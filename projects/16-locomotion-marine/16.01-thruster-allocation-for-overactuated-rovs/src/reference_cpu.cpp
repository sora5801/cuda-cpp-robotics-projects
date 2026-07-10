// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference AND one-time setup math for
//                     project 16.01 (Thruster allocation for overactuated
//                     ROVs, batched QP)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), same as every project here:
//   1) CORRECTNESS ORACLE — thruster_allocate_cpu is a dead-simple sequential
//      twin of the GPU kernel; main.cu asserts the two agree element-wise.
//   2) TEACHING BASELINE — reading this file next to kernels.cu shows
//      exactly what "porting to the GPU" changed here: NOTHING about the
//      math, only WHERE the per-problem loop runs (a CPU for-loop over k vs.
//      one GPU thread per k) and where the shared matrices live (a plain
//      array here vs. __constant__ memory there).
//
// This file ALSO owns the ONE-TIME HOST SETUP MATH (build_allocation_matrix,
// build_qp_matrices, power_iteration_lambda_max, cholesky_solve_spd) that
// main.cu runs once at startup before either path launches. That code is
// not "a competing implementation of the kernel" — it is plain host
// arithmetic the kernel and the oracle both CONSUME (the H/BtW2 matrices,
// the step size) — 08.01 sets the same precedent by putting its shared
// "plant stepper" in this same file rather than duplicating it.
//
// Rules for the ORACLE portion of this file: plain C++17, no CUDA headers,
// no hand-vectorization, no cleverness. If the reference is clever, it can
// be wrong, and then the oracle lies.
//
// Read this after: kernels.cuh.  Companion GPU kernel: kernels.cu.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>     // sqrtf
#include <cstdio>    // fprintf for the Cholesky non-SPD diagnostic
#include <cstdlib>
#include <limits>    // std::numeric_limits<float>::quiet_NaN() (non-SPD diagnostic)

// ===========================================================================
// Section 1 — one-time host setup math (§ single-source geometry -> QP).
// ===========================================================================

// ---------------------------------------------------------------------------
// build_allocation_matrix — B (row-major, kNDof x kNThr) from the
// kThrusterPos / kThrusterDir tables in kernels.cuh.
//
// Row 0-2 of column i is simply d_i (thruster i's force contributes to the
// body wrench's FORCE rows exactly along its own mounting direction). Row
// 3-5 of column i is the MOMENT arm r_i x d_i — the textbook "force at a
// point produces a moment about the origin" cross product, taught in every
// statics course and re-derived here in code instead of hidden in a matrix
// library. THEORY.md "the math" derives this from first principles (torque
// = r x F) and connects it to the specific geometry table in kernels.cuh.
// ---------------------------------------------------------------------------
void build_allocation_matrix(float* B)
{
    for (int i = 0; i < kNThr; ++i) {
        const float rx = kThrusterPos[i][0], ry = kThrusterPos[i][1], rz = kThrusterPos[i][2];
        const float dx = kThrusterDir[i][0], dy = kThrusterDir[i][1], dz = kThrusterDir[i][2];

        // Force rows: thruster i's unit force direction, unchanged.
        B[0 * kNThr + i] = dx;   // Fx row
        B[1 * kNThr + i] = dy;   // Fy row
        B[2 * kNThr + i] = dz;   // Fz row

        // Moment rows: tau_i = r_i x d_i (right-hand rule; body frame is
        // x-forward/y-starboard/z-down per kernels.cuh — right-handed, so
        // the standard cross-product formula applies unmodified).
        B[3 * kNThr + i] = ry * dz - rz * dy;  // Mx row
        B[4 * kNThr + i] = rz * dx - rx * dz;  // My row
        B[5 * kNThr + i] = rx * dy - ry * dx;  // Mz row
    }
}

// ---------------------------------------------------------------------------
// build_qp_matrices — form H, BtW2, and Q from B and the diagonal weights W.
//
// Q       = B^T W^2 B + eps*I     (kNThr x kNThr, SYMMETRIC POSITIVE DEFINITE
//                                   because eps > 0 — even though B^T W^2 B
//                                   ALONE is only positive SEMI-definite:
//                                   rank(B) = 6 < kNThr = 8, so B^T W^2 B has
//                                   a 2-dimensional null space — THEORY.md
//                                   "the math" ties this to the physical
//                                   "internal squeeze" redundancy directions.)
// H       = 2*Q                   (the QP's actual Hessian: J(u) expands to
//                                   0.5 u^T H u - g^T u + const — see the
//                                   derivation in kernels.cu's kernel header)
// BtW2    = B^T W^2                (so g = 2*BtW2*tau for any commanded wrench)
//
// All O(kNThr^2 * kNDof) work, run ONCE — no reason to hand-optimize.
// ---------------------------------------------------------------------------
void build_qp_matrices(const float* B, const float* W, float eps,
                       float* H, float* BtW2, float* Q)
{
    // W2[d] = W[d]^2 — the diagonal of W^T W (W is diagonal, so W^T W is too).
    float W2[kNDof];
    for (int d = 0; d < kNDof; ++d) W2[d] = W[d] * W[d];

    // BtW2 = B^T * diag(W2): column d of B, scaled by W2[d], transposed into
    // row i of BtW2. BtW2[i*kNDof + d] = B[d*kNThr + i] * W2[d].
    for (int i = 0; i < kNThr; ++i)
        for (int d = 0; d < kNDof; ++d)
            BtW2[i * kNDof + d] = B[d * kNThr + i] * W2[d];

    // Q = BtW2 * B + eps*I : Q[i][j] = sum_d BtW2[i][d]*B[d][j], then add eps
    // on the diagonal. Symmetric by construction (BtW2*B = B^T W^2 B), so we
    // could exploit that and compute only the upper triangle — but kNThr=8
    // makes the full O(64*6) computation trivial, and writing it out in full
    // keeps this function's intent (and its output layout) unambiguous.
    for (int i = 0; i < kNThr; ++i) {
        for (int j = 0; j < kNThr; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < kNDof; ++d)
                acc += BtW2[i * kNDof + d] * B[d * kNThr + j];
            if (i == j) acc += eps;
            Q[i * kNThr + j] = acc;
            H[i * kNThr + j] = 2.0f * acc;   // H = 2*Q
        }
    }
}

// ---------------------------------------------------------------------------
// power_iteration_lambda_max — classic power method for the largest
// eigenvalue of a symmetric n x n matrix M (row-major).
//
// Algorithm (THEORY.md "the math" walks the intuition): start from a fixed
// vector v0, repeatedly apply v <- M*v then renormalize; v converges to the
// dominant eigenvector, and the Rayleigh quotient v^T M v / v^T v converges
// to the dominant eigenvalue. We start from the all-ones vector (normalized)
// — a deliberately UNINTERESTING, fully reproducible starting point (no RNG
// anywhere in this project's setup math), and iterate a fixed kPowerIters
// times rather than checking for convergence, exactly like the QP kernel's
// own fixed-iteration discipline (kernels.cu). For an 8x8 matrix this
// converges to FP32 precision in well under 100 iterations whenever the top
// two eigenvalues are not nearly equal (true here — THEORY.md reports the
// actual measured spectrum).
// ---------------------------------------------------------------------------
float power_iteration_lambda_max(const float* M, int n, int iters)
{
    // Fixed start: the normalized all-ones vector. Any generic starting
    // vector works (power iteration fails only for a measure-zero set of
    // starts exactly orthogonal to the dominant eigenvector); all-ones is
    // simply the simplest reproducible choice.
    float v[kNThr];   // n <= kNThr for every caller in this project
    float norm0 = 0.0f;
    for (int i = 0; i < n; ++i) { v[i] = 1.0f; norm0 += 1.0f; }
    norm0 = std::sqrt(norm0);
    for (int i = 0; i < n; ++i) v[i] /= norm0;

    float lambda = 0.0f;
    for (int it = 0; it < iters; ++it) {
        float Mv[kNThr];
        for (int i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) acc += M[i * n + j] * v[j];
            Mv[i] = acc;
        }
        // Rayleigh quotient with the OLD (unit-norm) v: lambda = v . (M v).
        float rq = 0.0f;
        for (int i = 0; i < n; ++i) rq += v[i] * Mv[i];
        lambda = rq;

        // Renormalize Mv into the next v (guard against a degenerate M=0).
        float norm = 0.0f;
        for (int i = 0; i < n; ++i) norm += Mv[i] * Mv[i];
        norm = std::sqrt(norm);
        if (norm < 1e-20f) break;   // M is (numerically) the zero matrix
        for (int i = 0; i < n; ++i) v[i] = Mv[i] / norm;
    }
    return lambda;
}

// ---------------------------------------------------------------------------
// cholesky_solve_spd — solve Q x = b for SPD Q (n x n, row-major) via
// in-place lower-triangular Cholesky Q = L L^T, then forward/back
// substitution. This is 33.01's per-thread batched algorithm, here run
// ONCE, sequentially, for a single n=8 system — the "pseudoinverse gate"
// (main.cu Stage 2b) uses it to compute the closed-form damped weighted
// pseudoinverse x* = Q^-1 (B^T W^2 tau) that the unsaturated QP solutions
// must match (THEORY.md "the math" derives WHY the unconstrained optimum of
// this exact QP has that closed form — it is the normal-equations solution
// of a weighted, Tikhonov-damped least squares problem).
//
// Non-SPD input (should never happen here — Q is SPD by construction for
// eps > 0) is reported loudly rather than silently producing garbage; see
// 33.01's identical NaN-on-failure policy for the reasoning.
// ---------------------------------------------------------------------------
void cholesky_solve_spd(const float* Q, const float* b, float* x, int n)
{
    float L[kNThr * kNThr] = { 0.0f };   // n <= kNThr for every caller here

    for (int j = 0; j < n; ++j) {
        float diag = Q[j * n + j];
        for (int p = 0; p < j; ++p) diag -= L[j * n + p] * L[j * n + p];
        if (diag <= 0.0f) {
            std::fprintf(stderr,
                "cholesky_solve_spd: matrix is not SPD (pivot %d = %g) — "
                "this should not happen for Q = B^T W^2 B + eps*I with eps>0; "
                "check the eps/weights configuration in kernels.cuh\n", j, diag);
            // std::numeric_limits::quiet_NaN(), not a literal 0.0f/0.0f divide
            // (MSVC rejects that as a compile-time div-by-zero error): NaN
            // propagates through any downstream arithmetic, so a caller
            // cannot mistake a failed solve for a real answer.
            for (int i = 0; i < n; ++i) x[i] = std::numeric_limits<float>::quiet_NaN();
            return;
        }
        const float l_jj = std::sqrt(diag);
        L[j * n + j] = l_jj;
        const float inv_l_jj = 1.0f / l_jj;
        for (int i = j + 1; i < n; ++i) {
            float s = Q[i * n + j];
            for (int p = 0; p < j; ++p) s -= L[i * n + p] * L[j * n + p];
            L[i * n + j] = s * inv_l_jj;
        }
    }

    // Forward: L y = b (reuse x[] to hold y, then overwrite with the final x).
    for (int i = 0; i < n; ++i) {
        float s = b[i];
        for (int p = 0; p < i; ++p) s -= L[i * n + p] * x[p];
        x[i] = s / L[i * n + i];
    }
    // Back: L^T x = y.
    for (int i = n - 1; i >= 0; --i) {
        float s = x[i];
        for (int p = i + 1; p < n; ++p) s -= L[p * n + i] * x[p];
        x[i] = s / L[i * n + i];
    }
}

// ===========================================================================
// Section 2 — the CPU oracle (the twin of kernels.cu's kernel).
// ===========================================================================

// ---------------------------------------------------------------------------
// pgd_step — ONE projected-gradient-descent update, shared by the batched
// oracle and the tracing function below so the two can never drift apart.
// Line-by-line identical to the GPU kernel's inner loop body (kernels.cu).
// ---------------------------------------------------------------------------
static inline void pgd_step(float* u, const float* g, const float* H,
                            const float* umax, float step)
{
    float grad[kNThr];
    for (int i = 0; i < kNThr; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < kNThr; ++j) acc += H[i * kNThr + j] * u[j];
        grad[i] = acc - g[i];
    }
    for (int i = 0; i < kNThr; ++i) {
        float cand = u[i] - step * grad[i];
        cand = cand < -umax[i] ? -umax[i] : cand;
        cand = cand > umax[i] ? umax[i] : cand;
        u[i] = cand;
    }
}

// ---------------------------------------------------------------------------
// thruster_allocate_cpu — sequential twin of pgd_allocate_kernel, one QP at
// a time. Same H/BtW2/step (computed once by main.cu and passed to BOTH
// paths — never recomputed here), same fixed iteration count, same
// zero-thrust cold start. main.cu's §5 gate compares this function's output
// against the GPU kernel's, element by element.
// ---------------------------------------------------------------------------
void thruster_allocate_cpu(int count,
                           const float* tau, const float* umax,
                           const float* H, const float* BtW2, float step,
                           int iters, float* u_out)
{
    for (int k = 0; k < count; ++k) {
        const float* t = &tau[k * kNDof];
        const float* um = &umax[k * kNThr];

        // g = 2 * BtW2 * t (identical formula to the kernel's).
        float g[kNThr];
        for (int i = 0; i < kNThr; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < kNDof; ++j) acc += BtW2[i * kNDof + j] * t[j];
            g[i] = 2.0f * acc;
        }

        float u[kNThr] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (int it = 0; it < iters; ++it) pgd_step(u, g, H, um, step);

        for (int i = 0; i < kNThr; ++i) u_out[k * kNThr + i] = u[i];
    }
}

// ---------------------------------------------------------------------------
// thruster_allocate_trace_cpu — the SAME algorithm for a SINGLE problem,
// logging the QP objective J(u_k) and the raw wrench residual ||Bu_k-tau||
// at every iteration (including k=0, the zero-thrust start).
//
// Why the objective is the RIGOROUSLY guaranteed monotone quantity, and the
// residual is reported for context only: for an L-smooth convex f and a
// convex constraint set, the projected-gradient / proximal-gradient method
// with step <= 1/L satisfies the classical "descent lemma"
// (Beck & Teboulle 2009, Prop. 3.1): F(u_{k+1}) <= F(u_k) for
// F(u) = f(u) + indicator(u in box), i.e. exactly J(u_k) here. The raw
// residual ||Bu_k - tau|| is only ONE PART of J (the other part is the
// eps*||u||^2 effort penalty) — nothing in the theorem promises it alone is
// monotone. main.cu's Stage 2c checks J (the rigorous claim) and reports the
// residual alongside it for the reader to inspect (THEORY.md "how we verify
// correctness" discusses this precisely, including whether the residual
// happens to be monotone too for this project's specific test wrench).
// ---------------------------------------------------------------------------
void thruster_allocate_trace_cpu(const float* tau, const float* umax,
                                 const float* B, const float* W,
                                 const float* H, const float* BtW2,
                                 float eps, float step, int iters,
                                 float* J_trace, float* residual_trace)
{
    float W2[kNDof];
    for (int d = 0; d < kNDof; ++d) W2[d] = W[d] * W[d];

    float g[kNThr];
    for (int i = 0; i < kNThr; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < kNDof; ++j) acc += BtW2[i * kNDof + j] * tau[j];
        g[i] = 2.0f * acc;
    }

    // log_state — compute J(u) and ||Bu - tau|| for the CURRENT u and append
    // both to the trace at index `idx`. Local lambda-like helper (a plain
    // static function would need u/idx passed explicitly — inlined here for
    // readability at the two call sites below).
    auto log_state = [&](const float* u, int idx) {
        float Bu[kNDof];
        for (int d = 0; d < kNDof; ++d) {
            float acc = 0.0f;
            for (int i = 0; i < kNThr; ++i) acc += B[d * kNThr + i] * u[i];
            Bu[d] = acc;
        }
        float weighted_sq = 0.0f, residual_sq = 0.0f;
        for (int d = 0; d < kNDof; ++d) {
            const float r = Bu[d] - tau[d];
            weighted_sq += W2[d] * r * r;
            residual_sq += r * r;
        }
        float effort = 0.0f;
        for (int i = 0; i < kNThr; ++i) effort += u[i] * u[i];
        J_trace[idx] = weighted_sq + eps * effort;
        residual_trace[idx] = std::sqrt(residual_sq);
    };

    float u[kNThr] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    log_state(u, 0);   // iteration 0 = the zero-thrust starting point
    for (int it = 0; it < iters; ++it) {
        pgd_step(u, g, H, umax, step);
        log_state(u, it + 1);
    }
}
