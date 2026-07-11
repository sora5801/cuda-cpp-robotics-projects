// ===========================================================================
// kernels.cu — GPU implementation for project 01.12
//              Visual servoing: image-Jacobian control loop entirely on GPU
//              (teaching core: eye-in-hand IBVS, batched convergence-basin study)
//
// The big idea
// ------------
// Each GPU thread owns ONE closed IBVS loop: it holds a camera pose in
// registers, and for up to kMaxSteps ticks it (1) projects the 4 target
// points, (2) assembles the 6x6 damped normal equations of the image
// Jacobian, (3) solves them for a twist, (4) integrates the pose, and
// (5) checks convergence — exactly the rollout-farm idiom 08.01 teaches for
// MPPI's OPEN-loop rollouts, applied here to a whole CLOSED feedback loop.
// K loops never interact, so one thread per loop is the natural mapping.
//
// What is NEW here beyond 08.01/33.01/09.01:
//   * the "rollout" IS the controller under test (closed loop, not a
//     candidate future scored by an external cost) — no softmin blend, no
//     host-side weighting: each thread's own local decisions ARE the answer;
//   * a 6x6 damped LEAST-SQUARES solve every step (33.01's Cholesky idiom,
//     applied to Gauss-Newton normal equations instead of a single SPD
//     system read from memory — here A is ASSEMBLED per thread, per step,
//     from 4 rank-2 point contributions, never materializing the dense 8x6
//     interaction matrix L itself — see build_normal_equations below);
//   * REAL per-thread control flow divergence: threads `break` out of their
//     loop at DIFFERENT step counts (fast-converging vs. pathological loops)
//     — unlike 08.01, where every thread runs the exact same T iterations.
//     THEORY.md §GPU-mapping teaches what this costs a warp.
//   * register pressure is genuinely higher than 08.01's 4-float cart-pole
//     state: a camera pose (7) plus a 6x6 normal matrix (36) is the
//     project's honest register story — see THEORY.md §GPU-mapping.
//
// All model constants and layouts come from kernels.cuh — the single source
// shared with the CPU oracle. Per the twin-vs-shared ruling documented at
// the top of reference_cpu.cpp, the ALGORITHMIC CORE below (quaternion
// math, the per-point Jacobian row, the Cholesky solve, the pose
// integrator) is a DELIBERATE INDEPENDENT reimplementation of the CPU
// oracle's — not a shared __host__ __device__ helper — so that the
// GPU-vs-CPU comparison in main.cu actually tests something.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// __constant__ memory: the target's 4 world-frame points and the desired
// feature vector s*. Every thread of every loop, every step, reads these
// SAME 20 floats (see kernels.cuh "GOAL POSE") — constant memory caches and
// BROADCASTS a uniform read to a whole warp in one transaction, the same
// reasoning 09.01 uses for its robot model. 80 bytes total: trivially small.
// ---------------------------------------------------------------------------
__constant__ float c_target_pts[12];   // 4 points * (X,Y,Z) world-frame, m
__constant__ float c_s_star[8];        // 4 points * (x,y) desired normalized coords

void set_target_and_goal(const float target_pts_world[12], const float s_star[8])
{
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_pts, target_pts_world, 12 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_s_star,     s_star,           8 * sizeof(float)));
}

// ===========================================================================
// Small device-side math helpers. All __device__ __forceinline__: they are
// called from inside the hot per-step loop and must not pay a function-call
// tax; __forceinline__ asks nvcc to inline unconditionally (it usually would
// anyway at -O2/-O3, but the request documents the intent).
// ===========================================================================

// quat_conj — conjugate (w,x,y,z) -> (w,-x,-y,-z): for a UNIT quaternion this
// IS the inverse rotation, used below to go world-frame -> camera-frame.
__device__ __forceinline__ void quat_conj(const float q[4], float out[4])
{
    out[0] =  q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

// quat_mul — Hamilton product out = a ⊗ b, (w,x,y,z) order throughout (the
// repo convention, CLAUDE.md §12). Composing rotations: a ⊗ b applies b
// FIRST, then a, when both are expressed the same way this project uses
// them (see ibvs_integrate: q_new = q ⊗ q_delta, RIGHT-multiplying by a
// BODY-frame increment — the standard rule for integrating a rotation given
// an angular velocity expressed in the rotating body's own frame).
__device__ __forceinline__ void quat_mul(const float a[4], const float b[4], float out[4])
{
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

// quat_normalize — renormalize in place. Every integration step introduces a
// tiny drift away from unit norm (FP32 rounding through sin/cos and the
// Hamilton product); THEORY.md §numerics discusses why we renormalize EVERY
// step rather than letting error accumulate over up to 400 steps.
__device__ __forceinline__ void quat_normalize(float q[4])
{
    const float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    const float inv = rsqrtf(n2);   // fast reciprocal sqrt intrinsic — a unit
                                     // quaternion's norm is always close to 1,
                                     // so the single-Newton-iteration
                                     // intrinsic is accurate enough here
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}

// rotate_by_quat — rotate vector v (3) by unit quaternion q, out = q v q*.
// Uses the well-known "double cross product" expansion (Rodrigues via
// quaternion) instead of building a 3x3 rotation matrix first: for a SINGLE
// vector this is fewer FLOPs and fewer registers than materializing R.
//   out = v + 2*qw*(qv x v) + 2*(qv x (qv x v)),   qv = (q1,q2,q3)
__device__ __forceinline__ void rotate_by_quat(const float q[4], const float v[3], float out[3])
{
    const float qvx = q[1], qvy = q[2], qvz = q[3];
    // t = qv x v
    const float tx = qvy*v[2] - qvz*v[1];
    const float ty = qvz*v[0] - qvx*v[2];
    const float tz = qvx*v[1] - qvy*v[0];
    // u = qv x t
    const float ux = qvy*tz - qvz*ty;
    const float uy = qvz*tx - qvx*tz;
    const float uz = qvx*ty - qvy*tx;
    out[0] = v[0] + 2.0f*q[0]*tx + 2.0f*ux;
    out[1] = v[1] + 2.0f*q[0]*ty + 2.0f*uy;
    out[2] = v[2] + 2.0f*q[0]*tz + 2.0f*uz;
}

// rotate_by_conj_quat — rotate v by the INVERSE of q, i.e. world-frame ->
// camera-frame (see kernels.cuh "CAMERA POSE LAYOUT"). Conjugating first and
// reusing rotate_by_quat keeps there being exactly ONE rotation formula to
// get right; the conjugate is three sign flips, effectively free.
__device__ __forceinline__ void rotate_by_conj_quat(const float q[4], const float v[3], float out[3])
{
    float qc[4];
    quat_conj(q, qc);
    rotate_by_quat(qc, v, out);
}

// ---------------------------------------------------------------------------
// chol6_inplace — factor a 6x6 SYMMETRIC POSITIVE-DEFINITE matrix A into
// A = R Rᵀ, R lower-triangular, IN PLACE (R overwrites A's lower triangle;
// the strict upper triangle is left untouched and must never be read again).
//
// Why in-place: the hot batch kernel never needs the pre-factorization A
// after this call (unlike the verification kernel below, which copies A out
// FIRST) — reusing the storage saves 36 registers/thread that 4096
// concurrently-resident threads would otherwise all pay for (THEORY.md
// §GPU-mapping quantifies the register story).
//
// Method: classic textbook column-by-column Cholesky (the same algorithm
// 33.01's batched_cholesky_solve teaches for general N; here N=6 is fixed
// and the loops fully unroll into straight-line code). A is guaranteed SPD
// by construction — it is LᵀL (a Gram matrix, PSD) plus kDampingMu*I (a
// strictly positive diagonal shift) — so the sqrt argument below is
// mathematically always positive; the max(...,tiny) guard is defensive
// against FP32 rounding driving a diagonal pivot to a hair below zero in a
// genuinely near-singular configuration (see THEORY.md §numerics for when
// this can happen and why the clamp is honest, not a silent wrong-answer).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void chol6_inplace(float A[6][6])
{
#pragma unroll
    for (int i = 0; i < 6; ++i) {
#pragma unroll
        for (int j = 0; j <= i; ++j) {
            float sum = A[i][j];
#pragma unroll
            for (int k = 0; k < j; ++k) sum -= A[i][k] * A[j][k];
            if (i == j) {
                A[i][i] = sqrtf(fmaxf(sum, 1e-12f));   // diagonal pivot (defensive clamp, see above)
            } else {
                A[i][j] = sum / A[j][j];
            }
        }
    }
}

// chol6_solve — solve (R Rᵀ) x = b given R (A's lower triangle, from
// chol6_inplace) via forward substitution (R y = b) then back substitution
// (Rᵀ x = y) — the standard O(N^2) Cholesky solve, versus O(N^3) for a
// general factorization+solve; N=6 here so the difference is small in
// absolute terms but the PATTERN is the one that matters at real robot
// sizes (33.01's THEORY.md quantifies it for N up to a manipulator's DoF).
__device__ __forceinline__ void chol6_solve(const float R[6][6], const float b[6], float x[6])
{
    float y[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        float sum = b[i];
#pragma unroll
        for (int k = 0; k < i; ++k) sum -= R[i][k] * y[k];
        y[i] = sum / R[i][i];
    }
#pragma unroll
    for (int i = 5; i >= 0; --i) {
        float sum = y[i];
#pragma unroll
        for (int k = i + 1; k < 6; ++k) sum -= R[k][i] * x[k];   // Rᵀ[i][k] = R[k][i]
        x[i] = sum / R[i][i];
    }
}

// cholesky_diag_ratio — the cheap conditioning PROXY (kernels.cuh "OUTPUT
// LAYOUT"): min diagonal entry of the Cholesky factor over the max. A
// well-conditioned A has comparable pivots (ratio near 1); a
// near-singular A drives one pivot toward the damping floor while others
// stay large (ratio -> 0). This is NOT the true condition number (that
// would need the actual eigenvalues/singular values of A, expensive at
// this size on a GPU thread) — it is a byproduct of work already done
// (the Cholesky factor we just computed for the solve), honestly labeled
// a proxy everywhere it is used (THEORY.md §numerics).
__device__ __forceinline__ float cholesky_diag_ratio(const float R[6][6])
{
    float lo = R[0][0], hi = R[0][0];
#pragma unroll
    for (int i = 1; i < 6; ++i) {
        const float d = R[i][i];
        lo = fminf(lo, d);
        hi = fmaxf(hi, d);
    }
    return lo / fmaxf(hi, 1e-12f);
}

// ===========================================================================
// ibvs_compute_step — ONE IBVS control step from `pose`: project the 4
// target points, assemble the damped Gauss-Newton normal equations of the
// image Jacobian, solve for the twist. This is the function BOTH GPU
// kernels below call — the batch time-stepper and the single-step
// verification kernel — so there is exactly one GPU implementation of the
// control law (its independent CPU twin lives in reference_cpu.cpp).
//
// Smart accumulation (why we never materialize the 8x6 interaction matrix
// L): the textbook derivation stacks 4 per-point 2x6 Jacobian blocks into
// one 8x6 matrix L, then forms LᵀL (6x6) and Lᵀe (6). But LᵀL is a SUM over
// points of each point's OWN 2x6 block contribution — Σᵢ Lᵢᵀmy Lᵢ — so we
// can accumulate A += Lᵢᵀ Lᵢ and b += Lᵢᵀ eᵢ point-by-point, in registers,
// and never store the dense 8x6 L at all. This is the same "normal
// equations via rank-k accumulation" pattern used in any Gauss-Newton
// solver (bundle adjustment, ICP) — 4 points here, thousands of points in
// a real solver, same idea. It also HALVES the register footprint this
// kernel would otherwise pay for L (48 floats) on top of A (36).
//
// Parameters:
//   pose            : [7] this loop's CURRENT camera pose (kernels.cuh layout).
//   variant         : which (x,y,Z) triple feeds the Jacobian ROW formula
//                     (kernels.cuh "CONTROLLER VARIANTS"); the ERROR always
//                     uses the true current features regardless of variant.
//   v_out           : [6] OUT — the computed twist v_c = -λ·x.
//   A_out           : [36] OUT (row-major 6x6) OR nullptr — the damped
//                     normal matrix BEFORE factorization (verification use).
//   b_out           : [6] OUT OR nullptr — the right-hand side Lᵀe.
//   feat_out        : [8] OUT — current normalized features (x,y) per point.
//   err_norm_out    : [1] OUT — L2 norm of (feat - s*).
//   cond_proxy_out  : [1] OUT — this step's conditioning proxy (0,1].
//   zmax_out        : [1] OUT — max TRUE depth (m) among the 4 points this step.
//   featmax_out     : [1] OUT — max |x| or |y| among the 4 points this step.
// ===========================================================================
__device__ void ibvs_compute_step(const float pose[7], int variant,
                                  float v_out[6], float* A_out, float* b_out,
                                  float feat_out[8], float* err_norm_out,
                                  float* cond_proxy_out, float* zmax_out, float* featmax_out)
{
    const float p[3] = { pose[0], pose[1], pose[2] };
    const float q[4] = { pose[3], pose[4], pose[5], pose[6] };

    // A (the 6x6 normal matrix) and b (its right-hand side) accumulate
    // across the 4 points below — registers, zero-initialized once.
    float A[6][6];
#pragma unroll
    for (int i = 0; i < 6; ++i)
#pragma unroll
        for (int j = 0; j < 6; ++j) A[i][j] = 0.0f;
    float b[6] = { 0,0,0,0,0,0 };

    float sumSqErr = 0.0f;   // running Σ(feature error)^2 -> L2 norm at the end
    float zmax = 0.0f;       // running max TRUE depth across the 4 points
    float featmax = 0.0f;    // running max |normalized coordinate|

    // -----------------------------------------------------------------
    // Per-point loop: project, error, per-point Jacobian row, accumulate.
    // Unrolled (kNumPoints=4 is a compile-time constant) — no loop
    // overhead, and the compiler can interleave the 4 points' independent
    // work across the pipeline.
    // -----------------------------------------------------------------
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // World-frame vector from camera to point i, then into camera frame
        // (see kernels.cuh "CAMERA POSE LAYOUT": world->camera uses q's
        // conjugate). c_target_pts is __constant__ memory (broadcast read).
        const float Pw[3] = {
            c_target_pts[i*3 + 0] - p[0],
            c_target_pts[i*3 + 1] - p[1],
            c_target_pts[i*3 + 2] - p[2]
        };
        float Pc[3];
        rotate_by_conj_quat(q, Pw, Pc);

        const float Z = Pc[2];                 // TRUE depth (m) — the ground truth this simulation knows
        const float x = Pc[0] / Z;              // current normalized image coordinates
        const float y = Pc[1] / Z;
        feat_out[2*i + 0] = x;
        feat_out[2*i + 1] = y;

        const float ex = x - c_s_star[2*i + 0]; // feature error ALWAYS uses the true current feature —
        const float ey = y - c_s_star[2*i + 1]; // only the JACOBIAN below substitutes per variant
        sumSqErr += ex*ex + ey*ey;

        zmax    = fmaxf(zmax, Z);
        featmax = fmaxf(featmax, fmaxf(fabsf(x), fabsf(y)));

        // Choose the (xj,yj,Zj) triple feeding the interaction-matrix ROW
        // formula — the ONLY place the three controller variants differ
        // (kernels.cuh "CONTROLLER VARIANTS"). variant is a runtime int
        // (not templated): a project-scale kernel could specialize per
        // variant to shave a few predicated selects, but that would cost
        // THREE separate kernel bodies to keep in sync with the CPU
        // oracle — this repo's teaching-over-cleverness call (CLAUDE.md
        // §1) is one branch, read once, understood for all three variants.
        float xj, yj, Zj;
        if (variant == kVariantTrueDepth) {
            xj = x; yj = y; Zj = Z;
        } else if (variant == kVariantFixedDepth) {
            xj = x; yj = y; Zj = kGoalStandoff;
        } else { // kVariantDesiredJacobian
            xj = c_s_star[2*i + 0]; yj = c_s_star[2*i + 1]; Zj = kGoalStandoff;
        }

        // The classical per-point interaction-matrix block (THEORY.md
        // derives this from d/dt(X/Z, Y/Z) under a rigid-body camera
        // twist): two rows mapping v_c=(vx,vy,vz,wx,wy,wz) to (ẋ,ẏ).
        const float invZ = 1.0f / Zj;
        const float Lrow0[6] = { -invZ, 0.0f,  xj*invZ,      xj*yj,        -(1.0f+xj*xj),  yj };
        const float Lrow1[6] = {  0.0f, -invZ, yj*invZ,      1.0f+yj*yj,   -xj*yj,        -xj };

        // Accumulate this point's rank-2 contribution into the NORMAL
        // EQUATIONS: A += Lrowᵀ Lrow (both rows), b += Lrowᵀ · (ex,ey).
        // This is the "never materialize the dense L" trick from the
        // function header — we go straight from two 6-vectors to a 6x6
        // rank-2 update, the same total FLOP count as building L then
        // multiplying, at a fraction of the register footprint.
#pragma unroll
        for (int r = 0; r < 6; ++r) {
            b[r] += Lrow0[r]*ex + Lrow1[r]*ey;
#pragma unroll
            for (int c = 0; c < 6; ++c)
                A[r][c] += Lrow0[r]*Lrow0[c] + Lrow1[r]*Lrow1[c];
        }
    }

    // Levenberg damping: A = LᵀL + μI. LᵀL alone is only POSITIVE
    // SEMI-definite (rank <= 6, and can be lower still near a genuine
    // kinematic singularity of the 4-point configuration); adding μI on
    // the diagonal makes A strictly SPD so chol6_inplace's sqrt argument
    // is always positive by construction, AND it is the textbook
    // Levenberg-Marquardt regularization that keeps the solved twist
    // bounded when LᵀL is nearly singular (THEORY.md §numerics discusses
    // the damping-vs-conditioning trade honestly — too much μ slows
    // convergence, too little lets a near-singular step fire a huge twist).
#pragma unroll
    for (int i = 0; i < 6; ++i) A[i][i] += kDampingMu;

    if (A_out) {
#pragma unroll
        for (int i = 0; i < 6; ++i)
#pragma unroll
            for (int j = 0; j < 6; ++j) A_out[i*6 + j] = A[i][j];
    }
    if (b_out) {
#pragma unroll
        for (int i = 0; i < 6; ++i) b_out[i] = b[i];
    }

    // Factor IN PLACE (A becomes its own Cholesky factor — see
    // chol6_inplace's header for why this matters for register pressure)
    // and solve for x = pinv(L)-like least-squares direction.
    chol6_inplace(A);
    float x[6];
    chol6_solve(A, b, x);
    const float condProxy = cholesky_diag_ratio(A);

    // The control law: v_c = -λ · x  (kernels.cuh step 4).
#pragma unroll
    for (int i = 0; i < 6; ++i) v_out[i] = -kLambda * x[i];

    *err_norm_out   = sqrtf(sumSqErr);
    *cond_proxy_out = condProxy;
    *zmax_out       = zmax;
    *featmax_out    = featmax;
}

// ---------------------------------------------------------------------------
// ibvs_feature_error_norm — cheap re-projection ONLY (no Jacobian, no
// solve): used once, after a non-converged loop's LAST integration step, to
// report the feature error AT the fully-advanced final pose. (A converged
// loop already has the correct final error from the step that triggered
// convergence — it never integrates again after that — so this function is
// only needed for the "ran out of steps" exit path; see ibvs_batch_kernel.)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ibvs_feature_error_norm(const float pose[7])
{
    const float p[3] = { pose[0], pose[1], pose[2] };
    const float q[4] = { pose[3], pose[4], pose[5], pose[6] };
    float sumSqErr = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float Pw[3] = {
            c_target_pts[i*3+0] - p[0], c_target_pts[i*3+1] - p[1], c_target_pts[i*3+2] - p[2]
        };
        float Pc[3];
        rotate_by_conj_quat(q, Pw, Pc);
        const float x = Pc[0] / Pc[2], y = Pc[1] / Pc[2];
        const float ex = x - c_s_star[2*i+0], ey = y - c_s_star[2*i+1];
        sumSqErr += ex*ex + ey*ey;
    }
    return sqrtf(sumSqErr);
}

// ---------------------------------------------------------------------------
// ibvs_integrate — advance the camera pose by one control tick under the
// constant twist v (zero-order hold, exactly how a real control period
// applies a command — the same convention 08.01 uses for the cart-pole's
// force). SE(3) integration, EXACT in rotation / first-order in
// translation (THEORY.md justifies this split): the rotation update uses
// the quaternion EXPONENTIAL MAP (closed-form, exact for a constant angular
// velocity over dt — no linearization error there at all), while the
// position update is a first-order Euler step dp = R·v_lin·dt. The full
// se(3) matrix exponential also has a translation/rotation COUPLING term
// of order dt^2 that this omits; at kDt=0.01 s the coupling term is
// negligible next to the primary one (THEORY.md §numerics quantifies it) —
// the honest trade is exactness where it is cheap (rotation) and a
// well-understood O(dt^2) approximation where the alternative would cost
// noticeably more registers and arithmetic for a correction this small.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ibvs_integrate(float pose[7], const float v[6], float dt)
{
    const float vlin[3] = { v[0], v[1], v[2] };
    const float wang[3] = { v[3], v[4], v[5] };
    const float q[4]    = { pose[3], pose[4], pose[5], pose[6] };

    // Position: dp/dt = R_wc · v_lin (linear velocity is expressed in the
    // CAMERA's own frame; rotate it into world frame before integrating).
    float dp[3];
    rotate_by_quat(q, vlin, dp);
    pose[0] += dt * dp[0];
    pose[1] += dt * dp[1];
    pose[2] += dt * dp[2];

    // Orientation: exact quaternion exponential of the body-frame angular
    // velocity over this tick. theta = w_ang*dt is the rotation VECTOR
    // (axis * angle) swept this step.
    const float theta[3] = { dt*wang[0], dt*wang[1], dt*wang[2] };
    const float angle = sqrtf(theta[0]*theta[0] + theta[1]*theta[1] + theta[2]*theta[2]);
    float qDelta[4];
    if (angle > 1e-8f) {
        const float half = 0.5f * angle;
        const float s = sinf(half) / angle;      // = sinc(angle/2)/2, but written as sin/angle for clarity
        qDelta[0] = cosf(half);
        qDelta[1] = theta[0] * s;
        qDelta[2] = theta[1] * s;
        qDelta[3] = theta[2] * s;
    } else {
        // Small-angle fallback (avoids 0/0): exp(theta/2) ≈ 1 + theta/2.
        qDelta[0] = 1.0f;
        qDelta[1] = 0.5f * theta[0];
        qDelta[2] = 0.5f * theta[1];
        qDelta[3] = 0.5f * theta[2];
    }
    // RIGHT-multiply: q_delta is expressed in the CAMERA's own (body)
    // frame, so it composes on the right of the current world-frame
    // orientation (see quat_mul's header comment).
    float qNew[4];
    quat_mul(q, qDelta, qNew);
    quat_normalize(qNew);   // guard against the drift quat_normalize's own header describes
    pose[3] = qNew[0]; pose[4] = qNew[1]; pose[5] = qNew[2]; pose[6] = qNew[3];
}

// ===========================================================================
// ibvs_batch_kernel — the rollout-farm kernel: one thread = one closed IBVS
// loop, up to kMaxSteps ticks, early-exit on convergence.
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x + threadIdx.x
// owns loop k. Grid: ceil(K/256) x 256 (repo default; ragged tail guarded).
//
// REGISTER PRESSURE (read this before profiling and being surprised):
// this kernel's persistent per-thread state is a 7-float pose plus, inside
// ibvs_compute_step, a 6x6 normal matrix (36 floats) and its Cholesky
// factor (computed IN PLACE in the same 36 — see chol6_inplace) — roughly
// 45-60 live floats at the deepest point, noticeably more than 08.01's
// 4-float cart-pole state + RK4 scratch (~30). Expect nvcc to report more
// registers/thread here and correspondingly LOWER occupancy (fewer
// resident warps per SM) than 08.01's kernel on the same GPU — THEORY.md
// §GPU-mapping discusses this honestly and how to check it
// (`-Xptxas -v` register counts) rather than asserting a number that would
// vary by compiler version.
//
// WARP DIVERGENCE FROM EARLY EXIT (the other new lesson vs. 08.01): a
// warp's 32 threads run in lockstep: the hardware does not retire a warp
// until EVERY thread in it has finished. A `break` is per-thread — a
// thread whose loop CONVERGES at step 40 stops doing USEFUL work, but its
// warp keeps re-issuing instructions (with that thread's execution
// predicated off) until the WARP's slowest thread finishes — a retreat-
// pathology loop that never converges runs the full kMaxSteps. Because
// this kernel's loop-index-to-cohort mapping is CONTIGUOUS (kernels.cuh
// "COHORTS": all NOMINAL indices first, then DECAY, then RETREAT), most
// warps are drawn entirely from ONE cohort — a warp of 32 consecutive
// RETREAT-cohort threads pays the full kMaxSteps together (little wasted
// divergence WITHIN that warp, since they mostly agree on "keep going"),
// while a warp straddling a cohort BOUNDARY, or one nominal warp
// containing a single slow outlier, is bottlenecked by its slowest member.
// This is the general GPU lesson: early exit only saves WALL-CLOCK time to
// the extent whole WARPS agree to stop, not individual threads — see
// README Exercise for measuring it directly on this kernel.
// ===========================================================================
__global__ void ibvs_batch_kernel(const float* __restrict__ init_poses, // [K*7]
                                  int variant, int K,
                                  const int*   __restrict__ trace_idx,   // [trace_count] loop indices to log, or nullptr
                                  int trace_count,
                                  float* __restrict__ out_converged,     // [K]
                                  float* __restrict__ out_steps,         // [K]
                                  float* __restrict__ out_final_err,     // [K]
                                  float* __restrict__ out_cond_min,      // [K]
                                  float* __restrict__ out_zmax,          // [K]
                                  float* __restrict__ out_featmax,       // [K]
                                  float* __restrict__ out_trace)         // [trace_count*(kMaxSteps+1)*kTraceRowStride]
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;                                    // ragged-tail guard

    // This thread's starting pose — 4 uniform-per-thread-but-distinct
    // reads (each thread reads its OWN 7 floats: this is a coalesced,
    // per-thread-distinct global read, unlike the __constant__ broadcast
    // reads of the target/goal above).
    float pose[7];
#pragma unroll
    for (int i = 0; i < 7; ++i) pose[i] = init_poses[k*7 + i];

    // Is this loop one of the small documented traced subset? trace_count
    // is <= kTraceCount (8) — a linear scan of 8 ints is cheap next to a
    // 6x6 Cholesky solve, and every thread in the grid does the SAME 8
    // comparisons against the SAME small array (broadcast-friendly reads).
    int traceSlot = -1;
    for (int i = 0; i < trace_count; ++i)
        if (trace_idx[i] == k) { traceSlot = i; break; }

    float condMin = 1.0f;   // best-possible proxy value; we track the MIN (worst) seen
    float zMax = 0.0f, featMax = 0.0f;
    bool converged = false;
    int stepsUsed = 0;      // overwritten every iteration below (break or natural loop end)
    float lastErrNorm = 0.0f;

    for (int t = 0; t < kMaxSteps; ++t) {
        float v[6], feat[8], errNorm, condProxy, zStep, featStep;
        ibvs_compute_step(pose, variant, v, nullptr, nullptr, feat, &errNorm, &condProxy, &zStep, &featStep);

        condMin = fminf(condMin, condProxy);
        zMax    = fmaxf(zMax, zStep);
        featMax = fmaxf(featMax, featStep);
        lastErrNorm = errNorm;

        if (traceSlot >= 0) {
            float* row = out_trace
                       + (size_t)traceSlot * (kMaxSteps + 1) * kTraceRowStride
                       + (size_t)t * kTraceRowStride;
            row[0] = (float)t;
            row[1] = pose[0]; row[2] = pose[1]; row[3] = pose[2];
#pragma unroll
            for (int i = 0; i < 8; ++i) row[4 + i] = feat[i];
        }

        if (errNorm < kConvergeEps) {
            converged = true;
            stepsUsed = t;
            break;   // EARLY EXIT — see the kernel header's warp-divergence discussion
        }

        ibvs_integrate(pose, v, kDt);
        stepsUsed = t + 1;   // if the loop ends naturally next iteration test, this is how many we ran
    }

    // Honest final error: a CONVERGED loop's lastErrNorm is already the
    // error AT the pose we stopped at (we broke BEFORE integrating again);
    // a loop that spent its whole budget has been integrated one step
    // PAST its last evaluated error, so re-evaluate at the true final pose
    // (kernels.cu ibvs_feature_error_norm's header explains why).
    const float finalErr = converged ? lastErrNorm : ibvs_feature_error_norm(pose);

    out_converged[k] = converged ? 1.0f : 0.0f;
    out_steps[k]      = (float)stepsUsed;
    out_final_err[k]  = finalErr;
    out_cond_min[k]   = condMin;
    out_zmax[k]       = zMax;
    out_featmax[k]    = featMax;
}

// ===========================================================================
// ibvs_single_step_kernel — the verification kernel: ONE step from each of
// `count` sampled poses, exposing v/A/b/e for the tight Jacobian/
// pseudoinverse gate (kernels.cuh launch_ibvs_single_step). One thread per
// sampled pose — the same mapping idea as the batch kernel, at a much
// smaller, purpose-built scale (count is typically ~16, not 4096).
// ===========================================================================
__global__ void ibvs_single_step_kernel(const float* __restrict__ poses7, int count, int variant,
                                        float* __restrict__ out_v,    // [count*6]
                                        float* __restrict__ out_A,    // [count*36]
                                        float* __restrict__ out_b,    // [count*6]
                                        float* __restrict__ out_e)    // [count*8]
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float pose[7];
#pragma unroll
    for (int j = 0; j < 7; ++j) pose[j] = poses7[i*7 + j];

    float v[6], feat[8], errNorm, condProxy, zmax, featmax;
    ibvs_compute_step(pose, variant, v, &out_A[i*36], &out_b[i*6], feat, &errNorm, &condProxy, &zmax, &featmax);

#pragma unroll
    for (int c = 0; c < 6; ++c) out_v[i*6 + c] = v[c];
#pragma unroll
    for (int c = 0; c < 8; ++c) out_e[i*8 + c] = feat[c] - c_s_star[c];
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================
void launch_ibvs_batch(int K, int variant, const float* d_init_poses,
                       const int* d_trace_idx, int trace_count,
                       float* d_out_converged, float* d_out_steps,
                       float* d_out_final_err, float* d_out_cond_min,
                       float* d_out_zmax, float* d_out_featmax,
                       float* d_out_trace)
{
    if (K < 1 || !d_init_poses) {
        std::fprintf(stderr, "launch_ibvs_batch: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;                          // repo default geometry (warp multiple, good occupancy)
    const int blocks = (K + threads - 1) / threads;    // ceil(K/threads): cover every loop
    ibvs_batch_kernel<<<blocks, threads>>>(d_init_poses, variant, K,
                                           d_trace_idx, trace_count,
                                           d_out_converged, d_out_steps, d_out_final_err,
                                           d_out_cond_min, d_out_zmax, d_out_featmax, d_out_trace);
    CUDA_CHECK_LAST_ERROR("ibvs_batch_kernel launch");
}

void launch_ibvs_single_step(int count, const float* d_poses7, int variant,
                             float* d_out_v, float* d_out_A,
                             float* d_out_b, float* d_out_e)
{
    if (count < 1 || !d_poses7) {
        std::fprintf(stderr, "launch_ibvs_single_step: invalid arguments (count=%d)\n", count);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 64;    // count is small (~16); a single block comfortably covers it
    const int blocks = (count + threads - 1) / threads;
    ibvs_single_step_kernel<<<blocks, threads>>>(d_poses7, count, variant, d_out_v, d_out_A, d_out_b, d_out_e);
    CUDA_CHECK_LAST_ERROR("ibvs_single_step_kernel launch");
}
