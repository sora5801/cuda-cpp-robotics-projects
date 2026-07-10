// ===========================================================================
// kernels.cu — GPU implementation for project 15.01
//              Minimum-snap trajectory optimization batched over waypoint
//              sets (quadrotor-style 2-D waypoint flight)
//
// The big idea
// ------------
// A "batch" here is not one huge linear system — it is 10,000 SMALL ones
// (32x32), each belonging to a different waypoint set, and none of them
// touch each other. That is exactly 33.01's lesson ("robotics almost never
// needs one big matrix factorized fast; it needs a million tiny ones
// factorized simultaneously") pushed to a larger per-problem N:
//
//     one thread  =  one whole waypoint set (BOTH axes), solved in place.
//
// What is NEW here relative to 33.01's N=3/4/6 Cholesky kernel
// --------------------------------------------------------------
// 33.01's matrices (up to 6x6 = 36 floats) fit comfortably in registers
// after full unrolling. THIS project's system is 32x32 = 1024 floats — an
// order of magnitude more data than the ~255-register/thread hardware
// ceiling can hold. The compiler CANNOT place `float A[32*32]` in the
// register file; it spills to LOCAL memory — a per-thread private region
// that physically lives in the same off-chip DRAM as global memory (cached
// through L1/L2, same latency class as an uncached global load). This is
// not a bug or an oversight: it is the honest, DOCUMENTED reality of
// scaling the thread-per-problem pattern past the register budget, and
// THEORY.md §the-GPU-mapping walks through exactly why it still wins:
// thousands of INDEPENDENT 32x32 solves running in parallel, each thread's
// local array serviced by the same memory subsystem that serves global
// memory but with none of the cross-thread coalescing story to worry about
// (no two threads ever touch the same byte of A/b/x). A CPU doing this
// sequentially pays the FULL O(K*N^3) cost with only ILP to hide latency;
// the GPU pays the SAME per-thread cost but amortizes it over thousands of
// threads in flight — the win is occupancy-driven latency hiding, not
// register-file locality (contrast with 33.01's kernel comment, which is
// the register-resident end of this exact same design spectrum).
//
// Also new: every thread's constraint MATRIX A is IDENTICAL (the row layout
// in kernels.cuh depends only on which derivatives are pinned/continuous,
// never on the waypoint VALUES) — only the right-hand side b differs
// between problems (and between the x and y solve of the SAME problem).
// This kernel deliberately does NOT exploit that: each thread reassembles
// and re-eliminates A from scratch, twice (once per axis) — the honest,
// maximally-parallel "thread owns the whole problem" pattern this repo
// teaches everywhere else (08.01, 33.01, 09.01), at the cost of redundant
// work a smarter "factor A once, broadcast L/U, solve K*2 right-hand
// sides" design would avoid. README Exercise 4 asks you to build that
// smarter version and measure the win.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>
#include <math_constants.h>         // CUDART_NAN_F: canonical device NaN constant (33.01's NaN-on-failure policy)

// ---------------------------------------------------------------------------
// falling_factorial_dev — returns j! / (j-d)! = j*(j-1)*...*(j-d+1), the
// coefficient that appears when you differentiate tau^j exactly d times:
//
//     d^d/dtau^d [ tau^j ] = falling_factorial(j,d) * tau^(j-d)     (j >= d)
//                          = 0                                     (j <  d)
//
// Every row of the constraint matrix (position, endpoint-derivative,
// interior-continuity) is built from this one identity evaluated at
// tau = 0 or tau = 1 — see assemble_minsnap_system below. d never exceeds
// kMaxContinuityDeriv (6), so the loop is tiny and branchless-in-practice
// once unrolled by the compiler.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float falling_factorial_dev(int j, int d)
{
    if (j < d) return 0.0f;          // d-th derivative of tau^j is 0 when j < d
    float r = 1.0f;
#pragma unroll
    for (int t = 0; t < kMaxContinuityDeriv; ++t)   // d <= kMaxContinuityDeriv always; extra iterations are no-ops
        if (t < d) r *= static_cast<float>(j - t);
    return r;
}

// ---------------------------------------------------------------------------
// assemble_minsnap_system — build the 32x32 constraint matrix A and 32-
// vector b for ONE axis of ONE waypoint set, following the row layout that
// kernels.cuh documents as the single source of truth (position rows 0..7,
// endpoint-derivative rows 8..13, interior-continuity rows 14..31).
//
// Key simplification used throughout: every constraint in this project is
// evaluated at tau = 0 or tau = 1 (segment boundaries) — NEVER an interior
// tau. That means every "row" collapses to one of two shapes:
//   tau = 1 (row_at_tau1(d)): coefficients c_j for j = d..7 all appear,
//            weighted by falling_factorial(j,d), because 1^(j-d) = 1.
//   tau = 0 (row_at_tau0(d)): only c_d survives, weighted by d! = falling_
//            factorial(d,d), because 0^(j-d) is 1 only when j == d.
// No powf() call is needed anywhere — every nonzero entry is an EXACT
// small integer (<= 7! / 1! = 5040, representable exactly in FP32's 24-bit
// mantissa), which keeps this assembly step numerically exact; all the
// rounding in this kernel happens in the elimination that follows.
//
// Parameters:
//   wp  — [kNumWaypoints] this axis's waypoint coordinates (m), one number
//         per waypoint (the caller passes wp_x or wp_y separately).
//   A   — [kSysN*kSysN] OUT: row-major, zeroed then filled (32x32 = 1024
//         floats — LOCAL memory; see the file header for why).
//   b   — [kSysN] OUT: right-hand side.
// ---------------------------------------------------------------------------
__device__ void assemble_minsnap_system(const float wp[kNumWaypoints], float* A, float* b)
{
    // Plain (NOT unrolled) loop: A already lives in local memory regardless
    // of indexing style (1024 floats is ~4x the register file's per-thread
    // ceiling, so the compiler cannot register-allocate it either way) —
    // unrolling a 1024-iteration store loop would only bloat code size for
    // no placement benefit. Contrast with 33.01, where forcing constant
    // indices via unrolling was WHAT MADE register placement possible at
    // all; here that lever does nothing, so we do not pull it.
    for (int i = 0; i < kSysN * kSysN; ++i) A[i] = 0.0f;   // most entries stay 0 (see shapes above)

    // ---- Rows 0..7: position interpolation, 2 per segment -----------------
    // p_s(0) = wp[s]   (tau=0 row for d=0: only c_{s,0}, weight 0!=1)
    // p_s(1) = wp[s+1] (tau=1 row for d=0: every c_{s,j}, weight 1 — since
    //                   falling_factorial(j,0) is the empty product = 1)
    // kNumSegments (4) is small and FIXED, so unrolling this outer loop is
    // still cheap and lets the compiler constant-fold the row/segment
    // arithmetic below it — unlike the 1024-wide zeroing loop above, there
    // is a real (if small) benefit here.
#pragma unroll
    for (int s = 0; s < kNumSegments; ++s) {
        const int row0 = 2 * s, row1 = 2 * s + 1;
        A[row0 * kSysN + s * kCoeffsPerSegment + 0] = 1.0f;
        b[row0] = wp[s];
#pragma unroll
        for (int j = 0; j < kCoeffsPerSegment; ++j)
            A[row1 * kSysN + s * kCoeffsPerSegment + j] = 1.0f;
        b[row1] = wp[s + 1];
    }

    // ---- Rows 8..13: GLOBAL endpoint conditions ----------------------------
    // Start (segment 0, tau=0): velocity, accel, jerk = 0 (d=1,2,3).
    // End   (segment kNumSegments-1, tau=1): same three derivatives = 0.
    // These encode "hover-to-hover" flight: the quadrotor starts and ends at
    // rest, not merely at the right position (THEORY.md §the-problem).
    for (int d = 1; d <= kMaxEndpointDeriv; ++d) {
        const int row_start = 8 + (d - 1);
        A[row_start * kSysN + 0 * kCoeffsPerSegment + d] = falling_factorial_dev(d, d);
        b[row_start] = 0.0f;

        const int row_end = 8 + kMaxEndpointDeriv + (d - 1);
        const int last_seg = kNumSegments - 1;
        // j's trip count depends on d (a variable-bound loop) — left as a
        // plain loop; the compiler is free to unroll it once d is known at
        // an outer unrolled level, we do not force it (see zeroing-loop note).
        for (int j = d; j < kCoeffsPerSegment; ++j)
            A[row_end * kSysN + last_seg * kCoeffsPerSegment + j] = falling_factorial_dev(j, d);
        b[row_end] = 0.0f;
    }

    // ---- Rows 14..31: interior continuity ----------------------------------
    // 3 interior waypoints (between segments m and m+1, m=0,1,2); each pins
    // derivatives d=1..6 (velocity..pop) continuous across the join. Why all
    // the way to d=6 and not fewer: kernels.cuh's header derives the exact
    // degrees-of-freedom count (8 + 6 + 3*6 = 32) that FORCES this — with
    // only vel/acc/jerk (d<=3) pinned at the interior, the system would be
    // underdetermined (short 9 equations for 32 unknowns).
#pragma unroll
    for (int m = 0; m < kNumInterior; ++m) {
#pragma unroll
        for (int d = 1; d <= kMaxContinuityDeriv; ++d) {
            const int row = 14 + m * kMaxContinuityDeriv + (d - 1);
            // Left segment m, evaluated at tau=1: every c_{m,j}, j=d..7.
            // (variable-bound loop — not force-unrolled, see the earlier note)
            for (int j = d; j < kCoeffsPerSegment; ++j)
                A[row * kSysN + m * kCoeffsPerSegment + j] = falling_factorial_dev(j, d);
            // Right segment m+1, evaluated at tau=0: only c_{m+1,d}, negated
            // (continuity means LEFT minus RIGHT equals zero).
            A[row * kSysN + (m + 1) * kCoeffsPerSegment + d] = -falling_factorial_dev(d, d);
            b[row] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// solve_minsnap_system — Gaussian elimination WITH PARTIAL PIVOTING on the
// 32x32 system built above, in place: A and b are destroyed, x receives the
// solution. Line-by-line twin of reference_cpu.cpp's solve_minsnap_system
// (only fabsf/sqrtf-vs-std:: spellings differ, the usual reason GPU/CPU
// comparisons use a tolerance rather than bit-equality — THEORY.md
// §numerical-considerations).
//
// Why partial pivoting (unlike 33.01's Cholesky, which needed none): this
// matrix is NOT symmetric positive definite — it mixes position rows
// (small integer weights) with high-derivative continuity rows (weights up
// to 7!/1! = 5040), so the naive "divide by whatever is on the diagonal"
// elimination can hit a near-zero or badly-scaled pivot. Partial pivoting
// (swap in the largest-magnitude candidate from the remaining rows of the
// SAME column before dividing) is the standard, cheap fix — THEORY.md
// walks the conditioning story in full, including why working in
// NORMALIZED tau (not real seconds) is what keeps this problem tractable
// in FP32 at all.
//
// Returns true if every pivot cleared the singularity floor (this
// constraint layout is provably nonsingular for any waypoint values, so a
// false here would mean a real bug, not a bad waypoint set); on false, the
// caller fills the output with NaN (33.01's fail-loud policy) rather than
// silently returning a wrong answer.
// ---------------------------------------------------------------------------
__device__ bool solve_minsnap_system(float* A, float* b, float* x)
{
    constexpr int N = kSysN;
    constexpr float kSingularFloor = 1e-6f;   // relative to typical pivot magnitudes (O(1)..O(5040)); see THEORY.md

    // ---- Forward elimination with partial pivoting -------------------------
    for (int col = 0; col < N; ++col) {
        int piv = col;
        float best = fabsf(A[col * N + col]);
        for (int r = col + 1; r < N; ++r) {
            const float v = fabsf(A[r * N + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < kSingularFloor) return false;    // should never trigger — see comment above

        if (piv != col) {
            for (int c = col; c < N; ++c) {
                const float tmp = A[col * N + c];
                A[col * N + c] = A[piv * N + c];
                A[piv * N + c] = tmp;
            }
            const float tmp_b = b[col];
            b[col] = b[piv];
            b[piv] = tmp_b;
        }

        const float inv_pivot = 1.0f / A[col * N + col];  // ONE division per column, reused below
        for (int r = col + 1; r < N; ++r) {
            const float factor = A[r * N + col] * inv_pivot;
            if (factor != 0.0f) {   // most rows are already structurally zero in column `col` — skip real work
                for (int c = col; c < N; ++c)
                    A[r * N + c] = fmaf(-factor, A[col * N + c], A[r * N + c]);
                b[r] = fmaf(-factor, b[col], b[r]);
            }
        }
    }

    // ---- Back substitution ---------------------------------------------------
    for (int row = N - 1; row >= 0; --row) {
        float sum = b[row];
        for (int c = row + 1; c < N; ++c)
            sum = fmaf(-A[row * N + c], x[c], sum);
        x[row] = sum / A[row * N + row];
    }
    return true;
}

// ===========================================================================
// The batched minimum-snap kernel: one thread = one waypoint set, both axes.
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x + threadIdx.x
// owns waypoint set k. Grid: ceil(K/256) x 256 (repo default; ragged tail
// guarded, matching every other batched kernel in this repository).
//
// Memory spaces per thread:
//   registers : wp_x[5], wp_y[5] (10 floats — the waypoint coordinates)
//   LOCAL     : A[1024], b[32], x[32] — see the file header for why this
//               is local memory, not registers, and why that is still the
//               right call at K ~ 10,000.
//   global    : one strided read of 10 floats (waypoints), one coalesced-
//               ish write of 64 floats (coeffs) per thread — see the
//               launcher comment below for the coalescing honesty note.
// No shared memory (waypoint sets share nothing with each other), no
// atomics, no divergence beyond the tail guard and the (never-taken in
// practice) singular-system branch.
// ===========================================================================
__global__ void minsnap_batch_kernel(const float* __restrict__ waypoints,  // [K*10] batch-contiguous, layout: kernels.cuh
                                     float*       __restrict__ coeffs,     // [K*64] OUT, layout: kernels.cuh
                                     int K)                                // number of waypoint sets
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's waypoint-set index
    if (k >= K) return;                                    // ragged-tail guard

    // Load this problem's 5 waypoints, split into per-axis coordinate
    // arrays. This is a STRIDED read (each thread's 10 floats are
    // contiguous, but consecutive THREADS' 10-float blocks are also
    // contiguous, so a warp's reads DO coalesce into 320-byte-aligned
    // bursts) — the interleaved x,y layout costs nothing here because we
    // read the whole small block per thread rather than striding across
    // threads within one waypoint's fields.
    float wp_x[kNumWaypoints], wp_y[kNumWaypoints];
#pragma unroll
    for (int i = 0; i < kNumWaypoints; ++i) {
        wp_x[i] = waypoints[k * kWaypointFloatsPerSet + i * 2 + 0];
        wp_y[i] = waypoints[k * kWaypointFloatsPerSet + i * 2 + 1];
    }

    // The 32x32 system, its RHS, and the solution — LOCAL memory (see file
    // header). Declared once and REUSED across both axes: assemble()
    // rezeroes A fully, so there is no stale state from the x solve
    // leaking into the y solve.
    float A[kSysN * kSysN];
    float b[kSysN];
    float x[kSysN];

    assemble_minsnap_system(wp_x, A, b);
    const bool ok_x = solve_minsnap_system(A, b, x);
#pragma unroll
    for (int i = 0; i < kSysN; ++i)
        coeffs[k * kCoeffsPerSet + 0 * kSysN + i] = ok_x ? x[i] : CUDART_NAN_F;

    assemble_minsnap_system(wp_y, A, b);   // A, b fully rebuilt — safe to reuse the same local arrays
    const bool ok_y = solve_minsnap_system(A, b, x);
#pragma unroll
    for (int i = 0; i < kSysN; ++i)
        coeffs[k * kCoeffsPerSet + 1 * kSysN + i] = ok_y ? x[i] : CUDART_NAN_F;
}

// ===========================================================================
// Host launcher (declared in kernels.cuh).
// ===========================================================================
void launch_minsnap_batch(int K, const float* d_waypoints, float* d_coeffs)
{
    if (K < 1 || !d_waypoints || !d_coeffs) {
        std::fprintf(stderr, "launch_minsnap_batch: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }

    const int threads = 256;                          // repo default: warp multiple, light per-block resource cost
    const int blocks = (K + threads - 1) / threads;    // ceil(K/threads); ragged tail guarded in the kernel
    minsnap_batch_kernel<<<blocks, threads>>>(d_waypoints, d_coeffs, K);
    CUDA_CHECK_LAST_ERROR("minsnap_batch_kernel launch");
}
