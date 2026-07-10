// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 15.01
//                     Minimum-snap trajectory optimization batched over
//                     waypoint sets (quadrotor-style 2-D waypoint flight)
//
// One job (declared in kernels.cuh): minsnap_batch_cpu — the ORACLE twin of
// the GPU kernel. Same constraint layout (kernels.cuh is the single source
// of truth for row indices), same Gaussian elimination with partial
// pivoting, sequential over k. main.cu runs it against the GPU on the SAME
// batch and requires agreement within a documented tolerance — the §5
// GPU-vs-CPU gate. It also serves as the honest timing baseline: "a CPU
// manages a few thousand of these sequentially" is MEASURED here, not
// asserted (CLAUDE.md §12).
//
// The assembly and elimination functions below are deliberate, documented
// duplicates of the __device__ versions in kernels.cu (CLAUDE.md §4 self-
// containment rule) — diff the two files and the ONLY differences are
// fabsf/sqrtf-vs-std:: spellings and the missing __device__ qualifiers.
// Keeping the MATH identical is what makes the GPU-vs-CPU comparison in
// main.cu meaningful; keeping the CODE readable-in-isolation is what makes
// this file trustworthy as ground truth (a reader can verify it by eye
// without knowing anything about CUDA).
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared model constants, layouts, signatures

#include <cmath>         // std::fabs
#include <limits>        // std::numeric_limits<float>::quiet_NaN()

// ---------------------------------------------------------------------------
// falling_factorial — host twin of kernels.cu's falling_factorial_dev. See
// that file for the full derivation comment (not repeated here — the MATH
// must stay identical, per the file header above).
// ---------------------------------------------------------------------------
static float falling_factorial(int j, int d)
{
    if (j < d) return 0.0f;
    float r = 1.0f;
    for (int t = 0; t < d; ++t) r *= static_cast<float>(j - t);
    return r;
}

// ---------------------------------------------------------------------------
// assemble_minsnap_system — host twin of kernels.cu's device function.
// Builds the 32x32 constraint matrix A (row-major) and RHS b for one axis
// of one waypoint set. Row layout is kernels.cuh's single source of truth:
// rows 0..7 position, 8..13 endpoint derivatives, 14..31 interior
// continuity. See kernels.cu for the full derivation comments.
// ---------------------------------------------------------------------------
static void assemble_minsnap_system(const float wp[kNumWaypoints], float* A, float* b)
{
    for (int i = 0; i < kSysN * kSysN; ++i) A[i] = 0.0f;

    // Position interpolation, rows 0..7.
    for (int s = 0; s < kNumSegments; ++s) {
        const int row0 = 2 * s, row1 = 2 * s + 1;
        A[row0 * kSysN + s * kCoeffsPerSegment + 0] = 1.0f;
        b[row0] = wp[s];
        for (int j = 0; j < kCoeffsPerSegment; ++j)
            A[row1 * kSysN + s * kCoeffsPerSegment + j] = 1.0f;
        b[row1] = wp[s + 1];
    }

    // Global endpoint conditions (velocity, accel, jerk = 0), rows 8..13.
    for (int d = 1; d <= kMaxEndpointDeriv; ++d) {
        const int row_start = 8 + (d - 1);
        A[row_start * kSysN + 0 * kCoeffsPerSegment + d] = falling_factorial(d, d);
        b[row_start] = 0.0f;

        const int row_end = 8 + kMaxEndpointDeriv + (d - 1);
        const int last_seg = kNumSegments - 1;
        for (int j = d; j < kCoeffsPerSegment; ++j)
            A[row_end * kSysN + last_seg * kCoeffsPerSegment + j] = falling_factorial(j, d);
        b[row_end] = 0.0f;
    }

    // Interior continuity (derivatives 1..6 at each of 3 interior waypoints), rows 14..31.
    for (int m = 0; m < kNumInterior; ++m) {
        for (int d = 1; d <= kMaxContinuityDeriv; ++d) {
            const int row = 14 + m * kMaxContinuityDeriv + (d - 1);
            for (int j = d; j < kCoeffsPerSegment; ++j)
                A[row * kSysN + m * kCoeffsPerSegment + j] = falling_factorial(j, d);
            A[row * kSysN + (m + 1) * kCoeffsPerSegment + d] = -falling_factorial(d, d);
            b[row] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// solve_minsnap_system — host twin of kernels.cu's device function: plain
// Gaussian elimination with partial pivoting, in place. See kernels.cu for
// the full "why partial pivoting / why this can't be singular" commentary.
// ---------------------------------------------------------------------------
static bool solve_minsnap_system(float* A, float* b, float* x)
{
    const int N = kSysN;
    const float kSingularFloor = 1e-6f;

    for (int col = 0; col < N; ++col) {
        int piv = col;
        float best = std::fabs(A[col * N + col]);
        for (int r = col + 1; r < N; ++r) {
            const float v = std::fabs(A[r * N + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < kSingularFloor) return false;

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

        const float inv_pivot = 1.0f / A[col * N + col];
        for (int r = col + 1; r < N; ++r) {
            const float factor = A[r * N + col] * inv_pivot;
            if (factor != 0.0f) {
                for (int c = col; c < N; ++c)
                    A[r * N + c] -= factor * A[col * N + c];
                b[r] -= factor * b[col];
            }
        }
    }

    for (int row = N - 1; row >= 0; --row) {
        float sum = b[row];
        for (int c = row + 1; c < N; ++c)
            sum -= A[row * N + c] * x[c];
        x[row] = sum / A[row * N + row];
    }
    return true;
}

// ---------------------------------------------------------------------------
// minsnap_batch_cpu — all K waypoint sets, one after another (the GPU gives
// each its own thread). Same NaN-on-singular policy as the kernel (never
// expected to trigger for this constraint layout — a NaN in the output
// would mean a genuine bug, not a bad waypoint set).
// ---------------------------------------------------------------------------
void minsnap_batch_cpu(int K, const float* waypoints, float* coeffs)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();

    float A[kSysN * kSysN];
    float b[kSysN];
    float x[kSysN];
    float wp_x[kNumWaypoints], wp_y[kNumWaypoints];

    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < kNumWaypoints; ++i) {
            wp_x[i] = waypoints[k * kWaypointFloatsPerSet + i * 2 + 0];
            wp_y[i] = waypoints[k * kWaypointFloatsPerSet + i * 2 + 1];
        }

        assemble_minsnap_system(wp_x, A, b);
        const bool ok_x = solve_minsnap_system(A, b, x);
        for (int i = 0; i < kSysN; ++i)
            coeffs[k * kCoeffsPerSet + 0 * kSysN + i] = ok_x ? x[i] : nan;

        assemble_minsnap_system(wp_y, A, b);
        const bool ok_y = solve_minsnap_system(A, b, x);
        for (int i = 0; i < kSysN; ++i)
            coeffs[k * kCoeffsPerSet + 1 * kSysN + i] = ok_y ? x[i] : nan;
    }
}
