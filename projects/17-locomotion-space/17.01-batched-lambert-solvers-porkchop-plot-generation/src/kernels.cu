// ===========================================================================
// kernels.cu — GPU implementation for project 17.01
//              Batched Lambert solvers + porkchop plot generation
//
// The big idea
// ------------
// A porkchop plot is 262,144 (512*512) INDEPENDENT questions: "if I leave
// body 1's orbit at epoch t1 and arrive at body 2's orbit at epoch t2, how
// much delta-v does that cost?" Answering one question means solving
// Lambert's problem (a transcendental root-find on the universal anomaly
// z) — cells share no data and write no shared output, so ONE GPU THREAD
// PER CELL is the natural mapping: the exact "one thread = one small
// numerical problem" pattern 33.01 teaches for batched linear algebra,
// here applied to a batched nonlinear ROOT-FIND instead of a batched
// linear solve.
//
// What is NEW here beyond 33.01/09.01/08.01/07.09:
//   * the per-thread work is a FIXED-ITERATION BISECTION on a transcendental
//     equation (33.01/09.01 solve algebra; 08.01 integrates an ODE; this
//     kernel roots a Stumpff-function equation — same "identical scheme on
//     both paths" discipline, different equation);
//   * a documented, GENUINE mathematical singularity inside the batch (the
//     Lambert equations break at a transfer angle of exactly 180 deg) that
//     the kernel must detect and flag rather than silently propagate NaN;
//   * NO INPUT ARRAY IS READ AT ALL — every thread's inputs (t1, t2, the
//     scenario) are either the tiny by-value LambertScenario struct or
//     derived purely from the thread's own index. This is even more
//     favorable than SAXPY's memory-bound profile: this kernel does zero
//     global-memory READS and only ONE coalesced WRITE per output array —
//     it is purely ARITHMETIC-BOUND (trig- and sqrt-heavy), the opposite
//     end of the roofline from a bandwidth-bound map.
//   * WARP DIVERGENCE that is geographically clustered rather than random:
//     the five kStatus* outcomes are NOT scattered independently per cell —
//     they form contiguous REGIONS of the (t1, t2) grid (a masked diagonal
//     band, a long-way half-plane, a thin near-singular ring). Neighboring
//     threads (same warp) usually share a region and hence a status, so
//     divergence cost concentrates at REGION BOUNDARIES, not everywhere
//     (THEORY.md §the-gpu-mapping measures this).
//
// All constants and the scenario layout come from kernels.cuh, the single
// source shared with the CPU oracle; every function below is a deliberate,
// line-by-line twin of the one in reference_cpu.cpp (diff the files — only
// sinf/cosf vs std::sin/std::cos spellings and __device__ qualifiers
// differ, the reason the GPU-vs-CPU comparison is tolerance-based).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cmath>                    // nanf() — a quiet NaN we can hand back per the kStatus* policy
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// orbital_rate — Kepler's third law for a circular orbit: n = sqrt(mu/r^3).
// mu = 1 in canonical units (kernels.cuh), so this is simply r^-1.5. n is
// the body's constant angular rate (rad/TU); THEORY.md §the-math derives
// this from F = m*omega^2*r = G*M*m/r^2.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float orbital_rate(float r)
{
    return rsqrtf(r * r * r);   // rsqrtf: 1/sqrt(x), one hardware-accelerated
                                // reciprocal-sqrt op instead of a divide
                                // after sqrtf — the standard GPU idiom for
                                // "I only ever wanted the reciprocal root".
}

// ---------------------------------------------------------------------------
// body_state — a circular heliocentric orbit's position AND velocity at
// canonical time t, closed form (no integration — a circular orbit is
// exactly solvable, THEORY.md §the-math). Phase convention: angle = n*t,
// i.e. both bodies sit on the +x axis at t = 0 (kernels.cuh header note).
//
//   r   — orbit radius, LU.           n  — angular rate, rad/TU (orbital_rate(r)).
//   t   — canonical time, TU.
//   pos — OUT: heliocentric position, LU.   vel — OUT: heliocentric velocity, LU/TU.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void body_state(float r, float n, float t, Vec2* pos, Vec2* vel)
{
    float s, c;
    sincosf(n * t, &s, &c);     // one call for both sin and cos of the same
                                // angle — cheaper than sinf()+cosf() separately
                                // and exactly what "position AND velocity of a
                                // circular orbit" needs (they share the angle).
    pos->x = r * c;  pos->y = r * s;
    // Velocity is tangential (perpendicular to position), magnitude r*n —
    // differentiate (r cos(nt), r sin(nt)) w.r.t. t directly.
    vel->x = -r * n * s;  vel->y = r * n * c;
}

// ---------------------------------------------------------------------------
// stumpff_c / stumpff_s — the Stumpff functions C(z), S(z) that let the
// universal-variable Lambert formulation cover elliptical (z>0), parabolic
// (z=0), and hyperbolic (z<0) transfers with ONE set of equations (no
// per-conic-type branching in the Lambert solver itself — THEORY.md derives
// the closed forms). Near z = 0 the closed forms are a removable 0/0
// (sin/cos or sinh/cosh of a near-zero argument divided by near-zero z);
// this project uses the Taylor series in that neighborhood instead of
// letting FP32 catastrophically cancel — the numerics lesson THEORY.md
// §numerical-considerations calls "the Stumpff series switchover".
// ---------------------------------------------------------------------------
__device__ __forceinline__ float stumpff_c(float z)
{
    if (z > 1e-6f) {
        float sq = sqrtf(z);
        return (1.0f - cosf(sq)) / z;
    } else if (z < -1e-6f) {
        float sq = sqrtf(-z);
        return (coshf(sq) - 1.0f) / (-z);
    } else {
        // Taylor series C(z) = 1/2 - z/24 + z^2/720 - z^3/40320 + ...
        // (the z=0 case, C(0)=1/2, is just this series' leading term).
        return 0.5f - z * (1.0f / 24.0f) + z * z * (1.0f / 720.0f) - z * z * z * (1.0f / 40320.0f);
    }
}

__device__ __forceinline__ float stumpff_s(float z)
{
    if (z > 1e-6f) {
        float sq = sqrtf(z);
        return (sq - sinf(sq)) / (sq * sq * sq);
    } else if (z < -1e-6f) {
        float sq = sqrtf(-z);
        return (sinhf(sq) - sq) / (sq * sq * sq);
    } else {
        // Taylor series S(z) = 1/6 - z/120 + z^2/5040 - z^3/362880 + ...
        return (1.0f / 6.0f) - z * (1.0f / 120.0f) + z * z * (1.0f / 5040.0f) - z * z * z * (1.0f / 362880.0f);
    }
}

// ---------------------------------------------------------------------------
// y_of_z — the universal formulation's auxiliary quantity (THEORY.md's
// "the math" derives it from the Lagrange f/g coefficients). Clamped to
// kYFloor: the bisection bracket's FAR ends can transiently make the
// bracketed expression slightly negative before the root is found (sqrt of
// a negative number would poison every later iteration with NaN); the
// floor only ever engages away from the true root (kernels.cuh comment).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float y_of_z(float z, float r1n, float r2n, float A)
{
    float C = stumpff_c(z);
    float S = stumpff_s(z);
    float y = r1n + r2n + A * (z * S - 1.0f) / sqrtf(C);
    return y < kYFloor ? kYFloor : y;
}

// ---------------------------------------------------------------------------
// tof_of_z — the time-of-flight (TU) a given universal anomaly z implies,
// for this cell's fixed (r1n, r2n, A). This is F(z) + target_tof in
// root-finding language; the bisection below roots F(z) = tof_of_z(z) - tof.
// Monotonic increasing in z over the elliptical single-revolution branch
// this project searches (THEORY.md §the-algorithm) — the fact that makes
// bisection on a bracketed sign change valid at all.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float tof_of_z(float z, float r1n, float r2n, float A)
{
    float C = stumpff_c(z);
    float S = stumpff_s(z);
    float y = y_of_z(z, r1n, r2n, A);
    float chi = sqrtf(y / C);                    // the universal anomaly's own "radius"
    return chi * chi * chi * S + A * sqrtf(y);    // mu = 1, so no sqrt(mu) divisor
}

// ---------------------------------------------------------------------------
// solve_cell — the whole per-thread computation: classify the cell, and if
// it is a genuine short-way Lambert instance, bisect for z, recover the
// transfer velocities via the Lagrange f/g coefficients, and return the
// total impulsive delta-v. This is a deliberate, line-by-line twin of
// solve_cell_cpu() in reference_cpu.cpp — the function the kernel below
// calls once per thread.
//
//   i, j    — this cell's (departure, arrival) grid indices.
//   sc      — the scenario (grid_n, window, orbit radii, TOF band).
//   status  — OUT: one of the kStatus* codes (kernels.cuh).
// Returns: total delta-v (LU/TU) if *status == kStatusOk, else a quiet NaN.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float solve_cell(int i, int j, const LambertScenario& sc, int* status)
{
    const float dt = sc.window_tu / static_cast<float>(sc.grid_n);
    const float t1 = static_cast<float>(i) * dt;      // departure epoch, TU
    const float t2 = static_cast<float>(j) * dt;      // arrival epoch, TU
    const float tof = t2 - t1;                         // time of flight, TU

    // ---- structural mask: is this even a candidate transfer duration? ----
    if (!(tof > sc.min_tof_tu) || !(tof < sc.max_tof_tu)) {
        // (NaN-safe comparisons written as "!(a > b)" rather than "a <= b":
        // if tof were ever NaN — it never is here, t1/t2 are finite by
        // construction — the negated form still masks instead of solving.)
        *status = kStatusMaskedTof;
        return nanf("");
    }

    const float n1 = orbital_rate(sc.r1_au);
    const float n2 = orbital_rate(sc.r2_au);
    Vec2 r1v, v1v, r2v, v2v;
    body_state(sc.r1_au, n1, t1, &r1v, &v1v);
    body_state(sc.r2_au, n2, t2, &r2v, &v2v);
    const float r1n = sc.r1_au;   // exact for a circular orbit (both bodies
    const float r2n = sc.r2_au;   // never leave their own orbit radius)

    // ---- transfer angle: the PROGRADE (counterclockwise) sweep from r1 to
    // r2, via atan2(cross_z, dot) — robust across the whole circle, unlike
    // acos(dot/(|r1||r2|)) which loses the sign (THEORY.md §the-math). ----
    const float dot = r1v.x * r2v.x + r1v.y * r2v.y;
    const float cross_z = r1v.x * r2v.y - r1v.y * r2v.x;
    float dtheta = atan2f(cross_z, dot);
    if (dtheta < 0.0f) dtheta += 2.0f * kPi;   // fold into [0, 2*pi)

    if (fabsf(dtheta - kPi) < kEpsSingularRad) {
        *status = kStatusNearSingular;          // THE Lambert singularity (kernels.cuh)
        return nanf("");
    }
    if (dtheta > kPi) {
        *status = kStatusLongWay;               // out of scope for v1 (README exercise)
        return nanf("");
    }

    // ---- the Lambert "A" constant (THEORY.md derives this stable form,
    // A = sqrt(2 r1n r2n) * cos(dtheta/2), from the textbook
    // sin(dtheta)*sqrt(r1n r2n/(1-cos dtheta)) via the half-angle identity
    // — the naive form is a 0/0 trap as dtheta -> 0; this one is not). ----
    const float A = sqrtf(2.0f * r1n * r2n) * cosf(0.5f * dtheta);

    // ---- bisection on z: identical bracket and iteration count to
    // reference_cpu.cpp (kernels.cuh's kBisectZLo/Hi/Iters) — the §5
    // GPU-vs-CPU gate depends on both paths taking EXACTLY the same steps. ----
    float lo = kBisectZLo, hi = kBisectZHi;
    float flo = tof_of_z(lo, r1n, r2n, A) - tof;
    float fhi = tof_of_z(hi, r1n, r2n, A) - tof;
    if ((flo > 0.0f) == (fhi > 0.0f)) {
        *status = kStatusNonConverged;          // bracket did not sign-change (rare; counted, not hidden)
        return nanf("");
    }
#pragma unroll 4
    for (int it = 0; it < kBisectIters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float fmid = tof_of_z(mid, r1n, r2n, A) - tof;
        // No early exit (file header comment): every thread in the batch
        // takes exactly kBisectIters steps, on both GPU and CPU.
        if ((fmid > 0.0f) == (flo > 0.0f)) { lo = mid; flo = fmid; }
        else                               { hi = mid; }
    }
    const float z = 0.5f * (lo + hi);

    // ---- recover the transfer velocities via the Lagrange f, g, gdot
    // coefficients (THEORY.md §the-math) — the standard closing step of
    // every universal-variable Lambert solver. ----
    const float y = y_of_z(z, r1n, r2n, A);
    const float f    = 1.0f - y / r1n;
    const float g    = A * sqrtf(y);              // mu = 1
    const float gdot = 1.0f - y / r2n;

    Vec2 vt1, vt2;   // transfer-orbit velocity at departure / arrival
    vt1.x = (r2v.x - f * r1v.x) / g;   vt1.y = (r2v.y - f * r1v.y) / g;
    vt2.x = (gdot * r2v.x - r1v.x) / g; vt2.y = (gdot * r2v.y - r1v.y) / g;

    // ---- total impulsive delta-v: the two instantaneous burns that patch
    // the transfer orbit onto each body's own circular orbit. ----
    const float dv1 = hypotf(vt1.x - v1v.x, vt1.y - v1v.y);
    const float dv2 = hypotf(v2v.x - vt2.x, v2v.y - vt2.y);

    *status = kStatusOk;
    return dv1 + dv2;
}

// ===========================================================================
// The batched Lambert-solve kernel: one thread = one (departure, arrival)
// grid cell. Thread-to-data mapping: idx = blockIdx.x*blockDim.x+threadIdx.x
// owns cell (i, j) = (idx % grid_n, idx / grid_n) — row-major, matching the
// PGM artifact's own layout (kernels.cuh).
//
// Memory per thread: ZERO global reads (every input is index-derived or the
// tiny by-value `sc`, which the launch configuration below broadcasts to
// every thread via kernel parameter space — a fast, cached, read-only
// path); TWO coalesced writes at the very end (deltav[idx], status[idx]).
// No shared memory, no atomics — cells never interact, by construction.
// ===========================================================================
__global__ void lambert_grid_kernel(LambertScenario sc,
                                    float* __restrict__ deltav,   // [grid_n*grid_n] OUT: delta-v (LU/TU) or NaN
                                    int*   __restrict__ status,   // [grid_n*grid_n] OUT: kStatus* code
                                    int total_cells)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_cells) return;                   // ragged-tail guard

    const int i = idx % sc.grid_n;    // departure-epoch index
    const int j = idx / sc.grid_n;    // arrival-epoch index

    int st = kStatusOk;
    const float dv = solve_cell(i, j, sc, &st);
    deltav[idx] = dv;
    status[idx] = st;
}

// ===========================================================================
// Host launcher (declared in kernels.cuh).
// ===========================================================================
void launch_lambert_grid(const LambertScenario& sc, float* d_deltav, int* d_status)
{
    if (sc.grid_n < 1 || !d_deltav || !d_status) {
        std::fprintf(stderr, "launch_lambert_grid: invalid arguments (grid_n=%d)\n", sc.grid_n);
        std::exit(EXIT_FAILURE);
    }

    const int total = sc.grid_n * sc.grid_n;
    const int threads = 256;                          // repo default geometry (warp multiple, good occupancy)
    const int blocks = (total + threads - 1) / threads;
    lambert_grid_kernel<<<blocks, threads>>>(sc, d_deltav, d_status, total);
    CUDA_CHECK_LAST_ERROR("lambert_grid_kernel launch");
}
