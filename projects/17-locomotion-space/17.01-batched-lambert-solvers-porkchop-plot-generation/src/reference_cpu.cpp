// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 17.01
//                     Batched Lambert solvers + porkchop plot generation
//
// The oracle twin of kernels.cu's batched solver: the SAME per-cell
// algorithm (body_state, Stumpff series, the identical bisection bracket
// and fixed iteration count from kernels.cuh), evaluated sequentially over
// every grid cell instead of one-thread-per-cell. main.cu runs both on the
// committed scenario and requires agreement within a documented tolerance
// (CLAUDE.md §5's GPU-vs-CPU gate) — a real bug (wrong index arithmetic, a
// mis-transcribed sign, a bracket typo) shifts SOME cell's answer by much
// more than float rounding ever could, so the gate is sharp in practice.
//
// Why a CPU twin exists at all (CLAUDE.md §5, repeated here because this
// project's "GPU version" is unusually easy to get subtly wrong — a sign
// error in the atan2 transfer-angle formula, for instance, silently turns
// every "short way" cell into a "long way" cell and vice versa, and NOTHING
// about the code crashes; only a disagreeing oracle catches it):
//   1) CORRECTNESS ORACLE — a dead-simple sequential version a reader can
//      verify by eye and step through cell by cell in a debugger.
//   2) TEACHING BASELINE — read this file first, then kernels.cu; the only
//      real differences are __device__ qualifiers, sinf/cosf vs
//      std::sin/std::cos, and sincosf (a CUDA-only intrinsic with no
//      portable host equivalent, so this file calls std::sin/std::cos
//      separately — a one-line, documented, harmless divergence).
//
// Rules for this file: plain C++17, no CUDA headers (this file is compiled
// by the HOST compiler, cl.exe — kernels.cuh carries no __global__/
// __device__ declarations at all in this project, so no __CUDACC__ fence
// is even needed here, unlike some sibling projects).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // shared scenario struct, status codes, algorithm constants

#include <cmath>         // std::sin/cos/sqrt/atan2/hypot/fabs — float overloads throughout

// ---------------------------------------------------------------------------
// Host twins of the device model functions (kernels.cu carries the full
// physics/numerics commentary — not repeated here; the MATH must stay
// identical between the two files, which is the entire point).
// ---------------------------------------------------------------------------

static float orbital_rate(float r)
{
    return 1.0f / std::sqrt(r * r * r);   // no host rsqrtf(); this is the
                                           // portable equivalent — a divide
                                           // plus a sqrt instead of one
                                           // reciprocal-sqrt instruction.
}

static void body_state(float r, float n, float t, Vec2* pos, Vec2* vel)
{
    const float theta = n * t;
    const float s = std::sin(theta), c = std::cos(theta);   // two calls: no
                                                              // portable sincosf()
                                                              // on the host path
    pos->x = r * c;  pos->y = r * s;
    vel->x = -r * n * s;  vel->y = r * n * c;
}

static float stumpff_c(float z)
{
    if (z > 1e-6f) {
        float sq = std::sqrt(z);
        return (1.0f - std::cos(sq)) / z;
    } else if (z < -1e-6f) {
        float sq = std::sqrt(-z);
        return (std::cosh(sq) - 1.0f) / (-z);
    } else {
        return 0.5f - z * (1.0f / 24.0f) + z * z * (1.0f / 720.0f) - z * z * z * (1.0f / 40320.0f);
    }
}

static float stumpff_s(float z)
{
    if (z > 1e-6f) {
        float sq = std::sqrt(z);
        return (sq - std::sin(sq)) / (sq * sq * sq);
    } else if (z < -1e-6f) {
        float sq = std::sqrt(-z);
        return (std::sinh(sq) - sq) / (sq * sq * sq);
    } else {
        return (1.0f / 6.0f) - z * (1.0f / 120.0f) + z * z * (1.0f / 5040.0f) - z * z * z * (1.0f / 362880.0f);
    }
}

static float y_of_z(float z, float r1n, float r2n, float A)
{
    float C = stumpff_c(z);
    float S = stumpff_s(z);
    float y = r1n + r2n + A * (z * S - 1.0f) / std::sqrt(C);
    return y < kYFloor ? kYFloor : y;
}

static float tof_of_z(float z, float r1n, float r2n, float A)
{
    float C = stumpff_c(z);
    float S = stumpff_s(z);
    float y = y_of_z(z, r1n, r2n, A);
    float chi = std::sqrt(y / C);
    return chi * chi * chi * S + A * std::sqrt(y);
}

// ---------------------------------------------------------------------------
// solve_cell_cpu — the sequential twin of kernels.cu's solve_cell(). Same
// classification order, same bracket, same fixed iteration count, same f/g
// recovery — diff this against kernels.cu's solve_cell() line by line.
// ---------------------------------------------------------------------------
static float solve_cell_cpu(int i, int j, const LambertScenario& sc, int* status)
{
    const float dt = sc.window_tu / static_cast<float>(sc.grid_n);
    const float t1 = static_cast<float>(i) * dt;
    const float t2 = static_cast<float>(j) * dt;
    const float tof = t2 - t1;

    if (!(tof > sc.min_tof_tu) || !(tof < sc.max_tof_tu)) {
        *status = kStatusMaskedTof;
        return std::nanf("");
    }

    const float n1 = orbital_rate(sc.r1_au);
    const float n2 = orbital_rate(sc.r2_au);
    Vec2 r1v, v1v, r2v, v2v;
    body_state(sc.r1_au, n1, t1, &r1v, &v1v);
    body_state(sc.r2_au, n2, t2, &r2v, &v2v);
    const float r1n = sc.r1_au;
    const float r2n = sc.r2_au;

    const float dot = r1v.x * r2v.x + r1v.y * r2v.y;
    const float cross_z = r1v.x * r2v.y - r1v.y * r2v.x;
    float dtheta = std::atan2(cross_z, dot);
    if (dtheta < 0.0f) dtheta += 2.0f * kPi;

    if (std::fabs(dtheta - kPi) < kEpsSingularRad) {
        *status = kStatusNearSingular;
        return std::nanf("");
    }
    if (dtheta > kPi) {
        *status = kStatusLongWay;
        return std::nanf("");
    }

    const float A = std::sqrt(2.0f * r1n * r2n) * std::cos(0.5f * dtheta);

    float lo = kBisectZLo, hi = kBisectZHi;
    float flo = tof_of_z(lo, r1n, r2n, A) - tof;
    float fhi = tof_of_z(hi, r1n, r2n, A) - tof;
    if ((flo > 0.0f) == (fhi > 0.0f)) {
        *status = kStatusNonConverged;
        return std::nanf("");
    }
    for (int it = 0; it < kBisectIters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float fmid = tof_of_z(mid, r1n, r2n, A) - tof;
        if ((fmid > 0.0f) == (flo > 0.0f)) { lo = mid; flo = fmid; }
        else                               { hi = mid; }
    }
    const float z = 0.5f * (lo + hi);

    const float y = y_of_z(z, r1n, r2n, A);
    const float f    = 1.0f - y / r1n;
    const float g    = A * std::sqrt(y);
    const float gdot = 1.0f - y / r2n;

    Vec2 vt1, vt2;
    vt1.x = (r2v.x - f * r1v.x) / g;    vt1.y = (r2v.y - f * r1v.y) / g;
    vt2.x = (gdot * r2v.x - r1v.x) / g; vt2.y = (gdot * r2v.y - r1v.y) / g;

    const float dv1 = std::hypot(vt1.x - v1v.x, vt1.y - v1v.y);
    const float dv2 = std::hypot(v2v.x - vt2.x, v2v.y - vt2.y);

    *status = kStatusOk;
    return dv1 + dv2;
}

// ---------------------------------------------------------------------------
// lambert_grid_cpu — every cell, in simple row-major order (declared in
// kernels.cuh). O(grid_n^2) cells times O(kBisectIters) Stumpff evaluations
// each — the honest sequential cost that makes the kernel's parallel speed-
// up mean something (main.cu's [time] line measures both).
// ---------------------------------------------------------------------------
void lambert_grid_cpu(const LambertScenario& sc, float* deltav, int* status)
{
    for (int j = 0; j < sc.grid_n; ++j) {
        for (int i = 0; i < sc.grid_n; ++i) {
            const int idx = j * sc.grid_n + i;
            int st = kStatusOk;
            deltav[idx] = solve_cell_cpu(i, j, sc, &st);
            status[idx] = st;
        }
    }
}
