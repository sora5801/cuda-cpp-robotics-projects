// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 04.01
//                     Massive particle filter localization
//                     (2-D range-beam MCL teaching core)
//
// Three jobs in this project (all declared in kernels.cuh):
//
//   1. pf_predict_cpu / pf_weight_cpu — the ORACLE twins of the two GPU
//      kernels: same motion model, same RNG, same ray-march, same beam
//      model, sequential over k. main.cu runs them against the GPU on
//      identical step-0 inputs and requires agreement within documented
//      tolerances — the §5 GPU-vs-CPU gate. pf_weight_cpu is also the
//      honest timing baseline: "a CPU core manages thousands of particles
//      per scan" is measured here, not asserted.
//
//   2. pf_uniform01_cpu / pf_gaussian_pair_cpu — the host copies of the
//      portable RNG, exported so main.cu draws its initial particle cloud
//      and its resampling pick from the SAME generator family. One RNG
//      design for the whole demo, defined in two twinned places.
//
//   3. (implicitly) the readable statement of the algorithm: this file is
//      the version to read FIRST — the kernels are these loops with the
//      `for (k ...)` turned into threads.
//
// The functions below are line-by-line twins of the __device__ versions in
// kernels.cu — deliberate, documented duplication (diff the files: float
// function spellings and __restrict__ aside, the bodies match). The
// ray-march is written CONTRACTION-SAFE on both sides (lone multiplies +
// running adds) so, given bit-identical inputs, the twins visit
// bit-identical map cells — the full reasoning lives with the device copy
// in kernels.cu and in THEORY.md §numerics; it is not repeated here.
//
// Rules for this file: plain C++17, no CUDA headers, no OpenMP, no
// cleverness — if the reference is clever, it can be wrong, and then the
// oracle lies. (Compiled by cl.exe; kernels.cuh is host-safe by design.)
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared constants, layouts, signatures

#include <cmath>         // std::sqrt, std::log, std::sin, std::cos, std::floor

// ---------------------------------------------------------------------------
// The portable RNG family — host twins (device copies + full commentary in
// kernels.cu; the MATH must stay identical, so it is not re-explained here).
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32_host(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float pf_uniform01_cpu(uint32_t& state)          // (0, 1] — never 0, log-safe
{
    return (xorshift32_host(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

void pf_gaussian_pair_cpu(uint32_t& state, float& g0, float& g1)
{
    const double u1 = static_cast<double>(pf_uniform01_cpu(state));
    const double u2 = static_cast<double>(pf_uniform01_cpu(state));
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = kTwoPi * u2;
    g0 = static_cast<float>(r * std::cos(a));
    g1 = static_cast<float>(r * std::sin(a));
}

static inline uint32_t pf_seed_host(uint32_t base, int k, int step)
{
    uint32_t s = base
               ^ (2654435761u * static_cast<uint32_t>(k + 1))
               ^ (972663749u * static_cast<uint32_t>(step + 1));
    if (s == 0u) s = 0x9E3779B9u;
    xorshift32_host(s);
    xorshift32_host(s);
    return s;
}

// ---------------------------------------------------------------------------
// pf_predict_cpu — all K particles through the motion model, one after
// another (the GPU gives each its own thread). Same noise draws: particle k
// at step t seeds identically here and on the device, so the only CPU-vs-GPU
// differences are libm trig ulps and FMA contraction in the smooth pose
// update — which is why main.cu's predict gate is a small ABSOLUTE
// tolerance, not bit equality.
// ---------------------------------------------------------------------------
void pf_predict_cpu(int K, int step, float odo_v, float odo_w,
                    float* px, float* py, float* pth)
{
    for (int k = 0; k < K; ++k) {
        uint32_t rng = pf_seed_host(kBaseSeed, k, step);
        float gv, gw, gx, gy;
        pf_gaussian_pair_cpu(rng, gv, gw);     // twist noise
        pf_gaussian_pair_cpu(rng, gx, gy);     // positional roughening

        const float v = odo_v + kSigmaV * gv;
        const float w = odo_w + kSigmaW * gw;

        const float th = pth[k];               // OLD heading for the position step
        px[k] += v * std::cos(th) * kDt + kSigmaXY * gx;
        py[k] += v * std::sin(th) * kDt + kSigmaXY * gy;
        pth[k] = th + w * kDt;                 // heading turns after — twin order
    }
}

// ---------------------------------------------------------------------------
// raycast_range_host — expected range by fixed-step marching; twin of
// raycast_range_dev (kernels.cu carries the physics and the
// contraction-safety commentary). std::floor on a float returns the same
// value floorf does; the (int) cast then matches the device exactly.
// ---------------------------------------------------------------------------
static float raycast_range_host(float x, float y, double ang,
                                const unsigned char* map,
                                int map_w, int map_h, float inv_res)
{
    const float dir_x = static_cast<float>(std::cos(ang));
    const float dir_y = static_cast<float>(std::sin(ang));
    const float step_x = dir_x * kRayStep;     // lone multiplies — exact 2^-3 scaling
    const float step_y = dir_y * kRayStep;

    float sx = x;
    float sy = y;
    float r = 0.0f;

    for (int i = 0; i < kMaxRaySteps; ++i) {
        sx += step_x;                          // pure adds — same rounding as the GPU
        sy += step_y;
        r += kRayStep;
        const int ix = static_cast<int>(std::floor(sx * inv_res));
        const int iy = static_cast<int>(std::floor(sy * inv_res));
        if (ix < 0 || iy < 0 || ix >= map_w || iy >= map_h)
            return r;                          // off the map: treat as a hit
        if (map[iy * map_w + ix])
            return r;                          // first occupied cell ends the beam
    }
    return kRMax;                              // max return
}

// ---------------------------------------------------------------------------
// pf_weight_cpu — all K particles scored against the scan, sequentially.
// This loop is the demo's honest CPU baseline for the [time] line: the same
// ~10^8 map lookups the kernel does per scan at K = 10^5, on one core.
// ---------------------------------------------------------------------------
void pf_weight_cpu(int K,
                   const float* px, const float* py, const float* pth,
                   const unsigned char* map, int map_w, int map_h,
                   float inv_res, const float* scan, float* logw)
{
    for (int k = 0; k < K; ++k) {
        const float x = px[k];
        const float y = py[k];
        const double th = static_cast<double>(pth[k]);   // beam angles in double — twin

        float sq_sum = 0.0f;                   // sum of squared innovations (m^2)
        for (int b = 0; b < kNumBeams; ++b) {
            const double ang = th + (static_cast<double>(b) * (kTwoPi / kNumBeams) - kPi);
            const float z_hat = raycast_range_host(x, y, ang, map, map_w, map_h, inv_res);
            const float dz = scan[b] - z_hat;
            sq_sum += dz * dz;
        }
        logw[k] = -sq_sum * (1.0f / (2.0f * kSigmaZ * kSigmaZ));
    }
}
