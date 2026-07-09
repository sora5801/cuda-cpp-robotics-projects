// ===========================================================================
// kernels.cu — GPU implementation for project 04.01
//              Massive particle filter localization
//              (teaching core: 2-D range-beam Monte Carlo localization)
//
// The big idea
// ------------
// A particle filter holds K pose hypotheses and, every sensor tick, asks
// each one two questions: "where would you be now, given the odometry?"
// (PREDICT) and "how well does the scan the robot just took match the scan
// YOU would have taken from your pose?" (WEIGHT). Both questions are
// per-particle-independent, so the GPU mapping is the by-now-familiar
// thread-per-problem pattern (33.01/09.01/08.01) — here the "problem" is
// one pose hypothesis, and the weight question costs ~1,000 occupancy-grid
// lookups (16 beams x up to 64 ray-march steps). At K = 100,000 that is
// ~10^8 lookups per scan: the arithmetic-free, memory-latency-soaked kind
// of work GPUs hide effortlessly and CPUs grind through.
//
// What is NEW here beyond 33.01/09.01/07.09/08.01:
//   * an IN-KERNEL counter-based RNG: noise for particle k at step t is a
//     pure function of (seed, k, t) — no cuRAND, no per-thread state arrays,
//     bit-identical on CPU and GPU (integer ops are exact everywhere);
//   * a DIVERGENT inner loop (each ray stops when IT hits a wall): warps
//     execute until their slowest lane finishes — tolerated and measured
//     honestly rather than hidden (THEORY.md §GPU-mapping discusses the
//     sorted-by-pose and likelihood-field remedies production uses);
//   * CONTRACTION-SAFE arithmetic in the ray-march: the marching position
//     advances by running additions of pre-scaled steps (never a*b+c in one
//     expression), so nvcc cannot fuse what MSVC would round differently —
//     the march is discontinuous (one cell decides hit/miss), so unlike
//     08.01's smooth costs, a last-ulp difference here could flip a whole
//     beam. Bit-identical inputs => bit-identical cells visited.
//
// All constants, layouts, and frames come from kernels.cuh — the single
// source shared with the CPU oracle; every function below is a deliberate
// line-by-line twin of the one in reference_cpu.cpp.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// The portable RNG family (device twins of reference_cpu.cpp's host copies).
//
// xorshift32 — Marsaglia's 3-shift generator: full 2^32-1 period, three
// integer ops, and (the property we exploit) bit-identical output on every
// compiler and device. Not a statistical marvel — fine for exploration
// noise, and README Exercise 3 swaps in cuRAND Philox to compare.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t xorshift32_dev(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// uniform01 — map the top 24 bits to (0, 1]: never returns 0, so log() in
// Box–Muller below is always safe. Same construction as the repo's other
// projects (08.01) — top bits because xorshift's low bits are its weakest.
__device__ __forceinline__ float uniform01_dev(uint32_t& state)
{
    return (xorshift32_dev(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// gaussian_pair — one Box–Muller transform => TWO independent N(0,1) draws.
// The transcendental step runs in DOUBLE and is cast to float at the end:
// cheap insurance for the tails, and it shrinks the CPU-vs-GPU libm
// difference to below the float rounding step (THEORY.md §numerics).
__device__ __forceinline__ void gaussian_pair_dev(uint32_t& state, float& g0, float& g1)
{
    const double u1 = static_cast<double>(uniform01_dev(state));
    const double u2 = static_cast<double>(uniform01_dev(state));
    const double r = sqrt(-2.0 * log(u1));   // radius of the 2-D Gaussian point
    const double a = kTwoPi * u2;            // its angle — uniform on the circle
    g0 = static_cast<float>(r * cos(a));
    g1 = static_cast<float>(r * sin(a));
}

// pf_seed — the counter-based trick: derive particle k's step-t RNG state
// directly from (base, k, t). Two large odd multipliers spread consecutive
// ids/steps across the state space; two warm-up rounds decorrelate the
// still-similar seeds xorshift is sensitive to. Twin in reference_cpu.cpp.
__device__ __forceinline__ uint32_t pf_seed_dev(uint32_t base, int k, int step)
{
    uint32_t s = base
               ^ (2654435761u * static_cast<uint32_t>(k + 1))      // Knuth's 2^32/phi
               ^ (972663749u * static_cast<uint32_t>(step + 1));   // another large odd constant
    if (s == 0u) s = 0x9E3779B9u;          // xorshift32's one forbidden state is 0
    xorshift32_dev(s);                     // warm-up round 1
    xorshift32_dev(s);                     // warm-up round 2
    return s;
}

// ===========================================================================
// PREDICT kernel: one thread = one particle through the motion model.
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x + threadIdx.x
// owns particle k. Grid: ceil(K/256) x 256 (repo default; tail guarded).
//
// Memory spaces: three coalesced read-modify-writes (px, py, pth — the SoA
// layout contract in kernels.cuh makes each a consecutive-float warp
// access); everything else lives in registers. No shared memory, no
// atomics: particles never interact during prediction — by construction.
//
// The model (unicycle, Euler, order matters and is shared with the data
// generator): position advances along the OLD heading, then the heading
// turns. Noise: the odometry twist is perturbed per particle (kSigmaV/W) —
// the filter's admission that encoders lie — plus a small positional
// "roughening" (kSigmaXY) so resampled clones separate (THEORY §algorithm).
// ===========================================================================
__global__ void pf_predict_kernel(int K, int step, float odo_v, float odo_w,
                                  float* __restrict__ px,    // [K] x (m), in/out
                                  float* __restrict__ py,    // [K] y (m), in/out
                                  float* __restrict__ pth)   // [K] theta (rad, unwrapped), in/out
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's particle
    if (k >= K) return;                                    // ragged-tail guard

    // Deterministic per-(particle, step) noise: 2 Box–Muller pairs = 4 draws.
    uint32_t rng = pf_seed_dev(kBaseSeed, k, step);
    float gv, gw, gx, gy;
    gaussian_pair_dev(rng, gv, gw);        // twist noise
    gaussian_pair_dev(rng, gx, gy);        // positional roughening

    const float v = odo_v + kSigmaV * gv;  // THIS particle's belief of the twist
    const float w = odo_w + kSigmaW * gw;

    const float th = pth[k];               // OLD heading — used for the position step
    px[k] += v * cosf(th) * kDt + kSigmaXY * gx;
    py[k] += v * sinf(th) * kDt + kSigmaXY * gy;
    pth[k] = th + w * kDt;                 // heading turns AFTER the position step
    // (theta stays unwrapped on purpose — only sin/cos of it are consumed;
    // kernels.cuh documents the single wrap point, and it is not here.)
}

// ===========================================================================
// The sensor model: expected range by fixed-step ray-marching (device twin).
//
// From pose (x, y) along world angle `ang`: step kRayStep at a time (first
// sample ONE step out), return the range at the first occupied or off-map
// cell, else kRMax ("max return"). Off-map counts as a hit because the map
// has border walls — a ray only leaves the map if the particle itself is
// already outside, and such particles deserve their terrible score.
//
// CONTRACTION SAFETY (the reason this function looks the way it does):
// the position advances by running ADDS of pre-scaled steps (step_x/step_y,
// each produced by a LONE multiply), and the cell index by a LONE multiply
// plus floorf. None of these expressions has the a*b+c shape, so nvcc has
// nothing to fuse into an FMA — given bit-identical (x, y, ang inputs to
// the float cast), CPU and GPU visit bit-identical cells. A fused
// `x0 + i*step` form would round differently on the two paths, and this
// function is DISCONTINUOUS in its inputs: one flipped cell near a corner
// can change the returned range by meters, not ulps (THEORY.md §numerics).
//
// Beam direction in DOUBLE: |cos_double - cos_msvc| is a couple of double
// ulps (~1e-16), far below the float cast's rounding step (~6e-8), so the
// float direction is identical on both paths essentially always.
// ===========================================================================
__device__ __forceinline__ float raycast_range_dev(
    float x, float y, double ang,                    // start pose (m) + world beam angle (rad)
    const unsigned char* __restrict__ map,           // [map_h*map_w] occupancy, row-major
    int map_w, int map_h, float inv_res)             // grid dims (cells), 1/cell-size (1/m)
{
    const float dir_x = static_cast<float>(cos(ang));  // double trig, float cast (see above)
    const float dir_y = static_cast<float>(sin(ang));
    const float step_x = dir_x * kRayStep;   // lone multiplies — nothing to contract;
    const float step_y = dir_y * kRayStep;   // kRayStep = 0.125 = 2^-3, so these are exact scalings

    float sx = x;         // marching sample position (m) — advances by pure adds
    float sy = y;
    float r = 0.0f;       // range marched so far (m); 0.125 sums are exact in FP32

    for (int i = 0; i < kMaxRaySteps; ++i) {
        sx += step_x;                                   // pure add: same rounding on CPU & GPU
        sy += step_y;
        r += kRayStep;
        const int ix = static_cast<int>(floorf(sx * inv_res));  // lone multiply + floor:
        const int iy = static_cast<int>(floorf(sy * inv_res));  // world (m) -> cell index
        if (ix < 0 || iy < 0 || ix >= map_w || iy >= map_h)
            return r;                                   // left the map: treat as a hit
        if (map[iy * map_w + ix])
            return r;                                   // first occupied cell: the beam ends here
    }
    return kRMax;                                       // nothing within range: max return
}

// ===========================================================================
// WEIGHT kernel: one thread = one particle scored against the whole scan.
//
// Thread-to-data mapping: thread k owns particle k; it marches all
// kNumBeams rays sequentially. Grid: ceil(K/256) x 256.
//
// Memory spaces per thread:
//   registers : pose, direction, marching state (~30 regs)
//   global    : px/py/pth[k]  — coalesced SoA reads (one each);
//               scan[b]       — UNIFORM read (all threads, same address):
//                               L2/read-only-cache broadcast, like 08.01's
//                               u_nom[t];
//               map[iy*w+ix]  — the interesting one: DIVERGENT reads, but
//                               the whole sample map is 4 KiB, so after
//                               first touch every lookup is an L1/L2 hit —
//                               the polar opposite of 07.09's cache-hostile
//                               big-grid access pattern, worth comparing;
//               logw[k]       — one coalesced write at the end.
// Divergence, honestly: lanes in a warp hold DIFFERENT poses, so their rays
// hit walls after different step counts; SIMT predication makes the warp
// pay for its slowest lane, every beam. With near-converged clouds the
// poses (hence ranges) are similar and the waste is small; a freshly
// scattered cloud pays more. Production remedies in THEORY §GPU-mapping.
// ===========================================================================
__global__ void pf_weight_kernel(int K,
                                 const float* __restrict__ px,    // [K] x (m)
                                 const float* __restrict__ py,    // [K] y (m)
                                 const float* __restrict__ pth,   // [K] theta (rad)
                                 const unsigned char* __restrict__ map,  // [map_h*map_w] occupancy
                                 int map_w, int map_h, float inv_res,
                                 const float* __restrict__ scan,  // [kNumBeams] measured ranges (m)
                                 float* __restrict__ logw)        // [K] OUT: log-likelihoods
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's particle
    if (k >= K) return;                                    // ragged-tail guard

    const float x = px[k];
    const float y = py[k];
    const double th = static_cast<double>(pth[k]);   // promoted once: beam angles are
                                                     // computed in double (see raycast)

    // Sum of squared range innovations over the fan. Accumulated raw and
    // scaled ONCE at the end: one multiply instead of sixteen, and the
    // single spot where GPU FMA contraction may differ from the CPU's
    // two-step rounding is this smooth accumulation — harmless at ~1e-7
    // relative, unlike the discontinuous march (which is kept exact).
    float sq_sum = 0.0f;
    for (int b = 0; b < kNumBeams; ++b) {
        // Beam b's world angle: theta + (-pi + b*2pi/16), all in double.
        const double ang = th + (static_cast<double>(b) * (kTwoPi / kNumBeams) - kPi);
        const float z_hat = raycast_range_dev(x, y, ang, map, map_w, map_h, inv_res);
        const float dz = scan[b] - z_hat;    // innovation: measured minus expected (m)
        sq_sum += dz * dz;
    }

    // Gaussian beam model in log space (kernels.cuh explains why log):
    //   log p(scan | pose) = -sum_b dz_b^2 / (2 sigma_z^2)  (+ const, dropped)
    logw[k] = -sq_sum * (1.0f / (2.0f * kSigmaZ * kSigmaZ));
}

// ===========================================================================
// Host launchers (declared in kernels.cuh). Both own the same grid math and
// the mandatory post-launch error check (CLAUDE.md §6.1 rule 7).
// ===========================================================================
void launch_pf_predict(int K, int step, float odo_v, float odo_w,
                       float* d_px, float* d_py, float* d_pth)
{
    if (K < 1 || !d_px || !d_py || !d_pth) {
        std::fprintf(stderr, "launch_pf_predict: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;                      // repo default geometry (warp multiple,
    const int blocks = (K + threads - 1) / threads;  // good occupancy on sm_75..sm_89)
    pf_predict_kernel<<<blocks, threads>>>(K, step, odo_v, odo_w, d_px, d_py, d_pth);
    CUDA_CHECK_LAST_ERROR("pf_predict_kernel launch");
}

void launch_pf_weight(int K,
                      const float* d_px, const float* d_py, const float* d_pth,
                      const unsigned char* d_map, int map_w, int map_h,
                      float inv_res, const float* d_scan, float* d_logw)
{
    if (K < 1 || !d_px || !d_py || !d_pth || !d_map || !d_scan || !d_logw
        || map_w < 1 || map_h < 1 || inv_res <= 0.0f) {
        std::fprintf(stderr, "launch_pf_weight: invalid arguments (K=%d, map %dx%d)\n",
                     K, map_w, map_h);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;
    const int blocks = (K + threads - 1) / threads;
    pf_weight_kernel<<<blocks, threads>>>(K, d_px, d_py, d_pth,
                                          d_map, map_w, map_h, inv_res,
                                          d_scan, d_logw);
    CUDA_CHECK_LAST_ERROR("pf_weight_kernel launch");
}
