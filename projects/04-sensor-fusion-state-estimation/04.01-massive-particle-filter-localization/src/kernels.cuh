// ===========================================================================
// kernels.cuh — interface for project 04.01
//               Massive particle filter localization
//               (teaching core: 2-D range-beam Monte Carlo localization)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the filter loop), kernels.cu (the GPU
// predict/weight kernels), and reference_cpu.cpp (their line-by-line CPU
// twins). Everything all three must agree on — frames, particle layout,
// the motion model, the sensor model, the noise generator, and every model
// constant — is defined HERE, once (CLAUDE.md §12).
//
// The particle filter in five lines (THEORY.md derives it properly):
//   1. Keep K weighted pose hypotheses ("particles") (x, y, theta).
//   2. PREDICT: push every particle through the odometry twist + noise.
//   3. WEIGHT: for every particle, ray-cast what the range sensor WOULD
//      see from that pose and score it against what it DID see.
//   4. ESTIMATE: the weighted mean of the cloud is the pose estimate.
//   5. RESAMPLE: clone high-weight particles, drop low-weight ones; repeat.
// Steps 2–3 are O(K) independent per-particle work — one GPU thread per
// particle — and step 3 does ~1,000 map lookups per particle. That is why
// the catalog calls for 10^5–10^6 particles "GPU likelihoods": a CPU core
// manages thousands of particles per scan; the GPU manages a million.
// Resampling (step 5) stays on the HOST in this teaching version — O(K),
// trivial, and keeping it in plain C++ keeps the algorithm visible
// (the GPU prefix-sum version is README Exercise 5).
//
// FRAMES & UNITS (SI, CLAUDE.md §12) — shared with the data generator:
//   World frame: origin at the map's lower-left corner, x right, y up,
//   right-handed; heading theta measured CCW from +x, in radians, kept
//   UNWRAPPED in particle state (only sin/cos of it are ever consumed;
//   the single defined wrap point is main.cu's heading-error helper).
//   Map: occupancy grid, kMapRes m/cell; cell (ix,iy) covers
//   [ix*res,(ix+1)*res) x [iy*res,(iy+1)*res); 0 = free, 1 = occupied.
//
// PARTICLE LAYOUT — structure of arrays (SoA), documented once here:
//   px[k], py[k]  particle k's position (m, world frame)
//   pth[k]        particle k's heading (rad, unwrapped)
//   logw[k]       particle k's UNNORMALIZED log-likelihood (see weight fn)
// Why SoA and not an array of {x,y,th} structs: the kernels read/write one
// field for all particles at a time, so SoA makes every warp access 32
// CONSECUTIVE floats — perfectly coalesced. An array-of-structs would
// stride 12 bytes and waste 2/3 of every memory transaction (the same
// layout lesson 33.01 measured and 08.01 applied to its noise array).
//
// DETERMINISM CONTRACT (THEORY.md §numerics tells the full story):
//   * Per-particle noise comes from an IN-KERNEL xorshift32 seeded by
//     (base seed, particle id, step) — no cuRAND, no state arrays: the
//     draw for particle k at step t is a pure function of (k, t), identical
//     on CPU and GPU (integer ops are bit-exact everywhere).
//   * The ray-march in the weight function is written contraction-safe
//     (lone multiplies + running additions, never a*b+c in one expression)
//     so nvcc cannot fuse it into FMAs the CPU would round differently —
//     given bit-identical poses, CPU and GPU march the SAME cells.
//   * Beam directions are computed in DOUBLE precision then cast to float,
//     shrinking the CPU-vs-GPU trig difference below the float cast step.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH   // classic include guard: safe on every compiler
#define PROJECT_KERNELS_CUH

#include <cstdint>            // uint32_t for the seeds (host and device)

// ---------------------------------------------------------------------------
// Sensor model constants — shared verbatim by the GPU weight kernel, the CPU
// oracle, and (mirrored in Python) scripts/make_synthetic.py. A mismatch here
// would make the filter score measurements against the wrong physics.
// ---------------------------------------------------------------------------
constexpr int   kNumBeams = 16;       // range beams per scan, full 360 deg fan
constexpr float kRMax = 8.0f;         // max sensor range (m); farther = "max return"
constexpr float kRayStep = 0.125f;    // ray-march step (m) = half a map cell —
                                      // fine enough that a beam cannot jump
                                      // OVER an occupied cell diagonally...
                                      // almost (THEORY.md §algorithm owns the
                                      // corner-cutting caveat honestly)
constexpr int   kMaxRaySteps = 64;    // = kRMax / kRayStep (march count cap)
constexpr float kSigmaZ = 0.15f;      // Gaussian range-noise std-dev ASSUMED by
                                      // the filter (m). Deliberately larger than
                                      // the 0.10 m the generator injects: the
                                      // slack absorbs the ray-march quantization
                                      // and keeps weights from collapsing
                                      // (THEORY.md §algorithm, "sigma inflation")

// ---------------------------------------------------------------------------
// Motion model constants (unicycle robot, odometry-driven prediction).
// ---------------------------------------------------------------------------
constexpr float kDt = 0.1f;           // odometry/scan period (s) -> 10 Hz
constexpr float kSigmaV = 0.10f;      // per-particle linear-velocity noise (m/s);
                                      // > the generator's 0.05 odometry noise so
                                      // the cloud stays wider than the true error
constexpr float kSigmaW = 0.12f;      // per-particle angular-velocity noise (rad/s)
constexpr float kSigmaXY = 0.01f;     // additive position "roughening" (m): keeps
                                      // resampled clones from being exact copies,
                                      // the classic defense against sample
                                      // impoverishment (THEORY.md §algorithm)

// ---------------------------------------------------------------------------
// Filter defaults (main.cu may override K from the CLI; everything else is
// part of the taught, tuned setup).
// ---------------------------------------------------------------------------
constexpr int      kDefaultK = 100000;    // particles (the catalog's 10^5 floor;
                                          // --particles 1000000 also works)
constexpr uint32_t kBaseSeed = 42u;       // repo-law seed for ALL noise streams
constexpr float kInitSigmaPos = 0.30f;    // initial cloud spread around the known
constexpr float kInitSigmaTh = 0.15f;     //   start pose (m, rad) — pose TRACKING;
                                          //   global localization is Exercise 4
constexpr float kRmseGateM = 0.15f;       // closed-loop success gate: position
                                          // RMSE vs ground truth must beat this
                                          // (measured ~0.03 m — wide margin on
                                          // purpose, README §Expected output)

constexpr double kPi = 3.14159265358979323846;    // double: beam angles are
constexpr double kTwoPi = 6.28318530717958647692; // computed in double precision

// ---------------------------------------------------------------------------
// launch_pf_predict — propagate all K particles by one odometry twist + noise.
//
//   K       : particle count (>= 1).
//   step    : time-step index (seeds the per-particle noise stream — the same
//             (k, step) pair always draws the same noise, on CPU and GPU).
//   odo_v   : measured linear velocity for this step (m/s, body frame).
//   odo_w   : measured angular velocity (rad/s, CCW positive).
//   d_px/d_py/d_pth : DEVICE pointers, K floats each — particle poses
//             (m, m, rad; world frame; layout contract above), updated
//             IN PLACE:  p += (odo + noise) twist integrated over kDt.
//
// Launch: one thread per particle, 256-thread blocks (grid math + reasoning
// with the kernel in kernels.cu). Cost per thread: ~2 Box–Muller pairs +
// a handful of flops — this kernel is launch-latency-bound, not compute-bound;
// it exists to keep the particles resident on the GPU between weight calls.
// ---------------------------------------------------------------------------
void launch_pf_predict(int K, int step, float odo_v, float odo_w,
                       float* d_px, float* d_py, float* d_pth);

// ---------------------------------------------------------------------------
// launch_pf_weight — score all K particles against one range scan.
//
//   K       : particle count (>= 1).
//   d_px/d_py/d_pth : DEVICE pointers, K floats each — particle poses (READ).
//   d_map   : DEVICE pointer, map_w*map_h bytes — occupancy grid, row-major
//             cell (ix,iy) at map[iy*map_w + ix], 0 free / 1 occupied.
//   map_w/map_h : grid dimensions in cells (sample: 64x64 -> the whole map
//             is 4 KiB — it lives entirely in L1/L2 after first touch).
//   inv_res : 1 / cell size (1/m; sample: 4.0 exactly — chosen so the
//             world->cell conversion is a lone exact multiply).
//   d_scan  : DEVICE pointer, kNumBeams floats — the MEASURED ranges (m),
//             beam b at body-relative angle (-pi + b*2pi/kNumBeams).
//   d_logw  : DEVICE pointer, K floats OUT — per-particle unnormalized
//             log-likelihood:  -sum_b (z_b - zhat_b)^2 / (2*kSigmaZ^2).
//             Log space on purpose: 16 beams multiply 16 Gaussians, and the
//             product underflows float long before the SUM of exponents
//             misbehaves. The host exponentiates after subtracting the max
//             (main.cu step 3) — the same softmax hygiene as 08.01's softmin.
//
// Launch: one thread per particle, 256-thread blocks. This is the project's
// hot loop: kNumBeams ray-marches of up to kMaxRaySteps map lookups each,
// per particle (~10^8 lookups per scan at K = 10^5).
// ---------------------------------------------------------------------------
void launch_pf_weight(int K,
                      const float* d_px, const float* d_py, const float* d_pth,
                      const unsigned char* d_map, int map_w, int map_h,
                      float inv_res, const float* d_scan, float* d_logw);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — line-by-line twins of the kernels,
// sequential over k. main.cu runs both paths on identical inputs at step 0
// and requires agreement within documented tolerances (the §5 gate; the
// exact tolerance reasoning lives with the verify stage in main.cu).
// ---------------------------------------------------------------------------
void pf_predict_cpu(int K, int step, float odo_v, float odo_w,
                    float* px, float* py, float* pth);

void pf_weight_cpu(int K,
                   const float* px, const float* py, const float* pth,
                   const unsigned char* map, int map_w, int map_h,
                   float inv_res, const float* scan, float* logw);

// ---------------------------------------------------------------------------
// Host-side RNG helpers (also defined in reference_cpu.cpp — the SAME
// xorshift32/Box–Muller the twins use). main.cu consumes these for the
// initial particle cloud and the resampling pick, so every random number in
// the whole demo flows from one documented, portable generator family.
//   pf_uniform01_cpu    : advance state, return a float in (0, 1].
//   pf_gaussian_pair_cpu: advance state, return two independent N(0,1) draws.
// ---------------------------------------------------------------------------
float pf_uniform01_cpu(uint32_t& state);
void  pf_gaussian_pair_cpu(uint32_t& state, float& g0, float& g1);

#endif // PROJECT_KERNELS_CUH
