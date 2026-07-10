// ===========================================================================
// kernels.cuh — interface for project 32.02
//               CUDA Graphs for jitter-free fixed-rate perception-control loops
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the three-mode measurement study), kernels.cu
// (the eight GPU kernels that make up ONE "tick" of a fixed-rate
// perception-control loop), and reference_cpu.cpp (the plain-C++ twin of the
// whole tick, used as the correctness oracle). Everything all three must
// agree on — sizes, the plant model borrowed from 08.01, the tick's data
// layout, and the measurement-record shapes main.cu logs to CSV — is defined
// HERE, once (CLAUDE.md §12 "one source of truth" rule).
//
// What this project studies (read main.cu's header first if you have not)
// -------------------------------------------------------------------------
// This is NOT a new controller. It is a LATENCY/JITTER MEASUREMENT STUDY of
// how a realistic, small, multi-kernel robot "tick" behaves when the SAME
// eight kernels + two device-to-device copies are launched three different
// ways every 4 ms (250 Hz):
//   (A) twelve individual host API calls, straight to a CUDA stream — the
//       naive way every learner starts with;
//   (B) the identical twelve calls captured ONCE into a CUDA Graph
//       (cudaStreamBeginCapture/cudaStreamEndCapture) and replayed with a
//       single cudaGraphLaunch() every tick;
//   (C) the same graph, but with one input (the raw sensor buffer) double-
//       buffered and repointed every tick via cudaGraphExecKernelNodeSetParams
//       — the technique real systems use when a captured node's ARGUMENT
//       (not just the memory it points at) must legitimately change.
// The three modes must be mathematically IDENTICAL — same kernels, same
// data, same launch geometry — so THEORY.md's numerics section can argue
// (and main.cu's VERIFY stage checks) that their outputs are bit-identical.
// Only the ORCHESTRATION differs, and that is exactly what is measured.
//
// The tick pipeline (eight kernels, deliberately tiny — see README "What
// this computes" for the sizing argument: launch overhead must be a visible
// fraction of the tick, which only happens if the tick's real GPU work is
// small):
//
//   1. sensor_scale_bias_kernel   — perception: raw counts -> physical units (map)
//   2. sensor_smooth_kernel       — perception: 3-tap denoise (stencil)
//   3. state_predict_kernel       — estimation: constant-velocity predict
//   4. state_correct_kernel       — estimation: fixed-gain measurement fusion (reduction + fuse)
//   5. mppi_rollout_kernel        — planning: K rollouts x T steps (the "fat" kernel; 08.01's
//                                   cart-pole dynamics, reused BY NAME at K=512,T=16 instead of
//                                   08.01's K=4096,T=50 — see README System context)
//   6. cost_min_kernel            — planning: softmin's numerical-safety min-reduction
//   7. softmin_weight_kernel      — planning: importance weights + their sum (reduction)
//   8. control_blend_kernel       — control: weighted-noise blend into the nominal plan (reduction)
//   + two cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, ...) "publish" copies (not __global__
//     kernels, but still CUDA API calls captured as graph nodes — see main.cu) that hand the
//     tick's two outputs to stable buffers a downstream consumer could safely read.
//
// STATE LAYOUT — float x[4], reusing 08.01's cart-pole layout VERBATIM (same
// names, same units) because this project's job is to wrap a REPRESENTATIVE
// MPPI-shaped workload, not invent new physics:
//     x[0] = p      generalized position (m)
//     x[1] = pdot   generalized velocity (m/s)
//     x[2] = theta  generalized angle (rad)
//     x[3] = thdot  generalized angular velocity (rad/s)
// This project does NOT close the loop against an evolving simulated plant
// (08.01 already teaches that); see README "Limitations & honesty" for why
// that is the right scope for a measurement study.
//
// NOISE LAYOUT — eps is stored TRANSPOSED, eps[t*K + k], for the same
// coalescing reason 08.01 documents in its kernels.cuh: at rollout step t,
// a warp's K-indexed reads become consecutive floats.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cuda_runtime.h>   // cudaStream_t — every launcher below takes an explicit stream
                            // (never the default stream 0) because CUDA Graph capture
                            // (cudaStreamBeginCapture) requires a stream we fully control;
                            // using the same launcher on the same stream for all three modes
                            // keeps the ORCHESTRATION the only difference between them.

// ---------------------------------------------------------------------------
// Tick-pipeline sizing constants — the SINGLE SOURCE for every buffer size,
// grid/block dimension, and CSV column main.cu, kernels.cu, and
// reference_cpu.cpp all share. Change one number here and every file that
// matters follows (that is the entire point of a ".cuh contract").
//
// Sizing argument (README "What this computes" restates this for the
// learner): the whole tick's GPU work is deliberately kept to roughly
// 100-300 microseconds of DEVICE time so that per-launch host/driver
// overhead (measured in low single-digit microseconds per call on Windows
// WDDM, THEORY.md derives the path) is a VISIBLE fraction of the tick, not
// noise beneath it. That is only true if every kernel here is small; a
// bigger, more "realistic" workload would bury the very effect this project
// exists to measure.
// ---------------------------------------------------------------------------
constexpr int kSensorN   = 512;  // raw sensor channel count (a small reduced-resolution
                                 // range scan — think a down-sampled 2D safety LiDAR ring,
                                 // SYSTEM_DESIGN.md §2.1's AMR sensor suite, shrunk to fit
                                 // one thread block); unitless raw counts on input
constexpr int kHorizon   = 16;   // MPPI horizon T (08.01 uses 50; this project uses a 0.16 s
                                 // look-ahead at kDt below — plenty to shape a small tick,
                                 // far too short for real swing-up, which is not this
                                 // project's job, see README §Limitations)
constexpr int kRollouts  = 512;  // MPPI rollout count K (08.01 uses 4096; 512 keeps the
                                 // "fat" kernel's own cost small enough that launch overhead
                                 // remains visible next to it — the sizing argument above)
constexpr int kNX        = 4;    // state dimension (the 08.01 cart-pole layout, reused)

// ---------------------------------------------------------------------------
// Plant + MPPI constants — copied VERBATIM from 08.01's kernels.cuh (same
// physical plant, same tuned weights) because this project's rollout kernel
// is 08.01's rollout kernel at smaller K/T, not a new derivation. See
// ../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/src/kernels.cuh
// for the physics derivation (THEORY.md §The problem restates the essentials
// this project actually needs: the DAG shape, not the swing-up story).
// ---------------------------------------------------------------------------
constexpr float kMassCart    = 1.0f;
constexpr float kMassPole    = 0.1f;
constexpr float kPoleHalfLen = 0.5f;
constexpr float kGravity     = 9.81f;
constexpr float kUmax        = 10.0f;   // control clamp (N) — applies to the blended plan

constexpr float kDt          = 0.01f;   // rollout integration step (s) -> 0.16 s horizon at T=16
constexpr float kSigma       = 2.5f;    // exploration noise std-dev (N), same as 08.01
constexpr float kLambda      = 0.5f;    // softmin temperature, same as 08.01

// Stage-cost weights — same shape as 08.01's, unchanged (this project never
// tunes the controller; it only needs the SAME arithmetic shape every tick).
constexpr float kWAngle = 10.0f;
constexpr float kWThdot = 0.1f;
constexpr float kWPos   = 0.5f;
constexpr float kWPdot  = 0.05f;
constexpr float kWCtrl  = 0.001f;

// ---------------------------------------------------------------------------
// Perception + estimation constants (stages 1-4). These model a deliberately
// SIMPLIFIED, synthetic sensor-to-measurement relationship — honestly, this
// is NOT a physically rigorous range-to-position model (see README
// §Limitations); its only job is to give the pipeline a realistic multi-
// kernel SHAPE (map -> stencil -> predict -> fuse) with real, checkable
// numerics, not to teach sensor modeling (that is domains 01-03's job).
// ---------------------------------------------------------------------------
constexpr float kSensorGain   = 1.0f / 4096.0f;  // raw counts [0,4095] -> [0,1)
constexpr float kSensorBias   = -0.5f;           // recenter to [-0.5, 0.5)
constexpr float kMeasScale    = 1.0f;            // proxy-measurement scale (unitless -> m, synthetic)
constexpr float kMeasOffset   = 0.0f;
constexpr float kCorrGain     = 0.35f;            // fixed complementary-filter gain (0..1); a real
                                                  // EKF replaces this with a covariance-derived
                                                  // Kalman gain — see THEORY.md §numerics

// ===========================================================================
// Measurement records — what main.cu logs, one row per tick, per mode. These
// are POD structs so a row is a single fwrite-shaped line to CSV; every field
// documents its unit so demo/out/latency_histogram.csv is self-describing
// even opened cold in a spreadsheet.
// ===========================================================================

// One tick's timing, for one mode. All time fields are HOST-clock
// microseconds from QueryPerformanceCounter (Windows) or
// std::chrono::steady_clock (portable fallback) EXCEPT gpu_exec_ms, which is
// the GPU's own device-timeline duration from a pair of cudaEvents — the
// three are deliberately different clocks measuring different things
// (THEORY.md §measurement methodology explains why all three matter).
struct TickMeasurement {
    int    tick;         // tick index within the MEASURED window [0, kMeasuredTicks)
    double submit_us;     // host time spent issuing this tick's GPU work (no waiting)
    double latency_us;    // end-to-end: submit-start to GPU-completion-CONFIRMED (submit + wait)
    double gpu_exec_ms;   // device-timeline duration of this tick's GPU work (cudaEvent pair)
    double period_us;     // wall time since the PREVIOUS tick's scheduled start (pacing achieved)
};

// Aggregate statistics over one mode's measured run — the row main.cu writes
// to demo/out/jitter_summary.csv, and the numbers RESULT: gates on (mean-
// only, conservative margins — see main.cu's gate comments for why tail
// percentiles are reported as [info], never gated).
struct JitterStats {
    double mean, p50, p95, p99, max;   // all in the same unit as the input series (us or ms)
};

// ---------------------------------------------------------------------------
// tick_pipeline_inputs_t — everything ONE tick's GPU work reads that main.cu
// must supply fresh: the raw sensor sample and the rollout exploration
// noise. Both are generated DETERMINISTICALLY on the host from a tick index
// (see main.cu's tick_inputs()) so all three modes see BIT-IDENTICAL inputs
// at the same global tick index — the precondition for the bit-identical
// OUTPUT comparison THEORY.md §numerics argues for.
// ---------------------------------------------------------------------------
struct TickInputsHost {
    float sensor_raw[kSensorN];             // raw counts, see kSensorGain/kSensorBias above
    float eps[kHorizon * kRollouts];        // exploration noise, TRANSPOSED: eps[t*K + k]
};

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees the __global__ declarations ----

// Stage 1 — perception: y[i] = kSensorGain*x[i] + kSensorBias, a pure map.
__global__ void sensor_scale_bias_kernel(const float* __restrict__ x,
                                         float*       __restrict__ y);

// Stage 2 — perception: 3-tap moving-average stencil, clamped boundary.
__global__ void sensor_smooth_kernel(const float* __restrict__ x,
                                     float*       __restrict__ y);

// Stage 3 — estimation: constant-velocity predict (kNX threads; trivially
// small on purpose — see README "the fat kernel is the exception, not the rule").
__global__ void state_predict_kernel(const float* __restrict__ x_est_prev,
                                     float*       __restrict__ x_pred);

// Stage 4 — estimation: mean-of-smoothed-sensor reduction + fixed-gain fuse
// into the predicted state's position channel only.
__global__ void state_correct_kernel(const float* __restrict__ smoothed,
                                     const float* __restrict__ x_pred,
                                     float*       __restrict__ x_est_new);

// Stage 5 — planning, THE FAT KERNEL: K rollouts x T RK4 steps of 08.01's
// cart-pole dynamics, one thread per rollout. See kernels.cu for the full
// physics commentary (deliberately not repeated here — it is 08.01's, cited).
__global__ void mppi_rollout_kernel(const float* __restrict__ x0,      // [kNX]
                                    const float* __restrict__ u_nom,   // [kHorizon]
                                    const float* __restrict__ eps,     // [kHorizon*kRollouts]
                                    float*       __restrict__ cost);   // [kRollouts] OUT

// Stage 6 — planning: single-block shared-memory MIN reduction over the K
// rollout costs (the softmin numerical-safety subtraction 08.01 does on the
// host; here it must live on the GPU or the graph would need a host round trip).
__global__ void cost_min_kernel(const float* __restrict__ cost,
                                float*       __restrict__ s_min);

// Stage 7 — planning: w_k = exp(-(cost_k - s_min)/lambda), plus their sum
// (single-block shared-memory SUM reduction).
__global__ void softmin_weight_kernel(const float* __restrict__ cost,
                                      const float* __restrict__ s_min,
                                      float*       __restrict__ weights,  // [kRollouts] OUT
                                      float*       __restrict__ w_sum);   // [1] OUT

// Stage 8 — control: for every horizon step t (one block per t), the
// weighted blend Σ_k w_k eps[t*K+k] / w_sum is reduced and folded into
// u_nom[t] in place, then clamped to ±kUmax.
__global__ void control_blend_kernel(float*       __restrict__ u_nom,     // [kHorizon] IN/OUT
                                     const float* __restrict__ eps,       // [kHorizon*kRollouts]
                                     const float* __restrict__ weights,   // [kRollouts]
                                     const float* __restrict__ w_sum);    // [1]

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launch wrappers — declared OUTSIDE the __CUDACC__ fence so any
// translation unit may call them (only their DEFINITIONS in kernels.cu need
// nvcc). Every wrapper takes an explicit cudaStream_t: main.cu uses the SAME
// stream for all three modes, so graph capture (which records whatever is
// enqueued on a stream between Begin/EndCapture) sees exactly these calls.
// ---------------------------------------------------------------------------
void launch_sensor_scale_bias(const float* d_x, float* d_y, cudaStream_t stream);
void launch_sensor_smooth(const float* d_x, float* d_y, cudaStream_t stream);
void launch_state_predict(const float* d_x_est_prev, float* d_x_pred, cudaStream_t stream);
void launch_state_correct(const float* d_smoothed, const float* d_x_pred,
                          float* d_x_est_new, cudaStream_t stream);
void launch_mppi_rollout(const float* d_x0, const float* d_u_nom, const float* d_eps,
                         float* d_cost, cudaStream_t stream);
void launch_cost_min(const float* d_cost, float* d_s_min, cudaStream_t stream);
void launch_softmin_weight(const float* d_cost, const float* d_s_min,
                           float* d_weights, float* d_w_sum, cudaStream_t stream);
void launch_control_blend(float* d_u_nom, const float* d_eps,
                          const float* d_weights, const float* d_w_sum, cudaStream_t stream);

// ---------------------------------------------------------------------------
// CPU reference (reference_cpu.cpp) — the oracle twin of the WHOLE tick, not
// just one kernel. main.cu's VERIFY stage runs this once, on tick 0's exact
// inputs, and requires agreement with the GPU path within a documented
// tolerance (the repo's §5 gate). See reference_cpu.cpp for why the whole
// pipeline is one function instead of eight, mirroring kernels.cu's stages.
//
//   sensor_raw   : [kSensorN] host pointer, raw counts (kernels.cuh layout)
//   eps          : [kHorizon*kRollouts] host pointer, TRANSPOSED layout
//   x_est_prev   : [kNX] host pointer, the state estimate entering this tick
//   u_nom_inout  : [kHorizon] host pointer, IN the previous plan, OUT the
//                  blended-and-clamped new plan (mirrors control_blend_kernel)
//   x_est_out    : [kNX] OUT, this tick's corrected state estimate
//   cost_out     : [kRollouts] OUT, every rollout's total cost (the
//                  fine-grained comparison main.cu's VERIFY stage uses)
// ---------------------------------------------------------------------------
void tick_pipeline_cpu(const float* sensor_raw, const float* eps,
                       const float* x_est_prev, float* u_nom_inout,
                       float* x_est_out, float* cost_out);

#endif // PROJECT_KERNELS_CUH
