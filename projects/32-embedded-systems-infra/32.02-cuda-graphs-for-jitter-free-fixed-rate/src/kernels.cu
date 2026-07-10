// ===========================================================================
// kernels.cu — GPU implementation for project 32.02
//              CUDA Graphs for jitter-free fixed-rate perception-control loops
//
// Eight small kernels, one "tick" of a fixed-rate perception-control loop.
// Read kernels.cuh first (the contract: sizes, layouts, the 08.01-borrowed
// plant). Read this file top to bottom — the stages run in exactly this
// order, each depending on the previous, which is also the order main.cu
// enqueues them in (and therefore the order CUDA Graph capture records
// them in: stream order becomes graph dependency order, see main.cu).
//
// Why eight kernels this small, on purpose
// -----------------------------------------
// Every kernel below could be one thread's worth of work in a bigger kernel.
// Keeping them separate is DELIBERATE: a real perception-control tick is
// built from independently-authored, independently-tested stages (a
// perception team owns stages 1-2, an estimation team owns 3-4, planning
// owns 5-7, controls owns 8) that get composed, not fused, in production —
// and each cudaLaunchKernel() call the fusion avoids is exactly the
// per-call host/driver overhead this project measures (THEORY.md "what a
// kernel launch actually costs"). Fusing everything into one kernel would
// make the launch-overhead STORY disappear along with the launches.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (a
// line-by-line twin of every stage below, in plain sequential C++).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ===========================================================================
// STAGE 1 — sensor_scale_bias_kernel: perception, a pure MAP.
//
// y[i] = kSensorGain*x[i] + kSensorBias for i in [0, kSensorN). Converts
// synthetic raw sensor counts into the physical-ish units the rest of the
// pipeline expects (README §Limitations: the mapping is intentionally
// simplified — see kernels.cuh's comment on kSensorGain/kSensorBias).
//
// Launch: one block, kSensorN threads (512) — kSensorN fits in one block
// (<=1024 on sm_75+), so a naive one-thread-per-element map needs no grid-
// stride loop; every thread's global id IS its element index.
// Memory: one coalesced read of x[i], one coalesced write of y[i] per
// thread — textbook bandwidth-bound map, no shared memory, no divergence.
// ===========================================================================
__global__ void sensor_scale_bias_kernel(const float* __restrict__ x,
                                         float*       __restrict__ y)
{
    const int i = threadIdx.x;                 // one block, one thread per sensor channel
    y[i] = kSensorGain * x[i] + kSensorBias;
}

void launch_sensor_scale_bias(const float* d_x, float* d_y, cudaStream_t stream)
{
    sensor_scale_bias_kernel<<<1, kSensorN, 0, stream>>>(d_x, d_y);
    CUDA_CHECK_LAST_ERROR("sensor_scale_bias_kernel launch");
}

// ===========================================================================
// STAGE 2 — sensor_smooth_kernel: perception, a STENCIL.
//
// y[i] = (x[i-1] + x[i] + x[i+1]) / 3, boundary-clamped (i-1 and i+1 read
// the nearest valid index at the array's ends instead of going out of
// bounds — the standard "clamp to edge" stencil boundary, cheaper than a
// branch-free ghost cell for an array this small). A 3-tap moving-average
// denoise — the smallest stencil that is still genuinely a stencil (reads
// NEIGHBORS, not just its own index, unlike stage 1's map).
//
// Launch: same geometry as stage 1 (one block, kSensorN threads). Memory:
// each thread reads three elements; adjacent threads' reads overlap by two
// of three, so global memory traffic is still ~1x (not 3x) once the L1/L2
// cache lines both neighbors already pulled in for stage 1's read are
// reused — no shared-memory tiling needed at this size (a bigger stencil
// project, e.g. image convolution in domain 01, DOES tile; see THEORY.md
// §GPU mapping for when the crossover happens).
// ===========================================================================
__global__ void sensor_smooth_kernel(const float* __restrict__ x,
                                     float*       __restrict__ y)
{
    const int i = threadIdx.x;
    const int lo = i > 0 ? i - 1 : 0;                    // clamp-to-edge left neighbor
    const int hi = i < kSensorN - 1 ? i + 1 : kSensorN - 1;  // clamp-to-edge right neighbor
    y[i] = (x[lo] + x[i] + x[hi]) * (1.0f / 3.0f);
}

void launch_sensor_smooth(const float* d_x, float* d_y, cudaStream_t stream)
{
    sensor_smooth_kernel<<<1, kSensorN, 0, stream>>>(d_x, d_y);
    CUDA_CHECK_LAST_ERROR("sensor_smooth_kernel launch");
}

// ===========================================================================
// STAGE 3 — state_predict_kernel: estimation, constant-velocity PREDICT.
//
// The cheapest honest model of "where will we be next tick": integrate
// position/angle forward by the LAST known velocity/angular-velocity,
// leave the velocities themselves unchanged (no acceleration is known at
// predict time — that is what stage 4's measurement is for). This is
// DELIBERATELY a cruder model than the rollout kernel's full nonlinear RK4
// cart-pole dynamics (stage 5) — a real estimator's predict step is
// usually a cheap linear(ized) model, while the PLANNER is where the full
// nonlinear model earns its keep. THEORY.md §The math names this contrast
// explicitly; it is a genuine, common production trade-off, not a shortcut
// unique to this demo.
//
// Launch: one block, kNX threads (4) — the smallest kernel in the tick, on
// purpose: it exists to show that not every stage in a real pipeline is
// big, and every one of them still pays the SAME per-launch overhead this
// project measures (README "the fat kernel is the exception, not the rule").
// ===========================================================================
__global__ void state_predict_kernel(const float* __restrict__ x_est_prev,
                                     float*       __restrict__ x_pred)
{
    const int i = threadIdx.x;   // one of the 4 state channels (kernels.cuh layout)
    if (i == 0)      x_pred[0] = x_est_prev[0] + x_est_prev[1] * kDt;  // p += pdot*dt
    else if (i == 1) x_pred[1] = x_est_prev[1];                        // pdot unchanged
    else if (i == 2) x_pred[2] = x_est_prev[2] + x_est_prev[3] * kDt;  // theta += thdot*dt
    else             x_pred[3] = x_est_prev[3];                        // thdot unchanged
}

void launch_state_predict(const float* d_x_est_prev, float* d_x_pred, cudaStream_t stream)
{
    state_predict_kernel<<<1, kNX, 0, stream>>>(d_x_est_prev, d_x_pred);
    CUDA_CHECK_LAST_ERROR("state_predict_kernel launch");
}

// ===========================================================================
// STAGE 4 — state_correct_kernel: estimation, REDUCTION + fixed-gain FUSE.
//
// Two jobs in one kernel, in sequence, synchronized by __syncthreads():
//   (a) mean of the smoothed sensor array — a classic shared-memory tree
//       reduction, kSensorN=512 threads -> 1 value in log2(512)=9 steps;
//   (b) thread 0 folds that mean (scaled into a synthetic "measurement",
//       kernels.cuh's kMeasScale/kMeasOffset) into the predicted state's
//       POSITION channel only, via a FIXED complementary-filter gain
//       (kCorrGain). A real EKF/UKF (domain 04) instead computes a
//       time-varying Kalman gain from propagated covariance — this project
//       intentionally uses the cheap fixed-gain version because the
//       covariance math would add real complexity for zero measurement
//       value here (THEORY.md §numerics says so explicitly; the point of
//       this kernel is to BE a shared-memory reduction in the tick's DAG,
//       not to be a good estimator).
//
// Launch: one block, kSensorN threads (512) — the same thread count as
// stages 1-2, so the same block can immediately reuse the smoothed array
// stage 2 just wrote without any reshaping.
// Memory: SHARED memory for the reduction tree (2 KiB for 512 floats) —
// the first use of shared memory in this tick; global memory is read once
// per thread (coalesced) and the fused state is written by thread 0 only.
// ===========================================================================
__global__ void state_correct_kernel(const float* __restrict__ smoothed,
                                     const float* __restrict__ x_pred,
                                     float*       __restrict__ x_est_new)
{
    __shared__ float partial[kSensorN];   // one slot per thread; reduced in place below

    const int tid = threadIdx.x;
    partial[tid] = smoothed[tid];
    __syncthreads();   // every thread's read must land before anyone reduces

    // Standard power-of-two tree reduction: each round halves the live
    // thread count and folds the upper half onto the lower half. kSensorN
    // is 512 = 2^9, so this terminates in exactly 9 rounds with no ragged
    // tail to guard — one of the reasons kSensorN was chosen as a power of
    // two (THEORY.md §GPU mapping elaborates).
    for (int stride = kSensorN / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();   // every round's writes must be visible before the next round reads
    }

    if (tid == 0) {
        const float mean = partial[0] / static_cast<float>(kSensorN);
        const float measured_p = kMeasScale * mean + kMeasOffset;   // synthetic position proxy
        // Fixed-gain complementary fusion: blend the prediction toward the
        // measurement by kCorrGain (0 = ignore the sensor, 1 = trust it
        // completely). Velocities/angle pass through untouched — this
        // simplified sensor never measures them (kernels.cuh honesty note).
        x_est_new[0] = x_pred[0] + kCorrGain * (measured_p - x_pred[0]);
        x_est_new[1] = x_pred[1];
        x_est_new[2] = x_pred[2];
        x_est_new[3] = x_pred[3];
    }
}

void launch_state_correct(const float* d_smoothed, const float* d_x_pred,
                          float* d_x_est_new, cudaStream_t stream)
{
    state_correct_kernel<<<1, kSensorN, 0, stream>>>(d_smoothed, d_x_pred, d_x_est_new);
    CUDA_CHECK_LAST_ERROR("state_correct_kernel launch");
}

// ===========================================================================
// STAGE 5 — mppi_rollout_kernel: planning, THE FAT KERNEL.
//
// This is 08.01's mppi_rollouts_kernel, verbatim in its physics, at a
// smaller K/T (kernels.cuh: K=512, T=16 vs 08.01's K=4096, T=50). It is
// reused rather than re-derived on purpose — see README "System context":
// this project's whole point is to wrap a REPRESENTATIVE MPPI-shaped tick
// in CUDA Graphs, and 08.01 IS the repo's canonical MPPI-shaped workload.
// Full physics commentary (the Lagrangian derivation, why RK4, why the
// cost is wrap-free) lives in 08.01's kernels.cu/THEORY.md and is not
// repeated here; what follows are the parts specific to being ONE STAGE
// inside a bigger captured tick rather than the whole demo.
//
// Thread-to-data mapping: thread k = threadIdx.x owns rollout k (one
// block, kRollouts=512 threads — fits in one block, unlike 08.01's
// K=4096 which needs a multi-block grid; no ragged-tail guard needed).
// Memory: x0/registers exactly as 08.01; eps read via the TRANSPOSED
// layout (kernels.cuh) for coalescing; cost[k] one coalesced write.
// ===========================================================================
__device__ __forceinline__ void tick_cartpole_deriv(const float* x, float u, float* xdot)
{
    const float sin_th = sinf(x[2]);   // precise sinf/cosf — same reasoning as 08.01: the
    const float cos_th = cosf(x[2]);   // rollouts integrate unwrapped angles across many steps

    const float total_mass = kMassCart + kMassPole;
    const float ml = kMassPole * kPoleHalfLen;

    const float tmp = (u + ml * x[3] * x[3] * sin_th) / total_mass;
    const float th_acc = (kGravity * sin_th - cos_th * tmp)
        / (kPoleHalfLen * (4.0f / 3.0f - kMassPole * cos_th * cos_th / total_mass));
    const float p_acc = tmp - ml * th_acc * cos_th / total_mass;

    xdot[0] = x[1];
    xdot[1] = p_acc;
    xdot[2] = x[3];
    xdot[3] = th_acc;
}

__device__ __forceinline__ void tick_rk4_step(float* x, float u, float dt)
{
    float k1[kNX], k2[kNX], k3[kNX], k4[kNX], xt[kNX];

    tick_cartpole_deriv(x, u, k1);
#pragma unroll
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(0.5f * dt, k1[i], x[i]);
    tick_cartpole_deriv(xt, u, k2);
#pragma unroll
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(0.5f * dt, k2[i], x[i]);
    tick_cartpole_deriv(xt, u, k3);
#pragma unroll
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(dt, k3[i], x[i]);
    tick_cartpole_deriv(xt, u, k4);

#pragma unroll
    for (int i = 0; i < kNX; ++i)
        x[i] += dt * (1.0f / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
}

__device__ __forceinline__ float tick_stage_cost(const float* x, float u)
{
    const float upright = 1.0f - cosf(x[2]);
    return kWAngle * upright
         + kWThdot * x[3] * x[3]
         + kWPos   * x[0] * x[0]
         + kWPdot  * x[1] * x[1]
         + kWCtrl  * u * u;
}

__global__ void mppi_rollout_kernel(const float* __restrict__ x0,
                                    const float* __restrict__ u_nom,
                                    const float* __restrict__ eps,
                                    float*       __restrict__ cost)
{
    const int k = threadIdx.x;   // this thread's rollout index (one block covers all K=512)

    float x[kNX];
#pragma unroll
    for (int i = 0; i < kNX; ++i) x[i] = x0[i];   // every rollout starts from the SAME estimate

    float S = 0.0f;
    for (int t = 0; t < kHorizon; ++t) {
        float u = u_nom[t] + eps[t * kRollouts + k];   // transposed layout: coalesced across k
        u = fminf(fmaxf(u, -kUmax), kUmax);            // clamp BEFORE integrating (08.01's reasoning)
        tick_rk4_step(x, u, kDt);
        S += tick_stage_cost(x, u);
    }
    cost[k] = S;
}

void launch_mppi_rollout(const float* d_x0, const float* d_u_nom, const float* d_eps,
                         float* d_cost, cudaStream_t stream)
{
    mppi_rollout_kernel<<<1, kRollouts, 0, stream>>>(d_x0, d_u_nom, d_eps, d_cost);
    CUDA_CHECK_LAST_ERROR("mppi_rollout_kernel launch");
}

// ===========================================================================
// STAGE 6 — cost_min_kernel: planning, MIN reduction.
//
// 08.01 subtracts S_min from every cost before exp() on the HOST (the
// classic softmin overflow guard: exp(0) instead of exp(-10^4)). This
// project cannot do that — a host round trip in the middle of the tick
// would break graph capture (a captured graph cannot pause for the CPU to
// read a value back and branch on it; see THEORY.md §GPU mapping). So the
// min-reduction moves onto the GPU as its own tiny kernel, feeding stage 7.
//
// Launch/memory: identical shared-memory tree-reduction shape to stage 4's
// mean, just MIN instead of SUM (fminf, associative-enough for a reduction
// tree — see THEORY.md §numerics on why min needs no compensated-sum care
// unlike the weighted sums in stages 7-8).
// ===========================================================================
__global__ void cost_min_kernel(const float* __restrict__ cost,
                                float*       __restrict__ s_min)
{
    __shared__ float partial[kRollouts];

    const int tid = threadIdx.x;
    partial[tid] = cost[tid];
    __syncthreads();

    for (int stride = kRollouts / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fminf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) s_min[0] = partial[0];
}

void launch_cost_min(const float* d_cost, float* d_s_min, cudaStream_t stream)
{
    cost_min_kernel<<<1, kRollouts, 0, stream>>>(d_cost, d_s_min);
    CUDA_CHECK_LAST_ERROR("cost_min_kernel launch");
}

// ===========================================================================
// STAGE 7 — softmin_weight_kernel: planning, elementwise exp + SUM reduction.
//
// w_k = exp(-(cost_k - s_min)/lambda) for every rollout, plus their sum
// (needed to normalize the blend in stage 8). Same shared-memory tree
// pattern as stage 6, but every thread ALSO writes its own w_k to global
// memory first — stage 8 needs the full per-rollout weight array, not just
// its sum, so this kernel produces BOTH outputs from one pass over the data.
// ===========================================================================
__global__ void softmin_weight_kernel(const float* __restrict__ cost,
                                      const float* __restrict__ s_min,
                                      float*       __restrict__ weights,
                                      float*       __restrict__ w_sum)
{
    __shared__ float partial[kRollouts];

    const int tid = threadIdx.x;
    const float w = expf(-(cost[tid] - s_min[0]) / kLambda);   // s_min subtracted: exp(~0), never exp(-huge)
    weights[tid] = w;          // stage 8 reads this back per-rollout
    partial[tid] = w;          // ...and this copy feeds the sum reduction below
    __syncthreads();

    for (int stride = kRollouts / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0) w_sum[0] = partial[0];
}

void launch_softmin_weight(const float* d_cost, const float* d_s_min,
                           float* d_weights, float* d_w_sum, cudaStream_t stream)
{
    softmin_weight_kernel<<<1, kRollouts, 0, stream>>>(d_cost, d_s_min, d_weights, d_w_sum);
    CUDA_CHECK_LAST_ERROR("softmin_weight_kernel launch");
}

// ===========================================================================
// STAGE 8 — control_blend_kernel: control, T independent weighted reductions.
//
// For every horizon step t, fold the importance-weighted noise back into
// the nominal plan: u_nom[t] += Σ_k w_k*eps[t*K+k] / w_sum, then clamp to
// ±kUmax — 08.01's host-side blend loop (its main.cu step 4), moved onto
// the GPU because, again, a captured graph cannot round-trip to the host
// mid-tick.
//
// Thread-to-data mapping: ONE BLOCK PER HORIZON STEP (gridDim.x =
// kHorizon = 16), kRollouts (512) threads per block. blockIdx.x = t
// (which timestep this block owns), threadIdx.x = k (which rollout this
// thread's term comes from). Each block does its own independent
// shared-memory sum-reduction over eps[t*K + k]*w_k, exactly like stage
// 7's sum but T=16 of them in parallel instead of one — the natural GPU
// mapping once you notice the T blend equations share no data.
// ===========================================================================
__global__ void control_blend_kernel(float*       __restrict__ u_nom,
                                     const float* __restrict__ eps,
                                     const float* __restrict__ weights,
                                     const float* __restrict__ w_sum)
{
    __shared__ float partial[kRollouts];

    const int t   = blockIdx.x;    // this block's horizon step
    const int k   = threadIdx.x;   // this thread's rollout index
    partial[k] = weights[k] * eps[t * kRollouts + k];
    __syncthreads();

    for (int stride = kRollouts / 2; stride > 0; stride >>= 1) {
        if (k < stride) partial[k] += partial[k + stride];
        __syncthreads();
    }

    if (k == 0) {
        float u = u_nom[t] + partial[0] / w_sum[0];
        u_nom[t] = fminf(fmaxf(u, -kUmax), kUmax);   // same clamp discipline as the rollout kernel
    }
}

void launch_control_blend(float* d_u_nom, const float* d_eps,
                          const float* d_weights, const float* d_w_sum, cudaStream_t stream)
{
    control_blend_kernel<<<kHorizon, kRollouts, 0, stream>>>(d_u_nom, d_eps, d_weights, d_w_sum);
    CUDA_CHECK_LAST_ERROR("control_blend_kernel launch");
}
