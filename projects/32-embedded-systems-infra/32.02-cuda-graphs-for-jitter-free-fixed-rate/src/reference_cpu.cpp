// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 32.02
//                     (CUDA Graphs for jitter-free fixed-rate perception-
//                     control loops)
//
// One job: tick_pipeline_cpu() is the ORACLE TWIN of the whole eight-stage
// GPU tick (kernels.cu), not of one kernel — main.cu's VERIFY stage runs it
// once, on tick 0's exact inputs, and requires agreement with the GPU path
// (any of the three modes; they are required to agree with EACH OTHER
// exactly, so checking mode A here is checking all three) within a
// documented relative tolerance. See THEORY.md §How we verify correctness
// for the tolerance's derivation.
//
// This file is compiled by the HOST compiler (cl.exe), never nvcc — the
// __CUDACC__ fence in kernels.cuh hides every __global__ declaration from
// it, so only the plain data (constants, structs) and this function's own
// prototype are visible. Every stage below is a sequential, single-threaded
// twin of the identically-numbered stage in kernels.cu; read them side by
// side, not in isolation — the ONLY intended difference is "loop over K/T"
// vs "thread k, T-step loop," never the arithmetic itself.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>     // std::sin/cos/exp/fabs — host trig/transcendental twins of sinf/cosf/expf
#include <vector>

// ---------------------------------------------------------------------------
// Stage 1+2 twin — sensor_scale_bias + sensor_smooth, fused into one pass
// for brevity (the GPU keeps them as two kernels for the reasons kernels.cu
// explains; the CPU oracle only needs to compute the SAME numbers, not
// reproduce the kernel boundary).
// ---------------------------------------------------------------------------
static void sensor_preprocess_cpu(const float* raw, float* smoothed)
{
    std::vector<float> scaled(kSensorN);
    for (int i = 0; i < kSensorN; ++i)
        scaled[i] = kSensorGain * raw[i] + kSensorBias;

    for (int i = 0; i < kSensorN; ++i) {
        const int lo = i > 0 ? i - 1 : 0;
        const int hi = i < kSensorN - 1 ? i + 1 : kSensorN - 1;
        smoothed[i] = (scaled[lo] + scaled[i] + scaled[hi]) * (1.0f / 3.0f);
    }
}

// ---------------------------------------------------------------------------
// Stage 3+4 twin — state_predict + state_correct.
// ---------------------------------------------------------------------------
static void state_estimate_cpu(const float* smoothed, const float* x_est_prev,
                               float* x_est_new)
{
    // Stage 3: constant-velocity predict (line-by-line twin of state_predict_kernel).
    float x_pred[kNX];
    x_pred[0] = x_est_prev[0] + x_est_prev[1] * kDt;
    x_pred[1] = x_est_prev[1];
    x_pred[2] = x_est_prev[2] + x_est_prev[3] * kDt;
    x_pred[3] = x_est_prev[3];

    // Stage 4: mean of the smoothed array (a plain sequential sum here —
    // the GPU's shared-memory TREE reduction sums the exact same 512
    // values in a different ORDER, which is why main.cu compares with a
    // tolerance rather than demanding bit-equality against this file;
    // THEORY.md §numerics names float-reduction non-associativity as the
    // reason), then the same fixed-gain fuse as state_correct_kernel.
    double sum = 0.0;   // double accumulator: the honest way to sum 512 floats
                        // without the CPU path's own rounding pattern becoming
                        // a SECOND source of GPU-CPU divergence on top of the
                        // reduction-order one already in play (CLAUDE.md §12).
    for (int i = 0; i < kSensorN; ++i) sum += smoothed[i];
    const float mean = static_cast<float>(sum / kSensorN);
    const float measured_p = kMeasScale * mean + kMeasOffset;

    x_est_new[0] = x_pred[0] + kCorrGain * (measured_p - x_pred[0]);
    x_est_new[1] = x_pred[1];
    x_est_new[2] = x_pred[2];
    x_est_new[3] = x_pred[3];
}

// ---------------------------------------------------------------------------
// Stage 5 twin — the cart-pole rollout batch, 08.01's dynamics (see
// 08.01/src/reference_cpu.cpp for the identical derivation; duplicated here
// per CLAUDE.md §4's deliberate-duplication rule rather than reached across
// project folders).
// ---------------------------------------------------------------------------
static void cartpole_deriv_cpu(const float* x, float u, float* xdot)
{
    const float sin_th = std::sin(x[2]);
    const float cos_th = std::cos(x[2]);

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

static void rk4_step_cpu(float* x, float u, float dt)
{
    float k1[kNX], k2[kNX], k3[kNX], k4[kNX], xt[kNX];

    cartpole_deriv_cpu(x, u, k1);
    for (int i = 0; i < kNX; ++i) xt[i] = x[i] + 0.5f * dt * k1[i];
    cartpole_deriv_cpu(xt, u, k2);
    for (int i = 0; i < kNX; ++i) xt[i] = x[i] + 0.5f * dt * k2[i];
    cartpole_deriv_cpu(xt, u, k3);
    for (int i = 0; i < kNX; ++i) xt[i] = x[i] + dt * k3[i];
    cartpole_deriv_cpu(xt, u, k4);

    for (int i = 0; i < kNX; ++i)
        x[i] += dt * (1.0f / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
}

static float stage_cost_cpu(const float* x, float u)
{
    const float upright = 1.0f - std::cos(x[2]);
    return kWAngle * upright
         + kWThdot * x[3] * x[3]
         + kWPos   * x[0] * x[0]
         + kWPdot  * x[1] * x[1]
         + kWCtrl  * u * u;
}

static void mppi_rollouts_cpu(const float* x0, const float* u_nom, const float* eps,
                              float* cost)
{
    for (int k = 0; k < kRollouts; ++k) {
        float x[kNX];
        for (int i = 0; i < kNX; ++i) x[i] = x0[i];

        float S = 0.0f;
        for (int t = 0; t < kHorizon; ++t) {
            float u = u_nom[t] + eps[t * kRollouts + k];
            u = u < -kUmax ? -kUmax : (u > kUmax ? kUmax : u);
            rk4_step_cpu(x, u, kDt);
            S += stage_cost_cpu(x, u);
        }
        cost[k] = S;
    }
}

// ---------------------------------------------------------------------------
// Stage 6+7+8 twin — softmin weights (with the same S_min subtraction the
// GPU's cost_min_kernel computes) and the T weighted blends.
// ---------------------------------------------------------------------------
static void softmin_blend_cpu(const float* cost, const float* eps, float* u_nom_inout)
{
    // Stage 6: S_min.
    float s_min = cost[0];
    for (int k = 1; k < kRollouts; ++k) if (cost[k] < s_min) s_min = cost[k];

    // Stage 7: weights + their sum. double accumulator for w_sum — the
    // same reasoning as 08.01's host blend: kRollouts=512 tiny weights
    // must not lose the small ones to float rounding before they are used.
    std::vector<float> w(kRollouts);
    double w_sum = 0.0;
    for (int k = 0; k < kRollouts; ++k) {
        w[k] = std::exp(-(cost[k] - s_min) / kLambda);
        w_sum += w[k];
    }

    // Stage 8: T independent weighted blends, same clamp as control_blend_kernel.
    for (int t = 0; t < kHorizon; ++t) {
        double acc = 0.0;
        const float* eps_t = &eps[static_cast<size_t>(t) * kRollouts];
        for (int k = 0; k < kRollouts; ++k) acc += static_cast<double>(w[k]) * eps_t[k];
        float u = u_nom_inout[t] + static_cast<float>(acc / w_sum);
        u_nom_inout[t] = u < -kUmax ? -kUmax : (u > kUmax ? kUmax : u);
    }
}

// ---------------------------------------------------------------------------
// tick_pipeline_cpu — the whole tick, in the same eight-stage order as
// kernels.cu, declared in kernels.cuh.
// ---------------------------------------------------------------------------
void tick_pipeline_cpu(const float* sensor_raw, const float* eps,
                       const float* x_est_prev, float* u_nom_inout,
                       float* x_est_out, float* cost_out)
{
    std::vector<float> smoothed(kSensorN);
    sensor_preprocess_cpu(sensor_raw, smoothed.data());          // stages 1-2
    state_estimate_cpu(smoothed.data(), x_est_prev, x_est_out);  // stages 3-4
    mppi_rollouts_cpu(x_est_out, u_nom_inout, eps, cost_out);    // stage 5
    softmin_blend_cpu(cost_out, eps, u_nom_inout);               // stages 6-8
    // (The two device-to-device "publish" copies in kernels.cu have no CPU
    // twin: they move data, they do not compute anything — nothing here to
    // verify beyond "the bytes arrived," which main.cu's cross-mode array
    // comparison already covers.)
}
