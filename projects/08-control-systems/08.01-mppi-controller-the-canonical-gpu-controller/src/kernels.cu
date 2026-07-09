// ===========================================================================
// kernels.cu — GPU implementation for project 08.01
//              MPPI controller — the canonical GPU controller
//              (teaching core: force-limited cart-pole swing-up)
//
// The big idea (CLAUDE.md §6.2 uses this very kernel as its style example)
// ------------------------------------------------------------------------
// MPPI steers by SAMPLING. Each GPU thread simulates ONE candidate control
// sequence ("rollout") of the system dynamics over the horizon, adds up its
// cost, and the host then blends all sequences with softmin weights. K
// rollouts are fully independent → one thread per rollout is the natural
// GPU mapping (K ~ thousands; a CPU manages dozens). The same
// thread-per-problem pattern as 33.01/09.01 — here the "problem" is a whole
// 1-second simulated future of the plant.
//
// What is NEW here beyond 33.01/09.01/07.09:
//   * a full ODE integrator (RK4) living inside the kernel loop;
//   * a COST FUNCTIONAL accumulated along a trajectory — the object
//     optimal control actually optimizes;
//   * the coalescing fix applied from the start: the noise array is stored
//     TRANSPOSED (eps[t*K + k]) so that at each time step a warp's 32
//     noise reads are consecutive floats — one 128-byte transaction —
//     instead of strides of T floats (the layout 33.01 taught the cost of);
//   * uniform reads (u_nom[t]: every thread, same address) served by L2/
//     read-only cache — the same-address-read spectrum runs 09.01's
//     __constant__ broadcast → this → 07.09's divergent global reads.
//
// All model constants and layouts come from kernels.cuh — the single
// source shared with the CPU oracle; the dynamics function below is a
// deliberate line-by-line twin of the one in reference_cpu.cpp.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// cartpole_deriv — the plant's equations of motion: xdot = f(x, u).
//
// Frictionless cart-pole in the standard form (θ = 0 upright; gravity is
// the destabilizing term — THEORY.md derives these from the Lagrangian):
//
//     tmp   = (u + m_p·l·θ̇²·sinθ) / (m_c + m_p)
//     θ̈     = (g·sinθ − cosθ·tmp) / ( l·(4/3 − m_p·cos²θ/(m_c+m_p)) )
//     p̈     = tmp − m_p·l·θ̈·cosθ / (m_c + m_p)
//
// Units: SI throughout (m, m/s, rad, rad/s; u in N). State layout is the
// kernels.cuh contract. __forceinline__ + fixed-size arrays keep everything
// in registers (the 33.01 lesson, applied).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cartpole_deriv(const float* x, float u, float* xdot)
{
    const float sin_th = sinf(x[2]);   // precise sinf/cosf, not the __sinf
    const float cos_th = cosf(x[2]);   // intrinsics — same reasoning as 09.01:
                                       // intrinsic error grows with |θ|, and
                                       // rollouts integrate UNWRAPPED angles
                                       // that pass ±π routinely during swing-up

    const float total_mass = kMassCart + kMassPole;
    const float ml = kMassPole * kPoleHalfLen;

    // tmp = acceleration the cart would have if the pole were a point mass
    // riding along (force + centrifugal term from the swinging pole).
    const float tmp = (u + ml * x[3] * x[3] * sin_th) / total_mass;

    // Pole angular acceleration: gravity torque vs the cart's reaction.
    const float th_acc = (kGravity * sin_th - cos_th * tmp)
        / (kPoleHalfLen * (4.0f / 3.0f - kMassPole * cos_th * cos_th / total_mass));

    // Cart acceleration: tmp minus the pole's back-reaction.
    const float p_acc = tmp - ml * th_acc * cos_th / total_mass;

    xdot[0] = x[1];      // ṗ
    xdot[1] = p_acc;     // p̈
    xdot[2] = x[3];      // θ̇
    xdot[3] = th_acc;    // θ̈
}

// ---------------------------------------------------------------------------
// rk4_step — classic 4th-order Runge–Kutta under ZERO-ORDER HOLD (u constant
// across the step, exactly how a 50 Hz controller drives a real actuator).
//
// Why RK4 and not Euler? The swing-up trajectory has fast pole dynamics near
// the bottom (θ̇ up to ~8 rad/s); at dt = 0.02 s Euler's O(dt²) local error
// visibly distorts the pendulum's energy, and an MPC "optimizing" a wrong
// model steers the wrong plant. RK4's O(dt⁵) local error makes the model
// trustworthy at this dt for 4 extra derivative evaluations — the classic
// robotics accuracy/cost trade (THEORY.md §numerics quantifies it).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void rk4_step(float* x, float u, float dt)
{
    float k1[kNX], k2[kNX], k3[kNX], k4[kNX], xt[kNX];   // all registers (fixed size, literal indices)

    cartpole_deriv(x, u, k1);
#pragma unroll
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(0.5f * dt, k1[i], x[i]);
    cartpole_deriv(xt, u, k2);
#pragma unroll
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(0.5f * dt, k2[i], x[i]);
    cartpole_deriv(xt, u, k3);
#pragma unroll
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(dt, k3[i], x[i]);
    cartpole_deriv(xt, u, k4);

    // x += dt/6 · (k1 + 2k2 + 2k3 + k4) — the standard RK4 blend.
#pragma unroll
    for (int i = 0; i < kNX; ++i)
        x[i] += dt * (1.0f / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
}

// ---------------------------------------------------------------------------
// stage_cost — what "good" means, evaluated at every step of every rollout.
//
// Weights from kernels.cuh (the tuning story is THEORY.md's):
//   angle: kWAngle·(1 − cosθ) — smooth, minimal at upright, and WRAP-FREE
//          (cos never cares that rollouts integrate θ past ±π; using θ²
//          here would punish a pole that swung the "long way" up, which is
//          exactly the kind of accidental prior that ruins swing-up);
//   damping/position/effort terms: standard quadratics.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float stage_cost(const float* x, float u)
{
    const float upright = 1.0f - cosf(x[2]);   // 0 at top, 2 at bottom
    return kWAngle * upright
         + kWThdot * x[3] * x[3]
         + kWPos   * x[0] * x[0]
         + kWPdot  * x[1] * x[1]
         + kWCtrl  * u * u;
}

// ===========================================================================
// The MPPI rollout kernel: one thread = one candidate future.
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x + threadIdx.x
// owns rollout k. Grid: ceil(K/256) × 256 (repo default; ragged tail
// guarded).
//
// Memory spaces per thread and per step t:
//   registers : the simulated state x[4] + RK4 scratch (~30 regs)
//   global    : u_nom[t]      — UNIFORM read (all threads, same address):
//                               served by the read-only/L2 cache path at
//                               broadcast-like cost after first touch;
//               eps[t*K + k]  — COALESCED read thanks to the transposed
//                               layout (warp reads 32 consecutive floats;
//                               the layout decision lives in kernels.cuh);
//               cost[k]       — one coalesced write at the very end.
// No shared memory, no atomics, no divergence beyond the tail guard: the
// rollouts never interact — by construction, and that is the whole point.
// ===========================================================================
__global__ void mppi_rollouts_kernel(const float* __restrict__ x0,     // [4] shared start state (current plant state)
                                     const float* __restrict__ u_nom,  // [T] nominal control sequence (N)
                                     const float* __restrict__ eps,    // [T*K] noise, TRANSPOSED: eps[t*K + k]
                                     float*       __restrict__ cost,   // [K] OUT: total rollout costs
                                     int K)                            // number of rollouts
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's rollout index
    if (k >= K) return;                                    // ragged-tail guard

    // Every rollout starts from the SAME state — the plant's now. Four
    // uniform reads, then registers for the whole simulated second.
    float x[kNX];
#pragma unroll
    for (int i = 0; i < kNX; ++i) x[i] = x0[i];

    float S = 0.0f;    // this rollout's accumulated cost (unitless, weighted)

    // March the horizon: at each step take the nominal control, add MY
    // noise, clamp to what the actuator can actually do, integrate, pay.
    for (int t = 0; t < kHorizon; ++t) {
        // Clamp BEFORE integrating — the rollout must experience the same
        // saturation the real actuator will impose, or MPPI optimizes with
        // forces the motor cannot deliver (a classic sim-to-real gap in
        // miniature; THEORY.md §algorithm).
        float u = u_nom[t] + eps[t * K + k];
        u = fminf(fmaxf(u, -kUmax), kUmax);

        rk4_step(x, u, kDt);
        S += stage_cost(x, u);
    }

    cost[k] = S;   // one coalesced write; the host's softmin blend consumes this
}

// ===========================================================================
// Host launcher (declared in kernels.cuh).
// ===========================================================================
void launch_mppi_rollouts(int K, const float* x0,
                          const float* d_u_nom, const float* d_eps,
                          float* d_cost)
{
    if (K < 1 || !x0 || !d_u_nom || !d_eps || !d_cost) {
        std::fprintf(stderr, "launch_mppi_rollouts: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }

    // The start state is 16 bytes — upload it fresh each call. A dedicated
    // device buffer per call keeps the API stateless; the alloc/free cost
    // is trivial next to K×T RK4 steps (and pooling is domain-32 business).
    float* d_x0 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x0, kNX * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x0, x0, kNX * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;                      // repo default geometry
    const int blocks = (K + threads - 1) / threads;
    mppi_rollouts_kernel<<<blocks, threads>>>(d_x0, d_u_nom, d_eps, d_cost, K);
    CUDA_CHECK_LAST_ERROR("mppi_rollouts_kernel launch");

    // Free after the launch: cudaFree synchronizes with the work using the
    // buffer, so this is safe — and it keeps the function self-contained.
    CUDA_CHECK(cudaFree(d_x0));
}
