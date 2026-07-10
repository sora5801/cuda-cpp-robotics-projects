// ===========================================================================
// kernels.cu — GPU implementation for project 10.03
//              Massively parallel robot sim (Isaac-Gym-style: one robot,
//              10,000 environments)
//
// The big idea
// ------------
// One thread = one ENVIRONMENT (not one rollout of a single plant, as in
// 08.01 — a whole independent copy of the robot, with its own randomized
// mass/length and its own episode clock). All N threads step in LOCKSTEP:
// every thread performs step t before any thread performs step t+1, which
// is what makes this an Isaac-Gym-style farm rather than N unrelated
// simulations that happen to share a GPU.
//
// What is NEW here beyond 08.01/09.01/33.01:
//   * STATE THAT PERSISTS ACROSS THE WHOLE RUN in GLOBAL memory (SoA
//     arrays), not a fresh x0 handed to a stateless kernel every call —
//     see kernels.cuh's layout comment for why that forces the SoA choice.
//   * THE ENTIRE T-STEP RUN IS ONE KERNEL LAUNCH. 08.01's MPPI loop needed
//     a host round-trip every 20 ms (the softmin blend and the "apply to
//     the real plant" step are host-side decisions the next tick depends
//     on). Nothing here depends on a host decision between ticks — the
//     controller, the fail/cap/reset logic, and next tick's input are all
//     fully determined by the PREVIOUS tick's on-device state — so the
//     per-thread loop over `steps` lives INSIDE the kernel. This is the
//     concrete, measurable payoff of a resident farm: main.cu times ONE
//     launch covering N*steps environment-ticks and reports an AGGREGATE
//     env-steps/second throughput, not a per-tick latency (contrast
//     08.01's per-20ms-tick GPU-ms figure).
//   * Per-environment RESET happens INLINE, not via a second kernel. A
//     literal "reset kernel" would need to know, from the HOST side,
//     which environments just failed — but with the whole run fused into
//     one launch, no host code runs between ticks to make that decision.
//     The correct (and, not coincidentally, higher-throughput) design is
//     what real GPU physics farms actually do: each thread checks its OWN
//     termination condition after its OWN step and resets itself in place
//     — no synchronization with any other thread is needed, because
//     environments never interact (the 08.01 lesson: independence is what
//     makes "one thread per unit of work" the right mapping at all).
//
// All model constants, the SoA layout, the controller, and the reset rule
// come from kernels.cuh — the single source shared with the CPU oracle;
// farm_init_one_env/farm_step_one_env are LITERALLY the same functions the
// CPU path calls (reference_cpu.cpp), not hand-copied twins (kernels.cuh's
// file header explains why this project departs from 08.01's duplication
// rule here).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (CLAUDE.md §6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ===========================================================================
// init_farm_kernel — one thread per environment, run ONCE per FarmBuffers
// instance before any step kernel: seeds the env's RNG stream, draws its
// domain-randomized mc/mp/l, zeros its metrics, and takes the first
// episode reset (draws the first initial angle).
//
// Thread-to-data mapping: thread i = blockIdx.x*blockDim.x + threadIdx.x
// owns environment i. Grid: ceil(N/256) x 256 (repo default; ragged tail
// guarded) — see launch_farm_init below for the reasoning.
//
// Memory behavior: every array in `buf` is written EXACTLY ONCE per
// thread here, at a coalesced address buf.<field>[i] (consecutive threads
// -> consecutive addresses, the SoA payoff described in kernels.cuh).
// No shared memory, no atomics: environments never share data.
// ===========================================================================
__global__ void init_farm_kernel(FarmBuffers buf, int N, uint32_t base_seed,
                                 float dr_mc, float dr_mp, float dr_l, float theta0_range)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's environment index
    if (i >= N) return;                                    // ragged-tail guard

    // Local registers for this env's fields; farm_init_one_env (kernels.cuh)
    // fills them, then we do ONE coalesced write of each field back to
    // global memory — cheaper than reading/writing global memory field by
    // field inside the shared function (which does not know it is on a GPU).
    float x, xdot, theta, thdot, mc, mp, l;
    int ep_step, steps_balanced, reset_count;
    uint32_t rng;

    farm_init_one_env(x, xdot, theta, thdot, mc, mp, l,
                      ep_step, rng, steps_balanced, reset_count,
                      env_seed(base_seed, i), dr_mc, dr_mp, dr_l, theta0_range);

    buf.x[i] = x;  buf.xdot[i] = xdot;  buf.theta[i] = theta;  buf.thdot[i] = thdot;
    buf.mass_cart[i] = mc;  buf.mass_pole[i] = mp;  buf.pole_half_len[i] = l;
    buf.rng_state[i] = rng;
    buf.ep_step[i] = ep_step;
    buf.steps_balanced[i] = steps_balanced;
    buf.reset_count[i] = reset_count;
}

// ===========================================================================
// step_farm_kernel — one thread per environment, `steps` ticks run
// INTERNALLY in a register-resident loop (see the file header for why the
// whole run is one launch). This is the project's hot loop and its only
// kernel that does real floating-point work.
//
// Memory behavior per thread:
//   READ ONCE at entry: buf.x[i], buf.xdot[i], buf.theta[i], buf.thdot[i],
//     buf.mass_cart[i], buf.mass_pole[i], buf.pole_half_len[i],
//     buf.rng_state[i], buf.ep_step[i], buf.steps_balanced[i],
//     buf.reset_count[i] — eleven coalesced 128-byte-per-warp transactions
//     (four state fields change every tick; mass/length are read-only for
//     the whole run — READ ONCE and keep in registers rather than
//     re-reading global memory 1000 times per environment).
//   REGISTERS for the whole `steps`-tick loop: no global memory traffic
//     AT ALL between the entry read and the exit write — this is the
//     entire performance story of fusing the loop into the kernel.
//   WRITE ONCE at exit: the same eleven fields, coalesced.
// No shared memory (environments share nothing), no atomics, no
// divergence beyond the tail guard: control flow inside the per-tick
// helper (the fail/cap/reset branch) DOES differ across threads once
// environments start failing/resetting at different times, but each
// branch is O(1) work (a handful of scalar ops), so the warp-divergence
// cost is small and bounded — very different from, say, an iterative
// solver where diverged threads could spin for wildly different lengths.
// ===========================================================================
__global__ void step_farm_kernel(FarmBuffers buf, int N, int steps,
                                 float Kx, float Kxd, float Kth, float Kthd,
                                 float theta0_range, int episode_cap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Entry: pull this environment's entire state into registers. mc/mp/l
    // are per-env CONSTANTS for this call (domain randomization happened
    // once, in init_farm_kernel) — reading them here, once, instead of
    // inside the loop is the difference between 1 global read and 1000.
    float x = buf.x[i], xdot = buf.xdot[i], theta = buf.theta[i], thdot = buf.thdot[i];
    const float mc = buf.mass_cart[i], mp = buf.mass_pole[i], l = buf.pole_half_len[i];
    uint32_t rng = buf.rng_state[i];
    int ep_step = buf.ep_step[i];
    int steps_balanced = buf.steps_balanced[i];
    int reset_count = buf.reset_count[i];

    // The whole run for THIS environment, entirely in registers: control,
    // integrate, classify, maybe reset — `steps` times, zero global
    // memory traffic in between. This loop is where essentially all of
    // this kernel's arithmetic happens (THEORY.md §GPU mapping counts it).
    for (int t = 0; t < steps; ++t) {
        farm_step_one_env(x, xdot, theta, thdot, mc, mp, l,
                          ep_step, rng, steps_balanced, reset_count,
                          Kx, Kxd, Kth, Kthd, theta0_range, episode_cap);
    }

    // Exit: one coalesced write per field, same layout as the entry read.
    buf.x[i] = x;  buf.xdot[i] = xdot;  buf.theta[i] = theta;  buf.thdot[i] = thdot;
    buf.rng_state[i] = rng;
    buf.ep_step[i] = ep_step;
    buf.steps_balanced[i] = steps_balanced;
    buf.reset_count[i] = reset_count;
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================

// launch_farm_init — see kernels.cuh for the contract.
void launch_farm_init(const FarmBuffers& buf, int N, uint32_t base_seed,
                      float dr_mc, float dr_mp, float dr_l, float theta0_range)
{
    if (N < 1) {
        std::fprintf(stderr, "launch_farm_init: invalid N=%d\n", N);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;                        // repo default: warp multiple, good occupancy
    const int blocks = (N + threads - 1) / threads;  // ceil(N/threads): cover every environment

    init_farm_kernel<<<blocks, threads>>>(buf, N, base_seed, dr_mc, dr_mp, dr_l, theta0_range);
    CUDA_CHECK_LAST_ERROR("init_farm_kernel launch");
}

// launch_farm_step — see kernels.cuh for the contract. Grid math is
// IDENTICAL to launch_farm_init (same N, same environment-per-thread
// mapping); `steps` only changes how much work each thread does internally,
// not the launch shape.
void launch_farm_step(const FarmBuffers& buf, int N, int steps,
                      float Kx, float Kxd, float Kth, float Kthd,
                      float theta0_range, int episode_cap)
{
    if (N < 1 || steps < 1) {
        std::fprintf(stderr, "launch_farm_step: invalid N=%d steps=%d\n", N, steps);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    step_farm_kernel<<<blocks, threads>>>(buf, N, steps, Kx, Kxd, Kth, Kthd,
                                          theta0_range, episode_cap);
    CUDA_CHECK_LAST_ERROR("step_farm_kernel launch");
}
