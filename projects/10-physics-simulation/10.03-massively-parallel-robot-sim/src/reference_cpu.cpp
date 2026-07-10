// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 10.03
//                     Massively parallel robot sim (Isaac-Gym-style: one
//                     robot, 10,000 environments)
//
// Two jobs in this project (both declared in kernels.cuh):
//
//   1. farm_init_cpu / farm_step_cpu — the ORACLE twins of
//      init_farm_kernel / step_farm_kernel: same per-env logic
//      (farm_init_one_env / farm_step_one_env, LITERALLY shared via
//      kernels.cuh — see that file's header for why this project departs
//      from 08.01's hand-duplication rule), sequential over environments
//      instead of one-thread-per-environment. main.cu runs these against
//      the GPU on a 256-environment subset and requires agreement within
//      documented tolerances — the §5 GPU-vs-CPU gate.
//
//   2. energy_conservation_cpu — a SEPARATE diagnostic that has nothing to
//      do with the farm: it integrates ONE undriven (u=0), unbounded (no
//      fail box, no reset) cart-pole and reports how much the RK4
//      integrator's total-energy estimate drifts over 1000 steps. This
//      checks the INTEGRATOR itself, independent of the controller or the
//      randomized farm — an "analytic-style gate" a single CPU trajectory
//      is enough to exercise (no GPU parallelism needed or used).
//
// Because farm_init_one_env/farm_step_one_env/cartpole_energy are defined
// ONCE in kernels.cuh (as HD inline functions that compile to plain host
// functions here, since __CUDACC__ is not defined when cl.exe processes
// this file), this file contains almost no physics of its own — it is
// mostly the ORCHESTRATION around those shared functions: looping over
// environments, and, for the energy experiment, looping over time and
// recording a trace. Rules for this file otherwise unchanged from 08.01:
// plain C++17, no CUDA headers (kernels.cuh's HD macro guarantees this —
// __host__/__device__ never appear when __CUDACC__ is undefined), no
// hand-vectorization, no cleverness. If the reference is clever, it can be
// wrong, and then the oracle lies.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared model, controller, reset logic, and RNG — the CONTRACT

// ---------------------------------------------------------------------------
// farm_init_cpu — sequential twin of init_farm_kernel: for each
// environment i, seed its stream and call farm_init_one_env exactly as
// the GPU kernel does (same seed derivation via env_seed, same shared
// function) — the only difference is "for each i" instead of "thread i".
// ---------------------------------------------------------------------------
void farm_init_cpu(const FarmBuffers& buf, int N, uint32_t base_seed,
                   float dr_mc, float dr_mp, float dr_l, float theta0_range)
{
    for (int i = 0; i < N; ++i) {
        float x, xdot, theta, thdot, mc, mp, l;
        int ep_step, steps_balanced, reset_count;
        uint32_t rng;

        farm_init_one_env(x, xdot, theta, thdot, mc, mp, l,
                          ep_step, rng, steps_balanced, reset_count,
                          env_seed(base_seed, i), dr_mc, dr_mp, dr_l, theta0_range);

        buf.x[i] = x; buf.xdot[i] = xdot; buf.theta[i] = theta; buf.thdot[i] = thdot;
        buf.mass_cart[i] = mc; buf.mass_pole[i] = mp; buf.pole_half_len[i] = l;
        buf.rng_state[i] = rng;
        buf.ep_step[i] = ep_step;
        buf.steps_balanced[i] = steps_balanced;
        buf.reset_count[i] = reset_count;
    }
}

// ---------------------------------------------------------------------------
// farm_step_cpu — sequential twin of step_farm_kernel: for each
// environment i, run `steps` ticks of farm_step_one_env back to back —
// the GPU gives each environment its own thread; the CPU gives each
// environment its own turn. Same shared function, same per-tick logic.
// ---------------------------------------------------------------------------
void farm_step_cpu(const FarmBuffers& buf, int N, int steps,
                   float Kx, float Kxd, float Kth, float Kthd,
                   float theta0_range, int episode_cap)
{
    for (int i = 0; i < N; ++i) {
        float x = buf.x[i], xdot = buf.xdot[i], theta = buf.theta[i], thdot = buf.thdot[i];
        const float mc = buf.mass_cart[i], mp = buf.mass_pole[i], l = buf.pole_half_len[i];
        uint32_t rng = buf.rng_state[i];
        int ep_step = buf.ep_step[i];
        int steps_balanced = buf.steps_balanced[i];
        int reset_count = buf.reset_count[i];

        for (int t = 0; t < steps; ++t) {
            farm_step_one_env(x, xdot, theta, thdot, mc, mp, l,
                              ep_step, rng, steps_balanced, reset_count,
                              Kx, Kxd, Kth, Kthd, theta0_range, episode_cap);
        }

        buf.x[i] = x; buf.xdot[i] = xdot; buf.theta[i] = theta; buf.thdot[i] = thdot;
        buf.rng_state[i] = rng;
        buf.ep_step[i] = ep_step;
        buf.steps_balanced[i] = steps_balanced;
        buf.reset_count[i] = reset_count;
    }
}

// ---------------------------------------------------------------------------
// energy_conservation_cpu — integrate ONE undriven (u=0), UNBOUNDED cart-
// pole for `steps` RK4 ticks from initial angle theta0 (all other state
// zero), recording total mechanical energy at every sample INCLUDING t=0
// (energy_out must have steps+1 slots) via the shared cartpole_energy()
// formula. No fail box, no reset: this experiment exists to see the
// integrator's own error, not the farm's episode logic, so nothing here
// may reset or clamp the trajectory (THEORY.md §how we verify explains
// why a "free" trajectory is the honest way to expose RK4 drift).
//
// final_state_out, if non-null, receives [p, pdot, theta, thetadot] after
// the last step — main.cu logs it as an [info] line for a curious reader,
// it is not part of any pass/fail gate.
// ---------------------------------------------------------------------------
void energy_conservation_cpu(int steps, float theta0,
                             float mc, float mp, float l,
                             float* energy_out, float* final_state_out)
{
    float x[kNX] = { 0.0f, 0.0f, theta0, 0.0f };

    energy_out[0] = cartpole_energy(x[0], x[1], x[2], x[3], mc, mp, l);
    for (int t = 0; t < steps; ++t) {
        rk4_step(x, /*u=*/0.0f, mc, mp, l, kDt);   // UNDRIVEN: u is fixed at zero, never clamped
                                                    // (kUmax would never bind here anyway)
        energy_out[t + 1] = cartpole_energy(x[0], x[1], x[2], x[3], mc, mp, l);
    }

    if (final_state_out) {
        final_state_out[0] = x[0]; final_state_out[1] = x[1];
        final_state_out[2] = x[2]; final_state_out[3] = x[3];
    }
}
