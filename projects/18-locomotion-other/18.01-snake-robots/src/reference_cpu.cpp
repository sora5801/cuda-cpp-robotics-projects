// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 18.01
//                     (Snake robots: serpenoid gait sweeps, anisotropic
//                     friction)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), same as every project here:
//   1) It is the CORRECTNESS ORACLE for the §5 VERIFY gate: main.cu spot-
//      checks kVerifyCount of the GPU sweep's own results by recomputing
//      them from scratch, sequentially, on the CPU, and requiring agreement
//      within a documented tolerance.
//   2) It is the vehicle for every DIAGNOSTIC run that is not part of the
//      GPU sweep itself — the zero-amplitude, isotropic-friction, and
//      turning-bias verification GATES, plus the best-gait trajectory
//      artifact. These are single trajectories (thousands of steps, one
//      gait each); a GPU launch would be pure overhead for work this small,
//      so — mirroring 10.03's energy_conservation_cpu — they live here,
//      CPU-only, by design.
//
// This file itself is almost entirely PLUMBING: every actual physics
// formula (the serpenoid gait, the anisotropic friction law, the
// prescribed-joint Newton-Euler step) lives ONCE, as the __host__ __device__
// inline functions in kernels.cuh, and is called UNCHANGED from here and
// from kernels.cu's GPU kernel (see that header's file comment for why this
// sharing is what makes the §5 tolerance number meaningful rather than a
// guess). Nothing below hand-copies a formula — if it looks like plumbing,
// it is.
//
// Read this after: kernels.cuh (the physics). Companion: kernels.cu (the
// GPU kernel this file's sweep_cpu() is the oracle for).
// ===========================================================================

#include "kernels.cuh"

// ---------------------------------------------------------------------------
// sweep_cpu — recompute GaitResult for exactly the requested indices,
// sequentially, using the SAME decode_gait()/simulate_gait() the GPU kernel
// calls. `indices` need not be sorted or contiguous — main.cu picks a
// stride-sampled subset spanning the whole (amplitude, phase, frequency)
// grid so the gate exercises every axis, not just one corner of it.
// ---------------------------------------------------------------------------
void sweep_cpu(const int* indices, int n_indices, const GaitGridParams& grid, const SimParams& sim,
              float mu_t, float mu_n, GaitResult* out)
{
    for (int k = 0; k < n_indices; ++k) {
        const GaitParams gp = decode_gait(indices[k], grid, mu_t, mu_n);
        simulate_gait(gp, sim, out[k]);
    }
}

// ---------------------------------------------------------------------------
// run_single_gait_cpu — simulate one fully-specified gait. A one-line
// wrapper around simulate_gait(), but naming it separately gives main.cu's
// three verification gates (zero-amplitude, isotropic-friction, turning-
// bias) a clear, self-documenting call site instead of three inline
// simulate_gait() calls scattered through the orchestration code.
// ---------------------------------------------------------------------------
void run_single_gait_cpu(const GaitParams& gp, const SimParams& sim, GaitResult& out)
{
    simulate_gait(gp, sim, out);
}

// ---------------------------------------------------------------------------
// run_single_gait_logged_cpu — same simulation as run_single_gait_cpu, but
// re-implements simulate_gait's loop HERE (rather than calling it) so it
// can sample (t, x, y, yaw) every log_stride steps along the way — the
// data for demo/out/best_gait_path.csv. The per-step PHYSICS is still the
// single shared snake_step() from kernels.cuh; only the "sample and stash a
// row every so often" bookkeeping is new here, which is why this is a
// reference_cpu.cpp addition rather than a THIRD kernels.cuh function that
// would otherwise need to be written HD (host+device) for no reason — the
// trajectory artifact is a CPU-only, one-gait, one-time report.
//
// Returns the number of rows actually written (min(desired, max_log_rows));
// row 0 is always t=0 (the true starting pose), so a caller plotting this
// file sees the whole trajectory including the origin.
// ---------------------------------------------------------------------------
int run_single_gait_logged_cpu(const GaitParams& gp, const SimParams& sim, int log_stride,
                               int max_log_rows, float* t_log, float* x_log, float* y_log,
                               float* yaw_log, GaitResult& out)
{
    float x = 0.0f, y = 0.0f, yaw = 0.0f;
    float vx = 0.0f, vy = 0.0f, yaw_rate = 0.0f;
    float path_len = 0.0f, effort = 0.0f;

    int n_rows = 0;
    // Row 0: the true initial pose, before any stepping.
    if (n_rows < max_log_rows) {
        t_log[n_rows] = 0.0f; x_log[n_rows] = x; y_log[n_rows] = y; yaw_log[n_rows] = yaw;
        ++n_rows;
    }

    for (int s = 0; s < sim.n_steps; ++s) {
        const float t = static_cast<float>(s) * sim.dt_s;
        snake_step(x, y, yaw, vx, vy, yaw_rate, t, sim.dt_s,
                  gp.amp_r, gp.beta_r, gp.omega_rps, gp.gamma_r,
                  sim.link_len_m, sim.link_mass_kg, sim.gravity_mps2, gp.mu_t, gp.mu_n,
                  path_len, effort);

        // Sample AFTER every log_stride-th completed step (s is 0-based, so
        // "step number s+1 has just finished" is (s+1) % log_stride == 0).
        if (((s + 1) % log_stride) == 0 && n_rows < max_log_rows) {
            t_log[n_rows] = static_cast<float>(s + 1) * sim.dt_s;
            x_log[n_rows] = x; y_log[n_rows] = y; yaw_log[n_rows] = yaw;
            ++n_rows;
        }
    }

    out.final_x_m = x;
    out.final_y_m = y;
    out.final_yaw_r = yaw;
    out.distance_m = sqrtf(x * x + y * y);
    out.path_length_m = path_len;
    out.straightness = (path_len > 1.0e-9f) ? (out.distance_m / path_len) : 0.0f;
    out.effort_j = effort;
    const float total_weight_n = static_cast<float>(kNLinks) * sim.link_mass_kg * sim.gravity_mps2;
    out.cot = effort / (total_weight_n * fmaxf(out.distance_m, 1.0e-6f));

    return n_rows;
}
