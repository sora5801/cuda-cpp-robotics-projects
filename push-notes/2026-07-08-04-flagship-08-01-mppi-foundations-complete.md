# Push note — 2026-07-08-04: flagship 08.01 mppi foundations complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

**The foundation set is complete.** Flagship 08.01 — MPPI, the catalog's "canonical GPU
controller" — closes the four foundation projects (33.01 → 09.01 → 07.09 → 08.01) that CLAUDE.md
§11 ordered first because later flagships reuse their patterns. This one is the payoff: the first
project where the GPU *controls* something. A force-limited cart-pole swings itself up from
hanging and balances, driven closed-loop at 50 Hz by 4096 GPU-simulated futures per tick. The demo
writes the trajectory as a plottable artifact, and the whole loop — noise, rollouts, softmin
blend, receding horizon — fits in one readable file pair. With the foundations battle-tested over
four builds, the standards are ready for parallel worker batches on the remaining 32 flagships.

## What changed

- **[projects/08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/](../projects/08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/)** —
  complete: cart-pole dynamics (from the Lagrangian) + RK4 + stage cost + the rollout kernel
  ([`src/kernels.cu`](../projects/08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/src/kernels.cu)),
  oracle twin + plant stepper ([`src/reference_cpu.cpp`](../projects/08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/src/reference_cpu.cpp)),
  the full MPPI closed loop with verify stage and trajectory artifact
  ([`src/main.cu`](../projects/08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/src/main.cu)),
  scenario sample + generator, full README / THEORY / PRACTICE, all markers resolved.
- **[docs/STATUS.md](../docs/STATUS.md)** — 08.01 → `done` (**4/505**; Phase 1 foundations 4/4).

## New projects (didactic blurbs)

**08.01 — MPPI controller** (★ beginner, domain 08, flagship). Teaches sampling-based MPC end to
end: perturb the plan (Gaussian noise, transposed layout so rollout reads coalesce), simulate K
futures (one thread per rollout — the pattern from 33.01/09.01 now driving a control loop),
weight by exponentiated negative cost, blend, act, recede. The plant is the classic force-limited
cart-pole, chosen because swing-up defeats every linear controller and *emerges* from the
optimization. New CUDA concepts: an RK4 integrator inside a kernel, cost functionals along
trajectories, and the uniform-read/coalesced-read/divergent-read spectrum completed across the
four foundation projects. The single most interesting thing to look at: plot
`demo/out/trajectory.csv` and watch the controller discover energy pumping.

## How to build & run

```powershell
projects\08-control-systems\08.01-mppi-controller-the-canonical-gpu-controller\demo\run_demo.ps1
# then plot demo\out\trajectory.csv (theta_rad vs t_s) to see the swing-up
```

## What to study here

The four foundations are designed as a sequence — if you are starting the repo here, read
33.01 → 09.01 → 07.09 → 08.01. Within 08.01: `README.md` → `THEORY.md` §The problem (why
swing-up is the honest benchmark) and §The math (where the exponential weights come from) →
`src/main.cu` (the loop) → `src/kernels.cu` (the kernel). First exercises: plot the artifact
(Exercise 1), then break the temperature λ both ways (Exercise 2).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-08):

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 6 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** worst relative rollout-cost deviation **1.8e-07** over 4096 rollouts
  (tol 1e-3).
- **Closed-loop behavior:** swing-up completed at step 113 (t ≈ 2.3 s); pole balanced
  (|θ| < 0.2 rad) for the **final 287 steps** (success threshold: final 100); final state
  p = 0.033 m, θ = 0.006 rad.
- Timing (single-shot teaching artifacts): rollout set ≈ 0.8–1.4 ms GPU kernel vs ≈ 18–21 ms
  single-core CPU; closed-loop average ≈ 0.3–0.4 ms GPU kernel per 20 ms tick.
- `tools/verify_project.py`: **all structural gates PASS** (4/4 foundation flagships green).

## Known limitations / TODOs

- Plant-as-model (zero mismatch), host-side softmin + per-tick noise upload (didactic
  transparency; Exercises 3–4 remove them), cart-pole rung only — all documented in README
  §Limitations with the full-strength sim-only caveat (this project's output is a force command).
- Worker dispatch was blocked by the Claude session limit all afternoon; all four foundations
  were built by the lead inline. Batch dispatch for the remaining 32 flagships resumes when the
  limit resets.

## Next push preview

Phase 1 continues: worker batches (8–16 parallel) on the remaining 32 flagships, starting with the
★ entries (01.02 stereo SGM, 02.06 ICP, 04.01 particle filter, 05.01 TSDF, …), one polished
project per domain, pushed in ~6-flagship batches per CLAUDE.md §11.
