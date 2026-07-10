# Push note — 2026-07-10-06: flagship 18.01 snake serpenoid

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 18.01 — **snake robots: serpenoid gait sweeps** — is done: 8,192 (amplitude, phase,
frequency) gait combinations of a 12-link planar snake simulated for 8 s each, entirely on the
GPU (78 ms — 840 million gait-steps/second), finding the classic serpenoid speed ridge at
0.54 m/s. The physics gates carry the teaching: a zero-amplitude gait displaces exactly 0.0 m
(analytic identity), and making friction *isotropic* collapses propulsion to **6.3%** of the
anisotropic optimum — the "anisotropy is the necessary ingredient of undulation" theorem measured
as a pass/fail gate rather than asserted. The granular/DEM half of the catalog bullet ships as a
documented milestone pointing at 10.10, per the §2/§13 reduced-scope rule (stated in README §13).

## What changed

- **[projects/18-locomotion-other/18.01-snake-robots/](../projects/18-locomotion-other/18.01-snake-robots/)** —
  complete: prescribed-joint 3-DOF formulation (the snake's shape is known at every t; only the
  head pose is dynamic — the clean teaching trick, documented with the math), smoothed anisotropic
  Coulomb model, thread-per-gait sweep kernel, CPU oracle, four physics gates, sweep-surface CSV +
  heatmap PGM + best-gait path artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 18.01 → `done` (**22/505**).

## New projects (didactic blurbs)

**18.01 — Snake serpenoid sweeps** (intermediate, domain 18, flagship). Hirose's serpenoid curve,
why biology converged on it, and the force-decomposition derivation of why scales/wheels
(μ_t ≪ μ_n) turn body waves into thrust. GPU content: the thread-per-simulation sweep family
(08.01/10.03's pattern) at its most transcendental-bound (~3 billion sinf/cosf calls). Honesty
notes worth reading: the fastest gait is not the straightest, the optimal β sits at the sweep's
own boundary, and a sign-inverted turning gate at a degenerate gait led to a documented
investigation — all kept, none polished away. The single most interesting thing to look at:
`demo/out/sweep_surface.pgm` — the speed ridge as a picture.

## How to build & run

```powershell
projects\18-locomotion-other\18.01-snake-robots\demo\run_demo.ps1
# then open demo\out\sweep_surface.pgm and plot demo\out\best_gait_path.csv
```

## What to study here

`THEORY.md` §The problem (the anisotropic-friction propulsion derivation with the ASCII force
diagram — the heart) → §The algorithm (the prescribed-joint 3-DOF trick) → `src/kernels.cu`.
First exercise: extend the sweep grid past the boundary-sitting β optimum and see where the ridge
really peaks.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 12 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** worst final-position deviation 1.4e-06 m over 32 spot-checked gaits ×
  8,000 chained FP32 steps (tol 1e-3).
- **Physics gates:** zero-amplitude displacement exactly 0.0; isotropic-friction speed 6.3% of
  anisotropic best (bound 20%); turning bias shifts yaw +1.014 rad at a well-conditioned interior
  gait (bound +0.05); amplitude ridge peaks in the interior, clearing both grid edges.
- **Sweep:** 8,192 gaits finite; best gait A=0.373 rad, β=0.100 rad, ω=3.14 rad/s → 0.538 m/s,
  straightness 0.752, COT 1.465.
- Timing (teaching artifact): 78 ms GPU for the full sweep.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Flat-ground anisotropic-friction model (the published lateral-undulation simplification);
  granular/DEM coupling documented-only → 10.10; planar gaits only (sidewinding/rolling
  documented); prescribed joints (perfect tracking assumption, stated).

## Next push preview

19.01 antipodal grasp scoring, then 20.01 GelSight tactile processing closes batch 1e.
