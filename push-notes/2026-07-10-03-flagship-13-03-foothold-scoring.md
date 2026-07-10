# Push note — 2026-07-10-03: flagship 13.03 foothold scoring

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 13.03 — **foothold scoring kernels** — is done: the quadruped terrain-evaluation pipeline
(slope via least-squares plane fits, roughness from plane residuals, distance-to-hazard, fused
score with friction-derived vetoes) plus a batched foothold-selection consumer, all gated against
a designed terrain whose features *are* the ground truth (the 15° ramp measures 15.007°). The
push's real teaching treasure is a documented bug story: a variable-shadowing bug existed
**identically in both the GPU kernel and its CPU twin** — so the twin comparison passed while both
were wrong — and only the analytic flat-region gate caught it. That is the strongest possible
real-world argument for this repository's two-tier verification doctrine, and THEORY.md §How we
verify correctness now tells it as a case study.

## What changed

- **[projects/13-locomotion-legged/13.03-foothold-scoring-kernels/](../projects/13-locomotion-legged/13.03-foothold-scoring-kernels/)** —
  complete: slope/roughness kernel (in-thread Cramer 3×3 plane fits), bounded distance-to-hazard,
  fusion with hard vetoes (NaN, slope > atan μ — derived), 1000-query foothold selection, CPU
  twin, four stage-isolated gates + four analytic terrain gates, score-map PGM + selections CSV
  artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 13.03 → `done` (**19/505**).

## New projects (didactic blurbs)

**13.03 — Foothold scoring** (intermediate, domain 13, flagship). Where legged locomotion meets
terrain: support polygons and friction cones turn a coefficient of friction into a slope limit
(derived), elevation-map pathologies (holes, drift) turn into hard vetoes, and three per-cell
scores fuse into the map a footstep planner consumes. GPU content: windowed gathers, stage
isolation for verification, per-query argmax selection. The single most interesting thing to look
at: the shared-bug case study in THEORY — why "GPU matches CPU" can be a false comfort, and what
catches it.

## How to build & run

```powershell
projects\13-locomotion-legged\13.03-foothold-scoring-kernels\demo\run_demo.ps1
# then open demo\out\foothold_score.pgm — hazard bands around the step, rocks, and hole
```

## What to study here

`THEORY.md` §The problem (friction cones → the slope limit) and §How we verify correctness (the
shared-bug case study) → `src/kernels.cu`. First exercise: lower μ to 0.3 and watch the ramp
band flip from walkable to vetoed.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 6 stable lines matched, exit 0.
- **Stage-isolated GPU-vs-CPU gates:** slope 1.4e-06 rad / roughness 1.5e-08 m; edge distance
  exact (0.0); fusion 1.2e-07; selection indices 0/1000 mismatched.
- **Analytic terrain gates:** flat control max slope 1.30° (bound 3.4°), mean score 0.987;
  ramp measured 15.007° vs constructed 15.00° (tol ±1.5°); step edge hard-vetoed to exactly 0
  with the far-cell distance saturating at the exact 0.20 m cap; 1000/1000 selections valid and
  in-radius.
- **Two real bugs found by the gates and kept as case studies:** the twin-invisible shadowing bug
  (caught by the analytic gate) and an FMA boundary asymmetry flipping 1/1000 selections (caught
  by the exact-match gate) — both documented in THEORY.md.
- Timings (teaching artifacts): slope/roughness 0.47 ms, edge 0.32 ms, fusion 0.15 ms, selection
  0.016 ms on the GPU.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Single-scale plane-fit window, static map, disc-search selection (kinematic reachability and
  gait timing are 13.02/13.08's jobs — cross-referenced); terrain committed as a recipe, not a
  grid (documented design).

## Next push preview

14.02 traversability costmaps closes batch 1d; then batch 1e (16.01 thruster allocation, 18.01
snake serpenoid, 19.01 grasp scoring, 20.01 GelSight).
