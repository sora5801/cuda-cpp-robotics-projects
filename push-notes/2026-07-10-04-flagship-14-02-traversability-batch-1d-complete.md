# Push note — 2026-07-10-04: flagship 14.02 traversability batch 1d complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 14.02 — **traversability costmaps fusing semantics + geometry** — is done, closing
**batch 1d** (11.01, 12.01, 13.03, 14.02; **20/505 overall, 20 of 36 flagships**). The project is
built around the two cases that make sensor fusion a real problem rather than an average: a water
pool that geometry calls perfectly flat (**semantics must veto — and does, 224/224 cells**) and a
vegetation patch that geometry calls dangerously rough (**semantics must rescue — and does,
0/1870 vetoed, at a visibly reduced speed limit**). The vehicle limits are derived, not asserted:
for the illustrative platform, rollover geometry (26.57°) governs over traction (34.99°), and the
step limit falls out of a friction-cone corner-contact argument. A per-cell max-safe-speed layer
makes the output directly consumable by 14.01's MPPI.

## What changed

- **[projects/14-locomotion-wheeled/14.02-traversability-costmaps-fusing-semantics/](../projects/14-locomotion-wheeled/14.02-traversability-costmaps-fusing-semantics/)** —
  complete: geometric layer (dual-window slope/step/roughness), confidence-weighted semantic
  layer, fusion with asymmetric hard vetoes, speed-limit layer, CPU twin (double-precision plane
  fits, documented), designed-disagreement gates, three artifacts (two PGMs + the transect CSV),
  full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 14.02 → `done` (**20/505**).

## New projects (didactic blurbs)

**14.02 — Traversability fusion** (intermediate, domain 14, flagship). Why geometry and semantics
*disagree* — grass lies to LiDAR, water lies to everything — and how a fusion rule encodes risk
policy: confidence-immune water vetoes (asymmetric risk), pessimistic fallbacks for low-confidence
labels ("don't trust" degrades to "don't know", never to "assume cheap"), and an honest
weighted-vs-MAX fusion comparison run live in the demo. Deliberately paired with 13.03: same
elevation-map substrate, discrete footholds for legs there vs continuous drivability for wheels
here. The single most interesting thing to look at: `demo/out/layers.csv` plotted — all three
layers plus fusion along a transect crossing every feature.

## How to build & run

```powershell
projects\14-locomotion-wheeled\14.02-traversability-costmaps-fusing-semantics\demo\run_demo.ps1
# then open demo\out\traversability.pgm + speed_limit.pgm, plot demo\out\layers.csv
```

## What to study here

Batch 1d as a set spans the deploy chain: simulate the sensor (11.01), run the network (12.01),
score the terrain (13.03 legs / 14.02 wheels). Within 14.02: `THEORY.md` §The problem (why the
channels disagree — the physics) and the fusion-policy discussion → `src/kernels.cu`. First
exercise: drop the water veto's confidence immunity and find the scenario where that kills the
robot on paper.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 6 stable lines matched, exit 0.
- **Stage-isolated GPU-vs-CPU gates:** slope 1.1e-06 rad; step exact; roughness 2.2e-08;
  semantic/fusion ~6e-08; veto reasons 0/65,536 mismatched; speed exact.
- **Designed-disagreement gates:** flat-but-water vetoed 224/224 (geometry alone scored it 0.10 —
  "safe"); rough-but-vegetation 0/1,870 vetoed (geometry alone 0.44 — "bad"), speed reduced to
  2.22 of 2.50 m/s; ditch geometrically vetoed to exactly 1.0 despite cheap GRAVEL labels; berm
  measured 17.86° vs constructed 18.00° (tol ±1.5°).
- A live weighted-vs-MAX fusion comparison prints as a teaching diagnostic (directional, honestly
  labeled not-a-gate in this scenario).
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- No NaN/dropout modeling (deliberate scope: the fusion story stays central; 13.03 covers the
  NaN-veto discipline), single vehicle parameter set (illustrative, dated), static maps.
- A ditch-center plane-fit cancellation (symmetric geometry fitting to zero slope) is kept as a
  numerics case study in THEORY.md.

## Next push preview

Batch 1e: 16.01 thruster allocation QP, 18.01 snake serpenoid sweeps, 19.01 antipodal grasp
scoring, 20.01 GelSight tactile processing.
