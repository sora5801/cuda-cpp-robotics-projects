# Push note — 2026-07-10-13: flagship 27.04 composite layup

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 27.04 — **composite layup optimization + Tsai–Wu failure envelope sweeps** — opens batch
1g: the full classical-laminate-theory chain (lamina Q → transformed Q̄(θ) → ABD assembly →
per-ply stresses → Tsai–Wu first-ply failure) derived in THEORY.md and swept over 256 symmetric
8-ply stacks × 18 load cases on the GPU (1.5 ms), plus two 128×128 failure envelopes. The design
lesson lands as measured numbers: for mixed loading the quasi-isotropic stack wins (factor 1.46 —
with the honest CLT identity that all 24 angle-permutations tie, because membrane response
depends only on the angle multiset), while for aligned loading the all-0° stack wins by a factor
of 10. The build's best moment is a caught physics subtlety: the first isotropy gate accidentally
tested an elliptical (Nx,Ny) slice rather than rotation invariance; the fix applies a genuine
tensor rotation and proves algebraically that F66 = 3F11 is what isotropy requires — all kept in
THEORY.md and the gate's comments.

## What changed

- **[projects/27-materials-manufacturing/27.04-composite-layup-optimization-tsai-wu-failure/](../projects/27-materials-manufacturing/27.04-composite-layup-optimization-tsai-wu-failure/)** —
  complete: shared host/device CLT+Tsai–Wu core, layup-sweep + envelope kernels, full-recompute
  CPU oracle, four analytic gates (all ~1e-7), ranking CSV + two envelope PGMs + contour CSVs,
  full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 27.04 → `done` (**29/505**).

## New projects (didactic blurbs)

**27.04 — Composite layup + Tsai–Wu** (intermediate, domain 27, flagship). Anisotropy as a design
variable: composites let you *place stiffness where the loads are*, and CLT is the algebra that
makes that placement computable. The failure-criteria taxonomy is taught honestly (max-stress vs
interactive criteria, Tsai–Wu's interaction term and its known controversy). The single most
interesting thing to look at: the two envelope PGMs side by side — the quasi-isotropic stack's
rounded envelope vs the cross-ply's angular one; the shape *is* the design decision.

## How to build & run

```powershell
projects\27-materials-manufacturing\27.04-composite-layup-optimization-tsai-wu-failure\demo\run_demo.ps1
# then open demo\out\envelope_best.pgm + envelope_cross.pgm, read layup_ranking.csv
```

## What to study here

`THEORY.md` §The math (the CLT chain, then the F66 = 3F11 isotropy proof) → `src/kernels.cuh`
(the shared physics core) → the gate design in `main.cu`. First exercise: add ±30° to the angle
alphabet and see whether anything beats quasi-isotropic on the mixed set.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 14 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** worst relative deviation 1.8e-06 over 37,376 fully-recomputed points
  (tol 1e-3).
- **Analytic gates (all ~1e-7):** single 0° ply fails at exactly Xt·t / Xc·t; isotropic-degenerate
  material direction-independent under properly-rotated uniaxial load (spread 1.0e-07); cross-ply
  A11 = A22 and Q̄(0) = Q exact; load-factor homogeneity exact.
- **Design results:** mixed-set winner [0/−45/90/45]s at 1.4623 (24 CLT-tied permutations
  documented); aligned-set winner all-0° at 10.0000.
- Timing (teaching artifact): sweep 1.5 ms + envelopes 0.13 ms GPU.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Membrane loads on symmetric laminates (bending/coupling B-D behavior documented as the
  extension), first-ply failure only (progressive failure documented), synthetic material
  constants labeled as such (coupon testing named as the real-world anchor in PRACTICE).

## Next push preview

28.01 real-time FEM soft arm, then 29.05 ultrasound beamforming and 30.01 the agriculture bundle
close batch 1g.
