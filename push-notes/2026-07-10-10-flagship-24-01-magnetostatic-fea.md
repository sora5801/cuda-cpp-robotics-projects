# Push note — 2026-07-10-10: flagship 24.01 magnetostatic fea

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 24.01 — **2-D magnetostatic field solving + motor cogging sweeps** — is done: a
variable-coefficient elliptic PDE solver (red-black SOR on the magnetic vector potential,
harmonic-mean interface reluctivities) computes the field of a 4-pole/6-slot permanent-magnet
motor cross-section, Maxwell-stress-tensor torque is integrated in the air gap, and a
5-arc-fraction × 24-rotor-angle sweep (120 batched field solves, ~1.2 s GPU) answers a real
design question: **peak cogging torque has a genuine non-monotonic minimum at magnet-arc fraction
0.70** (2.04 N·m/m vs 7.22 at full arc) — the classic motor-design heuristic, emerging from
Maxwell's equations rather than being asserted. The analytic gates are physics classics:
Ampère's law on a current annulus (0.19% error, bore field ~2.5e-08 T where theory says zero) and
normal-flux continuity across an air/iron interface (0.35%).

## What changed

- **[projects/24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/](../projects/24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/)** —
  complete: batched red-black SOR kernel (blockIdx.z = rotor-angle variant), equivalent-current
  PM model, MST contour torque kernel, CPU twin, two analytic gates + cogging physics sanity
  (mean-zero, pole-pitch periodicity), field PGM + waveform CSV artifacts, full README / THEORY /
  PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 24.01 → `done` (**26/505**).

## New projects (didactic blurbs)

**24.01 — Magnetostatic FEA + cogging sweeps** (★ beginner, domain 24, flagship). The
physics-deepest project so far: Maxwell → magnetostatics → vector potential → equivalent
magnetizing currents → the Maxwell stress tensor, each derived in THEORY.md; plus the classic FVM
lesson of *why* interface coefficients take the harmonic mean (a flux-tube series argument). GPU
content: variable-coefficient stencil iteration (31.01's family) batched across sweep variants in
one launch. A genuine FP-reproducibility finding is documented: the interface gate measures 0.35%
in Release but 2.6% in Debug (contraction differences through thousands of SOR sweeps) — the
tolerance is set from that measured spread. The single most interesting thing to look at:
`demo/out/field_magnitude.pgm` — flux crowding in the teeth, exactly like the textbook figures,
because it *is* the textbook equation.

## How to build & run

```powershell
projects\24-actuators-motors\24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque\demo\run_demo.ps1
# then open demo\out\field_magnitude.pgm and plot demo\out\cogging_waveforms.csv
```

## What to study here

`THEORY.md` end to end — this is the repo's electromagnetic-physics anchor (24.02 optimization,
24.03 FOC, and 25.07 coil design all build on it) → `src/kernels.cu` (the stencil + batching).
First exercise: change the slot count to 9 and predict the new cogging period from lcm(P,S)
before running.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 10 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** worst |ΔA| = 2.9e-07 Wb/m over 65,536 nodes (tol 2e-5).
- **Analytic gates:** Ampère annulus 0.19% at three radii, bore residual 2.5e-08 T; interface
  flux continuity 0.35% (tol 5%, spread-calibrated).
- **Design sweep:** arc 0.60→5.48, **0.70→2.04 (min)**, 0.80→3.57, 0.90→6.67, 1.00→7.22 N·m/m;
  cogging mean-zero sanity ≤ 0.0009 of peak on all fractions; pole-pitch periodicity exact to
  print precision.
- Timing (teaching artifact): one 1500-sweep solve ≈ 30 ms GPU vs 416 ms CPU; full 120-solve
  sweep ≈ 1.16 s.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Regular-grid FDM/FVM (unstructured FEM documented as production — FEMM/Maxwell/JMAG named),
  linear materials (no saturation curve — documented), 2-D per-unit-length torque units stated
  honestly, no winding-current torque study (that is the ripple half — an exercise plus 24.02).

## Next push preview

25.01 battery electro-thermal simulation, then 26.01 topology optimization closes batch 1f.
