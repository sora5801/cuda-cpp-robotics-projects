# Push note — 2026-07-10-12: flagship 26.01 topology optimization batch 1f complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 26.01 — **SIMP topology optimization on GPU** — is done, closing **batch 1f** (21.04,
24.01, 25.01, 26.01; **28/505 overall, 28 of 36 flagships**). The catalog calls this the flagship
design project, and it delivers the full classic pipeline: matrix-free Jacobi-preconditioned CG
plane-stress FEA (gather kernel, zero atomics, element stiffness derived by Gauss quadrature and
cross-checked against the 99-line-topopt constant), SIMP penalization, mesh-independence
filtering, and optimality-criteria updates — run on two load cases: the canonical MBB beam and a
robot motor-flange L-bracket, both converging to textbook diagonal-strut topologies at exactly the
0.4000 target volume. The FEA is verified the way FEM is supposed to be: a **patch test**
reproducing the exact linear field at 2.8e-6, and a cantilever matching **Timoshenko** beam theory
at 0.8% with the residual honestly attributed to Q4 shear locking. Warm-started CG (400 → 44
iterations across outer iterations) keeps the whole two-case demo at ~8 s.

## What changed

- **[projects/26-mechanical-design-structures/26.01-topology-optimization-on-gpu-for-lightweight/](../projects/26-mechanical-design-structures/26.01-topology-optimization-on-gpu-for-lightweight/)** —
  complete: matrix-free K·u gather kernel + PCG, sensitivity + filter kernels, host OC bisection,
  CPU twin, patch/beam/optimization gates, two topology PGMs + convergence CSV artifacts, full
  README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 26.01 → `done` (**28/505**).

## New projects (didactic blurbs)

**26.01 — Topology optimization** (★ beginner, domain 26, flagship). Where robot lightweighting
actually comes from: linear elasticity → Q4 stiffness → compliance minimization, SIMP's
push-to-0/1 penalization logic, the checkerboard pathology and why filtering fixes it, and the
GPU lesson of the batch — **matrix-free beats assembled-sparse** for structured-grid FEA (no
matrix, no atomics, one gather per node). The `Emin/E0 = 1e-3` conditioning trade for
Jacobi-only preconditioning is documented rather than hidden. The single most interesting thing
to look at: `demo/out/topology_bracket.pgm` — a motor-bracket truss the optimizer *invented*.

## How to build & run

```powershell
projects\26-mechanical-design-structures\26.01-topology-optimization-on-gpu-for-lightweight\demo\run_demo.ps1
# then open demo\out\topology_mbb.pgm and topology_bracket.pgm (~8 s run)
```

## What to study here

Batch 1f as a set is the robot-as-machine column: watch the human (21.04), design the motor
(24.01), power it (25.01), and hold it together with the least metal (26.01). Within 26.01:
`THEORY.md` §The math (Q4 stiffness + the OC derivation) and the checkerboard/filter story →
`src/kernels.cu` (the matrix-free matvec). First exercise: switch the filter off and watch the
checkerboard pathology appear.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 11 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** displacement 4.3e-3 rel (float CG, documented), compliance 5.2e-8.
- **FEM gates:** patch test 2.8e-6 rel (tol 1e-3); tip deflection 0.8% vs Timoshenko (Q4 shear
  locking attributed, Euler–Bernoulli comparison shown too).
- **Optimization gates (both cases):** volume exactly 0.4000; 100% connected solid; compliance
  monotone from iteration 6 (worst uptick 0.07% vs 0.5% slack). MBB compliance 0.0904 J; bracket
  0.0359 J.
- Timing (teaching artifact): both 80-iteration cases ≈ 7.5 s GPU total; warm-start effect
  400 → 44 CG iterations.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- 2-D plane stress (3-D and level-set documented — the bullet's slash), single load case per run,
  Jacobi-only preconditioning with the documented Emin trade. Retrospective queue unchanged
  (template LNK4099; twin-vs-shared ruling; CMake data-path resolver hardening — seen again here).

## Next push preview

Batch 1g: 27.04 composite layup + Tsai–Wu sweeps, 28.01 real-time FEM soft arm, 29.05 ultrasound
beamforming, 30.01 the agriculture bundle (milestone 1: fruit detect + localize).
