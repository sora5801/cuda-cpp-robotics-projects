# Push note — 2026-07-08-02: flagship 09.01 batched fk

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Foundation flagship #2: **09.01 batched forward kinematics** is complete to the Definition of Done.
One GPU thread computes the full FK chain of a synthetic 6-DoF arm for its own configuration —
200,000 configurations per launch — with the robot model broadcast from `__constant__` memory. It
extends 33.01's thread-per-problem pattern with the kinematics toolkit every later project leans
on: rigid-transform composition, Rodrigues' formula, Shepperd's stable matrix→quaternion
conversion in the repo's (w,x,y,z) convention, and double-cover-aware pose comparison. The catalog
calls FK "the foundation for everything above"; it is now in place.

## What changed

- **[projects/09-dynamics-kinematics/09.01-batched-forward-kinematics/](../projects/09-dynamics-kinematics/09.01-batched-forward-kinematics/)** —
  complete: FK kernel + constant-memory model + host API
  ([`src/kernels.cu`](../projects/09-dynamics-kinematics/09.01-batched-forward-kinematics/src/kernels.cu)),
  layout contracts ([`src/kernels.cuh`](../projects/09-dynamics-kinematics/09.01-batched-forward-kinematics/src/kernels.cuh)),
  oracle twin ([`src/reference_cpu.cpp`](../projects/09-dynamics-kinematics/09.01-batched-forward-kinematics/src/reference_cpu.cpp)),
  two-stage driver with hemisphere-aware comparator
  ([`src/main.cu`](../projects/09-dynamics-kinematics/09.01-batched-forward-kinematics/src/main.cu)),
  synthetic 6-DoF-arm sample (~5.3 KiB, seed 42) +
  [`scripts/make_synthetic.py`](../projects/09-dynamics-kinematics/09.01-batched-forward-kinematics/scripts/make_synthetic.py),
  full README / THEORY / PRACTICE, data & demo READMEs, all scaffold markers resolved.
- **[docs/STATUS.md](../docs/STATUS.md)** — 09.01 → `done` (**2/505**).

## New projects (didactic blurbs)

**09.01 — Batched forward kinematics** (★ beginner, domain 09, flagship). Teaches how "where is
the hand?" is computed and why sampling-based robotics needs it 10⁵ at a time: the chain
`p ← p + R·t; R ← R·R_fix·Rot(axis, q)` is inherently sequential along the arm but embarrassingly
parallel across configurations. New CUDA concept beyond 33.01: `__constant__` memory and its
warp-broadcast read path for the shared robot model. The single most interesting thing to look at:
`mat3_to_quat` in `src/kernels.cu` — Shepperd's four-branch stable conversion, with the
double-cover story told at the comparator in `main.cu`.

## How to build & run

```powershell
projects\09-dynamics-kinematics\09.01-batched-forward-kinematics\demo\run_demo.ps1
# or: open build\batched-forward-kinematics.sln in VS 2026, Release|x64, build, run the exe
```

## What to study here

Project `README.md` (System context — how FK feeds IK/grasping/planning) → `THEORY.md` §The
problem (where a robot model's numbers physically come from: castings, gearboxes, encoders,
calibration) → `src/kernels.cuh` (the layout contracts) → `src/kernels.cu` (constant memory,
Rodrigues, Shepperd). Exercises to try first: add a tool transform (README Exercise 1); then
batched numerical Jacobians (Exercise 4) as the bridge to 09.02.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-08):

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 7 stable lines matched `demo/expected_output.txt`,
  exit 0. Worst GPU-vs-CPU deviations: **8.9e-08 m** position (tol 1e-4 m), **1.8e-07** quaternion
  (tol 1e-4, hemisphere-aligned), norm invariant 1.8e-07.
- Observed timing (single-shot teaching artifact, kernel-only vs one CPU core): FK ×200,000
  configs ≈ 0.19 ms GPU vs ≈ 48–54 ms CPU (~250–280×).
- `tools/verify_project.py`: **all structural gates PASS**.

## Known limitations / TODOs

- Revolute-only single serial chain, FP32, synthetic archetype model (no URDF loader) — scoping
  documented in the project README §Limitations; trees/prismatic/tool frames are exercises and
  domain-09 siblings.
- Worker dispatch remains blocked by the Claude session limit (resets 4:20 PM); flagships are
  being built by the lead inline. 07.09 and 08.01 remain in this foundation sequence.

## Next push preview

Foundation flagship #3: 07.09 jump-flooding Voronoi/distance transforms — the repo's first
grid/stencil-pattern project. Then 08.01 MPPI to close the foundation set.
