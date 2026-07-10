# Push note — 2026-07-10-14: flagship 28.01 soft arm fem

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 28.01 — **real-time FEM soft arm + model-based control** — is done: a 1,440-element
corotational-linear soft arm (2-D rotation extraction derived in closed form via atan2 of the
deformation gradient) steps at dt = 30 µs on the GPU, runs **1.69× faster than reality** at this
mesh (measured, not promised), identifies its own tip-deflection-per-tension Jacobian by probing
the model, and closed-loop tracks four tip setpoints through antagonistic tendon tensions with
zero overshoot and sub-0.2 mm steady-state error. The physics gates triangulate the model from
three directions: static tip deflection 0.5% off Euler–Bernoulli, first bending mode 1.4% off the
analytic cantilever frequency, and unactuated energy drift bounded at 4.4% (symplectic Euler's
signature, continuing 10.03's energy-gate lineage).

## What changed

- **[projects/28-soft-robotics/28.01-real-time-fem-soft-arm-model-model-based-control/](../projects/28-soft-robotics/28.01-real-time-fem-soft-arm-model-model-based-control/)** —
  complete: corotational Q4 force kernel (scatter-with-atomics assembly — the documented contrast
  to 26.01's gather), symplectic integration kernel, tendon actuation, Jacobian identification +
  PI tip controller, CPU twin, three analytic gates + four setpoint gates + the real-time gate,
  trajectory/snapshot/frame artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 28.01 → `done` (**30/505**).

## New projects (didactic blurbs)

**28.01 — Real-time FEM soft arm** (★ beginner, domain 28, flagship). Why soft robots break the
rigid-body abstraction, why plain linear FEM explodes at large rotations (spurious volume growth)
and how corotational decomposition fixes it, where the explicit-dynamics stability limit comes
from (c = √(E/ρ) — derived), and what "model-based control" honestly means when the model is the
plant you probe. The assembly-strategy taxonomy (scatter+atomics vs coloring vs gather) is this
project's GPU lesson, deliberately paired against 26.01. The single most interesting thing to look
at: `demo/out/tip_trajectory.csv` plotted — four clean steps from a controller whose only model
is a number it measured.

## How to build & run

```powershell
projects\28-soft-robotics\28.01-real-time-fem-soft-arm-model-model-based-control\demo\run_demo.ps1
# then plot demo\out\tip_trajectory.csv and demo\out\arm_snapshots.csv (~7 s run)
```

## What to study here

`THEORY.md` §The problem (why soft robotics is hard) and the corotational derivation →
`src/kernels.cu` (the scatter assembly and its race story) → the identification-then-control
design in `main.cu`. First exercise: switch to pure linear elasticity and watch the large-rotation
artifact appear — the reason corotational exists, demonstrated.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead (the builder's session was cut by the account
limit after its demo passed; the lead re-ran every gate and confirmed the one noted doc
discrepancy had already been fixed):

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 16 stable lines matched, exit 0 (~7 s).
- **GPU-vs-CPU gate:** worst |dx| = 6.0e-08 m, |dv| = 5.8e-04 m/s over 500 steps.
- **Analytic gates:** static deflection 3.98 vs 4.00 mm EB (0.5%); f₁ = 2.000 vs 2.029 Hz
  analytic (1.4%); unactuated energy drift 4.4% peak (bound 8%, symplectic).
- **Control gates:** Jacobian identified at 1.20e-02 m/N; all four setpoints — 0.0% overshoot,
  steady-state errors 0.04–0.18 mm (bound 0.3 mm).
- **Real-time gate:** 11.58 simulated s in 6.84 wall s → factor 1.69× (≥ 1× required).
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- 2-D plane stress, tendon-line actuation abstraction (pneumatic chambers → 28.03), quasi-static
  identified gain (dynamic model-based control documented as the extension), SOFA named as the
  production framework the title nods to.

## Next push preview

29.05 ultrasound beamforming (medical domain — educational-only framing per §8), then 30.01 the
agriculture bundle closes batch 1g.
