# 28.01 — Real-time FEM soft-arm model + model-based control (GPU SOFA-style)

**Difficulty:** ★ beginner · **Domain:** 28. Soft Robotics

> Catalog bullet (source of truth, verbatim): `★ Real-time FEM soft-arm model + model-based control (GPU SOFA-style)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**A soft robot has no joints to put encoders on — its "state" is a continuum, and controlling it
means controlling a deformable body through a physics model that runs faster than reality.** This
project builds that entire loop, small enough to read in an afternoon: a 2-D cantilevered soft arm
(120×12 bilinear elements of a synthetic 1 MPa elastomer-class material) simulated with **explicit
corotational-linear FEM** on the GPU — one thread per element scatters forces with atomics, one
thread per node integrates with symplectic Euler at a CFL-derived 30 µs timestep — plus two
antagonistic tendons and a **model-based task-space controller** that first *measures its own
plant's tip Jacobian by probing the FEM model*, then tracks tip setpoints with a PI loop through
tendon-tension differentials. The demo verifies the GPU against a CPU twin, checks the physics
against three *analytic* anchors (Euler-Bernoulli statics, the cantilever mode frequency, energy
conservation), runs the closed loop through four setpoints, and reports a measured **real-time
factor of ~1.7×** on the reference GPU. Everything the catalog bullet names is implemented; the
"GPU SOFA-style" framing (what SOFA does differently and better) is documented in
[THEORY.md](THEORY.md) §Where this sits in the real world.

## What this computes & why the GPU helps

Every 30 µs timestep: 1,440 elements × (rotation extraction + an 8×8 stiffness matvec + a scatter)
followed by 1,573 nodes × (damping + tendon forces + a symplectic-Euler update) — ~33,000 steps per
simulated second, forever.

- **Pattern:** stencil-shaped **scatter with atomics** (one thread per element, `atomicAdd` into
  shared nodes — the deliberately-taught dual of 26.01's gather; the race story is told in
  [`src/kernels.cu`](src/kernels.cu)'s header) + a per-node **map** for integration.
- **Measured reality:** the CPU twin steps at ~92 µs/step single-core; the GPU two-kernel step
  averages ~18 µs wall including launch overhead — which is the honest, interesting regime: at this
  mesh size the hot loop is **submission-bound, not arithmetic-bound** (removing one API call per
  step measurably cut 27.9 → 17.9 µs/step; THEORY.md §The GPU mapping).
- **Why that matters:** real-time factor is the product spec here. Model-based soft control needs
  the model to outrun the arm; the measured 1.7× (11.6 simulated seconds in 6.8 wall seconds,
  across every phase including sensing and logging) is what makes the controller in this very demo
  possible.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **soft-robotics modeling slot** — a real-time plant model that sits
  between design and control. It is simultaneously *simulation* (cross-cutting layer) and the
  *model inside a model-based controller* (control layer): soft robots have no joint-angle
  abstraction, so the FEM model IS the state representation the rest of the stack reasons about.
- **Upstream inputs:** an arm **design** — geometry, material, actuator routing. In this repo's
  domain that means the actuator/arm concepts of 28.03 (PneuNet-style bending actuators) and its
  siblings; here the design arrives as the committed scenario file (message shape: a parameter
  struct, `data/sample/arm_scenario.csv`).
- **Downstream consumers:** soft-arm **controllers and planners** (28.x's control-oriented
  projects) that need tip poses, Jacobians, and fast rollouts; in this demo the consumer is
  built-in — the identified-Jacobian PI controller consuming tip positions (message shape: a
  `PoseStamped`-like tip state at ~333 Hz).
- **Rate / latency budget:** **the real-time factor IS the budget.** A model-based soft controller
  must integrate the model at least as fast as wall clock — anything below 1× and the "model-based"
  premise collapses to open-loop guessing. Measured here: 33 kHz physics stepping (30 µs dt),
  333 Hz control ticks, real-time factor 1.7× with every overhead included. Production stacks
  (SOFA-based surgical/continuum work) fight for exactly this number at richer meshes.
- **Reference robot(s):** the **soft-arm archetype** — continuum manipulators (surgical
  catheter/endoscope robots, trunk-like manipulators) and **soft grippers** (tendon- and
  pneumatically-driven fingers wrapping produce or delicate parts). Of this repo's five reference
  robots, the nearest slot is the manipulator work cell's gripper end — where compliance replaces
  force control.
- **In production:** the **SOFA framework** (the bullet's namesake) is the real version of this —
  implicit integration, richer elements, contact, GPU plugins; **Elastica** covers slender-rod
  soft dynamics (Cosserat rods); **MuJoCo's** deformable bodies serve RL-scale soft simulation.
  This project is those tools' teaching core: explicit, visible, one file per idea.
- **Owning team:** soft robotics R&D (SYSTEM_DESIGN item 5) — typically a simulation-plus-controls
  group inside a soft-robotics company or lab; adjacent: materials/mechanical (owns the arm the
  model must match) and controls (consumes the model).

## The algorithm in brief

- **Corotational-linear FEM** — per element, extract the rotation from the deformation gradient in
  closed form (2-D polar decomposition: one `atan2`), compute linear-elastic force in the unrotated
  frame, rotate back. Stable at large rotations where pure linear FEM grows spurious volume. →
  [THEORY.md](THEORY.md) §The math (including the derivation and the linear-FEM failure demo).
- **Explicit symplectic-Euler dynamics** — lumped mass, no linear solve; dt derived from the CFL
  bound (wave speed c = √(E/ρ) ≈ 30 m/s → dt ≤ h/c; we run at 0.45× that). → THEORY §numerics.
- **Rayleigh damping, split honestly** — α (mass-proportional) damps the slow bending mode; β
  (stiffness-proportional) is small, pays explicit integration's stability tax, and — as this
  project *measured* — is what keeps the warped-stiffness corotational force from self-exciting. →
  THEORY §Numerical considerations (the flutter detective story).
- **Tendon actuation** — two antagonistic distributed line forces along the top/bottom fibers;
  differential tension bends the arm (the bimetallic-strip mechanism); co-contraction bias is
  bounded by the arm's own Euler buckling load (a real soft-robot design constraint, derived in
  [`src/kernels.cuh`](src/kernels.cuh)).
- **Model-based tip control** — identify the quasi-static tip Jacobian by probing the model itself
  (apply ΔT, settle, measure), then PI on tip error with gains derived from the identified J and a
  loop-shaping argument (resonant loop gain < 1). → THEORY §The model-based-control story.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/real-time-fem-soft-arm-model-model-based-control.sln`](build/real-time-fem-soft-arm-model-model-based-control.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/real-time-fem-soft-arm-model-model-based-control.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
The FEM assembly, integrator, controller, and even the PGM rasterizer are hand-rolled teaching code.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU
and every analytic gate):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
the **three artifacts to plot** (tip trajectory, arm snapshots, deformed-mesh image).

## Data

The committed sample is a **scenario definition**, not recordings:
`data/sample/arm_scenario.csv` (~2.4 KB, synthetic, zero RNG) — the soft arm's model constants
(cross-checked at startup against [`src/kernels.cuh`](src/kernels.cuh); a mismatch refuses to run)
plus the runtime controller scenario (probe size, PI tuning, setpoint sequence — editable without
rebuilding). Regenerate byte-identically with `python scripts/make_synthetic.py`. No public dataset
applies — the ground truth here is *analytic* — so `scripts/download_data.ps1` is an honest no-op.
Details, field docs, and the SHA-256: [`data/README.md`](data/README.md).

## Expected output

Sixteen stable lines — banner, `PROBLEM:`, `MESH:`, `SCENARIO:`, `VERIFY: PASS`, three
`GATE ...: PASS`, `IDENTIFY: PASS`, four `SETPOINT n: PASS`, `ARTIFACT:`, `REALTIME: PASS`,
`RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). What each verdict measured on the reference
RTX 2080 SUPER (each printed on its adjacent `[info]` line):

1. **VERIFY** — 500 steps of full dynamics, GPU scatter+atomics vs CPU sequential twin: worst
   |Δx| ≈ 6e-8 m, worst |Δv| ≈ 7e-4 m/s against tolerances 1e-5 m / 5e-3 m/s (the tolerance is
   reassociation-aware: atomics sum in hardware order; THEORY §How we verify).
2. **GATE static-deflection** — tip sag under a 0.02 N tip load: measured 3.977 mm vs
   Euler-Bernoulli's 4.000 mm (0.6% error; 30% allowance).
3. **GATE first-mode-frequency** — zero-crossing estimate 2.000 Hz vs analytic 2.029 Hz (1.4%
   error; 20% allowance).
4. **GATE energy-conservation** — peak total-energy drift 4.3% over 4 α-undamped periods (8%
   bound, with the measured drift budget documented in `src/main.cu`).
5. **IDENTIFY** — tip Jacobian 1.20e-2 m/N from a 0.18 N probe.
6. **SETPOINT 0–3** — targets ±2.11 / +1.06 / 0 mm: rise-to-90% ≈ 1.16 s, overshoot 0.0%,
   steady-state error 0.04–0.18 mm (bounds: reached, ≤60%, ≤0.3 mm).
7. **REALTIME** — 11.6 simulated s in 6.8 wall s = **1.7×** (≥1× required).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the contract: mesh/DOF layout, material record with
   units, the CFL-derived dt, the damping split (and why β is load-bearing), the tendon model and
   its buckling bound. Every derivation the code relies on lives here, once.
2. [`src/kernels.cu`](src/kernels.cu) — the heart: `elem_force_kernel` (rotation extraction →
   local force → **atomicAdd scatter**, with the full gather-vs-scatter race story in the header)
   and `node_integrate_kernel` (symplectic Euler + the zero-after-consume trick that cut the
   per-step API overhead by a third). The single most interesting thing: the whole "real-time FEM"
   is just these two kernels, ~33,000 times per simulated second.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the deterministic sequential twin (fixed
   summation order — the thing the GPU's atomics deliberately don't have), plus the shared
   derivations: the 8×8 element stiffness from Gauss quadrature, lumped mass, and the energy
   diagnostics.
4. [`src/main.cu`](src/main.cu) — the staged pipeline: scenario load + model cross-check → verify
   → analytic gates → identify → closed loop → artifacts → real-time factor. The PI controller is
   ~40 lines in plain sight, including the anti-windup.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **SOFA framework** — the bullet's namesake: the open-source soft-body simulation framework used
  across surgical and soft robotics, with implicit integrators, contact, and GPU plugins. This
  project is its explicit, single-file teaching core; compare SOFA's implicit-Euler + CG solve with
  our explicit stepping (THEORY §real world).
- **Müller et al., "Interactive Virtual Materials" (2004)** — the warped-stiffness corotational
  method this project implements — including its documented instability, which this project
  *measured* (THEORY §numerics); Chao et al. (2010) and McAdams et al. (2011) are the "fix it
  properly" follow-ups.
- **Elastica (Gazzola lab)** — Cosserat-rod soft dynamics: the 1-D slender-body alternative to
  2-D/3-D FEM for arms; the right tool when the arm is truly rod-like.
- **MuJoCo (deformable/flex bodies)** — soft bodies inside an RL-scale simulator; the
  performance-vs-fidelity trade at industrial scale.
- **Della Santina, Katzschmann, Rus et al.** — the soft-robot *control* literature this project's
  identified-Jacobian controller is a 1-DOF miniature of (model-based control of soft arms via
  reduced models and measured Jacobians).

## Exercises

1. **Plot the loop:** `demo/out/tip_trajectory.csv` → tip and setpoint vs time, then the tension
   differential. Find the rise, the (absent) overshoot, and the integrator slowly erasing the last
   0.1 mm. Then scatter a snapshot from `arm_snapshots.csv` to see the bent arm.
2. **Break linear FEM (the gate as a lab):** in `src/kernels.cu`'s `elem_force_kernel`, force
   `theta = 0` (rotation extraction off — pure linear FEM), rebuild, run. Watch the static gate
   inflate the deflection and the energy gate explode as rotations grow — then explain both from
   THEORY §why linear FEM fails.
3. **Meet the flutter:** set the ring's β to exactly 0 in `src/main.cu`, rebuild, and watch the
   energy gate fail with monotonic growth. Measure the growth rate at dt and dt/2 and convince
   yourself it is *not* an integrator artifact (THEORY §numerics tells you what it is).
4. **Retune the controller from the file:** edit `PI_MARGIN_ALPHA` to 0.6 in
   `data/sample/arm_scenario.csv` (no rebuild needed) and run — the loop rings at the arm's
   resonance and the SSE gates fail. Derive why from the resonant-loop-gain argument in
   `src/main.cu`'s gain comment, then find the largest margin that still passes.
5. **Go variational (the real fix):** implement the missing ∂R/∂x term of the corotational force
   (Chao et al. 2010) in a copy of `elem_force_kernel`, and show the fully-undamped ring now
   conserves energy — the flutter was never inevitable, just the price of the standard shortcut.

## Limitations & honesty

- **2-D plane stress, not a 3-D arm.** The teaching core is deliberately planar; the mesh/DOF
  layout, scatter assembly, and corotational extraction all generalize to 3-D (8-node hexes, a real
  polar decomposition) at ~4× the code and none of the new ideas. THEORY §real world maps the path.
- **The tendon model is an abstraction** — distributed axial line forces along the fibers (the
  standard lumped model of an embedded/bonded actuator), not a routed cable with friction, slack,
  and pulleys. It also applies forces in the fixed −x direction, which bakes in the P−δ
  (buckling-proximity) softening that the identified Jacobian honestly absorbs. PRACTICE §1
  describes what real tendon routing adds.
- **The plant is the model** — the controller drives the same FEM it identified against (zero
  model mismatch, deliberately ideal). Real soft robots add hysteresis, creep, and manufacturing
  spread; that gap is exactly why the *identify-then-control* workflow taught here (rather than
  trusting an analytic model) is the field's default.
- **β damping is load-bearing** — the classic warped-stiffness corotational force self-excites
  without it (measured: ~6.4/s amplitude growth from rest, dt-independent). We kept the standard
  force and documented the crutch instead of silently hiding it; Exercise 5 is the proper fix.
- **The energy gate runs at β = 5e-6 s, not literal zero** (mode-1 damping ζ₁ = 3.2e-5 — 0.16%
  energy over the window) for exactly that reason; the drift bound's measured budget is in
  `src/main.cu`.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER, sm_75), stated
  where measured. The real-time factor is measured end-to-end and holds with margin on the
  reference GPU; it is not a promise about other hardware.
- **Sim-validated only (CLAUDE.md §1):** this project's output is tendon tension commands — code
  of exactly the kind that would move hardware. Everything here ran only against the simulated
  arm; nothing is safety-certified, and any hardware use would demand the full testing ladder
  (PRACTICE §3) plus an independent safety envelope.
