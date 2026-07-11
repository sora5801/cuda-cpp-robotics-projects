# 35.01 — Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics

**Difficulty:** [R&D] research · **Domain:** 35. Micro & Nano Robotics

> Catalog bullet (source of truth, verbatim): `Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics [R&D]`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds two things and glues them together. **First**, a GPU field solver for a 4-coil
electromagnet arrangement (two orthogonal Helmholtz-style pairs around a small workspace) — the
Biot-Savart law, summed over 720 straight wire segments, evaluated on a 256×256 grid. **Second**, a
low-Reynolds-number swarm of 1000 simulated superparamagnetic microrobots that this field's *gradient*
pulls around the workspace, steered by an open-loop, offline-designed 3-waypoint current schedule. A
learner who studies this project should come away understanding: why the Biot-Savart law is a natural
GPU map-reduce; how linearity turns 4 expensive field solves into unlimited cheap ones; why physics at
micron scale is dominated by viscosity, not inertia; and why "pulling" a paramagnetic bead with a
magnetic field gradient can only ever attract, never push.

**Scope (CLAUDE.md §2, §13 — this is an `[R&D]` catalog bullet, shipped as a reduced-scope teaching
version):** IMPLEMENTED — the 4-coil Biot-Savart field solver (with three independent analytic physics
gates, not just a GPU-vs-CPU check), the linearity-exploiting field-combination pipeline, and the
open-loop, low-Reynolds-number swarm dynamics under a fixed 3-waypoint schedule. DOCUMENTED-ONLY (the
research frontier — see [THEORY.md "Where this sits in the real world"](THEORY.md#where-this-sits-in-the-real-world)):
closed-loop (vision/fluoroscopy-feedback) control, heterogeneous swarms, and bead-bead magnetic/
hydrodynamic interactions.

## What this computes & why the GPU helps

Two GPU computations, chained by the field's linearity:

- **The field map: a batched map+reduce.** `B(x) = (mu0/4*pi) * sum_over_segments(I*dl x r / |r|^3)` —
  one thread per one of 65536 grid points, each reducing over 720 coil segments. Pattern: **map+reduce**
  (independent output cells, each a sum over a shared input). This runs 4 times (once per coil, at unit
  current — the catalog bullet's named GPU hook), producing 4 reusable "basis" maps.
- **The swarm: an agent farm.** 1000 robots, each a fully independent thread holding its own `(x,y)`
  state in registers and looping over an entire schedule phase's worth of Euler steps internally.
  Pattern: **batched independent simulation** (the same shape as [08.01](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/)'s
  rollouts and [22.01](../../22-multi-robot-swarms/22.01-100k-agent-swarm-simulator/)'s swarm agents).
- **The bridge: linearity.** Any coil-current vector's field is a linear combination of the 4 basis
  maps — a trivial GPU "map" kernel — so the swarm never re-touches the 720-segment sum; a 4-neighbor
  stencil kernel then precomputes the force-generating gradient once per schedule phase.
- **Measured reality (RTX 2080 SUPER):** the 4 basis-map solves take ~0.9-1.1 ms of GPU kernel time vs.
  ~77-80 ms for ONE coil on one CPU core; the full 3-phase, 1000-robot, 900-step swarm run takes
  ~94-103 ms of GPU kernel time.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** SYSTEM_DESIGN's own framing names this domain explicitly — "soft/medical/field/
  micro/modular robotics (28–30, 35, 36) are the same [autonomy] stack under unusual physics." This
  project spans **perception→state estimation** (the field map stands in for what a real system would
  build from camera/fluoroscopy feedback of bead positions) and **control/actuation** (the current
  schedule is the direct analogue of a torque/force command in a macro-scale robot) — compressed,
  because at this `[R&D]` teaching scope the "sensor" is the ground-truth simulated swarm state, not a
  real image pipeline (see "Limitations & honesty").
- **Upstream inputs:** in a real system, a TASK GOAL (e.g. "move payload/cell cluster to region X") from
  a higher-level planner, plus a MEASURED swarm state from imaging (closed-loop — not implemented here;
  this teaching version's "upstream input" is instead the committed scenario's fixed coil geometry and
  current-schedule design, loaded once at start).
- **Downstream consumers:** the coil DRIVER ELECTRONICS (current amplifiers — PRACTICE.md §2) that would
  turn a commanded current vector into real coil current, and, in a closed-loop system, a vision/state-
  estimation pipeline that would consume the ACTUAL swarm position this project's dynamics model
  predicts.
- **Rate / latency budget:** honestly, **not a fixed-Hz control loop in this teaching version** — the
  demo computes an entire multi-second schedule offline, once, then simulates it. A REAL closed-loop
  system's rate is set by its FEEDBACK sensor: camera-based magnetic microrobot control platforms
  typically run their control loop at the camera's frame rate (tens of Hz, matching SYSTEM_DESIGN item
  1's perception band, 30-60 Hz), well above this bead's own response bandwidth (its "corner frequency"
  — the rate at which `gamma*v = F` can track a changing `F` — is set by the drag/inertia ratio, itself
  microsecond-scale at this size, i.e. utterly non-limiting; the imaging loop is always the bottleneck).
- **Reference robot(s):** **none of SYSTEM_DESIGN's five reference robots** use this capability — it is
  its own class of machine (a bench-top instrument, not a mobile/legged/aerial/manipulator robot). The
  honest reference archetype is the **magnetic-microrobot manipulation platform** (PRACTICE.md §2-3
  names OctoMag-class research systems and magnetic-tweezer instruments) — a fundamentally different
  physical form factor (a coil array around a stationary sample stage) from every SYSTEM_DESIGN
  reference robot.
- **In production:** research-stage platforms (OctoMag and its academic descendants at ETH Zürich and
  peer institutions) and single-molecule magnetic-tweezer instruments (commercial and academic) are the
  closest real systems; there is, as of this writing, no widely commercialized "magnetic microrobot
  swarm" product — this whole capability sits at the preclinical/research end of the pipeline
  (PRACTICE.md §4 is explicit about this).
- **Owning team:** **research** (this is the honest answer for an `[R&D]` bullet) — in a company pursuing
  this capability, it would sit with a dedicated micro/nano-robotics or biomedical-devices research
  group, adjacent to controls/autonomy (who would own any closed-loop extension) and, eventually,
  regulatory affairs (SYSTEM_DESIGN item 5; PRACTICE.md §4 elaborates for the medical-device path).

## The algorithm in brief

- **Discretized Biot-Savart law** — each of 4 coils approximated as a 180-segment polygon; per-segment
  contribution `dB = (mu0/4*pi)*I*(dl x r)/|r|^3`, summed per grid cell. → [THEORY.md §The math](THEORY.md#the-math)
- **Linearity of Maxwell's equations, exploited** — 4 per-unit-current basis maps combine into any
  current configuration's field via a cheap weighted sum; no re-summing segments after the first 4
  solves. → [THEORY.md §The math](THEORY.md#linearity-the-field-of-any-current-combination-is-a-linear-combination)
- **The Helmholtz condition** — coil offset = radius/2 (separation = radius) flattens the field between
  an opposing pair; verified, not just asserted (`GATE_HELMHOLTZ`). → [THEORY.md §The math](THEORY.md#the-helmholtz-condition-why-offset--radius2-flattens-the-field)
- **Superparamagnetic gradient-pulling force**, `F = k*grad(|B|^2/2)`, derived from `m=(V*chi/mu0)*B`
  and the curl-free-region identity `(B.grad)B = grad(|B|^2/2)`. → [THEORY.md §The math](THEORY.md#the-superparamagnetic-force-law-f--gradm--b-derived-to-f--k--gradb2)
- **Low-Reynolds-number, first-order (inertia-free) dynamics** — `v = F/gamma` (Stokes drag), explicit
  Euler; derived from a real Reynolds-number calculation, not assumed. → [THEORY.md §The problem](THEORY.md#the-problem--physics--engineering-first)
- **Open-loop 3-waypoint schedule**, designed offline by forward-simulating the same linear model once
  for a single point. → [THEORY.md §The algorithm](THEORY.md#the-algorithm)

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/magnetic-microrobot-swarms.sln`](build/magnetic-microrobot-swarms.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/magnetic-microrobot-swarms.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuRAND (the swarm's one random draw, its initial cluster, is generated deterministically on the host
with a fixed-seed xorshift32 + Box-Muller, mirroring [08.01](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/)'s
host-noise pattern); no cuBLAS/cuFFT/Thrust — every kernel is hand-written so the whole pipeline is
visible in `src/kernels.cu`.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at — including the **three artifacts to open**.

## Data

The committed sample, `data/sample/microswarm_scenario.csv` (470 bytes), is a **design, not a
recording** (like [24.01](../../24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/)'s
motor cross-section): coil geometry, fluid/bead material constants, and the open-loop schedule's drive
magnitude — every field a fixed engineering constant, synthetic by definition (CLAUDE.md §8). Regenerate
with `python scripts/make_synthetic.py` (deterministic; no seed needed for the scenario itself — the
swarm's one random draw is seeded independently, inside the demo, from the scenario's own `SWARM` row).
No public dataset applies to a from-scratch coil/swarm design; `scripts/download_data.ps1` is an honest
no-op. Full field documentation, units, and the committed SHA-256: [`data/README.md`](data/README.md).

## Expected output

Fifteen stable lines — banner, `PROBLEM:`, `SCENARIO:`, and eleven `VERIFY_*:`/`GATE_*:`/`ARTIFACT:`
verdicts plus `RESULT:` — checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt).
**Eight independent verification stages**, all measured on the reference machine (RTX 2080 SUPER, sm_75,
Release build; see [THEORY.md "How we verify correctness"](THEORY.md#how-we-verify-correctness) for the
full derivation of each):

1. **`VERIFY_FIELD`** — GPU vs. independent CPU basis-map field (coil 0): measured worst `1.09e-11 T`
   (tol `5e-9 T`).
2. **`VERIFY_DYNAMICS`** — GPU vs. independent CPU swarm result after 300 chained Euler steps, all 1000
   robots: measured worst `1.54e-8 m` (tol `1e-7 m`).
3. **`GATE_ONAXIS`** — a single loop's on-axis field vs. the textbook closed form (never touches the
   grid): measured max relative error `2.54e-4` (tol 1%).
4. **`GATE_HELMHOLTZ`** — the East+West pair's flatness over the actual 8 mm workspace: measured
   variation `1.76e-3` (tol 2%).
5. **`GATE_DIVERGENCE`** — full-3D `div(B) ~ 0` at 5 interior points, a mixed current configuration:
   measured max normalized `1.02e-4` (tol `1e-3`).
6. **`GATE_ATTRACT`** — each of the 4 coils, energized alone, must pull the swarm toward itself: measured
   East `+342 um`, West `-335 um`, North `+360 um`, South `-317 um` in 50 steps (margin `5 um`).
7. **`GATE_WAYPOINTS`** — the real 1000-robot swarm's centroid vs. the offline single-particle plan at
   each of 3 phases: measured `12.2 um`, `12.6 um`, `13.3 um` (tol `300 um`).
8. **`GATE_BOUNDS`** — every recorded snapshot of every robot stays finite and inside the mapped
   workspace, the whole run.

**Why no raw number appears on a checked line:** this project chains hundreds of sequential FP32 Euler
steps and thousands of Biot-Savart segment sums; compiler FMA-contraction differences across platforms
and optimization levels can shift the last bits of such a chain without changing any verdict — following
[24.01](../../24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/)'s precedent,
only PASS/FAIL verdicts are checked, and every measured number lives on an unchecked `[info]`/`[time]`
line in the program's actual output.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: scenario loading → coil geometry →
   4 basis-map solves → `VERIFY_FIELD` → 3 analytic gates → the illustrative artifact → the offline
   "planning pass" → `GATE_ATTRACT` → the real 3-phase swarm run (with `VERIFY_DYNAMICS`,
   `GATE_WAYPOINTS`, `GATE_BOUNDS` woven through it) → the trajectory/density artifacts. The single most
   interesting thing to look at: the "planning pass" — a 1-robot GPU run that designs the schedule the
   1000-robot run is later checked against, using the exact same kernels both times.
2. [`src/kernels.cuh`](src/kernels.cuh) — the whole project's contract in one place: the `CoilSegment` /
   `SwarmScenario` / `Float4` layouts, every unit, and the shared `HOSTDEV` numerical helpers
   (`biot_savart_contribution`, `bilinear_sample`) that keep the GPU kernels and the CPU oracle
   byte-for-byte the same formula.
3. [`src/kernels.cu`](src/kernels.cu) — the 4 kernels, each documented with its OWN thread-mapping and
   memory-behavior reasoning: `biot_savart_basis_kernel` (map+reduce), `combine_field_kernel` (pure map,
   linearity), `gradient_b2_kernel` (stencil), `swarm_step_kernel` (agent farm).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the four CPU oracle twins, built on the SAME
   `HOSTDEV` helpers as the GPU kernels — read it side by side with `kernels.cu` to see exactly what
   parallelization changed (a `for` loop over cells/robots became "one thread per cell/robot").
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Kummer et al., "OctoMag: An Electromagnetic System for 5-DOF Wireless Micromanipulation" (IEEE
  Trans. Robotics, 2010)** — the direct real-world archetype for this project's coil arrangement and
  gradient-pulling force law, scaled up to 8 coils and full closed-loop vision-feedback control. Read it
  to see exactly what this teaching version omits (closed-loop control, 5-DOF field synthesis) and why.
  ★ Illustrative reference title/venue — verify exact citation before relying on it (dated 2026-07-10).
- **Magnetic tweezers literature** (widely used in single-molecule biophysics) — the same
  `F = k*grad(|B|^2)` force law this project derives, applied to calibrated force spectroscopy on single
  molecules rather than robotic transport; a good source for the superparamagnetic-bead force derivation
  from a different application angle.
- **FEMM / Ansys Maxwell** (also named in [24.01](../../24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/)'s
  prior art) — production magnetostatic field solvers; this project's closed-form Biot-Savart approach
  is the special case that applies when the geometry is thin wires in vacuum/air rather than saturating
  iron, so no PDE relaxation is needed at all.
- **[08.01 MPPI](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/)** and
  **[22.01 swarm simulator](../../22-multi-robot-swarms/22.01-100k-agent-swarm-simulator/)** — this
  repo's other "agent farm" kernels; study their register-residency and launch-configuration reasoning,
  directly reused by this project's `swarm_step_kernel`.
- **[24.01 magnetostatic FEA](../../24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/)**
  — this project's electromagnetics sibling; its THEORY.md derives Maxwell's equations for magnetostatics
  in depth (this project builds directly on that groundwork rather than re-deriving it) and its
  red-black SOR kernel is the stencil-pattern sibling to this project's `gradient_b2_kernel`.

## Exercises

1. **Plot the artifacts:** `demo/out/field_magnitude.pgm` (view directly, or convert with ImageMagick/
   GIMP) and `demo/out/swarm_trajectory.csv` (plot `centroid_x_mm`/`centroid_y_mm` vs. `t_s` to see the
   3-phase path; overlay the 5 sample robots to see the swarm's spread). Confirm by eye that the swarm
   traces the North→East→South path the schedule commands.
2. **Widen the tolerance intuition:** re-run `GATE_DIVERGENCE` with `h` set back to `1e-6` (edit
   `main.cu`'s divergence block, rebuild) and watch it fail — then try `h=1e-3` and see truncation error
   start to grow the other way. Find, empirically, the widest `h` range that still passes.
3. **Add a 4th waypoint:** extend the schedule (`phase_coil[]`/`phase_name[]` in `main.cu`) with one more
   single-coil phase and confirm `GATE_WAYPOINTS` still tracks it — the planning pass needs no changes,
   only the phase list.
4. **Measure the attract-only limit:** try to construct a current vector that PUSHES the swarm away from
   a point rather than toward a coil (hint: "The math" derives why you cannot, with a fixed positive
   `chi_eff`) — then read what real 8-coil systems do instead (README "Prior art").
5. **[R&D-adjacent] Sketch closed-loop control:** using the SAME `combine_field_kernel`/
   `gradient_b2_kernel` pipeline, sketch (in comments, no need to fully implement) what a per-tick
   current-vector solve would look like if the swarm's centroid were measured every tick instead of
   planned offline — THEORY.md "Where this sits in the real world" names the shape of this extension.

## Limitations & honesty

- **Reduced-scope `[R&D]` teaching version (CLAUDE.md §2, §13):** the catalog bullet's "Biot-Savart field
  computation + swarm dynamics" is implemented in full for a 4-coil planar arrangement and open-loop
  dynamics; a full research system (OctoMag-class) uses more coils, closed-loop vision feedback, and
  synthesizes local field extrema for genuine multi-directional steering — none of that is implemented
  here, and THEORY.md "Where this sits in the real world" names exactly what each omission costs.
- **Attract-only steering, physically, not just as an implementation gap:** a superparamagnetic bead's
  force always points toward higher `|B|` — this project's open-loop schedule works WITH that constraint
  (single-coil phases), it does not (and, with this coil count/current model, cannot) synthesize a
  repulsive or laterally-steered pull.
- **No bead-bead interactions:** every robot is fully independent (no magnetic dipole-dipole force, no
  hydrodynamic coupling through the shared fluid) — an honest simplification that is fine for this
  project's TRANSPORT task and would need addressing for any SELF-ASSEMBLY behavior.
- **Homogeneous swarm:** every robot shares identical bead radius and susceptibility; heterogeneous
  swarms (different robots responding differently to the same field) are a named open research direction,
  not modeled.
- **Deterministic dynamics by default:** Brownian motion is derived and quantified (THEORY.md "The
  problem") to be honestly small relative to the deterministic drift at this project's bead size and
  timestep (~0.2 um vs. ~11 um per step) — but this ratio is PARAMETER-DEPENDENT, and a smaller bead or
  weaker field would flip it; the demo is deterministic, not because thermal noise is always negligible
  in this size regime, but because it is negligible AT THESE SPECIFIC, DOCUMENTED numbers.
- **2D workspace, idealized medium:** robots move in a plane with uniform, Newtonian fluid properties;
  no obstacles, no confinement geometry, no imaging depth/scattering — PRACTICE.md §1 and §3 name the
  real physical complications a fielded system would face.
- **Sim-validated only, and this project's output is not a hardware command in the usual sense (CLAUDE.md
  §1):** this demo's "current schedule" is a data structure consumed by a simulation, not a real coil
  driver — but the general caveat still applies in full: nothing here is safety-certified, no
  microrobot-swarm hardware exists in this repository's scope, and any real electromagnet/current-driver
  system built from this project's numbers would need the full engineering and (for any medical
  application) regulatory validation ladder PRACTICE.md §3-4 describe, starting from a proper safety
  review of the actual amplifier/coil hardware (PRACTICE.md §2) before any current ever flows.
- **Illustrative physical constants (bead size, susceptibility, current):** chosen to be plausible and
  internally consistent (data/README.md documents the reasoning), not measured from a real bead/coil
  system — dated 2026-07-10, verify current values before relying on any of them for real design work.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), never a benchmark
  claim.
