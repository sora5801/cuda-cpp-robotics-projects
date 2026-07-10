# 24.01 — 2D magnetostatic FEA solver on GPU → motor torque-ripple/cogging parameter sweeps

**Difficulty:** ★ beginner · **Domain:** 24. Actuators & Motors

> Catalog bullet (source of truth, verbatim): `★ 2D magnetostatic FEA solver on GPU → motor torque-ripple/cogging parameter sweeps`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds a small GPU solver for the magnetic field inside a permanent-magnet motor
cross-section, and uses it to answer a real motor-design question: **how much of each magnet pole
should physically cover, to make cogging torque — the "detent" feel a slotted motor has even with the
windings switched off — as small as possible?** A `256×256` grid represents a simplified 4-pole,
6-slot motor cross-section (rotor iron core, alternating-polarity radial magnets, air gap, slotted
stator with iron teeth and back iron). A batched, GPU-accelerated iterative solver finds the magnetic
vector potential field for each of 5 candidate magnet pole-arc fractions × 24 rotor positions (120
independent field solves total, batched 24-wide per arc fraction — one kernel-launch sequence per arc
fraction instead of 120 separate ones), the torque at each position is extracted via the Maxwell
stress tensor, and the resulting cogging-torque waveforms are compared to find the arc fraction with
the smallest peak. The demo also runs two INDEPENDENT physics checks against textbook closed-form
answers (Ampere's law for a current-carrying annulus; flux continuity across a straight material
interface) — not just a GPU-vs-CPU agreement check — because a field solver's correctness is a claim
about physics, and physics is what should be checked against.

**Scope note (CLAUDE.md §2):** the catalog bullet's "FEA" is used the way industry uses it loosely
("2D FEA motor design"). What is actually implemented is a finite-difference/finite-volume
discretization of the identical governing PDE a linear-element FEA solver would assemble, on a
regular grid rather than an unstructured mesh — the ratified teaching discretization; see
[THEORY.md "Where this sits in the real world"](THEORY.md#where-this-sits-in-the-real-world) for the
honest comparison to unstructured-mesh production tools. The load-current (FOC-driven) torque case is
documented but not swept — cogging (zero-current) torque is the catalog bullet's explicit target and
this project's full implemented scope.

## What this computes & why the GPU helps

The computation is a **variable-coefficient elliptic PDE relaxation, batched across independent
problem instances**:

- **Per solve:** `−∇·(ν∇A_z) = J` on a 256×256 grid — a 5-point stencil with reluctivity (`ν = 1/μ`)
  averaged harmonically at material faces (air/iron/magnet jump by ~2000×), relaxed by **red-black
  Successive Over-Relaxation** (SOR): one thread per grid cell, in-place, checkerboard-colored to make
  a normally-sequential Gauss-Seidel update race-free on a GPU. Pattern: **stencil**.
- **Across the sweep:** 5 arc fractions × 24 rotor angles = 120 independent field solves. Each arc
  fraction's 24 rotor-angle variants are **batched into one kernel-launch sequence** (`blockIdx.z`
  selects the variant) — the "many independent small problems" pattern is exactly as GPU-friendly as
  "one big problem," provided the batch axis rides outside the per-variant layout so coalescing is
  unaffected. Pattern: **batched independent solves**.
- **Measured reality (RTX 2080 SUPER):** one 256×256 variant solved to full convergence (1500
  red+black sweep-pairs) takes **~29 ms on the GPU vs. ~455 ms on one CPU core** (~15× — a teaching
  artifact, not a benchmark, see [Limitations](#limitations--honesty)); the full 120-solve sweep
  (batched 24-wide, 5 launch sequences) completes in **~1.07 s total** on the GPU.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** cross-cutting **infrastructure — the actuator-design slot**, not a runtime
  perception/planning/control layer. This is a design-time, OFFLINE engineering tool: it runs once
  (or a handful of times) per motor design, long before a robot ships, not once per control tick. It
  sits in the same "infrastructure: compute, comms, power, **mechanical structure**" cross-cutting
  band SYSTEM_DESIGN item 1 names for the physical machine itself.
- **Upstream inputs:** the motor's PERFORMANCE REQUIREMENTS (peak/continuous torque, speed range,
  mass and volume envelope, thermal budget) — in this repo's terms, exactly what a **whole-robot
  actuator selection optimizer (catalog 24.14)** would hand off once it has decided "this joint needs
  a custom motor, not a catalog part": pole/slot count candidates, target torque, envelope
  constraints. This project takes those requirements' GEOMETRIC consequence (a candidate cross-
  section) as its starting point.
- **Downstream consumers:** the motor's DRIVE ELECTRONICS and CONTROL design. Most directly, **FOC
  simulation (catalog 24.03)** — current-loop tuning grids and sensorless-observer banks — consumes
  the motor THIS project designs: FOC's current-loop gains, its torque-per-amp constant, and its
  cogging-compensation feedforward table (if any) all depend on the field solution and cogging
  waveform this project produces. The mechanical design (rotor/stator lamination stack, magnet
  procurement) is the other downstream consumer, on the hardware side (PRACTICE §1–2).
- **Rate/latency budget:** **none, in the control-loop sense — and that is the honest point.** This
  is an offline design tool: SYSTEM_DESIGN item 1's Hz/latency bands (camera 30–60 Hz, state estimator
  100–400 Hz, whole-body control 0.5–1 kHz, …) describe RUNTIME software; this project's "budget" is
  engineering-iteration turnaround — how many candidate designs an engineer can evaluate in a work
  session. The measured ~1.07 s per 5-arc-fraction sweep is what makes a much LARGER design-space
  search (README Exercise 4) tractable on a single desktop GPU, which is the actual value the "GPU"
  half of this catalog bullet buys over a CPU-only tool.
- **Reference robot(s):** every reference robot with a JOINT ultimately has a motor behind it, but
  CUSTOM motor-field design (as opposed to selecting a catalog motor) concentrates in the
  **quadruped** (13x: custom high-torque-density actuators are a competitive differentiator — foothold
  scoring, leg dynamics, and the actuator's own torque ripple all interact) and the **6-DoF
  manipulator work cell** (09/19x: joint actuators where smoothness at low speed — exactly what
  cogging degrades — matters for fine manipulation). SYSTEM_DESIGN's warehouse-AMR and quadrotor
  reference robots typically use CATALOG motors (24.14's territory) rather than bespoke field-solved
  designs.
- **In production:** a purpose-built motor-design suite (FEMM, Ansys Maxwell, JMAG, Motor-CAD — see
  [Prior art](#prior-art--further-reading)) with an unstructured, adaptively-refined mesh, a nonlinear
  (saturating) iron model, and often a full electromagnetic-thermal-structural co-simulation loop —
  this project's linear, fixed-grid, 2D-magnetostatic core is the teaching foundation those tools
  build on (THEORY.md "Where this sits in the real world" details every gap).
- **Owning team:** **actuation/hardware engineering** (motor design, sometimes a dedicated "electric
  machines" specialist role) — adjacent to mechanical design (who builds what this project's field
  solution constrains — PRACTICE §1), controls/electronics (who owns the FOC drive this project's
  output feeds — PRACTICE §3), and the actuator-selection/systems team (24.14) who decided a custom
  motor was warranted in the first place (SYSTEM_DESIGN item 5).

## The algorithm in brief

- **2D magnetostatic vector-potential formulation** — Maxwell's equations reduced to a scalar elliptic
  PDE `−∇·(ν∇A_z) = J` for the z-component of the magnetic vector potential, with `B = curl(A)` giving
  the in-plane field. → [THEORY.md §The math](THEORY.md#the-math)
- **Permanent-magnet modeling via equivalent magnetizing current** — `J_m = curl(M)`, computed by a
  central finite difference of the rasterized (piecewise-constant) magnetization field — reproduces
  the textbook "bound surface current at each pole edge" automatically. → [THEORY.md §The math](THEORY.md#the-math)
- **Harmonic-mean face reluctivity at material interfaces** — the correct averaging for flux crossing
  an iron/air/magnet boundary in series, derived from a 1D flux-tube argument. → [THEORY.md §Numerical considerations](THEORY.md#numerical-considerations)
- **Red-black Successive Over-Relaxation (SOR)** — a checkerboard-colored, in-place, GPU-parallel
  Gauss-Seidel relaxation of the 5-point variable-coefficient stencil, batched across independent
  rotor-angle variants in one kernel-launch sequence. → [THEORY.md §The GPU mapping](THEORY.md#the-gpu-mapping)
- **Maxwell stress tensor torque extraction** — cogging torque from a circular air-gap contour
  integral of `Br·Bθ`, derived from first principles. → [THEORY.md §The math](THEORY.md#the-math)

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/2d-magnetostatic-fea-solver-on-gpu-motor-torque.sln`](build/2d-magnetostatic-fea-solver-on-gpu-motor-torque.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/2d-magnetostatic-fea-solver-on-gpu-motor-torque.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuBLAS/cuFFT/cuSOLVER: the red-black SOR solver is a hand-written stencil kernel, deliberately, so the
whole elliptic-solve algorithm is visible in `src/kernels.cu` rather than hidden behind a sparse-linear-
algebra library call.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU and
two independent physics gates, runs the design sweep):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including the
**two artifacts to open** (a field-magnitude image and the cogging-waveform design plot).

## Data

The committed sample is a **motor design**, not a recording: `data/sample/motor_scenario.csv`
(405 bytes, synthetic — every field is a fixed design constant, not measured or downloaded) — grid
resolution, cross-section geometry, materials, and the cogging sweep plan (5 magnet pole-arc
fractions × 24 rotor angles). Regenerate with `python scripts/make_synthetic.py` (deterministic — no
seed needed, nothing here is random). No public dataset applies to a from-scratch motor cross-section
design; `scripts/download_data.ps1` is an honest no-op. Full field documentation, units, and the
committed SHA-256: [`data/README.md`](data/README.md).

## Expected output

Ten stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, `ANALYTIC_AMPERE:`,
`ANALYTIC_INTERFACE:`, `SWEEP:`, `PHYSICS:`, `ARTIFACT:`, `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Four independent verification stages, all
measured on the reference machine (RTX 2080 SUPER, sm_75, Release build):

1. **`VERIFY`** — the §5 GPU-vs-CPU gate: one representative motor variant solved by the GPU batched
   solver and the CPU twin must agree within `|dA| ≤ 2e-5` Wb/m (measured worst: `2.948e-07` Wb/m).
2. **`ANALYTIC_AMPERE`** — a uniform-current annulus in air, solved on the same solver, must match
   Ampere's law's closed form (zero field in the bore; the correct `B(r)` elsewhere) within 5%
   (measured: ~0.19% relative error at three sample radii; bore field `2.5e-8` T).
3. **`ANALYTIC_INTERFACE`** — a straight air/iron interface must show continuous normal `B` within 5%
   (measured: ~0.35% in Release, ~2.6% in Debug — see [THEORY.md](THEORY.md#numerical-considerations)
   for why the tolerance carries that much headroom).
4. **`PHYSICS`** — every cogging waveform integrates to ~zero net torque (measured `|mean|/peak`:
   `0.0000`–`0.0009` across all five arc fractions) and repeats after one magnet pole pitch (measured
   `|diff|/peak = 0.0000` against an independent solve).

**The design result** (an `[info]` line, not stable — see below): the sweep finds its minimum peak
cogging torque at magnet arc fraction **0.70** (peak `2.0444` N·m/m) vs. `0.60` (`5.4841`), `0.80`
(`3.5724`), `0.90` (`6.6671`), `1.00` (`7.2236`) N·m/m — a genuine, non-monotonic minimum, not the
largest or smallest arc fraction tested. Torque is reported per unit axial stack length (N·m/m — the
honest 2D unit, THEORY.md explains why); for an illustrative 30 mm stack, the same numbers read as
`61–217` mN·m — illustrative scaling, not a claimed real-motor result.

**Why no raw number appears on a checked line:** everything here is deterministic FP32 arithmetic (no
RNG), but ~3000 chained sweep passes accumulate compiler-FMA-contraction differences across platforms
and optimization levels (measured and quantified in THEORY.md) — so, following 31.01's precedent, only
PASS/FAIL verdicts are checked; every measured number lives on an unchecked `[info]`/`[time]` line.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: scenario loading → motor
   rasterization → VERIFY → two analytic gates → the batched sweep → physics sanity → artifacts. The
   single most interesting thing to look at: `rasterize_motor()`'s five-region classification and how
   little code it takes to turn "pole count, slot count, arc fraction, rotor angle" into a full
   material map.
2. [`src/kernels.cuh`](src/kernels.cuh) — the batched grid layout contract, the solver constants, and
   why the geometry/materials live in a runtime-loaded scenario rather than compile-time constants.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: the red-black SOR kernel. Read the header comment
   first — it explains why red-black needs no ping-pong buffer, and why the batch dimension costs
   nothing in memory coalescing.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the sequential twin; diff it against
   `kernels.cu`'s kernel body — only the indexing machinery differs, not the arithmetic.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **FEMM (Finite Element Method Magnetics, David Meeker)** — the free, widely-used 2D magnetostatic/
  electrostatic solver this project's scope most directly mirrors; its documentation is an excellent,
  freely available derivation of the same vector-potential formulation and equivalent-current PM
  model this project implements, on an unstructured triangular mesh instead of a regular grid.
- **Ansys Maxwell / JMAG / Motor-CAD** — the commercial motor-design suites production engineers use:
  nonlinear (saturating) iron, adaptive meshing, full electromagnetic-thermal-structural coupling, and
  automated cogging/torque-ripple optimization loops — everything THEORY.md's "Where this sits in the
  real world" section names this project as the linear, 2D, unsaturated core of.
- **Zhu & Howe, "Influence of Design Parameters on Cogging Torque in Permanent Magnet Machines"
  (IEEE Trans. Energy Conversion, 2000)** — the classic analytic-and-FEA study of exactly this
  project's design question (magnet arc, slot opening, skew); read it to see how far the closed-form
  analysis goes before FEA becomes necessary, and how production designers cross-check the two.
  Hendershot & Miller's "Design of Brushless Permanent-Magnet Machines" is the standard textbook
  covering the same ground at book length.
  ★ Illustrative reference title/venue — verify exact citation before relying on it (dated 2026-07-10).
- **cuRobo / OMPL / Drake** — not directly related to motor design, but the repo's standard "study the
  production tool, reimplement the teaching core" pattern this project follows for FEMM/Ansys Maxwell
  is the same relationship those projects have to their own production counterparts elsewhere in this
  repo — worth reading once for the pattern.

## Exercises

1. **Plot the artifacts:** `demo/out/field_magnitude.pgm` (view directly, or convert with
   ImageMagick/GIMP) and `demo/out/cogging_waveforms.csv` (theta vs. torque, one column per arc
   fraction). Confirm by eye that the arc fraction `[info] design result:` names really does have the
   smallest peak-to-peak swing.
2. **Contour-radius sensitivity:** the Maxwell stress integral is theoretically contour-independent
   within the air gap (THEORY.md derives why); edit `r_contour` in `main()`'s sweep stage to the gap's
   inner and outer thirds instead of the midpoint, rebuild, and measure how much the reported torque
   actually shifts at this grid's resolution — an honest measurement of a coarse teaching grid's limit.
3. **Widen the sweep:** add more arc fractions (e.g., 0.05 steps from 0.5 to 1.0) and confirm the
   minimum found stays near 0.70, or find a sharper one — the batched-solve pattern in `kernels.cu`
   needs zero changes, only the scenario CSV's `SWEEP_ARCS` row.
4. **Scale up the design space:** add a second swept parameter (e.g., `SLOT_OPEN` fraction) and batch
   over the full 2D grid of (arc fraction × slot-open fraction × rotor angle) variants — the point of
   this exercise is to feel how much of a bigger search the GPU's batching headroom buys you (the
   measured ~1.07 s for 120 solves suggests thousands of solves are still comfortably interactive).
5. **[R&D-adjacent] Add saturation:** wrap the existing linear solve in an outer Newton-Raphson (or
   simple fixed-point) loop that updates `mu_r_iron` per node from a small tabulated B-H curve and
   re-solves — THEORY.md "Where this sits in the real world" sketches exactly this extension.

## Limitations & honesty

- **Regular grid, not an unstructured FEA mesh.** Geometry is rasterized onto a fixed 256×256
  Cartesian grid rather than meshed with elements that conform to curved boundaries — a "staircase"
  approximation of every circular/angled edge, and a fixed (not locally refined) resolution, including
  in the air gap (only ~5 cells wide at this grid's resolution) where the field varies fastest. This
  is the ratified teaching discretization (CLAUDE.md §2) — mathematically the same PDE, a simpler
  implementation, honestly labeled "FEA-class" rather than claimed to be a production mesh solver.
- **Linear materials — no magnetic saturation.** Iron's permeability is a FIXED `mu_r = 2000`, not a
  saturating B-H curve; the model is only accurate below the iron's saturation flux density (this
  project's fields stay comfortably below it, but a design pushed harder would need the nonlinear
  extension THEORY.md and Exercise 5 describe).
- **Magnetostatic only — no eddy currents, no time variation.** The rotor is evaluated at a sequence
  of FIXED angles ("stroboscopic" snapshots), not as a continuously spinning source; a spinning
  rotor's changing flux induces eddy currents in solid/laminated iron that this project does not model
  (THEORY.md "Where this sits in the real world").
- **2D — no 3D end effects.** The reported torque (N·m/m, per unit axial length) is exact only for an
  infinitely long stack; a real motor's finite stack length fringes flux at both ends, an effect this
  project does not compute (and reports the per-length unit honestly instead of hiding the gap behind
  an assumed stack length).
- **No thermal or mechanical coupling.** Magnet strength (`Br`) drops with temperature; mechanical
  deflection/eccentricity under load shifts the air gap. Neither is modeled here — see PRACTICE §1–2
  for where these effects live on a real motor.
- **Cogging (zero-current) torque only** is swept, per the catalog bullet's explicit target; the
  load-current (FOC-driven) torque case shares this project's exact solver (set `J_free` nonzero in
  the windings) but is not exercised by the shipped demo.
- **Sim-validated only, and honestly not the project whose caveat matters most here:** this project's
  output is an OFFLINE design number, not a real-time command to hardware (see "System context" —
  there is no control loop here to caveat). The repo-wide caveat (CLAUDE.md §1) still applies in the
  general sense — nothing here is safety-certified, and a motor built from this project's output would
  need the full engineering validation ladder PRACTICE §3 describes before it drives anything.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), never a benchmark
  claim.
