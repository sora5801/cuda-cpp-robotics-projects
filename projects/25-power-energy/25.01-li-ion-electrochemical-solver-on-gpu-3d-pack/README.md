# 25.01 — Li-ion electrochemical (P2D/SPMe) solver on GPU + 3D pack thermal simulation + cooling-design sweeps

**Difficulty:** ★ beginner · **Domain:** 25. Power & Energy

> Catalog bullet (source of truth, verbatim): `★ Li-ion electrochemical (P2D/SPMe) solver on GPU + 3D pack thermal simulation + cooling-design sweeps`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> This project never commands real hardware (it is a design-time simulation tool, not a runtime
> controller); see [Limitations & honesty](#limitations--honesty) for the scoping this implies.

## Overview

This project simulates a 24-cell robot battery pack (4×3×2, e.g. a warehouse-AMR pack) at two coupled
physical scales at once. **Per cell**, it solves the **Single Particle Model (SPM)** — the simplest
rung of the electrochemical ladder the catalog bullet names ("P2D/SPMe") that still contains a genuine
solid-state lithium diffusion PDE and real Butler-Volmer reaction kinetics: one spherical particle per
electrode (anode, cathode), 20 radial finite-volume shells each. **Across the pack**, it solves the
pack's **3-D anisotropic heat equation** on a 32×24×16 voxel grid, with each cell as a heat source and
one face of the pack cooled by a convective (Robin) boundary condition. The two PDEs are coupled
**electro-thermally**: each cell's local temperature Arrhenius-scales its own diffusivity and reaction
rate, and the resulting heat generation feeds the next thermal step — so 12 pack DESIGNS (6 cooling
coefficients × {bottom-plate, side-plate}), all driven by the identical 20-minute AMR duty-cycle
mission, diverge from an identical start purely because of how well each one manages heat.

**What is implemented vs. documented-only** (CLAUDE.md §2/§13 scoping): the **SPM** tier is fully
implemented, GPU-accelerated, and verified below. **SPMe** (adds an electrolyte-phase concentration
state) and full **P2D** (resolves the porous electrode's through-thickness position, coupling many
particles per electrode instead of one representative particle) are the next two rungs of the ladder
the catalog bullet names — their governing equations and exactly where SPM's assumptions break down are
documented in [`THEORY.md`](THEORY.md) "Where this sits in the real world", not silently implemented.
The **3-D pack thermal solver and the 12-design cooling sweep are fully implemented** — that half of
the bullet is not a reduced-scope teaching version, it is the real thing at this repo's grid resolution.

A learner who studies this project should come away understanding: why lithium intercalation is a
diffusion problem (and why that makes fast charging hard); how Butler-Volmer kinetics connect a
particle's surface concentration to a real voltage number; why battery packs need pack-level (not
just cell-level) thermal design; and a very concrete, physically-grounded lesson that **more cooling
capacity is not automatically "better"** — it can trade peak temperature for pack imbalance (see
[Expected output](#expected-output)).

> **Template placeholder notice.** This section previously described the scaffold's SAXPY smoke test.
> `src/` now contains this project's real implementation; nothing here is a toolchain placeholder.

## What this computes & why the GPU helps

Two independent PDEs, each turned into a **batched stencil** — the repo's recurring "many independent
grid problems solved in one launch" lesson (24.01 batches rotor angles; this project batches pack
*designs*):

- **Solid-phase diffusion** (`electrochem_fv_kernel`): a spherical finite-volume stencil, batched over
  every (design, cell, electrode) — up to 576 independent spheres solved in one kernel launch, one
  thread per (particle, radial shell).
- **Pack heat equation** (`thermal_step_kernel`): an anisotropic 3-D Cartesian finite-volume stencil
  with a Robin boundary condition on one face, batched over every design — one thread per (design,
  voxel).

Both are **map/stencil** patterns: each output element depends only on itself and a handful of fixed
neighbors, so one GPU thread per element is the natural mapping, and the *batch* axis (which design,
which particle) rides for free alongside the spatial axes. The honest parallelism story at this
project's scope (24 cells, 12 designs) is discussed in [Limitations & honesty](#limitations--honesty)
and [`THEORY.md`](THEORY.md) "The GPU mapping" — this is exactly where that honesty matters most.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** cross-cutting infrastructure — power & energy — alongside simulation/digital-twin
  tooling, not in the perception→planning→control pipeline itself. It is a **design-time** tool
  (pack engineering, cooling-plate selection), not a component that runs on the robot.
- **Upstream inputs:** a `MissionProfile`-shaped current-vs-time curve — this project's own synthetic
  AMR duty cycle here, but in a real workflow it would come from **fleet/energy planning (25.05,
  by name)**: route length, payload, duty cycle, and ambient-temperature assumptions for the robot
  class being designed.
- **Downstream consumers:** pack mechanical/thermal design (which cooling plate to build,
  answered here) and, by name, **battery management system SOC/SOH estimation (25.02, by name)** — the
  same SPM equations this project solves offline at pack-design time are exactly what 25.02's *runtime*
  estimator (an EKF/UKF or reduced-order model over the same states) would use online, at kHz-to-Hz
  rates, with far less compute budget than a design-time sweep can spend.
- **Rate/latency budget:** **none at runtime** — this is an offline design tool, run once (or a handful
  of times) per pack revision, not a component in any control loop. Contrast with 25.02: a real BMS
  SOC estimator runs at 1–10 Hz on an MCU-class part, with a millisecond-scale budget per cycle. This
  project's honest rate is "however long an engineer is willing to wait for a sweep" (here: ~8 seconds
  on desktop GPU hardware, [Expected output](#expected-output)) — never claimed otherwise.
- **Reference robot(s):** the **warehouse AMR** reference design (`SYSTEM_DESIGN.md` reference robot 1)
  most directly — a 24-cell pack sized like an AMR's traction/hotel-load battery, driven by an AMR-shaped
  duty cycle. The **quadruped** reference design (reference robot 3) is the other natural user of this
  class of tool — legged robots pack cells into tight, thermally awkward volumes where a bottom-vs-side
  cold-plate choice is a real engineering question.
- **In production:** a commercial pack-design workflow would use PyBaMM or a COMSOL-class multiphysics
  battery module for the electrochemistry, and a dedicated CFD/FEA thermal tool (or the same COMSOL
  module) for the pack thermal side — see [Prior art & further reading](#prior-art--further-reading).
- **Owning team:** power & energy engineering (pack design, cell selection, BMS algorithms) — see
  [`PRACTICE.md`](PRACTICE.md) §4 for the fuller org picture and adjacent teams.

## The algorithm in brief

- **Spherical finite-volume solid diffusion** — Fick's second law in spherical coordinates,
  discretized with harmonic-consistent shell volumes/face areas and a Neumann (flux) boundary
  condition carrying the applied current ([`THEORY.md`](THEORY.md) "The algorithm").
- **Butler-Volmer kinetics, closed-form inversion** — the symmetric (α=0.5) case inverts exactly to
  `η = (2RT/F)·asinh(i/(2·i0))`, no Newton iteration needed (THEORY.md "The math").
- **Synthetic OCV curves** — closed-form polynomials shaped to the right qualitative behavior
  (graphite's low flat plateau; a smoother NMC-like cathode slope) — see [Data](#data).
- **Arrhenius electro-thermal coupling** — each cell's own temperature scales its diffusivity and
  reaction-rate prefactor, closing the loop between the two PDEs (THEORY.md "The math").
- **Anisotropic 3-D pack heat equation** — explicit finite-volume with a Robin (convective) boundary
  condition on exactly one face per design, five faces adiabatic (THEORY.md "The algorithm").
- **Batched cooling-design sweep** — 12 designs (6 `h` values × {bottom, side}) advanced together every
  step by both batched kernels (THEORY.md "The GPU mapping").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/li-ion-electrochemical-solver-on-gpu-3d-pack.sln`](build/li-ion-electrochemical-solver-on-gpu-3d-pack.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration (or `Debug|x64` — both build clean, zero warnings).
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/li-ion-electrochemical-solver-on-gpu-3d-pack.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only
(no cuBLAS/cuFFT/Thrust/etc.; the stencils are hand-written, per this project's teaching mandate).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at. Total run time on the reference RTX 2080 SUPER: ~8–10 seconds (built exe already present).

## Data

The committed sample (`data/sample/pack_scenario.csv`) is entirely **synthetic** — hand-chosen teaching
parameters at physically plausible orders of magnitude, generated deterministically (no RNG) by
`scripts/make_synthetic.py`. There is no real, licensable "24-cell robot pack" dataset to download, and
CLAUDE.md §8 explicitly steers away from trying to pass off invented numbers as real measurements — so
every value is labeled synthetic everywhere it appears (this README, `data/README.md`, and the CSV's
own comment header). Full field documentation, units, and the derivation behind the specific numbers
(diffusivities, mission currents, sweep values) are in [`data/README.md`](data/README.md) and
[`THEORY.md`](THEORY.md) "The math".

## Expected output

Nine gates, each printing a stable `PASS`/`FAIL` line (`src/main.cu`'s file header narrates the full
pipeline). Measured on the reference RTX 2080 SUPER (numbers will vary slightly by GPU/CPU — only the
PASS/FAIL verdicts are checked, CLAUDE.md determinism discipline extended across FP32/platform ulps):

| Gate | What it checks | Measured (reference run) | Tolerance |
|------|-----------------|---------------------------|-----------|
| `VERIFY` | GPU electrochemistry + thermal kernels vs. their CPU twins | worst \|Δc\| ≈ 0, worst \|ΔT\| ≈ 0 | 5×10⁻² mol/m³, 5×10⁻⁴ K |
| `ANALYTIC_DIFFUSION` | Quasi-steady `c_surf − c_avg` vs. closed form `jR/(5D)` | 70.25 vs. 80.00 mol/m³ (12.2% — a measured, understood `O(1/kNShells)` discretization bias, not a bug) | 15% |
| `ANALYTIC_COULOMB` | Total mole change vs. exactly-integrated applied flux | rel. error 0.35% | 1% |
| `ANALYTIC_THERMAL` | Steady single-face pack vs. exact energy balance `P=hAΔT` | 9.26 vs. 9.60 K (3.5%) | 5% |
| `PHYSICS` | Thermal-grid energy conservation, every design | worst rel. error 0.70% | 2% |

The sweep itself (`SWEEP:` / `[info] design result:` lines) reports the actual finding: with this
project's anisotropic conductivity (`kz` << `kx`,`ky` — a real wound/stacked cell's through-plane
bottleneck), **bottom-plate cooling barely responds to `h`** (all six bottom designs land within 0.01 K
of each other, ~308.0 K peak) because internal conduction, not the boundary, is the bottleneck, while
**side-plate cooling scales strongly with `h`** — the best design (`h`=500 W/(m²K), side) reaches the
*lowest* peak temperature (307.8 K) but the *largest* cell-to-cell spread (4.2 K, vs. 0.05 K for the
weakest bottom design) — a genuine peak-temperature-vs-balance trade-off, not a monotonic "more cooling
is better" result. `demo/out/design_sweep.csv` has the full 12-row table.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the shared contract: array layouts, `ElectrodeGeom`/
   `PackThermalParams`/`DesignPoint` structs, units and sign conventions. Read this FIRST — everything
   else assumes it.
2. [`src/kernels.cu`](src/kernels.cu) — the two GPU kernels: `electrochem_fv_kernel` (spherical FV
   diffusion) and `thermal_step_kernel` (anisotropic 3-D heat equation with the Robin boundary term).
   The most interesting kernel to read closely is `thermal_step_kernel`'s boundary-condition branch —
   it is the whole cooling-design sweep in about 15 lines.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — line-by-line CPU twins of both kernels.
4. [`src/main.cu`](src/main.cu) — the orchestration: scenario loading, the OCV/Butler-Volmer/voltage
   bookkeeping (`compute_bookkeeping`), the mission loop (`run_sweep_gpu`), the verify slice, the three
   analytic-gate fixtures, and the artifact writers. Long, but each stage is a clearly labeled section
   matching the file header's numbered pipeline.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and why they are copied, not shared.

## Prior art & further reading

- **PyBaMM** (Python Battery Mathematical Modelling) — the open-source reference for SPM/SPMe/P2D and
  the whole electrochemical ladder this project's SPM tier is the first rung of; study its model
  hierarchy documentation to see SPMe/P2D built out fully.
- **COMSOL Multiphysics — Battery Design Module** — the commercial standard for coupled
  electrochemical-thermal pack simulation (P2D + full 3-D thermal + mechanical), the production-grade
  version of exactly what this project teaches a reduced form of.
- **Newman's P2D model** (the original 1975 porous-electrode theory this whole ladder descends from) —
  study the porous-electrode + concentrated-solution-theory derivation this project's SPM deliberately
  simplifies away.
- **ANSYS Fluent / STAR-CCM+ battery pack thermal modules** — production CFD tools for exactly the
  32×24×16-voxel-scale (and far larger) pack thermal problem this project's `thermal_step_kernel` solves
  in miniature, with real turbulent coolant-flow CFD instead of a fixed convective coefficient `h`.
- **NREL's battery aging / thermal-runaway literature** — for where "cell-to-cell temperature spread"
  (this project's `max_spread` metric) actually matters: uneven aging and, at the extreme, cascading
  thermal runaway.

Study these; do not copy code wholesale — reimplement didactically and credit (CLAUDE.md §4.1).

## Exercises

1. **Sweep the mission duration.** Extend `MISSION`'s `duration_s` to 3600 s (1 hour) and re-run. Does
   bottom-plate cooling start responding to `h` once the mission is long enough to approach the pack's
   thermal time constant? (THEORY.md "Numerical considerations" discusses why 1200 s is short relative
   to that time constant.)
2. **Add a third cooling-face option.** Extend `CoolFace` (kernels.cuh) with `kCoolTopZ`, update both
   kernels' boundary branch and `main.cu`'s `build_designs`, and re-sweep — does cooling the *far* side
   from the heat-generating cells behave differently from cooling the *near* side?
3. **Warm-start the sweep.** Currently every design starts from the identical `T_init`/`c0_frac` state.
   Modify `run_sweep_gpu` to start from a previous run's final state (e.g., simulate a second mission
   back-to-back) and see how designs that ran hot compound their disadvantage.
4. **Profile the per-step host↔device traffic.** `run_sweep_gpu`'s comment names a gather-kernel
   alternative to the full-array temperature copy-back every step — implement `extract_cell_temps` as a
   device kernel instead, and measure the wall-time change (README's `[time]` lines already report the
   baseline).
5. **Implement SPMe's electrolyte-concentration state** (documented in THEORY.md "Where this sits in
   the real world"): add a second, 1-D-in-x diffusion PDE for electrolyte lithium-ion concentration, and
   make the exchange-current-density prefactor depend on it instead of the constant assumed here — the
   next rung of the ladder this project's README §1 names but does not implement.

## Limitations & honesty

- **Electrochemical ladder scope:** this project implements the **SPM** tier only. SPMe and full P2D
  (both named in the catalog bullet) are documented — governing equations and where SPM breaks down —
  in [`THEORY.md`](THEORY.md) "Where this sits in the real world", not implemented; Exercise 5 is a
  concrete on-ramp to SPMe.
- **All electrochemical/thermal parameters are synthetic** (see [Data](#data)) — plausible orders of
  magnitude, never claimed to match a real cell, pack, or dataset.
- **Shared pack current.** Every cell sees the identical commanded current at every instant (a
  documented simplification — a series pack under simple, non-actively-balancing BMS control); only
  each cell's local temperature, not its current, differs — see `kernels.cuh`'s header comment.
- **One representative particle per electrode per cell**, not a real electrode's billions of particles —
  the SPM's own foundational assumption, discussed in depth in THEORY.md "The problem".
- **Reversible (entropic) heat is omitted**; only irreversible ohmic + activation heat is modeled
  (`compute_bookkeeping` in `main.cu` documents this explicitly).
- **kNShells=20 introduces a measured ~12% discretization bias** in the diffusion analytic gate — a
  real, understood, shell-count-limited numerical effect (shrinks to ~6% at 40 shells), not a bug; the
  gate's tolerance is set from the measurement, not guessed (`main.cu`'s `run_diffusion_fixture`
  documents the full measured convergence table).
- **A 20-minute mission is short relative to the pack's own thermal time constant** at this project's
  thermal mass — which is *why* bottom-plate cooling barely differentiates across `h` values (see
  [Expected output](#expected-output) and Exercise 1), not a bug in the sweep.
- **No safety-hardware claims of any kind.** This is a **design-time simulation tool** — it never
  commands current into, or reads sensors from, real hardware, and its output is not a BMS runtime
  estimate (25.02, the sibling project, is where a runtime SOC/SOH estimator belongs). Nothing here is
  a substitute for UN 38.3 transport testing, UL 1642/2054, or IEC 62133 cell/pack safety certification
  — see [`PRACTICE.md`](PRACTICE.md) §4 for the regulatory orientation.
