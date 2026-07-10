# 31.01 — Hamilton-Jacobi reachability: level-set grid solvers (stencil ops — GPU-perfect)

**Difficulty:** ★ beginner · **Domain:** 31. Safety, Verification & Testing

> Catalog bullet (source of truth, verbatim): `★ Hamilton-Jacobi reachability: level-set grid solvers (stencil ops — GPU-perfect)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**Reachability answers the safety question exhaustively: from which states can the robot still
reach a target within T seconds, over ALL possible control choices — not just the ones a sampled
test happened to try?** This project computes that set for the simplest robot dynamics that still
teaches every real lesson: the **double integrator** (position `x`, velocity `v`, acceleration
command `u` with `|u| <= umax`) — the standard local model for one axis of almost anything that
moves (a wheel, a joint, a rotor thrust direction). It solves the Hamilton-Jacobi partial
differential equation that governs the **backward reachable tube (BRT)** on a 256x256 grid over
state space, backward in time from a target set, using a Lax-Friedrichs numerical Hamiltonian with
upwinding and an explicit CFL-limited "sweep" — one GPU thread per grid cell per sweep, ping-pong
buffers, the exact stencil pattern taught in 07.09 with real PDE math in the stencil body. The
**zero sublevel set** of the resulting value field `V(x,v)` is the reachable set: `V <= 0` means
"some control gets you to the target within the horizon."

What makes this project special, and the reason it is featured as a flagship: the double
integrator's minimum-time-to-origin problem has a **closed-form answer** (the textbook bang-bang
solution, one switch on a parabolic switching curve). The target set is deliberately chosen as a
sublevel set of that closed-form time-to-origin function, `T*(x,v)`, rather than an arbitrary box —
which means the *entire* resulting value field also has a closed form
(`V_exact = max(T*(x,v) - (t0+T), -t0)`, derived in `THEORY.md`). The demo therefore checks the
GPU's numeric answer against **pure mathematics**, cell by cell, not just against another program.
This project is implemented in full: the GPU solver, the CPU twin, the analytic oracle, and the
demo pipeline (scenario load -> solve -> GPU/CPU twin check -> analytic check -> PGM/CSV artifacts)
are all real, nothing here is a documented-only stub.

## What this computes & why the GPU helps

Per sweep: 256x256 = 65,536 independent cell updates, each combining its own value with its four
face neighbors through a small closed-form expression (a handful of divides, one `fabsf`, one
`fminf`) — no cell needs to know about any other cell **beyond its immediate neighborhood**, and no
cell needs information from a *later* point in the sweep. That is the canonical **stencil**
pattern: one thread per output cell, reading a fixed 5-point neighborhood, writing one output.

- **Pattern:** stencil (map-with-neighbors) — the same shape as a Jacobi iteration or 07.09's jump
  flood, here carrying real PDE math (a numerical Hamiltonian + artificial dissipation) instead of
  a distance-field update.
- **Why the GPU helps:** ~100 explicit sweeps for this scenario, each touching every one of 65,536
  cells independently — a textbook data-parallel workload with a small, uniform, coalesced memory
  footprint (see `src/kernels.cu` for why `i`, the position axis, is the fast/contiguous axis).
- **Measured reality (RTX 2080 SUPER, this scenario):** ~50–70 ms on one CPU core vs. ~1–3 ms of
  GPU kernel time (~20–40x, teaching artifact — see [Limitations](#limitations--honesty)).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **safety monitor**, a cross-cutting layer that sits *beside* the
  perception -> planning -> control pipeline, not inside it (SYSTEM_DESIGN §1's "cross-cutting"
  row). It runs in two phases: **offline**, compute the safe set once (this demo); **online**,
  cheaply check the current state estimate against the precomputed field every tick.
- **Upstream inputs:** a validated **system dynamics model** (here, the double-integrator bound
  `|u| <= umax` — one axis' worth of any real robot's actuator limits) and the live **state
  estimate** (message shape: `JointState`/`Odometry`-like `(x, v)`, per SYSTEM_DESIGN §3.6).
- **Downstream consumers:** a **safety filter or monitor** that reads the precomputed value field
  and either vetoes/overrides a controller's command (the CBF-filter pattern, catalog 31.04) or
  raises an alarm/fallback when the state is about to leave the safe set. 08.01's MPPI controller
  is the natural guarded client — a sampling controller commanding force/torque is exactly the kind
  of output this monitor would watch, per Chain A in `SYSTEM_DESIGN.md` §4.1.
- **Rate / latency budget:** the **offline solve** (this demo) is not rate-constrained — it runs
  once, ahead of time, in a few milliseconds of GPU time for this grid. The **online check** (is
  the current state inside the precomputed `V <= 0` set?) is a single array lookup and must keep up
  with the state estimator, 100–400 Hz (SYSTEM_DESIGN §1.1); at 1 kHz control-loop rates the check
  still fits comfortably because it does no PDE work at runtime, only a memory read.
- **Reference robot(s):** the **warehouse AMR** (§2.1) — Chain A in `SYSTEM_DESIGN.md` §4.1 places
  "31.01 HJ reachability" explicitly as the safety monitor watching the state estimator alongside
  the mapping/planning/control spine — and the **quadrotor** (§2.4), where a reachability-style
  safe set (geofence, battery-return envelope, attitude recovery bound) plays the same watchdog
  role beside the 0.5–1 kHz flight controller.
- **In production:** the open-source `hj_reachability` (JAX) and `OptimizedDP` toolboxes, and
  Mitchell's original `helperOC`/ToolboxLS (MATLAB), compute exactly this kind of value function at
  higher dimension and with more numerical machinery (WENO, adaptive grids); production safety
  filters more often use **control barrier functions (CBFs)** evaluated at kHz (catalog 31.04) as
  the cheap online cousin of an offline-computed reachable set — see README §11.
- **Owning team:** built and owned by **QA & functional safety** (SYSTEM_DESIGN §5.1's row for
  domain 31), working alongside **controls & autonomy** (who own the dynamics model and the
  controller being watched) — the safety monitor's whole job is to be a second, independently
  authored opinion the controls team's code does not get to override.

## The algorithm in brief

- **Backward reachable tube (BRT) via the level-set method** — encode the target as the zero
  sublevel set of a value function, solve a Hamilton-Jacobi PDE backward in time, read the answer
  off the sign of the result. -> [THEORY.md §the-math](THEORY.md#the-math)
- **Closed-form minimum-time double-integrator dynamics** — the target set is a sublevel set of the
  bang-bang minimum-time-to-origin function `T*(x,v)`, which is *also* the analytic oracle the
  numeric answer is checked against. -> [THEORY.md §the-algorithm](THEORY.md#the-algorithm)
- **Local Lax-Friedrichs (LxF) numerical Hamiltonian + upwinding** — one-sided difference pairs per
  axis, the Hamiltonian evaluated at their average, plus an artificial-dissipation term sized
  exactly to the Hamiltonian's steepest slope in each axis. -> [THEORY.md §gpu-mapping](THEORY.md#the-gpu-mapping)
- **Freezing (`min` with 0)** — the value may only decrease sweep-to-sweep, turning "reach at
  exactly time T" into "reach within T," the tube robotics actually needs.
  -> [THEORY.md §the-math](THEORY.md#the-math)
- **Explicit, CFL-limited timestepping** — `dt` is not a tuning knob; it is bounded by how fast
  information can cross one grid cell. -> [THEORY.md §numerical-considerations](THEORY.md#numerical-considerations)
- **Two-stage verification** — the repo's standard GPU-vs-CPU twin gate, *plus* a check against the
  closed-form bang-bang solution (the feature of this project).
  -> [THEORY.md §how-we-verify-correctness](THEORY.md#how-we-verify-correctness)

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/hamilton-jacobi-reachability.sln`](build/hamilton-jacobi-reachability.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/hamilton-jacobi-reachability.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
No cuBLAS/cuFFT/Thrust/CUB; the whole solver is one hand-written stencil kernel.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is a **scenario**, not recordings: `data/sample/double_integrator_scenario.csv`
(~0.8 KiB, synthetic, no RNG) — the grid geometry, the acceleration bound, the target level, and the
reachability horizon. Everything else (the initial level function, the PDE sweeps, the analytic
oracle) is computed inside the demo from that scenario plus closed-form mathematics; there is no
public dataset to fetch (`scripts/download_data.ps1`/`.sh` are honest no-ops). Details and the
exact reasoning behind the committed horizon: [`data/README.md`](data/README.md).

## Expected output

Two independent verification stages, both required for `RESULT: PASS`:

1. **VERIFY (the repo-standard §5 gate):** the GPU value field and a plain-C++ CPU twin, run from
   the *same* initial condition with the *identical* per-cell update expression, must agree within
   `max |V_gpu - V_cpu| <= 1e-3` over the whole 65,536-cell field. Measured worst case on this
   scenario: `~1.7e-5` — three orders of magnitude inside tolerance; the two paths differ only in
   FP32 rounding order.
2. **ANALYTIC (the check this project exists to feature):** every cell is classified `V <= 0`
   ("reachable") or not, and that classification must match the **closed-form** bang-bang
   minimum-time solution *exactly*, everywhere **except** a documented 3-grid-cell boundary band
   around the true boundary (a first-order numerical scheme cannot place a moving front more
   precisely than a few cells — `THEORY.md` measures this honestly rather than asserting it).
   Measured on this scenario: 0 disagreements outside the band, 230 (out of 1,556 band cells)
   inside it — verification against mathematics, not against another program.

Both checks, plus two written artifacts, are required for the demo to exit 0. The canonical stable
lines live in [`demo/expected_output.txt`](demo/expected_output.txt); `[info]`/`[time]` lines carry
the full numeric detail (worst error, tube size, sup-error) but are deliberately unchecked because
they are exact-but-machine-adjacent, not because they are unimportant — read them.

## Code tour

A guided reading order through `src/` (also the order `main.cu`'s own header comment recommends):

1. [`src/kernels.cuh`](src/kernels.cuh) — the shared contract: grid layout, the `HjGrid` struct,
   the sign convention for `V`, and `kBandCells`/`kTwinTol` with their measured justification.
2. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — `hj_solve_cpu` (the GPU kernel's line-by-line
   twin) and `min_time_to_origin` (the independent analytic oracle — read this to see the bang-bang
   switching-curve math made concrete).
3. [`src/kernels.cu`](src/kernels.cu) — the heart: `hj_sweep_kernel`, one thread per cell, the
   Lax-Friedrichs Hamiltonian + dissipation + freeze, and the ping-pong launcher.
4. [`src/main.cu`](src/main.cu) — the full pipeline: load scenario, build the initial level
   function from `min_time_to_origin`, solve on GPU and CPU, VERIFY, then the ANALYTIC stage (the
   most interesting code in this project — read the boundary-band computation closely), then write
   the PGM/CSV artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Mitchell, "A Toolbox of Level Set Methods" (2005) / `helperOC`, `ToolboxLS`** — the original
  MATLAB reachability toolbox this project's numerics (LxF Hamiltonian, freezing PDE) come from;
  the canonical reference for the level-set method taught here.
- **Bansal, Chen, Herbert, Tomlin, "Hamilton-Jacobi Reachability: A Brief Overview and Recent
  Advances" (2017)** — the clearest modern tutorial on the tube/BRT formulation this project
  implements; read it alongside `THEORY.md §the-math`.
- **`hj_reachability` (JAX, UC Berkeley HJ team) / `OptimizedDP`** — modern, GPU/vectorization-
  friendly reimplementations of the same PDE machinery, at higher dimension and with WENO-order
  schemes instead of this project's first-order LxF.
- **Catalog 31.04 CBF safety filters** — the cheap *online* cousin of an offline-computed reachable
  set: instead of a precomputed grid, a control barrier function bounds the safe set's boundary
  analytically and filters commands at kHz. Compare the two philosophies directly.
- **Catalog 34.06 "High-dimensional Hamilton-Jacobi via tensor decompositions [R&D]"** — where this
  exact PDE goes when the state dimension outgrows a dense grid (this project's [Limitations](#limitations--honesty)
  explains why 2-D is where a dense grid stops being practical).
- **Osher & Fedkiw, "Level Set Methods and Dynamic Implicit Surfaces"** — the numerical-methods
  textbook behind upwinding, Lax-Friedrichs, and the whole level-set toolkit used here.

## Exercises

1. **Read the field, not just the verdict.** Open `demo/out/value_function.pgm` in any image viewer
   that reads P5 PGM (or convert with any tool) and `demo/out/brs_boundary.csv` in a plotting
   library; overlay the CSV's `t_min_s` column against `t_target + horizon` and watch it hug a
   near-constant value along the boundary — *that* near-constancy is the analytic check, visible to
   the eye.
2. **Reproduce the horizon measurement yourself.** Edit `HORIZON` in
   `data/sample/double_integrator_scenario.csv` up toward 1.0 or 1.5 s, rebuild nothing (no C++
   changes needed), rerun the demo, and watch `ANALYTIC` fail with a growing
   `disagreements outside band` count. This is the exact experiment `THEORY.md §numerical-considerations`
   describes in words — now you have the numbers on your own GPU.
3. **Shrink `kBandCells` to 2** (the float64 reference measurement's exact answer — see
   `kernels.cuh`'s comment) at the committed 0.4 s horizon and watch `ANALYTIC` **fail** — real FP32
   rounding at the moving front costs one more cell of margin than an idealized float64 model
   predicts. `kBandCells=3` (the shipped value) is the smallest value that actually passes on this
   exact scenario; you have now reproduced that measurement yourself instead of taking it on faith.
4. **Add a shared-memory tile** to `hj_sweep_kernel` (each interior value is currently re-read by up
   to 4 neighbor threads straight from global memory/L2) and measure whether it moves the needle at
   this problem size — the honest 07.09 exercise, repeated here with real PDE math in the kernel.
5. **Extend to a literal box target.** Replace the min-time-sublevel-set initial condition with a
   simple axis-aligned box `l0(x,v) = max(|x|-x0, |v|-v0)` and derive (or numerically bound) the
   corresponding closed-form minimum-time-to-box solution for the analytic check — a genuinely
   harder verification problem than this project's point-target case.

## Limitations & honesty

- **Horizon reduced from the originally-scoped ~1.5 s to a measured 0.4 s.** First-order explicit
  Lax-Friedrichs dissipation compounds with every sweep and does **not** shrink by refining the
  timestep (verified: 408 -> 10,200 sweeps over the same horizon changed the boundary-band
  requirement by under 1%) — it is a genuine, textbook property of long-time integration with a
  first-order scheme, not a bug worked around here. At the originally-floated 1.5 s horizon this
  same scheme needs a ~13-cell excused band on this grid, which would have made the analytic
  check's whole point ("verification against mathematics, not fudge factors") ring hollow. The
  committed 0.4 s horizon (total elapsed-time budget `t0+T = 1.0 s`) was chosen by direct
  measurement to keep the excused band at 3 cells — see `THEORY.md §numerical-considerations` for
  the full sweep-count-vs-band-width table and `data/README.md`/`scripts/make_synthetic.py` for the
  scenario-level documentation of this decision.
- **Target set is a min-time sublevel set, not a literal box.** This is a deliberate choice (not a
  scope deviation dressed up): it makes the *entire* value field, not just probe points, checkable
  against a closed form, which is a strictly stronger verification than a box target would allow.
  Exercise 5 sketches the box-target alternative.
- **Curse of dimensionality, stated honestly.** This project's dense 256x256 grid is *only*
  practical because the double integrator is 2-D. Reachability's core numerical cost is
  exponential in state dimension (a dense grid at this resolution in 6-D would need ~256^6 cells —
  computationally absurd on any hardware). Real systems (a quadrotor, a manipulator) need either
  dimension reduction (decoupled axes, as `THEORY.md` discusses), or the research-frontier
  techniques of catalog 34.06 (tensor decompositions) and neural/deep reachability methods — this
  project teaches the 2-D core, honestly, not a false promise of scaling.
- **First-order accuracy, on purpose.** This project uses local Lax-Friedrichs, not the
  higher-order WENO schemes production toolboxes (`hj_reachability`, `helperOC`) use — a deliberate
  teaching trade (CLAUDE.md §1: "a slower/simpler kernel a learner can follow beats a faster one
  they cannot"), with the accuracy cost measured and documented rather than hidden.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), and vary run to
  run (CPU ~50–70 ms, GPU ~1–3 ms observed) with first-launch driver/JIT overhead unamortized;
  never a benchmark claim.
- **Sim-validated only, not safety-certified (CLAUDE.md §1, §8):** this project computes a
  safety-style reachability metric **didactically**. It is not a certified implementation of any
  safety standard, has not been validated against real sensor noise or model mismatch, and must
  never be treated as a substitute for a certified safety chain (hardwired E-stops, safety-rated
  monitors — see `PRACTICE.md §3`). Any real use of a reachability-style safety monitor demands the
  full testing ladder (simulation -> HIL -> bench -> free running) and an independently authored,
  professionally validated implementation.
