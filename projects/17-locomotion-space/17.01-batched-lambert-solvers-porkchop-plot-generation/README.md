# 17.01 — Batched Lambert solvers + porkchop plot generation

**Difficulty:** ★ beginner · **Domain:** 17. Locomotion — Space

> Catalog bullet (source of truth, verbatim): `★ Batched Lambert solvers + porkchop plot generation`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**A porkchop plot answers one question 262,144 times: "if I leave now and arrive later, what does it
cost?"** For every (departure epoch, arrival epoch) cell on a 512×512 grid, this project solves
Lambert's problem — given two positions and a flight time, find the orbit that connects them — between
two SYNTHETIC coplanar circular heliocentric orbits (an Earth-like body at 1 AU, a Mars-like body at
1.524 AU, canonical units where the Sun's `mu = 1`, no ephemeris data needed) and reports the total
impulsive delta-v of that transfer. Plotting delta-v over the grid produces the classic **porkchop
plot** mission designers use to pick a launch window — the same tool behind every "launch window"
headline for a Mars mission. The whole catalog bullet is implemented: the batched universal-variable
Lambert solver (Stumpff functions, a fixed-iteration bisection identical on GPU and CPU) AND the
porkchop plot artifact it produces. The short-way (Type I, transfer angle < 180°) branch is
implemented; the long-way (Type II) branch is documented as [Exercise 1](#exercises).

## What this computes & why the GPU helps

Per grid cell: classify the transfer geometry, then (for the ~18% of cells that are genuine
short-way, in-time-of-flight-band candidates) bisect 60 fixed iterations on the universal anomaly `z`,
each iteration evaluating the Stumpff functions `C(z)`, `S(z)` via `sin`/`cos`/`sinh`/`cosh` — a
transcendental root-find, not a closed-form formula.

- **Pattern:** batched solve — one thread per grid cell, exactly the "one thread owns one small
  numerical problem" pattern 33.01 teaches for batched linear algebra, here applied to a batched
  nonlinear root-find instead of a batched matrix solve. Cells share no data and write no shared
  output: 262,144 completely independent instances of the same equation.
- **Measured reality:** see the `[time]` line in your own run — this kernel does **zero global-memory
  reads** (every input is index-derived) and only two coalesced writes at the end, so unlike a
  bandwidth-bound map (SAXPY) it is purely **arithmetic-bound** (trig- and sqrt-heavy); the GPU wins
  by running 262,144 independent 60-iteration root-finds in parallel instead of one after another.
- **A genuine numerical surprise, not just a speed story:** the Lambert equations are mathematically
  singular at a transfer angle of exactly 180° — precisely where the Hohmann optimum sits. This
  project's NaN policy (kernels.cuh) turns that into a teaching feature rather than a bug report.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** upstream of the whole autonomy stack in §1's diagram — this is **mission
  design**, not a sense→plan→control loop. It answers "when do we leave and how much propellant does
  it cost," an input to every later phase, not a box inside the phase-to-phase pipeline. It is closest
  in spirit to the PLANNING layer's "global route" (§1: "global planner (route), 0.1–1 Hz, event
  driven") scaled up from minutes to months.
- **Upstream inputs:** mission goals and constraints (destination body, launch-vehicle capability,
  acceptable delta-v budget) — not sensor data; there is no `PointCloud` or `JointState` here. The
  "sensor" this project's real-world counterpart reads is an ephemeris (README §11, §Limitations).
- **Downstream consumers:** trajectory design and guidance, navigation & control (GN&C) — the chosen
  (departure epoch, arrival epoch, delta-v budget) feeds low-thrust or impulsive trajectory
  optimization (17.02+ in this domain), which in turn hands a reference trajectory to the spacecraft's
  onboard guidance and the flight team's orbit-determination/navigation loop.
- **Rate / latency budget:** **OFFLINE, by nature — say so honestly.** This is mission design, run
  once (or a handful of times) months to years before launch, not a control loop with a Hz figure
  (SYSTEM_DESIGN.md item 1's table has no row for this). The "budget" that matters is wall-clock
  turnaround for a mission designer iterating on launch-window trade studies — this demo's whole
  262,144-cell grid solves in low single-digit milliseconds of GPU kernel time (see your own `[time]`
  line), which is what makes an *interactive* porkchop-plot tool (drag the launch date, watch the
  contours update) possible at all.
- **Reference robot(s):** none of the five §2 reference robots directly (they are all *in-atmosphere
  or on-the-ground* autonomy archetypes) — this project's "robot" is the **spacecraft archetype**
  itself: an interplanetary probe or crewed vehicle, the machine every domain-17 project ultimately
  serves. `PRACTICE.md` grounds that archetype in real hardware and mission operations.
- **In production:** GMAT (NASA/AFRL, open source), poliastro / pykep (ESA-adjacent open-source
  astrodynamics libraries), and the mission-design tools built around Izzo's Lambert solver (the
  fast, robust, closed-form-where-possible algorithm most modern flight-dynamics software actually
  uses — README §11) all solve this exact problem, at production robustness and with real
  ephemerides.
- **Owning team:** mission design / flight dynamics, inside the broader controls & autonomy /
  GN&C organization (SYSTEM_DESIGN.md item 5) — adjacent to systems engineering (who set the
  delta-v budget this trade study serves) and propulsion (who owns the number this project outputs).

## The algorithm in brief

- **Circular-orbit state** — closed-form position/velocity for each body at any epoch (Kepler's
  third law gives the constant angular rate) — no integration needed. → [THEORY.md](THEORY.md) §the-math.
- **Universal-variable Lambert solver** — Stumpff functions `C(z)`, `S(z)` (with a Taylor-series
  switchover near `z=0`) let one set of equations cover elliptical/parabolic/hyperbolic transfers; a
  fixed-bracket, fixed-iteration bisection on `z` — IDENTICAL scheme on GPU and CPU, the §5 gate's
  precondition. → THEORY.md §the-algorithm, §numerical-considerations.
- **The NaN policy** — five cell-status outcomes (OK, masked time-of-flight, long-way scope
  exclusion, near-singular, non-converged), each counted and reported, never silently propagated.
  → THEORY.md §numerical-considerations.
- **The Hohmann ground truth** — an independent, closed-form vis-viva computation the grid's own
  minimum is checked against — verification against pure mathematics, not just against another piece
  of code. → THEORY.md §how-we-verify-correctness.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/batched-lambert-solvers-porkchop-plot-generation.sln`](build/batched-lambert-solvers-porkchop-plot-generation.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/batched-lambert-solvers-porkchop-plot-generation.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
Lambert solver (Stumpff functions + bisection) is hand-rolled by design (CLAUDE.md §1: this is exactly
the kind of small transcendental solve worth writing once, not the kind that justifies a dependency).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
**the porkchop picture to open**.

## Data

The committed sample is a **scenario**, not recordings: `data/sample/lambert_scenario.csv`
(771 bytes, synthetic, no RNG) — both orbit radii, the shared epoch window, the accepted
time-of-flight band, and the grid resolution. No public dataset applies here — this project's bodies
are synthetic circular orbits, not real ephemerides — so `scripts/download_data.ps1` is an honest
no-op (it also documents where real NASA JPL ephemerides live, for the honest out-of-scope next step).
Details: [`data/README.md`](data/README.md).

## Expected output

Nine stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY: PASS`, `NAN POLICY: PASS`,
`ANALYTIC: PASS`, `ARTIFACT:`, `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Three independent verifications:
**(1)** the §5 GPU-vs-CPU gate — every cell's status and, for OK cells, delta-v, computed by the
kernel and by [`src/reference_cpu.cpp`](src/reference_cpu.cpp) must agree (measured deviations on the
`[info]` line — orders of magnitude inside the documented bound); **(2)** the NaN-policy gate — the
fraction of attempted cells that are near-singular or non-converged stays small (measured on an
`[info]` line, well under the documented bound); **(3)** the ANALYTIC gate — the grid's own minimum
delta-v and time-of-flight land within a documented small window of the closed-form Hohmann optimum
(measured gap on an `[info]` line — this is verification against pure mathematics, not just
self-consistency). Success thresholds carry generous, documented margins over the measured values so
platform-level float rounding cannot flip the verdict (`src/main.cu`, `THEORY.md` §how-we-verify).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the whole project's contract in one file: canonical units
   (with the full SI conversion table derived in the header), the scenario layout, the five-way NaN
   policy, and the bisection's fixed algorithm constants. Read this first — it is unusually
   comment-dense on purpose, because every other file assumes it.
2. [`src/main.cu`](src/main.cu) — the driver: load the scenario, run GPU + CPU over the whole grid,
   VERIFY, the NaN-policy census, the ANALYTIC Hohmann check, then the artifact writers.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: `body_state` (closed-form circular orbits),
   `stumpff_c`/`stumpff_s` (with the series switchover), `solve_cell` (classify, then bisect), and the
   one-thread-per-cell kernel itself. The single most interesting thing: `solve_cell`'s classification
   order — read it alongside kernels.cuh's NaN-policy comment.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the line-by-line CPU twin; diff it against
   kernels.cu to see exactly what "GPU-vs-CPU identical algorithm" means in practice.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Izzo, D. (2015), "Revisiting Lambert's problem"** — the modern, fast, singularity-robust Lambert
  algorithm most production astrodynamics software actually uses (via a Householder iteration on a
  cleverly chosen variable, not Stumpff bisection); this project's universal-variable/bisection
  approach is the classic textbook method (Curtis, Vallado, Bate–Mueller–White) chosen here for its
  transparency, not its speed.
- **poliastro / pykep** — open-source Python astrodynamics libraries (ESA-adjacent lineage) that wrap
  Izzo's solver and real ephemerides into a porkchop-plot-ready API; compare their API shape to this
  project's `LambertScenario`.
- **GMAT (General Mission Analysis Tool, NASA/AFRL)** — the free, open-source, production mission-design
  tool used for real interplanetary trajectory work; its Lambert/porkchop tooling is this project's
  most direct professional counterpart.
- **NASA JPL Horizons / SPICE** — the real ephemeris sources (`scripts/download_data.ps1`'s DECISION
  note) that would replace this project's closed-form circular orbits with actual planetary positions.
- **Curtis, H., "Orbital Mechanics for Engineering Students"** — the textbook derivation of the
  universal-variable Lambert algorithm this project implements; THEORY.md follows its structure.

## Exercises

1. **The long way home.** Implement the Type II (long-way, transfer angle > 180°) branch — the
   status-2 (`kStatusLongWay`) cells this project currently excludes. Hint: the stable-`A` formula
   `A = sqrt(2 r1n r2n) cos(dtheta/2)` needs its sign flipped for the long way; work out why from
   THEORY.md §the-math and re-derive the Lagrange `f`/`g` recovery.
2. **Shrink the singular ring.** Halve `kEpsSingularRad` in `kernels.cuh`, rebuild, and re-run.
   Does the ANALYTIC gap to the Hohmann optimum shrink? Does the NaN-policy fraction change? Explain
   both from THEORY.md's numerics section.
3. **Sharpen the picture.** Edit `data/sample/lambert_scenario.csv`'s `GRID_N` to 1024 (regenerate
   with `python scripts/make_synthetic.py --grid-n 1024`, do not commit) and re-run. Measure how much
   closer the grid minimum gets to the Hohmann optimum — and how much longer the CPU reference takes.
4. **Real ephemerides.** Swap `body_state()`'s closed-form circular orbit for real Earth/Mars vectors
   from a JPL Horizons query at a few sample epochs (interpolate between them) — everything downstream
   of `body_state()` is unchanged; this is the honest next step `scripts/download_data.ps1` documents.
5. **Multi-revolution transfers.** This project's bracket (`kBisectZLo`/`kBisectZHi` in kernels.cuh)
   searches only the zero-revolution branch. Research what changes for `N`-revolution Lambert
   solutions (there are two per `N >= 1`) and sketch how the bracket and classification would need to
   change to add them.

## Limitations & honesty

- **Synthetic circular orbits, not real ephemerides.** Both bodies are idealized circles (Mars' real
  orbit is ~9.3% eccentric); the porkchop *shape* this produces is qualitatively right but the
  specific dates and delta-v numbers are not a real Earth–Mars mission's numbers. This is a documented,
  deliberate scoping choice (CLAUDE.md §2), not an oversight — see `scripts/download_data.ps1`.
- **Short-way (Type I) transfers only in v1.** The long-way (Type II) branch is excluded by scope
  (status `kStatusLongWay`), not by a solver limitation — [Exercise 1](#exercises) implements it.
- **Single-revolution transfers only.** No multi-revolution (N>0) Lambert branches — [Exercise 5].
- **Coplanar only.** Both orbits share the ecliptic-like plane by construction (z=0 always); a real
  interplanetary transfer usually needs a small plane-change component, which this project's 2D
  formulation cannot represent.
- **FP32 throughout the solver, FP64 for the independent Hohmann ground truth.** The two are
  deliberately different precisions and different code paths (`main.cu`'s `hohmann_ground_truth`) so
  the ANALYTIC check never shares rounding, let alone logic, with the thing it verifies.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **Not a control loop, not safety-relevant.** This project's output is a mission-design number
  (a launch-window recommendation), never a real-time actuation command — the repo-wide
  sim-validated-only caveat (CLAUDE.md §1) applies in its mildest form here: nothing in this project
  could move real hardware even indirectly without months of independent trajectory design,
  navigation, and operations work standing between this number and a spacecraft's engines.
