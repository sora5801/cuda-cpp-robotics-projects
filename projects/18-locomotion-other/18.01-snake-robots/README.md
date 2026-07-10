# 18.01 — Snake robots: serpenoid gait sweeps coupled to granular sim

**Difficulty:** intermediate · **Domain:** 18. Locomotion — Everything Else

> Catalog bullet (source of truth, verbatim): `Snake robots: serpenoid gait sweeps coupled to granular sim`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A snake robot moves by bending its body into a traveling wave — Hirose's **serpenoid curve** — and
letting **anisotropic friction** with the ground turn that side-to-side wiggle into forward thrust.
This project builds a GPU **design-space sweep**: for a planar 12-link snake on flat ground, it
simulates **8,192 candidate gaits** (every combination of wave amplitude `A`, inter-joint phase
offset `beta`, and temporal frequency `omega` on a 32×32×8 grid), each for 8 simulated seconds, and
finds the fastest one. One GPU thread simulates one gait end to end — the same "thread = one
independent trajectory" pattern as 08.01's MPPI rollouts and 10.03's environment farm, here applied
to searching a gait's parameters instead of a control sequence or a training curriculum.

**What is implemented vs. documented-only** (the catalog bullet bundles two ideas — CLAUDE.md §2):
serpenoid gait sweeps on a rigid, flat, anisotropic-friction ground model are **fully implemented and
GPU-accelerated** here. The bullet's second half — coupling to a full **granular (DEM) simulation**
of the ground itself — is **documented only**: THEORY.md §"Where this sits in the real world" explains
what that coupling would add and points at project 10.10 (this repo's DEM/granular-media flagship),
which this project's friction law is a standard, published *reduction* of (CLAUDE.md §13's "simplest
correct teaching version" rule). Measured on the reference machine (RTX 2080 SUPER): the whole
8,192-gait sweep runs in **~80 ms** of GPU kernel time.

## What this computes & why the GPU helps

Each of the 8,192 gaits is an independent 8,000-step (8 s @ 1 ms) time integration of a 3-degree-of-
freedom rigid-body model (the snake's head pose) driven by a 12-link forward-kinematics pass every
step. **Pattern: batched simulation (a parameter-space "map")** — one thread per gait, zero
communication between threads, exactly like a Monte-Carlo sweep. Nothing here is memory-bandwidth
bound (unlike SAXPY): every gait's inputs are four scalars and its state lives entirely in registers,
so the kernel is **compute-bound on transcendental throughput** — sinf/cosf calls dominate the inner
loop (THEORY.md §The GPU mapping counts them: ~46 calls/step × 8,000 steps × 8,192 gaits ≈ 3 billion
transcendental evaluations, done in ~80 ms).

- **Pattern:** thread-per-independent-simulation batched sweep (08.01/10.03's family, applied to
  gait-parameter search instead of control-sequence sampling or curriculum training).
- **Measured reality:** 8,192 gaits × 8,000 steps run in ~80 ms of GPU kernel time (Release,
  RTX 2080 SUPER) — roughly 800 million gait-steps/second. A single-threaded CPU oracle takes ~34 ms
  for just 32 of those gaits (the §5 verification subset), i.e. the full sweep would cost roughly
  8.7 seconds sequentially — a ~110x teaching-artifact speed-up from turning 8,192 independent
  trajectories into 8,192 independent threads.
- **Why not memory-bound:** each thread's "input" is four floats (amplitude, phase, frequency, plus
  the shared friction pair) and its "output" is six floats; there is no large array to stream — the
  bottleneck is arithmetic (mostly `sinf`/`cosf`), not bytes moved.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** **gait design / offline planning**, not a real-time control-loop box. A gait
  sweep like this one runs at design time or, at best, at the **global-planner band**
  (SYSTEM_DESIGN §1.1: 0.1–1 Hz or event-driven "pick a new gait") — never inside the 10–50 Hz local
  planner or the 0.5–1 kHz whole-body control loop that actually executes it.
- **Upstream inputs:** a *task/terrain specification* (flat ground, friction estimate — here a
  scenario file; on a real robot, terrain classification from perception, domain 01/02) and a *gait
  library or teleop command* choosing which region of gait-space to search or select from — the
  message-shaped analogue is closest to a `nav_msgs/Path`-style "go this way" intent, several levels
  above `JointState`.
- **Downstream consumers:** a **joint-space trajectory generator / servo controller**: the winning
  `(A, beta, omega, gamma)` becomes a per-joint reference angle stream `phi_j(t)`, published at the
  servo loop's rate (message shape: `sensor_msgs/JointState`-like position/velocity references, one
  per actuated joint) and tracked by each joint's local PID/current-loop hardware.
- **Rate / latency budget:** the sweep itself is **not** latency-critical (it can run once, offline,
  or occasionally online for terrain re-adaptation — well under the global-planner row). What *is*
  latency-critical, and entirely out of this project's scope, is the **downstream joint execution**:
  real snake-robot joint controllers close their position loops at hundreds of Hz to kHz
  (SYSTEM_DESIGN §1.1's whole-body-control / motor-current-loop rows) — this project assumes that
  tracking is perfect (THEORY.md §Numerical considerations) precisely because that loop is a separate,
  much faster problem this project does not model.
- **Reference robot(s):** **none of the five** SYSTEM_DESIGN reference robots (AMR, manipulator cell,
  quadruped, quadrotor, AV stack) — the snake is its own archetype, closest in spirit to
  SYSTEM_DESIGN §2.3's quadruped (both are "many actuated joints, ground-contact-driven locomotion")
  but with **zero unactuated compliant legs** and continuous body contact instead of discrete
  footholds. Real applications: **pipe and duct inspection** (oil & gas, water/sewer utilities,
  nuclear facility piping), **search-and-rescue** in rubble (confined, irregular voids no wheeled/
  legged robot fits through), and **minimally-invasive surgical/endoscopic** continuum devices (a
  distant cousin sharing the "long, thin, many-joint" body plan, though very different actuation and
  scale — see PRACTICE.md).
- **In production:** a real snake-robot stack replaces this offline sweep with either (a) a
  **precomputed gait table** indexed by estimated terrain friction, looked up in milliseconds, or
  (b) an **online gait optimizer** (CMA-ES, Bayesian optimization, or a learned policy) running on
  measured terrain feedback — this project's brute-force grid sweep is the pedagogical ancestor of
  both, and the GPU pattern (many independent trajectory evaluations in parallel) is exactly what a
  production online optimizer would also want.
- **Owning team:** controls/autonomy (SYSTEM_DESIGN §5.1), specifically whoever owns "locomotion" for
  a non-standard morphology — on a small team this is the same mechanical+controls engineers who
  built the robot; adjacent teams are embedded/firmware (who own the joint servo loop this project's
  output feeds) and mechanical engineering (who own the friction-determining surface of the physical
  snake, PRACTICE.md §1).

## The algorithm in brief

- **Hirose's serpenoid curve** — `phi_j(t) = A*sin(omega*t + j*beta) + gamma` prescribes every
  joint's angle exactly, as a known function of time; no joint dynamics are solved. →
  [THEORY.md](THEORY.md) §The math.
- **Anisotropic Coulomb friction** — each link resists sliding ALONG its own axis with a LOW
  coefficient (`mu_t`) and sliding ACROSS its axis with a HIGH coefficient (`mu_n`); this asymmetry
  is proven (and measured, via the isotropic-friction gate) to be the one ingredient a symmetric wiggle
  needs to produce net thrust. → THEORY §The problem — physics & engineering first.
- **The prescribed-joint 3-DOF reduction** — with the shape known, only the head's pose
  `(x, y, yaw)` is dynamic: Newton's law for the whole snake (constant mass + a nominal moment of
  inertia), semi-implicit Euler at `dt = 1 ms`. → THEORY §The algorithm.
- **The GPU sweep** — one thread per `(A, beta, omega)` grid point (32×32×8 = 8,192), each thread
  running its own 8,000-step integration and reducing it to speed/straightness/cost-of-transport. →
  THEORY §The GPU mapping.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/snake-robots.sln`](build/snake-robots.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/snake-robots.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. This
project deliberately calls `sinf`/`cosf` as two separate calls rather than the combined `sincosf`
intrinsic, specifically because `sincosf` is a CUDA/POSIX extension with **no MSVC host-CRT
equivalent** — `src/reference_cpu.cpp` must compile under plain `cl.exe` (see `src/kernels.cuh`'s
`link_friction_force` comment).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at, including how to plot the sweep surface.

## Data

The committed sample, `data/sample/snake_scenario.csv` (~0.9 KiB, synthetic, no RNG — this project's
physics has no randomness anywhere, CLAUDE.md §8), is a **task definition**, not a recording: the
snake's geometry (12 links × 0.10 m), the ground's anisotropic friction coefficients
(`mu_t=0.10`, `mu_n=0.70`), the 3-D sweep grid's shape and ranges, and the per-gait simulation
duration (8 s @ 1 ms). Regenerate it with `python scripts/make_synthetic.py` (deterministic — the
same command always produces the same bytes). Full field-by-field documentation, units, and the
committed file's SHA-256: [`data/README.md`](data/README.md).

## Expected output

Twelve stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, four `GATE_*:` lines, three
`ARTIFACT:` lines, and `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). **Two independent kinds of verification**:

1. **The §5 GPU-vs-CPU gate (`VERIFY:`):** 32 of the sweep's own 8,192 gaits (stride-sampled across
   the whole grid) are recomputed from scratch, sequentially, on the CPU, using the exact same
   `snake_step()` physics (`src/kernels.cuh`) the GPU kernel calls. Measured worst-case final-position
   deviation after 8,000 chained FP32 steps: **1.371e-06 m**, against a documented tolerance of
   **1.0e-03 m** (~700x margin — see THEORY.md §Numerical considerations for why the residual gap is
   `sinf`/`cosf` implementation differences, not a logic bug).
2. **Four physics gates**, each a small CPU-only diagnostic simulation:
   - `GATE_ZERO_AMPLITUDE`: a gait with `A=0` must produce **exactly 0.0 m** displacement (measured:
     `0.000e+00 m` — this one is not approximate, THEORY.md proves it from the equations themselves).
   - `GATE_ISOTROPIC_FRICTION`: the fastest discovered gait, re-run with `mu_t=mu_n` (friction made
     isotropic), must propel far more weakly. Measured: **0.0339 m/s vs. 0.5382 m/s anisotropic**
     (6.3% of the anisotropic speed, against a documented 20% bound) — the anisotropy-necessity
     theorem, measured.
   - `GATE_TURNING_BIAS`: adding a turning bias `gamma` at a representative mid-range gait must shift
     the final heading by a measurable, documented-sign amount. Measured: **+1.014 rad** shift for
     `gamma=+0.15 rad` (bound: `>= +0.05 rad`).
   - `GATE_AMPLITUDE_RIDGE`: at the best gait's `beta`/`omega`, speed vs. amplitude must peak in the
     *interior* of the swept range, not at either boundary — the classic serpenoid result. Measured:
     `0.0048 m/s` (min amplitude) `< 0.5382 m/s` (peak) `> 0.0460 m/s` (max amplitude) — the peak
     clears both edges by well over the documented 15% margin.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **start here.** Every physical constant, the gait
   parameterization, and — the heart of the whole project — `snake_step()`, the `__host__ __device__`
   function that IS the physics (forward kinematics, anisotropic friction, the 3-DOF Newton-Euler
   update). Read `link_friction_force()` first, then `snake_step()`, then `simulate_gait()`.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernel, deliberately thin: decode a gait index, call
   `simulate_gait()`, write six floats. The thinness IS the point (see the file's header comment).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the CPU oracle for the §5 gate, plus every
   single-gait diagnostic (zero-amplitude, isotropic-friction, turning-bias, the best-gait trajectory
   logger) — all built on the SAME shared `snake_step()`, never a hand-copied twin.
4. [`src/main.cu`](src/main.cu) — orchestration: load the scenario, launch the sweep, run the §5 gate,
   find the fastest gait, run the four physics gates, write the three artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — CLAUDE.md §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Hirose, S. (1993), *Biologically Inspired Robots: Snake-Like Locomotors and Manipulators*** — the
  serpenoid curve's origin; the book that turned "how do snakes move" into "how do we build a robot
  that moves that way." This project's `phi_j(t)` formula is Hirose's, unchanged.
- **Gray, J. (1946), "The mechanism of locomotion in snakes"** — the biomechanical case for
  anisotropic (scale-driven) ground friction as the propulsion mechanism; the biology this project's
  `link_friction_force()` idealizes into a Coulomb law.
- **Hu, D. L., Nirody, J., Scott, T., Shelley, M. J. (2009), "The mechanics of slithering locomotion"**
  — resistive-force-theory analysis of snake propulsion (the same friction-decomposition family this
  project uses), with the friction-ratio-dependent speed result THEORY.md's "where this sits in the
  real world" section cites.
- **CMU Biorobotics Lab's modsnake / HEBI Robotics' snake-arm products** — real serial-elastic-actuator
  snake robots; PRACTICE.md's hardware section names their actuation approach.
- **Transeth, A. A., Pettersen, K. Y., Liljebäck, P. (2009), "A survey on snake robot modeling and
  locomotion"** — a broad, more rigorous survey of exactly the modeling choices this project makes a
  teaching-scoped version of (including full multibody dynamics vs. this project's prescribed-joint
  reduction).
- **Project 10.10 (this repo)** — full DEM/granular-media simulation; the "coupled to granular sim"
  half of this project's catalog bullet that is documented, not implemented, here (THEORY.md's real-
  world section explains the gap precisely).

## Exercises

1. **Plot the artifacts:** `demo/out/sweep_surface.csv` → a heatmap or 3-D surface of speed vs.
   `(A, beta)`; `demo/out/best_gait_path.csv` → the winning gait's head trajectory. `sweep_surface.pgm`
   is already a viewable (if low-res) heatmap — open it in any image viewer.
2. **Chase the beta boundary:** the measured best `beta` sits at the LOW edge of the swept range
   (`beta_min = 0.10 rad` — see README §Limitations). Lower `BETA_MIN_R` in
   `data/sample/snake_scenario.csv` (or regenerate with `--beta-min-r`) and see whether an even
   longer-wavelength gait wins, and where the trend finally turns over.
3. **Straighten the winner:** the fastest gait is not the straightest (`straightness=0.752`). Modify
   `main.cu`'s argmax to instead maximize `speed * straightness`, or add a straightness floor, and
   compare the gait it finds against the pure-speed winner.
4. **Break the friction ratio:** sweep `MU_N/MU_T` itself (regenerate several scenarios with different
   `--mu-t`/`--mu-n`) and plot best-speed vs. anisotropy ratio — at what ratio does propulsion
   effectively vanish?
5. **Lift the register footprint:** `kNLinks` is a compile-time constant (`src/kernels.cuh`'s file
   header explains why). Investigate what it would take to make link count a runtime scenario field —
   dynamic shared/global scratch per thread, or a second kernel variant templated on link count.

## Limitations & honesty

- **Rigid-body, flat-ground, quasi-static-leaning approximation.** This is the standard, published
  reduced-scope teaching model for lateral undulation (resistive-force-theory-style anisotropic
  Coulomb friction) — not a full rigid multibody simulation (Featherstone-class dynamics with a real
  mass matrix) and not coupled to any deformable/granular ground model. Full DEM coupling is project
  10.10's territory (README §Prior art); THEORY.md's real-world section states exactly what that would
  add.
- **Perfect joint tracking.** Every joint is assumed to hit its prescribed serpenoid angle exactly,
  every instant — real servo joints track with some lag/error. This is what makes the "only 3 DOF are
  dynamic" reduction tractable at all (THEORY.md §The algorithm); a real robot's joint-tracking error
  would feed back into the friction forces this project treats as exact.
- **Constant nominal moment of inertia.** The head's rotational dynamics use `I_eff` computed once
  from the snake's STRAIGHT-line configuration, not recomputed from the instantaneous bent shape —
  documented in `src/kernels.cuh`'s `snake_step()` comment; THEORY.md's numerics section discusses the
  error this costs.
- **The fastest gait found is not the straightest.** The sweep's speed-maximizing gait measured
  `straightness = 0.752` (its path is ~33% longer than its net displacement) and drifts its own heading
  by ~1.7 rad over the 8 s run even with no turning bias (`gamma=0`) — a genuine, measured result of
  optimizing pure speed with an asymmetric starting phase, not a bug (the §5 gate independently
  confirms GPU and CPU agree on this trajectory to 1.4 microns). See Exercise 3.
- **The measured best `beta` sits at the sweep's own lower boundary** (`beta_min = 0.10 rad`), meaning
  the true (unconstrained) optimum may lie at an even longer wavelength than this project's grid
  tested — reported honestly rather than narrowed after the fact; see Exercise 2.
- **Cost-of-transport is a teaching-scale proxy**, not a calibrated robot's measured COT: the joint-
  torque estimate feeding it is itself an approximation (a free-body "cut" argument that neglects
  sub-chain inertia — THEORY.md derives and names this simplification explicitly).
- **All timings are teaching artifacts**, single-shot, one machine (RTX 2080 SUPER), never a benchmark
  claim (CLAUDE.md §12).
- **Sim-validated only (CLAUDE.md §1):** this project's output (a winning gait's `A, beta, omega,
  gamma`) could, in principle, be handed to a real snake robot's joint controllers as a reference
  trajectory. Nothing here has been validated against real hardware, real ground friction, or a real
  actuation chain — see PRACTICE.md §3 for the testing ladder any such use would require.
