# 15.01 — Minimum-snap trajectory optimization batched over waypoint sets

**Difficulty:** ★ beginner · **Domain:** 15. Locomotion — Aerial

> Catalog bullet (source of truth, verbatim): `★ Minimum-snap trajectory optimization batched over waypoint sets`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A quadrotor asked to fly through 5 waypoints needs more than a path — it needs a **trajectory**: a
position *as a function of time*, smooth enough that the rotor thrusts it implies never spike or
reverse abruptly. **Minimum-snap** trajectories are the standard answer: pick, among all smooth
paths through the waypoints, the one that minimizes the integral of squared *snap* (the 4th time
derivative of position) — smooth by construction, and, as THEORY.md derives from first principles,
directly tied to how gently the rotor thrusts have to change. This project builds that trajectory
**batched**: 10,000 independent 5-waypoint sets, each turned into a fixed-size **32×32 linear
system per axis** (2 axes, 4 segments, 8 polynomial coefficients each) and solved with **one GPU
thread per waypoint set**, in-thread Gaussian elimination with partial pivoting — the same
thread-per-problem batched-solve pattern 33.01 teaches at N=3/4/6, scaled to a per-problem size (32)
that no longer fits in registers. The demo verifies every one of the 10,000 solved trajectories two
ways — against a CPU oracle, and against the mathematical definition of the constraints it was
solved for — and writes a dense-sampled, plottable trajectory (plus a small rasterized image) for
one hand-picked "slalom" waypoint set.

## What this computes & why the GPU helps

Per waypoint set: 2 axes × one 32×32 dense Gaussian elimination each (~O(32³/3) ≈ 11,000
multiply-adds) plus O(32²) back-substitution — roughly 25,000 flops of pure, **independent** linear
algebra. Across the default batch (K=10,000) that is ~250 million flops of embarrassingly parallel
work.

- **Pattern:** batched (small) linear solve — one thread = one whole waypoint set (both axes),
  entirely self-contained; zero interaction between threads, by construction (the repo's
  thread-per-problem pattern, same family as 33.01's Cholesky and 08.01's rollouts).
- **Measured reality:** the batch of 10,000 solves takes ~11–12 ms of GPU kernel time vs ~55–60 ms
  on one CPU core (single-shot, teaching-artifact numbers — see [Expected output](#expected-output)).
- **New relative to 33.01:** each system here is 32×32 = 1,024 floats — far past the ~255-register
  hardware ceiling, so it lives in per-thread **local memory**, not registers. `src/kernels.cu`'s
  file header explains why that is still the right design at batch scale, and THEORY.md §the-GPU-
  mapping contrasts it directly with 33.01's register-resident N=6 case.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **local planning / trajectory-generation** layer — the boundary between
  state estimation and control, quoted by name in SYSTEM_DESIGN.md §4.4's aerial chain ("Chain D"):
  `[04.07 VIO backend] → [15.01 minimum-snap trajectories] → [15.03/08.x MPPI/NMPC control]`.
- **Upstream inputs:** waypoints from a global/mission planner (domain 06, or a human operator /
  mission script) plus the *current* state estimate — a `T_world_base` pose and velocity, the shape
  SYSTEM_DESIGN.md §3.6's `Header`+pose structs describe — used to seed the first waypoint and, in a
  full implementation, the initial velocity/acceleration boundary conditions.
- **Downstream consumers:** a trajectory-TRACKING controller — differential-flatness-based
  geometric control, NMPC, or a sampling controller like 08.01's MPPI adapted to the quadrotor
  (SYSTEM_DESIGN.md names `15.03 MPPI/NMPC quadrotor control` as the immediate consumer) — which
  reads `(x(t), ẋ(t), ẍ(t))` off this project's polynomials at its own control rate and turns them
  into attitude/thrust setpoints via the differential-flatness map (THEORY.md §the-problem).
- **Rate / latency budget:** SYSTEM_DESIGN.md §1.1 places local-planner replanning at **10–50 Hz**
  (20–100 ms per replan) — this demo's *whole 10,000-trajectory batch* solves in ~12 ms of GPU
  kernel time, i.e. a single ONE-trajectory replan (this project's actual per-tick unit) is a
  negligible fraction of that budget; the batch size here is a teaching/statistics choice (verify
  10,000 independent instances at once), not a claim that a real flight computer replans 10,000
  trajectories per tick.
- **Reference robot(s):** the **quadrotor** (SYSTEM_DESIGN.md §2.4) — its block diagram names
  `TRAJECTORY PLANNING [15 →] minimum-snap / time-optimal, replan 10–50 Hz` explicitly, sitting
  between state estimation and flight control.
- **In production:** ethz-asl's `mav_trajectory_generation` and PX4's internal trajectory generators
  solve the SAME class of problem (often the true free-derivative QP — see
  [Limitations & honesty](#limitations--honesty)); the batched GPU angle here (many independent
  waypoint sets solved at once) is this project's own teaching extension for exploring/verifying
  planner behavior at scale, not something a single onboard flight computer needs — see README
  §11 for the production tools this teaches toward.
- **Owning team:** controls/autonomy (SYSTEM_DESIGN.md §5.1) — trajectory generation typically sits
  with the same team that owns state estimation and the tracking controller; adjacent teams include
  simulation (validates the trajectory against the vehicle's real dynamic limits) and, for anything
  that flies, flight-test/safety (PRACTICE.md §3–4).

## The algorithm in brief

- **Differential flatness** — a quadrotor's flat outputs are position and yaw; snap (the 4th
  derivative of position) maps directly to the rotor thrusts' rate of change, which is *why*
  minimizing it is the physically-motivated smoothness objective, not an arbitrary choice. →
  [THEORY.md](THEORY.md) §the-problem.
- **Per-axis, per-segment degree-7 polynomials** in NORMALIZED segment time `tau ∈ [0,1]` — 4
  segments × 8 coefficients = 32 unknowns per axis, per waypoint set. → THEORY §the-math.
- **A closed-form, fully-determined 32×32 linear system** — NOT a free-variable QP: position
  interpolation (8 eqns) + zero velocity/accel/jerk at the two flight endpoints (6 eqns) + interior
  continuity of velocity through **pop** (derivatives 1–6, 18 eqns) exactly saturates the 32 degrees
  of freedom. THEORY.md derives this count and is explicit about how it differs from Mellinger &
  Kumar's original free-derivative formulation. → THEORY §the-math.
- **In-thread Gaussian elimination with partial pivoting**, one thread per waypoint set (both
  axes), operating on per-thread LOCAL memory (32×32 = 1,024 floats — past the register budget). →
  THEORY §the-GPU-mapping.
- **Two independent verification stages** — GPU-vs-CPU coefficient agreement (the §5 gate), and a
  from-scratch, double-precision residual check against the constraint DEFINITIONS (interpolation,
  endpoint derivatives, interior continuity) plus an analytic snap-cost sanity check. → THEORY
  §how-we-verify-correctness.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/minimum-snap-trajectory-optimization-batched.sln`](build/minimum-snap-trajectory-optimization-batched.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/minimum-snap-trajectory-optimization-batched.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
The batched solve is hand-rolled (no cuSOLVER/cuBLAS) precisely because this project *is* the
explanation of what a batched linear-algebra library does inside (CLAUDE.md §1); README §11 points
at cuSOLVER's batched routines for the production alternative.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU
and against the constraint definitions):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
**where the plottable trajectory and the path image land**.

## Data

The committed sample is 5 **hand-designed, named waypoint sets** (`data/sample/waypoint_sets.csv`,
820 bytes, synthetic) — `straight_line`, `right_angle`, `slalom` (required — used for the artifact),
`s_curve`, `big_loop`. They occupy the front of every batch the demo runs; the remaining ~9,995
sets of the default 10,000-set batch are generated in-demo from a documented fixed seed (bounded
box, minimum waypoint spacing). No public dataset applies — a waypoint set is 5 numbers a mission
planner or a human would supply, not a recording; `scripts/download_data.ps1` is an honest no-op.
Details: [`data/README.md`](data/README.md).

## Expected output

Seven stable lines — banner, `PROBLEM:`, `SAMPLE:`, `VERIFY:`, `CONSTRAINTS:`, `ARTIFACT:`,
`RESULT:` — checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt).
Two distinct, independent verifications, run on the full K=10,000-set batch:

1. **VERIFY (the §5 GPU-vs-CPU gate):** every one of the 640,000 solved coefficients (10,000 sets ×
   2 axes × 32 coefficients/axis) computed by the kernel and by
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp) must agree within relative tolerance 5e-3 (floor
   1) — measured worst case: **~6.4e-4**, roughly 8× headroom.
2. **CONSTRAINTS (verification against the mathematical definition, independent of the CPU
   oracle):** for every one of the 10,000 sets, re-evaluated with a separately-coded,
   double-precision evaluator —
   - waypoint interpolation error ≤ 1e-3 m (measured worst: **~5.1e-5 m**, ~20× headroom);
   - endpoint zero-derivative (velocity/accel/jerk) error ≤ 1e-2 (measured worst: **~2.3e-4**, ~44×
     headroom);
   - interior continuity jump (relative, floor 1) ≤ 1e-2 (measured worst: **~5.0e-4**, ~20×
     headroom);
   - the analytic snap-cost integral finite and non-negative for all 10,000 sets (measured range on
     this batch: **[1.5e3, 1.4e6]**, both bounds finite and positive).

   Success thresholds carry an order of magnitude or more of headroom above the measured values so
   ordinary FMA/rounding differences across GPU architectures (sm_75/86/89) cannot flip the verdict
   — only a genuine bug (which shifts these numbers at O(1), not at the FP32-rounding scale) does.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the project's one-place contract: waypoint/coefficient
   layouts, the 32-constant problem shape, and the exact row-index derivation of the 32×32 system
   (read this first — everything else refers back to it).
2. [`src/kernels.cu`](src/kernels.cu) — the GPU heart: `assemble_minsnap_system` (the constraint
   matrix, built from one small identity evaluated only at `tau=0` or `tau=1`),
   `solve_minsnap_system` (Gaussian elimination with partial pivoting), and the kernel that calls
   both, twice, per thread. The file header's "what is new relative to 33.01" section is the single
   most important comment in this project.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the oracle twin: line-by-line identical math,
   diff it against `kernels.cu` to see exactly what parallelizing changed (nothing about the
   algorithm — only the surrounding loop).
4. [`src/main.cu`](src/main.cu) — orchestration: batch construction (sample sets + seeded random
   fill), the GPU/CPU calls, the VERIFY and CONSTRAINTS stages (`check_batch` and
   `eval_segment_derivs` are the independent-verification heart), and the artifact writer.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Mellinger & Kumar (2011), "Minimum Snap Trajectory Generation and Control for Quadrotors"** —
  the source paper: the differential-flatness argument for why snap is the right objective, and the
  original free-derivative QP formulation this project's closed-form spline deliberately simplifies
  (see [Limitations & honesty](#limitations--honesty)).
- **Richter, Bry & Roy, "Polynomial Trajectory Planning for Aggressive Quadrotor Flight"** — the
  numerically robust "mapping matrix" / unconstrained-QP formulation that solves for the FREE
  interior derivatives via the cost Hessian — the production-grade version of the problem this
  project scopes down to a fixed square linear system.
- **ethz-asl `mav_trajectory_generation`** — a widely used open-source C++ implementation of the
  free-derivative minimum-snap QP, with time allocation; compare its API and its handling of
  interior free variables against this project's fixed, fully-continuous construction.
- **PX4 Autopilot** — the production flight-control firmware a real quadrotor's trajectory would
  feed into (position/attitude control loops at the rates SYSTEM_DESIGN.md §1.1 documents);
  PRACTICE.md §3 discusses where a trajectory generator like this one would run relative to PX4.
- **cuSOLVER's batched dense solvers** (`cusolverDnSgetrfBatched` / `cusolverDnSgetrsBatched`) — the
  production GPU library for exactly "many small independent linear systems"; this project
  hand-rolls the same idea (in-thread Gaussian elimination) so the mechanics are never a black box
  (CLAUDE.md §1) — swapping in cuSOLVER is README Exercise 4's natural extension.

## Exercises

1. **Plot the artifact:** `demo/out/trajectory.csv` → `x_m` vs `y_m` for the slalom shape, and
   `vx_ms`/`vy_ms` vs `t_s` to see the smooth speed profile pumping through each turn. Also open
   `demo/out/slalom_path.pgm` directly in an image viewer for a zero-setup sanity check.
2. **Time allocation:** replace `kSegmentDurationS`'s fixed equal segment time with a heuristic
   (e.g., duration proportional to Euclidean waypoint distance, or a trapezoidal-velocity estimate)
   and re-derive which constant becomes per-segment instead of global — note carefully which parts
   of `assemble_minsnap_system` stay tau-only and which now need a `T_seg[s]` scale factor.
3. **Extend to 3-D:** add a `z` axis (a third 32×32 solve per waypoint set, same code shape) and
   extend the artifact/PGM code to show altitude. Nothing about the constraint layout changes.
4. **Exploit the shared matrix:** every thread currently re-assembles and re-eliminates the SAME
   32×32 matrix `A` twice (kernels.cu's file header calls this out explicitly). Factor `A` once
   (host or a single kernel), broadcast `L`/`U`, and have each thread do only forward/back
   substitution for its own right-hand side(s) — turning `O(K·N³)` into `O(N³ + K·N²)`. Measure the
   speed-up and explain why it is NOT proportional to the flop-count reduction alone (memory
   traffic, occupancy).
5. **Implement the true free-derivative QP** (Mellinger & Kumar / Richter-Bry-Roy): leave interior
   snap/crackle/pop free instead of continuity-pinned, and solve the reduced KKT system for the
   values that minimize total snap cost. Compare the resulting trajectory's cost against this
   project's closed-form spline on the same waypoint sets — by how much does the "true" minimum
   improve on the simplified construction, and where does the difference show up visually?

## Limitations & honesty

- **Not the free-derivative QP.** This project's 32×32 system is a **closed-form, fully-determined
  spline construction** (interior continuity through pop, degree 6) chosen specifically because the
  degrees-of-freedom count (8 + 6 + 3×6 = 32) makes it solvable by plain Gaussian elimination — no
  quadratic-cost optimization happens at runtime. Mellinger & Kumar's original minimum-snap QP
  instead leaves the interior velocity/accel/jerk (and everything above) free at each waypoint and
  chooses their values by minimizing the true snap-squared cost, which requires solving a larger
  KKT system (or the reduced "mapping matrix" system). THEORY.md §the-math derives the exact DOF
  arithmetic and is explicit that this project's trajectory is a well-known, standard simplification
  — not literally the argmin of the snap functional in general. Exercise 5 builds the real thing.
- **Fixed, equal segment times.** `kSegmentDurationS` is one constant shared by all 4 segments of
  every waypoint set; real time allocation (segments proportional to distance, or optimized
  jointly with the trajectory) is Exercise 2, documented and not implemented here.
- **2-D only.** The x/y ratified scope; z is Exercise 3 (the code shape is identical — one more
  32×32 solve per set).
- **Redundant per-thread work, by design.** Every thread reassembles and re-eliminates the SAME
  constraint matrix — see `kernels.cu`'s file header and Exercise 4. This is the honest,
  maximally-parallel "thread owns the whole problem" pattern the repo teaches everywhere (08.01,
  33.01, 09.01), traded deliberately against a smarter factor-once design.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **Sim-validated only (CLAUDE.md §1):** this project's output is a *trajectory* that would feed a
  flight controller commanding real rotors. Everything here is pure computation, verified only
  against a CPU oracle and its own mathematical definition — no vehicle dynamics, no actuator
  limits, and no safety envelope are modeled. Any real-hardware use would need the full testing
  ladder (PRACTICE.md §3) plus an independent flight-safety review; nothing here is a certified or
  flight-ready trajectory generator.
