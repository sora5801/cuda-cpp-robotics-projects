# 16.01 — Thruster allocation for overactuated ROVs (batched QP)

**Difficulty:** intermediate · **Domain:** 16. Locomotion — Marine & Underwater

> Catalog bullet (source of truth, verbatim): `Thruster allocation for overactuated ROVs (batched QP)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

An ROV (remotely operated vehicle) with 8 thrusters and only 6 degrees of freedom (DOF) to control is
**overactuated on purpose** — the extra thrusters buy fault tolerance and full station-keeping
authority (THEORY.md explains why). Somebody still has to answer, every control tick: *given a
commanded body wrench (3 forces + 3 moments), which of the 8 signed thruster forces produce it,
without asking any thruster for more than it can give?* That is a small, structured **box-constrained
quadratic program (QP)** — one 8-unknown, 6-equation least-squares problem per commanded wrench, solved
subject to `|u_i| <= u_max_i`. This project builds that QP, solves **thousands of them at once** on the
GPU with one thread per problem (the same batched-solve pattern as 33.01's linear systems and 08.01's
rollouts), and verifies the result two ways that matter more than "GPU equals CPU": against the
**closed-form pseudoinverse** ground truth for easy (unsaturated) wrenches, and against the
**KKT optimality condition** for hard (saturated) ones. A second demo stage then answers the question a
real ROV operator actually cares about — *if thruster i seizes, how much worse does the vehicle's
wrench-tracking get?* — by re-solving the same batch nine times (nominal + one failure per thruster).

## What this computes & why the GPU helps

Per problem: form an 8-element gradient from an 8x8 matrix-vector product, project onto a box, repeat
500 times. That is ~500 x 80 flops = 40,000 flops — utterly trivial for one GPU thread, and far too
small to parallelize *inside* one problem. What is not trivial is that a real allocator faces
**thousands of these per second** (a whole planned trajectory's wrench sequence, or — this project's
second demo stage — the same batch re-solved once per candidate thruster failure).

- **Pattern:** batched solve — one GPU thread owns one whole QP (all 8 unknowns, both matrix-vector
  products, and the 500-iteration optimizer loop) entirely in registers. Zero interaction between
  threads, by construction — the same shape as 33.01's batched Cholesky and 08.01's MPPI rollouts.
- **A textbook __constant__-memory case:** the QP's 8x8 Hessian and 8x6 gradient-forming matrix are
  *identical for every thread in every launch* (they depend only on the vehicle's fixed geometry, never
  on the commanded wrench) — every thread reads the exact same address, every iteration, which CUDA's
  constant cache serves at near-register cost after the first touch. See
  [`src/kernels.cu`](src/kernels.cu) for the measurement and the read-pattern spectrum this sits on.
- **Measured reality:** batch-allocating all 500 commanded wrenches in the committed demo trajectory
  (500 QPs x 500 iterations each = 250,000 8x8-matvec-plus-projection steps) takes a fraction of a
  millisecond of GPU kernel time versus several milliseconds on one CPU core — see the demo's `[time]`
  line for this run's numbers (a teaching artifact, not a benchmark claim).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **control-allocation boundary between control and actuation** — the last
  computation before individual actuator setpoints. It consumes a *body-frame wrench command* (the
  output of a control law) and produces *per-thruster force setpoints* (the input to motor drivers); it
  never sees sensors, a map, or a plan directly.
- **Upstream inputs:** a commanded 6-DOF wrench `tau_cmd` (message shape: like `geometry_msgs/Wrench`)
  from a station-keeping controller (a DP PID) or a sampling controller such as **16.09's
  docking-under-current MPPI** — named because that project is this one's most natural upstream
  neighbor: it plans a wrench trajectory against a current disturbance, and this project turns each
  planned wrench into thruster commands.
- **Downstream consumers:** 8 signed force setpoints, one per thruster — each still one layer above real
  hardware (PRACTICE.md §1-2 covers the force-to-RPM layer this project deliberately stops short of);
  downstream is the motor-driver/ESC firmware that turns a force setpoint into a commanded RPM.
- **Rate / latency budget:** SYSTEM_DESIGN §1.1's table has no row literally named "control allocation";
  it runs *inside* whatever control loop calls it, at that loop's rate. For a DP/station-keeping ROV
  that is typically the **local-planner/control band, 10-50 Hz** (SYSTEM_DESIGN §1.1); tightly-coupled
  designs that fold allocation into a faster inner loop can push it toward the **0.5-1 kHz whole-body
  control band**. Either way the budget is dominated by "one allocation per control tick," and this
  demo's measured GPU kernel time for a WHOLE 500-wrench batch (see the `[time]` line) is already far
  below a single 20 ms tick's budget — headroom this project intentionally over-provisions (the real
  payoff of the GPU here is the two *batch* use cases: a whole trajectory at once, and the
  fault-tolerance sweep, both below).
- **Reference robot(s):** **none of the five §2 reference robots directly** (they are the warehouse AMR,
  manipulator cell, quadruped, quadrotor, and AV stack — all in-air or on-the-ground archetypes). This
  project's "robot" is the **ROV/AUV archetype** itself — the underwater vehicle every domain-16 project
  ultimately serves. [`PRACTICE.md`](PRACTICE.md) grounds that archetype in real hardware and operations.
- **In production:** general-purpose QP solvers (OSQP, qpOASES) or classical generalized-inverse thrust
  allocation with saturation handling (Fossen 2011; DP-industry allocation, e.g. Kongsberg-class systems)
  replace this project's hand-rolled projected gradient descent — see "Prior art" below.
- **Owning team:** vehicle **controls & autonomy** (SYSTEM_DESIGN §5.1's org map lists domains 13-18,
  including this one, under that team), working closely with electrical/embedded (who own the motor
  drivers this project's output ultimately reaches).

## The algorithm in brief

- **Allocation-matrix construction** — stack each thruster's force direction and moment arm
  (`tau_i = r_i x d_i`) into the 6x8 matrix `B`, once, from the vehicle's geometry. →
  [THEORY.md](THEORY.md) §The math.
- **Box-constrained QP** — `min ||W(Bu - tau)||^2 + eps||u||^2` s.t. `|u_i| <= u_max_i`; `eps` keeps the
  problem well-posed along `B`'s 2-dimensional null space (the "internal squeeze" redundancy an
  8-thruster/6-DOF vehicle always has). → THEORY §The math.
- **Projected gradient descent (PGD)** — a *fixed* number of gradient-plus-clamp steps, step size
  `1/L` from a host power iteration on the QP's Hessian; the box projection IS the QP theory this
  project teaches (no general-polytope solver needed). → THEORY §The algorithm.
- **Thread-per-QP batching** — thousands of independent problems, one GPU thread each, the shared
  Hessian in `__constant__` memory. → THEORY §The GPU mapping.
- **Two ground-truth optimality gates** — unsaturated solutions must match the closed-form **damped
  weighted pseudoinverse**; saturated solutions must satisfy the box **KKT** fixed-point condition. →
  THEORY §How we verify correctness.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/thruster-allocation-for-overactuated-rovs.sln`](build/thruster-allocation-for-overactuated-rovs.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/thruster-allocation-for-overactuated-rovs.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuBLAS/cuSOLVER: the matrices here are 6x8/8x8, small enough that hand-rolled register code beats any
library call's launch overhead (the same call CLAUDE.md §5 and 33.01 make for their sizes).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at, including both CSV artifacts.

## Data

The committed sample is **synthetic** (CLAUDE.md §8 default), generated by
[`scripts/make_synthetic.py`](scripts/make_synthetic.py) with a fixed seed (42):

- `data/sample/wrench_batch.csv` — the file the program actually loads: a 500-sample, 10 Hz commanded
  wrench trajectory (50 s) modeling an ROV holding station against a fluctuating current, with two
  deliberate "docking correction" bursts sized to saturate several thrusters at once. 472 of the 500
  rows come out unsaturated, 28 saturated — both of this project's optimality gates get real exercise.
- `data/sample/rov_geometry.csv` — a **human-readable documentation copy** of the 8-thruster geometry
  that is actually compiled into [`src/kernels.cuh`](src/kernels.cuh) (the program's real source of
  truth for the vehicle's geometry — CLAUDE.md §12 single-sourcing). This file is not read at runtime.

No public dataset applies (this is a synthetic teaching vehicle, not a specific commercial ROV's
telemetry); `scripts/download_data.ps1`/`.sh` are honest no-ops. Full field/unit documentation, the
regeneration command, and checksums: [`data/README.md`](data/README.md).

## Expected output

Ten stable lines: banner, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, three `GATE-*:` lines, two `ARTIFACT:`
lines, and `RESULT:` — checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt).
Four independent checks, all measured on the reference machine (RTX 2080 SUPER):

1. **§5 GPU-vs-CPU gate (`VERIFY`)** — the GPU kernel and [`src/reference_cpu.cpp`](src/reference_cpu.cpp)
   allocate the identical 500-wrench batch; worst per-thruster-force disagreement must be `<= 5e-3 N`
   (measured: `2.7e-05 N`).
2. **`GATE-PSEUDOINV`** — every *unsaturated* solution (472/500 rows) must match the closed-form damped
   weighted pseudoinverse (an independent 8x8 Cholesky solve) within `0.05 N` (measured worst: `6.1e-05 N`).
3. **`GATE-KKT`** — every *saturated* solution (28/500 rows) must satisfy the box-constrained KKT
   fixed-point condition (its projected-gradient residual near zero) within `0.05 N` (measured worst:
   `5.7e-06 N`).
4. **`GATE-MONOTONE`** — on this project's motivating worked example (THEORY.md §How we verify
   correctness), the QP objective must be non-increasing across all 500 projected-gradient iterations
   (measured: `0` violations, with a documented FP32-rounding slack of `1e-3`).

Two artifacts: `demo/out/allocation.csv` (commanded vs. achieved wrench and all 8 thruster forces, per
row of the batch) and `demo/out/failure_analysis.csv` (the nine-configuration fault-tolerance sweep —
see "Code tour" and THEORY.md for how to read it).

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in order: one-time setup (build the allocation
   matrix, form the QP, power-iterate the step size) -> load the wrench batch -> VERIFY -> the three
   optimality gates -> write `allocation.csv` -> the nine-configuration failure-analysis sweep -> write
   `failure_analysis.csv`.
2. [`src/kernels.cuh`](src/kernels.cuh) — the project's one-place contract: the thruster geometry table,
   the wrench/force layouts, the QP hyperparameters, and — spelled out in the header comment — exactly
   why this project uses the marine (Fossen/SNAME) body frame instead of the repo's usual one.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernel: one thread per QP, `__constant__` memory for the
   shared Hessian, a fixed-iteration projected-gradient loop. The single most interesting thing to look
   at: how the box projection (one `fminf`/`fmaxf` pair) is the *entire* constrained-optimization theory
   this kernel needs.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — TWO jobs in one file: the CPU oracle (a sequential
   twin of the kernel), and the one-time host setup math (building `B`, forming the QP, the power
   iteration, and the Cholesky solve the pseudoinverse gate uses as ground truth).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — CLAUDE.md §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Fossen (2011), *Handbook of Marine Craft Hydrodynamics and Motion Control*** — the standard marine
  vehicle-control reference; this project's body-frame convention, allocation-matrix formulation, and
  the general damped-pseudoinverse-with-saturation allocation story all follow its treatment.
- **DP (dynamic positioning) thrust allocation literature** (surveyed in Sørensen 2011, "A survey of dp
  control systems"; Johansen & Fossen 2013, "Control allocation — a survey") — the industrial lineage of
  "turn a commanded wrench into constrained actuator commands," including the generalized-inverse and
  quadratic-programming formulations this project's QP is a teaching-scale version of.
- **ArduSub / BlueOS** — the open-source autopilot stack that actually flies BlueROV2-class vehicles;
  its real motor-mixing matrices are a live example of the exact geometry -> allocation-matrix pipeline
  this project teaches, at production scope.
- **OSQP** and **qpOASES** — production general-purpose QP solvers (ADMM and active-set, respectively)
  that would replace this project's hand-rolled projected gradient descent in a real deployment; compare
  their convergence guarantees and per-solve cost against the simple fixed-iteration loop here.
- **cuSOLVER / cuBLAS batched routines** — NVIDIA's batched dense-linear-algebra libraries (33.01's
  territory); there is no off-the-shelf *constrained* batched-QP primitive in the CUDA math libraries,
  which is precisely why this project hand-rolls the solver (CLAUDE.md §1: no black boxes).

## Exercises

1. **Warm-start the optimizer.** The kernel cold-starts every QP at `u=0` (README "Limitations"). Pass
   in the *previous tick's* solution as the starting point instead and measure how few iterations are
   needed to reconverge for a slowly-varying wrench trajectory like the committed sample.
2. **Retune the weights.** `kWeight` in `kernels.cuh` is uniform. Increase the yaw (`Mz`) weight (a
   common DP choice — heading hold matters more than exact position during a hold) and rerun the demo;
   inspect `allocation.csv` for how the achieved wrench's Mz-tracking improves at the expense of the
   other axes.
3. **Add an early-exit convergence check.** Break the kernel's loop once the projected-gradient norm
   drops below a threshold, profile the result in Nsight Compute, and explain the warp-divergence cost
   against the fixed-iteration version's uniform control flow (THEORY.md "The GPU mapping" sets up the
   trade-off).
4. **Asymmetric thruster limits.** Real thrusters produce more forward than reverse bollard thrust
   (PRACTICE.md §2). Extend the box from `[-u_max, u_max]` to `[u_min_i, u_max_i]` per thruster and
   rerun the failure analysis with a realistic ~1.25:1 forward/reverse ratio.
5. **Break the geometry on purpose.** Perturb `kThrusterPos`/`kThrusterDir` slightly (simulating
   manufacturing tolerance or a bent thruster mount) so `B` picks up small cross-coupling terms, and
   compare how gracefully the QP handles it versus a naive (unconstrained, unweighted) pseudoinverse.

## Limitations & honesty

- **Regularization bias is real and measured, not hidden.** `eps=0.1` (chosen for fast, well-conditioned
  convergence — THEORY.md derives the condition number, ~41) attenuates the vehicle's *weakest* authority
  directions substantially: this geometry's pitch (`My`) response retains only `~47%` of a commanded
  pitch moment even when nothing saturates, roll (`Mx`) `~62%`, yaw (`Mz`) `~71%`, with surge/sway/heave
  much closer to full authority (`~95-98%`) — THEORY.md §Numerical considerations derives every number
  from the allocation matrix's eigenstructure. This is *why* the failure-analysis artifact measures
  **degradation relative to the nominal baseline**, not an absolute tracking-error threshold — see the
  comment in `main.cu`'s failure-analysis stage.
- **Symmetric, static thruster limits.** `u_max` is the same in both directions and the same for every
  wrench in the batch (except where the failure sweep deliberately zeros one out) — real thrusters are
  forward/reverse-asymmetric and their achievable thrust droops with battery voltage (PRACTICE.md §2;
  Exercise 4 addresses the asymmetry).
- **Force allocation only — not the force-to-RPM layer.** This project stops at *signed thruster force*;
  the real prop-law mapping force to commanded RPM (`T = k*n*|n|`, THEORY.md "the problem") and
  thruster-thruster wake interaction are named and discussed but not modeled (THEORY.md §Where this sits
  in the real world).
- **Zero-thrust cold start, fixed iteration count.** Every QP solves from scratch every time (no
  warm-starting — Exercise 1) for exactly `kPgdIters` steps regardless of difficulty (no early-exit
  convergence check — Exercise 3): a deliberate simplicity/determinism trade over peak efficiency.
- **Synthetic geometry and synthetic wrenches.** The 8-thruster layout is this project's own didactic
  vehicle (sized like a small/medium observation-class ROV), not reverse-engineered from any commercial
  vehicle's CAD; the wrench trajectory is a synthetic docking-under-current scenario, not recorded
  telemetry — both labeled synthetic everywhere they appear (CLAUDE.md §8).
- **Sim-validated only, not safety-certified (CLAUDE.md §1 at full strength here):** this project's
  output is a set of *thruster force commands* — the same category of output as 08.01's controller. Every
  number in this repo comes from allocating a synthetic wrench trajectory in software; nothing here has
  ever driven a real thruster, and any real-hardware use would demand the full testing ladder
  (PRACTICE.md §3) plus an independent safety envelope.
