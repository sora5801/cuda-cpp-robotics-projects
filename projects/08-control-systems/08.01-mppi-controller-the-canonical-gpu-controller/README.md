# 08.01 — MPPI controller — the canonical GPU controller: cart-pole → quadrotor → AGV → off-road racer

**Difficulty:** ★ beginner · **Domain:** 8. Control Systems

> Catalog bullet (source of truth, verbatim): `★ MPPI controller — the canonical GPU controller: cart-pole → quadrotor → AGV → off-road racer`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**MPPI (Model Predictive Path Integral) control steers by sampling futures.** Every 20 ms it asks:
"if I perturbed my current plan 4096 different ways, simulated each for one second, which
perturbations lead somewhere good?" — then blends the answers, weighted by how good, into the next
plan. No gradients, no linearization: it works on genuinely nonlinear problems where classical
controllers give up, *provided* you can simulate thousands of futures per tick. That proviso is
the GPU. This project builds the complete loop on the classic teaching plant — a **force-limited
cart-pole** that must *pump energy* to swing its pole up from hanging (no linear controller can) —
and runs it closed-loop: swing-up in ~2.3 s, then balance. The bullet's ladder (cart-pole →
quadrotor → AGV → off-road racer) shares this exact code shape; only the dynamics function and
cost grow. This is the catalog's ★ entry into all of sampling-based control, and CLAUDE.md §6.2
uses this very kernel as its commenting-style example.

## What this computes & why the GPU helps

Per 20 ms control tick: K=4096 rollouts × T=50 RK4 steps × ~90 flops each ≈ 18M flops of pure,
independent simulation — then one cheap softmin blend.

- **Pattern:** batched sampling — one thread = one rollout (a whole simulated future in registers);
  zero interaction between rollouts, by construction.
- **Measured reality:** the rollout set takes ~0.3–1.4 ms of GPU kernel time vs ~21 ms on one CPU
  core — the difference between "4096 samples at 50 Hz" being a controller and being a paper.
- **Layout lesson applied:** the noise array is stored **transposed** (`eps[t*K+k]`) so each
  simulation step reads coalesced (33.01 taught the cost of the naive layout; here the fix is
  applied from the start and explained in [`src/kernels.cuh`](src/kernels.cuh)).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **control layer** — the last software box before actuation; MPPI
  specifically spans local-planner and controller roles (it plans a horizon *and* emits the next
  actuation command).
- **Upstream inputs:** the current state estimate (message shape: joint/body state à la
  `nav_msgs/Odometry` or `sensor_msgs/JointState` — here the 4-float cart-pole state from
  [`src/kernels.cuh`](src/kernels.cuh)), a dynamics model, and a cost function encoding the task
  (from planning: goals, obstacle fields like 07.09's).
- **Downstream consumers:** the actuation chain — force/torque setpoints to motor drives (here,
  one force command applied to the simulated plant each tick).
- **Rate / latency budget:** MPPI-class controllers run at 10–50 Hz on real robots
  (SYSTEM_DESIGN item 1); this demo runs the full loop at 50 Hz with ~0.3 ms average GPU kernel
  per tick — an order of magnitude of headroom the bigger plants on the bullet's ladder consume.
- **Reference robot(s):** the **quadruped** (sampling whole-body/locomotion control) and the
  **off-road AGV/AV** (MPPI's original home: aggressive driving) most directly; the quadrotor
  chain in SYSTEM_DESIGN's composition map slots an MPPI/NMPC at the same position.
- **In production:** MPPI descendants run in AutoRally/off-road racing stacks, sampling MPC in
  MuJoCo-MPC, and NVIDIA's rollout-based planners; where certification demands determinism,
  gradient MPC (08.03/08.04) or classical control takes the actuation seat with sampling methods
  above it.
- **Owning team:** controls/autonomy (SYSTEM_DESIGN item 5) — with the safety team owning the
  envelope around it (see PRACTICE §4).

## The algorithm in brief

- **MPPI update** — sample `u_k = clamp(u_nom + ε_k)`, simulate, cost `S_k`; weights
  `w_k = exp(−(S_k−S_min)/λ)`; blend `u_nom += Σw_kε_k/Σw`; apply `u_nom[0]`; shift; repeat. →
  [THEORY.md](THEORY.md) §The math (including where that exponential comes from).
- **RK4 integration** of the cart-pole ODE under zero-order hold — the model must be trustworthy
  or MPC optimizes fiction. → THEORY §Numerical considerations.
- **Cost shaping** — `(1−cosθ)` for uprightness (smooth, wrap-free), quadratics on velocities,
  position, effort. The tuning story is told, not hidden. → THEORY §The algorithm.
- **Receding horizon** — the plan is a rolling window; yesterday's tail seeds today's plan.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/mppi-controller-the-canonical-gpu-controller.sln`](build/mppi-controller-the-canonical-gpu-controller.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/mppi-controller-the-canonical-gpu-controller.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only
(noise is host-generated for reproducibility; on-device cuRAND is Exercise 4).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
the **trajectory artifact to plot**.

## Data

The committed sample is a **scenario**, not recordings: `data/sample/cartpole_scenario.csv`
(~0.4 KiB, synthetic, no RNG) — start hanging at rest, run 400 steps at 50 Hz. Noise, rollouts,
and the simulated plant are generated in-demo from documented fixed seeds. No public dataset
applies; `scripts/download_data.ps1` is an honest no-op. Details: [`data/README.md`](data/README.md).

## Expected output

Six stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY: PASS`, `ARTIFACT:`, `RESULT: PASS` —
checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Two distinct
verifications: **(1)** the §5 GPU-vs-CPU gate — iteration 0's 4096 rollout costs computed by the
kernel and by [`src/reference_cpu.cpp`](src/reference_cpu.cpp) must agree within rel 1e-3 (measured
worst: ~1.8e-07); **(2)** the control check — |θ| < 0.2 rad for every one of the final 100 steps
(measured: balanced for the final 287, final θ = 0.006 rad). Success thresholds carry wide margins
so platform ulp differences in the host-generated noise cannot flip the verdict.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole MPPI loop in plain sight: noise → rollouts → softmin →
   blend → act → shift; plus the verify stage and the artifact writer.
2. [`src/kernels.cuh`](src/kernels.cuh) — the model constants, state layout, and the transposed
   noise-layout decision (the project's one-place contracts).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the oracle twin *and* the plant stepper (the
   project's single angle-wrap point lives here).
4. [`src/kernels.cu`](src/kernels.cu) — the heart: cart-pole dynamics from the Lagrangian, RK4,
   the stage cost, and the rollout kernel. The single most interesting thing: how small the kernel
   is — the entire "GPU controller" fits on one screen once the pattern is right.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Williams, Aldrich & Theodorou (2017), "Model Predictive Path Integral Control"** — the MPPI
  paper (information-theoretic derivation of the exponential weights; our THEORY.md walks the
  intuition, they own the proof).
- **AutoRally** (Georgia Tech) — MPPI driving a real off-road car at speed; the bullet's last rung
  in the wild, noise generated on-GPU with cuRAND.
- **MuJoCo-MPC** — sampling MPC (including MPPI variants) as a polished research tool; compare its
  planner/estimator split with this demo's loop.
- **Drake / acados / OSQP** — the gradient-MPC world (08.03/08.04's territory); know both families
  and when each wins.
- **Sutton & Barto's cart-pole** — the plant's lineage; the same equations under RL instead of MPC
  (project 12.06 trains the same plant with PPO).

## Exercises

1. **Plot the artifact:** `demo/out/trajectory.csv` → θ vs t. Identify the energy-pumping
   oscillations, the swing-up moment, and the catch. Then plot `u_N` and find where the force
   limit saturates.
2. **Break the temperature:** run with λ = 0.05 and λ = 50 (edit `kLambda`, rebuild) and explain
   both failure modes (greedy collapse vs. uniform mush) from the weight formula.
3. **Fuse the update:** move the softmin blend onto the GPU (a weighted-reduction kernel over the
   transposed eps) and eliminate the per-tick eps download — measure the tick time before/after.
4. **On-device noise:** replace the host noise with cuRAND (Philox, per-rollout streams) and
   quantify what the 800 KB-per-tick upload was costing; document the determinism trade you made.
5. **Climb the ladder:** swap the dynamics/cost for a planar quadrotor (6-state) — nothing else in
   the loop changes; that invariance is the whole point of the pattern.

## Limitations & honesty

- **The plant is the model** — the controller drives the same RK4 cart-pole it rolls out
  (zero model mismatch, deliberately ideal). Real MPPI lives or dies by model quality; robustness
  under mismatch is Exercise 5 territory and the [R&D] frontier (08.05 tube MPC).
- **Host-side softmin + per-tick noise upload** — didactic transparency over peak performance;
  Exercises 3–4 remove both and the header comments say what production stacks do.
- **Cart-pole only** — the ladder's later rungs (quadrotor, AGV, racer) are documented
  (THEORY §real-world), not implemented here.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **Sim-validated only, and this one matters here (CLAUDE.md §1):** this project's output is a
  *force command* — the archetype of code whose consumers move hardware. Everything here ran only
  against the simulated plant; nothing is safety-certified, no real-robot claim is made, and any
  hardware use would demand the full testing ladder (PRACTICE §3) plus an independent safety
  envelope. That is the repo-wide caveat at full strength.
