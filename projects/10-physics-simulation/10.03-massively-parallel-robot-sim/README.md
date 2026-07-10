# 10.03 — Massively parallel robot sim (Isaac-Gym-style: one robot, 10,000 environments)

**Difficulty:** ★ beginner · **Domain:** 10. Physics Simulation

> Catalog bullet (source of truth, verbatim): `★ Massively parallel robot sim (Isaac-Gym-style: one robot, 10,000 environments)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**Isaac Gym's actual trick isn't a smarter physics engine — it's running the SAME engine ten thousand
times at once, entirely on the GPU, so an RL algorithm can generate months of simulated robot
experience in hours.** This project builds that pattern, completely, on a tractable plant: 08.01's
force-limited cart-pole, reused verbatim, generalized so its mass and pole length become
**per-environment random variables** instead of shared constants. `N = 10,000` independent copies of
that robot — each with its own randomized dynamics and its own episode clock — are stepped in lockstep
for `T = 1,000` ticks (20 simulated seconds each) inside **one GPU kernel launch**, under a fixed,
pole-placed balance controller applied identically everywhere. Three ingredients make this an
"RL training farm" and not just a batch of independent demos: **parallel environments**, **domain
randomization** (mass/length drawn per environment, once, from a documented range), and **episode
reset** (each environment fails or times out and restarts on its own clock). The demo verifies the
whole thing three independent ways — a GPU-vs-CPU state comparison, a farm-wide finiteness/reset-count
gate, and a from-first-principles energy-conservation check on the RK4 integrator itself — and reports
the number that actually matters for a training farm: **aggregate environment-steps per second**
(several billion on the reference GPU), not any single environment's latency.

## What this computes & why the GPU helps

Per farm run: `N × T` independent RK4 integrations of a 4-state ODE, plus a scalar linear controller
evaluation and an episode-termination check at every one of those steps — for the default scenario,
10 million per-tick evaluations, ~40 million derivative calls.

- **Pattern:** batched, STATEFUL simulation — one thread owns one environment's state, resident in
  GLOBAL memory across the whole run (not a fresh input every call, the way 08.01's rollouts are);
  environments never interact, by construction, so this is the "many independent copies" pattern
  taken to its persistent, farm-scale conclusion.
- **Measured reality:** the entire 10,000-environment, 1,000-tick farm run completes in ~1–2 ms of GPU
  kernel time on the reference RTX 2080 SUPER — an aggregate throughput in the **several-billion
  environment-steps/second** range (single-shot, teaching artifact, not a benchmark claim; see the
  demo's `[time]` line for the exact run's number).
- **Layout lesson (this project's central new one):** state lives in **structure-of-arrays** (SoA) —
  `x[N], xdot[N], theta[N], thdot[N]` — not the array-of-structs a single stateless kernel call would
  use. [`src/kernels.cuh`](src/kernels.cuh) explains why persistent, lockstep-accessed state makes SoA
  the coalescing-correct choice here, in contrast to 08.01/09.01's fully-in-register single-call kernels.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **cross-cutting simulation & digital-twin layer** (SYSTEM_DESIGN §1's
  cross-cutting row, domain 10) — it does not sit inside the sense→plan→act pipeline at all; it
  generates the EXPERIENCE that trains or validates blocks that do (here: a controller like 08.01's;
  in general, any policy).
- **Upstream inputs:** a robot model (here: 08.01's cart-pole equations of motion, reused) and a task
  scenario (`data/sample/farm_scenario.csv` — environment count, run length, domain-randomization
  ranges, controller gains; message-shape equivalent: a training-job config, not a live topic).
- **Downstream consumers:** an **RL training loop** (12.06's PPO/GAE kernels, cited in
  SYSTEM_DESIGN.md's Chain C, would consume this exact pattern's `steps_balanced`-shaped return signal
  to update a policy instead of this project's fixed gains) and **controller-tuning sweeps** (08.10-style
  projects that validate a candidate gain set against a randomized population before it is trusted,
  exactly what this project's FARM gate does in miniature).
- **Rate / latency budget:** this project has NO real-time rate budget — it runs offline, as fast as
  possible (SYSTEM_DESIGN item 1.2: "classical GPU territory... simulation" is exactly this — a 10–100
  ms control-loop budget is meaningless here; the number that matters is aggregate throughput, reported
  honestly as such, not disguised as a latency figure).
- **Reference robot(s):** the **quadruped** most directly — SYSTEM_DESIGN §2.3 and Chain C
  (`docs/SYSTEM_DESIGN.md` §4.3) name "10.03 10k-env parallel robot sim" explicitly as the offline half
  of the quadruped's sim→real training loop, with 09.03's Featherstone ABA/RNEA standing in for this
  project's closed-form cart-pole once the plant has real articulated legs. The **6-DoF manipulator
  work cell** (§2.2) is the other natural user: validating a grasp or motion-planning policy across
  randomized payload mass/friction before trusting it on a real arm.
- **In production:** Isaac Gym / Isaac Lab (NVIDIA, PhysX-backed), MuJoCo MJX, and Brax (both
  JAX/XLA-backed) are the production-grade versions of exactly this pattern — see README §11 and
  THEORY.md §real-world for what they add.
- **Owning team:** simulation & tools (SYSTEM_DESIGN item 5.1: "Sim environments, digital twins, CI
  farms, internal libs" → domains 10, 11, 33, 34) — see PRACTICE.md §4 for the adjacent-team breakdown.

## The algorithm in brief

- **The plant** — 08.01's force-limited cart-pole equations of motion, generalized to per-environment
  runtime mass/length instead of compile-time constants. → [THEORY.md §the problem](THEORY.md#the-problem--physics--engineering-first).
- **RK4 integration** with explicit `fmaf()` fused multiply-adds at every stage (a deliberate,
  documented step beyond 08.01's discipline, aimed at minimizing GPU/CPU divergence). → THEORY §numerical considerations.
- **Fixed linear state-feedback controller**, pole-placed on the nominal linearization, applied
  identically in every environment — one policy, stress-tested across a randomized population. →
  THEORY §the math.
- **Domain randomization** — `mass_cart`/`mass_pole`/`pole_half_len` drawn once per environment from a
  documented ±20%/±30%/±15% envelope. → THEORY §domain randomization.
- **Episode reset** — fail (`|x|>2.4 m` or `|θ|>12°`) or cap (200 ticks) triggers an in-place reset with
  a freshly-drawn initial angle, fused into the same kernel that does the stepping (no separate "reset
  kernel" launch — THEORY.md explains why that is the correct design). → THEORY §the GPU mapping.
- **SoA state layout** — the project's central new GPU lesson, contrasted with 08.01/09.01's
  register-only, single-call kernels. → THEORY §the GPU mapping.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/massively-parallel-robot-sim.sln`](build/massively-parallel-robot-sim.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/massively-parallel-robot-sim.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
farm's RNG is a hand-rolled `xorshift32` shared between host and device (not cuRAND) specifically so
the GPU-vs-CPU verify stage compares bit-identical random streams — see [`src/kernels.cuh`](src/kernels.cuh).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
the **two CSV artifacts to plot**.

## Data

The committed sample is a **farm scenario**, not recordings: `data/sample/farm_scenario.csv`
(~1.0 KiB, synthetic, no RNG) — environment count, run length, episode cap, seed, the
domain-randomization envelope, and the controller gains. Every per-environment mass/length draw,
every initial angle, and every mid-run reset is generated **inside the demo, on the GPU**, from this
file's `SEED` field. No public dataset applies; `scripts/download_data.ps1` is an honest no-op.
Details: [`data/README.md`](data/README.md).

## Expected output

Nine stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY: PASS`, `FARM: PASS`, `ENERGY: PASS`, two
`ARTIFACT:` lines, and `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Three independent verifications, all measured
on the reference machine (RTX 2080 SUPER, CUDA 13.3):

1. **VERIFY** — the §5 GPU-vs-CPU gate on a 256-environment, 220-tick subset: worst state deviation
   4.77×10⁻⁷ (tolerance 1×10⁻³), `reset_count` exact match 256/256, `steps_balanced` worst diff 0
   (slack 3).
2. **FARM** — the full 10,000-environment run: every environment's state finite; every environment's
   `reset_count` measured exactly 5 (the provable minimum, given `episode_cap=200` and `T=1000`),
   gated against a documented `[5,12]` range.
3. **ENERGY** — an undriven, unbounded cart-pole's mechanical energy should be exactly conserved;
   measured maximum relative drift over 1000 RK4 steps: 1.045×10⁻⁵ against a documented 1×10⁻³ bound —
   this IS the RK4 integrator's own truncation error, made visible as a number.

Success thresholds carry wide, documented margins (THEORY.md derives and calibrates every one) so
platform-to-platform sinf/cosf ULP differences cannot flip the verdict.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the project's real center of gravity: the SoA state layout,
   the domain-randomization/reset design, and — new relative to every prior flagship — the SHARED
   `HD` (host-and-device) inline functions that main.cu, kernels.cu, and reference_cpu.cpp all call
   verbatim, with the reasoning for why this project departs from 08.01's hand-duplication rule.
2. [`src/kernels.cu`](src/kernels.cu) — `init_farm_kernel` and `step_farm_kernel`: the entire `T`-tick
   run fused into ONE launch per stage. The single most interesting thing to look at: how little code
   the kernels themselves contain, because the physics/controller/reset LOGIC already lives in
   kernels.cuh — the kernels are almost pure "load registers, loop, store registers."
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the oracle twin (sequential over environments
   instead of parallel) PLUS the completely separate energy-conservation diagnostic.
4. [`src/main.cu`](src/main.cu) — the three-stage orchestration: VERIFY, FARM, ENERGY, then the two
   CSV artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Makoviychuk et al. (2021), "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot
  Learning"** — the paper this project's catalog bullet and title are named after; the source of the
  "one robot, thousands of environments, entirely GPU-resident" idea this project teaches in miniature.
- **Isaac Lab** (NVIDIA's successor framework built on Isaac Sim/PhysX) — the current production home
  of this pattern, with full articulated-body dynamics, contacts, and sensor simulation this teaching
  core omits (THEORY.md §real-world).
- **MuJoCo MJX** — the same batched-environment idea implemented via JAX/XLA compilation instead of
  hand-written CUDA kernels; compare its auto-differentiability trade-off against this project's
  hand-rolled transparency.
- **Brax** (Google DeepMind) — another JAX-native massively-parallel physics engine, popular in RL
  research for exactly the throughput story this project measures.
- **Tobin et al. (2017), "Domain Randomization for Transferring Deep Neural Networks from Simulation
  to the Real World"** — the paper that gives this project's `dr_mass_cart`/`dr_mass_pole`/`dr_len`
  design its research grounding (THEORY.md §domain randomization).
- **08.01 (MPPI controller)** — the plant and RK4 integrator this project reuses verbatim; read it
  first if you have not already (this README assumes it).
- **12.06 (RL training kernels, planned)** — the project that would replace this farm's fixed gains
  with a learned policy, keeping everything else in this project unchanged (SYSTEM_DESIGN.md Chain C).

## Exercises

1. **Plot the artifacts:** `demo/out/env_metrics.csv` — scatter `balanced_fraction` against
   `mass_pole_kg` to see whether domain randomization visibly stresses the controller (measured: it
   barely does, at these ranges — that is itself the finding; see Exercise 2). `demo/out/energy_drift.csv`
   — plot `energy_j` vs `t_s` and watch the RK4 drift at a scale you can actually see by zooming in on
   `drift_rel`.
2. **Find the controller's failure boundary.** The measured farm result is that EVERY environment
   resets exactly 5 times (never fails early) at the committed randomization ranges — the fixed gains
   are robust across the whole envelope. Widen `DR_MASS_POLE`/`DR_LEN` in the scenario CSV (regenerate
   with `scripts/make_synthetic.py --dr-mass-pole 0.9 --dr-len 0.5`) or push `THETA0_RANGE` toward the
   12° fail threshold until `reset_count` starts exceeding 5 for some environments — you have just
   found this fixed-gain policy's actual robustness boundary, the thing domain-randomization testing
   exists to discover.
3. **Weaken the controller on purpose.** Move the pole-placement poles closer to the imaginary axis
   (e.g. recompute gains for poles at `{-0.8, -1.0, -1.2±0.3j}` — THEORY.md shows the linearization) and
   observe `reset_count` spread across the farm for the first time — connect the smaller stability
   margin to the eigenvalue analysis in THEORY §the math.
4. **Scale up.** Raise `N` in the scenario file toward 100,000 or 1,000,000 and re-measure aggregate
   env-steps/second — does it scale linearly? At what `N` does GPU memory (8 GiB on the reference card)
   become the limit, and how would you compute that limit from `FarmBuffers`' per-environment byte count?
5. **Add a second robot to the ladder.** Generalize `cartpole_deriv`/`rk4_step` to a different 4-state
   (or larger) plant — nothing else in the farm (SoA layout, domain randomization, reset logic) should
   need to change; that invariance is the whole point of the pattern (THEORY.md §real-world's closing note).

## Limitations & honesty

- **No contacts, no articulation.** The cart-pole never touches anything but its own frictionless
  track; a real Isaac-Gym-scale farm simulates contact-rich, multi-link robots (THEORY.md §real-world
  names what that adds: LCP/impulse solvers, Featherstone ABA/RNEA).
- **The controller is fixed, not learned.** This project stress-TESTS a hand-derived gain vector across
  a randomized population; it does not train anything. 12.06 is where the training algorithm lives —
  everything here is designed so that project can drop in unchanged (THEORY.md §real-world).
- **Domain randomization here is narrow: only mass and length.** Real sim-to-real work randomizes
  friction, sensor noise, latency, visual appearance (for vision-based policies), and more —
  PRACTICE.md §3 is explicit that this project's randomization narrows, but does not close, the
  sim-to-real gap.
- **The `reset_count` result is "boringly robust" — and that is reported honestly, not massaged.** At
  the committed randomization ranges, every one of 10,000 environments resets exactly 5 times (the
  provable minimum); the controller never actually fails. Exercise 2 shows how to find the boundary
  where it does. A less honest write-up would have widened the ranges until something "interesting"
  happened and called that the headline result — this one reports what was actually measured at the
  documented, defensible ranges.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), kernel-only where
  labeled; the aggregate throughput figure varies noticeably run to run (measured range on the
  reference machine: roughly 6–8.5 billion env-steps/second) because the farm run is only ~1–2 ms of
  wall clock, comparable to normal system timing jitter at that scale.
- **Sim-validated only (CLAUDE.md §1):** this project's direct output (a fixed gain vector, some CSV
  metrics) never commands real hardware, and nothing here is validated against a physical robot —
  everything ran only in simulation. Any project downstream that DOES command hardware using a policy
  developed through this pattern inherits the full CLAUDE.md §1 caveat and the testing ladder in
  PRACTICE.md §3, unconditionally.
