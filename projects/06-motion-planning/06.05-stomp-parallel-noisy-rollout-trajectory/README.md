# 06.05 — STOMP: parallel noisy-rollout trajectory optimization (born for GPU)

**Difficulty:** ★ beginner · **Domain:** 6. Motion Planning

> Catalog bullet (source of truth, verbatim): `★ STOMP: parallel noisy-rollout trajectory optimization (born for GPU)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**STOMP (Stochastic Trajectory Optimization for Motion Planning) plans by sampling.** Given a start,
a goal, and a field of obstacle costs, it holds a nominal trajectory — here **64 interior waypoints**
of a 2-D point robot between a fixed start and goal — and, each iteration, samples **1024 noisy
variations** of it, scores every one against the obstacle field on the GPU, and nudges *each waypoint
separately* toward wherever the good samples went. No gradients of the cost are needed: it works on a
non-convex, obstacle-carved landscape where a straight line drives through three obstacles, and turns
it into a smooth, collision-free route in ~16 iterations. The catalog calls STOMP "born for GPU" for
one reason — the K noisy rollouts are completely independent, so one GPU thread per rollout is the
natural mapping (the **same pattern 08.01's MPPI uses**; STOMP is its planning cousin). This project
builds the complete loop and writes a picture of the result you can open. The demo produces
`demo/out/trajectory.csv` (the final path) and `demo/out/costfield.pgm` (the cost field with the path
burned in). Everything is implemented; no component is documented-only.

## What this computes & why the GPU helps

Per iteration: K=1024 noisy trajectories × (N+1)=65 segments × kSegSamples=8 field lookups of pure,
independent scoring — then a cheap per-waypoint softmin blend on the host.

- **Pattern:** batched sampling — one thread = one noisy rollout (a whole candidate path scored in
  registers); zero interaction between rollouts, by construction.
- **Measured reality (this machine, RTX 2080 SUPER):** the K-rollout scoring set takes ~0.6–0.7 ms of
  GPU kernel time vs ~6.3 ms on one CPU core (~9× — a teaching artifact, not a benchmark).
- **Layout lesson applied:** the noise arrays are stored **transposed** (`eps[j*K+k]`) so each
  waypoint's 32 warp reads are coalesced — the same fix 08.01 and 33.01 teach, applied here from the
  start (explained in [`src/kernels.cuh`](src/kernels.cuh)).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)). This is the exact chain in SYSTEM_DESIGN §4.1: a distance/obstacle
field feeds STOMP, which feeds a tracking controller.

- **Stack position:** the **local/short-horizon planner** — it turns a route and a cost map into an
  actual trajectory for the controller to track.
- **Upstream inputs:** an obstacle-cost field (message shape: a `costmap`/distance field — here built
  on the host from the committed obstacle spec; on a real robot it comes from **07.09 jump-flooding
  SDF** or a `nav2` costmap fused from the map), plus start and goal poses (from the global planner /
  behavior layer, shape `PoseStamped`).
- **Downstream consumers:** a trajectory follower / tracking controller — the emitted path (shape
  `nav_msgs/Path` or a `JointTrajectory`) is tracked by an MPC or PID+feedforward controller; it
  conceptually chains straight into **08.01 MPPI** (SYSTEM_DESIGN §4.1) or a simpler pure-pursuit tracker.
- **Rate / latency budget:** a local re-planner replans at **10–50 Hz** (SYSTEM_DESIGN §1.1 / §4.1
  quote this band for the planner box). This demo's per-iteration GPU scoring is ~0.6 ms, so even a
  full 16-iteration re-plan (~10 ms) fits inside a 10 Hz budget with headroom; a fielded planner would
  warm-start from the previous plan and run far fewer iterations per tick.
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN §2.1: domain 06 is its navigation
  planner) and the **6-DoF manipulator work cell** (§2.2: arm motion planning, domains 06/07) — STOMP
  is used in both mobile and manipulator planning.
- **In production:** a shipping stack surrounds this with a global planner (route → lattice/hybrid-A*),
  a proper 3-D or configuration-space collision model, dynamic-obstacle prediction, and a controller
  that tracks the plan; STOMP or a GPU trajectory optimizer (cuRobo) fills the local-planning seat.
- **Owning team:** the **controls & autonomy** team (SYSTEM_DESIGN §5.1 org map: domains 04–09/23),
  adjacent to perception (which supplies the cost map) and to the safety team (which bounds the motion).

## The algorithm in brief

- **Smooth-noise sampling** — perturb the trajectory with `eps = M z`, where `z` is per-waypoint white
  noise and `M` is derived from `R^-1`, `R = A^T A` the finite-difference acceleration matrix. `M`'s
  columns are smooth basis functions, so the perturbed paths are smooth and their ends barely move. →
  [THEORY.md](THEORY.md) §The math (this is the single idea that most distinguishes STOMP).
- **GPU scoring** — one thread integrates one noisy path's obstacle cost along its segments (dense
  enough to catch a thin obstacle) plus a smoothness term. → THEORY §The GPU mapping.
- **Per-waypoint softmin update** — at *each* waypoint, weight the K perturbations by their local cost
  and blend; smooth the whole update through `M`. This per-waypoint weighting is STOMP's signature —
  contrast MPPI's single per-**whole-trajectory** softmin (08.01). → THEORY §The algorithm.
- **Iterate to a plateau** — stop when the cost stops improving (here ~16 iterations). Start and goal
  never move.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/stomp-parallel-noisy-rollout-trajectory.sln`](build/stomp-parallel-noisy-rollout-trajectory.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/stomp-parallel-noisy-rollout-trajectory.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only (the
noise is host-generated for reproducibility; the smoothing matrix M is inverted with plain host C++;
the obstacle scoring is a hand-rolled kernel by design).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including the
**cost-field image and trajectory to plot**.

## Data

The committed sample is a **scenario**, not recordings: `data/sample/obstacle_scenario.csv`
(~0.6 KiB, synthetic) — a 10×10 m map, a start at (1,1), a goal at (9,9), and 3 circular obstacles
straddling the diagonal so the straight-line initialization collides. The obstacle-cost *field* is
inflated from that spec on the host at load time; noise and rollouts are generated in-demo from
documented fixed seeds. No public dataset applies; `scripts/download_data.ps1` is an honest no-op.
Details: [`data/README.md`](data/README.md).

## Expected output

Six stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY: PASS`, `ARTIFACT:`, `RESULT: PASS` —
checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Two distinct
verifications:

1. **The §5 GPU-vs-CPU gate:** iteration 0's 1024 rollout costs computed by the kernel and by
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp) must agree within rel 1e-3 (measured worst on this
   machine: **2.2e-07** in Release, exactly **0** in Debug where `-G` disables FMA fusion).
2. **The end-to-end verdict:** the final trajectory must be collision-free with margin (max field value
   along it below 25, i.e. ≥0.30 m clear of every obstacle) **and** its total cost must fall below 5%
   of the straight-line value. Measured: max field along the final path **0.000** (fully clear, ≥0.6 m
   from every obstacle) and total cost **591.2 → 0.0013** (the collision cost is eliminated entirely).
   The thresholds sit far from the achieved behavior so platform low-bit differences cannot flip the verdict.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole STOMP loop in plain sight: build the cost field → precompute
   `M` → sample smooth noise → GPU score → per-waypoint softmin update → iterate; plus the verify stage,
   the collision/cost verdict, and the artifact writers.
2. [`src/kernels.cuh`](src/kernels.cuh) — the trajectory layout, the cost-field/grid convention, and the
   transposed noise-layout decision (the project's one-place contracts).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the scoring oracle twin *and* the single-path
   evaluator used for convergence and the final verdict.
4. [`src/kernels.cu`](src/kernels.cu) — the heart: the bilinear field sampler, the segment cost integral,
   and the one-thread-per-rollout scoring kernel. The single most interesting thing: how the kernel
   outputs a **per-waypoint** cost array, which is what lets STOMP's update differ from MPPI's.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Kalakrishnan, Chitta, Theodorou, Pastor & Schaal (2011), "STOMP: Stochastic Trajectory Optimization
  for Motion Planning"** — the paper (the per-timestep probability update and the `M = R^-1` smoothing
  matrix are theirs; our THEORY.md walks the construction, they own the derivation).
- **MoveIt STOMP planner plugin** — STOMP as a production planning plugin for real arms; see how it
  plugs into `planning_interface` and consumes a full collision world instead of a 2-D field.
- **CHOMP (Covariant Hamiltonian Optimization for Motion Planning)** — the *gradient-based* cousin:
  same obstacle+smoothness objective, but it descends the gradient of a differentiable cost/SDF. Learn
  the trade — CHOMP is faster when the cost is smooth and differentiable, STOMP is derivative-free and
  explores better around thin obstacles where gradients mislead (THEORY §real-world).
- **OMPL (Open Motion Planning Library)** — the sampling-based *planning* alternative (RRT*, PRM); know
  when you want a planner that finds *a* path vs. an optimizer that *improves* one.
- **cuRobo (NVIDIA)** — a modern GPU trajectory optimizer for manipulators; this project is the
  one-screen teaching version of what cuRobo does at scale over full robot geometry.

## Exercises

1. **Look at the artifact:** open `demo/out/costfield.pgm` (any image viewer) and plot
   `demo/out/trajectory.csv` — watch the path bend around the obstacle cluster. Then move an obstacle in
   `scripts/make_synthetic.py` (seed unchanged) and see the route change.
2. **Break the temperature:** raise/lower `kSensitivity` (the softmin `h`) and explain both failure
   modes — too greedy (chases one lucky sample) vs. too flat (averages toward no motion).
3. **Jagged vs. smooth noise:** replace `eps = M z` with plain per-waypoint noise `eps = z` (skip the
   matvec) and watch the perturbed trajectories — and the result — turn jagged. This is the whole point
   of `M`.
4. **On-device noise:** generate the noise on the GPU with cuRAND (and do the `M z` mix on-device) to
   remove the per-iteration upload; quantify what it was costing and document the determinism trade.
5. **Climb the dimensions:** extend the waypoints from 2-D to a 6-DoF arm's joint space — nothing in the
   loop's *shape* changes; only the cost (a real collision check) and the dimension grow. That
   invariance is why STOMP scales to real robots.

## Limitations & honesty

- **2-D point robot, analytic obstacles** — a real planner works in configuration space against a full
  collision model; here the "robot" is a point and obstacles are circles inflated into a cost field.
  The math (smooth noise, per-waypoint update, `M`-smoothing) is exactly the real thing; the geometry is
  the teaching simplification (THEORY §real-world).
- **Self-contained cost field** — the field is inflated on the host from the committed obstacle spec, not
  taken from project 07.09 (self-containment rule, §4). On a real robot 07.09's jump-flooding SDF is
  where this field would come from.
- **Host-side update + per-iteration noise upload** — didactic transparency over peak performance;
  Exercise 4 moves both onto the GPU and the headers say what production stacks do.
- **The final cost is essentially zero** — the straight-line init drives through all three obstacles, so
  almost the entire cost is collision cost, which STOMP eliminates; we certify the verdict as a stable
  "final < 5% of initial" rather than a fragile astronomically-large ratio (see the note in
  [`src/main.cu`](src/main.cu)).
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **Sim-validated only, and it matters here (CLAUDE.md §1):** this project's output is a *trajectory* — a
  motion command in the making. Everything ran only against a synthetic map; nothing is safety-certified,
  no real-robot claim is made, and any hardware use would demand the full testing ladder (PRACTICE §3)
  plus an independent safety envelope.
