# 22.01 — 100k-agent swarm simulator: flocking, pheromone grids, stigmergy

**Difficulty:** ★ beginner · **Domain:** 22. Multi-Robot Systems & Swarms

> Catalog bullet (source of truth, verbatim): `★ 100k-agent swarm simulator: flocking, pheromone grids, stigmergy`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**100,000 independent agents, one shared GPU grid, and no central planner.** Each agent looks only at
the handful of neighbors within 1 m of it and applies three local rules — separate, align, cohere —
and the flock that emerges is a *global* behavior nobody programmed directly. On top of that, agents
also deposit and sense a **pheromone field**: a 256×256 grid that diffuses and evaporates, the way an
ant trail spreads and fades. Agents steer weakly along its gradient, so the field becomes a second,
indirect communication channel — **stigmergy**, coordination through the shared environment rather
than through messages between agents. A learner who studies this project comes away understanding
*two* GPU patterns at once: the counting-sort spatial grid that turns an O(N²)-impossible neighbor
search into something that runs at 100,000 agents and 20 Hz, and the same 5-point stencil that
appears everywhere from image blur to fluid simulation, here doing double duty as an environmental
memory. The demo prints a GPU-vs-CPU correctness gate, then runs the full 100k-agent, 300-step flock
and writes two heatmap images plus a position snapshot so the emergent flocking is *visible*, not
just asserted.

**Bundled-bullet note (CLAUDE.md §2).** The catalog bullet names three ideas — flocking, pheromone
grids, stigmergy — and they are **not three separate features here but one coherent system**: the
pheromone grid *is* the stigmergy mechanism (agents coordinate indirectly by reading a field other
agents wrote), and flocking is the local rule set that runs alongside it every step. Both are fully
implemented in `src/`. What is **documented-only** (THEORY.md "Where this sits in the real world") is
the wider *family* of stigmergy patterns real swarm-robotics research explores beyond a single-channel
attractive trail: multi-pheromone systems (e.g., separate "food" and "home" trails, ant-colony-style),
repulsive/warning trails, and quorum-sensing aggregation. Those are named and scoped, not built —
README Exercises 4 and 5 point at the concrete next steps.

## What this computes & why the GPU helps

Two computations run every simulated step, and a third — the neighbor search — exists purely to make
the first one affordable:

- **Neighbor search (map + counting sort):** at N = 100,000 agents, testing every pair against the
  1 m interaction radius is 10¹⁰ distance tests *per step* — no processor does that at interactive
  rates. A **uniform grid** with cell size exactly equal to the interaction radius turns this into an
  O(N × ~30) gather: histogram agents into cells (`map`, atomicAdd), exclusive-scan the 65,536-cell
  histogram (`scan`), scatter agent indices into sorted bins (`map`), then each agent inspects only
  its 3×3 neighborhood of cells (a small, *provably complete*, `gather`).
- **The flock step (map):** one GPU thread per agent — gather up to ~30 neighbor candidates, fold them
  into Reynolds' separation/alignment/cohesion sums, sample the pheromone gradient, integrate. Every
  agent's decision is independent of every other agent's *this step*, so one thread per agent
  saturates the GPU with zero inter-thread coordination beyond the read-only neighbor gather.
- **The pheromone step (stencil):** one GPU thread per grid cell — a classic 5-point Laplacian
  (diffuse), an exponential decay (evaporate), and a deposit read from the same histogram the
  neighbor search already built. Structurally identical to a single step of 2-D heat diffusion or
  image blur; 07.09 is the repo's other grid/stencil teaching project.

Measured on the reference machine (RTX 2080 SUPER, sm_75): the flock kernel runs 100,000 agents in
**~0.45 ms**, the pheromone stencil in **~0.014 ms**, and the bin/scan/scatter phase (which includes
two host↔device round trips for the histogram) in **~0.37 ms** — under 1 ms of GPU-side work per
20 Hz control tick, with an order of magnitude of headroom (see [`THEORY.md`](THEORY.md) for the full
timing story and why the CPU oracle takes 1.5+ seconds at a *much smaller* N).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** spans **planning and control** for a *fleet* rather than one robot — each
  agent's flock step is simultaneously its local trajectory decision (where do I go next) and its
  actuation command (its next velocity), the same collapsed plan/control role 08.01's MPPI plays for
  one robot, here run N times per tick, in parallel, plus the cross-cutting **multi-drone
  coordination** box on SYSTEM_DESIGN's quadrotor diagram (§2.4).
- **Upstream inputs:** each agent's own state estimate (position/velocity — message shape
  `nav_msgs/Odometry` or a lighter `geometry_msgs/PoseStamped` + `TwistStamped` pair) and, in place of
  direct inter-agent messages, the *neighbor state it can locally sense* (bearing/range to nearby
  agents — this demo simplifies that to exact global-frame neighbor positions, a scoped idealization
  spelled out in Limitations below) plus the shared pheromone field (a `nav_msgs/OccupancyGrid`-shaped
  scalar field in production terms).
- **Downstream consumers:** the actuation chain of each individual agent (velocity/attitude commands
  to its own flight or drive controller — the same "downstream is a `Twist`" boundary 08.01 crosses);
  at the fleet level, a ground-station monitor or fleet-operations dashboard consumes the aggregate
  flock statistics (mean alignment, density) this demo prints.
- **Rate / latency budget:** this demo runs the full pipeline at a nominal **20 Hz** (`kDt = 0.05 s`),
  inside the local-planner/controller band (SYSTEM_DESIGN §1.1: 10–50 Hz) — realistic for a
  coordination layer that sits above each agent's own faster attitude loop (0.5–1 kHz, off this
  project's scope; SYSTEM_DESIGN §1.1). Measured GPU cost per tick (~0.8 ms total) leaves comfortable
  headroom inside a 50 ms tick budget even before considering that a real fleet's per-agent compute
  would be distributed, not centralized (PRACTICE §3).
- **Reference robot(s):** the **quadrotor** — SYSTEM_DESIGN §2.4 names "multi-drone coordination [22 →]"
  explicitly as a block feeding quadrotor flight control; a ground-robot swarm (warehouse AMR fleet,
  §2.1) is the same coordination-algorithm family at a different physical scale.
- **In production:** real multi-agent collision-avoidance layers more often use ORCA/RVO2
  (reciprocal velocity obstacles — collision-free by construction, unlike boids) or learned
  coordination policies; large-scale crowd/swarm simulation for design and testing runs in NVIDIA
  Warp or Isaac Sim. See README §11 for the specific tools.
- **Owning team:** the flocking/coordination algorithm itself sits with **controls & autonomy**
  (SYSTEM_DESIGN §5.1); SYSTEM_DESIGN's org map maps domain 22 explicitly to **fleet operations**
  ("22 fleet coordination") once a swarm is actually deployed, with simulation & tools owning the
  digital-twin environment this project is a teaching-scale version of (PRACTICE §4).

## The algorithm in brief

- **Uniform-grid counting sort** — histogram (atomicAdd), host exclusive scan, scatter — the
  same three-kernel spatial-binning pattern used in SPH fluid solvers and production perception
  pipelines. → [THEORY.md](THEORY.md) §The algorithm.
- **Reynolds' boids rules** (separation / alignment / cohesion), hat-weighted so every contribution
  goes smoothly to zero at the interaction radius — the numerics choice that makes the GPU-vs-CPU
  gate robust to ulp-level neighbor-set boundary flips. → THEORY §Numerical considerations.
- **Pheromone stencil** — deposit (from the neighbor-search histogram, reused) + 5-point Laplacian
  diffusion + exponential decay, zero-flux boundary. → THEORY §The math.
- **Clamped semi-implicit Euler integration** — acceleration and speed clamps applied by *scaling*
  (direction-preserving), not per-axis clipping — the same "rules propose, the clamp disposes"
  discipline as 08.01's force limit.
- **Lockstep GPU-vs-CPU verification** at a small deterministic N, because flocking is
  chaotic and a free-running comparison would amplify benign floating-point reordering into meters.
  → THEORY §How we verify correctness.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/100k-agent-swarm-simulator.sln`](build/100k-agent-swarm-simulator.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/100k-agent-swarm-simulator.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
exclusive scan runs on the host in plain C++ (a deliberate teaching choice, not a missing dependency —
see README Exercise 3 for the CUB/Thrust alternative); spawning is host-generated xorshift32, not
cuRAND, for cross-run reproducibility.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at — including the two heatmap images worth opening in an image viewer.

## Data

The committed sample is a **scenario**, not recordings: `data/sample/swarm_scenario.csv` (457 bytes,
synthetic, no RNG) — 100,000 agents, 300 steps at 20 Hz, spawn seed 42. The agents themselves (uniform
positions, random headings) are generated in-demo from that seed with the repo's portable xorshift32
generator; correctness comes from the lockstep CPU oracle plus the end-of-run flock statistics, not
from stored ground truth. No public dataset applies — a swarm simulator's "data" is its scenario, the
same precedent 08.01 set for controllers. Details and the checksum: [`data/README.md`](data/README.md).

## Expected output

Six stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY: PASS`, `ARTIFACT:`, `RESULT: PASS` —
checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Two independent
verifications: **(1)** the §5 GPU-vs-CPU lockstep gate — a small deterministic swarm (N = 4,096) run
100 steps in lockstep against [`src/reference_cpu.cpp`](src/reference_cpu.cpp)'s O(N²) brute-force
oracle, position/velocity/pheromone agreeing within absolute tolerance 1e-3 each (measured worst
deviation: ~1.5e-05 m position, ~1.2e-07 m/s velocity, ~1.2e-07 pheromone — roughly 65×–8,000× inside
tolerance); **(2)** the headline check — all 100,000 agents stay bounded inside the 256 m arena, and
the flock's mean local velocity alignment reaches ≥ 0.5 (measured: ~0.97, against a random-start
baseline of ~0). Success thresholds carry wide margins so the documented run-to-run atomic-ordering
ulps ([THEORY.md](THEORY.md) §Numerical considerations) can never flip the verdict.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: scenario load → lockstep verify
   stage → headline 100k-agent run → metrics → PGM/CSV artifacts. Start with the file header comment,
   which states the per-step pipeline order once.
2. [`src/kernels.cuh`](src/kernels.cuh) — the shared contract: agent state layout, arena/grid geometry,
   the bin-layout documentation, and (most important) the **determinism contract** — read this before
   the kernels, it explains why the tolerances in `THEORY.md` are shaped the way they are.
3. [`src/kernels.cu`](src/kernels.cu) — the four GPU kernels: `bin_count_kernel`, `bin_scatter_kernel`,
   `flock_step_kernel` (the heart — read `accumulate_neighbor` and `finish_agent` first, they are
   shared building blocks), and `pheromone_step_kernel`.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the brute-force oracle; a deliberate line-by-line
   twin of `kernels.cu`'s rule math, differing only in *how* neighbors are found (loop vs. grid).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Reynolds (1987), "Flocks, Herds, and Schools: A Distributed Behavioral Model"** — the boids paper;
  the source of the three rules this project implements almost verbatim, minus the hat-weighting
  (a numerics addition explained in THEORY.md).
- **ORCA / RVO2 (van den Berg et al.)** — reciprocal velocity obstacles: the production alternative to
  boids for multi-agent collision *avoidance* specifically, because boids' potential-field rules are
  not collision-free by construction the way ORCA is. Compare its guarantees against this project's
  soft separation rule.
- **NVIDIA Warp / Isaac Sim crowd & swarm simulation** — the differentiable, GPU-native descendant of
  exactly this pattern (uniform-grid neighbor search + per-agent rules) at production scale, used for
  training and testing autonomy policies against simulated crowds/fleets.
- **Bitcraze Crazyflie + Crazyswarm/Crazyswarm2** — the closest real hardware to this demo's spirit: a
  physical swarm of small quadrotors coordinated by a shared ROS 2 stack; compare their per-agent
  message shapes against this project's SoA state layout.
- **Dorigo et al., ant colony optimization & stigmergic swarm robotics (IRIDIA)** — the research
  lineage of the pheromone-grid mechanism: real robot swarms using virtual (radio-broadcast) or
  physical trail markers for indirect coordination, the "documented-only" variants this README's
  bundled-bullet note points at.
- **SPH (smoothed-particle hydrodynamics) neighbor search** — the counting-sort uniform grid in
  `kernels.cu` is the identical data structure fluid solvers (and NVIDIA FleX) use for particle
  neighbor queries; recognizing the pattern here is the transferable lesson.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Tune the flock.** Edit `kWSep`/`kWAli`/`kWCoh` in `kernels.cuh` and rebuild. Push cohesion far
   above alignment and watch the flock collapse into a milling ball instead of a moving stream; push
   separation to zero and watch agents overlap. No code restructuring — just intuition-building.
2. **Unweight the rules** (referenced in `kernels.cu`'s `accumulate_neighbor`). Replace the hat-weighted
   sums with Reynolds' original *unweighted* neighbor averages (every neighbor inside the radius counts
   equally, weight 1). Rebuild and rerun the GPU-vs-CPU gate — does it still pass at the same
   tolerance, or do you need to loosen it? Explain why in terms of the radius-boundary discontinuity
   THEORY.md describes.
3. **Move the scan onto the GPU** (referenced in `main.cu`'s `gpu_step`). Replace the host exclusive
   scan over the 65,536-cell histogram with a Thrust `exclusive_scan` or a CUB `DeviceScan` call, and
   remove the two PCIe round trips it currently costs. Measure the bin-phase time before and after.
4. **Give agents variable pheromone deposits** (referenced in `kernels.cu`'s `pheromone_step_kernel`).
   Right now every agent deposits the same `kDeposit`, so the histogram *is* the deposit map and stays
   bit-exact (integer atomics associate). Make deposit depend on agent state (e.g., speed) — this
   forces a `float atomicAdd` per deposit, which does **not** associate. Measure how much larger the
   pheromone tolerance in the GPU-vs-CPU gate must become, and document why in a comment.
5. **Chase memory locality** (referenced in `kernels.cu`'s `flock_step_kernel`, two parts). First,
   reorder agent state into bin order after each scatter (so neighbor gathers become contiguous
   reads) and measure the flock kernel's speedup. Second, go further: load one block's worth of
   candidate agents into shared memory before the neighbor loop, and measure again. This is the
   project's clearest "GPU memory hierarchy" lesson — quantify it, don't just implement it.

## Limitations & honesty

- **Global, exact neighbor sensing is a simulation convenience, not a deployment architecture.** Every
  agent here reads exact global-frame positions/velocities of its neighbors from a shared grid built
  once per step. A real swarm agent only has local, noisy, range-limited sensing (onboard camera/UWB/
  radio) of nearby teammates — PRACTICE.md §3 spells out what a real per-agent architecture looks like.
- **Synchronous, centralized computation.** All 100,000 agents step in lockstep on one GPU. Real
  swarms are physically distributed and asynchronous — each robot runs its own local step on its own
  clock. This demo is honestly a *simulation/design tool*, not a model of how the compute would be
  deployed (PRACTICE §2–§3).
- **One pheromone channel, one attractive trail.** The catalog's "stigmergy" is scoped here to a
  single deposit-diffuse-decay-gradient loop; multi-channel and repulsive-trail variants are
  documented, not implemented (README overview, Exercise 4, THEORY.md §real world).
- **No obstacle field.** Agents avoid each other and the arena walls; there is no static-obstacle or
  no-fly-zone layer, which any deployed swarm would need (07.09's distance-field machinery is the
  natural extension, left as future work rather than scope creep here).
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), never a benchmark
  claim (CLAUDE.md §12).
- **Sim-validated only, not safety-certified (CLAUDE.md §1):** this project's steering rules could, in
  a real deployment, become a velocity command sent to physical hardware. Everything here ran only
  against the simulated agent model; nothing is safety-certified, and any real-swarm use would demand
  the full staged testing ladder in PRACTICE.md §3 plus an independent per-unit safety envelope
  (geofencing, RTH failsafes) — the repo-wide caveat, restated because this project's output is
  exactly the kind that could command motion.
