# 23.01 — GPU costmaps: inflation, raytrace clearing, multi-layer fusion

**Difficulty:** ★ beginner · **Domain:** 23. Navigation Stack (Mobile Robots)

> Catalog bullet (source of truth, verbatim): `★ GPU costmaps: inflation, raytrace clearing, multi-layer fusion`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**This is the navigation stack's centerpiece: the layer that turns "what the sensors just saw" into
"where it is safe to drive."** A simulated warehouse AMR crosses a 12.8 m x 12.8 m room dotted with
four pillar obstacles, replanning ten times a second. Every tick: a simulated 360-beam LiDAR scan
feeds a three-kernel GPU pipeline that builds a fused costmap (raytrace mark/clear -> bounded-radius
inflation -> per-cell max fusion, mirroring Nav2's `costmap_2d` layer stack almost exactly), then a
fourth GPU kernel scores 4096 candidate velocity pairs against that costmap using the Dynamic Window
Approach (DWA), and the best safe one drives a simulated differential-drive robot one step closer to
the goal. The catalog bullet names three costmap components (inflation, raytrace clearing,
multi-layer fusion); this project implements all three **and** wires them into a real consumer — a
closed navigation loop — rather than stopping at "the costmap looks right in isolation," because a
costmap that nobody drives against does not teach the reason costmaps exist.

## What this computes & why the GPU helps

Per 10 Hz control tick: a 256x256 (65,536-cell) costmap rebuild from a 360-beam scan, then 4096
independent 2-second trajectory rollouts scored against that costmap — three genuinely different GPU
access patterns in one pipeline, not one pattern repeated:

- **Raytrace (map/scatter with a race):** one thread per LiDAR **beam** (360 threads) — a Bresenham
  grid walk that marks the obstacle it hit and clears everything in front of it. Different beams
  legitimately race for the same cell near an obstacle's silhouette; `atomicMax` resolves the race
  deterministically in the safe direction (mark always beats clear) rather than avoiding it.
- **Inflation (bounded stencil / gather):** one thread per **cell** (65,536 threads) — a
  `(2R+1)^2` bounded-radius gather that finds the nearest lethal cell and writes a distance-decay
  cost. Self-contained brute force (no jump-flooding dependency, per this project's scope), and
  exact **integer** arithmetic throughout — see [`src/kernels.cu`](src/kernels.cu) for why that
  choice makes the whole costmap byte-exact against its CPU oracle.
- **DWA scoring (sampling rollout, 08.01's pattern reused for scoring):** one thread per **(v,w)
  sample** (4096 threads) — each thread forward-simulates one candidate arc independently, exactly
  MPPI's "one thread, one whole simulated future" mapping, now producing a score to ARGMIN instead of
  a control blend to apply.
- **Measured reality (RTX 2080 SUPER, one machine, teaching artifact):** one full costmap update
  (raytrace + inflation + fusion) takes ~0.7-0.8 ms of GPU kernel time vs ~24 ms on one CPU core
  (~30-33x); the 4096-sample DWA pass takes ~0.2-0.25 ms vs ~2.2 ms CPU (~9-11x). Both numbers vary
  run to run (a single-shot teaching artifact, never a benchmark claim) — see
  [`demo/expected_output.txt`](demo/expected_output.txt)'s header for the exact measurement conditions.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the boundary between **state estimation/world model** and **planning** —
  costmaps are SYSTEM_DESIGN.md §1's own example of where perception output becomes a planning
  input (`[04,05 ->] state estimation/mapping` feeds `[23,06 ->] navigation stack` in the warehouse
  AMR block diagram, §2.1); DWA then spans planning and control, emitting the actuation command
  directly (the same dual role 08.01's MPPI plays for a different plant).
- **Upstream inputs:** a `sensor_msgs/LaserScan`-shaped scan (this project simulates it; 04.01/11.01
  are where a fuller sensor-noise model would live), the current pose (`nav_msgs/Odometry`-shaped —
  here the plant's own `[x,y,theta]`), and a static `nav_msgs/OccupancyGrid` (a prebuilt map — here
  `data/sample/world_map.pgm`, standing in for what 05.01 TSDF-fusion-style mapping would hand off in
  a fuller chain).
- **Downstream consumers:** the base controller — a `geometry_msgs/Twist` command
  (`linear.x = v, angular.z = w`) applied to the differential-drive plant every tick; on a real AMR
  this crosses into the wheel drives' velocity loops (SYSTEM_DESIGN.md §6.1's actuation chain).
- **Rate / latency budget:** SYSTEM_DESIGN.md §1.1 puts the costmap update at **5-20 Hz** and the
  local planner at **10-50 Hz**; this demo runs BOTH at 10 Hz (`kDtControl = 0.1 s`) with ~0.3-1.1 ms
  of combined GPU kernel time per tick measured — one to two orders of magnitude of headroom inside
  the 100 ms budget, most of which goes to the (host-side, deliberately simple) LiDAR scan simulation
  and the two small host<->device transfers, not the GPU compute itself.
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN.md §2.1) by name — this project
  literally implements that block diagram's `[23,06 ->] NAVIGATION STACK` box, and belongs to §4.1's
  Chain A (the AMR spine), sitting between `[05.01 TSDF fusion]`/`[07.09 jump-flooding distance
  field]` upstream and `[08.01 MPPI]`/wheel controllers downstream — this project's own DWA plays the
  role Chain A's diagram assigns to `[06.05 STOMP]`, the local-trajectory step.
- **In production:** Nav2's `costmap_2d` (`StaticLayer` + `ObstacleLayer` + `InflationLayer`,
  updated incrementally, not fully recomputed — THEORY.md §real-world discusses the trade) feeding
  `DWB` (Nav2's DWA-family local planner), both running on the robot's onboard compute continuously.
- **Owning team:** controls/autonomy (SYSTEM_DESIGN.md §5.1) — the navigation stack is this team's
  most-shipped deliverable on a wheeled robot; perception hands it maps and scans, safety/HRI teams
  watch its output (PRACTICE.md §3-4).

## The algorithm in brief

- **Raytrace mark/clear** — Bresenham-walk each LiDAR beam through the grid, marking its endpoint
  lethal (if it hit something) and clearing every cell in front of it; resolved deterministically
  under `atomicMax` where beams disagree. -> [THEORY.md](THEORY.md) §The GPU mapping.
- **Bounded-radius inflation** — for every cell, find the nearest lethal cell within
  `kInflationRadiusCells` (~10 cells / 0.5 m) and decay a cost with squared distance (exact integer
  arithmetic, not `exp()`/`sqrt()`). -> THEORY §Numerical considerations.
- **Per-cell max fusion** — `master = max(static, obstacle, inflation)`, the scaffold's SAXPY map
  pattern doing real work. -> THEORY §The GPU mapping.
- **Dynamic Window Approach** — sample the `(v,w)` window reachable within one control period given
  acceleration limits, forward-simulate each 2 s under RK4, score obstacle cost + goal progress +
  heading alignment, argmin over the admissible set. -> THEORY §The math (including DWA's known
  local-minima failure mode and how this project's scenario sidesteps it honestly).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/gpu-costmaps.sln`](build/gpu-costmaps.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/gpu-costmaps.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuBLAS/cuFFT/Thrust: every kernel (raytrace, inflation, fusion, DWA scoring) is hand-rolled.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
**how to view the two artifacts** (a viewable costmap image and a plottable driven-path CSV).

## Data

The committed sample is **synthetic**, generated (and solvability-VERIFIED, not just asserted) by
[`scripts/make_synthetic.py`](scripts/make_synthetic.py): `data/sample/world_map.pgm` (a 256x256
occupancy grid — border walls plus four pillar obstacles) and `data/sample/scenario.csv` (start pose,
goal, step cap). No public dataset applies — a solvable synthetic room with full ground truth teaches
the costmap+DWA pipeline at least as well as any real building scan would, without a license question
attached; `scripts/download_data.ps1` is an honest, documented no-op. Details, checksums, and field
documentation: [`data/README.md`](data/README.md).

## Expected output

Nine stable lines — banner, `PROBLEM:`, `MAP:`, `SCENARIO:`, two `VERIFY` lines, two `ARTIFACT:`
lines, and `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). **Two independent verifications**, both
described in full in [`THEORY.md`](THEORY.md) §How we verify correctness:

1. **Costmap gate — byte-exact.** One full GPU costmap update cycle (raytrace + inflation + fusion)
   compared cell-by-cell against the plain-C++ CPU oracle. Measured: **0/65,536 cells differ.** Every
   layer in this pipeline is pure integer arithmetic (a deliberate design choice — THEORY.md
   explains why), so "byte-exact" is not an approximation of the real bar, it IS the bar.
2. **DWA gate — tolerance.** One scoring pass over the tick-0 dynamic window (4096 trig-heavy
   rollouts) compared against the CPU oracle within relative tolerance 1e-3. Measured worst
   deviation: **2.188e-07** — ~4500x inside the tolerance, the same "the gate has enormous headroom
   against real bugs, and floating-point noise never gets close to it" story 08.01 tells.
3. **Closed-loop success.** The robot must reach the goal (within 0.3 m) inside the 500-step cap
   AND never enter a lethal-cost cell along the driven path. Measured: **goal reached in 288/500
   steps, 0 lethal-cell entries, 0 emergency brakes triggered.**

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the whole project's contract in one file: grid geometry,
   cost semantics (mirroring Nav2's byte convention), the LiDAR scan layout, the `(v,w)` sampling
   window layout, and every tuning constant. Read this FIRST — everything else assumes it.
2. [`src/main.cu`](src/main.cu) — the closed loop in plain sight: sense -> GPU costmap -> GPU DWA ->
   pick -> drive -> repeat, plus the two-part verify stage and the strict PGM/CSV loaders.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: four kernels, three GPU patterns. The single most
   interesting thing to read: `raytrace_kernel`'s file-header essay working through the mark/clear
   race step by step and showing why `atomicMax` — not a mutex, not avoidance — is the right fix.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the oracle twins (note `raytrace_beam_cpu`'s
   use of `std::max` instead of a plain assignment — it computes the SAME reduction the GPU's atomics
   compute, which is what makes the byte-exact comparison meaningful) and the diff-drive plant.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Nav2's `costmap_2d`** — the production version of this project's layer stack (`StaticLayer`,
  `ObstacleLayer`, `InflationLayer`, `VoxelLayer` for 3D), updated incrementally in bounded windows
  rather than fully recomputed every tick (THEORY.md §real-world contrasts this with our approach).
- **Nav2's `DWB` (Dynamic Window local planner)** — a pluggable, production DWA/TEB-family local
  planner; compare its critic-plugin scoring architecture with this project's fixed weighted sum.
- **Fox, Burgard & Thrun (1997), "The Dynamic Window Approach to Collision Avoidance"** — the
  original DWA paper; THEORY.md's admissibility discussion cites it directly.
- **Bresenham (1965), "Algorithm for computer control of a digital plotter"** — the integer line
  algorithm `raytrace_kernel`/`raytrace_beam_cpu` both implement, unchanged in shape 60 years later.
- **OpenCV / PCL raycasting utilities and ROS's `laser_geometry`** — the production tooling for
  turning a real LiDAR message into the grid operations this project hand-rolls for teaching.
- **08.01's MPPI controller** — the direct ancestor of this project's `dwa_score_kernel`: same
  "one thread, one simulated future" mapping, here reused for scoring/argmin instead of a softmin
  blend. Read it alongside this project's kernel to see how far one pattern generalizes.

## Exercises

1. **Plot the artifacts.** `demo/out/path.csv`'s `x_m` vs `y_m` over `demo/out/costmap.pgm` (as a
   background image) — see the robot's actual line around each pillar and how close it gets to the
   inflation gradient without ever touching lethal.
2. **Break the dynamic window.** Set `kAccelV`/`kAccelW` in `src/kernels.cuh` an order of magnitude
   smaller (a much less nimble robot) and explain the effect on `[info] emergency brakes:` and on how
   tightly the path hugs the pillars.
3. **Feel DWA's local-minimum failure mode.** Replace the four pillars in
   `scripts/make_synthetic.py`'s `build_world()` with the earlier full-width alternating-gap "slalom"
   layout described in that function's own comment, regenerate, and watch the robot get pinned
   against a wall face short of the step cap — then read THEORY.md §The math to see why, and why a
   real stack puts a GLOBAL planner in front of DWA specifically to prevent this.
4. **Pack the obstacle layer.** `obstacle_layer` uses one `int` (4 bytes) per cell because CUDA has
   no native 1-byte `atomicMax` (kernels.cu explains the trade). Implement a packed 1-byte-per-cell
   version using a 32-bit compare-and-swap loop on the containing word, and measure the memory
   traffic and kernel-time difference on the inflation/fusion kernels that read it afterward.
5. **Close the form.** `unicycle_rk4_step` numerically integrates a constant-`(v,w)` unicycle, which
   has an exact closed-form circular-arc solution. Derive it, substitute it into the DWA rollout, and
   measure both the score difference against the RK4 version and the kernel-time change.

## Limitations & honesty

- **No global planner.** This project's DWA is purely REACTIVE — there is no A*/Dijkstra route search
  above it. The committed scenario's obstacles are deliberately small, isolated pillars (not a
  full-width maze) specifically so a reactive planner can solve it without hitting DWA's well-known
  local-minima failure mode; THEORY.md discusses that failure mode honestly, including a version of
  this world (Exercise 3) that reliably triggers it.
- **Obstacle layer is redundant with the static layer in this demo.** The world is fully static and
  known a priori (`data/sample/world_map.pgm` IS the ground truth), so the STATIC layer alone would
  already guarantee safety. The OBSTACLE layer is still rebuilt from a simulated scan every tick to
  teach the real GPU raytrace-mark/clear pattern (and its honest data race) production costmaps need
  for objects a static map genuinely cannot know about — this demo just never exercises that case.
- **Point-robot-plus-inflation, not a real footprint.** Safety margins come entirely from the
  inflation radius, not from sweeping an actual robot footprint polygon through the costmap (Nav2
  supports both; this project teaches only the inflation-radius approximation).
- **Forward-only, no reverse.** `kVMin = 0` — the robot cannot back up. Combined with the
  reactive-only scope above, this is why the scenario avoids anything resembling a dead end.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **Sim-validated only (CLAUDE.md §1):** this project's output is a velocity command — the same
  archetype of code 08.01 flags at full strength. Everything here ran only against the simulated
  plant and a simulated LiDAR; nothing is safety-certified, no real-robot claim is made, and any
  hardware use would demand the full testing ladder (PRACTICE.md §3) plus an independent safety
  envelope (a certified safety-rated scanner, not this simulated one — PRACTICE.md §3 is explicit
  about that distinction).
