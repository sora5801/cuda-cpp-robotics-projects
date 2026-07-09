# 09.01 — Batched forward kinematics (10⁵ configurations — the foundation for everything above)

**Difficulty:** ★ beginner · **Domain:** 9. Robot Dynamics & Kinematics

> Catalog bullet (source of truth, verbatim): `★ Batched forward kinematics (10⁵ configurations — the foundation for everything above)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Forward kinematics (FK) is the "where is the hand?" function: given a robot's joint angles, compose
the chain of rigid transforms to get the end-effector pose. This project computes FK for a 6-DoF
serial arm **for 200,000 joint configurations at once** — one GPU thread per configuration, the
robot model broadcast from constant memory — and verifies every pose against a plain-C++ CPU
oracle. FK is the catalog's own words: *the foundation for everything above* — batched IK, grasp
reachability, sampling controllers all evaluate FK in bulk. After studying it you will know how
rigid transforms compose (`p ← p + R·t; R ← R·R_fix·Rot(axis, q)`), what Rodrigues' formula and a
numerically-stable matrix→quaternion conversion look like in real code, and why GPU constant memory
exists. The demo prints a PASS/FAIL verdict plus honest timing lines.

## What this computes & why the GPU helps

One FK evaluation is ~200 flops — trivial. The bottleneck is that sampling-based robotics never
wants one: an IK solver with random restarts (09.05) runs FK per seed per iteration; a
grasp-reachability ranker scores 10⁵ candidate configurations; an MPPI controller (08.01) evaluates
pose costs across thousands of rollouts.

- **Pattern:** batched map — one thread = one configuration's entire FK chain, computed in
  registers; zero inter-thread communication.
- **The structural fact:** the chain itself is inherently *sequential* (joint *j* needs joint
  *j−1*'s frame), so parallelism lives **across the batch**, never along the chain.
- **New GPU concept vs 33.01:** the robot model is identical for every thread → it rides in
  `__constant__` memory, whose cache *broadcasts* a uniform read to a whole warp in one
  transaction ([THEORY.md](THEORY.md) §The GPU mapping).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the kinematics core of the **planning/control layers** — FK is the inner-loop
  primitive of manipulation planning, IK, and reachability analysis (domain 09 feeds domains 06,
  08, 19).
- **Upstream inputs:** a robot description (the URDF/SRDF in a real stack — here a 10-float-per-
  joint model, layout in [`src/kernels.cuh`](src/kernels.cuh)) and batches of candidate joint
  configurations `q` (shaped like arrays of ROS 2 `sensor_msgs/JointState.position`).
- **Downstream consumers:** message-shaped poses (`geometry_msgs/Pose`: position + (w,x,y,z)
  quaternion) consumed by IK error terms (09.05), grasp-reachability ranking (19.08), workspace
  atlases (09.09), and pose-cost terms inside sampling planners/controllers (06.x, 08.01).
- **Rate / latency budget:** inside a 10–50 Hz planner tick or an interactive IK query
  (SYSTEM_DESIGN item 1); the measured ~0.2 ms kernel for 2×10⁵ configurations means FK is
  effectively *free* at those rates — which is exactly what makes massive sampling strategies
  viable.
- **Reference robot(s):** the **6-DoF manipulator work cell** directly (its IK/planning chain in
  SYSTEM_DESIGN's composition map); the **quadruped** analogously (leg-chain FK for foothold and
  whole-body control, domains 13/09).
- **In production:** cuRobo's batched kinematics, Isaac Lab / physics-engine FK layers, or CPU
  Pinocchio/KDL/MoveIt when batch sizes are small; fused into bigger kernels (dynamics, rollout
  costs) exactly as taught here.
- **Owning team:** motion planning / controls within an autonomy group; the robot *model* itself is
  owned jointly with mechanical engineering (they produce the geometry this code consumes) —
  SYSTEM_DESIGN item 5.

## The algorithm in brief

- **Chain composition** — per joint: `p ← p + R·t_j`, `R ← R·R_fix_j`, `R ← R·Rot(axis_j, q_j)`;
  end pose = final `(R, p)`. → [THEORY.md](THEORY.md) §The algorithm.
- **Rodrigues' formula** — axis-angle → rotation matrix; the workhorse of revolute joints.
- **Shepperd's method** — numerically-stable rotation-matrix → quaternion (largest-divisor branch
  selection), output in the repo's **(w,x,y,z)** order, double-cover handled by consumers.
- **Constant-memory model** — uploaded once via `set_robot_model()` (the URDF-at-startup pattern).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/batched-forward-kinematics.sln`](build/batched-forward-kinematics.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/batched-forward-kinematics.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Fully **synthetic** (labeled so everywhere): `data/sample/fk_sample.csv` (~5.3 KiB, committed) — a
generic 6-DoF anthropomorphic arm (an archetype invented for teaching, **no vendor's product**)
plus 64 joint configurations, generated by
[`scripts/make_synthetic.py`](scripts/make_synthetic.py) with seed 42, byte-identical on
regeneration. No public dataset applies (a robot model is just numbers), so
`scripts/download_data.ps1` is an honest no-op. Format, frames, and checksum:
[`data/README.md`](data/README.md).

## Expected output

Seven stable lines — banner, `PROBLEM:`, `MODEL:`, `SAMPLE:`, `SAMPLE RESULT: PASS`, `BATCH:`,
`RESULT: PASS` — checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt)
(`[info]`/`[time]` lines deliberately unchecked). Verification: every pose is computed twice — GPU
vs the oracle in [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — and compared within **1e-4 m**
(position) and **1e-4 per quaternion component after hemisphere alignment** (q and −q encode the
same rotation; the comparator aligns via the dot-product sign before differencing). A quaternion
**norm invariant** (‖q‖ = 1 ± 1e-4) is checked on both paths as a self-consistency gate. Measured
worst deviations on the reference machine: ~9e-08 m and ~1.8e-07 — three orders inside tolerance.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — arguments, the strict sample loader, model validation, the two
   verification stages, the hemisphere-aware comparator, the output contract.
2. [`src/kernels.cuh`](src/kernels.cuh) — the **layout contracts** (model rows, configurations,
   message-shaped poses) — the one place they are defined.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the oracle twin.
4. [`src/kernels.cu`](src/kernels.cu) — the heart: constant-memory model, Rodrigues, Shepperd, and
   the FK kernel. The single most interesting thing: how `__constant__ float c_model[]` turns "every
   thread reads the same robot" from a bandwidth problem into a broadcast.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **cuRobo** (NVIDIA) — production massively-batched kinematics/IK/planning; its FK is this
  project's pattern industrialized (fused, SoA layouts, thousands of arms at once).
- **Pinocchio** — the reference CPU rigid-body kinematics/dynamics library; compare its spatial-
  algebra formulation with our explicit (R, p) composition.
- **KDL / MoveIt** — the classic ROS kinematics stack; what "one FK at a time" production code
  looks like, and why batching required a rethink.
- **Isaac Lab / PhysX articulations** — FK embedded inside GPU physics stepping (the fusion story).
- **Featherstone, *Rigid Body Dynamics Algorithms*** — the book behind spatial transforms; the
  bridge to 09.03 (batched ABA/RNEA dynamics).
- **Shepperd (1978), "Quaternion from rotation matrix"** — the four-branch stable conversion
  implemented here.

## Exercises

1. **Add a tool transform:** extend the model with a fixed `T_flange_tool` applied after the last
   joint (one more model row with a fixed identity "joint") and update the sample — the cheapest
   way to feel how the layout contract propagates.
2. **All link poses:** write every intermediate `T_base_linkj` (not just the end-effector) into a
   `count × nj × 7` buffer — the shape collision checkers (07.x) need. Watch the output-write cost
   grow; explain it with the coalescing story.
3. **Prismatic joints:** add a joint-type flag to the model row (revolute/prismatic — translation
   along the axis instead of rotation about it) and handle both in the kernel; note the warp-
   divergence question and why it is harmless for a *uniform* model (every thread takes the same
   branch at each j).
4. **Batched numerical Jacobians:** central differences over the 6 joints (13 FK evaluations per
   configuration, all batched) — the stepping stone to project 09.02's analytic Jacobians.
5. **Move the model to plain global memory** and measure the difference with Nsight — quantify what
   the constant-cache broadcast actually buys at this size.

## Limitations & honesty

- **Revolute joints only**, single serial chain, no tool transform — the smallest teaching core;
  trees/branches (humanoids) and prismatic joints are Exercises/09.x siblings.
- **FP32** — right for FK at ~1 m scale (deviations ~1e-7 m); long chains or km-scale mechanisms
  would revisit precision.
- **The model is synthetic** — a generic arm archetype, deliberately not any vendor's kinematics;
  loading real URDFs is the production step (cuRobo/Pinocchio do it) and is out of scope here.
- **Single stream, plain transfers, single-shot timings** — the 243× teaching speed-up observed on
  the reference machine is kernel-only vs one CPU core and varies with clocks; never read it as a
  benchmark.
- **Sim-validated only:** FK output here drives comparisons and prints — nothing commands hardware;
  consumers that do (planners/controllers) carry the safety caveat themselves.
