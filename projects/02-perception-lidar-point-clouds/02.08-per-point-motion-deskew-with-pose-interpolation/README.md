# 02.08 — Per-point motion deskew with pose interpolation

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Per-point motion deskew with pose interpolation`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A spinning LiDAR does not take a snapshot — it takes ~360 tiny snapshots per sweep, one per azimuth
step, each correct in a *slightly different* sensor pose if the platform is moving. Naively stacking
those raw returns together (exactly what a driver does with no motion compensation) smears the scan: a
flat wall becomes a slanted or thickened blur, and every downstream consumer — registration, mapping,
planning — inherits the distortion. This project builds the fix: **per-point motion deskew**, using
pose interpolation to re-express every point in one common reference frame. Given the platform's
trajectory over the sweep (position + orientation samples), each point's own firing-time pose is
estimated by interpolating between the two bracketing samples — **LERP** for position, **SLERP** for
orientation — and the point is rigidly re-projected into the frame at the sweep's end. The project
teaches this on four synthetic motion cohorts (straight, a constant-yaw-rate arc, an aggressive
sinusoidal yaw wiggle, and a stationary identity control) and measures — never assumes — how much it
matters, and how much the *sampling rate* of the trajectory itself matters. Everything named in the
catalog bullet is implemented: pose interpolation (both LERP and full geodesic SLERP), and the
per-point re-projection it drives.

## What this computes & why the GPU helps

Per point: two pose interpolations (a tiny binary search + LERP + SLERP against a 2- or 21-sample
trajectory) and one rigid transform (a quaternion composition + a vector rotation). Every point's
output depends only on its own timestamp and local coordinates plus one shared reference pose — no
point ever touches another.

- **Pattern:** a pure **map** — the textbook embarrassingly-parallel case (one thread per point, zero
  interaction), the same shape as the repository's very first SAXPY placeholder, applied to real
  quaternion/rigid-body math instead of a scalar multiply-add.
- **Measured reality (this machine, RTX 2080 SUPER):** all 18,426 points across 4 cohorts × 2 regimes
  (8 kernel launches) deskew in a total of **0.11–0.34 ms of GPU kernel time** — a small fraction of
  the 50–100 ms a 10–20 Hz spinning LiDAR allows per sweep (see "System context" below).
- **Where the parallelism buys nothing (and says so):** the pose-interpolation trajectory itself is
  TINY (2 or 21 samples) — the per-point binary search that walks it costs a handful of comparisons,
  nowhere near memory-bound. The real payoff of the GPU here is *fitting comfortably inside the rate
  budget at zero design effort*, not raw throughput — this project is honest about that (see
  `THEORY.md` "The GPU mapping").

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the tail end of the **perception / sensor-driver layer**, immediately after raw
  point extraction and immediately before every geometric consumer — the first stage that must be
  correct or every stage after it inherits a bias no amount of downstream cleverness can undo.
- **Upstream inputs:** raw per-point returns with **per-point timestamps** from the LiDAR driver (the
  message shape [`02.01`](../../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/src/kernels.cuh)'s
  16-beam scan machinery produces, extended here with a firing time per point — PRACTICE.md §1 is
  honest about which real drivers actually expose this); and a **pose trajectory** over the sweep from
  the state estimator — named here by project, not left abstract:
  [`04.01`](../../../04-sensor-fusion-state-estimation/04.01-massive-particle-filter-localization/README.md)
  (particle-filter localization) or
  [`04.03`](../../../04-sensor-fusion-state-estimation/04.03-imu-preintegration-on-gpu-for-batch/README.md)
  (IMU preintegration) are the kind of upstream source a real stack would poll or subscribe to for
  exactly this array of `(t, position, quaternion)` samples.
- **Downstream consumers:** every geometric registration and mapping stage that assumes a point cloud
  shares ONE frame — named explicitly, because an undeskewed input silently poisons all three:
  [`02.06`](../../02.06-icp-point-to-point-point-to-plane-gicp/README.md) ICP (correspondences drift
  toward the smear), [`02.07`](../../02.07-ndt-scan-matching/README.md) NDT (that project's own
  "corridor" degeneracy lesson gets *worse* with smeared walls — a thicker wall is a less-informative
  normal distribution), and [`05.01`](../../../05-slam-mapping-localization/05.01-tsdf-fusion-kinectfusion/README.md)
  TSDF mapping (a smeared input never integrates into a clean surface, however many frames you fuse).
- **Rate / latency budget:** a spinning LiDAR sweeps at 10–20 Hz (SYSTEM_DESIGN item 1), so the WHOLE
  deskew pass must complete well inside 50–100 ms; measured here: well under 1 ms for the demo's point
  counts (see "What this computes" above) — orders of magnitude of headroom for a real full-density
  sweep (hundreds of thousands of points).
- **Reference robot(s):** the **warehouse AMR** and the **autonomous-vehicle stack** (SYSTEM_DESIGN §2)
  most directly — both carry a spinning LiDAR as a primary exteroceptive sensor and both feed it
  straight into registration/mapping.
- **In production:** a dedicated deskew/undistortion node (or a library call inside the driver itself)
  runs this every sweep, before anything else touches the cloud; see "Where this sits in the real
  world" in `THEORY.md` for how LOAM-lineage stacks and tightly-coupled LIO-SAM/FAST-LIO differ from
  this project's decoupled (deskew-then-estimate) shape.
- **Owning team:** perception/localization (SYSTEM_DESIGN item 5) — this is exactly the kind of
  correctness-critical glue code that lives at the perception/estimation boundary.

## The algorithm in brief

- **Per-point firing-time model** — a point's timestamp is derived from its azimuth within the sweep
  (the same 16-beam spinning-LiDAR geometry [`02.01`](../../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/src/kernels.cuh)
  cites from `01.18`, extended with per-firing time) → [`THEORY.md`](THEORY.md) "The problem".
- **Pose interpolation** — position **LERP**, orientation **SLERP** (proper geodesic interpolation,
  with the double-cover sign-flip handled and a small-angle/near-parallel fallback), between the two
  trajectory samples bracketing a point's firing time → THEORY.md "The math".
- **Two sampling regimes taught side by side** — a DENSE (~200 Hz-equivalent, 21-sample) regime and a
  SPARSE (2-sample, start/end only — implicit constant-velocity assumption) regime, using the
  IDENTICAL interpolation code on a shorter array → THEORY.md "The algorithm".
- **Per-point rigid re-projection** — every point re-expressed in the sweep-end reference frame via the
  relative transform between its own interpolated pose and the reference pose → THEORY.md "The math".

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/per-point-motion-deskew-with-pose-interpolation.sln`](build/per-point-motion-deskew-with-pose-interpolation.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/per-point-motion-deskew-with-pose-interpolation.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuBLAS/cuFFT/Thrust/cuRAND: every quaternion/vector primitive is hand-written (see `src/kernels.cuh`)
and there is no runtime randomness anywhere in this pipeline.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including the
**top-view triptych artifact**.

## Data

The committed sample is **synthetic** (the repo default, CLAUDE.md §8):
`data/sample/deskew_scan.bin` (~579 KiB, 18,426 points across 4 motion cohorts) — a 16-beam spinning
LiDAR sweeping a simple room while the platform follows four different ground-truth trajectories, with
every point carrying its exact analytic "instantaneous truth" twin (details:
[`data/README.md`](data/README.md)). No public dataset applies — motion deskew needs the platform's
*exact continuous trajectory* during a sweep, which no recorded dataset hands you cleanly;
`scripts/download_data.ps1` is an honest no-op.

## Expected output

Six independent gates, each printed as a stable `PASS`/`FAIL` line, checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt):

1. **`VERIFY`** — the §5 GPU-vs-CPU twin gate (tol 1e-4 m; measured worst deviation: **2.4e-6 m**).
2. **`SLERP_CORRECTNESS`** — a >90° quaternion pair's geodesic angle + double-cover sign-flip
   invariance (measured angle error: **5.6e-7 rad**; sign-flip error: **1.2e-7**).
3. **`IDENTITY_CONTROL`** — the stationary cohort's deskew must be a no-op (measured max displacement:
   **0.0 m**, exact to float precision).
4. **`RESTORATION`** — dense-sampled deskew vs. analytic instantaneous truth, all 3 moving cohorts,
   with the undeskewed baseline reported alongside (measured mean error, undeskewed → deskewed):
   straight 0.746 m → **0.000000 m**, arc 2.318 m → **0.000187 m**, wiggle 1.938 m → **0.096 m**.
5. **`SAMPLING_LESSON`** — wiggle cohort sparse/dense error ratio: measured **19.2×** (floor 5×);
   straight cohort dense-vs-sparse agreement: both **0.000000 m** (constant velocity is exact there).
6. **`DOWNSTREAM_PAYOFF`** — a wall plane-fit RMS thickness, straight cohort: skewed **0.239 m** →
   deskewed **0.0056 m** (matching the analytic truth's 0.0056 m exactly) — a **42×** tightening.

All numbers above are from an actual run on this project's reference machine; see
[`THEORY.md`](THEORY.md) "How we verify correctness" for the full measurement discussion.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: load → VERIFY → SLERP unit test →
   deskew both regimes on the GPU → four independent gates → artifacts.
2. [`src/kernels.cuh`](src/kernels.cuh) — the project's one real chapter: the trajectory/point layout
   contracts, and every quaternion/interpolation/deskew primitive, each derived in its own comment
   block. Read `deskew_one_point`'s comment for the whole algorithm in one place.
3. [`src/kernels.cu`](src/kernels.cu) — the one `__global__` kernel (a thin thread-per-point wrapper
   around `kernels.cuh`'s shared math) plus the `__constant__`-memory trajectory upload.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the CPU twin; read its file header for exactly
   what the GPU-vs-CPU comparison does and does NOT prove (and what closes that gap).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h` (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Zhang & Singh, "LOAM: Lidar Odometry and Mapping in Real-time" (2014)** — the lineage this
  project's decoupled deskew-then-register shape descends from; LOAM's own motion compensation uses
  the same LERP/SLERP-between-poses idea, driven by the odometry estimate from the PREVIOUS sweep.
- **Shan & Englot, "LIO-SAM" (2020)** and **Xu et al., "FAST-LIO2" (2022)** — the production frontier:
  motion compensation folded INSIDE a tightly-coupled LiDAR-inertial estimator (undistort using the
  IMU's own high-rate propagation, iterated with the scan-matching residual) instead of this project's
  decoupled "deskew first, register second" split — see THEORY.md "Where this sits in the real world".
- **Sola, "Quaternion kinematics for the error-state Kalman filter" (2017)** — the quaternion-algebra
  reference this project's `quat_rotate`/`quat_slerp` derivations cite directly.
- **PCL's `pcl::MovingLeastSquares` / motion-undistortion utilities** and **Open3D's** point-cloud
  transform pipeline — the production libraries whose per-point transform loop this project's kernel
  reimplements didactically on the GPU.
- **`sensor_msgs/PointCloud2` with a per-point `time` field (ROS/ROS 2 driver convention)** — the real
  packet-level source of the timestamps this project assumes; see `PRACTICE.md` §1 for the honest story
  of which drivers actually populate it.

## Exercises

1. **Plot the artifacts:** `demo/out/errors_wiggle.csv` → error vs. `t_s`. Identify WHERE in the sweep
   the sparse regime's error spikes, and check it lines up with the wiggle's oscillation peaks.
2. **Break the sampling rate:** halve `kDenseSamples` in `kernels.cuh` (regenerate the sample) and
   measure how the wiggle cohort's dense-regime error grows — find the sample rate where it starts
   approaching the sparse regime's error.
3. **Change the reference instant:** re-derive `deskew_one_point` for `t_ref` = sweep START instead of
   END, and confirm (by re-running VERIFY) that restoration quality is unchanged — only which frame the
   output lands in changes (THEORY.md "The math" explains why).
4. **Grow the trajectory, watch the binary search earn its keep:** feed `find_bracket_index` a
   1,000-sample trajectory (a plausible size for a longer time window) and profile linear-scan vs.
   binary-search — at `kDenseSamples=21` the two are indistinguishable; find where that stops being true.
5. **Climb toward tight coupling:** sketch (no need to implement) how you would fold this project's
   `deskew_one_point` INSIDE an iterated scan-matching loop, using the CURRENT iteration's estimated
   trajectory instead of a precomputed one — the FAST-LIO2 direction cited above.

## Limitations & honesty

- **Range-only, translation-and-rotation, no scan-matching feedback loop.** This project deskews using
  a GIVEN trajectory; it does not estimate that trajectory itself (that is domain 04's and 05's job).
  Production LiDAR-inertial systems increasingly fold deskew INSIDE the estimator (see "Where this sits
  in the real world" above) — a genuinely more accurate, more complex design this project deliberately
  does not build, to keep the interpolation math the whole lesson.
- **Synthetic timestamps, honestly scoped.** Real LiDAR drivers vary widely in whether they expose
  per-point firing times at all (`PRACTICE.md` §1) — this project assumes they do, which is the
  increasingly-common case (Velodyne/Ouster/Hesai all support it), but is not universal.
  - **Simplified scene, deliberately.** No boxes, no dynamic objects — the trajectory is this project's
  entire teaching payload; [`02.01`](../../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md)
  is where scene complexity is the lesson.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled; this
  project's point counts are small enough that GPU-vs-CPU speed-up is not the point (both easily clear
  the rate budget) — the POINT is correctness under interpolation, measured against known ground truth.
- **Not safety-certified; sim-validated only (CLAUDE.md §1).** This project's output feeds registration
  and mapping, not actuation directly — but a bad deskew degrades every downstream localization
  estimate a real robot's safety envelope depends on. Nothing here is validated against real sensor
  timing jitter, PTP synchronization error, or IMU/LiDAR extrinsic miscalibration; see `PRACTICE.md` §1
  and §3 for what a real integration must additionally account for.
