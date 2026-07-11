# 01.17 — Camera-LiDAR / camera-camera extrinsic calibration (batched reprojection-error optimization)

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Camera-LiDAR / camera-camera extrinsic calibration (batched reprojection-error optimization)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Extrinsic calibration answers one question: **where, exactly, is sensor B relative to sensor A?**
This project recovers that answer — a 6-degree-of-freedom rigid transform — for two sensor pairs
using the *same* GPU optimizer: (1) a camera and a LiDAR mounted on the same rig, and (2) two cameras
in a stereo-like pair. Both problems reduce to the identical mathematical shape: given a set of 3-D
points measured in one sensor's frame and their observed 2-D pixels in the other sensor's camera,
find the rigid transform that makes every projected point land on its observed pixel. The project
implements a **batched Levenberg-Marquardt (LM)** solver for that reprojection-error minimization, with
analytic Jacobians, entirely on the GPU, and demonstrates it two ways: a single calibration run
(GPU-accelerated per-iteration assembly) and a **1024-way parallel "farm"** of independent LM runs from
randomized starting guesses — the tool this project uses to answer *"how bad can the initial guess be
and still converge?"*, and to demonstrate the single most important practical lesson in calibration:
**pose diversity, not view count, is what makes a calibration well-conditioned.**

A learner who works through this project will understand: how to parameterize a 3-D rotation for
gradient-based optimization (the so(3) exponential map), how to derive an analytic Jacobian for a
camera-projection residual by hand, how Levenberg-Marquardt's damping trades off gradient descent
against Gauss-Newton, and two genuinely different ways to map an optimization problem onto a GPU
(parallel-over-*data* vs. parallel-over-*independent-trials*). The demo produces four artifacts: LM
loss-vs-iteration curves, a convergence-basin scatter plot, a metrics CSV, and — the "money shot" — a
camera image with LiDAR points reprojected onto it, in different colors, before and after calibration.

Both scenarios named in the catalog bullet are fully implemented (not documented-only): camera-LiDAR
and camera-camera calibration share every kernel, every CPU oracle, and every LM hyperparameter — see
[The algorithm in brief](#the-algorithm-in-brief) for exactly what is shared and why.

## What this computes & why the GPU helps

The computation is **nonlinear least-squares reprojection-error minimization**:

```
minimize over (R, t):   sum_i || project(R * p_i + t; K) - uv_i ||^2
```

where `p_i` are 3-D points in the source sensor's frame, `uv_i` are their observed pixels in the
destination camera, and `K` is the destination camera's known intrinsics (from 01.16). The GPU pattern
is **batched-solve applied two different ways**, both taught in this one project:

- **Correspondence-parallel reduction** — for ONE calibration estimate, each of the (here, 48)
  correspondences contributes one 2x6 Jacobian block; a shared-memory tree reduction folds all of them
  into a single 6x6 normal-equation system (`J^T J`, `J^T r`) per LM iteration. This is a *reduce* over
  independent per-point contributions, the same shape as a dot product or a histogram.
- **Optimization-parallel farm** — for the SAME correspondence set, 1024 threads each run an entire,
  independent, ~20-iteration LM trajectory from their own randomized starting guess, with no
  cross-thread communication at all. This is a *map* where the per-thread "element" is not a data point
  but an entire iterative algorithm — the pattern 08.01 (MPPI) uses for rollouts, applied here to
  optimization restarts instead.

The two are contrasted directly: the first parallelizes the correspondences of ONE optimization; the
second parallelizes MANY independent optimizations of a SMALL, fixed correspondence set. THEORY.md "The
GPU mapping" argues in detail which regime dominates as problem size changes.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** cross-cutting infrastructure, upstream of every multi-sensor perception and
  fusion stage — exactly like 01.16's intrinsic calibration (cited there as "not a 30 Hz pipeline
  stage... a one-time (or periodic) offline computation whose OUTPUT every other perception project
  silently assumes is already correct"). Extrinsic calibration is that same statement one level up:
  it is the glue that lets outputs from DIFFERENT sensors be expressed in one frame at all.
- **Upstream inputs:** per-view corner/fiducial detections and known camera intrinsics from **01.16**
  (checkerboard/ChArUco detection); raw 3-D fiducial measurements from a LiDAR (this project's
  camera-LiDAR scenario) — both consumed here as `PointCloud`-shaped 3-D points (§3.6) and pixel
  coordinates a real system would receive as `Image`-shaped detections.
- **Downstream consumers:** every fusion project in **04.xx** (state estimation cannot fuse a camera
  detection and a LiDAR return into one Kalman update without knowing their relative pose); LiDAR-camera
  painting/coloring kernels in **02.xx** (e.g. 02.17, which projects LiDAR points through exactly the
  `T_camera_lidar` this project solves for); any multi-camera rig project named in **01.07**. All of
  them consume the output as a `T_camera_lidar`/`T_camera2_camera1` rigid-transform constant, not a
  per-frame message — calibration runs once (or periodically), not in the sensing loop.
- **Rate / latency budget:** NOT a real-time stage — no row of the §1.1 rate table applies. A factory or
  field calibration run is offline, taking seconds to minutes; this project's own demo runs its full
  nine-stage verification (including a 1024-way multi-start farm) in low milliseconds of GPU time,
  which matters only insofar as it makes interactive re-calibration on a service bench practical.
- **Reference robot(s):** the **autonomous-vehicle stack** (domains 01/02/03/04/05/06/14/31/32) is the
  primary reference robot here — AV sensor rigs typically combine 4-12 cameras with several LiDARs, all
  needing pairwise or rig-wide extrinsics — and the **warehouse AMR** (02/04/05/23/06/08/25/31), whose
  camera-LiDAR fusion for obstacle detection depends on exactly this transform.
- **In production:** a dedicated calibration tool run at end-of-line (factory) and after any service
  event that could shift a sensor mount — see `PRACTICE.md` §3 for the recalibration triggers and §4 for
  the manufacturing-QA framing. Real tools: Kalibr, ROS 2's `camera_calibration`/`extrinsic_calibration`
  packages, and vendor-specific AV factory calibration cells (README "Prior art" below).
- **Owning team:** perception / calibration engineering (SYSTEM_DESIGN.md §5.1's "Perception" row), with
  heavy overlap into mechanical engineering (rig/fixture design, `PRACTICE.md` §1) and manufacturing/QA
  (per-unit calibration as a production test step, `PRACTICE.md` §4).

## The algorithm in brief

- **so(3) exponential-map parameterization** — the unknown rotation is represented as a 3-vector
  axis-angle "twist," turned into a rotation matrix via Rodrigues' formula (`so3_exp` in
  [`src/kernels.cuh`](src/kernels.cuh)); translation is a plain additive 3-vector. See THEORY.md "The
  math" for the full retraction the LM solver uses each iteration.
- **Analytic reprojection Jacobian ("the classic 2x6")** — derived by hand from the chain rule through
  the pinhole projection and the rotation perturbation, in `residual_and_jacobian()`; THEORY.md "The
  math" derives every line, and `main.cu`'s `jacobian_check` stage verifies it numerically at runtime
  (central-difference gate).
- **Levenberg-Marquardt with Marquardt (diagonal) damping** — damped Gauss-Newton, 6x6 Cholesky solve
  per iteration (`cholesky6_solve`), adaptive lambda, max 20 iterations. THEORY.md "The algorithm"
  derives the damping rule and cites 33.01/01.12's small-dense-solve pattern for the Cholesky itself.
- **Correspondence-parallel GPU assembly** — `assemble_normal_equations_kernel` in
  [`src/kernels.cu`](src/kernels.cu), a shared-memory tree reduction reusing 02.06 ICP's 6x6
  upper-triangle packing convention (extended by one scalar for the loss).
- **Optimization-parallel GPU farm** — `multistart_lm_farm_kernel`, one thread per independent LM
  trajectory, K=1024, the 08.01/01.12 "thread-per-problem" idiom applied to restart diversity instead of
  rollouts or robot configurations.
- **The degeneracy lesson** — the SAME camera-LiDAR extrinsic, solved from two synthetic pose cohorts
  (pose-diverse vs. near-coplanar), demonstrates measurably worse `J^T J` conditioning and measurably
  worse recovered accuracy on the coplanar cohort — echoing 01.16's own Zhang-calibration finding that
  view COUNT is not the same as pose DIVERSITY.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/camera-lidar-camera-camera-extrinsic-calibration.sln`](build/camera-lidar-camera-camera-extrinsic-calibration.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/camera-lidar-camera-camera-extrinsic-calibration.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only,
same as the repo default. No cuBLAS/cuSOLVER/Thrust: the linear algebra here is a fixed 6x6 solve and a
6x6 eigen-decomposition, small and instructive enough to hand-roll (CLAUDE.md §5's "hand-roll unless it
teaches nothing" rule) — see `src/kernels.cuh`'s `cholesky6_solve` and `jacobi_eigen_symmetric6`.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Three synthetic correspondence sets (48 correspondences each: 12 target poses x 4 fiducials) plus one
tiny synthetic background image, all generated by `scripts/make_synthetic.py` with a fixed seed:

- `cam_lidar_diverse.csv` — camera-LiDAR pair, target poses spanning wide depth (1.5-4.0 m) and
  orientation (up to ~50 deg) — the well-conditioned cohort used for the recovery, noise-scaling, and
  zero-noise gates.
- `cam_lidar_coplanar.csv` — the SAME ground-truth extrinsic, but 12 poses held near one depth and
  orientation — the deliberately degenerate cohort the DEGENERACY gate compares against the above.
- `cam_cam_diverse.csv` — camera-camera pair, 12 diverse poses relative to camera 1.
- `cam_background.pgm` — a 160x120 synthetic grayscale scene, the backdrop for the overlay artifact.

Every file stores the EXACT (noise-free) correspondences plus the ground-truth extrinsic; `main.cu`
adds sensor noise itself, deterministically, at three documented levels — see
[`data/README.md`](data/README.md) for the full field/units documentation and
[`scripts/make_synthetic.py`](scripts/make_synthetic.py) to regenerate. All synthetic, MIT-licensed
like the rest of the repo (CLAUDE.md §8).

## Expected output

The demo runs **nine verification stages**, each printing a stable `PASS`/`FAIL` line (exact measured
numbers live on the adjacent, unchecked `[info]` lines — see `src/main.cu`'s "Output contract" comment
for why): a jacobian-vs-numeric-diff check, a GPU-vs-CPU single-assembly twin, a GPU-vs-CPU full-trajectory
twin, a GPU-vs-CPU multi-start-subset twin, the convergence-basin coverage gate, camera-LiDAR and
camera-camera recovery-accuracy gates, a noise-scaling sanity check, the degeneracy (conditioning) gate,
and a zero-noise exactness anchor for both scenarios. All tolerances are **measured, then margined**
(run once, the actual worst-case deviation recorded, the threshold set with documented headroom above
it — 08.01's technique) — see `demo/README.md` for the full table of measured numbers and margins.

Four artifacts land in `demo/out/`: `convergence_curves.csv` (LM loss vs. iteration, GPU- and CPU-driven
trajectories side by side), `basin_scatter.csv` (1024 rows: initial perturbation magnitude vs. whether
that start converged), `gates_metrics.csv` (every gate's key numbers in one file), and `overlay.ppm` —
the true detected pixels (green), the pre-calibration reprojection (red), and the post-calibration
reprojection (blue) of the camera-LiDAR correspondences, drawn onto the synthetic background image.

The canonical stable lines live in [`demo/expected_output.txt`](demo/expected_output.txt).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **start here.** The full contract: correspondence/pose
   layouts, the LM hyperparameters, the shared camera-model math (so3_exp, the analytic Jacobian), and
   the two kernel interfaces. Read the file header in full before anything else.
2. [`src/main.cu`](src/main.cu) — the nine-stage orchestrator: data loading, noise synthesis, the
   host-orchestrated single-trajectory LM (`run_lm_gpu`), and every gate.
3. [`src/kernels.cu`](src/kernels.cu) — the two GPU kernels: `assemble_normal_equations_kernel`
   (correspondence-parallel reduction) and `multistart_lm_farm_kernel` (the optimization-parallel farm
   — the most interesting kernel in the project; read its header comment on why it uses double precision).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle twins, with a file
   header explaining exactly what is and is not shared with the GPU path, and why.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h` (data-file/artifact-dir
   resolution — this project uses both `find_data_file` and `resolve_out_dir`).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Kalibr** (ETH Zurich) — the de facto open-source multi-camera/IMU/LiDAR extrinsic calibration
  toolbox; its `camera-camera` mode solves almost exactly this project's camera-camera problem, with a
  full spline-based continuous-time backend this project deliberately does not build.
- **OpenCV's `stereoCalibrate`/`calibrateHandEye`** — the workhorse for camera-camera and hand-eye
  extrinsics; study its Levenberg-Marquardt implementation (`cv::LevMarq`) for a production-grade
  version of this project's `run_lm_cpu`/`run_lm_gpu`.
  https://docs.opencv.org/
- **ROS 2 / Autoware's `extrinsic_calibration_manager` and `tier4/CalibrationTools`** — real AV-stack
  camera-LiDAR calibration pipelines, including the interactive/manual-refinement UI a production tool
  needs beyond this project's fully-automated batch solve.
- **Zhang (2000), "A Flexible New Technique for Camera Calibration"** — the intrinsic-calibration paper
  01.16 implements; its pose-diversity observability argument is the exact one this project's
  DEGENERACY gate demonstrates for extrinsics.
- **Madsen, Nielsen & Tingleff, "Methods for Non-Linear Least Squares Problems"** — the standard,
  freely-available reference for the Levenberg-Marquardt damping rule this project implements
  (Marquardt's diagonal scaling variant).
- **Barfoot, "State Estimation for Robotics"** — the standard robotics-flavored treatment of SE(3)/SO(3)
  retractions and Jacobians; read its chapter on Lie groups for the FULLY-coupled se(3) exponential this
  project's "decoupled" SO(3) x R^3 retraction simplifies (THEORY.md "Numerical considerations" names
  the difference explicitly).

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Add lens distortion.** `pinhole_project` assumes an ideal pinhole; extend it with radial/tangential
   distortion terms (01.16's Zhang calibration ignores them too — extend both together) and re-derive
   the Jacobian's extra columns (or verify them purely numerically via `jacobian_check`'s technique).
2. **Widen the degeneracy study.** Generate a THIRD cohort at an intermediate diversity level and show
   the condition number varying continuously, not just diverse-vs-coplanar — turn `DEGENERACY` into a
   3-point curve like `NOISE_SCALING` already is.
3. **Fuse the block-reduction and farm kernels.** Right now the farm kernel scans all 48 correspondences
   SERIALLY per thread. For a much larger correspondence count, redesign it so a WARP (not a thread)
   owns one optimization, using warp shuffles to reduce the per-iteration sum across the warp's 32
   lanes — a hybrid of this project's two GPU regimes.
4. **On-device cuRAND.** The multi-start farm currently reads its randomized initial guesses from the
   portable xorshift32 generator (bit-reproducible across platforms, per 08.01's convention). Swap in
   cuRAND's device API and measure whether it changes the basin-convergence percentage.
5. **Joint camera-LiDAR-camera calibration.** Currently the two scenarios are solved independently.
   Extend `run_lm_gpu` to solve BOTH `T_camera_lidar` and `T_camera2_camera1` in one combined normal
   system (a 12-parameter state) sharing a common reference frame — the "rig-wide" calibration a real AV
   factory line actually runs.

## Limitations & honesty

- **No lens distortion.** Both scenarios use an ideal pinhole model (same simplification 01.16 makes for
  intrinsics) — Exercise 1 above extends it.
- **Camera-camera has no source-point noise, by design.** The 3-D points fed into the camera-camera
  solve are treated as exact, standing in for a per-view PnP pose from a REFERENCE camera (assumed
  already well-calibrated via 01.16's pipeline) composed with the board's exact manufactured geometry.
  A fully joint solve (Exercise 5) would instead propagate that reference camera's own uncertainty.
- **LiDAR noise is isotropic Cartesian Gaussian, not the true range/bearing (spherical) noise model** a
  real LiDAR exhibits — THEORY.md "Numerical considerations" names the real model this approximates and
  why the simplification is unlikely to change the qualitative lesson (pose diversity vs. conditioning).
- **The "decoupled" SO(3) x R^3 retraction**, not the fully-coupled se(3) exponential — THEORY.md "The
  math" and "Numerical considerations" name the difference and cite Barfoot's book for the full version.
- **Small, fixed correspondence counts (48).** Real rig calibration often uses hundreds of detections;
  the GPU kernels are written for arbitrary `n` (Exercise 3 asks what changes at n=50,000), but the
  committed sample stays tiny per CLAUDE.md §8, and the timing numbers reported are correspondingly tiny
  (a few milliseconds) — a teaching artifact, not a benchmark claim.
- **The multi-start farm's perturbation range (kernels.cuh's `kBasinMaxRotRad`/`kBasinMaxTransM`) was
  deliberately widened beyond "comfortable"** specifically so the basin gate finds a genuine convergence
  boundary (measured: ~77% of 1024 randomized starts up to ~137 degrees / 1.2 m from identity converge)
  rather than reporting an uninteresting 100% — see THEORY.md "How we verify correctness" for the exact
  numbers and why a narrower range would have taught nothing about basin SIZE.
- **Everything here is sim-validated only, on synthetic data, and not safety-certified** (CLAUDE.md §1).
  A recovered extrinsic feeding a real robot's perception stack must be validated against an independent
  physical measurement (see `PRACTICE.md` §3's field-validation procedure) before being trusted for
  anything safety-relevant.
