# 01.03 — Optical flow: pyramidal Lucas-Kanade, Farneback, census-transform flow

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Optical flow: pyramidal Lucas-Kanade, Farneback, census-transform flow`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This is a **bundled catalog bullet** (CLAUDE.md §2): it names three optical-flow methods, and this
project implements two of them fully and teaches the third:

- **IMPLEMENTED — dense pyramidal Lucas-Kanade**: a 3-level image pyramid, per-pixel Scharr gradients,
  a 5x5 structure-tensor + mismatch-vector solve, 3 warped-resampling iterations per level, and
  coarse-to-fine flow propagation, with a per-pixel CONFIDENCE output (the structure tensor's small
  eigenvalue — the aperture problem, made numeric).
- **IMPLEMENTED — census-transform block-matching flow**: a 5x5 (24-bit) census signature per pixel,
  brute-force Hamming block matching over a 13x13 search window with parabolic sub-pixel refinement,
  and a forward/backward (left-right) consistency check producing a validity mask.
- **DOCUMENTED ONLY — Farneback polynomial-expansion flow**: taught in full in
  [`THEORY.md`](THEORY.md#farneback-polynomial-expansion-milestone-3--documented-only-not-implemented)
  — the math (local quadratic signal model, closed-form displacement from polynomial coefficients) is
  derived to the point a learner could implement it, but no kernel exists for it here (see
  [Limitations & honesty](#limitations--honesty) for the scoping rationale).

A learner studying this project sees the SAME correspondence problem attacked two structurally
different ways — one that linearizes raw intensity (LK), one that compares local intensity RANK ORDER
(census) — on four synthetic frame pairs with EXACT, closed-form ground-truth flow: a pure translation,
a rotation+zoom (spatially-varying flow, the pyramid's reason to exist), the same translation with an
added brightness gradient (isolating brightness robustness), and a zero-motion negative control. The
demo runs both methods on all four pairs, verifies every GPU kernel against an independent CPU oracle,
and checks the result against eight independent gates built from the analytic ground truth — not just
"GPU agrees with CPU," but "both agree with reality."

> **Template placeholder notice.** This section documents the real implementation; the scaffold's SAXPY
> toolchain-validation placeholder has been fully replaced — every file under `src/` implements this
> project's real algorithm.

## What this computes & why the GPU helps

**Dense per-pixel correspondence between two frames** — for every one of `kW*kH = 19,200` pixels,
answer "where did this point go?" Both methods are, at their core, a **MAP** over pixels (embarrassingly
parallel — one GPU thread per pixel does independent work with no cross-thread communication) composed
with occasional **STENCIL** reads (a small fixed window around each thread's own pixel):

- **Lucas-Kanade** is a map+stencil PER PYRAMID LEVEL: gradients, structure tensor, and the iterative
  mismatch/solve are each a per-pixel kernel reading a local 5x5 window. The one place parallelism runs
  out is BETWEEN levels — level `L+1` needs level `L`'s finished flow field (see "The algorithm in
  brief") — so the level loop is host-orchestrated and sequential while every kernel WITHIN a level
  saturates the GPU.
- **Census matching** is a map with an in-thread SEARCH loop: each thread evaluates all 169 candidate
  displacements for its own pixel independently — more arithmetic per thread than a plain stencil, but
  still embarrassingly parallel across pixels (no pyramid, no cross-level dependency at all).

**Measured reality** (RTX 2080 SUPER, sm_75, Release, this project's committed sample; see "Expected
output" for the full line): **~3.7-5.2 ms** of total GPU kernel time covers BOTH methods on ALL FOUR
scene pairs (five VERIFY stages plus every gate's flow field), versus **~36 ms** for the CPU reference
computing the five verified stages alone — a real speed-up, and, more importantly for a robot, comfortably
inside a 30-60 Hz camera's per-frame budget (see "System context").

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** **perception**, specifically the boundary between raw imagery and state
  estimation — flow is a *measurement*, not yet an estimate: it feeds a filter/estimator, it does not
  replace one.
- **Upstream inputs:** rectified grayscale frames — the message-shaped output of project **01.01**'s
  full GPU image pipeline (debayer → undistort → rectify → resize), named explicitly because this
  project's `read_pgm()`-loaded frames are exactly the format 01.01 produces: undistorted, rectified,
  ready for pixel-correspondence math with no lens-distortion confound.
- **Downstream consumers:** dense per-pixel flow (with confidence/validity) feeds **01.21**'s scene-flow
  module (adding depth to lift 2-D flow into 3-D motion), a **04.07**-class sliding-window VIO backend
  (flow-derived velocity as a measurement into visual-inertial fusion), and dynamic-object segmentation
  (moving pixels stand out against the ego-motion-consistent background flow field a static scene would
  produce). A learned-flow deployment (RAFT-family — see `THEORY.md`) would ship via the SAME
  TensorRT-plus-custom-kernel pattern project **12.01** teaches, replacing this project's classical
  kernels with a trained network's forward pass behind the identical `(u, v, confidence)` interface.
- **Rate / latency budget:** SYSTEM_DESIGN.md's quadrotor reference architecture puts cameras at
  **30-60 Hz** feeding state estimation at **100-400 Hz** output (item 1's rate table) — flow must
  finish well under one camera period to avoid becoming the pipeline's bottleneck. **What this
  implementation measures**: ~1-2 ms of GPU kernel time for ONE method on ONE frame pair at this
  project's 160x120 teaching resolution — comfortable headroom at 30-60 Hz even before considering that
  a production deployment would run at a higher, camera-native resolution (see "Limitations & honesty").
  **What production stacks achieve**: NVIDIA's dedicated optical-flow hardware ACCELERATOR (VPI, Turing
  and later — `THEORY.md` "Where this sits in the real world") computes a census-like cost volume in
  fixed-function silicon at real-time HD frame rates with near-zero CUDA-core load; classical
  CPU-only LK-family trackers (still common in resource-constrained flight controllers) run at a few
  hundred sparse keypoints per frame at similar rates, not this project's DENSE every-pixel field.
- **Reference robot(s):** the **quadrotor** (velocity/hover-assist estimation when GPS is degraded or
  absent — the PX4Flow sensor module's lineage, a purpose-built camera+gyro+MCU board that runs a
  correlation/LK-style flow algorithm onboard and outputs `(vx, vy, quality)` directly to the flight
  controller; project **15.12** "precision landing: fiducial + optical-flow fusion" is this repo's
  closest sibling use case) and the **warehouse AMR** (visual odometry augmenting wheel odometry on
  slick or obstructed floors, particularly useful where wheel slip corrupts the primary odometry
  source).
- **In production:** a shipping stack would add an explicit OUTLIER-REJECTION/robust-fusion layer
  consuming this project's confidence/validity output (not just thresholding it, as this project's gates
  do, but feeding it as a per-measurement weight into an EKF/factor-graph estimator), likely a
  fixed-function hardware accelerator instead of CUDA cores for the census-style search (VPI), and — for
  metric (not just pixel-domain) velocity — a height/depth source to scale flow into physical units (see
  `PRACTICE.md` §3).
- **Owning team:** perception (SYSTEM_DESIGN.md item 5) — working closely with controls/autonomy (the
  consumer of flow-derived velocity) and, for a flight vehicle specifically, the flight-software team
  owning the PX4Flow-class sensor's onboard firmware (`PRACTICE.md` §4).

## The algorithm in brief

- **Dense pyramidal Lucas-Kanade** — build a 3-level image pyramid (2x area-average decimation per
  level, citing project 01.01's identical anti-aliasing argument); at each level, compute Scharr
  gradients and the 5x5 structure tensor ONCE, then iterate 3 times: bilinear-warp frame 1 by the
  running flow estimate, accumulate the mismatch vector, solve the 2x2 normal equations, clamp, update;
  upsample the result (bilinear + **x2 magnitude scale**) to seed the next finer level. → [THEORY.md
  "The math"](THEORY.md#structure-tensor-and-mismatch-vector-lucas-kanade) for the full derivation
  (including the sign convention, verified on paper against a linear-ramp test case) and [THEORY.md
  "The algorithm"](THEORY.md#milestone-1--dense-pyramidal-lucas-kanade) for the pseudocode/complexity.
- **Aperture problem as CONFIDENCE** — the structure tensor's small eigenvalue, computed once per level,
  reported at level 0 as a per-pixel trust signal (large = well-constrained 2-D texture, small = flat
  region or a straight edge with a locally invisible motion component). → [THEORY.md "The aperture
  problem"](THEORY.md#the-aperture-problem-from-first-principles).
- **Census-transform block matching** — a 24-bit rank-order signature per pixel (5x5 window, center
  excluded), brute-force Hamming winner-take-all over a 13x13 search window (reusing project 01.04's
  `__popc()` lesson), parabolic sub-pixel refinement, and a forward/backward consistency check for a
  validity mask. → [THEORY.md "Census transform and the Hamming
  metric"](THEORY.md#census-transform-and-the-hamming-metric) for the rank-order invariance proof.
- **Farneback (documented only)** — local quadratic (second-order) signal model fit per pixel via
  weighted least squares; displacement recovered in closed form from the fitted polynomial coefficients.
  → [THEORY.md "Farneback polynomial
  expansion"](THEORY.md#farneback-polynomial-expansion-milestone-3--documented-only-not-implemented).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/optical-flow.sln`](build/optical-flow.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/optical-flow.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. Every
kernel (pyramid, gradients, structure tensor, LK iteration, census transform/match/consistency) is
hand-rolled; no cuBLAS/cuFFT/Thrust/CUB is linked.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at — including how to read the **HSV flow color wheel** the artifacts use.

## Data

Four synthetic ground-truth-known frame pairs, all sharing one reference frame (`scene_a.pgm`, a
"hashed multi-scale texture" scene — deliberately NOT a checkerboard, see `data/README.md` for why dense
correspondence needs richer local structure than 01.04's sparse-feature checkerboard scene provides):
pure translation, rotation+zoom, translation+brightness-ramp, and zero-motion. Generated by
`scripts/make_synthetic.py` (seed 42, deterministic, no downloads); details, checksums, and the exact
ground-truth transform parameters in [`data/README.md`](data/README.md). No public dataset applies
(pixel-dense optical-flow benchmarks — Middlebury, MPI-Sintel, KITTI-flow — are research/non-commercial
or registration-gated; `scripts/download_data.ps1`/`.sh` are honest, permanent no-ops).

## Expected output

Every VERIFY and GATE line is stable PASS/FAIL text (no embedded numbers — cross-GPU-portable); measured
numbers live on `[info]` lines. The canonical lines are in
[`demo/expected_output.txt`](demo/expected_output.txt); the following is the FULL measured output on the
reference machine (RTX 2080 SUPER, sm_75, Release|x64) — a teaching artifact, not a benchmark claim:

- **5 GPU-vs-CPU VERIFY stages, all PASS**: `gradient` (bit-exact, max diff 0.000000 — Scharr taps are
  integers scaled by an exact power-of-two `1/32`), `lk_flow` (the full pyramidal pipeline's final flow
  field, tolerance-checked, max diff 0.0000 px on the committed sample), `census_transform` (bit-exact,
  0/19,200 signature mismatches), `census_match` (integer WTA cost bit-exact, sub-pixel diff 0.0000 px),
  `census_flow` (the full pipeline, max diff 0.0000 px, 0/19,200 validity-mask mismatches).
- **8 independent gates, all PASS**:

  | Gate | Measured | Tolerance |
  |------|----------|-----------|
  | `translation_lk` | mean EPE 0.1429 px (12,996 confident px) | <= 0.35 px |
  | `translation_census` | mean EPE 0.2716 px (10,107 valid px) | <= 0.45 px |
  | `rotation_zoom_lk` | mean EPE 1.8142 px (pyramidal, 12,996 confident px) | <= 2.50 px |
  | `pyramid_advantage` | single-level 7.4702 px vs. pyramidal 1.8142 px = **4.12x** | >= 1.50x |
  | `brightness_robustness_census` | mean EPE 0.6411 px (2.36x its own translation-scene EPE) | <= 1.00 px AND <= 3.00x |
  | `zero_motion_lk` | mean \|flow\| 0.0000 px | <= 0.20 px |
  | `zero_motion_census` | mean \|flow\| 0.2776 px | <= 0.35 px |
  | `confidence_mask_sanity` | rejected 2.2875 px > accepted 1.8142 px | rejected > accepted |

- **Reported, not gated** (an honest contrast, not a failure): LK's mean EPE on the SAME
  brightness-ramped scene is **17.6463 px** — over 27x worse than census's 0.6411 px on the identical
  scene, the brightness-constancy assumption breaking exactly as `THEORY.md` predicts.
- **Timing**: ~3.7-5.2 ms total GPU kernel time (both methods, all four scene pairs, five VERIFY stages
  plus every gated flow field); ~36-37 ms CPU reference time (the five VERIFIED stages only).

Tolerances were set with a documented margin over these measured numbers (the repo-wide "measured, then
margined" convention — see `THEORY.md` "How we verify correctness" for the empirical process, including
a real gradient-normalization bug this process caught, that arrived at both the numbers and the margins).

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: load the four scene pairs, five VERIFY stages, compute
   every flow field the gates need, eight gates, artifacts. Start here to see the whole pipeline shape.
2. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced contract: image/pyramid geometry, the
   census offset table, every tolerance/threshold constant with its justification, every kernel and
   launch-wrapper signature.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels. Read `scharr_gradient_kernel`'s header FIRST
   for the gradient-normalization lesson (the single most important numerics point in this project),
   then `lk_iterate_kernel` (the LK solve), then `census_match_kernel` (the search + sub-pixel fit).
   `run_pyramidal_lk_gpu`/`run_census_flow_gpu` at the bottom are the host orchestration loops.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle; compare
   `lk_iterate_cpu` against `lk_iterate_kernel` side by side to see "the same algorithm, twice."
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `find_data_file`/`resolve_out_dir`
   (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Lucas & Kanade (1981)** and **Bouguet's pyramidal implementation notes** (Intel/OpenCV technical
  report) — the original algorithm and the practical coarse-to-fine recipe this project's pyramid loop
  follows.
- **Baker & Matthews (2004), "Lucas-Kanade 20 Years On"** — the unifying framework (forward/inverse,
  additive/compositional) this project's THEORY.md cites for exactly which LK variant `lk_iterate_kernel`
  implements.
- **Zabih & Woodfill (1994)** — the original census transform paper; this project's rank-order
  invariance argument is theirs, made explicit for a robotics audience.
- **Farnebäck (2003), "Two-Frame Motion Estimation Based on Polynomial Expansion"** — Milestone 3's
  source, taught in full in `THEORY.md` though not implemented here.
- **OpenCV** (`cv::calcOpticalFlowPyrLK`, `cv::cuda::DensePyrLKOpticalFlow`, `cv::cuda::
  FarnebackOpticalFlow`, `cv::StereoSGBM`'s census-family cost) — the production reference
  implementations of every method this project teaches.
- **NVIDIA VPI / the NVIDIA Optical Flow hardware accelerator** — the fixed-function-silicon production
  descendant of this project's census milestone (Turing and later GPUs).
- **Teed & Deng (2020), "RAFT"** — the learned-flow state of the art this project's classical methods
  predate and are still compared against; `THEORY.md` "Where this sits in the real world" places it
  honestly relative to this project's approach.
- **PX4Flow** (open-hardware optical-flow sensor module) — the closest real-world hardware analogue to
  what this project's `main.cu` computes, at flight-vehicle power/compute budgets; see `PRACTICE.md` §2.

## Exercises

1. **Plot the artifacts.** Open `demo/out/flow_lk_rotzoom.ppm` and `flow_census_translation.ppm`
   alongside `flow_color_wheel.ppm`; verify by eye that the translation scene's HSV wheel is a
   near-uniform color patch (constant flow) while the rotation+zoom scene shows a smoothly rotating hue
   pattern radiating from the image center.
2. **Try a 7x7 LK window.** Change `kLkWindowRadius` (kernels.cuh) from 2 to 3, rebuild, and measure the
   effect on `rotation_zoom_lk`'s EPE and on the confident-pixel COUNT — THEORY.md's "The algorithm"
   predicts the trade-off (better conditioning vs. larger border loss); confirm it.
3. **Break the gradient normalization on purpose.** Comment out the `* (1.0f / 32.0f)` scale in BOTH
   `scharr_gradient_kernel` and `scharr_gradient_cpu`, rebuild, and watch `translation_lk` fail —
   reproduce, in five minutes, the real bug THEORY.md's "How we verify correctness" describes finding.
4. **Implement census cost aggregation.** Real stereo/flow pipelines often SUM Hamming cost over a
   further support window (not just the raw single-pixel census cost) before the winner-take-all step —
   implement it and measure whether `translation_census`'s EPE improves.
5. **Shared-memory tile `census_match_kernel`.** THEORY.md's "The GPU mapping" names the tiling
   opportunity (neighboring threads' 169-candidate searches overlap heavily); implement a
   `(blockDim + 2*kCensusSearchRadius)^2` shared-memory tile of `census_tgt` per block and measure the
   kernel-time change with `nvprof`/Nsight Compute.

## Limitations & honesty

- **Farneback is documented, not implemented** — the catalog bullet names three methods; per CLAUDE.md
  §2's bundled-bullet rule, this project implements the two most instructive-to-CONTRAST methods (a
  linearized-intensity solver vs. a rank-order block matcher) fully and teaches the third's math and
  algorithm completely in `THEORY.md`, honestly scoped rather than partially/badly implemented.
- **Census's rotation+zoom scene is reported, not gated.** The rotation+zoom pair's flow is generally
  non-integer AND large (up to ~12 px at the frame corners) — well outside census's fixed 13x13 search
  radius in places, and confounded by the sub-pixel floor `THEORY.md` derives. Only the translation and
  brightness-ramp pairs (both small, and translation deliberately exact-integer) are gated for census;
  `main.cu` computes and could report the rotation+zoom census result, but this project does not claim
  it as a tested guarantee.
- **160x120 teaching resolution, not camera-native.** Real cameras feeding a flow front end are typically
  VGA (640x480) or larger; this project's small resolution keeps the CPU oracle fast (a few tens of
  milliseconds) and keeps the whole pipeline easy to read end to end in one sitting. The algorithms
  themselves are resolution-independent (every kernel takes `W,H` as runtime parameters, not compile-time
  constants); scaling up is a data/constant change, not a rewrite (see Exercises).
- **No occlusion model beyond census's incidental left-right check.** Neither method here explicitly
  reasons about disocclusion; a pixel with genuinely no correspondence in the other frame gets SOME
  answer from both methods (LK always outputs a locally-optimal flow even where none is meaningful;
  census's LR-consistency check catches many but not all such cases as a side effect, not by design).
- **A real gradient-scale bug was caught and fixed while building this project** (see `THEORY.md` "How
  we verify correctness" for the full story) — recorded here deliberately, per CLAUDE.md's
  never-fabricate rule: an honest account of what went wrong and how the repo's twin-plus-independent-
  gate verification discipline caught it is as instructive as the final numbers.
- **Sim-validated only, and this one matters here (CLAUDE.md §1):** this project's output — a per-pixel
  velocity/displacement field — is the archetype of a measurement that could feed a robot's CONTROL loop
  (a quadrotor's hover-assist controller, an AMR's odometry). Everything here ran only against synthetic,
  ground-truth-known imagery on a desktop GPU; nothing is safety-certified, no real-camera or
  real-hardware claim is made, and any real deployment would need the full testing ladder
  (`PRACTICE.md` §3: simulation → recorded playback → tethered bench → free running with hard limits)
  plus an independent safety envelope around whatever consumes this project's output.
