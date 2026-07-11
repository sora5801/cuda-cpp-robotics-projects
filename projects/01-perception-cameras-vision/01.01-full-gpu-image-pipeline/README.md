# 01.01 — Full GPU image pipeline: debayer → undistort → rectify → resize → normalize, zero CPU copies

**Difficulty:** ★ beginner · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `★ Full GPU image pipeline: debayer → undistort → rectify → resize → normalize, zero CPU copies`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Every robot camera driver runs the same five classical steps before a single pixel reaches
perception: turn the raw Bayer mosaic into color (**debayer**), remove the physical lens's
distortion (**undistort**), correct a small sensor-mounting misalignment (**rectify**), shrink the
image to the resolution the downstream network expects (**resize**), and convert it to a zero-mean,
unit-std float tensor (**normalize**). This project implements all five stages on the GPU, teaches
each one individually, then FUSES undistort+rectify+resize into a single kernel and measures exactly
how many fewer bytes that kernel moves through memory compared to running them as five separate
kernels — the kernel-fusion lesson that is this project's centerpiece. "Zero CPU copies" describes
the pipeline chain itself: from the moment the raw Bayer image lands on the GPU to the moment the
final normalized tensor is downloaded, every intermediate stage's image stays in device memory —
nothing round-trips through the host in between (the demo's `demo/out/` artifacts are an additional,
separate debug step that DOES download each stage for visualization; the pipeline itself does not
need to). All five stages are implemented and run on a fully synthetic, ground-truth-labeled Bayer
scene (checkerboard + smooth color gradient + three colored disks); nothing here is bundled or
scoped down from the catalog bullet.

## What this computes & why the GPU helps

Every stage below is a **MAP**: each output pixel is computed independently of every other output
pixel (the one partial exception is normalize's mean/std, a **tree reduction**, not a map — see
`THEORY.md`). That makes the whole pipeline "embarrassingly parallel" — the natural GPU mapping is
one thread per output pixel, and the interesting engineering question is not "can this run in
parallel" but **how many kernels to use**: this project builds the same three middle stages
(undistort+rectify+resize) both as three separate kernels (STAGED, easy to read, each stage a full
image round-trip through global memory) and as one fused kernel (FUSED, harder to read, no
intermediate image ever materializes) and measures the difference.

- **Debayer**: a *stencil* — each thread reads a 3x3 neighborhood of the raw mosaic.
- **Undistort+rectify**: a *gather* with a precomputed lookup table — each thread reads exactly one
  LUT entry and bilinear-samples one location.
- **Resize**: a *reduction over a fixed 2x2 window* — an area-average box filter.
- **Normalize**: a *two-pass tree reduction* (mean/std) followed by a *map* (the affine transform).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the SENSOR-FACING FRONT of the Perception layer — the boundary between
  `[SENSORS]` and `[PERCEPTION]` in SYSTEM_DESIGN.md's stack diagram. This is the code that turns
  raw silicon output into the first image any other algorithm in this repository can consume.
- **Upstream inputs:** the camera driver / image-sensor-processor hardware, delivering a raw Bayer
  frame — modeled here as `sensor_msgs/Image`-shaped (SYSTEM_DESIGN.md §3.6), `channels=1`,
  8-bit, `frame_id="camera_raw"`.
- **Downstream consumers:** literally every vision project in this repository — 01.02 (stereo depth)
  consumes a RECTIFIED pair; 01.04-style feature/detector projects and every domain-12 (ML/AI) model
  consume the final NORMALIZED float tensor. Produces `sensor_msgs/Image`-shaped output,
  `channels=3` (rectified stages) or a raw `float*` tensor (post-normalize), `frame_id="camera_rect"`.
- **Rate / latency budget:** camera -> perception is **30-60 Hz**, budget **< 1 frame (16-33 ms)
  end-to-end** (SYSTEM_DESIGN.md §1.1) — this project's entire staged pipeline measures well under a
  millisecond on an RTX 2080 SUPER at 384x288 (see "Expected output" below), leaving the whole
  latency budget for the perception network that consumes its output.
- **Reference robot(s):** the warehouse AMR (camera-based obstacle/lane perception) and the 6-DoF
  manipulator work cell (eye-in-hand or fixed camera calibration) both run this exact front-end
  before any downstream perception; the autonomous-vehicle stack runs one instance PER camera.
- **In production:** replaced/surrounded by libargus (Jetson ISP), V4L2 + a vendor ISP driver, or a
  GPU library (OpenCV `cv::cuda`, NVIDIA VPI) — README "Prior art" expands on each.
- **Owning team:** perception / camera-systems engineering (SYSTEM_DESIGN.md §5.1's "Perception" row,
  domains 01/02/03/20) — the team that owns calibration pipelines and the ISP-to-perception boundary.

## The algorithm in brief

- **RGGB bilinear demosaic** ([`THEORY.md#the-algorithm`](THEORY.md#the-algorithm)) — reconstruct 3
  channels/pixel from the 1-channel-per-pixel Bayer mosaic by averaging same-color neighbors.
- **Brown-Conrady lens distortion model** (`k1, k2` radial, `p1, p2` tangential) plus a small
  rectifying rotation, combined into one **inverse-mapped remap LUT** — precomputed once, reused by
  every consumer of the image ([`THEORY.md#the-gpu-mapping`](THEORY.md#the-gpu-mapping)).
  Inverse mapping (walk OUTPUT pixels backward to their INPUT location) is what makes every output
  pixel land somewhere in the input, with no holes — the classical alternative (forward-warping the
  input) cannot make that guarantee.
- **Exact area-average downscale** by `kResizeFactor=2` — the anti-aliasing-correct filter for an
  integer decimation ratio.
- **Deterministic (atomic-free) two-pass mean/std reduction**, then a per-channel affine normalize —
  the ML-consumer convention every camera-fed neural net expects.
- **Kernel fusion**: undistort+rectify+resize collapsed into one kernel, avoiding the intermediate
  full-resolution image's global-memory round trip entirely (`THEORY.md`'s memory-traffic derivation).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/full-gpu-image-pipeline.sln`](build/full-gpu-image-pipeline.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/full-gpu-image-pipeline.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — this project uses only the CUDA runtime and the
C++17 standard library (CLAUDE.md §5's default dependency budget); no cuBLAS/cuFFT/Thrust/etc. is
needed for any stage.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is fully **synthetic** (per CLAUDE.md §8): a 384x288 RGGB Bayer scene
(checkerboard + smooth color gradient + three colored disks) with dense, exact ground truth,
generated by [`scripts/make_synthetic.py`](scripts/make_synthetic.py) from a fixed seed (42). The
raw Bayer image is produced by inverse-warping the ground-truth ideal image through the SAME camera
model the pipeline uses (rotate, Brown-Conrady distort, mosaic) — the exact inverse of what the
pipeline is asked to undo. Full field documentation, checksums, and regeneration instructions in
[`data/README.md`](data/README.md).

## Expected output

The demo runs the pipeline STAGED and FUSED, verifies every GPU kernel against an independent CPU
oracle (`VERIFY:`), then checks seven physical gates (`GATE <name>:`) — see
[`demo/README.md`](demo/README.md) for what each one means. Measured on the reference machine (RTX
2080 SUPER, sm_75, Release|x64):

- `VERIFY: PASS` — every kernel matches its CPU twin within a documented per-stage tolerance (uint8
  stages: <= 1.5/255; the remap LUT: <= 0.002 px; normalize stats: <= 0.005; normalize apply: <=
  0.05 in normalized units — the last of these is *derived*, not guessed: a legitimate +-1 uint8
  rounding difference between GPU's FMA-contracted and CPU's non-contracted bilinear interpolation,
  divided by the image's per-channel std (~65), propagates to ~0.015 normalized units; see
  `THEORY.md` "Numerical considerations").
- All seven gates `PASS`, with these MEASURED values (printed as non-diffed `[info]` lines, since
  they can shift by a few ULP across GPU architectures — see `src/main.cu`'s output-contract note):
  `roundtrip` max error 0.00000 px; `straightness_rectified` boundary spread 0.7395 px;
  `distortion_negative_control` (raw, uncorrected) boundary spread 1.3227 px; `color_fidelity`
  smooth-region mean error 0.1463 / 255 (edge region 7.8932 / 255, reported only); `resize_conservation`
  max channel-mean drift 0.2059 / 255; `normalize` |mean| ~4.5e-8, |std-1| ~4e-6; `fused_vs_staged`
  max difference 0.0187 normalized-tensor units.
- Timing (teaching artifact, not a benchmark, CLAUDE.md §12): staged pipeline ~0.35-0.7 ms total,
  fused pipeline ~0.25-0.6 ms total (single-shot, JIT-inclusive); derived memory traffic for the
  undistort+rectify+resize portion: staged moves 2,073,600 bytes, fused moves 1,410,048 bytes — a
  **32.0% reduction**, purely from not writing/reading the intermediate full-resolution image.

The canonical stable lines live in [`demo/expected_output.txt`](demo/expected_output.txt).

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: loads the sample, runs both pipelines, verifies against
   the CPU oracle, runs the seven gates, writes artifacts.
2. [`src/kernels.cuh`](src/kernels.cuh) — the camera model (single source of truth), every image
   layout, and every launcher's contract. Read this before anything else in `src/`.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels. Start with `debayer_kernel`, then
   `build_remap_lut_kernel` + `bilinear_sample_rgb`, then compare `remap_bilinear_kernel` +
   `resize_area2x_kernel` (STAGED) against `fused_kernel` (FUSED) side by side — that comparison IS
   the project. Finish with the three `normalize_*_kernel`s and their determinism argument.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins (see its header for
   exactly which functions are shared with the GPU path and why).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h` (data-file / artifact-dir
   resolution across the VS/run_demo/CMake launch layouts).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **libargus** (NVIDIA Jetson) — the production camera-capture + ISP API that runs debayer, lens
  correction, and tone-mapping on dedicated ISP hardware, not the GPU; this project's stages mirror
  its pipeline shape.
- **V4L2 + a vendor ISP driver** (Linux) — the standard kernel-level camera capture interface; most
  embedded Linux camera stacks sit on top of it.
- **OpenCV `cv::cuda`** — `cv::cuda::demosaicing`, `cv::cuda::remap`, `cv::cuda::resize` are the
  production, heavily-optimized GPU equivalents of every stage here; `cv::initUndistortRectifyMap`
  builds exactly the kind of LUT this project's `build_remap_lut_kernel` computes.
- **NVIDIA VPI (Vision Programming Interface)** — a production Jetson/desktop library with a
  dedicated `Remap` algorithm and hardware-accelerated backends (PVA, VIC) beyond the GPU alone.
  ArrayFire and NPP (NVIDIA Performance Primitives) offer similar building blocks.
  Where this repo's teaching build stops, tuned kernels like theirs take over.
- **Malvar-He-Cutler demosaicing** (Malvar, He & Cutler, 2004) — the gradient-corrected demosaic
  variant named in `THEORY.md`; this project implements the simpler bilinear baseline and documents
  MHC as the improved next step (README Exercise 2).
- **Kalibr / ROS `camera_calibration`** — the standard tools that would actually MEASURE the k1/k2/p1/p2
  and fx/fy/cx/cy this project hardcodes from a real checkerboard capture.

## Exercises

1. **Visualize the LUT.** Write the remap LUT's `(u-x, v-y)` displacement field as a false-color PPM
   (hue = direction, brightness = magnitude) — see visually where distortion is strongest (corners)
   and where the rectifying rotation dominates (which direction does it shift?).
2. **Implement Malvar-He-Cutler demosaicing** and compare its `color_fidelity` gate value against the
   bilinear baseline's 0.1463 — MHC's gradient-correction term should measurably reduce error near
   the checkerboard's high-frequency edges (the `edge-region` number this project reports but does not
   gate on).
3. **Fuse debayer into the remap kernel too.** `kernels.cuh`'s file header names this "possible in
   principle but hairy" — try it: the fused kernel would need to bilinear-sample the RAW Bayer
   mosaic directly (four different color-plane interpolations depending on which of the four RGGB
   phases each sample position lands in) instead of a clean 3-channel image. Measure whether the
   extra code complexity is worth the further memory-traffic reduction.
4. **Swap the deterministic normalize reduction for an atomicAdd-based one** and verify (by running the
   demo many times) that the atomic version's mean/std occasionally differ in the last few bits while
   this project's tree-based version does not — the concrete lesson behind THEORY.md's determinism
   argument.
5. **Add a second rectifying axis** (rotation about X as well as Y) and re-derive whether the
   straightness gate's boundary (a vertical line) still isolates rotation from distortion cleanly, or
   whether a second gate (e.g. a horizontal boundary) becomes necessary.

## Limitations & honesty

- **One shared intrinsic matrix K.** The raw (distorted) and rectified (undistorted) cameras share
  one `(fx, fy, cx, cy)` — a real calibration often lets the rectified camera use a *different*
  (usually slightly smaller) K to control how much border gets cropped. This project keeps K fixed to
  isolate the distortion-removal and rotation lessons from that separate concern (named in
  `kernels.cuh`'s file header).
- **No focal-length change on resize's boundary.** Undistort+rectify does not change resolution (only
  the RESIZE stage does) — a real ISP sometimes folds a resolution change into the same remap pass;
  this project keeps them as separate, individually-teachable stages (and re-fuses them deliberately
  in the FUSED kernel to teach exactly that folding).
- **Debayer stays outside the fusion.** The catalog's "zero CPU copies" pipeline never leaves the
  GPU, but the FUSED kernel specifically only fuses undistort+rectify+resize — Exercise 3 explores
  why fusing debayer in too is possible but harder.
- **Global shutter assumed.** Rolling-shutter skew (each row exposed at a slightly different time,
  common on cheap CMOS sensors) is not modeled; `THEORY.md` "Where this sits in the real world"
  discusses it honestly as a limitation of every stage here.
- **Synthetic-only.** The scene is fully analytic; no real sensor noise (shot noise, read noise,
  fixed-pattern noise, hot pixels) is modeled — a real ISP's debayer stage also has to be robust to
  all of those, which this teaching version does not need to be.
- **No safety implication.** This project only transforms images; it never commands motion. The
  general repo caveat (CLAUDE.md §1) still applies to anything built on top of it: sim-validated only,
  not safety-certified.
