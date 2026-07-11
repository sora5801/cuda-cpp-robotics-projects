# 01.13 — Canny + Hough line/circle detection for industrial alignment

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Canny + Hough line/circle detection for industrial alignment`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds the classic industrial machine-vision alignment pipeline, end to end, on the GPU:
**Canny edge detection** (Gaussian smoothing, Sobel gradients, non-max suppression, double-threshold
hysteresis) feeding **two Hough transforms** (a full line-parameter sweep and a known-radius circle
search), whose detections are fed into a small **least-squares rigid-alignment solve**. The demo scene
is a synthetic machined plate — 4 straight edges, 3 drilled holes of known distinct radii — rendered
under a *known* in-plane offset and rotation, exactly the kind of error a fixture, conveyor belt, or
robot pick introduces on a real inspection line. The pipeline recovers that offset from pixels alone and
reports it, which is *literally* what a factory vision station does before a robot corrects its grip or
a PLC rejects a mis-seated part.

Every stage named in the catalog bullet is implemented for real, not stubbed: both Hough transforms run
as genuine GPU kernels with their own CPU twins; the double- vs. single-threshold hysteresis comparison
is a deliberately engineered scene feature (a weak-but-connected "scratch mark"), not just a code path;
and the alignment solve uses the *detected* circle centers, not the ground truth, cross-checked against
it only for grading. Two things are honestly simplified and documented as such throughout: (1) the
non-max-suppression step uses the classic 4-sector direction quantization rather than interpolated NMS,
and (2) this project works entirely in pixel units — it does not model a camera's mm-per-pixel intrinsic
calibration (see [Limitations & honesty](#limitations--honesty) and [`PRACTICE.md`](PRACTICE.md) §3 for
how a real station adds that).

## What this computes & why the GPU helps

Three different GPU parallelization patterns, back to back, on one image:

- **Canny (map + bounded stencil):** Gaussian blur, Sobel gradients, and non-max suppression are each a
  *map* — one thread per pixel, reading a small fixed neighborhood, writing one output. Pure
  memory-bandwidth-bound work with zero data dependency between threads.
- **Hysteresis (iterative propagation):** a CCL-style repeated-sweep kernel where a weak pixel's fate
  depends on its neighbors' *current* state — the whole image is re-scanned every sweep until nothing
  changes, the same fixed-point pattern project 01.06's connected-component labeling uses.
- **Hough voting (scatter with atomics):** one thread per *edge* pixel fans out into many
  `atomicAdd`-based writes into a shared accumulator — a genuinely data-dependent workload (only ~1-2%
  of pixels are edges) and this project's clearest illustration of when scatter beats gather, and of why
  **integer atomics make the accumulator ORDER-INDEPENDENT and bit-exact**, in sharp contrast to
  float-atomic accumulation elsewhere in this repo, whose results are only reproducible run-to-run, not
  bit-identical to an independent CPU implementation.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — the classical (non-learned) machine-vision layer that sits right
  after image acquisition and feeds a geometric world model, not a neural one. It is the **workhorse of
  industrial machine vision**: fixture/part alignment, pick verification, and dimensional gauging all
  run some variant of this exact pipeline.
- **Upstream inputs:** a rectified `Image` (undistorted, geometrically corrected — project **01.01**'s
  GPU image pipeline) that has been **flat-fielded** (uniform illumination — project **01.09**'s
  photometric vignetting calibration); metrology-grade alignment needs *both*, since uncorrected lens
  distortion or vignetting silently biases every recovered (dx, dy, dtheta).
- **Downstream consumers:** the work-cell PLC or robot controller's **offset-correction** logic — a
  6-DoF manipulator (project domain **19**, manipulation & grasping) that adjusts its pick pose by
  exactly the recovered `AlignmentResult`, or a reject/accept signal on a conveyor gate.
- **Rate / latency budget:** a real inspection cell runs at the line's **parts-per-minute** rate — a
  120 ppm line gives ~500 ms per part, camera trigger to PLC handshake, of which vision processing is
  usually a small slice (tens of ms) so the robot/actuator has time to act. This implementation's
  *measured* pipeline time on the committed sample (RTX 2080 SUPER) is in the single-digit milliseconds
  (see the `[time]` line in `demo/expected_output.txt`'s companion run) — comfortably inside that budget
  even before considering that a real 2-5 MP industrial camera image is far larger than this project's
  teaching-sized 320x240.
- **Reference robot(s):** the **6-DoF manipulator work cell** (this project's primary reference robot —
  fixture alignment feeding a pick/place or assembly task) and, secondarily, the **AMR** reference robot
  during **docking**, where the exact same line+alignment idea locates a charging dock or pallet marker.
- **In production:** OpenCV's or NVIDIA NPP's Canny/Hough (GPU-accelerated, heavily tuned) for the
  detection stages, and a proper 2-D/3-D pose solver (or a commercial metrology suite — see
  [Prior art](#prior-art--further-reading)) for the alignment fit; camera calibration (project **01.16**)
  turns this project's pixel-space answer into millimeters.
- **Owning team:** machine vision / manufacturing engineering — see [`PRACTICE.md`](PRACTICE.md) §4 for
  the fuller org picture (adjacent teams, typical titles, and the regulatory/business context).

## The algorithm in brief

- **Canny edge detection** — separable Gaussian smoothing, Sobel gradients, 4-sector-quantized
  non-max suppression, double-threshold classification, and CCL-style iterative hysteresis promotion.
  See [`THEORY.md`](THEORY.md) "The algorithm" and "The GPU mapping".
- **Hough line transform** — point-line duality accumulator over (theta, rho), built by one
  `atomicAdd`-scatter kernel per edge pixel across all 180 candidate angles, with a **fixed-point Q16
  theta table** that makes the accumulator provably bit-exact between GPU and CPU. See
  [`THEORY.md`](THEORY.md) "The math" (the duality derivation) and "Numerical considerations".
- **Hough circle transform, known-radius + gradient-directed** — because the drilled holes have known,
  distinct nominal radii, the search collapses from a 3-D `(cx,cy,r)` volume to `NUM_HOLES` independent
  2-D planes, and each edge pixel votes at only 2 candidate centers per plane (along its own measured
  gradient direction) instead of an entire candidate circle. See [`THEORY.md`](THEORY.md) "The
  algorithm".
- **Sub-bin/sub-pixel refinement** — a 3-point parabola fit on each accumulator axis, the same
  closed-form idea project 01.02's stereo disparity refinement uses. See [`THEORY.md`](THEORY.md) "The
  math".
- **Rigid alignment least squares** — a small 4-unknown `(cos, sin, tx, ty)` linear normal-equations
  solve from the detected hole correspondences (the production-scale version of this exact idea is
  project 33.01's batched small-matrix linalg). See [`THEORY.md`](THEORY.md) "The math".

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/canny-hough-line-circle-detection-for-industrial.sln`](build/canny-hough-line-circle-detection-for-industrial.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/canny-hough-line-circle-detection-for-industrial.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none. This project links only the CUDA runtime and the C++17
standard library — every stage (Canny, both Hough transforms, the alignment solve) is hand-rolled, on
purpose (see [Prior art](#prior-art--further-reading) for what a production stack would use instead).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is fully synthetic (CLAUDE.md §8's default) — a rendered scene of a machined plate
with known ground-truth geometry and a known applied alignment offset, plus a "negative control" image
with no plate at all. `scripts/make_synthetic.py --seed 42` regenerates it byte-for-byte; no public
dataset applies here (see `data/README.md` for why a synthetic scene actually teaches *more* than a
photograph would for this specific task — the ground truth a photo lacks is exactly what alignment
gates need). Full provenance, checksums, and per-file field documentation: [`data/README.md`](data/README.md).

## Expected output

The demo prints a `VERIFY:` line per twin category — float-tolerance for Gaussian/Sobel/NMS, **exact**
integer equality for the hysteresis edge map, and a **bit-exact** integer-atomics comparison for the
Hough line accumulator (this project's headline determinism result; the Hough circle accumulator is
verified at peak level instead, an honestly-documented exception — see `THEORY.md` "How we verify
correctness") — followed by 6 independent `GATE <name>: PASS` lines (`line_recovery`, `circle_recovery`,
`alignment`, `edge_quality`, `hysteresis_lesson`, `negative_control`) and a final `RESULT: PASS` only if
every verify and every gate holds. It also writes 4 artifacts to `demo/out/`: the Canny edge map, a
log-stretched Hough line accumulator image, an overlay of detected lines/circles/alignment vector on the
original scene, and a CSV of every gate's measured value. The canonical stable lines live in
[`demo/expected_output.txt`](demo/expected_output.txt); `demo/README.md` explains every artifact and
every line prefix.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced data contracts: image/part geometry, Canny
   thresholds, both Hough accumulator layouts (including the fixed-point theta table), and every
   kernel/launcher/CPU-oracle declaration. Read this first — everything else assumes it.
2. [`src/kernels.cu`](src/kernels.cu) — the 8 GPU kernels, in pipeline order. Start with
   `hough_lines_vote_kernel` for the scatter-with-atomics case study, or `hysteresis_propagate_sweep_kernel`
   for the iterative-propagation pattern.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins, including the
   queue-based flood fill that reaches the *same* hysteresis fixed point via a structurally different
   algorithm than the GPU's synchronous sweeps.
4. [`src/main.cu`](src/main.cu) — orchestration: data loading, the verify stage (GPU vs. CPU, stage by
   stage), the host-only peak-extraction/alignment-solve analysis, all 6 gates, and artifact writing.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data-file/artifact-directory
   resolution.
6. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the scene renderer; read alongside
   `kernels.cuh` SECTION 2/3, whose geometry constants it deliberately duplicates.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **OpenCV** (`cv::Canny`, `cv::HoughLines`/`HoughLinesP`, `cv::HoughCircles`, and the CUDA module's GPU
  versions) — the industry-standard open-source implementation of every stage here; its Hough circle
  transform also smooths the accumulator before peak-picking, the same trick this project's
  `extract_circle_peaks` uses.
- **NVIDIA Performance Primitives (NPP)** — `nppiFilterCannyBorder` and friends: a vendor-tuned,
  production-grade GPU Canny implementation with the same conceptual stages as this project's kernels.
- **Duda & Hart (1972), "Use of the Hough Transformation to Detect Lines and Curves in Pictures"** — the
  original (theta, rho) parameterization this project's line accumulator implements directly.
  Canny (1986), "A Computational Approach to Edge Detection" — the original algorithm, including the
  double-threshold hysteresis idea this project's engineered scratch mark demonstrates.
- **Halcon (MVTec) and Cognex VisionPro/In-Sight** — the commercial machine-vision metrology suites a
  real factory inspection station runs; both include calibrated line/circle "caliper" tools that are
  the production-grade descendants of this project's Hough peaks, plus certified gauge repeatability
  tooling (see [`PRACTICE.md`](PRACTICE.md) §4).
- **Project 33.01 (batched small-matrix linalg)** — the GPU-batch-scale version of this project's tiny
  4x4 alignment-solve normal equations.
- **Project 01.16 (checkerboard/ChArUco detection)** — the camera calibration this project's pixel-only
  answer needs before it means millimeters on a real robot.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Plot the accumulator.** Load `demo/out/hough_lines_accum.pgm` in any image viewer and identify the
   4 bright peaks by eye; compute by hand which (theta, rho) each corresponds to and check it against
   the `[info]` lines the demo printed.
2. **Interpolated NMS.** Replace the 4-sector quantized non-max suppression with true bilinear
   interpolation of the two off-grid neighbor magnitudes (THEORY.md "Numerical considerations" describes
   the trade-off this exercise resolves) and re-measure the `edge_quality` gate's precision/recall.
3. **Gradient-informed theta windowing.** This project's line kernel sweeps all 180 theta bins per edge
   pixel (the didactic default); implement the production optimization — only sweep a narrow window of
   theta bins near the pixel's OWN gradient direction (perpendicular to the edge) — and measure the
   speed-up and any accuracy change.
4. **Separable accumulator smoothing.** `extract_circle_peaks`'s windowed sum is a brute-force
   `O(W*H*(2k+1)^2)` box filter; replace it with a separable two-pass prefix-sum box filter and measure
   the speed-up on a larger image.
5. **A second scene.** Regenerate `data/sample/` with a different `--dx/--dy/--dtheta-deg` (or a
   different seed) via `scripts/make_synthetic.py`, rebuild, and confirm every gate still passes —
   or, deliberately push the offset large enough that a hole rotates outside the image, and watch which
   gate fails first.

## Limitations & honesty

- **4-sector NMS, not interpolated.** The non-max-suppression step quantizes the gradient direction to
  the nearest of 4 axis/diagonal directions rather than interpolating the two off-grid neighbor
  magnitudes. This can thicken an edge by roughly one extra pixel at unlucky angles — quantified against
  this project's own `edge_quality` gate in `THEORY.md`.
- **Pixel units only — no camera calibration.** This project never converts pixels to millimeters; a
  real inspection station needs an intrinsic/extrinsic calibration (project 01.16) to do that. The
  recovered `(dx, dy, dtheta)` is therefore an image-plane measurement, not a metrology-certified one.
- **Full 180-bin theta sweep, not the production-optimized gradient-windowed version.** Implemented
  didactically in full (see Exercise 3) — the windowed optimization is documented but not implemented.
- **Hough circle accumulator is not bit-exact.** Unlike the line accumulator's fixed-point-table
  determinism, the circle accumulator's vote position depends on the (float-tolerance-only) Sobel
  gradient direction; it is verified at the peak level instead, honestly documented in
  `reference_cpu.cpp` and `THEORY.md`.
- **Windowed circle-peak extraction has a radius-collision limitation.** Nominal radii closer together
  than the peak-extraction window's reach can cross-contaminate each other's accumulator plane (a real
  effect, measured and fixed on this scene by choosing the window radius smaller than any pairwise
  radius gap — see `main.cu`'s `CIRCLE_PEAK_WINDOW` comment). A production system would separate
  candidate radii by more than this margin, or use a genuinely 3-D-aware peak search.
- **Synthetic scene, single lighting/texture model.** The brushed-metal texture, vignette, and noise are
  simplified analytic models, not a physically based rendering — see `THEORY.md` "The problem" for the
  honest physics this stands in for, and one paragraph on backlight-vs-brightfield lighting choices a
  real station makes that this project does not model.
- **Not safety-certified; no motion is commanded here.** This project only computes a measurement; if
  that measurement were ever wired to move a real robot (as `PRACTICE.md` §3 describes), the same
  sim-validated-only caveat as every control/planning project in this repo applies (CLAUDE.md §1).
