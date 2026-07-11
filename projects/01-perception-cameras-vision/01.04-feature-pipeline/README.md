# 01.04 — Feature pipeline: FAST/Harris detection, ORB descriptors, brute-force Hamming matcher

**Difficulty:** ★ beginner · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `★ Feature pipeline: FAST/Harris detection, ORB descriptors, brute-force Hamming matcher`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds the classical **sparse-feature front end** of visual odometry (VO) and SLAM,
entirely on the GPU, and proves it works by recovering a **known** camera motion from two synthetic
images. It detects corners two ways — FAST-9 (fast, all-integer, the workhorse of real-time VO) and
Harris (the textbook structure-tensor detector, taught side by side for comparison) — describes each
FAST corner with an oriented, rotation-compensated 256-bit binary descriptor (ORB's "oriented rBRIEF"),
and matches descriptors between two views with a brute-force Hamming-distance search. The two views are
the SAME analytically-rendered scene under a **known** rotation + translation + brightness change, so
the demo can check its own answer: do the matches recovered by the pipeline actually correspond to the
true camera motion? Every component named in the catalog bullet (FAST, Harris, ORB, brute-force
Hamming matching) is fully implemented — nothing here is documented-only.

## What this computes & why the GPU helps

Three stages, three different GPU access patterns, all embarrassingly parallel once you see the shape
of the work:

- **Detect** — a per-PIXEL **map** (every pixel's FAST score / Harris response is independent of every
  other pixel) followed by a per-pixel **stencil + atomic-compaction** step (3x3 non-max suppression).
- **Describe** — detection cuts ~65,536 pixels down to a few hundred keypoints; the natural mapping
  flips to **one thread per keypoint**, each doing more work (a ~700-sample disk sum, 256 pixel-pair
  comparisons) over a much smaller list.
- **Match** — brute-force Hamming matching is a **map over queries**, each thread scanning every train
  descriptor with a tight integer inner loop (`popcount(a XOR b)` via the hardware `__popc` instruction).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception, at the boundary with State estimation — this project's OUTPUT
  (matched keypoint pairs) is exactly the raw material a visual-odometry / visual-SLAM front end feeds
  to its back-end optimizer.
- **Upstream inputs:** a rectified, undistorted grayscale `Image` — named explicitly: project **01.01**
  (`Full GPU image pipeline`, this domain's flagship) is the canonical upstream stage, ending exactly
  where this project begins (its `rectified.ppm`/normalized output is the kind of frame this project's
  `scene_a.pgm`/`scene_b.pgm` stand in for, synthetically). A raw, un-rectified frame would corrupt the
  rotation-recovery gate below, since the pinhole/rotation math this project's ground-truth transform
  assumes presumes a calibrated, distortion-free image.
- **Downstream consumers:** the matched keypoint pairs (this project's `matches.csv`) are the classic
  input to **05.xx SLAM & mapping** (visual odometry front ends estimate frame-to-frame motion from
  exactly this kind of correspondence set — e.g. essential-matrix / PnP pose estimation, bundle
  adjustment) and to **01.16/01.17 calibration** projects, which use the SAME detect-and-match idea
  (chessboard/ArUco corners, multi-camera correspondence) for a different purpose.
- **Rate / latency budget:** 30-60 Hz camera input (SYSTEM_DESIGN.md §1.1); a feature front end
  typically must finish well under one frame period (a few milliseconds is the real-time target for VGA-
  scale images) so the downstream state estimator gets a fresh correspondence set every frame, not a
  stale one.
- **Reference robot(s):** the **quadrotor** (SYSTEM_DESIGN.md §2.4 — visual-inertial odometry's `STATE
  ESTIMATION [04]` block consumes exactly this kind of feature track) and the **warehouse AMR**
  (SYSTEM_DESIGN.md §2.1 — vision-based loop closure / relocalization inside `LOCALIZATION & MAPPING
  [04, 05]` uses the same detect-describe-match primitive, alongside the AMR's primary LiDAR sensing).
- **In production:** a calibrated camera driver feeding rectified frames at 30-60 Hz; GPU-accelerated
  feature extraction (NVIDIA VPI, OpenCV's `cv::cuda` module) or a learned front end (SuperPoint/
  SuperGlue-class networks) in place of this project's hand-rolled FAST/ORB; a robust estimator
  (5-point/8-point RANSAC, or a full visual-SLAM stack like ORB-SLAM3) consuming the matches instead of
  this project's four teaching gates.
- **Owning team:** Perception (with close collaboration with the SLAM/state-estimation sub-team on the
  correspondence format) — see [`PRACTICE.md`](PRACTICE.md) §4 for the fuller organizational picture.

## The algorithm in brief

- **FAST-9 corner detection** — the Bresenham circle of 16 pixels, a contiguous-9-point brightness-arc
  test, with the classical "high-speed" 4-point pre-filter. See [`THEORY.md`](THEORY.md#the-algorithm).
- **Harris corner detection** — Sobel gradients, a box-windowed structure tensor, the
  `det(M) - k*trace(M)^2` response. See [`THEORY.md`](THEORY.md#the-algorithm).
- **3x3 non-max suppression + top-N selection** — both detectors, GPU stencil + atomic compaction, then
  a deterministic host-side sort. See [`THEORY.md`](THEORY.md#the-gpu-mapping).
- **Intensity-centroid orientation** — Rosin's corner-orientation estimator, quantized into 30 discrete
  bins (matching OpenCV ORB's own implementation choice). See [`THEORY.md`](THEORY.md#the-math).
- **Oriented rBRIEF (ORB) description** — 256 rotated, precomputed intensity-pair comparisons packed
  into a 256-bit descriptor. See [`THEORY.md`](THEORY.md#the-algorithm).
- **Brute-force Hamming matching** — all-pairs `popcount(XOR)`, best + second-best per query, Lowe
  ratio test + mutual-consistency cross-check. See [`THEORY.md`](THEORY.md#the-algorithm).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/feature-pipeline.sln`](build/feature-pipeline.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/feature-pipeline.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only
(no cuBLAS/cuFFT/Thrust/CUB; every kernel here is hand-rolled, per this project's "no black boxes"
teaching goal).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is fully synthetic (CLAUDE.md §8 default): two analytically-rendered 256x256
grayscale views of the SAME scene under a KNOWN similarity transform (`scene_a.pgm`, `scene_b.pgm`),
plus one UNRELATED scene (`neg_scene_c.pgm`) used only as a negative control. Regenerate with
`python scripts/make_synthetic.py` (deterministic — byte-identical output every run). Full provenance,
checksums, and field documentation: [`data/README.md`](data/README.md).

## Expected output

`RESULT: PASS` requires FOUR `VERIFY(...)` lines and FOUR `GATE ...` lines to all read `PASS`:

- **`VERIFY(fast)`** — the FAST-9 score map and the final, sorted keypoint list agree **bit-for-bit**
  between the GPU kernel and the independent CPU reference (`src/reference_cpu.cpp`) — an all-integer
  computation has no floating-point rounding to excuse a mismatch.
- **`VERIFY(harris)`** — the Sobel gradients (exact) and the Harris response map (float) agree within a
  documented **relative** tolerance (2e-3; measured 5.2e-4 on the committed sample — Harris responses
  span ~13 orders of magnitude on this scene, so a relative bound is the honest comparison; see
  `THEORY.md`).
- **`VERIFY(describe)`** — per-keypoint orientation agrees within tolerance (0.01 rad; measured exactly
  0 on the committed sample) AND lands in the same 12-degree bin as the CPU reference for every
  keypoint; the 256-bit ORB descriptors then agree **bit-for-bit**.
- **`VERIFY(hamming)`** — the brute-force best/second-best Hamming distances (both match directions)
  agree **bit-for-bit** (pure integer popcount + reduction).
- **`GATE ground_truth_transform`** — of the accepted A-B matches, ≥90% must land within 5.0 px of
  where the KNOWN transform predicts (measured: 92.3%, 60/65).
- **`GATE rotation_recovery`** — the median orientation delta of matched pairs must equal the known
  12.0-degree rotation within 1.0 deg (measured: 11.51 deg, |error| = 0.49 deg).
- **`GATE repeatability`** — ≥50% of scene-A FAST keypoints must have a scene-B FAST keypoint within 3 px
  of their transformed location — bypasses descriptors/matching entirely (measured: 63.7%, 86/135).
- **`GATE negative_control`** — matches against the UNRELATED scene, scored the SAME way as Gate 1, must
  land near the transform ≤10% of the time (measured: 0.0%, 0/17) — proof the matcher/gates are not
  self-confirming.

The canonical stable-line contract lives in [`demo/expected_output.txt`](demo/expected_output.txt); the
full measured numbers (including the `[info]`-line detail) are recorded in `src/main.cu`'s tolerance
comment block and in `THEORY.md` "How we verify correctness".

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **start here.** The single-sourced contract: image geometry,
   FAST/Harris/ORB constants, every shared struct, and — most importantly — the header comment
   explaining exactly how this project achieves BIT-EXACT descriptors despite a TOLERANT orientation
   angle (the project's central design decision).
2. [`src/kernels.cu`](src/kernels.cu) — the eight GPU kernels: `fast_score_kernel`,
   `sobel_gradient_kernel` + `harris_response_kernel`, the two `nms_select_*_kernel`s,
   `orientation_kernel`, `describe_kernel`, `hamming_match_kernel`.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of every kernel above;
   read it side by side with `kernels.cu` to see what "the same algorithm, twice" looks like.
4. [`src/main.cu`](src/main.cu) — orchestration: loads the three scenes, runs detect/describe/match on
   the GPU, runs and compares every CPU twin, computes the four ground-truth gates, writes the
   artifacts. The most interesting kernel to look at first is `describe_kernel` (`kernels.cu`) — the
   256-comparison rBRIEF loop is the heart of ORB.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s multi-candidate data/output
   directory resolution.
6. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the analytic scene renderer and the
   ground-truth transform, in Python (the language-independent twin of `main.cu`'s transform math).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Rosten & Drummond, "Machine learning for high-speed corner detection" (2006)** — the original FAST
  paper, including the trained-decision-tree speedup this project's textbook quick-test stands in for.
- **Harris & Stephens, "A Combined Corner and Edge Detector" (1988)** — the original structure-tensor
  corner detector.
- **Rublee, Rabaud, Konolige, Bradski, "ORB: an efficient alternative to SIFT or SURF" (2011)** — the
  oriented-rBRIEF descriptor this project implements, including the real (learned, not random) sampling
  pattern and the 30-bin discretized-rotation trick this project deliberately reproduces.
- **OpenCV (`cv::cuda::ORB`, `cv::cuda::FastFeatureDetector`, `cv::BFMatcher`)** — the production,
  battle-tested implementation of everything here; read its ORB source (`orb.cpp`) for the real
  `bit_pattern_31_` sampling table this project's random one stands in for.
- **NVIDIA VPI (Vision Programming Interface)** — GPU-accelerated FAST/Harris/ORB as a shipped SDK on
  Jetson and dGPU, the production analogue of every kernel in this project.
- **ORB-SLAM3 (Campos, Elvira, Rodríguez, Montiel, Tardós, 2021)** — the reference real-time visual-
  SLAM system built on exactly this front end (FAST + oriented BRIEF + brute-force/bag-of-words
  matching); this project's `01.04` numbering deliberately mirrors ORB-SLAM3's own pipeline shape.
- **SuperPoint / SuperGlue (DeTone, Malisiewicz, Rabinovich 2018; Sarlin et al. 2020)** — the modern,
  learned successor to hand-crafted detect-describe-match, named honestly in
  [`THEORY.md`](THEORY.md#where-this-sits-in-the-real-world) as where this field is headed.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. Change `kFastThreshold` in `kernels.cuh` (try 10 and 40) and re-run the demo. Watch the FAST
   keypoint counts and `GATE repeatability`'s fraction move — a direct, hands-on look at the classic
   detector precision/recall trade-off.
2. Add a THIRD detector-comparison metric: instead of (or in addition to) `harris_vs_fast_overlap`,
   compute the fraction of Harris-A keypoints that pass the SAME repeatability test (transform +
   nearest-Harris-B-keypoint) FAST keypoints do — do the two detectors have similar repeatability, even
   though they disagree on WHICH pixels are corners?
3. Implement a k-d tree (or a grid-based spatial hash) for the matcher and compare its result against
   `hamming_match_cpu`'s brute-force answer — they should agree exactly on which matches are found
   (though not necessarily explore descriptors in the same order); time both and see at what `nTrain`
   the smarter structure starts winning (THEORY.md "Where this sits in the real world" discusses why
   binary descriptors keep brute force competitive far longer than float descriptors do).
4. Swap `kOrientBins` from 30 to, say, 8 or 60 and observe `verify(orientation bin agreement)` — at
   what bin count does GPU/CPU orientation-measurement noise start landing keypoints in different bins
   (breaking the bit-exact descriptor guarantee kernels.cuh's header derives)?
5. Replace `build_orb_base_pattern`'s random sampling pattern with a small, hand-designed one (e.g., a
   regular grid or a spiral) and measure `GATE ground_truth_transform`'s inlier fraction — does a
   "smarter" pattern out-perform random sampling, and by how much? (OpenCV's real, LEARNED pattern is
   the answer the ORB paper gives — see "Prior art" above.)

## Limitations & honesty

- **The ORB sampling pattern is random, not learned.** Real ORB (OpenCV) uses a pattern trained offline
  for low bit-correlation and high variance across a large image corpus; this project's
  `build_orb_base_pattern()` uses an isotropic, seeded-random pattern instead (kernels.cuh's header
  explains why this is an honest substitute, not a shortcut, for teaching the MECHANICS of oriented
  rBRIEF — every step is faithful; only the specific 256 pairs differ from OpenCV's).
- **Harris keypoints never reach description/matching.** The catalog bullet's "Harris detection" is
  fully implemented and independently GPU-vs-CPU verified, but only FAST keypoints feed the describe/
  match/gate pipeline (mirroring real ORB-SLAM-style systems, which are built on FAST, not Harris).
  Harris is reported via a detection-only comparison (`harris_vs_fast_overlap`) — see `THEORY.md`.
- **256x256 images, not VGA/HD.** Kept small so the demo (GPU + CPU reference, three images) runs in
  milliseconds with zero setup; the algorithms themselves are resolution-independent and the same
  kernels would run unmodified — just slower on the (still-simple, unoptimized) CPU twin — at 640x480
  or 1920x1080.
- **No lens distortion, no real sensor noise.** The synthetic scenes are analytically rendered and
  anti-aliased (see `data/README.md`), but carry none of a real camera's vignetting, Bayer-pattern
  demosaicing artifacts, or photon noise — project **01.01** teaches that pipeline stage; this project
  assumes it already happened.
- **Not safety-certified; sim-validated only.** This is a perception building block, not a control or
  actuation system — it never commands hardware motion directly — but if a downstream project of yours
  chains this into anything that DOES move a real robot, treat every claim here as simulation-only until
  independently validated on real hardware (CLAUDE.md §1, §8).
