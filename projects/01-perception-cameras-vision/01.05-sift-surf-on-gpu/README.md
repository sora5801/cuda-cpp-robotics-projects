# 01.05 — SIFT/SURF on GPU (harder, warp-level reductions)

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `SIFT/SURF on GPU (harder, warp-level reductions)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A from-scratch, GPU-accelerated implementation of SIFT (Scale-Invariant Feature Transform): a Gaussian
scale-space pyramid, Difference-of-Gaussian (DoG) extrema detection with sub-pixel/sub-scale refinement,
warp-cooperative orientation-histogram and 128-D descriptor computation, and brute-force L2 matching —
verified stage-by-stage against an independent CPU reference and checked against six ground-truth gates
on a synthetic image pair related by a **known 1.5x zoom + 20° rotation** (a REAL scale change, the
capability 01.04's single-scale FAST/ORB pipeline cannot demonstrate). SURF is **documented-only**:
`THEORY.md` teaches its integral-image and box-filter-Hessian ideas to implementable depth, but no SURF
code ships — see [Limitations & honesty](#limitations--honesty) for the scoping rationale. A learner who
studies this project will understand: why Gaussian is the mathematically unique scale-space kernel; how
DoG approximates the scale-normalized Laplacian; how to refine a discrete extremum to sub-pixel/sub-scale
precision via a small linear solve; and — the catalog's specific hook — how to map a per-keypoint
histogram-reduction problem onto a WARP (32 threads, `__shfl_down_sync` tree reduction) instead of a
single thread or a naive shared-memory-atomic scheme, and why that choice matters.

## What this computes & why the GPU helps

SIFT is a PIPELINE of different computational patterns chained together, each parallelized differently:
Gaussian blur and DoG are **map/stencil** operations (one thread per pixel, exactly 01.04's precedent);
DoG-extrema detection is a **3x3x3 stencil + atomic compaction** (01.04's 2-D non-max-suppression pattern,
extended one dimension); sub-pixel refinement is a **per-candidate iterative solve** (one thread per
candidate, variable iteration count); and — the centerpiece — orientation assignment and descriptor
construction are **per-keypoint histogram reductions**, mapped one WARP per keypoint: each of the 32
lanes accumulates a private partial histogram over its share of the sampling patch (zero contention, by
construction), then a five-step `__shfl_down_sync` tree reduction folds the 32 partials into the final
histogram using only register-to-register communication — no shared memory, no atomics. Matching is a
**brute-force all-pairs** map (one thread per query descriptor).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception (feature extraction & matching), feeding State estimation / World model
  — specifically the loop-closure and relocalization sub-problem of SLAM & mapping.
- **Upstream inputs:** a rectified/undistorted grayscale `Image` from 01.01 (GPU image pipeline) — this
  project assumes calibration has already removed lens distortion, since SIFT's own scale/orientation
  math assumes locally-linear geometry.
- **Downstream consumers:** 05.xx SLAM projects' loop-closure and place-recognition stages (matched
  features + implied relative pose feed a pose-graph correction); 01.17-style wide-baseline calibration
  (matching two views of a calibration target from very different distances/angles) is a second natural
  consumer, named as a `PointCloud`/`Image`-adjacent "Matches" message shape in
  `docs/SYSTEM_DESIGN.md`'s interface conventions.
- **Rate / latency budget:** SIFT is the EXPENSIVE end of the feature-matching spectrum — this project's
  own measurement (Release, RTX 2080 SUPER, ~130-190 total keypoints across three 256x256 images) is
  ~7-9 ms of GPU kernel time; a real photograph with thousands of keypoints costs proportionally more.
  It is realistically run at KEYFRAME rate (a few Hz, event-triggered on "have I moved enough / do I
  need to check for a loop closure"), not every camera frame — contrast 01.04's ORB/Hamming pipeline,
  which is cheap enough to run near frame rate and is what a real-time front end actually uses.
- **Reference robot(s):** the warehouse AMR (loop closure / re-localization after a lift interruption)
  and the 6-DoF manipulator work cell (wide-baseline registration — recognizing a part or fixture from a
  very different viewpoint/distance than it was first observed).
- **In production:** `cv::SIFT`/`cv::cuda::SIFT` (patent-expired, in mainline OpenCV since 2020) at
  scale; increasingly, learned features (SuperPoint, DISK) that solve the same correspondence problem
  end-to-end from data — see `THEORY.md` "Where this sits in the real world" for the full comparison.
- **Owning team:** Perception (adjacent to SLAM/Mapping and Simulation & Tools) — see PRACTICE.md §4.

## The algorithm in brief

- **Gaussian scale space** — a 2-octave x 2-interval pyramid built by INCREMENTAL separable blurring
  (blur just enough extra sigma to reach the next level, not re-blur from scratch) — see `THEORY.md`
  "The math" for the variance-additivity identity this relies on, and "The problem" for why Gaussian is
  the axiomatically UNIQUE correct kernel.
- **Difference-of-Gaussian (DoG) extrema** — DoG approximates the scale-normalized Laplacian-of-Gaussian
  (Lindeberg's operator for scale-covariant blob detection); extrema are found via a 3x3x3 stencil
  (own layer + layer above + layer below) with atomic compaction — see `THEORY.md` "The math"/"The
  algorithm".
- **Sub-pixel/sub-scale refinement** — a quadratic Taylor fit around each candidate, refined via an
  iterative 3x3 linear solve (Cramer's rule), with Lowe's contrast-threshold and principal-curvature
  edge rejection — see `THEORY.md` "The math" (and project 33.01 for the batched-small-matrix-solve
  pattern this is a single-instance version of).
- **Orientation assignment** — a 36-bin gradient-magnitude histogram over a Gaussian-weighted patch,
  built via **warp-level reduction** (32 lanes, private partial histograms, `__shfl_down_sync` tree
  reduce) — the catalog's centerpiece hook; see `THEORY.md` "The GPU mapping".
- **128-D descriptor** — a 4x4 spatial x 8 orientation-bin trilinear-interpolated histogram, built with
  the SAME warp-cooperative pattern, then L2-normalized, clipped at 0.2, and re-normalized.
- **Brute-force L2 matching** — squared-L2 distance, Lowe ratio test + mutual cross-check, contrasted
  directly against 01.04's Hamming/binary matcher (see `THEORY.md` "Where this sits in the real world").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/sift-surf-on-gpu.sln`](build/sift-surf-on-gpu.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/sift-surf-on-gpu.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — this project links only the CUDA runtime + C++17
standard library (every kernel is hand-rolled: no cuBLAS/cuFFT/Thrust/CUB).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

What the committed sample is (synthetic by default, per CLAUDE.md §8), how to regenerate or download it,
and its licensing. Details and provenance in [`data/README.md`](data/README.md).

Three 256x256 grayscale PGMs, entirely synthetic (`scripts/make_synthetic.py`, seeds 42 and 999, no
downloads): `scene_a.pgm` (reference view), `scene_b.pgm` (the SAME scene under a KNOWN similarity
transform — 1.5x zoom + 20° rotation + translation + brightness offset — rendered by inverse-
transforming each output pixel back into scene A's analytic scene function, so the ground truth is
EXACT, never a raster-warp approximation), and `neg_scene_c.pgm` (an unrelated scene, seed 999, the
negative control). The scene content is hashed multi-scale checkerboard patches (continuous, not
alternating, per-cell colors — a SIFT-specific extension of 01.04's self-similarity fix, see
`THEORY.md`/`make_synthetic.py`) plus disks and a gradient background, deliberately spanning MULTIPLE
physical sizes so different octaves/intervals of the pyramid each have real content to detect.

## Expected output

What success looks like, and how the GPU result is checked against the CPU reference
(`src/reference_cpu.cpp`) within a documented tolerance. The canonical lines live in
[`demo/expected_output.txt`](demo/expected_output.txt).

The demo runs the full 6-stage pipeline on all three images, verifying GPU-vs-CPU agreement at every
stage (scale space, DoG extrema, refine, orientation, describe, match), then checks six independent
gates. On the reference machine (RTX 2080 SUPER, sm_75, Release):

```
VERIFY(scale space + detect + orient + describe): PASS   (max|gpu-cpu| ~3-6e-7 across the pyramid; 0 boundary-tie mismatches; theta/descriptor agreement below printable float32 precision)
VERIFY(match): PASS                                       (max|gpu-cpu| dist_sq = 1.192e-07)
GATE scale_recovery: PASS         (median matched-pair scale ratio 1.4281 vs ground truth 1.50, 4.8% error)
GATE rotation_recovery: PASS      (median delta -20.25 deg vs expected -20.0 deg -- see the sign-convention note in main.cu/THEORY.md)
GATE transform_inlier: PASS       (4/15 = 26.7% of accepted matches land within 6px of the known transform)
GATE scale_repeatability: PASS    (20/78 = 25.6% of scene-A keypoints re-found in scene B at the predicted location+scale)
GATE negative_control: PASS       (0/14 = 0% of A-vs-unrelated-scene matches land near the true transform)
GATE descriptor_normalization: PASS (every descriptor's L2 norm within 2.5e-7 of 1; max component 0.367 <= 0.42)
RESULT: PASS
```

Every number above is real, measured on this exact committed sample — never fabricated (CLAUDE.md §8).
`gates_metrics.csv` (written to `demo/out/`) carries the full metric/tolerance/verdict table.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: loads the three images, builds both pyramids (GPU + CPU),
   runs detect/orient/describe/match with staged verification, computes the six gates, writes artifacts.
2. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced data-layout contract: every struct,
   constant, and the shared Gaussian-weight-table builder, with the full twin-independence argument in
   its header comment.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels. Start with `gaussian_blur_h/v_kernel` (the
   simplest), then `dog_extrema_candidates_kernel` (01.04's NMS pattern, one dimension bigger), then
   `refine_keypoint_kernel` (the 3x3 solve), and finish with `orientation_kernel`/`describe_kernel` — the
   warp-level-reduction centerpiece, extensively commented on the "why a warp, why not atomics" argument.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle; its header explains
   exactly which numerical choices (float vs. double, sequential vs. warp-strided) were made deliberately
   to isolate specific sources of GPU-vs-CPU divergence.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (IJCV 2004)** — the original SIFT
  paper; every formula in `THEORY.md` traces back to this.
- **OpenCV `cv::SIFT` / `cv::cuda::SIFT`** — the production, patent-expired-since-2020 reference
  implementation; read its source after this project to see the identical algorithm at real scale.
- **VLFeat (`vl_sift`)** — Andrea Vedaldi's extremely well-documented open-source SIFT; the best place
  to check this project's small numerical choices (histogram smoothing, grid-edge interpolation) against
  a widely-trusted reference.
- **Bay et al., "SURF: Speeded-Up Robust Features" (ECCV 2006)** — the integral-image / box-filter-
  Hessian speed-oriented cousin this project teaches (THEORY.md) but does not implement — study for the
  full algorithm and the speed/discriminability tradeoff it represents.
- **DeTone, Malisiewicz, Rabinovich, "SuperPoint" (2018) and Tyszkiewicz et al., "DISK" (2020)** — the
  modern, LEARNED successors to hand-designed descriptors like SIFT — study these to see how a CNN
  sidesteps the exact discriminability limitation this project's own matching-threshold story measures.
- **Project 33.01 (batched small-matrix linear algebra)** — the same 3x3-solve pattern this project's
  `refine_keypoint_kernel` uses once per candidate, there studied at true batched GPU scale.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Visualize the scale space.** Dump every Gaussian and DoG pyramid level as a PGM (extend `main.cu`'s
   artifact-writing code) and inspect how a checkerboard corner's DoG response changes across octaves.
2. **Measure the naive shared-atomic histogram's cost.** Implement the "naive" alternative
   `orientation_kernel`'s header describes (each lane does `atomicAdd(&shared_hist[bin], weight)`
   directly) as a SECOND kernel, and time both versions with `GpuTimer` at increasing keypoint counts —
   at what count does the warp-shuffle version's advantage become measurable?
3. **Implement SURF's integral image.** `THEORY.md` teaches the prefix-sum formula; write
   `build_integral_image_kernel` (a parallel scan — see project 34.xx-adjacent scan patterns) and verify
   an arbitrary rectangle sum against a brute-force CPU sum.
4. **Add a second octave count and re-measure the gates.** Change `kNumOctaves`/`kIntervals` in
   `kernels.cuh`, regenerate expected numbers, and see how the scale-recovery/repeatability gates'
   margins change as the pyramid samples scale more finely.
5. **Replace the synthetic scene with a real photograph pair.** Capture two photos of the same textured
   object at different distances/angles, convert to the project's PGM format, and see whether real
   photographic texture (unlike this project's geometric checkerboards) lets the Lowe ratio test run at
   its classic 0.75 threshold instead of this project's measured, scene-specific 0.92.

## Limitations & honesty

- **SURF is documented-only**, not implemented, per the catalog's bundled-bullet scoping rule (CLAUDE.md
  §2): `THEORY.md` "Where this sits in the real world" teaches its integral-image and box-filter-Hessian
  ideas to implementable depth, and states explicitly why shipping it would double this project's kernel
  count to demonstrate the identical warp-reduction lesson on a different feature formula.
- **Reduced scale-space scope.** This project uses 2 octaves x 2 intervals (vs. Lowe's/production SIFT's
  typical 4+ octaves x 3 intervals over a 2x-upsampled image) — a smaller but algorithmically IDENTICAL
  pipeline, chosen for a legible keypoint count and fast build/run time on a 256x256 teaching image. See
  `kernels.cuh`'s header for the full scoping justification.
- **The single-scale-detector contrast is documented, not re-implemented.** The catalog brief asks
  (where cheap) for a "FAST-from-01.04-style single-scale repeatability" number for direct contrast with
  this project's scale-aware repeatability gate. Re-implementing a second detector inside this project
  (rather than importing 01.04's code, which would violate the self-containment rule) was judged NOT
  cheap given this project's scope, so the comparison is instead explained conceptually here and in
  THEORY.md: a single-scale detector's fixed pixel-radius corner test cannot follow a keypoint's apparent
  size across a real zoom, so its "matched" locations under a 1.5x scale change would drift or vanish
  entirely — exactly the gap `scale_repeatability`'s scale-band check exists to measure that a
  single-scale method has no analogous way to pass.
- **Matching-threshold tuning is scene-specific, and said so out loud.** `kLoweRatioSift=0.92` and
  `kMinL2DistSq=0.15` (see `THEORY.md` "How we verify correctness" for the full derivation) are
  MEASURED-then-margined for THIS project's synthetic checkerboard-and-disk content, which — a genuine,
  root-caused finding — turns out to be LESS discriminative for SIFT's descriptor than real photographic
  texture would be (a right-angle corner's shape is far more generic, once rotation/scale-normalized,
  than a natural photograph's texture). A real deployment would re-measure both against its own imagery.
- **Synthetic-only data, honestly labeled everywhere.** No public dataset applies (see
  `scripts/download_data.ps1`'s documented no-op) — every pixel is analytically rendered, and every
  ground-truth number (the transform, matched-pair labels) is exact by construction, not estimated.
- **Sim-validated only.** This project's output (matched keypoints, an implied relative pose) is never
  fed to real actuators in this repository, and nothing here is safety-certified (CLAUDE.md §1). A real
  deployment feeding a SLAM back-end's pose correction from this pipeline's output should validate on the
  hardware-testing ladder PRACTICE.md §3 describes before trusting it near real motion.
