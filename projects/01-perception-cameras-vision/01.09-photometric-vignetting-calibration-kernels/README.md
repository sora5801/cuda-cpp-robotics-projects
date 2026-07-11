# 01.09 — Photometric/vignetting calibration kernels

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Photometric/vignetting calibration kernels`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Before any photometric algorithm can trust a pixel value, the camera itself must be characterized: every
lens darkens its corners, every sensor's pixels respond to light slightly differently from their
neighbors, and every sensor adds a small fixed offset pattern even with the shutter closed. This project
builds the **factory/bring-up calibration** that measures and removes those spatial effects, from
first-principles CUDA kernels — the step that sits between "raw sensor bytes" and "a pixel value you can
compare across the frame."

The sensor model (single-sourced in `src/kernels.cuh`, derived from the physics in `THEORY.md`):

```
I(x,y) = g(x,y) * L(x,y) + o(x,y) + noise
g(x,y) = V(x,y) * PRNU(x,y)      <- multiplicative field (optical vignette x per-pixel gain)
o(x,y) = DSNU(x,y)               <- additive field (fixed-pattern offset / black level)
```

The demo runs the full calibration pipeline on the GPU: average 16 dark frames to recover the additive
field `o`, average 16 uniformly-illuminated flat frames and normalize to recover the *nonparametric*
multiplicative gain map `g` (the industrial flat-field standard), fit a compact *parametric* model
`V(r) = 1 + a2*r^2 + a4*r^4 + a6*r^6` to that map's radial falloff (a small host least-squares solve), and
finally apply `(I - o) / g` to a natural test scene — verified stage-by-stage against an independent CPU
reference, and graded against **six independent gates** tied to the exactly-known synthetic ground truth
(never routed through the pipeline being graded). All six pass on the reference machine — see
[Expected output](#expected-output) and `demo/gates_metrics.csv`.

**Composition with 01.08 (HDR exposure fusion), stated up front.** 01.08-hdr-exposure-fusion-tone-mapping
taught the camera's *per-pixel-independent* response curve — the CRF, one 1-D lookup table `g(z) =
ln(exposure)` identical at every pixel. This project teaches the complementary, *spatially-varying* terms
that sit **before** that curve in a real pipeline: the vignette and per-pixel gain/offset fields differ by
pixel, not by code value. A real radiometric pipeline composes both: flat-field correct first (this
project), THEN apply the CRF to convert corrected code values into linear radiance (01.08). Isolating the
two here — a deliberate simplification stated honestly — lets each project teach its own physics without
conflating "which pixel" effects with "which brightness" effects; see
[Limitations & honesty](#limitations--honesty).

## What this computes & why the GPU helps

Every stage is one of three GPU patterns (see `kernels.cu`'s file header for the full essay):

- **Map** — per-pixel, independent of every other pixel: dark-subtraction (`elementwise_sub_kernel`),
  center-normalization (`affine_kernel`), and the correction itself (`correction_kernel`, the "reason to
  exist" kernel: `(I - o) / max(g, floor)`). One thread per pixel saturates memory bandwidth.
- **Map-of-reductions** — `stack_mean_kernel`: n=19,200 *independent* per-pixel reductions running in
  parallel, each a short serial sum over the stack's 16 frames (too few frames per pixel to justify a
  shared-memory tree reduction *per pixel* — see THEORY.md "The GPU mapping" for the crossover argument).
- **Reduction / scatter-reduction** — `roi_mean_reduce_kernel` (a classic shared-memory tree reduction +
  one atomicAdd per block, MASKED to a rectangular ROI, pinning the gain map's scale) and
  `radial_bin_kernel` (a genuinely different pattern: a histogram scatter-reduce, where every one of
  19,200 threads atomicAdd's directly into ONE of 44 small global bins by its own pixel's radius — see
  `kernels.cu` for why this is contrasted deliberately against the first reduction's shared-memory staging).
- **Small batched-solve (host, not GPU)** — the parametric radial fit is a single 3x3 normal-equations
  solve over 44 binned points; THEORY.md "The GPU mapping" explains why a problem this tiny has no
  meaningful GPU parallelization (33.01-batched-small-matrix-linalg is where this repo teaches the
  GPU-batched version, at a problem size where batching many such solves actually pays for itself).

The GPU payoff here is not any single kernel's raw speed (these are 160x120 calibration frames — the CPU
oracle finishes the *entire seven-kernel pipeline* in well under a millisecond too, see
`demo/expected_output.txt`'s timing lines, a teaching artifact, never a benchmark claim). The payoff is
architectural, and it shows up on the correction side specifically: `correction_kernel` is the ONE stage
of this pipeline that must run **every frame**, at full camera rate, on a real robot — and a pure per-pixel
map is exactly the pattern that scales linearly with resolution and pays off hardest at 4K/60fps.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception, at the camera **front-end / ISP boundary** — even more upstream than
  01.08's radiometric linearization: a real pipeline applies THIS project's flat-field correction to raw
  sensor bytes first, then 01.08's CRF-based radiance recovery, then everything else (feature detection,
  stereo, SLAM front-ends) that assumes a spatially-uniform pixel value.
- **Upstream inputs:** raw sensor frames from the camera driver/ISP (01.01-full-gpu-image-pipeline by
  name — its debayer/rectify stages are what would run immediately before or alongside this one on a real
  Bayer sensor), plus the ONE-TIME factory/bring-up calibration frames (dark + flat stacks) this project's
  `data/sample/` mimics, message-shaped as repeated `Image` (01.01's row-major uint8 convention).
- **Downstream consumers:** every algorithm that assumes a photometrically well-behaved pixel. Two named
  explicitly: **photometric direct visual odometry/SLAM** (DSO-class — Engel, Koltun & Cremers' *Direct
  Sparse Odometry*, which explicitly requires a per-camera photometric calibration file of exactly this
  shape, vignette map + response function, before its direct intensity-matching residuals mean anything —
  see [Prior art](#prior-art--further-reading)); and **01.07-fisheye-omnidirectional-unwarping-and-multi**'s
  surround-view stitching *by name* — its blended seams between adjacent camera views are exactly where an
  UNcorrected vignette falloff becomes visible as a photometric mismatch (one camera's overlap region reads
  darker than its neighbor's), the seam artifact this project's correction removes at the source rather
  than papering over with seam-blending alone. General inspection/metrology cameras (reflectance or
  dimensional measurement) are a third consumer: an uncorrected vignette reads as a spatially-varying
  measurement bias across the field of view.
- **Rate / latency budget — two DIFFERENT budgets, stated honestly (THEORY.md "Where this sits in the
  real world" expands on both):** the CALIBRATION half of this project (dark/flat-stack averaging, the
  radial fit) runs **OFFLINE**, once per camera at manufacturing bring-up or after a lens swap/major
  temperature shift — seconds to minutes, dominated by capturing 32 frames, not by the sub-millisecond GPU
  compute measured in `demo/expected_output.txt`. The CORRECTION half (`correction_kernel`) runs
  **PER-FRAME**, at the camera's own live rate — 30-60 Hz for the manipulator work cell's vision camera,
  10-20 Hz for a LiDAR-class depth sensor's RGB counterpart on an AMR (`docs/SYSTEM_DESIGN.md` item 1) —
  and must fit comfortably inside that frame's few-millisecond perception budget; a single elementwise
  map over a few-megapixel frame is exactly the kind of GPU work that does.
- **Reference robot(s):** the **6-DoF manipulator work cell** (`docs/SYSTEM_DESIGN.md` §2.2 lists domain
  **01** explicitly for its wrist/frame-mounted 3D camera's vision stage — inspection and pose-estimation
  tasks are directly sensitive to a spatially-biased pixel value) and the **warehouse AMR** (§2.1's sensor
  suite includes 2-4 depth cameras alongside LiDAR; any camera-based obstacle or fiducial detection on that
  suite inherits the same dependency).
- **In production:** camera vendors ship a **factory NUC (non-uniformity correction) table** — exactly
  this project's `gain_recovered`/`dsnu_recovered` maps — baked into the sensor's own ISP or firmware,
  applied in hardware before a single byte reaches the host (see `PRACTICE.md` §3). DSO and similar
  direct-methods SLAM systems ship (or require the user to run) a standalone photometric calibration tool
  producing this exact file shape. Machine-vision inspection systems characterize their sensors against
  the **EMVA 1288** standard (`PRACTICE.md` §4), which formalizes PRNU/DSNU measurement procedures this
  project's dark/flat-stack pipeline is a teaching-scale version of.
- **Owning team:** perception / camera-systems engineering owns the algorithm and the recovered
  calibration file format; **manufacturing test engineering** owns running this calibration on every unit
  at EVT/DVT/PVT bring-up (`docs/SYSTEM_DESIGN.md` §5.2 explicitly lists "calibration pipelines (01, 02)"
  as an EVT/DVT/PVT-stage deliverable) — see `PRACTICE.md` §4.

## The algorithm in brief

- **Dark-stack calibration** (`stack_mean_kernel` on 16 dark frames) — recovers the additive field `o(x,y)`
  by averaging away read noise, `1/sqrt(N)` at a time. See [`THEORY.md`](THEORY.md#the-math).
- **Flat-stack calibration + center-normalize** (`stack_mean_kernel` + `elementwise_sub_kernel` +
  `roi_mean_reduce_kernel` + `affine_kernel`) — recovers the *nonparametric* multiplicative gain map
  `g(x,y)`, the industrial flat-field standard, pinned to ~1 near the image's geometric center. See
  [`THEORY.md`](THEORY.md#the-algorithm).
- **Radial binning + parametric least-squares fit** (`radial_bin_kernel` + the shared host
  `fit_vignette_radial_ls`) — bins the nonparametric map by distance from center, fits the compact model
  `V(r) = 1 + a2*r^2 + a4*r^4 + a6*r^6` (cites 33.01-batched-small-matrix-linalg for the general small-solve
  pattern). See [`THEORY.md`](THEORY.md#the-math).
- **Correction** (`correction_kernel`) — `(I - o) / max(g, floor)`, applied to a natural test scene
  rendered through the SAME ground-truth model. See [`THEORY.md`](THEORY.md#numerical-considerations).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/photometric-vignetting-calibration-kernels.sln`](build/photometric-vignetting-calibration-kernels.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/photometric-vignetting-calibration-kernels.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — this project links only the CUDA runtime
(`cudart_static.lib`) and the C++17 standard library, the repo's default budget (CLAUDE.md §5). The
parametric fit's 3x3 solve is small and one-time enough that hand-rolled host Gaussian elimination is both
simpler and the more honest teaching choice (`kernels.cu`'s `solve3x3`), the same call 01.08's
`crf_solve_debevec` makes for its own one-time calibration solve.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is **entirely synthetic** (CLAUDE.md §8 default), generated by
`scripts/make_synthetic.py` (fixed seed 42, deterministic, reproducible bit-for-bit): 16 dark frames, 16
uniformly-illuminated flat frames, and one natural test scene (a textured background plus five
identical-radiance "gray card" swatches — one centered, four near the corners), all rendered through a
KNOWN decentered cos^4 vignette x hashed PRNU/DSNU sensor model. The committed files also include the
exact ground-truth additive and multiplicative fields (raw float32 dumps) and every generation parameter
(`params.csv`). Full field-by-field documentation, checksums, and the regeneration command are in
[`data/README.md`](data/README.md).

## Expected output

The demo builds (if needed), loads the calibration stacks + test scene, runs the full pipeline on the GPU
**and** on an independent CPU oracle, and reports:

- **`VERIFY:`** — GPU matches the CPU reference within a documented per-stage tolerance, for all
  independently-implemented pipeline stages (stack mean, dark subtract, center normalize, radial bin,
  correction). See `src/main.cu`'s tolerance constants and `THEORY.md` "How we verify correctness" for why
  each bound is sized the way it is.
- **Six `GATE` lines**, each an independent check against ground truth (not a GPU-vs-CPU comparison):
  `dsnu_recovery`, `gain_recovery`, `radial_fit`, `noise_averaging`, `correction_efficacy`, `flatness` —
  see [The algorithm in brief](#the-algorithm-in-brief) and `THEORY.md` for what each one measures and why.
- **`ARTIFACT:`** — every artifact in `demo/out/`: `vignette_true.pgm` (the true cos^4 field alone),
  `gain_recovered.pgm` (the nonparametric estimate), `dsnu_recovered.pgm` (contrast-stretched — the true
  excursion is only +-2 code-value units), `scene_uncorrected.pgm`/`scene_corrected.pgm` (open these side
  by side — the corners visibly brighten toward uniform after correction), `radial_profile.csv` (true vs.
  nonparametric vs. fitted `V(r)`, for plotting), and `gates_metrics.csv` (every measured number,
  machine-readable).
- **`RESULT: PASS`** only when VERIFY and all six gates pass. The canonical stable lines (no timings, no
  device names — those vary machine to machine) live in [`demo/expected_output.txt`](demo/expected_output.txt).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced data-layout contract: the sensor model, the
   frame-major stack layout (and why), problem geometry, the center-normalization ROI, the radial-binning
   geometry, every kernel/launcher/reference signature, and the shared parametric-fit declaration.
2. [`src/kernels.cu`](src/kernels.cu) — the six reusable GPU kernels (SECTION 1: start with
   `stack_mean_kernel` for the map-of-reductions pattern, then `radial_bin_kernel` for the scatter-reduce
   histogram pattern — the two most instructive kernels here), their launch wrappers (SECTION 2), and the
   shared radial least-squares solver (SECTION 3, `solve3x3` + `fit_vignette_radial_ls`).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of every kernel above, plus
   the note on why the least-squares fit is deliberately *shared*, not duplicated.
4. [`src/main.cu`](src/main.cu) — orchestration: load the stacks + scene, run the GPU pipeline and the CPU
   oracle, VERIFY, call the shared fit ONCE, evaluate the six gates, write every artifact. Read the six
   gate blocks in order — each is a self-contained lesson in grading a calibration pipeline against ground
   truth that never routes through the pipeline itself.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s multi-candidate file
   resolution (this project reads 33 committed frames from `data/sample/` and writes 7 artifacts to
   `demo/out/`).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Kang, Chandrasekaran & Fossum, "Vignetting and Its Elimination," Sensors & Actuators / imaging
  literature; and the classic cos^4 falloff derivation** — the physics `vignette_v()`/THEORY.md "The
  problem" derive from solid angle + projected aperture.
- **EMVA 1288** — the machine-vision industry's formal standard for characterizing PRNU, DSNU, dark
  current, and sensor noise; this project's dark/flat-stack pipeline is a teaching-scale version of the
  measurement procedures it standardizes (`PRACTICE.md` §4).
- **Engel, Koltun & Cremers, "Direct Sparse Odometry," PAMI 2018 (DSO)** — the direct-VO system this
  project's output composes with by name (see [System context](#system-context--where-this-sits-in-a-robot));
  study its own `photometric_calibration` tool, which recovers exactly a vignette map + response function
  from a video sequence rather than a dedicated dark/flat rig.
- **Goldman, "Vignette and Exposure Calibration and Compensation," PAMI 2010** — a from-a-single-sequence
  (no calibration rig) alternative to this project's dedicated dark/flat-stack approach; a natural
  follow-up read once this project's rig-based method is understood.
- **OpenCV's `cv::fisheye`/lens-calibration tooling and camera vendor NUC utilities** — the production,
  battle-tested implementations of factory flat-field calibration; compare their API surface and file
  formats against this project's `gain_true.bin`/`dsnu_true.bin` convention.
- **01.08-hdr-exposure-fusion-tone-mapping-for-outdoor** (this repo) — the complementary per-pixel-
  INDEPENDENT radiometric story (the CRF); see [Overview](#overview) for the exact composition order.
- **33.01-batched-small-matrix-linalg** (this repo) — the GPU-batched version of the small dense solve
  this project's `fit_vignette_radial_ls` performs once, on the host.

## Exercises

1. **Plot `radial_profile.csv`.** Overlay the true, nonparametric, and fitted curves and see how tightly
   they track — then look at where they diverge (very small and very large radii, the least-densely-
   sampled bins).
2. **Fit the decentering too.** This project deliberately fixes the radial fit's center at the GEOMETRIC
   image center (see README "Limitations & honesty"). Extend `fit_vignette_radial_ls` (or add a small
   nonlinear refinement loop around it) to also estimate `cx, cy`, and measure whether `radial_fit`'s
   residual-ratio gate moves closer to 1.0.
3. **Vary `kGainFloor`.** Push a few pixels of `gain_true.bin` artificially close to zero (simulating dead
   pixels) in a copy of `scripts/make_synthetic.py`'s output, rerun, and watch `correction_kernel`'s
   division-by-small-gain hazard become visible in `scene_corrected.pgm`.
4. **Implement a per-block partial-histogram radial_bin.** `radial_bin_kernel` atomicAdd's directly into
   global memory (see its header for the honest contention discussion); stage per-block partial bins in
   shared memory first and measure the kernel-time change at a much larger resolution.
5. **Chain this project's output into 01.08.** Apply this project's `correction_kernel` to 01.08's four
   raw exposures BEFORE Debevec-Malik radiance merge, and see whether `01.08`'s `radiance_reconstruction`
   gate's error improves — the exact composition [System context](#system-context--where-this-sits-in-a-robot)
   describes.

## Limitations & honesty

- **Linear sensor model — the CRF is out of scope here (by design).** This project composes with, but does
  not reimplement, 01.08's camera response function; see [Overview](#overview) for why isolating the
  spatial terms is the deliberate scoping choice, not an oversight.
- **The parametric fit assumes the GEOMETRIC image center, not the true (decentered) optical center.** A
  real calibration pipeline does not know the lens's decentering in advance; this project's radial fit
  therefore carries a small, honestly-measured systematic bias from that mismatch (see
  `kernels.cuh`'s `kCenterRoi*` comment and Exercise 2 for the extension that removes it).
- **The `flatness` gate is a self-consistency check, not a held-out generalization test.** It applies the
  recovered correction to frame 0 of the SAME flat stack the calibration was computed from — informative
  (a broken correction would fail it too), but weaker evidence than `correction_efficacy`'s gate, which
  grades against the fully independent `scene.pgm`. See `src/main.cu`'s GATE 6 comment.
- **Sensor noise is a simplified read+shot Gaussian model** (THEORY.md "Numerical considerations" states
  the simplification explicitly), not a rigorous Poisson-photon simulator — adequate for teaching the
  `1/sqrt(N)` averaging law and forcing the pipeline to be noise-robust, without a second free parameter
  this project does not need.
- **Grayscale, single-sensor scope.** No color-filter-array PRNU/vignette cross-talk (a real Bayer sensor's
  R/G/B channels vignette very slightly differently) and no multi-camera cross-calibration — both real,
  standard extensions this project's scope deliberately excludes.
- **Not safety-certified; sim-validated only.** Nothing in this repository is safety-certified (CLAUDE.md
  §1). This project's output is a calibration file/corrected image, never a control signal, but the same
  caveat applies repo-wide: any use on real hardware is the owner's decision and responsibility.
