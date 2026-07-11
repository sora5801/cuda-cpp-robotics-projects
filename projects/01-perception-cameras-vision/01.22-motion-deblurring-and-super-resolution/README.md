# 01.22 — Motion deblurring and super-resolution for inspection zoom

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Motion deblurring and super-resolution for inspection zoom`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

An inspection robot's camera has to read a serial number, a crack, or a fine label pattern that its
sensor's native resolution cannot quite resolve — and it usually has to do this while the robot (or
its zoom stage) is moving, which blurs the frame further. This project studies the two classic
image-restoration problems that answer "how do you recover detail the sensor didn't cleanly
capture": **motion deblurring** (removing a KNOWN motion blur from one frame) and **multi-frame
super-resolution** (combining several low-resolution frames, each capturing a different sub-pixel
phase of the same scene, into one higher-resolution estimate). Both are implemented in full —
naive-inverse / Wiener / Richardson-Lucy deconvolution for milestone 1, shift-and-add + iterative
back-projection (IBP) vs. a bicubic baseline for milestone 2 — on one shared synthetic inspection
scene (a flat patch, a high-contrast edge, a row of dot-matrix glyphs, a hashed texture patch, and
three bar-chart frequency groups). The demo produces every restored image plus two convergence
curves and a machine-readable gate report; a bar-chart "money shot" crop shows, by eye, super-
resolution recovering a pattern that single-frame upscaling provably cannot.

## What this computes & why the GPU helps

**Milestone 1 (deblurring)** is a **map** in two different domains: a frequency-domain pointwise
divide (naive inverse / Wiener filter, one thread per complex FFT bin — cuFFT does the O(N log N)
transform itself) and a spatial-domain **stencil** (Richardson-Lucy's two dense 15x15 circular
convolutions per iteration, one thread per output pixel, the same mapping family as 01.11's
bilateral filter). **Milestone 2 (super-resolution)** contrasts a **scatter** (shift-and-add: many
low-res samples land on overlapping output cells, so every write is an atomic add) against a
**gather** (iterative back-projection's forward-simulate and back-project steps: every output pixel
reads a bounded, statically-known set of inputs, so no atomics are needed at all) — the same
geometric relationship inverted two different ways, a genuinely instructive GPU-mapping contrast.
None of these kernels is compute-heavy per pixel; the GPU wins here the way it wins on most image
kernels in this repository — by giving every pixel (or every complex bin, or every reference
group) its own thread and letting the hardware's massive parallelism absorb what would be a slow
nested loop on a CPU.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — an image-RESTORATION stage that sits between raw sensor capture
  and downstream recognition, not a sensing or estimation stage of its own.
- **Upstream inputs:** a calibrated, denoised `Image` from 01.01's GPU image pipeline (undistorted,
  photometrically corrected) and 01.11's low-light denoising; motion metadata (encoder/IMU-derived
  velocity and exposure timing, in the same spirit as 01.10's rolling-shutter correction) that turns
  into this project's known motion-blur PSF or the known sub-pixel frame shifts.
- **Downstream consumers:** an OCR engine or a defect/anomaly classifier (domain 12, ML & AI) that
  needs the restored, higher-fidelity `Image` to read a serial number or score a surface defect
  reliably; a human inspector's live-review UI, in near-line workflows.
- **Rate / latency budget:** this is an OFFLINE / NEAR-LINE processing stage, not a closed-loop
  30-60 Hz perception stage (`docs/SYSTEM_DESIGN.md` §1.1) — a few hundred iterative-solver
  iterations per frame (30 Richardson-Lucy iterations, 12 IBP iterations here) do not fit a
  real-time budget on typical inspection hardware; a queued/batch pipeline processing frames after
  capture (seconds, not milliseconds, per frame) is the realistic deployment shape (PRACTICE.md §3).
- **Reference robot(s):** an inspection crawler/AMR-mounted camera rig (the closest of the
  `docs/SYSTEM_DESIGN.md` §2 reference robots is the warehouse AMR — a fixed or pan-tilt inspection
  camera riding a mobile base is a direct variant of that reference design, generalized here to
  static/track-mounted inspection rigs too, PRACTICE.md §3).
- **In production:** a computational-photography ISP stage for multi-frame merge (phone cameras do
  a production version of milestone 2 every time you take a photo — THEORY.md "Where this sits in
  the real world"), plus, increasingly, a learned (CNN/transformer) restoration network trained end
  to end rather than hand-derived per-method — named honestly in README §11.
- **Owning team:** perception/inspection (a specialization within the Perception team,
  `docs/SYSTEM_DESIGN.md` §5.1) — often shared with the ML/data team once a learned restoration
  model replaces the classical filters this project teaches.

Motion blur exists here for an economic reason, not an academic one: 01.10 derives motion blur as
the camera's EXPOSURE INTEGRAL over its motion during a shot — an inspection robot that stops for
every frame is slow and expensive to operate; one that images WHILE moving is fast but blurs, which
is exactly the trade-off deblurring exists to soften (README "Prior art" and THEORY.md "The
problem" cite 01.10 by name for the physics).

## The algorithm in brief

Bullet list of the key algorithms this project implements; link to [`THEORY.md`](THEORY.md) for depth.

- **Naive inverse filtering** (frequency domain, cuFFT) — the textbook, UNREGULARIZED inverse of a
  known blur; deliberately shown exploding into noise at the PSF's spectral near-zeros
  ([`THEORY.md` § The math](THEORY.md#the-math)).
- **Wiener deconvolution** (frequency domain, cuFFT) — the MMSE-optimal regularized inverse filter,
  derived from first principles ([`THEORY.md` § The math](THEORY.md#the-math)).
- **Richardson-Lucy deconvolution** (spatial domain, iterative) — the multiplicative Poisson-MLE
  update, an alternative that never divides by a PSF spectral zero
  ([`THEORY.md` § The algorithm](THEORY.md#the-algorithm)).
- **PSF-mismatch honesty test** — Wiener deconvolution run with a deliberately wrong PSF angle,
  measuring non-blind deconvolution's sensitivity to an inaccurate motion estimate.
- **Shift-and-add** (atomic GPU scatter) — combine 8 known-sub-pixel-shifted low-res frames onto a
  2x grid by bilinear splatting ([`THEORY.md` § The GPU mapping](THEORY.md#the-gpu-mapping)).
- **Iterative back-projection (IBP)** (deterministic GPU gather) — refine the shift-and-add
  estimate by minimizing reprojection error against every low-res frame
  ([`THEORY.md` § The algorithm](THEORY.md#the-algorithm)).
- **Bicubic upscaling** — the single-frame baseline that cannot recover aliased detail, included as
  the point of comparison for the two methods above.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/motion-deblurring-and-super-resolution.sln`](build/motion-deblurring-and-super-resolution.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/motion-deblurring-and-super-resolution.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: this project links **cuFFT** (`cufft.lib`, DLL-backed at
runtime via `cufft64_*.dll`) in addition to the CUDA runtime — a CUDA-toolkit-default library
(CLAUDE.md §5), not an extra install beyond the toolkit every project already requires. No fallback
path is needed: cuFFT ships with every CUDA Toolkit 13.3 install.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

100% synthetic, generated by [`scripts/make_synthetic.py`](scripts/make_synthetic.py) (fixed seed
42, no external dataset — full provenance, checksums, and field documentation in
[`data/README.md`](data/README.md)). The committed sample (`data/sample/`, ~72 KB) is the shared
inspection scene (`truth.pgm`), its motion-blurred+noisy version (`blurred.pgm`) plus the exact and
a deliberately-wrong PSF (`psf_truth.csv` / `psf_mismatch.csv`), and 8 low-resolution frames at
known quarter-pixel shifts (`lr_frame_0.pgm`..`lr_frame_7.pgm`, `shifts_truth.csv`). Regenerate with
`python scripts/make_synthetic.py`.

## Expected output

The demo restores the blurred frame three ways (naive inverse / Wiener / Richardson-Lucy),
super-resolves the 8 low-res frames two ways (shift-and-add+IBP / bicubic baseline), verifies every
GPU result against an independent CPU twin (tolerances documented in `src/main.cu`, from ~0.0002 DN
for the well-conditioned Wiener filter up to a looser tolerance for the atomic-scatter shift-and-add
step — see `src/reference_cpu.cpp`'s header for the twin-independence ruling), and checks 6
independent, ground-truth-based gates. On the reference machine (RTX 2080 SUPER): blurred PSNR
14.99 dB -> Wiener 18.11 dB (+3.13 dB) and Richardson-Lucy 17.34 dB (+2.35 dB), while the naive
inverse filter DROPS to 10.71 dB (-4.28 dB, worse than doing nothing — the designed failure); the
PSF-mismatch run degrades Wiener from 18.11 dB to 11.07 dB (-7.04 dB) when the assumed motion angle
is off by 25 degrees. On the bar-chart frequency group below the low-res grid's Nyquist limit,
super-resolution's pattern CORRELATES with ground truth at 0.984 vs. bicubic's 0.215 (bicubic
reproduces a similarly-contrasted but wrong, aliased pattern — see `src/main.cu`'s
`bar_pattern_correlation()` comment); whole-image SR PSNR beats bicubic by +3.36 dB. The canonical
lines are in [`demo/expected_output.txt`](demo/expected_output.txt); every measured number above
appears in `demo/out/gates_metrics.csv` after a run.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: loads the shared scene + PSFs + low-res frames, runs
   both milestones' GPU pipelines, verifies each against its CPU twin, evaluates all 6 gates, writes
   every artifact.
2. [`src/kernels.cuh`](src/kernels.cuh) — the full data contract: geometry, PSF/blur parameters, SR
   shift-table shape, scene-layout rectangles, and every kernel/launcher declaration — read this
   before anything else in `src/`.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels themselves: start with `naive_inverse_kernel`
   / `wiener_kernel` (the frequency-domain pair), then `shift_and_add_kernel` vs.
   `forward_simulate_kernel`/`backproject_kernel` (the scatter-vs-gather contrast, the most
   interesting pair in this project).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle, including a
   from-scratch radix-2 FFT (the CPU FFT twin this project's task brief calls for).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact-directory
   resolution.
6. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — how the shared scene, the motion-blur
   PSF, and the 8 aliased low-res frames are generated (the "honest way to make aliased LR frames"
   the task brief asks for — see that script's header).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **OpenCV** (`cv2.filter2D`, `cv2.dft`, and the `photo` module's `deconvolution` sample) — the
  practical, well-tested implementations of exactly the Wiener/Richardson-Lucy filters this project
  derives from scratch.
- **scikit-image `restoration`** (`richardson_lucy`, `wiener`, `unsupervised_wiener`) — a clean,
  well-documented Python reference implementation to compare numbers against.
- **Dabov et al., "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering" (BM3D)**
  and its deblurring extensions — the state of the art in classical (non-learned) restoration,
  cited already in 01.11's kernels.cuh for the denoising half of this repo's restoration story —
  study it for how far hand-crafted priors can go before learned models take over.
- **Farsiu, Robinson, Elad, Milanfar, "Fast and Robust Multiframe Super Resolution" (2004)** — the
  paper this project's shift-and-add + IBP pipeline is a simplified teaching version of.
- **Irani & Peleg, "Improving Resolution by Image Registration" (1991)** — the original iterative
  back-projection paper; this project's `backproject_kernel` is a direct, simplified descendant.
- **DnCNN / SRCNN / EDSR / Real-ESRGAN (learned restoration, 2016-present)** — where production
  restoration has moved: a CNN/transformer trained end to end on paired blurred/sharp or
  low-res/high-res data usually beats hand-derived filters on real photographs, at the cost of
  needing training data and losing the interpretability this project's closed-form filters have.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. Change `kBlurAngleDeg`/`kBlurLengthPx` in `kernels.cuh`, regenerate the data, and rebuild — watch
   how the PSF's spectral zeros (and hence the naive inverse filter's explosion pattern) move.
2. Add a shared-memory TILED version of `convolve_circular_kernel` (Richardson-Lucy's stencil),
   following 01.11's bilateral-filter tiling lesson — measure the speed-up.
3. Extend the bar-chart scene with a second orientation (horizontal or diagonal stripes) and a
   matching `bar_pattern_correlation` gate — this project scoped to one orientation only (see
   "Limitations" below).
4. Implement a per-frequency (rather than constant) Wiener regularizer `K(f)` using a rough
   power-spectrum estimate of the scene — compare its PSNR against the constant-`K` version here.
5. Replace `forward_simulate_kernel`'s single bilinear sample with a small-area box-integration
   sensor model (closer to how `make_synthetic.py` actually generated the low-res frames) and
   measure whether IBP's reprojection error converges faster or to a lower floor.

## Limitations & honesty

What is simplified, what is synthetic, and what would differ in production.

- **Both restoration problems are NON-BLIND.** The motion-blur PSF and the frame-to-frame sub-pixel
  shifts are GIVEN (from the synthetic generator, standing in for motion metadata a real system
  would derive from encoders/IMU or explicit registration — PRACTICE.md §3). Blind deconvolution
  (estimating the PSF from the blurred image alone) and blind SR (estimating shifts from the frames
  themselves) are real, harder problems, documented here but not implemented.
- **The Wiener filter uses a single CONSTANT regularizer `K`**, not the frequency-dependent
  noise-to-signal power ratio a production Wiener filter estimates — a standard, named
  simplification (THEORY.md "Where this sits in the real world").
- **IBP's forward model is one bilinear sample per low-res pixel**, not the small-area
  box-integration a real sensor performs (and that `make_synthetic.py` itself uses to generate the
  low-res frames) — a stated mismatch between the forward model IBP assumes and the true image
  formation process (Exercise 5 above explores closing this gap).
- **The bar-chart test scene uses ONE stripe orientation** (vertical only) — horizontal/diagonal
  aliasing behaves differently and is left as an exercise, not implemented.
- **Both milestones assume a STATIC scene and a KNOWN, single-source-of-blur camera motion** — real
  inspection scenes can have independently moving content (a conveyor, a vibrating part) that this
  project's forward models do not account for.
- **Not safety-certified, not for real-hardware motion control.** This project only restores
  IMAGES; it never commands a robot. Still, per CLAUDE.md §1/§8: everything here is study material,
  validated only against synthetic ground truth, never validated against a certified imaging or
  metrology standard, and no output should be treated as calibrated measurement or legal/inspection
  evidence without independent verification (PRACTICE.md §4).
