# 01.15 — Background subtraction for fixed-workspace cells

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Background subtraction for fixed-workspace cells`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A camera bolted to a work cell never moves — so, statistically, "the background" is whatever a
pixel usually looks like, and "foreground" is whatever just changed. This project builds three
increasingly capable per-pixel background models — **frame differencing** (a fixed reference
frame), a **running single Gaussian** (an adaptive mean + variance per pixel), and **MOG-lite**
(a 3-mode Gaussian mixture per pixel, the Stauffer-Grimson idea, simplified for teaching) — and
runs all three, on both GPU and an independent CPU oracle, against one 160-frame synthetic
sequence built from **five deliberately designed events**: an arm sweeping through twice, a box
permanently placed on the bench, a slow +15% illumination ramp, and a status lamp blinking between
two legitimate brightness states. Every event has exact, closed-form ground truth, so the demo
does not just check "does the GPU match the CPU" — it checks **which of the three models gets the
right answer, and why**, with five independent gates (README "Expected output") including a
closed-form prediction of how many frames a background model needs to "forget" a newly placed
object. All three models named in the catalog bullet's implied comparison (naive vs. adaptive vs.
multi-modal) are fully implemented; nothing here is a documented-only stub.

## What this computes & why the GPU helps

Three per-pixel classifiers, each independent across pixels within a frame:

- **Frame differencing:** `|I(t) - reference| > threshold` — a pure **map**, stateless, one
  comparison per pixel, computed for the WHOLE 160-frame sequence in a single kernel launch (no
  frame-to-frame dependency at all).
- **Running single Gaussian:** classify against `(mean, variance)`, then update both by
  exponential moving average — a per-FRAME **map** (one thread per pixel), but the sequence of 160
  frames is a genuine **serial recurrence**: frame `t`'s state depends on frame `t-1`'s, so the GPU
  mapping is "one kernel launch per frame," not "one thread per (frame, pixel)."
- **MOG-lite (K=3):** classify against the nearest of three per-pixel Gaussian modes, update
  weights/means/variances, rank by confidence, and decide background vs. foreground from the
  ranked cumulative weight — the same per-frame **map** + serial-recurrence shape as the single
  Gaussian, with a small **K=3 sort** inside every thread (see "The GPU mapping" in `THEORY.md` for
  the warp-divergence story this creates).

All three, plus the shared 3x3 **morphological open** (erode+dilate, a **stencil**) that cleans
every raw mask, are embarrassingly parallel over pixels — 12,288 independent decisions per frame,
naturally one GPU thread each. The GPU's job here is not to hide a slow algorithm; it is to keep
160 frames x 3 models x 12,288 pixels comfortably inside a real camera's 33 ms/frame budget with
room to spare (measured: **~4-10 ms total** for the whole 325-kernel-launch pipeline on an RTX 2080
SUPER — see `[time]` in `demo/expected_output.txt`'s companion run).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — a classical (non-learned) machine-vision layer that watches a
  **fixed** camera's field of view and turns "what changed" into a small, structured event stream.
  It is a cross-cutting WATCHER, not a stage in the perception→planning→control chain: it runs
  continuously, independent of any single manipulation cycle, the same architectural role
  SYSTEM_DESIGN.md gives 21.04's speed-and-separation monitor (a *different* fixed-camera watcher,
  covered below).
- **Upstream inputs:** a rectified `Image` from a **stationary** camera — project **01.01**'s GPU
  image pipeline (undistortion) — ideally also flat-fielded (project **01.09**'s photometric
  vignetting calibration, which would shrink exactly the illumination-drift failure mode this
  project's `illumination_drift` gate demonstrates) and, in poor lighting, denoised (project
  **01.11**'s low-light denoising, which would lower this project's `NOISE_SIGMA` operating point).
  Camera geometry must be genuinely fixed — the very first assumption THEORY.md's "The problem"
  section makes explicit.
- **Downstream consumers:** two different kinds of alarms, both message-shaped as a small event
  record (foreground mask + region ID + timestamp, SYSTEM_DESIGN.md §3.6-style): (1) an
  **intrusion alarm** feeding the cell's supervisory logic when the E1/E5-style events fire — the
  same didactic role project **21.04**'s speed-and-separation monitor plays for HUMAN intrusions
  specifically (see below, with its caveat carried over verbatim), and (2) a **pick-verify
  trigger** — "did the bin's contents change since the last cycle?" — that would hand off to
  project **01.14**'s template-matching pick-verification for the actual part-identity check, this
  project only ever answering "something changed," never "what changed to."
- **Rate / latency budget:** matches the fixed camera's own frame rate — this project targets a
  **30 Hz** budget (33 ms/frame; SYSTEM_DESIGN.md item 1's camera-perception boundary), against
  which the measured full-sequence pipeline (three models, GPU+CPU, morphology, all five gates) at
  **~4-10 ms total for 160 frames** is a two-orders-of-magnitude margin — the realistic per-frame
  incremental cost (one `sg_step` + one `mog_step` launch) is a small fraction of a millisecond,
  comfortably inside budget even stacked with other perception work on the same GPU.
- **Reference robot(s):** the **6-DoF manipulator work cell** (its own bullet in
  SYSTEM_DESIGN.md §2.2, watching the bench for an unexpected intrusion or a completed pick/place)
  and the **warehouse AMR** (SYSTEM_DESIGN.md §2.1, watching a depot's fixed induction/staging
  station the same way).
- **In production:** OpenCV's `cv::BackgroundSubtractorMOG2` / `KNN` (the direct descendants of the
  algorithm this project teaches), or ViBe-style sample-consensus models for higher robustness —
  see [Prior art](#prior-art--further-reading). Certified safety functions that must react to a
  human intrusion do **not** use background subtraction at all — see the caveat below and
  `PRACTICE.md` §4.
- **Owning team:** perception/cell-controls engineering (workspace-monitoring and pick-verification
  triggers) — see [`PRACTICE.md`](PRACTICE.md) §4 for the fuller org picture, continuing project
  01.13/01.14's framing for this same 01.x sub-domain.

> **Not a safety device.** This project's output could plausibly gate real motion (an "intrusion
> detected, pause the cell" signal) — CLAUDE.md §1/§8 requires the caveat explicitly here: this is a
> **sim-validated, didactic** intrusion signal, never a certified protective function. Project
> **21.04** (speed-and-separation monitoring) is this repository's dedicated treatment of
> human-safety monitoring, and it carries the identical caveat: *"DIDACTIC IMPLEMENTATION — NOT A
> CERTIFIED SAFETY FUNCTION."* Real collaborative-safety systems use certified hardware (safety-rated
> laser scanners, light curtains, pressure-sensitive mats) per ISO/TS 15066 and ISO 13849 — see
> `PRACTICE.md` §4 and SYSTEM_DESIGN.md item 6's regulatory map.

## The algorithm in brief

- **Frame differencing** — a static reference frame, stateless thresholding. See `THEORY.md`
  "The algorithm" and kernels.cu's `frame_diff_kernel`.
- **Running single Gaussian (EMA background model)** — per-pixel `(mean, variance)`, updated by
  exponential moving average every frame ("blind" update), classified by a `k*sigma` test. See
  `THEORY.md` "The math" for the EMA time-constant derivation and the closed-form absorption-time
  formula the `absorption` gate checks against, and `kernels.cu`'s `sg_step_kernel`.
  - Includes a **variance ceiling** (`SG_VAR_CEIL`) discovered empirically while building this
    project — see `THEORY.md` "Numerical considerations" for the runaway-desensitization bug it
    fixes and why it makes the closed-form prediction possible in the first place.
- **MOG-lite (K=3 Gaussian mixture, Stauffer & Grimson 1999, simplified)** — three
  `(weight, mean, variance)` modes per pixel; match-to-nearest within `2.5 sigma`; weight EMA
  update; matched mode's mean/variance EMA update; replace-weakest-mode on no match; background =
  the smallest, highest-confidence (`weight/sigma`) set of modes whose cumulative weight reaches
  `0.8`. This project's didactic heart — see `THEORY.md` "The math" and "The algorithm" for the
  full step-by-step derivation, and `kernels.cu`'s `mog_step_kernel` for the GPU mapping
  (including a worked warp-divergence estimate).
- **3x3 morphological open (erode, then dilate)** — post-processing shared by all three models'
  masks, removing salt-and-pepper misclassifications before any gate reads a mask. Same
  8-connected, zero-padded convention project **30.01**'s fruit-mask cleanup stage uses — see
  `THEORY.md` "The algorithm" for why opening (not closing) is the right operator here.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/background-subtraction-for-fixed-workspace-cells.sln`](build/background-subtraction-for-fixed-workspace-cells.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/background-subtraction-for-fixed-workspace-cells.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none.** This project links only the CUDA runtime and
the C++17 standard library — three hand-rolled background models and one hand-rolled morphological
filter need no external library (CLAUDE.md §5 default budget).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

160 committed grayscale PGM frames (128x96, ~1.88 MiB total) — the **entire** designed sequence,
so the demo runs fully offline with zero downloads and zero Python at run time. Synthetic, generated
by `scripts/make_synthetic.py` (fixed seed 42; hand-rolled xorshift32 for sensor noise, a stateless
integer hash for the fixed spatial texture — see that script's header for why the two are kept
conceptually separate). Full field documentation, the exact size-decision math (why 128x96 and not
the catalog-illustrative 240x180), and representative checksums live in
[`data/README.md`](data/README.md).

## Expected output

The canonical run (see [`demo/expected_output.txt`](demo/expected_output.txt) for the exact stable
lines) reports, on the reference machine (RTX 2080 SUPER, sm_75, Release):

- **`VERIFY(<model>): PASS`** for all three models — 0 / 1,966,080 raw-mask elements differ between
  GPU and CPU for frame differencing (bit-exact, no accumulation); 0 differ for single-Gaussian and
  MOG-lite too, with final-state `max|gpu-cpu|` of 0.000008 (variance/weight fields) — see
  `THEORY.md` "How we verify correctness" for why a small nonzero tolerance is the honest choice
  even though this particular run is bit-exact.
- **`GATE intrusion_detection: PASS`** — mean IoU vs. the arm's exact ground-truth footprint over
  every E1+E5 frame: single-Gaussian 0.6075, MOG-lite 0.5072 (frame-diff 0.6221, reported only —
  it is not gated here because it can legitimately detect a real intruder; its designed failure is
  drift, not intrusion).
- **`GATE illumination_drift: PASS`** — late-sequence (post-drift) false-positive rate:
  single-Gaussian 0.0172, MOG-lite 0.0000 (both comfortably under the 0.03 ceiling), while
  **frame-diff hits 0.3151** — over 10x the 0.10 floor this gate asserts it must exceed. This is
  the designed failure, asserted, not just observed.
- **`GATE absorption: PASS`** — the single-Gaussian model's closed-form absorption-time prediction
  (`THEORY.md` "The math") is **19 frames**; the measured frames-until-absorbed on the actual box
  event is **18 frames** — a 0.947x ratio, comfortably inside the documented [0.5x, 2.0x] gate. MOG
  reaches its own steady state in 2 frames (reported only — no closed form for MOG's mode-swap
  dynamics; see `THEORY.md`).
- **`GATE bimodal_lesson: PASS`** — false-positive rate AT THE BLINKING LAMP: MOG-lite 0.0000
  (learns both brightness states as separate modes), single-Gaussian **1.0000** — it flags the lamp
  as foreground on every single frame in the measured window, because one Gaussian cannot
  represent two legitimately different brightnesses at the same pixel. The dramatic gap (0.0 vs.
  1.0) is the whole reason this project builds a mixture model at all.
- **`GATE noise_floor: PASS`** — early, event-free, drift-negligible false-positive rate:
  single-Gaussian 0.0000, MOG-lite 0.0004, both far under the 0.02 ceiling.

Every gate's exact tolerance, and the measurement it was derived from, is documented as a comment
directly above its `constexpr` in [`src/main.cu`](src/main.cu) — never invented before the fact
(CLAUDE.md §12).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced contract: image/sequence geometry, the
   five designed events' exact numeric schedule, every model's threshold/rate constants, and the
   MOG state's mode-major memory layout. Read this FIRST — every other file defers to it.
2. [`src/kernels.cu`](src/kernels.cu) — the four GPU kernels (frame-diff, single-Gaussian step,
   MOG-lite step, morphological erode/dilate) and their launch wrappers. `mog_step_kernel` is the
   most interesting kernel in the project — start there for the warp-divergence story.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of every model
   (deliberately coded differently from the kernels where the algorithm allows — see this file's
   own header for the "written twice, on purpose" ruling).
4. [`src/main.cu`](src/main.cu) — orchestration: load the sequence, run CPU+GPU, verify, then the
   five gate functions (each documents its own derivation) and the artifact writers.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s multi-candidate
   `data/sample/` and `demo/out/` resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Stauffer & Grimson, "Adaptive background mixture models for real-time tracking" (1999)** — the
  paper this project's MOG-lite model directly simplifies; read it for the full (non-"lite")
  update-rate derivation (`rho` proportional to the matched Gaussian's density, not a fixed
  constant) and the original K=3..5 recommendation.
- **OpenCV `cv::BackgroundSubtractorMOG2`** — the mixture-of-Gaussians model's production
  descendant (Zivkovic 2004/2006): adaptive per-pixel component count, shadow detection, and a
  learning-rate schedule this project's fixed `MOG_LR_PARAM`/`MOG_LR_WEIGHT` only approximates.
- **OpenCV `cv::BackgroundSubtractorKNN`** — a non-parametric (K-nearest-neighbor sample
  consensus) alternative to a Gaussian mixture; learn it for how it sidesteps the Gaussian
  assumption entirely.
- **Barnich & Van Droogenbroeck, "ViBe: A universal background subtraction algorithm for video
  sequences" (2011)** — a sample-based (not parametric) model with an unusually clever
  neighbor-diffusion update rule; a different, influential answer to the same problem.
- **Learned change detection (e.g., "FgSegNet", "BSUV-Net")** — modern CNN-based approaches that
  outperform classical models on hard benchmarks (dynamic backgrounds, camouflage) at the cost of
  needing labeled training data this project's fully-synthetic approach never requires.
- **Project 30.01 (agriculture)** — the morphological-opening precedent this project's
  post-processing stage follows rather than reinvents.

## Exercises

1. **Widen the morphology.** Iterate the erode/dilate pass twice instead of once (a stronger
   opening) and re-measure `intrusion_detection`'s mean IoU — does it improve, or does the arm's
   thin "forearm" link start disappearing?
2. **Tune MOG's mode count.** `MOG_K` is fixed at 3 in `kernels.cuh`. Extend the kernels to K=4 or
   K=5 (the register arrays are currently hardcoded to 3 — see `mog_step_kernel`'s doc-comment) and
   see whether the `bimodal_lesson` gate's margin changes.
3. **Add a "conservative" update.** This project's single-Gaussian model updates blindly, every
   frame, regardless of classification (`THEORY.md` "Where this sits in the real world" explains
   why). Add a variant that skips the update on detected-foreground pixels and compare its
   `absorption` behavior — does the box ever get absorbed at all?
4. **Use BOTH Box-Muller outputs.** `scripts/make_synthetic.py`'s `gaussian_pair()` discards its
   second sample; use it (alternate frames, or interleave x/y) and confirm the committed sample's
   checksums change but every gate still passes.
5. **Try a fourth event.** Add a second, dimmer lamp with a different blink period and see whether
   MOG-lite's fixed K=3 modes are still enough for a pixel that must now track THREE recurring
   appearances (bench, lamp-high, lamp-low) simultaneously with a genuine intrusion passing through.

## Limitations & honesty

- **The arm does not rotate.** "Two-link articulated chain" here means two rectangles that
  translate together, not a revolute joint — a documented simplification (`kernels.cuh` SECTION 2)
  that keeps the ground-truth mask an exact, closed-form rectangle union instead of a rasterized
  rotated polygon, without weakening the intrusion-detection lesson.
- **No radiometric calibration.** Pixel intensities are unitless synthetic camera counts in
  `[0, 255]`; there is no exposure/gain model beyond the designed illumination ramp (E3) — a real
  camera's auto-exposure would fight this project's models in ways `THEORY.md` "The problem"
  discusses but does not simulate.
- **Sensor noise is a simplified constant-sigma Gaussian**, not true shot noise (which scales with
  signal) — an honest simplification representing a read-noise-dominated regime; see `THEORY.md`
  "The problem."
- **Single-Gaussian's update is "blind"** (updates every frame regardless of classification), a
  simplification that makes the absorption-time closed form exact for this project but differs
  from production "conservative update" designs — see `THEORY.md` "Where this sits in the real
  world" and Exercise 3.
- **Not a safety device.** See the "Not a safety device" callout in System context above and
  `PRACTICE.md` §4 — this project's intrusion signal is sim-validated and didactic only, never a
  certified protective function, regardless of how compelling the demo's numbers look.
