# 01.14 — Template matching (NCC) at scale for pick verification

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Template matching (NCC) at scale for pick verification`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A robot picks parts into a 24-slot tray; a camera photographs the tray after the cycle; this project
answers, per slot, the question every pick-and-place cell must answer before it moves on: **is the
right part here, correctly placed?** It implements normalized cross-correlation (NCC) template
matching, searched over a small `+-8` px translation window and a 5-angle rotation set, and — the
catalog bullet's "at scale" — computes the full 24-slot x 15-template x 289-offset score volume
(104,040 evaluations) THREE different ways on the GPU to teach the classic redundant-work-elimination
acceleration ladder: a **naive** kernel that re-derives its own window statistics every time, an
**integer sum-table** kernel that looks them up in O(1) from a whole-tray integral image, and a
**shared-memory** kernel that additionally caches the search window and template on-chip. All three are
cross-checked against each other and against an independent CPU oracle; the verified scores then drive
a slot-by-slot **OK / WRONG_PART / EMPTY** verdict, an offset/rotation recovery, and 5 independent gates
that each teach one concrete lesson about NCC in practice.

Every piece named in the catalog bullet is implemented for real: NCC itself (derived and verified bit-
exact up to its final square root), the "at scale" batched multi-kernel acceleration ladder (measured,
not asserted), and "pick verification" as an actual classification/localization pipeline with a designed
synthetic tray whose ground truth exercises every corner case — a correctly offset part, a rotated part,
a wrong part, an empty slot, and a shadowed part (the NCC-vs-SSD illumination story). Nothing here is
stubbed or hand-waved; see [Limitations & honesty](#limitations--honesty) for the handful of honestly
simplified pieces (a bounded, not full-image, search window; a 2-D rotation set, not scale/perspective).

## What this computes & why the GPU helps

Zero-normalized cross-correlation (NCC): a similarity score in `[-1, +1]` between a template and an
image window that is invariant to any *global affine* brightness change (`w -> a*w + b`, `a>0`) — the
reason NCC, not raw pixel-difference matching, is the industry-standard similarity measure under real
lighting variation. This project's parallelization story has three layers:

- **The score volume is an embarrassingly parallel 3-D MAP.** Every one of the 104,040
  `(slot, template, offset)` evaluations is independent — no evaluation reads another's result — so the
  natural GPU mapping is one thread per evaluation, `grid.x=slot, grid.y=template, block=(dx,dy)`.
- **A prefix-SCAN acceleration structure.** The whole-tray integral image (sum and sum-of-squares) is
  built by a classic 2-pass separable prefix scan (row-parallel, then column-parallel), turning every
  later window-statistics query into an O(1) lookup instead of an O(T^2) re-scan.
- **A REDUCE inside every evaluation.** The direct correlation term (the numerator every NCC evaluation
  needs) is an O(T^2) sum over the template's pixels — unavoidable without an FFT reformulation
  (`THEORY.md` cites the crossover argument) — and the SHARED-MEMORY kernel's whole point is making that
  reduction read on-chip memory instead of global memory.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — a classical (non-learned) machine-vision check that closes the loop
  on a manipulation action, not an open-loop sensing stage. It sits AFTER an action (a pick or place),
  not before one, which is what distinguishes "pick verification" from generic object detection.
- **Upstream inputs:** a rectified, flat-fielded `Image` of the tray — project **01.01**'s GPU image
  pipeline (undistortion) and project **01.09**'s photometric vignetting/flat-field calibration, named
  explicitly because this project's own `illumination_robustness` gate exists precisely because
  upstream flat-fielding is never perfect in practice.
- **Downstream consumers:** the cell's PLC or robot controller's re-pick/retry logic and project
  domain **19** (manipulation & grasping) — specifically **19.01**'s antipodal grasp scoring, which a
  failed `WRONG_PART`/`EMPTY` verdict here would trigger a re-pick or fault into. A verified `OK` with a
  recovered offset can also feed a downstream pose-correction step, exactly like project 01.13's
  alignment output.
- **Rate / latency budget:** a pick-and-place cell's cycle time sets the budget — a 60 parts/minute cell
  gives 1 s/part end to end, of which vision verification is normally a small slice. This
  implementation's *measured* full 104,040-evaluation score volume (all three GPU variants, RTX 2080
  SUPER, Release) is **under a millisecond** (see `[time]` lines in `demo/expected_output.txt`'s
  companion run) — orders of magnitude inside that budget even before considering that a real
  industrial tray/camera would be larger than this project's teaching-sized 324x220 image.
- **Reference robot(s):** the **6-DoF manipulator work cell** (this project's primary reference robot —
  pick-and-place / bin-picking verification) and, secondarily, any **AMR** reference robot performing
  automated tote/kit staging.
- **In production:** OpenCV's `cv::matchTemplate(..., TM_CCOEFF_NORMED)` (the exact algorithm this
  project hand-rolls) for simple cases; commercial geometric/edge-based pattern matching (Cognex
  PatMax-class, Halcon's shape-based matching) for anything needing real rotation/scale robustness — see
  [Prior art](#prior-art--further-reading) and `THEORY.md` "Where this sits in the real world" for why.
- **Owning team:** machine vision / manufacturing engineering — see [`PRACTICE.md`](PRACTICE.md) §4 for
  the fuller org picture, continuing project 01.13's framing.

## The algorithm in brief

- **Zero-normalized cross-correlation (NCC)** — derived from the raw integer sums
  `S_w, S_t, S_ww, S_tt, S_wt` (window/template sum, sum-of-squares, and cross-product), and why
  correlation (not raw pixel difference) is the right similarity measure under brightness-affine change.
  See [`THEORY.md`](THEORY.md) "The math".
- **Integral images (2-pass separable prefix scan)** — a whole-tray running sum and sum-of-squares
  table, built once, that turns every window's mean/energy into an O(1) box query. See
  [`THEORY.md`](THEORY.md) "The GPU mapping" and "Numerical considerations" (the uint32-vs-uint64
  overflow analysis).
- **Three NCC kernels, one acceleration ladder** — naive (O(T^2) window-stat re-scan per evaluation),
  sum-table (O(1) window stats via the integral image), and shared-memory (sum-table's O(1) stats PLUS
  on-chip caching of the O(T^2) correlation loop's own reads). See [`THEORY.md`](THEORY.md) "The GPU
  mapping" for the measured speed-ups.
- **A 5-angle rotation set** — evaluating a template pre-rotated at `-6,-3,0,+3,+6` degrees and keeping
  the best score, the direct (and only, at this scope) answer to NCC's lack of rotation invariance. See
  [`THEORY.md`](THEORY.md) "The math" and "How we verify correctness" for the measured score-vs-angle
  falloff.
- **Slot classification + offset/rotation recovery** — host-side downstream analysis (not part of the
  GPU/CPU twin — see `src/reference_cpu.cpp`'s independence ruling) that turns the verified score volume
  into an OK/WRONG_PART/EMPTY verdict per slot, checked against known synthetic ground truth by 5
  independent gates.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/template-matching-at-scale-for-pick-verification.sln`](build/template-matching-at-scale-for-pick-verification.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/template-matching-at-scale-for-pick-verification.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none. This project links only the CUDA runtime and the C++17
standard library — every stage (integral-image scan, all three NCC kernels, classification) is
hand-rolled, on purpose (see [Prior art](#prior-art--further-reading) for what a production stack would
use instead).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is fully synthetic (CLAUDE.md §8's default) — a rendered 24-slot tray plus a
15-template golden reference set, with per-slot ground truth for exactly what part (if any) is really in
each slot, at what pose. `scripts/make_synthetic.py --seed 42` regenerates it byte-for-byte; no public
dataset applies here (see `data/README.md` for why a synthetic tray actually teaches *more* than a
photograph would for this specific task — the per-slot ground truth a photo lacks is exactly what the
classification/localization gates need). Full provenance, checksums, and per-file field documentation:
[`data/README.md`](data/README.md).

## Expected output

The demo prints 3 `VERIFY:` lines (whole-tray integral images and window statistics GPU vs CPU
**bit-exact** integer comparisons; the full NCC score volume, all 3 GPU kernels vs the CPU oracle, within
a float tolerance — **measured 0.0** worst-case on the committed scene, a real result, not a fudge
factor), followed by 5 independent `GATE <name>: PASS` lines (`variant_consistency`, `classification`,
`localization`, `rotation_lesson`, `illumination_robustness`) and a final `RESULT: PASS` only if every
verify and every gate holds. It writes 5 artifacts to `demo/out/`: a colored verdict overlay on the tray,
a score-map visualization for the rotated slot, the measured score-vs-rotation-angle curve, the full
24-row per-slot score table, and every gate's measured value/bound. The canonical stable lines live in
[`demo/expected_output.txt`](demo/expected_output.txt); `demo/README.md` explains every artifact and
every line prefix.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced data contracts: tray/slot/window geometry,
   the template/rotation-set layout, the score-volume and integral-image layouts, the NCC algebra
   (derived once, in comments), and every kernel/launcher/CPU-oracle declaration. Read this first.
2. [`src/kernels.cu`](src/kernels.cu) — the 8 GPU kernels. Start with `ncc_naive_kernel`, then
   `ncc_sumtable_kernel`, then `ncc_shared_kernel` to see the acceleration ladder step by step; the
   integral-image scan pair (`integral_row_scan_kernel`/`integral_col_scan_kernel`) is the
   scan-pattern case study.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins, including the
   single-pass 2-D recurrence integral image (a structurally different algorithm from the GPU's 2-pass
   separable scan, reaching the identical result — see its header).
4. [`src/main.cu`](src/main.cu) — orchestration: data loading, the verify stage, the host-only
   classification/localization/rotation/illumination analysis, all 5 gates, and artifact writing.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data-file/artifact-directory
   resolution.
6. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the tray/template renderer; read alongside
   `kernels.cuh` SECTIONS 1-2, whose geometry constants it deliberately duplicates.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **OpenCV** (`cv::matchTemplate` with `TM_CCOEFF_NORMED`, and `cv::integral` for the sum-table trick) —
  the industry-standard open-source implementation of exactly this algorithm; its GPU module
  (`cv::cuda::TemplateMatching`) is the production version of this project's three hand-rolled kernels.
- **Lewis (1995), "Fast Normalized Cross-Correlation"** — the classic reference deriving the
  sum-table-accelerated NCC algebra this project implements (the same `S_w, S_ww, S_wt` sums).
- **Cognex PatMax / VisionPro, MVTec Halcon shape-based matching** — the commercial geometric/edge-based
  pattern matchers a real production line uses instead of pixel-correlation NCC specifically because
  they are robust to rotation AND scale, not just a small pre-rotated angle set — see
  [`THEORY.md`](THEORY.md) "Where this sits in the real world" for exactly what they do differently.
- **Project 03.01 (FMCW radar cube + CFAR)** — the repo's own cuFFT precedent this project's THEORY.md
  cites for the FFT-domain correlation alternative and its crossover-size argument.
- **Project 01.13 (Canny + Hough for industrial alignment)** — this project's direct predecessor in
  domain 01's industrial-vision arc: shares the synthetic-scene-with-known-ground-truth philosophy, the
  bit-exactness-where-possible discipline, and the machined-part visual style.
- **Project 33.01 (batched small-matrix linalg)** — the production-scale pattern this project's
  per-template constant-stats table and per-slot integral-image box queries are a small, single-GPU-call
  instance of.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Plot the score map.** Load `demo/out/score_map_rotated_slot.pgm` and `score_vs_angle.csv` together
   and confirm by eye that the brightest point in the score map sits at the applied `(0,0)` offset, and
   that the angle curve's peak sits at the rotation-set template closest to the true 24-degree rotation.
2. **Full-image search.** This project's search is windowed to `+-8` px around each slot's KNOWN nominal
   center (README "Limitations"). Remove that assumption for one slot — search the entire tray for the
   best match of its expected template — and measure how much slower (and, with 24 near-identical parts
   on the tray, how much more AMBIGUOUS) unconstrained search is.
3. **A 4th rotation angle regime.** Regenerate the scene with the ROTATED cohort's angle swept from 0 to
   30 degrees in 2-degree steps (`scripts/make_synthetic.py`), rerun, and reproduce the full
   single-vs-rotation-set falloff curve this project's THEORY.md reports only 3 points of.
4. **FFT-domain correlation.** Implement the numerator's O(T^2) correlation sum via cuFFT-based
   convolution instead (THEORY.md "Where this sits in the real world" sketches the crossover argument)
   and measure the break-even template size on this GPU.
5. **A finer rotation set.** Increase `NUM_ROT` and tighten `ROTATION_DEG`'s spacing (`kernels.cuh`),
   rebuild, and measure how the `rotation_lesson` gate's recovered score changes as the set's angular
   resolution improves — and how much extra GPU time the larger score volume costs.

## Limitations & honesty

- **Windowed, not full-image, search.** The `+-8` px translation search is centered on each slot's
  KNOWN nominal location — the realistic pick-verification scoping (the robot commanded a specific
  slot), not a generic scene-wide template search. A production system searching an unconstrained scene
  needs either a much larger search window or a coarse-to-fine pyramid (see `THEORY.md`).
- **2-D rotation only, no scale or perspective.** The 5-angle rotation set handles small in-plane
  rotation; it does not model scale change (camera-to-part distance variation) or out-of-plane
  perspective (a tilted part) — real production systems use geometric/edge-based matchers (see
  [Prior art](#prior-art--further-reading)) precisely because they generalize across all of these.
- **The rotation test angle (24 degrees) was chosen to teach the lesson clearly, not to model a
  realistic robot placement tolerance.** MEASURED (`scripts/make_synthetic.py`'s ROTATED-cohort comment,
  `THEORY.md`): all 3 synthetic part shapes are NCC-robust to rotation errors under roughly 10-15
  degrees at this template size — a real, honest finding about compact, roughly-convex silhouettes, not
  a limitation of NCC per se. 24 degrees was the smallest angle that MEASURABLY separated single-template
  from rotation-set recovery on this scene with real margin; a robot with a genuinely large orientation
  tolerance would need either a denser rotation set or a rotation-invariant matcher.
- **The illumination gradient is deliberately close to affine.** The `shadow` cohort's dimming is a
  smooth, moderate gradient across one slot's window (`THEORY.md` "The math" derives NCC's *global*
  affine-brightness invariance) — a strong, sharply local shadow would erode NCC's robustness too; this
  project demonstrates the case NCC is actually built to handle, and says so.
- **Pixel units only — no camera calibration.** Like project 01.13, this project never converts pixels
  to millimeters; see `PRACTICE.md` §3 for how a real station adds that.
- **Synthetic scene, simplified texture/noise/illumination models.** Hashed per-pixel texture and
  uniform sensor noise stand in for real machined-surface texture and photon/read noise — see
  `THEORY.md` "The problem" for the physics this simplifies.
- **Not safety-certified; no motion is commanded here.** This project only computes a classification; if
  that classification ever triggered a real re-pick or reject action (as `PRACTICE.md` §3 describes),
  the same sim-validated-only caveat as every control/planning project in this repo applies
  (CLAUDE.md §1).
