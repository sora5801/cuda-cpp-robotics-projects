# 01.24 — Transparent/reflective object detection via polarization imaging

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Transparent/reflective object detection via polarization imaging`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Every optical depth sensor a robot owns — stereo, structured light, time-of-flight — assumes light
bounces once, diffusely, off an opaque surface. Glass and bare metal break that assumption: glass
mostly transmits (a stereo pair sees straight through it, or sees a random specular glint), and metal
mirror-reflects (it shows you whatever is behind the camera, not itself). Both are close to invisible
in a plain intensity image too, when the scene behind them happens to be a similar brightness. This
project builds a **division-of-focal-plane (DoFP) polarization camera pipeline** that finds them anyway,
using a physical fact intensity throws away: **specular reflection off glass or metal polarizes light,
diffuse reflection off matte surfaces mostly does not** (the Fresnel equations, derived in
[`THEORY.md`](THEORY.md)). The demo renders a synthetic scene — a matte background plus a flat glass
pane, a curved glass dome, and a brushed metal bar — reconstructs the per-pixel Stokes parameters from
the camera's four polarizer-angle measurements, computes the Degree and Angle of Linear Polarization
(DoLP/AoLP), and detects the objects from DoLP alone. The centerpiece measurement (`GATE detection` in
`demo/expected_output.txt`) runs the **identical** detection pipeline on plain intensity too and shows
it fails on the glass — proving, not just asserting, why this modality earns its place on a robot.

All five catalog-bundle components are implemented in full (no reduced-scope milestones): the DoFP
mosaic model, per-angle demosaic, Stokes/DoLP/AoLP estimation, the free Malus self-consistency check,
and dual-signal (DoLP vs. intensity) detection.

## What this computes & why the GPU helps

The whole pipeline (demosaic, Stokes, DoLP/AoLP, the Malus residual, thresholding) is a **MAP**: every
output pixel is an independent, closed-form function of a small, fixed neighborhood of inputs — no
pixel's result depends on another pixel's *result* (only on other pixels' *inputs*), so one GPU thread
per pixel is the natural mapping and the whole chain launches embarrassingly parallel across ~16,000
pixels. The one exception is connected-component labeling, a fixed-point **iterative stencil** (each
sweep is still a map; convergence needs several sweeps) — the same GPU pattern 01.06's fiducial decoder
and 01.21's scene-flow detector use, cited by name in [`kernels.cu`](src/kernels.cu).

- **Demosaic** (Stage 1): map, one thread reconstructs all 4 polarizer-angle channels per pixel.
- **Stokes / DoLP / AoLP / Malus residual** (Stages 2-4): maps, one thread per pixel each.
- **Detection** (Stage 5): threshold (map) -> morphological open (3x3 stencil) -> connected-component
  labeling (iterative stencil with atomics) -> size filter (map) — run **twice**, once per signal.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** SENSORS -> **PERCEPTION** (SYSTEM_DESIGN.md §1). This is a perception-layer
  detector: it turns a raw sensor capture into an object mask, exactly parallel to 01.06's fiducial
  detector or 30.01's crop detector, just fed by a different modality.
- **Upstream inputs:** a DoFP polarization camera's raw mosaic frame — the message-shaped equivalent of
  a `PolarizationImage` (4x the bit depth of a plain `Image`, one super-pixel phase tag per pixel).
- **Downstream consumers:** a `TransparentObjectMask` (or fused into a standard `Image`-shaped
  obstacle/segmentation mask) feeding costmap construction (**14.02** GPU traversability costmaps) for
  navigation around glass storefronts/railings, or a grasp-candidate filter (**19.01** antipodal grasp
  scoring) that must know a "transparent" region is real geometry, not a sensor dropout.
- **Rate / latency budget:** camera-perception boundary, **30-60 Hz**, **< 1 frame (16-33 ms) end-to-end**
  (SYSTEM_DESIGN.md §1.1) — this project's whole pipeline runs in a fraction of a millisecond on a
  128x128 test frame (see "Expected output"), leaving ample headroom for a full-resolution version.
- **Reference robot(s):** the **warehouse AMR** (glass storefronts, railings, doors — SYSTEM_DESIGN.md
  §2.1) and the **6-DoF manipulator work cell** (bin picking of glass/metal parts — §2.2), which is
  exactly why `PRACTICE.md` grounds this project in both.
- **In production:** a dedicated DoFP sensor (this project's PRACTICE.md §2 names an illustrative real
  part) bolted onto — or replacing a frame of — the existing perception camera, feeding a
  polarization-aware segmentation head or a classical DoLP-threshold detector like this one at higher
  resolution, fused with the depth-based obstacle stack rather than replacing it (glass still needs a
  depth estimate from SOME modality; polarization tells you *that* something is there and roughly
  *what material*, not directly *how far*).
- **Owning team:** perception (with a research/algorithms specialization in "exotic sensing
  modalities") — SYSTEM_DESIGN.md §5.1's org map.

## The algorithm in brief

- **DoFP mosaic model** — a 2x2 repeating super-pixel of linear polarizers at 0/45/90/135 degrees (the
  polarization analogue of a Bayer color filter array). [`THEORY.md` "The problem"](THEORY.md#the-problem--physics--engineering-first).
- **Per-angle bilinear demosaic** — reconstruct all 4 angle channels at every pixel from the
  spatially-multiplexed mosaic (01.23's Bayer-demosaic kinship, generalized from 3 phases to 4).
  [`THEORY.md` "The GPU mapping"](THEORY.md#the-gpu-mapping).
- **Stokes parameter estimation** — the (unweighted) least-squares solution of Malus's law sampled at 4
  angles. [`THEORY.md` "The math"](THEORY.md#the-math).
- **DoLP / AoLP** — degree and angle of linear polarization, including the half-angle wrap.
  [`THEORY.md` "The math"](THEORY.md#the-math).
- **Malus self-consistency residual** — a FREE 1-DOF invariant (4 measurements, 3 parameters) that
  catches demosaic/registration bugs with no ground truth at all. [`THEORY.md` "How we verify correctness"](THEORY.md#how-we-verify-correctness).
- **Detection: threshold + morphological open + connected-component labeling + size filter** — run
  twice (DoLP signal, intensity-contrast signal), citing **01.06**/**01.21**'s connected-component
  pattern and **01.21**'s size-filtering lesson. [`THEORY.md` "The algorithm"](THEORY.md#the-algorithm).
- **Fresnel physics closed form** — the independent analytic prediction the `fresnel_anchor` gate checks
  the pipeline's own measurement against. [`THEORY.md` "The math"](THEORY.md#the-math).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/transparent-reflective-object-detection.sln`](build/transparent-reflective-object-detection.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/transparent-reflective-object-detection.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only
(no cuFFT/cuBLAS/Thrust; every stage here is a map or a small stencil, hand-rolled per CLAUDE.md §5).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

100% synthetic, generated by `scripts/make_synthetic.py` — a physics forward model that computes true
per-pixel Stokes parameters from the real Fresnel equations (glass) and a documented phenomenological
curve (metal), renders them through Malus's law into the four polarizer-angle channels, adds sensor
noise, and mosaics down to what a real DoFP sensor would actually capture. Full field documentation,
checksums, and regeneration instructions in [`data/README.md`](data/README.md).

## Expected output

Every stage is checked GPU-vs-CPU (`VERIFY(...)` lines) against tight float tolerances, and six
independent physics/detection GATEs check the pipeline's *output* against ground truth or closed-form
physics — never routed through either implementation. Measured on the reference machine (RTX 2080
SUPER, sm_75, Release|x64, default args):

| Check | Measured | Tolerance/floor |
|-------|----------|------------------|
| VERIFY(demosaic/stokes/dolp/aolp/malus_residual) | max\|gpu-cpu\| = 0.00000 (all five) | 0.01 DN / 0.001 / ~0.06 deg |
| VERIFY(detection_dolp / detection_intensity) | 0/16,384 mismatched pixels (both) | bit-exact |
| `GATE stokes_accuracy` | DoLP MAE (interior) = 0.0141; AoLP circular MAE (high-DoLP interior) = 0.82 deg | <= 0.05; <= 5.0 deg |
| `GATE malus_consistency` | mean\|residual\| = 3.78 DN | <= 6.0 DN |
| `GATE fresnel_anchor` | measured pane DoLP = 0.5343 vs. closed-form 0.5347 (\|diff\|=0.00034) | <= 0.02 |
| `GATE detection` | DoLP recall: glass=97.0%, metal=82.0%; **intensity recall on glass = 0.0%** | glass>=85%, metal>=65%, intensity-glass<=5% |
| `GATE brewster_sweep` | closed-form peak at 56 deg (true Brewster = 56.31 deg) | \|diff\| <= 2.0 deg |
| `GATE negative_control` | matte-only scene: mean DoLP=0.028, detected pixels=0 | exactly 0 |

The `detection` gate's two recall numbers are the whole point stated as data: **DoLP-based detection
finds 97% of the glass; plain-intensity detection finds 0%** of the exact same glass, because it was
built (by construction) to have zero intensity contrast against its background. The GPU pipeline runs
in well under a millisecond on this 128x128 test frame (see `[time]` lines; a teaching artifact, not a
benchmark). Six artifacts land in `demo/out/` — see [`demo/README.md`](demo/README.md) for what each
shows.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the data contract: the DoFP mosaic layout, scene-object
   geometry, detection constants, and the shared Fresnel physics helper. Read this first — every other
   file assumes it.
2. [`src/kernels.cu`](src/kernels.cu) — the 12 GPU kernels: demosaic, Stokes, DoLP/AoLP, Malus
   residual, threshold, morphological open, connected-component labeling, size filter.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of every stage (a
   genuinely different algorithm for connected-component labeling — union-find vs. the GPU's
   label-propagation sweep).
4. [`src/main.cu`](src/main.cu) — orchestration: load data, run both pipelines, VERIFY, run the six
   GATEs, write artifacts.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the physics forward model. Read this
   alongside `THEORY.md` "The problem" to see the Fresnel equations rendered into pixels.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `find_data_file`/`resolve_out_dir`.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **OpenCV's `cv::ppf` / polarization module** — production-grade DoFP demosaic and Stokes/DoLP/AoLP
  computation; compare its edge-aware demosaic against this project's plain bilinear baseline.
- **Sony's IMX250MZR/IMX253MZR datasheets** — the real sensor family this project's mosaic model is
  illustrative of; read the actual per-pixel polarizer extinction-ratio specs (PRACTICE.md §2).
- **Wolff & Boult (1991), "Constraining object features using a polarization reflectance model"** — the
  classical paper connecting Fresnel-derived DoLP to material/shape recognition, the intellectual root
  of this whole project.
- **Kalra et al. (2020), "Deep Polarization Cues for Transparent Object Segmentation" (ClearGrasp-era
  work)** — the learned successor to this project's hand-thresholded detector; read it to see what a
  CNN adds over a DoLP threshold.
- **Fischer et al. and other shape-from-polarization surveys** — the AoLP-as-surface-normal-cue idea
  this project's dome (radial AoLP) and metal bar (constant AoLP) sections illustrate directly.
- **cuRobo / MoveIt grasp pipelines** — where a `TransparentObjectMask` like this project's output
  would actually plug into a real bin-picking stack (see README "System context" and 19.01).

## Exercises

1. Change `kDolpThreshold` in `kernels.cuh` and re-run — watch `GATE negative_control` start failing as
   background noise crosses the threshold, and read why in `THEORY.md` "Numerical considerations".
2. Implement an edge-aware polarization demosaic (the polarization analogue of 01.23's Malvar-He-Cutler
   upgrade over plain bilinear) and measure how much it tightens `stokes_accuracy`'s interior MAE.
3. Add a fourth object: a matte surface tilted near grazing incidence. Predict its DoLP from the
   Fresnel equations before running the demo, then check your prediction against `dolp.pgm`.
4. Replace the metal bar's phenomenological DoLP curve with the real complex-refractive-index Fresnel
   equations (THEORY.md "Where this sits in the real world" gives the starting formula) and compare.
5. Extend `run_detection_gpu`/`run_detection_cpu` to report IoU (not just recall) against
   `truth_maps.csv`'s labels, and investigate the demosaic edge-bleed effect `kInteriorMarginPx` exists
   to exclude (main.cu's `[info]` line for `stokes_accuracy` reports how many pixels it drops).

## Limitations & honesty

- **The DoFP super-pixel layout is illustrative, not vendor-verified.** The 0/45/90/135 arrangement in
  `kernels.cuh` matches the *style* of a real DoFP sensor's construction but its exact orientation is
  not checked against a specific datasheet — PRACTICE.md §2 says so explicitly and dates the claim.
  Real integration work MUST verify against the actual part's datasheet.
- **The metal object's DoLP is a documented phenomenological curve, not the real physics.** Real metals
  are conductors and need complex-refractive-index Fresnel equations; this project scopes that out
  (stated in `THEORY.md` "Where this sits in the real world" and Exercise 4 above) to keep the glass
  physics — the project's real teaching payload — front and center.
- **Demosaic is plain bilinear, not edge-aware.** The "instantaneous FOV" artifact (each 2x2
  super-pixel's four measurements come from four *different* physical photosites) is documented
  honestly in `kernels.cu` and worked around with an interior-margin exclusion in the accuracy gate, not
  fixed — Exercise 2 is the fix.
- **128x128 test canvas, one lighting condition, one incidence-angle set per object.** A production
  system runs at full sensor resolution and must handle varying illumination, multiple simultaneous
  incidence angles per surface (curved/faceted glass), and non-specular partial occlusion — none
  modeled here.
- **Synthetic data throughout**, labeled everywhere it appears (CLAUDE.md §8) — chosen because it is the
  only way to get exact per-pixel ground-truth DoLP/AoLP a real photograph never provides (`data/README.md`).
- **Sim-validated only.** This project's output is a detection mask, not a control command — it does
  not itself move hardware — but any downstream costmap/grasp-planning consumer that *does* command
  motion (14.02, 19.01) must be validated in simulation before any real-hardware use; nothing here is
  safety-certified (CLAUDE.md §1).
