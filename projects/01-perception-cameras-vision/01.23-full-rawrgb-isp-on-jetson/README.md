# 01.23 — Full RAW→RGB ISP on Jetson (Argus + custom CUDA stages)

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Full RAW→RGB ISP on Jetson (Argus + custom CUDA stages)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds the **complete classical RAW→RGB image signal processor (ISP)** as one CUDA C++
pipeline: black-level subtraction, lens-shading (vignetting) correction, defective-pixel correction,
automatic white balance (two independent estimators), demosaic (the Malvar-He-Cutler gradient-
corrected method, plus a bilinear baseline for comparison), the color-correction matrix, and gamma —
the same eight stages every real camera ISP runs, in the same order, on the same kind of RAW10
sensor data. Every stage is implemented, measured, and gated against synthetic ground truth: a
160×120 synthetic RAW sensor is generated from a hand-authored scene (a 24-patch color chart, a
dedicated white-balance card, and a hashed texture region) rendered through a documented forward
sensor model (spectral crosstalk, two illuminants, lens shading, a committed defect map, shot+read
noise), so every correction stage has an exact, known target to be checked against.

**Scoping (read this first).** The catalog bullet asks for "Argus + custom CUDA stages" on Jetson.
Argus (NVIDIA's libargus camera API) and Jetson's fixed-function hardware ISP do not exist on the
owner's desktop RTX 2080 SUPER and cannot be emulated there. Per `CLAUDE.md` section 5's policy for
hardware-dependent projects, this project builds the **desktop-runnable teaching core** — the
complete RAW→RGB radiometric pipeline as pure CUDA, running on synthetic sensor data — and documents
the real Jetson/Argus/libargus deployment path in full below and in `THEORY.md`/`PRACTICE.md`,
without ever faking a Jetson number. See [Limitations & honesty](#limitations--honesty) and
[The Jetson story](#the-jetson-story) for the complete accounting.

> **Template placeholder notice.** This project's `src/` has been fully replaced — there is no
> SAXPY placeholder remaining. Every file below is the real implementation.

## What this computes & why the GPU helps

Every ISP stage is a **map** (black level, shading, white balance, CCM, gamma — one thread, one
pixel, no neighbor reads) or a small **stencil** (defect correction: a same-Bayer-phase 4-neighbor
median; demosaic: a 5×5 gather) except AWB gain estimation, which is a **reduction** (a deterministic
block-tree sum+max over the whole mosaic, extending sibling flagship 01.01's normalize pattern to
compute two estimators — gray-world and white-patch — in one pass). None of the eight stages has any
inter-pixel data dependency a naive `map` mapping can't express, which is exactly why an ISP is the
textbook first GPU program in real camera pipelines: every stage saturates memory bandwidth long
before it saturates compute.

- **Pattern:** map (6 of 8 stages) / stencil (defect correction, demosaic) / block-tree reduce (AWB).
- **Fusion:** stages 1–4 (black level → shading → defect → white balance) also ship as ONE fused
  kernel — see "The algorithm in brief" and `kernels.cuh`'s "Fusion economics" for why this
  particular fusion is *cheaper than free* per non-defective pixel (measured: 66.7% memory-traffic
  reduction, staged 0.14–0.19 ms vs fused ~0.05 ms on the reference machine).
- **Measured reality (RTX 2080 SUPER, sm_75, Release, single-shot):** stages 1–4 staged ≈0.16 ms,
  fused ≈0.05 ms; the CPU oracle for the whole D65 pipeline runs in ≈3 ms. At this project's teaching
  resolution the absolute numbers are trivial — the *shape* of the win (fusion eliminates three
  materialized intermediates) is the lesson, not the millisecond count; see "Where this sits in the
  real world" in `THEORY.md` for how the same argument scales to a real 12–48 MP sensor.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the very first box in the **perception** layer — stage zero of every
  camera-bearing robot. Everything downstream (detection, SLAM, visual servoing) consumes RGB or
  grayscale frames this stage produces; nothing upstream of it exists except the sensor and the lens.
- **Upstream inputs:** a raw Bayer sensor frame over MIPI CSI-2 (message shape: a `RawImage` —
  width, height, bit depth, Bayer pattern, timestamp — analogous to `sensor_msgs/Image` with an
  unrectified Bayer encoding), plus per-device calibration data (black level, the defect map, the
  lens-shading gain map, the CCM) loaded once at bring-up.
- **Downstream consumers:** *everything* that touches vision. This repo's own siblings show the
  split honestly: **01.01** (`full-gpu-image-pipeline`) is a **geometry-focused** teaching pipeline
  (debayer → undistort → rectify → resize → normalize) that assumes a *already-demosaiced, already
  radiometrically clean* input — this project is what actually produces that clean input from a raw
  sensor. **01.04** (feature pipeline), **01.11** (low-light denoising), and **01.02** (stereo depth)
  all consume this stage's `Image` output by name.
- **Rate / latency budget:** a real camera ISP runs at the sensor's frame rate — 30–60 Hz for a
  perception camera, sometimes higher for a global-shutter industrial sensor — with a total pipeline
  latency budget on the order of a few milliseconds per frame so it does not eat into downstream
  processing time (SYSTEM_DESIGN.md item 1). This project's teaching-resolution (160×120) measured
  time (≈0.3 ms for every stage combined, both illuminants, both staged and fused) is not a
  benchmark at real sensor resolutions — see "The Jetson story" below for the honest scaling math.
- **Reference robot(s):** the **warehouse AMR** (a fixed forward-facing camera feeding
  obstacle/lane perception) and the **inspection-class field/manipulation camera** (where color and
  radiometric accuracy — not just geometry — genuinely matter, e.g. defect or produce-quality
  inspection) are this project's most direct fits in `docs/SYSTEM_DESIGN.md`'s five reference robots.
- **In production:** on Jetson, most of stages 1–6 would run in the **hardware ISP** (via libargus)
  before a single CUDA kernel ever sees a pixel — see "The Jetson story" for exactly which stages
  custom CUDA typically replaces or supplements, and why.
- **Owning team:** camera systems / ISP engineering — a specialized sub-team of perception that
  usually also owns camera bring-up, calibration, and tuning (SYSTEM_DESIGN.md item 5; PRACTICE.md
  section 4 names the adjacent teams and roles).

## The algorithm in brief

- **Black level + saturation** — subtract the sensor's dark-current/ADC offset, clamp to the usable
  code range. → [THEORY.md "The problem"](THEORY.md#the-problem--physics--engineering-first)
- **Lens shading correction** — divide by a radial polynomial gain map `V(r) = 1 + a2·r² + a4·r⁴`
  (the same functional family as sibling **01.09**'s vignetting model, reduced to two terms). →
  [THEORY.md "The math"](THEORY.md#the-math)
- **Defective-pixel correction** — a committed factory defect list (loaded at runtime, broadcast via
  `__constant__` memory), corrected by the median of same-Bayer-phase neighbors. →
  [THEORY.md "The algorithm"](THEORY.md#the-algorithm)
- **White balance** — **gray-world** (assume the scene averages to neutral gray) AND **white-patch /
  max-RGB** (assume the brightest pixel is a white/specular highlight), both implemented as one GPU
  reduction; the project also demonstrates gray-world's textbook failure mode on a dominant-color
  scene. → [THEORY.md "Illuminant physics"](THEORY.md#the-math)
- **Demosaic** — **Malvar-He-Cutler** (Malvar, He & Cutler 2004): a gradient-corrected linear
  interpolation that measurably beats a **bilinear** baseline (also implemented, as the explicit
  comparison point sibling **01.01** deferred). → [THEORY.md "MHC demosaic derivation"](THEORY.md#the-algorithm)
- **Color-correction matrix (CCM)** — a 3×3 matrix, derived **by hand** in `THEORY.md` as the inverse
  of the synthetic sensor's documented spectral crosstalk matrix. → [THEORY.md "The math"](THEORY.md#the-math)
- **Gamma / sRGB encode** — the exact piecewise sRGB transfer function (perceptual coding), continuing
  sibling **01.08**'s tone-curve lineage. → [THEORY.md "sRGB gamma purpose"](THEORY.md#the-math)
- **Stage fusion** — stages 1–4 also run as one fused kernel; `THEORY.md` "The GPU mapping" continues
  01.01's staged-vs-fused economics for this project's specific (much cheaper) fusion case.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/full-rawrgb-isp-on-jetson.sln`](build/full-rawrgb-isp-on-jetson.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/full-rawrgb-isp-on-jetson.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md section 5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
libargus, no OpenCV, no TensorRT (this project is deliberately hand-rolled CUDA end to end, per
CLAUDE.md section 5's default policy).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU and
all ten physical gates):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at, and the list of visual artifacts written to `demo/out/`.

## Data

The committed sample is **100% synthetic** (CLAUDE.md section 8's default): two RAW10-in-uint16
sensor mosaics (D65 and tungsten illuminant) of a hand-authored 160×120 scene — a 24-patch
Macbeth-style color chart, a dedicated white-balance reference card, and a hashed texture region —
rendered through a documented forward sensor model (spectral crosstalk, per-illuminant gain, radial
lens shading, a committed 16-entry defect map, shot+read noise, 10-bit quantization). No public
dataset applies (`scripts/download_data.ps1`/`.sh` are honest no-ops — see `data/README.md` for why).
Regenerate with `python scripts/make_synthetic.py` (deterministic, seed 42). Full field
documentation, sizes, and SHA-256 checksums: [`data/README.md`](data/README.md).

## Expected output

Ten independent gates plus a per-stage GPU-vs-CPU `VERIFY`, all measured on the reference machine
(RTX 2080 SUPER, sm_75, Release, `demo/run_demo.ps1`):

| Gate | Measured | Tolerance |
|------|----------|-----------|
| `VERIFY` (GPU vs CPU, all 9 stages) | max\|gpu-cpu\| = 0.000000 for every float stage (bit-exact on this GPU); gamma 1.0 DN | ≤5e-4 (float stages), ≤1.5 DN (gamma) |
| `black_level_residual` | mean\|residual\| = 0.00236 | ≤0.030 |
| `shading_flatness` | inner/outer radial gap = 0.00005; mean\|residual\| = 0.00264 | ≤0.030 each |
| `defect_recovery` | mean\|corrected-truth\| at 16 defects = 0.00114; false corrections = 0 | ≤0.050; ≤1e-5 |
| `demosaic_psnr` | MHC 33.31 dB vs bilinear 31.79 dB — **gap 1.52 dB** | gap ≥1.0 dB |
| `awb_accuracy` | rel. gain error — D65 gray 0.058/white 0.013, tungsten gray 0.204/white 0.230 | ≤0.26 each |
| `awb_red_crop_failure` (negative control) | gray-world R-gain on the red-heavy crop = 0.437 vs true 1.0 (deviation 0.563) | must deviate ≥0.15 |
| `ccm_color_chart` | mean patch RGB-distance 6.4–7.7 / max 13.5–16.4 (0–255 scale, 24 patches) | ≤18.0 mean, ≤40.0 max |
| `tungsten_wrong_awb_negative_control` | chart error correct-AWB 21.2 vs wrong-AWB 38.9 — **1.84× inflation** | ≥1.5× |
| `end_to_end_psnr` | D65 26.36 dB, tungsten 23.47 dB (vs the reference rendering) | ≥24.0 dB / ≥21.5 dB |
| `fused_vs_staged` | max\|fused-staged\| = 0.000000 | ≤1e-4 |

Fusion economics (stages 1–4, D65, idealized no-cache-reuse byte model): staged 345,600 bytes vs
fused 115,200 bytes — **66.7% memory-traffic reduction**; measured staged ≈0.16–0.19 ms vs fused
≈0.05 ms. Every number above is a *measured-then-margined* tolerance (CLAUDE.md section 8: never
fabricated) — small floating-point drift across GPU architectures (different FMA-contraction
choices) is expected and is why tolerances carry margin above the measured value, not equality.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — start here: the whole sensor/ISP model (geometry, RAW10
   convention, shading polynomial, crosstalk matrix, CCM derivation pointer, MHC coefficient tables,
   the 24-patch chart layout) lives in one heavily-commented header every other file agrees with.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels. Read stages 1–4 first (the simplest maps),
   then the AWB reduction (extends 01.01's block-tree pattern), then the demosaic section (the
   project's centerpiece — `mhc_eval()` and the four coefficient tables), then CCM/gamma.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of every kernel; read
   it side by side with `kernels.cu` to see exactly what parallelized (the twin-independence ruling
   in its file header explains what is shared data vs. independently retyped algorithm).
4. [`src/main.cu`](src/main.cu) — orchestration: loads both illuminants' RAW mosaics, runs staged and
   fused pipelines, runs VERIFY, then all ten gates (each is a self-contained block — read
   `demosaic_psnr` and `awb_red_crop_failure` first, the two most interesting).
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the forward sensor model (the exact
   inverse of what the ISP undoes) and the scene layout (chart, AWB card, hashed texture).
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `paths.h` (copied, not shared — CLAUDE.md
   section 4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md section 4.1).

- **Malvar, He & Cutler (2004), "High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned
  Color Images"** — the demosaic algorithm this project's centerpiece kernel implements; read the
  original paper for the full derivation this project's `THEORY.md` summarizes.
- **NVIDIA libargus / Jetson Multimedia API** — the real camera-capture and ISP-control API this
  project's scoping note stands in for; see "The Jetson story" below and `PRACTICE.md` section 3 for
  the concrete API shape.
- **libcamera** (open-source, Linux) — a full software camera stack including AWB/AE/demosaic
  algorithms in the open, worth reading end to end once this project's stages feel familiar.
- **rawpy / LibRaw / dcraw** — open-source RAW decoders implementing (among others) AHD and other
  demosaic algorithms; a good place to compare MHC against its contemporaries.
- **01.01 (`full-gpu-image-pipeline`)** — this repo's geometry-focused sibling; read its "Limitations
  & honesty" for exactly which radiometric stages it *assumed away* and this project implements.
- **01.09 (`photometric-vignetting-calibration-kernels`)** — this project's lens-shading model is a
  2-term truncation of that project's 3-term radial polynomial; read it for the full calibration story
  (fitting the polynomial from data, rather than assuming it known, as this project does).

## Exercises

1. **Look at the ringing.** Open `demo/out/demosaiced_mhc_d65.ppm` and `demosaiced_bilinear_d65.ppm`
   side by side (any PPM viewer, or `python -c "from PIL import Image; Image.open(...).show()"`).
   Find the hashed-texture region's hard block edges and see MHC's sharper reconstruction — then find
   a place where MHC *overshoots* (a color that doesn't exist in the original palette). That overshoot
   is the same gradient-correction mechanism that makes MHC win on average.
2. **Break gray-world on purpose.** Shrink `kRedCropW`/`kRedCropH` in `kernels.cuh` to cover only the
   single reddest patch and re-run — watch `awb_red_crop_failure`'s measured deviation grow.
3. **Feel the tungsten cast.** Open `final_tungsten.ppm` vs `final_tungsten_wrong_awb.ppm` — the
   second one is what a camera with a stuck D65 white-balance preset produces under indoor lighting.
4. **Tune the shading polynomial.** Change `kShadeA2`/`kShadeA4` in `kernels.cuh` to model a more
   extreme (say, 50%) corner falloff, regenerate the sample, and watch `shading_flatness`'s residual
   change — at what falloff does `kShadeGainFloor`'s guard actually start mattering?
5. **Fuse further.** CCM and gamma are both pure per-pixel maps with no neighbor reads — fuse them
   into one kernel (`ccm_gamma_fused_kernel`) the way stages 1–4 are fused, and measure the (small,
   honestly reported) additional saving.

## Limitations & honesty

- **Desktop teaching core, not a Jetson deployment (CLAUDE.md section 5).** This project never runs
  on Jetson hardware, never links libargus, and never touches the Jetson hardware ISP. Every
  measured number in this README is from the owner's desktop RTX 2080 SUPER. "The Jetson story"
  below documents the real deployment path honestly, as a **documented hardware path**, never as a
  fabricated result.
- **Illustrative, not certified, color science.** The 24-patch chart's reference values, the
  crosstalk matrix, the illuminant gains, and the noise model are hand-chosen, physically-motivated
  numbers — not measurements of a real sensor, and not the certified X-Rite ColorChecker values.
  `data/README.md` "Provenance & honesty notes" states this explicitly.
- **No dynamic defect detection.** The defect map is a fixed, committed list (a factory calibration
  stand-in); a real ISP also runs online defect detection to catch pixels that degrade after
  shipping. `THEORY.md` "Where this sits in the real world" names this gap.
- **No local tone mapping / HDR.** This project's gamma stage is a single global sRGB curve; a modern
  camera ISP layers local tone mapping, HDR fusion (see sibling **01.08**), and noise reduction on
  top. Out of scope here — this project is the classical *radiometric* core only.
- **Small teaching resolution (160×120).** Chosen so the CPU oracle, the committed sample, and the
  build all stay fast and tiny; "The Jetson story" gives the honest scaling arithmetic to real sensor
  resolutions.
- **Sim-validated only where it matters (CLAUDE.md section 1).** This project's output is an image,
  not a motion command — the weakest form of the repo-wide safety caveat applies, but is stated for
  completeness: nothing here is certified for any downstream safety-relevant use.

## The Jetson story

*(Continues "System context" above; grounded further in [`PRACTICE.md`](PRACTICE.md) section 3.)*

A real Jetson camera pipeline for a robot rarely runs a hand-written CUDA ISP end to end — it is
built from three layers, and knowing where custom CUDA stages like this project's actually earn their
keep is the whole point of the catalog bullet this project answers:

1. **The sensor + CSI-2 driver.** A MIPI CSI-2 camera (e.g., an IMX-series sensor on a Jetson-
   compatible carrier board) streams packed RAW10/RAW12 over 2–4 data lanes to Jetson's dedicated CSI
   receiver hardware, which a V4L2 kernel driver exposes as a capture device.
2. **libargus + the fixed-function hardware ISP.** NVIDIA's libargus is the capture/control API;
   under it, Jetson SoCs carry a **dedicated hardware ISP block** (part of the VIC/ISP engine, not
   the GPU) that does black level, lens-shading correction, defect correction, Bayer-domain AWB/AE
   statistics collection, demosaic, color correction, and tone mapping — in fixed-function silicon,
   at the sensor's full frame rate, for a fraction of the GPU's power draw. **This is stages 1–7 of
   this project, already built, in hardware**, on every Jetson.
3. **Where custom CUDA stages make sense (the catalog's actual premise).** Given the hardware ISP
   exists, why write CUDA stages at all? Three honest reasons, each a real production pattern: (a)
   **algorithm control** — the hardware ISP's demosaic/AWB/tone-curve are fixed or only lightly
   tunable through vendor tuning tools; a robotics team that needs a *specific* algorithm (a
   depth-aware demosaic near a LiDAR-projected edge, a task-specific AWB target, a linear-light output
   for a downstream neural network instead of a display-gamma one) writes it in CUDA and either
   bypasses the hardware ISP or runs alongside it on a subset of stages; (b) **non-standard sensors**
   — thermal, event, or scientific sensors the hardware ISP was never built for still need *some* of
   these stages (black level, defect correction) and there is no hardware path for them; (c)
   **research/teaching transparency** — exactly this project's reason: a fixed-function ISP is a
   black box, and understanding what it does requires building the stages by hand at least once.

- **CSI-2/driver reality.** Bringing up a new sensor on Jetson means writing or adapting a V4L2/kernel
  driver plus a libargus sensor description (resolution, Bayer pattern, black level, gain/exposure
  register maps) — real, nontrivial embedded-systems work `PRACTICE.md` section 1 sketches; this
  project assumes that work is done and starts from "a RAW mosaic already exists in memory."
- **Rate at real resolutions.** A production automotive/robotics camera runs 30–60 Hz at 1–8 MP;
  this project's 160×120 teaching resolution is ≈480–2600× fewer pixels. Every kernel here is
  memory-bandwidth-bound and embarrassingly parallel across pixels, so the honest scaling estimate is
  roughly linear in pixel count (measured stages 1–4 fused: ≈0.05 ms at 19,200 px → a back-of-envelope
  ≈2.4–13 ms at 1–5 MP on the *same* GPU, before accounting for the RTX 2080 SUPER's far larger
  memory bandwidth than a Jetson Orin's — a real projection exercise, not a claim, left to Exercise 6
  in spirit and `THEORY.md`'s "Where this sits in the real world" in detail).
