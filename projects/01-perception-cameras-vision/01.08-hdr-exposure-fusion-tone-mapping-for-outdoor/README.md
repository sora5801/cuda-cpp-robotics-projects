# 01.08 — HDR exposure fusion + tone mapping for outdoor robots

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `HDR exposure fusion + tone mapping for outdoor robots`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

An outdoor scene routinely exceeds what one 8-bit exposure can capture: a robot camera looking across a
sunlit parking lot with a shaded loading dock and a vehicle's dark underbody easily spans 4-5 decades of
radiance, while a single sensor exposure resolves roughly 2. This project builds, teaches, and directly
compares the **two classic answers** to that gap, both from a bracket of four exposures
(`1/1000, 1/125, 1/30, 1/8` s) of the *same* synthetic outdoor scene:

- **Path A — HDR reconstruction + tone mapping.** Recover the camera's response function (CRF) from the
  bracket itself (Debevec & Malik's `gsolve`, solved on the host as a small dense linear system), merge
  the four exposures into one linear-radiance image, then compress that radiance down to a displayable
  image two ways: a **global** Reinhard operator and a **local**, pyramid-based base/detail split.
- **Path B — Mertens exposure fusion.** Skip radiance and the CRF entirely: blend the four *display*
  images directly, weighted per-pixel by contrast x well-exposedness, combined across scales with a
  Laplacian pyramid — and, as an explicit in-demo comparison, the same blend done the *naive* way (one
  full-resolution weighted average, no pyramid) so the halo/ghosting failure mode Path B's multiscale
  blending exists to fix is visible, not just asserted.

Both paths, and the naive baseline, are implemented on the **GPU** (a dozen small reusable CUDA kernels —
per-pixel maps, 3x3/5x5 stencils, a shared-memory reduction, and 4-way weighted combines) with an
**independent CPU twin** for every kernel, verified against each other, and graded against **six
independent gates** tied to ground truth: does the recovered CRF match the known synthetic curve; does
the recovered radiance match the exact synthetic radiance; is the tone-mapped output monotonic and
correctly ranged; does fusion/tone-mapping actually beat every single input exposure (the entire reason
this project exists); is fine texture preserved in both a deep-shadow and a highlight region; and is the
naive blend measurably more prone to haloing than the multiscale one. All six pass on the reference
machine — see [Expected output](#expected-output) and `demo/gates_metrics.csv`.

**Grayscale scope, stated up front:** this project's synthetic scene is single-channel (no color). That
is a deliberate simplification — HDR reconstruction and tone mapping are luminance problems first, and
grayscale removes a whole axis of complexity (debayering, white balance) this project is not about.
Mertens' classic third weight term (saturation = standard deviation across R/G/B) has no meaning here and
is honestly dropped rather than faked — see [Limitations & honesty](#limitations--honesty).

## What this computes & why the GPU helps

Every stage in both paths is one of four GPU patterns, reused deliberately across the whole project
rather than hand-rolled per stage (see `kernels.cu`'s file header):

- **Map** — per-pixel, independent of every other pixel: the CRF-weighted radiance merge, the Reinhard
  display map, log/exp/affine domain conversions, uint8-to-float normalization. One thread per pixel
  saturates memory bandwidth, exactly like every map kernel in this repository.
- **Stencil** — a small fixed neighborhood per output pixel: the 5x5 Gaussian blur+downsample that builds
  every Gaussian pyramid level (`gaussian_reduce_kernel`), the bilinear upsample that reconstructs them
  (`bilinear_expand_kernel`), and Mertens' 3x3 Laplacian contrast measure.
- **Reduction** — many-to-one: `luminance_log_sum_kernel` computes Reinhard's log-average scene luminance
  with a textbook shared-memory tree reduction plus one atomic add per block, the pattern this repo's
  reductions all use once and then get to assume.
- **Batched-solve** — Debevec-Malik's CRF recovery is a small (~320x320) dense linear least-squares
  system, solved once per capture with hand-rolled Gaussian elimination on the **host** (see
  `crf_solve_debevec`'s header for why this is the one deliberately *non*-GPU stage: a one-time
  calibration this small has too little work to amortize a kernel launch).

The GPU payoff is not any single kernel's speed (these images are 160x120 — the CPU oracle finishes the
*entire* pipeline in single-digit milliseconds too, see `demo/expected_output.txt`'s timing lines,
reported as a teaching artifact, never a benchmark claim). The payoff is architectural: a real
1920x1080-or-larger camera feed, run through this exact same kernel sequence, is where the constant-time
per-pixel maps and the O(pixels) reductions this project teaches actually separate a real-time pipeline
from an offline one.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception, specifically the camera **front-end / ISP boundary** — upstream of
  every vision algorithm in this repository's domain 01 (feature detection, stereo, optical flow,
  SLAM front-ends), because those algorithms all implicitly assume a well-exposed 8-bit (or tone-mapped)
  image. HDR fusion sits between raw sensor capture and everything else.
- **Upstream inputs:** a bracketed capture — `N` LDR frames (here `N=4`) at known exposure times,
  message-shaped as repeated `Image` (01.01's convention: row-major uint8, documented width/height) plus
  a small `ExposureBracket` metadata struct (per-frame shutter time, gain). Produced by the camera driver
  or ISP (see 01.01-full-gpu-image-pipeline, cited by name: that project's debayer/rectify stages are
  what would run *before* this one on a real Bayer sensor; this project starts from already-debayered,
  already-grayscale frames).
- **Downstream consumers:** every perception consumer in this repository that assumes correctly-exposed
  input — 01.04-feature-pipeline by name: FAST/Harris corner and ORB descriptor quality both collapse in
  clipped regions (a fully-saturated or fully-black patch has no gradient to detect a corner from, and no
  local contrast for a descriptor to discriminate on), which is precisely the failure this project's
  `dynamic_range_coverage` and `detail_preservation` gates make quantitative. Stereo depth, optical flow,
  and SLAM front-ends inherit the same dependency one level further downstream.
- **Rate / latency budget:** bracketed HDR is fundamentally **not free-running video-rate** — it needs
  `N` sequential exposures of a (nominally) static scene, so at `N=4` and the exposure times used here
  (up to 1/8 s for the longest frame) one full bracket takes on the order of ~150 ms of *capture* time
  alone, before this project's ~1-2 ms of *compute* (see `demo/expected_output.txt`'s `[time]` lines) —
  see [Limitations & honesty](#limitations--honesty) for the motion/ghosting consequence this implies,
  and `docs/SYSTEM_DESIGN.md` item 1 for where perception's usual 30-60 Hz budget sits by comparison. A
  production system either accepts a slow HDR-bracket cadence for a genuinely static scene (mapping,
  calibration, periodic re-exposure of a fixed workspace) or replaces bracketing with **sensor-level HDR**
  silicon that captures the full dynamic range in one exposure — see below.
- **Reference robot(s):** the outdoor **AMR** (delivery/warehouse-yard robots crossing between direct sun
  and shaded loading docks) and **agriculture** field robots (30.01-crop-detection-and-yield-estimation
  by name: row-crop imagery under harsh, variable outdoor illumination is a canonical HDR use case) —
  both named as reference robots in `docs/SYSTEM_DESIGN.md`. AV stacks exiting a tunnel into direct
  sunlight face the identical problem, though at video rate, which is exactly why they favor the
  sensor-level HDR path described below rather than this project's bracketing approach.
- **In production:** two real alternatives to what this project teaches, both worth knowing by name (see
  THEORY.md "Where this sits in the real world" for depth): (1) **software HDR from a bracket**, exactly
  this project's Path A/B, done at scale by OpenCV's `createCalibrateDebevec`/`createMergeMertens`, or by
  a production ISP's HDR block; (2) **sensor-level HDR** — dual-conversion-gain or split-pixel sensors
  (e.g., Sony's DOL-HDR staggered-exposure readout) that capture multiple exposures *within one sensor
  read*, eliminating the inter-frame motion problem this project's bracketing approach cannot avoid.
  Real-time robots exiting tunnels or entering shadow overwhelmingly prefer option (2); this project
  teaches option (1) because it is the one a learner can build and verify from first principles.
- **Owning team:** perception / camera-systems engineering — the team that owns sensor selection, ISP
  tuning, and the driver boundary between silicon and the rest of the autonomy stack
  (`docs/SYSTEM_DESIGN.md` item 5).

## The algorithm in brief

- **Debevec-Malik CRF recovery** (`crf_solve_debevec`, host) — log-domain weighted least squares with a
  smoothness prior, solved as ~320x320 normal equations via hand-rolled Gaussian elimination (cites
  33.01-batched-small-matrix-linalg for the general small-dense-solve pattern). See
  [`THEORY.md`](THEORY.md#the-math).
- **Radiance merge** (`radiance_merge_kernel`) — hat-weighted, per-pixel log-domain combination of the
  four exposures through the recovered CRF into one linear HDR radiance image, with an explicit fallback
  for pixels clipped in every exposure. See [`THEORY.md`](THEORY.md#the-algorithm).
- **Global tone mapping** (`run_reinhard_global_gpu`) — Reinhard's photographic operator: a log-average-
  luminance GPU reduction, then a per-pixel `L/(1+L)` display map, strictly bounded in `[0,1)`. See
  [`THEORY.md`](THEORY.md#the-math).
- **Local tone mapping** (`run_local_tonemap_gpu`) — a "bilateral-grid-lite" base/detail split: a
  2-level Gaussian pyramid built in log-radiance (citing 01.03-optical-flow's pyramid-kernel precedent),
  the coarse base compressed toward the scene mean, the detail layer mildly boosted, recombined and
  min-max normalized. See [`THEORY.md`](THEORY.md#the-algorithm).
- **Mertens exposure fusion** (`run_mertens_gpu`) — per-pixel, per-exposure weights (contrast x
  well-exposedness — saturation dropped, grayscale scene), normalized to sum to 1, blended via a
  Laplacian pyramid (weights in a Gaussian pyramid, image bands in a Laplacian pyramid, Burt & Adelson
  1983). See [`THEORY.md`](THEORY.md#the-algorithm).
- **Naive single-scale blend** (`run_mertens_gpu`'s other output) — the SAME normalized weights applied
  as one full-resolution weighted average, no pyramid — the explicit failure-case baseline the
  `halo_check` gate quantifies. See [`THEORY.md`](THEORY.md#numerical-considerations).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/hdr-exposure-fusion-tone-mapping-for-outdoor.sln`](build/hdr-exposure-fusion-tone-mapping-for-outdoor.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/hdr-exposure-fusion-tone-mapping-for-outdoor.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — this project links only the CUDA runtime
(`cudart_static.lib`) and the C++17 standard library, the repo's default budget (CLAUDE.md §5). No
cuBLAS/cuSOLVER: the CRF's dense solve is small and one-time enough that hand-rolled host Gaussian
elimination is both simpler and the more honest teaching choice (see `kernels.cu`'s
`solve_linear_system` header for the full reasoning).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is **entirely synthetic** (CLAUDE.md §8 default), generated by
`scripts/make_synthetic.py` (fixed seed 42, deterministic, reproducible bit-for-bit): an analytic
outdoor scene (sun disk, sky, open shade, sunlit concrete with a painted line marking, a deep shadow
rectangle "under a parked vehicle", hashed-noise texture everywhere, and a noise-free monotonic
calibration strip used only by the `tone_map_range` gate) spanning ~5 decades of radiance, rendered
through a **known** analytic camera response function into four clipped, noisy 8-bit exposures. The
committed files also include the exact ground-truth radiance (a raw float32 dump) and every generation
parameter (`params.csv`). Full field-by-field documentation, checksums, and the regeneration command are
in [`data/README.md`](data/README.md).

## Expected output

The demo builds (if needed), loads the four-exposure bracket, runs the shared CRF calibration once, runs
both HDR paths on the GPU **and** on an independent CPU oracle, and reports:

- **`VERIFY:`** — GPU matches the CPU reference within a documented per-stage tolerance, for all five
  independently-implemented pipeline stages (radiance merge, Reinhard, local tone map, naive blend,
  Mertens fusion). See `src/main.cu`'s tolerance constants and `THEORY.md` "How we verify correctness"
  for why each bound is sized the way it is.
- **Six `GATE` lines**, each an independent check against ground truth (not a GPU-vs-CPU comparison):
  `crf_recovery`, `radiance_reconstruction`, `tone_map_range`, `dynamic_range_coverage`,
  `detail_preservation`, `halo_check` — see [The algorithm in brief](#the-algorithm-in-brief) and
  `THEORY.md` for what each one measures and why.
- **`ARTIFACT:`** — every artifact in `demo/out/`: the four labeled input exposures, `reinhard_global.pgm`,
  `local_tonemap.pgm`, `mertens_fusion.pgm`, `naive_blend.pgm` (open these side by side — the halo at the
  painted-line/concrete boundary is visible in `naive_blend.pgm` and much reduced in `mertens_fusion.pgm`),
  `crf_curve.csv` (recovered vs. known-true CRF, for plotting), and `gates_metrics.csv` (every measured
  number, machine-readable).
- **`RESULT: PASS`** only when VERIFY and all six gates pass. The canonical stable lines (no timings, no
  device names — those vary machine to machine) live in [`demo/expected_output.txt`](demo/expected_output.txt).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced data-layout contract: image geometry, the
   exposure-time array, the CRF representation, the pyramid layout, every kernel/launcher/reference
   signature. Read this first — everything else implements this contract.
2. [`src/kernels.cu`](src/kernels.cu) — the dozen reusable GPU kernels (SECTION 1), their launch
   wrappers (SECTION 2), the two HDR paths' host orchestration (`run_reinhard_global_gpu`,
   `run_local_tonemap_gpu`, `run_mertens_gpu`, SECTION 3), and the shared CRF solver
   (`crf_solve_debevec`, SECTION 4) — start with `radiance_merge_kernel`, then `gaussian_reduce_kernel` /
   `bilinear_expand_kernel` (the pyramid primitives both HDR paths reuse), then `run_mertens_gpu` (the
   most involved orchestration — it produces both the real fusion AND the naive baseline).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of every primitive above,
   plus the note on which ONE function (`crf_solve_debevec`) is deliberately *shared*, not duplicated,
   and why.
4. [`src/main.cu`](src/main.cu) — orchestration: load the bracket, run the shared CRF calibration once,
   run both paths on GPU and CPU, VERIFY, evaluate the six gates, write every artifact. Read the six gate
   blocks in order — each is a self-contained lesson in how to grade an HDR pipeline against ground truth.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s multi-candidate file
   resolution (needed here because this project, unlike the SAXPY placeholder, actually reads
   `data/sample/` and writes `demo/out/`).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Debevec & Malik, "Recovering High Dynamic Range Radiance Maps from Photographs," SIGGRAPH 1997** —
  the original `gsolve` algorithm this project's `crf_solve_debevec` implements (with normal equations +
  Gaussian elimination in place of the paper's SVD, for the "small dense system, hand-rollable" teaching
  goal — see `kernels.cu`).
- **Mertens, Kautz & Van Reeth, "Exposure Fusion," Pacific Graphics 2007** — Path B's algorithm; study
  the full three-term weight formula (this project drops the saturation term — see
  [Limitations & honesty](#limitations--honesty)) and its Laplacian-pyramid blending.
- **Burt & Adelson, "The Laplacian Pyramid as a Compact Image Code," 1983** — the REDUCE/EXPAND pyramid
  formalism both `gaussian_reduce_kernel`/`bilinear_expand_kernel` implement a simplified version of.
- **Reinhard, Stark, Shirley & Ferwerda, "Photographic Tone Reproduction for Digital Images," SIGGRAPH
  2002** — the global operator `run_reinhard_global_gpu` implements (the simple form; study the paper's
  full white-point-burnout extension as a follow-up).
- **OpenCV's `createCalibrateDebevec` / `createMergeDebevec` / `createMergeMertens` / `Tonemap`** — the
  production, battle-tested implementations of every algorithm here; compare their API surface and
  default parameters against this project's `main.cu` constants.
- **01.01-full-gpu-image-pipeline** (this repo) — the debayer/rectify stage that would run *before* this
  project on a real Bayer sensor; also the PGM/PPM I/O and grid/block launch-config conventions this
  project reuses.
- **01.03-optical-flow** (this repo) — the pyramid-kernel precedent (`downsample_area2x_kernel`,
  `run_pyramidal_lk_gpu`'s orchestration pattern) this project's own pyramid primitives and
  `run_..._gpu` functions follow.

## Exercises

1. **Plot `crf_curve.csv`.** Overlay the recovered and true curves (after accounting for the reported
   `crf_offset` — see `THEORY.md` "Numerical considerations") and see where the recovery is best (dense
   sample support) and worst (near-clipped code values).
2. **Vary the exposure bracket size.** Rerun with only 2 exposures (edit `kNumExposures`-adjacent
   constants and `scripts/make_synthetic.py`) and watch `dynamic_range_coverage` degrade — how few
   exposures can still beat the best single exposure?
3. **Implement separable Gaussian reduce.** `gaussian_reduce_kernel` does a direct 5x5 (25-tap)
   convolution; replace it with two 1D 5-tap passes (10 taps total) and measure the kernel-time change on
   a larger synthetic image.
4. **Add a true polyphase EXPAND.** `bilinear_expand_kernel` is a simplified stand-in for Burt &
   Adelson's zero-insertion + 4x-kernel EXPAND (see its header) — implement the textbook version and
   check whether `halo_check`'s ratio improves.
5. **Re-add a 3-channel scene and Mertens' saturation term.** Extend `scripts/make_synthetic.py` to RGB,
   add the `std(R,G,B)` weight term back into `mertens_raw_weight_kernel`, and see how much the fused
   result changes.

## Limitations & honesty

- **Grayscale scene, by design** (see [Overview](#overview)) — Mertens' saturation weight term is
  dropped, not faked; a production Mertens implementation on a color camera should restore it.
- **Static-scene teaching scope.** A four-exposure bracket spanning up to 1/8 s per frame takes on the
  order of ~150 ms to capture; this project's synthetic scene is perfectly static across that window by
  construction. A real moving robot or scene introduces inter-frame motion, causing GHOSTING in the
  merged/fused result — neither this project's Debevec-Malik merge nor its Mertens fusion includes motion
  compensation or ghost rejection (both real, standard extensions in production HDR pipelines, and both
  out of scope here — see [System context](#system-context--where-this-sits-in-a-robot) for why real-time
  robots facing this problem generally prefer sensor-level HDR over bracketing in the first place).
- **The CRF is recovered up to an unknown scale.** Debevec-Malik radiance is only defined up to an
  additive constant in log space (equivalently, a multiplicative constant in linear radiance) — a
  well-documented, expected property of the algorithm, not a bug. Tone mapping is invariant to it (both
  Reinhard's key-scaled reduction and local tone mapping's min-max normalize absorb any global scale);
  this project's ground-truth-facing gates correct for it explicitly and say so in their `[info]` lines
  (see `main.cu`'s `crf_offset` computation and `THEORY.md` "The math").
- **Simplified pyramid EXPAND.** `bilinear_expand_kernel` uses plain bilinear upsampling rather than
  Burt & Adelson's textbook zero-insertion + 4x-kernel EXPAND — adequate for this project's didactic goal
  (see Exercise 4) but not spectrally identical to the classic operator.
- **The most extreme edge in the scene is deliberately NOT the halo_check test site.** The shadow/
  concrete boundary (~450x radiance contrast) is severe enough that this project's simplified pyramid
  shows some genuine reconstruction ringing there too, alongside the naive blend's noise-driven scatter —
  `halo_check` instead scans the gentler (3x) painted-line/concrete edge, where the classic
  weight-map-switching halo this gate is meant to teach is unambiguous. See `main.cu`'s `kHaloScanY`
  comment for the full, honest reasoning.
- **Not safety-certified; sim-validated only.** Nothing in this repository is safety-certified
  (CLAUDE.md §1). This project's output is a display/perception image, never a control signal, but the
  same caveat applies repo-wide: any use on real hardware is the owner's decision and responsibility.
