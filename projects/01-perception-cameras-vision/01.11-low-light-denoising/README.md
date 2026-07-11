# 01.11 — Low-light denoising (bilateral, non-local means, fast BM3D variant)

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Low-light denoising (bilateral, non-local means, fast BM3D variant)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A robot working the night shift sees photon-starved images: a warehouse AMR's camera at 2 a.m., an
inspection arm under a dim enclosure light, a drone at dusk. This project builds three denoisers that
recover a usable image from a frame where the brightest pixel carries an **expected signal of only 40
photoelectrons** — a deliberately extreme, honestly-labeled low-light operating point that makes every
committed sample visibly, heavily noisy. All three named methods in the catalog bullet are implemented
in full: **bilateral filtering** (built twice — a naive global-memory stencil and a shared-memory
TILED version, verified bit-identical, so the tiling speedup is a measured fact, not a claim),
**non-local means (NLM)** (patch-similarity weighted averaging — the expensive one), and
**BM3D-lite** (block-match 16 similar patches, a joint 2-D DCT + 1-D Haar transform, hard-threshold,
invert, sparsity-weighted aggregate — the hard-thresholding first stage of full BM3D; the second,
collaborative-Wiener stage is documented, not implemented — see [Limitations](#limitations--honesty)).
A fifth kernel, a plain **Gaussian blur** using bilateral's own spatial term with the range term
removed, is a *designed negative control*: it must reduce noise in flat regions and it must
**fail** to preserve the test scene's high-contrast edge — proving the edge-preservation gate can
actually tell a denoiser from something that merely blurs (measured: it retains only 16% of the clean
edge's gradient magnitude, against bilateral/NLM/BM3D-lite's 88-98%).

## What this computes & why the GPU helps

Three GPU patterns, one demo:

- **Bilateral (STENCIL):** every output pixel is an independent 9x9 joint spatial x range Gaussian
  weighted average — the canonical *map-of-a-local-reduction* pattern, one thread per pixel. The
  SHARED-MEMORY TILED version is the same stencil with cooperative halo loading — measured **~9-20x**
  faster than the naive version on the reference machine (run-to-run variance is real; both are
  proven bit-identical, so the speedup is purely a memory-traffic effect, not a numerics change).
- **NLM (SEARCH):** every output pixel searches a 13x13 window of candidate 5x5 patches (169
  candidates x 25-pixel patch comparisons = 4,225 squared differences per pixel) — the most
  arithmetically expensive of the three (measured ~200 ms on one CPU core vs ~0.5-0.9 ms on the GPU
  for this 200x150 frame).
- **BM3D-lite (BATCHED BLOCK PROCESSING):** one thread per *reference group* (not per pixel!) —
  1,813 groups for this frame, each doing its own block search, joint transform, threshold, and
  scatter-accumulate via `atomicAdd`. The unusual "one thread = one whole group's work" mapping is
  the same *shape* as project 08.01's "one thread = one whole rollout", applied to a very different
  payload (THEORY.md "The GPU mapping" draws the comparison out).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **perception / camera ISP** layer — specifically the noise-reduction (NR)
  block that sits *inside* a production image signal processor, downstream of demosaic and upstream
  of anything that assumes a clean image.
- **Upstream inputs:** a debayered, radiometrically-calibrated frame — project **01.01**'s full GPU
  image pipeline (debayer -> undistort -> rectify -> resize) produces exactly this shape of output,
  and project **01.09**'s photometric/vignetting calibration is what characterizes and removes the
  *spatial* (per-pixel fixed-pattern: DSNU/PRNU/vignette) part of a sensor's noise story that this
  project deliberately does NOT model — this project isolates the remaining *temporal*,
  signal-dependent shot+read noise 01.09 calls out as a distinct problem. In a real ISP the order is:
  01.09-style flat-field/dark-frame correction FIRST (removes the fixed spatial pattern), THEN this
  project's temporal denoising (removes the frame-to-frame random noise) — composing them the other
  way around would try to denoise noise that is still hiding a spatial bias.
- **Downstream consumers:** anything that assumes a roughly clean image — most concretely project
  **01.04**'s feature pipeline (FAST/Harris corner detection is a local-gradient method; heavy shot
  noise manufactures spurious high-gradient pixels and kills real corners under a fixed detection
  threshold tuned in daylight — a noisy frame is not just "the same image with worse SNR", it
  measurably breaks a daylight-tuned pipeline), plus any downstream low-light perception (night-time
  obstacle detection, visual odometry front-ends).
- **Rate / latency budget:** a real camera pipeline runs at 30-60 Hz (SYSTEM_DESIGN item 1); this
  project's teaching-scale 200x150 frame is not that camera. Honest extrapolation: NLM measured
  ~0.5-0.9 ms and BM3D-lite ~8.2 ms of GPU kernel time on 30,000 pixels; a 1920x1080 frame has ~69x
  more pixels, and NLM/BM3D-lite are *not* linear-only in pixel count once occupancy and cache
  behavior are accounted for — even optimistic linear scaling alone already lands NLM at ~35-60 ms and
  BM3D-lite at ~550+ ms per frame, both well past a 33 ms (30 Hz) budget. This is the honest, measured
  answer to "could you just run BM3D on every frame": no, not this un-optimized teaching kernel, not at
  camera resolution. **What production stacks do instead:** fixed-function ISP hardware NR blocks
  (a handful of cycles per pixel, silicon-implemented bilateral/NLM-family filters) or a single-pass
  learned denoiser (a small CNN, one forward pass per frame, trained self-supervised — Noise2Noise-
  class methods — see [Prior art](#prior-art--further-reading)).
- **Reference robot(s):** the **warehouse AMR** running a night shift (SYSTEM_DESIGN's reference AMR:
  02/04/05/23/06/08/25/31/32 — this project's output would feed 01.04-style features into 05/23) and
  an **inspection** platform working under a dim enclosure light (a fixed camera cell, closer to the
  manipulator work-cell reference robot's perception stack: 01/19/09/06/07/08/21/24).
- **In production:** most shipping cameras never run bilateral/NLM/BM3D at all — the ISP's
  fixed-function NR block does a cheaper, hardware-pipelined equivalent, and increasingly a small
  learned denoiser runs instead of (or after) it. This project's three methods are exactly the
  *classical* family that hardware NR blocks and learned denoisers both grew out of and are graded
  against in the literature.
- **Owning team:** perception / ISP (camera pipeline engineers, image-quality engineers) — the same
  team that owns 01.01's debayer/rectify pipeline and 01.09's calibration; the boundary with the ML
  team is exactly the classical-vs-learned-denoiser line above.

## The algorithm in brief

- **Bilateral filtering** — a 9x9 joint spatial x range Gaussian: weight a neighbor by *both* how
  close it is (space) and how similar its intensity is (range/photometric) — the range term is what
  lets it average away noise while stopping at edges. Built naive AND shared-memory tiled, verified
  bit-identical. → [THEORY.md](THEORY.md) §The math, §The GPU mapping.
- **Non-local means (NLM)** — the self-similarity prior: instead of "nearby pixels are probably
  similar" (bilateral's assumption), NLM asks "similar-LOOKING patches, wherever they are in a 13x13
  neighborhood, are probably the same underlying signal" and averages by patch-similarity weight.
  → THEORY §The math (the self-similarity argument), §The GPU mapping (why it's the expensive one).
- **BM3D-lite** — block-match 16 similar 8x8 patches into a 3-D stack, transform (2-D DCT per patch +
  1-D Haar across the stack — both orthonormal, hence noise-variance-preserving), HARD-THRESHOLD every
  coefficient at `2.7*sigma`, invert, and aggregate with a sparsity-based weight. The first stage of
  real BM3D; the second (collaborative Wiener) stage is documented, not built — the "-lite" in the
  name. → THEORY §The math (collaborative filtering), §Numerical considerations (why the threshold is
  scale-correct), §Where this sits in the real world (the full two-stage pipeline).
- **Gaussian-blur negative control** — bilateral's spatial term alone, no range awareness at all: the
  controlled experiment that proves edge_preservation is measuring something real (task brief).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/low-light-denoising.sln`](build/low-light-denoising.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/low-light-denoising.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
8x8 DCT is a hand-rolled fixed-basis matrix multiply (no cuFFT; THEORY.md "The GPU mapping" explains
why a transform this small has no meaningful library-call speedup).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including the
artifact list (denoised PGMs + residual heatmaps + `gates_metrics.csv`) to open and look at.

## Data

The committed sample is fully synthetic (CLAUDE.md §8): a hand-designed 200x150 test scene —
3-octave hashed multi-scale texture, three flat patches (dark/mid/bright), one high-contrast step
edge, one fine-detail stripe ruling — rendered clean (`clean.pgm`, ground truth) and through this
project's Poisson-shot + Gaussian-read noise model (`noisy.pgm`, the only frame any denoiser sees).
Noise is drawn via **exact Poisson sampling** (Knuth's inversion algorithm, xorshift32-driven) rather
than the more common Gaussian shot-noise approximation — chosen because the darkest committed patch
sits at an expected signal of only ~4.4 electrons, exactly where a Gaussian approximation visibly
disagrees with the true discrete, non-negative Poisson distribution. Regenerate with
`python scripts/make_synthetic.py` (fixed seed 42, byte-identical every run); no public dataset applies
(`scripts/download_data.ps1` is an honest no-op — see `data/README.md`).

## Expected output

Fifteen stable lines — banner, `PROBLEM:`, `DATA:`, five `VERIFY(method):`, five `GATE name:`,
`ARTIFACT:`, `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Every VERIFY and GATE below is a real,
measured number from the reference machine (RTX 2080 SUPER, sm_75):

- **VERIFY (GPU vs CPU, per method):** bilateral max|gpu-cpu|=0.0001 DN (tol 0.05), gaussian
  0.0000-0.0001 DN (tol 0.02), NLM 0.0001 DN (tol 0.15), BM3D-lite 0.0002 DN (tol 3.00 — the loosest,
  because its GPU aggregation is `atomicAdd`, an unordered float summation, against a fixed-order
  double-accumulated CPU oracle — see THEORY.md). The bilateral naive-vs-tiled check is
  **bit-identical**: max diff 0.000000 DN, exactly 0, by construction (kernels.cu's header proves why).
- **GATE psnr_improvement (whole image, need >= 2.0 dB over noisy=18.75 dB):** bilateral 25.06 dB
  (+6.31), gaussian 21.06 dB (+2.31 — even a naive blur measurably improves PSNR; that is precisely
  why edge_preservation is a *separate*, necessary gate), NLM 31.48 dB (+12.73), BM3D-lite 30.99 dB
  (+12.24).
- **GATE edge_preservation (need >= 55% of the clean step's 176.0 DN gradient):** bilateral 98%, NLM
  89%, BM3D-lite 88%, **Gaussian baseline 16% — FAILS as designed** (the negative control's asserted
  failure; main.cu prints a warning if it ever unexpectedly passes).
- **GATE flat_noise_floor (need <= 55% of noisy std, worst case across 3 flat patches):** bilateral
  45% (bright patch), gaussian 18%, NLM 13%, BM3D-lite 13%.
- **GATE method_ordering (texture-ROI PSNR, reported honestly):** noisy 18.88 dB, bilateral 26.86 dB,
  NLM 37.41 dB, BM3D-lite 36.30 dB — **NLM slightly beats BM3D-lite here**, which *differs* from the
  typical BM3D-lite >= NLM >= bilateral expectation; see [Limitations](#limitations--honesty) for the
  honest investigation.
- **GATE noise_model_sanity (measured/predicted std ratio, need in [0.80, 1.20]):** dark 1.025, mid
  0.922, bright 0.886 — the synthetic generator's noise matches the analytic Poisson+read-noise
  prediction this project's own code independently re-derives.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the whole project's contract in one file: geometry, the
   noise model (and why it's shared, single-sourced HD code), the scene-layout rectangles every gate
   reads, and every method's parameters. Read this FIRST.
2. [`src/kernels.cu`](src/kernels.cu) — the five GPU kernels. Start with `bilateral_naive_kernel`
   (the simplest stencil), then `bilateral_tiled_kernel` right after it (the tiling lesson, side by
   side), then `nlm_kernel`, then the BM3D-lite pair (`bm3d_group_kernel` is the single most
   interesting kernel in the project — one thread doing block-matching, a 2-D DCT, a 1-D Haar, hard
   thresholding, and atomic scatter-accumulation).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the four independent CPU oracles; read
   `bm3d_lite_cpu` beside `bm3d_group_kernel`/`bm3d_finalize_kernel` to see the same algorithm,
   independently retyped, with a fixed-order double accumulator standing in for the GPU's atomics.
4. [`src/main.cu`](src/main.cu) — orchestration: load data, run all five GPU + four CPU paths, VERIFY,
   five independent gates, ten artifacts.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the noise generator (exact Poisson via
   Knuth inversion) and the hashed multi-scale scene texture.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `paths.h` (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Tomasi & Manduchi (1998), "Bilateral Filtering for Gray and Color Images"** — the original
  bilateral filter paper; the joint spatial x range weighting this project implements verbatim.
- **Buades, Coll & Morel (2005), "A Non-Local Algorithm for Image Denoising"** — the NLM paper; the
  self-similarity argument THEORY.md walks through.
- **Dabov, Foi, Katkovnik & Egiazarian (2007), "Image Denoising by Sparse 3-D Transform-Domain
  Collaborative Filtering"** — the BM3D paper; this project implements exactly its first
  (hard-threshold) stage and documents the second (collaborative Wiener) stage it omits.
- **OpenCV's `cv::bilateralFilter` / `cv::fastNlMeansDenoising` / `cv::xphoto::bm3dDenoising`** — the
  production-grade, heavily-optimized implementations of all three methods; compare their integral-
  image/separable-kernel tricks against this project's teaching-straightforward versions.
- **Lehtinen et al. (2018), "Noise2Noise"** and the self-supervised denoising literature it spawned
  (Noise2Void, Neighbor2Neighbor) — the learned-denoiser family named in "System context" as what a
  real-time ISP increasingly runs instead of BM3D.
- **Mildenhall et al. (2018), "Burst Denoising with Kernel Prediction Networks"** — a concrete example
  of the "learned, real-time, low-light" answer to this project's rate-honesty discussion.

## Exercises

1. **Plot the residual heatmaps:** open all four `residual_*.pgm` artifacts side by side. Find the
   fine-detail ruling in each — every method visibly loses it (the spatial support of every filter here
   is coarser than the 4-px stripe period); describe WHY in terms of each method's own mechanism.
2. **Break bilateral's range term:** set `kBilateralSigmaRange` to a very large value (rebuild) and
   confirm it converges toward the Gaussian baseline's edge-destroying behavior — you have just
   proven, empirically, what the range term is *for*.
3. **Investigate the method_ordering surprise:** the measured order is NLM >= BM3D-lite >= bilateral,
   not the typical BM3D-lite >= NLM >= bilateral. Vary `kBm3dThreshLambda` and `kNlmH` and see whether
   a different, still-honest parameter choice changes the ranking — then read
   [Limitations](#limitations--honesty)'s discussion of the likely cause before you conclude anything.
4. **Add the missing search-window integral image (NLM):** THEORY.md documents, but does not build,
   the classic O(1)-per-candidate patch-SSD acceleration. Implement it and measure the speedup.
5. **Climb toward full BM3D:** implement the second (collaborative Wiener) stage, using the
   hard-threshold result as the oracle spectrum real BM3D's Wiener step needs — THEORY.md "Where this
   sits in the real world" sketches exactly what is required.

## Limitations & honesty

- **BM3D-lite is genuinely a reduced scope**, named honestly: hard-thresholding first stage only (no
  collaborative Wiener second stage), a Kaiser-window-free uniform-within-patch aggregation, and — the
  numerically significant one — **ONE global assumed sigma** (`kBm3dAssumedSigmaDn`, calibrated at
  mid-gray) for the hard threshold, even though this project's own noise is signal-DEPENDENT
  (heteroskedastic). Real low-light BM3D pipelines apply a variance-stabilizing (Anscombe) transform
  before BM3D and invert it after — documented in THEORY.md "Where this sits in the real world", not
  implemented here.
- **The method_ordering surprise is reported honestly, not smoothed over:** measured texture-ROI PSNR
  has NLM (37.41 dB) slightly ahead of BM3D-lite (36.30 dB), differing from the typical BM3D-lite >=
  NLM expectation from the literature. The likely cause: this project's synthetic background is a
  SMOOTH, continuously-correlated multi-octave hashed texture (bilinear-interpolated value noise) —
  exactly the regime where NLM's soft, continuous patch-similarity weighting has an advantage over
  BM3D-lite's HARD top-16 block selection plus a single hard threshold (which can zero out genuine
  low-contrast texture detail that falls below the threshold). A full two-stage BM3D (Wiener
  refinement) would likely narrow or reverse this gap — see Exercise 3 and THEORY.md.
- **BM3D-lite's kernel is not performance-optimized:** `bm3d_group_kernel`'s local 16x8x8 patch stack
  (1,024 floats/thread) far exceeds a healthy register budget and spills to local memory — correct,
  but not how a production BM3D-GPU implementation would shape this kernel (one BLOCK per group,
  shared-memory cooperation, is the fix — named in `kernels.cu`'s header, not built).
- **Rate honesty (README "System context"):** every timing here is measured on a 200x150 TEACHING
  frame; extrapolated to a real 1920x1080 camera, NLM and especially BM3D-lite blow well past a 30 Hz
  budget on this un-optimized code — the honest reason production ISPs use fixed-function hardware NR
  or a single-pass learned denoiser instead (see "System context" for the numbers).
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **The noise model omits fixed-pattern (spatial) noise on purpose** — DSNU/PRNU/vignetting is 01.09's
  subject, cited by name; composing the two projects in the right order is discussed in "System
  context".
- **Sim-validated only (CLAUDE.md §1):** every gate here grades pixels against synthetic ground truth;
  no real-camera claim is made, and this project's output is never used to command motion of real
  hardware — if a denoised frame from code like this ever fed a perception stack driving a real robot,
  the full testing ladder (`PRACTICE.md` §3) applies before trusting it.
