# 01.20 — Time-of-flight raw processing: phase unwrapping, flying-pixel removal

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Time-of-flight raw processing: phase unwrapping, flying-pixel removal`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A continuous-wave indirect-ToF (iToF) camera never times an individual light pulse: it modulates its
IR illuminator continuously and, per pixel, cross-correlates the returning light against four
internally shifted reference copies of the same waveform ("4-tap" demodulation) — the round-trip
distance shows up as a PHASE shift, recovered with an `atan2`. This project implements the two
problems every raw iToF pipeline must solve before that phase becomes a usable point cloud: a single
modulation frequency's phase only resolves distance MODULO an ambiguity range (**phase unwrapping** —
solved here with a second, lower frequency and a CRT-style integer-consistency search), and a real
sensor PIXEL integrates light from more than one surface at silhouette edges, producing points that
hang in space between the real surfaces (**flying pixels** — detected here via a local depth-
discontinuity test plus an amplitude-suppression test, both derived from the physical phasor-mixing
mechanism that causes them). A synthetic room-scale scene (a tilted wall, a sphere, and a box's raised
top face, rendered 1.5-5.6 m away with realistic sensor noise and a deliberately low-reflectivity
cohort) is rendered through a physically-derived forward model that genuinely produces both effects,
so the demo MEASURES — not just claims — that naive single-frequency ranging aliases on the far wall,
that dual-frequency unwrapping recovers metric depth (with an honestly-measured, nonzero wrap-decision
failure rate), and that the flying-pixel detector catches a real majority of the scene's designed
mixed-return edge pixels with zero false positives on clean interior surfaces. Every stage runs on the
GPU and is checked against an independent CPU oracle; nine further gates check the decoded/
reconstructed results against the synthetic ground truth.

Everything named in the catalog bullet is implemented: phase unwrapping (dual-frequency) and
flying-pixel removal. Nothing here is a documented-only stub.

## What this computes & why the GPU helps

Six stages. Five are pure per-pixel **MAP**s (the output of pixel `i` depends only on the inputs at
pixel `i`); the sixth is this project's one **STENCIL** — a genuinely new GPU-mapping idea beyond its
direct kin, project 01.19 (which is a pure map top to bottom):

- **Extract phase/amplitude** — per camera pixel, per frequency, combine 4 correlation taps via
  `atan2`/`sqrt` into a wrapped phase and a modulation-amplitude confidence signal (the SAME arithmetic
  01.19's phase-shift decode uses, applied to a temporal correlator instead of a spatial fringe
  pattern).
- **Single-frequency depth** — per camera pixel, the naive (and, beyond the ambiguity range, WRONG)
  depth from one frequency's phase alone — the aliasing this project measures directly.
- **Dual-frequency unwrap** — per camera pixel, a small (<=3-candidate) integer-consistency search
  that finds which wrap of the fine channel agrees with the coarse channel's unambiguous estimate.
- **Confidence mask** — per camera pixel, an amplitude-floor threshold.
- **Flying-pixel detect** — per camera pixel, THIS PROJECT'S STENCIL kernel: gather the 3x3
  neighborhood's depths/amplitudes and apply two physically-motivated tests (a depth-discontinuity test
  and an amplitude-suppression test) to flag mixed-return pixels.
- **Back-projection** — per camera pixel, pinhole back-projection to a metric `(x,y,z)` point.

Because five of six stages are zero-cross-pixel-dependency MAPs, the natural mapping is one thread per
pixel with no shared memory and no atomics anywhere (same argument as 01.19). The one exception —
`flying_pixel_detect_kernel` — is still race-free (every thread only ever WRITES its own output pixel;
it merely READS its neighbors), so it needs no atomics either, but it is the first kernel in this
project's pipeline whose output genuinely depends on more than one pixel's input: a genuine stencil,
not a map. See [`THEORY.md`](THEORY.md) "The GPU mapping" for the full argument, including why this
project's kernels are bandwidth-bound, not compute-bound, and honestly reports a modest measured
speed-up at this problem's small size (19,200 pixels — a toy compared to a real 1-2 MP iToF sensor).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** `Sensors → Perception`, the raw-sensor-to-point-cloud reconstruction step that
  feeds everything downstream of it (SYSTEM_DESIGN.md §1) — the SAME stack position 01.19 occupies,
  from a different physical sensing modality.
- **Upstream inputs:** the raw 4-tap (or, with unwrapping, 8-tap) correlation frame stack from an iToF
  sensor (`Image`-shaped messages, one per tap, 8-16 bit — SYSTEM_DESIGN.md §3.6), plus a one-time
  **calibration** (camera intrinsics from a project like 01.16, and the sensor's own per-pixel fixed-
  pattern-noise / "wiggling error" correction — PRACTICE.md §1).
- **Downstream consumers:** a `PointCloud` (SYSTEM_DESIGN.md §3.6) in the camera frame — consumed by
  obstacle avoidance and local costmapping (23.xx navigation stack), near-field manipulation sensing
  (19.xx grasp planning, where a wrist- or hand-mounted iToF module sees what a farther-mounted stereo
  rig cannot), and gesture/proximity sensing for HRI (21.xx).
- **Rate / latency budget:** a real iToF sensor streams continuously — 4 taps (or 8, dual-frequency)
  captured back-to-back at the sensor's own frame rate, typically netting **15-30 Hz** depth frames
  (SYSTEM_DESIGN.md §1's ~30-60 Hz camera row, at the lower end because of the multi-tap capture
  burden) — a fundamentally different rate regime from 01.19's structured light (which spends a whole
  multi-frame BURST per static scan). This project's own measured full-pipeline compute time
  (`[time]` line, printed live) sits comfortably under a millisecond at this problem's 19,200-pixel
  size — the real bottleneck for frame rate is the SENSOR's own multi-tap capture and readout, not this
  project's decode math (THEORY.md "The GPU mapping" makes the bandwidth case).
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN.md §2.1 — a compact, forward-facing iToF
  module is a common cheap near-field obstacle sensor) and the **6-DoF manipulator work cell**
  (SYSTEM_DESIGN.md §2.2 — a wrist-mounted iToF module for close-range grasp-approach sensing where a
  longer-baseline stereo or structured-light rig cannot focus).
- **In production:** Kinect-v2-class / Azure-Kinect-class consumer iToF, PMD Technologies / Infineon
  REAL3 and Melexis MLX75027-class industrial/automotive iToF modules — see README "Prior art" for
  what to study and PRACTICE.md for the hardware/BOM story.
- **Owning team:** perception / 3-D sensing, inside the Perception org (SYSTEM_DESIGN.md §5.1: "01,
  02, 03, 20"), the SAME team that owns 01.19 — this project and 01.19 are the two active-triangulation
  and active-timing 3-D sensing options that same team evaluates against each other for a given robot's
  range/precision/compactness needs (README "Prior art" makes the contrast explicit).

## The algorithm in brief

- **4-tap correlation demodulation** — per pixel, per frequency, `atan2`/`sqrt` on differenced
  correlation taps recovers a wrapped phase and a modulation-amplitude confidence signal, with the
  ambient/DC term cancelling exactly ([`THEORY.md`](THEORY.md#the-math)).
- **Single-frequency aliasing (why one frequency is not enough)** — a single CW frequency's phase only
  resolves distance modulo `c/(2f)`; this project's far wall is placed specifically beyond that range
  so the aliasing is real and measured, not asserted ([`THEORY.md`](THEORY.md#the-problem--physics--engineering-first)).
- **Dual-frequency phase unwrapping (the CRT-style consistency search)** — a coarse, low-frequency
  channel whose own ambiguity range covers the whole scene resolves WHICH cycle a fine, high-frequency
  channel's wrapped phase belongs to, by searching the small set of integer wrap hypotheses and keeping
  the one where both frequencies agree ([`THEORY.md`](THEORY.md#the-math)).
- **Flying-pixel detection via phasor mixing** — a pixel straddling a depth discontinuity integrates
  the AREA-WEIGHTED SUM of two surfaces' correlation phasors, decoding to a phase (and depth) belonging
  to NEITHER surface; detected here via a local depth-discontinuity test plus an amplitude-suppression
  test, both derived from — not merely correlated with — this physical mechanism
  ([`THEORY.md`](THEORY.md#the-math)).
- **Pinhole back-projection** — per pixel, the final unwrapped, flying-pixel-filtered depth becomes a
  metric `(x,y,z)` point via the same pixel-center camera-ray convention as 01.17/01.19.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/time-of-flight-raw-processing.sln`](build/time-of-flight-raw-processing.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/time-of-flight-raw-processing.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only
(no cuBLAS/cuFFT/Thrust; the small plane/sphere normal-equations solves in the reconstruction gates
are hand-rolled Gaussian elimination on the host, see `src/main.cu`).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

100% synthetic — a ray-cast, phasor-mixing-rendered scene (tilted wall + sphere + box) rendered under
2 modulation frequencies x 4 taps with realistic sensor noise, plus exact ground truth (metric depth,
surface identity, and an independently-computed flying-pixel label) that no photograph or vendor SDK
could provide. Regenerate with `python scripts/make_synthetic.py --seed 42`. Full field documentation,
size accounting, checksums, and the "how the sample was tuned" measurement log:
[`data/README.md`](data/README.md).

## Expected output

The demo runs all **6 pipeline stages** on the GPU and an independent CPU oracle (as two fully separate
end-to-end cascades) and requires exact or near-exact agreement (`VERIFY:` lines) — measured on the
committed sample, the worst GPU-vs-CPU phase disagreement is **4.8x10^-7 rad** (float rounding, far
inside the `1e-3` rad tolerance) and every stage's flag/wrap-count decisions agree with zero
mismatches. It then runs **9 independent gates** against the synthetic ground truth (`GATE:` lines):
freq1 phase extraction matches the analytic truth within a **0.029** rad mean error on clean pixels;
adding 35 counts of ambient IR to every tap changes the decoded phase by **exactly 0** (the offset-
cancellation algebra); the naive single-frequency depth shows a gross (>= half the ambiguity range)
error on **100%** of the far wall's background pixels (the designed aliasing, genuinely reproduced,
not asserted); dual-frequency unwrapping recovers metric depth to a **12.9 mm** mean error with the
correct integer wrap count on **98.1%** of confident pixels (a small, honestly-measured wrap-decision
failure rate — THEORY.md derives why it is never exactly zero); the flying-pixel detector catches
**62%** of the scene's 108 designed mixed-return edge pixels with **zero** false positives on clean
interior surfaces; the reconstructed wall fits its truth within **14.1 mm** RMS, the sphere's radius
within **3.3%**, and the box step's height within **0.01%**; and the amplitude mask rejects **100%**
of the deliberately low-reflectivity dark cohort with zero survivors. One further diagnostic
(`noise_scaling`, not gated) shows the measured depth-error standard deviation shrinking from
**18.7 mm** to **13.7 mm** as amplitude rises from the `[35,50)` to the `[50,255)` counts bucket —
confirming THEORY.md's derived `sigma_Z ~ 1/B` law on this exact sample. The canonical stable lines
live in [`demo/expected_output.txt`](demo/expected_output.txt); every measured number above is
reproduced live as an `[info]` line (not diffed — see `demo/README.md`) on each run.

Nine labeled artifacts land in `demo/out/` every run: two sample tap frames, the single-frequency
wrapped-depth map (the aliasing visual), the dual-frequency unwrapped depth map, the flying-pixel mask,
the point cloud (CSV), a before/after pair of orthographic profile renders (PPM — flying pixels visibly
hang in space in the "before" render and are gone in the "after"), and a full-precision gate-metrics
CSV.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the sensor model, frequency/tap parameters, and the whole
   shared contract every other file uses. Read this FIRST.
2. [`src/main.cu`](src/main.cu) — orchestration: load the 8-frame tap stack + ground truth, run all six
   stages as two independent GPU/CPU cascades, the nine gates, the artifacts.
3. [`src/kernels.cu`](src/kernels.cu) — the six GPU kernels (the heart of the project; Stage 5 is the
   one to read closely — the project's only stencil).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independently-written CPU twin of every
   stage, plus the full derivations in comments.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Lange, "3D Time-of-Flight Distance Measurement with Custom Solid-State Image Sensors"** (PhD
  thesis, 2000) — the foundational derivation of the 4-tap CW-ToF correlation principle this project
  implements.
- **Microsoft Azure Kinect / Kinect for Windows v2 SDK documentation** — a real production iToF SDK's
  depth-mode options, exposure/frequency configuration, and its own flying-pixel / edge-confidence
  filtering — compare its filtering knobs to this project's `kFlyingDepthJumpM`/`kFlyingAmplitudeRatio`.
- **Godbaz, Cree & Dorrington, "Understanding and ameliorating non-linear phase and amplitude
  responses in AMCW lidar"** (2012) and the broader multi-frequency-unwrapping literature — the
  standard references for extending this project's 2-frequency search to 3+ frequencies for robustness.
- **Whyte, Streeter, Cree & Dorrington, "Application of lidar techniques to time-of-flight range
  imaging"** — a survey covering flying-pixel/mixed-pixel detection approaches, including the
  amplitude- and edge-based filters this project's Stage 5 draws on.
- **PCL (Point Cloud Library)** — what a real pipeline would do with this project's output point cloud
  next (filtering, registration, meshing) — same downstream story as 01.19.
- **01.02 (this repo, passive stereo)**, **01.19 (this repo, structured light)**, and **03.01 (this
  repo, FMCW radar)** — the competing/complementary 3-D sensing modalities named by contrast throughout
  this project's docs.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Add a third frequency.** Extend `kernels.cuh`'s frequency set to three CW channels and generalize
   `dual_freq_unwrap_kernel`'s search to a 3-way consistency search — measure whether the wrap-count
   correctness rate (`unwrap_recovery` gate) improves at the SAME noise level.
2. **Tighten or loosen the amplitude floor.** Sweep `kDefaultAmplitudeFloor` and watch the
   `dark_cohort` and `phase_extraction` gates trade off (see `data/README.md` "How the sample was
   tuned" for the measurements that picked 18).
3. **Shared-memory tiling for the stencil.** Implement a tiled version of
   `flying_pixel_detect_kernel` (load each block's tile plus a 1-pixel halo into `__shared__` memory)
   and PROFILE whether it actually beats the current global-memory-only version at this problem size
   (THEORY.md "The GPU mapping" predicts it should not, yet, at 19,200 pixels — verify that prediction).
4. **Sweep the flying-pixel thresholds.** Reproduce the precision/recall frontier `data/README.md`
   records for `kFlyingAmplitudeRatio` and plot it — then try a THIRD criterion (e.g. cross-frequency
   amplitude-ratio consistency) and see whether it moves the frontier.
5. **A second, oblique edge cohort.** Add a grazing-incidence surface to `make_synthetic.py` (foreshortened,
   low effective amplitude even without any albedo change) and check whether the SAME amplitude floor
   that correctly rejects the dark cohort also — correctly or incorrectly — rejects this optically
   different but physically legitimate low-signal case (THEORY.md "Numerical considerations" previews
   why it should behave similarly).

## Limitations & honesty

- **Single-scene, static-scene demo.** Every depth estimate here needs 8 sequentially captured tap
  frames (4 per frequency); this project's synthetic sample is a single static instant — motion
  between taps (a real, honest cost of multi-tap CW ToF) is named in THEORY.md but not simulated.
- **Flat radiometric model.** `scripts/make_synthetic.py` models signal amplitude as `albedo * GAIN`
  with no `1/Z^2` illumination/collection falloff (the SAME simplification 01.19 makes for its own
  radiometric model) — a real iToF camera's amplitude (and hence depth precision) genuinely degrades
  with range even on a uniform-albedo surface; this project's noise-vs-amplitude relationship
  (`noise_scaling`) is real and derived, but the AMPLITUDE-vs-range relationship is simplified away.
- **Two-surface, single-bounce flying-pixel model only.** This project's forward model and detector
  both target the SPATIAL two-surface-per-pixel mixing case (a silhouette edge). General MULTIPATH
  (a single ray receiving light via multiple bounces, e.g. a concave corner or a shiny floor) is a
  deeper, related failure mode named honestly in THEORY.md "Where this sits in the real world" as a
  documented-only extension, not implemented here.
- **No public dataset used.** `data/README.md` "Why no public dataset" explains why: available public
  iToF corpora ship already vendor-decoded depth, not the raw 4-tap correlation frames this project's
  math operates on, and none carries the independent flying-pixel ground truth this project's
  `flying_pixel` gate needs.
- **Camera calibration is assumed, not performed.** This project consumes fixed, known camera
  intrinsics (`kernels.cuh`) exactly as a project like 01.16 would produce them — no calibration
  routine, and no per-pixel fixed-pattern-noise / "wiggling error" correction (PRACTICE.md §1), is
  implemented here.
- **Not safety-certified; sim-validated only.** This project computes a point cloud from synthetic
  data. If used to steer a real robot's motion (e.g. near-field obstacle avoidance or grasp approach),
  that downstream motion is the owner's decision and responsibility — see PRACTICE.md §3 for the
  hardware-testing ladder every real deployment should climb.
