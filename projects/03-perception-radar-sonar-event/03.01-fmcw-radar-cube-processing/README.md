# 03.01 — FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection

**Difficulty:** ★ beginner · **Domain:** 3. Perception — Radar, Sonar, Event & Exotic Sensors

> Catalog bullet (source of truth, verbatim): `★ FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

An FMCW ("frequency-modulated continuous wave") radar sweeps a chirp, mixes the echo with a copy of
itself, and samples the result many times per chirp, many chirps per frame, across several receive
antennas. That raw block of complex samples — the "radar cube" — hides three physical quantities
(range, velocity, angle) inside three different kinds of phase drift, and three FFTs pull them back
out. This project builds the **complete classic pipeline**, end to end, on a synthetic but physically
grounded 77 GHz automotive-band cube: synthesize the raw ADC cube from a committed 6-target scene,
window + FFT it into range bins, window + FFT the chirps into Doppler bins, noncoherently integrate
the antennas into a range-Doppler power map, run **two** CFAR (constant-false-alarm-rate) detectors
— cell-averaging (CA) and ordered-statistic (OS) — over that map, and finally FFT a zero-padded
antenna snapshot per detection into an azimuth estimate. Every stage named in the catalog bullet is
implemented (not just documented): range FFT, Doppler FFT, angle FFT, CA-CFAR, and OS-CFAR.

The demo's target scene is deliberately built to teach one specific, textbook CFAR failure mode: two
of the six targets sit only 1.5 m / 3 range bins and 0.9 m/s / 3 Doppler bins apart, one 6.7x stronger
than the other. **CA-CFAR misses the weak one** (the strong neighbor drags its local training-window
average up, raising its own threshold above the weak return); **OS-CFAR still finds it** (a single
contaminated training cell cannot move a 75th-percentile rank statistic much). Learners see this
happen, not just read about it: `demo/out/detections.csv` and the printed `[info]` lines report both
detectors' actual results side by side, measured, on this exact scene.

## What this computes & why the GPU helps

The pipeline computes, per 262,144-sample cube (256 range samples x 128 chirps x 8 antennas): two
batched 1-D FFTs (range and Doppler, ~50k complex butterflies total), a per-cell noncoherent power
integration, a 2-D CFAR stencil over all 32,768 range-Doppler cells (each reading ~200 training-cell
neighbors), and a small zero-padded FFT per surviving detection.

- **Pattern:** batched, independent transforms (map, dominates the FLOP count) + a stencil with a
  wide, fixed-size neighborhood (CFAR) + a tiny map+reduce per detection (angle peak-finding). Every
  one of the Ns/Nc/detection-count independent transforms/cells is embarrassingly parallel — the
  reason a real radar's 10-20 Hz frame rate needs a GPU (or an FFT-accelerator SoC) at all.
- **cuFFT does the heavy lifting** (batched C2C, in three different layout flavors — advanced-layout
  outer-axis, advanced-layout middle-axis-looped, and plain contiguous — see [`src/kernels.cu`](src/kernels.cu)
  for what each call computes and why); the CPU oracle instead uses a **hand-rolled O(N^2) DFT**,
  affordable only because it runs once, not at frame rate (see [`src/reference_cpu.cpp`](src/reference_cpu.cpp)).
- **Measured reality (RTX 2080 SUPER):** the GPU pipeline (synthesize -> both FFTs -> integrate -> both
  CFAR detectors) runs in ~33 ms; the CPU O(N^2) oracle takes ~440 ms for the same work — a ~13x
  teaching artifact, not a benchmark (single-shot, one machine).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **perception** layer, at the sensor-adjacent end: this project turns raw
  ADC samples directly into detections (range, velocity, angle) — the earliest possible point a radar
  front end hands off to software (SYSTEM_DESIGN §1's "SENSORS -> PERCEPTION" boundary).
- **Upstream inputs:** raw complex baseband ADC samples from the radar's RF front end (mixer + ADC
  bank) — the Ns x Nc x Na cube this project's `synthesize_cube_kernel` stands in for. On real
  hardware this arrives over a dedicated high-speed interface (LVDS/CSI2-class) from the radar SoC,
  not a general bus.
- **Downstream consumers:** a detection list — this project's `Detection` struct (range_m, vel_mps,
  az_deg, power), the radar analogue of SYSTEM_DESIGN §3.6's message-shaped structs — feeding
  **multi-target tracking / sensor fusion** (domain 04; e.g. a future 04.04-style tracker), which
  associates detections across frames into tracked objects with velocity and existence confidence.
- **Rate / latency budget:** automotive/robotics radars run at **10-20 Hz** frame rates with a
  per-frame budget under ~50-100 ms (SYSTEM_DESIGN §1.1's LiDAR/radar-class row) — this project's
  measured 33 ms GPU pipeline fits that budget with headroom; the 440 ms CPU oracle does not, which is
  exactly the point of building the GPU path.
- **Reference robot(s):** the **autonomous-vehicle stack** (SYSTEM_DESIGN §2.5) most directly — radar
  is the only sensor modality on that list that keeps working in rain, fog, and glare, which is why AV
  stacks fuse it with cameras and LiDAR rather than treating it as a redundant copy of either. It also
  appears in the AV composition chain (§4) feeding fusion/tracking exactly as this project's downstream
  consumer does.
- **In production:** a vendor radar SoC's on-chip DSP or accelerator (TI AWR/IWR-class, NXP, Infineon)
  runs this exact range-Doppler-angle-CFAR pipeline in firmware and ships a detection list over
  CAN-FD/Ethernet — see README "Prior art" and `PRACTICE.md` §2-3 for the hardware and integration
  picture.
- **Owning team:** perception (SYSTEM_DESIGN §5.1) — specifically a radar-signal-processing sub-team,
  adjacent to sensor fusion/tracking (who consume this project's output) and to electrical engineering
  (who own the RF front end this project's cube stands in for).

## The algorithm in brief

- **Cube synthesis** — one GPU thread per complex sample, closed-form phasor sum over all targets plus
  deterministic per-sample noise. -> [THEORY.md](THEORY.md) §The math.
- **Range FFT** — Hann-windowed batched cuFFT C2C along the fast-time axis; beat frequency -> range.
  -> THEORY.md §The GPU mapping.
- **Doppler FFT** — Hann-windowed batched cuFFT C2C along the slow-time (chirp) axis; inter-chirp phase
  drift -> velocity. -> THEORY.md §The GPU mapping.
- **Noncoherent antenna integration** — average power across antennas -> the range-Doppler map CFAR runs
  on. -> THEORY.md §The algorithm.
- **CA-CFAR** — cell-averaging constant-false-alarm-rate detection: threshold = alpha x mean(training
  cells). -> THEORY.md §The algorithm.
- **OS-CFAR** — ordered-statistic CFAR: threshold = alpha x (a fixed percentile of the SORTED training
  cells) — robust to a minority of contaminated cells where CA-CFAR is not. -> THEORY.md §The algorithm.
- **Angle FFT** — zero-padded cuFFT C2C across the 8-antenna ULA snapshot at each detection; antenna
  phase step -> azimuth. -> THEORY.md §The math.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/fmcw-radar-cube-processing.sln`](build/fmcw-radar-cube-processing.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/fmcw-radar-cube-processing.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **cuFFT** (`cufft.lib`/`CUDA::cufft`), a CUDA Toolkit
library — always present with any CUDA Toolkit 13.3 install (the repo's baseline requirement), so
there is no separate install step and no fallback path; see the `.vcxproj` and `CMakeLists.txt`
comments for exactly why it is linked and what it computes.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at, including the **range-Doppler image and detections CSV** artifacts.

## Data

The committed sample is **synthetic by construction** (CLAUDE.md §8's default): a radar configuration
record (`data/sample/radar_params.csv` — chirp/antenna parameters, cross-checked against the values
compiled into `src/kernels.cuh`) plus a fixed 6-target ground-truth list
(`data/sample/targets.csv` — range, velocity, azimuth, amplitude). The raw ~2 MB ADC cube itself is
**never written to disk**: both the GPU kernel and the CPU oracle synthesize it, in code, from these
two tiny files plus a fixed noise seed, deterministically. No public dataset applies — real radar cube
recordings are essentially never published raw (see `data/README.md` for why) —
`scripts/download_data.ps1`/`.sh` are honest permanent no-ops. Full provenance, every field's units,
and the design story behind the target list: [`data/README.md`](data/README.md).

## Expected output

Two independent verifications, mirroring the "hand-rolled CPU oracle + ground-truth ladder" pattern
this repo uses throughout:

1. **The §5 GPU-vs-CPU VERIFY gate:** the GPU's range-Doppler power map is compared cell-by-cell
   (all 32,768 cells) against the CPU O(N^2) DFT oracle's; worst relative deviation must be within 5%
   (measured: 0.55%). Catches indexing/layout/window/FFT bugs — any such bug shifts cell values by
   orders of magnitude, not fractions of a percent.
2. **GROUND-TRUTH gates:** every one of the 6 injected targets must be found by OS-CFAR within one
   resolution cell of range/velocity (0.50 m / 0.30 m/s) and 3 degrees of azimuth (measured: all 6
   found, worst errors 0.055 m / 0.137 m/s / 0.95 deg — see the derivation of these bounds in
   THEORY.md); false alarms must stay within a generous bound of 8 (measured: 0, for both detectors);
   and the CA-vs-OS masking comparison must show its documented shape — CA-CFAR misses the close
   pair's weak target, OS-CFAR finds it (measured: exactly that, every run, deterministically).

Artifacts: `demo/out/range_doppler.pgm` (a viewable log-magnitude image of the GPU's range-Doppler map
with OS-CFAR detections marked) and `demo/out/detections.csv` (every detection from both detectors,
matched against ground truth with per-field errors). The canonical stable console lines live in
[`demo/expected_output.txt`](demo/expected_output.txt); exact measured numbers (which vary in the
low digits across GPU architectures — cuFFT's internal algorithm differs sm_75 vs sm_86 vs sm_89) are
printed on `[info]`/`[time]` lines the demo script does not diff.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the single-source contract: the raw-cube memory layout, the
   physical parameters and every resolution/ambiguity formula, the CFAR geometry, and every function
   signature. Read this FIRST — everything else in the project cites it.
2. [`src/main.cu`](src/main.cu) — orchestration: load the scenario, run the GPU pipeline, run the CPU
   oracle, VERIFY, apply the ground-truth gates, write the artifacts.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU pipeline itself (the heart of the project). The single
   most interesting thing to read: the `launch_range_fft` / `launch_doppler_fft` pair's header
   comments, which derive exactly why one is a single cuFFT call and the other must loop a plan over
   the outer axis — the project's main "advanced data layout" lesson.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the O(N^2)-DFT correctness oracle; read it beside
   `kernels.cu` to see the exact same formulas without cuFFT's help.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and why they are copied, not shared.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **TI mmWave SDK / AWR-class radar reference design** — the production-firmware version of exactly
  this pipeline (range/Doppler/CFAR/angle) running on a dedicated DSP; TI's application notes on
  range-Doppler processing and CFAR are the closest real-world analogue of THEORY.md's derivations.
- **NXP S32R / Infineon radar reference designs** — alternative automotive radar SoC vendors with
  their own published signal-processing chains; comparing their block diagrams to this project's
  pipeline is a good exercise in "same physics, different silicon."
- **Richards, "Fundamentals of Radar Signal Processing"** — the standard graduate textbook covering
  matched filtering, Doppler processing, and CFAR theory (CA, GO, SO, OS variants) in full rigor;
  THEORY.md's CFAR derivation is a condensed, didactic version of this book's treatment.
- **Rohling (1983), "Radar CFAR Thresholding in Clutter and Multiple Target Situations"** — the
  original ordered-statistic CFAR paper; the masking scenario this project measures is exactly the
  failure mode Rohling's OS-CFAR was designed to fix.
- **cuFFT documentation (NVIDIA)** — the authoritative reference for `cufftPlanMany`'s advanced-layout
  parameters (istride/idist/ostride/odist), used three different ways in `src/kernels.cu`.
- **04.x multi-target tracking projects (this repo)** — the natural downstream consumer of this
  project's `Detection` list; study the chain README "System context" describes.

## Exercises

1. **Plot the artifact:** open `demo/out/range_doppler.pgm` (any PGM-capable viewer, or Python
   `PIL.Image.open`) and find the 6 bright crosses. Then open `demo/out/detections.csv` and match each
   cross to its row.
2. **Break the window:** remove the Doppler-axis Hann window (`launch_hann_window_doppler`) and rerun.
   Watch the false-alarm count explode — the exact failure this project's own development hit first
   (see THEORY.md "Numerical considerations" for the full story) — and explain why from the rectangular
   window's sidelobe level.
3. **Tune the close pair:** edit `data/sample/targets.csv`'s last two rows to move the pair 2 more
   range bins apart (regenerate is not required — just edit the committed file) and rerun; find the
   separation at which even CA-CFAR starts detecting the weak target, and relate it to the CFAR guard
   band width in `src/kernels.cuh`.
4. **Add a third detector:** implement greatest-of (GO-CFAR) or smallest-of (SO-CFAR) — split the
   training cells into leading/lagging halves and take the max/min of their means — and add it to the
   comparison. THEORY.md "Where this sits in the real world" names the trade each variant makes.
5. **Climb to imaging radar:** increase `kNa` and add a second receive row (a small 2-D array) to
   estimate ELEVATION as well as azimuth — the generalization to modern imaging/4D radar THEORY.md's
   last section describes.

## Limitations & honesty

- **Decoupled range-Doppler model.** This project treats fast-time (range) and slow-time (Doppler)
  phase as independent accumulators. Real FMCW radar has a small "range-Doppler coupling" term (a fast
  target's Doppler shift biases its apparent range within a chirp) that this teaching model ignores —
  THEORY.md "Numerical considerations" quantifies the bias this would introduce and names the
  production fixes (alternating chirp slopes, TDM-MIMO scheduling).
- **No radar range equation.** Target `amp` is an arbitrary, unitless reflection scale, not a
  calibrated RCS/power computed from the 1/R^4 radar range equation — realistic relative ordering
  (near/strong vs. far/weak targets), but not calibrated dBm/dBsm units. See THEORY.md "The problem."
  PGM/CSV power values are correspondingly **relative, uncalibrated units**, not calibrated dBm.
  RCS-derived amplitudes are a natural follow-on exercise.
  Detections.csv power values are `10*log10(power)` on this same uncalibrated scale.
- **CFAR alpha constants are empirically calibrated, not closed-form.** They target a Pfa = 1e-4 against
  a noise-ONLY calibration run; a real, non-homogeneous scene (several targets whose Hann-window
  sidelobes overlap) realizes a somewhat different effective Pfa than the idealized formula predicts —
  exactly the "non-homogeneous clutter" caveat every CFAR textbook states, here made concrete and
  measured rather than asserted (THEORY.md "How we verify correctness").
  This project also found and fixed a genuine noise-generator bug during development — a
  linearly-seeded per-sample xorshift stream produced measurably non-white noise until an
  avalanche-hash mixing step was added; see the `hash32_mix` comment in `src/kernels.cu` for the full
  story, kept in the code as an honest teaching artifact rather than smoothed over.
- **8-element ULA angular resolution is coarse** (~14 degrees full beamwidth at broadside) — realistic
  for a single small array, far below a modern imaging-radar's hundreds of virtual channels (MIMO).
  The zero-padded angle FFT improves peak *localization*, not true resolving power — stated explicitly
  in `src/kernels.cuh` and THEORY.md wherever it matters.
- **Single frame, no tracking.** This project stops at per-frame detections; associating detections
  across frames into tracks (velocity smoothing, existence confidence, false-alarm rejection over
  time) is domain 04's job, named in "System context" as the downstream consumer.
- **Not safety-relevant as built.** This project's output is a detection list, not a control command —
  the sim-validated-only caveat (CLAUDE.md §1) applies in spirit (nothing here is a certified sensor
  chain) but not in the "commands real hardware" sense; no additional caveat is needed beyond the
  repo-wide one.
