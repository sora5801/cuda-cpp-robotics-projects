# 01.10 — Rolling-shutter correction using IMU rates

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Rolling-shutter correction using IMU rates`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A CMOS "rolling shutter" (RS) camera does not expose every row of a frame at the same instant — it
scans row by row, and this project's synthetic sensor takes 25 ms to read out a whole frame. If the
camera rotates during that 25 ms (a drone in flight, a handheld scan), each row sees the world through
a slightly different orientation, and straight things in the world come out sheared, wobbled, or
"jello"-warped in the captured image. This project undoes exactly that: given nothing but 200 Hz gyro
(angular-rate) samples, it reconstructs the image an ideal *global-shutter* (GS) camera would have taken
at the frame's reference instant, entirely from the rolling-shutter capture and the gyro trace — no
translation is modeled (pure rotation only, the classic drone/handheld case).

The demo builds one synthetic scene with a *known* geometry (a hashed, non-repeating background texture
plus one bright, exactly-vertical marker line), captures it twice — once as an ideal global-shutter
reference and once as the row-sequential rolling-shutter frame a real sensor would have produced — and
supplies two gyro traces for the same true motion: a **clean** one and a **degraded** one (constant bias
+ noise, the realistic uncalibrated-sensor case). The GPU kernel corrects the RS frame using each gyro
trace in turn; eight independent gates (see [Expected output](#expected-output)) check that the
correction is geometrically real, numerically converged, and — even when the gyro is imperfect — still
far better than doing nothing. Every component named in the catalog bullet ("rolling-shutter correction"
and "IMU rates") is implemented in full; nothing here is scoped down to a documented-only stub.

## What this computes & why the GPU helps

The computation is a **per-pixel map with a small embedded search**: for every one of the
384x288 = 110,592 OUTPUT (reference-view) pixels, resolve which row of the captured rolling-shutter
frame it came from (a 3-iteration fixed-point search over a tiny, precomputed per-row rotation lookup
table), then bilinearly sample that source pixel. Every output pixel's search is completely independent
of every other pixel's — a textbook GPU *map* — so one thread per pixel is the natural mapping; see
`kernels.cu`'s `rs_correct_kernel` for the launch reasoning (including why a 2-D 32x8 thread block is
chosen specifically to align one warp with one output row, so the row lookup table's early reads
broadcast to a whole warp from GPU **constant memory** — the same broadcast pattern 09.01's robot model
uses).

The one thing that does **not** run on the GPU is turning the gyro samples into that per-row rotation
table: it is a small (order-of-hundreds-of-steps), inherently *sequential* recurrence over a tiny amount
of data, so it stays on the host (`kernels.cu`'s file header explains the design choice in full — this
is a deliberate "recompute vs. LUT" GPU-mapping lesson, not an oversight).

- **Pattern:** map (one thread per output pixel; embarrassingly parallel).
- **Bottleneck parallelized:** the per-pixel row-time search + bilinear sample, 110,592 independent
  instances.
- **What stays on the host, and why:** the gyro-to-rotation integration (sequential, tiny) — see
  `kernels.cu`.

## System context — where this sits in a robot

Full architecture reference: [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md).
Physical/commercial grounding: [`PRACTICE.md`](PRACTICE.md).

- **Stack position:** the tail end of **perception's image pipeline**, immediately after image
  acquisition and immediately before every geometric consumer — it sits between the camera driver and
  domains 01 (features/flow), 02/05 (SLAM/mapping), and any visual servoing loop (21). Conceptually it
  slots in right after 01.01's debayer/undistort/rectify pipeline and right before 01.03 (optical flow)
  or 01.04 (feature tracking) in a real chain.
- **Upstream inputs:** a rolling-shutter `Image` (row-sequential 8-bit frame, known `t_line`/readout
  time) and an `ImuSample` stream (200 Hz angular rate, camera/body frame) — message-shaped like ROS 2's
  `sensor_msgs/Image` + `sensor_msgs/Imu.angular_velocity`, per docs/SYSTEM_DESIGN.md's interface
  conventions. **Time-sync honesty:** this project assumes the camera and IMU timestamps are already in
  one common, perfectly-synchronized clock — a real rig needs hardware timestamping (FSYNC/strobe) or an
  online time-offset estimator to get this; see `PRACTICE.md` §3.
- **Downstream consumers:** any geometric algorithm that assumes a single, shared camera pose per frame —
  named explicitly: 01.03 (optical flow — RS skew corrupts the flow field near frame edges), 01.04
  (feature tracking — RS-distorted descriptors drift), and 05.xx SLAM/VIO front-ends, for whom rolling
  shutter is one of the **top real-world sources of pose error** on a rotating platform if left
  uncorrected (THEORY.md "Where this sits in the real world" names the production alternative: modeling
  RS directly inside the estimator).
- **Rate/latency budget:** cameras in this repo's reference robots run 30-60 Hz
  (docs/SYSTEM_DESIGN.md §1.1); this correction must fit inside that per-frame budget (a few ms on this
  project's 384x288 test size — see `[time]` lines in the demo output) so it never becomes the frame-rate
  bottleneck.
- **Reference robot(s):** the **quadrotor** (the classic "jello" platform — fast yaw/pitch/roll jitter
  during aggressive flight) and a **handheld/tripod-free 3-D scanner or AR handset** (human hand tremor
  at handheld scan rates). Both appear in docs/SYSTEM_DESIGN.md §2.
- **In production:** most production VIO/SLAM stacks do not correct the IMAGE at all — they model the
  rolling shutter directly INSIDE the state estimator (per-row timestamps folded into the optimization,
  e.g. VINS-Fusion's and OKVIS's RS-aware variants), which is more accurate but couples the correction to
  the estimator; a small minority of systems (broadcast video, some AR pipelines) do correct the image
  itself, closer to what this project teaches. THEORY.md expands on both paths.
- **Owning team:** perception, adjacent to embedded/firmware (who own the camera/IMU driver and hardware
  timestamping this project assumes) — docs/SYSTEM_DESIGN.md §5.1.

## The algorithm in brief

- **Body-rate quaternion integration** (exponential-map, exact for piecewise-constant angular velocity) —
  turns sparse 200 Hz gyro samples into a dense orientation trajectory. [`THEORY.md` "The math"](THEORY.md#the-math).
- **Pure-rotation row homography**, `H(v) = K * R_rel(v) * K^-1` — the same `K*R*K^-1` construction
  01.01 uses for its rectifying rotation, here re-evaluated per output row instead of once per image.
  [`THEORY.md` "The math"](THEORY.md#the-math).
- **3-iteration fixed-point row-time search** — the output pixel's source ROW determines the sample
  TIME determines the ROTATION determines the source ROW; iterating converges this circular dependency.
  [`THEORY.md` "The algorithm"](THEORY.md#the-algorithm) derives the contraction argument.
- **Bilinear resampling** with an honest invalid/out-of-frame flag (no silent clamp-to-edge).
  [`THEORY.md` "Numerical considerations"](THEORY.md#numerical-considerations).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/rolling-shutter-correction-using-imu-rates.sln`](build/rolling-shutter-correction-using-imu-rates.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/rolling-shutter-correction-using-imu-rates.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only
(no cuBLAS/cuFFT/cuRAND/Thrust; the linear algebra here is small enough to hand-roll, which is the point).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

100% synthetic, generated by [`scripts/make_synthetic.py`](scripts/make_synthetic.py) (fixed seed 42, a
tiny hand-rolled xorshift32 PRNG, standard library only). Full details, field-by-field, checksums, and
regeneration command: [`data/README.md`](data/README.md). No public dataset applies here — rolling-
shutter correction needs analytically-known ground truth (what an ideal global-shutter camera would have
seen) that no real capture can provide after the fact, so synthetic authorship is not just the repo
default but the only honest option (`data/README.md` explains further).

## Expected output

`.\demo\run_demo.ps1` builds (if needed), runs on the committed sample, and prints a `VERIFY:` line plus
eight `GATE <name>: PASS` lines before the final `RESULT: PASS`. Every verdict is backed by a measured
number printed on an adjacent `[info]` line (not diffed — see `src/main.cu`'s output-contract comment);
the ones below are from an actual run on the reference machine (RTX 2080 SUPER, sm_75, Release|x64):

| Gate | Measured | What it proves |
|------|----------|-----------------|
| `quat_integration_analytic` | \|measured-analytic\| = 0.000026 rad | the exponential-map integrator matches a closed-form rotation exactly, bypassing every camera-model primitive |
| `VERIFY` (GPU vs CPU) | max\|gpu-cpu\| = 1.0000 (uint8 scale) | the kernel and its CPU twin agree within one rounding-class ULP |
| `restoration` | corrected (clean gyro) masked mean\|err\| = 0.8335 / 255 | the corrected image is very close to the true global-shutter reference |
| `restoration_negative_control` | uncorrected masked mean\|err\| = 3.9893 / 255 | doing nothing is ~4.8x worse — the correction is doing real, measurable work |
| `straightness_corrected` | marker-line spread = 0.5219 px | the scene's known-straight line reads straight after correction |
| `straightness_negative_control` | RAW marker-line spread = 4.8504 px | the SAME line reads visibly sheared before correction (~5 px peak column shift — see `demo/out/rotation_profile.csv`) |
| `row_time_convergence` | max\|iter3-iter2\| = 0.00221 px | the 3-iteration fixed-point search has converged to well under a hundredth of a pixel |
| `gyro_degradation` | degraded corrected mean\|err\| = 1.3196 (clean 0.8335, uncorrected 3.9893) | an uncalibrated (biased+noisy) gyro makes correction visibly worse than the clean case, but still ~3x better than nothing |
| `valid_coverage` | 98.16% of output pixels | rows near the top/bottom of the frame lose the most source pixels (their row-time is furthest from the reference) |

Artifacts land in `demo/out/`: `rs_input.pgm`/`ground_truth_gs.pgm` (the two committed frames, copied
through for convenience), `corrected.pgm` (this project's output), `uncorrected_diff.pgm` /
`corrected_diff.pgm` (grayscale `|image - ground_truth|` heatmaps — visually compare them to see the
correction's effect), `rotation_profile.csv` (per-row relative-rotation angle — how much correction each
row needed), and `gates_metrics.csv` (every gate's measured value in one machine-readable file).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **start here.** The camera model, the quaternion algebra, the
   row-homography derivation, and every layout/constant three files must agree on, documented once.
2. [`src/kernels.cu`](src/kernels.cu) — the ONE GPU kernel (`rs_correct_kernel`): the per-pixel
   fixed-point search + bilinear sample, and why the gyro integration is deliberately NOT a second kernel.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of that kernel; read it
   side by side with `kernels.cu` to see exactly what "the loop became threads" means here.
4. [`src/main.cu`](src/main.cu) — orchestration: gyro integration, the row-LUT build, both GPU runs
   (clean/degraded gyro), VERIFY, all eight gates, and every artifact write.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **OKVIS / VINS-Fusion (RS-aware variants)** — production VIO systems that model the rolling shutter
  directly inside the state estimator (per-row timestamps in the optimization) rather than correcting the
  image first; the production-grade alternative to this project's approach (THEORY.md expands).
- **Kalibr** — the standard open-source camera/IMU calibration toolbox; among other things it estimates a
  camera's line-readout time and the camera-IMU time offset, the two numbers this project simply assumes
  known (`PRACTICE.md` §3).
- **OpenCV's `videostab` module** — classic 2-D rolling-shutter/wobble stabilization for video, a
  simpler (translation+affine, no IMU) cousin of this project's rotation-only, IMU-driven approach.
- **Forssén & Ringaby, "Rectifying rolling shutter video from hand-held devices" (CVPR 2010)** — the
  paper that popularized exactly this problem: per-row camera-pose interpolation and warping.
- **Project 01.01 (full GPU image pipeline)** — the `K*R*K^-1` inverse-mapping remap doctrine and the
  straightness-gate pattern this project generalizes from one fixed rotation to a time-varying one.
- **Project 09.01 (batched forward kinematics)** — the `(w,x,y,z)` quaternion convention and the
  `__constant__`-memory "upload once, broadcast to every thread" pattern this project reuses.

## Exercises

1. **Change the readout time.** Halve `kReadoutTimeS` in `kernels.cuh` (and the matching constant in
   `make_synthetic.py`) and regenerate the sample. Does the RS skew shrink? Does `restoration_negative_control`
   still pass its floor?
2. **Sweep the fixed-point iteration count.** Try `kFixedPointIters = 1` and `2`. Watch
   `row_time_convergence`'s measured delta grow, and see how far `restoration`'s error grows with it.
3. **Break the time sync.** Add a constant offset to every gyro sample's `t_s` in
   `read_gyro_csv`'s input (or in the generator) and observe which gate catches it first.
4. **Model translation.** The project assumes pure rotation. Add a small constant camera translation to
   `make_synthetic.py`'s scene rendering (still using the flat reference-plane texture) and watch the
   restoration gate degrade as the near-field-parallax assumption breaks — quantify how much translation
   this scene's synthetic "distance" can tolerate before the gate fails.
5. **Slerp instead of lerp.** `lerp_row_quat` in `kernels.cuh` linearly interpolates quaternion
   components. Implement a proper SLERP and measure whether `row_time_convergence`'s or `restoration`'s
   numbers actually change at this project's row-to-row rotation scale (THEORY.md predicts they should not).

## Limitations & honesty

- **Pure rotation only — no translation.** The row homography assumes the camera never translates during
  the frame; THEORY.md derives why this is exact for an infinitely distant scene and quantifies when it
  breaks down for a NEARBY scene (near-field parallax: a translating camera samples different world
  points row to row, not just different orientations of the same point).
- **Perfect camera-IMU time sync assumed.** Real rigs need hardware timestamping or online time-offset
  estimation (`PRACTICE.md` §3); this project's gyro and frame timestamps are synthetic and exact.
- **No IMU-camera extrinsic (lever-arm) calibration.** The gyro is assumed co-located with, and axis-
  aligned to, the camera; a real rig's IMU sits some distance away and at some small misalignment, both
  of which need calibration (Kalibr; `PRACTICE.md` §2-3).
- **Linear (not physically calibrated) gyro degradation model.** The "degraded" gyro trace adds a
  constant bias plus uniform (not Gaussian) noise — illustrative, not a datasheet-derived IMU error model.
- **Small-rotation regime.** The synthetic rotation amplitudes (a few degrees) keep every division in the
  homography well-conditioned; THEORY.md notes where a much larger rotation would need extra care.
- **Sim-validated only.** Nothing in this project runs on real hardware or commands motion; it is an
  offline image-correction demo over synthetic data (CLAUDE.md §1, §8).
