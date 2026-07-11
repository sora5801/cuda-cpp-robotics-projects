# Push note — 2026-07-11-01: batch 2c — domain 01, from calibration to closed loop

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2c (**47/505**; domain 01 at 11/24) walks the camera from *trustworthy pixels* to
*commanding motion*: photometric calibration (01.09), rolling-shutter correction (01.10),
low-light denoising (01.11), and image-based visual servoing (01.12) — the domain's first
closed-loop control project, built in 08.01's rollout-farm idiom. Highlights: 01.09 proves its
decomposition semantics with a gate showing the radial fit's residual is PRNU-scale (ratio
1.009) and confirms the 1/√N averaging law at 2.000/4.005; 01.10 teaches the row-time
fixed-point search and restores a 4.85 px rolling-shutter skew to 0.52 px against an
independently-rendered global-shutter truth; 01.11 lands all three named denoisers **including
BM3D-lite** with a designed Gaussian negative control that passes noise floors but fails edge
preservation at 16% — and reports an honest ordering surprise (NLM edging BM3D-lite on smooth
hashed texture) with cause analysis; 01.12 runs 4,096 IBVS loops × 3 controller variants in
5.1 ms, fits the feature-error decay rate to within 0.4% of the commanded λ, and *gates the
classic failure*: a near-180° cohort must exhibit camera retreat (100% detected, conditioning
correlation ≈ 0 proving the pathology geometric).

## What changed

- **[projects/01-perception-cameras-vision/01.09-photometric-vignetting-calibration-kernels/](../projects/01-perception-cameras-vision/01.09-photometric-vignetting-calibration-kernels/)** —
  dark/flat-stack calibration, cos⁴+PRNU/DSNU recovery, parametric radial fit, correction
  kernels; 6 gates.
- **[projects/01-perception-cameras-vision/01.10-rolling-shutter-correction-using-imu-rates/](../projects/01-perception-cameras-vision/01.10-rolling-shutter-correction-using-imu-rates/)** —
  gyro quaternion integration (09.01 conventions), per-row quaternion LUT in constant memory,
  row-time fixed-point remap; 8 gates including both negative controls and a gyro-bias study.
- **[projects/01-perception-cameras-vision/01.11-low-light-denoising/](../projects/01-perception-cameras-vision/01.11-low-light-denoising/)** —
  bilateral (naive + tiled, bit-identical, 16.0× speedup), NLM, BM3D-lite hard-thresholding
  stage (hand-rolled 8×8 DCT + 1-D Haar); exact Poisson noise gated against its analytic
  prediction; 5 gates.
- **[projects/01-perception-cameras-vision/01.12-visual-servoing/](../projects/01-perception-cameras-vision/01.12-visual-servoing/)** —
  batched IBVS convergence-basin study, 3 variants, analytic decay gate, retreat-pathology
  gate; sim-validated-only safety caveat per §8.
- **[docs/STATUS.md](../docs/STATUS.md)** — 01.09–01.12 → `done` (**47/505**).

## New projects (didactic blurbs)

**01.09 — Photometric/vignetting calibration.** Why corners are 26% darker (cos⁴ derived from
solid-angle first principles), how flat-fielding recovers it, and two free statistics lessons:
noise averaging follows 1/√N to under 1%, and a well-designed gate can prove a *decomposition*
(fit residual ≈ PRNU magnitude), not just a fit. Correction takes center-vs-corner disparity
from 26.2% to 0.11% on held-out scene content.

**01.10 — Rolling-shutter correction.** The jello effect from CMOS readout physics, and the
circular dependency at the heart of correction — source row → sample time → rotation → source
row — resolved by a taught fixed-point iteration (converges to 0.0022 px in 3 steps). The
degraded-gyro study is the honest bridge to VIO: an 8°/s bias still beats doing nothing 3×,
but the gap is why estimators track bias online.

**01.11 — Low-light denoising.** Photon statistics make low light a *signal-dependent* noise
problem (SNR ~ √signal); bilateral, NLM, and BM3D-lite answer it with three different priors.
The tiled bilateral is the repo's cleanest shared-memory lesson yet: bit-identical output,
16× faster. Most interesting artifact: the four residual heatmaps side by side —
`residual_gaussian_baseline.pgm` shows exactly the edge energy the negative control destroyed.

**01.12 — Visual servoing.** The lens becomes part of the plant: control error lives on the
image plane, the interaction matrix is derived from the projection derivative, and a GPU farm
of 4,096 closed loops maps the convergence basin that a single textbook example can't show —
including the built-in pathology (camera retreat at large optical-axis rotations) as a gated,
100%-detected feature of the geometry, not a bug.

## How to build & run

```powershell
projects\01-perception-cameras-vision\01.09-photometric-vignetting-calibration-kernels\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.10-rolling-shutter-correction-using-imu-rates\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.11-low-light-denoising\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.12-visual-servoing\demo\run_demo.ps1
```

## What to study here

The batch reads as one lesson in *trusting a pixel enough to act on it*: 01.09 makes intensity
mean something, 01.10 makes geometry mean something under motion, 01.11 recovers both from
noise, 01.12 closes the loop. Study 01.12's basin_map.ppm next to THEORY's retreat derivation.
Exercise: chain 01.09's correction in front of 01.11's denoisers and measure whether flat-field
correction changes the noise-model-sanity gate (hint: gain scaling scales shot noise too).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-11), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (12/12, 14/14, 15/15, 15/15);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder.

- **01.09:** twins exact to 9.2e-4 worst (radial-bin atomics); gain 0.19% mean rel error; DSNU
  corr 0.966; 1/√N ratios 2.000/4.005; disparity 26.2% → 0.11%.
- **01.10:** quaternion analytic check 2.6e-5 rad; convergence 0.0022 px; MAE 3.99 → 0.83;
  skew 4.85 → 0.52 px; degraded gyro (8°/s bias) still ≤60% of uncorrected.
- **01.11:** naive-vs-tiled bit-identical; PSNR +6.31/+12.73/+12.24 dB; Gaussian control fails
  edges at 16% (bar 55%) as designed; noise-model sanity 0.89–1.03.
- **01.12:** trajectory twin 8.9e-8 worst-step; decay fit 2.0079 vs λ=2.0; basins
  97.9/96.5/96.3%; retreat 100% detected; Debug-vs-Release comparison isolates FMA fusion as
  the sole rounding source (0.0 divergence in Debug).

## Known limitations / TODOs

- 01.09 fits the geometric (not true decentered) optical center — stated, with the extension
  as an exercise. 01.10 assumes perfect camera-IMU time sync and pure rotation (parallax
  honesty in THEORY). 01.11's BM3D-lite is stage 1 only (Wiener stage documented); σ assumed
  global. 01.12 is kinematic (no robot dynamics), 4-point planar target, sim-only per §8.

## Next push preview

Batch 2d: 01.13 Canny+Hough, 01.14 template matching (NCC), 01.15 background subtraction,
01.16 checkerboard/ChArUco detection — the industrial-vision quartet.
