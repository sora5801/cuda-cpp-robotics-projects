# Push note — 2026-07-11-03: batch 2e — the 3-D sensing quartet

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2e (**55/505**; domain 01 at 19/24) is the 3-D sensing arc: extrinsic calibration (01.17)
glues the sensors together, depth completion (01.18) densifies sparse LiDAR with RGB guidance,
and two active-ranging siblings — structured light (01.19) and indirect time-of-flight (01.20) —
share one deliberate teaching thread: the *same* 4-step phase arithmetic carries projector
columns in one and time delays in the other, and both gate its ambient-cancellation property at
exactly 0.0 rad. The batch's designed lessons all landed by measurement: 01.17's coplanar-pose
cohort shows calibration degeneracy at **110× worse conditioning** (pose diversity beats view
count — 01.16's Zhang finding made mechanistic); 01.18's texture-trap and camouflage-edge duals
quantify precisely where the "image edges ≈ depth edges" prior helps (1.27× at true edges) and
where it lies (3.63× at color-camouflaged depth edges — gated so the demo must *show* the
failure); 01.19 measures the Gray-vs-binary boundary argument at **30.8×** with a dedicated
20,000-probe stress kernel; and 01.20 generates flying pixels from the *physics* (area-weighted
phasor mixing at edges) rather than painting them, then removes them at 100% precision.

## What changed

- **[projects/01-perception-cameras-vision/01.17-camera-lidar-camera-camera-extrinsic-calibration/](../projects/01-perception-cameras-vision/01.17-camera-lidar-camera-camera-extrinsic-calibration/)** —
  batched LM with derived analytic se(3) Jacobian (gated vs central differences), 1,024-start
  convergence-basin farm in 3.0 ms, coplanar degeneracy cohort; 11 gates/verifies.
- **[projects/01-perception-cameras-vision/01.18-depth-completion/](../projects/01-perception-cameras-vision/01.18-depth-completion/)** —
  scan-line LiDAR projection (atomicMin float-encode z-buffer), Perona–Malik edge-aware
  diffusion with compile-time CFL assert, IDW baseline, texture-trap + camo-edge duals; 6 gates.
- **[projects/01-perception-cameras-vision/01.19-structured-light-decoding-for-3d-scanners/](../projects/01-perception-cameras-vision/01.19-structured-light-decoding-for-3d-scanners/)** —
  Gray code + 4-step phase shift + the production hybrid, ray-plane triangulation, three
  analytic surface gates; 13 gates/verifies.
- **[projects/01-perception-cameras-vision/01.20-time-of-flight-raw-processing/](../projects/01-perception-cameras-vision/01.20-time-of-flight-raw-processing/)** —
  4-tap CW demodulation, designed single-frequency aliasing, CRT-style dual-frequency
  unwrapping, phasor-mixed flying pixels detected against independent sub-ray truth; 15
  gates/verifies.
- **[docs/STATUS.md](../docs/STATUS.md)** — 01.17–01.20 → `done` (**55/505**).

## New projects (didactic blurbs)

**01.17 — Extrinsic calibration.** Why extrinsics drift (the aluminum-mount thermal arithmetic
is in THEORY), the analytic reprojection Jacobian derived and *checked by gate*, and a
convergence-basin study a textbook can't show: 77.3% of starts within 2.4 rad/1.2 m converge —
measured after the builder widened the range because 100% convergence taught nothing. Zero-noise
anchors recover the transform to machine precision.

**01.18 — Depth completion.** The edges-coincide prior treated as a *prior*: derived, exploited
(guided diffusion beats RGB-blind IDW), then interrogated with designed counterexamples — the
painted checkerboard on a flat wall must not become a depth edge, and the gray box against a
gray wall must visibly smear (both gated). CFL stability enforced at compile time.

**01.19 — Structured light.** Why the light *carries* the texture that passive stereo (01.02)
lacks; Gray's single-bit-boundary property proven then measured at 30.8× over plain binary;
hybrid Gray+phase reaching 0.071-column correspondence (3.76× over Gray alone); metric
reconstruction to 3.79 mm plane RMS. The dark-stripe cohort is 99.1% confidence-rejected with
zero hallucinated survivors.

**01.20 — Time of flight.** The same atan2 as 01.19 on a different physical carrier — a
deliberate pairing for study. Single-frequency aliasing demonstrated on 100% of the
beyond-ambiguity wall; dual-frequency unwrapping recovers 12.9 mm mean depth with an honest
98.06% wrap-correct rate (the failure probability derived in THEORY); flying pixels exist here
because phasors sum — the before/after X-Z profile pair is the money shot.

## How to build & run

```powershell
projects\01-perception-cameras-vision\01.17-camera-lidar-camera-camera-extrinsic-calibration\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.18-depth-completion\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.19-structured-light-decoding-for-3d-scanners\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.20-time-of-flight-raw-processing\demo\run_demo.ps1
```

## What to study here

Read 01.19 and 01.20 back to back — the same four samples and the same atan2, once across space
and once across time, is the cleanest "one math, two sensors" pairing in the repo so far. Then
01.17's degeneracy gate next to 01.16's Zhang finding: two projects independently measuring the
same truth (geometry diversity, not data volume, buys observability). Exercise: feed 01.20's
point cloud into 01.18's completion as the sparse input and reason about which failure taxonomy
applies.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-11), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (19/19, 14/14, 24/24, 28/28);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder
(01.20's builder disclosed one incidental read-only `git status` — no state changed).

- **01.17:** Jacobian analytic-vs-numeric 1.5e-3; basin 792/1024; recovery 0.31°/7.4 mm and
  0.12°/0.87 mm; degeneracy 110×/9.5×; zero-noise at machine precision.
- **01.18:** twins to 9.5e-7 m over 1,400 diffusion iterations; guided 0.92 m vs IDW 1.21 m MAE;
  trap 1.35× bounded; camo 3.63× demonstrated; density sweep monotone.
- **01.19:** Gray decode exact (0/30,000); hybrid 0.071 cols (3.76×); Gray-vs-binary 30.8×;
  ambient invariance 0.0 rad; plane/sphere/step 3.79 mm/1.42%/0.45%; dark stripe 99.1%.
- **01.20:** six twins near-exact; aliasing 14,601/14,601; unwrap 12.9 mm/98.06%; flying pixels
  100% precision/62% recall vs independent truth; plane/sphere/step 14.1 mm/3.26%/0.01%; dark
  cohort 100%.

## Known limitations / TODOs

- 01.17: isotropic (not spherical) LiDAR noise, camera-camera source poses treated exact —
  both flagged. 01.18: fixed-radius IDW, single diffusion parameter set (sensitivity reported).
  01.19: temporal coding assumes a static scene (motion honesty; single-shot methods named).
  01.20: multipath documented-only; flying-pixel recall 62% at 100% precision (the
  didactic 3×3 detector — production alternatives named).

## Next push preview

Batch 2f closes domain 01: 01.21 scene flow from RGB-D pairs, 01.22 motion deblurring +
super-resolution, 01.23 RAW→RGB ISP on Jetson (desktop-runnable teaching core per §5), 01.24
transparent/reflective object detection via polarization.
