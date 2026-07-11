# Push note — 2026-07-11-00: batch 2b — domain 01 intermediates, first four

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2b delivers the first four intermediate projects of domain 01 (**43/505**): optical flow,
SIFT, fisheye/surround-view, and HDR. Together they finish the "geometry and photometry of one
camera" arc that the ★ trio opened. Recurring theme of this batch: **every project carries a
designed negative control, and two of them caught real bugs the CPU twin could not see** —
01.03's analytic translation gate exposed a ×32 Scharr normalization error living identically in
both the GPU path and its twin (the third confirmed save for the two-tier doctrine after 13.03
and 01.06), and 01.05 turned a *failed assumption* into material: rotation/scale-normalized
synthetic corners collapse into a low-dimensional descriptor family, an empirically-validated
finding that reframes the classic ratio test and motivates RANSAC. Bundled bullets were scoped
per §2: Farneback (01.03) and SURF (01.05) ship documented-only, taught to implementable depth
in their THEORY.md files and stated in each README §13.

## What changed

- **[projects/01-perception-cameras-vision/01.03-optical-flow/](../projects/01-perception-cameras-vision/01.03-optical-flow/)** —
  dense 3-level pyramidal Lucas–Kanade (structure-tensor confidence = the aperture problem made
  visible) + 5×5 census block flow with forward-backward validity; 8 analytic gates; HSV
  flow-wheel artifacts.
- **[projects/01-perception-cameras-vision/01.05-sift-surf-on-gpu/](../projects/01-perception-cameras-vision/01.05-sift-surf-on-gpu/)** —
  full SIFT (separable scale space, DoG extrema + Cramer sub-pixel refine, warp-shuffle
  orientation histograms and 128-D descriptors, L2 ratio+mutual matching); 6 gates.
- **[projects/01-perception-cameras-vision/01.07-fisheye-omnidirectional-unwarping-and-multi/](../projects/01-perception-cameras-vision/01.07-fisheye-omnidirectional-unwarping-and-multi/)** —
  equidistant fisheye model, rectilinear + cylindrical unwarps, four-camera bird's-eye stitch
  through true 3-D extrinsics with feathered seams; 7 gates; ray-cast synthetic parking lot.
- **[projects/01-perception-cameras-vision/01.08-hdr-exposure-fusion-tone-mapping-for-outdoor/](../projects/01-perception-cameras-vision/01.08-hdr-exposure-fusion-tone-mapping-for-outdoor/)** —
  Debevec–Malik CRF recovery + radiance merge + Reinhard/local tone mapping vs Mertens pyramid
  fusion, with a naive blend as the measured failure baseline; 6 gates.
- **[docs/STATUS.md](../docs/STATUS.md)** — 01.03, 01.05, 01.07, 01.08 → `done` (**43/505**;
  domain 01: 7/24).

## New projects (didactic blurbs)

**01.03 — Optical flow.** Brightness constancy from image formation and exactly where it breaks:
the census transform's rank-order invariance is *measured* against LK on the same
brightness-ramped scene (0.64 px vs 17.6 px EPE), and the pyramid earns a 4.12× advantage over
single-level at an equal iteration budget. Most interesting artifact: `flow_lk_rotzoom.ppm`
against the HSV wheel.

**01.05 — SIFT on GPU.** The warp chapter: one 32-lane warp per keypoint, `__shfl_down_sync`
histogram trees vs naive shared atomics, and the float non-associativity lesson measured at
literal print-precision zero divergence from the sequential twin. Scale invariance proven by
gate: median matched scale ratio 1.428 vs the true 1.5, rotation to 0.25°. Honest finding kept
in THEORY: why synthetic corners make canonical Lowe matching hard — and why real pipelines end
in RANSAC.

**01.07 — Fisheye + surround view.** Why r = f·θ buys a hemisphere (and what it costs), inverse
mapping through true 3-D extrinsics, and the flat-ground assumption made quantitative: 10.9/255
BEV error on flat asphalt vs 37.3 near tall objects (3.41×) — the ghosting every parking-display
engineer knows, derived then measured. A lane edge bowing 60.9 px in the fisheye straightens to
0.29 px after unwarp.

**01.08 — HDR.** The dynamic-range problem from photometry (sun ≈ 100 klx, underbody ≈ 10 lx vs
a ~60–70 dB sensor), then both classic answers compared: radiance reconstruction (CRF recovered
to 0.058 ln-units with the scale ambiguity corrected *explicitly*) and Mertens fusion — with the
naive single-scale blend kept as a gated failure case (halo ratio 1.52×). Local tonemap covers
99.97% well-exposed vs 86.4% for the best single exposure.

## How to build & run

```powershell
projects\01-perception-cameras-vision\01.03-optical-flow\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.05-sift-surf-on-gpu\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.07-fisheye-omnidirectional-unwarping-and-multi\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.08-hdr-exposure-fusion-tone-mapping-for-outdoor\demo\run_demo.ps1
```

## What to study here

The batch's meta-lesson is verification design: read 01.03's THEORY §verification (the ×32
gradient bug the twin couldn't see), then 01.05's descriptor-collapse finding, then compare with
01.06's corner-margin case from batch 2a. Exercise: feed 01.03's brightness-ramp scene to
01.04's ORB matcher and predict — then measure — which of its gates survive.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10/11), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (18/18, 13/13, 13/13, 12/12);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder.

- **01.03:** census bit-exact (0/19,200 signatures, WTA costs exact); translation EPE 0.14 px
  (LK) / 0.27 px (census); pyramid advantage 4.12×; census brightness robustness 0.64 px vs LK
  17.6 px; zero-motion and confidence-mask-sanity controls pass.
- **01.05:** DoG extrema sets exactly equal; warp reductions at 0.0 printable divergence;
  0/155 match-index mismatches; scale recovery 4.8% error; rotation 0.25°; negative control
  0/14; inlier gate honestly floored and annotated (see the project's in-demo honesty note).
- **01.07:** model roundtrip 0.0 px; straightness 0.29 px vs 60.9 px raw (negative control);
  BEV flat 10.9 vs object-region 37.3 (flat-ground control 3.41×); seams 17.6; coverage 100%.
- **01.08:** twins ≤3.3e-3 worst stage; CRF 0.058 ln-units; radiance 4.3% relative over
  19,174/19,200 unclipped pixels; coverage/detail gates beat every single exposure; halo 1.52×.

## Known limitations / TODOs

- 01.03: Farneback documented-only (§2 bundle rule); 01.05: SURF documented-only, matcher
  thresholds are scene-specific tuning (documented with the descriptor-collapse analysis);
  01.07: static rig, no photometric mismatch between cameras (stated); 01.08: static-scene
  bracketing (motion/ghosting documented as the production problem, HDR-native sensors named).

## Next push preview

Batch 2c: 01.09 photometric/vignetting calibration, 01.10 rolling-shutter correction with IMU
rates, 01.11 low-light denoising, 01.12 visual servoing — domain 01 continues in ID order.
