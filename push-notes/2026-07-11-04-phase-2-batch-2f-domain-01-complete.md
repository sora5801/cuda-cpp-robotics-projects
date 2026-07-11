# Push note — 2026-07-11-04: batch 2f — domain 01 complete (24/24)

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2f (**59/505**) finishes **domain 01 — cameras & vision — in full: 24 of 24 projects
done**, the first domain completed under Phase 2. The closing four: scene flow (01.21) brings
motion into 3-D and carries the batch's process story — held at the lead gate once, then a
finisher *root-caused* its object-motion direction flip to least-squares conditioning (proven by
truth-mask instrumentation, fixed with a fixed-rotation fit: cos −0.91 → 0.946) and *proved* the
segmentation IoU is bounded by a genuine two-frame limitation (one coherent disocclusion blob no
size filter can remove — sweep table in THEORY). Deblurring + super-resolution (01.22) gates its
failure physics (the naive inverse must end up worse than doing nothing; a 25° PSF error costs a
measured 7 dB) and documents a metric lesson — aliased moiré gave contrast a false near-tie, so
the SR gate was rebuilt on pattern correlation (0.984 vs 0.215). The full RAW→RGB ISP (01.23)
lands the §5 hardware-dependent policy: eight radiometric stages, all nine twins bit-exact,
Jetson path documented and never faked. And polarization imaging (01.24) closes the domain on
first principles: ground truth computed from the Fresnel equations, the analyzer meeting the
physics at |Δ| = 0.00034, Brewster's angle found 0.31° from atan(n) — and the money shot the
domain deserves: glass invisible in intensity (recall 0.0%), glowing in DoLP (recall 97.0%).

## What changed

- **[projects/01-perception-cameras-vision/01.21-scene-flow-from-rgb-d-pairs/](../projects/01-perception-cameras-vision/01.21-scene-flow-from-rgb-d-pairs/)** —
  pyramidal LK → metric 3-D lift → IRLS+Horn robust ego-motion (0.017°/0.9 mm with the mover
  present vs 0.28°/36 mm naive, both gated) → residual segmentation with CCL + size filter;
  worker + finisher (see Summary).
- **[projects/01-perception-cameras-vision/01.22-motion-deblurring-and-super-resolution/](../projects/01-perception-cameras-vision/01.22-motion-deblurring-and-super-resolution/)** —
  Wiener (+3.13 dB) / Richardson–Lucy (+2.35 dB) / gated naive-inverse failure (−4.28 dB);
  8-frame shift-and-add + IBP super-resolution resolving below-Nyquist bars; cuFFT twinned
  against an independent CPU radix-2 FFT.
- **[projects/01-perception-cameras-vision/01.23-full-rawrgb-isp-on-jetson/](../projects/01-perception-cameras-vision/01.23-full-rawrgb-isp-on-jetson/)** —
  complete 8-stage ISP (black level → shading → defects → dual AWB → MHC demosaic → CCM →
  sRGB), per-stage generator truth, both AWB failure modes designed and asserted, desktop
  teaching core per §5 with the Jetson/Argus path documented.
- **[projects/01-perception-cameras-vision/01.24-transparent-reflective-object-detection/](../projects/01-perception-cameras-vision/01.24-transparent-reflective-object-detection/)** —
  DoFP mosaic → Stokes → DoLP/AoLP → detection run identically on two signals; Fresnel anchor,
  Brewster sweep, Malus 1-DOF self-consistency invariant, matte negative control.
- **[docs/STATUS.md](../docs/STATUS.md)** — 01.21–01.24 → `done` (**59/505; domain 01: 24/24**).

## New projects (didactic blurbs)

**01.21 — Scene flow.** Robust estimation as the dominant-motion assumption, made quantitative:
the mover corrupts a naive fit 38× worse in translation. The disocclusion-blob finding is the
honest deep lesson — two-frame residual segmentation *cannot* distinguish "object moved" from
"world revealed," which is why production systems reason over time.

**01.22 — Deblurring + SR.** Where restoration's information comes from (and where it can't):
PSF zeros amplify noise 22× in the naive inverse; SR's detail comes from aliasing sampled at
sub-pixel phases — and a metric that rewards amplitude without truth will lie to you.

**01.23 — Full ISP.** Every DN a robot ever sees passed through this chain; here every stage has
exact truth, and the two AWB estimators fail exactly where theory says (dominant-color scenes;
wrong illuminant = 1.84× chart-error cast).

**01.24 — Polarization.** The modality that sees what stereo, structured light, and ToF all miss
(each named with its failure reason): Fresnel reflection polarizes, Brewster maximizes, and a
2×2 mosaic of analyzers measures it at frame rate. The generator and analyzer meet at the
physics to 3.4e-4.

## How to build & run

```powershell
projects\01-perception-cameras-vision\01.21-scene-flow-from-rgb-d-pairs\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.22-motion-deblurring-and-super-resolution\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.23-full-rawrgb-isp-on-jetson\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.24-transparent-reflective-object-detection\demo\run_demo.ps1
```

## What to study here

Domain 01 now reads as one course: pixels made trustworthy (01.01–01.12), pixels made
*decisive* (01.13–01.16), pixels made 3-D (01.17–01.20), and pixels made honest about their
limits (01.21–01.24). For this batch specifically: 01.24's THEORY (Fresnel → Brewster →
Stokes) is the cleanest physics-first chapter in the domain; 01.21's disocclusion analysis and
01.22's metric lesson are the two best examples yet of *investigating* a weak number instead of
tuning it away. Exercise: run 01.23's tungsten output through 01.24's detection and reason
about what an illuminant cast does to DoLP (spoiler: nothing — derive why).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-11), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (19/19, 17/17, 16/16, 23/23);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder.

- **01.21:** worker + finisher (lead held the first delivery); ego-motion byte-identical
  pre/post fix; CCL + size-filter twins bit-exact; static control 1.57%.
- **01.22:** six twins tight (cuFFT vs independent CPU FFT); naive inverse −4.28 dB asserted;
  PSF mismatch −7.04 dB; SR correlation 0.984 vs 0.215; IBP monotone 12/12.
- **01.23:** nine twins bit-exact (0.000000); MHC +1.52 dB over bilinear; defect recovery zero
  false corrections; chart 7.7/255 mean; fusion 66.7% traffic saving at 0.000000 divergence.
- **01.24:** seven twins exact (detection masks 0/16,384); Fresnel anchor 0.00034; Brewster
  0.31°; glass recall 97.0% DoLP vs 0.0% intensity; negative control exactly 0 detections.

## Known limitations / TODOs

- 01.21: two-frame scope (temporal accumulation documented as the production fix); object-motion
  magnitude honestly ungated. 01.22: non-blind deconvolution (blind documented); known shifts
  for SR (registration documented). 01.23: teaching-scale resolution, desktop-only measurements
  (Jetson numbers deliberately absent). 01.24: illustrative DoFP layout (vendor calibration
  realities documented); metal signature phenomenological.

## Next push preview

Domain 02 — LiDAR & point clouds (19 remaining, 02.06 ICP done as flagship): batch 2g starts
with 02.01 voxel-grid downsampling with GPU spatial hashing and continues in ★-then-ID order.
