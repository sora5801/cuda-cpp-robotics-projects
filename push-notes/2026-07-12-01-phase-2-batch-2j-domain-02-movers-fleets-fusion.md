# Push note — 2026-07-12-01: batch 2j — movers, fleets, and fusion

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2j (**75/505**; domain 02 at 17/20) covers online moving-object segmentation (02.14),
fleet-uplink compression (02.15), multi-LiDAR merging with extrinsic refinement (02.16), and
LiDAR-camera fusion (02.17). The batch's signature is *derived properties asserted by
experiment*: 02.14 derives the two-sided residual sign semantics and then requires each cohort
to fire the derived direction (oncoming negative, receding positive); 02.15 derives the octree
quantization bound and measures distortion at 0.9962 of it — tight, not just safe — while
Huffman sits inside the Shannon [H, H+1) band at every sweep row; 02.16 derives its drift
thresholds from an aligned control cohort and closes the detect→refine→validate loop to
sub-millimeter; and 02.17 closes a cross-project loop — 01.17's analytic pixel-displacement
formula, measured at last, matches at 1.09×. Honest findings kept throughout: MIN-fusion's
just-stopped blind spot derived as a property (02.14), the compressibility advantage living in
node counts rather than entropy ratios (02.15's gate redesigned around the true metric), a
9.8×10⁹ observability contrast from wall geometry (02.16), and an occlusion check that barely
worked at exact-pixel resolution until widened — with its over-filtering cost stated (02.17).

## What changed

- **[projects/02-perception-lidar-point-clouds/02.14-moving-object-segmentation-from-sequential-scans/](../projects/02-perception-lidar-point-clouds/02.14-moving-object-segmentation-from-sequential-scans/)** —
  range-image residual MOS over M previous scans, MIN-fusion, CCL cleanup; disocclusion
  mitigation measured at 50×; window study (precision 44.6%→100%).
- **[projects/02-perception-lidar-point-clouds/02.15-point-cloud-compression-for-fleet-uplink/](../projects/02-perception-lidar-point-clouds/02.15-point-cloud-compression-for-fleet-uplink/)** —
  Morton-prefix octree + canonical Huffman with scan-composed GPU bit-packing; R-D sweep;
  structured 12.2× vs pathological 6.2× (the manifold argument quantified).
- **[projects/02-perception-lidar-point-clouds/02.16-multi-lidar-merging-extrinsic-refinement/](../projects/02-perception-lidar-point-clouds/02.16-multi-lidar-merging-extrinsic-refinement/)** —
  3-sensor rig, plane-residual drift detection, point-to-plane LM refinement to 0.11/0.31 mm,
  zero-drift control, dedup bookkeeping exact.
- **[projects/02-perception-lidar-point-clouds/02.17-lidar-camera-projection-coloring-fusion-kernels/](../projects/02-perception-lidar-point-clouds/02.17-lidar-camera-projection-coloring-fusion-kernels/)** —
  point coloring with z-buffer visibility (89.1%→0.7% wrong-color), depth painting verified
  against an independent per-pixel minimum, calibration-sensitivity sweep vs the 01.17 formula.
- **[docs/STATUS.md](../docs/STATUS.md)** — 02.14–02.17 → `done` (**75/505**).

## New projects (didactic blurbs)

**02.14 — Online MOS.** Motion seen as range-image residuals, with the sign logic derived
before it is trusted; the disocclusion band that fooled 01.21's camera version returns in LiDAR
form and is beaten by multi-scan consistency (26.4%→0.0%). The just-stopped car that MIN-fusion
cannot see is the honest cost of that choice — derived in the kernel comment.

**02.15 — Compression.** Why LiDAR compresses: surfaces are 2-D manifolds, quantified by a
designed incompressible twin cohort. The octree is a string problem over Morton codes; the
bit-packer is the prefix scan's classic encore; and the distortion bound is proven tight by
measurement.

**02.16 — Multi-LiDAR.** The field-maintenance loop: detect drift from plane residuals
(thresholds earned from the control rig), refine with point-to-plane LM, validate back under
the band — plus the domain's most dramatic number: wall-orientation diversity changing Hessian
conditioning by nine orders of magnitude.

**02.17 — LiDAR-camera fusion.** Complementary physics joined at a calibrated transform: the
parallax occlusion band derived from the baseline, the z-buffer earning its keep in color, and
edge bleeding measured at its true magnitude (boundary points 287× worse). The sensitivity
curve turns 01.17's motivating arithmetic into data.

## How to build & run

```powershell
projects\02-perception-lidar-point-clouds\02.14-moving-object-segmentation-from-sequential-scans\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.15-point-cloud-compression-for-fleet-uplink\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.16-multi-lidar-merging-extrinsic-refinement\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.17-lidar-camera-projection-coloring-fusion-kernels\demo\run_demo.ps1
```

## What to study here

The batch pairs well with its predecessors: 02.14 against 02.13 (online detection vs offline
removal — the same physics, two problem statements), 02.16 against 01.17 (target-based factory
calibration vs data-driven field refinement), 02.17 against 01.18 (the same projection core,
two products). Study 02.15's entropy-payoff redesign as the repo's cleanest example of
rebuilding a gate around what measurement actually shows. Exercise: run 02.14's mover mask into
02.13's carving accumulator and reason about whether online MOS can replace offline removal
(hint: the just-stopped car answers it).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-12), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (17/17, 21/21, 18/18, 14/14);
`tools/verify_project.py` all structural gates PASS.

- **02.14:** twins exact/near-exact; sign semantics asserted per cohort; statics 0/1,538;
  mitigation 50×; 1.75 ms vs 50 ms budget.
- **02.15:** seven verify stages bit-exact; distortion/bound ratio 0.9962; Huffman in band at
  8/8 rows; roundtrip exact. Builder disclosed one read-only sub-agent used to fact-check a
  citation the lead's own brief had suggested inaccurately — corrected; briefs tightened.
- **02.16:** recovery 0.0°/0.11 mm and 0.0°/0.31 mm; validation loop closes; observability
  contrast 9.8e9× gated; zero-drift control < 0.35 mm; dedup 8,135 = 7,003 + 1,132 exact.
- **02.17:** projection/z-buffer twins exact; occlusion cohort 89.1% → 0.7%; depth painting
  0.0 m vs independent re-derivation; sensitivity 1.09×/1.26× of analytic; accounting exact.

## Known limitations / TODOs

- 02.14: pose-quality coupling stated (MOS bounded by localization); MIN-fusion blind spot
  derived. 02.15: decode host-serial (block-wise parallel decode documented); uint32 occupancy
  simplification noted. 02.16: zone-assigned correspondences (piecewise-planar scene
  simplification, documented); loop consistency [info]-only pending pose-graph (05.xx).
  02.17: static-scene fusion (02.08 deskew assumed upstream); 5×5 visibility window
  over-filters near edges (the measured trade, Exercise 2).

## Next push preview

Batch 2k closes domain 02: 02.18 weather filtering (DROR/LIOR), 02.19 PointPillars/CenterPoint
voxelization + scatter, 02.20 LiDAR intensity calibration.
