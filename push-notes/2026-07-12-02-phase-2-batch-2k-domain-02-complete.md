# Push note — 2026-07-12-02: batch 2k — domain 02 complete (20/20)

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2k (**78/505**) finishes **domain 02 — LiDAR & point clouds — in full: 20 of 20 projects
done**, the second domain completed under Phase 2. The closers: weather filtering (02.18)
derives snow-hit probabilities from a Beer-Lambert scatterer model and lands the domain's oldest
lesson — the 1/r² density trap — as a gated two-directional assertion (generic SOR falsely
kills 60.5% of far-range real points; physics-derived DROR kills 5.9%); PointPillars
voxelization (02.19) bridges to learned detection with the determinism thread's ML-preprocessing
finale (atomic slot-claiming changes *which points survive* under reordering — measured — while
the sorted path keeps the analytic answer bit-identically) and closes end-to-end without
TensorRT per the §5 pattern; and intensity calibration (02.20) replays 01.09's flat-field story
on 16 laser channels — gauge freedom taught, observability stated as graph connectivity (the
disconnected channel *flagged*, not hallucinated), and 02.18's measured LIOR dependency closed
with a 130/308 decision-flip experiment. The domain now reads as one course: raw points →
downsampled, segmented, clustered, searched, registered, deskewed, described, recognized,
cleaned, compressed, merged, fused, weather-proofed, ML-ready, and calibrated.

## What changed

- **[projects/02-perception-lidar-point-clouds/02.18-weather-filtering/](../projects/02-perception-lidar-point-clouds/02.18-weather-filtering/)** —
  SOR/DROR/LIOR on a Beer-Lambert scatterer forward model; snow/rain P/R gates; the dust-plume
  hard case measured (DROR 94.6% vs LIOR 63.3% recall in the core, mechanism explained); LIOR's
  calibration dependency quantified (−5.9 pp) as the 02.20 handoff.
- **[projects/02-perception-lidar-point-clouds/02.19-pointpillars-centerpoint-voxelization-scatter/](../projects/02-perception-lidar-point-clouds/02.19-pointpillars-centerpoint-voxelization-scatter/)** —
  9-D pillar features + fixed PFN-lite + NCHW scatter + designed-response head; cap-truncation
  determinism study; 6/6 detection closure, zero false peaks; TRT tensor contract documented,
  zero fabricated numbers.
- **[projects/02-perception-lidar-point-clouds/02.20-lidar-intensity-calibration-across-channels/](../projects/02-perception-lidar-point-clouds/02.20-lidar-intensity-calibration-across-channels/)** —
  log-linear per-channel gain solve over the shared-voxel graph; gains to 5.8% worst; spread
  collapse 5.5×; unobservable-channel assertion; the LIOR decision-flip demo.
- **[docs/STATUS.md](../docs/STATUS.md)** — 02.18–02.20 → `done` (**78/505; domain 02: 20/20**).

## New projects (didactic blurbs)

**02.18 — Weather filtering.** Precipitation as physics, not noise: hit probabilities from
cross-section × density × path length, weak returns from partial beam interception, and the
reason DROR's radius *must* grow with range derived from the same beam geometry that 02.01
taught. The dust plume is the honest hard case: dense enough scatter statistically resembles a
surface.

**02.19 — PointPillars/CenterPoint preprocessing.** The impedance mismatch between sparse
irregular LiDAR and dense regular tensors, bridged by layout engineering — and the repo's
determinism doctrine arriving where it matters most commercially: reproducible ML features.
Max-pool is permutation-invariant *except when the cap truncates*; the interaction is taught,
measured, and solved by ordering.

**02.20 — Intensity calibration.** Sixteen lasers, one world: relative gains from shared
observations with no targets, the gauge fixed, the unobservable flagged. The before/after
histograms are the visual; the 130 flipped LIOR decisions are the proof that calibration is not
cosmetic.

## How to build & run

```powershell
projects\02-perception-lidar-point-clouds\02.18-weather-filtering\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.19-pointpillars-centerpoint-voxelization-scatter\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.20-lidar-intensity-calibration-across-channels\demo\run_demo.ps1
```

## What to study here

Domain 02 closes with its cross-project loops actually closed: 02.18 measured a dependency and
02.20 resolved it; 02.19 produces exactly what 12.01 consumes; 02.16 refines what 01.17
calibrates. Read the domain's determinism thread end to end (02.01 ordering → 02.02 operators →
02.04 canonical form → 02.19 ML features) — it is the repo's most complete single-theme course
so far. Exercise: chain 02.18's DROR output into 02.19's pillarization and measure what a
snowstorm does to the detection-closure gate with and without filtering.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-12), each project re-verified independently by the lead after the builder's
self-gate — all three: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (24/24, 15/15, 10/10);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder.

- **02.18:** twins exact; DROR 96/97% P/R snow, 96/98% rain; preservation 95.6%; the SOR
  far-range gate asserted both directions; range-stratified table written.
- **02.19:** feature/PFN/scatter/head twins ≤3.8e-6; roundtrip bit-identical; sorted cap
  bit-identical ×3 vs atomic 32/30/20 differing slots shuffled; closure 6/6, zero false peaks.
- **02.20:** four verify stages ≤3e-6; gains 5.8% worst (gauge-aligned); collapse 5.5×;
  multi-material delta 4.4 pp; channel 15 flagged unobservable with 15/15 others recovered.
  Two real bugs found by gates (voxel-boundary graph cut; weakly-linked cluster conditioning).

## Known limitations / TODOs

- 02.18: brute-force O(n²) neighbor search (spatial indexing ceded to 02.05/02.09 by design);
  physics constants illustrative and dated. 02.19: designed-response head (not learned — the
  honesty that keeps it a preprocessing project); FP16/INT8 documented-only. 02.20: single-scan
  self-calibration (temporal aggregation documented); committed scene stays in the 1/r² regime
  (plateau taught, not exercised).

## Next push preview

Domain 03 — radar, sonar & event cameras (12 remaining after the 03.01 FMCW+CFAR flagship):
batch 2l opens with 03.02 and proceeds in ★-then-ID order.
