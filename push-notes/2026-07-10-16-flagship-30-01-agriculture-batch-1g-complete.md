# Push note — 2026-07-10-16: flagship 30.01 agriculture batch 1g complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 30.01 — **agriculture, milestone 1: fruit detection + 3-D localization + ripeness** — is
done, closing **batch 1g** (27.04, 28.01, 29.05, 30.01; **32/505 overall, 32 of 36 flagships**).
This is the repo's bundled-bullet showcase: the catalog packs seven agriculture components into
one bullet, and per the §2 rule this project implements milestone 1 fully (classical HSV →
morphology → GPU connected-component labeling → 3-D back-projection → ripeness on synthetic RGB-D
orchard scenes) while the other six ship as documented milestones — with `fruit_map.csv`
explicitly seeding milestone 7's yield map. Two verification designs stand out: the CCL check runs
**two different algorithms** (GPU label propagation vs CPU union-find) canonicalized to exact
integer equality; and the build *derived* a real optics correction — a camera sees a sphere's
near surface, biasing depth by 2R/3 — which cut localization error from ~2.7 cm to **1.8 mm
mean**. Occlusion honesty throughout: 20/24 detectable fruit found (gate ≥80%), and both false
positives are the *designed* cross-depth merge pairs, identified by fruit ID.

## What changed

- **[projects/30-field-robotics/30.01-agriculture/](../projects/30-field-robotics/30.01-agriculture/)** —
  complete: HSV/mask/morphology kernels, iterative label-propagation CCL (order-independence
  argued), atomics-based component stats, near-surface-corrected 3-D localization, ripeness
  scoring, CPU oracle (different-algorithm CCL), detection/localization/ripeness gates,
  detections PGM + fruit-map CSV artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 30.01 → `done` (**32/505**).

## New projects (didactic blurbs)

**30.01 — Agriculture milestone 1** (★ beginner, domain 30, flagship). The classical perception
pipeline that teaches the geometry deep-learning production systems stand on: why HSV separates
ripeness from shading, how label propagation converges to a unique fixed point, and where a
3-D localization error budget actually comes from (depth quantization + centroid bias + the
derived near-surface offset). Honest framing: deep detectors are production reality; this is the
didactic baseline that makes their outputs geometrically meaningful. The single most interesting
thing to look at: `demo/out/detections.pgm` — 22 rings on the orchard, including the two honest
merge-case "mistakes" the data was designed to cause.

## How to build & run

```powershell
projects\30-field-robotics\30.01-agriculture\demo\run_demo.ps1
# then open demo\out\detections.pgm and read demo\out\fruit_map.csv
```

## What to study here

Batch 1g as a set spans materials → soft bodies → medical imaging → field perception: the
repo's breadth argument in four projects. Within 30.01: `THEORY.md`'s localization error budget
(the near-surface derivation) and the CCL convergence sketch → `src/kernels.cu`. First exercise:
lower the ripeness floor below 0.35 and watch the green-on-green case become honestly unsolvable
— then read why spectral sensing exists.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 10 stable lines matched, exit 0.
- **GPU-vs-CPU gates:** HSV bit-exact (trig-free design); mask exact (0/307,200); CCL labels
  exact after canonicalization (0/7,059) across two different algorithms; stats 4.3e-07.
- **Ground-truth gates:** 20/24 detectable fruit (83% vs ≥80%), 2 false positives = the designed
  merge pairs; localization mean 1.8 mm / max 6.9 mm (gate 15 mm); radius mean 0.9 mm (gate
  6 mm); ripeness Spearman ρ = 0.998 (gate ≥0.70).
- Timing (teaching artifact): GPU pipeline ≈ 4.5 ms (CCL-dominated, host-round-trip honesty).
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Milestone 1 of 7 (the other six documented per the bundle rule — README §13 lists implemented
  vs documented-only); classical color pipeline as the didactic baseline (deep detectors named as
  production); ripeness scoped to hue-resolvable range (the green-on-green honesty).

## Next push preview

Batch 1h — the final four flagships: 32.02 CUDA Graphs control loop, 34.03 ergodic control (FFT),
35.01 magnetic microswarm fields, 36.03 lattice-robot kinematics. Then the §11 standards
retrospective push.
