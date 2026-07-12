# 02.14 — Moving-object segmentation from sequential scans

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Moving-object segmentation from sequential scans`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project answers one question a spinning LiDAR must answer **every single scan, in real time**:
*of the points I just measured, which ones belong to something moving right now?* It does this
**online**, from a short buffered window of the CURRENT scan plus its **M=4** immediately preceding
scans — never from a long map history. The method is the classical (non-learned) LiDAR-MOS lineage
(Chen et al., IROS 2021): organize each scan into a 16x360 **range image** (02.12's thesis — image
neighbors are sensor-sphere neighbors), **reproject** each previous scan into the current sensor's
frame (02.08's deskew formula, reused here across scans instead of within one sweep), take the
**signed residual** between current and reprojected-previous range at every cell, **fuse** the M
residuals with a MIN rule that is deliberately resistant to one-off disocclusion artifacts, and clean
up the surviving candidates with a range-image connected-component filter (02.12's union-find
lineage). The demo grades the result against a designed scene with **four differently-moving cars**
(a lateral crosser, a pure radial approach, a pure radial departure, and a car that just stopped) plus
two static "honesty" cohorts (a long wall and a thin pole), and writes a residual-image PGM where
movers visibly glow.

This project is the **online, dual** of sibling project [02.13](../02.13-dynamic-point-removal/) —
see "The algorithm in brief" below for the explicit contrast.

## What this computes & why the GPU helps

The computation is a chain of four **per-cell / per-point MAPs and one SCATTER**, run once per
incoming scan: (1) scatter the current scan's points into a range image (nearest-wins per cell); (2)
for each of the 4 previous scans, reproject every point into the current frame and scatter it into
its own range image — 4 independent, embarrassingly parallel scatters; (3) for every current-scan
cell, a MAP over its own cell computing the fused residual against up to 4 other images; (4) a range-
image adjacency SCATTER (build "both cells look like a mover" edges) followed by a small, generic
GPU union-find (the repo's lock-free path-halving sweep, reused from 02.04/02.12) to remove speckle.
Every stage maps one GPU thread to one point or one cell, with zero cross-thread dependency except
the deliberately tiny, provably-converging union-find sweep — the same shape 02.02/02.12/02.13 use,
because a spinning LiDAR's own beam organization is *the* natural unit of GPU parallelism for this
whole family of algorithms.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception, right where LiDAR point-cloud processing (domain 02) hands off to
  state estimation/tracking (04) and mapping (05) — SYSTEM_DESIGN.md §1's `SENSORS -> PERCEPTION ->
  STATE ESTIMATION` boundary. Online MOS is the **fork point**: every current-scan point leaves this
  module already tagged mover-or-static.
- **Upstream inputs:** posed, organized LiDAR scans — a `PointCloud` message per scan (SYSTEM_DESIGN.md
  §3.6) plus its `T_world_lidar` pose, exactly the output of **02.07** (ground-plane-removed organized
  scans) and **02.08** (per-point motion-deskewed scans, named by ID because this project's reprojection
  formula literally IS 02.08's deskew formula applied across scans instead of within one sweep).
- **Downstream consumers — TWO lanes, split by this project's own output label:**
  - **Movers -> tracking & prediction.** Points labeled MOVING are the natural input to a multi-object
    tracker (domain **04.xx**, e.g. a multi-hypothesis or JPDA tracker) and from there to
    SYSTEM_DESIGN.md's **Prediction** layer ("where will things be in 1-10 s?") — a mover this project
    finds is a mover the planner needs a forecast for.
  - **Statics -> mapping.** Points labeled STATIC are safe to fuse into a persistent map (domain
    **05.xx**, e.g. TSDF fusion or an occupancy grid) — this is the ONLINE, per-scan complement to
    **02.13**'s OFFLINE dual: 02.13 cleans a map after the fact from a long history; this project keeps
    movers out of the map from the moment each scan arrives, using only a short window.
- **Rate / latency budget:** MUST run once per incoming scan at the sensor's native **10-20 Hz**
  (SYSTEM_DESIGN.md §1.1's "LiDAR -> perception/mapping" boundary: < 1 scan period, 50-100 ms) — a
  stale mover/static split is worse than none, because the tracker and the map both start trusting a
  wrong label. `main.cu`'s `timing` gate measures the canonical M=4 pass's actual GPU kernel time
  against a 50 ms (20 Hz) budget on this project's demo scale (measured: ~1-4 ms — comfortable margin
  at this cell count; THEORY.md "The GPU mapping" discusses how this would change at a real sensor's
  full point count).
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN.md §2.1: "POINT-CLOUD PERCEPTION [02]"
  feeding "LOCALIZATION & MAPPING [04,05]" — exactly this project's fork) and the **autonomous-vehicle
  stack** (§2.5: "LIDAR [02]" feeding "FUSION & TRACKING [04]" — both reference robots operate *among
  people*, where a missed mover is the failure mode that matters most).
- **In production:** a shipping stack replaces this project's fixed-threshold classical residual test
  with a learned model trained end-to-end on exactly this range-image-residual representation
  (LiDAR-MOS/4DMOS — THEORY.md "Where this sits in the real world"), and often fuses in a semantic
  prior (person/vehicle detectors) alongside the geometric test this project teaches.
- **Owning team:** Perception (SYSTEM_DESIGN.md §5.1: domains 01/02/03/20) builds and tunes this
  module; its output is a direct dependency of the Controls & Autonomy team's tracker and mapper.

## The algorithm in brief

- **Range-image organization** of the current scan by native (ring, azimuth) — 02.02/02.12's
  nearest-wins encoded-atomicMin scatter, reused verbatim (`kernels.cuh` "Range-image geometry").
- **Cross-scan reprojection**: 02.08's rigid pose-composition deskew formula, applied here between
  DIFFERENT SCANS (current <- previous) instead of within one continuous sweep — see
  [THEORY.md "The math"](THEORY.md#the-math).
- **Signed residual + the two-sided sign derivation**: arrival (negative) vs. departure/revealed-
  background (positive) — see [THEORY.md "The math"](THEORY.md#the-math), verified independently by
  the `sign_semantics` gate.
- **Multi-scan MIN-fusion**, the disocclusion-resistant evidence rule — see
  [THEORY.md "The algorithm"](THEORY.md#the-algorithm) and `kernels.cu`'s `residual_fuse_kernel` comment.
- **Range-image connected-component cleanup**, 02.12's generic lock-free GPU union-find reused over a
  brand-new edge predicate — see [THEORY.md "The GPU mapping"](THEORY.md#the-gpu-mapping).
- **The offline dual, named explicitly**: [02.13](../02.13-dynamic-point-removal/) answers "which map
  voxels were only ever transiently occupied, over a long history?"; this project answers "which
  CURRENT points are moving right now, from a short window?" — contrasted in full in
  [THEORY.md "The problem"](THEORY.md#the-problem--physics--engineering-first).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/moving-object-segmentation-from-sequential-scans.sln`](build/moving-object-segmentation-from-sequential-scans.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/moving-object-segmentation-from-sequential-scans.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none. This project uses only the CUDA runtime + C++17
standard library (CLAUDE.md §5's default dependency budget) — no cuBLAS/cuFFT/Thrust/CUB; every
kernel (including the generic union-find sweep) is hand-rolled.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Synthetic-only (CLAUDE.md §8 default): `scripts/make_synthetic.py` ray-casts a real 16-beam spinning
LiDAR (closed-form ray/box and ray/cylinder intersection, so ground truth is exact) through an
analytic scene across 5 sensor positions, producing a 5-scan window with per-point ground truth —
including a `disocclusion_band` flag computed by re-ray-casting from every scan's own sensor position
(the exact, analytic definition the `disocclusion_mitigation` gate reads). No public dataset is used:
online MOS needs an object-level, per-window ground truth (which points are moving *right now*) that
no public LiDAR benchmark hands you directly at this granularity. Full provenance, checksums, and
per-column documentation: [`data/README.md`](data/README.md).

## Expected output

**Verification (four independent GPU-vs-CPU stages, CLAUDE.md §5):**

| Stage | Comparison | Tolerance | Measured (this run) |
|-------|-----------|-----------|----------------------|
| Current-scan organize | GPU vs. CPU range image + ground-truth payload | bit-exact (no trig) | 0 mismatches / 5,760 cells |
| Reprojection (x4 previous scans) | GPU vs. CPU reprojected range image | 5 mm, <=2% of populated cells | 0/6,972 outside tolerance |
| Residual fusion (M=4) | GPU vs. CPU, fed the SAME verified range images | bit-exact (pure arithmetic) | 0 mismatches / 5,760 cells |
| Range-image CCL | GPU vs. CPU edge set (set-equality) + union-find (bit-exact partition) | exact | edge sets identical; 0 root mismatches |

**Gates (independent of the twin comparisons above — checked against ground truth the algorithm never
sees):**

| Gate | Bound | Measured (this run) |
|------|-------|----------------------|
| `mover_detection` (per-cohort recall + overall IoU) | crossing >=90%, oncoming >=75%, receding >=90%, IoU >=65% | 98.5% / 87.9% / 100.0% / IoU 74.4% |
| `sign_semantics` (oncoming negative, receding positive) | >=80% of each cohort | 87.9% / 100.0% |
| `static_precision` (WALL+POLE false-positive rate) | <=5.0% | 0.0% (1,538 points) |
| `disocclusion_mitigation` (FP-rate improvement, M=1 vs. M=4, wall disocclusion band) | >=3.0x | 50.00x (26.4% -> 0.0% of 261 points) |
| `timing` (full canonical M=4 MOS pass) | < 50 ms (20 Hz budget) | ~1-4 ms |

**[info]-only measurements (honestly reported, not gated):**

- `temporal_boundary`: the `stopped_car` cohort (moving through scans 0-3, held stationary between the
  last previous scan and now) is recalled at **0.0%** for every window size M=1/2/4 — a *derived*
  property of MIN-fusion (kernels.cu's `residual_fuse_kernel` comment), not a bug: the freshest
  (lag-1) comparison alone already reads near-zero once an object stops, and MIN always includes it.
- `window_size` study: overall recall/precision/IoU at M=1 (72.5%/44.6%/38.2%) vs. M=2
  (74.4%/95.1%/71.6%) vs. M=4 (74.4%/100.0%/74.4%) — more history buys precision, not recall, at the
  cost of buffering M scans of latency.

Artifacts (`demo/out/`, git-ignored): `residual_image.pgm` (fused |residual|, movers glow),
`label_vs_truth.ppm` (a confusion map: TP green / FP red / FN orange / TN gray),
`disocclusion_band.ppm` (the wall's disocclusion-band cells highlighted magenta), plus
`per_cohort_metrics.csv` and `gates_metrics.csv`. Canonical stable lines in
[`demo/expected_output.txt`](demo/expected_output.txt); numbers above are `[info]`-line measurements,
never diffed (CLAUDE.md §12).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — start here: the file header derives the whole method in five
   steps, then the range-image/pose-algebra contract every other file shares.
2. [`src/main.cu`](src/main.cu) — orchestration: load the 5-scan window, run the four verify stages,
   run the five gates, write the artifacts. Read the file header's "What this program does" first.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels; `residual_fuse_kernel`'s comment is the most
   interesting one in the project (it derives *why MIN*, with the disocclusion-mitigation trade-off
   spelled out in full).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the scene: four cars, each teaching a
   different facet of the residual sign/fusion story, plus the disocclusion-band ground-truth derivation.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **LiDAR-MOS** (Chen, Mersch, Behley et al., IROS 2021 / RA-L) — the range-image-residual
  representation this project implements classically; the paper trains a small CNN directly on the
  residual images this project computes by hand.
- **4DMOS** (Mersch et al., 2022) — the sequel: a spatio-temporal (4-D sparse conv) network that
  remembers motion state across a LONGER window than 4 scans, directly addressing this project's
  measured `temporal_boundary` limitation (a just-stopped object).
- **Removert** (Kim & Kim, 2020) — this project's own sibling **02.13**'s closest real-world analogue:
  offline, multi-resolution range-image comparison for map cleaning (the OFFLINE dual named throughout
  this README).
- **RAFT-3D / scene flow** (see **01.21** in this repo) — the camera-side dual of this project's
  disocclusion problem; 01.21's THEORY.md derives the SAME occlusion-boundary false-positive mechanism
  in the RGB-D/optical-flow domain, cited directly in this project's THEORY.md.
- **PCL / OpenCV CUDA** — production point-cloud and image toolkits with range-image and connected-
  component primitives comparable to this project's hand-rolled versions.
- **nvblox** — NVIDIA's GPU-accelerated mapping library; a production target for this project's
  "statics" output lane.

## Exercises

1. **Vote fusion.** Swap `residual_fuse_kernel`'s MIN rule for a majority-vote rule (>= K of M
   comparisons exceed threshold) and re-run the `window_size` / `disocclusion_mitigation` gates — does
   vote recover the `stopped_car` cohort that MIN misses? What does it cost on the disocclusion band?
2. **A fifth scan.** Extend `kMaxWindowM` to 8 and regenerate a longer synthetic sequence — does the
   `window_size` study's precision-vs-recall trade continue, or plateau?
3. **Non-identity orientation.** Give the sensor a slow yaw sweep in `make_synthetic.py` (the pose
   algebra in `kernels.cuh` is already fully general) and confirm the reprojection VERIFY stage still
   passes — this exercises the quaternion path the demo's identity-orientation scenario never does.
4. **Pose-error sensitivity, measured.** THEORY.md "Numerical considerations" derives the coupling
   between localization quality and MOS quality analytically; add a small synthetic bias to the
   previous-scan poses fed into `reproject_scatter_cpu` (leave the canonical GPU path untouched) and
   measure how `static_precision` degrades — turn the derivation into a live number.
5. **Learned MOS.** Dump this project's residual images as a small labeled dataset (the ground truth
   is already computed) and train a tiny CNN to reproduce LiDAR-MOS's classification — compare its
   `mover_detection` numbers against this project's fixed-threshold gates.

## Limitations & honesty

- **Fixed threshold, not learned.** `kDynamicThresholdM` is a single constant, derived from range
  noise and measured against this scene (THEORY.md "The math" / "Numerical considerations") — a real
  LiDAR-MOS deployment trains a network on exactly this residual representation instead (see "Prior
  art").
- **MIN-fusion cannot recover a just-stopped mover**, by construction (the `temporal_boundary`
  [info] measurement, 0% recall at every M) — a documented, derived property, not a bug; 4DMOS's
  learned temporal memory is the production fix (see "Prior art").
- **Insufficient-evidence cells default to STATIC.** A cell with no valid comparison in ANY included
  previous scan cannot be proven to have moved and is conservatively left unlabeled-as-static (the
  same "no evidence -> no removal" stance 02.13 takes) — a newly-revealed area is invisible to this
  method until it has been observed at least twice.
- **Cars are axis-aligned boxes, not heading-oriented.** The synthetic scene's dynamic objects are
  AABBs regardless of travel direction (02.13's identical simplification) — real object detectors
  recover oriented bounding boxes.
- **Sensor orientation is identity throughout the demo scenario** (translation-only ego-motion,
  02.13's identical documented scope cut) — the reprojection algebra itself is fully general
  (Exercise 3 exercises the cut path).
- **Small demo scale.** 5,760 range-image cells is tiny next to a real sensor's 100k+ points/scan; the
  `timing` gate's wide margin (~1-4 ms against a 50 ms budget) is a teaching artifact of this scale,
  not a production benchmark (THEORY.md "The GPU mapping" discusses how occupancy/bandwidth would
  matter more at full resolution).
- **Sim-validated only.** Every number in this README comes from the committed synthetic scene. If
  this classifier's mover/static split were ever wired into a real robot's planner or map, it would
  need real-sensor validation and is **not** safety-certified (CLAUDE.md §1, §8) — see
  [PRACTICE.md](PRACTICE.md) §4 for the stakes of getting the split wrong in either direction.
