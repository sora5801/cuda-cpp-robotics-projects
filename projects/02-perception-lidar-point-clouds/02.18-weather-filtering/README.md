# 02.18 — Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A LiDAR driving through snow, rain, or a dust cloud does not just see the road, walls, and cars —
it also sees the weather. Every falling snowflake, raindrop, and airborne dust particle that happens
to sit in a beam's path can scatter enough light back to register as a "real" return, and a naive
downstream consumer (a clustering step, an occupancy map, a costmap) has no way to tell that speckle
apart from an actual obstacle. This project builds three outlier filters that remove it: **SOR**
(Statistical Outlier Removal, the generic textbook baseline, included specifically to demonstrate
*why it fails* on a spinning LiDAR's naturally range-dependent point density), **DROR** (Dynamic
Radius Outlier Removal, Charron et al. 2018 — the fix, a search radius that grows with range), and
**LIOR** (Low-Intensity Outlier Removal, this project's teaching version of the intensity-based
filter family — weather returns are systematically dim). All three are implemented, GPU-accelerated,
and measured against a physically-derived synthetic scene with three independent weather conditions
(SNOW, RAIN, DUST) overlaid on the same real structure — see [`THEORY.md`](THEORY.md) for the full
Beer-Lambert-extinction forward model that generates it and [`scripts/make_synthetic.py`](scripts/make_synthetic.py)
for the code. A learner who finishes this project will understand not just *how* to remove weather
noise from a point cloud, but *why* the naive approach breaks specifically because of LiDAR beam
geometry, and where that break stops mattering (a dense enough scatterer cloud fools even the fix).

## What this computes & why the GPU helps

The computation is **brute-force all-pairs neighborhood search, run once per point**: for every point
in a scan, count how many other points fall within some radius (DROR: a radius that scales with that
point's own range; LIOR: a fixed radius), or find its K nearest neighbors and their mean distance
(SOR). This is the classic *map with an embedded reduction*: one GPU thread owns one point and does an
independent O(n) scan over every other point — n threads, each doing O(n) work, O(n²) total, with zero
communication between threads. It parallelizes perfectly (no point's neighborhood computation depends
on any other point's), and at this project's scale (roughly one to two thousand points per scan) the
whole O(n²) pass finishes in well under a millisecond on a desktop GPU (measured below) — fast enough
that a real pipeline could run it as a pre-clustering filter every scan, at full LiDAR rate.

## System context — where this sits in a robot

Full stack reference: [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md). Physical/
commercial grounding: [`PRACTICE.md`](PRACTICE.md).

- **Stack position:** perception (`SYSTEM_DESIGN.md` §1) — specifically the **point-cloud
  pre-processing boundary between the raw sensor and every downstream perception/mapping consumer**.
  It sits in the same pipeline slot as project 02.13 (dynamic point removal) and 02.03 (ground
  segmentation): a filter that runs on the raw scan BEFORE anything else trusts it.
- **Upstream inputs:** a raw `PointCloud` (`SYSTEM_DESIGN.md` §3.6: `xyz` + `intensity`) straight off
  the LiDAR driver, at the sensor's native scan rate. One honest line on what this project does NOT
  model: a real driver can report **dual returns** (both a weak weather echo and a stronger surface
  echo for the same beam) — this project's synthetic sensor and its filters are **single-return
  only** (see "Limitations").
- **Downstream consumers:** everything. A snowstorm of unfiltered false points chokes Euclidean
  clustering (project 02.04 — clusters explode into thousands of one-point "objects"), corrupts
  accumulated maps (project 02.13's map-maintenance ledger, project 05.01's TSDF fusion — false
  occupancy baked in forever), and terrifies a costmap-based planner (project 23.01 — every snowflake
  becomes a phantom obstacle the robot swerves to avoid, or worse, an emergency stop). Getting this
  filter right or wrong is the difference between an all-weather robot and one that refuses to leave
  the garage in a light flurry.
- **Rate/latency budget:** must fit inside the LiDAR→perception boundary, `SYSTEM_DESIGN.md` §1.1:
  **10-20 Hz**, sub-scan-period latency (50-100 ms). Measured on this project's committed scans (up
  to ~1,200 points/scan, brute-force O(n²)): all three filters combined finish in low single-digit
  milliseconds of GPU kernel time per scan (see "Expected output") — comfortable headroom at this
  point count; THEORY.md's "Where this sits in the real world" section discusses what changes at a
  production LiDAR's full point count (60,000-300,000+ points/scan).
- **Reference robot(s):** the autonomous-vehicle stack (`SYSTEM_DESIGN.md` §2.5 — winter/dust-storm
  operation is a real ODD, Operational Design Domain, boundary) and the warehouse/outdoor AMR
  (`SYSTEM_DESIGN.md` §2.1 — any site with an outdoor yard, loading dock, or seasonal weather).
- **In production:** DROR (Charron et al.) and its statistical-filter descendants ship in research and
  some production autonomy stacks as a pre-clustering step; intensity-based filters (this project's
  LIOR is a teaching version of that family) are common in industrial/agricultural LiDAR pipelines
  where calibrated intensity is already available. Autoware and PCL both ship SOR-family filters as a
  baseline (README "Prior art").
- **Owning team:** **perception** (`SYSTEM_DESIGN.md` §5.1) owns the LiDAR pipeline this filter lives
  in; it is a direct dependency for controls & autonomy's costmap/planning stack, which is why a
  regression here is a cross-team incident, not just a perception bug.

## The algorithm in brief

- **SOR (Statistical Outlier Removal)** — the K-nearest-neighbor mean distance, thresholded against
  the whole cloud's mean + `kSorStdMult` standard deviations. Included as the *baseline that fails*:
  it assumes uniform point density, which a spinning LiDAR's fixed angular sampling grid never gives
  you (THEORY.md "The problem" derives the 1/r² density falloff this project's `sor_far_range_failure`
  gate measures directly).
- **DROR (Dynamic Radius Outlier Removal, Charron et al. 2018)** — a search radius that GROWS with
  range, `r_search(r) = max(beta * alpha * r, r_min)`, derived from the sensor's own angular
  resolution (THEORY.md "The math"); a point is an outlier if too few neighbors fall within its own
  radius.
- **LIOR (Low-Intensity Outlier Removal)** — this project's teaching version of the intensity-based
  filter family: a point is an outlier if its intensity is below a threshold AND it is locally sparse
  within a small fixed radius (THEORY.md "The problem" derives why weather returns are physically dim
  — partial beam interception by a millimeter-scale particle).
- **The physics forward model** — a Beer-Lambert single-scatter extinction argument turns a particle
  density + cross-section + path length into a per-beam scatter probability, used to generate the
  synthetic SNOW/RAIN/DUST scans this project's filters are measured against (THEORY.md "The problem").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/weather-filtering.sln`](build/weather-filtering.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/weather-filtering.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none. Only the CUDA runtime + C++17 standard library
(CLAUDE.md §5 default budget) — all three filters and their brute-force radius/KNN search are
hand-rolled, no cuBLAS/Thrust/etc. (that acceleration structure is project 02.05's/02.09's job, cited
throughout).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Fully synthetic (CLAUDE.md §8 default): three independent weather captures (SNOW/RAIN/DUST) of the
SAME static scene (a ground plane, two walls at ~8 m and ~32 m, and a parked car), each ray-cast by a
real 16-beam spinning LiDAR model and overlaid with a Beer-Lambert-extinction scatterer field.
Generated by [`scripts/make_synthetic.py`](scripts/make_synthetic.py) (fixed seed 42, xorshift32 +
Box-Muller noise) — no public dataset used; see that script for why (the short version: recovering an
exact per-point real/scatterer ground-truth label from a real weather-LiDAR dataset like CADC or WADS
would need hand-annotation this repo cannot do automatically, and CADC in particular carries a
non-commercial research license — `scripts/download_data.ps1`/`.sh` state this decision in full).
Full field documentation, checksums, and provenance: [`data/README.md`](data/README.md).

## Expected output

**The pipeline (from an actual run on an RTX 2080 SUPER, CUDA 13.3, Release|x64 — every number below
is measured, never invented):**

- **VERIFY (six GPU-vs-CPU gates, run on the SNOW scan, n=1,074 points — representative: no kernel
  branches on which weather scan it is fed):**
  - SOR mean-KNN-distance: worst `|gpu-cpu|` = `9.5e-07` m over 1,074 points (tol 1e-3 m) — then
    classify (same host-computed threshold fed to both paths): 0 mask mismatches.
  - DROR neighbor count: 0 mismatches of 1,074 points (exact integers) — then classify: 0 mismatches.
  - LIOR neighbor count: 0 mismatches of 1,074 points (exact integers) — then classify: 0 mismatches.
- **DROR / LIOR precision+recall floors (snow + rain cohorts, all PASS):**

  | Filter | Cohort | Precision | Recall |
  |--------|--------|-----------|--------|
  | DROR | snow | 96.3% | 97.0% |
  | DROR | rain | 95.6% | 97.7% |
  | LIOR | snow | 65.1% | 91.1% |
  | LIOR | rain | 79.6% | 92.1% |
  | SOR (reported, not gated) | snow | 74.4% | 66.7% |
  | SOR (reported, not gated) | rain | 84.8% | 52.6% |

- **`real_point_preservation`** (real points correctly KEPT, all 3 scans combined): DROR **95.6%**,
  LIOR **91.6%** (floors 85%/80%).
- **`sor_far_range_failure`** (the headline lesson, both directions asserted in one gate): of 119
  far-range (>= 25 m) real points across all 3 scans, **SOR falsely removes 60.5%** of them (floor
  35%) while **DROR falsely removes only 5.9%** of the SAME cohort (ceiling 15%).
- **`dust_plume_honesty`** (measured, NOT floor-gated by design): inside the dust plume's 777-point
  core, DROR scores precision 98.2% / recall 94.6%; LIOR scores precision 98.8% / recall 63.3% — LIOR
  is the one that struggles more here, a genuinely non-obvious finding this project measures rather
  than assumes (THEORY.md "Numerical considerations" explains why: LIOR's fixed companion radius is
  larger than DROR's own range-scaled radius at the plume's short range, so it saturates into "looks
  dense enough to keep" sooner as scatterer density rises).
- **`[info]` combined** (production practice): union(DROR, LIOR) raises recall to 98.5%/98.1%/94.9%
  (snow/rain/dust) at some precision cost; intersection raises precision to 96.8%/97.2%/89.0% at some
  recall cost.
- **`[info]` intensity_dependence**: LIOR recall on the snow scan drops from 91.1% (clean, as-
  calibrated intensity) to 85.2% under a documented miscalibration perturbation — the project 02.20
  dependency, quantified rather than just asserted.
- **Timing** (teaching artifact, not a benchmark): all 3 scans, all 3 filters, six kernel launches per
  scan — **~2.3-2.5 ms total GPU kernel time**, single-shot, one machine.

Tolerance: SOR's statistic (a floating-point sum of square roots) is compared with a tight 1e-3 m
tolerance; every other VERIFY comparison (DROR/LIOR neighbor counts, and every classify stage given
its already-verified input) is **exact** — see THEORY.md "How we verify correctness" for why each
tolerance is what it is.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — read this FIRST: all three filters' contracts (the point-
   record layout, every parameter, the dynamic-radius formula) are documented once, here.
2. [`src/kernels.cu`](src/kernels.cu) — the six GPU kernels: `sor_mean_knn_dist_kernel` (the K-nearest
   search), `dror_neighbor_count_kernel` (the heart of the project — the range-scaled radius),
   `lior_neighbor_count_kernel` (the deliberate fixed-radius contrast), and their three classify
   kernels.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independently-typed CPU twins; read side by
   side with `kernels.cu` to see exactly what stayed shared (the formulas) and what was retyped (the
   search loops).
4. [`src/main.cu`](src/main.cu) — orchestration: load data, the VERIFY stage, 12 gates, 3 artifacts.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the physics forward model that generates
   the committed sample; its module docstring is the didactic derivation of everything DROR/LIOR
   exploit.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `find_data_file`/`resolve_out_dir`.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1):

- **Charron, Phillips & Waslander, "De-noising of Lidar Point Clouds Corrupted by Snowfall" (2018)**
  — the original DROR paper; this project's dynamic-radius formula is a direct, hand-rolled
  transcription of theirs.
- **Rusu, "Semantic 3D Object Maps for Everyday Manipulation in Human Living Environments" (PhD
  thesis, 2009) / PCL's `StatisticalOutlierRemoval`** — the SOR baseline this project's SOR filter
  reimplements and deliberately stress-tests.
- **Kurup & Bos, "DSOR: A Scalable Statistical Filter for Removing Falling Snow from LiDAR Point
  Clouds in Severe Winter Weather" (2021)** — a modern descendant that adds a dynamic distance
  threshold in the SOR family's spirit; study it for how production filters keep evolving past DROR.
- **The CADC (Canadian Adverse Driving Conditions) and WADS (Winter Adverse Driving dataSet) datasets**
  — the real snow-LiDAR data this project's synthetic scene stands in for (see "Data" above and
  `scripts/download_data.ps1` for why they are not used directly here).
- **Autoware.Universe's `pointcloud_preprocessor`** — a production autonomy stack that ships a
  DROR-family filter as a real pipeline node; study its parameter tuning and node topology.
- **Project 02.20 (LiDAR intensity calibration across channels)** — this project's forward-looking
  sibling: LIOR's intensity threshold assumes calibrated, range-compensated intensity, which is
  exactly what 02.20 studies in depth (not yet built as of this writing).

## Exercises

1. Re-run [`scripts/make_synthetic.py`](scripts/make_synthetic.py) with `SOR_STD` (in `kernels.cuh`,
   `kSorStdMult`) raised to 1.0 (PCL's typical default) and re-measure `sor_far_range_failure` — does
   the far/near contrast wash out? Why (THEORY.md "How we verify correctness" discusses the sweep that
   picked 0.5)?
2. Plot `demo/out/range_stratified.csv`'s false-removal columns against range band — the 1/r² density
   story, made visible.
3. Implement the UNION(DROR, LIOR) mask as a fourth first-class filter (main.cu already computes it
   for the `[info]` combined line) and add it as a gated filter in its own right — what floor would you
   set, and why?
4. `kernels.cuh`'s `kDrorAlphaRad` is hardcoded to the azimuth step. Real LiDARs have DIFFERENT
   azimuth and elevation angular resolutions — extend DROR to use the tighter of the two (THEORY.md
   "The math" discusses which one Charron et al.'s own formula uses and why).
5. Implement a tiled-into-shared-memory version of `dror_neighbor_count_kernel` (kernels.cu names the
   tradeoff at this project's point count) and measure whether it actually helps at n ~ 1,000-1,500
   points, or whether the L1/L2 cache already captures the reuse.

## Limitations & honesty

- **Sensor stays at the origin for all three weather captures** — a deliberate scope cut
  (`scripts/make_synthetic.py`'s module docstring): this project compares three INDEPENDENT captures
  of the identical static scene under three atmospheres for a clean A/B/C comparison, not one
  continuously moving platform observing changing weather in real time.
- **Single-return only** — a real dual-return LiDAR can report both a weak weather echo and a
  stronger surface echo for the same beam; this project's synthetic sensor and its filters model
  single-return only (a scatter event pre-empts the real surface return entirely). A dual-return
  extension is a natural next project.
- **Brute-force O(n²) search, no spatial index** — deliberate scope, not an oversight: at this
  project's point counts (roughly 1,000-1,500 points/scan) brute force finishes in well under a
  millisecond; a production LiDAR's full point count (60,000+/scan) would need the spatial
  acceleration structure project 02.05/02.09 build (cited throughout `kernels.cuh`).
- **Illustrative physics constants, dated 2026-07-12** — particle radii, number densities, and
  backscatter reflectances in `scripts/make_synthetic.py` are ORDER-OF-MAGNITUDE choices, not
  measured atmospheric physics (real values need Mie scattering theory at the sensor's wavelength,
  THEORY.md "Where this sits in the real world"); several were TUNED (documented at their source) so
  the three weather conditions produce comparably-sized, clearly-separated cohorts for a legible demo
  — the same "measured-then-margined, tuning stated honestly" discipline project 02.13 uses.
- **Not safety-certified.** Nothing here commands real hardware; if a production version of this
  pipeline ever fed a costmap that a real robot planned around, the usual sim-validated-only caveat
  applies in full (CLAUDE.md §1).
