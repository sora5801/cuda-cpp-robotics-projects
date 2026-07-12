# 02.16 — Multi-LiDAR merging + extrinsic refinement

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Multi-LiDAR merging + extrinsic refinement`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Most real robots big enough to need LiDAR at all carry more than one: a 360° roof unit plus one or
two corner units filling in the blind spot behind it (kernels.cuh's rig diagram). This project
teaches the two problems that creates. **Merging** — turning three separate point clouds, each in
its own sensor's frame, into one coherent cloud in the vehicle's frame — sounds trivial (transform
and concatenate) but has a real teaching problem hiding in it: the same physical surface gets seen
by two or three sensors in the overlap zones, and unless you deduplicate, your merged cloud has
phantom double density exactly where it matters most. **Extrinsic refinement** is the harder, more
interesting half: the transform you use to merge each sensor's cloud is only as good as its
calibration, and LiDAR mounts drift — vibration and thermal cycling loosen a bracket a fraction of a
degree and a few millimeters over months of fleet service. This project detects that drift from nothing
but the overlap geometry itself (two sensors' fitted planes of the same wall disagreeing is the
signal), refines it away with a point-to-plane optimization closely related to ICP, and validates
that the fix actually worked — a closed loop, run end to end on a synthetic rig with a known,
planted answer.

**What's implemented vs. documented-only.** Both catalog components — merging and extrinsic
refinement — are fully implemented and gated (see "Expected output" below for the ten independent
verification stages). What is *not* implemented is a full multi-way pose-graph solve across more
than two sensors at once (05.xx's job, named explicitly in THEORY.md "Where this sits in the real
world"); this project's "loop consistency" stage states that harder problem didactically ([info]
only, not gated) rather than solving it.

## What this computes & why the GPU helps

Four GPU-parallel stages, each a different point on the map/reduce spectrum (THEORY.md "The GPU
mapping" argues the full story): (1) the merge transform itself — a pure **map**, one thread per
point; (2) per-zone plane fitting — a **map into a bounded scatter-reduce** (atomics into just six
output slots, one per surface); (3) the refinement's normal-equation assembly — a **map-then-tree-
reduce**, the same 6×6 Gauss-Newton pattern 01.17/02.06 use for camera calibration and ICP, applied
here to a point-to-plane residual against a *fixed* zone target instead of a searched nearest
neighbor; (4) voxel-grid deduplication of the merged cloud — a **sort-and-compact** spatial-index
pattern (02.01/02.09's lineage). None of these are large by this repo's standards (a few thousand
points), but the *pattern* is exactly what a real fleet vehicle's onboard pipeline runs per scan, at
much larger scale, every 50-100 ms.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception, at the very front of the stack — this is the step that turns
  *raw per-sensor* LiDAR returns into the single coherent `PointCloud` every downstream perception
  project in this domain (02.01 voxel downsampling through 02.14 moving-object segmentation) assumes
  it is given. Extrinsic refinement is a slower, less-frequent side channel: a field-maintenance
  watchdog, not a per-scan step.
- **Upstream inputs:** three `PointCloud` streams (SYSTEM_DESIGN.md §3.6), one per LiDAR, each in its
  own sensor frame — plus each sensor's *believed* extrinsic (`T_base_lidar_i`, SYSTEM_DESIGN.md
  §3.3 naming), which starts out as the factory/CAD value and is what this project's refinement
  updates over time.
- **Downstream consumers:** the merged `PointCloud` feeds every single-cloud perception project in
  this domain by name — ground segmentation (02.03), clustering (02.04), normal estimation (02.09),
  ICP/NDT scan matching (02.06/02.07), and ultimately SLAM (05.xx) and the navigation stack (23.xx).
  The refined extrinsics feed back into the SAME merge step for every future scan, and into a fleet's
  calibration-management system as an audit trail.
- **Rate / latency budget:** merging runs **per scan, at the sensor's own rate — 10-20 Hz** (measured
  on this project's committed sample: ~134 ms wall-clock for the FULL demo, including every twin
  gate and both LM solves; the merge-plus-dedup steps alone are a small fraction of that and would
  comfortably clear a 50-100 ms per-scan budget at production point counts with the naive kernels
  shown here, per SYSTEM_DESIGN.md §1.1's "LiDAR → perception" row). Refinement runs on a completely
  different clock: **on EVENTS** — a scheduled calibration-health check (e.g. daily) or a triggered
  one (a detected collision, a large drift-detection residual) — never per-scan.
- **Reference robot(s):** the autonomous-vehicle stack (SYSTEM_DESIGN.md §2.5, "1-5 LiDARs") is the
  primary reference — multi-LiDAR coverage plus the heaviest calibration-maintenance burden in the
  five reference robots — and the warehouse AMR (§2.1) whenever a heavier platform carries more than
  one 3D LiDAR for full-perimeter coverage.
- **In production:** merging is a small, un-glamorous but load-bearing node in every real stack
  (Autoware's `pointcloud_preprocessor` fuses multi-LiDAR input exactly this way); extrinsic
  refinement in production is usually a "continuous calibration" service (see README §11) running
  offboard or on a fleet-management backend, not onboard in real time.
- **Owning team:** perception (SYSTEM_DESIGN.md §5.1) for the merge path; a **perception
  infrastructure / calibration** sub-team (straddling perception and fleet operations) for the
  refinement/drift-watchdog path — see `PRACTICE.md` §3-4.

## The algorithm in brief

- **Merge transform** — `x_base = T_base_lidar_i * x_lidar_i` per sensor, per point (THEORY.md "The
  algorithm").
- **Voxel-grid deduplication** — hash each merged point into a fixed-size cell, keep one
  representative per occupied cell (02.01's Method-B key packing, cited; THEORY.md "The algorithm").
- **PCA plane fitting** — mean-shifted covariance + eigendecomposition per (sensor, surface) zone,
  the smallest eigenvector is the plane normal (02.03/02.09's lineage, cited; THEORY.md "The math").
- **Plane-pair residual (drift detection)** — the geodesic angle and perpendicular offset between two
  sensors' fitted planes of the *same* physical surface; the observable this whole project is built
  on (THEORY.md "The math" derives the linearized sensitivity to a small extrinsic error).
- **Point-to-plane extrinsic refinement** — Levenberg-Marquardt on a 6-DoF SE(3) local
  parameterization (01.17's se(3)-adjacent retraction, cited), minimizing point-to-fixed-plane
  residuals (02.06's point-to-plane ICP linearization, cited, applied against a *zone-assigned*
  target instead of a nearest-neighbor search — THEORY.md explains why that swap is legitimate here).
- **Observability / conditioning** — the refinement Hessian's condition number, contrasting a
  single-plane zone set against three mutually orthogonal zones (01.17's coplanar-pose degeneracy
  lesson, recast; THEORY.md "Numerical considerations").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/multi-lidar-merging-extrinsic-refinement.sln`](build/multi-lidar-merging-extrinsic-refinement.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/multi-lidar-merging-extrinsic-refinement.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none beyond the CUDA toolkit + C++17 standard library.
`kernels.cu` includes `<thrust/...>` (header-only, part of the CUDA Toolkit) for the merged-cloud
dedup pipeline's sort step; the `.vcxproj`/`CMakeLists.txt` document the two MSVC/CCCL compiler flags
(`/Zc:preprocessor`, `/Zc:__cplusplus`) that Thrust's headers require under MSVC (02.01/02.09's
root-caused fix, reused verbatim).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Synthetic by necessity, not just by repo default (CLAUDE.md §8): extrinsic refinement needs a KNOWN
ground-truth mounting drift to be verifiable at all. `scripts/make_synthetic.py` builds a small
"yard" scanned by a 3-LiDAR rig, generating two cohorts — `aligned.csv` (every sensor at its exact
nominal pose) and `drifted.csv` (the front-corner sensors carry a documented ~0.8°/3 cm drift each,
a different vector per sensor). Full field documentation, checksums, and regeneration instructions
in [`data/README.md`](data/README.md).

## Expected output

Ten independent verification stages, each a stable `PASS`/`FAIL` line plus `[info]` line(s) with the
measured number(s) behind it: `TRANSFORM_TWIN`, `PLANE_FIT_TWIN`, `DRIFT_DETECTION`, `ASSEMBLY_TWIN`,
`TRAJECTORY_TWIN`, `RECOVERY_LEFT`/`RECOVERY_RIGHT` (the headline: does refinement recover each
sensor's true drift?), `VALIDATION_LOOP` (the closed loop — do plane residuals fall back under
threshold after refinement?), `OBSERVABILITY` (a single planar zone vs. three orthogonal zones —
a dramatic, measured condition-number contrast), and `DEDUP_ACCOUNTING` (GPU vs CPU voxel-grid
dedup, exact index-set agreement). Every GPU/CPU twin gate uses a documented, measured-then-margined
tolerance (`src/main.cu`'s tolerance block cites the exact number from an actual run — see THEORY.md
"How we verify correctness"). `demo/expected_output.txt` holds the canonical stable lines; the exact
tolerance-vs-measurement story is in `src/main.cu`'s comments.

Artifacts written to `demo/out/`: `topview_before.ppm`/`topview_after.ppm` (whole-scene context),
`topview_zoom_before.ppm`/`topview_zoom_after.ppm` (**the money shot** — a zoomed inset where the
centimeter-scale ghosting is actually visible against the 26-meter scene), `plane_residuals.csv`,
`gates_metrics.csv`.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — start here: the rig geometry (ASCII diagram), the
   sensor/surface/zone vocabulary, and every shared math primitive (SO(3), point-to-plane
   residual/Jacobian, voxel-key packing).
2. [`src/main.cu`](src/main.cu) — the ten-stage orchestration: load data, fit planes, detect drift,
   refine, validate, check observability and the zero-drift control, dedup, write artifacts.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels: the trivial merge map, the atomics-based
   plane-fit accumulation (contrast with #4's tree reduction), the refinement assembly kernel (the
   project's most interesting kernel), and the Thrust-backed dedup pipeline.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle twins; read its file
   header first for exactly what is and is not shared with the GPU path.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the synthetic rig/scene generator (the
   Python mirror of `kernels.cuh`'s rig constants).
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s multi-candidate file
   resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Autoware's `pointcloud_preprocessor` / `sensing` stack** — the open-source AV reference for
  multi-LiDAR concatenation, deskew, and cropping, the production version of this project's merge step.
- **Continuous/online extrinsic calibration products** (e.g. commercial fleet-calibration services,
  and open research such as *CalibNet*, *LIO-Calib*, and plane/edge-based LiDAR-LiDAR calibration
  papers) — the production version of this project's detect-then-refine loop, usually running
  offboard against fleet-wide logs rather than onboard per-scan.
- **PCL's `SampleConsensusModelPlane` / normal estimation** — the reference plane-fitting toolkit this
  project's PCA pipeline teaches toward.
- **GTSAM / Open3D pose-graph optimization** — the general multi-sensor consistency solver this
  project's [info]-only "loop consistency" stage gestures at without implementing (see 05.xx and
  THEORY.md "Where this sits in the real world").
- **01.17 (camera-LiDAR/camera-camera extrinsic calibration)** — this project's direct lineage: the
  same batched Levenberg-Marquardt machinery, se(3)-adjacent retraction, and Cholesky solve, applied
  here to LiDAR-LiDAR planes refined from field data instead of camera-target correspondences from a
  factory rig.
- **02.06 (ICP point-to-point/point-to-plane/GICP)** — the point-to-plane linearization this
  project's refinement residual is built on.

## Exercises

1. **Add a diagonal wall.** The scene's four walls are axis-aligned (normals along ±x, ±y). Add a
   45°-oriented wall segment and confirm the observability story is unchanged (three independent
   normal directions is what matters, not axis alignment).
2. **Widen the drift.** `kernels.cuh`'s `kDrift` table sets ~0.8°/3 cm. Try 3°/10 cm — does
   `RECOVERY_*` still pass with the current tolerances, or does the point-to-plane linearization
   start to break down (THEORY.md "Numerical considerations" discusses the linearization's validity
   range)?
3. **Implement the direct LEFT-RIGHT twin properly.** The loop-consistency stage computes a direct
   LEFT-RIGHT refinement but never GPU/CPU-twins it. Add that twin gate.
4. **Nearest-neighbor correspondence instead of zone assignment.** Swap the refinement's fixed
   zone-lookup correspondence for a real per-iteration nearest-neighbor search against the target
   sensor's raw points (02.06's approach) — does the answer change? Should it?
5. **A fourth sensor.** Add a rear-facing LiDAR and extend the zone-set/observability logic — what
   changes about the loop-consistency story with three independently-drifted sensors instead of two?

## Limitations & honesty

- **Zone-assignment correspondence, not nearest-neighbor search.** The refinement's "which target
  plane does this point belong to" decision is the point's `surface_id` tag (known at data-generation
  time), not a per-iteration search against the target sensor's raw points (02.06's approach). This
  is an honest, documented simplification valid because the scene is genuinely piecewise-planar and
  a point never migrates from one physical surface to another as the extrinsic is refined — see
  THEORY.md "The algorithm" for the full argument, and Exercise 4 above to explore replacing it.
- **FOV/visibility uses a simplified geometric model.** Which sensor "sees" a world point is decided
  by base-frame azimuth from the sensor's mounting origin (not full raycasting/occlusion) — a
  documented simplification in `scripts/make_synthetic.py`'s header, not a full sensor simulator
  (that is project 11.01's job).
- **No occlusion, no multi-path, no retroreflectivity.** The synthetic scene has clean line-of-sight
  everywhere within a sensor's FOV — a real yard has occluders.
- **Only two sensors are refined; only two are ever drifted.** MAIN never drifts in this project's
  scoping — real fleets recalibrate every sensor over time, including the reference one, which needs
  the fuller pose-graph treatment named in THEORY.md.
- **Loop consistency is diagnostic, not corrective.** The [info]-only stage exposes the graph-
  consistency idea; it does not close the loop by re-solving all extrinsics jointly (05.xx's job).
- **Sim-validated only.** This project's outputs are never applied to real hardware — if you were to
  feed a recovered extrinsic to an actual robot's calibration file, treat it as a candidate needing
  independent validation, not a certified result (CLAUDE.md §1, §8).
