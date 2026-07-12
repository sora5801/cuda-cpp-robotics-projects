# 02.09 — Normal + curvature estimation at millions of points/sec

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Normal + curvature estimation at millions of points/sec`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

For every point in a point cloud, this project computes a **surface normal** (which way the local
surface faces) and a **curvature proxy** (how much the local surface bends or breaks), fast enough for
LiDAR-rate throughput. It is the geometric-feature layer nearly every downstream point-cloud algorithm
in this repository leans on: point-to-plane ICP needs normals to converge well and stably; feature
descriptors like FPFH are built ON TOP of normals; ground/obstacle segmentation and grasp planning both
use surface orientation. The pipeline — per point, in one fused GPU kernel — is: find the K=16 nearest
neighbors (a compact voxel-hash index, reimplemented locally, cited from project
[`02.05`](../02.05-kd-tree-or-lbvh-construction-knn-radius-search/README.md)); fit the local covariance
with the mean-shift numerical trick; eigendecompose it with a hand-rolled Jacobi solver (cited from
[`02.03`](../02.03-ground-segmentation/README.md)); take the smallest-eigenvalue eigenvector as the
normal, oriented toward the sensor; and derive a **surface-variation** curvature proxy from the
eigenvalue ratios, plus a degeneracy flag for edges/corners/isolated points. The demo verifies the whole
pipeline against **closed-form** analytic surfaces (a plane, a sphere, a cylinder, and a sharp edge) —
so "the normal is correct" is checked against real geometric truth, not just GPU-vs-CPU self-consistency
— and then measures throughput on a 1,050,000-point workload.

> **Template placeholder notice.** This project's `src/` no longer contains the scaffolded SAXPY
> placeholder — it has been replaced by the real implementation described below and throughout
> `THEORY.md`/`PRACTICE.md`.

## What this computes & why the GPU helps

For point *p* with K nearest neighbors {q₁..qₖ} (including *p* itself): the mean-shifted covariance
matrix Cov = (1/K)Σ(qᵢ−mean)(qᵢ−mean)ᵀ; its eigendecomposition Cov = VΛVᵀ; the normal is the eigenvector
for the smallest eigenvalue λ₀ (the total-least-squares plane fit through the neighborhood — see
`THEORY.md` "The math"); curvature is the surface-variation proxy λ₀/(λ₀+λ₁+λ₂).

Every point's computation is **completely independent** of every other point's, once the shared spatial
index exists — the textbook GPU **map** pattern, one thread per point. The parallelized bottleneck is
therefore the whole per-point cost — a K-nearest-neighbor search PLUS a 3×3 eigenproblem, repeated N
times, embarrassingly parallel. What is genuinely non-trivial (and is where this project's engineering
lives) is keeping that per-thread cost cheap enough, and the neighbor search honest enough, that a
single kernel launch processes a million-plus points per second — see "The algorithm in brief" and
`THEORY.md` "The GPU mapping" for the register-pressure and memory-layout story.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — the geometric-feature layer directly above raw point-cloud
  preprocessing (voxel downsampling `02.01`, ROI crop `02.02`, motion deskew `02.08`) and directly below
  registration/feature/segmentation consumers.
- **Upstream inputs:** a `PointCloud` (interleaved xyz, meters, sensor frame) — typically already voxel-
  downsampled (`02.01`) and deskewed (`02.08`); the sensor origin, for normal orientation.
- **Downstream consumers, named:** point-to-plane / GICP scan matching (`02.06` — its faster, more
  stable convergence versus point-to-point ICP exists BECAUSE of this normal layer); FPFH descriptors
  (`02.10`, the very next project in this domain — FPFH is a histogram of pairwise ANGLES between
  points' normals, built directly on this project's output); ground/obstacle segmentation and moving-
  object segmentation (`02.13`/`02.14`, which use surface orientation and local flatness); grasp-quality
  scoring (`19.01` — antipodal grasp points are scored by how well two contact normals oppose each
  other).
- **Rate / latency budget:** an automotive-grade spinning LiDAR returns ~100k-500k points per 100 ms
  sweep (10 Hz) — 1M-5M points/sec sustained; a solid-state/flash LiDAR or an RGB-D camera can push
  higher point RATES at lower per-frame counts but tighter per-frame latency (30-60 Hz). This project's
  measured throughput (`GATE throughput`, `[info] throughput_measured`) is directly comparable to that
  budget: `Mpts/s >= sensor_points_per_sweep * sweep_rate_hz / 1e6` is the pass condition for keeping up
  in real time; see `THEORY.md` "Where this sits in the real world" for the arithmetic against this
  project's own measured numbers.
- **Reference robot(s):** the autonomous-vehicle stack (`docs/SYSTEM_DESIGN.md` reference robot 5 —
  domains 01/02/03/04/05/06/14/31/32) and the 6-DoF manipulator work cell (reference robot 2 — where
  normals feed grasp planning, `19.01`).
- **In production:** PCL's `NormalEstimationOMP`/`IntegralImageNormalEstimation`, Open3D's
  `estimate_normals`, or a learned normal estimator (PCPNet, DeepFit — README "Prior art") running on
  the same GPU tier as the rest of the perception stack; often FUSED into a single perception kernel
  chain (normals computed once, consumed by several downstream stages) rather than a standalone pass.
- **Owning team:** Perception (the team that also owns `01`-`05`); its output is a hard dependency for
  the Localization/Mapping team (`02.06`, SLAM) and the Manipulation team (`19.01`).

## The algorithm in brief

- **Voxel-hash spatial index** (K-nearest-neighbor engine) — sort points into a uniform grid by a
  packed 64-bit cell key (Thrust radix sort + boundary compaction), reimplemented compactly here, cited
  from [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md) (the key-packing
  formula) and [`02.05`](../02.05-kd-tree-or-lbvh-construction-knn-radius-search/README.md) (the
  bounded-heap KNN idea and the LBVH-vs-hash trade study this project's design decision is based on).
  See `THEORY.md` "The algorithm" for the full derivation.
- **Safe-radius ring expansion** — a genuinely new piece of engineering this project needed: scanning a
  fixed 3×3×3 cell stencil is NOT sufficient to guarantee the true K nearest neighbors (a query can sit
  anywhere inside its own cell); the search widens ring by ring until a provable safe-radius bound holds.
  See `THEORY.md` "The algorithm" for the proof and `kernels.cu`'s `estimate_normals_kernel` header for
  the implementation.
- **Mean-shifted covariance** — the numerically stable two-pass covariance accumulation (centroid first,
  then covariance around it), avoiding the catastrophic cancellation of the naive one-pass formula.
- **Cyclic Jacobi eigensolver** for symmetric 3×3 matrices, hand-rolled, cited from
  [`02.03`](../02.03-ground-segmentation/README.md)'s precedent (contrasted there against
  [`02.08`](../02.08-per-point-motion-deskew-with-pose-interpolation/README.md)'s closed-form Smith
  solver, which only returns the smallest eigenvalue — this project needs the full spectrum).
- **Sensor-oriented normal disambiguation** — the classic viewpoint sign-flip heuristic (PCL's
  `flipNormalTowardsViewpoint`), and the honest failure mode it has at grazing incidence.
- **Surface variation** (Pauly, Gross & Kobbelt 2002) as the curvature proxy — deliberately NOT mean or
  Gaussian curvature; `THEORY.md` "The math" defines the relationship precisely.
- **Eigenvalue-ratio degeneracy flagging** — a Demantke et al. (2011)-style dimensionality feature,
  reduced here to a single threshold on surface variation.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/normal-curvature-estimation-at-millions.sln`](build/normal-curvature-estimation-at-millions.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/normal-curvature-estimation-at-millions.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **Thrust** (header-only, ships with the CUDA Toolkit —
`kernels.cu`'s voxel-hash index build uses `thrust::stable_sort_by_key`/`reduce`/`copy_if`; no fallback
needed, it is always available). No other dependency beyond the CUDA Toolkit + C++17 standard library.
The `.vcxproj` carries the `/Zc:preprocessor /Zc:__cplusplus -std=c++17` flags Thrust's CCCL headers
require under MSVC — see the `CudaCompile` item group comments if retargeting this project.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample (`data/sample/normals_scan.bin`, ~263 KiB, 8,400 points) is **synthetic by
necessity, not just by convention**: normal/curvature estimation needs closed-form GROUND TRUTH to grade
against, and no public LiDAR dataset ships exact per-point normals/curvature for real scans. Four
analytic surfaces (tilted plane, sphere, cylinder, sharp edge) at three noise levels each, generated by
`scripts/make_synthetic.py --seed 42`. Full field-by-field format, provenance, and checksum in
[`data/README.md`](data/README.md).

## Expected output

The demo runs a correctness pass (GPU vs. three independently-coded CPU/brute-force oracles, plus
angular error against analytic ground truth — tolerances documented in `THEORY.md` "How we verify
correctness") followed by a throughput pass (1,050,000 points, GPU only). All `VERIFY`/`GATE` lines must
print `PASS` for `RESULT: PASS`. The canonical stable lines live in
[`demo/expected_output.txt`](demo/expected_output.txt); `[info]`/`[time]` lines carry real measured
numbers but are deliberately not diffed (CLAUDE.md §12 — some are GPU-architecture/timing-sensitive).
Artifacts (`normal_map.ppm`, `curvature_heatmap.ppm`, `degeneracy_map.ppm`, two CSVs) land in `demo/out/`.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: load the analytic-surface sample, build the voxel-hash
   index, run the GPU pipeline and its three independent CPU/brute-force oracles, every `VERIFY`/`GATE`,
   the throughput replication, and the artifact writers.
2. [`src/kernels.cuh`](src/kernels.cuh) — the shared data-layout contract: point/cohort structs, the
   voxel-hash key packing, the KNN tie-break total order, every tuned constant (K, cell size, ring cap,
   curvature threshold), and the full pipeline design rationale in its file header — read this first for
   the "why", before the code.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels: the voxel-hash index build, and
   `estimate_normals_kernel`, the fused per-point KNN → covariance → Jacobi eigensolve → normal →
   curvature → degeneracy pipeline (the heart of the project).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — three independently-typed CPU oracles: an
   unordered-map voxel-hash KNN + covariance/eigen twin, and a hash-free O(n) brute-force anchor — see
   its file header for the independence ruling each follows.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the analytic-surface truth engine: how the
   plane/sphere/cylinder/edge cohorts, their noise model, and their ground truth are generated.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and data/artifact path resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **PCL `NormalEstimation`/`NormalEstimationOMP`** — the reference implementation of exactly this
  algorithm (covariance PCA + viewpoint orientation) in a production point-cloud library; study its
  `setKSearch`/`setRadiusSearch` API and its integral-image variant for organized (depth-image) clouds.
- **Open3D `estimate_normals`** — a modern, GPU-capable equivalent with a similar API; also implements
  normal CONSISTENCY propagation (orienting a whole cloud's normals coherently, not just toward one
  sensor origin) — an extension this project deliberately scopes out (README "Limitations").
- **Pauly, Gross & Kobbelt, "Efficient Simplification of Point-Sampled Surfaces" (2002)** — the paper
  that names "surface variation" as a curvature/feature-saliency proxy; the exact formula this project's
  curvature output implements.
- **Demantke, Mallet, David & Vallet, "Dimensionality Based Scale Selection in 3D LiDAR Point Clouds"
  (2011)** — the eigenvalue-ratio "linearity/planarity/sphericity" feature family this project's
  degeneracy flag is a simplified instance of.
- **nvblox / Open3D CUDA backends** — production examples of exactly this kernel (per-point PCA normals)
  running at LiDAR/RGB-D rates on GPU; study their memory-layout choices (SoA point buffers, fused
  kernels) against this project's own.
- **PCPNet (Guerrero et al. 2018) / DeepFit (Ben-Shabat & Gould 2020)** — learned normal estimation:
  neural networks that predict normals (and sometimes curvature) directly from a local patch, trained to
  be robust to noise and sharp features in ways a fixed-K PCA fit cannot be — named honestly as the
  research-frontier alternative to everything in this project.

## Exercises

1. Retune `kernels.cuh`'s `kK` (try 8 and 32) and re-run the demo — how does the sphere/cylinder
   curvature-fit bias (`GATE sphere_normals`'s noise=none mean angular error) change? Does the edge
   cohort's degeneracy-flag rate change? Explain both trends from `THEORY.md`'s K-vs-noise-vs-bias
   derivation.
2. Implement normal-consistency propagation (a graph-coloring / minimum-spanning-tree pass over the KNN
   graph that flips normals to agree with their neighbors, à la Hoppe et al. 1992) as an alternative to
   this project's per-point viewpoint heuristic — measure how it changes `GATE orientation`'s grazing-
   incidence success rate.
3. Add a fifth analytic cohort — a torus, or a saddle (hyperbolic paraboloid, negative Gaussian
   curvature) — with its own closed-form normal, and extend `GATE curvature_ordering` to include it.
4. Profile `estimate_normals_kernel` with `nvcc --ptxas-options=-v` and Nsight Compute; measure actual
   achieved occupancy at `kK=16` versus a hypothetical `kK=8`, and connect the numbers to `THEORY.md`'s
   register-pressure discussion.
5. Replace the fixed `kCellSizeM` with a density-ADAPTIVE cell size (estimated from a quick point-count
   pass), and measure whether it changes the ring-expansion rate (`kDegenIsolated` counts) on a point
   cloud with deliberately non-uniform density.

## Limitations & honesty

- **Synthetic-only data, by necessity, not laziness.** No public dataset carries exact per-point normal
  and curvature ground truth for real LiDAR scans; every number this project's gates check is graded
  against analytically exact surfaces, not real-world data. `data/README.md` states this explicitly.
- **No normal-consistency propagation.** Every normal is oriented independently, toward ONE fixed sensor
  origin. A real multi-view fusion pipeline (or a single scan with self-occluding surfaces on both sides
  visible from different sensor positions) needs graph-based consistency propagation this project does
  not implement (see "Exercises" #2 and "Prior art").
- **Fixed K, fixed cell size.** Real LiDAR point density falls off as 1/r² with range (`02.05`'s THEORY.md
  derives this); this project's committed sample is deliberately near-uniform density per cohort so a
  single tuned cell size (`kCellSizeM`) works everywhere. A production pipeline over a full real scan
  would need either an adaptive cell size or per-range K, neither implemented here (see "Exercises" #5).
- **The curvature is a proxy, not differential-geometry curvature.** `THEORY.md` is explicit: surface
  variation correlates with true (mean/Gaussian) curvature on smooth surfaces but is not equal to it, and
  spikes at edges/corners for a different reason (neighborhood straddling two surfaces) than smooth
  bending. Do not feed this project's curvature output into a computation that needs true differential
  curvature without re-reading that section.
- **Grazing-incidence orientation is a known, measured, unfixed failure mode.** `GATE orientation`
  reports it honestly ([info] `orientation_grazing`) rather than hiding it; it is the expected behavior
  of the viewpoint sign-disambiguation heuristic, not a bug.
- **Sim-validated only.** This project's output is a pure geometric computation with no motion of any
  kind — the real-hardware safety caveat (CLAUDE.md §1) applies only insofar as its DOWNSTREAM consumers
  (ICP, grasp planning) command real motion; this project itself commands nothing.
