# 02.04 — Euclidean clustering via GPU union-find / connected components

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Euclidean clustering via GPU union-find / connected components`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project turns a ground-removed LiDAR point cloud into a list of **objects**: it groups points
into clusters using the classic Euclidean/single-linkage rule ("two points that are closer than a
distance `d` belong to the same object, transitively"), and computes that grouping on the GPU **two
different ways** on the exact same data — a lock-free **union-find** (path halving + atomicCAS
union-by-min) and iterative **min-label propagation** (the pattern this repo's 30.01/01.06/01.21
already use for image-grid connected components) — so a learner can *watch* the textbook complexity
gap between them instead of just reading about it. On the committed sample, union-find converges in
**2 sweeps**; label propagation needs **299 sweeps** on the same graph, because one designed feature of
the scene (a long, thin, curved chain of points — "the snake") has a large graph diameter, and label
propagation's convergence time is bounded by that diameter while union-find's is not (THEORY.md derives
why). The demo also builds two more designed scenarios into the same scene: two objects kept apart by
slightly more than `d` (they must stay two clusters) and two objects bridged by a thin chain of points
spaced slightly less than `d` (they must merge into one — Euclidean clustering's well-known "chaining"
failure mode, demonstrated honestly rather than hidden). Every stage is checked against an independent
CPU reference; the final partition is checked against ground truth computed the same way, by the
synthetic-data generator, before the pipeline ever runs.

Everything described above is implemented; nothing in the catalog bullet is documented-only.

## What this computes & why the GPU helps

The computation is **connected components of a proximity graph**: build an edge between every pair of
points within distance `d`, then find every connected component of that graph. Two GPU patterns are at
work:

- **Neighbor search is a *gather*.** Each point (a thread) scans a small, bounded neighborhood (a
  27-voxel stencil) and tests candidates independently — the same "one thread reads many memory
  locations, writes its own small output" shape as 02.01's hash-insert kernel, applied to a distance
  test instead of an accumulation.
- **Connected components is *iterative graph relaxation*.** Both algorithms process the whole edge list
  in parallel, every SWEEP, until nothing changes — a **map-reduce-like fixed-point iteration**, not a
  single pass. The GPU's advantage is that "process every edge" is trivially data-parallel *within* a
  sweep; the interesting story (and this project's teaching core) is how many sweeps each algorithm
  needs, which is a property of the *algorithm*, not of the hardware — see THEORY.md "The algorithm" for
  the O(diameter) vs. O(log diameter) derivation the demo measures a live instance of.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception. This project is the third stage of a perception chain this repo's
  domain builds project by project, explicitly: **02.01** (voxel-grid downsampling) thins a raw scan;
  **02.03** (ground segmentation) splits the thinned scan into ground / non-ground; **02.04 (this
  project)** turns the non-ground remainder into discrete object hypotheses. Say it plainly: this
  domain's projects are not independent exercises, they are a pipeline being assembled one project at a
  time, and 02.04 is where "points" first become "objects".
- **Upstream inputs:** a `PointCloud`-shaped array of non-ground points (02.03's output message shape,
  named in that project's README) — see [`data/README.md`](data/README.md) for exactly how this
  project's synthetic sample manufactures that hand-off with ground truth attached.
- **Downstream consumers:** per-cluster centroid/count/AABB feeds a multi-object tracker (domain
  **04.xx** sensor fusion & state estimation — associating this frame's clusters with last frame's
  tracks, e.g. project 04.01's particle filter), a motion predictor, and **23.01**'s GPU costmaps
  (obstacle inflation around each cluster's AABB) — all named here explicitly, the same "the pipeline
  keeps going" framing as the upstream side.
- **Rate / latency budget:** a spinning LiDAR delivers a full scan at 10–20 Hz (`docs/SYSTEM_DESIGN.md`
  item 1); a costmap/planner consumer typically wants results within a few milliseconds of the scan
  arriving so the 10–50 Hz local-planning loop is not stalled. Measured on the reference machine (RTX
  2080 SUPER): the full pipeline (voxel index + edges + union-find + relabel + stats) runs in **well
  under 1 ms** on the committed 1,469-point non-ground scene — comfortably inside even the tightest
  budget, with wide headroom for a much larger real scan (see [Expected output](#expected-output) for
  every measured number).
- **Reference robot(s):** the **AMR** (warehouse robot, domains 02/04/05/23/06/08/25/31/32) and the
  **AV stack** (01/02/03/04/05/06/14/31/32) both consume clustering directly — an AMR needs discrete
  obstacles to avoid in a costmap; an AV stack needs them to seed a tracker for pedestrians, cyclists,
  and other vehicles.
- **In production:** a real stack replaces this project's hand-rolled union-find/label-propagation pair
  with a tuned library implementation (PCL's `EuclideanClusterExtraction`, cuML/cuGraph's GPU connected
  components) and very often layers a *learned* instance segmentation model on top or instead (see
  [Prior art & further reading](#prior-art--further-reading)) specifically to fix the chaining failure
  mode this project demonstrates rather than hides.
- **Owning team:** perception (specifically the "obstacle detection" or "object proposal" sub-team,
  adjacent to the tracking/fusion team that consumes this output and the mapping team that owns 02.01's
  and 02.03's upstream stages) — `docs/SYSTEM_DESIGN.md` item 5 has the fuller org map.

## The algorithm in brief

- **Voxel binning at leaf = d** — 02.01's spatial-hash key-packing machinery, reused with the voxel edge
  length set EXACTLY equal to the cluster tolerance `d`, the condition that makes a 27-cell stencil
  provably sufficient to find every neighbor within `d` (see [`THEORY.md`](THEORY.md) "The math").
- **Neighbor-edge construction** — a sorted-array + binary-search voxel index (a second, different
  spatial-hash query technique from 02.01's open-addressing table) turns "which points are within `d`
  of me" into a single bounded scan per point ([`THEORY.md`](THEORY.md) "The GPU mapping").
- **GPU union-find (Method A)** — lock-free, path-halving find + atomicCAS union-by-min, run as
  repeated parallel sweeps over the edge list until convergence; THE new idea this project teaches
  ([`THEORY.md`](THEORY.md) "The algorithm", "The GPU mapping").
- **GPU min-label propagation (Method B)** — the 30.01/01.06/01.21 image-grid CCL pattern, adapted to
  an arbitrary edge list, run on the identical edges for a fair, measured complexity comparison
  ([`THEORY.md`](THEORY.md) "The algorithm").
- **Cluster relabeling (compact ids via a Thrust scan) + per-cluster statistics + min-size filtering** —
  turns sparse canonical roots into dense `[0,K)` ids, computes count/centroid/AABB per cluster, and
  rejects components smaller than `kMinClusterSize` as noise ([`THEORY.md`](THEORY.md) "The GPU mapping").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/euclidean-clustering-via-gpu-union-find.sln`](build/euclidean-clustering-via-gpu-union-find.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/euclidean-clustering-via-gpu-union-find.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none beyond the CUDA toolkit's header-only Thrust (used for
sort/scan/reduce/copy_if — see `src/kernels.cu`) and the C++17 standard library. If the build fails with
a `CCCL`, `/Zc:preprocessor`, or `__cplusplus` error, see the detailed comment in
`build/*.vcxproj`'s `CudaCompile` sections — a known CUDA 13.3 + Thrust + MSVC interaction, already
worked around there (the same fix 02.01 root-caused).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample is 100% synthetic (CLAUDE.md §8): 180,000 ground points (context only — this
project never clusters them) plus 1,469 non-ground points assembled into a scene designed to make four
specific claims checkable, with an exact ground truth computed by the SAME single-linkage rule the C++
pipeline implements. Regenerate with `python scripts/make_synthetic.py --seed 42` (byte-identical to the
committed file). Full field-by-field documentation, provenance, and checksums:
[`data/README.md`](data/README.md). No public dataset applies —
[`scripts/download_data.ps1`](scripts/download_data.ps1) is an honest no-op and says why.

## Expected output

Success means every `VERIFY`/`GATE` line in the demo's output reads `PASS`, and the program exits 0.
The canonical stable lines live in [`demo/expected_output.txt`](demo/expected_output.txt); this is what
an actual reference run (RTX 2080 SUPER, sm_75, Release|x64) printed:

| Check | What it verifies | Tolerance |
|-------|-------------------|-----------|
| `VERIFY(keys)` | GPU voxel keys vs. the shared CPU formula | bit-exact |
| `VERIFY(edges)` | GPU neighbor-edge set vs. an independent CPU `unordered_map`-based build | exact set equality |
| `VERIFY(union_find)` | GPU lock-free union-find's canonical roots vs. independent sequential CPU union-find | bit-exact (mathematically order-independent, see THEORY.md) |
| `VERIFY(label_propagation)` | GPU label-propagation's converged labels vs. the SAME CPU union-find partition | bit-exact |
| `VERIFY(stats)` | GPU per-cluster count/centroid/AABB vs. independent double-precision CPU accumulation | counts exact; centroid/AABB within 0.5 mm (measured 1.72e-05 m, see below) |
| `GATE partition_vs_truth` | GPU result vs. the generator's ground truth | bit-exact |
| `GATE stats_integrity` | counts sum to N; every centroid inside its own AABB | free invariant, exact |
| `GATE noise_filtering` | every scattered noise point ends up unclustered | exact |
| `GATE separation_test` | the two objects kept `d+0.10 m` apart stay two clusters | exact |
| `GATE chaining_test` | the two objects bridged by a `d-0.10 m`-spaced path merge into one | exact |
| `GATE snake_convergence` | label-propagation sweeps ≥ 50 AND union-find sweeps ≤ 20 on the same snake | measured: **299** vs. **2** |

Measured numbers from the reference run (also written to `demo/out/gates_metrics.csv` every run):

- 1,469 non-ground points → 24,259 candidate edges, 462 occupied voxels.
- **Union-find: 2 sweeps, 0.2–0.4 ms.** **Label propagation: 299 sweeps, 13–14 ms.** Same edges, same
  final partition, a ~50–65x sweep-count gap and timing gap — the complexity lesson, measured, not
  asserted.
- 37 raw connected components → 9 reported after `min_cluster_size=5` filtering (28 points rejected as
  scattered noise).
- `max_stats_delta_m = 1.717e-05` (the largest GPU-vs-CPU centroid/AABB disagreement observed, driving
  the documented 0.5 mm tolerance — CLAUDE.md's "measured then margined" rule).

Artifacts written to `demo/out/` (git-ignored, regenerated every run): `topview_truth.ppm` and
`topview_gpu_result.ppm` (top-view renders colored by truth vs. GPU-reported cluster id, side by side),
`topview_snake_highlight.ppm` (the long-diameter chain highlighted in magenta against everything else in
gray), `sweep_comparison.csv` (union-find vs. label-propagation sweep counts and timings), and
`gates_metrics.csv` (every measured number behind every line above).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — read this FIRST this time: the shared contract, and "THE
   UNION-FIND CHAPTER" walks the lock-free find/union algorithm end to end before any code runs it.
2. [`src/main.cu`](src/main.cu) — orchestration: load the scene, build the edge graph, run both GPU
   algorithms, verify everything, gate the four designed scenarios, write artifacts.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels: `build_edges_kernel` (neighbor search),
   `uf_union_sweep_kernel` (the heart of the project), `lp_sweep_kernel` (its point of comparison), and
   the Thrust-based relabeling pipeline.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle: a genuinely different
   edge-building data structure and a textbook sequential union-find, with the independence ruling
   explained in the file header (union-find is exactly the kind of "clever" algorithm that ruling exists
   for).
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the scene: read `build_truth_clusters()`
   to see the SAME algorithm this project's C++ implements, run once more in pure Python as ground truth.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, path resolution, and why they are copied,
   not shared.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **PCL `EuclideanClusterExtraction`** — the textbook CPU implementation (a KD-tree radius search +
  union-find) this project's algorithm is a from-scratch, GPU-parallel version of; read its docs for the
  production API shape.
- **cuML / cuGraph connected components** — NVIDIA's production GPU graph-connectivity libraries; study
  their (more sophisticated) parallel CC algorithms — this project's lock-free union-find is a teaching-
  scale relative of the same family.
- **DBSCAN** — the density-aware upgrade that fixes exactly the chaining failure mode this project
  demonstrates honestly (a sparse "bridge" of points fails DBSCAN's density requirement even though it
  passes single-linkage's simpler distance rule); documented-only here, not implemented.
- **Shiloach & Vishkin (1982), "An O(log n) parallel connected components algorithm"**, and Anderson &
  Woll (1991) / Jayanti & Tarjan (2016) on lock-free concurrent union-find — the theoretical lineage of
  the path-halving + union-by-min approach implemented here.
- **Tarjan (1975), "Efficiency of a Good But Not Linear Set Union Algorithm"** — the classical
  sequential O(alpha(n)) amortized union-find result THEORY.md cites for the algorithm's complexity story.
- **PointPillars / CenterPoint (project 12.xx-adjacent)** — the modern, *learned* alternative to
  geometric clustering for instance segmentation, named honestly: production perception stacks
  increasingly replace Euclidean clustering with a neural instance head precisely to avoid the chaining
  and occlusion failure modes THEORY.md discusses.

## Exercises

1. Change `kClusterToleranceM` in `src/kernels.cuh` (and `D_M` in `make_synthetic.py`, they must match)
   to `0.5` m and regenerate the sample — watch the separation-test gap shrink and, eventually, fail.
2. Add a "max cluster extent" post-filter (split any cluster whose AABB diagonal exceeds a threshold)
   and see it correctly split the chained pair back into two objects — the semantic fix THEORY.md names.
3. Implement `atomicMin`-based union-by-**rank** instead of union-by-**min** and measure whether sweep
   counts change (they should not, materially — the point is to see empirically that the CANONICAL id
   convention, not the balancing rule, is what this project's gates depend on).
4. Make the snake twice as long (bump `snake_radius`'s angular span in `make_synthetic.py`) and re-run —
   confirm label-propagation's sweep count roughly doubles while union-find's barely moves (the O(D) vs.
   O(log D) prediction, tested again at a different scale).
5. Port `build_edges_kernel`'s binary-search neighbor lookup to a hash-table lookup (reusing 02.01's
   exact insert/probe pattern instead) and compare measured edge-construction time.

## Limitations & honesty

- **Objects are filled synthetic point blocks, not ray-cast LiDAR returns.** `scripts/make_synthetic.py`
  samples boxes/poles/the snake as jittered lattices, not via a sensor model (that realism is 02.01's/
  11.01's/01.18's job, cited) — the right scoping choice for a project whose subject is the *clustering
  algorithm*, not sensor simulation, but a real scan's point density varies with range in ways this
  scene does not model.
- **Filtering, not semantic disambiguation.** The chaining-test gate is deliberately designed to PASS
  when two objects wrongly merge — this is honest single-linkage clustering behavior, not a bug; a real
  system adds a semantic or density-aware layer on top (see [Prior art](#prior-art--further-reading)).
- **min_cluster_size is a single global threshold.** A real system often uses range-dependent or
  density-adaptive thresholds (a pedestrian at 40 m returns far fewer points than one at 5 m); this
  project's fixed threshold is a deliberate simplification.
- **No temporal tracking.** Every cluster is computed fresh, per scan; identity across frames (needed to
  turn "cluster 3 this frame" into "the same pedestrian as last frame") is domain 04.xx's job, cited but
  not implemented here.
- Sim-validated only: this project's output is illustrative of a perception pipeline stage and is not
  connected to, nor validated against, any real robot or real sensor. If a downstream consumer of
  clustering output ever commands real robot motion (which it eventually would, in a real stack), that
  hardware safety caveat belongs to *that* project, not this one — but is restated here for honesty.
