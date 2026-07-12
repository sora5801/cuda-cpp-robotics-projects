# 02.12 — Range-image conversion + depth-clustering segmentation

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Range-image conversion + depth-clustering segmentation`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A spinning LiDAR does not natively hand you an "unorganized point cloud" — it hands you one range
value per (beam, azimuth-step), a 2-D **range image**, the same way a rolling-shutter camera hands you
one intensity per (row, column). This project (a) implements that conversion on the GPU, both
directions, and (b) uses the resulting image structure to cluster obstacles WITHOUT any 3-D
neighbor search: the Bogoslavskyi & Stachniss **depth-clustering** algorithm, which tests only the two
image-adjacent neighbors of every cell against an angular ("beta") criterion derived from the triangle
formed by two beams and the surface between their returns. Everything the catalog bullet names is
implemented: the range-image conversion kernels (unorganized→organized and back), a compact
range-image-native ground-removal pass, and the depth-clustering segmenter itself. As a teaching
contrast (not part of the catalog bullet, but essential to showing *why* the image-native approach
matters), the project also runs a compact voxel-hash Euclidean clustering pipeline (02.04's lineage) on
the exact same non-ground points and measures where and why the two disagree.

The demo runs both pipelines on one committed synthetic scene, verifies every GPU stage against an
independent CPU reference, and gates four designed teaching scenarios end to end — see
[Expected output](#expected-output) for the actual measured numbers from a reference run.

## What this computes & why the GPU helps

- **Range-image conversion** (both directions) is a **scatter/gather** pattern: unorganized→organized
  is a per-point scatter into a computed cell (races resolved by a nearest-wins encoded `atomicMin`,
  02.02's technique, cited); organized→unorganized is a per-cell stream compaction (an atomic-counter
  append at this project's cell count).
- **Ground removal** is a **stencil-per-column** pattern: `kAzimuthBins` (1024) independent threads,
  each walking its own 16-cell column sequentially — embarrassingly parallel across columns, a short
  bounded serial chain within one.
- **Depth-clustering edges** are an **image-stencil map**: one thread per cell, testing exactly two
  fixed neighbors (ring+1, and azimuth+1 with wrap-around) — no search, no sort, no spatial hash. This
  is the whole point: neighbor-finding, which dominates 02.04/02.05's runtime, is *free* here because
  the sensor already handed you the adjacency structure.
- **Clustering** (both the depth-image graph and the Euclidean-comparison graph) is the generic
  lock-free **GPU union-find** (path-halving find + union-by-min, 02.04's Method A, reused verbatim as
  a function of any edge list).
- **Euclidean comparison** reuses 02.01/02.04's **voxel-hash + 27-cell stencil** pattern (sort-based
  spatial index + binary search) — the explicit "how much work does neighbor-finding cost when you
  DON'T get it for free" baseline.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — the low-latency obstacle-segmentation lane, immediately downstream
  of the LiDAR driver and upstream of everything that needs "here are the obstacles" rather than "here
  are 100,000 points."
- **Upstream inputs:** the LiDAR driver's native, ring-major packet stream (message-shaped as
  `PointCloud` with per-point `ring`/`azimuth` metadata already attached — 02.02's organized-grid
  convention, cited by name) — NOT a pre-organized cloud; this project's own first stage builds the
  range image from that raw stream.
- **Downstream consumers:** an obstacle list (per-cluster centroid, count, AABB) feeding tracking/data
  association (04.xx) and costmap population (23.01's GPU costmaps, cited by name) at minimum added
  latency.
- **Rate / latency budget:** LiDAR → perception is a **10–20 Hz / <100 ms** boundary
  (`docs/SYSTEM_DESIGN.md` §1.1). This project's own measured pipeline time on the reference machine is
  a small fraction of a millisecond for the full committed scene (see "Expected output" below) —
  comfortably inside that budget even before any production-grade optimization, and the embedded angle
  matters here specifically: image-grid algorithms with no spatial-hash memory overhead fit Jetson-class
  compute budgets (32.xx, cited by name) far more easily than a full voxel-hash pipeline at real LiDAR
  point counts.
- **Reference robot(s):** the **warehouse AMR** (its LiDAR obstacle-avoidance loop is exactly this
  low-latency lane) and the **autonomous-vehicle stack** (production AV LiDAR pipelines — Apollo,
  Autoware — use range-image-native processing for precisely this reason; `docs/SYSTEM_DESIGN.md` §2.1,
  §2.5).
- **In production:** a real range-image obstacle segmenter would be one stage in a fused pipeline
  (ground removal → range-image segmentation → tracking → prediction), likely running on the vehicle's
  perception compute tier alongside camera-based detection, with the two fused downstream.
- **Owning team:** Perception (sometimes split into "LiDAR perception" as its own sub-team at larger
  robotics companies) — `docs/SYSTEM_DESIGN.md` §5.1.

## The algorithm in brief

- **Range-image conversion, both directions** — nearest-wins encoded-`atomicMin` scatter
  (unorganized→organized) and atomic-counter stream compaction (organized→unorganized); see
  [`THEORY.md`](THEORY.md) "The GPU mapping".
- **Range-image-native ground removal** — a column-wise vertical-angle walk against a virtual
  sensor-height reference point; see [`THEORY.md`](THEORY.md) "The math".
- **The beta criterion (Bogoslavskyi & Stachniss, IROS 2016)** — `beta =
  atan2(r2*sin(alpha), r1 - r2*cos(alpha))` over every image-adjacent obstacle-cell pair; large beta =
  continuous surface, small beta = a range discontinuity = an object boundary. Full triangle derivation
  in [`THEORY.md`](THEORY.md) "The math".
- **Generic lock-free GPU union-find** (02.04 Method A, cited) — clusters both the depth-image graph
  and the Euclidean-comparison graph with the identical three kernels.
- **Voxel-hash Euclidean comparison clustering** (02.04 lineage, cited) — the explicit "what if you had
  to find neighbors the 3-D way" baseline this project measures against.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/range-image-conversion-depth-clustering.sln`](build/range-image-conversion-depth-clustering.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/range-image-conversion-depth-clustering.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies: CUDA Toolkit libraries (Thrust, header-only, for the Euclidean-comparison
voxel index's sort/compaction — see `build/*.vcxproj`'s comments for the `/Zc:preprocessor` /
`/Zc:__cplusplus` flags this requires under MSVC) + C++17 standard library only. No fallback path is
needed since Thrust ships with the CUDA Toolkit itself.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Synthetic by default (CLAUDE.md §8): a ray-cast 16-beam LiDAR scan with exact ground truth, generated
by `scripts/make_synthetic.py` (fixed seed 42, xorshift32, stdlib-only). Full field-by-field format,
provenance, and checksum in [`data/README.md`](data/README.md).

## Expected output

Every `VERIFY(...)` line compares a GPU stage against an independently-written CPU twin
(`src/reference_cpu.cpp`); every `GATE ...:` line asserts a designed, measured teaching scenario. All
numbers below are from an ACTUAL run on the reference machine (RTX 2080 SUPER, sm_75, Release|x64, the
committed 7,957-point sample) — never fabricated (CLAUDE.md §12); see
[`demo/expected_output.txt`](demo/expected_output.txt) for the exact stable lines this repo's CI-style
check diffs against, and `demo/out/gates_metrics.csv` after any run for the complete measured table.

- **`VERIFY(range_image)`** — GPU nearest-wins `atomicMin` scatter is bit-exact vs. the independent CPU
  running-minimum scatter (0 mismatches), and neither of the two synthetic collision-test phantom
  points ever wins its cell (`GATE collision_resolution`).
- **`VERIFY(ground_removal)` / `GATE ground_removal`** — the GPU and CPU column-walks agree exactly; the
  predicted ground label reaches **precision 0.9789, recall 1.0000** against the generator's truth
  (floor 0.95 both).
- **`VERIFY(depth_edges)` / `VERIFY(union_find_depth)`** — the GPU beta-criterion edge set and its
  union-find partition are bit-exact vs. independent CPU twins.
- **`GATE partition_vs_truth`** — depth clustering recovers **person, big_box, and far_pole** each as a
  single, (near-)pure cluster: measured best-IoU **1.000 / 1.000 / 1.000** against the floor 0.85 (IoU
  computed only over cells that survived ground removal — see `src/main.cu`'s comment on why mixing in
  ground-removal edge effects would misattribute that stage's error to clustering). `wall_behind` is
  reported separately: it correctly SPLITS into 2 clusters covering 100% of its own points — the
  person standing in front of its middle occludes the strip directly behind it, disconnecting the
  visible left/right flanks in the image. That is the segmenter working correctly on an occluded
  object, not an error.
- **`GATE depth_gap_showcase` — the headline result.** The person stands a MEASURED 0.19 m (nearest
  visible-point distance) in front of the wall — smaller than the Euclidean comparison's 0.40 m
  tolerance. Depth clustering shares **0** clusters between person and wall (correctly separated, at
  any range — the beta criterion is range-RATIO based, not a fixed metric distance); the Euclidean
  comparison shares **1** cluster (it MERGES them, exactly the fixed-radius-clustering failure mode this
  project exists to demonstrate).
- **`GATE grazing_fragmentation` — the honest weakness.** A 13 m wall viewed at shallow, grazing
  incidence fragments into **13** depth clusters of size >= 5 (floor 3) — the beta criterion's
  documented failure mode, MEASURED. The Euclidean comparison, blind to viewing angle, keeps the same
  wall in only 5 clusters ([info] contrast).
- **`[info] thin_pole`** — the isolated 0.10 m post survives the min-cluster-size filter (raw component
  size 25 >= floor 5) with best-IoU 1.000 on this scene; `THEORY.md` discusses the case where it would
  not.
- **`GATE timing_payoff`** — both clustering paths' GPU time are measured with `cudaEvent`s; a reference
  run measured the depth-image path at roughly 3–5x FASTER than the voxel-hash Euclidean path on this
  scene (see the `[time] ... path total` lines any run prints — single-shot, teaching artifact, never a
  benchmark claim).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — read this FIRST: the shared contract, the six-stage pipeline
   walkthrough, the beta-criterion formula, every constant's derivation.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels, in pipeline order: range-image conversion,
   ground removal, depth-clustering edges (the beta criterion in CUDA), the generic union-find, the
   Euclidean-comparison voxel hash.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins; read side by side with
   `kernels.cu` to see exactly what stayed shared (data-layout formulas) vs. what was genuinely
   reimplemented (the "clever" algorithmic cores).
4. [`src/main.cu`](src/main.cu) — orchestration: load the raw scan, run both pipelines, verify every
   stage, compute IoUs/fragmentation counts, write artifacts. The most interesting reading here is the
   `truth_for_iou` restriction (why the clustering-quality metric must exclude ground-removal's own edge
   effects) and the `depth_gap_showcase` / `grazing_fragmentation` gates (how a designed scene becomes a
   measured assertion).
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the scene generator; its module docstring
   explains the occlusion-shadow geometry that had to be accounted for to make the depth-gap showcase
   actually work (a real lesson in why "looks right on paper" needs measurement).
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Bogoslavskyi & Stachniss, "Fast Range Image-Based Segmentation of Sparse 3D Laser Scans for
  Online Operation," IROS 2016** — the paper this project's depth-clustering core reimplements
  didactically; read it for the original angle-criterion derivation and their reported runtimes.
- **`depth_clustering`** (github.com/PRBonn/depth_clustering) — the paper authors' own open-source
  implementation; study its image representation and its handling of multi-echo/dual-return LiDAR
  (which this project's simplified scene does not model).
- **Autoware / Apollo LiDAR perception stacks** — production AV pipelines that use range-image-native
  processing for exactly the latency reasons this project's System-context section names.
- **RangeNet++ (Milioto et al., IROS 2019)** — the learned-segmentation successor to hand-crafted
  criteria: a CNN over the SAME range-image representation this project builds, replacing the beta
  criterion with a learned per-pixel classifier.
- **PCL's `EuclideanClusterExtraction`** — the classic 3-D-neighbor-search clustering baseline this
  project's Euclidean comparison stage reimplements didactically (cited in full from 02.04).
- **02.02 (organized/unorganized conversion)** and **02.03 (ground segmentation)** — this project's
  direct upstream lineage; read them first if the range-image concept or ground-removal terminology is
  new.

## Exercises

1. Change `kBetaThresholdDeg` in `kernels.cuh` (try 5 and 20) and rerun — watch `grazing_fragmentation`'s
   fragment count and `partition_vs_truth`'s IoUs move in opposite directions; find the value that
   fragments the grazing wall the LEAST while still separating the depth-gap pair.
   [`THEORY.md`](THEORY.md) "Numerical considerations" discusses the trade-off this exposes.
2. Add a third clustering method: min-label propagation (02.04's Method B) over the SAME depth-image
   edges, and compare its sweep count against union-find's on this project's grazing wall (which has a
   long, thin fragment chain — a natural "does propagation's O(diameter) bound show up here too?"
   question).
3. Extend `depth_edges_kernel` to consider EIGHT neighbors (add the two diagonal image neighbors) instead
   of four total (two forward) — does it change the grazing-wall fragment count? Why or why not,
   geometrically?
4. Implement dual-return handling in the synthetic generator (two ranges per ray, near-surface + far-
   surface) and extend `scatter_encode_kernel` to keep both — a real step toward what a production
   driver actually reports (see `PRACTICE.md` section 1).
5. Swap the min-cluster-size noise floor for a RANGE-DEPENDENT one (a real object subtends fewer image
   cells at range) and rerun `thin_pole` at several distances — at what range does it start dying to the
   filter with a FIXED floor, and does the adaptive floor fix it?

## Limitations & honesty

- **Ground is a single flat plane.** The column-wise angle walk is a genuine simplification against
  02.03's full RANSAC/CZM treatment, which handles ramps, plateaus, and multi-level terrain this
  project's ground-removal stage cannot (cited honestly at the point of use in `src/main.cu`'s
  `GATE ground_removal` line and in `THEORY.md` "Where this sits in the real world").
- **No dual/multi-echo returns.** Real LiDARs often report 2+ ranges per beam direction (near-surface
  and far-surface, e.g. through foliage or a windshield); this project's synthetic scan keeps the
  nearest return only, as most consumer/robotics LiDAR drivers default to. `PRACTICE.md` section 1
  discusses the real packet formats.
- **Single revolution, no motion deskew.** The scan is a single static sweep with no platform motion —
  02.08 (motion deskew) is the sibling project that would sit upstream of this one on a moving robot.
- **The beta-criterion threshold is a single global constant.** Real systems often adapt it locally
  (surface-normal-aware variants, or a range-dependent threshold); THEORY.md "Where this sits in the
  real world" names the direction production systems take this.
- **The Euclidean comparison is a reduced-scope reimplementation**, not 02.04 itself (no min-label
  propagation baseline, a single fixed tolerance) — sufficient for the timing/behavior contrast this
  project needs, not a substitute for 02.04's own, fuller teaching arc.
- This project computes no control or planning output and commands no hardware motion; nonetheless, per
  repo policy, note explicitly: everything here is **sim-validated only, not safety-certified**, and any
  downstream use commanding real hardware motion is the integrator's decision and responsibility.
