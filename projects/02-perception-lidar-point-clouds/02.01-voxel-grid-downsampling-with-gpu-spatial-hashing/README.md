# 02.01 — Voxel-grid downsampling with GPU spatial hashing

**Difficulty:** ★ beginner · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `★ Voxel-grid downsampling with GPU spatial hashing`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> This project only reads a synthetic point cloud and writes a smaller one; it commands no hardware,
> so the sim-validated-only caveat (CLAUDE.md §1) is inherited by its downstream consumers, not
> triggered here directly.

## Overview

This project is the first kernel a real LiDAR-fed robot pipeline runs on every scan: replace ~200k raw
points with a much smaller set of one-centroid-per-occupied-voxel representatives, on a fixed 20 cm
grid. It is built and verified **two ways at once** — an atomic open-addressing GPU hash table (Method
A, the catalog bullet's "spatial hashing") and a Thrust sort + fixed-order segmented reduction (Method
B) — so the same result can be checked from two different data-structure angles, and so the project
becomes a hands-on lesson in a genuinely subtle GPU topic: which parts of a parallel reduction are
reproducible run-to-run, and which are not, and why. The demo downloads nothing, downsamples a
committed 198,534-point synthetic scan (a 16-beam spinning-LiDAR sweep of an analytic room, plus two
deliberately adversarial point clusters designed to stress a hash table in opposite ways), verifies both
methods against independent CPU oracles, runs three physically/logically independent gates, quantifies
the hash table's probing behavior and each method's determinism with **repeated runs**, and writes a
before/after top-view render plus two CSV artifacts. Both the catalog's named method (spatial hashing)
and its natural, more-deterministic alternative (sort-based reduction) are fully implemented — nothing
in this bullet is documented-only.

## What this computes & why the GPU helps

**The computation:** partition N points into voxels of edge L via `floor(p/L)` per axis, then replace
every occupied voxel's points with their mean position (the centroid). Two GPU-parallel strategies for
the "which points share a voxel, and what is their sum" step:

- **Method A — a SCATTER pattern.** Every point independently computes where it wants to land (its
  voxel's hash-table slot) and races to get there with atomics — a scatter/hash-insert, the canonical
  shape for "unknown-size groups, fill them as you go."
- **Method B — a SORT-THEN-SEGMENTED-REDUCE pattern.** Every point's voxel key is computed, the whole
  array is sorted by key (turning "same voxel" into "adjacent in the sorted array"), then one thread per
  voxel walks its own contiguous run and sums it — a completely different, ORDER-based way to solve the
  identical grouping problem.

Both are embarrassingly parallel across N ~ 200k points (map + hash-insert or map + sort + segmented
reduce), the bottleneck a plain nested CPU loop would hit is exactly the O(N) or O(N log N) work every
GPU thread instead does concurrently, and — because LiDAR density falls off as 1/r² with range (derived
in `THEORY.md`) — the SIZE of the problem this collapses (200k points down to ~7k voxels on this sample)
is what makes every downstream O(N) or O(N²) perception algorithm affordable.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** perception — the FIRST point-cloud-processing stage after the LiDAR driver, inside
  SYSTEM_DESIGN's "POINT-CLOUD PERCEPTION [02 →] downsample, ground seg, clustering, deskew" box
  (§1, per-scan budget < 50 ms).
- **Upstream inputs:** a raw `PointCloud` (SYSTEM_DESIGN §3.6 message shape) from the LiDAR driver —
  the calibrated, deskewed points a real driver stack (project-class `02.08` per-point motion deskew,
  `02.20`-class intensity/channel calibration) hands off; this project's synthetic scan stands in for
  that driver's output.
- **Downstream consumers, named as message-shaped hand-offs:** the downsampled `PointCloud` feeds
  **[`02.06`](../02.06-icp-point-to-point-point-to-plane-gicp/README.md) ICP** directly (both projects
  use the identical interleaved-`float* xyz`-in-meters layout — SYSTEM_DESIGN's composition-map Chain A
  names exactly this hand-off), **[`02.04`](../02.04-euclidean-clustering-via-gpu-union-find/README.md)**
  Euclidean clustering (fewer points, same clusters, far cheaper neighbor search),
  **[`02.07`](../02.07-ndt-scan-matching/README.md)** NDT scan matching (which voxelizes anyway — this
  project's grid can seed NDT's), and **[`05.01`](../../05-slam-mapping-localization/05.01-tsdf-fusion-marching-cubes-mesh-extraction/README.md)**
  TSDF mapping (fusing fewer, more even points reduces integration cost per frame).
- **Rate / latency budget:** LiDAR delivers 10–20 Hz (SYSTEM_DESIGN §1.1); this stage must fit
  comfortably inside that scan period, well under the 50 ms per-scan budget the perception box above is
  given — this project's GPU Method A/B runs measure **well under 4 ms** at 198,534 points on an RTX
  2080 SUPER (`demo/expected_output.txt`'s companion `[time]` lines; see "Expected output" below).
- **Reference robot(s):** the warehouse AMR and the autonomous-vehicle stack (SYSTEM_DESIGN §2) — both
  name voxel/downsample as the first LiDAR-perception block; a 6-DoF manipulator cell would use this if
  it carries a scene-scanning LiDAR rather than only an eye-in-hand depth camera.
- **In production:** a tuned, vendor- or library-provided voxel filter (PCL's `VoxelGrid`, Open3D's
  `voxel_down_sample`, or a GPU path inside nvblox/cuPCL-class pipelines) — see "Prior art" below.
- **Owning team:** perception (the team that owns the LiDAR driver and the first few pipeline stages
  before hand-off to localization/mapping and planning teams — SYSTEM_DESIGN item 5).

## The algorithm in brief

- **Voxel key computation** — `floor(p/L)` per axis (mind the negative-coordinate truncation pitfall),
  packed into a 64-bit integer (21 bits/axis, biased) — [`THEORY.md` "The math"](THEORY.md#the-math).
- **Method A: open-addressing GPU hash table** — Teschner et al. 2003 spatial hash, linear probing,
  atomicCAS claim-or-probe insert, atomicAdd accumulation, atomic-counter compaction —
  [`THEORY.md` "The GPU mapping"](THEORY.md#the-gpu-mapping).
- **Method B: Thrust sort + fixed-order segmented reduction** — `thrust::stable_sort_by_key` (radix
  sort), boundary detection + stream compaction (`thrust::reduce` / `thrust::copy_if`), one thread per
  voxel doing a sequential fixed-order float sum — [`THEORY.md` "The GPU mapping"](THEORY.md#the-gpu-mapping).
- **Determinism, quantified, not assumed** — Method A's float-atomicAdd order is scheduler-dependent
  (measured, not just asserted); Method B's fixed summation order makes it bit-exact against a CPU twin
  and across repeated GPU runs — [`THEORY.md` "Numerical considerations"](THEORY.md#numerical-considerations).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/voxel-grid-downsampling-with-gpu-spatial-hashing.sln`](build/voxel-grid-downsampling-with-gpu-spatial-hashing.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/voxel-grid-downsampling-with-gpu-spatial-hashing.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

**Optional dependency:** Thrust (`thrust::stable_sort_by_key`/`reduce`/`copy_if`, Method B's sort +
compaction) — header-only, part of the CUDA Toolkit, no extra install or link step. **Toolchain note:**
CUDA 13.3's Thrust/CCCL headers require MSVC's conforming preprocessor and an explicit C++17 request to
nvcc's device-side front end; `build/*.vcxproj`'s `CudaCompile` sections pass `/Zc:preprocessor`,
`/Zc:__cplusplus`, and `-std=c++17` with a full explanation of why each is needed — nothing to configure
by hand, but worth reading if you ever see a `CCCL`-flavored compile error in your own Thrust-using
project.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample (`data/sample/lidar_scan.bin`, ~2.3 MiB, 198,534 points) is 100% synthetic
(CLAUDE.md §8 default): a 16-beam spinning-LiDAR scan of an analytic 16×16 m room (ray-cast against a
floor, four walls, and three obstacle boxes — no library, hand-rolled slab/plane intersection), plus two
adversarial regions (a 3,000-point dense cluster and a 150-point sparse grid) appended to stress-test the
hash table. Regenerate with `python scripts/make_synthetic.py --seed 42` (byte-identical to the
committed file). Full provenance, exact field layout, and the measured composition (revolutions cast,
hit rate, etc.) are in [`data/README.md`](data/README.md).

## Expected output

An actual run on the reference machine (RTX 2080 SUPER, sm_75, Release|x64) produces:

- **`VERIFY(keys)`, `VERIFY(method_b)`: PASS, bit-exact** — GPU voxel keys and Method B's entire output
  match their CPU twins to the last bit (198,534 keys checked one by one; 7,132 voxels checked row by
  row).
- **`VERIFY(method_a)`: PASS** — Method A's atomic-hash centroids agree with an *independent* CPU oracle
  (a `std::unordered_map`, double precision, sequential accumulation order — genuinely different
  structure/order/precision from the GPU path) to within **1.0×10⁻⁴ m**, measured-then-margined ~30×
  over the actual observed worst case of **3.34×10⁻⁶ m** (see `THEORY.md` "Numerical considerations" for
  where that number comes from).
- **All 4 gates PASS**: `cross_method_agreement` (A vs B agree to 4.29×10⁻⁶ m; occupancy exactly equal
  at 7,132 voxels both ways), `partition_invariant` (every one of 198,534 points accounted for exactly
  once, both methods), `centroid_containment` (every centroid geometrically inside its own voxel, both
  methods), `determinism_method_b` (3 independent runs, byte-identical output).
- **Honest, not gated, numbers** (`[info]` lines — see `demo/out/gates_metrics.csv` for every one):
  Method A's own 3-run determinism delta (~4–8×10⁻⁶ m — present, small, and reported, not hidden);
  occupancy 7,132 voxels vs a naive uniform-volume estimate of 96,550 (ratio 0.074 — LiDAR returns lie on
  a thin surface, not a filled volume); hash load factor 0.0136, mean probe length 0.072 (normal scan)
  vs 0.002 (adversarial region — see "Limitations" for the honest, slightly counter-intuitive story
  behind that number); downsample RMS distance 0.057 m (vs the 0.10 m "L/2" back-of-envelope intuition).
- Timings (single-shot, teaching artifacts, never a benchmark claim): GPU Method A ≈ 0.11 ms, GPU
  Method B ≈ 1.4–3.4 ms, CPU twins 1.3–7.3 ms.

The canonical stable-line contract lives in [`demo/expected_output.txt`](demo/expected_output.txt).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the shared contract: point-cloud layout, voxel-key packing,
   the spatial hash, the hash-table struct, every kernel/launcher/CPU-twin declaration, and WHY this
   header carries no `__host__ __device__` (read this first — it explains the whole project).
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels: `hash_insert_kernel` (Method A's atomicCAS
   claim-or-probe loop, the project's central new idea) and `launch_sort_based_downsample` (Method B's
   Thrust orchestration + `segmented_reduce_kernel`'s fixed-order sum).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the three CPU twins, and (in its file header) the
   independence ruling explaining exactly which of the three is bit-exact-by-design, which is
   independent-by-design, and why.
4. [`src/main.cu`](src/main.cu) — orchestration: load data, run both methods (three times each), every
   VERIFY/GATE, the hash-stats/occupancy/downsample-quality measurements, and the artifact writers.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the scene, the beam model (cited from
   01.18), and the two adversarial regions.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/output path resolution.

## Prior art & further reading

- **PCL `VoxelGrid`** — the textbook CPU reference implementation this project's *definition* of
  "voxel downsampling" matches (centroid-per-occupied-voxel); study it for its leaf-size auto-tuning and
  its `pcl::VoxelGridCovariance` variant that keeps per-voxel covariance for NDT-style consumers.
- **Open3D `voxel_down_sample` / `voxel_down_sample_and_trace`** — a modern, well-documented CPU
  implementation with the `_and_trace` variant that returns exactly the point→voxel mapping this
  project's `partition_invariant`/`downsample_quality` gates compute internally but do not expose as an
  API — worth reading for how a production library shapes that as a public contract.
- **cuPCL / nvblox-class GPU preprocessing** — NVIDIA's GPU point-cloud libraries implement voxel
  filtering (among other stages) as one kernel in a larger fused pipeline; study these for how a
  production stack avoids the extra host round-trips this teaching project deliberately keeps explicit.
- **Teschner et al. 2003, "Optimized Spatial Hashing for Collision Detection of Deformable Objects"** —
  the spatial hash function (large-prime XOR/multiply) Method A's `spatial_hash()` implements verbatim;
  read it for the collision-probability analysis behind the prime choice.
- **NVIDIA Thrust / CUB documentation** — `thrust::stable_sort_by_key`'s radix-sort internals and
  `cub::DeviceSegmentedReduce` (the production-grade, load-balanced version of this project's
  one-thread-per-voxel `segmented_reduce_kernel` — see `THEORY.md` "Where this sits in the real world").

## Exercises

1. **Tune the leaf size.** Re-run with `kVoxelLeafM` set to 0.05 m and 0.50 m (edit `kernels.cuh`,
   rebuild). Watch `occupancy_analytics` and `downsample_quality` move in opposite directions — smaller
   leaf → more voxels, smaller RMS error. Where does the point-count reduction stop being worth it?
2. **Break the floor-vs-truncate fix on purpose.** Temporarily change `voxel_coord()` in `kernels.cuh`
   to use `(int32_t)(p/leaf)` instead of `std::floor`, rebuild, and watch which gate catches it first
   (hint: it will not be `VERIFY(keys)` alone — reason about why `centroid_containment` is the one that
   would actually fire, and for which points).
3. **Force a hash collision cluster.** Shrink `HashTableGPU`'s capacity sizing (lower `kTargetLoadFactor`
   in `kernels.cuh` toward 0.9) and re-run — watch `hash_stats`'s mean/max probe length climb, and relate
   the change to the birthday-bound arithmetic in `THEORY.md` "The math".
4. **Replace Method A's compaction** with a `thrust::copy_if`-based stream compaction (mirroring Method
   B's own boundary compaction) instead of the atomic-counter push-back `hash_compact_kernel` uses — does
   it change any VERIFY/GATE result? Should it?
5. **Add a KD-tree nearest-voxel query** (project `02.05`'s territory) instead of `downsample_quality`'s
   "a point's own voxel is its nearest representative" shortcut, and confirm the two give the same
   answer (they must, given `partition_invariant` — this exercise is really about seeing WHY they must).

## Limitations & honesty

- **The scan is synthetic, labeled everywhere it appears** (`data/README.md`, `DATA:` line, every
  artifact) — an analytic room + boxes, not a recording. See `PRACTICE.md` §1 for what a real LiDAR
  packet stream looks like in contrast.
- **6 accumulated revolutions, not 1.** A single realistic 16-beam sweep produces roughly 30–40k points
  per revolution, well short of the N ~ 200k this catalog bullet scopes the project at; the sample
  accumulates 6 independent (independently-noised) sweeps of a static scene, standing in for a short
  integration dwell — an honest, documented scope choice (`scripts/make_synthetic.py`'s module
  docstring), not a claim that any one real sweep contains this many points.
- **Method A's compaction is the simplest correct one, not the fastest.** A single atomic counter for
  the whole table (rather than a scan-based, more work-efficient compaction like Method B's own boundary
  step) is a deliberate "teach the pattern once" choice — `kernels.cu`'s `hash_compact_kernel` comment
  names the trade-off.
- **Method B's per-voxel reduction has load imbalance by design.** One thread walks a whole voxel's
  point run sequentially — cheap and bit-exact-friendly, but the thread covering the 3,000-point dense
  cluster's voxel does far more work than its warp-mates. `THEORY.md` "Where this sits in the real world"
  names `cub::DeviceSegmentedReduce` as the production fix, and why this project does not use it.
- **The adversarial regions did not stress the hash table the way their name suggests — and that itself
  is the honest finding, reported, not hidden.** `hash_stats` measured mean/max probe length LOWER on
  the adversarial index range than on the normal scan (0.002/1 vs 0.072/4). The reason: probe length is
  driven by how many DISTINCT voxel keys collide at the same hash slot, not by how many points share one
  key — repeated inserts of the SAME key resolve on the very first probe (an `old == my_key` match), so
  cramming thousands of points into one voxel is nearly free for probing, and this project's generously
  sized table (capacity from worst-case N, not from the much smaller actual occupancy — load factor
  0.0136 in practice) leaves little room for real collisions anywhere. See `THEORY.md` "The math" for
  the arithmetic and Exercise 3 above for how to force the effect the catalog bullet's name evokes.
- **Sim-validated only, not safety-certified** (CLAUDE.md §1) — this project itself commands no
  hardware, but every downstream consumer named in "System context" that *does* command motion inherits
  that caveat, and `PRACTICE.md` says so at the point it matters.
