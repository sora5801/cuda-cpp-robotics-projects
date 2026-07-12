# 02.05 — KD-tree or LBVH construction + KNN/radius search on GPU

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `KD-tree or LBVH construction + KNN/radius search on GPU`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds a **linear BVH (LBVH)** — a spatial index tree — over a ~200,000-point LiDAR scan
**entirely on the GPU, from scratch, every scan**: Morton (Z-order) codes → GPU sort → Tero Karras's
parallel radix-tree construction → bottom-up AABB propagation, with no host involvement and no
recursion anywhere. It then answers two kinds of neighbor query through that tree — **fixed-radius
search** and **K-nearest-neighbor (KNN, K=8)** — and contrasts both, honestly and measurably, against
the fixed-radius **voxel-hash** technique this domain already teaches in projects
[`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md) and
[`02.04`](../02.04-euclidean-clustering-via-gpu-union-find/README.md). Every build stage and every
query path is checked against an independently-written CPU twin, and a third, tree-free O(N·Q)
brute-force oracle catches anything a shared bug between a kernel and its twin might hide.

The catalog bullet's two named techniques are both implemented, not just one: the **LBVH** is the
project's primary subject (the "new" GPU-programming idea it teaches — parallel-from-scratch tree
construction), and the **voxel-hash fixed-radius search** (the KD-tree-adjacent alternative the bullet
also names) is reimplemented compactly as the deliberate point of comparison, reusing this domain's
own established sort+segment machinery. On the committed data: a radius search at `r = 0.5 m` finds
**1,215 neighbors** in a designed dense cluster and **0** in a designed sparse region at the identical
radius — the physically-motivated failure mode THEORY.md derives from LiDAR's 1/r² density falloff —
while KNN finds exactly **8** neighbors at both locations, just much farther away at the sparse one.
That is this project's whole engineering lesson, made numeric.

## What this computes & why the GPU helps

**Construction** is a *sort + independent-per-node build*: every point's 3-D position becomes a 30-bit
Morton code (a `map`), the array is sorted once (a well-known parallel primitive, `thrust::sort`), and
then — the genuinely new idea — every one of the tree's N−1 internal nodes computes its own two
children **independently**, from the sorted array alone, with **zero dependency on any other node's
result**. That independence is what makes the whole tree buildable in one parallel pass with no
recursion and no per-level synchronization; the bottleneck it removes is the SERIAL, level-by-level
dependency a textbook top-down tree build has (each level must finish before the next starts).

**Query** (radius search and KNN) is an embarrassingly parallel `map`: one GPU thread owns one query
end-to-end (traverses the tree, collects results, sorts them) — 2,000 independent problems, one thread
each, no communication between threads at all.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — infrastructure layer. Neighbor search is not a stage of the
  pipeline that produces its own message; it is the inner-loop primitive **half the domain's other
  stages are built on top of**. Named consumers in this very domain: **02.09** (normal/curvature
  estimation — every point's normal is a plane fit through its K or radius neighbors); **02.10** (FPFH
  descriptors — a histogram over each point's neighborhood, computed twice per point at two radii);
  **02.06** (ICP — nearest-neighbor correspondence search between two clouds every iteration); **02.04**
  (Euclidean clustering — a fixed-radius neighbor query IS the graph-edge step that union-find runs on).
  This project ships the index those stages assume already exists.
- **Upstream inputs:** a `PointCloud`-shaped buffer (SYSTEM_DESIGN.md §3.6) — raw or already
  downsampled/ground-filtered (02.01/02.03's output slots directly in) — plus a set of query points
  (often the SAME cloud, self-querying for per-point neighborhoods).
- **Downstream consumers:** any stage that asks "what is near this point" — see the four named above,
  plus loop-closure search (02.11), moving-object segmentation (02.14), and any learned point-cloud
  network's neighborhood-pooling layer (12.xx).
- **Rate / latency budget:** a spinning LiDAR delivers a new scan at 10–20 Hz (50–100 ms/scan budget,
  SYSTEM_DESIGN.md §1). On this project's committed 199,404-point scan, the FULL LBVH rebuild (Morton +
  sort + radix-tree + AABB) measures **~4–5 ms** on the reference GPU — comfortably inside that budget,
  with room left for the queries themselves (measured throughput below). A production perception stack
  would rebuild the tree once per scan and answer thousands to millions of queries against it before the
  next scan arrives.
- **Reference robot(s):** the **autonomous-vehicle stack** (domains 01/02/03/04/05/06/14/31/32 —
  SYSTEM_DESIGN.md's reference robot 5) and the **warehouse AMR** (02/04/05/23/06/08/25/31/32 —
  reference robot 1) both depend on fast neighbor search: the AV's registration/segmentation pipeline
  runs at full LiDAR rate; the AMR's local costmap and clustering stages query a smaller, slower-moving
  cloud but on cheaper embedded compute, where the hash-vs-tree tradeoff this project measures matters
  even more.
- **In production:** PCL's `KdTreeFLANN` (CPU, k-d tree) or a GPU library such as cuML's/FAISS's
  neighbor search (README "Prior art") would typically replace a hand-rolled LBVH; NVIDIA's own OptiX
  and nvblox use LBVH-family structures internally for ray tracing and TSDF neighbor queries
  respectively — this project's construction is the same *family* of algorithm, taught from first
  principles.
- **Owning team:** perception infrastructure / "core" perception — the unglamorous team that every
  perception feature team (segmentation, registration, tracking) depends on and rarely thinks about
  until it is too slow (PRACTICE.md §4 names this explicitly).

## The algorithm in brief

- **Morton (Z-order) encoding** — interleave 10-bit-per-axis quantized coordinates into one 30-bit
  code; sorting by this code approximates a spatial sort (THEORY.md "The math").
- **Karras's parallel radix-tree construction** (2012) — every internal node's range and children
  computed independently via a longest-common-prefix search over the sorted key array; the central new
  idea this project teaches (THEORY.md "The algorithm", "The GPU mapping").
- **Bottom-up AABB propagation** via an atomic "second-arrival" race — every leaf climbs its parent
  chain; a `__threadfence()`-guarded atomic counter lets exactly one thread per internal node compute
  its bounding box, once, race-free (THEORY.md "Numerical considerations").
- **Stack-based BVH traversal** — radius search (fixed threshold) and KNN (a bounded max-heap with a
  shrinking prune radius) — both a small, fixed-size per-thread stack whose depth is bounded by a
  bit-counting proof, not a balance guarantee (THEORY.md "The GPU mapping", contrasted with
  [`11.01`](../../11-sensor-sim-digital-twins/11.01-gpu-lidar-simulator/README.md)'s top-down BVH).
- **Fixed-radius voxel hashing** — the domain's existing technique (02.01 Method B / 02.04 lineage):
  sort points into a spatial hash keyed by voxel, then a 27-cell stencil + binary search per query.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/kd-tree-or-lbvh-construction-knn-radius-search.sln`](build/kd-tree-or-lbvh-construction-knn-radius-search.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/kd-tree-or-lbvh-construction-knn-radius-search.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none beyond the CUDA toolkit + C++17 standard library.
`kernels.cu` uses **Thrust** (header-only, part of the CUDA Toolkit) for the sort and boundary-compaction
stages; no extra library file is linked (see the `.vcxproj`'s `CudaCompile` comments for the
`/Zc:preprocessor` / `/Zc:__cplusplus` MSVC flags Thrust's headers require under CUDA 13.3, a known
interaction root-caused in project 02.01 and reused here verbatim).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample (`data/sample/lbvh_scan.bin`, ~2.3 MiB) is **synthetic** (CLAUDE.md §8 default):
the same 16-beam spinning-LiDAR room scan project 02.01 pioneered (cited, reimplemented — not
imported, per the self-containment rule), extended with two adversarial regions and 2,000 query points
tuned specifically for this project's KNN-vs-fixed-radius contrast. `scripts/make_synthetic.py`
regenerates it byte-for-byte from seed 42. Full provenance, field format, and checksums in
[`data/README.md`](data/README.md).

## Expected output

The demo builds the LBVH on both GPU and CPU, verifies bit-exact agreement at every stage (Morton
codes, sort order, radix-tree topology, AABBs — all exact, not tolerance-bounded; see THEORY.md
"Numerical considerations" for why), runs 2,000 radius/KNN queries through the tree and the voxel-hash
baseline, cross-checks GPU against three independent tiers (a CPU tree twin, a CPU hash twin, and a
tree-free CPU brute-force oracle), and gates a designed density-contrast scenario. On the reference
machine (RTX 2080 SUPER), every `VERIFY`/`GATE` line reads `PASS` and the demo writes a top-view PNG-like
PPM plus three CSVs to `demo/out/`. Canonical stable lines: [`demo/expected_output.txt`](demo/expected_output.txt).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the data-layout contract: read the long file-header comment
   first — it is the map of the whole project (the four-stage pipeline, the LBVH node layout, the
   contrast with 11.01's top-down BVH, and the domain contrast with voxel hashing).
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels, in pipeline order: Morton encoding,
   `build_radix_tree_kernel` (the Karras construction — the heart of the project), `propagate_aabb_kernel`
   (the atomic second-arrival trick, and its `__threadfence()` — read that comment), the two BVH
   traversal kernels, then the voxel-hash baseline.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — start with its file header: the independence
   ruling explains exactly which functions are shared data-layout arithmetic and which are
   independently retyped algorithmic twins, and why the brute-force oracle at the bottom exists as a
   THIRD, structurally unrelated verification tier.
4. [`src/main.cu`](src/main.cu) — orchestration: load data, run every stage on GPU and CPU, every
   `VERIFY`/`GATE` line, the artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h` (data/artifact resolution).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"** (HPG
  2012) and its companion **NVIDIA Developer Blog series "Thinking Parallel, Part III: Tree
  Construction on the GPU"** — the exact construction algorithm this project implements; read the
  original for the full generality (LBVH, octrees, and k-d trees from the SAME sorted-key idea).
- **PCL's `KdTreeFLANN`** — the CPU production baseline for radius/KNN search most robotics stacks
  reach for first; study its API shape (this project's query signatures echo it) and its FLANN backend.
- **cuML's / FAISS's GPU neighbor search** — production GPU nearest-neighbor libraries at a much larger
  scale (millions to billions of points, approximate methods); study their batching and index-choice
  tradeoffs once this project's exact, from-scratch version is understood.
- **NVIDIA OptiX** — production GPU ray tracing built on LBVH-family acceleration structures; this
  project's construction is the same algorithm family OptiX's BVH builder uses internally, at a much
  smaller, fully-visible scale.
- **nvblox** — NVIDIA's GPU TSDF-mapping library; its voxel-hash and neighbor-query internals are the
  production-grade descendant of this project's voxel-hash baseline.
- Project **11.01** (this repo) — the hand-built, top-down, median-split BVH this project is
  deliberately contrasted against; read its `THEORY.md` "The algorithm" for the depth-bound argument
  this project's own bit-counting bound is compared to.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Nearer-child-first traversal.** Both traversal kernels push both children unconditionally; a
   production traverser visits the AABB-nearer child first so the farther one can be culled sooner by a
   tightened bound. Measure the change in `traversal_stats.csv`'s mean nodes-visited.
2. **Variable radius per query.** Extend `radius_search_bvh_kernel` to take a per-query radius array
   instead of one global `kRadiusM`, and re-run the density-contrast comparison with a radius that
   scales with range (THEORY.md "The problem" derives the physically-motivated formula) — this is
   exactly the query shape the voxel-hash baseline CANNOT serve without rebuilding at a new leaf size.
3. **K as a runtime parameter.** `kQueryK` is a compile-time constant sized for register-resident
   heaps. Make it a launch parameter with a `local`-memory heap instead, and measure the throughput
   cost of larger K.
4. **Approximate KNN via early termination.** Stop the KNN traversal once the heap has been full for M
   consecutive prunable subtrees without improving — measure the accuracy/speed tradeoff against the
   exact brute-force oracle.
5. **Expanding-ring hash KNN.** Give the voxel-hash baseline a KNN mode by growing the stencil radius
   ring by ring until K points are found (THEORY.md "Where this sits in the real world" names this as
   the classical fix) — then compare its throughput against the LBVH's KNN honestly.

## Limitations & honesty

- **No rebuild-vs-refit choice.** The tree is rebuilt from scratch every call; a streaming system would
  consider refitting (recomputing only AABBs, keeping topology) between full rebuilds — not implemented
  here (PRACTICE.md §1 discusses the tradeoff).
- **KNN heap is register/local-memory bound at compile-time K=8.** A much larger K would spill more
  aggressively; not measured here (see Exercise 3).
- **The voxel-hash baseline only supports ONE fixed radius per run** (`leaf == kRadiusM`), by design —
  the whole point of the domain contrast. A real system needing both fast fixed-radius AND adaptive
  queries would maintain both structures, as this project's own comparison demonstrates is sometimes
  the right engineering call.
- **Radius-search result canonicalization (insertion sort) is O(k²) in the result count.** Fine for
  typical/sparse results, a measurable cost for the adversarial dense-cluster query (visible in
  `traversal_stats.csv` and the query-throughput numbers) — a documented, deliberate simplicity-over-
  speed choice for exact-equality verification, not a production design (THEORY.md "Numerical
  considerations").
- **Synthetic data only; not safety-certified.** This project's neighbor-search results could feed a
  real robot's perception pipeline, but nothing here is validated beyond the synthetic scene and the
  gates above — sim-validated only, per CLAUDE.md §1.
