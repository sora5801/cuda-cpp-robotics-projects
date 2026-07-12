# 02.04 — Euclidean clustering via GPU union-find / connected components: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

A spinning LiDAR does not report "there is a pedestrian at 12 meters" — it reports a cloud of
independent range returns, each one a photon time-of-flight measurement off *some* surface. Turning that
cloud into "there is one object here, another object there" is an inference the sensor never makes for
you; it is entirely the job of the software downstream, and it rests on a single physical prior:

> **Physical contiguity.** A rigid or articulated real-world object is (mostly) spatially CONTIGUOUS —
> its surface returns are close to each other in 3-D space, and there is (usually) a visible gap of
> empty space between it and the next object.

This prior is doing real work, and it is worth stating exactly where it can fail, because both failure
modes are load-bearing lessons of this project:

1. **Under-segmentation (chaining).** Two DIFFERENT physical objects can have points that happen to lie
   closer to each other than to their own object's far side — a pedestrian standing near a wall, a
   shopping cart touching a person's leg, two parked cars whose mirrors nearly touch. A purely
   *geometric* proximity rule cannot tell these apart from a single object with an unusual shape; the
   physical-contiguity prior alone is simply not enough information. This project's **chaining test**
   builds exactly this failure into the synthetic scene on purpose, and *asserts that it happens* — the
   honest lesson is not "avoid this," it is "know that single-linkage clustering does this, and that
   production systems layer something else on top to fix it" (see "Where this sits in the real world").
2. **Over-segmentation (occlusion splitting).** A SINGLE physical object can be occluded by something in
   front of it, or self-occluded by its own geometry (a car's near side blocks LiDAR from seeing gaps
   *through* it, but a pedestrian's legs might scan as two separate blobs at certain ranges/angles), so
   its visible returns split into two spatially separated point groups that a purely geometric rule
   reports as two objects. This project's scene does not manufacture this case directly (it needs
   temporal/semantic context this repo's later 04.xx/12.xx projects add), but it is named here as the
   dual failure mode every learner should know about symmetrically with chaining.

**Engineering constraints a real system imposes on top of the physics:** the point-to-point distance
threshold `d` this project calls the "cluster tolerance" has to be tuned against LiDAR RANGE-DEPENDENT
POINT DENSITY (a beam's angular resolution means point spacing on a surface grows roughly linearly with
range — 02.01's THEORY.md derives the 1/r^2 area-density falloff this follows from), against the
sensor's own RANGE NOISE floor (too tight a `d` and single-surface returns fail to connect due to noise;
too loose and unrelated nearby surfaces merge), and against the LATENCY BUDGET a 10–20 Hz scan rate
imposes (README "System context" — a clustering stage that cannot keep up with the sensor is worse than
useless, it is a growing queue).

## The math

**Single-linkage clustering, stated precisely.** Given a point set `P = {p_0, ..., p_{n-1})` in R^3 and
a distance threshold `d > 0`, define an undirected graph `G = (V, E)` with `V = P` and

```
E = { (p_i, p_j) : i != j, ||p_i - p_j||_2 <= d }
```

Then **Euclidean clustering with tolerance `d` is, by definition, computing the connected components of
`G`** — two points are in the same cluster if and only if there is a PATH between them in `G`, i.e. a
chain of points each within `d` of the next (this is exactly what makes it "single-linkage": a cluster
only needs ONE close pair anywhere along a chain to bridge two otherwise-distant regions — the formal
statement of the chaining hazard above). This project's job is computing this partition, fast, on
thousands of points, and doing it TWO independent algorithmic ways to compare their behavior.

**The canonical-labeling convention.** A connected component's CANONICAL id, throughout this project, is
the MINIMUM point index among its members. This is not an arbitrary choice: it is the value BOTH
algorithms below naturally converge to (union-by-min for union-find; min-label flooding for label
propagation), which is exactly what lets `main.cu` compare their outputs to each other and to the
generator's independently-computed ground truth with a single, bit-exact equality test — no relabeling,
no isomorphism search, no Hungarian-algorithm matching needed.

**Why the 27-cell voxel stencil is EXACT, not approximate, when leaf `L >= d`.** Bin every point into a
voxel `(floor(x/L), floor(y/L), floor(z/L))`. Claim: if `||p - q|| <= d <= L`, then `p` and `q`'s voxel
indices differ by at most 1 along every axis. Proof: for any single axis `a`, `|p_a - q_a| <= ||p-q|| <=
d <= L` (a coordinate difference along one axis is never larger than the full Euclidean distance).
Suppose the voxel indices differed by 2 or more along axis `a`; then there would have to be at least one
FULL voxel width of separation between the two points' positions along that axis, forcing `|p_a - q_a|
>= L` — contradicting the bound above (strict for a difference of 2+; the boundary case of exactly `L`
is handled by the half-open voxel convention `[k*L, (k+1)*L)`, see 02.01's `voxel_coord` derivation,
cited). So every axis's voxel index differs by at most 1, i.e. `q` lies in the 3x3x3 block of voxels
centered on `p`'s voxel — 27 cells including `p`'s own. Setting `L = d` exactly (this project's choice)
is the TIGHTEST valid leaf size: any smaller and the proof breaks (an adjacent-voxel pair could then be
farther than `d` apart, missing real edges — this project does not do that); any larger and voxels hold
more points each for no correctness benefit, only more wasted stencil-scan work.

**Union-find complexity — the classical (sequential) result.** With path compression AND union by
rank/size, a sequence of `m` union/find operations on `n` elements takes `O(m * alpha(n))` total time,
`alpha` the inverse Ackermann function (Tarjan 1975). `alpha(n) <= 4` for any `n` representable in this
universe (it grows unimaginably slowly — even `n` = the number of atoms in the observable universe gives
`alpha(n) <= 5`), so in practice this is "amortized constant time per operation." This is a SEQUENTIAL
result; it says nothing directly about a parallel GPU implementation's SWEEP count, which is a different
question this project answers empirically (see "The algorithm" below) and grounds in the pointer-jumping
literature (Shiloach & Vishkin 1982) rather than restating Tarjan's proof, which does not transfer
directly to the batch-parallel setting.

**Label propagation's O(diameter) bound, derived.** Define the graph diameter `D` = the longest
shortest-path (in graph hops) between any two connected vertices. In one sweep of
`lp_sweep_kernel`, a vertex's label can only decrease by "hearing about" a smaller label from a DIRECT
neighbor (one hop). By induction, after `k` sweeps, the smallest label in a component has propagated to
every vertex within `k` hops of its origin. For the FARTHEST vertex (at hop-distance `D` from the
component's minimum-index vertex, in the worst case), full convergence therefore needs AT LEAST `D`
sweeps — this project's snake is engineered (see `scripts/make_synthetic.py`) to be a near-exact PATH
GRAPH (each interior point has exactly 2 neighbors), whose diameter equals `point_count - 1` exactly,
making this bound tight and its 299-sweep measurement essentially the theoretical prediction, not an
approximation of it.

## The algorithm

**Pipeline, step by step** (see `src/kernels.cuh`'s file header for the full version with kernel names):

1. Compute each non-ground point's voxel key at leaf = `d` (02.01 lineage).
2. Sort points by voxel key (Thrust `stable_sort_by_key`), mark run boundaries, compact into a dense
   `(unique_key, seg_start)` index — 02.01 Method B's exact pipeline, reused for a different purpose.
3. For each point, scan its 27-voxel stencil (binary search over `unique_key`), test each candidate's
   actual squared distance, and atomically append qualifying `(i, j)`, `i < j` edges.
4. **Method A — GPU union-find:** repeated parallel sweeps over the edge list; each sweep, every edge's
   two endpoints `find()` their roots (with path halving) and, if different, `union()` them (union-by-min
   via a lock-free atomicCAS retry loop). Repeat until a sweep changes nothing. Finalize: one more full
   compression pass so every point's parent is DIRECTLY its true root.
5. **Method B — GPU label propagation:** the SAME edge list, repeated parallel sweeps of "flood the
   smaller label across every edge" (`atomicMin` in both directions), until nothing changes. No
   compression of any kind.
6. **Relabel + stats + filter:** sort-by-root + `inclusive_scan` (a second, different Thrust compaction
   idiom from step 2's `copy_if`) turns sparse canonical roots into dense `[0,K)` ids; per-cluster
   count/centroid/AABB accumulate via atomics; components under `kMinClusterSize` are marked noise.

**Complexity, serial vs. parallel:** building the naive `O(n^2)` all-pairs distance graph is what a
first-principles implementation would do; the voxel stencil (step 3) turns this into `O(n * k)` where `k`
is the expected number of points within a small, bounded neighborhood (a scene-density-dependent
constant, not a function of `n` — see "The GPU mapping" below for the specific number this project's
scene produces). Both clustering algorithms (steps 4–5) do `O(E)` work PER SWEEP, `E` the edge count; the
CRITICAL DIFFERENCE this project measures is the NUMBER of sweeps: union-find needs `O(log D)` sweeps
(each sweep's path halving roughly halves every remaining path length, a pointer-jumping/list-ranking
argument — Shiloach & Vishkin 1982), while label propagation needs `O(D)` sweeps (no compression, one
hop per sweep, "The math" derives this exactly). On the committed scene (`D` ~= 298, the snake's
diameter), `log2(298) ~= 8.2` predicts single-digit-to-low-double-digit union-find sweeps; the MEASURED
number is 2 — even better than the log-bound predicts, because union-by-min's compression is more
aggressive than a plain list-ranking halving analysis assumes (many of the snake's internal unions
happen to be resolved in earlier sweeps thanks to the OTHER, non-snake edges converging first and
"seeding" useful compressed pointers) — an honest empirical note, not a claim the log bound is loose in
general.

## The GPU mapping

**Thread-to-data mapping, by kernel:**

- `build_edges_kernel`: one thread per POINT (a *gather*: reads many other points' positions, writes
  only its own contribution to a SHARED atomic-counter-indexed edge array). Memory behavior: `xyz` reads
  for the 27-stencil's candidate points are effectively RANDOM ACCESS (voxel neighbors are not adjacent
  in the array) — the opposite of a coalesced map like 02.01's SAXPY-style kernels; this is the honest
  cost of a spatial-index lookup, mitigated only by the small working-set size per point (few dozen
  candidates, see below) fitting comfortably in L2 cache across nearby threads in a warp.
- `uf_union_sweep_kernel` / `lp_sweep_kernel`: one thread per EDGE. `parent`/`label` arrays live in
  GLOBAL memory throughout (no shared-memory tiling): unlike a stencil kernel, which edge touches which
  two vertices is DATA-DEPENDENT and essentially random across the point index space, so there is no
  spatially-local tile to stage in shared memory the way a structured-grid kernel would. This is the
  honest cost of a GRAPH algorithm on a GPU: the memory-access PATTERN is exactly what makes graph
  algorithms harder to accelerate than dense linear algebra or image processing, and this project's
  measured 0.2–14 ms range (thousands of edges) versus what a compute-bound kernel of similar edge count
  would achieve is a live demonstration of that gap.
- Relabeling: `thrust::stable_sort_by_key` (radix sort under the hood — 02.01's kernels.cu explains what
  that computes) + `thrust::inclusive_scan` (a Blelloch-style work-efficient parallel prefix sum, `O(n)`
  work / `O(log n)` depth) — library calls chosen because a hand-rolled version would re-implement
  exactly these two well-known primitives with no teaching benefit beyond what 02.01/33.01 already cover
  for sort and scan respectively.

**Why `kMaxEdgesPerPoint = 256` (the density number "The algorithm" above promised):** the densest
region of the synthetic scene is a filled-lattice object at spacing `g = 0.15 m`; a point deep inside one
has every OTHER filled point within `floor(d/g) = 2` grid steps as a geometric candidate, and the count
of integer lattice points inside a Euclidean ball of radius 2 grid steps is bounded by the CONTINUOUS
ball volume `(4/3)*pi*2^3 ~= 33.5` — `src/kernels.cuh`'s comment on this constant shows the full margined
derivation to 256. MEASURED on the committed scene: 24,259 total edges over 1,469 points, an average of
under 17 edges per point — comfortably inside the bound with a wide safety margin (`edge_overflow_count`
in `demo/out/gates_metrics.csv` reads 0 every run).

**Memory hierarchy used:** global memory for every array (points, keys, edges, parent/label, stats) —
no shared memory, no texture/constant memory. This is a DELIBERATE simplification (CLAUDE.md "teaching
beats cleverness"): a production graph-connectivity library (cuGraph) does use warp-cooperative and
block-level techniques to reduce global-memory traffic for the find/union primitive, but adding that here
would obscure the algorithmic lesson (path halving, lock-free union) behind a memory-optimization lesson
that belongs in a different project (see "Where this sits in the real world").

## Numerical considerations

**Almost none — by design.** After the initial voxel-key packing (integer, bit-exact) and the neighbor
distance test (a float comparison against `d^2`, using the SAME formula on both GPU and CPU so any
disagreement is a genuine algorithmic bug, not float drift), EVERY subsequent stage of the CORE
clustering pipeline — edge construction dedup, union-find, label propagation, relabeling — is PURE
INTEGER arithmetic on point indices. There is no accumulation, no summation order, and hence no float
non-determinism anywhere in the partition computation itself; this is the same "all-integer after
binning" property project 36.03 (lattice-robot kinematics) notes for its own connectivity computation,
cited here for the identical reason. This is WHY `VERIFY(union_find)` and `VERIFY(label_propagation)`
can demand BIT-EXACT agreement rather than a tolerance — a rare and pleasant property for a GPU project
in this repository.

**Where float non-determinism DOES appear: per-cluster statistics.** `stats_accumulate_kernel`'s
`atomicAdd` calls for the centroid sum execute in a hardware-scheduler-decided order, exactly the same
"Method A" story 02.01's THEORY.md tells for its hash-table centroids — the SET of points per cluster is
exact (integer), but the SUM's rounding depends on add order. `VERIFY(stats)` therefore compares against
an independent CPU accumulation within a MEASURED-then-margined tolerance rather than bit-exactly (README
"Expected output" has the measured number, 1.717e-05 m, and the ~29x margin applied).

**Voxel-key edge cases:** the boundary of the half-open voxel convention `[k*L, (k+1)*L)` (02.01's
`voxel_coord` derivation, cited) is handled identically on GPU and CPU by sharing the same `floor()`
formula; no separate edge-case code exists to drift between the two.

## How we verify correctness

Two tiers, per CLAUDE.md's independence ruling (`src/reference_cpu.cpp`'s file header states this in
full): **shared data-layout formulas** (voxel key packing) are single-sourced and checked by
`VERIFY(keys)`; **algorithmic cores** (edge construction, union-find) are re-implemented INDEPENDENTLY on
the CPU with different data structures (`std::unordered_map` instead of sorted-array-and-binary-search;
plain sequential recursion instead of parallel lock-free sweeps) specifically because union-find is
"clever" enough that a shared implementation could hide an identical bug on both sides of a comparison.

The STRONGEST gate available here is `VERIFY(union_find)`: because union-by-min's final partition is
mathematically ORDER-INDEPENDENT (any correct sequence of unions over the same edge set converges to
identical canonical roots, "The math" above), a from-scratch SEQUENTIAL CPU implementation and a
massively-parallel GPU implementation are expected to agree BIT-FOR-BIT, not within a tolerance — and
they do, every run, on the committed scene. `GATE partition_vs_truth` goes one step further: it compares
against ground truth computed by a COMPLETELY SEPARATE program (`scripts/make_synthetic.py`, in Python,
before the C++ pipeline is even built) using the identical distance rule — three independent
implementations (GPU CUDA C++, CPU C++, and Python) converging on the same bits is about as strong a
correctness statement as a single project can make.

## Where this sits in the real world

**PCL's `EuclideanClusterExtraction`** does the same single-linkage computation on a single CPU core, via
a KD-tree radius search feeding a sequential union-find — algorithmically similar to this project's CPU
oracle, at production scale and tuning. **cuML/cuGraph's connected-components implementations** are the
GPU-production analog of this project's Method A, using more sophisticated (and less didactically
transparent) techniques — hybrid BFS/label-propagation hand-offs, warp-cooperative unions, load-balanced
work queues — to squeeze out throughput this project's teaching-scale kernels intentionally leave on the
table (see "The GPU mapping"'s note on memory hierarchy). **DBSCAN** is the standard "fix" for the
chaining failure mode this project demonstrates honestly: by requiring a MINIMUM NEIGHBOR COUNT (density)
rather than just a distance threshold, a single sparse bridging point (exactly what this project's
chaining-test bridge is) fails DBSCAN's core-point test and does not connect two dense regions — DBSCAN
is documented here, not implemented, as the natural "exercise 6" for a learner who wants to fix the
failure this project deliberately leaves visible. Modern production perception stacks (PointPillars,
CenterPoint, and their descendants) increasingly skip geometric clustering entirely in favor of a LEARNED
instance-segmentation head that reasons jointly over density, intensity, and (in a multi-frame system)
motion — named here honestly as the direction the field has moved, not as a claim that geometric
clustering is obsolete (it remains the cheapest, most interpretable, zero-training-data baseline, which
is exactly why it is still taught, and still shipped as a fallback in many real systems today).
