# 02.05 — KD-tree or LBVH construction + KNN/radius search on GPU: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**A "neighborhood" is a proxy for a physical surface.** A LiDAR point cloud is a discrete sample of
continuous surfaces — the floor, a wall, an obstacle. No single point carries surface information (a
plane needs at least 3 points to define; a reliable normal or curvature estimate needs many more, to
average out sensor noise). Every stage that reasons about *shape* — normal estimation (02.09), local
descriptors (02.10), ICP correspondence (02.06), clustering (02.04) — starts by asking "which other
points, near this one, plausibly sample the SAME surface patch?" That question is exactly what this
project answers, as fast and as correctly as possible, for every point in a scan.

**Why a FIXED radius is physically wrong, derived from beam geometry.** A spinning mechanical LiDAR
fires beams at fixed ANGULAR increments — `Δθ` in azimuth (`2π / AZIMUTH_STEPS`, e.g. ≈0.00245 rad for
this project's 2,560-step scan) and `Δφ` in elevation (fixed by the beam count, e.g. 2° ≈ 0.0349 rad
for a 16-beam unit). At range `R`, two ADJACENT beams' returns off a surface roughly perpendicular to
the beam land approximately `R·Δθ` apart in azimuth and `R·Δφ` apart in elevation — **point spacing
grows LINEARLY with range**, so **areal point density falls off as `1/R²`** (twice the range → a beam
footprint covering 4× the area → 1/4 the points per unit area, for the same angular resolution).
Concretely, at `R = 5 m` this project's scan geometry gives azimuth spacing ≈1.2 cm and elevation
spacing ≈17 cm; at `R = 20 m` those grow to ≈4.9 cm and ≈70 cm. A radius `r = 0.5 m` comfortably spans
several beams' worth of returns at 5 m, but at 20 m the ELEVATION spacing alone (70 cm) exceeds the
query DIAMETER (`2r = 1.0 m`) — the same fixed radius that worked nearby can return **zero** neighbors
far away, or (moving the other direction, toward a dense cluster of near-range returns or a
reflective/oversampled surface) can return **hundreds to thousands**. Neither failure is a bug in the
sensor; it is a direct consequence of fixed ANGULAR sampling meeting variable RANGE. This project's
committed data demonstrates the qualitative effect with two DESIGNED regions (a dense cluster and an
isolated sparse region) rather than simulating the full range-dependent beam-divergence physics inline
— see [`11.01`](../../11-sensor-sim-digital-twins/11.01-gpu-lidar-simulator/THEORY.md) for where that
physics IS derived and simulated in full; this project's contribution is the SEARCH STRUCTURE that
would let a real system query "how far do I need to look" adaptively instead of committing to one `r`.

**Engineering constraints a real robot imposes.** A perception stack must answer this question for
every point, every scan, at the sensor's frame rate (10–20 Hz, 50–100 ms/frame) — for a scan with
`N ~ 10⁵–10⁶` points and potentially just as many queries, a naive `O(N²)` all-pairs search (≥10¹⁰
operations) is unusable; the entire point of a spatial index is to turn that into
`O(N log N)` build + `O(log N)` per query.

## The math

**Morton (Z-order) encoding.** Given a point `p = (x,y,z)` inside a bounding box `[lo, hi]³`, quantize
each axis to a `b`-bit integer (`b = 10` here, `kMortonBitsPerAxis`):
`xᵢ = ⌊1023 · clamp((x−loₓ)/(hiₓ−loₓ), 0, 1)⌋`, similarly for `yᵢ, zᵢ`. The 30-bit Morton code
interleaves their bits: `code = Σₖ (xᵢ[k]·2^(3k) + yᵢ[k]·2^(3k+1) + zᵢ[k]·2^(3k+2))`. **Locality
property (stated didactically, proof sketch):** two points whose HIGH bits agree on all three axes
necessarily lie in the same large octree cell (their Morton codes share the same high-order bits by
construction); as codes get closer numerically, they are guaranteed to share progressively MORE
high-order bits, hence lie in progressively smaller shared octree cells. This is exactly the recursive
structure of an OCTREE traversed in a fixed child order — Morton order literally *is* octree
pre-order. The well-known caveat (this project measures it, does not just assert it — see
`[info] morton_locality`): two points can be geometrically close but numerically far apart in Morton
order when they straddle a high-order bit boundary (e.g. `x = 0.4999` vs `x = 0.5001` differ in the
TOP bit of that axis) — sorted-order adjacency is a strong statistical tendency toward spatial
locality, not a guarantee for every pair.

**Duplicate codes and the augmented key.** At 30-bit resolution over ~200k points, many points quantize
to the identical cell (`kernels.cuh`'s `augmented_key` comment; this project's data measures this
directly via the Morton-locality gap). Karras's construction needs a TOTAL ORDER with no ties;
appending each point's unique 32-bit original index to the low bits of a 64-bit key
(`augmented_key = (morton << 32) | index`) makes every key pairwise distinct, resolving the tie
deterministically by ascending original index and removing an entire class of degenerate cases from
the construction algorithm below (Karras's original paper handles this with a special-cased branch;
this project's index-augmentation makes that branch provably unreachable — kernels.cu's
`build_radix_tree_kernel` comment names exactly where).

**Karras's key lemma (construction, stated precisely; proof sketch).** Define `δ(i,j)` = the length of
the longest common PREFIX, in bits, of the augmented keys at SORTED positions `i` and `j` (`−1` if `j`
is out of range — Karras's "−∞" sentinel). The entire radix-tree construction is: for internal node
`i ∈ [0, n−2]`, its RANGE `[first, last]` is the unique maximal interval containing `i` such that every
position within shares a longer prefix with `i` than the immediate neighbor OUTSIDE the interval (on
the growth-direction side) does. **The lemma:** this range is EXACTLY the set of sorted positions in
the subtree rooted at the binary-radix-TRIE node that this internal node represents — i.e., the whole
construction is nothing but a **binary trie over the augmented keys**, and each internal node's
children split at the position where the shared prefix length strictly increases beyond the range's
own common prefix (`δ(first,last)`). Because a trie's levels correspond one-to-one with BIT POSITIONS
(examined from the most significant bit down), **and no bit position can gate more than one branch
along any single root-to-leaf path** (once two keys are separated at bit `b`, both children's subtrees
only ever examine bits AFTER `b`), **tree depth is bounded by the number of bits that CAN differ
between two keys.** The augmented key is 64 bits wide, but its top 2 bits are always 0 (the 30-bit
Morton code occupies bits 61..32; nothing occupies bits 63..62) — so only 62 bits can ever differ
between two distinct points, and **root-to-leaf depth is bounded by 62.** `kBvhStackSize = 64` is 2
nodes of pure headroom over this proven bound — not a "probably enough" guess.

**KNN complexity.** A single query visits, in the worst analyzed case, every node whose bounding box
lies within the current K-th-best distance — bounded above by the SAME `O(depth)` traversal cost as
radius search when the data is well-distributed (expected `O(log N + K)` for the classic uniform-random
analysis of k-d-tree-family KNN, a folklore result this project measures rather than re-derives; see
"How we verify correctness" and `[info] traversal_stats`). The heap itself: `O(log K)` per candidate
insertion/replacement, `K` fixed at 8 here.

## The algorithm

Four build stages, then two traversal algorithms, all detailed with full launch-configuration reasoning
in `kernels.cu` (read that file's per-kernel comments for the GPU specifics; this section is the
serial algorithmic skeleton):

1. **Morton encoding** — `O(N)`, embarrassingly parallel map.
2. **Sort** — `O(N log N)` serial / `O(N)` expected-work parallel radix sort (`thrust::sort`); because
   augmented keys are unique, ANY correct sort (stable or not) produces the identical result.
3. **Radix-tree construction** — for each of the `N−1` internal nodes, `O(log(range length))` via
   exponential+binary search for the range, then `O(log(range length))` again for the split position —
   `O(log N)` per node in the typical case, `O(N log N)` total SERIAL work, but **every node computed
   independently** so the PARALLEL depth is `O(log N)`, not `O(N log N)` (see "The GPU mapping").
4. **AABB propagation** — `O(N)` total work (each of the `2N−1` nodes' AABB computed once), parallel
   depth `O(tree depth)` — every leaf climbs toward the root, with exactly one thread "winning" the
   right to compute each internal node (the atomic second-arrival rule, kernels.cu).
5. **Radius search / KNN traversal** — `O(nodes visited)` per query, empirically close to
   `O(log N)` for radius search (well-separated queries) and higher for KNN or dense-region queries
   (measured, not assumed — `[info] traversal_stats`).

## The GPU mapping

**The independence argument — the whole point of this project's construction.** A top-down BVH build
(11.01's median-split approach, cited throughout) must finish splitting a node before its children CAN
be split — level `k+1` depends on level `k`'s result, an inherently SEQUENTIAL dependency chain across
`O(log N)` levels, however many threads are thrown at any ONE level. Karras's construction removes that
dependency entirely: internal node `i`'s range and children are a pure FUNCTION of the sorted key array
— nothing about node `i`'s computation reads any OTHER internal node's result. `build_radix_tree_kernel`
therefore launches ALL `N−1` internal-node computations in ONE kernel, ONE pass, with zero
synchronization between them — an `O(1)`-*synchronization-round* construction (as opposed to
`O(log N)` synchronization rounds for a top-down build), at the cost of `O(log(range))` redundant
`δ()` re-evaluation per node instead of information reuse across levels. This is the exact trade this
project's file-header comment names as "CONTRAST WITH 11.01": exact balance guarantee + sequential
levels vs. no balance guarantee + fully parallel construction.

**Warp divergence in traversal.** Every thread in a warp handles a DIFFERENT query, at a DIFFERENT
stack depth, testing a DIFFERENT node, taking a DIFFERENT branch (prune / descend-leaf / descend-both)
— structurally the same DIVERGENT TRAVERSAL problem 11.01's ray-BVH traversal has (that project's
THEORY.md measures and names it first in this repository; this project's `[info] traversal_stats` line
is the same measurement applied to KNN and radius search). The measured effect on the committed data:
mean nodes visited per radius-search query is small (queries in empty/sparse regions prune almost
everything), but the MAXIMUM (a query landing in the dense cluster) is over 5,000 — one thread in a
warp doing far more work than its 31 neighbors, who then idle. This is the SAME kind of load-imbalance
cost 02.01's `segmented_reduce_kernel` comment names for its own adversarial dense cluster, applied
here to tree traversal instead of segmented reduction.

**Stack/heap in registers vs. local memory — an honest accounting.** `kBvhStackSize = 64` ints
(256 bytes) and `kQueryK = 8` heap entries (2×32 bytes) are declared as local arrays inside each
kernel. The CUDA compiler places small, statically-indexed arrays in REGISTERS when it can prove every
access pattern is resolvable at compile time; a 64-entry stack with data-dependent push/pop indices is
a realistic candidate for **spilling to local memory** (a per-thread region of DEVICE memory, cached
through L1/L2 but not free) rather than staying fully register-resident — this project does not claim
otherwise. The honest teaching point: this is exactly the tradeoff a fixed-size stack makes against a
dynamically-sized one (`std::vector`-style, impossible in a `__global__` kernel without a slow
per-thread heap allocator) — bounded worst-case memory footprint, known at compile time, in exchange
for possibly-spilled (not always-register) storage. Profiling this precisely (Nsight Compute's
register/local-memory reports) is named as a follow-up in README "Exercises", not performed here.

## Numerical considerations

**Morton quantization resolution, worked at this project's scale.** The scene's padded bounding box
spans roughly 20 m per axis (a 16 m room plus a small margin); at `kMortonBitsPerAxis = 10`
(1,024 cells/axis), each cell is `20 / 1024 ≈ 2 cm` wide. Two points closer than 2 cm on EVERY axis can
share a Morton cell (measured directly by the collision rate implicit in how often the augmented key's
index tie-break actually resolves ties — not separately reported, but the reason the index augmentation
exists at all rather than being a defensive-only measure). This resolution only affects TREE
CONSTRUCTION locality (which points end up spatially near each other in the SORTED array, and hence how
efficient — not how CORRECT — traversal is); every distance test in this project uses the point's real
`float` coordinates, never its quantized code, so quantization can never produce a wrong query answer,
only a less-efficient tree.

**Race conditions and the `__threadfence()` this project shipped without, briefly.** During
development, `propagate_aabb_kernel`'s atomic "second-arrival" AABB fill initially used a bare
`atomicAdd(&flags[parent], 1u)` with no memory fence before it. This PASSED `VERIFY(topology)` (tree
SHAPE is independent of AABB content) but FAILED `VERIFY(aabb)` intermittently, and — more
interestingly — the resulting WRONG AABBs still passed `VERIFY(radius_bvh)` (the CPU twin traverses the
SAME, equally-wrong GPU tree, so the two agree with each other) while failing `GATE brute_force_anchor`
(the tree-free oracle, which has no way to inherit the bug). The root cause: `atomicAdd`'s ATOMICITY
guarantees the counter update itself is race-free, but says NOTHING about when this thread's PRIOR,
ORDINARY writes (the child's own AABB, written just before the atomic) become visible to OTHER threads
on OTHER SMs — the GPU's relaxed memory model permits those two writes to become visible to a remote
observer out of program order. `__threadfence()`, inserted immediately before the `atomicAdd`, forces
every earlier global write BY THIS THREAD to be visible to every other thread before the fence
returns — restoring the "by the time you see my flag increment, you can see my AABB write" guarantee
the whole second-arrival argument depends on. This is a textbook instance of the general CLAUDE.md
"race conditions, atomics, determinism" numerical-hazards category, made concrete: **atomics order
THEMSELVES; they do not, by default, order everything that happened before them, on other threads'
behalf.** See `kernels.cu`'s `propagate_aabb_kernel` for the fix and this exact story, in place.

**Float distance ties and the documented tie-break.** `knn_less(a, b)` (kernels.cuh) orders KNN
candidates by `(dist2, index)` lexicographically: closer distance wins; an EXACT `dist2` tie is broken
by the smaller original point index. Because every implementation (GPU kernel, CPU twin, brute-force
oracle) applies this SAME total order at every comparison — not just the final sort — the K-smallest
set under this order is a well-defined, ORDER-INDEPENDENT function of the point set (see kernels.cuh's
`KnnCandidate` comment for the full argument), which is what lets every `VERIFY`/`GATE` in this project
demand EXACT equality on KNN results rather than a tolerance.

**AABB exactness (extending 02.02's lineage).** Unlike a SUM (order-dependent under floating-point
rounding — 02.01's `segmented_reduce_kernel` documents this for centroids), `min`/`max` NEVER round:
IEEE-754 `min(a,b)`/`max(a,b)` simply SELECTS one of its two exact inputs. The min/max of a FIXED SET
of floats is therefore a function of the SET alone, independent of the order pairs are combined in —
so the GPU's atomic-flag AABB propagation and the CPU's single-threaded post-order traversal, given the
SAME (already-verified-identical) tree topology, are mathematically guaranteed to produce BIT-IDENTICAL
AABBs. `VERIFY(aabb)` demands exact equality on exactly this basis, not a measured-then-margined
tolerance.

**Query-time distance comparisons — an honest residual risk, measured not just assumed.** Radius/KNN
threshold comparisons (`dist2 <= r²`, heap replacement) use `float` arithmetic that COULD, in
principle, round differently between nvcc's device code (which may contract `a*a + b*b` into fused
multiply-adds) and cl.exe's host code (which does not, by default) — a difference of at most ~1 ULP.
This could, in a pathological case, flip an inclusion decision for a point sitting EXACTLY on a radius
or K-th-distance boundary. This project does not special-case that risk; it MEASURES it: every
`VERIFY(radius_bvh)`/`VERIFY(knn_bvh)`/`GATE brute_force_anchor` run on the full committed dataset (a
naturally-distributed synthetic scan, never deliberately placed at a boundary) reports EXACT agreement,
zero mismatches — the same honest "measured, not assumed" stance 02.01/08.01 take for their own
float-rounding tolerances, applied here to a case where the measured outcome happens to be zero
disagreement rather than a nonzero-but-bounded one.

## How we verify correctness

Three independent tiers, layered (CLAUDE.md §5's independence ruling, applied in full in
`reference_cpu.cpp`'s file header — read that first for the complete accounting of what is shared vs.
independently retyped):

1. **GPU-vs-CPU twins**, one per stage: `VERIFY(morton)`, `VERIFY(sort)`, `VERIFY(topology)`,
   `VERIFY(aabb)` — all EXACT (integer arithmetic or order-independent min/max; no float-rounding
   tolerance needed at any construction stage). `VERIFY(radius_bvh)` / `VERIFY(knn_bvh)` compare the
   GPU kernel against an independently-retyped CPU traversal over the SAME (already-verified) tree —
   this proves the TRAVERSAL logic is faithfully parallelized, but is BLIND to any bug that would
   corrupt both the tree and both traversals identically (the exact scenario the `__threadfence()` bug
   above demonstrates: this tier alone did not catch it).
2. **An independent voxel-hash oracle**, `VERIFY(radius_hash)` — a completely different data structure
   (`std::unordered_map`, no tree at all) reaching the same radius-search answer, cross-checked against
   the BVH path by `GATE hash_vs_bvh_agreement`.
3. **The brute-force anchor**, `GATE brute_force_anchor` — an `O(N·Q)` linear scan sharing NO code with
   any tree, hash, or traversal logic anywhere in the project, run against 1,000 sampled queries. This
   is the tier that actually caught the `__threadfence()` bug during development (tier 1 could not,
   because both the GPU kernel and its CPU twin traversed the identical, identically-wrong tree).

`GATE tree_validity` adds two FREE structural invariants that need no oracle at all: every leaf
reachable from the root exactly once (a corrupted or cyclic tree would fail this immediately), and
every internal node's AABB exactly contains both children's AABBs (a propagation bug — including the
one above, before the fix — fails this directly, independent of any CPU comparison).

## Where this sits in the real world

Production point-cloud libraries rarely hand-roll an LBVH for CPU-side neighbor search: **PCL's
`KdTreeFLANN`** wraps FLANN's k-d tree, tuned over two decades for exactly this workload on CPU.
GPU-scale neighbor search increasingly uses **cuML's** or **FAISS's** GPU-native indexes — at the
scale those libraries target (millions to billions of points), APPROXIMATE nearest-neighbor methods
(IVF, product quantization, learned indexes) trade a small, bounded error for large speedups this
project's EXACT construction does not attempt. NVIDIA's **OptiX** ray-tracing SDK builds and traverses
LBVH-family acceleration structures internally for ray-primitive intersection — the SAME algorithm
family this project implements from scratch, at production scale and with hardware ray-tracing-core
acceleration this project's software traversal does not have access to (11.01's README names OptiX as
the natural "build your own before touching it" next step for THAT project's raycasting; this project
is the natural next step for the TREE CONSTRUCTION half of the same lineage). **nvblox** (NVIDIA's GPU
TSDF-mapping library) and other real-time mapping systems typically favor voxel-hash-family structures
over trees for their OWN core data structure (spatial hashing scales better for uniformly-dense
volumetric data), which is exactly this project's own honest finding for uniform, small-radius queries
— the hash-vs-tree tradeoff this project measures is a live, real engineering decision in production
systems, not a pedagogical simplification. Where this project's teaching version would most need
extending for production use: incremental refit instead of full rebuild for streaming scans
(PRACTICE.md §1), approximate/early-terminating KNN for latency-critical paths (README Exercise 4), and
GPU-native construction libraries (e.g., NVIDIA's own internal LBVH builders, or CUB's device-wide
primitives) in place of this project's hand-rolled kernels — all deliberately NOT used here, per
CLAUDE.md's "build your own before touching the library" teaching mandate.
