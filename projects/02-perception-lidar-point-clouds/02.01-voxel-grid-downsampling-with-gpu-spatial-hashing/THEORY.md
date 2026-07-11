# 02.01 — Voxel-grid downsampling with GPU spatial hashing: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

### Why LiDAR density falls off as 1/r² (derive it from beam geometry)

A spinning mechanical LiDAR samples the world on a fixed **angular** grid, not a fixed **spatial** grid.
Project [`01.18`](../../01-perception-cameras-vision/01.18-depth-completion/THEORY.md) derives the
16-beam direction formula this project's scanner reuses verbatim:

```
d(el, az) = ( cos(el) cos(az),  cos(el) sin(az),  sin(el) )
```

with fixed elevation steps `Δel` (2° between the 16 beams here) and fixed azimuth steps `Δaz` (the
sensor's rotation resolution). Consider a beam that hits a flat surface roughly perpendicular to it, at
range `r`. Its two angular NEIGHBORS — the next azimuth step and the next beam's elevation step — hit
the SAME surface at points separated, to first order (small-angle approximation, valid because `Δel` and
`Δaz` are a few tenths of a degree to a couple of degrees), by

```
Δx_azimuth  ≈ r · Δaz          (arc length along the azimuth direction)
Δx_elevation ≈ r · Δel          (arc length along the elevation direction)
```

So the little patch of surface "represented" by one LiDAR return has area

```
A(r) ≈ (r · Δaz) · (r · Δel) = r² · Δaz · Δel
```

— it GROWS as `r²`. Point **density** (points per unit surface area) is the reciprocal,

```
ρ(r) ≈ 1 / A(r) = 1 / (r² · Δaz · Δel)  ∝  1/r²
```

A point 2 m from the sensor represents a patch 1/4 the area of one at 4 m — near-field surfaces are
sampled 4× as densely. (The full radiometric picture also includes a `cos(θ_incidence)` factor —
grazing rays cover MORE surface area per angular step, which project
[`11.01`](../../11-sensor-sim-digital-twins/11.01-gpu-lidar-simulator/THEORY.md) treats in depth for
*returned intensity*; for point-count density alone, the `1/r²` term dominates and is what this
project's committed sample actually exhibits — its dense near-field returns near the sensor's own
adversarial-cluster box are the extreme case of exactly this effect.) **This is the entire reason voxel
downsampling exists**: without it, every downstream algorithm (ICP correspondence search, clustering,
mapping integration) spends most of its O(N) or O(N²) work on near-field points that carry almost no
additional shape information beyond their neighbors, while comparatively starving far-field coverage.

### What a centroid representative preserves — and what it throws away

Replacing a voxel's points with their mean POSITION preserves the surface's coarse shape (the mean of
points scattered near a locally-flat patch sits very close to that patch — this project's
`downsample_quality` measurement, 0.057 m RMS at a 0.20 m leaf, quantifies exactly how close on the
committed sample) and is trivially fast to compute (one running sum, CLAUDE.md's "teaching beats
cleverness" choice over anything fancier). It DESTROYS everything about the points' internal
distribution: **surface normals** and **local curvature** (project
[`02.09`](../02.09-normal-curvature-estimation-at-millions/README.md)'s entire subject) cannot be
recovered from a centroid alone — a voxel containing a sharp edge and a voxel containing a flat patch of
the same area can produce statistically similar centroids. Point-to-plane ICP consumers that need
normals (project [`02.06`](../02.06-icp-point-to-point-point-to-plane-gicp/README.md)) must estimate
them on the DOWNSAMPLED points, not inherit them from this stage — a real interface constraint this
project's README "System context" names explicitly.

## The math

### Voxel coordinates: floor(), not truncation

Frame convention: SI meters, LiDAR sensor frame (origin at the sensor, +x forward, +z up — CLAUDE.md
§12), matching [`02.06`](../02.06-icp-point-to-point-point-to-plane-gicp/src/kernels.cuh)'s
`PointCloud` layout so this project's output is a drop-in ICP input. Given leaf size `L > 0`, point `p`'s
voxel coordinate along one axis is

```
v = floor(p / L)
```

C++'s `(int)(p/L)` **truncates toward zero**, which only agrees with `floor` for `p ≥ 0`. For `p < 0` it
is wrong: `p = -0.3 m`, `L = 0.20 m` ⇒ `p/L = -1.5` ⇒ `(int)(-1.5) = -1` (truncated), but
`floor(-1.5) = -2`. The voxel spanning `[-0.4, -0.2)` is the one that actually CONTAINS `x = -0.3`;
`[-0.2, 0.0)` (what truncation gives) does not. `kernels.cuh`'s `voxel_coord()` uses `std::floor` (device
side: `floorf`) unconditionally, for every sign of `p`. This project's scene spans both signs of every
axis (the room is centered on the sensor), so this is not a corner case — roughly half the points would
land in the wrong voxel without it.

### Packing (vx, vy, vz) into a 64-bit key

Each axis gets 21 bits: a **biased (offset-binary)** encoding, `u = v + 2²⁰`, maps the signed range
`[-2²⁰, 2²⁰-1]` onto the unsigned range `[0, 2²¹-1]`, then

```
key = ux | (uy << 21) | (uz << 42)        (63 bits used; bit 63 always 0)
```

**Overflow bound, worked out:** at this project's `L = 0.20 m`, the representable per-axis range is

```
±2²⁰ voxels × 0.20 m/voxel = ±1,048,576 × 0.20 = ±209,715.2 m ≈ ±210 km
```

— from an origin at the sensor. This project's scene is a 16 m room; the bound has roughly SEVEN orders
of magnitude of headroom. 21 bits was a deliberate, checked choice, not an arbitrary one — a robot
operating over city-scale distances from a single fixed origin would need to either re-origin
periodically (the standard practice) or grow the field width, and this arithmetic is exactly how you
would size that decision.

### The spatial hash function, and why THESE primes

Method A hashes `(vx, vy, vz)` (not the packed key directly) via the Teschner et al. 2003 formula:

```
h(vx, vy, vz) = (vx · p1) XOR (vy · p2) XOR (vz · p3)         mod 2³²  (unsigned wraparound)
p1 = 73,856,093   p2 = 19,349,663   p3 = 83,492,791   (all prime, the paper's published constants)
```

Multiplying by large, unrelated primes before XOR-ing spreads SPATIALLY NEARBY integer coordinates
(exactly what a LiDAR scan produces — two points 20 cm apart on the same surface differ by `±1` in at
most one or two axes) across the full 32-bit hash range, avoiding the visibly clustered, cache-line-like
patterns a naive `(vx*C + vy)*C + vz` hash would produce for a grid-structured input.

### Collision probability at this project's load factor — the birthday-bound arithmetic

For `m` items hashed uniformly at random into `C` slots, the probability that AT LEAST ONE pair collides
(the classic birthday-paradox estimate) is

```
P(≥1 collision) ≈ 1 - exp(-m² / (2C))
```

This project sizes `C` (capacity) from the WORST-CASE point count `N` — `C = next_pow2(⌈N / 0.5⌉)` —
because occupied-voxel count `m` is unknown ahead of time and can never exceed `N`. On the committed
sample: `N = 198,534` → `C = 524,288`; but the ACTUAL occupied count is only `m = 7,132` (points cluster
onto shared voxels far more than the worst case assumes). Plugging the REALIZED numbers in:

```
P(≥1 collision) ≈ 1 - exp(-7132² / (2·524288)) = 1 - exp(-48.6) ≈ 1.0
```

— a collision (two DIFFERENT voxel keys hashing to the same table slot) is essentially certain
somewhere in the table, and the measured data confirms it: `hash_stats`'s probe-length histogram shows
22 inserts needing 4 probes. But Knuth's classic average-case bound for linear probing,

```
E[probes, successful search]   ≈ ½ (1 + 1/(1-α))            α = m/C = load factor
E[probes, unsuccessful search] ≈ ½ (1 + 1/(1-α)²)
```

evaluated at this project's REALIZED load factor `α = m/C = 7132/524288 ≈ 0.0136` gives
`E[successful] ≈ 1.01`, `E[unsuccessful] ≈ 1.03` — both barely above 1 probe. The measured mean was
**0.072** on the normal scan region, even lower than either formula predicts, because most of this
project's `N` insert *attempts* are not "first claim of a brand-new key" (Knuth's unsuccessful-search
case) but "the 27th point landing in an already-claimed voxel" (a successful search for a key seeded
earlier at low load factor, which resolves fast) — see "Numerical considerations" and README
"Limitations" for the full, honest hash_stats story, including why the ADVERSARIAL region measured
*lower* probe lengths than the ordinary scan.

## The algorithm

**Shared first stage (both methods).** For each of `N` points, compute its voxel key: `O(N)` work,
trivially data-parallel (`compute_keys_kernel`, one thread per point).

**Method A — atomic open-addressing hash table.**
1. Allocate a table of `C` slots (keys + float sums + count), reset every slot to EMPTY.
2. For each point, hash its key to a home slot, then linear-probe: `atomicCAS` claim-or-match, then
   `atomicAdd` accumulate. Terminates in at most `C` probes (proven not to loop forever as long as
   `m < C`, which the capacity sizing above guarantees).
3. Compact: one thread per SLOT, claimed slots grab a dense output row via an atomic counter.

Serial cost: `O(N)` expected (amortized `O(1)` per insert at this load factor) plus `O(C)` for reset and
compaction (`C ≥ 2N` by construction, so this term also stays `O(N)`). Parallel cost: the SAME `O(N)`
(insert) / `O(C)` (reset+compact) work spread across thousands of concurrent GPU threads, each doing
`O(1)`-ish work (a short, usually-1-iteration probe loop).

**Method B — sort + fixed-order segmented reduction.**
1. Stable-sort the `(key, point_index)` pairs by key: turns "same voxel" into "contiguous run in a
   sorted array". `O(N log N)` serial cost (a comparison sort's bound; the GPU radix sort Thrust actually
   runs is `O(N · w/b)` for a `w`-bit key processed `b` bits per pass — see "The GPU mapping").
2. Mark segment boundaries (`O(N)`, one comparison per position) and compact them into a `seg_start`
   array via stream compaction (`O(N)`).
3. One thread per VOXEL (not per point) walks its own segment sequentially, summing floats in a FIXED
   order. `O(N)` total work (every point visited exactly once, across however many voxel-threads there
   are), but load-imbalanced across threads (see "Where this sits in the real world").

Both methods do asymptotically the same `Θ(N)` amount of real work; Method B pays an extra `log`-ish
factor for the sort in exchange for the determinism story below.

## The GPU mapping

### Method A: the SCATTER pattern, and the atomicCAS claim-or-probe loop

Each thread starts knowing only ITS OWN point; it does not know in advance which other threads share its
voxel. The natural mapping is therefore **scatter**: every thread independently computes a destination
(hash slot) and writes there, racing against every other thread that might compute the SAME destination.
`atomicCAS(address, compare, val)` is the hardware primitive that makes this race-safe: it atomically
reads `*address`, compares to `compare`, and writes `val` ONLY if they matched, indivisibly — no other
thread's `atomicCAS` on the same address can interleave inside it. The claim-or-probe loop
(`kernels.cu`'s `hash_insert_kernel`, walked through step by step in its own comment) is the canonical
form of this pattern on a GPU: try to claim the home slot; if another thread already claimed it for the
SAME key, join it; if a DIFFERENT key is there, linear-probe onward. `atomicAdd` on the per-slot sums is
the same idea applied to accumulation instead of slot ownership. **Memory:** the table lives in global
memory (no shared-memory staging — occupancy per voxel is unknown ahead of a launch, and different
warps' points scatter to essentially random slots, so there is no locality to stage); reads/writes are
one 8-byte (key) or three 4-byte (sums) global transactions per thread, uncoalesced by construction
(scatter has no useful spatial pattern to coalesce).

### Method B: the ORDER (sort-then-reduce) pattern

The GPU mapping here is the opposite philosophy: instead of racing to a shared destination, first
establish a canonical ORDER (the sort) that turns "who shares my group" into "who is next to me in the
array" — then every group's reduction becomes an ordinary, race-free, per-thread sequential walk.
`thrust::stable_sort_by_key` runs a **radix sort** on the 64-bit keys under the hood: repeated stable
partitioning by a few bits of the key at a time (least-significant-first), each pass fully data-parallel
(a histogram + prefix-sum + scatter, conceptually the SAME counting-sort-plus-prefix-sum idea project
[`22.01`](../../22-multi-robot-swarms/22.01-100k-agent-swarm-simulator/src/kernels.cuh)'s
spatial-binning `counts[]`/`starts[]`/`bin_agents[]` layout uses at a smaller scale for neighbor binning
— Thrust's radix sort is that same idea, generalized to a 64-bit key and heavily tuned). We use the
LIBRARY here specifically for its documented **stability** guarantee (equal keys keep their input order)
— hand-rolling a stable multi-pass radix sort correctly is real, fiddly work that would teach little
beyond what `22.01`'s simpler single-pass counting sort already covers (CLAUDE.md rule 6: use the
library, explain what it computes and why not hand-rolled). `thrust::reduce` (segment count) and
`thrust::copy_if` (boundary-position compaction) are both standard **stream compaction** building
blocks, explained at their call sites in `kernels.cu`. The final `segmented_reduce_kernel` is a plain map
over voxels — global memory reads following the sorted permutation (a gather, not a scatter: each
voxel-thread reads its OWN contiguous run of point indices, no atomics anywhere).

### The scatter-vs-order duality — the same story elsewhere in this repo

This project's Method A (scatter + atomics, nondeterministic order, simple to write) vs. Method B (sort
into a fixed order first, then race-free reduction) is the SAME fundamental choice this repository's FEM
assembly projects face in their own domain: project
[`28.01`](../../28-soft-robotics/28.01-real-time-fem-soft-arm-model-model-based-control/THEORY.md)'s
"Scatter + atomics" row in its element-assembly comparison table names the identical trade-off —
`atomicAdd` into shared nodes, non-deterministic summation order, simplest to implement — against a
GATHER alternative (each output reads from its known contributors, race-free by construction, the
strategy project [`26.01`](../../26-mechanical-design-structures/26.01-topology-optimization-on-gpu-for-lightweight/THEORY.md)
chooses for the same mesh-assembly problem, precisely BECAUSE its every-node-knows-its-neighbors mesh
structure makes gather cheap). Method B is not literally a gather (it still needs a sort to discover
group membership first, since a point cloud carries no built-in neighbor structure the way a mesh does)
— but the PAYOFF is the same one 26.01 chose gather for: a fixed, known access pattern buys
determinism that atomics structurally cannot.

## Numerical considerations

**Precision.** Everything here is FP32 except reference_cpu.cpp's Method-A oracle (see below) — this
project never needs FP64 accuracy for a 20 cm voxel on a scene spanning tens of meters (float32's ~7
decimal digits of precision is >> the physical measurement noise the synthetic scan itself injects,
±1.5 cm range noise per `scripts/make_synthetic.py`).

**Method A's float-accumulation nondeterminism — measured, not just asserted.** `atomicAdd` serializes
concurrent writes to the same address in an order the GPU's hardware scheduler decides — a property of
the SILICON, not the program, and it is not guaranteed to repeat between runs (different warp
scheduling, different launch-time GPU state). Floating-point addition is **not associative**
(`(a+b)+c ≠ a+(b+c)` in general, due to rounding), so summing the same set of floats in different orders
CAN produce different final bit patterns. This project MEASURES the effect rather than hand-waving it:
`main.cu` runs the entire Method-A pipeline three independent times and compares matching voxels' float
centroids across runs. On the reference machine's actual run: **max delta across 3 runs ≈ 4–8×10⁻⁶ m** —
present, small (float32 ULP at these ~1–20 m coordinate magnitudes is already ~10⁻⁶–10⁻⁷ m, so this is a
handful of ULPs, exactly what accumulating ~2–30 floats in a different order would produce), and reported
honestly in the `[info] determinism_method_a` line rather than hidden or asserted away.

**Method B's fixed-order design is what buys bit-exactness — a parallel worth stating explicitly.**
Projects [`01.13`](../../01-perception-cameras-vision/01.13-canny-hough-line-circle-detection-for-industrial/README.md)
and `01.14` earn their determinism by staying in INTEGER arithmetic throughout (integers add
associatively — order never matters). Method B's underlying quantity (a centroid) is inherently a real
number, so that trick is unavailable; instead, Method B fixes the SUMMATION ORDER itself (stable-sort by
key, then walk each voxel's run sequentially in ascending original-point-index order) and relies on
IEEE-754 float addition being FULLY DETERMINED by a fixed operand sequence — `a + b`, executed with
round-to-nearest-even and no fused-multiply-add reordering (there is no multiply in this reduction, and
this repo never enables `--use_fast_math`, CLAUDE.md §5), gives the SAME bit pattern on any IEEE-754-
compliant adder, CPU or GPU. `reference_cpu.cpp`'s `sort_based_downsample_cpu` reproduces the identical
order (`std::stable_sort`'s stability guarantee mirrors `thrust::stable_sort_by_key`'s exactly), which is
why `VERIFY(method_b)` compares bit-exact rather than within a tolerance — and it is why the demo's own
3-repeated-run `determinism_method_b` gate also compares bit-exact, on the SAME machine, run to run:
Method B's radix sort is a deterministic algorithm (its result depends only on the input, never on
thread-scheduling accidents), so "fixed order" holds run-to-run on one GPU, not just against the CPU.

**Why Method A's independent CPU oracle uses DOUBLE, not float.** `reference_cpu.cpp`'s
`hashmap_downsample_cpu` accumulates in `double` deliberately — the same "give the oracle more precision
than the thing under test" choice project
[`02.06`](../02.06-icp-point-to-point-point-to-plane-gicp/src/reference_cpu.cpp)'s
`build_normal_system_cpu` makes. A double-precision, sequential-order, `std::unordered_map`-structured
sum is close enough to the "true" mathematical answer that Method A's float/atomic/hash-table result can
be compared against it as ground truth, with the tolerance (measured then margined — `main.cu`'s
`kToleranceMethodA_m`, 1.0×10⁻⁴ m against a measured worst case of 3.34×10⁻⁶ m) absorbing float32's own
rounding budget, not the oracle's.

## How we verify correctness

Three tiers, matching `reference_cpu.cpp`'s independence ruling (read that file's header for the full
reasoning behind which twin plays which role):

1. **`VERIFY(keys)` — bit-exact, shared-formula check.** GPU (kernels.cu's device-side transcription) vs
   CPU (kernels.cuh's shared `voxel_coord`/`pack_voxel_key`) key for all 198,534 points, compared
   exactly (integers — no tolerance needed or wanted). This is the gate that would catch a typo between
   the header's canonical formula and kernels.cu's necessarily-duplicated device copy (kernels.cuh's file
   header explains why that duplication exists).
2. **`VERIFY(method_b)` — bit-exact, deliberately-matched-order twin.** Every one of Method B's 7,132
   output rows (centroid, count, key) compared with `==`, not a tolerance — see "Numerical
   considerations" above for why that is achievable and honest, not a trick.
3. **`VERIFY(method_a)` — tolerance-based, genuinely independent twin.** Different data structure
   (`std::unordered_map` vs. this project's open-addressing table), different order (sequential point
   index vs. GPU thread scheduling), different precision (double vs. float) — see "Numerical
   considerations". Occupancy (voxel SET and per-voxel COUNTS) is checked EXACTLY (pure integer
   bookkeeping, order-independent); centroid VALUES are checked within the measured-then-margined
   tolerance.

Two further gates are **independent of both twins** — computed from a method's OWN output, checking a
property that would be violated by an internal pipeline bug even if it happened to still agree with a
CPU reference (the reference_cpu.cpp independence ruling requires exactly this kind of check whenever any
code is shared token-for-token, which the key-computation formula is):
`partition_invariant` (every point counted exactly once — Σcounts == N, both methods) and
`centroid_containment` (every centroid geometrically inside its own voxel's AABB — a free consequence of
convexity that a real accumulation bug, e.g. summing the wrong point into a voxel, could still violate
even if it did not change the point COUNT).

Edge cases exercised by the committed sample by construction: negative coordinates on every axis (the
floor-vs-truncate pitfall, "The math" above); a voxel receiving ~3,000 points (the dense adversarial
cluster — stresses Method A's accumulation precision, which is exactly why it, not the ordinary scan,
sizes `kToleranceMethodA_m`); voxels receiving exactly 1 point (the sparse adversarial region — the
degenerate case of every gate and every statistic above); and the ordinary scan's broad range of
per-voxel counts (1 to a few dozen) in between.

## Where this sits in the real world

**PCL's `VoxelGrid`** is the textbook reference this project's centroid-per-occupied-voxel definition
matches exactly; it runs single-threaded on CPU with a `std::unordered_map`-like accumulation (closer in
spirit to this project's Method-A CPU *oracle* than to either GPU method) and adds leaf-size
auto-suggestion and a covariance-tracking variant (`VoxelGridCovariance`) for NDT consumers this project
does not attempt. **Open3D's `voxel_down_sample`** is a modern equivalent with a documented
`_and_trace` variant that returns the point→voxel mapping explicitly — this project computes that exact
mapping internally (every gate needs it) but never exposes it as a first-class output the way Open3D
does; that would be a natural, small extension (see Exercise 5). **cuPCL / nvblox-class GPU
preprocessing** fuses voxel filtering into a larger pipeline (often directly feeding an SDF/TSDF
integration stage) as ONE kernel among several, sharing device memory and avoiding the host round-trips
this teaching project deliberately keeps explicit and visible (every stage here is its own launch with
its own comment, on purpose — CLAUDE.md "teaching beats cleverness"). **`cub::DeviceSegmentedReduce`**
(and Thrust's own `reduce_by_key`) is the production-grade version of this project's hand-written,
one-thread-per-voxel `segmented_reduce_kernel`: it load-balances a single wide reduction across MANY
threads per segment using a parallel tree, which matters exactly when segment sizes are highly skewed —
this project's own 3,000-point dense cluster is precisely that skew, and its one thread does far more
work than its neighbors in Method B's current design. The reason this project does not reach for that
library call: a parallel tree reduction's summation order is an implementation detail of the library
(documented as unspecified, and free to change between CUDA versions), which would make the bit-exact
CPU comparison this project relies on for Method B impossible to promise — the one-thread-per-voxel
design trades peak throughput for a determinism guarantee this teaching project is built to demonstrate.
A production pipeline that does not need bit-exact reproducibility (most do not) should prefer the
library call.
