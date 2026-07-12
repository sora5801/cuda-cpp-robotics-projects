# 02.15 — Point cloud compression (octree/entropy) for fleet uplink: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

A fleet of robots builds maps. Every robot that explores accumulates a point cloud (or a fused
occupancy grid, or a mesh — but underneath almost all of them, a set of 3-D points) describing
what it has seen, and a real fleet operation needs those observations back at a central place: a
cloud service that merges maps across robots, a human supervisor's dashboard, a nightly
map-quality audit, another robot that will later navigate the same building. That trip — robot to
cloud — is the **uplink**, and it is not free. It rides on the SAME radio link (Wi-Fi indoors,
cellular outdoors) the robot also uses for teleoperation video, fleet telemetry, and safety
heartbeats, and that link has a real, physical, often embarrassingly small budget: indoor
warehouse Wi-Fi shared across dozens of robots, or a cellular data plan billed by the gigabyte
(PRACTICE.md §1 works this arithmetic in dollars). A raw point cloud is enormous: a single
200,000-point tile — the scale this project's demo actually uses — stored as three IEEE-754
float32 values per point is `200,000 × 3 × 4 = 2,400,000` bytes, **2.3 MiB**, for one small tile
of one building. A fleet of 50 robots each uploading a handful of tiles a day adds up to real
gigabytes, real minutes, real dollars — quantified honestly, with this project's own measured
numbers, in the `[info] fleet arithmetic` line the demo prints and in PRACTICE.md §1.

**Why can this be compressed at all?** Because of a physical fact about the world robots map:
*most of a building's volume is empty air.* A robot's LiDAR, camera-stereo rig, or accumulated SLAM
map only ever reports points where something solid actually is — floors, walls, furniture,
shelving — and every solid object's *boundary* (the only part a sensor can see) is a **2-D
surface embedded in 3-D space**. A flat floor is a 2-D plane; a box is six 2-D faces; even a
crumpled, complicated surface is still locally 2-dimensional. This is the entire physical basis
for point-cloud compression, and it is the exact fact this project's two committed clouds are
built to demonstrate side by side:

- **`structured_map.bin`** — 200,000 points sampled on 30 flat surfaces (a floor, four walls, five
  boxes' visible faces) inside a 20 m × 20 m room. Real surfaces, real 2-D structure.
- **`pathological_cube.bin`** — the SAME 200,000 points, but scattered *uniformly at random* inside
  a matching cube — geometry with **no** surface structure: a 3-D-volume-filling cloud, the
  physically implausible case no real sensor ever produces, kept as the designed negative control.

Any spatial data structure that adapts to WHERE the data is (rather than storing every point in a
dense fixed-resolution grid) should need dramatically fewer "cells" to describe the surface cloud
than the volume-filling one — and that gap, quantified precisely, is this project's payoff (see
"How we verify correctness" below for the exact measured numbers).

**The engineering constraint this project's lossy stage answers to:** a robot's map does not need
infinite precision. A warehouse AMR navigating a 1 cm-tolerance aisle does not need sub-millimeter
wall positions; a human reviewing a dashboard map does not need better resolution than their
screen. Every real fleet-map format therefore accepts a bounded, **quantified** position error in
exchange for a large size reduction — this project's octree depth `D` is exactly that dial, and
the rate-distortion sweep (`demo/out/rd_curve.csv`) is the engineering trade made visible: coarser
`D` → far fewer bits, larger (but still *bounded and provable*) position error.

## The math

### Notation

- A point cloud is `N` points `p_i = (x_i, y_i, z_i) ∈ ℝ³`, meters, right-handed map frame (+z
  up — SYSTEM_DESIGN.md §3, applied to a static map tile rather than a live sensor scan).
- The scene's bounding **cube**: `aabb.min ∈ ℝ³`, side length `L = aabb.extent_m` (meters). Padded
  to a CUBE (equal extent on every axis, `kernels.cuh`'s `SceneAABB`) — not a general box — so
  every leaf cell at every depth is a literal cube, which is what makes the distortion bound below
  a one-line formula instead of a three-axis mess.
- **Depth** `D` (integer, this project sweeps `D ∈ {8, 9, 10, 11}`): the octree is subdivided `D`
  levels deep. Leaf cell size (the finest grid resolution): `leaf_m = L / 2^D`.
- A point's **quantized cell coordinate**: `i_a = clamp(⌊(p_a − aabb.min_a) / leaf_m⌋, 0, 2^D−1)`
  for each axis `a ∈ {x,y,z}` — `kernels.cuh`'s `quantize_axis`.

### The octree as a prefix tree over Morton strings

**Morton encoding (Z-order).** Interleave the bits of `(i_x, i_y, i_z)` — each a `D`-bit unsigned
integer — into one `3D`-bit integer `code`, bit `3b+0/1/2` of `code` holding bit `b` of
`i_x/i_y/i_z` respectively (`kernels.cuh`'s `morton_encode`; the same axis-to-bit convention
02.05's `morton_encode30` uses for its LBVH, generalized here to a depth swept across `8..11`
instead of fixed at 10). This is a bijection between the `2^{3D}` grid cells and the integers
`[0, 2^{3D})`.

**Claim: sorting codes ascending recursively orders the octree.** Define, for a code and a level
`ℓ ∈ [0, D]`, its **level-`ℓ` prefix**: the top `3ℓ` bits, `node_prefix(code, D, ℓ) = code ≫ 3(D−ℓ)`.
Because bit-shift is order-preserving (`a ≤ b ⟹ a≫k ≤ b≫k` for non-negative integers), if an array
of codes is sorted ascending by the FULL code, then for *every fixed* `ℓ`, the sequence
`node_prefix(code_i, D, ℓ)` is *also* non-decreasing across the sorted array. Consequently: **two
points share a level-`ℓ` octree node if and only if their level-`ℓ` prefixes are equal**, and
because the sorted-array sequence of those prefixes is non-decreasing, every group of equal
prefixes is a single *contiguous run*. Finding an octree level's nodes therefore reduces to a
STRING problem — "find where consecutive sorted codes' shared prefix length changes" — solvable
with a boundary-detection PREDICATE, not a pointer-chasing tree walk. This is the load-bearing
insight the whole GPU construction (below) exploits.

**The occupancy byte.** Each internal node at level `ℓ` (covering points sharing one level-`ℓ`
prefix) has up to 8 children at level `ℓ+1`, indexed by the next 3 bits of the code
(`child_octant(code, D, ℓ) = (code ≫ 3(D−ℓ−1)) & 7`, bit 0 = x, bit 1 = y, bit 2 = z). One byte per
node records which children exist: `occupancy = OR over points in this node of (1 ≪ child_octant)`.
The concatenation of every node's occupancy byte, level by level, root first, IS the compressed
geometry code — this project's Stage 1 output. Total geometry bytes `M` = total internal node
count across all `D` levels.

**Nesting property (used by `GATE rate_monotonic`).** Because `node_prefix(code, D, ℓ)` for `ℓ <
D` does not depend on `D` at all beyond requiring `D ≥ ℓ`, the level-0..D′ structure of the
depth-`D` octree, for any `D′ < D`, is *identical* to the depth-`D′` octree of the same cloud. So
as `D` grows, the octree can only ever **add** finer structure, never revise coarser structure:
total node count `M(D)` is non-decreasing in `D`, and — since finer cells are strictly smaller —
maximum reconstruction error is non-increasing in `D`. This is a provable structural fact, not an
empirical trend, and `main.cu` gates on it directly.

### The quantization distortion bound (used by `GATE distortion_bound`)

Decoding a leaf reconstructs its CENTER: `center_a = aabb.min_a + (i_a + 0.5) · leaf_m`. A point
`p` that quantized into that leaf satisfies `|p_a − center_a| ≤ leaf_m/2` on every axis (by
construction of `quantize_axis`'s floor). The worst case — `p` sitting at a cell **corner** — gives
the reconstruction error its tight upper bound, the leaf cube's **half-diagonal**:

```
error(p) ≤ (leaf_m · √3) / 2
```

This is the `distortion_bound_m` column of `rd_curve.csv`, derived from `leaf_m = L/2^D` alone —
no measurement needed to know it in advance, and the demo GATES the measured maximum error against
it (with a tiny float-rounding slack, not a correctness fudge). In this project's own run, the
measured worst-case ratio (max error / bound) across all 8 rows was **0.9962** — a point sitting
almost exactly on a cell corner, confirming the bound is *tight*, not merely safe.

### Shannon entropy and Huffman optimality (used by `GATE entropy_bound`)

Model the occupancy-byte stream as `M` i.i.d. draws from a 256-symbol alphabet with measured
probabilities `p_s = count_s / M`. The **Shannon entropy** `H = − Σ p_s log₂ p_s` (bits/symbol) is
the information-theoretic lower bound on the average code length ANY uniquely-decodable code can
achieve. **Huffman's theorem** (Huffman 1952): the canonical Huffman construction (repeatedly merge
the two least-frequent symbols) produces an *optimal* prefix-free code, and its average length
`L̄` is provably bounded:

```
H ≤ L̄ < H + 1
```

(the "within one bit of entropy" guarantee — the +1 slack is the unavoidable cost of assigning
each symbol a *whole number* of bits; arithmetic/range coding, discussed under "Where this sits in
the real world", removes that slack at the cost of losing the trivial byte-at-a-time decodability
this project relies on). `GATE entropy_bound` checks this band directly against the MEASURED `H`
and `L̄` for every row of the sweep.

### Canonical Huffman — why it is deterministic

A Huffman TREE is not unique: ties in frequency during construction can be broken arbitrarily
without changing the OPTIMAL average length. This project pins that ambiguity down with one
documented total order — `huffman_merge_key(freq, id)` (`kernels.cuh`) — so two structurally
DIFFERENT construction algorithms (a heap in `kernels.cu`'s `build_huffman_table`, a linear scan in
`reference_cpu.cpp`'s `build_huffman_table_cpu`) provably converge on the identical set of code
LENGTHS. Given only that length multiset, the **canonical** bit-pattern assignment (sort symbols by
`(length, symbol)`, assign codes in that order, `code_{k} = (code_{k−1}+1) ≪ (len_k − len_{k−1})`)
is then a pure FORMULA with no remaining ambiguity at all — the property `VERIFY huffman_table`
checks bit-for-bit.

## The algorithm

**Stage 1 — geometry (the octree), level by level, `ℓ = 0..D−1`:**

1. Compute each point's depth-`D` Morton code (a MAP: `O(n)` work, fully parallel).
2. Sort the codes ascending (`O(n log n)` comparisons, but see "The GPU mapping" for the radix-sort
   reality).
3. For each level `ℓ`: mark node BOUNDARIES in the sorted array (a MAP comparing neighbors),
   exclusive-SCAN the boundary flags into per-point node labels, then SCATTER each point's child-
   octant bit into its node's occupancy word (an atomic reduce).

Total work: `O(n·D)` across all levels (a small constant multiple of one pass over the points) —
serial depth is `O(D)` sequential level-launches, each internally `O(log n)` deep (the scan) —
compare a naive serial pointer-based octree build, which is `O(n·D)` work too but with `Θ(n·D)`
SERIAL depth (one point insertion at a time, following pointers).

**Stage 2 — entropy coding, once per depth:**

1. Histogram the `M`-byte occupancy stream over 256 bins (`O(M)`, parallel atomics).
2. Build the canonical Huffman table from the histogram (`O(k log k)` for `k ≤ 256` observed
   symbols — small and effectively serial; no GPU pattern here, see "The GPU mapping").
3. Look up each symbol's code LENGTH (a MAP), exclusive-SCAN lengths into per-symbol BIT offsets,
   SCATTER each symbol's variable-length code into the packed output (`O(M)` work, `O(log M)`
   scan depth).

**Decode** (host-only; see "The GPU mapping" for why): walk a bit-trie built from the table,
`O(total encoded bits)` serial work, THEN expand the octree level by level from the decoded byte
stream (`O(M)`), producing every leaf's reconstructed center.

## The GPU mapping

**Stage 1's central composition — the SAME scan primitive answering two different questions.**
This repo's stream-compaction scan (`02.02`'s two-level Blelloch exclusive scan, copied and cited
verbatim in `kernels.cu`) is normally used to turn 0/1 "keep this element" flags into destination
SLOTS for a compaction scatter. This project repurposes the *identical* kernel for a different
question — "which node does this point belong to?" — by scanning `is_start` (0/1 "does a new node
begin here") flags into per-point node LABELS. The one-line correction needed to make that
repurposing correct — `node_id[i] = scan_exclusive[i] + is_start[i] − 1` (a continuation point is
OFF BY ONE relative to the node it actually belongs to if you use the raw exclusive-scan value
directly) — is a real bug this project's own `VERIFY octree_levels` gate caught during
development (see `kernels.cu`'s `compute_occupancy_kernel` comment for the full derivation) — an
honest record that repurposing a primitive for a new question is easy to get subtly wrong, and
exactly why independent CPU twins exist.

**Stage 1's atomic scatter.** `compute_occupancy_kernel`: one thread per point, `atomicOr` its
child-octant bit into its node's occupancy WORD. Why a full `uint32_t` per node rather than one
byte: CUDA's `atomicOr` has no native byte-granular overload — only 32-/64-bit words. The
production fix is a manual "read the containing aligned word, shift the mask into the target
byte's position, atomicOr the word" trick; this project takes the simpler, equally-correct
teaching choice of one whole word per node (4× the memory of a byte array, utterly negligible at
this project's node counts — even the pathological cohort's worst-case ~966,000 nodes at `D=11`
is under 4 MiB) and narrows to a packed byte stream on the HOST afterward, once no more atomics
are needed.

**Stage 1's sort.** `thrust::sort` on the 64-bit codes dispatches to `cub::DeviceRadixSort` — an
8-pass, 8-bits-per-pass LSD radix sort, `O(n)` work, memory-bandwidth bound, no comparison
function needed (unlike the `O(n log n)` comparison sort the complexity analysis above cites as
the naive baseline). No paired "value" array is uploaded: this codec only ever needs the code
MULTISET, never which original point produced which code, so an *unstable* sort over
possibly-equal keys is correct and sufficient — a deliberate simplification versus 02.05's LBVH,
whose augmented 64-bit key exists specifically because IT needs to recover per-point identity
after sorting.

**Stage 1's SIZE BOUND, and why Stage 2's scan uses a different primitive.** The per-level
boundary scan always operates on an array sized `n` (the FIXED point count, 200,000 in this
project's committed clouds) — safely under the hand-rolled two-level Blelloch scan's
`512×512=262,144`-element bound. Stage 2's code-LENGTH scan, by contrast, operates on an array
sized `M` (the total octree NODE count), which is DATA-DEPENDENT and, on the pathological cohort
at `D=11`, reaches **965,984** — comfortably past that bound. `kernels.cu` therefore uses
`thrust::exclusive_scan` (CUB's single-pass "decoupled look-back" chained scan, no size limit) for
Stage 2 specifically, and the hand-rolled, bound-CHECKED Blelloch scan (which fails loudly rather
than silently truncating if the bound is ever exceeded) for Stage 1 — the repo's `teach the
internals once, reach for the production primitive where its assumptions would otherwise break`
discipline, both primitives used for a real, documented reason rather than habit.

**Why the canonical Huffman table build is HOST-ONLY.** At most 256 observed symbols: a heap-based
merge sequence is `O(k log k) ≤ O(256 · 8)` — a few thousand operations, dwarfed by the surrounding
`O(n)`/`O(M)` GPU kernels. There is no meaningful thread-per-something GPU mapping for a problem
this small and this inherently sequential (each merge step depends on the PREVIOUS merge's
result) — a correct teaching choice to leave it on the host, not an oversight (CLAUDE.md §1: no
black boxes, but also no manufactured parallelism where none pays off).

**Why decode is host-only, and resists parallelism entirely.** ENCODE parallelizes cleanly because
every symbol's output BIT OFFSET is knowable in advance from a scan over LENGTHS alone — no symbol
needs any OTHER symbol's DECODED VALUE first. DECODE has the opposite shape: a variable-length
prefix code means decoding symbol `i` requires knowing exactly how many bits symbol `i−1`
consumed, which requires having ALREADY decoded symbol `i−1` — a strictly serial dependency chain
with no scan-shaped escape hatch. The production answer real decoders use (documented here, not
implemented, per this project's ratified scope) is **block-wise decode**: periodically insert
byte-aligned restart points into the stream (sacrificing a small amount of rate — each restart
point forfeits mid-byte packing efficiency) so `K` independent decoders can each start from a known
bit offset and run in parallel, trading a documented rate cost for `K`-way decode parallelism.

## Numerical considerations

**All-integer after quantization — say it plainly.** Once a point is quantized to `(i_x, i_y,
i_z)`, every downstream Stage-1 operation (Morton encode, sort, prefix comparison, child-octant
extraction) is **exact integer arithmetic** — no floating-point rounding, no race-order
sensitivity, no ULP hand-wringing. This is the domain convention this repo's other Morton/hash
projects (02.01, 02.05) also rely on, and it is WHY `VERIFY morton`, `VERIFY sort`, `VERIFY
octree_levels`, `VERIFY histogram`, and `VERIFY encode_bitstream` are all gated **bit-exact**, not
tolerance-bounded — there is no legitimate source of GPU/CPU disagreement left once the codes
themselves agree.

**The one place floating point still matters: `quantize_axis`'s normalize-then-scale.** Computing
`⌊(p_a − lo) / L · 2^D⌋` involves one division and one multiply in FP32; a point sitting almost
exactly on a cell boundary could, in principle, round to the cell on either side depending on
compiler/architecture FMA fusion differences (the same class of hazard 02.02's edge-cohort
generator documents for its own predicate thresholds). This project's synthetic data does not
place points deliberately AT such boundaries (unlike 02.02's adversarial edge cohort — a
deliberate scoping difference: this project's GPU-vs-CPU gates are already bit-exact on the
INTEGER codes because `compute_codes_cpu` and the GPU kernel both compute `point_to_code` from the
SAME shared formula, so the concern is moot for `VERIFY morton` specifically — see "How we verify
correctness"), and the `compute_cube_aabb` padding (0.5% margin) keeps real data comfortably away
from the `[0, 2^D)` extremes either way.

**Canonical Huffman code length, worst case.** For 256 symbols, the theoretical worst-case maximum
Huffman code length (a deliberately adversarial Fibonacci-like frequency distribution) is 255
bits — this project stores codes in a `uint32_t` (32-bit budget), which would silently truncate
such a pathological table. This never occurs with REAL measured histograms at this project's
scale (hundreds of thousands of samples over 256 symbols cannot produce a Fibonacci-degenerate
distribution by chance), and the measured maximum code length across every row of this project's
own sweep was well under 16 bits — but the theoretical possibility is named here honestly rather
than silently assumed away (CLAUDE.md §13).

**The degenerate single-symbol edge case.** If an occupancy stream ever contained only ONE distinct
byte value, canonical Huffman construction cannot assign it a genuine 0-bit code (a real
bitstream must consume at least one bit per symbol to know where one ends and the next begins);
`build_huffman_table`/`build_huffman_table_cpu` both force a 1-bit code in that case (the same fix
DEFLATE uses). This never occurs in this project's committed data (every depth's occupancy stream
uses far more than one pattern — the coarsest observed alphabet, at `D=8` pathological, still uses
well over a hundred of the 256 possible symbols) but is handled, not assumed away.

**Determinism.** Every stage is deterministic given the input data: no RNG, no floating-point
summation order sensitivity (the only reductions are integer atomicOr/atomicAdd over a
COMMUTATIVE, ASSOCIATIVE operation — bitwise OR and integer addition are exactly order-independent,
unlike float addition), so repeated runs on the same GPU produce bit-identical results. The
synthetic input data itself is generated with a fixed xorshift32 seed (42) for the same reason.

## How we verify correctness

Two tiers, matching the ruling in `src/reference_cpu.cpp`'s file header:

**Tier 1 — GPU-vs-CPU twins, bit-exact, at the canonical depth D=10, structured cloud** (the
`VERIFY` stage `main.cu` runs before trusting anything else):

| Check | What it compares | Tolerance |
|-------|-------------------|-----------|
| `morton` | GPU vs CPU Morton codes, every point | bit-exact (shared formula; all-integer, see above) |
| `sort` | `thrust::sort` vs `std::sort`, full array | bit-exact (a total order on possibly-equal keys is unique) |
| `octree_levels` | GPU (scan+atomic) vs CPU (sequential accumulator) node counts + occupancy bytes, every level | bit-exact — this is the check that caught the `node_id` off-by-one bug documented in "The GPU mapping" above |
| `histogram` | GPU vs CPU 256-bin histogram | bit-exact (integer counting) |
| `huffman_table` | Heap-based vs linear-scan canonical-Huffman builders, all 256 symbols | bit-exact (canonical form is deterministic — see "The math") |
| `encode_bitstream` | GPU map+scan+scatter vs CPU serial bit-writer, packed bytes + bit count | bit-exact |
| `roundtrip` | `decode(encode(cloud))` vs the independently-computed ground-truth leaf set | bit-exact, in order |

**Tier 2 — analytic/independent gates, every row of the sweep** (both cohorts, all four depths):

- `lossless_roundtrip` — the correctness anchor, re-checked at every depth (not just the canonical
  one), against `octree_unique_leaf_codes` — a trivial dedup pass sharing NO code with the
  decoder, per the "a component with no GPU counterpart is verified end-to-end, against an
  independent ground truth" ruling.
- `distortion_bound` — measured max error vs. the DERIVED half-diagonal formula (no CPU/GPU
  comparison at all — an analytic gate, the strongest kind, because it does not depend on any
  implementation being "independent," only on geometry).
- `rate_monotonic` — the nesting property proven in "The math", checked as a hard inequality on
  measured node counts and errors across the depth sweep.
- `entropy_bound` — measured `L̄` inside the theorem-guaranteed `[H, H+1)` band.
- `entropy_payoff` — see the honest account below; this gate does NOT compare the internal
  Huffman/raw ratio across cohorts (a real finding from this project's own development run showed
  that comparison points the WRONG way), but the end-to-end compression ratio, where structured
  wins decisively and consistently (measured **12.2×** vs **6.2×** at D=10 — a 2.0× margin, and
  never less than 1.65× anywhere in the sweep).

**A genuinely interesting finding, kept honest rather than smoothed over.** An early version of
this project's `entropy_payoff` gate asserted that the structured cloud's Huffman-compressed
occupancy stream should be a SMALLER fraction of its raw stream than the pathological cloud's —
i.e., that entropy coding should reward structure MORE on top of the octree stage. The measured
data flatly contradicts that: at D=10, structured's Huffman/raw ratio is **0.620**, while
pathological's is **0.504** — pathological's byte stream is actually MORE compressible BY
HUFFMAN ALONE. The reason, once you look at it, is physically sensible: a sparse, scattered
cloud's rare non-empty octree nodes disproportionately contain exactly ONE point, so their
occupancy byte disproportionately has exactly one bit set — concentrating the whole histogram onto
just 8 of the 256 possible symbols (measured Shannon entropy 3.98 bits at D=10) — while a
surface's nodes see a WIDER variety of multi-bit crossing patterns as a plane slices through a
cell at different angles and offsets (measured entropy 4.93 bits). The real compression story
(structured needs a **2.4×** smaller octree — 316,830 vs 766,003 nodes at D=10 — hence far fewer
total bits even before entropy coding) lives entirely in STAGE 1, not Stage 2, and the gate was
rewritten to measure the metric that is actually true, rather than the one that was assumed true
before the data was ever run. Both numbers are printed (`rd_curve.csv`'s `shannon_entropy_bits` /
`huffman_avg_bits` columns) so a learner can see this contrast directly, not just take the gate's
word for it.

## Where this sits in the real world

- **PCL's `OctreePointCloudCompression`** (Point Cloud Library) is the closest direct relative:
  occupancy-octree geometry coding plus entropy coding of the resulting stream, with optional
  double-buffering to encode only the CHANGED nodes between consecutive frames (a temporal
  compression axis this project does not implement — every depth's octree is built from scratch;
  see PRACTICE.md §3 for the production version's incremental-update discussion).
- **Google/MPEG's Draco and MPEG's G-PCC (Geometry-based Point Cloud Compression)** are the modern,
  standardized production answer: G-PCC's "octree geometry coding" mode is structurally the same
  idea taught here (occupancy bytes per node), but its entropy stage uses **context-adaptive
  binary arithmetic coding (CABAC)** — coding each of the 8 occupancy bits with a probability
  conditioned on the PARENT node's occupancy pattern and NEIGHBORING nodes' occupancy — which
  captures the local geometric correlation this project's flat, context-FREE 256-symbol Huffman
  table cannot (documented, not implemented, per the ratified scope below).
- **LASzip** (the LAS/LAZ point-cloud format's compressor) targets a different point-cloud SHAPE —
  large-scale airborne/terrestrial LiDAR surveys with per-point attributes (intensity, return
  number, classification) — using a differently-tuned predictive + entropy-coding pipeline, but
  the SAME core idea: exploit spatial coherence, then entropy-code the residual.
- **Arithmetic/range coding, honestly**: as "The math" notes, Huffman's `+1` bit slack per symbol
  is real; arithmetic coding removes it, approaching the Shannon bound arbitrarily closely at the
  cost of losing simple byte-at-a-time decodability (a real, non-parallelizable serial
  dependency EVEN STRONGER than this project's Huffman decoder — arithmetic decode cannot restart
  at a symbol boundary at all without an explicit flush). Production point-cloud codecs (G-PCC,
  Draco) use range coding for exactly this reason; this project teaches the simpler, byte-boundary-
  friendly Huffman alternative and documents range coding as the natural next step (README
  "Exercises").
- **Run-length encoding (RLE)**, named honestly as an alternative worth knowing: RLE would compress
  RUNS of identical occupancy bytes cheaply, which helps almost NOT AT ALL here, because — as this
  project's own measured histograms show — consecutive nodes rarely share the exact same
  occupancy pattern even in highly compressible structured data (the 256-symbol ALPHABET is
  skewed, but any given symbol rarely repeats many times consecutively in Morton-sorted order).
  Entropy coding the WHOLE alphabet (Huffman) is the right tool here; RLE is the right tool for a
  different kind of redundancy this project's data does not exhibit.
- **What a full research/production version would add beyond this teaching core**: temporal
  (frame-to-frame) delta coding for incrementally-updated maps; context-adaptive entropy coding
  (CABAC-style, conditioning on parent/neighbor occupancy); per-point attribute compression
  (intensity, semantic labels, color) alongside geometry; a parallel BLOCK-WISE decoder (see "The
  GPU mapping"); and a learned (neural) entropy model, an active research direction MPEG's G-PCC
  successor efforts are exploring.
