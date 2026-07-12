# 02.19 — PointPillars/CenterPoint voxelization + scatter kernels feeding TensorRT: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics.

## The problem — physics & engineering first

A spinning or solid-state LiDAR reports a **point cloud**: an unordered bag of returns
`(x, y, z, intensity)`, one per laser pulse that hit something. Physically, this is a *sparse sample of
a 3-D surface* — the returns lie on whatever object surfaces were in the beam's path, with density that
falls off as ~1/r² with range (beam divergence spreads a fixed angular resolution over a growing area;
[`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md) derives this exactly).
Crucially, a point cloud has **no fixed shape**: frame to frame, the point count varies, the memory
addresses of "the points near the ego vehicle" vary, and there is no notion of "pixel (3, 7)" the way a
camera image has one.

A **learned** 3-D object detector, however, is (almost always) a convolutional or matmul-heavy neural
network, and neural network layers want **fixed-shape, dense, regular tensors** — the same shape every
forward pass, so the compute graph, the memory allocations, and (critically for deployment) the
TensorRT engine's kernel selection can all be decided once, ahead of time, rather than re-planned every
frame. This is the **impedance mismatch** this project is entirely about bridging: turning a variable-
size, sparse, unordered bag of points into a fixed-shape, dense tensor a network — and, downstream,
TensorRT — can consume without surprises.

**PointPillars** (Lang et al. 2019) resolves the mismatch with a specific, elegant choice: discretize
the ground plane into a bird's-eye-view (BEV) grid of "pillars" — vertical columns with **no z-split at
all**. Every point's (x,y) decides which pillar it belongs to; z only participates inside each pillar's
feature computation, never in the binning itself. This is a genuine physical simplification: it throws
away the *vertical* position of features within a pillar's column (a car's roof and a pedestrian's head
both just contribute to "this pillar has points up to z=1.5 m"), in exchange for a 2-D (not 3-D) dense
tensor — dramatically less memory and compute than a full 3-D voxel grid, at the cost of some vertical
discriminative power. **CenterPoint** (and 3-D voxel backbones generally) keep a coarse z-split instead,
trading some of that memory/compute back for vertical structure. This project implements PointPillars'
z-collapsed pillars as the primary path and measures the CenterPoint-style 3-D voxel alternative
directly (`[info] pillar_vs_voxel`) so the trade is not asserted, it is *shown*.

The engineering constraint that makes this a genuinely hard systems problem, not just a formatting
exercise: **the number of occupied pillars is not known until runtime**, and **each pillar can be
overrun by an arbitrary number of points** (a nearby wall, a dense cluster of foliage, or — in this
project's synthetic scene — a deliberately adversarial cluster). A production system must (a) cap each
pillar's point count to bound the tensor's inner dimension, (b) decide *which* points to keep when a
pillar overflows, and (c) do all of this on a GPU where thousands of points are binned concurrently. The
"which points to keep" decision turns out to be a genuine **numerical determinism** hazard, not just a
detail — see "Numerical considerations" below.

## The math

**Point cloud.** `points ∈ ℝ^{N×4}`, row *i* = `(x_i, y_i, z_i, r_i)` (meters, meters, meters, unitless
reflectance), BEV/ego frame, +x forward, +y left, +z up (CLAUDE.md §12).

**The BEV grid.** Pillar size `s = 0.4 m`; grid `N_x = N_y = 200` pillars; window
`x ∈ [x_min, x_min + N_x·s)`, `y ∈ [y_min, y_min + N_y·s)`, with `x_min = y_min = -40 m` here
(`kernels.cuh`). A point's pillar indices:

```
i_x = ⌊(x - x_min) / s⌋,    i_y = ⌊(y - y_min) / s⌋
key = i_y · N_x + i_x                                   (dense cell id, 0 <= key < N_x·N_y)
```

`⌊·⌋` must be a true **floor**, not truncation-toward-zero: for a negative-side point (e.g.
`x = -0.3 m, s = 0.4 m` in a grid with `x_min=0`), `(int)(-0.3/0.4) = 0` truncates, but
`⌊-0.75⌋ = -1` is the physically correct pillar (the one spanning `[-0.4, 0.0)`).
`std::floor`/`floorf` gets this right unconditionally
([`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)'s "The math" derives
the identical pitfall for its voxel grid).

A point is **out of window** (excluded from every downstream stage) when `i_x ∉ [0,N_x)`,
`i_y ∉ [0,N_y)`, or `z ∉ [z_min, z_max]` (the vertical window, separate from the (x,y) key — z never
participates in `pillar_key_of` itself, only in the per-pillar feature computation and the CenterPoint
voxel key).

**CenterPoint-style 3-D voxel key**, the same grid split into `Z = 2` uniform z-bands of thickness
`b = (z_max - z_min)/Z`:

```
i_z = ⌊(z - z_min) / b⌋,   0 <= i_z < Z
voxel_key = i_z · (N_x·N_y) + key
```

Bounded (like the pillar grid — see "The GPU mapping" for why this project needs **no** spatial hash
table, unlike 02.01's unbounded scan). At `z_min=-3, z_max=5, Z=2`: `b = 4 m`, splitting near
`z = -3+4 = 1.0 m` — roughly ground/wheel-height (band 0) vs. car-body/roof-height (band 1) in this
project's synthetic scene.

**Per-pillar point cap and truncation.** Cap `C = 32`. If a pillar's arrival count `n_p > C`, only `C`
of the `n_p` points are *kept* — which `C` depends on the binning **method** (see "The algorithm"). Kept
points populate slots `0..C-1`; slots `≥` the kept count are zero-padded (the fixed tensor-shape
requirement — every pillar's row is exactly `C×D` regardless of how many points actually landed there).

**The 9-D per-point feature vector** (PointPillars' `D=9`), for kept point *k* in pillar *p* with kept
set mean `(x̄_p, ȳ_p, z̄_p)` (over the *kept* points only — see "Numerical considerations" for why that
choice matters) and the pillar's fixed geometric center `(c_x, c_y) = (x_min + (i_x+0.5)s,\ y_min +
(i_y+0.5)s)`:

```
f = [ x_k, y_k, z_k, r_k,                     — raw geometry + reflectance
      x_k - x̄_p, y_k - ȳ_p, z_k - z̄_p,        — offset from the pillar's KEPT-point mean ("shape" term)
      x_k - c_x,  y_k - c_y ]                 — offset from the pillar's fixed geometric center ("anchor" term)
```

Why both offset terms, when either alone looks redundant? The **mean-offset** terms describe the
point's position *relative to the local point cloud shape inside this pillar* — translation-invariant,
so the same local shape (say, a car's roof edge) produces the same mean-offset features no matter where
in the world it is. But mean-offset ALONE cannot tell the network *where in the pillar* (near an edge vs.
centered) the point sits, because the mean itself moves with however many/which points are kept — a
network could not distinguish "3 points hugging one edge" from "3 points spread evenly" using only
mean-offsets computed from those same 3 points. The **center-offset** terms anchor each point against
the pillar's fixed geometry (independent of which points survived truncation), recovering that missing
positional information. This is the concrete version of the "translation-invariance vs. absolute
position" tension that also motivates keeping the raw `x,y` terms small-weighted in this project's PFN
(`data/README.md`'s weight-design note) — absolute (x,y) alone tells the network "you are at world
position (20, 20)," useful for range-dependent effects but actively unhelpful for recognizing the same
LOCAL shape wherever it appears.

**PFN-lite** (the fixed, non-learned stand-in for PointPillars' learned Pillar Feature Net). A linear
layer `W ∈ ℝ^{4×9}, b ∈ ℝ^4` (fixed, `data/sample/pfn_lite_weights.csv`, seed 42 — never trained) maps
each kept point's 9-D feature to 4 channels, ReLU'd, then **max-pooled across the pillar's kept points**:

```
h_k = ReLU(W f_k + b) ∈ ℝ^4            for each kept point k
g_p = max_k h_k                        (elementwise max over the pillar's kept points)
```

plus two EXPLICIT, hand-computed channels appended (not part of the linear layer — documented as such
so the "designed vs. learned" line stays honest): **occupancy** `= kept_count/C ∈ [0,1]`, and **height
extent** `= (max_k z_k - min_k z_k)/h_norm`, clamped `[0,1]`, `h_norm = 2 m`. Final pillar feature
`pillar_feat_p = [occupancy, height_extent, g_p] ∈ ℝ^6`.

**Why max-pool = permutation invariance (the Deep Sets argument).** A pillar's kept points have no
canonical order — they are a *set*, not a sequence — yet the network needs one fixed-size vector per
pillar. `max` (like `sum` or `mean`) is a **symmetric function**: `max(h_1,...,h_n) = max(h_{σ(1)},
...,h_{σ(n)})` for any permutation `σ`. Zaheer et al.'s "Deep Sets" (2017) formalizes this: any function
on an unordered set that must be permutation-invariant can be written as `ρ(pool_i φ(x_i))` for some
per-element embedding `φ` and pooling operator `pool` — exactly the shape `ρ = identity, φ = ReLU(Wx+b),
pool = max` here. **Except**: this invariance is over the *set that reaches the pooling operator* — it
says nothing about which points are *in* that set. The truncation policy (below) decides that set, and
a nondeterministic truncation policy makes the pillar feature nondeterministic too, even though the
pooling step itself is perfectly permutation-invariant. This is the exact seam kernels.cuh's
cap_truncation gate probes.

**The BEV canvas.** `canvas ∈ ℝ^{C_ch×H×W}` (NCHW, `C_ch=6, H=W=200`), initialized to zero; occupied
pillar *p* at `(i_x,i_y)` writes `canvas[:, i_y, i_x] = pillar_feat_p`. Unoccupied cells stay exactly
zero — the "sparse-to-dense" step.

**The toy head.** Two 3×3 convolutions (`kernels.cuh`'s `kSmoothKernel3x3`, `kSharpenKernel3x3`) with an
elementwise occupancy gate between them:

```
S = conv(canvas[1], smooth_kernel)                 — smooth the height-extent plane
G = S ⊙ canvas[0]                                   — gate by (unsmoothed) occupancy
H = conv(G, sharpen_kernel)                          — the final heatmap
```

Peak extraction: pixel `(i_y,i_x)` is a **candidate** iff `H[i_y,i_x] > threshold` and it is the
strict local maximum (deterministic tie-break: an equal-valued neighbor at a smaller flattened index
wins) in its `(2r+1)×(2r+1)` window, `r=2`. NMS then greedily keeps the highest-scoring candidate,
suppressing every remaining candidate within `d_nms` pillars (Euclidean, in pillar units).

## The algorithm

**Method A — atomic per-pillar slot claim.** One thread per point:
```
key = pillar_key_of(x,y,z)                  // -1 if out of window
if key < 0: return
slot = atomicAdd(&point_count[key], 1)       // "my claimed slot index"
if slot < C: raw_points[key][slot] = (x,y,z,r)
```
`atomicAdd` returns the value **before** the increment — the classic GPU counting-allocator idiom.
Complexity: `O(N)` work, fully parallel; the only serialization is per-CELL (points in different pillars
never contend). Correctness of the COUNT is guaranteed by atomicity; correctness of WHICH points occupy
slots `0..C-1` when a pillar overflows depends on the **order** the hardware serializes concurrent
`atomicAdd` calls — a real, load-bearing caveat, not a footnote (see "Numerical considerations").

**Method B — sort + fixed-order truncation** (reusing
[`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)'s Method B lineage
directly):
1. Filter out-of-window points (`key < 0`) via stream compaction (`thrust::copy_if`).
2. `thrust::stable_sort_by_key(keys, keys+n_valid, original_index)` — sorts by pillar key; **stability**
   guarantees that within any pillar's run, points remain in ASCENDING ORIGINAL INDEX order regardless of
   which order they arrived in.
3. Mark segment boundaries (`mark_boundaries_kernel`, a 1-neighbor comparison map), then compact via
   `reduce` (count) + `copy_if` (offsets) — exactly 02.01's boundary-compaction pattern.
4. `sorted_bin_kernel`: one thread per occupied SEGMENT, copies the FIRST `min(run_len, C)` points of
   that fixed sorted order into the dense per-cell storage.

Complexity: the sort dominates at `O(N log N)` in general (Thrust's radix sort is `O(N · key_width /
radix_bits)` for integer keys, effectively linear here since the key width is small and fixed); every
other step is `O(N)` or `O(#occupied cells)`. Because the ORDER used for truncation is now a pure
function of original point index (not of hardware scheduling), the surviving point set is a
**deterministic function of the input** — the property Method A does not have.

**The same machinery, two key functions.** `launch_sort_and_compact()` is written generically over "N
points, a per-point integer key, C possible cells" — it has no idea whether the key came from
`pillar_key_of` (PointPillars) or `voxel_key_of` (CenterPoint-style). This is not a coincidence of
implementation convenience: at the kernel level, PointPillars and CenterPoint's voxel encoders truly
ARE the same binning machinery with a different key formula — the catalog bullet names both networks in
one breath for exactly this reason.

## The GPU mapping

**Thread-to-data mapping**, kernel by kernel:
- `compute_*_keys_kernel`, `atomic_bin_kernel`, `mark_boundaries_kernel`, `elementwise_mul_kernel` —
  one thread per POINT (or per array element), grid-stride loop (08.01/02.01's idiom: correct for any
  `n`, lets the caller pick grid size for occupancy rather than being forced into `⌈n/block⌉` blocks).
- `pfn_stats_kernel`, `augment_features_kernel` (one thread per (pillar,slot) pair), `pfn_lite_kernel`,
  `scatter_kernel`, `gather_kernel`, `gather_occupied_cell_kernel`, `sorted_bin_kernel` — one thread per
  OCCUPIED PILLAR (or per pillar×slot pair for `augment_features_kernel`'s finer parallelism).
- `conv3x3_kernel`, `peak_extract_kernel` — one thread per OUTPUT PIXEL, 2-D `(16,16)` blocks (the
  standard image-kernel default; `200×200` needs `⌈200/16⌉²=13²=169` blocks).

**Memory hierarchy.** Everything here uses **global memory** only — no shared memory, no textures. Why:
the reductions involved are tiny (`≤32` points per pillar for the PFN, `9` neighbors per pixel for the
conv) — small enough that register-resident per-thread loops (the compiler keeps a `float acc` and a
handful of temporaries in registers automatically) already avoid repeated global traffic *within* one
thread's work; the classic shared-memory win (staging a TILE of data once, reused by many threads in a
block) does not apply because neighboring threads in `augment_features_kernel`/`pfn_lite_kernel` do NOT
share input data (each pillar's points are private to that pillar), and `conv3x3_kernel`'s 3×3
neighborhoods overlap only slightly for `H=W=200` — a real, large-image conv stencil kernel WOULD stage
a halo tile into shared memory (Exercise territory; THEORY.md flags it rather than hiding it).

**Why no spatial hash table** (the direct contrast with
[`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)): 02.01's voxel grid is
UNBOUNDED — a LiDAR scan's extent is not known ahead of time, so there is no way to allocate one array
slot per possible voxel; a hash table (with its open-addressing probe chains, load-factor tuning, and
`atomicCAS` claim loop) is the only option. A BEV detector's input window, by contrast, is a **design
choice** — "we care about objects within ±40 m" — so the grid is bounded and SMALL (`40,000` cells)
enough that every pillar gets a dense, unique, hash-free array slot: `cell = i_y·N_x + i_x` directly
addresses `point_count[cell]` and `raw_points[cell]`. This project's binning is therefore *simpler* than
02.01's despite solving a structurally similar problem — a genuine lesson about when a bounded domain
changes the right data structure.

**Why the scatter is NCHW, not HWC** (`scatter_kernel`'s comment expands this): one occupied pillar's
`scatter` writes land in `C_ch=6` DIFFERENT, far-apart channel-planes — not coalesced across channels
within one thread. An HWC (channel-interleaved) canvas would make that ONE pillar's write perfectly
contiguous instead. But the conv stage that reads the canvas runs `H·W·9` times per pass (two passes,
every pixel, every one of 9 taps) — vastly more often than the `num_occupied` scatter writes — and a
conv thread's 3×3 neighborhood is naturally contiguous WITHIN one NCHW channel plane, not across
interleaved channels. NCHW trades a slower, rare write for a faster, frequent read — the right trade
here, and the reason production frameworks (and TensorRT) default to NCHW for conv-heavy tensors too.

**Occupancy and launch configuration.** 1-D kernels use 256-thread blocks (a warp multiple, the repo
default, good occupancy on sm_75..sm_89) capped at 4,096 blocks with a grid-stride loop absorbing any
remainder. 2-D kernels use `16×16=256`-thread blocks for the same total-thread-per-block reasoning,
tiled over the `200×200` canvas.

**Library calls.** `thrust::stable_sort_by_key` — a RADIX sort under the hood for integer keys:
repeated stable partitioning by a few bits of the key at a time, LSB to MSB, `O(n · key_width/radix_bits)`
work, fully data-parallel per pass (conceptually the same idea
[`22.01`](../../22-multi-robot-swarms/22.01-100k-agent-swarm-sim/README.md)'s hand-rolled counting-sort
neighbor binning implements by hand, at smaller scale — Thrust's version is that idea, generalized and
tuned). `thrust::reduce`/`thrust::copy_if` — a standard parallel tree reduction and stream-compaction
primitive respectively; both dispatch to well-tuned CUB kernels underneath. We use Thrust here
specifically for its STABILITY guarantee on the sort (the foundation of Method B's determinism claim),
not merely for convenience.

## Numerical considerations

**The central numerical/determinism story: atomic truncation order.** CUDA's memory model guarantees
`atomicAdd` is atomic — no two threads ever read-modify-write the same counter simultaneously — but it
makes **no promise about the ORDER** concurrent `atomicAdd` calls from different threads/blocks are
serialized in. On a real GPU, that order is a function of the hardware's warp/block scheduler, which in
turn depends on occupancy, memory-system timing, and (on a busy system) contention from unrelated work.
**Empirically**, identical kernel launches on identical inputs, on a QUIET GPU, often DO reproduce the
same schedule — this project's own `[info] cap_truncation: atomic SAME-order` line measures exactly
this and, on the reference machine, sometimes shows zero variance across 3 repeats. That empirical
stability is NOT a language guarantee, and the moment the ARRIVAL ORDER of points changes — a virtual
certainty across a fleet (packet reordering, multi-return interleaving, multi-sensor merge order,
different DMA timing) — the atomicAdd claim order changes with it, silently, with no error and no
warning. This project demonstrates that exact failure mode reliably (not depending on fragile same-order
scheduler luck) by re-running Method A over 3 independently-shuffled copies of the SAME logical point
set and measuring how many of the 32 kept points differ (`[info] cap_truncation: atomic SHUFFLED-order`
— on the reference machine, tens of the 32 slots differ between shuffles). The downstream consequence is
real: a pillar's mean, its 9-D features, its PFN-lite output, and therefore the network's input all
change, silently, from run to run, on IDENTICAL sensor data — a genuine ML-reproducibility bug, not an
academic curiosity, hiding inside what looks like "just a preprocessing kernel."

**Why Method B is bit-exact, not merely close.** `sorted_bin_kernel` (GPU) and `sorted_bin_cpu` (the CPU
oracle) both compute a kept pillar's mean by summing FLOAT values in the SAME fixed order — ascending
original point index, guaranteed by both `thrust::stable_sort_by_key` and `std::stable_sort`'s identical
stability contract. IEEE-754 float addition is deterministic given identical operands in identical
order (no reassociation, no fused-multiply-add ambiguity for a chain of plain `+=`), so the GPU and CPU
sums round IDENTICALLY, bit for bit — this project's `VERIFY(binning)` measures the actual worst
observed difference (`3.8e-6` on the reference machine, well inside noise from the tensor's OTHER,
genuinely-not-bit-exact fields) rather than asserting it blindly.

**Where 1e-4, not bit-exactness, is the honest bound.** The PFN-lite's linear layer (`W·f + b`, 9
multiply-adds per channel) and the conv head's 3×3 weighted sums are exactly the kind of arithmetic
where nvcc's device compiler and MSVC's `cl.exe` host compiler may legally fuse a multiply and an add
into a single-rounding FMA differently (or not at all) — the IEEE-754 standard permits, but does not
require, FMA fusion, and the two compilers' default choices need not agree. The resulting per-operation
difference is at most ~1 ULP; accumulated over 9-36 operations it stays comfortably under `1e-4` for
this project's value ranges — measured, not merely asserted (`VERIFY(pfn)`/`VERIFY(head)`'s `[info]`
lines print the actual worst-case difference every run, currently ~1e-6 to 1e-7 — far inside the bound,
leaving generous headroom against a different GPU architecture's rounding).

**Mean-over-kept-points, not mean-over-arrived-points.** This project computes each pillar's mean
`(x̄,ȳ,z̄)` over the KEPT points only (post-truncation), matching common real-world implementations
(e.g. mmdetection3d) — the network only ever sees the kept points, so a mean including dropped points
would describe a set the network cannot observe. This is a documented design choice, not the only
possible one.

**FP16/INT8 (documented-only).** A real TensorRT deployment (12.01, 12.03) would quantize this
pipeline's tensors to FP16 or INT8 for inference throughput — this project stays FP32 throughout and
does not measure or claim any quantization-error number; `[info] trt_handoff` names the real path
without fabricating results for it.

## How we verify correctness

Two tiers, matching [`docs/PROJECT_TEMPLATE/src/reference_cpu.cpp`](../../../docs/PROJECT_TEMPLATE/src/reference_cpu.cpp)'s
independence ruling:

1. **Twin comparison** (`VERIFY(...)` lines) — the GPU's Method B (deterministic) pipeline vs.
   `reference_cpu.cpp`'s independently-coded twin, stage by stage: `keys` (bit-exact — both call the
   SAME shared `kernels.cuh` formula, but the GPU path is a hand-transcribed `__device__` copy, so a
   typo in either copy is caught here), `binning` (occupied-pillar list + kept counts + the 9-D feature
   tensor — bit-exact per the ordering argument above), `pfn` (the fixed linear layer + max-pool, 1e-4),
   `scatter` (a pure copy — bit-exact), `head` (the two conv passes + gate, 1e-4), `peaks` (the final
   detection list, position-exact). The sort itself is genuinely independent: `thrust::stable_sort_by_key`
   (GPU) vs. `std::stable_sort` (CPU) — different libraries, different algorithm implementations, sharing
   only the STABILITY *contract*, not code.
2. **Independent gates** (`GATE ...` lines) — checks that do NOT route through the twin comparison, so a
   bug shared by both the GPU and CPU implementations (impossible here for `pillar_key_of` itself, since
   it is single-sourced — but very possible for, say, a consistently-wrong pillar-center formula) would
   still be caught:
   - **`layout_roundtrip`** — `gather(scatter(x)) == x` on every occupied pillar: a pure bookkeeping
     identity (no arithmetic), so ANY mismatch is a genuine indexing bug, not a numerical one.
   - **`cap_truncation`** — the determinism study above, PLUS a hand-provable anchor: the sorted
     method's cap-stress pillar keeps points `{0,...,31}` by original-stream index (a fact derivable by
     hand from the truncation rule and the generator's known contiguous placement — never merely
     observed).
   - **`feature_semantics`** — a hand-constructed 3-point pillar (`x,y,z,intensity` chosen so the mean
     and pillar-center arithmetic are easy to verify by hand) run through the pipeline; every one of the
     3×9 output values is compared against an ANALYTICALLY computed (double-precision, worked in a
     comment in `main.cu`) expected value, 1e-4 tolerance — the "free exactness anchor" that does not
     depend on trusting either implementation's internal consistency.
   - **`detection_closure`** — the end-to-end proof that the LAYOUT is right, not just each stage in
     isolation: every one of the 6 synthetic cars must produce a detection within 3 m of its true
     center, and every detection must map back to a real car (zero false peaks on the cap-stress pillar
     or clutter). A garbage layout — swapped x/y, a wrong pillar-center formula, a broken scatter index
     — would still likely pass the PER-STAGE twin comparisons (both paths would compute the SAME wrong
     thing) but would almost certainly fail `detection_closure`, because the wrongness would show up as
     misplaced or absent detections against INDEPENDENTLY known ground truth.

## Where this sits in the real world

**PointPillars** (Lang et al., CVPR 2019) and **CenterPoint** (Yin, Zhou, Krähenbühl, CVPR 2021) are
real, widely-deployed 3-D detectors; this project implements their voxelization/scatter PREPROCESSING
faithfully (the same key formulas, the same 9-D feature vector, the same scatter step) while replacing
their LEARNED components with fixed/hand-designed stand-ins, documented at every site. What a real
trained PFN adds beyond this project's fixed linear layer: many more channels (64-128 typical, not 4),
multiple linear+BN+ReLU layers (not one), and weights trained end-to-end with the detection head on a
labeled dataset (KITTI, nuScenes, Waymo) — so the 4 "meaningless" channels this project's PFN-lite
produces would, after training, become genuinely discriminative shape descriptors. What a real trained
head adds beyond this project's hand-designed 2-layer conv: a much deeper CNN backbone (typically a
small ResNet-style network over the BEV canvas), an SSD-style anchor mechanism or CenterPoint's
learned center-heatmap regression (this project's toy head is architecturally closer to CenterPoint's
heatmap idea than to PointPillars' anchor boxes, though neither is implemented in full), and
box-size/orientation regression heads this project does not attempt at all.

**NVIDIA's CUDA-PointPillars** (open-source reference implementation, part of the DeepStream/TAO
ecosystem) implements this exact preprocessing pipeline — voxelization, feature generation, scatter —
as production CUDA kernels feeding a TensorRT-deployed learned PFN and backbone; its kernel-level
architecture (dense per-cell storage for a bounded BEV window, a scatter kernel, FP16 inference) matches
this project's design closely, at production scale (~100k-point scans, `P_max` in the thousands to tens
of thousands, sub-10ms end-to-end preprocessing budgets). Project
[`12.01`](../../12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/README.md) is this
project's sibling for the deployment half of that story — the actual TensorRT engine build, custom
pre/post-processing plugins, and the optional-SDK/plain-CUDA-fallback pattern this project's README
cites as precedent.

**Sparse convolution libraries** (spconv, MinkowskiEngine, TorchSparse) exist precisely because of the
economics `[info] sparsity_economics` measures: once channel counts grow into the hundreds (a real
trained backbone), materializing a DENSE `[C,H,W]` canvas for a scene that is typically <20% occupied
wastes the large majority of memory bandwidth and compute on cells that are always zero. Production 3-D
detectors increasingly run sparse convolutions over the OCCUPIED-cell list directly, only ever
materializing a dense tensor (if at all) at the very last stage before a dense detection head — this
project's single scatter-then-dense-conv design is the simple, teachable version of a much larger
engineering space.

**`[R&D]` note:** this catalog bullet is not tagged `[R&D]`, so no reduced-scope research framing
applies; the simplifications above (fixed weights, no orientation regression, no `P_max` padding) are
ordinary teaching-scope decisions (CLAUDE.md §13), not research-frontier gaps.
