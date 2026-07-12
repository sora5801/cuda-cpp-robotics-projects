# 02.02 — ROI crop, passthrough, organized↔unorganized conversion kernels: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Why a LiDAR return goes invalid

A spinning mechanical LiDAR (this project's model: 16 beams, -15°..+15° elevation in 2° steps,
identical to project 01.18's teaching camera-LiDAR rig) fires a laser pulse along each beam
direction at every azimuth step and times the round trip of the reflected light (`range = c * dt / 2`,
`c` = speed of light). A cell in the resulting ring x azimuth grid is **invalid** — no return — for
one of two physically distinct reasons this project's synthetic data models separately, even though
both end up as the same NaN sentinel (a real driver's packet cannot tell them apart either):

1. **Geometric miss.** The beam simply does not hit anything within the sensor's maximum range
   (`MAX_RANGE_M` = 20 m here). This project's virtual room deliberately has **no ceiling** — any
   beam tilted enough to clear the 1.5 m wall top before reaching a wall escapes to open sky and
   returns nothing. Real robots see this constantly: a beam grazing the horizon, or angled above the
   tallest object in range, sees nothing and reports nothing.
2. **Absorption / specular deflection / max-range falloff / sun noise (the sensor-model dropout).**
   Even when geometry says "there is a surface here," the return can still be lost:
   - **Absorption**: matte black, foam, or highly-absorptive materials return too little energy for
     the receiver's noise floor.
   - **Specular deflection**: glass, polished metal, or wet surfaces at a grazing angle reflect the
     beam AWAY from the receiver rather than back along its path (the same physics that makes mirrors
     invisible to a LiDAR pointed at the wrong angle).
   - **Max-range falloff**: received power falls off as `1/r^2` (the inverse-square law, same beam
     divergence physics 02.01's THEORY.md derives for point density) — near the sensor's rated max
     range, ordinary-reflectivity targets can drop below the noise floor even though a highly
     reflective target at the same range would not.
   - **Sun noise**: outdoor daytime operation adds background photon flux across the receiver's
     entire passband; a weak return can be swamped by ambient solar infrared, especially at low sun
     angles (dawn/dusk, a known operational weak point of time-of-flight LiDAR).

   This project models this whole family as one lumped, independent **5% dropout probability**
   applied to every otherwise-successful geometric hit (`DROPOUT_PROB` in
   `scripts/make_synthetic.py`) — an honest simplification: a real sensor's dropout rate is angle-,
   material-, and range-dependent, not a flat constant, but a flat rate is enough to teach the
   downstream consequence (organized grids have holes; algorithms that assume a dense grid break).

### Why the organized grid is the sensor's native geometry, and why that matters

A spinning LiDAR does not produce "a list of 3D points" — it produces a stream of **(ring, azimuth,
range)** triples, one per firing, in a fixed, predictable order (every ring fires at every azimuth
step, azimuth step monotonically increasing as the head spins). The **organized grid** —
`kNumBeams` rows (rings) x `kAzimuthBins` columns (azimuth), ring-major, one cell per (ring, azimuth)
pair, invalid cells NaN — is nothing more than this native firing order laid out as a 2D array. It
is the sensor's OWN geometry, not a derived convenience.

This matters because **organized structure turns neighbor-finding from a SEARCH into an INDEX
LOOKUP**. Given a point at organized index `(r, a)`, its four immediate angular neighbors are simply
`(r±1, a)` and `(r, a±1)` — an O(1) array access. An **unorganized** point list has thrown that
structure away: finding the same neighbors requires a spatial search (a k-d tree, a voxel hash — the
machinery of projects 02.05/02.01). Range-image algorithms (project **02.12**, depth-clustering on a
range image) and fast normal estimation (project **02.09**) exist specifically to exploit this O(1)
lookup — they would be meaningfully slower, or need an entirely different algorithm, on an
unorganized cloud. This project's organized->unorganized kernel is therefore not "the normal
direction" and unorganized->organized "the unusual one" — both directions are equally real production
operations, chosen by whichever consumer is next in the pipeline.

### The engineering constraint: this must be nearly free

Every kernel in this project sits **between** the LiDAR driver and everything else in the perception
stack (README "System context"). A spinning LiDAR reports at 10-20 Hz; the entire perception +
planning + control loop downstream has a total latency budget on the order of tens of milliseconds.
A crop/passthrough/conversion step that costs even 1-2 ms per scan is a measurable tax on that
budget — these are meant to be near-free "glue," not a computational bottleneck (README "System
context" states the measured numbers on the owner's GPU). That constraint is what makes this
project's compaction primitive — a scan with cost `O(n)` and no wasted work — the right tool: a naive
"copy everything, then remove the bad ones with erase-remove" approach on the GPU would either need
multiple passes or non-coalesced writes; the scan-then-scatter pattern below does the whole job in
two kernel launches with fully coalesced memory access.

## The math

### Notation

- `p = (x, y, z)`, meters, `lidar` sensor frame (right-handed, x-forward/y-left/z-up,
  SYSTEM_DESIGN.md §3.2).
- `flags[i] in {0, 1}` — the keep/drop decision for point `i` under some predicate.
- `scan[i] = sum(flags[0..i))` — the **exclusive prefix sum** ("exclusive scan") of `flags`.

### The compaction identity (why a scan solves compaction)

**Claim:** if `flags[i] = 1` (point `i` is kept), its destination index in the compacted output is
exactly `scan[i]`.

**Proof sketch:** `scan[i]` counts how many points BEFORE `i` were kept. Every one of those earlier
kept points has ALREADY claimed a destination index in `{0, ..., scan[i]-1}` (by the same argument,
inductively) — none of them can claim `scan[i]` or later. No LATER point can claim an index before
`scan[i]` either, because `scan` is monotonically non-decreasing (`flags >= 0`) and a later point's
scan value is `>= scan[i] + flags[i] = scan[i] + 1 > scan[i]`. So `scan[i]` is available, and it is
the ONLY consistent index for point `i` to land at if every kept point writes to `scan[<its index>]`.
This is exactly what `compact_scatter_kernel` does (kernels.cu), and it is why the compaction is
**stable** (order-preserving): `scan` is monotonic in `i`, so two kept points `i < j` always satisfy
`scan[i] < scan[j]` (since `scan[j] >= scan[i] + flags[i] = scan[i] + 1`) — their RELATIVE order
survives compaction exactly.

**Why order preservation matters downstream:** many consumers implicitly rely on point order —
matching a compacted cloud's points back to auxiliary per-point data (intensity, ring/azimuth,
timestamp) computed BEFORE the crop requires the crop to preserve order (or ship a permutation, which
this project's `orig_idx` output arrays effectively are). A non-stable compaction would silently
scramble that correspondence — a real, hard-to-detect bug class this project's `GATE
order_preservation` exists to catch by construction, not by luck.

### Work/depth analysis: naive serial scan vs. Blelloch work-efficient scan

For an array of `n` elements:

| Algorithm | Work (total operations) | Depth (parallel steps) |
|---|---|---|
| Naive serial scan (one thread, a running sum) | `O(n)` | `O(n)` |
| Hillis-Steele scan (the simplest parallel scan: every element adds its neighbor `2^d` away, for `d = 0..log2(n)`) | `O(n log n)` | `O(log n)` |
| Blelloch work-efficient scan (this project: up-sweep reduce + down-sweep distribute) | `O(n)` | `O(log n)` |

**Derivation, Blelloch up-sweep:** round `d` (0-indexed) combines `n / 2^(d+1)` pairs, so total work
across `log2(n)` rounds is `sum_{d=0}^{log2(n)-1} n/2^(d+1) = n * (1/2 + 1/4 + ... ) = n - 1 = O(n)`.
The down-sweep is symmetric (also `O(n)`), so total work is `O(n)`, matching the SERIAL algorithm's
work exactly, while depth is `O(log n)` because each round runs its active threads in parallel. This
is the whole point of "work-efficient": Hillis-Steele's `O(n log n)` does `log n` TIMES more work
than necessary to buy the same `O(log n)` depth; Blelloch buys the same depth for free.

### Frustum plane derivation (the "five-plane test")

The camera has intrinsics `fx, fy` (focal lengths, pixels), `cx, cy` (principal point, pixels), and
image size `W x H`. The **pinhole projection** of a camera-frame point `p_cam = (x, y, z)` (z > 0,
forward) is `u = fx*x/z + cx`, `v = fy*y/z + cy`. A point is visible iff `0 <= u <= W-1` and
`0 <= v <= H-1` and `z > 0` (in front of the camera).

Multiplying the pixel-bounds inequalities through by `z` (valid since `z > 0` preserves inequality
direction) turns each into a **linear** test with no division — the whole point of doing this
algebra once, on paper, instead of computing a division per point per test on the GPU:

```
u >= 0        =>  fx*x + cx*z >= 0                    (LEFT plane:   n_left   = (fx, 0, cx))
u <= W-1      =>  (W-1-cx)*z - fx*x >= 0               (RIGHT plane:  n_right  = (-fx, 0, W-1-cx))
v >= 0        =>  fy*y + cy*z >= 0                     (TOP plane:    n_top    = (0, fy, cy))
v <= H-1      =>  (H-1-cy)*z - fy*y >= 0               (BOTTOM plane: n_bottom = (0, -fy, H-1-cy))
z >= near     =>  z - near >= 0                        (NEAR plane:   n_near   = (0, 0, 1), offset -near)
```

Each of `n_left/right/top/bottom` is the normal of a plane THROUGH THE CAMERA ORIGIN containing one
edge of the image — this is the standard view-frustum construction (four side planes fan out from
the eye point; only `near`/`far` do not pass through the origin). This project omits the traditional
**far** plane: passthrough/box already bound range, and a camera's frustum has no PHYSICAL far
plane the way a rasterizer's clip volume has an arbitrary one — see README "Limitations" for the
explicit scoping note.

A point `p_lidar` is tested by first transforming to the camera frame,
`p_cam = R * p_lidar + t` (`kTCameraLidar`, reused verbatim from project 01.18's derivation of a roof
LiDAR above a windshield-height camera — see that project's THEORY.md "The math" for the rotation's
derivation), then evaluating the five inequalities above. `is_in_frustum()` in `kernels.cuh` is this
exact five-dot-product test.

### The 64-bit encoded atomicMin key

Extends project 01.18's uint-encoded-atomicMin z-buffer trick from 32 bits to 64. For any two
POSITIVE, FINITE `float`s `a < b`, their IEEE-754 bit patterns, reinterpreted as unsigned 32-bit
integers, satisfy `bits(a) < bits(b)` — the exponent occupies the high bits and dominates the
comparison, so integer ordering matches float ordering exactly (01.18 derives this; every range value
here is a physical distance, always `>= 0`, so the general negative-float case does not apply).
Packing `(bits(range) << 32) | point_index` into a 64-bit key means `atomicMin` over that key
resolves "smallest range, ties broken by smallest index" in ONE hardware-native atomic operation, with
**no lock, no critical section, and a result independent of thread execution order** — see "Numerical
considerations" below for why that independence is not a hand-wave.

## The algorithm

### Predicate -> scan -> scatter (the compaction template, six times)

Every one of this project's six kernels reduces to the SAME three-stage template:

```
1. PREDICATE:  flags[i] = keep(point[i]) ? 1 : 0        -- one thread per point, pure map
2. SCAN:       scan[i]  = exclusive_prefix_sum(flags)    -- THE primitive (see "The math")
3. SCATTER:    if flags[i]: out[scan[i]] = point[i]      -- one thread per point, pure map
```

- **Passthrough**: `keep(p) = z in [zmin, zmax]`.
- **Box ROI**: `keep(p) = p` inside an axis-aligned box.
- **Frustum crop**: `keep(p)` = the five-plane test above.
- **Fused**: `keep(p) = passthrough(p) AND box(p) AND frustum(p)`, one predicate kernel, one
  scan, one scatter.
- **Chained**: passthrough's OUTPUT feeds box as its INPUT, box's output feeds frustum — three
  independent runs of the whole three-stage template, each on a (shrinking) array.
- **Organized -> unorganized**: `keep(p) = NOT is_invalid_point(p)` (a NaN test) applied to the
  flattened organized grid — literally the same template with the simplest possible predicate.

**Why chained and fused reach the identical answer (a correctness invariant, not a coincidence):**
logical AND is associative and commutative, and each stage's compaction preserves relative order (the
identity above). Applying `A`, then `B` to `A`'s survivors, then `C` to `B`'s survivors keeps exactly
the points satisfying `A(p) AND B(p) AND C(p)`, in their ORIGINAL relative order — the same set, same
order, as one pass testing `A(p) AND B(p) AND C(p)` directly. `GATE fused_vs_chained` verifies this
holds in practice, not just on paper.

### Unorganized -> organized: scatter with collision resolution

The opposite direction cannot use predicate->scan->scatter, because it is not a FILTER — every input
point is kept, but MULTIPLE points can target the SAME output cell (a genuine `many-to-one` mapping,
unlike compaction's `subset, order-preserving` mapping). The algorithm is instead:

```
1. for every point i (in parallel): compute its (ring, azimuth) cell, race
   atomicMin(&cell_encoded[cell], pack_range_index(range(i), i))
2. for every cell c (in parallel): decode the winner (if any) from cell_encoded[c],
   copy its xyz, or write NaN if no point ever targeted this cell
```

`atomicMin` guarantees the SMALLEST encoded key survives regardless of which thread's atomic executed
last — the "nearest wins" policy this project's collision test measures directly
(`GATE collision_accounting`).

### Complexity

Every kernel here is `O(n)` work, `O(log n)` or `O(1)` depth (the scan is `O(log n)` depth; every
predicate/scatter/atomicMin pass is `O(1)` depth, one kernel launch, embarrassingly parallel). The
two-level scan composition (kernels.cu) extends a single block's `O(kScanElemsPerBlock)`-element
native capacity to an arbitrary `n` at the cost of one extra small kernel launch (scanning the block
sums) plus a broadcast pass — `O(n)` work overall, unchanged asymptotically.

## The GPU mapping

### The scan: two levels, three kernel launches

See `kernels.cu`'s "THE SCAN CHAPTER" comment block for the full walkthrough (up-sweep/down-sweep
diagrams, the two-level composition, and the bank-conflict discussion) — reproduced here at a higher
level for the reader who wants the shape before the detail:

```
Level 1 (many blocks, one per 512-element span):
    blelloch_block_scan_kernel  -->  local exclusive scan PER BLOCK + this block's TOTAL sum
Level 2 (ONE block, over the (small) array of block totals):
    blelloch_block_scan_kernel  -->  exclusive scan of the block totals (their GLOBAL offsets)
Combine:
    add_block_offsets_kernel    -->  add each block's global offset to its local scan values
```

**Memory hierarchy:** the up-sweep/down-sweep tree runs entirely in **shared memory**
(`__shared__ int temp[512]`, 2 KiB per block) — global memory is far too slow (hundreds of cycles of
latency) for the `O(log n)` rounds of tiny read-modify-write steps this algorithm performs; shared
memory's much lower latency and per-block locality is what makes an in-place tree scan on-chip
practical at all. The predicate and scatter kernels, by contrast, use NO shared memory — every
thread's work is fully independent (a pure map), so there is nothing to share.

**Bank conflicts (the honesty CLAUDE.md asks for, not silently optimized away):** shared memory has
32 banks; a warp's 32 threads accessing 32 DIFFERENT banks is one free transaction. This kernel's
up-sweep/down-sweep index formula `offset*(2*tid+1)-1` strides by `offset` (1, 2, 4, ..., 256 across
the 9 rounds) — once `offset >= 32`, every active thread in a warp lands on indices sharing the SAME
bank, serializing what should be one transaction into up to 32. The standard fix pads every index by
`idx + (idx >> 5)` (`GPU Gems 3` ch. 39) so no two threads ever share a bank at any stride; this
project deliberately does NOT implement that padding, trading a measurable-but-bounded slowdown (see
`[info] scan_scaling`'s timings) for a kernel body a learner can read in one sitting — CLAUDE.md §1's
"explain the faster version in comments" applies exactly here.

**Occupancy:** `kScanBlockThreads = 256` threads scan `kScanElemsPerBlock = 512` elements per block —
a good default: a multiple of the 32-thread warp, enough threads to hide the up-sweep/down-sweep's
`__syncthreads()` latency behind other warps' work, and a small enough shared-memory/register
footprint that many blocks fit per SM simultaneously on sm_75..sm_89.

**What Thrust computes and why it differs:** `thrust::exclusive_scan` dispatches to CUB's
`DeviceScan::ExclusiveSum`, a **single-pass "decoupled look-back" chained scan** — ONE kernel launch
whose blocks communicate their running prefix through global-memory flags, each block briefly polling
its LEFT NEIGHBOR rather than waiting through a separate reduce-then-broadcast pass. That is a
genuinely different (and, at scale, faster) algorithm from the three-launch approach taught here — see
"Where this sits in the real world" below.

### The frustum/box/passthrough predicates: pure map, no shared memory

Every predicate kernel reads exactly one point's `(x, y, z)` (3 coalesced floats — adjacent threads
read adjacent addresses, since `xyz` is interleaved) and writes exactly one `int` flag. No
inter-thread communication, no shared memory, no bank-conflict story — the simplest possible GPU
mapping, included here specifically as the CONTRAST to the scan above: not every kernel needs the
scan's machinery, and recognizing which ones don't is itself a lesson.

### Scatter-vs-compact duality (continuity with 02.01)

02.01's `hash_compact_kernel` performs a conceptually IDENTICAL "each source claims a dense output
slot" operation via an atomic counter (`atomicAdd` on `num_occupied`) rather than a scan. Both are
valid GPU compaction strategies with a real trade-off: an atomic counter is simpler to write but gives
NO ordering guarantee (the order survivors land in the output is whatever order their atomics happen
to execute in — acceptable for 02.01, where survivors are voxel centroids with no meaningful "order,"
but WRONG here, where order preservation is a stated correctness requirement). The scan-based approach
costs more code (the whole two-level composition) to buy a DETERMINISTIC, order-preserving mapping.
Choosing between them is a real design decision every GPU compaction problem faces.

## Numerical considerations

- **Predicate boundaries under FMA-adjacent rounding (02.03's precedent, cited):** every predicate
  here is an inequality (`>=`, `<=`) on quantities that are EXACT COPIES of loaded `float`s — no
  arithmetic happens before the comparison (unlike 02.03's ground-plane distance, which involves a
  dot product that CAN round differently between a fused and non-fused multiply-add). The frustum
  test IS an inequality on a computed expression (`fx*x + cx*z`), so it inherits 02.03's caveat: the
  GPU compiler may fuse `fx*x + cx*z` into a single FMA (one rounding step) while a differently-
  compiled CPU path might round twice — normally worth a documented ULP tolerance. This project
  avoids needing one anyway: `frustum_compact_cpu`/`is_in_frustum` are SHARED (single-sourced,
  `kernels.cuh`) between the GPU device transcription and the CPU oracle's arithmetic EXPRESSION
  (`kFx * cx + kCx * cz`, byte-for-byte identical operator sequence in both files), and both are
  compiled from the SAME source-level expression with the SAME operator ordering, so any FMA fusion
  the compiler chooses to apply, it applies (or does not apply) consistently to both — verified
  empirically by `GATE frustum_geometry`'s exact (not tolerance-margined) match on this project's
  hardware/toolchain. The **edge cohort**'s ±1e-4 boundary-straddling points are the adversarial test
  of this claim: if FMA fusion ever DID diverge between the two code paths, a boundary point would be
  the first place it would show up as a predicate disagreement, and it does not.
- **The organized<->unorganized encode-packing bit budget:** `pack_range_index` uses all 32 bits of a
  `float`'s bit pattern for range (no truncation — a full IEEE-754 float32, exact bit-for-bit) and all
  32 bits of `uint32_t` for the point index, fitting comfortably in a `uint64_t` with room to spare
  (a `uint32_t` index covers up to ~4.29 billion points — far beyond any array this project or a
  realistic single-scan LiDAR pipeline touches).
- **Why the collision winner is exact (min is order-independent; sums are not):** the minimum
  operator is commutative and associative — `min(a, b) = min(b, a)` and the minimum of a SET does not
  depend on visitation order. The GPU's `atomicMin` race visits points in an unpredictable,
  hardware-scheduled order; the CPU twin's running-minimum loop visits them in a fixed index order;
  both compute "the minimum (range, index) pair among every point that targeted this cell" — the same
  well-defined set operation, so they agree EXACTLY, with **no tolerance**. Contrast this with 02.01's
  Method A (an `atomicAdd`-accumulated voxel centroid): a SUM's floating-point result genuinely
  depends on accumulation order (rounding does not commute with addition the way it commutes with
  min/max), so THAT project needs a measured tolerance where THIS project needs none. Recognizing
  which GPU idiom you are using — and which claim it can and cannot make about determinism — is the
  numerical lesson this project and 02.01 make as a matched pair.
- **The azimuth-bin-edge trap (an honest, caught-in-development bug — see `data/README.md`'s "A note
  on the azimuth-bin-center fix"):** generating rays at the EXACT lower edge of each azimuth bin
  placed every angle precisely on `floor()`'s decision boundary; a sub-ULP `cos`/`sin`/`atan2`
  round-trip error flipped ~49% of reconstructed bins down by one. Casting at the bin CENTER instead
  buys a full half-bin-width (~0.003 rad) of margin — the general lesson: never place a value you will
  later `floor()`/`round()` reconstruct EXACTLY on the rounding boundary if you can instead place it
  at the boundary's midpoint for free.
- **Two-level scan's scaling limit:** the level-2 pass scans block sums in a single
  `kScanElemsPerBlock`-capacity block, so this implementation's exact ceiling is
  `kScanElemsPerBlock^2 = 262,144` elements; `launch_scan_blelloch` checks this at runtime and fails
  loudly (never silently truncates) if exceeded — see kernels.cu's comment for what a third level
  would require.

## How we verify correctness

Two tiers, per the repo's independence ruling (see `reference_cpu.cpp`'s file header for the full
statement specialized to this project):

1. **Twin comparison (GPU vs. CPU), bit-exact, no tolerance anywhere in this project.** Unlike most
   GPU-vs-CPU comparisons in this repository, EVERY comparison here is an exact integer/bit-exact
   float check, not a tolerance-margined one — because every quantity compared is either (a) an
   INTEGER (scan values, counts, collision bookkeeping) or (b) a float that is a verbatim COPY (never
   the result of GPU-vs-CPU-diverging arithmetic) from input to output. The CPU twins are
   algorithmically INDEPENDENT of the GPU's scan-based/atomicMin-based mechanisms (a plain serial
   filter loop for compaction, a plain running-minimum for the organized scatter) — see
   `reference_cpu.cpp`'s header for exactly what is shared (geometric formulas — a data-layout
   contract) versus independently structured (the algorithms).
2. **The edge cohort — an adversarial, analytically-placed test set,** not a fuzz test: 39 points
   sitting at ±1e-4 around every predicate's exact threshold, checked by `GATE predicate_correctness`
   / `GATE frustum_geometry` for correct inclusive/exclusive boundary behavior on BOTH sides of every
   threshold this project defines.
3. **Structural gates independent of any oracle:** `GATE order_preservation` checks the GPU's OWN
   output is strictly increasing in original index — a property that must hold regardless of whether
   the CPU twin is even correct, catching a class of bug (a scatter writing to the wrong slot) a
   count-only comparison could miss if it happened to produce the right COUNT by coincidence.
   `GATE collision_accounting`'s `valid_in == occupied + collisions` identity is checked from TWO
   independently-computed traversals (cell-space and point-space) per implementation (GPU and CPU
   separately), not asserted as a tautology.

Every project-specific tolerance question this project might have needed — FMA rounding in the
frustum predicate, accumulation order in the collision scatter — resolves to "none needed," and each
resolution is argued above, not assumed.

## Where this sits in the real world

- **PCL's `CropBox` and `PassThrough` filters** (`pcl::CropBoxFilter`, `pcl::PassThrough`) implement
  exactly the predicate half of this project's pipeline (single-threaded CPU, `pcl::ExtractIndices`
  underneath for the actual compaction) — the production tools this project's passthrough/box crop
  teach toward. PCL predates widespread GPU point-cloud tooling; GPU-accelerated equivalents live in
  vendor SDKs (NVIDIA's cuPCL, Isaac ROS) rather than upstream PCL itself.
- **Thrust/CUB are the production scan/compaction primitives.** `thrust::copy_if` /
  `cub::DeviceSelect::Flagged` implement predicate-based compaction directly (skipping the
  hand-written predicate-then-scan-then-scatter dance this project spells out for teaching) on top of
  the SAME decoupled-look-back single-pass scan this project's `launch_scan_thrust` calls into —
  production code reaches for `cub::DeviceSelect` directly rather than composing scan+scatter by hand.
- **`sensor_msgs/PointCloud2`'s `is_dense` field** is ROS 2's version of this project's
  organized/valid distinction: `is_dense = false` signals "this cloud may contain NaN/Inf points,"
  exactly the organized grid's invalid-cell convention — `PRACTICE.md` §3 discusses the honesty (and
  frequent dishonesty, in practice) of that flag in real driver output.
- **nvblox / Isaac ROS's `point_cloud_transport` and TensorRT-accelerated cropping nodes** are the
  closest production analogs of this project's GPU predicate kernels — camera-frustum-based LiDAR
  cropping for sensor fusion is a real, shipped pattern in NVIDIA's Isaac perception stack, not a
  pedagogical invention.
- **What full production pipelines do differently:** real LiDAR drivers apply vehicle-body /
  self-hit masking (PRACTICE.md §1) BEFORE any of this project's crops, often baked into firmware or
  a fixed per-unit calibration mask rather than a per-frame kernel; production frustum crops often
  fuse the crop directly into the camera-LiDAR COLORING kernel (project 02.17) rather than running as
  a separate pass, exactly the fusion lesson `GATE fused_vs_chained` teaches in miniature.
