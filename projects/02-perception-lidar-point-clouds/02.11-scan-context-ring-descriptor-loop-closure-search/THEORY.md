# 02.11 — Scan Context / ring-descriptor loop-closure search: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

A mobile robot estimates its own motion by integrating noisy measurements — wheel odometry, IMU
preintegration, scan-to-scan ICP. Every one of those estimates carries a small error, and small errors
**accumulate**: after driving a kilometer of corridors, a robot's belief about its own position can be
meters away from the truth, even though every individual motion estimate was locally excellent. This is
not a bug to be fixed by a better sensor — it is a consequence of dead reckoning being a *sum of
approximations*, and the sum's variance grows without bound as long as the robot only ever looks at
where it *just* was.

The fix requires a fundamentally different kind of measurement: recognizing that the robot's CURRENT
surroundings match somewhere it has ALREADY mapped, independent of the accumulated drift in its pose
estimate. This is **place recognition**, and when it closes a "loop" in the trajectory graph — connecting
today's keyframe back to one from minutes or hours ago — it is called **loop closure**. Geometrically, a
detected loop closure is a global positional constraint that a pose-graph optimizer can use to
redistribute the accumulated drift across the *whole* trajectory, snapping a bent corridor straight
again. It is one of the few mechanisms in SLAM that can correct errors *after* they have already grown
large, rather than only slowing their growth.

**Why this is a genuinely hard perception problem, physically.** A LiDAR return is a noisy, partial,
viewpoint-dependent sample of 3-D structure: the SAME physical corner, revisited from a different heading
or a parked delivery cart's width to one side, produces a POINT CLOUD that shares almost no point with the
first visit (different beam angles hit different surface patches) even though the underlying geometry is
identical. Any place-recognition descriptor must therefore summarize a scan by something more stable than
raw points — a statistic robust to the sensor's own viewpoint. Scan Context's answer: describe a place by
the **envelope of its persistent vertical structure** (the tallest thing in each direction, at each
range) rather than by the specific points a single sweep happened to hit. A parked car, a pedestrian, or
a gust of dust in front of a wall is TRANSIENT — it may or may not be there on the next visit — while the
wall's own height, at that range and bearing, is a property of the PLACE, not of the moment. Taking the
MAX height per cell (rather than, say, the mean or the point count) is a deliberate bet that the
persistent structure usually wins the max, because transient clutter is rarely TALLER than the
architecture behind it. This is the physical/engineering reasoning "the math" below turns into a formula.

**Engineering constraints a real robot imposes:** loop closure runs on live keyframes at a modest but
nonzero rate (README "System context" — roughly once per keyframe, ~1–2 Hz), so the search cost per
query must stay small even as the database of past keyframes grows into the thousands over a long
mission; a single false positive corrupts the map (PRACTICE.md §1), so precision matters more than
raw recall; and the sensor's own mounting height, beam pattern, and max range all leave fingerprints in
what "the same place" even looks like — this project's own synthetic sensor model (16 channels, -18° to
+12° elevation, 40 m max range) is not incidental, it directly shapes which cells of the descriptor ever
get real data (see "numerical considerations" below).

## The math

**Coordinate setup.** A keyframe scan is a set of points `p_i = (x_i, y_i, z_i)` in the SENSOR FRAME
(origin at the sensor, +x forward along the robot's current heading, +y left, +z up —
`docs/SYSTEM_DESIGN.md` §3.2's body convention). Define, for point `i`:

```
range_i  = sqrt(x_i^2 + y_i^2)        (planar/horizontal range from the sensor)
azimuth_i = atan2(y_i, x_i)  folded into [0, 2*pi)
```

**The descriptor.** Partition range into `kNumRing = 20` equal rings over `[0, kSensorMaxRangeM)` and
azimuth into `kNumSector = 60` equal sectors over `[0, 2*pi)`. The Scan Context matrix is

```
SC[r][s] = max { z_i : point i falls in ring r, sector s },   r in [0,20), s in [0,60)
```

(`kEmptyZ`, a sentinel far below any physically reachable height, if no point ever lands in that cell —
see "numerical considerations" for why this exact choice matters).

**Rotation becomes a column shift — the derivation.** Suppose the robot revisits the same physical place
but with heading rotated by `delta` (`delta` in world-frame yaw). A world point that appeared at
sensor-relative azimuth `az` on the first visit appears at sensor-relative azimuth `az - delta` on the
second (its WORLD position is unchanged; only the sensor's own reference direction rotated by `delta`).
Every point's RANGE is unchanged by a rotation of the sensor about its own origin. So: the ring a point
falls in is unchanged, and the sector shifts by a constant amount for every point simultaneously —
`floor((az - delta) / sector_width) = floor(az/sector_width) - shift` for the integer
`shift = round(delta / sector_width)` (exact when `delta` is a multiple of the sector width, a close
approximation otherwise — the descriptor's fundamental yaw QUANTIZATION, `360/60 = 6` degrees). Since
every point's ring is unchanged and every point's sector shifts by the SAME `shift`, the whole matrix
just has its columns cyclically permuted:

```
SC_rotated[r][s] = SC_original[r][(s + shift) mod 60]
```

This is the entire trick: **comparing two matrices under every possible column shift and taking the best
one searches over relative yaw for free**, rather than requiring yaw to already be known (contrast with
ICP, which needs a good initial guess).

**The distance metric.** For two matrices `A` (query) and `B` (candidate) and a trial shift `k`, define
the per-column cosine distance and the matrix distance as its mean over sectors:

```
col_dist(s, k) = 1 - cos_angle( A[:, s], B[:, (s+k) mod 60] )       (0 = identical direction, 1 = orthogonal, 2 = opposite)
dist(A, B, k)  = (1/60) * sum_{s=0}^{59} col_dist(s, k)
D(A, B)        = min_{k=0}^{59} dist(A, B, k)
shift*         = argmin_k dist(A, B, k)      <-- the free yaw estimate: shift* * 6 degrees
```

`D(A,B)` close to 0 means "very likely the same place"; `shift*` is the relative-yaw estimate a
downstream aligner can start from instead of identity (README's `yaw_handoff` illustration).

**The ring key.** `RK[r] = (# non-empty sectors in ring r) / 60`, a 20-dimensional vector in `[0,1]^20`.
Because every column's occupancy is unaffected by which shift is applied (a shift permutes WHICH sector
holds a value, never WHETHER a given ring has one), `RK` is EXACTLY rotation-invariant — no shift search
needed to compare it. Comparing two ring keys by L1 distance is a coarse, cheap "how much vertical
structure surrounds me, at each range" fingerprint, used to rank the whole database and keep only the
closest `kRingKeyPrefilterBudget` candidates for the expensive `D(A,B)` search — a projection from
1200 numbers down to 20, discarding all azimuthal detail on purpose for speed.

## The algorithm

1. **Build** (once per new keyframe): for every point, compute `(ring, sector)` and scatter its height
   into the running max of that cell — O(N) in point count, embarrassingly parallel across points.
2. **Ring key** (once per new keyframe): for every ring, count non-empty sectors — O(20 x 60) per scan,
   trivial.
3. **Prefilter** (once per query, against the whole database): L1 distance from the query's ring key to
   every valid (temporally-gapped) candidate's ring key, keep the closest `kRingKeyPrefilterBudget` —
   O(database_size x 20).
4. **Search** (once per query, against the prefiltered candidates only): for every candidate and every
   of 60 shifts, the mean column-cosine distance — O(budget x 60 x 60) = O(budget x 3600), the
   project's dominant cost, and exactly why step 3 exists: without it this is O(database_size x 3600),
   growing without bound as the mission gets longer.
5. **Decide**: if the minimum distance found is below `kScDistanceThreshold`, declare a loop closure at
   the argmin candidate, with `shift*` as the yaw estimate; hand off to geometric verification
   (README "System context").

**Complexity, serial vs. parallel.** Step 1 is O(N) serial (N ~ 1000–1600 points/scan here) — GPU:
O(N / P) with P threads, one atomic max per point. Step 4 is O(budget x 3600) serial per query — GPU:
O(60 rings-worth-of-work) per (candidate, shift) BLOCK, with `budget x 60` blocks running concurrently,
so wall-clock scales with the SLOWEST block (60 sequential ring reads + a `log2(64)=6`-round reduction)
rather than with the total work — this is why the measured sweep (README "Expected output") stays under
a millisecond per query even as the candidate set grows into the hundreds.

## The GPU mapping

**Kernel 1 — `sc_build_kernel` (the scatter).** One thread per POINT, across every scan in the batch at
once (`total_points` threads total — main.cu launches this ONCE for all 160 committed keyframes, not
once per keyframe, because there is no dependency between scans at build time). Each thread: reads its
own `(x,y,z)` (coalesced — thread `i` reads `xyz[3i..3i+2]`, consecutive threads read consecutive
memory), computes `(ring, sector)`, and calls `atomicMaxFloat` on that cell of `sc_all`. The WRITE
address is data-dependent — a genuine scatter, not a coalesced write — and that is the correct GPU
pattern here: many independent producers (points), one shared accumulator per bucket (cells), the same
shape as a histogram or a spatial hash insert (02.01's voxel grid, cited). `atomicMaxFloat` itself is
built from `atomicCAS` in a compare-and-retry loop (CUDA has no native `atomicMax` for `float` — every
CUDA program that needs one builds it this way), directly generalizing this repo's 01.06
`atomicMin64`/`atomicMax64` CAS-loop idiom from a 64-bit integer key to a 32-bit float value: read the
current bit pattern, reinterpret it as `float` for the actual comparison (never for ORDERING the raw
bits — the alternative order-preserving-uint-encoding trick works but hides the comparison being made;
this project prefers the direct, honest compare), retry only if another thread's write raced ahead.

**Kernel 2 — `ring_key_kernel` (the small reduce).** One thread per `(scan, ring)` pair — small enough
(`n_scans x 20` = 3200 threads for the committed sample) that a 60-iteration sequential inner loop per
thread, no block cooperation, is the right level of engineering; a fancier reduction would add code
without adding speed at this scale.

**Kernel 3 — `sc_shift_distance_kernel` (candidate x shift, the project's named GPU mapping).** Grid
`dim3(num_candidates, 60)` — one BLOCK per `(candidate, shift)` pair. Block size 64 threads (`kNumSector`
padded to the next warp multiple; the 4 extra lanes are guarded off). Thread `t < 60` owns SECTOR `t` of
the QUERY matrix and computes `col_dist(t, shift)` against the CANDIDATE's shifted column; a
shared-memory tree reduction (64 → 32 → … → 1, a clean power of two, no ragged tail) sums the 60 real
contributions and thread 0 writes the block's one output float.

*Memory coalescing, worked example.* `col_dist` loops over `r = 0..19`. At a fixed `r`, thread `t`
reads `sc_query[r*60 + t]` — across the 64 threads of the block that is the CONTIGUOUS span
`sc_query[r*60 .. r*60+59]`, one coalesced 128-byte-class transaction per ring, 20 transactions total for
the query side. The CANDIDATE read at the same `r` is `sc_cand[r*60 + (t+shift) mod 60]` — a
WARP-LEVEL PERMUTATION of the exact same 60-float span (every lane reads a DIFFERENT offset within it,
just not offset `t`). Hardware coalescing keys off the SET of addresses a warp touches in one
transaction, not off which lane reads which byte within that set — so the wrap-around shift costs
NOTHING extra: the candidate side is exactly as coalesced as the query side. This is precisely why
`kNumSector` was chosen as the FAST-VARYING (contiguous) index of the Scan Context matrix layout in
`kernels.cuh` — the layout decision exists *for* this kernel, the same "pick the layout the hot kernel
wants" lesson 08.01's transposed noise array teaches (cited in `kernels.cuh`).

*Why candidate x shift, not candidate-only with a per-thread shift loop?* An alternative mapping — one
thread per candidate, looping over 60 shifts and 20 rings internally — would need only `num_candidates`
threads (worse occupancy on any modern GPU, which wants thousands of resident threads) and would forgo
the block-level SHARED read of each ring's 60-float span across the 60 shift-values that all read it
(each shift would independently re-read the same query row from global memory, 60 times over). The
chosen mapping trades a bit more code (the reduction) for both better occupancy and a genuinely smaller
memory-traffic footprint per useful FLOP.

## Numerical considerations

**Precision.** Every kernel here is FP32 throughout — Scan Context's cosine-distance arithmetic has no
numerically stiff step (no matrix inversion, no small-angle expansion), so FP32 is the natural choice
and no project in this repo's lineage uses FP64 for it.

**Determinism and the scatter-max.** `max` is ORDER-INDEPENDENT: however many threads race to update the
same cell, in whatever interleaving the scheduler chooses, the final value is exactly
`max(every value ever attempted)` — no rounding drift, no partial-sum artifact (contrast with a running
SUM, whose float rounding genuinely depends on evaluation order). This is what lets
`VERIFY(scan_context)` (main.cu) compare the GPU-built matrix against a plain sequential CPU max and
expect near-EXACT agreement — 99.74% of the committed sample's 192,000 cells match bit-for-bit; the
remaining ~0.26% are RING/SECTOR BINNING ties (not max-value disagreements): `atan2f`/`sqrtf` (device) and
`std::atan2`/`std::sqrt` (host) are not guaranteed bit-identical, so a point whose true azimuth or range
sits within a few ULP of a bin boundary can occasionally land one bin apart between the two paths. The
tolerance the VERIFY gate uses (§9's structural check) absorbs exactly this, and no more.

**The shift-distance reduction.** The GPU's block-level TREE reduction and the CPU oracle's plain
SEQUENTIAL sum combine the same 60 numbers in a DIFFERENT order — floating-point addition is not
associative, so the two sums can differ by a few ULP even though every underlying `col_dist` value is
identical. `VERIFY(shift_distance)` therefore uses a small ABSOLUTE tolerance (`2e-4`) rather than exact
equality — the honest way to compare a reduction across two different summation orders (the same
reasoning 08.01/02.10 apply to their own reductions, cited).

**Two real bugs this project's own development caught — worth re-deriving, not hiding.**

1. *The empty-cell sentinel had to be the seed of the running MAX, not a value the max could be beaten
   by.* An early version of this file used `0.0f` as the "no point here" sentinel, reasoning that a real
   return is rarely near exactly zero. That reasoning was correct but irrelevant: this project's sensor
   sits `kSensorHeightM = 1.6` m above the ground it rides on, so a GROUND return reads sensor-frame
   `z ≈ -1.6` — a legitimate, informative NEGATIVE height — and `-1.6` never beats a `0.0f` SEED under a
   running max. The bug silently discarded almost every ground-only cell (kept only cells a building
   wall also touched), turning a descriptor meant to summarize "how tall is the structure here" into one
   that mostly recorded "is there a building directly here or not". The fix: `kEmptyZ = -1000.0f`, the
   original Scan Context paper's own convention — a value no physically reachable sensor-frame height
   can ever exceed. General lesson: a running-max/min sentinel must be a genuine BOUND on the value
   range, not merely "an unlikely value".
2. *Once `kEmptyZ` is a large negative number, it must never enter the cosine-distance arithmetic
   directly.* A column with one real 0.3 m reading and nineteen `-1000.0f` "empty" rings would have its
   cosine GEOMETRY dominated by the nineteen sentinels, not the one real number. `column_cosine_distance`
   therefore MASKS every cell first (empty → contributes `0.0` to dot/norm; a separate `any_real` flag
   tracks whether the column had data at all) before doing any arithmetic — and the masking uncovered a
   THIRD, more subtle issue: a column empty in BOTH scans is AGREEING evidence ("nothing out there, in
   both visits") and must score distance `0.0`, not the maximum `1.0` an earlier version of this rule
   gave it. That earlier rule inflated the distance between two byte-for-byte IDENTICAL revisits by
   roughly one point of distance per column BOTH scans happened to leave empty — on this project's own
   committed sample, that pushed genuine same-place revisits (which should score ~0) up past 0.6,
   comfortably above any sane detection threshold. `kernels.cuh`'s `column_cosine_distance` documents the
   final three-way rule (both empty → 0.0, one-sided empty → 1.0, both real → cosine distance) in full.
3. *A keyframe-sampling asymmetry, not aliasing, was the first suspect for poor rotated-cohort
   recall.* Segments were originally sampled into `n = round(length/spacing)` keyframes at
   `t = (k+0.5)/n`; the segment's "anchor" (its representative keyframe for curated ground truth) used
   `k = n//2`. For EVEN `n` this lands at `t = 0.625`, not the true midpoint `t = 0.5` — harmless on its
   own, but it breaks the ONE property a forward pass and a reversed pass of the SAME physical edge need
   to share: `position_forward(0.5) == position_reverse(0.5)` (both equal `0.5*A + 0.5*B`) holds ONLY at
   `t=0.5`; at `t=0.625` the two anchors sit on OPPOSITE sides of the true midpoint, roughly 6 m apart on
   a 25 m block. `scripts/make_synthetic.py`'s `sample_keyframes()` now forces every segment's sample
   count to the next ODD integer, so `k = n//2 = (n-1)/2` lands EXACTLY at `t=0.5` for every direction.
   General lesson: a "curated ground-truth generator" is itself part of the system under test — its own
   subtle bugs manifest as the SAME symptom (poor measured recall) as a bug in the algorithm being
   evaluated, and distinguishing the two took exactly the kind of diagnostic sweep §"how we verify
   correctness" below describes.

**The yaw sign convention.** `shift* * (360/60)` degrees is compared against the synthetic world's known
TRUE relative yaw two ways (`+shift` and `-shift`, `main.cu`'s `rotation_invariance` gate) and the smaller
error is reported — resolved EMPIRICALLY against ground truth rather than re-derived from the ray-casting
convention by hand, and found to need no correction on the committed sample (mean error 0.0°, i.e. the
`+shift` convention is already correct given this project's azimuth-folding convention
`world_az = heading + sensor_az`).

## How we verify correctness

**Twin-vs-shared ruling for this project** (the template's general ruling, `docs/PROJECT_TEMPLATE/src/reference_cpu.cpp`,
resolved here): the small, deterministic, formulaic pieces — `ring_index_from_range`,
`sector_index_from_xy`, `column_cosine_distance`, `ring_key_l1_distance` — are SHARED, declared plain
`inline` C++ in `kernels.cuh` and called DIRECTLY by `reference_cpu.cpp` (duplicating a four-line index
formula would be pure transcription with no independence value); `kernels.cu` carries its OWN literal
`__device__` copy of each (required because a plain `inline` function compiled by nvcc is host-only —
02.10's identical precedent, cited). The AGGREGATION LOOPS — the parallel atomic scatter (`kernels.cu`)
vs. the sequential compare-and-replace (`reference_cpu.cpp`); the block-level tree reduction vs. the
plain sequential sum — are INDEPENDENTLY reimplemented with no structural resemblance, which is exactly
where a real GPU-only bug (wrong thread-to-cell mapping, a race, a reduction-order bug) would surface as
a GPU-vs-CPU mismatch.

**Tier 1 — GPU-vs-CPU VERIFY gates** (does the parallel implementation match the sequential reference?):
`VERIFY(scan_context)` (near-exact, scatter-max is order-independent), `VERIFY(ring_key)` (a cascade of
the above, looser floor), `VERIFY(shift_distance)` (small absolute tolerance, reduction-order only —
both paths are fed the SAME upstream matrices deliberately, so this stage does not conflate its own
tolerance with `VERIFY(scan_context)`'s already-measured disagreement).

**Tier 2 — independent GATEs against GROUND TRUTH** (does the ALGORITHM produce the right ANSWER, a
question twin agreement cannot address — the repo's independence ruling explicitly requires this second
tier): `loop_detection`, `rotation_invariance`, `lateral_sensitivity`, `negative_cohort`,
`ringkey_prefilter` — every one scored against the synthetic world's KNOWN true poses and curated revisit
labels (`data/sample/loop_pairs.csv`), never against the GPU/CPU peer. The `yaw_handoff` illustration
adds a THIRD, orthogonal check: does the recovered yaw actually help a downstream aligner (compact ICP)
converge, independent of whether the distance/threshold machinery agrees with anything at all.

**The operating threshold, chosen with margin, not curve-fitting.** `main.cu`'s diagnostic development
run (documented, not fabricated) found every genuine same-place revisit in the committed sample scoring
`<= 0.03` and the closest genuine aliasing confound (two different, structurally similar streets)
scoring `>= 0.13`; `kScDistanceThreshold = 0.10` sits deliberately in the MIDDLE of that gap (CLAUDE.md
§12: "success thresholds carry wide margins"), not immediately below the nearest negative example — a
threshold chosen by hugging one boundary would be one GPU-architecture's worth of ULP drift away from
flipping a verdict; this one has roughly 0.07–0.09 of headroom on each side, several orders of magnitude
larger than any plausible cross-architecture transcendental-function disagreement (the ~0.26% binning-tie
rate above, propagated through a 1200-cell matrix and a 60-column mean, moves the aggregate distance by
an amount far smaller than that margin).

## Where this sits in the real world

**Scan Context / SC++ (Kim & Kim, and later Kim, Choi & Kim, "Scan Context++", T-RO 2021)** is the
lineage this project reimplements didactically — the original paper (and its widely-used open-source
`irapkaist/scancontext` release) is the direct ancestor of every design choice named above; SC++ adds a
polar-context VARIANT descriptor and a two-phase (translation-then-rotation) search that handles large
lateral displacement better than the single-shift search this project implements, precisely the honest
limitation the `lateral_sensitivity` gate measures rather than papering over.

**OverlapNet / LCDNet / learned place recognition** replace the hand-built ring x sector statistic with a
CNN trained end-to-end on range images or raw point clouds, typically outperforming hand-crafted
descriptors on the hardest cases (extreme lateral offset, seasonal/lighting change for camera-based
variants) at the cost of needing training data and losing the "read the code, understand exactly what it
computes" property this repo values (CLAUDE.md §1). A production SLAM stack today often runs BOTH: a
cheap geometric prefilter (this project's shape) ahead of a learned re-ranker.

**The bag-of-visual-words alternative (DBoW2, ORB-SLAM's own loop detector)** solves the same problem
from IMAGES instead of point clouds: quantize local image features (ORB, the same family
[01.04](../../01-perception-cameras-vision/01.04-feature-pipeline/README.md)'s feature pipeline builds)
into a visual vocabulary, and compare scans by the TF-IDF-weighted overlap of their visual words — the
same "compress a scan into a compact signature, compare signatures cheaply" shape as Scan Context, one
level up the sensing stack. Real autonomous-vehicle and warehouse-AMR stacks frequently run a LiDAR
geometric detector (this project's family) and a camera visual-word detector (DBoW2's family) in
parallel and fuse their candidates, since the two fail on largely uncorrelated scene types.
