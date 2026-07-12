# 02.03 — Ground segmentation: RANSAC plane fit; Patchwork++-style GPU port: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**What "ground" physically is to a robot.** Ground is the load-bearing surface a robot's wheels, feet,
or hull rest on and move across — the one thing every mobile robot must know the shape of before it
moves at all. Everything a robot does downstream of perception assumes an answer to "where can I be": a
wheeled robot's suspension model, a legged robot's foothold planner, a marine robot's draft clearance —
all consume some version of "ground" (or its analogue: seafloor, deck, terrain). Ground segmentation is
the perception module that turns a raw point cloud into that answer.

**Why LiDAR ground returns dominate the point budget — and why that is a mixed blessing.** A spinning
LiDAR emits beams at fixed elevation angles. For a beam at elevation `θ` (measured from horizontal,
negative = downward) from a sensor mounted at height `H` above a flat floor, the beam reaches the floor
at range

```
r(θ) = H / sin(|θ|)          (θ < 0, i.e. pointing down)
```

This has two consequences that matter for this project:

1. **Ground returns are geometrically privileged.** Unlike an obstacle, which only reflects a beam if it
   happens to be in that beam's path, the floor reflects *every* sufficiently-downward beam, at every
   azimuth — a flat floor is a huge, always-present target. This is why naive point-cloud processing
   (voxel downsampling, clustering, registration) is dominated by near-field ground returns unless it is
   removed first — the same 1/r² density story [02.01](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)
   derives for LiDAR returns generally applies doubly to ground, since ground is the one surface visible
   at *every* azimuth simultaneously.
2. **Ground coverage is a set of discrete RINGS, not a continuous sheet.** Because `r(θ)` is fixed per
   beam elevation, a flat floor produces concentric *rings* of returns — one ring per beam that reaches
   the floor within sensor range — with **gaps** between rings where no beam happens to intersect the
   floor at all. This project's own beam table (16 beams, −15°…+15° in 2° steps) has only 5 elevations
   steep enough to reach the floor within `MAX_RANGE_M = 16 m` (computed from `r(θ)` above: −15°→5.8 m,
   −13°→6.7 m, −11°→7.9 m, −9°→9.6 m, −7°→12.3 m; shallower beams overshoot the range limit and simply
   return nothing). This is not a simulation artifact — real 16-beam mechanical LiDARs have exactly this
   ring structure, which is *why* real Patchwork/Patchwork++ deployments use 32-, 64-, or 128-beam
   sensors for dense ground coverage. This project's own `scripts/make_synthetic.py` had to be tuned
   around this fact once (see its module docstring's "Why the canopy center sits at r=6.2 m" note) — a
   real, load-bearing lesson about sparse-beam LiDAR the hard way.

**The engineering constraint this project's ramp/plateau scene exercises.** Real ground is rarely one
plane: ramps, curbs, speed bumps, multi-level parking structures, and uneven terrain are the norm, not
the exception. A ground-segmentation algorithm's job is to track the *local* support surface even as its
orientation and height change — the central tension this project's two milestones make concrete.

## The math

**Notation.** A point `p = (x, y, z)ᵀ ∈ ℝ³`, meters, in the LiDAR sensor frame (origin at the sensor,
+x forward, +z up — CLAUDE.md §12). A plane is represented `(n, d)` with unit normal `n = (nₓ,n_y,n_z)ᵀ`
and offset `d` such that every point `p` on the plane satisfies `n·p + d = 0`. The **signed distance**
from an arbitrary point `p` to the plane is `dist(p) = n·p + d` (positive on the side `n` points toward).

**RANSAC's probabilistic guarantee.** Given a point set with a true inlier fraction `w` (fraction of all
points that truly lie on the target plane, within the inlier threshold), the probability that a single
random 3-point draw is *all inliers* is `w³` (three independent draws, sampled without replacement, but
`w³` is the standard approximation for `N ≫ 3`). The probability that **at least one** of `K` independent
draws is all-inliers is `1 - (1-w³)ᴷ`. Setting this to a target success probability `p` and solving for
`K`:

```
1 - (1-w³)^K = p
(1-w³)^K = 1-p
K·log(1-w³) = log(1-p)
K = log(1-p) / log(1-w³)              (*)
```

This project's `ransac_formula` gate computes `(*)` with the **measured** `w` from the flat-only run
(RANSAC's home turf, where the whole point of the formula check is meaningful) and `p = 0.999`, then
asserts `K_needed ≤ K = 1024` — an *analytic* check that the fixed hypothesis budget is actually enough,
not just an empirical "it worked this time."

**Least-squares plane fitting (why the smallest eigenvector).** Given a set of `m` points `{pᵢ}`, the
least-squares plane minimizes the sum of squared **perpendicular** distances:

```
minimize over (n,d), |n|=1:   Σᵢ (n·pᵢ + d)²
```

Differentiating with respect to `d` and setting to zero gives `d = -n·p̄` (`p̄` = the centroid) — the
optimal plane always passes through the centroid. Substituting back, the objective becomes
`Σᵢ (n·(pᵢ-p̄))² = n^T C n` where `C = (1/m) Σᵢ (pᵢ-p̄)(pᵢ-p̄)^T` is the 3×3 **covariance matrix** of the
points. Minimizing `n^T C n` subject to `|n|=1` is the classic Rayleigh-quotient problem: the minimizer
is the eigenvector of `C` with the **smallest eigenvalue** (the direction of *least* variance — exactly
"perpendicular to the flattest spread of points," which is what a plane normal should be). This is why
both milestones' plane fits — RANSAC's refinement and every CZM patch's fit — reduce to "build a 3×3
covariance matrix, find its smallest eigenvector" (`fit_plane_from_cov_accum` in `kernels.cuh`).

**The CZM's polar partition geometry.** The concentric-zone model partitions the (x,y) plane around the
sensor into `Z` **zones** by range `r = √(x²+y²)` (`kCzmZoneEdgesM = [0.5, 4, 8, 14, 20]` meters, 4
zones), each zone split radially into `R` **rings** (`kCzmRingsPerZone = 2`), each ring split
angularly into a **zone-specific** number of **sectors** (`kCzmZoneSectors = [32, 24, 16, 8]`, near to
far). The **why** for density-adaptive sector counts: LiDAR point density falls off as `1/r²`
([02.01](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)'s derivation) — a fixed
sector angular width `Δφ` covers an arc length `r·Δφ` that *grows* with range, but the point density
*shrinks* as `1/r²`, so the point *count* in a fixed-angle, fixed-radial-depth patch falls roughly as
`1/r` even before accounting for the radial depth also needing to grow to keep patch *area* comparable.
Finer sectors near, coarser sectors far, keeps each patch's point COUNT in a comparable range across the
whole scan (roughly 300–5,000 points/patch, measured on this project's committed scene) — too few points
starves the plane fit (`kCzmMinPatchPoints = 10` is the hard floor), too many wastes the fine local
resolution the whole method exists for.

## The algorithm

**Milestone 1 — RANSAC, step by step:**

1. **Generate K=1,024 hypotheses.** For each hypothesis index `k`, derive a starting RNG state from
   `(global_seed, k)` via a counter-based mix (no shared/sequential state — see "The GPU mapping"),
   draw 3 point indices, and fit the plane through them via the cross-product formula
   `n = (p₁-p₀) × (p₂-p₀)`, normalized (reject and retry up to 8 times if the cross product's magnitude
   is too small — a near-collinear/coincident triplet, whose normal direction is numerically
   meaningless).
2. **Evaluate all K hypotheses against all N points** — for each hypothesis, count points with
   `|n·p+d| ≤ threshold` (0.08 m). Serial cost: `O(K·N)`; this is the whole reason this step is worth
   parallelizing (see "The GPU mapping" below).
3. **Select the best hypothesis** — the one with the most inliers (ties broken toward the lowest index,
   a deterministic rule so GPU and CPU pick the same winner).
4. **Refine** — gather the winning hypothesis's inliers, accumulate their 3×3 covariance, take the
   smallest eigenvector (see "The math" above).
5. **Classify** every point against the refined plane at the same 0.08 m threshold — RANSAC's final
   ground/not-ground answer.

Complexity: step 1 is `O(K)`; step 2 is `O(K·N)` (the dominant term, `≈165M` evaluations on this
project's scene); steps 3–5 are `O(K)`, `O(N)`, `O(N)`. A serial CPU implementation pays the full
`O(K·N)` sequentially; the GPU parallelizes across both `K` and (within each hypothesis) `N`.

**Milestone 2 — the concentric-zone model, step by step:**

1. **Assign every point a patch id** from its `(zone, ring, sector)` via the polar formula above —
   `O(N)`, embarrassingly parallel (pure function of one point's `(x,y)`).
2. **Group points by patch** — sort by patch id (turns "which points share a patch" into "a contiguous
   run in a sorted array," `O(N log N)` via radix sort).
3. **Per column (a `(zone,sector)` pair), fit ring 0 then ring 1 IN ORDER:**
   a. **Seed selection.** Ring 0 (no prior information): seed = points within `[min_z, min_z + 0.20 m]`
      of the patch's own minimum height (a height-*margin* rule — see "Numerical considerations" for why
      this differs from Patchwork++'s actual percentile rule). Ring 1 (if ring 0 passed its tests): seed
      = points within `±0.30 m` of a height *predicted* by ring 0's fitted plane, evaluated at the seed
      centroid — this is the **region-growing** step: ring 1's search window tracks ring 0's plane's
      SLOPE, not just its height, so on the ramp, ring 1's window is already centered on the *higher*
      ground the ramp produces at that range, instead of blindly re-searching from a flat-ground
      assumption.
   b. **Fit** the seed points' plane (covariance + smallest eigenvector, as above).
   c. **Test uprightness** (angle between the fitted normal and vertical `≤ 30°` — chosen to admit the
      scene's 8° ramp with a wide margin while still rejecting near-vertical surfaces like a box's side
      face) **and flatness** (RMS residual of seed points to the fitted plane `≤ 6 cm` — rejects a patch
      whose "seed" is actually a mix of surfaces, e.g. straddling an obstacle's edge).
   d. **Classify** every point in the patch (not just its seed) against the fitted plane at `≤ 5 cm` —
      if the patch failed either test, every one of its points is classified non-ground (a conservative,
      safety-leaning default: an *uncertain* patch is not asserted to be ground).
4. Track a per-column height **carry**: after a ring passes, its fitted plane's height prediction at the
   seed centroid becomes the *next* ring's search center; a ring that fails leaves the carry unchanged
   (in a `kCzmRingsPerZone = 2` scene this only matters between ring 0 and ring 1, but the rule
   generalizes to more rings — see README "Exercises").

Complexity: step 1 is `O(N)`; step 2 is `O(N log N)` (dominated by the sort, though with only 161
distinct keys it runs close to radix sort's best case); step 3 is `O(patches × points-per-patch) = O(N)`
total work, but spread across only `kCzmNumColumns = 80` independent units of *coarse-grained*
parallelism (contrast with Milestone 1's 1,024-wide hypothesis parallelism — see "The GPU mapping").

## The GPU mapping

**RANSAC hypothesis generation** (`ransac_generate_hypotheses_kernel`): grid-stride over `K=1024`
hypotheses, one thread per hypothesis. Each thread's RNG state derives *only* from `(global_seed, its own
index, retry attempt)` — a **counter-based** stream (Salmon et al.'s term for exactly this pattern),
deliberately NOT a shared sequential stream with a cursor, because a shared cursor would need a
serializing atomic and defeat the entire point of parallel generation. This is the "generate K
independent random samples with K independent threads, no coordination" pattern that appears throughout
GPU Monte Carlo code (see also [08.01](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/README.md)'s
per-rollout noise streams for the same idea at a different scale).

**RANSAC evaluation** (`ransac_evaluate_hypotheses_kernel`) — **the K×N regime, the project's
GPU-parallelism centerpiece.** Mapped **one block per hypothesis** (`grid.x = K = 1024`, `block =
256`): every thread in hypothesis `k`'s block grid-strides over all `N` points, testing the inlier
threshold, then a shared-memory tree reduction folds the block's 256 partial counts into one integer.
The alternative mapping — thread-per-`(hypothesis, point-chunk)` with a *global* atomic accumulator per
hypothesis — trades this kernel's clean, atomic-free block-local reduction for finer-grained
parallelism; it would matter if `K` were small and `N` were enormous (too few blocks to fill the GPU).
Here it is the opposite: `K=1024` blocks already comfortably saturate an RTX 2080 SUPER's 46 SMs many
times over, so the simpler, atomic-free mapping wins on both clarity and performance for this problem
size — the same "farm the independent unit of work across blocks" framing
[08.01](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/README.md)'s MPPI
rollout kernel uses for its `K` independent trajectories.

**RANSAC refinement** (`ransac_accumulate_inliers_kernel` + `ransac_refine_kernel`) — a genuinely
different, SMALLER-scale mapping: one grid-stride kernel accumulates the winning hypothesis's inlier
covariance via `atomicAdd` into 9 global floats (order-nondeterministic — see "Numerical
considerations"), then a **deliberately serial `<<<1,1>>>` kernel** runs the eigensolver on that single
3×3 matrix. This K=1 "batched" solve is a teaching bridge to the CZM's genuinely-parallel version below.

**CZM patch assignment** (`czm_compute_patch_ids_kernel`) — a plain grid-stride **map**, one thread per
point (this project's simplest kernel: no shared memory, no reduction, purely a function of each
point's own `(x,y)`).

**CZM sort + boundary search** (`launch_czm_sort_and_index`) — `thrust::stable_sort_by_key` (a radix
sort under the hood — see `kernels.cu`'s comment for what it computes and why the library beats a
hand-rolled version here) turns "every point's patch id" into "patch ids sorted, points permuted along";
`thrust::lower_bound`'s **vectorized** form then finds all 161 patch-boundary indices in one
device-parallel call (161 independent binary searches, not 161 kernel launches).

**CZM fit + classify** (`czm_fit_and_classify_kernel`) — **the project's smaller-scale, patch-parallel
mapping, a deliberate contrast with RANSAC's evaluation kernel:** one block per **column**
(`kCzmNumColumns = 80` blocks, `block = 128` threads), each block processing its column's two rings
*sequentially* (a genuine data dependency — ring 1 may need ring 0's fitted plane) via four
collaborative passes (min-z, seed covariance, flatness residual, classification), each a standard
shared-memory tree reduction (`block_reduce_sum_dev` / `block_reduce_min_dev`). With only 80 blocks,
this launch does **not** saturate the GPU the way RANSAC's 1,024-block evaluation does — an honest,
named limit: CZM's parallelism is bounded by the number of *patches* (a scene-design choice), not by
`N`. A production system processing many scans per second would instead batch multiple SCANS' CZM fits
into one launch (grid.x = scans × columns) to keep the GPU fed; this project's demo runs one scan, so
that batching is left to README "Exercises."

**Why the min-z reduction runs unconditionally, even for empty patches:** every branch in
`czm_fit_and_classify_kernel` that contains a `__syncthreads()`-bearing call (the block reductions)
tests a value computed **identically by every thread in the block** (`npts`, loaded from the same
`patch_start[]` entries; `s_is_ground`, a shared broadcast) — never a per-thread-varying condition. CUDA
requires every thread in a block to reach the *same* `__syncthreads()` call; a block-uniform branch
guarantees that trivially, at the cost of a few wasted cycles on empty/undersized patches (cheap,
measured negligible next to the kernel's total runtime).

## Numerical considerations

**Precision.** Both milestones fit planes in **FP32** (matching the point cloud's native precision), but
every CPU twin accumulates covariance sums in **double** — the same "give the oracle more precision than
the thing under test" choice [02.01](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)'s
`hashmap_downsample_cpu` makes for its independent oracle.

**FMA contraction — the source of this project's "near-bit-exact, not exact" tolerances.** `kernels.cuh`'s
plain functions (`plane_from_triplet`, `czm_compute_patch_id`, ...) are LITERAL SOURCE TEXT shared with
their `__device__` transcriptions in `kernels.cu` — but nvcc's device compilation pass and cl.exe's host
pass are each free to fuse a `a*b + c*d` pattern into a fused-multiply-add (FMA) instruction
independently, producing results that differ by up to ~1 ULP (unit in the last place) even from
byte-identical source and byte-identical inputs. Measured on this project's committed scene: RANSAC
hypothesis planes differ by at most **0.0396°** in normal angle and **3×10⁻⁶ m** in offset between the
GPU and its CPU twin — geometrically meaningless, but enough to fail a literal `==` comparison on the
stored floats. `main.cu`'s `VERIFY(hypotheses_*)` gates therefore compare within a small angle/offset
tolerance (0.05°/1mm — margined roughly 25× and 300× over the measured maxima), not bit-exact —
learn this lesson once here; it recurs in every FP32 GPU/CPU comparison in this repository.

**Boundary-straddling points.** `VERIFY(ransac_eval_*)` and `VERIFY(patch_ids)` compare INTEGER
quantities (inlier counts; patch ids) derived from continuous thresholds/bin edges — a point whose
distance-to-plane or polar (r,azimuth) coordinate lands within float ULP of a decision boundary can be
classified on opposite sides by GPU vs. CPU rounding, flipping an integer result even though no "bug"
exists. Measured: 9/1024 hypotheses' inlier counts differ by ±1 point (out of ~40,000–160,000 points per
hypothesis) on the full-scene run; 2/161,836 points' patch assignment differs. Both are margined by
*count*, not eliminated — for a continuously-distributed random scene, a handful of exact-boundary
ties is the expected, honest outcome (the same empirical argument
[02.01](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)'s Method-B bit-exactness
claim rests on: no point is *adversarially* placed on a boundary).

**Determinism: RANSAC's atomics vs. the CZM's synchronized reductions.** `ransac_accumulate_inliers_kernel`
accumulates the winning hypothesis's covariance via `atomicAdd` — the summation ORDER depends on GPU
thread scheduling and is **not** run-to-run deterministic (mirroring
[02.01](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)'s Method A exactly, and for
the same reason: this step runs once per RANSAC call, not thousands of times, so the engineering trade
favors atomics' simplicity over a sort-based fixed-order reduction). By contrast, `czm_fit_and_classify_kernel`
uses ONLY shared-memory tree reductions (`block_reduce_sum_dev`/`block_reduce_min_dev`) — no atomics
anywhere — so CZM's output is **bit-for-bit reproducible run to run** on fixed input (verified: 5
consecutive clean-build runs of the committed scene produced byte-identical gate numbers). This
determinism was load-bearing during this project's own development: an early, buggy version of a debug
instrumentation pass masked what should have been a clean, deterministic result, and re-running from a
clean rebuild is what confirmed the kernel itself has no race.

**Covariance conditioning on thin patches.** The one-pass covariance formula
`C = E[pp^T] - p̄p̄^T` (used by `fit_plane_from_cov_accum`) is numerically less stable than a two-pass
(subtract the mean first, then accumulate) formula when a patch's points are tightly clustered far from
the origin — the classic "catastrophic cancellation" risk. This project's patches sit at ranges up to
20 m with sub-meter spread, so the risk is real but small (measured: the direct PCA fit to the ramp's
true ground points recovers the 8° slope to within **0.0003°** — see the `slope_accuracy` `[info]` line);
a production system fitting patches far from the sensor origin (e.g., in a world frame instead of a
sensor-relative frame) should use the two-pass formula instead — see [33.01](../../33-foundational-libraries/33.01-batched-small-matrix-linalg/README.md)
for that technique's general treatment.

**Degenerate triplets.** A 3-point plane fit is undefined when the 3 points are collinear (the cross
product vanishes). `plane_from_triplet` rejects a triplet whose cross-product magnitude falls below
`kRansacMinCrossNormM2` and retries (up to 8 attempts per hypothesis) — measured on this scene: 0
hypotheses ever exhaust all 8 attempts (the point cloud is dense and well-spread enough that a
degenerate draw is rare and a retry almost always succeeds).

## How we verify correctness

Every GPU computation in this project has an independent CPU oracle in `reference_cpu.cpp`, following
one of two rulings (`kernels.cuh`'s file header states which applies to each):

- **Shared-formula, drift-detecting (near-bit-exact):** `ransac_generate_hypotheses_cpu` and
  `czm_compute_patch_ids_cpu` call `kernels.cuh`'s own plain functions directly — the SAME formula the
  GPU's `__device__` code transcribes. `VERIFY(hypotheses_*)` and `VERIFY(patch_ids)` exist to catch
  DRIFT between the header and its device transcription (a typo in either copy would show up as a large,
  systematic mismatch — not the small, ULP-scale differences these gates actually tolerate).
- **Independent, tolerance-compared:** `ransac_evaluate_hypotheses_cpu` (its own nested loop, no
  shared-memory reduction), `ransac_refine_cpu` (double precision, sequential order, no atomics — a
  genuinely different computation from the GPU's float/atomic path), and `czm_fit_and_classify_cpu`
  (a `std::vector<std::vector<int>>` per-patch bucket list — a different DATA STRUCTURE than the GPU's
  Thrust-sorted contiguous array — double-precision sequential accumulation, its own Jacobi
  eigensolver call). `VERIFY(ransac_eval_*)`, `VERIFY(ransac_refine_*)`, and `VERIFY(czm_fit)` compare
  these against the GPU's output within measured-then-margined tolerances (CLAUDE.md §12).

**Ground-truth gates (`main.cu`'s 6 `GATE ...:` lines)** are a *third*, independent layer: they do not
compare GPU against CPU at all, but compare each milestone's **classification output** against the
scene's exact, generator-known ground truth (`h_ground_label`/`h_zone_id`, loaded straight from
`data/sample/ground_scan.bin`). This is what proves the *algorithms* work, not just that the GPU agrees
with the CPU — a subtle but important distinction: two implementations could agree with each other while
both being wrong relative to physical ground truth, which is exactly what the ground-truth gates catch
that the VERIFY gates cannot.

## Where this sits in the real world

**RANSAC in production** looks much like this project's Milestone 1 — PCL's `SACSegmentation` with a
`SACMODEL_PLANE` model is the textbook version most robotics engineers meet first, and Autoware's older
`ray_ground_filter` uses a scan-line RANSAC variant. Production systems add: multiple sequential RANSAC
passes (remove the dominant plane, re-run for a second one — handles a simple two-level case without a
full CZM), robust loss functions beyond a hard inlier/outlier threshold (MSAC, MLESAC), and often a
Kalman-filtered plane estimate across frames instead of a fresh fit every scan.

**Patchwork++ in production** (Lee, Jung, Yoon, Kim, *"Patchwork++: Fast and Robust Ground Segmentation
Solving Partial Under-Segmentation Using 3D Point Cloud,"* IEEE RA-L 2022 — the real system this
project's Milestone 2 ports at reduced scope) adds, beyond this teaching version:

- **Adaptive Ground Likelihood Estimation (A-GLE)** — a per-patch Gaussian model over (flatness,
  elevation, uprightness) fit online, replacing this project's fixed thresholds
  (`kCzmUprightMaxDeg`/`kCzmFlatnessMaxRmsM`) with data-driven ones.
- **Reflected Noise Removal (RNR)** — LiDAR returns below the true ground plane (multi-path reflections
  off wet/reflective surfaces) get explicitly detected and discarded before seed selection; this project
  has no wet-surface model (see PRACTICE.md "rain-wet reflections").
- **Temporal Ground Revert (TGR)** — a patch's "ground" verdict from the *previous* frame biases the
  current frame's decision, smoothing flicker on genuinely ambiguous patches (this project processes one
  static scan; README "Exercises" sketches adding this).
- **Percentile-based seed selection** — the paper selects the lowest N% of a patch's points by height,
  not this project's fixed height-margin band; the paper's choice adapts better to very sparse patches.

**The original Patchwork** (Lim, Oh, Myung, IEEE RA-L 2021) is cited specifically for the concentric-zone
model's zone/ring/sector partition — this project's `kCzmZoneEdgesM`/`kCzmZoneSectors` follow that
paper's density-adaptive-sector idea, though the exact numeric configuration here is this project's own
(tuned to its own beam table and scene scale, not copied from the paper's LiDAR/scene assumptions).
