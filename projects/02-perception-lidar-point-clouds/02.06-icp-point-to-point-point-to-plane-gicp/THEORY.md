# 02.06 — ICP: point-to-point → point-to-plane → GICP, all batched: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**What a LiDAR return physically is.** A spinning or solid-state LiDAR emits a laser pulse (or a
continuously modulated beam) and measures the time until reflected light returns: `range = c * t_flight
/ 2`, where `c ≈ 3×10^8 m/s` is the speed of light and the factor of 2 accounts for the round trip. A
single "point" in a point cloud is therefore not a mathematical point at all — it is the estimated
range along one specific beam direction at one specific instant, converted to Cartesian `(x,y,z)` via
that beam's known angle. Three physical effects turn that estimate into the noisy, structured data this
project's ICP has to align:

- **Range/timing noise.** The time-of-flight measurement has finite precision — photodetector shot
  noise, timing-circuit jitter, and (for solid-state/FMCW sensors) phase-noise in the modulated signal
  all contribute. Commodity automotive/robotics LiDAR typically quotes **1σ range noise of 1–3 cm** at
  moderate range; short-range or higher-grade units can reach a few millimeters. This project's
  synthetic noise (`NOISE_SIGMA_M = 0.005` m, i.e. 5 mm — see `scripts/make_synthetic.py`) sits at the
  better end of that real-world range, chosen so the ICP problem is meaningfully noisy without being so
  noisy that a beginner's first read of the convergence curve is dominated by noise rather than by the
  point-to-point-vs-point-to-plane comparison this project teaches.
- **Beam divergence.** A laser beam is not infinitesimally thin — it diverges by a small angle (often
  quoted in milliradians) as it travels, so at range `R` the illuminated spot has diameter roughly
  `R * divergence_angle`. On a surface with texture or an edge inside that spot, the returned range is
  some function (often the nearest-surface or a power-weighted blend) of everything the spot touched —
  the physical origin of "mixed pixels" at depth discontinuities (e.g., a point that lands *between* a
  foreground box edge and the wall behind it). This project's synthetic scene has no beam-divergence
  model — every point is sampled exactly on a primitive's surface — a documented simplification (see
  [Limitations & honesty](README.md#limitations--honesty)).
- **Angular/timestamp precision and motion distortion.** A spinning LiDAR sweeps its beam across ~100 ms
  per revolution while the vehicle carrying it keeps moving, so raw points within one sweep are each
  valid at a *slightly different pose* — "motion distortion" that must be corrected (deskewed) before
  registration is meaningful. This project's synthetic clouds have **no** motion distortion (every point
  in a cloud is generated at the SAME instant, from the SAME pose) — it assumes that correction already
  happened upstream (project 02.08's job; see [System context](README.md#system-context--where-this-sits-in-a-robot)).

**The engineering constraint this project is built around.** A LiDAR-based localization loop runs at
**10–20 Hz** with a **< 100 ms** latency budget per scan (`docs/SYSTEM_DESIGN.md` item 1's rate table) —
whatever ICP variant a robot runs must converge to a usable pose estimate well inside that window, every
scan, forever. That budget is *why* the iteration count this project measures (point-to-plane's 6
iterations vs. point-to-point's 48 on the same scene, both figures in `README.md`) is not a curiosity: at
~5 ms/iteration on 30,000 points, 6 iterations is 25–30 ms (comfortably inside budget with margin for
everything else the perception stack does that scan cycle) while 48 iterations is ~235 ms (blows the
whole per-scan budget on registration alone). The iteration-count difference is a *real-time systems*
fact, not just a numerical curiosity.

## The math

**Notation.** A point cloud is a set of points `{p_i} ⊂ ℝ³` (meters, right-handed frame). We are given a
SOURCE cloud `{p_i}` and a TARGET cloud `{q_j}`, and we seek the rigid transform `T = (R, t)` — `R ∈
SO(3)` a rotation, `t ∈ ℝ³` a translation (m) — that best aligns the transformed source onto the target:
`x_i = R p_i + t` should land near some `q_j`. Following `docs/SYSTEM_DESIGN.md` §3 and 09.01's
convention, we call this `T_target_source` ("the source cloud's frame, expressed in the target cloud's
frame") and store it as the repo-standard pair: translation `t ∈ ℝ³` plus a unit quaternion `q =
(w,x,y,z)` (scalar-first, CLAUDE.md §12).

**The ICP problem statement.** Given an unknown correspondence between source and target points, find
`T` minimizing a per-point cost summed over corresponding pairs. Since the TRUE correspondence is
unknown, ICP alternates two steps to a local optimum:

```
repeat until converged:
  1. CORRESPOND: for the CURRENT estimate T, find each transformed source point's best target match.
  2. UPDATE: given those correspondences (now treated as fixed/known), improve T to reduce the cost.
```

This is coordinate descent on a joint (correspondence, transform) optimization — provably converges to
a LOCAL minimum of the resulting cost (Besl & McKay 1992), which is why a reasonable initial estimate
(here: identity, appropriate for scan-to-scan motion that is genuinely small — THEORY.md's synthetic
ground truth, 5–10° / 0.2–0.4 m, is exactly this regime) matters.

**Point-to-point cost.** The classic (Besl & McKay 1992) formulation minimizes summed squared Euclidean
distance:

```
E_pp(T) = Σ_i  || R p_i + t − q_{c(i)} ||²
```

where `c(i)` is point `i`'s current correspondence. The closed-form minimizer over ALL of `SO(3) × ℝ³`
at once is the Kabsch/Umeyama SVD solution (center both clouds, SVD the cross-covariance, recover `R`
from the SVD factors, then `t` from the centroids) — the textbook answer, and what production libraries
(PCL's default `TransformationEstimationSVD`) implement.

**Point-to-plane cost (Chen & Medioni 1992; linearization: Low 2004).** Instead of penalizing full
Euclidean distance to the matched point, penalize distance projected onto the matched TARGET point's
surface normal `n_j` (a unit vector, computed once per target cloud — see "PCA normal estimation"
below):

```
E_pl(T) = Σ_i  ( n_{c(i)} · (R p_i + t − q_{c(i)}) )²
```

There is no simple closed form for this minimization over the full nonlinear `SO(3)`, so it is solved by
LINEARIZING around the current estimate and taking a Gauss-Newton step — which is also how this
project's POINT-TO-POINT variant is implemented, for a reason worth stating up front: **unifying both
variants under the same linearized machinery lets one 6×6-normal-equation solver, one reduction kernel,
and one SE(3) update serve both** (the repo's "teach the pattern, not just the algorithm" ethos —
CLAUDE.md §1). The cost relative to Besl & McKay's exact SVD solve: point-to-point here needs several
Gauss-Newton steps to reach the SAME answer the closed form gives in one — visible in this project's own
measured iteration counts (point-to-point takes MORE iterations than point-to-plane, but on a
per-iteration cost basis both variants cost nearly the same, `README.md`'s "Expected output" table).

**The linearization, derived.** Let `x_i = R p_i + t` be the CURRENT transformed source point (already
in the target frame). Parameterize a SMALL update to `T` as a twist `δ = (ω, v) ∈ ℝ⁶` — `ω` a small
rotation vector (rad, about target-frame axes), `v` a translation increment (m, target frame) — applied
on the LEFT (world/target-frame composition, matching how `x_i` is already expressed):

```
T_new ≈ Exp(ω) · T_old        (rotation composes via the exact SO(3) exponential — see Numerical
                                considerations for why we do NOT also need the coupled se(3) "V matrix"
                                for the translation half)
x_i' ≈ x_i + ω × x_i + v = x_i − [x_i]_× ω + v      (first-order in ω; [x_i]_× is the skew-symmetric
                                                      cross-product matrix of x_i)
```

**Point-to-point Jacobian.** The residual is the full 3-vector `r_i = x_i − q_i`; from `x_i' ≈ x_i −
[x_i]_× ω + v`, the Jacobian of `x_i'` w.r.t. `δ = [ω; v]` is the 3×6 matrix `J_i = [ −[x_i]_×  |  I ]`.
Gauss-Newton's normal equations are `(Σ J_iᵀJ_i) δ = −Σ J_iᵀ r_i`. Multiplying out `J_iᵀJ_i` (using
`[x_i]_×ᵀ = −[x_i]_×`, a property of every skew-symmetric matrix) gives CLOSED FORMS for each point's
6×6 contribution (implemented verbatim in `kernels.cu`'s `kPointToPoint` branch):

```
H_rot,rot   = |x_i|² I − x_i x_iᵀ      (3x3, symmetric)
H_rot,trans = [x_i]_×                  (3x3; the lower-left block is its transpose, by symmetry)
H_trans,trans = I                      (3x3 — EVERY valid point contributes EXACTLY the identity here,
                                         a clean fact worth checking by eye against kernels.cu)
g_rot   = x_i × r_i
g_trans = r_i
```

**Point-to-plane Jacobian.** The residual is the SCALAR `e_i = n_i · (x_i − q_i)`. Substituting the same
`x_i' ≈ x_i − [x_i]_×ω + v` and using the scalar triple product identity `a·(b×c) = c·(a×b)` gives

```
e_i(δ) ≈ e_i0 + (x_i × n_i)·ω + n_i·v = e_i0 + Jᵢᵀδ,     J_i = [ x_i × n_i ;  n_i ]  (a 6-vector)
```

so `H_i = J_i J_iᵀ` is a RANK-1 6×6 outer product and `g_i = J_i e_i0` — both implemented verbatim in
`kernels.cu`'s `kPointToPlane` branch. This is exactly Low (2004)'s linear point-to-plane system (his
`a_i = p_i × n_i`, our `x_i × n_i` — same quantity, `x_i` being the ALREADY-linearized-around point).

**Why point-to-plane converges faster on structured scenes — the "sliding" argument.** Consider ONE
infinite flat wall. Point-to-point's cost has a real gradient pulling a source point toward its exact
matched point on the wall — including the component TANGENT to the wall, which is a red herring: sliding
a point sideways along an infinite flat surface changes NOTHING about how well the surfaces overlap
(after the slide, the point is still exactly on the wall, just matched to a different nearby point).
Point-to-point still "sees" and fights this tangential mismatch every iteration, wasting Gauss-Newton
steps correcting an error that does not actually matter, and re-discovering a NEW (equally irrelevant)
tangential mismatch after the correspondences update. Point-to-plane's cost is `(n·(x−q))²` — it is
EXACTLY ZERO for any point already lying in the matched plane, regardless of where on that plane it
lands. The tangential "sliding" direction costs point-to-plane NOTHING, so Gauss-Newton spends every
step purely on the directions that matter (closing the gap in the NORMAL direction, and — critically —
the ROTATION directions a plane constraint still fully determines, since a plane's normal direction
still rotates with the object). With THREE mutually orthogonal planes (this project's floor + 2 walls,
meeting at a corner) all six DOF are constrained collectively, and point-to-plane reaches that
constrained optimum in far fewer steps because it never wastes steps on tangential noise. A single
INFINITE plane, by contrast, leaves 3 DOF (2 in-plane translation + rotation about the normal) truly
unconstrained for EITHER variant — which is why this project's scene needs the corner (see
`scripts/make_synthetic.py`'s comments on `sample_wall_b`).

**GICP — the catalog bullet's third rung (documented here, not implemented — see README §Limitations).**
Generalized-ICP (Segal, Haehnel & Thrun 2009) generalizes BOTH variants above into one framework: instead
of matching points to points (point-to-point) or points to planes (point-to-plane), it matches LOCAL
COVARIANCE ELLIPSOIDS. Every point `p_i` (in BOTH clouds, not just the target) gets its own small local
covariance `C_i` (estimated the same PCA way this project already computes target normals from — but a
FULL 3×3 covariance, not just its smallest eigenvector). The cost becomes

```
E_gicp(T) = Σ_i  d_iᵀ ( C_i^{q} + R C_i^{p} Rᵀ )⁻¹ d_i,     d_i = q_i − (R p_i + t)
```

— a Mahalanobis distance under the COMBINED (rotated source + target) local covariance, rather than a
plain Euclidean or single-normal-projected one. Point-to-plane falls out as the special case where the
target's covariance is "flat" (near-zero variance along its normal, isotropic in-plane) and the source's
covariance is isotropic (a point has no shape); point-to-point falls out when BOTH covariances are
isotropic. GICP is strictly more general and, for many real scans (partial planes, cylindrical/curved
surfaces, sparse/anisotropic sampling), converges more robustly than either simpler variant — at the
cost of a per-point 3×3 matrix inverse (or a small linear solve) INSIDE the reduction, TWO covariance
estimates instead of one, and a genuinely different (and heavier) Jacobian derivation. Implementing it
to this project's own standard (a from-scratch derivation, a GPU kernel, a CPU oracle, a convergence
measurement) is real additional scope beyond a "teach the progression" project — see README's reduced-
scope statement and Exercise 4, which sketches exactly how to extend this project's `IcpMode` to add it.

## The algorithm

Per ICP iteration (steps labeled as in `main.cu`'s `run_icp`):

1. **Transform** the ORIGINAL source cloud by the CURRENT estimate `T_est` (never by re-transforming the
   previous iteration's already-transformed cloud — avoids compounding floating-point transform error
   across iterations). `O(N)`, GPU map.
2. **Correspond**: for every transformed source point, brute-force scan the WHOLE target cloud for its
   nearest neighbor; reject matches beyond `kDefaultMaxCorrDist`. `O(N·M)`, GPU map + per-thread search
   (parallel across `N`; serial within a thread over `M`).
3. **Build the normal system**: for every VALID correspondence, compute its 27-scalar contribution
   (mode-dependent — the closed forms above) and reduce all `N` contributions into one 6×6 `H` and
   6-vector `g`. `O(N)` per-point work, `O(log(block size))` reduction depth, GPU map + two-stage
   reduction (kernels.cuh/kernels.cu).
4. **Solve** `H·δ = −g` for the twist increment `δ = [ω; v]` via a 33.01-style Cholesky (host, double
   precision, `O(1)` — one 6×6 system, not a batch).
5. **Update** `T_est` — compose `ω`'s exact quaternion exponential onto the rotation, add `v` to the
   translation (Numerical considerations explains the asymmetry).
6. **Check convergence**: if `|ω|` and `|v|` are both below tiny thresholds, stop; else repeat (up to
   `kMaxIcpIters`).

**Complexity.** Per iteration: `O(N)` transform + `O(N·M)` correspondence search (the dominant term,
`N,M ~ 10⁴` here) + `O(N)` reduction + `O(1)` solve. Serial CPU cost for the SAME iteration would be
identical asymptotically but with the `O(N·M)` term fully sequential — the reason a CPU "manages" a
single ICP iteration on 30,000×30,000 points in ~230 ms (measured — `[time]` line in the VERIFY stage's
GPU-vs-CPU comparison on a SINGLE correspondence search, not the whole loop) while the GPU does the same
search in a few milliseconds: `N·M` independent distance evaluations is the textbook embarrassingly-
parallel workload.

**PCA normal estimation (once per target cloud, before the loop above starts).** For every target point
`j`: brute-force-scan the target cloud for its `k=16` nearest neighbors (SAME `O(M)`-per-thread search
shape as step 2, run `M` times — once per target point); form their 3×3 covariance about the local
centroid; eigendecompose via `kJacobiSweeps` cyclic Jacobi rotations; the eigenvector of the SMALLEST
eigenvalue is the surface normal (the direction the local neighborhood varies LEAST — for a flat patch,
that is the "thickness" direction, i.e. the normal; for a genuinely curved or noisy patch, it is the
best FLAT approximation in a least-squares sense). `O(M²)` total (once), not repeated per ICP iteration.

## The GPU mapping

```
Kernel 1  transform_cloud_kernel        : 1 thread / source point,  O(1) work/thread
Kernel 2  find_correspondences_kernel   : 1 thread / source point,  O(M) work/thread (scan)
Kernel 3  estimate_normals_kernel       : 1 thread / target point,  O(M) work/thread (scan) + O(1) eigen
Kernel 4  build_normal_system_kernel    : 1 thread / source point,  O(1) work/thread + BLOCK REDUCTION
```

**Brute-force search honesty (Kernels 2 & 3).** Every thread in a warp executes the SAME loop bound (`M`,
the target-cloud size) with no data-dependent branch INSIDE the loop — at loop step `m`, all 32 lanes
read `tgt_xyz[m*3 .. m*3+2]`, the SAME address, simultaneously. This is a BROADCAST, not a gather: the
L1/L2 cache serves one 12-byte read (rounded to a cache-line fetch) to the whole warp, not 32 scattered
reads. This is the same "uniform read" shape 08.01's `u_nom[t]` uses and 09.01's `__constant__` model
generalizes — kernels.cuh places this project's correspondence search explicitly on that same spectrum.
A KD-tree/BVH search (02.05's project) would cut the asymptotic cost from `O(M)` to `O(log M)` per query
at the price of a DIVERGENT, pointer-chasing traversal — different lanes of a warp would follow
different tree paths, destroying the broadcast property. This project's brute force is the honest,
simplest-to-verify choice for TEACHING the reduction pattern that follows; 02.05's dedicated project (and
this project's README Exercise 2) is where the acceleration structure itself is the lesson.

**The reduction, in detail (Kernel 4 — this project's central new GPU concept).** Every valid
correspondence contributes 27 numbers (21 unique entries of a symmetric 6×6 `H`, plus 6 entries of `g`)
that must be SUMMED across up to 30,000 points into ONE final answer — unlike every other kernel in this
project (and most of the repository's flagships), this is not "one thread, one independent result."

*Stage 1 (GPU, within a block).* Each thread computes its point's 27-scalar contribution into
REGISTERS, then the whole block cooperatively sums those contributions using a classic binary-TREE
REDUCTION through SHARED memory: write all 27 values per thread into a `[27 × threads]` shared array,
then repeatedly halve the "active" thread count, each active thread adding its partner `s` slots to the
right, for `log2(blockDim)` steps. The block's single combined 27-scalar answer is written to
`block_partials[blockIdx.x]`.

*Why CHANNEL-MAJOR shared memory (`sdata[channel][thread]`, flattened `sdata[c*threads + tid]`)?* During
the reduction, at step `s`, active thread `tid` reads `sdata[c*threads + tid]` and
`sdata[c*threads + tid + s]` — for a FIXED channel `c`, consecutive threads address CONSECUTIVE shared-
memory banks (stride 1: the textbook conflict-free access pattern on hardware with 32 banks). A
thread-major layout (`sdata[thread][channel]`, i.e. `sdata[tid*27 + c]`) would instead put a THREAD's 27
channels contiguously but make consecutive THREADS' access to the SAME channel `c` stride by 27 floats —
not a clean multiple of the 32-bank hardware, so it neither guarantees conflicts NOR guarantees their
absence the way channel-major does. Channel-major's one cost is the INITIAL write from registers into
shared memory: 27 strided stores per thread rather than one contiguous 27-float store — a good trade,
because the reduction (not the initial write) runs `log2(blockDim)` times.

*Why `kThreadsReduce = 128`, not the repo-default 256?* Shared-memory budget: `27 floats × threads × 4
bytes`. At 128 threads that is 13.5 KiB/block — comfortable headroom under Turing's 48 KiB
static-shared-memory-per-block default, leaving room for MULTIPLE resident blocks per SM (more warps to
hide memory latency behind). At 256 threads the buffer doubles to 27 KiB, and on some configurations
that permits only ONE resident block per SM — a real occupancy cost for a kernel that does `O(1)`
arithmetic per point and is therefore reduction- and launch-bound, not compute-bound, making occupancy
(not raw FLOPs) the thing worth protecting here.

*Stage 2 (HOST).* `main.cu` sums the (few hundred) block-partial rows into the final `H`, `g` with a
simple loop, accumulating in DOUBLE — small enough (`num_blocks × 27` floats, typically a few thousand)
that a plain host loop is both correct AND the clearest possible teaching code, the SAME choice 08.01
makes for its softmin blend rather than writing a second tiny GPU kernel (README's Exercise 5 names that
optimization explicitly).

**Register pressure (Kernel 3, PCA normals).** Each thread tracks its `k=16` candidate neighbors across
FOUR parallel arrays (distance + x + y + z, 64 floats) plus covariance/Jacobi scratch — a genuinely
register-heavy kernel, the same honest trade 33.01's `N=6` Cholesky makes: fewer resident warps per SM,
but this kernel runs ONCE per target cloud (not per ICP iteration), so its occupancy cost is paid once
and amortized over the whole registration, not multiplied by the iteration count.

**Constant-ish parameter passing (`Rigid3`).** Unlike 09.01's robot model — the SAME for a whole batch of
thousands of configurations, justifying a one-time `__constant__`-memory upload via `set_robot_model()`
— this project's `T_est` CHANGES every ICP iteration, so kernels.cuh passes it as an ordinary BY-VALUE
kernel parameter instead: the CUDA compiler places kernel parameters in a small dedicated parameter bank
every thread reads with `__constant__`-like broadcast efficiency, with no separate upload call needed.
This is the third point on a "how do 30000 threads read a handful of shared floats" spectrum that runs
through the repository: 09.01's `__constant__` symbol (static, batch-wide) → 08.01's uniform GLOBAL read
of `u_nom[t]` (changes every tick, read `T` times per kernel) → this project's kernel PARAMETERS (changes
every tick, read exactly once per thread) — cheapest to update, matched to how often it actually changes.

## Numerical considerations

- **FP32 kernels, FP64 host reduction.** Every GPU kernel computes in `float` (the repo default,
  CLAUDE.md §12); the two host-side sums — the block-partial-to-final reduction (stage 2 above) and
  `reference_cpu.cpp`'s oracle accumulation — use `double`, both for the SAME reason: summing thousands
  of `float`-computed terms in `float` accumulates rounding error that grows with term count, while a
  `double` accumulator absorbs that error at negligible cost (the sums are `O(num_blocks)` or `O(N)`
  scalar additions — microseconds on the host either way). This mirrors 08.01's softmin-weight
  accumulation exactly, applied here to a least-squares system instead of an importance-sampling weight.
- **Float summation order (why the GPU-vs-CPU tolerance is relative, not exact).** The GPU path sums
  each block's 128 (or fewer, ragged tail) per-point contributions via a BINARY TREE (depth
  `log2(128)=7`), then sums `num_blocks` block totals SEQUENTIALLY on the host; the CPU oracle sums all
  `N` per-point contributions SEQUENTIALLY, directly into a `double`. Floating-point addition is not
  associative — `(a+b)+c` and `a+(b+c)` can differ in their last bit — so these are genuinely DIFFERENT
  (but equally valid) summation orders of the SAME mathematical sum. Measured worst-case relative
  deviation on this project's 30,000-point main pair: `1.5e-07` (point-to-point), `3.3e-08`
  (point-to-plane) — both far under the `1e-3` gate (a `~10,000×` margin), the same "the tolerance
  exists for reordering, not for bugs" story 08.01 and 33.01 tell (a genuine indexing/formula bug shifts
  results at order `1`, not order `1e-7`).
- **Rotation parameterization: exact quaternion exponential for `ω`, simple additive `t += v` for the
  translation — NOT the full coupled `se(3)` exponential.** The MATH's derivation (§The math) shows
  `x_i' ≈ x_i − [x_i]_×ω + v` to FIRST ORDER in `ω`; a fully "correct" Lie-group retraction would compose
  `T_new = Exp(ξ)·T_old` for the FULL twist `ξ=(ω,u)`, which mixes `u` through a coupling matrix `V(ω) =
  I + (1−cos|ω|)/|ω|² [ω]_× + (|ω|−sin|ω|)/|ω|³ [ω]_×²` before it becomes a translation — i.e. the
  "exact" update would be `t_new = Exp(ω) t_old + V(ω) u`, not `t_old + v`. This project uses the
  SIMPLER additive retraction (`R_new = Exp(ω)·R_old` via the exact quaternion exponential — cheap and
  numerically robust for any `|ω|`, not just small ones — but `t_new = t_old + v` directly), which is
  the standard simplification real point-to-plane ICP implementations use (Low 2004's own presentation;
  PCL's `TransformationEstimationLLS`/`PointToPlaneLLS`). It is correct here specifically BECAUSE
  Gauss-Newton RE-LINEARIZES from scratch every iteration: any first-order error the simplified
  retraction introduces at one iteration is corrected by the NEXT iteration's fresh linearization around
  the updated (and still-valid, `R` renormalized) estimate. Contrast this with 09.01, whose STATIC
  joint-chain forward-kinematics composition has NO such iterative self-correction and therefore MUST use
  the exact composition at every joint — the same "why is this project's SE(3) handling different from
  09.01's" question kernels.cuh's header comment and `main.cu`'s `run_icp` both flag explicitly.
- **Quaternion normalization drift.** `q_est` is renormalized after every update (`quat_normalize` in
  `main.cu`) — `kMaxIcpIters` (60) applications of a Hamilton product could otherwise let `|q|` drift
  from 1 by an accumulating float32 rounding error; renormalizing every step keeps that drift at machine
  epsilon rather than letting it compound (CLAUDE.md §12's general quaternion-hygiene rule).
- **The Jacobi eigensolve's numerical robustness.** THEORY.md's earlier note (see kernels.cu's own
  header comment) matters concretely here: this project's PCA normal estimation constantly feeds the
  eigensolver COVARIANCE MATRICES with one near-zero eigenvalue and two nearly-equal larger ones — the
  textbook "nearly degenerate" case that trips up closed-form (cubic-equation) 3×3 eigenvalue formulas
  (their trig branch needs care near repeated roots). Cyclic Jacobi has no special case at all: the SAME
  straight-line rotation code handles every input, degenerate or not, which is why it is the chosen
  method here (Numerical Recipes §11.1; `kJacobiSweeps = 8` is generous for a 3×3 — 3–5 sweeps is the
  textbook convergence estimate).
- **Correspondence tie-breaking and the VERIFY stage's exact-index claim.** Both the GPU kernel and the
  CPU oracle scan target index `m = 0..M-1` in the SAME order with the SAME strict `<` comparison, so the
  identical index wins on both paths whenever no two candidate distances are EXACTLY tied at the bit
  level — true with overwhelming probability for continuous, noisy synthetic geometry, and specifically
  checked (not just assumed) only at ITERATION 0, where the initial 5–10° / 0.2–0.4 m misalignment keeps
  every true nearest neighbor's distance far from its runner-up's (THEORY.md's own "How we verify
  correctness" section below states the measured result: 0/30,000 mismatches).
- **The 6×6 solve's damping.** `kDamping = 1e-3` is added to `H`'s diagonal before every Cholesky solve —
  the exact "`JᵀJ + λI`" Levenberg-Marquardt-style regularization 33.01's own header comment names as a
  real-world use of its batched Cholesky kernel. It is negligible next to `H`'s own diagonal magnitude
  (point-to-point's translation block alone accumulates exactly `num_valid_correspondences` per pair —
  thousands, here) and exists purely as safety margin; no measured run on this project's wall-dominated
  scenes has come close to needing it (the corner geometry keeps `H` well-conditioned throughout).

## How we verify correctness

Three independent checks, because ICP can be *numerically right and behaviorally wrong* (or the reverse)
— the same reasoning 08.01's two-tier verification uses, extended here to three tiers because this
project has three genuinely different things that could break:

1. **Correspondence exact-index check (VERIFY).** On pair 0's identity-transform (iteration-0) inputs,
   the GPU kernel's and the CPU oracle's nearest-neighbor index for every one of 30,000 source points
   must match EXACTLY (not within tolerance — see the tie-breaking note above for why exactness is a
   reasonable ask here). Measured: 0/30,000 mismatches. This catches indexing bugs, off-by-one tail
   handling, and gate-logic errors instantly — a single wrong index is not a rounding error, it is a
   different point entirely.
2. **Normal-system relative-tolerance check (VERIFY), both modes.** The GPU's (float-reduced,
   double-summed) `H`/`g` must agree with the CPU oracle's (fully double-accumulated) `H`/`g` within
   relative tolerance `1e-3` (floored at `max(1, |value|)` the same way 08.01/33.01 floor their
   tolerances). Measured worst case: `1.5e-07` (point-to-point), `3.3e-08` (point-to-plane) — ~10,000×
   margin, catching formula bugs (which shift results at order 1) while tolerating float summation-order
   noise (which does not).
3. **Ground-truth pose gate (CHECK), all 4 runs.** The only check that validates the WHOLE closed loop —
   correspondence search, normal estimation, reduction, solve, AND the SE(3) update — end to end, against
   a value nothing in the pipeline ever sees directly (the committed ground-truth transform). A bug that
   leaves every individual kernel "locally correct" but breaks the LOOP (a sign error in the update, a
   forgotten renormalization, a correspondence gate set wrong) would sail through checks 1–2 and fail
   here. Measured worst case (pair1, point-to-point): `0.119°` rotation error, `0.007` m translation error
   — both roughly `7-8×` under the `1.00°` / `0.050` m gate.

A fourth check — **point-to-plane converges in fewer iterations than point-to-point on the wall-dominated
main pair** — validates the project's CENTRAL TAUGHT CLAIM specifically (not just "does ICP work", but
"does the thing THEORY.md argues for actually happen"): measured `6` vs. `48` iterations on pair 0 (an
`8×` difference), `5` vs. `32` on pair 1 (`6.4×`). This is what turns "we implemented two ICP variants"
into "we taught why one beats the other, and measured it."

All four checks run against the COMMITTED synthetic pairs (`data/sample/`), so the whole verification
suite runs OFFLINE, deterministically, with no network access and no hardware — `demo/run_demo.ps1`
reproduces every number in this document.

## Where this sits in the real world

- **PCL's `IterativeClosestPoint`** implements point-to-point (SVD, the Besl & McKay closed form — not
  this project's linearized Gauss-Newton point-to-point), and
  `TransformationEstimationPointToPlaneLLS` implements almost exactly this project's point-to-plane
  linearization, with a `pcl::search::KdTree` for correspondence search instead of a brute-force scan.
- **Open3D's `registration_icp`** offers point-to-point, point-to-plane, AND (as of recent releases)
  colored-ICP variants, with a voxel-hashed or KD-tree correspondence backend and, in newer builds, a
  CUDA tensor backend — the production answer to this project's brute-force search.
- **GICP (Segal, Haehnel & Thrun 2009) and FAST-GICP** are the production answer to this project's
  documented-only third rung: FAST-GICP in particular is a real-time GPU/multithreaded implementation
  used in several open-source LiDAR-odometry stacks, built on exactly the covariance-to-covariance
  formulation derived above.
- **KISS-ICP (Vizzo et al. 2023)** shows the opposite lesson from GICP's added complexity: a
  DELIBERATELY simple point-to-point pipeline, with careful engineering around it (adaptive
  correspondence thresholds, constant-velocity motion prediction, voxel downsampling) rather than a
  fancier cost function, competitive with much more complex LiDAR-odometry stacks on public benchmarks —
  a reminder that "which ICP variant" is one knob among many in a real pipeline, not the whole story.
- **What production LiDAR odometry (05.09's territory) adds beyond this project:** motion-distortion
  correction (02.08) BEFORE registration; a KD-tree or voxel-hashed correspondence search; scan-to-MAP
  registration (not just scan-to-scan) once a map exists, so drift does not accumulate unbounded; a
  proper uncertainty/covariance estimate on the recovered pose (feeding a pose graph or an EKF/factor-
  graph backend — domain 04/05's territory); and robust-kernel weighting (Huber/Cauchy losses on the
  residuals) so a handful of bad correspondences (dynamic objects, mixed pixels) cannot dominate the
  solve the way an un-weighted least-squares system can.
- **GPU-accelerated production LiDAR stacks** (nvblox's registration path, Autoware's NDT/ICP
  localization) put this exact correspondence-search-then-reduce shape on the GPU for the same reason
  this project does: it is the part of the loop that scales with point count, and point count is exactly
  what a modern high-resolution LiDAR gives you a lot of.
