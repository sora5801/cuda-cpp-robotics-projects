# 02.10 — FPFH descriptors + RANSAC global registration: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

A LiDAR returns a **point cloud**: samples of a surface's geometry, in the sensor's own reference
frame, at the instant it was scanned. Two scans of the *same* physical scene, taken from two *different*
sensor poses, describe the same surfaces in two different coordinate systems. **Registration** is the
problem of finding the rigid transform `T` (rotation + translation, six degrees of freedom for a rigid
body — no scale, no shear, because a LiDAR measures true metric distance) that maps one scan's points
onto the other's.

[02.06 (ICP)](../02.06-icp-point-to-point-point-to-plane-gicp/THEORY.md) solves this by *iterative
refinement*: start from a guess, find nearest-point correspondences, fit a small correction, repeat.
That only works when the initial guess is already close — physically, when the correspondence search
"nearest point" is likely to find the TRUE matching point, which requires the two scans to already
nearly overlap. A robot that just woke up (no odometry since power-on), just recovered from a wheel
slip or an e-stop (no reliable prior pose), or is trying to merge two independently-built maps (no
shared coordinate frame at all) has **no such guess**. This is the **global registration** problem, and
it needs a fundamentally different tool: instead of "assume we're close and refine," it must ask "what
LOCAL GEOMETRY looks distinctive enough to match across scans, with no positional prior at all?"

The physical answer is: the **shape of a small neighborhood around a point** — is it flat, is it a
sharp edge, is it a rounded corner, does its normal turn quickly or slowly as you move across the
surface — is a property of the *object*, not of the *sensor pose that observed it*. A LiDAR return from
the corner of a crate looks like a corner whether the sensor was 2 m north of it or 3 m southeast and
rotated 140 degrees. This project's engineering task is to turn that physical intuition into a **number
that is provably unchanged by any rigid sensor motion** — a *pose-invariant local-shape descriptor* —
so two points in two different scans can be compared and matched purely on local geometry, no position
information involved at all.

**Engineering constraints that make this hard, not just "compute a number":** real LiDAR returns are
noisy (range noise ~mm-cm), the local neighborhood used to describe a point is finite and therefore
itself position-dependent (which `k` neighbors get included shifts slightly as the exact scan pattern
changes), and — the constraint this project's synthetic scene is deliberately built to expose — many
real surfaces are LOCALLY SELF-SIMILAR: a flat floor patch looks like every other flat floor patch, no
matter how distinctive a *global* view of the room would be. A descriptor computed from too small or
too generic a neighborhood cannot tell one floor tile from another. This project's crate and cylindrical
pillar exist specifically to give the scene some genuinely distinctive local geometry (sharp edges,
constant curvature) alongside the inevitable self-similar flat regions — the same "aliasing" hazard
[01.04](../../01-perception-cameras-vision/01.04-feature-pipeline/README.md) and
[01.05](../../01-perception-cameras-vision/01.05-sift-surf-on-gpu/README.md) teach for 2-D image
patches (a blank wall photographed from any angle looks the same; a corner does not), arriving in 3-D.

### The Darboux frame — why the angle triplet is pose-invariant

Take a query point `p_q` with unit normal `n_q`, and one of its neighbors `p_k` with unit normal `n_k`.
Define the **Darboux frame** rooted at the query point:

```
u = n_q                                  (the query's own normal)
d = (p_k - p_q) / |p_k - p_q|            (unit direction from query to neighbor)
v = (u x d) / |u x d|                    (perpendicular to u, in the plane containing u and d)
w = u x v                                (completes a right-handed orthonormal basis)
```

      n_q = u
       ^
       |    n_k
       |   ^
       |  /
       | /
   p_q *------> d --------* p_k
       \
        v  (into/out of the u-d plane, perpendicular to both)

Three angles describe how the neighbor's normal `n_k` sits relative to this frame:

```
alpha = v . n_k          (how much n_k tilts out of the u-d plane)
phi   = u . d             (how much the connecting line leans away from the query's own normal)
theta = atan2(w . n_k, u . n_k)   (n_k's azimuth around u, in the v-w plane)
```

**Why this triplet cannot change under a rigid transform.** Apply any rotation `R` and translation `t`
to *both* points and *both* normals together (`p -> R*p + t`, `n -> R*n` — normals rotate but do not
translate, because they are directions, not positions): the difference `p_k - p_q` becomes
`R*(p_k - p_q)` (translation cancels), so `d` becomes `R*d`; `u` becomes `R*n_q`; and `v`, `w`, being
built from cross products of already-rotated vectors, become `R*v`, `R*w`. Every dot product in the
triplet formula is therefore between two vectors that both picked up the *same* rotation `R`, and
`(R*a).(R*b) = a.b` for any rotation (rotations preserve inner products — this is literally the
definition of a rotation matrix, `R^T*R = I`). `atan2` of two such preserved dot products is likewise
unchanged. **The triplet depends only on the RELATIVE geometry of the two points and their normals —
never on where the sensor was standing.** This is the entire reason FPFH works as a matching tool
across two independently-posed scans: THE SAME PHYSICAL NEIGHBORHOOD produces the SAME angle triplet
regardless of which scan (and therefore which sensor pose) it was measured from. `main.cu`'s
`descriptor_invariance` gate measures exactly this claim, using GROUND-TRUTH point correspondences (not
the algorithm's own matching) so the measurement is honest.

## The math

**Normals (STAGE 1).** For a point `q` and its `K`-nearest-neighbor set (`K = 20` here), the local
surface normal is the eigenvector of the SMALLEST eigenvalue of the neighborhood's mean-shifted
covariance matrix `C = (1/K) * sum_i (p_i - mean)(p_i - mean)^T` — the total-least-squares plane
through the neighborhood (identical derivation to
[02.09](../02.09-normal-curvature-estimation-at-millions/THEORY.md)'s "The math", cited, not repeated
here). Oriented toward the cloud's own centroid (a robust stand-in for "the interior side" on this
project's mostly-enclosing room geometry — a simpler convention than 02.06's sensor-origin orientation,
suited to a *static* cloud with no privileged viewpoint).

**SPFH (STAGE 2).** For query point `q`, `SPFH(q)` is three 11-bin histograms (one per angle above),
each built by computing the Darboux triplet against every one of `q`'s `K` neighbors and incrementing
the bin the corresponding angle falls into, then normalizing each 11-bin block to sum to 1:

```
for each neighbor k of q:
    (alpha, phi, theta) = darboux_triplet(n_q, p_q, n_k, p_k)
    hist_alpha[bin(alpha)] += 1;  hist_phi[bin(phi)] += 1;  hist_theta[bin(theta)] += 1
SPFH(q) = concat(hist_alpha, hist_phi, hist_theta) / K        (33 dims, three blocks each summing to 1)
```

This is `O(K)` per point — the "**S**implified" in SPFH: true **PFH** (Rusu et al.'s earlier, un-fast
version) instead computes a joint histogram over EVERY PAIR *within* the K-neighborhood (`K*(K-1)/2`
pairs, each needing the "which point is the source" symmetry-breaking rule PFH adds precisely because
neither point is privileged as "the query" the way SPFH's `p_q` always is) — `O(K^2)` per point,
`O(n*K^2)` total. SPFH sidesteps this by always fixing `p_q` as the Darboux frame's origin, at the cost
of only directly capturing pairs that include the query itself.

**FPFH (STAGE 3) — recovering PFH's descriptive power in `O(K)`.** The "**F**ast" step re-weights each
point's own SPFH with its neighbors' ALREADY-COMPUTED SPFH:

```
FPFH(q) = SPFH(q) + (1/K) * sum_{k in neighbors(q)} (1/dist(q,k)) * SPFH(k)
FPFH(q) = FPFH(q) / sum(FPFH(q))                      (L1-normalize the full 33-dim vector)
```

Read this as **two rings**: ring 1 is `q`'s own `K` neighbors (the sum, directly); ring 2 is each of
those neighbors' OWN `K`-neighborhoods — but ring 2's information arrives **for free**, already baked
into `SPFH(k)` by a PRIOR kernel launch, with no second explicit traversal. The result approximates what
a true `O(K^2)` PFH histogram over the full 2-ring neighborhood would show, at `O(K)` cost per point —
`O(n*K)` total, not `O(n*K^2)`. This is the complexity trade the catalog bullet names explicitly, and it
is why FPFH — not PFH — is the descriptor every modern point-cloud library (PCL, Open3D) actually ships.

**Descriptor matching + the ratio test (STAGE 4).** For each source point `s`, find its nearest
(`d1`) and second-nearest (`d2`) target descriptor by squared Euclidean distance over the 33 dimensions.
Accept the match only if `d1 <= kMatchRatioMax^2 * d2` (`kMatchRatioMax = 0.95` here — looser than
SIFT's classic 0.7-0.8, because a 33-D geometric histogram over a partially self-similar room scene is
inherently less separable than a 128-D SIFT descriptor over a richly textured photograph; measured, not
assumed, in the `descriptor_distance_histogram.csv` artifact). The logic: if the SECOND-best match is
nearly as good as the best, the match is ambiguous and more likely to be geometrically wrong than
correct — exactly [01.04](../../01-perception-cameras-vision/01.04-feature-pipeline/THEORY.md)'s
lesson for 2-D image features, unchanged in spirit for 3-D shape histograms.

**RANSAC over correspondence triplets (STAGE 5).** Given a correspondence set `C = {(s_i, t_i)}` of
size `nc` (post-ratio-test), classical point-cloud RANSAC samples **3 correspondences**, fits the rigid
transform that would make all 3 exact, and counts how many of the OTHER `nc` correspondences that
transform explains within a threshold (`kRansacInlierThresholdM`). This is 02.03's plane-RANSAC pattern
(3 *points* define a plane) generalized: **3 correspondence *pairs* define a rigid transform** (given 3
non-collinear point pairs with no reflection ambiguity, the least-squares rigid fit is unique).

**Prescreen: why check edge lengths before fitting.** A rigid transform preserves every pairwise
distance: `|R*a + t - (R*b + t)| = |R*(a-b)| = |a-b|` for any rotation `R` (rotations preserve vector
length) — translation cancels entirely. So if `(s0,t0), (s1,t1), (s2,t2)` were ALL correct
correspondences of the same true transform, then `|s0-s1|` must equal `|t0-t1|` (and likewise for the
other two pairs) to within noise. If even ONE of the three is a wrong correspondence, this equality
almost always fails by a large margin (this scene's characteristic length scale is meters; a wrong
match's positional error is essentially uncorrelated with the true one's). Checking three subtractions
and three compares — `edge_length_prescreen` — catches the overwhelming majority of bad triplets
**before** ever running a fit (measured: `prescreen_efficiency`'s `[info]` line reports the exact
rejected fraction on the committed data). This is what makes RANSAC-over-correspondences fast: the
expensive step (a 4×4 eigensolve) only runs on triplets that already look geometrically self-consistent.

**Horn's closed-form rigid fit.** Given `count >= 3` correspondence pairs, minimize
`sum_i |R*s_i + t - t_i|^2` over rigid transforms `(R, t)`. Horn (1987) shows the optimum is found in
closed form: subtract each set's centroid, form the 3×3 cross-covariance
`M = sum_i (s_i - c_s)(t_i - c_t)^T`, pack it into the 4×4 SYMMETRIC "key matrix"

```
N = [ tr(M)          M23-M32        M31-M13        M12-M21      ]
    [ M23-M32        M11-M22-M33    M12+M21        M31+M13      ]
    [ M31-M13        M12+M21       -M11+M22-M33    M23+M32      ]
    [ M12-M21        M31+M13        M23+M32       -M11-M22+M33  ]
```

(`Mij` = row `i`, column `j` of `M`, 1-indexed; `tr(M) = M11+M22+M33`), and take the eigenvector of `N`'s
LARGEST eigenvalue as the optimal rotation quaternion `(w,x,y,z)`. Translation follows in closed form:
`t = c_t - R*c_s`. This is a genuinely different eigenproblem from STAGE 1's normal-fitting one — there
the SMALLEST eigenvalue's eigenvector is wanted (least total-squared perpendicular distance to a
plane); here the LARGEST eigenvalue's eigenvector is wanted (this is Horn's algebraic identity: the sum
of squared alignment residuals, expressed via the quaternion, is minimized exactly when the quadratic
form `q^T*N*q` is maximized, i.e. at `N`'s top eigenvector — Horn's paper derives this from the
quaternion representation of a rotated-point residual). The SAME cyclic-Jacobi machinery
(`jacobi_eigen_4x4`, this project's own extension of the repo's 3×3 pattern to 6 off-diagonal pairs
instead of 3) diagonalizes both — different algebra, same numerical tool.

**The point-to-plane ICP handoff (STAGE 6, condensed from 02.06's fuller derivation, cited).** For the
transformed source point `x = R*p + t` matched to target point `q` with target normal `n`, the
point-to-plane residual is `e = n . (x - q)` — the component of the misalignment ALONG the surface
normal (converges faster than point-to-point because it does not penalize a point sliding along a flat
surface, which contributes no true error there). Linearizing around a small perturbation
`x_new = Exp(w)*x + v ~= x + w cross x + v`:

```
de/dw = x cross n     (a 3-vector: the "rotation part" of the Jacobian row)
de/dv = n              (a 3-vector: the "translation part")
J = [de/dw ; de/dv]   (6 entries, this project's [wx,wy,wz,vx,vy,vz] order, 02.06's convention)
```

Every matched point contributes `H += J^T*J` (6×6) and `g += J^T*e` (6×1); the Gauss-Newton step solves
`H*delta = -g` (a 33.01-style Cholesky solve, host-side, double precision) and folds `delta` back via
`R <- Exp(delta.w)*R`, `t <- t + delta.v` — 02.06's exact convention, reused unchanged here.

## The algorithm

**End to end, per scan pair:**

1. **STAGE 1** (per point, both clouds): brute-force KNN (`k=20`) → mean-shifted covariance → Jacobi
   eigensolve → sensor-agnostic normal, oriented toward the cloud's own centroid. `O(n^2)` (KNN) +
   `O(n*k)` (covariance/eigensolve).
2. **STAGE 2** (per point, both clouds): SPFH via the Darboux triplet against each of `k` neighbors.
   `O(n*k)`.
3. **STAGE 3** (per point, both clouds): FPFH via the weighted neighbor-SPFH re-accumulation. `O(n*k)`.
4. **STAGE 4** (per source point): nearest+second-nearest target FPFH (brute force, 33-D), ratio test.
   `O(n_src * n_tgt * 33)`.
5. **STAGE 5** (per hypothesis, `K = 8192` of them): sample 3 correspondences (retry up to 8 times on a
   degenerate/duplicate draw) → prescreen → Horn fit → score against the whole correspondence set `nc`.
   `O(K * nc)` total, fully parallel across `K`.
6. Host: `select_best_hypothesis` (argmax inlier count, ties toward lowest index) — `O(K)`, sequential
   (K is small; not worth a reduction kernel, the same "know when NOT to parallelize" lesson 02.03
   teaches for its own hypothesis selection).
7. Host: gather the best hypothesis's inlier correspondences, refit via the SAME `rigid_fit_horn` over
   all of them (not just 3) — `O(#inliers)`, sequential, an `O(1)`-shaped step not worth a kernel launch
   (mirroring 08.01's host-side softmin blend and 02.06's host-side 6×6 solve).
8. **STAGE 6**: up to `kIcpMaxIters = 10` point-to-plane ICP iterations from the refit, using the
   already-computed target normals. `O(iters * n_src * n_tgt)` (brute-force correspondence search per
   iteration).

**Serial cost** for a single CPU thread following this same recipe: dominated by STAGE 4's matching
(`O(n_src*n_tgt*33)`, tens to hundreds of millions of scalar ops at this project's scale) and STAGE
1/2/3's KNN (`O(n^2)`). **Parallel cost**: every stage maps to one GPU thread per independent unit of
work (a point, or a hypothesis), so wall-clock scales with the LONGEST per-thread inner loop, not the
total operation count — the entire reason this pipeline runs comfortably in real time at this project's
scale despite doing tens of millions of scalar comparisons.

## The GPU mapping

**Histogram kernels: per-thread-private, not atomic.** STAGE 2/3's SPFH/FPFH kernels give EVERY point
its OWN 33-entry histogram, computed entirely by ONE thread and written to ONE output row. No two
threads ever touch the same output location, so **no atomics are used or needed** — this is a
deliberate, argued choice (`kernels.cu`'s file header spells it out): the alternative mapping (one
thread per (point, neighbor) *pair*, scattering votes into a shared per-point histogram) would need
atomics for no benefit, since `K=20` is far too small to be worth splitting one point's tiny amount of
work across multiple threads — the GPU's real parallelism budget is far better spent across the
thousands of independent POINTS, not within one point's 20-entry inner loop.

**STAGE 6's accumulator: atomics, and why that is the right call HERE.** `icp_accumulate_kernel` is the
direct contrast: EVERY matched source point contributes to the SAME shared 27-double accumulator (`H`'s
21 upper-triangle entries + `g`'s 6) — there is no way to give each thread a private copy of "the
answer" because the 6×6 linear system genuinely IS the sum over all points. 02.06 solves this with a
shared-memory block-tree reduction (partial sums per block, finished on the host); this project uses
plain `atomicAdd(double)` (native since sm_60, well within this repo's sm_75 floor) directly into global
memory instead — a documented SIMPLIFICATION, not an oversight: at this project's `n_src` (a few
thousand points, not 02.06's larger runs), atomic contention on 27 locations is cheap, and the code is a
third the size of a full block reduction. Exercise 4 asks you to build the block-reduction version and
measure the actual speed difference at this scale.

**The RANSAC hypothesis farm: hypothesis-parallel, not point-parallel.** `ransac_hypotheses_kernel` maps
ONE THREAD PER HYPOTHESIS, not one thread per correspondence — 02.03's identical "farm" pattern, cited.
Each thread does a genuinely serial amount of work internally (sample, prescreen, fit, score against all
`nc` correspondences), but the `K=8192` hypotheses are completely independent of each other, so the
natural GPU mapping parallelizes ACROSS hypotheses. This mirrors 08.01's rollout-per-thread mapping: the
"batch of independent small trials" pattern recurs throughout this repository whenever a stochastic
search replaces an exhaustive one.

**Memory hierarchy.** Every kernel here uses registers/local memory for its per-thread working set
(the KNN heap, the histogram, the RANSAC hypothesis's own small buffers) and reads its inputs from
global memory with no shared-memory tiling — at this project's point counts (~1.5k-3.2k/scan), the
working sets are small enough that occupancy is not bandwidth-bound the way 02.09's million-point
throughput target is; THEORY.md's honest answer for "why no shared memory" is "it would not move the
needle at this scale, and 02.09/02.01 already teach that pattern where it matters."

## Numerical considerations

- **Two-pass covariance** (STAGE 1): mean first, THEN the covariance around it — not the textbook
  one-pass `E[pp^T] - mean*mean^T` formula, which loses precision catastrophically when points sit far
  from the coordinate origin but tightly clustered locally (identical reasoning to
  [02.09](../02.09-normal-curvature-estimation-at-millions/THEORY.md#numerical-considerations), cited).
- **Boundary binning**: `angle_to_bin` clamps its fractional position to `[0, 1-eps)` before truncating
  to an integer bin, so a value exactly AT the upper edge (`alpha=1.0`, `theta=pi`) lands in the LAST
  bin rather than one-past-the-end — this shared formula runs identically (same float32 arithmetic) on
  GPU and CPU, so `VERIFY(spfh)`/`VERIFY(fpfh)` measure near-exact (not merely "close") agreement in
  practice (see the demo's `[info]` lines).
- **Normal-sign consistency**: every normal is oriented toward its cloud's own centroid (STAGE 1),
  feeding directly into the Darboux frame's `u = n_q` — an INCONSISTENT sign convention between two
  scans of the same physical point would flip `alpha` and `theta`'s signs unpredictably and break
  `descriptor_invariance`; the centroid-orientation rule is simple enough to apply identically in every
  independent scan (no cross-scan coordination needed) while still being CONSISTENT per physical point
  across scans for this project's mostly-convex room geometry.
- **Degenerate triplets**: `rigid_fit_horn` guards against a near-zero cross-covariance trace (all three
  points nearly coincident even after the edge-length prescreen — defensive, not expected to fire given
  the prescreen already ran) by returning `false` rather than a garbage transform; callers must check.
- **Jacobi sweep counts**: `kJacobiSweeps3 = 8` (3×3, three off-diagonal pairs) and `kJacobiSweeps4 = 14`
  (4×4, six off-diagonal pairs — more pairs need more sweeps to reach the same float32 precision,
  measured via `VERIFY(ransac_refit)`'s tight rotation/translation tolerance against an independent
  double-precision oracle, which passed with ~0.0000 deg / 0.0000 m residual on the committed data —
  comfortable margin under the sweep count chosen).
- **Determinism**: every RNG draw (KNN tie-breaks via `knn_less`, RANSAC's `hypothesis_seed`) is fixed
  and counter-based — the same committed data + the same seed reproduces byte-identical results, the
  precondition for `VERIFY(ransac)`'s bit-exact-checkable hypothesis generation.

## How we verify correctness

Two tiers, per `reference_cpu.cpp`'s file-header ruling (restated per-stage in `kernels.cuh`'s
"Twin-vs-shared ruling"):

- **GPU-vs-CPU twins** (`VERIFY(...)` lines): KNN (exact-match required — a shared `knn_less` total
  order makes this achievable, not merely aspirational), normals (>=95% of points with cosine agreement
  >=0.98 — STAGE 1 uses TWO INDEPENDENT eigensolves, 02.09's stricter choice, so near-isotropic
  covariances can legitimately disagree slightly), SPFH/FPFH (>=98% of bins within 0.05 absolute
  tolerance — the Darboux/binning FORMULA is shared, so agreement is normally near-exact; the tolerance
  accommodates the small normal-agreement slack propagating through), descriptor matching (>=95% index
  agreement), the RANSAC hypothesis farm (>=95% of hypotheses' valid-flag+inlier-count agree — hypothesis
  GENERATION calls the shared `rigid_fit_horn` directly so this is close to bit-exact-checkable; scoring
  is independently looped), the RANSAC refit (real float refit vs. a FULLY INDEPENDENT double-precision
  oracle — `ransac_refit_cpu`'s own Jacobi 4×4, its own accumulation, 02.03's `ransac_refine_cpu`
  precedent), and the ICP point-to-plane system (GPU atomic-float accumulation vs. CPU sequential-double
  accumulation, relative tolerance — 02.06's identical "more precise by construction" ruling for the CPU
  side).
- **Independent, ground-truth/invariant gates** (`GATE ...` lines) — the tier that proves the ALGORITHM
  is right, not merely that two implementations agree: `descriptor_invariance` (mean L1 distance between
  FPFH computed independently in each scan's own frame, for GROUND-TRUTH-identical physical points —
  the pose-invariance property itself, measured), `registration_recovery` (recovered pose vs. the TRUE
  pose from `pairs_meta.csv`, never touching the GPU/CPU comparison at all), `icp_negative_control`
  (local ICP FROM IDENTITY must FAIL by a wide margin — the negative control that proves global
  registration is doing real work, not redundant with what ICP alone could already do), and
  `ransac_formula` (the classical iteration-count requirement, computed independently from the SAME
  measured inlier ratio the demo's own RANSAC run produces — an analytic sanity check on the budget, not
  a comparison to any oracle).

This two-tier structure exists precisely because twin agreement alone is BLIND to a bug that lives
identically in both the shared header formula and its device transcription (`reference_cpu.cpp`'s file
header names the exact flagship-13.03 incident that motivated this rule repo-wide) — the ground-truth
gates above are what actually validate the Horn/Darboux/RANSAC MATH, not merely its faithful
parallelization.

## Where this sits in the real world

**PCL's `SampleConsensusPrerejective`** is the production-grade version of exactly this pipeline
(FPFH match → RANSAC-with-prerejection → refit), with additional refinements this teaching version
scopes out: adaptive correspondence rejection thresholds, a configurable polygon-similarity check (this
project's edge-length prescreen generalized to more than 3 points at once), and multi-threaded (not
GPU) parallelism. **Open3D's** `registration_ransac_based_on_feature_matching` is the same shape again,
with a modern Python-first API and a `CorrespondenceCheckerBasedOnEdgeLength` that names this project's
prescreen explicitly.

**Neighbor engine crossover.** This project's brute-force KNN and brute-force descriptor matching are
the honest, simplest-correct choice at ~1.5k-3.2k points/scan — `O(n^2)` here is a few million
operations, microseconds on any modern GPU. [02.09](../02.09-normal-curvature-estimation-at-millions/THEORY.md)'s
voxel-hash index becomes the right tool once point counts reach the millions (its own catalog promise:
"at millions of points/sec") — the crossover is roughly where the constant-factor overhead of building
and querying a spatial index becomes smaller than the quadratic brute-force cost, typically in the tens
of thousands to low millions of points depending on hardware. A production LiDAR relocalization service
running on submaps of that size would use a KD-tree or the voxel hash, not this project's brute force.

**Where hand-crafted FPFH is being replaced.** The modern research frontier (explicitly OUT of scope
here, named honestly) is **learned** local descriptors — FCGF (Fully Convolutional Geometric Features),
D3Feat, and others — trained end-to-end on large point-cloud datasets, typically more distinctive (fewer
false matches on self-similar geometry, this project's own honest struggle on its flat floor/wall
patches) at the cost of needing training data and a learned model rather than a closed-form geometric
formula. **TEASER++** (Yang, Shi & Carlone, 2020) replaces RANSAC's random sampling with a certifiably
globally-optimal robust estimator (graduated non-convexity + a graph-theoretic max-clique inlier
selection) that tolerates far higher outlier rates than RANSAC's probabilistic guarantee — relevant
precisely when, as this project's own committed scene demonstrates, the raw correspondence-set inlier
ratio can be as low as ~10%. Both are named here as the honest "what a real system reaches for next,"
not reimplemented.
