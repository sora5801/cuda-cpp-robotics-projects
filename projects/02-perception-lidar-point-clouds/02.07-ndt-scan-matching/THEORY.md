# 02.07 — NDT scan matching (Autoware-style map localizer): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

### Why a small patch of scanned surface is well modeled as Gaussian

A LiDAR beam measures range along one ray. A rigid, locally-flat surface — a wall, a floor, a door
— intersected by many beams from nearby directions produces points that, in a small neighborhood,
scatter around the TRUE surface with error from three physical sources: (1) the sensor's own range
noise (a few centimeters for a mechanical LiDAR, driven by time-of-flight jitter and receiver SNR —
this project models it as isotropic Gaussian, σ=0.02 m, a simplification of the true 1-D-along-beam
noise, noted honestly below); (2) the beam's finite angular spread hitting a slightly non-planar
patch; (3) sub-voxel surface texture (paint, mortar lines, dust). All three are small, additive, and
roughly independent of position within the patch — exactly the conditions under which a Gaussian is
the right local model (the same physical argument 02.03's ground-plane fit and 02.09's per-point
normal/curvature estimation make from a different angle, and directly related to 02.06's PCA-based
surface-normal estimation: **the covariance ellipsoid's smallest-eigenvalue direction IS the
surface normal**, the same PCA object 02.06 extracts for point-to-plane ICP — see "The GPU mapping"
below for exactly how this project reuses that construction, just for a different downstream job).

A single voxel's sample covariance is therefore not a statistical nicety — it is a genuine local
model of a small piece of the physical world: an ANISOTROPIC ellipsoid, thin along the surface
normal (little true variation once sensor noise is subtracted) and wide in-plane (the surface
extends across the voxel). A flat wall's voxel covariance can have a smallest eigenvalue that is
orders of magnitude below its largest — physically correct, but numerically dangerous once you
invert it (see "Numerical considerations").

### Map representation economics — why compile the map into Gaussians at all

A raw point cloud is a list of individually-meaningless samples: any single point carries no
information about the surface's SHAPE, only its own noisy position. A voxelized Gaussian map
throws away the individual points and keeps only the SUFFICIENT STATISTICS of each local patch
(mean + covariance — everything a Gaussian likelihood needs) — a real, measurable compression, not
just an algorithmic convenience:

```
Raw map.bin (this project):        40,000 points x 12 bytes (xyz, fp32)      = 480,000 bytes
Fine NDT grid (this project):         196 occupied voxels x ~40 bytes
                                       (mean[3] + inv_cov[6] + count, fp32)   =   7,840 bytes
                                                                    ratio ~=  61x
```

That 61x is MEASURED on this project's own committed map (`data/README.md`), not a textbook claim.
For a real building-scale map (thousands of square meters, not 180 m²) the ratio only improves,
because point DENSITY grows with survey time while voxel COUNT grows with surface AREA — the same
economic argument that makes 02.01's voxel downsampling worthwhile applied one level further: NDT
does not just downsample the map, it replaces it with a compact PARAMETRIC model a localizer can
score against with no search at all. This is why Autoware ships `.pcd` maps that NDT loads and
voxelizes ONCE at startup, not per scan — the whole point of the representation is amortizing this
compression across every localization tick for the vehicle's entire operating lifetime.

### Engineering constraints

A production NDT localizer answers "where am I" inside a 10 Hz budget (SYSTEM_DESIGN item 1) shared
with every other perception/planning stage downstream of it — a late pose is as bad as a wrong one,
because the EKF fusing it (04.xx) assumes a bounded latency. The map itself is a physical asset with
its own engineering constraints (survey accuracy, storage, staleness as the environment changes) —
PRACTICE.md §1 covers the unsung "keeping the map current" operational burden a pure-algorithm
treatment skips entirely.

## The math

### Frames, notation, and the local parameterization (reused, cited)

`T_map_scan = (R, t)`: `x_map = R * x_scan + t`, `R` row-major 3x3, `t` in meters — 02.06's
`T_target_source` convention with "map" playing target's role. The pose is optimized in LOCAL
6-vector coordinates `delta = [omega(3); v(3)]` — 01.17's EXACT decoupled SO(3) x R^3
parameterization, cited and reimplemented in `kernels.cuh` (`retract`, `so3_exp`, `skew3`):

```
R_new = Exp(omega) * R          (exact Rodrigues; omega a LEFT/world-frame perturbation)
t_new = t + v
```

### The voxel Gaussian and the Mahalanobis score

Voxel `k` stores mean `mu_k` (m) and REGULARIZED covariance `Sigma_k` (m²) from its member map
points (build details below). For a scan point `x` transformed to `y = R*x + t`, let `q = y - mu_k`
(`k` = the voxel `y` falls in). The (unnormalized) Gaussian density at `y` is
`exp(-1/2 * q^T Sigma_k^-1 q)`.

### The mixture model and the d1/d2 derivation (Biber & Straßer 2003; Magnusson 2009)

A real scan is not ALL inliers — dynamic objects, multipath, and sensor glitches produce points
that belong to no voxel's true surface (this project's `TRUE_OUTLIER_FRACTION`, kernels.cuh's
`kAssumedOutlierRatio`). Model the density at a point as a two-component mixture:

```
p(x) = c1 * exp(-1/2 * q^T Sigma^-1 q)  +  c2
```

`c1 = 10*(1 - outlier_ratio)` (an inlier weight — PCL's exact constant), `c2 = outlier_ratio /
resolution^3` (a uniform outlier DENSITY spread over one voxel's volume — `resolution` enters here,
which is why `d1`/`d2` are recomputed per resolution level). We want a smooth, closed-form
approximation to `-log p(x)` that a Newton optimizer can differentiate cheaply. Because `p(x) ->
c2` as `q -> infinity` (pure outlier regime), define the SHIFTED target that vanishes there:

```
target(m) = -log(p(x)) - (-log(c2)),   m = q^T Sigma^-1 q   (Mahalanobis distance^2)
          = -log( (c1/c2)*exp(-m/2) + 1 )
```

Approximate `target(m)` by the functional form `-d1*exp(-d2/2 * m)` (which correctly -> 0 as
`m -> infinity`), matching VALUES at two points — `m=0` (perfect alignment) and `m=1` (one
Mahalanobis sigma, PCL's choice of second point):

```
m=0:  -d1 = target(0) = -log(c1/c2 + 1)              =>  d1 = log((c1+c2)/c2)
m=1:  -d1*exp(-d2/2) = target(1) = -log((c1/c2)*exp(-1/2) + 1)
      =>  d2 = -2*log( log((c1/c2)*exp(-1/2)+1) / d1 )
```

**Both `d1` and `d2` must come out POSITIVE**: `score(x) = -d1*exp(-d2/2*m)` has to be MOST
NEGATIVE at `m=0` (best possible alignment — minimizing this score means maximizing fit) and rise
toward 0 as `m` grows. `kernels.cuh`'s `ndt_compute_d1_d2()` implements exactly the closed form
above — see "Numerical considerations" for a real sign bug this exact derivation caught during
development.

### The gradient (chain rule through R*x+t into the Mahalanobis form)

Per-point score `S_i(p) = -d1*exp(-d2/2*m_i(p))`, `m_i(p) = q_i^T C^-1 q_i`, `q_i = y_i(p) - mu`
(mu treated as constant — the voxel ASSIGNMENT is frozen during one linearization, exactly like
ICP freezes correspondences, THEORY §Numerical considerations below discusses the consequence).

```
dS_i/dp_k = -d1 * exp(-d2/2*m_i) * (-d2/2) * dm_i/dp_k
dm_i/dp_k = 2 * q_i^T C^-1 * dq_i/dp_k = 2 * (C^-1 q_i)^T * J_{:,k}     (C^-1 symmetric)

  =>  dS_i/dp = d1*d2*f_i * J_i^T b_i,     f_i = exp(-d2/2*m_i),  b_i = C^-1 q_i
```

`J_i` (3x6, `dq_i/dp = dy_i/dp` since `mu` is constant) is 01.17's EXACT rotation-Jacobian formula,
reused for a 3-vector residual instead of a 2-vector reprojection:

```
J_i = [ -[R*x_i]_x  |  I_3 ]      (cols 0-2: omega: -[RP]_x, the LEFT-perturbation formula;
                                    cols 3-5: v: identity)
```

### The Hessian (Gauss-Newton term MINUS a curvature correction)

Differentiate `dS_i/dp_k` again w.r.t. `p_l`. The Gauss-Newton approximation drops the second
derivative of `J` itself (the same approximation 02.06's ICP and 01.17's calibration make — valid
because `J` is evaluated once per linearization, at the CURRENT pose, not as a function of `p`):

```
d2(S_i)/dp_k dp_l  =  d1*d2*f_i * [ (J_i^T C^-1 J_i)_{kl}  -  d2 * (J_i^T b_i)_k * (J_i^T b_i)_l ]
```

The first term, `J^T C^-1 J`, is EXACTLY ICP's Gauss-Newton shape (with `C^-1` playing the role of
a per-point weight matrix instead of the identity — NDT's Hessian generalizes point-to-point ICP's
`J^T J` to an anisotropic, per-voxel-weighted version). The SECOND term is new: it comes from the
Gaussian's OWN curvature (the second derivative of `exp(-d2/2*m)` w.r.t. `m`) and has NO analogue in
ICP's linear residual. Because it is SUBTRACTED, it can push an eigenvalue of `H` negative far from
the optimum — NDT's Hessian is **not guaranteed positive semi-definite**, unlike ICP's. This is the
single most consequential mathematical difference between the two algorithms' optimization
behavior, and it drives the damping-scheme choice below.

### Multi-resolution smoothing — why coarse voxels widen the basin

A coarse voxel spans more physical area per Gaussian, so (a) its covariance is typically LESS
anisotropic (a 2 m voxel is more likely to include a corner, a doorway edge, or floor+wall together
than a 1 m voxel is, giving it a rounder, less needle-like covariance — THEORY continues this in
"Numerical considerations"), and (b) each voxel's Gaussian, being wider, has a SLOWER-decaying
`exp(-d2/2*m)` in absolute position space (fewer voxels means the `resolution^3` term inside `c2`
is larger, which — working through `ndt_compute_d1_d2` — increases `d1` and decreases `d2` at
coarse resolution relative to fine: this project measures `d1=4.80,d2=0.218` at 2.0 m vs.
`d1=2.77,d2=0.363` at 1.0 m). A smaller `d2` means the Gaussian bump decays MORE SLOWLY with
Mahalanobis distance — literally a wider attraction basin in the objective landscape. Running
coarse first lets the optimizer find the right NEIGHBORHOOD with a forgiving objective, before
fine resolution sharpens the estimate. THEORY §How we verify correctness reports the MEASURED
effect: 65% vs. 7.5% cohort convergence at the same total iteration budget, at this scene's
smallest perturbation bin.

## The algorithm

1. **Build the voxel grid(s)** from the map cloud (once, reused across every registration): for
   each map point, `voxel_index()` (dense, direct — see "The GPU mapping") locates its voxel;
   accumulate count + position sum (PASS 1); finalize means; accumulate centered outer products
   (PASS 2); finalize each voxel's covariance — regularize (floor small eigenvalues), invert.
   Voxels with fewer than `kMinPointsPerVoxel` (5) points are marked invalid (the NDT analogue of
   ICP's correspondence rejection). Complexity: `O(N)` two passes over `N` map points plus
   `O(V)` finalize work over `V` voxels — `V << N` always (this project: `V`=196/58, `N`=40,000).
2. **Assemble score/gradient/Hessian** at the current pose estimate: for each scan point, transform
   by the current `(R,t)`, look up its voxel (O(1)), compute `m`, `f`, the 3x6 Jacobian, and fold
   its contribution into a running 28-scalar `[H21|g6|score]` record. Complexity: `O(n)` in the
   scan point count `n`, embarrassingly parallel across points.
3. **Damped Newton step**: solve `(H + lambda*diag(|H|)) * delta = -g` (sign-safe scaled damping,
   "Numerical considerations" explains why), retract, ACCEPT only if the new pose's score improves
   (else grow `lambda` and retry the SAME `H`/`g` — classic Levenberg-Marquardt, and REQUIRED here
   because `H` can be indefinite, unlike Gauss-Newton systems that always accept).
4. **Repeat step 2-3** up to `kMaxItersCoarse` (12) at the coarse grid, then up to `kMaxItersFine`
   (15) at the fine grid, starting from the coarse stage's result.
5. **The ICP contrast** (`icp_point_to_point_cpu`, compact reimplementation, 02.06 cited for the
   full treatment): the SAME retraction and 6x6-solve machinery, but a brute-force nearest-neighbor
   correspondence search each iteration (`O(n*m)`, `n`=scan points, `m`=target points) instead of a
   voxel lookup, and `H = J^T J` (always PSD, so Marquardt's classic multiplicative damping applies
   — the direct textbook contrast to NDT's sign-safe additive scheme).

## The GPU mapping

### Dense, direct-indexed grid — not hashed (a deliberate departure from 02.01)

02.01 hashes because a raw LiDAR SCAN's occupied-voxel set is sparse and, in principle, unbounded
(a streaming point cloud could span an unknown extent). An NDT MAP is the opposite: built ONCE from
a bounded, KNOWN survey area. Even this project's FINE (1.0 m) grid is only `17*9*4 = 612` voxels —
three orders of magnitude smaller than a hash table would ever need to be at this scale. Direct
indexing (`voxel_index()`: three `floor()`s, a bounds check, one linear-index multiply-add) is O(1)
with NO probe loop, NO collision handling, and NO `atomicCAS` insert path — strictly simpler AND
faster than hashing here. A city-scale HD map (kilometers, not meters) would flip this trade back
toward hashing or spatial tiling — README "Limitations" says so.

### The voxel-build kernels — point-parallel scatter, voxel-parallel gather

Four kernels, in the only correct order (means must finalize before covariance can center on them):
`ndt_voxel_accum_sum_kernel` (point-parallel, `atomicAdd(double*)`) -> `ndt_finalize_means_kernel`
(voxel-parallel, trivial divide) -> `ndt_voxel_accum_cov_kernel` (point-parallel, `atomicAdd(double*)`
again) -> `ndt_finalize_cov_kernel` (voxel-parallel: 3x3 Jacobi eigensolve, regularize, invert).
`atomicAdd` on `double*` has been native SASS since compute capability 6.0 — sm_75 and above pay no
emulation penalty, so there is no performance reason to fall back to float accumulation here (unlike
the assembly kernel below, where float IS the measured right trade).

### The 28-scalar reduction (01.17's exact packing, cited)

`ndt_assemble_kernel`: one thread per scan point computes its local `[H21(21) | g6(6) | score(1)]`
contribution in registers, writes it to a `blockDim.x * 28`-float SHARED memory buffer, and a
stride-halving tree reduction (`O(log blockDim.x)` steps) folds all threads' rows into one
28-scalar row per block — `block_partials`. The host finishes the cross-block sum in DOUBLE
precision (the same "GPU float-reduced, host double-summed" split every assembly-kernel project in
this repo uses). Why a tree, not per-thread atomics into one global record? 28 separate atomic
contention points per block vs. `O(log n)` parallel, contention-free adds — and the tree keeps the
GPU path's OWN rounding bit-reproducible within a block (only the intentionally-unordered
cross-block sum lives in double, where reordering sensitivity is negligible).

### Memory access pattern — a genuine gather, unlike 08.01's broadcast reads

Every thread in a warp reads the SAME `scan_xyz[i]` layout (coalesced), but each thread's
TRANSFORMED point can land in a DIFFERENT voxel — `grid.mean`/`grid.inv_cov6` reads are a
data-dependent GATHER, not a broadcast (unlike 08.01's uniform `u_nom[t]` read, which every thread
reads at the SAME address). At this project's scale (196 valid fine voxels, ~12.5 KB of mean+
inv_cov6 data) the whole active voxel table fits comfortably in L2 cache, so the gather's latency
is largely hidden — a much bigger map would need to think harder about this access pattern
(README Exercise territory).

## Numerical considerations

### Two-pass mean/covariance, not Welford, not the naive one-pass trick

Three ways to compute a running covariance: (1) the NAIVE one-pass raw-second-moment trick
(`E[x^2] - E[x]^2`, computed from `sum(x)` and `sum(x^2)` alone) — the fastest, but subtracts two
similarly-sized large numbers when the mean is far from the origin (this project's map spans x in
[0,16] m; a wall voxel's raw `sum(x^2)` at x~13 m with a tiny true variance is a textbook
catastrophic-cancellation setup); (2) Welford's single-pass, numerically-stable running update —
elegant for a SEQUENTIAL accumulator, but its parallel/batched merge formula adds real complexity
for a payoff (avoiding one extra kernel launch + one extra pass over the map) this project's scale
does not need; (3) TWO-PASS (mean first, then centered covariance) — reads every point twice, but
each pass is a plain sum of well-scaled (centered, for pass 2) quantities, sidestepping cancellation
entirely. This project uses (3), in DOUBLE precision throughout the accumulation, and documents
the trade rather than hiding it.

### Covariance regularization — why thin voxels (walls!) are exactly the risky ones

A voxel containing a genuinely flat surface patch has a raw sample covariance with one eigenvalue
near ZERO (no true variation along the surface normal, once sensor noise floors it). Inverting that
covariance directly would blow the Mahalanobis distance up arbitrarily for any point with even a
tiny offset along that direction — mathematically correct (an infinitely confident constraint) but
numerically catastrophic (a single-precision float cannot represent that confidence, and the
resulting Hessian entries saturate). `regularize_and_invert_cov3()` floors every eigenvalue to
`kEigenFloorRatio * (largest eigenvalue)` before inverting — this project measured
`kEigenFloorRatio=0.001` (PCL-typical) as TOO aggressive for its cohort (near-degenerate wells that
trapped the optimizer within millimeters of a bad local minimum) and settled on `0.01` after
measuring the cohort convergence rate at both settings — a real, reported tuning decision, not an
assumed default.

### Two real bugs this project's own gates caught during development

1. **A sign bug in `ndt_compute_d1_d2`.** An earlier version of the `d1` formula (transcribed
   incorrectly from memory before the from-scratch re-derivation above was written down) produced a
   NEGATIVE `d1`. The symptom was dramatic and immediate: `score_sanity` failed (score at ground
   truth was NOT the best score — everything was inverted), and cohort convergence was 0%. The fix
   is the closed-form re-derivation above, verified by hand against the numeric value BEFORE being
   committed.
2. **A damping-scale-then-symmetry bug in the Newton solve.** After fixing (1), convergence was
   still poor (~5%): a flat `lambda*I` damping addition, with `lambda` a FIXED constant, was
   negligible against this project's `H` diagonal (which reaches `1e5`-`1e7` — inv_cov along a
   near-planar voxel's normal direction is large BY DESIGN, see above), so the FIRST Newton step
   from any real offset was essentially undamped and overshot. Scaling `lambda`'s START by the
   assembled Hessian's own max diagonal magnitude helped (5.6% -> 16.7%... actually the FIRST fix
   alone reached 16.7% only combined with proper accept/reject; see the commit history in
   `kernels.cuh`'s comments) but a SECOND, subtler issue remained: NDT's 6 parameters do not share
   one physical scale — rotation entries carry units of "meters per radian" (`dy/domega ~ |R*x|`,
   points meters away), translation entries are dimensionless-in-meters (`dy/dv = I`) — so a SINGLE
   flat `lambda` could be simultaneously too large for one block and too small for the other. The
   fix, in `cholesky6_solve_flat`: `A_ii += lambda * max(|H_ii|, floor)` — scaled PER PARAMETER
   (Marquardt's adaptivity) but ADDED as a positive quantity regardless of `H_ii`'s sign
   (Levenberg's sign-safety, required because `H` can be indefinite — Marquardt's own
   `(1+lambda)*H_ii` would make a NEGATIVE diagonal entry MORE negative, the wrong direction).
   Combined with proper accept/reject step control, cohort convergence went from ~5% to a STABLE
   13-17% (the exact figure has moved slightly across this project's history as the cohort sample
   size and a scan-generation bug, below, were fixed — never because of an iteration budget
   change). Raising the iteration budget further (12/15 -> 20/25, and again to 60/80 during the
   finisher pass below) did NOT change the result — the remaining un-converged trials are genuine
   local minima of this scene's objective at these perturbation magnitudes, not an
   iteration-starved optimizer (README "Limitations" repeats this).

### Hessian conditioning in the corridor — the degeneracy this project measures, not just asserts

The `degenerate_axis` `[info]` report (main.cu STAGE I, never gated pass/fail — 01.17's degeneracy-
gate lineage, cited by name) computes `H` at the ground-truth pose from two subsets of the SAME
scan: points within 4 m of the sensor (mostly corridor walls/floor — the room and pillar excluded)
versus the FULL scan (room + pillar visible). Measured on this project's data: the near-field-only
Hessian has condition ratio **~346**, its weakest eigenvector loading almost entirely (0.99) on
`vx` — translation ALONG the corridor's long axis, exactly the "sliding down a featureless hallway"
degeneracy a human intuition predicts. The FULL scan's condition ratio drops to **~97** once the
far room wall and pillar are visible (their much greater range gives large leverage on the
translation estimate along the corridor's axis) — a measured, not asserted, demonstration that
GEOMETRIC DIVERSITY, not just point COUNT, is what resolves a degenerate direction.

### A finisher pass's diagnosis: the smallest-perturbation bin IS the corridor-axis degeneracy

A lead review of this project asked a sharp question: production NDT, started from a NEAR-TRUTH
initial guess (this project's smallest bin, 0.2 m / 5°), should converge close to 100% of the time
on a well-structured scene — so why did this scene's smallest bin measure only ~65-67%? A finisher
pass instrumented every unconverged trial (`main.cu`'s `failure_diagnosis` `[info]` line) and found
the failures were NOT dominated by the weak `vz` axis STAGE I names for the FULL scan (only
12/208 = 6% of ALL failed trials, across every bin, were "Z-only blockers" — cases where the XY/
rotation error alone would have passed but Z drift alone pushed the trial over the threshold; Z is
not even perturbed by the cohort generator, so any Z error here is optimizer DRIFT, not
unconverged recovery). Re-running one representative failing trial's FULL score trajectory with
**5x the iteration budget** (60 coarse + 80 fine, vs. the shipped 12+15) left its final score
UNCHANGED to two decimal places at BOTH resolution stages — conclusive evidence of a genuine
stationary point, not an iteration-starved one.

The real explanation was geometric, and simple once measured: `main.cu`'s `bin0_corridor_axis_split`
`[info]` line classifies each smallest-bin trial's initial XY offset DIRECTION as within 20° of the
corridor's long axis (0°/180°) or not, and reports separate convergence rates for each group.
Measured on this project's committed cohort: perturbations ALONG the corridor axis converge **0%**
(0/9); every other direction converges **84%** (26/31) — almost exactly the "≥85-90%" a near-truth
NDT registration should achieve, once the ONE direction this scene's own `degenerate_axis` report
already names as weak is excluded. This is not a special case invented to explain away a bad
number — it is the SAME degeneracy STAGE I measures, showing up directly in the basin/convergence
gate a chapter later, and it is exactly the failure mode a real Autoware-style localizer handles
by fusing NDT's pose with wheel odometry (which DOES observe forward travel along a corridor) — a
generalization of this project's own existing "roll/pitch/Z come from a wheel/IMU prior, not the
LiDAR match" scoping note (README "Limitations", `scripts/make_synthetic.py`'s cohort-generation
comment) to the horizontal axis a corridor scene makes degenerate.

### Honest negative results: what did NOT move the smallest-bin number

Before accepting the corridor-axis explanation above, the finisher pass tried every lever CLAUDE.md
§13's "measure, do not guess" discipline suggests, each REVERTED after measurement showed no
durable improvement (all numbers below are cohort-wide convergence, this scene, this cohort):

- **Assumed outlier ratio (`kAssumedOutlierRatio`).** Swept 0.05 (matching the TRUE injected
  fraction) through 0.80. The shipped value (0.40) was already at or near a local optimum in every
  sweep; nearby values were WORSE (0.30 -> 8.9%, 0.05 -> 8.9%, vs. 0.40's 13.3-16.7% depending on
  cohort/RNG state at the time of the sweep) and there was no smooth trend to exploit — a real,
  reported non-result, not a tuning opportunity.
- **Fine voxel size.** Tried 0.5 m (half the shipped 1.0 m): overall convergence WORSENED slightly
  (16.7% -> 14.4% at the time of this specific test) and the SAME trials that were stuck at 1.0 m
  were still stuck, unmoved, at 0.5 m — direct evidence the failures are not a voxel-boundary
  discretization artifact.
- **Coarse voxel size.** Tried 1.5 m (down from the shipped 2.0 m): convergence COLLAPSED to 1.1%
  — the coarse stage's WIDE basin (the entire reason multi-resolution exists, "Multi-resolution
  smoothing" above) depends on the coarse Gaussian being wide, and 2.0 m is load-bearing, not an
  arbitrary round number.
- **Eigenvalue floor ratio (`kEigenFloorRatio`).** Tried 0.05 (up from the shipped 0.01, already
  itself a measured choice per "Covariance regularization" above): convergence COLLAPSED to 0.0%.
  0.01 is a genuine local optimum for this scene, confirmed a second time.
- **Accept/reject retry budget and damping granularity.** Tried raising the per-iteration retry cap
  from 8 to 40 (no change at all to the specific stuck trials) and separately tried a much finer
  damping escalation (`kLambdaUp` 10.0 -> 2.0, more retries): convergence WORSENED (16.7% -> 8.9%)
  — gentler damping let the optimizer wander into DIFFERENT, generally worse, local minima rather
  than reach the same ones faster.
- **A backtracking line search along the Newton direction** (standard in production NDT/LM solvers,
  tried two ways: replacing the accept/reject retry entirely, and as a last-resort fallback only
  after the existing retry budget was exhausted). The REPLACEMENT variant reshuffled which
  individual trials converged (some rescued, some newly broken) for a net-negative overall change;
  the LAST-RESORT fallback never once found an improving point along the least-damped direction for
  any trial that the existing scheme had already given up on — independent confirmation that those
  trials are at genuine stationary points in every nearby direction tried, not merely poorly
  reached ones. Neither variant is in the shipped code (CLAUDE.md's "teaching beats cleverness":
  unproven complexity earns no place here).
- **A REAL fix that WAS kept: the outlier RNG desync bug.** `scripts/make_synthetic.py`'s
  `generate_scan()` used to draw an outlier's replacement "wrong depth" from the SAME RNG stream
  every INLIER point's range noise also reads from — an extra draw taken only on the outlier
  branch silently desynchronized the "clean" and "with-outliers" paired scans' noise for every beam
  after the first outlier, even though the function's own docstring claimed beam-for-beam
  alignment. This produced a real, reproducible symptom: an early version of this project's
  `outlier_robustness` gate measured the WITH-outliers cohort converging MORE often than the
  outlier-FREE one (16.7% vs. 11.1%) — backwards, and a straightforward consequence of comparing
  two scans that were not actually a single-variable-toggled pair past their first ~20 beams, not
  evidence that outliers somehow help. The fix: give the outlier depth draw its own independent
  stream (`rng_outlier_depth`), so consuming it can never perturb the noise stream every inlier
  beam also reads from. Re-measured after the fix (and after the cohort size increase below, to
  give the comparison real statistical power): clean 13.8% vs. with-outliers 13.3% — nearly
  identical, and now falling on the EXPECTED side (outliers degrade, if anything, not improve).
- **Cohort sample size.** `COHORT_TRIALS_PER_BIN` was raised from 15 to 40 (90 -> 240 total trials)
  because the smallest bin's 15-trial sample was demonstrably too small to trust: a single trial
  landing near the corridor-degenerate axis is worth 6.7 percentage points on its own. Re-measuring
  at 40 (and, during tuning, spot-checked at 60) trials/bin gave a materially more STABLE reading
  (65-68% across sample sizes) — this did not change the ACHIEVABLE rate, but it made the
  MEASUREMENT trustworthy, and it is what makes the on-axis/off-axis split above (9 and 31 trials
  respectively) a large enough sample to draw a real conclusion from.

### Precision choices, summarized

`float` (FP32) for all per-point geometry (positions, Jacobians, the score/gradient/Hessian
per-point math) — matches every other project in this repo and keeps the transcendental `expf()`
call (the assembly kernel's dominant per-point cost) cheap. `double` (FP64) for: voxel-build
accumulators (many-point sums, real cancellation risk — see above), the cross-block/cross-point
REDUCTION of H/g/score (summing many similarly-signed float terms), and all Newton-solve bookkeeping
(`lambda`, the 6x6 Cholesky itself — a `~50` FLOP solve, utterly free next to the point-parallel
assembly, so there is no performance reason to stay in float there).

## How we verify correctness

Nine gated stages (`main.cu`), each printing a stable `PASS`/`FAIL` line:

- **`VOXEL_STATS_TWIN`** — the GPU-built grid (both resolutions) vs. `build_ndt_grid_cpu`'s
  independent two-pass sequential accumulator: EXACT agreement on which voxels are valid and their
  point counts (pure integer bookkeeping); tight relative tolerance (mean 1e-3, inv_cov 1e-2 — the
  looser inv_cov bound accounts for the Jacobi eigensolve's own iterative rounding) on the
  regularized statistics. Measured: mean rel dev 5.9e-8, inv_cov rel dev 1.5e-6 — both far inside
  tolerance; the bound exists for a real bug, not to pass a coincidence.
- **`JACOBIAN_CHECK`** — the CALCULUS gate (01.17's exact discipline): the analytic gradient
  (`ndt_assemble_cpu`, containing the chain-rule `J`/`H` code) vs. a CENTRAL-DIFFERENCE numeric
  gradient computed by calling ONLY `ndt_total_score_cpu` (which shares the score formula but
  contains none of the analytic Jacobian code) at perturbed poses. A real numerical subtlety this
  gate's implementation had to handle: NDT's per-point score is only PIECEWISE smooth —
  `voxel_index()` is a `floor()`-based step function, so a point within an eps-induced shift of a
  voxel FACE can flip which Gaussian it scores against, a genuine discontinuity. The gate excludes
  points within a documented safety margin (1 cm, an order of magnitude above the worst-case
  eps-induced shift) of any voxel boundary — THEORY, not a swept-under-the-rug tolerance loosening.
  Measured worst relative deviation: 1.7e-3 (tolerance 5e-2).
- **`ASSEMBLY_TWIN`** — one score/gradient/Hessian assembly at the ground-truth pose, full
  `scan_main` cloud (5,122 points), GPU block-reduced vs. CPU sequential-double. Measured worst
  relative deviation: 1.8e-6 (tolerance 5e-3).
- **`TRAJECTORY_TWIN`** — one full coarse->fine trajectory, GPU-orchestrated
  (`run_ndt_multires_gpu`) vs. independently-written CPU control flow (`run_ndt_multires_cpu`),
  measured-then-margined (08.01's technique): both converge to the SAME final pose (measured: GPU
  rot 1.283°/trans 0.0771 m vs. CPU IDENTICAL to 4 significant figures — 0° rotation deviation,
  2.1e-6 m translation deviation, comfortably inside the tolerance) — proper accept/reject damping
  made this MUCH tighter than an early always-accept version, whose GPU/CPU paths diverged into
  different local minima entirely (see "Numerical considerations"). The twin trial is now FOUND by
  searching the cohort for its first bin-2 (0.8 m / 15°) member rather than a raw index — a raw
  index ("trial 30") silently pointed at a DIFFERENT bin once `COHORT_TRIALS_PER_BIN` changed from
  15 to 40, a real bug the finisher pass caught and fixed (`main.cu`'s STAGE D comment tells the
  story); this trial happens to converge cleanly, unlike an earlier (mis-indexed) version of this
  gate whose displayed trajectory plateaued at a failed registration.
- **`SCORE_SANITY`** — a free monotonicity check: score at ground truth beats score at all 240
  cohort initial guesses. Measured: 0/240 violations.
- **`CONVERGENCE` / `ACCURACY`** — the 240-trial cohort, NDT multi-resolution: measured 32/240 =
  13.3% converged (floor gate >=10%, comfortable margin below the measured, stable value); among
  converged trials, mean 51.3 mm / 1.02°, worst 77.1 mm / 2.67° (gate <100 mm / <4°).
- **`BASIN_CONTRAST`** — the SAME 240 trials, NDT multi-resolution (13.3%) vs. NDT fine-only at the
  IDENTICAL total iteration budget (1.2%) vs. the compact ICP contrast (24.2%) — gated on
  multi-resolution being at least as robust as fine-only (a design-guaranteed claim, verified with
  a small slack for cohort-level noise); ICP's own number is reported honestly, not gated in
  either direction, because "does NDT beat ICP" is the thing this project measures, not assumes.
  Measured HONESTLY: in this project's small, simple, single-room-scale scene, ICP is MORE
  sample-efficient than NDT overall (24.2% vs. 13.3% converged) and more accurate among its
  converged trials besides (36.6 mm / 0.73° vs. NDT's 51.3 mm / 1.02° — an `[info]`-only
  `accuracy contrast` line main.cu prints alongside `basin_contrast` for exactly this comparison).
  A finisher pass tried every honest lever available (see "Numerical considerations"'s "Honest
  negative results") to change this and could not, honestly, make NDT dominate this particular
  scene's cohort — the real, taught lesson is WHY not (correspondence search is genuinely cheap and
  accurate at this map's scale, `m`=724 points; NDT's `O(1)`-per-point advantage over ICP's
  `O(n*m)` only pays for itself once `m` is large enough that brute-force search becomes the
  bottleneck — "Where this sits in the real world" and README "Limitations" say this plainly, not
  as an excuse but as the actual scientific finding).
- **`OUTLIER_ROBUSTNESS`** — the same cohort, WITH vs. WITHOUT the documented 5% outlier fraction
  (a beam-aligned "wrong-depth return" paired scan, `scripts/make_synthetic.py`): measured mean
  converged error is essentially unchanged (54.0 mm clean vs. 51.3 mm with outliers — different
  trials converge in each run, so the two means are not directly comparable point-for-point), while
  the WORST converged error rises modestly (77.8 mm clean vs. 77.1 mm with outliers — nearly
  identical) and convergence rate is nearly identical too (13.8% clean vs. 13.3% with outliers) — a
  real, GRACEFUL, and (unlike an earlier buggy version of this measurement, "Honest negative
  results" above) properly ISOLATED comparison: same 240 trials, same paired beam-for-beam noise,
  differing ONLY in whether outliers were injected.

## Where this sits in the real world

- **Autoware `ndt_scan_matcher`** — the production ROS 2 node this project's system context names
  directly. Differences from this teaching core: a real map is millions of points (Autoware
  typically pre-voxelizes with `pcl::VoxelGrid` and stores the NDT structure persistently, not
  rebuilding it per run); initial pose comes from GNSS or a previous localization result, not a
  blind perturbation cohort (this project's whole basin study is a stand-in for "how forgiving is
  the initial-guess requirement," a question a real system answers once at startup, not every
  tick); multi-threaded voxel search (`ndt_omp`) rather than this project's dense array; and a
  health-monitoring layer around the raw NDT score (PRACTICE.md §3 covers this).
- **PCL `NormalDistributionsTransform`** — the reference open-source CPU implementation whose
  `outlier_ratio_`/`resolution_` parameters this project's `kAssumedOutlierRatio`/`kLeafFine` mirror
  directly; its Newton solve differs in damping details but shares the same `d1`/`d2` mixture math.
- **ICP family (02.06, this repo)** — the direct algorithmic contrast this project measures against
  on identical data. Measured HONESTLY, this project's small, simple, single-room-scale scene does
  NOT favor NDT overall: ICP converges more of the cohort (24.2% vs. NDT's 13.3%) and is more
  accurate among its converged trials (36.6 mm/0.73° vs. 51.3 mm/1.02°) — most visibly at the
  smallest-perturbation bin, where an accurate nearest-neighbor correspondence beats a discretized
  voxel lookup outright (100% vs. 65% convergence at 0.2 m / 5°, and even that 65% is itself mostly
  explained by the corridor-axis degeneracy above, not a voxelization weakness). The teaching point
  this project's own measurements support is narrower than "NDT wins basin width": NDT's real
  advantage is `O(1)`-per-point voxel lookup vs. ICP's `O(n*m)` brute-force correspondence SEARCH —
  a cost that is invisible at this project's scale (`m`=724 ICP target points) and becomes decisive
  once the map is large enough that a nearest-neighbor search itself is the bottleneck (a city-block
  HD map, not a single building) — see "Where this sits in the real world" and README "Limitations"
  for the honest scope of that claim, and Exercise 4 for how to measure the crossover directly by
  GPU-accelerating the ICP contrast and re-running both algorithms against a much larger map.
- **Learned localizers** — named honestly, not dismissed: place-recognition networks (e.g.,
  learned global descriptors feeding a coarse initial guess — the role 02.11's scan-context
  descriptor plays classically) and end-to-end learned point-cloud registration (e.g., predicting a
  pose directly from a scan/map pair) are an active research direction that can widen the "initial
  guess" basin further than multi-resolution NDT alone — an open problem this teaching core does
  not attempt, consistent with CLAUDE.md §13's "state the real frontier, do not fake it" guidance.
