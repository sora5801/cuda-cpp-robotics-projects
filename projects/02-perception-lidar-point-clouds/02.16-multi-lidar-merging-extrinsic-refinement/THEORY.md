# 02.16 — Multi-LiDAR merging + extrinsic refinement: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**Why one LiDAR is never enough.** A spinning 360° LiDAR mounted on a vehicle's roof sees almost
everything — except its own vehicle. The car's own hood, hull, or chassis blocks a wedge of the
field of view directly in front of and behind the sensor at close range, and tall/wide vehicles
create additional near-field blind zones along their flanks. Corner-mounted units fill exactly these
gaps: a front-left and front-right LiDAR, angled outward, cover the near-field zones the roof unit
cannot reach, at the cost of a narrower field of view each. `kernels.cuh`'s rig diagram documents
this project's concrete geometry: MAIN (roof, 360°) plus LEFT/RIGHT (front corners, ~55° outward
yaw, 180° FOV wedge each). The two corner sensors' wedges overlap only in a narrow forward cone —
directly behind the vehicle is a real, physical blind zone even with all three sensors, exactly
matching production AV designs (README "System context" and PRACTICE.md §1).

**Why mounts drift.** A LiDAR bracket is a rigid mechanical joint between the sensor housing and the
vehicle body — bolted, sometimes shimmed, occasionally bonded. Three physical mechanisms move it by
a small, slow amount over a fleet vehicle's service life:

- **Vibration-induced fastener loosening.** Bolted joints under cyclic vibration (road input, panel
  resonance) can lose preload over months; a bracket that was torqued correctly at installation can
  develop a fraction of a degree of play as the joint's friction lock degrades. This is a textbook
  mechanical-engineering failure mode (self-loosening under transverse vibration, the mechanism
  Junker's classic vibration-test work characterizes), not unique to robotics.
- **Thermal cycling.** A sensor housing and its mounting bracket are rarely the same material (an
  aluminum or steel bracket, a polymer/aluminum sensor housing); differential thermal expansion
  across a day-night or seasonal cycle applies small cyclic stress to the joint, which — combined
  with vibration — accelerates the loosening above. Over enough cycles, a residual, non-recovering
  offset can remain.
- **Impact and handling.** A curb strike, a cargo-bay bump during a sensor swap, or a wash-bay water
  jet can nudge a bracket by more than either mechanism above in one event — the same story
  01.17's PRACTICE.md tells for camera rigs, applied here to LiDAR mounts.

This project's synthetic rig plants a **~0.8° / 3 cm** drift on the two corner sensors (a different
vector per sensor — `kernels.cuh`'s `kDrift` table) — an illustrative, not measured, magnitude
consistent with "a bracket that has quietly loosened," chosen to be large enough to matter for
downstream perception (a 0.8° angular error over a 10 m range is a **14 cm** positional error at
that range — see "The math" below) and small enough that it would NOT be obvious from a quick visual
inspection of the mount.

**The engineering constraint this creates.** A fleet cannot practically re-run a factory calibration
rig (a checkerboard target array, a controlled bay, a technician) every time a sensor might have
drifted — that does not scale to hundreds of vehicles running continuously. The industry answer is
**online/continuous calibration**: use the vehicle's own sensing of the world it is already driving
through as the calibration target, running in the background, flagging (or auto-correcting) drift
without ever taking the vehicle out of service. That is exactly the detect-then-refine loop this
project implements in miniature.

## The math

### Notation

SI units throughout; `T_parent_child` names a rigid transform "child expressed in parent"
(SYSTEM_DESIGN.md §3.3). `R` is a row-major 3×3 rotation, `t` a 3-vector translation, so
`x_parent = R * x_child + t`. `Exp(ω)` is the SO(3) exponential (Rodrigues' formula) of an
axis-angle vector `ω` (radians): `R = I + sin(θ)/θ [ω]_x + (1-cos θ)/θ² [ω]_x²`, `θ = |ω|`.

### The merge transform

Trivial by design: `x_base_i = T_base_lidar_i * x_lidar_i` for every point, per sensor `i`. The
whole teaching content of "merging" is NOT this formula — it is what happens next.

### Deduplication

Two (or three) sensors observing the same physical surface produce two (or three) INDEPENDENT
samples of it, each with its own noise realization and (if the extrinsic is wrong) its own
systematic offset. Concatenating without deduplication inflates point density non-uniformly — dense
in overlap zones, sparse elsewhere — which biases every downstream density-sensitive algorithm
(voxel occupancy counts, nearest-neighbor statistics, ground-plane RANSAC inlier counts). Voxel-grid
deduplication (02.01's Method-B hashing, cited) hashes each point into a fixed-size cell and keeps
one representative per occupied cell — a map (hash) into a sort-and-compact spatial index, not an
optimization.

### Plane fitting (PCA)

Given a set of `m` points `{p_j}` believed to lie on one plane, the total-least-squares fit is: compute
the centroid `c = (1/m) Σ p_j`, the mean-shifted covariance `Cov = (1/m) Σ (p_j - c)(p_j - c)ᵀ`, and
eigendecompose. The eigenvector of the SMALLEST eigenvalue is the plane normal `n` (the direction of
least variance — the direction "the data barely moves in," which for a flat surface plus isotropic
noise is the surface normal); `c` is a point on the plane. This is exactly 02.09's per-point normal
estimation pipeline, applied here at ZONE granularity (hundreds of points spanning meters) instead of
per-point KNN-neighborhood granularity (tens of points spanning centimeters) — same math, different
scale.

### The drift observable: plane-pair residuals

This is the "linearized sensitivity" this project is built on. Let a sensor's TRUE extrinsic be a
small perturbation of its NOMINAL (believed) one:

```
R_true = Exp(δω) · R_nom          (a LEFT/world-frame rotation perturbation)
t_true = t_nom + δt
```

A raw sensor-frame point `p` on a TRUE world plane `(n, d)` (i.e. `n·x_true = d`) satisfies
`x_true = R_true·p + t_true`. If we (incorrectly) transform `p` through the NOMINAL extrinsic
instead — exactly what happens when a drifted sensor's data is merged using its stale calibration —
we get `x_believed = R_nom·p + t_nom`. To first order in the small quantities `δω, δt`:

```
x_true - x_believed ≈ [δω]_x · (R_nom p) + δt = [δω]_x · q + δt,      q := x_believed - t_nom
```

(`[δω]_x` is the skew-symmetric cross-product matrix of `δω` — this is EXACTLY the same
`[R·p_src]_x` term `point_to_plane_residual_and_jacobian` in `kernels.cuh` uses for the refinement's
Jacobian; detection and refinement are two views of the same linear operator, one run in reverse.)

Substituting into the true-plane equation and rearranging, the BELIEVED points satisfy, to first
order:

```
n · x_believed  ≈  d  -  n·δt  -  δω·(q × n)
```

Two regimes fall out of this one line:

- **Near a fixed reference point `q₀`** (say the zone's centroid), the constant part of the
  right-hand side, `-n·δt - δω·(q₀×n)`, is an **apparent OFFSET shift** — this is what
  `plane_pair_residual()`'s `offset_m` measures (the perpendicular distance from one sensor's fitted
  centroid to another sensor's fitted plane).
- The part that VARIES with in-plane position (`Δq = q - q₀`), `-δω·(Δq×n)`, is a **linear function
  of position across the plane** — which is precisely what a plane fit reads as a NORMAL TILT. This
  is what `angle_deg` measures.
- **The offset term scales with RANGE.** `|δω·(q₀×n)| ≤ |δω| · |q₀| · sin(∠(q₀,n))` — an angular
  drift produces an apparent offset that grows with distance from the sensor to the observed
  surface. At this project's ~8-12 m wall ranges, the planted **0.8° (≈ 0.014 rad)** drift alone
  predicts an offset contribution on the order of `0.014 × 10 m ≈ 14 cm` for a plane seen nearly
  edge-on to the range vector — consistent in ORDER OF MAGNITUDE with this project's measured
  drifted-rig offsets (tens of millimeters to several centimeters, `demo/out/plane_residuals.csv`;
  the exact number also depends on the angle between the range vector and the plane normal, which
  the simple bound above does not resolve). This range-amplification is why a LARGE-baseline scene
  (walls tens of meters out, not a tabletop rig) is a much MORE sensitive drift detector for angular
  error than translation error alone would suggest.
- **A single plane only measures a PROJECTION of `(δω, δt)` onto its own normal direction.** Fitting
  one plane and comparing it to a trusted reference recovers, at best, one scalar combination of the
  six drift parameters (dominated by the normal-aligned component of `δt` and the tilt-inducing
  components of `δω`) — never the full 6-DOF drift. This directly motivates the observability
  discussion below and the refinement's need for multiple non-parallel zones.

### Point-to-plane refinement (Levenberg-Marquardt)

Given a source point `p` (raw sensor frame) and a FIXED target plane `(n, c)` (base frame, from a
trusted sensor's fit), the residual and its Jacobian with respect to a local 6-vector
`δ = [δω; δv]` at the current estimate `T` (02.06's point-to-plane linearization, applied to a
zone-fixed target instead of a searched nearest neighbor):

```
r(T) = n · (R·p + t - c)                                    (signed distance, m)
J    = [ n·(-[R·p]_x)  |  n ]                                (1x6 row: d(r)/d(δω), d(r)/d(δv))
```

Stacking every point in the active zone set gives an over-determined system; Gauss-Newton/
Levenberg-Marquardt solves `(H + λ diag(H)) δ = -g` with `H = ΣJᵀJ`, `g = ΣJᵀr`, iterating the
retraction `R ← Exp(δω)·R`, `t ← t + δv` (01.17's exact update rule) until `δ` is tiny. `kernels.cuh`
carries the full hyperparameter table (`kMaxLmIters=20`, Marquardt damping schedule) — 01.17's
battle-tested numbers, unchanged, since this is the identical 6×6 SE(3) local-optimization shape.

## The algorithm

1. **Load** two cohorts (aligned/drifted), each `(sensor_id, surface_id, x, y, z)` rows, raw sensor
   frame.
2. **Fit** every needed plane: for each (sensor, cohort) pair, transform to a common frame (usually
   base, via that sensor's currently-believed extrinsic) and PCA-fit a plane per surface — O(n) per
   fit, embarrassingly parallel across points, one thread per point until the tiny final reduction.
3. **Detect**: compare MAIN's (trusted) plane against each side sensor's plane, per shared zone —
   O(zones) plane-pair residual evaluations, essentially free.
4. **Refine**: Levenberg-Marquardt, `kMaxLmIters=20` iterations max, each iteration one GPU assembly
   pass over the active zone's points (`O(n)` per pass, `O(n·iters)` total) plus an O(1) 6×6 host
   solve. **Why zone-assigned correspondence, not a nearest-neighbor search (02.06's approach)**: this
   project's scene is genuinely piecewise-planar and a point's owning surface never changes as `T` is
   refined (unlike ICP proper, where the transform can move a point from one region of a complex
   shape to another between iterations) — searching for the nearest point on a fixed set of ALREADY-
   FITTED planes would just rediscover the same zone assignment every time, at real extra cost and
   zero benefit. This is a documented, honest simplification, not a shortcut around the hard part.
5. **Validate**: re-fit the refined sensor's plane, re-run the SAME residual computation as step 3.
6. **Observability**: re-run the assembly at a RESTRICTED zone mask (one plane instead of three),
   compare `H`'s condition number.
7. **Merge + dedup**: transform every sensor's full cloud (map), voxel-hash + sort + compact (the
   dedup pipeline).

Complexity is linear in point count at every stage except the tiny (6×6, 3×3) linear-algebra
kernels, which are O(1) per call — the whole pipeline is bandwidth-bound, not compute-bound, exactly
like a real onboard perception pipeline at much larger point counts.

## The GPU mapping

Four distinct patterns, worth reading as a set (kernels.cu's file header argues this explicitly):

- **Pure map** (`transform_points(_multi)_kernel`): one thread per point, no cooperation, no shared
  state. The simplest possible GPU kernel — deliberately as trivial as the template's own SAXPY,
  because merging's REAL difficulty is not the transform.
- **Map into a bounded scatter-reduce** (`accumulate_centroid_kernel`/`accumulate_covariance_kernel`):
  thousands of threads, `atomicAdd` into just `kNumSurfaces = 6` output slots. Atomics are the RIGHT
  tool here specifically because the output is tiny and the operation runs ONCE per (sensor, cohort)
  — contention is bounded (at most 6 hot addresses) and there is no per-iteration cost to amortize.
- **Map-then-tree-reduce** (`assemble_point_to_plane_kernel`): the OPPOSITE choice, for the OPPOSITE
  reason. This reduction produces ONE 28-wide record and runs potentially dozens of times per
  refinement (every LM iteration, both the "at current T" and "at candidate T" evaluations) — the
  same 28-wide slot hammered by up to `kThreadsReduce=128` threads per block, over and over. A
  shared-memory tree reduction (01.17/02.06's block-tree-reduce-then-host-finishes-in-double split,
  cited) avoids the contention atomics would create at that call frequency. Memory hierarchy: GLOBAL
  for the read-only point/surface arrays (coalesced, one point per thread) and the block-partial
  output; SHARED for the `blockDim.x * 28` float scratch (14 KiB at `kThreadsReduce=128`,
  comfortably under the 48 KiB budget); REGISTERS for each thread's own 28-scalar accumulator before
  the copy into shared memory.
- **Sort-and-compact** (the dedup pipeline): `thrust::stable_sort_by_key` (a library call, used
  because sorting itself is not this project's teaching content — the VOXEL-GRID DEDUP ALGORITHM
  built on top of it is, CLAUDE.md §1's "no black boxes" rule applied honestly: we say explicitly
  what the library call computes and why we did not hand-roll it) followed by a boundary-scan kernel
  and a `thrust::copy_if` compaction (02.01/02.09's identical pattern, cited).

## Numerical considerations

- **Mean-shifted covariance, not the textbook one-pass formula.** `Cov = E[pp^T] - mean·mean^T` loses
  precision catastrophically when points sit far from the coordinate origin (this project's points
  are 8-12 m from the vehicle) but are tightly clustered locally (the whole point of "local"
  covariance) — the classic cancellation 02.09's THEORY.md derives in full. This project's
  `accumulate_centroid_kernel` / `accumulate_covariance_kernel` two-pass split avoids it.
- **The point-to-plane linearization's validity range.** The Jacobian above is exact for the RESIDUAL
  gradient (it does not approximate the plane, only the effect of a SMALL retraction step), but
  Gauss-Newton's implicit assumption — that one linear step gets you close enough for the next
  step's linearization to still be valid — degrades if the INITIAL error is large. This project's
  ~0.8°/3 cm drift is comfortably inside that range (measured: convergence in well under
  `kMaxLmIters=20` iterations, `demo/out/gates_metrics.csv`); README Exercise 2 asks what happens if
  you widen it.
- **Why several gates print rotation deviations of exactly 0.0000°.** `RECOVERY_LEFT`/`RIGHT`,
  `TRAJECTORY_TWIN`, and `ZERO_DRIFT_CONTROL` all measured a rotation deviation that printed as
  literally zero on the reference run. This is not a bug: with hundreds to nearly two thousand points
  per zone and 6 mm range noise, the LEAST-SQUARES rotation estimate's uncertainty is far smaller
  than FP32's angular resolution near identity — `rotation_angle_deg()` computes `acos((trace(R_err)
  -1)/2)`, and for a genuinely tiny `R_err`, `1 - cos(θ) ≈ θ²/2` underflows FP32's ~1.19e-7 epsilon
  well before `θ` reaches the ~1e-4° range where a human would call the difference "measurable" — the
  `acos` clamp then returns exactly 0. `src/main.cu`'s tolerance-block comment documents this
  explicitly and keeps every affected tolerance modestly above zero (both for honesty — "below our
  instrument's resolution" is a weaker claim than "exactly zero" — and for headroom against
  cross-GPU-architecture FMA reordering).
- **Reduction order is not deterministic run-to-run.** Both the atomics-based plane-fit accumulation
  and (to a lesser extent, since it is a fixed block-tree order) the refinement assembly can complete
  their additions in a different order between runs — visible in this project's own repeated runs as
  small (≤ 1 ULP-ish, always well inside the documented tolerance) drift in the printed `[info]`
  numbers. This is expected and does not affect any stable (gated) line.
- **Angle wrapping.** Every angle in this project (`rotation_angle_deg`, `plane_pair_residual`'s
  `angle_deg`) is a magnitude in `[0°, 180°]` from an `acos`, never a signed angle — no wrapping
  concerns arise.

## How we verify correctness

Two tiers, per the repo's twin-independence ruling (`reference_cpu.cpp`'s file header states the
full ruling as applied to this project):

- **GPU-vs-CPU twins** (`TRANSFORM_TWIN`, `PLANE_FIT_TWIN`, `ASSEMBLY_TWIN`, `TRAJECTORY_TWIN`,
  `DEDUP_ACCOUNTING`) — the ACCUMULATION LOOPS and, for plane fitting, the EIGENSOLVERS are
  independently written on each side; every one of the SHARED closed-form primitives
  (`point_to_plane_residual_and_jacobian`, `so3_exp`, the voxel-key packing) is exactly the kind of
  "system under test whose hand-duplication would be pure transcription" the ruling permits sharing.
  Every tolerance in `src/main.cu` is measured-then-margined: run once, read the actual worst-case
  deviation off the `[info]` line, then set the threshold with documented headroom (2-15x for most
  gates; see the tolerance block's comment for the handful that are kept modestly above an
  effectively-zero measurement instead).
- **The independent gate on the shared formula itself**: `RECOVERY_LEFT`/`RIGHT` and
  `ZERO_DRIFT_CONTROL` compare a refined extrinsic against ground truth computed by
  `scripts/make_synthetic.py` — a COMPLETELY SEPARATE program, a different language, sharing no code
  with the C++ refinement path at all. If `point_to_plane_residual_and_jacobian` had a sign error,
  every twin gate above would still pass (both sides share the bug), but the refined extrinsic would
  NOT converge toward the independently-generated true drift, and `RECOVERY_*` would fail. This is
  01.17's exact "zero-noise sanity gate is also the independent check on the shared camera-model
  formula" argument, restated for this project's geometry.

Edge cases exercised: the ALIGNED cohort (zero drift — `DRIFT_DETECTION`'s control side and
`ZERO_DRIFT_CONTROL`'s whole purpose), a deliberately RANK-DEFICIENT zone set (`OBSERVABILITY`'s
wall-front-only solve), and a genuinely INDEPENDENT small-sample estimate (`LOOP_CONSISTENCY`'s
direct LEFT-RIGHT refinement, only ~120 points across two zones — reported `[info]` only, not gated,
precisely because its own observability is thinner than the main-anchored refinements').

## Where this sits in the real world

**Merging.** Autoware's `pointcloud_preprocessor` package concatenates multiple LiDAR topics into one
with essentially this project's transform-then-optionally-downsample shape, at production point
counts (hundreds of thousands of points per scan) and with additional deskew (02.08's job, assumed
upstream here) and self-filter (removing points that hit the vehicle's own body) stages this project
omits for scope.

**Extrinsic calibration/refinement.** Production AV and robotics stacks split calibration into two
economically very different regimes:

- **Factory/offline calibration** — a controlled bay, calibration targets (checkerboards,
  retroreflective spheres, or structured rooms), a technician and a fixture, run once at
  manufacturing/commissioning time. High accuracy, high labor cost per vehicle, does not scale to
  monitoring hundreds of vehicles continuously.
- **Continuous/online calibration** — exactly this project's detect-then-refine loop, run against
  whatever the vehicle observes during normal operation (buildings, road edges, other structure),
  usually offboard or on a fleet backend rather than in real time onboard. Lower per-instance
  accuracy than a factory rig (fewer, noisier, less-controlled observations), but runs continuously
  and catches drift AS IT HAPPENS rather than at the next scheduled service interval. Research in
  this space (plane/edge-based LiDAR-LiDAR and LiDAR-camera online calibration, e.g. CalibNet-style
  and factor-graph-based approaches) generalizes this project's single-pair point-to-plane solve into
  a joint pose-graph optimization over every sensor pair simultaneously — the "loop consistency"
  question this project states didactically (`[info]` only) but does not solve.

**The economics, honestly.** A factory recalibration event (bay time, technician labor, vehicle
downtime) costs real money per occurrence; an online calibration MISS (a drifted sensor silently
degrading perception for weeks before anyone notices) costs more, in the form of degraded
perception quality feeding every downstream safety-relevant decision. This is why fleet operators
increasingly treat calibration health as a MONITORED metric (this project's `DRIFT_DETECTION` gate,
generalized to a continuously-running dashboard) rather than a one-time manufacturing step — see
`PRACTICE.md` §3-4 for the deployment pattern and the fleet-maintenance-cost framing.
