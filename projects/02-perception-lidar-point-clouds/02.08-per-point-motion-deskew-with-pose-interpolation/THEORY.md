# 02.08 — Per-point motion deskew with pose interpolation: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**A spinning LiDAR is a rolling shutter, in 3-D, at mechanical speed.** A mechanical LiDAR spins its
beam array around a vertical axis, firing a burst of laser pulses at fixed azimuth increments and
measuring range from time-of-flight. One full revolution — a "sweep" — takes `T_sweep` seconds, set by
the motor's spin rate: a common 10 Hz unit sweeps in 100 ms, a 20 Hz unit in 50 ms. **Firing is
sequential in azimuth, not simultaneous.** Beam group `k` fires at azimuth step `a`, at time

```
t(a) = t0 + (a / AZIMUTH_STEPS) * T_sweep,     a = 0 .. AZIMUTH_STEPS-1
```

exactly the same physical fact project [`01.10`](../../../01-perception-cameras-vision/01.10-rolling-shutter-correction-using-imu-rates/THEORY.md)
teaches for a CMOS camera sensor, whose rows expose sequentially rather than all at once — **read that
project's "The problem" section as this one's twin in camera form.** The two problems share one
sentence: *a sensor that samples the world over a nonzero time window, while something moves, produces
data that is only self-consistent if you know WHEN each sample was taken and undo the motion between
samples.* Where they differ is the geometry of the fix:

| | 01.10 Rolling shutter | 02.08 Motion deskew |
|---|---|---|
| What samples sequentially | image ROWS (position fixed by pixel geometry) | LiDAR BEAMS (position determined by firing time via azimuth) |
| What moves between samples | the CAMERA (extrinsic-only: pure rotation, this project's scope) | the PLATFORM (full SE(3): translation AND rotation) |
| The fix's shape | a per-ROW homography (`K R K^-1`) applied to pixel RAYS | a per-POINT rigid transform applied to already-metric 3-D points |
| Why translation matters here but not there | camera translation over ~25 ms of readout is sub-millimeter (negligible) at typical handheld/vehicle rates; LiDAR sweeps (~100 ms) are 4x longer and this project's speeds (up to 15 m/s) make TRANSLATION the dominant term, not just rotation | — |

Every point this project's LiDAR returns is measured in the sensor's **own, instantaneous body frame at
its firing time** — not in one shared frame. If the platform carrying the sensor is stationary, that
distinction is invisible: every firing-time frame IS the same frame. If the platform moves, it is not,
and naively concatenating raw per-point returns (exactly what a driver's packet does, and exactly what
this project calls the **"skewed"** scan) silently assumes a shared frame that does not exist. A flat
wall, sampled over 100 ms by a platform that moved 1.5 m during that time, is recorded as a set of
points spread across a *range* of true wall positions — the wall appears to thicken or slant.

**The engineering constraint a real robot imposes.** The fix — motion deskew — must complete well
inside the NEXT sweep's period, because it sits between the driver and everything else in the pipeline
(README "System context"): a 10 Hz spin rate allows ~100 ms; a 20 Hz unit allows ~50 ms. Miss that
budget and the perception stack falls behind the sensor, a much worse problem than any single frame's
distortion.

### The distortion-magnitude table (why this is not a minor correction)

For a platform translating at speed `v` (m/s) and rotating at rate `omega` (rad/s), the WORST-CASE
displacement/rotation baked into a single sweep of duration `T_sweep` is simply `v * T_sweep` and
`omega * T_sweep` — first-order, but the right order of magnitude to reason about whether deskew
matters. At `T_sweep = 100 ms` (this project's value, a realistic 10 Hz spin rate):

| Platform speed | Translation over one sweep | Yaw rate | Rotation over one sweep |
|---|---|---|---|
| 1 m/s (slow AMR) | 0.10 m | 0.5 rad/s (gentle turn) | 0.05 rad (2.9°) |
| 5 m/s (brisk walk / warehouse cart) | 0.50 m | 2.0 rad/s (moderate turn) | 0.20 rad (11.5°) |
| **15 m/s (this project's `straight`/`arc` cohorts, ≈54 km/h)** | **1.50 m** | **6.0 rad/s (this project's `arc` cohort)** | **0.60 rad (34.4°)** |
| 30 m/s (highway, ≈108 km/h) | 3.00 m | — | — |

A 1.5 m translation error, or a 34° orientation error, is not a rounding-level defect — it is larger
than most obstacles a robot needs to avoid, and larger than the width of most corridors
[`02.07`](../../02.07-ndt-scan-matching/README.md) NDT registers against. This is why motion deskew is
not optional above walking speed, and why this project measures — never assumes — exactly how much it
recovers (see "How we verify correctness").

## The math

**State: a rigid pose trajectory.** The platform's pose over the sweep is a function of time,
`(p(t), q(t))`: `p(t)` the WORLD position (m), `q(t)` a UNIT quaternion, repo order `(w,x,y,z)`,
"body expressed in world" (`T_world_body` — [`09.01`](../../../09-dynamics-kinematics/09.01-batched-forward-kinematics/src/kernels.cuh)'s
convention, cited; `docs/SYSTEM_DESIGN.md` §3.3/3.4). This project never has access to the CONTINUOUS
function — only to `n` discrete samples of it, `{(t_i, p_i, q_i)}`, in ascending time order. Two
"regimes" of that discretization are compared side by side: DENSE (`n = 21`, ~200 Hz-equivalent spacing
across the 100 ms sweep) and SPARSE (`n = 2`, just the sweep's start and end sample).

**Interpolating between two samples.** Given a query time `t` bracketed by samples `i` and `i+1`, let
`alpha = (t - t_i) / (t_{i+1} - t_i) ∈ [0,1]`. Position interpolates by **LERP**:

```
p(t) ≈ (1 - alpha) * p_i + alpha * p_{i+1}
```

This is EXACT when the platform's true velocity is constant over `[t_i, t_{i+1}]` — position lives in
the flat vector space R³, and a constant-velocity trajectory IS a straight line in that space, which
LERP reproduces exactly (this project's `straight` cohort proves this numerically — see "How we verify
correctness").

**Why rotation cannot use the same trick — the curved-manifold argument.** Unit quaternions representing
3-D rotations live on `S³`, the 4-D unit sphere — a CURVED manifold, not a flat vector space.
Component-wise LERP between two quaternions, `(1-alpha)*q_i + alpha*q_{i+1}`, computes a point on the
CHORD connecting them through the interior of the sphere, not a point ON the sphere — renormalizing that
chord point projects it back onto the sphere, but along a path that does not trace the sphere's
geodesic (the shortest ON-SPHERE path, which is what a rotation trajectory at constant angular velocity
actually traces). For small angular gaps between samples the chord and the geodesic nearly coincide
(this project's fallback below exploits exactly that); for the tens-of-degrees gaps the SPARSE regime
can span, they diverge measurably — this is a geometric fact, not a numerical inconvenience, and it is
why **SLERP** (Spherical Linear intERPolation) exists.

**SLERP, derived.** Two unit quaternions `q_i, q_{i+1}` separated by geodesic angle
`theta = acos(dot(q_i, q_{i+1}))` (their 4-D dot product; the "how far apart" measure on the sphere).
The interpolant that moves along the geodesic at CONSTANT ANGULAR SPEED is:

```
slerp(q_i, q_{i+1}, alpha) = [sin((1-alpha)*theta) * q_i + sin(alpha*theta) * q_{i+1}] / sin(theta)
```

This is the unique curve on `S³` connecting the two quaternions with constant angular velocity — the
exact rotational analogue of LERP's constant LINEAR velocity on a flat space. It is why this project's
`arc` cohort (constant yaw RATE — a genuinely constant angular velocity) sees SLERP between just its
START and END samples reproduce the TRUE continuous orientation EXACTLY at every intermediate instant:
a constant-angular-velocity rotation traces the identical geodesic SLERP interpolates (measured in "How
we verify correctness" below).

**The double cover (why sign matters).** The map from unit quaternions to rotations is 2-to-1: `q` and
`-q` represent the IDENTICAL rotation (negate all four components; the rotation matrix
`R(q) = R(-q)` is unchanged — check the `quat_to_mat3`-style formula, every term is quadratic in the
components). This means `dot(q_i, q_{i+1})` can be NEGATIVE even when `q_i` and `q_{i+1}` represent
nearby rotations, simply because one of them happened to be expressed with the "opposite sign" of the
same physical orientation. Interpolating the raw formula above in that case walks the LONG way around
the 4-sphere — a rotation of MORE than the true angle between them, sometimes wildly wrong. The fix:
if `dot(q_i, q_{i+1}) < 0`, negate `q_{i+1}` (this changes nothing about what it represents) before
interpolating — guaranteeing the SHORTEST path. [`src/kernels.cuh`](src/kernels.cuh)'s `quat_slerp`
implements exactly this, and `main.cu`'s `SLERP_CORRECTNESS` gate exercises it directly with a designed
sign-flipped pair.

**The reference frame and the per-point transform.** Fix a reference instant `t_ref` — this project
uses the SWEEP END, `t_ref = T_sweep` (not the start; see the callout below for why the choice barely
matters mathematically but does matter for the freshness of the output). Let `(p_i, q_i)` be the
interpolated pose at a point's own firing time `t_i`, and `(p_ref, q_ref)` the interpolated pose at
`t_ref` (computed ONCE per sweep). A point measured as `P_local` in the sensor's own frame at `t_i` has
WORLD position `P_world = p_i + R(q_i) * P_local`. Re-expressing that SAME world point in the reference
sensor's frame:

```
P_ref = R(q_ref)^-1 * (P_world - p_ref)
      = R(conj(q_ref)) * (p_i - p_ref)  +  R(conj(q_ref)) * R(q_i) * P_local
      = t_rel                            +  R(q_rel) * P_local
```

using `R(a)^-1 = R(conj(a))` for a unit quaternion, and `R(a) * R(b) = R(a ⊗ b)` for the Hamilton
product `⊗` (this repo's convention):

```
q_rel = conj(q_ref) ⊗ q_i          (the relative ROTATION, ref -> i)
t_rel = R(conj(q_ref)) * (p_i - p_ref)   (the relative TRANSLATION, expressed in the ref frame)
```

This is `deskew_one_point` in [`src/kernels.cuh`](src/kernels.cuh), verbatim.

**Why the choice of `t_ref` does not change how WELL deskew works, only which frame the answer lands
in.** Every point undergoes the identical two-step recipe (interpolate its own pose, compose against the
reference pose) regardless of which instant is chosen as reference — the interpolation ERROR at a given
point's firing time is unaffected by where `t_ref` sits. What changes is only the CONSTANT rigid offset
applied to every point in the sweep (moving `t_ref` from start to end just re-anchors the whole output).
This project picks the sweep END because (a) it is the freshest ego-pose estimate available when the
scan finishes arriving — a downstream planner reacting to "this scan, right now" is reacting to the most
current position, and (b) it matches the convention most real drivers use when stamping a scan's
timestamp (PRACTICE.md §1).

## The algorithm

Per sweep (the numbered steps are labeled in `main.cu`):

1. **Load** the sweep's raw points — each with a firing time `t_i` and local coordinates `P_local` —
   and the trajectory samples covering `[t0, t_ref]` (either regime).
2. **Compute the reference pose ONCE**: `interpolate_pose(traj, n, t_ref)` → `(p_ref, q_ref)`.
3. **For every point, in parallel** (this is the pure map — see "The GPU mapping"):
   a. `find_bracket_index` — binary search for the two trajectory samples bracketing `t_i`.
   b. `interpolate_pose` — LERP the position, SLERP the orientation, at `t_i`.
   c. `deskew_one_point` — compose `q_rel`/`t_rel` against the reference pose and apply the rigid
      transform to `P_local`.
4. **Output**: every point, now expressed in the single reference frame — safe to hand to a consumer
   that assumes one frame per cloud (README "System context").

Complexity per point: O(log n) for the bracket search (n = 2 or 21 here — see "The GPU mapping" for why
this never dominates), O(1) for the interpolation and transform (a fixed handful of FMAs, one `acosf`,
two `sinf`). Total: O(N log n) for N points, fully parallel across points.

## The GPU mapping

```
one thread = one point i:
  read t_points[i], xyz_local[i]     (coalesced: adjacent threads, adjacent addresses)
  read g_traj[...]                   (UNIFORM broadcast: every thread reads the SAME bytes,
                                       from __constant__ memory — see kernels.cu's g_traj comment)
  read ref_pose                      (passed BY VALUE: a small POD broadcast, no pointer chase)
  compute deskew_one_point(...)      (registers only: ~15 floats live at once, no spilling
                                       at this function's size)
  write xyz_out[i]                   (coalesced)
grid = ceil(N/256), block = 256      (repo-default warp multiple; kernels.cu's launch comment)
```

No shared memory (points share no data with each other — the embarrassingly-parallel property the
catalog bullet names explicitly), no atomics (each point's output is independent), and — worth stating
honestly — **no meaningful occupancy story at this project's scale**: a few thousand points per cohort
is nowhere near enough work to saturate an RTX-class GPU's thousands of cores; the measured 0.03–0.15 ms
per kernel launch (README "Expected output") is dominated by launch/synchronization overhead, not
compute. The teaching point of this project's GPU mapping is not "look how fast" — it is "look how
TRIVIAL the parallel version is once the per-point math is right", and that a real full-density sweep
(hundreds of thousands of points, this repo's usual N) would scale this SAME kernel, unchanged, straight
into a regime where the GPU's parallelism is the only way to stay inside a 50 ms budget.

**Why the bracketing-sample binary search does not dominate.** `find_bracket_index` runs `O(log n)`
comparisons against `g_traj`'s time field — at `n = 21` (DENSE) that is ~5 comparisons; at `n = 2`
(SPARSE) it is a single branch. Both are noise next to even ONE point's global memory transaction. The
teaching value is the PATTERN, not the speedup: a production deskew node ingesting a longer window of a
200 Hz pose stream (n in the hundreds) is exactly where `O(log n)` binary search starts mattering over a
linear scan — README Exercise 4 asks the reader to find that crossover by profiling it directly, rather
than trusting this claim.

**`__constant__` memory for the trajectory (cite 09.01/01.10's precedent).** Every thread in a launch
reads the identical trajectory bytes — a textbook broadcast pattern, and the same reasoning
[`09.01`](../../../09-dynamics-kinematics/09.01-batched-forward-kinematics/src/kernels.cuh)'s robot
model and [`01.10`](../../../01-perception-cameras-vision/01.10-rolling-shutter-correction-using-imu-rates/src/kernels.cuh)'s
per-row LUT both give for the same memory-space choice. `kernels.cu` re-uploads `g_traj` between the
dense and sparse regimes for the SAME cohort — mirroring 01.10's "upload once per variant, launch many"
shape, because this project's whole point is comparing what changes when the uploaded trajectory's
RESOLUTION changes.

## Numerical considerations

- **SLERP's small-angle fallback IS the Taylor limit, not a separate approximation.** As two
  quaternions converge (`dot -> 1`), the exact formula's `sin(theta)` denominator vanishes and the
  computation becomes an unstable `0/0` (undefined exactly at `dot == 1` — the common case of two
  IDENTICAL consecutive samples, e.g. this project's `stationary` cohort, where EVERY sample pair is
  identical). Since `sin(x) ≈ x` for small `x` (first-order Taylor), the exact weights
  `sin((1-t)*theta)/sin(theta) -> (1-t)` and `sin(t*theta)/sin(theta) -> t` as `theta -> 0` — i.e. plain
  LERP-then-renormalize is the ANALYTIC LIMIT of SLERP, not a different, cruder thing standing in for
  it. `kernels.cuh`'s `quat_slerp` switches to this branch below `dot > 0.9995` (≈1.8° geodesic gap),
  chosen with headroom over float rounding noise and comfortably below every non-degenerate gap this
  project's dense-regime trajectories produce.
- **Quaternion normalization drift (CLAUDE.md §12).** Every `quat_mul`/`quat_slerp` result is
  renormalized before use (`quat_normalize`, called inside both) — chained products drift off the unit
  sphere from float rounding; left unchecked over many compositions this would corrupt `R(q)`'s
  orthogonality. This project's chains are short (one `quat_mul` per point, in `q_rel`'s composition) so
  drift is negligible per-call, but the discipline is applied unconditionally, not "when it seems to
  matter" — the same rule CLAUDE.md §12 states for the whole repo.
- **Timestamp precision: float vs. double seconds, the ULP arithmetic.** Firing times in this project
  span `[0, 0.1]` s. FP32 has a 24-bit significand; at magnitude ~0.1, one ULP is
  `2^-24 * 2^floor(log2(0.1)) ≈ 2^-24 * 2^-4 ≈ 3.7e-9` s — under 4 nanoseconds. The fastest motion this
  project models (the `wiggle` cohort's peak angular rate, ≈55 rad/s) would need a timing error of
  order `1e-6` s to bias an angle by `1e-4` rad — three orders of magnitude looser than one FP32 ULP at
  this timescale. **Decision: float32 timestamps throughout** (matching the point coordinates'
  precision, and every trajectory sample's), not double — the precision loss is unmeasurable at this
  project's speeds and durations, and using one FP32 pipeline end-to-end (no host-side double
  round-tripping) keeps the GPU/CPU twin comparison honest (both paths round identically). A vehicle
  sweeping for MINUTES instead of 100 ms, with timestamps measured from process start rather than
  sweep-relative, is the regime where double-precision (or a sweep-relative float, as here) genuinely
  matters — see PRACTICE.md §1 for how real drivers avoid this trap.
- **Determinism.** There is no runtime randomness anywhere in this pipeline (the ONLY randomness in the
  whole project is the fixed-seed range noise baked into the committed sample file at generation time —
  `../scripts/make_synthetic.py`, seed 42). Every gate's measured number is therefore reproducible
  bit-for-bit on a given machine; the only source of run-to-run variation is FMA-fusion/intrinsic
  differences between nvcc's device code and cl.exe/nvcc's host code — bounded by the VERIFY gate's
  tolerance (measured worst case: 2.4e-6 m).

## How we verify correctness

This project's twin comparison (GPU kernel vs. `reference_cpu.cpp`) is structurally **blind to bugs
inside the shared math** (`quat_slerp`, `quat_rotate`, `deskew_one_point` in `kernels.cuh`) — both paths
call the IDENTICAL functions, so an error there would reproduce identically on both sides and the twins
would agree perfectly while both being wrong (the independence ruling `reference_cpu.cpp`'s file header
states in full). This project therefore carries FIVE gates in two tiers:

**Tier 1 — the twin (`VERIFY`):** catches indexing, launch-configuration, host/device data-marshaling,
and memory-layout bugs — anything where the GPU and CPU paths could plausibly DISAGREE despite sharing
the same math. Measured worst deviation across 4 cohorts × 2 regimes: **2.4e-6 m** (tol 1e-4 m).

**Tier 2 — independent, analytic-ground-truth gates**, each targeting a DIFFERENT failure mode the twin
cannot see, because each compares against `../scripts/make_synthetic.py`'s continuous trajectories or a
closed-form formula — neither of which routes through `deskew_one_point` at all:

- **`SLERP_CORRECTNESS`** — catches a wrong SLERP formula, a missing double-cover fix, or a broken
  small-angle fallback, independent of any point data: a >90° quaternion pair's geodesic-angle
  progression (measured error: 5.6e-7 rad) and sign-flip invariance (measured: 1.2e-7).
- **`IDENTITY_CONTROL`** (stationary cohort) — catches any bug that makes deskew ADD motion where none
  exists (an off-by-one in the reference pose, an accidental extra rotation). Measured max displacement:
  **0.0 m** (exact — every trajectory sample is bit-identical in this cohort, so the relative transform
  is the identity to float precision by construction).
- **`RESTORATION`** (straight/arc/wiggle) — catches a deskew that runs but produces the WRONG geometry,
  by comparing against the analytic instantaneous-truth twin every point carries. The undeskewed
  baseline is reported alongside as a negative control — it MUST be dramatically worse, or the cohort is
  not actually testing anything:

  | Cohort | Undeskewed mean error | Dense-deskewed mean error | Improvement |
  |---|---|---|---|
  | straight | 0.746 m | 0.000000 m | exact (constant velocity + constant heading) |
  | arc | 2.318 m | 0.000187 m | ~12,400× |
  | wiggle | 1.938 m | 0.096 m | ~20× |

- **`SAMPLING_LESSON`** — catches a deskew that is only "accidentally" correct because the trajectory
  happened to be sampled finely enough, by comparing the DENSE and SPARSE regimes directly:
  - wiggle cohort: sparse-regime mean error (1.852 m) is **19.2×** the dense-regime mean error
    (0.096 m) — the 2-sample regime, missing 2.5 oscillation cycles entirely, barely improves on the
    undeskewed baseline (1.938 m) at all.
  - straight cohort: dense and sparse regimes AGREE (both 0.000000 m to float precision) — a
    consistency check that only holds because constant velocity makes 2-sample interpolation exact
    (see "The math"); if it failed, something in the interpolation code path itself would be broken.
- **`DOWNSTREAM_PAYOFF`** — a compact PCA plane-fit RMS ("How production robotics stacks measure a flat
  surface's thickness", 02.03 ground-segmentation's lineage) on a wall slice of the straight cohort:
  skewed **0.239 m** RMS thickness → deskewed **0.0056 m** (matching the analytic truth's 0.0056 m
  exactly) — a concrete, measured answer to "why does this matter downstream" beyond an abstract error
  number.

## Where this sits in the real world

- **LOAM's lineage** (Zhang & Singh 2014) established the decoupled shape this project builds: deskew a
  sweep using the best available motion estimate (LOAM uses the PREVIOUS sweep's own odometry output as
  that estimate — a bootstrap this project sidesteps by assuming the trajectory is simply given),
  THEN register the deskewed cloud. Every LOAM-descended stack (LeGO-LOAM, LIO-mapping, and many
  commercial AMR/AV stacks) inherits this two-stage structure.
- **The production frontier is TIGHT coupling.** LIO-SAM (Shan & Englot 2020) and FAST-LIO/FAST-LIO2
  (Xu et al. 2021/2022) fold motion compensation INSIDE the state estimator itself: the IMU's own
  high-rate propagation supplies the per-point pose (replacing this project's precomputed trajectory
  samples with a continuously-updated one), and the scan-matching residual feeds back into the SAME
  filter that produces the next sweep's undistortion — deskew and localization become one iterated
  loop, not two separate stages. This project's decoupled version is the right teaching entry point
  (the interpolation math is identical) but is NOT what a modern production LiDAR-inertial stack ships.
- **Driver-level packet timestamps, honestly.** This project assumes every point carries a usable firing
  time. Real drivers vary: Velodyne/Ouster/Hesai packet formats DO expose per-point (or per-firing-group)
  timestamps, often PTP-synchronized to the vehicle's clock domain — but not every ROS/ROS 2 driver
  wrapper propagates that field into `PointCloud2`'s optional `time`/`t` channel, and some older or
  simpler LiDAR products expose only a per-PACKET (not per-point) timestamp, requiring a coarser
  approximation of firing time from azimuth alone (still better than none — PRACTICE.md §1 has the full
  honesty on this).
