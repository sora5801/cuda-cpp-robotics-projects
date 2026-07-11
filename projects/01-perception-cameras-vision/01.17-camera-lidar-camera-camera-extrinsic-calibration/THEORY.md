# 01.17 — Camera-LiDAR / camera-camera extrinsic calibration (batched reprojection-error optimization): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### What a LiDAR measures vs. what a camera measures

These two sensors are physically nothing alike, and that difference is *why* calibrating them together
is hard and *why* it needs to be accurate:

- A **LiDAR** is an *active, time-of-flight ranging* sensor. It fires a pulsed (or FMCW-modulated) laser
  and measures the round-trip time (or phase shift) to convert directly into a **range** along the beam
  direction; the beam's bearing comes from a rotating mirror or MEMS scanner's encoder. A LiDAR return is
  therefore a genuine 3-D measurement (range x bearing x elevation → Cartesian x,y,z) with error dominated
  by *timing jitter along the range axis* (millimeter-scale for a good time-of-flight LiDAR) and a much
  smaller angular error from encoder/beam-divergence (milliradian-scale).
- A **camera** is a *passive, bearing-only* sensor. Each pixel tells you the DIRECTION a ray of light
  arrived from — nothing about how far away its source was. A single camera frame carries no depth
  information at all; depth is only recoverable indirectly (stereo triangulation, motion, or — as here —
  a *known* 3-D structure like a calibration target).

Fusing them means overlaying a "distance-accurate, angle-so-so" sensor onto an "angle-accurate
(sub-pixel, via the lens), distance-blind" sensor. The only way to combine "LiDAR return at (x,y,z) in
the lidar frame" with "camera detection at pixel (u,v)" into one consistent belief about the world is to
know **exactly** where the LiDAR's frame sits relative to the camera's frame — the rigid transform
`T_camera_lidar` this project solves for. Camera-camera calibration solves the analogous problem for two
cameras that must jointly triangulate (stereo) or hand off detections (multi-camera rigs).

### Why "exactly" — the range-dependent error budget

A small extrinsic error does not stay small once you look far away. Split the error into a rotation part
(`delta_theta`, rad) and a translation part (`delta_t`, m), and consider a point at range `R` (m) from the
sensor:

- A **rotation** error tilts every observed ray by `delta_theta`, **regardless of range** — its pixel
  effect is roughly `fx * delta_theta` (px), constant with distance.
- A **translation** error shifts the sensor's assumed origin by `delta_t`, whose ANGULAR effect
  (`delta_t / R` rad) **shrinks with range** — its pixel effect is roughly `fx * delta_t / R` (px).

Worked example with a representative long-range automotive camera (`fx ~= 1000` px) at `R = 50` m: a
`0.1 deg` (`1.75e-3` rad) rotation error costs about `1000 * 1.75e-3 ~= 1.75` px, while a `10` mm
translation error costs only about `1000 * 0.01 / 50 ~= 0.2` px. **Rotation accuracy matters far more
than translation accuracy at range** — the opposite of most people's intuition, and the reason
production calibration tools (README "Prior art") report rotation in arc-minutes, not just translation
in millimeters.

### Why extrinsics drift — the physical causes, quantified

Nothing about a mechanical mount is perfectly rigid over time. Three physical effects this project's
`PRACTICE.md` §3 turns into a recalibration *procedure*:

- **Thermal expansion.** A sensor rig's mounting bar is commonly aluminum (`alpha ~= 23e-6 /K`, a typical
  6061-T6 value). Over a `deltaT = 40` K swing (a cold morning to a sun-baked afternoon on a vehicle
  roof — entirely realistic), a `L = 0.30` m baseline between two sensors changes length by
  `deltaL = L * alpha * deltaT = 0.30 * 23e-6 * 40 ~= 2.8e-4` m, **0.28 mm**. That is smaller than this
  project's own *measured* translation-recovery accuracy under moderate sensor noise (7.44 mm — see "How
  we verify correctness" below), so thermal expansion alone will not visibly break a calibration in one
  cycle — but it is a real, non-zero, *repeating* effect (every day, every season) that accumulates
  alongside the next two, and a rig spanning a longer baseline (say, front-to-rear LiDAR on a 4 m vehicle)
  sees the same strain scaled up proportionally.
- **Vibration-induced fastener loosening.** Bolted sensor mounts on a moving vehicle or legged robot see
  constant small-amplitude vibration; without thread-locking compound or belleville washers, bolt preload
  decays over thousands of vibration cycles, and a mount that was rigid at installation develops
  microscopic play — a rotation error, not (usually) a translation error, since play tends to show up as
  a small rocking freedom rather than a clean linear shift.
- **Crash/service events.** A single hard bump, a curb strike, or a service technician removing and
  reinstalling a sensor bracket can shift an extrinsic by DEGREES in one event — orders of magnitude
  larger than thermal drift, and the reason `PRACTICE.md` §3 lists "any service event touching a sensor
  mount" as an unconditional recalibration trigger, not just a scheduled one.

## The math

### Notation

- `p_src` (m): a 3-D point in the SOURCE sensor's frame (LiDAR frame for the camera-LiDAR scenario;
  camera 1's frame for the camera-camera scenario).
- `T_dest_src = (R, t)`: the unknown rigid transform (SYSTEM_DESIGN.md §3.3 naming: "source expressed in
  dest"), `R` a 3x3 rotation matrix, `t` a 3-vector translation (m), both expressed in the DEST frame.
- `K = (fx, fy, cx, cy)` (px): the DEST camera's known pinhole intrinsics (01.16's output, consumed here
  by name), in the OPTICAL frame convention (z-forward depth axis, x-right, y-down — the documented
  REP-103 exception SYSTEM_DESIGN.md §3.2 allows for camera optics).
- `uv_obs` (px): the observed pixel the correspondence's point projects to.

### The parameterization: SO(3) local coordinates and the retraction

`R` cannot be optimized directly with plain gradient steps — `R + deltaR` is not, in general, a valid
rotation matrix (it is not orthonormal). The standard fix, used throughout robotics and vision
(Barfoot's *State Estimation for Robotics*, cited in README "Prior art"), is to optimize in **local
coordinates** on the rotation manifold and **retract** back onto it after every step.

A rotation's local coordinates are a 3-vector `omega` (an *axis-angle* or "so(3) log-rotation" vector:
direction = rotation axis, magnitude `theta = |omega|` = rotation angle, radians). The map from local
coordinates to an actual rotation matrix is the **SO(3) exponential map**, Rodrigues' rotation formula:

```
R = Exp([omega]_x) = I + (sin(theta)/theta) [omega]_x + ((1-cos(theta))/theta^2) [omega]_x^2
```

where `[omega]_x` is the skew-symmetric ("cross-product") matrix of `omega`. This closed form comes from
solving the ODE `dR/dtheta = [axis]_x R` (a rotating frame's own defining equation) — see any robotics
Lie-group reference for the derivation; `kernels.cuh`'s `so3_exp()` implements exactly this formula,
including the small-angle branch (`theta < 1e-8`) that avoids the `0/0` in `sin(theta)/theta` as
`theta -> 0`, using the first-order Taylor truncation `R ~= I + [omega]_x` there (error `O(theta^2)`,
negligible below the branch threshold — see "Numerical considerations").

This project's LM solver perturbs the CURRENT estimate `(R, t)` by a **local 6-vector**
`delta = [omega; v]` (omega: rad, v: m) via the **retraction**:

```
R(delta) = Exp(omega) * R          t(delta) = t + v
```

This is the **decoupled SO(3) x R^3** retraction (rotation perturbed multiplicatively and EXACTLY via
Rodrigues; translation perturbed additively). It is *not* the fully-coupled se(3) exponential, which
would instead couple the translation through a second matrix `V(omega)`:

```
t(delta)_coupled = t + V(omega) * v,    V(omega) = I + ((1-cos theta)/theta^2)[omega]_x + ((theta-sin theta)/theta^3)[omega]_x^2
```

"Numerical considerations" below explains exactly why skipping `V(omega)` is a safe simplification for
this project's LM steps (which are small once near convergence) and names the coupled version this
simplifies (README Prior art's Barfoot reference derives it in full).

### The analytic reprojection Jacobian — "the classic 2x6," derived

The residual for one correspondence is
`r(delta) = project(R(delta) * p_src + t(delta); K) - uv_obs`, a 2-vector (px). LM needs
`J = d(r)/d(delta)`, a 2x6 matrix, at `delta = 0` (i.e., at the CURRENT estimate). Three steps:

**Step 1 — how the camera-frame point moves.** Let `Pcam = R*p_src + t` (the point in the dest camera's
frame at the current estimate) and `RP = R*p_src` (the ROTATED-ONLY point, i.e. `Pcam - t`). First-order
expand the retraction:

```
Pcam(delta) = Exp(omega)*R*p_src + t + v
           ~= (I + [omega]_x) * RP + t + v          (first-order Exp, valid near delta=0)
            = Pcam + [omega]_x * RP + v
```

Using the skew-symmetric identity `[a]_x b = -[b]_x a` (swapping the order of a cross product flips its
sign), `[omega]_x * RP = -[RP]_x * omega`. So:

```
d(Pcam)/d(omega) = -[RP]_x        (a 3x3 matrix — NOT -[Pcam]_x; the rotation perturbs R*p_src, and
                                    translation t is untouched by a rotation perturbation)
d(Pcam)/d(v)     =  I_3
```

**Step 2 — how the pixel moves as the camera-frame point moves.** Differentiate the pinhole formula
`u = fx*Xc/Zc + cx`, `v = fy*Yc/Zc + cy` (`Pcam = (Xc,Yc,Zc)`) directly:

```
J_proj = d(u,v)/d(Pcam) =  [ fx/Zc,   0,      -fx*Xc/Zc^2 ]
                           [ 0,       fy/Zc,  -fy*Yc/Zc^2 ]
```

**Step 3 — chain them.** `J = J_proj * [ d(Pcam)/d(omega) | d(Pcam)/d(v) ] = J_proj * [ -[RP]_x | I_3 ]`,
a 2x6 matrix — exactly what `residual_and_jacobian()` in `src/kernels.cuh` computes (that function's own
comment repeats this derivation beside the code, so the two never drift apart). `main.cu`'s
`jacobian_check` stage verifies this formula numerically at runtime (see "How we verify correctness").

### Levenberg-Marquardt as damped Gauss-Newton

Stacking every correspondence's `r_i`/`J_i`, the Gauss-Newton step solves the LINEARIZED least-squares
problem `min_delta || r + J*delta ||^2`, whose normal equations are `(J^T J) delta = -J^T r`. Gauss-Newton
alone can overshoot badly far from the optimum (it assumes the linearization is globally valid).
**Levenberg-Marquardt** fixes this by damping:

```
(J^T J + lambda * diag(J^T J)) delta = -J^T r
```

This is **Marquardt's** variant of the damping (scaling the DIAGONAL of `H = J^T J`, not adding a flat
`lambda * I`, which was Levenberg's original 1944 proposal). The reason this project uses Marquardt's
variant: the state mixes ROTATION (rad, `omega`) and TRANSLATION (m, `v`) — two physically different
units whose `H` diagonal entries can differ by orders of magnitude in scale. A flat `lambda * I` damps
both equally in absolute terms, over-damping whichever parameter has the smaller natural curvature; the
`diag(H)` scaling damps each parameter proportionally to ITS OWN curvature, which is scale-invariant
(Madsen/Nielsen/Tingleff, cited in README "Prior art", derive and recommend exactly this). `lambda`
adapts every iteration: shrink (`x kLambdaDown = 0.3`) on an accepted (cost-decreasing) step to press
toward pure Gauss-Newton's fast local convergence; grow (`x kLambdaUp = 10.0`) on a rejected step to fall
back toward gradient descent's global robustness. `src/kernels.cuh` documents every constant; every
implementation (GPU assembly-driven, CPU single-trajectory, GPU farm, CPU farm) uses the SAME
single-sourced numbers, so "how damping adapts" cannot silently drift between the four LM loops this
project runs (only "how the loop is WRITTEN" differs — reference_cpu.cpp's header explains why that
distinction matters for the twin gates).

### Observability and the degeneracy lesson

Why does a near-coplanar pose cohort make the solve worse-conditioned? Intuitively: `H = J^T J`'s
CONDITION NUMBER (largest eigenvalue / smallest) measures how "stretched" the cost surface is along
different directions in the 6-D parameter space. A large condition number means SOME combination of
parameters is barely constrained by the data — the loss barely changes as you move along that
direction, so a tiny amount of noise in `g = J^T r` gets AMPLIFIED into a large error in `delta` along
that direction (the Cholesky solve divides by something close to zero).

When every correspondence sits near the SAME depth and the SAME viewing direction (this project's
coplanar cohort), a specific rotation (tilting the target about an axis roughly IN the image plane) and
a specific translation (shifting along the LiDAR's boresight, adjusting apparent scale) produce nearly
identical PIXEL motion for every point — the classical **bas-relief-style ambiguity**: the data cannot
tell "the target rotated slightly" from "the target moved slightly closer/farther and rotated a
compensating amount" when every point shares one depth and one viewing angle. Vary the depth AND the
viewing angle across views (the diverse cohort), and this ambiguity breaks: a rotation now produces a
DIFFERENT pixel pattern than a translation would, at every different depth/angle, and `H` becomes
well-conditioned. This is 01.16's Zhang-calibration finding (view diversity beats view count) restated
for extrinsics instead of intrinsics — and this project's `DEGENERACY` gate turns the qualitative
argument into a measured number (see "How we verify correctness").

## The algorithm

1. **Initialize** `(R, t)` — either a deterministic "rough prior" offset from ground truth (the
   single-trajectory stages) or a randomized perturbation from the identity transform (the multi-start
   farm — see "The GPU mapping").
2. **Assemble** the normal equations: for every correspondence `i`, compute `r_i`, `J_i` (the analytic
   formula above), and accumulate `H = sum_i J_i^T J_i` (packed as 21 upper-triangle scalars),
   `g = sum_i J_i^T r_i` (6 scalars), and `cost = sum_i r_i^T r_i` (1 scalar) — 28 scalars total.
   Serial cost: `O(N)` point evaluations (`N` = correspondence count, 48 here). Parallel cost (the
   correspondence-parallel kernel): `O(N/P + log P)` for `P` threads — the reduction tree's depth.
3. **Solve** the damped `6x6` system via Cholesky decomposition (`O(6^3) = O(216)`, but with the fixed
   constant fully unrolled by the compiler in practice — negligible next to step 2 at any realistic `N`).
4. **Retract** to a candidate `(R', t')` and **re-assemble** (cost only) at the candidate.
5. **Accept** if the candidate's cost is lower (update the estimate, shrink `lambda`, check convergence:
   `||delta||` below a threshold or the cost decrease is negligible); **reject** otherwise (keep the old
   estimate, grow `lambda`, retry next iteration from the SAME point).
6. **Repeat** up to `kMaxLmIters = 20` times.

One full single-trajectory run costs at most `20 * 2 * N` point evaluations (`~1920` for `N=48`, the
"x2" from the current-state assembly plus the candidate-cost check each iteration). The multi-start farm
runs `kMultiStartK = 1024` of these INDEPENDENTLY and in parallel — see "The GPU mapping."

## The GPU mapping

Two genuinely different parallel structures, taught side by side (kernels.cuh's file header names them
"regimes"; here is the argument for when each dominates):

### Regime 1 — correspondence-parallel (`assemble_normal_equations_kernel`)

One thread per correspondence; a block-wide shared-memory TREE REDUCTION folds every thread's 28-scalar
contribution into one row of `block_partials` (`kThreadsPerBlock = 128` threads/block, so `log2(128) = 7`
reduction steps instead of 127 serial adds). Memory: correspondences are read from GLOBAL memory with a
COALESCED access pattern (adjacent threads, adjacent `p_obs`/`uv_obs` records); the reduction lives in
SHARED memory (`128 * 28 * 4` bytes `= 14336` bytes/block, well under the 48 KiB default budget); each
thread's own per-correspondence `local[28]` accumulator lives in REGISTERS. The host finishes the sum
ACROSS blocks in double precision (02.06's ICP convention, cited in `kernels.cuh`) — with `N=48` and
`kThreadsPerBlock=128`, that is exactly ONE block, so the cross-block step is trivial here, but the
kernel is written to be correct for any `N` (README Exercise 3 asks what changes at `N=50,000`, where
the cross-block sum becomes the interesting part).

This regime wins when `N` (the correspondence count) is LARGE: with thousands of correspondences, a
SINGLE optimization already has enough independent work to saturate every SM, and there is only one
estimate to maintain (no reason to run many in parallel).

### Regime 2 — optimization-parallel (`multistart_lm_farm_kernel`)

One thread per INDEPENDENT LM trajectory; each thread draws its own randomized starting guess and runs
its ENTIRE up-to-20-iteration loop serially, re-scanning all `N=48` correspondences itself every
iteration, with ZERO cross-thread communication. Memory: every thread reads the SAME `p_obs`/`uv_obs`
arrays in the SAME order — a BROADCAST read pattern (cheap: the memory system serves one cache line to
many threads at once) rather than the assembly kernel's coalesced-but-distinct-per-thread pattern. Each
thread's state (`H21`/`g6` in double, the current and candidate `Rigid3`) lives in registers, spilling to
thread-local memory once the register file is exhausted at high occupancy — a deliberate
throughput/register-pressure trade this kernel does not tune away (CLAUDE.md §1: "a slower kernel a
learner can follow beats a fast one they cannot").

This regime wins when `N` is SMALL but you want MANY independent trials — exactly this project's shape
(`N=48`, `K=1024` restarts). **Measured**: the farm completes all 1024 trajectories in `~3.0` ms of GPU
time (`~2.97-2.99` ms across repeated runs on this project's reference GPU) — roughly `1024 * 20 * 2 * 48
~= 2.0` million residual/Jacobian evaluations in that window. Running the SAME 1024 restarts through
Regime 1 instead — one host-orchestrated LM trajectory (up to 20 assembly-kernel LAUNCHES) per restart —
would issue on the order of `1024 * 20 * 2 = 40,960` separate kernel launches, each doing only 48 threads
of work (a tiny fraction of even one SM). At a typical CUDA launch overhead of several microseconds per
launch (a well-known order-of-magnitude figure, not something this project measured directly for this
hypothetical), that alone is on the order of **hundreds of milliseconds** — two orders of magnitude
slower than the farm's measured ~3 ms, entirely from launch/PCIe overhead rather than compute. This is
the concrete version of "the natural mapping when N is small and K is large": Regime 2 amortizes the
fixed per-launch cost across a single kernel invocation instead of paying it thousands of times.

## Numerical considerations

- **Rotation-parameterization singularities.** Rodrigues' formula (`so3_exp`) has NO singularity for any
  rotation angle — unlike Euler angles (gimbal lock at 90 deg pitch), axis-angle is well-behaved
  everywhere except the removable `theta -> 0` limit (handled by the small-angle branch above). Because
  every LM iteration uses `so3_exp` as a LOCAL retraction (small `delta`, not a globally-accumulated
  state), the optimizer's own state never approaches any problematic region. The multi-start farm's
  randomized INITIAL perturbation can have a large magnitude (up to `kBasinMaxRotRad = 2.4` rad, `~137`
  deg) — but that is a single, exact `so3_exp` evaluation at construction time, not a repeated local
  increment, so it carries none of the accumulation risk a repeated approximate update would.
- **`J^T J` conditioning — measured.** The degeneracy gate's actual numbers: condition-number proxy
  (largest/smallest eigenvalue of the converged `H`, via `jacobi_eigen_symmetric6`, 01.16's construction
  reimplemented here — cited) `~= 259` on the pose-diverse cohort vs. `~= 28,600` on the coplanar cohort
  — **a ~110x worse condition number** from pose geometry alone, with everything else (ground truth,
  noise level, LM settings) held fixed. Translation-recovery error follows: `~9.1` mm (diverse) vs.
  `~86.9` mm (coplanar), a **~9.5x** accuracy penalty.
- **Float accumulation — the twin discrepancy, explained.** The GPU assembly kernel accumulates each
  block's 28 scalars via a FLOAT shared-memory tree reduction; the CPU oracle casts each per-correspondence
  contribution to DOUBLE before a straight sequential accumulation. A single-shot comparison of the two
  (the `ASSEMBLY_TWIN` gate) measures a worst relative deviation of `1.65e-7` — consistent with FP32
  epsilon (`~1.2e-7`) accumulated over a modest (7-level) reduction tree. Chained across 20 LM
  iterations, where each iteration's tiny discrepancy feeds a NONLINEAR update that slightly changes the
  NEXT iteration's linearization point, this amplifies (the same "chained-RK4 divergence" story 08.01
  tells for its own iterative integration): the `TRAJECTORY_TWIN` and `MULTISTART_TWIN` gates measure a
  final-pose deviation of `~0.040` deg / `~4e-7`-`6e-5` m between the GPU-orchestrated and CPU-only
  trajectories — still tiny in absolute terms, but ~5 orders of magnitude larger than the single-shot
  discrepancy that seeded it, which is exactly why those two gates carry LOOSER, "measured-then-margined"
  tolerances (`0.2` deg) rather than the assembly gate's tight one (`1e-4` relative).
- **The decoupled SO(3) x R^3 retraction vs. the coupled se(3) exponential.** "The math" above named the
  simplification (`t(delta) = t + v` instead of `t(delta) = t + V(omega)*v`). `V(omega)` deviates from
  the identity matrix by `O(theta)` for small `theta` — and LM's `delta` is, BY CONSTRUCTION, small once
  the iteration is near convergence (that is what "the linearization is locally valid" means). Far from
  convergence, where `delta` can be larger, LM's own damping (`lambda` growing on rejected steps) keeps
  actual accepted steps small regardless — so the coupling term's omission changes at most the SPEED of
  convergence, never the converged ANSWER (a fixed point of the iteration is a fixed point regardless of
  which valid local parameterization reached it). The FULLY-coupled version (Barfoot, cited) is the
  fix if a learner wants to verify this claim by comparing iteration counts to convergence.
- **The LiDAR noise model is a documented simplification.** A real time-of-flight LiDAR's noise is
  SPHERICAL/anisotropic in the sensor's own polar frame: tight along the RANGE axis (mm-scale, from
  timing jitter) but growing with range along the TANGENTIAL/bearing axes (a fixed angular resolution
  translates to a growing lateral distance at longer range). This project's `apply_noise()` instead adds
  ISOTROPIC CARTESIAN Gaussian noise (`kNoiseMed.sigma_p_src_m = 1` cm on every axis, uniformly) — most
  accurate near boresight at moderate range, increasingly wrong for off-axis or very-long-range returns.
  The simplification does not change the qualitative DEGENERACY lesson (pose diversity vs. conditioning
  is a geometric fact about the correspondence layout, not the noise model), but a learner extending this
  project to real sensor data should replace it with a proper range/bearing noise model.
- **The `jacobian_check` epsilon/tolerance tradeoff.** Central differencing in FP32 with `eps = 1e-3`
  balances truncation error (`O(eps^2) ~= 1e-6`) against floating-point ROUNDING error in the residual
  itself (`O(ulp/eps) ~= 1e-7/1e-3 = 1e-4`) — the achievable numeric-Jacobian accuracy is therefore
  ROUNDING-dominated, at roughly `1e-4` absolute in quantities of order `1`-`100` (px per unit
  radian/meter). `kJacobianRelTol = 5e-2` leaves generous headroom above that floor; the measured worst
  relative deviation is `1.52e-3` — about 33x inside the tolerance.

## How we verify correctness

This project's twin-independence ruling (the full text lives in `src/reference_cpu.cpp`'s file header —
read it once, it is the canonical statement): the camera model and SO(3) retraction (`so3_exp`,
`pinhole_project`, `residual_and_jacobian`) are SHARED between the GPU and CPU paths (`CALIB_HD` in
`kernels.cuh`) because reimplementing a seven-line closed-form formula twice by hand would be pure
transcription, not independent verification. The LM ITERATION CONTROL FLOW (damping, accept/reject,
convergence) and the per-iteration ACCUMULATION (block-reduction on the GPU vs. a plain loop on the CPU)
are independently written in every case. Because sharing the camera model could in principle hide a bug
from every twin comparison at once, this project carries two gates that do NOT depend on twin agreement
at all: `jacobian_check` (numeric differencing, independent of whether the shared PROJECTION formula
matches reality) and the ZERO-NOISE gates (recovery against ground truth generated by an INDEPENDENT
Python reimplementation of the same camera model — a bug in the shared C++ formula would show up as a
zero-noise recovery FAILURE regardless of GPU/CPU agreement).

Every tolerance below is **measured, then margined**: the project was run once, the worst-case
deviation recorded on an `[info]` line, and the threshold set with documented headroom above it (08.01's
technique, cited throughout). Measured on the reference machine (NVIDIA RTX 2080 SUPER, sm_75,
Release|x64; also verified on Debug|x64, where FP codegen differs and every gate still passed with
comparable margins):

| Gate | What it checks | Measured | Tolerance | Headroom |
|------|-----------------|----------|-----------|----------|
| `JACOBIAN_CHECK` | analytic vs. central-difference numeric Jacobian | `1.52e-3` rel | `5e-2` rel | ~33x |
| `ASSEMBLY_TWIN` | one GPU vs. CPU normal-equation assembly | `1.65e-7` rel | `1e-4` rel | ~600x |
| `TRAJECTORY_TWIN` | full 20-iteration GPU-orchestrated vs. CPU-only trajectory | `3.96e-2` deg / `4.02e-7` m | `0.2` deg / `5e-4` m | ~5x / >1000x |
| `MULTISTART_TWIN` | 64 GPU farm threads reproduced on the CPU | 64/64 converged-classification agreement; worst `3.96e-2` deg / `5.62e-5` m among 52 both-converged | `0.2` deg / `5e-4` m | ~5x / ~9x |
| `BASIN` | fraction of 1024 randomized starts that reach the true optimum | `792/1024 = 77.3%` | `>= 40%` | (a genuine measured basin boundary, not a near-100%/near-0% edge case) |
| `RECOVERY_CAM_LIDAR` | best-of-1024 accuracy vs. ground truth, moderate noise | `0.305` deg / `7.44` mm | `0.6` deg / `15` mm | ~2x / ~2x |
| `RECOVERY_CAM_CAM` | best-of-256 accuracy vs. ground truth, moderate noise | `0.122` deg / `0.87` mm | `0.6` deg / `15` mm | ~5x / ~17x |
| `NOISE_SCALING` | recovery error at 3 noise levels, monotone within slack | low `0.140` deg/`0.99` mm, med `0.256` deg/`29.4` mm, high `1.218` deg/`76.8` mm | monotone +slack | (see below) |
| `DEGENERACY` | coplanar vs. diverse condition number AND translation error | condition `110.3x` worse, translation `9.52x` worse | both `> 3x` | ~37x / ~3x |
| `ZERO_NOISE_*` | noise-free recovery, both scenarios | `0.0` deg exactly / `~5e-4` mm | `0.01` deg / `0.2` mm | >>100x |

`NOISE_SCALING`'s translation error jumps sharply between "low" (`0.10` px / `3` mm sensor noise) and
"med" (`0.50` px / `10` mm) — from under 1 mm to nearly 30 mm — a genuinely nonlinear response, not a
bug: with only 48 correspondences, the LEAST-constrained parameter direction (README/THEORY's
conditioning discussion) amplifies noise more than proportionally once the noise crosses a level where
it starts competing with the correspondences' own geometric signal. This is itself a small, honest
illustration of the same conditioning story the DEGENERACY gate makes explicit with pose geometry instead
of noise level.

## Where this sits in the real world

Production extrinsic calibration goes well beyond this project's teaching core in several directions
(README "Prior art" names the tools; here is what they add):

- **Continuous-time / spline-based backends** (Kalibr) — instead of discrete target poses, a full
  trajectory spline lets a moving rig calibrate against a static target (or vice versa) using EVERY
  frame, not a curated set of 12 views, and jointly estimates time offsets/synchronization between
  sensors — a real problem this project sidesteps entirely by assuming perfectly synchronized, already-
  associated correspondences.
- **Robust loss functions.** Real correspondence sets contain OUTLIERS (a mis-detected corner, a LiDAR
  return off a reflective surface). Production LM implementations (OpenCV's `cv::LevMarq`, Ceres Solver)
  wrap the squared residual in a robust kernel (Huber, Cauchy) that down-weights large residuals instead
  of letting a single bad correspondence dominate `H`/`g` — this project's plain sum-of-squares assumes
  clean data, appropriate for a controlled calibration-rig capture but not a target-less "calibrate from
  driving footage" pipeline.
- **Target-less / online calibration** — an active research area (the catalog's `[R&D]` spirit, though
  this bullet was not itself tagged `[R&D]`): instead of a physical checkerboard, use natural scene
  features (edges, planes, learned keypoints) visible to both sensors and continuously refine the
  extrinsic during normal operation, catching drift (this project's PRACTICE.md §3 "recalibration
  triggers") automatically instead of requiring a scheduled offline procedure. Open problems: robustness
  to scenes with too little cross-modal structure (a featureless highway), and distinguishing "the
  extrinsic drifted" from "the world moved" without a known-static target.
- **Rig-wide joint calibration** — a real AV sensor suite calibrates ALL cameras and ALL LiDARs
  simultaneously in one large joint optimization (hundreds to thousands of parameters), not pairwise —
  README Exercise 5 sketches the smallest step toward this (jointly solving both this project's
  scenarios), which is still far short of a full rig-wide bundle adjustment.
