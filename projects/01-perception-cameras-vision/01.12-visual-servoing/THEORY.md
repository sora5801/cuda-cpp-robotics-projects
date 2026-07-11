# 01.12 — Visual servoing: image-Jacobian control loop entirely on GPU: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### The sensor physics: perspective projection

A pinhole camera maps a 3-D point `P = (X, Y, Z)` expressed in the **camera's own frame** (origin at
the optical center, `Z` along the optical axis, pointing at the scene) to a 2-D point on the image
plane through simple similar triangles: a ray from `P` through the optical center crosses a plane one
unit of focal length away at

```
x = X / Z         y = Y / Z
```

These `(x, y)` — **normalized image coordinates** — are this project's *features*. (A real camera also
applies a focal length and a pixel-coordinate offset, `u = f_x·x + c_x`; we work in the
already-calibrated, unit-focal-length normalized plane throughout, the standard simplification for
teaching the control law — a real system's intrinsic calibration would be applied once, upstream, by
the detector.) The frame convention for `(X,Y,Z)` here is the standard **machine-vision optical
frame** — `x`-right, `y`-down, `z`-forward — a deliberate, documented deviation from the repo's
default body convention (`kernels.cuh`'s file header states this explicitly, per CLAUDE.md §12);
`z`-forward is what makes `Z` "depth," a physically meaningful, always-positive quantity in front of
the lens.

### Why image-space control is a genuinely different problem than Cartesian control

A Cartesian controller (e.g. project 08.01's MPPI, or a joint-space PID) acts on a plant whose
*output* is a position/velocity/pose — a linear or near-linear function of the actuators, for small
motions. Here, the "plant" a visual servo controller drives is **the composition of the robot's rigid
motion with the camera's perspective projection** — and that projection is fundamentally, permanently
nonlinear: `x = X/Z` is a *division*, not a sum. Two consequences that make IBVS its own discipline
rather than "the same PID with a camera bolted on":

1. **The controller never knows Cartesian position at all.** IBVS closes the loop directly on `(x,y)`
   pixels-normalized-to-unit-focal-length — no triangulation, no pose estimate, no map. This is the
   entire appeal (robust to camera/hand-eye calibration error in ways a pose-estimating controller is
   not) and the entire cost (the controller can only ever "see," and therefore only ever correct,
   whatever is *observable* through 8 numbers — see "The math" below for what that limits).
2. **The mapping from a 3-D twist to a 2-D feature velocity depends on depth `Z`, which the controller
   does not directly measure.** A camera moving toward a point and a camera moving parallel to it
   produce *different* feature velocities for the *same* Cartesian speed, scaled by `1/Z`. Every
   practical IBVS system must therefore either measure `Z` (stereo/RGBD — this project's "true-depth"
   variant, an upper bound) or *assume* one (this project's "fixed-depth" and "desired-Jacobian"
   variants — what most real systems actually do, per README's Prior art).

### The retreat pathology, derived geometrically

Classic IBVS has one famous failure mode, reproduced on purpose by this project's RETREAT cohort
(`kernels.cuh` "COHORTS"): start the camera at (nearly) the **goal position** but rotated by **≈180°
about its own optical axis**. A perfectly reasonable-looking control law drives the camera to
**physically retreat** — increase its distance from the target by 3× or more — instead of simply
rotating back. Why?

Rotating 180° about the optical axis maps every feature `s*_i` (the goal position) to **its mirror
image through the image center**, `s_i ≈ -s*_i` (to first order, for a rotation about the principal
point — true here because position error is near zero). The control law `v_c = -λ·L̂⁺·(s - s*)` moves
every feature along the **straight line in image space** from `s_i` to `s*_i` — but the straight line
from `-s*_i` to `+s*_i` passes *through the image center*. So the controller's plan, read purely in
image space, is: "shrink every feature's distance from the center to (near) zero, then swing them back
out to the goal."

Now ask: **what Cartesian motion cheaply shrinks every feature's image radius toward zero,
simultaneously, for all 4 points?** Not a rotation about the optical axis (`ω_z`) — that term's
contribution to `(ẋ,ẏ)` is `(y, -x)·ω_z`, a *pure rotation of the image points around the center*; it
never changes any point's radius `r = √(x²+y²)` at all, so it cannot explain the "shrink" phase by
itself. Radius does shrink, though, under **retreat** (`v_z > 0`, moving away along the optical axis):
with the 3-D point's lateral offset `(X,Y)` roughly fixed and `Z` growing, `x=X/Z` and `y=Y/Z` both
shrink *for every point at once* — a single scalar knob (`v_z`) that uniformly shrinks all 4 points'
radii together is a very cheap way, in a least-squares sense, to explain a "shrink toward center"
image motion shared by every point in the configuration. The damped-least-squares solve (`chol6_solve`
in `kernels.cu`) finds exactly this: a solution dominated by `v_z`, not `ω_z` — literal backward
motion where a human would just spin the camera back around. THEORY's numerics section explains why
this is **not** a conditioning/singularity artifact (measured in this project's `conditioning_honesty`
info line): `L` is perfectly well-conditioned near this configuration — the pathology is a *geometric*
property of the straight-line path the linear control law commits to, re-derived at every instant as
the (still large) rotation error persists.

## The math

**Notation** (SI units, right-handed frames, `T_parent_child` transform convention — all per
`kernels.cuh` and `docs/SYSTEM_DESIGN.md` §3): camera pose `pose = (p, q)`, `p ∈ ℝ³` the camera's
position in the WORLD frame (m), `q = (w,x,y,z)` a unit quaternion, `T_world_cam`'s rotation (repo
scalar-first order). A world point `P_w` is seen in the camera frame as `P_c = R_wcᵀ(P_w - p)`.

### The interaction matrix, derived per point

A point *fixed in the world* has, in the moving camera's own frame, the well-known rigid-body
velocity relation (a standard result: the point's apparent velocity relative to the camera is minus
the camera's own velocity, corrected for rotation)

```
Ṗ = -v_c - ω_c × P          (v_c, ω_c: camera twist, expressed IN the camera's own frame)
```

Componentwise, with `P=(X,Y,Z)`, `v_c=(v_x,v_y,v_z)`, `ω_c=(ω_x,ω_y,ω_z)`:

```
Ẋ = -v_x - ω_y Z + ω_z Y
Ẏ = -v_y - ω_z X + ω_x Z
Ż = -v_z - ω_x Y + ω_y X
```

Differentiate `x = X/Z` via the quotient rule, `ẋ = Ẋ/Z - x·Ż/Z`, substitute, and simplify (every term
regrouped in terms of `x,y,Z` only — no `X,Y` survive, exactly the reduction that makes the control
law implementable from features alone):

```
ẋ = -v_x/Z + 0·v_y + (x/Z)·v_z +  xy·ω_x   - (1+x²)·ω_y +  y·ω_z
ẏ =  0·v_x - v_y/Z + (y/Z)·v_z + (1+y²)·ω_x -   xy·ω_y   -  x·ω_z
```

i.e. `(ẋ, ẏ)ᵀ = L_pt(x,y,Z) · v_c`, the classical **per-point 2×6 interaction-matrix block**:

```
L_pt(x,y,Z) = ⎡ -1/Z    0    x/Z    xy       -(1+x²)   y  ⎤
              ⎣  0    -1/Z   y/Z    1+y²      -xy      -x ⎦
```

(this is exactly `Lrow0`/`Lrow1` in `kernels.cu`/`reference_cpu.cpp` — code and derivation
line-for-line). Stacking all `kNumPoints=4` points' rows gives the full `8×6` interaction matrix `L`,
so that the WHOLE feature vector obeys `ṡ = L(s,Z)·v_c`.

### The control law

Define the feature error `e = s - s*` (goal features `s*` are a compile-time-derivable constant of
this project's fixed target+goal pose — `build_target_and_goal_cpu`). We want `e → 0`; the simplest
law that achieves exponential error decay **when `L` is known exactly and square** is
`v_c = -λ·L⁻¹·e`. Here `L` is `8×6` (over-determined: 8 features, 6 twist DOF — a deliberate choice,
since a real 4-point fiducial marker is exactly this over-determined), so we use the **Moore-Penrose
left pseudoinverse** via the normal equations, DAMPED (Levenberg-Marquardt) for numerical safety:

```
(L̂ᵀL̂ + μI) x = L̂ᵀ e          v_c = -λ·x
```

`μ` (`kDampingMu`) is a small Levenberg damping constant; the `6×6` system is SPD by construction
(`L̂ᵀL̂` is PSD, `+μI` makes it strictly positive definite) and solved by hand-rolled Cholesky
(`chol6_inplace`/`chol6_solve` — the exact small-SPD-solve idiom project 33.01 teaches, applied here
to Gauss-Newton normal equations assembled fresh every control step rather than read from memory).

### Local stability sketch — and its honest limit

With `s*` constant, `ė = ṡ = L·v_c = -λ·L·L̂⁺·e`. If the Jacobian estimate is exact (`L̂=L`, this
project's `true-depth` variant), let `P = L·L⁺ = L(LᵀL)⁻¹Lᵀ` — a symmetric, idempotent **projector**
onto `range(L)` (a rank-≤6 subspace of `ℝ⁸`, `L`'s row/feature space). Then `ė = -λ·P·e`. Decompose
`e = e_∥ + e_⊥` (in and orthogonal to `range(L)`): `ė_∥ = -λ·e_∥` — **exact** exponential decay for
the component of error the twist can actually influence — while `ė_⊥ = 0` **at that instant** (the
component of error outside `range(L)` is, momentarily, un-drivable). Because `L` itself changes as the
pose evolves, `e_⊥` is not a true constant of the whole trajectory — but its existence is why this
project's `exponential_decay` gate is honestly scoped to a **small-error, pure-translation** cohort
(`kernels.cuh` "COHORTS"): for the *minimal* IBVS case (exactly 3 points, 6 features = 6 DOF, `L`
square and generically invertible), `P = I` and the decay is exact globally; with 4 points (8
features, this project's choice, matching a real square fiducial), the decay is only *locally* clean
— which is exactly what the gate measures (a log-linear fit over the first ~0.6 s, not a claim of
lifelong pure-exponential behavior) and what was measured to hold within 1% on this machine (README
"Expected output").

### The depth-estimate question — the three variants

`L_pt` needs `Z`, the true depth — unavailable from a monocular feature detector alone on a real
robot. Three ways to fill it in (`kernels.cuh` "CONTROLLER VARIANTS", all implemented, all run over
the identical batch of starting poses):

- **`kVariantTrueDepth`** — use the exact current `Z` (only possible in simulation/with a depth
  sensor); the reference upper bound.
- **`kVariantFixedDepth`** — use current `(x,y)` but a *constant* `Z = kGoalStandoff` for every point,
  every step — the textbook practical approximation (assume the target stays near its expected
  working distance).
- **`kVariantDesiredJacobian`** — use the *desired* features `s*` and `Z*` for the **entire**
  interaction matrix, literally constant over the whole loop (`L(s*,Z*)`) — the other classical
  textbook choice, cheaper to compute (in principle — see the GPU-mapping note on why this
  implementation recomputes it anyway) and provably locally stable near the goal by the same argument
  above evaluated at `e≈0`.

## The algorithm

```
build_target_and_goal_cpu()              -> 4 world points, s* (once, closed form)
generate_batch_init_poses_cpu()          -> K starting poses across 3 cohorts (once, host)
upload target/goal to __constant__ memory; upload init poses to device
for each of 3 variants:
  launch K threads, one per loop:
    pose = init_poses[k]
    for t in 0..kMaxSteps:
      for each of 4 points:
        project point into camera frame -> (x,y,Z), feature error (ex,ey)
        pick (xj,yj,Zj) per variant -> per-point interaction-matrix rows
        accumulate A += rowsᵀrows, b += rowsᵀ(ex,ey)      # never materialize dense L (GPU mapping ↓)
      A += μI; Cholesky-factor A in place; solve for x; v_c = -λ x
      if ‖e‖ < kConvergeEps: record converged, break      # early exit (GPU mapping ↓)
      integrate pose by v_c over dt (SE(3), see numerics)
    record steps used, final error, worst conditioning proxy, max depth/feature excursion
```

**Complexity, per loop, per step:** O(4 points × 6²) ≈ O(150) FLOPs to assemble `A,b`, plus one `6×6`
Cholesky factor+solve, O(6³/3) ≈ O(72) FLOPs — both trivial next to, say, 08.01's per-step RK4 (which
itself is trivial per rollout; the *batch* is where the work is). **Serial cost** for `K` loops,
`T` steps each: `O(K·T)` of this per-step work — embarrassingly parallel across `K`, exactly like
08.01's rollouts, which is the whole reason this is a GPU project (see next section).

## The GPU mapping

**Thread-to-data mapping:** thread `k = blockIdx.x·blockDim.x + threadIdx.x` owns closed loop `k`,
256-thread blocks, `ceil(K/256)` blocks — the same thread-per-problem geometry as 08.01/33.01/09.01.
The loops never interact: no shared memory, no atomics, no inter-thread communication beyond the tiny
broadcast reads of the target/goal (`__constant__` memory, mirroring 09.01's robot-model broadcast:
every thread reads the SAME 20 floats every step, the textbook constant-memory use case).

**What is new here versus 08.01's rollout farm** (08.01's kernel is a `Read this after` for this
project — its header comment is worth re-reading first):

- **The "rollout" is a whole CLOSED loop, not a scored candidate.** 08.01's threads all run exactly
  `kHorizon` steps and hand a *cost* back to the host for softmin blending; this project's threads run
  their own feedback law to (possibly early) completion and hand back the final verdict directly — no
  second stage.
- **Register pressure is genuinely higher.** 08.01's persistent per-thread state is a 4-float
  cart-pole state plus RK4 scratch (~30 registers). This project's per-thread state is a 7-float pose
  plus, inside `ibvs_compute_step`, a `6×6` normal matrix (36 floats, factored **in place** into its
  own Cholesky factor — see `chol6_inplace`'s header for why that in-place reuse specifically matters
  here) — on the order of 45-60 live floats at the deepest point. Expect nvcc to report more
  registers/thread and correspondingly lower occupancy than 08.01's kernel on the same GPU; check with
  `-Xptxas -v` rather than trusting a number stated here (compiler versions change register
  allocation) — this project states the *shape* of the trade honestly rather than a specific figure
  that would go stale.
- **Never materialize the dense interaction matrix.** The textbook derivation stacks 4 point-Jacobian
  blocks into an explicit `8×6` matrix `L`, then forms `LᵀL` and `Lᵀe`. Since `LᵀL = Σᵢ Lᵢᵀ Lᵢ` is a
  SUM over points, `ibvs_compute_step` accumulates `A += Lᵢᵀ Lᵢ` and `b += Lᵢᵀ eᵢ` **point by point**,
  in registers, and never stores the `8×6` matrix at all — the same "normal equations via rank-`k`
  accumulation" pattern any Gauss-Newton solver (bundle adjustment, ICP) uses at scale, here with `k=2`
  contributions from `4` points instead of thousands. This roughly halves what the dense-`L` approach
  would cost in registers (no separate 48-float `L` on top of the 36-float `A`).
- **Real early-exit divergence, for the first time in this repo's control projects.** 08.01's threads
  all run the SAME `kHorizon` iterations — no divergence from loop length. Here, `break` on
  convergence means DIFFERENT threads finish at DIFFERENT step counts. CUDA's SIMT execution model
  does not retire a warp until **every one of its 32 threads** has finished — a `break`ing thread's
  warp keeps re-issuing instructions (with that thread predicated off, doing no useful work) until the
  warp's SLOWEST thread also finishes. Because this kernel's loop-index-to-cohort mapping is
  CONTIGUOUS (`kernels.cuh` "COHORTS": all NOMINAL indices first, then DECAY, then RETREAT), most
  32-thread warps are drawn from a single cohort — a warp of 32 RETREAT-cohort threads mostly agrees
  "keep going" (little wasted divergence *within* that warp, since the whole warp pays the full
  `kMaxSteps` together), while a warp straddling a cohort BOUNDARY, or containing one slow outlier from
  an otherwise-fast cohort, is bottlenecked by its single slowest member — the general GPU lesson that
  **early exit only saves wall-clock time to the extent whole WARPS agree to stop**, not individual
  threads (README Exercise 2 asks you to measure this directly).

## Numerical considerations

- **Precision:** FP32 throughout (repo default; the CPU oracle uses `float` too, with `std::` math
  function spellings — see `reference_cpu.cpp`).
- **Damping vs. conditioning near-singular `L`.** `A = LᵀL` alone is only positive **semi**-definite,
  and can drop rank below 6 near a true kinematic singularity of the 4-point configuration (e.g. all 4
  points nearly collinear in the image, which cannot happen for this project's fixed square target at
  any pose actually reachable from the documented cohorts, but the code does not special-case that
  away — damping is the general-purpose guard). `μI` (`kDampingMu = 0.05`) makes `A` strictly SPD by
  construction, so `chol6_inplace`'s `sqrt` argument is mathematically guaranteed positive; the
  `fmaxf(sum, 1e-12f)` clamp in the code is a defensive belt-and-suspenders against FP32 rounding, not
  a load-bearing correctness mechanism. Too much damping slows convergence (the effective step shrinks
  toward `0`); too little lets a near-singular step fire an unphysically large twist — `kDampingMu` was
  chosen empirically small relative to `A`'s typical diagonal magnitude (`O(1)`–`O(40)` across the
  configurations this project visits) and is exactly the parameter the `conditioning_honesty` `[info]`
  line probes (measured whole-batch correlation between the worst conditioning proxy encountered and
  eventual non-convergence: `r≈0.14` on this machine — weakly positive as expected, and explicitly
  weak for the RETREAT cohort specifically, `r≈0.00` — because that cohort's failure is the geometric
  pathology derived above, not a numerical-conditioning one; conflating the two would be a real
  scientific error this project takes care not to make).
- **The conditioning proxy is a proxy, honestly.** `cholesky_diag_ratio` returns
  `min(diag(R))/max(diag(R))` of the Cholesky factor — cheap (it reuses work already being done for
  the solve) but NOT the true condition number of `A` (that needs `A`'s actual eigenvalues/singular
  values, expensive at this size on a GPU thread). It correlates with conditioning (a near-singular
  `A` drives one pivot toward the damping floor while others stay large) without being it — every
  place this project reports it says "proxy," never "condition number."
- **Float accumulation over up to 400 steps.** Each step's twist, integration, and re-projection
  introduce ~1 ULP-scale rounding; over 400 compounding steps this is why the single-loop trajectory
  twin (README "Expected output") uses a LOOSE late-step tolerance (5e-3) after a TIGHT early-step one
  (2e-4) — an explicit, measured acknowledgment that GPU-vs-CPU agreement on a chaotic-adjacent
  feedback loop degrades gracefully with trajectory length, not a claim of bit-exact agreement forever.
- **FMA fusion is the dominant source of GPU-vs-CPU disagreement, not a bug.** The
  Jacobian/pseudoinverse twin (a SINGLE step, no compounding) still shows worst deviations up to
  `~7e-3` in `A`'s entries in a **Release** build — traced to nvcc's device code aggressively fusing
  multiply-adds (single rounding) versus cl.exe's host code under `/fp:precise` (potentially separate
  rounding), compounded through the point-accumulation loop for the RETREAT-style near-singular sample
  poses where `A`'s entries themselves reach into the tens. A **Debug** (unoptimized, `-Od`/no FMA
  fusion pressure) build of this exact project shows **EXACT 0.0** agreement on the same comparison —
  direct, measured proof the algorithm is identical on both paths and the Release-build gap is purely
  a compiler-optimization rounding artifact, not an indexing/sign/layout bug (see `main.cu`'s tolerance
  constants for the exact measured numbers this is calibrated against).
- **Quaternion normalization drift.** Every `ibvs_integrate` call renormalizes the updated quaternion
  (`quat_normalize`) rather than trusting the exponential-map update to stay exactly unit-norm across
  hundreds of steps of FP32 accumulation — the standard, cheap (`rsqrtf`) guard against drift.
- **Angle wrapping:** not applicable in the usual sense — this project never stores or compares a bare
  angle; orientation is always a quaternion, and the only "angle" that appears explicitly is the
  per-step rotation-vector magnitude inside `ibvs_integrate`'s exponential map, which is well-defined
  for any real value (no wrap needed).

## How we verify correctness

Per `reference_cpu.cpp`'s twin-vs-shared ruling (quoted there in full): the CONTROL LAW (quaternion
math, the per-point interaction-matrix row, the `6×6` damped solve, the SE(3) integrator) is
implemented **twice, independently** — `kernels.cu`'s `__device__` functions and
`reference_cpu.cpp`'s plain-C++ twins never call each other or share code — while target/goal geometry
and the initial-pose RNG (pure SETUP data, never part of the algorithm under test) are single-sourced.

**Three GPU-vs-CPU twin comparisons**, at three grains (see README "Expected output" for the measured
numbers each run prints):

1. **Single-loop trajectory twin** — one loop's full ≤400-step trajectory, tight early / loose late
   tolerance (numerics above explains why).
2. **Jacobian/pseudoinverse twin** — the intermediate linear algebra (`v`, `A`, `b`, `e`) at 16 sampled
   poses spanning small offsets, large offsets, and near-180° rotations, across all 3 variants — the
   finest-grained check, isolating a single control step from any compounding.
3. **Batch-statistics twin** — a 128-loop subset of the actual K=4096 batch, fully re-simulated on the
   CPU, compared on converged-flag agreement, steps-to-converge, and final error.

**Three INDEPENDENT gates** that route through **neither** twin's shared assumptions — satisfying the
ruling's requirement for at least one verification path a bug common to both twins could not hide
behind:

- **`exponential_decay`** — a closed-form CONTROL-THEORY prediction (`ė_∥ ≈ -λe_∥` from "The math"
  above), fit from the actual trajectory data and compared to the compile-time constant `λ`. A bug that
  broke BOTH the GPU and CPU implementations identically (the exact failure mode the twin-vs-shared
  ruling warns about, and the one that bit flagship 13.03 in this repo's own history) would still be
  caught here, because this gate does not ask "do the two paths agree," it asks "does the measured
  behavior match the independently-derivable physics."
- **`retreat_pathology`** — a known QUALITATIVE result from the visual-servoing literature (Chaumette
  1998 and the tutorials in README's Prior art), not a number derived from either implementation.
- **`convergence_basin`** — while not a closed-form prediction, it is a STRUCTURAL sanity check (a
  correctly implemented controller should converge from modest initial errors) independent of the
  twins' bit-level agreement.

## Where this sits in the real world

- **ViSP** (Inria's Visual Servoing Platform) is the production-grade open-source implementation of
  everything derived above — its `vpFeaturePoint`, `vpServo`, and `vpAdaptiveGain` classes are the
  "grown-up" versions of `ibvs_compute_step`, `kVariantFixedDepth`, and a tunable `λ`, respectively.
  Study it; this project reimplements the CORE loop from first principles rather than wrapping it.
- **PBVS (Position-Based Visual Servoing)** sidesteps the retreat pathology entirely by estimating full
  3-D pose first (via the 4 points + known geometry — a PnP solve) and controlling in Cartesian space;
  the cost is a dependency on an accurate camera model and pose estimate this project's pure
  image-space approach never needs.
- **Hybrid / 2.5-D visual servoing** (Malis, Chaumette & Boudet 1999) is the practical modern default:
  decouple translation (controlled in image space, IBVS-style) from rotation (controlled from a
  partial pose estimate), which provably avoids the retreat pathology's cause (a poorly-conditioned
  coupling between translation and rotation error) without PBVS's full pose-estimation cost.
- **Learned visual servoing** replaces the hand-derived interaction matrix (or the whole control law)
  with a learned mapping from image observations to twist commands — an active research direction this
  project does not implement, named honestly rather than approximated badly.
