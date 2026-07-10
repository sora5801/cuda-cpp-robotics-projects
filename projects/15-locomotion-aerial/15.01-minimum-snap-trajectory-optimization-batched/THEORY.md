# 15.01 — Minimum-snap trajectory optimization batched over waypoint sets: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**A quadrotor is under-actuated in a very specific way.** It has 4 independent rotor speeds and 6
degrees of freedom (position x,y,z and orientation roll/pitch/yaw) — 4 controls for 6 DOF. What
makes it flyable at all is that the 4 rotor speeds map, through the airframe geometry, to exactly 4
useful quantities: one **total thrust** (along the body's own "up" axis) and three **body torques**
(roll, pitch, yaw). Position is controlled *indirectly*: to accelerate horizontally, the vehicle must
first **tilt** so that part of its thrust points sideways, then untilt once the acceleration is no
longer needed. Every horizontal position command is therefore secretly an ATTITUDE command in
disguise.

**Differential flatness makes that disguise explicit and exact (Mellinger & Kumar, 2011).** Newton's
second law for the vehicle's center of mass, in the world frame, is

```
m * x_ddot(t) = R(t) * [0, 0, T(t)]^T  -  m * g * z_world           (x in meters, T in newtons)
```

— the ONLY force other than gravity is the thrust vector, which always points along the body's own
z-axis, scaled by the scalar thrust `T(t)`. Rearranging: the desired **thrust direction** at any
instant is completely determined by the desired acceleration `x_ddot(t)` (it must cancel gravity and
supply the acceleration — a vector you can compute the instant you know `x_ddot`). That is the first
rung of differential flatness: *given a smooth position trajectory, the required body orientation
(up to yaw) falls out of its SECOND derivative, algebraically, no differential equation to integrate.*

Climb one derivative at a time and the same trick keeps working:

- **Acceleration `x_ddot`** ⟶ required thrust DIRECTION (hence required attitude, up to yaw).
- **Jerk `x_dddot`** ⟶ how fast that direction must ROTATE ⟶ required body ANGULAR VELOCITY.
- **Snap `x_ddddot`** ⟶ how fast the angular velocity must change ⟶ required body ANGULAR
  ACCELERATION ⟶ (through the vehicle's moment of inertia) the required TORQUES ⟶ the required
  RATE OF CHANGE of the 4 individual rotor speeds.

**This is the physical reason snap is the objective, not an arbitrary choice of "4th derivative
sounds smooth."** A trajectory with huge snap somewhere demands huge, fast changes in rotor speed at
that instant — exactly the kind of command that saturates motor bandwidth, draws current spikes the
ESCs and battery may not deliver, and excites airframe vibration. **Minimizing the integral of
squared snap along the whole flight is, to first order, minimizing how hard the rotors have to work
to fly it** — a genuine actuator-effort proxy, derived from the airframe's own physics, not a
cosmetic smoothness heuristic. (The chain stops at snap for the widely-used quadrotor model because
position and yaw are the *flat outputs*: the full state and every control input can be written as an
algebraic — not differential — function of these two signals and a *finite* number of their
derivatives, up to the 4th derivative of position and the 2nd of yaw. Yaw is left constant here — a
scoping choice; extending yaw is a natural next project, not attempted here.)

**The engineering frame this project deliberately does NOT model:** real airframes have a finite
thrust-to-weight ratio (bounds max acceleration), finite motor electrical/mechanical time constants
(bound how fast rotor speed can actually change, regardless of what the trajectory *asks* for),
finite battery/ESC current, and propeller/airframe vibration modes that a too-aggressive trajectory
can excite. A minimum-snap trajectory is a **smoothness proxy**, not a constrained optimal-control
solution — it does not know the vehicle's thrust limit and will happily ask for accelerations the
airframe cannot deliver if the waypoints are close together and the segment time is short. Production
systems check feasibility post-hoc (does the required thrust/angular-rate profile stay inside
measured vehicle limits?) or fold time allocation into the optimization so tight turns get more time.
This project fixes segment duration (README Exercise 2) and never checks feasibility against a
vehicle model — stated honestly here and in [Limitations & honesty](README.md#limitations--honesty).

## The math

**Per-axis, per-segment polynomial.** For one axis (x or y) and one segment `s = 0..3`, position is
a degree-7 polynomial in NORMALIZED segment time `tau ∈ [0,1]`:

```
p_s(tau) = c_{s,0} + c_{s,1}*tau + c_{s,2}*tau^2 + ... + c_{s,7}*tau^7        (8 unknowns per segment)
```

4 segments × 8 coefficients = **32 unknowns per axis, per waypoint set** — `kSysN` in
[`src/kernels.cuh`](src/kernels.cuh). Real time is `t = (s + tau) * kSegmentDurationS`; because
EVERY segment shares the same fixed duration `T`, a `tau`-domain derivative and a physical-time
derivative differ only by a constant factor `1/T^d` — a fact used below to justify working entirely
in `tau` (never dividing by `T`) inside the constraint matrix.

**Why degree 7, specifically.** Minimizing `∫(p⁗)² dt` (the 4th-derivative-squared functional,
`k=4`) over a space of polynomials is a classical calculus-of-variations problem whose Euler-Lagrange
solution is a polynomial of degree `2k−1 = 7` on each smooth piece (the same logic that makes natural
cubic splines, `k=2`, degree `3 = 2·2−1`, the minimizers of curvature-squared). Degree 7 is not an
arbitrary "let's have some extra coefficients" choice — it is the smallest polynomial degree for
which minimizing snap is even a well-posed problem with enough freedom to satisfy realistic boundary
and continuity conditions.

**Constraint counting — why THIS project's system is square (32 equations, 32 unknowns), and how
that differs from the textbook minimum-snap QP.** Three kinds of linear constraint apply:

| # | Constraint | Count | Rows in `kernels.cuh` |
|---|------------|-------|------------------------|
| 1 | Position interpolation: `p_s(0)=wp_s`, `p_s(1)=wp_{s+1}`, every segment | 2×4 = **8** | 0..7 |
| 2 | Zero velocity/accel/jerk (`d=1,2,3`) at the flight's two extreme ends | 2×3 = **6** | 8..13 |
| 3 | Interior continuity: derivative `d` of the left segment at `tau=1` equals derivative `d` of the right segment at `tau=0`, for `d = 1..d_max`, at each of the 3 interior waypoints | 3×d_max | 14..(13+3·d_max) |

Rows 1 and 2 total 14, fixed by the physical setup (interpolate every waypoint; start and end at
rest — "rest" meaning zero velocity, acceleration, AND jerk, a "hover-to-hover" flight). For the
system to be square (as many equations as the 32 unknowns), row 3 must contribute exactly `32 − 14 =
18` equations, and since there are 3 interior waypoints, `3 * d_max = 18  ⟹  d_max = 6`. **The
interior continuity is therefore forced to run through the 6th derivative ("pop")** — not a
stylistic choice to go "as smooth as possible," but the unique value of `d_max` that makes 8 + 6 +
3·d_max land on exactly 32.

**This is where this project's construction and the textbook minimum-snap QP (Mellinger & Kumar
2011; Richter, Bry & Roy) genuinely diverge, and it is worth being precise about it.** In the
original formulation, interior continuity is only REQUIRED up through jerk (`d_max = 3`, the minimum
needed for the trajectory to have a well-defined, finite snap everywhere — a discontinuous jerk would
mean an infinite spike in snap at that instant). That gives `8 + 6 + 3·3 = 23` equality constraints
for 32 unknowns — **9 degrees of freedom short of square**. The textbook method resolves those 9 DOF
(the interior waypoints' snap, crackle, and pop values — deliberately left free) by treating the
whole problem as a **quadratic program**: minimize `Σ_segments ∫(p⁗)² dt` (a quadratic form in the
coefficient vector, `c^T Q c`) subject to the 23 linear equality constraints, and solve the resulting
KKT system (or the smaller, equivalent "free-derivative" reduced system many implementations use).
**That KKT solve is qualitatively more work than a square linear solve** — it needs the cost's
Hessian `Q`, not just the constraint rows.

This project instead **pins the remaining 9 degrees of freedom as EQUALITY constraints** — continuity
of snap, crackle, and pop across every interior waypoint, exactly saturating the DOF budget — so the
whole problem collapses to a single square 32×32 linear system, solvable by plain Gaussian
elimination with no cost matrix in sight. The result is a genuinely smooth, `C^6`-continuous
degree-7 spline through the waypoints (visually and qualitatively very close to a true minimum-snap
trajectory — both are "as smooth as this polynomial family can make it") — but it is **not, in
general, the exact minimizer of `∫‖snap‖² dt`** the way the free-derivative QP's solution is; the
extra continuity we impose is a strong SUFFICIENT condition for smoothness, not the variational
NECESSARY condition the true optimum satisfies. [Limitations & honesty](README.md#limitations--honesty)
states this plainly, and README Exercise 5 asks you to build the true QP and measure the gap.
`kernels.cuh`'s header comment is the single source of truth for this row layout; `kernels.cu` and
`reference_cpu.cpp` implement it identically.

**The row-building identity.** Every constraint in this project is evaluated only at the segment
boundaries `tau=0` or `tau=1` — never an interior point — which collapses the general derivative
identity

```
d^d/dtau^d [ tau^j ] = j! / (j-d)! * tau^(j-d)      (j >= d;  else 0)
```

into two simple shapes: at `tau=1`, every coefficient `c_j` with `j >= d` contributes (since
`1^(j-d)=1` for any exponent); at `tau=0`, only `c_d` survives (since `0^(j-d)` is 1 only when
`j=d`, else 0). No `powf()` call is needed anywhere in assembly — every nonzero matrix entry is a
small EXACT integer (`falling_factorial(j,d) <= 7!/1! = 5040`), representable exactly in FP32's
24-bit mantissa.

**The analytic snap-cost integral.** The 4th derivative of a degree-7 polynomial is a CUBIC in
`tau`:

```
p''''(tau) = 24*c_4 + 120*c_5*tau + 360*c_6*tau^2 + 840*c_7*tau^3   =  a_0 + a_1*tau + a_2*tau^2 + a_3*tau^3
```

so `∫_0^1 (p'''')^2 dtau` has a closed form via the moment identity `∫_0^1 tau^(m+n) dtau =
1/(m+n+1)`:

```
∫_0^1 (p'''')^2 dtau  =  Σ_{m=0}^{3} Σ_{n=0}^{3}  a_m * a_n / (m+n+1)
```

— exact, no numerical quadrature, computed by `segment_snap_integral` in `src/main.cu`. Converting to
real time: `dt = T*dtau` and the 4th time-derivative carries a `1/T^4` factor, so the SQUARED
integral picks up `1/T^8` from squaring and one factor of `T` back from the `dt→dtau` substitution —
net **`1/T^7`** (`compute_snap_cost`).

## The algorithm

Per waypoint set, per axis (both run identically on GPU and CPU — the only difference is one thread
vs. a sequential loop):

1. **Assemble** the 32×32 matrix `A` and RHS `b` from the row plan above (`O(N²)` work, dominated by
   zeroing the mostly-sparse matrix).
2. **Forward-eliminate with partial pivoting**: for each of the 32 columns, find the largest-magnitude
   candidate pivot among the remaining rows, swap it into place, then eliminate it from every row
   below (`O(N³/3) ≈ 11,000` multiply-adds).
3. **Back-substitute** (`O(N²) ≈ 1,000` multiply-adds) to recover the 32 coefficients.
4. Repeat steps 1–3 for the other axis.

**Complexity.** Serial cost per waypoint set: `O(2·N³)` (two axes); across a batch of `K` sets,
`O(K·2·N³) ≈ 220` million flops for the default `K=10,000` — measured at ~55–60 ms on one CPU core
(~4 Gflop/s, plausible for scalar FP32 with the memory traffic Gaussian elimination on a 1024-float
local array implies). The GPU performs the SAME total flop count (no algorithmic shortcut — every
thread does its own full elimination) but in parallel across thousands of resident threads; measured
kernel time is ~11–12 ms, a modest (~5×) speed-up compared to, say, 08.01's rollout kernel (tens to
hundreds of ×) — the GPU-mapping section below explains why this kernel's memory-access pattern
caps the win well short of that.

## The GPU mapping

```
one thread = one waypoint set, both axes, solved end to end

registers : wp_x[5], wp_y[5]                (10 floats — the waypoint coordinates)
LOCAL     : A[32*32], b[32], x[32]           (1,024 + 32 + 32 = 1,088 floats ≈ 4.3 KB/thread)
grid      : ceil(K/256) x 256                (repo default; ragged tail guarded)
```

**Why LOCAL memory, not registers (the direct contrast with 33.01).** 33.01's largest matrix (`N=6`,
Cholesky) is 36 floats — after full loop unrolling, the compiler can place every element in its own
register (~48 registers/thread total), the fastest storage the chip has. This project's system is
`32×32 = 1,024` floats for `A` alone — **roughly 4× the entire per-thread register budget** (the
hardware ceiling is 255 registers/thread on Turing/Ampere/Ada). The compiler simply cannot place an
array this large in the register file; it spills to **local memory** — a per-thread PRIVATE region
that physically lives in the same off-chip DRAM as global memory, serviced through the same L1/L2
cache hierarchy, at roughly the same latency class as an *uncached* global load. This is not a
missed optimization or a bug to fix — it is the honest, documented reality of scaling the
thread-per-problem pattern past the point where the "problem" fits in registers, and it is why
`kernels.cu`'s zeroing loop (1,024 stores) is deliberately left as a plain loop rather than force-
unrolled: unrolling only helps when it enables register placement, and here it cannot.

**Why it still wins at batch scale.** A CPU doing this sequentially pays the full `O(K·2·N³)` cost
with only instruction-level parallelism (superscalar execution, a few in-flight instructions) to hide
memory latency. The GPU pays the SAME per-thread cost, but with (at `K=10,000`, 256 threads/block)
~40 resident blocks' worth of INDEPENDENT threads available to the scheduler — while one thread's
local-memory load is in flight, the SM can execute other threads' arithmetic. This is
occupancy-driven LATENCY HIDING, not register-file locality (the mechanism 33.01's kernel relies on)
— a different point on the same "thread-per-problem" design spectrum, and the reason the measured
speed-up here (~5×) is real but far more modest than 33.01's or 08.01's: with ~4.3 KB of local
traffic per thread and 256 threads/block, a block's local-memory footprint (~1.1 MB) vastly exceeds
a Turing SM's L1 capacity (~96–128 KB), so a meaningful fraction of this kernel's traffic misses L1
and lands on L2/DRAM — a bandwidth story 33.01's tiny register-resident matrices never faced.

**A design choice left deliberately unexploited.** Every thread's constraint matrix `A` is
IDENTICAL — the row layout depends only on which derivatives are pinned or continuous, never on the
waypoint VALUES — so in principle `A` could be factored ONCE (host-side, or by a single kernel) and
every thread could do only the cheap `O(N²)` forward/back substitution for its own right-hand side,
turning `O(K·2·N³)` into `O(N³ + K·2·N²)`. This kernel does NOT do that: every thread independently
reassembles and re-eliminates the same matrix, twice. That is the deliberate, maximally-parallel
"thread owns the whole problem" pattern this repository teaches everywhere (08.01, 33.01, 09.01) —
it generalizes trivially to problems where the matrix genuinely DOES vary per-thread (unequal
segment times, missing waypoints, per-set constraint choices), which is the common real case; README
Exercise 4 asks you to build the smarter shared-factorization version and measure what it wins (and
what complexity it costs).

**Contrast with the production tool.** cuSOLVER's batched dense solvers
(`cusolverDnSgetrfBatched`/`getrsBatched`) implement exactly "many independent small linear systems,"
with far more sophisticated internal blocking/memory management than this hand-rolled kernel — this
project is the from-scratch explanation of what such a call does inside (CLAUDE.md §1); README §11
points there for the production path.

## Numerical considerations

- **FP32 throughout, no fast-math.** `--use_fast_math` is deliberately not enabled (matching the
  repo's Release settings) — this kernel's arithmetic is already at the edge of comfortable FP32
  precision (see conditioning, below), and relaxed-precision intrinsics would only make that worse.
- **Conditioning of the high-order polynomial basis — and why NORMALIZED time is what makes this
  problem tractable in FP32 at all.** The nonzero entries of `A` range from 1 (position rows) up to
  `falling_factorial(7,6) = 5040` (pop-continuity rows) — already a ~3.7-decade spread. Now imagine
  the (tempting, and WRONG) alternative of writing the polynomial directly in *real* segment time
  `t ∈ [0,T]` instead of normalized `tau ∈ [0,1]`. With a segment duration of, say, `T=4 s` (a
  realistic slower-flight value — this project's own demo happens to use `T=1 s`, which numerically
  masks the effect, so this example is deliberately a different, illustrative `T`), a term like
  `c_7 * t^7` would need `t^7` evaluated up to `4^7 = 16,384` — combined with the derivative weights
  above, matrix entries would range from 1 to *millions*, seven or more decades. Partial pivoting
  cannot fix a genuinely ill-conditioned matrix; it only protects against a merely *badly ordered*
  well-conditioned one. Working entirely in `tau ∈ [0,1]` (as `kernels.cuh`'s layout mandates and
  `kernels.cu`/`reference_cpu.cpp` implement) keeps every power bounded by 1, so the ONLY spread left
  is the falling-factorial coefficients themselves (~3.7 decades) — well within FP32's comfortable
  range with partial pivoting. This is a structural design decision (never divide by `T` inside the
  matrix), not a runtime check — it is correct for ANY segment duration, not merely the `T=1 s` this
  demo happens to use.
- **Partial pivoting, not Cholesky (contrast with 33.01).** This matrix is NOT symmetric positive
  definite — it deliberately mixes position rows (small integer weights) with high-derivative
  continuity rows (weights into the thousands), so naive "divide by whatever sits on the diagonal"
  elimination risks a near-zero or badly scaled pivot. Partial pivoting (swap in the
  largest-magnitude candidate from the remaining rows of the same column) is the standard, cheap fix
  — `solve_minsnap_system`'s singularity floor (`1e-6`, relative to typical pivot magnitudes) should
  never trigger for this constraint layout: because `A` depends only on the FIXED row structure and
  never on waypoint values, its rank is a structural property, not a per-instance one — it is either
  always full rank or never, and this project's 10,000-random-plus-5-hand-designed-set batch never
  observed a singular hit.
- **FP32 chain depth and the GPU-vs-CPU tolerance.** 32 elimination steps, each doing an `O(N)` row
  update, means each solved coefficient is the result of a long chain of FMA-or-not operations — the
  GPU kernel uses explicit `fmaf` (one rounding step per multiply-add); `reference_cpu.cpp` uses
  plain `*`/`-=` (the host compiler's choice of fusion, which may differ). That is the standard
  reason a GPU-vs-CPU comparison needs a tolerance, not bit-equality (the same story 08.01 and 33.01
  tell) — measured worst-case relative coefficient deviation on this project's batch: **~6.4e-4**,
  against a documented tolerance of `5e-3` (~8× headroom — enough to absorb ordinary
  architecture-to-architecture FMA differences, tight enough that a real indexing/pivoting bug, which
  shifts results at `O(1)` relative scale, trips it immediately).
- **Determinism.** No atomics, no shared memory, no cross-thread interaction of any kind — the batch
  is embarrassingly parallel by construction, so results are bit-reproducible run to run on a given
  machine (the only variation possible is architecture-to-architecture FMA/rounding differences,
  which the tolerances above are sized to absorb).
- **NaN-on-failure policy.** If `solve_minsnap_system` ever DID hit the singularity floor (never
  observed, but defended against per the repo's fail-loud convention — 33.01's precedent), that
  waypoint set's coefficients are filled with NaN rather than a plausible-looking wrong answer;
  `check_batch`'s finiteness check in `main.cu` would catch it immediately.

## How we verify correctness

Two INDEPENDENT checks, because a linear solve can be numerically close to a wrong answer in ways a
single check might miss (and because a bug shared between the assembly code and a naive re-check
using the SAME formula would cancel out and hide):

1. **VERIFY — the §5 GPU-vs-CPU gate.** Every one of the 640,000 solved coefficients (K=10,000 sets
   × 2 axes × 32 coefficients) computed by the kernel and by `reference_cpu.cpp`'s line-by-line twin
   must agree within relative tolerance `5e-3` (floor 1). This catches indexing, layout, or pivoting
   bugs that diverge the two paths — but it CANNOT, by itself, catch a bug present identically in
   BOTH `kernels.cu` and `reference_cpu.cpp` (a wrong row-index formula copy-pasted into both files
   would still "agree" perfectly).
2. **CONSTRAINTS — verification against the mathematical DEFINITION, independent of the CPU oracle.**
   `main.cu`'s `eval_segment_derivs` re-evaluates every solved polynomial (using its OWN,
   separately-written loops — deliberately NOT reusing `assemble_minsnap_system`'s
   `falling_factorial` helper) in DOUBLE precision, and measures three residual categories across
   every one of the 10,000 sets:
   - **Waypoint interpolation error** — `|p_s(0) − wp_s|`, `|p_s(1) − wp_{s+1}|` for every segment;
     measured worst: **~5.1e-5 m** against a `1e-3 m` tolerance (~20× headroom).
   - **Endpoint zero-derivative error** — `|velocity|, |accel|, |jerk|` at the two flight ends;
     measured worst: **~2.3e-4** against `1e-2` (~44× headroom).
   - **Interior continuity jump** — `|left(d) − right(d)|`, relative (floor 1), for `d=1..6` at
     every interior waypoint; measured worst: **~5.0e-4** against `1e-2` (~20× headroom — the
     tightest margin of the three, honestly reflecting that this residual involves the largest,
     least-well-conditioned matrix entries).
   
   Using DOUBLE precision in the check (while the SOLVE stays FP32) isolates the measured residual to
   the solve's own error — the check itself contributes negligibly by comparison. Plus a **cost
   sanity check**: the analytic snap integral (exact, no quadrature) must be finite and non-negative
   for every set — measured range on this project's default batch: `[1.5e3, 1.4e6]`, both bounds
   finite and strictly positive (as expected: even the `straight_line` sample set, whose y-axis
   solves to all-zero coefficients exactly, has a nonzero x-axis snap integral, because the
   rest-to-rest boundary conditions force a nontrivial acceleration profile even along a straight
   line — see `data/README.md`'s per-set notes).

Both checks pass with an order of magnitude or more of headroom above their measured values — enough
that ordinary FMA/rounding differences across GPU architectures cannot flip the verdict, while a
genuine bug (which shifts these numbers at `O(1)`, not at the FP32-rounding scale) trips them
immediately.

## Where this sits in the real world

- **ethz-asl's `mav_trajectory_generation`** implements the TRUE free-derivative minimum-snap QP
  (with time allocation) and has flown on real research quadrotors — the production comparison for
  everything this project simplifies (see [The math](#the-math)'s DOF-counting discussion).
- **PX4 Autopilot** is where a real vehicle's trajectory-following control loop actually runs
  (position/attitude control at the rates `docs/SYSTEM_DESIGN.md` §1.1 documents). In practice, a
  companion computer running something like this project does NOT ship raw polynomial coefficients
  to the flight controller; it evaluates the polynomial at the controller's own rate and streams
  position/velocity/acceleration setpoints (e.g., MAVLink `SET_POSITION_TARGET_LOCAL_NED`) —
  PRACTICE.md §3 discusses this split in more detail.
- **cuSOLVER's batched dense solvers** are the production GPU library for exactly this class of
  problem ("many independent small linear systems") — see [The GPU mapping](#the-gpu-mapping)'s
  contrast above.
- **Time-optimal / minimum-time extensions** (jointly optimizing segment durations, not just fixing
  them — README Exercise 2) and **feasibility-checking** (verifying the resulting thrust/angular-rate
  profile stays inside a real vehicle's measured limits) are standard production additions this
  project does not attempt — see [Limitations & honesty](README.md#limitations--honesty).
- **What the full version adds** beyond this teaching core: the free-derivative QP (README Exercise
  5), time allocation (Exercise 2), 3-D + yaw (Exercise 3), corridor/obstacle constraints (inequality
  constraints — a genuinely harder QP), and closed-loop replanning against a moving state estimate
  rather than a single static batch.
