# 16.01 — Thruster allocation for overactuated ROVs (batched QP): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why an ROV carries more thrusters than it has degrees of freedom.** A rigid body moving in 3D has
6 degrees of freedom (DOF): three translations (surge `Fx`, sway `Fy`, heave `Fz`) and three rotations
(roll `Mx`, pitch `My`, yaw `Mz`). The *minimum* number of independent actuators that can command all
6 DOF is 6 — but almost no ROV ships with exactly 6 thrusters. This project's vehicle carries **8**, for
two physical reasons that have nothing to do with the math being harder:

1. **Fault tolerance.** A subsea vehicle cannot pull over. A tether-managed ROV working under a ship, or
   an AUV on a multi-hour survey, loses a thruster to a fouled prop, a flooded housing, or a burnt motor
   winding often enough that "the vehicle keeps working with one thruster dead" is a real requirement,
   not a nicety. With exactly 6 independent thrusters, losing *any one* leaves the vehicle unable to
   command at least one DOF at all; with 8 arranged so that any two groups of 4 already span 3 DOF each
   (this project's geometry, below), losing one thruster degrades authority in that group but never
   removes a DOF outright — this project's **failure-analysis demo stage measures exactly that
   degradation**, not just asserts it exists.
2. **Full authority for dynamic positioning (DP) and station-keeping.** Real ROVs spend much of their
   working life not moving at all in the net sense — holding position and heading against current while
   a manipulator arm does work, or hovering for a camera survey. That requires the vehicle to be able to
   push in *any* direction at *any* moment without needing to reorient first (unlike a car, which can
   only accelerate forward). A minimal 6-thruster layout can do this in principle, but only by running
   every thruster near its rated limit for common commands; redundant thrusters spread the load and keep
   more headroom in reserve for the current's next gust.

**Thruster physics — what "thruster force" is standing in for.** A marine thruster is a ducted propeller
spun by an electric (or hydraulic) motor. Its steady-state thrust is set by the propeller law
`T = k * n * |n|` (`n` = propeller speed, signed; `k` a thruster-specific constant folding in propeller
pitch/diameter and fluid density) — thrust grows with the *square* of speed, hence `n*|n|` rather than
`n^2`, to keep the sign (reverse thrust needs negative `n`). This project's QP allocates **force `u_i`
directly**, deliberately stopping one layer above the real control input (commanded RPM): the
`n = sign(u)*sqrt(|u|/k)` inversion is a simple, well-documented next step (README "Limitations"), and
skipping it here keeps the QP itself — the actual subject of this project — linear and clean. Two more
honest simplifications, named so the gap to a real vehicle is never invisible:

- **Bollard thrust and asymmetry.** A thruster's *bollard thrust* (thrust at zero vehicle speed, the
  relevant number for station-keeping) is not the same forward and reverse — ducted thrusters are
  shaped to accelerate flow one way, so reverse thrust is typically `~70-80%` of forward. This project
  uses a **symmetric** `+-u_max` box (README "Limitations"; Exercise 4 asks you to fix it).
- **Thruster-thruster interaction.** Two thrusters mounted close together (this project's four vertical
  thrusters sit at the vehicle's corners, `~0.3-0.4 m` apart) can partially ingest each other's wake,
  reducing combined thrust below the sum of their individual bollard ratings. Production allocators
  either mount thrusters far enough apart to make this negligible or measure and correct for it; this
  project's allocation matrix `B` assumes **no interaction** (each thruster's contribution is exactly
  linear and independent) — the standard first-order assumption, and the one every allocation-matrix
  formulation in the literature (README "Prior art") starts from too.

**The engineering frame this project's QP must respect.** Allocation is not a one-shot calculation: it
runs inside a real-time control loop (System context, README), so it inherits that loop's tick budget —
whatever the QP does must finish well inside the tick, for *every* commanded wrench the controller might
ever issue, not just easy ones. And its output drives current through motor windings: a solution that
ignores `u_max` is not a rounding error, it is a command the hardware cannot execute, silently distorting
the achieved wrench in a way the upstream controller never asked for (the motivating example in
"How we verify correctness" makes this concrete with real numbers).

## The math

**Body frame.** This project uses `x`-forward, `y`-starboard (right), `z`-down — the Fossen/SNAME marine
convention (Fossen 2011), a deliberate, documented deviation from the repo's default `x`-forward/`y`-left/
`z`-up (CLAUDE.md §12 explicitly permits this "unless a domain standard says otherwise (state it)" — see
[`src/kernels.cuh`](src/kernels.cuh) for the full reasoning). Units SI throughout: meters, Newtons,
Newton-meters.

**The allocation matrix.** Thruster `i` sits at body-frame position `r_i` (m) and pushes along a fixed
unit direction `d_i`; a signed force `u_i` (N) along that direction produces a body wrench

```
force  contribution:  d_i * u_i                      (N)
moment contribution:  (r_i x d_i) * u_i               (N*m)   — the textbook "torque = r x F"
```

Stacking all 8 thrusters' contributions linearly gives the wrench from any force vector `u in R^8` as
`tau = B*u`, where `B` is the **6x8 allocation matrix**, column `i` equal to `[d_i; r_i x d_i]`. `B` is
built once, in code, from the geometry table in `kernels.cuh` — see `build_allocation_matrix` in
[`src/reference_cpu.cpp`](src/reference_cpu.cpp).

**This project's `B`, printed out** (rows `Fx,Fy,Fz,Mx,My,Mz`; columns `H1..H4,V1..V4`; `c45 = sqrt(2)/2`):

```
Fx  [  c45   c45   c45   c45    0     0     0     0  ]
Fy  [ -c45   c45   c45  -c45    0     0     0     0  ]
Fz  [   0     0     0     0    -1    -1    -1    -1  ]
Mx  [   0     0     0     0  -0.20  0.20  -0.20  0.20 ]
My  [   0     0     0     0   0.15  0.15  -0.15 -0.15 ]
Mz  [-0.2475 0.2475 -0.2475 0.2475   0     0     0     0  ]
```

Notice the **block structure**: the 4 horizontal thrusters (columns 1-4) touch only `Fx,Fy,Mz`; the 4
vertical thrusters (columns 5-8) touch only `Fz,Mx,My`. This is by design (both groups are mounted
co-planar with the vehicle's own axes — the header comment in `kernels.cuh` derives it geometrically)
and it is a real design pattern, not a simplification for this project: separating "the group that
handles horizontal motion" from "the group that handles vertical/rotational trim" is exactly how
production ROV thruster layouts are drawn (README "Prior art", ArduSub's real motor-mixing matrices show
the same block pattern). `rank(B) = 6` — full row rank, so *every* wrench is reachable in the
unconstrained (`eps=0`, no saturation) sense; the interesting physics starts once `+-u_max` bites.

**The QP.** Given a commanded wrench `tau_cmd in R^6`, allocation solves

```
minimize_u   J(u) = || W (B*u - tau_cmd) ||^2  +  eps * ||u||^2
subject to   -u_max_i  <=  u_i  <=  u_max_i        for i = 0..7
```

`W` is a diagonal weight on which wrench components matter most (this project uses `W = I`, README
"Limitations" discusses retuning it). `eps > 0` is a small **Tikhonov regularization** term with two
jobs: (a) it makes the *unconstrained* problem well-posed even though `B` has an 8-column, 6-row shape
(2-dimensional null space — "internal squeeze" force combinations that produce zero net wrench, see
"Numerical considerations"); (b) among near-tied solutions it prefers **less total thrust effort**, a
physically sensible tie-break (less current draw, less heat). Expanding the objective:

```
J(u) = u^T (B^T W^2 B + eps I) u  -  2 (B^T W^2 tau_cmd)^T u  +  const
     = 0.5 u^T H u  -  g^T u  +  const,       H = 2(B^T W^2 B + eps I),   g = 2 B^T W^2 tau_cmd
```

`H` (the QP's Hessian) and `g` (its linear term, a function of `tau_cmd`) are exactly what
`build_qp_matrices` and the kernel compute — see [`src/kernels.cuh`](src/kernels.cuh) for the precise
layout.

**KKT conditions for a box-constrained QP.** For a convex QP with only box constraints, the optimality
(KKT) condition at a feasible point `u*` has an unusually clean form: `u*` must be a **fixed point of
its own projected-gradient step**, for *any* step size `t > 0`:

```
u*  =  Proj_box( u* - t * grad J(u*) ),     Proj_box(v)_i = clip(v_i, -u_max_i, u_max_i)
```

Intuitively: at the optimum, either the gradient in coordinate `i` is exactly zero (an *interior*
optimum in that coordinate — the classic unconstrained `grad J = 0`), or coordinate `i` sits exactly on
a box face and the gradient pushes *further into* that face (so clamping cancels the would-be move).
This single fixed-point condition is what "**GATE-KKT**" tests numerically (below) — no Lagrange
multipliers need to be extracted or interpreted, because the projection already encodes them.

**The step size — why `1/L` guarantees progress.** `H` is symmetric positive definite (`eps > 0` makes
it strictly so), so `J` is convex and `L`-smooth with `L = lambda_max(H)`, the Hessian's largest
eigenvalue — the **Lipschitz constant of `grad J`**. The classical result for projected/proximal
gradient descent on an `L`-smooth convex function with a convex constraint set (Beck & Teboulle 2009,
the ISTA/FISTA paper's Proposition 3.1) is: taking step size `step <= 1/L` guarantees the objective is
**non-increasing at every iteration** — `J(u_{k+1}) <= J(u_k)`, always, no line search needed. This
project estimates `L` **once, on the host, at startup**, by the classical **power iteration**: starting
from a fixed vector, repeatedly apply `v <- H*v`, renormalize, and the Rayleigh quotient `v^T H v`
converges to `L` (`power_iteration_lambda_max` in `reference_cpu.cpp`, `kPowerIters = 100` steps — the
same "compute the hard number once on the host, cheaply and transparently" move 09.01 makes for its
Jacobians). Measured for this project's geometry (`eps = 0.1`): `L = 8.2000`, `step = 1/L = 0.121951`.

**The closed-form ground truth: the damped weighted pseudoinverse.** If no box constraint is ever
active, the QP reduces to the *unconstrained* minimizer of the same quadratic — set `grad J(u) = 0`:

```
H u* = g   =>   u* = (B^T W^2 B + eps I)^-1 (B^T W^2 tau_cmd)
```

This is exactly the classical **Tikhonov-damped, weighted pseudoinverse allocation** every marine-control
textbook derives (Fossen 2011; README "Prior art"). It is *not* an approximation of the QP's answer for
unsaturated wrenches — it **is** the QP's answer, by construction, whenever the unconstrained optimum
happens to already satisfy the box. That equivalence is what "**GATE-PSEUDOINV**" tests (below): it is a
statement about this exact problem's structure, not a heuristic sanity check.

## The algorithm

**Projected gradient descent (PGD), per problem:**

```
u <- 0                                    # cold start (README "Limitations")
g <- 2 * BtW2 * tau_cmd                   # linear term, computed once per problem
repeat kPgdIters (= 500) times:
    grad <- H*u - g                       # 8x8 matvec
    u    <- clip(u - step*grad, -u_max, u_max)   # gradient step + box projection
return u
```

**Why projected gradient descent, and not something fancier?** Three honest reasons, in order of
importance for a *teaching* repository (CLAUDE.md §1):

1. **The projection IS the lesson.** A box is the one constraint set whose Euclidean projection is
   *trivial* — an independent per-component clip, no linear system to solve. That triviality is what
   makes "solve a constrained QP" collapse into "add one clamp to gradient descent," which is exactly
   the concept this project exists to teach. A general-polytope constraint (this project does not have
   one) would need the projection *itself* to be another optimization problem — see "Where this sits in
   the real world" for what production solvers do instead.
2. **It is provably correct, and the proof is checkable by a learner.** The descent lemma above is a few
   lines of algebra a second-year optimization course covers; there is no black-box convergence
   guarantee to take on faith (contrast with, say, trusting an interior-point solver's internal barrier
   schedule).
3. **It batches embarrassingly.** Every problem's loop is identical in shape and iteration count — the
   ideal case for one-thread-per-problem GPU execution (next section).

**Complexity.** Per problem: `g` costs `8*6 = 48` multiply-adds (once); each of `kPgdIters` iterations
costs an `8x8` matvec (`64` multiply-adds) plus 8 clamps — `~500 * 64 = 32,000` multiply-adds total,
`O(kPgdIters * kNThr^2)`. Serial cost for a batch of `K` problems: `O(K * kPgdIters * kNThr^2)`. Parallel
cost (one thread per problem, enough threads to fill the GPU): `O(kPgdIters * kNThr^2)` — independent of
`K`, exactly the payoff a batched-solve pattern is supposed to deliver (33.01 makes the identical
argument for its batched Cholesky).

**Fixed iterations, no early exit — a deliberate choice.** A convergence check (`stop when
||grad_proj|| < tol`) would let *easy* problems finish sooner — but different threads in the same warp
would then take different numbers of loop iterations, and a warp only retires when its *slowest* lane
finishes (THEORY "The GPU mapping" below quantifies the cost). This project keeps every thread's
iteration count identical and verifies convergence **after the fact** instead (the KKT/pseudoinverse
gates) — README Exercise 3 asks you to build the early-exit version and measure the trade yourself.

## The GPU mapping

```
one thread  =  one whole QP (all 8 unknowns), solved entirely in registers
grid: ceil(K / 256) blocks x 256 threads          (repo default geometry)

per thread, every iteration:
  c_H[i*8+j]     __constant__ read, UNIFORM address across the whole warp
                 (every thread wants H[i][j] — the SAME H — every iteration)
  c_BtW2[i*6+j]  __constant__ read, UNIFORM address (used once, before the loop)
  u[8], grad[8]  REGISTERS, private per thread, updated kPgdIters times
tau[k], umax[k]  global reads, ONCE, at the top (coalesced across k)
u_out[k]         global write, ONCE, at the bottom (coalesced across k)
```

**Why `__constant__` memory, specifically (not just "global, read-only").** `H` and `BtW2` do not merely
happen to be read-only — they are *identical for every thread in the batch*, because they depend only on
the vehicle's fixed geometry, never on which wrench a given thread is solving. That is the textbook use
case for CUDA's constant memory: a small (here, `112` floats `= 448` bytes, far under the 64 KiB budget)
region backed by a dedicated on-chip cache optimized for exactly this access pattern — every thread in a
warp requesting the *same* address in the *same* cycle, served as one broadcast read instead of 32
separate ones. This sits at the "always-uniform" end of the same read-pattern spectrum 08.01 walks for
its `u_nom` (uniform, served by the general L2/read-only path) and 09.01 for its rotation table
(`__constant__`, this project's exact pattern) — worth comparing side by side.

**Register budget.** Per thread: `u[8]`, `grad[8]`, `g[8]`, `t[6]`, `um[8]` — `~38` live floats at peak,
comfortably inside the `255`-register-per-thread hardware ceiling, and small enough that `256` threads
per block do not starve the SM's register file the way 33.01's `N=6` Cholesky kernel (`~48` registers
just for one matrix) can. Because every loop bound (`kNThr`, `kNDof`, `kPgdIters`) is a compile-time
constant, `#pragma unroll` turns both matvecs into straight-line FMA code with no loop-branch overhead —
the same unrolling argument 33.01 makes for its fixed-`N` kernels.

**Occupancy character.** No shared memory (nothing is reused *between* threads — every problem is fully
independent, so shared memory would only cost bookkeeping for no benefit), no atomics, and the only
divergence is the ordinary ragged-tail guard (`k >= count`). Because `kPgdIters` is fixed and identical
for every thread (the "no early exit" decision above), a launched warp's 32 lanes execute the *exact
same instruction stream* for the *exact same number of iterations* — the healthiest possible case for
SIMT hardware, and the reason this project could add an early-exit convergence check (Exercise 3) only
at a measurable divergence cost, not for free.

**Why not a batched cuSOLVER/cuBLAS call?** There is no off-the-shelf CUDA library primitive for a
*constrained* batched QP (cuSOLVER's batched routines solve unconstrained linear systems and
eigenproblems — exactly the primitive this project's own `cholesky_solve_spd` reference-oracle helper
uses, at `N=8`, once, not batched). Production GPU-accelerated constrained optimization (README "Where
this sits in the real world") is a much newer and less standardized area than dense batched linear
algebra; hand-rolling here is not a stopgap for a missing library call, it is close to the actual
research/production frontier.

## Numerical considerations

**Precision.** FP32 throughout, matching every other project in this repo — the achieved forces are
`O(1-40) N`, well within FP32's clean dynamic range, and 500 sequential FP32 FMA-chain iterations
accumulate rounding error slowly (measured below).

**Conditioning — the real story, not just a number.** `H`'s eigenvalues for this project's geometry
(`eps=0.1`) range from `lambda_min = eps = 0.1` to `lambda_max = 4.1`, so `L = 8.2`, `step = 1/L =
0.121951`, and the condition number `kappa = lambda_max/lambda_min = 41`. That number alone says "PGD
converges in a few hundred iterations" (true — measured, this project's `kPgdIters = 500` carries
comfortable headroom) but hides *where* the ill-conditioning comes from and what it costs physically.
Decomposing `B^T B` (`W=I` here) by eigenvalue reveals the vehicle's **per-DOF authority spectrum**:

| Eigenvalue of `B^T B` | Physical direction (wrench space) | Gain `= lambda/(lambda+eps)` at `eps=0.1` |
|---|---|---|
| `0` (x2) | the 2 "internal squeeze" null directions — zero net wrench | n/a (unobservable, by design) |
| `0.09` | pure pitch, `My` | `47%` |
| `0.16` | pure roll, `Mx` | `62%` |
| `0.245` | pure yaw, `Mz` | `71%` |
| `2.0` (x2, degenerate) | surge/sway, `Fx`/`Fy` | `95%` |
| `4.0` | heave, `Fz` | `98%` |

The "gain" column is the fraction of a *commanded* wrench component the **unconstrained** damped
pseudoinverse actually *achieves* along that pure direction — a direct consequence of the closed-form
solution derived above (`H`'s eigenvalue `lambda` in a direction becomes a scalar gain `lambda/(lambda+
eps)` in that same direction, the standard Tikhonov-damping result). **Pitch is this vehicle's weakest
authority direction** because its moment arm (`kVx = 0.15 m`, the vertical thrusters' fore/aft offset)
is smaller than roll's (`kVy = 0.20 m`) — a *geometry* fact, not a solver artifact, and it is exactly why
this project's failure-analysis stage (README, `main.cu`) measures wrench-tracking **degradation
relative to the nominal baseline** rather than against an absolute threshold: even a perfectly healthy
vehicle "under-tracks" pitch and roll commands by design, once `eps` damping is in the picture. Choosing
a smaller `eps` recovers more authority at the cost of a worse-conditioned (slower-converging) QP — the
`eps`-sweep table in this project's design notes (reproduced in README "Limitations") makes the
trade-off's shape explicit: `eps=0.01` gives `kappa~401` (needs thousands of PGD iterations for the same
accuracy), `eps=1.0` gives `kappa=5` (converges in a few dozen iterations, but the weakest directions
drop to `~9%` gain) — there is no "correct" `eps`, only a documented choice for this project's iteration
budget.

**FP32 accuracy, measured (not assumed).** On the reference machine (RTX 2080 SUPER, `kPgdIters=500`):

- GPU-vs-CPU (identical FP32 algorithm, different compilers: nvcc vs. cl.exe) worst per-thruster-force
  disagreement over the whole 500-wrench batch: `2.7e-05 N`.
- QP vs. closed-form pseudoinverse (unsaturated rows, `472/500`): worst deviation `6.1e-05 N`.
- KKT projected-gradient residual (saturated rows, `28/500`): worst `5.7e-06 N`.
- Objective monotonicity: once `J(u_k)` converges (this project's motivating example converges to
  `J~1319.31` by iteration `~35` of `500`), *every later iteration recomputes the same 8x8 matvec from a
  numerically-settled `u`* — pure FP32 rounding noise at that magnitude is `~J * 2^-23 ~ 1.6e-4`; the
  measured worst observed uptick was `3.7e-4` (a handful of such ticks, all after convergence, never
  during the real descent). `main.cu`'s monotonicity gate carries a documented `1e-3` slack, `~3x`
  headroom over that measurement — never large enough to mask a genuine ascent early in the run, where
  `J` is still dropping by whole integers per step.

**No angle wrapping, no quaternions, no stiff ODE here.** Unlike a controller that integrates
orientation over time (08.01's cart-pole, 09.01's kinematics), this project is a **static per-tick map**
— it has no state carried between calls (the cold-start-at-zero decision, above) and touches no angles
directly (moments are linear quantities, not orientations). The robotics-specific numerical hazards
CLAUDE.md §4.2 asks every THEORY.md to consider (angle wrapping, quaternion drift, stiff integration) are
therefore **N/A here by construction** — worth stating explicitly rather than silently omitting.

## How we verify correctness

Four independent checks, because a batched optimizer can be *numerically close to the CPU oracle and
still not actually solving the right problem* (a bug in the QP formulation itself would reproduce
identically on both paths):

**1) The §5 GPU-vs-CPU gate.** The kernel and `reference_cpu.cpp`'s `thruster_allocate_cpu` run the
*exact same* fixed-iteration PGD algorithm against the *exact same* `H`/`BtW2`/`step` (computed once,
shared by both paths — never recomputed per-path, so there is only one place the QP's setup math could
disagree with itself). Catches indexing bugs, layout mismatches, and launch-configuration errors — any
such bug shifts results at order `1`, not order `1e-5`.

**2) `GATE-PSEUDOINV` — closed-form ground truth for the easy case.** As derived in "The math," any
*unsaturated* QP solution must equal `Q^-1(B^T W^2 tau)`, computed independently via an 8x8 Cholesky
factorization (`cholesky_solve_spd`, the `N=8` case of 33.01's algorithm, run once — not batched — as
the oracle). This is not "does the optimizer look reasonable" — it is "does this exact optimizer solve
this exact problem," checked against closed-form linear algebra a reader can verify by hand.

**3) `GATE-KKT` — the fixed-point optimality condition for the hard case.** For every *saturated*
solution, the projected-gradient residual `||u - Proj_box(u - step*grad J(u))||` must be near zero (the
KKT condition derived in "The math"). Unlike the pseudoinverse gate, this has no independent
closed-form target — it is a self-consistency check on the returned `u` itself, which is exactly what
makes it meaningful for the box-active case where no simple closed form exists.

**4) `GATE-MONOTONE` — the motivating worked example, re-used as a runtime check.** The wrench
`tau_cmd = (-18.33, 1.91, 0, 0, 0, -62.99)` N/N/N/Nm/Nm/Nm (`kMotivatingWrench`, `kernels.cuh`) is a
demanding combined surge-correction + yaw-correction command. Working through it end to end:

```
unconstrained damped pseudoinverse:  u* = ( 38.37, -50.72,  39.66, -52.00,  0,0,0,0 )  N   <- H2,H4 want
                                                                                            far beyond +-40N
naive approach ("solve unconstrained, then clip"):
   u_clip = ( 38.37, -40.00,  39.66, -40.00,  0,0,0,0 )     achieved = (-1.39,  0.91, 0,0,0, -39.11)
                                                             |achieved Fx| / |commanded Fx| =  7.6%
                                                             angle(achieved, commanded)     = 14.2 deg

this project's QP (properly re-optimizes under the SAME box):
   u_qp   = ( 29.63, -40.00,  32.09, -40.00,  0,0,0,0 )     achieved = (-12.92, 1.74, 0,0,0, -35.07)
                                                             |achieved Fx| / |commanded Fx| = 70.5%
                                                             angle(achieved, commanded)     =  4.1 deg
```

**This is the whole reason naive pseudoinverse-then-clip is dangerous, made concrete.** Both approaches
saturate the *same* two thrusters (H2, H4), and both "achieve less than commanded" — that part is
unavoidable once the box actually binds. But naive clipping does not just under-deliver *magnitude*, it
distorts *direction*: because H2 and H4 also carry most of the "surge" information the pseudoinverse
packed into them (they were asked to go far past their limit specifically to also help with `Fx`), simply
truncating them at `+-40N` throws away almost all of the surge command while barely touching the yaw
command — the achieved wrench points `14.2 deg` away from what was asked for, and recovers only `7.6%`
of the commanded forward force. The QP, faced with the *identical* box constraint, does not just clip —
it **redistributes**: it pulls H1 and H3 back from their pseudoinverse values (`38.37 -> 29.63`,
`39.66 -> 32.09`) to partially compensate for what H2/H4 can no longer deliver, landing at a wrench only
`4.1 deg` off-axis and recovering `70.5%` of the commanded surge — because the QP's objective, unlike
"clip and hope," is *literally defined* as "get as close to the commanded wrench as this box allows"
(confirmed by the objective values themselves: `J(u_clip) = 1482.6 > J(u_qp) = 1319.3` — the QP provably
found the better feasible point, not just a different one). `main.cu`'s monotonicity gate re-runs this
exact example with per-iteration objective tracing (`thruster_allocate_trace_cpu`) and asserts `J(u_k)`
never increases across all 500 steps — turning the worked example above from a one-time illustration
into a standing regression test.

The wrench batch itself (`data/sample/wrench_batch.csv`) is committed so the whole check runs offline,
and the `allocation.csv` artifact (README) makes every row's achieved-vs-commanded comparison inspectable
directly, not just pass/fail.

## Where this sits in the real world

- **Fossen's textbook treatment (Fossen 2011)** derives the same damped-weighted-pseudoinverse allocation
  this project's ground-truth gate checks against, and extends it with *thrust-direction* constraints
  (some real thrusters — azimuthing/vectored ones — can also rotate, adding an angle to allocate, not
  just a magnitude) that this project's fixed-direction thrusters do not need.
- **Production DP (dynamic positioning) allocators** (README "Prior art": Sørensen 2011, Johansen &
  Fossen 2013) generalize this project's QP in several directions this teaching version deliberately
  skips: **rate limits** (a thruster cannot jump from `-40N` to `+40N` instantly — real allocators
  penalize `du/dt`, turning the problem into a QP over a short horizon, not a single instant),
  **quadratic thrust-vs-power costs** (since real power draw grows faster than linearly with thrust,
  production cost functions are not always the simple quadratic this project uses), and **general
  linear/polytope constraints** beyond a box (e.g., "these two thrusters together cannot exceed the
  bus's current limit") — exactly the case where the trivial box-projection this project relies on
  stops being trivial, and production solvers reach for **active-set** (qpOASES) or **ADMM** (OSQP)
  methods instead of plain projected gradient descent.
- **ArduSub's real motor-mixing matrices** are this project's `B` matrix at production scope: hand-tuned
  (or CAD-derived) per-vehicle, with saturation handled by a simpler prioritized-scaling scheme (scale
  *all* thruster commands down together until none exceeds its limit) rather than a full QP re-solve —
  a real, shipped simplification that trades this project's directional accuracy (the motivating example
  above) for lower computational cost on a small flight-controller MCU, not a GPU. This project's GPU
  angle is specifically about the case ArduSub's design does not target: solving **thousands** of
  allocation problems at once (a planned trajectory, or a fault-tolerance sweep), where the extra
  optimality accuracy is affordable because the hardware is available.
- **What a full research-grade version would add beyond this teaching core:** azimuthing-thruster angle
  allocation (mixed continuous/discrete optimization), the RPM-command layer and its own actuator
  dynamics (spin-up lag, deadband), measured thruster-interaction correction terms, and — the actual GPU
  research frontier here — running the *whole* allocation QP inside a kHz-class control loop with CUDA
  Graphs (32.02's territory) rather than this project's one-shot-per-tick launch.
