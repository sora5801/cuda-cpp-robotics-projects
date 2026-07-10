# 31.01 — Hamilton-Jacobi reachability: level-set grid solvers (stencil ops — GPU-perfect): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why reachability *is* the safety question.** Every mobile robot obeys Newton: it has momentum,
and momentum takes time and distance to shed. A car doing 15 m/s cannot stop in zero meters no
matter how hard the brakes clamp; a quadrotor descending fast cannot arrest its fall instantly no
matter how much thrust the rotors can spin up. This gives rise to the concept of an
**inevitable collision state (ICS)**: a state from which *no* sequence of legal controls can avoid
some bad outcome within the time available. A perception-and-planning stack that only checks "is
there an obstacle in my current path?" can walk a robot into an ICS well before the obstacle is
close, simply by ignoring how much stopping distance the robot's own dynamics demand. Reachability
answers the *complementary* and more useful question directly: **from which states can the robot
still (with the right control) reach some goal/safe region within a given time budget?** Compute
that set once, offline, and a runtime monitor need only ask "is my current state still inside it?"
— a single array lookup, cheap enough to run every control tick, and *exhaustive* over every
possible control sequence rather than the finite handful a sampled test could ever try.

**The double integrator as every 1-DoF axis's model.** This project's plant is deliberately the
simplest one that still has real momentum: position `x` [m], velocity `v` [m/s], acceleration
command `u` [m/s^2] bounded `|u| <= umax`, with dynamics

```
xdot = v            (position changes at the current velocity)
vdot = u            (velocity changes at the commanded acceleration, up to the actuator's limit)
```

This is not a toy in the dismissive sense — it is the **local model for one axis of almost every
real actuator**: a car's longitudinal speed vs. throttle/brake authority, a drone's altitude vs.
thrust margin, a robot arm joint's angular velocity vs. torque limit (near a fixed configuration,
where the effective inertia is roughly constant). Multi-axis, multi-body robots decompose
(imperfectly, but usefully) into a handful of such 1-DoF reachability problems, which is exactly
why this is the catalog's entry point into all of reachability (★ beginner) rather than a
simplified toy that teaches nothing transferable.

**Engineering constraints a real safety monitor imposes.** A monitor built on this idea must (a)
finish its offline solve well within whatever planning cadence updates the target/horizon (seconds,
not hours — this is why the numerical method must be GPU-fast, not just correct), (b) answer its
online query (state inside the set or not?) within the control loop's hard deadline (a single array
lookup easily clears even a 1 kHz budget — SYSTEM_DESIGN §1.1), and (c) be conservative in the
right direction: a monitor that is *too permissive* (calls an unsafe state safe) is worse than
useless, while one that is *too conservative* (calls a safe state unsafe) merely costs performance.
The numerical dissipation this project measures and documents (§Numerical considerations) pushes
the *numeric* boundary slightly inward of the true one at this grid resolution — the safer
direction, and worth knowing explicitly rather than discovering by accident.

## The math

**State and target.** State `(x, v) in R^2`. A **target set** `T = {(x,v) : l(x,v) <= 0}` for some
scalar **level function** `l`. This project's target is a sublevel set of the closed-form
minimum-time-to-origin function (derived below): `l(x,v) = T*(x,v) - t0`, so
`T = {T*(x,v) <= t0}` — every state that can reach the *exact origin* within `t0` seconds. `t0`
(the scenario's `TTARGET`) is a target-set sizing knob, not part of the dynamics.

**The Hamiltonian.** For dynamics `xdot = f(x,v,u)` and a candidate value gradient
`p = (px, pv) = grad(V)`, define

```
H(x, p) = min over |u| <= umax of  p . f(x, v, u)
        = min over |u| <= umax of  (px*v + pv*u)
        = px*v - umax*|pv|                                  (the inner min picks u = -umax*sign(pv))
```

`H` is the **min**, not the max, because this is an *existential* reachable set: "does *some*
control make progress toward the target?" (A *robust/avoid* reachability problem — "for *all*
disturbances" — would use `max` instead; that is a different, harder catalog entry.)

**The backward reachable tube (BRT) PDE — dynamic-programming sketch.** Define
`V(x,v,tau) = min over control policies, min over s in [0,tau], of l(xi(s))`, where `xi` evolves
FORWARD under the dynamics for `s` seconds starting at `(x,v)`, and `tau` is elapsed **backward**
time (`tau = 0` is "now," `tau = T` is "T seconds before now" — the horizon we solve to). In words:
*V is the best (lowest) value of `l` the robot can achieve along its own future, minimized over
every control it could apply, and over every instant up to `tau` seconds out.* The one-step Bellman
recursion `V(x,tau) = min(l(x), min_u V(x + f(x,u)*dtau, tau - dtau))` becomes, in the limit
`dtau -> 0` (the standard dynamic-programming-to-PDE limit, via Taylor expansion of the second
term and `min` distributing over the two branches):

```
dV/dtau = min( 0,  H(x, grad V) )                         V(x, 0) = l(x)
```

This is the **tube** equation (as opposed to a plain `dV/dtau = H` transport equation): the `min`
with 0 is the "freeze" that turns "value at exactly tau" into "best value seen anywhere in
`[0,tau]`" — the value may only ever *decrease*, matching the intuition that a state which was ever
inside the target set counts as reachable, permanently, no matter what happens after. The
**backward reachable tube at horizon T** is `{(x,v) : V(x,v,T) <= 0}` — every state from which the
target is reachable at *some* point within `T` seconds, which is what a robot's safety monitor
actually wants (as opposed to "reachable at *exactly* `T`," which a plain transport equation would
give and which is a much less useful set for safety purposes).

**Where the exact solution formula comes from.** For *this specific* target (a min-time sublevel
set of the *same* dynamics the PDE governs), the value function has a closed form. Along the
control policy that is time-optimal to the origin, `T*` decreases at the fastest possible rate
along the *whole trajectory to the origin* (a standard sub-arc-optimality property of time-optimal
control: truncating a time-optimal trajectory yields a time-optimal trajectory to the new
endpoint), so `l(xi(s)) = T*(x,v) - s - t0` for `s <= T*(x,v)`, and `l(xi(s)) = -t0` (its floor,
achieved by coasting at rest at the origin) for `s >= T*(x,v)`. Minimizing over `s in [0, tau]`
gives:

```
V_exact(x, v, tau) = max( T*(x,v) - (t0 + tau), -t0 )
```

At the demo's horizon `tau = T` (the scenario's `HORIZON`), the boundary of the BRT is exactly
`T*(x,v) = t0 + T` — this is the number the analytic verification stage checks the numeric field
against, and this whole derivation is independently re-verified in the investigation that shaped
this project's parameters (a brute-force forward simulation of the bang-bang policy, tracking
`min_s l(xi(s))` directly, agrees with this formula to simulation-integrator precision).

**The closed-form minimum-time-to-origin solution (the analytic oracle).** By Pontryagin's
maximum principle, the time-optimal control for the double integrator is **bang-bang with at most
one switch**: full acceleration one way, then full acceleration the other, arriving at the origin
exactly as velocity reaches zero. The switch happens on the **switching curve**
`x = -v*|v| / (2*umax)` (both branches of the parabola in one formula — verify: for `v <= 0` this
is `x = v^2/(2*umax)`, the arc reached by coasting in under `u = +umax`; for `v >= 0` it is
`x = -v^2/(2*umax)`, the mirror arc under `u = -umax`). Off the curve:

```
right of the curve (x + v|v|/(2 umax) > 0): thrust u = -umax first, then u = +umax to arrive
    T*(x,v) = ( v + 2*sqrt( v^2/2 + umax*x ) ) / umax

left of the curve:  thrust u = +umax first, then u = -umax to arrive (the mirror image)
    T*(x,v) = (-v + 2*sqrt( v^2/2 - umax*x ) ) / umax

on the curve: single arrival phase, no switch
    T*(x,v) = |v| / umax
```

implemented, in double precision, in `min_time_to_origin()` (`src/reference_cpu.cpp`) — the
independent oracle every numeric answer in this project is checked against.

## The algorithm

1. **Load the scenario** (grid size, domain, `umax`, `t0`, horizon `T`) — six numbers, the whole
   problem definition (`data/sample/double_integrator_scenario.csv`).
2. **Build the initial level function** `l0(x,v) = T*(x,v) - t0` on the host, in double precision
   (the oracle's own precision), narrowed to FP32 once and fed identically to both solver paths.
3. **March `n_sweeps` explicit backward-time sweeps.** Each sweep: every cell reads its own value
   and its four face neighbors from a snapshot buffer, computes the Lax-Friedrichs numerical
   Hamiltonian plus dissipation, freezes with `min(0, ...)`, and writes into the other (ping-pong)
   buffer. `n_sweeps = ceil(T / dt_max)`, with `dt = T / n_sweeps` landing the last sweep exactly on
   the horizon (`main.cu`'s derived-parameters block).
4. **VERIFY**: compare the GPU field against the CPU twin's field, cell by cell (max-norm).
5. **ANALYTIC**: compare the GPU field's `V <= 0` classification against `T*(x,v) <= t0+T`, cell by
   cell, excusing a documented band around the true boundary.
6. **Write artifacts**: the value field as a PGM image, and the numeric front (boundary cells) as a
   CSV with the analytic `T*` alongside each point.

**Complexity.** Serial (CPU): `O(nx * nv * n_sweeps)` — for this scenario, `256*256*109 ≈ 7.1M`
cell-updates, each O(1) work (measured: ~50–70 ms on one core). Parallel (GPU): the same total work,
but `n_sweeps` sequential **launches** of `nx*nv` fully independent per-cell updates each — the
per-sweep work is embarrassingly parallel (measured: ~1–3 ms total, i.e., each ~65,536-cell sweep
finishes in a few microseconds of kernel time on a mid-range GPU). The *sweeps themselves* are
inherently sequential (each depends on the previous one's output) — this is a `parallel-map` inside
a `serial-loop`, not a fully parallel algorithm; that sequential loop is exactly what a much larger,
research-grade 3D+ reachability solve would want to shrink via better numerics (§Where this sits in
the real world), not something this project's GPU mapping can remove.

## The GPU mapping

```
one thread = one grid cell (i, j):  i = position index (fast axis), j = velocity index (slow axis)
grid: ceil(nx/16) x ceil(nv/16) blocks of 16x16 threads (07.09's square-tile default, since the
      stencil reaches along BOTH axes symmetrically)

per thread, per sweep:
  global reads : center + 4 face neighbors from `in` (5 floats; the +/-1 x-neighbors are
                 CONSECUTIVE addresses within a warp's row -> coalesced; the +/-nx v-neighbors
                 are whole-row strides the L2 serves across the block's rows of threads)
  registers    : the 5 stencil values + 4 one-sided differences (~20 registers; small enough that
                 occupancy is not the bottleneck at this problem size)
  global write : 1 coalesced write to `out`
  shared mem   : NONE — each interior value is re-read by at most 4 neighbor threads, and at this
                 grid size the L2 cache already serves that reuse; a shared-memory tile (README
                 Exercise 4) is the natural next step to *measure*, not assume, the honest 07.09
                 position on this exact tradeoff.
  atomics      : NONE, divergence: only the tail-guard (`i >= nx || j >= nv`) and the branchless
                 ghost-cell selects at the domain border.
```

This is precisely 07.09's stencil pattern (ping-pong buffers, thread-per-cell, boundary ghost
handling, ragged-tile guards) with the stencil *body* replaced: instead of a distance-field
minimum, each cell evaluates a small piece of numerical PDE (§The math). The **library-call
budget is zero** — no cuBLAS/cuFFT/Thrust; every arithmetic step in the kernel is hand-rolled and
commented in place (CLAUDE.md §1's "no black boxes," trivially satisfied because there is no
library to hide behind).

**Why `ax = |v_j|` needs no artificial dissipation at all (and is not a missed optimization).**
`dH/dpx = v` exactly (linear term), and `v_j` is a **grid coordinate**, not part of the unknown
solution — so the "local Lax-Friedrichs" formula for the `px*v` term, worked out algebraically,
collapses to *exact upwinding*: `vj*pxp` when `vj >= 0`, `vj*pxm` when `vj < 0` (verify by
substituting `vj = a`, expanding `0.5*a*(pxm+pxp) + 0.5*|a|*(pxp-pxm)`, and simplifying the two
cases separately). Zero excess numerical smearing in that axis, "for free," because `v` being a
known grid coordinate rather than an unknown field value removes any ambiguity about the transport
direction.

## Numerical considerations

- **Precision:** FP32 throughout the PDE solve (both GPU and CPU paths), FP64 for the analytic
  oracle and the initial-condition setup (built once on the host, so its extra precision costs
  nothing at runtime and keeps the oracle "beyond suspicion" — `kernels.cuh`'s own framing).
- **CFL stability, and why it is a LAW, not a knob.** Explicit sweeps are stable only while
  information cannot cross more than one grid cell per sweep:
  `dt * (max|v|/dx + umax/dv) <= kCfl` (`kCfl = 0.5`, half the theoretical limit, for margin).
  `main.cu` computes `n_sweeps = ceil(T * rate / kCfl)` and `dt = T / n_sweeps` from this bound —
  `dt` is *derived*, never chosen freely.
- **Long-time-integration dissipation — measured, not assumed.** First-order Lax-Friedrichs
  dissipation, in the semi-discrete (`dt -> 0`) limit, does **not vanish with the timestep** — it
  is a *spatial* (grid-resolution) effect that **compounds with every sweep**. Measured directly
  (a standalone float64 reimplementation of the exact per-cell update, on this scenario's grid):
  shrinking the CFL number 25x (408 -> 10,200 sweeps over the same 1.5 s horizon) changed the
  boundary-band requirement by under 1%, ruling out a temporal-accuracy explanation. Sweeping the
  **horizon** at fixed grid, however, shows the required excused band growing with sweep count:

  | Horizon T (s) | sweeps | required boundary band (cells, this grid) |
  |---|---|---|
  | 0.3 | 82  | 2  |
  | 0.4 | 109 | **2** (committed scenario; `kBandCells = 3` for margin) |
  | 0.6 | 164 | 4  |
  | 1.0 | 272 | 8  |
  | 1.5 | 408 | 13 |

  This is why the committed scenario's horizon is 0.4 s (`t0 + T = 1.0 s` total elapsed-time
  budget) rather than a longer number: a longer horizon is not wrong, it simply has to *pay* for a
  proportionally wider excused band, and a wide band would undercut the entire point of "verify
  against pure mathematics" (README Exercise 2 lets you reproduce this table on your own machine).
  Refining the **grid** at fixed horizon does not rescue this either — worst-case boundary error in
  *grid cells* actually **grows** with resolution (128 -> 512 grid: 7 -> 22 cells), because the
  *physical* smearing width shrinks only very slowly with `dx` (an exponent around 0.18 was
  measured, far below the naive O(dx) first-order rate) near the solution's genuine non-smooth
  features: the origin's cone singularity in `T*`, the switching-curve kink, and — worst of all —
  the reachable tube's own extremal corner (where the boundary curve's tangent is vertical, a true
  cusp of the continuous solution that *no* grid-based first-order scheme resolves sharply). None
  of this is a defect unique to this implementation: it is the standard, textbook behavior of
  monotone first-order schemes for Hamilton-Jacobi equations near kinks (Crandall-Lions/Barles-
  Souganidis convergence theory guarantees convergence to the viscosity solution as `dx -> 0`, but
  says nothing about the *rate*, which degrades exactly at such non-smooth features).
- **Monotonicity, verified by hand.** The full per-cell update (LxF Hamiltonian + dissipation +
  freeze) was checked to have non-negative partial derivatives with respect to every one of its 5
  stencil inputs — a hand Jacobian computation confirms this (worked out during this project's
  numerics investigation) — the structural property (Crandall-Lions monotonicity) that a
  discretization needs before convergence theory says anything at all.
- **Boundary policy:** linear-extrapolation ghost cells at the domain edges (`2*center -
  opposite_neighbor`), which makes the one-sided slopes agree at the border and adds no dissipation
  of its own there. Verified harmless for this scenario: the true reachable tube's extent
  (`|x| <= umax*(t0+T)^2/2 ≈ 0.4 m`, `|v| <= umax*(t0+T) ≈ 0.8 m/s`) stays well over 100 grid cells
  from every domain edge on the committed `[-3,3] x [-2,2]` domain — confirmed empirically too: a
  3x-larger domain at the same resolution changes the measured boundary-band requirement by under
  0.5%, ruling out edge-artifact contamination as an explanation for the dissipation above.
- **Determinism:** no RNG anywhere in this project; FP32 results are bit-identical run-to-run on
  one machine. Across platforms, compiler FMA-contraction choices can differ in last-ulp terms —
  which is why no stable output line carries a floating-point field value, only PASS/FAIL against
  wide-margin tolerances.
- **Angle wrapping / quaternion drift:** not applicable — this project's state space
  (position, velocity) has no periodic or unit-norm components.

## How we verify correctness

Two independent checks, because a PDE solver can be *numerically self-consistent and still wrong*:

1. **VERIFY — the §5 GPU-vs-CPU twin gate.** `hj_sweep_kernel` (GPU) and `hj_sweep_cell` (CPU) are
   a deliberate line-by-line twin — same FP32 arithmetic, same ping-pong discipline, same
   expressions (`kernels.cu` and `reference_cpu.cpp` diff almost line for line). Any indexing,
   layout, sign, or upwinding-direction bug shifts the two paths' results at order 1, not at FP32's
   ~1e-7 rounding floor; the measured worst disagreement, `~1.7e-5`, sits three orders of magnitude
   inside the `1e-3` tolerance, i.e., comfortably in "this is just rounding" territory.
2. **ANALYTIC — the check this project exists to feature.** Every cell's `V <= 0` classification is
   compared against the **closed-form** bang-bang solution (§The math), not against another run of
   the same code. Because the discretization's error is concentrated in a band around the moving
   front (not scattered randomly across the grid — verified: with the committed scenario, *every*
   one of the 230 disagreeing cells sits inside the documented 3-cell band; the count outside the
   band is exactly 0), the excused-band design isolates "the scheme is not infinitely sharp" from
   "the scheme is wrong," and the numbers above back that claim rather than asserting it.

Both checks run from the *same* initial condition and the *same* solved field — VERIFY establishes
that the GPU is trustworthy relative to a from-scratch CPU implementation, and ANALYTIC establishes
that *both* of them are trustworthy relative to mathematics itself. A bug that happened to affect
GPU and CPU identically (a shared logic error in the update expression, say) would sail through
VERIFY and get caught by ANALYTIC instead — which is exactly what happened during this project's
own development: a first implementation attempt at 1.5 s horizon passed VERIFY perfectly while
ANALYTIC correctly flagged (at the *original* `kBandCells=2`) a genuine measurement gap between the
intended scope and what the numerics could honestly deliver, which is what led to the horizon
being re-measured down to 0.4 s rather than the band being silently widened without justification.

## Where this sits in the real world

- **`hj_reachability` (JAX) and `OptimizedDP`** solve the same PDE family at higher dimension
  (typically up to 5–6 states) using higher-order (WENO) spatial schemes, adaptive/narrow-band
  grids, and GPU vectorization via JAX/XLA rather than a hand-written CUDA kernel — the production
  descendants of exactly the numerics taught here, minus this project's first-order simplicity.
- **Mitchell's `helperOC`/`ToolboxLS`** is the original MATLAB reference implementation this
  project's Lax-Friedrichs scheme is drawn from; still the pedagogical gold standard for the field.
- **Control barrier functions (CBFs, catalog 31.04)** are the production-favored *online* cousin:
  instead of a precomputed grid, a CBF bounds the safe set's boundary with an analytic function and
  filters commands in real time at kHz, trading this project's exhaustive-but-offline guarantee for
  a cheap, differentiable, always-on one. Many real safety architectures use both: an offline
  reachability analysis to *design* a conservative CBF, then the CBF at runtime.
- **The curse of dimensionality is real and is the field's central open problem** (catalog 34.06,
  "High-dimensional Hamilton-Jacobi via tensor decompositions [R&D]"). A dense grid's cost is
  exponential in state dimension; this project's 2-D, 65,536-cell grid is *the* regime where dense
  grids remain the right tool. Production systems facing 6+ states (a full quadrotor, a
  manipulator) turn to: decomposition into lower-dimensional subsystems (as this project's "every
  1-DoF axis" framing suggests), neural/deep reachability (learned value functions, no grid at
  all), or the tensor-decomposition and sums-of-squares/SDP relaxations that make up much of
  catalog domain 34's research frontier. This project teaches the foundational 2-D core those
  methods all still build on — the same PDE, the same verification philosophy, a different way of
  representing the value function.
