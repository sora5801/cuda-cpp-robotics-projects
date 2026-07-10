# 17.01 — Batched Lambert solvers + porkchop plot generation: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**The two-body problem.** Ignore every gravitational influence except the Sun (an excellent
approximation for a spacecraft far from any planet), and Newton's law of gravitation plus his second
law give one body's motion around a much heavier one:

```
r_ddot = -mu * r_hat / r^2          (vector form: r_ddot = -mu * r / |r|^3)
```

where `r` is the position vector from the Sun, `r = |r|`, and `mu = G*M_sun` is the Sun's
**standard gravitational parameter** (SI value `1.32712440018e20 m^3/s^2` — note `G` and `M_sun`
never appear separately in orbital mechanics; only their product does, and it is known far more
precisely than either factor alone, because it is measured directly from spacecraft tracking). This
one differential equation, integrated, produces every closed conic orbit — circle, ellipse, parabola,
hyperbola — depending on the body's energy. **Kepler's three laws** are exactly the qualitative
content of this equation's solutions: (1) orbits are conics with the Sun at a focus, (2) equal areas
in equal times (angular momentum conservation), (3) `T^2 proportional to a^3` (worked out below).

**The engineering frame.** A real mission does not get to choose when the destination planet is
"there" — Earth and Mars are both independently orbiting the Sun, so a transfer trajectory must be
timed to meet a moving target with a moving departure point. That timing problem, done right, is a
mission's **launch window**: a range of departure dates (and, for each, a range of viable arrival
dates) where a physically realizable, propellant-affordable trajectory exists. Missing a launch
window by even days can mean waiting a full **synodic period** (worked out below) — for Earth-Mars,
about 26 months — for the next opportunity. The porkchop plot is the mission designer's tool for
seeing the whole window at once instead of solving one departure/arrival pair at a time; this project
GPU-accelerates exactly that "solve it 262,144 times" step. There is no sensor noise, no real-time
deadline, and no actuator here (this is offline mission design, README §System-context) — but there
is real engineering stake: propellant is the single most expensive, most mass-constrained resource on
an interplanetary spacecraft, and the difference between a good and a mediocre launch-window choice is
routinely hundreds to thousands of m/s of delta-v, which is the difference between a mission that is
possible and one that is not.

## The math

### Canonical units

Every quantity below is expressed in **canonical units**: a length unit `LU = 1 AU`, and the Sun's
`mu` fixed to exactly `1` by *choosing* the time unit `TU` that makes it so:

```
TU = sqrt(LU^3 / GM_sun_SI) = sqrt((1.495978707e11 m)^3 / 1.32712440018e20 m^3/s^2)
   = 5,022,642.89 s = 58.132441 mean solar days = 0.159158 Julian years
```

(`kernels.cuh`'s file header carries the full derivation and the resulting SI conversion table.)
This is standard practice in orbital mechanics: it keeps every number in this project's FP32
arithmetic near order 1 instead of order `1e11` or `1e20`, and it makes the physics equations below
read exactly as their textbook form with `mu = 1` dropped out entirely.

### Kepler's third law, and a circular orbit's closed form

For a circular orbit of radius `r`, the gravitational force supplies exactly the centripetal
acceleration: `mu / r^2 = omega^2 * r`, so the angular rate is

```
n = omega = sqrt(mu / r^3)              (this project's orbital_rate(r), mu = 1: n = r^-1.5)
```

and the position/velocity at canonical time `t` (phase angle `theta = n*t`, both bodies starting at
`theta = 0` — the convention `kernels.cuh` documents and this section works out the consequence of)
are the closed form `body_state()` implements directly — no integration needed, because a circular
orbit is *exactly* solvable:

```
r(t) = r * (cos(n t), sin(n t))          v(t) = r*n * (-sin(n t), cos(n t))
```

Kepler's third law falls out immediately: the period is `T = 2*pi/n = 2*pi*sqrt(r^3/mu)`, i.e.
`T^2 proportional to r^3` (here `a = r` since the orbit is circular).

### The synodic period, and why the porkchop plot has its shape

Earth-like body 1 (`r1 = 1 LU`, `n1 = 1 rad/TU`) and Mars-like body 2 (`r2 = 1.524 LU`,
`n2 = r2^-1.5 = 0.531524 rad/TU`) orbit at *different* angular rates, so their relative geometry
(the angle between them, as seen from the Sun) repeats with period

```
T_syn = 2*pi / |n1 - n2| = 13.412 TU   (~2.14 Earth years — the real Earth-Mars synodic period)
```

**This is why a porkchop plot looks like a plot at all, rather than a single number.** A launch
window is not "a" date — good departure/arrival pairs recur once per synodic period, and *within*
one synodic period the relative geometry sweeps through every possible phase angle, which is exactly
why this project's `WINDOW_TU = 28.0` (~2.09 synodic periods — "pick clean round numbers," README's
ratified scope) guarantees the grid contains at least one full occurrence of every transfer geometry,
including the Hohmann-optimal one (next subsection). Real porkchop plots for real missions look
"loopy" (nested contour bowties) for exactly this reason: the plot is a 2D slice through a
periodically-repeating 1D relative-geometry story.

### The Lambert problem — statement

Given two position vectors `r1`, `r2` (here: each body's position at its own epoch, `body_state()`
above) and a time-of-flight `TOF = t2 - t1`, **Lambert's problem** asks: find the two-body orbit that
passes through both points, taking exactly `TOF` to go from the first to the second. The answer is a
pair of velocity vectors `v1_transfer`, `v2_transfer` — the velocity a spacecraft must have at
departure and will have at arrival to fly that connecting orbit. This project's delta-v is then simply
the two "patch" burns:

```
delta_v = |v1_transfer - v1_body(t1)| + |v2_body(t2) - v2_transfer|
```

(leave the departure body's own orbital velocity onto the transfer orbit; leave the transfer orbit
onto the arrival body's own orbital velocity — two impulsive burns, the textbook idealization real
propulsion systems approximate with a short powered arc).

### The universal-variable formulation

The classic difficulty: Lambert's problem has *different* closed forms for elliptical, parabolic, and
hyperbolic transfer orbits. The **universal variable** `z` and the **Stumpff functions** `C(z)`,
`S(z)` unify all three into one set of equations (this is the textbook derivation — Curtis §5.3,
Bate–Mueller–White §5.3 — condensed here to what this project implements):

```
C(z) = (1 - cos(sqrt(z))) / z          z > 0 (elliptical)
S(z) = (sqrt(z) - sin(sqrt(z))) / z^1.5

C(z) = (cosh(sqrt(-z)) - 1) / (-z)     z < 0 (hyperbolic)
S(z) = (sinh(sqrt(-z)) - sqrt(-z)) / (-z)^1.5

C(0) = 1/2, S(0) = 1/6                  z = 0 (parabolic — Taylor series limits)
```

Given the transfer angle `dtheta` (next subsection) and the constant

```
A = sin(dtheta) * sqrt(r1n*r2n / (1 - cos(dtheta)))     [the textbook form — see the numerics
                                                          section below for why this project uses
                                                          the algebraically equivalent, numerically
                                                          nicer form A = sqrt(2*r1n*r2n)*cos(dtheta/2)]
```

the auxiliary quantity `y(z)`, the "universal anomaly" `chi(z)`, and the time-of-flight as a function
of `z` are:

```
y(z)   = r1n + r2n + A*(z*S(z) - 1) / sqrt(C(z))
chi(z) = sqrt(y(z) / C(z))
TOF(z) = chi(z)^3 * S(z) + A*sqrt(y(z))          (mu = 1: no sqrt(mu) divisor)
```

`TOF(z)` is monotonically increasing in `z` over the branch this project searches (hyperbolic,
`z -> -inf`, gives `TOF -> 0`; the elliptical single-revolution boundary, `z -> 4*pi^2`, gives
`TOF -> infinity`) — that monotonicity is *why* a bracketed root-find on `z` is valid at all: solving
Lambert's problem is finding the `z` with `TOF(z) = TOF_target`. Once `z` is found, the **Lagrange
coefficients** recover the transfer velocities directly:

```
f    = 1 - y/r1n            g = A*sqrt(y)            gdot = 1 - y/r2n
v1_transfer = (r2 - f*r1) / g          v2_transfer = (gdot*r2 - r1) / g
```

### The transfer angle, and the "short way"

The transfer angle `dtheta` — how far around the Sun the trajectory sweeps — is computed from the two
position vectors via

```
dot     = r1 . r2
cross_z = r1.x*r2.y - r1.y*r2.x
dtheta  = atan2(cross_z, dot)     folded into [0, 2*pi)
```

`atan2` (rather than `acos(dot/(r1n*r2n))`) is used because `acos` throws away the sign — it cannot
distinguish "80° counterclockwise" from "80° clockwise," which for two BODIES ORBITING THE SAME
PROGRADE DIRECTION matters: `dtheta` computed this way is the **prograde sweep** from `r1` to `r2`,
consistent with both bodies' actual direction of motion. This project solves the **short way**
(`dtheta < pi`, historically "Type I"): `dtheta > pi` (the long way, "Type II", would require the
transfer orbit to sweep more than half the circle to stay prograde) is out of scope for `v1`
(README Exercise 1 implements it).

### The Hohmann transfer — the closed-form optimum this project verifies against

For a transfer between two **coplanar circular** orbits specifically (not general Lambert geometry),
classical orbital mechanics proves the delta-v-minimal transfer is the **Hohmann transfer**: a
transfer ellipse tangent to both circles, departing at `dtheta` exactly `180 deg` (periapsis at `r1`,
apoapsis at `r2`). Its semi-major axis, half-period (the flight time), and delta-v come straight from
**vis-viva** (`v^2 = mu*(2/r - 1/a)`, the two-body energy equation solved for speed):

```
a_h    = (r1 + r2) / 2                    = 1.262 LU
TOF_h  = pi * a_h^1.5                     = 4.453884 TU     (half the transfer ellipse's period)
v1c = sqrt(mu/r1) = 1.000000               v2c = sqrt(mu/r2) = 0.810042    (circular speeds)
vp  = sqrt(2/r1 - 1/a_h) = 1.098912         va  = sqrt(2/r2 - 1/a_h) = 0.721071  (transfer-orbit speeds)
delta_v_h = (vp - v1c) + (v2c - va)        = 0.187883 LU/TU
```

(all six-decimal values above are this project's own scenario — `r1=1`, `r2=1.524` — computed by
`main.cu`'s independent, double-precision `hohmann_ground_truth()`, never by the Lambert solver
itself). **The Hohmann geometry is exactly the universal-variable formulation's singularity**
(`dtheta = pi` makes `A = 0` — see Numerical considerations): the transfer this project is built to
*find the neighborhood of* is precisely the one point its general-purpose solver cannot evaluate. This
tension — and how the grid works around it — is this project's central numerical lesson.

### Where in the grid the Hohmann alignment recurs

Both bodies start at phase angle 0 at `t=0` (`kernels.cuh`'s convention). The Hohmann-optimal
departure epoch (where body 2 sits exactly `pi` ahead of body 1's departure position, timed so it
*stays* there for `TOF_h` more of its own motion) solves

```
t1* * (n2 - n1) = pi - n2*TOF_h   (mod 2*pi)     =>     t1* = 11.759 TU (mod T_syn = 13.412 TU)
```

— which lands inside this project's `[0, 28)` TU departure window (at `t1* = 11.759` TU, with
`t2* = t1* + TOF_h = 16.213` TU also inside the `[0, 28)` arrival window), guaranteeing the grid
contains a neighborhood of the true optimum to search near. A second, later occurrence
(`t1* + T_syn = 25.171` TU) falls partly outside the window and is not fully usable — one is enough.

## The algorithm

Per grid cell `(i, j)` — `t1 = i*dt`, `t2 = j*dt`, `dt = WINDOW_TU / GRID_N` (both axes share `dt`):

1. **Mask on time-of-flight.** `tof = t2 - t1`; if outside `(MIN_TOF_TU, MAX_TOF_TU)`, stop —
   `kStatusMaskedTof`. Serial cost: O(1). This structurally excludes the near-zero-duration
   (unphysical, numerically degenerate) and unreasonably-slow (not a sane mission candidate) regions.
2. **Compute both bodies' state** at `t1`, `t2` — closed form, O(1) (the-math above).
3. **Compute the transfer angle** `dtheta` via `atan2` — O(1). If within `kEpsSingularRad` of `pi`:
   stop, `kStatusNearSingular` (the Hohmann-adjacent singularity). If `dtheta > pi`: stop,
   `kStatusLongWay` (out of scope for `v1`).
4. **Bisect for `z`.** Evaluate `TOF(z) - tof_target` at the two ends of a fixed bracket
   `[kBisectZLo, kBisectZHi] = [-60, 39]`; if they do not bracket a sign change, stop,
   `kStatusNonConverged`. Otherwise run **exactly `kBisectIters = 60` iterations** of bisection — no
   early exit (Numerical considerations explains why) — each iteration evaluating `TOF(z)` once
   (two Stumpff calls). Serial cost per attempted cell: O(60) Stumpff evaluations, each O(1)
   (a handful of `sin`/`cos`/`sqrt` calls).
5. **Recover the transfer velocities** via the Lagrange coefficients and compute `delta_v` — O(1).

**Complexity:** O(1) per cell for masked/long-way/near-singular cells (the majority — see the census
below), O(`kBisectIters`) = O(60) transcendental-function evaluations for the ~18% of cells that reach
step 4. Total grid: O(`GRID_N^2`) cells, each O(1) to O(60) — **embarrassingly parallel**: every cell's
answer is independent of every other cell's, with no shared state, no reduction, no communication —
the ideal case for a batched-solve GPU kernel (the same shape as 33.01's batched linear algebra, here
with a transcendental root-find as "the small numerical problem" instead of a matrix factorization).

**Measured cell census** (this project's committed scenario, `GRID_N=512`, `262,144` total cells;
values on your own run's `[info] cell census` line will match closely):
roughly two-thirds of cells are masked by the time-of-flight band, about a fifth are excluded by the
long-way scope decision, a couple of percent of the *remaining attempted* cells are near-singular, and
the rest (about a sixth of the grid) get a real delta-v. Non-convergence (a failed bracket) is
vanishingly rare given the validated `[-60, 39]` bracket — see the exact measured numbers on your own
run's `[info]` lines, quoted honestly rather than hand-derived here.

## The GPU mapping

```
one thread = one grid cell (i, j), idx = j*grid_n + i  (row-major, matches the PGM artifact layout)
grid = ceil(total_cells/256) x 256   (repo default; ragged tail guarded)

per thread:
  READS:  ZERO global-memory reads — every input is either the tiny by-value
          LambertScenario struct (broadcast via kernel parameter space, a
          fast read-only path) or derived purely from the thread's own idx.
  WRITES: deltav[idx], status[idx] — two coalesced writes, once, at the end.
  COMPUTE: 0 to ~60*O(1) transcendental evaluations, branch-dependent (below).
```

This is the opposite end of the roofline from SAXPY (33.01/the scaffold placeholder): SAXPY moves
12 bytes per 2 FLOPs (bandwidth-bound); this kernel moves 8 bytes (one float + one int) per cell after
up to ~60 iterations of several transcendental calls each (arithmetic-bound, GPU-bound on special-
function throughput, not memory bandwidth).

**Warp-divergence honesty — the point the project brief specifically asks for.** The five
`kStatus*` outcomes are NOT scattered randomly across the grid: they form **contiguous geometric
regions** (a diagonal band from the time-of-flight mask, a half-plane from the long-way exclusion, a
thin ring from the near-singular band). Because `idx = j*grid_n + i` puts 32 *consecutive* `i` values
(one warp's worth) into the same warp for a fixed `j`, and neighboring `i` at fixed `j` usually sit in
the *same* geometric region, **most warps are status-homogeneous** — either every thread in the warp
returns in O(1) (masked/long-way/near-singular) or every thread runs the full ~60-iteration bisection.
Divergence cost — a warp where SOME threads finish in O(1) while others run 60 iterations, leaving the
finished threads idle — concentrates at **region BOUNDARIES**: the thin set of warps whose 32 cells
straddle the time-of-flight mask edge, the `dtheta = pi` line, or the `dtheta = pi/2` diagonal that
separates masked/attempted geometries. This is a general lesson beyond this project: threshold-based
classification kernels divergence-cost is a boundary-length problem, not a cell-count problem — a
smoother (larger) region layout costs less divergence than the same total area cut into many thin
slivers, even at equal total "attempted work."

**Occupancy:** each thread's working set is small (a handful of floats — no arrays, no loops with
dynamic bounds) — register pressure is light, and with no shared memory and no atomics (cells never
interact, by construction) occupancy is limited only by the usual block-size/register trade, not by
this kernel's own memory footprint.

## Numerical considerations

- **Why `A = sqrt(2*r1n*r2n)*cos(dtheta/2)` and not the textbook
  `sin(dtheta)*sqrt(r1n*r2n/(1-cos(dtheta)))`.** The two are algebraically identical (half-angle
  identity: `1 - cos(dtheta) = 2*sin^2(dtheta/2)`, `sin(dtheta) = 2*sin(dtheta/2)*cos(dtheta/2)`), but
  the textbook form is a **removable 0/0** as `dtheta -> 0` (both `sin(dtheta)` and `1-cos(dtheta)`
  vanish) — exactly the kind of catastrophic-cancellation trap FP32 cannot be trusted near. The
  half-angle form has no such issue: it is smooth and well-conditioned for every `dtheta` in `(0, pi)`,
  including near 0. This project uses the stable form everywhere and treats the textbook form as
  algebra-only (never evaluated in code) — a small, concrete lesson in *which* algebraically-equal
  formula to trust in floating point.
- **The GENUINE singularity is at `dtheta = pi`, not `dtheta = 0`.** `A -> 0` as `dtheta -> pi`
  (`cos(pi/2) = 0`) regardless of which formula computes it — this is not a numerical artifact, it is
  a **mathematical fact about the Lambert problem**: at a transfer angle of exactly 180°, the two
  position vectors and the Sun are collinear, so the transfer PLANE (which must contain all three) is
  not uniquely determined — infinitely many planes satisfy the geometry. This project's coplanar 2D
  setup sidesteps the plane-ambiguity itself (there is only one plane, z=0), but the *algebra*
  (`A=0` makes `y(z)`'s formula divide by zero) still breaks, because the universal-variable
  derivation assumes `A != 0` throughout. `kStatusNearSingular` (within `kEpsSingularRad = 2 deg` of
  `pi`) is this project's honest, documented acknowledgment of a real textbook edge case — not a bug
  to be "fixed" by a smaller epsilon, but a fact to be worked around (the Hohmann optimum genuinely
  sits at this project's solver's blind spot, which is *why* the ANALYTIC gate checks the grid's
  *nearest approach* to Hohmann rather than expecting an exact hit).
- **The Stumpff series switchover.** `C(z)`, `S(z)`'s closed forms (`(1-cos(sqrt(z)))/z` etc.) are
  ALSO a removable 0/0 as `z -> 0` — this time in the *algorithm's own working variable*, which the
  bisection sweeps directly through zero on every attempted cell whose optimal `z` is near the
  parabolic boundary. Rather than trust FP32 to resolve a 0/0 near machine epsilon (it will not — the
  numerator and denominator both underflow toward zero at different relative rates, producing garbage
  or a stray NaN), this project switches to the **Taylor series** for `|z| < 1e-6`:
  `C(z) = 1/2 - z/24 + z^2/720 - ...`, `S(z) = 1/6 - z/120 + z^2/5040 - ...` — smooth, well-conditioned
  polynomials that agree with the closed forms to FP32 precision well outside the switchover radius
  (verify this yourself: evaluate both forms at `z = 1e-5` and compare). This is the general pattern
  for "removable singularity in a special function": closed form away from the point, series near it,
  a documented, tested crossover radius in between.
- **Fixed-iteration bisection, no early exit, identical on GPU and CPU.** A converging Newton's method
  would typically finish in under 10 iterations instead of always running all 60 — but Newton's
  iteration COUNT would then depend on the specific cell's `z` trajectory, and comparing "GPU thread
  k's answer after N_k GPU-rounding iterations" against "CPU cell k's answer after N_k
  CPU-rounding iterations" is a much weaker guarantee than comparing two paths that took textually
  IDENTICAL steps (the same reasoning 08.01 applies to its fixed RK4 step count). Bisection trades
  iteration efficiency for this guarantee cheaply: 60 steps shrinks the initial width-99 bracket to
  `99/2^60 ~= 8.6e-17`, far below FP32's `~1.2e-7` relative epsilon — the extra ~30 iterations beyond
  what FP32 could even distinguish are deliberate, cheap, documented overkill for a clean comparison.
- **The `y(z)` floor (`kYFloor = 1e-6`).** The FAR ends of the fixed bracket (`z` near `-60` or `39`)
  can transiently evaluate `y(z) < 0` for geometries far from this cell's true root — `sqrt` of a
  negative number would poison the rest of that cell's bisection with NaN before the search ever
  narrows toward the real answer. Clamping `y` to a tiny positive floor only ever engages during those
  early, far-from-root iterations; by the time bisection has converged (kBisectIters=60 is vast
  overkill, see above), the floor is nowhere near the final `z`.
- **FP32 throughout the solver; FP64 for the independent Hohmann ground truth.** The Lambert solver
  (GPU and CPU) works entirely in FP32 — consistent with the rest of this repository's default and
  perfectly adequate for canonical-unit magnitudes near 1. `main.cu`'s `hohmann_ground_truth()`
  deliberately uses `double` and calls neither `stumpff_c`/`stumpff_s` nor the bisection — an
  independent code path in an independent precision is what makes the ANALYTIC gate a check against
  *mathematics*, not merely a self-consistency check of one code path against itself.

## How we verify correctness

Three genuinely independent checks, because none of them alone rules out every class of bug:

1. **The §5 GPU-vs-CPU gate (VERIFY).** The kernel and `reference_cpu.cpp`'s `solve_cell_cpu()` are
   line-by-line twins — same classification order, same bracket, same fixed iteration count, same
   Lagrange recovery — differing only in `sinf/cosf/rsqrtf/sincosf` vs `std::sin/cos/sqrt` spellings.
   Every cell's status must match (within a small, documented count of boundary-threshold ulp flips —
   `main.cu`'s `kMaxStatusMismatches`), and OK cells' delta-v must agree within relative tolerance
   1e-3. This catches indexing bugs, a mistyped constant, a sign error in `atan2`'s arguments — the
   "the code computes something self-consistent, but is it the RIGHT something" class of bug that a
   single implementation can never catch on its own.
2. **The NaN-policy gate.** The fraction of *attempted* cells (short-way, valid time-of-flight) that
   end up near-singular or non-converged must stay below a documented bound. This catches a bracket
   that no longer covers the true root (a regression that would silently balloon the non-converged
   count) or an `kEpsSingularRad` set so wide it swallows a large chunk of legitimate cells.
3. **The ANALYTIC gate (verification against pure mathematics) — the one that matters most.** Nothing
   about check 1 or 2 can catch a Lambert solver that is *self-consistent but physically wrong* — a
   solver with a subtly incorrect Lagrange-coefficient formula, say, could still agree with its own CPU
   twin bit-for-bit and never hit the singular/non-converged branches, while computing a delta-v field
   that has nothing to do with real orbital mechanics. The Hohmann ground truth (an independent,
   double-precision, textbook-formula computation that never calls the Lambert solver) is the check
   that catches exactly that class of bug: **for two coplanar circular orbits, the global delta-v
   minimum over ALL departure/arrival epochs is PROVABLY the Hohmann value** (a real theorem of
   orbital mechanics, not a property of this project's code). A grid search can only *approach* that
   continuous optimum, never beat it (sampling a smooth function at finite resolution never finds a
   value below the true minimum near a well-behaved minimum), so the grid's own best cell must land
   at-or-above the Hohmann delta-v, within a documented small window whose size is set by the grid's
   resolution and the near-singular exclusion band (`README`'s ratified scope: "measure and document
   the gap" — done on the `[info]` lines of every run, with the exact measured numbers, not asserted
   in advance). The time-of-flight at that minimum cell is checked against the closed-form Hohmann
   half-period the same way. Together, checks 1–3 mean a bug would have to (a) agree with an
   independent CPU implementation, (b) stay within the documented degenerate-cell budget, AND
   (c) still land near the one point real orbital mechanics says is optimal — a combination that is
   very hard for a wrong implementation to satisfy by accident.

## Where this sits in the real world

- **Izzo's algorithm (2015) is what production software actually runs**, not the Stumpff-bisection
  method this project implements. Izzo's approach reformulates the problem around a single scalar
  variable with better-behaved derivatives and uses a Householder (higher-order Newton) iteration that
  typically converges in 2–3 steps with no bracket needed and handles the multi-revolution case
  natively. This project's textbook universal-variable/bisection method is chosen for **transparency**
  (every step is a named, derivable quantity; a bisection needs no derivative and its convergence
  behavior is trivial to reason about) over production speed — exactly the CLAUDE.md §1 "teaching
  beats cleverness" trade, made explicit.
- **poliastro / pykep** wrap Izzo's solver (poliastro; pykep is the ESA-lineage C++/Python library
  Izzo himself contributed to) with real ephemeris queries, unit handling, and porkchop-plot
  convenience functions — the open-source software layer a real mission-design engineer would
  actually use day to day, sitting on the exact algorithm this project's Exercise 1/4/5 would need to
  reach production parity.
- **GMAT** (NASA Goddard / Air Force Research Lab, open source) is the free, full mission-design tool:
  Lambert targeting, real ephemerides (via SPICE), trajectory optimization, and operational-grade
  numerical propagators — the professional-grade version of this entire project's pipeline, and a
  legitimate next tool to learn after this one.
- **Real interplanetary missions** additionally handle: 3D (non-coplanar) transfers with a genuine
  plane-change cost; patched-conic approximations that stitch together planetary sphere-of-influence
  segments (departure hyperbola -> heliocentric transfer -> arrival hyperbola) rather than this
  project's pure heliocentric two-body model; finite-burn (not impulsive) propulsion, which for
  low-thrust electric propulsion turns the whole problem into a continuous trajectory-optimization
  problem instead of a Lambert lookup (domain 17's later projects); and launch-vehicle-specific
  constraints (declination of the launch asymptote, C3 energy limits) that further restrict which
  Lambert solutions in a porkchop plot are actually flyable. This project's porkchop plot is the first,
  necessary, and genuinely representative first step of that whole pipeline — not a toy unrelated to
  it.
