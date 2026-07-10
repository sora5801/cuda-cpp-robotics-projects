# 18.01 — Snake robots: serpenoid gait sweeps coupled to granular sim: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why snakes bother to wiggle.** A snake has no legs and no wheels; its only tools for moving are its
own body shape and whatever it is touching. Biologists (Gray, 1946) traced the mechanism to the
snake's **scales**: they are angled so that sliding the belly *forward* meets almost no resistance,
while sliding *sideways* meets a great deal — an **anisotropic friction surface**, engineered by
evolution the way a ratchet is engineered by a machinist. Hirose (1993) turned this observation into
an equation robotics could build from: the **serpenoid curve**, a body shape whose local curvature
varies sinusoidally along its length and sinusoidally in time — the mathematical description of
"a wave traveling down the body." This project asks a very concrete engineering question about that
curve: **for a snake with a GIVEN friction anisotropy, which serpenoid wave (what amplitude, what
wavelength, what speed) moves it fastest?** That is a genuine design-space search — exactly the kind
of embarrassingly parallel sweep a GPU is built for.

**The physics of the ground contact.** Model each of the snake's 12 rigid links as resting on flat
ground, in continuous contact, carrying its share of the snake's weight. When a link slides, Coulomb
friction opposes the sliding with a force proportional to the normal load (here, simply the link's own
weight `m_link * g` — there is no vertical dynamics in this top-down, flat-ground model) and
*independent of speed* once sliding, in the direction opposing the local velocity. The one twist that
makes locomotion possible: **the coefficient is not the same in every direction**. Along the link's own
long axis (the direction its "scales" or wheels would glide), the coefficient `mu_t` (tangential) is
LOW; across the link (the direction its side would have to plow through the ground), the coefficient
`mu_n` (normal) is HIGH. THEORY's derivation below shows precisely why `mu_t << mu_n` is not just
helpful but *necessary* — with `mu_t = mu_n`, the propulsion this project measures collapses to 6.3%
of its anisotropic value (README §Expected output, `GATE_ISOTROPIC_FRICTION`).

**The engineering frame.** A real snake robot's joints are driven by embedded servo motors closing a
position loop at hundreds of Hz to kHz (SYSTEM_DESIGN §1.1) — several orders of magnitude faster than
anything this project computes. This project's entire job is *upstream* of that loop: pick the
`(A, beta, omega)` that will be commanded, once, before the robot starts moving (or occasionally, if
terrain friction is re-estimated online) — see README §System context. That separation of concerns
(slow gait *design* vs. fast joint *execution*) is why the "perfect joint tracking" assumption below
is defensible: this project is not claiming a real servo tracks its reference with zero error, only
that the SHAPE this project searches over is the one a real controller would be told to track.

**The anisotropic-friction propulsion mechanism, derived.** Consider one link, moving with center
velocity `v` that is generally NOT aligned with the link's own long axis (because the body is bending
under it). Decompose `v` into a component `v_t` along the link (tangent direction `e_t`) and a
component `v_n` across it (normal direction `e_n`):

```
  ONE LINK, mid-wave — its own long axis is the diagonal line; the wave
  gives it a velocity v that points somewhere ELSE:

                          e_n  (normal: HIGH friction mu_n)
                           ^
                           |          v  (this link's actual center
                           |         /|   velocity, set by the wave)
                           |        / |
                           |       /  |  v_n = v . e_n
                           |      /   |  (the ACROSS-link component —
                           |     /    |   the wave's "sideways kick")
       ====================|====/=====|============>  e_t (tangent:
              link i's body (a rigid rod)             LOW friction mu_t)
                             \__________/
                                 v_t = v . e_t
                            (the ALONG-link component —
                             the "glide" component)

  Friction force:  F = -mu_t*(m*g)*sign(v_t)*e_t  -mu_n*(m*g)*sign(v_n)*e_n
                        \_____small, weak brake____/  \____big, strong brake___/

  F is NOT anti-parallel to v (it would be, if mu_t = mu_n) — it is
  rotated TOWARD e_t, because the machine almost completely cancels v_n
  but barely touches v_t.
```

**Why that rotation produces net forward thrust — the non-holonomic argument.** Push `mu_n` toward
infinity (a thought experiment; this project uses a large but finite value) and the friction law stops
being a *force* and becomes a *constraint*: `v_n -> 0` at every instant, for every link — the link is
physically forbidden from sliding sideways, exactly like a wheel or an ice skate's blade (this is a
**non-holonomic velocity constraint**, the same mathematical object that lets a bicycle or a parked car
translate sideways through a sequence of steering-and-rolling moves that individually never move it
sideways at all — the "parallel parking" trick, formalized as **geometric mechanics** / the
"falling cat" theorem). With every link pinned against sideways slip but free to glide fore-and-aft,
the ONLY way the snake's *internal* shape change (the traveling wave) can be realized at all is by the
whole body *translating* — the wave does not just wiggle in place, it is kinematically forced to walk
the body forward as it passes. Set `mu_t = mu_n` and that preferred direction disappears: friction
resists whatever the local velocity happens to be, in whatever direction it happens to point, with no
"easy way out" — the traveling wave still moves the body a little (this project's isotropic gate
measures a real but small residual, not exactly zero — a fully reciprocal, time-symmetric gait would
be closer to the classical *scallop theorem*'s exact-zero result for isotropic viscous drag, but a
traveling serpenoid wave is not perfectly reciprocal), yet loses the overwhelming majority of its
thrust. `mu_n / mu_t = 7.0` in this project's committed scenario (`0.70 / 0.10`) is squarely in the
regime real engineered snake-robot skins/wheels target (Hirose's own robots used passive wheels for
exactly this reason — PRACTICE.md §2).

## The math

**State (SI, right-handed, top-down world frame — SYSTEM_DESIGN §3.2's `x`-forward/`y`-left/`z`-up
convention specialized to a flat-ground, 2-D problem):** the snake's dynamic degrees of freedom are
the HEAD link's pose and velocity, `x, y` (m), `yaw` (rad, 0 = +x axis, CCW positive),
`vx, vy` (m/s), `yaw_rate` (rad/s). Every other link's pose/velocity is forward kinematics off this
state (below) — never an independent dynamic variable.

**The serpenoid gait (Hirose, 1993).** Joint `j` (`j = 0 .. 10`, connecting link `j` and link `j+1`)
carries the PRESCRIBED angle

```
phi_j(t) = A * sin(omega*t + j*beta) + gamma
```

with `A` the amplitude (rad), `beta` the inter-joint phase offset (rad, sets the body wavelength:
`wavelength ~= 2*pi*link_len / beta` for small `beta`), `omega` the temporal frequency (rad/s), and
`gamma` a common offset (rad, the turning bias — zero for the main speed sweep). Its time derivative,
needed everywhere below, is `phidot_j(t) = A*omega*cos(omega*t + j*beta)`.

**Forward kinematics (the shape).** With link 0 (the head) at `(x, y)`, orientation `yaw`, every other
link's orientation is the CUMULATIVE sum of the joint angles between it and the head:

```
theta_0 = yaw
theta_{j+1} = theta_j + phi_j(t),   j = 0 .. 10
```

and, treating each link as a rigid rod of length `L` (`link_len_m`) whose two ends meet its neighbors:

```
p_0 = (x, y)
p_{i+1} = p_i + (L/2)*e_t(theta_i) + (L/2)*e_t(theta_{i+1}),   e_t(th) = (cos th, sin th)
```

Differentiating both recurrences in time (with `theta_dot_0 = yaw_rate`, `theta_dot_{j+1} =
theta_dot_j + phidot_j(t)`) gives every link's angular and linear velocity as an EXACT, closed-form
function of the current state and the current gait phase — `src/kernels.cuh`'s `snake_step()` computes
both recurrences in a single forward pass.

**Anisotropic friction (per link `i`, THEORY's derivation above).** With `n = (-sin(theta_i),
cos(theta_i))` the link's normal and `v_i` its center velocity:

```
v_t = v_i . e_t(theta_i)          v_n = v_i . n
f_t = -mu_t * (m_link*g) * v_t / sqrt(v_t^2 + eps^2)      (smoothed signum — see Numerics)
f_n = -mu_n * (m_link*g) * v_n / sqrt(v_n^2 + eps^2)
F_i = f_t * e_t(theta_i) + f_n * n                          (force, world frame, N)
```

**Newton-Euler for the whole snake (the prescribed-joint 3-DOF reduction).** Because the shape is
KNOWN (not solved for), the only unknowns are the head's pose — summing every link's friction force
and its torque about the head position `(x, y)` gives the whole-body force/torque balance directly,
with NO constraint-force solve required (contrast a general N-link multibody chain, which needs one):

```
F_net     = sum_i F_i
tau_net   = sum_i (p_i - (x,y)) x F_i           (2-D cross product, scalar z-component)
a         = F_net / M,        M = N_LINKS * m_link
alpha     = tau_net / I_eff,  I_eff = (1/3) * M * (N_LINKS*L)^2   (nominal rod-about-its-end inertia)
```

Semi-implicit Euler advances `(vx, vy, yaw_rate)` from `(a, alpha)` first, then `(x, y, yaw)` from the
UPDATED velocity — `src/kernels.cuh`'s Numerical considerations below explains why that ordering
matters.

**Cost of transport.** A per-joint actuator torque is ESTIMATED (not solved for — no joint dynamics
exist to solve) via a free-body "cut" argument: cutting the chain at joint `k`, the torque the
actuator must supply to hold the prescribed angle against the ground reaction is approximated as the
net friction torque of every link DOWNSTREAM of the cut, about the joint's own location:

```
tau_k ~= sum_{i > k} (p_i - joint_k) x F_i
```

(computed in `O(N_LINKS)` via a backward running suffix sum — `snake_step()`'s second pass). The raw
effort integral and the normalized cost of transport are then

```
effort = integral sum_k |tau_k(t) * phidot_k(t)| dt          (J; a work-magnitude proxy)
COT    = effort / (M * g * distance)                          (unitless; standard robotics COT form)
```

## The algorithm

Per gait (the whole body of `simulate_gait()`), for `n_steps = T_sim / dt` steps (committed scenario:
8,000 steps of 1 ms each):

1. **Forward pass** (`O(N_LINKS)`, link 0 → 11): evaluate the serpenoid formula at every joint,
   accumulate each link's orientation/position/velocity from the recurrences above, compute each
   link's anisotropic friction force, and accumulate the whole-body `F_net`/`tau_net` as you go.
2. **Backward suffix pass** (`O(N_LINKS)`, link 11 → 1): the per-joint torque estimate for the cost-
   of-transport integral, via a single running suffix sum (not the naive `O(N_LINKS^2)` all-pairs sum
   the free-body argument would suggest at face value).
3. **Semi-implicit Euler integration**: `O(1)` — advance the 6-element body state.
4. Repeat 8,000 times; reduce the resulting trajectory to `(distance, path_length, straightness,
   effort, cot)` (`O(1)` at the end).

**Complexity.** One gait costs `O(N_LINKS * n_steps)` serial work (~96,000 link-visits for the
committed scenario). The full sweep is `O(G * N_LINKS * n_steps)` total work across `G = 8,192`
INDEPENDENT gaits — serially, `O(G)` times the single-gait cost (measured: ~34 ms for 32 gaits on one
CPU core extrapolates to ~8.7 s for all 8,192); in parallel, one thread per gait turns the `G` factor
into a **width** the GPU fills instead of a **multiplier** the wall clock pays (measured: ~80 ms for
all 8,192 — see §The GPU mapping).

## The GPU mapping

```
one thread = one gait g = (a_idx*n_beta + b_idx)*n_omega + w_idx
grid = ceil(G/128) x 128           (see kernels.cu for why 128, not the
                                     repo-default 256 — larger per-thread
                                     register footprint than 08.01/10.03)

per thread g, for its whole 8,000-step lifetime:
  registers/local : the ENTIRE state (6 floats) + snake_step()'s per-link
                     scratch (6*N_LINKS + (N_LINKS-1) = 83 floats) — no
                     global memory touched between the kernel's start and
                     its six final writes
  compute          : ~46 sinf/cosf calls per step (1 for the head's own
                     orientation once, then 2 per joint for the serpenoid
                     phase + 2 per joint for the new link orientation,
                     x 11 joints) x 8,000 steps x 8,192 threads
                     ~= 3.0 BILLION transcendental evaluations
  global writes    : SIX coalesced floats at the very end (distance,
                     straightness, cot, effort, final_x, final_y) — a
                     warp's 32 threads write 32 consecutive floats to
                     each output array, the textbook coalesced pattern
```

**Why this kernel is compute-bound, not memory-bound (contrast SAXPY/33.01).** Every thread's "input"
(four gait scalars plus the shared friction pair, all KERNEL PARAMETERS — effectively free, broadcast
reads) and "output" (six floats) are tiny; the entire cost is the `sinf`/`cosf`-heavy arithmetic of
8,000 sequential integration steps. Measured: **~80 ms of GPU kernel time for the whole 8,192-gait
sweep** on an RTX 2080 SUPER — roughly 800 million gait-steps/second, i.e. the Special Function Units
computing billions of `sinf`/`cosf` calls, not the memory controller, set the wall clock.

**Register footprint and occupancy.** `snake_step()`'s per-link scratch arrays (`PX, PY, FX, FY, TX,
TY[N_LINKS]`, `phidot[N_LINKS-1]` — 83 floats) are much larger than 08.01's 4-float rollout state or
10.03's per-environment scalars — this project's kernel therefore likely runs at lower occupancy per
SM than either sibling. It does not need higher occupancy to saturate the GPU: `G = 8,192` gaits is
already enough independent work to fill an RTX 2080 SUPER's 46 SMs many times over even with a modest
number of resident warps per SM, so — unlike 08.01/10.03 — this kernel's launch wrapper never needed a
block-count cap.

**What a CUDA library would do here:** nothing — there is no batched-solve, no matrix, no reduction
library call anywhere in this kernel; it is 8,192 independent scalar-and-small-array integrations, the
purest "many small problems, one thread each" pattern in this repo's sampling family.

## Numerical considerations

- **The Coulomb-friction stiffness problem.** True Coulomb friction is a DISCONTINUOUS function of
  velocity (a signum). Integrating that directly at ANY finite `dt` makes the friction force chatter
  once a link's velocity oscillates near zero — every step can flip its sign, injecting spurious
  numerical energy. The standard, honest fix (used throughout `link_friction_force()`) is a SMOOTHED
  signum, `v / sqrt(v^2 + eps^2)`, with `eps = 0.01 m/s` chosen well below the gaits' typical link
  speeds (the fastest measured gait's head alone moves at ~0.54 m/s average, and individual links move
  faster still during a wave's peak) — large enough to kill the chatter, small enough that genuine
  sliding is barely perturbed.
- **Why semi-implicit (symplectic) Euler, not explicit Euler.** Updating velocity from the CURRENT
  force, then position from the UPDATED velocity, is unconditionally better-behaved for velocity-
  dependent force laws like Coulomb friction than the fully explicit alternative (which can pump
  energy at the friction discontinuity) — the same class of reasoning that leads game/robot physics
  engines to symplectic integrators. It has a second, purely bookkeeping benefit here: `path_len_accum
  += |v_new|*dt` is then the EXACT polyline length of the path actually taken, not an approximation of
  it, because `(dx, dy) = v_new*dt` by construction.
- **RK4 was not used, deliberately.** 08.01/10.03 both use RK4 for their smooth ODEs; this project's
  right-hand side is only `C^1` (the smoothed-signum friction law, not smooth), so a higher-order
  integrator buys little extra accuracy per step while costing 4x the `sinf`/`cosf` evaluations already
  measured as the bottleneck (§The GPU mapping) — semi-implicit Euler at `dt=1 ms` is the honest match
  for a problem whose stiffness lives at the friction law, not the rotational kinematics.
- **`fmaf()` discipline: NOT enforced here, unlike 10.03.** 10.03's `kernels.cuh` explicitly writes
  every multiply-add as `fmaf()` to remove host/device rounding-order differences as a source of GPU-
  vs-CPU divergence, leaving only `sinf`/`cosf` as the residual. This project does NOT — ordinary `+=`/
  `*` are used throughout `snake_step()`, so nvcc's default FP-contraction (device code silently fuses
  `a*b+c` into one rounding step) and cl.exe's default non-fusing behavior (`/fp:precise`, two rounding
  steps) both contribute to the measured GPU-vs-CPU gap, alongside `sinf`/`cosf`. The measured result
  — **1.371e-06 m worst-case divergence after 8,000 chained steps** — shows this extra discipline was
  not NEEDED to hit a documented, comfortable tolerance (1.0e-03 m, ~700x margin) for this project's
  verification purposes; Exercise material for a learner who wants to chase the last order of
  magnitude of reproducibility is to add it and re-measure.
- **Determinism.** There is no RNG anywhere in this project (CLAUDE.md §8) — every gait's trajectory is
  a pure function of its four parameters, bit-reproducible on a GIVEN machine/build across runs. Only
  cross-architecture `sinf`/`cosf` implementation differences (or a different compiler's FP-contraction
  choices) could shift results on ANOTHER machine — which is exactly why the checked demo output
  (`demo/expected_output.txt`) carries no raw trajectory numbers, only PASS/FAIL verdicts with measured,
  documented margins (README §Expected output).
- **Why the fastest gait is not the straightest — a numerics-adjacent honesty note.** Every gait starts
  "mid-wave" (`simulate_gait()`'s initial condition is `x=y=yaw=0` at rest, but `phi_j(0)` is generally
  NOT zero — see `kernels.cuh`). That arbitrary starting phase is not symmetric between the two
  "halves" of a wave cycle, so a gait's net yaw drift over a FINITE run (8 s, a handful of gait
  periods) is a genuine, deterministic, non-cancelling artifact of that starting phase — not integrator
  error. README §Limitations documents the measured consequence.

## How we verify correctness

Two independent kinds of check, because a design-space search can be *numerically right and
behaviorally uninteresting* (or vice versa) — the same two-kinds-of-check philosophy 08.01 and 10.03
use, applied to a sweep instead of a controller or a farm:

1. **The §5 GPU-vs-CPU gate:** 32 of the sweep's OWN 8,192 gait indices (stride-sampled: `idx[k] =
   k*G/32`, spanning every corner of the amplitude/phase/frequency grid, not just one region) are
   recomputed from scratch, sequentially, on the CPU, calling the exact same `snake_step()` the kernel
   calls (`src/kernels.cuh`'s `HD` sharing — see that file's header). Final head position is compared,
   absolute tolerance, because position legitimately passes near zero for slow/degenerate gaits (a
   relative tolerance would be meaningless there — the same reasoning 10.03 gives for its own absolute
   state tolerance). Measured worst case: **1.371e-06 m** over 8,000 chained FP32 steps, against a
   documented bound of **1.0e-03 m** — ~700x headroom, sized to comfortably absorb a different GPU
   architecture's independently-rounded `sinf`/`cosf` while still catching any real indexing/formula
   bug instantly (such a bug would show up at order 1, e.g. a wrong `beta` sign misplacing an entire
   gait by tens of centimeters, not micrometers).
2. **Four physics gates**, each testing a PREDICTION this section derived, not just "did the numbers
   change": zero amplitude must give exactly zero displacement (a statement about `snake_step()`'s own
   algebra, not a tolerance-bounded approximation — measured `0.000e+00 m`); isotropic friction at the
   best gait must collapse propulsion (measured 6.3% of the anisotropic speed, against a 20% bound);
   a turning bias must shift final heading in the documented direction (measured **+1.014 rad** for
   `gamma=+0.15 rad` at a representative interior gait — see the note below on WHY an interior gait,
   not the sweep's own speed-optimal one, is the honest place to measure this); and speed vs. amplitude
   must show the classic interior ridge (measured: peak clears both grid edges by well over 10x the
   documented 15% margin).
   - **A design decision made from a real measurement, documented honestly:** the turning-bias gate is
     evaluated at a grid-CENTER gait, not at `best_gp` (the sweep's fastest gait). Early development
     measured the "obvious" version — apply `gamma` directly to `best_gp` — and found the SIGN
     appeared to invert (`best_gp` sits at the swept `beta` range's own lower boundary and already
     accumulates ~1.7 rad of its own yaw drift over 8 s even at `gamma=0`, §Numerics above; near that
     edge case, the turning bias's effect is not simply additive relative to the raw final yaw). Rather
     than force the "expected" sign at an admittedly near-degenerate operating point, the gate moved to
     a well-conditioned interior gait and compares final yaw WITH vs. WITHOUT the bias (isolating
     gamma's own effect from the gait's intrinsic drift) — the honest fix, not a tuned-until-it-passes
     one (CLAUDE.md's "never fake a pass").

## Where this sits in the real world

- **Granular/DEM coupling (the catalog bullet's other half — project 10.10).** This project's ground
  model is a *reduced, published* simplification: rigid anisotropic Coulomb friction on a flat,
  non-deformable surface. A full treatment couples the snake to a discrete-element (DEM) simulation of
  actual grains (sand, gravel, loose soil) — each grain a simulated rigid body, contact forces resolved
  between thousands to millions of grain-grain and grain-snake contacts per step. That coupling would
  let this project answer questions it currently cannot: does the optimal gait change when the ground
  can be pushed aside rather than just slid across (real sand-swimming snakes/robots exploit exactly
  this, e.g. the "sandfish" lizard studies Hu et al. 2009 build on)? Does a wave that works on rigid
  ground fail (or improve!) on loose sand? 10.10's DEM engine is the natural upstream/downstream
  partner project for exactly this question.
- **Full multibody dynamics.** Transeth et al.'s 2009 survey catalogs the full range of snake-robot
  models, from this project's prescribed-joint kinematic-dominant reduction up through complete
  Featherstone-class articulated-body dynamics with real joint actuator dynamics, backlash, and
  friction in the JOINTS themselves (not just the ground contact) — production snake-robot control
  research (and this repo's own 09.01, batched forward kinematics/Jacobians) is the on-ramp to that
  fuller treatment.
- **3-D gaits.** This project is planar (lateral undulation only) by catalog-bullet scope. Real snake
  robots also perform **sidewinding** (a 3-D gait that lifts sections of the body clear of the ground
  entirely, trading continuous friction contact for a sequence of discrete footholds — used on loose
  sand where lateral undulation performs poorly) and **rolling/helical** gaits (for pipe-interior
  locomotion, PRACTICE.md §1) — both require a 3-D (not top-down) contact and gravity model this
  project's scope deliberately excludes; documented here, not implemented.
- **Real hardware comparison.** CMU's Biorobotics Lab modsnake robots and HEBI Robotics' commercial
  snake-arm/snake-robot products use serial elastic actuator modules with PASSIVE WHEELS at each
  segment — engineering their own anisotropic friction directly into the hardware, rather than relying
  on a scaled skin, which is both more controllable and more repeatable than a biological or bio-
  inspired surface texture (PRACTICE.md §1-2 for the construction detail). Their onboard controllers
  close joint-tracking loops at rates this project's "perfect tracking" assumption stands in for.
- **What the full research version adds beyond this teaching core:** online (not offline-swept) gait
  adaptation from measured terrain friction; full rigid-multibody dynamics with joint-level actuator
  models; DEM-coupled deformable-terrain locomotion (10.10); 3-D gait families (sidewinding, rolling);
  and closed-loop heading correction (README Exercise 3 is this project's on-ramp to that last one).
