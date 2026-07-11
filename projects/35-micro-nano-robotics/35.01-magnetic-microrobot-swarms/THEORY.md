# 35.01 — Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

### Why the micro-world needs its own physics before it needs a robot

Everything about how this project's "robots" move is a consequence of one fact: at the size of a
bacterium, **the physics that dominates your intuition (inertia, gravity) is negligible, and physics
you never think about (viscosity, thermal noise, surface forces) dominates instead.** This section
derives that fact with real numbers, because a controller designed on the wrong intuition (e.g. "coast
after the force turns off") would be wrong by ten orders of magnitude here.

**Scale analysis: the Reynolds number.** The Reynolds number `Re = rho*v*a/mu` (dimensionless) compares
inertial forces (`rho*v^2*a^2`, from momentum) to viscous forces (`mu*v*a`, from Stokes drag) for a body
of size `a` moving at speed `v` through a fluid of density `rho` and dynamic viscosity `mu`. For this
project's committed scenario — a bead of radius `a = 5 um` moving at the demo's own measured drift speed
(`GATE_ATTRACT`'s East-only run: ~342 um displacement in 50 steps of `dt=0.5 s` = 25 s, so
`v ~ 13.7 um/s`) through water (`rho = 1000 kg/m^3`, `mu = 1.0e-3 Pa*s`):

```
Re = (1000 kg/m^3) * (1.37e-5 m/s) * (5e-6 m) / (1.0e-3 Pa*s) ~ 7e-5
```

(`main.cu`'s own `[info] Re ~` line prints a similar order-of-magnitude estimate, `~1.1e-4`, from a
slightly different representative-velocity proxy — both agree on the number that matters: **`Re` is
five orders of magnitude below 1.**) At `Re << 1` (the "Stokes flow" or "creeping flow" regime),
inertia is not just small, it is *irrelevant*: if every force on the bead vanished right now, it would
stop moving within nanoseconds (compare the momentum relaxation time `m/gamma` — a bead's mass is
picograms, its drag coefficient is ~1e-7 N*s/m, giving a stopping time near a microsecond, itself tiny
next to this project's 0.5-second timestep). This is *why* the dynamics this project simulates are
**first-order**: `velocity = force / drag`, full stop — there is no `F = m*a` term to integrate, no
momentum to conserve, no coasting. Section "The math" below makes this the actual equation of motion.

**Why "swimming" is weird down here — Purcell's scallop theorem, honestly.** A famous consequence of
`Re << 1` is E. M. Purcell's **scallop theorem**: any *reciprocal* motion (a stroke that looks the same
played forwards or backwards, like a scallop's single hinge opening and closing) produces **zero net
displacement** at low Reynolds number, because the fluid's response is purely determined by the
instantaneous shape, with no inertia to "remember" the stroke's direction. This is why real
microorganisms swim with corkscrewing flagella or beating cilia (genuinely non-reciprocal motions), not
by flapping a single paddle. **This project's microrobots do not swim at all** — they are pulled by an
*external* magnetic field gradient (a body force applied from outside, like gravity pulling a ball, not
a self-generated stroke), so the scallop theorem simply does not apply to them; it is mentioned here
because it is the single most important idea in low-Re locomotion, and an honest theory section names
what its own project is *not* doing, and why the difference matters (35.02, "Low-Reynolds-number
swimming: Stokes-flow boundary element solvers," is this repository's project about the scallop
theorem's actual subject — self-propelled microswimmers).

**Why surface/viscous forces dominate over gravity.** A sphere's weight (net of buoyancy) scales with
volume, `~a^3`; viscous drag scales with `~a`. Their ratio shrinks as `a^2` as the bead gets smaller. At
`a = 5 um`, a bead of typical density (~1.5x water) settles under gravity at a terminal (Stokes) velocity
of order `0.01-0.1 um/s` — one to three orders of magnitude below this project's magnetically-driven
drift speed. Gravity is not literally zero here, but it is a small, constant, downward bias this
project's 2D in-plane model does not need to resolve (PRACTICE.md §1 discusses the real hardware's
vertical confinement).

**Thermal noise: computing whether Brownian motion matters, honestly.** A bead in a fluid is
continuously kicked by thermal collisions with fluid molecules — Brownian motion. The Stokes-Einstein
relation gives the diffusion coefficient `D = k_B*T / (6*pi*mu*a)`, and the RMS displacement over a time
`dt` along one axis is `sqrt(2*D*dt)`. For this project's bead (`a=5 um`) in water at room temperature
(`T=293 K`, `k_B=1.380649e-23 J/K`) over one integration step (`dt=0.5 s`):

```
D  = (1.380649e-23 * 293) / (6*pi*0.001*5e-6)  ~ 4.3e-14 m^2/s
RMS step = sqrt(2*D*0.5)                        ~ 0.21 um
```

`main.cu` computes and prints this exact number every run (`[info] Brownian RMS step ~ 0.207 um`),
alongside the deterministic drift over the same step (`~10.9 um` at this scenario's typical gradient) —
**the deterministic drift is roughly 50x the thermal jiggle at this bead size and timestep.** That ratio
is *why* this project's default is a **deterministic** simulation with Brownian motion switched off
(THEORY.md "Numerical considerations" restates this as a modeling decision, not just a measurement): the
physics genuinely supports treating the swarm's *centroid* motion as a deterministic drift-dominated
process at this scale, while honestly noting that (a) a much smaller bead, or (b) a much weaker
gradient, would flip this ratio and make Brownian motion the dominant term — exactly the regime
real single-molecule and nanoparticle magnetic manipulation experiments must contend with (35.03,
"Brownian dynamics of nanorobots," is this repository's project for that regime).

**The engineering task.** Given all of the above, the task this project teaches is: **given a fixed
arrangement of electromagnet coils around a small workspace, compute the magnetic field they produce,
and use the field's *gradient* to pull a swarm of magnetically susceptible microparticles along a
planned path** — the foundational capability behind targeted drug delivery, microscale assembly, and
lab-on-chip particle sorting (README "System context" and PRACTICE.md §3–4 place this in the real
research and commercial landscape). The engineering constraints a real system imposes — coil heating,
amplifier bandwidth, workspace imaging for feedback, field linearity limits — are named honestly in
PRACTICE.md rather than modeled here; this project's teaching core is the **field computation and the
open-loop dynamics**, the ratified reduced scope for this `[R&D]` catalog bullet (README "Limitations &
honesty").

## The math

### Biot-Savart from Maxwell (building on 24.01's groundwork)

[`24.01`](../../24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/THEORY.md)
already derives magnetostatics from Maxwell's equations for a *continuous* current distribution
(`-div(nu*grad(A_z)) = J`, solved as a field-continuum PDE). This project needs the same underlying
law — Ampere's law in its differential form, `curl(B) = mu0*J` (magnetostatic, no displacement current)
— but for a **discrete set of thin wires** rather than a continuous current sheet, so it uses the
integral form's direct consequence instead: the **Biot-Savart law**,

```
B(r) = (mu0/4*pi) * INTEGRAL[ I * dl x (r - r') / |r - r'|^3 ]
```

(`r'` ranges over the wire, `dl` is an infinitesimal length element of it, `r` is the field point). This
is the exact solution of `curl(B)=mu0*J`, `div(B)=0` for a filament current (24.01 solves the same
Maxwell equations with a mesh-based PDE relaxation because its geometry — magnet/iron/air regions with
different permeabilities — has no closed-form Green's function; this project's geometry — thin wires in
vacuum/air, `mu_r ~ 1` everywhere relevant — has one, and Biot-Savart *is* that Green's function,
integrated). Discretizing each of this project's 4 circular coils into `segs_per_coil` straight
segments turns the integral into the sum the catalog bullet names explicitly:

```
B(r) = (mu0/4*pi) * SUM_over_segments[ I_c * dl_s x (r - r_s) / |r - r_s|^3 ]
```

where `r_s` is segment `s`'s midpoint, `dl_s` is its vector (direction = current sense, magnitude =
segment length), and `I_c` is the ampere-turn current on segment `s`'s coil `c`. `kernels.cuh`'s
`biot_savart_contribution` is this formula, verbatim, for one segment; `kernels.cu`'s
`biot_savart_basis_kernel` is the sum over all segments, one thread per field point.

**Why "ampere-turns," not amperes.** A real coil is not one loop of wire but `N` turns wound together;
each turn carries the same current `I_wire`, and by superposition their fields simply add — `N` turns at
`I_wire` amps produce the same field as one turn at `N*I_wire` amps. This project's `I` is always this
product (PRACTICE.md §2 gives illustrative `N`/`I_wire` splits for the committed scenario's 500 A-turns)
— a bookkeeping simplification that costs nothing physically because Biot-Savart is linear in current
(see below), and matters for READ ability: a control engineer reading "500 A-turns" instantly knows to
divide by the coil's actual turn count to get the wire current an amplifier must supply.

### Linearity: the field of ANY current combination is a linear combination

Biot-Savart is **linear in the source current** — doubling `I` doubles `B` everywhere, and the field of
two coils driven simultaneously is the *sum* of each coil's field driven alone (superposition, a direct
consequence of Maxwell's equations being linear in vacuum/air). Formally, if `B_c(x)` is the field coil
`c` produces per unit ampere-turn (this project's "basis map"), then for ANY current vector
`I = (I_E, I_W, I_N, I_S)`:

```
B(x; I) = I_E*B_E(x) + I_W*B_W(x) + I_N*B_N(x) + I_S*B_S(x)
```

This single fact is exploited **twice** in this project (kernels.cuh's file header names both uses): it
is *why* the expensive 720-segment Biot-Savart sum needs to run only 4 times total (once per coil, at
unit current) rather than once per current configuration ever needed — every subsequent field (each of
the 3 schedule phases, the GATE_ATTRACT probes, the illustrative artifact) is a cheap 4-term weighted sum
of already-computed maps (`combine_field_kernel`); and it is *how* this project's open-loop schedule was
DESIGNED (see "The algorithm" below) — a single forward simulation using this same linear model, run
once, offline, before the real swarm ever moves.

### The Helmholtz condition: why offset = radius/2 flattens the field

A single circular loop's on-axis field, `B(z) = mu0*I*R^2 / (2*(R^2+z^2)^1.5)` (this project's
`GATE_ONAXIS` closed form, `z` measured from the loop's own center along its axis), is a *peaked*
function — maximal at `z=0`, falling off on both sides. Two identical coils sharing an axis, separated
by distance `d`, with currents circulating in the SAME absolute sense (so their on-axis fields *add*),
produce a combined field that is symmetric about their shared midpoint by construction; the FIRST
derivative of that combined field vanishes at the midpoint automatically (odd symmetry). The **Helmholtz
condition**, `d = R` (this project's coils: `offset_m = R/2` on each side of the origin, i.e. separation
`= R`), is the SPECIFIC separation at which the SECOND derivative *also* vanishes at the midpoint —
found by differentiating the two-coil sum twice with respect to `z` and solving for the `d` that zeroes
the result at `z=0`. With both the first and second derivatives zero, the field near the midpoint is flat
to third order in `z` — the textbook "Helmholtz pair" result this project's `GATE_HELMHOLTZ` verifies
numerically (measured variation over the actual 8 mm workspace: ~0.18%, comfortably inside the
documented 2% tolerance) rather than merely asserting.

### The superparamagnetic force law: F = grad(m . B), derived to F = k * grad(|B|^2)

A **superparamagnetic** bead (this project's microrobot model — a polymer/silica bead embedded with
iron-oxide nanoparticles, PRACTICE.md §2 names commercial examples) has NO permanent magnetization; it
acquires a magnetic moment `m` proportional to the LOCAL applied field, in the field's own direction:

```
m = (V * chi_eff / mu0) * B
```

(`V` = bead volume, `chi_eff` = dimensionless effective volume-susceptibility contrast between the bead
and the surrounding fluid, both `chi_eff` and `V` fixed per-bead constants in `SwarmScenario`). The
potential energy of a magnetic dipole `m` in a field `B` is `U = -m . B`, and force is the negative
gradient of potential energy, so:

```
F = -grad(U) = grad(m . B) = (V*chi_eff/mu0) * grad(B . B) ... [substituting m, and B.B is not
                                                                 quite what grad(m.B) gives yet — see below]
```

Being careful (`m` itself depends on position through `B(x)`, but for the STANDARD superparamagnetic
force derivation used throughout the magnetic-manipulation literature, one treats `m` as instantaneously
aligned and proportional to the local `B`, and asks for the force on that induced dipole in the AMBIENT
field gradient — the `(m . grad)B` form, not `grad(m.B)`, which would double-count the field's own
position dependence through `m`). The correct, standard result (Cummings' law):

```
F = (V*chi_eff/mu0) * (B . grad) B
```

**The key simplification this project exploits**: in a CURRENT-FREE region (everywhere this project's
swarm lives — the workspace has coils around it, not inside it, so Ampere's law gives `curl(B) = 0`
there), the identity `(B . grad) B = grad(|B|^2 / 2)` holds EXACTLY (a standard vector-calculus identity:
`grad(|B|^2/2) = (B.grad)B + B x curl(B)`, and the cross term vanishes when `curl(B)=0`). Therefore:

```
F = (V*chi_eff/mu0) * grad(|B|^2 / 2) = k_force * grad(|B|^2/2),   k_force = V*chi_eff/mu0
```

This is `kernels.cuh`/`kernels.cu`'s exact force law (`Fx = k_force*0.5*dB2dx`, `Fy = k_force*0.5*dB2dy`
in `swarm_step_kernel`/`swarm_step_cpu`) — **the force depends on the GRADIENT OF THE FIELD MAGNITUDE
SQUARED, not on the field's direction.** This is why a bead is always pulled toward regions of HIGHER
`|B|` — i.e. always toward whichever coil is more strongly energized — and, honestly, why this project's
open-loop coil-switching strategy can only ever *attract*, never *repel* (README "Limitations &
honesty" restates this as a real, physically-fundamental limitation of pure gradient pulling with
paramagnetic beads, not an implementation gap).

### Stokes drag and the first-order equation of motion

At `Re << 1` ("The problem" above), a sphere of radius `a` moving at velocity `v` through a fluid of
viscosity `mu` experiences a drag force `F_drag = -6*pi*mu*a*v` (Stokes' law — derived by solving the
Navier-Stokes equations with the inertial term dropped, the formal statement of "creeping flow"). Newton's
second law, `m*dv/dt = F_magnetic + F_drag`, with `m` a bead's picogram-scale mass and the momentum
relaxation timescale microseconds (far below this project's 0.5 s step), collapses to **quasi-static
force balance**: `F_magnetic + F_drag ~ 0`, i.e.

```
v = F_magnetic / gamma,   gamma = 6*pi*mu_fluid*a_bead   (Stokes drag coefficient, N*s/m)
```

— a FIRST-ORDER ODE for position, no velocity state, no acceleration term. `swarm_step_kernel`/
`swarm_step_cpu` integrate this with explicit (forward) Euler: `x_{t+1} = x_t + (F(x_t)/gamma)*dt`.

## The algorithm

**Step 1 — discretize the coil geometry** (`main.cu`'s `generate_coil_segments`, run once, host-side,
`O(NUM_COILS * segs_per_coil)`): each of the 4 coils becomes `segs_per_coil` (180 in the committed
scenario) straight `CoilSegment`s approximating its circle — a regular polygon inscribed in the true
circle, `720` segments total (the catalog bullet's named figure).

**Step 2 — compute 4 basis field maps** (`biot_savart_basis_kernel`, called once per coil):
`O(grid_n^2 * n_segs)` — `256^2 * 720 ~= 4.7e7` segment evaluations PER COIL, `~1.9e8` total for all 4.
This is the expensive step and the only one that ever touches the segment list.

**Step 3 — combine, per schedule phase** (`combine_field_kernel`): `O(grid_n^2)` per call — a linear
combination of the 4 precomputed basis maps (the linearity argument above), reused for every subsequent
field configuration this demo needs (3 schedule phases, 4 GATE_ATTRACT probes, 1 illustrative artifact —
8 total calls, each `O(grid_n^2)`, vs. re-running step 2's `O(grid_n^2 * n_segs)` sum 8 times).

**Step 4 — precompute the force-generating gradient, per phase** (`gradient_b2_kernel`): `O(grid_n^2)` —
a 4-neighbor central-difference stencil of `|B|^2` over the combined map, producing `(dB2/dx, dB2/dy)`
at every cell ONCE per phase, so step 5 never has to re-derive a gradient at run time.

**Step 5 — advance the swarm** (`swarm_step_kernel`, called once per schedule phase, looping internally):
`O(n_robots * steps)` bilinear-interpolation-plus-Euler-update operations, `steps` explicit-Euler
sub-steps per launch. This is the cheapest-per-operation step in the whole pipeline (a handful of flops
per robot per step) and, by construction (steps 3–4 having already paid the field-computation cost),
also the ONLY step whose cost scales with the swarm size or the simulated duration.

**Designing the open-loop schedule (offline, from the linear model).** The catalog bullet's ratified
scope is OPEN-LOOP control: the coil currents as a function of TIME, fixed in advance, with no feedback
from measured robot positions at run time (closed-loop control — README "Limitations & honesty" and
"Where this sits in the real world" below — is the documented research step beyond this teaching
version). This project's schedule was designed by running steps 3–5 FORWARD, ONCE, for a single
representative point starting at the workspace origin, through a 3-phase candidate schedule (energize
North only, then East only, then South only, each held for `steps_per_phase * dt_s` seconds at the
scenario's `I0` ampere-turns) — `main.cu`'s "planning pass," using this program's OWN kernels so the plan
and the reported tracking tolerance are computed by the identical numerical path. The resulting 3
phase-end positions ARE the waypoints `GATE_WAYPOINTS` later checks the REAL, 1000-robot, dispersed
swarm's centroid against — the swarm's actual dynamics never read this plan back; it exists purely to
define what "success" means for the reported comparison (a genuine feedforward design, not a disguised
feedback loop).

**Why single-coil-only phases.** Section "The math" above derived that pure gradient-pulling with a
superparamagnetic bead can only ATTRACT, never repel — the honest consequence is that steering
"diagonally" by combining two coils does not simply add their pull directions (the combined `|B|^2` has
cross terms — see "The math"), so the simplest, most PHYSICALLY LEGIBLE open-loop strategy is to move
axis-by-axis: energize the ONE coil whose direction matches the next leg of the path, for a duration long
enough (from the offline plan) to cover the needed distance, then switch. This is also literally the
same physical situation `GATE_ATTRACT` verifies — the schedule's 3 phases and the attraction gate's 4
probes exercise the identical code path with the identical physical claim ("energizing coil `c` alone
pulls the swarm toward coil `c`"), just at different current-vector-to-outcome checkpoints.

## The GPU mapping

Four kernels, four DIFFERENT thread-to-data mappings, deliberately — the project's structure IS a small
tour of the GPU patterns that recur across this whole repository:

```
Kernel                    Threads =           Pattern            Memory behavior
------------------------  ------------------  -----------------  ------------------------------------
biot_savart_basis_kernel  1 per grid cell      map + per-thread   broadcast reads (segs[], ~20 KB,
                           (65536)              REDUCE over        L2-resident after warm-up);
                                               720 segments        coalesced output writes
combine_field_kernel      1 per grid cell      pure map            coalesced reads (coil-major layout)
                                                                   + coalesced writes
gradient_b2_kernel        1 per grid cell      STENCIL             4-neighbor reads (overlapping between
                                               (4-neighbor)         adjacent threads -> L2 reuse)
swarm_step_kernel         1 per ROBOT (1000)   agent farm,         SCATTERED bilinear reads (each robot
                                               register-resident    at its own position); no inter-
                                               T-step loop          thread communication, no atomics
```

**Kernel 1 (`biot_savart_basis_kernel`)** is the catalog bullet's named GPU hook: the classic
"embarrassingly parallel field map," one thread per evaluation point, independent of every other thread
by construction — the textbook case for the GPU's SIMT model. `kernels.cu`'s own header comments detail
occupancy/memory reasoning; the short version is that `segs[]` (720 structs, ~20 KB) is small enough to
live in cache across the kernel's whole run, making this kernel COMPUTE-bound (cross products + rsqrt),
not memory-bound, despite reading the segment array 65536 times over.

**Kernel 2 (`combine_field_kernel`)** is the cheapest kernel in the pipeline and the direct GPU expression
of the linearity argument in "The math": no segment loop, no neighbor reads, just 4 multiply-adds per
output cell. It exists because paying kernel 1's cost 8 times (once per configuration this demo needs)
instead of 4 times (once per coil, ever) would be strictly worse for no benefit — precomputing basis maps
and combining them cheaply is the general pattern for ANY linear operator applied to MANY input
combinations (the same idea underlies, e.g., precomputed basis functions in FEM, or impulse-response
convolution in signal processing).

**Kernel 3 (`gradient_b2_kernel`)** is this project's STENCIL kernel — the same access pattern
[`24.01`](../../24-actuators-motors/24.01-2d-magnetostatic-fea-solver-on-gpu-motor-torque/THEORY.md#the-gpu-mapping)'s
red-black SOR solver teaches (a 5-point stencil there; a 4-neighbor central difference here), deliberately
WITHOUT shared-memory tiling: at `256x256` cells, `(Bx,By)` together are 512 KB — too large for one SM's
shared memory to hold as a single tile, AND this kernel runs only 3 times total (once per schedule
phase), not thousands of times as 24.01's iterative relaxation does — so the "measure before tiling"
lesson applies directly: L2-cache reuse between neighboring threads' overlapping reads is already
sufficient at this call count, and tiling would add code complexity for no measurable benefit here
(README Exercise territory: 24.01 is the sibling project where the SAME stencil pattern, run thousands
of times, DOES reward tiling).

**Kernel 4 (`swarm_step_kernel`)** is this repository's "agent farm" pattern
([`08.01`](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/THEORY.md#the-gpu-mapping)'s
rollout kernel, [`22.01`](../../22-multi-robot-swarms/22.01-100k-agent-swarm-simulator/THEORY.md)'s swarm
agents): one thread owns one robot's ENTIRE trajectory for the whole phase, held in registers, looped
internally — a single kernel LAUNCH performs an entire schedule phase's worth of Euler steps
(`steps_per_phase=300` in the committed scenario), not 300 separate launches, amortizing launch overhead
to nothing. Unlike kernels 1–3, this kernel's memory access is SCATTERED (each robot bilinearly samples
`(dB2x,dB2y)` at its OWN, generally-different position) rather than a clean per-cell pattern — at only
1000 robots x 4 corner-reads each, this is nowhere near a bottleneck (the maps are 512 KB, comfortably
L2-resident), but it is worth naming as the reason a MUCH larger swarm would eventually want texture
memory's dedicated 2D-locality caching (README Exercise territory again). No atomics anywhere in this
kernel: every robot's trajectory is fully independent — this project does not model bead-bead magnetic
dipole interactions or hydrodynamic coupling between robots (an honest, named limitation, README
"Limitations & honesty" and "Where this sits in the real world" below).

## Numerical considerations

- **Precision:** FP32 throughout (device and host), the repo default. Field magnitudes in this
  scenario range from ~1e-5 T (single-coil basis maps, unit current) to ~1.5-2.5e-2 T (combined fields
  at 500-800 A-turns) — comfortably within FP32's dynamic range and precision for every operation this
  project performs.
- **The 1/r^3 singularity guard (`biot_savart_contribution`):** a small epsilon is added to `r^2` before
  cubing, defensively — in THIS project's geometry, the workspace never gets closer than roughly
  `coil_offset_m - workspace_half_m ~ 6 mm` to any coil segment, so the guard is never actually
  exercised by the committed scenario, but a general Biot-Savart routine should never assume its
  caller's geometry keeps it safe (the same defensive-programming argument as a softening length in
  N-body gravity kernels).
- **The bilinear-sample clamp guard (`bilinear_sample`):** a SEPARATE guard, protecting against an
  out-of-bounds MEMORY READ (not a numerical singularity) if a robot's position ever drifts past the
  mapped grid's edge. This project's tuned parameters keep every robot comfortably inside the mapped
  region (measured: the swarm's extent stays within roughly ±3.3 mm of an 8 mm-wide, ±4 mm-half-width
  workspace — see README "Expected output"), so this is also a defensive floor, not a frequently-hit
  path.
- **Finite-difference step size — a genuine, measured trade-off (GATE_DIVERGENCE):** the divergence
  sanity check differentiates `B` numerically via central differences. A first attempt at `h = 1e-6 m`
  (1 micron) measured a normalized divergence of `6.8e-3` — an order of magnitude ABOVE the intended
  `1e-3` tolerance, not because the physics was wrong but because FP32's ~7 decimal digits of precision
  cannot cleanly subtract two `~1e-2 T` field values that differ by only `~1e-6` (the true derivative
  scale times a 1-micron step) — classic catastrophic cancellation. Widening the step to `h = 1e-4 m`
  (still 200x smaller than the 20 mm coil radius, so TRUNCATION error stays negligible) measured
  `1.02e-4` — comfortably inside tolerance. This is the textbook finite-difference step-size trade-off
  (too small: rounding error dominates; too large: truncation error dominates) made concrete with real,
  measured numbers rather than asserted from a textbook plot.
- **Determinism:** the ONLY randomness in this whole demo is the swarm's INITIAL cluster (host
  `Xorshift32` + Box-Muller, fixed seed from the scenario's `SWARM` row) — the dynamics themselves are
  fully deterministic (no per-step noise; "The problem" above derives and quantifies why Brownian motion
  is honestly small enough at this bead size/timestep to omit by default). A different bead size or a
  weaker field would change that calculus — see "Where this sits in the real world."
- **Angle wrapping / quaternions:** not applicable to this project (`N/A because` this project has no
  orientation state at all — beads are modeled as point masses with an induced magnetic moment, not
  rigid bodies with attitude; CLAUDE.md §12's angle-wrapping discipline has nothing to wrap here).
- **State-vector layout:** every robot's state is exactly `(x, y)` in meters, workspace-frame,
  Structure-of-Arrays (`rx[]`, `ry[]`) for GPU coalescing — documented once in `kernels.cuh`'s file
  header, honored identically by the GPU kernel and the CPU oracle.

## How we verify correctness

**Eight independent stages**, because (as `08.01`'s THEORY.md argues for its own controller) a physics
simulation can be numerically self-consistent yet behaviorally wrong, or vice versa — no single check
catches everything:

1. **`VERIFY_FIELD`** — the GPU's coil-0 basis map vs. an INDEPENDENT CPU implementation
   (`biot_savart_basis_cpu`, sharing only the HOSTDEV numerical CORE, not the loop structure) at the
   same inputs. Measured worst case: `1.09e-11 T` against a `5e-9 T` tolerance (basis values are
   `~1e-5 T`, so this is roughly 5 orders of magnitude below signal — FP32 ULP-level agreement).
2. **`VERIFY_DYNAMICS`** — the GPU's phase-0 swarm result vs. a FULLY INDEPENDENT CPU path (its own
   basis map, combine, and gradient, not a copy of the GPU's intermediate results) after 300 chained
   Euler steps for all 1000 robots. Measured worst case: `1.54e-8 m` against a `1e-7 m` tolerance
   (positions are `~1e-3 m` scale, so `1e-4` relative headroom over ~300 chained sequential updates).
3. **`GATE_ONAXIS`** — the discretized field vs. the textbook closed form
   `B=mu0*I*R^2/(2*(R^2+z^2)^1.5)`, NEVER touching the 256x256 grid. Measured max relative error:
   `2.54e-4` (0.025%) against a 1% tolerance — this is the polygon-vs-circle discretization error at
   180 segments, not a GPU/CPU numerics question at all.
4. **`GATE_HELMHOLTZ`** — the East+West pair's flatness over the ACTUAL workspace extent (not an
   arbitrarily generous region). Measured variation: `1.76e-3` (0.18%) against a 2% tolerance.
5. **`GATE_DIVERGENCE`** — full 3D `div(B) ~ 0` at 5 interior points under a MIXED (asymmetric) current
   configuration, via a properly-tuned finite-difference stencil ("Numerical considerations" above).
   Measured max normalized `|div B|*R/|B|`: `1.02e-4` against a `1e-3` tolerance.
6. **`GATE_ATTRACT`** — for EACH of the 4 coils individually, a probe swarm must drift toward that coil.
   Measured (50 steps, `dt=0.5 s`): East `+342 um`, West `-335 um`, North `+360 um`, South `-317 um` along
   their respective expected axes (margin: `5 um`) — every coil's sign matches the "attracts toward
   itself" prediction from "The math," with three orders of magnitude of headroom over the margin.
7. **`GATE_WAYPOINTS`** — the REAL 1000-robot swarm's centroid, at the end of each phase, vs. the offline
   single-particle plan. Measured distances: `12.2 um`, `12.6 um`, `13.3 um` for the 3 phases, against a
   `300 um` tolerance — roughly 20-25x headroom, and the residual itself has a clean physical
   explanation (the plan is a single point; the real swarm is a ~0.3 mm-spread Gaussian cluster, so its
   centroid tracks the plan closely but not exactly, especially where the field's gradient is not
   perfectly uniform across the cluster's extent).
8. **`GATE_BOUNDS`** — every recorded snapshot (91 rows, every 10 steps) of every one of the 1000
   robots' positions is checked finite and within the mapped workspace, for the entire run.

Every measured number above is the ACTUAL output of a real run on the reference machine (RTX 2080
SUPER, sm_75, Release build) — see `demo/expected_output.txt`'s header comment for why the CHECKED
output lines carry only PASS/FAIL verdicts (not these numbers) while every one of these numbers is
printed honestly on an unchecked `[info]` line.

## Where this sits in the real world

- **OctoMag and its descendants** (ETH Zurich's Institute for Robotics and Intelligent Systems, and the
  broader magnetic-manipulation research community it seeded) are this project's real-world archetype:
  an 8-coil (not 4) electromagnetic system arranged around a workspace, driving CLOSED-LOOP control of
  magnetic microrobots using real-time camera or fluoroscopic position feedback — the single biggest gap
  between this teaching version and a real system. **Closed-loop control is the documented next step**:
  it would replace this project's offline-planned, fixed schedule with a per-tick optimization (e.g. a
  small QP or gradient step) solving "what current vector best moves the swarm's MEASURED centroid
  toward the CURRENT target," using the exact same linear field model this project already has —
  README Exercise territory names this explicitly.
- **Magnetic tweezers** (a mature, widely-used single-molecule biophysics tool) use a similar
  gradient-pulling principle at a much smaller scale (single beads, often near a permanent magnet or a
  simpler coil pair rather than a full steerable array) to apply calibrated forces/torques to individual
  molecules — the same `F = k*grad(|B|^2)` physics this project derives, applied to force SPECTROSCOPY
  rather than robotic transport.
- **Closed-loop control's real requirements** this teaching version does not implement: a vision or
  fluoroscopy feedback loop (PRACTICE.md §3), a real-time controller running at camera frame rate
  (typically tens of Hz, not this demo's offline-computed schedule), and — critically — a CONTROL LAW
  that handles the fact that pure gradient-pulling can only attract (never repel): real systems either
  use MORE coils (8, as in OctoMag) to synthesize local field MINIMA that trap/steer a target
  independent of nearby obstacles, or accept the attract-only constraint and plan paths that respect it.
- **Heterogeneous swarms** — this project's `k_force`/`gamma` are IDENTICAL for every robot (same bead
  size, same susceptibility). Real swarms of many small robots for tasks like distributed sensing or
  parallel micro-assembly increasingly want DIFFERENT robots to respond differently to the SAME field
  (by size, coating, or magnetic anisotropy) so a single global field can address them selectively — an
  active, open research problem this project's uniform-swarm model does not touch.
- **In-vivo and lab-on-chip integration challenges** (PRACTICE.md §4 gives the regulatory/business
  framing): a real deployment fights viscosity variation (blood is non-Newtonian; different tissues have
  wildly different effective viscosity), imaging depth limits (fluoroscopy dose, ultrasound resolution,
  optical scattering), and the sheer difficulty of generating strong enough gradients at a useful
  standoff distance through tissue — this project's clean, obstacle-free, directly-imaged 2D workspace
  sidesteps every one of these, honestly, as the reduced teaching scope this `[R&D]` catalog bullet
  ships as (README "Limitations & honesty").
- **Bead-bead interactions** (magnetic dipole-dipole forces between nearby robots, and hydrodynamic
  coupling through the fluid they share) are not modeled — at this project's swarm density and field
  strengths they are a secondary effect for TRANSPORT (the demo's task), but they are the DOMINANT effect
  for chain/aggregate SELF-ASSEMBLY behaviors that real magnetic-particle systems also exhibit and that
  some research platforms deliberately exploit — a natural extension this project's per-robot-independent
  kernel design (no atomics, no inter-thread communication) would need to break to add.
