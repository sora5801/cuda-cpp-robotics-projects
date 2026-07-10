# 24.01 — 2D magnetostatic FEA solver on GPU → motor torque-ripple/cogging parameter sweeps: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**Why a motor needs a field solver at all.** A permanent-magnet (PM) motor makes torque by arranging
magnets and current-carrying conductors so that the magnetic field they jointly produce pushes the
rotor around. Getting the torque *magnitude* right is high-school physics (`F = I L × B`); getting the
torque **smooth** is not — it is one of the harder problems in electromechanical design, and it cannot
be solved by hand. The culprit is **cogging torque**: even with *zero* winding current, a PM rotor
spinning past a slotted iron stator feels a periodic torque, because the magnetic circuit's *reluctance*
(its resistance to flux, the magnetic analogue of electrical resistance) changes as each magnet
sweeps past each slot opening and each tooth. The rotor "wants" to sit where the reluctance is lowest —
exactly the ratcheting, detent feel you get turning an unpowered stepper motor or a bicycle dynamo by
hand. Left unmanaged, cogging causes audible noise, vibration, and — in precision or low-speed
applications (robot joints included) — position-dependent torque error a controller cannot easily
compensate. **This project builds the smallest possible tool that can predict it: a solver for the
magnetic field around a slotted PM motor cross-section, differentiated at a circular contour into a
torque number, repeated across rotor angle and one design parameter (magnet pole-arc fraction) to
answer the design question every motor engineer asks: "how much of each pole should the magnet
cover, to make cogging as small as possible?"**

**The physics, from Maxwell's equations down.** A motor's magnetic field is governed by two of
Maxwell's four equations (the *magnetostatic* pair — no time-varying fields, appropriate because we
evaluate the field at a sequence of *fixed* rotor angles, not a moving one; THEORY §"Where this sits
in the real world" discusses what changes when the rotor is actually spinning):

```
∇ × H = J          (Ampere's law: circulating H is sourced by free current density J)
∇ · B = 0          (no magnetic monopoles — every field line closes on itself)
```

together with the material relation `B = μ H` (linear, isotropic materials — iron below saturation,
air, and a PM's *recoil* line — the simplification THEORY §"Numerical considerations" is honest
about) and, for permanent magnets specifically, `B = μ₀(H + M)`, where **M** is the magnet's intrinsic
*magnetization* (A/m) — the source of its field even with zero external current.

**The engineering frame.** This solver answers a **design-time, offline** question, not a real-time
control-loop question (SYSTEM_DESIGN item 1's rate bands do not apply the way they do to a
perception or control kernel — see README "System context" for the honest rate/latency framing). But
the physics it computes feeds directly into engineering decisions that DO have hard consequences at
robot-runtime: a badly-cogging actuator degrades a legged robot's foot-placement precision, a
manipulator's low-speed smoothness, and a drone motor's acoustic and vibration signature. The
engineering constraint that makes this a *design* problem rather than a closed-form one is
**geometry**: the moment a stator has discrete slots (which every wound motor does — you cannot wind
copper into a smooth iron ring), the field has no closed-form solution, and you are down to
numerically solving a PDE over the actual (or an idealized) cross-section.

## The math

**Reducing 3D magnetostatics to a 2D scalar problem.** A motor's active length (the "stack") is
mostly *extruded* along its shaft axis (z): the cross-section — magnets, slots, teeth — repeats
(ignoring end effects) all the way down the stack. Every free current and every magnet in this
project's cross-section is therefore z-directed (or, for permanent magnets, has a z-directed
*equivalent* current — derived below): `J = J_z(x,y) ẑ`. Because `∇ · B = 0` always, `B` can be
written as the curl of a vector potential `A`; for a purely z-directed source, `A` itself reduces to a
single **scalar** field `A_z(x,y)` (the x and y components of A are unnecessary — this is the entire
"2D" in "2D magnetostatic FEA"):

```
A = A_z(x,y) ẑ
B = ∇ × A = ( ∂A_z/∂y,  −∂A_z/∂x,  0 )        →  Bx = ∂A_z/∂y,   By = −∂A_z/∂x
```

Note `∇ · B ≡ 0` is now true **by construction**, for *any* smooth `A_z` — one Maxwell equation is
free. Only Ampere's law is left to enforce.

**The governing PDE.** Substitute `H = νB − νμ₀M` (from `B = μH + μ₀M`, with **reluctivity**
`ν = 1/μ` — the magnetic analogue of electrical resistivity) into `∇ × H = J`, using `B = ∇×A`:

```
∇ × (ν ∇×A) = J + ∇×(ν μ₀ M)
```

Taking the z-component (the only nonzero one, by the same argument as above) and expanding
`∇×(ν ∇×A)` for `A = A_z ẑ` gives, after the standard vector-identity simplification:

```
        −∇·(ν ∇A_z) = J_z + [∇×(ν μ₀ M)]_z
```

For every material in this project the recoil permeability of a PM is close enough to that of air
(`μ_r_magnet ≈ 1.05`) that `ν μ₀ ≈ 1` to within a few percent inside the magnet — the standard
simplification production PM-motor tools also make for a *linear* magnet model — reducing the source
term to the textbook **equivalent magnetizing current**:

```
        −∇·(ν ∇A_z) = J_z + J_m,          J_m = [∇×M]_z = ∂M_y/∂x − ∂M_x/∂y
```

**This is THE governing equation this project solves** — a linear, second-order, elliptic PDE with a
**variable coefficient** `ν(x,y)` that jumps by a factor of ~2000 between iron and air. Everything
else in this file explains how to discretize it, solve it, and check the answer.

**Modeling the permanent magnets (M, and why `J_m` is the right source).** A magnet is modeled as
uniformly magnetized in its rest direction: for a *radially* magnetized surface-mount pole (this
project's rotor), `M = ±M₀ r̂` inside the magnet and `M = 0` outside, alternating sign pole to pole
(`M₀ = Br/μ₀`, from the magnet's remanence `Br` — a manufacturer-quoted material property, ~1.2 T for
a modern NdFeB grade). Because `M` is **piecewise constant**, its curl is a distribution concentrated
at the magnet's boundaries — physically, the well-known "equivalent bound surface current"
`K_b = M × n̂` every PM-motor textbook derives for each pole edge. For a radially magnetized pole this
surface current sheet sits at the pole's two *angular* edges (where the field jumps from `M₀ r̂` to
zero) and is z-directed with strength `M₀` — exactly the reluctance-modulating "wall of current" that
makes each pole act like a distributed winding. **Instead of hand-deriving that surface geometry for
every pole edge, this project computes `J_m` by taking a CENTERED FINITE DIFFERENCE of the
rasterized (piecewise-constant) M field on the very same grid the solver uses** (`equivalent_current`
in `src/main.cu`). A finite difference of a step function is a discrete approximation to a Dirac delta
— smeared over about one grid cell — reproducing the bound surface current automatically, at exactly
the solver's own resolution, with far less code and far less room for a sign error than deriving
`M × n̂` region by region.

**Boundary condition — why Dirichlet `A_z = 0` at the domain edge is not a "free space" cheat.** A
level curve of `A_z` (a curve along which `A_z` is constant) is, by the definition `B = ∇×A`,
*everywhere tangent to* `B` — it **is** a field line. Fixing `A_z = 0` along the *entire* outer
boundary of the solve domain therefore forces that boundary to be a single closed field line: **no
flux crosses it**. This is the standard "flux barrier" truncation every commercial motor-FEA tool
(FEMM, Ansys Maxwell, JMAG) offers, and it is a legitimate modeling choice — not an infinite-domain
approximation — *provided* the boundary sits somewhere flux genuinely does not want to cross. Placing
it just outside the stator's solid back iron (which is deliberately made thick enough to carry the
whole pole's flux without saturating) satisfies that condition to good approximation; this project's
committed scenario does exactly that (`data/README.md` documents the radii).

## The algorithm

**Step 1 — rasterize the cross-section.** For a given magnet pole-arc fraction and rotor angle,
classify every grid node into one of five concentric regions (rotor core / magnet ring / air gap /
stator teeth+slots / stator back iron) by radius and, within the magnet and tooth rings, by angle —
producing the reluctivity field `ν(x,y)` and the magnetization field `M(x,y)` (`rasterize_motor` in
`src/main.cu`). Complexity: `O(nx·ny·max(P,S))` — a few hundred thousand cheap comparisons, done once
per rotor-angle *variant* on the host (this is SETUP, not the taught hot loop — the 31.01/08.01
precedent this project follows).

**Step 2 — build the source term.** Central-difference `M` into `J_m` (`equivalent_current`); add any
free (winding) current `J_free`, which is **always zero** for the cogging sweep by definition — cogging
is measured with the windings open.

**Step 3 — discretize the PDE.** On the square grid (`dx = dy = h`), the variable-coefficient
Laplacian `−∇·(ν∇A)` becomes, at every INTERIOR node `(i,j)`, a 5-point stencil (full derivation in
"The GPU mapping" below):

```
A(i,j) = [ νE·A(i+1,j) + νW·A(i-1,j) + νN·A(i,j+1) + νS·A(i,j-1) + h²·J(i,j) ] / (νE+νW+νN+νS)
```

with **face reluctivities** `νE = ½(ν(i,j)+ν(i+1,j))`, etc. — the harmonic-mean averaging derived
below. Boundary nodes are pinned at `A = 0` (never updated — the Dirichlet condition above).

**Step 4 — solve iteratively.** Repeatedly sweep every interior node with the update above (this is
`RED-BLACK SUCCESSIVE OVER-RELAXATION`, detailed in the next section) until the field has converged —
here, for a FIXED, measured-sufficient number of sweeps rather than a residual-triggered stop
(explained in "The GPU mapping"). Serial cost per sweep: `O(nx·ny)`; a direct solve of the same
`~65,000×65,000` sparse linear system (one unknown per node) via, say, a sparse Cholesky factorization
would cost far more in both time and memory for a system this large and this variable-coefficient —
iterative relaxation is the standard choice for regular-grid elliptic PDEs of this size, precisely
because each sweep is `O(N)` and embarrassingly parallel over nodes.

**Step 5 — extract B, then torque.** `B = curl(A)` by central differences (`curl_A`); torque follows
from the Maxwell stress tensor integrated around a circular contour in the air gap (derived below,
`maxwell_stress_torque`). Repeat steps 1–5 for every `(arc fraction, rotor angle)` pair in the sweep —
**batched**, so that all 24 rotor-angle solves for one arc fraction run as ONE kernel-launch sequence
(the next section's central lesson).

**Why cogging torque exists, precisely.** Cogging torque is (minus) the derivative, with respect to
rotor angle `θ`, of the magnetic **co-energy** `W'(θ)` stored in the field at zero current:
`T_cog(θ) = −dW'/dθ`. Co-energy depends on `θ` only because the *reluctance seen by the magnets*
changes as they sweep past alternating iron teeth (low reluctance) and slot openings (high
reluctance) — if the stator bore were a smooth, unslotted iron cylinder, `W'` would not depend on `θ`
at all (full rotational symmetry) and cogging would be exactly zero. Slots are unavoidable (windings
must go somewhere), so cogging is unavoidable in slotted machines — the entire point of this
project's *sweep* is that the **size** of the effect is a controllable design choice (this project
controls it via magnet pole-arc fraction; production designs also skew the slots, use fractional-slot
winding layouts, or add dummy slots — README "Prior art" and PRACTICE §4 name the tools that do this
systematically).

**The Maxwell stress tensor — deriving the torque formula.** The Maxwell stress tensor gives the
force per unit area a magnetic field exerts across any surface, in terms of the field alone (no need
to know what is physically touching that surface — a purely field-based bookkeeping device, exactly
like using a control volume in fluid dynamics). In 2D, at a point on a circle of radius `r` with
outward normal `r̂`, the SHEAR stress (force per area, tangential direction, the component that does
rotational work) is:

```
σ_rθ = Br·Bθ / μ₀
```

Multiplying by the lever arm `r`, integrating around the full circle (`dl = r dθ`), and — because
this is a 2D problem — reporting the result PER UNIT AXIAL LENGTH (see "Numerical considerations" for
why this unit is the honest one), gives:

```
T'(θ) = ∮ r · σ_rθ dl = ∮ r · (Br·Bθ/μ₀) · r dθ  =  (r² / μ₀) ∮ Br(θ)·Bθ(θ) dθ           [N·m / m]
```

evaluated at any contour radius `r` sitting entirely in the air gap. By Ampere's law and the
divergence theorem, this integral is provably **independent of the exact contour radius**, as long as
no current or magnetization crosses the annulus between the contour and the true source (a standard,
useful property — "Where this sits in the real world" and the README's contour-sensitivity discussion
report what this project MEASURES for that independence at this grid's resolution, honestly, rather
than just asserting the theorem).

## The GPU mapping

**The stencil is a MAP over an elliptic relaxation, extending 07.09/31.01's pattern.** Both prior
grid-PDE flagships in this repo (07.09's jump-flooding SDF, 31.01's Hamilton-Jacobi reachability) are
stencils: every cell updates from a small neighborhood, every sweep, independently of every OTHER
cell's update **within the same sweep**. This project's stencil is new in two ways:

1. **A genuinely variable coefficient.** `ν(x,y)` jumps by ~2000× between iron and air, computed
   fresh from the neighboring NODE values every sweep (`νE = ½(ν_c + ν(i+1,j))`, etc.) rather than
   read from a separate precomputed face-coefficient array. This is a deliberate RECOMPUTE-vs-STORE
   trade: `nu` is one float per node (not per face), so recomputing four face averages costs four
   extra multiply-adds but saves an entire array's worth of memory bandwidth every sweep — worthwhile
   here because this kernel is bandwidth-, not compute-, bound (see the roofline discussion below).

2. **RED-BLACK Gauss-Seidel instead of ping-pong Jacobi.** Gauss-Seidel converges roughly TWICE as
   fast as Jacobi for the same stencil (it uses each sweep's freshest neighbor values immediately,
   rather than the previous sweep's), but is normally *sequential*: cell `i` needs cell `i-1`'s
   ALREADY-UPDATED value within the same pass. The **checkerboard trick** breaks that dependency:
   color the grid by `(i+j)` parity; every RED cell's four face neighbors are all BLACK and vice
   versa, so updating every red cell in parallel — reading only (unmodified-this-pass) black
   neighbors — computes EXACTLY the same numbers a sequential Gauss-Seidel sweep over the red cells
   would, and vice versa for black. Two color-restricted launches per "sweep pair," **in place** (no
   ping-pong buffer needed — the color-guard IS the synchronization, enforced by CUDA stream order
   between the two launches, exactly as 31.01's ping-pong swap is enforced by stream order between
   iterations).

3. **SOR (Successive Over-Relaxation) rides on top for free.** The same stencil, blended:
   `A ← A + ω·(A_gs − A)`, `ω ∈ (0,2)`. `ω=1` is plain Gauss-Seidel; this project measured (below)
   that `ω=1.97` — close to the theoretical stability limit `ω→2` for a grid this size — cuts the
   sweep count needed for good convergence by roughly **an order of magnitude**.

**Batching — the project's second GPU lesson.** `blockIdx.z` selects the variant (rotor angle); every
variant shares the SAME grid geometry (`FeaGrid g`, passed by value — small, uniform, read by every
thread, ideal for the constant-cache-backed kernel-argument path) but has its own `nu`/`Jsrc`/`A`
slice, laid out **batch-major**: `F[b*ny*nx + j*nx + i]` (kernels.cuh's layout contract). The batch
axis rides OUTSIDE the per-variant `(i,j)` layout specifically so that within ONE variant, a warp's
32 threads (consecutive `i`, same `j`, same `b`) still read 32 CONSECUTIVE floats — batching costs
nothing in coalescing. **This is the project's central lesson: "solve B independent small problems"
is exactly as GPU-friendly as "solve one big problem," as long as the layout respects it** — one
kernel-launch sequence solves all 24 rotor angles for one arc fraction, instead of 24 separate
sequences paying their own launch overhead.

**Why a FIXED sweep count, not a residual-triggered stop.** A per-cell or per-variant "converged,
stop early" test would make different threads (or different `blockIdx.z` batch elements) finish at
different times — exactly the kind of DATA-DEPENDENT divergent control flow that breaks a uniform,
lock-step launch's throughput (every launch must still wait for the slowest lane). Fixing the sweep
count at a value MEASURED to comfortably over-converge every fixture this project tests (see
"Numerical considerations") keeps every thread in every launch doing the same, predictable amount of
work — the standard GPU-relaxation trade of "a few wasted sweeps" for "zero divergence and a simple,
predictable launch."

**Memory hierarchy.** Per thread, per sweep: 5 reads from `nu` (center + 4 face neighbors — the `i±1`
neighbors are consecutive addresses within a warp's row, coalesced; the `j±1` neighbors are whole-row
strides the L2 cache serves across the block's rows), 4 reads from `A`, 1 read from `Jsrc`, 1 write to
`A` — roughly 40 bytes moved per cell per pass for ~15 flops of arithmetic. **No shared memory**: each
node's value is reused by at most 4 neighbor threads, and the L2 cache covers that reuse at this grid
size (the same documented choice 07.09/31.01 make — a shared-memory tile is README Exercise territory,
worth measuring rather than assuming). **No atomics**; the only divergence is the checkerboard color
test and the tail/border guards, both cheap (a predicated early return, not a serialized branch).

## Numerical considerations

**FP32 throughout — and a measured, honest limit of "converged."** The solver runs entirely in FP32
(the repo's default, and iterative relaxation is well-conditioned enough at this grid size that
double precision buys nothing visible). Residual-norm tracking during development (a standalone
NumPy/float64 prototype, not shipped code) showed the solved field's PHYSICAL quantities (the sampled
B field, the torque integral) stop changing to five significant figures once the residual has dropped
by roughly **9–11 orders of magnitude** from its initial value — after that point, further sweeps
reduce the ITERATION error well below the DISCRETIZATION error (the ~0.2% gap between this solver's
answer and the exact Ampere's-law closed form, which no amount of extra sweeping can close — only a
finer grid can). `N_SWEEPS = 1500` sweep-pairs at `ω = 1.97` (the committed scenario) was chosen from
that measurement with comfortable headroom, not tuned to "just barely pass."

**Why harmonic averaging, not arithmetic, at a material interface — the classic FVM lesson.**
Consider 1D flux flowing perpendicular through two adjacent cells of reluctivity `ν₁, ν₂`, each of
length `h/2` (the standard "flux tube" argument). Reluctance adds IN SERIES, exactly like electrical
resistance: `R_total = R₁ + R₂ = ν₁·(h/2) + ν₂·(h/2)` (per unit area). The SINGLE EFFECTIVE
reluctivity `ν_eff` that reproduces the same total reluctance over the full length `h` satisfies
`ν_eff · h = ν₁·(h/2) + ν₂·(h/2)`, i.e. **`ν_eff = (ν₁+ν₂)/2` — the ARITHMETIC MEAN OF RELUCTIVITY**.
Because `ν = 1/μ`, this is algebraically IDENTICAL to the HARMONIC MEAN of the two permeabilities:
`μ_eff = 2μ₁μ₂/(μ₁+μ₂)`. (Check: `1/μ_eff = (μ₁+μ₂)/(2μ₁μ₂) = ½(1/μ₁+1/μ₂) = ν_eff`. ✓) Using the
ARITHMETIC mean of `μ` instead (a natural-looking but WRONG shortcut) over-weights the high-
permeability material and can violate flux continuity at a sharp interface by a large margin — this
project's harmonic-mean face coefficient (`νE = ½(ν_c + ν_neighbor)`, arithmetic in `ν` = harmonic in
`μ`) is the reason `ANALYTIC_INTERFACE` passes to a fraction of a percent (measured below) instead of
tens of percent.

**Angle wrapping — the project's one defined wrap point.** Every angular test in `rasterize_motor`
(which pole window, which slot window a node falls in) wraps the angle difference into `(−π, π]` via
`wrap_pi` — the ONE place in this codebase angles are wrapped (CLAUDE.md §12 discipline). The solver
itself never sees an angle; it only sees the `nu`/`Jsrc` fields the wrapped classification produces.

**Debug-vs-Release floating-point reproducibility — a real, measured finding, not a footnote.**
Because `ω = 1.97` sits close to the SOR stability boundary (`ω → 2`), the iteration is more sensitive
than a well-damped one to exactly WHEN rounding happens — and Debug (`-G`, unoptimized SASS) vs.
Release (`-lineinfo`, optimized SASS) genuinely execute a DIFFERENT sequence of floating-point
operations (the compiler reorders adds, contracts multiply-adds differently, etc. — bit-for-bit
reproducibility across optimization levels was never a floating-point guarantee, only a same-build
one). Measured on the reference machine: the `ANALYTIC_INTERFACE` gate's relative flux jump is
`~0.35%` in Release and `~2.6%` in Debug — the SAME converged-to-machine-precision solution, a
DIFFERENT last few bits. `src/main.cu`'s gate tolerance (5%) was set from this MEASUREMENT, not
guessed, with headroom on both sides: comfortably above what either build configuration produces,
comfortably below the tens-of-percent error a real indexing or averaging bug would cause. The
GPU-vs-CPU VERIFY gate is, interestingly, the OPPOSITE story: in Debug (both paths unoptimized, no
FMA contraction on either side) the two fields come out **bit-identical** (`worst |dA| = 0.000e+00`);
in Release, compiler FMA-contraction differences between `nvcc`'s device code and `cl.exe`'s host
code reappear as the measured `2.948e-07` Wb/m gap — see "How we verify correctness" for the
tolerance this sets.

**2D torque units — reported honestly, not silently scaled.** Because this is a 2D (per-unit-length)
field solve, the Maxwell-stress torque integral naturally comes out in **N·m per meter of axial stack
length**, not N·m. This project reports EXACTLY that unit everywhere (`kernels.cuh`, every torque
printout, the CSV artifact header) rather than silently multiplying by an assumed stack length — a
real motor's torque is this number times its actual stack length in meters; README §"Expected output"
shows an illustrative (clearly labeled) scale-up for a plausible stack length.

**Contour-radius sensitivity — measured, not asserted.** The Maxwell stress integral is theoretically
independent of the exact contour radius within the air gap (see "The math"). At this grid's
resolution (the air gap is only ~5 cells wide), main.cu evaluates the contour at the gap's geometric
midpoint; README documents the measured spread when the contour is moved to the gap's inner and outer
thirds instead, so a learner can see the (small, honestly reported) sensitivity a coarse teaching grid
introduces, rather than trusting the theorem blindly.

## How we verify correctness

Four independent checks, layered so a bug has nowhere to hide:

1. **GPU-vs-CPU twin (`VERIFY`)** — one representative motor variant (a non-trivial rotor angle,
   deliberately not aligned with any pole/slot symmetry axis), solved by the exact same red-black SOR
   update expression on the GPU (`kernels.cu`) and sequentially on the CPU (`reference_cpu.cpp`),
   same sweep count, same pass order. Measured worst `|A_gpu − A_cpu|`: `2.948e-07` Wb/m in Release
   (bit-identical, `0.0`, in Debug — see "Numerical considerations"), against a tolerance of `2e-5`
   Wb/m — comfortable headroom over measured FP32 rounding, while an indexing/layout/averaging bug
   would shift the field at the scale of its own magnitude (`~1e-3`–`1e-2` Wb/m for this problem), not
   at the `1e-7` level rounding alone produces. Catches indexing, layout, and pass-order bugs on the
   solver ITSELF — the algorithm this project teaches.

2. **Analytic gate A — Ampere's law (`ANALYTIC_AMPERE`)** — an INDEPENDENT fixture (a uniform-current
   annulus in air, not the motor) solved on the SAME solver, checked against a closed-form textbook
   answer: zero field in the bore (no enclosed current), the correct `B(r)` growth inside the annulus
   and 1/r decay outside. This checks the solver against MATHEMATICS, not against another copy of
   itself — the strongest kind of check this repository can make (31.01's `min_time_to_origin` gate is
   this project's direct precedent). Measured relative error: `~0.19%` at three sample radii (well
   inside the 5% gate); the bore's residual field is `2.5e-8` T against a `2%`-of-scale threshold.

3. **Analytic gate B — flux continuity (`ANALYTIC_INTERFACE`)** — a second independent fixture (a
   straight air/iron interface driven by a current strip), checking that the field's NORMAL component
   is continuous across the interface — the textbook boundary condition, and the most DIRECT possible
   check on the harmonic-mean face-averaging this solver's stencil performs. Measured: `~0.35%`
   (Release) / `~2.6%` (Debug) against a 5% gate (see "Numerical considerations" for why the tolerance
   sits where it does).

4. **Physics sanity (`PHYSICS`)** — checks the SOLVER'S OWN OUTPUT against physical law rather than
   against a fixture: every cogging waveform must integrate to ~zero net torque over the sampled
   period (measured `|mean|/peak` ratio: `0.0000`–`0.0009` across all five arc fractions — cogging
   does no net work, by the energy-conservation argument in "The math"), and one magnet pole pitch
   must be a true structural period of the geometry, checked with an INDEPENDENT solve at
   `θ = pole_pitch` compared against the `θ = 0` sample (measured `|diff|/peak = 0.0000` — the two
   solves land on the same answer to the precision the solver converges to). Together these two checks
   would catch a symmetry-breaking bug (a rasterizer angle-wrap error, a mis-indexed batch slot) that
   neither the twin comparison nor the analytic gates — which never look at the actual motor geometry
   — could see.

The full four-stage measured result on the reference machine (RTX 2080 SUPER, sm_75, Release build):
`VERIFY: PASS`, `ANALYTIC_AMPERE: PASS`, `ANALYTIC_INTERFACE: PASS`, `PHYSICS: PASS`, and the sweep
itself finds a genuine, non-monotonic minimum in peak cogging torque at magnet arc fraction **0.70**
(peak `2.0444` N·m/m) against `0.60` (`5.4841`), `0.80` (`3.5724`), `0.90` (`6.6671`), and `1.00`
(`7.2236`) N·m/m — the qualitative result the catalog bullet's design question exists to surface.

## Where this sits in the real world

**"FEA" vs. what this project actually implements.** Production tools (FEMM, Ansys Maxwell, JMAG,
Motor-CAD) discretize the SAME governing PDE this project derives above using an **unstructured
triangular (or quadratic) mesh**, refined wherever the geometry demands it — dense in the air gap and
at slot corners, coarse in the solid back iron. That buys two things a regular grid cannot: the mesh
BOUNDARY conforms exactly to curved/angled geometry (no "staircase" approximation of a circular pole
arc), and resolution can be spent where the field varies fastest (this project's fixed 256×256 grid
gives the air gap only ~5 cells, visibly limiting precision right where torque is computed — the
`ANALYTIC_AMPERE`/`ANALYTIC_INTERFACE` gates' ~0.2%–0.35% residual errors are largely THIS
discretization-resolution effect, not solver-algorithm error). This project's regular-grid finite-
difference/finite-volume scheme is the RATIFIED TEACHING DISCRETIZATION — it solves the identical PDE
a linear-triangular-element FEA solver assembles, with a far simpler implementation (no mesh generator,
no element-shape-function bookkeeping), at the honest cost of geometric fidelity and localized
resolution.

**Nonlinear iron (saturation) — the biggest physics simplification here.** Real electrical steel does
NOT have a constant `μ_r`; its B-H curve saturates around 1.5–1.8 T, and above that "knee" the
effective permeability drops sharply, changing both the field shape and the torque. Production tools
solve this with an OUTER Newton-Raphson (or fixed-point/successive-substitution) loop around the exact
linear solve this project performs: guess `μ_r(B)` per element from the last iteration's `B`, re-solve
for `A`, recompute `B`, repeat until `μ_r` stops changing. This project's LINEAR `μ_r_iron = 2000`
model (chosen well below typical saturation flux densities) is a deliberate, documented teaching
simplification — the "reduced-scope teaching version" this catalog bullet's difficulty allows; the
outer nonlinear loop is straightforward to add on top of the exact linear solver here (README
Exercises).

**What is entirely absent here.** (1) **Eddy currents / AC loss** — this is a magnetOSTATIC (DC, zero
frequency) solver; a spinning rotor's changing flux induces eddy currents in solid conductors and
laminated iron that this project does not model (that is the *magnetodynamic*, `∇×E = −∂B/∂t`-coupled
problem — a natural "climb the ladder" extension, akin to how 08.01's MPPI ladder documents a bigger
plant rather than implementing it). (2) **3D end effects** — a real stack has finite length, and flux
fringes at both ends; this project's per-unit-length answer is exact only for an infinitely long
stack, an approximation that degrades for short, fat motors. (3) **Thermal and mechanical coupling** —
magnet strength drops with temperature (documented, not modeled), and mechanical eccentricity/
deflection under load shifts the air gap (also not modeled). Every production tool named above extends
in exactly these three directions; this project's linear, 2D, DC core is the shared foundation all of
them build on, and is exactly what the catalog bullet's difficulty (★ beginner) asks this project to
teach cleanly rather than exhaustively.
