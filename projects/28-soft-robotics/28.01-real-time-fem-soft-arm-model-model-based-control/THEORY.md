# 28.01 — Real-time FEM soft-arm model + model-based control (GPU SOFA-style): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Every symbol gets
> units on first use; frames are right-handed, 2-D in the arm's body plane (x along the arm from
> the fixed base, y up through the height, z out of plane).

## The problem — physics & engineering first

**Why soft robots are hard.** A rigid robot is a blessing of abstraction: a handful of numbers per
joint (angle, velocity, torque) fully describes an arm, because "rigid" means the material's
internal state is irrelevant. A soft robot revokes that blessing. A silicone arm is a *continuum* —
its configuration is a displacement **field** with, in principle, infinitely many degrees of
freedom; there is no joint to put an encoder on, no joint angle for a PID loop to servo, and no
Denavit–Hartenberg table to look up. Three consequences drive everything in this project:

1. **The model IS the state.** To know "where the tip is," you must model the whole body. Soft
   robotics therefore leans on continuum mechanics discretized by finite elements — and the model
   must run in real time if a controller is to use it, which is why this project's headline number
   is a measured real-time factor.
2. **Geometry goes nonlinear before material does.** Elastomers stretch by percent-scale strains
   in normal service, but the arm *rotates* by tens of degrees. Small-strain, large-rotation is the
   canonical soft-arm regime — and exactly the regime where the cheapest FEM (linear) breaks and
   the corotational method earns its keep (§The math).
3. **Actuation is indirect.** You cannot torque a continuum; you squeeze it (pneumatic chambers),
   pull it (tendons), or heat it (SMA). Force enters distributed over the body, and the map from
   actuation to task-space motion is itself a modeling problem — which is why this project's
   controller *measures* that map instead of assuming it.

**The physical system.** A cantilevered rectangular arm: length L = 0.24 m, height H = 0.024 m,
out-of-plane thickness t = 0.02 m, of a synthetic elastomer-class material — Young's modulus
E = 1 MPa (real silicones run ~0.05–5 MPa; 1 MPa is squarely "soft"), Poisson's ratio ν = 0.40,
density ρ = 1100 kg/m³. The base (x = 0) is bonded to a rigid mount; everything else moves. Two
tendon-like actuation channels run along the top and bottom faces: pulling the top fiber harder
than the bottom compresses the top more than the bottom, and the *strain differential across the
cross-section* bends the arm — the same mechanism that curls a bimetallic strip, a PneuNet
bending actuator (28.03's subject), or your own finger's tendon pair.

**The engineering constraints a real version faces** (PRACTICE.md grounds each): elastomer
properties drift with temperature and age (±20% modulus is normal); silicone creeps and exhibits
hysteresis, so today's Jacobian is not next month's; tendons introduce friction and slack;
E ~ 1 MPa means gravity visibly sags anything this size (this teaching model omits gravity so the
analytic gates stay clean — an explicitly-labeled simplification); and the co-contraction that
keeps both tendons taut *axially compresses the arm*, which buckles like any slender column if
overdone (§The math derives the bound; it sized this project's tension budget).

## The math

**Elasticity in one paragraph.** Strain measures relative deformation (dimensionless); stress
(Pa) is force per area; an elastic material maps one to the other. For small strains the map is
linear (Hooke's law), characterized by E and ν. In 2-D **plane stress** (thin plate, free faces:
σ_zz = 0) the 3-entry strain vector ε = (ε_xx, ε_yy, γ_xy) maps to stress σ = D·ε through

```
D = E/(1-ν²) · [ 1   ν   0        ]
               [ ν   1   0        ]
               [ 0   0   (1-ν)/2  ]
```

**FEM in one paragraph.** Chop the domain into elements (here Q4: 4-node bilinear quadrilaterals,
120×12 of them, each h×h with h = 2 mm). Within an element, displacement interpolates the 4 corner
values through shape functions N_a(ξ,η). Differentiating the shape functions gives the 3×8
strain-displacement matrix B (strain from corner displacements); the element's 8×8 stiffness is
K_e = t·∫ Bᵀ D B dA, evaluated exactly by 2×2 Gauss quadrature
([`src/reference_cpu.cpp`](src/reference_cpu.cpp)'s `compute_KE_hat` performs the integral
numerically at startup — no magic matrix is hardcoded anywhere). For a *square* element the h²
from the area integral cancels the (1/h)² inside BᵀDB exactly, so K_e = E·t·K̂ where K̂ is a
dimensionless unit matrix — the derivation is spelled out in [`src/kernels.cuh`](src/kernels.cuh)
and is why the code carries one scalar `Et` instead of per-element matrices.

**Why linear FEM fails at large rotation — the spurious-volume artifact.** Linear FEM measures
strain as ε = ½(∇u + ∇uᵀ) — the *linearized* strain, valid only for small displacement gradients.
Rigid-rotate an element by θ with zero deformation: u = (R(θ) − I)X gives
∇u = R − I = [[cosθ−1, −sinθ], [sinθ, cosθ−1]], whose symmetric part is (cosθ−1)·I ≠ 0. The
element "sees" an isotropic strain of cosθ−1 ≈ −θ²/2 — a pure rotation reads as *compression*,
growing quadratically with angle. A linear-FEM arm that bends visibly therefore also
swells/shrinks spuriously and fights its own rotation: at 30° of tip rotation the phantom strain
is ~13% — larger than any real strain in the arm. README Exercise 2 lets you watch this happen
(set θ_e = 0 in the force kernel and run the gates).

**The corotational fix, derived.** Separate rotation from deformation *per element*: compute the
deformation gradient F = ∂x/∂X (the 2×2 matrix mapping reference to current directions), factor
F = R·S into rotation R and symmetric stretch S (polar decomposition), measure strain in the
*unrotated* frame, rotate the resulting force back. In 2-D the rotation has a closed form worth
deriving once: the best rotation maximizes alignment, trace(RᵀF); with R(θ) = [[c, −s], [s, c]],

```
trace(RᵀF) = c·(F11 + F22) + s·(F21 − F12)
```

which is maximized (differentiate w.r.t. θ, set to zero — the second-derivative check picks the
maximum branch) at

```
theta = atan2(F21 − F12,  F11 + F22)
```

— one `atan2`, no eigendecomposition (the scope's "polar-decomposition-lite"; it IS the exact 2-D
polar decomposition whenever det F > 0, i.e. short of element inversion). The element force is
then the **warped stiffness** form (Müller et al. 2004):

```
f_e = Et · R · K̂ · (Rᵀ x_e − X_e)
```

with x_e the current corner positions and X_e the reference ones. At rest F = I → θ = 0 → f = 0
exactly; under pure rotation Rᵀx_e − X_e = 0 → f = 0 exactly — the artifact is gone by
construction. (Fine print: this force drops the ∂R/∂x variation and is *not* exactly a potential
gradient — a fact that comes back with teeth in §Numerical considerations.)

**The deformation gradient from shape functions.** F = Σ_a x_a ⊗ ∇N_a evaluated at the element
center, where the physical-space gradients for a square element are constants:
∇N_a = (ξ_a/(2h), η_a/(2h)) with corner signs ξ_a, η_a ∈ {−1,+1} (`grad_n_physical` in
kernels.cuh). Four fused multiply-adds per F entry; F = I exactly at rest.

**Dynamics.** Newton per node: M·ẍ = f_ext − f_int(x, ẋ) with M the *lumped* (diagonal) mass —
each element parcels its mass ρ·t·h² equally to its 4 corners (row-sum lumping; what makes
explicit integration solve-free). Damping is Rayleigh, C = α·M + β·K, giving per-mode damping
ratio

```
zeta(omega) = alpha/(2*omega) + beta*omega/2
```

— α damps LOW modes (ζ ∝ 1/ω), β damps HIGH modes (ζ ∝ ω). This project uses α = 3.8 s⁻¹ (sized
for ζ₁ = 0.15 at the first bending mode: α = 2ζ₁ω₁) and β = 2×10⁻⁵ s (sized by the *stability
tax*, below). The β force folds into the same element matvec as elasticity:
f = Et·R·K̂·(u_local + β·v_local) — one 8×8 multiply serves both.

**Explicit integration and the CFL bound, derived.** Semi-implicit (symplectic) Euler:

```
v <- v + dt * f(x, v)/m        (velocity FIRST, from the current state's force)
x <- x + dt * v                 (position second, from the NEW velocity)
```

Why the order matters: for an oscillatory mode this update is area-preserving in phase space
(determinant exactly 1), so the scheme conserves a "shadow" energy near the true one instead of
spiraling outward like naive explicit Euler — the property gate (iii) measures. The price of any
explicit scheme is conditional stability. Sound crosses one element in h/c seconds, where

```
c = sqrt(E/rho) = sqrt(1e6 / 1100) = 30.15 m/s
```

(dimensional check: Pa·m³/kg = (N·m)/(m²·kg) = m²/s²), and an explicit step propagates
information one element per step — dt must not outrun the physics: dt ≤ h/c = 66.3 µs. Formally,
undamped symplectic Euler is stable for ω·dt ≤ 2 and the highest mesh frequency is ω_max ≈ 2c/h;
the two statements coincide. We run dt = 30 µs — a documented 0.45 safety ratio (the catalog
bullet's illustrative "~2e-4 s" would sit 3× *above* this mesh's bound; this project's E/h choice
tightens the bound, and we honor the bound, not the round number). Note what the CFL couples:
**stiffer or finer → smaller dt**. This is *the* reason soft robots (E ~ MPa) get real-time
explicit FEM while steel (E ~ 200 GPa, c ≈ 5,000 m/s) does not — softness buys a ~170× larger
stable timestep at equal mesh and density, a genuinely deep fact about why this subfield exists.

**The stability tax on β.** The damped mode update carries the factor 1 − dt·(α + β·ω²); at ω_max
the β·ω² term explodes for any β sized to damp the *first* mode (β = 0.024 s would give
dt·β·ω_max² ≈ 640 — instant NaN, and yes, this project's first draft measured exactly that). Hence
the split: α handles mode 1 (cost dt·α = 10⁻⁴, nothing), and β = 2×10⁻⁵ keeps dt·β·ω_max² ≈ 0.55,
safely inside. Full derivation with numbers: kernels.cuh's damping comment.

**Tendons, and the buckling budget.** Each tendon is a distributed axial line force: every
non-base node on its fiber row receives T/120 in −x (toward the base). This is the standard lumped
abstraction of an *embedded/bonded* tendon or fiber-reinforced actuator — deliberately distinct
from a free frictionless cable, which, being straight and taut, would load only its endpoints; the
distributed form is what actually produces distributed bending moment. Estimate the bending: the
tendon force applied beyond station x is ΔT·(L−x)/L at height ±H/2, so the section moment is
M(x) ≈ ΔT·(H/2)·(L−x)/L, and the unit-load integral gives tip deflection
δ = ΔT·(H/2)·L²/(3EI) — predicting J = δ/ΔT ≈ 10.0 mm/N against the measured 12.0 mm/N; the ~20%
gap is mostly the P−δ softening below (which the *identified* J honestly absorbs and an assumed
analytic model would have missed). Both tendons carry a co-contraction bias (cables cannot push),
and the bias pays a structural price: summed tension compresses the arm axially, and a cantilevered
column buckles at

```
P_cr = pi^2 * E * I / (2L)^2 = 0.987 N,     I = t*H^3/12 = 2.304e-8 m^4
```

The chosen bias (0.25 N per tendon → 0.5 N total ≈ 51% of P_cr) keeps the arm below buckling but
*measurably softened* — lateral stiffness scales like (1 − P/P_cr) near critical load. Real
tendon-driven continuum robots live with exactly this constraint: over-tension the backbone and it
buckles.

**The model-based-control story.** "Model-based" here means the controller's gain comes from the
model, not hand-tuning against the plant. The honest version for soft robots is
**identification**: analytic soft-arm models are chronically wrong (material spread, hysteresis,
actuator coupling — and even here, the P−δ effect), so practitioners *measure* the
input-to-task-space Jacobian: apply a known input step, settle, read the output. This project does
precisely that against its own FEM (probe ΔT = 0.18 N → settle → J = δ_tip/ΔT = 12.0 mm/N), then
closes a PI loop:

```
dT = Kp*e + Ki*Integral(e dt),      e = y_ref - y_tip
Kp = margin/|J|      (loop gain `margin` at DC — and ~ margin/(2*zeta) at the arm's
                      resonance, so margin = 0.3 keeps the resonant loop gain ~0.7 < 1:
                      measured lesson — margin 0.6 left a persistent +/-0.5 mm limit cycle)
Ki = Kp/Ti, Ti = 0.15 s   (integrator crossover ~ margin/Ti = 2 rad/s, well below the
                           ~9-12.7 rad/s arm resonance)
```

with conditional-integration anti-windup (freeze the integral while the tension clamp saturates in
the error's direction — without it, a clamped transient winds up an integral that must unwind
through massive overshoot) and the differential mapped antagonistically, T = bias ± ΔT/2, the
clamp chosen so both tendons stay taut. Measured closed loop: rise-to-90% ≈ 1.16 s, zero
overshoot, steady-state error 0.04–0.18 mm on ±2.1 mm steps.

## The algorithm

Per timestep (33,333 per simulated second; both kernels in [`src/kernels.cu`](src/kernels.cu)):

1. **Element pass** (one thread per element): gather 4 corners' x and v (8+8 floats) → F (4 FMAs
   per entry) → θ = atan2(F21−F12, F11+F22) → local combo u_local + β·v_local (rotate by Rᵀ,
   subtract reference) → f_local = Et·K̂·combo (8×8 matvec, 128 FMAs) → rotate by R and
   **atomicAdd** the negative into the 4 corners' force slots.
2. **Node pass** (one thread per node): read the assembled force (then zero it for the next step —
   the zero-after-consume contract), add −α·m·v and this step's tendon share (± the analytic
   gate's point load), a = f/m, symplectic-Euler update; base nodes reset to their rest pose
   instead (the Dirichlet condition, enforced rather than solved).

Around that inner loop, `main.cu` stages: **verify** (500 steps GPU-vs-CPU) → **gate i** (settle
under a tip load, compare Euler-Bernoulli) → **gates ii + iii** (release, ring 4 periods:
zero-crossing frequency + energy record) → **identify** (probe, settle, J) → **closed loop**
(4 setpoints × 660 ticks × 100 substeps) → artifacts → the real-time factor.

Complexity per step: O(nelem) force work + O(nnode) integration, embarrassingly parallel within
each pass; the CPU twin is the same O, walked sequentially (measured 92 µs vs 18 µs per step — and
§The GPU mapping explains why the GPU number is overhead-bound, not compute-bound). Settling uses
two documented tricks: a **boosted α = 12 s⁻¹** (dynamic relaxation — equilibrium is
damping-independent, so heavier damping just gets there ~3× sooner), and a settle criterion that
requires max|v| below threshold on **5 consecutive checks** — velocity dips through zero at every
oscillation turnaround, so a single check can fire mid-transient at ~20% overshoot (measured: it
once reported a 4.84 mm "static" deflection that was really a swinging arm, and inflated the
identified J by 14%; the consecutive-checks rule fixed both).

## The GPU mapping

```
elem_force_kernel:    1 thread = 1 element   (1440 threads, 256/block)
  registers : corner x,v (16f), F (4f), combo/f_local (8f each) — no spills at this size
  constant  : K_hat (64 floats) — every thread, every launch: the textbook broadcast
  global    : 16 reads (corner data, L2-served), 8 atomicAdd writes (contended <= 4-way)

node_integrate_kernel: 1 thread = 1 node     (1573 threads, 256/block)
  global    : force read + zeroed in place, x/v read+write — all perfectly coalesced
              (node index = thread index; i is the fast axis, repo convention)
```

**The assembly-strategy taxonomy** (the load-bearing design decision — contrast 26.01):

| Strategy | Thread owns | Writes | Price | Natural home |
|---|---|---|---|---|
| **Gather** (26.01's matvec) | a node | its own 2 DOF only | redundant *reads* (each element's data read by ≤4 node-threads) + per-node incident-element logic | matrix-free solvers, where the node-side product K·p is the goal |
| **Scatter + atomics** (this project) | an element | ≤8 `atomicAdd` into nodes it does not own | write contention (≤4-way here) + **non-deterministic summation order** | explicit assembly, where the per-element quantity (the rotation!) is the expensive part |
| Scatter + graph coloring | an element, within a color | plain writes, no atomics | one launch per color (4 for a 2-D grid), less parallelism per launch, extra bookkeeping | determinism-critical explicit codes |

Why scatter here when 26.01 chose gather over the *same* mesh neighborhood: the corotational
rotation is an **element** quantity — extracting θ needs all 4 of an element's corners at once, so
the natural "one thread, one problem" unit is the element. A gather formulation would recompute
each element's rotation up to 4× (once per adjacent node-thread) or stage rotations through an
extra kernel and buffer. We take the direct mapping, accept the atomics, and *document the price*:
the race is resolved correctly by hardware (no update is lost), but floating-point addition is not
associative, so per-node sums round differently run to run — which is exactly why the §5 gate is
tolerance-based (§How we verify) and why we did NOT choose coloring: at 1,440 elements the demo is
launch-overhead-bound already, and 4× the launches would cost more than the atomics do. (Coloring
becomes the right answer at determinism-critical scale; saying when each wins is part of the
lesson.)

**The performance regime, measured honestly.** This mesh is *small* for a GPU — 1,440 threads
cannot fill an RTX 2080 SUPER's 48 SMs even once. The per-step wall cost (~18 µs) is dominated by
kernel-launch/submission overhead, not arithmetic: folding the force-buffer zeroing into the node
kernel (dropping one cudaMemset per step — 3 API calls to 2) cut a measured **27.9 → 17.9 µs per
step**, a 36% saving from *not calling something*. Two lessons: (1) real-time small-model physics
is a **latency** problem, not a throughput problem — the exact regime CUDA Graphs (32.02) exists
for, and the honest explanation of why "small but fast forever" is its own discipline; (2) the GPU
still wins ~5× over the CPU twin here, and the margin grows with mesh size — the same two kernels
at 100× the elements would saturate the machine without changing a line.

## Numerical considerations

- **FP32 on the whole hot path**; double only for one-time derivations (K̂'s quadrature) and
  diagnostic accumulations (energy sums — 165 samples of ~10⁻⁵ J would otherwise bury the drift
  signal in accumulation rounding). Positions are O(0.1 m) with µm-scale steps — comfortably
  inside float's ~7 digits.
- **Atomics nondeterminism policy (decided, not hidden):** GPU force sums reassociate run to run.
  We keep the CPU twin *deterministic* (fixed element order) and compare within a
  reassociation-aware tolerance, rather than forcing GPU determinism via graph coloring — the
  taxonomy above says why. Measured after 500 steps: worst |Δx| ≈ 6×10⁻⁸ m, worst
  |Δv| ≈ 7×10⁻⁴ m/s; tolerances 10⁻⁵ m / 5×10⁻³ m/s sit ~100×/~8× above that noise and well below
  bug scale (~10⁻³ m+). Velocities get the looser bound on purpose: high-frequency mesh waves are
  phase-sensitive amplifiers of ulp noise, while positions integrate it away.
- **NaN discipline (learned the hard way):** `NaN > tol` is false — a naive tolerance check
  *passes* on NaN; `fabs(NaN) > m` is false — a naive max-velocity settle check reads a blown-up
  state as "settled"; and casting NaN to int is UB that hung this project's Bresenham rasterizer
  for five minutes. The first draft's unstable damping produced NaN that sailed through both
  checks. Every gate now tests `isfinite` explicitly, settle treats non-finite as +∞, and the
  rasterizer clamps. When a check *can* be fooled by NaN, it eventually will be.
- **The warped-stiffness flutter (this project's measured detective story — and why β is
  load-bearing):** with damping fully off, the model self-excites from *exact rest*: total energy
  grows from a ~10⁻¹¹ J rounding seed at ~6.4/s amplitude e-fold rate — at dt AND dt/2 (so not an
  integrator/CFL effect), with an amplitude-independent injection rate (so not a large-rotation
  effect). Cause: the warped force f = R·K̂·(Rᵀx − X) omits the ∂R/∂x variation, so its tangent is
  non-symmetric — a force field with curl, whose orbits can extract net work; a known pathology of
  the classic corotational shortcut (Chao et al. 2010 and McAdams et al. 2011 give the variational
  fix). Measured responses: β = 2×10⁻⁵ s quenches it completely (energy then *decays* 1.4% over 2
  periods); even β = 2.5×10⁻⁶ quenches it. Policy: keep the standard force (it is what the field's
  fast implementations actually use), document the crutch instead of hiding it, run the energy gate
  with a featherweight β = 5×10⁻⁶ s (mode-1 ζ = 3.2×10⁻⁵), and make the proper variational fix
  README Exercise 5.
- **Float energy drift, budgeted:** the 8% energy-gate bound decomposes as: ~4% leak of mode-1
  energy into β-damped mesh modes via the nonlinear rotation coupling (measured to be
  β-INdependent: 4.1% at β = 2.5×10⁻⁶ vs 5.3% at 2×10⁻⁵ — the coupling sets the rate, β only
  disposes of what arrives), 0.16% direct β dissipation of mode 1, ~0.02% symplectic bounded
  oscillation (ω₁·dt = 3.8×10⁻⁴). Measured total: 4.3%. A real defect blows through instantly —
  the fully-undamped flutter measured +41% over *half* the window.
- **Angle wrapping / quaternion drift:** N/A in 2-D — `atan2` returns (−π, π] and θ is consumed
  inside cos/sin within the same step, never accumulated across steps (CLAUDE.md §12's
  wrap-discipline satisfied trivially; stated for completeness).
- **Q4 honesty:** bilinear quads over-stiffen in bending (shear locking), and ν = 0.40 was chosen
  partly because ν → 0.5 (true rubber) volumetrically locks Q4 badly. The static gate's 30%
  allowance covers coarse-mesh exercises; the shipped mesh measures 0.6% — 12 elements through the
  depth is enough.

## How we verify correctness

Five independent checks, because a simulator can be numerically right and physically wrong — or
both right and still useless to a controller:

1. **GPU-vs-CPU twin (§5 gate; `VERIFY:`)** — 500 steps of full dynamics (damping + asymmetric
   probe-magnitude tendon load) from identical state through both paths; worst-DOF position and
   velocity deviations within the reassociation-aware tolerances above; every value finite.
   Catches indexing, layout, rotation-extraction, and scatter bugs (each shifts results 4+ orders
   above the reassociation noise).
2. **Analytic statics (`GATE static-deflection:`)** — settle under a 0.02 N tip load; tip sag vs
   Euler-Bernoulli δ = P·L³/(3EI) = 4.000 mm. Measured: 3.977 mm, 0.6% error, against a 30%
   allowance (which covers Q4 locking and 2-D-continuum-vs-1-D-beam effects — 26.01's honesty
   precedent about discretization, reused).
3. **Analytic dynamics (`GATE first-mode-frequency:`)** — release and ring; count RISING zero
   crossings of the tip trace (FFT-free frequency estimation: linearly interpolate each crossing
   time, average the spacings, invert) against
   f₁ = (1.875104²/2π)·√(EI/(ρ·A·L⁴)) = 2.029 Hz — 1.875104 being the first root of
   cos(kL)·cosh(kL) = −1, the cantilever's frequency equation. Measured: 2.000 Hz, 1.4% error,
   against a 20% allowance.
4. **Energy conservation (`GATE energy-conservation:`)** — the same ring's total energy
   (kinetic + corotational elastic PE, both computed by the CPU-side diagnostics on downloaded
   state) must stay within 8% of its initial value — 10.03's energy-gate precedent, adapted with
   the measured drift budget above.
5. **Closed-loop behavior (`IDENTIFY:` / `SETPOINT n:`)** — the Jacobian must be measurable and
   finite; every setpoint must be reached (90% rise inside the hold window), overshoot ≤ 60%, and
   the mean |error| over the final 10% of the hold ≤ 0.3 mm — catching everything the pointwise
   checks cannot: actuation-map sign errors, windup, mistuned gains, an unstable loop.

The scenario is committed (`data/sample/`, cross-checked against the compiled constants at
startup), the run needs no downloads, and the artifacts (`demo/out/`) make every claim
*inspectable* — plot the tip trace and read the rise times off it yourself.

## Where this sits in the real world

- **SOFA** (the bullet's "SOFA-style") is the field's real framework: implicit integration
  (backward Euler with CG/direct solves — unconditionally stable, so dt is chosen for accuracy,
  not CFL), corotational *and* hyperelastic elements, contact, constraints, and GPU plugins. The
  trade this project teaches by contrast: explicit = two trivial kernels + a brutal dt bound;
  implicit = one linear solve per step (26.01's matrix-free CG machinery is exactly what that
  needs!) + freedom in dt. Surgical simulators and SOFA's soft-robotics plugin sit implicit;
  games/haptics often sit explicit.
- **Elastica** replaces 2-D/3-D FEM with Cosserat rods (a 1-D centerline with directors) — the
  right reduction for slender arms, orders-of-magnitude cheaper, standard in octopus-arm-style
  research. **MuJoCo**'s deformable/flex bodies serve RL pipelines needing thousands of parallel
  rollouts (10.03's territory). IPC-class methods own the guaranteed-intersection-free high end.
- **The sim-to-real gap for soft robots** is the field's defining pain: elastomer batch spread,
  the Mullins effect (first stretches permanently soften silicone), creep, hysteresis, temperature
  drift, and hand-fabrication tolerances mean even a *perfect* FEM of the CAD is wrong about the
  physical arm on the bench. That is why the identify-then-control workflow this project teaches —
  measure the Jacobian, close a modest loop around it — is the practical baseline, and why serious
  stacks re-identify online or wrap learned residuals around the physics (12.x's territory).
- **What the full version adds:** 3-D hexahedral/tetrahedral elements with a true polar
  decomposition; hyperelastic laws (Neo-Hookean, Ogden) for real strain levels; implicit
  integration; contact; gravity and pre-strained equilibria; routed-cable tendon models with
  friction and slack; and on the control side, model-order reduction (POD, Koopman) or learned
  dynamics feeding MPC — the current research frontier for soft manipulators.
- **This project's honest position:** the smallest complete instance of the whole idea — a physics
  model fast enough to sit inside its own controller, verified against analytic truth, with its
  numerical skeletons (CFL, locking, flutter, buckling) exposed and measured instead of hidden.
  Everything bigger is more of the same, plus a linear solver.
