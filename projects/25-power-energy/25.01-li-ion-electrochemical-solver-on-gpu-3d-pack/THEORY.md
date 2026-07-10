# 25.01 — Li-ion electrochemical (P2D/SPMe) solver on GPU + 3D pack thermal simulation + cooling-design sweeps: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics. Define every symbol, unit, and frame
> on first use.

## The problem — physics & engineering first

### Why lithium intercalation is a diffusion problem

A lithium-ion cell stores energy by moving Li⁺ ions between two **intercalation host** materials — a
graphite anode and a layered-oxide cathode (NMC, NCA, LFP, …) — through an electrolyte that only
conducts ions, never electrons. Each host material is built from countless microscopic **particles**
(graphite flakes, oxide grains), each a few microns across. When current flows, Li⁺ crosses the
particle's *surface* via an electrochemical reaction, but getting from that surface to the particle's
*interior* — where most of the storage capacity actually lives — has no shortcut: it happens by
**solid-state diffusion**, ion by ion, driven purely by a concentration gradient, obeying Fick's second
law exactly the way heat, dye in water, or dopants in silicon do. There is no way to "push" lithium into
a particle faster than diffusion allows; you can only raise the *surface* concentration (by driving more
current) and wait for the gradient to relax inward. **This is the single physical fact that shapes
almost everything about how a Li-ion cell can be used**: fast charging is hard because diffusion is
slow (diffusivities here are of order 10⁻¹⁴–10⁻¹³ m²/s — a lithium ion needs roughly `R_p²/D` seconds
to equilibrate across a single particle, which for a 6 micron particle at `D`=3×10⁻¹⁴ m²/s is **1200
seconds — twenty minutes** for one particle radius, the exact number this project's own particles use);
charging faster than diffusion can keep up drives the *surface* concentration to the material's physical
maximum long before the *bulk* is full, which is exactly the condition that triggers lithium metal
plating (a real safety failure mode) rather than intercalation. Every fast-charging protocol in industry
is, underneath, a diffusion-management problem.

### Engineering constraints a real pack imposes

A robot's battery pack is not one particle — it is dozens of cells, packed tightly for volumetric energy
density, generating heat that has nowhere obvious to go. Three constraints this project takes seriously:

- **Diffusion sets a hard ceiling on usable current**, independent of how much power electronics can
  deliver — a controls/electrical engineer cannot "fix" this with a better inverter.
- **Heat generation scales roughly with `I²`** (ohmic) plus a smaller `I`-linear activation term (Butler-
  Volmer overpotential), so aggressive duty cycles (an AMR's stop-start driving, or a legged robot's
  burst-torque gaits) concentrate heat in short bursts, not steadily — exactly the shape this project's
  mission profile has (§ below).
- **Real cell/pack construction is anisotropic.** A cylindrical or prismatic cell is built by winding
  (or stacking) thin electrode/separator layers — heat conducts easily *along* those layers (through
  metal current-collector foils) but poorly *across* them (through many separator/electrolyte
  interfaces). This project's `kx`,`ky` (in-plane) vs. `kz` (through-plane) split is not a decoration —
  it is the single most consequential engineering fact about pack thermal design, and it is exactly what
  produces this project's headline finding (README "Expected output"): a bottom cold plate fights the
  *low-conductivity* axis and barely helps; a side cold plate rides the *high-conductivity* axes and
  helps a lot.

## The math

### State, units, frames

Everything is SI. Concentration `c` [mol/m³]; temperature `T` [K, always — never Celsius, anywhere in
this codebase]; voltage/OCV/overpotential `η` [V]; current `I` [A]; molar flux `j` [mol/(m²s)]; heat `q`
[W, per cell] or volumetric heat `q_vol` [W/m³, per voxel]; thermal conductivity `k` [W/(m·K)]. Full
struct layouts and the sign convention for `j` (positive = leaving the particle) live in
[`src/kernels.cuh`](src/kernels.cuh)'s header comment — read it before this section; it is not
repeated here.

### Solid-phase diffusion (Fick's second law, spherical)

Each electrode particle is modeled as a sphere of radius `R_p`. Concentration depends only on radius `r`
and time `t` (spherical symmetry — no angular dependence, the SPM's core simplifying assumption):

```
∂c/∂t = (1/r²) ∂/∂r ( D r² ∂c/∂r )        0 ≤ r ≤ R_p
```

Boundary conditions: regularity at the center (`∂c/∂r|_{r=0} = 0` — no flux crosses a point), and a
**Neumann** (flux) condition at the surface carrying the applied current:

```
-D ∂c/∂r|_{r=R_p} = j
```

where `j` [mol/(m²s)] is this electrode's molar flux, **positive when Li is leaving the particle**
(delithiation). `main.cu` maps a cell current `I_cell` [A] to `j` via `j = I_cell / (F·A_surf)`, where
`A_surf` is this electrode's total active particle surface area *per cell* (`kernels.cuh`) and `F` is
Faraday's constant (96 485 C/mol — the charge per mole of electrons/monovalent ions). During discharge
(`I_cell` > 0 by this project's convention) the anode delithiates (`j_a` > 0) and the cathode lithiates
(`j_c` < 0).

### Butler-Volmer kinetics and its closed-form inversion

The **reaction rate at a particle's surface** — how fast Li⁺ actually crosses the electrode/electrolyte
interface — is not instantaneous; it needs an **overpotential** `η` (a voltage penalty beyond the
equilibrium open-circuit voltage) to proceed at a given rate. The Butler-Volmer equation is the standard
model, derived from transition-state theory applied to the forward/backward electron-transfer reaction:

```
i = i0 · [ exp(α_a F η / RT) − exp(−α_c F η / RT) ]
```

`i` [A/m²] is the reaction current density, `i0` [A/m²] the **exchange current density** (how fast the
forward and backward reactions balance at equilibrium — effectively the reaction's "readiness"), and
`α_a`, `α_c` are transfer coefficients (fractions, summing to 1 for an elementary one-electron step).
This project uses the **symmetric case** `α_a = α_c = 0.5` everywhere, which lets the equation be
inverted **in closed form** (`sinh` is odd, so the two exponentials combine into `2·sinh`):

```
i = 2·i0·sinh(Fη / 2RT)   =>   η = (2RT/F)·asinh( i / (2·i0) )
```

— no Newton iteration needed, a genuine numerical-methods win this project takes deliberately
(`bv_overpotential()` in `main.cu` is exactly this one line). `i0` itself depends on how much lithium
room is left in the particle — this project uses the standard SPM simplification
`i0(x,T) = i0_ref(T)·√(x(1−x))`, `x = c_surf/c_max` (vanishes as an electrode approaches empty or full,
qualitatively correct BV behavior without needing an electrolyte-concentration state — see "The problem"
below for exactly what this assumption costs).

### Terminal voltage and heat generation

```
V_cell = OCV_cathode(x_c) − OCV_anode(x_a) + η_c − η_a − I_cell·R_ohm
q_cell = I_cell · ( [OCV_cathode(x_c) − OCV_anode(x_a)] − V_cell )
```

The sign convention (derived from the flux convention above: during discharge `η_c`<0, `η_a`>0) makes
both overpotentials and the ohmic term subtract from voltage during discharge and *add* heat regardless
of current direction — `q_cell` ≥ 0 both charging and discharging, exactly as physics requires (energy
dissipation, not creation, either way). **This project omits the reversible (entropic) heat term**
`T·(dU/dT)·I/F`, which needs an entropy-coefficient curve per electrode this teaching model does not
carry — an honest, documented omission (README "Limitations"), not an oversight.

### Synthetic OCV curves

`ocv_cathode(x) = 4.2 − 0.6x − 0.6x⁸` and `ocv_anode(x) = 0.1 + 0.5·exp(−30x) + 0.05(1−x)` (`main.cu`).
Both are monotonic (checked analytically: their derivatives never change sign on `[0,1]`) and shaped to
the right *qualitative* behavior — the anode's characteristic sharp early drop to a long low plateau
(graphite's real staging-transition shape), the cathode's smoother slope (an NMC-like layered oxide) —
**without being fit to any real dataset** (CLAUDE.md §8: never fabricate data passed off as real). At
this project's initial stoichiometries (`x_a`=0.75, `x_c`=0.40) the synthetic curves give a full-cell
OCV of ≈3.85 V — a plausible nominal Li-ion voltage, a useful sanity check that the curves are shaped
sensibly even though they are invented.

### Arrhenius electro-thermal coupling

Every rate constant in this model — diffusivity `D`, exchange-current prefactor `i0_ref` — is scaled by
temperature via the Arrhenius relation, referenced to 298.15 K (25 °C):

```
A(T) = A_25 · exp( −Ea/R · (1/T − 1/T_ref) )
```

This is the electro-thermal coupling the catalog bullet asks for, made concrete: a hotter cell diffuses
lithium faster and reacts more readily (lower kinetic overpotential) — the same physical reason real
packs age unevenly when some cells run hotter than others (see "Where this sits in the real world").
`main.cu` evaluates this in `double`: `T` sits close to `T_ref` (within a few kelvin for most of this
project's mission), so `1/T − 1/T_ref` is a **difference of two nearly equal numbers** — the classic
catastrophic-cancellation hazard that `float` alone would make noisy; doing the whole Arrhenius
evaluation in `double` (narrowing only the final result to `float` for the kernel) keeps this exact.

### The quasi-steady closed form (this project's analytic gate i)

Under a **constant** surface flux `j` (leaving-positive convention) and no other source, the particle's
average concentration drifts linearly (`d c_avg/dt = −3j/R_p`, from a volume-integrated mass balance:
total moles change at rate `−j·A_particle`, divide by volume `V = A_particle·R_p/3`), while the *shape*
of the profile relative to that average relaxes to a fixed quasi-steady form. Writing `c(r,t) = c_avg(t)
+ f(r)` and substituting into Fick's second law, `f` satisfies a steady ODE whose solution (regularity
at `r=0`, zero-mean over the volume, and the surface flux BC) is:

```
f(r) = −j·r²/(2·R_p·D) + 3j·R_p/(10D)          =>     f(R_p) = c_surf − c_avg = −j·R_p/(5D)
```

With this project's sign convention (`j` leaving-positive), **insertion** (`j`<0) gives `c_surf − c_avg
= (−j)·R_p/(5D) > 0` — the surface runs *ahead* of the average while lithium is flowing in, exactly as
intuition demands. This is the textbook quasi-steady result used to build reduced-order SPM models
(the "3-parameter model" of Subramanian et al. and its descendants) — a genuine, derivable closed form,
not an approximation invented for this project. `main.cu`'s `run_diffusion_fixture()` checks it directly
(README "Expected output" has the measured numbers, including the honest discretization-bias story).

### Coulomb counting (analytic gate ii)

Summing the shell mass balance over the whole particle **telescopes exactly**: every interior face's
flux appears with opposite sign in its two neighboring shells' balances and cancels; only the two
boundary terms survive (zero at `r=0`, the imposed `j` at `r=R_p`). So, **exactly**, regardless of `D`
or the profile's shape:

```
d(total moles)/dt = −j·A_particle
```

Forward-Euler applied uniformly to every shell preserves this identity **at every discrete timestep**
(the same telescoping argument applies term-by-term to the discrete update) — mass conservation here is
not an approximate outcome of a "good enough" scheme, it is an algebraic consequence of writing the
update in flux-divergence form at all. `run_diffusion_fixture()` checks measured `Δmoles` against
`−j·A_particle·t` and finds agreement to <0.4% (README) — the residual is pure FP32 accumulation
rounding over tens of thousands of steps, not scheme error, which is exactly what the tight 1%
tolerance is designed to catch (an indexing/sign bug would blow this apart by orders of magnitude, not
fractions of a percent).

### The pack heat equation

```
ρc_p ∂T/∂t = ∂/∂x(k_x ∂T/∂x) + ∂/∂y(k_y ∂T/∂y) + ∂/∂z(k_z ∂T/∂z) + q_vol
```

with **anisotropic** conductivity (`k_x`,`k_y` in-plane, `k_z` through-plane, "The problem" above),
`q_vol` [W/m³] the per-voxel heat source (each cell's `q_cell` spread uniformly over its 8×8×8 voxel
block), adiabatic (zero-flux) Neumann conditions on five of the six pack faces, and a **Robin**
(convective) condition on the sixth, design-selected face:

```
-k_n ∂T/∂n|_face = h·(T_face − T_coolant)
```

`n` the outward normal, `h` [W/(m²K)] the design's convective coefficient, `T_coolant` fixed.

### The exact steady-state energy balance (analytic gate iii)

At true steady state (`∂T/∂t=0` everywhere), integrating the heat equation over the whole pack volume
and applying the divergence theorem makes every *interior* conduction term vanish (heat only moves
around, it does not appear or disappear) and every *adiabatic* face contribute zero — leaving:

```
∫∫ h·(T_face − T_coolant) dA  =  ∫∫∫ q_vol dV   =   P_total
```

i.e. **every watt generated must leave through the one active face**, exactly, regardless of internal
gradients — this is a **global energy-conservation identity**, not a lumped-parameter (small-Biot-number)
approximation. `main.cu`'s `run_thermal_fixture()` runs a uniformly-heated, single-face fixture to near
steady state and checks the *average* boundary temperature against `T_coolant + P_total/(h·A_face)` —
3.5% measured error (README), attributable to "not perfectly at steady state yet" plus the FV boundary
discretization, with 5% tolerance leaving real margin.

## The algorithm

### Spherical finite-volume discretization

`R_p` is divided into `kNShells`=20 uniform-thickness shells, cell-centered: shell `s` spans
`[s·dr, (s+1)·dr)`, `dr = R_p/kNShells`. Shell volume and face areas use the **exact** spherical
formulas (not an approximation): `V_s = (4/3)π(r_out³−r_in³)`, `A = 4πr²` at each face. The FV update is

```
V_s · dc_s/dt = A_in·F_in − A_out·F_out,     F = −D·(c_far − c_near)/dr   (Fick's law between shell centers)
```

with `F_in=0` at `s=0` (center symmetry) and `F_out=j` at `s=kNShells−1` (the surface flux BC). Complexity:
`O(kNShells)` per particle per step, embarrassingly parallel across shells (each depends only on its
two immediate neighbors) and across particles (fully independent spheres) — `O(1)` parallel time per
step given enough threads.

### 3-D pack finite-volume discretization

Per axis, per voxel: an **interior** voxel uses the standard second difference
`k·(T₋−2T₀+T₊)/d²`; a voxel on an **adiabatic** boundary uses a **zero-gradient ghost**
(`T_ghost=T₀`), which algebraically collapses the same formula to the one-sided
`k·(T_neighbor−T₀)/d²` (substituting `T_ghost` for the missing neighbor); a voxel on **this design's
cooling face** adds, on top of that one-sided conduction term, a Robin source
`h·(T_coolant−T₀)/d` — the standard "half control volume touches the coolant directly" cell-centered
boundary treatment (a textbook Patankar-style Practice-B boundary node). Summing all three axes and
this step's `q_vol`, then dividing by `ρc_p`, gives `dT/dt`; forward Euler integrates it. Complexity:
`O(1)` per voxel per step (6 neighbor reads, a handful of flops), embarrassingly parallel across voxels
and across the 12 batched designs.

### Serial vs. parallel cost

One mission step (24 cells × 2 electrodes × 20 shells = 960 particle-shell updates, 12288 pack voxels,
× 12 batched designs): serially, `O(12 × (960 + 12288))` ≈ 159,000 scalar FV updates per step, 12000
steps → ≈1.9 billion scalar updates for the full sweep. The reference CPU path (`reference_cpu.cpp`)
does exactly this, sequentially, and is why `main.cu` only runs it on a small verify slice (§ below) —
the full sweep is a GPU-only exercise by design.

## The GPU mapping

### Thread-to-data mapping (both kernels; full detail in `kernels.cu`'s header comments)

`electrochem_fv_kernel`: one thread per (particle, shell) — flat index decoded as `s = idx % kNShells`
(shell, fastest axis), `p = idx / kNShells` (particle, `= (design×kNCells+cell)×2+electrode`). A 1-D
grid-stride-free launch, `256` threads/block, `ceil(total/256)` blocks. `thermal_step_kernel`: one
thread per (design, voxel) — `(i,j)` mapped to an 8×8 tile (`blockIdx.x`,`blockIdx.y`), `blockIdx.z`
decoded into `(design, k)` — extending 24.01's "batch rides in `blockIdx.z`" idiom from 2-D to 3-D.

### Memory hierarchy

Both kernels are small-working-set stencils: **no shared memory** is used, in both cases for the same
documented reason 07.09/31.01/24.01 give — each value is reused by at most a handful of immediate
neighbors, and the L2 cache covers that reuse at these array sizes (kNShells=20 floats per particle;
12288 voxels per design). **Registers** hold each thread's local scratch (face areas/volumes for the
sphere; the six neighbor reads for the voxel). `ElectrodeGeom`/`PackThermalParams`/`DesignPoint` arrive
**by value** (tiny structs, constant-cache-backed on arrival) — the same reasoning 24.01's `FeaGrid`
documents for uniform per-launch data every thread reads identically.

### Where the REAL parallelism lives (an honest accounting)

At this project's ratified scope — 24 cells, 12 designs, a 32×24×16 grid — the thermal kernel launches
147,456 threads and the electrochemistry kernel launches 11,520: both comfortably fill an RTX 2080
SUPER's 46 SMs, but neither is a *large* problem by GPU standards, and this project's own measured
numbers say so honestly: the 200-step verify slice shows the GPU **barely** beating (sometimes losing
to) the single-core CPU twin (README "Expected output" — a `B=1`, tiny-array workload dominated by
kernel-launch and PCIe-copy overhead, not compute). **The real parallelism this project's design is
built to exploit is the BATCH axis** — B designs solved in one launch sequence instead of B separate
sequential sweeps — and that story only gets *better* at scale: a real pack-engineering Monte Carlo
sweep (manufacturing-tolerance variation across thousands of cells, or an aging study across a 1000-cell
fleet-representative pack) would push the batch dimension from 12 to thousands, at which point this
same code's GPU advantage becomes the whole point (the full 12-design, 1200 s sweep already runs in
about 1.2 seconds of *kernel* time — README "Expected output" — the remaining wall-clock time is
per-step host↔device orchestration, exactly the "GPU pays off at scale" lesson repeated at a different
knob than 24.01/31.01 turn).

### What CUDA library calls this project uses

None. Every stencil is hand-rolled, per this project's teaching mandate (CLAUDE.md §5 default
dependency budget: CUDA runtime + C++17 standard library only) — there is no cuSOLVER/cuBLAS call
hiding a tridiagonal or sparse-matrix solve here, because the explicit time-stepping this project uses
never needs one (see "Numerical considerations" below for why explicit stepping is the right choice at
this project's timescales).

## Numerical considerations

### Precision

FP32 throughout the PDE state (concentration, temperature) — the repo's default (CLAUDE.md §5); all
HOST-side setup and bookkeeping (mission-profile math, OCV/Butler-Volmer/voltage, the Arrhenius
evaluation) is deliberately `double`, narrowed to `float` only at the point where it crosses into a
kernel argument — the same "setup beyond suspicion, only the solver is FP32 taught/measured" discipline
24.01's `kPi` and rasterization functions establish.

### Stability — the diffusion CFL

Explicit forward Euler on the spherical FV scheme is stable only while the innermost shell's own rate
constant, `≈3D/dr²`, satisfies `dt_e · 3D/dr² ≲ 1`. For this project's anode particle (`R_p`=6 μm,
`D`=3×10⁻¹⁴ m²/s, `dr = R_p/20` = 300 nm): `3D/dr² ≈ 1 s⁻¹`, so `dt_e` needs to stay well under 1 s —
this project's `dt_e = dt_thermal/n_sub = 0.1/5 = 0.02 s` sits comfortably inside that bound (measured:
`n_sub=1`, i.e. `dt_e`=0.1 s, is *also* stable at these parameters — the diffusion time constant is
~1200 s, far slower than a 0.1 s thermal tick; `n_sub`=5 is chosen for a different reason, resolving the
mission profile's abrupt current-step transitions a bit more finely, not for raw stability headroom).

### Stability — the thermal CFL

Explicit FTCS for the anisotropic heat equation is stable while
`dt_thermal ≤ ρc_p / [2·(k_x/dx² + k_y/dy² + k_z/dz²)]`. At this project's scenario values the bound
works out to ≈1.41 s — the committed `dt_thermal`=0.1 s carries a **14× margin**, checked and printed
by `main.cu` (`thermal_cfl_margin()`) *before* the mission loop runs a single step, exactly the
"check the CFL before trusting the run" discipline 31.01's `kCfl` documents. The thermal fixture used
for analytic gate iii deliberately uses FIXTURE-ONLY properties (`ρc_p`=4×10⁴, far below the real
scenario's 2×10⁶) to make the fixture's own thermal time constant short enough to reach steady state
in a tractable number of explicit steps — its own CFL margin (≈2×, still comfortably stable) is
independently checked in `run_thermal_fixture()`'s derivation comment.

### The measured kNShells=20 discretization bias

A standalone Python re-implementation of the exact FV scheme (used to prototype this project before
writing the CUDA/C++ twins) shows the quasi-steady analytic gate's ~12% gap is **not** a
convergence-in-time issue (identical at 2× and 80× the particle relaxation time) but an
**O(1/kNShells)** finite-volume discretization bias: 10 shells → 23%, 20 shells → 12%, 40 shells → 6%
— halving each time shell count doubles, the textbook signature of a first-order boundary-representation
error (the innermost/outermost shells are *centered* half a shell inside the true `r=0`/`r=R_p`, so
their discrete values run slightly behind the continuum profile's true endpoint values). `kNShells=20`
is this project's ratified scope; the analytic gate's 15% tolerance is set directly from this
measurement (a documented, derived number, not a padded guess) — README Exercise (implicitly, by
extension of this table) is to rerun at 40+ shells and watch the bias shrink further.

### Angle wrapping, quaternions, ill-conditioned Jacobians

None of these apply to this project — there are no angles, orientations, or Jacobians anywhere in this
model (a scalar diffusion PDE and a scalar heat equation, both purely radial/Cartesian). The
Arrhenius-evaluation cancellation hazard (§ above) is this project's actual numerical-hazard analogue,
and is handled the same way those hazards are handled elsewhere in the repo: identified explicitly and
computed in higher precision.

### Determinism

No RNG anywhere in this project — the mission profile, OCV curves, and Butler-Volmer inversion are all
deterministic closed-form math; the PDE state evolves by deterministic FP32 arithmetic. Bit-for-bit
reproducibility holds on one machine, one build; across compilers/platforms, FMA-contraction choices can
differ in the last few ulps after thousands of chained steps — following 24.01/31.01's precedent, no
STABLE (checked) output line carries a raw floating-point number, only PASS/FAIL verdicts against
tolerances with real, measured headroom (the table in README "Expected output" reports the actual
measured numbers on the reference machine, on `[info]` lines, honestly, without pretending they are
checked constants).

## How we verify correctness

Four independent layers, each catching a different class of bug:

1. **GPU-vs-CPU twin** (`VERIFY`): both kernels' plain-C++ oracles (`reference_cpu.cpp`) run against
   the GPU on a small (`B`=1, 200-step) slice, fed *identical* per-step inputs (§ "How the verify slice
   is driven" — `main.cu`'s `run_verify_slice()`), so any divergence is attributable only to the two PDE
   implementations, not to different driving data. Tolerances (`kTwinTolConc`, `kTwinTolTemp` in
   `kernels.cuh`) are set from measured FP32 rounding scale (a few ulps compounded over ~1000 chained
   FMA-bearing operations), with orders of magnitude of headroom below what an indexing/layout bug
   would produce.
2. **Closed-form analytic gates** (`ANALYTIC_DIFFUSION`/`ANALYTIC_COULOMB`/`ANALYTIC_THERMAL`): standalone
   fixtures checked against derivations in "The math" above — genuinely independent of whether the GPU
   and CPU merely agree *with each other* (they could both be wrong the same way; these gates check
   against actual mathematics).
3. **Physics sanity on the real run** (`PHYSICS`): the mission sweep's own thermal-grid energy balance,
   checked for every one of the 12 designs, using host-side bookkeeping that is INDEPENDENT of (not
   derived from) the kernel's own internal arithmetic — a genuine regression check, not a tautology.
4. **Spot-checked physical plausibility**: the synthetic OCV curves' resulting nominal cell voltage
   (~3.85 V) and the mission's SOC swing (~15–40% over 20 minutes, "The math" / `data/README.md`) are
   sanity-checked by hand against real Li-ion cell behavior, even though no number here is claimed to
   match a real cell.

For a stochastic-algorithm project this section would also cover fixed-seed/statistical comparison —
not applicable here (no RNG anywhere, § above).

## Where this sits in the real world

### SPMe and full P2D — the rest of the ladder

**SPMe** (Single Particle Model with electrolyte) adds a second PDE: 1-D-in-x diffusion of Li⁺ through
the electrolyte-filled separator and porous electrodes,

```
ε ∂c_e/∂t = ∂/∂x( D_e^eff ∂c_e/∂x ) + (1−t⁺)/F · a·j(x)
```

(`ε` porosity, `D_e^eff` effective electrolyte diffusivity, `t⁺` the lithium transference number, `a`
the electrode's specific interfacial area) — and makes the exchange-current density `i0` depend on the
LOCAL electrolyte concentration `c_e(x)` instead of this project's constant-`c_e` assumption. This is
exactly what SPM cannot capture: **at high C-rates, electrolyte concentration develops real
through-thickness gradients** (it depletes near the separator on the discharging electrode), and
ignoring that (as SPM does) makes SPM systematically over-predict achievable rate capability — the
precise reason production battery-management tools graduate to SPMe or full P2D for anything beyond
mild C-rates. Full **P2D** (Doyle-Fuller-Newman / Newman's model) goes one step further: instead of ONE
representative particle per electrode, it solves a coupled ensemble of particles distributed through the
porous electrode's thickness, each with its own local current density and local electrolyte state — the
production-grade model class this project's SPM is a deliberate, honestly-scoped simplification of.

### How production stacks differ

**PyBaMM** implements the full ladder (SPM → SPMe → P2D → thermally-coupled variants) in Python with
automatic differentiation and a real nonlinear/DAE solver (CasADi) instead of this project's fixed
forward-Euler stepping — production electrochemistry tools almost universally use **implicit** or
semi-implicit time integration precisely because real P2D models are numerically stiffer (through
tightly-coupled electrolyte/solid-phase equations) than this project's cleanly-decoupled SPM. **COMSOL's
Battery Design Module** couples P2D-class electrochemistry to a full 3-D FEA thermal (and optionally
mechanical/stress) model on an unstructured mesh — the closest commercial analogue to this project's
"electrochemistry + 3-D pack thermal" combination, at production fidelity and with real CAD-derived
geometry instead of this project's regular-grid cell blocks.

### The pack-design lesson, stated for a reader who skips straight to this section

The headline sweep result (README "Expected output" — bottom-plate cooling is conduction-limited and
barely responds to `h`; side-plate cooling is boundary-limited and responds strongly, at the cost of
larger cell-to-cell spread) is not a curiosity of this project's specific numbers — it is the real
reason production pack engineers care about **where** current collectors and thermal interface material
route heat, not just how strong the coolant loop is. A pack design that achieves an excellent *average*
or *peak* temperature by cooling along the low-conductivity axis can simultaneously create the worst
*cell-to-cell* imbalance, which is the metric that actually drives uneven long-term aging (hotter cells
lose capacity faster — an Arrhenius effect on the SAME degradation mechanisms this project's electro-
thermal coupling models, just extended to a slower, cycle-count timescale this project does not
simulate) and, eventually, pack-level capacity fade driven by its worst cell. "Which cooling design is
best" genuinely depends on which failure mode a given robot's duty cycle and service life care about
more — this project gives a reader the simulation machinery to ask that question quantitatively instead
of by rule of thumb.
