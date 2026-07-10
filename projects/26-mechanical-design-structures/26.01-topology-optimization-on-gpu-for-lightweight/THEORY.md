# 26.01 — Topology optimization (SIMP) on GPU for lightweight links and brackets — flagship design project: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**The question this project answers:** given a design domain (a rectangle you're allowed to put
material in), a place it's bolted down, a place a load is applied, and a material budget (say,
"use at most 40% of the volume"), **where should the material go** to make the resulting part as
*stiff* as possible? This is not a shape you sculpt by hand — it is the output of an optimization
algorithm that starts from a uniform gray slab and, over dozens of iterations, decides element by
element whether each tiny patch of the domain should be solid, void, or (briefly, on the way to a
decision) something in between.

**Why this matters for a robot.** Every link and bracket on a robot is a compromise between two
things that pull in opposite directions: it must be *stiff* (a floppy link ruins a manipulator's
positioning accuracy; a floppy bracket lets a motor shift under load and destroys a gear mesh) and
it must be *light* (every gram of link mass is a gram the actuator upstream of it has to
accelerate — for a legged robot or a drone, mass is the single most expensive resource in the
whole system, since it multiplies through the entire kinematic chain). Traditional design puts
material everywhere "just in case" and calls it conservative; topology optimization instead asks
the physics directly: *for this exact load path, where is material actually doing work, and where
is it dead weight?* The answer is very often a strange, organic-looking lattice of struts that a
human designer would not have sketched — and it is provably (within the model's assumptions) the
stiffest structure achievable at that material budget. This is now a standard step in the design of
aerospace brackets, legged-robot links, and 3D-printed manipulator components (README "Prior art").

**The physics: linear elasticity, plane stress.** We model the 2D cross-section of a flat part
(think of a bracket cut from sheet stock, or a link's web) as a continuum obeying Hooke's law:
stress is linear in strain. "Plane stress" is the standard 2D idealization for a THIN part loaded
in its own plane: the out-of-plane stress component `sigma_zz` is assumed zero (the part is free to
thin or bulge slightly at its free faces — nothing is clamping it in the third direction). This is
distinct from "plane strain" (used for a thick slice deep inside a long prismatic body, where
`epsilon_zz=0` instead) — the wrong choice measurably changes the effective stiffness by a factor
of `(1-nu^2)`, so THEORY (this section) commits to plane stress and every formula below assumes it.

**The engineering constraints a real bracket imposes**, which this project's `PRACTICE.md`
elaborates: manufacturing tolerances (a topology-optimized organic shape is easy to 3D print but
hard to CNC mill — PRACTICE §1), fatigue at re-entrant (concave) corners (this project's L-bracket
case has exactly such a corner, and THEORY revisits it below), and the honest fact that the
2D idealization ignores out-of-plane buckling, which a real thin bracket can suffer even when its
*in-plane* compliance is excellent (README "Limitations & honesty").

## The math

### Governing equation

For a 2D linear-elastic body occupying a domain `Omega` with prescribed zero displacement on part
of the boundary (`Gamma_u`, the "Dirichlet" boundary — where the part is clamped) and prescribed
traction on the rest (`Gamma_t`, the "Neumann" boundary — where a load is applied), static
equilibrium under plane-stress Hooke's law is a linear PDE. Its finite-element **discretization**
is the linear system this project actually solves:

```
K U = F
```

- `K` — the **global stiffness matrix** (`ndof x ndof`, `ndof = 2 * n_nodes` — 2 displacement
  degrees of freedom, `ux` and `uy`, per node). `K` is symmetric positive semi-definite (positive
  DEFINITE once the Dirichlet boundary removes rigid-body motion) and, for a structured grid,
  extremely SPARSE — each row has at most 18 nonzeros (a node's own 2 dofs plus its 8 neighbors'
  16 — the "3x3 stencil" the GPU mapping section returns to).
- `U` — the unknown nodal **displacement** vector (m), `U = [ux_0, uy_0, ux_1, uy_1, ...]`.
- `F` — the applied nodal **force** vector (N), zero everywhere except at loaded nodes.

`K` is never assembled as an explicit sparse matrix in this project — see "The GPU mapping" for why.

### The element: bilinear (Q4) plane stress

The domain is discretized into a **structured grid of square elements**, each with 4 corner nodes
(`node0`..`node3`, CCW — the exact indexing contract lives in
[`src/kernels.cuh`](src/kernels.cuh)'s "MESH & DOF LAYOUT" comment). Displacement inside an element
is interpolated **bilinearly** from its 4 corner values via shape functions defined on a `[-1,1]^2`
parent square (`(xi,eta)` are the parent coordinates; `(xi_i,eta_i)` are node `i`'s parent-corner
signs, `(-1,-1),(1,-1),(1,1),(-1,1)`):

```
N_i(xi,eta) = 1/4 (1 + xi_i*xi) (1 + eta_i*eta),   i = 0..3
```

Strain is the symmetric gradient of displacement; substituting the bilinear interpolation gives the
**strain-displacement matrix `B`** (3x8: rows `[exx, eyy, gamma_xy]`, columns the 8 nodal dofs):

```
dN_i/dx = dN_i/dxi * dxi/dx,   dN_i/dy = dN_i/deta * deta/dy      (chain rule through the Jacobian)

B = [ dN0/dx    0     dN1/dx    0    dN2/dx    0    dN3/dx    0  ]
    [   0     dN0/dy    0     dN1/dy   0     dN2/dy   0     dN3/dy]
    [ dN0/dy  dN0/dx  dN1/dy  dN1/dx dN2/dy  dN2/dx dN3/dy  dN3/dx]
```

Plane-stress Hooke's law gives the **material matrix `D`** (factoring `E` out explicitly —
`D = E * D_hat(nu)`, `D_hat` a pure function of Poisson's ratio):

```
D_hat(nu) = 1/(1-nu^2) * [ 1   nu        0     ]
                          [ nu   1        0     ]
                          [ 0    0   (1-nu)/2   ]
```

### The element stiffness matrix, derived (not hardcoded)

The element stiffness is the standard FEM weak-form integral, evaluated by **2x2 Gauss quadrature**
(exact for this bilinear-times-bilinear integrand — zero quadrature error, not an approximation):

```
K_e = t * INTEGRAL_over_element  B^T D B  dA
    = t * E * SUM_{gauss points} w_p * B(xi_p,eta_p)^T D_hat B(xi_p,eta_p) * det(J)
```

`t` is the out-of-plane thickness; this project uses the same convention as project 24.01's
torque-per-meter-of-stack-length: **`t = 1 m`**, i.e. every reported force/compliance number is
"per meter of out-of-plane thickness" — a real bracket of thickness `t_real` scales forces (and
mass) by `t_real` (PRACTICE §1 makes this concrete for the two committed scenarios).

**Why the element size `h` never appears.** Mapping the `[-1,1]^2` parent square to a physical
`h x h` square element gives `x = h/2*(xi+1)`, so `dx/dxi = h/2` (and identically for `y`). The
Jacobian determinant is therefore `det(J) = h^2/4`, while every entry of `B` carries a factor
`dxi/dx = 2/h` from the chain rule above. `B^T D B` therefore scales as `(2/h)^2`, and multiplying
by `det(J) = h^2/4` gives `(2/h)^2 * (h^2/4) = 1` — **the `h`'s cancel exactly**, for ANY square
element, leaving an integral that depends only on `nu` (through `D_hat`) and is otherwise pure
geometry on the fixed parent domain. This project computes that integral numerically, in double
precision, at startup (`compute_KE_hat()` in
[`src/reference_cpu.cpp`](src/reference_cpu.cpp) — shared by both the GPU and CPU paths, so there
is exactly one derivation, not two that could silently drift) — calling the result `KE_hat`
(DIMENSIONLESS): every element's real stiffness is exactly

```
K_e = E(rho_e) * KE_hat      (t = 1 m folded in, per the convention above)
```

`compute_KE_hat(nu=0.3)` reproduces `KE_hat[0][0] = 0.494505`, the exact value quoted by the
"99-line topopt" reference implementation (Sigmund 2001; Andreassen et al. 2011) — a genuine
cross-check that this project's from-scratch Gauss-quadrature derivation matches the literature,
printed as an `[info]` line every run.

### SIMP: penalizing "50% material"

The design variable is a per-element **density** `rho_e in [0,1]`. A naive linear interpolation
`E(rho) = rho * E0` lets the optimizer cheat by using lots of "50% material" — cheap in the volume
constraint, structurally almost as good as solid in a linear model, but physically meaningless (you
cannot manufacture "50% aluminum"). **SIMP** (Solid Isotropic Material with Penalization) instead
uses

```
E(rho_e) = Emin + rho_e^p (E0 - Emin),    p = 3 (fixed, this project's ratified scope)
```

Because `p > 1`, a unit of density bought at `rho=0.5` returns only `0.5^3 = 0.125` of the
stiffness a unit at `rho=1` would — intermediate densities become a BAD trade relative to their
volume cost, and the optimizer is driven toward `rho in {0, 1}` "for free," as a side effect of
minimizing compliance under the SAME volume constraint. `p=3` is the textbook value: `p<1` fails to
penalize (concave, rewards intermediate densities); very large `p` makes the sensitivity landscape
so steep that OC's bisection struggles to make stable progress (large jumps that overshoot,
oscillate). `Emin` (never exactly 0) keeps `K` positive definite everywhere — a literal zero would
make void elements contribute an exactly-singular local stiffness block.

### Compliance and its sensitivity

**Compliance** `c = F^T U` (Joules) is this project's objective: the work the load does moving
through the resulting deflection — LOWER compliance means a STIFFER structure for that load. Using
`U = K^-1 F` and the fact that `dK/drho_e` is zero everywhere except element `e`'s own 8x8 block,
the classical **adjoint-free** sensitivity (self-adjoint because compliance minimization is a
special case where the adjoint problem equals the primal problem) is:

```
dc/drho_e = - u_e^T (dK_e/drho_e) u_e = - dE/drho_e * u_e^T KE_hat u_e
dE/drho_e = p * rho_e^(p-1) * (E0 - Emin) = 3 rho_e^2 (E0-Emin)          (p=3)
```

where `u_e` is element `e`'s 8 local displacements, gathered from its 4 corner nodes' solved `U`.
This is a genuinely CHEAP per-element quadratic form once `U` is known — the entire reason SIMP
with the adjoint trick is tractable: no second linear solve is needed per iteration, only one FEA
solve plus one small element-parallel kernel (`elem_sensitivity_kernel` in
[`src/kernels.cu`](src/kernels.cu)).

### The checkerboard pathology, and why filtering fixes it

A well-known failure mode of naive SIMP: the optimizer discovers that alternating solid/void
elements in a checkerboard pattern are ARTIFICIALLY stiff **in the discretized model** — the
coarse Q4 discretization underestimates a checkerboard's true (physical) compliance, so the
optimizer "cheats" the mesh rather than solving the real physics, and the result is
mesh-DEPENDENT (refine the mesh and you get a DIFFERENT, finer checkerboard, not a converged
shape). **Sigmund's density filter** (Sigmund 1997, 2001) fixes this by making each element's
EFFECTIVE sensitivity a distance-weighted average of its neighbors' *raw* sensitivities within a
radius `rmin` (this project uses `rmin = 2.4` elements — kernels.cuh's `kFilterRMin`):

```
dc_filt_e = ( SUM_f w(e,f) * rho_f * dc_raw_f )  /  ( rho_e * SUM_f w(e,f) )
w(e,f) = max(0, rmin - dist(e,f))
```

A single isolated solid element surrounded by void (the checkerboard's building block) now has its
sensitivity DILUTED by its void neighbors' near-zero `dc_raw` contribution (weighted by their
near-zero `rho_f`) — the checkerboard is no longer attractive to the optimizer. The `rho_f` weight
in the numerator (not present in the earliest 1997 formulation, added in the widely-used later
form) additionally protects near-void elements from division blow-up in the denominator. **This is
also the mesh-independence fix**: `rmin` fixes a PHYSICAL length scale (in element units) that the
optimizer cannot resolve features finer than, so refining the mesh refines the *representation* of
a converged shape rather than discovering new, spurious, ever-finer structure (README Exercise 2:
disable the filter and watch the checkerboard reappear).

### The Optimality Criteria (OC) update

Minimizing `c(rho)` subject to `mean(rho) <= volfrac` (over active elements) and box constraints
`0<=rho_e<=1`, the KKT stationarity condition at an interior optimum sets the objective's gradient
proportional to the constraint's gradient: `dc/drho_e = lambda * dV/drho_e` for some Lagrange
multiplier `lambda >= 0` (with `dV/drho_e = 1`, a uniform-element-volume convention). The classic
**heuristic** OC update (Bendsoe 1995) rearranges this into a fixed-point iteration:

```
Be = -dc_filt_e / lambda                    (should equal 1 at the exact optimum)
rho_new_e = clip( rho_e * sqrt(Be),  rho_e - move,  rho_e + move,  [0,1] )
```

`sqrt(Be)` is a damped multiplicative step (bigger `Be` -> "this element wants more material" ->
grow it) and the `move` limit (this project: 0.2, the standard value) prevents any one element's
density from swinging wildly in one iteration — the stabilizer that keeps this greedy heuristic
well-behaved despite having no formal convergence proof (unlike a proper primal-dual interior-point
method). `lambda` is found by **bisection**: for a fixed `lambda`, the formula above gives a
definite `rho_new` and hence a definite `mean(rho_new)`; increasing `lambda` monotonically
DECREASES `mean(rho_new)` (a larger multiplier makes every `Be` smaller), so bisecting `lambda`
until `mean(rho_new) == volfrac` is a 1D root-find with a monotone function — always converges,
and only costs `O(n_elements)` work per bisection step (cheap; this project runs 60 steps, tiny
next to one FEA solve). Because this project uses REAL PHYSICAL units (unlike the classic 99-line
code's nondimensional `E=1`), `lambda`'s natural scale is not known in advance, so
`src/main.cu`'s `oc_update()` DISCOVERS a bracket by geometric doubling rather than hardcoding
`[0, 1e9]` — see that function's comment for the derivation.

### The level-set alternative (documented, not implemented)

The catalog bullet offers "SIMP/level-set." **Level-set topology optimization** represents the
solid/void boundary implicitly as the zero contour of a scalar function `phi(x,y)` (solid where
`phi>0`, void where `phi<0`), and evolves `phi` by a Hamilton-Jacobi PDE driven by the same shape
sensitivity SIMP computes, `d(phi)/dt + V |grad(phi)| = 0` with velocity `V` derived from the
boundary's normal sensitivity. Its chief advantage over SIMP is a CRISP boundary at every iteration
(no intermediate-density "gray" elements to interpret or penalize away) and natural topological
changes via boundary merging/splitting; its chief cost is the extra machinery of narrow-band
reinitialization (`|grad(phi)|=1` must be periodically restored) and a generally slower, more
delicate convergence than SIMP's blunt-but-robust OC update. This project implements the SIMP
lineage exclusively (the ratified scope); a from-scratch level-set implementation would reuse this
project's FEA solver as its physics core and add a level-set advection kernel — a natural README
Exercise for a learner who wants to extend this project.

## The algorithm

**Per outer SIMP iteration** (the numbered steps are labeled in `run_simp()` in
[`src/main.cu`](src/main.cu)):

1. **FEA solve**: `K(rho) U = F` via matrix-free Jacobi-preconditioned CG (>95% of the arithmetic —
   "The GPU mapping" below). WARM-STARTED from the previous iteration's `U`.
2. **Sensitivity**: one small GPU kernel computes, per element, `ce = E(rho_e) q_e` and
   `dc_raw = -dE/drho_e q_e` where `q_e = u_e^T KE_hat u_e` (a per-element quadratic form).
3. **Filter**: one small GPU kernel produces `dc_filt` from `dc_raw` and `rho` (the formula above).
4. **OC update + bisection** (host): produces `rho_new` meeting the volume constraint exactly (to
   bisection tolerance).
5. Repeat until `max_outer` iterations (this project: 80) or the design has settled (max per-element
   density change `< 0.01` after iteration 6 — `run_simp()`'s early-stop check).

**Complexity per outer iteration**: `O(CG_iters * ndof)` for the FEA solve (the dominant term —
CG_iters ranges from ~400 (capped, cold start) down to tens once warm-starting kicks in — see "The
GPU mapping"), `O(n_elements)` for sensitivity/filter/OC. A DIRECT solve (assemble `K`, Cholesky
factorize) would cost `O(ndof * bandwidth^2)` per iteration for a banded direct solver — for this
project's mesh sizes the CG approach is already competitive on a CPU and, on a GPU, is the only
approach that maps cleanly at all (next section).

## The GPU mapping

### Matrix-free CG: why assembly-free beats assembled-sparse on a GPU here

A "textbook" FEA solver ASSEMBLES a global sparse stiffness matrix (e.g., in CSR format) once per
density field, then hands it to a sparse direct or iterative solver. This project deliberately does
NOT assemble `K`. Two reasons, one GPU-specific:

1. **`K` changes every outer iteration** (it depends on `rho`, which the optimizer updates every
   step) — an assembly step that rebuilds a sparse structure (or even just its nonzero VALUES,
   holding the sparsity pattern fixed) every iteration is pure overhead compared to recomputing
   contributions on the fly from the density field, which is already sitting in GPU memory.
2. **Sparse matrix-vector products are a poor GPU fit at this scale.** CSR SpMV is
   latency/bandwidth-bound with irregular memory access (row pointers, column indices, gather-heavy)
   — the classical GPU sparse-linear-algebra literature spends a lot of effort just making SpMV
   efficient. A MATRIX-FREE method sidesteps the problem entirely: conjugate gradient never needs
   `K` itself, only the ACTION `K*p` for an arbitrary vector `p` — and for a structured FEM mesh,
   that action is exactly "each node sums a fixed, small, precomputable contribution from its
   (at most 4) incident elements" — dense, regular, perfectly-known-in-advance memory access. This
   project computes `K*p` with the `matvec_gather_kernel` in
   [`src/kernels.cu`](src/kernels.cu): one thread per NODE, reading its incident elements' densities
   and the tiny shared `KE_hat` constant, writing its own 2-dof output. No matrix is ever built,
   stored, or synchronized — the density field itself *is* `K`, implicitly.

### GATHER vs. SCATTER — the race-condition story

The alternative GPU mapping is one thread per ELEMENT, each computing its local `K_e * p_e`
contribution and **scattering** the 8 results into the (up to 4) nodes that own them via
`atomicAdd` — since multiple elements share every interior node, multiple threads would race to
write the SAME output entries, and atomics are mandatory to serialize them correctly. This
project's kernel instead launches one thread per NODE, and each thread **gathers** — reads (never
writes) its up-to-4 incident elements' data and owns the single output entry it alone writes. Zero
atomics, zero write races, BY CONSTRUCTION, at the cost of redundant reads (a node shared by 4
elements is read by all 4 of THEIR gather passes, and a node's own value is re-read by up to 4
neighboring node-threads' gathers) — but reads are cheap and cacheable (the GPU's L2 cache absorbs
this reuse at this problem's scale, exactly the reasoning project 24.01 documents for its own FEA
stencil), while atomics genuinely serialize contending writes. **GATHER is the right default
whenever the output's owner is known up front** — true here because a structured Q4 mesh's
node-to-incident-element map is fixed, small, and computable from indices alone (the "quadrant"
table in `kernels.cu`'s `quadrant_elem()`). A scatter+atomics variant is README Exercise 4 — try it
and measure the atomic-contention cost directly.

### The stencil: richer than a scalar Poisson problem

Projects 07.09/24.01/31.01 solve SCALAR 2D PDEs with the classic 5-point stencil (4 neighbors +
self). This project's PDE is VECTOR (2 dofs per node) with BILINEAR (not just nearest-neighbor)
coupling, so a node's true stiffness support is its full **3x3 neighborhood** (9 nodes, 18 dofs) —
the same 4 incident elements every structured-Q4-mesh node has, just now contributing a small
DENSE 2x8 block each instead of a single scalar weight. `matvec_gather_kernel` implements this
directly: for each of a node's (up to 4) incident elements, gather that element's 8 local values
(its own 4 corner nodes' `x`), then contract against the appropriate 2 rows of `KE_hat`.

### Memory hierarchy

- **`__constant__` memory**: `KE_hat` (64 floats) and the filter weight table (25 floats) — EVERY
  thread in EVERY relevant kernel reads the exact same bytes every launch, the textbook broadcast
  use case for the constant cache (the same reasoning projects 08.01/09.01 document for their own
  per-launch-uniform data).
- **Global memory (L2-cached reuse, no shared memory)**: `rho`, `x`/`U`, `fixed` — read redundantly
  by neighboring node-threads (the GATHER trade above); at this project's mesh sizes (thousands to
  tens of thousands of elements) the working set fits comfortably in an RTX-class GPU's L2, so an
  explicit shared-memory tile (which WOULD help at much larger grids, or on GPUs with smaller L2)
  is left as README Exercise 4's natural next step rather than built in from the start — the same
  "measure before adding shared memory" discipline project 24.01's FEA solver documents.
- **Registers**: each node/element thread's local 8-float gathered vector and running sums —
  small, fixed-size arrays that stay register-resident (no spilling at this problem's element
  size), the same discipline project 08.01's rollout kernel documents for its own fixed-size state.

### Reduction: the CG dot products

Conjugate gradient needs three scalar dot products per iteration (`dot(p,Kp)` for `alpha`,
`dot(r,r)` for the stopping test, `dot(r,z)` for `beta`). `reduce_dot_kernel` computes these with
the standard two-level pattern: each thread accumulates a grid-stride partial sum, a
power-of-two shared-memory tree reduction collapses each BLOCK's partial sums to one value, and the
(at most a few thousand) per-block partials are summed on the HOST in double precision — a
deliberate choice (matching project 08.01's "keep the small trailing arithmetic on the host, in
plain sight" call for its softmin blend): the final sum is microseconds of work, and finishing it
in plain C++ keeps the whole CG recursion (`launch_topo_cg_solve` in
[`src/kernels.cu`](src/kernels.cu)) readable end-to-end instead of requiring a second reduction
kernel. README Exercise 5 asks you to fuse this into a device-side two-level reduction and measure
what the host round-trip costs per CG iteration.

### Warm-starting: the performance decision that makes the demo's time budget work

`launch_topo_cg_solve`'s `d_U` parameter is IN/OUT: `run_simp()` never resets it to zero between
outer iterations (only at the very first, cold-start iteration). Once the design has moved past its
first few, wildly-changing iterations, consecutive densities `rho` differ only slightly, so the
PREVIOUS solution is already an excellent initial guess for the NEXT solve — measured on this
project's own MBB reference run: the first (cold-start) FEA solve needs the full `kMaxCgIters=400`
cap to reach `1e-3` relative residual; by the final iterations, warm-started CG converges in as few
as **44-259 iterations** depending on the case (the `[info]` lines each run print the measured
first/last counts). This is not a minor tweak — without it, every one of the 80 outer iterations
would pay the cold-start cost, and the demo's ~8-second measured runtime (well inside the ~90s
budget this project documents) would instead run several times longer.

## Numerical considerations

- **FP32 throughout, with a documented conditioning trade.** SIMP's classic `Emin/E0` ratio is
  `1e-9` (Andreassen et al. 2011) — an extreme stiffness contrast between "solid" and "void"
  elements that production topology-optimization codes handle with GEOMETRIC or ALGEBRAIC
  MULTIGRID preconditioners (the standard answer to ill-conditioned elasticity systems with
  heterogeneous coefficients). This project's matrix-free CG uses only a JACOBI (diagonal)
  preconditioner — chosen because it maps to a trivial elementwise GPU kernel with no setup cost,
  keeping the CG solver's teaching core small and legible. A `1e-9` contrast under Jacobi-only
  preconditioning would need thousands of CG iterations per solve to reach even a modest relative
  residual — incompatible with this project's documented ~90-second demo budget across 80 outer
  iterations x 2 load cases. This project therefore uses **`Emin/E0 = 1e-3`**: still functionally
  void (0.1% of solid stiffness — negligible next to any solid element sharing a node with it) but
  keeping the linear system's condition number tractable for plain Jacobi-CG. This is a documented,
  measured trade (the [info] lines print achieved CG iteration counts and residuals every run), not
  a hidden shortcut — README Exercise 3 asks you to try `1e-6` and watch CG iteration counts (and
  runtime) grow accordingly.
- **The CG stopping rule**: relative residual `||r|| / ||F|| < 1e-3` (`kCgRelTol`), capped at
  `kMaxCgIters=400` regardless. Both are DELIBERATELY loose for the OUTER SIMP loop: a design that
  is about to change again next iteration does not need its FEA solved to machine precision — only
  the CANTILEVER BEAM and PATCH TEST analytic gates (below) tighten this to `1e-7` over up to 4000
  iterations, because THEY are checking the solver's asymptotic correctness, not feeding an
  optimizer.
- **No transcendental functions in the SIMP interpolation.** `p=3` is fixed by this project's
  ratified scope, so `E(rho)`/`dE/drho` hand-multiply `rho*rho*rho` instead of calling
  `powf(rho,3)` / `pow(rho,3)`. This buys two things: one fewer transcendental call (cheap, but not
  the main point), and — more importantly — nvcc's `powf` and MSVC's `pow` are NOT guaranteed
  bit-identical for the same input, so calling either would inject an extra, uncontrolled source of
  GPU-vs-CPU divergence into the §5 verify gate. A hand-rolled cube is pure multiplication, which
  both compilers execute identically up to standard FP rounding — one less variable in the
  tolerance story below.
- **Determinism.** Every array (mesh, densities, RHS, boundary conditions) is built from fixed,
  file-sourced constants — no RNG anywhere in this project. Given the same GPU/driver/compiler, a
  run is bit-reproducible; ACROSS different GPU architectures, CG's iteration-dependent rounding can
  in principle diverge by a handful of iterations (never changing the CONVERGED answer beyond the
  solver's own tolerance) — the reason every stable output line is a PASS/FAIL verdict with a
  documented margin, never a raw number (the same discipline projects 08.01/24.01 follow).

## How we verify correctness

Four independent checks, because a topology optimizer can be *numerically correct and structurally
uninteresting* (or vice versa) — no single gate would catch every class of bug:

1. **The §5 GPU-vs-CPU twin gate (VERIFY)**: one full inner iteration (CG solve + sensitivity +
   filter) on a small, INTERMEDIATE-density problem (`rho=volfrac` everywhere — deliberately
   exercising the full SIMP `E(rho)` interpolation, unlike the two analytic gates below which use
   `rho=1` exactly), run through the GPU kernels and the sequential CPU oracle
   (`reference_cpu.cpp`). Measured on this project's own reference run: worst relative displacement
   deviation `4.3e-3` (gate: `1e-2`), compliance relative deviation `5.2e-8`, worst filtered-
   sensitivity relative deviation `6.4e-8` (gate for both: `5e-3`). The displacement gate is the
   loosest because a CAPPED, non-fully-converged CG solve (`kCgRelTol=1e-3`) genuinely limits how
   tightly GPU and CPU float-rounding paths can agree at the per-DOF level — the residual-level
   tolerance CLAUDE.md's brief calls for, not solution bit-equality.
2. **The patch test (analytic gate)**: a solid rectangular strip under uniform x-tension (left edge
   `ux=0` everywhere + one corner `uy=0` — a determinate 2D support; consistent nodal loads on the
   right edge for a uniform traction `sigma0`) must reproduce the EXACT closed-form field
   `ux(x,y)=exx*x, uy(x,y)=eyy*y` with `exx=sigma0/E0, eyy=-nu*exx` — because Q4 elements are
   COMPLETE to degree 1 (exact for any linear displacement field) and this project's applied loads
   are the exact consistent-nodal-load equivalent of the assumed uniform traction, the ONLY error
   source is CG's own convergence tolerance (tightened to `1e-7` for this gate). Measured: relative
   error `2.8e-6` against a `1e-3` gate — headroom exists specifically because this is THE textbook
   FEM correctness check, and a real indexing/sign/assembly bug would blow past it by orders of
   magnitude, not fractions of a percent.
3. **The cantilever beam gate (analytic gate)**: a solid cantilever's tip deflection under a point
   shear load must match Euler-Bernoulli theory (`delta = PL^3/3EI`) plus the Timoshenko shear
   correction (`+ PL/(kGA)`, `k=5/6` rectangular) within a documented allowance. Measured: FEA
   deflection sits `0.8%` BELOW the Timoshenko value (gate: `5%`) — this SMALL, HONEST gap is real
   physics, not a tolerance fudge: fully-integrated bilinear Q4 elements are well known in the FEA
   literature to exhibit **shear locking** in bending-dominated problems (the element's bilinear
   displacement field cannot represent pure bending without also generating spurious shear strain
   energy, artificially stiffening the element) — at this project's mesh depth (8 elements through
   the beam's height), the effect is modest but measurable, and this gate's tolerance is set from
   the ACTUAL measurement with headroom, not guessed. README Exercise 3 (coarsen `nely` to 2-3
   elements) makes the locking dramatically worse and is the fastest way to SEE it.
4. **Optimization sanity (MBB / BRACKET gates)**: three checks per case, all measured, none
   assumed — (a) the achieved volume fraction matches the target within `0.02` (bisection's own
   convergence, verified end to end); (b) compliance is monotone non-increasing from iteration 6
   onward within a `0.5%` per-step slack (OC's well-documented early climb away from a
   structurally-poor uniform start is EXPECTED and excluded — measured on MBB: `0.69 J -> 1.37 J`
   peak around iteration 4, then monotonically down to `0.090 J`; the largest near-convergence
   uptick measured anywhere in either run is `~0.07%`, a 7x margin under the gate); (c) the largest
   4-connected component of "solid" elements (`rho>0.5`) holds at least `90%` of all solid material
   — the checkable proxy for "this design is a STRUCTURE, not disconnected dust." Both cases
   measured **100%** — every run so far has converged to a single connected component.

## Where this sits in the real world

- **Production tools**: Altair OptiStruct, Dassault (TOSCA/Simulia), nTopology, and Ansys
  Mechanical's topology optimization module are the commercial-grade descendants of exactly this
  algorithm — unstructured, adaptively-refined FEM meshes (not this project's fixed structured
  grid), manufacturing constraints built directly into the optimization (minimum feature size,
  overhang-angle constraints for 3D printing, draw-direction constraints for casting/milling —
  PRACTICE §1), multi-load-case and fatigue-aware objectives, and industrial-strength linear
  solvers (sparse direct or multigrid-preconditioned iterative, not a from-scratch Jacobi-CG).
- **The "99-line"/"88-line" topopt lineage**: Sigmund's 1999/2001 99-line MATLAB code and
  Andreassen et al.'s 2011 88-line successor are the field's standard TEACHING implementations —
  this project is a direct, GPU-ported descendant, and `compute_KE_hat()`'s cross-check against
  their published `KE[0][0]=0.494505` constant is a literal, checkable link back to that lineage.
- **Level-set methods** (Osher & Sethian's level-set framework, applied to topology optimization by
  Wang, Wang & Guo 2003 and Allaire, Jouve & Toader 2004) are SIMP's main alternative — see "The
  math" above for the comparison; production tools increasingly offer BOTH, since each has domains
  where it is more natural (SIMP: broad exploration from a blank domain; level-set: refining an
  already-reasonable shape with crisp boundaries).
- **What the full research version adds** beyond this teaching core: unstructured/adaptive meshing,
  multigrid or algebraic-multigrid preconditioning (removing the `Emin` conditioning trade this
  project documents above), multiple simultaneous load cases and stress (not just compliance)
  objectives, manufacturing-aware constraints, and — increasingly — machine-learned surrogate
  models that predict a near-optimal topology directly, using classical SIMP only to polish the
  final result (an active [R&D] research direction this project's teaching core is a legitimate
  on-ramp toward, not a substitute for).
