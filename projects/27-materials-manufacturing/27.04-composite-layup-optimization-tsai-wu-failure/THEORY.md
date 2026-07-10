# 27.04 — Composite layup optimization + Tsai-Wu failure envelope sweeps: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**Why composites at all — specific stiffness and strength.** Every gram carried by a flying or
walking robot is a gram fighting gravity and a gram of reflected inertia at every joint that moves
it. The metric that matters for a structural material in these machines is not stiffness or strength
alone but stiffness *per unit mass* (specific stiffness `E/rho`) and strength *per unit mass*
(specific strength `sigma/rho`). Unidirectional carbon-fiber/epoxy has a specific stiffness roughly
3-5x aluminum's and a specific strength roughly 5-10x steel's *along the fiber direction* — but only
along that direction. Perpendicular to the fibers, the same material is weaker and more compliant
than the epoxy resin holding it together. This is not a flaw to engineer around; it is the entire
point.

**Anisotropy as a design variable.** A slab of aluminum has the same stiffness in every direction —
you cannot "aim" its strength. A composite laminate is a stack of thin sheets (**plies**, ~0.1-0.2 mm
each), each one a set of parallel fibers in resin, and each ply can be oriented at whatever angle the
designer chooses. Stack enough plies at enough angles and you have placed material stiffness exactly
where the load needs it and nowhere else — the core idea this whole project teaches. A wing spar that
only ever bends about one axis can be built almost entirely from 0-degree plies (fiber aligned with
the span) and be lighter than any isotropic alternative; a panel that must resist load from an
unknown direction needs a *mix* of angles. Deciding that mix — the **layup** — is a genuine
combinatorial design problem, and this project builds the tool that scores candidates.

**Micromechanics, briefly and honestly.** A ply's own stiffness (`E1` along the fiber, `E2` across
it, `G12` in shear) is itself derivable from the fiber and matrix's individual properties and volume
fraction via **rule-of-mixtures** relations (`E1 approx Vf*Ef + Vm*Em` is the simplest form; real
micromechanics — Halpin-Tsai equations, finite-element unit-cell homogenization — refines this).
This project does **not** implement micromechanics: it starts one level up, from already-known ply
engineering constants (as a designer reading a material datasheet would), and teaches everything from
there upward — classical laminate theory and failure prediction. Say so honestly: the fiber/matrix
interaction itself is out of scope (a `[R&D]`-adjacent micromechanics project's territory).

**The engineering constraint this project answers.** A structural engineer with a known set of loads
(from a load case, a flight envelope, a topology-optimized bracket's reaction forces — SYSTEM_DESIGN
item 4's interface convention: loads arrive as data, from upstream analysis) and a fixed material
system must choose: how many plies, at what angles, in what order? The design space is combinatorial
— with an alphabet of 4 angles and 8 plies (4 independent, by symmetry — the math below explains why)
there are `4^4 = 256` candidates even before considering thickness or asymmetric stacks. **Score
every one of them, fast, and rank by margin against every load direction the part must survive** is
exactly the computation this project maps onto the GPU.

## The math

**Notation, defined once (SI throughout, CLAUDE.md §12):** ply orientation `theta` in degrees in
every data structure, converted to radians at the one point `kernels.cuh` defines
(`kDegToRad`) immediately before any `sinf`/`cosf` call. `theta` is measured counter-clockwise from
the laminate `x`-axis to the ply's fiber (material `1`) axis — the project's one documented angle
convention. Ply properties live in Pa (moduli, strengths) and m (thickness); laminate loads
`(Nx, Ny, Nxy)` are **stress resultants** — force per unit laminate WIDTH, N/m, **not** a stress —
the standard CLT load unit, obtained by integrating stress through the laminate thickness.

### 1. Ply stiffness in material axes

A thin orthotropic ply under **plane stress** (`sigma_3 = tau_13 = tau_23 = 0` — valid because the
laminate is thin compared to its in-plane extent) has a reduced stiffness matrix `Q` relating stress
to strain **in the ply's own material axes** (1 = along fiber, 2 = across fiber):

```
[sigma1]   [Q11 Q12  0 ] [eps1   ]
[sigma2] = [Q12 Q22  0 ] [eps2   ]
[tau12 ]   [0   0   Q66] [gamma12]
```

with (from inverting the plane-stress compliance matrix built from the four independent engineering
constants `E1, E2, G12, nu12` — `nu21` is *not* independent, fixed by Maxwell-Betti reciprocity
`nu21/E2 = nu12/E1`):

```
nu21 = nu12 * E2/E1
Q11 = E1 / (1 - nu12*nu21)      Q22 = E2 / (1 - nu12*nu21)
Q12 = nu12*E2 / (1 - nu12*nu21) = nu21*E1 / (1 - nu12*nu21)
Q66 = G12
```

`Q66 = G12` directly — shear is fully decoupled from normal response *in material axes*; it becomes
coupled only after rotation (next section), which is exactly the mechanism that makes off-axis plies
useful for resisting shear/mixed loads.

### 2. Rotating into laminate axes: `Qbar(theta)`

A ply oriented at angle `theta` in the laminate's `x`-`y` frame has a **transformed** stiffness
`Qbar(theta)` — the standard "transformed reduced stiffness" identities (Jones, *Mechanics of
Composite Materials*), with `c = cos(theta)`, `s = sin(theta)`:

```
Qbar11 = Q11*c^4 + 2*(Q12+2*Q66)*s^2*c^2 + Q22*s^4
Qbar22 = Q11*s^4 + 2*(Q12+2*Q66)*s^2*c^2 + Q22*c^4
Qbar12 = (Q11+Q22-4*Q66)*s^2*c^2 + Q12*(s^4+c^4)
Qbar66 = (Q11+Q22-2*Q12-2*Q66)*s^2*c^2 + Q66*(s^4+c^4)
Qbar16 = (Q11-Q12-2*Q66)*c^3*s - (Q22-Q12-2*Q66)*c*s^3
Qbar26 = (Q11-Q12-2*Q66)*c*s^3 - (Q22-Q12-2*Q66)*c^3*s
```

`Qbar16`/`Qbar26` are the **shear-extension coupling** terms, zero only at `theta = 0` or `90`
degrees — this is the algebraic fact behind "an off-axis ply leaks normal load into shear
deformation," the mechanism a mixed layup exploits to resist load from more than one direction.
At `theta = 0`: `c=1, s=0` kills every `s`-bearing term, so `Qbar == Q` **exactly** —
`GATE_CLT_SANITY` (`src/main.cu`) checks this identity directly against the code that computes it.

### 3. Assembling the laminate: the `A` matrix, and why `B = 0`

Classical laminate theory integrates stress through the laminate thickness to get force and moment
**resultants**. For `N` plies, ply `k` occupying `[z_{k-1}, z_k]` (measured from the laminate
midplane):

```
A_ij = sum_k Qbar_ij(ply k) * (z_k - z_{k-1})              extensional stiffness
B_ij = sum_k Qbar_ij(ply k) * (z_k^2 - z_{k-1}^2) / 2       bending-extension coupling
D_ij = sum_k Qbar_ij(ply k) * (z_k^3 - z_{k-1}^3) / 3       bending stiffness
```

`A` depends on ply THICKNESS, never on `z`'s sign — so `A` depends only on the **multiset** of ply
angles, never their stacking order. This project's `assemble_A()` computes exactly `A` (thickness
times `Qbar`, summed) — never `B` or `D`, because this project only ever applies pure in-plane
(**membrane**) loads and only ever builds **symmetric** laminates. A symmetric stack — ply `k` and
ply `(N-1-k)` share the same angle and thickness, mirrored about the midplane — makes every `z_k^2`
term cancel EXACTLY against its mirror partner (`B_ij` is a sum of `z^2` differences; a symmetric
laminate pairs every positive-`z` ply with an identical negative-`z` ply, and `z^2` does not care
about sign), so `B = 0` **exactly**, not approximately. With `B=0` and no applied moment, the
laminate's curvature is exactly zero and every ply shares one uniform midplane strain — the
simplification that makes this project's whole computation a single 3×3 linear solve instead of a
coupled 6×6 one.

**Solving for strain.** For a symmetric laminate under pure membrane load:

```
[Nx ]   [A11 A12 A16] [eps_x0    ]
[Ny ] = [A12 A22 A26] [eps_y0    ]
[Nxy]   [A16 A26 A66] [gamma_xy0]
```

a symmetric 3×3 system, solved once per (layup, load case) by `solve3x3_sym()` — Cramer's rule,
fully unrolled, in registers (the "33.01-style small batched solve" this project's scope calls for).

### 4. Per-ply stress recovery

Every ply shares `(eps_x0, eps_y0, gamma_xy0)` (no curvature, so no `z`-dependence). Transform into
ply `k`'s own material axes (engineering shear-strain convention throughout — `gamma`, not tensor
strain `gamma/2` — the CLT textbook convention):

```
eps1    = eps_x0*c^2 + eps_y0*s^2 + gamma_xy0*s*c
eps2    = eps_x0*s^2 + eps_y0*c^2 - gamma_xy0*s*c
gamma12 = 2*(eps_y0-eps_x0)*s*c + gamma_xy0*(c^2-s^2)
```

then Hooke's law with the ply's **unrotated** `Q` (each ply "feels" its own local material behavior):

```
sigma1 = Q11*eps1 + Q12*eps2      sigma2 = Q12*eps1 + Q22*eps2      tau12 = Q66*gamma12
```

### 5. The Tsai-Wu failure criterion

A ply fails when its stress state crosses a **quadratic failure surface** in `(sigma1, sigma2,
tau12)` space:

```
F1*sigma1 + F2*sigma2 + F11*sigma1^2 + F22*sigma2^2 + F66*tau12^2 + 2*F12*sigma1*sigma2 = 1
```

with strength parameters derived from the five measured lamina strengths (`Xt, Xc, Yt, Yc, S12` —
all magnitudes):

```
F1  = 1/Xt - 1/Xc         F2  = 1/Yt - 1/Yc          (LINEAR terms)
F11 = 1/(Xt*Xc)           F22 = 1/(Yt*Yc)             F66 = 1/S12^2   (QUADRATIC terms)
F12 = -0.5 * sqrt(F11*F22)                                            (INTERACTION term)
```

**Why an interactive criterion at all — the taxonomy.** The simpler alternatives are **max-stress**
(fail if `|sigma1| > X`, independently of `sigma2`/`tau12`) and **max-strain** (same idea, on
strains) — both treat the three stress components as failing independently, drawing a rectangular
(max-stress) or slightly rotated rectangular (max-strain) failure region. Real composite failure is
not independent: a ply under combined transverse tension AND shear fails at LOWER stress in either
component than either alone would predict (the classic "combined loading is worse than either load
alone" observation) — an interactive criterion like Tsai-Wu captures this coupling with its
cross-term `F12`, at the cost of needing a genuinely experimentally-hard-to-measure biaxial data
point to fix `F12` precisely.

**Why `-1/2` — the interaction term, honestly.** `F12` cannot be derived from the five uniaxial/pure-
shear strengths alone — it is fundamentally a **biaxial** strength property, and a real biaxial test
(pressurized tube, cruciform specimen) is expensive and rare. Tsai & Hahn's widely-used
recommendation `F12 = -0.5*sqrt(F11*F22)` is not derived from first principles; it is chosen because
it keeps the quadratic failure surface a **closed, bounded** ellipsoid across the widest practical
range of strength ratios without requiring the extra test — a pragmatic default, not a law of
physics. Production tools (README "Prior art") let an engineer override it with a measured value
when one exists, and the choice is a documented point of controversy in the composites literature
(some authors argue for `F12=0`, others for values derived from a specific micromechanical model) —
this project uses the standard `-1/2` default and says so plainly rather than presenting it as
uniquely correct.

**A genuinely elegant, checkable consequence of `-1/2`.** Substituting `sigma1 = Xt` into the
1-D (`sigma2=tau12=0`) form of the criterion:
`F11*Xt^2 + F1*Xt = Xt^2/(Xt*Xc) + Xt*(1/Xt - 1/Xc) = Xt/Xc + 1 - Xt/Xc = 1` — **exactly** satisfied,
for ANY `Xt, Xc`. The same substitution with `sigma1 = -Xc` also gives exactly 1. In other words,
`F1` and `F11` are calibrated so that `Xt` and `-Xc` are the **exact algebraic roots** of the 1-D
Tsai-Wu quadratic — this is not a coincidence, it is how `F1`/`F11` are *defined*, and it is the
closed form `GATE_SINGLE_PLY_CLOSED_FORM` checks directly (§How we verify correctness, below).

**A second, less obvious consequence — when does `-1/2` make the criterion ROTATION-invariant?**
Write the quadratic part of the criterion for the case `F11=F22=Fn` (equal-strength-in-every-
direction, the isotropic-degenerate case `GATE_ISOTROPIC_ENVELOPE` builds) using the algebraic
identities `sigma1^2+sigma2^2 = 0.5*(sigma1+sigma2)^2 + 0.5*(sigma1-sigma2)^2` and
`sigma1*sigma2 = 0.25*(sigma1+sigma2)^2 - 0.25*(sigma1-sigma2)^2`:

```
Fn*(sigma1^2+sigma2^2) + 2*F12*sigma1*sigma2 + F66*tau12^2
  = (0.5*Fn+0.5*F12)*(sigma1+sigma2)^2 + (0.5*Fn-0.5*F12)*(sigma1-sigma2)^2 + F66*tau12^2
```

`(sigma1+sigma2)` is the **first stress invariant** (trace) — automatically the same in any rotated
frame. `(sigma1-sigma2)^2 + 4*tau12^2` is (four times) the **second invariant** — also rotation-
invariant. The expression above is a rotation-invariant combination of these two invariants
**exactly when** the coefficient of `(sigma1-sigma2)^2` equals `1/4` the coefficient of `tau12^2`,
i.e. `0.5*Fn - 0.5*F12 = F66/4`, i.e.:

```
F66 = 2*(Fn - F12)
```

With the standard `F12 = -0.5*sqrt(Fn*Fn) = -0.5*Fn`: `F66 = 2*(Fn + 0.5*Fn) = 3*Fn` — the
`GATE_ISOTROPIC_ENVELOPE` condition `S12 = sqrt(Xt*Xc/3) = F0/sqrt(3)` (§How we verify correctness
derives the `S12` form from `F66=3*F11`). This is a genuinely elegant, checkable property: the
SAME `-1/2` normalization that calibrates `F1`/`F11` to the uniaxial strengths is *also* exactly
what a rotation-invariant 2-D quadratic failure form requires, once the shear term is set correctly
— two independently-motivated requirements meeting at the same normalization is not an accident of
this project's test design, it is a real structural property of the Tsai-Wu form.

**The load-scaling factor.** Because every stress component above is **linear** in the applied load
(a chain of a linear solve and linear Hooke's-law evaluations), scaling the load by `lambda` scales
every stress by `lambda` too. Substituting into the failure criterion:

```
a*lambda^2 + b*lambda - 1 = 0
    a = F11*s1^2 + F22*s2^2 + F66*t12^2 + 2*F12*s1*s2      (quadratic terms, at UNIT load scale)
    b = F1*s1 + F2*s2                                        (linear terms, at UNIT load scale)
```

Solved in **closed form** (the quadratic formula, smallest positive root — `solve_lambda()`) for the
exact load factor at which THIS ply reaches its own failure surface. The laminate's first-ply-failure
factor is the **minimum** over its plies — the standard, conservative design criterion (§Where this
sits in the real world discusses progressive/last-ply alternatives).

## The algorithm

Per (layup, load case) or (Nx,Ny grid point):

1. **Decode** the layup index into 8 ply angles (`decode_layup`) — base-4 digit decomposition of a
   `layup_id in [0, 256)` into 4 independent angles, mirrored for symmetry.
2. **Assemble `A`** — loop over 8 plies, rotate `Q` into `Qbar(theta_k)`, accumulate thickness-
   weighted sums. `O(kNPlies)` — 8 trig-pair evaluations, no loop-carried dependency beyond the
   accumulator.
3. **Solve** the 3×3 system for `(eps_x0, eps_y0, gamma_xy0)` — Cramer's rule, `O(1)` (a fixed
   handful of multiply-adds and one divide).
4. **Loop over 8 plies again** — transform strain into material axes, Hooke's law, Tsai-Wu `(a,b)`,
   closed-form `lambda`, track the running minimum.

**Complexity, serial vs. parallel.** One (layup, case) evaluation is `O(kNPlies)` = `O(1)` in the
problem's own terms (8 is a compile-time constant, never scaled up). The SWEEP is
`O(kNLayups * n_cases)` serially — 4,096-4,608 independent evaluations for the committed scenario —
or `O(1)` parallel-depth on a GPU with enough threads to cover them all at once (this project's whole
problem is comfortably smaller than one wave across an RTX 2080 SUPER's 46 SMs). The ENVELOPE sweep
is `O(kEnvGridN^2)` = 16,384 points, same story. There is, notably, **no loop over TIME anywhere in
this project** — every evaluation is a single pass through a short, fixed-depth function chain, the
key numerical-conditioning difference from this repo's integrator-heavy projects (§Numerical
considerations expands this).

**Why stacking sequence is enumerated but does not change the score.** `A_ij` is a SUM over plies —
addition is commutative, so reordering the sum's terms (i.e. reordering the plies) never changes
`A`. The catalog bullet's "4^4 = 256 stack sequences" is honored literally (every ordered sequence is
scored), but the reader should expect, and this project's own measured output demonstrates, exact
ties among every permutation of the same angle multiset (measured: 24 = `4!` layups tied for the
`MIXED`-set win, exactly the permutations of "one ply each of 0/45/-45/90"). This is a genuine
finding about membrane-only laminate response, not a defect in the sweep design — stacking sequence
DOES matter for bending stiffness `D` and for secondary effects (delamination initiation, thermal
warpage) this project's pure-membrane scope never touches (README §Limitations).

## The GPU mapping

```
layup_sweep_kernel: one thread per (layup_id, case_id) PAIR
  g = blockIdx.x*blockDim.x + threadIdx.x
  layup_id = g / n_cases    (slowest-varying)
  case_id  = g % n_cases    (fastest-varying)
  -> decode_layup (registers) -> laminate_failure_factor (registers) -> ONE coalesced write

envelope_kernel: one thread per (Nx,Ny) grid point, FIXED layup
  g = blockIdx.x*blockDim.x + threadIdx.x
  i = g / kEnvGridN  (row -> Ny)     j = g % kEnvGridN  (column -> Nx)
  -> envelope_grid_point -> laminate_failure_factor (registers) -> ONE coalesced write
```

**Memory hierarchy.** `Lamina`/`AngleAlphabet`/`Layup8` arrive as **kernel parameters** (passed by
value — a few dozen bytes, broadcast to every thread at effectively zero cost, the same
"parameter-memory uniform read" spot on 08.01's memory-spectrum note as `u_nom[t]`). The sweep
kernel's load-case array is a small **global-memory** pointer read (at most 16 `LoadCase` = 192
bytes total, served from the L2/read-only cache path after first touch by any thread in a block).
Every thread's entire per-problem state — the 8-float `angles[]` array, the running `worst_lambda`
scalar, all the intermediate `Q`/`Qbar`/`A`/strain/stress values — lives in **registers**: nothing
here is large enough to need shared memory, and no thread ever reads another thread's data (the
defining property of a design-space sweep — by construction, not by discipline). No atomics, no
divergence beyond the standard ragged-tail guard.

**Occupancy, honestly.** This project's entire problem (at most a few thousand sweep threads, 16,384
envelope threads) is **far smaller** than what saturates a modern GPU's thousands of concurrent
threads — unlike 08.01's 4,096-rollout MPPI controller or 18.01's multi-thousand-gait sweep, this
project never needed a grid-size cap or careful occupancy tuning; `ceil(total/256)` blocks cover the
whole problem in a small handful of waves. The teaching point here is the **mapping** — how a
materials design search becomes independent GPU threads — not raw throughput; measured kernel time
for the entire sweep is under 2 ms on the reference machine, and the interesting number is not "how
fast" but "how much design space explored per interactive iteration" (README §System context).

**No CUDA library is linked.** `solve3x3_sym` is a fully unrolled 3×3 Cramer's-rule evaluation —
6 stored matrix entries, a handful of multiply-adds, one divide. Dispatching this to cuSOLVER (built
for batches of much larger systems, with real launch/setup overhead per call) would teach nothing a
by-hand derivation does not already teach at this size, and would cost more than it saves — the same
judgment call CLAUDE.md §5 asks every project to make explicitly (`CMakeLists.txt`'s comment states
this decision at the point a reader would look for it).

## Numerical considerations

- **Precision: FP32 throughout**, matching the repo default. Every physics function lives in
  `kernels.cuh` as a `__host__ __device__` inline (the 18.01 pattern) — the GPU kernel and the CPU
  oracle run the textually IDENTICAL source, so the only possible divergence between them is
  `sinf`/`cosf`/`sqrtf`'s independently-rounded host vs. device implementations, propagated through
  a SHORT, fixed-depth chain (assemble `A` -> solve 3x3 -> per-ply stress -> Tsai-Wu quadratic).
- **Why this project's VERIFY tolerance is far tighter than 08.01's or 18.01's.** Those projects
  chain hundreds to thousands of timesteps, so tiny per-step rounding differences accumulate
  (08.01 measures ~1e-7 relative divergence after 50 RK4 steps; 18.01 measures ~1.4e-6 m after 8,000
  steps). This project has **no time loop at all** — one pass through the chain above, so rounding
  never compounds. Measured worst-case GPU-vs-CPU relative deviation across the FULL sweep+envelope
  (37,376 points, no spot-sampling): `1.76e-6` — the VERIFY tolerance (`1e-3`) therefore carries
  roughly **500x** headroom, a wide margin chosen so the gate is robust to a different GPU
  architecture's independently-rounded transcendentals while still catching any real indexing or
  formula bug instantly (those show up at order 1, not 1e-6).
- **Conditioning: Pa vs. GPa, an honest scale note.** `Q`/`Qbar` entries are ~1e9-1e11 Pa; `A` matrix
  entries (`Qbar * t_ply`, `t_ply ~ 1e-4` m) land around 1e6-1e7 (N/m); load resultants are ~1e5-1e6
  N/m. The 3x3 system's condition number is set by the ratio `E1/E2` (here ~13.5) — comfortably
  well-conditioned; Cramer's rule (division by one 3x3 determinant) is numerically fine at this
  scale. A laminate mixing wildly different ply materials (a hybrid layup, out of this project's
  scope) could ill-condition `A` far more severely — worth knowing before reaching for Cramer's rule
  on a bigger or worse-conditioned system (33.01's territory for a general treatment).
- **The `kEnvFactorClamp` near-zero-load singularity.** `laminate_failure_factor` genuinely diverges
  to `+infinity` as the applied load approaches `(0,0,0)` (zero load cannot fail anything) —
  `envelope_factor_at()` clamps both the "no load" case (`|N| < kEnvFactorMinLoad`) and the general
  large-factor case to `kEnvFactorClamp = 5.0` for storage/display. Because `kEnvFactorClamp >> 1`,
  this clamp never moves the interesting factor=1 boundary — only flattens the field deep inside the
  safe region, and every file that touches the field (kernel, CPU oracle, PGM writer, contour
  extractor) applies the identical documented clamp so it cannot itself become a source of GPU-vs-CPU
  disagreement.
- **No angle-wrapping, no quaternions, no stiff ODEs, no ill-conditioned Jacobians near a
  singularity** — none of CLAUDE.md §12's usual robotics numerical hazards apply here: every angle in
  this project is a fixed design parameter read once per evaluation, never integrated or accumulated,
  so there is no periodicity/wrap concern anywhere in the codebase.

## How we verify correctness

Five independent checks, because a materials analysis can be *numerically self-consistent and still
physically wrong* (an implementation bug that both the GPU kernel and CPU oracle share equally would
sail through a GPU-vs-CPU comparison):

1. **The §5 GPU-vs-CPU gate (`VERIFY`).** Full-array recomputation (not spot-sampled — see §Numerical
   considerations) of both sweep case sets and both envelopes by the CPU oracle, `kernels.cuh`'s
   shared `HD` functions called identically from both paths. Catches indexing, layout, and launch
   bugs; measured worst-case `1.76e-6` relative deviation, tolerance `1e-3`.
2. **`GATE_SINGLE_PLY_CLOSED_FORM`.** A one-ply "laminate" at `theta=0` under pure `+-Nx` — for a
   SINGLE ply, `A` reduces to `Q*t` exactly and, because there is only one material through the
   thickness, the ply's own stress is `Nx/t` **algebraically**, no solve needed. Combined with the
   §the-math result that `Xt`/`-Xc` are exact roots of the 1-D Tsai-Wu quadratic, the failure load is
   `Xt*t` (tension) / `Xc*t` (compression) **exactly**. Measured relative error: `~1e-7` (pure FP32
   rounding through the quadratic-formula chain) against a tolerance of `1e-3`.
3. **`GATE_ISOTROPIC_ENVELOPE`.** The isotropic-degenerate material (§the-math derives both the
   elastic condition `Q12+2*Q66=Q11` and the strength condition `F66=3*F11` this gate is built from)
   must give the SAME failure strength regardless of a rotated uniaxial load's orientation — see
   §the-math's "genuinely elegant" derivation for exactly what is being checked and why the naive
   "sweep `(Nx,Ny)` with `Nxy=0`" version of this test is subtly WRONG (the gate's own comment in
   `main.cu` tells this story, including the bug this project's own development caught). Measured
   spread: `~1e-7` relative.
4. **`GATE_CLT_SANITY`.** The `[0/90/0/90]s` cross-ply laminate must give `A11=A22` **exactly** (equal
   counts of `0`- and `90`-degree plies swap `Q11 <-> Q22` between the two sums — a symmetry argument
   independent of any specific numeric material values) and `Qbar(theta=0)` must reproduce `Q`
   exactly (the `c=1,s=0` algebraic identity §the-math states). Measured relative deviation:
   `~1e-7` (A11 vs A22) and exactly `0` (Qbar(0) vs Q, since `cosf(0.0f)` and `sinf(0.0f)` are exact
   in FP32).
5. **`GATE_LOAD_HOMOGENEITY`.** Because stress is linear in load, `factor(k*N) = factor(N)/k` for any
   `k > 0` — checked on 4 representative `(layup, case, k)` samples spanning both sweep sets.
   Measured worst relative deviation: `0` on the reference machine (the sample chain happens to hit
   FP32-exact cases; not claimed to be exact in general, only within the documented tolerance).

Together these five checks exercise every piece of the pipeline (the linear solve, the rotation
formulas, the quadratic failure solve, the layup decode) against an independently-derivable ground
truth, not just against each other.

## Where this sits in the real world

- **HyperSizer, ANSYS Composite PrepPost, Altair Hyperlaminate** (README "Prior art") implement this
  exact CLT + Tsai-Wu (or Hashin, Puck, LaRC — see below) machinery at production scale: thousands of
  load cases per part, coupled to a full finite-element mesh (so strain varies point-to-point instead
  of this project's one uniform membrane strain), manufacturing constraints (minimum ply-drop
  spacing, balanced/symmetric-stack rules, maximum contiguous same-angle plies to limit
  micro-cracking), and often multiple competing failure criteria evaluated simultaneously.
- **Beyond first-ply failure.** Production analyses commonly run a **progressive failure** or
  **last-ply-failure** model: when a ply's matrix fails (a `sigma2`/`tau12`-dominated failure mode),
  its transverse and shear stiffness are knocked down (not necessarily to zero) and the laminate is
  RE-SOLVED under the same load — repeating until the fiber-dominated plies also fail. This can show
  meaningfully more reserve strength than first-ply-failure predicts, at the cost of a much more
  involved (iterative, mode-dependent) analysis. README Exercise 4 sketches a minimal version.
- **Beyond Tsai-Wu.** Modern composite design increasingly uses criteria that separate FIBER failure
  from MATRIX failure explicitly — **Hashin's criterion** (four distinct sub-criteria: tensile/
  compressive fiber failure, tensile/compressive matrix failure) and **Puck's criterion** / the
  **LaRC03/04** family (which additionally models the fracture-plane angle of matrix cracking) are
  the current state of the art for composite failure prediction, specifically because a single
  interactive quadratic like Tsai-Wu cannot distinguish "this ply failed because the fiber broke" from
  "this ply failed because the matrix cracked" — a distinction that matters enormously for what a
  progressive-failure model should do next.
- **What a full progressive-failure / multi-criterion tool needs beyond this project:** an FEA mesh
  (not one uniform-strain patch), a stiffness-degradation model, an iterative nonlinear solve per
  load step, and typically several allowable failure criteria evaluated together with engineering
  judgment about which one governs — a genuinely larger piece of software than this project's
  closed-form sweep, and the reason composite structural analysis remains a specialist discipline
  (PRACTICE §4) rather than a fully-automated one.
