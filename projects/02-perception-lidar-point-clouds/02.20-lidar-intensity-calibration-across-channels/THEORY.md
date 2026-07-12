# 02.20 — LiDAR intensity calibration across channels: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### What LiDAR intensity physically is

A time-of-flight LiDAR beam leaves the sensor as a short pulse of laser light, travels to a surface,
scatters, and a fraction of that scattered light returns to the sensor's receiver — an avalanche
photodiode (APD) or similar detector, chosen because it multiplies a weak optical signal internally
(the "avalanche gain") before it ever reaches an amplifier, which is what makes single-digit-photon
returns from tens of meters away detectable at all. The *range* comes from the pulse's round-trip
time; the *intensity* comes from the received optical power — how many photons came back, converted
to an analog voltage by the APD, amplified, and digitized into whatever integer or float value ends
up in the point cloud's `intensity` field.

Every stage of that chain has a gain that can differ, beam to beam:

- **The laser diode's output power.** Sixteen separate laser diodes (one per beam) age differently and
  were never perfectly matched at manufacturing time.
- **The APD's avalanche gain.** APD gain depends on the reverse-bias voltage across the diode and on
  temperature; small manufacturing and thermal-path differences across sixteen physically separate
  APDs on a spinning head produce small, persistent gain differences.
- **The receive optics' alignment.** Each beam's return light must land precisely on its own APD's
  active area; sub-millimeter alignment differences from assembly attenuate some beams more than
  others, permanently.
- **AGC (automatic gain control) — an important honesty note.** Some real LiDARs apply *per-return*
  automatic gain control to keep the analog signal in range before digitizing — which, if present,
  changes what "intensity" even means (a return's reported intensity partly reflects the AGC setting
  chosen for THAT return, not a fixed transfer function). This project's forward model does **not**
  include AGC: it models a fixed per-channel gain, the simpler and still very real failure mode
  (aging/alignment drift, not per-shot AGC state). A real deployment must first confirm its own
  sensor's intensity output is AGC-free (or account for AGC) before trusting ANY per-channel
  calibration, this project's or a vendor's — see PRACTICE.md §1.

The net effect: the SAME physical surface, at the SAME range and angle, produces a systematically
different reported intensity depending on which of the sixteen beams happened to see it. That
per-channel multiplicative factor is this project's unknown, `g[ch]`.

### The robotics task and its engineering constraints

Intensity calibration is not a per-frame, real-time computation — it is closer to a periodic
maintenance task (PRACTICE.md §1 details the cadence). The engineering constraints that matter here
are different from a perception project's:

- **Data budget, not compute budget.** The bottleneck is not "can I compute this fast enough for the
  next frame" — it is "do I have ENOUGH overlapping observations." A single static scan may not give
  every channel pair enough shared voxels (this project's own build hit this directly — see
  "Numerical considerations" below); a real deployment accumulates over a drive, or several.
- **Noise floor at grazing incidence.** A surface seen at a shallow angle returns very little light
  (Lambert's law, derived below) — this project's own ground-plane returns get close enough to the
  sensor's noise floor that some register as exactly zero intensity (see `data/README.md`), a real,
  physically honest phenomenon this project's numerical guards (below) must handle, not paper over.
- **Thermal drift.** APD gain is temperature-dependent; a calibration performed on a cold morning and
  applied at midday introduces its own small error — one reason real calibration tables are
  periodically refreshed rather than computed once forever (PRACTICE.md §1).

## The math

**Notation.** SI units throughout; sensor frame, sensor at the origin (x-forward/y-left/z-up,
`SYSTEM_DESIGN.md` §3.2). For a point `p ∈ ℝ³` (meters), range `r = ||p||₂` (meters), and a struck
surface with unit outward normal `n`, the ray's unit direction is `d = p / r` and the **incidence
angle** `θ` is the angle between `-d` and `n`; `cos(θ) = |d · n|`.

### The forward model

```
I = g[ch] · R · f(r) · cos(θ) + ε
```

- `I` — the measured intensity (unitless, ≥ 0, the point cloud's `intensity` field).
- `g[ch]` — the **unknown** this project recovers: channel `ch`'s scalar gain (unitless, this
  project's true values span 0.60–1.40, `kernels.cuh`'s ground-truth comment).
- `R` — the struck surface's Lambertian reflectivity (unitless, `(0, 1]`), unknown to the algorithm
  (never estimated; canceled algebraically, below).
- `f(r)` — the **range-falloff** term, derived next.
- `cos(θ)` — the Lambertian incidence-angle term, derived next.
- `ε` — sensor noise (this project's generator: small multiplicative + additive Gaussian, documented
  in `scripts/make_synthetic.py`).

### Deriving `f(r)`: the radar-equation shape, plateaued near range

For a solid surface that fills the ENTIRE beam (true whenever the beam's footprint is smaller than
the surface, the common case), the classic single-scatter radar-equation argument gives received
power falling as `1/r²`: the outgoing beam's power density falls as `1/r²` over the outbound trip
(inverse-square spreading), the surface re-radiates a fraction of what it received (Lambert's law,
below — a PROPERTY of the surface, not of range), and the RECEIVER'S APERTURE subtends a solid angle
that *also* shrinks as `1/r²` on the return trip. Two factors of `1/r²` would give `1/r⁴` (the
classic *radar* equation for a POINT target) — but a solid, beam-filling surface is not a point
target: it re-radiates from an area that itself GROWS as `r²` (the beam footprint widens with range),
which exactly cancels one of the two `1/r²` factors, leaving the familiar **`1/r²`** falloff for an
extended Lambertian surface — the same argument, and the same "range-compensated intensity as a
diagnostic" convention, project 02.18's docstring cites for why a raw intensity channel is usable at
all without first inverting range.

Near the sensor, this breaks down: the receive optics have a finite focal working range, and a target
too close is out of focus — a real receiver's response FLATTENS rather than diverging as `r → 0` (the
"near-range defocus plateau"). This project's model captures both regimes in one continuous, if not
differentiable, curve:

```
f(r) = ( r_plateau / max(r, r_plateau) )²
```

`f(r_plateau) = 1` by construction (a convenient reference scale, not a physical unit); `f` is
constant for `r ≤ r_plateau` and falls as `1/r²` beyond it. `kernels.cuh`'s `kRangePlateauM = 4.0` m.
**Honesty note** (restated from README "Limitations"): this project's committed scene never puts a
surface closer than ~8 m, so the demo never actually SAMPLES the plateau regime — the model teaches
the shape; the data teaches the `1/r²` regime alone.

### Deriving `cos(θ)`: Lambert's cosine law

A Lambertian (ideal diffuse) surface's radiance is the SAME in every viewing direction, but the POWER
it intercepts from an incoming beam of fixed cross-section is proportional to the projected area the
beam actually illuminates — which shrinks as `cos(θ)` as the beam becomes more oblique to the surface
normal (the same `cos(θ)` that makes noon sunlight feel more intense than sunset light striking the
same patch of ground at a grazing angle). This project floors `cos(θ)` at 0.02 in the generator (a
real receiver never reads exactly zero) and reports the physical consequence honestly: grazing-angle
ground returns are dim enough to brush this project's noise floor (`data/README.md`).

### Why channels differ — restated as the unknown

Given the SAME forward model applies to every channel, a shared voxel `v` observed by channels
`c₁, c₂, …` gives, for each channel present,

```
I_{v,c} = g[c] · R_v · f(r_{v,c}) · cos(θ_{v,c}) + ε
```

Dividing out the KNOWN part (`f`, `cos θ`, both computable from the point's own geometry — see "The
algorithm") and taking a log turns the remaining UNKNOWN product `g[c] · R_v` into a **sum**:

```
y_{v,c} := log( I_{v,c} / (f(r_{v,c})·cos(θ_{v,c})) )  ≈  log(g[c]) + log(R_v) + noise
```

This is the single equation the rest of this document, and the whole codebase, builds on.
`kernels.cuh`'s `corrected_log_intensity()` computes `y_{v,c}` for one point; the least-squares
system below turns MANY such equations, across many voxels, into 16 numbers.

### The least-squares gain model

Let `x_c = log(g[c])` (16 unknowns) and `μ_v = log(R_v)` (one unknown per shared voxel — but note:
**never actually solved for**, see below). For a voxel `v` with channel set `C_v` (`|C_v| = k_v ≥ 2`)
and per-(voxel,channel) MEAN observation `y_{v,c}` (averaging over however many points of channel `c`
landed in voxel `v`), the joint least-squares objective is

```
J(x, μ) = Σ_v Σ_{c ∈ C_v} ( y_{v,c} − x_c − μ_v )²
```

**Profiling out `μ_v` analytically.** For FIXED `x`, the `μ_v` that minimizes voxel `v`'s inner sum is
its own least-squares estimate: `μ_v* = ȳ_v − x̄_v`, where `ȳ_v` is the mean of `y_{v,c}` over `C_v`
and `x̄_v` is the mean of `x_c` over the SAME `C_v`. Substituting back:

```
J(x) = Σ_v Σ_{c ∈ C_v} ( r_{v,c} − (x_c − x̄_v) )²,      r_{v,c} := y_{v,c} − ȳ_v
```

`r_{v,c}` is computable from the DATA ALONE (no unknowns) — it is a per-voxel, already-mean-zero
residual (`Σ_{c∈C_v} r_{v,c} = 0` by construction). This is exactly the standard **fixed-effects
elimination** trick from two-way ANOVA (row effects = channels, column effects = voxels): eliminate
the nuisance column effect algebraically, leaving a pure function of the row effects. The result is
never "biased" by a wrong guess at `R_v` — it never *needs* one.

**The normal equations are a graph Laplacian.** Minimizing `J(x)` gives, per voxel, a **centering
projector** contribution `P_v = I_{k_v} − (1/k_v)·𝟙𝟙ᵀ` (the `k_v × k_v` matrix that subtracts the
mean), embedded at rows/columns `C_v` of a `16 × 16` matrix, summed over every shared voxel:

```
A = Σ_v P_v   (embedded),      b = Σ_v r_v   (embedded)
solve:  A x = b
```

`A`'s OFF-DIAGONAL nonzero structure is precisely "which channel pairs co-occur in a shared voxel" —
a weighted adjacency matrix. This makes `A` a (voxel-weighted) **graph Laplacian** over the 16
channels, and standard graph-Laplacian theory says: `A`'s null space dimension equals the number of
CONNECTED COMPONENTS of that graph. One connected graph → one null direction (the global gauge, next);
a channel with NO edges at all (never shared a voxel) has its own trivial 1-dimensional null
component — `A`'s diagonal entry for that channel is EXACTLY zero (kernels.cuh SECTION 5 states this
precisely; `kernels.cu`'s `solve_channel_gains` generalizes it to ALSO catch a channel stranded in a
smaller, separate cluster — "Numerical considerations" below explains why that generalization was
necessary in practice, not just in principle).

**Gauge freedom.** Adding a constant to every `x_c` and subtracting it from every `μ_v` leaves every
residual — hence `J` — unchanged: only *relative* gains are observable from intensity ratios alone,
never an absolute scale (the same "one global scale unobservable" honesty project 01.09/01.17 name for
their own decompositions). This project fixes the gauge by convention: `mean(x_c) = 0` over the
OBSERVABLE channels. Implementation-wise (`kernels.cu` SECTION 8): add a small ridge term
`λ/m · 𝟙𝟙ᵀ` to the reduced `m × m` system (`m` = observable channel count). Because `𝟙` is EXACTLY in
`A`'s null space (`P_v · 𝟙_{k_v} = 0` for every voxel, by construction of a centering projector) and
`b` is EXACTLY orthogonal to `𝟙` (every `r_v` sums to zero over its own support), this ridge term
provably: (a) leaves the solution's non-gauge component untouched (the added rank-1 term lives
entirely in the direction `A` was singular along, so it cannot perturb any other eigen-direction), and
(b) pins `mean(x) = 0` exactly. This is not an approximation — in exact arithmetic the ridge-regularized
solve returns the UNIQUE minimum-norm least-squares solution with that gauge convention, for any
`λ > 0`; `λ` only matters for FLOATING-POINT conditioning (`kGaugeLambda = 1e-2`, chosen to sit
comfortably above `A`'s round-off floor without dominating its real eigenvalues, which are `O(1)`–`O(10)`
at this project's scale).

## The algorithm

1. **Per-point forward-model inversion** (`point_features_kernel`/`_cpu`): for each point, classify
   which known plane it struck (`classify_normal_family`, purely from its own `xyz` against
   `kernels.cuh`'s scene constants — never from a hidden label), compute `r`, `f(r)`, `cos(θ)`, and
   `y = corrected_log_intensity(I, f(r), cos θ)` (with numerical floors — see below). Serial cost:
   `O(n)`; embarrassingly parallel (every point independent).
2. **Voxel binning + per-(voxel,channel) accumulation** (`bin_accumulate_kernel`/`_cpu`): assign each
   point a voxel index (round-half-up binning — "Numerical considerations" derives why, not
   `voxel_coord`'s naive floor), then accumulate `Σy` and a count per `(voxel, channel)` pair. Serial
   cost: `O(n)`; the accumulation step is a scatter-reduce (parallel, but destination collisions
   require atomics on the GPU path — see "The GPU mapping").
3. **Least-squares assembly** (`assemble_ls_kernel`/`_cpu`): for every voxel with `k_v ≥ 2` channels
   present (the "shared" subset — usually a small fraction of occupied voxels), compute its mean
   `ȳ_v`, its residuals `r_{v,c}`, and accumulate its `P_v`/`r_v` contribution into the global `A`/`b`
   (`channel_ls_accumulate`, kernels.cuh SECTION 5). Serial cost: `O(V·k̄²)` where `V` is the occupied-
   voxel count and `k̄` the mean channels-per-voxel (`k̄ ≤ 16` always, typically 2–4) — negligible.
4. **The shared solve** (`solve_channel_gains`, kernels.cu SECTION 8, called ONCE): connected-
   components analysis (union-find, `O(16²)` at this problem's fixed size) to find the DOMINANT
   observation cluster, then Gaussian elimination with partial pivoting over that cluster's reduced,
   gauge-ridged system (`O(m³)`, `m ≤ 16` — microseconds).
5. **Gain application** (`apply_gain_kernel`/`_cpu`): divide the recovered gain back out of every
   point's range/incidence-compensated intensity — a pure map, `O(n)`.

Complexity end to end: `O(n + V·k̄² + m³)` — for this project's scale (`n ~ 10³`, `V ~ 10²`, `m ≤ 16`),
every stage after the initial per-point pass is essentially free; the per-point pass dominates and is
the only stage a real deployment's larger `n` (a full sweep: 10⁵–10⁶ points) would need to keep GPU-
accelerated at scale — steps 3–5 stay cheap even there, since `V` and `m` do not grow with raw point
count nearly as fast as `n` does.

## The GPU mapping

- **Stage 1 (map):** one thread per point, `blockDim = 256` (a warp multiple; good occupancy on
  sm_75–sm_89), `gridDim = ceil(n/256)`. Global memory only — `xyz`/`intensity` read once each,
  coalesced (adjacent threads read adjacent point records); no shared memory, because no data is
  reused across threads (every point's computation is fully local).
- **Stage 2 (scatter-reduce):** same launch shape; each thread's destination
  (`sum_log[voxel*16+channel]`) depends on its OWN point's data, so different threads' writes
  legitimately collide — `atomicAdd` is required (the same trade project 01.09's `radial_bin_kernel`
  makes: correctness over determinism of summation ORDER, verified against an independently-ordered
  CPU oracle instead of demanding bit-identical results — see "How we verify correctness"). No shared-
  memory staging: the destination array (`numVoxels × 16` floats) is far too large and too sparsely
  hit per block to benefit from a block-local partial-histogram stage at this project's scale (that
  optimization is real at MUCH higher point counts — README Exercise territory, not this project's
  core lesson).
- **Stage 3 (scatter-reduce, coarser grain):** one thread per VOXEL, not per point — `gridDim =
  ceil(numVoxels/256)`. Each qualifying thread's `k_v ≤ 16` channel ids and means live in **registers**
  (`chans[16]`/`y[16]`, small fixed-size local arrays — no shared memory needed: there is nothing to
  reduce WITHIN a voxel across threads, only the small number of ATOMIC writes each voxel makes into
  the tiny (256+16-float) global `A`/`b` destination, contended only by the sub-percent of threads
  that actually qualify (`k_v ≥ 2`).
- **Stage 4 (the solve): NOT on the GPU, by design.** `kernels.cuh` SECTION 8 states the reasoning in
  full: a `≤ 16 × 16` dense solve has no meaningful GPU mapping — the kernel launch overhead alone
  (microseconds) would dwarf the `O(16³)` arithmetic (nanoseconds). Project 33.01 (batched small-
  matrix linear algebra) is where this pattern DOES earn a GPU kernel — at a scale where THOUSANDS of
  independent small systems amortize the launch cost, which this project's single system never
  approaches.
- **Stage 5 (map):** identical shape to stage 1.

No CUDA library calls are used anywhere in this project (no cuBLAS/cuSOLVER/Thrust) — every kernel is
hand-rolled, and the one place a library WOULD classically appear (a batched or single dense solve) is
exactly the place this project argues a library (or even a GPU kernel at all) is not the right tool at
this scale.

## Numerical considerations

- **Precision.** All per-point/per-voxel arithmetic is FP32 on both paths; the least-squares solve
  runs internally in FP64 (`kernels.cu` SECTION 8) — the ridge term's magnitude (`λ/m ~ 10⁻³`) sits
  close enough to `A`'s real eigenvalues (`O(1)`–`O(10)`) that FP32 elimination would visibly round it
  away over several pivot steps; FP64 keeps that safely separated from machine epsilon.
- **Log of noisy near-zero intensities — the floor/guard.** Grazing-incidence ground returns can be
  extremely dim (`data/README.md`: some clamp to exactly 0.0 after noise); dividing by a small
  `f(r)·cos(θ)` and taking a log of a near-zero or exactly-zero quantity would produce `-∞`/NaN.
  `kernels.cuh`'s `corrected_log_intensity()` floors BOTH the divisor (`kDenomFloor = 1e-4`) and the
  quotient (`kIntensityFloor = 1e-4`) before `logf()` — a small, honest bias on the dimmest points
  (this project's ground points, which is exactly why they never end up in a SHARED voxel that drives
  the gain solve — the floor protects the pipeline from crashing on data that was never going to be
  useful for calibration anyway, not from silently corrupting the answer).
- **Determinism and the atomic-order tolerance.** `atomicAdd` reorders float summation
  nondeterministically across GPU runs (thread-scheduling dependent); the CPU oracle sums in FIXED,
  sequential point-index order, in DOUBLE precision — "give the oracle better precision", the same
  asymmetry projects 01.09/02.01 use for their own atomic reductions. Measured on the reference
  machine: `sum_log` max `|gpu-cpu|` ≈ 1e-6, the assembled `A`/`b` max `|gpu-cpu|` ≈ 3e-6 — both a
  tiny fraction of the 3e-3 tolerance margined in, comfortable headroom for a different GPU or a
  different run's atomic interleaving.
- **A fixed-origin voxel grid is more fragile than it looks — a real bug this project's build caught.**
  `voxel_coord()` originally matched project 02.01's plain `floor(v/leaf)`, which places a bin BOUNDARY
  at every multiple of `leaf`, including exactly zero. This project's beam fan is symmetric about
  elevation 0 (16 evenly spaced elevations, no exact-center beam) — the two beams straddling elevation
  0 sit on OPPOSITE sides of that zero boundary at EVERY azimuth column, for EVERY leaf size, because a
  linear function through the origin never moves its own zero-crossing. The measured consequence
  (before the fix): the 16-channel graph split cleanly into TWO disconnected 8-channel halves,
  regardless of leaf size swept from 0.2 m to 1.0 m. Switching to round-half-up binning
  (`floor(v/leaf + 0.5)`, i.e. bin CENTERS at multiples of `leaf` rather than bin EDGES) moved the
  boundary away from the physically special coordinate zero and reconnected the whole graph at
  `leaf = 0.5` m. **The general lesson, not just this project's fix:** a synthetic (or real) scene
  whose structure happens to align with a fixed analysis grid's boundaries can produce a
  connectivity failure a naive leaf-size sweep alone will not reveal, because the SAME structural
  misalignment persists at every leaf size tested. A real deployment should treat unexpectedly poor
  channel-graph connectivity as a signal to check for exactly this kind of aliasing, not just "not
  enough data."
- **Connected components, not just nonzero-degree — a second, related bug this project's OWN gates
  caught.** `README`'s `multi_material_robustness` gate initially compared error over "every channel
  with `A_cc > 0`" on a restricted (single-material) subgraph and produced a numerically meaningless
  (effectively infinite) error for one channel: that channel's diagonal was nonzero (it DID share a
  voxel with something), but its only edge led into a SMALL cluster disconnected from the rest — a
  SECOND, unresolved gauge freedom the single global ridge term does not fix (the ridge only pins the
  mean over channels that are ALL mutually reachable; a second component has its OWN unconstrained
  relative offset, which shows up as a tiny — not exactly zero — eigenvalue and an amplified, not
  merely undefined, solution). The fix (kernels.cu SECTION 8): explicit connected-components analysis
  (union-find over `A`'s nonzero off-diagonal structure), solving ONLY the largest component and
  flagging every other channel — genuinely isolated ones AND ones stranded in a smaller cluster alike
  — unobservable. This is the SAME graph-connectivity idea "The math" states in principle, discovered
  to need this exact generalization empirically, while building this project's own restricted-cohort
  gate.
- **Fair comparison over a shrinking channel set.** Once observability is per-solve (different solves
  can flag DIFFERENT subsets of the 16 channels observable), comparing "worst error over 16 channels"
  against "worst error over 6" is not apples-to-apples — a real trap this project's `multi_material_
  robustness` gate hit directly (see `main.cu`'s `compare_solves_over_intersection`): both solves are
  gauge-realigned using the MEAN OVER THEIR INTERSECTION ONLY before comparing, and the intersection
  size is itself reported and floor-checked.

## How we verify correctness

Two independent tiers, per this repo's twin-independence ruling (`reference_cpu.cpp`'s file header):

1. **GPU-vs-CPU twins, every pipeline stage** (the VERIFY block): per-point features (tight FP32
   tolerance — the shared formula should agree almost to the bit), voxel indices (EXACT — integer
   arithmetic), per-(voxel,channel) accumulation and the assembled `A`/`b` (atomic-order tolerance,
   measured then margined, above), and the final gain application (tight tolerance). This proves the
   GPU kernels are FAITHFUL to the same computation the CPU oracle performs — it says nothing about
   whether that computation is CORRECT.
2. **Independent gates against ground truth never seen by the algorithm** (`gain_recovery`,
   `consistency_improvement`, `multi_material_robustness`, `unobservable_channel`): these compare the
   pipeline's OUTPUT against the generator's TRUE per-channel gains, a from-scratch cross-channel
   consistency metric, a restricted-cohort re-solve, and a deliberately engineered observability
   failure — none of them route through the shared 16×16 solve's own internal correctness, because
   (per the SECTION-8 precedent this project follows from project 01.09) a SHARED function is blind to
   bugs inside itself. This tier is what actually proves the algorithm is right.

Tolerances throughout are set by MEASURING the actual value on the reference machine (RTX 2080 SUPER,
sm_75) and adding margin — quoted at each constant's definition in `main.cu`, never set AT the
measured value (the same discipline projects 01.09/02.18 state explicitly for their own tolerances).

## Where this sits in the real world

Real LiDAR vendors (Velodyne, Ouster, Hesai, and others) measure per-channel intensity response at the
factory, against calibrated reflectance targets under controlled illumination, and ship the resulting
correction table baked into firmware — the intensity a driver reports has usually already had SOME
per-channel correction applied. What this project teaches is the FIELD-VERIFICATION and self-
calibration technique an integrator or fleet operator uses without a lab: confirm the factory table
still holds (detector aging drifts it over months to years — PRACTICE.md §1), or recover a working
approximation for a sensor whose factory table is unavailable or stale, using nothing but the sensor's
own overlapping observations. The self-calibration idea itself — decompose a set of noisy pairwise or
overlapping measurements into per-sensor and per-target unknowns via least squares, fix the gauge by
convention, and flag whatever the observation graph cannot reach — is a general pattern that recurs
across robotics and computer vision under many names: **camera photometric/vignetting self-calibration
from overlapping images** (project 01.09, this project's direct camera-side twin, cited throughout);
**radiometric calibration from image sequences** in the HDR/computational-photography literature;
**bundle adjustment's own gauge-freedom handling** (a pose graph has the identical "only relative poses
are observable, fix one to anchor the rest" structure); and **sensor-network self-calibration**
generally (any fleet of nominally-identical sensors whose individual gains drift). For LiDAR
specifically, intensity-aware SLAM and place-recognition work (scan-context variants that incorporate
intensity, cited in README "Prior art") is a direct consumer that assumes exactly the kind of
calibration this project performs is already done.
