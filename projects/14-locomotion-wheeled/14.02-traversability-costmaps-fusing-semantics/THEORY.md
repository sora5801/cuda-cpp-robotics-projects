# 14.02 — Traversability costmaps fusing semantics + geometry: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why a wheel needs more than "is the ground there?"** A wheeled vehicle's tire maintains continuous
rolling contact with the ground — unlike 13.03's legged robot, which commits to one discrete foot
placement per step, a wheeled vehicle's contact patch is *always* there, tracking the terrain
underneath it. That sounds forgiving, and mostly it is: a wheel shrugs off small bumps a foot would
need to plan around. But it fails in ways a foot does not, and understanding those two hard-failure
modes — **rollover** and **high-centering/step-climb failure** — is where this project's two hard vetoes
come from.

**Rollover — the geometry of tipping over sideways.** Model the vehicle as a rigid box of track width
`T` (the lateral distance between left and right wheel contact lines, meters) and center-of-gravity
height `h` above the ground plane (meters), standing still on a slope of angle `theta` (the worst case:
the slope runs *across* the vehicle's direction of travel, a "side-slope" or "cross-slope"). Gravity
`m*g` acts straight down through the CoG. Resolve moments about the downhill wheel's contact line — the
vehicle is on the verge of rolling over exactly when the CoG's vertical projection reaches that contact
line:

```
tan(theta_rollover) = T / (2*h)
```

(`T/2` is the horizontal distance from vehicle centerline to a wheel's contact line; the vehicle
tips when the *horizontal* offset of the CoG from that contact line — driven by `h*sin(theta)` acting
against `T/2*cos(theta)` — reaches zero; the small-slope-independent ratio `tan(theta) = (T/2)/h` is the
standard static-rollover threshold used throughout ground-vehicle dynamics). A vehicle with a WIDE,
LOW stance (large `T`, small `h`) tolerates steep cross-slopes; a NARROW, TALL vehicle — exactly what an
off-road platform becomes once you bolt a LiDAR mast and a compute box on top — tips much sooner. This
project's illustrative dimensions (`kTrackWidthM=0.6 m`, `kCogHeightM=0.6 m`, PRACTICE.md §2 dates and
caveats them) give `theta_rollover = atan(0.6/(2*0.6)) = atan(0.5) = 26.57 deg`.

**Traction — the same friction-cone argument 13.03 uses, applied to a rolling contact instead of a
point.** A tire's traction is bounded by Coulomb friction exactly like 13.03's foot pad: on a slope of
angle `theta`, the tire can hold without slipping only while `tan(theta) <= mu` (13.03's THEORY.md
derives this from resolving gravity into normal and tangential components at the contact). This
project's illustrative tire-on-dirt/gravel `kWheelMu = 0.7` gives `theta_traction = atan(0.7) = 34.99 deg`.

**Which one governs?** `slope_limit_rad = min(theta_traction, theta_rollover)` — the MORE restrictive of
the two, because either failure mode alone is enough to end the drive. For this project's illustrative
vehicle, **rollover governs** (26.57 deg < 34.99 deg): a wide friction budget does not help a narrow,
top-heavy vehicle that tips over well before its tires would ever slip. A wider, lower vehicle (larger
`T`, smaller `h`) would flip this — traction would govern instead — and the demo's own `PROBLEM:` line
prints which one is binding every run, so this is not a fact you have to take on faith; change
`kTrackWidthM`/`kCogHeightM` in `kernels.cuh` and rebuild to see the crossover for yourself (README
Exercise territory).

**Step-climb — the geometry of a wheel meeting a vertical-ish edge.** Consider a rigid wheel of radius
`r` (meters) approaching a step (or ledge) of height `h_step`. The wheel first touches the step at its
corner. Quasi-statically (ignoring momentum — the conservative, "coasting" case; a moving vehicle with
torque and momentum can sometimes climb more than this bound predicts, a limitation this project states
honestly, THEORY §Where this sits in the real world), the wheel's center sits a horizontal distance
`sqrt(r^2 - (r-h_step)^2)` from the corner, and the line from the corner contact point to the wheel
center makes an angle `phi` with the vertical where

```
cos(phi) = (r - h_step) / r      i.e.      h_step = r*(1 - cos(phi))
```

For the wheel to climb WITHOUT SLIPPING at that corner contact, the resultant contact force (weight
support plus whatever forward push is driving the climb) must stay inside the friction cone at the
corner — the SAME friction-cone condition as the slope-traction argument above, just applied to a point
contact against a corner instead of a contact against a flat slope, giving a maximum admissible contact
angle `phi_max = atan(mu)`. Substituting:

```
step_limit_m = r * (1 - cos(atan(mu)))
```

This project reuses `kWheelMu = 0.7` for BOTH derivations (one constant, two independent physical
consequences — THEORY.md's single-source spirit) and an illustrative wheel radius `kWheelRadiusM = 0.20 m`,
giving `step_limit_m = 0.20*(1 - cos(34.99 deg)) = 0.20*0.1808 = 0.0362 m` (3.6 cm) — the exact number
the demo's `PROBLEM:` line prints. This is a deliberately CONSERVATIVE, passive bound: it says nothing
about what torque and forward momentum can accomplish (§Where this sits in the real world discusses
real rover-mobility analyses that go further).

**Why grass lies to geometry, and water lies to both.** A LiDAR or stereo depth sensor measures RETURN
RANGE, not "the ground" — it reports wherever the beam's reflection came from. Off-road, that
distinction matters enormously:

- **Tall grass / vegetation** scatters a beam's return unpredictably across the canopy's height, not
  the true ground surface beneath it. A patch of tall grass over perfectly flat, drivable dirt can
  therefore look — to a purely GEOMETRIC pipeline — like rough, unpredictable terrain: locally elevated
  roughness and small apparent slope discontinuities that have NOTHING to do with what a wheel would
  actually experience rolling through it. This project's `VEGBUMP` synthetic feature stands in for
  exactly this sensor physics (§The algorithm below works through why it elevates `roughness_m`
  specifically, not `slope_rad`).
- **Water** is the opposite failure in both channels at once: it is often the single FLATTEST, most
  geometrically inviting surface in a scene (a still pool reflects a near-perfect plane back at a
  depth sensor, or — worse — SPECULARLY reflects the sky and returns almost no usable range at all,
  which a naive pipeline can interpret as "no obstacle" rather than "no data"). Geometry alone can be
  actively MISLED by water, not just uninformed. And critically, water hides what actually matters —
  depth, current, a soft or scoured bottom — from EITHER channel: no amount of geometric or semantic
  confidence can certify what is underneath. This is the physical reasoning behind treating a semantic
  WATER classification as an unconditional veto (§The two-channel fusion problem, below) rather than one
  more signal to be averaged in.

## The math

**Problem statement.** Given a row-major elevation grid `elevation_m[row*W+col]` (meters,
`kGridW=kGridH=256`, `kCellM=0.10 m` — the layout contract in [`src/kernels.cuh`](src/kernels.cuh)) and
co-registered semantic arrays `semantic_class[row*W+col]` (a `uint8_t` in `[0,6)`, the six-class
palette below) and `confidence[row*W+col]` (float in `[0,1]`), compute per cell: a slope
`slope_rad in [0,pi/2)`, a step height `step_height_m >= 0`, a roughness `roughness_m >= 0`, a
geometric cost `geo_cost in [0,1]`, a semantic cost `semantic_cost in [0,1]`, a fused cost
`fused_cost in [0,1]`, and a speed limit `speed_limit_mps in [0, kVMaxMps]`.

**The six-class palette and prior costs** (single-sourced in `kernels.cuh`, repeated here for
reference):

| Class | Prior cost | Physical reasoning |
|---|---|---|
| `CLASS_DIRT` | 0.05 | firm, bare, compacted ground — the easy baseline |
| `CLASS_GRAVEL` | 0.10 | loose but firm, good drainage, good traction |
| `CLASS_GRASS` | 0.20 | short/mown ground cover — mild traction hit, ground itself visible/trustworthy |
| `CLASS_VEGETATION` | 0.45 | tall grass/brush — occludes the true ground; may hide a small hazard |
| `CLASS_WATER` | 1.00 | independently hard-vetoed regardless of this prior (§below) |
| `CLASS_UNKNOWN` | 0.65 | no confident label of any kind |

**Least-squares plane fit** (identical derivation to 13.03's, restated in this project's own notation
and window size). For a cell at `(row,col)`, gather every neighbor in the `(2*kFitRadiusCells+1)^2 = 7x7`
window, with cell-centered local coordinates `x_i = dc*kCellM`, `y_i = dr*kCellM` and height `z_i`. Fit
`z = a*x + b*y + c` minimizing `E(a,b,c) = sum_i (z_i - a*x_i - b*y_i - c)^2`. Setting the three partial
derivatives to zero gives the normal equations

```
| Sxx  Sxy  Sx | |a|   |Sxz|
| Sxy  Syy  Sy | |b| = |Syz|
| Sx   Sy   n  | |c|   |Sz |
```

solved by Cramer's rule (`src/kernels.cu`'s `solve_plane_3x3` — see its header comment for the full
three-determinant expansion; 13.03's kernels.cu derives it in even more detail).

**Slope**, exactly as 13.03 derives it: `slope_rad = atan(sqrt(a^2+b^2))` — the angle between the
fitted plane's steepest-ascent direction and horizontal.

**Roughness**, exactly as 13.03: `roughness_m = sqrt((1/n) * sum_i (z_i - a*x_i - b*y_i - c)^2)` — the
population standard deviation of the fit's own residuals, computed in a SECOND pass once `(a,b,c)` are
known.

**Step height — this project's own addition, a DIFFERENT window.** Over the SEPARATE, tighter
`(2*kStepRadiusCells+1)^2 = 5x5` window, `step_height_m = max_i(z_i) - min_i(z_i)` — the raw elevation
swing, no fitting at all. Why not just read this off the wide window's own residuals? Because a
LEAST-SQUARES fit MINIMIZES total squared error by finding the best AVERAGE trend — a single sharp step
near the center of a 7x7 window gets partially "absorbed" into the fitted slope `(a,b)` rather than
showing up as a large, unambiguous residual, especially when the step is small relative to the window
(a symmetric V-shape, like this project's ditch sampled at its exact center, is the extreme case:
THEORY.md's own verification section below measures this directly). A raw max-min gather over a
SMALLER window, sized to roughly one wheel/track contact patch, sidesteps the fitting entirely and
answers the question a wheel actually needs answered: "how much does the ground under me change over
the next few tens of centimeters?"

**The two hard-veto limits**, derived in §The problem above:

```
slope_limit_rad = min( atan(kWheelMu), atan(kTrackWidthM / (2*kCogHeightM)) )     = 0.4636 rad (26.57 deg), rollover-governed
step_limit_m    = kWheelRadiusM * (1 - cos(atan(kWheelMu)))                        = 0.0362 m
```

**Geometric cost.** With `slope_cost = clamp(slope/slope_limit, 0, 1)`, `step_cost =
clamp(step/step_limit, 0, 1)`, `rough_cost = clamp(roughness/kRoughnessMaxM, 0, 1)`:

```
geo_cost = clamp( kWeightSlope*slope_cost + kWeightStep*step_cost + kWeightRough*rough_cost, 0, 1 )
```

with `kWeightSlope=0.4, kWeightStep=0.3, kWeightRough=0.3` (sum to 1.0 — a convex combination, so
`geo_cost` stays in `[0,1]` by construction).

**Semantic cost — the confidence-weighting blend.** `semantic_cost = confidence*prior[class] +
(1-confidence)*kPessimisticPriorCost`, a convex combination between "trust the label completely"
(confidence=1: `semantic_cost = prior[class]`) and "trust it not at all" (confidence=0: `semantic_cost
= kPessimisticPriorCost = prior[CLASS_UNKNOWN] = 0.65`). This equality is deliberate, not coincidental:
a label the segmentation net itself is not confident in carries LESS information than no label at all
would be dangerous to assume — the honest floor for "I don't trust this" is exactly "I don't know",
never "I'll assume it's cheap". This is the physical intuition from §The problem made precise: unknown
grass MIGHT hide a ditch; a low-confidence "probably dirt" reading should not get to claim dirt's cheap
0.05 prior just because the argmax happened to land there.

**Fusion — the central formula of this project.**

```
geo_veto = isnan(slope) OR (slope > slope_limit_rad) OR (step_height > step_limit_m)
sem_veto = (semantic_class == CLASS_WATER)                          # regardless of confidence — see below

fused_cost = 1.0                                                     if geo_veto OR sem_veto
           = clamp( kWeightGeo*geo_cost + kWeightSem*semantic_cost, 0, 1 )     otherwise
```

with `kWeightGeo = kWeightSem = 0.5` (equal trust by default; §The two-channel fusion problem discusses
when a real system would weight these unevenly).

**Speed limit — a curvature-free stopping-distance bound.** Model straight-line braking:
`v_final^2 = v_initial^2 - 2*a*d`. Setting `v_final = 0` (must be able to fully stop) and solving for
the initial speed that exhausts the available deceleration `a_avail` over a fixed distance `kStopDistM`:

```
a_avail(cost) = kSafetyFraction * kWheelMu * kGravityMps2 * (1 - cost)
v_limit(cost) = min( kVMaxMps, sqrt(2 * a_avail(cost) * kStopDistM) )
```

`kSafetyFraction = 1/3` RESERVES two-thirds of the tire's total friction budget for simultaneous
steering — the classic **friction-circle** argument (a tire's combined lateral+longitudinal force
magnitude is bounded by `mu*F_normal`, not each independently; braking hard while also turning consumes
budget from both). "Curvature-free" means precisely this: the bound assumes braking along the CURRENT
heading (no turn), using only a SENSOR-RANGE/reaction-distance property (`kStopDistM`), never an
assumed path curvature — because at costmap-build time, no particular path has been chosen yet. A
downstream sampling controller (14.01's MPPI) is exactly where curvature-specific reasoning belongs: it
evaluates concrete candidate trajectories, each with its own curvature, against this SPEED CEILING as
one term in its own running cost.

## The algorithm

Per costmap (all four kernels are `O(W*H)` cells, launched as one thread per cell):

1. **Geometric layer**: gather up to `7x7=49` neighbors for the plane fit (`O(k1^2)` per cell,
   `k1=kFitRadiusCells=3`), solve for `(a,b,c)`, derive slope, run a second `49`-neighbor pass for
   roughness, then a SEPARATE `5x5=25`-neighbor max-min gather (`k2=kStepRadiusCells=2`) for step
   height. **Complexity:** `O(W*H*(k1^2+k2^2))` serial; embarrassingly parallel across cells (no cell's
   OUTPUT depends on another cell's output, only on shared input heights — 13.03's exact reasoning).
2. **Semantic layer**: one class lookup, one confidence blend per cell. `O(W*H)`, trivially parallel.
3. **Fusion**: two comparisons, one blend, per cell — reads the four already-computed per-cell numbers.
   `O(W*H)`, trivially parallel.
4. **Speed limit**: one sqrt, one clamp, per cell. `O(W*H)`, trivially parallel.

**Working through WHY the ditch's exact center defeats the wide-window slope fit (a real design lesson
this project's own analytic gate had to learn — kept visible here rather than smoothed over).** This
project's `DITCH` feature is a symmetric trapezoidal V: a linear descent, a flat bottom, a linear ascent
back up, with the two walls at EQUAL and OPPOSITE grades. Sample the WIDE (0.7 m) fit window centered
exactly at the geometric midpoint of the flat bottom: the window's coverage extends symmetrically up
BOTH walls. A least-squares PLANE fit to a symmetric V is trying to find the single best-fit TILT — and
a symmetric V has zero net tilt at its center (the up-slope on one side exactly cancels the down-slope
on the other in the normal-equations sums). The V-shape does not disappear; it shows up entirely as
ROUGHNESS (residual), not slope. This project's own `main.cu` initially sampled Gate C — the "ditch must
be vetoed" analytic check — at exactly this symmetric center and measured a LOW fused cost (0.28, not
vetoed) purely from this cancellation, not from any kernel bug (VERIFY's stage-isolated GPU-vs-CPU gates
passed the entire time; this was a GATE-DESIGN mistake, not an algorithm mistake). The fix, visible in
`main.cu`'s Gate C: sample the WALL, not the symmetric bottom, where the plane fit sees a genuinely
one-sided ~45-degree grade rather than a self-cancelling V. §How we verify correctness returns to this
as a case study in why an analytic gate's SAMPLING LOCATION is as load-bearing as its threshold.

## The GPU mapping

```
All four kernels: 2-D launch, thread (col,row) owns cell (col,row)
    block = 16x16 = 256 threads (warp-friendly square tile — 13.03's exact launch geometry)
    grid  = ceil(W/16) x ceil(H/16)  =  16x16 blocks exactly at 256x256
```

**Why every kernel here is 2-D per-cell, with NO per-query stage (a deliberate contrast with 13.03).**
13.03 chains three per-cell map/stencil kernels into a fourth, per-QUERY batched-search kernel, because
a foot planner asks "where, near this ONE nominal point, should I actually step?" — a genuinely
different-shaped problem (a small number of independent local searches) from the map-building stages
before it. A COSTMAP has no equivalent question: EVERY cell needs an answer, not a handful of nominated
points, because the downstream consumer (14.01's MPPI) evaluates thousands of ROLLOUTS, each sampling
the costmap at whatever cell its own trajectory currently occupies — the "query" happens entirely
DOWNSTREAM, in a different project, against the finished costmap array. This project's four kernels are
therefore a more UNIFORM pipeline shape than 13.03's: every stage is "one thread per cell," a smaller
conceptual surface but a genuinely different lesson (a costmap-BUILDING pipeline looks different from a
costmap-CONSUMING one, and this repo's domain-14/domain-23 sibling projects and 14.01 show both halves).

**Memory hierarchy: global only, no shared memory — and why not (yet), same story as 13.03.** The
geometric layer's two windows re-read heavily overlapping cells across neighboring threads (up to 49x
and 25x redundant global-memory reads respectively, for the wide and tight windows). A shared-memory
TILED version would have each thread block cooperatively load its patch of the grid (plus a
halo of `kFitRadiusCells`/`kStepRadiusCells` neighbor cells) into shared memory ONCE, then have every
thread in the block read from that fast on-chip copy — the textbook stencil optimization (README
Exercise 5). Not built here for the same reason 13.03 states: at this map's size (65,536 cells,
sub-millisecond kernels already, §What this computes measures it), the L2 cache absorbs most of the
redundancy in practice, and the UNTILED version keeps the thread-to-data mapping — the concept this
project exists to teach — visible without shared-memory index bookkeeping standing in front of it.

**The six-class palette lives in `__device__` global memory, not `__constant__`.** `kClassPriorCost`
(kernels.cuh) is a 6-entry table every thread in `semantic_layer_kernel`/`fusion_kernel` reads — the
textbook use case for CUDA's `__constant__` memory space (a small, read-only, broadcast-friendly table
the hardware caches specially). This project keeps it as a plain `__device__` array instead, for one
practical reason worth stating honestly: `__constant__` and `__device__` global arrays behave
identically for a table this small (6 floats = 24 bytes — both fit comfortably in L1/L2 and both get
broadcast-read efficiently by a warp indexing the same small array), and using `__device__` lets the
SAME array declaration compile under both nvcc (device code) and cl.exe (`reference_cpu.cpp`'s host-only
oracle) via one `#ifdef __CUDACC__` fence — see the header comment on `kClassPriorCost` in
`kernels.cuh` for the exact mechanism. README Exercise territory: switch it to `__constant__` and
profile whether it measurably changes anything at this table size (it should not, and demonstrating
that empirically is itself the lesson).

**No atomics, no divergence beyond loop bounds and the veto branches** — every kernel's only branching
is window-boundary clipping and the (data-dependent, but spatially COHERENT — terrain features span
many cells, not single ones) hard-veto conditions, so neighboring cells in a warp almost always take the
same branch.

## Numerical considerations

- **FP32 throughout the GPU path**, matching the repo default; elevations are `O(0.01-1.0 m)`, so
  absolute float32 precision (~1e-7 relative) is many orders below anything these thresholds care about.
- **The CPU oracle deliberately uses DOUBLE precision in the plane fit** — a documented departure from
  13.03's float-both-sides choice (`reference_cpu.cpp`'s file header states this explicitly). The
  reasoning: this project's fit window is larger (49 samples vs. 13.03's 25) and this project's
  terrain has sharper local features (the berm/ditch/vegetation-bump edges) than 13.03's gentler ramp
  and step, so a genuinely higher-precision oracle is a stronger correctness check than a second
  float32 implementation of the identical formula would be — at the cost of Stage 1's VERIFY tolerance
  needing to absorb a real (if still small) float32-vs-float64 gap rather than a pure few-ULP rounding
  gap. Measured worst case: `1.05e-6 rad` slope, `2.24e-8 m` roughness, well inside the documented
  `2e-3 rad`/`2e-4 m` tolerances (main.cu's `kSlopeTolRad`/`kRoughTolM`) with wide margin.
- **Step height needs NO tolerance margin at all.** `step_height_m = max - min` over identical input
  floats on both paths is a pure comparison operation with no accumulation and no fused-multiply-add
  ambiguity (contrast 13.03's foothold-selection kernel, whose disc-membership test IS an
  FMA-vs-non-FMA rounding trap) — measured `max|dstep| = 0.000e+00` exactly, every run, and the demo's
  VERIFY gate holds it to an EXACT match rather than a tolerance.
- **The symmetric-V cancellation is a NUMERICAL fact, not a bug**, worth restating from §The algorithm:
  a least-squares plane fit to a perfectly symmetric valley genuinely has near-zero fitted slope at the
  exact center, by the mathematics of the normal equations, not by any implementation error. This
  project's own Gate C sampling-location fix (§The algorithm) is the visible trace of this project
  learning that lesson during construction — left in the text specifically because CLAUDE.md's
  "no black boxes" spirit asks every reader to internalize it before writing a similar gate elsewhere.
- **Confidence is defensively clamped to `[0,1]`** at the point of use (both `semantic_layer_cpu` and
  `semantic_layer_kernel`) even though the synthetic generator already produces values in `[0.05,0.99]`
  — a real segmentation net's softmax output is mathematically guaranteed to lie in `[0,1]`, but nothing
  stops an upstream bug (or a different data source entirely) from handing this pipeline something
  outside that range, and a fusion cost silently exceeding `[0,1]` would corrupt every downstream
  consumer's assumptions.
- **Angle handling.** `slope_rad in [0, pi/2)` by construction (`atan` of a non-negative argument) — no
  periodic wrapping is possible or needed here, exactly 13.03's reasoning for the same quantity.

## How we verify correctness

**Two independent layers, because this pipeline can fail in two different ways a single check would
miss** (13.03's exact framing, restated for this project's own scenario):

1. **VERIFY — four STAGE-ISOLATED GPU-vs-CPU gates**, each kernel-under-test and its CPU oracle fed the
   IDENTICAL upstream arrays (the CPU oracle's own prior-stage output, uploaded to the device for the
   GPU kernel), so no error can accumulate BETWEEN stages. Measured, this run (RTX 2080 SUPER): Stage 1
   (geometric layer, genuinely independent float32-vs-float64 arithmetic on the SAME raw elevation)
   `1.05e-6 rad` slope, `0` step (exact), `2.24e-8 m` roughness, 0 NaN-pattern mismatches over 65,536
   cells; Stage 2 (semantic layer, pinned inputs) `5.96e-8` semantic-cost difference; Stage 3 (fusion,
   pinned inputs) `5.96e-8` geo-cost and fused-cost difference, `0` veto-reason mismatches out of
   65,536 (an EXACT match — `veto_reason` is a discrete classification, and once fed bit-identical
   pinned floats, both paths' comparisons must agree exactly, the same bar 13.03 holds its
   foothold-selection cell index to); Stage 4 (speed limit, pinned inputs) `0` (exact — pure sqrt/min
   of identical floats).
2. **Two DESIGNED-DISAGREEMENT analytic gates, run on the real, all-GPU, end-to-end pipeline** (never
   mixed with CPU numbers) — the project's own central claim, checked against the scenario's own known
   ground truth: Gate A measures mean geo_cost 0.1043 (bound `<0.25`, "geometry alone looks safe")
   over the water pool's interior (224 cells, inset one fit-window-radius from the pool's edges) while
   ALL 224 cells are vetoed and ALL 224 carry the semantic-veto bit — semantics wins independent of how
   good geometry looks. Gate B measures mean geo_cost 0.4442 (bound `>0.35`, "geometry alone looks
   bad") over the vegetation patch's interior (1870 cells) while 0/1870 are hard-vetoed, the mean fused
   cost (0.4591) stays at or under the 0.60 validity bound, and the mean speed limit (2.224 m/s) is
   measurably below the 2.50 m/s cruise cap while staying well above zero — semantics rescues the cell,
   at reduced speed. A third, NON-gated diagnostic reports what a MAX-based ("worst channel wins")
   fusion rule would have produced on this same patch (mean 0.4872, worst single cell 0.5374) —
   consistently more pessimistic than this project's weighted blend, though in this project's own
   committed scenario not pessimistic enough to cross the 0.60 validity bound itself (§The two-channel
   fusion problem discusses the general argument; README Exercise 2/3 invite pushing the terrain until
   it does, with your own measured numbers).
3. **Two pure-geometry analytic gates**, also end-to-end GPU: the ditch's descending-wall midpoint
   (NOT its symmetric flat-bottom center — §The algorithm explains why that distinction is load-bearing)
   is hard-vetoed (fused_cost exactly 1.0000, geo-veto bit set) despite being labeled cheap GRAVEL
   (0.10 prior) — geometry vetoes on its own terms; the berm's measured mean slope (17.861 deg, sampled
   one fit-window-radius in from its own transition kinks) tracks its constructed 18.00 deg within
   ±1.5 deg — a pure geometry-fidelity check, the same kind 13.03's ramp gate performs.

The scenario is entirely synthetic and entirely known (README §Data) SPECIFICALLY so this second and
third layer are possible — a real, recorded elevation+semantic map has no designed disagreement case to
check a fusion rule against, and no constructed berm angle to check a measured slope against.

## The two-channel fusion problem

**Why two INDEPENDENT hard vetoes, not one combined threshold?** A single "combined danger score past
some threshold" veto would force a design choice this project deliberately avoids: HOW BAD does
semantics have to be to matter as much as a real geometric hazard, and vice versa? The two failure modes
this project models are not commensurable. A slope past the rollover limit WILL tip the vehicle over —
a fact about rigid-body statics, true regardless of what is painted on the ground. A WATER
classification means the pipeline genuinely does not know what is beneath the surface — a fact about
missing information, true regardless of how gentle the visible surface looks. Keeping the two vetoes
INDEPENDENT (either one alone forces `fused_cost = 1.0`) means neither channel's confidence can talk the
other one down from a veto it is sure about — exactly the asymmetric-risk argument below, made
structural rather than tunable.

**The asymmetric-risk argument for the water veto's confidence-immunity.** Every OTHER semantic
class's cost is confidence-weighted (§The math's convex blend) — so why does WATER alone ignore
confidence entirely? Because the two possible errors are not symmetric. A FALSE POSITIVE water
classification (the pipeline says "water" when it is actually just wet, dark dirt) costs a vehicle a
minor detour around a patch that was actually fine — an efficiency loss. A FALSE NEGATIVE (the pipeline
says "not water" — or worse, an UNCERTAIN "maybe water" gets averaged down to a cheap cost — when it was
actually water) risks the vehicle driving into standing water of unknown depth, current, or bottom
condition: a safety loss, potentially catastrophic, and NOT something a few extra percentage points of
"probably fine" softens. When the cost of one error type vastly exceeds the cost of the other, the
textbook decision-theoretic answer is NOT "average the two error costs weighted by confidence" — it is
"bias hard toward the safer error." This project encodes that bias structurally: any ARGMAX water
classification, regardless of confidence, is treated as an unconditional veto.

**Conservative (MAX) fusion vs. optimistic (weighted-blend) fusion — the design choice this project
makes, and its failure modes.** Everywhere the hard vetoes do not fire, this project blends
`geo_cost` and `semantic_cost` with a WEIGHTED AVERAGE (`fused_cost = kWeightGeo*geo_cost +
kWeightSem*semantic_cost`). A different, equally legitimate design is MAX-fusion — `fused_cost =
max(geo_cost, semantic_cost)`, "the worse channel decides":

- **Weighted-blend (this project's choice).** A confident, cheap reading from ONE channel can pull down
  a mediocre reading from the OTHER — this is exactly the "semantics rescues a geometrically noisy
  cell" story Gate B demonstrates with real numbers. The failure mode: if one channel is systematically
  and CONFIDENTLY wrong across a wide area (not just noisy but actually miscalibrated — e.g. a
  segmentation net trained on a different biome confidently mislabeling a genuine hazard as benign
  vegetation), the blend can dilute a real warning from the other channel into a merely "moderate" cost
  instead of the emphatic veto it deserves.
- **MAX-fusion (the conservative alternative).** ROBUST to either channel's blind spots: if either
  channel is bad, the fused cost is bad, full stop — no averaging-away of a genuine hazard. The failure
  mode is the mirror image, and this project's own `main.cu` measures it directly (§How we verify
  correctness, item 2): a channel that is merely NOISY (not wrong, just imprecise — exactly what
  vegetation-canopy LiDAR returns are) can dominate the max and flag a cell invalid that a human, or a
  better-calibrated model, would recognize as perfectly fine. Overly conservative fusion is not a free
  lunch — it can make a genuinely usable off-road corridor look impassable everywhere the map is merely
  UNCERTAIN, not actually dangerous, which in a real deployment either strands the vehicle or trains an
  operator to distrust (and eventually override) the safety layer entirely — its own kind of risk.

This project ships the weighted blend as the PRIMARY rule (because README's own worked example, Gate B,
is exactly the "rescue a noisy-but-fine cell" story this repo wants to teach) while measuring the
conservative alternative on the same data, honestly, every run (§How we verify correctness item 2) —
so a reader sees BOTH design points' real, measured behavior on the same terrain rather than taking
either argument on faith. A production system's real answer is usually neither extreme: per-channel
CONFIDENCE-DEPENDENT weighting (a channel reporting low confidence contributes less to the blend,
naturally sliding the fused rule toward MAX-like conservatism exactly where a channel admits it does
not know, and toward optimistic blending exactly where both channels are confident) — README Exercise 4
sketches this extension.

## Where this sits in the real world

- **ETH Zurich/ANYbotics's `elevation_mapping`/`grid_map` ecosystem** (13.03's closest production
  analogue) increasingly layers a LEARNED semantic classifier on top of its geometric traversability
  layer — exactly this project's two-channel story, at production scale and with real per-cell
  UNCERTAINTY propagated from sensor noise models (a layer this project omits, README §Limitations).
- **NASA/JPL Mars-rover mobility and traversability analysis** derives hard slope and step limits from
  rover geometry (wheel radius, suspension travel, rollover margins) using far more detailed
  wheel-terramechanics models than this project's friction-cone/rollover-geometry teaching version —
  including the ACTIVE-climbing case this project's passive, quasi-static step-limit derivation
  explicitly excludes (a moving rover with torque and momentum can sometimes exceed the passive bound;
  §The problem states this honestly). Published rover mobility papers are the right next read for
  anyone who wants the full version of this project's `step_limit_m` derivation.
- **DARPA RACER-era off-road autonomy research** fuses geometric LiDAR-derived costmaps with LEARNED
  semantic traversability classifiers for high-speed unstructured-terrain driving — the closest modern
  research analogue to this project's exact fusion shape, typically at far higher update rates and with
  learned (not hand-tuned) fusion weights.
- **What the full version adds beyond this teaching core:** per-cell height/label UNCERTAINTY
  propagated from real sensor noise models (not just a simulated softmax confidence); confidence-
  dependent fusion weighting instead of a fixed 0.5/0.5 split (§The two-channel fusion problem,
  Exercise 4); an ACTIVE (torque-and-momentum-aware) step-climb model instead of this project's passive
  quasi-static bound; and, at the research frontier, end-to-end LEARNED traversability estimation that
  never separates "geometric" from "semantic" evidence at all, instead training directly on raw sensor
  data with a self-supervised traversability label (a research direction domains 10/12's sim-to-real
  loop, SYSTEM_DESIGN §4.3, feeds).
