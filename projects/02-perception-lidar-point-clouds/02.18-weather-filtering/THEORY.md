# 02.18 — Weather filtering: snow/rain/dust outlier removal (DROR/LIOR): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### LiDAR versus precipitation: what actually happens to a beam

A time-of-flight LiDAR fires a short pulse of infrared light (typically 905 nm or 1550 nm) and times
how long it takes for enough of that light to bounce back to the receiver to cross a detection
threshold. In clear air, essentially all of the pulse's energy either reflects off the first solid
surface it meets or escapes into open sky. Snow, rain, and dust introduce a THIRD outcome: the pulse
can hit an airborne particle — an ice crystal, a water droplet, a grain of mineral dust — long before
it reaches whatever solid surface is behind it.

Two physical facts determine what the sensor sees when that happens, and both matter for this
project's filters:

**1. Partial beam interception.** A real LiDAR beam is not an infinitely thin ray; it diverges with
range, so by the time it has traveled a distance `r`, it illuminates a roughly circular disk of area

```
A_footprint(r) = pi * (r * theta_div / 2)^2
```

where `theta_div` is the beam's full divergence angle (radians) — an illustrative order of magnitude
for a spinning automotive LiDAR is a few milliradians (`kernels.cuh`'s `kDrorAlphaRad` uses the
sensor's ANGULAR SAMPLING resolution, a related but distinct quantity discussed below; the
`BEAM_DIVERGENCE_RAD` constant in `scripts/make_synthetic.py` is the beam's own physical width). A
millimeter-scale particle sitting inside that disk intercepts only a tiny FRACTION of the beam's
power — approximately

```
fraction_intercepted(r) = sigma / A_footprint(r)
```

where `sigma = pi * a^2` is the particle's geometric cross-section (`a` its radius). The returned
intensity is that fraction times the particle's own backscatter reflectance `rho_type` (a material
property — ice, water, and mineral dust all backscatter differently, and much of that difference is
wavelength-dependent, a subject this project deliberately does not model at that depth — see "Where
this sits in the real world"). Two consequences follow directly, and both are load-bearing for LIOR:

- **Scatterer returns are systematically DIM.** A solid surface fills the WHOLE beam footprint and
  reflects according to its own material's reflectance; a millimeter-scale particle fills a
  vanishingly small fraction of it. This is a real, physical, unavoidable signal — not an artifact of
  this project's synthetic data.
- **That dimness gets WORSE with range, faster than a solid surface's does.** `A_footprint(r)` grows
  as `r^2`, so `fraction_intercepted(r)` shrinks as `1/r^2` even BEFORE accounting for the beam's own
  round-trip power falloff — while a solid surface's calibrated, range-compensated intensity stays
  roughly flat with range (that compensation is exactly what a properly calibrated intensity channel
  does — see project 02.20 and this project's `intensity_dependence` gate). A snowflake at 5 m already
  reads dim; the same snowflake at 30 m reads much dimmer still, relative to a real surface's constant
  baseline.

**2. Beer-Lambert extinction — the probability a beam gets intercepted at all.** Whether a beam
encounters a particle in the first place is a classic radiative-transfer question: given a medium with
particle number density `N` (particles/m^3) and per-particle cross-section `sigma` (m^2), the
probability that a beam travels a path length `L` (meters) WITHOUT being intercepted is

```
P(no interaction over L) = exp(-N * sigma * L)
```

the same exponential-decay law that governs radar attenuation, light attenuation through fog, and
neutron absorption in a reactor shield — "N * sigma" is the medium's EXTINCTION COEFFICIENT (units
1/m), and the argument `N * sigma * L` is the OPTICAL DEPTH the beam has traveled through. The
probability of AT LEAST ONE interaction is therefore `p_hit = 1 - exp(-N * sigma * L)`
(`scripts/make_synthetic.py`'s `try_scatter` implements exactly this). Given that an interaction
occurred, WHERE along the path it happened is not uniformly distributed — a Poisson process's first
event is more likely to happen early in a dense medium than late, following the truncated exponential
distribution `sample_scatter_range` samples via inverse-CDF (derived in full in that function's
docstring).

**Attenuation honesty.** Not every intercepted beam produces a usable return: a strong-enough
scattering event can attenuate the pulse below the receiver's detection threshold entirely, producing
NO return rather than a weak one. `scripts/make_synthetic.py`'s `kPLostGivenScatter`-style probability
(named `SNOW_P_LOST`/`RAIN_P_LOST`/`DUST_P_LOST` per weather type) models this honestly rather than
pretending every scatter event is detectable.

### The engineering constraint this project's filters exist to fix

A spinning LiDAR samples the world on a FIXED ANGULAR GRID — a fixed number of elevation rings and a
fixed azimuth step. That grid has a direct geometric consequence for point SPACING: two adjacent
beams, separated by angle `alpha` (radians), diverge to a linear separation of approximately

```
spacing(r) ~= alpha * r
```

at range `r` — this is simple arc-length geometry, the same "arc length = angle * radius" relationship
from a first trigonometry course. A solid surface sampled by this grid therefore has POINT DENSITY
(points per unit area) that falls off as `1/r^2` — the identical 1/r^2 story project 02.01 teaches
from the opposite direction (a uniform-density point cloud's voxel occupancy count falls off with
range for the same reason). This project's `WALL_FAR` object (`scripts/make_synthetic.py`, ~32 m) and
the naturally shallow-elevation-ring ground returns both exist specifically to make this density
falloff observable and gradeable (`data/README.md`'s per-cohort tallies, `demo/out/range_stratified.csv`).

A filter that assumes UNIFORM point density — SOR, this project's deliberately-included baseline —
cannot tell "a real surface, sampled sparsely because it is far away" from "an isolated airborne
scatterer, sampled sparsely because it is genuinely alone in space." Both look identical to a
fixed-neighbor-count, fixed-distance-threshold test. DROR's whole contribution is recognizing that the
search radius itself should grow with range at exactly the rate real-surface spacing does — "The math"
below derives the formula.

### Engineering constraints a real system imposes

- **Noise floor:** a real LiDAR's receiver has a fixed sensitivity; very weak scatterer returns (this
  project's "attenuation honesty" lost fraction) simply never cross it.
- **Latency:** this filter must run inside the perception boundary's sub-scan-period budget
  (README "System context" — 10-20 Hz, well under 100 ms).
- **False-negative cost vs. false-positive cost are NOT symmetric:** a missed snowflake (false
  negative) adds one spurious point a downstream clustering step usually absorbs; a falsely-removed
  real point (false positive) can delete a genuine obstacle from the map. This asymmetry is why this
  project gates `real_point_preservation` as its own floor, separate from raw recall (README "Expected
  output").

## The math

**Symbols** (SI units throughout, sensor frame, sensor at the origin — CLAUDE.md §12): `p_i in R^3`
point `i`'s position (m); `r_i = ||p_i||` its range (m); `I_i in [0,1]` its intensity; `n` the number
of points in one scan.

**SOR (Statistical Outlier Removal).** For each point `i`, let `d_i` be the mean Euclidean distance to
its `K` nearest neighbors (`K = kSorK = 8`):

```
d_i = (1/K) * sum_{j in KNN_K(i)} ||p_i - p_j||
```

Let `mu = mean(d_1..d_n)` and `sigma = std(d_1..d_n)` over the WHOLE scan. Point `i` is an outlier iff

```
d_i > mu + beta_sor * sigma,        beta_sor = kSorStdMult = 0.5
```

**DROR (Dynamic Radius Outlier Removal).** For each point `i`, define a per-point search radius that
GROWS with that point's own range:

```
r_search(r_i) = max(beta_dror * alpha * r_i,  r_min)
```

where `alpha` (radians) is the sensor's native ANGULAR SAMPLING RESOLUTION — this project uses the
azimuth step (`kDrorAlphaRad = 1 deg = 0.0174533 rad`), matching Charron et al.'s own use of the
sensor's horizontal angular resolution; `beta_dror = kDrorBeta = 3.0` is a safety multiplier (Charron
et al. report good results for beta roughly 3-6); `r_min = kDrorRMin = 0.05` m floors the radius near
`r_i -> 0`. Let `c_i` = the number of OTHER points within `r_search(r_i)` of point `i`. Point `i` is an
outlier iff

```
c_i < k_min,        k_min = kDrorKMin = 3
```

**Why this formula, derived from "The problem" above:** a real surface's point spacing at range `r` is
`~= alpha * r` (the arc-length argument). Setting `r_search(r) proportional to alpha * r` means the
search radius grows at EXACTLY the rate real-surface spacing grows, so a real point's own angular
neighbors (on the same ring, on adjacent rings) remain within `r_search` at ANY range — DROR's radius
is the density falloff's own inverse, baked into the search itself. An isolated scatterer, whose
nearest "neighbors" (if any) are other randomly-scattered particles rather than points sampled from a
coherent surface, does not benefit from this scaling: its neighbor count stays low regardless of range
(until the surrounding scatterer field gets dense enough to defeat this — see "Numerical
considerations" and the `dust_plume_honesty` measurement).

**LIOR (Low-Intensity Outlier Removal).** Point `i` is an outlier iff BOTH:

```
I_i < I_thresh                                          (I_thresh = kLiorIntensityThresh = 0.05)
AND
c'_i < k'_min     where c'_i = |{j != i : ||p_i - p_j|| <= R_fixed}|
                        R_fixed = kLiorRadius = 0.35 m,  k'_min = kLiorKMin = 2
```

`R_fixed` is deliberately NOT range-scaled — the simplicity/cost tradeoff LIOR makes relative to DROR
(README/`kernels.cuh`). The "AND" (not "OR") is the guard against throwing away a real but genuinely
dark, densely-and-coherently-sampled surface (a grazing-angle asphalt ground return, this project's
`kCohortGround` — the physics-first reason: real intensity for a Lambertian surface is
`rho * cos(theta_incidence)`, and `cos(theta_incidence)` can be small at a shallow viewing angle purely
from geometry, with no weather involved at all).

## The algorithm

All three filters share the SAME two-stage shape:

1. **STATISTIC** — one pass over the scan computing a per-point number (SOR: mean-KNN-distance; DROR/
   LIOR: neighbor count within a radius). Serial cost: O(n) points, each doing an O(n) scan over every
   other point (SOR additionally maintains a size-K sorted insertion array per point) — O(n^2) total,
   or O(n^2 log K) if a real heap were used for SOR (this project's insertion-sorted array is O(n*K)
   per point, i.e. still O(n^2) total since K is a small constant — see "The GPU mapping").
2. **CLASSIFY** — a trivial O(n) threshold compare (SOR/LIOR additionally need one O(n) reduction —
   the mean/std of the STATISTIC array — computed once, host-side, between the two stages; see "The
   GPU mapping" for why this project does not also teach a GPU reduction kernel here).

Total complexity: **O(n^2)** per filter per scan, dominated entirely by the STATISTIC stage. At this
project's point counts (n ~ 1,000-1,500/scan) that is on the order of 1-2 million distance evaluations
per filter per scan — trivial for a GPU, and measured (README) at well under a millisecond of kernel
time even for all three filters across all three scans combined.

**No spatial index.** A production system at real LiDAR point counts (60,000+ points/scan) would build
a k-d tree, a uniform grid, or an LBVH first (project 02.05's/02.09's territory, cited throughout) and
turn each O(n) inner scan into an O(log n) or O(1)-amortized query. This project deliberately stays at
brute-force scope (README "Limitations") because the teaching focus here is the FILTERING MATH — what
counts as "too sparse" or "too dim," and why — not spatial acceleration structures, which are already
taught in depth elsewhere in this repository.

## The GPU mapping

**Thread-to-data mapping (all six kernels):** one thread owns one point. `kernels.cu`'s six kernels
(`sor_mean_knn_dist_kernel`, `sor_classify_kernel`, `dror_neighbor_count_kernel`,
`dror_classify_kernel`, `lior_neighbor_count_kernel`, `lior_classify_kernel`) all use the identical
grid-stride pattern: `i = blockIdx.x*blockDim.x + threadIdx.x`, striding by `gridDim.x*blockDim.x`.
The STATISTIC kernels each run an inner `for (j = 0; j < n; ++j)` loop reading every other point's
`xyz` — this is the classic ALL-PAIRS GPU pattern (n independent workers, each scanning the same
shared input array), and it needs no communication or synchronization between threads at all: point
`i`'s neighborhood computation touches no other thread's output.

**Memory hierarchy.** `xyz` is read from GLOBAL memory by every thread, `O(n)` times each — `O(n^2)`
total traffic, the dominant cost at this project's scale. No SHARED memory is used: at n in the low
thousands, the whole `xyz` array (`n*3*4` bytes — under 24 KB even at n=2,000) is small enough that the
L1/L2 cache already captures nearly all of the reuse a hand-tiled shared-memory version would buy;
`kernels.cu` names the tiled version as this project's Exercise rather than its baseline, and Exercise
5 (README) asks the learner to measure whether it actually helps here. No CONSTANT or TEXTURE memory
is used — there is no small, read-only, broadcast-pattern table this project's kernels would benefit
from (contrast project 02.13's `__constant__` scan-origin table, cited for the pattern).

**Registers.** SOR's K-nearest search keeps two size-`kSorK` (8) arrays (`best_d2[8]`, `best_idx[8]`)
entirely in registers — small enough (16 registers of bookkeeping, on top of the loop's own locals)
that no spilling occurs at this project's block size (`kernels.cu`'s launch-configuration comment).

**Squared-distance comparisons.** Every THRESHOLD comparison in this project (DROR's and LIOR's radius
tests) compares squared distances against a squared radius, avoiding one `sqrtf()` per candidate pair
in the hottest loops — `x < y` iff `x^2 < y^2` for `x, y >= 0`, so this changes no comparison's outcome
(`kernels.cuh`'s `squared_distance3` docstring). SOR's mean distance needs the TRUE distance (it is
averaging distances, not just comparing them), so it cannot avoid the `sqrtf()` — the one place in
this project's kernels where that micro-optimization does not apply.

**No CUDA library calls.** Every kernel here is hand-rolled (CLAUDE.md §1 "no black boxes"); this
project's default dependency budget is the CUDA runtime + C++17 standard library only (README "Build").

**The host-side reduction (a deliberate scoping choice).** SOR's global `mu`/`sigma` and the classify
threshold they produce are computed on the HOST, from the (small, already GPU-computed) per-point
statistic array copied back — `main.cu`'s `run_gpu_pipeline`. A GPU reduction kernel (parallel
tree-sum) would be a legitimate alternative and a good exercise, but at n ~ 1,000-1,500 elements a
serial host loop finishes in microseconds; this project's teaching budget goes to the radius/KNN
SEARCH pattern instead — GPU reduction is taught in depth elsewhere (08.01's softmin weight blend,
23.01's costmap reductions).

## Numerical considerations

**Precision.** All device-side arithmetic is FP32 (`float`), matching the CPU reference's `float`
computation — the repo-standard choice for LiDAR-scale coordinates (tens of meters, well within FP32's
~7-decimal-digit precision budget with room to spare).

**Determinism and the SOR tie-break.** SOR's K-nearest search must produce the IDENTICAL set of K
neighbors on the GPU and the CPU for the mean-distance comparison to be meaningful (not merely
"close by luck"). `kernels.cuh`'s `dist_less` fixes a total order — smaller squared distance wins; on
an EXACT tie, the smaller point index wins — the same tie-break discipline projects 02.05/02.09 use
for their own K-nearest searches, reimplemented here for this project's much smaller K (8, versus
those projects' spatially-indexed searches over millions of points). Because both the GPU kernel and
the CPU reference scan candidates `j = 0..n-1` in the SAME order under the SAME total order, the
selected K-set — and, generally, its exact sum — agree closely; `main.cu`'s VERIFY stage uses a tight
(not bit-exact) `1e-3` m tolerance rather than claiming bit-identical output, because `sqrtf()` on the
GPU and `sqrtf()`/`std::sqrt` on the host are both IEEE-754-conformant but not GUARANTEED bit-identical
across every compiler/architecture pair — the honest way to compare (the measured worst-case deviation
in "Expected output," `9.5e-07` m, illustrates just how tight that agreement actually is in practice).

**Why DROR/LIOR's neighbor counts verify EXACTLY (integers, zero tolerance).** A neighbor count is an
INTEGER built entirely from `<=` comparisons of squared distances against a squared radius — no
accumulation, no summation order to disagree about. The GPU and CPU each independently evaluate the
SAME `dx*dx+dy*dy+dz*dz <= radius^2` expression per candidate pair; IEEE-754 comparison operators are
bit-exact across conformant implementations, so the two counts either agree exactly or reveal a real
bug (an indexing error, a boundary-condition mistake) — `main.cu`'s VERIFY stage measured 0 mismatches
across 1,074 points for both filters (README "Expected output"), the expected outcome, not generous luck.

**The classify-stage verify trick (why it is exact by construction, not by luck).** `main.cu`'s
classify-stage comparisons (SOR/DROR/LIOR) feed the SAME already-verified statistic array to BOTH the
GPU classify kernel and the CPU classify function, using the SAME host-computed threshold scalar.
Given byte-identical inputs and a deterministic `>`/`<` comparison, the two paths cannot disagree —
this is project 02.13's "ledger-then-classify" precedent, reapplied here three times. It tests the
CLASSIFY logic in isolation from any statistic-stage floating-point drift, which is exactly what a
useful verify gate should isolate.

**Range-dependent hazards specific to this project.** DROR's `r_search(r)` formula divides no
quantity by `r` (it MULTIPLIES), so there is no singularity at `r -> 0`; the `r_min` floor exists
purely so a point extremely close to the sensor still gets a sane, nonzero search radius rather than
one that shrinks toward zero. LIOR's partial-interception intensity model (`scripts/make_synthetic.py`)
floors `range_m` at 0.5 m before computing the beam footprint area specifically to avoid a
divide-by-near-zero spike in the intensity formula at very short range — a numerical, not physical,
safety floor, documented at its source.

**When DROR's own assumption breaks — the dust plume core.** DROR's search radius scaling
assumes real-surface point spacing is the RELEVANT comparison. A sufficiently DENSE scatterer field —
this project's tuned dust plume (`scripts/make_synthetic.py`'s `DUST_DENSITY_PER_M3` comment documents
the tuning) — can produce enough nearby scatterer points that a scatterer point's OWN neighbor count
crosses `k_min` too, i.e. the scatterer field starts to statistically resemble a coherent surface.
Measured honestly (the `dust_plume_honesty` gate, never floor-gated by design): at this project's
tuned density, DROR still discriminates reasonably well (precision 98.2%, recall 94.6% inside the
plume core) — but LIOR does WORSE there (precision 98.8%, recall only 63.3%), a genuinely non-obvious
result. The reason: LIOR's FIXED companion radius (0.35 m) is LARGER than DROR's own range-scaled
radius at the plume's short range (3-7 m, where `r_search(5) = max(3*0.0175*5, 0.05) ~= 0.26` m) — so
as scatterer density rises, LIOR's fixed, comparatively generous radius accumulates "enough nearby
points to look dense" SOONER than DROR's tighter, physically-derived radius does. This is exactly the
kind of parameter interaction that only shows up when you MEASURE, not assume — and it is the honest
answer to "does LIOR simply win where DROR fails," which this project's own numbers say no to.

## How we verify correctness

The CPU reference (`src/reference_cpu.cpp`) independently retypes every STATISTIC and CLASSIFY
function from scratch (the "twin, not shared" ruling `reference_cpu.cpp`'s file header states in
full) — only pure formula bookkeeping (`squared_distance3`, `range3`, `dist_less`, `dror_search_radius`)
is shared token-for-token via `kernels.cuh`'s `HD` (host-and-device) functions, because sharing a
four-line formula is transcription, not the algorithm under test; the actual O(n) search LOOPS are
independently written in both files. `main.cu`'s VERIFY stage runs, on the SNOW scan (n=1,074 points,
representative — no kernel branches on which weather scan it is fed):

1. **SOR mean-KNN-distance**, GPU vs. CPU, tolerance `1e-3` m (worst case measured `9.5e-07` m — "Numerical considerations" explains why this is tight, not bit-exact).
2. **SOR classify**, given the SAME (CPU-computed) mean-distance array and threshold fed to both paths — exact.
3. **DROR neighbor count**, GPU vs. CPU, exact integers (0 mismatches measured).
4. **DROR classify**, given the SAME (GPU-computed, already-verified-exact) count array — exact.
5. **LIOR neighbor count**, GPU vs. CPU, exact integers (0 mismatches measured).
6. **LIOR classify**, given the SAME count array and shared intensity input — exact.

Every downstream gate (README "Expected output") then uses the now-certified GPU pipeline, run on ALL
THREE scans — normal production use of verified kernels, not a second verification pass (the same
reasoning project 02.13's file header states for its own "secondary analysis" stage: re-deriving a
second, independent GPU pass over already-certified code exercises no new code path).

**Edge cases exercised:** the near-sensor floor (LIOR's intensity model, DROR's `r_min`), the dense
dust-plume core (both filters' failure mode measured honestly, above), and the far-range sparse-real
cohort (`sor_far_range_failure` — SOR's designed failure). This project is fully deterministic (no
stochastic algorithm at runtime — all randomness is baked into the committed data file once, by
`scripts/make_synthetic.py`'s fixed-seed generator), so no statistical/seeded-comparison strategy is
needed for the filters themselves.

## Where this sits in the real world

**DROR** (Charron, Phillips & Waslander 2018) is the real, published algorithm this project's DROR
filter reimplements from the paper's own formula — production and research LiDAR-preprocessing
pipelines that target snow (Autoware.Universe's `pointcloud_preprocessor`, several published
autonomous-driving snow-removal studies) use DROR or a close descendant as a standard baseline.
**LIOR**, as implemented here, is this project's OWN teaching version of the broader intensity-based
weather-filtering family — real systems in this family (and DSOR, Kurup & Bos 2021, README "Prior
art") typically combine an intensity signal with SOME density/geometric signal, exactly the
"dim AND sparse" structure this project's LIOR uses, though production tunings and exact formulas
vary by sensor and target weather condition.

**Real snow-LiDAR datasets** — CADC (Canadian Adverse Driving Conditions) and WADS (Winter Adverse
Driving dataSet) — are the datasets a production team would validate against; this project's synthetic
scene is not a substitute for that validation, only a controlled, fully-labeled teaching environment
(README "Data" states why neither is used directly here).

**What a production all-weather stack does differently:**

- **Spatial acceleration.** At full LiDAR point counts (60,000-300,000+ points/scan), brute-force
  O(n^2) search is not viable; production systems build a k-d tree or uniform grid per scan (project
  02.05/02.09's territory) and turn every neighbor query into O(log n) or better.
- **Sensor-level physics beyond this project's scope.** Real backscatter cross-sections depend on Mie
  scattering theory at the sensor's specific wavelength (905 nm or 1550 nm), particle size
  DISTRIBUTIONS (not a single representative radius, as this project uses per weather type), and
  polarization effects some sensors exploit specifically to discriminate weather from solid targets.
  This project's illustrative constants (dated 2026-07-12, README "Limitations") are order-of-magnitude
  teaching values, not a substitute for that deeper physics.
- **Dual-return exploitation.** A real dual-return sensor can report both a scatterer echo and an
  attenuated surface echo for the same beam — extra information this project's single-return scope cut
  (README "Limitations") does not model; a production system that has dual returns available uses them
  as an additional, often very strong, weather-vs-real signal.
- **Adaptive/learned filtering.** Some modern production and research systems learn a
  weather-classification model (per-point or per-scan) rather than relying purely on hand-tuned
  geometric/intensity thresholds — this project's fixed-threshold approach is the classical,
  fully-interpretable baseline that any learned approach is measured against.
- **Fusion across modalities.** A full autonomy stack rarely trusts LiDAR-only weather filtering in
  isolation — camera and radar (which respond very differently to precipitation) provide corroborating
  or contradicting evidence that a production perception stack fuses in (project 04's territory).
