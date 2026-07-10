# 13.03 — Foothold scoring kernels: slope, roughness, edge distance from elevation maps: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why a foot needs more than "is the ground there?"** A wheel distributes its weight continuously
along a rolling contact patch; a legged robot's foot touches the world at one small, discrete point,
chosen once per step and then trusted for the whole stance phase. Choosing badly is not a comfort
problem, it is a **stability** problem: a quadruped's support polygon (the convex hull of its planted
feet, projected onto the ground) must contain the robot's center of mass projection (static stability)
or its zero-moment point / capture point (dynamic stability) or the robot falls. A single bad foothold
— one that slips, crumbles, or simply is not where the controller thought it was — can pull the whole
support polygon out from under the robot in under 100 ms (the same 0.5-1 kHz whole-body-control
deadline SYSTEM_DESIGN §1.1 marks as *hard*: miss it and the robot falls). This project computes,
per grid cell, the three physical quantities a real foot placement depends on before a controller ever
gets to plan a trajectory to it.

**The friction cone — where the hard slope limit comes from.** Treat the foot-ground contact as a
point subject to Coulomb friction: the ground can push back on the foot with a normal force `f_n`
(perpendicular to the local surface) and a tangential force `f_t` (along the surface) bounded by
`|f_t| <= mu * f_n`, where `mu` is the friction coefficient between the foot pad and the ground.
Consider a foot resting on a slope of angle `theta` from horizontal, supporting weight `m*g` with no
other forces (the simplest, most conservative case — a real standing/walking robot's other joint
torques and momentum add to this, which is exactly why production stacks build in additional margin,
README §Limitations). Resolving gravity into components normal and tangential to the slope:

```
f_n = m*g*cos(theta)      (pushes the foot into the ground)
f_t = m*g*sin(theta)      (pulls the foot down the slope)
```

The foot holds without slipping exactly when `f_t <= mu*f_n`, i.e. `m*g*sin(theta) <= mu*m*g*cos(theta)`,
i.e. `tan(theta) <= mu`. The steepest slope a foot with friction coefficient `mu` can stand on,
therefore, is

```
theta_limit = atan(mu)
```

— a **hard physical limit**, not a tuning knob, and exactly the reason `fusion_kernel`
([`src/kernels.cu`](src/kernels.cu)) treats "slope past this limit" as an absolute veto (score forced
to `0.0f`) rather than a penalty that merely lowers the score. This project uses `mu = 0.6` (a
rubber-like foot pad on packed dirt or concrete — an illustrative, dated value, see
[`PRACTICE.md`](PRACTICE.md) §2), giving `theta_limit = atan(0.6) = 0.5404 rad = 30.96 deg` — the
exact number the demo prints on its `PROBLEM:` line every run.

**The engineering reality of a real elevation map.** The height grid this project consumes is never
ground truth in a fielded system: it is built by fusing noisy depth measurements over a moving,
vibrating robot, and it inherits at least three characteristic pathologies that this project's
synthetic terrain deliberately reproduces so the kernels have something honest to be tested against:
*noise* (every real depth sensor has a few millimeters of range jitter — this project's smooth ripple
field stands in for it, chosen deterministic and analytically bounded specifically so THEORY can state
an exact worst case rather than an empirical one, below); *drift* (a moving robot's own pose
uncertainty smears the map over time — not modeled here, a genuine limitation, README §Limitations);
and *holes* (specular surfaces, occlusion behind an obstacle, or a sensor simply running out of range
leave cells with **no measurement at all** — this project's rectangular NaN "hole" stands in for that).
A kernel that quietly treats a hole as height-zero would confidently plan a foot onto a cliff it never
actually measured; §Numerical considerations below is the discipline this project uses to make sure
that never happens.

## The math

**Problem statement.** Given a row-major height grid `height_m[row*W+col]` (meters, NaN = unknown,
`kGridW=kGridH=256`, `kCellM=0.02` m — the layout contract in
[`src/kernels.cuh`](src/kernels.cuh)), compute per cell: a slope `slope_rad in [0, pi/2)`, a roughness
`roughness_m >= 0`, an edge distance `edge_dist_m in [0, kEdgeSearchRadiusCells*kCellM]`, and a fused
score `score in [0,1]`. Then, given a list of query points `(x_m, y_m)`, return for each the best
cell within a fixed search disc.

**Least-squares plane fit.** For a cell at `(row,col)`, gather every non-NaN neighbor in the
`(2*kFitRadius+1)^2` window (5x5 for `kFitRadius=2`), with cell-centered local coordinates
`x_i = dc*kCellM`, `y_i = dr*kCellM` (dc,dr the neighbor's column/row offset) and height `z_i`. Fit
`z = a*x + b*y + c` minimizing the sum of squared residuals

```
E(a,b,c) = sum_i (z_i - a*x_i - b*y_i - c)^2
```

Setting `dE/da = dE/db = dE/dc = 0` gives the **normal equations** — the standard "design-matrix
transpose times design matrix" result for linear least squares (with design matrix rows `[x_i, y_i, 1]`):

```
| Sxx  Sxy  Sx | |a|   |Sxz|          Sxx = sum x_i^2      Sxy = sum x_i*y_i    Sx = sum x_i
| Sxy  Syy  Sy | |b| = |Syz|          Syy = sum y_i^2      Sy  = sum y_i        Sz = sum z_i
| Sx   Sy   n  | |c|   |Sz |          Sxz = sum x_i*z_i    Syz = sum y_i*z_i    n  = sample count
```

`src/kernels.cu`'s `solve_plane_3x3` solves this by Cramer's rule (three 3x3 determinants over one —
see its header comment for the full derivation and why a hand-written solve, not a library, is the
right size here).

**Slope from the fitted plane.** The plane's steepest-ascent direction has horizontal unit vector
`(a,b)/sqrt(a^2+b^2)` and a directional derivative (rise per unit horizontal run) of exactly
`sqrt(a^2+b^2)` — the gradient magnitude. The angle that steepest-ascent line makes with horizontal is
therefore

```
slope = atan( sqrt(a^2 + b^2) )
```

(equivalently, the angle between the plane's normal `(-a,-b,1)` and vertical `(0,0,1)` — both give the
identical `cos(slope) = 1/sqrt(a^2+b^2+1)`; the code uses the `atan(gradient)` form because `atanf`
stays well-conditioned as `a,b -> infinity`, where the `acos` form's derivative blows up).

**Roughness from the fit's residuals.** With `(a,b,c)` known, `roughness_m = sqrt( (1/n) * sum_i (z_i -
a*x_i - b*y_i - c)^2 )` — the population standard deviation of the plane fit's residuals (§Numerical
considerations explains the population-vs-sample-statistic choice).

**The friction-cone slope limit**, derived in §The problem: `slope_limit_rad = atan(kFrictionMu)`.

**Fusion.** With `slope_score = clamp(1 - slope/slope_limit, 0, 1)`, `rough_score =
clamp(1 - roughness/kRoughnessMaxM, 0, 1)`, `edge_score = clamp(edge_dist/kEdgeSafeDistM, 0, 1)`:

```
score = 0                                                        if height is NaN, OR
                                                                   slope is NaN (degenerate fit), OR
                                                                   slope > slope_limit_rad
       = w_slope*slope_score + w_rough*rough_score + w_edge*edge_score     otherwise
```

with `w_slope=0.4, w_rough=0.3, w_edge=0.3` (sum to 1.0 — a convex combination, so `score` stays in
`[0,1]` by construction whenever the veto does not fire).

## The algorithm

Per elevation map (the four map-scale kernels, each `O(W*H)` cells):

1. **Slope + roughness** (one `solve_plane_3x3` per cell): gather up to `(2k+1)^2` neighbors (`k =
   kFitRadius = 2`, so up to 25 — fewer at a hole's edge or the map boundary), solve for `(a,b,c)`,
   derive slope, then a **second** window pass computes the residual sum for roughness (two passes
   because roughness needs the plane the first pass just solved for — THEORY can't compute a
   residual against an unknown plane). **Complexity:** `O(W*H*k^2)` serial; **fully parallel** across
   cells (§The GPU mapping) since no cell's fit depends on another cell's *output*, only on shared
   *input* heights.
2. **Edge distance** (one bounded search per cell): classify every cell hazardous or not (unknown
   height, degenerate/unknown fit, slope past the limit, or roughness past `kRoughnessMaxM`), then for
   every non-hazard cell, brute-force-search a `(2R+1)^2` window (`R = kEdgeSearchRadiusCells = 10`,
   441 cells) for the nearest hazard, clamped to a circular disc and capped at `R*kCellM` if none is
   found. **Complexity:** `O(W*H*R^2)` serial (the most expensive kernel in the pipeline — measured
   ~40 ms on one CPU core at this map's size, README §What this computes); embarrassingly parallel
   across cells, same reasoning as step 1.
3. **Fusion** (pure per-cell arithmetic, no window): `O(W*H)`, trivially parallel — the four inputs
   (height, slope, roughness, edge distance) are already sitting in memory from steps 1-2.
4. **Foothold selection** (one bounded argmax per query): for each of `N` queries (this demo: 1000),
   walk a `~(2R'+1)^2` disc (`R' = ceil(kFootholdSearchRadiusM/kCellM) = 5`, so up to 121 cells,
   `kFootholdSearchRadiusM = 0.10 m`) of the fused score grid, keep the best (deterministic
   raster-order, strict-`>` tie-break — §The GPU mapping and §How we verify correctness explain why
   the tie-break rule is load-bearing, not cosmetic). **Complexity:** `O(N*R'^2)`, independent of `W*H`
   once the score grid exists — a completely different *shape* of parallel problem from steps 1-3
   (batched search over a small, fixed data structure, vs. a dense map over every cell).

## The GPU mapping

```
Steps 1-3 (per-cell kernels): 2-D launch, thread (col,row) owns cell (col,row)
    block = 16x16 = 256 threads (warp-friendly square tile)
    grid  = ceil(W/16) x ceil(H/16)  =  16x16 blocks exactly at 256x256

Step 4 (per-query kernel): 1-D launch, thread q owns query q
    block = 256 threads, grid = ceil(N/256)  (4 blocks for N=1000)
```

**Why 2-D for the map kernels, 1-D for selection?** The map kernels' natural index IS a 2-D `(col,
row)` pair — CUDA's `dim3` grid/block exists exactly for this, and writing `blockIdx.x/y,
threadIdx.x/y` keeps the kernel's indexing math a direct mirror of the row-major array math
(`row*kGridW+col`) instead of a derived `q -> (row,col)` unpacking. The selection kernel's natural
index is a flat query list, so 1-D (the same shape as 08.01's "thread = rollout") is the natural fit.
Contrast this deliberately with 09.01's `__constant__` broadcast and 07.09's divergent global reads —
this project's third point on that same "how threads read shared data" spectrum: **windowed, mostly
coalesced global reads**, discussed next.

**Memory hierarchy: global only, no shared memory — and why not (yet).** Every kernel here reads
`height_m`/`slope_rad`/etc. straight from global memory, with heavy REDUNDANT re-reads: in the
slope/roughness kernel, each of a 5x5 window's cells is re-read by up to 25 *different* threads (every
thread whose own window overlaps that cell); in edge-distance, up to 441x. A shared-memory TILED
version would have each thread block cooperatively load its patch of the grid (plus a
`kFitRadius`/`kEdgeSearchRadiusCells`-wide "halo" of neighbor cells) into shared memory ONCE, then have
every thread in the block read from that fast on-chip copy instead of global memory repeatedly — the
textbook stencil/convolution optimization, and genuinely the next thing to build here (README
Exercise 4). It is not built in this teaching version for the same reason 08.01's blend stays on the
host: at this map's size (65536 cells, sub-millisecond kernels already), the L2 cache absorbs most of
the redundancy in practice, and the UNTILED version keeps the thread-to-data mapping — the concept this
project exists to teach — visible without shared-memory index bookkeeping standing in front of it.

**Occupancy contrast, by design.** The three map kernels launch 65536 threads (256 blocks of 256 at
minimum SM occupancy multiples, filling an RTX 2080 SUPER's 46 SMs many times over); the selection
kernel launches only `N=1000` threads (4 blocks) — a WIDE, CHEAP kernel next to three MEDIUM,
work-heavier ones, on purpose: a real gait planner never needs a million foothold queries per tick,
only as many as it has swing legs and candidate footholds under consideration. Occupancy at this scale
is intentionally low; README Exercise 5 asks you to reason about whether a warp-per-query
re-mapping would even help here (it would not, until `N` grows by orders of magnitude — the classic
"more blocks, not fewer, cheaper threads" lesson small batched kernels teach).

**No atomics, no divergence beyond loop bounds.** Every kernel's only branching is window-boundary
clipping (`if (row<0 || ...) continue`) and the veto/degenerate-fit early returns — data-independent
across threads in the sense that matters for warp efficiency (neighboring cells in a warp almost
always take the same branch, since terrain features span many cells, not single ones).

## Numerical considerations

- **FP32 throughout**, matching the repo default; heights are O(0.01-0.3 m), so absolute float32
  precision (~1e-7 relative) is many orders below anything these thresholds care about.
- **Near-degenerate plane fits.** `solve_plane_3x3` refuses to solve (returns `false`, propagating
  `NaN` slope/roughness) when `n < 3` (fewer than 3 samples cannot determine a plane's 3 free
  parameters) or `|det(M)| < 1e-9` (numerically collinear survivors — e.g. a window clipped down to a
  single row or column of neighbors near two adjacent map edges). Both are treated identically by
  every downstream consumer: a `NaN` slope is one of `fusion_kernel`'s two hard-veto conditions and
  one of `is_hazard_cell`'s four hazard conditions — "I could not certify this cell's geometry" is
  itself treated as unsafe, never silently ignored.
- **NaN propagation discipline.** A hole cell's *own* NaN height propagates immediately (no fit is
  even attempted). A *neighbor's* NaN height is **excluded from the fit's sums entirely** (`if
  (isnan(zi)) continue;` inside the gather loop) — never substituted with zero. Zero would invent a
  cliff at the hole's boundary that was never actually measured; exclusion instead simply shrinks the
  window's sample count (down to the map-edge-clipping `n<3` guard above, in the worst case). This is
  the single discipline point THEORY.md's "no black boxes" spirit asks every reader to internalize
  before touching any real elevation-mapping code.
- **Angle handling.** `slope_rad in [0, pi/2)` by construction (`atan` of a non-negative argument) —
  no periodic wrapping is possible or needed here (contrast CLAUDE.md §12's `(-pi,pi]` convention,
  which governs orientation states like the pole angle in 08.01, not a one-sided geometric quantity
  like this one).
- **Population vs. sample statistic for roughness.** Dividing the residual sum of squares by `n`
  (population std-dev) rather than `n - 3` (the unbiased "residual degrees of freedom" estimator that
  accounts for the 3 fitted parameters) understates roughness by a factor of `sqrt(n/(n-3))` — about
  7% at `n=25`, the common case. This is a stated simplification: the thresholds this feeds
  (`kRoughnessMaxM = 0.02 m`) have far more slack than 7% built into their choice, and the *n*-divisor
  keeps both the GPU kernel and the CPU oracle trivially, visibly identical formulas.
- **Two REAL numerical bugs this project shipped and caught — read this section, it is the point of
  having two independent verification layers (§How we verify correctness expands on both):**
  1. *A variable-shadowing bug that neither GPU-vs-CPU tolerance check caught.* An early draft named
     the plane-fit intercept `c` (from `z = a*x + b*y + c`) — and a few lines later, inside the
     residual-computation loop, ALSO named the loop's per-cell column index `c` (`const int c = col +
     dc`). C++ scoping rules mean the inner loop's `c` **shadowed** the outer `c`: the residual line
     `zi - (a*xi + b*yi + c)` silently used the *column index* (an int, ~10-250, implicitly converted
     to float) instead of the intercept (~0.002 m). The measured symptom: roughness came out around
     **25 meters** instead of a few tenths of a millimeter — but because `kernels.cu`'s kernel and
     `reference_cpu.cpp`'s oracle are DELIBERATE, line-by-line duplicates (CLAUDE.md §5), the SAME
     copy-paste mistake existed in both, so their cross-check disagreement (the GPU-vs-CPU VERIFY
     gate) measured a difference of only ~4.6e-5 m — comfortably inside a naive tolerance, because
     *both sides were wrong in nearly the same way*. The bug was caught only by an ANALYTIC gate
     against terrain ground truth (the flat-control region's mean fused score measured 0.39, nowhere
     near the >0.95 an almost-perfectly-flat region should score) — the exact failure mode this
     project's two-layer verification strategy exists to catch. Both files now name the intercept
     `c0` and the neighbor column index `nc`, with a comment at the fix site explaining why.
  2. *An FMA-vs-non-FMA rounding asymmetry the GPU-vs-CPU gate DID catch.* The foothold-selection
     kernel's disc-membership test, `ddx*ddx + ddy*ddy > rad2_m`, is exactly the shape a compiler can
     fuse into a single `fma(ddy,ddy,ddx*ddx)` — nvcc does this by default on the GPU; MSVC's
     `/fp:precise` (this repo's default, CLAUDE.md §5 — reproducible floats over `--use_fast_math`)
     does not fuse the same way on the CPU. The fused and unfused computations round differently by a
     few ULPs, which can — for a cell sitting almost exactly on the disc boundary — flip whether that
     cell is visited AT ALL, not merely rescore it. Measured: 1 of 1000 queries picked a genuinely
     different cell (score differing by 3.2e-3, not a rounding-scale amount) between the two paths.
     The fix (both `kernels.cu` and `reference_cpu.cpp`, identically): inflate the disc test by a
     fixed `1e-6f` epsilon, `> rad2_m + kDiscEps`, wide enough to absorb the FMA-rounding gap and
     narrow enough (~0.05 mm effective radius change) to be geometrically meaningless. After the fix,
     all 1000 selections match their CPU oracle's cell index EXACTLY (not just within tolerance) —
     see §How we verify correctness for why exactness is the right bar for this specific stage.

## How we verify correctness

**Two independent layers, because this pipeline can fail in two different ways a single check would
miss.** A four-kernel PIPELINE (unlike 08.01's single rollout kernel or 07.09's single JFA kernel) adds
a failure mode neither of those single-kernel projects has to worry about: errors COMPOUNDING across
stages, and — as the numerics section's bug #1 shows — a bug that is IDENTICAL in both the "independent"
GPU and CPU implementations, which a same-input-both-sides diff cannot see by construction. This
project's answer is two layers that catch different things:

1. **VERIFY — four STAGE-ISOLATED GPU-vs-CPU gates.** Rather than run the full GPU pipeline and the
   full CPU pipeline end-to-end and diff only the final score (which would let a boundary-cell
   disagreement in an early stage — e.g. a hazard classification that flips between the two paths due
   to ordinary float rounding — cascade into a much larger, harder-to-attribute difference several
   stages later), `main.cu` feeds each kernel-under-test and its CPU oracle the **identical** upstream
   arrays (the CPU oracle's own prior-stage output, uploaded to the device for the GPU kernel). Stage 1
   (slope/roughness) is the only stage whose two paths perform genuinely independent floating-point
   arithmetic on the same raw input (measured worst case: 1.4e-6 rad slope, 1.5e-8 m roughness, 0 NaN
   pattern mismatches over 65536 cells). Stages 2-3 (edge distance, fusion) are pinned to bit-identical
   inputs and — being pure comparisons/arithmetic with no reduction across many terms — measure at or
   near exact agreement (0 and 1.19e-7 respectively). Stage 4 (selection), also pinned, is held to an
   **exact** cell-index match across all 1000 queries — achievable, and required, specifically because
   its inputs are pinned identical floats and its only operations are comparisons and an argmax, not
   new arithmetic that could round differently between the two compilers (numerics bug #2 above is
   exactly the counter-example that motivated tightening this gate to exact-match in the first place).
2. **Four ANALYTIC gates against the terrain's own known ground truth**, run on the real, all-GPU,
   end-to-end pipeline (never mixed with CPU numbers): the flat control region's slope must sit under
   an analytic bound derived from the background ripple's own closed-form worst-case gradient
   (`amplitude * 2*pi/wavelength * sqrt(2) = 0.002 * 2*pi/0.5 * sqrt(2) ≈ 2.04 deg`, measured actual:
   1.30 deg, well inside a 3.4 deg margin that absorbs the plane fit's own smoothing); the ramp's
   measured mean slope, sampled one fit-window-radius in from its own transition edges, must track the
   RECIPE's constructed 15.00 deg angle (measured: 15.01 deg, tolerance ±1.5 deg); the step's edge cell
   must be hard-vetoed to exactly 0 while a cell placed beyond the fit-window-plus-search-radius reach
   from the step must show its edge distance SATURATED at the search cap (measured: exactly 0.0000 and
   0.2000 m respectively); and every one of 1000 foothold selections must be both valid (score above
   `kValidThreshold`) and within its search disc (measured: 1000/1000 on both counts). This layer is
   what caught numerics bug #1 — a bug the cross-implementation diff, by its very construction, could
   not see.

The terrain is entirely synthetic and entirely known (README §Data) SPECIFICALLY so this second layer
is possible — a real, recorded elevation map has no ramp angle to check a measured slope against.

## Where this sits in the real world

- **ETH Zurich's `elevation_mapping`/`grid_map` ecosystem** (and its commercial descendants at
  companies like ANYbotics) is the closest production analogue: a fused, variance-aware 2.5-D grid with
  traversability layers computed on top, running on real quadrupeds today. The biggest gap from this
  teaching core: **per-cell height UNCERTAINTY**, propagated from sensor noise models through the map
  fusion and INTO the traversability scoring (a high-confidence flat cell and a low-confidence flat
  cell should not score identically) — this project treats every known height as exact (README
  Exercise 3 sketches the extension).
- **MIT Cheetah-family controllers** favor cheaper, more heuristic foothold scoring evaluated at much
  higher rates, trading this project's full per-cell least-squares fit for speed — a legitimate design
  point at the opposite end of the fidelity/latency trade this project's ~1 ms GPU budget sits well
  inside of.
- **Boston Dynamics Spot / ANYbotics ANYmal** (public materials only — internals are proprietary) ship
  this general capability class in fielded, commercial quadrupeds; their published work discusses
  learned/semantic terrain classification (identifying grass vs. gravel vs. ice by appearance, not just
  geometry) as a layer ABOVE geometric scoring like this project's — a fusion of perception (domains
  01-03) and this project's geometric layer that production stacks increasingly build.
- **The gait-planning integration** (this project's own honest gap, README §Limitations): a production
  footstep planner does not fix a nominal point and search a small disc around it — it searches a
  MUCH larger candidate region JOINTLY with gait timing and centroidal dynamics (13.02's Centroidal
  MPC, SYSTEM_DESIGN §4.3's Chain C), often evaluating thousands of candidate footholds per leg per
  planning cycle. This project's fixed nominal-point-plus-small-disc search is the simplest version of
  that search that still demonstrates the GPU pattern (a batched argmax) the full version scales up.
- **What the full version adds beyond this teaching core:** uncertainty-aware fusion, learned/semantic
  terrain classification fused with geometry, joint foothold-and-gait optimization instead of a fixed
  nominal point, and — at the research frontier — foothold selection integrated directly into a
  learned locomotion policy (SYSTEM_DESIGN §4.3's offline sim-to-real RL loop, domains 10/12) rather
  than a separate geometric scoring stage at all.
