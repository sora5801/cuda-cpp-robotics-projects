# 02.13 — Dynamic point removal (raycast free-space carving): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**A LiDAR beam is a physical fact about a whole line, not just its endpoint.** A time-of-flight
LiDAR emits a pulse of light and measures how long it takes to come back: `range = c * t_round_trip
/ 2`, where `c` is the speed of light. The instrument reports one number — the distance to whatever
reflected the pulse — and most pipelines throw everything else away, keeping only the 3-D point
`P = origin + direction * range`. But the physics tells you more than that one point: for the pulse
to have traveled the FULL distance `range` before reflecting, **nothing could have been in its way
for the entire length of that path**. If something had been there, the pulse would have reflected
off it instead, and the reported range would have been shorter. A single beam is therefore evidence
of two different things at once: "something is here" (at the endpoint) and "nothing is *anywhere
between here and there*" (along the whole ray). Elfes' 1989 occupancy-grid insight — arguably the
single most important idea in mapping robotics — is exactly this: **free-space evidence is just as
real as occupied-space evidence, and a mapping system that only keeps hits is throwing away half of
what every sensor return actually told it.**

**Why this matters for MOVING objects specifically.** Consider a parked car that a robot's LiDAR
sees during scan 1, then drives away before scan 6. Scan 1's beams that hit the car recorded real
points — those points are not wrong, they were a faithful photograph of that instant. The problem is
temporal: a naive mapping system that just accumulates every scan's points into one cloud has no way
to know the car is gone by scan 6, because it never asked. But scan 6's beams, aimed at the same
physical location, now travel straight through it (nothing is there anymore) and either hit
something further away or return nothing at all — and BOTH of those outcomes are free-space evidence
for that exact spot. This project's whole method is: accumulate that free-space evidence across many
scans, and treat "this spot has been proven empty far more often than it was proven occupied" as
proof that whatever occupied it has moved.

**Engineering reality this project simplifies.** A real spinning LiDAR (e.g. a 16- or 32-beam
mechanical unit) fires beams at a fixed elevation table and a continuously rotating azimuth, at
range-dependent noise floors of 1-3 cm (this project models 2 cm, RANGE_NOISE_SIGMA_M in
`scripts/make_synthetic.py`), with a maximum range set by return-signal strength (weaker for dark or
grazing-incidence surfaces — a real effect this project does not model; every hit here is a clean
specular-enough return within `kMaxRangeM`). Real systems must also budget for the sensor's own
mounting vibration and thermal drift in range calibration; this project holds the sensor's mounting
geometry and calibration perfect (PRACTICE.md discusses what that costs in practice).

## The math

**Frames and units** (SI throughout; right-handed, x-forward/y-left/z-up world frame — CLAUDE.md
§12). Beam `i` fires from `scan_id[i]`'s sensor position `o = origin(scan_id[i])` (meters, world
frame) along a unit direction `d = dir[i]` (world frame — sensor orientation is identity throughout
this project's scenario, `kernels.cuh`'s file header). Its recorded endpoint, when it is a hit, is

```
P_i = o + d * range_i                                          (1)
```

**The voxel grid** is a dense array of cubic cells, edge length `L = kVoxelSizeM`, covering a world-
space box `[origin, origin + N*L)` per axis (`kGridOriginX/Y/Z`, `kGridNX/NY/NZ`). A world point `p`
maps to an integer voxel coordinate by

```
voxel_coord(p, origin, L) = floor((p - origin) / L)                         (2)
```

`floor`, not truncation toward zero, is required for correctness on the negative side of the grid —
the same pitfall project 02.01's `voxel_coord` documents in full (a naive `(int)` cast rounds a
negative `p / L` the WRONG way at the origin-relative sign boundary).

**The ledger.** For every voxel `v`, this project accumulates three non-negative integer counters
over all `N` beams in a batch (K scans):

```
hits[v]              = #{ beams whose HIT endpoint's voxel is v }
pass_from_hit[v]      = #{ (beam, voxel-visited) pairs where the beam eventually hit
                            something ELSE, but crossed v en route }
pass_from_maxrange[v] = #{ (beam, voxel-visited) pairs where the beam never hit
                            anything at all, but crossed v en route }
```

with two counting rules that make the ledger a well-defined partition of "what every beam's march
said about every voxel it visited":

- **Endpoint exclusion.** The voxel a beam HITS is counted in `hits`, never also in `pass_from_hit`
  for that same beam — a beam contributes exactly ONE increment total to its own endpoint voxel, and
  zero or more PASS increments to every OTHER voxel it crossed.
- **Max-range beams carve their own terminal voxel too.** A beam with no return marches the FULL
  `kMaxRangeM` and marks every voxel it crosses as PASS, *including* its nominal terminal voxel
  (there is no "hit" to exclude it from, since nothing was hit) — "beams that hit nothing still
  carve," the catalog bullet's own phrase.

**The classification ratio.** For a recorded point `P_i` landing in voxel `v = voxel_coord(P_i, ...)`,
define

```
passes(v) = pass_from_hit[v] + pass_from_maxrange[v]
score(v)  = passes(v) / (hits[v] + passes(v))          if hits[v] + passes(v) > 0, else 0    (3)
label(v)  = DYNAMIC  if score(v) >= kDynamicThreshold (0.6)
            STATIC   otherwise
```

**Why a ratio, and why THIS threshold's statistics matter.** `score(v)` is, informally, "what
fraction of the times a beam had an opinion about voxel `v`, that opinion was 'empty'." A voxel that
is genuinely permanent structure, observed `H` times as a hit and never legitimately passed through
(`passes(v) = 0`), scores exactly 0 — perfectly confident STATIC. A voxel a car occupied for exactly
ONE scan and that 9 later scans' beams then travel straight through scores `9/10 = 0.9` — confidently
DYNAMIC. The interesting (and, this project measures honestly, NOISY) regime is a voxel with a
SINGLE hit and a SINGLE incidental pass — from an unrelated beam merely grazing nearby — which scores
EXACTLY `0.5`, a coin flip between two single, low-confidence observations. This is precisely the
regime `kDynamicThreshold = 0.6` (rather than the naive `0.5` "simple majority") is chosen to move
away from: requiring evidence to be a *clear* majority, not a bare one, measurably reduces false
removals of once-observed static structure without costing any margin on the genuinely dynamic
cohorts, whose scores this project's committed scene measures at 0.7-1.0 (README "Expected output"
states the exact numbers, both at 0.5 and 0.6, from the design-iteration history).

**The DDA march (Amanatides & Woo, 1987) — derived from first principles.** A ray `o + t*d` crosses
voxel boundaries at specific parameter values `t`. Because the grid is axis-aligned, the boundary
crossings on each axis form an arithmetic sequence, and the algorithm tracks, per axis `a`:

```
step[a]   = sign(d[a])                          (which way the voxel index moves on this axis)
tDelta[a] = L / |d[a]|                           (how much t it takes to cross ONE voxel on this axis)
tMax[a]   = t at which the CURRENT voxel's boundary is first crossed on this axis
```

`tMax[a]` is initialized from the voxel the ray starts in: if `step[a] = +1`, the next boundary is
at world coordinate `origin[a] + (voxel[a]+1)*L`; if `step[a] = -1`, it is at `origin[a] +
voxel[a]*L`; solving `o[a] + t*d[a] = boundary` for `t` gives the initial `tMax[a]`. At every step of
the march, the algorithm advances along whichever axis has the SMALLEST `tMax` (that is the next
boundary crossing on ANY axis), updates that voxel index by `step[a]`, and adds `tDelta[a]` to that
axis's `tMax` so the next comparison stays fair. This visits every voxel the ray passes through, in
order, using only comparisons and additions — no per-step `sqrt`, no resampling, and (this project's
own numerical contribution, not in the 1987 paper, which assumes an exact analytic stopping distance)
an INTEGER stopping condition: the target voxel is computed directly from the endpoint via (2), and
the march simply runs until the current voxel EQUALS the target voxel, rather than comparing
accumulated `t` against a float distance (see "Numerical considerations" for why this matters).

## The algorithm

Step by step, per beam (`kernels.cu`'s `carve_one_beam` / `reference_cpu.cpp`'s
`carve_one_beam_cpu` — two independent typings of the same steps):

1. Compute the START voxel from the beam's origin via (2). This voxel is NEVER marked (the
   self-carve guard — see "Numerical considerations").
2. Compute the TARGET voxel from the beam's endpoint `o + d * range` via (1)-(2), directly (not by
   marching).
3. If start == target (a near-zero-range degenerate beam), mark a hit if applicable and stop — no
   march needed.
4. Otherwise, initialize `step`/`tDelta`/`tMax` per axis (the math above) and march: repeatedly
   advance along the smallest-`tMax` axis, and at each new voxel, either (a) mark it HIT and stop
   (if it is the target and the beam is a real hit), (b) mark it PASS and stop (if it is the target
   and the beam is a max-range beam), or (c) mark it PASS and continue (any other voxel), with a
   grid-bounds check that stops the march early if the ray exits the local map before reaching its
   target (the honest behavior of a BOUNDED local map, not a bug).
5. Classification (a separate, later pass, `classify_kernel`/`classify_cpu`): for every recorded HIT
   point, recompute its own voxel via (2) and evaluate (3).

**Complexity.** Per beam, the march visits O(voxels crossed) ≈ O(range / L) voxels — for this
project's `range` up to 20 m and `L = 0.2 m`, at most ~100-300 voxels (bounded by `kMaxDDASteps =
300`, comfortable headroom over the grid's own diagonal). Serially, carving `N` beams costs
`O(N * range / L)`; this project's whole 28,800-beam, 10-scan carve measures ~13-15 ms single-
threaded on the reference machine (README "Expected output"). Classification is `O(N)` — one voxel
lookup per point, no march. In parallel across `N` independent beam-threads, the carve's wall-clock
cost is dominated by the SLOWEST beam's march (a few hundred steps) plus whatever serialization the
hottest few voxels' atomics impose (measured, not modeled, in the "GPU mapping" section below).

## The GPU mapping

**Thread-to-data mapping:** one thread per BEAM (`carve_kernel`), the same thread-per-independent-
problem shape as 08.01's rollouts or 09.01's forward-kinematics batch — here the "problem" is a
whole voxel march instead of an ODE integration or a matrix chain. Beams never read or write each
other's data except through the shared ledger.

**Memory hierarchy:**
- **Registers** hold the entire march state per thread (`cx,cy,cz`, `tMaxX/Y/Z`, `tDeltaX/Y/Z`,
  `stepx/y/z`) — small, fixed-size, no spilling risk at this scale.
- **`__constant__` memory** holds the `kNumScans`-sized array of sensor origins
  (`set_scan_origins()`), read via `c_scan_origin[scan_id[i]*3 + ...]`. Because beams are stored
  scan-major (`kernels.cuh`'s file header "BEAM RECORD LAYOUT"), most warps' 32 threads share the
  SAME `scan_id`, making this a broadcast read served at essentially zero extra cost — the same
  "everyone reads the same bytes, put it in constant memory" reasoning 02.08's trajectory upload and
  09.01's model-parameter upload both give.
- **Global memory, read side:** `scan_id[i]`, `dir[i*3+0..2]`, `is_hit[i]`, `range[i]` — each read
  exactly once per thread. `dir` is stored interleaved (matching the repo's `PointCloud` convention,
  02.01/02.06/`SYSTEM_DESIGN.md` §3.6) rather than split into `dx[]/dy[]/dz[]`; the honest cost is
  that a warp's three-float read is not as tightly coalesced as a split layout would give (README
  Exercise 5 asks the learner to measure the difference).
- **Global memory, write side (the atomics):** every voxel visited by every beam issues exactly one
  `atomicAdd` to one of `hits[v]`/`pass_from_hit[v]`/`pass_from_maxrange[v]`. This is the *scatter*
  regime — the same taxonomy entry 02.01's hash-table insert names for its own atomicCAS/atomicAdd
  pattern — chosen here over a two-pass count-then-write scheme because the total write volume
  (a few hundred thousand increments) is small and the access pattern is NOT uniformly hot: **atomic
  contention is structurally concentrated near the sensor.** Every beam in a scan starts at the same
  voxel and immediately fans out; in the first few marching steps a large fraction of that scan's
  ~2,880 beams are all still near that shared starting region, so near-sensor voxels receive
  atomicAdd traffic from a large fraction of the scan's beams, while a voxel 15 m away is typically
  touched by only the handful of beams whose specific direction threads that spot. Measured on this
  project's committed scene (README "Expected output"): voxels within 3 m of any sensor position
  average ~14 pass-increments each (one voxel peaks at 875), while voxels beyond 10 m average ~2 —
  roughly a 7x hotspot ratio. This is the SAME observation OctoMap's own documentation makes about
  its `insertPointCloud`: atomics-per-voxel is the right tool precisely BECAUSE the contended region
  is small relative to the whole grid, not despite it. An all-atomics design that was hot EVERYWHERE
  would instead want the shared-memory local-histogram trick projects 07.09/23.01 teach (README
  Exercise 4).
- **Divergence:** beams whose march is short (a nearby hit) finish long before beams that travel the
  full `kMaxRangeM` — an intrinsic workload imbalance for a variable-length algorithm on fixed-width
  SIMT hardware. It costs cycles (some lanes idle while others finish their march), never
  correctness: every beam still visits exactly the voxels ITS OWN march requires.
- **`carve_trace_kernel`** reuses the identical `carve_one_beam` device function with the ledger
  pointers replaced by `nullptr` and a trace-output buffer supplied instead — one implementation,
  two call sites, so the verify stage's instrumented march can never silently drift from the bulk
  carve's real one (`kernels.cu`'s file header).
- **`classify_kernel`** is a plain per-point *map*: three GATHER reads (`hits[v]`,
  `pass_from_hit[v]`, `pass_from_maxrange[v]`, scattered across the grid, no coalescing to exploit —
  the same honest stencil-adjacent-read story 05.01's marching-cubes corner loads tell) and one
  coalesced write each of `score_out`/`label_out`.

## Numerical considerations

**Determinism, and why this project can require EXACT (not tolerance-based) GPU/CPU agreement.**
The DDA march contains no transcendental function — only `floor`, multiply, add, divide, and
compare. Every multiply-add is spelled out as an explicit `fmaf()` (device) / `std::fmaf()` (host)
in the SAME order on both paths (the same determinism contract project 05.01's TSDF integration
uses), so the two paths execute IDENTICAL IEEE-754 operations and produce bit-identical voxel
sequences, ledger counts, and classification scores. This project's VERIFY stage therefore checks
exact equality throughout (README "Expected output": 0 mismatches on all three gates, worst score
deviation `0.000e+00`) — a stronger and more legible guarantee than the ~1e-3 relative tolerances
transcendental-function-heavy projects like 08.01 must accept.

**The self-carve guard.** The voxel containing the sensor is never marked PASS for any beam. Why
this is not merely a convenience: without it, EVERY one of a scan's beams would mark the sensor's own
voxel as PASS on its very first step, accumulating up to `kTotalBeams` (28,800) essentially
meaningless increments in one voxel — trivial to obtain, and telling you nothing, since the sensor's
own location is where the robot physically sits, not a location any OTHER measurement observed as
free. The march therefore only begins counting from the FIRST voxel boundary the beam actually
crosses.

**Integer stopping condition vs. float distance comparison.** A tempting alternative to computing
the target voxel directly (as this project does) is to accumulate `t` during the march and stop once
`t >= range`. The two are NOT equivalent at a voxel boundary: floating-point rounding in the
accumulated `tMax` sum can disagree by up to a few ULPs with a freshly-computed `range`, occasionally
selecting a DIFFERENT voxel than the one the endpoint actually lands in when the true crossing sits
exactly on (or very near) a boundary — a genuinely awkward bug class, because it manifests as an
off-by-one-voxel error only for a small, boundary-dependent fraction of beams. Comparing already-
decided INTEGER voxel coordinates (this project's choice) removes the ambiguity entirely: there is no
"nearly equal" for integers.

**Grazing incidence and grid-boundary alignment — the discretization false-positive story, derived.**
Consider a flat static surface (this project's wall) whose front face sits at world coordinate `y_0`.
If `y_0` happens to land EXACTLY on a voxel boundary (i.e. `(y_0 - origin_y) / L` is an integer), a
hit point on that surface, perturbed by the sensor's ± range noise `sigma`, lands in voxel
`floor((y_0 + noise - origin_y)/L)` — and because `noise` straddles zero, ROUGHLY HALF of all hits on
that exact surface location fall one voxel SHORT of the boundary (into the open-space voxel
immediately in front of the wall) purely by noise, not by any real occupancy change. That
"immediately in front of the wall" voxel is, geometrically, the FINAL-APPROACH voxel for essentially
every beam that ever reaches the wall from any direction — a voxel that legitimately accumulates
enormous PASS traffic. A hit accidentally landing there, alongside that huge pass count, scores
close to 1.0: a FALSE dynamic classification of a perfectly permanent wall. This project measured
this failure mode directly while building its demo (a first attempt at the scene put the wall's
front face at exactly `y = 7.8`, `origin_y = -16.0`, `L = 0.2` — a perfect boundary alignment — and
saw ~30% of otherwise-generic wall points falsely removed); the fix was to offset the ENTIRE VOXEL
GRID's origin by a small (0.07 m) amount that shares no common factor with the scene's round-number
geometry, so no surface lands on a boundary by coincidence (`kernels.cuh`'s `kGridOriginX/Y/Z`
comment derives this in full — the same lesson, from the opposite direction, as 05.01's "+0.5 centers
the voxel sample" TSDF comment: sampling AT a boundary is always the numerically fragile choice).

**Sub-voxel-thin geometry is a DIFFERENT, unavoidable discretization effect, kept as this project's
intentional honesty cohort.** The origin-offset fix above removes an ACCIDENT (a coincidental
alignment this project's own scene design created); it does nothing for a genuinely thin object like
the 4 cm-radius pole, which is smaller than a fifth of one 20 cm voxel edge in ANY grid alignment.
Any beam that geometrically misses the pole by even a few centimeters shares the pole's OWN voxel
with beams from other scans/angles that legitimately pass straight through open space right next to
it — there is no origin offset that fixes this, because the mismatch is between the OBJECT's physical
scale and the VOXEL's physical scale, not a coordinate coincidence. This project measures and reports
the resulting ~100% false-positive rate on the pole honestly (README "Expected output") rather than
hiding it by, say, enlarging the pole to be voxel-sized — the whole point of including it is to teach
that a voxel grid has a resolution floor, and objects thinner than that floor are simply unreliable to
carve, in any real system, not just this teaching one. A wall's free x-direction END (the wall_edge
cohort) sits between these two extremes: the wall is a large, easily-resolved target overall, but
right at its terminus, grazing rays from oblique viewing angles legitimately clip past the corner —
measured at 5.2%, meaningfully higher than the wall's generic-face rate but far below the pole's,
exactly the gradient the physical picture predicts.

**Precision.** All geometry and the ledger's ratio arithmetic are FP32; the ledger's COUNTS
themselves are exact 32-bit unsigned integers (no precision loss possible in an `atomicAdd(uint,1)`
accumulation, and no overflow risk at this project's scale — the largest single-voxel count measured
is 875, four orders of magnitude below `UINT32_MAX`).

## How we verify correctness

Three tiers, each catching a different bug class (the independence ruling in
`reference_cpu.cpp`'s file header explains why more than one tier is required):

1. **DDA trace exact** (a documented 48-beam subset: one max-range beam plus one hit per cohort,
   padded sequentially to reach the subset size) — the GPU's `carve_trace_kernel` and the CPU's
   `carve_trace_one_beam_cpu` (independently typed) must produce IDENTICAL ordered voxel-index
   sequences, entry for entry. This is the strongest possible check on the algorithm itself: it is
   blind to nothing the march touches.
2. **Hit/pass ledger exact** (the full 28,800-beam carve) — `carve_kernel` (GPU) vs. `carve_cpu`
   (CPU, independently typed) must produce IDENTICAL `hits`/`pass_from_hit`/`pass_from_maxrange`
   arrays, all 768,000 voxels × 3 counters. Order-independent by construction (integer addition
   commutes — the same reasoning project 02.02's counting kernels rely on), so this passes
   regardless of GPU thread-scheduling order.
3. **Classification exact given the ledger** — `classify_kernel` (GPU) vs. `classify_cpu` (CPU)
   must agree on every label exactly and every score within float headroom (measured `0.000e+00`
   worst deviation, since the ratio computation is pure integer-to-float division with no
   transcendental step).

These three tiers together prove the GPU and CPU implementations of the ALGORITHM agree — but per
the independence ruling, they cannot prove the algorithm ITSELF is measuring the right thing (a bug
shared by both independently-typed copies, e.g. an inverted comparison, would sail through all three
unnoticed). The project's FIVE GATES close that gap by comparing the FINAL classification against
GROUND TRUTH loaded straight from `data/sample/beams.csv`'s `truth_dynamic` column — data the
carving/classification code never sees:

- `ghost_removal` / `late_leaver` / `static_preservation` compare classification labels against
  object-identity ground truth (was this point ever part of something that moved?).
- `free_space_consistency` checks two ACCOUNTING invariants that hold by construction if (and only
  if) the ledger and classification code are internally consistent (no retained point sits in a
  `hits==0` voxel; `sum(hits)` exactly equals the count of hit beams) — an INDEPENDENT sanity check
  that routes through neither the CPU nor GPU twin's shared voxel-indexing helper in the way that
  matters (it re-derives each point's voxel from scratch and cross-checks against the ledger's own
  bookkeeping).
- `max_range_carving` checks a specific, designed sub-cohort (the isolated ghost, present only in
  scan 0) whose carving evidence, by the scene's geometry, MUST come almost entirely from
  `pass_from_maxrange`, not `pass_from_hit` — a targeted analytic check, not a GPU/CPU comparison.

Stochastic elements: only the SENSOR NOISE is randomized (fixed seed 42, xorshift32,
`scripts/make_synthetic.py`), baked once into the committed sample file — the C++ pipeline itself is
entirely deterministic given that file, so no seeded/statistical comparison strategy is needed at
runtime (contrast 08.01's per-tick noise, which must be regenerated and compared statistically).

## Where this sits in the real world

**OctoMap** (Hornung et al., 2013) is the direct production descendant of this project's ledger: a
probabilistic hit/miss occupancy octree, updated with the SAME "cast every beam, mark the endpoint
occupied and everything before it free" principle, but using a LOG-ODDS Bayesian update (accumulating
`log(p/(1-p))` per observation, clamped, thresholded) instead of this project's raw hit/pass ratio —
log-odds naturally handles an unbounded, streaming sequence of observations without this project's
implicit assumption of "K scans, then classify once," and degrades gracefully under sensor noise in a
way a bare ratio does not. **Removert** (Kim & Kim, 2020) and **ERASOR** (Lim et al., 2021) are the
modern LiDAR-SPECIFIC dynamic-point-removal literature: both compare a query scan against an
accumulated map using range-image or pseudo-occupancy representations rather than per-beam
raycasting, trading this project's per-beam exactness for the speed needed to clean city-scale maps
in seconds rather than this project's ms-scale toy grid. Dynamic-aware SLAM front ends (e.g.
DynaSLAM, DS-SLAM in the visual-SLAM literature; LIO-SAM-adjacent dynamic filters in the LiDAR
literature) fold a version of this removal decision directly into the mapping loop rather than as a
post-session batch pass, closing the loop this project's README "System context" names explicitly
(an uncleaned map corrupts the very localization used to build the next map). This project's
reduced scope relative to all of the above: a bounded local grid rather than an octree or streaming
representation, a batch rather than incremental update, and a raw ratio rather than a Bayesian
log-odds filter — each a deliberate teaching simplification, named honestly in README "Limitations."
