# 20.01 — GelSight/DIGIT processing: contact patch, shear field via optical flow, slip detection in real time: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Every symbol, unit,
> and frame is defined on first use.

## The problem — physics & engineering first

**How a vision-based tactile sensor turns touch into an image.** A GelSight-style sensor is, from the
outside, a small soft pad. Inside: a layer of clear, compliant **elastomer gel** (silicone rubber,
typically a few millimeters thick) bonded to a rigid, transparent backing; a **camera** looks up through
that backing at the gel's *inner* surface; and the gel's inner surface is lit from the side by internal
LEDs (production sensors typically use three or more colors from different directions specifically so
each color's brightness encodes surface tilt in a different direction — see "Where this sits in the real
world" below). When an object presses against the gel's *outer* surface, the gel **deforms to match the
object's shape** — it wraps around whatever touches it, the way a rubber glove wraps around a finger.
The camera does not see the object; it sees the gel's deformed *inner* surface, illuminated so that its
local geometry shows up as brightness and shadow. This is the whole trick: **touch becomes a geometry
problem, and geometry becomes a photometry problem a camera can already solve at 30-60 Hz.**

**Why the gel is printed with a dot grid.** A perfectly smooth gel surface deforming under shear
(tangential sliding) produces almost no *visible* signal at the pixel level — the same texture just
moves. GelSight and DIGIT sensors solve this the same way a wind tunnel visualizes airflow: print a
regular, high-contrast marker (a grid of dark dots) on the gel's surface, and every marker's own visible
motion becomes a direct, trackable sample of the surface's underlying displacement field. Without
markers, measuring shear would require the sensor to distinguish surface texture that has genuinely
moved from surface texture that only *looks* different because of shading changes — an ill-posed
problem for a low-textured, deforming, unevenly-lit gel. With markers, it collapses into "find N known
dots, see how far each one moved" — the sparse tracking problem this project's shear-field kernel
solves.

**The physical task, precisely.** Given a video stream of the gel's inner surface, recover, per frame:
(1) **which area is in contact** and how much force/depth is behind it, (2) **how the touched material
is sliding** relative to its undeformed rest state (the shear field), and (3) **whether grip is about to
fail** (slip). All three exist because a robot gripper needs them *before* an object falls: contact area
tells you if you are touching anything at all; shear tells you the object is being dragged rather than
gripped cleanly; slip is the leading indicator that friction is losing the fight against gravity/inertia
*before* the object visibly moves in the world.

**The engineering frame.** A tactile sensor sits in the same real-time budget as any other camera-rate
perception source (30-60 Hz, SYSTEM_DESIGN.md §1.1) — but unlike most perception, its OUTPUT feeds a
control loop with a genuinely tight reaction requirement: once slip begins, the object accelerates under
gravity immediately, so the "detect slip -> raise grip force" loop is commonly budgeted around **~10
ms**, i.e. one to two camera frames, not the tens-of-milliseconds latitude a navigation stack enjoys
(README "System context" states this honestly, including what this project does NOT model: sensor read
latency, USB transfer, and the force controller's own tick). Gel wear, illumination drift, and
mechanical compliance are the field-reality constraints named in [`PRACTICE.md`](PRACTICE.md) §1.

## The math

**Notation.** Pixel coordinates `(x, y)` in pixels (px), `x` rightward, `y` downward (the repo's image
convention). Physical lengths in millimeters (mm), converted via the fixed scale `kPxPerMm` (px/mm,
`kernels.cuh`). All formulas below are implemented verbatim in `main.cu`'s `compute_frame_state`,
`hertz_contact_radius_px`, `detectable_radius_px`, and `marker_displacement` — this section derives what
those functions only evaluate.

### Hertzian contact radius (the footprint)

A rigid sphere of radius `R` (mm) presses into an elastic half-space by a mutual-approach depth `delta`
(mm) — the standard small-deflection Hertz contact problem (Johnson, *Contact Mechanics*, 1985, ch. 3).
Approximating the sphere near its pole by a paraboloid, `h(r) approx r^2 / (2R)` for radial distance `r`
from the contact axis, elastic (Hertzian) contact theory gives the **contact radius**

```
a = sqrt(R * delta)                      (mm; multiply by kPxPerMm for px)
```

This project uses this formula directly for BOTH the sphere indenter (`R = kSphereRadiusMm`) and, via
the identical parabolic-height derivation applied to a cylinder, the edge/line-contact indenter
(`R = kEdgeRadiusMm`, contact distance measured as pure horizontal offset rather than radial — see "The
algorithm" below and `main.cu`'s `contact_distance_px`). `a` is the TRUE mechanical footprint radius —
not necessarily what a camera-and-threshold contact detector can see (next).

### The shading model, and why it predicts a smaller VISIBLE radius than `a`

Inside the contact radius, this project darkens each pixel by an amount proportional to a **paraboloid
local-depth profile** consistent with the Hertz boundary (depth = `delta` at the center, exactly 0 at
`r = a`):

```
depth(r) = delta * (1 - (r/a)^2)                      for r <= a       (mm)
darkening(r) = kShadeGainPerMm * depth(r)                                (gray levels)
```

This is the project's stated "intensity-proxy indentation" simplification (README "Limitations") — a
stand-in for real photometric-stereo depth reconstruction. Because `darkening(r)` is written in closed
form, the radius at which it first crosses the contact-mask detector's threshold `T = kContactMaskThreshold`
is EXACT, not approximated: solve `kShadeGainPerMm * delta * (1 - (r/a)^2) = T` for `r`:

```
a_detect = a * sqrt( 1 - T / (kShadeGainPerMm * delta) )        (px, a_detect < a whenever T > 0)
```

This is not a numerical artifact this project papers over — it is the honest, closed-form consequence
of using a finite intensity threshold on a shading profile that reaches zero contrast exactly at the
mechanical boundary. The CONTACT ground-truth gate (`main.cu`) compares the algorithm's measured patch
area against `pi * a_detect^2`, not `pi * a^2`, for exactly this reason (README "Expected output"
measured `a_detect approx 46.6 px` against `a approx 49.0 px` at max depth — a real, ~5% shrinkage a
learner can verify by hand from the formula above).

### The Cattaneo-Mindlin partial-slip annulus

Once the object translates tangentially while pressed, friction does not release uniformly across the
contact patch. Cattaneo-Mindlin contact theory (Johnson ch. 7) shows that for a tangential load fraction
`s = T_tangential / (mu * N)` in `[0, 1]` (a normalized ratio of applied tangential force to the limiting
static-friction force), a central **stick zone** of radius `c <= a` remains glued to the object while the
outer **annulus** `c < r <= a` has already begun to slip:

```
c(s) = a * (1 - s)^(1/3)
```

`c = a` at `s = 0` (nothing has slipped yet); `c -> 0` as `s -> 1` (the whole patch is sliding). This
project ramps `s` linearly from 0 to 1 across the SLIP phase's 40 frames (README-documented, honest
teaching simplification: this models a sustained applied shear whose effective tangential loading slowly
saturates the interface, not additional gross macroscopic motion — the object's COMMANDED position is
held fixed throughout the SLIP phase; see `kernels.cuh`'s phase-layout comment).

Each marker's displacement is then a documented, continuous function of its distance `r` from the
current contact center:

```
frac(r) = 1                                                        if r <= c
frac(r) = kStickResidualFrac + (1 - kStickResidualFrac) * (a-r)/(a-c)   if c < r <= a
frac(r) = 0 (never touched)                                        if r > a
displacement(r) = commanded_shear_px * frac(r)
```

`kStickResidualFrac` (0.15) is the small residual motion kinetic friction still imparts even in the
slipping annulus — a marker at the very edge of contact is not perfectly stationary, it is dragged a
little.

### 2-D rigid (Procrustes) fit — the slip-scoring math

Given a set of `n` marker rest positions `P_i` and their measured (detected) positions `Q_i`, the
optimal rigid transform (rotation `theta` + translation `t`, no scale) minimizing
`sum_i |Q_i - (R(theta) P_i + t)|^2` has a closed form in 2-D that needs no SVD. Treat each centered
point as a complex number: `z_p = (P_i - centroid_P)`, `z_q = (Q_i - centroid_Q)` (as `x + i*y`). Then

```
theta = atan2( Im(sum_i z_q,i * conj(z_p,i)),  Re(sum_i z_q,i * conj(z_p,i)) )
t = centroid_Q - R(theta) * centroid_P
```

(`main.cu`'s `rigid_fit_and_slip` implements this in real arithmetic — `Re = sum(qx*px + qy*py)`,
`Im = sum(qy*px - qx*py)` — the same expression, expanded.) The **residual** of marker `i` is
`|Q_i - (R(theta) P_i + t)|`; the **slip score** is the fraction of in-contact markers whose residual
exceeds `kResidualSlipThresholdPx`.

## The algorithm

Per frame (the numbered steps are labeled in `main.cu`'s render/pipeline loop):

1. **Render** (host): evaluate `compute_frame_state(t)` (the closed-form physics above) then paint the
   gel background (baseline gray, minus shading, plus fixed texture noise) and blend every marker on
   top at its physics-predicted, rounded-to-nearest-pixel position.
2. **GPU: contact mask** — `|frame - baseline| >= threshold` per pixel (map), then binary erosion then
   dilation (a morphological OPEN — see "The GPU mapping" for erosion vs. dilation as MIN/MAX filters
   and why open, not close, is the right operator for isolated-speckle noise).
3. **GPU: patch stats** — area and (unnormalized) centroid sums via atomic accumulation.
4. **GPU: marker detect** — one thread per marker, local-minimum search in a small window around the
   marker's known rest position (NOT a whole-image blob detector — see "The GPU mapping").
5. **GPU: marker track** — displacement (detected - rest), a validity flag (was the window's minimum
   dark enough to trust?), and an in-contact flag (sampled from the mask at the marker's REST pixel,
   not its detected pixel — see the reasoning in `kernels.cu`'s `track_markers_kernel` header: contact
   membership is a question about the marker's undeformed IDENTITY, not its current drawn position).
6. **Host: rigid fit + slip score** — the closed-form Procrustes fit above, over in-contact + valid
   markers only; slip is declared when the score crosses `kSlipScoreDeclareThreshold`.

**Complexity.** Steps 2-3 are `O(W*H)` per frame (map/stencil/reduction — trivially parallel, 76,800
independent-ish pixels). Steps 4-5 are `O(num_markers * window_area)` = `O(221 * 17^2)` ~ 64K comparisons
per frame — still tiny. Step 6 is `O(num_markers)` — the reason it stays on the host (below).

**Why search near rest instead of whole-image blob detection + assignment.** A "textbook" tactile
tracker would detect ALL dark blobs in the whole image (a stream-compaction pattern: threshold, local-
minimum test, atomically append survivors to a candidate list) and then solve an assignment problem
(nearest-neighbor or Hungarian matching) between that unordered candidate list and known marker
identities. This project takes the simpler, still-legitimate path real systems ALSO use once the sensor
is calibrated: search a small window around each marker's KNOWN rest position. Two honest reasons this
is not a shortcut: (1) it is CHEAPER — `O(num_markers * window)` instead of `O(W*H)` for detection plus
an `O(num_markers * candidates)` matching pass; (2) it sidesteps the identity-assignment problem
entirely, because searching in a window ALREADY tied to a specific rest position IS the assignment. The
cost, stated honestly: this project's synthetic scene gives ground-truth marker identity for free (each
image lattice cell IS a known marker), which a REAL system does not have on its very first frame (it
must calibrate the rest grid once, from a genuine blob-detection pass, before this project's search-
near-rest approach applies from frame 2 onward) — see "Where this sits in the real world."

## The GPU mapping

```
contact_mask_kernel / erode3_kernel / dilate3_kernel / patch_stats_kernel:
  one thread = one PIXEL.   grid: ceil(320/16) x ceil(240/16) blocks, 16x16 threads/block
  (320, 240 both multiples of 16 in THIS image -> no ragged tail here, but the
   bounds check stays; a kernel correct only for exact multiples is a bug waiting
   for the next resolution)

  contact_mask: map, fully independent, coalesced reads/writes -> no shared memory
  erode3/dilate3: STENCIL, radius 1 -> each output reads 9 inputs; neighboring
  threads' 3x3 windows overlap 9x -> the textbook shared-memory-tile case (load a
  halo once per block). This project keeps the naive global-memory version
  deliberately (measured well under 1 ms total for BOTH passes at 320x240) —
  07.09's jump-flooding kernels are this repo's worked shared-memory-tile example
  for a project where the tile actually earns its complexity.
  patch_stats: map + atomicAdd (3 independent counters) -> no shared-memory
  block-reduce, stated as a deliberate "teaching beats cleverness" call at this
  image size (kernels.cu's header comment measures and names the tradeoff).

detect_markers_kernel / track_markers_kernel:
  one thread = one MARKER (221 threads, NOT 76,800).  grid: ceil(221/128) blocks
  A brute-force per-marker window search (17x17 = 289 comparisons/thread for
  detect) is the textbook case where "just use more threads at finer
  granularity" would be WRONG: a per-pixel kernel here would need a second
  compaction pass to even produce a marker list, for a problem that only has
  221 answers to find in the first place. Matching kernel granularity to
  problem granularity (not always "one thread per pixel") is this project's
  clearest GPU-mapping lesson.
```

**Why the rigid fit stays on the host.** `O(num_markers)` ~ 221 points, a handful of trig calls and
sums — microseconds of scalar work. Fusing it into a GPU kernel would need either a second reduction
kernel (more code) or a single-thread kernel (defeats the purpose of using a GPU at all) for a
computation this repo has already named, at this exact scale, in 08.01's softmin blend: keeping it on
the host puts the ENTIRE slip-scoring algorithm on one screen of plain C++, immediately below the kernel
call whose output it consumes — worth more didactically than the marginal (zero, at n=221) performance
gain.

## Numerical considerations

- **Marker rendering has a PROVABLY unique per-blob minimum, by construction, not by luck.** Every
  marker's rendered profile blends from `kMarkerDarkGray = 60` (exact, at the rounded center pixel,
  `blend = 1` so the gel background contributes NOTHING there) out to the surrounding gel value at the
  marker's edge. The gel background itself never drops below approximately
  `kGelBaselineGray - kShadeGainPerMm*kIndentDepthMaxMm - kTextureNoiseAmplitude` = `180 - 84 - 4 = 92`
  anywhere in the whole image (worst case: maximum indentation depth, at the shading center, at the
  most negative texture-noise draw) — comfortably above 60. So no pixel outside a marker's own exact
  center can ever be as dark as that center, which is exactly what makes `detect_markers_kernel`'s
  argmin search well-posed with no risk of a genuine tie.
- **Non-overlap margins are also proven, not assumed.** `kMarkerMarginPx (9) > kSearchRadiusPx (8)`
  guarantees every marker's search window stays fully in-bounds (no border clamping branches).
  `kSearchRadiusPx (8) < kMarkerSpacingPx/2 (9)` guarantees one marker's search window can never reach
  a NEIGHBORING marker's rest cell. The maximum possible marker excursion (`kShearTotalPx = 5px`) is
  small enough that two adjacent markers' rendered blobs (radius 4px each) can never touch:
  `spacing(18) - excursion(5) - 2*radius(4) = 5 px > 0` minimum gap, even in the worst case both
  markers shift toward each other. These are arithmetic facts about the constants in `kernels.cuh`, not
  claims verified only empirically.
- **Why the SHEAR gate measures EXACTLY 0.00 px error.** During the SHEAR phase every in-contact marker
  gets the IDENTICAL commanded shear value (no `r`-dependence: `frac = 1` uniformly, full stick). Since
  every marker's rest x-coordinate is an exact integer and the SAME float `shear` is added to all of
  them before rounding, `round(rest_x + shear)` is `rest_x + round(shear)` for every marker — an
  algebraic identity of `round()`, not a coincidence — so all in-contact markers land on the identical
  integer pixel shift. There is no per-marker rounding variance to average away, hence the perfect
  measured agreement (README "Expected output").
- **Why SLIP-phase measurements are noisier.** In the annulus, `frac(r)` varies continuously with each
  marker's own `r`, so each marker's `round(rest + shear*frac(r))` lands independently — genuine, small
  (<=0.5 px) quantization noise appears ON TOP OF the true physical signal, which is why the measured
  slip-score curve rises in small discrete steps (each step = 1/22 markers, the count active during
  this scenario's slip phase) rather than smoothly matching the continuous area-fraction ground-truth
  model.
- **A known, honest limitation of the ORDINARY least-squares rigid fit.** Once slipping markers begin
  to outnumber stuck ones (late in the SLIP phase), the fit is no longer dominated by the stuck core —
  it drifts toward whatever the numerical majority is doing, which can pull residuals for the FEW
  remaining truly-stuck markers upward too. This project's slip-ONSET gate is unaffected (onset happens
  early, while stuck markers still dominate), but a learner plotting the full `slip_timeline.csv` will
  see this effect directly in the score's non-monotonic behavior late in the run — exactly the standard
  argument for a robust estimator (RANSAC, IRLS) over ordinary least squares once outliers are a
  sizeable fraction of the data (Exercise territory, README).
- **Exact (not tolerance-based) GPU/CPU verification.** Every pipeline kernel (contact mask, erosion,
  dilation, patch stats, marker detect, marker track) operates on integers and threshold comparisons —
  no floating-point accumulation chain exists anywhere in the checked path (the RENDERER uses floats,
  but both GPU and CPU consume the SAME already-rendered uint8 bytes). This is why `main.cu`'s VERIFY
  stage demands bit-for-bit equality on all 100 frames rather than a documented tolerance (contrast with
  08.01's RK4 dynamics, which genuinely need one).

## How we verify correctness

Two independent kinds of check, because a tactile pipeline can be *numerically exact and physically
wrong* (e.g. a correctly-implemented kernel fed a broken sensor model) just as easily as the reverse:

1. **The exact GPU/CPU gate, every frame.** `reference_cpu.cpp` is a literal sequential twin of every
   kernel in `kernels.cu` — same loops, same tie-break rule (detect_markers' window scan order:
   `dy` outer ascending, `dx` inner ascending, strict less-than to update — documented explicitly in
   both files as the one place a "mathematically equivalent but differently-ordered" rewrite would
   silently break bit-exactness on a tied window minimum). `main.cu` runs BOTH paths on ALL 100 frames
   and demands zero mismatches across mask bytes, patch stats, and every marker's detected position,
   intensity, displacement, validity, and contact membership — not a sampled subset (measured: **0
   mismatches**).
2. **Three ground-truth gates, physics vs. measurement.** CONTACT (patch area/centroid vs. the
   closed-form `a_detect` footprint), SHEAR (mean tracked displacement vs. the commanded translation),
   and SLIP (detected onset frame vs. the Cattaneo-Mindlin-derived modeled onset, computed from the
   SAME formula the scenario is built from, never hardcoded) — each threshold set with documented
   headroom over an ACTUAL measured run (README "Expected output"), never asserted from theory alone
   (CLAUDE.md §8).

The scenario is committed (`data/sample/`) so the whole check runs offline; the three demo artifacts
(`contact_mask.pgm`, `shear_field.csv`, `slip_timeline.csv`) make the pipeline's behavior *inspectable*,
not just pass/fail.

## Where this sits in the real world

- **Real GelSight/DIGIT sensors reconstruct actual depth, not an intensity proxy.** Production sensors
  illuminate with (typically) three or more distinctly-colored, directionally-separated LEDs; each
  color channel's brightness at a pixel is approximately proportional to the surface normal's alignment
  with that light's direction (a Lambertian-style photometric-stereo relationship), and a calibrated
  lookup or learned model integrates the resulting gradient field into an actual depth map. This
  project's single-scalar "darker = more indented" shading is a deliberately simplified stand-in for
  that machinery — the honest v1 scope stated throughout this project.
- **Marker tracking in production** typically runs a real blob detector (not this project's
  search-near-rest shortcut) on the FIRST frame to calibrate the rest grid, then tracks frame-to-frame
  with a Kalman filter or simple nearest-neighbor gating — closer to this project's approach once
  calibrated, but with the initial-detection problem this project's synthetic ground truth sidesteps.
- **TacTip** (Bristol Robotics Laboratory) is a instructive contrast in TRANSDUCTION, not just
  algorithm: pins on a soft, air-or-gel-filled dome, camera watches pin-TIP motion rather than a flat
  gel's shading — its marker-tracking software is very close to this project's shear-field kernel, but
  the mechanical sensor it feeds is a completely different design (PRACTICE §2).
- **uSkin** (XELA Robotics) abandons vision entirely: 3-axis magnetic (Hall-effect) taxels under a soft
  cover — no camera, no marker tracking, direct per-taxel force vectors instead of an image pipeline.
  Comparing it against GelSight/DIGIT/TacTip is the classic tactile-sensor-family tradeoff: vision-based
  sensors get very high SPATIAL resolution (a whole camera's worth of pixels) cheaply, at the cost of an
  image-processing pipeline and a fragile gel; magnetic/capacitive skins get robust, low-latency
  per-taxel signals at much lower spatial resolution.
- **Slip detection in production** increasingly uses LEARNED models (vibration/texture classification,
  or the catalog's own [`20.04`](../20.04-learned-slip-prediction-fused-into-the-grasp/README.md),
  "Learned slip prediction fused into the grasp control loop") rather than this project's purely
  geometric rigid-fit-residual approach — this project's analytical method is the interpretable
  baseline a learned model should be measured against, not a claim that geometry alone is what ships.
- **Robust fitting** (RANSAC, iteratively-reweighted least squares) is the standard production fix for
  this project's named ordinary-least-squares limitation (Numerical considerations, above) once a
  sizeable fraction of markers are genuinely slipping.
