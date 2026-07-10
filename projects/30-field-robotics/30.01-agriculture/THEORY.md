# 30.01 — Agriculture: fruit detection + 3D localization + ripeness; weed-vs-crop segmentation at frame rate; per-plant spray targeting; crop-row following; canopy volume from LiDAR; under-canopy navigation; yield mapping: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md section 4.2).

## The problem — physics & engineering first

**Why a fruit even LOOKS different when it is ripe: the biology.** A fruit's skin color is set by its
pigment mix. Unripe fruit is dominated by **chlorophyll** — green, because chlorophyll absorbs red and
blue light and reflects green. As a fruit ripens, chlorophyll breaks down (a genuinely enzymatic,
programmed process — climacteric fruit like apples and tomatoes trigger it via ethylene signaling), and
pigments that were always present but masked become visible: **carotenoids** (yellows, oranges — the
same pigment family as carrots) and, in many fruit, **anthocyanins** (reds, synthesized fresh during
ripening, not merely unmasked). This is why ripening color sweeps green -> yellow -> orange -> red in
that specific order for so many fruit species, and it is why this project's synthetic scene models
ripeness as a smooth HUE sweep (`hue = 120*(1-ripeness)` degrees, green at 120 through red at 0 —
`scripts/make_synthetic.py`) rather than an arbitrary color table: the model is a simplified but
physically-motivated stand-in for a real, well-understood biological process.

**Why detecting this from a camera is genuinely hard — the engineering.** Four compounding difficulties
a real orchard robot faces, each of which this project's synthetic scene deliberately includes:

1. **Illumination.** Outdoor light varies by orders of magnitude across a day (full sun to overcast),
   direction (harsh midday shadows vs. flat diffuse light), and is often supplemented by an on-robot
   light for under-canopy or low-light work. A fruit's RAW color (RGB) confounds its reflectance
   spectrum with however much of that variable light happens to be hitting it — see "The math" below for
   how HSV addresses this. This project's scene renders each fruit with real Lambertian shading from an
   on-robot ring light (`light_dir ~= -normalize(point)`, PRACTICE.md section 2 discusses why this
   hardware choice is realistic for the domain) plus a per-pixel ambient floor, so a fruit's LIT and
   SHADOWED sides genuinely differ in brightness — detection must survive that, not just work on flat
   disks.
2. **Occlusion.** Fruit grows in clusters, behind leaves, behind branches, and behind OTHER fruit at
   different depths. The committed scene's two designed cross-depth merge cases (README "Limitations")
   and one fully-occluded fruit are not edge cases added for difficulty's sake — they are the norm in a
   real canopy, and a perception system that is never tested against them is untested against the
   actual job.
3. **Color variance.** Even fruit of the same species and similar ripeness varies in exact hue, and (the
   hardest case, scoped OUT of this milestone — see "Numerical considerations" and README "Limitations")
   an unripe fruit's color can be nearly identical to the leaves around it.
4. **Bandwidth and latency.** A field robot moving at even modest speed needs its perception to keep
   pace with either its own motion (for navigation-adjacent milestones) or its harvesting cycle
   (SYSTEM_DESIGN section 1.1's 30-60 Hz camera row; README "System context" derives the harvest-cycle
   argument for why this project's ~3-4 ms pipeline has ample headroom against either).

**The engineering frame for the sensor itself.** This project consumes an RGB-D frame — a color camera
plus a depth channel, exactly the sensor class named in SYSTEM_DESIGN's manipulator-work-cell reference
robot (section 2.2) and this project's own agricultural analog (README "System context"). Real depth
sensing outdoors is itself hard (structured-light and ToF sensors both struggle in direct sunlight;
PRACTICE.md section 2 discusses the hardware alternatives) — this project models a STRUCTURED-LIGHT-like
noise curve (quadratic in range, "The math" below) as an honest stand-in for that engineering reality,
not a claim that any specific sensor was simulated exactly.

## The math

**Camera model.** Pinhole projection, `fx=fy=525` px (the documented Kinect-v1/TUM-RGBD focal length —
`data/README.md`), principal point `(cx,cy)=(320,240)`. A 3-D point `(x,y,z)` in the camera frame
(optical convention: x-right, y-down, z-forward — SYSTEM_DESIGN section 3.2) projects to pixel
`(u,v) = (fx*x/z + cx, fy*y/z + cy)`. This project's ray parametrization (shared by the renderer in
`scripts/make_synthetic.py` and the back-projection in `src/main.cu`) writes a camera ray through pixel
`(x,y)` as `d = ((x+0.5-cx)/fx, (y+0.5-cy)/fy, 1)` — deliberately UNNORMALIZED, so that a point
`t*d` on the ray has `z`-coordinate exactly `t`: **the ray parameter IS the depth in meters**, with no
extra division. This is why the renderer's ray-sphere intersection (`gen_scene` in the generator) reads
its hit depth directly off the quadratic's root, and why back-projection (below) is a single multiply.

**RGB -> HSV, derived.** Given `(r,g,b) in [0,1]^3`, let `cmax = max(r,g,b)`, `cmin = min(r,g,b)`,
`delta = cmax - cmin` (the "chroma"). Then:

```
V = cmax                                     (value: the brightest channel present)
S = delta / cmax             (cmax > 0)      (saturation: chroma relative to brightness)
H = 60 * ( (g-b)/delta mod 6 )                if cmax == r
H = 60 * ( (b-r)/delta + 2 )                  if cmax == g
H = 60 * ( (r-g)/delta + 4 )                  if cmax == b
```

**Why this separates color from lighting.** Scale every channel by the same brightness factor `k > 0`
(exactly what uniform Lambertian shading does to a surface's reflected color): `r' = kr, g' = kg, b' =
kb`. Then `cmax' = k*cmax`, `delta' = k*delta`, so `S' = delta'/cmax' = delta/cmax = S` — **saturation
is scale-invariant** — and every ratio inside the hue formula, e.g. `(g'-b')/delta' = k(g-b)/(k*delta) =
(g-b)/delta`, is IDENTICAL — **hue is exactly scale-invariant**. Only `V' = k*cmax` changes. This is not
an approximation: for a pure brightness scaling, HSV's H and S are mathematically unchanged and V
absorbs all of it. Real shading is not a perfect uniform scale (there is also an additive ambient term —
this project's `value = ambient_floor + (1-ambient_floor)*diffuse`, ambient_floor > 0, is exactly such a
term), which is WHY hue is only *almost* invariant to shading in the rendered scene (a small amount does
leak in, visible in the mask thresholds' margins — "The algorithm" below) rather than a claim that HSV
is a perfect color/lighting separator on real cameras (it is not: white balance, sensor response
non-linearity, and specular highlights all leak some lighting information into hue too — a real system
calibrates for this; this project's synthetic scene is honest about testing the IDEAL case).

**3-D localization — the surface-vs-center depth bias, derived.** A camera does not see a sphere's
CENTER; it sees the near hemisphere of its SURFACE, and every pixel-count-weighted average this pipeline
computes (the robust depth estimate) is therefore an average of SURFACE depths, not the center depth.
Parametrize the visible surface by its radial distance `rho in [0, r]` from the sphere's PROJECTED
center (i.e., `rho` is the pixel-space radius of the point, converted back to a length via the same
`z/fx` scale as everywhere else): a point at that radius sits at height `h(rho) = sqrt(r^2 - rho^2)`
above the sphere's "equatorial" plane (the plane through the center, perpendicular to the viewing axis),
so its depth is `Z(rho) = Zc - h(rho)` (`Zc` = the true center depth; the near pole, `rho=0`, is closest
at `Zc - r`; the silhouette edge, `rho=r`, is at exactly `Zc`). **Averaging uniformly over PROJECTED
AREA** (area element `2*pi*rho drho` — the natural weighting for a pixel-count-weighted mean, since
equal screen area means equal pixel count):

```
mean_h = (1 / (pi*r^2)) * INTEGRAL[0, r] of  h(rho) * 2*pi*rho  drho
       = (2 / r^2) * INTEGRAL[0, r] of  rho * sqrt(r^2 - rho^2)  drho
       = (2 / r^2) * [ -(1/3) * (r^2 - rho^2)^(3/2) ]  from 0 to r
       = (2 / r^2) * (1/3) * r^3
       = (2/3) * r
```

So `mean_surface_depth = Zc - (2/3)*r`, i.e. **the true center is `(2/3)*radius` FARTHER than the
visible surface's mean depth** — a closed-form, derivable systematic bias, independent of Z and of any
sensor noise. `src/main.cu`'s `build_detections()` applies exactly this correction:
`center_z_m = surface_depth_m + (2/3)*radius_m`. **Measured effect** (this project's committed scene,
before vs. after the correction): mean 3-D localization error dropped from **~2.7 cm to ~1.8 mm** — this
bias, not sensor noise or pixel quantization, is the dominant term in this project's localization error
budget, confirmed by direct measurement matching the closed-form prediction (e.g. a 4.7 cm-radius fruit
predicts a `(2/3)*0.047 = 3.13 cm` bias; the measured pre-correction error for that fruit was ~3.3 cm).

**Radius back-projection.** A connected component of `N` foreground pixels, treated as a filled disk of
area `N`, has screen radius `r_px = sqrt(N/pi)` (from `N = pi*r_px^2`). Similar triangles (the same
relation the camera model above already establishes: a real length `L` at depth `Z` subtends `fx*L/Z`
pixels) inverts to a real-world radius `r_m = r_px * Z / fx`.

**Ripeness.** The exact inverse of the synthetic generator's forward model: `ripeness = clamp((120 -
mean_hue_deg)/120, 0, 1)`. This is a DERIVED, closed-form inverse of a KNOWN forward model — not an
independently-fit color rule — which is exactly what makes the demo's rank-correlation gate meaningful
as a pipeline-correctness check (rather than a claim about real-world ripeness estimation accuracy; see
"How production stacks do this differently" below for the honest gap).

## The algorithm

Seven stages, each with a serial-vs-parallel complexity note (`N = W*H = 307,200` pixels; `K` = number
of connected components, typically a few dozen):

1. **RGB -> HSV** — O(N), embarrassingly parallel (pure map).
2. **Fruit mask** — O(N), embarrassingly parallel. Thresholds (`src/kernels.cuh`): `hue < 85 deg`
   (fruit ripeness 0.35-1.0 maps to hue 0-78 deg; foliage sits at 100-140 deg — a comfortable 15-degree
   margin), `sat > 0.55` (the PRIMARY discriminator against dark branch strokes: measured fruit
   saturation 0.65-0.90 vs. branch 0.40-0.50), `value > 0.22` (a backstop against near-black
   shadow/branch pixels; measured fruit value never drops below ~0.27 even fully shadowed, thanks to
   the `ambient_floor` term in "The math" above). Each threshold is chosen from the ACTUAL measured
   separation in the committed scene, not an arbitrary guess — and each resolves a SPECIFIC confusion:
   hue resolves fruit-vs-foliage, saturation resolves fruit-vs-branch, value is the shadow backstop. No
   single channel resolves every confusion in this scene — this is the concrete lesson behind CLAUDE.md
   section 1's "teach the geometry, do not hide the limits": a learner who only reads about HSV
   thresholding in the abstract does not learn that a real scene usually needs more than one gate, and
   which gate does which job.
3. **Morphological opening** — O(N) per pass (a fixed 3x3 stencil), two passes (erode, dilate).
   8-CONNECTED structuring element, chosen deliberately DIFFERENT from stage 4's 4-connectivity: opening
   judges a pixel's dense LOCAL neighborhood in every direction (a diagonal-only speck is still
   speckle), while CCL decides object membership, where the simpler, more conservative 4-connected
   choice is the deliberate teaching default (`src/kernels.cu`'s morphology kernel header expands this).
4. **Connected-component labeling** — O(N) per sweep, O(component diameter) sweeps to converge (measured
   ~56-64 on the committed scene — see "Numerical considerations"). The PARALLEL algorithm (label
   propagation) vs. the SERIAL alternative (union-find, O(N * alpha(N)) with path compression,
   effectively linear) is this project's central algorithmic teaching point — see "The GPU mapping" for
   the parallel version and "How we verify correctness" for why running BOTH is not redundant, it is the
   verification strategy.
5. **Per-component statistics** — O(N) atomic-scatter accumulation (pass 1: count/position/bbox/hue/
   depth-sum) + O(N) atomic-scatter re-accumulation (pass 2: robust inlier depth) + O(N) elementwise maps
   (mean, finalize) + a final O(N) host scan to enumerate the (small, O(K)) set of canonical component
   roots. Total: dominated by the O(N) pixel passes, not the O(K) component count — exactly the
   "GPU does the pixel work, host does the tiny bookkeeping" division of labor 08.01 establishes for
   this repo (README "Code tour" names the shared pattern).
6. **3-D localization** — O(K), pure host arithmetic (the derivation above, applied per component).
7. **Ripeness** — O(K), a single division and clamp per component.

## The GPU mapping

```
Stages 1-3 (HSV, mask, morphology):  one thread per pixel, 1-D grid over the flat
  index i = y*W+x (NOT a grid-stride loop — N is fixed at 307,200 by the committed
  scene, so one generous launch covers it with no meaningful occupancy difference;
  see kernels.cu's file header for the full reasoning). Morphology is a 3x3 STENCIL:
  9 global reads per output pixel, no shared-memory tiling (the committed scene's
  modest N makes the L2 cache do most of the work a hand-rolled shared-memory halo
  would optimize further -- README Exercise territory, not implemented here for
  the same "teaching beats cleverness" reason 33.01 states first).

Stage 4 (CCL): ccl_init_kernel is a map. ccl_propagate_sweep_kernel is a STENCIL
  (4 neighbor reads) + ATOMIC (label[i] updated via atomicMin) run REPEATEDLY from
  the host until a device-side "changed" flag (atomicOr) reports no motion. This
  is the project's only MULTI-KERNEL-LAUNCH LOOP -- see "Numerical considerations"
  for its honest cost.

Stage 5 (component stats): component_stats_init_kernel and the two mean/finalize
  kernels are maps over the SAME dense [W*H]-sized index space the atomic kernels
  use (see kernels.cuh's file header for why this dense, uncompacted layout is the
  deliberate choice -- avoiding a GPU stream-compaction kernel that would teach a
  DIFFERENT lesson than this project's). component_stats_pass1/pass2 are ATOMIC
  SCATTERS: many pixels (a whole fruit's blob) write concurrently to the SAME
  per-component slot -- atomicAdd/atomicMin/atomicMax are not an optimization here,
  they are CORRECTNESS (a plain += would silently drop concurrent updates).

No shared memory anywhere in this project -- deliberately. Every kernel's working
set per pixel is a handful of neighbor reads (morphology, CCL) or independent
global reads (everything else); the natural shared-memory opportunity (tiling a
block's morphology halo) is real but secondary to this project's two headline GPU
lessons (HSV/mask thresholding and provably-correct parallel CCL) -- named as
README Exercise 2's cousin, not built here.
```

**cuRAND/cuBLAS/Thrust/CUB:** none used. Every stage is a hand-rolled kernel — the point of a repo whose
rule is "no black boxes" (CLAUDE.md section 1): a production GPU CCL library (e.g., inside PCL's or
Open3D's clustering, or 02.04's own union-find variant) would very likely use a library-quality
implementation of exactly this pattern; this project builds the simplest CORRECT version by hand so a
learner sees every atomic and every convergence argument explicitly.

## Numerical considerations

- **Precision:** FP32 throughout (HSV values, depth in meters, all per-component sums). No FP64 is used
  or needed — the largest quantities involved (a sum of ~1700 hue values around 0-360, or ~1700 depth
  values around 1-5 m) are nowhere near FP32's ~7-decimal-digit precision limit.
- **HSV/mask determinism:** the RGB->HSV conversion (derived in "The math") uses only `+`, `-`, `*`,
  `/`, and comparisons — NO transcendental functions (no `atan2f`, unlike some HSV formulations) — so
  IEEE-754 guarantees the GPU and CPU compute the SAME result to the same rounding, modulo at most a
  1-ULP difference from FMA (fused-multiply-add) fusion differences between `nvcc` and `cl.exe`.
  Measured worst-case divergence on the committed scene: **exactly 0.0000 degrees / 0.00e+00 / 0.00e+00**
  — genuinely bit-identical, not merely "within tolerance" — which is why `main.cu`'s HSV tolerance
  (`kHsvTol = 1e-3`) has enormous headroom and was never close to being exercised.
- **Connected-component labels: EXACT equality, not tolerance, and why that is achievable.**
  `ccl_propagate_sweep_kernel`'s convergence argument (full sketch in `kernels.cuh`'s file header,
  restated briefly): every label only ever DECREASES (atomicMin never increases a value) and is bounded
  below by 0, so the sequence per pixel is monotone and bounded -> it MUST converge in finitely many
  sweeps. At the fixed point, `label[p] <= label[q]` for every foreground neighbor `q`, which forces
  every pixel's converged label down to (and never below) the TRUE minimum linear index reachable via
  foreground connectivity from it — a unique value, independent of the SCHEDULE (which threads ran in
  which sweep). This is the identical argument that proves Bellman-Ford correct for shortest paths with
  all-ZERO edge weights (label propagation with `min` IS that algorithm, specialized to a 0-weight grid
  graph). Because this fixed point is a MATHEMATICAL PROPERTY of the mask, not an artifact of any one
  algorithm's bookkeeping, the CPU oracle can use a COMPLETELY DIFFERENT algorithm (raster-scan
  union-find with path compression — O(N * alpha(N)), effectively linear, the standard SERIAL CCL
  algorithm) and, after both sides canonicalize to the SAME convention ("label = minimum linear index in
  the component" — trivial for the GPU side, which IS that convention already; one extra pass for the
  union-find side, `reference_cpu.cpp`'s `ccl_union_find_cpu`), the two MUST produce bit-identical
  integer label arrays if both are correctly implemented. Measured: **0 mismatches over 7,059 compared
  foreground pixels** — exact equality achieved, not approximated.
- **Sweep count is a scheduling detail, NOT part of the correctness guarantee.** How MANY sweeps it takes
  to reach the fixed point depends on the actual order concurrent threads apply their `atomicMin`
  updates within and across sweeps (Gauss-Seidel-style: a neighbor updated earlier in the SAME sweep is
  seen immediately, which is why this typically converges faster than a strict double-buffered Jacobi
  scheme would) — measured **56-64 sweeps** across repeated runs on the committed scene, i.e. genuinely
  a few sweeps of run-to-run variance, while the FINAL LABELS themselves are exactly reproducible every
  time (the point of the convergence proof above). This is why sweep count is printed on the unchecked
  `[time]` line, never a diffed stable line.
- **Where the GPU pays for this loop's simplicity:** each sweep needs a `cudaMemcpy` of one `int` back
  to the host to check the "did anything change?" flag — a synchronizing round trip, ~56-64 times per
  frame. Measured: the CCL stage takes ~2.5-3.3 ms of this project's ~3-4 ms total GPU time, i.e. it
  DOMINATES — not because the per-sweep KERNEL is slow (it visits 307,200 pixels doing a handful of
  comparisons each — sub-millisecond), but because of the host round-trip tax. Production fixes
  (README Exercise 5): a fixed sweep budget (no host check at all — safe if the budget provably exceeds
  the scene's worst-case diameter), or a device-side termination check via a persistent kernel / CUDA
  Graphs (32.02's territory) that never leaves the GPU.
- **Float atomics are NOT bit-order-independent, and this project says so out loud.** `comp_sum_hue` and
  `comp_sum_depth` accumulate via `atomicAdd(float*, float)`; floating-point addition is not
  associative, so in principle a different thread-scheduling order could produce a different last bit.
  Measured: bit-stable across every run and both Debug/Release configurations on the reference GPU (the
  component sizes here — tens to ~1700 pixels — keep any such divergence far below anything visible at
  the printed precision), but `main.cu`'s output deliberately keeps these MEASURED numbers on unchecked
  `[info]` lines and only diffs the PASS/FAIL verdict against a fixed threshold — the same discipline
  08.01 uses for its noise-driven trajectory numbers, for the identical reason.
- **Robust depth estimator, sanity-checked against its own assumption.** The inlier band
  (`kInlierSigmaMul * kDepthNoiseK * mean_z^2`) uses the SAME noise-model constant the synthetic
  generator used to CREATE the noise — a realistic assumption (a fielded system calibrates its sensor's
  own noise curve once), not circular reasoning about THIS scene specifically: the pipeline never reads
  the generator's code or its RNG, only the same publicly-documented sensor noise SHAPE.

## How we verify correctness

Three independent layers, because a perception pipeline can be numerically self-consistent (GPU==CPU)
while still being WRONG about the world (missing fruit, wrong 3-D positions) — and the reverse is also
possible, so both must be checked:

1. **GPU-vs-CPU, stage by stage (the VERIFY line).** HSV and mask: tolerance-based (see "Numerical
   considerations" — measured essentially exact). Connected-component labels: EXACT integer equality
   after canonicalization (measured: 0/7,059 mismatches — see "Numerical considerations" for the full
   argument for why this is achievable, not merely hoped for). Final per-fruit statistics (3-D center,
   radius, ripeness): small relative tolerance (measured worst case ~4-7e-7, far inside the documented
   1e-2 gate — the residual comes from float summation order, not a bug).
2. **Detection-rate / false-positive gate against exact 3-D ground truth (the DETECT line) — occlusion
   honesty, by fruit ID.** The committed scene's 25 ground-truth fruit include ONE fully-occluded fruit
   (0 visible pixels — genuinely undetectable from this view by any algorithm, correctly excluded from
   the 24-fruit "detectable" denominator) and TWO designed cross-depth merge pairs: two fruit at
   noticeably different depths (roughly 1.2 m and 0.5 m apart in Z) whose projected silhouettes touch in
   the 2-D image. Because this pipeline's connected-component stage groups pixels by color and 2-D
   connectivity ONLY (never by depth discontinuity — a deliberate Milestone-1 scoping choice, README
   Exercise 3 sketches the fix), each pair is reported as ONE blob, at a centroid and depth that matches
   NEITHER true fruit well (their true 3-D positions are 3D-distant even though 2-D-adjacent — see
   "The math" for why perspective occlusion of physically distant objects is a real, distinct
   phenomenon from occlusion of nearby ones). This yields exactly **20/24 individually correct matches
   and 2 unmatched merge-blobs** — the demo's threshold (`kMinDetectionRate = 0.80`, `kMaxFalsePositives
   = 2`) is set to the MEASURED value with a small margin, not padded to hide a failure, and this file
   states plainly why 100% is not achievable on the designed data: the two merge cases are a real,
   inherent limitation of image-space-only connected-component fruit detection, not noise. A pipeline
   that reported 24/24 on this scene would either be exploiting a lucky threshold or, more likely, doing
   depth-aware splitting the README already scopes out — the honest number here is the teaching content.
3. **Localization/radius/ripeness accuracy against exact 3-D ground truth (LOCALIZE / RIPENESS lines).**
   Localization: mean 1.8 mm / max 6.9 mm against a 15 mm gate — see "The math" for the derived
   surface-to-center correction that produced this (pre-correction: mean ~2.7 cm). Radius: mean 0.9 mm /
   max 2.6 mm against a 6 mm gate. Ripeness: Spearman RANK correlation (not absolute-value agreement —
   deliberately: the hue->ripeness model is a documented simplification of real ripeness, so what this
   pipeline can honestly promise is getting the ORDER right, which rank correlation measures directly
   and which is provably insensitive to any monotone reparametrization of the color model — "Where this
   sits in the real world" below states what a calibrated absolute ripeness estimate would need beyond
   this). Measured rho = 0.998 against a >=0.70 gate.

The scene is committed (`data/sample/`) so the whole three-layer check runs offline, deterministically,
with zero downloads — and the two visual artifacts (`demo/out/detections.pgm`,
`demo/out/fruit_map.csv`) make every one of these numbers independently inspectable, not just
pass/fail.

## Where this sits in the real world

- **Fruit detection in production is dominated by learned detectors** (YOLO-family fine-tunes, and
  academic systems like DeepFruits, MinneApple, Fuji-SfM) trained on thousands of labeled orchard
  images, typically fused with stereo or ToF depth for 3-D localization through essentially the SAME
  camera-model math this project derives by hand ("The math" above). This project's classical
  HSV+morphology+CCL pipeline is the didactic floor underneath that: it teaches the geometry a learned
  detector's box or mask output still has to pass through, and it is honest that its raw color-threshold
  detection accuracy would not be competitive on a real, unscoped scene (README "Limitations" — green
  fruit, shadows, specular highlights, and cluttered leaves are all genuinely harder for a classical
  color pipeline than for a CNN trained on real variety).
- **Ripeness estimation in production** increasingly goes beyond visible-light hue: near-infrared (NIR)
  reflectance, firmness sensing (mechanical or acoustic), and sometimes destructive sampling (Brix
  sugar-content meters) are all real inputs a commercial system fuses, precisely because hue alone
  (this project's entire ripeness signal) genuinely cannot resolve firmness or sugar content — two
  fruit at identical hue can differ meaningfully in eating/market readiness. This project's Spearman
  rank-correlation gate (not absolute agreement) is the honest reflection of that limit, stated in "How
  we verify correctness" above.
- **The six documented-only milestones**, each a natural, separately-scoped extension of this exact
  code and data shape:
  - **Weed-vs-crop segmentation at frame rate** — the SAME HSV+mask+CCL pipeline shape, retrained
    thresholds (or a learned classifier) distinguishing crop-plant color/shape/row-position from weeds;
    "frame rate" names the same GPU-throughput argument this project's timing numbers already establish.
  - **Per-plant spray targeting** — consumes this project's (or the weed-detection variant's) 3-D
    positions directly, converting each into an actuator command (a sprayer nozzle's on/off or aim
    angle) — the first milestone in this bundle whose output would command real hardware, hence
    PRACTICE.md section 3's full staged-testing-ladder caveat applies to it, not to Milestone 1.
  - **Crop-row following** — a navigation-stack problem (SYSTEM_DESIGN domain 23), typically LiDAR- or
    vision-based row detection feeding a lateral controller; shares this project's camera/sensor layer
    but not its per-object detection algorithm.
  - **Canopy volume from LiDAR** — a 3-D reconstruction problem (SYSTEM_DESIGN domains 02/05, e.g. this
    repo's TSDF-fusion-style flagship), estimating a tree's biomass/volume from point-cloud sweeps —
    complementary to, not overlapping with, this project's per-fruit detection.
  - **Under-canopy navigation** — GNSS is unreliable or absent under a dense canopy; this milestone is a
    state-estimation problem (SYSTEM_DESIGN domain 04/05), not a perception-detection one.
  - **Yield mapping** — temporal/spatial FUSION of this project's per-frame output (exactly what
    `demo/out/fruit_map.csv` seeds) across a robot's whole traverse, transformed from camera frame into
    a world/map frame via the robot's own localization — the natural next milestone to build on top of
    this one, and the one nearest to this project's existing code.
- **What a full research/production version needs beyond this teaching core:** depth-aware
  instance segmentation (splitting the two designed merge cases correctly — README Exercise 3), a
  learned detector for open-set color/shape robustness, temporal tracking across frames (avoiding
  double-counting the same fruit from a moving platform — directly relevant to Milestone 7), and a
  calibrated multi-sensor ripeness estimate beyond hue alone.
