# 01.13 — Canny + Hough line/circle detection for industrial alignment: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**What an edge physically is.** A camera pixel measures irradiance — photons per unit area per unit
time arriving at that sensor site, converted to a digital number by the sensor's response curve. An
*edge* in the image is a place where that measured irradiance changes rapidly with position. Physically,
that rapid change comes from one of a few distinct causes, and it matters which one, because they behave
differently under blur and noise:

- **A reflectance discontinuity** — the material or its finish changes (e.g., the boundary between a
  machined aluminum plate and the darker table it sits on). The *surface normal* does not change; only
  the fraction of incident light reflected toward the camera does.
- **A geometric discontinuity** — the surface normal itself changes abruptly (a corner, a chamfer, the
  rim of a drilled hole), so a specular or glancing highlight appears or disappears as the surface turns
  away from the light. Machined metal is especially prone to this: a drilled hole's edge is often a
  small chamfer (a deburring pass), and the light catches that chamfer differently than the flat face
  around it — which is *exactly* why this project renders holes as a genuinely darker interior (the
  chamfer's shadow/occlusion) rather than merely a different flat shade.
- **An occlusion boundary** — one surface hides another (not really present in this project's flat-part
  scene, but the dominant edge type in general robot vision).

**Lighting's role (one honest paragraph).** Real machine-vision stations choose lighting geometry
specifically to *turn the physics above into contrast*. **Backlighting** (the part silhouetted against a
bright, diffuse light source behind it) produces the cleanest possible binary silhouette edges — ideal
for measuring an external profile or a through-hole's diameter, because it makes reflectance and surface
finish irrelevant; the part is either between the camera and the light (dark) or not (bright). It cannot
see surface features like scratches, though, or non-through holes. **Brightfield / front lighting**
(what this project's scene simulates: a lit, reflective plate) shows surface detail — scratches, blind
holes, printed marks — at the cost of being sensitive to specular highlights, surface finish, and
shadow direction, which is why real installations spend real engineering effort on diffuse ring
lights or dome lighting to tame those highlights (see [`PRACTICE.md`](PRACTICE.md) §1). This project's
brushed-metal texture and shallow vignette are a simplified stand-in for that brightfield reality; see
[Numerical considerations](#numerical-considerations) for what a physically-based render would add.

**Engineering constraints a real station imposes.** A production camera's noise floor (a handful of
gray levels of read/shot noise at typical gains), the achievable optical resolution (pixels-per-mm set
by lens and standoff distance), the line's required cycle time (Rate/latency in
[`README.md`](README.md#system-context--where-this-sits-in-a-robot)), and mechanical vibration from
nearby machinery (which blurs edges further if exposure time is not short enough) all set the practical
floor on how weak an edge can be and still be reliably found — the exact tension this project's
`CANNY_T_LOW`/`CANNY_T_HIGH` thresholds and the engineered "scratch mark" scene feature are built to
teach concretely.

## The math

**Frame and units.** Image coordinates: `x` right, `y` DOWN (standard raster convention), origin at the
top-left pixel, units of pixels. This is a purely 2-D, in-image-plane problem — there is no 3-D pose or
camera extrinsic here, so CLAUDE.md §12's `T_parent_child`/quaternion conventions do not apply; the one
transform this project reasons about is the 2-D rigid map defined below.

**Gaussian smoothing.** The 5-tap kernel `w = [1,4,6,4,1]/16` is the row of Pascal's triangle
`(1+1)^4`, which is the discrete binomial approximation to a Gaussian with `sigma ~= sqrt(n/4) = 1.0` px
for `n=4` (the number of "coin flips" folded into the kernel). Convolving with it twice, separably
(horizontal pass then vertical pass), is mathematically IDENTICAL to a single 2-D Gaussian convolution
because a 2-D Gaussian is exactly the outer product of two 1-D Gaussians:
`G(x,y) = G(x)*G(y)`. This project chose `sigma ~= 1.0` px specifically because the smallest feature
that must SURVIVE blurring is the smallest hole radius (6 px) — a much wider kernel would blur a 6-px
hole's boundary into a shallow, hard-to-threshold ramp before Sobel ever sees it.

**Sobel gradient.** The two 3x3 stencils
`Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]`, `Gy = Gx^T`
approximate the partial derivatives `dI/dx`, `dI/dy` of the (now-smoothed) image intensity `I(x,y)`. The
gradient vector `(gx, gy)` points in the direction of STEEPEST INCREASE of intensity; its magnitude
`|grad I| = sqrt(gx^2+gy^2)` measures edge STRENGTH, and its direction `atan2(gy,gx)` is PERPENDICULAR
to the edge itself (the edge runs along the direction where intensity does not change). `Gx`'s
positive-side weights sum to `1+2+1=4`, so a raw convolution over an 8-bit image reports a gradient 4x
too large relative to true "intensity change per pixel of edge steepness" — this project divides by 4
(`SOBEL_SCALE`) at the source so every downstream number (thresholds, votes, gate bounds) means what it
says.

**Non-max suppression, geometrically.** An edge, before NMS, is a multi-pixel-wide RIDGE of high
gradient magnitude (the blur spreads what was a sharp step into a smooth ramp). NMS keeps only the ridge
CREST: at each pixel, walk one step forward and one step backward along the LOCAL gradient direction
(perpendicular to the edge, i.e. across it) and keep the pixel only if its magnitude is the local
maximum among those three samples. This project quantizes the continuous gradient direction to the
nearest of 4 compass/diagonal directions (0/45/90/135 degrees) rather than interpolating the two
off-grid neighbor magnitudes — see [Numerical considerations](#numerical-considerations) for the
quantified cost of that choice.

**The Hough line transform, from point-line duality.** Every non-vertical, non-degenerate line can be
written in NORMAL (point-normal) form:
`x*cos(theta) + y*sin(theta) = rho`,
where `theta in [0, pi)` is the angle of the line's NORMAL vector and `rho` is the signed perpendicular
distance from the image origin to the line. The DUALITY: fix a point `(x0,y0)` and vary `theta` — the
equation traces a SINUSOID `rho(theta) = x0*cos(theta) + y0*sin(theta)` in `(theta,rho)` space (this is
just `x0,y0` "voting for every line that could pass through it"). Two points that lie on the SAME true
line produce two sinusoids that INTERSECT at that line's own `(theta,rho)` — and N collinear points
produce N sinusoids that all pass through the same point, which is exactly what an accumulator that
counts votes per `(theta,rho)` cell measures: **the true line is the accumulator's peak.**

**Rigid alignment as a linear least-squares problem.** The applied scene transform is
`q = R(dtheta) * p + c_img + t`, where `p` is a point in the plate's LOCAL frame, `R(dtheta) =
[[cos dtheta, -sin dtheta],[sin dtheta, cos dtheta]]`, `c_img = (IMG_CX, IMG_CY)` is the image position
of the plate's nominal (untransformed) center, and `t = (dx, dy)` is the applied translation. This is
NOT linear in `dtheta`, but substituting `a = cos(dtheta)`, `b = sin(dtheta)` makes it LINEAR in the 4
unknowns `(a, b, tx, ty) = (a, b, dx, dy)`:
```
qx - IMG_CX = a*px - b*py + tx
qy - IMG_CY = b*px + a*py + ty
```
Each detected-hole correspondence `(p_local, q_detected)` contributes 2 of these linear equations; with
`NUM_HOLES=3` correspondences (6 equations, 4 unknowns) the system is overdetermined, and the
least-squares solution minimizes total squared residual via the normal equations
`(A^T A) x = A^T b` — a `4x4` linear system solved once per demo run by Gauss-Jordan elimination in
`main.cu::gauss_solve4` (project 33.01's batched small-matrix linalg is the GPU-batch-scale version of
exactly this operation, run for thousands of independent small systems at once instead of one).
`dtheta = atan2(b, a)` recovers the rotation from the fitted `(a,b)` (which need not be exactly
unit-length after a least-squares fit — `atan2` is invariant to positive scaling, so this is fine).

**Sub-bin/sub-pixel refinement.** An accumulator cell is a discrete bin; the TRUE peak of a smooth
underlying vote-density function generally falls between bins. Given 3 equally-spaced samples
`(c_lo, c, c_hi)` centered on the discovered integer peak, fitting a parabola `y = A*t^2 + B*t + C`
through the 3 points and solving for its vertex gives a closed-form sub-bin offset:
`t* = 0.5*(c_lo - c_hi) / (c_lo - 2*c + c_hi)`
(clamped to `[-0.5, 0.5]`, since anything larger means a DIFFERENT bin was actually the true peak). This
project applies it independently along each accumulator axis (theta and rho for lines; x and y for
circles) — the same closed-form idea project 01.02's stereo-disparity sub-pixel refinement fits to its
cost curve around the integer-disparity minimum.

## The algorithm

Step by step, with the complexity of each stage (`W,H` = image dimensions; `E` = number of edge pixels
surviving hysteresis, typically 1-2% of `W*H`; `T=180` = Hough theta bins; `K=NUM_HOLES=3`):

1. **Gaussian blur, separable, `O(W*H)`.** Two passes, each `O(1)` work per pixel (5 taps) —
   `O(10*W*H)` total, versus `O(25*W*H)` for a naive 2-D 5x5 convolution.
2. **Sobel gradients, `O(W*H)`.** One 3x3-stencil pass, `O(9*W*H)`.
3. **Non-max suppression, `O(W*H)`.** One pass; each pixel reads its own gradient plus 2 neighbors.
4. **Double-threshold classify, `O(W*H)`.** Pure per-pixel map, no neighbor reads.
5. **Hysteresis promotion, `O(sweeps * W*H)` worst case.** Each sweep is `O(W*H)` (every weak pixel
   checks its 8 neighbors); the number of sweeps is bounded by the LONGEST weak-pixel CHAIN reachable
   from any strong seed (at most the image diagonal, `~400` sweeps) but MEASURED on this project's
   scene at 46 sweeps (see [How we verify correctness](#how-we-verify-correctness)) — real scenes
   converge far faster than the theoretical worst case because chains are short.
6. **Finalize edge map, `O(W*H)`.**
7. **Hough line voting, `O(E * T)`.** Every edge pixel votes across all 180 theta bins — the
   PRODUCTION optimization (gradient-informed theta windowing, README Exercise 3) would restrict each
   pixel to a narrow window of theta bins near its OWN gradient direction (since a pixel's own measured
   gradient already estimates the perpendicular-to-edge direction fairly well), cutting this to
   `O(E * window_width)`. This project implements the full, didactic `O(E*T)` sweep — with `E` typically
   a few hundred pixels and `T=180`, that is on the order of `10^5` accumulator writes, trivially fast
   even unoptimized, which is *why* the didactic full sweep is the right default here: the "expensive"
   case only bites at production image/edge-count scales this teaching project does not reach.
8. **Hough circle voting, `O(E * K)`.** Each edge pixel votes at 2 candidate centers per KNOWN radius —
   `O(2*E*K)` total, versus a GENERIC (unknown-radius) circle Hough transform's
   `O(E * R_range)` fan-out per pixel (voting along an entire candidate circle of centers for every
   candidate radius), which for a realistic radius search range explodes past what a single GPU pass can
   do without heavy memory-footprint tricks. Knowing the radius set in advance — a genuinely available
   fact on a real factory floor, where the CAD model specifies every hole — is what tames it.
9. **Peak extraction + sub-bin refinement, `O(T*R_bins)` for lines / `O(K*W*H)` for circles.** Host-only,
   not GPU-parallelized (see [The GPU mapping](#the-gpu-mapping) for why that is the right call at this
   accumulator size).
10. **Alignment least squares, `O(K)`.** A fixed `4x4` linear solve, independent of image size.

## The GPU mapping

**The dominant mapping: one thread per pixel.** Every stage through the edge map (Gaussian blur, Sobel,
NMS, classify, hysteresis, finalize) uses the SAME flat 1-D grid of 256-thread blocks over
`W*H = 76,800` threads — the repo's standard idiom (see `kernels.cu`'s `flat_grid` helper). No shared
memory is used even for the stencil kernels (Gaussian, Sobel, NMS all re-read neighboring pixels from
GLOBAL memory): at this problem size (320x240, a few MB of traffic total) the L1/L2 cache already
captures most of the reuse a hand-written shared-memory tile would target, and the didactic clarity of
"read exactly the neighbors this pixel needs, right where the math is written" outweighs the modest
bandwidth win — see README Exercise 4 for the separable-prefix-sum speedup a hot path WOULD want.

**Hysteresis: iterative, synchronized between kernel launches.** Each `hysteresis_propagate_sweep_kernel`
launch is a plain map (one thread per pixel, checks its own 8 neighbors), but the ALGORITHM is
iterative: `main.cu` launches it repeatedly, checking a single `atomicOr`-accumulated `changed` flag
after each launch, until a sweep changes nothing. This is the same pattern project 01.06's connected-
component labeling uses for its label-propagation fixed point — CUDA has no efficient "wait until no
thread anywhere wants to write" primitive WITHIN one kernel launch at this problem's irregular-chain
shape, so the natural mapping is "repeat the whole-grid kernel until convergence," accepting some
redundant re-scanning of already-settled pixels in exchange for a simple, provably-correct fixed point.

**Hough voting: SCATTER with atomics, not gather.** The natural alternative mapping — one thread PER
ACCUMULATOR CELL, each looping over every edge pixel asking "do you vote for me?" — would cost
`O(cells * E)` (144,180 cells x hundreds of edge pixels), far worse than the SCATTER mapping's
`O(E * 180)` used here, because the "owning" side (edge pixels) is far smaller than the "target" side
(accumulator cells). Scatter wins whenever that asymmetry holds; GATHER wins in the opposite case — a
dense, regular target with few, expensive-to-enumerate sources (contrast this with, e.g., an FEA
stiffness-matrix assembly, where each output DOF gathers from a small fixed set of neighboring elements
rather than scattering from an unbounded set of contributors). `atomicAdd` on `int` is what makes the
scatter mapping SAFE: many threads may target the same cell concurrently; the hardware serializes
conflicting atomics on one address into SOME total order, and because integer addition is exactly
associative and commutative (no rounding, unlike float addition), the FINAL COUNT is identical
regardless of that order — this is the load-bearing fact behind the project's bit-exact line-accumulator
claim (see [Numerical considerations](#numerical-considerations)).

**Constant memory for the theta table.** `g_cos_fixed`/`g_sin_fixed` (kernels.cu) are `__constant__`
arrays: every thread in the line-voting kernel launch reads the SAME 180 entries in the SAME order (a
`for` loop over all theta bins) — a textbook BROADCAST access pattern, which constant memory's small
per-SM cache is built for (one fetch serves an entire warp reading the same address on the same cycle,
versus 32 separate global-memory transactions). At `180*4 bytes*2 arrays = 1,440` bytes total, it is a
tiny fraction of the 64 KiB constant window with room to spare.

**Peak extraction and the alignment solve stay on the HOST.** After the accumulators are built (and
verified against the CPU oracle), `main.cu`'s peak-finding and the `4x4` alignment solve run as plain
sequential host code — deliberately, for two reasons. First, they are TINY: the line accumulator has
144,180 cells (a full scan is microseconds even sequentially) and the alignment solve is one `4x4`
system. Second, and more important pedagogically: this mirrors flagship project 08.01's MPPI controller,
whose softmin control-blending step is ALSO deliberately left on the host after the GPU rollout kernel —
seeing the WHOLE downstream algorithm in a few dozen lines of plain C++ teaches more than a
micro-optimized fused kernel would, and nothing here is on the robot's real-time critical path at this
scale (see [`README.md`](README.md)'s rate/latency budget).

## Numerical considerations

**Precision.** Every image-processing stage (blur, Sobel, NMS) uses FP32 throughout, never re-quantizing
intermediate results back to `uint8` between stages — doing so would throw away exactly the sub-integer
precision Sobel needs to compute an accurate gradient from an already-smoothed (fractional-valued)
image.

**Float-tolerance vs. exact vs. bit-exact — three different verification tiers in ONE project, and why:**

| Stage | Tier | Why |
|-------|------|-----|
| Gaussian blur, Sobel, NMS | Float tolerance (`~1e-2`, `~5e-2` for NMS's `sqrt`) | Independently-compiled GPU (nvcc) and CPU (cl.exe) code computing the identical formula can still round differently — GPU compilers aggressively fuse multiply-adds into single-rounding FMA instructions; host compilers may round twice. Up to ~1 ULP divergence per operation, compounding slightly through the pipeline. The exact same story as this repo's SAXPY placeholder's documented tolerance. |
| Hysteresis edge map (state array) | EXACT (bit-for-bit) | The promotion rule is proven monotonic and convergent to a UNIQUE fixed point regardless of visit order (kernels.cu's kernel comment) — GPU (synchronous repeated sweeps) and CPU (queue-based flood fill) are two structurally different algorithms reaching the SAME fixed point, and integer state values (0/1/2) have no rounding at all. |
| Hough LINE accumulator | BIT-EXACT | Integer `atomicAdd` is exactly associative/commutative (no float rounding), AND the vote ADDRESS itself is computed with a shared Q16 fixed-point theta table (see below) instead of independently-rounded floating-point `cosf`/`sinf` — removing BOTH sources of GPU/CPU divergence at once. |
| Hough CIRCLE accumulator | Peak-level tolerance (an honest exception) | The vote position depends on the per-pixel Sobel gradient DIRECTION, which is only float-tolerant (see row 1) — a fixed-point table cannot help here because the input itself is continuous, data-dependent, and not from a lookup table. Verified by comparing the independently-extracted PEAKS of the GPU- and CPU-built accumulators (center within ~1 px, votes within a small margin) rather than claiming false bit-exactness. |

**The fixed-point theta table, in detail (why it exists).** A rho value landing within 1 ULP of a bin
boundary could round to a DIFFERENT integer `rho_bin` on GPU vs. CPU if each independently evaluated
`cosf(theta)`/`sinf(theta)` in floating point — silently breaking the "bit-exact accumulator" claim for
that one vote, undetectable except by the exact comparison this project actually performs. The fix:
compute `cos_fixed[t] = round(cos(t*THETA_STEP) * 65536)` (`Q16`) ONCE, in double precision, on the
host, and hand the IDENTICAL `int32_t` array to both the GPU (via `upload_hough_constants` ->
`__constant__` memory) and the CPU oracle (as a plain function argument). From that point on, the vote
address `x*cos_fixed[t] + y*sin_fixed[t]`, and its rounding to `rho_bin` via integer bias-and-shift, is
PURE INTEGER ARITHMETIC — operations the C++ and IEEE standards specify exactly, with zero rounding-mode
or FMA-contraction ambiguity left anywhere in the computation.

**The circle-accumulator radius-collision lesson (numerical, not conceptual).** A genuine hole boundary
pixel's vote scatters across a few neighboring cells due to sub-pixel rounding of its OWN measured
gradient direction — MEASURED on this project's scene: a raw single-cell peak for the smallest hole
carried only 14 of an expected ~38 votes; summing a small window around it recovered 44. But a window
wide enough to recover that scatter can ALSO scoop up a systematic, non-random contribution from a
DIFFERENT hole: an edge pixel on a TRUE circle of radius `R`, voting with a WRONG candidate radius `r`,
lands on a small ring of radius `|R-r|` around that OTHER hole's own true center (a direct consequence
of the vote formula — work it out: `center +/- r*(unit gradient)` where the gradient direction is exact
for the pixel's true circle). When `|R-r|` is smaller than the peak-extraction window's reach, that
ring's votes concentrate into the window sum as if they belonged there. This project's 3 nominal radii
(6, 8, 10) differ by exactly 2 — MEASURED to collide at window radius 2 (the r=6 plane's reported peak
silently became the r=8 hole's location) and to be safely separated at window radius 1 (see
`main.cu`'s `CIRCLE_PEAK_WINDOW` comment for the full numbers). A production system would either widen
the radius spacing or use a smarter, genuinely 3-D-aware non-max suppression across the radius axis too.

**The hysteresis engineering lesson (a rendering/numerics interaction, not a detection bug).** An early
version of the scratch mark was rendered as a thin (2 px) two-edged "valley" — and MEASURED to produce
almost no usable gradient at all: the two nearby opposing edges partially cancelled under the 5-tap
Gaussian blur (whose effective support is comparable to the valley's own width), and separately, a bare
T-junction where a weak edge meets a strong one at 90 degrees MEASURED a genuine Gaussian-blur
corner-rounding gap (the local gradient dipped below even `T_LOW` for a pixel or two exactly at the
junction), breaking 8-connectivity so hysteresis never even started propagating. The fix — a wider,
one-sided band with a short, deep "seed" segment right at the junction — is documented in
`scripts/make_synthetic.py`'s `SCRATCH_WIDTH`/`SCRATCH_SEED_LEN` comments; the resulting scene is what
the `hysteresis_lesson` gate actually measures (double-threshold recovers the WHOLE mark; a single high
threshold recovers only the short deep seed).

**Angle wrapping.** Hough line `theta` lives in `[0, pi)`, not `[0, 2*pi)`, because a line and the "same
line described from the opposite normal direction" (`theta + pi`, `rho -> -rho`) are the IDENTICAL line
— including both would double-count every line. `dtheta` (the applied/recovered rotation) is a full
signed angle in radians, wrapped defensively (`main.cu`'s alignment gate) to `(-180, 180]` degrees before
comparing recovered vs. truth, though this scene's 7-degree applied rotation never approaches that
wraparound in practice.

**Determinism.** The whole pipeline is deterministic given the same input image: no RNG appears
anywhere in `src/` (only `scripts/make_synthetic.py` uses one, seeded, to build the input). The two
"stochastic-looking" quantities in the printed output — the hysteresis sweep count and the GPU/CPU
timing figures — are informative `[info]`/`[time]` lines, deliberately excluded from the diffed stable
output, exactly per this repo's output-contract convention.

## How we verify correctness

Every stage is checked twice: an INDEPENDENT CPU oracle (`reference_cpu.cpp`, written separately from
the GPU kernels per the template's twin-independence ruling) for the GPU-vs-CPU comparison, PLUS a set
of 6 gates that do not route through any shared code at all — the two-tier structure the template's
independence ruling requires whenever a shared helper (here: the Q16 theta table, and the deliberately
UN-twinned peak-extraction/alignment-solve analysis) could otherwise hide a shared bug (the cautionary
tale cited in `reference_cpu.cpp`'s header: flagship 13.03's identical variable-shadowing bug lived in
BOTH paths and only an independent analytic gate caught it).

**Tier 1 — the twin comparison** (see the table in [Numerical considerations](#numerical-considerations)
for which tolerance tier each stage uses). Every run of the demo performs this comparison LIVE, on the
committed scene, and refuses to proceed to the gates at all if any twin disagrees (`RESULT: FAIL`
immediately, `main.cu`'s VERIFY stage).

**Tier 2 — the 6 independent gates**, each measuring something the twin comparison CANNOT (whether the
DETECTION, not just the GPU/CPU agreement, is actually right):

- **`line_recovery`** — every one of the 4 true edges is matched by SOME detected Hough-line peak within
  3 degrees / 3 px (bounds measured-then-margined: this scene's actual votes for the shortest true edges
  were 68, comfortably above the `HOUGH_LINE_PEAK_MIN_VOTES=30` floor, well above the noise a handful of
  unrelated edge pixels could accumulate).
- **`circle_recovery`** — every one of the 3 true holes is matched by its OWN known-radius plane's peak
  within 2 px, at or above `HOUGH_CIRCLE_PEAK_MIN_VOTES=30` — set with real margin above the ~8-10 vote
  noise floor MEASURED elsewhere in the same accumulator planes, and real margin below the 44/57/73
  votes MEASURED at the true centers.
- **`alignment`** — the actual business measurement: the LS-recovered `(dx, dy, dtheta)` within 2 px / 1.5
  degrees of the truth transform baked into the scene. MEASURED on the committed sample: recovered
  `dx=7.99 px, dy=-4.90 px, dtheta=6.72 deg` against applied `dx=8.00, dy=-5.00, dtheta=7.00` — well
  inside the bound.
- **`edge_quality`** — precision (per DETECTED pixel: is it within 1.5 px of a true edge?) and recall
  (per point SAMPLED along the true curves at ~1 px arc-length spacing: is a detected pixel nearby?).
  These two are DELIBERATELY measured with different denominators — an earlier version of this gate
  divided detected-pixel count by the analytic mask's PIXEL AREA for recall too, and that measured
  recall around 0.33 even on a detector later shown (by the arc-length version) to recover ~99% of the
  true perimeter LENGTH: a 1.5-px-wide band is roughly 3 px wide, so a perfectly-thin 1-px detected
  curve can score at most ~1/3 "recall" against its own area no matter how good the detector is — a
  metric-definition bug, not a detector problem, and a genuinely useful lesson in gate design.
- **`hysteresis_lesson`** — the designed comparison: on the engineered scratch segment, double-threshold
  hysteresis must recover >= 70% of its sampled length (MEASURED: 100%) while a SINGLE high threshold
  applied uniformly must recover <= 30% (MEASURED: 20%, from the short deep "seed" zone alone) — see
  [Numerical considerations](#numerical-considerations) for the full rendering story behind these
  numbers.
- **`negative_control`** — run the identical pipeline on `negative_control.pgm` (same texture/vignette/
  noise, zero plate) and require ZERO accepted line peaks and ZERO accepted circle planes above their
  vote thresholds. MEASURED: 0 and 0, confirming the thresholds are not simply "always fire."

## Where this sits in the real world

Production machine-vision stacks run the SAME conceptual pipeline with three real differences: (1) they
use vendor-tuned, heavily-optimized Canny/Hough implementations (OpenCV, NVIDIA NPP) rather than
hand-rolled teaching kernels — see [`README.md`](README.md#prior-art--further-reading) for exactly which
calls; (2) they calibrate the camera (project 01.16) so every measurement this project reports in pixels
comes out in millimeters, with a documented uncertainty; and (3) commercial metrology suites (Halcon,
Cognex) wrap line/circle detection in CALIPER tools with sub-pixel accuracy specifications, gauge R&R
statistical acceptance testing (see [`PRACTICE.md`](PRACTICE.md) §4), and certified repeatability under
varying lighting — none of which this teaching project claims. The core mathematical ideas (point-line
duality, known-geometry-constrained search, hysteresis, least-squares rigid registration) are, however,
EXACTLY what those production systems compute underneath the tooling.
