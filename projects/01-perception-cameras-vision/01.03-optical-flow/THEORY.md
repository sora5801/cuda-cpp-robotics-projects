# 01.03 — Optical flow: pyramidal Lucas-Kanade, Farneback, census-transform flow: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### What a camera actually measures, and what "flow" means

A camera's sensor integrates incident photons over each pixel's well for an exposure interval and
quantizes the result to an intensity `I(x, y, t)` — here `x, y` are pixel COLUMN/ROW indices (not
metric distances; SYSTEM_DESIGN.md's SI-everywhere convention applies to physical robot quantities, but
raw pixel intensity has no SI unit, it is sensor-referred) and `t` is the frame's capture time. Between
two frames `t` and `t+dt`, a point on some physical surface projects to a different pixel location if
either the camera or the surface moved. **Optical flow** is the per-pixel apparent 2-D displacement
`(u, v)` — in pixels — such that the intensity pattern in frame `t+dt` "looks like" the pattern in
frame `t`, shifted by `(u, v)`. It is called *apparent* motion deliberately: flow is a property of the
IMAGE, not necessarily of the 3-D scene — a rotating uniformly-lit sphere produces zero flow (nothing in
its image changes) despite genuine 3-D motion, and a stationary light source sweeping a shadow across a
static wall produces non-zero flow despite zero 3-D motion. This project studies image-plane flow;
recovering 3-D motion/structure from it is the job of downstream modules (visual odometry, structure
from motion — see "System context" in `README.md`).

### Brightness constancy — the assumption everything below rests on, and where it breaks

Every algorithm in this project (and nearly every classical optical-flow method) assumes
**brightness constancy**: a physical scene point's measured intensity does not change between frames,
only its pixel location does —

```
I(x + u, y + v, t + dt) = I(x, y, t)
```

This is a strong physical claim, and real cameras violate it constantly:

- **Illumination change** — auto-exposure, a cloud passing, a light flickering. This project's pair
  (c) (`scene_b_translation_bright.pgm`) deliberately adds a smooth spatial brightness ramp to test
  exactly this (see "How we verify correctness").
- **Specular reflection** — a shiny surface's apparent brightness depends on the VIEWING ANGLE, not
  just the surface itself; it can look completely different from two viewpoints even with zero motion.
- **Occlusion / disocclusion** — a point visible in frame `t` may be hidden behind something in frame
  `t+dt` (or vice versa): there is no correct flow vector for such a pixel at all, only a "no
  correspondence exists" answer. Neither algorithm here has an explicit occlusion model; census's
  left-right consistency check (below) is the closest thing to one, catching many (not all) such cases
  as a byproduct.
- **Non-Lambertian shading** — most real materials are not perfectly diffuse; apparent brightness shifts
  with the light-camera-surface geometry as the camera moves, even for a perfectly rigid, well-lit
  scene.

The two implemented methods respond to violations of this assumption very differently, and that
difference is this project's central teaching point (developed fully in "The math" and demonstrated by
the `brightness_robustness_census` gate in `main.cu`):

- **Lucas-Kanade minimizes a squared INTENSITY DIFFERENCE.** Its normal equations are built directly
  from `I1(warped) - I0` — a real physical brightness change enters that difference exactly like a
  motion-induced one, and LK cannot tell them apart. A brightness violation biases the flow estimate.
- **Census encodes only intra-window RANK ORDER** ("is neighbor `k` brighter-or-equal to the center").
  Any transform that preserves local rank order — in particular, any smoothly-varying additive term
  that changes little across one small window — leaves nearly every comparison bit unchanged. Census
  degrades gracefully where LK degrades badly.

### The aperture problem, from first principles

Consider a small window sliding over the image; ask how much the window's content changes if shifted by
`(u, v)`. The sum of squared differences

```
E(u, v) = sum_{(dx, dy) in window} [ I(x+dx+u, y+dy+v) - I(x+dx, y+dy) ]^2
```

is, for small `(u, v)`, well-approximated by `E(u, v) ~= [u v] M [u v]^T`, where `M` is the
**structure tensor** derived formally below. `M`'s eigenvalues classify the window:

- **Both eigenvalues small** — a flat region (a wall, the sky, an out-of-focus blur): `E` barely
  changes for any shift, so no direction of motion is recoverable. Neither LK nor census has anything
  to work with here; this project's confidence output (LK's small eigenvalue) and validity mask
  (census's left-right consistency) both correctly flag these pixels as untrustworthy.
- **One eigenvalue large, one small** — a straight EDGE. Shifting perpendicular to the edge changes `E`
  a lot; shifting ALONG it barely changes `E` at all. This is the **aperture problem**: looking through
  a small window at a straight edge, only the motion component perpendicular to the edge is locally
  observable — the tangential component is invisible no matter how sharp the edge is (project 01.04's
  THEORY.md derives the identical taxonomy for FAST/Harris corner detection; this project turns the same
  eigen-analysis into a continuous per-pixel CONFIDENCE rather than a binary corner/not-corner decision).
- **Both eigenvalues large** — genuine 2-D texture (or a corner): a shift in ANY direction changes `E`
  substantially, and the flow at this pixel is well constrained. This project's synthetic scene (a
  hashed multi-scale value-noise field, deliberately NOT a checkerboard — see "How we verify
  correctness") is built specifically to make this the common case almost everywhere, because DENSE
  flow — unlike 01.04's sparse-feature pipeline, which can simply discard low-texture pixels — must
  report *something* at every pixel and be honest about how much to trust it.

### Engineering constraints a real robot imposes

A real optical-flow front end lives under hard timing and hardware budgets absent from this teaching
demo: a rolling-shutter CMOS sensor exposes each row microseconds after the row above it, so a
fast-rotating robot sees real geometric skew within one "frame" that neither algorithm here models
(project 01.10 addresses rolling-shutter correction specifically); motion blur from a slow shutter
relative to scene motion violates brightness constancy in a spatially-varying way no simple gain/offset
model captures; and a flow front end feeding a flight or ground controller at 30-60 Hz has a
single-digit-millisecond compute budget per frame — this project's ~4 ms measured GPU time (both
methods, all four scene pairs; see README "Expected output") is well inside that budget on a desktop
GPU, though a real embedded deployment (Jetson-class SoC, no discrete GPU) would need to re-measure on
that hardware, not assume desktop numbers transfer (see `PRACTICE.md` §2).

## The math

### Structure tensor and mismatch vector (Lucas-Kanade)

Starting from `E(u,v) = sum_w [I1(x+dx+u,y+dy+v) - I0(x+dx,y+dy)]^2` over a window `w` of pixels
`(dx,dy)`, first-order Taylor-expand `I1` around the CURRENT flow estimate `(u,v)` in an INCREMENT
`(du,dv)`:

```
I1(x+dx+u+du, y+dy+v+dv)  ~=  I1_w(x+dx,y+dy)  +  Ix0(x+dx,y+dy)*du  +  Iy0(x+dx,y+dy)*dv
```

where `I1_w` denotes I1 already sampled ("warped") at the current `(u,v)`, and — this is the specific
approximation this project's `lk_iterate_kernel` makes, and the one that makes repeated iteration cheap
— the spatial derivative is taken from **`I0`'s gradient**, fixed for the whole level, rather than
recomputing `I1`'s gradient at the moving warped location every iteration (an "inverse-compositional"-
style formulation; Baker & Matthews 2004 name and compare all four Lucas-Kanade variants). Substituting
`It = I1_w(x+dx,y+dy) - I0(x+dx,y+dy)` (the current mismatch) and minimizing `sum_w [Ix0*du + Iy0*dv -
(-It)]^2` over `(du,dv)` gives the normal equations

```
M [du; dv] = b,     M = [ Sxx  Sxy ]     Sxx = sum_w Ix0^2      bx = sum_w Ix0*It
                        [ Sxy  Syy ]     Syy = sum_w Iy0^2      by = sum_w Iy0*It
                                          Sxy = sum_w Ix0*Iy0
```

and the COMPOSITIONAL update rule for a pure-translation warp (composition of two translations is just
addition, and inverting a translation just negates it) is `(u,v) <- (u,v) - M^{-1}b`. Solved in closed
form via Cramer's rule for the symmetric 2x2 system:

```
det = Sxx*Syy - Sxy^2
du  = -(Syy*bx - Sxy*by) / det          (this project's kernels.cu/reference_cpu.cpp lk_iterate)
dv  = -(-Sxy*bx + Sxx*by) / det
```

`M`'s two eigenvalues are `lambda = trace(M)/2 +- sqrt((trace(M)/2)^2 - det(M))` (closed form for a
symmetric 2x2 matrix); this project reports `lambda_min` as the per-pixel CONFIDENCE (see "The aperture
problem" above — `structure_tensor_kernel` computes it once per pyramid level, before any iteration).

**Sanity check by construction (worth deriving once, on paper):** for a perfectly linear 1-D image
`I0(x) = x` and true displacement `u_true` (so `I1(x) = x - u_true`), starting from `u=0`: `It =
I1(x) - I0(x) = -u_true` everywhere, `Ix0 = 1`, `bx = -u_true`, `Sxx = 1`, giving `du = -(-u_true)/1 =
u_true` in ONE step — Newton's method converges immediately for a locally linear signal, exactly as
expected. Real (non-linear) image content needs the `kLkIterationsPerLevel` repeats this project runs.

### Pyramid displacement bound — why coarse-to-fine is not optional

The Taylor linearization above is only valid while `(u,v)` stays small enough that `I1`'s second-order
term is negligible over the window — informally, "the true displacement must be within about one pixel
of the linearization point for one Newton step to make real progress," and even with several iterations
a `k` x `k` window's LOCAL gradient signal only carries reliable information about motions on the order
of the window's own extent. This project's rotation+zoom scene produces flow magnitudes up to roughly a
dozen pixels at the frame's corners (`README.md`'s "The algorithm in brief" states the measured worst
case) — far outside a single 5x5 window's direct capture range. The pyramid's mechanism: at pyramid
level `L` (built by `kNumLevels-1` applications of 2x area-average decimation — see "The GPU mapping"),
a true image-plane displacement of `D` pixels at level 0 appears as `D / 2^L` pixels at level `L` — the
SAME physical motion, viewed at coarser spatial resolution, is a SMALLER pixel displacement, and
therefore within the linearization's reach. `run_pyramidal_lk_gpu`/`pyramidal_lk_cpu` solve at the
coarsest level first (small residual, converges reliably from a zero start), then **upsample the
flow field bilinearly AND multiply both components by 2** (`upsample_flow_kernel`/`upsample_flow_cpu` —
the factor of 2 is the pyramid's entire reason to exist: undo the `1/2^L` shrinkage one level at a
time) before iterating again at the next finer level, whose REMAINING residual (true flow minus the
propagated coarse estimate) is now small even though the true flow itself was not. README's
`pyramid_advantage` gate measures this directly: the SAME total iteration budget, spent with vs. without
this hierarchical initialization, differs by several times in accuracy on the rotation+zoom scene
(measured ~4.1x — see README "Expected output").

### Census transform and the Hamming metric

A pixel's **census signature** encodes, for each of the `kCensusBits = 24` neighbors in a 5x5 window
(the center excluded), a single bit: `1` if that neighbor's intensity is `>=` the center's, `0`
otherwise (Zabih & Woodfill 1994). Formally, for center pixel `p` and neighbor offsets `{q_k}`:

```
census(p)_k = [ I(p + q_k) >= I(p) ]          k = 0 .. 23
```

**Why this is invariant to ANY monotonically increasing brightness transform** (not merely a uniform
additive/multiplicative one, the stronger claim this project's brightness-robustness gate exercises):
let `f` be any strictly increasing function (a gamma curve, a smooth local gain, an additive offset —
anything order-preserving). Then `a < b <=> f(a) < f(b)` for all `a, b`, so applying `f` to every pixel
in the image leaves `I(p+q_k) >= I(p)` TRUE or FALSE exactly as before, for every `k`, at every pixel —
the signature, and therefore the Hamming distance between any two signatures, is UNCHANGED. This is
strictly stronger than FAST's/ORB's brightness-robustness claim (project 01.04's THEORY.md), which only
survives affine (uniform gain + offset) changes, because a rank-order argument works for the whole
family of monotonic curves, not just the linear ones.

Matching between two signatures `s1, s2` uses the **Hamming distance** — the number of differing bits —
computed as `popcount(s1 XOR s2)`: XOR sets a bit exactly where the two signatures disagree, and
population count sums them. This is a single hardware instruction on modern GPUs (`__popc()`, see "The
GPU mapping"), which is WHY binary descriptors and census-style matching displaced float-vector (SSD/
SAD) matching in latency-sensitive robotics pipelines: comparing two 24-bit signatures costs one XOR
plus one POPC, versus dozens of multiply-accumulates for an equivalent-size float window SSD.

**Sub-pixel refinement** fits a 1-D parabola through 3 Hamming-cost samples `c(-1), c(0), c(+1)` around
the winning integer displacement, independently along each axis:

```
offset = 0.5 * (c(-1) - c(+1)) / (c(-1) - 2*c(0) + c(+1))
```

the closed-form vertex of the unique parabola through those three points — a documented approximation
(a full 2-D quadratic fit through all 9 neighbors would use more information; "Where this sits in the
real world" names this as the production refinement).

### Farneback polynomial expansion (Milestone 3 — documented only, not implemented)

Farneback's method (Gunnar Farnebäck, 2003, "Two-Frame Motion Estimation Based on Polynomial
Expansion") replaces both LK's single first-order Taylor term and census's binary comparisons with a
LOCAL QUADRATIC (second-order) signal model, fit independently around every pixel:

```
f(x) ~= x^T A x + b^T x + c          (x is the 2-D offset from the expansion point)
```

where `A` (2x2 symmetric), `b` (2-vector), and `c` (scalar) are found by a WEIGHTED LEAST-SQUARES fit
of this quadratic to the actual intensity values in a window around each pixel (the weights are
typically a separable Gaussian-like kernel, both for spatial locality and to de-emphasize samples far
from the expansion point). Given two such quadratic models `f1` (around the point in frame 1) and `f2`
(the SAME point translated by the true displacement `d` in frame 2), and assuming the true image
signal is genuinely quadratic (so `f2(x) = f1(x - d)` exactly), algebraic expansion of `f1(x-d)` and
matching it term-by-term against `f2`'s coefficients gives:

```
A2*x + b2  =  A1*(x - d) + b1  =  A1*x - A1*d + b1        for all x
```

Matching the constant (x-independent) terms: `b2 = b1 - A1*d`, i.e. (assuming `A1` is well-conditioned,
the exact structure-tensor-style eigenvalue argument this project's confidence output already makes)

```
d = A1^{-1} * (b1 - b2) / 2                (using A ~= (A1+A2)/2 in practice for better conditioning)
```

— a CLOSED-FORM per-pixel displacement estimate from a SINGLE polynomial-coefficient comparison, no
iteration required at a given scale (though real implementations still run it in a pyramid, both for
large-motion capture range and because the local-quadratic assumption itself only holds over small
neighborhoods). The polynomial coefficients `A, b, c` are computed once per frame via a separable
convolution against a fixed basis (this is the expensive step: several convolutions per pixel to
extract 6 independent quadratic-fit coefficients, versus LK's 2 gradient convolutions or census's single
comparison pass) — OpenCV's `calcOpticalFlowFarneback` and its CUDA counterpart implement exactly this
pipeline. Farneback sits BETWEEN this project's two implemented methods in character: like LK, it
regresses on raw intensity (inheriting the SAME brightness-constancy fragility this project's gate
demonstrates for LK); like census, it produces a genuinely DENSE field with no explicit windowed-SSD
search; unlike either, it is naturally suited to a REGION-GROWING confidence/smoothness prior (the
polynomial coefficients vary smoothly across neighboring pixels, which production Farneback
implementations exploit to propagate estimates and reduce noise — this project's LK confidence and
census validity mask are simpler, purely-local analogues of the same idea).

## The algorithm

### Milestone 1 — dense pyramidal Lucas-Kanade

```
1. Build the image pyramid (levels 0 = full res .. kNumLevels-1 = coarsest) for BOTH frames,
   via repeated 2x area-average decimation.                          O(W*H) total (geometric series)
2. Initialize the coarsest USED level's flow field to (0,0).                              O(1)
3. For level L = (num_levels-1) down to 0:
     a. Compute Scharr gradients (Ix0, Iy0) of level L's frame 0.                    O(W_L*H_L)
     b. Compute the structure tensor (Sxx,Syy,Sxy) and confidence (lambda_min)
        over a 5x5 window at every pixel, ONCE for this level.                       O(W_L*H_L * 25)
     c. Repeat kLkIterationsPerLevel times:
          - bilinear-warp frame 1 by the running flow estimate over the 5x5 window,
            accumulate the mismatch vector (bx,by), solve the 2x2 system, clamp,
            add the increment to the running flow field (in place).                  O(W_L*H_L * 25)
     d. If L > 0: bilinear-upsample the flow field to level L-1's resolution
        and multiply both components by 2 (see "The math").                          O(W_{L-1}*H_{L-1})
4. Level 0's final flow field and confidence map are the answer.
```

Serial (CPU) cost: dominated by step 3c, `O(kNumLevels * kLkIterationsPerLevel * W*H)` (the geometric
sum over levels is dominated by level 0). Parallel (GPU) cost: EVERY per-pixel operation within one
kernel call is embarrassingly parallel (`W_L*H_L` independent threads); the SEQUENCE of kernel calls
across levels and iterations is NOT parallelizable (level L+1 needs level L's finished result) — see
"The GPU mapping" for why this specific shape (parallel-within, sequential-across) is the single most
important mapping idea in this project.

### Milestone 2 — census-transform block-matching flow

```
1. Compute the 24-bit census signature at every pixel of BOTH frames.                O(W*H * 24)
2. For every eligible pixel in frame 0 (the "reference"):
     search all (2R+1)^2 candidate displacements in frame 1 (the "target"),
     Hamming winner-take-all, then parabolic sub-pixel refinement.        O(W*H * (2R+1)^2) = O(W*H*169)
3. Repeat step 2 with frame 1 as reference, frame 0 as target (the "backward" pass).  O(W*H*169)
4. Left-right consistency check: for each forward match, verify the backward match
   (sampled near the forward-predicted target) points back close to the origin.      O(W*H)
```

No pyramid, no iteration — census's search RADIUS (not a linearization) directly bounds its capture
range; the trade-off (versus LK) is quadratic cost in the search radius rather than linear cost in
iteration count, and integer-only (no sub-window-accuracy-below-the-parabola) localization at its core.

## The GPU mapping

**The one big idea:** every kernel in this project maps ONE GPU THREAD to ONE PIXEL (or, for census
matching, one REFERENCE pixel with an in-thread search loop over candidates) — a MAP or STENCIL
pattern, entirely independent across threads within a single kernel launch. What is NOT parallel is the
SEQUENCE of kernel launches: pyramid levels must run coarse-to-fine in order (each depends on the
previous level's finished flow field), and LK's iterations within one level must run in order (each
reads the flow field the previous iteration wrote). `run_pyramidal_lk_gpu` (`kernels.cu`) is therefore
HOST orchestration code — a C++ loop issuing one kernel launch after another — not itself a kernel; this
"host loop drives a chain of otherwise-parallel kernels" shape recurs throughout this repository
whenever an algorithm has an inherently sequential outer structure (compare 08.01 MPPI's per-tick host
loop).

**Memory hierarchy choices** (all documented as the project's teaching-simplicity default, with the
tiled alternative named as an exercise rather than implemented — the repeated pattern of every stencil
kernel in this repository):

- **`__constant__` memory** for the census offset table (`kCensusDxDev/kCensusDyDev` — mirroring project
  01.04's identical use for its FAST circle table): every thread in every block reads the SAME 24 fixed
  offsets on every launch, the textbook case for the constant cache's single-broadcast-per-warp
  behavior.
- **Global memory, no shared-memory tiling**, for every stencil (Scharr, structure tensor, census
  transform, LK's warped mismatch accumulation, census's block-matching search): each thread
  independently re-reads its own window from global memory. `census_match_kernel`'s 169-candidate
  search is the kernel where this costs the most — neighboring threads' search windows overlap heavily
  (a pixel one column over searches almost the identical 169 target locations, shifted by one column),
  so a shared-memory TILE of `census_tgt` sized `(blockDim + 2*kCensusSearchRadius)^2` per block, staged
  once and reused by every thread in the block, is the natural next optimization — left as README's
  census-tiling exercise so a learner derives the tile-size arithmetic themselves, rather than reading
  it pre-solved.
- **Registers** for every per-thread accumulator (`Sxx/Syy/Sxy`, `bx/by`, the running `best_cost` in
  census matching) — private per thread, read/written many times per kernel body, the fastest memory
  available; no thread ever needs another thread's accumulator.

**Occupancy**: every kernel launches a 16x16 (256-thread) 2-D block grid sized to cover the level's
`W_L x H_L` (or the full `kW x kH` for census) with a ragged-tail guard — at this project's resolutions
(from 40x30 up to 160x120) that is dozens to low hundreds of blocks, comfortably saturating an RTX 2080
SUPER's 46 SMs many times over even at the coarsest pyramid level.

## Numerical considerations

**Gradient NORMALIZATION is load-bearing, not cosmetic** (the single most important numerics lesson
this project teaches, root-caused empirically while building it — see "How we verify correctness" for
the measured before/after). The raw integer Scharr convolution over-reports the true per-pixel intensity
derivative by a factor of 32 (the stencil's positive-side weights sum to 16, and a central difference
spans 2 pixels: `16 * 2 = 32`). Lucas-Kanade's solved step is `M^{-1}*b`; `M` is QUADRATIC in the
gradient's scale (every term is a product of two gradient samples) while `b` is LINEAR in it (one
gradient sample times a raw intensity difference that does not itself scale) — so an unnormalized
gradient scaled by `k` shrinks the SOLVED DISPLACEMENT by a factor of `k`. This does not look like a
crash or an obviously-wrong answer: it looks exactly like "slow convergence," because each Newton step
still points in the correct direction, just takes a step `k` times too small — a genuinely easy bug to
mistake for "needs more iterations" rather than "needs correct units." The fix, `scharr_gradient_kernel`
dividing by `32.0f`, is an EXACT float32 operation (32 is a power of two — only the exponent changes, no
rounding, no loss of the bit-exactness the gradient stage's VERIFY claims).

**Bilinear-warp interpolation error.** `lk_iterate_kernel`'s mismatch accumulation samples `I1` at a
fractional coordinate every iteration; bilinear interpolation is exact for content that is locally
BILINEAR (rare) and introduces a smoothing/blurring bias otherwise — high-frequency content (this
project's finest 4 px noise octave) is attenuated slightly by every resample, a small but real
contributor to residual error beyond what the linearization itself explains. Real implementations
sometimes use higher-order (bicubic) resampling for the same reason census's sub-pixel accuracy is
inherently limited by its integer core (see below).

**Float vs. double accumulation is a DELIBERATE independent-path choice, not carelessness.**
`kernels.cu`'s structure-tensor and mismatch accumulations run in `float` (matching the GPU's native
arithmetic); `reference_cpu.cpp`'s twins accumulate the SAME sums in `double` before narrowing to
`float` at the end — an intentionally DIFFERENT numerical path from the GPU's (the same choice project
01.04's Harris CPU twin makes, for the identical reason: a shared accumulation order would make the twin
comparison blind to an accumulation-order bug). This is why `VERIFY(lk_flow)` is TOLERANCE-checked
(0.25 px), not bit-exact, while `VERIFY(gradient)`, `VERIFY(census_transform)`, and the integer half of
`VERIFY(census_match)` ARE bit-exact — every operation in those three is either an integer sum/XOR/POPC
or an exact power-of-two scale, with no accumulation-order sensitivity at all.

**Census's inherent sub-pixel floor.** Because the WTA search only ever considers INTEGER
displacements, and the true displacement in the rotation+zoom pair is generally NOT an integer, even a
perfect parabolic fit can only partially recover the fractional part — and the fit itself degrades when
the underlying image content changes rapidly relative to the search step (a sub-pixel rendering
residual between two frames sampled at slightly different sub-pixel offsets can flip a census bit that
an exact-integer-motion scene would never flip). This project's translation ground truth is
DELIBERATELY chosen as an exact integer, `(3.0, -3.0)` px (see "How we verify correctness" for why —
this was an empirical fix, not the original design), specifically so census's gated "exact" translation
test is not confounded by this inherent floor; the rotation+zoom scene (non-integer flow almost
everywhere) is NOT gated for census for exactly this reason (README states this scoping honestly).

**Angle wrapping / determinism.** No angle-wrapping hazard exists in this project (flow is a Cartesian
displacement, never an angle); the HSV flow-visualization's `atan2` IS an angle, wrapped to `(-180,
180]` for hue mapping, but that wrapping is purely cosmetic (an artifact-rendering choice, not a
numerical correctness concern). Every RNG (`make_synthetic.py`'s hashed lattice noise) is seeded
(seed 42) and hand-rolled (xorshift32, never a language-standard-library RNG) for cross-language,
cross-run reproducibility (CLAUDE.md §12).

## How we verify correctness

**Bit-exact twins** (tolerance 0, GPU must equal CPU to the last bit): `census_transform` (pure integer
comparisons), the integer half of `census_match` (WTA displacement and Hamming cost — `__popc()` on the
GPU vs. the hand-rolled SWAR `popcount32_portable()` on the CPU, both computing the identical population
count), and the `gradient` stage (Scharr taps are exact integers scaled by an EXACT power-of-two `/32`
— see "Numerical considerations").

**Tolerance-checked twins** (deliberately DIFFERENT float accumulation order/precision — see above):
the full `lk_flow` pipeline (0.25 px on the final flow field, after 3 levels x 3 iterations of
float-vs-double bilinear-warp accumulation), the sub-pixel half of `census_match` (0.05 px, one
parabola evaluation), and the full `census_flow` pipeline (0.1 px plus an exact validity-mask match — the
mask itself is a `<=` threshold comparison on an already-tolerance-checked residual, so it is checked
for EXACT agreement, not tolerance, since a borderline residual could theoretically flip on ULP
differences between platforms — the committed sample's measured margin around the threshold made this a
non-issue in practice, see README).

**Independent, ground-truth-based GATES** (the class of bug a GPU-vs-CPU twin comparison structurally
cannot catch — both implementations could share an identical conceptual bug and still agree perfectly
with each other; project 01.04's `reference_cpu.cpp` header names the general principle, and this
project's OWN build process is a worked example of exactly that failure mode, described next).

**A real bug this process caught (worth recording honestly, CLAUDE.md's no-fabrication rule cuts both
ways — mistakes made and fixed are as instructive as the final numbers):** the unnormalized-gradient bug
described in "Numerical considerations" was present, IDENTICALLY, in both `kernels.cu`'s device code and
`reference_cpu.cpp`'s independently-written host code — a shared conceptual error (both authors, though
independent, made the exact same simplifying omission), so `VERIFY(lk_flow)` passed throughout (GPU and
CPU agreed with each other almost perfectly, both being wrong the SAME way). Only the `translation_lk`
gate against the KNOWN analytic translation caught it: measured mean flow on the translation pair was
`(0.64, -0.47)` against a ground truth of `(3.4, -2.7)` — a large, unmistakable, physically-impossible
error for a task with an exact closed-form answer. This is precisely why every project in this
repository carries BOTH a twin comparison and at least one twin-independent gate (the general ruling in
`docs/PROJECT_TEMPLATE/src/reference_cpu.cpp`'s header) — the twin proves the GPU is faithful to the
CPU; only an independent, ground-truth-based check proves the CPU (and, transitively, the GPU) is
faithful to reality.

**A second empirical fix, for completeness:** the FIRST version of this project's synthetic scene
weighted its texture octaves toward COARSE spatial scales (cell sizes 40/20/10/5 px, weights
45/28/17/10%), which is what dense LK's smooth-gradient assumption wants but starved census's 5x5 window
of genuine local contrast — nearby candidate shifts within a locally near-linear gradient region produce
near-identical (or exactly tied) census signatures, so the winner-take-all search had little real
information to disambiguate a precise match (`translation_census` measured mean EPE ~1.5-2.1 px across
several octave-weighting attempts, well outside a defensible tolerance). Two changes fixed it: (1)
rebalancing the octaves toward finer scales (32/16/8/4 px cells, weights 20/25/30/25%, with the finest
cell smaller than the 5x5 window itself — see `data/README.md`), and (2) — the change that mattered
more, isolated by testing it alone — choosing an EXACT-INTEGER ground-truth translation `(3.0, -3.0)`
px rather than a sub-pixel one, removing the inherent sub-pixel floor described in "Numerical
considerations" as a confound. After both fixes: `translation_census` mean EPE ~0.27 px (10,107/14,976
census-eligible pixels valid); see README "Expected output" for the full measured line.

## Where this sits in the real world

**OpenCV** ships production implementations of BOTH this project's methods and the documented-only
third: `cv::calcOpticalFlowPyrLK` (sparse, at named keypoints — this project's dense variant tracks
every pixel instead) and its CUDA counterpart `cv::cuda::SparsePyrLKOpticalFlow` /
`cv::cuda::DensePyrLKOpticalFlow`; `cv::optflow::calcOpticalFlowSF`/stereo `StereoBM`/`StereoSGBM` use
census-family costs (SGBM specifically borrows census/Hamming-style matching terms in its cost volume);
`cv::calcOpticalFlowFarneback` / `cv::cuda::FarnebackOpticalFlow` implement Milestone 3 exactly as
derived above. Production LK implementations differ from this project mainly in: robust (not plain
least-squares) error norms to downweight outlier pixels within a window; explicit per-pixel tracking
QUALITY output (closely related to this project's confidence, but often ALSO checking forward-backward
consistency the way this project's census stage does, giving LK the same occlusion-robustness trick);
and a trained/tuned decision for how many pyramid levels and iterations to spend, rather than this
project's fixed, honestly-labeled `kNumLevels=3`/`kLkIterationsPerLevel=3`.

**NVIDIA VPI** (Vision Programming Interface, the production successor to the older NVIDIA Optical Flow
SDK) ships a dedicated, FIXED-FUNCTION hardware optical-flow ACCELERATOR on Turing-and-later GPUs (a
block entirely separate from the CUDA cores this project's kernels run on) that computes a census-like
cost volume in silicon at real-time frame rates with near-zero CUDA-core load — the direct commercial
descendant of this project's Milestone 2 idea, at a hardware level this teaching project cannot reach
(THEORY does not claim to reproduce fixed-function silicon, only the algorithm it embodies).

**Learned (RAFT-era) optical flow** (Teed & Deng 2020's RAFT, and its many successors) replaced the
classical pipeline above with an end-to-end trained network: a learned per-pixel feature encoder (in
place of raw intensity/gradient/census), an explicit ALL-PAIRS correlation volume (a learned,
differentiable generalization of census's brute-force search), and a recurrent update operator (in
place of LK's fixed Newton iteration) that is itself TRAINED to converge in few steps. RAFT-family
methods now define the accuracy state of the art on public benchmarks (Sintel, KITTI-flow) by a wide
margin over classical LK/Farneback/census — but at real computational cost (a full forward pass through
a deep network per frame pair, versus this project's few-millisecond classical kernels) and with the
usual learned-model caveats (out-of-distribution scenes, dataset bias, and — for a robot — the
verification burden of trusting a black-box network's output versus this project's fully-inspectable,
hand-derivable normal-equations solve). Project 12.xx's learned-perception track is where this
repository takes up that side of the story; this project's classical, closed-form methods remain the
right first thing to understand, and often the right thing to actually ship, when the compute budget or
the certification story doesn't accommodate a trained network (see `PRACTICE.md` §4).
