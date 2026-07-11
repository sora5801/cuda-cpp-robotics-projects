# 01.05 — SIFT/SURF on GPU (harder, warp-level reductions): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Why "scale" is a real physical problem, not a software inconvenience

A pinhole camera projects a 3-D point at distance $Z$ onto the image plane at height $y = f \cdot Y / Z$,
where $f$ is the focal length and $Y$ the point's real-world height. Move the camera twice as close
($Z \to Z/2$) and the SAME physical feature — a corner, a bolt head, a QR-code module — doubles in
apparent size on the sensor. A robot that only ever recognizes a landmark at ONE apparent size is
useless the moment it approaches, backs away, or a teammate with a different camera/lens sees the same
place. This is not a corner case: every mobile robot re-visiting a location (loop closure), every
manipulator approaching a part it first saw from across a work cell, and every multi-robot team sharing
a map faces a genuine, unavoidable scale change between two views of the same thing. Project 01.04
(feature-pipeline) sidesteps this: its FAST/ORB pipeline finds corners at ONE fixed pixel scale, so its
ground-truth pair uses rotation + translation only. This project's whole reason to exist is the piece
01.04 could not do: recovering correspondence across a REAL zoom.

### Where the blur comes from, physically

A real lens has a finite aperture, so it does not form a perfect point image of a point source — it
forms an Airy disk / point-spread function (PSF), and the sensor's own pixel pitch adds further
low-pass filtering (anti-aliasing filters exist precisely to prevent moiré from undersampling). The net
effect: every real photograph is already smoothed by SOME amount before a single pixel is read out.
`kSigmaInputAssumed = 0.5` (px) in `src/kernels.cuh` is this project's stand-in for that pre-existing
optical/sensor blur — the same assumption Lowe's original SIFT paper makes, and the reason the FIRST
Gaussian-blur step in the pyramid blurs from 0.5, not from 0, up to `kSigma0 = 1.6`.

### Engineering constraints a real robot imposes on this stage

- **Latency budget.** A perception pipeline feeding a 10-50 Hz local planner has, at most, tens of
  milliseconds per frame for EVERYTHING upstream of planning — and SIFT, as this project measures
  honestly (see "How we verify correctness"), is the SLOW end of the feature-matching spectrum compared
  to 01.04's ORB.
- **Repeatability under real noise.** Sensor read noise, JPEG/H.264 compression artifacts, and rolling-
  shutter skew all perturb the exact pixel values a second view of the same place produces — a detector
  that only works on pixel-perfect synthetic renders teaches nothing about the real failure mode.
- **Power and thermal.** Every extra millisecond of GPU time on an embedded SoC (Jetson-class, see
  PRACTICE.md) is joules the battery does not get to spend on motors.

## The math

### The Gaussian scale-space and why THIS kernel

A continuous "scale-space" of an image $I(x,y)$ is a family $L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)$,
convolving with a 2-D Gaussian of standard deviation $\sigma$ (pixels):

$$G(x,y,\sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/2\sigma^2}$$

Why the Gaussian, and not, say, a box filter or a median? Witkin (1983) and Koenderink (1984) proved a
**scale-space uniqueness result**: if you demand that a linear, shift-invariant, isotropic smoothing
family (a) creates NO new local extrema as $\sigma$ increases (a "causality"/non-creation-of-structure
axiom — coarser scales should only ever SIMPLIFY the signal, never invent new detail) and (b) obeys a
semigroup law $G_{\sigma_1} * G_{\sigma_2} = G_{\sqrt{\sigma_1^2+\sigma_2^2}}$ (blurring twice composes
into one blur of the combined variance — the exact identity `src/main.cu`'s pyramid-building loop uses
to blur incrementally, level to level, instead of re-blurring from scratch each time), the kernel is
FORCED to be Gaussian, uniquely, up to the choice of $\sigma$. This is why every credible scale-space
detector (SIFT included) uses a Gaussian, not an arbitrary "blur"-shaped kernel: it is the only choice
that does not accidentally fabricate features at coarse scales.

**Separability**, the property `launch_gaussian_blur`'s two-pass kernel exploits, follows algebraically
from the Gaussian's exponential form:
$$G(x,y,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{-x^2/2\sigma^2} \cdot \frac{1}{\sqrt{2\pi}\sigma}e^{-y^2/2\sigma^2} = g(x,\sigma)\,g(y,\sigma)$$
so a 2-D convolution with an $(2r+1)\times(2r+1)$ kernel ($O(r^2)$ taps per pixel) factors into two 1-D
passes of $(2r+1)$ taps each ($O(2r)$ taps per pixel) — the exact saving `kernels.cu`'s header explains.

### Difference-of-Gaussian ≈ scale-normalized Laplacian-of-Gaussian

The Gaussian obeys the **heat/diffusion equation** in $\sigma$: $\frac{\partial G}{\partial \sigma} =
\sigma \nabla^2 G$ (this is literally the heat equation with $\sigma^2/2$ playing the role of time).
Approximating the derivative by a finite difference between two nearby scales $k\sigma$ and $\sigma$:
$$\frac{\partial G}{\partial \sigma} \approx \frac{G(k\sigma) - G(\sigma)}{k\sigma - \sigma}
\quad\Rightarrow\quad G(k\sigma) - G(\sigma) \approx (k-1)\,\sigma\,\nabla^2 G(\sigma)$$
so the Difference-of-Gaussian, $D = L(\cdot,\cdot,k\sigma) - L(\cdot,\cdot,\sigma)$, is proportional to
$\sigma^2 \nabla^2 L$ — the **scale-normalized Laplacian**, exactly the operator Lindeberg (1994) proved
is necessary to make blob detection SCALE-INVARIANT in the first place (an un-normalized Laplacian's
response magnitude shrinks as $\sigma$ grows, so its raw extrema drift toward small scales; the
$\sigma^2$ factor cancels that drift). DoG is therefore not an ad-hoc trick — it is a cheap way (one
subtraction per pixel, already-computed pyramid levels) to approximate the ONE operator whose extrema
are provably scale-covariant: if a blob doubles in physical size, its DoG extremum moves to a scale
level exactly one octave higher, which is precisely what the `scale_recovery`/`scale_repeatability`
gates below measure and confirm.

### Pyramid geometry (this project's parameters, single-sourced in `kernels.cuh`)

$$\sigma_i = \sigma_0 \cdot k^{\,i}, \qquad k = 2^{1/s}, \qquad i = 0 \ldots s{+}2$$

with $\sigma_0 = 1.6$ and $s = $ `kIntervals` $= 2$ (so $k = \sqrt{2}$), giving `kImagesPerOctave` $= 5$
Gaussian levels and `kDogPerOctave` $= 4$ DoG levels per octave, of which the interior
`kFirstExtremaLayer..kLastExtremaLayer` $= [1,2]$ are searched for extrema (each needs a DoG layer above
AND below). Octave $o{+}1$'s level 0 is a $2\times$ nearest-neighbor decimation of octave $o$'s level
`kIntervals` image (sigma $2\sigma_0$ relative to ITS grid) — Lowe's standard "half the resolution,
double the sigma" octave step, safe from aliasing precisely because that source image is ALREADY blurred
to $2\sigma_0$ (see `kernels.cu`'s `downsample2x_kernel` header).

### Sub-pixel/sub-scale refinement (Brown & Lowe's method)

Model $D$ near an integer candidate $\mathbf{x}_0=(x_0,y_0,i_0)$ by its 2nd-order Taylor expansion in
the offset $\mathbf{z}=(dx,dy,ds)$:
$$D(\mathbf{z}) \approx D_0 + \nabla D^{\!\top}\mathbf{z} + \tfrac12\,\mathbf{z}^{\!\top} H \mathbf{z}$$
where $\nabla D = (D_x,D_y,D_s)$ (central differences) and $H$ is the $3\times3$ Hessian
$\begin{psmallmatrix}D_{xx}&D_{xy}&D_{xs}\\D_{xy}&D_{yy}&D_{ys}\\D_{xs}&D_{ys}&D_{ss}\end{psmallmatrix}$.
Setting $\nabla_{\mathbf{z}} D = 0$ gives the stationary offset $\mathbf{z}^\star = -H^{-1}\nabla D$ — a
**3x3 linear solve per candidate**, the single-instance analogue of the BATCHED small-matrix solves
project 33.01 studies (there: thousands of independent small systems solved in parallel; here: one
system per candidate, solved by one thread, iterated up to `kMaxRefineIters`$=5$ times with re-centering
when $|\mathbf{z}^\star|$ exceeds 0.5 in any axis — see `kernels.cu`'s `refine_keypoint_kernel`).

### Orientation histogram and the 128-D descriptor

**Orientation**: for each pixel in a Gaussian-weighted disk of radius $3\cdot1.5\cdot\sigma_{\text{oct}}$
around the keypoint, accumulate $m = \sqrt{g_x^2+g_y^2}$ (gradient magnitude) into a 36-bin histogram of
$\theta=\operatorname{atan2}(g_y,g_x)$, weighted by $e^{-(dx^2+dy^2)/2(1.5\sigma_{\text{oct}})^2}$. The
peak bin (parabolically sub-bin-interpolated) becomes the dominant orientation; any OTHER local peak
$\geq 0.8\times$ the maximum spawns an additional, independently-oriented copy of the same keypoint.

**Descriptor**: sample a $4\times4$ grid of cells, each spanning `kDescScaleFactor`$\cdot\sigma_{oct}=
3\sigma_{oct}$ pixels, ROTATED by $-\theta$ (the dominant orientation) so the whole grid is expressed in
a canonical, rotation-invariant frame; each sample's gradient (also expressed relative to $\theta$) votes
into 8 orientation bins per cell via TRILINEAR interpolation across (row, col, orientation) — 128
numbers total. Final: $L2$-normalize $\to$ clip each component at 0.2 $\to$ re-normalize (see
"Numerical considerations" for why clipping and why re-normalizing CAN push a component back above 0.2).

## The algorithm

| Stage | Serial cost | This project's GPU mapping |
|---|---|---|
| Gaussian pyramid | $O(N \cdot L \cdot r)$, $N{=}$px, $L{=}$levels, $r{=}$blur radius | one thread per OUTPUT pixel, 2 passes/level (separable) |
| DoG | $O(N \cdot D)$, $D{=}$DoG levels | one thread per pixel, pure map |
| Extrema | $O(N \cdot D_{\text{interior}} \cdot 26)$ | one thread per pixel, 3x3x3 stencil + atomic compaction |
| Refine | $O(C \cdot \text{iters} \cdot 27)$, $C{=}$candidates | one thread per candidate, iterative 3x3 solve |
| Orientation | $O(K \cdot A)$, $K{=}$keypoints, $A{=}$patch area | **one WARP per keypoint**, private-then-shuffle-reduce |
| Descriptor | $O(K \cdot A' \cdot 8)$, $A'{=}$larger window | **one WARP per keypoint**, same pattern, 128 bins |
| Match | $O(Q \cdot T \cdot 128)$, $Q,T{=}$query/train counts | one thread per query, brute-force all-pairs |

Every row up through Refine is a direct extension of 01.04's map/stencil/atomic-compaction patterns
(the 3x3x3 extrema test literally IS `nms_select_fast_kernel`'s 3x3 test, one dimension bigger).
Orientation and Descriptor are where this project's assigned GPU hook — warp-level reductions — actually
lives; see "The GPU mapping" below for the full argument.

## The GPU mapping

### The warp chapter (the centerpiece)

`orientation_kernel` and `describe_kernel` both launch **one block of exactly 32 threads (one warp) per
keypoint** (`orientation_kernel<<<n, kWarpSize>>>` — see `kernels.cu`). The reasoning, worked in full in
that file's header, boils down to a granularity argument: a keypoint's sampling patch can be hundreds to
thousands of pixels (far too much serial work for one thread) but the PER-KEYPOINT problem is still too
small, and the histogram accumulation too contention-prone, to hand one thread per SAMPLE PIXEL across
ALL keypoints in one flat launch. A warp is the natural middle granularity: 32-way parallelism per
keypoint, entirely self-contained in one block.

The kernel proceeds in two phases with a specific, load-bearing memory strategy:

```
Phase 1 (embarrassingly parallel):           Phase 2 (the warp-shuffle TREE reduction):
  lane 0  --> local_hist[36] (private)          for each of the 36 bins:
  lane 1  --> local_hist[36] (private)             v = local_hist[bin]           (this lane's partial)
  lane 2  --> local_hist[36] (private)             v += shfl_down(v, 16)         |  32 partials -> 16
  ...                                               v += shfl_down(v, 8)         |  16 -> 8
  lane 31 --> local_hist[36] (private)              v += shfl_down(v, 4)         |   8 -> 4
  (each lane strides over 1/32 of the                v += shfl_down(v, 2)         |   4 -> 2
   patch's pixels, ZERO cross-lane traffic)          v += shfl_down(v, 1)         |   2 -> 1
                                                     lane 0 writes hist[bin] = v  (register->register only)
```

**Why not the naive `atomicAdd(&shared_hist[bin], weight)` per sample?** With only 36 (or 128) shared-
memory slots and 32 lanes voting simultaneously, the birthday paradox guarantees frequent collisions —
and collisions on a hardware atomic SERIALIZE. Worse, the collision rate is HIGHEST exactly where SIFT
is designed to work best: near a real corner or blob, most samples' gradients cluster in 1-2 dominant
directions, so most atomics pile onto the SAME one or two bins. The local-accumulate-then-shuffle-reduce
scheme sidesteps this entirely: Phase 1 touches no shared state at all (perfectly parallel, by
construction), and Phase 2's cross-lane communication happens via `__shfl_down_sync`, a single-cycle-
class register-to-register operation with no memory traffic whatsoever — the fastest form of
inter-thread communication a GPU offers.

**Two different fan-out strategies, side by side, on purpose.** `dog_extrema_candidates_kernel` uses
ATOMIC COMPACTION (a global counter, `atomicAdd`) because the number of extrema found is genuinely
unbounded and data-dependent ahead of time. `orientation_kernel`'s multi-orientation spawn instead uses
FIXED SLOTS: block `kp_idx` owns a private sub-range `out[kp_idx*kMaxOrientedPerKeypoint .. +cap)` and
fills it with a purely LOCAL counter — no atomics, no cross-block contention, AND (the property this
project's staged verification needs) a fully DETERMINISTIC output order that lines up index-for-index
with the CPU twin's natural sequential order. The lesson: atomics are the right tool when fan-out is
UNBOUNDED and shared; fixed slots are the right (cheaper, deterministic) tool when fan-out is BOUNDED and
known ahead of time, per producer. Getting this distinction backwards here — using a global atomic
counter for orientation spawning, which this project's FIRST implementation did — produced an
output order that depends on which GPU block happens to finish first, silently breaking any attempt at
a clean, index-aligned GPU-vs-CPU comparison (root-caused during development; see `main.cu`'s
`compact_oriented()` for the fix).

### Everything else, briefly

The Gaussian blur, DoG, and downsample kernels are one-thread-per-pixel maps/stencils, deliberately
WITHOUT shared-memory tiling — an honestly-named simplification (see `kernels.cu`'s header): at blur
radii up to `kMaxGaussRadius`$=24$, a tile large enough to help a 16x16 block needs $(16{+}48)\times16$
floats of shared memory, a real optimization this project's complexity budget was spent elsewhere on
(the warp kernels) instead of chasing. `refine_keypoint_kernel` and `match_l2_kernel` are one-thread-
per-item kernels (variable-iteration-count and brute-force-inner-loop respectively) — no shared memory
needed, since neither shares data across threads.

### Occupancy, for the numbers actually measured on this project's scale

An RTX 2080 SUPER has 46 SMs. `orientation_kernel`/`describe_kernel` launch `n` blocks of 32 threads
(1 warp each) — for this project's typical keypoint counts (tens to ~90 per image, per the measured
numbers in "How we verify correctness"), that is far fewer blocks than 46 SMs can run concurrently: the
GPU is NOT saturated, and total kernel time (measured ~7-9 ms for the ENTIRE pipeline across all three
images, Release) is dominated by fixed per-launch overhead, not by real parallel throughput. This is
named honestly rather than hidden: SIFT at real-camera resolutions and keypoint counts (thousands, not
tens) is exactly the regime where this one-warp-per-keypoint mapping starts to actually fill the GPU —
see "Where this sits in the real world" for how production implementations exploit that.

## Numerical considerations

### Precision and the twin strategy, stage by stage

This project's GPU-vs-CPU comparison strategy is deliberately DIFFERENT at each stage (full argument in
`kernels.cuh`'s and `reference_cpu.cpp`'s headers) because each stage has a different SOURCE of possible
divergence:

- **Scale space (blur+DoG)**: both sides consume the SAME shared Gaussian weight table (built once by
  `build_gaussian_kernel_1d`, single-sourced data — see "How we verify correctness" for why this is data,
  not algorithm), so any measured difference isolates summation-order/FMA-fusion effects ONLY. Measured
  on the committed sample: max$|$gpu$-$cpu$|$ $\approx 3$-$6\times10^{-7}$ — machine-precision float32
  noise (float32 has $\approx 7$ significant decimal digits; this divergence sits right at that floor).
- **DoG extrema**: candidate SETS are compared, not element-wise, because the (float-tolerance-close,
  not bit-identical) pyramids feed a STRICT inequality test — a pixel sitting exactly at that boundary
  can legitimately flip sides between the two pipelines. Measured: 0 boundary-tie mismatches on the
  committed sample (133/106 candidates, all common) — the float-level agreement above is tight enough
  that no candidate happened to sit exactly on a tie in this run; a different GPU architecture's FMA
  scheduling could, in principle, produce a handful, which is exactly why this is a SET comparison and
  not a stricter one.
- **Refine**: an iterative, RE-CENTERING solve — a different rounding at iteration 1 could, in principle,
  walk toward a different integer neighbor and diverge onto a different PATH, not just a different
  answer along the same path. This project's CPU twin deliberately stays in FLOAT precision (not double,
  unlike 01.04's Harris precedent) specifically to avoid manufacturing that kind of divergence — see
  `reference_cpu.cpp`'s header for the full reasoning. Measured: identical accepted counts both images
  (78/78, 50/50).
- **Orientation / Descriptor — THE numerics lesson this project teaches.** The GPU sums each histogram
  bin via a 32-lane warp-shuffle TREE (5 pairwise-reduction steps, each pair added in an order that
  depends on which lanes are still "active"); the CPU sums the SAME per-sample contributions
  SEQUENTIALLY, one at a time, in raster order. **Floating-point addition is not associative** —
  $(a+b)+c$ and $a+(b+c)$ can differ once rounding enters — so these two summation ORDERS can, in
  principle, disagree in the last few bits even though every individual per-sample term is computed
  identically on both sides. Measured on the committed sample: max$|$gpu$-$cpu$|$ theta $= 0.00000$ rad
  and max$|$gpu$-$cpu$|$ descriptor component $=0.00000$ (below float32's printable precision at 5
  decimals) — the reordering effect is real but, for these keypoint counts and patch sizes, smaller than
  can be printed; `main.cu`'s tolerances (`kTolOrientationRad=0.05`, `kTolDescriptorComp=0.02`) are set
  from this measurement with honest margin, not tightened to the observed zero (a different GPU, a
  larger scene, or more keypoints could measure a real, nonzero value).
- **Match (L2)**: both sides run on the SAME (already GPU-computed) descriptor arrays, in the SAME
  dimension order — the one stage with no reduction-order freedom left to diverge on. Measured:
  max$|$gpu$-$cpu|$ dist$^2$ $=1.192\times10^{-7}$, 0 best-index mismatches.

### Angle wrapping and the row-axis sign convention (a real, worth-teaching gotcha)

`orientation_kernel` computes $\theta=\operatorname{atan2}(g_y,g_x)$ with $g_y$ built (central
difference `img[y-1]-img[y+1]`) to be POSITIVE when the image is brighter UPWARD on screen — so
increasing $\theta$ sweeps COUNTER-CLOCKWISE as actually displayed. `main.cu`'s `forward_transform`,
meanwhile, applies a textbook rotation matrix directly to `(x, row)` — and because pixel ROWS increase
DOWNWARD on screen, that "+theta" parameter is a CLOCKWISE rotation as displayed. Both conventions are
individually, internally correct; they simply point opposite ways. The consequence, measured directly
on the committed sample (every genuinely-corresponding keypoint pair's orientation delta clusters near
$-20°$, not $+20°$, for a $+20°$ transform parameter): `main.cu`'s `rotation_recovery` gate compares
against $-$`kTransformThetaDeg`, with the derivation spelled out at that gate's definition rather than
silently flipping a sign. This is a classic image-processing trap (any time a "math" rotation is applied
to array indices without accounting for which way rows run) — internalize it, because it recurs in every
project that mixes a geometric transform convention with an image-gradient convention.

### Descriptor normalization's overshoot (why the gate's ceiling is above the clip value)

Clipping each of the 128 components at 0.2 can only ever SHRINK the vector's norm (never grow it), so
the SECOND $L2$-normalize (renormalizing the clipped vector back to unit length) necessarily SCALES UP —
meaning a component that was exactly at the 0.2 clip boundary can end up measurably ABOVE 0.2 after this
step. Measured on the committed sample: max component $=0.3671$ across 155 descriptors — real,
reproducible, expected SIFT-descriptor behavior, not a bug (`main.cu`'s `descriptor_normalization` gate
uses a MEASURED ceiling, $0.42$, not a naive "must stay $\leq 0.2$" bound, for exactly this reason).

## How we verify correctness

Every stage's CPU reference (`reference_cpu.cpp`) is an INDEPENDENTLY hand-typed re-implementation, per
the repo's twin-independence ruling (see `kernels.cuh`'s header for the full statement): data-LAYOUT
contracts (structs, constants, the Gaussian weight-table builder) are single-sourced and shared —
duplicating a closed-form weight formula would be pure transcription, not independence — while every
ALGORITHMIC core (the convolution loop, the extrema test, the 3x3 solve, the histogram accumulation, the
matcher) is written twice. Staged verification (see "Numerical considerations" above for the tolerance
at each stage) means each stage's twin comparison uses the PREVIOUS stage's GPU-computed, already-
verified output as shared input to both sides — exactly 01.04's precedent of sharing the quantized
orientation bin between its GPU and CPU describe paths, extended here to the scale-space image and the
refined keypoint list.

**Independent gates** (none routed through any GPU-vs-CPU comparison — each checks something the twins
structurally cannot): `scale_recovery` (median matched-pair scale ratio $=1.4281$ vs. ground truth
$1.50$, $4.8\%$ relative error), `rotation_recovery` (median delta $-20.2523°$ vs. expected $-20.0°$,
$0.25°$ error), `transform_inlier` ($4/15=26.7\%$ of accepted matches land within 6 px of the true
transform), `scale_repeatability` ($20/78=25.6\%$ of scene-A keypoints re-found in scene B at the
transform-predicted location AND scale band — the gate 01.04's single-scale FAST structurally cannot
pass under a real zoom; see README "Limitations & honesty" for why FAST is documented, not
re-implemented, for a head-to-head number), `negative_control` ($0/14=0\%$ of A-vs-unrelated-scene
matches land near the true transform — proof the matcher and gates are not self-confirming), and
`descriptor_normalization` (every descriptor's $L2$ norm within $2.49\times10^{-7}$ of 1, max component
$0.3671 \le 0.42$).

**The matching-threshold story, told honestly.** Early development used Lowe's classic ratio
`kLoweRatioSift=0.75`. Measured on the committed sample, this let through as few as 1-6 accepted
matches, several of which were geometrically WRONG despite passing both the ratio test and mutual
cross-check — root-caused (see README "Limitations & honesty" and the scene-design notes in
`scripts/make_synthetic.py`) to a genuine property of this project's synthetic geometric content: a
right-angle checkerboard corner, once rotation/scale-normalized, lives in a low-dimensional shape
family, so a meaningful minority of geometrically UNRELATED keypoint pairs coincidentally resemble each
other in 128-D descriptor space — occasionally MORE closely than a true match resembles itself. A
measured "too good to be true" signature (unrelated pairs at squared-L2 distance $\approx 0.02$-$0.07$,
while every confirmed TRUE correspondence measured $\approx 0.3$-$0.98$) motivated `kMinL2DistSq=0.15`,
an explicit floor rejecting implausibly-perfect matches, alongside a loosened `kLoweRatioSift=0.92`
(measured-then-margined, like every other tolerance in this file) — real photographic content would
never need either adjustment, and that gap is itself the honest finding: SIFT needs richer texture than
pure synthetic geometry provides, discussed further below.

## Where this sits in the real world

**OpenCV's `cv::SIFT`** (patent-expired since 2020, now in the main `opencv` module, not
`opencv_contrib`) implements the SAME Lowe 1999/2004 algorithm this project teaches, at PRODUCTION
scale: typically 4 octaves x 3 intervals over a $2\times$-upsampled input, yielding hundreds to
thousands of keypoints per real photograph — the "where the warp mapping actually saturates a GPU"
regime this project's THEORY.md "GPU mapping" section names but cannot reach at teaching scale.
**OpenCV's CUDA module** (`cv::cuda::SIFT`, and historically `xfeatures2d::SURF_CUDA`) parallelizes
essentially this project's same stage list, at that larger scale — study it after this project to see
the identical mapping applied where it matters. **VLFeat**'s `vl_sift` is the classic, extremely
well-documented open-source REFERENCE implementation (Andrea Vedaldi's) worth reading line-by-line for
exactly the small numerical choices (histogram smoothing, exact edge-of-grid interpolation rules) this
project's `THEORY.md`/`kernels.cu` comments call out. **Learned features — SuperPoint (DeTone et al.
2018) and DISK (Tyszkiewicz et al. 2020)** are the modern successor: a CNN trained end-to-end to jointly
predict keypoints AND descriptors directly from data, which sidesteps exactly the "hand-designed
descriptor's discriminative power depends on the richness of local texture" limitation this project's
"How we verify correctness" section measured directly — a learned descriptor can, in principle, learn
to discriminate SIMPLE geometric corners just as well as richly-textured ones, at the cost of needing a
trained model and a GPU inference pass instead of a closed-form formula. Production SLAM/VO stacks
(ORB-SLAM3, COLMAP, many industrial AMR localization stacks) still favor HAND-DESIGNED binary
descriptors (ORB, per 01.04) for the real-time front end specifically because of the Hamming-vs-L2 cost
gap this project measures directly (128 float subtract-multiply-adds vs. 8 uint32 XOR-POPCs per
comparison — see `kernels.cu`'s `match_l2_kernel` header), reaching for float descriptors like SIFT
mainly in offline/batch contexts (structure-from-motion, one-time map building) where latency matters
less than raw discriminative power.

**SURF (documented-only in this project — see README "Limitations & honesty" for the scoping)** is
Bay et al.'s 2006 speed-oriented answer to the same scale-invariance problem, built on two ideas this
section teaches to implementable depth without shipping code: (1) **integral images** — a single
$O(N)$ pre-pass ($S(x,y) = \sum_{x'\le x, y'\le y} I(x',y')$, computed via a 2-D prefix sum) that lets
the sum of ANY axis-aligned rectangle be read in $O(1)$ time (four array lookups: bottom-right $-$
top-right $-$ bottom-left $+$ top-left) — the same prefix-sum idea underlying the classic
Viola-Jones face detector's Haar features; and (2) a **box-filter Hessian approximation**: SURF replaces
SIFT's Gaussian-derivative Hessian ($D_{xx}D_{yy}-D_{xy}^2$) with a determinant built from simple
RECTANGULAR (box-shaped) second-derivative approximations, computable via a HANDFUL of integral-image
lookups regardless of the filter's size — meaning SURF's "scale space" is built by GROWING the box
filter's size at CONSTANT cost per evaluation, rather than repeatedly re-blurring the image (SIFT's
approach, which gets more expensive at coarser scales). SURF's 64-D descriptor similarly uses Haar-
wavelet responses (simple $+1/-1$ rectangular filters, also $O(1)$ via integral images) summed over a
$4\times4$ grid, in place of SIFT's raw-gradient orientation histogram. The upshot: SURF trades some of
SIFT's descriptor richness for roughly a $3$-$5\times$ real-world speed-up (Bay et al.'s reported
figures), a genuinely different point on the same speed/discriminability tradeoff this project's ratio-
test story above already surfaces directly — implementing it fully would double this project's kernel
count for a second matcher demonstrating the identical warp-level-reduction lesson on a different feature
formula, which the catalog's `[R&D]`/bundled-bullet scoping rule (CLAUDE.md §2, §13) explicitly permits
documenting rather than shipping.
