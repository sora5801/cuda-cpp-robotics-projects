# 01.21 — Scene flow from RGB-D pairs: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

A robot's RGB-D camera reports, at every instant, two things per pixel: a color and a depth (the
distance from the camera's optical center to whatever the ray through that pixel hit — Pcam.z, the
pinhole "z-buffer" convention, not Euclidean range). Consecutive frames, a fraction of a second
apart, give two such (color, depth) pairs. Physically, THREE things can have changed a scene point's
apparent position between those two frames:

1. **The camera moved.** Every static point in the world appears to shift, by an amount and
   direction that depends only on the camera's own rigid motion (rotation + translation) and that
   point's depth — this is *motion parallax*, the same geometric fact that makes distant mountains
   appear to move slower than a nearby fence post from a moving car.
2. **The scene point itself moved.** A person walking, a door swinging, a box sliding off a
   conveyor — genuine independent motion, unrelated to the camera.
3. **Nothing moved, but the correspondence is ambiguous or invalid** — occlusion (a surface visible
   in frame0 is hidden behind something in frame1, or vice versa), specular highlights that don't
   move with the surface, or sensor noise.

**Scene flow** is the dense field of true 3-D displacement vectors this process induces. Given only
the pixels (no external motion sensor), (1) and (2) look identical at any single pixel — a rigid-
motion-induced shift and a genuine independent motion are both just "this point moved from here to
there." The entire intellectual content of this project is the classical way to *tell them apart*:
(1) is caused by ONE rigid transform, shared by every static pixel in the scene simultaneously; (2)
is a LOCAL deviation from that shared transform, confined to the pixels of whatever is actually
moving on its own. Fit the ONE dominant transform robustly (so a minority of independently-moving
pixels cannot corrupt it), and every pixel that still disagrees afterward is either genuinely moving
or a hazard from case (3) — cataloged and bounded, not silently ignored.

**Why RGB-D (not stereo, not monocular) makes this metric.** A single camera's optical flow gives
only 2-D pixel displacement — motion toward or away from the camera is invisible (scale-ambiguous)
without a second source of depth. RGB-D's depth channel supplies exactly that missing dimension:
every pixel's flow can be *lifted* into a real, metric 3-D displacement (meters, not "some unknown
multiple of meters"), which is what makes a physically meaningful rigid-motion fit possible at all.

**Depth sensing physics (the engineering constraint that drives this project's noise model).** Most
depth cameras (structured-light like the original Kinect, active or passive stereo, some ToF
variants) recover depth from a measured *disparity* — the shift, in pixels, between corresponding
points in two views (two physical cameras, or a camera and a projected pattern treated as a virtual
second view). The classical triangulation relation is

```
z = f * B / d
```

where `f` is focal length (pixels), `B` is the baseline (meters, the physical separation between
the two viewpoints), and `d` is disparity (pixels). Disparity is measured to some roughly CONSTANT
resolution `Δd` (set by pixel pitch and matching precision, not by depth) — but because `z` depends
on the *reciprocal* of `d`, the depth measurement's OWN uncertainty grows quadratically with range:

```
dz/dd = -f*B/d^2 = -z^2/(f*B)   =>   sigma_z ~ z^2 * sigma_d / (f*B)
```

This is precisely why real RGB-D sensors are precise (millimeters) up close and imprecise
(centimeters, worsening toward decimeters) at their far range — and it is why this project's
synthetic depth noise (kernels.cuh's `kDepthNoiseAM`, `kDepthNoiseB`; the exact same constants
`scripts/make_synthetic.py` uses to noise the committed sample) is modeled as `sigma_z(z) = A + B*z^2`:
a constant electronics/quantization floor `A` plus a quadratic-in-range term standing in for the
`sigma_d/(f*B)` disparity-quantization mechanism above. `kDepthNoiseB` is deliberately kept modest
(THEORY.md "Numerical considerations" quantifies why) so the demo's noise floor stays well under the
object-motion signal even at this scene's ~9 m depth — an honest, stated simplification of the real
mechanism's typical magnitude, not a claim that real sensors are this quiet at long range.

**Engineering constraints a real robot imposes.** Color and depth streams must be time-synchronized
(PRACTICE.md §1 discusses the rolling-shutter/global-shutter and inter-sensor sync hazards this
implies); the whole pipeline must run inside the sensor's own frame period (README "System context"
— 15–30 Hz, tens of milliseconds); and — the reason this project bothers with *robust* fitting at
all — a real scene reliably contains SOME independently moving content (people, other robots,
doors), so "assume everything is static" is never a safe simplification outside a lab bench test.

## The math

**Frames and notation.** SI units throughout; right-handed frames (CLAUDE.md §12). Camera-optical
frame convention (z-forward, x-right, y-down — the standard pinhole convention, matching 01.18):
a pixel `(px, py)` with depth `d` back-projects to the camera-frame 3-D point

```
P = ( (px + 0.5 - cx)/fx * d,  (py + 0.5 - cy)/fy * d,  d )
```

(the `+0.5` is the pixel-CENTER convention — it must match whatever convention rendered the depth
map exactly, or every recovered point carries a systematic sub-pixel bias; `kernels.cu`'s
`backproject()` and `scripts/make_synthetic.py`'s `camera_ray_cam_frame()` are the two independent
places this convention is asserted, and they must agree).

**Raw scene flow.** Let `x` be a pixel in frame0 with depth `D0(x)`, and let `u(x) = (u,v)` be the
2-D optical flow taking `x` to its corresponding location `x+u` in frame1, with depth `D1(x+u)`
(bilinearly sampled, since `x+u` is sub-pixel). Define

```
P1 = backproject(x, D0(x))            -- in camera0's OWN frame
P2 = backproject(x+u, D1(x+u))        -- in camera1's OWN frame
F  = P2 - P1                          -- the "raw" scene flow this project's kernels.cuh calls F
```

`P1` and `P2` live in **different** reference frames (camera0's and camera1's own optical frames,
respectively) — `F` is not yet a physical displacement vector, it is the *observed* 3-D coordinate
change as measured independently by each frame's own sensor. This is a standard, useful
simplification valid because the two exposures are close together in time (the two frames are
"almost" sharing a frame) — and it is precisely why the next step (ego-motion fitting) is needed to
make it physically meaningful.

**Ego-motion as a rigid registration problem.** For a STATIC point, `P1` and `P2` are the SAME
physical point viewed from two nearby camera poses — related by a rigid transform:

```
P2 = T_gt(P1) = R_gt * P1 + t_gt      (for every STATIC pixel, up to sensor noise)
```

`T_gt` is *not* simply "the camera's motion" — it is the transform that predicts a static point's
camera1-frame coordinates FROM its camera0-frame coordinates, which is algebraically the *inverse*
composition of the camera's own physical displacement, expressed in the right basis. Concretely
(`scripts/make_synthetic.py`'s module docstring derives every step): if the camera's own world-frame
motion is `(R_ego, t_ego)` — camera1 sits at `CAM_POS0 + t_ego` with orientation `R_ego` relative to
world/BODY axes (x-forward, y-left, z-up) — and `M` is the FIXED permutation matrix converting
body-frame vectors to camera-OPTICAL-frame vectors (`M = [[0,-1,0],[0,0,-1],[1,0,0]]`, exactly
`kernels.cu`'s/`make_synthetic.py`'s `body_to_cam`), then

```
T_gt = (R_gt, t_gt) = ( M R_ego^T M^T,  M (-R_ego^T t_ego) )
```

**The M-conjugation is load-bearing, not decorative** — this project's own build got it wrong once
and it is worth understanding exactly why. `R_ego` is a rotation matrix whose NUMBERS are expressed
in BODY axes (e.g. "rotate 3° about the up axis"). `P1`/`P2` live in camera OPTICAL axes (a
*different*, fixed rotation away from body axes). A rotation matrix's entries are basis-dependent:
applying `R_ego^T` directly to an optical-frame vector silently rotates about the WRONG axis (a
rotation about the body's "up" axis becomes, once permuted into optical coordinates, a rotation
mixing the camera's OWN x/z axes — not x/y). The fix is the standard similarity-transform rule for
re-expressing a linear operator in a different basis: `A` acting on body-frame vectors becomes
`M A M^-1 = M A M^T` (since `M` is an orthogonal permutation) acting on optical-frame vectors. This
project's build caught the bug by fitting the (correct) Horn solve to *exact, noise-free* synthetic
correspondences and comparing the recovered rotation's axis against the intended one — an
independent, closed-form sanity check that is exactly the kind of "known-answer gate" CLAUDE.md's
twin-independence ruling asks every shared-code path to carry (see `reference_cpu.cpp`'s header).

**The moving object's residual.** The object additionally translates by `t_obj` (world/body frame,
no rotation — `kernels.cuh`'s file header explains why this simplification was chosen). Composing
through the identical algebra:

```
P2_obj = T_gt(P1_obj) + c_gt,      c_gt = M (R_ego^T t_obj)
```

`c_gt` is a CONSTANT vector (the same for every object pixel, since the object's motion has no
rotation) — this is the mathematical reason residual segmentation works at all: apply the correctly
recovered `T_gt` to every pixel's `P1`; static pixels' residuals collapse to ~sensor noise, object
pixels' residuals collapse to `~c_gt`, a fixed, non-zero, THRESHOLDABLE vector.

**Robust rigid-motion fitting (Horn's method).** Given weighted correspondences `{(P1_i, P2_i, w_i)}`,
the weighted least-squares rigid fit minimizing `sum_i w_i |R P1_i + t - P2_i|^2` has a classical
closed form (Horn, 1987): compute weighted centroids `mu1, mu2`; form the cross-covariance
`S[a][b] = E_w[P1_a P2_b] - mu1_a mu2_b` (the "E[XY]-E[X]E[Y]" identity, letting the covariance be
accumulated in ONE pass with no prior knowledge of the centroids — the same packed-accumulator idea
01.17's 28-scalar reduction uses, cited); build the 4×4 symmetric "key matrix"

```
N = [ Sxx+Syy+Szz   Syz-Szy       Szx-Sxz       Sxy-Syx     ]
    [ Syz-Szy       Sxx-Syy-Szz   Sxy+Syx       Szx+Sxz     ]
    [ Szx-Sxz       Sxy+Syx      -Sxx+Syy-Szz   Syz+Szy     ]
    [ Sxy-Syx       Szx+Sxz       Syz+Szy      -Sxx-Syy+Szz ]
```

The unit eigenvector of `N`'s MOST POSITIVE eigenvalue is the optimal rotation quaternion `(w,x,y,z)`;
translation follows as `t = mu2 - R(q) mu1`. `N`'s trace is identically zero (sum each diagonal
entry above: every `Sxx`/`Syy`/`Szz` term appears with coefficient `+1` twice and `-1` twice),
so its eigenvalues are not all the same sign in general — the target is the largest *signed*
eigenvalue, which is not automatically the largest in *magnitude* (see "The GPU mapping" below for
why this matters to how it is computed).

**Robustness.** A single unweighted fit over ALL pixels lets the moving object's `~c_gt`-sized
residuals bias the "dominant" transform toward a compromise between the background's true motion and
the object's — exactly the failure this project's `ego_motion_robustness` gate is designed to
demonstrate. Iteratively Reweighted Least Squares (IRLS) fixes this: fit once (uniform weights),
compute each point's residual under the current fit, derive a ROBUST scale from the residuals'
median absolute value (`sigma_robust = 1.4826 * median(|r|)` — the standard MAD-to-Gaussian-sigma
conversion, robust because the median itself tolerates up to ~50% contamination before breaking),
down-weight points whose residual exceeds a few robust-sigmas via the Tukey biweight
`w = (1-u^2)^2` for `|u|<1` (`u = r / (c * sigma_robust)`, `c=4.685` a standard tuning constant for
95% Gaussian efficiency), zero for `|u|>=1`, and refit. Repeated `kIrlsIterations` times, this
converges toward a fit dominated by the (majority) static background, having progressively excluded
the (minority) mover.

## The algorithm

1. **Pyramid build** (2 levels): area-average 2× downsample of both grayscale frames.
2. **Per level, coarse→fine:** Scharr gradients → 5×5 structure tensor (+ its small eigenvalue as a
   per-pixel confidence, the aperture-problem signal) → `kLkIterationsPerLevel` forward-additive LK
   refinement steps (bilinear-warp frame1, solve the local 2×2 normal equations, clamp the step) →
   bilinear-upsample the flow (with a ×2 magnitude scale) to seed the next finer level.
   Complexity: `O(W*H*window^2)` per level per iteration, embarrassingly parallel across pixels
   WITHIN a level; sequential ACROSS levels (coarse must finish before fine starts).
3. **3-D lift:** for every pixel whose confidence clears `kMinConfidenceForLift` and whose D0 is
   valid, bilinear-sample D1 at the flow-shifted location; if the 4 taps disagree by more than
   `kDepthEdgeGuardM`, reject (straddling a real depth edge); else back-project both `P1` and `P2`.
   `O(W*H)`, embarrassingly parallel.
4. **Robust ego-motion fit (`kIrlsIterations` rounds):** accumulate the weighted 16-scalar
   covariance record over all valid points (`O(n)`, a reduction), solve for `(R,t)` via Horn's
   method (`O(1)` — a fixed-size 4×4 eigenproblem, independent of `n`), compute residuals (`O(n)`,
   a map), derive the robust scale and next round's weights (`O(n log n)` for the median — the one
   genuinely sequential-feeling step, done on the host).
5. **Residual segmentation:** threshold the final round's residual magnitude against a noise-derived
   bound (`O(n)`), 3×3 morphological open (erode then dilate, `O(n)` each), then CONNECTED-COMPONENT
   LABELING (iterative 4-connected label propagation, `O(n * sweeps)`, `sweeps` bounded by the
   largest component's diameter — see "The GPU mapping") followed by a size-count-then-filter pass
   (`O(n)`, an atomic scatter then a map) that drops components no larger than
   `kMinComponentSizePx`.
6. **Object motion:** a robust (IRLS `kIrlsIterations` rounds, Tukey biweight — the SAME machinery
   as step 4) FIXED-ROTATION offset fit restricted to the segmented mask: `R` is held at the
   already-recovered `R_robust` and only the translation offset is re-estimated each round from the
   SAME weighted-covariance-reduce accumulator step 4 already uses (only its `sum_w`/`sum_w*P1`/
   `sum_w*P2` rows are needed — the cross-product rows a free Horn fit would need are simply
   unused). See "Numerical considerations" for why this replaced a free 6-DOF Horn fit.

## The GPU mapping

**Stages 1–3 and 5** are the repo's most familiar shape: one thread per pixel, no cross-thread
communication within a kernel, global memory only (a documented simplification — THEORY.md's
sibling projects 01.03/01.04 name the shared-memory-tiled speedup as an exercise; this project
inherits that same honest scoping). The one SEQUENTIAL dependency is the pyramid level loop (level
L+1's flow must exist before level L can use it as a starting point) — everything WITHIN a level is
embarrassingly parallel.

**Stage 4's reduction is the project's one new GPU idea.** `weighted_covariance_reduce_kernel`
gives each thread one point's 16-scalar contribution (`kernels.cuh`'s `kCovarWidth` layout: 1 sum
of weights + 3 weighted-P1 + 3 weighted-P2 + 9 weighted-outer-products), stages it in SHARED memory
(`kThreadsReduce=128` threads × 16 floats × 4 bytes = 8 KiB per block — comfortably within budget
alongside several resident blocks on sm_75..sm_89), and collapses it via the standard binary-tree
shared-memory reduction (halve the active thread count each step, each surviving thread adds its
mirror's record into its own). The HOST then sums the per-block partial records in DOUBLE precision
— the same "GPU reduces in float across a tree, host finishes the cross-block sum in double" split
02.06's ICP normal-equation assembly and 01.17's calibration use, cited — before handing the 16
doubles to `build_rigid_from_covariance16`.

**Why the eigenproblem is solved by POWER ITERATION, not a full Jacobi sweep** (contrast with
02.06's 3×3 Jacobi eigensolve for its own, different, symmetric-matrix problem): a full eigensolve
computes ALL 4 eigenpairs of the 4×4 `N`; this project only ever needs the DOMINANT one. Plain power
iteration (repeatedly apply `N` to a vector and renormalize) converges to the eigenvector of LARGEST
MAGNITUDE — but "The math" showed `N`'s trace is zero, so the target (largest SIGNED) eigenvalue is
not guaranteed to be largest in magnitude. The fix is a spectral SHIFT: Gershgorin's circle theorem
bounds every eigenvalue's magnitude by the sum of a row's (or, more crudely and just as validly
here, the WHOLE matrix's) absolute entries — call this bound `shift`. Adding `shift * I` to `N`
shifts every eigenvalue by the same `+shift` without touching any eigenvector; because `shift` is at
least as large as the most-negative eigenvalue's magnitude, EVERY shifted eigenvalue becomes
non-negative, so the original LARGEST (signed) eigenvalue is now unambiguously the largest in
magnitude too — guaranteeing plain power iteration converges to the right vector. This entire solve
is a fixed-size (4×4), `O(1)`-per-call HOST routine (not a kernel): it runs once per IRLS round on
16 already-reduced scalars, far too small to parallelize usefully, and is SHARED between the GPU
path and the CPU reference oracle (this file's twin-independence discussion covers why that sharing
is safe here, and what independent gate it needs — see "How we verify correctness").

**Connected-component labeling is the project's SECOND departure from thread-per-pixel-with-no-
communication** (after Stage 4's reduction): `ccl_propagate_sweep_kernel` still launches one thread
per pixel, but each thread reads its FOUR NEIGHBORS' labels (a genuine, if small, cross-thread
dependency within a kernel launch) and `atomicMin`s a smaller label back in. Because a label can
only ever DECREASE and is bounded below by 0, repeated sweeps provably converge to a UNIQUE fixed
point (the component's minimum linear pixel index) regardless of scheduling — the SAME "monotone
atomic + repeat until no change" idiom 01.06's and 30.01's own CCL stages use (cited in
`kernels.cuh`), just re-derived and re-typed here. The HOST owns the sweep LOOP itself (reading back
one `int` — "did anything change this sweep?" — after every launch), exactly like this project's own
Milestone-3 IRLS loop; on this project's 128×96 demo, convergence measures at a few dozen sweeps
(`main.cu`'s `[info] connected_components` line reports the exact count each run), comfortably under
the `kMaxCclSweeps=256` safety cap derived from this scene's `max(kW,kH)=128` worst-case component
diameter.

**Occupancy/bandwidth notes.** Every per-pixel kernel here is memory-bound (a handful of FLOPs per
byte moved) at this project's 128×96 demo scale — the GPU-vs-CPU speed-up (`[time]` lines) is a
launch-overhead-dominated, small-problem artifact, not a meaningful benchmark (CLAUDE.md §12); at a
real sensor's full resolution (e.g. 640×480 or larger) the SAME kernels become genuinely
bandwidth-bound and the block-sizes/shared-memory choices here would matter for real throughput.

## Numerical considerations

**Precision.** All GPU kernels operate in FP32; the reduction and the Horn solve accumulate in
FP64 on the host (`double c16[16]` in `main.cu`, `double` throughout `build_rigid_from_covariance16`)
— a fixed, small (16-scalar, then 4×4) computation where the precision cost of double accumulation
is negligible and the payoff (a rotation/translation accurate to sub-millimeter, sub-0.02° on this
project's demo) is real.

**Depth-noise propagation into 3-D position error (the theoretical half of the segmentation
threshold).** Back-projection is `P(px,py,d) = d * ray_dir(px,py)`, `ray_dir = ((px+0.5-cx)/fx,
(py+0.5-cy)/fy, 1)`. Holding pixel position fixed, `dP/dd = ray_dir(px,py)` — a first-order argument
valid because depth noise (millimeters to a few centimeters) is small relative to depth itself
(meters). `|ray_dir|` is minimized (=1) at the principal point and grows toward the image corners;
this project's `kMaxRayBoundFactor` (kernels.cuh) is `|ray_dir|` evaluated at the image corner —
`sqrt((64/118)^2 + (48/116)^2 + 1) ≈ 1.2104` — a single, slightly conservative bound used everywhere
rather than a per-pixel-varying one. Combining independent noise contributions from BOTH `P1` (via
`D0`) and `P2` (via `D1`) at similar depth gives a combined bound of roughly
`kMaxRayBoundFactor * sqrt(2) * sigma_z(z)` for the position-error scale at depth `z`.

**Why the OPERATIONAL segmentation threshold uses the MEASURED residual spread, not this formula
directly (an honest, characterized gap).** This project's build measured the theoretical bound
above (evaluated at this scene's mean depth, ≈7.6 m) at ≈0.017 m, while the ACTUAL measured
residual spread (via the same MAD-based robust scale the IRLS loop already computes) was
≈4–5× larger. The theoretical model above accounts ONLY for depth noise; it omits a second,
comparably-sized contributor — **2-D flow position uncertainty**, propagated through the SAME
back-projection Jacobian. Even a flow error of a fraction of a pixel at the ~9 m depths in this
scene corresponds to several centimeters of lateral 3-D position error (a fraction of a pixel times
depth/focal-length) — comparable to or larger than the depth-noise-only term. Rather than silently
fold an unexplained fudge factor into the theoretical formula, this project's segmentation threshold
is set directly from THIS RUN's own measured residual spread (`kSegThresholdKSigma` robust-sigmas
above the MAD-based scale), with the theoretical depth-noise-only bound printed alongside as an
honest order-of-magnitude cross-check (`main.cu`'s `noise_derivation` `[info]` line) — physically
motivated in FORM (why residuals have a scale worth thresholding at all, and roughly why), grounded
in MEASUREMENT for the operational VALUE.

**Flow endpoint-error outliers (a measured, characterized limitation, not a hidden one).** The
committed demo's 2-D flow MEDIAN endpoint error is ≈0.25 px — Lucas-Kanade recovers the true motion
to a fraction of a pixel almost everywhere — but the MEAN is ≈2.4 px, because a genuine minority
(~15–20% of "confident" pixels, by measurement) sit at occlusion/disocclusion boundaries where the
brightness-constancy assumption breaks: the frame0 content simply does not correspond to anything
correct in frame1 (occluded, revealed, or view-dependent shading change). Raising the confidence
gate (an early attempt in this project's build) does NOT reliably filter these — confidence measures
frame0's OWN local texture richness, which says nothing about whether a valid frame1 correspondence
exists at all. This is the textbook occlusion limitation of any brightness-constancy optical-flow
method (see "Where this sits in the real world"); README "Limitations & honesty" names it directly
and the gates in `main.cu` are bounded from the MEASURED mean, not an aspirational target.

**Connected-component size filtering: measured effect, and why it is modest, not dramatic (an
honest finding, not a tuning miss).** `kMinComponentSizePx=10` (kernels.cuh) is derived from the
morphological opening operator's own math: a component made of exactly ONE erosion-surviving pixel
dilates back into a solid 3×3=9-pixel block — the theoretical MINIMUM the opening can produce — so
`>9` requires at least a second independently-surviving pixel. On the committed dynamic pair, the
post-morphology mask has 13 connected components; sweeping every possible size floor against the
KNOWN truth mask (an offline analysis, not something the pipeline can do at runtime — it does not
know where the object is) shows NO floor does dramatically better than the shipped one:

| Size floor | tp | fp | fn | precision | recall | IoU |
|-----------:|---:|---:|---:|----------:|-------:|----:|
| 1 (no filter, post-morphology only) | 123 | 302 | 178 | 0.289 | 0.409 | 0.204 |
| 10 (shipped `kMinComponentSizePx`) | 114 | 257 | 187 | 0.307 | 0.379 | 0.204 |
| 16 | 99 | 233 | 202 | 0.298 | 0.329 | 0.185 |
| 28 | 99 | 206 | 202 | 0.325 | 0.329 | 0.195 |
| 49 | 57 | 200 | 244 | 0.222 | 0.189 | 0.114 |
| best possible (grid search vs. truth, for reference only — NOT how the floor is chosen) | 114 | 257 | 187 | 0.329 | 0.379 | 0.213 |

The reason: visualizing the 13 components individually (comparing `moving_mask_postmorph.pgm` to
`truth_mask.pgm` pixel-by-pixel) shows the true object is FRAGMENTED by the morphological open into
several small pieces (9, 9, 15, 48, and part of a 59-pixel and a 198-pixel component), while the
SINGLE LARGEST false-positive contributor is not scattered speckle but ONE coherent 198-pixel blob
sitting immediately adjacent to the object — a disocclusion-boundary artifact (background revealed,
or occluded, by the moving box; the computed `\|P2-P1\|` there is large and genuinely non-zero, just
not caused by the tracked object). This blob is comparable in size to the object's OWN largest
fragment, so no size floor can keep one and drop the other — floors small enough to keep the real
15/9-pixel object fragments also keep same-size noise blobs elsewhere in the frame, and floors large
enough to drop the 198-pixel FP blob also drop genuine object fragments (see the `floor=49` row:
recall collapses). `kMinComponentSizePx=10` is kept anyway because it is (a) mathematically
DERIVED, not fit to this scene's answer, and (b) measurably better on PRECISION (0.289→0.307) with
IoU unchanged — a real, if modest, improvement; the honest limitation is documented rather than
hidden or force-fit to a bigger number (README "Limitations & honesty", Exercise 5 names the natural
next step: a forward-backward flow-consistency check that could catch the disocclusion region at
its SOURCE, in the lifting stage, rather than trying to discriminate it after the fact by shape.)

**Object motion: root-causing a near-opposite-direction result (a genuine bug, now fixed and
explained).** This project's build originally fit `T_obj` with a FREE (unconstrained rotation)
unweighted Horn solve restricted to the segmented mask — the natural first design, since it reuses
`build_rigid_from_covariance16` unchanged. On the committed dynamic pair it recovered an offset with
`cos(angle)` vs. the known `c_gt` of **-0.91** (magnitude 2.5× too large) — nearly the OPPOSITE
direction. Root-causing this (rather than assuming either "it's just mask noise" or "there's a
frame/axis bug") required isolating the two candidate causes:

1. **Is the mask's false-positive contamination alone enough to explain it?** Refitting the SAME
   free 6-DOF Horn solve restricted to the EXACT TRUTH mask (zero false positives by construction)
   still gave `cos(angle) ≈ -0.25` — poor, not just "a bit worse." Contamination is not the whole
   story.
2. **Is there a frame/axis-convention bug** (the same CLASS of bug this project's ego-motion stage
   root-caused once already — see "The math"'s M-conjugation story)? Feeding the fit EXACT
   ground-truth 2-D+3-D flow (`data/sample/truth_scene_flow.bin`, bypassing the pipeline's own
   estimated flow ENTIRELY) at truth-mask pixels, with the EXACT known `T_gt`, reproduces `c_gt`
   BIT-EXACTLY (`(-0.2996, 0.0000, 0.0157)` recovered vs. `(-0.2996, 0.0000, 0.0157)` truth, to
   printed precision). The geometry and convention are correct — ruling out a frame bug.

With (1) and (2) both narrowing the search, the actual cause is CONDITIONING: a free 6-DOF Horn fit
recovers `t = mu2 - R(q) mu1`, and on a SMALL (order 300 points), SPATIALLY NARROW (one box face, at
this scene's ~7-8 m range) point set, a modest rotation estimation error — inevitable given the
pipeline's own estimated (not exact) flow/depth feeding boundary-adjacent pixels — gets multiplied
by the ~7-8 m "lever arm" from the fit's centroid to the camera origin, producing a translation error
far larger than the mover's own ~0.3 m true signal. The background's ego-motion fit does not suffer
this because it has thousands of well-distributed points; the object's fit, restricted to a few
hundred points on one small surface, is inherently in the ill-conditioned regime.

**The fix (shipped): hold the rotation fixed.** Because the scene's object motion is translation-
only (documented already — "The math" above), the object's TRUE rotation is EXACTLY `R_robust`
(the already-accurate, well-conditioned background fit). Holding `R` fixed at `R_robust` and
robustly (IRLS+Tukey, reusing the exact machinery "Robustness" above describes) estimating only the
translation offset removes the ill-conditioned rotation solve entirely. Measured effect (dynamic
pair, final mask): `cos(angle)` improves from -0.91 to **≈0.95** (well past the 0.9 "well-aligned"
bar); magnitude ratio improves from 2.5× to **≈0.4-0.5×** (better, but still short of the 0.75-1.25×
a promotion to a GATE would need). The residual magnitude gap is consistent with a mixed-pixel/
partial-volume bias: bilinear depth sampling at boundary-adjacent pixels (a large fraction of a
~300-pixel object silhouette) blends object and background depth, which SHRINKS (never grows) the
recovered 3-D displacement at those pixels — a systematic, not random, effect, so it survives
robust averaging rather than cancelling out. `object_motion` stays `[info]`-only, honestly, with the
ACTUAL measured quality (not a blanket "solve ok") printed every run.

**Angle/rotation numerics.** Rotation error between two `Rigid3` estimates is measured via the trace
identity `cos(theta) = (trace(R_a^T R_b) - 1) / 2` (`main.cu`'s `rotation_angle_deg_between`),
clamped to `[-1,1]` before `acos` to guard against a microscopic floating-point excursion outside
the mathematically valid range for a true rotation-matrix product.

**Determinism.** No RNG at runtime — the ONLY randomness anywhere in this project is the offline,
fixed-seed (42) synthetic data generator. Every GPU kernel here uses a plain reduction tree (no
atomics, no non-deterministic ordering), so a rerun on the same GPU reproduces bit-identical
results; a different GPU architecture may differ in the last few ULPs of FMA-contraction behavior,
comfortably inside every VERIFY tolerance below.

## How we verify correctness

Two tiers, per CLAUDE.md's twin-independence ruling (`reference_cpu.cpp`'s header):

**Tier 1 — per-stage GPU-vs-CPU twins**, on the REAL loaded data (not synthetic toy inputs), each
independently implemented (kernels.cu vs. reference_cpu.cpp):

| Stage | Tolerance | Measured (reference run) |
|-------|-----------|---------------------------|
| Pyramidal LK flow | 0.05 px | 1.2e-2 px |
| 3-D lift (P1/P2/validity) | 1e-4 m, 0 validity mismatches | 9.5e-7 m, 0 mismatches |
| Weighted covariance reduce (non-uniform weights, IRLS round 1) | 1e-4 relative | 1.6e-9 relative |
| Residuals (IRLS round 1) | 1e-4 m | 9.8e-7 m |
| Threshold mask | bit-exact | bit-exact |
| Morphological open | bit-exact | bit-exact |
| Connected-component labels (GPU label-propagation vs. CPU union-find, canonicalized) | bit-exact | bit-exact (33 sweeps to convergence) |
| Component size filter | bit-exact | bit-exact |

**Tier 2 — independent, closed-form/known-answer gates**, exercising `build_rigid_from_covariance16`
(the one SHARED, un-twinned routine — see "The math" for why sharing it is permitted) against the
scene's KNOWN ground truth, never against a second copy of itself:

| Gate | Bound | Measured (reference run) |
|------|-------|---------------------------|
| `flow_2d` (mean EPE, confident+valid pixels) | < 3.20 px | 2.44 px (median 0.25 px — see Numerical considerations) |
| `scene_flow_3d` (mean 3-D EPE) | < 0.38 m | 0.26 m |
| `ego_motion` (robust rotation / translation error vs. known camera motion, WITH the mover present) | < 0.30° / < 8.0 mm | 0.017° / 0.9 mm |
| `ego_motion_robustness` (naive-vs-robust improvement) | ≥1.5× | naive: 0.28°/36 mm vs. robust: 0.017°/0.9 mm |
| `object_segmentation` (IoU vs. known object mask) | > 0.15 | 0.204 (precision 0.307, recall 0.379) |
| `static_negative_control` (segmented moving-pixel fraction, ego-motion-only pair) | < 5.0% | 1.6% |
| `noise_derivation` (raw pre-morphology false-positive rate on the negative control) | < 20.0% | 14.5% |

All bounds are MEASURED-then-margined (CLAUDE.md convention): none is an aspirational target chosen
before running real data. `object_motion` is reported `[info]`-only — cos(angle) ≈0.95 (well past a
0.9 bar) but magnitude ratio ≈0.4-0.5× (short of a 0.75-1.25× bar), see "Numerical considerations"
for the full root-cause story and why direction and magnitude are graded, and will be promoted,
independently.

## Where this sits in the real world

**RAFT-3D** (Teed & Deng, 2021) and its descendants are the current state of the art in dense scene
flow: a learned optical-flow backbone (RAFT) extended with a per-pixel `SE3` field and iterative
updates, trained end-to-end on large synthetic-plus-real datasets. It solves the SAME occlusion
problem this project's Limitations section names honestly — by learning features far more robust to
appearance change and texturelessness than Scharr-gradient brightness constancy — while keeping the
same conceptual decomposition (a locally-rigid motion field) this project teaches by hand.

**KITTI Scene Flow** is the standard real-world benchmark for this task, evaluated with the same
family of metrics this project's gates use (end-point error, outlier rate) on real automotive stereo
sequences with LiDAR-derived ground truth — a genuinely harder problem (real sensor noise,
uncontrolled lighting, real occlusion at every car/pedestrian boundary) than this project's clean
synthetic scene, which is exactly why a real system needs the learned robustness RAFT-3D-class
methods provide.

**Dynamic SLAM systems** (DynaSLAM, DS-SLAM, and their production descendants) use a moving-object
mask structurally identical to this project's output to REJECT dynamic-object feature
correspondences before they corrupt the map or the pose estimate — the downstream consumer named in
README "System context." Some (like this project) segment via geometric consistency after an
ego-motion fit; others use a learned semantic segmentation (person/car detectors) as a prior. A
production stack often combines both: semantics propose CANDIDATE dynamic regions, geometric
residual analysis (this project's technique) CONFIRMS which candidates are actually moving right now
versus parked/stationary instances of a dynamic-capable class.
