# 02.17 — LiDAR-camera projection/coloring fusion kernels: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Define every symbol,
> unit, and frame on first use.

## The problem — physics & engineering first

### Two complementary, physically incompatible measurements

A LiDAR and a camera measure genuinely different physical quantities, and fusion is only interesting
*because* they are complementary rather than redundant:

- A **LiDAR** is an active, time-of-flight ranging sensor ([`01.20`](../../01-perception-cameras-vision)'s
  domain covers the tactile/contact analog; the LiDAR-specific physics is
  [`01.02`](../../01-perception-cameras-vision)/[`01.18`](../../01-perception-cameras-vision/01.18-depth-completion)'s territory, cited): it fires a
  pulsed or FMCW-modulated beam and times its round trip, converting directly into a **range** along the
  beam. A LiDAR return is a genuine 3-D measurement with error dominated by timing jitter along the range
  axis (millimeter-scale) — but it carries **no appearance information at all**: a red box and a gray box
  at the same range and orientation return numerically identical points.
- A **camera** is a passive, bearing-only, *photon-counting* sensor: each pixel integrates the light
  arriving from one direction over the exposure window, encoding reflectance/appearance with no direct
  depth at all. Depth is only recoverable indirectly (stereo, motion, or — as here — a *known* second
  ranging sensor).

Fusion buys exactly what each sensor lacks: geometry from time-of-flight, appearance from photons. The
catch is that "put the appearance where the geometry is" only works cleanly when both sensors see the
same point from the same place — and they never do, because two physical instruments cannot occupy the
same volume of space. That single fact — a nonzero **baseline** between the LiDAR and camera origins —
is the entire reason this project's hardest problem (occlusion) exists at all.

### The occlusion geometry this project's scene is built to expose (derived)

Let a camera sit at height `h_cam` and a LiDAR at height `h_lidar` (both measured from a common ground
plane, meters, `z` up), separated by a vertical **baseline** `b = h_lidar - h_cam` (this project's rig:
`b = 1.80 - 1.50 = 0.30` m — IDENTICAL to 01.18/02.02's own rig, cited). An opaque occluder's top edge
sits at height `h_top` and range `r1` from the rig; a background surface sits at range `r2 > r1`. For a
sensor at height `h_s` to see a background point at height `h_bg` *over* the occluder's top edge, similar
triangles along the line of sight give the height of that line of sight AT the occluder's range:

```
height_at_occluder(h_s, h_bg) = h_s + (r1/r2)·(h_bg - h_s)
```

The sensor clears the occluder exactly when `height_at_occluder >= h_top`, which rearranges to a minimum
background height:

```
h_bg >= h_s + (r2/r1)·(h_top - h_s)              (the clearance condition, per sensor)
```

Evaluate this for the camera (`h_s = 1.50`, `h_top = 1.60`, `r2/r1 = 12/4 = 3`) and the LiDAR
(`h_s = 1.80`, same `h_top`, `r2`, `r1`):

```
camera clears when h_bg >= 1.50 + 3·(1.60-1.50) = 1.80 m
LiDAR  clears when h_bg >= 1.80 + 3·(1.60-1.80) = 1.20 m
```

**Background points with `h_bg` in `(1.20, 1.80)` m are therefore genuinely visible to the LiDAR but
hidden from the camera** — this project's occlusion cohort, `scripts/make_synthetic.py`'s designed
geometry (and confirmed there per-point by an independent second ray cast, not just this formula). The
cohort's HEIGHT WIDTH has a clean closed form (subtract the two thresholds):

```
Δz_cohort = h_bg,camera_threshold - h_bg,lidar_threshold = b · (r2/r1 - 1)
```

With this rig's numbers: `Δz_cohort = 0.30 · (3 - 1) = 0.60` m — matching the derivation above exactly.
**This formula is why the z-buffer visibility pass is not optional, and not a corner case**: for *any*
nonzero baseline `b` and *any* foreground/background depth discontinuity with `r2 > r1`, the cohort width
grows linearly with `b` and with the range ratio `r2/r1`. A taller sensor mast (bigger `b`, common on
roof-rack AV rigs) or a nearer occluder relative to the background (bigger `r2/r1`, common in cluttered
urban scenes) makes the cohort *larger*, not smaller — naive point coloring gets *worse*, not better, as
real-world rigs scale up. A same-origin sensor (`b = 0`) would have zero cohort by this formula — occlusion-
by-parallax is a direct, quantified consequence of the baseline, not an approximation artifact.

### Engineering constraints that shape the algorithm

- **LiDAR angular sparsity vs. pixel-grid density.** This project's committed scan (31 elevations ×
  120 azimuths) spaces adjacent returns roughly 1–3 px apart on the 160×120 image (measured:
  `data/README.md`). An occlusion check that only trusts the EXACT pixel a point lands on therefore often
  finds no competing z-buffer evidence even where the occluder plainly covers that pixel in the dense
  camera image — "The GPU mapping" below derives why this project's occlusion check searches a small
  pixel window instead.
- **Range noise.** This project's synthetic LiDAR carries σ = 2 cm range noise (illustrative MEMS/ToF-
  class unit, `data/README.md`) — an occlusion check that demanded EXACT depth equality would reject
  every truly-visible point too.
- **Camera-LiDAR synchronization and platform motion.** A real fusion pipeline additionally needs the
  camera exposure and the LiDAR sweep to be time-aligned (hardware FSYNC/PTP — `PRACTICE.md` §1) and, on
  a moving platform, per-point motion deskew before projection
  ([`02.08`](../02.08-per-point-motion-deskew-with-pose-interpolation), named, not implemented here —
  README "Limitations"). This project's single static synthetic frame sidesteps both; a real rig cannot.

## The math

**Notation.** `P_lidar = (x,y,z)` (m): a LiDAR-frame point (LidarPointF, `src/kernels.cuh`).
`T_camera_lidar = (R, t)`: the extrinsic, `R` row-major 3×3, `P_camera = R·P_lidar + t` — the IDENTICAL
`Rigid3` shape and numeric values 01.17 solves for and 01.18/02.02 already consume (cited, not
re-derived; this project's own contribution is the OCCLUSION geometry above and the SENSITIVITY analysis
below). Depth always means `Pcam.z` (never Euclidean range — the pinhole/z-buffer convention every
project in this family shares).

### Pinhole projection (shared with 01.17/01.18/02.02)

```
u = fx·Xc/Zc + cx        v = fy·Yc/Zc + cy         (Pcam = (Xc,Yc,Zc), Zc > 0)
```

`fx=154, fy=152, cx=80, cy=60` (px) — this project's teaching camera, numerically identical to
01.17/01.18/02.02's. Points with `Zc <= 0` (behind the camera) or `Zc > kMaxDepthM = 20` m (the LiDAR's
own maximum-range cutoff — 01.18's THEORY.md derives the eye-safety link budget, cited) are dropped.

### The z-buffer's encoded-atomicMin trick (reused from 01.18, cited)

CUDA has no native `atomicMin` for `float`. Reinterpreting a POSITIVE, finite float's IEEE-754 bit
pattern as `uint32_t` preserves numeric ordering (the exponent occupies the high bits and dominates the
comparison), so an integer `atomicMin` on the reinterpreted bits keeps the smallest depth with no
transformation needed — every depth this project encodes is a LiDAR return in front of the camera
(`Zc > 0` checked first), so the general negative-float case (01.18 documents it, not needed here) never
runs.

### Bilinear sampling

For a continuous projected coordinate `(u,v)` and a normalized `[0,1]` planar RGB image, clamp
`(u,v)` to `[0,W-1]×[0,H-1]` (clamp-to-edge), then blend the four surrounding pixel taps
`(x0,y0),(x0+1,y0),(x0,y0+1),(x0+1,y0+1)` with weights `(1-tx)(1-ty), tx(1-ty), (1-tx)ty, tx·ty` where
`tx = u - floor(u)`, `ty = v - floor(v)`. Nearest-neighbor would silently discard exactly the sub-pixel
information the pinhole projection computed — and, as "Numerical considerations" shows, bilinear's OWN
blending across a true color edge is this project's clearest non-occlusion failure mode.

### The occlusion depth-consistency check, windowed

A point at rounded pixel `(px,py)` (round-half-up, `floor(x+0.5)` — the SAME rule the z-buffer uses to
choose ITS pixel, so the two never disagree about which pixel a point "belongs to") is accepted as
*visible* iff:

```
exists (nx,ny) in [px-R, px+R] x [py-R, py+R]  such that
    |zc(point) - decode(d_encoded[ny,nx])| <= band_m
```

with `R = kOcclusionWindowRadiusPx = 2` (a 5×5 window) and `band_m = kOcclusionBandM = 0.30` m. "The GPU
mapping" and "Numerical considerations" below derive why BOTH the window and the band are necessary, and
"How we verify correctness" reports the measured effect of widening `R` from 0 (single pixel) to 2.

### Calibration-error perturbations and their predicted pixel effect (01.17, re-derived here)

01.17's THEORY.md splits an extrinsic error into a rotation part `δθ` (rad) and a translation part `δt`
(m) and derives their pixel effects at a point of range `R` (m):

```
rotation:    Δpixel ≈ f·δθ                (range-INDEPENDENT — every ray tilts by the same angle)
translation: Δpixel ≈ f·δt / R            (SHRINKS with range — a fixed offset subtends a smaller angle further away)
```

This project's sweep applies each error TYPE in isolation so the two formulas can be checked separately
(`src/main.cu`'s `perturb_rotation`/`perturb_translation`):

- **Rotation sweep**: `R' = Ry(θ)·R_base` — an EXTRA small rotation about the CAMERA's own Y ("down")
  axis composed on the LEFT of the existing extrinsic (`θ ∈ {0.2°, 0.5°, 1.0°}`), `t` untouched. `Ry`
  mixes the camera's X/Z axes, and X is exactly the axis the pinhole formula divides into `u` — so this
  perturbation produces a mostly-HORIZONTAL pixel shift, predicted `Δu ≈ fx·θ` for a point near the
  optical axis (off-axis points pick up a smaller second-order correction, absorbed into this project's
  measured headroom below).
- **Translation sweep**: `t' = t_base + δ·(1,0,0)` (the camera's own "right" axis, `δ ∈ {1,2,5}` cm), `R`
  untouched. Predicted `Δpixel ≈ fx·δ/R̄`, `R̄` the mean camera-frame depth of the considered points.

### Color-boundary crossing (the sensitivity curve's headline number)

For each perturbation level, a point's sampled color is compared against its OWN zero-perturbation
baseline sample (not against ground truth — this measures how much the CALIBRATION ERROR moves the
sampled color, independent of whether the baseline sample was itself accurate). A "boundary crossing" is
`max(|Δr|,|Δg|,|Δb|) > kColorBoundaryThresh = 0.25` (normalized `[0,1]`, the same max-channel measure
01.18's conductance and this project's `color_dist` both use) — chosen well below the scene's smallest
true-surface color distance (README "Data"'s palette) so a flip genuinely means "drifted onto a
different surface," not sensor/rounding noise.

## The algorithm

Four kernels (`src/kernels.cu`), each O(N) in the point count `N` (a few thousand) and embarrassingly
parallel across points; `src/main.cu` orchestrates them into the two directions plus the sweep:

1. **`project_zbuffer_kernel`** — for each point, project (drop if invalid), `atomicMin` the encoded
   depth into its pixel. `O(N)` serial, `O(N/P)` parallel plus rare atomic contention near silhouette
   edges (where several beams round to the same pixel). This pass alone is Direction B's product.
2. **`project_points_kernel`** — the shared geometric core: project every point (valid or not) to
   continuous `(u,v)`, camera depth `zc`, and an in-frustum flag. `O(N)`, no shared state.
3. **`sample_bilinear_kernel`** — four-tap bilinear gather at `(u,v)` from the guidance image. `O(N)`
   (each thread does `O(1)` work — 4 taps × 3 channels).
4. **`check_occlusion_kernel`** — for each point, scan its `(2R+1)²` pixel window of kernel 1's z-buffer
   for the nearest evidence and compare within the band. `O(N · (2R+1)²)` — with `R=2` a 25-tap scan per
   point, still trivial at this point count (measured: a fraction of a millisecond, `[time]` lines).

**The sensitivity sweep** (evaluation-only, `main.cu`) re-runs kernels 2+3 six times (three rotation
levels, three translation levels) at a perturbed `T_camera_lidar`, comparing each run's `(u,v,color)`
against the SAME baseline run's outputs — `O(6·N)` total, negligible next to the ray-casting that built
the scene.

## The GPU mapping

**Scatter then gather — the two-pass fusion taxonomy.** Kernel 1 is a SCATTER: one thread per INPUT
point, writing to a data-dependent, possibly-colliding OUTPUT pixel — exactly the situation atomics
exist for (01.18's kernels.cu names this same pattern for its own z-buffer stage, cited). Kernels 2–4 are
GATHERS/MAPS: one thread per point, reading from fixed (image-shaped) inputs, writing its own private
output slot — no collisions, no atomics. Splitting the pipeline this way — SCATTER once to build shared
evidence (the z-buffer), then GATHER many times against that evidence — is the general pattern any
fusion kernel that needs "does something else occupy MY output location" has to use; 01.18's own
projection+z-buffer stage collapses scatter-then-immediate-use into one kernel because it has no
DOWNSTREAM gather that needs the z-buffer as a *read-only* input from OTHER threads' points — this
project does (kernel 4 reads kernel 1's output from a DIFFERENT set of points' neighborhoods), which is
exactly why it stays a separate pass here.

**Per-point independence, four times over.** Every kernel maps one thread to one LiDAR point (kernel 1)
or one point's already-computed data (kernels 2–4) — the same "one thread owns one independent unit of
work" idiom 08.01/09.01/33.01 establish, applied here to fusion instead of rollouts or FK. No shared
memory anywhere: each kernel's per-thread working set (a handful of floats, kernel 4's 25-tap window) is
small enough that registers suffice, and — unlike 01.11's bilateral filter, which tiles overlapping reads
into shared memory to amortize a much wider stencil — kernel 4's window reads are individually cheap
enough (25 global loads of 4 bytes each) that a shared-memory tile would spend more effort managing the
tile than it would save (THEORY.md is honest that a production-resolution version, thousands of points
per pixel neighborhood, would revisit this trade).

**Memory hierarchy.** Every kernel uses GLOBAL memory only, `__restrict__` on every read-only pointer.
At this project's scale (a few thousand points, a 160×120×3 image) the ENTIRE working set is a few
hundred kilobytes — well under any GPU's L2 cache, so occupancy and bandwidth are not the bottleneck; the
kernels are latency-bound on a handful of dependent loads per thread, which is why the whole four-kernel
pipeline measures well under a millisecond (`[time]` lines).

## Numerical considerations

- **Encoded-atomicMin lineage.** Covered in "The math"; the risk case (a NaN or non-positive depth
  reaching the encoder) cannot occur here — `zc <= 0` is filtered before the encode call, and the
  ray-cast synthetic data never produces a NaN/Inf depth by construction (same argument 01.18 makes for
  its own z-buffer, cited).
- **Bilinear sampling at a true color boundary — the edge-bleeding measurement.** A LiDAR point projected
  within roughly one pixel of a real object silhouette samples a WEIGHTED BLEND of two different
  surfaces' colors — physically honest (the pinhole model has no way to know a discontinuity exists
  there) but measurably worse than interior sampling. This project's own committed run: boundary-pixel
  points (classified by a strong RGB gradient at their rounded pixel, the same max-channel measure
  `kColorBoundaryThresh` uses) show a naive-coloring wrong-color rate of 41.5% (n=193) vs. 0.14% (n=2767)
  for interior points — a **287× ratio**, printed on an `[info]` line every run, never gated pass/fail
  (README "Expected output" explains why: this is a sub-pixel/parallax reality to TEACH, not a bug to
  fix — see 01.18's own "edges coincide" prior discussion for the companion RGB/depth-edge story).
- **The occlusion window's over/under-filtering trade-off, measured.** `R=0` (exact pixel only, an
  earlier version of this project) let 56.5% of the ground-truth-occluded cohort through wrongly colored
  — barely better than the 89.1% naive baseline — because the sparse occluder scan simply had no return
  on most cohort points' exact pixels (the angular-sparsity argument in "The problem" above, made
  concrete). Widening to `R=2` (a 5×5 window, this project's shipped value) drops that to 0.7% wrong —
  but at a real cost: `coloring_accuracy` (measured on ground-truth-VISIBLE points) drops from a
  hypothetical near-100% at `R=0` to 76.9% at `R=2`, because the wider search now also finds unrelated,
  closer evidence near real depth EDGES and over-filters some genuinely visible points there too.
  README Exercise 2 asks the learner to sweep `R` and watch this trade-off directly — there is no
  "correct" window size independent of the scan's own density, only a measured trade curve.
- **Sub-pixel rounding discipline.** The z-buffer's pixel index (`floor(u+0.5)`) and the occlusion
  check's own re-derivation of that SAME index must agree exactly, or a point could be compared against
  the WRONG pixel's z-buffer evidence — `check_occlusion_kernel`'s doc-comment states this as the
  contract kernel 1 and kernel 4 share.
- **Determinism.** Every kernel here is a pure per-point map/scatter with no floating-point reduction
  order to worry about (unlike 08.01's softmin sum or 02.06's ICP normal-equation accumulation) — the
  GPU-vs-CPU VERIFY tolerances below are tight (1e-4–1e-5) precisely because there is no accumulated
  iteration to loosen them, the SAME reasoning 01.18's single-shot stages (projection, conductance) use.

## How we verify correctness

**Tier 1 — GPU-vs-CPU twins**, per the repo's independence ruling (`reference_cpu.cpp`'s file header):
every kernel's algorithmic core (the rigid transform + pinhole formula, the z-buffer compare, the
bilinear blend, the windowed occlusion search) is written TWICE, independently. Measured on the
committed sample (RTX 2080 SUPER, Release|x64; also verified Debug|x64):

| Kernel | Tolerance | Measured worst |
|---|---|---|
| `project+z-buffer` | 1e-4 m, zero empty-pixel mismatches | 0.0 m exact, 0 mismatches |
| `project-points` | 1e-3 px (u,v), 1e-4 m (zc), zero in-frustum mismatches | 1.5e-5 px, 0.0 m, 0 mismatches |
| `bilinear-sample` | 1e-5 | 6.0e-8 |
| `occlusion-check` | ≤1% visibility-flag mismatches | 0.0% (0/3368) |

The occlusion-check tolerance is not bit-exact by design: the GPU path decodes ITS OWN encoded z-buffer;
the CPU path reads an independently-computed plain-float z-buffer. The two z-buffers already agree
within 1e-4 m (tier above); a point sitting almost exactly on the `±band_m` boundary can in principle
still flip which side of the compare it lands on when the two buffers' sub-1e-4 rounding differs — the
same "chained-comparison" story 01.17's `TRAJECTORY_TWIN` gate tells for its own multi-stage pipeline
(cited). Zero is the expectation and the measured result; a handful is tolerated and would be reported,
not hidden.

**Tier 2 — independent gates against ground truth** (`scripts/make_synthetic.py`'s true_r/g/b/visible
columns, which never touch either verified code path):

| Gate | What it checks | Measured | Threshold | Headroom |
|---|---|---|---|---|
| `frustum_accounting` | bookkeeping: in+out==total, colored+filtered==in-frustum | exact | exact | — |
| `coloring_accuracy` | ground-truth-visible points colored within tol 0.12 of true color | 76.9% (2483/3230) | ≥ 70% | ~7 pp |
| `occlusion_correctness` (WITHOUT check) | naive coloring's wrong-color rate on the occluded cohort | 89.1% (n=138) | ≥ 80% | ~9 pp |
| `occlusion_correctness` (WITH check) | checked coloring's wrong-color rate, SAME cohort | 0.7% | ≤ 5% | ~7× |
| `depth_image_fidelity` | painted depth vs. an independently re-derived per-pixel minimum | 0.0 m exact | ≤ 1e-4 m | — |
| `sensitivity_curve` (analytic consistency) | smallest-level measured/predicted pixel-displacement ratio | rotation 1.09×, translation 1.26× | within 4× (either direction) | ~3.5×/~3.2× |
| `sensitivity_curve` (monotonicity) | flip fraction non-decreasing with \|perturbation\|, both sweeps | rotation 0.26%→2.73%→5.22%; translation 0.00%→0.00%→1.53% | non-decreasing | — |

`depth_image_fidelity`'s cross-check deliberately does NOT reuse `project_zbuffer_cpu`: `main.cu` carries
a THIRD, independent, evaluation-only re-implementation of the projection formula
(`project_to_pixel_eval`) purely so this gate does not silently depend on either verified code path being
correct — the same "a gate that does not route through either twin" discipline 01.17's `jacobian_check`
and zero-noise gates use (cited).

**Edge cases exercised:** points behind the camera (`zc<=0`) and beyond `kMaxDepthM`; points outside the
image after rounding; an entirely empty occlusion-check window (conservatively NOT visible, never a
silent "assume visible"); the `dist==0`/self-sample early return in bilinear sampling is not applicable
here (bilinear always has 4 defined taps) but the border-clamp path IS exercised (points near the image
edge, `kernels.cu`'s `bilinear_sample_device` clamp branch).

## Where this sits in the real world

- **PointPainting** (Vora et al., 2020) and its descendants paint LiDAR points with per-pixel SEMANTIC
  labels from a camera network (not raw RGB) before feeding a 3-D detector — this project's Direction A
  is the geometric plumbing (projection, occlusion-aware association) any such scheme depends on;
  production systems additionally handle lens distortion, multi-camera coverage, and — critically — run
  the occlusion reasoning at the SEMANTIC-network's resolution and confidence, not a raw z-buffer window.
- **ROS 2's `image_geometry`/`depth_image_proc`** ship the hardened, distortion-aware version of this
  project's `project_to_pixel_eval`/pinhole formula as reusable library calls; `depth_image_proc`'s
  point-cloud-to-depth-image and depth-image-to-point-cloud nodelets are Direction B's production
  analogs.
- **Colorized mapping products** (colored TSDF/voxel maps, nvblox-style) consume point-coloring output
  shaped exactly like this project's `cloud_topview.ppm`/`cloud_sideview.ppm` artifacts, at scale and
  over many accumulated frames rather than one.
- **What the full version adds** beyond this teaching core: lens distortion correction, multi-camera
  fusion (choosing which camera's pixel to sample when several see the same point), learned occlusion
  reasoning (a network trained to predict visibility rather than a fixed-radius window), hardware
  time-synchronization (`PRACTICE.md` §1) instead of an assumed single static frame, and — where the
  fused output feeds a safety-relevant consumer — a validated confidence/uncertainty estimate per painted
  point, not just a binary visible/filtered flag.
