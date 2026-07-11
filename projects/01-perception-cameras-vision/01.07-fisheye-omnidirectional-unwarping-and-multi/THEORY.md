# 01.07 — Fisheye/omnidirectional unwarping and multi-camera surround-view stitching: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Why a lens cannot simply "see wider"

A conventional (rectilinear) camera lens forms an image by projecting a 3-D ray at angle `theta` from
the optical axis onto a flat sensor at radius `r = f*tan(theta)`, where `f` is the focal length in
pixels. This is not a design choice one can tune away: it is the geometry of a flat image plane and a
single, well-behaved (low-distortion) optical center — the same trigonometry that makes a shadow on a
wall grow without bound as the light source approaches the wall's own plane. As `theta -> 90 degrees`,
`tan(theta) -> infinity`: a rectilinear sensor would need infinite area to capture a ray arriving
perpendicular to the optical axis, and any FOV that even approaches 180 degrees suffers extreme,
unusable magnification near the edge (a straight highway lane marking near the edge of a 170-degree
rectilinear photo would stretch across a huge fraction of the image). A pinhole/rectilinear lens
physically CANNOT see a full hemisphere, let alone the 185-degree-class FOV automotive surround-view
and security cameras routinely ship.

A fisheye lens solves this with real optics, not a projection trick: a strongly negative (concave)
front element intercepts marginal rays at large `theta` and bends them MUCH more aggressively than a
simple lens would, trading angular linearity in the pinhole sense for bounded, well-behaved image-
space compression. The lens designer chooses (via the exact curvature and spacing of several
elements) which `r(theta)` relationship the finished lens approximates; the common families, all
converging to the SAME `r ~ f*theta` behavior for SMALL `theta` (where every lens looks pinhole-like)
but diverging for large `theta`:

| Family | `r(theta)` | Behavior as `theta -> 90 deg` and beyond |
|---|---|---|
| Rectilinear (pinhole) | `f*tan(theta)` | Diverges to infinity at 90 deg — cannot represent theta >= 90 deg at all |
| Stereographic | `2f*tan(theta/2)` | Finite, but still grows without bound as theta -> 180 deg; conformal (preserves local angles) |
| **Equidistant (this project)** | `f*theta` | Exactly LINEAR in theta, finite and well-behaved for theta up to 180 deg |
| Equisolid angle | `2f*sin(theta/2)` | Finite, saturates smoothly; equal-AREA (a fixed solid angle maps to a fixed image area — the common "35mm-camera fisheye" spec) |
| Orthographic | `f*sin(theta)` | Saturates at `r=f` when `theta=90 deg` — CANNOT represent theta > 90 deg at all |

This project's rig uses the **equidistant** model — the textbook "ideal fisheye" and a real,
purchasable lens spec (many security/panoramic lenses are sold as equidistant or close to it) — for
two reasons named again in `kernels.cuh`'s file header: it is the simplest family to teach, and its
`theta = r/f` inverse is exact and closed-form (no iterative undistort, in direct contrast to 01.01's
Brown-Conrady pinhole-distortion model). Real fisheye lenses deviate from any pure family by a
few percent; **the Kannala-Brandt polynomial** `r = f*(theta + k1*theta^3 + k2*theta^5 + k3*theta^7 +
k4*theta^9)` is the production generalization every real calibration toolchain (OpenCV `fisheye`,
Kalibr) fits per-lens — this project's `k_i` are implicitly all zero, a genuine simplification named
here and in `README.md`'s "Limitations & honesty".

### Vignetting is physics, not a rendering defect

No lens illuminates its full theoretical FOV with uniform brightness. A fisheye lens's front element
has a finite physical aperture; rays arriving at very large `theta` graze the lens barrel and internal
baffling, are partially absorbed, and in a real sensor the corners of a rectangular image sensor
sitting behind a circular image circle receive NO light at all past the design radius. This project
models that honestly: the equidistant formula is mathematically well-defined for any `theta` up to 180
degrees, but `kFishValidHalfFovRad` (92.5 degrees, half of the lens's 185-degree-class spec) marks
where the real lens's illumination stops — `scripts/make_synthetic.py` renders everything past that
radius pure black (the vignette), and every unwarp/BEV computation in this project treats it as
"outside the illuminated circle," never sampling scene content there.

### The flat-ground assumption — the geometry that makes BEV possible, and where it breaks

A single 2-D camera image loses depth: every pixel tells you a RAY, not a POINT. Reconstructing a
top-down (bird's-eye) view from camera images therefore requires knowing, for every ray, how far along
it the visible surface actually is — information a monocular camera does not provide on its own. The
industry-standard trick (and this project's): **assume every ray's first solid surface is the flat
ground plane, Z=0, in the vehicle's own frame.** Under that assumption, a ray's ground intersection is
a simple, closed-form computation (intersect a line with a plane — this project's `bev_pixel_to_ground`
+ `rig_camera_to_bev_sample`, run in reverse from a ground point to a camera pixel), no depth sensor
needed. For a genuinely flat-ground point — a lane marking, a patch of bare asphalt — the assumption is
exactly true, and the reconstruction is exact up to lens/calibration/blend error (this project's
`bev_ground_truth` gate measures exactly this residual).

For a TALL object — a parked car's bumper, a curb, a pedestrian, the 3 objects this project's
synthetic scene places on purpose — the assumption is **false**: the camera ray hits the object's
surface well before it would have reached the ground. The BEV compositor, having no way to know this,
paints the object's color onto whatever ground point that ray direction geometrically implies — a
point that is typically much FARTHER from the camera than the object actually is (a nearby object
occludes a large swath of "implied ground" behind it, along the ray). The result is a classic **radial
smear**: the object's silhouette appears to bleed outward from the mounting camera's position across
the ground plane, often well past where the object physically stands. This project's
`flat_ground_assumption` gate measures this deliberately, as a negative control (README "The algorithm
in brief"): the same ground-truth comparison used to prove flat-ground reconstruction is ACCURATE on
real ground also proves it is INACCURATE — by a large, consistent margin — near tall objects. This is
not a bug to fix; it is the central, honest limitation every real BEV product ships with (PRACTICE.md
§4), and the reason production systems either restrict BEV to true ground content, fuse in a real depth
source (stereo, LiDAR, radar) for anything above the ground plane, or accept the ghosting as a known,
documented UX limitation.

### Engineering constraints on the real hardware

A real fisheye module for this use case faces: **mechanical tolerance** (a fisheye lens's fixed,
extreme distortion means a mounting-angle error of even 1-2 degrees shifts the effective FOV boundary
by a visible amount — much more sensitive than a narrow-FOV lens); **thermal drift** (the housing and
lens barrel expand with temperature, shifting the effective focal length by a small but, at automotive
temperature ranges, non-negligible amount — production systems recalibrate or compensate); **vibration**
(a vehicle-mounted camera sees continuous low-amplitude vibration; a fisheye's high angular sensitivity
near the FOV edge means even sub-pixel vibration blur is proportionally worse there than at the image
center); and **contamination** (mud, water, ice on the lens is a real, common failure mode for a
downward/outward-facing surround-view camera — PRACTICE.md §1 covers cleaning/self-heating mitigations).

## The math

**Frames.** Vehicle body frame: right-handed, x-forward, y-left, z-up, origin at ground level under the
vehicle's geometric center (CLAUDE.md §12). Camera-optical frame (the stated exception, same as 01.01):
z-forward (down the optical axis), x-right, y-down.

**The equidistant fisheye model.** For a camera-frame ray direction `(X, Y, Z)` (any positive scale):

```
theta = atan2( hypot(X, Y), Z )              in [0, pi]   -- angle from optical axis
phi   = atan2( Y, X )                        in (-pi, pi] -- azimuth around optical axis
r     = f * theta                                          -- THE equidistant model
u     = cx + r * cos(phi)
v     = cy + r * sin(phi)
```

The inverse (pixel `(u,v)` -> unit ray direction) is exact and closed-form:

```
du, dv = u - cx, v - cy
r      = hypot(du, dv)
theta  = r / f          -- exact, no search
phi    = atan2(dv, du)
(X, Y, Z) = ( sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) )
```

**Rig extrinsics.** Each of the 4 rig cameras has a pose `T_vehicle_cam` (CLAUDE.md §12 convention:
parent = vehicle, child = camera): a mount translation `t` (meters, vehicle frame) and a rotation
`R_vehicle_cam` whose COLUMNS are the camera's own `(x_cam, y_cam, z_cam)` axes expressed in vehicle
components. `kernels.cuh` stores the equivalent `R_cam_vehicle = R_vehicle_cam^T` as ROWS (a pure
bookkeeping choice — see that file's `CameraExtrinsic` comment — so the hot path needs no transpose).
Every camera shares one derivation: a NOMINAL (untilted, facing directly +x/-x/+y/-y) orthonormal basis,
then a single rotation by `tilt = 45 deg` about the camera's own right (`x_cam`) axis:

```
x_cam = x0                                  (rotation axis, unchanged)
y_cam = cos(tilt)*y0 - sin(tilt)*z0
z_cam = sin(tilt)*y0 + cos(tilt)*z0
```

A vehicle-frame ground point `P = (X, Y, 0)` maps into camera coordinates by
`P_cam = R_cam_vehicle * (P - mount)`; `P_cam`'s own z-component is that ray's forward depth in the
camera's frame, and `(P_cam.x, P_cam.y, P_cam.z)` feeds the equidistant forward-projection formula
above to find which fisheye pixel shows that ground point.

**Homography as a special case.** A reader who has seen automotive BEV done with OpenCV's
`warpPerspective` may wonder why this project does not use a single 3x3 homography matrix. For a
PINHOLE camera, the flat-ground assumption (Z=0) collapses the general 3-D projection into exactly a
3x3 linear map in homogeneous coordinates — a homography IS "assume a flat ground plane, then a
pinhole camera's projection becomes linear." This project's cameras are fisheye (the equidistant model
above is not a linear projective map of `(X,Y,Z)`), so no single 3x3 matrix captures it: every BEV pixel
in this project is walked explicitly through the 3-D chain (ground point -> vehicle-to-camera rigid
transform -> fisheye projection) instead — the "inverse mapping done through 3-D, not a 2-D homography"
distinction the catalog bullet and README call out.

## The algorithm

**Half 1 (per-camera unwarp), for each output surface:**
1. For every OUTPUT pixel `(xo, yo)`, compute its camera-frame ray via that surface's own unprojection
   (`pinhole_unproject_rect` for the rectilinear sub-FOV; `cyl_unproject`'s azimuth/elevation sweep for
   the cylindrical panorama).
2. `fisheye_project()` that ray into the fisheye source image's pixel coordinates — write one
   `RemapSample{u,v}` into a LUT (this step is purely geometric: it depends only on the output pixel and
   the fixed camera model, never on image content, so it is computed ONCE, not per frame in a real
   video pipeline).
3. Bilinear-gather: for every output pixel, read its LUT entry and bilinear-sample the fisheye image.

Serial cost: `O(W_out * H_out)` for both the LUT build and the gather, each pixel `O(1)` work
(constant-time trig for the LUT, 4 texel reads + a lerp for the gather) — fully parallel, no
data-dependent branching beyond a boundary clamp.

**Half 2 (BEV compositor), for every BEV output pixel:**
1. `bev_pixel_to_ground()` — which ground-plane point `(X,Y,0)` this pixel represents.
2. For each of the 4 rig cameras: transform the ground point into that camera's frame, check it is in
   front of the camera and inside the design FOV, compute a linear feather weight, and (if visible)
   bilinear-sample that camera's fisheye image.
3. Accumulate a weighted sum over however many cameras (0 to 4, though realistically 0-2 given this
   rig's geometry) are visible; write the normalized blend and a coverage bitmask.

Serial cost: `O(W_bev * H_bev * 4)` — a small, FIXED per-pixel constant (never data-dependent beyond
early-exit skips), the exact shape that makes "4-camera loop in one thread" the right GPU mapping (see
"The GPU mapping" below).

## The GPU mapping

Every kernel in this project maps one GPU THREAD to one OUTPUT PIXEL, over a 2-D thread grid matching
the output image's own shape (`blockIdx.{x,y}*blockDim.{x,y}+threadIdx.{x,y}` — no flattening
arithmetic needed, unlike a 1-D grid-stride map over a linearized array). This is the natural
generalization of 01.01's 1-D map to a 2-D image, and it recurs across all 4 kernels in `kernels.cu`
(`build_rect_lut_kernel`, `build_cyl_lut_kernel`, `remap_bilinear_kernel`, `bev_compose_kernel`) —
learning one teaches all four.

**LUT-vs-recompute trade** (the same lesson 01.01's remap-LUT teaches, generalized): the rectilinear and
cylindrical LUTs are purely geometric (independent of pixel content), so this project computes them
ONCE, in their own kernels, and reuses them for every subsequent bilinear-gather call — exactly 01.01's
`launch_build_remap_lut` precedent. A real video pipeline processing many frames per second would
compute the LUT once at startup (or once per calibration) and reuse it for every frame thereafter,
making the LUT-build cost amortize toward zero; this project's demo computes it fresh each run purely
for pedagogical self-containment.

**Why one thread per BEV pixel with an IN-THREAD 4-camera loop, not one thread per (pixel, camera) pair
with a cross-thread reduction** (`kernels.cu`'s `bev_compose_kernel` header states this in full; summary
here): the blend is a tiny, FIXED-size (exactly 4-term) weighted sum. Splitting it across 4 threads
would need either a second kernel launch to reduce the 4 partial results (an extra global-memory round
trip to move a 4-element sum) or atomic writes into the shared output pixel (a genuine race this
design has no need to invite) — both strictly worse than simply accumulating 4 terms in one thread's
REGISTERS, the fastest memory this GPU has, with zero synchronization. `kBevW*kBevH = 102,400` threads
is already comfortably more than an RTX 2080 SUPER's 46 SMs need for full occupancy; splitting the
camera loop across 4x more threads would not measurably improve occupancy, only add coordination cost.

**Memory hierarchy.** Every kernel here is a light-compute, moderate-memory-traffic gather: global
memory holds the source image(s) and the LUT; each thread reads a handful of texels (4 for one
bilinear sample, up to 16 for the BEV kernel's worst case of 4 cameras x 4 texels each) and writes one
output pixel — no shared memory, no reuse between threads (each thread's source samples are
data-dependent on ITS OWN output pixel, unlike a stencil where neighboring threads read overlapping
input, so shared-memory tiling would not help here), no atomics. Registers hold the 4-camera
accumulator in `bev_compose_kernel` and nothing else of note; occupancy is bandwidth-bound, not
register- or shared-memory-limited.

**No CUDA library calls** (cuBLAS/cuFFT/Thrust/CUB) — every operation here is either constant-time
per-thread trigonometry or a small, hand-written bilinear gather, exactly the granularity this repo's
dependency policy (CLAUDE.md §5) reserves for hand-rolled teaching code rather than a library call.

## Numerical considerations

**The equidistant model's benign singularities.** `phi = atan2(Y, X)` (forward) and
`phi = atan2(dv, du)` (inverse) are mathematically undefined only when both arguments are exactly zero
(the dead-center ray, `theta = 0` / `r = 0`). In both directions this is HARMLESS: the equidistant
formula multiplies `phi`'s cosine/sine by `r`, which is exactly zero at that point regardless of what
arbitrary value `atan2(0,0)` returns — the output pixel lands on `(cx, cy)` (forward) or the ray lands
on the optical axis `(0,0,1)` (inverse) either way. This is a genuine, if narrow, numerics lesson: not
every "0/0"-shaped expression is a real hazard — the surrounding algebra can render it moot, and it is
worth checking explicitly (as this project's code comments do) rather than assuming every division by
a near-zero quantity is dangerous.

**theta near and past 90 degrees.** `theta = atan2(hypot(X,Y), Z)` is well-conditioned across its ENTIRE
`[0, pi]` domain — unlike, say, `acos(Z / |v|)`, whose derivative blows up as the argument approaches
±1 (i.e. as `theta` approaches 0 or `pi`), `atan2` of two arguments that are never SIMULTANEOUSLY zero
(the ray direction is never the zero vector) has no such blow-up. This is precisely why the equidistant
model handles `theta > 90 degrees` — the exact regime a pinhole/rectilinear model cannot even
represent — without any special-casing: PART 1 of `kernels.cuh` and this project's `model_roundtrip`
gate both exercise angles well past 90 degrees for exactly this reason.

**Bilinear sampling at the fisheye rim.** Near the design FOV boundary (`theta` approaching
`kFishValidHalfFovRad`), a bilinear sample's 4 texels may straddle the transition from real scene
content to the vignetted black region — this project's clamp-to-edge boundary policy (never sampling
outside `[0,W)x[0,H)`) avoids reading garbage memory, but a sample landing exactly at the boundary can
still blend a small amount of "black" into an otherwise-valid pixel. The linear feather weight
(`kFeatherBandRad`, ramping to zero over the last ~8.6 degrees before the boundary) is this project's
mitigation: pixels near the edge are down-weighted in the BEV blend precisely where this contamination
would otherwise be most visible.

**LUT quantization.** `RemapSample` stores `(u,v)` as `float` — single precision, ~7 significant
decimal digits — comfortably more precision than an 8-bit image needs (a fisheye image at 320x240 never
needs sub-1e-4-pixel accuracy to look correct), so LUT quantization contributes nothing measurable to
this project's VERIFY tolerances; the dominant GPU-vs-CPU divergence source is the same one 01.01
documents — nvcc's default FMA-contraction of the bilinear interpolation's `a + (b-a)*t` into one
rounding step on device, versus two rounding steps on host without `/fp:fast` — bounded at ~1 uint8
unit, which is exactly this project's `kTolUint8 = 1.5` tolerance.

**Determinism.** Every computation in this project (camera model, rig geometry, bilinear sampling,
BEV blend) is a deterministic function of fixed inputs — no RNG, no atomics that could reorder a
floating-point sum, no iterative solver with a data-dependent trip count. The ONLY randomness anywhere
in this project's pipeline lives in `scripts/make_synthetic.py`'s xorshift32-hashed value noise, which
is itself fully deterministic given the fixed seed (42) — the committed sample is bit-for-bit
reproducible.

## How we verify correctness

**Tier 1 — VERIFY (GPU vs CPU twin).** Every kernel (`build_rect_lut_kernel`, `build_cyl_lut_kernel`,
`remap_bilinear_kernel`, `bev_compose_kernel`) is compared, element-wise, against
`reference_cpu.cpp`'s independently-retyped plain-C++ implementation. Per this repo's twin-independence
ruling (`reference_cpu.cpp`'s file header): `fisheye_project`/`fisheye_unproject`/
`pinhole_unproject_rect`/`cyl_unproject`/`rig_camera_to_bev_sample` are SHARED (the camera model and
rig geometry are DATA — the physical lens and rig — not "the algorithm under test"), while the
bilinear sampling, the LUT-build loop structure, and the entire BEV multi-camera blend + coverage
accumulation are retyped a SECOND time, independently, in `reference_cpu.cpp`. Measured on the
reference machine (RTX 2080 SUPER, sm_75), Release: LUT max error 0.0000305 px (tol 0.001), remap/BEV
max error 1.0000 uint8 units (tol 1.5, the FMA-contraction bound above), coverage bitmask exact (0
difference, tol 0.5 — pure integer computation).

**Tier 2 — 7 independent physical gates**, each checking something the twin comparison CANNOT (a bug
shared between `kernels.cu` and `reference_cpu.cpp` — e.g. a wrong rig mount coordinate baked into
`kernels.cuh` — would pass VERIFY perfectly and only be caught here):

- **`model_roundtrip`** — the equidistant model's forward/inverse formulas are hand-retyped a THIRD
  time, in double precision, entirely bypassing `kernels.cuh`, and exercised over a `theta` grid
  including angles past 90 degrees (measured: 0.0000000 px max error — the equidistant model's inverse
  is exact and closed-form, so this is essentially a floating-point-rounding-only test, in sharp
  contrast to 01.01's iterative Brown-Conrady roundtrip gate).
- **`straightness_rectilinear` / `distortion_negative_control`** — a genuinely world-straight ground-
  plane edge (the boundary line in the synthetic scene) is located via a from-scratch, host-side
  threshold-crossing detector, and a least-squares best-fit LINE (not a constant-mean spread, since
  this edge is not vertical in image space) measures its residual. Measured: 0.2947 px in the
  rectilinear unwarp (tol <= 1.0), 60.8862 px in the RAW fisheye image (tol >= 40.0, the negative
  control proving real curvature is being corrected).
- **`bev_ground_truth`** — the stitched BEV output is compared to `data/sample/bev_ground_truth.ppm`
  (an exact, camera-free orthographic render of the ground texture) in pixels that are covered,
  away from any tall object's footprint, and away from any camera-boundary seam. Measured: mean
  |error| 10.9495 (0-255 scale, tol <= 15.0) — see README "Limitations & honesty" for what dominates
  this residual.
- **`flat_ground_assumption`** — the SAME ground-truth comparison, restricted to pixels near a tall
  object's footprint: measured mean |error| 37.3365 (tol >= 20.0, a floor proving the assumption's
  failure is real and large, not a rounding artifact) — 3.41x the flat-region error.
- **`seam_consistency`** — in every BEV pixel seen by exactly 2 cameras, both cameras' PRE-blend
  samples of the SAME ground point are independently re-sampled (via a THIRD, gate-local bilinear
  sampler, never `kernels.cu`'s or `reference_cpu.cpp`'s) and compared. Measured: mean |A-B| 17.5811
  (tol <= 24.0) — proof the 4 cameras' extrinsics and the shared model are mutually consistent.
- **`coverage`** — every BEV pixel within a 3.4 m design radius of the vehicle center is checked for
  at least 1 contributing camera. Measured: 100.000% covered (tol >= 99.5%); 63.36% of the full BEV
  crop is seen by 2+ cameras (reported, not gated — an overlap AMOUNT is a rig-design choice).

Every tolerance above is a floor/ceiling calibrated with margin over the ACTUAL measured value on the
committed sample (never fabricated — `src/main.cu`'s tolerance-block comment records every number),
robust to the small, legitimate cross-GPU-architecture float drift 01.01's output-contract note
describes.

## Where this sits in the real world

Production fisheye undistort (README "Prior art"): OpenCV's `fisheye` module implements exactly this
project's equidistant-family LUT-build-and-remap pattern, generalized to the Kannala-Brandt polynomial
and driven by a checkerboard-based calibration procedure (Kalibr, or OpenCV's own
`fisheye::calibrate`) rather than a hand-specified model — PRACTICE.md §3 describes that calibration
procedure for a real rig. NVIDIA DriveWorks and VPI ship production, GPU-accelerated multi-camera
surround-view modules that solve exactly this project's Half 2, at automotive-grade resolution and
frame rate, with additional production concerns this project deliberately omits: photometric
(exposure/white-balance) blending across cameras, temporal smoothing/ghosting suppression, and
calibrated per-vehicle extrinsics from an on-line or factory calibration step rather than a hardcoded
rig. The flat-ground-assumption failure this project measures honestly is an ACTIVE research area, not
a solved problem — production systems mitigate it (never eliminate it) via higher-resolution sensors,
multi-frame temporal fusion, or a genuine depth source (stereo pairs, structure-from-motion, or a
LiDAR/radar fusion layer) for content above the ground plane; this project's teaching version stops at
demonstrating the failure clearly and honestly, which is itself the first step toward understanding
why those mitigations exist.
