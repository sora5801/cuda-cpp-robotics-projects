// ===========================================================================
// kernels.cuh — interface for project 01.07
//               Fisheye/omnidirectional unwarping and multi-camera
//               surround-view stitching (bird's-eye view, BEV)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver), kernels.cu (the GPU kernels),
// and reference_cpu.cpp (the independent CPU oracle) — every camera-model
// formula, rig constant, image layout, and sentinel all three must agree on
// lives HERE, once (CLAUDE.md §12), exactly the discipline sibling flagship
// 01.01 (full GPU image pipeline) uses for its Brown-Conrady pinhole model.
// This file follows 01.01's documentation style closely and contrasts with
// it constantly: 01.01 undoes a NARROW-FOV pinhole lens with polynomial
// (Brown-Conrady) distortion; this project undoes a WIDE-FOV (185-class)
// fisheye lens with an angle-linear (equidistant) projection. Read 01.01's
// kernels.cuh first if you have not — the LUT + inverse-mapping + bilinear
// pattern below is deliberately the same shape, applied to a very different
// camera model.
//
// ===========================================================================
// PART 1 — THE FISHEYE CAMERA MODEL (single source of truth)
// ===========================================================================
//
// Why a pinhole camera CANNOT see 185 degrees (the physics, briefly; full
// derivation in THEORY.md "The problem")
// -----------------------------------------------------------------------
// A pinhole/rectilinear lens obeys r = f*tan(theta): a ray at angle theta
// from the optical axis lands at image radius r = f*tan(theta). As
// theta -> 90 deg, tan(theta) -> infinity — a rectilinear image plane would
// need to be infinitely large to capture even a 180-degree FOV, and any
// FOV approaching 180 deg suffers extreme, unusable stretching near the
// edge (this is exactly why 01.01's rectilinear undistort/rectify stage
// never has to worry about "the lens saw past 90 degrees"). A fisheye lens
// is built (extra negative lens elements bending marginal rays, THEORY.md
// "The problem" + PRACTICE.md §1) to trade that infinite-radius blowup for
// bounded, PREDICTABLE angular compression instead — see the projection
// families below. The trade is real: fisheye images are NOT
// perspective-correct (straight 3-D lines curve, THEORY.md), which is
// precisely what this project's straightness gate measures.
//
// Projection function families (name -> r(theta); f = focal length in
// PIXELS, theta = angle from the optical axis in RADIANS; THEORY.md "The
// math" derives each from a physical lens-design argument and tabulates
// r(theta) at several angles):
//   rectilinear (pinhole, 01.01's model)   r = f * tan(theta)      — blows up at 90 deg
//   stereographic                          r = 2f * tan(theta/2)   — conformal (angles preserved locally)
//   EQUIDISTANT  <-- THIS PROJECT'S MODEL  r = f * theta           — angle-LINEAR, bounded for all theta
//   equisolid angle                        r = 2f * sin(theta/2)   — equal-AREA (common in "fisheye" marketing)
//   orthographic                           r = f * sin(theta)      — saturates before 90 deg (theta>90 impossible)
//
// This project uses EQUIDISTANT (r = f*theta) as the one, single-sourced
// model for every camera in the rig, for two deliberate teaching reasons:
//   1) It is the textbook "ideal fisheye" every intro treatment leads with,
//      and real equidistant-class lenses exist (surveillance/security
//      "fisheye" lenses are frequently sold on this spec).
//   2) It has a CLOSED-FORM projection AND a CLOSED-FORM inverse (theta =
//      r/f) — no fixed-point iteration anywhere in this file, a direct and
//      informative contrast with 01.01's Brown-Conrady model, whose inverse
//      has NO closed form and needs 20+ iterations (01.01 kernels.cuh's
//      file header). THEORY.md documents the Kannala-Brandt polynomial
//      r = f*(theta + k1*theta^3 + k2*theta^5 + ...) as the production
//      generalization real calibration toolchains (OpenCV `fisheye`,
//      Kalibr) fit per-lens; this project's k_i are implicitly all zero.
//
// Camera-optical frame (same stated exception 01.01 uses, CLAUDE.md §3.1's
// documented-exception clause): z-forward (down the optical axis), x-right,
// y-down (row 0 = image top). A 3-D ray direction (X, Y, Z) in this frame
// (not necessarily unit length — every formula below is scale-invariant,
// see "Numerical considerations" in THEORY.md) has:
//   theta = angle from the optical axis (+Z)  = atan2(hypot(X,Y), Z)   in [0, pi]
//   phi   = azimuth around the optical axis    = atan2(Y, X)           in (-pi, pi]
// fisheye_project() below turns (theta, phi) into a pixel via r=f*theta and
// (u,v) = (cx + r*cos(phi), cy + r*sin(phi)); fisheye_unproject() is its
// exact algebraic inverse. Both are HD (__host__ __device__) so kernels.cu
// (GPU) and reference_cpu.cpp (CPU oracle) — AND main.cu's rig-geometry
// helper below — all read the identical formula (CLAUDE.md §12); per this
// repo's twin-independence ruling (see reference_cpu.cpp's file header),
// sharing the camera-model formula itself is permitted (it is DATA — the
// physical lens — not "the algorithm under test") PROVIDED the project
// carries a verification gate that does NOT route through it: main.cu's
// "model roundtrip" gate hand-retypes both formulas independently, in
// double precision, over a theta grid that includes angles past 90 degrees
// (the exact regime a pinhole model cannot even represent).
//
// One fisheye lens, one set of intrinsics (kFishFx/kFishCx/kFishCy below),
// used for EVERY camera in this project — the single "physical lens" this
// whole project is built around, mounted 5 different ways: once as a
// stand-alone camera (Half 1 — unwarp), and once per rig position (Half 2 —
// 4-camera surround view, reusing the SAME rendered front-camera image as
// Half 1's single-camera input, so the two halves visibly share data).
//
// ===========================================================================
// PART 2 — HALF 1: SINGLE-CAMERA UNWARPING (two output projections)
// ===========================================================================
//
// A fisheye image is geometrically HONEST (every pixel really is where the
// lens put it) but perceptually unusable directly: 01.01's straightness
// argument applies in reverse here — a fisheye image bends every 3-D
// straight line that does not pass through the optical axis, which is
// disorienting for a human viewer or a downstream detector trained on
// rectilinear images. "Unwarping" re-projects the SAME captured light onto
// a friendlier output surface. This project builds two, teaching the
// projection-surface trade-off (THEORY.md "The algorithm"):
//
//   RECTILINEAR (pinhole) sub-FOV — perspective-correct (3-D lines stay
//     straight, the property 01.01's whole rectify stage exists to
//     restore), but only for a NARROW sub-FOV (this project: ~45 deg half
//     angle) — pinhole's r=f*tan(theta) blowup (see PART 1) means a wide
//     rectilinear re-projection of fisheye content stretches the periphery
//     unusably. Good for "look straight ahead," bad for "see everything."
//   CYLINDRICAL panorama — wraps the WIDE azimuth range around a cylinder
//     (radius = focal length) instead of a flat plane: horizontal lines at
//     the camera's own height stay straight, but verticals bow slightly off
//     that height — a deliberate, bounded trade that lets this project's
//     cylindrical output cover +-80 deg of azimuth (an amount that would
//     make a rectilinear image explode in width) at +-35 deg of elevation.
//     This is the same "projection surface" idea panoramic-photo stitchers
//     (Hugin, PTGui) use, taught here from a single fisheye source instead
//     of many overlapping narrow-FOV photos.
//
// Both are built the SAME way 01.01 builds its remap stage — INVERSE
// mapping (for every OUTPUT pixel, ask "which fisheye pixel does my ray
// come from?"), because inverse mapping guarantees a value at every output
// pixel with no holes, while forward-warping the fisheye image pixel by
// pixel would leave gaps wherever the fisheye's non-uniform pixel density
// under- or over-samples the output (01.01 kernels.cuh derives this
// argument for Brown-Conrady; it applies unchanged here). GPU: precompute a
// LUT of source (u,v) once per output pixel (purely geometric, content-
// independent, exactly 01.01's launch_build_remap_lut pattern), then a
// generic bilinear-gather kernel reads it — see kernels.cu.
//
// ===========================================================================
// PART 3 — HALF 2: FOUR-CAMERA SURROUND VIEW (bird's-eye view, BEV)
// ===========================================================================
//
// The flat-ground assumption (the idea that makes automotive/AMR BEV
// possible at all, and the idea this project's negative-control gate is
// built to falsify honestly)
// -----------------------------------------------------------------------
// A BEV compositor answers a purely GEOMETRIC question for every output
// (top-down) pixel: "which fisheye camera pixel(s) show the light that left
// THIS point on the ground?" That question is well-posed only if you know
// how far away the ground is along every ray — which a single 2-D camera
// image never tells you (LIDAR/stereo would; this project's 4 fisheye
// cameras do not). The industry-standard trick (and this project's, THEORY.md
// "The problem"): ASSUME every ray's first solid surface is the flat ground
// plane Z=0 at its geometrically-implied distance. For actual ground
// points, that assumption is exactly true and the reconstruction is exact
// (up to lens/calibration/blend error). For a TALL OBJECT (a parked car
// bumper, a pedestrian, a pallet), the assumption is FALSE: the ray hits
// the object first, so the BEV paints the object's color onto whatever
// ground point that ray direction would have reached had the object not
// been there — a radial "ghost"/smear stretching away from that camera's
// mount position. This project's flat_ground_assumption gate demonstrates
// exactly this failure on purpose (main.cu), the reason every real BEV
// product (Bosch, Continental multi-camera surround-view ECUs; see
// PRACTICE.md §4) either restricts itself to true top-down ground content
// or fuses in a real depth source for anything taller than the ground.
//
// This is an INVERSE mapping done through 3-D, not a 2-D homography (the
// distinction the catalog bullet and README both call out): a homography
// is the CLOSED-FORM special case of this same flat-ground assumption for a
// PINHOLE camera (a homography IS "assume Z=0, then the perspective map
// collapses to a 3x3 linear map in homogeneous coordinates" — a fact
// covered in THEORY.md "The math"); this project's cameras are fisheye
// (equidistant, not a linear projective map), so no single 3x3 matrix
// captures it — every BEV pixel is walked explicitly through 3-D: ground
// point -> vehicle-frame ray -> camera-frame ray (rig extrinsics) ->
// fisheye pixel (PART 1's model). rig_camera_to_bev_sample() below is that
// walk, shared (same twin-independence exception as fisheye_project/
// unproject: this is rig GEOMETRY/DATA, not the blending algorithm under
// test) by kernels.cu's BEV kernel and reference_cpu.cpp's independent BEV
// twin; the surrounding bilinear sampling and multi-camera blend accumulate
// are retyped independently in each (see reference_cpu.cpp's file header).
//
// The rig — 4 identical fisheye cameras (PART 1's ONE lens model), 4 mounts
// -----------------------------------------------------------------------
// Vehicle body frame (CLAUDE.md §12): right-handed, x-forward, y-left,
// z-up, origin at ground level under the vehicle's geometric center (so a
// camera's mount Z IS its height above the ground plane it must see).
// T_parent_child convention: T_vehicle_cam[i] describes camera i EXPRESSED
// IN the vehicle frame — mount position (translation) plus a 3x3 rotation
// whose ROWS are the camera's own (x_cam, y_cam, z_cam) unit axes, each
// written in vehicle-frame (bx, by, bz) components. Because the rig only
// ever needs vehicle-point -> camera-point (never the reverse — no ray
// casting happens in the GPU/CPU code, only in the synthetic generator,
// which does its own rig construction independently in Python), storing
// the rows this way lets rig_camera_to_bev_sample() apply the rotation with
// a single matrix-vector product and NO transpose at the hot call site:
//     P_cam = R_cam_vehicle * (P_vehicle - mount),   R_cam_vehicle = R_vehicle_cam^T
// (R is orthonormal, so "transpose" is just "the rows I already chose to
// store" — see CameraExtrinsic below.)
//
// ASCII rig diagram (top-down, +x forward is UP on the page, +y left is
// LEFT on the page — matches the vehicle-frame convention above):
//
//                              +x (forward)
//                                  ^
//                                  |
//                         .--[FRONT, tilt 45d down]--.
//                        /     mount (2.0, 0.0, 0.6)  \
//                       /                               \
//      [LEFT]---mount (0.0, 1.0, 1.1)          mount (0.0,-1.0, 1.1)---[RIGHT]
//   +y <--   facing +y, tilt 45d down    facing -y, tilt 45d down   --> -y
//                       \                               /
//                        \     mount (-2.0, 0.0, 0.6)  /
//                         '--[REAR,  tilt 45d down]---'
//                                  |
//                                  v
//                              -x (rearward)
//
// Every mount tilts 45 degrees DOWN from horizontal (about the camera's own
// right axis — see the derivation above CameraExtrinsic below), a single
// illustrative rig angle (PRACTICE.md §1 dates and discusses real rig
// tilts, typically 35-55 degrees depending on vehicle height and desired
// near-field coverage). All 4 cameras share PART 1's ONE lens model, so
// the ONLY per-camera difference anywhere in this file is WHERE it is
// bolted on and WHICH way it points — exactly how a real surround-view kit
// is specified (PRACTICE.md §2).
//
// IMAGE LAYOUTS (row-major, uint8 RGB interleaved unless noted — identical
// convention to 01.01's, so a PPM write is a direct memcpy everywhere):
//   FISHEYE   (kFishW x kFishH): captured/rendered camera images, one per
//     rig position (+ the stand-alone Half-1 camera, which reuses FRONT's).
//   RECTILINEAR / CYLINDRICAL (kRectW x kRectH / kCylW x kCylH): Half-1's
//     two unwarp outputs.
//   BEV (kBevW x kBevH): Half-2's stitched top-down output, plus a
//     kBevW x kBevH single-channel COVERAGE bitmask (bit i set = camera i
//     contributed to this pixel; see kCamFront..kCamRear below).
//   RemapSample LUT: identical struct to 01.01's, sized to the OUTPUT
//     resolution here (there is no separate resize stage in this project,
//     so — unlike 01.01 — the LUT is exactly the size of the image it
//     feeds, one entry per output pixel).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>     // atan2f/hypotf/sinf/cosf — every trig call in this file funnels through these

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. Identical
// trick to 01.01 (and 18.01, 10.03, ...): a function that must be textually
// IDENTICAL on both sides of the host/device boundary, so kernels.cu's GPU
// kernels and reference_cpu.cpp's plain-C++ oracle both call
// fisheye_project()/fisheye_unproject()/rig_camera_to_bev_sample() UNCHANGED.
// Sharing the source removes "did I transcribe the trig correctly twice?"
// as an explanation for a GPU-vs-CPU mismatch — what remains is genuinely
// numerics (THEORY.md "Numerical considerations"), which VERIFY exists to
// catch, and genuine algorithmic bugs in the retyped bilinear/blend code,
// which the model-roundtrip gate (bypassing this file) exists to catch.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// kPi — used only for a couple of defensive clamps below (theta cannot
// exceed pi for a physically-forward-looking ray; see fisheye_unproject).
// Not a general-purpose "angle wrapping" constant (CLAUDE.md §12 angle-
// wrap rule applies to STATE that accumulates over time; this project has
// none — every angle here is recomputed fresh from a fixed pixel or 3-D
// point each call, so there is nothing to wrap).
// ---------------------------------------------------------------------------
constexpr float kPi = 3.14159265f;

// ===========================================================================
// PART 1 constants — the ONE fisheye lens (PART 1's file-header derivation).
// kFishW x kFishH mirrors the task's "~320x240" fisheye views: big enough
// that the ~185-degree FOV is not a handful of pixels, small enough that
// four of them plus a synthetic ray-cast render stay fast and tiny to
// commit (CLAUDE.md §8).
// ---------------------------------------------------------------------------
constexpr int kFishW = 320;                          // fisheye image width, px
constexpr int kFishH = 240;                          // fisheye image height, px
constexpr float kFishCx = (kFishW - 1) * 0.5f;        // principal point x, px = 159.5 (exact image center; see 01.01's "-0.5" note)
constexpr float kFishCy = (kFishH - 1) * 0.5f;        // principal point y, px = 119.5

// kFishFx: the equidistant model's ONE focal-length constant (fx=fy — a
// real fisheye's image circle is round, not elliptical, so there is no
// separate horizontal/vertical focal length the way a pinhole model can
// have). Chosen (not derived) as a clean number; kFishValidHalfFovRad below
// is the SEPARATE, independently-chosen "how much of theta-space is inside
// the illuminated lens circle" bound — decoupling the two lets both be
// simple literals instead of one being solved in terms of the other.
// r = kFishFx * theta at theta = kFishValidHalfFovRad works out to
// 74.0 * 1.614430 = 119.47 px, just inside kFishCy = 119.5 — i.e. this
// project's chosen (fx, valid half-FOV) pair makes the illuminated circle
// inscribe almost exactly in the image HEIGHT, with the four CORNERS
// (r up to hypot(kFishCx,kFishCy) = 199.3 px, theta = 199.3/74 = 2.694 rad
// = 154.4 deg) extending well past the design FOV into the vignetted region
// a real fisheye lens goes dark in (THEORY.md "The problem": vignetting is
// PHYSICS — marginal rays at very large theta are attenuated/absorbed by
// the lens barrel and housing, not a rendering defect) — the synthetic
// generator paints that region black for exactly this reason.
constexpr float kFishFx = 74.0f;                      // equidistant focal length, px/radian

// kFishFullFovDeg: documentation/printf constant only (185-degree-class
// lens, matching the catalog bullet). kFishValidHalfFovRad is the value
// every formula below actually uses — HALF the full FOV, in radians,
// 92.5 deg = 92.5 * pi/180 = 1.6144295... rounded to float (same
// hand-rounding discipline 01.01 uses for kRectCos/kRectSin, so every
// translation unit links against the identical bit pattern).
constexpr float kFishFullFovDeg = 185.0f;             // documentation/printf only
constexpr float kFishValidHalfFovRad = 1.614430f;     // 92.5 deg — theta beyond this is outside the illuminated circle

// kFeatherBandRad: BEV blend weights ramp linearly to zero over this many
// radians of theta approaching kFishValidHalfFovRad (see
// rig_camera_to_bev_sample) — a soft "feather" rather than a hard cutoff,
// so the seam between "this camera stops contributing" and "the next one
// takes over" is a blend, not a visible hard edge (THEORY.md "The GPU
// mapping" discusses the alternative, a hard binary mask, and why a linear
// ramp is simpler to reason about and verify than a smoothstep).
constexpr float kFeatherBandRad = 0.15f;              // ~8.6 deg feather band width, radians

// ---------------------------------------------------------------------------
// RemapSample — one LUT entry: the (u, v) floating-point coordinate, in the
// FISHEYE image, that a given OUTPUT pixel should bilinearly sample.
// Identical shape to 01.01's RemapSample (a plain struct, not CUDA's
// float2 — CLAUDE.md §1 no-black-box-types rule); here the LUT is sized to
// the OUTPUT resolution directly (rectilinear or cylindrical), not to a
// shared full-resolution grid, because this project has no separate resize
// stage between "compute source pixel" and "use it".
// ---------------------------------------------------------------------------
struct RemapSample {
    float u;   // source column in the kFishW x kFishH fisheye image, may be fractional or out-of-range
    float v;   // source row    in the kFishW x kFishH fisheye image, may be fractional or out-of-range
};

// ---------------------------------------------------------------------------
// fisheye_project — PART 1's forward map: a camera-frame ray direction
// (X, Y, Z), ANY positive scale (every formula below is scale-invariant —
// hypot/atan2 only ever see RATIOS of X, Y, Z), to the fisheye PIXEL that
// ray lands on under the equidistant model r = f*theta.
//
// Steps:
//   1. theta = angle from the optical axis (+Z) = atan2(hypot(X,Y), Z).
//      atan2's two-argument form is well-conditioned over the WHOLE range
//      theta in [0, pi] — unlike asin/acos of a ratio, it never divides by
//      a quantity that can hit zero (THEORY.md "Numerical considerations"
//      contrasts this with acos(Z/|v|), which loses precision badly as
//      theta -> 0 because acos's derivative blows up there).
//   2. phi = azimuth around the optical axis = atan2(Y, X). Ill-defined
//      (arbitrary) only at X=Y=0 (theta=0, dead-center ray) — but harmless
//      there: r = f*0 = 0, so cos(phi)/sin(phi)'s arbitrary value gets
//      multiplied by zero (the du/dv below both land on exactly (cx,cy)
//      regardless of phi's value). A genuinely benign singularity, worth
//      naming explicitly rather than silently hoping nobody notices
//      (THEORY.md elaborates).
//   3. r = kFishFx * theta (the model itself — the ONE line that is
//      "equidistant" rather than any other projection family in PART 1's
//      table).
//   4. (u, v) = (kFishCx + r*cos(phi), kFishCy + r*sin(phi)).
//
// Parameters: X, Y, Z — camera-frame ray direction (need not be unit).
// Returns: RemapSample{u, v} — may lie outside [0,kFishW) x [0,kFishH) for
//   theta beyond the image bounds; callers clamp when bilinear-sampling
//   (kernels.cu / reference_cpu.cpp) or gate on visibility first (BEV path,
//   PART 3 — rig_camera_to_bev_sample checks theta against
//   kFishValidHalfFovRad BEFORE calling this, so BEV never samples the
//   vignetted corner region on purpose).
// ---------------------------------------------------------------------------
HD inline RemapSample fisheye_project(float X, float Y, float Z)
{
    const float r_xy = hypotf(X, Y);           // sqrt(X^2+Y^2), the "how far off-axis" radius in 3-D
    const float theta = atan2f(r_xy, Z);       // angle from +Z axis, in [0, pi] — see step 1 above
    const float phi = atan2f(Y, X);            // azimuth around +Z axis, in (-pi, pi] — see step 2 above
    const float r = kFishFx * theta;           // THE equidistant model: r is LINEAR in theta
    RemapSample s;
    s.u = kFishCx + r * cosf(phi);
    s.v = kFishCy + r * sinf(phi);
    return s;
}

// ---------------------------------------------------------------------------
// fisheye_unproject — PART 1's exact algebraic INVERSE of fisheye_project:
// a fisheye pixel (u, v) to the UNIT-length camera-frame ray direction that
// lands there. Unlike 01.01's Brown-Conrady undistort (a 20-iteration
// fixed-point search — see that file's header), this is five lines of
// closed-form trig: the equidistant model's whole didactic point.
//
// Steps (the exact reverse of fisheye_project's four steps):
//   1. (du, dv) = (u - kFishCx, v - kFishCy);  r = hypot(du, dv).
//   2. theta = r / kFishFx (r = f*theta, solved for theta — no search).
//      Defensively clamped to [0, pi]: a pixel far enough outside the
//      image that r > f*pi has no physical ray under this model (the
//      lens would have to see BEHIND itself); clamping instead of letting
//      theta run unbounded keeps sin/cos below well-defined and keeps this
//      function total (never NaN) for any float (u, v) — a robustness
//      property main.cu's roundtrip gate exercises directly.
//   3. phi = atan2(dv, du) — arbitrary only at r=0 (u=cx, v=cy), harmless
//      by the same argument as fisheye_project's step 2 (sin(theta)=0 -> X=Y=0
//      regardless of phi).
//   4. Direction = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)) —
//      a unit vector by construction (sin^2+cos^2=1 twice over).
//
// Parameters: u, v — fisheye pixel coordinates (any float; see step 2).
// Outputs: X, Y, Z — unit-length camera-frame ray direction.
// ---------------------------------------------------------------------------
HD inline void fisheye_unproject(float u, float v, float& X, float& Y, float& Z)
{
    const float du = u - kFishCx;
    const float dv = v - kFishCy;
    const float r = hypotf(du, dv);
    float theta = r / kFishFx;                 // r = f*theta, solved directly — the whole point of equidistant
    if (theta > kPi) theta = kPi;              // defensive clamp — see step 2 above
    const float phi = atan2f(dv, du);
    const float s = sinf(theta);
    X = s * cosf(phi);
    Y = s * sinf(phi);
    Z = cosf(theta);
}

// ===========================================================================
// PART 2 constants + helpers — Half 1's two unwarp output surfaces. Both
// output cameras look straight down the SAME optical axis the fisheye
// itself uses (no extra rotation — "dewarping" re-projects the SAME view
// onto a friendlier surface, it does not re-aim the camera), so their
// unproject helpers hand back a ray directly in the fisheye's own
// camera-optical frame, ready for fisheye_project() above.
// ===========================================================================

// Rectilinear (pinhole) sub-FOV output — kRectFx == kRectCx makes the
// horizontal half-FOV EXACTLY atan(kRectCx/kRectFx) = atan(1) = 45 degrees
// (a clean, documented number); the resulting vertical half-FOV is
// atan(kRectCy/kRectFx) = atan(74.5/99.5) = 36.8 degrees — comfortably
// inside the fisheye's kFishValidHalfFovRad (92.5 deg) with wide margin
// even at the output image's four corners (worst case ~51 deg — see
// kernels.cu's LUT-build kernel header for the measured corner theta).
constexpr int kRectW = 200;
constexpr int kRectH = 150;
constexpr float kRectCx = (kRectW - 1) * 0.5f;   // 99.5
constexpr float kRectCy = (kRectH - 1) * 0.5f;   // 74.5
constexpr float kRectFx = 99.5f;                 // == kRectCx -> exactly 45 deg horizontal half-FOV

// pinhole_unproject_rect — rectilinear OUTPUT pixel -> camera-frame ray
// (unnormalized; fisheye_project is scale-invariant, see its header). The
// textbook pinhole inverse: (X,Y,Z) = ((xo-cx)/fx, (yo-cy)/fy, 1).
HD inline void pinhole_unproject_rect(int xo, int yo, float& X, float& Y, float& Z)
{
    X = (static_cast<float>(xo) - kRectCx) / kRectFx;
    Y = (static_cast<float>(yo) - kRectCy) / kRectFx;   // square pixels: same fx for both axes
    Z = 1.0f;
}

// Cylindrical panorama output — columns sweep AZIMUTH (phi, around the
// camera's vertical/y axis... more precisely around the axis perpendicular
// to the horizontal sweep, see the derivation below), rows sweep
// ELEVATION (el, tilt up/down from the horizontal). +-80 deg azimuth would
// make a rectilinear image's width diverge (tan(80 deg) = 5.7 -> the image
// would need to be ~11x as wide as it is tall); the cylinder handles it in
// a plain kCylW x kCylH rectangle with bounded, PREDICTABLE (not
// accelerating) horizontal compression — THEORY.md "The algorithm" derives
// why (the cylinder's own circumference grows linearly with radius, the
// same reason a world map's Mercator-style cylindrical projection stays
// finite while a gnomonic/rectilinear one cannot reach the poles).
constexpr int kCylW = 280;
constexpr int kCylH = 120;
constexpr float kCylAzMaxRad = 1.396263f;   // 80 deg, radians (80*pi/180)
constexpr float kCylElMaxRad = 0.610865f;   // 35 deg, radians (35*pi/180)

// cyl_unproject — cylindrical OUTPUT pixel -> camera-frame ray. Column 0 is
// the LEFTMOST azimuth (-kCylAzMaxRad, i.e. toward +X_cam/+phi... see
// below) and column kCylW-1 is the rightmost (+kCylAzMaxRad); row 0 is the
// TOP (elevation +kCylElMaxRad, tilted up) and row kCylH-1 is the bottom
// (tilted down) — "up" in the y-down camera-optical frame means a NEGATIVE
// Y component, hence the "-sinf(el)" below.
//
// Derivation (camera-optical frame: z-forward, x-right, y-down): sweep
// azimuth "az" in the x-z plane first (as if elevation were zero: ray =
// (sin(az), 0, cos(az)) — a unit circle in the horizontal plane, az=0 is
// straight ahead), THEN tilt that ray up/down by elevation "el" about the
// camera's OWN x (right) axis, which only touches the y and z components:
//     X = sin(az) * cos(el)
//     Y = -sin(el)
//     Z = cos(az) * cos(el)
// (a unit vector by construction: X^2+Y^2+Z^2 = cos^2(el)*(sin^2(az)+cos^2(az)) + sin^2(el) = 1).
HD inline void cyl_unproject(int xo, int yo, float& X, float& Y, float& Z)
{
    const float az = -kCylAzMaxRad + static_cast<float>(xo) * (2.0f * kCylAzMaxRad) / static_cast<float>(kCylW - 1);
    const float el =  kCylElMaxRad - static_cast<float>(yo) * (2.0f * kCylElMaxRad) / static_cast<float>(kCylH - 1);
    const float cel = cosf(el);
    X = sinf(az) * cel;
    Y = -sinf(el);
    Z = cosf(az) * cel;
}

// ===========================================================================
// PART 3 constants + helpers — Half 2's 4-camera rig and BEV compositor.
// ===========================================================================

// Camera indices — this exact order is the bit layout of the BEV coverage
// mask (bit i set = camera i contributed to this output pixel; see
// kernels.cu's bev_compose_kernel) AND the switch-case order in
// rig_camera_to_bev_sample() below AND the argument order every BEV
// launcher/reference takes AND the file order ../scripts/make_synthetic.py
// writes/main.cu loads the 4 committed fisheye PPMs in. One order,
// cross-referenced everywhere (CLAUDE.md §12).
constexpr int kCamFront = 0;
constexpr int kCamLeft  = 1;
constexpr int kCamRight = 2;
constexpr int kCamRear  = 3;
constexpr int kNumRigCameras = 4;

// CameraExtrinsic — T_vehicle_cam for one rig camera: mount = camera's
// optical center in vehicle frame (meters); m[9] = R_cam_vehicle, ROW-
// MAJOR, row 0 = camera's x_cam (right) axis in vehicle (bx,by,bz)
// components, row 1 = y_cam (down), row 2 = z_cam (forward/viewing
// direction) — so P_cam.k = row_k . (P_vehicle - mount) (this file's PART
// 3 header explains why storing ROWS avoids a transpose at the call site).
struct CameraExtrinsic {
    float mount[3];   // camera optical center, vehicle frame, meters (bx, by, bz)
    float m[9];        // R_cam_vehicle, row-major (row0=x_cam, row1=y_cam, row2=z_cam, each in vehicle components)
};

// ---------------------------------------------------------------------------
// rig_extrinsic_for — the rig, derived once (by hand, in this file's PART 3
// header ASCII diagram section) and hardcoded here as the same kind of
// hand-rounded, bit-identical-across-translation-units literal 01.01 uses
// for kRectCos/kRectSin. Every camera shares ONE tilt angle (45 degrees
// down from horizontal, about the camera's own right/x_cam axis — the
// physical motion of "aiming the lens more at the ground") applied to a
// per-camera NOMINAL (untilted) basis that only depends on which way the
// camera faces (+-x or +-y, vehicle-frame) — see the per-case comments
// below for each one's x0/y0/z0 and the general tilt formula (x_cam=x0,
// y_cam=cos(t)*y0-sin(t)*z0, z_cam=sin(t)*y0+cos(t)*z0). Tilt trig,
// hand-rounded once: cos(45 deg) = sin(45 deg) = sqrt(2)/2 = 0.70710678.
// ../scripts/make_synthetic.py independently re-derives the SAME 4
// matrices from the SAME nominal-basis + tilt construction (its own Python
// trig, double precision) — two independent languages computing the same
// rig geometry from the same physical description is this project's
// cross-check that the numbers below are not a transcription slip
// (mirrors 01.01's 3-way camera-model cross-check).
//
// A SWITCH over 4 literal cases — not a namespace-scope array — is a
// deliberate toolchain choice, not a style preference: a local variable
// initialized from compile-time float literals compiles identically on
// host and device with no extra flags or storage-class annotations,
// whereas a namespace-scope ARRAY read by both host code (main.cu's gates,
// reference_cpu.cpp) and device code (kernels.cu's kernel) would need
// explicit __constant__/__device__ placement on the device side — which is
// then NOT directly readable from plain host code, defeating the whole
// point of one HD-shared function. This is exactly the kind of CUDA
// memory-space rule CLAUDE.md's "no black boxes" mandate asks to surface,
// not paper over (THEORY.md "Numerical considerations" revisits it).
// ---------------------------------------------------------------------------
HD inline CameraExtrinsic rig_extrinsic_for(int cam)
{
    CameraExtrinsic e{};
    switch (cam) {
    case kCamFront:
        // FRONT — mount at the front bumper center, 0.6 m up, tilted 45 deg
        // down. Nominal basis: x0=-by (right, facing +x), y0=-bz (down),
        // z0=+bx (forward).
        e.mount[0] = 2.0f; e.mount[1] = 0.0f; e.mount[2] = 0.6f;
        e.m[0] = 0.0f; e.m[1] = -1.0f; e.m[2] = 0.0f;                       // x_cam = -by
        e.m[3] = -0.70710678f; e.m[4] = 0.0f; e.m[5] = -0.70710678f;       // y_cam = -sin(t)*bx - cos(t)*bz
        e.m[6] =  0.70710678f; e.m[7] = 0.0f; e.m[8] = -0.70710678f;       // z_cam =  cos(t)*bx - sin(t)*bz
        break;
    case kCamLeft:
        // LEFT — mount at the left mirror, 1.0 m out / 1.1 m up, tilted 45
        // deg down. Nominal basis: x0=+bx (right, facing +y), y0=-bz (down), z0=+by (forward).
        e.mount[0] = 0.0f; e.mount[1] = 1.0f; e.mount[2] = 1.1f;
        e.m[0] = 1.0f; e.m[1] = 0.0f; e.m[2] = 0.0f;                        // x_cam = +bx
        e.m[3] = 0.0f; e.m[4] = -0.70710678f; e.m[5] = -0.70710678f;       // y_cam = -sin(t)*by - cos(t)*bz
        e.m[6] = 0.0f; e.m[7] =  0.70710678f; e.m[8] = -0.70710678f;       // z_cam =  cos(t)*by - sin(t)*bz
        break;
    case kCamRight:
        // RIGHT — mount at the right mirror, mirrored across the vehicle's
        // x-axis from LEFT. Nominal basis: x0=-bx, y0=-bz, z0=-by.
        e.mount[0] = 0.0f; e.mount[1] = -1.0f; e.mount[2] = 1.1f;
        e.m[0] = -1.0f; e.m[1] = 0.0f; e.m[2] = 0.0f;                       // x_cam = -bx
        e.m[3] = 0.0f; e.m[4] =  0.70710678f; e.m[5] = -0.70710678f;       // y_cam =  sin(t)*by - cos(t)*bz
        e.m[6] = 0.0f; e.m[7] = -0.70710678f; e.m[8] = -0.70710678f;       // z_cam = -cos(t)*by - sin(t)*bz
        break;
    default:   // kCamRear
        // REAR — mount at the rear bumper center, mirrored across the
        // vehicle's y-axis from FRONT. Nominal basis: x0=+by, y0=-bz, z0=-bx.
        e.mount[0] = -2.0f; e.mount[1] = 0.0f; e.mount[2] = 0.6f;
        e.m[0] = 0.0f; e.m[1] = 1.0f; e.m[2] = 0.0f;                        // x_cam = +by
        e.m[3] =  0.70710678f; e.m[4] = 0.0f; e.m[5] = -0.70710678f;       // y_cam =  sin(t)*bx - cos(t)*bz
        e.m[6] = -0.70710678f; e.m[7] = 0.0f; e.m[8] = -0.70710678f;       // z_cam = -cos(t)*bx - sin(t)*bz
        break;
    }
    return e;
}

// BEV output layout — a kBevW x kBevH top-down crop of the ground plane
// centered on the vehicle, kBevRangeM meters in every direction (an
// 8 m x 8 m patch at 320x320 px = 0.025 m/px, i.e. 2.5 cm/pixel ground
// sampling distance — a realistic surround-view resolution, PRACTICE.md
// §2). Row 0 = farthest FORWARD (+X), row kBevH-1 = farthest REARWARD
// (-X); column 0 = farthest LEFT (+Y), column kBevW-1 = farthest RIGHT
// (-Y) — i.e. the image reads like a map with the vehicle facing "up".
constexpr int kBevW = 320;
constexpr int kBevH = 320;
constexpr float kBevRangeM = 4.0f;   // ground half-extent in both X and Y, meters

// bev_pixel_to_ground — BEV OUTPUT pixel -> the ground-plane point
// (X, Y, 0) it represents, vehicle frame, meters. See kBevW/kBevH's
// comment above for the row/column -> +-X/+-Y orientation.
HD inline void bev_pixel_to_ground(int xo, int yo, float& X, float& Y)
{
    Y = kBevRangeM - static_cast<float>(xo) * (2.0f * kBevRangeM) / static_cast<float>(kBevW - 1);
    X = kBevRangeM - static_cast<float>(yo) * (2.0f * kBevRangeM) / static_cast<float>(kBevH - 1);
}

// kMinForwardZ — a ground point whose camera-frame Z (forward depth) is at
// or below this is treated as "behind (or grazing) the camera", never
// visible — guards the division-free but still ray-direction-dependent
// visibility test below from a degenerate near-zero-forward-depth point
// (THEORY.md "Numerical considerations").
constexpr float kMinForwardZ = 0.05f;   // meters

// ---------------------------------------------------------------------------
// rig_camera_to_bev_sample — PART 3's heart: given a rig camera index and a
// GROUND-PLANE point (X, Y, 0) in vehicle-frame meters, decide whether that
// camera can see it and, if so, where to sample its fisheye image and how
// much to trust that sample (feather weight). Shared HD helper (this file's
// PART 3 header explains the twin-independence exception this falls under
// — rig GEOMETRY/DATA, not the blending algorithm).
//
// Steps:
//   1. Vehicle-frame point -> camera-frame point: P_cam = R_cam_vehicle *
//      (P_ground - mount) (a plain 3x3 matrix-vector product against the
//      stored ROWS — see CameraExtrinsic's comment for why no transpose is
//      needed here).
//   2. Visibility gate 1: P_cam.z (the ray's own forward/depth component)
//      must exceed kMinForwardZ — otherwise the ground point is behind or
//      grazing the camera, not a valid ray at all.
//   3. theta = angle of the camera-frame ray from its own optical axis
//      (same formula fisheye_project uses internally, computed directly
//      here rather than by calling fisheye_project twice, since we need
//      theta for BOTH the visibility gate and the feather weight before
//      committing to computing (u,v)).
//   4. Visibility gate 2: theta must be <= kFishValidHalfFovRad — beyond
//      that the ray falls in the fisheye's vignetted/unmodeled region
//      (PART 1's kFishFx comment).
//   5. Feather weight: 1.0 for theta comfortably inside the valid circle,
//      ramping LINEARLY to 0.0 over the last kFeatherBandRad radians
//      before the boundary — see kFeatherBandRad's comment for why linear.
//   6. (u, v) via fisheye_project(P_cam) — computed only once both gates
//      pass, since it is meaningless (and possibly numerically extreme)
//      for out-of-FOV rays.
//
// Parameters: cam — one of kCamFront/Left/Right/Rear. X, Y — ground point,
//   vehicle-frame meters (Z is implicitly 0, the flat-ground assumption).
// Outputs: u, v — fisheye pixel to sample (valid only if visible=true is
//   also returned via the return value). weight — feather weight in
//   [0, 1] (valid only if visible).
// Returns: true iff this camera can see (X, Y, 0) at all (both gates
//   passed AND weight > 0 — a pixel exactly at the boundary contributes
//   zero weight, so callers may treat "visible but weight==0" as
//   equivalent to invisible; the two are kept distinct in the return value
//   for the coverage gate, which counts "visible" as "in the design FOV
//   footprint" regardless of how faded the feather makes it).
// ---------------------------------------------------------------------------
HD inline bool rig_camera_to_bev_sample(int cam, float X, float Y,
                                        float& u, float& v, float& weight)
{
    const CameraExtrinsic cx = rig_extrinsic_for(cam);
    const float dx = X - cx.mount[0];
    const float dy = Y - cx.mount[1];
    const float dz = 0.0f - cx.mount[2];        // ground point's Z is 0 (the flat-ground assumption itself)

    // Step 1: vehicle-frame delta -> camera-frame point (rows of m = camera axes).
    const float pcx = cx.m[0] * dx + cx.m[1] * dy + cx.m[2] * dz;   // x_cam component
    const float pcy = cx.m[3] * dx + cx.m[4] * dy + cx.m[5] * dz;   // y_cam component
    const float pcz = cx.m[6] * dx + cx.m[7] * dy + cx.m[8] * dz;   // z_cam component (forward depth)

    // Step 2: behind-camera gate.
    if (pcz <= kMinForwardZ) return false;

    // Step 3: angle from this camera's own optical axis.
    const float r_xy = hypotf(pcx, pcy);
    const float theta = atan2f(r_xy, pcz);

    // Step 4: outside-the-illuminated-circle gate.
    if (theta > kFishValidHalfFovRad) return false;

    // Step 5: linear feather over the last kFeatherBandRad radians.
    const float edge = kFishValidHalfFovRad - theta;   // radians of headroom before the boundary
    weight = (edge >= kFeatherBandRad) ? 1.0f : (edge / kFeatherBandRad);
    if (weight <= 0.0f) return false;                  // exactly at (or past) the boundary: no contribution

    // Step 6: project into this camera's fisheye pixel grid.
    const RemapSample s = fisheye_project(pcx, pcy, pcz);
    u = s.u;
    v = s.v;
    return true;
}

// ===========================================================================
// GPU launch wrappers (kernels.cu). Every wrapper computes its own launch
// geometry, launches, and calls CUDA_CHECK_LAST_ERROR — main.cu never
// touches <<<...>>> syntax directly (same discipline as 01.01).
// ===========================================================================

// Half 1a — precompute the rectilinear-output LUT (purely geometric,
// content-independent, computed once). d_lut OUT: kRectW*kRectH RemapSample.
void launch_build_rect_lut(RemapSample* d_lut);

// Half 1b — precompute the cylindrical-output LUT. d_lut OUT: kCylW*kCylH RemapSample.
void launch_build_cyl_lut(RemapSample* d_lut);

// Generic bilinear-gather remap — shared by BOTH Half-1 outputs (called
// twice with different LUT/dims, exactly like 01.01's remap_bilinear
// reused across its staged pipeline). d_src: srcW*srcH*3 uint8 fisheye
// image. d_lut: outW*outH RemapSample. d_out OUT: outW*outH*3 uint8.
void launch_remap_bilinear(const unsigned char* d_src, const RemapSample* d_lut,
                           unsigned char* d_out, int srcW, int srcH, int outW, int outH);

// Half 2 — the BEV compositor: one thread per BEV pixel, 4-camera loop
// in-thread (kernels.cu's kernel header argues this regime). d_front/left/
// right/rear: kFishW*kFishH*3 uint8 each. d_bev OUT: kBevW*kBevH*3 uint8
// (blended). d_coverage OUT: kBevW*kBevH uint8 (per-pixel camera bitmask,
// bit i = camera i contributed — see kCamFront..kCamRear above).
void launch_bev_compose(const unsigned char* d_front, const unsigned char* d_left,
                        const unsigned char* d_right, const unsigned char* d_rear,
                        unsigned char* d_bev, unsigned char* d_coverage);

// ===========================================================================
// CPU references (reference_cpu.cpp) — INDEPENDENT reimplementations (per
// the project template's twin-independence ruling — see that file's
// header) of every launcher above, same math, plain nested loops, "_cpu"
// suffix. fisheye_project/fisheye_unproject/pinhole_unproject_rect/
// cyl_unproject/rig_camera_to_bev_sample above are the deliberate,
// documented exception (shared camera-model + rig-geometry formulas); the
// bilinear sampling, the LUT-build loop structure, and the whole BEV
// multi-camera blend/coverage accumulation are all typed a second time
// here, independently, so the GPU-vs-CPU comparison in main.cu is not
// blind to bugs in the geometry-consuming code.
// ===========================================================================
void build_rect_lut_cpu(RemapSample* lut);
void build_cyl_lut_cpu(RemapSample* lut);
void remap_bilinear_cpu(const unsigned char* src, const RemapSample* lut,
                        unsigned char* out, int srcW, int srcH, int outW, int outH);
void bev_compose_cpu(const unsigned char* front, const unsigned char* left,
                     const unsigned char* right, const unsigned char* rear,
                     unsigned char* bev, unsigned char* coverage);

#endif // PROJECT_KERNELS_CUH
