// ===========================================================================
// kernels.cuh — interface for project 02.17
//               LiDAR-camera projection/coloring fusion kernels
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (staged verification + gates + artifacts),
// kernels.cu (the GPU kernels), and reference_cpu.cpp (the independent CPU
// oracle twins). Per this repo's twin-independence ruling (see
// reference_cpu.cpp's file header): ONLY data-layout — structs and numeric
// constants — is single-sourced here; the ALGORITHMIC CORE (the rigid
// transform + pinhole projection, the encoded-atomicMin z-buffer, bilinear
// sampling, the occlusion depth-consistency test) is written TWICE,
// independently, once in kernels.cu (device) and once in reference_cpu.cpp
// (host) — so main.cu's VERIFY stage actually catches bugs instead of
// comparing a formula to itself.
//
// THE TWO DIRECTIONS THIS PROJECT TEACHES (catalog bullet: "LiDAR-camera
// projection/coloring fusion kernels")
// -------------------------------------------------------------------------
//   DIRECTION A — POINT COLORING (LiDAR -> camera): put a camera pixel's
//     color onto each LiDAR point. The hard part is OCCLUSION: a point on a
//     far surface can still project into the SAME image pixel a near
//     surface occupies (two sensors at different origins see around corners
//     differently — THEORY.md derives this project's exact occlusion
//     geometry from the camera/LiDAR baseline). Naively sampling whatever
//     pixel a point lands on paints occluded points with the WRONG (near
//     surface's) color; this project measures that failure directly (the
//     "naive" path) and fixes it with an honest z-buffer visibility check
//     (the "checked" path) — both gated in main.cu (GATE occlusion_correctness).
//   DIRECTION B — DEPTH PAINTING (LiDAR -> image plane): the SAME z-buffer
//     projection pass IS a sparse depth image — the fused, RGBD-like product
//     (no completion/densification here; 01.18 is the completion sibling,
//     cited, not reimplemented).
//
// FOUR KERNELS, FOUR SMALL GPU PATTERNS (each documented at its definition
// in kernels.cu):
//   1) project_zbuffer_kernel   — SCATTER + atomicMin (01.18's z-buffer
//      trick, cited): one thread per POINT, nearest-wins per PIXEL. This
//      kernel's output IS Direction B's product (a sparse depth image) and
//      Direction A's occlusion oracle.
//   2) project_points_kernel    — MAP: one thread per point, the pinhole
//      projection alone (01.18/01.17's formula, cited), producing continuous
//      (u,v) pixel coordinates, camera-frame depth, and an in-frustum flag —
//      the shared geometric core BOTH directions and the calibration-error
//      sensitivity sweep build on.
//   3) sample_bilinear_kernel   — MAP: one thread per point, bilinear color
//      sampling at (u,v) (01.01 lineage, cited) — the NAIVE coloring path
//      (Direction A without the occlusion fix).
//   4) check_occlusion_kernel   — MAP: one thread per point, compares this
//      point's own camera-frame depth against the z-buffer's PIXEL winner
//      (from kernel 1, same T) within a documented band; the CHECKED
//      coloring path is "kernel 3's color, kept only where kernel 4 says
//      visible" — the occlusion fix, earning its keep in color.
//
// The calibration-error sensitivity sweep (main.cu, README/THEORY "The
// algorithm in brief") reuses kernels 2+3 at PERTURBED T_camera_lidar values
// — no new kernel needed; only main.cu's evaluation-only host code changes.
//
// CAMERA MODEL & EXTRINSIC — IDENTICAL numeric convention to 01.17/01.18/02.02
// (cited throughout, not re-derived): pinhole intrinsics (fx,fy,cx,cy) in the
// OPTICAL frame (z-forward, x-right, y-down); depth means Pcam.z, never
// Euclidean range; Rigid3 { R[9] row-major; t[3] } with P_dst = R*P_src + t;
// kTCameraLidar is numerically IDENTICAL to 01.18/02.02's own constant (the
// same roof-LiDAR-over-windshield-camera rig — see THEORY.md "The math" for
// the derivation, owned by 01.18, cited here). This project is the reason
// 01.17's calibration and 01.18/02.02's fixed extrinsic MATTER: it is where a
// wrong T_camera_lidar becomes a visibly wrong-colored point cloud (the
// calibration-error sensitivity sweep quantifies exactly this).
//
// UINT-ENCODED ATOMIC MIN — see 01.18's kernels.cuh file header for the full
// derivation (reused verbatim here, cited): CUDA has no atomicMin for float;
// reinterpreting a POSITIVE finite float's IEEE-754 bits as uint32_t preserves
// ordering, so atomicMin on the reinterpreted bits keeps the smallest depth.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t/uint8_t — the z-buffer encoding word and boolean-ish flags

// ===========================================================================
// Image / camera constants — IDENTICAL to 01.17/01.18/02.02's teaching camera
// (cited; not re-derived) so this project's numbers are directly comparable
// to its upstream (01.17's calibration, 01.18's projection) and sibling
// (02.02's frustum crop) projects.
// ===========================================================================
constexpr int   kImageWidth  = 160;   // px
constexpr int   kImageHeight = 120;   // px
constexpr int   kImagePixels = kImageWidth * kImageHeight;
constexpr float kFx = 154.0f;         // px focal length x (~56.5 deg horizontal FOV)
constexpr float kFy = 152.0f;         // px focal length y (~44.7 deg vertical FOV)
constexpr float kCx = 80.0f;          // px principal point x
constexpr float kCy = 60.0f;          // px principal point y

// A point at exactly this range or beyond is out of the LiDAR's own maximum
// range (eye-safety-limited transmit power — 01.18's THEORY.md derives the
// link budget, cited).
constexpr float kMaxDepthM = 20.0f;   // m

// Sentinel meaning "no value here" — depth is always positive, so a negative
// sentinel can never be confused with a real reading (01.18's convention).
constexpr float kInvalidDepth = -1.0f;

// ---------------------------------------------------------------------------
// Rigid3 — a rigid-body transform, IDENTICAL shape to 01.17/01.18/02.02:
// P_dst = R * P_src + t, R stored ROW-MAJOR (R[r*3+c]). A pure data-layout
// struct (no member functions) — safe from both nvcc (.cu) and cl.exe
// (reference_cpu.cpp) with no __host__/__device__ decoration needed.
// ---------------------------------------------------------------------------
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation
    float t[3];   // translation, meters, in the DESTINATION (camera) frame
};

// kTCameraLidar — P_camera = R * P_lidar + t. NUMERICALLY IDENTICAL to
// 01.18/02.02's own constant (a roof-mounted LiDAR, x-forward/y-left/z-up,
// 0.30 m above and 0.05 m behind a windshield-height camera, zero relative
// tilt — the rotation is a clean axis PERMUTATION: camera-z=lidar-x,
// camera-x=-lidar-y, camera-y=-lidar-z). Derivation owned by 01.18's
// THEORY.md "The math" (cited, not re-derived) — this project's own
// THEORY.md instead derives the OCCLUSION geometry this fixed rig produces
// (why a LiDAR mounted higher than the camera creates the exact "sees over
// the occluder, camera doesn't" cohort scripts/make_synthetic.py builds).
constexpr Rigid3 kTCameraLidar = {
    { 0.0f, -1.0f,  0.0f,
      0.0f,  0.0f, -1.0f,
      1.0f,  0.0f,  0.0f },
    { 0.0f, -0.30f, -0.05f }
};

// ---------------------------------------------------------------------------
// LidarPointF — one LiDAR return, POSITION ONLY, IN THE LIDAR'S OWN FRAME
// (meters). IDENTICAL shape to 01.18's struct. This is exactly what
// data/sample/lidar_points.csv's x,y,z columns store and what every kernel
// below consumes. The CSV's true_r/true_g/true_b/visible/surface columns are
// EVALUATION-ONLY ground truth: main.cu parses them into separate host
// arrays that never touch a kernel or a CPU-twin function — see the file
// header's "independent gate" note and scripts/make_synthetic.py's own
// header for why they must stay outside both verified code paths.
// ---------------------------------------------------------------------------
struct LidarPointF {
    float x, y, z;   // meters, LiDAR frame
};

// ---------------------------------------------------------------------------
// Occlusion depth-consistency band + search window (Direction A, kernel 4).
// A LiDAR point at pixel (px,py) is accepted as "visible" only if its OWN
// camera-frame depth is within kOcclusionBandM of the NEAREST z-buffer
// evidence found within a (2*kOcclusionWindowRadiusPx+1) square window
// centered on (px,py) — never exactly equal at the single pixel, and never
// restricted to that single pixel either, for two compounding reasons this
// project's own measurements exposed (README/THEORY document the numbers):
//   (a) LiDAR range noise (make_synthetic.py: sigma=0.02 m) and multiple
//       beams landing on one discretized pixel from slightly different
//       points on a continuous surface both perturb a truly-visible point's
//       depth away from any single winning value — the BAND absorbs this.
//   (b) A sparse LiDAR's own angular sampling (this project's committed scan
//       spaces adjacent returns ~1-3 px apart, README "Data") means the
//       OCCLUDING surface frequently has NO return on the EXACT pixel a
//       hidden point behind it lands on, even though the occluder visibly
//       covers that pixel in the dense camera image. Restricting the check
//       to one exact pixel therefore MISSES most real occlusions (measured:
//       WITH-check wrong-color rate barely improved on the occlusion cohort)
//       — the WINDOW widens the search for occluder evidence to the
//       occluder's own local neighborhood without needing every pixel
//       individually painted, at the cost of occasionally over-filtering a
//       truly-visible point near a real depth edge (README "Limitations").
// THEORY.md "The math" derives this project's designed occlusion gap: the
// RED occluder sits ~4 m out, the GREEN background it hides ~12 m out — an
// ~8 m true discontinuity, far past this band, so the band trivially
// separates "same surface, sensor noise" from "genuinely different surface,
// occlusion" once the window finds the occluder's evidence at all.
// 0.30 m mirrors 01.18's own depth-discontinuity threshold (kDepthEdgeThreshM,
// cited) — tuned there for a similar noise/geometry regime.
constexpr float kOcclusionBandM = 0.30f;          // m
constexpr int   kOcclusionWindowRadiusPx = 2;     // search a 5x5 pixel window (measured-then-chosen, README "Data")

// ---------------------------------------------------------------------------
// Calibration-error sensitivity sweep constants (main.cu's evaluation-only
// orchestration reuses kernels 2+3 at each of these T_camera_lidar
// perturbations — README/THEORY document the 01.17-derived analytic
// consistency check). Rotation perturbs R only (about the camera's Y/"down"
// axis — a yaw-like mounting error, producing a mostly-HORIZONTAL pixel
// shift, THEORY.md derives why); translation perturbs t only (along the
// camera's X/"right" axis — a lateral mounting error).
// ---------------------------------------------------------------------------
constexpr int   kNumSensitivityLevels = 3;
constexpr float kSensitivityRotDeg[kNumSensitivityLevels]   = { 0.2f, 0.5f, 1.0f };   // degrees
constexpr float kSensitivityTransCm[kNumSensitivityLevels]  = { 1.0f, 2.0f, 5.0f };   // centimeters

// A sampled color is judged to have "crossed a designed color boundary"
// relative to the zero-perturbation baseline when the max-absolute-channel
// difference (normalized [0,1], the SAME measure 01.18's conductance uses,
// cited) exceeds this. The scene's four surfaces (README: red/green/blue/
// yellow/gray) are chosen with large mutual color distances specifically so
// this threshold cleanly separates "still sampling the same surface" from
// "drifted onto a neighboring one" — README/THEORY document the measured
// flip-fraction curve this threshold produces.
constexpr float kColorBoundaryThresh = 0.25f;   // normalized [0,1] max-channel diff

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// project_zbuffer_kernel — scatter N LiDAR points into the WxH image with a
// nearest-wins z-buffer (kernel 1 — see file header). d_encoded:
// [kImagePixels] device array the CALLER must cudaMemset to 0xFF bytes
// (UINT32_MAX, the "empty pixel" sentinel) before the launch. T is passed BY
// VALUE (a `constexpr` global has no device-side storage unless additionally
// marked __device__/__constant__ — 01.18's convention, cited) — the SAME
// parameter every caller (Direction A/B, the sensitivity sweep) supplies,
// which is exactly what lets the sensitivity sweep reuse this kernel at
// PERTURBED T values with zero code duplication.
__global__ void project_zbuffer_kernel(const LidarPointF* __restrict__ d_pts,
                                       int n_pts,
                                       Rigid3 T,
                                       uint32_t* __restrict__ d_encoded);

// project_points_kernel — the shared geometric core (kernel 2): one thread
// per point, pure pinhole projection, NO z-buffer, NO color. d_u/d_v: [n_pts]
// OUT, continuous (unrounded) pixel coordinates (meaningful even OUTSIDE the
// image — the sensitivity sweep measures sub-pixel displacement here).
// d_zc: [n_pts] OUT, camera-frame depth (Pcam.z; <=0 or >kMaxDepthM means
// "behind camera or out of range", reflected in d_in_frustum). d_in_frustum:
// [n_pts] OUT, 1 iff zc in (0, kMaxDepthM] AND the ROUNDED pixel (same
// floor(x+0.5) convention kernel 1 uses) lands inside [0,W)x[0,H).
__global__ void project_points_kernel(const LidarPointF* __restrict__ d_pts,
                                      int n_pts,
                                      Rigid3 T,
                                      float* __restrict__ d_u,
                                      float* __restrict__ d_v,
                                      float* __restrict__ d_zc,
                                      uint8_t* __restrict__ d_in_frustum);

// sample_bilinear_kernel — kernel 3: one thread per point, bilinear color
// sample at (d_u[i], d_v[i]) from the PLANAR normalized-[0,1] guidance image
// d_rgb ([3*kImagePixels]: plane 0 = red, plane 1 = green, plane 2 = blue —
// 01.18's layout, cited). Points with d_in_frustum[i]==0 write (0,0,0) — a
// documented, honest "no color" rather than an uninitialized read. d_color:
// [3*n_pts] OUT, INTERLEAVED (r,g,b) per point — a small per-point array read
// straight back by main.cu's gates, so interleaved (not the image's own
// planar layout) is the natural shape here.
__global__ void sample_bilinear_kernel(const float* __restrict__ d_u,
                                       const float* __restrict__ d_v,
                                       const uint8_t* __restrict__ d_in_frustum,
                                       int n_pts,
                                       const float* __restrict__ d_rgb,
                                       float* __restrict__ d_color);

// check_occlusion_kernel — kernel 4: one thread per point, the occlusion fix.
// Recomputes this point's OWN rounded pixel from (d_u[i], d_v[i]), scans the
// (2*kOcclusionWindowRadiusPx+1) square NEIGHBORHOOD of that pixel in the
// z-buffer (d_encoded, produced by kernel 1 with the SAME T this point was
// projected with) for the NEAREST (smallest-depth) evidence found anywhere
// in the window, decodes it, and accepts this point as "visible" iff
// |d_zc[i] - nearest_in_window| <= band_m AND the window actually HAS any
// evidence at all (an entirely empty window — no LiDAR point landed
// anywhere nearby under THIS z-buffer pass — cannot confirm visibility, so
// the point is conservatively marked NOT visible; see kernels.cu for why
// "no evidence" must not default to "assume visible", and kernels.cuh's
// kOcclusionWindowRadiusPx comment for why a single exact pixel is not
// enough). d_visible: [n_pts] OUT.
__global__ void check_occlusion_kernel(const float* __restrict__ d_u,
                                       const float* __restrict__ d_v,
                                       const float* __restrict__ d_zc,
                                       const uint8_t* __restrict__ d_in_frustum,
                                       int n_pts,
                                       const uint32_t* __restrict__ d_encoded,
                                       float band_m,
                                       uint8_t* __restrict__ d_visible);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launchers — declared OUTSIDE the __CUDACC__ fence so any translation
// unit may call them; only their DEFINITIONS (kernels.cu) need nvcc. Each
// owns its kernel's grid/block math and the mandatory post-launch error check.
// ---------------------------------------------------------------------------
void launch_project_zbuffer(const LidarPointF* d_pts, int n_pts, Rigid3 T, uint32_t* d_encoded);
void launch_project_points(const LidarPointF* d_pts, int n_pts, Rigid3 T,
                           float* d_u, float* d_v, float* d_zc, uint8_t* d_in_frustum);
void launch_sample_bilinear(const float* d_u, const float* d_v, const uint8_t* d_in_frustum, int n_pts,
                            const float* d_rgb, float* d_color);
void launch_check_occlusion(const float* d_u, const float* d_v, const float* d_zc,
                            const uint8_t* d_in_frustum, int n_pts,
                            const uint32_t* d_encoded, float band_m, uint8_t* d_visible);

// ---------------------------------------------------------------------------
// CPU reference twins (reference_cpu.cpp) — the independent oracle for each
// kernel (see this header's file comment and reference_cpu.cpp's file header
// for the sharing ruling). All pointers below are HOST pointers.
// ---------------------------------------------------------------------------

// project_zbuffer_cpu — sequential nearest-wins z-buffer (no atomics needed:
// single-threaded, so a plain "is this closer than what's there" compare
// suffices — the GPU's atomicMin trick exists ONLY because many threads race
// on the same pixel). out_depth: [kImagePixels], pre-filled with
// kInvalidDepth by this function (01.18's convention).
void project_zbuffer_cpu(const LidarPointF* pts, int n_pts, Rigid3 T, float* out_depth);

// project_points_cpu — independent host twin of kernel 2. u,v,zc,in_frustum:
// [n_pts] OUT (in_frustum as 0/1 stored in a uint8_t array).
void project_points_cpu(const LidarPointF* pts, int n_pts, Rigid3 T,
                        float* u, float* v, float* zc, uint8_t* in_frustum);

// sample_bilinear_cpu — independent host twin of kernel 3. rgb: HOST
// [3*kImagePixels] planar normalized-[0,1] image (same layout the GPU path
// reads). color: [3*n_pts] OUT, interleaved.
void sample_bilinear_cpu(const float* u, const float* v, const uint8_t* in_frustum, int n_pts,
                         const float* rgb, float* color);

// check_occlusion_cpu — independent host twin of kernel 4. depth: HOST
// [kImagePixels] plain float depth map (project_zbuffer_cpu's OWN output —
// NOT the GPU's encoded array; this keeps the CPU path free of the
// atomicMin-encoding detail entirely, exactly as 01.18's CPU twin does).
// visible: [n_pts] OUT.
void check_occlusion_cpu(const float* u, const float* v, const float* zc, const uint8_t* in_frustum,
                         int n_pts, const float* depth, float band_m, uint8_t* visible);

#endif // PROJECT_KERNELS_CUH
