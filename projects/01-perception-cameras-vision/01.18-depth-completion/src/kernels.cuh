// ===========================================================================
// kernels.cuh — interface for project 01.18
//               Depth completion: sparse LiDAR + RGB -> dense depth
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (staged-verification driver + evaluation
// gates), kernels.cu (the GPU kernels), and reference_cpu.cpp (the
// independent CPU oracle twins). Per this repo's twin-independence ruling
// (see reference_cpu.cpp's file header), ONLY data-layout — structs and
// numeric constants — is single-sourced here; the ALGORITHMIC CORE
// (projection math, conductance formula, diffusion update, IDW weights) is
// written TWICE, independently, once in kernels.cu (device) and once in
// reference_cpu.cpp (host) — so the GPU-vs-CPU comparison in main.cu's
// VERIFY stage actually catches bugs instead of comparing a formula to
// itself.
//
// THE PIPELINE (four stages, four kernel/launcher/CPU-twin triples below)
// -------------------------------------------------------------------------
//   1) PROJECTION + Z-BUFFER — scatter each LiDAR return (measured in the
//      LIDAR's own frame) into the camera image via the extrinsic
//      T_camera_lidar (the SAME Rigid3 shape and R*p+t convention project
//      01.17 calibrates — cited throughout; here we CONSUME a fixed,
//      already-known extrinsic rather than solving for it). Multiple LiDAR
//      points can land on the same pixel (a near surface partially
//      occluding a far one); the NEAREST wins — an honest z-buffer, not an
//      average, because averaging near and far returns at an occlusion
//      boundary would fabricate a physically meaningless mid-depth.
//   2) IDW BASELINE — inverse-distance-weighted interpolation from sparse
//      samples in a fixed-radius pixel window. Blind to the RGB image on
//      purpose: it is the "no prior" baseline the edge-aware method must
//      beat, and the yardstick the gates below measure against.
//   3) EDGE-AWARE ANISOTROPIC DIFFUSION — the project's MAIN densification
//      method. Depth is treated as the state of a 2-D heat-diffusion PDE
//      whose conductivity is GATED by the RGB image's own gradient (the
//      "edges coincide" prior: a strong image edge probably marks a depth
//      discontinuity, so diffusion should not smooth across it). Sparse
//      LiDAR samples are Dirichlet boundary conditions, reasserted every
//      iteration, so the PDE is really solving "fill in the pixels I don't
//      know, without ever overwriting the pixels I do."
//   4) EVALUATION — not a kernel at all; main.cu compares both densified
//      fields against the SYNTHETIC scene's exact dense truth and reports
//      the gates named in README "Expected output" / THEORY.md "How we
//      verify correctness": overall accuracy, edge quality, the texture-
//      trap and camouflage-edge honesty checks, and input fidelity.
//
// CAMERA MODEL — pinhole, intrinsics (fx, fy, cx, cy) in the SAME naming
// convention 01.16/01.17 use, OPTICAL frame (z-forward depth axis, x-right,
// y-down — the documented REP-103 exception SYSTEM_DESIGN.md notes for
// camera optics). depth in this project ALWAYS means Pcam.z (the pinhole
// z-buffer convention), never Euclidean range from the camera center — the
// same convention every RGB-D sensor and depth-completion paper uses.
//
// EXTRINSIC — Rigid3 { float R[9] row-major; float t[3] }, IDENTICAL shape
// and P_cam = R*P_src + t convention to 01.17's calibration output (cited).
// This project's extrinsic (kTCameraLidar below) is a FIXED, already-solved
// constant — an illustrative roof-LiDAR-above-windshield-camera rig, the
// kind of number 01.17's optimizer would hand you. See THEORY.md "The math"
// for the full derivation of the rotation (a clean axis permutation between
// the LiDAR's x-forward/y-left/z-up sensor frame and the camera's optical
// frame) and PRACTICE.md §1 for the physical mounting story.
//
// UINT-ENCODED ATOMIC MIN (the z-buffer's concurrency trick) — CUDA has no
// atomicMin for float. The standard fix reinterprets each float's IEEE-754
// bit pattern as an unsigned int and runs atomicMin on THAT: for two
// positive, finite floats, larger value implies larger bit pattern (the
// exponent lives in the high bits and dominates the comparison), so integer
// ordering matches float ordering exactly with NO transformation needed.
// (The fully general version — floats that may be negative — needs one
// extra step: flip all bits if the sign bit is set, else flip only the sign
// bit, so negative floats sort as smaller integers too; every LiDAR depth
// in this project is a positive number in front of the camera, so the
// simple positive-only case is what kernels.cu actually uses — see its
// encode_depth_for_zbuffer() for both branches, documented for completness.)
//
// DIFFUSION STABILITY (a real numerics gate, not just a chosen constant) —
// the explicit (forward-Euler) update below is a weighted average of a
// pixel's four neighbors; THEORY.md "The math" derives the CFL-style bound
// dt <= 1 / (sum of the four neighbor conductances), and because Perona-
// Malik conductance is bounded in (0, 1], the WORST case is dt <= 0.25.
// kDiffusionDt is checked against this bound at runtime in main.cu (the
// "STABILITY" gate) — the safety margin is a compile-time constant here,
// not something the demo merely asserts and hopes.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t — the z-buffer's atomic-min encoding word

// ===========================================================================
// Image / camera constants — the ONE camera every stage of the pipeline
// agrees on. 160x120 matches 01.17's teaching camera exactly (same fx/fy/
// cx/cy too) so a learner who studied that project recognizes the numbers;
// it also keeps the committed PPM/PGM samples kilobyte-sized (CLAUDE.md §8).
// ===========================================================================
constexpr int   kImageWidth  = 160;   // px
constexpr int   kImageHeight = 120;   // px
constexpr int   kImagePixels = kImageWidth * kImageHeight;
constexpr float kFx = 154.0f;         // px focal length x (~56.5 deg horizontal FOV at this width)
constexpr float kFy = 152.0f;         // px focal length y (~44.7 deg vertical FOV at this height)
constexpr float kCx = 80.0f;          // px principal point x (image-center-ish)
constexpr float kCy = 60.0f;          // px principal point y

// A point at exactly this range or beyond (or one whose ray never hits the
// scene, i.e. "sky") is treated as OUT OF RANGE — real LiDARs have a finite
// maximum range set by eye-safety-limited transmit power and receiver
// sensitivity (THEORY.md "The problem" derives the eye-safety link budget).
constexpr float kMaxDepthM = 20.0f;   // m

// Sentinel meaning "no value here" for BOTH the sparse depth map (no LiDAR
// return landed on this pixel) and the ground-truth depth map (this pixel's
// camera ray hit nothing — sky). Depth is physically always positive, so a
// negative sentinel can never be confused with a real reading.
constexpr float kInvalidDepth = -1.0f;

// ---------------------------------------------------------------------------
// Rigid3 — a rigid-body transform, IDENTICAL shape to 01.17's calibration
// output: P_dst = R * P_src + t, R stored ROW-MAJOR (R[r*3+c]). A pure
// data-layout struct (no member functions) — safe to include from both
// nvcc (.cu) and cl.exe (reference_cpu.cpp) translation units with no
// __host__/__device__ decoration needed.
// ---------------------------------------------------------------------------
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation: R[0..2]=row0, R[3..5]=row1, R[6..8]=row2
    float t[3];   // translation, meters, in the DESTINATION frame
};

// kTCameraLidar — P_camera = R * P_lidar + t. Derived in THEORY.md "The
// math" from a roof-mounted LiDAR (x-forward, y-left, z-up sensor frame,
// the same convention the vehicle BODY frame uses — SYSTEM_DESIGN.md
// interface conventions) sitting 0.30 m above and 0.05 m behind a
// windshield-height camera with NO relative tilt: the rotation is a clean
// axis PERMUTATION (camera-z = lidar-x, camera-x = -lidar-y,
// camera-y = -lidar-z), not a general Rodrigues rotation — the simplest
// non-trivial extrinsic there is, and a deliberately readable one to learn
// the projection formula on before facing 01.17's general-R case.
constexpr Rigid3 kTCameraLidar = {
    { 0.0f, -1.0f,  0.0f,
      0.0f,  0.0f, -1.0f,
      1.0f,  0.0f,  0.0f },
    { 0.0f, -0.30f, -0.05f }
};

// ---------------------------------------------------------------------------
// LidarPointF — one LiDAR return, IN THE LIDAR'S OWN FRAME (meters, the
// x-forward/y-left/z-up sensor convention kTCameraLidar's derivation
// assumes). This is exactly what data/sample/lidar_points.csv stores and
// exactly what project_zbuffer_kernel/_cpu consume — the single point where
// "what a LiDAR point IS" is defined for every stage downstream.
// ---------------------------------------------------------------------------
struct LidarPointF {
    float x, y, z;   // meters, LiDAR frame (see above)
};

// ===========================================================================
// Densification hyperparameters — shared by the GPU launcher and the CPU
// twin so "how many iterations", "how strong is the noise floor", etc. can
// never silently drift between the two paths (CLAUDE.md §12: state layouts
// and shared constants are single-sourced).
// ===========================================================================

// IDW baseline: a FIXED-RADIUS window search (not a true k-nearest-neighbor
// search — THEORY.md "The algorithm" names this simplification explicitly).
// At the committed sample's ~5% LiDAR density, average nearest-neighbor
// spacing is ~4-5 px, so a 16 px radius reliably captures several samples.
constexpr int   kIdwRadiusPx = 16;     // px, window half-width (Chebyshev)
constexpr float kIdwPower    = 2.0f;   // IDW exponent: weight = 1 / dist^power

// Anisotropic diffusion: explicit (forward-Euler) update, Perona-Malik
// conductance g = exp(-(grad/K)^2) gating each of the 4 axis-aligned edges.
// kConductanceK is in NORMALIZED luminance units (grayscale scaled to
// [0,1]) — see THEORY.md "Numerical considerations" for how this value was
// chosen relative to the scene's designed RGB contrasts.
constexpr float kConductanceK   = 0.12f;   // normalized-luminance gradient scale
constexpr float kDiffusionDt    = 0.20f;   // forward-Euler step (stability bound: dt <= 0.25, see file header)
constexpr int   kDiffusionIters = 1400;    // iteration count (README/THEORY document the convergence measurement)

// A REAL numerics gate, not merely an assertion in prose: this is the exact
// CFL-style bound the file header derives (dt <= 1 / sum-of-neighbor-
// conductances, worst case 4 neighbors at conductance 1.0 each -> 0.25). If
// a future edit ever raises kDiffusionDt past this bound, the BUILD fails
// here, at the one place that can prove it, instead of the demo silently
// producing an oscillating/diverging depth field at runtime.
static_assert(kDiffusionDt <= 0.25f,
             "kDiffusionDt exceeds the forward-Euler stability bound dt <= 1/4 "
             "(THEORY.md 'The math' derives this from the discrete Laplacian's "
             "worst-case neighbor conductance sum) -- lower kDiffusionDt or "
             "prove a tighter bound before changing this.");

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// project_zbuffer_kernel — scatter N LiDAR points into the WxH image with a
// nearest-wins z-buffer. One thread per POINT (a scatter, not the usual
// per-pixel map/stencil — see kernels.cu for the full mapping discussion).
// d_pts: [n_pts] device array, LIDAR-frame points. d_encoded: [kImagePixels]
// device array the CALLER must cudaMemset to 0xFF bytes (= UINT32_MAX, the
// "empty pixel" sentinel) before the launch; updated via atomicMin using
// the encoding scheme documented in the file header. t_camera_lidar is
// passed BY VALUE (not read from a global) — a `constexpr` struct declared
// at namespace scope has no device-accessible STORAGE unless additionally
// marked `__device__`/`__constant__`; passing the (tiny, 48-byte) extrinsic
// as an ordinary kernel argument sidesteps that entirely and is the same
// by-value convention 01.17's kernels use for their own SE(3) estimate
// (cited in this header's file comment) — the value lives in the kernel's
// parameter memory, which every thread can read regardless of where the
// caller's copy originally lived.
__global__ void project_zbuffer_kernel(const LidarPointF* __restrict__ d_pts,
                                       int n_pts,
                                       Rigid3 t_camera_lidar,
                                       uint32_t* __restrict__ d_encoded);

// compute_conductance_kernel — one thread per pixel (a map/stencil hybrid:
// each thread reads its 2 forward neighbors). d_rgb: [3*kImagePixels]
// PLANAR (not interleaved) normalized-[0,1] color image — d_rgb[0..N) is
// the red plane, d_rgb[N..2N) green, d_rgb[2N..3N) blue (N=kImagePixels).
// Full COLOR (not grayscale-luminance) conductance is a deliberate choice:
// two surfaces can differ strongly in hue while landing at nearly the same
// luminance (this project's own near_box-vs-ground pair does, by
// measurement — THEORY.md "Numerical considerations" shows the numbers) —
// collapsing to grayscale first would blind the conductance gate to those
// edges entirely, which is a worse, ACCIDENTAL version of the camo-edge
// failure mode this project studies on purpose. The edge weight is the
// MAX absolute per-channel difference (not the Euclidean color distance):
// a strong step in any one channel is exactly as informative as a strong
// step in all three for "is this probably an object boundary", and max()
// is one comparison cheaper than a square-root per edge per iteration.
// d_g_right/d_g_down OUT: [kImagePixels] Perona-Malik conductance gating
// the edge from (x,y) to its right/below neighbor respectively (border
// pixels write conductance 0 for the neighbor that would fall outside the
// image — THEORY.md discusses why that is the correct, not merely
// convenient, boundary condition).
__global__ void compute_conductance_kernel(const float* __restrict__ d_rgb,
                                           float* __restrict__ d_g_right,
                                           float* __restrict__ d_g_down);

// diffusion_step_kernel — ONE forward-Euler iteration, ping-pong style (the
// 07.09 jump-flooding precedent, cited in kernels.cu): reads d_in, writes
// d_out — never in place, so every thread in the pass sees a consistent
// snapshot of the previous iteration (Jacobi update, not Gauss-Seidel).
// d_anchor: [kImagePixels] sparse depth (kInvalidDepth where absent) — the
// Dirichlet boundary condition, reasserted unconditionally at every pixel
// that has one, every single iteration.
__global__ void diffusion_step_kernel(const float* __restrict__ d_in,
                                      const float* __restrict__ d_g_right,
                                      const float* __restrict__ d_g_down,
                                      const float* __restrict__ d_anchor,
                                      float* __restrict__ d_out);

// idw_kernel — one thread per output pixel; each does its own fixed-radius
// window search (kIdwRadiusPx) over d_sparse for valid samples. d_sparse,
// d_out: [kImagePixels]. Pixels with NO sample in the window fall back to
// d_sparse's own value if present, else 0 (documented, honest corner case —
// see kernels.cu).
__global__ void idw_kernel(const float* __restrict__ d_sparse,
                           float* __restrict__ d_out);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launchers — declared OUTSIDE the __CUDACC__ fence so any translation
// unit may call them; only their DEFINITIONS (kernels.cu) need nvcc. Each
// owns its kernel's grid/block math, the mandatory post-launch error check,
// and (for launch_diffusion) the WHOLE iterate-and-ping-pong unit of work —
// the same "launcher owns the schedule" shape 07.09's launch_jfa uses.
// ---------------------------------------------------------------------------

// launch_project_zbuffer — projects d_pts (DEVICE array, n_pts LiDAR-frame
// points) into d_encoded (DEVICE array, kImagePixels uint32_t, ALREADY
// memset to 0xFF by the caller). After the call, d_encoded holds, per
// pixel, either UINT32_MAX (no point landed there) or the winning point's
// encoded depth — decode with the same bit-reinterpret used to encode it
// (main.cu does this decode; it is a trivial, non-algorithmic step, see
// that file).
void launch_project_zbuffer(const LidarPointF* d_pts, int n_pts, uint32_t* d_encoded);

// launch_compute_conductance — d_rgb: DEVICE [3*kImagePixels] planar
// normalized-[0,1] color (see compute_conductance_kernel's doc-comment for
// the layout and why full color, not grayscale). d_g_right, d_g_down:
// DEVICE [kImagePixels] OUT. Exposed as its own launcher (not just an
// internal step of launch_diffusion) so main.cu can VERIFY it against
// compute_conductance_cpu independently — the texture-trap and camo-edge
// gates hinge entirely on this formula being correct, so it earns its own
// twin check (README/THEORY "How we verify correctness").
void launch_compute_conductance(const float* d_rgb, float* d_g_right, float* d_g_down);

// launch_diffusion — runs the FULL kDiffusionIters-step anisotropic
// diffusion densification and leaves the result in d_out (DEVICE array,
// kImagePixels). d_sparse: DEVICE array, kImagePixels, the Dirichlet
// anchors (kInvalidDepth elsewhere). d_rgb: DEVICE array, [3*kImagePixels]
// planar normalized-[0,1] color guidance image (same layout as
// compute_conductance_kernel). unknown_seed: the value every
// UNKNOWN (non-anchor) pixel starts the PDE at — THEORY.md "Numerical
// considerations" explains why this must be a plausible depth (the mean of
// the valid sparse samples, computed by the caller who already holds the
// sparse array on the host) rather than an out-of-range sentinel: a region
// whose conductance gate is closed on every side (the checkerboard texture
// trap is the designed example) can only reach a Dirichlet anchor if one
// falls inside it, and any pixel that never reaches one keeps its SEED
// value for the whole run — a sentinel there is an obviously-wrong answer;
// the mean is an honest "we have no local evidence" fallback. Internally
// allocates and frees its own conductance and ping-pong scratch buffers —
// the caller supplies only inputs, the seed, and the final-result buffer
// (mirrors launch_jfa's contract).
void launch_diffusion(const float* d_sparse, const float* d_rgb, float unknown_seed, float* d_out);

// launch_idw — inverse-distance-weighted baseline densification. d_sparse,
// d_out: DEVICE arrays, kImagePixels.
void launch_idw(const float* d_sparse, float* d_out);

// ---------------------------------------------------------------------------
// CPU reference twins — the independent oracle for each stage (see this
// header's file comment and reference_cpu.cpp's file header for the
// sharing ruling). All operate on plain HOST arrays.
// ---------------------------------------------------------------------------

// project_zbuffer_cpu — sequential nearest-wins z-buffer (no atomics
// needed: single-threaded, so a plain "is this closer than what's there"
// compare suffices — the GPU's atomicMin trick exists ONLY because many
// threads race on the same pixel; the CPU twin is the simple version that
// motivates why the GPU needs the trick at all). out: [kImagePixels],
// pre-filled with kInvalidDepth by this function.
void project_zbuffer_cpu(const LidarPointF* pts, int n_pts, float* out_depth);

// compute_conductance_cpu — independent host implementation of the same
// Perona-Malik formula compute_conductance_kernel computes. Signature
// mirrors the kernel's; used both directly (VERIFY stage) and internally
// by diffusion_densify_cpu.
void compute_conductance_cpu(const float* rgb, float* g_right, float* g_down);

// diffusion_densify_cpu — runs kDiffusionIters forward-Euler steps
// sequentially (its own ping-pong pair, allocated internally) and writes
// the final field to out. Independent from launch_diffusion end to end.
// unknown_seed: same meaning and same value the caller passes to
// launch_diffusion for this input — the twin comparison in main.cu's
// VERIFY stage requires both paths start from an IDENTICAL initial field.
void diffusion_densify_cpu(const float* sparse, const float* rgb, float unknown_seed, float* out);

// idw_densify_cpu — independent host implementation of the fixed-radius
// IDW baseline.
void idw_densify_cpu(const float* sparse, float* out);

#endif // PROJECT_KERNELS_CUH
