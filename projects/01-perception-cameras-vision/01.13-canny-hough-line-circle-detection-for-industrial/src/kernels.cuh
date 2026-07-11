// ===========================================================================
// kernels.cuh — single-sourced data contracts + kernel/reference declarations
//               for project 01.13 (Canny + Hough line/circle detection for
//               industrial alignment)
//
// Role in the project
// --------------------
// This header is the ONE place that defines the image geometry, the
// synthetic part's nominal geometry, and every accumulator layout used by
// both the GPU path (kernels.cu), the CPU oracle (reference_cpu.cpp), and
// the orchestration/analysis code (main.cu). Per the template's independence
// ruling (see reference_cpu.cpp's file header): data-layout contracts are
// single-sourced and shared; the ALGORITHMIC CORE of each stage is written
// twice, independently.
//
// The part-geometry constants below (PART_HALF_W/H, HOLE_LOCAL_*,
// HOLE_RADIUS, EDGE_THETA0/RHO0_LOCAL) also appear, duplicated WITH A
// COMMENT POINTING HERE, in scripts/make_synthetic.py — Python cannot
// #include a C++ header, so that is the one place in this project where
// deliberate, documented duplication (CLAUDE.md §4 self-containment spirit)
// is unavoidable. If you change a number here, change it there too.
//
// Why ".cuh"? See docs/PROJECT_TEMPLATE/src/kernels.cuh's header — the short
// version: __global__ declarations are fenced behind #ifdef __CUDACC__ so
// this same file can be #included by reference_cpu.cpp, which cl.exe (not
// nvcc) compiles.
//
// Read this after: README.md/THEORY.md (the algorithm walkthrough). Read
// this before: kernels.cu, reference_cpu.cpp, main.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>

// ===========================================================================
// SECTION 1 — image geometry
// ===========================================================================

// 320x240 chosen per the project brief: big enough that Canny/Hough teach
// their real O(edge_pixels * theta_bins) cost, small enough that the whole
// pipeline (including the O(W*H*180) Hough line sweep, ~13.8M inner-loop
// iterations at worst) runs in milliseconds on a desktop GPU and seconds on
// one CPU core (see THEORY.md "The algorithm" for the exact operation count).
static constexpr int IMG_W = 320;
static constexpr int IMG_H = 240;
static constexpr int IMG_PIXELS = IMG_W * IMG_H;

// Pixel index convention used EVERYWHERE in this project: row-major,
// idx = y * IMG_W + x, x in [0, IMG_W), y in [0, IMG_H). Image-plane
// convention: +x right, +y DOWN (standard image/raster convention, not a
// right-handed 3-D robot frame — there is no 3-D pose here, only a 2-D
// in-plane alignment, so CLAUDE.md §12's T_parent_child/quaternion
// conventions do not apply; the one frame-like quantity, the 2-D rigid
// transform (dx, dy, dtheta), is defined precisely in SECTION 3 below.
static inline int pixel_index(int x, int y) { return y * IMG_W + x; }

// A handful of named constants used throughout: PI, and the fact that Hough
// line angles are periodic with period PI (a line and the "same line rotated
// 180 degrees" are identical), not 2*PI (see THEORY.md "The math").
static constexpr float PI_F = 3.14159265358979323846f;
static constexpr float HALF_PI_F = 1.57079632679489661923f;

// ===========================================================================
// SECTION 2 — the synthetic part's NOMINAL geometry (local frame)
// ===========================================================================
// The scene is a rectangular machined plate (like a laser-cut mounting
// bracket) with 4 straight edges and 3 drilled alignment/mounting holes of
// KNOWN, DISTINCT nominal radii — exactly the "known geometry" that lets an
// industrial vision station search a small parameter space instead of a
// generic 3-D Hough circle transform (THEORY.md "The algorithm").
//
// All positions below are in the plate's own LOCAL frame: origin at the
// plate's geometric center, +x right, +y down (matching image convention so
// the local->image transform in SECTION 3 is a pure rotate+translate, no
// axis flip to track). Units: pixels (this teaching project does not model
// a camera's intrinsics/mm-per-pixel calibration — see README "Limitations"
// and PRACTICE.md §3 for how a real station gets mm out of this).

static constexpr float PART_HALF_W = 70.0f;   // plate half-width,  local +/-x extent (px)
static constexpr float PART_HALF_H = 45.0f;   // plate half-height, local +/-y extent (px)

static constexpr int NUM_HOLES = 3;
// Hole i has KNOWN nominal radius HOLE_RADIUS[i] — this is the whole trick
// that tames the Hough circle transform from a 3-D (cx,cy,r) search down to
// NUM_HOLES independent 2-D (cx,cy) searches (THEORY.md). Radii are
// distinct on purpose: it lets circle_recovery match "the hole with radius
// 8" unambiguously, exactly like a real work-order that specifies which
// tapped hole is which.
static constexpr float HOLE_LOCAL_X[NUM_HOLES] = {  45.0f, -40.0f,  5.0f };  // px, local frame
static constexpr float HOLE_LOCAL_Y[NUM_HOLES] = { -15.0f, -20.0f, 30.0f };  // px, local frame
static constexpr float HOLE_RADIUS[NUM_HOLES]  = {   6.0f,   8.0f, 10.0f };  // px, KNOWN nominal radii

// The 4 edges as NOMINAL (local-frame) lines, stored the way the Hough
// accumulator will express them once transformed into the image: a line is
// { theta in [0, PI), rho } with the point-normal form  x*cos(theta) +
// y*sin(theta) = rho  (THEORY.md "The math" derives this from point-line
// duality). theta0/rho0_local below are the line's parameters when
// expressed relative to the PLATE'S OWN LOCAL ORIGIN (before the rigid
// transform in SECTION 3 is applied) — i.e. what you would measure if the
// plate's center were the image center and it were unrotated.
// Order: left, right, top, bottom.
static constexpr int NUM_EDGES = 4;
static constexpr float EDGE_THETA0[NUM_EDGES]      = { 0.0f, 0.0f, HALF_PI_F, HALF_PI_F };
static constexpr float EDGE_RHO0_LOCAL[NUM_EDGES]  = { -PART_HALF_W, PART_HALF_W, -PART_HALF_H, PART_HALF_H };
// (left: theta=0 normal points +x, signed distance -HALF_W from local
//  origin; right: same normal, +HALF_W; top: theta=PI/2 normal points +y
//  [downward, since +y is down], signed distance -HALF_H (top is the
//  SMALLER-y edge); bottom: +HALF_H. Cross-checked against the transform in
//  SECTION 3 and against scripts/make_synthetic.py's rasterizer — see
//  THEORY.md "How we verify correctness" for the worked numeric example.)

// The engineered "weak-but-connected" scratch mark used for the hysteresis
// lesson (README/THEORY "single- vs double-threshold"): a shallow, low-
// contrast scribe line from the midpoint of the TOP edge straight into the
// plate interior. It is 8-connected to the top edge's own strong Canny
// pixels, so double-threshold hysteresis should recover it by propagation
// while a single high threshold, applied uniformly, should not (its own
// gradient magnitude is deliberately below T_HIGH everywhere). Endpoints in
// the plate's local frame; scripts/make_synthetic.py rasterizes it and
// records its transformed image-frame endpoints in data/sample/truth.csv
// for the gate to check directly (no re-derivation needed at verify time).
static constexpr float SCRATCH_LOCAL_X0 = 0.0f;
static constexpr float SCRATCH_LOCAL_Y0 = -PART_HALF_H;         // starts ON the top edge
static constexpr float SCRATCH_LOCAL_X1 = 0.0f;
static constexpr float SCRATCH_LOCAL_Y1 = -PART_HALF_H + 25.0f; // 25 px into the interior

// ===========================================================================
// SECTION 3 — the applied rigid transform (the "truth" a real line reports)
// ===========================================================================
// The synthetic scene renders the plate under an in-plane offset+rotation
// applied about the IMAGE center (IMG_CX, IMG_CY) — i.e. when
// (dx, dy, dtheta) = (0, 0, 0), the plate's local origin sits exactly at the
// image center. image_point = R(dtheta) * local_point + (IMG_CX, IMG_CY) +
// (dx, dy). This is exactly the transform an industrial cell's fixture
// error, conveyor jitter, or robot pick offset would apply to a nominally-
// centered part; recovering (dx, dy, dtheta) from the image IS the alignment
// measurement (README "The industrial story", THEORY.md "The math").
static constexpr float IMG_CX = 160.0f;  // px, image x of the plate's NOMINAL (untransformed) center
static constexpr float IMG_CY = 120.0f;  // px, image y of the plate's NOMINAL (untransformed) center

// ===========================================================================
// SECTION 4 — Canny stage constants
// ===========================================================================

// 5-tap binomial approximation to a Gaussian, sigma ~= 1.0 px (Pascal's
// triangle row 4, normalized: 1,4,6,4,1 sums to 16). Chosen over a "true"
// continuous Gaussian sample for two reasons: (1) power-of-two normalization
// (divide by 16, a bit-shift) keeps GPU/CPU rounding identical bit-for-bit
// in more cases than an arbitrary sigma would, and (2) it is the exact
// textbook Canny smoothing kernel — see THEORY.md "The math" for the sigma
// derivation and why a wider kernel would blur the 6-10 px hole radii away.
// NOTE on the __device__ guard: a plain `constexpr` ARRAY (unlike a scalar
// constexpr like PI_F above) is not reliably usable when INDEXED BY A
// RUNTIME VALUE inside device code on this toolchain — nvcc needs an actual
// device-memory home for it, hence the explicit __device__ qualifier. The
// #ifdef keeps this legal for cl.exe too (which compiles this same header
// via reference_cpu.cpp and does not know the __device__ keyword at all).
#ifdef __CUDACC__
__device__
#endif
static constexpr float GAUSS_WEIGHTS[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };
static constexpr int GAUSS_RADIUS = 2; // 5 taps = 2*radius+1

// Sobel 3x3 stencil: Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = Gx^T. Each stencil's
// POSITIVE-side weights sum to 1+2+1 = 4 — an UNNORMALIZED convolution sum
// therefore reports a gradient 4x too large relative to true intensity-per-
// pixel units. This is the exact lesson project 01.03 root-caused for its
// Scharr stencil (there: weights sum to 16, scale factor 1/32 because the
// positive-side sum there is 16 folded through additional structure-tensor
// scaling); the general rule this project restates: ALWAYS DIVIDE BY THE
// STENCIL'S POSITIVE-WEIGHT SUM, and SAY SO in the code, or every downstream
// threshold (T_LOW/T_HIGH below) silently means "4x the number you think it
// does." See kernels.cu's sobel_gradient_kernel for the applied division.
static constexpr float SOBEL_SCALE = 1.0f / 4.0f;

// Double-threshold hysteresis bounds, on the SOBEL-SCALED gradient magnitude
// (0..~360 range for an 8-bit image: max |gx|,|gy| ~= 255, magnitude up to
// ~255*sqrt(2) ~= 360, before Gaussian smoothing softens real edges well
// below that ceiling). Values below were MEASURED on this project's actual
// synthetic scene (see THEORY.md "How we verify correctness" for the
// histogram this was tuned against) — not guessed: strong plate/hole
// boundaries after blur land at roughly 90-180, the deliberately weak
// scratch mark at roughly 28-45, and background brushed-metal texture noise
// stays under 12.
static constexpr float CANNY_T_LOW = 20.0f;
static constexpr float CANNY_T_HIGH = 55.0f;

// Hard cap on hysteresis sweep count — a safety net, not a tuning knob: the
// promotion fixed point is reached in at most (image diagonal) sweeps in the
// worst case (one pixel promoted per sweep along a maximal weak chain); 400
// is generous headroom over the ~15-25 sweeps this scene actually needs
// (measured, printed as an [info] line by main.cu every run).
static constexpr int HYSTERESIS_MAX_SWEEPS = 400;

// edge_state values written by classify_threshold_kernel / promoted by
// hysteresis_propagate_sweep_kernel. Plain named constants (not an enum
// class) because both nvcc (device code) and cl.exe (host code, via this
// shared header) need to read/write them as a plain unsigned char.
static constexpr unsigned char EDGE_NONE = 0;
static constexpr unsigned char EDGE_WEAK = 1;
static constexpr unsigned char EDGE_STRONG = 2;

// ===========================================================================
// SECTION 5 — Hough LINE accumulator layout
// ===========================================================================
// Point-line duality (THEORY.md derives this): every line is
// x*cos(theta) + y*sin(theta) = rho, theta in [0, PI) (a line and its
// theta+PI counterpart are the identical line — only the SIGN of rho would
// flip — so restricting theta to a half-turn avoids double-counting every
// line twice), rho in [-RHO_MAX, +RHO_MAX] where RHO_MAX bounds the largest
// distance any image pixel can have from the origin.
static constexpr int HOUGH_THETA_BINS = 180;                  // 1 degree per bin
static constexpr float HOUGH_THETA_STEP = PI_F / static_cast<float>(HOUGH_THETA_BINS);
// ceil(sqrt(320^2 + 240^2)) = ceil(400.0) = 400 px — the image diagonal,
// the largest |rho| any line through the image can have.
static constexpr int HOUGH_RHO_MAX = 400;
static constexpr int HOUGH_RHO_BINS = 2 * HOUGH_RHO_MAX + 1;  // 801: rho_bin = round(rho) + RHO_MAX
static constexpr long long HOUGH_LINE_ACCUM_CELLS =
    static_cast<long long>(HOUGH_THETA_BINS) * static_cast<long long>(HOUGH_RHO_BINS); // 144,180 int cells

// ---- The FIXED-POINT theta table: what makes the line accumulator BIT-EXACT
// ----------------------------------------------------------------------------
// The headline determinism claim of this project ("integer atomics make the
// accumulator order-independent") only survives contact with reality if the
// VOTE ADDRESS itself — the (theta_bin, rho_bin) pair each edge pixel writes
// to — is ALSO bit-identical on GPU and CPU. Plain floating-point cosf/sinf
// evaluated separately by nvcc (device) and cl.exe (host) are NOT guaranteed
// to round identically (the same 1-ULP FMA-contraction story as the SAXPY
// placeholder's tolerance note) — a rho value landing 1 ULP from a .5
// boundary could round to a DIFFERENT integer bin on the two paths, silently
// breaking bit-exactness for that one vote.
//
// The fix: quantize cos(theta)/sin(theta) to Q16 FIXED-POINT integers ONCE,
// on the host, and hand the IDENTICAL int32 table to both paths (GPU via
// upload_hough_constants() -> __constant__ memory, CPU as a plain array
// argument to hough_lines_accum_cpu). From that point on, the vote address
// is computed with PURE INTEGER ARITHMETIC (multiply, add, integer round via
// bias-and-shift) — operations the IEEE/C++ standards specify EXACTLY, with
// no rounding-mode or contraction ambiguity left at all. This table is a
// SHARED DATA-LAYOUT CONTRACT under the reference_cpu.cpp independence
// ruling (constants, not algorithm) — the vote-SCATTER loop that USES it
// remains independently written in kernels.cu vs reference_cpu.cpp.
static constexpr int HOUGH_FIXED_SHIFT = 16;                      // Q16: 16 fractional bits
static constexpr int32_t HOUGH_FIXED_SCALE = 1 << HOUGH_FIXED_SHIFT;  // 65536

// build_hough_theta_table_fixed — fills cos_fixed[t]/sin_fixed[t] =
// round(cos/sin(t * THETA_STEP) * 65536) for t in [0, HOUGH_THETA_BINS),
// using double precision (plenty of headroom below float rounding) so the
// ONE table used everywhere is as accurate as this teaching project needs.
// Called exactly once, on the host, by main.cu; the same int32_t arrays are
// then handed to BOTH launch paths — see the header note above.
static inline void build_hough_theta_table_fixed(int32_t* cos_fixed, int32_t* sin_fixed)
{
    for (int t = 0; t < HOUGH_THETA_BINS; ++t) {
        const double theta = static_cast<double>(t) * static_cast<double>(HOUGH_THETA_STEP);
        cos_fixed[t] = static_cast<int32_t>(std::lround(std::cos(theta) * static_cast<double>(HOUGH_FIXED_SCALE)));
        sin_fixed[t] = static_cast<int32_t>(std::lround(std::sin(theta) * static_cast<double>(HOUGH_FIXED_SCALE)));
    }
}

// Minimum vote count (out of up to ~150 edge pixels on the longer plate
// edges) for a Hough-space local maximum to be accepted as a real line
// candidate; MEASURED against this scene (see THEORY.md), well above the
// few-vote noise floor any short chain of unrelated edge pixels can create.
static constexpr int HOUGH_LINE_PEAK_MIN_VOTES = 30;

// ===========================================================================
// SECTION 6 — Hough CIRCLE accumulator layout
// ===========================================================================
// Because the 3 drilled holes have KNOWN, distinct nominal radii
// (HOLE_RADIUS above), the accumulator is NUM_HOLES independent 2-D (cx,cy)
// planes — not one 3-D (cx,cy,r) volume. THEORY.md "The algorithm" derives
// why the full 3-D transform is O(edge_pixels * W * H * R_range) and
// explodes, while this known-radius version is O(edge_pixels * NUM_HOLES).
static constexpr long long HOUGH_CIRCLE_ACCUM_CELLS =
    static_cast<long long>(NUM_HOLES) * static_cast<long long>(IMG_W) * static_cast<long long>(IMG_H);

// Minimum vote count for a circle-accumulator peak, measured AFTER
// main.cu's extract_circle_peaks() windowed-sum smoothing (see that
// function's comment for why raw single-cell votes undercount badly). A
// hole of radius r has circumference ~= 2*pi*r edge pixels (r=6 -> ~38,
// r=10 -> ~63); each votes at TWO candidate centers (both signs along its
// measured gradient, see kernels.cu) but only the correctly-signed one
// lands near the true center — so the windowed sum there recovers
// approximately ONE circumference's worth of votes, scattered by rounding
// across a handful of neighboring cells. MEASURED on this project's own
// scene: true-center windowed sums of 44/57/73 (for r=6/8/10) against a
// background noise floor (other edges' votes landing elsewhere in the same
// plane) of at most ~20 — this threshold sits with real margin on both sides.
static constexpr int HOUGH_CIRCLE_PEAK_MIN_VOTES = 30;

// ===========================================================================
// SECTION 7 — plain data structures shared by main.cu's analysis code
// ===========================================================================
// These are NOT part of the GPU/CPU twin comparison (see reference_cpu.cpp's
// independence ruling, and main.cu's "peak extraction and alignment are not
// twinned" note) — they are the shapes the verified accumulators get turned
// into by the single, host-only peak-extraction/alignment code in main.cu.

struct DetectedLine {
    float theta;   // radians, [0, PI)
    float rho;     // px, sub-bin-refined
    int votes;     // raw accumulator vote count at the (unrefined) peak bin
};

struct DetectedCircle {
    float cx, cy;  // px, sub-pixel-refined image-frame center
    float r;       // px, the KNOWN nominal radius of this accumulator plane
    int votes;
};

// Result of the small (a,b,tx,ty) rigid-registration least-squares solve
// (THEORY.md "The math"; cites project 33.01's batched small-matrix linalg
// as the production-scale version of the same normal-equations idea).
struct AlignmentResult {
    float dx, dy;     // px, recovered translation
    float dtheta;     // rad, recovered rotation
    bool solved;       // false if fewer than 2 correspondences were available
};

#ifdef __CUDACC__  // ================= device-aware section (nvcc only) ====

// ---- Stage 1: separable Gaussian blur (two passes) -------------------------
// img is the raw uint8 input; tmp/blurred are float buffers (kept float
// throughout the pipeline rather than re-quantizing to uint8 between stages
// — re-quantizing would throw away exactly the sub-level precision Sobel
// needs, see THEORY.md "Numerical considerations").
__global__ void gaussian_blur_h_kernel(const uint8_t* __restrict__ img, int W, int H,
                                       float* __restrict__ tmp);
__global__ void gaussian_blur_v_kernel(const float* __restrict__ tmp, int W, int H,
                                       float* __restrict__ blurred);

// ---- Stage 2: Sobel gradients ----------------------------------------------
__global__ void sobel_gradient_kernel(const float* __restrict__ blurred, int W, int H,
                                      float* __restrict__ gx, float* __restrict__ gy);

// ---- Stage 3: gradient-direction non-max suppression -----------------------
__global__ void nms_kernel(const float* __restrict__ gx, const float* __restrict__ gy,
                           int W, int H, float* __restrict__ suppressed_mag);

// ---- Stage 4: double-threshold classification (reused for the single-
//      threshold variant too — see kernels.cu and main.cu) ------------------
__global__ void classify_threshold_kernel(const float* __restrict__ suppressed_mag,
                                          int W, int H, float t_low, float t_high,
                                          unsigned char* __restrict__ state);

// ---- Stage 5: hysteresis promotion, ONE sweep (CCL-style iterative
//      propagation — see kernels.cu header, cites 01.06/30.01) --------------
__global__ void hysteresis_propagate_sweep_kernel(unsigned char* __restrict__ state,
                                                   int W, int H, int* __restrict__ changed);

// ---- Stage 6: state -> binary edge map (0 / 255) ----------------------------
__global__ void finalize_edge_map_kernel(const unsigned char* __restrict__ state,
                                         int W, int H, unsigned char* __restrict__ edge_map);

// ---- Stage 7: Hough line voting (integer atomics -> order-independent) -----
__global__ void hough_lines_vote_kernel(const unsigned char* __restrict__ edge_map,
                                        int W, int H, int* __restrict__ accum);

// ---- Stage 8: Hough circle voting (gradient-directed, known-radius) --------
__global__ void hough_circles_vote_kernel(const unsigned char* __restrict__ edge_map,
                                          const float* __restrict__ gx,
                                          const float* __restrict__ gy,
                                          int W, int H, int* __restrict__ accum);

#endif // __CUDACC__ ---------------------------------------------------------

// ---- Host launch wrappers (own grid/block math + post-launch error check,
//      callable from any translation unit — see kernels.cu for each launch
//      configuration's reasoning) -------------------------------------------
void launch_gaussian_blur(const uint8_t* d_img, int W, int H, float* d_tmp, float* d_blurred);
void launch_sobel_gradient(const float* d_blurred, int W, int H, float* d_gx, float* d_gy);
void launch_nms(const float* d_gx, const float* d_gy, int W, int H, float* d_suppressed_mag);
void launch_classify_threshold(const float* d_suppressed_mag, int W, int H,
                               float t_low, float t_high, unsigned char* d_state);
// Runs ONE sweep and returns whether any pixel changed (via a device->host
// copy of the single int flag) — main.cu owns the sweep-until-converged loop
// and the sweep counter, exactly mirroring 01.06's CCL convergence pattern.
bool launch_hysteresis_sweep(unsigned char* d_state, int W, int H);
void launch_finalize_edge_map(const unsigned char* d_state, int W, int H, unsigned char* d_edge_map);
// Uploads the fixed-point theta table (SECTION 5) and the known hole radii
// (SECTION 2) into __constant__ device memory. Call ONCE before the first
// launch_hough_lines_vote / launch_hough_circles_vote of a run.
void upload_hough_constants(const int32_t* cos_fixed, const int32_t* sin_fixed);
void launch_hough_lines_vote(const unsigned char* d_edge_map, int W, int H, int* d_accum);
void launch_hough_circles_vote(const unsigned char* d_edge_map, const float* d_gx, const float* d_gy,
                               int W, int H, int* d_accum);

// ---- CPU reference oracle (reference_cpu.cpp) — INDEPENDENT reimplementation
//      of every twinned stage; see that file's header for the independence
//      ruling and which stages are/are not twinned. -------------------------
void gaussian_blur_cpu(const uint8_t* img, int W, int H, float* blurred);
void sobel_gradient_cpu(const float* blurred, int W, int H, float* gx, float* gy);
void nms_cpu(const float* gx, const float* gy, int W, int H, float* suppressed_mag);
void classify_threshold_cpu(const float* suppressed_mag, int W, int H,
                            float t_low, float t_high, unsigned char* state);
// Reaches the SAME hysteresis fixed point as the GPU's sweep loop via a
// completely different algorithm (a queue-based flood fill) — the point of
// the twin (see THEORY.md "How we verify correctness" and reference_cpu.cpp).
void hysteresis_propagate_cpu(unsigned char* state, int W, int H);
void finalize_edge_map_cpu(const unsigned char* state, int W, int H, unsigned char* edge_map);
// cos_fixed/sin_fixed: the SAME Q16 table built by build_hough_theta_table_fixed
// and uploaded to the GPU by upload_hough_constants — see SECTION 5's note on
// why this table (data), not the accumulation loop (algorithm), is shared.
void hough_lines_accum_cpu(const unsigned char* edge_map, int W, int H,
                           const int32_t* cos_fixed, const int32_t* sin_fixed, int* accum);
void hough_circles_accum_cpu(const unsigned char* edge_map, const float* gx, const float* gy,
                             int W, int H, int* accum);

#endif // PROJECT_KERNELS_CUH
