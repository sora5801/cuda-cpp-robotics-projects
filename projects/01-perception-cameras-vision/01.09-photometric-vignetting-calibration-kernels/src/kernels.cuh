// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 01.09
//               (Photometric/vignetting calibration kernels)
//
// Role in the project
// --------------------
// This header is the SINGLE-SOURCED data-layout contract for the whole
// project (the twin-independence ruling in reference_cpu.cpp's file header:
// layouts and constants are shared, the algorithmic CORE is written twice).
// Every file in src/ agrees here on:
//
//   * the SENSOR MODEL — I(x,y) = g(x,y)*L(x,y) + o(x,y) + noise, where
//     g = V*PRNU (multiplicative: optical vignette x per-pixel gain
//     nonuniformity) and o = DSNU (additive fixed-pattern offset). See
//     scripts/make_synthetic.py's file header for the full derivation and
//     THEORY.md "The math" for the physics behind it. This project composes
//     with 01.08's radiometric story by ISOLATING the spatial terms: 01.08
//     taught the per-pixel-independent camera response function (CRF, a
//     1-D curve, code value -> exposure, identical at every pixel); this
//     project teaches the PER-PIXEL, spatially-varying multiplicative/
//     additive fields that sit BEFORE the CRF in a real imaging pipeline
//     (see README "System context" for the exact composition order).
//   * the STACK layout — dark/flat calibration frames are stored FRAME-
//     MAJOR: stack[f*kN + p] is pixel p of frame f. This is deliberate (see
//     kernels.cu's stack_mean_kernel header): for a FIXED frame f, adjacent
//     threads (adjacent pixels p) touch adjacent addresses, so every frame
//     iteration inside the per-pixel reduction loop is a fully coalesced
//     read — the natural, correct layout for "reduce across N images, one
//     thread per pixel".
//   * PROBLEM GEOMETRY — kW/kH/kN (160x120, matching 01.03/01.08's
//     pyramid-friendly precedent, though this project builds no pyramid),
//     kNumDarkFrames/kNumFlatFrames (16 each, MUST MATCH
//     scripts/make_synthetic.py's N_DARK/N_FLAT), the CENTER-NORMALIZATION
//     ROI (the industrial flat-field convention: pin the nonparametric gain
//     map's scale to ~1 in a small region near the image center — see
//     kernels.cu's normalize step), the RADIAL-BINNING geometry (bins the
//     nonparametric gain map by distance from the GEOMETRIC image center,
//     feeding both radial_profile.csv and the parametric least-squares
//     fit), and kGainFloor (the division-by-small-gain numerical guard the
//     correction kernel applies at extreme corners — see THEORY.md
//     "Numerical considerations").
//
// Why ".cuh"? (CLAUDE.md §12) — device-only declarations (__global__ kernel
// signatures) are fenced behind #ifdef __CUDACC__ so this header stays
// includable by reference_cpu.cpp, which cl.exe (not nvcc) compiles.
//
// Kernel inventory (each a distinct GPU pattern; see kernels.cu for the
// full essay on each):
//   1) stack_mean        — MAP-of-per-pixel-REDUCTIONS across N frames.
//   2) elementwise_sub    — pure MAP (a - b).
//   3) affine              — pure MAP (scale*a + offset); reused for the
//                           center-normalize step.
//   4) roi_mean_reduce      — block-tree REDUCE (shared memory) + one atomicAdd
//                           per block into a double accumulator, MASKED to a
//                           rectangular ROI.
//   5) radial_bin           — SCATTER-REDUCE: every pixel atomicAdd's into ONE
//                           of a small number of global radial-distance bins
//                           (a "histogram" pattern, contrasted with #4's
//                           shared-memory-then-one-atomic-per-block pattern).
//   6) correction            — the "reason to exist" MAP: (I - o) / max(g, floor).
//
// Plus ONE shared, HOST-ONLY least-squares solver (SECTION 5, the 01.08
// crf_solve_debevec precedent) fitting the parametric radial vignette model
// V(r) = 1 + a2*r^2 + a4*r^4 + a6*r^6 to the (binned) nonparametric gain map.
//
// Read this after: main.cu.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ===========================================================================
// SECTION 1 — problem geometry (single-sourced; MUST MATCH
// scripts/make_synthetic.py's W/H/N_DARK/N_FLAT — see that script's module
// header for the cross-reference discipline).
// ===========================================================================

// Image dimensions in pixels. 160x120 matches 01.03/01.08's precedent
// (chosen there for pyramid-friendly halving; kept here purely for
// consistency across the domain and because it is a comfortably small,
// fast-to-process, easy-to-visualize frame size for a teaching demo).
constexpr int kW = 160;
constexpr int kH = 120;
constexpr int kN = kW * kH;   // total pixel count per frame, used everywhere as the flat element count

// Calibration stack sizes — "average N=16 dark frames" / "average N=16
// flat-field frames" (task brief). Also the N values the noise_averaging
// gate re-slices as N=1/4/16 subsets (main.cu).
constexpr int kNumDarkFrames = 16;
constexpr int kNumFlatFrames = 16;

// Center-normalization ROI (pixel-space, half-open [x0,x1) x [y0,y1)):
// pins the nonparametric gain map's scale via the mean of the DARK-
// SUBTRACTED flat stack over a small box near the GEOMETRIC image center
// (80, 60) — NOT the true optical center (83, 58): a real calibration
// pipeline does not know the lens's decentering in advance, so using the
// geometric center is the honest, deployable choice (see README
// "Limitations & honesty" and THEORY.md "The algorithm" for the
// parametric-fit consequence of this same scoping decision). 8x8=64
// pixels — small enough that "near the center" is true, large enough that
// per-frame noise averages down well below the +-2% PRNU signal.
constexpr int kCenterRoiX0 = 76, kCenterRoiX1 = 84;
constexpr int kCenterRoiY0 = 56, kCenterRoiY1 = 64;

// Radial-binning geometry for the nonparametric gain map, feeding both
// radial_profile.csv and the parametric least-squares fit (SECTION 5).
// kNumRadialBins * kRadialBinWidthPx = 110 px covers every pixel's distance
// from the geometric center: the farthest image corner from (80,60) is
// sqrt(80^2+60^2) = 100 px, so 110 px leaves comfortable headroom.
constexpr int kNumRadialBins = 44;
constexpr float kRadialBinWidthPx = 2.5f;

// Correction kernel's division-by-small-gain numerical floor (THEORY.md
// "Numerical considerations" derives why this is needed in general, even
// though this project's synthetic gain never approaches it — see main.cu's
// [info] line reporting the actual measured minimum recovered gain).
constexpr float kGainFloor = 0.05f;

// ===========================================================================
// SECTION 2 — device-only declarations (nvcc only; see the file header for
// why this fence exists). Each kernel below carries its one-line summary
// here; the full essay (thread mapping, memory behavior, numerics) lives
// with the DEFINITION in kernels.cu (CLAUDE.md §6.1).
// ===========================================================================
#ifdef __CUDACC__

// ---- 1) per-pixel mean across a frame-major stack of N frames -----------
__global__ void stack_mean_kernel(const float* __restrict__ stack,
                                  int numFrames, int n,
                                  float* __restrict__ out_mean);

// ---- 2) generic elementwise combine: out = a - b -------------------------
__global__ void elementwise_sub_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       int n, float* __restrict__ out);

// ---- 3) generic affine MAP: out = scale*in + offset ----------------------
__global__ void affine_kernel(const float* __restrict__ in, int n,
                              float scale, float offset,
                              float* __restrict__ out);

// ---- 4) masked ROI-mean REDUCE: sums pixels inside [x0,x1)x[y0,y1) -------
__global__ void roi_mean_reduce_kernel(const float* __restrict__ img,
                                       int W, int H,
                                       int x0, int x1, int y0, int y1,
                                       double* __restrict__ d_sum_accum);

// ---- 5) radial-distance SCATTER-REDUCE (histogram-style atomics) --------
__global__ void radial_bin_kernel(const float* __restrict__ gain, int W, int H,
                                  float cx, float cy,
                                  int numBins, float binWidthPx,
                                  float* __restrict__ d_bin_sum,
                                  int*   __restrict__ d_bin_count);

// ---- 6) the correction MAP: out = (I - dsnu) / max(gain, floor) ----------
__global__ void correction_kernel(const float* __restrict__ I,
                                  const float* __restrict__ dsnu,
                                  const float* __restrict__ gain,
                                  int n, float gainFloor,
                                  float* __restrict__ out);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// SECTION 3 — host-callable launch wrappers (visible to every translation
// unit; only their DEFINITIONS in kernels.cu require nvcc). Each owns its
// grid/block math and the mandatory post-launch error check.
// ===========================================================================
void launch_stack_mean(const float* d_stack, int numFrames, int n, float* d_out_mean);
void launch_elementwise_sub(const float* d_a, const float* d_b, int n, float* d_out);
void launch_affine(const float* d_in, int n, float scale, float offset, float* d_out);
void launch_roi_mean_reduce(const float* d_img, int W, int H,
                            int x0, int x1, int y0, int y1,
                            double* d_sum_accum);
void launch_radial_bin(const float* d_gain, int W, int H, float cx, float cy,
                       int numBins, float binWidthPx,
                       float* d_bin_sum, int* d_bin_count);
void launch_correction(const float* d_I, const float* d_dsnu, const float* d_gain,
                       int n, float gainFloor, float* d_out);

// ===========================================================================
// SECTION 4 — the CPU reference oracle (defined in reference_cpu.cpp).
// Declared here so main.cu and reference_cpu.cpp agree on every signature at
// COMPILE time (a drifted twin is a silent bug class of its own).
// ===========================================================================
void stack_mean_cpu(const float* stack, int numFrames, int n, float* out_mean);
void elementwise_sub_cpu(const float* a, const float* b, int n, float* out);
void affine_cpu(const float* in, int n, float scale, float offset, float* out);
double roi_sum_cpu(const float* img, int W, int H, int x0, int x1, int y0, int y1);
void radial_bin_cpu(const float* gain, int W, int H, float cx, float cy,
                    int numBins, float binWidthPx,
                    float* bin_sum, int* bin_count);
void correction_cpu(const float* I, const float* dsnu, const float* gain,
                    int n, float gainFloor, float* out);

// ===========================================================================
// SECTION 5 — the shared, HOST-ONLY parametric radial least-squares fit.
//
// Per the twin-independence ruling (see reference_cpu.cpp's file header for
// the full statement): this function is SHARED between the "GPU path" and
// the "CPU reference path" — it is a single 3x3 normal-equations solve over
// at most kNumRadialBins=44 data points (see THEORY.md "The GPU mapping"
// for why a problem this tiny has no meaningful GPU parallelization; it is
// the same "no GPU mapping for a dense micro-solve" call 01.08's
// crf_solve_debevec makes, and 33.01-batched-small-matrix-linalg is the
// project that DOES teach GPU-side batched small solves, at a scale where
// batching actually pays for itself). Because this solve is shared, the
// twin GPU-vs-CPU comparison is BLIND to bugs inside it — which is why the
// radial_fit gate (main.cu) is an INDEPENDENT check against the KNOWN
// analytic cos^4 curve, never against a second implementation of this
// function.
//
// Model (task brief, THEORY.md "The math"): V(r) = 1 + a2*r_n^2 + a4*r_n^4
// + a6*r_n^6, where r_n = r / rNorm is the radius NORMALIZED by rNorm (a
// documented scale, e.g. ~image half-diagonal) so the three basis columns
// [r_n^2, r_n^4, r_n^6] stay numerically well-conditioned (raw pixel-radius
// powers up to r^6 ~ 10^12 would otherwise destroy the normal equations'
// conditioning in FP32/FP64 — see THEORY.md "Numerical considerations").
// The intercept is FIXED at 1 (V(0)=1 by construction of the physical
// model), so only 3 unknowns are fit.
//
// Parameters:
//   bin_r      — [numPoints] bin-center radius, PIXELS (not normalized —
//                this function normalizes internally by rNorm).
//   bin_mean   — [numPoints] the nonparametric gain map's mean value in
//                that radial bin (only bins with count > 0 should be
//                passed in — see main.cu's call site).
//   numPoints  — number of valid (bin_r, bin_mean) pairs.
//   rNorm      — the radius normalization scale (pixels, > 0).
//   out_a2/a4/a6 — OUT: the three fitted coefficients (dimensionless, in
//                  the NORMALIZED r_n basis — main.cu un-normalizes when
//                  evaluating V_fitted(r) = 1 + a2*(r/rNorm)^2 + ...).
// Side effects: none beyond writing the three outputs. Complexity: O(numPoints)
// to build the 3x3 normal equations + O(1) (a fixed-size 3x3 solve).
// ===========================================================================
void fit_vignette_radial_ls(const float* bin_r, const float* bin_mean, int numPoints,
                            float rNorm, float& out_a2, float& out_a4, float& out_a6);

#endif // PROJECT_KERNELS_CUH
