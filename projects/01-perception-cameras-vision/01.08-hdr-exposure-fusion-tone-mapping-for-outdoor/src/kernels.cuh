// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 01.08
//               (HDR exposure fusion + tone mapping for outdoor robots)
//
// Role in the project
// --------------------
// This header is the SINGLE-SOURCED data-layout contract for the whole
// project (CLAUDE.md's twin-independence ruling: layouts and constants are
// shared, the algorithmic CORE is written twice — once here-driven on the
// GPU in kernels.cu, once independently in reference_cpu.cpp). Every file
// in src/ agrees on:
//
//   * the EXPOSURE STACK layout — W x H, N_EXPOSURES=4 uint8 images, the
//     four shutter times kExposureTimes[],
//   * the CRF REPRESENTATION — a 256-entry lookup table g[z] = ln(exposure)
//     recovered by Debevec-Malik (crf_solve_debevec, defined once, shared —
//     see its own header below for why that is the correct call per the
//     independence ruling),
//   * the PYRAMID layout — kNumLevels=3 Gaussian/Laplacian pyramid levels,
//     160x120 -> 80x60 -> 40x30, shared by BOTH HDR paths (local tone
//     mapping and Mertens fusion both build pyramids with the exact same
//     REDUCE/EXPAND primitives — see kernels.cu's file header for why that
//     reuse is a deliberate teaching choice, not corner-cutting),
//   * MERTENS WEIGHT semantics — contrast x well-exposedness (saturation
//     dropped: this project's scenes are grayscale, so the standard
//     Mertens "std across RGB channels" saturation term has no meaning
//     here; see README "Limitations & honesty" and THEORY.md for the
//     honest discussion of that simplification).
//
// Why ".cuh"? (CLAUDE.md §12) — device-only declarations (__global__ kernel
// signatures) are fenced behind #ifdef __CUDACC__ so this header stays
// includable by reference_cpu.cpp, which cl.exe (not nvcc) compiles.
//
// Naming convention used throughout this project's kernel set: every
// low-level kernel is a REUSABLE PRIMITIVE (elementwise map, stencil
// reduce/expand, or 4-way weighted combine) called several times, with
// different buffers, by the higher-level "run_..._gpu" orchestration
// functions that assemble the two HDR paths. Read kernels.cu's file header
// for the full pipeline diagram before reading the declarations below.
//
// Read this after: main.cu.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint8_t — the raw LDR pixel type everywhere in this project

// ===========================================================================
// SECTION 1 — problem geometry (single-sourced; MUST MATCH
// scripts/make_synthetic.py's W/H/EXPOSURE_TIMES_S/NUM_LEVELS — see that
// script's module header for the cross-reference discipline).
// ===========================================================================

// Image dimensions in pixels. 160x120 matches 01.03-optical-flow's precedent
// (chosen there, and reused here, because it halves cleanly twice: 160x120
// -> 80x60 -> 40x30 — exactly the depth our pyramids need, with no ragged
// odd-dimension edge cases to special-case in the REDUCE/EXPAND kernels).
constexpr int kW = 160;
constexpr int kH = 120;
constexpr int kN = kW * kH;   // total pixel count, used everywhere as the flat element count

// Four bracketed exposures (the task's documented shutter speeds). Index 0
// is the SHORTEST (darkest) exposure, index 3 the LONGEST (brightest).
constexpr int kNumExposures = 4;
constexpr float kExposureTimes[kNumExposures] = {
    1.0f / 1000.0f,   // exposure_0.pgm — freezes the sky/sun, everything else near-black
    1.0f / 125.0f,    // exposure_1.pgm
    1.0f / 30.0f,     // exposure_2.pgm
    1.0f / 8.0f       // exposure_3.pgm — exposes the deep shadow, sky/concrete blown out
};

// Pyramid depth for BOTH HDR paths (local tone mapping's base/detail split
// and Mertens' multiresolution blend). Level 0 = full res (160x120), level
// kNumLevels-1 = coarsest (40x30 at depth 3).
constexpr int kNumLevels = 3;

// level_w/level_h — the pyramid geometry helper every orchestration function
// and every caller that sizes a device buffer uses, so the "which level is
// how big" arithmetic exists in exactly ONE place (the 01.03 precedent this
// project explicitly follows). inline + constexpr-friendly plain functions:
// visible to host AND device code (no __host__/__device__ qualifier needed
// for a function this simple that only nvcc AND cl.exe both already parse
// as ordinary C++ — but see kernels.cu for the few spots that DO need the
// explicit dual qualifier).
inline int level_w(int level) { int w = kW; for (int i = 0; i < level; ++i) w /= 2; return w; }
inline int level_h(int level) { int h = kH; for (int i = 0; i < level; ++i) h /= 2; return h; }

// ===========================================================================
// SECTION 2 — the CRF (camera response function) representation.
//
// g[] is a 256-entry table: g[z] = ln(X) for the exposure X that the
// (recovered or, in scripts/make_synthetic.py, the KNOWN) response function
// maps to 8-bit code value z. This is Debevec & Malik's own representation
// (SIGGRAPH 1997) — see THEORY.md "The math" for the full derivation.
// ===========================================================================
constexpr int kCrfBins = 256;

// ===========================================================================
// SECTION 3 — device-only declarations (nvcc only; see the file header for
// why this fence exists). Every kernel below is documented in FULL where it
// is DEFINED (kernels.cu) — headers carry the one-line summary + the
// data-layout contract; kernels.cu carries the essay (CLAUDE.md §6.1).
// ===========================================================================
#ifdef __CUDACC__

// ---- 1) Debevec-Malik radiance merge (PATH A, stage 2) --------------------
// Per-pixel MAP: n=4 uint8 LDR samples + the recovered CRF (read from
// __constant__ memory, see g_crf_table below) + 4 known ln(exposure time)
// scalars -> ONE linear-domain radiance estimate per pixel.
__global__ void radiance_merge_kernel(const uint8_t* __restrict__ z0,
                                      const uint8_t* __restrict__ z1,
                                      const uint8_t* __restrict__ z2,
                                      const uint8_t* __restrict__ z3,
                                      int n,
                                      float ln_t0, float ln_t1, float ln_t2, float ln_t3,
                                      float* __restrict__ out_radiance);

// ---- 2) log-average luminance REDUCTION (PATH A, Reinhard stage 1) --------
// Block-level shared-memory tree reduction + one atomicAdd per block into a
// single double accumulator: sum_i ln(eps + radiance[i]). The host divides
// by n and exponentiates to get L_avg (see run_reinhard_global_gpu).
__global__ void luminance_log_sum_kernel(const float* __restrict__ radiance,
                                         int n, float eps,
                                         double* __restrict__ d_sum_accum);

// ---- 3) Reinhard global tone-map MAP (PATH A, Reinhard stage 2) ----------
// Ld = Lscaled / (1 + Lscaled), Lscaled = key_over_lavg * radiance[i].
// Strictly in [0, 1) for radiance[i] >= 0 — see THEORY.md "The math".
__global__ void reinhard_map_kernel(const float* __restrict__ radiance,
                                    int n, float key_over_lavg,
                                    float* __restrict__ out);

// ---- 4) Gaussian REDUCE (pyramid primitive, shared by both HDR paths) ----
// 5x5 Gaussian blur + 2x downsample: in is inW x inH, out is
// (inW/2) x (inH/2). Border: clamp-to-edge.
__global__ void gaussian_reduce_kernel(const float* __restrict__ in, int inW, int inH,
                                       float* __restrict__ out);

// ---- 5) bilinear EXPAND (pyramid primitive, shared by both HDR paths) ----
// Upsamples in (inW x inH) to an EXACT outW x outH via bilinear
// interpolation (outW/outH are normally 2x inW/inH, one pyramid level up).
__global__ void bilinear_expand_kernel(const float* __restrict__ in, int inW, int inH,
                                       float* __restrict__ out, int outW, int outH);

// ---- 6/7) elementwise combine (generic primitives, n-element flat arrays) -
__global__ void elementwise_sub_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       int n, float* __restrict__ out);   // out = a - b
__global__ void elementwise_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       int n, float* __restrict__ out);   // out = a + b

// ---- 8) affine MAP (generic primitive): out = scale*in + offset ----------
__global__ void affine_kernel(const float* __restrict__ in, int n,
                              float scale, float offset, float* __restrict__ out);

// ---- 9) natural-log MAP (generic primitive): out = ln(in + eps) ----------
__global__ void log_kernel(const float* __restrict__ in, int n, float eps,
                           float* __restrict__ out);

// ---- 10) uint8 -> [0,1) float MAP (generic primitive): out = in / 255 ----
__global__ void u8_to_unit_kernel(const uint8_t* __restrict__ in, int n,
                                  float* __restrict__ out);

// ---- 11) Mertens raw per-exposure weight (PATH B, weight stage) ----------
// STENCIL (3x3 Laplacian for contrast) + MAP (Gaussian well-exposedness):
// raw_weight(x,y) = |laplacian3x3(img)|^wc * exp(-(img-0.5)^2/(2*sigma^2))^we
// img01 is normalized [0,1] (see u8_to_unit_kernel). NOTE: the classic
// Mertens third term (saturation = std across RGB channels) is dropped —
// this project's scenes are single-channel; see kernels.cu's file header
// and README "Limitations & honesty" for the honest accounting of that
// simplification.
__global__ void mertens_raw_weight_kernel(const float* __restrict__ img01, int W, int H,
                                          float wc, float we, float sigma,
                                          float* __restrict__ out_weight);

// ---- 12) normalize 4 raw weights to sum to 1 at every pixel --------------
__global__ void normalize_weights4_kernel(const float* __restrict__ w0, const float* __restrict__ w1,
                                          const float* __restrict__ w2, const float* __restrict__ w3,
                                          int n,
                                          float* __restrict__ o0, float* __restrict__ o1,
                                          float* __restrict__ o2, float* __restrict__ o3);

// ---- 13) generic 4-way weighted sum (naive blend AND per-level fusion) ---
__global__ void weighted_sum4_kernel(const float* __restrict__ a0, const float* __restrict__ w0,
                                     const float* __restrict__ a1, const float* __restrict__ w1,
                                     const float* __restrict__ a2, const float* __restrict__ w2,
                                     const float* __restrict__ a3, const float* __restrict__ w3,
                                     int n, float* __restrict__ out);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// SECTION 4 — host-callable launch wrappers (visible to every translation
// unit; only their DEFINITIONS in kernels.cu require nvcc). Each owns its
// grid/block math and the mandatory post-launch error check.
// ===========================================================================
void upload_crf_table(const float* h_g256);   // cudaMemcpyToSymbol into g_crf_table (device __constant__)

void launch_radiance_merge(const uint8_t* d_z0, const uint8_t* d_z1,
                           const uint8_t* d_z2, const uint8_t* d_z3,
                           int n, float ln_t0, float ln_t1, float ln_t2, float ln_t3,
                           float* d_out_radiance);
void launch_luminance_log_sum(const float* d_radiance, int n, float eps, double* d_sum_accum);
void launch_reinhard_map(const float* d_radiance, int n, float key_over_lavg, float* d_out);
void launch_gaussian_reduce(const float* d_in, int inW, int inH, float* d_out);
void launch_bilinear_expand(const float* d_in, int inW, int inH, float* d_out, int outW, int outH);
void launch_elementwise_sub(const float* d_a, const float* d_b, int n, float* d_out);
void launch_elementwise_add(const float* d_a, const float* d_b, int n, float* d_out);
void launch_affine(const float* d_in, int n, float scale, float offset, float* d_out);
void launch_log(const float* d_in, int n, float eps, float* d_out);
void launch_u8_to_unit(const uint8_t* d_in, int n, float* d_out);
void launch_mertens_raw_weight(const float* d_img01, int W, int H, float wc, float we, float sigma,
                               float* d_out_weight);
void launch_normalize_weights4(const float* d_w0, const float* d_w1, const float* d_w2, const float* d_w3,
                               int n, float* d_o0, float* d_o1, float* d_o2, float* d_o3);
void launch_weighted_sum4(const float* d_a0, const float* d_w0, const float* d_a1, const float* d_w1,
                          const float* d_a2, const float* d_w2, const float* d_a3, const float* d_w3,
                          int n, float* d_out);

// ---- high-level GPU orchestration (allocate scratch, launch the sequence
//      of primitives above, free scratch — the 01.03 "run_..._gpu" pattern
//      this project follows for its two multi-stage HDR paths) ------------
void run_reinhard_global_gpu(const float* d_radiance, int n, float key, float* d_out_reinhard);
void run_local_tonemap_gpu(const float* d_radiance, int W, int H,
                           float compression_factor, float detail_boost,
                           float* d_out_tonemap);
void run_mertens_gpu(const uint8_t* d_z0, const uint8_t* d_z1, const uint8_t* d_z2, const uint8_t* d_z3,
                     int W, int H, float wc, float we, float sigma,
                     float* d_out_naive, float* d_out_fused);

// ===========================================================================
// SECTION 5 — the shared, HOST-ONLY Debevec-Malik CRF solver.
//
// Per the twin-independence ruling (see reference_cpu.cpp's file header for
// the full statement): this function is SHARED between the "GPU path" and
// the "CPU reference path" — there is no meaningful GPU parallelization of
// a ~320x320 dense linear solve (see THEORY.md "The GPU mapping" for why),
// so duplicating it in reference_cpu.cpp would be pure token-for-token
// transcription, exactly the case the ruling calls out as the exception.
// Because this solver is shared, the twin GPU-vs-CPU comparison is BLIND to
// bugs inside it — which is why this project's crf_recovery gate (main.cu)
// is an INDEPENDENT check against scripts/make_synthetic.py's KNOWN
// analytic curve, never against a second implementation of this function.
//
// Parameters:
//   z0..z3     — the four full W*H LDR exposures (device-independent; this
//                is host code, called with host-resident pixel buffers).
//   W, H       — image geometry.
//   t0..t3     — the four exposure times, SECONDS (not logs — this function
//                takes ln() internally, once, so callers never have to).
//   grid_n     — sample points per axis (grid_n x grid_n regular grid of
//                pixels feeds the solve — see kernels.cu for the exact
//                placement formula).
//   margin     — pixels of border excluded from the sample grid.
//   lambda     — smoothness-prior weight (THEORY.md "The math").
//   out_g256   — OUT: the recovered g[256] table, g[z] = ln(exposure).
// Side effects: none beyond writing out_g256. Complexity: see kernels.cu.
// ===========================================================================
void crf_solve_debevec(const uint8_t* z0, const uint8_t* z1, const uint8_t* z2, const uint8_t* z3,
                       int W, int H, float t0, float t1, float t2, float t3,
                       int grid_n, int margin, float lambda,
                       float* out_g256);

// ===========================================================================
// SECTION 6 — the CPU reference oracle (defined in reference_cpu.cpp).
// Declared here so main.cu and reference_cpu.cpp agree on every signature
// at COMPILE time (a drifted twin is a silent bug class of its own).
// ===========================================================================
// g256 is the SAME recovered CRF table crf_solve_debevec produced and
// upload_crf_table() uploaded to the GPU's __constant__ memory — passed
// here EXPLICITLY (rather than via a hidden global) so this reference
// stays a pure function of its arguments, easy to reason about in
// isolation. The CRF is common CALIBRATION INPUT to both paths, not part
// of what either twin computes — see reference_cpu.cpp's file header for
// why that keeps the independence ruling intact.
void radiance_merge_cpu(const uint8_t* z0, const uint8_t* z1, const uint8_t* z2, const uint8_t* z3,
                        int n, float ln_t0, float ln_t1, float ln_t2, float ln_t3,
                        const float* g256,
                        float* out_radiance);
double luminance_log_mean_cpu(const float* radiance, int n, float eps);
void reinhard_map_cpu(const float* radiance, int n, float key_over_lavg, float* out);
void gaussian_reduce_cpu(const float* in, int inW, int inH, float* out);
void bilinear_expand_cpu(const float* in, int inW, int inH, float* out, int outW, int outH);
void elementwise_sub_cpu(const float* a, const float* b, int n, float* out);
void elementwise_add_cpu(const float* a, const float* b, int n, float* out);
void affine_cpu(const float* in, int n, float scale, float offset, float* out);
void log_map_cpu(const float* in, int n, float eps, float* out);
void u8_to_unit_cpu(const uint8_t* in, int n, float* out);
void mertens_raw_weight_cpu(const float* img01, int W, int H, float wc, float we, float sigma,
                            float* out_weight);
void normalize_weights4_cpu(const float* w0, const float* w1, const float* w2, const float* w3,
                            int n, float* o0, float* o1, float* o2, float* o3);
void weighted_sum4_cpu(const float* a0, const float* w0, const float* a1, const float* w1,
                       const float* a2, const float* w2, const float* a3, const float* w3,
                       int n, float* out);

// High-level CPU orchestration mirroring the GPU run_..._cpu functions
// above one-for-one (same stage sequence, independent implementation).
void run_reinhard_global_cpu(const float* radiance, int n, float key, float* out_reinhard);
void run_local_tonemap_cpu(const float* radiance, int W, int H,
                           float compression_factor, float detail_boost,
                           float* out_tonemap);
void run_mertens_cpu(const uint8_t* z0, const uint8_t* z1, const uint8_t* z2, const uint8_t* z3,
                     int W, int H, float wc, float we, float sigma,
                     float* out_naive, float* out_fused);

#endif // PROJECT_KERNELS_CUH
