// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 01.03
//               (Optical flow: dense pyramidal Lucas-Kanade + census-
//               transform block-matching flow; Farneback is documented-only,
//               see THEORY.md "The algorithm" and README §13)
//
// Role in the project
// -------------------
// SINGLE-SOURCED CONTRACT between kernels.cu (GPU, nvcc), reference_cpu.cpp
// (independent CPU oracle, cl.exe) and main.cu (orchestration, nvcc). Per
// the repo's twin-independence ruling (docs/PROJECT_TEMPLATE/src/
// reference_cpu.cpp's header, reproduced by every project's reference_cpu.cpp
// file comment): data LAYOUT (image geometry, pyramid level sizing, the
// census offset table, struct shapes, tolerances-adjacent constants) is
// single-sourced HERE; the ALGORITHMIC CORE (gradient stencils, structure-
// tensor accumulation, the 2x2 solve, bilinear warping, census bit-packing,
// Hamming block matching, sub-pixel refinement, LR consistency) is written
// TWICE — once in kernels.cu, independently again in reference_cpu.cpp.
//
// The two implemented milestones this header describes
// ------------------------------------------------------
//   MILESTONE 1 — DENSE PYRAMIDAL LUCAS-KANADE (run_pyramidal_lk_gpu /
//     _cpu): a 3-level image pyramid (kNumLevels, built by the SAME
//     area-average 2x box filter project 01.01's resize_area2x_kernel uses
//     for anti-aliased decimation — see downsample_area2x_kernel's header
//     for the exact citation), dense per-pixel Scharr gradients, a 5x5
//     structure-tensor + mismatch-vector Lucas-Kanade solve with 3 warped
//     re-sampling iterations per level, coarse-to-fine flow-field
//     upsampling between levels. Confidence = the structure tensor's SMALL
//     eigenvalue (the aperture problem, made a first-class per-pixel
//     output — see structure_tensor_kernel's header and THEORY.md).
//   MILESTONE 2 — CENSUS-TRANSFORM BLOCK-MATCHING FLOW (run_census_flow_gpu
//     / _cpu): a 5x5 (24-bit) census signature per pixel (census_transform_
//     kernel), brute-force Hamming block matching over a small search
//     window with parabolic sub-pixel refinement (census_match_kernel,
//     reusing the __popc() lesson from project 01.04's hamming_match_kernel
//     — see that kernel's header), and a forward/backward (left-right)
//     consistency check producing a per-pixel validity mask
//     (census_consistency_kernel).
//   MILESTONE 3 — FARNEBACK polynomial-expansion flow is DOCUMENTED ONLY
//     (THEORY.md "The algorithm"; README §13 states the scoping honestly,
//     per CLAUDE.md §2's bundled-bullet rule) — no kernel exists for it.
//
// Why two very different dense-flow families in one project? They are the
// catalog bullet's two OTHER named methods, and they make the same point
// two different ways: LK regresses on raw intensity differences (fast,
// sub-pixel-accurate, but brittle under non-uniform brightness change and
// needs the pyramid for large motions); census regresses on intensity
// RANK ORDER within a small window (integer, coarser, but provably
// invariant to any monotonic brightness transform and — because it never
// linearizes — needs no pyramid to cover its search radius). THEORY.md
// "The problem" and the brightness-robustness gate in main.cu make this
// contrast the didactic centerpiece of the project.
//
// Why ".cuh"? (repeated from every project in this repo, CLAUDE.md §12)
// -----------------------------------------------------------------------
// This header is included by BOTH kernels.cu (nvcc) and reference_cpu.cpp
// (cl.exe, which has never heard of "__global__"). Every device-only
// declaration is fenced behind #ifdef __CUDACC__; plain structs/constants/
// functions used by all three files sit outside the fence.
//
// Read this after: main.cu.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint8_t, uint32_t — fixed-width types for images and census signatures
#include <cmath>     // fabsf/sqrtf (host use too — <cmath> is plain C++17, fine for cl.exe)

// ===========================================================================
// Image geometry — SINGLE-SOURCED. scripts/make_synthetic.py renders every
// committed sample frame at exactly this resolution; main.cu asserts the
// loaded PGM dimensions match before doing anything else (fail loud, never
// silently truncate — repo convention, see 01.01/01.04's sample loaders).
//
// 160x120 ("QQVGA"-class, 4:3) is chosen small enough that the CPU oracle
// (three pyramid levels, several warped-resample iterations each, a dense
// 13x13 census search at every pixel) finishes in well under a second, and
// large enough that the pyramid's three halvings (160->80->40, 120->60->30)
// land on exact integers at every level — no rounding-induced size drift
// between what downsample_area2x_kernel produces and what the next level's
// kernels expect (a real bug class this exact-power-of-two choice sidesteps
// entirely rather than documenting a fixup).
// ===========================================================================
constexpr int kW = 160;   // full-resolution (level 0) image width, px
constexpr int kH = 120;   // full-resolution (level 0) image height, px

// level_w/level_h — the (exact, since kW/kH are divisible by 2^(kNumLevels-1))
// dimensions of pyramid level L (0 = finest/full-res, increasing = coarser).
// Plain (not constexpr-device) inline functions: only ever called from HOST
// code (main.cu's orchestration, reference_cpu.cpp's pyramid loop, kernels.cu's
// launch wrappers) to size buffers and launch grids — never from a kernel
// body, so no __device__ qualifier is needed.
inline int level_w(int level) { return kW >> level; }
inline int level_h(int level) { return kH >> level; }

// ===========================================================================
// MILESTONE 1 constants — dense pyramidal Lucas-Kanade.
// ===========================================================================
constexpr int kNumLevels = 3;     // pyramid depth: level 0 (160x120) .. level 2 (40x30)

// LK window: 5x5 (radius 2), matching the census window size below — a
// deliberate pedagogical symmetry, not a coincidence (THEORY.md "The
// algorithm" discusses the 5x5-vs-7x7 trade-off explicitly: a 7x7 window
// averages the structure tensor over more samples, which helps conditioning
// in weakly-textured regions at the cost of a larger border loss and 96
// (vs 24) taps per pixel per iteration — 5x5 is the right teaching default
// for a hashed multi-scale texture scene that already has strong gradients
// almost everywhere; the repo's exercises invite trying 7x7).
constexpr int kLkWindowRadius = 2;

// Scharr gradient stencil's OWN minimal border (radius 1: a 3x3 stencil).
// Deliberately SMALLER than the structure-tensor border below — the exact
// two-tier border-arithmetic pattern project 01.04's kSobelBorder /
// kDetectBorder split documents in full (see that header's comment for the
// worked "why the smaller border must exist independently" argument this
// project reuses verbatim): the structure-tensor window later reads
// gx/gy at offsets up to +-kLkWindowRadius from its own center, so those
// samples must be REAL gradients, not zero-forced border padding.
constexpr int kGradBorder = 1;

// The structure-tensor / LK-iterate kernels' own eligible border: any pixel
// whose 5x5 window would read outside the region where gx/gy are valid.
constexpr int kLkBorder = kGradBorder + kLkWindowRadius;   // = 3

// Inner LK iterations per pyramid level (warp I1 with the running estimate,
// recompute the mismatch vector, resolve, accumulate) — 3, per the project
// brief. For the "no pyramid" ablation (README's pyramid_advantage gate),
// main.cu runs LK with kNumLevels=1 and kLkIterationsPerLevel*kNumLevels
// iterations at the single finest level, so BOTH runs spend the identical
// total refinement budget — isolating hierarchical initialization as the
// only variable (see main.cu's "pyramid_advantage" gate and THEORY.md).
constexpr int kLkIterationsPerLevel = 3;

// Per-iteration incremental step clamp, in pixels. The 2x2 solve can return
// a huge (du,dv) where the structure tensor is near-singular (the aperture
// problem: THEORY.md derives why det(M) -> 0 there) or where the current
// warp estimate is badly wrong (early iterations, large residual motion) —
// an unclamped update can overshoot into a region where the local linear
// (first-order Taylor) model is no longer valid at all, diverging instead
// of converging. Clamping each iteration's step is the standard, simplest
// safeguard (Bouguet's pyramidal-LK implementation notes use the same
// device); THEORY.md "Numerical considerations" discusses the alternative
// (a trust-region / Levenberg-style step) as the production refinement.
constexpr float kLkMaxStepPerIterPx = 4.0f;

// Degeneracy floor on det(M) = Sxx*Syy - Sxy^2. Below this, the structure
// tensor is treated as too close to singular to trust ANY solve (both
// eigenvalues small, or one near-zero — flat region or a pure straight
// edge with no cross-direction information) and the iteration leaves the
// flow at its current (pyramid-propagated) estimate rather than injecting
// numerical noise. Value chosen empirically on the committed scene's
// gradient magnitudes (Scharr taps up to +-16 intensity units per pixel
// squared and summed over 25 taps -> Sxx/Syy commonly in the 1e4..1e8
// range on textured pixels; kLkDetEpsilon sits many orders below the
// smallest det(M) this project's synthetic texture ever produces on a
// non-degenerate pixel — see main.cu's [info] line for the measured
// minimum det(M) actually encountered, and THEORY.md "How we verify
// correctness" for why this floor does not need to be delicately tuned).
constexpr float kLkDetEpsilon = 1.0f;

// ===========================================================================
// MILESTONE 2 constants — census-transform block-matching flow.
// ===========================================================================
constexpr int kCensusRadius = 2;         // 5x5 window (matches kLkWindowRadius — see above)
constexpr int kCensusBits   = 24;        // (2*kCensusRadius+1)^2 - 1 (center excluded): 5x5 minus itself

// The 24 (dx,dy) offsets of a 5x5 window EXCLUDING the center, in fixed
// raster order (row-major, dy outer, dx inner) — bit k of a census
// signature always means "is the neighbor at kCensusDx[k],kCensusDy[k]
// brighter-or-equal to the center". SINGLE-SOURCED as a macro literal
// (not a plain constexpr array) for the EXACT same CUDA-language reason
// project 01.04's kFastCircleX/Y documents in full: device kernels need
// their own __constant__-memory copy (a host-storage-class array is
// invisible to device code), and defining the VALUES once as a macro lets
// kernels.cu instantiate a __constant__ copy from the identical literal
// instead of hand-duplicating 24 numbers a second time (a drift risk this
// repo's single-sourcing rule exists to prevent).
#define CENSUS_DX_INIT { -2,-1,0,1,2,  -2,-1,0,1,2,  -2,-1,1,2,  -2,-1,0,1,2,  -2,-1,0,1,2 }
#define CENSUS_DY_INIT { -2,-2,-2,-2,-2,  -1,-1,-1,-1,-1,  0,0,0,0,  1,1,1,1,1,  2,2,2,2,2 }
constexpr int kCensusDx[kCensusBits] = CENSUS_DX_INIT;
constexpr int kCensusDy[kCensusBits] = CENSUS_DY_INIT;

constexpr int kCensusSearchRadius = 6;   // block-match search window: (2*6+1)^2 = 169 candidate displacements
// Total border a pixel needs from the image edge for a FULLY valid census
// match: its own census needs kCensusRadius, AND every candidate it might
// search up to kCensusSearchRadius away must ALSO have a valid census
// (itself kCensusRadius from ITS OWN edges) — so the match kernel's
// eligible region is inset by kCensusRadius + kCensusSearchRadius from
// every edge (see census_match_kernel's header for the worked diagram).
constexpr int kCensusBorder = kCensusRadius + kCensusSearchRadius;   // = 8

// Forward/backward (left-right) consistency tolerance, pixels. A pixel's
// validity mask bit is set iff |forward_flow(p) + backward_flow(p+forward_flow(p))|
// <= this bound — see census_consistency_kernel's header for the geometric
// argument (a correct match's backward flow should point almost exactly
// back where it came from; an occluded/wrong match's usually will not).
constexpr float kCensusConsistencyTolPx = 1.0f;

// ===========================================================================
// Shared numeric constant.
// ===========================================================================
constexpr float kPi = 3.14159265358979323846f;

// ===========================================================================
// popcount32_portable — Hamming weight of a 32-bit word, the classic SWAR
// bit-trick (see project 01.04's identical helper for the full fold-by-fold
// derivation this project reuses verbatim — "no black boxes" for the
// hardware __popc() instruction kernels.cu calls directly). Used by
// reference_cpu.cpp's Hamming-distance twin for census matching; only the
// low kCensusBits=24 bits of the input are ever meaningful here, but the
// full 32-bit trick costs nothing extra and needs no masking to stay
// correct (the high 8 bits are always 0 in a valid census signature).
// ---------------------------------------------------------------------------
inline int popcount32_portable(uint32_t v)
{
    v = v - ((v >> 1) & 0x55555555u);                          // every 2 bits -> count of set bits in that pair
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);          // every 4 bits -> count of set bits in that nibble
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;                          // every 8 bits -> count of set bits in that byte
    return static_cast<int>((v * 0x01010101u) >> 24);           // horizontal sum of the 4 byte-lanes, in the top byte
}

// ===========================================================================
// clamp_f / clampi — tiny shared numeric helpers, plain enough that writing
// them twice would be pure transcription (permitted by the twin-independence
// ruling's third bullet — see this file's header); nothing about EITHER
// twin's correctness depends on these being branch-for-branch identical.
//
// CUDA_HOSTDEV: expands to "__host__ __device__" so kernels.cu's device
// code can call these directly (they run on EVERY thread's own data, no
// different from any other per-thread scalar op) — but expands to nothing
// when reference_cpu.cpp includes this header, because cl.exe (the host
// compiler) has never heard of "__device__" and would fail to parse it.
// ===========================================================================
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
CUDA_HOSTDEV inline float clamp_f(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
CUDA_HOSTDEV inline int   clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// ===========================================================================
// MILESTONE 1 kernels — dense pyramidal Lucas-Kanade. Full documentation
// (the box-filter citation, the Scharr-vs-Sobel choice, the structure-
// tensor/eigenvalue derivation, the warp-and-solve iteration) sits with each
// kernel's DEFINITION in kernels.cu; declarations here just fix signatures.
// ===========================================================================

// downsample_area2x_kernel — exact 2x area-average box-filter decimation of
// a single-channel (grayscale) image, one thread per OUTPUT pixel. The
// single-channel twin of project 01.01's resize_area2x_kernel (see that
// kernel's header for the full anti-aliasing argument this project cites
// rather than re-deriving) — used here to build the LK image pyramid.
// in: [inW*inH] device IN. out: [ (inW/2)*(inH/2) ] device OUT.
__global__ void downsample_area2x_kernel(const uint8_t* __restrict__ in, int inW, int inH,
                                         uint8_t* __restrict__ out);

// scharr_gradient_kernel — per-pixel 3x3 Scharr Gx,Gy (a stencil). Border
// kGradBorder. gx_out/gy_out: [W*H] device OUT, float (exact integers —
// see kernels.cu's numerics note, same argument as 01.04's Sobel kernel).
__global__ void scharr_gradient_kernel(const uint8_t* __restrict__ img, int W, int H,
                                       float* __restrict__ gx_out, float* __restrict__ gy_out);

// structure_tensor_kernel — per-pixel 5x5-window structure tensor
// (Sxx,Syy,Sxy) from gx/gy, PLUS the tensor's small eigenvalue (the
// per-pixel CONFIDENCE this project reports — see THEORY.md's aperture-
// problem derivation). Border kLkBorder.
__global__ void structure_tensor_kernel(const float* __restrict__ gx, const float* __restrict__ gy,
                                        int W, int H,
                                        float* __restrict__ sxx_out, float* __restrict__ syy_out,
                                        float* __restrict__ sxy_out, float* __restrict__ min_eig_out);

// lk_iterate_kernel — ONE forward-additive Lucas-Kanade refinement step:
// bilinear-warp I1 by the RUNNING flow estimate, accumulate the 5x5
// mismatch vector against the PRECOMPUTED (gx,gy,Sxx,Syy,Sxy — computed
// once per level, reused every iteration since they are derived from I0
// only), solve the 2x2 system, clamp, and ADD the increment into
// flow_u/flow_v IN PLACE. Called kLkIterationsPerLevel times per level from
// main.cu's/kernels.cu's orchestration loop. Border kLkBorder (pixels
// outside it are left at their propagated flow, untouched).
__global__ void lk_iterate_kernel(const uint8_t* __restrict__ img0, const uint8_t* __restrict__ img1,
                                  int W, int H,
                                  const float* __restrict__ gx, const float* __restrict__ gy,
                                  const float* __restrict__ sxx, const float* __restrict__ syy,
                                  const float* __restrict__ sxy,
                                  float* __restrict__ flow_u, float* __restrict__ flow_v);

// upsample_flow_kernel — bilinear-upsample a coarse-level flow field to the
// next finer level's resolution AND scale both components by 2x (a
// displacement of N px at half resolution is a displacement of 2N px at
// full resolution — the whole point of coarse-to-fine propagation, THEORY.md
// derives this from the pyramid's geometry). One thread per FINE output
// pixel. coarse_*: [coarseW*coarseH] device IN. fine_*: [fineW*fineH] device
// OUT, where fineW==2*coarseW, fineH==2*coarseH exactly (kW/kH's power-of-2
// construction guarantees this — see this file's image-geometry comment).
__global__ void upsample_flow_kernel(const float* __restrict__ coarse_u, const float* __restrict__ coarse_v,
                                     int coarseW, int coarseH,
                                     float* __restrict__ fine_u, float* __restrict__ fine_v,
                                     int fineW, int fineH);

// ===========================================================================
// MILESTONE 2 kernels — census-transform block-matching flow.
// ===========================================================================

// census_transform_kernel — per-pixel 24-bit census signature (a STENCIL:
// each thread reads its own 5x5 neighborhood, entirely independent of every
// other thread). Border kCensusRadius. census_out: [W*H] device OUT
// (uint32_t, only the low 24 bits meaningful; border pixels get 0, which is
// never mistaken for a real signature because border pixels are also
// excluded from every downstream comparison by kCensusBorder).
__global__ void census_transform_kernel(const uint8_t* __restrict__ img, int W, int H,
                                        uint32_t* __restrict__ census_out);

// census_match_kernel — per REFERENCE-image pixel: brute-force search of
// every candidate displacement in [-R,R]x[-R,R] (the search loop runs
// IN-THREAD — see kernels.cu's header for the GPU-mapping argument and the
// shared-memory-tiling alternative named as an exercise), Hamming
// winner-take-all, then parabolic sub-pixel refinement along each axis.
// census_ref/census_tgt: [W*H] device IN (from census_transform_kernel;
// "ref"/"tgt" rather than "0"/"1" so the SAME kernel serves both the
// forward 0->1 and backward 1->0 passes the consistency check needs).
// Border kCensusBorder. flow_u/flow_v: [W*H] device OUT (sub-pixel,
// pixels; 0 outside the border). cost_min_out: [W*H] device OUT (the
// WINNING integer Hamming cost, 0..24 — used for reporting, NOT gated
// directly; the LR consistency check is this project's validity signal).
__global__ void census_match_kernel(const uint32_t* __restrict__ census_ref,
                                    const uint32_t* __restrict__ census_tgt,
                                    int W, int H,
                                    float* __restrict__ flow_u, float* __restrict__ flow_v,
                                    int* __restrict__ cost_min_out);

// census_consistency_kernel — per-pixel forward/backward consistency check
// (see kCensusConsistencyTolPx's comment for the geometric argument).
// fwd_u/fwd_v: [W*H] device IN (image0->image1 flow). bwd_u/bwd_v: [W*H]
// device IN (image1->image0 flow). valid_out: [W*H] device OUT, 1 = passed
// the consistency check, 0 = failed OR outside the census-eligible border.
__global__ void census_consistency_kernel(const float* __restrict__ fwd_u, const float* __restrict__ fwd_v,
                                          const float* __restrict__ bwd_u, const float* __restrict__ bwd_v,
                                          int W, int H,
                                          uint8_t* __restrict__ valid_out);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// Host-callable LAUNCH WRAPPERS — own the grid/block math + post-launch
// error check (CLAUDE.md §6.1 rule 7), visible to any translation unit
// (only their DEFINITIONS, in kernels.cu, need nvcc).
// ===========================================================================
void launch_downsample_area2x(const uint8_t* d_in, int inW, int inH, uint8_t* d_out);
void launch_scharr_gradient(const uint8_t* d_img, int W, int H, float* d_gx, float* d_gy);
void launch_structure_tensor(const float* d_gx, const float* d_gy, int W, int H,
                             float* d_sxx, float* d_syy, float* d_sxy, float* d_min_eig);
void launch_lk_iterate(const uint8_t* d_img0, const uint8_t* d_img1, int W, int H,
                       const float* d_gx, const float* d_gy,
                       const float* d_sxx, const float* d_syy, const float* d_sxy,
                       float* d_flow_u, float* d_flow_v);
void launch_upsample_flow(const float* d_coarse_u, const float* d_coarse_v, int coarseW, int coarseH,
                          float* d_fine_u, float* d_fine_v, int fineW, int fineH);

void launch_census_transform(const uint8_t* d_img, int W, int H, uint32_t* d_census);
void launch_census_match(const uint32_t* d_census_ref, const uint32_t* d_census_tgt, int W, int H,
                         float* d_flow_u, float* d_flow_v, int* d_cost_min);
void launch_census_consistency(const float* d_fwd_u, const float* d_fwd_v,
                               const float* d_bwd_u, const float* d_bwd_v, int W, int H,
                               uint8_t* d_valid);

// ---------------------------------------------------------------------------
// run_pyramidal_lk_gpu — the FULL Milestone-1 orchestration: builds the
// image pyramid, then loops coarse-to-fine calling the launch_ wrappers
// above (kNumLevels is always built; `num_levels` chooses how many are
// USED, so main.cu can call this with num_levels=1 for the "no pyramid"
// ablation the pyramid_advantage gate needs — see kLkIterationsPerLevel's
// comment). Defined in kernels.cu (it is host orchestration code, not a
// kernel, but lives there because it owns the device buffers between
// kernel launches — main.cu never sees a pyramid-level device pointer).
//
// Parameters: d_img0_full/d_img1_full — [kW*kH] device IN, level-0 frames.
//   num_levels — pyramid levels ACTUALLY used (1 or kNumLevels).
//   iters_per_level — LK iterations run at EACH used level.
//   d_flow_u_out/d_flow_v_out — [kW*kH] device OUT, final level-0 flow.
//   d_min_eig_out — [kW*kH] device OUT, final level-0 confidence.
// Side effects: allocates and frees its own scratch device buffers.
// ---------------------------------------------------------------------------
void run_pyramidal_lk_gpu(const uint8_t* d_img0_full, const uint8_t* d_img1_full,
                          int num_levels, int iters_per_level,
                          float* d_flow_u_out, float* d_flow_v_out, float* d_min_eig_out);

// ---------------------------------------------------------------------------
// run_census_flow_gpu — the FULL Milestone-2 orchestration: census-transform
// both frames, match forward (0->1) and backward (1->0), consistency-check.
// Defined in kernels.cu for the same "owns the device buffers" reason as
// run_pyramidal_lk_gpu above.
//
// Parameters: d_img0/d_img1 — [kW*kH] device IN.
//   d_flow_u_out/d_flow_v_out — [kW*kH] device OUT, forward (0->1) flow.
//   d_valid_out — [kW*kH] device OUT, LR-consistency validity mask.
// ---------------------------------------------------------------------------
void run_census_flow_gpu(const uint8_t* d_img0, const uint8_t* d_img1,
                         float* d_flow_u_out, float* d_flow_v_out, uint8_t* d_valid_out);

// ===========================================================================
// CPU reference (oracle) declarations — defined in reference_cpu.cpp.
// Per-stage twins mirror each kernel of the same name above (independently
// written — see this file's header); the two run_* orchestrators mirror
// the two host-orchestration functions above, ALSO independently written
// (their own pyramid loop / forward+backward+consistency sequence).
// ===========================================================================
void downsample_area2x_cpu(const uint8_t* in, int inW, int inH, uint8_t* out);
void scharr_gradient_cpu(const uint8_t* img, int W, int H, float* gx_out, float* gy_out);
void structure_tensor_cpu(const float* gx, const float* gy, int W, int H,
                          float* sxx_out, float* syy_out, float* sxy_out, float* min_eig_out);
void lk_iterate_cpu(const uint8_t* img0, const uint8_t* img1, int W, int H,
                    const float* gx, const float* gy,
                    const float* sxx, const float* syy, const float* sxy,
                    float* flow_u, float* flow_v);
void upsample_flow_cpu(const float* coarse_u, const float* coarse_v, int coarseW, int coarseH,
                       float* fine_u, float* fine_v, int fineW, int fineH);
void pyramidal_lk_cpu(const uint8_t* img0_full, const uint8_t* img1_full,
                      int num_levels, int iters_per_level,
                      float* flow_u_out, float* flow_v_out, float* min_eig_out);

void census_transform_cpu(const uint8_t* img, int W, int H, uint32_t* census_out);
void census_match_cpu(const uint32_t* census_ref, const uint32_t* census_tgt, int W, int H,
                      float* flow_u, float* flow_v, int* cost_min_out);
void census_consistency_cpu(const float* fwd_u, const float* fwd_v,
                            const float* bwd_u, const float* bwd_v, int W, int H,
                            uint8_t* valid_out);
void census_flow_cpu(const uint8_t* img0, const uint8_t* img1,
                     float* flow_u_out, float* flow_v_out, uint8_t* valid_out);

#endif // PROJECT_KERNELS_CUH
