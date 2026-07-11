// ===========================================================================
// kernels.cuh — interface & data contract for project 01.11
//               Low-light denoising (bilateral, non-local means, BM3D-lite)
//
// Role in the project
// --------------------
// The single-sourced contract between main.cu (orchestration + gates),
// kernels.cu (the five GPU denoisers), reference_cpu.cpp (the CPU oracle
// twins) and scripts/make_synthetic.py (the noisy-frame generator). Every
// geometry constant, noise-model formula, and scene-layout rectangle that
// more than one of those files must agree on lives HERE, once (CLAUDE.md
// §12) — a disagreement anywhere becomes a compile-time signature mismatch
// or an assertion failure, never a silent drift.
//
// RATIFIED SCOPE (CLAUDE.md §2 bundled-bullet rule; task brief) — the
// catalog bullet "Low-light denoising (bilateral, non-local means, fast
// BM3D variant)" bundles three named methods into ONE project. All three
// are implemented as milestones:
//   1. BILATERAL   — joint spatial x range Gaussian stencil, 9x9 window.
//                    Built TWICE (naive global-memory + shared-memory
//                    TILED) to teach the tiling speedup, quantified.
//   2. NON-LOCAL MEANS (NLM) — patch-similarity weighted average, 5x5
//                    patches over a 13x13 search window. The expensive one.
//   3. BM3D-LITE    — a REDUCED, honestly-named first stage of BM3D:
//                    block-match 16 similar 8x8 patches, stack them, apply
//                    a separable 3-D transform (2-D DCT per patch + 1-D
//                    Haar across the stack), HARD-THRESHOLD, invert, and
//                    aggregate with sparsity-weighted averaging. Real BM3D's
//                    SECOND stage (collaborative Wiener filtering using the
//                    hard-threshold result as an oracle) is documented only
//                    (THEORY.md "Where this sits in the real world") — the
//                    "lite" in the name is that omission, stated up front.
// Plus a GAUSSIAN-BLUR baseline: the SAME 9x9 window as bilateral, spatial
// term only (no range term) — a DESIGNED NEGATIVE CONTROL. It must PASS the
// flat-region noise-floor gate (it removes noise) and MUST FAIL the
// edge-preservation gate (it has no mechanism to avoid mixing across an
// edge) — main.cu asserts the failure, proving the gate can tell a real
// denoiser from something that merely blurs (task brief).
//
// THE LOW-LIGHT NOISE STORY (physics — THEORY.md "The problem" derives it)
// ---------------------------------------------------------------------
// A camera pixel counts photons, which arrive as a Poisson process — shot
// noise whose VARIANCE equals the MEAN signal, in electrons (this is why
// low light is qualitatively different from read-noise-dominated imaging:
// SNR ~ sqrt(signal), so darker regions are proportionally noisier, not
// just absolutely noisier — see signal_electrons_of_dn()/predicted_noise_
// std_dn() below and THEORY.md). This project studies a DELIBERATELY
// extreme operating point — scene peak kPeakElectrons = 40 electrons at
// code value 255 — so every committed frame is visibly, heavily noisy
// (the task brief's explicit target). scripts/make_synthetic.py draws
// EXACT Poisson samples via Knuth's multiplicative-inversion algorithm
// (xorshift32-driven, no library RNG; see that script's header for why
// exact sampling beats the Gaussian approximation at this operating
// point, where the darkest committed flat patch has an expected signal
// of only ~4.4 electrons) plus additive Gaussian READ noise, then
// quantizes to 8 bits — the noise generation itself happens ONCE, offline, in Python;
// the C++/CUDA side never draws random numbers, it only DENOISES the
// committed noisy frame and independently re-derives the analytic
// noise-variance PREDICTION (below) to sanity-check the generator
// (main.cu's noise_model_sanity gate).
//
// This project isolates shot+read noise + quantization; it deliberately
// does NOT model per-pixel fixed-pattern noise (DSNU/PRNU/vignetting) —
// that spatial, per-pixel-varying story is 01.09's (photometric/vignetting
// calibration), cited by name in README "System context". Composing the
// two (401.09's calibration THEN this project's denoising) is exactly the
// order a real ISP applies them (README "System context").
//
// TWIN-INDEPENDENCE RULING applied here (see reference_cpu.cpp's header
// for the full statement): the DATA-LAYOUT contracts below (geometry,
// scene-layout rectangles, noise-model formula, BM3D-lite's reference-patch
// grid-position formula) are SHARED HD (host+device) code — sharing them is
// the repo's rule, not an exception, because they are facts about the
// PROBLEM, not the algorithm under test. The algorithmic core of every
// denoiser (the stencil math, the patch search, the DCT/Haar transforms)
// is written TWICE, independently, in kernels.cu (GPU) and
// reference_cpu.cpp (CPU) — per-method VERIFY tolerances in main.cu are
// this project's twin-agreement gate; the four independent psnr/edge/flat/
// noise-model GATEs (main.cu) are what catch a bug hiding in shared code.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>     // sqrtf — used by the HD noise-model helpers below

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe (the 01.01
// precedent, kernels.cuh's own header there explains the mechanism). Used
// ONLY for the small, textbook DATA-CONTRACT helpers below (noise model,
// BM3D-lite grid-position arithmetic) — never for algorithmic cores, which
// this project deliberately writes twice (see file header).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// SECTION 1 — problem geometry. MUST MATCH scripts/make_synthetic.py's
// "MUST MATCH kernels.cuh" block.
// ===========================================================================
// 200x150 — a slightly LARGER canvas than 01.09's 160x120 precedent — this
// project's methods have real spatial reach (BM3D-lite's block-matching
// search extends up to kBm3dSearchRadius+kBm3dPatch =~14 px from a
// reference anchor), so the test scene needs flat/edge/texture regions
// large enough to carry a CONTAMINATION-FREE interior for the gates below
// to measure (see kFlatDark's comment) — 160x120 was tried first and its
// 24x24 flat patches measurably leaked neighboring texture into every
// filter's output near the patch border (main.cu's flat_noise_floor gate
// caught it directly: measured residual std that DEPENDED on how close a
// pixel sat to the patch edge, not on the denoiser's actual noise
// suppression). 200x150 buys the margin the layout below needs.
constexpr int kW = 200;             // frame width, px
constexpr int kH = 150;             // frame height, px
constexpr int kN = kW * kH;         // pixel count per frame (30,000)

// ===========================================================================
// SECTION 2 — the noise model (single source; scripts/make_synthetic.py
// mirrors these exact numbers with a "MUST MATCH kernels.cuh" comment and
// independently re-derives the formula in Python for the SAME reason
// main.cu re-derives it in C++: the noise_model_sanity gate must not trust
// the generator that produced the data it is grading).
//
// kPeakElectrons — the scene's BRIGHTEST pixel (code value 255) has an
//   EXPECTED signal of only 40 photoelectrons. This is an extreme low-light
//   operating point BY DESIGN (task brief): a well-lit frame carries
//   thousands of electrons per pixel; 40 is a very short exposure / high-ISO
//   underexposed frame, chosen specifically so every committed sample is
//   visibly, heavily noisy without needing any artificial gain boost.
// kReadNoiseE — additive, SIGNAL-INDEPENDENT Gaussian read noise, electrons
//   RMS. 2.0 e- sits in the middle of a real back-illuminated (BSI) CMOS
//   sensor's typical 1-3 e- rms range at low analog gain (PRACTICE.md §2
//   dates and sources this).
// kDnPerElectron — the SENSOR GAIN this project assumes: exactly enough
//   digital numbers per electron that code value 255 corresponds to
//   kPeakElectrons electrons (255/40 = 6.375 DN/e-). A real sensor's gain
//   is whatever the ADC's conversion factor happens to be; fixing it this
//   way keeps the whole noise story expressible in DN units without a
//   second free parameter — an honest teaching simplification, stated once.
// ===========================================================================
constexpr float kPeakElectrons = 40.0f;                          // e-, at code value 255
constexpr float kReadNoiseE = 2.0f;                               // e- rms, signal-independent
constexpr float kDnPerElectron = 255.0f / kPeakElectrons;         // = 6.375 DN/e-

// signal_electrons_of_dn — invert the sensor gain: how many electrons does
// a CLEAN (noise-free) code value represent under this project's assumed
// gain? Pure unit conversion, no noise involved.
HD inline float signal_electrons_of_dn(float clean_dn)
{
    return clean_dn / kDnPerElectron;
}

// predicted_noise_std_dn — the ANALYTIC prediction this project's
// noise_model_sanity gate (main.cu) checks the synthetic generator against,
// independently of how the generator actually drew its samples.
//
// Derivation (THEORY.md "The math" walks this in full): Poisson shot noise
// has Var[electrons] = mean[electrons] (a defining property of the Poisson
// distribution — not an approximation); read noise adds an INDEPENDENT
// Gaussian with variance kReadNoiseE^2 (variances of independent noise
// sources add); converting electrons -> DN through the fixed linear gain
// kDnPerElectron scales variance by gain^2 (Var[aX] = a^2 Var[X]). We
// deliberately OMIT the +1/12 quantization-noise term (8-bit rounding
// contributes standard deviation ~0.29 DN, negligible next to the ~20-40 DN
// shot+read noise this operating point produces — main.cu's [info] line
// reports the ratio so the omission is visibly harmless, not hidden).
HD inline float predicted_noise_std_dn(float clean_dn)
{
    const float signal_e = signal_electrons_of_dn(clean_dn);
    const float var_e = signal_e + kReadNoiseE * kReadNoiseE;   // Poisson + read variance, in e-^2
    return sqrtf(var_e) * kDnPerElectron;                        // -> DN via the fixed linear gain
}

// ===========================================================================
// SECTION 3 — the test scene's layout: flat patches, a high-contrast step
// edge, a fine-detail ruling, and a pure-texture ROI. MUST MATCH
// scripts/make_synthetic.py's identical block (the swatch-rectangle
// discipline of 01.09's main.cu, applied here to the SCENE rather than a
// calibration target). Every gate in main.cu reads pixels from these exact
// rectangles in the CLEAN ground truth and in every denoiser's output.
// ===========================================================================
struct Rect { int x0, x1, y0, y1; };   // half-open [x0,x1) x [y0,y1), pixel space

// Three FLAT (constant-radiance) patches at different brightness levels —
// the flat_noise_floor and noise_model_sanity gates' ROIs. Three levels
// deliberately span dark/mid/bright so noise_model_sanity can show the
// predicted std GROWING with signal (the signal-DEPENDENT shot-noise story;
// see the file header) rather than checking a single flat operating point.
//
// 48x48, comfortably larger than any method's spatial reach (bilateral 9x9,
// NLM 13x13 search + 5x5 patch = 17x17 effective, BM3D-lite's block-match
// can pull a matched patch up to kBm3dSearchRadius+kBm3dPatch =~14 px from
// a reference anchor). main.cu's gates measure only the INNER 16x16 (a
// 16-px erosion margin on every side — see main.cu's erode_rect()), so a
// pixel near the patch's own border blending with the surrounding hashed
// texture (a real, and correct, filter behavior) never contaminates the
// "how flat is this region after denoising" measurement — a 160x120 canvas
// with 24x24 patches was tried FIRST and measurably failed exactly this
// way (kW's comment tells the story).
//
// Bright-patch level is kept a comfortable margin below 255: predicted
// noise std GROWS with signal (the signal-dependent shot-noise story), so a
// patch too close to the 8-bit ceiling would have its upper noise tail
// clipped, biasing the measured std low (measured during tuning: code
// value 224 clipped enough tail to read ~17% low against the analytic
// prediction — code value 175 does not).
constexpr Rect kFlatDark{ 8, 56, 8, 56 };             // 48x48, code value 28  (~4.4 e-)
constexpr Rect kFlatMid{ 144, 192, 8, 56 };           // 48x48, code value 128 (~20.1 e-)
constexpr Rect kFlatBright{ 8, 56, 94, 142 };         // 48x48, code value 175 (~27.5 e-)
constexpr float kFlatDarkDn = 28.0f;
constexpr float kFlatMidDn = 128.0f;
constexpr float kFlatBrightDn = 175.0f;
constexpr int kFlatMeasureMargin = 16;   // erosion margin (px) main.cu applies before measuring any kFlat* rect

// A high-contrast STEP EDGE: within kEdgeRegion, every column < kEdgeStepX
// is flat at kEdgeLoDn and every column >= kEdgeStepX is flat at
// kEdgeHiDn — a hard, artifact-free 1-pixel transition the edge_preservation
// gate measures a horizontal finite difference across (main.cu). The step
// sits 33 px from either side of kEdgeRegion — comfortably beyond every
// method's spatial reach, so the measured gradient is never contaminated
// by the surrounding hashed texture the way an un-eroded flat patch would be.
constexpr Rect kEdgeRegion{ 70, 136, 64, 90 };        // 66x26
constexpr int kEdgeStepX = 103;                       // column where lo -> hi (33 px from each border)
constexpr float kEdgeLoDn = 24.0f;
constexpr float kEdgeHiDn = 200.0f;                   // delta 176 DN: unmistakably "high-contrast"

// A FINE-DETAIL ruling: 2-px-wide alternating stripes (4-px period) — far
// finer than any of the three denoisers' spatial support (9x9 bilateral
// window, 5x5 NLM patch, 8x8 BM3D-lite patch). Included in the whole-image
// PSNR sum and called out qualitatively in README "Exercises"/"Limitations"
// as where every method here visibly loses detail — no gate is keyed to it
// alone (an honest scoping choice: quantifying "detail loss" well needs a
// frequency-domain metric this project does not build).
constexpr Rect kFineDetail{ 144, 192, 94, 142 };      // 48x48
constexpr int kFineStripePeriod = 4;                  // px (2 px each of lo/hi)
constexpr float kFineLoDn = 50.0f;
constexpr float kFineHiDn = 200.0f;

// A pure hashed-texture ROI, guaranteed to overlap none of the rectangles
// above — the psnr_improvement / method_ordering gates' primary region:
// flat regions saturate every method's PSNR near the same ceiling (hiding
// the ranking signal), so ranking is measured where genuine self-similarity
// structure exists for NLM/BM3D-lite to exploit and bilateral cannot.
constexpr Rect kTextureRoi{ 64, 136, 8, 56 };          // 72x48

// ===========================================================================
// SECTION 4 — BILATERAL filter + its GAUSSIAN-BLUR negative control.
// Both use the SAME 9x9 window and spatial sigma; the gaussian baseline
// simply omits the range (photometric) term (kernels.cu's gaussian_blur_
// kernel is bilateral's spatial-only special case, written out separately
// so the negative control cannot silently rot out of sync with a future
// bilateral edit).
// ===========================================================================
constexpr int kBilateralRadius = 4;              // 9x9 window (2*4+1)
constexpr float kBilateralSigmaSpatial = 2.5f;   // px — most weight within the radius-4 window
constexpr float kBilateralSigmaRange = 40.0f;    // DN — tuned against this scene's measured
                                                  // ~18-42 DN noise std (README "Expected output"
                                                  // quotes the measured PSNR/edge numbers this
                                                  // value produces)

// ===========================================================================
// SECTION 5 — NON-LOCAL MEANS. 5x5 patches (kNlmPatchRadius=2), 13x13
// search window (kNlmSearchRadius=6) — THEORY.md "The GPU mapping" derives
// the O(search_area * patch_area) per-pixel cost this buys.
// ===========================================================================
constexpr int kNlmPatchRadius = 2;    // 5x5 patch (2*2+1)
constexpr int kNlmSearchRadius = 6;   // 13x13 search window (2*6+1)
constexpr float kNlmH = 38.0f;        // DN — filtering strength; tuned like sigma_range above

// ===========================================================================
// SECTION 6 — BM3D-LITE. 8x8 patches, stride-4 reference grid, 13x13 search
// (same search geometry as NLM — a deliberate consistency choice so README/
// THEORY can describe "the same neighborhood, two different treatments of
// the matches: NLM weights ALL candidates, BM3D-lite HARD-SELECTS the best
// 16 and jointly transforms them"), 16-block stacks, hard-threshold at
// lambda*sigma in the (orthonormal, hence noise-variance-preserving)
// transform domain (THEORY.md "Numerical considerations" proves the
// variance-preservation property this threshold relies on).
// ===========================================================================
constexpr int kBm3dPatch = 8;             // 8x8 reference/candidate patch
constexpr int kBm3dStride = 4;            // reference-grid spacing (50% overlap)
constexpr int kBm3dSearchRadius = 6;      // 13x13 search window around each reference anchor
constexpr int kBm3dStackSize = 16;        // similar blocks kept per group (power of 2: exact Haar)
constexpr float kBm3dThreshLambda = 2.7f; // the standard BM3D hard-threshold multiplier (Dabov et al.)

// kBm3dAssumedSigmaDn — real (AWGN-assuming) BM3D takes ONE noise sigma as
// input; our noise is signal-DEPENDENT (Section 2), so no single sigma is
// exactly right everywhere — an approximation stated honestly (THEORY.md
// "Where this sits in the real world" names the fix: a variance-stabilizing
// Anscombe transform before BM3D, inverted after). We pick the sigma this
// project's noise model predicts at a representative MID-GRAY signal
// (clean_dn = 128), computed BY HAND in double precision and rounded to
// float here (the kRectCos/kRectSin precedent in 01.01's kernels.cuh) so
// every translation unit links the identical constant instead of five
// independently-rounded runtime sqrtf() calls:
//     signal_e  = 128 / 6.375              = 20.0784313725... e-
//     var_e     = 20.0784... + 2.0^2       = 24.0784313725... e-^2
//     std_e     = sqrt(var_e)              = 4.9070... e-
//     std_dn    = std_e * 6.375            = 31.283... DN
constexpr float kBm3dAssumedSigmaDn = 31.283f;
constexpr float kBm3dThreshold = kBm3dThreshLambda * kBm3dAssumedSigmaDn;   // ~84.46 DN

// ---------------------------------------------------------------------------
// BM3D-lite reference-patch GRID GEOMETRY — a data-layout contract (per the
// twin-independence ruling, file header) shared verbatim by the GPU launch
// grid, the CPU reference's outer loop, and main.cu's artifact bookkeeping.
//
// Anchor positions along one axis of length `dim` walk 0, stride, 2*stride,
// ... up to the last position that keeps an 8-wide patch inside [0, dim),
// PLUS (only if the stride does not already land exactly there) one final
// FLUSH position clamped to dim-kBm3dPatch — so the last row/column of the
// image is always covered by at least one reference group even when
// (dim - kBm3dPatch) is not an exact multiple of kBm3dStride.
// ---------------------------------------------------------------------------
HD inline int bm3d_num_positions(int dim)
{
    const int last = dim - kBm3dPatch;                 // last legal anchor coordinate
    const int strided_count = last / kBm3dStride + 1;   // 0, stride, ..., <= last
    const bool need_flush = (last % kBm3dStride) != 0;  // stride does not already reach `last`
    return strided_count + (need_flush ? 1 : 0);
}

HD inline int bm3d_position(int i, int dim)
{
    const int last = dim - kBm3dPatch;
    const int strided_count = last / kBm3dStride + 1;
    if (i < strided_count) return i * kBm3dStride;      // a regular strided anchor
    return last;                                        // the single flush-clamped extra (if it exists)
}

// Reference-group grid size for this project's fixed kW x kH (used to size
// the GPU launch, the CPU loop bounds, and the atomic-aggregation buffers).
constexpr int kBm3dNumX = (kW - kBm3dPatch) / kBm3dStride + 1
                        + (((kW - kBm3dPatch) % kBm3dStride) != 0 ? 1 : 0);   // = 49 for kW=200
constexpr int kBm3dNumY = (kH - kBm3dPatch) / kBm3dStride + 1
                        + (((kH - kBm3dPatch) % kBm3dStride) != 0 ? 1 : 0);   // = 37 for kH=150
constexpr int kBm3dNumGroups = kBm3dNumX * kBm3dNumY;                         // = 1,813 groups

// ===========================================================================
// SECTION 7 — device-only kernel declarations (nvcc only; fenced so
// reference_cpu.cpp, compiled by cl.exe, never sees a __global__ signature).
// Full documentation (thread mapping, memory spaces, numerics) sits with
// each DEFINITION in kernels.cu (CLAUDE.md §6.1); one-line summaries here.
// ===========================================================================
#ifdef __CUDACC__

// 1) BILATERAL, naive global-memory stencil. img/out: kN floats, DN units.
__global__ void bilateral_naive_kernel(const float* __restrict__ img, int W, int H,
                                       float* __restrict__ out);

// 2) BILATERAL, shared-memory TILED stencil — same math, same summation
//    order as (1) (see kernels.cu), so outputs are bit-identical to (1).
__global__ void bilateral_tiled_kernel(const float* __restrict__ img, int W, int H,
                                       float* __restrict__ out);

// 3) GAUSSIAN-BLUR negative control — bilateral's spatial term ALONE.
__global__ void gaussian_blur_kernel(const float* __restrict__ img, int W, int H,
                                     float* __restrict__ out);

// 4) NON-LOCAL MEANS — one thread per output pixel; the expensive kernel.
__global__ void nlm_kernel(const float* __restrict__ img, int W, int H,
                           float* __restrict__ out);

// 5) BM3D-LITE, stage 1: one thread per REFERENCE GROUP. Block-matches,
//    transforms, thresholds, inverts, and atomically scatter-accumulates
//    into out_sum/out_weight (both kN floats) — see kernels.cu for why
//    atomics are unavoidable here (many groups write overlapping pixels).
__global__ void bm3d_group_kernel(const float* __restrict__ img, int W, int H,
                                  float* __restrict__ out_sum, float* __restrict__ out_weight);

// 6) BM3D-LITE, stage 2: finalize out[i] = out_sum[i] / out_weight[i], with
//    img (the original noisy frame) as a DEFENSIVE fallback for the
//    (should-never-trigger, given the coverage guarantee in Section 6)
//    out_weight[i] == 0 case — never divide by zero, never emit garbage.
__global__ void bm3d_finalize_kernel(const float* __restrict__ img,
                                     const float* __restrict__ out_sum,
                                     const float* __restrict__ out_weight,
                                     int n, float* __restrict__ out);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// SECTION 8 — host-callable launch wrappers (visible to every translation
// unit; only their DEFINITIONS in kernels.cu require nvcc). Each owns its
// grid/block math and the mandatory post-launch error check.
// ===========================================================================
void launch_bilateral_naive(const float* d_img, int W, int H, float* d_out);
void launch_bilateral_tiled(const float* d_img, int W, int H, float* d_out);
void launch_gaussian_blur(const float* d_img, int W, int H, float* d_out);
void launch_nlm(const float* d_img, int W, int H, float* d_out);

// launch_bm3d_lite — wraps BOTH kernels above plus the two scratch buffers
// (out_sum/out_weight, zeroed internally) behind the same one-call shape
// every other launcher in this file has. d_img/d_out: kN floats.
void launch_bm3d_lite(const float* d_img, int W, int H, float* d_out);

// ===========================================================================
// SECTION 9 — the CPU reference oracle (reference_cpu.cpp). Declared here so
// main.cu and reference_cpu.cpp agree on every signature at COMPILE time.
// ===========================================================================
void bilateral_cpu(const float* img, int W, int H, float* out);
void gaussian_blur_cpu(const float* img, int W, int H, float* out);
void nlm_cpu(const float* img, int W, int H, float* out);
void bm3d_lite_cpu(const float* img, int W, int H, float* out);

#endif // PROJECT_KERNELS_CUH
