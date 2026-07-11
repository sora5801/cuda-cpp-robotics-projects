// ===========================================================================
// kernels.cuh — interface & data contract for project 01.22
//               Motion deblurring and super-resolution for inspection zoom
//
// Role in the project
// --------------------
// The single-sourced contract between main.cu (orchestration + gates),
// kernels.cu (the GPU kernels + cuFFT wrappers), reference_cpu.cpp (the
// independent CPU oracle twins) and scripts/make_synthetic.py (the scene /
// blur / low-res-frame generator). Every geometry constant, PSF parameter,
// and scene-layout rectangle that more than one of those files must agree
// on lives HERE, once (CLAUDE.md §12) — make_synthetic.py mirrors the
// numbers below with "MUST MATCH kernels.cuh" comments (the 01.11 precedent
// this project follows) because Python cannot #include a .cuh file.
//
// RATIFIED SCOPE (two milestones; task brief) — the catalog bullet "Motion
// deblurring and super-resolution for inspection zoom" bundles two related
// inverse problems into ONE project, both about recovering detail an
// inspection camera's raw frame does not directly show:
//   MILESTONE 1 — MOTION DEBLURRING (non-blind): a KNOWN line PSF (derived
//     from constant-velocity motion during a global-shutter exposure — see
//     THEORY.md, which cites project 01.10's exposure-INTEGRAL derivation
//     for rolling shutter and specializes it to the simpler global-shutter
//     case). Three restorations of the SAME blurred+noisy frame: a Wiener
//     filter (frequency domain, cuFFT), a DELIBERATELY naive inverse filter
//     (frequency domain, same cuFFT machinery, EXPLODES at PSF spectral
//     zeros — the designed failure demo), and Richardson-Lucy (spatial
//     domain, multiplicative EM update, iterative). Plus a PSF-MISMATCH
//     honesty run: Wiener deconvolution with a deliberately wrong PSF angle.
//   MILESTONE 2 — MULTI-FRAME SUPER-RESOLUTION: N=8 low-res frames with
//     KNOWN sub-pixel shifts (quarter-LR-pixel lattice) are combined by
//     shift-and-add onto a 2x grid, refined by iterative back-projection
//     (IBP), and compared against bicubic upscaling of a single LR frame
//     (the baseline that CANNOT recover aliased detail — THEORY.md derives
//     why: single-frame upsampling adds no new SAMPLES, only smooths
//     existing ones, so information lost to aliasing at capture time is
//     gone; multi-frame SR works because each shifted frame samples a
//     DIFFERENT phase of the same band-limited scene, and combined they
//     approximate sampling at the finer 2x grid's Nyquist rate).
//
// THE SHARED SCENE (kW x kH, generated once by make_synthetic.py; see that
// script's header for the exact rasterization method) carries every region
// BOTH milestones' gates read: a flat patch (noise-floor honesty, ties to
// 01.11's flat_noise_floor gate by name), a high-contrast step edge (edge-
// sharpness, ties to 01.11's edge_gradient_mean by name and reuses its
// formula), a small "text-like" dot-matrix glyph row (NOT a real font — a
// hand-drawn 5x7 bitmap set, stated honestly), a deterministic hashed
// texture patch, and THREE vertical bar-chart frequency groups whose
// periods straddle the low-resolution grid's Nyquist limit — the
// sr_resolution gate's "money shot": multi-frame SR must resolve contrast
// at the FINE group that single-frame bicubic upscaling cannot.
//
// TWIN-INDEPENDENCE RULING applied here (see reference_cpu.cpp's header for
// the full statement, restated from 01.11's kernels.cuh): the DATA-LAYOUT
// contracts below (geometry, scene rectangles, PSF size, the shift-table
// SHAPE) are a shared PROBLEM-DEFINITION contract, not an algorithm under
// test — sharing them is the repo's rule, not an exception. The ALGORITHMIC
// core of every restoration method (the FFT itself, the Wiener/naive-
// inverse frequency-domain formulas, the Richardson-Lucy update, the
// shift-and-add splat, the IBP forward/back projection, bicubic
// interpolation) is written TWICE, independently: cuFFT + hand-written CUDA
// kernels here (kernels.cu) vs. a from-scratch radix-2 CPU FFT and
// from-scratch spatial loops in reference_cpu.cpp. Per-method VERIFY
// tolerances in main.cu are the twin-agreement gate; the independent
// PSNR/contrast/monotonicity GATEs (main.cu) are what catch a bug hiding in
// shared code (or in cuFFT itself, which this project treats as a trusted
// library the same way 03.01 does — see kernels.cu's file header for why).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>     // sinf/cosf/floorf — used by the HD geometry helpers below

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe (the 01.01/
// 01.11 precedent). Used ONLY for small textbook data-contract helpers
// (bilinear-footprint arithmetic, angle-to-unit-vector) — never for the
// algorithmic cores, which this project deliberately writes twice (see the
// twin-independence ruling above).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// ComplexF32 — a plain, CUDA-header-free complex sample (the 03.01
// precedent, cited by name per the task brief). reference_cpu.cpp is
// compiled by cl.exe and must never see <cufft.h> or CUDA's ComplexF32 vector
// type; kernels.cu reinterpret_casts ComplexF32* to cufftComplex* at every
// cuFFT call site (both are bit-for-bit {float re; float im;} — cufftComplex
// is itself typedef'd from ComplexF32 in cufft.h) so exactly ONE plain type
// describes "one complex frequency-domain sample" everywhere in this
// project, host and device alike.
// ---------------------------------------------------------------------------
struct ComplexF32 { float re, im; };

// ===========================================================================
// SECTION 1 — problem geometry. MUST MATCH scripts/make_synthetic.py's
// "MUST MATCH kernels.cuh" block.
//
// kW = kH = 128 = 2^7: a SQUARE, power-of-two canvas. Square keeps the bar-
// chart / glyph / texture layout simple to reason about; power-of-two in
// BOTH dimensions is what lets reference_cpu.cpp implement an independent,
// textbook radix-2 Cooley-Tukey FFT (see that file's header) instead of a
// general mixed-radix one — a real teaching simplification, stated once
// here rather than hidden in the FFT code.
// ===========================================================================
constexpr int kW = 128;              // truth/deblur-image width, px (also the SR 2x output grid width)
constexpr int kH = 128;              // truth/deblur-image height, px (also the SR 2x output grid height)
constexpr int kN = kW * kH;          // pixel count (16,384)

// R2C cuFFT's "advanced data layout" for a real WxH image: the complex
// spectrum is only W/2+1 columns wide (the other half is the redundant
// conjugate mirror of a real signal's spectrum — storing it would waste
// exactly half the memory for zero new information). kFreqW/kFreqN size
// every complex device buffer this project allocates for milestone 1.
constexpr int kFreqW = kW / 2 + 1;   // 65
constexpr int kFreqN = kFreqW * kH;  // 8,320 complex bins

// ===========================================================================
// SECTION 2 — the motion-blur PSF (point spread function).
//
// A camera translating at constant velocity during a GLOBAL-shutter
// exposure integrates the same scene ray-bundle over a straight line
// segment in image space (THEORY.md derives this from the exposure
// integral — the same physical idea project 01.10 uses for ROLLING
// shutter's per-row integral, specialized here to one global exposure
// window, cited by name in README "System context"). The result is a
// LINE PSF: a 1-D segment of length kBlurLengthPx at angle kBlurAngleDeg,
// anti-alias-rasterized into a kPsfSize x kPsfSize kernel that sums to 1.0
// (energy-preserving — the blur redistributes light, it does not create or
// destroy it).
//
// kBlurLengthPx=9.0, kBlurAngleDeg=20.0 are chosen so the blur is clearly
// visible against this project's ~10-14px glyph strokes and the bar-chart's
// 3-8px periods (a shorter blur would be nearly invisible; the README
// "Expected output" quotes the measured PSNR this combination produces).
//
// kMismatchAngleDeg = kBlurAngleDeg + 25.0: the PSF-mismatch honesty run
// (README/THEORY "Limitations") deconvolves the SAME blurred frame with a
// PSF rotated 25 degrees from the true motion direction — a plausible
// real-world error (e.g. a bad IMU/encoder-to-camera extrinsic calibration
// feeding the wrong direction into a motion-metadata-driven PSF, PRACTICE.md
// §3) — and main.cu MEASURES how much the reconstruction degrades.
//
// The PSF itself is NOT recomputed by this project's C++/CUDA code: it is
// generated ONCE by scripts/make_synthetic.py (the authoritative
// rasterizer) and committed as psf_truth.csv / psf_mismatch.csv — exactly
// the way a real inspection system would receive a PSF derived offline from
// encoder/IMU motion metadata (PRACTICE.md §3), not re-derive the line
// geometry on every frame. kPsfSize/kPsfRadius below size the dense kernel
// array every deconvolution method reads.
// ===========================================================================
constexpr int kPsfSize = 15;                    // odd: kernel spans [-7, +7] px
constexpr int kPsfRadius = kPsfSize / 2;        // 7
constexpr float kBlurLengthPx = 9.0f;           // px, motion length during the exposure
constexpr float kBlurAngleDeg = 20.0f;          // degrees from +x (image right), CCW
constexpr float kMismatchAngleDeg = kBlurAngleDeg + 25.0f;  // the deliberately WRONG PSF angle (45 deg)
constexpr float kBlurNoiseStdDn = 3.0f;         // additive Gaussian sensor noise, DN rms, post-blur

// ---------------------------------------------------------------------------
// Wiener / naive-inverse regularization constants.
//
// The Wiener filter (THEORY.md "The math" derives this from MMSE
// estimation) is  Hinv(f) = conj(H(f)) / (|H(f)|^2 + K)  where H is the
// PSF's frequency response and K approximates the noise-to-signal POWER
// ratio. Real Wiener filtering uses a per-frequency K = Sn(f)/Ss(f); this
// project uses the standard TEACHING SIMPLIFICATION of a single CONSTANT K
// (the "parametric Wiener filter" — THEORY.md "Where this sits in the real
// world" names the frequency-dependent version production ISPs use).
// kWienerK below was MEASURED (main.cu's [info] lines reproduce the
// measurement) then chosen to sit comfortably in the range that clears the
// wiener_recovery gate without over-smoothing — see README "Expected
// output" for the exact quoted PSNR this value produces.
//
// kNaiveInverseEpsilon is NOT a regularizer — it exists only so a literal
// H(f)=0 bin does not produce an IEEE division-by-zero (inf/nan, which
// would corrupt the WHOLE inverse FFT via that single bin, not just fail
// gracefully at it). At kPsfSize=15/length=9, the PSF's spectrum has
// several near-zero bins (a line PSF is sinc-like along its motion axis —
// THEORY.md plots this); dividing measured NOISE by a near-zero PSF
// response amplifies that noise by orders of magnitude — the naive_
// inverse_failure gate asserts the result is WORSE than doing nothing.
// ---------------------------------------------------------------------------
constexpr float kWienerK = 0.006f;              // measured/tuned; see README "Expected output"
constexpr float kNaiveInverseEpsilon = 1.0e-4f; // numerically necessary floor only, NOT a regularizer

// Richardson-Lucy (THEORY.md derives this from Poisson maximum-likelihood):
// the multiplicative EM update  x_{k+1} = x_k * ( PSF^T * (y / (PSF * x_k)) ).
// kRlIterations=30 is the task-documented iteration count; kRlEpsilon guards
// the division the same way kNaiveInverseEpsilon does (denominators can
// legitimately approach zero in dark image regions).
constexpr int kRlIterations = 30;
constexpr float kRlEpsilon = 1.0e-3f;

// ===========================================================================
// SECTION 3 — multi-frame super-resolution geometry.
//
// kLrScale=2: each low-resolution frame is HALF the truth canvas's linear
// size in each axis — a modest, teaching-scale SR factor (real computational-
// photography SR often pushes 2-4x; THEORY.md "Where this sits in the real
// world" names phone-camera multi-frame SR as the production analogue).
// kNumFrames=8 low-res frames, each a DIFFERENT sub-pixel phase of the same
// scene (make_synthetic.py's honest generation method: render a 4x
// supersampled master canvas once, extract a shifted window per frame, then
// BOX-DOWNSAMPLE by 8x total to LR resolution — the same anti-aliasing
// operator a real sensor's pixel-integration applies, so the LR frames are
// genuinely, physically aliased, not merely blurred).
// ===========================================================================
constexpr int kLrScale = 2;                          // SR upsampling factor (LR -> HR grid)
constexpr int kLrW = kW / kLrScale;                  // 64
constexpr int kLrH = kH / kLrScale;                  // 64
constexpr int kLrN = kLrW * kLrH;                    // 4,096 px per LR frame
constexpr int kNumFrames = 8;                        // low-res frames combined into one HR estimate
constexpr int kLrFramesN = kNumFrames * kLrN;        // 32,768 — total LR samples across all frames

constexpr int kIbpIterations = 12;                   // iterative back-projection iteration count
constexpr float kIbpStep = 0.6f;                     // IBP relaxation (step) factor; < 1 for stable convergence

// ---------------------------------------------------------------------------
// Shift — one LR frame's known sub-pixel registration, in LR-PIXEL units
// (NOT truth-pixel units — multiply by kLrScale to get truth/HR-pixel
// units). A real system derives this from encoder/IMU-driven motion
// metadata or explicit sub-pixel registration (PRACTICE.md §3); here it is
// GROUND TRUTH written by make_synthetic.py to data/sample/shifts_truth.csv
// and read back verbatim by main.cu — this project studies NON-BLIND SR
// (known shifts), the multi-frame analogue of milestone 1's non-blind
// deconvolution (blind SR/deblurring are both documented-only, README
// "Limitations").
// ---------------------------------------------------------------------------
struct Shift { float dx_lrpx, dy_lrpx; };

// ===========================================================================
// SECTION 4 — shared scene-layout rectangles. MUST MATCH
// scripts/make_synthetic.py's identical block (the 01.09/01.11 swatch-
// rectangle discipline). Every gate in main.cu reads pixels from these
// exact regions in the truth image and in every method's output.
// ===========================================================================
struct Rect { int x0, x1, y0, y1; };  // half-open [x0,x1) x [y0,y1), pixel space

// A flat (constant-radiance) patch — the noise-honesty report's ROI
// (main.cu ties this to 01.11's flat_noise_floor gate BY NAME: every
// restoration method here, like every denoiser there, trades noise
// suppression against detail recovery, and a flat region is where pure
// noise behavior is visible with no scene structure to confound it).
constexpr Rect kFlatRect{ 8, 44, 8, 40 };            // 36x32, truth value kFlatDn
constexpr float kFlatDn = 128.0f;

// A high-contrast vertical step edge — edge_gradient_mean() below reuses
// 01.11's exact formula (a horizontal finite difference across a known
// column) to measure how much of the step SURVIVES each restoration.
constexpr Rect kEdgeRegion{ 52, 120, 8, 40 };        // 68x32
constexpr int kEdgeStepX = 86;                        // column where lo -> hi
constexpr float kEdgeLoDn = 24.0f;
constexpr float kEdgeHiDn = 220.0f;                   // delta 196 DN

// The "text-like" dot-matrix glyph row — SEVEN hand-drawn 5x7 bitmap
// glyphs (NOT a real font; make_synthetic.py's GLYPH_* tables are the
// authoritative shapes), each glyph cell 10x14 truth px (2px per bitmap
// dot), laid out left to right at a 14px pitch.
constexpr Rect kGlyphsRegion{ 8, 120, 44, 68 };       // 112x24
constexpr int kGlyphCellPx = 2;                        // truth px per bitmap dot
constexpr int kGlyphPitchPx = 14;                      // px between glyph origins
constexpr float kGlyphLoDn = 20.0f;
constexpr float kGlyphHiDn = 235.0f;

// A deterministic hashed-texture patch (mimics a machined/labeled surface's
// stochastic micro-texture) — 4x4 truth-px blocks, three DN levels chosen
// by hashing the block index (xorshift32, make_synthetic.py's generator;
// no gate is keyed to this region alone — it is qualitative "does detail
// survive" context for the artifact crops, like 01.11's fine-detail region).
constexpr Rect kTextureRegion{ 8, 120, 72, 96 };      // 112x24
constexpr int kTextureBlockPx = 4;

// Three vertical bar-chart frequency groups (alternating lo/hi stripes
// varying along x) at PERIODS chosen to straddle the LOW-RESOLUTION grid's
// Nyquist limit — the sr_resolution gate's money shot (README/THEORY):
//   coarse: period 8 truth-px/cycle = 4 LR-px/cycle -- comfortably ABOVE
//     the LR Nyquist period (4 LR-px would need >=4 LR-px/cycle to be
//     resolvable; 4 sits exactly at the boundary -- see "mid" below), a
//     sanity-check group both bicubic and SR should resolve well.
//   mid: period 4 truth-px/cycle = 2 LR-px/cycle -- exactly the LR
//     Nyquist period (2 LR-samples/cycle is the textbook minimum) -- a
//     boundary case, reported but not gated.
//   fine: period 3 truth-px/cycle = 1.5 LR-px/cycle -- BELOW the LR
//     Nyquist period. A single LR frame CANNOT represent this frequency
//     without aliasing (Nyquist-Shannon: >=2 samples/cycle required); only
//     multi-frame SR's finer effective sampling (each frame a different
//     sub-pixel phase) can recover it. This is the gated frequency.
// All three at the SAME orientation (vertical stripes) — a scoping choice
// stated in README "Limitations"; horizontal/diagonal bar orientations are
// listed as an exercise, not implemented, to keep the canvas small.
// ===========================================================================
constexpr Rect kBarCoarseRegion{ 8, 40, 100, 120 };   // 32x20, period 8
constexpr Rect kBarMidRegion{ 44, 76, 100, 120 };     // 32x20, period 4
constexpr Rect kBarFineRegion{ 80, 116, 100, 120 };   // 36x20, period 3
constexpr int kBarPeriodCoarse = 8;
constexpr int kBarPeriodMid = 4;
constexpr int kBarPeriodFine = 3;
constexpr float kBarLoDn = 30.0f;
constexpr float kBarHiDn = 225.0f;

// ===========================================================================
// SECTION 5 — HD geometry helpers shared (as DATA-CONTRACT arithmetic, not
// algorithm) by the GPU gather kernels and the CPU twins.
// ===========================================================================

// bilinear_weights — given a continuous coordinate (in destination-grid
// units) and a destination size, return the top-left integer sample index
// and the four bilinear weights, CLAMPING to the valid range at borders
// (clamp-to-edge, the simplest defensible border rule for a small teaching
// canvas — THEORY.md "Numerical considerations" notes the alternative,
// wraparound, would be wrong here because these images are NOT periodic).
// Used by bicubic's simpler cousin (shift-and-add's splat target) and by
// IBP's forward/back gather steps.
struct BilinearSample { int x0, y0; float wx, wy; };  // x0,y0: floor sample; wx,wy in [0,1]: fractional part

HD inline BilinearSample bilinear_sample_at(float cx, float cy, int W, int H)
{
    // Clamp the continuous coordinate into the valid interpolation range
    // [0, W-1] x [0, H-1] BEFORE flooring, so a shift that lands slightly
    // outside the frame (this project's largest shift is 1.5 truth px,
    // README "Limitations") degrades gracefully to edge-clamped sampling
    // instead of reading out-of-bounds memory.
    if (cx < 0.0f) cx = 0.0f; if (cx > static_cast<float>(W - 1)) cx = static_cast<float>(W - 1);
    if (cy < 0.0f) cy = 0.0f; if (cy > static_cast<float>(H - 1)) cy = static_cast<float>(H - 1);
    int x0 = static_cast<int>(cx);              // floor (cx >= 0 here, so truncation == floor)
    int y0 = static_cast<int>(cy);
    if (x0 > W - 2) x0 = W - 2;                  // keep x0+1 in range (only matters when W==1, defensive)
    if (y0 > H - 2) y0 = H - 2;
    BilinearSample s;
    s.x0 = x0; s.y0 = y0;
    s.wx = cx - static_cast<float>(x0);
    s.wy = cy - static_cast<float>(y0);
    return s;
}

// ===========================================================================
// SECTION 6 — device-only kernel declarations (nvcc only; fenced so
// reference_cpu.cpp, compiled by cl.exe, never sees a __global__ signature).
// Full documentation (thread mapping, memory spaces, numerics) sits with
// each DEFINITION in kernels.cu; one-line summaries here.
// ===========================================================================
#ifdef __CUDACC__

// -- Milestone 1: frequency-domain pointwise ops (post-cuFFT). All operate
// on kFreqN complex bins, one thread per bin (a pure map).
__global__ void naive_inverse_kernel(const ComplexF32* __restrict__ blurred_freq,
                                     const ComplexF32* __restrict__ psf_freq,
                                     ComplexF32* __restrict__ out_freq);
__global__ void wiener_kernel(const ComplexF32* __restrict__ blurred_freq,
                              const ComplexF32* __restrict__ psf_freq,
                              ComplexF32* __restrict__ out_freq, float K);
__global__ void scale_real_kernel(float* __restrict__ img, int n, float scale);

// -- Milestone 1: spatial-domain circular convolution (Richardson-Lucy's
// two conv steps per iteration) — one thread per OUTPUT pixel, dense
// kPsfSize x kPsfSize stencil with wraparound indexing (matches the FFT
// path's circular-convolution semantics — see kernels.cu).
__global__ void convolve_circular_kernel(const float* __restrict__ img,
                                         const float* __restrict__ psf, // kPsfSize*kPsfSize
                                         float* __restrict__ out);
__global__ void divide_safe_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ out, int n, float eps);
__global__ void multiply_inplace_kernel(float* __restrict__ a, const float* __restrict__ b, int n);
__global__ void subtract_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ out, int n);

// -- Milestone 2: SR. shift_and_add is the project's SCATTER kernel (many
// LR samples land near the same HR cell — atomics are unavoidable, see
// kernels.cu); forward-simulate/back-project/bicubic are GATHER kernels
// (each output pixel reads a bounded, deterministic set of inputs).
__global__ void shift_and_add_kernel(const float* __restrict__ lr_frames,   // [kNumFrames*kLrN]
                                     const Shift* __restrict__ shifts,      // [kNumFrames]
                                     float* __restrict__ hr_sum,            // [kN], atomically accumulated
                                     float* __restrict__ hr_weight);        // [kN], atomically accumulated
__global__ void finalize_splat_kernel(const float* __restrict__ hr_sum, const float* __restrict__ hr_weight,
                                      const float* __restrict__ fallback, float* __restrict__ hr_out);
__global__ void forward_simulate_kernel(const float* __restrict__ hr_estimate,
                                        const Shift* __restrict__ shifts,
                                        float* __restrict__ lr_predicted);  // [kNumFrames*kLrN]
__global__ void backproject_kernel(const float* __restrict__ residual,     // [kNumFrames*kLrN]
                                   const Shift* __restrict__ shifts,
                                   float* __restrict__ hr_estimate,        // updated in place
                                   float step);
__global__ void bicubic_upscale_kernel(const float* __restrict__ lr, float* __restrict__ hr);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// SECTION 7 — host-callable launch wrappers (every translation unit sees
// these; only their DEFINITIONS in kernels.cu require nvcc). Each owns its
// grid/block math, the mandatory post-launch error check, and — for the
// cuFFT-backed ones — the plan lifecycle (see kernels.cu for why plans are
// created/destroyed per call here rather than cached: a teaching-clarity
// choice, discussed in that file's header).
// ===========================================================================
void launch_fft_forward_r2c(const float* d_img, ComplexF32* d_freq);         // [kN] -> [kFreqN]
void launch_fft_inverse_c2r(const ComplexF32* d_freq, float* d_img);         // [kFreqN] -> [kN], NORMALIZED (see kernels.cu)
void launch_naive_inverse(const ComplexF32* d_blurred_freq, const ComplexF32* d_psf_freq, ComplexF32* d_out_freq);
void launch_wiener(const ComplexF32* d_blurred_freq, const ComplexF32* d_psf_freq, ComplexF32* d_out_freq, float K);
void launch_scale_real(float* d_img, int n, float scale);

void launch_convolve_circular(const float* d_img, const float* d_psf, float* d_out);
void launch_divide_safe(const float* d_a, const float* d_b, float* d_out, int n, float eps);
void launch_multiply_inplace(float* d_a, const float* d_b, int n);
void launch_subtract(const float* d_a, const float* d_b, float* d_out, int n);

void launch_shift_and_add(const float* d_lr_frames, const Shift* d_shifts, float* d_hr_sum, float* d_hr_weight);
void launch_finalize_splat(const float* d_hr_sum, const float* d_hr_weight, const float* d_fallback, float* d_hr_out);
void launch_forward_simulate(const float* d_hr_estimate, const Shift* d_shifts, float* d_lr_predicted);
void launch_backproject(const float* d_residual, const Shift* d_shifts, float* d_hr_estimate, float step);
void launch_bicubic_upscale(const float* d_lr, float* d_hr);

// ===========================================================================
// SECTION 8 — the CPU reference oracle (reference_cpu.cpp). Declared here so
// main.cu and reference_cpu.cpp agree on every signature at COMPILE time.
// ComplexF32-shaped complex data is represented on the host as a plain struct
// (Complex64, defined in reference_cpu.cpp) to keep this header CUDA-vector-
// type-free outside the __CUDACC__ fence; main.cu marshals between the two
// explicitly (see main.cu's build_padded_psf()/to_complex() helpers).
// ===========================================================================
void naive_inverse_cpu(const float* blurred, const float* psf_padded, float* out);
void wiener_cpu(const float* blurred, const float* psf_padded, float K, float* out);
void richardson_lucy_cpu(const float* blurred, const float* psf, float* estimate_inout,
                         int iterations, float* mse_curve_out /* [iterations], vs blurred, may be null */);

void shift_and_add_cpu(const float* lr_frames, const Shift* shifts, float* hr_out);
void ibp_refine_cpu(const float* lr_frames, const Shift* shifts, float* hr_estimate_inout,
                    int iterations, float* rms_curve_out /* [iterations], may be null */);
void bicubic_upscale_cpu(const float* lr, float* hr);

#endif // PROJECT_KERNELS_CUH
