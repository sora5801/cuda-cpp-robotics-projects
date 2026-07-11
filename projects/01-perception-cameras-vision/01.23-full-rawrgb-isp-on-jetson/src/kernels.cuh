// ===========================================================================
// kernels.cuh — interface & sensor/ISP model for project 01.23
//               Full RAW->RGB ISP: black level -> lens shading -> defect
//               correction -> white balance -> demosaic (MHC + bilinear) ->
//               color-correction matrix -> gamma, staged AND fused (1-4),
//               desktop-CUDA teaching core of the catalog's Jetson/Argus bullet
//
// SCOPING (CLAUDE.md section 5, hardware-dependent policy)
// -----------------------------------------------------------------------
// The catalog bullet asks for "Argus + custom CUDA stages" on Jetson. Argus
// is NVIDIA's libargus camera-capture API and Jetson ships a fixed-function
// hardware ISP (VIC/ISP engine) neither of which exist, nor can be linked
// or emulated, on the owner's desktop RTX 2080 SUPER. Per CLAUDE.md section
// 5's explicit rule for hardware-dependent projects, this project builds the
// DESKTOP-RUNNABLE TEACHING CORE: the complete classical RAW->RGB radiometric
// pipeline as pure CUDA C++, running on synthetic sensor data, with the real
// Jetson/Argus/libargus deployment path documented (never faked) in
// THEORY.md "Where this sits in the real world" and PRACTICE.md section 3.
//
// Role in the project
// -------------------
// The single contract every translation unit (main.cu, kernels.cu,
// reference_cpu.cpp) and the Python generator (../scripts/make_synthetic.py)
// agree on: image geometry, the Bayer/RAW10-in-uint16 convention, the sensor
// model (black level, lens shading polynomial, spectral crosstalk / CCM,
// illuminant gains), the committed defect list's slot count, and the
// Malvar-He-Cutler demosaic coefficient tables. Every numeric constant here
// has a "MUST MATCH ../scripts/make_synthetic.py" cross-reference exactly
// like sibling flagship 01.01's camera model (CLAUDE.md paragraph 12) — a
// generator/pipeline disagreement here is the single most common way a
// synthetic-data ISP project silently teaches the wrong lesson.
//
// THE EIGHT-STAGE PIPELINE (THEORY.md derives every step's physics/math)
// -----------------------------------------------------------------------
//   RAW10-in-uint16 mosaic (one sample/pixel, RGGB, values 0..1023 stored in
//   a wider container — the same "unpack narrow ADC codes into a roomier
//   integer" convention every real RAW10/RAW12 driver uses)
//     -> (1) BLACK LEVEL + saturation:      raw -> normalized [0,~1] float
//     -> (2) LENS SHADING correction:        divide by a radial polynomial
//            gain map V(r) = 1 + a2 r^2 + a4 r^4 (01.09's model, 2-term)
//     -> (3) DEFECTIVE PIXEL correction:     committed defect list, median
//            of same-Bayer-phase neighbors, __constant__-memory broadcast
//     -> (4) WHITE BALANCE:                  gray-world AND white-patch
//            estimators (GPU reduction), gains applied per Bayer phase
//     [(1)-(4) also ship as ONE FUSED kernel — see "Fusion economics" below]
//     -> (5) DEMOSAIC:                       Malvar-He-Cutler (gradient-
//            corrected linear demosaic) AND a bilinear baseline, both GPU
//            kernels with independent CPU twins, so the pipeline can MEASURE
//            (not assert) the quality gap this project's THEORY.md predicts
//     -> (6) COLOR-CORRECTION MATRIX:        3x3 CCM = crosstalk^-1, derived
//            by hand in THEORY.md from the documented synthetic sensor's
//            spectral crosstalk matrix
//     -> (7) GAMMA / sRGB encode:            the exact piecewise sRGB
//            transfer function, linear float -> 8-bit display RGB
//
// Fusion economics (extends 01.01's staged-vs-fused lesson)
// -----------------------------------------------------------------------
// Stages (1)-(4) are each a per-pixel MAP with no cross-thread dependency
// EXCEPT stage (3): a defect pixel's correction reads its same-phase
// neighbors' STAGE-(1)+(2)-corrected values. 01.01 fused three stages whose
// dependency was on materialized DATA (bilinear samples of a full remapped
// image); here the dependency is on a materialized FORMULA (black-level
// subtract + shading divide is 4 FLOPs, entirely a function of the raw
// sample and the pixel's own (x,y) — nothing another kernel computed). That
// makes fusion of (1)-(4) unusually cheap: the fused kernel recomputes those
// 4 FLOPs inline for up to 4 extra neighbor taps ONLY for the handful of
// pixels that are actually defective (~16 of the whole image) — every other
// pixel pays ZERO extra compute versus the staged path, while STILL avoiding
// three full-resolution intermediate buffers round-tripping through global
// memory (the write-then-read traffic 01.01's fused_kernel header derives in
// general). main.cu measures and prints both the analytic byte savings and
// the wall-clock kernel-time comparison, exactly as 01.01 does.
//
// IMAGE / SENSOR LAYOUTS (row-major throughout)
// -----------------------------------------------------------------------
//   RAW mosaic       (kRawW x kRawH): uint16_t[y*W+x], RGGB tiling (see
//                     bayer_phase_at() below), values in [0, kWhiteLevel].
//   NORMALIZED plane (same WxH): float[y*W+x], one sample per pixel, the
//                     "sensor domain" value used by stages (1)-(5); can
//                     exceed 1.0 after white-balance gain (documented
//                     headroom, THEORY.md "Numerical considerations").
//   RGB float image  (kRawW x kRawH x 3): float[(y*W+x)*3+c], c in
//                     {0=R,1=G,2=B} — demosaic's output through gamma's
//                     input; the ONLY stage that changes sample count per
//                     pixel (1 -> 3).
//   FINAL sRGB image (kRawW x kRawH x 3): uint8_t[(y*W+x)*3+c] — identical
//                     layout to a binary PPM (P6) pixel (01.01's convention).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>

// HD — "__host__ __device__" under nvcc, nothing under cl.exe. Same trick as
// 01.01's kernels.cuh: a handful of formulas here are genuinely SHARED
// "hardware facts" (the Bayer phase lookup, the lens-shading polynomial, the
// MHC coefficient tables) that main.cu's gates, kernels.cu's kernels, AND
// ../scripts/make_synthetic.py (independently, in Python) must all agree on
// bit-for-bit; per the twin-independence ruling (see reference_cpu.cpp's
// header) sharing exactly these — and ONLY these — formulas is permitted
// because they are DATA (documented physical/geometric facts), never "the
// algorithm under test".
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// SECTION 0 — geometry & RAW10 sensor constants. MUST MATCH the "MUST MATCH
// kernels.cuh" block in ../scripts/make_synthetic.py.
// ===========================================================================

// Raw sensor resolution. Small on purpose (CLAUDE.md section 5: desktop-
// runnable, tiny committed sample) while staying big enough that a 6x4
// color chart, a dedicated AWB card, and a textured region are all legible
// at >= 20x20 px per feature (see the layout block, section 1, below).
constexpr int kRawW = 160;
constexpr int kRawH = 120;

// RAW10-in-uint16: the ADC is a 10-bit sensor (0..1023 DN, "data numbers"),
// but every raw sample is stored in a 16-bit container — exactly how real
// camera drivers unpack MIPI CSI-2's packed RAW10 stream into something a
// CPU/GPU can index a byte at a time (PRACTICE.md section 1 details the
// packed-vs-unpacked wire format this abstracts over).
constexpr int kBitDepth    = 10;
constexpr int kWhiteLevel  = 1023;   // 2^10 - 1: brightest representable DN
constexpr int kBlackLevel  = 64;     // DN a photosite reads with ZERO light
                                     // (dark current + ADC offset — every
                                     // real sensor has this; THEORY.md "The
                                     // problem" derives where it comes from)
// Usable code range after black-level subtraction — the denominator that
// turns DN into the [0,1] normalized float domain stages (2)-(7) work in.
constexpr int kSatRange = kWhiteLevel - kBlackLevel;   // 959 DN

// ---------------------------------------------------------------------------
// bayer_phase_at — which of the FOUR RGGB roles a raw pixel plays. Unlike
// 01.01's 3-way bayer_channel_at() (R/G/B), this project's demosaic (MHC)
// needs to tell the two green sub-lattices apart: a green pixel in a RED
// row (Gr) has RED neighbors left/right, while a green pixel in a BLUE row
// (Gb) has BLUE neighbors left/right — the two cases use TRANSPOSED MHC
// kernels (section 3). RGGB tiling, x increasing rightward, y increasing
// downward (image convention):
//
//     row y=0 (even): R  Gr  R  Gr ...
//     row y=1 (odd):  Gb  B  Gb  B ...
//     row y=2 (even): R  Gr  R  Gr ...   (repeats)
//
// Returns: 0=R, 1=Gr (green, red row), 2=Gb (green, blue row), 3=B.
// ---------------------------------------------------------------------------
HD inline int bayer_phase_at(int x, int y)
{
    const bool even_row = (y & 1) == 0;
    const bool even_col = (x & 1) == 0;
    if (even_row && even_col)  return 0;   // R
    if (even_row && !even_col) return 1;   // Gr
    if (!even_row && even_col) return 2;   // Gb
    return 3;                              // B
}

// phase_to_wb_channel — collapse the 4-way Bayer phase to the 3-way white-
// balance channel index (0=R,1=G,2=B): Gr and Gb share ONE white-balance
// gain (both are physically the same green color filter; the ISP's WB stage
// does not distinguish the two green sub-lattices, only demosaic does).
HD inline int phase_to_wb_channel(int phase)
{
    if (phase == 0) return 0;              // R
    if (phase == 3) return 2;              // B
    return 1;                              // Gr or Gb -> G
}

// ===========================================================================
// SECTION 1 — synthetic scene layout constants. Every field below is used by
// BOTH ../scripts/make_synthetic.py (to draw the scene) and main.cu (to know
// where each gate should sample). "MUST MATCH" applies to every constant.
//
// Layout (a Macbeth-style 24-patch color chart + a dedicated AWB reference
// card + a hashed texture region + a dark neutral background), all inside
// the kRawW x kRawH canvas:
//
//   +----------------------------------------------+
//   | 6x4 color chart          |  hashed texture    |
//   | (kChartX0,kChartY0)      |  region             |
//   |                          |  (kTexX0,kTexY0)    |
//   +--------------------------+                     |
//   | AWB card (kCardX0,Y0)    |                     |
//   +----------------------------------------------+
//         (remaining canvas: dark neutral background)
// ===========================================================================
constexpr int kChartCols  = 6;
constexpr int kChartRows  = 4;
constexpr int kPatchSize  = 20;    // px, each chart patch is kPatchSize^2
constexpr int kPatchGap   = 1;     // px between patches
constexpr int kChartX0    = 4;
constexpr int kChartY0    = 4;
constexpr int kChartW     = kChartCols * kPatchSize + (kChartCols - 1) * kPatchGap;   // 125
constexpr int kChartH     = kChartRows * kPatchSize + (kChartRows - 1) * kPatchGap;   // 83
// Patch (row r, col c)'s pixel rectangle, r in [0,kChartRows), c in [0,kChartCols):
//   x in [kChartX0 + c*(kPatchSize+kPatchGap), + kPatchSize)
//   y in [kChartY0 + r*(kPatchSize+kPatchGap), + kPatchSize)
// Chart spans x in [4, 4+6*20+5*1) = [4,129), y in [4, 4+4*20+3*1) = [4,87).

// The dedicated AWB reference card: a large, bright, spatially uniform
// neutral region below the chart — the synthetic stand-in for a photographer
// holding up a white-balance card. Deliberately SEPARATE from the chart's
// own grayscale ramp (row 3, section 2): it is bigger (more pixels to
// average -> less noise-sensitive) and brighter (drives the white-patch/
// max-RGB estimator, which needs a genuine near-white region to lock onto).
constexpr int kCardX0 = 4, kCardY0 = 91, kCardW = 86, kCardH = 20;
constexpr unsigned char kCardSrgb8[3] = { 230, 230, 230 };

// Hashed texture region: 4x4-px blocks, each colored from a small fixed
// palette by the seeded PRNG (../scripts/make_synthetic.py). This is the
// ONLY high-spatial-frequency content in the scene — the color chart and
// AWB card are deliberately flat (they exist to be AVERAGED), but demosaic
// quality can only be measured where the true image actually has per-pixel
// color detail for MHC's gradient correction to earn its keep over bilinear
// (THEORY.md "The algorithm" derives why a flat scene cannot discriminate
// the two demosaicers at all).
constexpr int kTexX0 = 133, kTexY0 = 4, kTexW = 23, kTexH = 107;
constexpr int kTexBlockSize = 4;
constexpr int kTexPaletteN = 8;
// sRGB8 palette, row-major {R,G,B}: strong primaries/secondaries + white/black
// so adjacent blocks create real color EDGES for demosaic to resolve.
constexpr unsigned char kTexPalette[kTexPaletteN][3] = {
    {220, 50, 50}, {50, 220, 50}, {50, 50, 220}, {220, 220, 50},
    {220, 50, 220}, {50, 220, 220}, {230, 230, 230}, {30, 30, 30}
};

// Background fill (everywhere not covered by chart/card/texture): a dark,
// spatially flat neutral — contributes additional neutral-ish content to
// the whole-image gray-world average without adding new colors to reason
// about.
constexpr unsigned char kBackgroundSrgb8[3] = { 30, 30, 30 };

// ---------------------------------------------------------------------------
// The 24-patch color chart's reference sRGB8 values (row-major, patch index
// = r*kChartCols+c). Loosely modeled on the FAMILIES of patches a classic
// X-Rite ColorChecker carries (skin tones, primaries/secondaries, a
// grayscale ramp) for pedagogical familiarity — these are ILLUSTRATIVE
// numbers chosen for this project, NOT the certified X-Rite colorimetric
// values (which require a licensed spectral measurement to reproduce
// honestly; CLAUDE.md section 8: never fabricate precision this project did
// not measure). Row 3 IS the grayscale ramp (white -> black, 6 steps) —
// one chart does double duty as both the CCM gate's target table and the
// "gray ramp" the project brief asks for.
//
// Row 1 (index 6,7,8 = orange/red/orange-yellow) is the RED-HEAVY CROP used
// by the designed AWB-failure gate (main.cu): a rectangle covering exactly
// those three contiguous patches, whose average color is strongly
// red/orange-dominant — gray-world assumes the AVERAGE of what it sees is
// neutral gray, so averaging only warm patches is exactly the scene gray-
// world is known to get wrong (THEORY.md "Where this sits in the real
// world" names this failure mode explicitly).
// ---------------------------------------------------------------------------
constexpr int kChartN = kChartRows * kChartCols;   // 24
constexpr unsigned char kChartRefSrgb8[kChartN][3] = {
    // row 0 — skin/earth/foliage family
    {115, 82, 68}, {194, 150, 130}, {98, 122, 157}, {87, 108, 67}, {133, 128, 177}, {103, 189, 170},
    // row 1 — warm family (cols 0-2 = the red-heavy crop)
    {214, 126, 44}, {193, 60, 56}, {222, 158, 46}, {94, 60, 108}, {157, 188, 64}, {56, 61, 150},
    // row 2 — cool/primary/secondary family
    {56, 80, 152}, {70, 148, 73}, {60, 150, 175}, {188, 84, 150}, {231, 199, 31}, {52, 126, 145},
    // row 3 — grayscale ramp: white -> black
    {243, 243, 242}, {200, 200, 200}, {160, 160, 160}, {120, 120, 120}, {85, 85, 85}, {52, 52, 52},
};

// Red-heavy crop rectangle (patches (1,0),(1,1),(1,2) — row index 1, cols 0-2):
constexpr int kRedCropX0 = kChartX0;                                       // 4
constexpr int kRedCropY0 = kChartY0 + 1 * (kPatchSize + kPatchGap);        // 25
constexpr int kRedCropW  = 3 * kPatchSize + 2 * kPatchGap;                 // 62
constexpr int kRedCropH  = kPatchSize;                                     // 20

// ===========================================================================
// SECTION 2 — the synthetic sensor's optical/electronic model. Every
// constant below MUST MATCH ../scripts/make_synthetic.py's forward model.
// ===========================================================================

// ---- Lens shading (vignetting): a RADIAL POLYNOMIAL gain map, the SAME
// functional family as 01.09's photometric-vignetting-calibration-kernels
// (V(r) = 1 + a2 r^2 + a4 r^4 + a6 r^6), reduced here to its first two even
// terms (a2, a4) — a 2-term truncation of 01.09's 3-term model, sufficient
// to produce a clearly visible, physically plausible falloff without a
// third free parameter this project has no independent gate for. r is
// normalized by kShadeRNorm (the image's own half-diagonal) so r=1 at the
// farthest corner from the optical center. ----------------------------------
constexpr float kShadeCx = (kRawW - 1) * 0.5f;
constexpr float kShadeCy = (kRawH - 1) * 0.5f;
// Half-diagonal in pixels, precomputed by hand (double precision, rounded to
// float here) for the same host/device bit-identity reason 01.01's
// kRectCos/kRectSin are precomputed rather than calling sqrtf() twice
// independently on host and device.
constexpr float kShadeRNorm = 99.30005f;   // hypot(79.5, 59.5), double precision then rounded
constexpr float kShadeA2 = -0.35f;         // V(1) = 1 - 0.35 + 0.10 = 0.75 -> 25% corner falloff
constexpr float kShadeA4 =  0.10f;
// Correction floor (01.09's kGainFloor precedent): guards 1/V(r) against a
// division blow-up if a future edit ever pushes V(r) near zero at the
// corner; V(1)=0.75 here so this floor is inactive on the committed scene,
// documented defensively (THEORY.md "Numerical considerations").
constexpr float kShadeGainFloor = 0.35f;

// shading_gain_at — V(r) at raw pixel (x,y), the MULTIPLICATIVE light loss
// the lens+microlens stack applies at that photosite. Shared HD "hardware
// fact" (see file header) — the generator applies this FORWARD (multiply),
// the ISP's shading-correction stage applies its reciprocal.
HD inline float shading_gain_at(int x, int y)
{
    const float dx = static_cast<float>(x) - kShadeCx;
    const float dy = static_cast<float>(y) - kShadeCy;
    const float r  = sqrtf(dx * dx + dy * dy) / kShadeRNorm;
    const float r2 = r * r;
    return 1.0f + kShadeA2 * r2 + kShadeA4 * r2 * r2;
}

// ---- Spectral crosstalk matrix M: row c = how strongly the c-th color
// filter (R, G, or B) responds to R/G/B light landing on it. Real color
// filters are not perfectly narrowband, so each photosite leaks some signal
// from its neighbors' colors (THEORY.md "The math" derives a CCM from this
// matrix by hand). Each row sums to 1.0 BY CONSTRUCTION: a perfectly
// spectrally FLAT (neutral gray) input reproduces itself at the sensor —
// crosstalk alone introduces NO color cast for neutral scenes, so any cast
// this project measures is attributable ENTIRELY to the illuminant gains
// below, not to crosstalk (a deliberate, documented pedagogical separation
// of the two physical effects). Stored as NINE NAMED SCALARS rather than a
// float[9]: a namespace-scope constexpr SCALAR is safely usable from both
// host and device code under nvcc (it is folded to an immediate at each use
// site, no runtime storage needed); an ARRAY read via pointer+index from
// device code needs real device-visible storage, which is why the (much
// larger) MHC tables in section 3 are handled differently — see that
// section's comment for the full explanation. -------------------------------
constexpr float kM00 = 0.72f, kM01 = 0.22f, kM02 = 0.06f;
constexpr float kM10 = 0.10f, kM11 = 0.78f, kM12 = 0.12f;
constexpr float kM20 = 0.06f, kM21 = 0.20f, kM22 = 0.74f;

// ---- The Color-Correction Matrix: CCM = M^-1, derived BY HAND in
// THEORY.md's "The math" section (full 3x3 inversion shown, not just
// asserted). Because M's rows sum to 1, M^-1's rows sum to 1 automatically
// (a row-sum-1 matrix's inverse shares the property: M*ones=ones implies
// ones=M^-1*ones) — NO separate white-point renormalization step is needed,
// unlike the general CCM-derivation case most real ISP tuning pipelines
// must handle (THEORY.md names this explicitly). -----------------------------
constexpr float kCCM00 =  1.44817f, kCCM01 = -0.39476f, kCCM02 = -0.05340f;
constexpr float kCCM10 = -0.17487f, kCCM11 =  1.38534f, kCCM12 = -0.21047f;
constexpr float kCCM20 = -0.07016f, kCCM21 = -0.34241f, kCCM22 =  1.41257f;

// ccm_apply_at — shared HD "hardware fact" (see file header): the CCM's 3x3
// matrix-vector product, sensor-domain (r,g,b) -> linear-sRGB (or,og,ob).
// Written with named scalars (not a loop over an array) precisely so it
// compiles identically, with zero device-storage concerns, in both
// kernels.cu's device kernel and reference_cpu.cpp's INDEPENDENT host
// caller — sharing this five-line formula is the same judgment call 01.01
// makes for distort_forward() (it is the documented camera-model DATA, not
// "the algorithm under test"; main.cu's ccm_color_chart gate is this
// project's independent check that does not merely trust this formula).
HD inline void ccm_apply_at(float r, float g, float b, float& out_r, float& out_g, float& out_b)
{
    out_r = kCCM00 * r + kCCM01 * g + kCCM02 * b;
    out_g = kCCM10 * r + kCCM11 * g + kCCM12 * b;
    out_b = kCCM20 * r + kCCM21 * g + kCCM22 * b;
}

// ---- Illuminant gains: the per-channel multiplicative SIGNAL a neutral
// (equal-energy) reflectance produces at the sensor under each light
// source, BEFORE crosstalk. D65 (daylight, ~6500K) is this project's
// NEUTRAL reference (gain 1,1,1 by definition/convention — the white-
// balance state that needs no correction). Tungsten (~2856K) is
// red-heavy/blue-poor, an illustrative ratio in the right ballpark for a
// warm incandescent source (not a spectroradiometer measurement — CLAUDE.md
// section 8 honesty). The TRUE AWB correction gain a perfect white-balance
// algorithm should recover is the reciprocal, normalized so the green
// channel keeps unit gain (the convention essentially every real camera
// ISP uses, since luminance is mostly carried by green). ---------------------
constexpr float kIlluminantD65Gain[3]      = { 1.00f, 1.00f, 1.00f };
constexpr float kIlluminantTungstenGain[3] = { 1.42f, 1.00f, 0.53f };
constexpr float kTrueAwbGainD65[3]      = { 1.00f, 1.00f, 1.00f };
// 1/1.42, 1/1.00, 1/0.53 :
constexpr float kTrueAwbGainTungsten[3] = { 0.70423f, 1.00000f, 1.88679f };

// ---- sRGB transfer function (the "gamma" stage) — the EXACT piecewise
// function from the sRGB standard, encoding direction (scene-linear [0,1]
// -> display-encoded [0,1]); the technically correct name is the sRGB OETF
// (opto-electronic transfer function) since "EOTF" formally names the
// OPPOSITE (display-encoded -> linear) direction — THEORY.md "The math"
// spells out this common naming mix-up honestly. Shared HD "hardware fact"
// (both the ISP's gamma stage and the generator's ground-truth renderer use
// the identical function). ---------------------------------------------------
HD inline float srgb_encode(float linear)
{
    linear = linear < 0.0f ? 0.0f : (linear > 1.0f ? 1.0f : linear);   // clip: the one clamp point (see file header)
    if (linear <= 0.0031308f) return 12.92f * linear;
    return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
}
// srgb_decode — the inverse (display-encoded -> scene-linear), used by the
// generator to turn the chart's documented sRGB8 reference values into the
// linear-light reflectances the forward sensor model actually needs.
HD inline float srgb_decode(float s)
{
    s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
    if (s <= 0.04045f) return s / 12.92f;
    return powf((s + 0.055f) / 1.055f, 2.4f);
}

// ---- Noise model: shot (signal-dependent) + read (constant) noise, the
// standard two-term sensor noise model (THEORY.md "The problem" derives
// both terms physically). sigma^2(signal_dn) = kReadNoiseDn^2 +
// kShotNoiseK*signal_dn, signal_dn measured ABOVE black level. -------------
constexpr float kReadNoiseDn = 2.0f;
constexpr float kShotNoiseK  = 0.02f;

// ===========================================================================
// SECTION 3 — Malvar-He-Cutler demosaic coefficient tables (Malvar, He &
// Cutler, "High-Quality Linear Interpolation for Demosaicing of Bayer-
// Patterned Color Images", 2004). Coefficients as commonly reproduced in the
// demosaicing literature; verify against the original paper before use
// outside this teaching context (CLAUDE.md section 8 honesty about
// sourcing). Every table sums to 8 (so a flat/constant scene demosaics back
// to itself exactly once divided by 8 — THEORY.md verifies this by hand for
// all four tables and explains the gradient-correction intuition behind the
// negative "Laplacian" taps).
//
// Kernel selection by Bayer phase (bayer_phase_at, section 0):
//   phase R  (0): missing G -> kMhcG,    missing B -> kMhcDiag
//   phase Gr (1): missing R -> kMhcA,    missing B -> kMhcB
//   phase Gb (2): missing B -> kMhcA,    missing R -> kMhcB
//   phase B  (3): missing G -> kMhcG,    missing R -> kMhcDiag
// Every table is a 5x5 stencil, row-major, offsets dy,dx in [-2,2] mapped to
// table index (dy+2)*5+(dx+2); the tap-gathering, border-clamping, and
// phase-selection LOGIC around these tables is the "algorithm under test"
// and is retyped INDEPENDENTLY in kernels.cu (device) and reference_cpu.cpp
// (host) — only the 25-number TABLES themselves (published data) are common
// to both, matching this project's ccm_apply_at carve-out.
//
// Device-storage note (why these arrays are declared TWICE in this project):
// a namespace-scope constexpr SCALAR (kCCM00 etc., above) is safely usable
// from device code under nvcc — it is folded to an immediate wherever it
// appears, no runtime storage required. A 25-element ARRAY read through a
// pointer at a runtime-computed offset is a different story: it needs real,
// device-visible STORAGE, which a plain host-side constexpr array (as
// declared below, for reference_cpu.cpp's benefit) does not reliably
// provide under nvcc. So kernels.cu additionally defines its OWN
// `__constant__`-qualified copies of these four tables (same 25 numbers
// each, side by side with these for easy comparison) purely so the GPU
// kernels have guaranteed device storage to read from — see that file's
// section 3 for the device-side declarations and CLAUDE.md's memory-
// hierarchy teaching goal (constant memory: every thread in a warp reads
// the SAME table, the textbook case constant memory's broadcast is for).
// ===========================================================================
constexpr int kMhcTaps = 25;

// G at R or at B: a plus-shaped bilinear-of-G term plus a Laplacian-of-the-
// OWN-channel correction (the two -1's at distance 2 use the CENTER's own
// native color, sampled at the same phase two pixels away).
constexpr float kMhcG[kMhcTaps] = {
     0,  0, -1,  0,  0,
     0,  0,  2,  0,  0,
    -1,  2,  4,  2, -1,
     0,  0,  2,  0,  0,
     0,  0, -1,  0,  0,
};
// R at Gr / B at Gb — HORIZONTAL emphasis (the target color's true samples
// sit directly left/right of a Gr/Gb pixel; the row-2 taps carry the bulk
// of the weight).
constexpr float kMhcA[kMhcTaps] = {
     0,    0,   0.5f, 0,    0,
     0,   -1,   0,   -1,    0,
    -1,    4,   5,    4,   -1,
     0,   -1,   0,   -1,    0,
     0,    0,   0.5f, 0,    0,
};
// R at Gb / B at Gr — VERTICAL emphasis, the exact transpose of kMhcA (the
// target color's true samples sit directly above/below instead).
constexpr float kMhcB[kMhcTaps] = {
     0,    0,  -1,   0,    0,
     0,   -1,   4,  -1,    0,
     0.5f, 0,   5,   0,    0.5f,
     0,   -1,   4,  -1,    0,
     0,    0,  -1,   0,    0,
};
// R at B / B at R — DIAGONAL: the target color's true samples sit only at
// the four diagonal neighbors, so this kernel is symmetric under 90-degree
// rotation (unlike kMhcA/kMhcB, which are each other's rotation).
constexpr float kMhcDiag[kMhcTaps] = {
     0,     0,   -1.5f, 0,     0,
     0,     2,    0,    2,     0,
    -1.5f,  0,    6,    0,    -1.5f,
     0,     2,    0,    2,     0,
     0,     0,   -1.5f, 0,     0,
};

// ===========================================================================
// SECTION 4 — committed defect-pixel list. A REAL sensor ships with a
// factory-measured defect map (stuck/hot/dead photosites) loaded from
// calibration storage at boot, not compiled into firmware — this project
// mirrors that realism: the list lives in data/sample/defect_list.csv (x,y),
// loaded at RUNTIME by main.cu, then broadcast to the GPU via
// __constant__ memory (kernels.cu) — the ONE stage in this pipeline that
// uses constant memory, contrasted with every other stage's global-memory
// traffic (THEORY.md "The GPU mapping" names why constant memory is the
// right choice here: every thread in the kernel reads the SAME small,
// read-only array, which is exactly what constant memory's broadcast cache
// is built for).
// ===========================================================================
constexpr int kMaxDefects = 64;   // upper bound on the committed list's length

// upload_defect_list — host wrapper (defined in kernels.cu) that copies the
// runtime-loaded defect coordinates into the __constant__ device array via
// cudaMemcpyToSymbol. xs/ys: HOST arrays of length count (count <= kMaxDefects).
void upload_defect_list(const int* xs, const int* ys, int count);

// ===========================================================================
// GPU launch wrappers (kernels.cu). Every wrapper computes its own launch
// geometry, launches, and calls CUDA_CHECK_LAST_ERROR (main.cu never writes
// <<<...>>> directly — CLAUDE.md paragraph 6's orchestration/parallelism
// split, same discipline as every project in this repo).
// ===========================================================================

// ---- STAGED path: one kernel per stage, each writing a full W*H
// intermediate to global memory (the traffic the FUSED kernel below avoids).
// d_raw: kRawW*kRawH uint16_t (RAW10-in-uint16). d_out: kRawW*kRawH float
// (normalized [0,~1.2] "sensor domain", one sample per pixel).
void launch_black_level(const uint16_t* d_raw, float* d_out, int W, int H);
void launch_lens_shading(const float* d_in, float* d_out, int W, int H);
// d_defect_count: number of valid entries in the just-uploaded constant list.
void launch_defect_correct(const float* d_in, float* d_out, int W, int H, int defect_count);
void launch_white_balance(const float* d_in, float* d_out, int W, int H,
                          float gain_r, float gain_g, float gain_b);

// ---- FUSED path: stages (1)-(4) in ONE kernel (file header "Fusion
// economics"). Same inputs/outputs as the staged chain end to end.
void launch_fused_bl_shading_defect_wb(const uint16_t* d_raw, float* d_out, int W, int H,
                                       int defect_count, float gain_r, float gain_g, float gain_b);

// ---- AWB statistics: a DETERMINISTIC (no atomics) two-level block-tree
// reduction over the shading+defect-corrected mosaic (01.01's
// normalize_block_stats precedent, EXTENDED to mix two reduction operators
// per lane: sum for gray-world, max for white-patch), producing PER-PHASE
// (R, G-combined, B) sum+max. d_in: kRawW*kRawH float (post stage 3, PRE
// white balance — the WB stage below is what CONSUMES these stats).
// d_block_sum3/d_block_max3 OUT: num_blocks*3 each. Per-phase pixel COUNTS
// are not reduced at runtime: for the even kRawW x kRawH used throughout
// this project, RGGB geometry fixes them exactly (R = B = W*H/4, G = W*H/2
// — every 2x2 tile contributes exactly one R, one B, two G), so
// awb_gains_from_stats_cpu/launch_awb_finalize compute them from W*H
// directly (kernels.cu/reference_cpu.cpp document the formula at the call
// site) rather than spending a reduction lane on a compile-time-known value.
void launch_awb_stats_block(const float* d_in, int W, int H,
                            double* d_block_sum3, float* d_block_max3,
                            int num_blocks);
// Finalize: <<<1,1>>>, sequential over the (small) block partials — the
// same "no atomics anywhere" determinism story as 01.01's normalize stage.
// gray_gain3/white_gain3 OUT: 3 floats each (R,G,B), G-normalized to 1.0.
void launch_awb_finalize(const double* d_block_sum3, const float* d_block_max3,
                         int num_blocks, int W, int H,
                         float* d_gray_gain3, float* d_white_gain3);

// ---- Demosaic: mosaic (post WB) -> 3-channel RGB. Both operate on the
// SAME input so their outputs are directly comparable (main.cu's PSNR gate).
void launch_demosaic_bilinear(const float* d_mosaic, float* d_rgb, int W, int H);
void launch_demosaic_mhc(const float* d_mosaic, float* d_rgb, int W, int H);

// ---- Color-correction matrix: per-pixel 3x3 matmul, sensor-RGB -> linear sRGB.
void launch_ccm_apply(const float* d_rgb_in, float* d_rgb_out, int W, int H);

// ---- Gamma encode: linear float RGB -> 8-bit display sRGB (the final artifact).
void launch_gamma_encode(const float* d_rgb_linear, unsigned char* d_rgb_srgb8, int W, int H);

// ===========================================================================
// CPU references (reference_cpu.cpp) — INDEPENDENT reimplementations of
// every launcher above (twin-independence ruling, see that file's header):
// same math, plain nested loops, "_cpu" suffix. bayer_phase_at(),
// shading_gain_at(), the kMhc* tables, srgb_encode/decode, and the sensor
// constants above are the deliberate, documented SHARED exception (they are
// data, not the algorithm under test); the neighbor-clamping, the median
// search, the reduction arithmetic, the demosaic stencil application, the
// CCM matmul, and the fused-kernel re-derivation are all typed a SECOND
// time, independently, in this file.
// ===========================================================================
void black_level_cpu(const uint16_t* raw, float* out, int W, int H);
void lens_shading_cpu(const float* in, float* out, int W, int H);
void defect_correct_cpu(const float* in, float* out, int W, int H,
                        const int* defect_x, const int* defect_y, int defect_count);
void white_balance_cpu(const float* in, float* out, int W, int H,
                       float gain_r, float gain_g, float gain_b);
void fused_bl_shading_defect_wb_cpu(const uint16_t* raw, float* out, int W, int H,
                                    const int* defect_x, const int* defect_y, int defect_count,
                                    float gain_r, float gain_g, float gain_b);
void awb_stats_cpu(const float* in, int W, int H, double sum3[3], float max3[3]);
// Per-phase pixel counts are DERIVED from W*H (RGGB geometry, see kernels.cuh
// section "AWB statistics"), not measured: countR=countB=W*H/4, countG=W*H/2.
void awb_gains_from_stats_cpu(const double sum3[3], const float max3[3], int W, int H,
                              float gray_gain3[3], float white_gain3[3]);
void demosaic_bilinear_cpu(const float* mosaic, float* rgb, int W, int H);
void demosaic_mhc_cpu(const float* mosaic, float* rgb, int W, int H);
void ccm_apply_cpu(const float* rgb_in, float* rgb_out, int W, int H);
void gamma_encode_cpu(const float* rgb_linear, unsigned char* rgb_srgb8, int W, int H);

#endif // PROJECT_KERNELS_CUH
