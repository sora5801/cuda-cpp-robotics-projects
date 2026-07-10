// ===========================================================================
// kernels.cuh — interface & single-source contract for project 29.05
//               Ultrasound: GPU beamforming (plane-wave delay-and-sum B-mode)
//               Milestone 1 of the catalog bullet "Ultrasound: GPU
//               beamforming, elastography, image-based servoing" — the
//               other two milestones ship documented-only (README §13,
//               THEORY.md "Where this sits in the real world").
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (orchestration: phantom loading, channel-data
// synthesis, artifacts, verification gates), kernels.cu (the GPU beamforming
// pipeline), and reference_cpu.cpp (the CPU oracle twins of that pipeline).
// Everything all three must agree on — array geometry, pulse/sampling
// parameters, the imaging grid, the channel-data memory layout, and the FIR
// low-pass design — is defined HERE, once (CLAUDE.md §12).
//
// THE PHYSICAL SCENARIO IN ONE PARAGRAPH (THEORY.md derives all of this):
// A linear array of kNumElements piezo elements fires ONE unsteered PLANE
// WAVE straight into the tissue (the simplest transmit scheme used by real
// "ultrafast" ultrasound — README "Prior art"). Point scatterers in a
// SYNTHETIC phantom (wire targets + a high-scattering inclusion region +
// many background speckle scatterers — data/README.md) each reflect a
// Gaussian-windowed sinusoidal pulse back to every element with a two-way
// time-of-flight set by geometry. main.cu synthesizes this "channel data"
// (one RF trace per element) directly from the phantom, in FP32, on the
// HOST — it is the project's synthetic SENSOR, not the taught algorithm.
// The taught algorithm is what turns that channel data into an image:
//   1. DAS  (delay-and-sum): reconstruct EVERY pixel of a 2-D image
//      independently by, for each element, computing the exact two-way
//      delay to that pixel, linearly interpolating the channel trace at
//      that delay, weighting by an f-number-limited Hann receive
//      apodization, and averaging over the active aperture. This is
//      "software beamforming" — the way ultrafast plane-wave imaging (and
//      this project) forms images, contrasted with classical scanline
//      beamforming in README/THEORY.
//   2. Envelope detection: the beamformed RF image still oscillates at the
//      carrier; per depth COLUMN we quadrature-demodulate (mix with
//      cos/sin at fc, referenced to each pixel's own two-way arrival time)
//      then low-pass FIR-filter along depth to recover the baseband I/Q,
//      whose magnitude is the envelope. (RATIFIED: Hilbert-free — see
//      THEORY.md "The algorithm" for why this is the correct, standard
//      substitute for a Hilbert transform under the narrowband assumption
//      the pulse satisfies.)
//   3. Log compression: 20*log10(envelope / max) mapped into a documented
//      display dynamic range — the actual B-mode image.
// Verification is BOTH a GPU-vs-CPU tolerance gate at each of these three
// stages AND four analytic/known-truth gates (wire localization, measured
// vs. derived resolution, inclusion contrast, closed-form delay sanity) —
// see main.cu's header comment and THEORY.md "How we verify correctness".
//
// CHANNEL-DATA MEMORY LAYOUT — float[kNumElements][kNumSamples], row-major:
//     offset(e, s) = e * kNumSamples + s
//   with e = element index (0..kNumElements-1), s = fast-time sample index
//   (0..kNumSamples-1, one RF trace per element, kSamplingFreqHz Hz).
//
// IMAGE MEMORY LAYOUT — float[kImageNz][kImageNx], row-major:
//     offset(iz, ix) = iz * kImageNx + ix
//   with iz = depth (axial) pixel index, 0 = SHALLOWEST (kImageZMinM);
//        ix = lateral pixel index, 0 = LEFT edge (kImageXMinM).
//   Row-major with iz slowest matches a top-to-bottom B-mode display and
//   the PGM writer in main.cu (rows = depth, like 03.01's range axis).
//
// ARRAY GEOMETRY — element e sits at lateral position
//     x_e = (e - (kNumElements - 1)/2) * kElementPitchM        (meters)
//   i.e. a centered linear array, aperture kApertureM wide. This ONE-LINE
//   formula is the project's single most-duplicated piece of physics
//   (channel synthesis, the DAS kernel, the DAS CPU oracle, and the delay
//   sanity check in main.cu each restate it) — CLAUDE.md's "deliberate,
//   documented duplication" rather than a shared __host__ __device__
//   helper, because reference_cpu.cpp is built by cl.exe with NO CUDA
//   headers (see the kernels.cu/reference_cpu.cpp file headers) and every
//   restatement is a one-liner cheap enough to keep byte-identical by eye.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>   // std::sin/cos/exp — used by the shared HOST-only helper below

// ===========================================================================
// PHYSICAL & ARRAY CONSTANTS — the single source of truth for channel-data
// synthesis, both beamforming paths (GPU + CPU), and every resolution
// formula quoted in README.md/THEORY.md. A 64-element, 0.3 mm-pitch, 5 MHz
// linear array is a plausible small-footprint research/teaching probe
// (comparable in spirit to a Verasonics L11-4v-class array — README "Prior
// art"); values are chosen for round, checkable derived numbers.
// ===========================================================================
constexpr float kSoundSpeedMps  = 1540.0f;   // c: speed of sound in soft tissue (the textbook
                                              // "tissue convention" value — THEORY.md "The problem")
constexpr int   kNumElements    = 64;        // linear array element count
constexpr float kElementPitchM  = 0.30e-3f;  // center-to-center element spacing (m) = 0.3 mm
constexpr float kApertureM      = (kNumElements - 1) * kElementPitchM;  // full aperture width (m), ~18.9 mm

constexpr float kCenterFreqHz   = 5.0e6f;    // fc: transducer/pulse center frequency (Hz) = 5 MHz
constexpr float kSamplingFreqHz = 40.0e6f;   // fs: channel-data ADC sample rate (Hz) = 40 MHz
                                              // (8x fc — comfortable oversampling for accurate delay
                                              // interpolation; THEORY.md "Numerical considerations")
constexpr float kWavelengthM    = kSoundSpeedMps / kCenterFreqHz;   // lambda = c/fc (m), ~0.308 mm

constexpr float kPulseCycles    = 2.5f;      // nominal number of carrier cycles in the transmit pulse
                                              // (a short, well-resolved imaging pulse — THEORY.md)
constexpr float kPulseDurationS = kPulseCycles / kCenterFreqHz;     // nominal pulse duration (s)
constexpr float kPulseSigmaS    = kPulseDurationS / 4.0f;           // Gaussian ENVELOPE std-dev (s):
                                              // chosen so the pulse's +/-2 sigma span covers roughly
                                              // the nominal duration above (THEORY.md derives the
                                              // resulting axial resolution from kPulseCycles directly)

// Channel-data trace length. Must cover every two-way delay the DAS kernel
// can address (on-axis AND the longest off-axis receive path at max image
// depth) plus a few pulse-widths of margin so a scatterer near the far edge
// of the window is not truncated. See main.cu's [info] line for the exact
// covered depth/time this buys; kNumSamples is intentionally generous.
constexpr int kNumSamples = 2048;

// ===========================================================================
// IMAGING GRID — the pixel-parallel DAS output. Odd kImageNx gives an exact
// on-axis pixel column at x=0 (the wire-target cross pattern is centered
// there — data/README.md), which keeps the delay-sanity and PSF checks in
// main.cu exact rather than "nearest pixel". kImageDzM is chosen well below
// the Nyquist spacing the envelope stage's axial quadrature demodulation
// needs (THEORY.md "Numerical considerations" derives the c/(4*fc) bound
// and shows this grid clears it with margin).
// ===========================================================================
constexpr float kImageXMinM = -9.6e-3f;   // lateral field of view: left edge (m)
constexpr float kImageXMaxM =  9.6e-3f;   // lateral field of view: right edge (m)
constexpr float kImageZMinM = 10.0e-3f;   // axial field of view: shallowest depth (m) — kept away
                                          // from z=0 so t_tx=z/c and the f-number aperture formula
                                          // never see a singularity (THEORY.md "Numerical considerations")
constexpr float kImageZMaxM = 30.0e-3f;   // axial field of view: deepest depth (m)
constexpr int   kImageNx = 257;           // lateral pixel count (odd: exact x=0 column)
constexpr int   kImageNz = 801;           // axial pixel count
constexpr float kImageDxM = (kImageXMaxM - kImageXMinM) / (kImageNx - 1);  // lateral pixel pitch (m)
constexpr float kImageDzM = (kImageZMaxM - kImageZMinM) / (kImageNz - 1);  // axial pixel pitch (m)

constexpr float kFNumber = 1.5f;   // receive f-number: active_half_aperture(z) = z / (2*kFNumber)
                                    // — THEORY.md "The algorithm" derives why this single knob sets
                                    // BOTH the sidelobe/grating-lobe trade AND (via kLateralResM
                                    // below) the lateral resolution, independent of depth.

// ---------------------------------------------------------------------------
// Derived, DOCUMENTED teaching numbers — the formulas README/THEORY quote
// and the numbers the resolution-measurement gate (main.cu) checks the
// simulated point-spread function against. Both are FIRST-ORDER estimates
// (THEORY.md "The math" derives each); main.cu measures the real thing and
// reports the ratio honestly rather than asserting the formula is exact.
// ---------------------------------------------------------------------------
constexpr float kAxialResM   = kPulseCycles * kWavelengthM / 2.0f;  // "spatial pulse length / 2"
constexpr float kLateralResM = kWavelengthM * kFNumber;             // lambda * F# (diffraction limit
                                                                     // of an f-number-apodized aperture)

// ===========================================================================
// ENVELOPE-DETECTION FIR LOW-PASS — a small, documented, Hann-windowed-sinc
// filter applied ALONG DEPTH (per lateral column) after quadrature mixing,
// to reject the 2*fc image term and keep only the baseband envelope
// (THEORY.md "The algorithm" derives the cutoff choice). kFirRadius=8 means
// 17 taps — enough stopband rejection to matter, small enough to read the
// whole filter in one screen (CLAUDE.md §1: teaching beats cleverness).
// build_lowpass_fir_taps() is the ONE place the tap formula lives: it is a
// plain host function (no CUDA syntax) so BOTH kernels.cu's launcher
// (uploads the taps into __constant__ memory once) and reference_cpu.cpp
// (fills a local array every call — 17 floats, too cheap to bother caching)
// call the exact same code — no duplication needed here, unlike the
// __global__ kernel bodies themselves (see the file header above).
// ===========================================================================
constexpr int kFirRadius = 8;                    // filter half-width (taps beyond the center)
constexpr int kFirTaps   = 2 * kFirRadius + 1;    // = 17
constexpr float kLpCutoffHz = kCenterFreqHz / 2.0f;  // reject the ~2*fc mixing image, pass the
                                                      // envelope's much narrower baseband bandwidth

// build_lowpass_fir_taps — fill taps[kFirTaps] with a Hann-windowed-sinc
// low-pass FIR, cutoff kLpCutoffHz, referenced to the "sample rate" the
// depth axis implies when treated as a two-way-time axis (fs_z = c /
// (2*dz) — THEORY.md derives this mapping). Normalized to unity DC gain
// (sum(taps) == 1) so the filter does not rescale the envelope's amplitude.
// Pure double-precision host arithmetic (filter DESIGN, not the hot path;
// precision here is free and avoids compounding rounding across 17 taps).
inline void build_lowpass_fir_taps(float* taps /* [kFirTaps] */)
{
    const double kPiD = 3.14159265358979323846;
    const double fs_z = static_cast<double>(kSoundSpeedMps) / (2.0 * static_cast<double>(kImageDzM));
    const double fc_norm = static_cast<double>(kLpCutoffHz) / fs_z;   // normalized cutoff, cycles/sample

    double sum = 0.0;
    for (int n = -kFirRadius; n <= kFirRadius; ++n) {
        // Ideal (infinite) low-pass impulse response, the classic sinc kernel;
        // the n==0 case is the removable singularity sin(x)/x -> 1 at x=0.
        const double sinc_val = (n == 0) ? (2.0 * fc_norm)
                                          : (std::sin(2.0 * kPiD * fc_norm * n) / (kPiD * n));
        // Hann taper so the TRUNCATED (17-tap) filter does not ring badly —
        // the window goes to (nearly) zero at the tap array's edges.
        const double w = 0.5 + 0.5 * std::cos(kPiD * static_cast<double>(n) / (kFirRadius + 1));
        const double h = sinc_val * w;
        taps[n + kFirRadius] = static_cast<float>(h);
        sum += h;
    }
    for (int i = 0; i < kFirTaps; ++i)
        taps[i] = static_cast<float>(taps[i] / sum);   // renormalize: unity DC gain
}

// ===========================================================================
// LOG COMPRESSION / DISPLAY
// ===========================================================================
constexpr float kDynamicRangeDb = 50.0f;   // display dynamic range: dB image is clamped to
                                            // [-kDynamicRangeDb, 0] before byte-mapping to the PGM

// ===========================================================================
// PHANTOM CROSS-CHECK CONSTANTS — the inclusion region's geometry. Loaded
// (and cross-checked) from data/sample/array_params.csv at run time
// (main.cu, mirroring 03.01's radar_params.csv cross-check): the phantom
// generator (scripts/make_synthetic.py) embeds these SAME numbers in its
// own source (documented cross-language duplication, CLAUDE.md §12) — a
// drift between the two is a loud, early SCENARIO: MISMATCH failure rather
// than a silently wrong contrast gate.
// ===========================================================================
constexpr float kInclusionCenterXM = -6.0e-3f;   // inclusion disk center, lateral (m)
constexpr float kInclusionCenterZM = 15.0e-3f;   // inclusion disk center, depth (m)
constexpr float kInclusionRadiusM  =  2.5e-3f;   // inclusion disk radius (m)

// ===========================================================================
// GPU launchers (defined in kernels.cu). Every one owns its own launch
// configuration and post-launch CUDA_CHECK_LAST_ERROR call (CLAUDE.md §6.1
// rule 7); main.cu calls only these, never a __global__ kernel directly.
// All pointer parameters are DEVICE pointers unless documented "HOST".
// ===========================================================================

// launch_das — Stage 1 (the project's star kernel): delay-and-sum
// beamform every pixel of the kImageNz x kImageNx image independently from
// the channel data. One thread per PIXEL (the classic pixel-parallel DAS
// mapping — kernels.cu's header comment derives why this, not
// one-thread-per-element, is the natural GPU mapping).
//   d_channel  : DEVICE [kNumElements*kNumSamples] RF channel data (see
//                the memory-layout note above).
//   d_rf_image : DEVICE [kImageNz*kImageNx] OUT — the real-valued
//                beamformed RF image (still oscillating at the carrier;
//                envelope detection is the next stage).
void launch_das(const float* d_channel, float* d_rf_image);

// launch_quadrature_demod — Stage 2a: mix the RF image with cos/sin at fc,
// referenced to each pixel's own on-axis two-way arrival time (a pointwise
// MAP; kernels.cu explains the phase formula). Produces the raw (unfiltered)
// I/Q pair every pixel needs before the low-pass stage below.
//   d_rf_image        : DEVICE [kImageNz*kImageNx] IN — launch_das's output.
//   d_i_raw, d_q_raw   : DEVICE [kImageNz*kImageNx] OUT — raw in-phase /
//                        quadrature components (pre low-pass).
void launch_quadrature_demod(const float* d_rf_image, float* d_i_raw, float* d_q_raw);

// launch_envelope_lowpass — Stage 2b: FIR-low-pass filter d_i_raw/d_q_raw
// ALONG DEPTH (per lateral column — a 1-D STENCIL, kFirRadius neighbors
// each side, edge-clamped) to reject the ~2*fc mixing image, then take the
// magnitude sqrt(I^2+Q^2) — the envelope. Uploads build_lowpass_fir_taps()'s
// taps into __constant__ memory on its FIRST call only (cached thereafter,
// the same "compute once, reuse" pattern 03.01 uses for its Hann windows).
//   d_i_raw, d_q_raw : DEVICE [kImageNz*kImageNx] IN — launch_quadrature_demod's output.
//   d_env            : DEVICE [kImageNz*kImageNx] OUT — the envelope (>= 0).
void launch_envelope_lowpass(const float* d_i_raw, const float* d_q_raw, float* d_env);

// launch_log_compress — Stage 3: 20*log10(env/env_max + eps), clamped to
// [-kDynamicRangeDb, 0] — a pointwise MAP. env_max is a HOST scalar (the
// caller's choice of normalization reference — main.cu uses each pipeline's
// OWN envelope maximum, so the GPU and CPU paths are independently
// self-normalized end to end; see main.cu's header comment).
//   d_env : DEVICE [kImageNz*kImageNx] IN.
//   d_db  : DEVICE [kImageNz*kImageNx] OUT, values in [-kDynamicRangeDb, 0].
void launch_log_compress(const float* d_env, float env_max, float* d_db);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — line-by-line oracle twins of the
// three GPU stages above (same formulas, std:: spellings). main.cu runs
// both pipelines on the IDENTICAL channel data and requires agreement at
// EACH stage within a documented tolerance (the project's three-stage §5
// GPU-vs-CPU VERIFY gate — CLAUDE.md §9).
// ---------------------------------------------------------------------------

// das_cpu — oracle twin of launch_das. channel/rf_image are HOST pointers,
// same layouts as the GPU version.
void das_cpu(const float* channel, float* rf_image);

// envelope_detect_cpu — oracle twin of launch_quadrature_demod +
// launch_envelope_lowpass FUSED into one function (the CPU path has no
// reason to split what the GPU splits for parallelism — CLAUDE.md §5's
// "clarity beats speed" rule for this file). rf_image/env are HOST pointers.
void envelope_detect_cpu(const float* rf_image, float* env);

// log_compress_cpu — oracle twin of launch_log_compress. env/db are HOST
// pointers; env_max is the CPU pipeline's OWN envelope maximum (see the
// launch_log_compress comment on independent self-normalization).
void log_compress_cpu(const float* env, float env_max, float* db);

#endif // PROJECT_KERNELS_CUH
