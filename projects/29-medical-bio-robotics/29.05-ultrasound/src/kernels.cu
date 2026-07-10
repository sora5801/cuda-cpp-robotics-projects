// ===========================================================================
// kernels.cu — GPU beamforming pipeline for project 29.05
//              Ultrasound: GPU beamforming (plane-wave delay-and-sum B-mode)
//
// The big idea (why DAS is "the classic GPU-beamforming argument")
// ------------------------------------------------------------------
// Every pixel of the output image is reconstructed INDEPENDENTLY: pixel
// (ix, iz) reads the channel data, computes its own per-element delays, and
// writes its own output. Nothing a pixel does depends on any other pixel —
// the textbook MAP pattern, at the scale that makes GPUs worth using: this
// project's kImageNz*kImageNx grid is ~206,000 independent pixel problems,
// each doing a small (64-element) inner loop. This is exactly how real
// "software beamformers" (Verasonics research systems, the PICMUS benchmark
// this project's plane-wave/DAS choices are modeled on — README "Prior
// art") use GPUs: one thread (or work-item) per output pixel, sweeping the
// SAME small per-element loop every real beamformer runs, just K = 200k
// times in parallel instead of one scanline at a time.
//
// The pipeline, kernel by kernel (matches kernels.cuh's launcher list and
// main.cu's call order):
//   1. das_kernel                — delay-and-sum, f-number Hann apodization
//                                  (map: 1 thread/pixel, inner loop/element)
//   2. quadrature_demod_kernel   — mix RF with cos/sin at fc               (map)
//   3. envelope_lowpass_kernel   — FIR low-pass along depth + magnitude    (1-D stencil)
//   4. log_compress_kernel       — 20*log10(env/max), clamped              (map)
//
// What is NEW here beyond 08.01/03.01's kernels:
//   * a per-pixel, per-element GEOMETRIC delay (not a lookup table) driving
//     LINEAR INTERPOLATION of a stored signal — the "why not nearest-
//     neighbor" story THEORY.md tells in full;
//   * a CONTINUOUS apodization weight that does two jobs in one formula:
//     Hann tapering AND f-number-limited (depth-growing) active aperture —
//     see das_kernel's header comment for the closed form;
//   * __constant__ memory for the envelope stage's FIR taps — every thread
//     in every block reads the SAME 17 floats, the textbook case for
//     __constant__'s per-SM broadcast cache (contrast with 09.01's
//     __constant__ use for a different reason: a whole robot's kinematic
//     tree, not a filter).
//
// All model constants and layouts come from kernels.cuh — the single
// source shared with the CPU oracle; das_kernel below is a deliberate
// line-by-line twin of das_cpu in reference_cpu.cpp (CLAUDE.md's
// "documented duplication" — reference_cpu.cpp has no CUDA headers, so it
// cannot include this file's __global__ bodies).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cmath>
#include <cstdio>

// A float constant so device code never silently promotes to double.
static constexpr float kPiF = 3.14159265358979323846f;

// ===========================================================================
// 1) Delay-and-sum beamforming — the project's star kernel.
// ===========================================================================

// ---------------------------------------------------------------------------
// das_kernel — reconstruct ONE pixel's beamformed RF value.
//
// Physics recap (THEORY.md "The math" derives every line): the array fires
// a single UNSTEERED PLANE WAVE straight down, so the transmit arrival time
// at depth z is the same for every lateral position:
//
//     t_tx(z) = z / c                                    (one-way descent)
//
// The echo from a point at (x, z) returns to element e (at lateral position
// x_e) along a straight line of length r_rx = sqrt((x-x_e)^2 + z^2), so the
// TOTAL two-way delay for element e is
//
//     t(x, z, e) = t_tx(z) + r_rx(x, z, e) / c
//
// — the exact formula this kernel, das_cpu, and main.cu's closed-form delay
// sanity check all restate (kernels.cuh's documented-duplication note).
//
// APODIZATION — one continuous formula does two jobs at once:
//
//     a(z) = z / (2 * kFNumber)              half-width of the ACTIVE
//                                             receive aperture at depth z
//                                             (grows with depth — THEORY.md
//                                             derives why: a fixed physical
//                                             aperture subtends a shrinking
//                                             angle as z grows, so f-number-
//                                             constant apodization widens
//                                             the *used* aperture to match,
//                                             until it saturates at the
//                                             array's full physical width)
//     u = (x_e - x) / a(z)                   element's normalized lateral
//                                             offset from the pixel, in
//                                             units of the active half-
//                                             aperture
//     weight(u) = 0.5 + 0.5*cos(pi*u), |u|<=1;   0 otherwise
//
// weight(u) is a HANN window shaped continuously by distance rather than a
// hard element cutoff: it is 1 at u=0 (element directly under the pixel)
// and tapers smoothly to 0 exactly at the f-number-limited aperture edge —
// so f-number-limited aperture growth and Hann sidelobe tapering are the
// SAME formula (THEORY.md "The algorithm" walks the derivation and shows
// the alternative "hard rectangular window" this avoids).
//
// INTERPOLATION — the fractional delay t(x,z,e)*fs almost never lands on an
// integer sample, so we LINEARLY INTERPOLATE the two bracketing channel
// samples. THEORY.md "Numerical considerations" derives why nearest-
// neighbor would ring: at fs=40 MHz (25 ns/sample) and fc=5 MHz, a half-
// sample delay error is a non-trivial fraction of one carrier period and
// shows up as visible axial ripple in the reconstructed image — linear
// interpolation is the cheap, adequate fix (a sinc/Lanczos interpolator
// would be more accurate still; README Exercise 2).
//
// Thread-to-data mapping: thread idx = blockIdx.x*blockDim.x + threadIdx.x
// owns FLAT pixel index idx = iz*kImageNx + ix (kernels.cuh's row-major
// image layout); iz = idx / kImageNx, ix = idx % kImageNx.
//
// Memory behavior: consecutive threads (consecutive idx) are consecutive ix
// at fixed iz for most of a warp's life (kImageNx=257 does not divide 32
// evenly, so one warp per few hundred straddles a row boundary — harmless).
// Consecutive-ix threads read NEARLY the same delay per element (their x
// differs by one pixel pitch, ~0.075 mm — far below one channel sample's
// worth of delay change), so warp-adjacent threads touch nearby channel
// memory addresses even though the access is not literally coalesced (each
// thread's per-element delay is data-dependent, not affine in threadIdx).
// __restrict__ on d_channel additionally hints the read-only/L2 cache path,
// which is what actually absorbs this "nearly-shared" locality across the
// warp — the honest middle ground between 08.01's perfectly uniform reads
// and 07.09's fully divergent ones (kernels.cuh cross-references this).
// ---------------------------------------------------------------------------
__global__ void das_kernel(const float* __restrict__ d_channel,   // [kNumElements*kNumSamples]
                           float*       __restrict__ d_rf_image)  // [kImageNz*kImageNx] OUT
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's flat pixel index
    const int total_pixels = kImageNz * kImageNx;
    if (idx >= total_pixels) return;                          // ragged-tail guard

    const int iz = idx / kImageNx;    // depth pixel index (0 = shallowest)
    const int ix = idx % kImageNx;    // lateral pixel index (0 = left edge)

    const float z = kImageZMinM + static_cast<float>(iz) * kImageDzM;   // pixel depth (m)
    const float x = kImageXMinM + static_cast<float>(ix) * kImageDxM;   // pixel lateral position (m)

    const float t_tx = z / kSoundSpeedMps;             // plane-wave one-way descent time (s)
    const float half_aperture = z / (2.0f * kFNumber);  // f-number-limited active half-aperture (m)

    float accum = 0.0f;        // weighted sum of interpolated channel samples
    float weight_sum = 0.0f;   // sum of apodization weights (normalizer — see the file header
                                // note: we AVERAGE rather than sum so the image's intensity
                                // scale does not drift purely from the active aperture growing
                                // with depth; THEORY.md/README "Limitations" name this choice)

    // Inner loop: every array element is a CANDIDATE contributor; the
    // continuous apodization weight above naturally zeroes out elements
    // outside the f-number-limited active aperture, so no separate
    // "which elements are active" bookkeeping is needed.
#pragma unroll 8
    for (int e = 0; e < kNumElements; ++e) {
        // Element geometry — the ONE-LINE formula kernels.cuh's file header
        // documents as this project's single most-duplicated piece of
        // physics (restated here, in das_cpu, and in main.cu's delay check).
        const float x_e = (static_cast<float>(e) - 0.5f * (kNumElements - 1)) * kElementPitchM;

        const float u = (x_e - x) / half_aperture;   // normalized offset, apodization argument
        if (fabsf(u) > 1.0f) continue;                // outside the active aperture: weight 0

        const float weight = 0.5f + 0.5f * cosf(kPiF * u);   // Hann taper, 1 at u=0

        const float dx = x_e - x;
        const float r_rx = sqrtf(dx * dx + z * z);          // receive path length (m)
        const float t_total = t_tx + r_rx / kSoundSpeedMps; // total two-way delay (s)

        const float sample_pos = t_total * kSamplingFreqHz; // fractional sample index
        const int   i0 = static_cast<int>(floorf(sample_pos));
        if (i0 < 0 || i0 + 1 >= kNumSamples) continue;       // delay outside the recorded trace

        const float frac = sample_pos - static_cast<float>(i0);
        const float s0 = d_channel[e * kNumSamples + i0];
        const float s1 = d_channel[e * kNumSamples + i0 + 1];
        const float sample_val = s0 + frac * (s1 - s0);      // linear interpolation

        accum += weight * sample_val;
        weight_sum += weight;
    }

    d_rf_image[idx] = (weight_sum > 0.0f) ? (accum / weight_sum) : 0.0f;
}

// ---------------------------------------------------------------------------
// launch_das — grid math + launch + error check (declared in kernels.cuh).
// ---------------------------------------------------------------------------
void launch_das(const float* d_channel, float* d_rf_image)
{
    const int total_pixels = kImageNz * kImageNx;
    const int block = 256;                                 // warp multiple, repo default
    const int grid = (total_pixels + block - 1) / block;    // ceil: covers every pixel

    das_kernel<<<grid, block>>>(d_channel, d_rf_image);
    CUDA_CHECK_LAST_ERROR("das_kernel launch");
}

// ===========================================================================
// 2a) Quadrature demodulation — pointwise mix with cos/sin at fc.
// ===========================================================================

// ---------------------------------------------------------------------------
// quadrature_demod_kernel — mix the beamformed RF image with the carrier,
// referenced to each pixel's OWN on-axis two-way arrival time t(z) = 2z/c
// (THEORY.md "The algorithm" derives why 2z/c — the on-axis approximation
// of the DAS delay formula above — is the right phase reference for a
// pixel-based, not scanline-based, beamformer).
//
//     I_raw(x,z) =  rf(x,z) * cos(2*pi*fc * 2z/c)
//     Q_raw(x,z) = -rf(x,z) * sin(2*pi*fc * 2z/c)
//
// This is the standard analytic-signal downconversion exp(-j*2*pi*fc*t):
// mixing a real narrowband signal with a coherent local oscillator shifts
// its spectrum down by fc, leaving the (slowly varying) envelope near DC
// and a rejected image near 2*fc — which envelope_lowpass_kernel removes
// next. Pure MAP: every pixel's output depends only on its own input pixel
// and its own (compile-time-derivable) depth, so no shared/neighbor reads
// are needed here — the stencil work is entirely in the NEXT kernel.
// ---------------------------------------------------------------------------
__global__ void quadrature_demod_kernel(const float* __restrict__ d_rf_image,  // [kImageNz*kImageNx]
                                        float* __restrict__ d_i_raw,           // [kImageNz*kImageNx] OUT
                                        float* __restrict__ d_q_raw)           // [kImageNz*kImageNx] OUT
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = kImageNz * kImageNx;
    if (idx >= total_pixels) return;

    const int iz = idx / kImageNx;
    const float z = kImageZMinM + static_cast<float>(iz) * kImageDzM;

    const float phase = 2.0f * kPiF * kCenterFreqHz * (2.0f * z / kSoundSpeedMps);
    const float rf = d_rf_image[idx];
    d_i_raw[idx] =  rf * cosf(phase);
    d_q_raw[idx] = -rf * sinf(phase);
}

void launch_quadrature_demod(const float* d_rf_image, float* d_i_raw, float* d_q_raw)
{
    const int total_pixels = kImageNz * kImageNx;
    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;

    quadrature_demod_kernel<<<grid, block>>>(d_rf_image, d_i_raw, d_q_raw);
    CUDA_CHECK_LAST_ERROR("quadrature_demod_kernel launch");
}

// ===========================================================================
// 2b) Envelope low-pass — a 1-D stencil ALONG DEPTH, then magnitude.
// ===========================================================================

// __constant__ memory: every thread in every block reads the SAME 17 tap
// values every call — the textbook case for __constant__'s per-SM
// broadcast cache (one cache line serves the whole warp, and the whole
// grid shares the same working set, unlike d_channel's per-thread-varying
// reads above). Filled ONCE by upload_fir_taps_once() below.
__constant__ float d_fir_taps[kFirTaps];

// upload_fir_taps_once — build the taps on the host (kernels.cuh's shared
// build_lowpass_fir_taps — the ONE place the filter formula lives) and copy
// them into __constant__ memory, but only on the FIRST call: cudaMemcpy is
// a few microseconds, trivial next to the kernel itself, but there is no
// reason to pay it every one of this demo's (single) call either — the
// same "compute once, cache in a device global" pattern 03.01 uses for its
// Hann windows (kernels.cuh's file header cross-references it).
static void upload_fir_taps_once()
{
    static bool uploaded = false;
    if (uploaded) return;
    float taps[kFirTaps];
    build_lowpass_fir_taps(taps);   // kernels.cuh: plain host function, no CUDA syntax
    CUDA_CHECK(cudaMemcpyToSymbol(d_fir_taps, taps, sizeof(taps)));
    uploaded = true;
}

// ---------------------------------------------------------------------------
// envelope_lowpass_kernel — FIR-low-pass I_raw/Q_raw ALONG DEPTH (per
// lateral column, i.e. along the iz axis at fixed ix) to reject the ~2*fc
// mixing image left over from quadrature_demod_kernel, then take the
// magnitude sqrt(I_f^2 + Q_f^2) — the envelope.
//
// This is a 1-D STENCIL: pixel (ix, iz) reads kFirTaps neighbors CENTERED
// on itself along iz, from I_raw/Q_raw — genuinely different data movement
// from every MAP kernel above (das_kernel's "neighbors" are computed
// geometrically per element, not read as fixed offsets; this kernel reads
// FIXED offsets of an ALREADY-COMPUTED image). Edge handling: CLAMPED
// (out-of-range iz2 reuses the nearest valid row) rather than zero-padded —
// zero-padding would pull the filtered value toward zero right at the
// image's shallow/deep edges, a visible dark band; clamping is the honest,
// simple choice and is documented as an edge artifact in README
// "Limitations" (production systems widen the acquired window instead).
//
// Thread-to-data mapping: identical flat-index scheme to das_kernel
// (idx -> iz, ix). Memory: for a FIXED tap offset k, consecutive threads
// (consecutive ix, same iz) read consecutive addresses
// I_raw[(iz+k)*kImageNx + ix] — perfectly COALESCED across the warp; the
// kFirTaps-iteration loop just repeats this kFirTaps times per thread. So
// despite being a stencil "down" the image, the WARP's memory traffic
// stays as clean as a pure map kernel — worth noticing against
// envelope_lowpass's superficial resemblance to a "hard" strided-access
// kernel (THEORY.md "The GPU mapping" makes this point explicitly).
// No shared-memory halo tiling is used (kFirRadius=8 means each pixel's
// stencil overlaps its neighbors' by up to 16 rows — real DAS/image-
// processing kernels tile a halo into shared memory to cut the resulting
// re-reads; README Exercise 4 asks you to add it and measure the win).
// ---------------------------------------------------------------------------
__global__ void envelope_lowpass_kernel(const float* __restrict__ d_i_raw,  // [kImageNz*kImageNx]
                                        const float* __restrict__ d_q_raw,  // [kImageNz*kImageNx]
                                        float* __restrict__ d_env)          // [kImageNz*kImageNx] OUT
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = kImageNz * kImageNx;
    if (idx >= total_pixels) return;

    const int iz = idx / kImageNx;
    const int ix = idx % kImageNx;

    float sum_i = 0.0f, sum_q = 0.0f;
#pragma unroll
    for (int k = -kFirRadius; k <= kFirRadius; ++k) {
        int iz2 = iz + k;
        // Clamp-to-edge (see header comment): reuse the boundary row rather
        // than reading out of bounds or treating the missing sample as 0.
        if (iz2 < 0) iz2 = 0;
        if (iz2 >= kImageNz) iz2 = kImageNz - 1;

        const float tap = d_fir_taps[k + kFirRadius];
        const int nidx = iz2 * kImageNx + ix;
        sum_i += tap * d_i_raw[nidx];
        sum_q += tap * d_q_raw[nidx];
    }

    d_env[idx] = sqrtf(sum_i * sum_i + sum_q * sum_q);
}

void launch_envelope_lowpass(const float* d_i_raw, const float* d_q_raw, float* d_env)
{
    upload_fir_taps_once();

    const int total_pixels = kImageNz * kImageNx;
    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;

    envelope_lowpass_kernel<<<grid, block>>>(d_i_raw, d_q_raw, d_env);
    CUDA_CHECK_LAST_ERROR("envelope_lowpass_kernel launch");
}

// ===========================================================================
// 3) Log compression — pointwise map, the final B-mode image.
// ===========================================================================

// ---------------------------------------------------------------------------
// log_compress_kernel — 20*log10(env/env_max + eps), clamped into
// [-kDynamicRangeDb, 0]. The human eye (and the ultrasound literature) work
// in decibels because tissue reflectivity spans many orders of magnitude
// (wire targets vs. faint speckle) that a linear display cannot show at
// once — THEORY.md "The problem" and "The algorithm" walk the dynamic-
// range argument in full. Pure MAP: every pixel is independent given the
// single scalar env_max (computed on the HOST — see kernels.cuh's comment
// on independent self-normalization per pipeline).
// eps=1e-6f floors the argument so a true-zero envelope pixel (weight_sum
// was 0 in das_kernel, or a perfectly destructive-interference speckle
// null) produces a large-but-finite negative dB rather than -inf/NaN,
// which the clamp below then saturates to -kDynamicRangeDb anyway.
// ---------------------------------------------------------------------------
__global__ void log_compress_kernel(const float* __restrict__ d_env,   // [kImageNz*kImageNx]
                                    float env_max,
                                    float* __restrict__ d_db)          // [kImageNz*kImageNx] OUT
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = kImageNz * kImageNx;
    if (idx >= total_pixels) return;

    const float ratio = d_env[idx] / env_max + 1e-6f;
    float db = 20.0f * log10f(ratio);
    db = fminf(0.0f, fmaxf(-kDynamicRangeDb, db));   // clamp into the display range
    d_db[idx] = db;
}

void launch_log_compress(const float* d_env, float env_max, float* d_db)
{
    const int total_pixels = kImageNz * kImageNx;
    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;

    log_compress_kernel<<<grid, block>>>(d_env, env_max, d_db);
    CUDA_CHECK_LAST_ERROR("log_compress_kernel launch");
}
