// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 29.05
//                     Ultrasound: GPU beamforming (plane-wave DAS B-mode)
//
// Three jobs in this project (all declared in kernels.cuh), one per GPU
// stage, so main.cu can gate GPU-vs-CPU agreement STAGE BY STAGE rather
// than only at the very end (a bug that cancels itself out across stages
// would slip past a single end-to-end check — CLAUDE.md §9's "documented
// tolerance" gate is meant to catch exactly this):
//
//   1. das_cpu             — the ORACLE twin of das_kernel: same delay
//      formula, same f-number/Hann apodization, same linear interpolation,
//      sequential over pixels and (inside each pixel) over elements.
//   2. envelope_detect_cpu — the oracle twin of quadrature_demod_kernel +
//      envelope_lowpass_kernel FUSED into one function (no reason to keep
//      them separate on a single CPU thread — CLAUDE.md §5's "clarity
//      beats speed" rule for this file).
//   3. log_compress_cpu    — the oracle twin of log_compress_kernel.
//
// Why does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), concretely for THIS project:
//   1) CORRECTNESS ORACLE. Beamforming bugs are easy to write and easy to
//      miss by eye: an off-by-one in the delay formula, a transposed
//      element-position sign, or a wrong apodization argument all still
//      PRODUCE an image — just a subtly (or badly) wrong one. A dead-
//      simple sequential version a reader can verify by eye gives ground
//      truth; main.cu runs both and asserts pixel-wise agreement within a
//      documented tolerance at each stage.
//   2) TEACHING BASELINE. Reading this file first, then kernels.cu, shows
//      exactly what parallelization changed: das_kernel's pixel loop
//      became "one thread per pixel"; the element loop inside stays
//      IDENTICAL logic, spelled with std:: instead of CUDA intrinsics.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP. If the reference is clever, it can be wrong, and then the
// oracle lies. (Compiled by the HOST compiler, cl.exe — kernels.cuh has no
// __CUDACC__-fenced declarations for this file to trip over.)
//
// Read this after: kernels.cu — then compare the two side by side; the
// das_cpu / das_kernel pair and the quadrature+lowpass block are each a
// deliberate line-by-line duplication.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>    // std::sin/cos/sqrt/log10, float versions
#include <vector>   // scratch I/Q buffers in envelope_detect_cpu

namespace {
constexpr float kPiF = 3.14159265358979323846f;
}

// ---------------------------------------------------------------------------
// das_cpu — sequential twin of das_kernel. See kernels.cu for the full
// physics derivation (delay formula, f-number/Hann apodization, why linear
// interpolation) — not repeated here; only the MATH must stay identical.
// channel/rf_image are HOST pointers using the same layouts as the GPU path
// (kernels.cuh: channel[e*kNumSamples+s], rf_image[iz*kImageNx+ix]).
// ---------------------------------------------------------------------------
void das_cpu(const float* channel, float* rf_image)
{
    for (int iz = 0; iz < kImageNz; ++iz) {
        const float z = kImageZMinM + static_cast<float>(iz) * kImageDzM;
        const float t_tx = z / kSoundSpeedMps;
        const float half_aperture = z / (2.0f * kFNumber);

        for (int ix = 0; ix < kImageNx; ++ix) {
            const float x = kImageXMinM + static_cast<float>(ix) * kImageDxM;

            float accum = 0.0f;
            float weight_sum = 0.0f;

            for (int e = 0; e < kNumElements; ++e) {
                // Same one-line element-geometry formula as das_kernel and
                // main.cu's delay sanity check (kernels.cuh's documented
                // duplication note).
                const float x_e = (static_cast<float>(e) - 0.5f * (kNumElements - 1)) * kElementPitchM;

                const float u = (x_e - x) / half_aperture;
                if (std::fabs(u) > 1.0f) continue;

                const float weight = 0.5f + 0.5f * std::cos(kPiF * u);

                const float dx = x_e - x;
                const float r_rx = std::sqrt(dx * dx + z * z);
                const float t_total = t_tx + r_rx / kSoundSpeedMps;

                const float sample_pos = t_total * kSamplingFreqHz;
                const int   i0 = static_cast<int>(std::floor(sample_pos));
                if (i0 < 0 || i0 + 1 >= kNumSamples) continue;

                const float frac = sample_pos - static_cast<float>(i0);
                const float s0 = channel[e * kNumSamples + i0];
                const float s1 = channel[e * kNumSamples + i0 + 1];
                const float sample_val = s0 + frac * (s1 - s0);

                accum += weight * sample_val;
                weight_sum += weight;
            }

            rf_image[iz * kImageNx + ix] = (weight_sum > 0.0f) ? (accum / weight_sum) : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// envelope_detect_cpu — sequential twin of quadrature_demod_kernel +
// envelope_lowpass_kernel, fused. Same phase reference (2z/c on-axis
// approximation), same Hann-windowed-sinc FIR (kernels.cuh's shared
// build_lowpass_fir_taps — the ONE place that formula lives, so this
// function and envelope_lowpass_kernel are guaranteed bit-identical taps),
// same clamp-to-edge boundary handling.
// ---------------------------------------------------------------------------
void envelope_detect_cpu(const float* rf_image, float* env)
{
    // Raw (pre-filter) I/Q for the whole image — the CPU has no reason to
    // avoid a second full-image buffer; kernels.cu splits this into two
    // kernels because a GPU launch boundary is where results become
    // visible to OTHER threads (envelope_lowpass_kernel's stencil needs
    // every pixel's quadrature_demod_kernel output to already exist).
    std::vector<float> i_raw(static_cast<size_t>(kImageNz) * kImageNx);
    std::vector<float> q_raw(static_cast<size_t>(kImageNz) * kImageNx);

    for (int iz = 0; iz < kImageNz; ++iz) {
        const float z = kImageZMinM + static_cast<float>(iz) * kImageDzM;
        const float phase = 2.0f * kPiF * kCenterFreqHz * (2.0f * z / kSoundSpeedMps);
        const float c = std::cos(phase);
        const float s = std::sin(phase);
        for (int ix = 0; ix < kImageNx; ++ix) {
            const int idx = iz * kImageNx + ix;
            const float rf = rf_image[idx];
            i_raw[static_cast<size_t>(idx)] =  rf * c;
            q_raw[static_cast<size_t>(idx)] = -rf * s;
        }
    }

    float taps[kFirTaps];
    build_lowpass_fir_taps(taps);   // kernels.cuh: identical formula to the GPU's __constant__ upload

    for (int iz = 0; iz < kImageNz; ++iz) {
        for (int ix = 0; ix < kImageNx; ++ix) {
            float sum_i = 0.0f, sum_q = 0.0f;
            for (int k = -kFirRadius; k <= kFirRadius; ++k) {
                int iz2 = iz + k;
                if (iz2 < 0) iz2 = 0;                       // clamp-to-edge, same as envelope_lowpass_kernel
                if (iz2 >= kImageNz) iz2 = kImageNz - 1;

                const float tap = taps[k + kFirRadius];
                const int nidx = iz2 * kImageNx + ix;
                sum_i += tap * i_raw[static_cast<size_t>(nidx)];
                sum_q += tap * q_raw[static_cast<size_t>(nidx)];
            }
            env[iz * kImageNx + ix] = std::sqrt(sum_i * sum_i + sum_q * sum_q);
        }
    }
}

// ---------------------------------------------------------------------------
// log_compress_cpu — sequential twin of log_compress_kernel.
// ---------------------------------------------------------------------------
void log_compress_cpu(const float* env, float env_max, float* db)
{
    const int total_pixels = kImageNz * kImageNx;
    for (int i = 0; i < total_pixels; ++i) {
        const float ratio = env[i] / env_max + 1e-6f;
        float d = 20.0f * std::log10(ratio);
        d = std::fmin(0.0f, std::fmax(-kDynamicRangeDb, d));
        db[i] = d;
    }
}
