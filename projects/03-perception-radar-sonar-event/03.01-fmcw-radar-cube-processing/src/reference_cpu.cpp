// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU oracle for project 03.01
//                     FMCW radar cube processing: range-Doppler-angle FFTs +
//                     CA/OS-CFAR detection
//
// WHY a naive O(N^2) DFT is the right choice for THIS oracle (and would be
// the WRONG choice for the GPU path in kernels.cu)
// ----------------------------------------------------------------------
// The whole point of a CPU reference is to be a piece of code a reader can
// verify BY EYE against the textbook definition of a DFT:
//
//     X[m] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*n*m/N)
//
// with no cleverness (no bit-reversal permutation, no radix-2/4
// butterflies, no in-place trickery) standing between the formula and the
// code. That clarity is worth its O(N^2) cost here because the cost is
// AFFORDABLE: this project's transforms are small (Ns=256, Nc=128) and run
// ONCE per demo, not thousands of times per second like the GPU path would
// need to in a real radar (10-20 Hz frame rate x range+Doppler FFTs every
// frame — see README "System context"). Concretely, computing every
// range-bin output for every (chirp, antenna) pair this way costs
// Ns^2 * Nc * Na = 256^2*128*8 ~= 67 million complex multiply-adds, and the
// Doppler pass costs Nc^2 * Ns * Na = 128^2*256*8 ~= 34 million more — about
// 100 million complex MACs total, well under a second of plain scalar C++
// on any modern CPU (measured: see the [time] line main.cu prints). A
// production system computing this every 50-100 ms forever could NOT
// afford O(N^2) and needs the O(N log N) Cooley-Tukey algorithm cuFFT
// implements in kernels.cu — but as a one-shot CORRECTNESS ORACLE, this
// file's job is to be obviously right, not fast.
//
// The one optimization taken: TWIDDLE-FACTOR TABLES. exp(-2*pi*i*n*m/N) is
// precomputed ONCE per axis (an N x N table) instead of calling sinf/cosf
// inside the innermost loop — this is not an algorithmic shortcut (still
// exactly O(N^2) multiply-adds), it just amortizes the transcendental
// calls, which is the difference between "seconds" and "tens of seconds"
// here and does not compromise the "obviously the DFT formula" claim above
// (the table literally stores the formula's own exp(...) terms).
//
// This file ALSO carries the CFAR detectors and the angle estimator as
// line-by-line twins of kernels.cu's kernels (same guard/training geometry,
// same alpha constants, same insertion sort) — the twin relationship the
// repo's whole verification strategy depends on (CLAUDE.md §5, §9).
//
// Rules for this file (as for every reference_cpu.cpp in the repo): plain
// C++17, NO CUDA or cuFFT headers — compiled by the HOST compiler (cl.exe),
// never nvcc. kernels.cuh's ComplexF32 (not cufftComplex) is exactly why
// that is possible here.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Host twins of kernels.cu's per-sample noise generator. Deliberately
// duplicated (not shared via a common header) so this file stays
// CUDA-header-free — see kernels.cuh's file header for the project-wide
// reasoning. The FORMULAS below are byte-for-byte identical to kernels.cu's
// __device__ __host__ versions (same operations, same float arithmetic, no
// double-precision detours) so the CPU and GPU cubes agree to near-ULP
// precision with ZERO data transfer between the two paths — the same
// noise-parity strategy 08.01 uses for its host-generated exploration
// noise, here re-derived per-sample instead of per-control-tick.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32_step(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// hash32_mix — byte-for-byte twin of kernels.cu's hash32_mix (Chris
// Wellons' "lowbias32" avalanche finalizer). See kernels.cu's comment for
// why hashing the per-sample seed (not just a linear offset) is required
// before running xorshift32 — without it, consecutive samples' noise is
// measurably non-white (this project's first implementation attempt hit
// exactly that bug; THEORY.md documents the before/after).
static inline uint32_t hash32_mix(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline float uniform01_from(uint32_t& state)
{
    return (xorshift32_step(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

static inline ComplexF32 complex_gaussian(uint32_t& state, float sigma)
{
    const float u1 = uniform01_from(state);
    const float u2 = uniform01_from(state);
    const float r  = std::sqrt(-2.0f * std::log(u1));
    const float t  = 6.28318530717958647692f * u2;
    ComplexF32 z;
    z.re = sigma * r * std::cos(t);
    z.im = sigma * r * std::sin(t);
    return z;
}

// ---------------------------------------------------------------------------
// synthesize_cube_cpu — sequential twin of synthesize_cube_kernel: same
// per-target phasor formula, same per-sample noise stream, one sample at a
// time instead of one GPU thread per sample. See kernels.cu's
// synthesize_cube_kernel for the full physics commentary (not repeated
// here — the MATH must stay identical, so only the loop structure differs).
// ---------------------------------------------------------------------------
void synthesize_cube_cpu(ComplexF32* cube, const RadarTarget* targets, int num_targets)
{
    const int total = kNs * kNc * kNa;
    for (int idx = 0; idx < total; ++idx) {
        const int a = idx % kNa;
        const int nc_idx = idx / kNa;
        const int c = nc_idx % kNc;
        const int n = nc_idx / kNc;

        float sig_re = 0.0f, sig_im = 0.0f;
        for (int t = 0; t < num_targets; ++t) {
            const RadarTarget tgt = targets[t];
            const float f_beat = 2.0f * tgt.range_m * kSlope / kC;
            const float f_d    = 2.0f * tgt.vel_mps / kLambda;
            const float az_rad = tgt.az_deg * (3.14159265358979323846f / 180.0f);

            const float phase = 6.28318530717958647692f *
                                    (f_beat * static_cast<float>(n) / kFs +
                                     f_d * static_cast<float>(c) * kChirpDur)
                               + 3.14159265358979323846f * std::sin(az_rad) * static_cast<float>(a);

            sig_re += tgt.amp * std::cos(phase);
            sig_im += tgt.amp * std::sin(phase);
        }

        uint32_t rng_state = hash32_mix(kNoiseSeed ^ hash32_mix(static_cast<uint32_t>(idx)));
        if (rng_state == 0u) rng_state = 1u;
        const ComplexF32 noise = complex_gaussian(rng_state, kNoiseStd);

        cube[idx].re = sig_re + noise.re;
        cube[idx].im = sig_im + noise.im;
    }
}

// ---------------------------------------------------------------------------
// Small complex-arithmetic helpers (this file avoids <complex> so its
// float rounding behavior stays as close as possible to kernels.cu's raw
// float re/im arithmetic — comparing apples to apples in the VERIFY gate).
// ---------------------------------------------------------------------------
static inline ComplexF32 cadd(ComplexF32 a, ComplexF32 b) { return { a.re + b.re, a.im + b.im }; }
static inline ComplexF32 cmul(ComplexF32 a, ComplexF32 b)
{
    return { a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

// build_twiddle_table — table[n*N + m] = exp(-2*pi*i*n*m/N), the textbook
// DFT kernel for a length-N transform, for ALL N*N (n, m) pairs. See this
// file's header comment for why precomputing this (rather than calling
// sinf/cosf inside the O(N^2) accumulation loop) is bookkeeping, not an
// algorithmic shortcut — the O(N^2) multiply-add structure is unchanged.
static std::vector<ComplexF32> build_twiddle_table(int N)
{
    std::vector<ComplexF32> table(static_cast<size_t>(N) * N);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < N; ++m) {
            const double angle = -2.0 * 3.14159265358979323846 * static_cast<double>(n) * m / static_cast<double>(N);
            table[static_cast<size_t>(n) * N + m].re = static_cast<float>(std::cos(angle));
            table[static_cast<size_t>(n) * N + m].im = static_cast<float>(std::sin(angle));
        }
    }
    return table;
}

// ---------------------------------------------------------------------------
// process_rd_map_cpu — Hann windows + O(N^2) range DFT + O(N^2) Doppler DFT
// + noncoherent antenna integration, all as the textbook formula operating
// directly on this file's [Ns][Nc][Na] copy of the cube (no in-place
// aliasing tricks — a fresh output buffer per transform stage, favoring
// clarity exactly as this file's header promises).
// ---------------------------------------------------------------------------
void process_rd_map_cpu(const ComplexF32* cube_in, float* rd_power)
{
    const size_t total = static_cast<size_t>(kNs) * kNc * kNa;
    std::vector<ComplexF32> cube(cube_in, cube_in + total);   // working copy

    // ---- Hann windows (identical formula to kernels.cu's ensure_hann_windows) ----
    std::vector<float> win_range(kNs), win_doppler(kNc);
    for (int i = 0; i < kNs; ++i)
        win_range[i] = 0.5f * (1.0f - std::cos(6.28318530717958647692f * static_cast<float>(i) / static_cast<float>(kNs - 1)));
    for (int i = 0; i < kNc; ++i)
        win_doppler[i] = 0.5f * (1.0f - std::cos(6.28318530717958647692f * static_cast<float>(i) / static_cast<float>(kNc - 1)));

    for (int n = 0; n < kNs; ++n)
        for (int c = 0; c < kNc; ++c)
            for (int a = 0; a < kNa; ++a) {
                ComplexF32& s = cube[static_cast<size_t>(n) * kNc * kNa + static_cast<size_t>(c) * kNa + a];
                s.re *= win_range[n];
                s.im *= win_range[n];
            }

    // ---- Range DFT: for every (c, a), transform the kNs samples along n ----
    // out[m] = sum_n cube[n,c,a] * twiddle_r[n*Ns+m]  (textbook DFT, THE
    // SAME sum cuFFT's range-axis batched plan computes in kernels.cu).
    const std::vector<ComplexF32> twiddle_r = build_twiddle_table(kNs);
    std::vector<ComplexF32> after_range(total);
    for (int c = 0; c < kNc; ++c) {
        for (int a = 0; a < kNa; ++a) {
            for (int m = 0; m < kNs; ++m) {
                ComplexF32 acc = { 0.0f, 0.0f };
                for (int n = 0; n < kNs; ++n) {
                    const ComplexF32 x = cube[static_cast<size_t>(n) * kNc * kNa + static_cast<size_t>(c) * kNa + a];
                    acc = cadd(acc, cmul(x, twiddle_r[static_cast<size_t>(n) * kNs + m]));
                }
                after_range[static_cast<size_t>(m) * kNc * kNa + static_cast<size_t>(c) * kNa + a] = acc;
            }
        }
    }

    // ---- Doppler window (applied to the range-FFT output, per kernels.cu) ----
    for (int n = 0; n < kNs; ++n)
        for (int c = 0; c < kNc; ++c)
            for (int a = 0; a < kNa; ++a) {
                ComplexF32& s = after_range[static_cast<size_t>(n) * kNc * kNa + static_cast<size_t>(c) * kNa + a];
                s.re *= win_doppler[c];
                s.im *= win_doppler[c];
            }

    // ---- Doppler DFT: for every (n, a), transform the kNc samples along c,
    // with an FFTSHIFT applied by simply writing output bin d directly to
    // the CENTERED index (d + kNc/2) mod kNc — the same remapping
    // np.fft.fftshift performs, done here at write time instead of as a
    // separate pass. ----
    const std::vector<ComplexF32> twiddle_d = build_twiddle_table(kNc);
    std::vector<ComplexF32> after_doppler(total);   // fftshifted: index d=kNc/2 is zero velocity
    for (int n = 0; n < kNs; ++n) {
        for (int a = 0; a < kNa; ++a) {
            for (int d = 0; d < kNc; ++d) {
                ComplexF32 acc = { 0.0f, 0.0f };
                for (int c = 0; c < kNc; ++c) {
                    const ComplexF32 x = after_range[static_cast<size_t>(n) * kNc * kNa + static_cast<size_t>(c) * kNa + a];
                    acc = cadd(acc, cmul(x, twiddle_d[static_cast<size_t>(c) * kNc + d]));
                }
                const int shifted = (d + kNc / 2) % kNc;
                after_doppler[static_cast<size_t>(n) * kNc * kNa + static_cast<size_t>(shifted) * kNa + a] = acc;
            }
        }
    }

    // ---- Noncoherent antenna integration -> rd_power[n,c] ----
    for (int n = 0; n < kNs; ++n) {
        for (int c = 0; c < kNc; ++c) {
            float acc = 0.0f;
            for (int a = 0; a < kNa; ++a) {
                const ComplexF32 s = after_doppler[static_cast<size_t>(n) * kNc * kNa + static_cast<size_t>(c) * kNa + a];
                acc += s.re * s.re + s.im * s.im;
            }
            rd_power[static_cast<size_t>(n) * kNc + c] = acc / static_cast<float>(kNa);
        }
    }
}

// NOTE: kernels.cu's launch_doppler_fft leaves cuFFT's NATURAL bin order
// (DC first, negative frequencies wrapped to the back half) untouched; the
// GPU pipeline instead calls a dedicated launch_fftshift_doppler kernel
// right afterward to remap into this same centered order (bin kNc/2 =
// zero velocity) before any later stage touches the cube. This function
// folds the identical remap into its OWN write index (the `shifted`
// variable above) so the two paths' rd_power arrays are directly,
// cell-for-cell comparable in main.cu's VERIFY gate with no extra pass.

// ---------------------------------------------------------------------------
// gather_training_cells_cpu — line-by-line twin of kernels.cu's
// gather_training_cells device function: same loop order, same guard-band
// skip logic, so cells[] is filled in IDENTICAL order on both paths (not
// that CA/OS's outputs depend on order, but the OS-CFAR sort's tie-breaking
// behavior is easiest to reason about when it does).
// ---------------------------------------------------------------------------
static int gather_training_cells_cpu(const float* rd, int i, int j, float* cells)
{
    int count = 0;
    for (int di = -kCfarHalf; di <= kCfarHalf; ++di) {
        const bool guard_row = (di >= -kCfarGuard && di <= kCfarGuard);
        for (int dj = -kCfarHalf; dj <= kCfarHalf; ++dj) {
            const bool guard_col = (dj >= -kCfarGuard && dj <= kCfarGuard);
            if (guard_row && guard_col) continue;
            cells[count++] = rd[(i + di) * kNc + (j + dj)];
        }
    }
    return count;
}

// cfar_ca_cpu / cfar_os_cpu — twins of cfar_ca_kernel / cfar_os_kernel.
// Same geometry, same alpha constants, same insertion sort (std::sort would
// be a one-line "cleverer" alternative; using the identical hand-rolled
// insertion sort as the GPU kernel keeps the two paths' floating-point
// SUMMATION/COMPARISON order identical, which matters for a bit-level-
// honest oracle even though the final rank value would not change).
void cfar_ca_cpu(const float* rd, unsigned char* det, float* thresh)
{
    float cells[kCfarNTrain];
    for (int i = 0; i < kNs; ++i) {
        for (int j = 0; j < kNc; ++j) {
            const int idx = i * kNc + j;
            if (i < kCfarHalf || i >= kNs - kCfarHalf || j < kCfarHalf || j >= kNc - kCfarHalf) {
                det[idx] = 0; thresh[idx] = 0.0f; continue;
            }
            gather_training_cells_cpu(rd, i, j, cells);
            float sum = 0.0f;
            for (int k = 0; k < kCfarNTrain; ++k) sum += cells[k];
            const float z = sum / static_cast<float>(kCfarNTrain);
            const float t = kAlphaCA * z;
            thresh[idx] = t;
            det[idx] = (rd[idx] > t) ? 1 : 0;
        }
    }
}

void cfar_os_cpu(const float* rd, unsigned char* det, float* thresh)
{
    float cells[kCfarNTrain];
    for (int i = 0; i < kNs; ++i) {
        for (int j = 0; j < kNc; ++j) {
            const int idx = i * kNc + j;
            if (i < kCfarHalf || i >= kNs - kCfarHalf || j < kCfarHalf || j >= kNc - kCfarHalf) {
                det[idx] = 0; thresh[idx] = 0.0f; continue;
            }
            gather_training_cells_cpu(rd, i, j, cells);
            for (int a = 1; a < kCfarNTrain; ++a) {
                const float key = cells[a];
                int b = a - 1;
                while (b >= 0 && cells[b] > key) { cells[b + 1] = cells[b]; --b; }
                cells[b + 1] = key;
            }
            const float z = cells[kOsRankIndex];
            const float t = kAlphaOS * z;
            thresh[idx] = t;
            det[idx] = (rd[idx] > t) ? 1 : 0;
        }
    }
}

// ---------------------------------------------------------------------------
// angle_estimate_cpu — O(kNaFft * kNa) zero-padded DFT twin of
// gather_angle_snapshots_kernel + launch_angle_fft + find_angle_peaks_kernel
// combined, for ONE detection. Because kNaFft-kNa of the kNaFft inputs are
// exactly zero (the zero-padding), the DFT sum only needs the kNa non-zero
// terms per output bin — still an O(N^2)-FAMILY algorithm (here
// O(kNaFft*kNa) = 64*8 = 512 multiply-adds, trivial), consistent with this
// file's "obviously correct, not fast" mandate.
// ---------------------------------------------------------------------------
float angle_estimate_cpu(const ComplexF32* cube, int kr, int kd)
{
    ComplexF32 snapshot[kNa];
    for (int a = 0; a < kNa; ++a)
        snapshot[a] = cube[static_cast<size_t>(kr) * kNc * kNa + static_cast<size_t>(kd) * kNa + a];

    int best_bin = 0;
    float best_mag2 = -1.0f;
    for (int k = 0; k < kNaFft; ++k) {
        ComplexF32 acc = { 0.0f, 0.0f };
        for (int a = 0; a < kNa; ++a) {
            const double angle = -2.0 * 3.14159265358979323846 * static_cast<double>(a) * k / static_cast<double>(kNaFft);
            const ComplexF32 tw = { static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle)) };
            acc = cadd(acc, cmul(snapshot[a], tw));
        }
        const float mag2 = acc.re * acc.re + acc.im * acc.im;
        if (mag2 > best_mag2) { best_mag2 = mag2; best_bin = k; }
    }

    const int k_centered = (best_bin < kNaFft / 2) ? best_bin : best_bin - kNaFft;
    float sin_theta = 2.0f * static_cast<float>(k_centered) / static_cast<float>(kNaFft);
    sin_theta = std::min(1.0f, std::max(-1.0f, sin_theta));
    return std::asin(sin_theta) * (180.0f / 3.14159265358979323846f);
}
