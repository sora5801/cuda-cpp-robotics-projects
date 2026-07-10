// ===========================================================================
// kernels.cu — GPU pipeline for project 03.01
//              FMCW radar cube processing: range-Doppler-angle FFTs +
//              CA/OS-CFAR detection
//
// The big idea
// ------------
// A radar "cube" is a 3-D block of complex baseband samples: fast-time
// (Ns samples per chirp) x slow-time (Nc chirps) x antenna (Na channels).
// Every axis is turned into a physical quantity by an FFT ALONG that axis
// — range from fast-time, velocity from slow-time, angle from antenna —
// and every one of those FFTs is EMBARRASSINGLY BATCHED: thousands of
// independent length-256/128/64 transforms that share nothing. That is
// exactly the shape cuFFT's batched planning API (cufftPlanMany) exists
// for, and this file's job is to feed it the right memory layout for each
// axis without ever copying/transposing the cube (CLAUDE.md §5's "explain
// every cuFFT call: what it computes, why not hand-rolled, the shapes").
//
// The pipeline, kernel by kernel (matches kernels.cuh's launcher list and
// main.cu's call order):
//   1. synthesize_cube_kernel   — build the raw cube (map: 1 thread/sample)
//   2. hann_window_*_kernel     — taper before each FFT   (map)
//   3. launch_range_fft         — cuFFT, ADVANCED layout  (library)
//   4. launch_doppler_fft       — cuFFT, ADVANCED layout, looped (library)
//   5. noncoherent_integrate    — antenna power average   (map+reduce, tiny)
//   6. cfar_ca / cfar_os        — 2-D CFAR detectors      (stencil)
//   7. gather_angle_snapshots   — per-detection gather     (map)
//   8. launch_angle_fft         — cuFFT, CONTIGUOUS batch (library)
//   9. find_angle_peaks         — per-detection argmax     (map+reduce, tiny)
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cufft.h>              // cuFFT: cufftPlanMany / cufftExecC2C (see each launcher below)
#include <cmath>
#include <cstdio>
#include <cstdlib>

// cuFFT calls return their OWN error enum (cufftResult), not cudaError_t —
// CUDA_CHECK (util/cuda_check.cuh) only understands cudaError_t, so cuFFT
// gets its own tiny check macro here, same do/while(0)-guarded shape and
// same "fail loud, fail at the call site" philosophy (CLAUDE.md §6.1 rule 7).
#define CUFFT_CHECK(call)                                                     \
    do {                                                                     \
        cufftResult cufft_err__ = (call);                                    \
        if (cufft_err__ != CUFFT_SUCCESS) {                                  \
            std::fprintf(stderr, "cuFFT error %d at %s:%d in '%s'\n",        \
                         static_cast<int>(cufft_err__), __FILE__, __LINE__,  \
                         #call);                                             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

// ===========================================================================
// 1) Cube synthesis
// ===========================================================================

// ---------------------------------------------------------------------------
// xorshift32 / uniform01 / box_muller_pair — the repo's portable, per-thread
// deterministic noise generator (same construction as 08.01's host-side
// generator, here run ON-DEVICE, one independent stream per SAMPLE rather
// than per control-tick). Key property: a sample's noise depends ONLY on
// its own flat index + a fixed seed — never on thread-scheduling order or
// on any other thread's state — so the result is bit-reproducible
// regardless of how the grid is scheduled, and reference_cpu.cpp can
// reproduce EXACTLY the same noise for the same index with no data
// transfer between the two paths (THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
__device__ __host__ inline uint32_t xorshift32_step(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// hash32_mix — Chris Wellons' "lowbias32" 32-bit integer finalizer (a
// well-characterized, publicly documented avalanche hash: every output bit
// depends on every input bit with near-zero measured bias). Used to turn a
// LINEARLY-INCREMENTING per-sample seed (kNoiseSeed + K*idx, consecutive
// samples differ by a constant) into a well-mixed xorshift32 START STATE.
//
// Why this step is NECESSARY, not decorative: xorshift32 is a LINEAR
// recurrence over GF(2). Seeding it directly from consecutive linear
// values and drawing output after only 1-2 steps leaves detectable
// structure between NEIGHBORING samples' noise — this project's first
// implementation attempt did exactly that and the resulting "noise" was
// visibly non-white: its FFT showed elevated, non-flat power at specific
// range bins, enough to blow past both CFAR detectors' thresholds by the
// hundreds (a false-alarm count orders of magnitude above the Pfa=1e-4
// design target). Hashing the seed first breaks the linear relationship
// between neighboring samples before xorshift32 ever runs, and IS the fix
// that restores a flat, IID-looking noise floor (see THEORY.md "Numerical
// considerations" for the measured before/after comparison).
__device__ __host__ inline uint32_t hash32_mix(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __host__ inline float uniform01_from(uint32_t& state)
{
    // Top 24 bits -> (0,1]; never exactly 0 (safe for logf() below) and
    // never exactly 1 (keeps the Box-Muller radius finite).
    return (xorshift32_step(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// One complex Gaussian sample (re, im independently N(0, sigma^2)) from a
// SINGLE Box-Muller pair (u1, u2): the classic transform emits TWO
// independent standard normals per pair, which we assign directly to the
// real and imaginary parts instead of drawing two separate pairs — half
// the transcendental calls of the naive approach, and no correlation
// concern (I and Q noise are physically independent thermal processes).
__device__ __host__ inline ComplexF32 complex_gaussian(uint32_t& state, float sigma)
{
    const float u1 = uniform01_from(state);
    const float u2 = uniform01_from(state);
    const float r  = sqrtf(-2.0f * logf(u1));
    const float t  = 6.28318530717958647692f * u2;   // 2*pi*u2
    ComplexF32 z;
    z.re = sigma * r * cosf(t);
    z.im = sigma * r * sinf(t);
    return z;
}

// ---------------------------------------------------------------------------
// synthesize_cube_kernel — build ONE complex sample of the raw radar cube.
//
// Physics recap (THEORY.md derives every line): for a target at range R,
// radial velocity v (positive = approaching), azimuth theta, the received
// baseband phasor at fast-time sample n, chirp c, antenna a is
//
//     amp * exp( j * 2*pi * (f_beat*n/fs + f_d*c*Tc) ) * exp( j*pi*sin(theta)*a )
//
//   f_beat = 2*R*S/c     (beat frequency: constant across a chirp, gives RANGE)
//   f_d    = 2*v/lambda  (Doppler frequency: constant across a frame,
//                         phase-accumulated chirp to chirp, gives VELOCITY)
//   pi*sin(theta)*a      (spatial phase step for a half-wavelength ULA;
//                         d = lambda/2 makes the 2*pi*d*sin(theta)/lambda
//                         steering formula collapse to pi*sin(theta) — see
//                         THEORY.md "The math")
// This is the DECOUPLED teaching model: range (fast-time phase) and
// Doppler (slow-time phase) are treated as two independent phase
// accumulators, ignoring the small "range-Doppler coupling" a real chirp
// exhibits when a target moves during the sweep itself. THEORY.md
// "Numerical considerations" quantifies the coupling this ignores and
// "Where this sits in the real world" explains how production stacks
// handle it (alternating up/down chirp slopes, TDM-MIMO scheduling).
//
// Thread-to-data mapping: ONE thread per complex sample, over the whole
// cube (Ns*Nc*Na threads = 262,144 for the default sizes). Thread i owns
// flat index i, decoded to (n, c, a) via the kernels.cuh layout contract
// idx = n*(Nc*Na) + c*Na + a. Every target's contribution is a scalar
// closed-form phasor — no shared state, no atomics, embarrassingly
// parallel: the purest MAP in this project.
// ---------------------------------------------------------------------------
__global__ void synthesize_cube_kernel(ComplexF32* __restrict__ cube,
                                       const RadarTarget* __restrict__ targets,
                                       int num_targets, int total_samples)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;

    // Decode idx -> (n, c, a). Antenna is fastest-varying (innermost),
    // matching kernels.cuh's layout contract exactly.
    const int a = idx % kNa;
    const int nc_idx = idx / kNa;      // combined (n, c) index
    const int c = nc_idx % kNc;
    const int n = nc_idx / kNc;

    float sig_re = 0.0f, sig_im = 0.0f;   // accumulate all targets' phasors
    for (int t = 0; t < num_targets; ++t) {
        const RadarTarget tgt = targets[t];
        const float f_beat = 2.0f * tgt.range_m * kSlope / kC;
        const float f_d    = 2.0f * tgt.vel_mps / kLambda;
        const float az_rad = tgt.az_deg * (3.14159265358979323846f / 180.0f);

        // Total phase = range term (fast-time) + Doppler term (slow-time)
        // + angle term (antenna) — three independent accumulators, summed
        // once because they multiply as phasors (exp(jA)*exp(jB) = exp(j(A+B))).
        const float phase = 6.28318530717958647692f *
                                (f_beat * static_cast<float>(n) / kFs +
                                 f_d * static_cast<float>(c) * kChirpDur)
                           + 3.14159265358979323846f * sinf(az_rad) * static_cast<float>(a);

        // Precise sinf/cosf (not the fast __sinf/__cosf intrinsics): phase
        // here can span many multiples of 2*pi (large n, c), where
        // intrinsic trig's relative error grows — same reasoning 08.01/
        // 09.01 give for precise trig near large or wrapped angles.
        sig_re += tgt.amp * cosf(phase);
        sig_im += tgt.amp * sinf(phase);
    }

    // Per-sample deterministic noise stream: hash (seed, idx) into a
    // well-mixed xorshift32 start state — see hash32_mix's comment above
    // for why the hash step (not just a linear seed offset) is required
    // for the noise to actually be spectrally flat/white.
    uint32_t rng_state = hash32_mix(kNoiseSeed ^ hash32_mix(static_cast<uint32_t>(idx)));
    if (rng_state == 0u) rng_state = 1u;   // xorshift's one forbidden state
    const ComplexF32 noise = complex_gaussian(rng_state, kNoiseStd);

    ComplexF32 out;
    out.re = sig_re + noise.re;
    out.im = sig_im + noise.im;
    cube[idx] = out;
}

void launch_synthesize_cube(ComplexF32* d_cube, const RadarTarget* d_targets, int num_targets)
{
    const int total = kNs * kNc * kNa;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    synthesize_cube_kernel<<<grid, block>>>(d_cube, d_targets, num_targets, total);
    CUDA_CHECK_LAST_ERROR("synthesize_cube_kernel launch");
}

// ===========================================================================
// 2) Hann windows
// ===========================================================================

// ---------------------------------------------------------------------------
// Why window at all? A finite-length DFT is mathematically equivalent to
// multiplying an infinite signal by a RECTANGULAR window, whose spectrum
// is a sinc with slowly-decaying (~-13 dB) sidelobes. A strong target's
// sidelobes leak into every other range/Doppler bin at that level — far
// above the noise floor — and get flagged by CFAR as spurious detections
// scattered across the strong target's whole row/column (exactly what an
// early version of this project's own prototype showed with NO Doppler
// window: dozens of extra "detections" at one strong target's range bin,
// spread across nearly every Doppler bin). A Hann taper
// w[i] = 0.5*(1 - cos(2*pi*i/(N-1))) trades a WIDER mainlobe (worse
// resolution — the reason kRangeResM/kVelResMps are quoted for a
// rectangular window and are, honestly, a slightly optimistic best case)
// for sidelobes below -31 dB, which is what actually makes CFAR usable on
// a scene with targets of very different strengths. See THEORY.md
// "Numerical considerations" for the resolution-vs-sidelobe trade in full.
// ---------------------------------------------------------------------------

// Precomputed window coefficients, uploaded once and reused by every call
// (the windows never change — recomputing sinf/cosf per element per call
// would be pure waste). Lazily initialized on first use.
static float* g_d_hann_range = nullptr;    // [kNs]
static float* g_d_hann_doppler = nullptr;  // [kNc]

static void ensure_hann_windows()
{
    if (g_d_hann_range) return;   // already built

    float h_range[kNs], h_doppler[kNc];
    for (int i = 0; i < kNs; ++i)
        h_range[i] = 0.5f * (1.0f - cosf(6.28318530717958647692f * static_cast<float>(i) / static_cast<float>(kNs - 1)));
    for (int i = 0; i < kNc; ++i)
        h_doppler[i] = 0.5f * (1.0f - cosf(6.28318530717958647692f * static_cast<float>(i) / static_cast<float>(kNc - 1)));

    CUDA_CHECK(cudaMalloc(&g_d_hann_range, kNs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_hann_doppler, kNc * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_d_hann_range, h_range, kNs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_hann_doppler, h_doppler, kNc * sizeof(float), cudaMemcpyHostToDevice));
}

// hann_window_range_kernel — multiply cube[n,c,a] by win[n] (a UNIFORM
// per-(c,a)-thread-group read of the small kNs-length window — served by
// the L2/read-only cache path, negligible bandwidth next to the cube
// read/write itself). One thread per complex sample: a pure elementwise map.
__global__ void hann_window_range_kernel(ComplexF32* __restrict__ cube,
                                         const float* __restrict__ win, int total_samples)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;
    const int n = idx / (kNc * kNa);          // range index is the SLOWEST-varying
    const float w = win[n];
    cube[idx].re *= w;
    cube[idx].im *= w;
}

// hann_window_doppler_kernel — multiply cube[n,c,a] (POST range-FFT) by
// win[c]. Same map shape; the only difference from the range window is
// which decoded index selects the coefficient.
__global__ void hann_window_doppler_kernel(ComplexF32* __restrict__ cube,
                                           const float* __restrict__ win, int total_samples)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;
    const int c = (idx / kNa) % kNc;          // doppler index is the MIDDLE axis
    const float w = win[c];
    cube[idx].re *= w;
    cube[idx].im *= w;
}

void launch_hann_window_range(ComplexF32* d_cube)
{
    ensure_hann_windows();
    const int total = kNs * kNc * kNa;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    hann_window_range_kernel<<<grid, block>>>(d_cube, g_d_hann_range, total);
    CUDA_CHECK_LAST_ERROR("hann_window_range_kernel launch");
}

void launch_hann_window_doppler(ComplexF32* d_cube)
{
    ensure_hann_windows();
    const int total = kNs * kNc * kNa;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    hann_window_doppler_kernel<<<grid, block>>>(d_cube, g_d_hann_doppler, total);
    CUDA_CHECK_LAST_ERROR("hann_window_doppler_kernel launch");
}

// ===========================================================================
// 3) & 4) The two big FFTs — cuFFT, explained per CLAUDE.md §6.1 rule 6
// ===========================================================================
//
// What cufftPlanMany computes (both calls below): a batch of independent
// 1-D complex-to-complex DFTs, X_k[m] = sum_n x_k[n] * exp(-2*pi*i*n*m/N),
// for k = 0..batch-1, each of length N — i.e. exactly what kernels.cuh's
// file header describes range/Doppler FFTs as doing, computed with the
// O(N log N) Cooley-Tukey algorithm instead of the O(N^2) sum. Why cuFFT
// and not a hand-rolled radix kernel: implementing a numerically-robust,
// arbitrary-batched, strided FFT is a multi-week project on its own (see
// 33.x-style foundational-library projects for what "hand-roll an FFT"
// actually entails) and would teach FFT internals at the expense of
// teaching RADAR — this project's actual subject. cuFFT is a CUDA TOOLKIT
// library (CLAUDE.md §5 default-allowed), and every call it makes is
// explained here in full, so nothing about it is a black box even though
// we do not reimplement it (the CPU oracle in reference_cpu.cpp, by
// contrast, DOES use a hand-rolled O(N^2) DFT — see that file's header
// for why the oracle can afford the naive algorithm where the GPU path
// should not).
//
// cufftPlanMany's "advanced data layout" parameters, used by both calls:
//   n[]      the length of EACH transform (Ns for range, Nc for Doppler)
//   inembed/onembed  set to n[] here (no embedding trick beyond stride)
//   istride/ostride  the ELEMENT distance (in complex samples) between
//                    consecutive samples WITHIN one transform
//   idist/odist      the ELEMENT distance between the START of consecutive
//                    batched transforms
//   batch            how many independent transforms this one plan runs
// This is the exact mechanism that lets cuFFT operate DIRECTLY on the
// [Ns][Nc][Na] cube with no transpose — see each function below for the
// concrete numbers and why they follow from kernels.cuh's layout contract.
// ===========================================================================

// launch_range_fft — transform the RANGE axis (n, the SLOWEST/outermost
// axis) in place, for every (chirp, antenna) pair.
//
// Layout math: sample (n, c, a) sits at offset n*(Nc*Na) + c*Na + a. Fixing
// (c, a) and varying n therefore walks offsets 0, Nc*Na, 2*Nc*Na, ... —
// istride = Nc*Na. And because the (c, a) PAIR itself enumerates exactly
// the offsets 0, 1, 2, ..., Nc*Na-1 (that IS the n=0 slice, contiguous by
// construction of the layout), the whole Nc*Na-batch is expressible with
// idist = 1: batch b's transform starts at offset b. One clean call, no
// host-side loop — the "easy" advanced-layout case (transformed axis is
// the OUTERMOST dimension).
void launch_range_fft(ComplexF32* d_cube)
{
    cufftHandle plan;
    int n[1] = { kNs };
    const int batch = kNc * kNa;
    const int istride = kNc * kNa;
    const int idist = 1;
    // inembed/onembed = n[] (no embedding beyond the stride itself).
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                             n, istride, idist,
                             n, istride, idist,
                             CUFFT_C2C, batch));
    // ComplexF32 <-> cufftComplex: both are exactly {float, float} in
    // memory (re/im == x/y) — a deliberate, documented, layout-verified
    // reinterpret_cast (see kernels.cuh's file header) so reference_cpu.cpp
    // never needs to include cufft.h.
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_cube),
                             reinterpret_cast<cufftComplex*>(d_cube), CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));
}

// launch_doppler_fft — transform the DOPPLER axis (c, the MIDDLE axis) in
// place, for every (range, antenna) pair.
//
// Layout math: fixing a range bin n, the sub-block cube[n, *, *] is
// CONTIGUOUS (offset n*Nc*Na .. n*Nc*Na+Nc*Na-1) and internally shaped
// exactly like a fresh [Nc][Na] slab: fixing an antenna a and varying c
// walks stride Na, and the Na antennas at c=0 are contiguous (offsets
// 0..Na-1) — i.e. INSIDE one range bin, a plan with batch = Na,
// istride = Na, idist = 1 is exactly the range-FFT trick again.
//
// The catch: unlike the range FFT, that "idist = 1 batch" trick only
// covers ONE range bin's Na antennas. Extending it across ALL Ns range
// bins at once would need offset(n, a) = n*(Nc*Na) + a to be an AFFINE
// function of a single combined batch index b = n*Na + a — and it is not
// (as a cycles 0..Na-1, offset climbs by 1 each step; the moment a wraps
// and n increments, offset jumps by Nc*Na - (Na-1) instead). cuFFT's
// batch parameter is a SINGLE linear loop, so no one cufftPlanMany call
// can express "batch over two axes with different strides" — this is a
// real, common limit of the advanced-layout API, not a corner we are
// cutting. THEORY.md "The GPU mapping" discusses the alternative
// (transpose the cube once so Doppler becomes the outer axis) and why we
// deliberately do NOT take it here.
//
// The fix used here: build ONE plan for "Na antennas, length-Nc
// transform" and EXECUTE it Ns times, offsetting the data pointer by one
// range bin (Nc*Na complex samples) each time. This is a standard,
// legitimate cuFFT idiom — "create once, exec many" — and it is exactly
// what a real pipeline does whenever its batch structure has more levels
// than cufftPlanMany's single loop can express.
void launch_doppler_fft(ComplexF32* d_cube)
{
    cufftHandle plan;
    int n[1] = { kNc };
    const int batch = kNa;
    const int istride = kNa;
    const int idist = 1;
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                             n, istride, idist,
                             n, istride, idist,
                             CUFFT_C2C, batch));

    cufftComplex* base = reinterpret_cast<cufftComplex*>(d_cube);
    for (int rbin = 0; rbin < kNs; ++rbin) {
        cufftComplex* slice = base + static_cast<size_t>(rbin) * kNc * kNa;   // this range bin's [Nc][Na] slab
        CUFFT_CHECK(cufftExecC2C(plan, slice, slice, CUFFT_FORWARD));
    }
    CUFFT_CHECK(cufftDestroy(plan));
}

// ---------------------------------------------------------------------------
// fftshift_doppler_kernel — remap cuFFT's natural Doppler bin order
// (0, +1, ..., kNc/2-1, -kNc/2, ..., -1 — DC first, negative frequencies
// wrapped to the back half) into a CENTERED order where index kNc/2 is
// zero velocity. Out-of-place (a straight permutation cannot safely alias
// its own input): each thread reads one sample and writes it to its
// shifted destination — a pure, embarrassingly parallel MAP, one thread
// per complex sample. This is also exactly what numpy's fftshift does,
// and exactly what reference_cpu.cpp folds into the WRITE index of its own
// Doppler DFT (process_rd_map_cpu) — the two paths must agree on which
// bin is "zero velocity" or every downstream comparison would be
// comparing physically different cells.
// ---------------------------------------------------------------------------
__global__ void fftshift_doppler_kernel(const ComplexF32* __restrict__ in,
                                        ComplexF32* __restrict__ out, int total_samples)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;
    const int a = idx % kNa;
    const int nc_idx = idx / kNa;
    const int c = nc_idx % kNc;
    const int n = nc_idx / kNc;

    const int c_shifted = (c + kNc / 2) % kNc;
    const size_t out_idx = static_cast<size_t>(n) * kNc * kNa + static_cast<size_t>(c_shifted) * kNa + a;
    out[out_idx] = in[idx];
}

void launch_fftshift_doppler(const ComplexF32* d_in, ComplexF32* d_out)
{
    const int total = kNs * kNc * kNa;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    fftshift_doppler_kernel<<<grid, block>>>(d_in, d_out, total);
    CUDA_CHECK_LAST_ERROR("fftshift_doppler_kernel launch");
}

// ===========================================================================
// 5) Noncoherent antenna integration -> range-Doppler POWER map
// ===========================================================================

// ---------------------------------------------------------------------------
// noncoherent_integrate_kernel — rd_power[n,c] = mean_a |cube[n,c,a]|^2.
//
// "Noncoherent" means we sum POWER (magnitude squared), not the complex
// AMPLITUDE — the antenna phase differences that will later reveal angle
// (step 8/9) would just cancel out under a coherent (complex) sum here,
// since different targets/noise realizations have unrelated antenna
// phase. Averaging power across Na independent looks at the SAME range-
// Doppler cell is a free SNR improvement for detection purposes (the
// classical "noncoherent integration gain", ~sqrt(Na) in amplitude
// terms) — exactly the reason a detector reads the RD map AFTER
// collapsing antennas, then estimates angle separately, only for cells
// that already cleared detection (step 7-9).
//
// Thread-to-data mapping: one thread per (n, c) output cell (Ns*Nc =
// 32,768 threads), each reading its Na antenna samples with STRIDE-1
// access (a is the fastest-varying axis) — a small, fully coalesced read
// per thread, then Na multiply-adds and one write. A stencil-flavored
// reduction, but over a tiny, fixed Na=8 — cheap enough that a serial
// per-thread loop beats any shared-memory tree-reduction machinery.
// ---------------------------------------------------------------------------
__global__ void noncoherent_integrate_kernel(const ComplexF32* __restrict__ cube,
                                             float* __restrict__ rd_power)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // combined (n,c) index
    if (idx >= kNs * kNc) return;

    const ComplexF32* base = cube + static_cast<size_t>(idx) * kNa;
    float acc = 0.0f;
#pragma unroll
    for (int a = 0; a < kNa; ++a) {
        const float re = base[a].re, im = base[a].im;
        acc += re * re + im * im;
    }
    rd_power[idx] = acc / static_cast<float>(kNa);
}

void launch_noncoherent_integrate(const ComplexF32* d_cube, float* d_rd_power)
{
    const int total = kNs * kNc;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    noncoherent_integrate_kernel<<<grid, block>>>(d_cube, d_rd_power);
    CUDA_CHECK_LAST_ERROR("noncoherent_integrate_kernel launch");
}

// ===========================================================================
// 6) 2-D CFAR detectors — CA (cell-averaging) and OS (ordered-statistic)
// ===========================================================================

// ---------------------------------------------------------------------------
// Shared training-window gather: copy every TRAINING cell (the kCfarWindow
// x kCfarWindow neighborhood MINUS the (2*kCfarGuard+1)^2 guard block
// around the cell under test) into a per-thread local array. Both CFAR
// kernels below call this — the guard/training GEOMETRY is identical; only
// what happens to `cells[]` afterward (mean vs. sorted rank) differs. That
// is the whole point of the comparison this project teaches (THEORY.md
// "The algorithm"): isolate ONE variable.
//
// Guard cells exist because the cell under test's own energy "bleeds" into
// its immediate neighbors (finite mainlobe width, even after windowing);
// including them in the training average would make the detector measure
// its OWN target's energy as if it were background clutter, raising its
// own threshold and desensitizing itself. Training cells estimate the
// LOCAL clutter/noise floor from cells assumed target-free.
//
// local memory note: kCfarNTrain = 200 floats (800 bytes) per thread is a
// genuinely large per-thread footprint — this spills to LOCAL memory
// (off-chip, cached) rather than living in registers. That is an honest,
// documented cost of a teaching-simple, fixed-size implementation;
// THEORY.md "Where this sits in the real world" names the production fix
// (a running/sliding-window sum for CA-CFAR avoids re-summing entirely;
// OS-CFAR's need for a full order statistic is the harder case, usually
// solved with a smaller window or an approximate/streaming selection
// algorithm rather than a full per-cell sort).
// ---------------------------------------------------------------------------
__device__ inline int gather_training_cells(const float* __restrict__ rd, int i, int j, float* __restrict__ cells)
{
    int count = 0;
#pragma unroll
    for (int di = -kCfarHalf; di <= kCfarHalf; ++di) {
        const bool guard_row = (di >= -kCfarGuard && di <= kCfarGuard);
#pragma unroll
        for (int dj = -kCfarHalf; dj <= kCfarHalf; ++dj) {
            const bool guard_col = (dj >= -kCfarGuard && dj <= kCfarGuard);
            if (guard_row && guard_col) continue;    // inside the guard block (incl. the CUT itself): skip
            cells[count++] = rd[(i + di) * kNc + (j + dj)];
        }
    }
    return count;   // always == kCfarNTrain; returned for the (documented, defensive) assert-by-construction
}

// cfar_ca_kernel — CELL-AVERAGING CFAR: threshold = kAlphaCA * mean(training).
// The classical CFAR detector: assumes the training cells are i.i.d.
// samples of the local noise/clutter power, and that their SAMPLE MEAN is
// a good estimate of its true mean — true when the window is genuinely
// homogeneous clutter, and BADLY biased the moment a second target (or a
// clutter edge) sits inside the window: one contaminated cell drags the
// arithmetic mean up, raising the threshold for every other cell in the
// window — including a genuinely weaker target sitting nearby (THEORY.md
// "The algorithm" walks the exact masking scenario this project measures).
__global__ void cfar_ca_kernel(const float* __restrict__ rd, unsigned char* __restrict__ det,
                               float* __restrict__ thresh)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // range index
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // doppler index
    if (i >= kNs || j >= kNc) return;
    const int idx = i * kNc + j;

    // Border cells narrower than the window's half-width cannot form a
    // full training window; they are simply never flagged (matches the
    // CPU oracle and the earlier Python design prototype exactly).
    if (i < kCfarHalf || i >= kNs - kCfarHalf || j < kCfarHalf || j >= kNc - kCfarHalf) {
        det[idx] = 0;
        thresh[idx] = 0.0f;
        return;
    }

    float cells[kCfarNTrain];
    gather_training_cells(rd, i, j, cells);

    float sum = 0.0f;
    for (int k = 0; k < kCfarNTrain; ++k) sum += cells[k];
    const float z = sum / static_cast<float>(kCfarNTrain);
    const float t = kAlphaCA * z;

    thresh[idx] = t;
    det[idx] = (rd[idx] > t) ? 1 : 0;
}

// cfar_os_kernel — ORDERED-STATISTIC CFAR: threshold = kAlphaOS *
// training[kOsRankIndex] (the kOsRankFrac-quantile of the SORTED training
// cells, e.g. the 75th percentile / "3rd quartile"). Sorting first and
// reading a FIXED RANK makes the statistic robust to a MINORITY of
// contaminated cells: a single strong interfering target contributes at
// most one (or a handful of) outlier value(s), which land at the TOP of
// the sorted list and never influence a rank chosen well below the
// maximum. That robustness is not free — THEORY.md quantifies the price:
// a slightly higher variance (hence, in practice, a slightly higher
// realized false-alarm rate) than CA-CFAR under pure homogeneous noise,
// which this project's own measured detections.csv shows honestly.
//
// Sort: a plain O(N^2) insertion sort over the kCfarNTrain-element local
// array. Deliberately the simplest correct sort, not the fastest — CLAUDE.md
// §1's "teaching beats cleverness" applied literally; THEORY.md names
// nth_element/quickselect (O(N) average, no full ordering needed since we
// only read ONE rank) as the production optimization this project skips.
__global__ void cfar_os_kernel(const float* __restrict__ rd, unsigned char* __restrict__ det,
                               float* __restrict__ thresh)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kNs || j >= kNc) return;
    const int idx = i * kNc + j;

    if (i < kCfarHalf || i >= kNs - kCfarHalf || j < kCfarHalf || j >= kNc - kCfarHalf) {
        det[idx] = 0;
        thresh[idx] = 0.0f;
        return;
    }

    float cells[kCfarNTrain];
    gather_training_cells(rd, i, j, cells);

    // Insertion sort, ascending. Simple, in-place, O(N^2) worst/typical —
    // fine at N = kCfarNTrain = 200 for a one-shot teaching demo (see the
    // kernel's header comment above for the honest cost/production note).
    for (int a = 1; a < kCfarNTrain; ++a) {
        const float key = cells[a];
        int b = a - 1;
        while (b >= 0 && cells[b] > key) {
            cells[b + 1] = cells[b];
            --b;
        }
        cells[b + 1] = key;
    }

    const float z = cells[kOsRankIndex];
    const float t = kAlphaOS * z;

    thresh[idx] = t;
    det[idx] = (rd[idx] > t) ? 1 : 0;
}

// Shared launch geometry for both CFAR kernels: a 2-D block over (doppler,
// range) — 16x16 threads is a comfortable warp-multiple default; the grid
// covers the full Ns x Nc map (border cells self-exclude inside the kernel).
static void launch_cfar_common(void (*kernel)(const float*, unsigned char*, float*),
                               const float* d_rd_power, unsigned char* d_det, float* d_thresh,
                               const char* what)
{
    dim3 block(16, 16);
    dim3 grid((kNc + block.x - 1) / block.x, (kNs + block.y - 1) / block.y);
    kernel<<<grid, block>>>(d_rd_power, d_det, d_thresh);
    CUDA_CHECK_LAST_ERROR(what);
}

void launch_cfar_ca(const float* d_rd_power, unsigned char* d_det, float* d_thresh)
{
    launch_cfar_common(cfar_ca_kernel, d_rd_power, d_det, d_thresh, "cfar_ca_kernel launch");
}

void launch_cfar_os(const float* d_rd_power, unsigned char* d_det, float* d_thresh)
{
    launch_cfar_common(cfar_os_kernel, d_rd_power, d_det, d_thresh, "cfar_os_kernel launch");
}

// ===========================================================================
// 7)-9) Per-detection angle estimation
// ===========================================================================

// gather_angle_snapshots_kernel — one thread per (detection, output bin).
// For bin < kNa: copy the real antenna sample cube[kr,kd,bin]. For
// bin >= kNa: write zero (the zero-padding that lets the kNaFft-point FFT
// INTERPOLATE the true kNa-point spectrum for a sharper peak read-out —
// it does not add true resolving power; see kernels.cuh's kNaFft comment).
__global__ void gather_angle_snapshots_kernel(const ComplexF32* __restrict__ cube,
                                              const int* __restrict__ kr, const int* __restrict__ kd,
                                              int num_det, ComplexF32* __restrict__ snapshots)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // combined (detection, bin) index
    if (idx >= num_det * kNaFft) return;
    const int det_i = idx / kNaFft;
    const int bin = idx % kNaFft;

    if (bin < kNa) {
        const size_t cube_idx = static_cast<size_t>(kr[det_i]) * kNc * kNa
                              + static_cast<size_t>(kd[det_i]) * kNa + bin;
        snapshots[idx] = cube[cube_idx];
    } else {
        snapshots[idx].re = 0.0f;
        snapshots[idx].im = 0.0f;
    }
}

void launch_gather_angle_snapshots(const ComplexF32* d_cube, const int* d_kr, const int* d_kd,
                                   int num_det, ComplexF32* d_snapshots)
{
    if (num_det <= 0) return;
    const int total = num_det * kNaFft;
    const int block = 128;
    const int grid = (total + block - 1) / block;
    gather_angle_snapshots_kernel<<<grid, block>>>(d_cube, d_kr, d_kd, num_det, d_snapshots);
    CUDA_CHECK_LAST_ERROR("gather_angle_snapshots_kernel launch");
}

// launch_angle_fft — the project's THIRD and simplest cuFFT usage: a plain
// CONTIGUOUS batch (istride = 1, idist = kNaFft — the DEFAULT layout
// cufftPlanMany reduces to when data is already packed batch-after-batch
// in memory, no advanced-layout parameters needed). Deliberately placed
// beside launch_range_fft/launch_doppler_fft so a reader sees the full
// spectrum of cuFFT batching: contiguous (here), strided-outer-axis
// (range), and strided-middle-axis-looped (Doppler).
void launch_angle_fft(ComplexF32* d_snapshots, int num_det)
{
    if (num_det <= 0) return;
    cufftHandle plan;
    int n[1] = { kNaFft };
    // batch = num_det, contiguous: n[]/1/kNaFft is cufftPlanMany's
    // "default" (unstrided) advanced-layout form, spelled out explicitly
    // here for symmetry with the two calls above rather than calling the
    // simpler cufftPlan1d + cufftExecC2C-with-implicit-batch overload.
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                             n, 1, kNaFft,
                             n, 1, kNaFft,
                             CUFFT_C2C, num_det));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_snapshots),
                             reinterpret_cast<cufftComplex*>(d_snapshots), CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));
}

// find_angle_peaks_kernel — one thread per detection: scan its kNaFft
// spectrum for the peak-MAGNITUDE bin (argmax |X[k]|), convert the bin
// index to a centered (fftshift-equivalent) index, then to azimuth via
// the ULA steering relation sin(theta) = 2*k_centered/kNaFft (derived in
// THEORY.md "The math" from the same d=lambda/2 spacing used in synthesis).
// A small, fully serial per-thread scan over 64 elements — far too small
// to parallelize further without paying more in synchronization than the
// scan itself costs (occupancy, not per-thread speed, is what matters
// here: num_det independent threads, not one thread doing everything).
__global__ void find_angle_peaks_kernel(const ComplexF32* __restrict__ snapshots,
                                        int num_det, float* __restrict__ az_deg_out)
{
    const int det_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (det_i >= num_det) return;

    const ComplexF32* spec = snapshots + static_cast<size_t>(det_i) * kNaFft;
    int best_bin = 0;
    float best_mag2 = -1.0f;
    for (int k = 0; k < kNaFft; ++k) {
        const float re = spec[k].re, im = spec[k].im;
        const float mag2 = re * re + im * im;
        if (mag2 > best_mag2) { best_mag2 = mag2; best_bin = k; }
    }

    // Un-shift: cuFFT's natural bin order is [0, +1, ..., +N/2-1, -N/2,
    // ..., -1] (DC first); recover the SIGNED, centered bin index.
    const int k_centered = (best_bin < kNaFft / 2) ? best_bin : best_bin - kNaFft;
    float sin_theta = 2.0f * static_cast<float>(k_centered) / static_cast<float>(kNaFft);
    sin_theta = fminf(1.0f, fmaxf(-1.0f, sin_theta));   // guard the asinf domain against fp roundoff at +/-1
    az_deg_out[det_i] = asinf(sin_theta) * (180.0f / 3.14159265358979323846f);
}

void launch_find_angle_peaks(const ComplexF32* d_snapshots, int num_det, float* d_az_deg)
{
    if (num_det <= 0) return;
    const int block = 64;
    const int grid = (num_det + block - 1) / block;
    find_angle_peaks_kernel<<<grid, block>>>(d_snapshots, num_det, d_az_deg);
    CUDA_CHECK_LAST_ERROR("find_angle_peaks_kernel launch");
}
