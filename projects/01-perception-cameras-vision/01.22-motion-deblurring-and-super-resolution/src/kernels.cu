// ===========================================================================
// kernels.cu — GPU kernels + cuFFT wrappers for project 01.22
//              Motion deblurring and super-resolution for inspection zoom
//
// Role in the project
// --------------------
// Every __global__ (GPU) kernel lives here, plus the host-side launch
// wrappers that own grid/block math AND (for the FFT-backed calls) the
// cuFFT plan lifecycle. Two families of GPU work:
//   1. milestone 1 (deblurring): naive_inverse_kernel / wiener_kernel /
//      scale_real_kernel (frequency-domain MAPs, one thread per complex
//      bin) + convolve_circular_kernel / divide_safe_kernel /
//      multiply_inplace_kernel (spatial-domain Richardson-Lucy).
//   2. milestone 2 (super-resolution): shift_and_add_kernel (the project's
//      one SCATTER kernel — atomics, see its header) + finalize_splat_kernel
//      / forward_simulate_kernel / backproject_kernel / bicubic_upscale_
//      kernel (all GATHER kernels — deterministic, no atomics).
//
// cuFFT as a "no black box" library call (CLAUDE.md §1, the 03.01
// precedent cited by name in kernels.cuh): cuFFT computes the SAME
// discrete Fourier transform this project's reference_cpu.cpp derives from
// scratch (a radix-2 Cooley-Tukey FFT, see that file's header) — an
// O(N log N) divide-and-conquer factorization of the O(N^2) DFT sum. We use
// cuFFT here rather than hand-rolling a CUDA FFT because a GOOD parallel
// FFT (bit-reversal permutation, twiddle-factor generation, multi-pass
// butterfly scheduling with shared-memory staging) is itself a multi-week
// teaching project on its own (THEORY.md sketches the shape); this
// project's subject is DECONVOLUTION, not FFT internals, so cuFFT plays the
// same role here that it plays in 03.01's radar-cube processing. The CPU
// twin (reference_cpu.cpp) hand-rolls its OWN independent FFT precisely so
// this project is not just "trust cuFFT" — the twin-agreement VERIFY gate
// in main.cu proves cuFFT and a from-scratch implementation agree.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cufft.h>               // cuFFT: cufftPlan2d / cufftExecR2C / cufftExecC2R (see the wrappers below)

#include "kernels.cuh"           // our own interface — keeps decl/def in sync at compile time
#include "util/cuda_check.cuh"   // CUDA_CHECK_LAST_ERROR for post-launch error surfacing

// ---------------------------------------------------------------------------
// CUFFT_CHECK — cuFFT returns its OWN error enum (cufftResult), not
// cudaError_t, so CUDA_CHECK (which formats cudaError_t) cannot check a
// cuFFT call directly. This macro is the 03.01 precedent verbatim: print the
// numeric code (cuFFT does not ship a cufftGetErrorString) and hard-exit,
// the same "never ignore a library error" discipline CUDA_CHECK enforces
// for the runtime API (CLAUDE.md §6.1 rule 7).
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                     \
    do {                                                                      \
        cufftResult cufft_err__ = (call);                                     \
        if (cufft_err__ != CUFFT_SUCCESS) {                                   \
            std::fprintf(stderr, "cuFFT error %d at %s:%d in '%s'\n",         \
                         static_cast<int>(cufft_err__), __FILE__, __LINE__,   \
                         #call);                                              \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                     \
    } while (0)

// ===========================================================================
// PART A — milestone 1: frequency-domain kernels (post-cuFFT pointwise ops).
// ===========================================================================

// naive_inverse_kernel — the DESIGNED FAILURE. Divides the blurred frame's
// spectrum by the PSF's spectrum with only a numerically-necessary epsilon
// floor (kNaiveInverseEpsilon; NOT a regularizer, see kernels.cuh). At a
// frequency bin where the PSF's response is near zero (a line PSF is
// sinc-like along its motion axis — THEORY.md plots this), this divides
// mostly-NOISE by a near-zero number: the result explodes. That explosion
// is the whole teaching point (README/THEORY "naive_inverse_failure").
//
// Thread mapping: one thread per COMPLEX frequency bin, i = blockIdx.x*
// blockDim.x + threadIdx.x, i in [0, kFreqN). A pure map: bin i's output
// depends only on bin i's two inputs — no shared memory, no cross-thread
// communication (identical mapping shape to SAXPY; see the scaffold notes
// this project replaced).
//
// Complex division a/b = a * conj(b) / |b|^2 — the standard numerically-
// sane form (avoids forming 1/b directly, which would itself divide by
// zero at |b|=0 before we ever get to use the epsilon floor).
__global__ void naive_inverse_kernel(const ComplexF32* __restrict__ blurred_freq,
                                     const ComplexF32* __restrict__ psf_freq,
                                     ComplexF32* __restrict__ out_freq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kFreqN) return;

    const ComplexF32 y = blurred_freq[i];  // Y(f): the blurred+noisy frame's spectrum at this bin
    const ComplexF32 h = psf_freq[i];      // H(f): the PSF's spectrum at this bin
    // |H(f)|^2 + epsilon: epsilon is orders of magnitude smaller than a
    // "real" regularizer would use (compare kWienerK below) — just enough
    // to keep this line finite, which is exactly what makes the explosion
    // at near-zero H(f) bins visible instead of literally infinite/NaN.
    const float denom = h.re * h.re + h.im * h.im + kNaiveInverseEpsilon;
    // X(f) = Y(f) * conj(H(f)) / denom  ==  Y(f) / H(f) when denom ~= |H(f)|^2
    out_freq[i].re = (y.re * h.re + y.im * h.im) / denom;
    out_freq[i].im = (y.im * h.re - y.re * h.im) / denom;
}

// wiener_kernel — the REGULARIZED inverse. Same complex-division shape as
// naive_inverse_kernel, but denom adds K (an estimate of the noise-to-
// signal POWER ratio, kernels.cuh "Wiener / naive-inverse regularization
// constants") INSTEAD of a numerically-necessary epsilon. Where H(f) is
// large, K is negligible and this behaves like the naive inverse (recovers
// detail); where H(f) is small, K DOMINATES the denominator and the filter
// deliberately rolls off toward zero instead of amplifying noise — the
// MMSE trade-off THEORY.md derives from first principles ("The math").
__global__ void wiener_kernel(const ComplexF32* __restrict__ blurred_freq,
                              const ComplexF32* __restrict__ psf_freq,
                              ComplexF32* __restrict__ out_freq, float K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kFreqN) return;

    const ComplexF32 y = blurred_freq[i];
    const ComplexF32 h = psf_freq[i];
    const float denom = h.re * h.re + h.im * h.im + K;   // the ONLY line that differs from naive_inverse_kernel
    out_freq[i].re = (y.re * h.re + y.im * h.im) / denom;
    out_freq[i].im = (y.im * h.re - y.re * h.im) / denom;
}

// scale_real_kernel — cuFFT's inverse transforms are UNNORMALIZED by
// convention (cufftExecC2R(FFT(x)) == N*x, not x — a documented cuFFT
// property that saves cuFFT a pass over the data when a caller wants to
// fold the 1/N scale into a later step; we do not, for clarity, so this
// kernel applies it explicitly). One thread per pixel, in place.
__global__ void scale_real_kernel(float* __restrict__ img, int n, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) img[i] *= scale;
}

// ---------------------------------------------------------------------------
// convolve_circular_kernel — dense kPsfSize x kPsfSize stencil with
// WRAPAROUND (circular) indexing, computing
//     out[y][x] = sum_{ky,kx} psf[ky][kx] * img[(y+ky-r+H)%H][(x+kx-r+W)%W]
// where r = kPsfRadius. Circular indexing is the SPATIAL-DOMAIN twin of
// frequency-domain multiplication (a discrete convolution theorem fact,
// THEORY.md "The math"): this formula, applied to the PSF placed at the
// SAME center convention as build_padded_psf() in main.cu, produces
// BIT-COMPARABLE results to going through cuFFT — the two Richardson-Lucy
// convolutions per iteration therefore never need cuFFT at all (an O(W*H*
// kPsfSize^2) direct stencil, cheap at this project's 128x128/15x15 sizes:
// ~2.76M taps per convolution).
//
// Thread mapping: one thread per OUTPUT pixel (i = flattened y*W+x, i in
// [0,kN)) — a classic 2-D stencil, one thread per output element, same
// family as 01.11's bilateral filter (cited by name: that project's
// kernels.cu walks the shared-memory TILING optimization this project does
// NOT apply — kPsfSize=15 already keeps each thread's working set small
// enough that global-memory re-reads are not this project's bottleneck;
// README "Exercises" suggests tiling as a follow-up).
//
// psf is a DEVICE pointer to exactly kPsfSize*kPsfSize floats — callers
// pass either the true PSF (RL's "reblur" convolution) or its 180-degree
// ROTATION (RL's "correlation" convolution, THEORY.md derives why
// correlation-with-the-flipped-kernel implements the adjoint operator the
// EM update needs) — kernels.cu never flips a kernel itself; main.cu
// prepares both device buffers once, up front (see build_psf_flipped()).
// ---------------------------------------------------------------------------
__global__ void convolve_circular_kernel(const float* __restrict__ img,
                                         const float* __restrict__ psf,
                                         float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kN) return;
    const int x = i % kW;
    const int y = i / kW;

    // Register accumulator: kPsfSize^2=225 taps read from GLOBAL memory
    // (img) per output pixel. No shared memory: unlike 01.11's bilateral
    // tiling lesson, this project keeps the stencil simple and spends the
    // teaching budget on the deconvolution MATH instead (README
    // "Exercises" names tiling as the natural next optimization).
    float acc = 0.0f;
    for (int ky = 0; ky < kPsfSize; ++ky) {
        // Wraparound row index: circular convolution needs img[(y+ky-r)
        // mod H], and C++'s % can return negative for negative operands,
        // so we add H before the mod (the standard idiom, seen throughout
        // this repo wherever wraparound indexing appears).
        const int sy = (y + ky - kPsfRadius + kH) % kH;
        for (int kx = 0; kx < kPsfSize; ++kx) {
            const int sx = (x + kx - kPsfRadius + kW) % kW;
            acc += psf[ky * kPsfSize + kx] * img[sy * kW + sx];
        }
    }
    out[i] = acc;
}

// divide_safe_kernel — out[i] = a[i] / max(b[i], eps). The FIRST of
// Richardson-Lucy's two per-iteration MAPs: the ratio (measured blurred
// frame) / (current estimate's simulated reblur). eps guards a legitimate
// near-zero denominator (a dark image region) the same defensive way
// kRlEpsilon is documented in kernels.cuh.
__global__ void divide_safe_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ out, int n, float eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        const float denom = b[i] > eps ? b[i] : eps;
        out[i] = a[i] / denom;
    }
}

// multiply_inplace_kernel — a[i] *= b[i]. Richardson-Lucy's SECOND
// per-iteration map: the estimate is multiplicatively corrected by the
// back-projected ratio (THEORY.md "The math" derives why this update
// provably increases the Poisson data likelihood every iteration and
// therefore never needs a step-size / learning-rate parameter, unlike
// gradient methods).
__global__ void multiply_inplace_kernel(float* __restrict__ a, const float* __restrict__ b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) a[i] *= b[i];
}

// subtract_kernel — out[i] = a[i] - b[i]. Used by IBP's per-iteration
// residual computation (main.cu: residual = actual_lr - predicted_lr,
// between forward_simulate_kernel and backproject_kernel below) — a plain
// elementwise map, the simplest kernel in this file.
__global__ void subtract_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = a[i] - b[i];
}

// ===========================================================================
// PART B — milestone 2: super-resolution kernels.
// ===========================================================================

// shift_and_add_kernel — THE ONE SCATTER KERNEL in this project (contrast
// with every other kernel here, which GATHERS). Each of kNumFrames LR
// frames' kLrN pixels is a physical measurement at a KNOWN sub-pixel
// location on the 2x HR grid (shifts[f], kernels.cuh); we bilinearly SPLAT
// that one measurement's value and weight into the (up to) four HR cells
// its footprint touches. Because eight independently-shifted frames land
// their footprints at DIFFERENT, overlapping HR locations, MANY threads
// (from different frames, or even different pixels of the SAME frame near
// a shared boundary) may write the SAME hr_sum[]/hr_weight[] cell in the
// same launch — a genuine race UNLESS every write is atomic. This is the
// scatter-vs-gather teaching contrast the task brief asks for: compare this
// kernel to forward_simulate_kernel/backproject_kernel below, which invert
// the SAME geometric relationship into a GATHER and need no atomics at all
// (THEORY.md "The GPU mapping" draws the contrast out fully; the 01.11
// BM3D-lite group kernel is this repo's earlier atomic-scatter precedent,
// cited by name in kernels.cuh's twin-independence note).
//
// Thread mapping: one thread per (frame, LR pixel) pair, flattened over
// kLrFramesN = kNumFrames*kLrN threads. frame = i / kLrN, lrIdx = i % kLrN.
__global__ void shift_and_add_kernel(const float* __restrict__ lr_frames,
                                     const Shift* __restrict__ shifts,
                                     float* __restrict__ hr_sum,
                                     float* __restrict__ hr_weight)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kLrFramesN) return;
    const int frame = i / kLrN;
    const int lrIdx = i % kLrN;
    const int lu = lrIdx % kLrW;   // this LR pixel's column
    const int lv = lrIdx / kLrW;   // this LR pixel's row

    const Shift s = shifts[frame];
    // This LR sample's CONTINUOUS location on the HR grid: an LR pixel at
    // integer (lu,lv) with sub-pixel registration shift (dx,dy) LR-px sits
    // at HR coordinate ((lu+dx)*scale + (scale-1)/2, ...) — the +
    // (scale-1)/2 term centers the LR pixel's footprint inside its scale x
    // scale HR block (kLrScale=2 -> +0.5), matching the box-DOWNSAMPLE
    // forward model make_synthetic.py used to CREATE this LR pixel
    // (kernels.cuh Section 3's header) so the round trip is geometrically
    // consistent.
    const float hx = (static_cast<float>(lu) + s.dx_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
    const float hy = (static_cast<float>(lv) + s.dy_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
    const float value = lr_frames[frame * kLrN + lrIdx];

    // Bilinear SPLAT: distribute `value` (with unit total weight) across
    // the four HR cells surrounding (hx,hy), weighted by area-overlap
    // (the standard bilinear "adjoint of gather" splat). bilinear_sample_
    // at() clamps to the valid HR range, so a footprint that would spill
    // past the border degrades to edge accumulation instead of a stray
    // out-of-bounds write.
    const BilinearSample bs = bilinear_sample_at(hx, hy, kW, kH);
    const float w00 = (1.0f - bs.wx) * (1.0f - bs.wy);
    const float w10 = bs.wx * (1.0f - bs.wy);
    const float w01 = (1.0f - bs.wx) * bs.wy;
    const float w11 = bs.wx * bs.wy;
    const int i00 = bs.y0 * kW + bs.x0;
    const int i10 = i00 + 1;
    const int i01 = i00 + kW;
    const int i11 = i01 + 1;

    // atomicAdd: unavoidable here (see the kernel header) — the ORDER in
    // which different threads' contributions land is nondeterministic,
    // which is exactly why main.cu's VERIFY tolerance for this method is
    // the loosest of the project (the 01.11 BM3D-lite precedent for
    // "atomic float vs. fixed-order double CPU twin").
    atomicAdd(&hr_sum[i00], value * w00); atomicAdd(&hr_weight[i00], w00);
    atomicAdd(&hr_sum[i10], value * w10); atomicAdd(&hr_weight[i10], w10);
    atomicAdd(&hr_sum[i01], value * w01); atomicAdd(&hr_weight[i01], w01);
    atomicAdd(&hr_sum[i11], value * w11); atomicAdd(&hr_weight[i11], w11);
}

// finalize_splat_kernel — hr_out[i] = hr_sum[i]/hr_weight[i], with
// `fallback` (main.cu passes the bicubic upscale) used defensively where
// weight is ~0 (should never trigger given 8 frames' quarter-pixel-lattice
// coverage — main.cu's [info] line reports how often it does, the 01.11
// BM3D-lite "never divide by zero, never emit garbage" discipline).
__global__ void finalize_splat_kernel(const float* __restrict__ hr_sum, const float* __restrict__ hr_weight,
                                      const float* __restrict__ fallback, float* __restrict__ hr_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kN) return;
    const float w = hr_weight[i];
    hr_out[i] = (w > 1.0e-6f) ? (hr_sum[i] / w) : fallback[i];
}

// forward_simulate_kernel — GATHER half of iterative back-projection's
// forward model: "if the true scene were the current HR estimate, what
// would frame f's sensor have measured at LR pixel (lu,lv)?" We invert
// shift_and_add_kernel's forward geometry algebraically and BILINEARLY
// SAMPLE (gather) the HR estimate at that continuous location — the
// bilinear kernel is this project's stand-in for the sensor's true box-
// integration PSF (README "Limitations" states this simplification; a
// closer forward model would gather a small area, not one bilinear sample
// — THEORY.md "Where this sits in the real world" names the fix).
// Because each output depends on a BOUNDED, statically-known set of four
// HR pixels, no two threads ever write the same output element: no
// atomics, unlike shift_and_add_kernel above (the scatter/gather contrast
// this project is built to teach).
//
// Thread mapping: one thread per (frame, LR pixel), same flattening as
// shift_and_add_kernel.
__global__ void forward_simulate_kernel(const float* __restrict__ hr_estimate,
                                        const Shift* __restrict__ shifts,
                                        float* __restrict__ lr_predicted)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kLrFramesN) return;
    const int frame = i / kLrN;
    const int lrIdx = i % kLrN;
    const int lu = lrIdx % kLrW;
    const int lv = lrIdx / kLrW;

    const Shift s = shifts[frame];
    const float hx = (static_cast<float>(lu) + s.dx_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
    const float hy = (static_cast<float>(lv) + s.dy_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
    const BilinearSample bs = bilinear_sample_at(hx, hy, kW, kH);
    const int i00 = bs.y0 * kW + bs.x0, i10 = i00 + 1, i01 = i00 + kW, i11 = i01 + 1;
    const float v = (1.0f - bs.wx) * (1.0f - bs.wy) * hr_estimate[i00]
                  + bs.wx * (1.0f - bs.wy) * hr_estimate[i10]
                  + (1.0f - bs.wx) * bs.wy * hr_estimate[i01]
                  + bs.wx * bs.wy * hr_estimate[i11];
    lr_predicted[i] = v;
}

// backproject_kernel — GATHER half of IBP's correction step. For every HR
// pixel, walk all kNumFrames residual maps (actual - predicted, computed by
// main.cu between forward_simulate and this call) and GATHER each frame's
// residual at the INVERSE-mapped LR coordinate — the algebraic inverse of
// shift_and_add_kernel's forward map, so "which LR sample would this HR
// pixel have contributed to" is answered by direct formula, not a search.
// hr_estimate[i] += step * (sum of kNumFrames bilinearly-gathered
// residuals) / kNumFrames — averaging over frames keeps the step stable
// regardless of kNumFrames (THEORY.md "Numerical considerations").
//
// Thread mapping: one thread per HR pixel (kN threads) — EVERY thread
// writes exactly ONE hr_estimate[] element (itself), reading up to
// kNumFrames*4 residual samples: no atomics, no cross-thread writes at all.
__global__ void backproject_kernel(const float* __restrict__ residual,
                                   const Shift* __restrict__ shifts,
                                   float* __restrict__ hr_estimate,
                                   float step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kN) return;
    const int hxi = i % kW;
    const int hyi = i / kW;

    float acc = 0.0f;
    for (int f = 0; f < kNumFrames; ++f) {
        const Shift s = shifts[f];
        // Invert forward_simulate_kernel's map: HR (hxi,hyi) -> continuous
        // LR coordinate for frame f. This is the exact algebraic inverse
        // of "hx = (lu+dx)*scale + (scale-1)/2" solved for lu.
        const float lu = (static_cast<float>(hxi) - 0.5f * (kLrScale - 1)) / kLrScale - s.dx_lrpx;
        const float lv = (static_cast<float>(hyi) - 0.5f * (kLrScale - 1)) / kLrScale - s.dy_lrpx;
        const BilinearSample bs = bilinear_sample_at(lu, lv, kLrW, kLrH);
        const int base = f * kLrN;
        const int i00 = base + bs.y0 * kLrW + bs.x0, i10 = i00 + 1, i01 = i00 + kLrW, i11 = i01 + 1;
        acc += (1.0f - bs.wx) * (1.0f - bs.wy) * residual[i00]
             + bs.wx * (1.0f - bs.wy) * residual[i10]
             + (1.0f - bs.wx) * bs.wy * residual[i01]
             + bs.wx * bs.wy * residual[i11];
    }
    hr_estimate[i] += step * (acc / static_cast<float>(kNumFrames));
}

// bicubic_upscale_kernel — the "cannot recover aliased detail" BASELINE
// (README/THEORY): upsamples a SINGLE LR frame (main.cu passes frame 0, the
// zero-shift reference) to the HR grid with the classic separable cubic
// convolution kernel (Keys, 1981; a=-0.5 — the parameter value that makes
// the kernel exactly reproduce a Taylor-series-accurate interpolant for
// smooth signals, the most common "bicubic" default). Because it uses ONE
// frame's samples, it can only ever interpolate BETWEEN measured samples —
// it cannot un-alias content the LR sampling grid never captured (THEORY.md
// "The problem" derives why: interpolation is a LOW-PASS operation on the
// samples it has, not a source of new information).
//
// Thread mapping: one thread per HR output pixel (kN threads), each
// gathering a 4x4 neighborhood of LR samples (16 reads) — a stencil, same
// shape family as convolve_circular_kernel but with data-dependent
// (fractional-position) weights instead of a fixed kernel array.
HD inline float cubic_weight(float t, float a)
{
    // Keys' cubic convolution kernel, |t|<=2 support:
    //   |t|<=1: (a+2)|t|^3 - (a+3)|t|^2 + 1
    //   1<|t|<2: a|t|^3 - 5a|t|^2 + 8a|t| - 4a
    const float at = t < 0.0f ? -t : t;
    if (at <= 1.0f) return (a + 2.0f) * at * at * at - (a + 3.0f) * at * at + 1.0f;
    if (at < 2.0f)  return a * at * at * at - 5.0f * a * at * at + 8.0f * a * at - 4.0f * a;
    return 0.0f;
}

__global__ void bicubic_upscale_kernel(const float* __restrict__ lr, float* __restrict__ hr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kN) return;
    const int hx = i % kW, hy = i / kW;
    const float a = -0.5f;   // the standard "bicubic" parameter (Keys 1981)

    // Map the HR pixel CENTER back to a continuous LR coordinate (the
    // standard "align pixel centers" convention: HR pixel 0 sits half an
    // HR-pixel in, which is a quarter-LR-pixel in — consistent with how
    // kLrScale=2 relates the two grids without a half-pixel offset error
    // at the image boundary).
    const float lx = (static_cast<float>(hx) + 0.5f) / kLrScale - 0.5f;
    const float ly = (static_cast<float>(hy) + 0.5f) / kLrScale - 0.5f;
    const int ix = static_cast<int>(floorf(lx));
    const int iy = static_cast<int>(floorf(ly));
    const float fx = lx - static_cast<float>(ix);
    const float fy = ly - static_cast<float>(iy);

    float acc = 0.0f;
    for (int dy = -1; dy <= 2; ++dy) {
        int sy = iy + dy; sy = sy < 0 ? 0 : (sy > kLrH - 1 ? kLrH - 1 : sy);   // clamp-to-edge border
        const float wy = cubic_weight(static_cast<float>(dy) - fy, a);
        for (int dx = -1; dx <= 2; ++dx) {
            int sx = ix + dx; sx = sx < 0 ? 0 : (sx > kLrW - 1 ? kLrW - 1 : sx);
            const float wx = cubic_weight(static_cast<float>(dx) - fx, a);
            acc += wx * wy * lr[sy * kLrW + sx];
        }
    }
    hr[i] = acc;
}

// ===========================================================================
// PART C — host-callable launch wrappers.
//
// Launch configuration reasoning (shared by every wrapper below, stated
// once): block=256 threads (warp multiple, the repo-wide default reasoned
// through in the scaffold's saxpy_kernel comment this project replaced);
// grid=ceil(n/block) with NO 4096-block cap here, unlike the SAXPY
// placeholder — this project's largest launch (kLrFramesN=32,768 threads)
// needs only 128 blocks, comfortably inside a single launch's natural
// grid-size limits, so a grid-stride loop is not needed for correctness
// (used anyway in the elementwise map kernels above for robustness against
// future size changes; the per-pixel kernels below use the simple "one
// thread, early-return" form because their thread counts are small and
// fixed by kN/kLrFramesN/kFreqN, matching the 01.11 per-pixel-stencil style).
// ===========================================================================
static inline int grid_for(int n, int block = 256) { return (n + block - 1) / block; }

// -- cuFFT wrappers -----------------------------------------------------------
// A fresh cufftPlan2d per call (rather than a cached, project-lifetime
// plan) is a deliberate teaching-clarity choice: every call site is then a
// single, self-contained "here is a complete FFT" statement a reader can
// study in isolation, at the cost of repeated plan-creation overhead
// (measured in main.cu's [time] line — a few hundred microseconds per call,
// negligible next to this project's other costs, and explicitly a teaching
// trade-off, not a claim about production performance; THEORY.md "Where
// this sits in the real world" notes production code caches plans).
void launch_fft_forward_r2c(const float* d_img, ComplexF32* d_freq)
{
    cufftHandle plan;
    // cufftPlan2d(..., ny, nx, ...): cuFFT names its 2-D plan arguments in
    // ROW-major (ny=height first) order — matches this project's row-major
    // [y][x] image layout with no transpose needed.
    CUFFT_CHECK(cufftPlan2d(&plan, kH, kW, CUFFT_R2C));
    // cufftExecR2C takes a non-const real input; our caller's d_img is
    // logically read-only but cuFFT's C API is not const-correct here — a
    // const_cast is the standard, documented workaround (03.01 uses the
    // same pattern), never a sign the data is actually mutated.
    CUFFT_CHECK(cufftExecR2C(plan, const_cast<float*>(d_img),
                             reinterpret_cast<cufftComplex*>(d_freq)));
    CUFFT_CHECK(cufftDestroy(plan));
}

void launch_fft_inverse_c2r(const ComplexF32* d_freq, float* d_img)
{
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, kH, kW, CUFFT_C2R));
    // C2R also DESTROYS its complex input buffer as working scratch space
    // (a documented cuFFT behavior) — every caller in this project treats
    // its frequency-domain buffers as single-use after an inverse transform,
    // which main.cu's orchestration respects (it never reads a frequency
    // buffer again after passing it here).
    CUFFT_CHECK(cufftExecC2R(plan, const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(d_freq)),
                             d_img));
    CUFFT_CHECK(cufftDestroy(plan));
    // cuFFT's C2R is UNNORMALIZED (see scale_real_kernel's header) —
    // every caller of launch_fft_inverse_c2r is responsible for the 1/N
    // scale; main.cu applies it explicitly right after this call so the
    // "unnormalized" fact is never silently baked into a magic constant.
}

void launch_naive_inverse(const ComplexF32* d_blurred_freq, const ComplexF32* d_psf_freq, ComplexF32* d_out_freq)
{
    naive_inverse_kernel<<<grid_for(kFreqN), 256>>>(d_blurred_freq, d_psf_freq, d_out_freq);
    CUDA_CHECK_LAST_ERROR("naive_inverse_kernel launch");
}

void launch_wiener(const ComplexF32* d_blurred_freq, const ComplexF32* d_psf_freq, ComplexF32* d_out_freq, float K)
{
    wiener_kernel<<<grid_for(kFreqN), 256>>>(d_blurred_freq, d_psf_freq, d_out_freq, K);
    CUDA_CHECK_LAST_ERROR("wiener_kernel launch");
}

void launch_scale_real(float* d_img, int n, float scale)
{
    scale_real_kernel<<<grid_for(n), 256>>>(d_img, n, scale);
    CUDA_CHECK_LAST_ERROR("scale_real_kernel launch");
}

void launch_convolve_circular(const float* d_img, const float* d_psf, float* d_out)
{
    convolve_circular_kernel<<<grid_for(kN), 256>>>(d_img, d_psf, d_out);
    CUDA_CHECK_LAST_ERROR("convolve_circular_kernel launch");
}

void launch_divide_safe(const float* d_a, const float* d_b, float* d_out, int n, float eps)
{
    divide_safe_kernel<<<grid_for(n), 256>>>(d_a, d_b, d_out, n, eps);
    CUDA_CHECK_LAST_ERROR("divide_safe_kernel launch");
}

void launch_multiply_inplace(float* d_a, const float* d_b, int n)
{
    multiply_inplace_kernel<<<grid_for(n), 256>>>(d_a, d_b, n);
    CUDA_CHECK_LAST_ERROR("multiply_inplace_kernel launch");
}

void launch_subtract(const float* d_a, const float* d_b, float* d_out, int n)
{
    subtract_kernel<<<grid_for(n), 256>>>(d_a, d_b, d_out, n);
    CUDA_CHECK_LAST_ERROR("subtract_kernel launch");
}

void launch_shift_and_add(const float* d_lr_frames, const Shift* d_shifts, float* d_hr_sum, float* d_hr_weight)
{
    shift_and_add_kernel<<<grid_for(kLrFramesN), 256>>>(d_lr_frames, d_shifts, d_hr_sum, d_hr_weight);
    CUDA_CHECK_LAST_ERROR("shift_and_add_kernel launch");
}

void launch_finalize_splat(const float* d_hr_sum, const float* d_hr_weight, const float* d_fallback, float* d_hr_out)
{
    finalize_splat_kernel<<<grid_for(kN), 256>>>(d_hr_sum, d_hr_weight, d_fallback, d_hr_out);
    CUDA_CHECK_LAST_ERROR("finalize_splat_kernel launch");
}

void launch_forward_simulate(const float* d_hr_estimate, const Shift* d_shifts, float* d_lr_predicted)
{
    forward_simulate_kernel<<<grid_for(kLrFramesN), 256>>>(d_hr_estimate, d_shifts, d_lr_predicted);
    CUDA_CHECK_LAST_ERROR("forward_simulate_kernel launch");
}

void launch_backproject(const float* d_residual, const Shift* d_shifts, float* d_hr_estimate, float step)
{
    backproject_kernel<<<grid_for(kN), 256>>>(d_residual, d_shifts, d_hr_estimate, step);
    CUDA_CHECK_LAST_ERROR("backproject_kernel launch");
}

void launch_bicubic_upscale(const float* d_lr, float* d_hr)
{
    bicubic_upscale_kernel<<<grid_for(kN), 256>>>(d_lr, d_hr);
    CUDA_CHECK_LAST_ERROR("bicubic_upscale_kernel launch");
}
