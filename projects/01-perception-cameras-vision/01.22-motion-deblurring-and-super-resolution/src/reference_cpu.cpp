// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.22
//                     Motion deblurring and super-resolution for inspection zoom
//
// WHY does a GPU repository ship a CPU implementation of everything?
// (CLAUDE.md §5 — restated from the template): this file is BOTH the
// correctness oracle main.cu's VERIFY step compares the GPU path against,
// AND the teaching baseline that makes "what did parallelizing this change"
// legible.
//
// INDEPENDENCE RULING applied throughout this file (the template's Phase-1
// retrospective, restated in kernels.cuh's header and obeyed here exactly):
//   * Data-layout contracts (kW/kH/kPsfSize geometry, Rect/Shift structs,
//     the bilinear_sample_at() footprint arithmetic, PSF padding placement)
//     are single-sourced in kernels.cuh and SHARED — duplicating a struct
//     layout or an index formula is not "independence", it is a second
//     place for the same bug to hide.
//   * The ALGORITHMIC CORE — the DFT itself, the Wiener/naive-inverse
//     pointwise formulas, the Richardson-Lucy multiplicative update, the
//     shift-and-add splat loop, the IBP forward/back-projection loop, the
//     bicubic cubic-convolution weights — is written TWICE, independently.
//     Below, "independently" specifically means: this file's Fourier
//     transform is a from-scratch RADIX-2 COOLEY-TUKEY FFT (bit-reversal
//     permutation + iterative butterfly passes, double precision, textbook
//     form), never calling cuFFT or any FFT library — a genuinely different
//     implementation from kernels.cu's cuFFT-based R2C/C2R path, not a
//     recompilation of the same code for the host. This project's VERIFY
//     choice is therefore the "CPU FFT twin" option named in the task
//     brief (as opposed to a spatial-domain direct-convolution twin) —
//     documented once, here, because it is the stronger check: it exercises
//     cuFFT's correctness too, not just the pointwise math around it.
//   * Per the ruling's third bullet, this project ALSO carries independent
//     gates that never route through fft2d() or bilinear_sample_at(): every
//     PSNR/contrast/monotonicity gate in main.cu compares against the
//     COMMITTED ground-truth truth.pgm or checks a physical invariant
//     (naive inverse must be WORSE than doing nothing; IBP's reprojection
//     error must fall monotonically) — never merely "does the GPU agree
//     with this file", which alone could not catch a bug shared by both.
//
// Rules for this file: plain C++17, no CUDA headers (kernels.cuh's
// __CUDACC__ fence hides every __global__ declaration from cl.exe), no
// hand-vectorization, no OpenMP, no cleverness — clarity beats speed here,
// always, per the template.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include <vector>
#include <cmath>
#include <algorithm>   // std::max/min/swap

#include "kernels.cuh"

// ===========================================================================
// PART A — a from-scratch radix-2 Cooley-Tukey FFT, double precision.
//
// Complex64 — a plain complex number for THIS FILE ONLY (distinct from
// kernels.cuh's ComplexF32, which is the host<->device single-precision
// data-exchange type). Using double here is deliberate: this file is the
// correctness ORACLE, so it should be the more numerically trustworthy of
// the two implementations, not a second copy of the GPU's float32 rounding.
// ===========================================================================
namespace {

struct Complex64 {
    double re = 0.0, im = 0.0;
    Complex64 operator+(const Complex64& o) const { return { re + o.re, im + o.im }; }
    Complex64 operator-(const Complex64& o) const { return { re - o.re, im - o.im }; }
    Complex64 operator*(const Complex64& o) const { return { re * o.re - im * o.im, re * o.im + im * o.re }; }
};

// fft1d — in-place iterative radix-2 Cooley-Tukey transform of a length-n
// (n MUST be a power of two — kW=kH=128=2^7, kernels.cuh's header explains
// why this project's canvas is square-power-of-two specifically so this
// textbook algorithm applies with no mixed-radix complication) complex
// sequence. invert=false computes the forward DFT, invert=true the inverse
// (self-normalizing: divides by n at the end, so fft1d(fft1d(x,false),true)
// == x up to floating-point rounding — verified implicitly every time
// naive_inverse_cpu/wiener_cpu round-trip through fft2d below).
//
// Algorithm (two classic passes):
//   1. Bit-reversal permutation: swap a[i] and a[reverse_bits(i)] so the
//      butterfly network below can work purely in-place, output-in-order.
//   2. log2(n) butterfly passes: pass `len` combines pairs of length-len/2
//      transforms (already computed by earlier, smaller passes) into
//      length-len transforms using the len-th roots of unity ("twiddle
//      factors") — the divide-and-conquer step that turns the O(n^2) DFT
//      sum into O(n log n) total work.
// This is the standard reference algorithm taught in every DSP course
// (Cooley & Tukey 1965); THEORY.md "The algorithm" walks the recursion this
// iterative form implements bottom-up.
static void fft1d(std::vector<Complex64>& a, bool invert)
{
    const int n = static_cast<int>(a.size());

    // -- bit-reversal permutation ------------------------------------------
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // -- butterfly passes: len = 2, 4, 8, ..., n -----------------------------
    for (int len = 2; len <= n; len <<= 1) {
        // The angle's SIGN differs between forward/inverse (the defining
        // property of a Fourier transform pair — e^{-i.} forward, e^{+i.}
        // inverse, or vice versa; either convention is internally
        // consistent as long as forward/inverse use OPPOSITE signs and the
        // 1/n normalization is applied exactly once, which is all that
        // matters for round-tripping and for the convolution theorem this
        // project relies on — THEORY.md "Numerical considerations").
        const double ang = 2.0 * 3.14159265358979323846 / static_cast<double>(len) * (invert ? 1.0 : -1.0);
        const Complex64 wlen{ std::cos(ang), std::sin(ang) };
        for (int i = 0; i < n; i += len) {
            Complex64 w{ 1.0, 0.0 };
            for (int j = 0; j < len / 2; ++j) {
                const Complex64 u = a[i + j];
                const Complex64 v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w = w * wlen;
            }
        }
    }

    if (invert) {
        for (auto& x : a) { x.re /= n; x.im /= n; }
    }
}

// fft2d — separable 2-D DFT: transform every ROW, then every COLUMN (the
// standard row-column decomposition; correct because the 2-D DFT is
// separable — THEORY.md "The math" shows the factorization). data is a
// row-major W*H complex buffer, transformed IN PLACE.
static void fft2d(std::vector<Complex64>& data, int W, int H, bool invert)
{
    std::vector<Complex64> row(W);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) row[x] = data[static_cast<size_t>(y) * W + x];
        fft1d(row, invert);
        for (int x = 0; x < W; ++x) data[static_cast<size_t>(y) * W + x] = row[x];
    }
    std::vector<Complex64> col(H);
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) col[y] = data[static_cast<size_t>(y) * W + x];
        fft1d(col, invert);
        for (int y = 0; y < H; ++y) data[static_cast<size_t>(y) * W + x] = col[y];
    }
}

} // namespace

// ===========================================================================
// PART B — milestone 1 CPU twins: naive inverse filter, Wiener filter,
// Richardson-Lucy.
// ===========================================================================

// naive_inverse_cpu / wiener_cpu — both promote the real blurred frame and
// the real (already wraparound-padded — a shared data-layout contract, see
// main.cu's build_padded_psf()) PSF to complex (zero imaginary part), run
// the from-scratch fft2d forward on each, apply the SAME regularized-
// division formula kernels.cu's naive_inverse_kernel/wiener_kernel apply
// (the shared MATH, independently coded here in double precision — see
// this file's header for why "same formula, independently implemented" is
// exactly what a twin means), inverse-transform, and keep the real part
// (the imaginary part is expected to be ~0 by construction: a real signal
// convolved/divided by a real-valued-spectrum-derived filter stays real,
// modulo floating-point noise — THEORY.md notes the residual imaginary
// magnitude as a sanity check main.cu could report).
void naive_inverse_cpu(const float* blurred, const float* psf_padded, float* out)
{
    std::vector<Complex64> Y(kN), H(kN);
    for (int i = 0; i < kN; ++i) { Y[i].re = static_cast<double>(blurred[i]); Y[i].im = 0.0; }
    for (int i = 0; i < kN; ++i) { H[i].re = static_cast<double>(psf_padded[i]); H[i].im = 0.0; }
    fft2d(Y, kW, kH, false);
    fft2d(H, kW, kH, false);

    std::vector<Complex64> X(kN);
    for (int i = 0; i < kN; ++i) {
        const double denom = H[i].re * H[i].re + H[i].im * H[i].im + static_cast<double>(kNaiveInverseEpsilon);
        X[i].re = (Y[i].re * H[i].re + Y[i].im * H[i].im) / denom;
        X[i].im = (Y[i].im * H[i].re - Y[i].re * H[i].im) / denom;
    }
    fft2d(X, kW, kH, true);   // self-normalizing inverse (see fft1d's header)
    for (int i = 0; i < kN; ++i) out[i] = static_cast<float>(X[i].re);
}

void wiener_cpu(const float* blurred, const float* psf_padded, float K, float* out)
{
    std::vector<Complex64> Y(kN), H(kN);
    for (int i = 0; i < kN; ++i) { Y[i].re = static_cast<double>(blurred[i]); Y[i].im = 0.0; }
    for (int i = 0; i < kN; ++i) { H[i].re = static_cast<double>(psf_padded[i]); H[i].im = 0.0; }
    fft2d(Y, kW, kH, false);
    fft2d(H, kW, kH, false);

    std::vector<Complex64> X(kN);
    const double Kd = static_cast<double>(K);
    for (int i = 0; i < kN; ++i) {
        const double denom = H[i].re * H[i].re + H[i].im * H[i].im + Kd;
        X[i].re = (Y[i].re * H[i].re + Y[i].im * H[i].im) / denom;
        X[i].im = (Y[i].im * H[i].re - Y[i].re * H[i].im) / denom;
    }
    fft2d(X, kW, kH, true);
    for (int i = 0; i < kN; ++i) out[i] = static_cast<float>(X[i].re);
}

// richardson_lucy_cpu — the spatial-domain multiplicative EM update
// (THEORY.md derives this from Poisson maximum-likelihood estimation):
//     reblur    = estimate (*) psf                       (forward blur)
//     ratio     = blurred / max(reblur, eps)              (data-fidelity ratio)
//     estimate *= ratio (*) psf_flipped                   (adjoint back-projection)
// repeated kRlIterations times. (*) denotes CIRCULAR convolution — the
// SAME wraparound-index formula as kernels.cu's convolve_circular_kernel,
// written independently here as a plain nested loop (no shared function;
// see this file's header). psf_flipped is psf rotated 180 degrees (built
// once, locally): for THIS project's motion-blur line PSF — centered and
// traversed at uniform velocity, hence POINT-SYMMETRIC about its own
// center, psf[-x,-y] == psf[x,y] — the flipped kernel happens to equal the
// original. The code still builds and uses a distinct psf_flipped buffer
// via the GENERAL 180-degree-rotation formula (not a symmetry shortcut) so
// it reads correctly for the general Richardson-Lucy algorithm, which does
// need a genuine flip for an asymmetric PSF (e.g. an accelerating camera's
// comet-trail blur — README "Exercises" suggests trying one).
//
// mse_curve_out[it] (if non-null) records mean((reblur - blurred)^2) BEFORE
// each iteration's update — a measurement-only, ground-truth-free
// convergence diagnostic (RL provably increases the Poisson data
// likelihood every iteration, so this residual is expected to fall
// monotonically-ish; main.cu writes it to demo/out/rl_convergence.csv).
void richardson_lucy_cpu(const float* blurred, const float* psf, float* estimate_inout,
                         int iterations, float* mse_curve_out)
{
    std::vector<float> psf_flipped(static_cast<size_t>(kPsfSize) * kPsfSize);
    for (int ky = 0; ky < kPsfSize; ++ky)
        for (int kx = 0; kx < kPsfSize; ++kx)
            psf_flipped[static_cast<size_t>(ky) * kPsfSize + kx] =
                psf[static_cast<size_t>(kPsfSize - 1 - ky) * kPsfSize + (kPsfSize - 1 - kx)];

    // convolve_circular — a plain nested loop, the CPU twin of kernels.cu's
    // convolve_circular_kernel (same formula, independently written: no
    // shared function crosses the host/device boundary here).
    auto convolve_circular = [](const std::vector<float>& img, const std::vector<float>& psfk,
                                std::vector<float>& out) {
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                double acc = 0.0;
                for (int ky = 0; ky < kPsfSize; ++ky) {
                    const int sy = (y + ky - kPsfRadius + kH) % kH;
                    for (int kx = 0; kx < kPsfSize; ++kx) {
                        const int sx = (x + kx - kPsfRadius + kW) % kW;
                        acc += static_cast<double>(psfk[static_cast<size_t>(ky) * kPsfSize + kx])
                             * static_cast<double>(img[static_cast<size_t>(sy) * kW + sx]);
                    }
                }
                out[static_cast<size_t>(y) * kW + x] = static_cast<float>(acc);
            }
        }
    };

    std::vector<float> estimate(estimate_inout, estimate_inout + kN);
    std::vector<float> psf_vec(psf, psf + static_cast<size_t>(kPsfSize) * kPsfSize);
    std::vector<float> reblur(kN), ratio(kN), correction(kN);

    for (int it = 0; it < iterations; ++it) {
        convolve_circular(estimate, psf_vec, reblur);

        if (mse_curve_out) {
            double acc = 0.0;
            for (int i = 0; i < kN; ++i) {
                const double d = static_cast<double>(reblur[i]) - static_cast<double>(blurred[i]);
                acc += d * d;
            }
            mse_curve_out[it] = static_cast<float>(acc / static_cast<double>(kN));
        }

        for (int i = 0; i < kN; ++i) {
            const float denom = reblur[i] > kRlEpsilon ? reblur[i] : kRlEpsilon;
            ratio[i] = blurred[i] / denom;
        }
        convolve_circular(ratio, psf_flipped, correction);
        for (int i = 0; i < kN; ++i) estimate[i] *= correction[i];
    }

    for (int i = 0; i < kN; ++i) estimate_inout[i] = estimate[i];
}

// ===========================================================================
// PART C — milestone 2 CPU twins: bicubic upscale, shift-and-add,
// iterative back-projection.
// ===========================================================================

namespace {
// cubic_weight_cpu — Keys' (1981) cubic convolution kernel, a=-0.5, written
// independently in double precision (the CPU twin of kernels.cu's
// cubic_weight — same well-known closed-form formula, no shared code).
double cubic_weight_cpu(double t, double a)
{
    const double at = t < 0.0 ? -t : t;
    if (at <= 1.0) return (a + 2.0) * at * at * at - (a + 3.0) * at * at + 1.0;
    if (at < 2.0)  return a * at * at * at - 5.0 * a * at * at + 8.0 * a * at - 4.0 * a;
    return 0.0;
}
} // namespace

void bicubic_upscale_cpu(const float* lr, float* hr)
{
    const double a = -0.5;
    for (int hy = 0; hy < kH; ++hy) {
        for (int hx = 0; hx < kW; ++hx) {
            const double lx = (static_cast<double>(hx) + 0.5) / kLrScale - 0.5;
            const double ly = (static_cast<double>(hy) + 0.5) / kLrScale - 0.5;
            const int ix = static_cast<int>(std::floor(lx));
            const int iy = static_cast<int>(std::floor(ly));
            const double fx = lx - static_cast<double>(ix);
            const double fy = ly - static_cast<double>(iy);

            double acc = 0.0;
            for (int dy = -1; dy <= 2; ++dy) {
                int sy = iy + dy; sy = sy < 0 ? 0 : (sy > kLrH - 1 ? kLrH - 1 : sy);
                const double wy = cubic_weight_cpu(static_cast<double>(dy) - fy, a);
                for (int dx = -1; dx <= 2; ++dx) {
                    int sx = ix + dx; sx = sx < 0 ? 0 : (sx > kLrW - 1 ? kLrW - 1 : sx);
                    const double wx = cubic_weight_cpu(static_cast<double>(dx) - fx, a);
                    acc += wx * wy * static_cast<double>(lr[static_cast<size_t>(sy) * kLrW + sx]);
                }
            }
            hr[static_cast<size_t>(hy) * kW + hx] = static_cast<float>(acc);
        }
    }
}

// shift_and_add_cpu — the CPU twin of kernels.cu's shift_and_add_kernel:
// the SAME bilinear-splat formula (a shared data-layout contract, see
// bilinear_sample_at() in kernels.cuh), but accumulated in DOUBLE
// precision with a FIXED iteration order (frame ascending, then row, then
// column) instead of the GPU's nondeterministic-order atomicAdd — the
// identical "atomic float vs. fixed-order double" twin shape 01.11's
// BM3D-lite established (cited in kernels.cuh), which is why main.cu grants
// this method the loosest VERIFY tolerance of the two SR methods.
void shift_and_add_cpu(const float* lr_frames, const Shift* shifts, float* hr_out)
{
    std::vector<double> sum(kN, 0.0), weight(kN, 0.0);

    for (int f = 0; f < kNumFrames; ++f) {
        const Shift s = shifts[f];
        for (int lv = 0; lv < kLrH; ++lv) {
            for (int lu = 0; lu < kLrW; ++lu) {
                const float hx = (static_cast<float>(lu) + s.dx_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
                const float hy = (static_cast<float>(lv) + s.dy_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
                const BilinearSample bs = bilinear_sample_at(hx, hy, kW, kH);
                const double value = static_cast<double>(lr_frames[static_cast<size_t>(f) * kLrN + lv * kLrW + lu]);
                const double w00 = (1.0 - bs.wx) * (1.0 - bs.wy);
                const double w10 = static_cast<double>(bs.wx) * (1.0 - bs.wy);
                const double w01 = (1.0 - bs.wx) * static_cast<double>(bs.wy);
                const double w11 = static_cast<double>(bs.wx) * static_cast<double>(bs.wy);
                const int i00 = bs.y0 * kW + bs.x0, i10 = i00 + 1, i01 = i00 + kW, i11 = i01 + 1;
                sum[i00] += value * w00; weight[i00] += w00;
                sum[i10] += value * w10; weight[i10] += w10;
                sum[i01] += value * w01; weight[i01] += w01;
                sum[i11] += value * w11; weight[i11] += w11;
            }
        }
    }

    // Defensive fallback (should not trigger — 8 frames on a quarter-pixel
    // lattice give dense coverage; see kernels.cu's finalize_splat_kernel
    // header): bicubic-upscale frame 0 (the zero-shift reference frame)
    // wherever a HR cell ends up with ~0 accumulated weight.
    std::vector<float> fallback(kN);
    bicubic_upscale_cpu(lr_frames /* frame 0 starts at offset 0 */, fallback.data());

    for (int i = 0; i < kN; ++i)
        hr_out[i] = (weight[i] > 1.0e-6) ? static_cast<float>(sum[i] / weight[i]) : fallback[i];
}

// ibp_refine_cpu — the CPU twin of kernels.cu's forward_simulate_kernel +
// subtract_kernel + backproject_kernel loop. Both forward and backward
// steps are GATHER operations (kernels.cu's header explains why no atomics
// are needed for either), so this CPU twin is deterministic by
// construction and needs no fixed-order-accumulation trick the way
// shift_and_add_cpu above does — main.cu therefore grants IBP a TIGHTER
// VERIFY tolerance than shift-and-add's atomic-scatter one.
void ibp_refine_cpu(const float* lr_frames, const Shift* shifts, float* hr_estimate_inout,
                    int iterations, float* rms_curve_out)
{
    std::vector<float> estimate(hr_estimate_inout, hr_estimate_inout + kN);
    std::vector<float> predicted(kLrFramesN), residual(kLrFramesN);

    for (int it = 0; it < iterations; ++it) {
        // -- forward-simulate: gather a bilinear sample of the CURRENT HR
        // estimate at every (frame, LR pixel)'s known continuous location —
        // the identical formula to forward_simulate_kernel, independently
        // looped here.
        for (int f = 0; f < kNumFrames; ++f) {
            const Shift s = shifts[f];
            for (int lv = 0; lv < kLrH; ++lv) {
                for (int lu = 0; lu < kLrW; ++lu) {
                    const float hx = (static_cast<float>(lu) + s.dx_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
                    const float hy = (static_cast<float>(lv) + s.dy_lrpx) * kLrScale + 0.5f * (kLrScale - 1);
                    const BilinearSample bs = bilinear_sample_at(hx, hy, kW, kH);
                    const int i00 = bs.y0 * kW + bs.x0, i10 = i00 + 1, i01 = i00 + kW, i11 = i01 + 1;
                    const float v = (1.0f - bs.wx) * (1.0f - bs.wy) * estimate[i00]
                                  + bs.wx * (1.0f - bs.wy) * estimate[i10]
                                  + (1.0f - bs.wx) * bs.wy * estimate[i01]
                                  + bs.wx * bs.wy * estimate[i11];
                    predicted[static_cast<size_t>(f) * kLrN + lv * kLrW + lu] = v;
                }
            }
        }

        double acc_sq = 0.0;
        for (int i = 0; i < kLrFramesN; ++i) {
            residual[i] = lr_frames[i] - predicted[i];
            acc_sq += static_cast<double>(residual[i]) * static_cast<double>(residual[i]);
        }
        if (rms_curve_out) rms_curve_out[it] = static_cast<float>(std::sqrt(acc_sq / static_cast<double>(kLrFramesN)));

        // -- back-project: for every HR pixel, gather the inverse-mapped
        // residual sample from all kNumFrames frames and accumulate the
        // averaged, step-scaled correction — the identical formula to
        // backproject_kernel, independently looped here.
        for (int hyi = 0; hyi < kH; ++hyi) {
            for (int hxi = 0; hxi < kW; ++hxi) {
                float acc = 0.0f;
                for (int f = 0; f < kNumFrames; ++f) {
                    const Shift s = shifts[f];
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
                estimate[static_cast<size_t>(hyi) * kW + hxi] += kIbpStep * (acc / static_cast<float>(kNumFrames));
            }
        }
    }

    for (int i = 0; i < kN; ++i) hr_estimate_inout[i] = estimate[i];
}
