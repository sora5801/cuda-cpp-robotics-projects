// ===========================================================================
// kernels.cu — GPU implementation for project 34.03
//              Ergodic control: spectral multiscale coverage (SMC)
//              (teaching core: single 2-D first-order agent, K=32x32 modes)
//
// The big idea
// ------------
// Two GPU jobs, run at very different rates:
//
//   (1) ONCE, at startup: turn the target density phi(x) into its 1024
//       Fourier (cosine) coefficients phi_k. This is a DCT ("discrete
//       cosine transform") computed via a single 2-D cuFFT call plus two
//       small bookkeeping kernels — the catalog bullet's named "FFT-based,
//       very GPU-friendly" hook.
//   (2) EVERY control step (6000 times over the demo): update all 1024
//       modes' running time-average c_k and their contribution to the
//       ergodic-descent direction. One thread PER MODE — genuinely
//       independent, embarrassingly parallel work, but small (K=1024
//       threads). This project is honest about that scale in the comments
//       below and in THEORY.md §GPU mapping: the teaching point is the
//       MAPPING (mode-parallel, not point-parallel), not a big speed-up at
//       this K — the pattern is exactly what a multi-agent or finer-K
//       extension would need to actually saturate a GPU.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cufft.h>                  // cuFFT: cufftPlan2d / cufftExecZ2Z (see launch_build_phi_k)
#include <cmath>
#include <cstdio>
#include <cstdlib>

// cuFFT returns its own error enum (cufftResult), not cudaError_t — same
// "fail loud, at the call site" macro shape as 03.01's CUFFT_CHECK.
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
// Shared device helpers — the basis L2-norm h_k. A DELIBERATE, DOCUMENTED
// DUPLICATE of the plain-C++ version in reference_cpu.cpp (CLAUDE.md §5's
// "the correctness oracle never depends on nvcc" rule means this cannot be
// a single shared __device__ __host__ function living in the header — see
// 08.01's cartpole_deriv / reference_cpu.cpp's twin for the exact same
// pattern applied to a different kernel).
//
// h_k = sqrt( integral_[0,1]^2 f_k_unnormalized(x)^2 dx ), the L2 norm of
// cos(k1*pi*x1)*cos(k2*pi*x2) on the unit square:
//   integral_0^1 cos(0*pi*x)^2 dx = 1        (k_i = 0: the constant 1)
//   integral_0^1 cos(k*pi*x)^2 dx = 1/2       (k_i > 0, any integer k)
// so h_k in {1, 1/sqrt(2), 1/2} depending on how many of (k1,k2) are zero.
// Dividing f_k's raw product by h_k makes {f_k} an ORTHONORMAL basis on
// [0,1]^2 — required for the Fourier-coefficient bookkeeping in THEORY.md
// §the math (c_k, phi_k, and Parseval's identity all assume orthonormality).
// ---------------------------------------------------------------------------
__device__ __forceinline__ double basis_norm_h_dev(int k1, int k2)
{
    const bool z1 = (k1 == 0), z2 = (k2 == 0);
    if (z1 && z2) return 1.0;
    if (z1 || z2) return 0.70710678118654752440;   // 1/sqrt(2)
    return 0.5;
}

// ===========================================================================
// 1) Target Fourier coefficients phi_k — computed ONCE via DCT-via-FFT.
// ===========================================================================
//
// THE DCT-I-VIA-FFT IDENTITY (derived in full in THEORY.md §the math; this
// comment gives the recipe, THEORY.md gives the proof):
//
// We want, for every mode (k1,k2), the double integral
//     I[k1,k2] = integral_[0,1]^2 phi(x1,x2) * cos(k1*pi*x1) * cos(k2*pi*x2) dx1 dx2
// approximated by the 2-D TRAPEZOIDAL RULE on an (N x N) grid, N=kPhiGridN,
// spacing h=1/(N-1), grid points x_n = n*h (n=0..N-1, endpoints INCLUDED).
//
// Step 1 — MIRROR: build an (M x M) array, M = kDctM = 2*(N-1), by
// reflecting the grid through BOTH endpoints of BOTH axes:
//     e[p,q] = phi_grid[ mirror(p), mirror(q) ],   mirror(i) = i if i<N,
//                                                              M-i otherwise
// This is exactly the classic "even-symmetric periodic extension" that
// turns a length-N Neumann-boundary (cosine) problem into a length-M
// periodic (Fourier) one — see build_even_extension_kernel below.
//
// Step 2 — TRANSFORM: cufftPlan2d + cufftExecZ2Z (forward) on e. Because e
// is even-symmetric about BOTH mirror axes, its 2-D DFT is PURELY REAL
// (to floating-point rounding) — no phase-correction multiply is needed,
// unlike some DCT-via-FFT recipes (the "pack two half-mirrors" trick used
// for DCT-II). That simplicity is exactly WHY this project mirrors through
// the endpoints (a DCT-I grid, endpoints included) rather than using
// cell-centered samples (which would need DCT-II's extra phase twiddle) —
// a documented convenience for a teaching implementation.
//
// Step 3 — EXTRACT + NORMALIZE: for k1,k2 < kK (32 << M/2 = 128, so no
// aliasing concern), read Re(E[k1,k2]) from the FFT output and rescale:
//     I[k1,k2]   = Re(E[k1,k2]) * (h^2 / 4)          (extract_phi_k_kernel)
//     phi_k[k1,k2] = I[k1,k2] / h_k                  (orthonormalize)
// The h^2/4 factor comes from the trapezoidal-rule <-> DCT-I algebra:
// 1-D trapezoidal integral = (h/2) * DCT-I coefficient (THEORY.md derives
// this identity), and the 2-D case is the tensor product of two 1-D ones.
//
// WHY cuFFT AND NOT A HAND-ROLLED DCT (CLAUDE.md §6.1 rule 6): a correct,
// numerically robust FFT (even a small 256-point 2-D one) is a
// multi-week project in its own right (see the 33.x-style foundational
// projects for what that actually entails); this project's subject is
// ERGODIC CONTROL, not FFT internals. cuFFT is a CUDA Toolkit library
// (CLAUDE.md §5's default-allowed set), and reference_cpu.cpp proves the
// result independently with NO FFT at all (an O(N^2*K) direct sum with
// precomputed cosine tables) — the transform is never a black box even
// though we do not reimplement it (the project's TRANSFORM-CORRECTNESS
// gate IS that independent check).
// ===========================================================================

// build_even_extension_kernel — write ONE cell of the (M x M) mirrored,
// zero-imaginary-part array from the (N x N) real target-density grid.
//
// Thread-to-data mapping: one thread per OUTPUT cell (p,q) in the M x M
// extended array (M=256 -> 65,536 threads) — a pure index-remapping MAP;
// every output cell reads exactly one input cell, no two threads race on
// the same memory, no shared memory needed.
__device__ __forceinline__ int mirror_index(int i, int n)
{
    // n = kPhiGridN, m = kDctM = 2*(n-1). For i < n: identity (the "real"
    // half of the period). For i >= n: reflect through the far endpoint —
    // i in [n, m) maps to source index m-i in [1, n-2] (see the file header
    // derivation; this is the "no phase twiddle needed" DCT-I mirror).
    return (i < n) ? i : (kDctM - i);
}

__global__ void build_even_extension_kernel(const double* __restrict__ phi_grid,
                                            cufftDoubleComplex* __restrict__ ext)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kDctM * kDctM;
    if (idx >= total) return;

    const int p = idx / kDctM;         // extended-array row    (mirrored axis 1)
    const int q = idx % kDctM;         // extended-array column (mirrored axis 2)
    const int sn = mirror_index(p, kPhiGridN);
    const int sm = mirror_index(q, kPhiGridN);

    cufftDoubleComplex v;
    v.x = phi_grid[sn * kPhiGridN + sm];   // real part = the mirrored density sample
    v.y = 0.0;                             // imaginary part: zero (phi is a real density)
    ext[idx] = v;
}

// extract_phi_k_kernel — read the FFT output's real part at (k1,k2) for
// every mode, rescale by the trapezoidal/DCT-I algebra factor (h^2/4), and
// orthonormalize by h_k. One thread per MODE (kNumModes = 1024) — a tiny
// map, dwarfed in cost by the FFT and the mirror kernel above; it exists as
// its own kernel (rather than folded into main.cu's host code) so the
// ENTIRE DCT pipeline — mirror, transform, extract — stays on-device
// between one H2D copy (the raw grid) and one D2H copy (phi_k), the same
// "no needless round trips" discipline 03.01's pipeline follows.
__global__ void extract_phi_k_kernel(const cufftDoubleComplex* __restrict__ fft_out,
                                     double* __restrict__ phi_k, double h_spacing)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kNumModes) return;

    const int k1 = idx / kK;
    const int k2 = idx % kK;
    const int e_idx = k1 * kDctM + k2;     // k1,k2 < kK = 32 << kDctM/2 = 128: no wraparound risk

    const double scale = (h_spacing * h_spacing) * 0.25;   // h^2/4 — see file header derivation
    const double I_k = fft_out[e_idx].x * scale;           // .x = real part (cufftDoubleComplex is {re,im})
    phi_k[idx] = I_k / basis_norm_h_dev(k1, k2);
}

void launch_build_phi_k(const double* d_phi_grid, double* d_phi_k)
{
    // The mirrored array lives only for the duration of this call — a
    // fresh allocation each time is fine because this function runs ONCE
    // per demo (setup, not the hot loop), unlike the persistent buffers
    // main.cu allocates outside the 6000-step control loop.
    cufftDoubleComplex* d_ext = nullptr;
    const size_t ext_count = static_cast<size_t>(kDctM) * kDctM;
    CUDA_CHECK(cudaMalloc(&d_ext, ext_count * sizeof(cufftDoubleComplex)));

    // Step 1: mirror. Grid-stride not needed at this size — one thread per
    // cell, block=256 (repo default), grid = ceil(65536/256) = 256 blocks.
    {
        const int block = 256;
        const int grid = (static_cast<int>(ext_count) + block - 1) / block;
        build_even_extension_kernel<<<grid, block>>>(d_phi_grid, d_ext);
        CUDA_CHECK_LAST_ERROR("build_even_extension_kernel launch");
    }

    // Step 2: transform. cufftPlan2d is cuFFT's SIMPLEST planning call — a
    // single, unbatched 2-D transform (contrast with 03.01's cufftPlanMany
    // advanced-layout batching: this project's cuFFT usage is deliberately
    // the plainest possible form, because there is exactly ONE transform to
    // run, ONCE, at startup). CUFFT_Z2Z: double-precision complex-to-complex
    // (matching this project's double-precision-throughout policy —
    // kernels.cuh's file header explains why). In-place (input == output
    // pointer) is safe and the standard cuFFT idiom when the caller does
    // not need the pre-transform data afterward (it does not, here).
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, kDctM, kDctM, CUFFT_Z2Z));
    CUFFT_CHECK(cufftExecZ2Z(plan, d_ext, d_ext, CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));

    // Step 3: extract + normalize. block=256, grid=ceil(1024/256)=4.
    {
        const double h_spacing = 1.0 / static_cast<double>(kPhiGridN - 1);
        const int block = 256;
        const int grid = (kNumModes + block - 1) / block;
        extract_phi_k_kernel<<<grid, block>>>(d_ext, d_phi_k, h_spacing);
        CUDA_CHECK_LAST_ERROR("extract_phi_k_kernel launch");
    }

    CUDA_CHECK(cudaFree(d_ext));
}

// ===========================================================================
// 2) The per-control-step SMC update — one thread per MODE.
// ===========================================================================
//
// smc_step_kernel implements, for EVERY mode k=(k1,k2) independently:
//   f_k(x)        = cos(k1*pi*x1)*cos(k2*pi*x2) / h_k          (this step's basis value)
//   S_k          += f_k(x)                                     (running sum, persists in d_S)
//   c_k           = S_k / n                                    (running time-AVERAGE)
//   Lambda_k      = (1 + k1^2 + k2^2)^(-1.5)                    (Sobolev weight)
//   grad f_k(x)   = ( -k1*pi*sin(k1*pi*x1)*cos(k2*pi*x2)/h_k ,
//                     -cos(k1*pi*x1)*k2*pi*sin(k2*pi*x2)/h_k )  (closed-form; no numerical diff needed)
//   B_k           = Lambda_k * (c_k - phi_k) * grad f_k(x)      (this mode's contribution)
// main.cu downloads d_Bx/d_By (a few KB) and reduces B = sum_k B_k on the
// HOST — the SAME deliberate choice 08.01 makes for its softmin blend:
// O(kNumModes) = O(1024) scalar adds is microseconds of plain C++, and
// keeping it there puts the WHOLE control law on one screen next to the
// kernel call (THEORY.md §GPU mapping names the on-GPU reduction as the
// natural next exercise, exactly as 08.01 Exercise 3 does for its blend).
//
// THREAD-TO-DATA MAPPING & HONESTY ABOUT SCALE: thread idx =
// blockIdx.x*blockDim.x+threadIdx.x owns mode (idx/kK, idx%kK). Every
// mode's update reads only x (broadcast, see below) and its OWN S_k/phi_k
// entries — genuinely independent, zero shared memory, zero atomics. At
// K=1024 this is SMALL parallelism (a handful of warps, not the tens of
// thousands of threads a perception kernel launches) — an honest limit of
// a SINGLE-agent, K=32-per-axis teaching instance. The GPU story scales
// exactly the way multi-agent SMC and finer-K variants need it to: N
// agents x K^2 modes is an N*K^2-thread launch with the identical kernel
// shape (THEORY.md §Where this sits in the real world names this as the
// first thing the full research version would exploit).
//
// MEMORY: x1, x2 are passed BY VALUE as kernel PARAMETERS (not a device
// pointer main.cu uploads, unlike 08.01's 4-float x0). Kernel parameters
// live in fast constant/parameter memory, automatically broadcast to every
// thread at zero extra setup cost — the simpler, and here sufficient,
// choice for a 2-scalar state (08.01's 4-float array needed a pointer only
// because passing an ARRAY by value through CUDA's C-linkage launch syntax
// is not idiomatic; two independent scalars have no such restriction).
// d_S/d_phi_k/d_c/d_Bx/d_By are read/written with ONE coalesced access per
// thread (idx is the natural array offset for all five) — textbook
// coalescing, the same reasoning as every map kernel in this repo.
// ===========================================================================
__global__ void smc_step_kernel(double x1, double x2,
                                const double* __restrict__ phi_k,
                                double* __restrict__ S,        // IN/OUT, persists across calls
                                int n,
                                double* __restrict__ c_out,
                                double* __restrict__ Bx_out,
                                double* __restrict__ By_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kNumModes) return;

    const int k1 = idx / kK;
    const int k2 = idx % kK;
    const double h = basis_norm_h_dev(k1, k2);

    const double cx1 = cos(static_cast<double>(k1) * kPi * x1);
    const double cx2 = cos(static_cast<double>(k2) * kPi * x2);
    const double f = (cx1 * cx2) / h;

    // Running sum -> running time-average. n is 1-based (this step counts).
    const double s_new = S[idx] + f;
    S[idx] = s_new;
    const double c = s_new / static_cast<double>(n);

    const double diff = c - phi_k[idx];
    const double kk = static_cast<double>(k1 * k1 + k2 * k2);
    const double lambda = 1.0 / ((1.0 + kk) * sqrt(1.0 + kk));   // (1+||k||^2)^(-1.5), avoids a general pow()

    const double sx1 = sin(static_cast<double>(k1) * kPi * x1);
    const double sx2 = sin(static_cast<double>(k2) * kPi * x2);
    const double dfdx1 = (-static_cast<double>(k1) * kPi * sx1 * cx2) / h;
    const double dfdx2 = (-cx1 * static_cast<double>(k2) * kPi * sx2) / h;

    c_out[idx] = c;
    Bx_out[idx] = lambda * diff * dfdx1;
    By_out[idx] = lambda * diff * dfdx2;
}

void launch_smc_step(double x1, double x2, const double* d_phi_k,
                     double* d_S, int n,
                     double* d_c, double* d_Bx, double* d_By)
{
    const int block = 256;                                  // repo default (warp multiple)
    const int grid = (kNumModes + block - 1) / block;        // = 4 blocks at K=1024 — see the honesty note above
    smc_step_kernel<<<grid, block>>>(x1, x2, d_phi_k, d_S, n, d_c, d_Bx, d_By);
    CUDA_CHECK_LAST_ERROR("smc_step_kernel launch");
}
