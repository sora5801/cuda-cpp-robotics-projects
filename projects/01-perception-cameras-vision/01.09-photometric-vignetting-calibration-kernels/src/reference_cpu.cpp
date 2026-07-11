// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.09
//                     (Photometric/vignetting calibration kernels)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5):
//
//   1) It is the CORRECTNESS ORACLE. GPU code fails in ways CPU code cannot:
//      wrong thread indexing, missed tail elements, race conditions, stale
//      device memory, bad transfers. A dead-simple sequential version that a
//      reader can verify BY EYE gives us ground truth; main.cu runs both and
//      asserts element-wise agreement within a documented tolerance.
//
//   2) It is the TEACHING BASELINE. Reading this file first, then
//      kernels.cu, shows exactly what parallelization changed — the same
//      per-pixel math, written once as an explicit loop and once as one
//      thread's body.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong, and
// then the oracle lies. (This file is compiled by the HOST compiler, cl.exe;
// the __CUDACC__ fence in kernels.cuh hides device declarations from it.)
//
// Independence ruling (Phase-1 standards retrospective — load-bearing,
// quoted in full from docs/PROJECT_TEMPLATE/src/reference_cpu.cpp):
// --------------------------------------------------------------------
//   * Data-layout contracts (structs, constants, indexing formulas) MUST be
//     single-sourced in kernels.cuh and shared.
//   * The ALGORITHMIC CORE should be written twice — independently, in the
//     simplest possible C++ here.
//   * A shared __host__ __device__ helper is permitted when duplicating it
//     would be pure token-for-token transcription — but then the twin
//     comparison is BLIND to bugs inside that helper, so the project MUST
//     also carry at least one verification gate that does not route through
//     the shared code.
//
// This project's application of the ruling: every kernel below (stack_mean,
// elementwise_sub, affine, roi_mean_reduce, radial_bin, correction) is
// written TWICE — independently, once here, once in kernels.cu — and
// GPU-vs-CPU VERIFY-checked in main.cu. The one piece of "the algorithm"
// that IS shared is the parametric least-squares fit (kernels.cu SECTION 3,
// declared in kernels.cuh SECTION 5) — a single 3x3 host solve with no
// meaningful GPU/CPU split to compare (see its own header for the full
// justification, mirroring 01.08's crf_solve_debevec). Because that fit is
// shared, the twin comparison is BLIND to bugs inside it; main.cu's
// radial_fit gate is the INDEPENDENT check that does not route through it —
// comparing the fit's OUTPUT against the known analytic cos^4 curve, never
// against a second implementation of the solver.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <cmath>       // std::sqrt

// ---------------------------------------------------------------------------
// stack_mean_cpu — sequential twin of stack_mean_kernel: out[p] = mean over
// f=0..numFrames-1 of stack[f*n + p]. Same frame-major layout, same double
// accumulator (so a legitimate difference in this project's results can
// never come from "one path uses more precision than the other" — see
// THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
void stack_mean_cpu(const float* stack, int numFrames, int n, float* out_mean)
{
    for (int p = 0; p < n; ++p) {
        double acc = 0.0;
        for (int f = 0; f < numFrames; ++f) {
            acc += static_cast<double>(stack[static_cast<size_t>(f) * n + p]);
        }
        out_mean[p] = static_cast<float>(acc / static_cast<double>(numFrames));
    }
}

// ---------------------------------------------------------------------------
// elementwise_sub_cpu — out[i] = a[i] - b[i]. The simplest possible loop;
// the twin of elementwise_sub_kernel.
// ---------------------------------------------------------------------------
void elementwise_sub_cpu(const float* a, const float* b, int n, float* out)
{
    for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

// ---------------------------------------------------------------------------
// affine_cpu — out[i] = scale*in[i] + offset. The twin of affine_kernel.
// ---------------------------------------------------------------------------
void affine_cpu(const float* in, int n, float scale, float offset, float* out)
{
    for (int i = 0; i < n; ++i) out[i] = scale * in[i] + offset;
}

// ---------------------------------------------------------------------------
// roi_sum_cpu — sum of img's pixels inside [x0,x1) x [y0,y1), returned as a
// double. The twin of roi_mean_reduce_kernel's REDUCTION (main.cu divides by
// the exact ROI pixel count itself, on both the GPU and CPU sides, so this
// function's contract deliberately mirrors what the kernel produces BEFORE
// that division: a raw sum, not a mean).
// ---------------------------------------------------------------------------
double roi_sum_cpu(const float* img, int W, int H, int x0, int x1, int y0, int y1)
{
    double sum = 0.0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            sum += static_cast<double>(img[static_cast<size_t>(y) * W + x]);
        }
    }
    (void)H;   // H is not needed for the loop bounds (y1 <= H is the caller's contract) but kept
               // in the signature for symmetry with every other (W,H,...) function in this file.
    return sum;
}

// ---------------------------------------------------------------------------
// radial_bin_cpu — twin of radial_bin_kernel's histogram scatter-reduce.
// Sequential, so no atomics are needed here at all — the CPU loop simply
// accumulates into bin_sum[bin]/bin_count[bin] directly, one pixel at a
// time. This is EXACTLY the property that makes the GPU version's atomics
// necessary in the first place: many threads racing to update the same few
// bins concurrently need hardware-arbitrated read-modify-write, while a
// single CPU thread visiting pixels one at a time never has that race.
// Caller must zero bin_sum/bin_count first (same contract as the kernel's
// cudaMemset requirement).
// ---------------------------------------------------------------------------
void radial_bin_cpu(const float* gain, int W, int H, float cx, float cy,
                    int numBins, float binWidthPx,
                    float* bin_sum, int* bin_count)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float dx = (static_cast<float>(x) + 0.5f) - cx;
            const float dy = (static_cast<float>(y) + 0.5f) - cy;
            const float r = std::sqrt(dx * dx + dy * dy);
            const int bin = static_cast<int>(r / binWidthPx);
            if (bin >= 0 && bin < numBins) {
                const int p = y * W + x;
                bin_sum[bin] += gain[p];
                bin_count[bin] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// correction_cpu — out[i] = (I[i] - dsnu[i]) / max(gain[i], gainFloor). The
// twin of correction_kernel — same clamp, same order of operations.
// ---------------------------------------------------------------------------
void correction_cpu(const float* I, const float* dsnu, const float* gain,
                    int n, float gainFloor, float* out)
{
    for (int i = 0; i < n; ++i) {
        const float g = gain[i] > gainFloor ? gain[i] : gainFloor;
        out[i] = (I[i] - dsnu[i]) / g;
    }
}
