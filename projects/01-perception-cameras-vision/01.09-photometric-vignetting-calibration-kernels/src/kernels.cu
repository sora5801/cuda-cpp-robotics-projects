// ===========================================================================
// kernels.cu — GPU kernels + shared host solve for project 01.09
//              (Photometric/vignetting calibration kernels)
//
// Role in the project
// --------------------
// This file has THREE layers, read in this order:
//
//   1) Six small, REUSABLE __global__ kernels (SECTION 1) — each a single
//      map/reduce/scatter-reduce primitive. None of them "is" the
//      calibration pipeline by itself; main.cu assembles them (dark-stack
//      mean -> flat-stack mean -> subtract -> center-normalize -> radial-
//      bin -> [host: shared LS fit] -> correct) exactly the way 01.08's
//      main.cu assembles ITS primitives into two HDR paths.
//   2) Host launch wrappers (SECTION 2) — one per kernel, owning the grid/
//      block math and the mandatory post-launch error check.
//   3) The shared, HOST-ONLY parametric radial least-squares fit
//      (SECTION 3) — see its declaration in kernels.cuh SECTION 5 for why
//      this one piece of "the algorithm" is deliberately NOT duplicated in
//      reference_cpu.cpp.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include <cmath>                 // sqrtf/atan2f/fabs — host+device math
#include <cstdio>
#include <cstdlib>               // std::exit/EXIT_FAILURE — the LS solver's loud-failure path
#include <algorithm>             // std::swap — solve3x3's partial-pivoting row swap

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ===========================================================================
// Launch-configuration constants (repo-wide convention — see 01.01/01.08).
// kBlock1D — threads/block for every kernel below: all six operate on FLAT
// n-element (or n=kN pixel) arrays with no 2D neighbor access, so a single
// 1D grid is the natural, simplest-to-read mapping (no separate kBlock2D
// needed here, unlike 01.08's stencil kernels).
// ===========================================================================
static constexpr int kBlock1D = 256;
static inline int grid1d(int n) { return (n + kBlock1D - 1) / kBlock1D; }

// ===========================================================================
// SECTION 1 — the __global__ kernels.
// ===========================================================================

// ---------------------------------------------------------------------------
// stack_mean_kernel — per-pixel mean across a FRAME-MAJOR stack of
// numFrames images: out[p] = (1/numFrames) * sum_f stack[f*n + p].
//
// This is a MAP-OF-REDUCTIONS: n INDEPENDENT reductions running in
// parallel, one per pixel, each a short serial sum over numFrames (16 in
// this project — far too small to justify a shared-memory tree reduction
// PER PIXEL; see THEORY.md "The GPU mapping" for the crossover argument).
// The natural GPU mapping is therefore "one thread per pixel, loop over
// frames" — embarrassingly parallel across the OUTER (pixel) axis, serial
// across the INNER (frame) axis.
//
// Memory behavior (why the FRAME-MAJOR layout in kernels.cuh matters): at
// loop iteration f, thread p reads stack[f*n + p]. Adjacent threads (p,
// p+1) read ADJACENT addresses at every iteration — a fully coalesced
// 128-byte-aligned transaction per warp, every iteration. A pixel-major
// layout (stack[p*numFrames + f]) would instead scatter each warp's reads
// numFrames elements apart at every iteration — the classic strided-access
// anti-pattern this project's layout choice avoids by construction.
//
// Parameters:
//   stack       — [numFrames*n] DEVICE pointer, frame-major (kernels.cuh).
//   numFrames   — frames to average (16 for both stacks in this project;
//                 the noise_averaging gate in main.cu also calls this with
//                 numFrames=1 and 4 on a shorter PREFIX of the same array).
//   n           — pixels per frame (kN in this project).
//   out_mean    — [n] OUT: per-pixel mean, coalesced write.
// ---------------------------------------------------------------------------
__global__ void stack_mean_kernel(const float* __restrict__ stack,
                                  int numFrames, int n,
                                  float* __restrict__ out_mean)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    // Double accumulator: even 16 FP32 terms of similar magnitude accrue
    // negligible drift, but the double costs nothing here (16 adds) and
    // keeps this kernel's numerics unquestionably not-the-bug when
    // debugging (THEORY.md "Numerical considerations" states this policy).
    double acc = 0.0;
    for (int f = 0; f < numFrames; ++f) {
        acc += static_cast<double>(stack[static_cast<size_t>(f) * n + p]);
    }
    out_mean[p] = static_cast<float>(acc / static_cast<double>(numFrames));
}

// ---------------------------------------------------------------------------
// elementwise_sub_kernel — out[i] = a[i] - b[i]. Used to dark-subtract the
// flat-stack mean: flat_minus_dsnu = flat_avg - dsnu_recovered (main.cu).
// A pure MAP: every output element depends on exactly one input pair.
// ---------------------------------------------------------------------------
__global__ void elementwise_sub_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       int n, float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

// ---------------------------------------------------------------------------
// affine_kernel — out[i] = scale*in[i] + offset. Reused for the center-
// normalize step (scale = 1/center_val, offset = 0) that turns the dark-
// subtracted flat average into the nonparametric gain map — the same
// "one generic primitive, several call sites" reuse 01.08's affine_kernel
// demonstrates for its own tone-mapping arithmetic.
// ---------------------------------------------------------------------------
__global__ void affine_kernel(const float* __restrict__ in, int n,
                              float scale, float offset,
                              float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = scale * in[i] + offset;
}

// ---------------------------------------------------------------------------
// roi_mean_reduce_kernel — sums the pixels of `img` that fall inside the
// rectangular ROI [x0,x1) x [y0,y1), accumulating into ONE double via the
// classic two-phase reduction (identical pattern to 01.08's
// luminance_log_sum_kernel — see its header for the full derivation of why
// shared memory + one atomicAdd per block beats a second kernel launch):
//
//   phase 1 (in shared memory): each thread converts its own pixel to
//     "value if inside the ROI, else 0.0" (the additive identity — the
//     MASKING trick that lets a single flat 1-D kernel implement a
//     rectangular ROI sum without a 2D grid or a separate compaction
//     pass), then a binary-tree reduction collapses the block to one sum.
//   phase 2: thread 0 of each block does ONE atomicAdd into the global
//     double accumulator — contention is bounded by the BLOCK count
//     (~75 for this project's kN=19,200), not the thread count.
//
// The caller (main.cu) divides the returned sum by the EXACT ROI pixel
// count (x1-x0)*(y1-y0), known analytically — no separate "count" reduction
// needed, unlike radial_bin_kernel below (whose bin assignment is NOT known
// in advance and genuinely needs a per-bin count).
//
// Parameters: img [W*H]; x0/x1/y0/y1 the ROI (half-open, must lie inside
// [0,W)x[0,H) — main.cu's kCenterRoi* constants satisfy this by
// construction); d_sum_accum OUT, caller must cudaMemset it to 0 first.
// ---------------------------------------------------------------------------
__global__ void roi_mean_reduce_kernel(const float* __restrict__ img,
                                       int W, int H,
                                       int x0, int x1, int y0, int y1,
                                       double* __restrict__ d_sum_accum)
{
    __shared__ float partial[kBlock1D];

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int n = W * H;

    float v = 0.0f;
    if (p < n) {
        const int x = p % W, y = p / W;   // recover 2D position from the flat index
        if (x >= x0 && x < x1 && y >= y0 && y < y1) v = img[p];   // 0.0f if outside the ROI (the mask)
    }
    partial[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_sum_accum, static_cast<double>(partial[0]));
}

// ---------------------------------------------------------------------------
// radial_bin_kernel — SCATTER-REDUCE (a "histogram" pattern, genuinely
// different from roi_mean_reduce_kernel's shared-memory-then-one-atomic-
// per-block reduction above): every one of kN=19,200 threads computes ITS
// OWN pixel's distance from (cx,cy), picks ONE of numBins radial-distance
// bins, and atomicAdd's DIRECTLY into that bin's global sum/count. There is
// no shared-memory staging here because, unlike a full-image reduction to
// ONE scalar, this kernel reduces to numBins=44 DIFFERENT scalars — a
// per-block partial-histogram-then-merge scheme exists (and is a real
// optimization; see README "Exercises") but is not worth the complexity
// for 44 bins at this problem size (THEORY.md "The GPU mapping" discusses
// the trade-off honestly): on average kN/numBins ~ 436 threads contend per
// bin, spread across MANY warps and blocks, and this is a ONE-TIME
// calibration step, not a per-frame hot path — atomics contention here
// costs microseconds, not milliseconds.
//
// Parameters:
//   gain            — [W*H] the nonparametric gain map to bin.
//   cx, cy          — GEOMETRIC center (kernels.cuh's constants document
//                      why geometric, not true optical, center is used).
//   numBins         — kNumRadialBins.
//   binWidthPx      — kRadialBinWidthPx; bin index = floor(r / binWidthPx).
//   d_bin_sum       — [numBins] OUT: sum of gain values in each bin (caller
//                      cudaMemsets to 0 first).
//   d_bin_count     — [numBins] OUT: pixel count in each bin (same).
// Pixels whose bin index would fall >= numBins (none do at this project's
// geometry — see kernels.cuh's headroom comment — but a real lens could be
// cropped differently) are silently excluded: an honest, documented
// boundary decision, not a silent out-of-bounds write.
// ---------------------------------------------------------------------------
__global__ void radial_bin_kernel(const float* __restrict__ gain, int W, int H,
                                  float cx, float cy,
                                  int numBins, float binWidthPx,
                                  float* __restrict__ d_bin_sum,
                                  int*   __restrict__ d_bin_count)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = W * H;
    if (p >= n) return;

    const int x = p % W, y = p / W;
    // Pixel-CENTER sampling (x+0.5, y+0.5) — matches scripts/make_synthetic.py's
    // vignette_v() convention, so the recovered radial profile lines up with
    // the true one at the SAME r values (no half-pixel bias between the two).
    const float dx = (static_cast<float>(x) + 0.5f) - cx;
    const float dy = (static_cast<float>(y) + 0.5f) - cy;
    const float r = sqrtf(dx * dx + dy * dy);

    const int bin = static_cast<int>(r / binWidthPx);
    if (bin >= 0 && bin < numBins) {
        atomicAdd(&d_bin_sum[bin], gain[p]);
        atomicAdd(&d_bin_count[bin], 1);
    }
}

// ---------------------------------------------------------------------------
// correction_kernel — THE reason this project exists: out[i] = (I[i] -
// dsnu[i]) / max(gain[i], gainFloor). A pure per-pixel MAP.
//
// The gainFloor clamp (THEORY.md "Numerical considerations" derives the
// general hazard): dividing by a SMALL gain amplifies whatever noise/error
// remains in the numerator — a pixel with true gain 0.01 would amplify a
// +-1 code-value read-noise residual into a +-100 code-value correction
// error. This project's synthetic gain never drops below ~0.62 (see
// main.cu's [info] line for the actual measured minimum), so the floor
// never engages here — it is included because every REAL flat-field
// correction needs one (a dead/underperforming pixel can have a near-zero
// recovered gain), and shipping a kernel that silently divides by a
// near-zero float in Release (no crash, just a wildly wrong, unflagged
// value) would be teaching the wrong lesson.
//
// Parameters: I [n] the raw observed image; dsnu [n] recovered additive
// offset; gain [n] recovered multiplicative field; gainFloor the numerical
// guard above; out [n] OUT, in the SAME units as (I - dsnu) — main.cu
// clamps to [0,255] only when writing a PGM artifact, never here.
// ---------------------------------------------------------------------------
__global__ void correction_kernel(const float* __restrict__ I,
                                  const float* __restrict__ dsnu,
                                  const float* __restrict__ gain,
                                  int n, float gainFloor,
                                  float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gain[i] > gainFloor ? gain[i] : gainFloor;
    out[i] = (I[i] - dsnu[i]) / g;
}

// ===========================================================================
// SECTION 2 — host launch wrappers. Each owns its grid/block math and the
// mandatory post-launch error check (CLAUDE.md §6.1 rule 7).
// ===========================================================================
void launch_stack_mean(const float* d_stack, int numFrames, int n, float* d_out_mean)
{
    stack_mean_kernel<<<grid1d(n), kBlock1D>>>(d_stack, numFrames, n, d_out_mean);
    CUDA_CHECK_LAST_ERROR("stack_mean_kernel launch");
}

void launch_elementwise_sub(const float* d_a, const float* d_b, int n, float* d_out)
{
    elementwise_sub_kernel<<<grid1d(n), kBlock1D>>>(d_a, d_b, n, d_out);
    CUDA_CHECK_LAST_ERROR("elementwise_sub_kernel launch");
}

void launch_affine(const float* d_in, int n, float scale, float offset, float* d_out)
{
    affine_kernel<<<grid1d(n), kBlock1D>>>(d_in, n, scale, offset, d_out);
    CUDA_CHECK_LAST_ERROR("affine_kernel launch");
}

void launch_roi_mean_reduce(const float* d_img, int W, int H,
                            int x0, int x1, int y0, int y1,
                            double* d_sum_accum)
{
    const int n = W * H;
    roi_mean_reduce_kernel<<<grid1d(n), kBlock1D>>>(d_img, W, H, x0, x1, y0, y1, d_sum_accum);
    CUDA_CHECK_LAST_ERROR("roi_mean_reduce_kernel launch");
}

void launch_radial_bin(const float* d_gain, int W, int H, float cx, float cy,
                       int numBins, float binWidthPx,
                       float* d_bin_sum, int* d_bin_count)
{
    const int n = W * H;
    radial_bin_kernel<<<grid1d(n), kBlock1D>>>(d_gain, W, H, cx, cy, numBins, binWidthPx,
                                               d_bin_sum, d_bin_count);
    CUDA_CHECK_LAST_ERROR("radial_bin_kernel launch");
}

void launch_correction(const float* d_I, const float* d_dsnu, const float* d_gain,
                       int n, float gainFloor, float* d_out)
{
    correction_kernel<<<grid1d(n), kBlock1D>>>(d_I, d_dsnu, d_gain, n, gainFloor, d_out);
    CUDA_CHECK_LAST_ERROR("correction_kernel launch");
}

// ===========================================================================
// SECTION 3 — the shared, HOST-ONLY parametric radial least-squares fit
// (declared in kernels.cuh SECTION 5 — read that comment first for the
// twin-independence-ruling justification and the exact model).
//
// Implementation: build the 3x3 normal equations A^T A x = A^T b for the
// basis [r_n^2, r_n^4, r_n^6] (r_n = r/rNorm) and target (bin_mean - 1),
// then solve the tiny symmetric 3x3 system by Gaussian elimination with
// partial pivoting — the general, teachable technique that scales to any
// N (33.01-batched-small-matrix-linalg is where this repo teaches the
// GPU-batched version, at a problem size where batching many such solves
// actually pays for itself; a single 3x3 solve here has no such
// opportunity — THEORY.md "The GPU mapping" states this honestly).
// ===========================================================================

// solve3x3 — Gaussian elimination with partial pivoting for a dense 3x3
// linear system A*x = b (A row-major, A[r*3+c]). Returns false (leaves x
// untouched) if A is numerically singular (|pivot| < 1e-12) — this project
// never hits that branch (the basis is well-conditioned by the rNorm
// normalization) but a linear solver that can silently divide by ~0 is
// exactly the kind of "no black boxes, no silent wrongness" failure this
// repo's commenting standard exists to prevent (CLAUDE.md §1).
static bool solve3x3(double A[3][3], double b[3], double x[3])
{
    // Partial pivoting: at each elimination step, swap in the row with the
    // LARGEST remaining leading coefficient — the standard numerical-
    // stability fix (without it, a small pivot can amplify rounding error
    // catastrophically; THEORY.md "Numerical considerations" cross-refers
    // this to the same "ill-conditioned solve" family as near-singular
    // robot Jacobians).
    for (int col = 0; col < 3; ++col) {
        int pivot_row = col;
        double best = std::fabs(A[col][col]);
        for (int r = col + 1; r < 3; ++r) {
            if (std::fabs(A[r][col]) > best) { best = std::fabs(A[r][col]); pivot_row = r; }
        }
        if (best < 1e-12) return false;   // numerically singular — caller must handle honestly
        if (pivot_row != col) {
            for (int c = 0; c < 3; ++c) std::swap(A[col][c], A[pivot_row][c]);
            std::swap(b[col], b[pivot_row]);
        }
        for (int r = col + 1; r < 3; ++r) {
            const double factor = A[r][col] / A[col][col];
            for (int c = col; c < 3; ++c) A[r][c] -= factor * A[col][c];
            b[r] -= factor * b[col];
        }
    }
    // Back-substitution.
    for (int r = 2; r >= 0; --r) {
        double acc = b[r];
        for (int c = r + 1; c < 3; ++c) acc -= A[r][c] * x[c];
        x[r] = acc / A[r][r];
    }
    return true;
}

void fit_vignette_radial_ls(const float* bin_r, const float* bin_mean, int numPoints,
                            float rNorm, float& out_a2, float& out_a4, float& out_a6)
{
    // Build A^T A (3x3, symmetric) and A^T b directly (never materialize
    // the numPoints x 3 design matrix A itself — with numPoints <=
    // kNumRadialBins=44, this is a trivial O(numPoints) accumulation loop,
    // the standard way to form small normal equations without extra
    // memory; THEORY.md "The math" derives WHY minimizing sum (basis.x -
    // target)^2 leads to exactly these equations).
    double AtA[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    double Atb[3] = {0, 0, 0};

    for (int i = 0; i < numPoints; ++i) {
        const double rn = static_cast<double>(bin_r[i]) / static_cast<double>(rNorm);
        const double rn2 = rn * rn, rn4 = rn2 * rn2, rn6 = rn4 * rn2;
        const double basis[3] = { rn2, rn4, rn6 };
        // Target: bin_mean - 1 (the intercept is FIXED at 1 — V(0)=1 by
        // construction of the physical model, see kernels.cuh SECTION 5).
        const double target = static_cast<double>(bin_mean[i]) - 1.0;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) AtA[r][c] += basis[r] * basis[c];
            Atb[r] += basis[r] * target;
        }
    }

    double x[3] = { 0.0, 0.0, 0.0 };
    const bool ok = solve3x3(AtA, Atb, x);
    if (!ok) {
        // Numerically singular normal equations would mean the radial bins
        // carried no usable spread of r values — should never happen with
        // this project's geometry (44 populated bins spanning ~0..100 px).
        // Fail LOUD rather than silently return zero coefficients (which
        // would make every downstream gate wrongly "pass" by comparing
        // against a flat V=1 curve).
        std::fprintf(stderr, "fit_vignette_radial_ls: normal equations are singular "
                             "(numPoints=%d) — check the radial-bin population\n", numPoints);
        std::exit(EXIT_FAILURE);
    }
    out_a2 = static_cast<float>(x[0]);
    out_a4 = static_cast<float>(x[1]);
    out_a6 = static_cast<float>(x[2]);
}
