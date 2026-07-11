// ===========================================================================
// kernels.cu — GPU kernels for project 01.05
//              SIFT on GPU: Gaussian scale space, DoG extrema, warp-level
//              orientation histograms, warp-level 128-D descriptors,
//              brute-force L2 matching.
//
// Big idea (the whole project in one paragraph)
// -----------------------------------------------
// SIFT is a PIPELINE of very different GPU workloads chained together —
// exactly why it makes a good "harder" teaching project after 01.04's
// single-scale FAST/ORB pipeline. Building the scale space is a MAP/STENCIL
// over millions of pixels (like 01.04's detectors). Finding extrema is a
// STENCIL + ATOMIC COMPACTION (same pattern, one dimension bigger: 3x3x3
// instead of 3x3). Refinement is a per-candidate, VARIABLE-ITERATION-COUNT
// loop — a natural one-thread-per-candidate kernel. But orientation and
// description are neither: each keypoint needs to visit HUNDREDS to
// THOUSANDS of surrounding pixels and REDUCE them into a small histogram —
// too much serial work for one thread, too little independent work to
// justify one thread per SAMPLE (with a scatter-add race on every bin).
// The right-sized unit is a WARP: 32 threads split the sampling work,
// each lane builds a PRIVATE partial histogram (no shared memory, no
// atomics, no contention at all while sampling), and a five-step
// __shfl_down_sync TREE REDUCTION folds the 32 partials into one final
// histogram per bin — the "warp-level reductions" this catalog entry is
// built to teach, used TWICE (orientation's 36 bins, description's 128).
//
// All shared layouts, constants, and the Gaussian weight-table builder are
// single-sourced in kernels.cuh — read that file's header first; it
// explains the shared-weights-vs-independent-convolution-loop split this
// file and reference_cpu.cpp both depend on.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (an
// INDEPENDENT re-implementation of every algorithmic core below — diff
// them side by side to see what "the same algorithm, twice" looks like).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>
#include <algorithm>   // std::min/max (host-side launch wrappers only)

// ---------------------------------------------------------------------------
// Launch geometry shared by every "one thread per pixel" kernel in this
// file: a 16x16 2-D block (256 threads, a warp multiple), matching 01.04's
// established occupancy default. Grids are sized per-call because octave 0
// and octave 1 have different W,H (256x256 vs 128x128) — unlike 01.04's
// fixed kW/kH, this project's geometry varies by octave, so grid2d() takes
// W,H as arguments instead of being a zero-argument helper.
// ---------------------------------------------------------------------------
static constexpr int kBlock2D = 16;

static inline dim3 grid2d(int W, int H)
{
    return dim3((W + kBlock2D - 1) / kBlock2D, (H + kBlock2D - 1) / kBlock2D);
}

// Launch geometry for "one thread per candidate/query" kernels (refine,
// match): a 1-D grid, block 128 (a warp multiple), no stride loop needed —
// candidate/query counts here are always small (<= kMaxDogCandidates).
static constexpr int kBlock1D = 128;
static inline int grid1d(int n) { return (n + kBlock1D - 1) / kBlock1D; }

// ===========================================================================
// 1) GAUSSIAN BLUR — separable 2-pass convolution. gaussian_blur_h_kernel
//    convolves along X (each thread reads `2*radius+1` HORIZONTAL
//    neighbors); gaussian_blur_v_kernel convolves along Y. Running both in
//    sequence (H then V, through a temp buffer) computes the full 2-D
//    Gaussian blur at O(2*radius) work per pixel instead of the O(radius^2)
//    a non-separable 2-D stencil would cost — separability is a property of
//    the Gaussian specifically (THEORY.md "The problem" proves it), not a
//    generic image-filtering trick.
//
// Thread-to-data mapping: thread (bx*16+tx, by*16+ty) owns OUTPUT pixel
// (x,y) = (blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y).
// Memory: reads `2*radius+1` texels from GLOBAL memory per thread, all from
// `src` (never written mid-pass, so no race even though adjacent threads'
// windows overlap heavily — see THEORY.md "GPU mapping" for the
// shared-memory-tiled version this teaching kernel deliberately skips: at
// radius up to kMaxGaussRadius=24, a tile large enough to help every thread
// in a 16x16 block would need (16+48)x16 floats in shared memory, a real
// optimization worth doing but not essential for a project whose complexity
// budget this repo has chosen to spend on the WARP kernels instead — named
// honestly, not silently skipped).
// Border handling: CLAMP-TO-EDGE (see kernels.cuh's header for why, vs.
// zero-padding).
// ===========================================================================
__global__ void gaussian_blur_h_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       int W, int H, const float* __restrict__ weights, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;   // ragged-tail guard (grid rounds up)

    float acc = 0.0f;   // accumulate in the SAME left-to-right tap order every call site uses (see kernels.cuh header: this order is what makes the GPU-vs-CPU blur comparison isolate summation-order effects cleanly)
    for (int i = -radius; i <= radius; ++i) {
        int sx = x + i;
        // clamp-to-edge: reflect out-of-range columns onto the nearest
        // valid one (min/max, not modulo — a hard boundary, not a wrap).
        if (sx < 0) sx = 0;
        if (sx >= W) sx = W - 1;
        acc += weights[i + radius] * src[y * W + sx];
    }
    dst[y * W + x] = acc;
}

__global__ void gaussian_blur_v_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       int W, int H, const float* __restrict__ weights, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        int sy = y + i;
        if (sy < 0) sy = 0;
        if (sy >= H) sy = H - 1;
        acc += weights[i + radius] * src[sy * W + x];
    }
    dst[y * W + x] = acc;
}

// launch_gaussian_blur — orchestrates ONE full 2-D blur: build the 1-D
// weight table on the host (shared data, see kernels.cuh), upload it,
// horizontal pass into d_tmp, vertical pass into d_dst. d_tmp is a
// caller-owned [W*H] scratch buffer (passed in rather than allocated here)
// so main.cu's octave-building loop can reuse ONE scratch buffer across
// every blur call instead of malloc/free-ing per level.
void launch_gaussian_blur(const float* d_src, float* d_dst, int W, int H, float sigma, float* d_tmp)
{
    float h_weights[kMaxGaussTaps];
    int radius = 0;
    build_gaussian_kernel_1d(sigma, h_weights, radius);   // shared host helper, kernels.cuh

    float* d_weights = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weights, static_cast<size_t>(2 * radius + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, static_cast<size_t>(2 * radius + 1) * sizeof(float), cudaMemcpyHostToDevice));

    gaussian_blur_h_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_src, d_tmp, W, H, d_weights, radius);
    CUDA_CHECK_LAST_ERROR("gaussian_blur_h_kernel launch");
    gaussian_blur_v_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_tmp, d_dst, W, H, d_weights, radius);
    CUDA_CHECK_LAST_ERROR("gaussian_blur_v_kernel launch");

    CUDA_CHECK(cudaFree(d_weights));
}

// ===========================================================================
// 1b) DOWNSAMPLE 2x — the between-octave step. A pure MAP: thread (x,y)
//     owns output pixel (x,y) and reads exactly ONE source texel, at
//     (2x,2y) -- nearest-neighbor decimation, no averaging/pre-filter of
//     its own. This is safe (not aliased) specifically because the image
//     being downsampled is ALWAYS the pyramid level whose blur has already
//     reached sigma = 2*kSigma0 (see main.cu's octave-building loop) --
//     that blur IS this step's anti-alias pre-filter, exactly Lowe's
//     original justification for reusing an already-computed pyramid
//     level instead of re-filtering from scratch.
// ===========================================================================
__global__ void downsample2x_kernel(const float* __restrict__ src, int srcW, int srcH, float* __restrict__ dst)
{
    const int dstW = srcW / 2, dstH = srcH / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;
    dst[y * dstW + x] = src[(2 * y) * srcW + (2 * x)];
}

void launch_downsample2x(const float* d_src, int srcW, int srcH, float* d_dst)
{
    const int dstW = srcW / 2, dstH = srcH / 2;
    downsample2x_kernel<<<grid2d(dstW, dstH), dim3(kBlock2D, kBlock2D)>>>(d_src, srcW, srcH, d_dst);
    CUDA_CHECK_LAST_ERROR("downsample2x_kernel launch");
}

// ===========================================================================
// 2) DoG SUBTRACT — the simplest kernel in this file: a pure MAP.
// ===========================================================================
__global__ void dog_subtract_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                    float* __restrict__ dst, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;
    dst[idx] = a[idx] - b[idx];   // DoG[i] = Gaussian[i+1] - Gaussian[i] (caller passes a=higher-sigma, b=lower-sigma)
}

void launch_dog_subtract(const float* d_a, const float* d_b, float* d_dst, int W, int H)
{
    dog_subtract_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_a, d_b, d_dst, W, H);
    CUDA_CHECK_LAST_ERROR("dog_subtract_kernel launch");
}

// ===========================================================================
// 3) DoG EXTREMA — a 3x3x3 STENCIL (26 neighbors: 8 in the SAME DoG layer,
//    9 in the layer ABOVE, 9 in the layer BELOW) + ATOMIC COMPACTION,
//    directly extending 01.04's nms_select_fast_kernel from a 2-D
//    neighborhood to a 3-D (space x space x scale) one.
//
// A candidate (x,y) is kept iff dog_center[x,y] is STRICTLY greater than
// all 26 neighbors, OR strictly less than all 26 (a "blob-ness" test: SIFT
// keypoints are LOCAL EXTREMA of the scale-normalized Laplacian
// approximation, dark or light blobs alike — THEORY.md derives why). The
// STRICT inequality is the same deliberate tie-breaking choice 01.04's
// nms_select_fast_kernel documents: a tie with any neighbor (in ANY of the
// 3 layers) suppresses the candidate, a small and CONSISTENT loss (both
// this kernel and dog_extrema_cpu use '>'/'<', never '>=' / '<=') that
// keeps the candidate SET well-defined even though DoG values are float
// (see main.cu's "boundary ties" VERIFY note for what happens when the
// GPU and CPU pyramids disagree by less than a float ULP right at a tie).
//
// contrast pre-filter: skip the full 26-neighbor scan for any pixel whose
// |D| is already below kContrastThreshold — cheap to check FIRST (one
// read, one compare) and it rejects the overwhelming majority of pixels
// (flat/low-texture regions), so this ordering matters for real throughput
// even though this teaching kernel does not otherwise chase performance.
// ===========================================================================
__global__ void dog_extrema_candidates_kernel(const float* __restrict__ dog_below,
                                              const float* __restrict__ dog_center,
                                              const float* __restrict__ dog_above,
                                              int W, int H, int octave, int layer,
                                              DogCandidate* __restrict__ out,
                                              int* __restrict__ counter, int max_candidates)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (x < kExtremaBorder || x >= W - kExtremaBorder || y < kExtremaBorder || y >= H - kExtremaBorder) return;

    const float center = dog_center[y * W + x];
    if (fabsf(center) < kContrastThreshold) return;   // cheap pre-filter (see header)

    bool is_max = true, is_min = true;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nidx = (y + dy) * W + (x + dx);
            // same-layer neighbors: skip the center itself (dx==dy==0)
            if (!(dx == 0 && dy == 0)) {
                const float v = dog_center[nidx];
                if (v >= center) is_max = false;
                if (v <= center) is_min = false;
            }
            const float vb = dog_below[nidx];
            if (vb >= center) is_max = false;
            if (vb <= center) is_min = false;
            const float va = dog_above[nidx];
            if (va >= center) is_max = false;
            if (va <= center) is_min = false;
        }
    }
    if (!is_max && !is_min) return;

    const int slot = atomicAdd(counter, 1);   // same unordered-append pattern as 01.04's NMS compaction
    if (slot < max_candidates) {
        out[slot] = DogCandidate{ octave, layer, x, y };
    }
    // slot >= max_candidates: silently dropped, *counter still reflects the
    // TRUE total so the host can detect and report an overflow.
}

int launch_dog_extrema(const float* d_dog_below, const float* d_dog_center, const float* d_dog_above,
                       int W, int H, int octave, int layer,
                       DogCandidate* d_out, int max_candidates)
{
    int* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    dog_extrema_candidates_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(
        d_dog_below, d_dog_center, d_dog_above, W, H, octave, layer, d_out, d_counter, max_candidates);
    CUDA_CHECK_LAST_ERROR("dog_extrema_candidates_kernel launch");

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_counter));

    if (h_count > max_candidates) {
        std::fprintf(stderr, "[warn] launch_dog_extrema(octave=%d,layer=%d): %d candidates found, capacity %d -- some were dropped\n",
                    octave, layer, h_count, max_candidates);
        h_count = max_candidates;
    }
    return h_count;
}

// ===========================================================================
// 4) REFINE — one thread per RAW candidate: the iterative quadratic-Taylor
//    sub-pixel/sub-scale solve (Brown & Lowe's method, as adopted by SIFT).
//
// D is modeled locally as a 2nd-order Taylor expansion around the integer
// candidate (x0,y0,layer0):
//     D(z) ~= D0 + grad(D)^T z + 0.5 z^T H z,    z = (dx,dy,ds)
// The offset that makes this quadratic STATIONARY (its gradient zero) is
// z* = -H^-1 grad(D) — a single 3x3 LINEAR SOLVE (the small-matrix-solve
// pattern project 33.01 studies in its batched form; this kernel is a
// single-instance, per-thread version of exactly that operation). If any
// component of z* exceeds 0.5 (the candidate's TRUE optimum lies closer to
// a NEIGHBORING integer sample than to this one), Lowe's algorithm
// RE-CENTERS the search at the nearest integer neighbor and repeats, up to
// kMaxRefineIters times — because the quadratic model is only trustworthy
// near the sample it was built around.
//
// Thread-to-data mapping: thread i owns candidates[i]. Memory: reads a
// handful of DoG samples per iteration from GLOBAL memory (no sharing
// between threads — candidates are typically sparse and scattered, so
// shared-memory tiling would rarely pay off here, unlike the dense
// per-pixel stencils above).
// ===========================================================================

// solve_3x3_device — Cramer's-rule solve of H*z = rhs for a 3x3 SYMMETRIC
// H (this project's Hessian). Returns false (singular/ill-conditioned,
// caller must reject the candidate) if |det(H)| is too small to trust.
// Written INDEPENDENTLY of reference_cpu.cpp's refine_keypoint_cpu solver
// (own file-local helper, no shared code with the CPU twin) -- per the
// twin-independence ruling, this is exactly the kind of small numerical
// core that should be re-derived, not copy-pasted, so a transcription bug
// in one does not silently hide behind the other (see kernels.cuh header).
static __device__ inline float det3x3_device(const float M[3][3])
{
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

static __device__ inline bool solve_3x3_device(const float H[3][3], const float rhs[3], float z[3])
{
    const float det = det3x3_device(H);
    if (fabsf(det) < 1e-12f) return false;   // near-singular Hessian: the quadratic model has no reliable stationary point here
    const float inv_det = 1.0f / det;

    // Cramer's rule: z_k = det(H with column k replaced by rhs) / det(H).
    // Cheap enough (a few dozen FLOPs) that a per-thread, per-iteration
    // solve is not a bottleneck even across kMaxRefineIters=5 iterations.
    const float Hx[3][3] = { {rhs[0], H[0][1], H[0][2]}, {rhs[1], H[1][1], H[1][2]}, {rhs[2], H[2][1], H[2][2]} };
    const float Hy[3][3] = { {H[0][0], rhs[0], H[0][2]}, {H[1][0], rhs[1], H[1][2]}, {H[2][0], rhs[2], H[2][2]} };
    const float Hz[3][3] = { {H[0][0], H[0][1], rhs[0]}, {H[1][0], H[1][1], rhs[1]}, {H[2][0], H[2][1], rhs[2]} };
    z[0] = det3x3_device(Hx) * inv_det;
    z[1] = det3x3_device(Hy) * inv_det;
    z[2] = det3x3_device(Hz) * inv_det;
    return true;
}

__global__ void refine_keypoint_kernel(const float* __restrict__ dog, int W, int H,
                                       const DogCandidate* __restrict__ candidates, int n,
                                       SiftKeypoint* __restrict__ out, int* __restrict__ accepted)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const DogCandidate c = candidates[i];
    int x = c.x, y = c.y, layer = c.layer;
    accepted[i] = 0;   // pessimistic default; only set to 1 on a clean accept below

    float z[3] = { 0.0f, 0.0f, 0.0f };   // (dx, dy, ds), the running sub-pixel/sub-scale offset

    for (int iter = 0; iter < kMaxRefineIters; ++iter) {
        // Bounds check EVERY iteration (re-centering can walk toward the
        // border, or off the searchable-layer range): D(x,y,layer) plus a
        // full 3x3x3 neighborhood (needed for gradient+Hessian below) must
        // stay inside [1, W-2] x [1, H-2] x [1, kDogPerOctave-2] -- the
        // layer bound is INTENTIONALLY tighter than [0, kDogPerOctave-1]
        // because the Ds/Dss finite differences read layer-1 and layer+1.
        if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1 || layer < 1 || layer >= kDogPerOctave - 1) return;

        const float* Dm = dog + (layer - 1) * W * H;   // DoG one layer BELOW
        const float* D0 = dog + layer * W * H;          // DoG at the current layer
        const float* Dp = dog + (layer + 1) * W * H;    // DoG one layer ABOVE

        // Central-difference gradient (see this kernel's header for the
        // Taylor-expansion derivation).
        const float Dx = 0.5f * (D0[y * W + (x + 1)] - D0[y * W + (x - 1)]);
        const float Dy = 0.5f * (D0[(y + 1) * W + x] - D0[(y - 1) * W + x]);
        const float Ds = 0.5f * (Dp[y * W + x] - Dm[y * W + x]);

        const float d0 = D0[y * W + x];
        const float Dxx = D0[y * W + (x + 1)] - 2.0f * d0 + D0[y * W + (x - 1)];
        const float Dyy = D0[(y + 1) * W + x] - 2.0f * d0 + D0[(y - 1) * W + x];
        const float Dss = Dp[y * W + x] - 2.0f * d0 + Dm[y * W + x];
        const float Dxy = 0.25f * (D0[(y + 1) * W + (x + 1)] - D0[(y - 1) * W + (x + 1)]
                                  - D0[(y + 1) * W + (x - 1)] + D0[(y - 1) * W + (x - 1)]);
        const float Dxs = 0.25f * (Dp[y * W + (x + 1)] - Dp[y * W + (x - 1)]
                                  - Dm[y * W + (x + 1)] + Dm[y * W + (x - 1)]);
        const float Dys = 0.25f * (Dp[(y + 1) * W + x] - Dp[(y - 1) * W + x]
                                  - Dm[(y + 1) * W + x] + Dm[(y - 1) * W + x]);

        const float Hmat[3][3] = { {Dxx, Dxy, Dxs}, {Dxy, Dyy, Dys}, {Dxs, Dys, Dss} };
        const float neg_grad[3] = { -Dx, -Dy, -Ds };

        if (!solve_3x3_device(Hmat, neg_grad, z)) return;   // singular Hessian: reject (see solve_3x3_device's header)

        if (fabsf(z[0]) < kRefineConvergeTol && fabsf(z[1]) < kRefineConvergeTol && fabsf(z[2]) < kRefineConvergeTol) {
            // Refined DoG value at the sub-pixel optimum (2nd-order Taylor
            // estimate) -- Lowe's stronger, POST-refinement contrast test.
            const float d_hat = d0 + 0.5f * (Dx * z[0] + Dy * z[1] + Ds * z[2]);
            if (fabsf(d_hat) < kContrastThreshold) return;   // refined contrast too weak: reject

            // Principal-curvature (edge) rejection using the 2x2 SPATIAL
            // block of the Hessian only (Dxx,Dyy,Dxy -- Lowe's original
            // edge test ignores the scale axis): a candidate on an EDGE has
            // one large and one small principal curvature (THEORY.md
            // connects this to 01.04's structure-tensor/Harris story --
            // same "ratio of eigenvalues" idea, different formula).
            const float tr = Dxx + Dyy;
            const float det2 = Dxx * Dyy - Dxy * Dxy;
            if (det2 <= 0.0f) return;   // negative/zero determinant: a saddle, never a blob -- reject
            const float ratio_test = (tr * tr) / det2;
            const float ratio_bound = (kEdgeRatioR + 1.0f) * (kEdgeRatioR + 1.0f) / kEdgeRatioR;
            if (ratio_test >= ratio_bound) return;   // edge-like: reject

            // ACCEPT -- write the refined keypoint.
            SiftKeypoint kp;
            kp.octave = c.octave;
            kp.layer = layer;
            kp.x_oct = static_cast<float>(x) + z[0];
            kp.y_oct = static_cast<float>(y) + z[1];
            kp.ds = z[2];
            const float scale2x = static_cast<float>(1 << c.octave);   // 2^octave: maps octave-local px -> original-image px
            kp.x_img = kp.x_oct * scale2x;
            kp.y_img = kp.y_oct * scale2x;
            kp.sigma_oct = sigma_at(static_cast<float>(layer) + z[2]);
            kp.sigma_img = kp.sigma_oct * scale2x;
            kp.contrast = fabsf(d_hat);
            out[i] = kp;
            accepted[i] = 1;
            return;
        }

        // Not converged: re-center at the NEAREST integer neighbor implied
        // by z (Lowe's re-centering rule) and iterate again. lroundf gives
        // round-half-away-from-zero, matching reference_cpu.cpp's std::lround
        // twin (kept algorithmically independent -- see that file).
        x += static_cast<int>(lroundf(z[0]));
        y += static_cast<int>(lroundf(z[1]));
        layer += static_cast<int>(lroundf(z[2]));
    }
    // Loop exhausted without converging: implicit reject (accepted[i] stays 0).
}

void launch_refine_keypoints(const float* d_dog, int W, int H,
                             const DogCandidate* d_candidates, int n,
                             SiftKeypoint* d_out, int* d_accepted)
{
    if (n <= 0) return;
    refine_keypoint_kernel<<<grid1d(n), kBlock1D>>>(d_dog, W, H, d_candidates, n, d_out, d_accepted);
    CUDA_CHECK_LAST_ERROR("refine_keypoint_kernel launch");
    // Compaction happens on the HOST (main.cu downloads d_out + d_accepted
    // and filters) rather than a second device pass -- n is small
    // (<= kMaxDogCandidates) so a host-side filter is simpler and cheap.
}

// ===========================================================================
// 5) ORIENTATION — ONE WARP (32 threads) PER KEYPOINT. THE centerpiece
//    kernel of this project (catalog hook: "warp-level reductions").
//
// Why a warp, and not one thread per keypoint or one thread per sample?
// -----------------------------------------------------------------------
//   * One thread per keypoint (like 01.04's orientation_kernel, which
//     sums ~700 samples SERIALLY per keypoint): correct, but leaves 31 of
//     every warp's 32 lanes computing a DIFFERENT keypoint's orientation
//     while doing the SAME amount of serial work each -- fine for ORB's
//     small integer centroid, wasteful once the per-keypoint patch grows
//     (SIFT's orientation patch can be 30+ px across, ~700+ samples, and
//     THIS project also weights by a Gaussian, computes atan2f per sample,
//     and needs a 36-slot HISTOGRAM, not a running sum -- more work per
//     keypoint than ORB's centroid by a wide margin).
//   * One thread per SAMPLE PIXEL, all keypoints' samples flattened into
//     one giant launch: maximal parallelism, but every sample must
//     scatter-add into ITS keypoint's histogram bin -- a genuine
//     cross-thread race needing atomics (see the "naive" alternative
//     below), and bookkeeping which sample belongs to which keypoint adds
//     real complexity for a small, already-cheap-per-thread workload.
//   * ONE WARP PER KEYPOINT is the sweet spot: 32-way parallelism per
//     keypoint (enough to matter — a ~700-sample patch becomes ~22
//     samples/lane), while keeping every keypoint's work self-contained
//     in one block, and avoiding cross-thread atomics ENTIRELY via the
//     scheme below.
//
// The scheme, in two phases:
//   Phase 1 (embarrassingly parallel, NO shared memory, NO atomics): each
//     lane strides over its 1/32 share of the patch's pixels and
//     accumulates weighted-magnitude contributions into its OWN PRIVATE
//     36-float histogram (`local_hist`). Lanes never touch each other's
//     data here -- zero contention, by construction.
//   Phase 2 (the warp-level REDUCTION): for EACH of the 36 bins, in turn,
//     every lane's local_hist[b] is summed across the warp via a
//     five-step __shfl_down_sync TREE (offsets 16,8,4,2,1 -- each step
//     halves the number of "active" partial sums, in O(log2(32))=5 steps
//     instead of 32 serial adds), landing the bin's TOTAL in lane 0's
//     register. Lane 0 alone then writes hist[b] to shared memory.
//
// The NAIVE alternative, for contrast (NOT implemented here, on purpose):
//     __shared__ float hist[36] = {0};
//     for (each of my strided samples) atomicAdd(&hist[bin], weight);
//   This is SHORTER code, but 32 lanes are hashing into only 36 shared-
//   memory slots -- by the birthday paradox, MANY samples across the warp
//   collide on the same bin in the same instruction, and every collision
//   SERIALIZES (hardware atomics resolve one write at a time per address).
//   A single dominant gradient direction (the common case near a real
//   corner/blob -- exactly what SIFT is designed to find) makes this
//   WORSE, not better: most samples pile into 1-2 bins, so most atomics
//   serialize. The local-then-shuffle-reduce scheme above touches shared
//   memory only kOriHistBins=36 times PER KEYPOINT (once per bin, by lane
//   0 only) and NEVER contends -- the entire accumulation phase is
//   perfectly parallel, and only the O(log2(32)) reduction phase pays for
//   cross-lane communication, via the fastest cross-lane path a GPU has
//   (register-to-register shuffle, no memory access at all).
//
// Numerics: each lane's local_hist lives in a 36-element per-thread array.
// Because the bin INDEX is data-dependent (computed from a sample's
// gradient angle), the compiler generally cannot keep this array in
// registers and instead spills it to per-thread LOCAL memory (a
// thread-private slice of the GPU's global memory space, cached through
// L1/L2 like any other memory) -- an honest, named performance tradeoff
// for a project processing at most a few hundred keypoints, not millions
// (THEORY.md "GPU mapping" discusses a shared-memory-per-warp alternative
// for higher-keypoint-count regimes).
// ===========================================================================

// parabolic_peak_offset — small, LANE-0-ONLY helper local to this file
// (never shared with reference_cpu.cpp -- see this file's header on
// internal helpers vs. cross-file sharing).
static __device__ inline float parabolic_peak_offset(float left, float center, float right)
{
    // Fit a parabola through 3 equally-spaced samples and return the
    // (sub-bin) offset of ITS peak from the center sample -- the standard
    // "3-point parabolic interpolation" used throughout signal processing
    // for sub-sample peak localization.
    const float denom = left - 2.0f * center + right;
    if (fabsf(denom) < 1e-12f) return 0.0f;   // three co-linear samples: no curvature to interpolate, treat the center as the peak
    return 0.5f * (left - right) / denom;
}

// emit_oriented_keypoint_slot — write ONE oriented-keypoint entry into a
// block's private fixed-slot sub-range and advance its LOCAL (non-atomic)
// counter, passed BY REFERENCE (ordinary C++ reference semantics work in
// device code exactly as on the host -- no CUDA-specific trick here, just
// a plain __device__ helper instead of a device lambda, so this file needs
// no --extended-lambda compiler flag).
static __device__ inline void emit_oriented_keypoint_slot(const SiftKeypoint& kp, float bin_interp, int num_bins,
                                                           OrientedKeypoint* my_out, int& spawned)
{
    float theta = bin_interp * (2.0f * kPi / static_cast<float>(num_bins));
    if (theta < 0.0f) theta += 2.0f * kPi;
    if (theta >= 2.0f * kPi) theta -= 2.0f * kPi;
    my_out[spawned].kp = kp;
    my_out[spawned].theta = theta;
    ++spawned;
}

__global__ void orientation_kernel(const float* __restrict__ gauss_oct, int W, int H,
                                   const SiftKeypoint* __restrict__ kps, int n,
                                   OrientedKeypoint* __restrict__ out, int* __restrict__ out_spawn_count)
{
    const int kp_idx = blockIdx.x;         // one BLOCK per keypoint
    if (kp_idx >= n) return;
    const int lane = threadIdx.x;          // 0..31 -- blockDim.x MUST be kWarpSize (see launch_orientation)
    // This block's PRIVATE output sub-range (see kernels.cuh's header for
    // why fixed slots, not an atomic-compacted list, are the right choice
    // here): no other block ever touches out[kp_idx*cap .. +cap), so lane
    // 0 below writes it with a purely local counter, no atomics at all.
    OrientedKeypoint* my_out = out + kp_idx * kMaxOrientedPerKeypoint;

    const SiftKeypoint kp = kps[kp_idx];
    const float* img = gauss_oct + kp.layer * W * H;   // this keypoint's own Gaussian pyramid image (see kernels.cuh header)

    const float sigma_w = kOriSigmaFactor * kp.sigma_oct;   // Gaussian weighting window std-dev
    const int radius = max(1, static_cast<int>(lroundf(kOriRadiusFactor * sigma_w)));
    const int cx = static_cast<int>(lroundf(kp.x_oct));
    const int cy = static_cast<int>(lroundf(kp.y_oct));
    const int side = 2 * radius + 1;
    const int num_samples = side * side;
    const float two_sigma_w_sq = 2.0f * sigma_w * sigma_w;

    // ---- Phase 1: strided, per-lane, PRIVATE accumulation (see header) ----
    float local_hist[kOriHistBins];
    #pragma unroll
    for (int b = 0; b < kOriHistBins; ++b) local_hist[b] = 0.0f;

    for (int s = lane; s < num_samples; s += kWarpSize) {
        const int dy = s / side - radius;
        const int dx = s % side - radius;
        const int x = cx + dx;
        const int y = cy + dy;
        if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) continue;         // stay inside the gradient stencil's needs
        if (dx * dx + dy * dy > radius * radius) continue;                // isotropic DISK, not the bounding square (same idea as 01.04's rBRIEF sampling disk)

        // Central-difference image gradient. Row-major layout means
        // INCREASING y is DOWN the image; using (y-1) minus (y+1) for the
        // vertical term keeps "gy positive" meaning "brighter upward" --
        // a standard image-processing sign convention, documented here so
        // atan2f(gy, gx) below produces angles matching a normal
        // right-handed on-screen picture rather than a flipped one.
        const float gx = img[y * W + (x + 1)] - img[y * W + (x - 1)];
        const float gy = img[(y - 1) * W + x] - img[(y + 1) * W + x];
        const float mag = sqrtf(gx * gx + gy * gy);

        float angle = atan2f(gy, gx);                 // [-pi, pi]
        if (angle < 0.0f) angle += 2.0f * kPi;         // wrap to [0, 2*pi)

        const float weight = expf(-static_cast<float>(dx * dx + dy * dy) / two_sigma_w_sq);   // Gaussian-weighted vote: samples farther from the keypoint count less

        int bin = static_cast<int>(floorf(angle * (static_cast<float>(kOriHistBins) / (2.0f * kPi))));
        if (bin < 0) bin = 0;
        if (bin >= kOriHistBins) bin = kOriHistBins - 1;   // guard the rare angle==2*pi float edge case

        local_hist[bin] += mag * weight;
    }

    // ---- Phase 2: the warp-shuffle TREE REDUCTION (see header) ------------
    __shared__ float hist[kOriHistBins];   // final, reduced histogram -- only lane 0 writes and reads it back (peak-finding is serial and cheap at 36 bins)
    #pragma unroll
    for (int b = 0; b < kOriHistBins; ++b) {
        float v = local_hist[b];
        // Butterfly/tree reduction: each step folds the "upper half" of
        // the still-active lanes into the "lower half" via a register-to-
        // register shuffle (no shared memory, no global memory, no
        // synchronization barrier needed -- shuffles are implicitly
        // warp-synchronous). After 5 steps (32 -> 16 -> 8 -> 4 -> 2 -> 1),
        // lane 0 holds the sum of all 32 lanes' local_hist[b].
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFFu, v, offset);   // full-warp mask: every lane in this block participates (blockDim.x == 32 exactly)
        }
        if (lane == 0) hist[b] = v;
    }

    if (lane != 0) return;   // everything below is serial, cheap (36-element scans), and needs only ONE lane

    // ---- Smoothing: Lowe's classic 5-tap circular [1,4,6,4,1]/16 pass,
    // one application -- damps single-bin noise spikes so parabolic
    // interpolation (below) fits a genuinely smooth local peak instead of
    // quantization jitter (THEORY.md "Numerical considerations"). --------
    float smoothed[kOriHistBins];
    #pragma unroll
    for (int b = 0; b < kOriHistBins; ++b) {
        const int m2 = (b - 2 + kOriHistBins) % kOriHistBins;
        const int m1 = (b - 1 + kOriHistBins) % kOriHistBins;
        const int p1 = (b + 1) % kOriHistBins;
        const int p2 = (b + 2) % kOriHistBins;
        smoothed[b] = (hist[m2] + hist[p2]) * (1.0f / 16.0f) + (hist[m1] + hist[p1]) * (4.0f / 16.0f) + hist[b] * (6.0f / 16.0f);
    }

    float max_val = 0.0f; int max_bin = 0;
    #pragma unroll
    for (int b = 0; b < kOriHistBins; ++b) if (smoothed[b] > max_val) { max_val = smoothed[b]; max_bin = b; }
    if (max_val <= 0.0f) { out_spawn_count[kp_idx] = 0; return; }   // a perfectly flat local patch (no gradient signal at all) -- vanishingly rare on real content, but honestly possible near a border; no orientation to assign, this keypoint is silently dropped

    // Local (non-atomic) fan-out counter -- see this kernel's header and
    // kernels.cuh's contract comment for why a purely local counter is
    // both correct and sufficient here (bounded, statically-known max
    // fan-out per block, exclusive output sub-range).
    int spawned = 0;

    // Primary peak: always emitted (it IS the maximum, so it trivially
    // clears the kOriPeakRatio bar) -- interpolated using its own
    // immediate circular neighbors even if one happens to tie it exactly.
    {
        const int m1 = (max_bin - 1 + kOriHistBins) % kOriHistBins;
        const int p1 = (max_bin + 1) % kOriHistBins;
        const float off = parabolic_peak_offset(smoothed[m1], smoothed[max_bin], smoothed[p1]);
        emit_oriented_keypoint_slot(kp, static_cast<float>(max_bin) + off, kOriHistBins, my_out, spawned);
    }
    // Secondary peaks: any OTHER strict local maximum at >= kOriPeakRatio
    // of the primary's height spawns an ADDITIONAL keypoint copy at that
    // orientation (Lowe's multi-orientation rule -- a corner where two
    // edges meet at a shallow angle genuinely has two "dominant"
    // directions, and treating it as one keypoint with one orientation
    // would silently throw away a real, matchable feature).
    #pragma unroll
    for (int b = 0; b < kOriHistBins && spawned < kMaxOrientedPerKeypoint; ++b) {
        if (b == max_bin) continue;
        const int m1 = (b - 1 + kOriHistBins) % kOriHistBins;
        const int p1 = (b + 1) % kOriHistBins;
        const float v = smoothed[b];
        if (v > smoothed[m1] && v > smoothed[p1] && v >= kOriPeakRatio * max_val) {
            const float off = parabolic_peak_offset(smoothed[m1], v, smoothed[p1]);
            emit_oriented_keypoint_slot(kp, static_cast<float>(b) + off, kOriHistBins, my_out, spawned);
        }
    }
    out_spawn_count[kp_idx] = spawned;
}

void launch_orientation(const float* d_gauss_oct, int W, int H, const SiftKeypoint* d_kps, int n,
                        OrientedKeypoint* d_out, int* d_spawn_count)
{
    if (n <= 0) return;
    orientation_kernel<<<n, kWarpSize>>>(d_gauss_oct, W, H, d_kps, n, d_out, d_spawn_count);
    CUDA_CHECK_LAST_ERROR("orientation_kernel launch");
}

// ===========================================================================
// 6) DESCRIBE — the SAME one-warp-per-keypoint, local-accumulate-then-
//    shuffle-reduce pattern as orientation_kernel, scaled from 36 bins to
//    128 (4x4 spatial cells x 8 orientation bins), with TRILINEAR (row x
//    col x orientation) soft binning instead of orientation's simple
//    nearest-bin assignment -- SIFT's actual descriptor formula (Lowe
//    2004 section 6.1), not a simplified stand-in.
//
// Why trilinear, not nearest-bin: a sample sitting near a cell/bin BOUNDARY
// should influence BOTH neighbors somewhat, not flip its entire vote to
// one side or the other from a 1-pixel jitter -- exactly the same
// robustness argument 01.04's rBRIEF avoided needing (binary tests have no
// such boundary) but SIFT's floating-point descriptor benefits from
// directly: soft binning is a big part of why SIFT descriptors are more
// robust to small geometric noise than a naive nearest-bin histogram.
// ===========================================================================
__global__ void describe_kernel(const float* __restrict__ gauss_oct, int W, int H,
                                const OrientedKeypoint* __restrict__ kps, int n,
                                SiftDescriptor* __restrict__ desc_out)
{
    const int kp_idx = blockIdx.x;
    if (kp_idx >= n) return;
    const int lane = threadIdx.x;

    const OrientedKeypoint okp = kps[kp_idx];
    const SiftKeypoint& kp = okp.kp;
    const float* img = gauss_oct + kp.layer * W * H;

    const float cos_t = cosf(okp.theta);
    const float sin_t = sinf(okp.theta);
    const float hist_width = kDescScaleFactor * kp.sigma_oct;   // pixels-per-cell, this keypoint's own scale

    // Radius covers the rotated d x d grid PLUS the +-1-cell interpolation
    // margin (sqrt(2) for the diagonal, (d+1)/2 half-extent) -- Lowe's
    // exact formula, defensively capped (see kDescMaxRadius's comment).
    int radius = static_cast<int>(lroundf(hist_width * 1.41421356f * (kDescGridSize + 1) * 0.5f));
    radius = max(1, min(radius, kDescMaxRadius));
    const int cx = static_cast<int>(lroundf(kp.x_oct));
    const int cy = static_cast<int>(lroundf(kp.y_oct));
    const int side = 2 * radius + 1;
    const int num_samples = side * side;

    const float half_d = kDescGridSize * 0.5f;             // 2.0 -- also the descriptor Gaussian window's sigma (Lowe: sigma = d/2)
    const float two_sigma_desc_sq = 2.0f * half_d * half_d;

    // ---- Phase 1: strided, per-lane, PRIVATE accumulation into all 128
    // bins (see orientation_kernel's header for the full "why a warp, why
    // private-then-reduce, why not naive atomics" argument -- identical
    // reasoning here, just 128 bins instead of 36). --------------------------
    float local_hist[kDescDims];
    #pragma unroll
    for (int b = 0; b < kDescDims; ++b) local_hist[b] = 0.0f;

    for (int s = lane; s < num_samples; s += kWarpSize) {
        const int dy = s / side - radius;
        const int dx = s % side - radius;
        const int x = cx + dx;
        const int y = cy + dy;
        if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) continue;

        // Rotate the sample offset by -theta (into the keypoint's own,
        // orientation-aligned frame) and rescale by 1/hist_width so
        // coordinates are in CELL units -- this is what makes the
        // descriptor ROTATION INVARIANT: every keypoint's sampling grid is
        // defined relative to ITS dominant direction, not the image axes.
        const float rx = (cos_t * dx + sin_t * dy) / hist_width;
        const float ry = (-sin_t * dx + cos_t * dy) / hist_width;

        // Shift so cell CENTERS land on integers 0..d-1 (Lowe's convention:
        // cell c spans roughly [c-0.5, c+0.5) in these units).
        const float rbin = rx + half_d - 0.5f;
        const float cbin = ry + half_d - 0.5f;
        if (rbin <= -1.0f || rbin >= kDescGridSize || cbin <= -1.0f || cbin >= kDescGridSize) continue;   // outside the 4x4 grid's interpolation support entirely

        const float gx = img[y * W + (x + 1)] - img[y * W + (x - 1)];
        const float gy = img[(y - 1) * W + x] - img[(y + 1) * W + x];
        const float mag = sqrtf(gx * gx + gy * gy);
        float angle = atan2f(gy, gx);
        if (angle < 0.0f) angle += 2.0f * kPi;

        float rel_angle = angle - okp.theta;   // rotate the GRADIENT direction into the keypoint's frame too -- same invariance argument, applied to orientation instead of position
        if (rel_angle < 0.0f) rel_angle += 2.0f * kPi;
        if (rel_angle >= 2.0f * kPi) rel_angle -= 2.0f * kPi;
        const float obin = rel_angle * (static_cast<float>(kDescOriBins) / (2.0f * kPi));   // continuous orientation-bin coordinate, [0, 8)

        const float gauss_w = expf(-(rx * rx + ry * ry) / two_sigma_desc_sq);   // descriptor-window Gaussian, in CELL units (Lowe: falls off across the whole 4x4 grid, de-emphasizing its edges)
        const float w = gauss_w * mag;

        // Trilinear soft-binning: distribute `w` across the (up to) 8
        // (row, col, orientation) bins surrounding (rbin, cbin, obin),
        // weighted by linear interpolation fractions along each axis.
        const int r0 = static_cast<int>(floorf(rbin));
        const int c0 = static_cast<int>(floorf(cbin));
        const int o0 = static_cast<int>(floorf(obin));
        const float rfrac = rbin - r0, cfrac = cbin - c0, ofrac = obin - o0;

        #pragma unroll
        for (int dr = 0; dr <= 1; ++dr) {
            const int rr = r0 + dr;
            if (rr < 0 || rr >= kDescGridSize) continue;
            const float wr = dr ? rfrac : (1.0f - rfrac);
            #pragma unroll
            for (int dc = 0; dc <= 1; ++dc) {
                const int cc = c0 + dc;
                if (cc < 0 || cc >= kDescGridSize) continue;
                const float wc = dc ? cfrac : (1.0f - cfrac);
                #pragma unroll
                for (int doo = 0; doo <= 1; ++doo) {
                    int oo = (o0 + doo) % kDescOriBins;
                    if (oo < 0) oo += kDescOriBins;   // orientation bins WRAP (0 and 7 are circular neighbors) -- unlike row/col, which are hard-edged
                    const float wo = doo ? ofrac : (1.0f - ofrac);
                    const int bin_idx = (rr * kDescGridSize + cc) * kDescOriBins + oo;
                    local_hist[bin_idx] += w * wr * wc * wo;
                }
            }
        }
    }

    // ---- Phase 2: warp-shuffle reduce all 128 bins (identical mechanism
    // to orientation_kernel, just 128 iterations of the same 5-step tree
    // instead of 36). ---------------------------------------------------------
    __shared__ float hist[kDescDims];
    #pragma unroll
    for (int b = 0; b < kDescDims; ++b) {
        float v = local_hist[b];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
        if (lane == 0) hist[b] = v;
    }

    if (lane != 0) return;

    // ---- L2-normalize -> clip at kDescClipValue -> RE-normalize. --------
    // WHY clip: a bright specular highlight or a saturated-sensor edge can
    // make ONE gradient direction locally dominate far more than
    // "normal" illumination would (a NONLINEAR camera-response effect, not
    // a simple brightness/contrast SCALE that L2-normalization alone
    // already handles) -- capping any single bin's contribution keeps a
    // few outlier pixels from swamping the whole descriptor, at the cost
    // of a small, deliberate loss of discriminative power for the
    // (rare) legitimately-huge-gradient case. Lowe's paper reports this
    // measurably improves matching robustness under real illumination
    // changes (THEORY.md cites the number).
    float norm_sq = 0.0f;
    #pragma unroll
    for (int b = 0; b < kDescDims; ++b) norm_sq += hist[b] * hist[b];
    const float inv_norm = (norm_sq > 1e-20f) ? rsqrtf(norm_sq) : 0.0f;

    float clipped[kDescDims];
    float norm2_sq = 0.0f;
    #pragma unroll
    for (int b = 0; b < kDescDims; ++b) {
        const float v = fminf(hist[b] * inv_norm, kDescClipValue);
        clipped[b] = v;
        norm2_sq += v * v;
    }
    // Numerics note (see THEORY.md for the full discussion): because
    // clipping can only ever SHRINK a component, the post-clip norm is
    // <= 1, so this second normalization SCALES UP -- meaning a component
    // that was exactly AT the 0.2 clip boundary can end up SLIGHTLY ABOVE
    // 0.2 after this step. This is expected, well-known SIFT descriptor
    // behavior, not a bug; main.cu's descriptor-normalization gate uses a
    // MEASURED, not a naive 0.2-exact, bound for this reason.
    const float inv_norm2 = (norm2_sq > 1e-20f) ? rsqrtf(norm2_sq) : 0.0f;
    #pragma unroll
    for (int b = 0; b < kDescDims; ++b) desc_out[kp_idx].v[b] = clipped[b] * inv_norm2;
}

void launch_describe(const float* d_gauss_oct, int W, int H, const OrientedKeypoint* d_kps, int n,
                     SiftDescriptor* d_desc)
{
    if (n <= 0) return;
    describe_kernel<<<n, kWarpSize>>>(d_gauss_oct, W, H, d_kps, n, d_desc);
    CUDA_CHECK_LAST_ERROR("describe_kernel launch");
}

// ===========================================================================
// 7) BRUTE-FORCE SQUARED-L2 MATCH — one thread per QUERY descriptor,
//    scanning every TRAIN descriptor (same all-pairs structure as 01.04's
//    hamming_match_kernel, now over 128 FLOATS instead of 8 uint32 WORDS
//    per comparison -- see this kernel's header for the measured cost
//    difference the project brief asks us to report).
//
// Why SQUARED L2 (no sqrtf in the inner loop): the ratio test and the
// max-distance cap only ever compare distances to EACH OTHER or to a
// threshold, and both operations are monotonic under x -> sqrt(x) for
// x >= 0 -- so "best <= ratio * second" on plain distances is EXACTLY
// equivalent to "best_sq <= ratio^2 * second_sq" on squared distances (see
// kernels.cuh's kLoweRatioSift comment), and skipping 2 sqrtf calls per
// (query, train) pair for every one of the O(nQuery*nTrain) comparisons
// is a real, free saving -- the one sqrtf a caller actually wants (for a
// human-readable distance in a CSV) is computed ONCE, after this kernel,
// on just the winning pair (see main.cu).
// ===========================================================================
__global__ void match_l2_kernel(const SiftDescriptor* __restrict__ query, int nQuery,
                                const SiftDescriptor* __restrict__ train, int nTrain,
                                float* __restrict__ best1_dist_sq, int* __restrict__ best1_idx,
                                float* __restrict__ best2_dist_sq, int* __restrict__ best2_idx)
{
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nQuery) return;

    const SiftDescriptor q = query[qi];   // one register/local-memory-resident copy, reused nTrain times

    float b1 = 1.0e30f; int i1 = -1;   // sentinel: strictly worse than any real squared-L2 distance (max possible, for unit vectors, is 4.0)
    float b2 = 1.0e30f; int i2 = -1;

    for (int ti = 0; ti < nTrain; ++ti) {
        const SiftDescriptor& t = train[ti];
        float dist_sq = 0.0f;
        #pragma unroll
        for (int d = 0; d < kDescDims; ++d) {
            const float diff = q.v[d] - t.v[d];
            dist_sq += diff * diff;   // the 128-float analogue of 01.04's popcount(q.w[w] ^ t.w[w]) -- a multiply+add per dimension instead of one XOR+POPC per 32 dimensions
        }
        if (dist_sq < b1) {
            b2 = b1; i2 = i1;
            b1 = dist_sq; i1 = ti;
        } else if (dist_sq < b2) {
            b2 = dist_sq; i2 = ti;
        }
    }

    best1_dist_sq[qi] = b1; best1_idx[qi] = i1;
    best2_dist_sq[qi] = b2; best2_idx[qi] = i2;
}

void launch_match_l2(const SiftDescriptor* d_query, int nQuery, const SiftDescriptor* d_train, int nTrain,
                     float* d_best1_dist_sq, int* d_best1_idx, float* d_best2_dist_sq, int* d_best2_idx)
{
    if (nQuery <= 0 || nTrain <= 0) return;
    match_l2_kernel<<<grid1d(nQuery), kBlock1D>>>(
        d_query, nQuery, d_train, nTrain, d_best1_dist_sq, d_best1_idx, d_best2_dist_sq, d_best2_idx);
    CUDA_CHECK_LAST_ERROR("match_l2_kernel launch");
}
