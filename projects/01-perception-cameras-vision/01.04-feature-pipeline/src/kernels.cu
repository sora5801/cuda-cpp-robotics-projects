// ===========================================================================
// kernels.cu — GPU kernels for project 01.04
//              Feature pipeline: FAST/Harris detection, ORB descriptors,
//              brute-force Hamming matcher
//
// Big idea (the whole project in one paragraph)
// -----------------------------------------------
// Every stage below is a MAP or a small STENCIL over an independent unit of
// work — a pixel (detection), a keypoint (orientation, description), or a
// query descriptor (matching) — which is exactly why sparse-feature
// pipelines are such a natural GPU fit: millions of pixels visited once,
// then a much smaller, embarrassingly-parallel per-keypoint stage, then an
// all-pairs comparison that is a map over queries with a small serial loop
// inside. Eight kernels, three families:
//   1. fast_score_kernel, sobel_gradient_kernel, harris_response_kernel
//        — per-PIXEL maps/stencils: one thread per pixel, entirely
//          independent, no shared memory (a teaching simplification named
//          honestly in each kernel's header — THEORY.md's "GPU mapping"
//          section derives the shared-memory-tiled faster version).
//   2. nms_select_fast_kernel, nms_select_harris_kernel
//        — per-pixel STENCIL + ATOMIC COMPACTION: reads a 3x3 neighborhood,
//          conditionally appends to a shared output list via atomicAdd.
//   3. orientation_kernel, describe_kernel, hamming_match_kernel
//        — per-KEYPOINT (or per-QUERY) maps: one thread per keypoint is the
//          natural mapping once detection has cut millions of pixels down
//          to a few hundred keypoints (argued in each kernel's header).
//
// All shared layouts, constants, and lookup-table builders are single-
// sourced in kernels.cuh — read that file's header first; it explains the
// bit-exact-vs-tolerance twin strategy this file and reference_cpu.cpp both
// depend on.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (an
// INDEPENDENT re-implementation of every algorithmic core below — diff
// them side by side to see what "the same algorithm, twice" looks like).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry shared by every "one thread per pixel" kernel in this
// file: a 16x16 2-D block (256 threads, a warp multiple) and a grid sized
// to exactly cover kW x kH with a ragged-tail guard inside each kernel —
// the same idiom 01.02's census_kernel uses (see that file's comment for
// the occupancy argument; at 256x256 this is 16x16 = 256 blocks, several
// times over an RTX 2080 SUPER's 46 SMs).
// ---------------------------------------------------------------------------
static constexpr int kBlock2D = 16;

static inline dim3 grid2d_pixels()
{
    return dim3((kW + kBlock2D - 1) / kBlock2D, (kH + kBlock2D - 1) / kBlock2D);
}

// Launch geometry for the "one thread per keypoint/query" kernels: a plain
// 1-D grid-stride-free launch (n is always small here — at most kTopNFast
// = 300 — so a single generous grid with a tail guard, no stride loop
// needed; contrast with the SAXPY placeholder's grid-stride loop, which
// exists for n potentially in the billions).
static constexpr int kBlock1D = 128;

static inline int grid1d(int n)
{
    return (n + kBlock1D - 1) / kBlock1D;
}

// ---------------------------------------------------------------------------
// Device-visible copies of the FAST circle-offset tables. kernels.cuh
// defines the VALUES once as macros (FAST_CIRCLE_X_INIT etc.) precisely so
// these __constant__-memory arrays and reference_cpu.cpp's plain host
// arrays (kFastCircleX/Y/kFastQuadIdx) can never drift apart — see that
// header's comment for why device code cannot simply read the host arrays
// directly (a CUDA language rule, not a workaround).
//
// __constant__ memory (as opposed to plain __device__ global memory) is
// the right home for this data because EVERY thread in EVERY block reads
// the SAME 16 (or 4) values on every launch: the constant cache broadcasts
// one read to an entire warp in a single transaction, versus each thread
// separately hitting global memory — the textbook constant-memory use case
// (uniform, read-only, small, hot).
// ---------------------------------------------------------------------------
__constant__ int kFastCircleXDev[kFastCircleN] = FAST_CIRCLE_X_INIT;
__constant__ int kFastCircleYDev[kFastCircleN] = FAST_CIRCLE_Y_INIT;
__constant__ int kFastQuadIdxDev[4] = FAST_QUAD_IDX_INIT;

// ===========================================================================
// 1) FAST-9 SCORE — a STENCIL kernel: thread (x,y) reads a fixed 16-pixel
//    ring (kFastCircleX/Y, radius 3) around itself and decides "corner or
//    not", with a numeric strength if so.
//
// The contiguous-arc test, illustrated (the Bresenham circle of 16 points,
// index 0 = north, clockwise — see kernels.cuh for the exact offsets):
//
//                14 15  0  1  2
//             13                3
//             12       P        4      P = center pixel (x,y)
//             11                5      ring = the 16 numbered points
//                10  9  8  7  6
//
//   FAST-9 asks: is there a run of >= 9 CONSECUTIVE ring points (wrapping
//   around) that are ALL brighter than I(P)+t, or ALL darker than I(P)-t?
//   If the run only needs to be found (not located precisely), a classic
//   pigeonhole argument gives a cheap pre-filter: points {0,4,8,12} are 4
//   apart around a 16-point ring, so ANY 9-length contiguous run must
//   include at least 3 of those 4 (a run of 9 skips at most 7 consecutive
//   points, and no 4-apart quartet can have 2 uncovered gaps of 7 without
//   covering >= 3 of the 4 marked points) — the "high-speed test" below.
//
// Thread-to-data mapping: thread (bx*16+tx, by*16+ty) owns pixel
// (x,y) = (blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y).
// Memory: reads up to 17 bytes (center + 16 ring samples) from GLOBAL
// memory per thread — no shared-memory tiling (a teaching simplification;
// THEORY.md's "GPU mapping" derives the tiled version, which would still
// help less here than in a large-window stencil like 01.02's 7x7 census,
// since 16 samples spread over a radius-3 ring already reuses little
// between adjacent threads). No atomics, no shared memory.
// ===========================================================================
__global__ void fast_score_kernel(const uint8_t* __restrict__ img, int* __restrict__ score_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kW || y >= kH) return;                 // ragged-tail guard (grid rounds up)
    const int idx = y * kW + x;

    // Border pixels: the 16-point ring would read outside the image.
    // score_out is 0 here, unconditionally (no corner is EVER reported
    // this close to the edge — a real limitation of a fixed-size detector,
    // documented in README "Limitations & honesty").
    if (x < kDetectBorder || x >= kW - kDetectBorder ||
        y < kDetectBorder || y >= kH - kDetectBorder) {
        score_out[idx] = 0;
        return;
    }

    const int Ip = static_cast<int>(img[idx]);        // center pixel intensity, 0..255

    // Sample the 16-point ring ONCE into registers (each point read from
    // GLOBAL memory exactly once — reused by both the quick test and the
    // full test below, which is the entire reason to hoist it out here
    // instead of re-reading img[] inline in each test).
    int ring[kFastCircleN];
    #pragma unroll
    for (int i = 0; i < kFastCircleN; ++i) {
        ring[i] = static_cast<int>(img[(y + kFastCircleYDev[i]) * kW + (x + kFastCircleXDev[i])]);
    }

    // ---- high-speed quick test (pigeonhole pre-filter, see header) -------
    int quad_bright = 0, quad_dark = 0;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int v = ring[kFastQuadIdxDev[q]];
        if (v > Ip + kFastThreshold) ++quad_bright;
        else if (v < Ip - kFastThreshold) ++quad_dark;
    }
    if (quad_bright < 3 && quad_dark < 3) {
        // Neither polarity can possibly have a 9-run among the full 16 —
        // proven by the pigeonhole argument above, so skip the O(16*9*2)
        // full test entirely. This is the "high-speed" half of FAST;
        // production FAST replaces this HAND-WRITTEN quick test with a
        // TRAINED DECISION TREE over all 16 comparisons (see THEORY.md).
        score_out[idx] = 0;
        return;
    }

    // ---- full contiguous-arc test + score, both polarities ---------------
    // For EACH polarity that passed the quick test, scan all 16 possible
    // arc starting points; for each start, check the 9 consecutive ring
    // indices (mod 16); if all 9 qualify, the arc's STRENGTH is the
    // MINIMUM per-point margin within it (how far every point in the arc
    // clears the threshold — a corner is only as strong as its weakest
    // qualifying point, a standard "worst case in the run" score). Track
    // the BEST (max) such strength over every start and both polarities:
    // this both answers "is P a corner" (best > 0) and ranks corners for
    // the NMS stage. Deterministic given the ring array (integer-only:
    // BIT-EXACT reproducible on any IEEE-754-agnostic integer ALU).
    int best_margin = 0;   // 0 == "not a corner"; a real corner always yields >= 1 (see kernels.cuh comment)

    if (quad_bright >= 3) {
        #pragma unroll
        for (int start = 0; start < kFastCircleN; ++start) {
            int margin = 2147483647;   // INT_MAX sentinel; shrinks to the arc's weakest point
            bool ok = true;
            #pragma unroll
            for (int j = 0; j < kFastArcLen; ++j) {
                const int v = ring[(start + j) % kFastCircleN];
                const int m = v - Ip - kFastThreshold;   // > 0 required to qualify as "bright"
                if (m <= 0) { ok = false; break; }
                if (m < margin) margin = m;
            }
            if (ok && margin > best_margin) best_margin = margin;
        }
    }
    if (quad_dark >= 3) {
        #pragma unroll
        for (int start = 0; start < kFastCircleN; ++start) {
            int margin = 2147483647;
            bool ok = true;
            #pragma unroll
            for (int j = 0; j < kFastArcLen; ++j) {
                const int v = ring[(start + j) % kFastCircleN];
                const int m = Ip - v - kFastThreshold;   // > 0 required to qualify as "dark"
                if (m <= 0) { ok = false; break; }
                if (m < margin) margin = m;
            }
            if (ok && margin > best_margin) best_margin = margin;
        }
    }

    score_out[idx] = best_margin;
}

void launch_fast_score(const uint8_t* d_img, int* d_score)
{
    fast_score_kernel<<<grid2d_pixels(), dim3(kBlock2D, kBlock2D)>>>(d_img, d_score);
    CUDA_CHECK_LAST_ERROR("fast_score_kernel launch");
}

// ===========================================================================
// 2) SOBEL GRADIENTS — a small STENCIL kernel (3x3 neighborhood -> two
//    derivative estimates). Border: kSobelBorder (1), NOT kDetectBorder —
//    see kernels.cuh's kSobelBorder comment for why Sobel gets its OWN,
//    smaller border (Harris's box window later reads out to +-2 from ITS
//    center, and needs those gradient values to be real, not zero-forced).
//
// Numerics: every Sobel weight is a small integer (-2,-1,0,1,2), and the
// input is uint8 (0..255), so Gx, Gy are EXACT integers up to magnitude
// 4*255 = 1020 — representable without rounding error in float32 (exact
// up to 2^24). Storing them as float (rather than int) is purely for
// convenience in harris_response_kernel's box-sum accumulation below;
// no precision is lost by this choice at these magnitudes.
// ===========================================================================
__global__ void sobel_gradient_kernel(const uint8_t* __restrict__ img,
                                      float* __restrict__ gx_out,
                                      float* __restrict__ gy_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kW || y >= kH) return;
    const int idx = y * kW + x;

    if (x < kSobelBorder || x >= kW - kSobelBorder ||
        y < kSobelBorder || y >= kH - kSobelBorder) {
        gx_out[idx] = 0.0f;
        gy_out[idx] = 0.0f;
        return;
    }

    // Read the 3x3 neighborhood once into named locals — clearer than
    // re-indexing img[] nine times inline, and the compiler coalesces
    // these loads across a warp regardless (adjacent threads read
    // adjacent columns of each row).
    const int i00 = img[(y - 1) * kW + (x - 1)], i01 = img[(y - 1) * kW + x], i02 = img[(y - 1) * kW + (x + 1)];
    const int i10 = img[ y      * kW + (x - 1)],                              i12 = img[ y      * kW + (x + 1)];
    const int i20 = img[(y + 1) * kW + (x - 1)], i21 = img[(y + 1) * kW + x], i22 = img[(y + 1) * kW + (x + 1)];

    // Standard Sobel 3x3 kernels:
    //   Gx = [-1 0 1; -2 0 2; -1 0 1]   (horizontal derivative: emphasizes vertical edges)
    //   Gy = [-1 -2 -1; 0 0 0; 1 2 1]   (vertical derivative: emphasizes horizontal edges)
    const int gx = (i02 + 2 * i12 + i22) - (i00 + 2 * i10 + i20);
    const int gy = (i20 + 2 * i21 + i22) - (i00 + 2 * i01 + i02);

    gx_out[idx] = static_cast<float>(gx);   // exact — see header numerics note
    gy_out[idx] = static_cast<float>(gy);
}

void launch_sobel_gradient(const uint8_t* d_img, float* d_gx, float* d_gy)
{
    sobel_gradient_kernel<<<grid2d_pixels(), dim3(kBlock2D, kBlock2D)>>>(d_img, d_gx, d_gy);
    CUDA_CHECK_LAST_ERROR("sobel_gradient_kernel launch");
}

// ===========================================================================
// 3) HARRIS RESPONSE — a STENCIL kernel over the GRADIENT images: thread
//    (x,y) sums (Gx^2, Gy^2, Gx*Gy) over a (2*kHarrisWinRadius+1)^2 = 5x5
//    box window (25 taps), forming the 2x2 structure tensor
//
//        M = [ Sxx  Sxy ]      Sxx = sum(Gx^2), Syy = sum(Gy^2),
//            [ Sxy  Syy ]      Sxy = sum(Gx*Gy)      (all summed over the window)
//
//    then the Harris-Stephens response R = det(M) - k*trace(M)^2
//    (= lambda1*lambda2 - k*(lambda1+lambda2)^2 in terms of M's
//    eigenvalues lambda1,lambda2 — THEORY.md derives WHY large-and-similar
//    eigenvalues, i.e. a large R, means "corner": a small window in EITHER
//    principal gradient direction produces a big brightness change).
//
// Why a box window (uniform weight) and not a Gaussian window (as Harris'
// original paper and OpenCV's cornerHarris use)? A box sum is a single
// nested loop with equal weights — easier to read line-by-line as the
// "sum gradients over a neighborhood" idea with zero extra machinery; a
// Gaussian window would need a separable 1-D kernel pass (more code, more
// kernels) for a response that is directionally smoother but not
// qualitatively different for this project's teaching purpose. THEORY.md
// names the Gaussian-window version as the production refinement.
//
// No shared-memory tiling (each thread re-reads its own 25-tap window from
// global memory independently) — the same honestly-named simplification
// as every stencil in this file; THEORY.md's "GPU mapping" derives the
// tiled version (a natural next step: cache a (16+4)x(16+4) tile of gx/gy
// per BLOCK in shared memory, since neighboring threads' windows overlap
// heavily at radius 2).
// ===========================================================================
__global__ void harris_response_kernel(const float* __restrict__ gx,
                                       const float* __restrict__ gy,
                                       float* __restrict__ response_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kW || y >= kH) return;
    const int idx = y * kW + x;

    if (x < kDetectBorder || x >= kW - kDetectBorder ||
        y < kDetectBorder || y >= kH - kDetectBorder) {
        response_out[idx] = 0.0f;
        return;
    }

    float Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f;
    #pragma unroll
    for (int wy = -kHarrisWinRadius; wy <= kHarrisWinRadius; ++wy) {
        #pragma unroll
        for (int wx = -kHarrisWinRadius; wx <= kHarrisWinRadius; ++wx) {
            const int widx = (y + wy) * kW + (x + wx);   // valid: kDetectBorder(3) >= kHarrisWinRadius(2) + kSobelBorder(1), see kernels.cuh
            const float gxv = gx[widx];
            const float gyv = gy[widx];
            Sxx += gxv * gxv;
            Syy += gyv * gyv;
            Sxy += gxv * gyv;
        }
    }

    const float det   = Sxx * Syy - Sxy * Sxy;
    const float trace  = Sxx + Syy;
    response_out[idx] = det - kHarrisK * trace * trace;
}

void launch_harris_response(const float* d_gx, const float* d_gy, float* d_response)
{
    harris_response_kernel<<<grid2d_pixels(), dim3(kBlock2D, kBlock2D)>>>(d_gx, d_gy, d_response);
    CUDA_CHECK_LAST_ERROR("harris_response_kernel launch");
}

// ===========================================================================
// 4) NMS + COMPACTION — thread (x,y), restricted to the keypoint-eligible
//    interior [kBorder, kW-kBorder) x [kBorder, kH-kBorder), reads its own
//    score and its 8 immediate neighbors' scores (a 3x3 stencil over the
//    ALREADY-COMPUTED score/response map — cheap, no recomputation) and
//    keeps the pixel iff it is a STRICT local maximum above the floor.
//
// Tie-breaking (why STRICT '>', not '>='): if two adjacent pixels have the
// EXACT same score, '>' means NEITHER is a local max, so both are
// suppressed — a small, deterministic, and CONSISTENT loss (both the GPU
// kernel and reference_cpu.cpp's independent NMS use the same '>' rule),
// which matters because main.cu compares the two paths' FINAL keypoint
// LISTS for bit-exact equality on FAST (see kernels.cuh's header): any
// tie-break ambiguity would make that comparison fragile. A real plateau
// (multiple pixels at the identical integer FAST score) is possible but
// rare on natural imagery; losing it is a documented, honest simplification
// (README "Limitations & honesty"), not a bug.
//
// Compaction: kept candidates are appended, UNORDERED (the order threads
// finish in is not deterministic — that's fine, main.cu SORTS the
// downloaded list by (score desc, y asc, x asc) before doing anything with
// it, which both restores determinism AND is the tie-break rule the CPU
// path's fast_nms_select_cpu() also applies internally).
// ===========================================================================
__global__ void nms_select_fast_kernel(const int* __restrict__ score,
                                       int* __restrict__ out_x,
                                       int* __restrict__ out_y,
                                       int* __restrict__ out_score,
                                       int* __restrict__ counter,
                                       int max_candidates)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kW || y >= kH) return;
    if (x < kBorder || x >= kW - kBorder || y < kBorder || y >= kH - kBorder) return;

    const int s = score[y * kW + x];
    if (s <= 0) return;   // not a corner at all

    bool is_max = true;
    #pragma unroll
    for (int dy = -1; dy <= 1 && is_max; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (score[(y + dy) * kW + (x + dx)] >= s) { is_max = false; break; }   // '>=': a tie also suppresses (see header)
        }
    }
    if (!is_max) return;

    const int slot = atomicAdd(counter, 1);   // reserve a unique output slot, whatever order threads arrive in
    if (slot < max_candidates) {
        out_x[slot] = x;
        out_y[slot] = y;
        out_score[slot] = s;
    }
    // slot >= max_candidates: candidate silently dropped (capacity
    // exceeded); *counter still reflects the TRUE total so the host can
    // detect and report an overflow rather than being fooled by silence.
}

// The FLOAT twin of the kernel above, over the Harris response map, with
// an extra pre-filter `thresh` (Harris responses can be a wide range of
// magnitudes — see kernels.cu numerics note below — so a floor separate
// from "> 0" is needed to reject the flat/near-flat majority of the image
// before even considering the 3x3 comparison).
__global__ void nms_select_harris_kernel(const float* __restrict__ response,
                                         float thresh,
                                         int* __restrict__ out_x,
                                         int* __restrict__ out_y,
                                         float* __restrict__ out_score,
                                         int* __restrict__ counter,
                                         int max_candidates)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kW || y >= kH) return;
    if (x < kBorder || x >= kW - kBorder || y < kBorder || y >= kH - kBorder) return;

    const float s = response[y * kW + x];
    if (s <= thresh) return;

    bool is_max = true;
    #pragma unroll
    for (int dy = -1; dy <= 1 && is_max; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (response[(y + dy) * kW + (x + dx)] >= s) { is_max = false; break; }
        }
    }
    if (!is_max) return;

    const int slot = atomicAdd(counter, 1);
    if (slot < max_candidates) {
        out_x[slot] = x;
        out_y[slot] = y;
        out_score[slot] = s;
    }
}

// launch_nms_select_fast/harris — own the ephemeral device counter (alloc,
// zero, launch, download the final count, free) so main.cu only has to
// manage the OUTPUT arrays it actually needs downstream. Returns the
// count ACTUALLY WRITTEN (clamped to max_candidates); if the true count
// exceeded capacity, main.cu compares against this function's raw count
// via a second, unclamped read — see the [info] line it prints.
int launch_nms_select_fast(const int* d_score, int* d_out_x, int* d_out_y, int* d_out_score, int max_candidates)
{
    int* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    nms_select_fast_kernel<<<grid2d_pixels(), dim3(kBlock2D, kBlock2D)>>>(
        d_score, d_out_x, d_out_y, d_out_score, d_counter, max_candidates);
    CUDA_CHECK_LAST_ERROR("nms_select_fast_kernel launch");

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_counter));

    if (h_count > max_candidates) {
        std::fprintf(stderr, "[warn] launch_nms_select_fast: %d candidates found, capacity %d -- some were dropped\n",
                    h_count, max_candidates);
        h_count = max_candidates;
    }
    return h_count;
}

int launch_nms_select_harris(const float* d_response, float thresh, int* d_out_x, int* d_out_y, float* d_out_score, int max_candidates)
{
    int* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    nms_select_harris_kernel<<<grid2d_pixels(), dim3(kBlock2D, kBlock2D)>>>(
        d_response, thresh, d_out_x, d_out_y, d_out_score, d_counter, max_candidates);
    CUDA_CHECK_LAST_ERROR("nms_select_harris_kernel launch");

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_counter));

    if (h_count > max_candidates) {
        std::fprintf(stderr, "[warn] launch_nms_select_harris: %d candidates found, capacity %d -- some were dropped\n",
                    h_count, max_candidates);
        h_count = max_candidates;
    }
    return h_count;
}

// ===========================================================================
// 5) ORIENTATION — one thread per KEYPOINT (not per pixel): detection has
//    already cut ~65,536 pixels down to at most kTopNFast = 300 keypoints,
//    so the natural GPU mapping flips from "one thread per pixel" to "one
//    thread per keypoint", each doing MORE work (a ~700-sample disk sum)
//    over a MUCH smaller work list — the right-sized granularity for the
//    data volume at this stage (THEORY.md "GPU mapping" makes this
//    argument explicitly, including why launching 300 threads still keeps
//    the GPU busy: an RTX 2080 SUPER can have thousands of threads
//    in flight, so 300 independent ~700-iteration loops is still
//    trivially cheap, just not "saturate every SM" cheap).
//
// The intensity centroid (Rosin 1999, adopted by ORB for orientation):
//   m10 = sum_{(dx,dy) in disk} dx * I(x+dx, y+dy)
//   m01 = sum_{(dx,dy) in disk} dy * I(x+dx, y+dy)
//   theta = atan2(m01, m10)
// Intuition: if the patch is uniformly lit, gray-weighted "mass" is
// balanced around the center and m10=m01=0 (no preferred direction). Real
// corners/edges are lopsided — more bright mass on one side — and the
// centroid vector points from the keypoint TOWARD that mass, which ORB
// treats as "the patch's dominant direction" for canceling out in-plane
// rotation (THEORY.md "The problem" derives why this needs to be
// canceled at all: brightness constancy holds along the LOCAL gradient
// direction, and BRIEF's raw pixel-pair comparisons are NOT rotation
// invariant on their own).
//
// Numerics: dx, dy, and I are all small integers (|dx|,|dy| <= 15, I in
// 0..255), so m10, m01 are accumulated in INT (exact, no rounding — see
// kernels.cuh's numerics note): the only floating-point operation in this
// entire kernel is the final atan2f() call, which isolates the "orientation
// tolerance" twin's entire source of GPU/CPU divergence to one hardware
// transcendental function versus the host's libm implementation.
// ===========================================================================
__global__ void orientation_kernel(const uint8_t* __restrict__ img,
                                   const int* __restrict__ kp_x,
                                   const int* __restrict__ kp_y,
                                   int n,
                                   float* __restrict__ theta_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int cx = kp_x[i];
    const int cy = kp_y[i];
    const int R = kOrientPatchRadius;

    int m10 = 0, m01 = 0;   // exact integer accumulators (see header numerics note)
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            if (dx * dx + dy * dy > R * R) continue;   // isotropic DISK, not the bounding square
            const int I = static_cast<int>(img[(cy + dy) * kW + (cx + dx)]);
            m10 += dx * I;
            m01 += dy * I;
        }
    }
    theta_out[i] = atan2f(static_cast<float>(m01), static_cast<float>(m10));
}

void launch_orientation(const uint8_t* d_img, const int* d_kp_x, const int* d_kp_y, int n, float* d_theta)
{
    if (n <= 0) return;
    orientation_kernel<<<grid1d(n), kBlock1D>>>(d_img, d_kp_x, d_kp_y, n, d_theta);
    CUDA_CHECK_LAST_ERROR("orientation_kernel launch");
}

// ===========================================================================
// 6) DESCRIBE — one thread per keypoint (same granularity argument as
//    orientation_kernel above). For each of the 256 precomputed, bin-
//    rotated sample pairs, read two pixel intensities and pack one
//    comparison bit — the rBRIEF test from the ORB paper (Rublee et al.
//    2011): bit_k = [ I(keypoint + pair_k.offset1) < I(keypoint + pair_k.offset2) ].
//
// Why this kernel achieves BIT-EXACT agreement with describe_cpu() despite
// running on different hardware: EVERY input is already an exact integer
// by the time this kernel runs (kernels.cuh's header explains why bin_idx
// is safe to treat as shared, validated data) — img[] is uint8, table[]
// is precomputed integer offsets, and "<" between two uint8 values has
// exactly one correct answer. There is no floating point ANYWHERE in this
// kernel. That is the entire trick.
// ===========================================================================
__global__ void describe_kernel(const uint8_t* __restrict__ img,
                                const int* __restrict__ kp_x,
                                const int* __restrict__ kp_y,
                                const int* __restrict__ bin_idx,
                                int n,
                                const RotatedOffset* __restrict__ table,
                                OrbDescriptor* __restrict__ desc_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int cx = kp_x[i];
    const int cy = kp_y[i];
    const int bin = bin_idx[i];
    const RotatedOffset* row = table + bin * kOrbNumPairs;

    OrbDescriptor d;
    #pragma unroll
    for (int w = 0; w < kOrbDescWords; ++w) d.w[w] = 0u;

    for (int k = 0; k < kOrbNumPairs; ++k) {
        const RotatedOffset o = row[k];
        // Defensive clamp: by construction (kBorder=16 > kOrientPatchRadius=15
        // and rotation preserves distance from the keypoint — see
        // kernels.cuh's kBorder comment) every sample stays in-bounds, but
        // an explicit clamp costs nothing and turns a hypothetical future
        // constant-tuning mistake into a silently-safe read instead of a
        // device-side out-of-bounds fault.
        int xa = cx + o.dx1, ya = cy + o.dy1;
        int xb = cx + o.dx2, yb = cy + o.dy2;
        xa = min(max(xa, 0), kW - 1); ya = min(max(ya, 0), kH - 1);
        xb = min(max(xb, 0), kW - 1); yb = min(max(yb, 0), kH - 1);

        const uint8_t Ia = img[ya * kW + xa];
        const uint8_t Ib = img[yb * kW + xb];
        if (Ia < Ib) {
            d.w[k >> 5] |= (1u << (k & 31));   // k>>5 == k/32 (word), k&31 == k%32 (bit position), LSB-first
        }
    }
    desc_out[i] = d;
}

void launch_describe(const uint8_t* d_img, const int* d_kp_x, const int* d_kp_y, const int* d_bin_idx,
                     int n, const RotatedOffset* d_table, OrbDescriptor* d_desc)
{
    if (n <= 0) return;
    describe_kernel<<<grid1d(n), kBlock1D>>>(d_img, d_kp_x, d_kp_y, d_bin_idx, n, d_table, d_desc);
    CUDA_CHECK_LAST_ERROR("describe_kernel launch");
}

// ===========================================================================
// 7) BRUTE-FORCE HAMMING MATCH — one thread per QUERY descriptor, each
//    scanning every TRAIN descriptor (an O(nQuery * nTrain) all-pairs
//    comparison, exactly what "brute-force" means, as opposed to an
//    approximate structure like a k-d tree or an LSH hash table — THEORY.md
//    discusses why binary descriptors make brute force practical at
//    real-time rates even though it is asymptotically the "naive" choice).
//
// Hamming distance between two 256-bit descriptors = the number of bit
// positions where they DIFFER = popcount(a XOR b). __popc() is a single
// SASS instruction (POPC) on every CUDA-capable GPU — hardware population
// count, the same operation reference_cpu.cpp's popcount32_portable()
// implements by hand with the classic SWAR bit-trick (see that function's
// comment in kernels.cuh for the "what would this cost without the
// instruction" answer, per CLAUDE.md's no-black-boxes rule).
//
// Why Hamming replaced L2/SSD for real-time matching: comparing two N-bit
// binary strings costs N/32 XOR+POPC instructions (8, here) versus an
// N-dimensional float L2 distance (many multiplies, a sqrt) for a
// descriptor of comparable discriminative power — THEORY.md's "The
// algorithm" section works the operation-count comparison in full; this
// is precisely why ORB/BRISK/FREAK-style binary descriptors displaced
// SIFT/SURF's float descriptors in latency-sensitive robotics pipelines.
//
// Best + second-best via a single pass (no sort, no second kernel): each
// thread keeps a 2-slot running "top-2 smallest" — classic online
// selection, O(nTrain) time, O(1) extra memory per thread.
// ===========================================================================
__global__ void hamming_match_kernel(const OrbDescriptor* __restrict__ query, int nQuery,
                                     const OrbDescriptor* __restrict__ train, int nTrain,
                                     int* __restrict__ best1_dist, int* __restrict__ best1_idx,
                                     int* __restrict__ best2_dist, int* __restrict__ best2_idx)
{
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nQuery) return;

    const OrbDescriptor q = query[qi];   // one register-resident copy, reused nTrain times

    int b1 = kOrbNumPairs + 1, i1 = -1;   // sentinel: strictly worse than any real distance (max real distance = 256)
    int b2 = kOrbNumPairs + 1, i2 = -1;

    for (int ti = 0; ti < nTrain; ++ti) {
        const OrbDescriptor t = train[ti];
        int dist = 0;
        #pragma unroll
        for (int w = 0; w < kOrbDescWords; ++w) {
            dist += __popc(q.w[w] ^ t.w[w]);   // hardware population-count instruction (see header)
        }
        if (dist < b1) {
            b2 = b1; i2 = i1;
            b1 = dist; i1 = ti;
        } else if (dist < b2) {
            b2 = dist; i2 = ti;
        }
    }

    best1_dist[qi] = b1; best1_idx[qi] = i1;
    best2_dist[qi] = b2; best2_idx[qi] = i2;
}

void launch_hamming_match(const OrbDescriptor* d_query, int nQuery, const OrbDescriptor* d_train, int nTrain,
                          int* d_best1_dist, int* d_best1_idx, int* d_best2_dist, int* d_best2_idx)
{
    if (nQuery <= 0 || nTrain <= 0) return;
    hamming_match_kernel<<<grid1d(nQuery), kBlock1D>>>(
        d_query, nQuery, d_train, nTrain, d_best1_dist, d_best1_idx, d_best2_dist, d_best2_idx);
    CUDA_CHECK_LAST_ERROR("hamming_match_kernel launch");
}
