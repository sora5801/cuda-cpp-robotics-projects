// ===========================================================================
// kernels.cu — GPU kernels for project 01.01
//              Full GPU image pipeline: debayer -> undistort -> rectify ->
//              resize -> normalize, staged AND fused
//
// Big idea (the whole project in one paragraph)
// -----------------------------------------------
// Every stage below is a MAP: one thread, one output pixel, no
// cross-thread communication (the normalize stage's reduction is the one
// exception — a tree, not a map, see its own header comment). That makes
// image pipelines an easy first GPU program... until you ask HOW MANY
// kernels to use. The staged path below spells out five independent maps,
// each reading/writing a full image to GLOBAL memory between stages — the
// obvious, readable way to write it. The fused kernel collapses
// undistort+rectify+resize into ONE map that never writes the intermediate
// full-resolution image at all — the SAME math, fewer bytes moved. Reading
// both side by side, with the memory-traffic accounting main.cu prints, IS
// this project's kernel-fusion lesson (THEORY.md "The GPU mapping" derives
// the byte counts this file's structure implies).
//
// All shared layouts, the camera model, and every constant live in
// kernels.cuh — read that file's header comment first; it is not repeated
// here. Companion oracle: reference_cpu.cpp (an INDEPENDENT line-by-line
// CPU twin of every kernel below, except the shared camera-model formulas
// documented in kernels.cuh).
//
// Read this after: kernels.cuh.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (paragraph 6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry shared by every "one thread per pixel" kernel in this
// file: a 16x16 2-D block (256 threads, a warp-multiple; at kFullW x kFullH
// = 384x288 that is 24x18 = 432 blocks — tens of times more than an RTX
// 2080 SUPER's 48 SMs need to stay fed) and a grid sized to exactly cover
// W x H with a ragged-tail guard inside each kernel (the same
// ceil-divide-the-grid, if-guard-the-thread idiom as 01.02's kernels.cu).
// ---------------------------------------------------------------------------
static constexpr int kBlock2D = 16;

static inline dim3 grid2d(int W, int H)
{
    return dim3((W + kBlock2D - 1) / kBlock2D, (H + kBlock2D - 1) / kBlock2D);
}

// clampi — clamp an integer index into [lo, hi]. __device__-only: every
// caller below is inside a kernel; the CPU twin in reference_cpu.cpp
// defines its own host-side copy (a two-line function is exactly the case
// where re-typing costs nothing and keeps the twins independent, per the
// project's twin-independence ruling — see reference_cpu.cpp's header).
__device__ inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ===========================================================================
// 1) DEBAYER — bilinear demosaic of an RGGB Bayer mosaic.
//
// Every raw pixel measures exactly ONE of {R, G, B} (whichever color
// filter physically sits over that photosite — see kernels.cuh's
// bayer_channel_at() diagram); debayer must FABRICATE the other two
// channels from neighbors of the same color. This is a STENCIL (each
// thread reads up to 8 neighbors of the input, a 3x3 footprint) producing
// 3 output channels per input pixel — the "3x more data out than in"
// growth that makes debayer the first stage a real ISP always runs.
//
// The four cases (kernels.cuh's diagram, restated in code terms):
//   center is R: G = avg(N,S,E,W); B = avg(NE,NW,SE,SW) (diagonals)
//   center is B: G = avg(N,S,E,W); R = avg(NE,NW,SE,SW) (diagonals)
//   center is G, on an R-row (y even): R = avg(E,W); B = avg(N,S)
//   center is G, on a  B-row (y odd) : B = avg(E,W); R = avg(N,S)
// Border pixels clamp neighbor indices to the image edge (clampi) rather
// than reading out of bounds — a small, honest bias (repeated edge pixels
// pull the estimate slightly toward the edge value) that a production ISP
// usually avoids by processing a slightly larger raw frame with true
// optical black margins; THEORY.md names this simplification explicitly.
//
// Thread-to-data mapping: thread (bx*16+tx, by*16+ty) owns BOTH the one
// raw input pixel (x,y) and the one RGB output pixel (x,y) — same (x,y),
// different buffers, because debayer does not change resolution.
// ===========================================================================
__global__ void debayer_kernel(const unsigned char* __restrict__ bayer,
                               unsigned char* __restrict__ rgb, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;                          // ragged-tail guard

    // Clamped neighbor coordinates — computed once, reused by whichever
    // branch below needs them (compilers coalesce the unused ones away).
    const int xm = clampi(x - 1, 0, W - 1), xp = clampi(x + 1, 0, W - 1);
    const int ym = clampi(y - 1, 0, H - 1), yp = clampi(y + 1, 0, H - 1);

    // 4-connected (N,S,E,W) and 4 diagonal (NE,NW,SE,SW) raw samples —
    // every one of these 8 reads is a scattered global-memory access (no
    // shared-memory tiling here, a teaching simplification named honestly
    // in THEORY.md, same spirit as 01.02's census kernel).
    const unsigned char n_ = bayer[ym * W + x];
    const unsigned char s_ = bayer[yp * W + x];
    const unsigned char e_ = bayer[y * W + xp];
    const unsigned char w_ = bayer[y * W + xm];
    const unsigned char ne = bayer[ym * W + xp];
    const unsigned char nw = bayer[ym * W + xm];
    const unsigned char se = bayer[yp * W + xp];
    const unsigned char sw = bayer[yp * W + xm];
    const unsigned char center = bayer[y * W + x];

    float R, G, B;
    const int ch = bayer_channel_at(x, y);
    if (ch == 0) {                                          // native R
        R = static_cast<float>(center);
        G = 0.25f * (static_cast<float>(n_) + s_ + e_ + w_);
        B = 0.25f * (static_cast<float>(ne) + nw + se + sw);
    } else if (ch == 2) {                                   // native B
        B = static_cast<float>(center);
        G = 0.25f * (static_cast<float>(n_) + s_ + e_ + w_);
        R = 0.25f * (static_cast<float>(ne) + nw + se + sw);
    } else {                                                 // native G
        G = static_cast<float>(center);
        if ((y & 1) == 0) {                                 // G on an R-row: horiz neighbors R, vert neighbors B
            R = 0.5f * (static_cast<float>(e_) + w_);
            B = 0.5f * (static_cast<float>(n_) + s_);
        } else {                                             // G on a B-row: horiz neighbors B, vert neighbors R
            B = 0.5f * (static_cast<float>(e_) + w_);
            R = 0.5f * (static_cast<float>(n_) + s_);
        }
    }

    const int o = (y * W + x) * 3;
    // Clamp to [0,255] before the round-to-nearest cast: the averages
    // above are convex combinations of uint8 inputs so they can NEVER
    // leave [0,255] in exact arithmetic, but we clamp anyway — cheap
    // insurance against a float rounding nudge landing at -epsilon or
    // 255+epsilon, which would otherwise WRAP (not clamp) on the cast to
    // unsigned char.
    rgb[o + 0] = static_cast<unsigned char>(fminf(fmaxf(R, 0.0f), 255.0f) + 0.5f);
    rgb[o + 1] = static_cast<unsigned char>(fminf(fmaxf(G, 0.0f), 255.0f) + 0.5f);
    rgb[o + 2] = static_cast<unsigned char>(fminf(fmaxf(B, 0.0f), 255.0f) + 0.5f);
}

void launch_debayer_rggb(const unsigned char* d_bayer, unsigned char* d_rgb, int W, int H)
{
    debayer_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_bayer, d_rgb, W, H);
    CUDA_CHECK_LAST_ERROR("debayer_kernel launch");
}

// ===========================================================================
// 2) BUILD REMAP LUT — one thread per FULL-RESOLUTION output pixel, calling
// the shared compute_source_pixel() (kernels.cuh) and storing the result.
// Purely geometric (no image data touched at all) — this is why it is
// computed ONCE and reused by every consumer below, mirroring how a real
// camera driver builds its undistort map at calibration time and replays
// it every frame rather than re-deriving it (THEORY.md "Where this sits in
// the real world").
// ===========================================================================
__global__ void build_remap_lut_kernel(RemapSample* __restrict__ lut, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    lut[y * W + x] = compute_source_pixel(x, y);
}

void launch_build_remap_lut(RemapSample* d_lut, int W, int H)
{
    build_remap_lut_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_lut, W, H);
    CUDA_CHECK_LAST_ERROR("build_remap_lut_kernel launch");
}

// ---------------------------------------------------------------------------
// bilinear_sample_rgb — sample an interleaved RGB image at a fractional
// coordinate (u, v), clamp-to-edge outside the image. __device__-only: the
// STAGED remap kernel, the FUSED kernel, and NO ONE else on the GPU side
// call this (reference_cpu.cpp defines its own independent host copy —
// see kernels.cuh's file header on why bilinear sampling is deliberately
// NOT shared between the CPU and GPU paths, unlike the camera model).
//
// Clamping strategy: clamp the CONTINUOUS coordinate first, then floor —
// this guarantees x0/y0/x1/y1 are all valid indices without a branch per
// corner, and produces the standard "replicate the edge pixel" boundary
// behavior used throughout this repo's image kernels (01.02's census
// kernel instead REJECTS border pixels; remapping cannot reject — every
// output pixel must get an answer, so clamping is the correct choice here
// and THEORY.md names the difference).
// ---------------------------------------------------------------------------
__device__ inline void bilinear_sample_rgb(const unsigned char* __restrict__ img,
                                           int W, int H, float u, float v,
                                           float out[3])
{
    u = fminf(fmaxf(u, 0.0f), static_cast<float>(W - 1));
    v = fminf(fmaxf(v, 0.0f), static_cast<float>(H - 1));
    const int x0 = static_cast<int>(floorf(u));
    const int y0 = static_cast<int>(floorf(v));
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    const float fx = u - static_cast<float>(x0);            // horizontal interpolation weight, [0,1]
    const float fy = v - static_cast<float>(y0);            // vertical interpolation weight, [0,1]

#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float v00 = static_cast<float>(img[(y0 * W + x0) * 3 + c]);
        const float v10 = static_cast<float>(img[(y0 * W + x1) * 3 + c]);
        const float v01 = static_cast<float>(img[(y1 * W + x0) * 3 + c]);
        const float v11 = static_cast<float>(img[(y1 * W + x1) * 3 + c]);
        const float top = v00 + (v10 - v00) * fx;            // interpolate along x at row y0
        const float bot = v01 + (v11 - v01) * fx;            // interpolate along x at row y1
        out[c] = top + (bot - top) * fy;                     // interpolate the two rows along y
    }
}

// ===========================================================================
// 3) STAGED UNDISTORT+RECTIFY — one thread per full-resolution output
// pixel: look up its LUT entry, bilinear-sample the debayered image,
// round, write. This MATERIALIZES the full kFullW x kFullH remapped image
// in global memory — the write this stage produces (and the read the
// resize stage below must then perform) is exactly the traffic the FUSED
// kernel later eliminates.
// ===========================================================================
__global__ void remap_bilinear_kernel(const unsigned char* __restrict__ rgb_in,
                                      const RemapSample* __restrict__ lut,
                                      unsigned char* __restrict__ rgb_out, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const RemapSample s = lut[y * W + x];
    float rgb[3];
    bilinear_sample_rgb(rgb_in, W, H, s.u, s.v, rgb);

    const int o = (y * W + x) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c)
        rgb_out[o + c] = static_cast<unsigned char>(fminf(fmaxf(rgb[c], 0.0f), 255.0f) + 0.5f);
}

void launch_remap_bilinear(const unsigned char* d_rgb_in, const RemapSample* d_lut,
                           unsigned char* d_rgb_out, int W, int H)
{
    remap_bilinear_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_rgb_in, d_lut, d_rgb_out, W, H);
    CUDA_CHECK_LAST_ERROR("remap_bilinear_kernel launch");
}

// ===========================================================================
// 4) STAGED RESIZE — exact kResizeFactor x area-average downscale. One
// thread per OUTPUT (resized) pixel; each thread reads exactly
// kResizeFactor^2 = 4 input texels and averages them. This is the correct
// anti-aliasing filter for an INTEGER decimation factor: a box filter
// whose support exactly matches the decimation ratio means every input
// pixel contributes to EXACTLY ONE output pixel with EQUAL weight — no
// input sample is dropped (as nearest-neighbor would) or double-counted
// (as a naive bilinear resize, which was never designed for downscaling,
// can do when the sampling grid aliases) — THEORY.md derives the
// aliasing argument in full.
// ===========================================================================
__global__ void resize_area2x_kernel(const unsigned char* __restrict__ rgb_in,
                                     unsigned char* __restrict__ rgb_out, int Wf, int Hf)
{
    const int Wr = Wf / kResizeFactor, Hr = Hf / kResizeFactor;
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= Wr || yo >= Hr) return;

    const int x0 = xo * kResizeFactor, y0 = yo * kResizeFactor;
    float acc[3] = { 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int dy = 0; dy < kResizeFactor; ++dy) {
#pragma unroll
        for (int dx = 0; dx < kResizeFactor; ++dx) {
            const int o = ((y0 + dy) * Wf + (x0 + dx)) * 3;
#pragma unroll
            for (int c = 0; c < 3; ++c) acc[c] += static_cast<float>(rgb_in[o + c]);
        }
    }
    const float norm = 1.0f / static_cast<float>(kResizeFactor * kResizeFactor);
    const int oo = (yo * Wr + xo) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c)
        rgb_out[oo + c] = static_cast<unsigned char>(acc[c] * norm + 0.5f);
}

void launch_resize_area2x(const unsigned char* d_rgb_in, unsigned char* d_rgb_out, int Wf, int Hf)
{
    const int Wr = Wf / kResizeFactor, Hr = Hf / kResizeFactor;
    resize_area2x_kernel<<<grid2d(Wr, Hr), dim3(kBlock2D, kBlock2D)>>>(d_rgb_in, d_rgb_out, Wf, Hf);
    CUDA_CHECK_LAST_ERROR("resize_area2x_kernel launch");
}

// ===========================================================================
// 5) FUSED UNDISTORT+RECTIFY+RESIZE — the centerpiece kernel. One thread
// per RESIZED output pixel (same thread count as resize_area2x_kernel, NOT
// the same as remap_bilinear_kernel — this is a 4x coarser grid than the
// staged remap). For each of the kResizeFactor^2 full-resolution
// sub-pixels that a staged pipeline would have materialized and then
// re-read, this kernel looks up the LUT and bilinear-samples the
// debayered image DIRECTLY INTO REGISTERS, accumulates, and writes the
// average ONCE. The intermediate full-resolution remapped image never
// exists in memory — not in global memory, not even transiently; each
// sample lives only as long as the local `float rgb[3]` below.
//
// What this buys (THEORY.md "The GPU mapping" derives the general
// formula; here is the concrete count at this project's WxH):
//   STAGED remap+resize moves  18.75 * W*H  bytes (read+write, idealized
//     no-cache-reuse model — see the note below); FUSED moves 12.75 * W*H.
//   The 6*W*H byte gap is EXACTLY the staged path's intermediate full-res
//   image: written once by remap_bilinear_kernel (3*W*H bytes) and read
//   once by resize_area2x_kernel (3*W*H bytes) — a round trip through
//   global memory that produces no new information, since the very next
//   kernel only ever reads each of those bytes once. Fusing the two
//   kernels deletes exactly that round trip; main.cu prints the derived
//   byte counts and the measured kernel-time comparison side by side.
//
// Honesty note (also in THEORY.md): the byte counts above assume NO
// L2/texture-cache reuse between adjacent threads' overlapping bilinear
// footprints — a worst-case, easy-to-derive-by-hand model. Real hardware
// caches aggressively (adjacent output pixels' bilinear samples overlap
// heavily), so the MEASURED kernel-time gap is usually smaller than the
// idealized byte-count gap predicts; both numbers are printed so the
// difference between "bytes we asked the memory system for" and "time it
// took" is itself visible.
//
// Numerics note: because the four sub-samples are averaged in FLOAT
// (registers) before the ONE rounding-to-uint8 at the very end, this
// kernel rounds once per output pixel, where the staged path rounds TWICE
// (once in remap_bilinear_kernel, once in resize_area2x_kernel) — a real,
// small, and expected difference between the fused and staged results
// (main.cu's fused-vs-staged gate documents and bounds it; THEORY.md
// "Numerical considerations" explains why double rounding is never
// IDENTICAL to single rounding).
// ===========================================================================
__global__ void fused_kernel(const unsigned char* __restrict__ rgb_in,
                             const RemapSample* __restrict__ lut_fullres,
                             unsigned char* __restrict__ rgb_out, int Wf, int Hf)
{
    const int Wr = Wf / kResizeFactor, Hr = Hf / kResizeFactor;
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= Wr || yo >= Hr) return;

    float acc[3] = { 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int dy = 0; dy < kResizeFactor; ++dy) {
#pragma unroll
        for (int dx = 0; dx < kResizeFactor; ++dx) {
            const int xf = xo * kResizeFactor + dx;          // full-res sub-pixel column
            const int yf = yo * kResizeFactor + dy;          // full-res sub-pixel row
            const RemapSample s = lut_fullres[yf * Wf + xf]; // reused LUT — see file header
            float rgb[3];
            bilinear_sample_rgb(rgb_in, Wf, Hf, s.u, s.v, rgb);   // sample straight from the debayered image
#pragma unroll
            for (int c = 0; c < 3; ++c) acc[c] += rgb[c];
        }
    }
    const float norm = 1.0f / static_cast<float>(kResizeFactor * kResizeFactor);
    const int oo = (yo * Wr + xo) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c)
        rgb_out[oo + c] = static_cast<unsigned char>(acc[c] * norm + 0.5f);   // ONE rounding step (see header note)
}

void launch_fused_undistort_rectify_resize(const unsigned char* d_rgb_in,
                                           const RemapSample* d_lut_fullres,
                                           unsigned char* d_rgb_out, int Wf, int Hf)
{
    const int Wr = Wf / kResizeFactor, Hr = Hf / kResizeFactor;
    fused_kernel<<<grid2d(Wr, Hr), dim3(kBlock2D, kBlock2D)>>>(d_rgb_in, d_lut_fullres, d_rgb_out, Wf, Hf);
    CUDA_CHECK_LAST_ERROR("fused_kernel launch");
}

// ===========================================================================
// 6) NORMALIZE — a three-kernel, DETERMINISTIC (no atomics) two-pass
// mean/std reduction over the resized image, per channel.
//
// The determinism choice (THEORY.md "Numerical considerations" expands
// this; CLAUDE.md paragraph 12 asks every project to state it explicitly):
// the FASTEST way to sum millions of numbers on a GPU is usually
// atomicAdd from every thread straight into one global accumulator — but
// atomicAdd's arrival order is whatever the scheduler happens to produce,
// so the bit pattern of a float sum built that way can differ between two
// runs of the IDENTICAL kernel on the IDENTICAL input (floating-point
// addition is not associative: (a+b)+c != a+(b+c) in general). This
// project instead uses a FIXED two-level tree: (a) each block reduces its
// own slice with an in-block shared-memory tree (a fixed, deterministic
// binary reduction order, identical every run because block/thread
// scheduling never changes the WITHIN-block reduction structure), then
// (b) a single thread sums the resulting (few hundred) block partials in
// a fixed sequential order. No atomic instruction appears anywhere in
// this pipeline — the price is one tiny, cheap extra kernel launch
// (launch_normalize_finalize); the payoff is a bit-reproducible result
// run after run on the same GPU, which is what lets main.cu's normalize
// gate assert an exact-ish tolerance instead of a statistical one.
// ===========================================================================

// 6a) Per-block partial sums. One thread per pixel; shared memory holds 6
// parallel reduction lanes (sum_r, sum_g, sum_b, sumsq_r, sumsq_g,
// sumsq_b), each blockDim.x doubles wide, laid out back to back:
// sdata[lane*blockDim.x + tid]. Accumulating in DOUBLE (not float) even
// though the source pixels are uint8_t: summing ~28,000 terms (this
// project's resized image, 192x144) in float32 loses real precision
// (float32 has ~7 decimal digits; the running sum-of-squares term can
// reach the millions, leaving only 2-3 digits of headroom for the next
// addend) — THEORY.md works this bound out in full.
__global__ void normalize_block_stats_kernel(const unsigned char* __restrict__ rgb,
                                              int n_pixels,
                                              double* __restrict__ block_sum3,
                                              double* __restrict__ block_sumsq3)
{
    extern __shared__ double sdata[];              // size = 6 * blockDim.x doubles (see launcher)
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;    // this thread's pixel index, may be >= n_pixels (tail)
    const int bd = blockDim.x;

    double r = 0.0, g = 0.0, b = 0.0;
    if (i < n_pixels) {
        r = static_cast<double>(rgb[i * 3 + 0]);
        g = static_cast<double>(rgb[i * 3 + 1]);
        b = static_cast<double>(rgb[i * 3 + 2]);
    }
    // Out-of-range threads (the ragged last block) contribute exactly 0 —
    // the tree reduction below sums ALL blockDim.x lanes unconditionally,
    // so a padded-with-zero tail is simpler and just as correct as an
    // extra branch in the loop below would be.
    sdata[0 * bd + tid] = r;
    sdata[1 * bd + tid] = g;
    sdata[2 * bd + tid] = b;
    sdata[3 * bd + tid] = r * r;
    sdata[4 * bd + tid] = g * g;
    sdata[5 * bd + tid] = b * b;
    __syncthreads();

    // Binary tree reduction: at each step, the first `stride` threads add
    // the partner `stride` away into their own slot; the ACTIVE thread
    // count halves every step, and thread 0 ends up holding the total —
    // the textbook shared-memory reduction, run once per lane (6x), same
    // stride schedule every launch (no data-dependent branching), which is
    // exactly the "fixed order" this kernel's determinism claim rests on.
    for (int stride = bd / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int lane = 0; lane < 6; ++lane)
                sdata[lane * bd + tid] += sdata[lane * bd + tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int blk = blockIdx.x;
        block_sum3[blk * 3 + 0]   = sdata[0 * bd];
        block_sum3[blk * 3 + 1]   = sdata[1 * bd];
        block_sum3[blk * 3 + 2]   = sdata[2 * bd];
        block_sumsq3[blk * 3 + 0] = sdata[3 * bd];
        block_sumsq3[blk * 3 + 1] = sdata[4 * bd];
        block_sumsq3[blk * 3 + 2] = sdata[5 * bd];
    }
}

void launch_normalize_block_stats(const unsigned char* d_rgb, int W, int H,
                                  double* d_block_sum3, double* d_block_sumsq3,
                                  int num_blocks)
{
    const int n_pixels = W * H;
    const size_t shmem_bytes = static_cast<size_t>(6) * kNormBlockSize * sizeof(double);
    normalize_block_stats_kernel<<<num_blocks, kNormBlockSize, shmem_bytes>>>(
        d_rgb, n_pixels, d_block_sum3, d_block_sumsq3);
    CUDA_CHECK_LAST_ERROR("normalize_block_stats_kernel launch");
}

// 6b) Finalize — <<<1,1>>>: exactly one thread, one block. This is
// deliberately NOT parallelized further: num_blocks is small (169 at this
// project's resized resolution — see main.cu), so a plain O(num_blocks)
// sequential loop costs microseconds, and keeping it single-threaded means
// there is only ONE possible summation order, period — no reduction tree
// to reason about at this stage at all. This is the second half of the
// "no atomics anywhere" determinism story (see the section 6 header).
__global__ void normalize_finalize_kernel(const double* __restrict__ block_sum3,
                                          const double* __restrict__ block_sumsq3,
                                          int num_blocks, long long n_pixels,
                                          float* __restrict__ mean3, float* __restrict__ std3)
{
    double sum[3] = { 0.0, 0.0, 0.0 };
    double sumsq[3] = { 0.0, 0.0, 0.0 };
    for (int blk = 0; blk < num_blocks; ++blk) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            sum[c]   += block_sum3[blk * 3 + c];
            sumsq[c] += block_sumsq3[blk * 3 + c];
        }
    }
    const double n = static_cast<double>(n_pixels);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const double mean = sum[c] / n;
        // Population variance E[x^2] - E[x]^2 (dividing by N, not N-1: we
        // want the moments of THIS image's pixel population, not an
        // unbiased estimate of some larger population it was drawn from —
        // THEORY.md contrasts this with a sample-variance use case).
        double var = sumsq[c] / n - mean * mean;
        if (var < static_cast<double>(kNormEps)) var = static_cast<double>(kNormEps);  // guard a flat channel
        mean3[c] = static_cast<float>(mean);
        std3[c]  = static_cast<float>(sqrt(var));
    }
}

void launch_normalize_finalize(const double* d_block_sum3, const double* d_block_sumsq3,
                               int num_blocks, long long n_pixels,
                               float* d_mean3, float* d_std3)
{
    normalize_finalize_kernel<<<1, 1>>>(d_block_sum3, d_block_sumsq3, num_blocks, n_pixels, d_mean3, d_std3);
    CUDA_CHECK_LAST_ERROR("normalize_finalize_kernel launch");
}

// 6c) Apply — one thread per pixel, the per-channel affine map. Reads the
// 3 mean/3 std values (already resident in device memory from 6b) once
// per thread; this is a MAP again, no reduction, back to full parallelism.
__global__ void normalize_apply_kernel(const unsigned char* __restrict__ rgb, float* __restrict__ out,
                                       int n_pixels, const float* __restrict__ mean3,
                                       const float* __restrict__ std3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
#pragma unroll
    for (int c = 0; c < 3; ++c)
        out[i * 3 + c] = (static_cast<float>(rgb[i * 3 + c]) - mean3[c]) / std3[c];
}

void launch_normalize_apply(const unsigned char* d_rgb, float* d_out, int W, int H,
                            const float* d_mean3, const float* d_std3)
{
    const int n_pixels = W * H;
    const int blocks = (n_pixels + kNormBlockSize - 1) / kNormBlockSize;
    normalize_apply_kernel<<<blocks, kNormBlockSize>>>(d_rgb, d_out, n_pixels, d_mean3, d_std3);
    CUDA_CHECK_LAST_ERROR("normalize_apply_kernel launch");
}
