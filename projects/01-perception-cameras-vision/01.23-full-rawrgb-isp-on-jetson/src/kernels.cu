// ===========================================================================
// kernels.cu — GPU kernels for project 01.23
//              Full RAW->RGB ISP: black level -> lens shading -> defect
//              correction -> white balance -> demosaic (MHC + bilinear) ->
//              CCM -> gamma, staged AND fused (stages 1-4)
//
// Every kernel here is a MAP (one thread, one output element) EXCEPT the AWB
// statistics kernel (a REDUCE, 01.01's deterministic block-tree pattern).
// The defect-correction and demosaic kernels are STENCILS (a thread reads a
// handful of neighbors, not just its own pixel) — kernels.cuh's file header
// walks the eight-stage pipeline; this file is where each stage's threads-
// to-data mapping and memory behavior are explained.
//
// Read this after: kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>

// ---------------------------------------------------------------------------
// Launch geometry shared by every "one thread per raw pixel" kernel in this
// file: a 1-D grid over the W*H mosaic (this project's images are small
// enough — kRawW*kRawH = 19,200 — that a 1-D index i=y*W+x with a single
// ceil-divide is simpler to read than a 2-D block, and every kernel below
// needs (x,y) anyway for bayer_phase_at()/shading_gain_at(), recovered by
// one div/mod). block=256: a warp multiple, the repo's standard default.
// ---------------------------------------------------------------------------
static constexpr int kBlock1D = 256;
static inline int grid1d(int n) { return (n + kBlock1D - 1) / kBlock1D; }

// clampi — clamp an integer index into [lo, hi]. __device__-only; the CPU
// twin in reference_cpu.cpp defines its own independent host-side copy
// (same two-line-function judgment call as 01.01's clampi/clampi_cpu split).
__device__ inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ===========================================================================
// STAGE 1 — BLACK LEVEL + saturation handling. Every photosite reads a
// nonzero DN even with zero light (dark current + ADC offset, kBlackLevel);
// subtracting it and dividing by the usable code range kSatRange maps raw
// DN -> a normalized [0,1] "sensor domain" float. "Saturation handling"
// here means clamping: a raw code below the black level (noise can push a
// dark pixel's ADC reading under its own offset) floors at 0 rather than
// going negative; kSatRange's own definition already caps the top end at
// exactly 1.0 for a raw value at kWhiteLevel, so the upper clamp below is a
// defensive no-op on well-formed input, not a load-bearing branch.
//
// Thread-to-data mapping: thread i owns raw pixel (i%W, i/W) — a pure MAP,
// no neighbor reads, so this is the cheapest possible kernel in the file
// (2 flops, 1 load, 1 store) and the natural "fold this into anything"
// candidate the FUSED kernel exploits.
// ===========================================================================
__global__ void black_level_kernel(const uint16_t* __restrict__ raw,
                                   float* __restrict__ out, int W, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int r = static_cast<int>(raw[i]);
    const float above_black = r > kBlackLevel ? static_cast<float>(r - kBlackLevel) : 0.0f;
    float norm = above_black / static_cast<float>(kSatRange);
    norm = fminf(norm, 1.0f);           // defensive clamp — see header note
    out[i] = norm;
}
void launch_black_level(const uint16_t* d_raw, float* d_out, int W, int H)
{
    const int n = W * H;
    black_level_kernel<<<grid1d(n), kBlock1D>>>(d_raw, d_out, W, H);
    CUDA_CHECK_LAST_ERROR("black_level_kernel launch");
}

// ===========================================================================
// STAGE 2 — LENS SHADING correction. Divides by the same radial polynomial
// V(r) the synthetic sensor multiplied by (kernels.cuh's shading_gain_at,
// the shared "hardware fact"), floored at kShadeGainFloor to guard the
// division (01.09's precedent — inactive at this project's chosen a2/a4,
// see kernels.cuh, but present because a correction stage that can silently
// divide by a near-zero gain is a real, reported failure mode worth
// guarding explicitly). Still a pure per-pixel MAP: shading_gain_at(x,y) is
// a closed-form function of the pixel's OWN coordinates, no neighbor reads.
// ===========================================================================
__global__ void lens_shading_kernel(const float* __restrict__ in,
                                    float* __restrict__ out, int W, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;
    const float gain = fmaxf(shading_gain_at(x, y), kShadeGainFloor);
    out[i] = in[i] / gain;
}
void launch_lens_shading(const float* d_in, float* d_out, int W, int H)
{
    const int n = W * H;
    lens_shading_kernel<<<grid1d(n), kBlock1D>>>(d_in, d_out, W, H);
    CUDA_CHECK_LAST_ERROR("lens_shading_kernel launch");
}

// ===========================================================================
// STAGE 3 — DEFECTIVE PIXEL correction. A real sensor's factory defect map
// is tiny (a few dozen photosites out of millions); this project's
// committed list (data/sample/defect_list.csv, kMaxDefects=64 slots) is
// broadcast to every thread via __constant__ memory (g_defect_x/g_defect_y
// below) — the textbook use case for CUDA's constant cache: every thread in
// the grid reads the IDENTICAL small array, so the cache serves the whole
// warp (indeed the whole grid) from one broadcast read instead of thrashing
// global-memory bandwidth on 19,200 redundant small reads.
//
// median4 — a FIXED 5-compare sorting network (not std::sort — GPU code
// avoids recursion/library calls in hot per-thread paths; a hand-unrolled
// network is deterministic, branch-shallow, and easy to verify by hand: see
// THEORY.md for a worked trace) that fully sorts 4 floats; the corrected
// value is the average of the two middle ("median") elements — the
// standard median-of-4 substitute (4 is not odd, so "the median" is this
// average by convention, same as many production defect-correction ISPs).
// ---------------------------------------------------------------------------
__device__ inline float median4(float a, float b, float c, float d)
{
    float t;
    if (a > b) { t = a; a = b; b = t; }
    if (c > d) { t = c; c = d; d = t; }
    if (a > c) { t = a; a = c; c = t; }
    if (b > d) { t = b; b = d; d = t; }
    if (b > c) { t = b; b = c; c = t; }
    return 0.5f * (b + c);
}

// Device-side defect list storage. `static` gives this translation unit its
// own private copy — see kernels.cuh section 4's header note: a
// __constant__ array declared in a header included by TWO nvcc translation
// units (main.cu and kernels.cu both compile as .cu files) would otherwise
// risk a duplicate-symbol link error; `static` sidesteps that by giving
// each TU internal linkage. Only kernels.cu's kernels dereference these —
// main.cu only ever calls the host wrapper upload_defect_list() below.
static __constant__ int g_defect_x[kMaxDefects];
static __constant__ int g_defect_y[kMaxDefects];

void upload_defect_list(const int* xs, const int* ys, int count)
{
    // cudaMemcpyToSymbol targets a __constant__/__device__ symbol by NAME
    // (compile-time resolved), unlike cudaMemcpy which takes a runtime
    // device pointer — the standard way to populate constant memory from
    // host-loaded calibration data (PRACTICE.md section 1 draws the real-
    // sensor parallel: a factory defect map loaded from calibration EEPROM
    // at boot, then held in fast on-chip constant storage for the ISP's
    // lifetime).
    if (count > kMaxDefects) count = kMaxDefects;   // defensive: never overrun the fixed slots
    CUDA_CHECK(cudaMemcpyToSymbol(g_defect_x, xs, static_cast<size_t>(count) * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_defect_y, ys, static_cast<size_t>(count) * sizeof(int)));
}

// Thread-to-data mapping: thread i owns raw pixel (x,y). EVERY thread scans
// the (tiny, <=64-entry) defect list once — 19,200 threads x <=64 compares
// = <=1.2M integer comparisons total, a rounding error next to the rest of
// the pipeline's cost, and far simpler than building a full-resolution
// boolean defect MASK just to avoid it. Non-defect threads pay ONLY that
// scan (no neighbor reads at all) — the common case is a pure MAP; only the
// rare defective pixel (~0.08% of this project's image) pays the STENCIL
// cost of four same-phase neighbor reads.
__global__ void defect_correct_kernel(const float* __restrict__ in,
                                      float* __restrict__ out, int W, int H,
                                      int defect_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    bool is_defect = false;
    #pragma unroll 4
    for (int k = 0; k < defect_count; ++k) {
        if (g_defect_x[k] == x && g_defect_y[k] == y) { is_defect = true; break; }
    }

    if (!is_defect) { out[i] = in[i]; return; }

    // Same-Bayer-phase neighbors sit exactly 2 pixels away in each axis
    // (RGGB repeats every 2 pixels) — the four orthogonal same-phase
    // neighbors (N,S,E,W at distance 2), border-clamped like every other
    // neighbor read in this repo's image kernels.
    const int xm = clampi(x - 2, 0, W - 1), xp = clampi(x + 2, 0, W - 1);
    const int ym = clampi(y - 2, 0, H - 1), yp = clampi(y + 2, 0, H - 1);
    const float n_ = in[ym * W + x];
    const float s_ = in[yp * W + x];
    const float e_ = in[y * W + xp];
    const float w_ = in[y * W + xm];
    out[i] = median4(n_, s_, e_, w_);
}
void launch_defect_correct(const float* d_in, float* d_out, int W, int H, int defect_count)
{
    const int n = W * H;
    defect_correct_kernel<<<grid1d(n), kBlock1D>>>(d_in, d_out, W, H, defect_count);
    CUDA_CHECK_LAST_ERROR("defect_correct_kernel launch");
}

// ===========================================================================
// STAGE 4 — WHITE BALANCE. Per-Bayer-phase gain multiply — a pure MAP, the
// simplest stage in the pipeline (the AWB *estimation* that produces
// gain_r/gain_g/gain_b is the interesting part, and it lives in the
// reduction kernels below; this kernel only *applies* gains someone else
// computed). Applying WB pre-demosaic (on the raw mosaic, not after) is the
// realistic choice production ISPs make too — see THEORY.md "Where this
// sits in the real world".
// ===========================================================================
__global__ void white_balance_kernel(const float* __restrict__ in,
                                     float* __restrict__ out, int W, int H,
                                     float gain_r, float gain_g, float gain_b)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;
    const int wbch = phase_to_wb_channel(bayer_phase_at(x, y));
    const float gain = wbch == 0 ? gain_r : (wbch == 2 ? gain_b : gain_g);
    out[i] = in[i] * gain;
}
void launch_white_balance(const float* d_in, float* d_out, int W, int H,
                          float gain_r, float gain_g, float gain_b)
{
    const int n = W * H;
    white_balance_kernel<<<grid1d(n), kBlock1D>>>(d_in, d_out, W, H, gain_r, gain_g, gain_b);
    CUDA_CHECK_LAST_ERROR("white_balance_kernel launch");
}

// ===========================================================================
// FUSED stages 1-4 — ONE kernel, one thread per raw pixel (kernels.cuh's
// file header "Fusion economics" derives why this fusion is unusually
// cheap). bl_shading_at() recomputes black-level+shading for ANY (x,y) from
// the RAW input directly — used for the thread's own pixel always, and for
// up to four same-phase NEIGHBORS only on the rare defective pixel. Because
// bl_shading_at is a pure function of raw[] and (x,y) (no dependency on any
// other kernel's output), recomputing it inline is exactly as correct as
// reading a materialized intermediate buffer would have been — the STAGED
// path's black_level_kernel + lens_shading_kernel outputs are literally
// what this function reproduces on demand, just never written to global
// memory. White balance is applied LAST (after the possible median), which
// is safe because all same-phase neighbors of a given thread share ONE
// gain: multiplying by a positive scalar commutes with taking a median (a
// order-statistic), so "median then scale" and "scale then median" give the
// identical result — main.cu's fused_vs_staged gate is the numeric proof.
// ===========================================================================
__device__ inline float bl_shading_at(const uint16_t* __restrict__ raw, int x, int y, int W, int H)
{
    x = clampi(x, 0, W - 1);
    y = clampi(y, 0, H - 1);
    const int r = static_cast<int>(raw[y * W + x]);
    const float above_black = r > kBlackLevel ? static_cast<float>(r - kBlackLevel) : 0.0f;
    float norm = fminf(above_black / static_cast<float>(kSatRange), 1.0f);
    const float gain = fmaxf(shading_gain_at(x, y), kShadeGainFloor);
    return norm / gain;
}

__global__ void fused_bl_shading_defect_wb_kernel(const uint16_t* __restrict__ raw,
                                                   float* __restrict__ out, int W, int H,
                                                   int defect_count,
                                                   float gain_r, float gain_g, float gain_b)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    bool is_defect = false;
    #pragma unroll 4
    for (int k = 0; k < defect_count; ++k) {
        if (g_defect_x[k] == x && g_defect_y[k] == y) { is_defect = true; break; }
    }

    float bl_sh;   // this pixel's black-level+shading-corrected value
    if (!is_defect) {
        bl_sh = bl_shading_at(raw, x, y, W, H);
    } else {
        const float n_ = bl_shading_at(raw, x, y - 2, W, H);
        const float s_ = bl_shading_at(raw, x, y + 2, W, H);
        const float e_ = bl_shading_at(raw, x + 2, y, W, H);
        const float w_ = bl_shading_at(raw, x - 2, y, W, H);
        bl_sh = median4(n_, s_, e_, w_);
    }

    const int wbch = phase_to_wb_channel(bayer_phase_at(x, y));
    const float gain = wbch == 0 ? gain_r : (wbch == 2 ? gain_b : gain_g);
    out[i] = bl_sh * gain;
}
void launch_fused_bl_shading_defect_wb(const uint16_t* d_raw, float* d_out, int W, int H,
                                       int defect_count, float gain_r, float gain_g, float gain_b)
{
    const int n = W * H;
    fused_bl_shading_defect_wb_kernel<<<grid1d(n), kBlock1D>>>(
        d_raw, d_out, W, H, defect_count, gain_r, gain_g, gain_b);
    CUDA_CHECK_LAST_ERROR("fused_bl_shading_defect_wb_kernel launch");
}

// ===========================================================================
// AWB STATISTICS — a DETERMINISTIC (no atomics) two-level block-tree
// reduction, extending 01.01's normalize_block_stats pattern from 6
// same-operator (sum) lanes to 6 MIXED-operator lanes: 3 SUM lanes (R,G,B —
// feed gray-world's "the average of a well-lit scene is gray" estimator)
// and 3 MAX lanes (R,G,B — feed white-patch's "the brightest pixel is a
// white/specular highlight" estimator). Both trees share ONE launch and one
// pass over the data — computing both estimators is barely more expensive
// than computing one, a real GPU-reduction lesson (the memory read is the
// bottleneck, not the handful of extra FLOPs per thread).
//
// Shared-memory layout: this kernel needs BOTH a double (sum) tree and a
// float (max) tree; CUDA gives a kernel exactly one `extern __shared__`
// array, so the two live back to back in one raw byte buffer, cast to the
// right type at the right offset (the launcher below sizes the allocation
// to fit both; see its comment for the exact byte count).
// ===========================================================================
__global__ void awb_stats_block_kernel(const float* __restrict__ in, int n_pixels, int W,
                                       double* __restrict__ block_sum3,
                                       float* __restrict__ block_max3)
{
    extern __shared__ unsigned char smem_raw[];
    double* ssum = reinterpret_cast<double*>(smem_raw);                       // 3*blockDim.x doubles
    float*  smax = reinterpret_cast<float*>(ssum + 3 * blockDim.x);           // 3*blockDim.x floats

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const int bd = blockDim.x;

    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;
    float  r_max = 0.0f, g_max = 0.0f, b_max = 0.0f;   // 0.0 is a safe neutral: all inputs are >= 0 (stage 1's clamp)
    if (i < n_pixels) {
        const int x = i % W, y = i / W;
        const int wbch = phase_to_wb_channel(bayer_phase_at(x, y));
        const float v = in[i];
        if (wbch == 0) { r_sum = static_cast<double>(v); r_max = v; }
        else if (wbch == 2) { b_sum = static_cast<double>(v); b_max = v; }
        else { g_sum = static_cast<double>(v); g_max = v; }
    }
    ssum[0 * bd + tid] = r_sum; ssum[1 * bd + tid] = g_sum; ssum[2 * bd + tid] = b_sum;
    smax[0 * bd + tid] = r_max; smax[1 * bd + tid] = g_max; smax[2 * bd + tid] = b_max;
    __syncthreads();

    // Same fixed binary-tree schedule as 01.01's normalize reduction (the
    // determinism argument is identical: stride halves every step, the
    // WITHIN-block order never depends on scheduling) — here applied to two
    // trees with two different combining operators (+= for sum, fmaxf for max).
    for (int stride = bd / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int lane = 0; lane < 3; ++lane) {
                ssum[lane * bd + tid] += ssum[lane * bd + tid + stride];
                smax[lane * bd + tid] = fmaxf(smax[lane * bd + tid], smax[lane * bd + tid + stride]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int blk = blockIdx.x;
        block_sum3[blk * 3 + 0] = ssum[0 * bd]; block_sum3[blk * 3 + 1] = ssum[1 * bd]; block_sum3[blk * 3 + 2] = ssum[2 * bd];
        block_max3[blk * 3 + 0] = smax[0 * bd]; block_max3[blk * 3 + 1] = smax[1 * bd]; block_max3[blk * 3 + 2] = smax[2 * bd];
    }
}
void launch_awb_stats_block(const float* d_in, int W, int H,
                            double* d_block_sum3, float* d_block_max3, int num_blocks)
{
    const int n_pixels = W * H;
    // Shared bytes = 3*block doubles (sum tree) + 3*block floats (max tree).
    const size_t shmem_bytes = static_cast<size_t>(3) * kBlock1D * sizeof(double)
                              + static_cast<size_t>(3) * kBlock1D * sizeof(float);
    awb_stats_block_kernel<<<num_blocks, kBlock1D, shmem_bytes>>>(d_in, n_pixels, W, d_block_sum3, d_block_max3);
    CUDA_CHECK_LAST_ERROR("awb_stats_block_kernel launch");
}

// awb_finalize — <<<1,1>>>, sequential over the (small) block partials, the
// second half of the "no atomics anywhere" determinism story. Per-phase
// pixel COUNTS are DERIVED from W*H, not reduced (kernels.cuh's comment on
// launch_awb_stats_block explains why: RGGB geometry fixes them exactly).
// Gray-world gain[c] = mean(G) / mean(c); white-patch gain[c] = max(G) /
// max(c) — both conventions keep the GREEN channel at unit gain (the
// standard camera-ISP convention, since green carries most of scene
// luminance and demosaic/CCM both treat it as the "reference" channel).
__global__ void awb_finalize_kernel(const double* __restrict__ block_sum3,
                                    const float* __restrict__ block_max3,
                                    int num_blocks, int W, int H,
                                    float* __restrict__ gray_gain3,
                                    float* __restrict__ white_gain3)
{
    double sum[3] = { 0.0, 0.0, 0.0 };
    float mx[3] = { 0.0f, 0.0f, 0.0f };
    for (int blk = 0; blk < num_blocks; ++blk) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            sum[c] += block_sum3[blk * 3 + c];
            mx[c] = fmaxf(mx[c], block_max3[blk * 3 + c]);
        }
    }
    const double countR = static_cast<double>(W) * H / 4.0;   // RGGB: R = B = W*H/4, G = W*H/2
    const double countG = static_cast<double>(W) * H / 2.0;
    const double countB = countR;
    const double meanR = sum[0] / countR, meanG = sum[1] / countG, meanB = sum[2] / countB;

    gray_gain3[1] = 1.0f;
    gray_gain3[0] = static_cast<float>(meanG / (meanR > 1e-8 ? meanR : 1e-8));
    gray_gain3[2] = static_cast<float>(meanG / (meanB > 1e-8 ? meanB : 1e-8));

    white_gain3[1] = 1.0f;
    white_gain3[0] = mx[1] / (mx[0] > 1e-6f ? mx[0] : 1e-6f);
    white_gain3[2] = mx[1] / (mx[2] > 1e-6f ? mx[2] : 1e-6f);
}
void launch_awb_finalize(const double* d_block_sum3, const float* d_block_max3,
                         int num_blocks, int W, int H,
                         float* d_gray_gain3, float* d_white_gain3)
{
    awb_finalize_kernel<<<1, 1>>>(d_block_sum3, d_block_max3, num_blocks, W, H, d_gray_gain3, d_white_gain3);
    CUDA_CHECK_LAST_ERROR("awb_finalize_kernel launch");
}

// ===========================================================================
// STAGE 5 — DEMOSAIC. Two independent kernels operating on the SAME
// white-balanced mosaic, so their outputs are directly comparable
// (main.cu's demosaic_psnr gate).
// ===========================================================================

// ---- 5a) Bilinear baseline — the same four-case algorithm as sibling
// flagship 01.01's debayer_kernel (distance-1 orthogonal/diagonal averages),
// independently retyped here for this project's float mosaic layout and
// 4-way phase split (01.01 operates on uint8 and a 3-way phase; the
// arithmetic pattern is the shared LINEAGE this project cites, the code is
// its own). Kept as this project's quality FLOOR: no gradient correction,
// so color fringing at edges is worse than MHC's — measuring exactly how
// much worse is this project's headline number (README "Expected output").
__global__ void demosaic_bilinear_kernel(const float* __restrict__ mosaic,
                                         float* __restrict__ rgb, int W, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;
    const int xm = clampi(x - 1, 0, W - 1), xp = clampi(x + 1, 0, W - 1);
    const int ym = clampi(y - 1, 0, H - 1), yp = clampi(y + 1, 0, H - 1);
    const float n_ = mosaic[ym * W + x], s_ = mosaic[yp * W + x];
    const float e_ = mosaic[y * W + xp], w_ = mosaic[y * W + xm];
    const float ne = mosaic[ym * W + xp], nw = mosaic[ym * W + xm];
    const float se = mosaic[yp * W + xp], sw = mosaic[yp * W + xm];
    const float center = mosaic[i];

    float R, G, B;
    const int phase = bayer_phase_at(x, y);
    if (phase == 0) {                                  // R site
        R = center; G = 0.25f * (n_ + s_ + e_ + w_); B = 0.25f * (ne + nw + se + sw);
    } else if (phase == 3) {                            // B site
        B = center; G = 0.25f * (n_ + s_ + e_ + w_); R = 0.25f * (ne + nw + se + sw);
    } else if (phase == 1) {                            // Gr site: R is horizontal, B is vertical
        G = center; R = 0.5f * (e_ + w_); B = 0.5f * (n_ + s_);
    } else {                                            // Gb site: B is horizontal, R is vertical
        G = center; B = 0.5f * (e_ + w_); R = 0.5f * (n_ + s_);
    }
    rgb[i * 3 + 0] = R; rgb[i * 3 + 1] = G; rgb[i * 3 + 2] = B;
}
void launch_demosaic_bilinear(const float* d_mosaic, float* d_rgb, int W, int H)
{
    const int n = W * H;
    demosaic_bilinear_kernel<<<grid1d(n), kBlock1D>>>(d_mosaic, d_rgb, W, H);
    CUDA_CHECK_LAST_ERROR("demosaic_bilinear_kernel launch");
}

// ---- 5b) Malvar-He-Cutler — the project's centerpiece kernel. Device-side
// __constant__ copies of kernels.cuh's four coefficient tables (see that
// header's section 3 for why these need their own device storage; `static`
// for the same per-translation-unit-linkage reason as g_defect_x/y above).
// Values are IDENTICAL to kernels.cuh's kMhcG/kMhcA/kMhcB/kMhcDiag — kept
// side by side here for a reader to diff by eye.
static __constant__ float d_kMhcG[kMhcTaps] = {
     0,  0, -1,  0,  0,
     0,  0,  2,  0,  0,
    -1,  2,  4,  2, -1,
     0,  0,  2,  0,  0,
     0,  0, -1,  0,  0,
};
static __constant__ float d_kMhcA[kMhcTaps] = {
     0,    0,   0.5f, 0,    0,
     0,   -1,   0,   -1,    0,
    -1,    4,   5,    4,   -1,
     0,   -1,   0,   -1,    0,
     0,    0,   0.5f, 0,    0,
};
static __constant__ float d_kMhcB[kMhcTaps] = {
     0,    0,  -1,   0,    0,
     0,   -1,   4,  -1,    0,
     0.5f, 0,   5,   0,    0.5f,
     0,   -1,   4,  -1,    0,
     0,    0,  -1,   0,    0,
};
static __constant__ float d_kMhcDiag[kMhcTaps] = {
     0,     0,   -1.5f, 0,     0,
     0,     2,    0,    2,     0,
    -1.5f,  0,    6,    0,    -1.5f,
     0,     2,    0,    2,     0,
     0,     0,   -1.5f, 0,     0,
};

// mhc_eval — apply one 5x5 MHC table centered at (x,y) against the RAW
// MOSAIC (single sample per pixel; NOT a per-channel-separated image — the
// whole point of MHC is that the raw mosaic's own Bayer periodicity puts
// the right same-phase and cross-phase samples at the right offsets
// automatically, kernels.cuh's section 3 comment walks this). #pragma
// unroll: the 5x5 loop bound is a compile-time constant, so the compiler
// fully unrolls it into 25 FMA-friendly straight-line instructions — no
// loop overhead, and the border clampi() calls become predictable branches
// the compiler can often convert to predicated moves.
__device__ inline float mhc_eval(const float* __restrict__ mosaic, int x, int y, int W, int H,
                                 const float* __restrict__ weights)
{
    float acc = 0.0f;
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            const float w = weights[(dy + 2) * 5 + (dx + 2)];
            if (w == 0.0f) continue;   // most of the 25 taps are exactly zero — skip the read+FMA
            const int nx = clampi(x + dx, 0, W - 1);
            const int ny = clampi(y + dy, 0, H - 1);
            acc += w * mosaic[ny * W + nx];
        }
    }
    return acc * 0.125f;   // / 8, the tables' documented normalization (kernels.cuh section 3)
}

// Thread-to-data mapping: thread i owns raw pixel (x,y) and writes ALL
// THREE output channels for it (the "1 sample in -> 3 samples out" stage,
// same growth 01.01's debayer names). Kernel SELECTION by phase mirrors
// kernels.cuh section 3's table exactly.
__global__ void demosaic_mhc_kernel(const float* __restrict__ mosaic,
                                    float* __restrict__ rgb, int W, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;
    const int phase = bayer_phase_at(x, y);
    const float native = mosaic[i];

    float R, G, B;
    if (phase == 0) {                                   // R site
        R = native;
        G = mhc_eval(mosaic, x, y, W, H, d_kMhcG);
        B = mhc_eval(mosaic, x, y, W, H, d_kMhcDiag);
    } else if (phase == 3) {                             // B site
        B = native;
        G = mhc_eval(mosaic, x, y, W, H, d_kMhcG);
        R = mhc_eval(mosaic, x, y, W, H, d_kMhcDiag);
    } else if (phase == 1) {                             // Gr site: R horizontal-emphasis, B vertical-emphasis
        G = native;
        R = mhc_eval(mosaic, x, y, W, H, d_kMhcA);
        B = mhc_eval(mosaic, x, y, W, H, d_kMhcB);
    } else {                                             // Gb site: B horizontal-emphasis, R vertical-emphasis
        G = native;
        B = mhc_eval(mosaic, x, y, W, H, d_kMhcA);
        R = mhc_eval(mosaic, x, y, W, H, d_kMhcB);
    }
    rgb[i * 3 + 0] = fmaxf(R, 0.0f);   // negative-lobe taps can undershoot near a hard edge; floor at 0
    rgb[i * 3 + 1] = fmaxf(G, 0.0f);   // (a real sensor signal cannot be negative — THEORY.md discusses
    rgb[i * 3 + 2] = fmaxf(B, 0.0f);   // this as MHC's one real failure mode, "ringing" near sharp edges)
}
void launch_demosaic_mhc(const float* d_mosaic, float* d_rgb, int W, int H)
{
    const int n = W * H;
    demosaic_mhc_kernel<<<grid1d(n), kBlock1D>>>(d_mosaic, d_rgb, W, H);
    CUDA_CHECK_LAST_ERROR("demosaic_mhc_kernel launch");
}

// ===========================================================================
// STAGE 6 — COLOR-CORRECTION MATRIX. A pure per-pixel MAP: 9 multiplies + 6
// adds per pixel, using the shared ccm_apply_at() (kernels.cuh) — no
// neighbor reads, the cheapest kind of kernel next to black-level.
// ===========================================================================
__global__ void ccm_apply_kernel(const float* __restrict__ rgb_in,
                                 float* __restrict__ rgb_out, int W, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    float or_, og, ob;
    ccm_apply_at(rgb_in[i * 3 + 0], rgb_in[i * 3 + 1], rgb_in[i * 3 + 2], or_, og, ob);
    rgb_out[i * 3 + 0] = or_; rgb_out[i * 3 + 1] = og; rgb_out[i * 3 + 2] = ob;
}
void launch_ccm_apply(const float* d_rgb_in, float* d_rgb_out, int W, int H)
{
    const int n = W * H;
    ccm_apply_kernel<<<grid1d(n), kBlock1D>>>(d_rgb_in, d_rgb_out, W, H);
    CUDA_CHECK_LAST_ERROR("ccm_apply_kernel launch");
}

// ===========================================================================
// STAGE 7 — GAMMA ENCODE. Linear float RGB (can be < 0 slightly from CCM's
// negative off-diagonal terms, or > 1 from WB/CCM gain — both real,
// expected headroom cases, THEORY.md "Numerical considerations" derives the
// bounds) -> clamp to [0,1] (the pipeline's highlight/shadow clip point,
// see kernels.cuh's srgb_encode) -> the exact sRGB piecewise transfer
// function -> round to 8-bit. The FINAL kernel in the pipeline; its output
// is the demo's headline artifact.
// ===========================================================================
__global__ void gamma_encode_kernel(const float* __restrict__ rgb_linear,
                                    unsigned char* __restrict__ rgb_srgb8, int W, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float s = srgb_encode(rgb_linear[i * 3 + c]) * 255.0f;
        rgb_srgb8[i * 3 + c] = static_cast<unsigned char>(fminf(fmaxf(s, 0.0f), 255.0f) + 0.5f);
    }
}
void launch_gamma_encode(const float* d_rgb_linear, unsigned char* d_rgb_srgb8, int W, int H)
{
    const int n = W * H;
    gamma_encode_kernel<<<grid1d(n), kBlock1D>>>(d_rgb_linear, d_rgb_srgb8, W, H);
    CUDA_CHECK_LAST_ERROR("gamma_encode_kernel launch");
}
