// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.01
//                     Full GPU image pipeline: debayer -> undistort ->
//                     rectify -> resize -> normalize
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5):
//
//   1) It is the CORRECTNESS ORACLE. GPU code fails in ways CPU code
//      cannot: wrong thread indexing, missed tail elements, race
//      conditions, stale device memory, bad transfers. A dead-simple
//      sequential version a reader can verify BY EYE gives us ground
//      truth; main.cu runs both and asserts element-wise agreement within
//      a documented tolerance.
//   2) It is the TEACHING BASELINE. Reading this file, then kernels.cu,
//      shows exactly what parallelization changed: every "for each output
//      pixel" loop below became "each thread owns one pixel" there — the
//      SAME five algorithms, one core vs. thousands of threads.
//
// Independence ruling applied to THIS file (the template's header states
// the general rule; here is exactly how it was applied):
//   * SHARED (kernels.cuh): bayer_channel_at(), distort_forward(),
//     compute_source_pixel() — the camera model / hardware-fact formulas.
//     These are DATA, in the sense the ruling means it (cf. 13.03's
//     dynamics-model precedent): re-typing the five-line Brown-Conrady
//     formula a second time would not exercise a different idea, only
//     risk a transcription slip that makes the oracle lie. main.cu's
//     "roundtrip" gate is the required INDEPENDENT check that does not
//     route through these functions (it re-derives distortion AND its
//     fixed-point inverse from scratch).
//   * INDEPENDENT (this file): the neighbor-clamping, the bilinear
//     sampling, the resize averaging, the fused-kernel re-derivation, and
//     the whole normalize reduction are all typed a SECOND time below,
//     from scratch, deliberately not calling anything in kernels.cu. Any
//     GPU-vs-CPU mismatch main.cu reports is therefore a real bug in one
//     of the two independent implementations, not a shared blind spot.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // RemapSample, camera-model constants + shared helpers, launcher signatures

#include <cmath>         // std::floor, std::sqrt

// ---------------------------------------------------------------------------
// clampi_cpu — host-side twin of kernels.cu's __device__ clampi(). Two
// lines; independently re-typed on purpose (see file header).
// ---------------------------------------------------------------------------
static inline int clampi_cpu(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// ===========================================================================
// debayer_rggb_cpu — sequential twin of debayer_kernel (kernels.cu). Same
// four-case bilinear demosaic, one pixel at a time, nested loops instead of
// a 2-D thread grid. bayer: W*H uint8_t (RGGB). rgb OUT: W*H*3 uint8_t.
// ===========================================================================
void debayer_rggb_cpu(const unsigned char* bayer, unsigned char* rgb, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int xm = clampi_cpu(x - 1, 0, W - 1), xp = clampi_cpu(x + 1, 0, W - 1);
            const int ym = clampi_cpu(y - 1, 0, H - 1), yp = clampi_cpu(y + 1, 0, H - 1);

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
            if (ch == 0) {                                     // native R
                R = static_cast<float>(center);
                G = 0.25f * (static_cast<float>(n_) + s_ + e_ + w_);
                B = 0.25f * (static_cast<float>(ne) + nw + se + sw);
            } else if (ch == 2) {                              // native B
                B = static_cast<float>(center);
                G = 0.25f * (static_cast<float>(n_) + s_ + e_ + w_);
                R = 0.25f * (static_cast<float>(ne) + nw + se + sw);
            } else {                                            // native G
                G = static_cast<float>(center);
                if ((y & 1) == 0) {
                    R = 0.5f * (static_cast<float>(e_) + w_);
                    B = 0.5f * (static_cast<float>(n_) + s_);
                } else {
                    B = 0.5f * (static_cast<float>(e_) + w_);
                    R = 0.5f * (static_cast<float>(n_) + s_);
                }
            }

            const int o = (y * W + x) * 3;
            const auto clamp255 = [](float v) {
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                return static_cast<unsigned char>(v + 0.5f);
            };
            rgb[o + 0] = clamp255(R);
            rgb[o + 1] = clamp255(G);
            rgb[o + 2] = clamp255(B);
        }
    }
}

// ===========================================================================
// build_remap_lut_cpu — sequential twin of build_remap_lut_kernel. Calls
// the SHARED compute_source_pixel() (kernels.cuh) — see file header for why
// this one function is allowed to be shared rather than re-typed.
// ===========================================================================
void build_remap_lut_cpu(RemapSample* lut, int W, int H)
{
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            lut[y * W + x] = compute_source_pixel(x, y);
}

// ---------------------------------------------------------------------------
// bilinear_sample_rgb_cpu — INDEPENDENT host re-typing of kernels.cu's
// __device__ bilinear_sample_rgb() (see file header: bilinear sampling is
// deliberately duplicated, not shared, so the twin comparison actually
// exercises the interpolation arithmetic).
// ---------------------------------------------------------------------------
static void bilinear_sample_rgb_cpu(const unsigned char* img, int W, int H,
                                    float u, float v, float out[3])
{
    if (u < 0.0f) u = 0.0f;
    if (u > static_cast<float>(W - 1)) u = static_cast<float>(W - 1);
    if (v < 0.0f) v = 0.0f;
    if (v > static_cast<float>(H - 1)) v = static_cast<float>(H - 1);

    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const int x1 = (x0 + 1 < W - 1) ? x0 + 1 : W - 1;
    const int y1 = (y0 + 1 < H - 1) ? y0 + 1 : H - 1;
    const float fx = u - static_cast<float>(x0);
    const float fy = v - static_cast<float>(y0);

    for (int c = 0; c < 3; ++c) {
        const float v00 = static_cast<float>(img[(y0 * W + x0) * 3 + c]);
        const float v10 = static_cast<float>(img[(y0 * W + x1) * 3 + c]);
        const float v01 = static_cast<float>(img[(y1 * W + x0) * 3 + c]);
        const float v11 = static_cast<float>(img[(y1 * W + x1) * 3 + c]);
        const float top = v00 + (v10 - v00) * fx;
        const float bot = v01 + (v11 - v01) * fx;
        out[c] = top + (bot - top) * fy;
    }
}

// ===========================================================================
// remap_bilinear_cpu — sequential twin of remap_bilinear_kernel: for every
// full-resolution output pixel, look up its LUT entry and bilinear-sample.
// ===========================================================================
void remap_bilinear_cpu(const unsigned char* rgb_in, const RemapSample* lut,
                        unsigned char* rgb_out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const RemapSample s = lut[y * W + x];
            float rgb[3];
            bilinear_sample_rgb_cpu(rgb_in, W, H, s.u, s.v, rgb);
            const int o = (y * W + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float v = rgb[c];
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                rgb_out[o + c] = static_cast<unsigned char>(v + 0.5f);
            }
        }
    }
}

// ===========================================================================
// resize_area2x_cpu — sequential twin of resize_area2x_kernel: exact
// kResizeFactor x area-average downscale, one output pixel at a time.
// ===========================================================================
void resize_area2x_cpu(const unsigned char* rgb_in, unsigned char* rgb_out, int Wf, int Hf)
{
    const int Wr = Wf / kResizeFactor, Hr = Hf / kResizeFactor;
    for (int yo = 0; yo < Hr; ++yo) {
        for (int xo = 0; xo < Wr; ++xo) {
            const int x0 = xo * kResizeFactor, y0 = yo * kResizeFactor;
            float acc[3] = { 0.0f, 0.0f, 0.0f };
            for (int dy = 0; dy < kResizeFactor; ++dy) {
                for (int dx = 0; dx < kResizeFactor; ++dx) {
                    const int o = ((y0 + dy) * Wf + (x0 + dx)) * 3;
                    for (int c = 0; c < 3; ++c) acc[c] += static_cast<float>(rgb_in[o + c]);
                }
            }
            const float norm = 1.0f / static_cast<float>(kResizeFactor * kResizeFactor);
            const int oo = (yo * Wr + xo) * 3;
            for (int c = 0; c < 3; ++c)
                rgb_out[oo + c] = static_cast<unsigned char>(acc[c] * norm + 0.5f);
        }
    }
}

// ===========================================================================
// fused_undistort_rectify_resize_cpu — sequential twin of fused_kernel: for
// every RESIZED output pixel, average the kResizeFactor^2 bilinear samples
// of the full-resolution sub-pixels in FLOAT, rounding once — the same
// single-rounding numerics as the GPU fused kernel (see kernels.cu's
// header note on why this differs slightly from the staged path).
// ===========================================================================
void fused_undistort_rectify_resize_cpu(const unsigned char* rgb_in, const RemapSample* lut_fullres,
                                        unsigned char* rgb_out, int Wf, int Hf)
{
    const int Wr = Wf / kResizeFactor, Hr = Hf / kResizeFactor;
    for (int yo = 0; yo < Hr; ++yo) {
        for (int xo = 0; xo < Wr; ++xo) {
            float acc[3] = { 0.0f, 0.0f, 0.0f };
            for (int dy = 0; dy < kResizeFactor; ++dy) {
                for (int dx = 0; dx < kResizeFactor; ++dx) {
                    const int xf = xo * kResizeFactor + dx;
                    const int yf = yo * kResizeFactor + dy;
                    const RemapSample s = lut_fullres[yf * Wf + xf];
                    float rgb[3];
                    bilinear_sample_rgb_cpu(rgb_in, Wf, Hf, s.u, s.v, rgb);
                    for (int c = 0; c < 3; ++c) acc[c] += rgb[c];
                }
            }
            const float norm = 1.0f / static_cast<float>(kResizeFactor * kResizeFactor);
            const int oo = (yo * Wr + xo) * 3;
            for (int c = 0; c < 3; ++c)
                rgb_out[oo + c] = static_cast<unsigned char>(acc[c] * norm + 0.5f);
        }
    }
}

// ===========================================================================
// normalize_stats_cpu — INDEPENDENT single-pass mean/variance computation.
// The CPU needs none of the GPU's block-then-finalize dance (THEORY.md
// contrasts the two): one sequential loop accumulating in double is
// ALREADY perfectly deterministic (there is only one possible summation
// order on one thread), so this file's "twin" of the GPU's three-kernel
// reduction is deliberately much simpler — the two are checked for
// AGREEMENT, not for using the same algorithm (CLAUDE.md paragraph 5: the
// oracle should be the simplest correct statement of the computation).
// rgb: W*H*3 uint8_t. mean3/std3 OUT: 3 doubles each (R,G,B).
// ===========================================================================
void normalize_stats_cpu(const unsigned char* rgb, int W, int H, double mean3[3], double std3[3])
{
    const long long n_pixels = static_cast<long long>(W) * static_cast<long long>(H);
    double sum[3] = { 0.0, 0.0, 0.0 };
    double sumsq[3] = { 0.0, 0.0, 0.0 };
    for (long long i = 0; i < n_pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double v = static_cast<double>(rgb[i * 3 + c]);
            sum[c] += v;
            sumsq[c] += v * v;
        }
    }
    const double n = static_cast<double>(n_pixels);
    for (int c = 0; c < 3; ++c) {
        const double mean = sum[c] / n;
        double var = sumsq[c] / n - mean * mean;                 // population variance (see kernels.cu note)
        if (var < static_cast<double>(kNormEps)) var = static_cast<double>(kNormEps);
        mean3[c] = mean;
        std3[c] = std::sqrt(var);
    }
}

// ===========================================================================
// normalize_apply_cpu — sequential twin of normalize_apply_kernel. Mean/std
// are passed as double (the CPU's natural accumulation precision) but the
// affine map is evaluated the SAME way the GPU does it — cast to float
// first, then divide — so the two paths' rounding behavior matches as
// closely as two independent implementations reasonably can.
// ===========================================================================
void normalize_apply_cpu(const unsigned char* rgb, float* out, int W, int H,
                         const double mean3[3], const double std3[3])
{
    const float mean_f[3] = { static_cast<float>(mean3[0]), static_cast<float>(mean3[1]), static_cast<float>(mean3[2]) };
    const float std_f[3]  = { static_cast<float>(std3[0]),  static_cast<float>(std3[1]),  static_cast<float>(std3[2]) };
    const long long n_pixels = static_cast<long long>(W) * static_cast<long long>(H);
    for (long long i = 0; i < n_pixels; ++i) {
        for (int c = 0; c < 3; ++c)
            out[i * 3 + c] = (static_cast<float>(rgb[i * 3 + c]) - mean_f[c]) / std_f[c];
    }
}
