// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.08
//                     (HDR exposure fusion + tone mapping for outdoor robots)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): this file is the CORRECTNESS
// ORACLE main.cu's VERIFY step checks the GPU path against, and it is the
// TEACHING BASELINE that makes kernels.cu's parallelization legible — read
// gaussian_reduce_cpu next to gaussian_reduce_kernel and the mapping from
// "for every output pixel" to "thread (ox,oy) owns one output pixel" is
// the whole lesson in one side-by-side comparison.
//
// Independence ruling (kernels.cuh SECTION 5 restates this in full) — how
// this project applies it:
//   * DATA-LAYOUT constants (kW, kH, kNumExposures, kNumLevels, kCrfBins,
//     kExposureTimes) are single-sourced in kernels.cuh and shared, per the
//     ruling's first bullet — divergent layouts are a bug class of their
//     own, not "independence".
//   * EVERY primitive below (radiance merge, log-luminance mean, Reinhard
//     map, Gaussian reduce, bilinear expand, elementwise combine, affine,
//     log, u8-to-unit, Mertens weight, weight normalize, weighted sum) is
//     re-typed HERE independently of kernels.cu — same math, a fresh loop
//     structure, so the twin comparison in main.cu actually catches
//     thread-indexing bugs, off-by-one tile boundaries, and race
//     conditions the GPU side alone could hide.
//   * crf_solve_debevec is the ruling's documented EXCEPTION: it is HOST-
//     ONLY code shared verbatim from kernels.cu (see kernels.cuh SECTION 5
//     for the full justification — there is no meaningful GPU
//     parallelization of a one-time ~320x320 dense solve). Because that
//     one function is shared, the GPU-vs-CPU twin comparison is BLIND to
//     bugs inside it; this project's crf_recovery gate (main.cu) is
//     therefore an INDEPENDENT check against scripts/make_synthetic.py's
//     KNOWN analytic CRF, never against a second implementation of the
//     solver — precisely the "at least one gate that does not route
//     through the shared code" the ruling requires.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — then compare stage by stage.
// ===========================================================================

#include <cmath>
#include <vector>

#include "kernels.cuh"

// ---------------------------------------------------------------------------
// hat_weight_cpu — independent re-typing of the triangular Debevec-Malik
// weight w(z) = min(z, 255-z) (see kernels.cu's hat_weight_device for the
// full rationale: this down-weights near-clipped/near-black samples).
// ---------------------------------------------------------------------------
static double hat_weight_cpu(int z)
{
    const int hi = 255 - z;
    return static_cast<double>(z < hi ? z : hi);
}

// ---------------------------------------------------------------------------
// radiance_merge_cpu — sequential twin of radiance_merge_kernel. For every
// pixel: hat-weighted average of the four (g(Z) - ln t) estimates in the
// log domain, with the same "clipped-everywhere" fallback (see the GPU
// kernel's header for the full derivation — repeated in brief here since
// this file must stand alone as a teaching artifact too).
// ---------------------------------------------------------------------------
void radiance_merge_cpu(const uint8_t* z0, const uint8_t* z1, const uint8_t* z2, const uint8_t* z3,
                        int n, float ln_t0, float ln_t1, float ln_t2, float ln_t3,
                        const float* g256,
                        float* out_radiance)
{
    const float lt[4] = { ln_t0, ln_t1, ln_t2, ln_t3 };
    // g256 is the SAME calibration result crf_solve_debevec produced and
    // main.cu also uploaded (via upload_crf_table) to the GPU path's
    // __constant__ memory — see kernels.cuh's declaration comment for why
    // taking it as an explicit parameter (rather than a hidden global)
    // keeps this function a pure, independently-reasoned twin: the CRF is
    // shared CALIBRATION INPUT to both paths, not part of what either twin
    // computes, so passing it in does not compromise the independence
    // ruling — only the per-pixel MERGE arithmetic below is what is being
    // verified, and that is written fresh here.
    const float* g = g256;

    for (int i = 0; i < n; ++i) {
        const int zz[4] = { z0[i], z1[i], z2[i], z3[i] };
        double wsum = 0.0, acc = 0.0;
        int z_max = -1;
        for (int j = 0; j < 4; ++j) {
            const double w = hat_weight_cpu(zz[j]);
            acc += w * (static_cast<double>(g[zz[j]]) - static_cast<double>(lt[j]));
            wsum += w;
            if (zz[j] > z_max) z_max = zz[j];
        }
        double ln_e;
        if (wsum > 1e-6) {
            ln_e = acc / wsum;
        } else {
            double lt_longest = lt[0], lt_shortest = lt[0];
            for (int j = 1; j < 4; ++j) {
                if (lt[j] > lt_longest)  lt_longest  = lt[j];
                if (lt[j] < lt_shortest) lt_shortest = lt[j];
            }
            ln_e = (z_max == 0) ? (static_cast<double>(g[0])   - lt_longest)
                                : (static_cast<double>(g[255]) - lt_shortest);
        }
        out_radiance[i] = static_cast<float>(std::exp(ln_e));
    }
}

// ---------------------------------------------------------------------------
// luminance_log_mean_cpu — MEAN of ln(eps + radiance) over n pixels (the
// same quantity luminance_log_sum_kernel's reduction computes the SUM of;
// main.cu compares sum/n against this directly). Plain running sum — no
// tree reduction needed on a single CPU core, which is exactly the point
// being taught: the GPU's shared-memory tree + atomic pattern exists to
// parallelize what is, sequentially, this one-line accumulation loop.
// ---------------------------------------------------------------------------
double luminance_log_mean_cpu(const float* radiance, int n, float eps)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::log(static_cast<double>(eps) + static_cast<double>(radiance[i]));
    return sum / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// reinhard_map_cpu — independent twin of reinhard_map_kernel.
// ---------------------------------------------------------------------------
void reinhard_map_cpu(const float* radiance, int n, float key_over_lavg, float* out)
{
    for (int i = 0; i < n; ++i) {
        const double l_scaled = static_cast<double>(key_over_lavg) * static_cast<double>(radiance[i]);
        out[i] = static_cast<float>(l_scaled / (1.0 + l_scaled));
    }
}

// ---------------------------------------------------------------------------
// gaussian_reduce_cpu — independent twin of gaussian_reduce_kernel: same
// 5x5 binomial blur + 2x downsample, same clamp-to-edge border rule,
// re-typed as nested loops over OUTPUT rows/cols (rather than the GPU's
// per-thread (ox,oy)) — a genuinely different control-flow shape computing
// the identical math, exactly what makes this comparison meaningful.
// ---------------------------------------------------------------------------
void gaussian_reduce_cpu(const float* in, int inW, int inH, float* out)
{
    const double tap[5] = { 1.0, 4.0, 6.0, 4.0, 1.0 };
    const int outW = inW / 2, outH = inH / 2;

    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            const int cx = ox * 2, cy = oy * 2;
            double acc = 0.0;
            for (int dy = -2; dy <= 2; ++dy) {
                int sy = cy + dy;
                if (sy < 0) sy = 0; else if (sy >= inH) sy = inH - 1;
                for (int dx = -2; dx <= 2; ++dx) {
                    int sx = cx + dx;
                    if (sx < 0) sx = 0; else if (sx >= inW) sx = inW - 1;
                    acc += tap[dy + 2] * tap[dx + 2] * static_cast<double>(in[sy * inW + sx]);
                }
            }
            out[oy * outW + ox] = static_cast<float>(acc / 256.0);
        }
    }
}

// ---------------------------------------------------------------------------
// bilinear_expand_cpu — independent twin of bilinear_expand_kernel: same
// pixel-center resampling convention, same clamp-to-edge border.
// ---------------------------------------------------------------------------
void bilinear_expand_cpu(const float* in, int inW, int inH, float* out, int outW, int outH)
{
    for (int oy = 0; oy < outH; ++oy) {
        double sy = (static_cast<double>(oy) + 0.5) * (static_cast<double>(inH) / outH) - 0.5;
        if (sy < 0.0) sy = 0.0; else if (sy > inH - 1.0) sy = inH - 1.0;
        const int y0 = static_cast<int>(sy);
        const int y1 = (y0 + 1 < inH) ? y0 + 1 : y0;
        const double fy = sy - y0;

        for (int ox = 0; ox < outW; ++ox) {
            double sx = (static_cast<double>(ox) + 0.5) * (static_cast<double>(inW) / outW) - 0.5;
            if (sx < 0.0) sx = 0.0; else if (sx > inW - 1.0) sx = inW - 1.0;
            const int x0 = static_cast<int>(sx);
            const int x1 = (x0 + 1 < inW) ? x0 + 1 : x0;
            const double fx = sx - x0;

            const double v00 = in[y0 * inW + x0], v10 = in[y0 * inW + x1];
            const double v01 = in[y1 * inW + x0], v11 = in[y1 * inW + x1];
            const double top = v00 * (1.0 - fx) + v10 * fx;
            const double bot = v01 * (1.0 - fx) + v11 * fx;
            out[oy * outW + ox] = static_cast<float>(top * (1.0 - fy) + bot * fy);
        }
    }
}

// ---------------------------------------------------------------------------
// elementwise_sub_cpu / elementwise_add_cpu / affine_cpu / log_map_cpu /
// u8_to_unit_cpu — the simplest possible loop for each; a bug in any of
// these one-liners would be a startling place for a bug to hide, but the
// repo convention is uniform independence, so they get their own loops too.
// ---------------------------------------------------------------------------
void elementwise_sub_cpu(const float* a, const float* b, int n, float* out)
{
    for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
}
void elementwise_add_cpu(const float* a, const float* b, int n, float* out)
{
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void affine_cpu(const float* in, int n, float scale, float offset, float* out)
{
    for (int i = 0; i < n; ++i) out[i] = scale * in[i] + offset;
}
void log_map_cpu(const float* in, int n, float eps, float* out)
{
    for (int i = 0; i < n; ++i) out[i] = std::log(in[i] + eps);
}
void u8_to_unit_cpu(const uint8_t* in, int n, float* out)
{
    for (int i = 0; i < n; ++i) out[i] = static_cast<float>(in[i]) / 255.0f;
}

// ---------------------------------------------------------------------------
// mertens_raw_weight_cpu — independent twin of mertens_raw_weight_kernel:
// same 3x3 Laplacian contrast + Gaussian well-exposedness, same two-term
// (no saturation — grayscale scene, see kernels.cuh) reduced formula.
// ---------------------------------------------------------------------------
void mertens_raw_weight_cpu(const float* img01, int W, int H, float wc, float we, float sigma,
                            float* out_weight)
{
    for (int y = 0; y < H; ++y) {
        const int ym = (y > 0) ? y - 1 : 0, yp = (y < H - 1) ? y + 1 : H - 1;
        for (int x = 0; x < W; ++x) {
            const int xm = (x > 0) ? x - 1 : 0, xp = (x < W - 1) ? x + 1 : W - 1;
            const double center = img01[y * W + x];
            const double lap = static_cast<double>(img01[y * W + xm]) + img01[y * W + xp]
                              + img01[ym * W + x] + img01[yp * W + x] - 4.0 * center;
            const double contrast = std::fabs(lap);
            const double d = center - 0.5;
            const double wellexposed = std::exp(-(d * d) / (2.0 * sigma * sigma));
            out_weight[y * W + x] = static_cast<float>(std::pow(contrast, static_cast<double>(wc))
                                                       * std::pow(wellexposed, static_cast<double>(we)));
        }
    }
}

// ---------------------------------------------------------------------------
// normalize_weights4_cpu / weighted_sum4_cpu — independent twins.
// ---------------------------------------------------------------------------
void normalize_weights4_cpu(const float* w0, const float* w1, const float* w2, const float* w3,
                            int n, float* o0, float* o1, float* o2, float* o3)
{
    for (int i = 0; i < n; ++i) {
        const double a0 = w0[i], a1 = w1[i], a2 = w2[i], a3 = w3[i];
        const double s = a0 + a1 + a2 + a3;
        if (s > 1e-6) {
            o0[i] = static_cast<float>(a0 / s); o1[i] = static_cast<float>(a1 / s);
            o2[i] = static_cast<float>(a2 / s); o3[i] = static_cast<float>(a3 / s);
        } else {
            o0[i] = o1[i] = o2[i] = o3[i] = 0.25f;
        }
    }
}
void weighted_sum4_cpu(const float* a0, const float* w0, const float* a1, const float* w1,
                       const float* a2, const float* w2, const float* a3, const float* w3,
                       int n, float* out)
{
    for (int i = 0; i < n; ++i)
        out[i] = a0[i] * w0[i] + a1[i] * w1[i] + a2[i] * w2[i] + a3[i] * w3[i];
}

// ===========================================================================
// High-level CPU orchestration — mirrors run_reinhard_global_gpu /
// run_local_tonemap_gpu / run_mertens_gpu in kernels.cu ONE FOR ONE (same
// stage sequence, so main.cu's per-stage VERIFY comparisons are meaningful)
// but built entirely from the independent _cpu primitives above.
// ===========================================================================

void run_reinhard_global_cpu(const float* radiance, int n, float key, float* out_reinhard)
{
    const double mean_ln = luminance_log_mean_cpu(radiance, n, 1e-6f);
    const double l_avg = std::exp(mean_ln);
    const float key_over_lavg = static_cast<float>(static_cast<double>(key) / l_avg);
    reinhard_map_cpu(radiance, n, key_over_lavg, out_reinhard);
}

void run_local_tonemap_cpu(const float* radiance, int W, int H,
                           float compression_factor, float detail_boost,
                           float* out_tonemap)
{
    const int n = W * H;
    const int w1 = W / 2, h1 = H / 2;
    const int w2 = w1 / 2, h2 = h1 / 2;
    const int n1 = w1 * h1, n2 = w2 * h2;

    std::vector<float> logL(static_cast<size_t>(n)), g1(static_cast<size_t>(n1)), g2(static_cast<size_t>(n2));
    std::vector<float> baseMid(static_cast<size_t>(n1)), baseFull(static_cast<size_t>(n)), detail(static_cast<size_t>(n));
    std::vector<float> baseComp(static_cast<size_t>(n)), detailBoost(static_cast<size_t>(n)), composite(static_cast<size_t>(n));

    log_map_cpu(radiance, n, 1e-6f, logL.data());
    gaussian_reduce_cpu(logL.data(), W, H, g1.data());
    gaussian_reduce_cpu(g1.data(), w1, h1, g2.data());
    bilinear_expand_cpu(g2.data(), w2, h2, baseMid.data(), w1, h1);
    bilinear_expand_cpu(baseMid.data(), w1, h1, baseFull.data(), W, H);
    elementwise_sub_cpu(logL.data(), baseFull.data(), n, detail.data());

    double mean_g2 = 0.0;
    for (int i = 0; i < n2; ++i) mean_g2 += g2[static_cast<size_t>(i)];
    mean_g2 /= static_cast<double>(n2);

    const float offset = static_cast<float>((1.0 - static_cast<double>(compression_factor)) * mean_g2);
    affine_cpu(baseFull.data(), n, compression_factor, offset, baseComp.data());
    affine_cpu(detail.data(), n, detail_boost, 0.0f, detailBoost.data());
    elementwise_add_cpu(baseComp.data(), detailBoost.data(), n, composite.data());

    float lo = composite[0], hi = composite[0];
    for (int i = 0; i < n; ++i) { const float v = composite[static_cast<size_t>(i)]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const float range = (hi - lo) > 1e-6f ? (hi - lo) : 1.0f;
    affine_cpu(composite.data(), n, 1.0f / range, -lo / range, out_tonemap);
}

void run_mertens_cpu(const uint8_t* z0, const uint8_t* z1, const uint8_t* z2, const uint8_t* z3,
                     int W, int H, float wc, float we, float sigma,
                     float* out_naive, float* out_fused)
{
    const uint8_t* z[kNumExposures] = { z0, z1, z2, z3 };
    const int dims_w[kNumLevels] = { W, W / 2, W / 4 };
    const int dims_h[kNumLevels] = { H, H / 2, H / 4 };
    const int n = W * H;

    std::vector<std::vector<float>> img(kNumExposures), rawW(kNumExposures), w0lvl(kNumExposures);
    for (int j = 0; j < kNumExposures; ++j) {
        img[j].assign(static_cast<size_t>(n), 0.0f);
        rawW[j].assign(static_cast<size_t>(n), 0.0f);
        w0lvl[j].assign(static_cast<size_t>(n), 0.0f);
        u8_to_unit_cpu(z[j], n, img[j].data());
        mertens_raw_weight_cpu(img[j].data(), W, H, wc, we, sigma, rawW[j].data());
    }
    normalize_weights4_cpu(rawW[0].data(), rawW[1].data(), rawW[2].data(), rawW[3].data(), n,
                           w0lvl[0].data(), w0lvl[1].data(), w0lvl[2].data(), w0lvl[3].data());

    weighted_sum4_cpu(img[0].data(), w0lvl[0].data(), img[1].data(), w0lvl[1].data(),
                      img[2].data(), w0lvl[2].data(), img[3].data(), w0lvl[3].data(), n, out_naive);

    // Gaussian pyramids of images (GI) and weights (GW); level 0 is img[j]/w0lvl[j].
    std::vector<std::vector<float>> GI[kNumExposures], GW[kNumExposures];
    for (int j = 0; j < kNumExposures; ++j) {
        GI[j].resize(kNumLevels);
        GW[j].resize(kNumLevels);
        GI[j][0] = img[j];
        GW[j][0] = w0lvl[j];
        for (int l = 1; l < kNumLevels; ++l) {
            const int nl = dims_w[l] * dims_h[l];
            GI[j][l].assign(static_cast<size_t>(nl), 0.0f);
            GW[j][l].assign(static_cast<size_t>(nl), 0.0f);
            gaussian_reduce_cpu(GI[j][l - 1].data(), dims_w[l - 1], dims_h[l - 1], GI[j][l].data());
            gaussian_reduce_cpu(GW[j][l - 1].data(), dims_w[l - 1], dims_h[l - 1], GW[j][l].data());
        }
    }

    // Laplacian bands.
    std::vector<std::vector<float>> LI[kNumExposures];
    for (int j = 0; j < kNumExposures; ++j) {
        LI[j].resize(kNumLevels);
        LI[j][kNumLevels - 1] = GI[j][kNumLevels - 1];
        for (int l = 0; l < kNumLevels - 1; ++l) {
            const int nl = dims_w[l] * dims_h[l];
            LI[j][l].assign(static_cast<size_t>(nl), 0.0f);
            std::vector<float> expanded(static_cast<size_t>(nl), 0.0f);
            bilinear_expand_cpu(GI[j][l + 1].data(), dims_w[l + 1], dims_h[l + 1], expanded.data(), dims_w[l], dims_h[l]);
            elementwise_sub_cpu(GI[j][l].data(), expanded.data(), nl, LI[j][l].data());
        }
    }

    // Fuse each level.
    std::vector<std::vector<float>> FL(kNumLevels);
    for (int l = 0; l < kNumLevels; ++l) {
        const int nl = dims_w[l] * dims_h[l];
        FL[l].assign(static_cast<size_t>(nl), 0.0f);
        weighted_sum4_cpu(LI[0][l].data(), GW[0][l].data(), LI[1][l].data(), GW[1][l].data(),
                          LI[2][l].data(), GW[2][l].data(), LI[3][l].data(), GW[3][l].data(), nl, FL[l].data());
    }

    // Reconstruct coarse-to-fine.
    std::vector<float> recon = FL[kNumLevels - 1];
    for (int l = kNumLevels - 2; l >= 0; --l) {
        const int nl = dims_w[l] * dims_h[l];
        std::vector<float> expanded(static_cast<size_t>(nl), 0.0f);
        bilinear_expand_cpu(recon.data(), dims_w[l + 1], dims_h[l + 1], expanded.data(), dims_w[l], dims_h[l]);
        std::vector<float> next(static_cast<size_t>(nl), 0.0f);
        elementwise_add_cpu(FL[l].data(), expanded.data(), nl, next.data());
        recon = next;
    }
    for (int i = 0; i < n; ++i) out_fused[i] = recon[static_cast<size_t>(i)];
}
