// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.03
//                     (Optical flow: dense pyramidal Lucas-Kanade + census-
//                     transform block-matching flow)
//
// WHY does a GPU repository ship a CPU implementation of everything? Two
// load-bearing reasons (CLAUDE.md §5) — the CORRECTNESS ORACLE (a dead-
// simple sequential version a reader can verify by eye) and the TEACHING
// BASELINE (reading this file, then kernels.cu, shows exactly what
// parallelization changed). See docs/PROJECT_TEMPLATE/src/reference_cpu.cpp
// for the repo-wide version of this argument in full, including the
// independence ruling this file follows:
//
//   * Data-LAYOUT contracts (image geometry, the census offset table, the
//     pyramid level sizing, tolerances) are single-sourced in kernels.cuh
//     and SHARED — divergent layouts would be a bug class of their own.
//   * The ALGORITHMIC CORE (gradient stencils, structure-tensor
//     accumulation, the 2x2 solve, bilinear warping, census bit-packing,
//     Hamming matching, sub-pixel refinement, LR consistency) is written
//     TWICE, independently, in the simplest possible C++ HERE.
//
// This project's specific twin strategy (mirrored from 01.04's identical
// FAST/Harris split — see that project's kernels.cuh header for the general
// argument): downsample and census stages are ALL-INTEGER, so they are
// BIT-EXACT twins. Scharr gradients are float but exact-integer-VALUED (see
// scharr_gradient_cpu's numerics note), so they are ALSO effectively
// bit-exact. Structure-tensor accumulation, the LK solve, bilinear warping,
// and census sub-pixel refinement are genuine floating point with a
// DIFFERENT accumulation order/precision than the GPU path (this file
// deliberately accumulates the structure tensor in DOUBLE — the same
// "independent numerical path" choice 01.04's Harris CPU twin makes, for
// the same reason: an honest, informative tolerance instead of an
// accidentally-bit-exact one that would hide a real accumulation-order bug).
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong, and
// then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <algorithm>   // std::min/max
#include <cmath>       // std::sqrt, std::floor, std::lround
#include <vector>

// ===========================================================================
// bilinear_sample_u8_cpu — the HOST-only twin of kernels.cu's
// bilinear_sample_u8 device helper. Deliberately a SEPARATE, independently
// written function (not a shared __host__ __device__ helper) — the exact
// design decision project 01.01's kernels.cuh documents for its own
// bilinear sampler, cited in kernels.cu's bilinear_sample_u8 header: small
// interpolation arithmetic is exactly where an independent implementation
// earns its keep as a correctness check.
// ---------------------------------------------------------------------------
static float bilinear_sample_u8_cpu(const uint8_t* img, int W, int H, float x, float y)
{
    x = std::min(std::max(x, 0.0f), static_cast<float>(W - 1));
    y = std::min(std::max(y, 0.0f), static_cast<float>(H - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);

    // Written as a weighted 4-tap sum (rather than kernels.cu's "two
    // horizontal lerps then one vertical lerp" shape) — algebraically the
    // same bilinear formula, arrived at independently, a different-looking
    // path to the same answer (see this file's header on why that matters).
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;
    return w00 * static_cast<float>(img[y0 * W + x0]) + w10 * static_cast<float>(img[y0 * W + x1]) +
           w01 * static_cast<float>(img[y1 * W + x0]) + w11 * static_cast<float>(img[y1 * W + x1]);
}

// ===========================================================================
// MILESTONE 1 — dense pyramidal Lucas-Kanade CPU twins.
// ===========================================================================

// downsample_area2x_cpu — all-integer 2x2 box average (bit-exact twin of
// downsample_area2x_kernel; see this file's header).
void downsample_area2x_cpu(const uint8_t* in, int inW, int inH, uint8_t* out)
{
    const int outW = inW / 2, outH = inH / 2;
    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            const int ix = ox * 2, iy = oy * 2;
            int sum = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    sum += static_cast<int>(in[(iy + dy) * inW + (ix + dx)]);
            out[oy * outW + ox] = static_cast<uint8_t>((sum + 2) / 4);
        }
    }
}

// scharr_gradient_cpu — independent host Scharr stencil, using a local 3x3
// array read (a different shape from kernels.cu's nine named locals — the
// same "different shape, same math" style 01.04's Sobel CPU twin uses).
void scharr_gradient_cpu(const uint8_t* img, int W, int H, float* gx_out, float* gy_out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (x < kGradBorder || x >= W - kGradBorder || y < kGradBorder || y >= H - kGradBorder) {
                gx_out[idx] = 0.0f; gy_out[idx] = 0.0f;
                continue;
            }
            int p[3][3];
            for (int wy = -1; wy <= 1; ++wy)
                for (int wx = -1; wx <= 1; ++wx)
                    p[wy + 1][wx + 1] = static_cast<int>(img[(y + wy) * W + (x + wx)]);

            const int gx = (3 * p[0][2] + 10 * p[1][2] + 3 * p[2][2]) - (3 * p[0][0] + 10 * p[1][0] + 3 * p[2][0]);
            const int gy = (3 * p[2][0] + 10 * p[2][1] + 3 * p[2][2]) - (3 * p[0][0] + 10 * p[0][1] + 3 * p[0][2]);
            // /32 normalization: an EXACT power-of-two float32 operation,
            // load-bearing for LK's scale-sensitive normal equations — see
            // kernels.cu's scharr_gradient_kernel header for the full
            // "why an unnormalized gradient silently breaks LK" derivation.
            gx_out[idx] = static_cast<float>(gx) * (1.0f / 32.0f);
            gy_out[idx] = static_cast<float>(gy) * (1.0f / 32.0f);
        }
    }
}

// structure_tensor_cpu — independent host structure tensor + min-eigenvalue,
// DOUBLE-accumulated (an independent numerical path from kernels.cu's float
// accumulation, on purpose — see this file's header).
void structure_tensor_cpu(const float* gx, const float* gy, int W, int H,
                          float* sxx_out, float* syy_out, float* sxy_out, float* min_eig_out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (x < kLkBorder || x >= W - kLkBorder || y < kLkBorder || y >= H - kLkBorder) {
                sxx_out[idx] = syy_out[idx] = sxy_out[idx] = min_eig_out[idx] = 0.0f;
                continue;
            }
            double sxx = 0.0, syy = 0.0, sxy = 0.0;
            for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
                for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
                    const int widx = (y + wy) * W + (x + wx);
                    const double gxv = static_cast<double>(gx[widx]);
                    const double gyv = static_cast<double>(gy[widx]);
                    sxx += gxv * gxv;
                    syy += gyv * gyv;
                    sxy += gxv * gyv;
                }
            }
            sxx_out[idx] = static_cast<float>(sxx);
            syy_out[idx] = static_cast<float>(syy);
            sxy_out[idx] = static_cast<float>(sxy);

            const double half_trace = 0.5 * (sxx + syy);
            const double det = sxx * syy - sxy * sxy;
            const double disc = std::max(half_trace * half_trace - det, 0.0);
            min_eig_out[idx] = static_cast<float>(half_trace - std::sqrt(disc));
        }
    }
}

// lk_iterate_cpu — independent host forward-additive LK step (see
// kernels.cu's lk_iterate_kernel for the full derivation this mirrors).
void lk_iterate_cpu(const uint8_t* img0, const uint8_t* img1, int W, int H,
                    const float* gx, const float* gy,
                    const float* sxx, const float* syy, const float* sxy,
                    float* flow_u, float* flow_v)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (x < kLkBorder || x >= W - kLkBorder || y < kLkBorder || y >= H - kLkBorder) continue;
            const int idx = y * W + x;

            const float Sxx = sxx[idx], Syy = syy[idx], Sxy = sxy[idx];
            const float det = Sxx * Syy - Sxy * Sxy;
            if (det < kLkDetEpsilon) continue;

            const float u = flow_u[idx], v = flow_v[idx];
            double bx = 0.0, by = 0.0;   // double accumulation: an independent numerical path from kernels.cu (see header)
            for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
                for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
                    const int widx = (y + wy) * W + (x + wx);
                    const float sx = static_cast<float>(x + wx) + u;
                    const float sy = static_cast<float>(y + wy) + v;
                    const float i1w = bilinear_sample_u8_cpu(img1, W, H, sx, sy);
                    const double it = static_cast<double>(i1w) - static_cast<double>(img0[widx]);
                    bx += static_cast<double>(gx[widx]) * it;
                    by += static_cast<double>(gy[widx]) * it;
                }
            }

            double ddu = -(static_cast<double>(Syy) * bx - static_cast<double>(Sxy) * by) / static_cast<double>(det);
            double ddv = -(-static_cast<double>(Sxy) * bx + static_cast<double>(Sxx) * by) / static_cast<double>(det);
            ddu = std::min(std::max(ddu, -static_cast<double>(kLkMaxStepPerIterPx)), static_cast<double>(kLkMaxStepPerIterPx));
            ddv = std::min(std::max(ddv, -static_cast<double>(kLkMaxStepPerIterPx)), static_cast<double>(kLkMaxStepPerIterPx));

            flow_u[idx] = u + static_cast<float>(ddu);
            flow_v[idx] = v + static_cast<float>(ddv);
        }
    }
}

// upsample_flow_cpu — independent host bilinear upsample + x2 magnitude
// scale (see kernels.cu's upsample_flow_kernel for the "why x2" argument).
void upsample_flow_cpu(const float* coarse_u, const float* coarse_v, int coarseW, int coarseH,
                       float* fine_u, float* fine_v, int fineW, int fineH)
{
    for (int fy = 0; fy < fineH; ++fy) {
        for (int fx = 0; fx < fineW; ++fx) {
            const float cx = (static_cast<float>(fx) + 0.5f) * 0.5f - 0.5f;
            const float cy = (static_cast<float>(fy) + 0.5f) * 0.5f - 0.5f;
            const float ccx = std::min(std::max(cx, 0.0f), static_cast<float>(coarseW - 1));
            const float ccy = std::min(std::max(cy, 0.0f), static_cast<float>(coarseH - 1));

            const int x0 = static_cast<int>(std::floor(ccx)), y0 = static_cast<int>(std::floor(ccy));
            const int x1 = std::min(x0 + 1, coarseW - 1), y1 = std::min(y0 + 1, coarseH - 1);
            const float wx = ccx - static_cast<float>(x0), wy = ccy - static_cast<float>(y0);

            auto bilerp = [&](const float* field) -> float {
                const float v00 = field[y0 * coarseW + x0], v10 = field[y0 * coarseW + x1];
                const float v01 = field[y1 * coarseW + x0], v11 = field[y1 * coarseW + x1];
                const float top = v00 * (1.0f - wx) + v10 * wx;
                const float bot = v01 * (1.0f - wx) + v11 * wx;
                return top * (1.0f - wy) + bot * wy;
            };

            const int fidx = fy * fineW + fx;
            fine_u[fidx] = 2.0f * bilerp(coarse_u);
            fine_v[fidx] = 2.0f * bilerp(coarse_v);
        }
    }
}

// pyramidal_lk_cpu — the FULL Milestone-1 oracle: an independently-written
// pyramid build + coarse-to-fine loop, mirroring run_pyramidal_lk_gpu's
// STRUCTURE (build once, loop coarse-to-fine) while every stage inside it
// is the independent CPU twin above — see this file's header for why the
// orchestration SHAPE may match (it is plumbing, not "the algorithmic
// core") while the math inside each stage must not.
void pyramidal_lk_cpu(const uint8_t* img0_full, const uint8_t* img1_full,
                      int num_levels, int iters_per_level,
                      float* flow_u_out, float* flow_v_out, float* min_eig_out)
{
    std::vector<uint8_t> img0[kNumLevels], img1[kNumLevels];
    std::vector<float> gx[kNumLevels], gy[kNumLevels];
    std::vector<float> sxx[kNumLevels], syy[kNumLevels], sxy[kNumLevels], min_eig[kNumLevels];
    std::vector<float> flow_u[kNumLevels], flow_v[kNumLevels];

    for (int L = 0; L < kNumLevels; ++L) {
        const size_t n = static_cast<size_t>(level_w(L)) * level_h(L);
        img0[L].resize(n); img1[L].resize(n);
        gx[L].resize(n); gy[L].resize(n);
        sxx[L].resize(n); syy[L].resize(n); sxy[L].resize(n); min_eig[L].resize(n);
        flow_u[L].assign(n, 0.0f); flow_v[L].assign(n, 0.0f);
    }

    img0[0].assign(img0_full, img0_full + static_cast<size_t>(kW) * kH);
    img1[0].assign(img1_full, img1_full + static_cast<size_t>(kW) * kH);
    for (int L = 1; L < kNumLevels; ++L) {
        downsample_area2x_cpu(img0[L - 1].data(), level_w(L - 1), level_h(L - 1), img0[L].data());
        downsample_area2x_cpu(img1[L - 1].data(), level_w(L - 1), level_h(L - 1), img1[L].data());
    }

    const int start_level = num_levels - 1;
    for (int L = start_level; L >= 0; --L) {
        const int Wl = level_w(L), Hl = level_h(L);
        scharr_gradient_cpu(img0[L].data(), Wl, Hl, gx[L].data(), gy[L].data());
        structure_tensor_cpu(gx[L].data(), gy[L].data(), Wl, Hl, sxx[L].data(), syy[L].data(), sxy[L].data(), min_eig[L].data());
        for (int it = 0; it < iters_per_level; ++it) {
            lk_iterate_cpu(img0[L].data(), img1[L].data(), Wl, Hl, gx[L].data(), gy[L].data(),
                          sxx[L].data(), syy[L].data(), sxy[L].data(), flow_u[L].data(), flow_v[L].data());
        }
        if (L > 0) {
            upsample_flow_cpu(flow_u[L].data(), flow_v[L].data(), Wl, Hl,
                             flow_u[L - 1].data(), flow_v[L - 1].data(), level_w(L - 1), level_h(L - 1));
        }
    }

    const size_t n0 = static_cast<size_t>(kW) * kH;
    std::copy(flow_u[0].begin(), flow_u[0].end(), flow_u_out);
    std::copy(flow_v[0].begin(), flow_v[0].end(), flow_v_out);
    std::copy(min_eig[0].begin(), min_eig[0].begin() + static_cast<long>(n0), min_eig_out);
}

// ===========================================================================
// MILESTONE 2 — census-transform block-matching flow CPU twins.
// ===========================================================================

// census_transform_cpu — bit-exact twin of census_transform_kernel.
void census_transform_cpu(const uint8_t* img, int W, int H, uint32_t* census_out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (x < kCensusRadius || x >= W - kCensusRadius || y < kCensusRadius || y >= H - kCensusRadius) {
                census_out[idx] = 0u;
                continue;
            }
            const int center = static_cast<int>(img[idx]);
            uint32_t sig = 0u;
            for (int k = 0; k < kCensusBits; ++k) {
                const int nx = x + kCensusDx[k], ny = y + kCensusDy[k];
                const int neighbor = static_cast<int>(img[ny * W + nx]);
                if (neighbor >= center) sig |= (1u << k);   // same polarity convention as census_transform_kernel — see kernels.cuh
            }
            census_out[idx] = sig;
        }
    }
}

// census_match_cpu — bit-exact WTA + cost, tolerance-checked sub-pixel
// refinement (independent re-implementation of census_match_kernel's search
// and parabola — see that kernel's header for the full derivation).
void census_match_cpu(const uint32_t* census_ref, const uint32_t* census_tgt, int W, int H,
                      float* flow_u, float* flow_v, int* cost_min_out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (x < kCensusBorder || x >= W - kCensusBorder || y < kCensusBorder || y >= H - kCensusBorder) {
                flow_u[idx] = 0.0f; flow_v[idx] = 0.0f; cost_min_out[idx] = kCensusBits + 1;
                continue;
            }
            const uint32_t ref_sig = census_ref[idx];
            int best_cost = kCensusBits + 1, best_dx = 0, best_dy = 0;
            for (int dy = -kCensusSearchRadius; dy <= kCensusSearchRadius; ++dy) {
                for (int dx = -kCensusSearchRadius; dx <= kCensusSearchRadius; ++dx) {
                    const uint32_t tgt_sig = census_tgt[(y + dy) * W + (x + dx)];
                    const int cost = popcount32_portable(ref_sig ^ tgt_sig);   // SWAR popcount — see kernels.cuh's header
                    if (cost < best_cost) { best_cost = cost; best_dx = dx; best_dy = dy; }
                }
            }
            cost_min_out[idx] = best_cost;

            float sub_dx = 0.0f, sub_dy = 0.0f;
            if (best_dx > -kCensusSearchRadius && best_dx < kCensusSearchRadius &&
                best_dy > -kCensusSearchRadius && best_dy < kCensusSearchRadius) {
                const int c_xm = popcount32_portable(ref_sig ^ census_tgt[(y + best_dy) * W + (x + best_dx - 1)]);
                const int c_xp = popcount32_portable(ref_sig ^ census_tgt[(y + best_dy) * W + (x + best_dx + 1)]);
                const int c_ym = popcount32_portable(ref_sig ^ census_tgt[(y + best_dy - 1) * W + (x + best_dx)]);
                const int c_yp = popcount32_portable(ref_sig ^ census_tgt[(y + best_dy + 1) * W + (x + best_dx)]);

                const float denom_x = static_cast<float>(c_xm - 2 * best_cost + c_xp);
                if (denom_x > 1e-3f) sub_dx = 0.5f * static_cast<float>(c_xm - c_xp) / denom_x;
                const float denom_y = static_cast<float>(c_ym - 2 * best_cost + c_yp);
                if (denom_y > 1e-3f) sub_dy = 0.5f * static_cast<float>(c_ym - c_yp) / denom_y;
            }
            flow_u[idx] = static_cast<float>(best_dx) + sub_dx;
            flow_v[idx] = static_cast<float>(best_dy) + sub_dy;
        }
    }
}

// census_consistency_cpu — independent host LR consistency check.
void census_consistency_cpu(const float* fwd_u, const float* fwd_v,
                            const float* bwd_u, const float* bwd_v, int W, int H,
                            uint8_t* valid_out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (x < kCensusBorder || x >= W - kCensusBorder || y < kCensusBorder || y >= H - kCensusBorder) {
                valid_out[idx] = 0u;
                continue;
            }
            const float fu = fwd_u[idx], fv = fwd_v[idx];
            int qx = static_cast<int>(std::lround(static_cast<float>(x) + fu));
            int qy = static_cast<int>(std::lround(static_cast<float>(y) + fv));
            qx = std::min(std::max(qx, kCensusBorder), W - 1 - kCensusBorder);
            qy = std::min(std::max(qy, kCensusBorder), H - 1 - kCensusBorder);

            const float bu = bwd_u[qy * W + qx], bv = bwd_v[qy * W + qx];
            const double res_x = static_cast<double>(fu) + bu, res_y = static_cast<double>(fv) + bv;
            const double residual = std::sqrt(res_x * res_x + res_y * res_y);
            valid_out[idx] = (residual <= static_cast<double>(kCensusConsistencyTolPx)) ? 1u : 0u;
        }
    }
}

// census_flow_cpu — the FULL Milestone-2 oracle, independently orchestrated
// (see pyramidal_lk_cpu's header for the "orchestration shape may match,
// math must not" ruling this mirrors).
void census_flow_cpu(const uint8_t* img0, const uint8_t* img1,
                     float* flow_u_out, float* flow_v_out, uint8_t* valid_out)
{
    const size_t n = static_cast<size_t>(kW) * kH;
    std::vector<uint32_t> census0(n), census1(n);
    census_transform_cpu(img0, kW, kH, census0.data());
    census_transform_cpu(img1, kW, kH, census1.data());

    std::vector<int> cost_fwd(n), cost_bwd(n);
    std::vector<float> bwd_u(n), bwd_v(n);
    census_match_cpu(census0.data(), census1.data(), kW, kH, flow_u_out, flow_v_out, cost_fwd.data());
    census_match_cpu(census1.data(), census0.data(), kW, kH, bwd_u.data(), bwd_v.data(), cost_bwd.data());
    census_consistency_cpu(flow_u_out, flow_v_out, bwd_u.data(), bwd_v.data(), kW, kH, valid_out);
}
