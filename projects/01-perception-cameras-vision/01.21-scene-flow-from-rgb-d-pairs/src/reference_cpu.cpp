// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.21
//                     (Scene flow from RGB-D pairs)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) it is the CORRECTNESS ORACLE
// main.cu's VERIFY stage compares every GPU stage against, element-wise,
// within a documented tolerance; (2) it is the TEACHING BASELINE — reading
// this file next to kernels.cu shows exactly what parallelization changed.
//
// Independence ruling (reproduced from docs/PROJECT_TEMPLATE/src/
// reference_cpu.cpp, binding for every project — see that file for the full
// text): data-LAYOUT contracts are single-sourced in kernels.cuh; the
// ALGORITHMIC CORE of every per-pixel/per-point stage below is written
// TWICE, independently (compare this file's loops to kernels.cu's kernels —
// same math, different code shape, never copy-pasted). The ONE exception is
// build_rigid_from_covariance16() (Horn's quaternion solve via shifted power
// iteration), which is SHARED from kernels.cuh — a small, textbook, non-
// approximating linear-algebra routine, exactly CLAUDE.md's "pure
// transcription" exemption (see that function's header for the full
// argument). Because it is shared, this project carries an INDEPENDENT gate
// that does not route through it: main.cu's ego_motion check compares the
// RECOVERED transform against the scene's known, closed-form ground truth
// (R_gt/t_gt), never against a second copy of the solve — exactly the
// "closed-form/analytic solution" tier the ruling requires alongside twin
// agreement (the flagship-13.03 lesson the ruling cites: twin agreement
// proves the GPU is faithful to the CPU path; only an independent, known-
// answer gate proves the SHARED code itself is right).
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong, and
// then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <vector>
#include <algorithm>

// ===========================================================================
// MILESTONE 1 — 2-level pyramidal Lucas-Kanade. Independently re-typed from
// kernels.cu's kernels (same math per 01.03's derivation, cited in
// kernels.cuh's header; a different code SHAPE here — plain nested loops,
// no thread/block indexing — is the natural CPU expression of the same
// per-pixel work kernels.cu spreads across threads).
// ===========================================================================

void downsample_area2x_cpu(const uint8_t* in, int inW, int inH, uint8_t* out)
{
    const int outW = inW / 2, outH = inH / 2;
    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            const int ix = ox * 2, iy = oy * 2;
            const int sum = static_cast<int>(in[iy * inW + ix]) + static_cast<int>(in[iy * inW + ix + 1]) +
                            static_cast<int>(in[(iy + 1) * inW + ix]) + static_cast<int>(in[(iy + 1) * inW + ix + 1]);
            out[oy * outW + ox] = static_cast<uint8_t>((sum + 2) / 4);
        }
    }
}

void scharr_gradient_cpu(const uint8_t* img, int W, int H, float* gx_out, float* gy_out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (x < kGradBorder || x >= W - kGradBorder || y < kGradBorder || y >= H - kGradBorder) {
                gx_out[idx] = 0.0f; gy_out[idx] = 0.0f;
                continue;
            }
            const int i00 = img[(y - 1) * W + (x - 1)], i01 = img[(y - 1) * W + x], i02 = img[(y - 1) * W + (x + 1)];
            const int i10 = img[y * W + (x - 1)],                                    i12 = img[y * W + (x + 1)];
            const int i20 = img[(y + 1) * W + (x - 1)], i21 = img[(y + 1) * W + x], i22 = img[(y + 1) * W + (x + 1)];
            const int gx = (3 * i02 + 10 * i12 + 3 * i22) - (3 * i00 + 10 * i10 + 3 * i20);
            const int gy = (3 * i20 + 10 * i21 + 3 * i22) - (3 * i00 + 10 * i01 + 3 * i02);
            gx_out[idx] = static_cast<float>(gx) / 32.0f;
            gy_out[idx] = static_cast<float>(gy) / 32.0f;
        }
    }
}

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
            double Sxx = 0.0, Syy = 0.0, Sxy = 0.0;   // double accumulator: an independent numerical path from
                                                      // the GPU's float accumulation, not just independent code
            for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
                for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
                    const int widx = (y + wy) * W + (x + wx);
                    const double gxv = gx[widx], gyv = gy[widx];
                    Sxx += gxv * gxv; Syy += gyv * gyv; Sxy += gxv * gyv;
                }
            }
            sxx_out[idx] = static_cast<float>(Sxx);
            syy_out[idx] = static_cast<float>(Syy);
            sxy_out[idx] = static_cast<float>(Sxy);
            const double half_trace = 0.5 * (Sxx + Syy);
            const double det = Sxx * Syy - Sxy * Sxy;
            const double disc = std::max(half_trace * half_trace - det, 0.0);
            min_eig_out[idx] = static_cast<float>(half_trace - std::sqrt(disc));
        }
    }
}

static float bilinear_sample_u8_cpu(const uint8_t* img, int W, int H, float x, float y)
{
    x = std::min(std::max(x, 0.0f), static_cast<float>(W - 1));
    y = std::min(std::max(y, 0.0f), static_cast<float>(H - 1));
    const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
    const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    const float i00 = static_cast<float>(img[y0 * W + x0]), i10 = static_cast<float>(img[y0 * W + x1]);
    const float i01 = static_cast<float>(img[y1 * W + x0]), i11 = static_cast<float>(img[y1 * W + x1]);
    const float top = i00 + (i10 - i00) * fx;
    const float bot = i01 + (i11 - i01) * fx;
    return top + (bot - top) * fy;
}

void lk_iterate_cpu(const uint8_t* img0, const uint8_t* img1, int W, int H,
                    const float* gx, const float* gy,
                    const float* sxx, const float* syy, const float* sxy,
                    float* flow_u, float* flow_v)
{
    for (int y = kLkBorder; y < H - kLkBorder; ++y) {
        for (int x = kLkBorder; x < W - kLkBorder; ++x) {
            const int idx = y * W + x;
            const float Sxx = sxx[idx], Syy = syy[idx], Sxy = sxy[idx];
            const float det = Sxx * Syy - Sxy * Sxy;
            if (det < kLkDetEpsilon) continue;
            const float u = flow_u[idx], v = flow_v[idx];
            float bx = 0.0f, by = 0.0f;
            for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
                for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
                    const int widx = (y + wy) * W + (x + wx);
                    const float i1w = bilinear_sample_u8_cpu(img1, W, H, static_cast<float>(x + wx) + u, static_cast<float>(y + wy) + v);
                    const float it = i1w - static_cast<float>(img0[widx]);
                    bx += gx[widx] * it; by += gy[widx] * it;
                }
            }
            float ddu = -(Syy * bx - Sxy * by) / det;
            float ddv = -(-Sxy * bx + Sxx * by) / det;
            ddu = clamp_f(ddu, -kLkMaxStepPerIterPx, kLkMaxStepPerIterPx);
            ddv = clamp_f(ddv, -kLkMaxStepPerIterPx, kLkMaxStepPerIterPx);
            flow_u[idx] = u + ddu; flow_v[idx] = v + ddv;
        }
    }
}

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
                const float top = v00 + (v10 - v00) * wx;
                const float bot = v01 + (v11 - v01) * wx;
                return top + (bot - top) * wy;
            };
            const int fidx = fy * fineW + fx;
            fine_u[fidx] = 2.0f * bilerp(coarse_u);
            fine_v[fidx] = 2.0f * bilerp(coarse_v);
        }
    }
}

void pyramidal_lk_cpu(const uint8_t* img0_full, const uint8_t* img1_full,
                      float* flow_u_out, float* flow_v_out, float* min_eig_out)
{
    std::vector<std::vector<uint8_t>> img0(kNumLevels), img1(kNumLevels);
    std::vector<std::vector<float>> gx(kNumLevels), gy(kNumLevels), sxx(kNumLevels), syy(kNumLevels), sxy(kNumLevels);
    std::vector<std::vector<float>> min_eig(kNumLevels), flow_u(kNumLevels), flow_v(kNumLevels);

    for (int L = 0; L < kNumLevels; ++L) {
        const size_t n = static_cast<size_t>(level_w(L)) * level_h(L);
        img0[L].resize(n); img1[L].resize(n);
        gx[L].resize(n); gy[L].resize(n);
        sxx[L].resize(n); syy[L].resize(n); sxy[L].resize(n);
        min_eig[L].resize(n); flow_u[L].assign(n, 0.0f); flow_v[L].assign(n, 0.0f);
    }
    std::copy(img0_full, img0_full + static_cast<size_t>(kW) * kH, img0[0].begin());
    std::copy(img1_full, img1_full + static_cast<size_t>(kW) * kH, img1[0].begin());
    for (int L = 1; L < kNumLevels; ++L) {
        downsample_area2x_cpu(img0[L - 1].data(), level_w(L - 1), level_h(L - 1), img0[L].data());
        downsample_area2x_cpu(img1[L - 1].data(), level_w(L - 1), level_h(L - 1), img1[L].data());
    }

    for (int L = kNumLevels - 1; L >= 0; --L) {
        const int Wl = level_w(L), Hl = level_h(L);
        scharr_gradient_cpu(img0[L].data(), Wl, Hl, gx[L].data(), gy[L].data());
        structure_tensor_cpu(gx[L].data(), gy[L].data(), Wl, Hl, sxx[L].data(), syy[L].data(), sxy[L].data(), min_eig[L].data());
        for (int it = 0; it < kLkIterationsPerLevel; ++it) {
            lk_iterate_cpu(img0[L].data(), img1[L].data(), Wl, Hl, gx[L].data(), gy[L].data(),
                          sxx[L].data(), syy[L].data(), sxy[L].data(), flow_u[L].data(), flow_v[L].data());
        }
        if (L > 0) {
            upsample_flow_cpu(flow_u[L].data(), flow_v[L].data(), Wl, Hl,
                             flow_u[L - 1].data(), flow_v[L - 1].data(), level_w(L - 1), level_h(L - 1));
        }
    }
    std::copy(flow_u[0].begin(), flow_u[0].end(), flow_u_out);
    std::copy(flow_v[0].begin(), flow_v[0].end(), flow_v_out);
    std::copy(min_eig[0].begin(), min_eig[0].end(), min_eig_out);
}

// ===========================================================================
// MILESTONE 2 — 3-D lifting (independent re-derivation of the same guard;
// note the CODE SHAPE differs from kernels.cu's kernel: this version reads
// as "compute both hazards, then decide", not "return early on the first
// hazard found" — a genuinely different control flow reaching the same
// result, not a transcription).
// ===========================================================================

static void backproject_cpu(float px, float py, float depth, float out[3])
{
    out[0] = ((px + 0.5f - kCx) / kFx) * depth;
    out[1] = ((py + 0.5f - kCy) / kFy) * depth;
    out[2] = depth;
}

void lift_scene_flow_cpu(const float* flow_u, const float* flow_v, const float* confidence,
                         const float* d0, const float* d1,
                         float* P1_out, float* P2_out, uint8_t* valid_out)
{
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int idx = y * kW + x;
            P1_out[3 * idx + 0] = P1_out[3 * idx + 1] = P1_out[3 * idx + 2] = 0.0f;
            P2_out[3 * idx + 0] = P2_out[3 * idx + 1] = P2_out[3 * idx + 2] = 0.0f;

            const bool source_ok = (d0[idx] != kInvalidDepth) && (confidence[idx] >= kMinConfidenceForLift);

            const float qx = static_cast<float>(x) + flow_u[idx];
            const float qy = static_cast<float>(y) + flow_v[idx];
            const bool in_bounds = (qx >= 0.0f && qx <= static_cast<float>(kW - 1) &&
                                    qy >= 0.0f && qy <= static_cast<float>(kH - 1));

            bool taps_ok = false, edge_ok = false;
            float d1v = 0.0f;
            if (in_bounds) {
                const int x0 = static_cast<int>(std::floor(qx)), y0 = static_cast<int>(std::floor(qy));
                const int x1 = std::min(x0 + 1, kW - 1), y1 = std::min(y0 + 1, kH - 1);
                const float d00 = d1[y0 * kW + x0], d10 = d1[y0 * kW + x1];
                const float d01 = d1[y1 * kW + x0], d11 = d1[y1 * kW + x1];
                taps_ok = (d00 != kInvalidDepth) && (d10 != kInvalidDepth) && (d01 != kInvalidDepth) && (d11 != kInvalidDepth);
                if (taps_ok) {
                    const float dmin = std::min(std::min(d00, d10), std::min(d01, d11));
                    const float dmax = std::max(std::max(d00, d10), std::max(d01, d11));
                    edge_ok = (dmax - dmin) <= kDepthEdgeGuardM;
                    if (edge_ok) {
                        const float fx = qx - static_cast<float>(x0), fy = qy - static_cast<float>(y0);
                        const float top = d00 + (d10 - d00) * fx;
                        const float bot = d01 + (d11 - d01) * fx;
                        d1v = top + (bot - top) * fy;
                    }
                }
            }

            if (source_ok && in_bounds && taps_ok && edge_ok) {
                float P1[3], P2[3];
                backproject_cpu(static_cast<float>(x), static_cast<float>(y), d0[idx], P1);
                backproject_cpu(qx, qy, d1v, P2);
                P1_out[3 * idx + 0] = P1[0]; P1_out[3 * idx + 1] = P1[1]; P1_out[3 * idx + 2] = P1[2];
                P2_out[3 * idx + 0] = P2[0]; P2_out[3 * idx + 1] = P2[1]; P2_out[3 * idx + 2] = P2[2];
                valid_out[idx] = 1u;
            } else {
                valid_out[idx] = 0u;
            }
        }
    }
}

// ===========================================================================
// MILESTONE 3 — residual + weighted covariance accumulation.
// ===========================================================================

void compute_residuals_cpu(int n, const float* P1, const float* P2, const uint8_t* valid, const Rigid3& T,
                           float* residual_vec_out, float* residual_mag_out)
{
    for (int i = 0; i < n; ++i) {
        if (!valid[i]) {
            residual_vec_out[3 * i + 0] = residual_vec_out[3 * i + 1] = residual_vec_out[3 * i + 2] = 0.0f;
            residual_mag_out[i] = 0.0f;
            continue;
        }
        const float p1[3] = { P1[3 * i], P1[3 * i + 1], P1[3 * i + 2] };
        float tp1[3];
        apply_rigid(T, p1, tp1);
        const double rx = static_cast<double>(tp1[0]) - P2[3 * i + 0];
        const double ry = static_cast<double>(tp1[1]) - P2[3 * i + 1];
        const double rz = static_cast<double>(tp1[2]) - P2[3 * i + 2];
        residual_vec_out[3 * i + 0] = static_cast<float>(rx);
        residual_vec_out[3 * i + 1] = static_cast<float>(ry);
        residual_vec_out[3 * i + 2] = static_cast<float>(rz);
        residual_mag_out[i] = static_cast<float>(std::sqrt(rx * rx + ry * ry + rz * rz));
    }
}

// weighted_covariance_accumulate_cpu — the direct (no block-partial staging
// needed) double-precision twin of weighted_covariance_reduce_kernel. Same
// kCovarWidth=16 layout (kernels.cuh), accumulated in ONE pass with a plain
// running sum — the CPU has no reduction TREE to independently re-derive,
// only the per-point CONTRIBUTION formula, which is written fresh here.
void weighted_covariance_accumulate_cpu(int n, const float* P1, const float* P2, const float* weight,
                                        double out16[16])
{
    for (int k = 0; k < 16; ++k) out16[k] = 0.0;
    for (int i = 0; i < n; ++i) {
        const double w = weight[i];
        if (w <= 0.0) continue;
        const double p1x = P1[3 * i], p1y = P1[3 * i + 1], p1z = P1[3 * i + 2];
        const double p2x = P2[3 * i], p2y = P2[3 * i + 1], p2z = P2[3 * i + 2];
        out16[0] += w;
        out16[1] += w * p1x; out16[2] += w * p1y; out16[3] += w * p1z;
        out16[4] += w * p2x; out16[5] += w * p2y; out16[6] += w * p2z;
        out16[7]  += w * p1x * p2x; out16[8]  += w * p1x * p2y; out16[9]  += w * p1x * p2z;
        out16[10] += w * p1y * p2x; out16[11] += w * p1y * p2y; out16[12] += w * p1y * p2z;
        out16[13] += w * p1z * p2x; out16[14] += w * p1z * p2y; out16[15] += w * p1z * p2z;
    }
}

// ===========================================================================
// MILESTONE 4 — residual segmentation.
// ===========================================================================

void threshold_mask_cpu(int n, const float* residual_mag, const uint8_t* valid, float threshold_m,
                        uint8_t* mask_out)
{
    for (int i = 0; i < n; ++i)
        mask_out[i] = (valid[i] && residual_mag[i] > threshold_m) ? 1u : 0u;
}

void erode3x3_cpu(const uint8_t* in, int W, int H, uint8_t* out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t v = 1u;
            for (int dy = -1; dy <= 1 && v; ++dy) {
                for (int dx = -1; dx <= 1 && v; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
                    if (!nb) v = 0u;
                }
            }
            out[y * W + x] = v;
        }
    }
}

void dilate3x3_cpu(const uint8_t* in, int W, int H, uint8_t* out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t v = 0u;
            for (int dy = -1; dy <= 1 && !v; ++dy) {
                for (int dx = -1; dx <= 1 && !v; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
                    if (nb) v = 1u;
                }
            }
            out[y * W + x] = v;
        }
    }
}

void morphological_open_cpu(uint8_t* mask_inout)
{
    std::vector<uint8_t> scratch(static_cast<size_t>(kPixels));
    erode3x3_cpu(mask_inout, kW, kH, scratch.data());
    dilate3x3_cpu(scratch.data(), kW, kH, mask_inout);
}

// ===========================================================================
// MILESTONE 4b — connected-component labeling + size filter (kernels.cuh's
// Milestone-4b constants block; independence ruling in this file's header).
// ===========================================================================

// uf_find / uf_union_toward_smaller — classic union-find with path-halving
// and "attach the larger root under the smaller" (so every component's root
// converges to its MINIMUM linear pixel index — the exact canonical-label
// convention the GPU's label propagation independently converges to; see
// kernels.cu's ccl_propagate_sweep_kernel header for that proof). 01.06's
// uf_find/uf_union_toward_smaller, cited; re-typed fresh here.
static int uf_find(std::vector<int>& parent, int x)
{
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];   // path-halving: point at grandparent while walking to the root
        x = parent[x];
    }
    return x;
}
static void uf_union_toward_smaller(std::vector<int>& parent, int a, int b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (a < b) parent[b] = a; else parent[a] = b;
}

// connected_components_cpu — classic Rosenfeld two-pass union-find, a
// DELIBERATELY different ALGORITHM from the GPU's iterative label
// propagation (see this function's declaration in kernels.cuh): pass 1
// unions every foreground pixel with its WEST and NORTH foreground
// neighbors (EAST/SOUTH get swept up when those pixels run their own
// west/north union — the standard two-neighbor Rosenfeld scan); pass 2
// resolves every foreground pixel to its component's canonical root.
void connected_components_cpu(const uint8_t* mask, int* label, int W, int H)
{
    std::vector<int> parent(static_cast<size_t>(W) * H);
    for (int i = 0; i < W * H; ++i) parent[i] = i;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (!mask[i]) continue;
            if (x > 0 && mask[i - 1]) uf_union_toward_smaller(parent, i, i - 1);
            if (y > 0 && mask[i - W]) uf_union_toward_smaller(parent, i, i - W);
        }
    }
    for (int i = 0; i < W * H; ++i)
        label[i] = mask[i] ? uf_find(parent, i) : kLabelNone;
}

// component_size_filter_cpu — the CPU twin of launch_component_size_filter:
// a direct sequential count-then-filter (no atomics needed for a single
// thread; 01.06's build_candidates_cpu makes the identical simplification
// for its own atomic-scatter GPU counterpart, cited).
void component_size_filter_cpu(const uint8_t* mask_in, const int* label, int min_size_px,
                               uint8_t* mask_out, int n)
{
    std::vector<int> size(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        if (mask_in[i]) size[label[i]] += 1;
    for (int i = 0; i < n; ++i)
        mask_out[i] = (mask_in[i] && size[label[i]] >= min_size_px) ? 1u : 0u;
}
