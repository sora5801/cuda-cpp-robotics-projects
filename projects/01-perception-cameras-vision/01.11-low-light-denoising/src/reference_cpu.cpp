// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.11
//                     Low-light denoising (bilateral, non-local means,
//                     BM3D-lite)
//
// Four independent CPU twins (bilateral_cpu, gaussian_blur_cpu, nlm_cpu,
// bm3d_lite_cpu), one per GPU launcher declared in kernels.cuh. Per this
// project's twin-independence ruling (the general statement lives in
// docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's header; kernels.cuh's file
// header restates it for THIS project): the DATA-LAYOUT contracts (image
// geometry, the noise-model formula, BM3D-lite's reference-grid position
// arithmetic) are shared HD code from kernels.cuh, but the ALGORITHMIC CORE
// of every denoiser below — the stencil math, the patch search, the DCT and
// Haar transforms — is retyped HERE independently, with NO shared function
// between this file and kernels.cu, so the GPU-vs-CPU comparison in main.cu
// is not blind to a bug hiding inside any one of them. This file is
// therefore both (a) the correctness oracle main.cu's VERIFY stage checks
// every GPU output against, and (b) the CPU-timing baseline that makes the
// [time] speed-up lines in main.cu legible.
//
// A note on cost: BM3D-lite here is genuinely expensive on one CPU core
// (block matching is O(groups x search_area x patch_area)) — main.cu's
// [time] line reports the measured wall-clock honestly; it is a teaching
// artifact (CLAUDE.md §12), never a benchmark claim.
//
// Read this after: kernels.cu — then compare the two side by side; the
// SHAPE of each function should look familiar, but no code is shared.
// ===========================================================================

#include "kernels.cuh"   // geometry, noise model, BM3D-lite grid layout, and every prototype below

#include <cmath>         // std::exp, std::sqrt, std::cos, std::fabs
#include <vector>
#include <algorithm>     // std::fill

// ---------------------------------------------------------------------------
// clamp_coord — clamp-to-edge border handling, the SAME convention
// kernels.cu's device copy applies (independently re-typed here, per the
// twin-independence ruling — see this file's header).
// ---------------------------------------------------------------------------
static inline int clamp_coord(int v, int n)
{
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

// ===========================================================================
// bilateral_cpu — sequential twin of bilateral_naive_kernel/
// bilateral_tiled_kernel. Same 9x9 window, same spatial+range weight
// formula, same fixed dy-outer/dx-inner accumulation order (not load-
// bearing for correctness on the CPU side, but keeping it lets a reader
// diff this loop against kernels.cu's line by line).
// ===========================================================================
void bilateral_cpu(const float* img, int W, int H, float* out)
{
    const float inv2ss = 1.0f / (2.0f * kBilateralSigmaSpatial * kBilateralSigmaSpatial);
    const float inv2sr = 1.0f / (2.0f * kBilateralSigmaRange * kBilateralSigmaRange);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float center = img[y * W + x];
            float wsum = 0.0f, vsum = 0.0f;
            for (int dy = -kBilateralRadius; dy <= kBilateralRadius; ++dy) {
                const int sy = clamp_coord(y + dy, H);
                for (int dx = -kBilateralRadius; dx <= kBilateralRadius; ++dx) {
                    const int sx = clamp_coord(x + dx, W);
                    const float v = img[sy * W + sx];
                    const float spatial_term = -static_cast<float>(dx * dx + dy * dy) * inv2ss;
                    const float diff = v - center;
                    const float range_term = -(diff * diff) * inv2sr;
                    const float w = std::exp(spatial_term + range_term);
                    wsum += w;
                    vsum += w * v;
                }
            }
            out[y * W + x] = vsum / wsum;
        }
    }
}

// ===========================================================================
// gaussian_blur_cpu — sequential twin of gaussian_blur_kernel (the negative
// control): the same 9x9 spatial-Gaussian weights, no range term at all.
// ===========================================================================
void gaussian_blur_cpu(const float* img, int W, int H, float* out)
{
    const float inv2ss = 1.0f / (2.0f * kBilateralSigmaSpatial * kBilateralSigmaSpatial);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float wsum = 0.0f, vsum = 0.0f;
            for (int dy = -kBilateralRadius; dy <= kBilateralRadius; ++dy) {
                const int sy = clamp_coord(y + dy, H);
                for (int dx = -kBilateralRadius; dx <= kBilateralRadius; ++dx) {
                    const int sx = clamp_coord(x + dx, W);
                    const float v = img[sy * W + sx];
                    const float w = std::exp(-static_cast<float>(dx * dx + dy * dy) * inv2ss);
                    wsum += w;
                    vsum += w * v;
                }
            }
            out[y * W + x] = vsum / wsum;
        }
    }
}

// ===========================================================================
// nlm_cpu — sequential twin of nlm_kernel: 5x5 patch, 13x13 search window,
// mean-squared patch distance weighted by exp(-d/h^2). The slow one on
// purpose — main.cu's [time] line is what makes "the expensive kernel"
// (README's words) a measured fact rather than an assertion.
// ===========================================================================
void nlm_cpu(const float* img, int W, int H, float* out)
{
    const int PR = kNlmPatchRadius;
    const int SR = kNlmSearchRadius;
    const float patchN = static_cast<float>((2 * PR + 1) * (2 * PR + 1));
    const float invH2 = 1.0f / (kNlmH * kNlmH);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float wsum = 0.0f, vsum = 0.0f;
            for (int oy = -SR; oy <= SR; ++oy) {
                const int qy = clamp_coord(y + oy, H);
                for (int ox = -SR; ox <= SR; ++ox) {
                    const int qx = clamp_coord(x + ox, W);

                    float ssd = 0.0f;
                    for (int py = -PR; py <= PR; ++py) {
                        const int ay = clamp_coord(y + py, H), by = clamp_coord(qy + py, H);
                        for (int px = -PR; px <= PR; ++px) {
                            const int ax = clamp_coord(x + px, W), bx = clamp_coord(qx + px, W);
                            const float diff = img[ay * W + ax] - img[by * W + bx];
                            ssd += diff * diff;
                        }
                    }
                    const float patch_dist = ssd / patchN;
                    const float w = std::exp(-patch_dist * invH2);
                    const float qv = img[qy * W + qx];
                    wsum += w;
                    vsum += w * qv;
                }
            }
            out[y * W + x] = vsum / wsum;
        }
    }
}

// ===========================================================================
// BM3D-lite CPU helpers — independent re-derivations of kernels.cu's
// dct8_basis/dct2d_forward/dct2d_inverse/haar_forward16/haar_inverse16
// (same textbook formulas, retyped from scratch in std:: functions instead
// of the device *f intrinsics — the twin-independence ruling, this file's
// header).
// ===========================================================================
static void dct8_basis_cpu(float basis[8][8])
{
    const double kPi = 3.14159265358979323846;
    for (int k = 0; k < 8; ++k) {
        const double alpha = (k == 0) ? std::sqrt(1.0 / 8.0) : std::sqrt(2.0 / 8.0);
        for (int n = 0; n < 8; ++n)
            basis[k][n] = static_cast<float>(alpha * std::cos(kPi / 8.0 * (n + 0.5) * k));
    }
}

static void dct2d_forward_cpu(const float basis[8][8], float p[8][8])
{
    float tmp[8][8];
    for (int u = 0; u < 8; ++u)
        for (int col = 0; col < 8; ++col) {
            float s = 0.0f;
            for (int n = 0; n < 8; ++n) s += basis[u][n] * p[n][col];
            tmp[u][col] = s;
        }
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float s = 0.0f;
            for (int n = 0; n < 8; ++n) s += tmp[u][n] * basis[v][n];
            p[u][v] = s;
        }
}

static void dct2d_inverse_cpu(const float basis[8][8], float p[8][8])
{
    float tmp[8][8];
    for (int row = 0; row < 8; ++row)
        for (int v = 0; v < 8; ++v) {
            float s = 0.0f;
            for (int u = 0; u < 8; ++u) s += basis[u][row] * p[u][v];
            tmp[row][v] = s;
        }
    for (int row = 0; row < 8; ++row)
        for (int col = 0; col < 8; ++col) {
            float s = 0.0f;
            for (int v = 0; v < 8; ++v) s += tmp[row][v] * basis[v][col];
            p[row][col] = s;
        }
}

static void haar_forward16_cpu(float v[16])
{
    float tmp[16];
    for (int len = 16; len > 1; len /= 2) {
        const int half = len / 2;
        for (int i = 0; i < half; ++i) {
            const float a = v[2 * i], b = v[2 * i + 1];
            tmp[i] = (a + b) * 0.70710678118654752f;
            tmp[half + i] = (a - b) * 0.70710678118654752f;
        }
        for (int i = 0; i < len; ++i) v[i] = tmp[i];
    }
}

static void haar_inverse16_cpu(float v[16])
{
    float tmp[16];
    for (int len = 2; len <= 16; len *= 2) {
        const int half = len / 2;
        for (int i = 0; i < half; ++i) {
            const float a = v[i], d = v[half + i];
            tmp[2 * i] = (a + d) * 0.70710678118654752f;
            tmp[2 * i + 1] = (a - d) * 0.70710678118654752f;
        }
        for (int i = 0; i < len; ++i) v[i] = tmp[i];
    }
}

// ===========================================================================
// bm3d_lite_cpu — sequential twin of bm3d_group_kernel + bm3d_finalize_
// kernel, fused into one pass over reference groups followed by one
// finalize pass — SAME two-stage aggregation shape (accumulate into
// out_sum/out_weight, then divide), but summed in a FIXED, deterministic
// group order (group 0, 1, 2, ... in raster order over the reference
// grid) instead of the GPU's unordered atomicAdd interleaving. That
// difference in summation ORDER (not in the algorithm) is exactly why
// main.cu's VERIFY tolerance for this method is the loosest of the four
// (THEORY.md "Numerical considerations" quantifies it).
// ===========================================================================
void bm3d_lite_cpu(const float* img, int W, int H, float* out)
{
    const int numX = bm3d_num_positions(W);
    const int numY = bm3d_num_positions(H);
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H);

    std::vector<double> sum(n, 0.0);      // double accumulators: this file's oracle can afford the
    std::vector<double> weight(n, 0.0);   // extra precision the GPU's atomicAdd(float) cannot (see below)

    float basis[8][8];
    dct8_basis_cpu(basis);

    for (int iy = 0; iy < numY; ++iy) {
        const int gy = bm3d_position(iy, H);
        for (int ix = 0; ix < numX; ++ix) {
            const int gx = bm3d_position(ix, W);

            // ---- block matching: top-kBm3dStackSize by ascending SSD -----
            float best_ssd[kBm3dStackSize];
            int best_cx[kBm3dStackSize], best_cy[kBm3dStackSize];
            for (int i = 0; i < kBm3dStackSize; ++i) best_ssd[i] = 3.402823466e38f;   // FLT_MAX, spelled out (no <cfloat> dependency)

            for (int oy = -kBm3dSearchRadius; oy <= kBm3dSearchRadius; ++oy) {
                const int cy = clamp_coord(gy + oy, H - kBm3dPatch + 1);
                for (int ox = -kBm3dSearchRadius; ox <= kBm3dSearchRadius; ++ox) {
                    const int cx = clamp_coord(gx + ox, W - kBm3dPatch + 1);

                    float ssd = 0.0f;
                    for (int r = 0; r < kBm3dPatch; ++r)
                        for (int c = 0; c < kBm3dPatch; ++c) {
                            const float diff = img[(gy + r) * W + (gx + c)] - img[(cy + r) * W + (cx + c)];
                            ssd += diff * diff;
                        }

                    if (ssd < best_ssd[kBm3dStackSize - 1]) {
                        int slot = kBm3dStackSize - 1;
                        while (slot > 0 && best_ssd[slot - 1] > ssd) {
                            best_ssd[slot] = best_ssd[slot - 1];
                            best_cx[slot] = best_cx[slot - 1];
                            best_cy[slot] = best_cy[slot - 1];
                            --slot;
                        }
                        best_ssd[slot] = ssd;
                        best_cx[slot] = cx;
                        best_cy[slot] = cy;
                    }
                }
            }

            // ---- gather -----------------------------------------------------
            float stack[kBm3dStackSize][kBm3dPatch][kBm3dPatch];
            for (int p = 0; p < kBm3dStackSize; ++p)
                for (int r = 0; r < kBm3dPatch; ++r)
                    for (int c = 0; c < kBm3dPatch; ++c)
                        stack[p][r][c] = img[(best_cy[p] + r) * W + (best_cx[p] + c)];

            // ---- forward transform: 2-D DCT per patch, then 1-D Haar
            // across the stack. ------------------------------------------
            for (int p = 0; p < kBm3dStackSize; ++p) dct2d_forward_cpu(basis, stack[p]);
            for (int u = 0; u < kBm3dPatch; ++u)
                for (int v = 0; v < kBm3dPatch; ++v) {
                    float vec[kBm3dStackSize];
                    for (int p = 0; p < kBm3dStackSize; ++p) vec[p] = stack[p][u][v];
                    haar_forward16_cpu(vec);
                    for (int p = 0; p < kBm3dStackSize; ++p) stack[p][u][v] = vec[p];
                }

            // ---- hard threshold ----------------------------------------
            int nonzero = 0;
            for (int p = 0; p < kBm3dStackSize; ++p)
                for (int u = 0; u < kBm3dPatch; ++u)
                    for (int v = 0; v < kBm3dPatch; ++v) {
                        if (std::fabs(stack[p][u][v]) < kBm3dThreshold) stack[p][u][v] = 0.0f;
                        else ++nonzero;
                    }

            // ---- inverse transform ---------------------------------------
            for (int u = 0; u < kBm3dPatch; ++u)
                for (int v = 0; v < kBm3dPatch; ++v) {
                    float vec[kBm3dStackSize];
                    for (int p = 0; p < kBm3dStackSize; ++p) vec[p] = stack[p][u][v];
                    haar_inverse16_cpu(vec);
                    for (int p = 0; p < kBm3dStackSize; ++p) stack[p][u][v] = vec[p];
                }
            for (int p = 0; p < kBm3dStackSize; ++p) dct2d_inverse_cpu(basis, stack[p]);

            // ---- accumulate, FIXED order (this group, raster scan over
            // its 16 patches x 64 pixels) — the CPU's determinism, absent
            // on the GPU's atomic path (see this function's header). -----
            const double w = 1.0 / (1.0 + static_cast<double>(nonzero));
            for (int p = 0; p < kBm3dStackSize; ++p)
                for (int r = 0; r < kBm3dPatch; ++r)
                    for (int c = 0; c < kBm3dPatch; ++c) {
                        const int px = best_cx[p] + c, py = best_cy[p] + r;
                        const size_t idx = static_cast<size_t>(py) * W + px;
                        sum[idx] += w * static_cast<double>(stack[p][r][c]);
                        weight[idx] += w;
                    }
        }
    }

    for (size_t i = 0; i < n; ++i)
        out[i] = (weight[i] > 1e-9) ? static_cast<float>(sum[i] / weight[i]) : img[i];
}
