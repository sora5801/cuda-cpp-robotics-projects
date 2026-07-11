// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.23
//                     Full RAW->RGB ISP: black level -> lens shading ->
//                     defect correction -> white balance -> demosaic
//                     (MHC + bilinear) -> CCM -> gamma
//
// WHY does a GPU repository ship a CPU implementation of everything?
// (CLAUDE.md section 5, restated briefly — see sibling 01.01's reference_cpu.cpp
// for the full essay this project follows verbatim):
//   1) It is the CORRECTNESS ORACLE main.cu's VERIFY step diffs the GPU
//      result against, stage by stage.
//   2) It is the TEACHING BASELINE: every "for each pixel" loop below is
//      exactly what kernels.cu's "each thread owns one pixel" parallelizes.
//
// Independence ruling applied to THIS file (mirrors 01.01's precedent):
//   * SHARED (kernels.cuh): bayer_phase_at(), phase_to_wb_channel(),
//     shading_gain_at(), srgb_encode()/srgb_decode(), ccm_apply_at(), and
//     the four kMhc* coefficient TABLES — documented "hardware fact" /
//     published-constant data, not the algorithm under test. Re-typing a
//     3x3 matrix-vector product or a 100-number published coefficient table
//     a second time would not exercise a different idea, only risk a
//     transcription slip that makes the oracle lie.
//   * INDEPENDENT (this file): the neighbor-clamping, the black-level/
//     saturation arithmetic, the median-of-4 sorting network, the defect-
//     list membership scan, the bilinear demosaic neighbor logic, the MHC
//     stencil TAP-GATHERING and phase-selection branching (which table for
//     which case — the "algorithm" built ON TOP of the shared tables), the
//     fused-kernel re-derivation, and the whole AWB reduction are ALL typed
//     a SECOND time below, from scratch, deliberately not calling anything
//     in kernels.cu. Any GPU-vs-CPU mismatch main.cu reports is therefore a
//     real bug in one of the two independent implementations.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <algorithm>   // std::sort (median-of-4, host side — no sorting-network discipline needed here)
#include <cmath>       // std::fabs

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

// median4_cpu — the CPU twin of kernels.cu's median4() sorting network.
// Independently typed as a plain 4-element std::sort + average-the-middle-
// two — CPU code has no reason to hand-roll a compare-exchange network the
// way the GPU kernel does (that discipline exists there to avoid a
// per-thread branch-heavy library call; here std::sort on 4 elements is
// simple, correct, and exactly as fast as anything else at this scale).
static inline float median4_cpu(float a, float b, float c, float d)
{
    float v[4] = { a, b, c, d };
    std::sort(v, v + 4);
    return 0.5f * (v[1] + v[2]);
}

// ===========================================================================
// STAGE 1 — black level + saturation handling (independent twin of
// black_level_kernel).
// ===========================================================================
void black_level_cpu(const uint16_t* raw, float* out, int W, int H)
{
    const int n = W * H;
    for (int i = 0; i < n; ++i) {
        const int r = static_cast<int>(raw[i]);
        const float above_black = r > kBlackLevel ? static_cast<float>(r - kBlackLevel) : 0.0f;
        float norm = above_black / static_cast<float>(kSatRange);
        if (norm > 1.0f) norm = 1.0f;
        out[i] = norm;
    }
}

// ===========================================================================
// STAGE 2 — lens shading correction (independent twin of lens_shading_kernel).
// ===========================================================================
void lens_shading_cpu(const float* in, float* out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            float gain = shading_gain_at(x, y);
            if (gain < kShadeGainFloor) gain = kShadeGainFloor;
            out[i] = in[i] / gain;
        }
    }
}

// ===========================================================================
// STAGE 3 — defective pixel correction (independent twin of defect_correct_kernel).
// ===========================================================================
void defect_correct_cpu(const float* in, float* out, int W, int H,
                        const int* defect_x, const int* defect_y, int defect_count)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            bool is_defect = false;
            for (int k = 0; k < defect_count; ++k) {
                if (defect_x[k] == x && defect_y[k] == y) { is_defect = true; break; }
            }
            if (!is_defect) { out[i] = in[i]; continue; }

            const int xm = clampi_cpu(x - 2, 0, W - 1), xp = clampi_cpu(x + 2, 0, W - 1);
            const int ym = clampi_cpu(y - 2, 0, H - 1), yp = clampi_cpu(y + 2, 0, H - 1);
            const float n_ = in[ym * W + x], s_ = in[yp * W + x];
            const float e_ = in[y * W + xp], w_ = in[y * W + xm];
            out[i] = median4_cpu(n_, s_, e_, w_);
        }
    }
}

// ===========================================================================
// STAGE 4 — white balance (independent twin of white_balance_kernel).
// ===========================================================================
void white_balance_cpu(const float* in, float* out, int W, int H,
                       float gain_r, float gain_g, float gain_b)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            const int wbch = phase_to_wb_channel(bayer_phase_at(x, y));
            const float gain = wbch == 0 ? gain_r : (wbch == 2 ? gain_b : gain_g);
            out[i] = in[i] * gain;
        }
    }
}

// ===========================================================================
// FUSED stages 1-4 (independent twin of fused_bl_shading_defect_wb_kernel).
// Same "recompute black-level+shading inline instead of reading a
// materialized buffer" structure as the GPU kernel, retyped from scratch.
// ===========================================================================
static float bl_shading_at_cpu(const uint16_t* raw, int x, int y, int W, int H)
{
    x = clampi_cpu(x, 0, W - 1);
    y = clampi_cpu(y, 0, H - 1);
    const int r = static_cast<int>(raw[y * W + x]);
    const float above_black = r > kBlackLevel ? static_cast<float>(r - kBlackLevel) : 0.0f;
    float norm = above_black / static_cast<float>(kSatRange);
    if (norm > 1.0f) norm = 1.0f;
    float gain = shading_gain_at(x, y);
    if (gain < kShadeGainFloor) gain = kShadeGainFloor;
    return norm / gain;
}

void fused_bl_shading_defect_wb_cpu(const uint16_t* raw, float* out, int W, int H,
                                    const int* defect_x, const int* defect_y, int defect_count,
                                    float gain_r, float gain_g, float gain_b)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            bool is_defect = false;
            for (int k = 0; k < defect_count; ++k) {
                if (defect_x[k] == x && defect_y[k] == y) { is_defect = true; break; }
            }
            float bl_sh;
            if (!is_defect) {
                bl_sh = bl_shading_at_cpu(raw, x, y, W, H);
            } else {
                const float n_ = bl_shading_at_cpu(raw, x, y - 2, W, H);
                const float s_ = bl_shading_at_cpu(raw, x, y + 2, W, H);
                const float e_ = bl_shading_at_cpu(raw, x + 2, y, W, H);
                const float w_ = bl_shading_at_cpu(raw, x - 2, y, W, H);
                bl_sh = median4_cpu(n_, s_, e_, w_);
            }
            const int wbch = phase_to_wb_channel(bayer_phase_at(x, y));
            const float gain = wbch == 0 ? gain_r : (wbch == 2 ? gain_b : gain_g);
            out[i] = bl_sh * gain;
        }
    }
}

// ===========================================================================
// AWB statistics (independent twin of awb_stats_block_kernel +
// awb_finalize_kernel, collapsed into ONE single-pass sequential
// accumulation — the CPU needs none of the GPU's block-then-finalize dance,
// same simplification 01.01's normalize_stats_cpu makes).
// ===========================================================================
void awb_stats_cpu(const float* in, int W, int H, double sum3[3], float max3[3])
{
    sum3[0] = sum3[1] = sum3[2] = 0.0;
    max3[0] = max3[1] = max3[2] = 0.0f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float v = in[y * W + x];
            const int wbch = phase_to_wb_channel(bayer_phase_at(x, y));
            sum3[wbch] += static_cast<double>(v);
            if (v > max3[wbch]) max3[wbch] = v;
        }
    }
}

void awb_gains_from_stats_cpu(const double sum3[3], const float max3[3], int W, int H,
                              float gray_gain3[3], float white_gain3[3])
{
    const double countR = static_cast<double>(W) * H / 4.0;   // RGGB: R = B = W*H/4, G = W*H/2
    const double countG = static_cast<double>(W) * H / 2.0;
    const double countB = countR;
    const double meanR = sum3[0] / countR, meanG = sum3[1] / countG, meanB = sum3[2] / countB;

    gray_gain3[1] = 1.0f;
    gray_gain3[0] = static_cast<float>(meanG / (meanR > 1e-8 ? meanR : 1e-8));
    gray_gain3[2] = static_cast<float>(meanG / (meanB > 1e-8 ? meanB : 1e-8));

    white_gain3[1] = 1.0f;
    white_gain3[0] = max3[1] / (max3[0] > 1e-6f ? max3[0] : 1e-6f);
    white_gain3[2] = max3[1] / (max3[2] > 1e-6f ? max3[2] : 1e-6f);
}

// ===========================================================================
// STAGE 5 — demosaic (independent twins). Bilinear: the same four-case
// distance-1 averaging as kernels.cu's baseline, retyped. MHC: the tap-
// gathering and phase-selection logic retyped from scratch, reading the
// SHARED kMhc* tables (kernels.cuh section 3 — the documented, permitted
// exception; see this file's header).
// ===========================================================================
void demosaic_bilinear_cpu(const float* mosaic, float* rgb, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            const int xm = clampi_cpu(x - 1, 0, W - 1), xp = clampi_cpu(x + 1, 0, W - 1);
            const int ym = clampi_cpu(y - 1, 0, H - 1), yp = clampi_cpu(y + 1, 0, H - 1);
            const float n_ = mosaic[ym * W + x], s_ = mosaic[yp * W + x];
            const float e_ = mosaic[y * W + xp], w_ = mosaic[y * W + xm];
            const float ne = mosaic[ym * W + xp], nw = mosaic[ym * W + xm];
            const float se = mosaic[yp * W + xp], sw = mosaic[yp * W + xm];
            const float center = mosaic[i];

            float R, G, B;
            const int phase = bayer_phase_at(x, y);
            if (phase == 0) {
                R = center; G = 0.25f * (n_ + s_ + e_ + w_); B = 0.25f * (ne + nw + se + sw);
            } else if (phase == 3) {
                B = center; G = 0.25f * (n_ + s_ + e_ + w_); R = 0.25f * (ne + nw + se + sw);
            } else if (phase == 1) {
                G = center; R = 0.5f * (e_ + w_); B = 0.5f * (n_ + s_);
            } else {
                G = center; B = 0.5f * (e_ + w_); R = 0.5f * (n_ + s_);
            }
            rgb[i * 3 + 0] = R; rgb[i * 3 + 1] = G; rgb[i * 3 + 2] = B;
        }
    }
}

// mhc_eval_cpu — the CPU twin of kernels.cu's mhc_eval() device function:
// same 5x5 tap gather, retyped as a plain double loop (no #pragma unroll —
// the compiler is free to unroll a 5x5 loop on its own; nothing here
// depends on it doing so).
static float mhc_eval_cpu(const float* mosaic, int x, int y, int W, int H, const float weights[kMhcTaps])
{
    float acc = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const float w = weights[(dy + 2) * 5 + (dx + 2)];
            if (w == 0.0f) continue;
            const int nx = clampi_cpu(x + dx, 0, W - 1);
            const int ny = clampi_cpu(y + dy, 0, H - 1);
            acc += w * mosaic[ny * W + nx];
        }
    }
    return acc * 0.125f;
}

void demosaic_mhc_cpu(const float* mosaic, float* rgb, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            const int phase = bayer_phase_at(x, y);
            const float native = mosaic[i];

            float R, G, B;
            if (phase == 0) {
                R = native;
                G = mhc_eval_cpu(mosaic, x, y, W, H, kMhcG);
                B = mhc_eval_cpu(mosaic, x, y, W, H, kMhcDiag);
            } else if (phase == 3) {
                B = native;
                G = mhc_eval_cpu(mosaic, x, y, W, H, kMhcG);
                R = mhc_eval_cpu(mosaic, x, y, W, H, kMhcDiag);
            } else if (phase == 1) {
                G = native;
                R = mhc_eval_cpu(mosaic, x, y, W, H, kMhcA);
                B = mhc_eval_cpu(mosaic, x, y, W, H, kMhcB);
            } else {
                G = native;
                B = mhc_eval_cpu(mosaic, x, y, W, H, kMhcA);
                R = mhc_eval_cpu(mosaic, x, y, W, H, kMhcB);
            }
            rgb[i * 3 + 0] = R > 0.0f ? R : 0.0f;
            rgb[i * 3 + 1] = G > 0.0f ? G : 0.0f;
            rgb[i * 3 + 2] = B > 0.0f ? B : 0.0f;
        }
    }
}

// ===========================================================================
// STAGE 6 — CCM (calls the SHARED ccm_apply_at() — see file header: this
// five-line matrix-vector product is documented camera-model DATA, the same
// carve-out 01.01 makes for distort_forward()).
// ===========================================================================
void ccm_apply_cpu(const float* rgb_in, float* rgb_out, int W, int H)
{
    const int n = W * H;
    for (int i = 0; i < n; ++i) {
        float or_, og, ob;
        ccm_apply_at(rgb_in[i * 3 + 0], rgb_in[i * 3 + 1], rgb_in[i * 3 + 2], or_, og, ob);
        rgb_out[i * 3 + 0] = or_; rgb_out[i * 3 + 1] = og; rgb_out[i * 3 + 2] = ob;
    }
}

// ===========================================================================
// STAGE 7 — gamma encode (calls the SHARED srgb_encode() — the exact
// standard piecewise function; sharing it is the whole point of "MUST
// MATCH", not a shortcut — see kernels.cuh's file header on why this
// function is HD-shared).
// ===========================================================================
void gamma_encode_cpu(const float* rgb_linear, unsigned char* rgb_srgb8, int W, int H)
{
    const int n = W * H;
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float s = srgb_encode(rgb_linear[i * 3 + c]) * 255.0f;
            if (s < 0.0f) s = 0.0f;
            if (s > 255.0f) s = 255.0f;
            rgb_srgb8[i * 3 + c] = static_cast<unsigned char>(s + 0.5f);
        }
    }
}
