// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.14
//                     (Template matching (NCC) at scale for pick verification)
//
// WHY does a GPU repository ship a CPU implementation of everything? See
// docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's header for the two
// load-bearing reasons (correctness oracle + teaching baseline) and the
// independence ruling this file follows, restated precisely for THIS
// project (kernels.cuh's file header, and project 01.13's precedent):
//
//   * DATA-LAYOUT CONTRACTS (tray/slot geometry, the template statistics
//     table S_t/S_tt) are single-sourced in kernels.cuh and SHARED — both
//     paths must agree on these bit-for-bit or a "disagreement" would just
//     be measuring two different problems, not a real bug.
//   * The ALGORITHMIC CORE of every twinned stage is written TWICE,
//     independently, in the simplest correct C++:
//       - build_integral_images_cpu uses the CLASSIC SINGLE-PASS 2-D
//         recurrence II(x,y) = img(x,y) + II(x-1,y) + II(x,y-1) - II(x-1,y-1)
//         — a genuinely DIFFERENT algorithm from kernels.cu's 2-PASS
//         SEPARABLE SCAN (row scan, then column scan), the same kind of
//         "different path to the same fixed point" independence project
//         01.13's queue-based flood fill achieves against the GPU's
//         synchronous sweep loop.
//       - window_stats_cpu and ncc_scores_cpu re-type the box-query and
//         correlation-sum loops from scratch, reading kernels.cu only to
//         confirm they compute the SAME documented quantity, not by
//         sharing code with it.
//
// What IS and IS NOT twinned in this project (THEORY.md "How we verify
// correctness" has the full table and the measured numbers):
//   TWINNED, BIT-EXACT (integer): the integral images (ii_sum, ii_sumsq) and
//                                   the window statistics (ws_sum, ws_sumsq).
//                                   Every intermediate here is exact integer
//                                   arithmetic — GPU and CPU must match
//                                   EXACTLY, not just closely.
//   TWINNED, float tolerance:     the final NCC score volume (104,040
//                                   evaluations) — the one place a sqrt/divide
//                                   introduces genuine floating-point
//                                   rounding that can differ by up to a few
//                                   ULP between nvcc's device sqrt() and
//                                   MSVC's host sqrt().
//   NOT twinned (single-sourced, downstream analysis, exactly the pattern
//   flagship 08.01 uses for its host-only softmin blend, and 01.13 uses for
//   its peak extraction/alignment solve): slot classification, offset/
//   rotation recovery, and the plain-SSD illumination comparison — all in
//   main.cu, checked by INDEPENDENT gates instead of a twin comparison.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — compare stage by stage.
// ===========================================================================

#include "kernels.cuh"

#include <cstring>   // std::memset

// ===========================================================================
// STAGE 1 — whole-tray integral images (CPU twin of integral_row_scan_kernel
// + integral_col_scan_kernel), via the CLASSIC SINGLE-PASS 2-D recurrence —
// see the file header for why this is a structurally different algorithm
// from the GPU's 2-pass separable scan, not a re-typed copy of it.
//
// Derivation of the recurrence (THEORY.md "The GPU mapping" gives the GPU
// side's inclusion-exclusion proof; this is the equivalent one-shot form):
// II(x,y) counts every pixel with x'<=x, y'<=y. II(x-1,y) already counts
// every such pixel with x'<=x-1 (all of column x's predecessors); II(x,y-1)
// already counts every one with y'<=y-1; both count the (x-1,y-1) rectangle
// TWICE (once from each direction), hence subtracting II(x-1,y-1) once.
// Adding the new pixel img(x,y) itself completes the count. A single
// sequential pass in row-major order visits (x-1,y) and (x,y-1) before
// (x,y) exactly when x,y are visited in increasing order — which a plain
// nested for-loop does automatically.
// ---------------------------------------------------------------------------
void build_integral_images_cpu(const uint8_t* img, uint32_t* ii_sum, uint64_t* ii_sumsq)
{
    // Zero the whole padded table first (row 0 / col 0 stay zero forever —
    // the same convention kernels.cu establishes via cudaMemset).
    std::memset(ii_sum, 0, sizeof(uint32_t) * static_cast<size_t>(II_CELLS));
    std::memset(ii_sumsq, 0, sizeof(uint64_t) * static_cast<size_t>(II_CELLS));

    for (int r = 1; r <= IMG_H; ++r) {
        for (int c = 1; c <= IMG_W; ++c) {
            const uint32_t v = img[static_cast<size_t>(r - 1) * IMG_W + (c - 1)];
            const uint32_t sum_above = ii_sum[ii_index(r - 1, c)];
            const uint32_t sum_left  = ii_sum[ii_index(r, c - 1)];
            const uint32_t sum_diag  = ii_sum[ii_index(r - 1, c - 1)];
            ii_sum[ii_index(r, c)] = v + sum_above + sum_left - sum_diag;

            const uint64_t vv = static_cast<uint64_t>(v) * v;
            const uint64_t sq_above = ii_sumsq[ii_index(r - 1, c)];
            const uint64_t sq_left  = ii_sumsq[ii_index(r, c - 1)];
            const uint64_t sq_diag  = ii_sumsq[ii_index(r - 1, c - 1)];
            ii_sumsq[ii_index(r, c)] = vv + sq_above + sq_left - sq_diag;
        }
    }
}

// ===========================================================================
// STAGE 2 — window statistics (CPU twin of window_stats_kernel). Plain
// triple-nested loop over slot x offset_y x offset_x, each cell an
// independent O(1) box query — re-typed from scratch (not calling any
// kernels.cu code, which the host compiler cannot see anyway).
// ---------------------------------------------------------------------------
void window_stats_cpu(const uint32_t* ii_sum, const uint64_t* ii_sumsq,
                      uint32_t* ws_sum, uint64_t* ws_sumsq)
{
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        for (int oy = 0; oy < NUM_OFFSETS_1D; ++oy) {
            const int dy = oy - SEARCH_RADIUS;
            for (int ox = 0; ox < NUM_OFFSETS_1D; ++ox) {
                const int dx = ox - SEARCH_RADIUS;
                const int x0 = slot_window_x0(slot) + SEARCH_RADIUS + dx;
                const int y0 = slot_window_y0(slot) + SEARCH_RADIUS + dy;
                const int x1 = x0 + TEMPLATE_SIZE, y1 = y0 + TEMPLATE_SIZE;

                const uint32_t S_w = ii_sum[ii_index(y1, x1)] - ii_sum[ii_index(y0, x1)]
                                    - ii_sum[ii_index(y1, x0)] + ii_sum[ii_index(y0, x0)];
                const uint64_t S_ww = ii_sumsq[ii_index(y1, x1)] - ii_sumsq[ii_index(y0, x1)]
                                     - ii_sumsq[ii_index(y1, x0)] + ii_sumsq[ii_index(y0, x0)];

                const long long idx = window_stats_index(slot, oy, ox);
                ws_sum[idx] = S_w;
                ws_sumsq[idx] = S_ww;
            }
        }
    }
}

// ===========================================================================
// STAGE 3 — the full NCC score volume oracle (CPU twin of ALL THREE GPU NCC
// kernels at once — they must all converge on this same value, up to float
// tolerance, regardless of which acceleration trick each one uses). Same
// box-query + direct-correlation structure as the GPU sum-table/shared
// kernels, independently typed here; the combining algebra
// (ncc_from_sums_cpu below) is a separate, from-scratch re-statement of
// kernels.cu's ncc_from_sums, not a shared function — see kernels.cuh
// SECTION 6 for the formula both are computing.
// ---------------------------------------------------------------------------
static float ncc_from_sums_cpu(int64_t S_w, int64_t S_ww, int64_t S_wt, int64_t S_t, int64_t S_tt)
{
    const int64_t N = TEMPLATE_PIXELS;
    const int64_t numerator_unnorm = N * S_wt - S_w * S_t;
    const int64_t var_w_unnorm = N * S_ww - S_w * S_w;
    const int64_t var_t_unnorm = N * S_tt - S_t * S_t;

    const double denom = std::sqrt(static_cast<double>(var_w_unnorm) * static_cast<double>(var_t_unnorm));
    if (denom < static_cast<double>(NCC_DENOM_EPS)) return 0.0f;
    return static_cast<float>(static_cast<double>(numerator_unnorm) / denom);
}

void ncc_scores_cpu(const uint8_t* img, const uint32_t* ii_sum, const uint64_t* ii_sumsq,
                    const uint8_t* templates, const int64_t* S_t, const int64_t* S_tt,
                    float* scores)
{
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        for (int tmpl = 0; tmpl < NUM_TEMPLATES; ++tmpl) {
            const uint8_t* tpl = templates + static_cast<size_t>(tmpl) * TEMPLATE_PIXELS;
            for (int oy = 0; oy < NUM_OFFSETS_1D; ++oy) {
                const int dy = oy - SEARCH_RADIUS;
                for (int ox = 0; ox < NUM_OFFSETS_1D; ++ox) {
                    const int dx = ox - SEARCH_RADIUS;
                    const int x0 = slot_window_x0(slot) + SEARCH_RADIUS + dx;
                    const int y0 = slot_window_y0(slot) + SEARCH_RADIUS + dy;
                    const int x1 = x0 + TEMPLATE_SIZE, y1 = y0 + TEMPLATE_SIZE;

                    const int64_t S_w = ii_sum[ii_index(y1, x1)] - ii_sum[ii_index(y0, x1)]
                                       - ii_sum[ii_index(y1, x0)] + ii_sum[ii_index(y0, x0)];
                    const int64_t S_ww = static_cast<int64_t>(ii_sumsq[ii_index(y1, x1)] - ii_sumsq[ii_index(y0, x1)]
                                        - ii_sumsq[ii_index(y1, x0)] + ii_sumsq[ii_index(y0, x0)]);

                    int64_t S_wt = 0;
                    for (int ty = 0; ty < TEMPLATE_SIZE; ++ty) {
                        const uint8_t* row = img + static_cast<size_t>(y0 + ty) * IMG_W + x0;
                        const uint8_t* trow = tpl + ty * TEMPLATE_SIZE;
                        for (int tx = 0; tx < TEMPLATE_SIZE; ++tx)
                            S_wt += static_cast<int64_t>(row[tx]) * trow[tx];
                    }

                    scores[score_index(slot, tmpl, oy, ox)] =
                        ncc_from_sums_cpu(S_w, S_ww, S_wt, S_t[tmpl], S_tt[tmpl]);
                }
            }
        }
    }
}
