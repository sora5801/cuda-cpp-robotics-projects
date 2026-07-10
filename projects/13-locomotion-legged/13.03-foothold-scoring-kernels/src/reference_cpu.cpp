// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 13.03
//                     (Foothold scoring kernels: slope, roughness, edge
//                     distance from elevation maps)
//
// Four oracle twins, one per GPU kernel in kernels.cu — same math, same
// window shapes, same tie-break rules, sequential over cells/queries
// instead of one-thread-per-item. Deliberate, documented duplication
// (CLAUDE.md §5): this file never includes kernels.cu, and it is compiled
// by the HOST compiler (cl.exe), never nvcc — the __CUDACC__ fence in
// kernels.cuh keeps every __global__ declaration invisible to it.
//
// Two jobs, as always in this repo:
//   1) CORRECTNESS ORACLE — main.cu's VERIFY stage feeds each GPU kernel
//      and its CPU twin here the SAME upstream inputs (see main.cu's
//      "stage-isolated" verification strategy) and diffs the outputs.
//   2) TEACHING BASELINE — reading this file next to kernels.cu shows
//      exactly what changed under parallelization: the outer loop over
//      cells/queries became "one thread owns one iteration"; the per-item
//      body is close to line-for-line identical (std::isnan/std::sqrt here,
//      isnan/sqrtf there).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>       // std::isnan, std::sqrt, std::atan, std::fabs, std::round, std::ceil
#include <limits>       // std::numeric_limits<float>::quiet_NaN()

namespace {
constexpr float kNaN() { return std::numeric_limits<float>::quiet_NaN(); }

// ---------------------------------------------------------------------------
// solve_plane_3x3 — HOST twin of kernels.cu's device function of the same
// name. Identical Cramer's-rule formulas, spelled with std::fabs instead of
// fabsf (float overloads resolve the same way; CLAUDE.md's "duplication is
// deliberate" rule applies verbatim here).
// ---------------------------------------------------------------------------
bool solve_plane_3x3(float Sxx, float Syy, float Sxy, float Sx, float Sy, float n,
                     float Sxz, float Syz, float Sz,
                     float* a, float* b, float* c)
{
    const float det = Sxx * (Syy * n - Sy * Sy)
                     - Sxy * (Sxy * n - Sy * Sx)
                     + Sx  * (Sxy * Sy - Syy * Sx);
    const float kDetEps = 1e-9f;
    if (std::fabs(det) < kDetEps) return false;

    const float det_a = Sxz * (Syy * n - Sy * Sy)
                       - Sxy * (Syz * n - Sy * Sz)
                       + Sx  * (Syz * Sy - Syy * Sz);
    const float det_b = Sxx * (Syz * n - Sy * Sz)
                       - Sxz * (Sxy * n - Sy * Sx)
                       + Sx  * (Sxy * Sz - Syz * Sx);
    const float det_c = Sxx * (Syy * Sz - Syz * Sy)
                       - Sxy * (Sxy * Sz - Syz * Sx)
                       + Sxz * (Sxy * Sy - Syy * Sx);

    *a = det_a / det;
    *b = det_b / det;
    *c = det_c / det;
    return true;
}

bool is_hazard_cell(int idx, const float* height_m, const float* slope_rad,
                    const float* roughness_m, float slope_limit_rad)
{
    if (std::isnan(height_m[idx])) return true;
    const float s = slope_rad[idx];
    if (std::isnan(s)) return true;
    if (s > slope_limit_rad) return true;
    const float rgh = roughness_m[idx];
    if (!std::isnan(rgh) && rgh > kRoughnessMaxM) return true;
    return false;
}
} // namespace

// ---------------------------------------------------------------------------
// slope_roughness_cpu — sequential twin of slope_roughness_kernel: same
// two-pass window scan (accumulate the fit, then the residuals) for every
// one of the kGridW*kGridH cells, one after another.
// ---------------------------------------------------------------------------
void slope_roughness_cpu(const float* height_m, float* slope_rad, float* roughness_m)
{
    for (int row = 0; row < kGridH; ++row) {
        for (int col = 0; col < kGridW; ++col) {
            const int idx = row * kGridW + col;

            if (std::isnan(height_m[idx])) {
                slope_rad[idx] = kNaN();
                roughness_m[idx] = kNaN();
                continue;
            }

            float Sxx = 0, Syy = 0, Sxy = 0, Sx = 0, Sy = 0, Sz = 0, Sxz = 0, Syz = 0;
            int n = 0;
            for (int dr = -kFitRadius; dr <= kFitRadius; ++dr) {
                const int nr = row + dr;
                if (nr < 0 || nr >= kGridH) continue;
                for (int dc = -kFitRadius; dc <= kFitRadius; ++dc) {
                    const int nc = col + dc;
                    if (nc < 0 || nc >= kGridW) continue;
                    const float zi = height_m[nr * kGridW + nc];
                    if (std::isnan(zi)) continue;
                    const float xi = dc * kCellM;
                    const float yi = dr * kCellM;
                    Sxx += xi * xi; Syy += yi * yi; Sxy += xi * yi;
                    Sx  += xi;      Sy  += yi;      Sz  += zi;
                    Sxz += xi * zi; Syz += yi * zi;
                    ++n;
                }
            }

            // See kernels.cu's twin function for why the intercept is named
            // `c0` and not `c` (a real shadowing bug this project hit and
            // fixed — a nearby loop's column index was also called `c`).
            float a, b, c0;
            if (n < 3 || !solve_plane_3x3(Sxx, Syy, Sxy, Sx, Sy, static_cast<float>(n),
                                          Sxz, Syz, Sz, &a, &b, &c0)) {
                slope_rad[idx] = kNaN();
                roughness_m[idx] = kNaN();
                continue;
            }

            slope_rad[idx] = std::atan(std::sqrt(a * a + b * b));

            float sum_r2 = 0.0f;
            for (int dr = -kFitRadius; dr <= kFitRadius; ++dr) {
                const int nr = row + dr;
                if (nr < 0 || nr >= kGridH) continue;
                for (int dc = -kFitRadius; dc <= kFitRadius; ++dc) {
                    const int nc = col + dc;
                    if (nc < 0 || nc >= kGridW) continue;
                    const float zi = height_m[nr * kGridW + nc];
                    if (std::isnan(zi)) continue;
                    const float xi = dc * kCellM;
                    const float yi = dr * kCellM;
                    const float resid = zi - (a * xi + b * yi + c0);
                    sum_r2 += resid * resid;
                }
            }
            roughness_m[idx] = std::sqrt(sum_r2 / static_cast<float>(n));
        }
    }
}

// ---------------------------------------------------------------------------
// edge_distance_cpu — sequential twin of edge_distance_kernel: identical
// bounded box-then-disc search, identical hazard predicate.
// ---------------------------------------------------------------------------
void edge_distance_cpu(const float* height_m, const float* slope_rad,
                       const float* roughness_m, float slope_limit_rad,
                       float* edge_dist_m)
{
    const int R = kEdgeSearchRadiusCells;
    for (int row = 0; row < kGridH; ++row) {
        for (int col = 0; col < kGridW; ++col) {
            const int idx = row * kGridW + col;
            if (is_hazard_cell(idx, height_m, slope_rad, roughness_m, slope_limit_rad)) {
                edge_dist_m[idx] = 0.0f;
                continue;
            }
            int best_d2 = R * R + 1;
            for (int dr = -R; dr <= R; ++dr) {
                const int r = row + dr;
                if (r < 0 || r >= kGridH) continue;
                for (int dc = -R; dc <= R; ++dc) {
                    const int c = col + dc;
                    if (c < 0 || c >= kGridW) continue;
                    const int d2 = dr * dr + dc * dc;
                    if (d2 > R * R || d2 >= best_d2) continue;
                    const int nidx = r * kGridW + c;
                    if (is_hazard_cell(nidx, height_m, slope_rad, roughness_m, slope_limit_rad))
                        best_d2 = d2;
                }
            }
            edge_dist_m[idx] = (best_d2 <= R * R)
                ? std::sqrt(static_cast<float>(best_d2)) * kCellM
                : static_cast<float>(R) * kCellM;
        }
    }
}

// ---------------------------------------------------------------------------
// fusion_cpu — sequential twin of fusion_kernel: identical hard-veto test,
// identical weighted blend.
// ---------------------------------------------------------------------------
void fusion_cpu(const float* height_m, const float* slope_rad,
                const float* roughness_m, const float* edge_dist_m,
                float slope_limit_rad, float* score)
{
    for (int idx = 0; idx < kGridW * kGridH; ++idx) {
        const float slope = slope_rad[idx];
        if (std::isnan(height_m[idx]) || std::isnan(slope) || slope > slope_limit_rad) {
            score[idx] = 0.0f;
            continue;
        }
        const float slope_score = std::fmin(std::fmax(1.0f - slope / slope_limit_rad, 0.0f), 1.0f);

        const float rgh = roughness_m[idx];
        const float rgh_safe = std::isnan(rgh) ? kRoughnessMaxM : rgh;
        const float rough_score = std::fmin(std::fmax(1.0f - rgh_safe / kRoughnessMaxM, 0.0f), 1.0f);

        const float edge_score = std::fmin(std::fmax(edge_dist_m[idx] / kEdgeSafeDistM, 0.0f), 1.0f);

        score[idx] = kWeightSlope * slope_score + kWeightRough * rough_score + kWeightEdge * edge_score;
    }
}

// ---------------------------------------------------------------------------
// foothold_selection_cpu — sequential twin of foothold_selection_kernel:
// identical raster-order disc walk and strict-'>' tie-break, so that when
// fed the IDENTICAL score grid the GPU kernel was fed, this oracle selects
// the IDENTICAL cell for every query (THEORY.md §How we verify correctness
// explains why this exactness is achievable here).
// ---------------------------------------------------------------------------
void foothold_selection_cpu(const float* score, const FootholdQuery* queries,
                            int num_queries, FootholdResult* results)
{
    for (int q = 0; q < num_queries; ++q) {
        const float qx = queries[q].x_m;
        const float qy = queries[q].y_m;
        const int col0 = static_cast<int>(std::lround(qx / kCellM));
        const int row0 = static_cast<int>(std::lround(qy / kCellM));

        const float rad_m = kFootholdSearchRadiusM;
        const float rad2_m = rad_m * rad_m;
        const int Rc = static_cast<int>(std::ceil(rad_m / kCellM));

        float best_score = -1.0f;
        int best_row = -1, best_col = -1;

        for (int dr = -Rc; dr <= Rc; ++dr) {
            const int r = row0 + dr;
            if (r < 0 || r >= kGridH) continue;
            for (int dc = -Rc; dc <= Rc; ++dc) {
                const int c = col0 + dc;
                if (c < 0 || c >= kGridW) continue;
                const float ddx = c * kCellM - qx;
                const float ddy = r * kCellM - qy;
                // See kernels.cu's foothold_selection_kernel for why this
                // epsilon exists (an FMA-vs-no-FMA rounding asymmetry at the
                // disc boundary) — it must match the GPU side EXACTLY.
                constexpr float kDiscEps = 1e-6f;
                if (ddx * ddx + ddy * ddy > rad2_m + kDiscEps) continue;
                const float s = score[r * kGridW + c];
                if (s > best_score) {
                    best_score = s;
                    best_row = r;
                    best_col = c;
                }
            }
        }

        FootholdResult res;
        res.row = best_row;
        res.col = best_col;
        if (best_row >= 0) {
            const float sx = best_col * kCellM - qx;
            const float sy = best_row * kCellM - qy;
            res.score = best_score;
            res.dist_m = std::sqrt(sx * sx + sy * sy);
            res.valid = (best_score >= kValidThreshold) ? 1 : 0;
        } else {
            res.score = 0.0f;
            res.dist_m = 0.0f;
            res.valid = 0;
        }
        results[q] = res;
    }
}
