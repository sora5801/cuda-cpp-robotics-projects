// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 14.02
//                     (Traversability costmaps fusing semantics + geometry)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), restated from 13.03 because they
// apply identically here:
//   1) It is the CORRECTNESS ORACLE — main.cu runs both paths on IDENTICAL,
//      PINNED inputs per stage and asserts agreement within a documented
//      tolerance (see main.cu's file header for why stage-isolation matters
//      for a multi-kernel pipeline like this one).
//   2) It is the TEACHING BASELINE — read this file, then kernels.cu, and
//      see exactly what parallelization changed (the loop became threads;
//      the per-cell body is line-for-line identical).
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — a dead-simple sequential version a reader can
// verify by eye. This file is compiled by the HOST compiler (cl.exe); the
// __CUDACC__ fence in kernels.cuh hides device-only declarations from it.
//
// Read this after: kernels.cu — then compare the two side by side; every
// function below has an EXACT namesake in kernels.cu with the identical
// per-cell formula, just addressed by (col,row) threads instead of a loop.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>     // std::sqrt, std::atan, std::cos, std::isnan
#include <algorithm> // std::min, std::max

namespace {

// clampf — clamp x into [lo, hi]. A tiny, repeatedly-used helper (both this
// file and kernels.cu define their OWN copy — deliberate duplication,
// CLAUDE.md §5 — so neither file depends on the other for something this
// small, and a reader of either file sees the whole computation in one place).
inline float clampf(float x, float lo, float hi)
{
    return (x < lo) ? lo : (x > hi ? hi : x);
}

} // namespace

// ===========================================================================
// geometric_layer_cpu — sequential twin of geometric_layer_kernel.
//
// Per cell: a least-squares plane fit over a (2*kFitRadiusCells+1)^2 window
// gives slope_rad and roughness_m (same two-pass structure 13.03 uses: the
// residual pass needs the plane the first pass just solved for); a SEPARATE,
// smaller (2*kStepRadiusCells+1)^2 window's max-min height swing gives
// step_height_m. THEORY.md §The math derives every formula below; §The
// algorithm explains why step-height needs its own, tighter window.
// ===========================================================================
void geometric_layer_cpu(const float* elevation_m,
                         float* slope_rad, float* step_height_m, float* roughness_m)
{
    for (int row = 0; row < kGridH; ++row) {
        for (int col = 0; col < kGridW; ++col) {
            const int idx = row * kGridW + col;

            // ---- Pass 1: least-squares plane fit over the WIDE window -----
            // Cell-centered local coordinates x_i=dc*kCellM, y_i=dr*kCellM;
            // fit z = a*x + b*y + c by the normal equations (THEORY.md's
            // "design matrix transpose times design matrix" derivation).
            double Sxx = 0, Syy = 0, Sxy = 0, Sx = 0, Sy = 0, Sz = 0, Sxz = 0, Syz = 0;
            int n = 0;
            for (int dr = -kFitRadiusCells; dr <= kFitRadiusCells; ++dr) {
                const int nr = row + dr;
                if (nr < 0 || nr >= kGridH) continue;          // clip at map edge
                for (int dc = -kFitRadiusCells; dc <= kFitRadiusCells; ++dc) {
                    const int nc = col + dc;
                    if (nc < 0 || nc >= kGridW) continue;
                    const double zi = elevation_m[nr * kGridW + nc];
                    const double xi = dc * static_cast<double>(kCellM);
                    const double yi = dr * static_cast<double>(kCellM);
                    Sxx += xi * xi; Syy += yi * yi; Sxy += xi * yi;
                    Sx  += xi;      Sy  += yi;      Sz  += zi;
                    Sxz += xi * zi; Syz += yi * zi;
                    ++n;
                }
            }

            // Cramer's-rule 3x3 solve (double precision on the CPU side —
            // deliberately: this oracle exists to catch GPU bugs, not to
            // reproduce GPU float32 rounding; main.cu's tolerance absorbs the
            // resulting few-ULP-to-few-tenths-of-a-percent gap, exactly as
            // 13.03 documents for its own plane fit).
            const double det = Sxx * (Syy * n - Sy * Sy)
                             - Sxy * (Sxy * n - Sy * Sx)
                             + Sx  * (Sxy * Sy - Syy * Sx);
            const double kDetEps = 1e-9;
            double a = 0.0, b = 0.0, c0 = 0.0;
            bool have_plane = (n >= 3) && (std::fabs(det) >= kDetEps);
            if (have_plane) {
                const double det_a = Sxz * (Syy * n - Sy * Sy)
                                   - Sxy * (Syz * n - Sy * Sz)
                                   + Sx  * (Syz * Sy - Syy * Sz);
                const double det_b = Sxx * (Syz * n - Sy * Sz)
                                   - Sxz * (Sxy * n - Sy * Sx)
                                   + Sx  * (Sxy * Sz - Syz * Sx);
                const double det_c = Sxx * (Syy * Sz - Syz * Sy)
                                   - Sxy * (Sxy * Sz - Syz * Sx)
                                   + Sxz * (Sxy * Sy - Syy * Sx);
                a = det_a / det; b = det_b / det; c0 = det_c / det;
            }

            // This project's terrain is always fully defined (no NaN holes —
            // README §Limitations), and the window always has >= 3 samples
            // near any interior/edge cell of a 256x256 grid with a radius-3
            // window, so `have_plane` failing here would indicate a genuine
            // degeneracy (perfectly collinear samples) rather than a hole;
            // treated identically to 13.03's degenerate case: NaN out.
            if (!have_plane) {
                slope_rad[idx] = std::nanf("");
                roughness_m[idx] = std::nanf("");
            } else {
                // slope = atan(|gradient|) = atan(sqrt(a^2+b^2)) — THEORY §math.
                slope_rad[idx] = static_cast<float>(std::atan(std::sqrt(a * a + b * b)));

                // ---- Pass 2: residual std-dev against the now-known plane --
                double sum_r2 = 0.0;
                for (int dr = -kFitRadiusCells; dr <= kFitRadiusCells; ++dr) {
                    const int nr = row + dr;
                    if (nr < 0 || nr >= kGridH) continue;
                    for (int dc = -kFitRadiusCells; dc <= kFitRadiusCells; ++dc) {
                        const int nc = col + dc;
                        if (nc < 0 || nc >= kGridW) continue;
                        const double zi = elevation_m[nr * kGridW + nc];
                        const double xi = dc * static_cast<double>(kCellM);
                        const double yi = dr * static_cast<double>(kCellM);
                        const double resid = zi - (a * xi + b * yi + c0);
                        sum_r2 += resid * resid;
                    }
                }
                // Population std-dev (divide by n, not n-3) — same choice as
                // 13.03, for the same reason: the thresholds this feeds have
                // far more slack than the ~7% bias this introduces.
                roughness_m[idx] = static_cast<float>(std::sqrt(sum_r2 / static_cast<double>(n)));
            }

            // ---- Step height: max-min swing over the TIGHTER window -------
            // Deliberately a SEPARATE gather from the plane fit: a sharp
            // discrete edge (a ditch lip, a berm crest) needs a small window
            // that does not get smoothed away by the wide least-squares fit
            // above (THEORY.md §The algorithm).
            float zmin =  1e30f, zmax = -1e30f;
            for (int dr = -kStepRadiusCells; dr <= kStepRadiusCells; ++dr) {
                const int nr = row + dr;
                if (nr < 0 || nr >= kGridH) continue;
                for (int dc = -kStepRadiusCells; dc <= kStepRadiusCells; ++dc) {
                    const int nc = col + dc;
                    if (nc < 0 || nc >= kGridW) continue;
                    const float zi = elevation_m[nr * kGridW + nc];
                    if (zi < zmin) zmin = zi;
                    if (zi > zmax) zmax = zi;
                }
            }
            step_height_m[idx] = zmax - zmin;   // always well-defined: the
                                                // window always contains at
                                                // least the cell itself.
        }
    }
}

// ===========================================================================
// semantic_layer_cpu — sequential twin of semantic_layer_kernel.
//
// Pure per-cell map: semantic_cost = confidence*prior + (1-confidence)*
// kPessimisticPriorCost — a convex blend between "trust the label" and
// "assume the conservative fallback", THEORY.md §The math's confidence-
// weighting derivation.
// ===========================================================================
void semantic_layer_cpu(const uint8_t* semantic_class, const float* confidence,
                        float* semantic_cost)
{
    for (int i = 0; i < kGridW * kGridH; ++i) {
        const uint8_t cls = semantic_class[i];
        const float prior = kClassPriorCost[cls];           // cls is always < kNumClasses by construction
        const float conf = clampf(confidence[i], 0.0f, 1.0f); // defensive clamp: a real softmax is already in [0,1]
        semantic_cost[i] = conf * prior + (1.0f - conf) * kPessimisticPriorCost;
    }
}

// ===========================================================================
// fusion_cpu — sequential twin of fusion_kernel.
//
// Two hard vetoes (fused_cost forced to EXACTLY 1.0, no partial credit):
//   * geometric: slope past slope_limit_rad, OR step past step_limit_m, OR a
//     degenerate/NaN plane fit (cannot certify the geometry at all);
//   * semantic: the ARGMAX class is CLASS_WATER, REGARDLESS of confidence —
//     THEORY.md §The two-channel fusion problem explains the asymmetric-risk
//     reasoning: a false "not water" costs a minor detour; a false "not
//     water" that was actually water risks the vehicle. Confidence still
//     modulates semantic_cost's CONTINUOUS value everywhere else — it never
//     softens this one veto.
// Everywhere else: a WEIGHTED BLEND of geo_cost (itself a weighted blend of
// three [0,1] sub-costs) and semantic_cost — letting a confident, cheap
// semantic reading pull down a geometrically noisy cell's fused cost, and
// vice versa (README/THEORY: the "rescue" story).
// ===========================================================================
void fusion_cpu(const float* slope_rad, const float* step_height_m,
                const float* roughness_m, const uint8_t* semantic_class,
                const float* semantic_cost, float slope_limit_rad,
                float step_limit_m, float* geo_cost, float* fused_cost,
                int32_t* veto_reason)
{
    for (int i = 0; i < kGridW * kGridH; ++i) {
        const float slope = slope_rad[i];
        const float step = step_height_m[i];
        const bool geo_veto = std::isnan(slope) || (slope > slope_limit_rad) || (step > step_limit_m);
        const bool sem_veto = (semantic_class[i] == CLASS_WATER);

        // geo_cost is computed REGARDLESS of veto status: a vetoed cell's
        // continuous geometric cost is still useful teaching/diagnostic
        // information (demo/out/layers.csv shows it), and fusion_kernel's
        // hard-veto branch below simply does not use it for that cell.
        const float slope_cost = clampf(std::isnan(slope) ? 1.0f : slope / slope_limit_rad, 0.0f, 1.0f);
        const float step_cost  = clampf(step / step_limit_m, 0.0f, 1.0f);
        const float rgh = roughness_m[i];
        const float rough_cost = clampf(std::isnan(rgh) ? 1.0f : rgh / kRoughnessMaxM, 0.0f, 1.0f);
        const float gc = clampf(kWeightSlope * slope_cost + kWeightStep * step_cost + kWeightRough * rough_cost,
                               0.0f, 1.0f);
        geo_cost[i] = gc;

        int32_t reason = kVetoNone;
        if (geo_veto) reason |= kVetoGeo;
        if (sem_veto) reason |= kVetoSem;
        veto_reason[i] = reason;

        if (reason != kVetoNone) {
            fused_cost[i] = 1.0f;   // hard veto — see the file header
        } else {
            fused_cost[i] = clampf(kWeightGeo * gc + kWeightSem * semantic_cost[i], 0.0f, 1.0f);
        }
    }
}

// ===========================================================================
// speed_limit_cpu — sequential twin of speed_limit_kernel.
//
// v_limit = min(kVMaxMps, sqrt(2 * a_avail(cost) * kStopDistM)), where
// a_avail(cost) = kSafetyFraction * kWheelMu * kGravityMps2 * (1 - cost) —
// a straight-line stopping-distance bound (curvature-free: it assumes
// braking along the current heading, never a turn radius) whose available
// deceleration DEGRADES linearly with fused cost. THEORY.md §The math
// derives this in full, including why it is the number 14.01's MPPI would
// consume as an additional running-cost/constraint term.
// ===========================================================================
void speed_limit_cpu(const float* fused_cost, float* speed_limit_mps)
{
    const float a_nominal = kSafetyFraction * kWheelMu * kGravityMps2;
    for (int i = 0; i < kGridW * kGridH; ++i) {
        const float cost = clampf(fused_cost[i], 0.0f, 1.0f);
        const float a_avail = a_nominal * (1.0f - cost);          // >= 0 always
        const float v_kinodynamic = std::sqrt(2.0f * a_avail * kStopDistM);
        speed_limit_mps[i] = std::min(kVMaxMps, v_kinodynamic);
    }
}
