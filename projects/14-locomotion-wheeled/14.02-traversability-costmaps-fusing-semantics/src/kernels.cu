// ===========================================================================
// kernels.cu — GPU implementation for project 14.02
//              Traversability costmaps fusing semantics + geometry
//
// The big idea
// ------------
// An off-road wheeled planner needs a per-cell verdict — "how much does it
// cost to drive here, and how fast may I go?" — built from TWO independent
// evidence channels: what the terrain's SHAPE says (geometry, from an
// elevation map) and what a segmentation net's LABEL says (semantics). This
// file builds that verdict in four passes, each a clean, independent GPU
// pattern, all sharing the SAME "one thread per cell" mapping (a simpler,
// more uniform shape than 13.03's mixed per-cell/per-query pipeline —
// THEORY.md §The GPU mapping explains why a costmap has no natural
// "query" stage the way a foothold planner does):
//
//   geometric_layer_kernel — STENCIL (two windows): a WIDE window's
//                            least-squares plane fit gives slope + roughness
//                            (13.03's technique, reused); a TIGHTER window's
//                            max-min swing gives step height — a second,
//                            independent hazard signal a smooth plane fit
//                            alone would blur away.
//   semantic_layer_kernel  — pure MAP: one class lookup + one confidence-
//                            weighted blend per cell. The simplest kernel
//                            in the file (compare 13.03's fusion_kernel).
//   fusion_kernel           — MAP with two independent HARD-VETO conditions
//                            (one geometric, one semantic) and a weighted
//                            blend everywhere else — the heart of this
//                            project's teaching point: two sensors can each
//                            veto on their own terms, and neither channel's
//                            confidence can talk the other one down from a
//                            hard veto it is sure about.
//   speed_limit_kernel      — pure MAP: a closed-form stopping-distance
//                            formula turns cost into m/s. This project's
//                            CONSUMER-FACING output (README §System context
//                            names 14.01's MPPI as the consumer by name).
//
// Every kernel here is ONE THREAD PER CELL — no shared memory, no atomics,
// no inter-thread communication (same "simplest correct mapping first"
// choice 13.03 documents; THEORY.md §The GPU mapping discusses the shared-
// memory tiling that would remove the geometric kernel's redundant window
// re-reads, and why it is not built here).
//
// Read this after: kernels.cuh. Companion oracle: reference_cpu.cpp (a
// deliberate near-line-by-line twin of every function below, spelled with
// std:: instead of CUDA intrinsics and double instead of float in the plane
// fit — CLAUDE.md §5's "duplication, not symlinks" rule; the float/double
// difference is exactly why main.cu's Stage 1 VERIFY gate uses a tolerance
// instead of demanding bit-exact agreement, see THEORY.md §Numerical
// considerations).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// device_clampf — clamp x into [lo, hi]. __forceinline__ __device__: this is
// called several times per thread across two kernels; inlining removes any
// call overhead and lets the compiler fold it into the surrounding FMA chain.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float device_clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

// ===========================================================================
// solve_plane_3x3 — closed-form solve of the 3x3 normal-equations system for
// a least-squares plane z = a*x + b*y + c fit through a window of samples.
//
// Identical derivation to 13.03's function of the same name (Cramer's rule,
// three 3x3 determinants over one) — repeated here, not shared, per
// CLAUDE.md §5's "deliberate duplication" rule: every project stays
// independently readable. See 13.03's kernels.cu for the full normal-
// equations derivation in comments; THEORY.md §The math repeats it for this
// project's own notation.
//
// Returns false ("degenerate window") when |det| is too small to trust —
// callers must treat that as "no plane could be fit here", never silently
// use garbage a/b/c.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool solve_plane_3x3(
    float Sxx, float Syy, float Sxy, float Sx, float Sy, float n,
    float Sxz, float Syz, float Sz,
    float* a, float* b, float* c)
{
    const float det = Sxx * (Syy * n - Sy * Sy)
                     - Sxy * (Sxy * n - Sy * Sx)
                     + Sx  * (Sxy * Sy - Syy * Sx);

    // Scale-aware degeneracy guard (13.03's reasoning applies verbatim: the
    // sums are built from exact multiples of kCellM, so a well-posed window's
    // determinant sits many orders of magnitude above float epsilon).
    const float kDetEps = 1e-9f;
    if (fabsf(det) < kDetEps) return false;

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

// ===========================================================================
// geometric_layer_kernel — one thread per grid cell.
//
// Thread-to-data mapping: 2-D launch, thread (blockIdx.x*16+threadIdx.x,
// blockIdx.y*16+threadIdx.y) owns cell (col, row) — the same natural 2-D
// mapping 13.03 uses for its per-cell kernels (the row-major array math
// row*kGridW+col mirrors the thread index math directly). Grid: ceil(W/16) x
// ceil(H/16) blocks of 16x16 = 256 threads.
//
// What each thread does: (1) gather up to (2*kFitRadiusCells+1)^2 = 49
// neighboring heights, solve the least-squares plane, derive slope, then a
// SECOND pass over the same window computes the residual std-dev
// (roughness) — 13.03's two-pass technique, unchanged; (2) SEPARATELY,
// gather the smaller (2*kStepRadiusCells+1)^2 = 25-cell window's max-min
// height swing for step_height_m — a different-shaped hazard signal a smooth
// plane fit alone cannot see (a single sharp step barely moves a wide
// window's LEAST-SQUARES slope, because the fit averages the step in with
// many flat neighbors — THEORY.md §The algorithm works this through
// numerically).
//
// Memory: everything read is GLOBAL (elevation_m); each thread's two windows
// overlap its neighbors' windows heavily (up to 49x and 25x redundant
// re-reads respectively) — THEORY.md §The GPU mapping discusses the shared-
// memory tiling left as an exercise, same trade-off 13.03 documents.
// ===========================================================================
__global__ void geometric_layer_kernel(const float* __restrict__ elevation_m,
                                       float* __restrict__ slope_rad,
                                       float* __restrict__ step_height_m,
                                       float* __restrict__ roughness_m)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // map-local x index
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // map-local y index
    if (col >= kGridW || row >= kGridH) return;               // ragged-tile guard
    const int idx = row * kGridW + col;

    // ---- Pass 1: accumulate the least-squares sums over the WIDE window --
    float Sxx = 0, Syy = 0, Sxy = 0, Sx = 0, Sy = 0, Sz = 0, Sxz = 0, Syz = 0;
    int n = 0;
    for (int dr = -kFitRadiusCells; dr <= kFitRadiusCells; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= kGridH) continue;                   // clip at the map edge
        for (int dc = -kFitRadiusCells; dc <= kFitRadiusCells; ++dc) {
            const int nc = col + dc;
            if (nc < 0 || nc >= kGridW) continue;
            const float zi = elevation_m[nr * kGridW + nc];
            const float xi = dc * kCellM;
            const float yi = dr * kCellM;
            Sxx += xi * xi; Syy += yi * yi; Sxy += xi * yi;
            Sx  += xi;      Sy  += yi;      Sz  += zi;
            Sxz += xi * zi; Syz += yi * zi;
            ++n;
        }
    }

    float a = 0.0f, b = 0.0f, c0 = 0.0f;
    const bool have_plane = (n >= 3) &&
        solve_plane_3x3(Sxx, Syy, Sxy, Sx, Sy, static_cast<float>(n), Sxz, Syz, Sz, &a, &b, &c0);

    if (!have_plane) {
        // This project's terrain has no holes (README §Limitations), so this
        // branch fires only for a genuinely degenerate (collinear) window —
        // propagate NaN rather than fabricate a slope, exactly 13.03's
        // discipline for "I could not certify this cell's geometry".
        slope_rad[idx] = nanf("");
        roughness_m[idx] = nanf("");
    } else {
        // slope = atan(|gradient|) = atan(sqrt(a^2+b^2)) — THEORY.md §math.
        slope_rad[idx] = atanf(sqrtf(a * a + b * b));

        // ---- Pass 2: residual std-dev against the now-known plane --------
        float sum_r2 = 0.0f;
        for (int dr = -kFitRadiusCells; dr <= kFitRadiusCells; ++dr) {
            const int nr = row + dr;
            if (nr < 0 || nr >= kGridH) continue;
            for (int dc = -kFitRadiusCells; dc <= kFitRadiusCells; ++dc) {
                const int nc = col + dc;
                if (nc < 0 || nc >= kGridW) continue;
                const float zi = elevation_m[nr * kGridW + nc];
                const float xi = dc * kCellM;
                const float yi = dr * kCellM;
                const float resid = zi - (a * xi + b * yi + c0);
                sum_r2 += resid * resid;
            }
        }
        roughness_m[idx] = sqrtf(sum_r2 / static_cast<float>(n));  // population std-dev; see reference_cpu.cpp note
    }

    // ---- Step height: max-min swing over the TIGHTER window --------------
    // A completely separate gather from the plane fit above — deliberately:
    // see the kernel header for why one window cannot serve both jobs.
    float zmin =  1e30f, zmax = -1e30f;
    for (int dr = -kStepRadiusCells; dr <= kStepRadiusCells; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= kGridH) continue;
        for (int dc = -kStepRadiusCells; dc <= kStepRadiusCells; ++dc) {
            const int nc = col + dc;
            if (nc < 0 || nc >= kGridW) continue;
            const float zi = elevation_m[nr * kGridW + nc];
            zmin = fminf(zmin, zi);
            zmax = fmaxf(zmax, zi);
        }
    }
    step_height_m[idx] = zmax - zmin;   // always well-defined: the window
                                        // always contains at least the cell
                                        // itself (dr=dc=0 is never clipped).
}

// ===========================================================================
// semantic_layer_kernel — one thread per grid cell. The simplest kernel in
// the file: one class lookup, one confidence blend (pure MAP — compare
// 33.01's SAXPY, or 13.03's fusion_kernel for the same "simplest pattern"
// role in that project).
//
// semantic_cost = confidence*prior[class] + (1-confidence)*kPessimisticPriorCost
// — a convex combination (confidence in [0,1]), so semantic_cost stays in
// [0,1] whenever prior[class] does, which every entry in kClassPriorCost
// does by construction (kernels.cuh).
// ===========================================================================
__global__ void semantic_layer_kernel(const uint8_t* __restrict__ semantic_class,
                                      const float* __restrict__ confidence,
                                      float* __restrict__ semantic_cost)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= kGridW || row >= kGridH) return;
    const int idx = row * kGridW + col;

    const uint8_t cls = semantic_class[idx];                   // this cell's argmax class id
    const float prior = kClassPriorCost[cls];                  // __constant__-fast: a 6-entry array
                                                                // the compiler keeps in registers/L1,
                                                                // not worth a dedicated __constant__
                                                                // memory declaration at this size
                                                                // (THEORY.md §The GPU mapping).
    const float conf = device_clampf(confidence[idx], 0.0f, 1.0f);  // defensive: a real softmax is already in [0,1]

    semantic_cost[idx] = conf * prior + (1.0f - conf) * kPessimisticPriorCost;
}

// ===========================================================================
// fusion_kernel — one thread per grid cell. Combines the geometric layer's
// three per-cell numbers and the semantic layer's cost into ONE fused cost,
// with two INDEPENDENT hard vetoes — the project's central teaching point.
//
// Hard vetoes (fused_cost forced to EXACTLY 1.0f, no partial credit):
//   * GEOMETRIC: slope past slope_limit_rad, OR step past step_limit_m, OR a
//     degenerate/NaN plane fit (cannot certify the geometry at all) —
//     physically will not hold or will not fit under the vehicle regardless
//     of what the terrain is LABELED as (THEORY.md §The problem derives
//     both limits from vehicle geometry/friction). This is why the DITCH
//     analytic gate (main.cu) passes even when the ditch is labeled cheap
//     gravel: geometry vetoes on its own terms.
//   * SEMANTIC: the ARGMAX class is CLASS_WATER, REGARDLESS of confidence —
//     an intentionally confidence-IMMUNE veto (THEORY.md §The two-channel
//     fusion problem's asymmetric-risk argument: a false "it's water" costs
//     a detour; a false "it's not water" that was actually water risks the
//     vehicle). This is why the WATER analytic gate passes even inside a
//     perfectly flat, geometrically ideal region: semantics vetoes on ITS
//     own terms, independent of how good geometry looks.
// Everywhere else: fused_cost = kWeightGeo*geo_cost + kWeightSem*semantic_cost
// — a convex blend, so a confident CHEAP semantic reading (e.g. VEGETATION's
// moderate prior, read with high confidence) pulls a geometrically noisy
// cell's fused cost DOWN below the hard-veto ceiling — the "semantics
// rescues, at reduced speed" story the VEGETATION analytic gate checks.
// ===========================================================================
__global__ void fusion_kernel(const float* __restrict__ slope_rad,
                              const float* __restrict__ step_height_m,
                              const float* __restrict__ roughness_m,
                              const uint8_t* __restrict__ semantic_class,
                              const float* __restrict__ semantic_cost,
                              float slope_limit_rad,
                              float step_limit_m,
                              float* __restrict__ geo_cost,
                              float* __restrict__ fused_cost,
                              int32_t* __restrict__ veto_reason)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= kGridW || row >= kGridH) return;
    const int idx = row * kGridW + col;

    const float slope = slope_rad[idx];
    const float step = step_height_m[idx];
    const bool geo_veto = isnan(slope) || (slope > slope_limit_rad) || (step > step_limit_m);
    const bool sem_veto = (semantic_class[idx] == CLASS_WATER);

    // The continuous geometric cost is computed REGARDLESS of veto status —
    // even a vetoed cell's "how bad does geometry alone think this is" is
    // useful diagnostic output (demo/out/layers.csv shows it on every row).
    const float slope_cost = device_clampf(isnan(slope) ? 1.0f : slope / slope_limit_rad, 0.0f, 1.0f);
    const float step_cost  = device_clampf(step / step_limit_m, 0.0f, 1.0f);
    const float rgh = roughness_m[idx];
    const float rough_cost = device_clampf(isnan(rgh) ? 1.0f : rgh / kRoughnessMaxM, 0.0f, 1.0f);
    const float gc = device_clampf(kWeightSlope * slope_cost + kWeightStep * step_cost + kWeightRough * rough_cost,
                                  0.0f, 1.0f);
    geo_cost[idx] = gc;

    int32_t reason = kVetoNone;
    if (geo_veto) reason |= kVetoGeo;
    if (sem_veto) reason |= kVetoSem;
    veto_reason[idx] = reason;

    fused_cost[idx] = (reason != kVetoNone)
        ? 1.0f
        : device_clampf(kWeightGeo * gc + kWeightSem * semantic_cost[idx], 0.0f, 1.0f);
}

// ===========================================================================
// speed_limit_kernel — one thread per grid cell. The project's CONSUMER-
// FACING output: turns a unitless [0,1] cost into an actionable m/s bound.
//
// v_limit = min(kVMaxMps, sqrt(2 * a_avail(cost) * kStopDistM)), where
// a_avail(cost) = kSafetyFraction * kWheelMu * kGravityMps2 * (1 - cost) —
// a curvature-free straight-line stopping-distance bound (THEORY.md §The
// math derives it from vf^2 = vi^2 - 2*a*d with vf=0). At cost=0, a_avail is
// the full reserved-fraction deceleration budget and the bound typically
// saturates at kVMaxMps (terrain is not the limiting factor); as cost rises
// toward 1, a_avail falls toward 0 and the bound falls toward 0 m/s exactly
// at the hard-veto ceiling — a cell fusion_kernel vetoed is a cell this
// kernel commands to a full stop, with no separate case needed (the formula
// is already continuous through cost=1).
// ===========================================================================
__global__ void speed_limit_kernel(const float* __restrict__ fused_cost,
                                   float* __restrict__ speed_limit_mps)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= kGridW || row >= kGridH) return;
    const int idx = row * kGridW + col;

    const float cost = device_clampf(fused_cost[idx], 0.0f, 1.0f);
    const float a_nominal = kSafetyFraction * kWheelMu * kGravityMps2;
    const float a_avail = a_nominal * (1.0f - cost);           // >= 0 always
    const float v_kino = sqrtf(2.0f * a_avail * kStopDistM);

    speed_limit_mps[idx] = fminf(kVMaxMps, v_kino);
}

// ===========================================================================
// Host launch wrappers (declared in kernels.cuh). All four kernels share the
// SAME 2-D "16x16 tile" launch geometry — a deliberately more uniform launch
// shape than 13.03's mixed 2-D/1-D pipeline, because every stage here is a
// per-CELL map/stencil; nothing in this project is a per-query batched
// search (THEORY.md §The GPU mapping).
// ===========================================================================
namespace {
dim3 cell_block() { return dim3(16, 16); }
dim3 cell_grid()
{
    return dim3((kGridW + 15) / 16, (kGridH + 15) / 16);
}
} // namespace

void launch_geometric_layer(const float* d_elevation_m,
                            float* d_slope_rad, float* d_step_height_m,
                            float* d_roughness_m)
{
    geometric_layer_kernel<<<cell_grid(), cell_block()>>>(
        d_elevation_m, d_slope_rad, d_step_height_m, d_roughness_m);
    CUDA_CHECK_LAST_ERROR("geometric_layer_kernel launch");
}

void launch_semantic_layer(const uint8_t* d_semantic_class,
                           const float* d_confidence, float* d_semantic_cost)
{
    semantic_layer_kernel<<<cell_grid(), cell_block()>>>(
        d_semantic_class, d_confidence, d_semantic_cost);
    CUDA_CHECK_LAST_ERROR("semantic_layer_kernel launch");
}

void launch_fusion(const float* d_slope_rad, const float* d_step_height_m,
                   const float* d_roughness_m, const uint8_t* d_semantic_class,
                   const float* d_semantic_cost, float slope_limit_rad,
                   float step_limit_m, float* d_geo_cost, float* d_fused_cost,
                   int32_t* d_veto_reason)
{
    fusion_kernel<<<cell_grid(), cell_block()>>>(
        d_slope_rad, d_step_height_m, d_roughness_m, d_semantic_class, d_semantic_cost,
        slope_limit_rad, step_limit_m, d_geo_cost, d_fused_cost, d_veto_reason);
    CUDA_CHECK_LAST_ERROR("fusion_kernel launch");
}

void launch_speed_limit(const float* d_fused_cost, float* d_speed_limit_mps)
{
    speed_limit_kernel<<<cell_grid(), cell_block()>>>(d_fused_cost, d_speed_limit_mps);
    CUDA_CHECK_LAST_ERROR("speed_limit_kernel launch");
}
