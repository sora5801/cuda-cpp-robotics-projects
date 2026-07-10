// ===========================================================================
// kernels.cu — GPU implementation for project 13.03
//              Foothold scoring kernels: slope, roughness, edge distance
//              from elevation maps
//
// The big idea
// ------------
// A quadruped's foot planner needs a per-cell verdict — "can I stand here?"
// — computed from nothing but a noisy height grid. This file builds that
// verdict in four passes, each a clean, independent GPU pattern:
//
//   slope_roughness_kernel  — STENCIL: each cell reads a small WINDOW of
//                              its neighbors (a local least-squares plane
//                              fit) and writes two numbers. Same family as
//                              image convolution/blur kernels.
//   edge_distance_kernel    — BOUNDED GATHER: each cell searches a larger
//                              (but still fixed-radius) window for the
//                              nearest HAZARD cell — a brute-force, capped
//                              distance transform. 07.09 solves the
//                              unbounded version of this problem in
//                              O(log R) passes with jump flooding; here the
//                              radius is small (10 cells) on purpose, so one
//                              dense O(R^2) pass per cell stays cheap and
//                              the kernel reads in one sitting — no
//                              multi-pass ping-pong buffers to reason about.
//   fusion_kernel            — pure MAP: combine four already-computed
//                              per-cell numbers into one score. The simplest
//                              pattern in the file (compare 33.01's SAXPY).
//   foothold_selection_kernel — BATCHED SEARCH: ~1000 independent queries,
//                              each one thread doing an ARGMAX over a small
//                              disc of the score grid it just built. Same
//                              "one thread = one independent problem" shape
//                              as 08.01's rollouts, but each thread's
//                              "problem" is a tiny 2-D search instead of a
//                              1-D time integration.
//
// All four kernels are ONE THREAD PER CELL (the first three) or ONE THREAD
// PER QUERY (the fourth) — no shared memory, no atomics, no inter-thread
// communication anywhere in this file. That is deliberate: CLAUDE.md's
// "teaching beats cleverness" pushes toward the simplest correct mapping
// first; THEORY.md §The GPU mapping discusses the shared-memory tiling that
// would speed up the windowed kernels, and why it is not implemented here.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (a
// deliberate line-by-line twin of every function below, spelled with
// std:: instead of CUDA intrinsics — CLAUDE.md §5's "duplication, not
// symlinks" rule).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ===========================================================================
// solve_plane_3x3 — closed-form solve of the 3x3 NORMAL-EQUATIONS system for
// a least-squares plane z = a*x + b*y + c fit through a window of samples.
//
// Why a hand-written 3x3 solve instead of a library (cuSOLVER etc.)?  The
// system is tiny (3 unknowns) and appears once per grid cell (65536 times
// per kernel launch) — a general dense-solver call per thread would dwarf
// the actual arithmetic with launch/setup overhead. Writing the closed form
// is also the whole POINT here (CLAUDE.md §1 "no black boxes": a 3x3 solve
// is exactly the size a robotics engineer should be able to write by hand).
//
// The normal equations for min_{a,b,c} sum_i (z_i - a*x_i - b*y_i - c)^2 are
//
//     | Sxx  Sxy  Sx | |a|   |Sxz|
//     | Sxy  Syy  Sy | |b| = |Syz|
//     | Sx   Sy   n  | |c|   |Sz |
//
// (the usual "design matrix transpose times design matrix" story — THEORY.md
// §The math derives this from the least-squares residual). Solved here by
// Cramer's rule: three 3x3 determinants over one. A FULLY SYMMETRIC window
// (no missing neighbors) would have Sx = Sy = Sxy = 0 by construction — the
// system would decouple into three trivial divisions — but this repo's
// elevation maps have holes (NaN cells) and map-edge clipping, both of which
// break that symmetry, so the general solve is what actually runs almost
// everywhere. Returns false (a "degenerate window" — near-collinear samples,
// or fewer than 3 non-collinear points) when |det| is too small to trust;
// callers must treat that as "no plane could be fit here", never silently
// use garbage a/b/c (THEORY.md §Numerical considerations).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool solve_plane_3x3(
    float Sxx, float Syy, float Sxy, float Sx, float Sy, float n,
    float Sxz, float Syz, float Sz,
    float* a, float* b, float* c)
{
    // det(M) via cofactor expansion along the top row.
    const float det = Sxx * (Syy * n - Sy * Sy)
                     - Sxy * (Sxy * n - Sy * Sx)
                     + Sx  * (Sxy * Sy - Syy * Sx);

    // Scale-aware degeneracy guard. All the sums above are built from
    // coordinates that are exact multiples of kCellM (never arbitrary real
    // numbers), so a genuinely well-posed window's determinant sits many
    // orders of magnitude above float epsilon; anything this small means
    // the surviving samples are (numerically) collinear, not merely
    // "close to" singular — THEORY.md walks the geometry that can cause it
    // (a window clipped down to a single row/column of neighbors).
    const float kDetEps = 1e-9f;
    if (fabsf(det) < kDetEps) return false;

    // Cramer's rule: replace one column of M with the RHS, take det/det.
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
// slope_roughness_kernel — one thread per grid cell.
//
// Thread-to-data mapping: 2-D launch, thread (blockIdx.x*16+threadIdx.x,
// blockIdx.y*16+threadIdx.y) owns cell (col, row) — the natural mapping for
// grid data (contrast with 08.01's 1-D "thread = rollout"; THEORY.md §The
// GPU mapping discusses why 2-D here). Grid: ceil(W/16) x ceil(H/16) blocks
// of 16x16 = 256 threads (a warp-friendly square tile).
//
// What each thread does: gather up to (2*kFitRadius+1)^2 = 25 neighboring
// heights (clipped to the grid, skipping NaN holes), solve the least-squares
// plane through them (solve_plane_3x3 above), then in a SECOND pass over the
// same window compute the residual std-dev against that plane. Two passes
// are required because roughness is defined relative to the fitted plane,
// which is not known until pass 1 finishes — trying to fuse the passes
// would need Welford-style running moments for THREE more accumulators for
// no real saving (the window is only 25 elements).
//
// Memory: everything read is GLOBAL (height_m; each thread's window
// overlaps its neighbors' windows heavily — up to (2k+1)^2 = 25x re-reads of
// the same cells across nearby threads). No shared memory is used: THEORY.md
// §The GPU mapping discusses the shared-memory TILING that would remove
// this redundancy and why the plain version is what ships here (teaching
// clarity first, CLAUDE.md §1). Every thread writes exactly two coalesced
// floats (slope_rad[idx], roughness_m[idx]) at the very end.
// ===========================================================================
__global__ void slope_roughness_kernel(const float* __restrict__ height_m,
                                       float* __restrict__ slope_rad,
                                       float* __restrict__ roughness_m)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // map-local x index
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // map-local y index
    if (col >= kGridW || row >= kGridH) return;               // ragged-tile guard
    const int idx = row * kGridW + col;

    // A cell with unknown height has no local surface to characterize —
    // NaN propagates immediately and unconditionally (THEORY.md's NaN
    // discipline: never invent geometry for a hole).
    if (isnan(height_m[idx])) {
        slope_rad[idx] = nanf("");
        roughness_m[idx] = nanf("");
        return;
    }

    // ---- Pass 1: accumulate the least-squares sums over the window -------
    // x_i, y_i are CELL-CENTERED local coordinates in meters (the fitted
    // plane is z = a*x + b*y + c about THIS cell, so the fitted height at
    // the cell itself is simply c — used nowhere here, but a useful sanity
    // fact when debugging).
    float Sxx = 0, Syy = 0, Sxy = 0, Sx = 0, Sy = 0, Sz = 0, Sxz = 0, Syz = 0;
    int n = 0;
    for (int dr = -kFitRadius; dr <= kFitRadius; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= kGridH) continue;                   // clip at the map edge
        for (int dc = -kFitRadius; dc <= kFitRadius; ++dc) {
            const int nc = col + dc;
            if (nc < 0 || nc >= kGridW) continue;
            const float zi = height_m[nr * kGridW + nc];
            if (isnan(zi)) continue;                            // skip holes: absent, not zero
            const float xi = dc * kCellM;
            const float yi = dr * kCellM;
            Sxx += xi * xi; Syy += yi * yi; Sxy += xi * yi;
            Sx  += xi;      Sy  += yi;      Sz  += zi;
            Sxz += xi * zi; Syz += yi * zi;
            ++n;
        }
    }

    // Fewer than 3 samples can't determine a plane (3 free parameters); the
    // 3x3 solve's own degeneracy guard catches collinear-but-numerous cases.
    // Both are the SAME "no plane could be fit here" outcome to a caller.
    //
    // NAMING NOTE (a real bug this repo hit and fixed, worth leaving visible
    // for the reader): the plane intercept below is deliberately named `c0`,
    // not `c` — a nearby loop's per-cell COLUMN index is also naturally
    // called `c`, and an earlier draft of this file used that name for both,
    // so the residual loop's `zi - (a*xi + b*yi + c)` silently read the
    // *loop's column index* (an int, implicitly converted to float) instead
    // of the plane's intercept. Both compiled cleanly and only the OUTPUT
    // NUMBERS were wrong (roughness computed as tens of meters instead of
    // millimeters) — the exact failure mode CLAUDE.md's "no black boxes" and
    // heavy commenting rules exist to catch faster. THEORY.md §Numerical
    // considerations tells the full story as a verification case study.
    float a, b, c0;
    if (n < 3 || !solve_plane_3x3(Sxx, Syy, Sxy, Sx, Sy, static_cast<float>(n),
                                  Sxz, Syz, Sz, &a, &b, &c0)) {
        slope_rad[idx] = nanf("");
        roughness_m[idx] = nanf("");
        return;
    }

    // Slope = angle between the plane's normal (proportional to (-a,-b,1))
    // and vertical (0,0,1). tan(slope) = |gradient| = sqrt(a^2+b^2) falls
    // straight out of that dot product — THEORY.md §The math derives it.
    slope_rad[idx] = atanf(sqrtf(a * a + b * b));

    // ---- Pass 2: residual std-dev against the now-known plane -------------
    float sum_r2 = 0.0f;
    for (int dr = -kFitRadius; dr <= kFitRadius; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= kGridH) continue;
        for (int dc = -kFitRadius; dc <= kFitRadius; ++dc) {
            const int nc = col + dc;
            if (nc < 0 || nc >= kGridW) continue;
            const float zi = height_m[nr * kGridW + nc];
            if (isnan(zi)) continue;
            const float xi = dc * kCellM;
            const float yi = dr * kCellM;
            const float resid = zi - (a * xi + b * yi + c0);
            sum_r2 += resid * resid;
        }
    }
    // Population std-dev (divide by n, not n-3): with n typically ~25 the
    // bias against the "unbiased" n-3 residual-dof estimator is a few
    // percent — negligible next to the hazard thresholds this feeds
    // (THEORY.md §Numerical considerations states the choice explicitly).
    roughness_m[idx] = sqrtf(sum_r2 / static_cast<float>(n));
}

// ===========================================================================
// edge_distance_kernel — one thread per grid cell.
//
// is_hazard(cell): unknown height, OR a degenerate/unknown plane fit
// (isnan(slope) also catches that), OR slope past the friction-derived
// limit, OR roughness past kRoughnessMaxM. This is a BROADER notion of
// "hazard" than fusion's hard veto (which uses only the first three) —
// deliberately: something too rough to stand on directly can still be
// worth pushing OTHER cells away from (this kernel), while still being
// eligible for a continuous, nonzero fused score of its own (fusion_kernel)
// rather than an absolute veto. THEORY.md §The algorithm names this
// distinction explicitly.
//
// Thread-to-data mapping: same 2-D tiling as slope_roughness_kernel. Each
// thread performs a BOUNDED, DENSE search over a (2*kEdgeSearchRadiusCells+1)
// square window (441 cells for radius 10), keeping the smallest squared
// distance to any hazard cell found, restricted to the circular disc
// (dr^2+dc^2 <= R^2) — a square SEARCH WINDOW but a circular DISTANCE
// metric, the standard "iterate a box, measure a circle" trick. If the
// window/disc holds no hazard at all, the distance SATURATES at the search
// radius (kEdgeSearchRadiusCells*kCellM) — a documented cap, not a true
// unbounded distance (07.09's jump-flooding kernel computes the unbounded
// version in O(log R) passes when the true value matters more than a cap).
// ===========================================================================
__device__ __forceinline__ bool is_hazard_cell(int idx,
                                               const float* height_m,
                                               const float* slope_rad,
                                               const float* roughness_m,
                                               float slope_limit_rad)
{
    if (isnan(height_m[idx])) return true;
    const float s = slope_rad[idx];
    if (isnan(s)) return true;                 // degenerate fit: cannot certify safety
    if (s > slope_limit_rad) return true;
    const float rgh = roughness_m[idx];
    if (!isnan(rgh) && rgh > kRoughnessMaxM) return true;
    return false;
}

__global__ void edge_distance_kernel(const float* __restrict__ height_m,
                                     const float* __restrict__ slope_rad,
                                     const float* __restrict__ roughness_m,
                                     float slope_limit_rad,
                                     float* __restrict__ edge_dist_m)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= kGridW || row >= kGridH) return;
    const int idx = row * kGridW + col;

    // A hazard cell IS its own nearest hazard: distance 0, no search needed.
    if (is_hazard_cell(idx, height_m, slope_rad, roughness_m, slope_limit_rad)) {
        edge_dist_m[idx] = 0.0f;
        return;
    }

    const int R = kEdgeSearchRadiusCells;
    int best_d2 = R * R + 1;                    // sentinel: "no hazard found yet"
    for (int dr = -R; dr <= R; ++dr) {
        const int r = row + dr;
        if (r < 0 || r >= kGridH) continue;
        for (int dc = -R; dc <= R; ++dc) {
            const int c = col + dc;
            if (c < 0 || c >= kGridW) continue;
            const int d2 = dr * dr + dc * dc;
            if (d2 > R * R || d2 >= best_d2) continue;   // outside the disc, or already beaten
            const int nidx = r * kGridW + c;
            if (is_hazard_cell(nidx, height_m, slope_rad, roughness_m, slope_limit_rad))
                best_d2 = d2;
        }
    }

    edge_dist_m[idx] = (best_d2 <= R * R)
        ? sqrtf(static_cast<float>(best_d2)) * kCellM     // a real, bounded distance
        : static_cast<float>(R) * kCellM;                  // saturated: nothing hazardous nearby
}

// ===========================================================================
// fusion_kernel — one thread per grid cell. The simplest kernel in the file:
// four already-computed numbers go in, one score comes out (pure MAP).
//
// Hard vetoes (score forced to EXACTLY 0.0f, no partial credit):
//   * unknown height (isnan(height_m))       — cannot stand where you can't see
//   * degenerate/unknown plane fit            — cannot certify slope at all
//   * slope past the friction-derived limit   — physically will not hold
//     (THEORY.md §The problem derives slope_limit_rad = atan(mu) from the
//     friction-cone condition: a foot cannot avoid slipping on a slope
//     steeper than this, regardless of how good everything else looks).
// Everything else is a WEIGHTED BLEND of three [0,1] sub-scores — a cell
// can be imperfect (a little rough, a little close to a hazard) and still
// score usably, which is the whole point of a continuous score instead of
// a second binary veto.
// ===========================================================================
__global__ void fusion_kernel(const float* __restrict__ height_m,
                              const float* __restrict__ slope_rad,
                              const float* __restrict__ roughness_m,
                              const float* __restrict__ edge_dist_m,
                              float slope_limit_rad,
                              float* __restrict__ score)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= kGridW || row >= kGridH) return;
    const int idx = row * kGridW + col;

    const float slope = slope_rad[idx];
    if (isnan(height_m[idx]) || isnan(slope) || slope > slope_limit_rad) {
        score[idx] = 0.0f;             // hard veto — see file header
        return;
    }

    // clamp(1 - x/limit, 0, 1): 1.0 at x=0, falls linearly to 0.0 at x=limit,
    // never negative past the limit (defensive: slope already passed the
    // veto above, so this clamp only ever bites for roughness/edge terms).
    const float slope_score = fminf(fmaxf(1.0f - slope / slope_limit_rad, 0.0f), 1.0f);

    const float rgh = roughness_m[idx];
    // Defensive fallback (unreachable given slope/roughness share one fit —
    // slope non-NaN implies roughness non-NaN by construction — kept because
    // a caller could one day decouple the two kernels' inputs).
    const float rgh_safe = isnan(rgh) ? kRoughnessMaxM : rgh;
    const float rough_score = fminf(fmaxf(1.0f - rgh_safe / kRoughnessMaxM, 0.0f), 1.0f);

    const float edge_score = fminf(fmaxf(edge_dist_m[idx] / kEdgeSafeDistM, 0.0f), 1.0f);

    score[idx] = kWeightSlope * slope_score + kWeightRough * rough_score + kWeightEdge * edge_score;
}

// ===========================================================================
// foothold_selection_kernel — one thread per QUERY (not per cell): a batched
// search over ~1000 independent nominal landing points, the pipeline's
// CONSUMER-FACING step (13.02/13.08 would call this once per swing leg).
//
// Thread-to-data mapping: 1-D, thread q = blockIdx.x*blockDim.x+threadIdx.x
// owns query q (compare 08.01's "thread = rollout" — same shape, different
// problem). With only ~1000 queries this kernel launches a HANDFUL of
// blocks — a deliberate contrast with the map kernels' full 65536-thread
// occupancy, discussed in THEORY.md §The GPU mapping ("a wide, cheap kernel
// next to three medium, work-heavy ones").
//
// Each thread converts its query's (x_m,y_m) to a nominal cell, then walks
// every cell within kFootholdSearchRadiusM (a small disc, ~50-80 cells for
// radius 5) keeping the ARGMAX score seen so far. Tie-breaking is
// DETERMINISTIC: the loop visits offsets in raster order (dr ascending,
// then dc ascending) and only replaces the incumbent on a STRICT '>' — so
// the first cell reached with the winning score always wins, on both the
// GPU and the CPU oracle (which walks the identical loop order). This
// matters because this kernel's own VERIFY gate (main.cu) is fed the exact
// same score grid on both paths and expects an EXACT index match — see
// THEORY.md §How we verify correctness for why that is achievable here
// where it would not be for the earlier float-heavy kernels.
// ===========================================================================
__global__ void foothold_selection_kernel(const float* __restrict__ score,
                                          const FootholdQuery* __restrict__ queries,
                                          int num_queries,
                                          FootholdResult* __restrict__ results)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_queries) return;

    const float qx = queries[q].x_m;
    const float qy = queries[q].y_m;
    const int col0 = static_cast<int>(lroundf(qx / kCellM));
    const int row0 = static_cast<int>(lroundf(qy / kCellM));

    const float rad_m = kFootholdSearchRadiusM;
    const float rad2_m = rad_m * rad_m;
    const int Rc = static_cast<int>(ceilf(rad_m / kCellM));   // cell-radius search box

    float best_score = -1.0f;    // strictly below any legal [0,1] score
    int best_row = -1, best_col = -1;

    for (int dr = -Rc; dr <= Rc; ++dr) {
        const int r = row0 + dr;
        if (r < 0 || r >= kGridH) continue;
        for (int dc = -Rc; dc <= Rc; ++dc) {
            const int c = col0 + dc;
            if (c < 0 || c >= kGridW) continue;
            // Circular disc in TRUE metric distance from the query point
            // (not the rounded nominal cell) — a query that lands near a
            // cell boundary still gets a fair, symmetric disc.
            const float ddx = c * kCellM - qx;
            const float ddy = r * kCellM - qy;
            // +kDiscEps: the compiler is free to fuse ddx*ddx+ddy*ddy into
            // an FMA on the GPU but not on the CPU (exactly the a*x+y story
            // 08.01's kernels.cu documents), which can round a cell EXACTLY
            // on the disc boundary a few ULPs differently on the two paths
            // and flip whether it is visited at all — not merely rescore
            // it. Inflating the disc by a fixed epsilon (identical on both
            // paths) removes that asymmetry; THEORY.md §Numerical
            // considerations has the measured-before/after story.
            constexpr float kDiscEps = 1e-6f;
            if (ddx * ddx + ddy * ddy > rad2_m + kDiscEps) continue;
            const float s = score[r * kGridW + c];
            if (s > best_score) {                 // STRICT '>': first-wins tie-break
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
        res.dist_m = sqrtf(sx * sx + sy * sy);
        res.valid = (best_score >= kValidThreshold) ? 1 : 0;
    } else {
        res.score = 0.0f;
        res.dist_m = 0.0f;
        res.valid = 0;
    }
    results[q] = res;
}

// ===========================================================================
// Host launch wrappers (declared in kernels.cuh). All four share the same
// 2-D "16x16 tile" launch geometry for the per-cell kernels and a 1-D
// "256 per block" geometry for the per-query kernel — the repo-default
// block sizes (CLAUDE.md §6.2's worked example uses 256 for the 1-D case).
// ===========================================================================
namespace {
// Shared 2-D launch config for the three per-cell kernels: 16x16 = 256
// threads/block (a warp-friendly square tile — the natural 2-D analogue of
// the repo's usual 256-thread 1-D default), grid sized to exactly cover
// kGridW x kGridH (256x256 / 16x16 = 16x16 blocks, no ragged tail at this
// grid size, though every kernel still guards it for robustness).
dim3 cell_block() { return dim3(16, 16); }
dim3 cell_grid()
{
    return dim3((kGridW + 15) / 16, (kGridH + 15) / 16);
}
} // namespace

void launch_slope_roughness(const float* d_height_m,
                            float* d_slope_rad, float* d_roughness_m)
{
    slope_roughness_kernel<<<cell_grid(), cell_block()>>>(d_height_m, d_slope_rad, d_roughness_m);
    CUDA_CHECK_LAST_ERROR("slope_roughness_kernel launch");
}

void launch_edge_distance(const float* d_height_m, const float* d_slope_rad,
                          const float* d_roughness_m, float slope_limit_rad,
                          float* d_edge_dist_m)
{
    edge_distance_kernel<<<cell_grid(), cell_block()>>>(
        d_height_m, d_slope_rad, d_roughness_m, slope_limit_rad, d_edge_dist_m);
    CUDA_CHECK_LAST_ERROR("edge_distance_kernel launch");
}

void launch_fusion(const float* d_height_m, const float* d_slope_rad,
                   const float* d_roughness_m, const float* d_edge_dist_m,
                   float slope_limit_rad, float* d_score)
{
    fusion_kernel<<<cell_grid(), cell_block()>>>(
        d_height_m, d_slope_rad, d_roughness_m, d_edge_dist_m, slope_limit_rad, d_score);
    CUDA_CHECK_LAST_ERROR("fusion_kernel launch");
}

void launch_foothold_selection(const float* d_score,
                               const FootholdQuery* d_queries, int num_queries,
                               FootholdResult* d_results)
{
    if (num_queries < 1) {
        std::fprintf(stderr, "launch_foothold_selection: invalid num_queries=%d\n", num_queries);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;
    const int blocks = (num_queries + threads - 1) / threads;
    foothold_selection_kernel<<<blocks, threads>>>(d_score, d_queries, num_queries, d_results);
    CUDA_CHECK_LAST_ERROR("foothold_selection_kernel launch");
}
