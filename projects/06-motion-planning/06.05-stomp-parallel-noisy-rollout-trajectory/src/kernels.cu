// ===========================================================================
// kernels.cu — GPU implementation for project 06.05
//              STOMP: parallel noisy-rollout trajectory optimization
//              (teaching core: 2-D point robot through an obstacle field)
//
// The big idea
// ------------
// STOMP optimizes by SAMPLING. Each GPU thread takes ONE noisy candidate
// trajectory (the nominal path plus its own smooth perturbation), integrates
// an obstacle-cost field along it, and reports (a) the per-waypoint local
// cost and (b) the total trajectory cost. The K noisy trajectories are fully
// independent → one thread per rollout is the natural GPU mapping (K ~ 1024
// here; a CPU manages a handful). This is the SAME thread-per-rollout pattern
// as 08.01's MPPI kernel — here the "rollout" is a whole candidate path, not
// a simulated future, and the accumulated quantity is a spatial cost integral
// instead of a control cost.
//
// What is specific to STOMP (vs 08.01 MPPI):
//   * We output a PER-WAYPOINT local cost array Sloc[j*K + k], not just one
//     scalar per rollout. STOMP's update reweights each waypoint SEPARATELY
//     (see the host update in main.cu) — a fundamentally different blend from
//     MPPI's single per-whole-trajectory softmin. The kernel must therefore
//     hand the host enough information to do that per-waypoint weighting.
//   * The rollout cost is a line integral of an obstacle-cost field, sampled
//     densely enough between waypoints to catch a thin obstacle (kSegSamples).
//
// The coalescing fix (inherited from 08.01/33.01): the noise arrays epsx/epsy
// are stored TRANSPOSED (eps[j*K + k]) so that at waypoint index j a warp's 32
// noise reads are consecutive floats — one memory transaction — instead of
// strides of N floats. The layout decision lives in kernels.cuh; this file
// depends on it.
//
// All constants, layouts, and the field/grid convention come from kernels.cuh
// — the single source shared with the CPU oracle; the field sampler and
// segment-cost function below are deliberate line-by-line twins of the ones in
// reference_cpu.cpp (documented duplication across the host/device boundary).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// sample_field — bilinear lookup of the obstacle-cost field at world (x, y).
//
// The field is a gw x gh grid of costs; world position (x, y) maps to grid
// coordinates (gx, gy) = (x / cell_m, y / cell_m). We read the four
// surrounding cells and blend them by the fractional position — bilinear
// interpolation makes the sampled cost CONTINUOUS in (x, y), which matters:
// STOMP nudges waypoints in tiny steps, and a piecewise-constant (nearest)
// lookup would give a flat, gradient-free cost that the optimizer cannot
// follow. Out-of-grid samples clamp to the nearest edge cell (the map is
// bordered by free space, so clamping to the edge cost is safe).
//
// Reads are from GLOBAL memory (the field is 256 KB — far too big for
// constant memory, and accessed at data-dependent addresses as the path
// wanders). On a real planner this field would live in a TEXTURE, whose
// hardware bilinear filter does exactly this blend for free (README
// Exercise) — we do it by hand here so nothing is a black box (CLAUDE.md §1).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sample_field(const float* __restrict__ field,
                                               int gw, int gh, float cell_m,
                                               float x, float y)
{
    // World → continuous grid coordinates.
    float gx = x / cell_m;
    float gy = y / cell_m;

    // Clamp into the valid interpolation range [0, g-1] so the +1 neighbour
    // below never steps off the array (branchless via fminf/fmaxf).
    gx = fminf(fmaxf(gx, 0.0f), static_cast<float>(gw - 1));
    gy = fminf(fmaxf(gy, 0.0f), static_cast<float>(gh - 1));

    const int ix = static_cast<int>(gx);      // lower-left cell index
    const int iy = static_cast<int>(gy);
    const int ix1 = (ix + 1 < gw) ? ix + 1 : ix;   // right/up neighbour (clamped at the edge)
    const int iy1 = (iy + 1 < gh) ? iy + 1 : iy;

    const float fx = gx - static_cast<float>(ix); // fractional position within the cell [0,1)
    const float fy = gy - static_cast<float>(iy);

    // Four corner costs (row-major field[iy*gw + ix]).
    const float c00 = field[iy  * gw + ix ];
    const float c10 = field[iy  * gw + ix1];
    const float c01 = field[iy1 * gw + ix ];
    const float c11 = field[iy1 * gw + ix1];

    // Bilinear blend: interpolate in x on both rows, then in y.
    const float c0 = c00 + fx * (c10 - c00);
    const float c1 = c01 + fx * (c11 - c01);
    return c0 + fy * (c1 - c0);
}

// ---------------------------------------------------------------------------
// segment_cost — obstacle cost accumulated along the segment (ax,ay)->(bx,by).
//
// A midpoint-rule line integral of the cost field: take kSegSamples evenly
// spaced sample points at parameter t = (s + 0.5)/kSegSamples, look up the
// field at each, and weight by the arc length step ds = seg_len/kSegSamples.
// The result has units of cost*metres — "how much obstacle cost this stretch
// of path drives through". Sampling BETWEEN waypoints (not just at them) is
// what lets a thin obstacle sitting mid-segment still register a cost; with
// only per-waypoint point samples a path could hop over a wall unpunished.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float segment_cost(const float* __restrict__ field,
                                               int gw, int gh, float cell_m,
                                               float ax, float ay,
                                               float bx, float by)
{
    const float dx = bx - ax;
    const float dy = by - ay;
    const float seg_len = sqrtf(dx * dx + dy * dy);
    const float ds = seg_len / static_cast<float>(kSegSamples);  // arc-length weight per sample

    float acc = 0.0f;
    // Midpoint samples: s = 0..kSegSamples-1 at t = (s + 0.5)/kSegSamples.
    for (int s = 0; s < kSegSamples; ++s) {
        const float t = (static_cast<float>(s) + 0.5f) / static_cast<float>(kSegSamples);
        const float x = ax + t * dx;
        const float y = ay + t * dy;
        acc += sample_field(field, gw, gh, cell_m, x, y);
    }
    return acc * ds;
}

// ---------------------------------------------------------------------------
// path_point — the s-th point of rollout k's full path, with the fixed
// endpoints substituted at the ends.
//
//   s = 0        -> start          (fixed; every thread reads the same value)
//   s = kN+1     -> goal           (fixed)
//   1 <= s <= kN -> noisy waypoint: theta[s-1] + eps[(s-1)*K + k]
//
// theta reads are UNIFORM (all threads, same address → served by the L2/
// read-only cache at broadcast-like cost). eps reads are COALESCED thanks to
// the transposed layout (adjacent threads k read adjacent floats). Both are
// recomputed on demand rather than cached in registers, because caching all
// kN=64 (x,y) waypoints would need 128 registers per thread and spill.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void path_point(int s, int k, int K,
                                            float sx, float sy, float gx, float gy,
                                            const float* __restrict__ theta_x,
                                            const float* __restrict__ theta_y,
                                            const float* __restrict__ epsx,
                                            const float* __restrict__ epsy,
                                            float* out_x, float* out_y)
{
    if (s == 0) { *out_x = sx; *out_y = sy; return; }          // fixed start
    if (s == kN + 1) { *out_x = gx; *out_y = gy; return; }     // fixed goal
    const int j = s - 1;                                       // interior waypoint index [0, kN)
    *out_x = theta_x[j] + epsx[j * K + k];                     // uniform theta + coalesced eps
    *out_y = theta_y[j] + epsy[j * K + k];
}

// ===========================================================================
// The STOMP scoring kernel: one thread = one noisy candidate trajectory.
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x + threadIdx.x owns
// rollout k. Grid: ceil(K/256) x 256 (repo default; ragged tail guarded).
//
// Outputs per thread k:
//   Sloc[j*K + k]  per interior waypoint j: obstacle cost of the TWO segments
//                  incident to that waypoint (the segment arriving at it and
//                  the segment leaving it). This double-counts each interior
//                  segment across its two endpoints — deliberately: a costly
//                  segment should pull BOTH of its waypoints, so the host's
//                  per-waypoint softmin rewards perturbations that clear
//                  either endpoint. This is the LOCAL state cost STOMP weights.
//   cost[k]        total trajectory cost = obstacle path-integral over all
//                  kN+1 segments (each counted once) + kWSmooth * smoothness,
//                  where smoothness = sum of squared discrete accelerations.
//                  Used only by the §5 GPU-vs-CPU verify gate.
//
// No shared memory (rollouts share nothing), no atomics; the only divergence
// is the ragged-tail guard. The rollouts never interact — by construction.
// ===========================================================================
__global__ void stomp_score_kernel(const float* __restrict__ field, int gw, int gh, float cell_m,
                                    float sx, float sy, float gx, float gy,     // fixed start/goal (m)
                                    const float* __restrict__ theta_x,          // [kN] nominal waypoints
                                    const float* __restrict__ theta_y,
                                    const float* __restrict__ epsx,             // [kN*K] noise, TRANSPOSED [j*K+k]
                                    const float* __restrict__ epsy,
                                    float* __restrict__ Sloc,                   // [kN*K] OUT local cost [j*K+k]
                                    float* __restrict__ cost,                   // [K]    OUT total cost
                                    int K)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's rollout index
    if (k >= K) return;                                    // ragged-tail guard

    // -- (1) Total obstacle cost: sum each of the kN+1 segments ONCE. --------
    // We walk the path point by point, keeping the previous point in
    // registers so each segment costs one path_point() fetch, not two.
    float px, py;                                          // "previous" path point
    path_point(0, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &px, &py);
    float total_obs = 0.0f;
    for (int s = 1; s <= kN + 1; ++s) {
        float cx, cy;
        path_point(s, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &cx, &cy);
        total_obs += segment_cost(field, gw, gh, cell_m, px, py, cx, cy);
        px = cx; py = cy;                                  // slide the window forward
    }

    // -- (2) Per-waypoint local cost + smoothness. ---------------------------
    // For interior waypoint j (path index s = j+1) we need its three
    // consecutive path points a=P[j], b=P[j+1], c=P[j+2]. Sloc = cost of the
    // two incident segments (a->b and b->c). Smoothness uses the discrete
    // acceleration a - 2b + c at that triple (units: m; squared → m^2).
    float smooth = 0.0f;
    for (int j = 0; j < kN; ++j) {
        float ax, ay, bx, by, cx, cy;
        path_point(j,     k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &ax, &ay); // P[j]
        path_point(j + 1, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &bx, &by); // P[j+1] (the waypoint)
        path_point(j + 2, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &cx, &cy); // P[j+2]

        const float sc_ab = segment_cost(field, gw, gh, cell_m, ax, ay, bx, by);
        const float sc_bc = segment_cost(field, gw, gh, cell_m, bx, by, cx, cy);
        Sloc[j * K + k] = sc_ab + sc_bc;                   // coalesced write (adjacent k adjacent)

        const float accx = ax - 2.0f * bx + cx;            // discrete acceleration (finite difference)
        const float accy = ay - 2.0f * by + cy;
        smooth += accx * accx + accy * accy;
    }

    cost[k] = total_obs + kWSmooth * smooth;               // total cost (verify gate only)
}

// ===========================================================================
// Host launcher (declared in kernels.cuh).
// ===========================================================================
void launch_stomp_score(int K,
                        const float* d_field, int gw, int gh, float cell_m,
                        const float* start2, const float* goal2,
                        const float* d_theta_x, const float* d_theta_y,
                        const float* d_epsx, const float* d_epsy,
                        float* d_Sloc, float* d_cost)
{
    if (K < 1 || !d_field || !start2 || !goal2 || !d_theta_x || !d_theta_y ||
        !d_epsx || !d_epsy || !d_Sloc || !d_cost) {
        std::fprintf(stderr, "launch_stomp_score: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }

    // Start and goal are 8 bytes each — pass them by value as scalars rather
    // than allocating device buffers. Every thread reads the same four floats,
    // so kernel arguments (which live in constant/parameter space, broadcast
    // to all threads) are the cheapest possible home for them.
    const float sx = start2[0], sy = start2[1];
    const float gx = goal2[0],  gy = goal2[1];

    const int threads = 256;                               // repo default geometry
    const int blocks = (K + threads - 1) / threads;        // ceil(K/threads): cover every rollout
    stomp_score_kernel<<<blocks, threads>>>(d_field, gw, gh, cell_m,
                                            sx, sy, gx, gy,
                                            d_theta_x, d_theta_y,
                                            d_epsx, d_epsy,
                                            d_Sloc, d_cost, K);
    CUDA_CHECK_LAST_ERROR("stomp_score_kernel launch");
}
