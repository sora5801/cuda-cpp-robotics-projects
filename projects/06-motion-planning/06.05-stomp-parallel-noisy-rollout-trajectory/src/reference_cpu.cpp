// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 06.05
//                     STOMP: parallel noisy-rollout trajectory optimization
//
// Two jobs in this project (both declared in kernels.cuh):
//
//   1. stomp_rollouts_cpu — the ORACLE twin of the GPU scoring kernel: same
//      field sampler, same segment integral, same smoothness term, sequential
//      over k. main.cu runs it against the GPU on identical inputs
//      (iteration 0) and requires agreement within a relative tolerance — the
//      §5 GPU-vs-CPU gate. It also serves as the honest timing baseline
//      ("a CPU manages a handful of rollouts" is measured here, not asserted).
//
//   2. evaluate_path_cost — score ONE full path on the host, for convergence
//      monitoring inside the optimization loop AND for the final collision
//      verdict. It uses a DENSER sub-sampling (kCheckSamples) than the scoring
//      path so the "collision-free with margin" test is strict about thin
//      obstacles, and it reports the maximum field value seen along the path
//      (the quantity the verdict tests). It shares the host field sampler with
//      the oracle, so the map is read one way on the host.
//
// The field sampler and segment-cost function below are line-by-line twins of
// the __device__ versions in kernels.cu — deliberate, documented duplication
// across the host/device boundary (diff the files: only the float-function
// spellings differ). This is exactly the pattern 08.01 uses for its dynamics.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization.
// If the reference is clever, it can be wrong, and then the oracle lies.
// (Compiled by the HOST compiler, cl.exe; kernels.cuh carries no CUDA
// constructs, so it includes cleanly here.)
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // shared constants, layouts, and signatures

#include <cmath>         // std::sqrt, std::fabs

// ---------------------------------------------------------------------------
// sample_field_cpu — bilinear lookup of the cost field (host twin of the
// device sample_field; see kernels.cu for the full commentary — the MATH must
// stay identical, so it is not repeated here).
// ---------------------------------------------------------------------------
static float sample_field_cpu(const float* field, int gw, int gh, float cell_m,
                              float x, float y)
{
    float gx = x / cell_m;
    float gy = y / cell_m;
    // Clamp into [0, g-1] so the +1 neighbour never steps off the array.
    if (gx < 0.0f) gx = 0.0f; else if (gx > gw - 1) gx = static_cast<float>(gw - 1);
    if (gy < 0.0f) gy = 0.0f; else if (gy > gh - 1) gy = static_cast<float>(gh - 1);

    const int ix = static_cast<int>(gx);
    const int iy = static_cast<int>(gy);
    const int ix1 = (ix + 1 < gw) ? ix + 1 : ix;
    const int iy1 = (iy + 1 < gh) ? iy + 1 : iy;

    const float fx = gx - static_cast<float>(ix);
    const float fy = gy - static_cast<float>(iy);

    const float c00 = field[iy  * gw + ix ];
    const float c10 = field[iy  * gw + ix1];
    const float c01 = field[iy1 * gw + ix ];
    const float c11 = field[iy1 * gw + ix1];

    const float c0 = c00 + fx * (c10 - c00);
    const float c1 = c01 + fx * (c11 - c01);
    return c0 + fy * (c1 - c0);
}

// ---------------------------------------------------------------------------
// segment_cost_cpu — obstacle line-integral along one segment, with a caller-
// chosen sample count (the oracle passes kSegSamples to match the kernel; the
// path evaluator passes the denser kCheckSamples). Host twin of the device
// segment_cost (which is hard-wired to kSegSamples).
// ---------------------------------------------------------------------------
static float segment_cost_cpu(const float* field, int gw, int gh, float cell_m,
                              float ax, float ay, float bx, float by, int nsamples)
{
    const float dx = bx - ax;
    const float dy = by - ay;
    const float seg_len = std::sqrt(dx * dx + dy * dy);
    const float ds = seg_len / static_cast<float>(nsamples);

    float acc = 0.0f;
    for (int s = 0; s < nsamples; ++s) {
        const float t = (static_cast<float>(s) + 0.5f) / static_cast<float>(nsamples);
        acc += sample_field_cpu(field, gw, gh, cell_m, ax + t * dx, ay + t * dy);
    }
    return acc * ds;
}

// path_point_cpu — the s-th path point of rollout k (host twin of the device
// path_point). See kernels.cu for the index/layout reasoning.
static void path_point_cpu(int s, int k, int K,
                           float sx, float sy, float gx, float gy,
                           const float* theta_x, const float* theta_y,
                           const float* epsx, const float* epsy,
                           float* out_x, float* out_y)
{
    if (s == 0)      { *out_x = sx; *out_y = sy; return; }
    if (s == kN + 1) { *out_x = gx; *out_y = gy; return; }
    const int j = s - 1;
    *out_x = theta_x[j] + epsx[j * K + k];
    *out_y = theta_y[j] + epsy[j * K + k];
}

// ---------------------------------------------------------------------------
// stomp_rollouts_cpu — all K rollouts, one after another (the GPU gives each
// its own thread). Reads the SAME transposed noise layout eps[j*K + k] the
// kernel reads — the layout is a data contract, not a GPU implementation
// detail, so the oracle honors it too. Fills cost[k] (the TOTAL cost) — the
// exact quantity the §5 verify gate compares against the kernel's cost[k].
// (The per-waypoint Sloc array is a GPU-only convenience the host update
// consumes; the verify gate checks the total, matching 08.01's approach.)
// ---------------------------------------------------------------------------
void stomp_rollouts_cpu(int K,
                        const float* field, int gw, int gh, float cell_m,
                        const float* start2, const float* goal2,
                        const float* theta_x, const float* theta_y,
                        const float* epsx, const float* epsy,
                        float* cost)
{
    const float sx = start2[0], sy = start2[1];
    const float gx = goal2[0],  gy = goal2[1];

    for (int k = 0; k < K; ++k) {
        // (1) Total obstacle cost: each of the kN+1 segments once.
        float px, py;
        path_point_cpu(0, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &px, &py);
        float total_obs = 0.0f;
        for (int s = 1; s <= kN + 1; ++s) {
            float cx, cy;
            path_point_cpu(s, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &cx, &cy);
            total_obs += segment_cost_cpu(field, gw, gh, cell_m, px, py, cx, cy, kSegSamples);
            px = cx; py = cy;
        }

        // (2) Smoothness: sum of squared discrete accelerations at each triple.
        float smooth = 0.0f;
        for (int j = 0; j < kN; ++j) {
            float ax, ay, bx, by, cx, cy;
            path_point_cpu(j,     k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &ax, &ay);
            path_point_cpu(j + 1, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &bx, &by);
            path_point_cpu(j + 2, k, K, sx, sy, gx, gy, theta_x, theta_y, epsx, epsy, &cx, &cy);
            const float accx = ax - 2.0f * bx + cx;
            const float accy = ay - 2.0f * by + cy;
            smooth += accx * accx + accy * accy;
        }

        cost[k] = total_obs + kWSmooth * smooth;
    }
}

// ---------------------------------------------------------------------------
// evaluate_path_cost — score ONE full path (P[0..npoints-1], endpoints
// included). Returns the total cost (obstacle path-integral + kWSmooth*
// smoothness) as a double, and writes the maximum field value sampled anywhere
// along the path into *out_max_field.
//
// Two roles:
//   * Convergence: main.cu calls this on the current NOMINAL trajectory each
//     iteration to watch the cost plateau. (The host built the field, so it
//     can score the noiseless path without a GPU round trip.)
//   * Verdict: after the loop, the returned max field value is tested against
//     the collision threshold, and the initial-vs-final total gives the
//     cost-reduction factor (both computed by THIS function, so the comparison
//     is apples-to-apples). Uses the dense kCheckSamples sub-sampling so a thin
//     obstacle a coarse sampling might skip cannot slip through the verdict.
// ---------------------------------------------------------------------------
double evaluate_path_cost(const float* field, int gw, int gh, float cell_m,
                          const float* px, const float* py, int npoints,
                          float* out_max_field)
{
    // Obstacle line-integral over all segments, and the running max field
    // value (checked at the dense sample points AND at the path vertices).
    double total_obs = 0.0;
    float  max_field = 0.0f;
    for (int s = 0; s + 1 < npoints; ++s) {
        total_obs += segment_cost_cpu(field, gw, gh, cell_m,
                                      px[s], py[s], px[s + 1], py[s + 1], kCheckSamples);
        // Sample the field densely along this segment for the collision test.
        const float dx = px[s + 1] - px[s];
        const float dy = py[s + 1] - py[s];
        for (int q = 0; q <= kCheckSamples; ++q) {
            const float t = static_cast<float>(q) / static_cast<float>(kCheckSamples);
            const float f = sample_field_cpu(field, gw, gh, cell_m, px[s] + t * dx, py[s] + t * dy);
            if (f > max_field) max_field = f;
        }
    }

    // Smoothness: squared discrete accelerations over the interior triples.
    double smooth = 0.0;
    for (int s = 1; s + 1 < npoints; ++s) {
        const double accx = static_cast<double>(px[s - 1]) - 2.0 * px[s] + px[s + 1];
        const double accy = static_cast<double>(py[s - 1]) - 2.0 * py[s] + py[s + 1];
        smooth += accx * accx + accy * accy;
    }

    if (out_max_field) *out_max_field = max_field;
    return total_obs + static_cast<double>(kWSmooth) * smooth;
}
