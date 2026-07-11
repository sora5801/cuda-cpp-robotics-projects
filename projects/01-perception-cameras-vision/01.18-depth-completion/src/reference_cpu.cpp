// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.18
//                     (Depth completion: sparse LiDAR + RGB -> dense depth)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): it is the CORRECTNESS ORACLE
// main.cu's VERIFY stage compares the GPU path against, and it is the
// TEACHING BASELINE that makes the GPU version legible as a transformation
// of something simple.
//
// Independence ruling applied in THIS file (see docs/PROJECT_TEMPLATE's
// reference_cpu.cpp for the full ruling text) — the choice this project
// makes:
//   * Data-layout contracts (Rigid3, LidarPointF, every constant) are
//     single-sourced in kernels.cuh and shared — divergent layouts would be
//     a bug class of their own, not independence.
//   * The ALGORITHMIC CORE of all four stages — the projection formula, the
//     Perona-Malik conductance, the diffusion update, the IDW weights — is
//     written HERE, completely independently from kernels.cu, in the
//     simplest possible sequential C++. None of it is shared as a
//     __host__ __device__ helper: every one of these formulas is short
//     enough (3-15 lines) that duplicating it is not "pure transcription"
//     of something too complex to write twice — it is exactly the kind of
//     formula this repo's ruling says SHOULD be independent, because that
//     is what makes the twin comparison in main.cu's VERIFY stage able to
//     catch a real bug (a sign error, an off-by-one neighbor, a swapped
//     axis) instead of comparing one formula to itself under a different
//     compiler.
//   * On top of the twin comparison, this project ALSO carries gates that
//     do not route through either implementation at all: the overall
//     accuracy, edge-quality, texture-trap, camo-edge, and input-fidelity
//     checks in main.cu all compare against the SYNTHETIC SCENE's exact
//     ray-cast ground truth, which lives entirely outside both kernels.cu
//     and this file — the independent gate the ruling requires even when
//     (as here) nothing is actually shared.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong,
// and then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side; every
// function below has a same-named counterpart there.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>       // sqrtf/expf/powf/fabsf/floorf — identical math library the GPU path also uses
#include <algorithm>   // std::max — used by max_channel_diff_cpu below
#include <vector>

// ---------------------------------------------------------------------------
// project_zbuffer_cpu — sequential nearest-wins z-buffer.
//
// Independent from project_zbuffer_kernel in the sense the file header
// promises: this is a plain "keep the smaller depth" compare, run one point
// at a time, in the ORDER the points were given — no atomics, because a
// single thread never races with itself. The GPU kernel needs its
// encode_depth_for_zbuffer/atomicMin trick only because MANY threads race on
// the same pixel; this loop is the reason that trick exists, made visible —
// depths here are compared directly as floats, no bit-encoding at all.
//
// Complexity: O(n_pts). out is (re)initialized to kInvalidDepth here so
// callers never need a separate "clear" step.
// ---------------------------------------------------------------------------
void project_zbuffer_cpu(const LidarPointF* pts, int n_pts, float* out_depth)
{
    for (int i = 0; i < kImagePixels; ++i) out_depth[i] = kInvalidDepth;

    for (int i = 0; i < n_pts; ++i) {
        const LidarPointF p = pts[i];

        // Same rigid transform as the GPU kernel, written out independently:
        // P_cam = R * P_lidar + t (kTCameraLidar, row-major R).
        const float* R = kTCameraLidar.R;
        const float xc = R[0] * p.x + R[1] * p.y + R[2] * p.z + kTCameraLidar.t[0];
        const float yc = R[3] * p.x + R[4] * p.y + R[5] * p.z + kTCameraLidar.t[1];
        const float zc = R[6] * p.x + R[7] * p.y + R[8] * p.z + kTCameraLidar.t[2];

        if (zc <= 0.0f || zc > kMaxDepthM) continue;

        const float inv_z = 1.0f / zc;
        const float u = kFx * xc * inv_z + kCx;
        const float v = kFy * yc * inv_z + kCy;

        const int px = static_cast<int>(std::floor(u + 0.5f));
        const int py = static_cast<int>(std::floor(v + 0.5f));
        if (px < 0 || px >= kImageWidth || py < 0 || py >= kImageHeight) continue;

        const int idx = py * kImageWidth + px;
        // Sequential "keep the nearest" — the single-threaded twin of the
        // GPU's atomicMin race resolution; same OUTCOME (smallest depth per
        // pixel wins), reached without any concurrency machinery at all.
        if (out_depth[idx] == kInvalidDepth || zc < out_depth[idx]) {
            out_depth[idx] = zc;
        }
    }
}

// max_channel_diff_cpu — the CPU twin's own copy of the color-edge-strength
// helper (max absolute per-channel difference over a PLANAR [3*kImagePixels]
// RGB buffer) — see compute_conductance_kernel's doc-comment in kernels.cuh
// for why full color, not grayscale.
static inline float max_channel_diff_cpu(const float* rgb, int a_idx, int b_idx)
{
    const float dr = std::fabs(rgb[a_idx]                     - rgb[b_idx]);
    const float dg = std::fabs(rgb[a_idx + kImagePixels]      - rgb[b_idx + kImagePixels]);
    const float db = std::fabs(rgb[a_idx + 2 * kImagePixels]  - rgb[b_idx + 2 * kImagePixels]);
    return std::max(dr, std::max(dg, db));
}

// ---------------------------------------------------------------------------
// compute_conductance_cpu — independent Perona-Malik conductance.
//
// Same formula as compute_conductance_kernel (g = exp(-(grad/K)^2)) and the
// same "conductance to the right/below neighbor, zero at the image border"
// layout — written here as two plain nested loops instead of a 2-D thread
// grid. Complexity: O(W*H).
// ---------------------------------------------------------------------------
void compute_conductance_cpu(const float* rgb, float* g_right, float* g_down)
{
    const float inv_k2 = 1.0f / (kConductanceK * kConductanceK);
    for (int y = 0; y < kImageHeight; ++y) {
        for (int x = 0; x < kImageWidth; ++x) {
            const int idx = y * kImageWidth + x;

            if (x + 1 < kImageWidth) {
                const float diff = max_channel_diff_cpu(rgb, idx, idx + 1);
                g_right[idx] = std::exp(-(diff * diff) * inv_k2);
            } else {
                g_right[idx] = 0.0f;
            }

            if (y + 1 < kImageHeight) {
                const float diff = max_channel_diff_cpu(rgb, idx, idx + kImageWidth);
                g_down[idx] = std::exp(-(diff * diff) * inv_k2);
            } else {
                g_down[idx] = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// diffusion_densify_cpu — kDiffusionIters sequential forward-Euler steps.
//
// Independent from launch_diffusion / diffusion_step_kernel: its own
// ping-pong pair (std::vector, not cudaMalloc), its own loop nest, its own
// Dirichlet-anchor branch. The UPDATE FORMULA is written out fresh here
// (not called from kernels.cu) — see this file's header for why that
// duplication is the deliberate, default choice for a project whose whole
// point is teaching this exact formula.
//
// Complexity: O(kDiffusionIters * W * H). At 160x120x400 iterations this
// is ~7.7M scalar updates — milliseconds on one CPU core, which is why
// main.cu's VERIFY stage can afford to run the FULL iteration count on
// both paths rather than truncating the CPU twin early.
// ---------------------------------------------------------------------------
void diffusion_densify_cpu(const float* sparse, const float* rgb, float unknown_seed, float* out)
{
    std::vector<float> g_right(static_cast<size_t>(kImagePixels));
    std::vector<float> g_down(static_cast<size_t>(kImagePixels));
    compute_conductance_cpu(rgb, g_right.data(), g_down.data());

    // Seed: anchors correct, unknown pixels start at unknown_seed (the mean
    // of the valid sparse samples, supplied by the caller) — the SAME
    // seeding choice launch_diffusion's seed_init_kernel makes (kernels.cu),
    // kept identical here so the two paths start from the same initial state.
    std::vector<float> buf_a(static_cast<size_t>(kImagePixels));
    for (int i = 0; i < kImagePixels; ++i) {
        const float v = sparse[i];
        buf_a[static_cast<size_t>(i)] = (v == kInvalidDepth) ? unknown_seed : v;
    }
    std::vector<float> buf_b(static_cast<size_t>(kImagePixels));

    float* cur = buf_a.data();
    float* nxt = buf_b.data();

    for (int it = 0; it < kDiffusionIters; ++it) {
        for (int y = 0; y < kImageHeight; ++y) {
            for (int x = 0; x < kImageWidth; ++x) {
                const int idx = y * kImageWidth + x;

                const float anchor = sparse[idx];
                if (anchor != kInvalidDepth) {
                    nxt[idx] = anchor;   // Dirichlet: fixed, every iteration
                    continue;
                }

                const float center = cur[idx];
                float flow = 0.0f;
                if (x > 0)                    flow += g_right[idx - 1]            * (cur[idx - 1]              - center);
                if (x + 1 < kImageWidth)      flow += g_right[idx]                * (cur[idx + 1]              - center);
                if (y > 0)                    flow += g_down[idx - kImageWidth]   * (cur[idx - kImageWidth]    - center);
                if (y + 1 < kImageHeight)     flow += g_down[idx]                 * (cur[idx + kImageWidth]    - center);

                nxt[idx] = center + kDiffusionDt * flow;
            }
        }
        float* tmp = cur; cur = nxt; nxt = tmp;   // ping-pong swap, same discipline as the GPU path
    }

    for (int i = 0; i < kImagePixels; ++i) out[i] = cur[i];
}

// ---------------------------------------------------------------------------
// idw_densify_cpu — independent fixed-radius inverse-distance-weighted
// baseline. Same window, same power, same "sample pixels pass through
// exactly / empty-window falls back to 0" contract as idw_kernel, written
// as plain nested loops. Complexity: O(W * H * kIdwRadiusPx^2).
// ---------------------------------------------------------------------------
void idw_densify_cpu(const float* sparse, float* out)
{
    for (int y = 0; y < kImageHeight; ++y) {
        for (int x = 0; x < kImageWidth; ++x) {
            const int idx = y * kImageWidth + x;
            const float here = sparse[idx];
            if (here != kInvalidDepth) {
                out[idx] = here;
                continue;
            }

            float wsum = 0.0f, vsum = 0.0f;
            const int x0 = x - kIdwRadiusPx < 0 ? 0 : x - kIdwRadiusPx;
            const int x1 = x + kIdwRadiusPx >= kImageWidth ? kImageWidth - 1 : x + kIdwRadiusPx;
            const int y0 = y - kIdwRadiusPx < 0 ? 0 : y - kIdwRadiusPx;
            const int y1 = y + kIdwRadiusPx >= kImageHeight ? kImageHeight - 1 : y + kIdwRadiusPx;

            for (int sy = y0; sy <= y1; ++sy) {
                for (int sx = x0; sx <= x1; ++sx) {
                    const float v = sparse[sy * kImageWidth + sx];
                    if (v == kInvalidDepth) continue;
                    const float dx = static_cast<float>(sx - x);
                    const float dy = static_cast<float>(sy - y);
                    const float dist = std::sqrt(dx * dx + dy * dy);
                    const float w = 1.0f / std::pow(dist, kIdwPower);
                    wsum += w;
                    vsum += w * v;
                }
            }
            out[idx] = (wsum > 0.0f) ? (vsum / wsum) : 0.0f;
        }
    }
}
