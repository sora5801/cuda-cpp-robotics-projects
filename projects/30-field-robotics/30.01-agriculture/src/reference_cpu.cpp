// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 30.01
//                     Agriculture, Milestone 1: fruit detection + 3-D
//                     localization + ripeness
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md section 5):
//   1) The CORRECTNESS ORACLE. main.cu runs both paths and compares.
//   2) The TEACHING BASELINE. Reading this file first, then kernels.cu,
//      shows exactly what parallelization changed.
//
// THIS FILE'S ONE DELIBERATE DEPARTURE from "line-by-line twin": the
// connected-component labeler. Every other stage below (HSV, mask,
// morphology) is a direct sequential twin of its kernels.cu counterpart —
// same formulas, same thresholds, one core instead of one thread per pixel.
// CCL is different ON PURPOSE: kernels.cu's GPU kernel implements iterative
// LABEL PROPAGATION (the ratified teaching algorithm — parallel-friendly,
// but asymptotically wasteful per THEORY.md's convergence analysis); this
// file implements classic RASTER-SCAN UNION-FIND (the standard SERIAL CCL
// algorithm — a single pass plus near-constant-time find/union). They are
// DIFFERENT algorithms solving the SAME well-defined problem (partition the
// mask into 4-connected components), so if both are correct they MUST agree
// on the partition — and after both are canonicalized to the SAME
// convention ("label = minimum linear pixel index in the component"), that
// agreement becomes an EXACT integer equality check in main.cu, not a
// tolerance (kernels.cuh's file header proves the GPU side's fixed point
// already IS that convention; this file's canonicalize_labels() function
// makes the union-find side match it).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <vector>

// ===========================================================================
// Stage 1 — RGB -> HSV (line-by-line twin of rgb_to_hsv_kernel)
// ===========================================================================
void rgb_to_hsv_cpu(const unsigned char* rgb, float* h, float* s, float* v, int W, int H)
{
    const int N = W * H;
    for (int i = 0; i < N; ++i) {
        const float r = rgb[i * 3 + 0] * (1.0f / 255.0f);
        const float g = rgb[i * 3 + 1] * (1.0f / 255.0f);
        const float b = rgb[i * 3 + 2] * (1.0f / 255.0f);

        const float cmax = std::fmax(r, std::fmax(g, b));
        const float cmin = std::fmin(r, std::fmin(g, b));
        const float delta = cmax - cmin;

        float hue_deg = 0.0f;
        if (delta > 1e-6f) {
            if (cmax == r)      hue_deg = 60.0f * std::fmod((g - b) / delta, 6.0f);
            else if (cmax == g) hue_deg = 60.0f * ((b - r) / delta + 2.0f);
            else                hue_deg = 60.0f * ((r - g) / delta + 4.0f);
            if (hue_deg < 0.0f) hue_deg += 360.0f;
        }
        const float sat = (cmax > 1e-6f) ? (delta / cmax) : 0.0f;
        const float val = cmax;

        h[i] = hue_deg;
        s[i] = sat;
        v[i] = val;
    }
}

// ===========================================================================
// Stage 2 — fruit mask (line-by-line twin of fruit_mask_kernel)
// ===========================================================================
void fruit_mask_cpu(const float* h, const float* s, const float* v, unsigned char* mask, int W, int H)
{
    const int N = W * H;
    for (int i = 0; i < N; ++i) {
        const bool fruit_like = (h[i] < kHueMaxDeg) && (s[i] > kSatMin) && (v[i] > kValMin);
        mask[i] = fruit_like ? 1u : 0u;
    }
}

// ===========================================================================
// Stage 3 — morphological opening (line-by-line twin of the erode/dilate
// kernels: same 3x3 full-square structuring element, same zero-padding).
// ===========================================================================
void morph_erode_cpu(const unsigned char* mask_in, unsigned char* mask_out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            unsigned char all_set = 1u;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const unsigned char nv = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                            ? mask_in[ny * W + nx] : 0u;
                    all_set &= nv;
                }
            }
            mask_out[y * W + x] = all_set;
        }
    }
}

void morph_dilate_cpu(const unsigned char* mask_in, unsigned char* mask_out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            unsigned char any_set = 0u;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const unsigned char nv = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                            ? mask_in[ny * W + nx] : 0u;
                    any_set |= nv;
                }
            }
            mask_out[y * W + x] = any_set;
        }
    }
}

// ===========================================================================
// Stage 4 — connected-component labeling via raster-scan UNION-FIND (the
// classic serial CCL algorithm — see the file header for why this is a
// DELIBERATELY different algorithm from the GPU's label propagation).
// ===========================================================================

// find_root — path-compressed find: follow parent[] pointers to the root,
// then re-point every visited node DIRECTLY at that root on the way back
// (the classic union-find speedup — makes the amortized cost of a long
// chain of finds nearly O(1) rather than O(chain length) each time).
static int find_root(std::vector<int>& parent, int i)
{
    int root = i;
    while (parent[root] != root) root = parent[root];
    // Second pass: compress every node on the path directly to root.
    while (parent[i] != root) {
        const int next = parent[i];
        parent[i] = root;
        i = next;
    }
    return root;
}

// union_pixels — merge the components containing a and b (no-op if already
// the same component). "Union by attaching the larger root to the smaller"
// is not implemented (the extra rank/size bookkeeping is not needed for
// this scene's small components — see the file header's asymptotic note);
// arbitrarily attaching root(a) under root(b) is correct, just not
// worst-case-optimal, which is exactly the kind of "teaching beats
// cleverness" call CLAUDE.md section 1 asks for in a CORRECTNESS ORACLE.
static void union_pixels(std::vector<int>& parent, int a, int b)
{
    const int ra = find_root(parent, a);
    const int rb = find_root(parent, b);
    if (ra != rb) parent[ra] = rb;
}

void ccl_union_find_cpu(const unsigned char* mask, int* label, int W, int H)
{
    const int N = W * H;
    std::vector<int> parent(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) parent[i] = i;   // everyone starts as their own root

    // Pass 1 (raster scan): for each foreground pixel, union with its
    // ALREADY-VISITED foreground neighbors (left and up only — right and
    // down have not been scanned yet, and scanning is symmetric so this
    // still discovers every 4-connected edge exactly once from the "later"
    // endpoint's perspective).
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (!mask[i]) continue;
            if (x > 0 && mask[i - 1]) union_pixels(parent, i, i - 1);
            if (y > 0 && mask[i - W]) union_pixels(parent, i, i - W);
        }
    }

    // Pass 2: resolve every foreground pixel's root (fully path-compressed
    // by now), giving an initial label that is SOME pixel index in the
    // component but NOT YET the canonical minimum (union-find's root is
    // whichever pixel ended up on top of the union tree, essentially
    // arbitrary — the next pass fixes that).
    for (int i = 0; i < N; ++i)
        label[i] = mask[i] ? find_root(parent, i) : kLabelNone;

    // Pass 3 — CANONICALIZATION (the step kernels.cuh's file header
    // promises): relabel every foreground pixel to the MINIMUM linear index
    // among all pixels sharing its root, matching the convention the GPU's
    // label-propagation kernel converges to on its own. First, find each
    // root's minimum member; second, rewrite every pixel's label through
    // that lookup.
    std::vector<int> root_min(static_cast<size_t>(N), N);   // N = sentinel "not seen yet" (> any real index)
    for (int i = 0; i < N; ++i) {
        if (label[i] == kLabelNone) continue;
        const int r = label[i];   // == find_root(parent, i), already resolved above
        if (i < root_min[r]) root_min[r] = i;
    }
    for (int i = 0; i < N; ++i) {
        if (label[i] == kLabelNone) continue;
        label[i] = root_min[label[i]];
    }
}

// ===========================================================================
// Stage 5 — per-component statistics (sequential twin of kernels.cu's five
// small GPU kernels, folded into one function since there is no staging
// benefit on a single CPU core). Same dense [H*W]-indexed array layout as
// the GPU side (kernels.cuh Stage 5), so main.cu's array-level comparison
// checks like-for-like.
// ===========================================================================
void component_stats_cpu(const unsigned char* mask, const int* label,
                         const float* h, const float* depth,
                         int* comp_count, int* comp_sum_x, int* comp_sum_y,
                         int* comp_min_x, int* comp_max_x,
                         int* comp_min_y, int* comp_max_y,
                         float* comp_sum_hue, float* comp_final_depth,
                         int W, int H)
{
    const int N = W * H;

    // Local (not caller-visible) accumulators for the two-pass robust depth
    // estimate — the GPU side exposes these as separate device arrays for
    // its multi-kernel staging; the CPU side has no staging to expose, so
    // they live only as function-local scratch (still O(H*W) floats, still
    // cheap — this whole function runs once per demo invocation).
    std::vector<float> comp_sum_depth(static_cast<size_t>(N), 0.0f);
    std::vector<float> comp_mean_depth(static_cast<size_t>(N), 0.0f);
    std::vector<float> comp_sum_depth_inlier(static_cast<size_t>(N), 0.0f);
    std::vector<int>   comp_count_inlier(static_cast<size_t>(N), 0);

    // ---- init: identity elements, same values as component_stats_init_kernel ----
    for (int i = 0; i < N; ++i) {
        comp_count[i] = 0;
        comp_sum_x[i] = 0;
        comp_sum_y[i] = 0;
        comp_min_x[i] = W;
        comp_max_x[i] = -1;
        comp_min_y[i] = H;
        comp_max_y[i] = -1;
        comp_sum_hue[i] = 0.0f;
    }

    // ---- pass 1: accumulate count/position/bbox/hue/depth, sequentially ----
    // (the GPU's atomics become a plain "+=" here: only one thread — the
    // whole CPU core — ever touches any comp_*[L] slot, so no race exists
    // to guard against; this is precisely the difference the kernel launch
    // parallelizes, spelled out in kernels.cu's own header comment.)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (!mask[i]) continue;
            const int L = label[i];
            comp_count[L] += 1;
            comp_sum_x[L] += x;
            comp_sum_y[L] += y;
            if (x < comp_min_x[L]) comp_min_x[L] = x;
            if (x > comp_max_x[L]) comp_max_x[L] = x;
            if (y < comp_min_y[L]) comp_min_y[L] = y;
            if (y > comp_max_y[L]) comp_max_y[L] = y;
            comp_sum_hue[L] += h[i];
            comp_sum_depth[L] += depth[i];
        }
    }

    // ---- mean depth (elementwise, guarded) ----
    for (int i = 0; i < N; ++i)
        comp_mean_depth[static_cast<size_t>(i)] =
            (comp_count[i] > 0) ? (comp_sum_depth[static_cast<size_t>(i)] / static_cast<float>(comp_count[i])) : 0.0f;

    // ---- pass 2: robust inlier re-accumulation, same sensor-noise band ----
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (!mask[i]) continue;
            const int L = label[i];
            const float mean_z = comp_mean_depth[static_cast<size_t>(L)];
            const float sigma_z = kDepthNoiseK * mean_z * mean_z;
            const float band = kInlierSigmaMul * sigma_z;
            if (std::fabs(depth[i] - mean_z) <= band) {
                comp_sum_depth_inlier[static_cast<size_t>(L)] += depth[i];
                comp_count_inlier[static_cast<size_t>(L)] += 1;
            }
        }
    }

    // ---- finalize: inlier mean, falling back to the pass-1 mean ----
    for (int i = 0; i < N; ++i) {
        const int ci = comp_count_inlier[static_cast<size_t>(i)];
        comp_final_depth[i] = (ci > 0)
            ? (comp_sum_depth_inlier[static_cast<size_t>(i)] / static_cast<float>(ci))
            : comp_mean_depth[static_cast<size_t>(i)];
    }
}
