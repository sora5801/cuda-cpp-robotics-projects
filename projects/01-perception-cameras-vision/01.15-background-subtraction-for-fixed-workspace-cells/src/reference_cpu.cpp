// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.15
//                     (Background subtraction for fixed-workspace cells)
//
// WHY does a GPU repository ship a CPU implementation of everything? See
// docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's header for the two
// load-bearing reasons (correctness oracle + teaching baseline) and the
// INDEPENDENCE RULING this project follows to the letter:
//
//   * Data-layout contracts (px_index, the mode-major MOG array layout, all
//     of kernels.cuh SECTION 3's thresholds) are single-sourced in
//     kernels.cuh and used AS-IS here — duplicating an indexing FORMULA
//     across files is a bug class, not independence (the ruling's own
//     words).
//   * The ALGORITHMIC CORE — the actual classify+update math for all three
//     models, and the K=3 mode ranking/sort — is written a SECOND time
//     here, independently, in the simplest sequential C++ that expresses
//     the same formulas kernels.cu's kernels implement. Where kernels.cu's
//     mog_step_kernel hand-unrolls a 3-element compare-swap sort in
//     registers (the GPU-appropriate small-N idiom), this file instead
//     sorts with std::stable_sort and a lambda comparator — genuinely
//     different code, not the same logic retyped, so a shared bug in "the
//     sort" cannot hide behind two identical-looking implementations.
//
// This project's independent verification gates (README "Verification",
// THEORY.md "How we verify correctness") are the analytic absorption-time
// check and the designed-event IoU/false-positive gates — these do NOT
// route through this file at all, so even a bug that somehow lived in BOTH
// this oracle and kernels.cu (the failure mode the ruling's 13.03 story
// warns about) would still be caught by an INDEPENDENT check, not just the
// twin comparison. See main.cu's gate stage.
//
// Rules for this file: plain C++17, no CUDA headers, no OpenMP, no
// cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <algorithm>   // std::stable_sort, std::max — the CPU twin's deliberately-different sort from the GPU's compare-swap network
#include <array>
#include <cmath>       // std::fabs, std::sqrt
#include <vector>      // scratch buffer for the two-pass CPU morphological open

// ===========================================================================
// Model 1 — frame differencing (see kernels.cu's frame_diff_kernel for the
// GPU twin; this is the exact sequential form the kernel parallelizes).
// ===========================================================================
void frame_diff_cpu(const unsigned char* frames, const unsigned char* reference,
                     unsigned char* mask_out, int n_pixels, int total_elems, float threshold)
{
    for (int i = 0; i < total_elems; ++i) {
        const int p = i % n_pixels;
        const float diff = std::fabs(static_cast<float>(frames[i]) - static_cast<float>(reference[p]));
        mask_out[i] = (diff > threshold) ? 1u : 0u;
    }
}

// ===========================================================================
// Model 2 — running single Gaussian, one frame's classify+update. Called
// SEQ_T-1 times from main.cu's host loop — see kernels.cu's sg_step_kernel
// doc-comment for the full derivation; the five lines below ARE that
// derivation with a for-loop instead of a thread index.
// ===========================================================================
void sg_step_cpu(const unsigned char* frame_t, float* mu, float* var,
                  unsigned char* mask_out, int n_pixels,
                  float alpha, float k_sigma, float var_floor, float var_ceil)
{
    for (int i = 0; i < n_pixels; ++i) {
        const float I      = static_cast<float>(frame_t[i]);
        const float mu_old  = mu[i];
        const float var_old = var[i];
        const float sigma   = std::sqrt(std::max(var_old, var_floor));
        const float diff     = I - mu_old;
        const bool  is_fg     = std::fabs(diff) > k_sigma * sigma;

        mu[i]  = mu_old + alpha * diff;
        // var_ceil caps the STORED variance — see kernels.cu's sg_step_kernel
        // doc-comment for why an uncapped blind update can desensitize the
        // detector in 1-2 frames instead of the intended ~20-frame mean
        // convergence.
        var[i] = std::min((1.0f - alpha) * var_old + alpha * diff * diff, var_ceil);
        mask_out[i] = is_fg ? 1u : 0u;
    }
}

// ===========================================================================
// Model 3 — MOG-lite, K=3. Same seven-step algorithm as mog_step_kernel
// (kernels.cu has the full narrative doc-comment; this file does not repeat
// it verbatim on purpose — read the two side by side). The CODE SHAPE below
// deliberately differs from the kernel: a small local Mode{weight,mean,var}
// struct plus std::stable_sort, versus the kernel's flat float[3] registers
// and hand-unrolled compare-swap network. Same formulas, independent
// implementation, per this file's header ruling.
// ===========================================================================
namespace {

// One pixel's one mode, bundled for readability in the sequential version
// (the GPU kernel keeps the three fields in separate arrays instead — see
// kernels.cuh SECTION 4 for why that split matters for coalescing; it does
// not matter here, since the CPU has no warps to coalesce for).
struct Mode {
    float weight;
    float mean;
    float var;
};

}  // namespace

void mog_step_cpu(const unsigned char* frame_t, float* weight, float* mean, float* var,
                   unsigned char* mask_out, int n_pixels, int mog_k,
                   float match_k_sigma, float lr_weight, float lr_param,
                   float var_init, float var_floor, float w_init_new, float bg_fraction)
{
    const int K = 3;   // this project's fixed MOG_K — see kernels.cuh SECTION 3
    (void)mog_k;        // accepted for signature symmetry with the GPU launcher; asserted equal to K by construction (main.cu always passes MOG_K)

    for (int i = 0; i < n_pixels; ++i) {
        // ---- 1) load this pixel's K modes from the mode-major arrays ----
        std::array<Mode, 3> modes{};
        for (int k = 0; k < K; ++k) {
            const int gi = k * n_pixels + i;
            modes[static_cast<size_t>(k)] = Mode{weight[gi], mean[gi], var[gi]};
        }

        const float I = static_cast<float>(frame_t[i]);

        // ---- 2) match test: closest mode within match_k_sigma sigmas ----
        int best_idx = -1;
        float best_abs_d = 0.0f;
        std::array<float, 3> d{};
        for (int k = 0; k < K; ++k) {
            const float sigma_k = std::sqrt(std::max(modes[static_cast<size_t>(k)].var, var_floor));
            d[static_cast<size_t>(k)] = I - modes[static_cast<size_t>(k)].mean;
            const float abs_d = std::fabs(d[static_cast<size_t>(k)]);
            const bool matched = abs_d <= match_k_sigma * sigma_k;
            if (matched && (best_idx < 0 || abs_d < best_abs_d)) {
                best_idx = k;
                best_abs_d = abs_d;
            }
        }

        // ---- 3) matched update, or replace-weakest on no match ----------
        if (best_idx >= 0) {
            for (int k = 0; k < K; ++k) {
                Mode& m = modes[static_cast<size_t>(k)];
                m.weight = (k == best_idx) ? (m.weight + lr_weight * (1.0f - m.weight))
                                            : ((1.0f - lr_weight) * m.weight);
            }
            Mode& mm = modes[static_cast<size_t>(best_idx)];
            const float dm = d[static_cast<size_t>(best_idx)];
            mm.mean = mm.mean + lr_param * dm;
            mm.var  = (1.0f - lr_param) * mm.var + lr_param * dm * dm;
        } else {
            // Weakest = first strict minimum weight (lowest index wins ties).
            int weakest = 0;
            float weakest_w = modes[0].weight;
            for (int k = 1; k < K; ++k) {
                if (modes[static_cast<size_t>(k)].weight < weakest_w) {
                    weakest = k;
                    weakest_w = modes[static_cast<size_t>(k)].weight;
                }
            }
            for (int k = 0; k < K; ++k) {
                if (k != weakest) modes[static_cast<size_t>(k)].weight *= (1.0f - lr_weight);
            }
            modes[static_cast<size_t>(weakest)] = Mode{w_init_new, I, var_init};
        }

        // ---- 4) renormalize to sum 1 -------------------------------------
        float sum_w = modes[0].weight + modes[1].weight + modes[2].weight;
        if (sum_w > 1e-6f) {
            for (auto& m : modes) m.weight /= sum_w;
        }

        // ---- 5) rank by confidence = weight / sigma, descending ---------
        // std::stable_sort (not the kernel's hand-unrolled compare-swap
        // network — see this file's header) over index 0..2; "stable"
        // guarantees equal-confidence modes keep their original (lowest-
        // index-first) relative order, matching the kernel's documented
        // tie-break by a different mechanism.
        std::array<int, 3> order = {0, 1, 2};
        std::array<float, 3> conf{};
        for (int k = 0; k < K; ++k) {
            conf[static_cast<size_t>(k)] = modes[static_cast<size_t>(k)].weight
                                          / std::sqrt(std::max(modes[static_cast<size_t>(k)].var, var_floor));
        }
        std::stable_sort(order.begin(), order.end(),
                          [&conf](int a, int b) { return conf[static_cast<size_t>(a)] > conf[static_cast<size_t>(b)]; });

        // ---- 6) accumulate sorted weights until crossing bg_fraction ----
        std::array<bool, 3> is_background = {false, false, false};
        float cum = 0.0f;
        for (int r = 0; r < K; ++r) {
            const int k = order[static_cast<size_t>(r)];
            cum += modes[static_cast<size_t>(k)].weight;
            is_background[static_cast<size_t>(k)] = true;
            if (cum >= bg_fraction) break;
        }

        // ---- 7) classify --------------------------------------------------
        const bool is_fg = (best_idx < 0) || !is_background[static_cast<size_t>(best_idx)];
        mask_out[i] = is_fg ? 1u : 0u;

        // ---- write back (mode-major, same layout the load used) ---------
        for (int k = 0; k < K; ++k) {
            const int gi = k * n_pixels + i;
            weight[gi] = modes[static_cast<size_t>(k)].weight;
            mean[gi]   = modes[static_cast<size_t>(k)].mean;
            var[gi]    = modes[static_cast<size_t>(k)].var;
        }
    }
}

// ===========================================================================
// Post-processing — 3x3 morphological open (erode, then dilate). See
// kernels.cu's morph_erode_kernel/morph_dilate_kernel doc-comment for the
// structuring-element and boundary-condition rationale (shared with project
// 30.01). This CPU twin fuses erode+dilate into ONE pass over the sequence
// with an internal scratch buffer, rather than two separate exported
// functions — a harmless, independent structural difference from the GPU
// path (which needs two kernel launches because it has no single-thread
// "do both stages for this pixel before moving on" option: dilation at
// pixel p can depend on erosion at a NEIGHBOR pixel that a different GPU
// thread computed, so the GPU must fully finish erosion everywhere — a
// kernel-launch barrier — before dilation reads it. The CPU, being
// single-threaded and sequential, still needs the same two full passes
// (dilation genuinely needs ALL of erosion's output, not just this pixel's)
// — this function performs them as two explicit loops internally so the
// same "erosion must fully finish first" rule holds here too.
// ===========================================================================
void morph_open_cpu(const unsigned char* mask_raw, unsigned char* mask_open,
                     int w, int h, int total_elems)
{
    std::vector<unsigned char> eroded(static_cast<size_t>(total_elems));
    const int n_pixels = w * h;

    // Pass 1: erode every frame independently (frame_base keeps neighbor
    // lookups from crossing into an adjacent frame — see kernels.cu).
    for (int i = 0; i < total_elems; ++i) {
        const int frame_base = (i / n_pixels) * n_pixels;
        const int p = i % n_pixels;
        const int x = p % w, y = p / w;
        unsigned char all_set = 1u;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                const unsigned char nv = (nx >= 0 && nx < w && ny >= 0 && ny < h)
                                        ? mask_raw[frame_base + ny * w + nx] : 0u;
                all_set &= nv;
            }
        }
        eroded[static_cast<size_t>(i)] = all_set;
    }

    // Pass 2: dilate the FULLY-eroded buffer.
    for (int i = 0; i < total_elems; ++i) {
        const int frame_base = (i / n_pixels) * n_pixels;
        const int p = i % n_pixels;
        const int x = p % w, y = p / w;
        unsigned char any_set = 0u;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                const unsigned char nv = (nx >= 0 && nx < w && ny >= 0 && ny < h)
                                        ? eroded[static_cast<size_t>(frame_base + ny * w + nx)] : 0u;
                any_set |= nv;
            }
        }
        mask_open[i] = any_set;
    }
}
