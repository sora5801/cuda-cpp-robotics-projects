// ===========================================================================
// kernels.cu — GPU kernels for project 01.15 (Background subtraction for
//              fixed-workspace cells)
//
// Role in the project
// --------------------
// All __global__ code lives here, plus the small host launch wrappers that
// own grid/block math (kept next to the kernel per repo convention — see
// docs/PROJECT_TEMPLATE/src/kernels.cu). Four kernels, one per teaching
// concept (CLAUDE.md §4): frame differencing, single-Gaussian EMA, MOG-lite,
// and 3x3 morphological open (erode+dilate) shared by all three models'
// masks. Full parameter/threshold documentation lives in kernels.cuh
// (SECTION 3) — this file documents the PER-THREAD WORK, not the constants.
//
// The three background models are all embarrassingly parallel over PIXELS
// (never over frames within a model — state at frame t depends on frame
// t-1, so the temporal loop lives in main.cu's host code, one kernel
// LAUNCH per frame; see THEORY.md "The GPU mapping" for why this is a
// sequence of small maps rather than one giant kernel). Frame differencing
// and morphology have no cross-frame state at all, so they each get ONE
// launch covering every (frame, pixel) pair in the whole sequence at once.
//
// Read this after: kernels.cuh.  Read this before: reference_cpu.cpp (the
// CPU twin — same math, no threads).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK_LAST_ERROR after every launch below
#include <cmath>                 // fabsf, sqrtf — device math intrinsics on NVCC's builtins

// ===========================================================================
// Model 1 — frame differencing (stateless MAP over the whole T*N array)
// ===========================================================================

// ---------------------------------------------------------------------------
// frame_diff_kernel — out[i] = |frames[i] - reference[i % n_pixels]| > threshold.
//
// Thread-to-data mapping: grid-stride over the FLAT (frame, pixel) index
// i = t*n_pixels + p. Because the reference frame never changes and every
// output element depends only on its own input element, this is the exact
// same MAP pattern the SAXPY placeholder taught — the only twist is the
// modulo to find which pixel of the (fixed) reference frame a given flat
// index belongs to. One launch covers all 160 frames: there is no reason to
// call this once per frame, because unlike the adaptive models there is no
// t-1 -> t dependency to serialize on.
//
// Parameters:
//   frames      — [total_elems] device, uint8 intensities, frame-major
//                 (frame t's pixels occupy [t*n_pixels, (t+1)*n_pixels)).
//   reference   — [n_pixels] device, the single captured reference frame
//                 (frame 0 in this project — see main.cu).
//   mask_out    — [total_elems] device OUT: 0/1 raw foreground flag.
//   n_pixels    — IMG_N (pixels per frame).
//   total_elems — SEQ_T * n_pixels.
//   threshold   — FRAME_DIFF_THRESHOLD (intensity units).
//
// Memory behavior: frames[] and mask_out[] are read/written with unit
// stride across the warp (coalesced); reference[] is re-read every
// n_pixels-th element by every thread, but its whole 12,288-byte footprint
// comfortably lives in L2 across the run, so this is not a bandwidth
// concern worth shared-memory tiling for a teaching kernel this small.
// ---------------------------------------------------------------------------
__global__ void frame_diff_kernel(const unsigned char* __restrict__ frames,
                                   const unsigned char* __restrict__ reference,
                                   unsigned char* __restrict__ mask_out,
                                   int n_pixels, int total_elems, float threshold)
{
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < total_elems; i += stride) {
        const int p = i % n_pixels;                              // which pixel of the reference frame
        const float diff = fabsf(static_cast<float>(frames[i]) - static_cast<float>(reference[p]));
        mask_out[i] = (diff > threshold) ? 1u : 0u;
    }
}

void launch_frame_diff(const unsigned char* d_frames, const unsigned char* d_reference,
                        unsigned char* d_mask, int n_pixels, int total_elems, float threshold)
{
    const int block = 256;
    int grid = (total_elems + block - 1) / block;
    if (grid > 4096) grid = 4096;   // grid-stride loop absorbs the remainder — same cap reasoning as the SAXPY placeholder
    frame_diff_kernel<<<grid, block>>>(d_frames, d_reference, d_mask, n_pixels, total_elems, threshold);
    CUDA_CHECK_LAST_ERROR("frame_diff_kernel launch");
}

// ===========================================================================
// Model 2 — running single Gaussian (ONE frame's classify+update per launch)
// ===========================================================================

// ---------------------------------------------------------------------------
// sg_step_kernel — one thread per PIXEL (not per frame — this kernel only
// ever sees one frame's worth of new samples; main.cu calls it SEQ_T-1
// times in a host loop, t = 1..SEQ_T-1, because mu/var at step t are a
// function of mu/var at step t-1: a genuine sequential recurrence across
// frames that cannot be parallelized over t without breaking correctness
// (see THEORY.md "The GPU mapping" for why "one thread per (frame,pixel)"
// is WRONG here, unlike frame differencing above).
//
// Per-pixel work (mirrors the EMA derivation in THEORY.md "The math"):
//   sigma = sqrt(max(var, var_floor))          — noise floor, see below
//   diff  = I(t) - mu                          — the innovation
//   fg    = |diff| > k_sigma * sigma           — classify BEFORE updating
//   mu   += alpha * diff                       — EMA mean update
//   var   = (1-alpha)*var + alpha*diff*diff    — EMA variance update
//
// Classify-then-update ordering matters: using yesterday's mu/var to judge
// today's sample is what makes "foreground" mean anything (a pixel cannot
// be foreground relative to a background estimate that has already eaten
// today's sample). This project uses a "blind" update — mu/var update
// EVERY frame regardless of the fg/bg call — a deliberate simplification
// documented in THEORY.md "Where this sits in the real world" (production
// systems often slow or skip updates on detected foreground, a "conservative
// update"); blind EMA is what makes the absorption-time closed form in
// THEORY.md "The math" exact rather than approximate.
//
// var_floor (SG_VAR_FLOOR): without it, a perfectly static run of identical
// uint8 samples drives var toward 0, and ANY future 1-intensity-unit sensor
// noise sample would then appear enormously many sigmas away and falsely
// fire foreground forever after. Flooring sigma at the sequence's own
// designed noise level (3.0, matching make_synthetic.py's NOISE_SIGMA)
// keeps the detector from becoming infinitely sensitive to noise it was
// never meant to react to.
//
// var_ceil (SG_VAR_CEIL): the mirror-image bug this project's own build
// surfaced (see kernels.cuh's SG_VAR_CEIL comment and THEORY.md "Numerical
// considerations" for the full story and the closed-form absorption-time
// derivation this ceiling makes possible): a single huge diff (a brand
// new object appearing) feeds alpha*diff*diff into var EVERY frame it
// keeps being blindly updated, which can balloon sigma fast enough to
// desensitize the detector in 1-2 frames via threshold inflation rather
// than genuine ~20-frame mean convergence. Capping the STORED variance
// (not just the sigma used for classification, which var_floor already
// bounds from below) keeps the detection threshold meaningful.
//
// Parameters: frame_t [n_pixels] device, this frame's samples; mu/var
// [n_pixels] device, IN/OUT model state; mask_out [n_pixels] device OUT.
// ---------------------------------------------------------------------------
__global__ void sg_step_kernel(const unsigned char* __restrict__ frame_t,
                                float* __restrict__ mu,
                                float* __restrict__ var,
                                unsigned char* __restrict__ mask_out,
                                int n_pixels,
                                float alpha, float k_sigma, float var_floor, float var_ceil)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;   // simple guard (not grid-stride): n_pixels = 12,288 fits one modest grid

    const float I        = static_cast<float>(frame_t[i]);
    const float mu_old    = mu[i];
    const float var_old   = var[i];
    const float sigma     = sqrtf(fmaxf(var_old, var_floor));
    const float diff       = I - mu_old;
    const bool  is_fg      = fabsf(diff) > k_sigma * sigma;

    mu[i]  = mu_old + alpha * diff;
    var[i] = fminf((1.0f - alpha) * var_old + alpha * diff * diff, var_ceil);
    mask_out[i] = is_fg ? 1u : 0u;
}

void launch_sg_step(const unsigned char* d_frame_t, float* d_mu, float* d_var,
                     unsigned char* d_mask_out, int n_pixels,
                     float alpha, float k_sigma, float var_floor, float var_ceil)
{
    const int block = 256;
    const int grid  = (n_pixels + block - 1) / block;   // 12,288/256 = 48 blocks exactly — no ragged tail this project's IMG_N happens to avoid, but the guard above handles it if a learner resizes the scene
    sg_step_kernel<<<grid, block>>>(d_frame_t, d_mu, d_var, d_mask_out, n_pixels, alpha, k_sigma, var_floor, var_ceil);
    CUDA_CHECK_LAST_ERROR("sg_step_kernel launch");
}

// ===========================================================================
// Model 3 — MOG-lite, K=3 modes per pixel (the didactic heart of this
// project — see README "The algorithm in brief" and THEORY.md "The math"
// for the full Stauffer & Grimson derivation this simplifies).
// ===========================================================================

// ---------------------------------------------------------------------------
// mog_step_kernel — one thread per pixel, one frame per launch (same
// sequential-launch discipline as sg_step_kernel and for the same reason:
// mode weights/means/variances at frame t depend on frame t-1).
//
// Per-pixel algorithm (mirrors THEORY.md "The math" step by step):
//   1. Load this pixel's K modes (weight, mean, var) into REGISTERS —
//      K=3 floats each, small enough that the compiler keeps them in
//      registers rather than spilling (checked in THEORY.md "The GPU
//      mapping" / Nsight occupancy note).
//   2. MATCH: for each mode k, is |I - mean_k| <= match_k_sigma * sigma_k?
//      Among matches, keep the CLOSEST (smallest |I - mean_k|); ties
//      (astronomically unlikely with real-valued noisy data) keep the
//      lowest index, simply because the scan below only replaces
//      best_idx on a STRICT improvement.
//   3a. If a mode matched (best_idx >= 0):
//         - weight EMA: matched mode moves toward 1, all others toward 0
//           (Stauffer-Grimson's w_k <- w_k + lr_w*(M_k - w_k), M_k=1 only
//           for the match) — THEORY.md shows this update conserves
//           sum(weight)=1 exactly whenever it started at 1 and exactly one
//           mode matches.
//         - mean/var EMA on the MATCHED mode only, using the pre-update
//           mean (the same diff already computed for the match test — no
//           mode "un-learns" from a sample it didn't explain).
//   3b. If NO mode matched: the weakest mode (lowest weight; ties keep the
//       lowest index, same "first strict improvement wins" scan) is
//       REPLACED outright — new mean = I, new var = var_init, new weight =
//       w_init_new — while every OTHER mode still decays as "unmatched."
//       This is how a pixel that starts single-modal (mode 0 = background)
//       grows a second, then third, recurring appearance over the sequence
//       (see THEORY.md's worked trace of the E4 blinking-lamp pixel).
//   4. RENORMALIZE the three weights to sum to 1 (a small numerical-hygiene
//      step: the "replace weakest" branch does not exactly conserve
//      sum=1 the way the matched branch provably does — see kernels.cuh
//      SECTION 3 / THEORY.md "Numerical considerations"). Guarded against
//      a (never-expected but always-checked) near-zero sum.
//   5. RANK the (possibly just-renormalized) modes by "confidence" =
//      weight / sigma — Stauffer-Grimson's own ranking, which favors
//      modes that are both FREQUENT (high weight) and TIGHT (low
//      variance) as more likely to be background than a rare, diffuse
//      mode. A hand-unrolled 3-element descending compare-swap sort (no
//      general sorting routine needed for K=3 — the CPU twin in
//      reference_cpu.cpp deliberately sorts a DIFFERENT way, std::stable_sort,
//      so the two are not the same code merely retyped).
//   6. ACCUMULATE sorted weights until the running sum reaches
//      bg_fraction (0.8): every mode visited up to and including the one
//      that crosses the threshold is "background."
//   7. CLASSIFY: foreground if no mode matched at all, OR if the matched
//      mode is not in that background set.
//
// WARP DIVERGENCE — the natural home for this lesson (CLAUDE.md brief):
// step 2's match loop and step 3's matched/no-match branch are DATA
// DEPENDENT per pixel. Within one 32-thread warp, some pixels (lanes) take
// the "matched mode 0" path, others "matched mode 2", others "no match,
// replace weakest" — the SIMT hardware executes ALL of these code paths
// SERIALLY for the warp, masking off the lanes that don't apply to each
// path, then reconverges. For K=3 this divergence is bounded and small (at
// most ~4 distinct paths: match k=0/1/2, or no-match), so the cost here is
// modest — but it is real, and it is why "small, fixed K, unrolled" is the
// GPU-friendly choice: a K=100 full Gaussian mixture would multiply this
// divergence cost per pixel far more than it would help a CPU thread doing
// the identical branch (a CPU pays for the branch it TAKES; a warp pays,
// approximately, for every DISTINCT branch ANY of its 32 lanes take). See
// THEORY.md "The GPU mapping" for the full argument and a back-of-envelope
// cost estimate.
//
// Parameters: frame_t [n_pixels]; weight/mean/var [mog_k*n_pixels],
// mode-major (kernels.cuh SECTION 4 — index k*n_pixels+i); mask_out
// [n_pixels] OUT. mog_k is passed as a runtime int but this kernel caps
// its LOCAL (register) arrays at 3 — a general-K version would need
// shared or global scratch instead of fixed-size registers; this project's
// K is fixed at 3 by design (kernels.cuh MOG_K), so the cap is never hit.
// ---------------------------------------------------------------------------
__global__ void mog_step_kernel(const unsigned char* __restrict__ frame_t,
                                 float* __restrict__ weight,
                                 float* __restrict__ mean,
                                 float* __restrict__ var,
                                 unsigned char* __restrict__ mask_out,
                                 int n_pixels, int mog_k,
                                 float match_k_sigma, float lr_weight, float lr_param,
                                 float var_init, float var_floor,
                                 float w_init_new, float bg_fraction)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;

    const int K = 3;   // register cap — see the doc-comment above; mog_k is expected == K in this project
    float w[K], m[K], v[K], d[K];
    bool  matched[K];

    // ---- 1) load this pixel's K modes (mode-major global -> registers) ---
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int gi = k * n_pixels + i;   // SECTION 4 layout: mode k, pixel i
        w[k] = weight[gi];
        m[k] = mean[gi];
        v[k] = var[gi];
    }

    const float I = static_cast<float>(frame_t[i]);

    // ---- 2) match test against every mode -----------------------------
    int best_idx = -1;
    float best_abs_d = 0.0f;   // only meaningful once best_idx >= 0
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float sigma_k = sqrtf(fmaxf(v[k], var_floor));
        d[k] = I - m[k];
        matched[k] = fabsf(d[k]) <= match_k_sigma * sigma_k;
        if (matched[k] && (best_idx < 0 || fabsf(d[k]) < best_abs_d)) {
            best_idx = k;
            best_abs_d = fabsf(d[k]);
        }
    }

    // ---- 3) matched-mode update, OR replace-weakest on no match -------
    if (best_idx >= 0) {
        // 3a. weight EMA: matched mode -> 1, everyone else -> 0 (see doc-comment).
#pragma unroll
        for (int k = 0; k < K; ++k) {
            w[k] = (k == best_idx) ? (w[k] + lr_weight * (1.0f - w[k]))
                                    : ((1.0f - lr_weight) * w[k]);
        }
        // Matched mode's mean/var EMA, using the ALREADY-COMPUTED d[] from
        // the match test above (the pre-update mean's innovation).
        m[best_idx] = m[best_idx] + lr_param * d[best_idx];
        v[best_idx] = (1.0f - lr_param) * v[best_idx] + lr_param * d[best_idx] * d[best_idx];
    } else {
        // 3b. find the weakest mode (first strict minimum wins ties).
        int weakest = 0;
        float weakest_w = w[0];
#pragma unroll
        for (int k = 1; k < K; ++k) {
            if (w[k] < weakest_w) { weakest = k; weakest_w = w[k]; }
        }
#pragma unroll
        for (int k = 0; k < K; ++k) {
            if (k != weakest) w[k] = (1.0f - lr_weight) * w[k];   // "unmatched" decay for everyone but the replacement
        }
        w[weakest] = w_init_new;
        m[weakest] = I;
        v[weakest] = var_init;
    }

    // ---- 4) renormalize weights to sum 1 (numerical hygiene — see doc) --
    float sum_w = w[0] + w[1] + w[2];
    if (sum_w > 1e-6f) {   // guard: never expected to be this small, but a silent divide-by-zero would be worse than a documented guard
#pragma unroll
        for (int k = 0; k < K; ++k) w[k] /= sum_w;
    }

    // ---- 5) rank by confidence = weight / sigma, descending ------------
    // Hand-unrolled 3-element compare-swap network (a general sort would
    // be overkill for K=3, and device-side std::sort is not available
    // without Thrust — see THEORY.md "The GPU mapping"). idxs[] starts in
    // mode-index order so ties (never swapped, since swaps use STRICT '<')
    // keep the lower original index first.
    float conf[K];
#pragma unroll
    for (int k = 0; k < K; ++k) conf[k] = w[k] / sqrtf(fmaxf(v[k], var_floor));
    int idxs[K] = {0, 1, 2};
    if (conf[idxs[0]] < conf[idxs[1]]) { int t = idxs[0]; idxs[0] = idxs[1]; idxs[1] = t; }
    if (conf[idxs[1]] < conf[idxs[2]]) { int t = idxs[1]; idxs[1] = idxs[2]; idxs[2] = t; }
    if (conf[idxs[0]] < conf[idxs[1]]) { int t = idxs[0]; idxs[0] = idxs[1]; idxs[1] = t; }

    // ---- 6) accumulate sorted weights until we cross bg_fraction -------
    bool is_background[K] = {false, false, false};
    float cum = 0.0f;
#pragma unroll
    for (int r = 0; r < K; ++r) {
        cum += w[idxs[r]];
        is_background[idxs[r]] = true;
        if (cum >= bg_fraction) break;   // this mode CROSSED the threshold — it and everything ranked above it are background
    }

    // ---- 7) classify ----------------------------------------------------
    const bool is_fg = (best_idx < 0) || !is_background[best_idx];
    mask_out[i] = is_fg ? 1u : 0u;

    // ---- write updated state back (mode-major, same layout as the load) -
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int gi = k * n_pixels + i;
        weight[gi] = w[k];
        mean[gi]   = m[k];
        var[gi]    = v[k];
    }
}

void launch_mog_step(const unsigned char* d_frame_t, float* d_weight, float* d_mean, float* d_var,
                      unsigned char* d_mask_out, int n_pixels, int mog_k,
                      float match_k_sigma, float lr_weight, float lr_param,
                      float var_init, float var_floor, float w_init_new, float bg_fraction)
{
    const int block = 256;
    const int grid  = (n_pixels + block - 1) / block;
    mog_step_kernel<<<grid, block>>>(d_frame_t, d_weight, d_mean, d_var, d_mask_out,
                                      n_pixels, mog_k, match_k_sigma, lr_weight, lr_param,
                                      var_init, var_floor, w_init_new, bg_fraction);
    CUDA_CHECK_LAST_ERROR("mog_step_kernel launch");
}

// ===========================================================================
// Post-processing — 3x3 morphological OPEN (erode, then dilate), applied to
// every model's raw mask before any gate reads it (README "Post-
// processing"). Same 8-connected, zero-padded convention as project
// 30.01's fruit-mask cleanup stage (projects/30-field-robotics/30.01-
// agriculture/src/kernels.cu) — cited here rather than reinvented, per
// CLAUDE.md's "study, do not copy" spirit applied WITHIN this repo too:
// the two kernels below are still written independently for this project's
// own T*N grid-stride shape (not copy-pasted), but the STRUCTURING ELEMENT
// and boundary rule are the same deliberate choice.
//
// WHY OPENING, TAUGHT ONCE HERE (see THEORY.md "The algorithm" for the
// full picture): the raw per-pixel classifiers above have no notion of
// SPATIAL coherence — a single noisy pixel that happens to cross its
// threshold is, to the classifier, indistinguishable from the first pixel
// of a real object. Erosion deletes anything that is not at least
// 1-pixel-margin thick everywhere (a lone speck vanishes entirely, because
// it has fewer than 9 set neighbors anywhere); dilation then regrows what
// erosion left standing by the same 1 pixel. Genuine objects (the arm, the
// box — tens of pixels across) survive with their footprint nearly intact;
// salt-and-pepper misclassifications, which this project's per-pixel
// thresholds WILL occasionally produce even inside the noise floor, do
// not. This is precisely why every downstream gate compares OPENED masks,
// not raw ones — a raw-mask false-positive rate would conflate "the
// classifier's per-pixel threshold" with "whether anyone bothered to clean
// it up," and this project wants to isolate the FIRST question.
// ===========================================================================

// morph_erode_kernel — grid-stride over the WHOLE T*N mask at once (no
// cross-frame dependency: frame t's opening only ever reads frame t's raw
// mask). x, y are recovered from the pixel-local part of the flat index;
// the frame index itself never participates in the neighbor offsets, so a
// neighbor lookup can never accidentally read an adjacent frame.
__global__ void morph_erode_kernel(const unsigned char* __restrict__ mask_in,
                                    unsigned char* __restrict__ mask_out,
                                    int w, int h, int total_elems)
{
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int n_pixels = w * h;
    for (; i < total_elems; i += stride) {
        const int frame_base = (i / n_pixels) * n_pixels;   // start of THIS frame's mask
        const int p = i % n_pixels;
        const int x = p % w, y = p / w;

        unsigned char all_set = 1u;
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                const unsigned char nv = (nx >= 0 && nx < w && ny >= 0 && ny < h)
                                        ? mask_in[frame_base + ny * w + nx] : 0u;   // zero-padding, see file header
                all_set &= nv;
            }
        }
        mask_out[i] = all_set;
    }
}

// morph_dilate_kernel — the mirror of erode: OR instead of AND over the
// same 3x3 zero-padded neighborhood.
__global__ void morph_dilate_kernel(const unsigned char* __restrict__ mask_in,
                                     unsigned char* __restrict__ mask_out,
                                     int w, int h, int total_elems)
{
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int n_pixels = w * h;
    for (; i < total_elems; i += stride) {
        const int frame_base = (i / n_pixels) * n_pixels;
        const int p = i % n_pixels;
        const int x = p % w, y = p / w;

        unsigned char any_set = 0u;
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                const unsigned char nv = (nx >= 0 && nx < w && ny >= 0 && ny < h)
                                        ? mask_in[frame_base + ny * w + nx] : 0u;
                any_set |= nv;
            }
        }
        mask_out[i] = any_set;
    }
}

void launch_morph_open(const unsigned char* d_mask_raw, unsigned char* d_mask_open,
                        unsigned char* d_scratch, int w, int h, int total_elems)
{
    const int block = 256;
    int grid = (total_elems + block - 1) / block;
    if (grid > 4096) grid = 4096;
    // erode(raw) -> scratch, then dilate(scratch) -> open. Two launches,
    // covering the entire sequence each time (see the kernels' own header
    // comment for why one launch suffices per stage).
    morph_erode_kernel<<<grid, block>>>(d_mask_raw, d_scratch, w, h, total_elems);
    CUDA_CHECK_LAST_ERROR("morph_erode_kernel launch");
    morph_dilate_kernel<<<grid, block>>>(d_scratch, d_mask_open, w, h, total_elems);
    CUDA_CHECK_LAST_ERROR("morph_dilate_kernel launch");
}
