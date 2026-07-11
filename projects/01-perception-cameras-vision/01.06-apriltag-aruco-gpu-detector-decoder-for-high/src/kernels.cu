// ===========================================================================
// kernels.cu — GPU kernels for project 01.06
//              AprilTag / ArUco GPU detector-decoder for high-rate fiducial
//              localization
//
// Ten launch wrappers, two families of launch geometry (kernels.cuh's file
// header names the contrast explicitly):
//   PIXEL-parallel (stages 1-2 and the stats scatter): one thread per pixel,
//     grid = ceil(W*H/256), block = 256 — the same launch idiom as 30.01.
//   CANDIDATE-parallel (stages 3-6): one thread per SURVIVING component
//     (typically single digits to a few dozen per scene), grid = ceil(n/32),
//     block = 32 — small enough that one warp usually covers the whole
//     launch; occupancy is irrelevant at this scale, CORRECTNESS and
//     per-thread SEQUENTIAL work (an 8x8 linear solve, a 36-point sample
//     loop) are what matters, which is exactly the lesson this half of the
//     pipeline teaches (THEORY.md "The GPU mapping" makes the contrast
//     explicit with both stages' actual measured occupancy).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cmath>

// ===========================================================================
// Small device-only helpers (NOT shared with the CPU twin — see
// reference_cpu.cpp's independence ruling: bilinear sampling and the 64-bit
// CAS-loop atomics are part of "the algorithm/mechanism under test", not
// pure data-layout, so each side types its own copy).
// ===========================================================================

// ---------------------------------------------------------------------------
// atomicMin64 / atomicMax64 — 64-bit unsigned min/max built from the ONE
// atomic primitive every CUDA-capable device guarantees for 64-bit words,
// atomicCAS. Native atomicMin/atomicMax on "unsigned long long" exist on
// sm_75+ (this repo's floor) but relying on the CAS-loop form here is a
// deliberate teaching choice (CLAUDE.md paragraph 1, "no black boxes"): it
// shows exactly HOW a compare-and-swap loop builds an arbitrary atomic
// read-modify-write from CAS alone, and is portable to any future GPU this
// project might target without checking library availability. The loop:
// read the current value, and if it is already <= (or >=) the new value we
// are done (early break, no wasted CAS); otherwise try to swap our value in,
// but only if nobody else changed the slot since our read (the CAS's
// "assumed" check) — if someone did, retry with the fresh value. This is the
// STANDARD compare-and-swap retry idiom, the same one that implements every
// lock-free data structure.
// ---------------------------------------------------------------------------
__device__ inline unsigned long long atomicMin64(unsigned long long* addr, unsigned long long val)
{
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        if (old <= val) return old;      // already <= : nothing to do, no CAS needed
        assumed = old;
        old = atomicCAS(addr, assumed, val);
    } while (assumed != old);            // if old != assumed, someone else won the race; retry
    return old;
}
__device__ inline unsigned long long atomicMax64(unsigned long long* addr, unsigned long long val)
{
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        if (old >= val) return old;
        assumed = old;
        old = atomicCAS(addr, assumed, val);
    } while (assumed != old);
    return old;
}

// ---------------------------------------------------------------------------
// bilerp_u8 / bilerp_f32 — bilinear sample of a [H*W] image at a fractional
// pixel coordinate (px, py), CLAMPING both the integer corner indices and
// the final query point to the image bounds (so a ray that overshoots the
// image edge by a few pixels — which the corner-refinement search
// deliberately allows, see launch_corner_refine below — degrades gracefully
// to an edge-replicated sample rather than reading out of bounds).
// ---------------------------------------------------------------------------
__device__ inline float bilerp_u8(const unsigned char* img, int W, int H, float px, float py)
{
    px = fminf(fmaxf(px, 0.0f), static_cast<float>(W - 1));
    py = fminf(fmaxf(py, 0.0f), static_cast<float>(H - 1));
    const int x0 = static_cast<int>(floorf(px));
    const int y0 = static_cast<int>(floorf(py));
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    const float tx = px - static_cast<float>(x0);
    const float ty = py - static_cast<float>(y0);
    const float v00 = static_cast<float>(img[y0 * W + x0]);
    const float v10 = static_cast<float>(img[y0 * W + x1]);
    const float v01 = static_cast<float>(img[y1 * W + x0]);
    const float v11 = static_cast<float>(img[y1 * W + x1]);
    const float top = v00 + (v10 - v00) * tx;
    const float bot = v01 + (v11 - v01) * tx;
    return top + (bot - top) * ty;
}
__device__ inline float bilerp_f32(const float* img, int W, int H, float px, float py)
{
    px = fminf(fmaxf(px, 0.0f), static_cast<float>(W - 1));
    py = fminf(fmaxf(py, 0.0f), static_cast<float>(H - 1));
    const int x0 = static_cast<int>(floorf(px));
    const int y0 = static_cast<int>(floorf(py));
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    const float tx = px - static_cast<float>(x0);
    const float ty = py - static_cast<float>(y0);
    const float v00 = img[y0 * W + x0];
    const float v10 = img[y0 * W + x1];
    const float v01 = img[y1 * W + x0];
    const float v11 = img[y1 * W + x1];
    const float top = v00 + (v10 - v00) * tx;
    const float bot = v01 + (v11 - v01) * tx;
    return top + (bot - top) * ty;
}

// ===========================================================================
// Stage 1 — adaptive threshold: separable box filter (local mean) + compare.
// ===========================================================================

// ---------------------------------------------------------------------------
// box_sum_h_kernel — horizontal half of the box filter: row_sum[y*W+x] =
// SUM over dx in [-r,r] of gray[y*W + clamp(x+dx,0,W-1)]. A STENCIL (each
// output reads 2r+1 inputs), CLAMP boundary handling (replicate the edge
// pixel) rather than zero-padding — zero-padding would pull the local mean
// DOWN near the image border and could spuriously mark a bright border
// region as "foreground", exactly the wrong direction for a detector meant
// to be conservative (THEORY.md "Numerical considerations").
//
// Numerical note (why this sum is EXACT float32 arithmetic, load-bearing for
// main.cu's VERIFY tolerance): every addend is a uint8 in [0,255]; a running
// sum of up to (2*kBoxRadius+1) <= 255 such terms never exceeds ~65025, far
// below float32's 2^24 exact-integer ceiling — so THIS sum has ZERO rounding
// error regardless of summation order, on ANY IEEE-754-compliant adder
// (device or host). The vertical pass below inherits the same property.
// ---------------------------------------------------------------------------
__global__ void box_sum_h_kernel(const unsigned char* __restrict__ gray,
                                 float* __restrict__ row_sum, int W, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    float s = 0.0f;
    for (int dx = -kBoxRadius; dx <= kBoxRadius; ++dx) {
        int nx = x + dx;
        nx = nx < 0 ? 0 : (nx >= W ? W - 1 : nx);   // clamp: replicate edge pixel
        s += static_cast<float>(gray[y * W + nx]);
    }
    row_sum[i] = s;
}
void launch_box_sum_h(const unsigned char* d_gray, float* d_row_sum, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    box_sum_h_kernel<<<grid, block>>>(d_gray, d_row_sum, W, H);
    CUDA_CHECK_LAST_ERROR("box_sum_h_kernel launch");
}

// ---------------------------------------------------------------------------
// box_sum_v_kernel — vertical half: sums row_sum over dy in [-r,r] (same
// clamp rule), then divides by the window AREA to finish the local MEAN.
// Chaining two 1-D passes (2*(2r+1) additions) instead of one 2-D pass
// ((2r+1)^2 additions) is the classic separable-filter GPU lesson: a 25x25
// box (625 taps) collapses into 25+25=50 taps per pixel, a >12x reduction in
// memory traffic and arithmetic for this project's kBoxRadius=12 — THEORY.md
// "The GPU mapping" derives the general (2r+1)^2 -> 2*(2r+1) saving.
// ---------------------------------------------------------------------------
__global__ void box_sum_v_kernel(const float* __restrict__ row_sum,
                                 float* __restrict__ local_mean, int W, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    float s = 0.0f;
    for (int dy = -kBoxRadius; dy <= kBoxRadius; ++dy) {
        int ny = y + dy;
        ny = ny < 0 ? 0 : (ny >= H ? H - 1 : ny);
        s += row_sum[ny * W + x];
    }
    const float area = static_cast<float>((2 * kBoxRadius + 1) * (2 * kBoxRadius + 1));
    local_mean[i] = s / area;
}
void launch_box_sum_v(const float* d_row_sum, float* d_local_mean, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    box_sum_v_kernel<<<grid, block>>>(d_row_sum, d_local_mean, W, H);
    CUDA_CHECK_LAST_ERROR("box_sum_v_kernel launch");
}

// ---------------------------------------------------------------------------
// adaptive_threshold_kernel — the "mean minus C" rule: mask=1 (foreground /
// candidate tag ink) iff gray < local_mean - kThreshBiasC. A pure elementwise
// MAP over two aligned [H*W] arrays; no neighbor reads (the box filter above
// already did all the neighborhood work), so this kernel is memory-bound and
// trivially simple — the "boring but correct" final step of the classic
// three-kernel separable-adaptive-threshold recipe.
// ---------------------------------------------------------------------------
__global__ void adaptive_threshold_kernel(const unsigned char* __restrict__ gray,
                                          const float* __restrict__ local_mean,
                                          unsigned char* __restrict__ mask, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    mask[i] = (static_cast<float>(gray[i]) < local_mean[i] - kThreshBiasC) ? 1u : 0u;
}
void launch_adaptive_threshold(const unsigned char* d_gray, const float* d_local_mean,
                               unsigned char* d_mask, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    adaptive_threshold_kernel<<<grid, block>>>(d_gray, d_local_mean, d_mask, N);
    CUDA_CHECK_LAST_ERROR("adaptive_threshold_kernel launch");
}

// ===========================================================================
// Stage 2 — connected-component labeling by iterative label propagation.
// Identical ALGORITHM and convergence argument to 30.01's stage 4 (cite:
// projects/30-field-robotics/30.01-agriculture/src/kernels.cu) — re-typed
// independently here for this project's own mask/label layout, per
// CLAUDE.md's "deliberate, documented duplication" self-containment rule
// (this is normal cross-PROJECT duplication, not the twin-independence
// ruling, which is about GPU-vs-CPU within one project).
//
// CONVERGENCE, briefly (full argument in 30.01's kernels.cu, reproduced in
// this project's THEORY.md "The algorithm"): every label only ever
// DECREASES (atomicMin), bounded below by 0, so it converges in finitely
// many sweeps to the UNIQUE fixed point label[p] = min over p's 4-connected
// foreground component of the linear pixel index — independent of thread
// scheduling. This is what lets main.cu compare the GPU's label image
// against the CPU's differently-algorithmed (union-find) oracle with EXACT
// integer equality after canonicalization.
// ---------------------------------------------------------------------------
__global__ void ccl_init_kernel(const unsigned char* __restrict__ mask,
                                int* __restrict__ label, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    label[i] = mask[i] ? i : kLabelNone;
}
void launch_ccl_init(const unsigned char* d_mask, int* d_label, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    ccl_init_kernel<<<grid, block>>>(d_mask, d_label, N);
    CUDA_CHECK_LAST_ERROR("ccl_init_kernel launch");
}

__global__ void ccl_propagate_sweep_kernel(const unsigned char* __restrict__ mask,
                                           int* __restrict__ label,
                                           int W, int H, int* __restrict__ changed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (!mask[i]) return;

    const int x = i % W, y = i / W;
    int best = label[i];
    if (x > 0     && mask[i - 1]) best = min(best, label[i - 1]);
    if (x < W - 1 && mask[i + 1]) best = min(best, label[i + 1]);
    if (y > 0     && mask[i - W]) best = min(best, label[i - W]);
    if (y < H - 1 && mask[i + W]) best = min(best, label[i + W]);

    if (best < label[i]) {
        atomicMin(&label[i], best);
        atomicOr(changed, 1);
    }
}
void launch_ccl_propagate_sweep(const unsigned char* d_mask, int* d_label, int W, int H, int* d_changed)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    ccl_propagate_sweep_kernel<<<grid, block>>>(d_mask, d_label, W, H, d_changed);
    CUDA_CHECK_LAST_ERROR("ccl_propagate_sweep_kernel launch");
}

// ===========================================================================
// Component statistics — PIXEL-parallel atomic scatter, dense [H*W]-indexed
// accumulators keyed directly by canonical label (same design as 30.01's
// Stage 5 — see that file's header for why compaction is deliberately NOT
// done on the GPU here).
// ===========================================================================
__global__ void component_stats_init_kernel(int* __restrict__ count,
                                            unsigned long long* __restrict__ sum_x,
                                            unsigned long long* __restrict__ sum_y,
                                            int* __restrict__ min_x, int* __restrict__ max_x,
                                            int* __restrict__ min_y, int* __restrict__ max_y,
                                            unsigned long long* __restrict__ key_min_sum,
                                            unsigned long long* __restrict__ key_max_sum,
                                            unsigned long long* __restrict__ key_min_diff,
                                            unsigned long long* __restrict__ key_max_diff,
                                            int W, int H, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    count[i] = 0;
    sum_x[i] = 0ull;
    sum_y[i] = 0ull;
    min_x[i] = W; max_x[i] = -1;    // identities: no real x reaches W or goes below 0
    min_y[i] = H; max_y[i] = -1;
    // Sentinels for the packed-key extrema: ~0ull is the LARGEST possible
    // 64-bit key (so the first real atomicMin always overwrites it); 0ull is
    // the SMALLEST (so the first real atomicMax always overwrites it) — see
    // pack_corner_key()'s file-header comment in kernels.cuh.
    key_min_sum[i]  = ~0ull;
    key_max_sum[i]  = 0ull;
    key_min_diff[i] = ~0ull;
    key_max_diff[i] = 0ull;
}
void launch_component_stats_init(int* d_count, unsigned long long* d_sum_x, unsigned long long* d_sum_y,
                                 int* d_min_x, int* d_max_x, int* d_min_y, int* d_max_y,
                                 unsigned long long* d_key_min_sum, unsigned long long* d_key_max_sum,
                                 unsigned long long* d_key_min_diff, unsigned long long* d_key_max_diff,
                                 int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    component_stats_init_kernel<<<grid, block>>>(d_count, d_sum_x, d_sum_y, d_min_x, d_max_x, d_min_y, d_max_y,
                                                  d_key_min_sum, d_key_max_sum, d_key_min_diff, d_key_max_diff,
                                                  W, H, N);
    CUDA_CHECK_LAST_ERROR("component_stats_init_kernel launch");
}

// ---------------------------------------------------------------------------
// component_stats_accumulate_kernel — every foreground pixel scatters into
// nine accumulators at index [label[p]]: a running count/sum (centroid), a
// running bbox (atomicMin/Max on int), and four PACKED-KEY extrema
// (atomicMin64/atomicMax64 on the (score,index) keys from kernels.cuh) that
// jointly recover the 4 "extreme corner" pixels — see the corner-refinement
// kernel below for what happens to them next.
// ---------------------------------------------------------------------------
__global__ void component_stats_accumulate_kernel(const unsigned char* __restrict__ mask,
                                                   const int* __restrict__ label,
                                                   int* __restrict__ count,
                                                   unsigned long long* __restrict__ sum_x,
                                                   unsigned long long* __restrict__ sum_y,
                                                   int* __restrict__ min_x, int* __restrict__ max_x,
                                                   int* __restrict__ min_y, int* __restrict__ max_y,
                                                   unsigned long long* __restrict__ key_min_sum,
                                                   unsigned long long* __restrict__ key_max_sum,
                                                   unsigned long long* __restrict__ key_min_diff,
                                                   unsigned long long* __restrict__ key_max_diff,
                                                   int W, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (!mask[i]) return;

    const int x = i % W, y = i / W;
    const int L = label[i];

    atomicAdd(&count[L], 1);
    atomicAdd(&sum_x[L], static_cast<unsigned long long>(x));
    atomicAdd(&sum_y[L], static_cast<unsigned long long>(y));
    atomicMin(&min_x[L], x); atomicMax(&max_x[L], x);
    atomicMin(&min_y[L], y); atomicMax(&max_y[L], y);

    const long long s = static_cast<long long>(x) + static_cast<long long>(y);   // "diagonal" score
    const long long d = static_cast<long long>(x) - static_cast<long long>(y);   // "anti-diagonal" score
    atomicMin64(&key_min_sum[L],  pack_corner_key(s, i));
    atomicMax64(&key_max_sum[L],  pack_corner_key(s, i));
    atomicMin64(&key_min_diff[L], pack_corner_key(d, i));
    atomicMax64(&key_max_diff[L], pack_corner_key(d, i));
}
void launch_component_stats_accumulate(const unsigned char* d_mask, const int* d_label,
                                       int* d_count, unsigned long long* d_sum_x, unsigned long long* d_sum_y,
                                       int* d_min_x, int* d_max_x, int* d_min_y, int* d_max_y,
                                       unsigned long long* d_key_min_sum, unsigned long long* d_key_max_sum,
                                       unsigned long long* d_key_min_diff, unsigned long long* d_key_max_diff,
                                       int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    component_stats_accumulate_kernel<<<grid, block>>>(d_mask, d_label, d_count, d_sum_x, d_sum_y,
                                                        d_min_x, d_max_x, d_min_y, d_max_y,
                                                        d_key_min_sum, d_key_max_sum, d_key_min_diff, d_key_max_diff,
                                                        W, H);
    CUDA_CHECK_LAST_ERROR("component_stats_accumulate_kernel launch");
}

// ===========================================================================
// Stage 3 — quad extraction: corner refinement. CANDIDATE-parallel: one
// thread owns one CandidateComponent, does ~4 short radial searches, and
// writes one QuadCorners. Contrast this launch geometry with every kernel
// above (THEORY.md "The GPU mapping" measures the actual grid/block/
// occupancy numbers for both families on this project's committed scene).
// ===========================================================================

// ---------------------------------------------------------------------------
// refine_one_corner — march from the component centroid TOWARD (and a bit
// PAST) the raw extreme pixel, sampling the real image, and return the
// sub-pixel distance along that ray where the image crosses from dark
// (foreground-like: gray < local_mean - bias) to light. The search takes the
// LAST such crossing within range (foreground components can have internal
// dark/light texture near their centroid — e.g. this project's payload
// cells — so the FIRST crossing walking outward is not reliable, but the
// LAST one before we are unambiguously in the background is: it is the
// physical edge of the component, by construction).
//
// HONESTY (README "Limitations & honesty", THEORY.md "The algorithm"): this
// is a 1-D search along a single, ALREADY-CHOSEN direction (centroid ->
// raw extreme pixel) — a real AprilTag detector fits a LINE to each of the
// four edges independently (from many gradient-clustered points) and
// intersects adjacent lines for the corner; that is robust to the extreme
// pixel itself being noisy. This project's search inherits whatever noise
// sits on that one ray. It is a deliberately smaller, tractable teaching
// version of the same idea (radial edge localization), not the production
// algorithm.
// ---------------------------------------------------------------------------
__device__ bool refine_one_corner(float cx, float cy, float rawx, float rawy,
                                  const unsigned char* gray, const float* local_mean,
                                  int W, int H, float& out_x, float& out_y)
{
    const float dx = rawx - cx, dy = rawy - cy;
    const float raw_dist = sqrtf(dx * dx + dy * dy);
    if (raw_dist < 1.0f) return false;              // degenerate: centroid == corner (near-empty component)
    const float ux = dx / raw_dist, uy = dy / raw_dist;

    // Search out to a SMALL FIXED margin past the raw extreme pixel — the
    // true geometric corner sits only slightly beyond the outermost
    // FOREGROUND pixel CCL found (a couple of pixels' worth of anti-
    // aliasing/rasterization slack from the synthetic scene's 5-tap blur,
    // never a large fraction of the whole centroid-to-corner distance).
    // An EARLIER version of this function used a MULTIPLICATIVE margin
    // (1.6x the distance) reasoning that "a little extra search range never
    // hurts" — it was wrong: for a corner 60px from the centroid that adds
    // 36px of extra search range, easily enough to wander past the tag
    // entirely into unrelated background/noise and lock onto a spurious
    // FAR-AWAY dark->light crossing (measured on this project's committed
    // scene: one tag's corner landed 31px from its true position before
    // this fix — see THEORY.md "Numerical considerations" for the full
    // story). A small ADDITIVE margin fixes it: it scales with the blur
    // radius, not with the tag's size.
    const float kCornerSearchMarginPx = 6.0f;
    const float max_t = raw_dist + kCornerSearchMarginPx;
    const int kSteps = 48;
    const float step = max_t / static_cast<float>(kSteps);

    float best_t = raw_dist;   // fallback: the raw pixel itself, if no crossing is found below
    float f_prev = 0.0f;       // "darkness" signal: (local_mean - bias) - gray; >0 means dark
    for (int k = 0; k <= kSteps; ++k) {
        const float t = step * static_cast<float>(k);
        const float px = cx + ux * t, py = cy + uy * t;
        const float g = bilerp_u8(gray, W, H, px, py);
        const float m = bilerp_f32(local_mean, W, H, px, py);
        const float f = (m - kThreshBiasC) - g;
        if (k > 0 && f_prev > 0.0f && f <= 0.0f) {
            // Linear interpolation for the zero-crossing between samples
            // k-1 (dark) and k (light) — a sub-pixel refinement of WHERE
            // along this 1-D ray the edge sits, cheap and exact for a
            // locally-linear brightness ramp (the box-filtered threshold
            // field is smooth by construction). We deliberately do NOT
            // break here: overwriting best_t keeps the LAST crossing seen
            // (see the function header note on why the last, not first,
            // dark->light transition is the physically correct one).
            const float t_prev = step * static_cast<float>(k - 1);
            const float frac = f_prev / (f_prev - f);   // in [0,1]
            best_t = t_prev + frac * step;
        }
        f_prev = f;
    }
    out_x = cx + ux * best_t;
    out_y = cy + uy * best_t;
    // Always succeeds once past the degenerate-centroid guard above: even
    // when no dark->light crossing is found (rare — e.g. a corner ray that
    // stays dark the whole search range), best_t's raw-pixel fallback is
    // still a usable, if less precise, corner estimate (README "Limitations
    // & honesty" names this fallback explicitly).
    return true;
}

__global__ void corner_refine_kernel(const CandidateComponent* __restrict__ cands, int n,
                                     const unsigned char* __restrict__ gray,
                                     const float* __restrict__ local_mean,
                                     int W, int H, QuadCorners* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const CandidateComponent& c = cands[i];
    QuadCorners q;
    q.valid = true;
    for (int k = 0; k < 4; ++k) {
        float rx, ry;
        const bool ok = refine_one_corner(c.centroid_x, c.centroid_y,
                                          c.raw_corner_x[k], c.raw_corner_y[k],
                                          gray, local_mean, W, H, rx, ry);
        q.x[k] = rx; q.y[k] = ry;
        if (!ok) q.valid = false;
    }
    out[i] = q;
}
void launch_corner_refine(const CandidateComponent* d_candidates, int n,
                          const unsigned char* d_gray, const float* d_local_mean,
                          int W, int H, QuadCorners* d_quads)
{
    if (n <= 0) return;
    const int block = 32, grid = (n + block - 1) / block;
    corner_refine_kernel<<<grid, block>>>(d_candidates, n, d_gray, d_local_mean, W, H, d_quads);
    CUDA_CHECK_LAST_ERROR("corner_refine_kernel launch");
}

// ===========================================================================
// Stage 4 — DLT homography, one 8x8 Gaussian-elimination solve per thread.
// ===========================================================================

// ---------------------------------------------------------------------------
// solve_8x8_partial_pivot — in-place Gaussian elimination with partial
// pivoting on an 8x9 augmented matrix A (8 unknowns, 1 RHS column appended
// at A[r][8]). DOUBLE precision throughout — see kernels.cuh's file header
// for why (small system, cheap even per-thread, and removes conditioning as
// a variable we have to fight with float32 — THEORY.md "Numerical
// considerations" quantifies the difference on this project's tag scales).
// Returns false if the largest available pivot in some column is below
// kPivotEps (a near-singular/degenerate quad — e.g. 3 nearly-collinear
// corners), in which case the candidate is dropped, never a silent
// wrong answer.
// ---------------------------------------------------------------------------
__device__ bool solve_8x8_partial_pivot(double A[8][9], double out_h[8])
{
    constexpr double kPivotEps = 1e-9;
    for (int col = 0; col < 8; ++col) {
        // Partial pivoting: swap in the row (at or below `col`) with the
        // largest |A[row][col]| — the standard numerical-stability measure
        // against dividing by a small number (THEORY.md derives the error
        // amplification a small pivot would cause).
        int piv = col;
        double piv_val = fabs(A[col][col]);
        for (int r = col + 1; r < 8; ++r) {
            if (fabs(A[r][col]) > piv_val) { piv_val = fabs(A[r][col]); piv = r; }
        }
        if (piv_val < kPivotEps) return false;   // degenerate system — caller marks invalid
        if (piv != col) {
            for (int k = 0; k < 9; ++k) { const double tmp = A[col][k]; A[col][k] = A[piv][k]; A[piv][k] = tmp; }
        }
        // Eliminate this column from every row below.
        for (int r = col + 1; r < 8; ++r) {
            const double f = A[r][col] / A[col][col];
            if (f == 0.0) continue;
            for (int k = col; k < 9; ++k) A[r][k] -= f * A[col][k];
        }
    }
    // Back-substitution.
    for (int r = 7; r >= 0; --r) {
        double s = A[r][8];
        for (int k = r + 1; k < 8; ++k) s -= A[r][k] * out_h[k];
        out_h[r] = s / A[r][r];
    }
    return true;
}

__global__ void homography_solve_kernel(const QuadCorners* __restrict__ quads, int n,
                                        Homography* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const QuadCorners& q = quads[i];
    Homography H;
    H.valid = false;
    if (!q.valid) { out[i] = H; return; }

    // Fixed tag-model corners, meters, SAME order as QuadCorners: TL, TR,
    // BR, BL (kernels.cuh's file header names this convention).
    const double half = static_cast<double>(kTagHalfM);
    const double MX[4] = { -half,  half, half, -half };
    const double MY[4] = { -half, -half, half,  half };

    // Build the 8x9 augmented system (kernels.cuh's file header derives the
    // two equations per correspondence; h33 fixed to 1).
    double A[8][9];
    for (int k = 0; k < 4; ++k) {
        const double X = MX[k], Y = MY[k];
        const double x = static_cast<double>(q.x[k]), y = static_cast<double>(q.y[k]);
        double* rowA = A[2 * k];
        rowA[0] = X; rowA[1] = Y; rowA[2] = 1.0; rowA[3] = 0.0; rowA[4] = 0.0; rowA[5] = 0.0;
        rowA[6] = -x * X; rowA[7] = -x * Y; rowA[8] = x;
        double* rowB = A[2 * k + 1];
        rowB[0] = 0.0; rowB[1] = 0.0; rowB[2] = 0.0; rowB[3] = X; rowB[4] = Y; rowB[5] = 1.0;
        rowB[6] = -y * X; rowB[7] = -y * Y; rowB[8] = y;
    }

    double h8[8];
    if (solve_8x8_partial_pivot(A, h8)) {
        for (int k = 0; k < 8; ++k) H.h[k] = h8[k];
        H.h[8] = 1.0;
        H.valid = true;
    }
    out[i] = H;
}
void launch_homography_solve(const QuadCorners* d_quads, int n, Homography* d_homographies)
{
    if (n <= 0) return;
    const int block = 32, grid = (n + block - 1) / block;
    homography_solve_kernel<<<grid, block>>>(d_quads, n, d_homographies);
    CUDA_CHECK_LAST_ERROR("homography_solve_kernel launch");
}

// ===========================================================================
// Stage 5 — perspective grid sampling + dictionary decode. CANDIDATE-
// parallel: one thread warp-samples all 36 cell centers of its own quad
// through H, thresholds each against the SAME local_mean field stage 1 built
// (single source of truth for "what is black" — kernels.cuh's file header),
// then tries the sampled 4x4 payload against the dictionary at all 4
// rotations. Parameter naming note: the image height is `imgH` (not `H`)
// specifically to avoid shadowing `Homography H` below — a real, if minor,
// naming trap worth flagging explicitly (CLAUDE.md paragraph 6: name
// non-obvious choices).
// ---------------------------------------------------------------------------
__global__ void grid_decode_kernel(const Homography* __restrict__ homs, int n,
                                   const unsigned char* __restrict__ gray,
                                   const float* __restrict__ local_mean, int W, int imgH,
                                   const uint16_t* __restrict__ dictionary, int num_dict_codes,
                                   int correction_capacity, Detection* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Start every field at an honest "rejected / not computed" default —
    // every early-return path below leaves a fully-defined, clearly-rejected
    // record rather than uninitialized memory.
    Detection d;
    d.candidate_index = i;
    d.border_ok = false;
    d.accepted = false;
    d.tag_id = -1;
    d.rotation = 0;
    d.hamming_distance = 999;
    d.pose_valid = false;
    for (int k = 0; k < 4; ++k) { d.corners_x[k] = 0.0f; d.corners_y[k] = 0.0f; }
    for (int k = 0; k < 9; ++k) d.R[k] = 0.0f;
    d.t[0] = d.t[1] = d.t[2] = 0.0f;

    const Homography H = homs[i];
    if (!H.valid) { out[i] = d; return; }

    const double cell = static_cast<double>(kTagSizeM) / static_cast<double>(kGridN);
    const double half = static_cast<double>(kTagHalfM);

    int border_errors = 0;
    uint16_t payload = 0;
    for (int r = 0; r < kGridN; ++r) {
        for (int c = 0; c < kGridN; ++c) {
            // Tag-frame cell CENTER, meters -> homogeneous image pixel via H
            // (kernels.cuh's file header: pixel_homog = H * (X,Y,1)).
            const double X = -half + (static_cast<double>(c) + 0.5) * cell;
            const double Y = -half + (static_cast<double>(r) + 0.5) * cell;
            const double w  = H.h[6] * X + H.h[7] * Y + H.h[8];
            const double px = (H.h[0] * X + H.h[1] * Y + H.h[2]) / w;
            const double py = (H.h[3] * X + H.h[4] * Y + H.h[5]) / w;

            const float g = bilerp_u8(gray, W, imgH, static_cast<float>(px), static_cast<float>(py));
            const float m = bilerp_f32(local_mean, W, imgH, static_cast<float>(px), static_cast<float>(py));
            const bool dark = g < (m - kThreshBiasC);

            if (is_border_cell(r, c)) {
                if (!dark) ++border_errors;   // tolerated up to kMaxBorderErrors -- see its doc comment
            } else if (dark) {
                payload = static_cast<uint16_t>(payload | (1u << payload_bit_index(r, c)));
            }
        }
    }
    d.border_ok = (border_errors <= kMaxBorderErrors);

    // Record the image-pixel location of the 4 grid CORNERS too (the tag
    // model's own corners, not just the refined quad — a small, honest
    // reprojection check a learner can compare against corners_x/y from
    // stage 3, see main.cu's corner-accuracy gate).
    const double MX[4] = { -half,  half, half, -half };
    const double MY[4] = { -half, -half, half,  half };
    for (int k = 0; k < 4; ++k) {
        const double w  = H.h[6] * MX[k] + H.h[7] * MY[k] + H.h[8];
        d.corners_x[k] = static_cast<float>((H.h[0] * MX[k] + H.h[1] * MY[k] + H.h[2]) / w);
        d.corners_y[k] = static_cast<float>((H.h[3] * MX[k] + H.h[4] * MY[k] + H.h[5]) / w);
    }

    if (!d.border_ok) { out[i] = d; return; }   // too many wrong border cells — see kMaxBorderErrors' doc comment

    // Degenerate-payload safeguard (all-black / all-white): a real dictionary
    // never assigns these, and a filled blob (a false-positive candidate)
    // reads exactly one of them — reject before even touching the
    // dictionary (kernels.cuh's popcount16 doc comment).
    const int ones = popcount16(payload);
    if (ones == 0 || ones == kPayloadBits) { out[i] = d; return; }

    // Try the sampled payload against the dictionary at all 4 in-plane
    // rotations; keep the globally closest (code, rotation) pair.
    uint16_t rot[4];
    rot[0] = payload;
    rot[1] = rotate_payload_90(rot[0]);
    rot[2] = rotate_payload_90(rot[1]);
    rot[3] = rotate_payload_90(rot[2]);

    int best_dist = 999, best_code = -1, best_rot = 0;
    for (int j = 0; j < num_dict_codes; ++j) {
        for (int rIdx = 0; rIdx < 4; ++rIdx) {
            const int hd = popcount16(static_cast<uint16_t>(rot[rIdx] ^ dictionary[j]));
            if (hd < best_dist) { best_dist = hd; best_code = j; best_rot = rIdx; }
        }
    }
    d.hamming_distance = best_dist;
    d.tag_id = best_code;
    d.rotation = best_rot;
    d.accepted = (best_dist <= correction_capacity);
    if (!d.accepted) { d.tag_id = -1; }   // rejected candidates report no ID (only the near-miss distance)

    out[i] = d;
}
void launch_grid_decode(const Homography* d_homographies, int n,
                        const unsigned char* d_gray, const float* d_local_mean, int W, int H,
                        const uint16_t* d_dictionary, int num_dict_codes, int correction_capacity,
                        Detection* d_detections)
{
    if (n <= 0) return;
    const int block = 32, grid = (n + block - 1) / block;
    grid_decode_kernel<<<grid, block>>>(d_homographies, n, d_gray, d_local_mean, W, H,
                                        d_dictionary, num_dict_codes, correction_capacity, d_detections);
    CUDA_CHECK_LAST_ERROR("grid_decode_kernel launch");
}

// ===========================================================================
// Stage 6 — pose from homography: classical K^-1*H column-normalization
// decomposition (THEORY.md "The math" derives every line below). Production
// systems refine this initial estimate with IPPE (README "Prior art").
// ===========================================================================
__global__ void pose_from_homography_kernel(const Homography* __restrict__ homs, int n,
                                            Detection* __restrict__ dets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Homography H = homs[i];
    Detection d = dets[i];
    d.pose_valid = false;
    if (!H.valid) { dets[i] = d; return; }

    // M = K^-1 * H, computed directly from K's closed-form inverse (K is
    // upper-triangular with unit (3,3) entry, so K^-1 is too — no general
    // 3x3 inverse needed here, unlike the DLT solve above).
    const double fx = static_cast<double>(kFx), fy = static_cast<double>(kFy);
    const double cx = static_cast<double>(kCx), cy = static_cast<double>(kCy);
    double M[3][3];
    M[0][0] = (H.h[0] - cx * H.h[6]) / fx;  M[0][1] = (H.h[1] - cx * H.h[7]) / fx;  M[0][2] = (H.h[2] - cx * H.h[8]) / fx;
    M[1][0] = (H.h[3] - cy * H.h[6]) / fy;  M[1][1] = (H.h[4] - cy * H.h[7]) / fy;  M[1][2] = (H.h[5] - cy * H.h[8]) / fy;
    M[2][0] = H.h[6];                       M[2][1] = H.h[7];                       M[2][2] = H.h[8];

    // Columns m1, m2, m3 of M: m1 ~ scale*r1, m2 ~ scale*r2, m3 ~ scale*t.
    double m1[3] = { M[0][0], M[1][0], M[2][0] };
    double m2[3] = { M[0][1], M[1][1], M[2][1] };
    double m3[3] = { M[0][2], M[1][2], M[2][2] };
    const double n1 = sqrt(m1[0]*m1[0] + m1[1]*m1[1] + m1[2]*m1[2]);
    const double n2 = sqrt(m2[0]*m2[0] + m2[1]*m2[1] + m2[2]*m2[2]);
    if (n1 < 1e-9 || n2 < 1e-9) { dets[i] = d; return; }   // degenerate homography column

    // Average the two column norms to recover ONE scale (in a perfect
    // homography they would already be equal; averaging is the standard,
    // numerically gentle way to split the difference — THEORY.md
    // "Numerical considerations" quantifies the residual asymmetry this
    // project actually measures).
    double scale = 2.0 / (n1 + n2);

    double r1[3] = { m1[0]*scale, m1[1]*scale, m1[2]*scale };
    double r2[3] = { m2[0]*scale, m2[1]*scale, m2[2]*scale };
    double t[3]  = { m3[0]*scale, m3[1]*scale, m3[2]*scale };

    // The tag must be IN FRONT of the camera (t_z > 0, optical convention,
    // z-forward): the homography's sign is ambiguous (H and -H describe the
    // same pixel mapping since it is used homogeneously), so flip the whole
    // triad if the naive decomposition put the tag behind the camera.
    if (t[2] < 0.0) {
        scale = -scale;
        for (int k = 0; k < 3; ++k) { r1[k] = m1[k]*scale; r2[k] = m2[k]*scale; t[k] = m3[k]*scale; }
    }

    // r1, r2 are each individually unit-norm by construction but are not
    // guaranteed exactly orthogonal (homography noise) — one Gram-Schmidt
    // step nudges r2 to be orthogonal to r1 before completing the frame with
    // the cross product, giving an (approximately) proper rotation matrix.
    // This is the cheap approximation THEORY.md contrasts with full SVD-
    // based orthogonalization and with IPPE.
    const double dot12 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];
    double r2o[3] = { r2[0] - dot12*r1[0], r2[1] - dot12*r1[1], r2[2] - dot12*r1[2] };
    const double n2o = sqrt(r2o[0]*r2o[0] + r2o[1]*r2o[1] + r2o[2]*r2o[2]);
    if (n2o < 1e-9) { dets[i] = d; return; }
    r2o[0] /= n2o; r2o[1] /= n2o; r2o[2] /= n2o;

    const double r3[3] = {
        r1[1]*r2o[2] - r1[2]*r2o[1],
        r1[2]*r2o[0] - r1[0]*r2o[2],
        r1[0]*r2o[1] - r1[1]*r2o[0]
    };

    // R's COLUMNS are r1, r2o, r3 (tag axes expressed in the camera frame) —
    // stored ROW-MAJOR per kernels.cuh's Detection::R documentation.
    d.R[0] = static_cast<float>(r1[0]); d.R[1] = static_cast<float>(r2o[0]); d.R[2] = static_cast<float>(r3[0]);
    d.R[3] = static_cast<float>(r1[1]); d.R[4] = static_cast<float>(r2o[1]); d.R[5] = static_cast<float>(r3[1]);
    d.R[6] = static_cast<float>(r1[2]); d.R[7] = static_cast<float>(r2o[2]); d.R[8] = static_cast<float>(r3[2]);
    d.t[0] = static_cast<float>(t[0]); d.t[1] = static_cast<float>(t[1]); d.t[2] = static_cast<float>(t[2]);
    d.pose_valid = true;
    dets[i] = d;
}
void launch_pose_from_homography(const Homography* d_homographies, int n, Detection* d_detections)
{
    if (n <= 0) return;
    const int block = 32, grid = (n + block - 1) / block;
    pose_from_homography_kernel<<<grid, block>>>(d_homographies, n, d_detections);
    CUDA_CHECK_LAST_ERROR("pose_from_homography_kernel launch");
}
