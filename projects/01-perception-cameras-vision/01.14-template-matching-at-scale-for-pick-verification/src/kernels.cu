// ===========================================================================
// kernels.cu — GPU kernels for project 01.14
//              (Template matching (NCC) at scale for pick verification)
//
// Role in the project
// -------------------
// Five kernels, in the order main.cu calls them:
//   1-2. integral_row_scan_kernel / integral_col_scan_kernel — build the
//        whole-tray integral images (SUM and SUM-OF-SQUARES) ONCE, a 2-pass
//        separable prefix scan (the acceleration structure every later NCC
//        evaluation reads from in O(1)).
//   3.   window_stats_kernel — the dedicated bit-exact-integer twin: window
//        sum/sum-of-squares per (slot, offset), read from the integral
//        images via O(1) box queries.
//   4-6. ncc_naive_kernel / ncc_sumtable_kernel / ncc_shared_kernel — the
//        SAME 104,040-evaluation NCC score volume, computed three ways, to
//        teach the "cache the redundant work" acceleration ladder:
//          naive    — every thread re-derives its own window statistics by
//                     directly re-scanning TEMPLATE_SIZE^2 pixels;
//          sumtable — window statistics come from the O(1) integral-image
//                     box query instead;
//          shared   — sumtable's O(1) statistics, PLUS the numerator's
//                     O(T^2) correlation loop reads its window/template
//                     pixels from SHARED memory instead of global memory.
//
// THE THREE-AXIS PARALLEL MAPPING (used by window_stats_kernel and all three
// NCC kernels): grid.x = slot, grid.y = template (NCC kernels only —
// window_stats has no template axis), block = (offset_x, offset_y). Every
// axis is embarrassingly parallel — no evaluation depends on any other — so
// this is a pure 3-D "one thread per independent problem instance" map,
// the same GPU-mapping idiom as 08.01's one-thread-per-rollout, extended to
// three independent axes instead of one.
//
// Read this after: kernels.cuh (the contracts). Read this before:
// reference_cpu.cpp (the independent CPU twins) and main.cu (orchestration).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>

// ---------------------------------------------------------------------------
// __constant__ memory — the template statistics table (kernels.cuh SECTION 5).
// EVERY thread scoring template t reads the SAME two numbers (S_t[t], S_tt[t])
// — a broadcast access pattern, exactly the reasoning project 01.13 gives for
// putting its Hough theta table in constant memory (kernels.cu file header
// there): a per-SM cache tuned for "every thread reads the same address on
// the same cycle" beats 289 separate global loads per warp. 15 templates *
// 2 int64 = 240 bytes — negligible against the 64 KiB constant window.
// ---------------------------------------------------------------------------
__constant__ int64_t g_S_t[NUM_TEMPLATES];
__constant__ int64_t g_S_tt[NUM_TEMPLATES];

void upload_template_stats(const int64_t* S_t, const int64_t* S_tt)
{
    CUDA_CHECK(cudaMemcpyToSymbol(g_S_t, S_t, sizeof(int64_t) * NUM_TEMPLATES));
    CUDA_CHECK(cudaMemcpyToSymbol(g_S_tt, S_tt, sizeof(int64_t) * NUM_TEMPLATES));
}

// ---------------------------------------------------------------------------
// ncc_from_sums — combine the 5 raw integer sums into one NCC score
// (kernels.cuh SECTION 6 documents and derives this algebra). A single
// __device__ helper reused by all three NCC kernels below: sharing this
// FINAL COMBINING STEP across kernels.cu's own three kernels is ordinary
// code reuse, not a violation of the GPU-vs-CPU twin-independence ruling
// (that ruling governs the kernels.cu/reference_cpu.cpp boundary — see
// kernels.cuh's file header — not reuse within one side of it).
//
// N: TEMPLATE_PIXELS. S_w/S_ww/S_wt vary per (slot,template,offset); S_t/S_tt
// are the constant-memory template stats. Promoting to double BEFORE the
// var_w*var_t product (rather than multiplying the two int64 variances
// directly) sidesteps a genuine, if exotic, int64 overflow: THEORY.md
// "Numerical considerations" works the worst-case bound (~2.9e19 for a
// perfectly bimodal 0/255 template+window) which exceeds even uint64_t's
// range, though no image this project renders comes remotely close.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ncc_from_sums(int64_t S_w, int64_t S_ww, int64_t S_wt,
                                               int64_t S_t, int64_t S_tt)
{
    const int64_t N = TEMPLATE_PIXELS;
    const int64_t numerator_unnorm = N * S_wt - S_w * S_t;
    const int64_t var_w_unnorm = N * S_ww - S_w * S_w;
    const int64_t var_t_unnorm = N * S_tt - S_t * S_t;

    const double denom = sqrt(static_cast<double>(var_w_unnorm) * static_cast<double>(var_t_unnorm));
    if (denom < static_cast<double>(NCC_DENOM_EPS)) return 0.0f;   // flat window or flat template — no signal
    return static_cast<float>(static_cast<double>(numerator_unnorm) / denom);
}

// ===========================================================================
// STAGE 1 — whole-tray integral images: a 2-pass separable prefix scan.
//
// Pass 1 (row scan): thread y owns row y, running a SEQUENTIAL accumulation
// across that row's W pixels — S(x,y) = sum_{x'<=x} img(x',y). H independent
// rows in parallel, exactly the same "parallel ACROSS independent lines,
// sequential WITHIN each line" idea as project 01.13's separable Gaussian
// blur (there: a 5-tap weighted stencil per line; here: a running sum).
//
// Pass 2 (col scan): thread c owns padded column c, running a sequential
// accumulation DOWN that column of pass 1's row-prefix table —
// II(x,y) = sum_{y'<=y} S(x,y'). Composing the two passes gives the true 2-D
// integral image (induction proof in THEORY.md "The GPU mapping"):
//     II(x,y) = sum_{y'<=y} S(x,y') = sum_{y'<=y} sum_{x'<=x} img(x',y')
//             = sum_{x'<=x, y'<=y} img(x',y')                       QED
//
// Both passes write IN PLACE (pass 2 reads pass 1's output and overwrites
// it) — safe because pass 2's threads each own a DISJOINT column, so no two
// threads ever touch the same memory. Padding (row 0 / col 0 all zero) is
// established by a cudaMemset before either kernel runs (launcher below);
// neither kernel ever writes there, so it stays zero throughout.
// ===========================================================================
__global__ void integral_row_scan_kernel(const uint8_t* __restrict__ img,
                                         uint32_t* __restrict__ ii_sum,
                                         uint64_t* __restrict__ ii_sumsq)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's row
    if (y >= IMG_H) return;

    // Running accumulators live in registers for the whole row — a classic
    // sequential prefix-sum loop, W=324 iterations, entirely memory-coalesced
    // within the row (consecutive x -> consecutive addresses).
    uint32_t running_sum = 0;
    uint64_t running_sumsq = 0;
    for (int x = 0; x < IMG_W; ++x) {
        const uint32_t v = img[y * IMG_W + x];
        running_sum += v;
        running_sumsq += static_cast<uint64_t>(v) * v;
        // Padded position: row y+1, col x+1 (row 0 / col 0 stay the zero
        // border established by the launcher's cudaMemset).
        ii_sum[ii_index(y + 1, x + 1)] = running_sum;
        ii_sumsq[ii_index(y + 1, x + 1)] = running_sumsq;
    }
}

__global__ void integral_col_scan_kernel(uint32_t* __restrict__ ii_sum,
                                         uint64_t* __restrict__ ii_sumsq)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x + 1;   // padded column, c in [1, IMG_W]
    if (c > IMG_W) return;

    uint32_t running_sum = 0;
    uint64_t running_sumsq = 0;
    for (int r = 1; r <= IMG_H; ++r) {
        // Read pass 1's row-prefix value sitting here, ADD it into the
        // running column total, and overwrite in place — see the file
        // header's induction proof for why this equals the true II(x,y).
        running_sum += ii_sum[ii_index(r, c)];
        running_sumsq += ii_sumsq[ii_index(r, c)];
        ii_sum[ii_index(r, c)] = running_sum;
        ii_sumsq[ii_index(r, c)] = running_sumsq;
    }
}

void launch_build_integral_images(const uint8_t* d_img, uint32_t* d_ii_sum, uint64_t* d_ii_sumsq)
{
    // Zero the WHOLE padded table first — this is what makes row 0 / col 0
    // the integral image's zero border without either kernel special-casing
    // it (kernels.cuh SECTION 4's padding convention).
    CUDA_CHECK(cudaMemset(d_ii_sum, 0, sizeof(uint32_t) * static_cast<size_t>(II_CELLS)));
    CUDA_CHECK(cudaMemset(d_ii_sumsq, 0, sizeof(uint64_t) * static_cast<size_t>(II_CELLS)));

    const int block = 64;   // H=220, W=324: small counts, a modest block keeps every SM fed
    integral_row_scan_kernel<<<(IMG_H + block - 1) / block, block>>>(d_img, d_ii_sum, d_ii_sumsq);
    CUDA_CHECK_LAST_ERROR("integral_row_scan_kernel launch");
    // Pass 2 depends on EVERY row of pass 1 being finished — kernels launched
    // into the same (default) stream execute in program order on the GPU, so
    // no explicit sync is needed between them; only the FINAL result
    // (read back on the host) needs a sync, which cudaMemcpy provides.
    integral_col_scan_kernel<<<(IMG_W + block - 1) / block, block>>>(d_ii_sum, d_ii_sumsq);
    CUDA_CHECK_LAST_ERROR("integral_col_scan_kernel launch");
}

// ===========================================================================
// STAGE 2 — window statistics via O(1) box query (the dedicated bit-exact
// integer twin main.cu's VERIFY stage compares against reference_cpu.cpp).
//
// Box-query algebra (standard integral-image inclusion-exclusion, derived in
// THEORY.md): the sum over [x0,x1) x [y0,y1) is
//     II(x1,y1) - II(x0,y1) - II(x1,y0) + II(x0,y0)
// with the PADDED table's convention meaning x0/y0/x1/y1 index directly —
// no +-1 juggling at the call site (kernels.cuh SECTION 4).
//
// Grid: one BLOCK per slot (24 blocks), one THREAD per offset (17x17=289) —
// no template axis here, because S_w/S_ww do not depend on which template is
// being scored (kernels.cuh SECTION 4's whole point).
// ===========================================================================
__global__ void window_stats_kernel(const uint32_t* __restrict__ ii_sum,
                                    const uint64_t* __restrict__ ii_sumsq,
                                    uint32_t* __restrict__ ws_sum,
                                    uint64_t* __restrict__ ws_sumsq)
{
    const int slot = blockIdx.x;
    const int ox = threadIdx.x, oy = threadIdx.y;                       // offset indices, [0, NUM_OFFSETS_1D)
    const int dx = ox - SEARCH_RADIUS, dy = oy - SEARCH_RADIUS;         // signed pixel offset, [-8, +8]

    const int x0 = slot_window_x0(slot) + SEARCH_RADIUS + dx;           // this candidate patch's top-left,
    const int y0 = slot_window_y0(slot) + SEARCH_RADIUS + dy;           // in ABSOLUTE tray-image pixels
    const int x1 = x0 + TEMPLATE_SIZE, y1 = y0 + TEMPLATE_SIZE;

    const uint32_t S_w = ii_sum[ii_index(y1, x1)] - ii_sum[ii_index(y0, x1)]
                        - ii_sum[ii_index(y1, x0)] + ii_sum[ii_index(y0, x0)];
    const uint64_t S_ww = ii_sumsq[ii_index(y1, x1)] - ii_sumsq[ii_index(y0, x1)]
                         - ii_sumsq[ii_index(y1, x0)] + ii_sumsq[ii_index(y0, x0)];

    const long long idx = window_stats_index(slot, oy, ox);
    ws_sum[idx] = S_w;
    ws_sumsq[idx] = S_ww;
}

void launch_window_stats(const uint32_t* d_ii_sum, const uint64_t* d_ii_sumsq,
                         uint32_t* d_ws_sum, uint64_t* d_ws_sumsq)
{
    const dim3 block(NUM_OFFSETS_1D, NUM_OFFSETS_1D);   // 17x17 = 289 threads: one per offset
    const dim3 grid(NUM_SLOTS);                          // 24 blocks: one per slot
    window_stats_kernel<<<grid, block>>>(d_ii_sum, d_ii_sumsq, d_ws_sum, d_ws_sumsq);
    CUDA_CHECK_LAST_ERROR("window_stats_kernel launch");
}

// ===========================================================================
// STAGE 3a — NAIVE NCC: every one of the 104,040 threads independently
// re-derives ITS OWN window statistics by re-scanning the TEMPLATE_SIZE^2
// window directly from global memory, THEN does the O(T^2) correlation loop.
//
// The redundancy this measures: S_w/S_ww do NOT depend on which template is
// being scored, yet this kernel recomputes them once per (slot,template,
// offset) triple — NUM_TEMPLATES=15 times more often than necessary for a
// fixed (slot,offset). That 15x redundant-work factor (not the O(T^2) vs
// O(1) complexity gap alone) is why the sum-table version below is
// dramatically faster, not just modestly faster — MEASURED in main.cu's
// [time] lines and discussed in THEORY.md "The GPU mapping".
// ===========================================================================
__global__ void ncc_naive_kernel(const uint8_t* __restrict__ img,
                                 const uint8_t* __restrict__ templates,
                                 float* __restrict__ scores)
{
    const int slot = blockIdx.x;
    const int tmpl = blockIdx.y;
    const int ox = threadIdx.x, oy = threadIdx.y;
    const int dx = ox - SEARCH_RADIUS, dy = oy - SEARCH_RADIUS;

    const int x0 = slot_window_x0(slot) + SEARCH_RADIUS + dx;
    const int y0 = slot_window_y0(slot) + SEARCH_RADIUS + dy;
    const uint8_t* tpl = templates + static_cast<size_t>(tmpl) * TEMPLATE_PIXELS;

    // Pass 1 of this thread's OWN private work: re-scan the window to get
    // S_w, S_ww — the O(T^2) cost the sum-table variant replaces with an
    // O(1) lookup (kernels.cuh SECTION 4/6).
    int64_t S_w = 0, S_ww = 0;
    for (int ty = 0; ty < TEMPLATE_SIZE; ++ty) {
        const uint8_t* row = img + static_cast<size_t>(y0 + ty) * IMG_W + x0;
        for (int tx = 0; tx < TEMPLATE_SIZE; ++tx) {
            const int64_t w = row[tx];
            S_w += w;
            S_ww += w * w;
        }
    }
    // Pass 2: the direct correlation sum S_wt — unavoidable in every variant
    // (kernels.cuh SECTION 6); re-reads the SAME window pixels a second time
    // here (naive pays for both passes from global memory, every thread).
    int64_t S_wt = 0;
    for (int ty = 0; ty < TEMPLATE_SIZE; ++ty) {
        const uint8_t* row = img + static_cast<size_t>(y0 + ty) * IMG_W + x0;
        const uint8_t* trow = tpl + ty * TEMPLATE_SIZE;
        for (int tx = 0; tx < TEMPLATE_SIZE; ++tx)
            S_wt += static_cast<int64_t>(row[tx]) * trow[tx];
    }

    scores[score_index(slot, tmpl, oy, ox)] = ncc_from_sums(S_w, S_ww, S_wt, g_S_t[tmpl], g_S_tt[tmpl]);
}

void launch_ncc_naive(const uint8_t* d_img, const uint8_t* d_templates, float* d_scores)
{
    const dim3 block(NUM_OFFSETS_1D, NUM_OFFSETS_1D);          // 289 threads: one per offset
    const dim3 grid(NUM_SLOTS, NUM_TEMPLATES);                  // 24 x 15 = 360 blocks
    ncc_naive_kernel<<<grid, block>>>(d_img, d_templates, d_scores);
    CUDA_CHECK_LAST_ERROR("ncc_naive_kernel launch");
}

// ===========================================================================
// STAGE 3b — SUM-TABLE NCC: S_w/S_ww come from an O(1) box query into the
// integral images built once by Stage 1 (shared across ALL 104,040
// evaluations) instead of each thread re-scanning its own window. The
// numerator S_wt is UNCHANGED from the naive kernel — still a direct O(T^2)
// correlation loop reading global memory (no shortcut exists for it without
// an FFT-domain reformulation — THEORY.md "Where this sits in the real
// world" cites project 03.01's cuFFT precedent and the crossover argument).
// ===========================================================================
__global__ void ncc_sumtable_kernel(const uint8_t* __restrict__ img,
                                    const uint32_t* __restrict__ ii_sum,
                                    const uint64_t* __restrict__ ii_sumsq,
                                    const uint8_t* __restrict__ templates,
                                    float* __restrict__ scores)
{
    const int slot = blockIdx.x;
    const int tmpl = blockIdx.y;
    const int ox = threadIdx.x, oy = threadIdx.y;
    const int dx = ox - SEARCH_RADIUS, dy = oy - SEARCH_RADIUS;

    const int x0 = slot_window_x0(slot) + SEARCH_RADIUS + dx;
    const int y0 = slot_window_y0(slot) + SEARCH_RADIUS + dy;
    const int x1 = x0 + TEMPLATE_SIZE, y1 = y0 + TEMPLATE_SIZE;

    // O(1): 4 global reads + 3 adds, REGARDLESS of TEMPLATE_SIZE — replaces
    // the naive kernel's 576-pixel re-scan above.
    const int64_t S_w = ii_sum[ii_index(y1, x1)] - ii_sum[ii_index(y0, x1)]
                       - ii_sum[ii_index(y1, x0)] + ii_sum[ii_index(y0, x0)];
    const int64_t S_ww = static_cast<int64_t>(ii_sumsq[ii_index(y1, x1)] - ii_sumsq[ii_index(y0, x1)]
                        - ii_sumsq[ii_index(y1, x0)] + ii_sumsq[ii_index(y0, x0)]);

    const uint8_t* tpl = templates + static_cast<size_t>(tmpl) * TEMPLATE_PIXELS;
    int64_t S_wt = 0;
    for (int ty = 0; ty < TEMPLATE_SIZE; ++ty) {
        const uint8_t* row = img + static_cast<size_t>(y0 + ty) * IMG_W + x0;
        const uint8_t* trow = tpl + ty * TEMPLATE_SIZE;
        for (int tx = 0; tx < TEMPLATE_SIZE; ++tx)
            S_wt += static_cast<int64_t>(row[tx]) * trow[tx];
    }

    scores[score_index(slot, tmpl, oy, ox)] = ncc_from_sums(S_w, S_ww, S_wt, g_S_t[tmpl], g_S_tt[tmpl]);
}

void launch_ncc_sumtable(const uint8_t* d_img, const uint32_t* d_ii_sum, const uint64_t* d_ii_sumsq,
                         const uint8_t* d_templates, float* d_scores)
{
    const dim3 block(NUM_OFFSETS_1D, NUM_OFFSETS_1D);
    const dim3 grid(NUM_SLOTS, NUM_TEMPLATES);
    ncc_sumtable_kernel<<<grid, block>>>(d_img, d_ii_sum, d_ii_sumsq, d_templates, d_scores);
    CUDA_CHECK_LAST_ERROR("ncc_sumtable_kernel launch");
}

// ===========================================================================
// STAGE 3c — SHARED-MEMORY NCC: sum-table's O(1) box query for S_w/S_ww,
// PLUS the block (one (slot,template) pair, 289 offset-threads) cooperatively
// stages its ENTIRE search window and its template into shared memory ONCE,
// so the O(T^2) numerator loop below reads on-chip memory instead of
// re-fetching mostly-overlapping global-memory bytes 289 separate times.
//
// Reuse argument (THEORY.md "The GPU mapping" quantifies it): adjacent
// offset-threads' TEMPLATE_SIZE x TEMPLATE_SIZE patches overlap by
// TEMPLATE_SIZE-1 columns/rows — the UNION of every patch any thread in this
// block touches is exactly the slot's WINDOW x WINDOW region (kernels.cuh
// SECTION 1 sizes WINDOW to guarantee this). Caching that union once (1,600
// bytes) plus the template (576 bytes) costs 2,176 bytes of the 48+ KiB a
// Turing/Ampere/Ada SM offers per block — trivial — and turns up to 289x
// redundant global reads of the same bytes into one read each.
//
// Memory spaces: shared (s_window, s_template — read many times per thread,
// on-chip, ~100x lower latency than global); global (ii_sum/ii_sumsq for the
// O(1) box query — unchanged from the sum-table kernel, already cheap
// enough that shared-caching it would not measurably help); registers (the
// per-thread S_w/S_ww/S_wt accumulators).
// ===========================================================================
__global__ void ncc_shared_kernel(const uint8_t* __restrict__ img,
                                  const uint32_t* __restrict__ ii_sum,
                                  const uint64_t* __restrict__ ii_sumsq,
                                  const uint8_t* __restrict__ templates,
                                  float* __restrict__ scores)
{
    __shared__ uint8_t s_window[WINDOW * WINDOW];        // 1,600 bytes: this slot's whole search region
    __shared__ uint8_t s_template[TEMPLATE_PIXELS];       // 576 bytes: this block's one template

    const int slot = blockIdx.x;
    const int tmpl = blockIdx.y;
    const int ox = threadIdx.x, oy = threadIdx.y;
    const int tid = oy * blockDim.x + ox;                 // linear thread id within the block, [0, 289)
    const int block_threads = blockDim.x * blockDim.y;    // 289

    // ---- Cooperative load: every thread in the block pulls a STRIDED slice
    // of the window/template into shared memory (a grid-stride-loop pattern
    // applied within a block instead of across a grid) so the load itself is
    // fully parallel across the 289 threads, not serialized onto one. -------
    const int base_x = slot_window_x0(slot), base_y = slot_window_y0(slot);
    for (int i = tid; i < WINDOW * WINDOW; i += block_threads) {
        const int wy = i / WINDOW, wx = i % WINDOW;
        s_window[i] = img[static_cast<size_t>(base_y + wy) * IMG_W + (base_x + wx)];
    }
    const uint8_t* tpl_g = templates + static_cast<size_t>(tmpl) * TEMPLATE_PIXELS;
    for (int i = tid; i < TEMPLATE_PIXELS; i += block_threads)
        s_template[i] = tpl_g[i];

    // Every thread in the block must see the FULLY-loaded shared arrays
    // before reading them below — the mandatory barrier after a cooperative
    // shared-memory load (omitting it is a classic race: some threads would
    // start their correlation loop while others are still writing s_window).
    __syncthreads();

    const int dx = ox - SEARCH_RADIUS, dy = oy - SEARCH_RADIUS;
    const int x0 = base_x + SEARCH_RADIUS + dx, y0 = base_y + SEARCH_RADIUS + dy;
    const int x1 = x0 + TEMPLATE_SIZE, y1 = y0 + TEMPLATE_SIZE;

    // S_w/S_ww: same O(1) global box query as the sum-table kernel — already
    // cheap (4 reads), so it stays on the integral image rather than adding
    // shared-memory bookkeeping that would not measurably help (see header).
    const int64_t S_w = ii_sum[ii_index(y1, x1)] - ii_sum[ii_index(y0, x1)]
                       - ii_sum[ii_index(y1, x0)] + ii_sum[ii_index(y0, x0)];
    const int64_t S_ww = static_cast<int64_t>(ii_sumsq[ii_index(y1, x1)] - ii_sumsq[ii_index(y0, x1)]
                        - ii_sumsq[ii_index(y1, x0)] + ii_sumsq[ii_index(y0, x0)]);

    // S_wt: THE optimized loop — local_x0/local_y0 are this thread's patch
    // origin WITHIN s_window (window-local coordinates, not tray-image
    // coordinates); every read below is shared memory, not global.
    const int local_x0 = SEARCH_RADIUS + dx, local_y0 = SEARCH_RADIUS + dy;
    int64_t S_wt = 0;
    for (int ty = 0; ty < TEMPLATE_SIZE; ++ty) {
        const uint8_t* wrow = s_window + (local_y0 + ty) * WINDOW + local_x0;
        const uint8_t* trow = s_template + ty * TEMPLATE_SIZE;
        for (int tx = 0; tx < TEMPLATE_SIZE; ++tx)
            S_wt += static_cast<int64_t>(wrow[tx]) * trow[tx];
    }

    scores[score_index(slot, tmpl, oy, ox)] = ncc_from_sums(S_w, S_ww, S_wt, g_S_t[tmpl], g_S_tt[tmpl]);
}

void launch_ncc_shared(const uint8_t* d_img, const uint32_t* d_ii_sum, const uint64_t* d_ii_sumsq,
                       const uint8_t* d_templates, float* d_scores)
{
    const dim3 block(NUM_OFFSETS_1D, NUM_OFFSETS_1D);
    const dim3 grid(NUM_SLOTS, NUM_TEMPLATES);
    ncc_shared_kernel<<<grid, block>>>(d_img, d_ii_sum, d_ii_sumsq, d_templates, d_scores);
    CUDA_CHECK_LAST_ERROR("ncc_shared_kernel launch");
}
