// ===========================================================================
// kernels.cu — GPU implementation for project 12.01
//              TensorRT deployment with custom CUDA pre/post kernels:
//              NMS, argmax decode, keypoint extraction
//
// Six kernels, each a MAP over a small, independent index space — the same
// "one thread per output element" pattern this whole repository teaches,
// applied to the specific shapes a deployed detector needs (CLAUDE.md §6.2
// style). None of them are large by GPU standards (the teaching image is
// tiny on purpose — see README "Limitations"); the point is to get the
// INDEXING, the MEMORY LAYOUT, and the PARALLEL-VS-SEQUENTIAL judgment
// calls exactly right, because those are what a real deployment reuses
// unchanged when the tensors get 1000x bigger (THEORY.md "GPU mapping").
//
// All shapes/constants come from kernels.cuh — the single source shared
// with the CPU oracle (reference_cpu.cpp); every function below is a
// deliberate, independent twin of one there (CLAUDE.md §5: a bug common to
// both would hide from the GPU-vs-CPU verify gate, so we do NOT share code
// between them, only the numeric contract in the header).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Repo-default launch geometry: 256 threads/block (warp multiple, good
// occupancy on sm_75..sm_89), grid = ceil(n/block). Every launcher below
// reuses this — see 33.01/08.01 for the same reasoning spelled out once.
// ---------------------------------------------------------------------------
static inline int grid_for(int n, int block) { return (n + block - 1) / block; }

// ===========================================================================
// KERNEL 1 — preprocess_kernel: the standard pre-kernel every deployed
// vision model needs. HWC uint8 [kSrcH,kSrcW,3] -> CHW float32
// [3,kNetH,kNetW] in one pass: bilinear resize + per-channel normalize +
// layout transpose.
//
// Thread-to-data mapping: thread i owns ONE output element of the CHW
// tensor. i = c*kNetH*kNetW + oy*kNetW + ox (channel-major, matching the
// output layout exactly — see the decomposition below). Grid: ceil(3*64*64
// / 256) = 48 blocks. This is a pure MAP (33.01's pattern): every output
// element is independent, so a grid-stride-free one-shot launch is enough.
//
// THE RESIZE MATH (half-pixel-center / "align_corners=false" convention —
// the OpenCV/PyTorch default, and the one this project standardizes on;
// THEORY.md derives it from "where does output pixel (ox,oy)'s CENTER fall
// in source coordinates?"):
//     scale = src_size / net_size                    (80/64 = 1.25 here)
//     src_coord = (dst_coord + 0.5) * scale - 0.5     (sub-pixel, may be
//                                                       negative at dst=0!)
//     clamp src_coord to [0, src_size-1]              (no extrapolation)
//     x0 = floor(src_coord); x1 = min(x0+1, src_size-1); frac = src_coord-x0
// Bilinear blends the 4 neighbors (v00,v01,v10,v11) with (1-frac)/frac
// weights along each axis, in that order (x first, then y) — the standard,
// separable bilinear formula.
//
// Memory behavior: src_hwc is read with a small, scattered 2x2 footprint
// per output pixel (interleaved RGB, so 3 output channels at the same
// (ox,oy) reuse the SAME 4 source pixels — a real optimization would cache
// them in shared memory or registers across the 3 channels; here every
// thread re-fetches independently for clarity, and at 80x80 input the
// entire image is a few L2 lines away regardless — see THEORY.md "GPU
// mapping" for the production note). net_chw is written once, coalesced
// within a channel plane (consecutive ox -> consecutive addresses).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void preprocess_kernel(const uint8_t* __restrict__ src_hwc,
                                  float* __restrict__ net_chw)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's CHW output index
    const int total = 3 * kNetH * kNetW;
    if (i >= total) return;                                // ragged-tail guard

    // Decompose the flat CHW index: channel-major, then row, then column —
    // MUST match how conv1 later reads this same buffer (kernels.cuh layout
    // convention comment).
    const int c  = i / (kNetH * kNetW);
    const int rem = i % (kNetH * kNetW);
    const int oy = rem / kNetW;
    const int ox = rem % kNetW;

    // Half-pixel-center source coordinate for this output pixel (see the
    // file-header derivation). scale_x/scale_y are compile-time-constant
    // ratios (80/64) but written as runtime floats for clarity.
    const float scale_x = static_cast<float>(kSrcW) / static_cast<float>(kNetW);
    const float scale_y = static_cast<float>(kSrcH) / static_cast<float>(kNetH);
    float sx = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
    float sy = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
    sx = clampf(sx, 0.0f, static_cast<float>(kSrcW - 1));
    sy = clampf(sy, 0.0f, static_cast<float>(kSrcH - 1));

    const int x0 = static_cast<int>(sx);                   // floor (sx >= 0 after clamp)
    const int y0 = static_cast<int>(sy);
    const int x1 = min(x0 + 1, kSrcW - 1);                  // clamp the "+1" neighbor at the border
    const int y1 = min(y0 + 1, kSrcH - 1);
    const float fx = sx - static_cast<float>(x0);           // fractional part in [0,1)
    const float fy = sy - static_cast<float>(y0);

    // Fetch the 4 neighbors of CHANNEL c only — src_hwc is interleaved, so
    // channel c of pixel (x,y) lives at src_hwc[(y*kSrcW + x)*3 + c].
    const float v00 = static_cast<float>(src_hwc[(y0 * kSrcW + x0) * 3 + c]);
    const float v01 = static_cast<float>(src_hwc[(y0 * kSrcW + x1) * 3 + c]);
    const float v10 = static_cast<float>(src_hwc[(y1 * kSrcW + x0) * 3 + c]);
    const float v11 = static_cast<float>(src_hwc[(y1 * kSrcW + x1) * 3 + c]);
    const float top = v00 * (1.0f - fx) + v01 * fx;         // blend along x at row y0
    const float bot = v10 * (1.0f - fx) + v11 * fx;         // blend along x at row y1
    const float resized = top * (1.0f - fy) + bot * fy;     // blend along y

    // Normalize (mean/std — kernels.cuh SECTION 1) and write in CHW order.
    // This single write both TRANSPOSES (HWC read -> CHW write) and
    // NORMALIZES — the two jobs the file header promises.
    net_chw[i] = (resized - kPixelMean) / kPixelStd;
}

void launch_preprocess(const uint8_t* d_src_hwc, float* d_net_chw)
{
    const int total = 3 * kNetH * kNetW;                    // 12288 output elements
    const int block = 256;
    preprocess_kernel<<<grid_for(total, block), block>>>(d_src_hwc, d_net_chw);
    CUDA_CHECK_LAST_ERROR("preprocess_kernel launch");
}

// ===========================================================================
// KERNEL 2 — conv2d_kernel: a GENERIC direct 2D convolution + optional
// ReLU. Reused UNCHANGED for conv1 (3->2ch), conv2 (2->2ch), and the 1x1
// detection head (2->6ch) — three launches, one kernel. This is the
// project's clearest "no black boxes" moment (CLAUDE.md §1): the exact
// arithmetic cuDNN/TensorRT fuse and tune for you is spelled out here.
//
// Thread-to-data mapping: thread i owns ONE output element (co,oy,ox).
// i = co*Hout*Wout + oy*Wout + ox (channel-major — matches every other
// tensor in this project). Each thread walks its OWN Cin*K*K receptive
// field sequentially — no shared memory, no cooperation between threads:
// output elements never share work, only (overlapping) READS of `in`,
// which the L1/L2 cache absorbs at these tiny sizes (conv1: 32*32*2=2048
// threads reading a 3*64*64=12288-float input; the whole input fits in a
// few KB, comfortably cached — see THEORY.md "GPU mapping" for why this
// stops being true at production image sizes, and what the standard fix
// (im2col+GEMM, or a tiled shared-memory conv) looks like).
//
// A 1x1 CONV IS A PER-CELL LINEAR LAYER: with K=1, pad=0, stride=1, the
// inner ky/kx loops degenerate to a single tap per input channel — exactly
// matrix-vector multiply at every spatial location. This is why the
// detection head (6 output "channels": 2 class scores + 4 box-regression
// values) is implemented as a conv layer at all: it is the standard,
// hardware-friendly way to express "a small MLP applied independently at
// every grid cell" (every modern single-stage detector head is built this
// way — see README "Prior art").
//
// Numerics: FP32 throughout; the accumulation ORDER is ci-then-ky-then-kx,
// documented because reference_cpu.cpp uses the SAME nesting (needed for
// the GPU-vs-CPU tolerance discussion in THEORY.md — different orders can
// legally produce different FP32 rounding for the same mathematical sum).
// ---------------------------------------------------------------------------
__global__ void conv2d_kernel(const float* __restrict__ in,
                              const float* __restrict__ w,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int Cin, int Cout, int Hin, int Win,
                              int K, int stride, int pad,
                              int Hout, int Wout,
                              bool relu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Cout * Hout * Wout;
    if (i >= total) return;

    const int co  = i / (Hout * Wout);
    const int rem = i % (Hout * Wout);
    const int oy  = rem / Wout;
    const int ox  = rem % Wout;

    float acc = b[co];                                      // start from this output channel's bias
    // Walk the receptive field: ci outermost, then ky, then kx — the
    // accumulation order reference_cpu.cpp mirrors exactly (see file header).
    for (int ci = 0; ci < Cin; ++ci) {
        for (int ky = 0; ky < K; ++ky) {
            const int iy = oy * stride - pad + ky;           // input row this tap reads
            if (iy < 0 || iy >= Hin) continue;                // zero-padding: skip out-of-bounds taps
            for (int kx = 0; kx < K; ++kx) {
                const int ix = ox * stride - pad + kx;        // input column this tap reads
                if (ix < 0 || ix >= Win) continue;
                const float wt = w[((co * Cin + ci) * K + ky) * K + kx];
                const float xv = in[(ci * Hin + iy) * Win + ix];
                acc += wt * xv;
            }
        }
    }
    if (relu && acc < 0.0f) acc = 0.0f;                       // ReLU: max(0,x), applied once at the end
    out[i] = acc;
}

void launch_conv2d(const float* d_in, const float* d_w, const float* d_b,
                   float* d_out,
                   int Cin, int Cout, int Hin, int Win,
                   int K, int stride, int pad, int Hout, int Wout,
                   bool relu)
{
    const int total = Cout * Hout * Wout;
    const int block = 256;
    conv2d_kernel<<<grid_for(total, block), block>>>(
        d_in, d_w, d_b, d_out, Cin, Cout, Hin, Win, K, stride, pad, Hout, Wout, relu);
    CUDA_CHECK_LAST_ERROR("conv2d_kernel launch");
}

// ===========================================================================
// KERNEL 3 — argmax_decode_kernel: the bullet's first named post-kernel.
// One thread per GRID CELL (kGridH*kGridW = 256): argmax over the
// kNumClasses score channels of the head's output, producing a per-cell
// winning class and its score. This is the classic "collapse the class
// axis" reduction every single-stage detector's decode step performs —
// here it is a trivial 2-way max (kNumClasses=2), but the kernel is
// written generically over kNumClasses so the pattern reads correctly at
// any class count.
//
// Thread-to-data mapping: thread idx owns cell (gx,gy) with
// idx = gy*kGridW + gx. head_out is [kHeadOut, kGridH, kGridW] channel-
// major (SECTION 1 layout); this kernel only reads the first kNumClasses
// channels (the box-regression channels are threshold_box_decode's job).
// No shared memory, no atomics, no divergence beyond the trivial tail
// guard — every thread does identical, independent work.
// ---------------------------------------------------------------------------
__global__ void argmax_decode_kernel(const float* __restrict__ head_out,
                                     int* __restrict__ best_class,
                                     float* __restrict__ best_score)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // cell index, 0..kGridH*kGridW-1
    const int ncells = kGridH * kGridW;
    if (idx >= ncells) return;

    int   winner = 0;
    // head_out channel 0 at this cell — the "class 0" score plane.
    float winner_score = head_out[0 * ncells + idx];
#pragma unroll
    for (int c = 1; c < kNumClasses; ++c) {
        const float s = head_out[c * ncells + idx];
        if (s > winner_score) { winner_score = s; winner = c; }  // strict '>' -> first max wins ties
    }
    best_class[idx] = winner;
    best_score[idx] = winner_score;
}

void launch_argmax_decode(const float* d_head_out, int* d_best_class, float* d_best_score)
{
    const int ncells = kGridH * kGridW;
    const int block = 256;
    argmax_decode_kernel<<<grid_for(ncells, block), block>>>(d_head_out, d_best_class, d_best_score);
    CUDA_CHECK_LAST_ERROR("argmax_decode_kernel launch");
}

// ===========================================================================
// KERNEL 4 — threshold_box_decode_kernel: score thresholding + box decode
// via ANCHOR ARITHMETIC, the bullet's second named step. One thread per
// grid cell; cells that pass the confidence gate atomically claim a slot
// in the compacted `candidates` array — a textbook GPU STREAM COMPACTION
// (turning a SPARSE boolean-masked set — "which of 256 cells fired?" —
// into a DENSE array NMS can index linearly).
//
// ANCHOR ARITHMETIC (the general formula — THEORY.md derives it; here the
// regression weights happen to be zero, see kernels.cuh SECTION 1, so the
// decode always resolves to the bare anchor, but the ARITHMETIC below is
// the real, general YOLO-style decode used by trained detectors):
//     cx = (gx + sigmoid(tx)) * kCellPx      cell-relative center, x
//     cy = (gy + sigmoid(ty)) * kCellPx      cell-relative center, y
//     w  = kAnchorPx * exp(tw)               anchor width scaled log-space
//     h  = kAnchorPx * exp(th)               anchor height scaled log-space
// sigmoid() keeps the center INSIDE the cell (a real regression head is
// only ever asked to nudge the anchor, never relocate it to another cell —
// that is what makes per-cell prediction well-posed at all); exp() makes
// the width/height regression symmetric in log-space (a box can shrink to
// zero but never go negative, and "twice as wide" and "half as wide" are
// equally easy targets for the (here: absent) learned weights to hit).
//
// ATOMICS (CLAUDE.md §6.1 rule 2 — documented explicitly): `count` is a
// SINGLE int in device memory, shared by every thread in the launch.
// atomicAdd(count, 1) is a read-modify-write the hardware serializes
// across all callers, returning to EACH caller the value *before* its own
// increment — i.e. a unique, race-free output slot per firing thread, with
// no two threads ever colliding on the same index. This is the simplest
// possible GPU compaction primitive: correct at any candidate density,
// and — because at most 256 threads ever call it here — costs nothing
// worth optimizing away (contrast with real detectors compacting tens of
// thousands of anchors, where atomics contention becomes a real profiling
// concern; THEORY.md's "GPU mapping" section discusses the escape hatches:
// per-block local compaction + a single cross-block atomic, or a prefix-
// sum/stream-compaction library call).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__global__ void threshold_box_decode_kernel(const int* __restrict__ best_class,
                                            const float* __restrict__ best_score,
                                            const float* __restrict__ head_out,
                                            Detection* __restrict__ candidates,
                                            int* __restrict__ count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ncells = kGridH * kGridW;
    if (idx >= ncells) return;

    const float score = best_score[idx];
    if (score <= kScoreThreshold) return;                     // background/weak cell: nothing to do

    const int gy = idx / kGridW;
    const int gx = idx % kGridW;
    const int cls = best_class[idx];

    // Box-regression channels live at head_out channels
    // [kNumClasses .. kNumClasses+3] = tx,ty,tw,th, same cell index.
    const float tx = head_out[(kNumClasses + 0) * ncells + idx];
    const float ty = head_out[(kNumClasses + 1) * ncells + idx];
    const float tw = head_out[(kNumClasses + 2) * ncells + idx];
    const float th = head_out[(kNumClasses + 3) * ncells + idx];

    const float cx = (static_cast<float>(gx) + sigmoidf_(tx)) * kCellPx;
    const float cy = (static_cast<float>(gy) + sigmoidf_(ty)) * kCellPx;
    const float w  = kAnchorPx * expf(tw);
    const float h  = kAnchorPx * expf(th);

    // Claim a compaction slot. See the file-header ATOMICS note. The guard
    // below is PROVABLY unreachable (at most ncells threads ever reach this
    // line, and kMaxCandidates == ncells exactly — kernels.cuh SECTION 1) —
    // kept anyway as a documented defensive check: silently corrupting
    // memory on a future refactor that changes kMaxCandidates is a far
    // worse failure than one dropped detection (CLAUDE.md "no black boxes"
    // extends to "no silent out-of-bounds writes").
    const int slot = atomicAdd(count, 1);
    if (slot >= kMaxCandidates) return;

    Detection det;
    det.score      = score;
    det.class_id   = cls;
    det.cell_index = idx;
    det.x0 = cx - w * 0.5f;
    det.y0 = cy - h * 0.5f;
    det.x1 = cx + w * 0.5f;
    det.y1 = cy + h * 0.5f;
    det.kp_x = -1.0f;    // sentinel: keypoint_extract_kernel fills this in later
    det.kp_y = -1.0f;
    candidates[slot] = det;
}

void launch_threshold_box_decode(const int* d_best_class, const float* d_best_score,
                                 const float* d_head_out,
                                 Detection* d_candidates, int* d_count)
{
    // Reset the compaction counter EVERY call — a stale count from a
    // previous frame/image would silently misplace every slot (see the
    // launcher's doc-comment in kernels.cuh).
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    const int ncells = kGridH * kGridW;
    const int block = 256;
    threshold_box_decode_kernel<<<grid_for(ncells, block), block>>>(
        d_best_class, d_best_score, d_head_out, d_candidates, d_count);
    CUDA_CHECK_LAST_ERROR("threshold_box_decode_kernel launch");
}

// ===========================================================================
// KERNEL 5 — iou_matrix_kernel: the bullet's third named step, NMS,
// starts here — the GENUINELY PARALLEL half of it. Given n candidate boxes
// (already sorted by score descending on the host — see main.cu), compute
// EVERY pairwise IoU up front, as one dense n*n matrix. This is an
// embarrassingly parallel ALL-PAIRS computation: iou[i][j] does not depend
// on iou[i'][j'] for any other pair, so one thread per (i,j) is the
// natural mapping — exactly like 33.01's batched map, just over box pairs
// instead of matrices.
//
// THE PARALLELISM TENSION IN NMS (why this kernel exists but greedy
// suppression, next, does NOT get one): classic greedy NMS is "sort by
// score, then for each survivor in order, suppress every LOWER-scored box
// it overlaps enough" — a chain of decisions where box j's fate depends on
// whether box i (i<j) itself survived, which depends on box i-1, and so
// on. That dependency chain is SEQUENTIAL by construction; parallelizing
// it correctly (e.g. NVIDIA's batched-NMS / matrix-NMS variants) requires
// real algorithmic cleverness (soft-NMS-style score decay, or computing
// suppression from the IoU matrix's UPPER-TRIANGULAR structure directly)
// that is out of scope for a teaching kernel. What DOES parallelize
// trivially — and is where the O(n^2) cost of NMS actually lives — is the
// IoU matrix itself, so that is what gets a kernel; the sequential scan
// that CONSUMES it runs on the host, in main.cu, exactly like 08.01 keeps
// its O(K*T) softmin blend on the host (same reasoning: trivial, honestly
// sequential bookkeeping does not deserve a kernel just because it CAN
// have one — THEORY.md discusses the real fix, matrix-NMS, in "where this
// sits in the real world").
//
// Thread-to-data mapping: thread idx owns pair (i,j) with idx = i*n + j.
// We compute the FULL symmetric n*n matrix (not just the upper triangle):
// at these candidate counts (tens, never more than kMaxCandidates=256) the
// ~2x redundant work is invisible, and a branch-free full matrix is
// simpler to read and to index from the host afterward. No shared memory
// (each thread's inputs — two Detection boxes — are read once each; L1
// absorbs the reuse across the row/column), no atomics.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float iou_device(const Detection& a, const Detection& b)
{
    const float ix0 = fmaxf(a.x0, b.x0);
    const float iy0 = fmaxf(a.y0, b.y0);
    const float ix1 = fminf(a.x1, b.x1);
    const float iy1 = fminf(a.y1, b.y1);
    const float iw = fmaxf(0.0f, ix1 - ix0);
    const float ih = fmaxf(0.0f, iy1 - iy0);
    const float inter = iw * ih;
    const float area_a = fmaxf(0.0f, a.x1 - a.x0) * fmaxf(0.0f, a.y1 - a.y0);
    const float area_b = fmaxf(0.0f, b.x1 - b.x0) * fmaxf(0.0f, b.y1 - b.y0);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

__global__ void iou_matrix_kernel(const Detection* __restrict__ boxes,
                                  int n,
                                  float* __restrict__ iou_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n;
    if (idx >= total) return;

    const int i = idx / n;
    const int j = idx % n;
    iou_out[idx] = (i == j) ? 1.0f : iou_device(boxes[i], boxes[j]);
}

void launch_iou_matrix(const Detection* d_boxes_sorted, int n, float* d_iou_out)
{
    if (n <= 0) return;                                      // nothing to do — avoid a 0-block launch
    const int total = n * n;
    const int block = 256;
    iou_matrix_kernel<<<grid_for(total, block), block>>>(d_boxes_sorted, n, d_iou_out);
    CUDA_CHECK_LAST_ERROR("iou_matrix_kernel launch");
}

// ===========================================================================
// KERNEL 6 — keypoint_extract_kernel: the bullet's third named post-kernel.
// One thread per SURVIVING (post-NMS) detection: a local-window argmax
// over the winning class's own score heatmap, refining a keypoint location
// near the detection. This project reuses the class-score map already
// computed by argmax_decode/threshold_box_decode as the heatmap (a
// CenterNet-style choice — see README "Prior art"): a real network would
// often learn an independent keypoint branch, but the KERNEL — a bounded
// local-window search for the strongest response near a candidate location
// — is identical either way, and that kernel is what this project teaches.
//
// Thread-to-data mapping: thread idx owns detection idx (n is tiny after
// NMS — typically single digits in this demo — so a single small block
// covers the whole launch). Each thread independently scans its own
// (2*kKeypointWinRadius+1)^2 window (5x5 = 25 cells here) of ONE heatmap
// channel (head_out[class_id]), clipped to the grid — no shared memory (no
// data reuse across threads: different detections generally read
// different, at most lightly overlapping, windows), no atomics.
//
// TIE-BREAKING (must match reference_cpu.cpp EXACTLY, bit for bit, or the
// GPU-vs-CPU verify gate could report a false failure on a plateau): the
// scan visits cells in ROW-MAJOR order (y outer, x inner) and keeps the
// FIRST strictly-greater value — i.e. "first max encountered, top-to-
// bottom, left-to-right". This project's flat-colored synthetic objects
// often DO plateau (several cells tied at the same peak score — see
// THEORY.md), which is exactly the case a documented, deterministic tie-
// break exists to make reproducible instead of undefined.
// ---------------------------------------------------------------------------
__global__ void keypoint_extract_kernel(Detection* __restrict__ detections,
                                        int n,
                                        const float* __restrict__ head_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Detection det = detections[idx];                          // local copy; write back once at the end
    const int gy = det.cell_index / kGridW;
    const int gx = det.cell_index % kGridW;
    const int ncells = kGridH * kGridW;
    const float* heatmap = head_out + det.class_id * ncells;   // this detection's class-score plane

    const int y_lo = max(0, gy - kKeypointWinRadius);
    const int y_hi = min(kGridH - 1, gy + kKeypointWinRadius);
    const int x_lo = max(0, gx - kKeypointWinRadius);
    const int x_hi = min(kGridW - 1, gx + kKeypointWinRadius);

    int best_y = gy, best_x = gx;                              // fallback: the detection's own cell
    float best_val = heatmap[gy * kGridW + gx];
    for (int y = y_lo; y <= y_hi; ++y) {                       // row-major scan — see tie-break note
        for (int x = x_lo; x <= x_hi; ++x) {
            const float v = heatmap[y * kGridW + x];
            if (v > best_val) {                                 // STRICT '>' -> first max wins ties
                best_val = v;
                best_y = y;
                best_x = x;
            }
        }
    }

    // Report the keypoint at the winning cell's CENTER, network-input
    // pixel space (the same cell-to-pixel convention as the box decode).
    det.kp_x = (static_cast<float>(best_x) + 0.5f) * kCellPx;
    det.kp_y = (static_cast<float>(best_y) + 0.5f) * kCellPx;
    detections[idx] = det;
}

void launch_keypoint_extract(Detection* d_detections, int n, const float* d_head_out)
{
    if (n <= 0) return;
    const int block = 32;                                      // n is always tiny post-NMS; one small block
    keypoint_extract_kernel<<<grid_for(n, block), block>>>(d_detections, n, d_head_out);
    CUDA_CHECK_LAST_ERROR("keypoint_extract_kernel launch");
}
