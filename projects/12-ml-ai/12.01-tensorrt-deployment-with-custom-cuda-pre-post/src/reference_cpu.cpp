// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 12.01
//                     TensorRT deployment with custom CUDA pre/post kernels
//
// The correctness oracle for the ENTIRE pipeline — every stage kernels.cu
// implements on the GPU, mirrored here as an independent, sequential, plain
// C++17 twin: preprocessing, both conv layers, the detection head (all via
// one generic conv2d_cpu, exactly like the GPU side reuses one conv2d
// kernel), argmax decode, threshold + anchor-arithmetic box decode, NMS
// (IoU + greedy suppression, both phases sequential here — see the
// parallelism-tension note in kernels.cu, kernel 5's header), and keypoint
// extraction.
//
// WHY a full pipeline twin, not just a final-answer check? Because a
// detector pipeline can be "right at the end, wrong in the middle" — e.g.
// a box-decode off-by-one that happens to still pass the ground-truth
// tolerance on THIS scene would slip through a final-answer-only check.
// main.cu therefore runs BOTH paths stage-by-stage and diffs every
// intermediate tensor (the preprocessed image, conv1/conv2 activations,
// the head's raw output) within a documented tolerance — CLAUDE.md §5's
// GPU-vs-CPU gate, applied at every stage instead of just the last one.
//
// DELIBERATE DUPLICATION (CLAUDE.md §5/§9): every function below is an
// INDEPENDENT re-implementation of its kernels.cu twin — same math, same
// accumulation order (documented per-function), different code, so a bug
// that would otherwise be common to "the algorithm" cannot silently agree
// with itself on both sides of the verify gate. The one exception is pure
// bookkeeping with no algorithmic content (std::sort's comparator, used
// identically by both main.cu's GPU-path candidate sort and nms_cpu below
// — sorting itself is a library call, not "the detector", exactly like
// 08.01 does not re-derive std::exp on both sides of its softmin).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <algorithm>   // std::stable_sort
#include <cmath>       // std::exp, std::fabs
#include <cstdint>

// ---------------------------------------------------------------------------
// preprocess_cpu — line-by-line twin of preprocess_kernel: bilinear resize
// (half-pixel-center convention) + per-channel normalize + HWC->CHW
// transpose. See kernels.cu KERNEL 1 for the full derivation; not repeated
// here — only the CPU-specific notes (plain loops instead of one thread
// per element; std:: spellings instead of CUDA intrinsics).
// ---------------------------------------------------------------------------
static float clampf_cpu(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

void preprocess_cpu(const uint8_t* src_hwc, float* net_chw)
{
    const float scale_x = static_cast<float>(kSrcW) / static_cast<float>(kNetW);
    const float scale_y = static_cast<float>(kSrcH) / static_cast<float>(kNetH);

    for (int c = 0; c < 3; ++c) {
        for (int oy = 0; oy < kNetH; ++oy) {
            float sy = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
            sy = clampf_cpu(sy, 0.0f, static_cast<float>(kSrcH - 1));
            const int y0 = static_cast<int>(sy);
            const int y1 = std::min(y0 + 1, kSrcH - 1);
            const float fy = sy - static_cast<float>(y0);

            for (int ox = 0; ox < kNetW; ++ox) {
                float sx = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
                sx = clampf_cpu(sx, 0.0f, static_cast<float>(kSrcW - 1));
                const int x0 = static_cast<int>(sx);
                const int x1 = std::min(x0 + 1, kSrcW - 1);
                const float fx = sx - static_cast<float>(x0);

                const float v00 = static_cast<float>(src_hwc[(y0 * kSrcW + x0) * 3 + c]);
                const float v01 = static_cast<float>(src_hwc[(y0 * kSrcW + x1) * 3 + c]);
                const float v10 = static_cast<float>(src_hwc[(y1 * kSrcW + x0) * 3 + c]);
                const float v11 = static_cast<float>(src_hwc[(y1 * kSrcW + x1) * 3 + c]);
                const float top = v00 * (1.0f - fx) + v01 * fx;
                const float bot = v10 * (1.0f - fx) + v11 * fx;
                const float resized = top * (1.0f - fy) + bot * fy;

                net_chw[(c * kNetH + oy) * kNetW + ox] = (resized - kPixelMean) / kPixelStd;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// conv2d_cpu — line-by-line twin of conv2d_kernel: same generic direct
// convolution, same accumulation order (ci outermost, then ky, then kx —
// matching kernels.cu's KERNEL 2 comment exactly, because chained FP32
// sums can legally round differently under a different order — the
// justification main.cu's stage-wise tolerance cites).
// ---------------------------------------------------------------------------
void conv2d_cpu(const float* in, const float* w, const float* b, float* out,
                int Cin, int Cout, int Hin, int Win,
                int K, int stride, int pad, int Hout, int Wout,
                bool relu)
{
    for (int co = 0; co < Cout; ++co) {
        for (int oy = 0; oy < Hout; ++oy) {
            for (int ox = 0; ox < Wout; ++ox) {
                float acc = b[co];
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int ky = 0; ky < K; ++ky) {
                        const int iy = oy * stride - pad + ky;
                        if (iy < 0 || iy >= Hin) continue;
                        for (int kx = 0; kx < K; ++kx) {
                            const int ix = ox * stride - pad + kx;
                            if (ix < 0 || ix >= Win) continue;
                            const float wt = w[((co * Cin + ci) * K + ky) * K + kx];
                            const float xv = in[(ci * Hin + iy) * Win + ix];
                            acc += wt * xv;
                        }
                    }
                }
                if (relu && acc < 0.0f) acc = 0.0f;
                out[(co * Hout + oy) * Wout + ox] = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// argmax_decode_cpu — twin of argmax_decode_kernel: per-cell argmax over
// the kNumClasses score channels, strict '>' so the first (lowest-index)
// class wins ties — the SAME rule the GPU kernel applies, so the two paths
// cannot disagree on a tie.
// ---------------------------------------------------------------------------
void argmax_decode_cpu(const float* head_out, int* best_class, float* best_score)
{
    const int ncells = kGridH * kGridW;
    for (int idx = 0; idx < ncells; ++idx) {
        int winner = 0;
        float winner_score = head_out[0 * ncells + idx];
        for (int c = 1; c < kNumClasses; ++c) {
            const float s = head_out[c * ncells + idx];
            if (s > winner_score) { winner_score = s; winner = c; }
        }
        best_class[idx] = winner;
        best_score[idx] = winner_score;
    }
}

// ---------------------------------------------------------------------------
// threshold_box_decode_cpu — twin of threshold_box_decode_kernel: same
// score gate, same anchor-arithmetic box decode. The GPU kernel compacts
// via atomicAdd on a shared counter (a hardware-serialized race); the CPU
// has no race to resolve at all — a single thread appending to a
// std::vector-like array via out_list->count++ IS the sequential special
// case of the same idea. Documenting that contrast is the point: the GPU
// needs atomics only because MANY threads compete for the SAME compaction
// counter simultaneously; one CPU thread never does.
// ---------------------------------------------------------------------------
void threshold_box_decode_cpu(const int* best_class, const float* best_score,
                              const float* head_out,
                              DetectionList* out_list)
{
    out_list->count = 0;
    const int ncells = kGridH * kGridW;

    for (int idx = 0; idx < ncells; ++idx) {
        const float score = best_score[idx];
        if (score <= kScoreThreshold) continue;

        const int gy = idx / kGridW;
        const int gx = idx % kGridW;
        const int cls = best_class[idx];

        const float tx = head_out[(kNumClasses + 0) * ncells + idx];
        const float ty = head_out[(kNumClasses + 1) * ncells + idx];
        const float tw = head_out[(kNumClasses + 2) * ncells + idx];
        const float th = head_out[(kNumClasses + 3) * ncells + idx];

        const float sig_tx = 1.0f / (1.0f + std::exp(-tx));
        const float sig_ty = 1.0f / (1.0f + std::exp(-ty));
        const float cx = (static_cast<float>(gx) + sig_tx) * kCellPx;
        const float cy = (static_cast<float>(gy) + sig_ty) * kCellPx;
        const float w  = kAnchorPx * std::exp(tw);
        const float h  = kAnchorPx * std::exp(th);

        if (out_list->count >= kMaxCandidates) break;   // see kernels.cu: provably unreachable, kept honest
        Detection& det = out_list->items[out_list->count++];
        det.score      = score;
        det.class_id   = cls;
        det.cell_index = idx;
        det.x0 = cx - w * 0.5f;
        det.y0 = cy - h * 0.5f;
        det.x1 = cx + w * 0.5f;
        det.y1 = cy + h * 0.5f;
        det.kp_x = -1.0f;
        det.kp_y = -1.0f;
    }
}

// ---------------------------------------------------------------------------
// iou_host — CPU twin of kernels.cu's iou_device. Written independently
// (deliberate duplication, see file header) even though the formula is
// identical — a bug in the intersection/union arithmetic on one side would
// otherwise silently agree with the same bug on the other.
// ---------------------------------------------------------------------------
static float iou_host(const Detection& a, const Detection& b)
{
    const float ix0 = std::max(a.x0, b.x0);
    const float iy0 = std::max(a.y0, b.y0);
    const float ix1 = std::min(a.x1, b.x1);
    const float iy1 = std::min(a.y1, b.y1);
    const float iw = std::max(0.0f, ix1 - ix0);
    const float ih = std::max(0.0f, iy1 - iy0);
    const float inter = iw * ih;
    const float area_a = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
    const float area_b = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// ---------------------------------------------------------------------------
// nms_cpu — sequential greedy NMS, the CPU's natural home for BOTH phases
// of the algorithm (kernels.cu splits them: a real IoU-matrix KERNEL for
// the parallel part, a host scan for the sequential part — see kernel 5's
// header for why). Here there is no parallel hardware to exploit, so both
// phases are just... a loop:
//
//   1. STABLE sort by score descending, ties broken by cell_index
//      ASCENDING. This exact comparator is also used, separately, by
//      main.cu when it sorts the GPU path's compacted candidates before
//      calling launch_iou_matrix — using the SAME deterministic rule on
//      both paths means a score tie (this project's flat-colored objects
//      produce several — see THEORY.md) resolves identically everywhere,
//      so the two paths' SURVIVOR SETS are directly comparable, not just
//      their cardinalities.
//   2. Greedy scan: walk survivors in sorted order; a box already marked
//      suppressed contributes nothing; otherwise it survives and suppresses
//      every LATER, SAME-CLASS box it overlaps by more than iou_threshold.
//      Class-aware NMS (never suppress across classes) mirrors real
//      detectors, which do not want a confident "red" box silently erased
//      by an overlapping "blue" box.
// ---------------------------------------------------------------------------
void nms_cpu(DetectionList* candidates, float iou_threshold, DetectionList* out_kept)
{
    std::stable_sort(candidates->items, candidates->items + candidates->count,
                     [](const Detection& a, const Detection& b) {
                         if (a.score != b.score) return a.score > b.score;   // higher score first
                         return a.cell_index < b.cell_index;                 // deterministic tie-break
                     });

    bool suppressed[kMaxCandidates] = { false };
    out_kept->count = 0;
    for (int i = 0; i < candidates->count; ++i) {
        if (suppressed[i]) continue;
        out_kept->items[out_kept->count++] = candidates->items[i];          // i survives
        for (int j = i + 1; j < candidates->count; ++j) {
            if (suppressed[j]) continue;
            if (candidates->items[j].class_id != candidates->items[i].class_id) continue;  // class-aware
            if (iou_host(candidates->items[i], candidates->items[j]) > iou_threshold)
                suppressed[j] = true;
        }
    }
}

// ---------------------------------------------------------------------------
// keypoint_extract_cpu — twin of keypoint_extract_kernel: same local-
// window argmax over the winning class's score heatmap, same row-major
// scan order and strict '>' tie-break (see kernels.cu KERNEL 6 for why
// that ordering must match bit-for-bit).
// ---------------------------------------------------------------------------
void keypoint_extract_cpu(DetectionList* detections, const float* head_out)
{
    const int ncells = kGridH * kGridW;
    for (int i = 0; i < detections->count; ++i) {
        Detection& det = detections->items[i];
        const int gy = det.cell_index / kGridW;
        const int gx = det.cell_index % kGridW;
        const float* heatmap = head_out + det.class_id * ncells;

        const int y_lo = std::max(0, gy - kKeypointWinRadius);
        const int y_hi = std::min(kGridH - 1, gy + kKeypointWinRadius);
        const int x_lo = std::max(0, gx - kKeypointWinRadius);
        const int x_hi = std::min(kGridW - 1, gx + kKeypointWinRadius);

        int best_y = gy, best_x = gx;
        float best_val = heatmap[gy * kGridW + gx];
        for (int y = y_lo; y <= y_hi; ++y) {
            for (int x = x_lo; x <= x_hi; ++x) {
                const float v = heatmap[y * kGridW + x];
                if (v > best_val) {
                    best_val = v;
                    best_y = y;
                    best_x = x;
                }
            }
        }
        det.kp_x = (static_cast<float>(best_x) + 0.5f) * kCellPx;
        det.kp_y = (static_cast<float>(best_y) + 0.5f) * kCellPx;
    }
}
