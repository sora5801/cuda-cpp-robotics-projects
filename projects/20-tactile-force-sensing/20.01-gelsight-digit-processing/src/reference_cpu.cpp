// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 20.01
//                     GelSight/DIGIT processing: contact patch, shear field
//                     via optical flow, slip detection in real time
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), and this project leans on both
// harder than most because it runs the comparison on EVERY frame of the
// 100-frame demo, not a single checkpoint:
//
//   1) It is the CORRECTNESS ORACLE. Every function below is a literal,
//      sequential restatement of its kernels.cu twin — same loops, same
//      tie-break rule, same bounds checks — so main.cu's VERIFY stage can
//      demand EXACT (bit-for-bit) agreement, not a floating-point
//      tolerance. Every operation in this pipeline (threshold compares,
//      min/max filters, integer atomic-equivalent sums, argmin search) is
//      integer or exact-threshold arithmetic on the SAME uint8 input bytes
//      both paths receive — there is no rounding anywhere for a tolerance
//      to paper over (kernels.cuh; contrast with 08.01's RK4 floating-point
//      chain, which genuinely needs one).
//
//   2) It is the TEACHING BASELINE. Read a function here, then its kernels.cu
//      twin: the transformation is always "the outer loop over pixels/
//      markers became one thread per pixel/marker" — nothing else changes.
//
// Rules for this file: plain C++17, no CUDA headers, no cleverness. This
// file is compiled by the HOST compiler (cl.exe); kernels.cuh's declarations
// here are all plain host function signatures (no __CUDACC__ fence needed —
// see kernels.cuh's header comment on why this project's header needs none).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

// ---------------------------------------------------------------------------
// contact_mask_cpu — sequential twin of contact_mask_kernel. See kernels.cu
// for the full doc comment; every function below repeats only what differs
// (the loop shape), not the algorithmic reasoning already written there.
// ---------------------------------------------------------------------------
void contact_mask_cpu(const unsigned char* frame, const unsigned char* baseline,
                      unsigned char* mask, int W, int H, int threshold)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            const int diff = static_cast<int>(frame[idx]) - static_cast<int>(baseline[idx]);
            const int adiff = diff < 0 ? -diff : diff;
            mask[idx] = (adiff >= threshold) ? 255 : 0;
        }
    }
}

// erode3_cpu / dilate3_cpu — sequential twins of erode3_kernel/dilate3_kernel.
void erode3_cpu(const unsigned char* in, unsigned char* out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            unsigned char min_val = 255;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const unsigned char v = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                           ? in[ny * W + nx] : 0;
                    if (v < min_val) min_val = v;
                }
            }
            out[y * W + x] = min_val;
        }
    }
}

void dilate3_cpu(const unsigned char* in, unsigned char* out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            unsigned char max_val = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const unsigned char v = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                           ? in[ny * W + nx] : 0;
                    if (v > max_val) max_val = v;
                }
            }
            out[y * W + x] = max_val;
        }
    }
}

// patch_stats_cpu — sequential accumulation (the CPU never needs atomics:
// one thread of control, no races by construction — the honest reason a
// serial reduction is just "add it up" while the GPU twin needs atomicAdd).
// Caller must zero *area/*sumx/*sumy first (same contract as the GPU launcher).
void patch_stats_cpu(const unsigned char* mask, int W, int H,
                     unsigned long long* area,
                     unsigned long long* sumx,
                     unsigned long long* sumy)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (mask[y * W + x] != 0) {
                *area += 1ULL;
                *sumx += static_cast<unsigned long long>(x);
                *sumy += static_cast<unsigned long long>(y);
            }
        }
    }
}

// detect_markers_cpu — sequential twin of detect_markers_kernel. The window
// scan order (dy outer ascending, dx inner ascending, strict less-than to
// update) is copied EXACTLY from the kernel — this is the one place where
// getting the loop order "slightly different but mathematically equivalent"
// would silently break bit-exact agreement on any frame with a tied window
// minimum, so it is called out here explicitly rather than left implicit.
void detect_markers_cpu(const unsigned char* frame, const Vec2f* rest_pos,
                        int num_markers, int W, int H, int search_radius,
                        Vec2f* detected_pos, int* min_intensity)
{
    for (int i = 0; i < num_markers; ++i) {
        const int cx = static_cast<int>(rest_pos[i].x + 0.5f);
        const int cy = static_cast<int>(rest_pos[i].y + 0.5f);

        int best_val = 256;
        int best_x = cx, best_y = cy;
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                const int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const int v = static_cast<int>(frame[ny * W + nx]);
                if (v < best_val) {
                    best_val = v;
                    best_x = nx;
                    best_y = ny;
                }
            }
        }
        detected_pos[i].x = static_cast<float>(best_x);
        detected_pos[i].y = static_cast<float>(best_y);
        min_intensity[i] = best_val;
    }
}

// track_markers_cpu — sequential twin of track_markers_kernel.
void track_markers_cpu(const Vec2f* detected_pos, const int* min_intensity,
                       const Vec2f* rest_pos, const unsigned char* mask,
                       int num_markers, int W, int H, int detect_threshold,
                       Vec2f* displacement, unsigned char* valid,
                       unsigned char* in_contact)
{
    for (int i = 0; i < num_markers; ++i) {
        displacement[i].x = detected_pos[i].x - rest_pos[i].x;
        displacement[i].y = detected_pos[i].y - rest_pos[i].y;
        valid[i] = (min_intensity[i] < detect_threshold) ? 1 : 0;

        const int rx = static_cast<int>(rest_pos[i].x + 0.5f);
        const int ry = static_cast<int>(rest_pos[i].y + 0.5f);
        const bool inb = (rx >= 0 && rx < W && ry >= 0 && ry < H);
        in_contact[i] = (inb && mask[ry * W + rx] != 0) ? 1 : 0;
    }
}
