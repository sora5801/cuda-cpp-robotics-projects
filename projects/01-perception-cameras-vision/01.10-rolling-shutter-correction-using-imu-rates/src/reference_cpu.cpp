// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.10
//                     (Rolling-shutter correction using IMU rates)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5):
//   1) It is the CORRECTNESS ORACLE — main.cu runs both rs_correct_kernel
//      (GPU) and rs_correct_cpu (here) on the SAME row LUT and RS frame,
//      and asserts element-wise agreement within a documented tolerance.
//   2) It is the TEACHING BASELINE — read this file first, then
//      kernels.cu's rs_correct_kernel: the per-pixel loop body is (by
//      design, for a MAP-pattern kernel) nearly identical; what changed is
//      "for each pixel" becoming "each thread owns one pixel".
//
// Independence ruling (see docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's
// file header for the full statement this repo adopted) — applied here:
//   * The quaternion algebra and the row-homography formula
//     (quat_normalize/quat_mul/quat_conj/quat_to_mat3/lerp_row_quat/
//     apply_row_rotation/bilinear_sample_gray, all in kernels.cuh) ARE the
//     camera/sensor model under test, and duplicating them here would be
//     pure token-for-token transcription — the repo's documented exception,
//     same one 01.01's compute_source_pixel()/distort_forward() and 09.01's
//     forward-kinematics primitives use. They are shared, HD, inline.
//   * The PER-PIXEL LOOP STRUCTURE — the fixed-point row-time search driving
//     those primitives — is typed a SECOND time, independently, right here,
//     rather than calling a shared "do_one_pixel()" function kernels.cu also
//     calls. That is what makes the GPU-vs-CPU comparison in main.cu able to
//     catch a genuine indexing/loop bug (a wrong iteration count, a stale
//     v_guess, an off-by-one in idx) rather than just re-confirming the two
//     sides share code.
//   * This project ALSO carries verification that bypasses the shared
//     primitives entirely (per the ruling's "at least one gate that does not
//     route through the shared code" requirement): main.cu's restoration
//     gate compares actual PIXEL CONTENT against an independently rendered
//     ground-truth image (never re-deriving the homography), and its
//     quaternion-integration self-check integrates a KNOWN constant angular
//     velocity and compares the result to the CLOSED-FORM analytic answer
//     (|omega|*dt), never calling lerp_row_quat/apply_row_rotation at all.
//
// Rules for this file: plain C++17, no CUDA headers, no cleverness — clarity
// beats speed here, always (this file is compiled by cl.exe, never nvcc; the
// __CUDACC__ fence in kernels.cuh hides __global__ declarations from it).
//
// Read this after: kernels.cu — then compare the two loop bodies side by side.
// ===========================================================================

#include "kernels.cuh"   // Quat + the shared HD camera-model primitives + this file's own prototype

#include <cmath>          // std::fabs

// ---------------------------------------------------------------------------
// rs_correct_cpu — sequential twin of kernels.cu's rs_correct_kernel.
//
// Parameters: see kernels.cuh's declaration for the full contract. All
// pointers are HOST pointers here (row_lut/rs_frame in, corrected/
// valid_mask/iter_delta out) — no device memory anywhere in this file.
//
// Complexity: O(W*H*kFixedPointIters) — a few hundred thousand quaternion-
// to-matrix conversions and bilinear samples; main.cu's [time] line reports
// the measured wall-clock cost, which is exactly the number the GPU kernel
// is being compared against for the "why bother with a GPU" lesson.
// ---------------------------------------------------------------------------
void rs_correct_cpu(const Quat* row_lut, int H,
                    const unsigned char* rs_frame, int W,
                    unsigned char* corrected, unsigned char* valid_mask, float* iter_delta)
{
    // Plain nested loops, row-major, matching the image layout documented
    // in kernels.cuh — deliberately the simplest correct statement of the
    // per-pixel search, no early-outs, no vectorization tricks.
    for (int yo = 0; yo < H; ++yo) {
        for (int xo = 0; xo < W; ++xo) {

            // Identical fixed-point search to the kernel's, typed
            // independently: seed the row guess at yo, refine it
            // kFixedPointIters times against the row LUT.
            float v_guess = static_cast<float>(yo);
            float xs = 0.0f, ys = 0.0f;
            float v_after_second_iter = 0.0f;

            for (int it = 0; it < kFixedPointIters; ++it) {
                const Quat q_rel = lerp_row_quat(row_lut, H, v_guess);
                float R[9];
                quat_to_mat3(q_rel, R);
                apply_row_rotation(R, static_cast<float>(xo), static_cast<float>(yo), xs, ys);

                if (it == kFixedPointIters - 2) v_after_second_iter = ys;
                v_guess = ys;
            }

            const int idx = yo * W + xo;
            iter_delta[idx] = std::fabs(ys - v_after_second_iter);

            int valid = 0;
            const float sample = bilinear_sample_gray(rs_frame, W, H, xs, ys, valid);
            corrected[idx] = valid ? static_cast<unsigned char>(sample + 0.5f) : static_cast<unsigned char>(0);
            valid_mask[idx] = static_cast<unsigned char>(valid);
        }
    }
}
