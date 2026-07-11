// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.07 (Fisheye/
//                     omnidirectional unwarping and multi-camera
//                     surround-view stitching)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), unchanged from every other
// project in this repository:
//
//   1) It is the CORRECTNESS ORACLE. GPU code fails in ways CPU code
//      cannot: wrong 2-D thread indexing (swap x/y and the picture still
//      "looks plausible" until you diff it), race conditions, stale device
//      memory, bad transfers. A dead-simple sequential version a reader
//      can verify BY EYE gives us ground truth; main.cu runs both and
//      asserts element-wise agreement within a documented tolerance.
//   2) It is the TEACHING BASELINE. Reading this file, then kernels.cu,
//      shows exactly what parallelization changed: every "for each output
//      pixel" double loop below became "each thread owns one pixel" there,
//      and the BEV compositor's "for each of 4 cameras" loop became the
//      SAME loop, just running once per GPU thread instead of once per CPU
//      iteration of the outer pixel loop.
//
// Independence ruling applied to THIS file (the template's header states
// the general rule; kernels.cuh's PART 1/PART 3 headers state exactly
// which functions fall under the shared-data exception; here is exactly
// how it plays out in code):
//   * SHARED (kernels.cuh): fisheye_project(), fisheye_unproject(),
//     pinhole_unproject_rect(), cyl_unproject(), rig_camera_to_bev_sample()
//     — the camera-model and rig-geometry formulas. These are DATA (the
//     physical lens and the physical rig), not "the algorithm under test"
//     (cf. 01.01's compute_source_pixel precedent). main.cu's "model
//     roundtrip" gate is the required INDEPENDENT check that does NOT
//     route through fisheye_project/unproject (it re-derives the
//     equidistant model from scratch, in double precision); main.cu's
//     "seam consistency" gate independently re-samples two cameras' actual
//     images at rig-consistent points using its OWN hand-written bilinear
//     sampler, exercising the rig geometry end-to-end from a third angle.
//   * INDEPENDENT (this file): the bilinear sampling, the LUT-build loop
//     structure, and the ENTIRE BEV multi-camera blend + coverage-bitmask
//     accumulation are all typed a SECOND time below, from scratch,
//     deliberately not calling anything in kernels.cu. Any GPU-vs-CPU
//     mismatch main.cu reports is therefore a real bug in one of the two
//     independent implementations, not a shared blind spot.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // RemapSample, camera-model constants + shared helpers, launcher signatures

#include <cmath>         // std::floor

// ---------------------------------------------------------------------------
// clampi_cpu — host-side twin of kernels.cu's __device__ clampi(). Two
// lines; independently re-typed on purpose (see file header).
// ---------------------------------------------------------------------------
static inline int clampi_cpu(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// ---------------------------------------------------------------------------
// bilinear_sample_rgb_cpu — INDEPENDENT host re-typing of kernels.cu's
// __device__ bilinear_sample_rgb() (file header: bilinear sampling is
// deliberately duplicated, not shared, so the twin comparison actually
// exercises the interpolation arithmetic). Same clamp-to-edge policy.
// ---------------------------------------------------------------------------
static void bilinear_sample_rgb_cpu(const unsigned char* img, int W, int H,
                                    float u, float v, unsigned char out[3])
{
    if (u < 0.0f) u = 0.0f;
    if (u > static_cast<float>(W - 1)) u = static_cast<float>(W - 1);
    if (v < 0.0f) v = 0.0f;
    if (v > static_cast<float>(H - 1)) v = static_cast<float>(H - 1);

    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const int x1 = clampi_cpu(x0 + 1, 0, W - 1);
    const int y1 = clampi_cpu(y0 + 1, 0, H - 1);
    const float fx = u - static_cast<float>(x0);
    const float fy = v - static_cast<float>(y0);

    for (int c = 0; c < 3; ++c) {
        const float v00 = static_cast<float>(img[(y0 * W + x0) * 3 + c]);
        const float v10 = static_cast<float>(img[(y0 * W + x1) * 3 + c]);
        const float v01 = static_cast<float>(img[(y1 * W + x0) * 3 + c]);
        const float v11 = static_cast<float>(img[(y1 * W + x1) * 3 + c]);
        const float top = v00 + (v10 - v00) * fx;
        const float bot = v01 + (v11 - v01) * fx;
        const float val = top + (bot - top) * fy;
        out[c] = static_cast<unsigned char>(val + 0.5f);
    }
}

// ===========================================================================
// build_rect_lut_cpu / build_cyl_lut_cpu — sequential twins of
// build_rect_lut_kernel / build_cyl_lut_kernel. Nested loops instead of a
// 2-D thread grid; each iteration calls the SHARED unproject + project
// helpers (kernels.cuh), same as the GPU kernels do.
// ===========================================================================
void build_rect_lut_cpu(RemapSample* lut)
{
    for (int yo = 0; yo < kRectH; ++yo) {
        for (int xo = 0; xo < kRectW; ++xo) {
            float X, Y, Z;
            pinhole_unproject_rect(xo, yo, X, Y, Z);
            lut[yo * kRectW + xo] = fisheye_project(X, Y, Z);
        }
    }
}

void build_cyl_lut_cpu(RemapSample* lut)
{
    for (int yo = 0; yo < kCylH; ++yo) {
        for (int xo = 0; xo < kCylW; ++xo) {
            float X, Y, Z;
            cyl_unproject(xo, yo, X, Y, Z);
            lut[yo * kCylW + xo] = fisheye_project(X, Y, Z);
        }
    }
}

// ===========================================================================
// remap_bilinear_cpu — sequential twin of remap_bilinear_kernel: for every
// output pixel, look up its LUT entry and bilinear-sample the fisheye
// source. Reused for both the rectilinear and cylindrical outputs by
// main.cu (same function, different lut/dims), mirroring the GPU side.
// ===========================================================================
void remap_bilinear_cpu(const unsigned char* src, const RemapSample* lut,
                        unsigned char* out, int srcW, int srcH, int outW, int outH)
{
    for (int yo = 0; yo < outH; ++yo) {
        for (int xo = 0; xo < outW; ++xo) {
            const int idx = yo * outW + xo;
            const RemapSample s = lut[idx];
            unsigned char rgb[3];
            bilinear_sample_rgb_cpu(src, srcW, srcH, s.u, s.v, rgb);
            out[idx * 3 + 0] = rgb[0];
            out[idx * 3 + 1] = rgb[1];
            out[idx * 3 + 2] = rgb[2];
        }
    }
}

// ===========================================================================
// bev_compose_cpu — sequential twin of bev_compose_kernel: for every BEV
// output pixel, walk the SAME 4-camera loop the GPU kernel runs per
// thread, calling the shared rig_camera_to_bev_sample() (kernels.cuh) and
// this file's OWN independent bilinear sampler, accumulating the SAME
// weighted blend + coverage bitmask, entirely independently of kernels.cu.
// ===========================================================================
void bev_compose_cpu(const unsigned char* front, const unsigned char* left,
                     const unsigned char* right, const unsigned char* rear,
                     unsigned char* bev, unsigned char* coverage)
{
    const unsigned char* const imgs[kNumRigCameras] = { front, left, right, rear };

    for (int yo = 0; yo < kBevH; ++yo) {
        for (int xo = 0; xo < kBevW; ++xo) {
            float X, Y;
            bev_pixel_to_ground(xo, yo, X, Y);

            float sum[3] = { 0.0f, 0.0f, 0.0f };
            float wsum = 0.0f;
            unsigned char cov = 0;

            for (int cam = 0; cam < kNumRigCameras; ++cam) {
                float u, v, weight;
                if (!rig_camera_to_bev_sample(cam, X, Y, u, v, weight)) continue;

                unsigned char rgb[3];
                bilinear_sample_rgb_cpu(imgs[cam], kFishW, kFishH, u, v, rgb);
                sum[0] += weight * static_cast<float>(rgb[0]);
                sum[1] += weight * static_cast<float>(rgb[1]);
                sum[2] += weight * static_cast<float>(rgb[2]);
                wsum += weight;
                cov = static_cast<unsigned char>(cov | (1u << cam));
            }

            const int idx = yo * kBevW + xo;
            if (wsum > 0.0f) {
                const float inv = 1.0f / wsum;
                bev[idx * 3 + 0] = static_cast<unsigned char>(sum[0] * inv + 0.5f);
                bev[idx * 3 + 1] = static_cast<unsigned char>(sum[1] * inv + 0.5f);
                bev[idx * 3 + 2] = static_cast<unsigned char>(sum[2] * inv + 0.5f);
            } else {
                bev[idx * 3 + 0] = 0;
                bev[idx * 3 + 1] = 0;
                bev[idx * 3 + 2] = 0;
            }
            coverage[idx] = cov;
        }
    }
}
