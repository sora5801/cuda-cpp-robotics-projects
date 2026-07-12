// ===========================================================================
// kernels.cu — GPU kernels for project 02.17
//              LiDAR-camera projection/coloring fusion kernels
//
// Four kernels, one shared geometric core, two teaching directions
// (kernels.cuh's file header names them "Direction A"/"Direction B" — this
// file's job is the WHY behind each kernel's own small GPU pattern):
//
//   1. project_zbuffer_kernel — a SCATTER: one thread per INPUT point (not
//      per output pixel). Threads race on shared output pixels; atomicMin
//      resolves the race (01.18's trick, cited, reused verbatim).
//   2. project_points_kernel  — a MAP: one thread per point, pure geometry,
//      no shared state at all — every other kernel/the sensitivity sweep
//      builds on this one's output.
//   3. sample_bilinear_kernel — a MAP: one thread per point, four
//      neighboring-pixel reads (a tiny, four-tap gather, the 01.01 lineage
//      cited in kernels.cuh).
//   4. check_occlusion_kernel — a MAP: one thread per point, one pixel read
//      from kernel 1's z-buffer output — the cheapest kernel here, and the
//      one that turns "coloring" into "fusion" (README/THEORY.md).
//
// All four are map/scatter-of-independent-points patterns: no shared memory,
// no cross-thread communication beyond kernel 1's atomics, no divergence
// beyond the tail guard and each kernel's own boundary branches — the
// pattern 08.01/09.01/33.01 establish for "one thread owns one independent
// unit of work" and 01.18 specializes to LiDAR-camera projection first.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>
#include <cstdint>

// ---------------------------------------------------------------------------
// encode_depth_for_zbuffer / decode_depth_from_zbuffer — order-preserving
// float<->uint32 map so atomicMin (integer-only on this hardware) can race
// many threads onto one pixel and keep the SMALLEST depth. Reused verbatim
// from 01.18's derivation (cited in kernels.cuh's file header): for any two
// POSITIVE, finite floats, the raw IEEE-754 bit pattern preserves numeric
// ordering (the exponent occupies the high bits and dominates the
// comparison) — no transformation needed. Every depth this project encodes
// is a LiDAR return in front of the camera (zc > 0 is checked before this is
// ever called), so the simple positive-only form is what actually runs; the
// fully general (possibly-negative) encoding is documented, not needed, in
// 01.18's kernels.cu, cited rather than repeated here.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t encode_depth_for_zbuffer(float depth_m)
{
    return __float_as_uint(depth_m);
}

__device__ __forceinline__ float decode_depth_from_zbuffer(uint32_t bits)
{
    return __uint_as_float(bits);
}

// ---------------------------------------------------------------------------
// project_point_device — the shared geometric core, DEVICE side: rigid
// transform LiDAR->camera (P_cam = R*P_lidar + t, kernels.cuh's Rigid3
// convention) then pinhole projection. Not shared as a __host__ __device__
// helper with reference_cpu.cpp on purpose (kernels.cuh/reference_cpu.cpp's
// twin-independence ruling: this ~10-line formula is exactly the kind of
// thing that SHOULD be written twice so the VERIFY stage's comparison means
// something) — reference_cpu.cpp writes its own copy independently.
//
// Outputs u,v as CONTINUOUS pixel coordinates (not rounded — the sensitivity
// sweep needs sub-pixel precision) and zc as the raw camera-frame depth
// (Pcam.z, the pinhole/z-buffer convention — never Euclidean range, 01.18's
// convention). Returns true iff zc is in (0, kMaxDepthM] AND the ROUNDED
// pixel (round-half-up, floor(x+0.5) — the SAME convention 01.18's z-buffer
// uses) lands inside the image; px_out/py_out receive that rounded pixel
// (valid only when the function returns true — callers that need them
// unconditionally, e.g. project_zbuffer_kernel's caller-side rounding,
// recompute directly rather than trusting an out-param on a false return).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool project_point_device(const LidarPointF& p, const Rigid3& T,
                                                      float& u, float& v, float& zc,
                                                      int& px_out, int& py_out)
{
    const float* R = T.R;
    const float xc = R[0] * p.x + R[1] * p.y + R[2] * p.z + T.t[0];
    const float yc = R[3] * p.x + R[4] * p.y + R[5] * p.z + T.t[1];
    zc              = R[6] * p.x + R[7] * p.y + R[8] * p.z + T.t[2];

    // u,v are meaningful (as CONTINUOUS coordinates, for the sensitivity
    // sweep's sub-pixel displacement measurement) even when zc is invalid —
    // but dividing by a non-positive zc would be nonsense, so guard first
    // and leave u,v at a defined (if unused) value in that case.
    if (zc <= 0.0f) { u = 0.0f; v = 0.0f; px_out = -1; py_out = -1; return false; }

    const float inv_z = 1.0f / zc;
    u = kFx * xc * inv_z + kCx;
    v = kFy * yc * inv_z + kCy;

    const int px = static_cast<int>(floorf(u + 0.5f));
    const int py = static_cast<int>(floorf(v + 0.5f));
    px_out = px;
    py_out = py;

    if (zc > kMaxDepthM) return false;
    if (px < 0 || px >= kImageWidth || py < 0 || py >= kImageHeight) return false;
    return true;
}

// ===========================================================================
// 1) PROJECTION + Z-BUFFER (Direction B's product; Direction A's occlusion
// oracle) — see kernels.cuh's file header and 01.18's kernels.cu (cited) for
// the full scatter/atomicMin discussion; this project's version differs only
// in taking T as a runtime PARAMETER (not a hardcoded kTCameraLidar) so the
// sensitivity sweep and the "occlusion, unfixed" baseline can reuse it at
// perturbed or repeated calls without a second kernel.
// ---------------------------------------------------------------------------
__global__ void project_zbuffer_kernel(const LidarPointF* __restrict__ d_pts,
                                       int n_pts,
                                       Rigid3 T,
                                       uint32_t* __restrict__ d_encoded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's LiDAR point index
    if (i >= n_pts) return;                                 // ragged-tail guard

    float u, v, zc; int px, py;
    if (!project_point_device(d_pts[i], T, u, v, zc, px, py)) return;   // nothing to write: out of range/frame

    const int idx = py * kImageWidth + px;
    // atomicMin on the ENCODED depth: many threads may race here (adjacent
    // beams near a silhouette edge routinely land on the same pixel); the
    // hardware's true read-modify-write guarantees the pixel ends up holding
    // exactly the smallest depth ANY thread offered, regardless of
    // scheduling order (01.18's derivation, cited).
    atomicMin(&d_encoded[idx], encode_depth_for_zbuffer(zc));
}

void launch_project_zbuffer(const LidarPointF* d_pts, int n_pts, Rigid3 T, uint32_t* d_encoded)
{
    const int threads = 256;                          // warp multiple, repo default (08.01/33.01)
    const int blocks = (n_pts + threads - 1) / threads;
    project_zbuffer_kernel<<<blocks, threads>>>(d_pts, n_pts, T, d_encoded);
    CUDA_CHECK_LAST_ERROR("project_zbuffer_kernel launch");
}

// ===========================================================================
// 2) PROJECT POINTS — the shared geometric core, no z-buffer, no color: a
// pure MAP, one thread per point, writing its own (u,v,zc,in_frustum) and
// touching no other thread's data at all. Every other kernel and the
// calibration-error sensitivity sweep (main.cu) builds on this one's output
// — keeping the raw projection as its OWN kernel (rather than folding it
// into kernel 1 or 3) is what lets the sweep re-run "just the geometry" at a
// perturbed T without paying for a z-buffer pass it does not need.
// ---------------------------------------------------------------------------
__global__ void project_points_kernel(const LidarPointF* __restrict__ d_pts,
                                      int n_pts,
                                      Rigid3 T,
                                      float* __restrict__ d_u,
                                      float* __restrict__ d_v,
                                      float* __restrict__ d_zc,
                                      uint8_t* __restrict__ d_in_frustum)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pts) return;

    float u, v, zc; int px, py;
    const bool in_frustum = project_point_device(d_pts[i], T, u, v, zc, px, py);

    d_u[i] = u;
    d_v[i] = v;
    d_zc[i] = zc;
    d_in_frustum[i] = in_frustum ? 1u : 0u;
}

void launch_project_points(const LidarPointF* d_pts, int n_pts, Rigid3 T,
                           float* d_u, float* d_v, float* d_zc, uint8_t* d_in_frustum)
{
    const int threads = 256;
    const int blocks = (n_pts + threads - 1) / threads;
    project_points_kernel<<<blocks, threads>>>(d_pts, n_pts, T, d_u, d_v, d_zc, d_in_frustum);
    CUDA_CHECK_LAST_ERROR("project_points_kernel launch");
}

// ===========================================================================
// 3) BILINEAR COLOR SAMPLING (Direction A, the NAIVE path — no occlusion
// awareness at all: "whatever pixel I land on is my color", the failure mode
// this project measures before fixing it with kernel 4).
//
// Bilinear, not nearest-neighbor, because a LiDAR point's projected (u,v) is
// almost never exactly on a pixel CENTER; nearest-neighbor rounding would
// silently discard sub-pixel information the pinhole model actually computed
// (01.01's lineage, kernels.cuh's file header) — and, as README/THEORY's
// "edge bleeding" honesty gate shows, it is bilinear's OWN blending across a
// true color edge that produces this project's most interesting failure at
// object silhouettes, distinct from the OCCLUSION failure kernel 4 fixes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void bilinear_sample_device(const float* __restrict__ rgb, float u, float v,
                                                        float& r, float& g, float& b)
{
    // Clamp-to-edge: a point whose projection lands fractionally outside the
    // last row/column of pixels (a routine sub-pixel rounding case near the
    // image border, not a bug) samples the border pixel repeated rather than
    // reading out of bounds.
    float uc = u < 0.0f ? 0.0f : (u > static_cast<float>(kImageWidth - 1) ? static_cast<float>(kImageWidth - 1) : u);
    float vc = v < 0.0f ? 0.0f : (v > static_cast<float>(kImageHeight - 1) ? static_cast<float>(kImageHeight - 1) : v);

    const int x0 = static_cast<int>(floorf(uc));
    const int y0 = static_cast<int>(floorf(vc));
    const int x1 = x0 + 1 < kImageWidth  ? x0 + 1 : x0;   // clamp the far tap at the last column/row too
    const int y1 = y0 + 1 < kImageHeight ? y0 + 1 : y0;
    const float tx = uc - static_cast<float>(x0);
    const float ty = vc - static_cast<float>(y0);

    const int i00 = y0 * kImageWidth + x0, i10 = y0 * kImageWidth + x1;
    const int i01 = y1 * kImageWidth + x0, i11 = y1 * kImageWidth + x1;

    // rgb is PLANAR: plane 0 = red [0,N), plane 1 = green [N,2N), plane 2 =
    // blue [2N,3N) (01.18's layout, cited) — each channel's four-tap bilinear
    // blend is independent, done once per channel below.
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float* plane = rgb + c * kImagePixels;
        const float top = plane[i00] * (1.0f - tx) + plane[i10] * tx;
        const float bot = plane[i01] * (1.0f - tx) + plane[i11] * tx;
        const float val = top * (1.0f - ty) + bot * ty;
        if (c == 0) r = val; else if (c == 1) g = val; else b = val;
    }
}

__global__ void sample_bilinear_kernel(const float* __restrict__ d_u,
                                       const float* __restrict__ d_v,
                                       const uint8_t* __restrict__ d_in_frustum,
                                       int n_pts,
                                       const float* __restrict__ d_rgb,
                                       float* __restrict__ d_color)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pts) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;   // documented "no color" default (see kernels.cuh)
    if (d_in_frustum[i]) bilinear_sample_device(d_rgb, d_u[i], d_v[i], r, g, b);

    d_color[3 * i + 0] = r;
    d_color[3 * i + 1] = g;
    d_color[3 * i + 2] = b;
}

void launch_sample_bilinear(const float* d_u, const float* d_v, const uint8_t* d_in_frustum, int n_pts,
                            const float* d_rgb, float* d_color)
{
    const int threads = 256;
    const int blocks = (n_pts + threads - 1) / threads;
    sample_bilinear_kernel<<<blocks, threads>>>(d_u, d_v, d_in_frustum, n_pts, d_rgb, d_color);
    CUDA_CHECK_LAST_ERROR("sample_bilinear_kernel launch");
}

// ===========================================================================
// 4) OCCLUSION CHECK (Direction A, the FIX). One thread per point: scan a
// small NEIGHBORHOOD of kernel 1's z-buffer around this point's own pixel
// (the same T this point was itself projected with — main.cu's contract,
// kernels.cuh) for the nearest evidence found anywhere nearby, and accept
// the point as "visible" only if its own depth is close to that nearest
// evidence. Cheapest-per-tap kernel in this project (a handful of global
// reads + decodes + a running min) — and the one that turns naive coloring
// into an honest fusion product.
//
// WHY A WINDOW, NOT JUST THE EXACT PIXEL (measured, not assumed — see
// kernels.cuh's kOcclusionWindowRadiusPx comment for the numbers): this
// project's LiDAR scan is angularly sparse enough that adjacent returns
// land several pixels apart. A background point hidden behind a foreground
// occluder (this scene's designed cohort) very often has NO occluder return
// on its OWN exact discretized pixel — even though the occluder plainly
// covers that pixel in the dense camera image — so an exact-pixel-only
// check finds no competing evidence and wrongly waves the hidden point
// through. Widening the search to the occluder's local neighborhood finds
// that evidence without needing every pixel individually painted.
// ---------------------------------------------------------------------------
__global__ void check_occlusion_kernel(const float* __restrict__ d_u,
                                       const float* __restrict__ d_v,
                                       const float* __restrict__ d_zc,
                                       const uint8_t* __restrict__ d_in_frustum,
                                       int n_pts,
                                       const uint32_t* __restrict__ d_encoded,
                                       float band_m,
                                       uint8_t* __restrict__ d_visible)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pts) return;

    if (!d_in_frustum[i]) { d_visible[i] = 0u; return; }

    // Recompute the SAME rounded pixel project_zbuffer_kernel used (the
    // identical floor(x+0.5) rule) — d_u/d_v are the continuous coordinates
    // kernel 2 already computed, so this is one floor() each, not a second
    // full projection.
    const int px = static_cast<int>(floorf(d_u[i] + 0.5f));
    const int py = static_cast<int>(floorf(d_v[i] + 0.5f));

    // Scan the (2R+1)x(2R+1) window for the smallest decoded depth found —
    // R is tiny (kernels.cuh: 2, a 5x5 = 25-tap window), so an unrolled
    // sequential scan (no shared memory, each thread's window overlaps its
    // neighbors' but at this problem size that redundant re-reading is not
    // worth a tiled optimization — CLAUDE.md's "teaching beats cleverness")
    // is the natural per-thread mapping.
    bool found = false;
    float nearest = 0.0f;
    for (int dy = -kOcclusionWindowRadiusPx; dy <= kOcclusionWindowRadiusPx; ++dy) {
        const int ny = py + dy;
        if (ny < 0 || ny >= kImageHeight) continue;
        for (int dx = -kOcclusionWindowRadiusPx; dx <= kOcclusionWindowRadiusPx; ++dx) {
            const int nx = px + dx;
            if (nx < 0 || nx >= kImageWidth) continue;
            const uint32_t bits = d_encoded[ny * kImageWidth + nx];
            if (bits == 0xFFFFFFFFu) continue;   // 0xFFFFFFFF: no LiDAR point landed on THIS window cell
            const float d = decode_depth_from_zbuffer(bits);
            if (!found || d < nearest) { nearest = d; found = true; }
        }
    }

    // An entirely empty window (no LiDAR point anywhere nearby under THIS
    // z-buffer pass) cannot CONFIRM this point is the nearest surface there
    // — and "assume visible when unsure" is exactly the bug this kernel
    // exists to prevent, so the conservative, documented answer is NOT
    // VISIBLE, not a silent pass.
    if (!found) { d_visible[i] = 0u; return; }

    d_visible[i] = (fabsf(d_zc[i] - nearest) <= band_m) ? 1u : 0u;
}

void launch_check_occlusion(const float* d_u, const float* d_v, const float* d_zc,
                            const uint8_t* d_in_frustum, int n_pts,
                            const uint32_t* d_encoded, float band_m, uint8_t* d_visible)
{
    const int threads = 256;
    const int blocks = (n_pts + threads - 1) / threads;
    check_occlusion_kernel<<<blocks, threads>>>(d_u, d_v, d_zc, d_in_frustum, n_pts, d_encoded, band_m, d_visible);
    CUDA_CHECK_LAST_ERROR("check_occlusion_kernel launch");
}
