// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.17
//                     (LiDAR-camera projection/coloring fusion kernels)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): it is the CORRECTNESS ORACLE
// main.cu's VERIFY stage compares the GPU path against, and it is the
// TEACHING BASELINE that makes the GPU version legible as a transformation
// of something simple.
//
// Independence ruling applied in THIS file (docs/PROJECT_TEMPLATE's
// reference_cpu.cpp carries the full ruling text; this is the choice this
// project makes, following 01.18's precedent exactly):
//   * Data-layout contracts (Rigid3, LidarPointF, every constant) are
//     single-sourced in kernels.cuh and shared — divergent layouts would be
//     a bug class of their own, not independence.
//   * The ALGORITHMIC CORE of all four kernels — the rigid transform +
//     pinhole projection, the z-buffer's nearest-wins compare, bilinear
//     sampling, the occlusion depth-consistency test — is written HERE,
//     completely independently from kernels.cu, in the simplest possible
//     sequential C++. None of it is shared as a __host__ __device__ helper:
//     every one of these formulas is short enough (4-20 lines) that
//     duplicating it is not "pure transcription" of something too complex to
//     write twice — it is exactly the kind of formula this repo's ruling
//     says SHOULD be independent, so main.cu's VERIFY stage comparison can
//     catch a real bug (a sign error, a swapped axis, an off-by-one tap)
//     instead of comparing one formula to itself under a different compiler.
//   * On top of the twin comparisons, this project ALSO carries gates that
//     do not route through either implementation at all: coloring_accuracy,
//     occlusion_correctness, and the sensitivity-curve analytic consistency
//     check all compare against scripts/make_synthetic.py's independent
//     ground truth (true_r/g/b, visible, and the 01.17-derived pixel-
//     displacement formula) — none of which lives in this file or in
//     kernels.cu (main.cu's own "independent gate" section, cited there).
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong, and
// then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side; every
// function below has a same-named counterpart there.
// ===========================================================================

#include "kernels.cuh"   // shared model constants, layouts, signatures

#include <cmath>         // std::floor, std::fabs

// ---------------------------------------------------------------------------
// project_point_cpu — independent host twin of kernels.cu's
// project_point_device: rigid transform + pinhole projection. Written from
// scratch here (not shared) per the independence ruling above.
// ---------------------------------------------------------------------------
static bool project_point_cpu(const LidarPointF& p, const Rigid3& T,
                              float& u, float& v, float& zc, int& px_out, int& py_out)
{
    const float* R = T.R;
    const float xc = R[0] * p.x + R[1] * p.y + R[2] * p.z + T.t[0];
    const float yc = R[3] * p.x + R[4] * p.y + R[5] * p.z + T.t[1];
    zc              = R[6] * p.x + R[7] * p.y + R[8] * p.z + T.t[2];

    if (zc <= 0.0f) { u = 0.0f; v = 0.0f; px_out = -1; py_out = -1; return false; }

    const float inv_z = 1.0f / zc;
    u = kFx * xc * inv_z + kCx;
    v = kFy * yc * inv_z + kCy;

    const int px = static_cast<int>(std::floor(u + 0.5f));
    const int py = static_cast<int>(std::floor(v + 0.5f));
    px_out = px;
    py_out = py;

    if (zc > kMaxDepthM) return false;
    if (px < 0 || px >= kImageWidth || py < 0 || py >= kImageHeight) return false;
    return true;
}

// ---------------------------------------------------------------------------
// project_zbuffer_cpu — sequential nearest-wins z-buffer. Independent from
// project_zbuffer_kernel in the sense the file header promises: this is a
// plain "keep the smaller depth" compare, run one point at a time in
// INPUT order — no atomics, because a single thread never races with
// itself (the GPU's atomicMin/encode trick exists only because MANY threads
// race on the same pixel; this loop is the reason that trick exists, made
// visible — depths here are compared directly as floats, no bit-encoding at
// all, 01.18's precedent).
//
// Complexity: O(n_pts). out_depth is (re)initialized to kInvalidDepth here
// so callers never need a separate "clear" step.
// ---------------------------------------------------------------------------
void project_zbuffer_cpu(const LidarPointF* pts, int n_pts, Rigid3 T, float* out_depth)
{
    for (int i = 0; i < kImagePixels; ++i) out_depth[i] = kInvalidDepth;

    for (int i = 0; i < n_pts; ++i) {
        float u, v, zc; int px, py;
        if (!project_point_cpu(pts[i], T, u, v, zc, px, py)) continue;
        const int idx = py * kImageWidth + px;
        if (out_depth[idx] == kInvalidDepth || zc < out_depth[idx]) out_depth[idx] = zc;
    }
}

// ---------------------------------------------------------------------------
// project_points_cpu — independent host twin of project_points_kernel.
// ---------------------------------------------------------------------------
void project_points_cpu(const LidarPointF* pts, int n_pts, Rigid3 T,
                        float* u, float* v, float* zc, uint8_t* in_frustum)
{
    for (int i = 0; i < n_pts; ++i) {
        int px, py;
        const bool ok = project_point_cpu(pts[i], T, u[i], v[i], zc[i], px, py);
        in_frustum[i] = ok ? 1u : 0u;
    }
}

// ---------------------------------------------------------------------------
// sample_bilinear_cpu — independent host twin of kernels.cu's
// bilinear_sample_device/sample_bilinear_kernel. Same clamp-to-edge/four-tap
// blend, written from scratch.
// ---------------------------------------------------------------------------
void sample_bilinear_cpu(const float* u, const float* v, const uint8_t* in_frustum, int n_pts,
                         const float* rgb, float* color)
{
    for (int i = 0; i < n_pts; ++i) {
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (in_frustum[i]) {
            float uc = u[i] < 0.0f ? 0.0f : (u[i] > static_cast<float>(kImageWidth - 1) ? static_cast<float>(kImageWidth - 1) : u[i]);
            float vc = v[i] < 0.0f ? 0.0f : (v[i] > static_cast<float>(kImageHeight - 1) ? static_cast<float>(kImageHeight - 1) : v[i]);

            const int x0 = static_cast<int>(std::floor(uc));
            const int y0 = static_cast<int>(std::floor(vc));
            const int x1 = x0 + 1 < kImageWidth  ? x0 + 1 : x0;
            const int y1 = y0 + 1 < kImageHeight ? y0 + 1 : y0;
            const float tx = uc - static_cast<float>(x0);
            const float ty = vc - static_cast<float>(y0);

            const int i00 = y0 * kImageWidth + x0, i10 = y0 * kImageWidth + x1;
            const int i01 = y1 * kImageWidth + x0, i11 = y1 * kImageWidth + x1;

            float chans[3];
            for (int c = 0; c < 3; ++c) {
                const float* plane = rgb + c * kImagePixels;
                const float top = plane[i00] * (1.0f - tx) + plane[i10] * tx;
                const float bot = plane[i01] * (1.0f - tx) + plane[i11] * tx;
                chans[c] = top * (1.0f - ty) + bot * ty;
            }
            r = chans[0]; g = chans[1]; b = chans[2];
        }
        color[3 * i + 0] = r;
        color[3 * i + 1] = g;
        color[3 * i + 2] = b;
    }
}

// ---------------------------------------------------------------------------
// check_occlusion_cpu — independent host twin of check_occlusion_kernel.
// Takes the PLAIN float depth map project_zbuffer_cpu produces (not an
// encoded uint32 array) — this keeps the CPU path entirely free of the
// atomicMin bit-encoding detail, which exists ONLY to make the GPU's race
// safe (01.18's convention: the CPU twin never needs the trick that motivated
// it). Because decode(encode(x)) == x exactly for the positive floats this
// project ever stores, the GPU kernel's decoded nearest-in-window value and
// this function's plain float nearest-in-window value are the SAME number
// whenever the underlying z-buffer passes agree — which the VERIFY stage
// checks independently, upstream of this function ever running. Scans the
// SAME (2*kOcclusionWindowRadiusPx+1) square window kernels.cu's kernel 4
// does (kernels.cuh's file comment explains why a single exact pixel is not
// enough), written independently here as a plain nested loop.
// ---------------------------------------------------------------------------
void check_occlusion_cpu(const float* u, const float* v, const float* zc, const uint8_t* in_frustum,
                         int n_pts, const float* depth, float band_m, uint8_t* visible)
{
    for (int i = 0; i < n_pts; ++i) {
        if (!in_frustum[i]) { visible[i] = 0u; continue; }
        const int px = static_cast<int>(std::floor(u[i] + 0.5f));
        const int py = static_cast<int>(std::floor(v[i] + 0.5f));

        bool found = false;
        float nearest = 0.0f;
        for (int dy = -kOcclusionWindowRadiusPx; dy <= kOcclusionWindowRadiusPx; ++dy) {
            const int ny = py + dy;
            if (ny < 0 || ny >= kImageHeight) continue;
            for (int dx = -kOcclusionWindowRadiusPx; dx <= kOcclusionWindowRadiusPx; ++dx) {
                const int nx = px + dx;
                if (nx < 0 || nx >= kImageWidth) continue;
                const float d = depth[ny * kImageWidth + nx];
                if (d == kInvalidDepth) continue;
                if (!found || d < nearest) { nearest = d; found = true; }
            }
        }
        if (!found) { visible[i] = 0u; continue; }   // no z-buffer evidence anywhere nearby -- see kernels.cu
        visible[i] = (std::fabs(zc[i] - nearest) <= band_m) ? 1u : 0u;
    }
}
