// ===========================================================================
// kernels.cu — GPU kernels for project 01.21 (Scene flow from RGB-D pairs)
//
// Big idea (the whole project in one paragraph)
// -----------------------------------------------
// Every kernel in Milestones 1-2-4 is a MAP or small STENCIL: one thread per
// pixel, entirely independent of every other pixel (the flow pyramid's
// coarse-to-fine LEVEL loop is the one place work is sequential — see
// run_pyramidal_lk_gpu, same shape as 01.03, cited). Milestone 3 introduces
// this project's one genuinely NEW GPU idea beyond thread-per-pixel: a
// REDUCTION (weighted_covariance_reduce_kernel) that turns kPixels
// independent per-point contributions into ONE small (16-scalar) record via
// a block-level shared-memory tree, then main.cu finishes the sum across
// blocks on the host (the exact "GPU partial reduce, host finishes it" split
// 02.06's ICP normal-equation assembly and 01.17's calibration use, cited).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (an
// INDEPENDENT re-implementation of every algorithmic core below).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry shared by every "one thread per pixel" kernel: a 16x16
// 2-D block (256 threads, a warp multiple — 01.03/01.04's repo-wide
// default), grid sized to cover the caller-supplied W x H with a ragged-
// tail guard inside each kernel body.
// ---------------------------------------------------------------------------
static constexpr int kBlock2D = 16;

static inline dim3 grid2d(int W, int H)
{
    return dim3((W + kBlock2D - 1) / kBlock2D, (H + kBlock2D - 1) / kBlock2D);
}

static inline int grid1d(int n, int block)
{
    return (n + block - 1) / block;
}

// ---------------------------------------------------------------------------
// bilinear_sample_u8 — sample a uint8 grayscale image at a fractional
// coordinate, clamp-to-edge outside the image. Identical role and clamping
// strategy to 01.03's bilinear_sample_u8 (cited); written fresh here rather
// than shared (01.03's header explains why: bilinear sampling is exactly
// the kind of small, easy-to-get-subtly-wrong arithmetic where an
// independent twin is worth more than a shared helper).
// ---------------------------------------------------------------------------
__device__ inline float bilinear_sample_u8(const uint8_t* __restrict__ img, int W, int H, float x, float y)
{
    x = fminf(fmaxf(x, 0.0f), static_cast<float>(W - 1));
    y = fminf(fmaxf(y, 0.0f), static_cast<float>(H - 1));
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float i00 = static_cast<float>(img[y0 * W + x0]);
    const float i10 = static_cast<float>(img[y0 * W + x1]);
    const float i01 = static_cast<float>(img[y1 * W + x0]);
    const float i11 = static_cast<float>(img[y1 * W + x1]);
    const float top = i00 + (i10 - i00) * fx;
    const float bot = i01 + (i11 - i01) * fx;
    return top + (bot - top) * fy;
}

// ===========================================================================
// MILESTONE 1 — 2-level pyramidal Lucas-Kanade. Every kernel below is a
// compact, independently-typed re-implementation of 01.03's Milestone 1
// (cited in each header); the derivations are not repeated here in full —
// see 01.03's kernels.cu and this project's own THEORY.md "The algorithm".
// ===========================================================================

// downsample_area2x_kernel — exact 2x area-average box filter (01.03's
// anti-aliasing argument, cited): the correct decimation filter because its
// support exactly matches the 2x ratio, so every input pixel contributes to
// exactly one output pixel with equal weight (no aliasing of this project's
// hashed/textured surfaces into the coarse level).
__global__ void downsample_area2x_kernel(const uint8_t* __restrict__ in, int inW, int inH,
                                         uint8_t* __restrict__ out)
{
    const int outW = inW / 2, outH = inH / 2;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;
    const int ix = ox * 2, iy = oy * 2;
    const int sum = static_cast<int>(in[iy * inW + ix]) + static_cast<int>(in[iy * inW + ix + 1]) +
                    static_cast<int>(in[(iy + 1) * inW + ix]) + static_cast<int>(in[(iy + 1) * inW + ix + 1]);
    out[oy * outW + ox] = static_cast<uint8_t>((sum + 2) / 4);   // integer round-to-nearest, bit-exact GPU/CPU
}

void launch_downsample_area2x(const uint8_t* d_in, int inW, int inH, uint8_t* d_out)
{
    downsample_area2x_kernel<<<grid2d(inW / 2, inH / 2), dim3(kBlock2D, kBlock2D)>>>(d_in, inW, inH, d_out);
    CUDA_CHECK_LAST_ERROR("downsample_area2x_kernel launch");
}

// scharr_gradient_kernel — per-pixel 3x3 Scharr Gx,Gy, normalized by the
// exact power-of-two 1/32 (01.03's derivation of why unnormalized gradients
// silently wreck the LK step size, cited — the normal equations are
// quadratic in the gradient while the mismatch vector is linear in it, so
// an unnormalized k=32 gradient shrinks every solved step by 32x).
__global__ void scharr_gradient_kernel(const uint8_t* __restrict__ img, int W, int H,
                                       float* __restrict__ gx_out, float* __restrict__ gy_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;
    if (x < kGradBorder || x >= W - kGradBorder || y < kGradBorder || y >= H - kGradBorder) {
        gx_out[idx] = 0.0f; gy_out[idx] = 0.0f;
        return;
    }
    const int i00 = img[(y - 1) * W + (x - 1)], i01 = img[(y - 1) * W + x], i02 = img[(y - 1) * W + (x + 1)];
    const int i10 = img[y * W + (x - 1)],                                  i12 = img[y * W + (x + 1)];
    const int i20 = img[(y + 1) * W + (x - 1)], i21 = img[(y + 1) * W + x], i22 = img[(y + 1) * W + (x + 1)];
    const int gx = (3 * i02 + 10 * i12 + 3 * i22) - (3 * i00 + 10 * i10 + 3 * i20);
    const int gy = (3 * i20 + 10 * i21 + 3 * i22) - (3 * i00 + 10 * i01 + 3 * i02);
    gx_out[idx] = static_cast<float>(gx) * (1.0f / 32.0f);
    gy_out[idx] = static_cast<float>(gy) * (1.0f / 32.0f);
}

void launch_scharr_gradient(const uint8_t* d_img, int W, int H, float* d_gx, float* d_gy)
{
    scharr_gradient_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_img, W, H, d_gx, d_gy);
    CUDA_CHECK_LAST_ERROR("scharr_gradient_kernel launch");
}

// structure_tensor_kernel — per-pixel 5x5-window structure tensor and its
// SMALL eigenvalue (the aperture-problem confidence, 01.03's derivation
// cited: a shift along the small-eigenvalue direction barely changes the
// window's SSD cost, so a large small-eigenvalue means the flow estimate is
// trustworthy in every direction).
__global__ void structure_tensor_kernel(const float* __restrict__ gx, const float* __restrict__ gy,
                                        int W, int H,
                                        float* __restrict__ sxx_out, float* __restrict__ syy_out,
                                        float* __restrict__ sxy_out, float* __restrict__ min_eig_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;
    if (x < kLkBorder || x >= W - kLkBorder || y < kLkBorder || y >= H - kLkBorder) {
        sxx_out[idx] = syy_out[idx] = sxy_out[idx] = min_eig_out[idx] = 0.0f;
        return;
    }
    float Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f;
    #pragma unroll
    for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
        #pragma unroll
        for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
            const int widx = (y + wy) * W + (x + wx);
            const float gxv = gx[widx], gyv = gy[widx];
            Sxx += gxv * gxv; Syy += gyv * gyv; Sxy += gxv * gyv;
        }
    }
    sxx_out[idx] = Sxx; syy_out[idx] = Syy; sxy_out[idx] = Sxy;
    const float half_trace = 0.5f * (Sxx + Syy);
    const float det = Sxx * Syy - Sxy * Sxy;
    const float disc = fmaxf(half_trace * half_trace - det, 0.0f);
    min_eig_out[idx] = half_trace - sqrtf(disc);
}

void launch_structure_tensor(const float* d_gx, const float* d_gy, int W, int H,
                             float* d_sxx, float* d_syy, float* d_sxy, float* d_min_eig)
{
    structure_tensor_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_gx, d_gy, W, H, d_sxx, d_syy, d_sxy, d_min_eig);
    CUDA_CHECK_LAST_ERROR("structure_tensor_kernel launch");
}

// lk_iterate_kernel — one forward-additive LK refinement step (01.03's
// derivation cited in full: linearize the warped re-sampling residual,
// solve the 2x2 normal equations via Cramer's rule, clamp the step).
__global__ void lk_iterate_kernel(const uint8_t* __restrict__ img0, const uint8_t* __restrict__ img1,
                                  int W, int H,
                                  const float* __restrict__ gx, const float* __restrict__ gy,
                                  const float* __restrict__ sxx, const float* __restrict__ syy,
                                  const float* __restrict__ sxy,
                                  float* __restrict__ flow_u, float* __restrict__ flow_v)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;
    if (x < kLkBorder || x >= W - kLkBorder || y < kLkBorder || y >= H - kLkBorder) return;

    const float Sxx = sxx[idx], Syy = syy[idx], Sxy = sxy[idx];
    const float det = Sxx * Syy - Sxy * Sxy;
    if (det < kLkDetEpsilon) return;

    const float u = flow_u[idx], v = flow_v[idx];
    float bx = 0.0f, by = 0.0f;
    #pragma unroll
    for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
        #pragma unroll
        for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
            const int widx = (y + wy) * W + (x + wx);
            const float sample_x = static_cast<float>(x + wx) + u;
            const float sample_y = static_cast<float>(y + wy) + v;
            const float i1w = bilinear_sample_u8(img1, W, H, sample_x, sample_y);
            const float it = i1w - static_cast<float>(img0[widx]);
            bx += gx[widx] * it; by += gy[widx] * it;
        }
    }
    float ddu = -(Syy * bx - Sxy * by) / det;
    float ddv = -(-Sxy * bx + Sxx * by) / det;
    ddu = clamp_f(ddu, -kLkMaxStepPerIterPx, kLkMaxStepPerIterPx);
    ddv = clamp_f(ddv, -kLkMaxStepPerIterPx, kLkMaxStepPerIterPx);
    flow_u[idx] = u + ddu; flow_v[idx] = v + ddv;
}

void launch_lk_iterate(const uint8_t* d_img0, const uint8_t* d_img1, int W, int H,
                       const float* d_gx, const float* d_gy,
                       const float* d_sxx, const float* d_syy, const float* d_sxy,
                       float* d_flow_u, float* d_flow_v)
{
    lk_iterate_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(
        d_img0, d_img1, W, H, d_gx, d_gy, d_sxx, d_syy, d_sxy, d_flow_u, d_flow_v);
    CUDA_CHECK_LAST_ERROR("lk_iterate_kernel launch");
}

// upsample_flow_kernel — bilinear-upsample + 2x magnitude scale (01.03's
// pyramid-propagation argument, cited: a flow vector is a pixel
// displacement, and the same physical motion spans twice as many pixels at
// twice the resolution).
__global__ void upsample_flow_kernel(const float* __restrict__ coarse_u, const float* __restrict__ coarse_v,
                                     int coarseW, int coarseH,
                                     float* __restrict__ fine_u, float* __restrict__ fine_v,
                                     int fineW, int fineH)
{
    const int fx = blockIdx.x * blockDim.x + threadIdx.x;
    const int fy = blockIdx.y * blockDim.y + threadIdx.y;
    if (fx >= fineW || fy >= fineH) return;
    const float cx = (static_cast<float>(fx) + 0.5f) * 0.5f - 0.5f;
    const float cy = (static_cast<float>(fy) + 0.5f) * 0.5f - 0.5f;
    const float ccx = fminf(fmaxf(cx, 0.0f), static_cast<float>(coarseW - 1));
    const float ccy = fminf(fmaxf(cy, 0.0f), static_cast<float>(coarseH - 1));
    const int x0 = static_cast<int>(floorf(ccx)), y0 = static_cast<int>(floorf(ccy));
    const int x1 = min(x0 + 1, coarseW - 1), y1 = min(y0 + 1, coarseH - 1);
    const float wx = ccx - static_cast<float>(x0), wy = ccy - static_cast<float>(y0);
    auto bilerp = [&](const float* __restrict__ field) -> float {
        const float v00 = field[y0 * coarseW + x0], v10 = field[y0 * coarseW + x1];
        const float v01 = field[y1 * coarseW + x0], v11 = field[y1 * coarseW + x1];
        const float top = v00 + (v10 - v00) * wx;
        const float bot = v01 + (v11 - v01) * wx;
        return top + (bot - top) * wy;
    };
    const int fidx = fy * fineW + fx;
    fine_u[fidx] = 2.0f * bilerp(coarse_u);
    fine_v[fidx] = 2.0f * bilerp(coarse_v);
}

void launch_upsample_flow(const float* d_coarse_u, const float* d_coarse_v, int coarseW, int coarseH,
                          float* d_fine_u, float* d_fine_v, int fineW, int fineH)
{
    upsample_flow_kernel<<<grid2d(fineW, fineH), dim3(kBlock2D, kBlock2D)>>>(
        d_coarse_u, d_coarse_v, coarseW, coarseH, d_fine_u, d_fine_v, fineW, fineH);
    CUDA_CHECK_LAST_ERROR("upsample_flow_kernel launch");
}

// run_pyramidal_lk_gpu — the FULL Milestone-1 orchestration: builds the
// 2-level pyramid, then a SEQUENTIAL coarse-to-fine loop (level 1's flow
// must finish before level 0 can use it as its starting point — the one
// place this project's flow stage is not embarrassingly parallel; every
// kernel WITHIN a level still is). 01.03's run_pyramidal_lk_gpu shape,
// cited, simplified to a FIXED 2 levels (no ablation parameter needed here).
void run_pyramidal_lk_gpu(const uint8_t* d_img0_full, const uint8_t* d_img1_full,
                          float* d_flow_u_out, float* d_flow_v_out, float* d_min_eig_out)
{
    uint8_t* d_img0[kNumLevels]; uint8_t* d_img1[kNumLevels];
    float* d_gx[kNumLevels]; float* d_gy[kNumLevels];
    float* d_sxx[kNumLevels]; float* d_syy[kNumLevels]; float* d_sxy[kNumLevels];
    float* d_min_eig[kNumLevels];
    float* d_flow_u[kNumLevels]; float* d_flow_v[kNumLevels];

    for (int L = 0; L < kNumLevels; ++L) {
        const size_t n = static_cast<size_t>(level_w(L)) * level_h(L);
        CUDA_CHECK(cudaMalloc(&d_img0[L], n));
        CUDA_CHECK(cudaMalloc(&d_img1[L], n));
        CUDA_CHECK(cudaMalloc(&d_gx[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gy[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sxx[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_syy[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sxy[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_eig[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_flow_u[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_flow_v[L], n * sizeof(float)));
    }

    CUDA_CHECK(cudaMemcpy(d_img0[0], d_img0_full, static_cast<size_t>(kW) * kH, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_img1[0], d_img1_full, static_cast<size_t>(kW) * kH, cudaMemcpyDeviceToDevice));
    for (int L = 1; L < kNumLevels; ++L) {
        launch_downsample_area2x(d_img0[L - 1], level_w(L - 1), level_h(L - 1), d_img0[L]);
        launch_downsample_area2x(d_img1[L - 1], level_w(L - 1), level_h(L - 1), d_img1[L]);
    }

    const int start_level = kNumLevels - 1;
    CUDA_CHECK(cudaMemset(d_flow_u[start_level], 0, static_cast<size_t>(level_w(start_level)) * level_h(start_level) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flow_v[start_level], 0, static_cast<size_t>(level_w(start_level)) * level_h(start_level) * sizeof(float)));

    for (int L = start_level; L >= 0; --L) {
        const int Wl = level_w(L), Hl = level_h(L);
        launch_scharr_gradient(d_img0[L], Wl, Hl, d_gx[L], d_gy[L]);
        launch_structure_tensor(d_gx[L], d_gy[L], Wl, Hl, d_sxx[L], d_syy[L], d_sxy[L], d_min_eig[L]);
        for (int it = 0; it < kLkIterationsPerLevel; ++it) {
            launch_lk_iterate(d_img0[L], d_img1[L], Wl, Hl, d_gx[L], d_gy[L], d_sxx[L], d_syy[L], d_sxy[L],
                              d_flow_u[L], d_flow_v[L]);
        }
        if (L > 0) {
            launch_upsample_flow(d_flow_u[L], d_flow_v[L], Wl, Hl,
                                 d_flow_u[L - 1], d_flow_v[L - 1], level_w(L - 1), level_h(L - 1));
        }
    }

    const size_t n0 = static_cast<size_t>(kW) * kH;
    CUDA_CHECK(cudaMemcpy(d_flow_u_out, d_flow_u[0], n0 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_flow_v_out, d_flow_v[0], n0 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_min_eig_out, d_min_eig[0], n0 * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int L = 0; L < kNumLevels; ++L) {
        CUDA_CHECK(cudaFree(d_img0[L])); CUDA_CHECK(cudaFree(d_img1[L]));
        CUDA_CHECK(cudaFree(d_gx[L])); CUDA_CHECK(cudaFree(d_gy[L]));
        CUDA_CHECK(cudaFree(d_sxx[L])); CUDA_CHECK(cudaFree(d_syy[L])); CUDA_CHECK(cudaFree(d_sxy[L]));
        CUDA_CHECK(cudaFree(d_min_eig[L]));
        CUDA_CHECK(cudaFree(d_flow_u[L])); CUDA_CHECK(cudaFree(d_flow_v[L]));
    }
}

// ===========================================================================
// MILESTONE 2 — 3-D lifting with the depth-consistency guard (kernels.cuh's
// file header stage 2 derives the physics; this is the direct translation).
// ===========================================================================

// backproject — pixel-center pinhole back-projection (MUST match
// scripts/make_synthetic.py's camera_ray_cam_frame formula exactly, see
// kernels.cuh's camera-model note). px,py may be fractional (the flow-
// shifted target location).
__device__ inline void backproject(float px, float py, float depth, float out[3])
{
    out[0] = ((px + 0.5f - kCx) / kFx) * depth;
    out[1] = ((py + 0.5f - kCy) / kFy) * depth;
    out[2] = depth;
}

__global__ void lift_scene_flow_kernel(const float* __restrict__ flow_u, const float* __restrict__ flow_v,
                                       const float* __restrict__ confidence,
                                       const float* __restrict__ d0, const float* __restrict__ d1,
                                       float* __restrict__ P1_out, float* __restrict__ P2_out,
                                       uint8_t* __restrict__ valid_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kW || y >= kH) return;
    const int idx = y * kW + x;

    P1_out[3 * idx + 0] = P1_out[3 * idx + 1] = P1_out[3 * idx + 2] = 0.0f;
    P2_out[3 * idx + 0] = P2_out[3 * idx + 1] = P2_out[3 * idx + 2] = 0.0f;
    valid_out[idx] = 0u;

    const float d0v = d0[idx];
    if (d0v == kInvalidDepth) return;                          // source pixel: no return (sky) here
    if (confidence[idx] < kMinConfidenceForLift) return;        // aperture-problem guard (01.03's confidence, cited)

    const float qx = static_cast<float>(x) + flow_u[idx];
    const float qy = static_cast<float>(y) + flow_v[idx];
    // Border guard: every one of the 4 bilinear taps below must be an
    // IN-BOUNDS array read (no clamping here, unlike bilinear_sample_u8 —
    // clamping a DEPTH sample near the image edge would silently reuse an
    // edge pixel's depth for an off-frame ray, fabricating a correspondence
    // that was never observed; rejecting is the honest choice).
    if (qx < 0.0f || qx > static_cast<float>(kW - 1) || qy < 0.0f || qy > static_cast<float>(kH - 1)) return;

    const int x0 = static_cast<int>(floorf(qx)), y0 = static_cast<int>(floorf(qy));
    const int x1 = min(x0 + 1, kW - 1), y1 = min(y0 + 1, kH - 1);
    const float d00 = d1[y0 * kW + x0], d10 = d1[y0 * kW + x1];
    const float d01 = d1[y1 * kW + x0], d11 = d1[y1 * kW + x1];
    if (d00 == kInvalidDepth || d10 == kInvalidDepth || d01 == kInvalidDepth || d11 == kInvalidDepth) return;

    // Depth-consistency guard (kernels.cuh's kDepthEdgeGuardM derivation):
    // 4 taps that disagree by more than the guard likely straddle a REAL
    // depth discontinuity, not sensor noise — reject rather than fabricate
    // a physically meaningless blended depth.
    const float dmin = fminf(fminf(d00, d10), fminf(d01, d11));
    const float dmax = fmaxf(fmaxf(d00, d10), fmaxf(d01, d11));
    if (dmax - dmin > kDepthEdgeGuardM) return;

    const float fx = qx - static_cast<float>(x0), fy = qy - static_cast<float>(y0);
    const float top = d00 + (d10 - d00) * fx;
    const float bot = d01 + (d11 - d01) * fx;
    const float d1v = top + (bot - top) * fy;

    float P1[3], P2[3];
    backproject(static_cast<float>(x), static_cast<float>(y), d0v, P1);
    backproject(qx, qy, d1v, P2);
    P1_out[3 * idx + 0] = P1[0]; P1_out[3 * idx + 1] = P1[1]; P1_out[3 * idx + 2] = P1[2];
    P2_out[3 * idx + 0] = P2[0]; P2_out[3 * idx + 1] = P2[1]; P2_out[3 * idx + 2] = P2[2];
    valid_out[idx] = 1u;
}

void launch_lift_scene_flow(const float* d_flow_u, const float* d_flow_v, const float* d_confidence,
                            const float* d_d0, const float* d_d1,
                            float* d_P1_out, float* d_P2_out, uint8_t* d_valid_out)
{
    lift_scene_flow_kernel<<<grid2d(kW, kH), dim3(kBlock2D, kBlock2D)>>>(
        d_flow_u, d_flow_v, d_confidence, d_d0, d_d1, d_P1_out, d_P2_out, d_valid_out);
    CUDA_CHECK_LAST_ERROR("lift_scene_flow_kernel launch");
}

// ===========================================================================
// MILESTONE 3 — residual + weighted covariance reduction.
// ===========================================================================

__global__ void compute_residuals_kernel(int n, const float* __restrict__ P1, const float* __restrict__ P2,
                                         const uint8_t* __restrict__ valid, Rigid3 T,
                                         float* __restrict__ residual_vec_out,
                                         float* __restrict__ residual_mag_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!valid[i]) {
        residual_vec_out[3 * i + 0] = residual_vec_out[3 * i + 1] = residual_vec_out[3 * i + 2] = 0.0f;
        residual_mag_out[i] = 0.0f;
        return;
    }
    const float p1[3] = { P1[3 * i], P1[3 * i + 1], P1[3 * i + 2] };
    float tp1[3];
    apply_rigid(T, p1, tp1);   // shared helper (kernels.cuh) — see its header for why sharing this is safe
    const float rx = tp1[0] - P2[3 * i + 0];
    const float ry = tp1[1] - P2[3 * i + 1];
    const float rz = tp1[2] - P2[3 * i + 2];
    residual_vec_out[3 * i + 0] = rx; residual_vec_out[3 * i + 1] = ry; residual_vec_out[3 * i + 2] = rz;
    residual_mag_out[i] = sqrtf(rx * rx + ry * ry + rz * rz);
}

void launch_compute_residuals(int n, const float* d_P1, const float* d_P2, const uint8_t* d_valid, Rigid3 T,
                              float* d_residual_vec_out, float* d_residual_mag_out)
{
    const int block = 256;
    compute_residuals_kernel<<<grid1d(n, block), block>>>(n, d_P1, d_P2, d_valid, T, d_residual_vec_out, d_residual_mag_out);
    CUDA_CHECK_LAST_ERROR("compute_residuals_kernel launch");
}

// weighted_covariance_reduce_kernel — thread i folds point i's (weighted)
// contribution to the kCovarWidth=16-scalar record (kernels.cuh derives the
// layout) into SHARED memory, then a standard power-of-two tree reduction
// (the same shape every reduction kernel in this repo uses, e.g. 02.06's
// build_normal_system_kernel) collapses blockDim.x threads' records down to
// ONE row per block. Shared-memory budget: kThreadsReduce(128) *
// kCovarWidth(16) * 4 bytes = 8 KiB — comfortably within a single SM's
// shared-memory budget on sm_75..sm_89 alongside several resident blocks.
__global__ void weighted_covariance_reduce_kernel(int n, const float* __restrict__ P1, const float* __restrict__ P2,
                                                   const float* __restrict__ weight,
                                                   float* __restrict__ block_partials)
{
    __shared__ float sdata[kThreadsReduce * kCovarWidth];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float rec[kCovarWidth];
    #pragma unroll
    for (int k = 0; k < kCovarWidth; ++k) rec[k] = 0.0f;

    if (i < n) {
        const float w = weight[i];
        if (w > 0.0f) {
            const float p1x = P1[3 * i], p1y = P1[3 * i + 1], p1z = P1[3 * i + 2];
            const float p2x = P2[3 * i], p2y = P2[3 * i + 1], p2z = P2[3 * i + 2];
            rec[0] = w;
            rec[1] = w * p1x; rec[2] = w * p1y; rec[3] = w * p1z;
            rec[4] = w * p2x; rec[5] = w * p2y; rec[6] = w * p2z;
            rec[7] = w * p1x * p2x; rec[8] = w * p1x * p2y; rec[9] = w * p1x * p2z;
            rec[10] = w * p1y * p2x; rec[11] = w * p1y * p2y; rec[12] = w * p1y * p2z;
            rec[13] = w * p1z * p2x; rec[14] = w * p1z * p2y; rec[15] = w * p1z * p2z;
        }
    }
    #pragma unroll
    for (int k = 0; k < kCovarWidth; ++k) sdata[threadIdx.x * kCovarWidth + k] = rec[k];
    __syncthreads();

    // Binary tree reduction within the block: at each step, the first half
    // of the still-active threads adds the second half's record into its
    // own (coalesced within a k-loop since consecutive threads' rec[k]
    // slots are kCovarWidth apart — a stride-kCovarWidth access, the
    // documented cost of packing many scalars per thread; kernels.cu's
    // 02.06 counterpart makes the identical trade for its 27-wide record).
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            #pragma unroll
            for (int k = 0; k < kCovarWidth; ++k)
                sdata[threadIdx.x * kCovarWidth + k] += sdata[(threadIdx.x + s) * kCovarWidth + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < kCovarWidth; ++k) block_partials[blockIdx.x * kCovarWidth + k] = sdata[k];
    }
}

int launch_weighted_covariance_reduce(int n, const float* d_P1, const float* d_P2, const float* d_weight,
                                      float* d_block_partials)
{
    const int nblocks = blocks_for(n, kThreadsReduce);
    weighted_covariance_reduce_kernel<<<nblocks, kThreadsReduce>>>(n, d_P1, d_P2, d_weight, d_block_partials);
    CUDA_CHECK_LAST_ERROR("weighted_covariance_reduce_kernel launch");
    return nblocks;
}

// ===========================================================================
// MILESTONE 4 — residual segmentation: threshold + 3x3 morphological open.
// ===========================================================================

__global__ void threshold_mask_kernel(int n, const float* __restrict__ residual_mag,
                                      const uint8_t* __restrict__ valid, float threshold_m,
                                      uint8_t* __restrict__ mask_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mask_out[i] = (valid[i] && residual_mag[i] > threshold_m) ? 1u : 0u;
}

void launch_threshold_mask(int n, const float* d_residual_mag, const uint8_t* d_valid, float threshold_m,
                           uint8_t* d_mask_out)
{
    const int block = 256;
    threshold_mask_kernel<<<grid1d(n, block), block>>>(n, d_residual_mag, d_valid, threshold_m, d_mask_out);
    CUDA_CHECK_LAST_ERROR("threshold_mask_kernel launch");
}

// erode3x3_kernel — output=1 iff EVERY pixel in the 3x3 window (out-of-
// bounds reads as 0) is 1 — a binary "AND over the neighborhood", the
// standard erosion primitive (30.01's morphological-cleanup pattern, cited).
__global__ void erode3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uint8_t v = 1u;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
            v = v & nb;
        }
    }
    out[y * W + x] = v;
}

// dilate3x3_kernel — output=1 iff ANY pixel in the 3x3 window is 1 (an
// "OR over the neighborhood"). erode-then-dilate (an OPENING) removes
// isolated single/few-pixel false positives without shrinking the surviving
// blob's silhouette (30.01's cited pattern).
__global__ void dilate3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uint8_t v = 0u;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
            v = v | nb;
        }
    }
    out[y * W + x] = v;
}

void launch_morphological_open(uint8_t* d_mask_inout)
{
    uint8_t* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scratch, static_cast<size_t>(kPixels)));
    erode3x3_kernel<<<grid2d(kW, kH), dim3(kBlock2D, kBlock2D)>>>(d_mask_inout, kW, kH, d_scratch);
    CUDA_CHECK_LAST_ERROR("erode3x3_kernel launch");
    dilate3x3_kernel<<<grid2d(kW, kH), dim3(kBlock2D, kBlock2D)>>>(d_scratch, kW, kH, d_mask_inout);
    CUDA_CHECK_LAST_ERROR("dilate3x3_kernel launch");
    CUDA_CHECK(cudaFree(d_scratch));
}

// ===========================================================================
// MILESTONE 4b — connected-component labeling + size filter (kernels.cuh's
// Milestone-4b constants block derives WHY this stage exists and its
// kMinComponentSizePx floor; read that comment first).
// ===========================================================================

// ccl_init_kernel — seed every foreground pixel's label with ITS OWN linear
// index (the only label a pixel can ever be sure of before any neighbor
// information propagates); background gets kLabelNone. One thread per
// pixel, pure map, n=kW*kH (01.06's ccl_init_kernel, cited; re-typed fresh).
__global__ void ccl_init_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    label[i] = mask[i] ? i : kLabelNone;
}

void launch_ccl_init(const uint8_t* d_mask, int* d_label, int W, int H)
{
    const int n = W * H, block = 256;
    ccl_init_kernel<<<grid1d(n, block), block>>>(d_mask, d_label, n);
    CUDA_CHECK_LAST_ERROR("ccl_init_kernel launch");
}

// ccl_propagate_sweep_kernel — one round of 4-connected label propagation:
// thread i looks at its own current label and its four foreground
// neighbors' CURRENT labels, keeps the minimum, and atomicMin's it back in.
// Because every label only ever DECREASES and is bounded below by 0, repeat
// sweeps (main.cu's loop) converge in finitely many rounds to the UNIQUE
// fixed point label[p] = min linear index over p's 4-connected component —
// independent of which order threads happen to run in this sweep or across
// sweeps (01.06's ccl_propagate_sweep_kernel, cited; re-typed fresh here for
// this project's own mask layout). `changed` (atomicOr) lets main.cu detect
// convergence without needing to know the true diameter in advance.
__global__ void ccl_propagate_sweep_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label,
                                           int W, int H, int* __restrict__ changed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
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

void launch_ccl_propagate_sweep(const uint8_t* d_mask, int* d_label, int W, int H, int* d_changed)
{
    const int n = W * H, block = 256;
    ccl_propagate_sweep_kernel<<<grid1d(n, block), block>>>(d_mask, d_label, W, H, d_changed);
    CUDA_CHECK_LAST_ERROR("ccl_propagate_sweep_kernel launch");
}

// component_size_count_kernel — every foreground pixel ATOMICALLY adds 1 to
// its canonical label's size bucket (the same "dense accumulator keyed by
// canonical label" atomic-scatter idea 01.06's component_stats_accumulate_
// kernel and 30.01's Stage 5 use, cited, simplified here to just the pixel
// COUNT — this project needs no centroid/bbox, only size). The caller must
// zero size_out first (main.cu's cudaMemset, mirroring how every other
// reduction scratch buffer in this project is prepared before accumulation).
__global__ void component_size_count_kernel(const uint8_t* __restrict__ mask, const int* __restrict__ label,
                                            int* __restrict__ size_out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!mask[i]) return;
    atomicAdd(&size_out[label[i]], 1);
}

// component_filter_kernel — pure per-pixel MAP: keep a mask pixel only if
// its OWN component's total size cleared kMinComponentSizePx. n=kW*kH.
__global__ void component_filter_kernel(const uint8_t* __restrict__ mask_in, const int* __restrict__ label,
                                        const int* __restrict__ size, int min_size,
                                        uint8_t* __restrict__ mask_out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mask_out[i] = (mask_in[i] && size[label[i]] >= min_size) ? 1u : 0u;
}

void launch_component_size_filter(const uint8_t* d_mask_in, const int* d_label, int min_size_px,
                                  uint8_t* d_mask_out, int n)
{
    int* d_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_size, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_size, 0, static_cast<size_t>(n) * sizeof(int)));
    const int block = 256;
    component_size_count_kernel<<<grid1d(n, block), block>>>(d_mask_in, d_label, d_size, n);
    CUDA_CHECK_LAST_ERROR("component_size_count_kernel launch");
    component_filter_kernel<<<grid1d(n, block), block>>>(d_mask_in, d_label, d_size, min_size_px, d_mask_out, n);
    CUDA_CHECK_LAST_ERROR("component_filter_kernel launch");
    CUDA_CHECK(cudaFree(d_size));
}
