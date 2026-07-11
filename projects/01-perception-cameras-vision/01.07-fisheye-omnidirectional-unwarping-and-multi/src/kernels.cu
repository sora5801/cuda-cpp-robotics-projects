// ===========================================================================
// kernels.cu — GPU kernels for project 01.07 (Fisheye/omnidirectional
//              unwarping and multi-camera surround-view stitching)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the small host-side launch
// wrappers that own the grid/block math (kernels.cuh's file header explains
// why that pairing lives together). Four kernels, in the order this project
// uses them (main.cu):
//   1. build_rect_lut_kernel   — Half 1a: rectilinear-output LUT (once, geometry only)
//   2. build_cyl_lut_kernel    — Half 1b: cylindrical-output LUT (once, geometry only)
//   3. remap_bilinear_kernel   — Half 1: bilinear-gather remap, reused for BOTH outputs
//   4. bev_compose_kernel      — Half 2: the 4-camera surround-view compositor
//
// Every kernel below is a pure MAP: kernels 1-3 map one OUTPUT pixel to one
// GPU thread (the same "one thread per output pixel" shape 01.01's
// launch_build_remap_lut / launch_remap_bilinear use); kernel 4 maps one
// BEV output pixel to one thread, with a short (4-iteration, always-the-
// same-trip-count) camera loop INSIDE that thread rather than one thread
// per (BEV pixel, camera) pair — kernel 4's own header below argues this
// choice explicitly.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"           // RemapSample, camera-model constants + shared HD helpers, launcher signatures
#include "util/cuda_check.cuh"   // CUDA_CHECK_LAST_ERROR for post-launch error surfacing

// ---------------------------------------------------------------------------
// clampi — plain integer clamp, used by the bilinear sampler's edge policy
// below. Trivial, but declared once so every clamp in this file reads the
// same way (mirrors 01.01's clampi_cpu, which is INDEPENDENTLY retyped in
// reference_cpu.cpp on purpose — this is the GPU side's own copy).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// ---------------------------------------------------------------------------
// bilinear_sample_rgb — device-side bilinear sampler shared by
// remap_bilinear_kernel AND bev_compose_kernel (both live in THIS file —
// reusing code within one side of the GPU-vs-CPU twin is ordinary
// engineering, not a violation of the twin-independence rule, which is
// specifically about NOT sharing code ACROSS the GPU/CPU boundary;
// reference_cpu.cpp types its own independent bilinear_sample_rgb_cpu).
//
// Clamp-to-edge boundary policy: a sample coordinate outside the image
// (which DOES happen — see fisheye_unproject's defensive clamp and PART 1's
// vignette-region note in kernels.cuh) is clamped to the nearest valid
// pixel rather than producing garbage; visually this reads as the border
// pixel "smearing" outward, a standard, honest choice for an unwarp demo
// (documented, not hidden — THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
__device__ void bilinear_sample_rgb(const unsigned char* __restrict__ img, int W, int H,
                                    float u, float v, unsigned char out[3])
{
    if (u < 0.0f) u = 0.0f;
    if (u > static_cast<float>(W - 1)) u = static_cast<float>(W - 1);
    if (v < 0.0f) v = 0.0f;
    if (v > static_cast<float>(H - 1)) v = static_cast<float>(H - 1);

    const int x0 = static_cast<int>(floorf(u));
    const int y0 = static_cast<int>(floorf(v));
    const int x1 = clampi(x0 + 1, 0, W - 1);
    const int y1 = clampi(y0 + 1, 0, H - 1);
    const float fx = u - static_cast<float>(x0);
    const float fy = v - static_cast<float>(y0);

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float v00 = static_cast<float>(img[(y0 * W + x0) * 3 + c]);
        const float v10 = static_cast<float>(img[(y0 * W + x1) * 3 + c]);
        const float v01 = static_cast<float>(img[(y1 * W + x0) * 3 + c]);
        const float v11 = static_cast<float>(img[(y1 * W + x1) * 3 + c]);
        const float top = v00 + (v10 - v00) * fx;
        const float bot = v01 + (v11 - v01) * fx;
        const float val = top + (bot - top) * fy;
        out[c] = static_cast<unsigned char>(val + 0.5f);   // round-to-nearest, clamp unnecessary (inputs are uint8)
    }
}

// ---------------------------------------------------------------------------
// build_rect_lut_kernel — Half 1a: one thread per RECTILINEAR output pixel.
// Thread (bx,tx),(by,ty) owns output pixel (xo,yo) = (blockIdx.x*blockDim.x
// + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y); a 2-D grid is the
// natural mapping for a 2-D image (no index-flattening arithmetic needed,
// unlike a 1-D grid-stride map over a flattened array).
//
// Per-thread work: pinhole_unproject_rect() -> fisheye_project() -> write
// one RemapSample. Purely geometric (depends only on (xo,yo) and the
// compile-time camera constants) — this is why it is precomputed ONCE and
// reused by every call to remap_bilinear_kernel with this LUT, exactly
// 01.01's launch_build_remap_lut rationale.
//
// Corner theta check (measured, not assumed — see kernels.cuh's kRectFx
// comment): the rectilinear output's four corners are the WORST-CASE ray
// angles this kernel ever computes. At (xo,yo)=(kRectW-1,kRectH-1):
// X=(199-99.5)/99.5=1.0, Y=(149-74.5)/99.5=0.749, Z=1 -> theta =
// atan2(hypot(1.0,0.749),1) = atan2(1.249,1) ~= 51.3 deg — comfortably
// under kFishValidHalfFovRad (92.5 deg), so this kernel's output pixels
// are ALWAYS drawn from inside the fisheye's illuminated circle; no
// separate visibility gate is needed here (contrast with bev_compose_kernel
// below, whose rays legitimately leave the circle for many BEV pixels).
//
// Memory: d_lut is WRITE-ONLY here (each thread writes exactly its own
// element, no shared memory, no atomics — an embarrassingly parallel map).
// ---------------------------------------------------------------------------
__global__ void build_rect_lut_kernel(RemapSample* __restrict__ d_lut)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= kRectW || yo >= kRectH) return;   // guard the ragged edge of the launch grid

    float X, Y, Z;
    pinhole_unproject_rect(xo, yo, X, Y, Z);
    d_lut[yo * kRectW + xo] = fisheye_project(X, Y, Z);
}

// ---------------------------------------------------------------------------
// build_cyl_lut_kernel — Half 1b: the cylindrical-output twin of the kernel
// above. Same shape, same reasoning; the only difference is which
// unproject function feeds fisheye_project (cyl_unproject's azimuth/
// elevation sweep instead of pinhole_unproject_rect's tangent-plane rays —
// kernels.cuh's PART 2 header derives both).
//
// Corner theta check: at (xo,yo)=(kCylW-1,0) (max azimuth, max elevation):
// az=80 deg, el=35 deg -> X=sin(80)*cos(35)=0.807, Y=-sin(35)=-0.574,
// Z=cos(80)*cos(35)=0.142 -> theta = atan2(hypot(0.807,0.574),0.142) ~=
// 81.9 deg — still under 92.5 deg, but with less margin than the
// rectilinear case above (10.6 deg vs ~41 deg) precisely because this
// output surface deliberately covers a WIDER FOV (kernels.cuh PART 2's
// projection-surface trade-off, made numerically concrete here).
// ---------------------------------------------------------------------------
__global__ void build_cyl_lut_kernel(RemapSample* __restrict__ d_lut)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= kCylW || yo >= kCylH) return;

    float X, Y, Z;
    cyl_unproject(xo, yo, X, Y, Z);
    d_lut[yo * kCylW + xo] = fisheye_project(X, Y, Z);
}

// ---------------------------------------------------------------------------
// remap_bilinear_kernel — Half 1's generic bilinear-gather: one thread per
// OUTPUT pixel, looks up its LUT entry, bilinear-samples the fisheye
// source image. Reused (via two separate launch_remap_bilinear() calls,
// see main.cu) for both the rectilinear and cylindrical outputs — the LUT
// and the output dimensions are the only things that differ, so one kernel
// serves both, exactly 01.01's remap_bilinear_kernel reused across
// pipeline stages.
//
// Thread mapping: 2-D grid, thread (xo,yo) owns output pixel (xo,yo) — same
// shape as the LUT-build kernels above (deliberately: a learner who
// understands one understands all three).
// ---------------------------------------------------------------------------
__global__ void remap_bilinear_kernel(const unsigned char* __restrict__ d_src,
                                      const RemapSample* __restrict__ d_lut,
                                      unsigned char* __restrict__ d_out,
                                      int srcW, int srcH, int outW, int outH)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= outW || yo >= outH) return;

    const int idx = yo * outW + xo;
    const RemapSample s = d_lut[idx];
    unsigned char rgb[3];
    bilinear_sample_rgb(d_src, srcW, srcH, s.u, s.v, rgb);
    d_out[idx * 3 + 0] = rgb[0];
    d_out[idx * 3 + 1] = rgb[1];
    d_out[idx * 3 + 2] = rgb[2];
}

// ---------------------------------------------------------------------------
// bev_compose_kernel — Half 2's centerpiece: one thread per BEV output
// pixel; INSIDE that thread, a fixed 4-iteration loop over the rig cameras
// (kernels.cuh's rig_camera_to_bev_sample for each), accumulating a
// weighted blend and a coverage bitmask.
//
// Why one thread per BEV pixel with an in-thread camera loop, rather than
// one thread per (BEV pixel, camera) pair (4x more threads, each doing
// 1/4 the work, then a cross-thread reduction to blend)?
//   * The blend is a tiny, FIXED-size (exactly 4-term) weighted sum — the
//     kind of reduction that is CHEAPER done by a single thread in
//     registers than coordinated across 4 threads via shared memory or
//     atomics. Registers are the fastest memory this GPU has (no bank
//     conflicts, no synchronization) and 4 accumulator terms cost nothing
//     next to the arithmetic already being done per camera.
//   * Every camera's visibility/weight test is DATA-DEPENDENT (some BEV
//     pixels are seen by 1 camera, some by 2, a very few by 0) — splitting
//     the loop across threads would need either a second kernel launch to
//     reduce partial results (extra global-memory round trip for a
//     4-element sum) or atomics into the shared output pixel (a genuine
//     race the in-thread version has no need to invite).
//   * kBevW*kBevH = 320*320 = 102,400 threads is already comfortably more
//     than this GPU's occupancy needs (an RTX 2080 SUPER has 46 SMs); the
//     4x-more-threads alternative would not measurably improve occupancy,
//     only add synchronization cost. THEORY.md "The GPU mapping" expands
//     this trade-off with the numbers measured on this project's demo.
//
// Thread mapping: 2-D grid over the kBevW x kBevH output, same shape as
// every other kernel in this file.
//
// Per-thread work: bev_pixel_to_ground() once, then for cam in
// {FRONT,LEFT,RIGHT,REAR}: rig_camera_to_bev_sample() (shared HD rig
// geometry — kernels.cuh's twin-independence note); if visible,
// bilinear_sample_rgb() the corresponding fisheye image and accumulate
// weight*rgb into a float sum plus weight into a float wsum, and set the
// camera's coverage bit. After the loop: if wsum>0, out = sum/wsum
// (the weighted-average blend); if wsum==0 (no camera sees this ground
// point at all — outside the rig's combined FOV footprint), out = 0
// (black) and coverage stays 0 — both read directly by main.cu's coverage
// gate.
// ---------------------------------------------------------------------------
__global__ void bev_compose_kernel(const unsigned char* __restrict__ d_front,
                                   const unsigned char* __restrict__ d_left,
                                   const unsigned char* __restrict__ d_right,
                                   const unsigned char* __restrict__ d_rear,
                                   unsigned char* __restrict__ d_bev,
                                   unsigned char* __restrict__ d_coverage)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= kBevW || yo >= kBevH) return;

    // Local array of the 4 fisheye source pointers, indexed by camera id
    // (kCamFront..kCamRear) — plain register-resident pointers, NOT a
    // separate device allocation (kernels.cuh PART 3's header explains why
    // this simpler form was chosen over a device array-of-pointers).
    const unsigned char* const imgs[kNumRigCameras] = { d_front, d_left, d_right, d_rear };

    float X, Y;
    bev_pixel_to_ground(xo, yo, X, Y);

    float sum[3] = { 0.0f, 0.0f, 0.0f };   // weighted color accumulator (registers)
    float wsum = 0.0f;                     // total feather weight seen so far
    unsigned char coverage = 0;            // per-camera contribution bitmask

    #pragma unroll
    for (int cam = 0; cam < kNumRigCameras; ++cam) {
        float u, v, weight;
        if (!rig_camera_to_bev_sample(cam, X, Y, u, v, weight)) continue;

        unsigned char rgb[3];
        bilinear_sample_rgb(imgs[cam], kFishW, kFishH, u, v, rgb);
        sum[0] += weight * static_cast<float>(rgb[0]);
        sum[1] += weight * static_cast<float>(rgb[1]);
        sum[2] += weight * static_cast<float>(rgb[2]);
        wsum += weight;
        coverage |= static_cast<unsigned char>(1u << cam);
    }

    const int idx = yo * kBevW + xo;
    if (wsum > 0.0f) {
        const float inv = 1.0f / wsum;
        d_bev[idx * 3 + 0] = static_cast<unsigned char>(sum[0] * inv + 0.5f);
        d_bev[idx * 3 + 1] = static_cast<unsigned char>(sum[1] * inv + 0.5f);
        d_bev[idx * 3 + 2] = static_cast<unsigned char>(sum[2] * inv + 0.5f);
    } else {
        d_bev[idx * 3 + 0] = 0;
        d_bev[idx * 3 + 1] = 0;
        d_bev[idx * 3 + 2] = 0;
    }
    d_coverage[idx] = coverage;
}

// ===========================================================================
// Launch wrappers — each computes its own launch geometry, launches, checks.
// Every kernel above uses a 2-D thread grid over its own output image, so
// every wrapper below shares the same block-size reasoning: 16x16 = 256
// threads/block, a warp-multiple (256/32=8 full warps) that keeps each
// block's footprint small enough for good occupancy on sm_75..sm_89 while
// still being large enough to amortize per-block launch overhead — the
// same "reasonable default, measure before trusting a single number"
// guidance as 01.01's saxpy-derived 256-thread blocks, adapted to 2-D.
// ===========================================================================

static constexpr int kBlock2D = 16;   // 16x16 = 256 threads/block, every launcher below

void launch_build_rect_lut(RemapSample* d_lut)
{
    const dim3 block(kBlock2D, kBlock2D);
    const dim3 grid((kRectW + kBlock2D - 1) / kBlock2D, (kRectH + kBlock2D - 1) / kBlock2D);
    build_rect_lut_kernel<<<grid, block>>>(d_lut);
    CUDA_CHECK_LAST_ERROR("build_rect_lut_kernel launch");
}

void launch_build_cyl_lut(RemapSample* d_lut)
{
    const dim3 block(kBlock2D, kBlock2D);
    const dim3 grid((kCylW + kBlock2D - 1) / kBlock2D, (kCylH + kBlock2D - 1) / kBlock2D);
    build_cyl_lut_kernel<<<grid, block>>>(d_lut);
    CUDA_CHECK_LAST_ERROR("build_cyl_lut_kernel launch");
}

void launch_remap_bilinear(const unsigned char* d_src, const RemapSample* d_lut,
                           unsigned char* d_out, int srcW, int srcH, int outW, int outH)
{
    const dim3 block(kBlock2D, kBlock2D);
    const dim3 grid((outW + kBlock2D - 1) / kBlock2D, (outH + kBlock2D - 1) / kBlock2D);
    remap_bilinear_kernel<<<grid, block>>>(d_src, d_lut, d_out, srcW, srcH, outW, outH);
    CUDA_CHECK_LAST_ERROR("remap_bilinear_kernel launch");
}

void launch_bev_compose(const unsigned char* d_front, const unsigned char* d_left,
                        const unsigned char* d_right, const unsigned char* d_rear,
                        unsigned char* d_bev, unsigned char* d_coverage)
{
    const dim3 block(kBlock2D, kBlock2D);
    const dim3 grid((kBevW + kBlock2D - 1) / kBlock2D, (kBevH + kBlock2D - 1) / kBlock2D);
    bev_compose_kernel<<<grid, block>>>(d_front, d_left, d_right, d_rear, d_bev, d_coverage);
    CUDA_CHECK_LAST_ERROR("bev_compose_kernel launch");
}
