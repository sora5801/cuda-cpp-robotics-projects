// ===========================================================================
// kernels.cu — GPU kernels for project 01.18 (Depth completion: sparse
//              LiDAR + RGB -> dense depth)
//
// Four kernels, three GPU patterns (kernels.cuh gives each one's full
// doc-comment; this file's job is the WHY behind the mapping):
//
//   1. project_zbuffer_kernel   — a SCATTER: one thread per INPUT point
//      (not per output pixel!). Threads write to unpredictable, possibly
//      colliding output locations, which is exactly why it needs an atomic.
//   2. compute_conductance_kernel — a MAP/STENCIL hybrid: one thread per
//      pixel, reading 2 forward neighbors, writing 2 outputs.
//   3. diffusion_step_kernel    — a STENCIL, ping-pong style: one thread
//      per pixel, reading its 4-neighborhood from the PREVIOUS iteration's
//      buffer, writing the new value to a SEPARATE buffer (07.09's
//      jump-flooding ping-pong precedent, cited below).
//   4. idw_kernel               — a bounded SEARCH: one thread per pixel,
//      each scanning a (2R+1)x(2R+1) window — the same "stencil with a
//      big window" shape as 01.11's bilateral kernels (cited below), just
//      searching for VALID samples instead of averaging every neighbor.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cmath>     // sqrtf, expf, fabsf — device math (nvcc provides device overloads)
#include <cstdint>

// ---------------------------------------------------------------------------
// encode_depth_for_zbuffer — order-preserving float->uint32 map, so
// atomicMin (integer-only on this hardware) can race many threads onto one
// pixel and keep the SMALLEST depth (the file header explains why smaller
// depth = nearer = the physically correct winner at an occlusion boundary).
//
// General trick (documented for completeness, per CLAUDE.md's "no black
// boxes" rule): reinterpreting an IEEE-754 float's bits as an unsigned
// integer preserves ordering for all POSITIVE floats already (the exponent
// occupies the high bits and dominates the comparison). Negative floats
// need one more step — flip every bit if negative, flip only the sign bit
// if non-negative — so the full two-branch version would read:
//     uint32_t bits = __float_as_uint(v);
//     return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
// Every depth this project encodes is a LiDAR return in front of the
// camera (Pcam.z > 0 is checked before this is ever called), so the
// negative branch never triggers here — we use the simple positive-only
// form and rely on the caller's z>0 guard, rather than pay for a branch
// that can never be taken. THEORY.md "Numerical considerations" discusses
// the (nonexistent, for this project) risk of a NaN or zero depth reaching
// this function.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t encode_depth_for_zbuffer(float depth_m)
{
    return __float_as_uint(depth_m);   // positive-float bit pattern IS the ordering (see above)
}

__device__ __forceinline__ float decode_depth_from_zbuffer(uint32_t bits)
{
    return __uint_as_float(bits);      // exact inverse of the encode above
}

// ===========================================================================
// 1) PROJECTION + Z-BUFFER
//
// Thread-to-data mapping: thread i owns LiDAR point i (a 1-D grid over
// n_pts) — a SCATTER, the opposite of the per-pixel maps elsewhere in this
// file. Each thread computes ONE output location (its point's pixel) and
// writes there; many threads can target the SAME pixel (two LiDAR returns
// whose rays, from the LIDAR's slightly different origin, happen to project
// near the same camera pixel — routine at object silhouettes), which is
// exactly the race atomicMin resolves.
//
// Math: P_cam = R * P_lidar + t (kTCameraLidar, the SAME Rigid3 convention
// 01.17 solves for). Depth is P_cam.z (the pinhole z-buffer convention, file
// header); pixel = (fx*x/z + cx, fy*y/z + cy). Points with z <= 0 (behind
// the camera — physically impossible for a real forward-looking rig, but a
// cheap defensive guard costs nothing) or landing outside the image are
// dropped; points beyond kMaxDepthM are dropped too (that is the LiDAR's own
// maximum-range cutoff, not a camera limitation — see THEORY.md).
// ---------------------------------------------------------------------------
__global__ void project_zbuffer_kernel(const LidarPointF* __restrict__ d_pts,
                                       int n_pts,
                                       Rigid3 t_camera_lidar,
                                       uint32_t* __restrict__ d_encoded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's LiDAR point index
    if (i >= n_pts) return;                                 // ragged-tail guard

    const LidarPointF p = d_pts[i];   // meters, LIDAR frame (kernels.cuh LidarPointF)

    // Rigid transform: P_cam = R*P_lidar + t, R row-major (kernels.cuh Rigid3).
    // t_camera_lidar arrived as a KERNEL PARAMETER (file header explains why:
    // the constexpr kTCameraLidar global has no device-side storage).
    const float* R = t_camera_lidar.R;
    const float xc = R[0] * p.x + R[1] * p.y + R[2] * p.z + t_camera_lidar.t[0];
    const float yc = R[3] * p.x + R[4] * p.y + R[5] * p.z + t_camera_lidar.t[1];
    const float zc = R[6] * p.x + R[7] * p.y + R[8] * p.z + t_camera_lidar.t[2];

    // Depth is Pcam.z (NOT Euclidean range) — the pinhole/z-buffer convention
    // this whole project uses (file header, kernels.cuh). Behind-camera and
    // beyond-max-range points contribute no evidence; nothing to write.
    if (zc <= 0.0f || zc > kMaxDepthM) return;

    const float inv_z = 1.0f / zc;
    const float u = kFx * xc * inv_z + kCx;
    const float v = kFy * yc * inv_z + kCy;

    // Round-to-nearest-pixel (floorf(x+0.5) is the standard, branch-free
    // round-half-up for non-negative x; pixel coords here are always >= a
    // few units inside a 160x120 image so the "round toward zero on
    // negatives" edge case never applies).
    const int px = static_cast<int>(floorf(u + 0.5f));
    const int py = static_cast<int>(floorf(v + 0.5f));
    if (px < 0 || px >= kImageWidth || py < 0 || py >= kImageHeight) return;

    const int idx = py * kImageWidth + px;
    // atomicMin on the ENCODED depth: many threads may race here (see file
    // header); the hardware performs a true read-modify-write, so no two
    // colliding writes can both "win" and no update is ever lost — the
    // pixel ends up holding exactly the smallest depth any thread offered.
    atomicMin(&d_encoded[idx], encode_depth_for_zbuffer(zc));
}

void launch_project_zbuffer(const LidarPointF* d_pts, int n_pts, uint32_t* d_encoded)
{
    const int block = 256;                          // warp multiple, standard default (see 08.01/33.01)
    const int grid = (n_pts + block - 1) / block;    // one thread per point, no grid-stride needed (n_pts is small: ~hundreds)
    // kTCameraLidar is read HERE, in host code (where a constexpr global has
    // perfectly ordinary storage), and handed to the kernel as an argument.
    project_zbuffer_kernel<<<grid, block>>>(d_pts, n_pts, kTCameraLidar, d_encoded);
    CUDA_CHECK_LAST_ERROR("project_zbuffer_kernel launch");
}

// ===========================================================================
// 2) CONDUCTANCE — Perona-Malik edge-stopping weight from the RGB (here,
// grayscale) guidance image: g = exp(-(grad/K)^2), THEORY.md "The math"
// derives this from the anisotropic-diffusion PDE. g -> 1 where the image
// is flat (diffuse freely, homogeneous region) and g -> 0 where the image
// has a strong edge (stop diffusing — "the edges-coincide prior" in
// action). Two conductances per pixel (to its RIGHT and BELOW neighbor)
// are enough to cover every one of the 4 axis-aligned edges in the grid
// exactly once each (pixel (x,y)'s LEFT neighbor's g_right[x-1,y] IS the
// edge between them — diffusion_step_kernel reads it from there).
//
// Thread-to-data mapping: one thread per pixel, 2-D grid (the same shape
// 01.11's bilateral kernels use for a per-pixel stencil).
// ---------------------------------------------------------------------------
// max_channel_diff — the color-edge strength between two pixels at plane
// offsets `a_idx`/`b_idx` in a PLANAR [3*kImagePixels] RGB buffer (file
// header: max absolute per-channel difference, not Euclidean distance).
__device__ __forceinline__ float max_channel_diff(const float* __restrict__ rgb, int a_idx, int b_idx)
{
    const float dr = fabsf(rgb[a_idx]                     - rgb[b_idx]);
    const float dg = fabsf(rgb[a_idx + kImagePixels]      - rgb[b_idx + kImagePixels]);
    const float db = fabsf(rgb[a_idx + 2 * kImagePixels]  - rgb[b_idx + 2 * kImagePixels]);
    return fmaxf(dr, fmaxf(dg, db));
}

__global__ void compute_conductance_kernel(const float* __restrict__ d_rgb,
                                           float* __restrict__ d_g_right,
                                           float* __restrict__ d_g_down)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kImageWidth || y >= kImageHeight) return;

    const int idx = y * kImageWidth + x;
    const float inv_k2 = 1.0f / (kConductanceK * kConductanceK);

    // Right edge: undefined (no neighbor) at the last column -> conductance
    // 0, which diffusion_step_kernel then correctly reads as "no flow
    // across the image boundary" (a natural/Neumann boundary, THEORY.md
    // "Numerical considerations" explains why zero-flux is the honest
    // choice here rather than wrapping or clamping).
    if (x + 1 < kImageWidth) {
        const float diff = max_channel_diff(d_rgb, idx, idx + 1);
        d_g_right[idx] = expf(-(diff * diff) * inv_k2);
    } else {
        d_g_right[idx] = 0.0f;
    }

    // Down edge: same story for the last row.
    if (y + 1 < kImageHeight) {
        const float diff = max_channel_diff(d_rgb, idx, idx + kImageWidth);
        d_g_down[idx] = expf(-(diff * diff) * inv_k2);
    } else {
        d_g_down[idx] = 0.0f;
    }
}

void launch_compute_conductance(const float* d_rgb, float* d_g_right, float* d_g_down)
{
    const dim3 block(16, 16);   // 256 threads/block, the standard 2-D stencil default (01.11 precedent)
    const dim3 grid((kImageWidth + block.x - 1) / block.x,
                    (kImageHeight + block.y - 1) / block.y);
    compute_conductance_kernel<<<grid, block>>>(d_rgb, d_g_right, d_g_down);
    CUDA_CHECK_LAST_ERROR("compute_conductance_kernel launch");
}

// ===========================================================================
// 3) DIFFUSION STEP — one forward-Euler iteration of
//        D_new(x,y) = D(x,y) + dt * [ g_left  * (D(x-1,y) - D(x,y))
//                                    + g_right * (D(x+1,y) - D(x,y))
//                                    + g_up    * (D(x,y-1) - D(x,y))
//                                    + g_down  * (D(x,y+1) - D(x,y)) ]
// where g_left(x,y) = g_right(x-1,y) and g_up(x,y) = g_down(x,y-1) — each
// edge's conductance is stored ONCE (by compute_conductance_kernel) and
// read from both sides here, so the diffusion is symmetric (flow x->y
// equals flow y->x) by CONSTRUCTION, not by luck.
//
// PING-PONG discipline (the 07.09 jump-flooding precedent, cited): read
// EVERY neighbor from d_in, write ONLY to d_out. If this kernel wrote back
// into the same buffer it read, a thread could read a neighbor's
// ALREADY-UPDATED value (from a different thread that happened to finish
// first) instead of the previous iteration's value — a Gauss-Seidel-like
// race whose result depends on scheduling order, i.e. NON-DETERMINISTIC.
// Two buffers make every iteration a clean, order-independent Jacobi
// update; launch_diffusion (below) owns swapping them each iteration.
//
// DIRICHLET ANCHORING: pixels with a valid sparse LiDAR sample (d_anchor
// != kInvalidDepth) are NOT diffused — they simply copy their anchor value
// straight through, every iteration. This is what "sparse samples are
// Dirichlet boundary conditions" means operationally: the PDE only ever
// gets to move the UNKNOWN pixels, and it re-derives every unknown pixel
// from scratch each step from ITS neighbors (which may themselves be
// anchors) — after enough iterations, influence has propagated from every
// anchor to every unknown pixel it can reach.
// ---------------------------------------------------------------------------
__global__ void diffusion_step_kernel(const float* __restrict__ d_in,
                                      const float* __restrict__ d_g_right,
                                      const float* __restrict__ d_g_down,
                                      const float* __restrict__ d_anchor,
                                      float* __restrict__ d_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kImageWidth || y >= kImageHeight) return;
    const int idx = y * kImageWidth + x;

    const float anchor = d_anchor[idx];
    if (anchor != kInvalidDepth) {
        d_out[idx] = anchor;    // Dirichlet: this pixel's value is FIXED, not diffused
        return;
    }

    const float center = d_in[idx];
    float flow = 0.0f;   // sum of g_neighbor * (D_neighbor - D_center) over all 4 neighbors

    if (x > 0) {
        const float g = d_g_right[idx - 1];              // the LEFT neighbor's rightward edge IS this edge
        flow += g * (d_in[idx - 1] - center);
    }
    if (x + 1 < kImageWidth) {
        const float g = d_g_right[idx];
        flow += g * (d_in[idx + 1] - center);
    }
    if (y > 0) {
        const float g = d_g_down[idx - kImageWidth];      // the UP neighbor's downward edge IS this edge
        flow += g * (d_in[idx - kImageWidth] - center);
    }
    if (y + 1 < kImageHeight) {
        const float g = d_g_down[idx];
        flow += g * (d_in[idx + kImageWidth] - center);
    }

    // Forward-Euler step. kDiffusionDt is checked against the CFL-style
    // stability bound dt <= 1/(sum of neighbor conductances) <= 0.25 at
    // startup in main.cu (the STABILITY gate) — this line trusts that
    // check rather than re-deriving it per pixel per iteration.
    d_out[idx] = center + kDiffusionDt * flow;
}

// seed_init_kernel — build the diffusion PDE's initial condition: anchors
// (d_sparse != kInvalidDepth) keep their measured value, every unknown
// pixel starts at `seed` (the caller-supplied mean of the valid sparse
// samples — see launch_diffusion's doc-comment in kernels.cuh for why a
// plausible depth beats an out-of-range sentinel here). A trivial one-line
// MAP kernel — its own kernel rather than folded into diffusion_step_kernel
// so the FIRST iteration of the loop below is identical to every other
// iteration (no special-cased "iteration 0" branch to keep in sync).
__global__ void seed_init_kernel(const float* __restrict__ d_sparse, float seed, float* __restrict__ d_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kImagePixels) return;
    const float v = d_sparse[i];
    d_out[i] = (v == kInvalidDepth) ? seed : v;
}

// launch_diffusion — owns the WHOLE densification: seed the PDE's initial
// condition, compute conductance (once), then kDiffusionIters ping-ponged
// steps, then hands the caller the final field. Scratch buffers are
// allocated/freed HERE so main.cu's orchestration stays about DATA, not
// about this method's internal iteration count.
void launch_diffusion(const float* d_sparse, const float* d_rgb, float unknown_seed, float* d_out)
{
    const size_t bytes = static_cast<size_t>(kImagePixels) * sizeof(float);

    float *d_g_right = nullptr, *d_g_down = nullptr;
    float *d_ping = nullptr, *d_pong = nullptr;
    CUDA_CHECK(cudaMalloc(&d_g_right, bytes));
    CUDA_CHECK(cudaMalloc(&d_g_down,  bytes));
    CUDA_CHECK(cudaMalloc(&d_ping,    bytes));
    CUDA_CHECK(cudaMalloc(&d_pong,    bytes));

    launch_compute_conductance(d_rgb, d_g_right, d_g_down);

    const int block1d = 256;
    const int grid1d = (kImagePixels + block1d - 1) / block1d;
    seed_init_kernel<<<grid1d, block1d>>>(d_sparse, unknown_seed, d_ping);
    CUDA_CHECK_LAST_ERROR("seed_init_kernel launch");

    const dim3 block(16, 16);
    const dim3 grid((kImageWidth + block.x - 1) / block.x,
                    (kImageHeight + block.y - 1) / block.y);

    float* cur = d_ping;
    float* nxt = d_pong;
    for (int it = 0; it < kDiffusionIters; ++it) {
        diffusion_step_kernel<<<grid, block>>>(cur, d_g_right, d_g_down, d_sparse, nxt);
        CUDA_CHECK_LAST_ERROR("diffusion_step_kernel launch");
        float* tmp = cur; cur = nxt; nxt = tmp;   // swap: next iteration reads what we just wrote
    }

    CUDA_CHECK(cudaMemcpy(d_out, cur, bytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_g_right));
    CUDA_CHECK(cudaFree(d_g_down));
    CUDA_CHECK(cudaFree(d_ping));
    CUDA_CHECK(cudaFree(d_pong));
}

// ===========================================================================
// 4) IDW BASELINE — inverse-distance-weighted interpolation, blind to the
// RGB image (the "no prior" baseline the edge-aware method must beat — the
// project's whole reason-to-exist gate compares against this).
//
// Thread-to-data mapping: one thread per OUTPUT pixel, 2-D grid. Each
// thread does its OWN bounded window search — the same "big stencil"
// shape 01.11's bilateral kernels use, just testing validity instead of
// always-accumulate.
//
// Anchors pass through exactly (an r=0 sample has "infinite" weight —
// handled as an explicit early-return so we never divide by the r=0
// distance). Pixels with NO valid sample anywhere in the window (rare at
// this project's density and radius, but possible near the image border
// where the window is clipped) fall back to 0.0f — a documented, honest
// "we have no information here" value rather than a silently wrong one;
// main.cu's evaluation gates report how often this fallback fires.
// ---------------------------------------------------------------------------
__global__ void idw_kernel(const float* __restrict__ d_sparse, float* __restrict__ d_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kImageWidth || y >= kImageHeight) return;
    const int idx = y * kImageWidth + x;

    const float here = d_sparse[idx];
    if (here != kInvalidDepth) {
        d_out[idx] = here;   // this pixel IS a sample — IDW should reproduce it exactly (input_fidelity gate)
        return;
    }

    float wsum = 0.0f, vsum = 0.0f;
    const int x0 = x - kIdwRadiusPx < 0 ? 0 : x - kIdwRadiusPx;
    const int x1 = x + kIdwRadiusPx >= kImageWidth ? kImageWidth - 1 : x + kIdwRadiusPx;
    const int y0 = y - kIdwRadiusPx < 0 ? 0 : y - kIdwRadiusPx;
    const int y1 = y + kIdwRadiusPx >= kImageHeight ? kImageHeight - 1 : y + kIdwRadiusPx;

    for (int sy = y0; sy <= y1; ++sy) {
        for (int sx = x0; sx <= x1; ++sx) {
            const float v = d_sparse[sy * kImageWidth + sx];
            if (v == kInvalidDepth) continue;
            const float dx = static_cast<float>(sx - x);
            const float dy = static_cast<float>(sy - y);
            const float dist = sqrtf(dx * dx + dy * dy);
            // dist > 0 always here (the dist==0 / here-is-a-sample case
            // already returned above), so this power is always well
            // defined — no epsilon-guard needed for kIdwPower == 2.
            const float w = 1.0f / powf(dist, kIdwPower);
            wsum += w;
            vsum += w * v;
        }
    }

    d_out[idx] = (wsum > 0.0f) ? (vsum / wsum) : 0.0f;   // documented fallback, see file header
}

void launch_idw(const float* d_sparse, float* d_out)
{
    const dim3 block(16, 16);
    const dim3 grid((kImageWidth + block.x - 1) / block.x,
                    (kImageHeight + block.y - 1) / block.y);
    idw_kernel<<<grid, block>>>(d_sparse, d_out);
    CUDA_CHECK_LAST_ERROR("idw_kernel launch");
}
