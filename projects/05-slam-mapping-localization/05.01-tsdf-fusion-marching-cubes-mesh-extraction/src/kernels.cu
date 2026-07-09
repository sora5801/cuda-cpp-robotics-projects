// ===========================================================================
// kernels.cu — GPU implementation for project 05.01
//              TSDF fusion (KinectFusion clone) + marching-cubes mesh
//              extraction
//
// The big idea
// ------------
// Dense mapping asks 2 million voxels the same two questions:
//   INTEGRATION: "camera, how far am I from the surface you just saw?"
//     One thread per VOXEL. Each voxel projects itself into the depth
//     image, compares the depth found there with its own distance along
//     the optical axis, and folds the (truncated) difference into its
//     running weighted average. No voxel ever touches another voxel —
//     a pure parallel MAP, the same thread-per-problem shape as 33.01,
//     just over a 3-D grid instead of a batch.
//   MESHING: "cell, does the surface pass through you, and where?"
//     One thread per 2x2x2 CELL. Classify the 8 corners against the zero
//     level set, look the pattern up in the precomputed 256-case table
//     (mc_tables.h, in __constant__ memory), interpolate the triangle
//     vertices on the cut edges, and APPEND them to a global buffer.
//     Appending from thousands of threads at once is this project's new
//     pattern: atomicAdd on a counter hands each triangle a unique slot.
//
// What is NEW here beyond 33.01/09.01/07.09/08.01:
//   * a 3-D grid mapped to a 1-D launch (index arithmetic instead of 07.09's
//     2-D blocks — with x the fastest axis both give coalesced access; 1-D
//     keeps the ragged-edge guard to one comparison);
//   * the ATOMIC-APPEND output pattern for variable-length results. The
//     honest alternative is TWO-PASS: pass 1 counts triangles per cell,
//     an exclusive prefix-sum (scan) turns counts into stable offsets,
//     pass 2 writes — deterministic ORDER and no atomic contention, at the
//     price of classifying every cell twice plus a scan. Production
//     libraries (VTK-m, NVIDIA's MC sample) do exactly that; we teach the
//     one-pass version because it is the minimal correct pattern and the
//     nondeterminism it introduces is ORDER-only (the triangle SET and
//     COUNT stay deterministic — verified against the CPU recount every
//     run). Exercise 4 upgrades this to the scan version.
//   * __constant__ memory used for a lookup TABLE all threads index
//     divergently-but-narrowly: warps hitting the same case row get a
//     broadcast; different rows serialize, but the table is 4 KiB and hot
//     in the constant cache — the right tool between 09.01's pure-uniform
//     reads and 07.09's scattered global reads.
//
// All layouts and constants come from kernels.cuh — the single source
// shared with the CPU twin; the integration function below is a deliberate
// line-by-line twin of tsdf_integrate_cpu (explicit fmaf's on BOTH sides
// make the two paths bit-identical, not merely close — kernels.cuh
// §determinism).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "mc_tables.h"              // the 256-case table (provenance in that header)
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Device copies of the marching-cubes tables, in __constant__ memory.
// Statically initialized from the same macros the CPU recount uses — one
// authoritative blob, two homes, zero drift (mc_tables.h explains).
// 256*16 + 2*12 bytes ≈ 4.1 KiB of the 64 KiB constant bank.
// ---------------------------------------------------------------------------
__constant__ signed char c_tri_table[256][16] = MC_TRI_TABLE_INITIALIZER;
__constant__ signed char c_edge_corner_a[12]  = MC_EDGE_CORNER_A_INITIALIZER;
__constant__ signed char c_edge_corner_b[12]  = MC_EDGE_CORNER_B_INITIALIZER;

// Repo default launch geometry: 256-thread 1-D blocks (works well across
// sm_75..sm_89; both kernels are memory-bound, so occupancy — not block
// shape — is what matters, and 256 gives full occupancy headroom).
static constexpr int kThreads = 256;

// ===========================================================================
// Kernel 0: reset the volume to the defined "never observed" state.
// One thread per voxel; a pure fill (same 3-line shape as 07.09's clear).
// tsdf = +1 ("far") is convention only — it stays meaningless until
// weight > 0; weight = 0 is the actual validity flag.
// ===========================================================================
__global__ void volume_clear_kernel(float* __restrict__ tsdf,    // [kVolN^3] OUT
                                    float* __restrict__ weight)  // [kVolN^3] OUT
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= kVolN * kVolN * kVolN) return;
    tsdf[v]   = 1.0f;
    weight[v] = 0.0f;
}

// ===========================================================================
// Kernel 1: TSDF integration — the KinectFusion measurement update.
//
// Thread-to-data mapping: thread v = blockIdx.x*blockDim.x + threadIdx.x
// owns voxel v; (ix,iy,iz) are decoded with x FASTEST, so consecutive
// threads read/write consecutive tsdf/weight addresses — fully coalesced
// (the kernels.cuh layout contract).
//
// Memory spaces per thread:
//   registers : the voxel's world/camera coordinates and the update math;
//   global    : depth[vi*W+ui] — a GATHER whose locality mirrors the
//               geometry: neighboring voxels project to neighboring pixels,
//               so warps touch compact depth regions and L2 serves most of
//               it (not perfectly coalesced — the honest cost of the
//               voxel-centric formulation);
//               tsdf[v]/weight[v] — one coalesced read + write each.
//   K, T      : kernel arguments (by value) — they live in the driver-managed
//               constant/param space; every thread reads the same values, a
//               broadcast (the 09.01 access-pattern lesson).
//
// Divergence: voxels behind the camera / outside the image / occluded
// beyond the truncation band exit early. Whole warps of far-away voxels
// exit together (spatial coherence again), so the guards cost little.
//
// DETERMINISM: every multiply-add below is an EXPLICIT fmaf(). nvcc would
// contract a*b+c into fma anyway (and MSVC would NOT) — writing it out
// makes the GPU and CPU twins execute identical IEEE operations, so the
// rounded pixel indices — where an ulp of drift could select a DIFFERENT
// depth sample and blow a voxel's value up by centimeters — match exactly
// (kernels.cuh §determinism; THEORY.md §numerics tells the pixel-boundary
// story in full).
// ===========================================================================
__global__ void tsdf_integrate_kernel(
    const float* __restrict__ depth,   // [K.height*K.width] z-depth (m); <=0 invalid
    Intrinsics                K,       // pinhole intrinsics (px)
    PoseRt                    T,       // T_cam_world: p_cam = R*p_world + t
    float*       __restrict__ tsdf,    // [kVolN^3] IN/OUT running TSDF (units of mu)
    float*       __restrict__ weight)  // [kVolN^3] IN/OUT running weights
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= kVolN * kVolN * kVolN) return;          // ragged-tail guard

    // Decode the flat index; x fastest (matches the linear layout, so this
    // is pure integer arithmetic, no lookup).
    const int ix = v % kVolN;
    const int iy = (v / kVolN) % kVolN;
    const int iz = v / (kVolN * kVolN);

    // Voxel CENTER in world coordinates (m). The +0.5 centers the sample in
    // the voxel — sampling corners instead would bias every distance by
    // half a voxel diagonal.
    const float px = fmaf(static_cast<float>(ix) + 0.5f, kVoxelSize, kVolOriginX);
    const float py = fmaf(static_cast<float>(iy) + 0.5f, kVoxelSize, kVolOriginY);
    const float pz = fmaf(static_cast<float>(iz) + 0.5f, kVoxelSize, kVolOriginZ);

    // World -> camera: p_cam = R * p_world + t (row-major R; 9 fmaf's).
    const float xc = fmaf(T.r[0], px, fmaf(T.r[1], py, fmaf(T.r[2], pz, T.t[0])));
    const float yc = fmaf(T.r[3], px, fmaf(T.r[4], py, fmaf(T.r[5], pz, T.t[1])));
    const float zc = fmaf(T.r[6], px, fmaf(T.r[7], py, fmaf(T.r[8], pz, T.t[2])));

    // Behind (or in the plane of) the camera: this frame cannot see it.
    if (zc <= 0.0f) return;

    // Pinhole projection to pixel coordinates, then round to the NEAREST
    // pixel center (floor(x+0.5), spelled identically in the CPU twin).
    // Nearest-neighbor depth lookup is the classic KinectFusion choice —
    // bilinear would blur across depth discontinuities and invent surface
    // where an object ends (THEORY.md §algorithm).
    const float u_px = fmaf(K.fx, xc / zc, K.cx);
    const float v_px = fmaf(K.fy, yc / zc, K.cy);
    const int ui = static_cast<int>(floorf(u_px + 0.5f));
    const int vi = static_cast<int>(floorf(v_px + 0.5f));
    if (ui < 0 || ui >= K.width || vi < 0 || vi >= K.height) return;  // outside the image

    const float d = depth[vi * K.width + ui];   // the observed surface (m along +z)
    if (d <= 0.0f) return;                      // no return at this pixel (sky)

    // PROJECTIVE signed distance: how far this voxel sits in FRONT of the
    // observed surface, measured along the optical axis (not along the ray,
    // and not the true Euclidean distance — the standard KinectFusion
    // approximation; its 1/cos(incidence) bias is quantified against the
    // analytic ground truth every run, THEORY.md §algorithm).
    const float sdf = d - zc;

    // Truncation, asymmetric on purpose:
    //   sdf < -mu : the voxel is hidden BEHIND the surface — this frame
    //               says nothing about it (it could be inside the object or
    //               behind it entirely). Integrating it would carve away
    //               geometry the camera never saw. Skip.
    //   sdf > +mu : confidently free space — clamp to +1 and integrate;
    //               free-space votes are what carve out noise blobs.
    if (sdf < -kTruncation) return;
    const float f = fminf(1.0f, sdf * (1.0f / kTruncation));  // to units of mu, in [-1, +1]

    // The running weighted average (per-frame observation weight = 1):
    //     F <- (F*W + f) / (W + 1),   W <- min(W + 1, kMaxWeight)
    // With W capped, old evidence cannot outvote change forever — every new
    // frame keeps at least 1/(kMaxWeight+1) influence. When W == 0 the
    // fmaf multiplies the meaningless initial F by exactly 0, so the first
    // observation lands unpolluted.
    const float w  = weight[v];
    const float wn = w + 1.0f;
    tsdf[v]   = fmaf(tsdf[v], w, f) / wn;
    weight[v] = fminf(wn, kMaxWeight);
}

// ===========================================================================
// Kernel 2: marching cubes — classify cells, emit triangles (atomic append).
//
// Thread-to-data mapping: thread c = blockIdx.x*blockDim.x + threadIdx.x
// owns the CELL whose min corner is voxel (ix,iy,iz), decoded x-fastest
// over the (kVolN-1)^3 cell grid (cells straddle 8 voxels, so there is one
// fewer per axis).
//
// Memory spaces per thread:
//   global    : 8 tsdf + 8 weight corner loads (neighbors in x are
//               coalesced; y/z neighbors land kVolN and kVolN^2 floats
//               apart — L2 catches most of it, same honest stencil story
//               as 07.09);
//               triangle writes — 9 floats per triangle at an
//               atomicAdd-reserved offset (append order = atomic order =
//               nondeterministic; SET and COUNT deterministic);
//   constant  : the case table (broadcast for warps on the same case);
//   registers : corner values, case index, interpolation math.
//
// Divergence, honestly: MOST cells are empty (all-outside or unobserved) and
// their whole warps retire after the classification loads. Warps containing
// surface cells diverge on the per-case triangle loop (0–5 triangles) —
// intrinsic to sparse surfaces in dense volumes; the two-pass/compaction
// variants exist precisely to regroup that work (header note above).
// ===========================================================================
__global__ void marching_cubes_kernel(
    const float* __restrict__ tsdf,      // [kVolN^3] fused TSDF (units of mu)
    const float* __restrict__ weight,    // [kVolN^3] weights (0 = unobserved)
    int                       max_tris,  // capacity of tri_verts (triangles)
    float*       __restrict__ tri_verts, // [max_tris*9] OUT packed triangles (m, world)
    int*         __restrict__ tri_count) // [1] IN/OUT global append counter
{
    constexpr int kCells = kVolN - 1;                       // cells per axis
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kCells * kCells * kCells) return;              // ragged-tail guard

    const int ix = c % kCells;                              // cell min corner, x fastest
    const int iy = (c / kCells) % kCells;
    const int iz = c / (kCells * kCells);

    // The 8 corner voxels, in the mc_tables.h numbering (0..3 bottom ring
    // CCW from (0,0), 4..7 the ring above). Offsets in (x,y,z):
    const int dx[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    const int dy[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    const int dz[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

    float f[8];               // corner TSDF values (units of mu)
    int   cubeindex = 0;      // bit i set = corner i inside (tsdf < 0)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int vi = ((iz + dz[i]) * kVolN + (iy + dy[i])) * kVolN + (ix + dx[i]);
        // Any unobserved corner disqualifies the cell: a zero crossing
        // against the +1 INITIAL value is not a surface, it is the edge of
        // what the cameras saw. (Skipping here is what keeps the mesh free
        // of a shell around the observed region.)
        if (weight[vi] == 0.0f) return;
        f[i] = tsdf[vi];
        cubeindex |= (f[i] < 0.0f) ? (1 << i) : 0;   // inside = negative = behind the surface
    }
    if (cubeindex == 0 || cubeindex == 255) return;  // fully outside / fully inside: no surface

    // World position of corner 0 (this cell's min voxel CENTER — the grid
    // of cell corners is the grid of voxel centers, where the TSDF samples
    // actually live).
    const float x0 = fmaf(static_cast<float>(ix) + 0.5f, kVoxelSize, kVolOriginX);
    const float y0 = fmaf(static_cast<float>(iy) + 0.5f, kVoxelSize, kVolOriginY);
    const float z0 = fmaf(static_cast<float>(iz) + 0.5f, kVoxelSize, kVolOriginZ);

    // Walk this case's triangle list (-1 terminated, at most 5 triangles).
    const signed char* row = c_tri_table[cubeindex];
    for (int t = 0; row[t] != -1; t += 3) {
        // Reserve a slot FIRST: atomicAdd returns the old value, so each
        // triangle gets a unique index even with thousands of concurrent
        // appenders. The counter keeps counting past max_tris (the host
        // checks it against the capacity); only the WRITE is skipped.
        const int slot = atomicAdd(tri_count, 1);
        if (slot >= max_tris) continue;

        float* out = tri_verts + static_cast<size_t>(slot) * 9;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int e = row[t + k];                 // which of the 12 edges
            const int a = c_edge_corner_a[e];         // its two corners
            const int b = c_edge_corner_b[e];

            // Linear interpolation to the zero crossing along edge a->b:
            //     s = f_a / (f_a - f_b)   in (0, 1)
            // Safe by construction: the table only lists sign-change edges,
            // so f_a and f_b straddle zero ("<" vs ">=") and the
            // denominator's magnitude is |f_a| + |f_b| > 0.
            const float fa = f[a];
            const float s  = fa / (fa - f[b]);

            // Vertex = corner_a + s * (corner_b - corner_a), built per axis
            // from the integer corner offsets (each 0 or 1 voxel step).
            const float ax = static_cast<float>(dx[a]), bx = static_cast<float>(dx[b]);
            const float ay = static_cast<float>(dy[a]), by = static_cast<float>(dy[b]);
            const float az = static_cast<float>(dz[a]), bz = static_cast<float>(dz[b]);
            out[k * 3 + 0] = fmaf(fmaf(s, bx - ax, ax), kVoxelSize, x0);
            out[k * 3 + 1] = fmaf(fmaf(s, by - ay, ay), kVoxelSize, y0);
            out[k * 3 + 2] = fmaf(fmaf(s, bz - az, az), kVoxelSize, z0);
        }
    }
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================

void launch_volume_clear(float* d_tsdf, float* d_weight)
{
    if (!d_tsdf || !d_weight) {
        std::fprintf(stderr, "launch_volume_clear: null volume pointer\n");
        std::exit(EXIT_FAILURE);
    }
    const int total = kVolN * kVolN * kVolN;
    volume_clear_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(d_tsdf, d_weight);
    CUDA_CHECK_LAST_ERROR("volume_clear_kernel launch");
}

void launch_tsdf_integrate(const float* d_depth, Intrinsics K,
                           PoseRt T_cam_world,
                           float* d_tsdf, float* d_weight)
{
    if (!d_depth || !d_tsdf || !d_weight || K.width < 1 || K.height < 1) {
        std::fprintf(stderr, "launch_tsdf_integrate: invalid arguments (%dx%d image)\n",
                     K.width, K.height);
        std::exit(EXIT_FAILURE);
    }
    const int total = kVolN * kVolN * kVolN;    // one thread per voxel
    tsdf_integrate_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(
        d_depth, K, T_cam_world, d_tsdf, d_weight);
    CUDA_CHECK_LAST_ERROR("tsdf_integrate_kernel launch");
}

void launch_marching_cubes(const float* d_tsdf, const float* d_weight,
                           int max_tris, float* d_tri_verts, int* d_tri_count)
{
    if (!d_tsdf || !d_weight || !d_tri_verts || !d_tri_count || max_tris < 1) {
        std::fprintf(stderr, "launch_marching_cubes: invalid arguments (max_tris=%d)\n",
                     max_tris);
        std::exit(EXIT_FAILURE);
    }
    const int cells = (kVolN - 1) * (kVolN - 1) * (kVolN - 1);   // one thread per cell
    marching_cubes_kernel<<<(cells + kThreads - 1) / kThreads, kThreads>>>(
        d_tsdf, d_weight, max_tris, d_tri_verts, d_tri_count);
    CUDA_CHECK_LAST_ERROR("marching_cubes_kernel launch");
}
