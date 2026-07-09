// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 05.01
//                     TSDF fusion + marching-cubes mesh extraction
//
// Two jobs in this project (both declared in kernels.cuh):
//
//   1. tsdf_integrate_cpu — the ORACLE TWIN of the integration kernel: same
//      projection, same truncation, same weighted average, sequential over
//      voxels instead of one thread each. main.cu fuses the same 4-frame
//      subset through both paths into separate volumes and requires
//      voxel-wise agreement within abs tol 1e-5 — the §5 GPU-vs-CPU gate.
//      It is also the honest timing baseline: "the CPU takes ~a second per
//      frame where the GPU takes ~a millisecond" is measured here, not
//      asserted.
//
//   2. marching_cubes_count_cpu — a COUNT-ONLY re-run of the marching-cubes
//      classification over the (downloaded) GPU volume. Because it reads
//      the exact same float values and applies the exact same comparisons
//      and the same table, its total must equal the GPU's atomic counter
//      EXACTLY — an order-independent check that no cell was dropped,
//      double-counted, or mis-classified by the atomic-append machinery.
//      (Geometry itself is checked differently: main.cu measures every
//      emitted vertex against the ANALYTIC scene SDF — a stronger oracle
//      than any CPU re-implementation could be. THEORY.md §verification.)
//
// The integration function below is a line-by-line twin of
// tsdf_integrate_kernel in kernels.cu — deliberate, documented duplication.
// Every multiply-add is an explicit std::fmaf matching the kernel's fmaf,
// so both paths execute the same IEEE-754 operations in the same order and
// the rounded pixel lookups are bit-identical (kernels.cuh §determinism —
// diff the two files: only the thread decode vs. the for-loops differ).
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared constants, layouts, struct definitions
#include "mc_tables.h"   // the same one-blob case tables the kernel uses

#include <cmath>         // std::fmaf, std::floor, std::fmin

// ---------------------------------------------------------------------------
// Host copies of the marching-cubes tables — initialized from the SAME
// macros as the __constant__ copies in kernels.cu (mc_tables.h owns the
// blob; see its header for the one-blob-two-homes reasoning).
// ---------------------------------------------------------------------------
static constexpr signed char kTriTable[256][16] = MC_TRI_TABLE_INITIALIZER;

// ---------------------------------------------------------------------------
// tsdf_integrate_cpu — fuse one depth frame, voxel by voxel (the kernel's
// twin; see kernels.cu for the full physics/units commentary — not repeated
// here, the MATH must stay identical and diffable).
// ---------------------------------------------------------------------------
void tsdf_integrate_cpu(const float* depth, Intrinsics K,
                        PoseRt T,
                        float* tsdf, float* weight)
{
    // Loop nest ordered z, y, x so the flat index v advances by 1 each
    // iteration — the same x-fastest order the GPU decodes, kept sequential.
    int v = 0;
    for (int iz = 0; iz < kVolN; ++iz) {
        for (int iy = 0; iy < kVolN; ++iy) {
            for (int ix = 0; ix < kVolN; ++ix, ++v) {
                // Voxel center in world (m) — same fmaf spelling as the kernel.
                const float px = std::fmaf(static_cast<float>(ix) + 0.5f, kVoxelSize, kVolOriginX);
                const float py = std::fmaf(static_cast<float>(iy) + 0.5f, kVoxelSize, kVolOriginY);
                const float pz = std::fmaf(static_cast<float>(iz) + 0.5f, kVoxelSize, kVolOriginZ);

                // World -> camera.
                const float xc = std::fmaf(T.r[0], px, std::fmaf(T.r[1], py, std::fmaf(T.r[2], pz, T.t[0])));
                const float yc = std::fmaf(T.r[3], px, std::fmaf(T.r[4], py, std::fmaf(T.r[5], pz, T.t[1])));
                const float zc = std::fmaf(T.r[6], px, std::fmaf(T.r[7], py, std::fmaf(T.r[8], pz, T.t[2])));
                if (zc <= 0.0f) continue;                       // behind the camera

                // Project; round to nearest pixel center (identical floor(x+0.5)).
                const float u_px = std::fmaf(K.fx, xc / zc, K.cx);
                const float v_px = std::fmaf(K.fy, yc / zc, K.cy);
                const int ui = static_cast<int>(std::floor(u_px + 0.5f));
                const int vi = static_cast<int>(std::floor(v_px + 0.5f));
                if (ui < 0 || ui >= K.width || vi < 0 || vi >= K.height) continue;

                const float d = depth[vi * K.width + ui];
                if (d <= 0.0f) continue;                        // no return at this pixel

                // Projective SDF + asymmetric truncation (kernels.cu explains).
                const float sdf = d - zc;
                if (sdf < -kTruncation) continue;               // occluded — unknown, skip
                const float f = std::fmin(1.0f, sdf * (1.0f / kTruncation));

                // Running weighted average, weight capped.
                const float w  = weight[v];
                const float wn = w + 1.0f;
                tsdf[v]   = std::fmaf(tsdf[v], w, f) / wn;
                weight[v] = std::fmin(wn, kMaxWeight);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// marching_cubes_count_cpu — classify every cell exactly as the kernel does
// and total the triangles, without emitting geometry.
//
// Same corner numbering, same "weight==0 disqualifies", same strict
// "tsdf < 0 = inside" comparison, same -1-terminated table rows — so the
// count is an exact invariant shared with the GPU pass regardless of the
// order its atomics fired in.
// ---------------------------------------------------------------------------
long long marching_cubes_count_cpu(const float* tsdf, const float* weight)
{
    // Corner offsets — must match kernels.cu / mc_tables.h numbering.
    static constexpr int dx[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    static constexpr int dy[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    static constexpr int dz[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

    constexpr int kCells = kVolN - 1;
    long long total = 0;                       // 64-bit: immune to any future volume growth

    for (int iz = 0; iz < kCells; ++iz) {
        for (int iy = 0; iy < kCells; ++iy) {
            for (int ix = 0; ix < kCells; ++ix) {
                int cubeindex = 0;             // bit i = corner i inside (tsdf < 0)
                bool valid = true;
                for (int i = 0; i < 8; ++i) {
                    const int vi = ((iz + dz[i]) * kVolN + (iy + dy[i])) * kVolN + (ix + dx[i]);
                    if (weight[vi] == 0.0f) { valid = false; break; }   // unobserved corner
                    if (tsdf[vi] < 0.0f) cubeindex |= (1 << i);
                }
                if (!valid || cubeindex == 0 || cubeindex == 255) continue;

                // Count this case's triangles by walking to the -1 terminator
                // — the same walk the kernel's emission loop performs.
                const signed char* row = kTriTable[cubeindex];
                for (int t = 0; row[t] != -1; t += 3) ++total;
            }
        }
    }
    return total;
}
