// ===========================================================================
// reference_cpu.cpp - plain-C++ CPU reference for project 02.20
//                      (LiDAR intensity calibration across channels)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5):
//   1) It is the CORRECTNESS ORACLE: main.cu runs both paths and asserts
//      element-wise agreement within a documented tolerance.
//   2) It is the TEACHING BASELINE: reading this file, then kernels.cu,
//      shows exactly what parallelization changed.
//
// Independence ruling (this project's application of the Phase-1 standards
// retrospective, restated in full in kernels.cuh's file header):
//   * Data-layout contracts and SHORT, closed-form FORMULAS (classify_
//     normal_family, range_falloff, corrected_log_intensity, voxel_coord,
//     flat_voxel_index) are single-sourced in kernels.cuh and shared,
//     token-for-token, with kernels.cu's device code - sharing a four-line
//     formula is transcription, not the algorithm under test.
//   * The ALGORITHMIC CORE - the ACCUMULATION LOOPS that decide iteration
//     order, precision, and data structure - is written TWICE,
//     independently:
//       - bin_accumulate_cpu accumulates in DOUBLE precision, sequentially
//         in point index order 0..n-1 (never atomics, never GPU thread-
//         scheduling order) - the same "independent order + better
//         precision" asymmetry projects 01.09/02.01 use for their own
//         atomic-vs-sequential reductions.
//       - assemble_ls_cpu independently re-derives kernels.cuh SECTION 5's
//         centering-projector formula IN DOUBLE (not by calling the shared
//         float channel_ls_accumulate() helper) - a stricter, differently-
//         precisioned oracle for the least-squares assembly stage, the same
//         asymmetry bin_accumulate_cpu uses one level up.
//   * solve_channel_gains() (kernels.cu SECTION 8) is SHARED (called ONCE,
//     the 01.09 SECTION-5 precedent for a dense micro-solve with no
//     meaningful GPU mapping) - main.cu's gain_recovery gate against
//     independent ground truth is what proves that shared solve is
//     trustworthy, never a second implementation of it.
//
// Read this after: kernels.cu - then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <cmath>

// ---------------------------------------------------------------------------
// point_features_cpu - independent host loop calling this project's SHARED
// forward-model/voxel-key formulas (kernels.cuh; see file header). Twin of
// point_features_kernel - since the per-point computation has no search or
// reduction to independently re-derive, this stage's VERIFY gate mainly
// catches indexing/transfer bugs, not algorithmic drift; the REAL
// independence lives one stage later (bin_accumulate_cpu).
// ---------------------------------------------------------------------------
void point_features_cpu(int n, const float* xyz, const float* intensity,
                         const GridBounds& grid, float* log_intensity, int32_t* voxel_idx)
{
    for (int i = 0; i < n; ++i) {
        const float p[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

        float normal[3];
        classify_normal_family(p, normal);

        const float r = point_range(p);
        const float r_safe = r > 1e-6f ? r : 1e-6f;
        const float cos_theta = std::fabs(p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2]) / r_safe;
        const float f_r = range_falloff(r);

        log_intensity[i] = corrected_log_intensity(intensity[i], f_r, cos_theta);

        const int32_t ix = voxel_coord(p[0], grid.leaf);
        const int32_t iy = voxel_coord(p[1], grid.leaf);
        const int32_t iz = voxel_coord(p[2], grid.leaf);
        voxel_idx[i] = flat_voxel_index(ix, iy, iz, grid);
    }
}

// ---------------------------------------------------------------------------
// bin_accumulate_cpu - INDEPENDENT twin of bin_accumulate_kernel (file
// header): sequential point-index order, DOUBLE accumulation - deliberately
// NOT the GPU's atomic/thread-scheduling order, and deliberately more
// precise, so the two paths' agreement is a real correctness signal, not a
// tautology. voxel_family is not recomputed here (main.cu's CPU pipeline
// path reuses point_features_cpu's classify_normal_family results directly
// via a second, tiny loop in main.cu - kept out of the hot accumulation loop
// below to keep this function's ONE JOB (the reduction) uncluttered).
// ---------------------------------------------------------------------------
void bin_accumulate_cpu(int n, const int32_t* channel, const float* log_intensity,
                         const int32_t* voxel_idx, int numVoxels,
                         double* sum_log_d, int32_t* count)
{
    for (int i = 0; i < numVoxels * kNumBeams; ++i) { sum_log_d[i] = 0.0; count[i] = 0; }
    for (int i = 0; i < n; ++i) {
        const int32_t vox = voxel_idx[i];
        if (vox < 0 || vox >= numVoxels) continue;
        const int32_t ch = channel[i];
        sum_log_d[static_cast<size_t>(vox) * kNumBeams + ch] += static_cast<double>(log_intensity[i]);
        count[static_cast<size_t>(vox) * kNumBeams + ch] += 1;
    }
}

// ---------------------------------------------------------------------------
// assemble_ls_cpu - INDEPENDENT twin of assemble_ls_kernel (file header):
// sequential voxel-index order, DOUBLE precision throughout, and the
// centering-projector formula (kernels.cuh SECTION 5) re-derived HERE
// directly rather than calling the shared float channel_ls_accumulate() -
// the stricter oracle this project's twin-independence ruling asks for.
// ---------------------------------------------------------------------------
void assemble_ls_cpu(int numVoxels, const double* sum_log_d, const int32_t* count,
                      double* A_d, double* b_d, int32_t* shared_voxel_count)
{
    for (int i = 0; i < kNumBeams * kNumBeams; ++i) A_d[i] = 0.0;
    for (int i = 0; i < kNumBeams; ++i) b_d[i] = 0.0;
    *shared_voxel_count = 0;

    int32_t chans[kNumBeams];
    double y[kNumBeams];

    for (int v = 0; v < numVoxels; ++v) {
        int k = 0;
        for (int c = 0; c < kNumBeams; ++c) {
            const int32_t cnt = count[static_cast<size_t>(v) * kNumBeams + c];
            if (cnt > 0) {
                chans[k] = c;
                y[k] = sum_log_d[static_cast<size_t>(v) * kNumBeams + c] / static_cast<double>(cnt);
                ++k;
            }
        }
        if (k < 2) continue;   // not a shared voxel (kernels.cuh SECTION 4/5)
        (*shared_voxel_count)++;

        double ybar = 0.0;
        for (int i = 0; i < k; ++i) ybar += y[i];
        ybar /= static_cast<double>(k);

        const double inv_k = 1.0 / static_cast<double>(k);
        for (int i = 0; i < k; ++i) {
            const int32_t ci = chans[i];
            const double r_i = y[i] - ybar;
            b_d[ci] += r_i;
            for (int j = 0; j < k; ++j) {
                const int32_t cj = chans[j];
                const double pij = (i == j) ? (1.0 - inv_k) : (-inv_k);
                A_d[ci * kNumBeams + cj] += pij;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// apply_gain_cpu - independent twin of apply_gain_kernel. Trivial per-point
// division; independence here mainly guards against an indexing typo, the
// same "even a one-line kernel gets a real twin" discipline the SAXPY
// template teaches from project zero.
// ---------------------------------------------------------------------------
void apply_gain_cpu(int n, const float* raw_intensity, const int32_t* channel,
                     const float* gain, float* corrected)
{
    for (int i = 0; i < n; ++i) {
        const int32_t ch = channel[i];
        corrected[i] = raw_intensity[i] / gain[ch];
    }
}
