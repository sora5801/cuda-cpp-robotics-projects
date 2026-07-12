// ===========================================================================
// reference_cpu.cpp - plain-C++ CPU reference for project 02.18
//                      Weather filtering: snow/rain/dust outlier removal
//                      (DROR/LIOR)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5):
//
//   1) It is the CORRECTNESS ORACLE. main.cu's VERIFY stage runs both paths
//      on the SAME committed scan and requires agreement (tight tolerance
//      for SOR's floating-point mean distance, EXACT for DROR/LIOR's
//      integer neighbor counts and for every classify mask - kernels.cuh's
//      file header names the exact scheme).
//   2) It is the TEACHING BASELINE - the sequential "for each point, scan
//      every other point" loop that kernels.cu's one-thread-per-point
//      kernels parallelize.
//
// TWIN vs SHARED - the independence ruling this project follows
// ----------------------------------------------------------------
// kernels.cuh already explains WHAT is shared (squared_distance3, range3,
// dist_less, dror_search_radius - pure formula bookkeeping, HD-qualified,
// compiled into both TUs from one source) and WHY: sharing a four-line
// formula is transcription, not the algorithm under test. WHAT IS NOT
// SHARED, and is instead typed out fresh below - the actual O(n) brute-
// force search loops:
//   * sor_mean_knn_dist_cpu's insertion-sorted K-nearest scan,
//   * dror_neighbor_count_cpu's / lior_neighbor_count_cpu's radius-count
//     scans.
// Each is retyped independently from kernels.cu's version - same ALGORITHM
// family (the repo's standard "same algorithm, independently transcribed"
// rule, cited throughout - see 02.09/02.13's own reference_cpu.cpp file
// headers for the identical reasoning), so a copy-paste bug in one cannot
// silently pass the other's mirror. The three CLASSIFY functions are
// single, trivial threshold comparisons; sharing THOSE would leave nothing
// for the verify gate to check, so they too are retyped (a two-line
// function is cheap insurance, not wasted effort).
//
// Rules for this file: plain C++17, no CUDA headers, no OpenMP, no
// cleverness - a reader should be able to verify every loop by eye.
//
// Read this after: kernels.cu - then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // shared HD helpers + this file's own prototypes: compiler-enforced
                          // signature agreement with what main.cu calls

#include <cmath>         // sqrtf

// ---------------------------------------------------------------------------
// sor_mean_knn_dist_cpu - sequential twin of sor_mean_knn_dist_kernel.
//
// Same insertion-sorted size-kSorK array, same dist_less total order, same
// candidate-discovery order (j = 0..n-1) - so the SET (and sum) of the K
// nearest neighbors is identical to the GPU path's, which is what makes the
// tight-tolerance (not merely "plausible") verify comparison meaningful
// (kernels.cuh file header).
// ---------------------------------------------------------------------------
void sor_mean_knn_dist_cpu(int n, const float* xyz, float* mean_dist)
{
    for (int i = 0; i < n; ++i) {
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

        float best_d2[kSorK];
        int32_t best_idx[kSorK];
        int count = 0;

        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
            const float d2 = squared_distance3(pi, pj);

            if (count < kSorK) {
                int pos = count;
                while (pos > 0 && dist_less(d2, static_cast<int32_t>(j), best_d2[pos - 1], best_idx[pos - 1])) {
                    best_d2[pos] = best_d2[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                    --pos;
                }
                best_d2[pos] = d2;
                best_idx[pos] = static_cast<int32_t>(j);
                ++count;
            } else if (dist_less(d2, static_cast<int32_t>(j), best_d2[kSorK - 1], best_idx[kSorK - 1])) {
                int pos = kSorK - 1;
                while (pos > 0 && dist_less(d2, static_cast<int32_t>(j), best_d2[pos - 1], best_idx[pos - 1])) {
                    best_d2[pos] = best_d2[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                    --pos;
                }
                best_d2[pos] = d2;
                best_idx[pos] = static_cast<int32_t>(j);
            }
        }

        float sum = 0.0f;
        for (int k = 0; k < count; ++k) sum += sqrtf(best_d2[k]);
        mean_dist[i] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
    }
}

// sor_classify_cpu - independent twin of sor_classify_kernel (a two-line
// function, retyped rather than shared - file header explains why).
void sor_classify_cpu(int n, const float* mean_dist, float threshold, int32_t* mask_out)
{
    for (int i = 0; i < n; ++i)
        mask_out[i] = (mean_dist[i] > threshold) ? 1 : 0;
}

// dror_neighbor_count_cpu - sequential twin of dror_neighbor_count_kernel:
// same per-point range-scaled radius (the SHARED dror_search_radius
// formula), independently-typed O(n) scan.
void dror_neighbor_count_cpu(int n, const float* xyz, int32_t* count_out)
{
    for (int i = 0; i < n; ++i) {
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
        const float radius = dror_search_radius(range3(pi));
        const float radius2 = radius * radius;

        int32_t cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
            if (squared_distance3(pi, pj) <= radius2) ++cnt;
        }
        count_out[i] = cnt;
    }
}

// dror_classify_cpu - independent twin of dror_classify_kernel.
void dror_classify_cpu(int n, const int32_t* count, int32_t* mask_out)
{
    for (int i = 0; i < n; ++i)
        mask_out[i] = (count[i] < kDrorKMin) ? 1 : 0;
}

// lior_neighbor_count_cpu - sequential twin of lior_neighbor_count_kernel:
// the FIXED kLiorRadius (kernels.cu's file header names the deliberate
// DROR contrast), independently-typed O(n) scan.
void lior_neighbor_count_cpu(int n, const float* xyz, int32_t* count_out)
{
    const float radius2 = kLiorRadius * kLiorRadius;

    for (int i = 0; i < n; ++i) {
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

        int32_t cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
            if (squared_distance3(pi, pj) <= radius2) ++cnt;
        }
        count_out[i] = cnt;
    }
}

// lior_classify_cpu - independent twin of lior_classify_kernel.
void lior_classify_cpu(int n, const float* intensity, const int32_t* count, int32_t* mask_out)
{
    for (int i = 0; i < n; ++i) {
        const bool dim = intensity[i] < kLiorIntensityThresh;
        const bool sparse = count[i] < kLiorKMin;
        mask_out[i] = (dim && sparse) ? 1 : 0;
    }
}
