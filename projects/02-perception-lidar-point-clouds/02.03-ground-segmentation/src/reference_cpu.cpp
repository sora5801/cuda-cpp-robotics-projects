// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.03
//                     (Ground segmentation: RANSAC plane fit; Patchwork++-
//                     style GPU port)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), restated here because this
// project's oracle is unusually varied in HOW it stays independent:
//
//   1) CORRECTNESS ORACLE. Five functions below give main.cu five VERIFY(...)
//      gates. Two are single-sourced-formula comparisons (bit-exact, drift
//      detectors — see kernels.cuh's file header for the ruling): hypothesis
//      generation and patch-id assignment call the SAME plain functions the
//      GPU kernels literally re-key as __device__ transcriptions, so any
//      typo in either copy shows up as a mismatch. Three are GENUINELY
//      INDEPENDENT re-implementations (own loop structure, own precision,
//      own data structures — the "twin" half of the ruling): hypothesis
//      evaluation, RANSAC refinement, and the CZM fit+classify. Each
//      function's comment below states which kind it is.
//
//   2) TEACHING BASELINE. Reading this file next to kernels.cu shows exactly
//      what "porting to the GPU" changed for each step: a per-hypothesis
//      loop became one block per hypothesis; a per-patch std::vector became
//      a Thrust sort + block-per-column; a double-precision sequential
//      covariance sum became float atomics. Same math, different hardware
//      mapping — the whole point of this repository.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP. If the reference is clever, it can be wrong, and then the
// oracle lies. (Compiled by the HOST compiler, cl.exe; kernels.cuh's
// __CUDACC__ fence hides device-only declarations from it.)
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "kernels.cuh"

// ===========================================================================
// MILESTONE 1 — RANSAC twins
// ===========================================================================

// ransac_generate_hypotheses_cpu — SHARED-FORMULA twin: calls kernels.cuh's
// own hypothesis_seed/pick_triplet_indices/plane_from_triplet directly (the
// exact functions kernels.cu's __device__ code transcribes). VERIFY
// (hypotheses) in main.cu compares this against the GPU's output bit-exact —
// a drift detector for the device transcription, not an algorithmic check.
void ransac_generate_hypotheses_cpu(int n, const float* xyz, uint32_t global_seed, int k,
                                    PlaneModel* hyp_plane, uint8_t* hyp_valid)
{
    for (int h = 0; h < k; ++h) {
        bool found = false;
        PlaneModel pl;
        for (int attempt = 0; attempt < kRansacMaxTripletAttempts && !found; ++attempt) {
            const uint32_t seed = hypothesis_seed(global_seed, h, attempt);
            int i0, i1, i2;
            pick_triplet_indices(seed, n, i0, i1, i2);
            const float p0[3] = { xyz[i0*3+0], xyz[i0*3+1], xyz[i0*3+2] };
            const float p1[3] = { xyz[i1*3+0], xyz[i1*3+1], xyz[i1*3+2] };
            const float p2[3] = { xyz[i2*3+0], xyz[i2*3+1], xyz[i2*3+2] };
            if (plane_from_triplet(p0, p1, p2, pl)) found = true;
        }
        hyp_plane[h] = pl;
        hyp_valid[h] = found ? 1u : 0u;
    }
}

// ransac_evaluate_hypotheses_cpu — INDEPENDENT twin: a single sequential
// K*N nested loop with NO shared-memory reduction, NO block-per-hypothesis
// structure — genuinely different code from ransac_evaluate_hypotheses_kernel,
// not a transcription. VERIFY(ransac_eval) in main.cu compares the per-
// hypothesis inlier COUNTS (integers) bit-exact — expected to hold in
// practice for this random synthetic scene (THEORY.md "Numerical
// considerations" explains why a boundary-straddling point is astronomically
// unlikely, the same empirical argument 02.01's Method-B bit-exactness
// claim rests on).
void ransac_evaluate_hypotheses_cpu(int n, const float* xyz, const PlaneModel* hyp_plane,
                                    const uint8_t* hyp_valid, float threshold, int k,
                                    int* hyp_inlier_count)
{
    for (int h = 0; h < k; ++h) {
        if (!hyp_valid[h]) { hyp_inlier_count[h] = -1; continue; }
        const PlaneModel pl = hyp_plane[h];
        int count = 0;
        for (int i = 0; i < n; ++i) {
            const float p[3] = { xyz[i*3+0], xyz[i*3+1], xyz[i*3+2] };
            const float dist = point_plane_signed_distance(pl, p);
            if (std::fabs(dist) <= threshold) ++count;
        }
        hyp_inlier_count[h] = count;
    }
}

// ransac_refine_cpu — INDEPENDENT twin: DOUBLE-precision sequential
// accumulation (ascending point index, no atomics, no thread-scheduling
// dependence) — deliberately different from the GPU's float/atomicAdd path
// (kernels.cu's ransac_accumulate_inliers_kernel comment explains why that
// path is order-nondeterministic). Compared within a measured-then-margined
// TOLERANCE (VERIFY(ransac_refine)), the same ruling class as 02.01's
// Method A vs. its independent hashmap oracle.
bool ransac_refine_cpu(int n, const float* xyz, PlaneModel plane, float threshold, PlaneModel& refined_out)
{
    double sx=0,sy=0,sz=0, sxx=0,sxy=0,sxz=0, syy=0,syz=0, szz=0;
    unsigned long long count = 0;
    for (int i = 0; i < n; ++i) {
        const float p[3] = { xyz[i*3+0], xyz[i*3+1], xyz[i*3+2] };
        const float dist = point_plane_signed_distance(plane, p);
        if (std::fabs(dist) <= threshold) {
            const double x = p[0], y = p[1], z = p[2];
            sx += x; sy += y; sz += z;
            sxx += x*x; sxy += x*y; sxz += x*z; syy += y*y; syz += y*z; szz += z*z;
            ++count;
        }
    }
    if (count < 3) return false;
    const double inv_n = 1.0 / static_cast<double>(count);
    const double mx = sx*inv_n, my = sy*inv_n, mz = sz*inv_n;
    const double cxx = sxx*inv_n - mx*mx, cxy = sxy*inv_n - mx*my, cxz = sxz*inv_n - mx*mz;
    const double cyy = syy*inv_n - my*my, cyz = syz*inv_n - my*mz, czz = szz*inv_n - mz*mz;
    const float packed[6] = { static_cast<float>(cxx), static_cast<float>(cxy), static_cast<float>(cxz),
                              static_cast<float>(cyy), static_cast<float>(cyz), static_cast<float>(czz) };
    float eigenvalues[3]; float eigenvectors[3][3];
    jacobi_eigen_3x3(packed, eigenvalues, eigenvectors);
    float nx = eigenvectors[0][0], ny = eigenvectors[0][1], nz = eigenvectors[0][2];
    if (nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }
    refined_out.nx = nx; refined_out.ny = ny; refined_out.nz = nz;
    refined_out.d  = -(nx * static_cast<float>(mx) + ny * static_cast<float>(my) + nz * static_cast<float>(mz));
    return true;
}

// ===========================================================================
// MILESTONE 2 — CZM twins
// ===========================================================================

// czm_compute_patch_ids_cpu — SHARED-FORMULA twin: calls kernels.cuh's own
// czm_compute_patch_id directly. VERIFY(patch_ids) compares this against the
// GPU's __device__ transcription, bit-exact — the drift detector.
void czm_compute_patch_ids_cpu(int n, const float* xyz, int* patch_id)
{
    for (int i = 0; i < n; ++i) patch_id[i] = czm_compute_patch_id(xyz[i*3+0], xyz[i*3+1]);
}

// ---------------------------------------------------------------------------
// czm_fit_and_classify_cpu — INDEPENDENT twin, genuinely different in THREE
// ways from czm_fit_and_classify_kernel: (a) DATA STRUCTURE — a
// std::vector<std::vector<int>> per-patch point list built with one linear
// pass (vs. the GPU's Thrust sort + patch_start index ranges); (b)
// ACCUMULATION ORDER/PRECISION — double-precision sums walking each patch's
// vector in ASCENDING ORIGINAL POINT INDEX order (vs. the GPU's float,
// grid-stride-order block reduction); (c) no shared-memory reduction at all
// (a plain sequential loop). Same ALGORITHM (min-z/carry seed rule, PCA fit
// via this header's shared jacobi_eigen_3x3, uprightness+flatness tests,
// region-growing height carry) — VERIFY(czm_fit) in main.cu compares
// is_ground flags and point classifications against the GPU's within a
// documented tolerance (not bit-exact — same ruling as ransac_refine_cpu).
// ---------------------------------------------------------------------------
void czm_fit_and_classify_cpu(int n, const float* xyz, const int* patch_id,
                              CzmPatchResult* patch_result, uint8_t* point_ground)
{
    std::vector<std::vector<int>> patches(static_cast<size_t>(kCzmNumPatches));
    for (int i = 0; i < n; ++i) {
        const int pid = patch_id[i];
        if (pid >= 0 && pid < kCzmNumPatches) patches[static_cast<size_t>(pid)].push_back(i);
        point_ground[i] = 0u;   // default non-ground; overwritten below for points in a passing ground patch
    }

    const double kPi = 3.14159265358979323846;

    for (int col = 0; col < kCzmNumColumns; ++col) {
        double carry_z = 0.0;
        bool carry_valid = false;

        for (int ring = 0; ring < kCzmRingsPerZone; ++ring) {
            const int patch = col * kCzmRingsPerZone + ring;
            const std::vector<int>& pts = patches[static_cast<size_t>(patch)];
            const int npts = static_cast<int>(pts.size());

            double min_z = 1.0e300;
            for (int idx : pts) min_z = std::min(min_z, static_cast<double>(xyz[idx*3+2]));

            double zlo, zhi;
            if (carry_valid) { zlo = carry_z - kCzmHeightCarryBandM; zhi = carry_z + kCzmHeightCarryBandM; }
            else              { zlo = min_z; zhi = min_z + kCzmSeedHeightMarginM; }

            // Sequential double-precision covariance accumulation, ascending
            // original point index (the vector's push_back order above).
            double sx=0,sy=0,sz=0, sxx=0,sxy=0,sxz=0, syy=0,syz=0, szz=0;
            unsigned int count = 0;
            for (int idx : pts) {
                const double x = xyz[idx*3+0], y = xyz[idx*3+1], z = xyz[idx*3+2];
                if (z >= zlo && z <= zhi) {
                    sx += x; sy += y; sz += z;
                    sxx += x*x; sxy += x*y; sxz += x*z; syy += y*y; syz += y*z; szz += z*z;
                    ++count;
                }
            }

            PlaneModel pl;
            bool is_ground = false;
            float upright_deg = 0.0f;
            if (count >= kCzmMinPatchPoints) {
                const double inv_n = 1.0 / static_cast<double>(count);
                const double mx = sx*inv_n, my = sy*inv_n, mz = sz*inv_n;
                const double cxx = sxx*inv_n - mx*mx, cxy = sxy*inv_n - mx*my, cxz = sxz*inv_n - mx*mz;
                const double cyy = syy*inv_n - my*my, cyz = syz*inv_n - my*mz, czz = szz*inv_n - mz*mz;
                const float packed[6] = { static_cast<float>(cxx), static_cast<float>(cxy), static_cast<float>(cxz),
                                          static_cast<float>(cyy), static_cast<float>(cyz), static_cast<float>(czz) };
                float eigenvalues[3]; float eigenvectors[3][3];
                jacobi_eigen_3x3(packed, eigenvalues, eigenvectors);
                float nx = eigenvectors[0][0], ny = eigenvectors[0][1], nz = eigenvectors[0][2];
                if (nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }
                pl.nx = nx; pl.ny = ny; pl.nz = nz;
                pl.d = -(nx * static_cast<float>(mx) + ny * static_cast<float>(my) + nz * static_cast<float>(mz));
                upright_deg = static_cast<float>(std::acos(std::min(std::max(static_cast<double>(nz), -1.0), 1.0)) * (180.0 / kPi));
                if (upright_deg <= kCzmUprightMaxDeg) is_ground = true;
            }

            float rms = 0.0f;
            if (is_ground) {
                double sumsq = 0.0;
                for (int idx : pts) {
                    const double x = xyz[idx*3+0], y = xyz[idx*3+1], z = xyz[idx*3+2];
                    if (z >= zlo && z <= zhi) {
                        const double dist = static_cast<double>(pl.nx)*x + static_cast<double>(pl.ny)*y
                                           + static_cast<double>(pl.nz)*z + static_cast<double>(pl.d);
                        sumsq += dist * dist;
                    }
                }
                rms = static_cast<float>(std::sqrt(sumsq / std::max(static_cast<double>(count), 1.0)));
                if (rms > kCzmFlatnessMaxRmsM) is_ground = false;
            }

            for (int idx : pts) {
                uint8_t g = 0u;
                if (is_ground) {
                    const float p[3] = { xyz[idx*3+0], xyz[idx*3+1], xyz[idx*3+2] };
                    const float dist = point_plane_signed_distance(pl, p);
                    g = (std::fabs(dist) <= kCzmClassifyDistM) ? 1u : 0u;
                }
                point_ground[idx] = g;
            }

            CzmPatchResult res;
            res.plane = pl;
            res.is_ground = is_ground ? 1 : 0;
            res.patch_point_count = static_cast<unsigned int>(npts);
            res.seed_point_count = count;
            res.rms_residual_m = rms;
            res.uprightness_deg = upright_deg;
            res.used_prior = carry_valid ? 1 : 0;
            patch_result[static_cast<size_t>(patch)] = res;

            if (is_ground) {
                const double mx = (count > 0) ? sx / static_cast<double>(count) : 0.0;
                const double my = (count > 0) ? sy / static_cast<double>(count) : 0.0;
                const double nz_safe = (std::fabs(static_cast<double>(pl.nz)) > 1.0e-3) ? static_cast<double>(pl.nz) : 1.0;
                carry_z = -(static_cast<double>(pl.d) + static_cast<double>(pl.nx) * mx + static_cast<double>(pl.ny) * my) / nz_safe;
                carry_valid = true;
            }
            // else: leave carry_z/carry_valid unchanged (see kernels.cu's matching comment).
        }
    }
}
