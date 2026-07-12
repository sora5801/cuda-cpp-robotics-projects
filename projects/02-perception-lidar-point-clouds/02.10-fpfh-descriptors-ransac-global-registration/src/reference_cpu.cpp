// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.10
//                     (FPFH descriptors + RANSAC global registration)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5):
//   1) It is the CORRECTNESS ORACLE — main.cu runs both paths and asserts
//      agreement within a documented tolerance.
//   2) It is the TEACHING BASELINE — reading this file, then kernels.cu,
//      shows exactly what parallelization changed.
//
// Independence ruling for THIS project — see kernels.cuh's file header
// "Twin-vs-shared ruling" for the full per-stage table. Summary:
//   * STAGE 1 (normals): INDEPENDENT eigensolve (jacobi_eigen_3x3_cpu below)
//     — 02.09's stricter choice, cited.
//   * STAGE 2/3 (SPFH/FPFH): the Darboux/binning FORMULA is shared
//     (kernels.cuh's darboux_triplet/angle_to_bin, called directly below);
//     the neighbor-gathering and accumulation LOOPS are this file's own.
//   * STAGE 5 (RANSAC hypotheses): the per-hypothesis Horn FIT is shared
//     (kernels.cuh's rigid_fit_horn, called directly — bit-exact-checkable
//     against the GPU's device transcription); the scoring loop is this
//     file's own.
//   * STAGE 5/6 (refit): ransac_refit_cpu below is a FULLY INDEPENDENT
//     double-precision reimplementation (its own jacobi_eigen_4x4_cpu, not
//     kernels.cuh's float rigid_fit_horn) — 02.03's ransac_refine_cpu
//     precedent, cited: the project's non-tautological numerical
//     cross-check on the Horn/Jacobi math itself.
//   * STAGE 6 (ICP): transform/correspondences/accumulation are all
//     independent sequential reimplementations — 02.06's identical ruling,
//     cited (that project's build_normal_system_cpu accumulates in double
//     while the GPU reduces in float, "the more precise of the two paths
//     by construction" — the same shape reused here for point-to-plane).
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <algorithm>   // std::fabs is in <cmath>; this is for the rare std::sort-style scan if ever needed

// ===========================================================================
// STAGE 1 helper — knn_search_cpu: an INDEPENDENT brute-force KNN — same
// O(n^2) algorithm SHAPE as the GPU kernel (there is no faster honest
// choice at this scale, see kernels.cu's file header) but a genuinely
// DIFFERENT data structure: a plain insertion into a SORTED std::vector
// (never a binary max-heap) — the "independent data structure" ruling
// 02.04/02.05/02.09 apply to their own KNN twins, reused here.
// ---------------------------------------------------------------------------
void knn_search_cpu(int n, const float* xyz, int32_t* neighbor_ids, float* neighbor_dist)
{
    for (int q = 0; q < n; ++q) {
        const float qp[3] = { xyz[q * 3 + 0], xyz[q * 3 + 1], xyz[q * 3 + 2] };

        // best[]: kept SORTED ascending by knn_less at all times (insertion
        // sort on every candidate that beats the current worst kept entry —
        // a different bookkeeping shape than the GPU's max-heap, though the
        // FINAL contents must agree exactly under the same total order).
        std::vector<std::pair<float, int32_t>> best;   // (dist2, id), size <= kFpfhK
        best.reserve(kFpfhK);

        for (int j = 0; j < n; ++j) {
            if (j == q) continue;
            const float dx = xyz[j * 3 + 0] - qp[0], dy = xyz[j * 3 + 1] - qp[1], dz = xyz[j * 3 + 2] - qp[2];
            const float d2 = dx * dx + dy * dy + dz * dz;

            if (static_cast<int>(best.size()) < kFpfhK) {
                // Insert in sorted position (linear scan from the back — fine
                // at kFpfhK==20).
                size_t pos = best.size();
                best.push_back({ d2, static_cast<int32_t>(j) });
                while (pos > 0 && knn_less(best[pos].first, best[pos].second, best[pos - 1].first, best[pos - 1].second)) {
                    std::swap(best[pos], best[pos - 1]);
                    --pos;
                }
            } else if (knn_less(d2, static_cast<int32_t>(j), best.back().first, best.back().second)) {
                best.back() = { d2, static_cast<int32_t>(j) };
                size_t pos = best.size() - 1;
                while (pos > 0 && knn_less(best[pos].first, best[pos].second, best[pos - 1].first, best[pos - 1].second)) {
                    std::swap(best[pos], best[pos - 1]);
                    --pos;
                }
            }
        }

        for (int a = 0; a < kFpfhK; ++a) {
            neighbor_ids[q * kFpfhK + a] = best[static_cast<size_t>(a)].second;
            neighbor_dist[q * kFpfhK + a] = std::sqrt(best[static_cast<size_t>(a)].first);
        }
    }
}

// ===========================================================================
// STAGE 1 — normals: jacobi_eigen_3x3_cpu is a fully INDEPENDENT cyclic-
// Jacobi implementation (same algorithm FAMILY as kernels.cu's
// d_jacobi_eigen_3x3, cited, but its own separate C++ — 02.09's
// "STEP 4's independence ruling" precedent, cited verbatim in kernels.cuh).
// ---------------------------------------------------------------------------
void jacobi_eigen_3x3_cpu(const float cov[6], float eigenvalues[3], float eigenvectors[3][3])
{
    double A[3][3] = {
        { cov[0], cov[1], cov[2] },
        { cov[1], cov[3], cov[4] },
        { cov[2], cov[4], cov[5] },
    };
    double V[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };
    const int pairs[3][2] = { {0,1}, {0,2}, {1,2} };

    for (int sweep = 0; sweep < kJacobiSweeps3; ++sweep) {
        for (int pi = 0; pi < 3; ++pi) {
            const int p = pairs[pi][0], q = pairs[pi][1];
            const double apq = A[p][q];
            if (std::fabs(apq) < 1.0e-14) continue;
            const double theta = (A[q][q] - A[p][p]) / (2.0 * apq);
            const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0);
            const double s = t * c;
            const double app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq; A[q][q] = aqq + t * apq;
            A[p][q] = 0.0; A[q][p] = 0.0;
            const int r = 3 - p - q;
            const double arp = A[r][p], arq = A[r][q];
            A[r][p] = A[p][r] = c * arp - s * arq;
            A[r][q] = A[q][r] = s * arp + c * arq;
            for (int i = 0; i < 3; ++i) {
                const double vip = V[i][p], viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }

    double ev[3] = { A[0][0], A[1][1], A[2][2] };
    double vec[3][3];
    for (int i = 0; i < 3; ++i) { vec[i][0] = V[0][i]; vec[i][1] = V[1][i]; vec[i][2] = V[2][i]; }
    for (int i = 1; i < 3; ++i) {
        const double ek = ev[i]; const double vk0 = vec[i][0], vk1 = vec[i][1], vk2 = vec[i][2];
        int j = i - 1;
        while (j >= 0 && ev[j] > ek) {
            ev[j + 1] = ev[j]; vec[j + 1][0] = vec[j][0]; vec[j + 1][1] = vec[j][1]; vec[j + 1][2] = vec[j][2];
            --j;
        }
        ev[j + 1] = ek; vec[j + 1][0] = vk0; vec[j + 1][1] = vk1; vec[j + 1][2] = vk2;
    }
    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = static_cast<float>(ev[i]);
        eigenvectors[i][0] = static_cast<float>(vec[i][0]);
        eigenvectors[i][1] = static_cast<float>(vec[i][1]);
        eigenvectors[i][2] = static_cast<float>(vec[i][2]);
    }
}

void estimate_normals_cpu(int n, const float* xyz, const int32_t* neighbor_ids,
                          float ref_x, float ref_y, float ref_z, float* out_normal)
{
    for (int q = 0; q < n; ++q) {
        const float qp[3] = { xyz[q * 3 + 0], xyz[q * 3 + 1], xyz[q * 3 + 2] };

        double mx = 0, my = 0, mz = 0;
        for (int a = 0; a < kFpfhK; ++a) {
            const int pid = neighbor_ids[q * kFpfhK + a];
            mx += xyz[pid * 3 + 0]; my += xyz[pid * 3 + 1]; mz += xyz[pid * 3 + 2];
        }
        mx /= kFpfhK; my /= kFpfhK; mz /= kFpfhK;

        double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
        for (int a = 0; a < kFpfhK; ++a) {
            const int pid = neighbor_ids[q * kFpfhK + a];
            const double dx = xyz[pid * 3 + 0] - mx, dy = xyz[pid * 3 + 1] - my, dz = xyz[pid * 3 + 2] - mz;
            cxx += dx * dx; cxy += dx * dy; cxz += dx * dz; cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
        }
        const float cov[6] = {
            static_cast<float>(cxx / kFpfhK), static_cast<float>(cxy / kFpfhK), static_cast<float>(cxz / kFpfhK),
            static_cast<float>(cyy / kFpfhK), static_cast<float>(cyz / kFpfhK), static_cast<float>(czz / kFpfhK)
        };

        float eigenvalues[3]; float eigenvectors[3][3];
        jacobi_eigen_3x3_cpu(cov, eigenvalues, eigenvectors);

        float nrm[3] = { eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2] };
        const float view[3] = { ref_x - qp[0], ref_y - qp[1], ref_z - qp[2] };
        const float dotv = nrm[0] * view[0] + nrm[1] * view[1] + nrm[2] * view[2];
        if (dotv < 0.0f) { nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2]; }

        out_normal[q * 3 + 0] = nrm[0]; out_normal[q * 3 + 1] = nrm[1]; out_normal[q * 3 + 2] = nrm[2];
    }
}

// ===========================================================================
// STAGE 2 — compute_spfh_cpu: calls kernels.cuh's SHARED darboux_triplet /
// angle_to_bin DIRECTLY (small, deterministic formula — 02.03's
// czm_compute_patch_id precedent: shared, not re-derived) over this file's
// OWN loop and OWN neighbor list (from knn_search_cpu above, not the GPU's).
// ---------------------------------------------------------------------------
void compute_spfh_cpu(int n, const float* xyz, const float* normal, const int32_t* neighbor_ids, float* out_spfh)
{
    for (int q = 0; q < n; ++q) {
        const float p_q[3] = { xyz[q * 3 + 0], xyz[q * 3 + 1], xyz[q * 3 + 2] };
        const float n_q[3] = { normal[q * 3 + 0], normal[q * 3 + 1], normal[q * 3 + 2] };

        float hist[kFpfhDim];
        for (int i = 0; i < kFpfhDim; ++i) hist[i] = 0.0f;

        for (int a = 0; a < kFpfhK; ++a) {
            const int kid = neighbor_ids[q * kFpfhK + a];
            const float p_k[3] = { xyz[kid * 3 + 0], xyz[kid * 3 + 1], xyz[kid * 3 + 2] };
            const float n_k[3] = { normal[kid * 3 + 0], normal[kid * 3 + 1], normal[kid * 3 + 2] };
            float alpha, phi, theta;
            darboux_triplet(n_q, p_q, n_k, p_k, alpha, phi, theta);   // SHARED header formula
            hist[0 * kFpfhBins + angle_to_bin(alpha, -1.0f, 1.0f)] += 1.0f;
            hist[1 * kFpfhBins + angle_to_bin(phi,   -1.0f, 1.0f)] += 1.0f;
            hist[2 * kFpfhBins + angle_to_bin(theta, -kPiF, kPiF)] += 1.0f;
        }

        const float inv_k = 1.0f / static_cast<float>(kFpfhK);
        for (int i = 0; i < kFpfhDim; ++i) out_spfh[q * kFpfhDim + i] = hist[i] * inv_k;
    }
}

// STAGE 3 — compute_fpfh_cpu: this file's own independent accumulation loop
// (same formula as kernels.cu's compute_fpfh_kernel, written separately).
void compute_fpfh_cpu(int n, const float* spfh, const int32_t* neighbor_ids, const float* neighbor_dist,
                      float* out_fpfh)
{
    for (int q = 0; q < n; ++q) {
        float acc[kFpfhDim];
        for (int i = 0; i < kFpfhDim; ++i) acc[i] = spfh[q * kFpfhDim + i];

        for (int a = 0; a < kFpfhK; ++a) {
            const int kid = neighbor_ids[q * kFpfhK + a];
            const float dist = neighbor_dist[q * kFpfhK + a];
            const float w = (dist > 1.0e-6f) ? (1.0f / dist) : 0.0f;
            for (int i = 0; i < kFpfhDim; ++i) acc[i] += w * spfh[kid * kFpfhDim + i];
        }
        const float inv_k = 1.0f / static_cast<float>(kFpfhK);
        for (int i = 0; i < kFpfhDim; ++i) acc[i] *= inv_k;

        float sum = 0.0f;
        for (int i = 0; i < kFpfhDim; ++i) sum += acc[i];
        const float inv_sum = (sum > 1.0e-9f) ? (1.0f / sum) : 0.0f;
        for (int i = 0; i < kFpfhDim; ++i) out_fpfh[q * kFpfhDim + i] = acc[i] * inv_sum;
    }
}

// ===========================================================================
// STAGE 4 — match_correspondences_cpu: an INDEPENDENT double loop (own
// nested-loop shape; the SAME O(n_src*n_tgt*33) algorithm as the GPU
// kernel — there is no cheaper honest choice at this scale, kernels.cuh's
// file header — but written separately so a coding slip in either path is
// exposed by VERIFY(match), not masked by shared code).
// ---------------------------------------------------------------------------
void match_correspondences_cpu(int n_src, const float* fpfh_src, int n_tgt, const float* fpfh_tgt,
                               uint8_t* out_matched, int32_t* out_best_idx,
                               float* out_dist1_sq, float* out_dist2_sq)
{
    for (int s = 0; s < n_src; ++s) {
        float best1 = 3.0e38f, best2 = 3.0e38f;
        int best1_idx = -1;
        for (int t = 0; t < n_tgt; ++t) {
            float d2 = 0.0f;
            for (int i = 0; i < kFpfhDim; ++i) {
                const float diff = fpfh_src[s * kFpfhDim + i] - fpfh_tgt[t * kFpfhDim + i];
                d2 += diff * diff;
            }
            if (d2 < best1) { best2 = best1; best1 = d2; best1_idx = t; }
            else if (d2 < best2) { best2 = d2; }
        }
        const bool accept = (best1_idx >= 0) && (best1 <= kMatchRatioMax * kMatchRatioMax * best2);
        out_matched[s] = accept ? 1u : 0u;
        out_best_idx[s] = best1_idx;
        out_dist1_sq[s] = best1;
        out_dist2_sq[s] = best2;
    }
}

// ===========================================================================
// STAGE 5 — ransac_hypotheses_cpu: hypothesis generation calls kernels.cuh's
// SHARED hypothesis_seed / pick_correspondence_triplet / edge_length_
// prescreen / rigid_fit_horn DIRECTLY (02.03's ransac_generate_hypotheses_cpu
// precedent, cited: no duplication needed for small deterministic host-
// callable math) — so VERIFY(hypotheses) in main.cu is a real bit-exact
// cross-check against the GPU's literal device transcription. The SCORING
// loop below is this file's own (independent) code.
// ---------------------------------------------------------------------------
void ransac_hypotheses_cpu(int nc, const float* corr_src_xyz, const float* corr_tgt_xyz,
                           uint32_t global_seed, int k,
                           uint8_t* out_valid, Rigid3* out_transform, int32_t* out_inlier_count)
{
    const float thresh2 = kRansacInlierThresholdM * kRansacInlierThresholdM;

    for (int h = 0; h < k; ++h) {
        bool got_valid_triplet = false;
        int i0 = -1, i1 = -1, i2 = -1;
        for (int attempt = 0; attempt < kRansacMaxTripletAttempts; ++attempt) {
            const uint32_t seed = hypothesis_seed(global_seed, h, attempt);
            if (!pick_correspondence_triplet(seed, nc, i0, i1, i2)) continue;
            const float* s0 = &corr_src_xyz[i0 * 3]; const float* s1 = &corr_src_xyz[i1 * 3]; const float* s2 = &corr_src_xyz[i2 * 3];
            const float* t0 = &corr_tgt_xyz[i0 * 3]; const float* t1 = &corr_tgt_xyz[i1 * 3]; const float* t2 = &corr_tgt_xyz[i2 * 3];
            if (edge_length_prescreen(s0, s1, s2, t0, t1, t2)) { got_valid_triplet = true; break; }
        }

        if (!got_valid_triplet) {
            out_valid[h] = 0u; out_inlier_count[h] = 0;
            continue;
        }

        float src3[9] = { corr_src_xyz[i0*3+0], corr_src_xyz[i0*3+1], corr_src_xyz[i0*3+2],
                          corr_src_xyz[i1*3+0], corr_src_xyz[i1*3+1], corr_src_xyz[i1*3+2],
                          corr_src_xyz[i2*3+0], corr_src_xyz[i2*3+1], corr_src_xyz[i2*3+2] };
        float tgt3[9] = { corr_tgt_xyz[i0*3+0], corr_tgt_xyz[i0*3+1], corr_tgt_xyz[i0*3+2],
                          corr_tgt_xyz[i1*3+0], corr_tgt_xyz[i1*3+1], corr_tgt_xyz[i1*3+2],
                          corr_tgt_xyz[i2*3+0], corr_tgt_xyz[i2*3+1], corr_tgt_xyz[i2*3+2] };
        Rigid3 T;
        if (!rigid_fit_horn(3, src3, tgt3, T.R, T.t)) {
            out_valid[h] = 0u; out_inlier_count[h] = 0;
            continue;
        }

        // Independent scoring loop (own code, not shared with kernels.cu).
        int inliers = 0;
        for (int c = 0; c < nc; ++c) {
            const float sp[3] = { corr_src_xyz[c * 3 + 0], corr_src_xyz[c * 3 + 1], corr_src_xyz[c * 3 + 2] };
            const float tp[3] = { corr_tgt_xyz[c * 3 + 0], corr_tgt_xyz[c * 3 + 1], corr_tgt_xyz[c * 3 + 2] };
            float xp[3]; apply_rigid(T, sp, xp);
            if (squared_distance3(xp, tp) <= thresh2) ++inliers;
        }

        out_valid[h] = 1u;
        out_transform[h] = T;
        out_inlier_count[h] = inliers;
    }
}

// ===========================================================================
// jacobi_eigen_4x4_cpu / ransac_refit_cpu — the FULLY INDEPENDENT double-
// precision refit oracle (02.03's ransac_refine_cpu precedent, cited): its
// OWN Jacobi 4x4 solve (double precision throughout, never calling
// kernels.cuh's float rigid_fit_horn), its OWN double-precision cross-
// covariance accumulation. This is the project's designated non-tautological
// numerical cross-check on the Horn/Jacobi math (kernels.cuh's file-header
// "Twin-vs-shared ruling"): a bug living identically in BOTH the shared
// float header function AND its device transcription would NOT be caught by
// simple twin agreement (both paths call literally the same formula) — a
// SEPARATE, higher-precision, independently-typed implementation is the
// check that closes that gap, the same lesson flagship 13.03 learned the
// hard way (reference_cpu.cpp's own file-header rule, restated here).
// ---------------------------------------------------------------------------
void jacobi_eigen_4x4_cpu(const double a_in[10], double eigenvalues[4], double eigenvectors[4][4])
{
    double A[4][4] = {
        { a_in[0], a_in[1], a_in[2], a_in[3] },
        { a_in[1], a_in[4], a_in[5], a_in[6] },
        { a_in[2], a_in[5], a_in[7], a_in[8] },
        { a_in[3], a_in[6], a_in[8], a_in[9] },
    };
    double V[4][4] = { {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1} };
    const int pairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };

    for (int sweep = 0; sweep < kJacobiSweeps4 + 4; ++sweep) {   // a few extra sweeps: double precision can profitably converge tighter
        for (int pi = 0; pi < 6; ++pi) {
            const int p = pairs[pi][0], q = pairs[pi][1];
            const double apq = A[p][q];
            if (std::fabs(apq) < 1.0e-15) continue;
            const double theta = (A[q][q] - A[p][p]) / (2.0 * apq);
            const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0);
            const double s = t * c;
            const double app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq; A[q][q] = aqq + t * apq;
            A[p][q] = 0.0; A[q][p] = 0.0;
            for (int i = 0; i < 4; ++i) {
                if (i == p || i == q) continue;
                const double aip = A[i][p], aiq = A[i][q];
                A[i][p] = c * aip - s * aiq; A[p][i] = A[i][p];
                A[i][q] = s * aip + c * aiq; A[q][i] = A[i][q];
            }
            for (int i = 0; i < 4; ++i) {
                const double vip = V[i][p], viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        eigenvalues[i] = A[i][i];
        for (int j = 0; j < 4; ++j) eigenvectors[i][j] = V[j][i];
    }
}

bool ransac_refit_cpu(int count, const float* src_xyz, const float* tgt_xyz, float R_out[9], float t_out[3])
{
    if (count < 3) return false;
    double cs[3] = { 0, 0, 0 }, ct[3] = { 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        cs[0] += src_xyz[i * 3 + 0]; cs[1] += src_xyz[i * 3 + 1]; cs[2] += src_xyz[i * 3 + 2];
        ct[0] += tgt_xyz[i * 3 + 0]; ct[1] += tgt_xyz[i * 3 + 1]; ct[2] += tgt_xyz[i * 3 + 2];
    }
    cs[0] /= count; cs[1] /= count; cs[2] /= count;
    ct[0] /= count; ct[1] /= count; ct[2] /= count;

    double M[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const double sx = src_xyz[i * 3 + 0] - cs[0], sy = src_xyz[i * 3 + 1] - cs[1], sz = src_xyz[i * 3 + 2] - cs[2];
        const double tx = tgt_xyz[i * 3 + 0] - ct[0], ty = tgt_xyz[i * 3 + 1] - ct[1], tz = tgt_xyz[i * 3 + 2] - ct[2];
        M[0] += sx * tx; M[1] += sx * ty; M[2] += sx * tz;
        M[3] += sy * tx; M[4] += sy * ty; M[5] += sy * tz;
        M[6] += sz * tx; M[7] += sz * ty; M[8] += sz * tz;
    }
    const double trace_abs = std::fabs(M[0]) + std::fabs(M[4]) + std::fabs(M[8]);
    if (trace_abs < 1.0e-12) return false;

    const double Sxx = M[0], Sxy = M[1], Sxz = M[2], Syx = M[3], Syy = M[4], Syz = M[5], Szx = M[6], Szy = M[7], Szz = M[8];
    const double n_packed[10] = {
        Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
        Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
        -Sxx + Syy - Szz, Syz + Szy,
        -Sxx - Syy + Szz
    };
    double eigenvalues[4]; double eigenvectors[4][4];
    jacobi_eigen_4x4_cpu(n_packed, eigenvalues, eigenvectors);

    int best = 0;
    for (int i = 1; i < 4; ++i) if (eigenvalues[i] > eigenvalues[best]) best = i;
    double qw = eigenvectors[best][0], qx = eigenvectors[best][1], qy = eigenvectors[best][2], qz = eigenvectors[best][3];
    const double qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qn < 1.0e-15) return false;
    qw /= qn; qx /= qn; qy /= qn; qz /= qn;

    double R[9];
    R[0] = 1.0 - 2.0 * (qy * qy + qz * qz);  R[1] = 2.0 * (qx * qy - qw * qz);         R[2] = 2.0 * (qx * qz + qw * qy);
    R[3] = 2.0 * (qx * qy + qw * qz);         R[4] = 1.0 - 2.0 * (qx * qx + qz * qz);  R[5] = 2.0 * (qy * qz - qw * qx);
    R[6] = 2.0 * (qx * qz - qw * qy);         R[7] = 2.0 * (qy * qz + qw * qx);         R[8] = 1.0 - 2.0 * (qx * qx + qy * qy);
    for (int i = 0; i < 9; ++i) R_out[i] = static_cast<float>(R[i]);

    const double tx = ct[0] - (R[0] * cs[0] + R[1] * cs[1] + R[2] * cs[2]);
    const double ty = ct[1] - (R[3] * cs[0] + R[4] * cs[1] + R[5] * cs[2]);
    const double tz = ct[2] - (R[6] * cs[0] + R[7] * cs[1] + R[8] * cs[2]);
    t_out[0] = static_cast<float>(tx); t_out[1] = static_cast<float>(ty); t_out[2] = static_cast<float>(tz);
    return true;
}

// ===========================================================================
// STAGE 6 — ICP CPU twins (02.06's identical independence ruling, cited):
// sequential, DOUBLE-precision accumulation, own loops.
// ---------------------------------------------------------------------------
void transform_cloud_cpu(int n, const float* src_xyz, const Rigid3& T, float* out_xyz)
{
    for (int i = 0; i < n; ++i) {
        const float p[3] = { src_xyz[i * 3 + 0], src_xyz[i * 3 + 1], src_xyz[i * 3 + 2] };
        float o[3]; apply_rigid(T, p, o);
        out_xyz[i * 3 + 0] = o[0]; out_xyz[i * 3 + 1] = o[1]; out_xyz[i * 3 + 2] = o[2];
    }
}

void icp_correspondences_cpu(int n_src, const float* cur_xyz, int n_tgt, const float* tgt_xyz,
                             float max_dist_m, int32_t* out_corr_idx, float* out_corr_dist2)
{
    const float gate2 = max_dist_m * max_dist_m;
    for (int s = 0; s < n_src; ++s) {
        const float p[3] = { cur_xyz[s * 3 + 0], cur_xyz[s * 3 + 1], cur_xyz[s * 3 + 2] };
        float best_d2 = 3.0e38f;
        int best_t = -1;
        for (int t = 0; t < n_tgt; ++t) {
            const float q[3] = { tgt_xyz[t * 3 + 0], tgt_xyz[t * 3 + 1], tgt_xyz[t * 3 + 2] };
            const float d2 = squared_distance3(p, q);
            if (d2 < best_d2) { best_d2 = d2; best_t = t; }
        }
        out_corr_idx[s] = (best_d2 <= gate2) ? best_t : -1;
        out_corr_dist2[s] = best_d2;
    }
}

// icp_accumulate_cpu — sequential double accumulation, NO atomics needed
// (single thread, deterministic summation order — the more precise of the
// two paths by construction, exactly 02.06's build_normal_system_cpu note).
void icp_accumulate_cpu(int n_src, const float* cur_xyz, const float* tgt_xyz, const float* tgt_normal,
                        const int32_t* corr_idx, double accum27[27])
{
    for (int i = 0; i < 27; ++i) accum27[i] = 0.0;
    const int row_start[6] = { 0, 6, 11, 15, 18, 20 };

    for (int s = 0; s < n_src; ++s) {
        const int t = corr_idx[s];
        if (t < 0) continue;
        const double x[3] = { cur_xyz[s * 3 + 0], cur_xyz[s * 3 + 1], cur_xyz[s * 3 + 2] };
        const double q[3] = { tgt_xyz[t * 3 + 0], tgt_xyz[t * 3 + 1], tgt_xyz[t * 3 + 2] };
        const double nrm[3] = { tgt_normal[t * 3 + 0], tgt_normal[t * 3 + 1], tgt_normal[t * 3 + 2] };
        const double diff[3] = { x[0] - q[0], x[1] - q[1], x[2] - q[2] };
        const double residual = nrm[0] * diff[0] + nrm[1] * diff[1] + nrm[2] * diff[2];
        const double J[6] = {
            x[1] * nrm[2] - x[2] * nrm[1],
            x[2] * nrm[0] - x[0] * nrm[2],
            x[0] * nrm[1] - x[1] * nrm[0],
            nrm[0], nrm[1], nrm[2]
        };
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) accum27[row_start[i] + (j - i)] += J[i] * J[j];
            accum27[21 + i] += J[i] * residual;
        }
    }
}
