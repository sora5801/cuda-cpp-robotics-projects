// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.16
//                     (Multi-LiDAR merging + extrinsic refinement)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// The template's two load-bearing reasons (CLAUDE.md §5) — correctness
// oracle and teaching baseline — restated for this project in
// docs/PROJECT_TEMPLATE/src/reference_cpu.cpp; read that header first if
// you have not already.
//
// Independence ruling, AS APPLIED HERE (01.17's ruling, cited verbatim,
// then worked through zone by zone for this project)
// --------------------------------------------------------------------------
//   * SHARED (kernels.cuh, CALIB_HD): the rigid-transform primitives
//     (so3_exp, mat3_vec/mul, retract, rigid3_compose/inverse) and
//     point_to_plane_residual_and_jacobian — THE formula this project
//     teaches. These are closed-form SO(3)/geometry formulas; hand
//     duplicating them would be pure transcription, exactly 01.17's
//     "system under test" exception. hidx()/blocks_for()/the voxel-key
//     packing are shared too, per 01.17's OWN precedent that data-layout
//     arithmetic (not "the algorithm") is fine to share.
//   * INDEPENDENT (written twice, on purpose):
//       - Plane fitting: fit_planes_cpu below does its OWN two-pass
//         mean-shifted-covariance accumulation (a plain sequential loop,
//         not kernels.cu's atomic scatter-reduce) AND calls its OWN
//         jacobi_eigen_3x3_cpu (NOT kernels.cuh's jacobi_eigen_3x3, which
//         main.cu uses only to decode the GPU path's covariance) — so
//         PLANE_FIT_TWIN genuinely exercises two independently-written
//         eigensolvers on independently-accumulated covariances, not one
//         algorithm compared against itself.
//       - assemble_point_to_plane_cpu: same shared residual/Jacobian
//         formula, but the ACCUMULATION LOOP (sequential, double-precision,
//         no reduction tree) is its own — this is the twin comparison
//         01.17/02.06 both use, applied here.
//       - run_refinement_lm_cpu: its OWN full LM trajectory — own damping/
//         accept-reject control flow — sharing only the residual/Jacobian
//         formula and the host-only cholesky6_solve (01.17's exception:
//         neither caller is "the GPU path", both are host orchestration
//         calling a generic textbook SPD solve).
//       - dedup_voxel_grid_cpu: an entirely different DATA STRUCTURE
//         (std::unordered_map, 02.09's HashMapCpu precedent, cited) than
//         the GPU's sort-and-compact pipeline — the two share only the
//         voxel-key FORMULA (kernels.cuh's pack_voxel_key), a data-layout
//         contract, not an algorithm.
//
// Why this is not paranoia (01.17's own warning, worth repeating): a bug
// that lives inside SHARED code is invisible to a twin comparison by
// construction — both sides compute the same wrong answer. This project's
// INDEPENDENT gate on the shared point_to_plane_residual_and_jacobian
// formula is the RECOVERY and ZERO_DRIFT_CONTROL gates: both compare a
// refined extrinsic against ground truth that scripts/make_synthetic.py
// computed in a COMPLETELY SEPARATE program (Python, not even the same
// language) — if the shared residual/Jacobian formula had a sign error,
// the LM solve would not converge to the true answer, and RECOVERY would
// fail even though every twin gate above it passed. Exactly 01.17's
// "zero-noise sanity gate is also the independent check on the shared
// camera-model formula" argument, restated for this project's geometry.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// transform_points_cpu — the twin of transform_points_kernel. A plain
// sequential loop; the formula itself (mat3_vec + add t) is the shared
// primitive, but the LOOP that drives it is written here independently
// (the GPU version is a per-thread map, not a host loop calling a device
// function — there is no code to accidentally share at the loop level).
// ---------------------------------------------------------------------------
void transform_points_cpu(int n, const float* src_xyz, const Rigid3& T, float* out_xyz)
{
    for (int i = 0; i < n; ++i) {
        const float p[3] = { src_xyz[i * 3 + 0], src_xyz[i * 3 + 1], src_xyz[i * 3 + 2] };
        float Rp[3];
        mat3_vec(T.R, p, Rp);
        out_xyz[i * 3 + 0] = Rp[0] + T.t[0];
        out_xyz[i * 3 + 1] = Rp[1] + T.t[1];
        out_xyz[i * 3 + 2] = Rp[2] + T.t[2];
    }
}

// ---------------------------------------------------------------------------
// jacobi_eigen_3x3_cpu — the CPU oracle's OWN cyclic-Jacobi 3x3 eigensolve.
// Same ALGORITHM FAMILY as kernels.cuh's jacobi_eigen_3x3 (02.03/02.09/
// 01.17 lineage, cited) but a SEPARATE implementation (see this file's
// header for why that separation matters for PLANE_FIT_TWIN).
// ---------------------------------------------------------------------------
void jacobi_eigen_3x3_cpu(const float cov[6], float eigenvalues[3], float eigenvectors[3][3])
{
    double A[3][3] = {
        { cov[0], cov[1], cov[2] },
        { cov[1], cov[3], cov[4] },
        { cov[2], cov[4], cov[5] }
    };
    double V[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };

    const int kSweeps = 8;
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(A[p][q]) < 1e-18) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                const double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A[p][q] = A[q][p] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = A[p][k] = c * akp - s * akq;
                    A[k][q] = A[q][k] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    int order[3] = { 0, 1, 2 };
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (A[order[j]][order[j]] < A[order[i]][order[i]]) { int tmp = order[i]; order[i] = order[j]; order[j] = tmp; }

    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = static_cast<float>(A[order[i]][order[i]]);
        for (int k = 0; k < 3; ++k) eigenvectors[i][k] = static_cast<float>(V[k][order[i]]);
    }
}

// ---------------------------------------------------------------------------
// fit_planes_cpu — the independent oracle's own two-pass (centroid, then
// mean-shifted covariance) accumulation, one surface at a time, followed by
// this file's OWN jacobi_eigen_3x3_cpu. Sign-orients toward
// kPlaneOrientRef exactly like kernels.cu's GPU decode step (a shared
// CONVENTION, not shared code — see kernels.cuh's Plane doc for why one
// fixed reference point correctly orients all four orthogonal surfaces).
// ---------------------------------------------------------------------------
void fit_planes_cpu(int n, const float* xyz, const int32_t* surface_id, Plane out_planes[kNumSurfaces])
{
    double sums[kNumSurfaces][3] = {};
    int32_t counts[kNumSurfaces] = {};
    for (int i = 0; i < n; ++i) {
        const int32_t s = surface_id[i];
        if (s < 0 || s >= kNumSurfaces) continue;
        sums[s][0] += xyz[i * 3 + 0];
        sums[s][1] += xyz[i * 3 + 1];
        sums[s][2] += xyz[i * 3 + 2];
        counts[s] += 1;
    }

    double centroid[kNumSurfaces][3] = {};
    for (int s = 0; s < kNumSurfaces; ++s) {
        out_planes[s] = kInvalidPlane;
        if (counts[s] < kMinPlanePoints) continue;
        centroid[s][0] = sums[s][0] / counts[s];
        centroid[s][1] = sums[s][1] / counts[s];
        centroid[s][2] = sums[s][2] / counts[s];
    }

    double cov[kNumSurfaces][6] = {};   // c00,c01,c02,c11,c12,c22
    for (int i = 0; i < n; ++i) {
        const int32_t s = surface_id[i];
        if (s < 0 || s >= kNumSurfaces || counts[s] < kMinPlanePoints) continue;
        const double dx = xyz[i * 3 + 0] - centroid[s][0];
        const double dy = xyz[i * 3 + 1] - centroid[s][1];
        const double dz = xyz[i * 3 + 2] - centroid[s][2];
        cov[s][0] += dx * dx; cov[s][1] += dx * dy; cov[s][2] += dx * dz;
        cov[s][3] += dy * dy; cov[s][4] += dy * dz; cov[s][5] += dz * dz;
    }

    for (int s = 0; s < kNumSurfaces; ++s) {
        if (counts[s] < kMinPlanePoints) continue;
        float cov_f[6];
        for (int k = 0; k < 6; ++k) cov_f[k] = static_cast<float>(cov[s][k] / counts[s]);

        float eigenvalues[3], eigenvectors[3][3];
        jacobi_eigen_3x3_cpu(cov_f, eigenvalues, eigenvectors);   // ascending: [0] is the plane normal

        float normal[3] = { eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2] };
        const float cx = static_cast<float>(centroid[s][0]);
        const float cy = static_cast<float>(centroid[s][1]);
        const float cz = static_cast<float>(centroid[s][2]);
        const float ref_dot = normal[0] * (kPlaneOrientRef[0] - cx) +
                              normal[1] * (kPlaneOrientRef[1] - cy) +
                              normal[2] * (kPlaneOrientRef[2] - cz);
        if (ref_dot < 0.0f) { normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; }

        out_planes[s].normal[0] = normal[0]; out_planes[s].normal[1] = normal[1]; out_planes[s].normal[2] = normal[2];
        out_planes[s].centroid[0] = cx; out_planes[s].centroid[1] = cy; out_planes[s].centroid[2] = cz;
        out_planes[s].valid = 1;
        out_planes[s].count = counts[s];
    }
}

// ---------------------------------------------------------------------------
// assemble_point_to_plane_cpu — the twin of assemble_point_to_plane_kernel:
// shared residual/Jacobian formula, INDEPENDENT sequential double-precision
// accumulation loop (no reduction tree, no block partials — a single
// running H21/g6/cost, exactly 01.17's assemble_normal_equations_cpu shape).
// ---------------------------------------------------------------------------
void assemble_point_to_plane_cpu(const float* p_src, const int32_t* surface_id, int n,
                                 const Rigid3& T, const Plane target_planes[kNumSurfaces], uint32_t zone_mask,
                                 double H21[21], double g6[6], double* cost_out)
{
    for (int a = 0; a < 21; ++a) H21[a] = 0.0;
    for (int a = 0; a < 6; ++a) g6[a] = 0.0;
    double cost = 0.0;

    for (int i = 0; i < n; ++i) {
        const int32_t s = surface_id[i];
        if (s < 0 || s >= kNumSurfaces) continue;
        if (!((zone_mask >> s) & 1u)) continue;
        if (!target_planes[s].valid) continue;

        const float p[3] = { p_src[i * 3 + 0], p_src[i * 3 + 1], p_src[i * 3 + 2] };
        float r, J[6];
        point_to_plane_residual_and_jacobian(T, p, target_planes[s].normal, target_planes[s].centroid, r, J);

        for (int a = 0; a < 6; ++a)
            for (int b = a; b < 6; ++b)
                H21[hidx(a, b)] += static_cast<double>(J[a]) * static_cast<double>(J[b]);
        for (int a = 0; a < 6; ++a)
            g6[a] += static_cast<double>(J[a]) * static_cast<double>(r);
        cost += static_cast<double>(r) * static_cast<double>(r);
    }
    *cost_out = cost;
}

// ---------------------------------------------------------------------------
// run_refinement_lm_cpu — an INDEPENDENTLY-WRITTEN full Levenberg-Marquardt
// trajectory: own loop, own damping/accept-reject flow (mirrors main.cu's
// GPU-orchestrated loop in STRUCTURE only — no code is shared beyond the
// residual/Jacobian formula and the host-only cholesky6_solve, per this
// file's header). Used by main.cu's TRAJECTORY_TWIN gate.
// ---------------------------------------------------------------------------
void run_refinement_lm_cpu(const float* p_src, const int32_t* surface_id, int n,
                           const Plane target_planes[kNumSurfaces], uint32_t zone_mask,
                           Rigid3 T_init, int max_iters,
                           Rigid3& out_T, double* loss_history, int& out_num_iters)
{
    Rigid3 T = T_init;
    double lambda = kLambdaInit;
    double H21[21], g6[6], cost;
    assemble_point_to_plane_cpu(p_src, surface_id, n, T, target_planes, zone_mask, H21, g6, &cost);

    int num_iters = 0;
    loss_history[num_iters++] = cost;

    for (int it = 0; it < max_iters; ++it) {
        double delta[6];
        bool ok = false;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
            ok = cholesky6_solve(H21, g6, lambda, delta);
            if (!ok) lambda *= kLambdaUp;
        }
        if (!ok) break;

        Rigid3 T_cand;
        retract(T, delta, T_cand);
        double H21n[21], g6n[6], cost_new;
        assemble_point_to_plane_cpu(p_src, surface_id, n, T_cand, target_planes, zone_mask, H21n, g6n, &cost_new);

        const double delta_norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] +
                                            delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);

        if (cost_new < cost) {
            const double rel_change = std::fabs(cost - cost_new) / (cost_new + 1.0e-12);
            T = T_cand;
            cost = cost_new;
            std::memcpy(H21, H21n, sizeof(H21));
            std::memcpy(g6, g6n, sizeof(g6));
            lambda *= kLambdaDown;
            if (lambda < kLambdaMin) lambda = kLambdaMin;
            loss_history[num_iters++] = cost;
            if (delta_norm < kConvergeDeltaNorm || rel_change < kConvergeCostRel) break;
        } else {
            lambda *= kLambdaUp;
            loss_history[num_iters++] = cost;
        }
    }

    out_T = T;
    out_num_iters = num_iters;
}

// ---------------------------------------------------------------------------
// dedup_voxel_grid_cpu — the independent oracle: an
// std::unordered_map<uint64_t,int32_t> voxel-key -> smallest-original-index
// map (02.09's HashMapCpu precedent, cited: a DIFFERENT data structure than
// the GPU's sort-and-compact pipeline, sharing only the voxel-key FORMULA).
// A single ascending pass (i = 0..n-1) means the map only ever inserts the
// FIRST (= smallest) index it sees for a given key — the identical tie-
// break rule the GPU's stable_sort_by_key path produces (kernels.cu's file
// header states this explicitly). Returns the number of unique voxels;
// out_representative_idx[0..return) holds the kept original indices, in
// the order their voxel was FIRST SEEN (not necessarily sorted by index —
// main.cu sorts both sides before an exact-set comparison).
// ---------------------------------------------------------------------------
int dedup_voxel_grid_cpu(int n, const float* xyz, float cell, int32_t* out_representative_idx)
{
    std::unordered_map<unsigned long long, int32_t> seen;
    seen.reserve(static_cast<size_t>(n) * 2);
    int num_unique = 0;
    for (int i = 0; i < n; ++i) {
        const float p[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
        const unsigned long long key = point_voxel_key(p, cell);
        if (seen.find(key) == seen.end()) {
            seen.emplace(key, i);
            out_representative_idx[num_unique++] = i;
        }
    }
    return num_unique;
}
