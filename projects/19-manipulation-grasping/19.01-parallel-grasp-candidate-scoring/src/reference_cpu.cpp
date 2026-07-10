// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 19.01
//                     (Parallel grasp-candidate scoring: antipodal sampling
//                     over point clouds)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5):
//
//   1) It is the CORRECTNESS ORACLE. Each function below is a line-by-line
//      sequential TWIN of its kernels.cu counterpart — same formulas, same
//      loop order, same tie-breaking policy — so main.cu's VERIFY stage can
//      assert GPU-vs-CPU agreement with confidence that any mismatch is a
//      real bug, not an apples-to-oranges comparison.
//   2) It is the TEACHING BASELINE. Reading estimate_normals_cpu next to
//      estimate_normals_kernel (or generate_candidates_cpu next to
//      generate_candidates_kernel) shows exactly what parallelization
//      changed: the outer "for every point" loop became "one thread per
//      point"; the per-point body is otherwise identical.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong, and
// then the oracle lies. (This file is compiled by the HOST compiler, cl.exe;
// the __CUDACC__ fence in kernels.cuh hides device-only declarations from it
// — grasp_hash_u32 is the one function BOTH sides of that fence define, so
// this file and kernels.cu call bit-identical code for the one stage
// required to match exactly — see kernels.cuh's grasp_hash_u32 comment.)
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cfloat>    // FLT_MAX
#include <cmath>     // std::sqrt, std::acos, std::atan, std::fabs

namespace {

// ---------------------------------------------------------------------------
// jacobi_rotate_cpu / jacobi_eigen_3x3_cpu — sequential twin of kernels.cu's
// jacobi_rotate / jacobi_eigen_3x3 (itself reused from project 02.06 — see
// kernels.cu's header comment). Same formulas; std::sqrt instead of CUDA's
// rsqrtf/sqrtf intrinsics (THEORY.md "Numerical considerations" names the
// resulting sub-ULP divergence and why the VERIFY tolerance absorbs it).
// ---------------------------------------------------------------------------
void jacobi_rotate_cpu(float A[3][3], float V[3][3], int p, int q)
{
    const float apq = A[p][q];
    if (std::fabs(apq) < 1e-12f) return;

    const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
    const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
    const float c = 1.0f / std::sqrt(t * t + 1.0f);
    const float s = t * c;

    const float app = A[p][p], aqq = A[q][q];
    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0.0f;
    A[q][p] = 0.0f;

    for (int r = 0; r < 3; ++r) {
        if (r == p || r == q) continue;
        const float arp = A[r][p], arq = A[r][q];
        A[r][p] = c * arp - s * arq;  A[p][r] = A[r][p];
        A[r][q] = s * arp + c * arq;  A[q][r] = A[r][q];
    }
    for (int r = 0; r < 3; ++r) {
        const float vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq;
        V[r][q] = s * vrp + c * vrq;
    }
}

void jacobi_eigen_3x3_cpu(float A[3][3], float V[3][3])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            V[i][j] = (i == j) ? 1.0f : 0.0f;

    for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
        jacobi_rotate_cpu(A, V, 0, 1);
        jacobi_rotate_cpu(A, V, 0, 2);
        jacobi_rotate_cpu(A, V, 1, 2);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// estimate_normals_cpu — sequential twin of estimate_normals_kernel. Same
// brute-force k-NN maintenance policy (unsorted array + tracked worst
// slot), same covariance + Jacobi + OUTWARD orientation steps (kernels.cu's
// estimate_normals_kernel header comment explains the outward-vs-02.06's-
// inward policy).
// ---------------------------------------------------------------------------
void estimate_normals_cpu(int n, const float* xyz, const float ref_point[3], float* normals)
{
    for (int j = 0; j < n; ++j) {
        const float qx = xyz[j * 3 + 0];
        const float qy = xyz[j * 3 + 1];
        const float qz = xyz[j * 3 + 2];

        float nb_d2[kPcaK], nb_x[kPcaK], nb_y[kPcaK], nb_z[kPcaK];
        for (int i = 0; i < kPcaK; ++i) nb_d2[i] = FLT_MAX;
        int   worst    = 0;
        float worst_d2 = FLT_MAX;

        for (int m = 0; m < n; ++m) {
            const float mx = xyz[m * 3 + 0];
            const float my = xyz[m * 3 + 1];
            const float mz = xyz[m * 3 + 2];
            const float dx = mx - qx, dy = my - qy, dz = mz - qz;
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < worst_d2) {
                nb_d2[worst] = d2; nb_x[worst] = mx; nb_y[worst] = my; nb_z[worst] = mz;
                worst = 0; worst_d2 = nb_d2[0];
                for (int i = 1; i < kPcaK; ++i) {
                    if (nb_d2[i] > worst_d2) { worst_d2 = nb_d2[i]; worst = i; }
                }
            }
        }

        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        for (int i = 0; i < kPcaK; ++i) { cx += nb_x[i]; cy += nb_y[i]; cz += nb_z[i]; }
        const float inv_k = 1.0f / static_cast<float>(kPcaK);
        cx *= inv_k; cy *= inv_k; cz *= inv_k;

        float cxx = 0.0f, cyy = 0.0f, czz = 0.0f, cxy = 0.0f, cxz = 0.0f, cyz = 0.0f;
        for (int i = 0; i < kPcaK; ++i) {
            const float dx = nb_x[i] - cx, dy = nb_y[i] - cy, dz = nb_z[i] - cz;
            cxx += dx * dx; cyy += dy * dy; czz += dz * dz;
            cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
        }

        float A[3][3] = { { cxx, cxy, cxz }, { cxy, cyy, cyz }, { cxz, cyz, czz } };
        float V[3][3];
        jacobi_eigen_3x3_cpu(A, V);

        int lo = 0;
        if (A[1][1] < A[lo][lo]) lo = 1;
        if (A[2][2] < A[lo][lo]) lo = 2;
        float nx = V[0][lo], ny = V[1][lo], nz = V[2][lo];

        const float inv_len = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
        nx *= inv_len; ny *= inv_len; nz *= inv_len;

        // OUTWARD orientation: flip toward-ref normals away from ref_point
        // (mirrors estimate_normals_kernel's flipped inequality exactly).
        const float to_ref_x = ref_point[0] - qx, to_ref_y = ref_point[1] - qy, to_ref_z = ref_point[2] - qz;
        if (nx * to_ref_x + ny * to_ref_y + nz * to_ref_z > 0.0f) {
            nx = -nx; ny = -ny; nz = -nz;
        }

        normals[j * 3 + 0] = nx;
        normals[j * 3 + 1] = ny;
        normals[j * 3 + 2] = nz;
    }
}

// ---------------------------------------------------------------------------
// generate_candidates_cpu — sequential twin of generate_candidates_kernel.
// Calls the SAME grasp_hash_u32 (kernels.cuh's #ifdef __CUDACC__ fence
// gives this file the plain-`inline` copy of the identical function body)
// and applies the identical search formulas in the identical order, so
// idx1/idx2 are expected to match the GPU kernel EXACTLY (main.cu's VERIFY
// stage checks this with zero tolerance, not a rel/abs bound).
// ---------------------------------------------------------------------------
void generate_candidates_cpu(int n, const float* xyz, const float* normals,
                             unsigned int seed, int num_candidates,
                             GraspCandidate* candidates)
{
    for (int k = 0; k < num_candidates; ++k) {
        const unsigned int idx1u = grasp_hash_u32(seed, static_cast<unsigned int>(k)) % static_cast<unsigned int>(n);
        const int idx1 = static_cast<int>(idx1u);

        const float p1x = xyz[idx1 * 3 + 0];
        const float p1y = xyz[idx1 * 3 + 1];
        const float p1z = xyz[idx1 * 3 + 2];
        const float n1x = normals[idx1 * 3 + 0];
        const float n1y = normals[idx1 * 3 + 1];
        const float n1z = normals[idx1 * 3 + 2];

        float best_perp2 = FLT_MAX;
        int   best_j     = -1;

        for (int j = 0; j < n; ++j) {
            if (j == idx1) continue;

            const float qx = xyz[j * 3 + 0];
            const float qy = xyz[j * 3 + 1];
            const float qz = xyz[j * 3 + 2];
            const float dx = qx - p1x, dy = qy - p1y, dz = qz - p1z;

            const float t = -(dx * n1x + dy * n1y + dz * n1z);
            if (t < kSearchTMinM || t > kSearchTMaxM) continue;

            const float perpx = dx + t * n1x;
            const float perpy = dy + t * n1y;
            const float perpz = dz + t * n1z;
            const float perp2 = perpx * perpx + perpy * perpy + perpz * perpz;
            if (perp2 > kSearchPerpTolM * kSearchPerpTolM) continue;

            const float njx = normals[j * 3 + 0];
            const float njy = normals[j * 3 + 1];
            const float njz = normals[j * 3 + 2];
            const float dotn = n1x * njx + n1y * njy + n1z * njz;
            if (dotn > kGenConeCosThreshold) continue;

            if (perp2 < best_perp2) {
                best_perp2 = perp2;
                best_j = j;
            }
        }

        candidates[k].idx1 = idx1;
        candidates[k].idx2 = best_j;
    }
}

// ---------------------------------------------------------------------------
// score_candidates_cpu — sequential twin of score_candidates_kernel. Same
// three gates (friction cone, width, clearance), same field layout.
// ---------------------------------------------------------------------------
void score_candidates_cpu(int n, const float* xyz, const float* normals,
                          const GraspCandidate* candidates, int num_candidates,
                          float mu, float w_min_m, float w_max_m,
                          GraspScore* scores)
{
    const float kPi = 3.14159265358979323846f;

    for (int c = 0; c < num_candidates; ++c) {
        const int idx1 = candidates[c].idx1;
        const int idx2 = candidates[c].idx2;

        GraspScore out;
        out.width_m = 0.0f; out.antipodal_cos = 0.0f;
        out.theta1_deg = 180.0f; out.theta2_deg = 180.0f;
        out.friction_ok = 0; out.width_ok = 0; out.clearance_ok = 0; out.feasible = 0;
        out.score = kRejectedScore;

        if (idx2 >= 0) {
            const float p1x = xyz[idx1 * 3 + 0], p1y = xyz[idx1 * 3 + 1], p1z = xyz[idx1 * 3 + 2];
            const float p2x = xyz[idx2 * 3 + 0], p2y = xyz[idx2 * 3 + 1], p2z = xyz[idx2 * 3 + 2];
            const float n1x = normals[idx1 * 3 + 0], n1y = normals[idx1 * 3 + 1], n1z = normals[idx1 * 3 + 2];
            const float n2x = normals[idx2 * 3 + 0], n2y = normals[idx2 * 3 + 1], n2z = normals[idx2 * 3 + 2];

            const float dx = p2x - p1x, dy = p2y - p1y, dz = p2z - p1z;
            const float width = std::sqrt(dx * dx + dy * dy + dz * dz);
            out.width_m = width;

            const float inv_w = 1.0f / width;
            const float ax = dx * inv_w, ay = dy * inv_w, az = dz * inv_w;

            const float dot_n1n2 = n1x * n2x + n1y * n2y + n1z * n2z;
            out.antipodal_cos = -dot_n1n2;

            float cos_t1 = -(ax * n1x + ay * n1y + az * n1z);
            cos_t1 = std::fmin(1.0f, std::fmax(-1.0f, cos_t1));
            out.theta1_deg = std::acos(cos_t1) * (180.0f / kPi);

            float cos_t2 = ax * n2x + ay * n2y + az * n2z;
            cos_t2 = std::fmin(1.0f, std::fmax(-1.0f, cos_t2));
            out.theta2_deg = std::acos(cos_t2) * (180.0f / kPi);

            const float alpha_deg = std::atan(mu) * (180.0f / kPi);
            out.friction_ok = (out.theta1_deg <= alpha_deg && out.theta2_deg <= alpha_deg) ? 1 : 0;

            out.width_ok = (width >= w_min_m && width <= w_max_m) ? 1 : 0;

            unsigned char clearance_ok = 1;
            for (int j = 0; j < n; ++j) {
                if (j == idx1 || j == idx2) continue;
                const float qx = xyz[j * 3 + 0], qy = xyz[j * 3 + 1], qz = xyz[j * 3 + 2];
                const float rx = qx - p1x, ry = qy - p1y, rz = qz - p1z;
                const float t = rx * ax + ry * ay + rz * az;
                if (t < kClearanceDeadzoneM || t > (width - kClearanceDeadzoneM)) continue;
                const float perpx = rx - t * ax;
                const float perpy = ry - t * ay;
                const float perpz = rz - t * az;
                const float perp2 = perpx * perpx + perpy * perpy + perpz * perpz;
                if (perp2 < kClearanceRadiusM * kClearanceRadiusM) { clearance_ok = 0; break; }
            }
            out.clearance_ok = clearance_ok;

            out.feasible = (out.friction_ok && out.width_ok && out.clearance_ok) ? 1 : 0;
            out.score = out.feasible ? out.antipodal_cos : kRejectedScore;
        }

        scores[c] = out;
    }
}
