// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.06
//                     ICP: point-to-point → point-to-plane → GICP, all batched
//
// The correctness ORACLE for every kernel in kernels.cu — sequential, plain
// C++17, no CUDA anywhere (CLAUDE.md §5). main.cu runs this against the GPU
// path on iteration 0's exact inputs and requires agreement within the
// documented tolerances (the VERIFY stage; THEORY.md §how-we-verify).
//
// Every function below is a deliberate, DOCUMENTED line-by-line twin of its
// __device__/__global__ counterpart in kernels.cu — same formulas, same
// layouts, only the float-function spellings and (for build_normal_system_cpu)
// the accumulator precision differ. Diff the two files side by side; where
// they disagree is where a bug would hide.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared layouts, constants, signatures

#include <cfloat>        // FLT_MAX — identical sentinel to kernels.cu
#include <cmath>         // std::sqrt, std::fabs

// ---------------------------------------------------------------------------
// transform_cloud_cpu — sequential twin of transform_cloud_kernel.
// ---------------------------------------------------------------------------
void transform_cloud_cpu(int n, const float* src_xyz, const Rigid3& T, float* out_xyz)
{
    for (int k = 0; k < n; ++k) {
        const float px = src_xyz[k * 3 + 0];
        const float py = src_xyz[k * 3 + 1];
        const float pz = src_xyz[k * 3 + 2];
        out_xyz[k * 3 + 0] = T.R[0] * px + T.R[1] * py + T.R[2] * pz + T.t[0];
        out_xyz[k * 3 + 1] = T.R[3] * px + T.R[4] * py + T.R[5] * pz + T.t[1];
        out_xyz[k * 3 + 2] = T.R[6] * px + T.R[7] * py + T.R[8] * pz + T.t[2];
    }
}

// ---------------------------------------------------------------------------
// find_correspondences_cpu — sequential twin of find_correspondences_kernel.
//
// Same scan order (m = 0..m_tgt-1), same strict '<' tie rule, so the
// EXACT SAME index wins on both paths for well-separated data — this is
// what main.cu's VERIFY stage checks bit-exactly (index equality, not a
// tolerance) at iteration 0, where the initial misalignment keeps every
// true nearest neighbor far from its runner-up (THEORY.md §how-we-verify).
// ---------------------------------------------------------------------------
void find_correspondences_cpu(int n_src, const float* cur_xyz,
                              int m_tgt, const float* tgt_xyz,
                              float max_dist_m,
                              int* corr_idx, float* corr_dist2)
{
    const float max_dist2 = max_dist_m * max_dist_m;
    for (int k = 0; k < n_src; ++k) {
        const float px = cur_xyz[k * 3 + 0];
        const float py = cur_xyz[k * 3 + 1];
        const float pz = cur_xyz[k * 3 + 2];

        float best_d2 = FLT_MAX;
        int   best_j  = -1;
        for (int m = 0; m < m_tgt; ++m) {
            const float dx = tgt_xyz[m * 3 + 0] - px;
            const float dy = tgt_xyz[m * 3 + 1] - py;
            const float dz = tgt_xyz[m * 3 + 2] - pz;
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_d2) { best_d2 = d2; best_j = m; }
        }

        corr_idx[k]   = (best_j >= 0 && best_d2 <= max_dist2) ? best_j : -1;
        corr_dist2[k] = best_d2;
    }
}

// ---------------------------------------------------------------------------
// jacobi_eigen_3x3_cpu — sequential twin of kernels.cu's jacobi_eigen_3x3 /
// jacobi_rotate. Identical algorithm (Numerical Recipes cyclic Jacobi on
// the fixed three off-diagonal pairs of a 3x3 symmetric matrix); the only
// difference is std::sqrt/std::fabs instead of the device intrinsics.
// ---------------------------------------------------------------------------
static void jacobi_rotate_cpu(float A[3][3], float V[3][3], int p, int q)
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

static void jacobi_eigen_3x3_cpu(float A[3][3], float V[3][3])
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

// ---------------------------------------------------------------------------
// estimate_normals_cpu — sequential twin of estimate_normals_kernel. Same
// brute-force k-NN maintenance policy (unsorted array + tracked worst slot),
// same covariance + Jacobi + orientation steps.
// ---------------------------------------------------------------------------
void estimate_normals_cpu(int m_tgt, const float* tgt_xyz,
                          const float ref_point[3], float* tgt_normals)
{
    for (int j = 0; j < m_tgt; ++j) {
        const float qx = tgt_xyz[j * 3 + 0];
        const float qy = tgt_xyz[j * 3 + 1];
        const float qz = tgt_xyz[j * 3 + 2];

        float nb_d2[kPcaK], nb_x[kPcaK], nb_y[kPcaK], nb_z[kPcaK];
        for (int i = 0; i < kPcaK; ++i) nb_d2[i] = FLT_MAX;
        int   worst    = 0;
        float worst_d2 = FLT_MAX;

        for (int m = 0; m < m_tgt; ++m) {
            const float mx = tgt_xyz[m * 3 + 0];
            const float my = tgt_xyz[m * 3 + 1];
            const float mz = tgt_xyz[m * 3 + 2];
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

        const float to_ref_x = ref_point[0] - qx, to_ref_y = ref_point[1] - qy, to_ref_z = ref_point[2] - qz;
        if (nx * to_ref_x + ny * to_ref_y + nz * to_ref_z < 0.0f) {
            nx = -nx; ny = -ny; nz = -nz;
        }

        tgt_normals[j * 3 + 0] = nx;
        tgt_normals[j * 3 + 1] = ny;
        tgt_normals[j * 3 + 2] = nz;
    }
}

// ---------------------------------------------------------------------------
// build_normal_system_cpu — sequential twin of build_normal_system_kernel,
// BOTH modes, with the numerically stronger accumulation policy documented
// in kernels.cuh: each point's 27-scalar contribution is computed with the
// SAME float32 formulas the kernel uses (so a genuine formula bug shows up
// at order-1 relative error, not buried in accumulation noise), but the
// running SUM across all n_src points is kept in DOUBLE — skipping the
// GPU path's block-tree-then-host-sum reduction entirely, so this oracle is
// strictly more precise than the path it is checking (THEORY.md §numerics).
// ---------------------------------------------------------------------------
void build_normal_system_cpu(int n_src, const float* cur_xyz,
                             const float* tgt_xyz, const float* tgt_normals,
                             const int* corr_idx, IcpMode mode,
                             double H21[21], double g6[6])
{
    for (int i = 0; i < 21; ++i) H21[i] = 0.0;
    for (int i = 0; i < 6; ++i)  g6[i]  = 0.0;

    for (int k = 0; k < n_src; ++k) {
        const int j = corr_idx[k];
        if (j < 0) continue;   // rejected correspondence: contributes nothing

        const float x0 = cur_xyz[k * 3 + 0], x1 = cur_xyz[k * 3 + 1], x2 = cur_xyz[k * 3 + 2];
        const float q0 = tgt_xyz[j * 3 + 0], q1 = tgt_xyz[j * 3 + 1], q2 = tgt_xyz[j * 3 + 2];
        const float r0 = x0 - q0, r1 = x1 - q1, r2 = x2 - q2;

        float h[21];
        float g[6];

        if (mode == kPointToPoint) {
            const float xx = x0 * x0, yy = x1 * x1, zz = x2 * x2;
            const float xy = x0 * x1, xz = x0 * x2, yz = x1 * x2;
            const float x2n = xx + yy + zz;

            h[0]  = x2n - xx; h[1]  = -xy;      h[2]  = -xz;
            h[3]  = 0.0f;     h[4]  = -x2;      h[5]  = x1;
            h[6]  = x2n - yy; h[7]  = -yz;
            h[8]  = x2;       h[9]  = 0.0f;     h[10] = -x0;
            h[11] = x2n - zz;
            h[12] = -x1;      h[13] = x0;       h[14] = 0.0f;
            h[15] = 1.0f;     h[16] = 0.0f;     h[17] = 0.0f;
            h[18] = 1.0f;     h[19] = 0.0f;
            h[20] = 1.0f;
            g[0] = x1 * r2 - x2 * r1;
            g[1] = x2 * r0 - x0 * r2;
            g[2] = x0 * r1 - x1 * r0;
            g[3] = r0; g[4] = r1; g[5] = r2;
        } else {   // kPointToPlane
            const float n0 = tgt_normals[j * 3 + 0];
            const float n1 = tgt_normals[j * 3 + 1];
            const float n2 = tgt_normals[j * 3 + 2];
            const float e0 = n0 * r0 + n1 * r1 + n2 * r2;

            const float a0 = x1 * n2 - x2 * n1;
            const float a1 = x2 * n0 - x0 * n2;
            const float a2 = x0 * n1 - x1 * n0;

            h[0]  = a0 * a0; h[1]  = a0 * a1; h[2]  = a0 * a2;
            h[3]  = a0 * n0; h[4]  = a0 * n1; h[5]  = a0 * n2;
            h[6]  = a1 * a1; h[7]  = a1 * a2;
            h[8]  = a1 * n0; h[9]  = a1 * n1; h[10] = a1 * n2;
            h[11] = a2 * a2;
            h[12] = a2 * n0; h[13] = a2 * n1; h[14] = a2 * n2;
            h[15] = n0 * n0; h[16] = n0 * n1; h[17] = n0 * n2;
            h[18] = n1 * n1; h[19] = n1 * n2;
            h[20] = n2 * n2;
            g[0] = a0 * e0; g[1] = a1 * e0; g[2] = a2 * e0;
            g[3] = n0 * e0; g[4] = n1 * e0; g[5] = n2 * e0;
        }

        // Accumulate this point's float32-computed contribution into the
        // double running sum — the ONE place this function is more precise
        // than the GPU path (kernels.cuh explains why that is by design).
        for (int c = 0; c < 21; ++c) H21[c] += static_cast<double>(h[c]);
        for (int c = 0; c < 6;  ++c) g6[c]  += static_cast<double>(g[c]);
    }
}
