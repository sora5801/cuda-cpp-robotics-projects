// ===========================================================================
// kernels.cu — GPU kernels for project 02.10 (FPFH descriptors + RANSAC
//              global registration): brute-force KNN (STAGE 1's neighbor
//              engine) + normals + SPFH + FPFH (STAGES 1-3) + descriptor
//              matching (STAGE 4) + the RANSAC hypothesis farm (STAGE 5) +
//              the point-to-plane ICP handoff kernels (STAGE 6).
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the host-side launch wrappers
// that own the grid/block math (CLAUDE.md paragraph 6.1 rule 2). Every
// constant, struct, and shared arithmetic helper is defined ONCE in
// kernels.cuh — read that file's long header comment FIRST; it is the map
// of this one, stage by stage.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR

// ===========================================================================
// Device transcriptions of kernels.cuh's shared plain-inline helpers.
// WHY DUPLICATED: those helpers are unqualified so cl.exe (reference_cpu.cpp)
// can see them too, which makes nvcc treat them as HOST-only and refuse to
// call them from a __global__ kernel (02.01/02.03/02.09's identical pattern
// — see kernels.cuh's file header for the full rationale and the VERIFY
// gates in main.cu that catch any drift between a header copy and its
// device transcription).
// ===========================================================================

__device__ __forceinline__ float d_squared_distance3(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ bool d_knn_less(float da, int32_t ia, float db, int32_t ib)
{
    if (da != db) return da < db;
    return ia < ib;
}

__device__ __forceinline__ void d_apply_rigid(const Rigid3& T, const float p[3], float out[3])
{
    out[0] = T.R[0] * p[0] + T.R[1] * p[1] + T.R[2] * p[2] + T.t[0];
    out[1] = T.R[3] * p[0] + T.R[4] * p[1] + T.R[5] * p[2] + T.t[1];
    out[2] = T.R[6] * p[0] + T.R[7] * p[1] + T.R[8] * p[2] + T.t[2];
}

// ---- STAGE 1: independent-per-02.09-lineage 3x3 Jacobi (device side) ------
// Same algorithm FAMILY as reference_cpu.cpp's jacobi_eigen_3x3_cpu (cited,
// 02.03/02.09's precedent), but NOT the same code — see kernels.cuh's
// "Twin-vs-shared ruling" for why STAGE 1 specifically gets two genuinely
// independent solves rather than one shared header call.
__device__ __forceinline__ void d_jacobi_eigen_3x3(const float cov[6], float eigenvalues[3], float eigenvectors[3][3])
{
    float A[3][3] = {
        { cov[0], cov[1], cov[2] },
        { cov[1], cov[3], cov[4] },
        { cov[2], cov[4], cov[5] },
    };
    float V[3][3] = { {1.0f,0.0f,0.0f}, {0.0f,1.0f,0.0f}, {0.0f,0.0f,1.0f} };

    #pragma unroll
    for (int sweep = 0; sweep < kJacobiSweeps3; ++sweep) {
        const int pairs[3][2] = { {0,1}, {0,2}, {1,2} };
        #pragma unroll
        for (int pi = 0; pi < 3; ++pi) {
            const int p = pairs[pi][0], q = pairs[pi][1];
            const float apq = A[p][q];
            if (fabsf(apq) < 1.0e-12f) continue;
            const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
            const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
            const float c = 1.0f / sqrtf(t * t + 1.0f);
            const float s = t * c;
            const float app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq; A[q][q] = aqq + t * apq;
            A[p][q] = 0.0f; A[q][p] = 0.0f;
            const int r = 3 - p - q;
            const float arp = A[r][p], arq = A[r][q];
            A[r][p] = A[p][r] = c * arp - s * arq;
            A[r][q] = A[q][r] = s * arp + c * arq;
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                const float vip = V[i][p], viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }
    float ev[3] = { A[0][0], A[1][1], A[2][2] };
    float vec[3][3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) { vec[i][0] = V[0][i]; vec[i][1] = V[1][i]; vec[i][2] = V[2][i]; }
    #pragma unroll
    for (int i = 1; i < 3; ++i) {
        const float ek = ev[i]; const float vk0 = vec[i][0], vk1 = vec[i][1], vk2 = vec[i][2];
        int j = i - 1;
        while (j >= 0 && ev[j] > ek) {
            ev[j + 1] = ev[j]; vec[j + 1][0] = vec[j][0]; vec[j + 1][1] = vec[j][1]; vec[j + 1][2] = vec[j][2];
            --j;
        }
        ev[j + 1] = ek; vec[j + 1][0] = vk0; vec[j + 1][1] = vk1; vec[j + 1][2] = vk2;
    }
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = ev[i];
        eigenvectors[i][0] = vec[i][0]; eigenvectors[i][1] = vec[i][1]; eigenvectors[i][2] = vec[i][2];
    }
}

// ---- STAGE 2's darboux_triplet + angle_to_bin: literal device copies of
// kernels.cuh's shared formula (bit-for-bit the same arithmetic; VERIFY
// gates in main.cu compare GPU output against the CPU twin that calls the
// header version directly — any drift between these two copies shows up
// there, exactly the pattern 02.03's czm_compute_patch_ids_kernel/_cpu use).
__device__ __forceinline__ void d_darboux_triplet(const float n_q[3], const float p_q[3],
                                                   const float n_k[3], const float p_k[3],
                                                   float& alpha, float& phi, float& theta)
{
    float d[3] = { p_k[0] - p_q[0], p_k[1] - p_q[1], p_k[2] - p_q[2] };
    const float dist = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const float inv_dist = (dist > 1.0e-9f) ? (1.0f / dist) : 0.0f;
    d[0] *= inv_dist; d[1] *= inv_dist; d[2] *= inv_dist;

    const float u[3] = { n_q[0], n_q[1], n_q[2] };
    float v[3] = { u[1] * d[2] - u[2] * d[1], u[2] * d[0] - u[0] * d[2], u[0] * d[1] - u[1] * d[0] };
    float vnorm = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (vnorm < 1.0e-6f) {
        const float helper[3] = { (fabsf(u[0]) < 0.9f) ? 1.0f : 0.0f, (fabsf(u[0]) < 0.9f) ? 0.0f : 1.0f, 0.0f };
        v[0] = u[1] * helper[2] - u[2] * helper[1];
        v[1] = u[2] * helper[0] - u[0] * helper[2];
        v[2] = u[0] * helper[1] - u[1] * helper[0];
        vnorm = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    const float inv_vnorm = 1.0f / vnorm;
    v[0] *= inv_vnorm; v[1] *= inv_vnorm; v[2] *= inv_vnorm;
    const float w[3] = { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };

    alpha = v[0] * n_k[0] + v[1] * n_k[1] + v[2] * n_k[2];
    phi   = u[0] * d[0] + u[1] * d[1] + u[2] * d[2];
    const float wn = w[0] * n_k[0] + w[1] * n_k[1] + w[2] * n_k[2];
    const float un = u[0] * n_k[0] + u[1] * n_k[1] + u[2] * n_k[2];
    theta = atan2f(wn, un);
}

__device__ __forceinline__ int d_angle_to_bin(float value, float lo, float hi)
{
    float frac = (value - lo) / (hi - lo);
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 0.99999994f) frac = 0.99999994f;
    int bin = static_cast<int>(frac * static_cast<float>(kFpfhBins));
    if (bin < 0) bin = 0;
    if (bin >= kFpfhBins) bin = kFpfhBins - 1;
    return bin;
}

// ---- STAGE 5's shared RNG + prescreen + Horn fit: literal device copies ---
__device__ __forceinline__ uint32_t d_xorshift32_step(uint32_t state)
{
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

__device__ __forceinline__ uint32_t d_hypothesis_seed(uint32_t global_seed, int k, int attempt)
{
    uint32_t s = global_seed ^ (0x9E3779B9u * static_cast<uint32_t>(k * 8 + attempt + 1));
    if (s == 0u) s = 1u;
    s = d_xorshift32_step(s); s = d_xorshift32_step(s);
    return s;
}

__device__ __forceinline__ bool d_pick_correspondence_triplet(uint32_t seed, int nc, int& i0, int& i1, int& i2)
{
    if (nc < 3) return false;
    uint32_t s = seed;
    s = d_xorshift32_step(s); i0 = static_cast<int>(s % static_cast<uint32_t>(nc));
    for (int guard = 0; guard < 8; ++guard) {
        s = d_xorshift32_step(s); i1 = static_cast<int>(s % static_cast<uint32_t>(nc));
        if (i1 != i0) break;
    }
    for (int guard = 0; guard < 8; ++guard) {
        s = d_xorshift32_step(s); i2 = static_cast<int>(s % static_cast<uint32_t>(nc));
        if (i2 != i0 && i2 != i1) break;
    }
    return (i0 != i1) && (i0 != i2) && (i1 != i2);
}

__device__ __forceinline__ bool d_edge_length_prescreen(const float s0[3], const float s1[3], const float s2[3],
                                                         const float t0[3], const float t1[3], const float t2[3])
{
    const float ds01 = sqrtf(d_squared_distance3(s0, s1));
    const float ds02 = sqrtf(d_squared_distance3(s0, s2));
    const float ds12 = sqrtf(d_squared_distance3(s1, s2));
    if (ds01 < kRansacMinEdgeLenM || ds02 < kRansacMinEdgeLenM || ds12 < kRansacMinEdgeLenM) return false;
    const float dt01 = sqrtf(d_squared_distance3(t0, t1));
    const float dt02 = sqrtf(d_squared_distance3(t0, t2));
    const float dt12 = sqrtf(d_squared_distance3(t1, t2));
    if (fabsf(ds01 - dt01) > kRansacEdgeLenTolM) return false;
    if (fabsf(ds02 - dt02) > kRansacEdgeLenTolM) return false;
    if (fabsf(ds12 - dt12) > kRansacEdgeLenTolM) return false;
    return true;
}

// d_jacobi_eigen_4x4 — literal device copy of Horn's 4x4 key-matrix
// eigensolve (kernels.cuh's jacobi_eigen_4x4, cited; see that header for the
// full derivation of WHY the largest eigenvalue's eigenvector is the
// optimal rotation quaternion — Horn 1987).
__device__ __forceinline__ void d_jacobi_eigen_4x4(const float a_in[10], float eigenvalues[4], float eigenvectors[4][4])
{
    float A[4][4] = {
        { a_in[0], a_in[1], a_in[2], a_in[3] },
        { a_in[1], a_in[4], a_in[5], a_in[6] },
        { a_in[2], a_in[5], a_in[7], a_in[8] },
        { a_in[3], a_in[6], a_in[8], a_in[9] },
    };
    float V[4][4] = { {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1} };
    const int pairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
    #pragma unroll
    for (int sweep = 0; sweep < kJacobiSweeps4; ++sweep) {
        #pragma unroll
        for (int pi = 0; pi < 6; ++pi) {
            const int p = pairs[pi][0], q = pairs[pi][1];
            const float apq = A[p][q];
            if (fabsf(apq) < 1.0e-12f) continue;
            const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
            const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
            const float c = 1.0f / sqrtf(t * t + 1.0f);
            const float s = t * c;
            const float app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq; A[q][q] = aqq + t * apq;
            A[p][q] = 0.0f; A[q][p] = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i == p || i == q) continue;
                const float aip = A[i][p], aiq = A[i][q];
                A[i][p] = c * aip - s * aiq; A[p][i] = A[i][p];
                A[i][q] = s * aip + c * aiq; A[q][i] = A[i][q];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float vip = V[i][p], viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        eigenvalues[i] = A[i][i];
        #pragma unroll
        for (int j = 0; j < 4; ++j) eigenvectors[i][j] = V[j][i];
    }
}

__device__ __forceinline__ bool d_rigid_fit_horn(int count, const float* src_xyz, const float* tgt_xyz,
                                                  float R_out[9], float t_out[3])
{
    if (count < 3) return false;
    float cs[3] = { 0, 0, 0 }, ct[3] = { 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        cs[0] += src_xyz[i * 3 + 0]; cs[1] += src_xyz[i * 3 + 1]; cs[2] += src_xyz[i * 3 + 2];
        ct[0] += tgt_xyz[i * 3 + 0]; ct[1] += tgt_xyz[i * 3 + 1]; ct[2] += tgt_xyz[i * 3 + 2];
    }
    const float inv_n = 1.0f / static_cast<float>(count);
    cs[0] *= inv_n; cs[1] *= inv_n; cs[2] *= inv_n;
    ct[0] *= inv_n; ct[1] *= inv_n; ct[2] *= inv_n;

    float M[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const float sx = src_xyz[i * 3 + 0] - cs[0], sy = src_xyz[i * 3 + 1] - cs[1], sz = src_xyz[i * 3 + 2] - cs[2];
        const float tx = tgt_xyz[i * 3 + 0] - ct[0], ty = tgt_xyz[i * 3 + 1] - ct[1], tz = tgt_xyz[i * 3 + 2] - ct[2];
        M[0] += sx * tx; M[1] += sx * ty; M[2] += sx * tz;
        M[3] += sy * tx; M[4] += sy * ty; M[5] += sy * tz;
        M[6] += sz * tx; M[7] += sz * ty; M[8] += sz * tz;
    }
    const float trace_abs = fabsf(M[0]) + fabsf(M[4]) + fabsf(M[8]);
    if (trace_abs < 1.0e-9f) return false;

    const float Sxx = M[0], Sxy = M[1], Sxz = M[2], Syx = M[3], Syy = M[4], Syz = M[5], Szx = M[6], Szy = M[7], Szz = M[8];
    const float n_packed[10] = {
        Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
        Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
        -Sxx + Syy - Szz, Syz + Szy,
        -Sxx - Syy + Szz
    };
    float eigenvalues[4]; float eigenvectors[4][4];
    d_jacobi_eigen_4x4(n_packed, eigenvalues, eigenvectors);

    int best = 0;
    #pragma unroll
    for (int i = 1; i < 4; ++i) if (eigenvalues[i] > eigenvalues[best]) best = i;
    float qw = eigenvectors[best][0], qx = eigenvectors[best][1], qy = eigenvectors[best][2], qz = eigenvectors[best][3];
    const float qn = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qn < 1.0e-12f) return false;
    const float inv_qn = 1.0f / qn;
    qw *= inv_qn; qx *= inv_qn; qy *= inv_qn; qz *= inv_qn;

    R_out[0] = 1.0f - 2.0f * (qy * qy + qz * qz);  R_out[1] = 2.0f * (qx * qy - qw * qz);         R_out[2] = 2.0f * (qx * qz + qw * qy);
    R_out[3] = 2.0f * (qx * qy + qw * qz);         R_out[4] = 1.0f - 2.0f * (qx * qx + qz * qz);  R_out[5] = 2.0f * (qy * qz - qw * qx);
    R_out[6] = 2.0f * (qx * qz - qw * qy);         R_out[7] = 2.0f * (qy * qz + qw * qx);         R_out[8] = 1.0f - 2.0f * (qx * qx + qy * qy);

    t_out[0] = ct[0] - (R_out[0] * cs[0] + R_out[1] * cs[1] + R_out[2] * cs[2]);
    t_out[1] = ct[1] - (R_out[3] * cs[0] + R_out[4] * cs[1] + R_out[5] * cs[2]);
    t_out[2] = ct[2] - (R_out[6] * cs[0] + R_out[7] * cs[1] + R_out[8] * cs[2]);
    return true;
}

// ===========================================================================
// STAGE 1 helper — brute-force KNN.
//
// Thread mapping: one thread per query point q. WHY brute force, not a
// spatial index: at this project's scale (~1.5k-3.2k points/scan), an
// O(n^2) all-pairs scan is a few million distance evaluations total —
// microseconds on any GPU here — and needs zero index-build machinery
// (no sort, no hash table, no tree). 02.06's launch_find_correspondences
// makes the identical call at a similar point count (cited); 02.09's
// voxel hash is the RIGHT tool once n reaches the millions this project
// explicitly does not target (THEORY.md "Where this sits in the real
// world" names the crossover).
// ---------------------------------------------------------------------------
__global__ void knn_search_kernel(int n, const float* __restrict__ xyz,
                                  int32_t* __restrict__ out_neighbor_ids,
                                  float* __restrict__ out_neighbor_dist)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const float qp[3] = { xyz[q * 3 + 0], xyz[q * 3 + 1], xyz[q * 3 + 2] };

    // Bounded max-heap of the kFpfhK BEST (smallest dist2) candidates seen
    // so far; heap[0] is always the current WORST kept candidate (02.05/
    // 02.09's identical "bounded top-K via a binary max-heap" pattern).
    float heap_d2[kFpfhK];
    int32_t heap_id[kFpfhK];
    int heap_size = 0;

    for (int j = 0; j < n; ++j) {
        if (j == q) continue;   // EXCLUDE self — SPFH pairs the query against its NEIGHBORS, never itself
        const float pp[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
        const float d2 = d_squared_distance3(pp, qp);

        if (heap_size < kFpfhK) {
            int c = heap_size++;
            heap_d2[c] = d2; heap_id[c] = j;
            while (c > 0) {
                const int parent = (c - 1) / 2;
                if (d_knn_less(heap_d2[parent], heap_id[parent], heap_d2[c], heap_id[c])) {
                    const float td = heap_d2[parent]; heap_d2[parent] = heap_d2[c]; heap_d2[c] = td;
                    const int32_t ti = heap_id[parent]; heap_id[parent] = heap_id[c]; heap_id[c] = ti;
                    c = parent;
                } else break;
            }
        } else if (d_knn_less(d2, j, heap_d2[0], heap_id[0])) {
            heap_d2[0] = d2; heap_id[0] = j;
            int c = 0;
            while (true) {
                const int l = 2 * c + 1, r = 2 * c + 2;
                int worst = c;
                if (l < kFpfhK && d_knn_less(heap_d2[worst], heap_id[worst], heap_d2[l], heap_id[l])) worst = l;
                if (r < kFpfhK && d_knn_less(heap_d2[worst], heap_id[worst], heap_d2[r], heap_id[r])) worst = r;
                if (worst == c) break;
                const float td = heap_d2[worst]; heap_d2[worst] = heap_d2[c]; heap_d2[c] = td;
                const int32_t ti = heap_id[worst]; heap_id[worst] = heap_id[c]; heap_id[c] = ti;
                c = worst;
            }
        }
    }

    // Insertion-sort the kFpfhK heap contents ascending by knn_less (kFpfhK
    // == 20: trivially cheap) so every consumer sees the SAME canonical
    // order — the precondition for an exact-equality GPU-vs-CPU gate.
    #pragma unroll
    for (int a = 1; a < kFpfhK; ++a) {
        const float kd = heap_d2[a]; const int32_t ki = heap_id[a];
        int b = a - 1;
        while (b >= 0 && d_knn_less(kd, ki, heap_d2[b], heap_id[b])) {
            heap_d2[b + 1] = heap_d2[b]; heap_id[b + 1] = heap_id[b];
            --b;
        }
        heap_d2[b + 1] = kd; heap_id[b + 1] = ki;
    }

    for (int a = 0; a < kFpfhK; ++a) {
        out_neighbor_ids[q * kFpfhK + a] = heap_id[a];
        out_neighbor_dist[q * kFpfhK + a] = sqrtf(heap_d2[a]);
    }
}

void launch_knn_search(int n, const float* d_xyz, int32_t* d_neighbor_ids, float* d_neighbor_dist)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    knn_search_kernel<<<grid, block>>>(n, d_xyz, d_neighbor_ids, d_neighbor_dist);
    CUDA_CHECK_LAST_ERROR("knn_search_kernel launch");
}

// ===========================================================================
// STAGE 1 — normals: one thread per point, mean-shifted covariance over its
// kFpfhK cached neighbors (02.09's "two-pass, not one-pass" precedent,
// cited: avoids the catastrophic cancellation of E[pp^T]-mean*mean^T at
// real-scale coordinates) -> d_jacobi_eigen_3x3 -> smallest eigenvector,
// oriented toward ref_point (this cloud's own centroid — see
// launch_estimate_normals; the SAME "orient toward an interior reference
// point" idea 02.06 uses for its target-cloud normals, cited, simpler here
// because this project has no dedicated sensor-origin concept per cloud —
// each cloud's own centroid is a robust stand-in for "the interior side"
// on this project's mostly-enclosing room geometry).
// ---------------------------------------------------------------------------
__global__ void estimate_normals_kernel(int n, const float* __restrict__ xyz,
                                        const int32_t* __restrict__ neighbor_ids,
                                        float ref_x, float ref_y, float ref_z,
                                        float* __restrict__ out_normal)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const float qp[3] = { xyz[q * 3 + 0], xyz[q * 3 + 1], xyz[q * 3 + 2] };

    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    for (int a = 0; a < kFpfhK; ++a) {
        const int pid = neighbor_ids[q * kFpfhK + a];
        mx += xyz[pid * 3 + 0]; my += xyz[pid * 3 + 1]; mz += xyz[pid * 3 + 2];
    }
    const float inv_k = 1.0f / static_cast<float>(kFpfhK);
    mx *= inv_k; my *= inv_k; mz *= inv_k;

    float cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
    for (int a = 0; a < kFpfhK; ++a) {
        const int pid = neighbor_ids[q * kFpfhK + a];
        const float dx = xyz[pid * 3 + 0] - mx, dy = xyz[pid * 3 + 1] - my, dz = xyz[pid * 3 + 2] - mz;
        cxx += dx * dx; cxy += dx * dy; cxz += dx * dz; cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
    }
    const float cov[6] = { cxx * inv_k, cxy * inv_k, cxz * inv_k, cyy * inv_k, cyz * inv_k, czz * inv_k };

    float eigenvalues[3]; float eigenvectors[3][3];
    d_jacobi_eigen_3x3(cov, eigenvalues, eigenvectors);

    float nrm[3] = { eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2] };
    const float view[3] = { ref_x - qp[0], ref_y - qp[1], ref_z - qp[2] };
    const float dotv = nrm[0] * view[0] + nrm[1] * view[1] + nrm[2] * view[2];
    if (dotv < 0.0f) { nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2]; }

    out_normal[q * 3 + 0] = nrm[0]; out_normal[q * 3 + 1] = nrm[1]; out_normal[q * 3 + 2] = nrm[2];
}

void launch_estimate_normals(int n, const float* d_xyz, const int32_t* d_neighbor_ids,
                             float ref_x, float ref_y, float ref_z, float* d_out_normal)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    estimate_normals_kernel<<<grid, block>>>(n, d_xyz, d_neighbor_ids, ref_x, ref_y, ref_z, d_out_normal);
    CUDA_CHECK_LAST_ERROR("estimate_normals_kernel launch");
}

// ===========================================================================
// STAGE 2 — SPFH. One thread per point q: darboux_triplet(q, neighbor) for
// each of q's kFpfhK neighbors, histogrammed into 3 SEPARATE 11-bin blocks
// (alpha/phi/theta), each block normalized to sum 1.
//
// NO ATOMICS ANYWHERE IN THIS KERNEL — the deliberate GPU-mapping choice
// THEORY.md "The GPU mapping" argues for explicitly: this is a PER-THREAD-
// PRIVATE histogram (each of the 33 bins for point q is touched ONLY by
// thread q, accumulated in registers/local memory, written ONCE at the
// end) rather than many threads racing to increment shared bins. The
// alternative that WOULD need atomics — e.g. one thread per (point,
// neighbor) PAIR scattering votes into a shared per-point histogram array
// — buys nothing here (kFpfhK=20 is too small to be worth splitting a
// single point's work across multiple threads) and would cost real
// contention for no benefit; "one thread, one point, one private
// histogram" is strictly better at this problem's shape. Contrast this
// with STAGE 6's icp_accumulate_kernel below, where MANY points genuinely
// contribute to the SAME shared accumulator and atomics are the right tool.
// ---------------------------------------------------------------------------
__global__ void compute_spfh_kernel(int n, const float* __restrict__ xyz, const float* __restrict__ normal,
                                    const int32_t* __restrict__ neighbor_ids,
                                    float* __restrict__ out_spfh)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const float p_q[3] = { xyz[q * 3 + 0], xyz[q * 3 + 1], xyz[q * 3 + 2] };
    const float n_q[3] = { normal[q * 3 + 0], normal[q * 3 + 1], normal[q * 3 + 2] };

    // Local (register/stack) histogram — private to this thread until the
    // final write, per the header comment above.
    float hist[kFpfhDim];
    #pragma unroll
    for (int i = 0; i < kFpfhDim; ++i) hist[i] = 0.0f;

    for (int a = 0; a < kFpfhK; ++a) {
        const int kid = neighbor_ids[q * kFpfhK + a];
        const float p_k[3] = { xyz[kid * 3 + 0], xyz[kid * 3 + 1], xyz[kid * 3 + 2] };
        const float n_k[3] = { normal[kid * 3 + 0], normal[kid * 3 + 1], normal[kid * 3 + 2] };
        float alpha, phi, theta;
        d_darboux_triplet(n_q, p_q, n_k, p_k, alpha, phi, theta);
        hist[0 * kFpfhBins + d_angle_to_bin(alpha, -1.0f, 1.0f)]           += 1.0f;
        hist[1 * kFpfhBins + d_angle_to_bin(phi,   -1.0f, 1.0f)]           += 1.0f;
        hist[2 * kFpfhBins + d_angle_to_bin(theta, -kPiF, kPiF)]           += 1.0f;
    }

    // Each 11-bin BLOCK independently normalized to sum 1 (Rusu et al.'s
    // SPFH convention — kernels.cuh's file header): exactly kFpfhK votes
    // fell into each of the 3 blocks, so dividing by kFpfhK normalizes all
    // three blocks in one pass.
    const float inv_k = 1.0f / static_cast<float>(kFpfhK);
    #pragma unroll
    for (int i = 0; i < kFpfhDim; ++i) out_spfh[q * kFpfhDim + i] = hist[i] * inv_k;
}

void launch_compute_spfh(int n, const float* d_xyz, const float* d_normal, const int32_t* d_neighbor_ids,
                         float* d_out_spfh)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    compute_spfh_kernel<<<grid, block>>>(n, d_xyz, d_normal, d_neighbor_ids, d_out_spfh);
    CUDA_CHECK_LAST_ERROR("compute_spfh_kernel launch");
}

// ===========================================================================
// STAGE 3 — FPFH: the weighted neighbor-SPFH re-accumulation ("the F in
// FPFH"). One thread per point q, reading the GLOBAL spfh[] array (already
// fully populated for EVERY point by STAGE 2 before this kernel launches —
// a hard ordering dependency main.cu's launch sequence enforces) at both
// q's own row AND each of q's kFpfhK neighbors' rows.
//
// Complexity story (THEORY.md "The algorithm" derives this fully): true
// PFH computes one histogram per point over ALL k*(k-1)/2 pairs WITHIN its
// neighborhood — O(n*k^2) total. FPFH instead computes ONE SPFH per point
// (O(n*k), STAGE 2) and here re-uses each neighbor's ALREADY-COMPUTED SPFH
// — which itself already summarized THAT neighbor's own k-neighborhood —
// as a cheap O(k) stand-in for the missing pairwise terms. This is the
// "two-ring" reading: ring 1 is q's own kFpfhK neighbors (read directly,
// this loop); ring 2 (each neighbor's OWN neighborhood) arrives for free,
// already baked into their SPFH rows, with NO second explicit traversal.
// Total cost: O(n*k) SPFH + O(n*k) here = O(n*k), not O(n*k^2) — the
// entire reason FPFH exists.
//
// Again NO ATOMICS: this thread reads MANY rows (read-only, no data race
// possible — every row was finalized by a PRIOR kernel launch) but writes
// only its own output row q. Same "per-thread-private output" mapping as
// STAGE 2, for the same reason.
// ---------------------------------------------------------------------------
__global__ void compute_fpfh_kernel(int n, const float* __restrict__ spfh,
                                    const int32_t* __restrict__ neighbor_ids,
                                    const float* __restrict__ neighbor_dist,
                                    float* __restrict__ out_fpfh)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    float acc[kFpfhDim];
    #pragma unroll
    for (int i = 0; i < kFpfhDim; ++i) acc[i] = spfh[q * kFpfhDim + i];   // start from this point's OWN SPFH

    for (int a = 0; a < kFpfhK; ++a) {
        const int kid = neighbor_ids[q * kFpfhK + a];
        const float dist = neighbor_dist[q * kFpfhK + a];
        const float w = (dist > 1.0e-6f) ? (1.0f / dist) : 0.0f;   // Rusu et al.'s 1/distance weighting: closer neighbors count more
        #pragma unroll
        for (int i = 0; i < kFpfhDim; ++i) acc[i] += w * spfh[kid * kFpfhDim + i];
    }
    const float inv_k = 1.0f / static_cast<float>(kFpfhK);
    #pragma unroll
    for (int i = 0; i < kFpfhDim; ++i) acc[i] *= inv_k;

    // L1-normalize the FULL 33-dim descriptor (ratified scope's explicit
    // instruction): every entry is non-negative (built from histogram
    // counts and positive weights), so the L1 norm is just the sum.
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kFpfhDim; ++i) sum += acc[i];
    const float inv_sum = (sum > 1.0e-9f) ? (1.0f / sum) : 0.0f;
    #pragma unroll
    for (int i = 0; i < kFpfhDim; ++i) out_fpfh[q * kFpfhDim + i] = acc[i] * inv_sum;
}

void launch_compute_fpfh(int n, const float* d_spfh, const int32_t* d_neighbor_ids, const float* d_neighbor_dist,
                         float* d_out_fpfh)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    compute_fpfh_kernel<<<grid, block>>>(n, d_spfh, d_neighbor_ids, d_neighbor_dist, d_out_fpfh);
    CUDA_CHECK_LAST_ERROR("compute_fpfh_kernel launch");
}

// ===========================================================================
// STAGE 4 — descriptor matching + ratio test. One thread per SOURCE point:
// scan all n_tgt target descriptors, track the best and second-best squared
// L2 distance (33-D) via a simple 2-slot running minimum (no heap needed —
// only the TOP TWO ever matter for a ratio test, 01.04's "don't build
// machinery a threshold test doesn't need" lesson again).
// ---------------------------------------------------------------------------
__global__ void match_correspondences_kernel(int n_src, const float* __restrict__ fpfh_src,
                                             int n_tgt, const float* __restrict__ fpfh_tgt,
                                             uint8_t* __restrict__ out_matched,
                                             int32_t* __restrict__ out_best_idx,
                                             float* __restrict__ out_dist1_sq,
                                             float* __restrict__ out_dist2_sq)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_src) return;

    float best1 = 3.0e38f, best2 = 3.0e38f;   // running (smallest, 2nd-smallest) squared L2 distance
    int best1_idx = -1;

    for (int t = 0; t < n_tgt; ++t) {
        float d2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < kFpfhDim; ++i) {
            const float diff = fpfh_src[s * kFpfhDim + i] - fpfh_tgt[t * kFpfhDim + i];
            d2 += diff * diff;
        }
        if (d2 < best1) { best2 = best1; best1 = d2; best1_idx = t; }
        else if (d2 < best2) { best2 = d2; }
    }

    const bool accept = (best1_idx >= 0) && (best1 <= kMatchRatioMax * kMatchRatioMax * best2);
    out_matched[s]  = accept ? 1u : 0u;
    out_best_idx[s] = best1_idx;
    out_dist1_sq[s] = best1;
    out_dist2_sq[s] = best2;
}

void launch_match_correspondences(int n_src, const float* d_fpfh_src, int n_tgt, const float* d_fpfh_tgt,
                                  uint8_t* d_out_matched, int32_t* d_out_best_idx,
                                  float* d_out_dist1_sq, float* d_out_dist2_sq)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n_src, block);
    match_correspondences_kernel<<<grid, block>>>(n_src, d_fpfh_src, n_tgt, d_fpfh_tgt,
                                                   d_out_matched, d_out_best_idx, d_out_dist1_sq, d_out_dist2_sq);
    CUDA_CHECK_LAST_ERROR("match_correspondences_kernel launch");
}

// ===========================================================================
// STAGE 5 — the RANSAC hypothesis farm (02.03's "farm" pattern, cited: one
// thread per hypothesis, K independent draws, no shared RNG state — see
// kernels.cuh's hypothesis_seed comment for why this is embarrassingly
// parallel BY CONSTRUCTION). Grid-stride over kRansacK is not needed at
// this K (a few thousand): one thread per hypothesis, one launch, covers
// it directly.
//
// Each thread: sample+prescreen up to kRansacMaxTripletAttempts triplets
// (retrying on a degenerate draw, mirroring 02.03's ransac_generate_
// hypotheses_kernel), fit via d_rigid_fit_horn, then SCORE by scanning the
// WHOLE gathered correspondence array (nc entries — a few hundred at this
// project's scale) counting inliers under kRansacInlierThresholdM. Total
// work per thread is O(nc); total kernel work is O(K*nc), fully parallel
// across the K threads.
// ---------------------------------------------------------------------------
__global__ void ransac_hypotheses_kernel(int nc, const float* __restrict__ corr_src_xyz,
                                         const float* __restrict__ corr_tgt_xyz,
                                         uint32_t global_seed, int k,
                                         uint8_t* __restrict__ out_valid,
                                         Rigid3* __restrict__ out_transform,
                                         int32_t* __restrict__ out_inlier_count)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= k) return;

    bool got_valid_triplet = false;
    int i0 = -1, i1 = -1, i2 = -1;
    for (int attempt = 0; attempt < kRansacMaxTripletAttempts; ++attempt) {
        const uint32_t seed = d_hypothesis_seed(global_seed, h, attempt);
        if (!d_pick_correspondence_triplet(seed, nc, i0, i1, i2)) continue;
        const float* s0 = &corr_src_xyz[i0 * 3]; const float* s1 = &corr_src_xyz[i1 * 3]; const float* s2 = &corr_src_xyz[i2 * 3];
        const float* t0 = &corr_tgt_xyz[i0 * 3]; const float* t1 = &corr_tgt_xyz[i1 * 3]; const float* t2 = &corr_tgt_xyz[i2 * 3];
        if (d_edge_length_prescreen(s0, s1, s2, t0, t1, t2)) { got_valid_triplet = true; break; }
    }

    if (!got_valid_triplet) {
        out_valid[h] = 0u;
        out_inlier_count[h] = 0;
        return;
    }

    float src3[9] = { corr_src_xyz[i0*3+0], corr_src_xyz[i0*3+1], corr_src_xyz[i0*3+2],
                      corr_src_xyz[i1*3+0], corr_src_xyz[i1*3+1], corr_src_xyz[i1*3+2],
                      corr_src_xyz[i2*3+0], corr_src_xyz[i2*3+1], corr_src_xyz[i2*3+2] };
    float tgt3[9] = { corr_tgt_xyz[i0*3+0], corr_tgt_xyz[i0*3+1], corr_tgt_xyz[i0*3+2],
                      corr_tgt_xyz[i1*3+0], corr_tgt_xyz[i1*3+1], corr_tgt_xyz[i1*3+2],
                      corr_tgt_xyz[i2*3+0], corr_tgt_xyz[i2*3+1], corr_tgt_xyz[i2*3+2] };
    Rigid3 T;
    if (!d_rigid_fit_horn(3, src3, tgt3, T.R, T.t)) {
        out_valid[h] = 0u;
        out_inlier_count[h] = 0;
        return;
    }

    int inliers = 0;
    const float thresh2 = kRansacInlierThresholdM * kRansacInlierThresholdM;
    for (int c = 0; c < nc; ++c) {
        const float sp[3] = { corr_src_xyz[c * 3 + 0], corr_src_xyz[c * 3 + 1], corr_src_xyz[c * 3 + 2] };
        const float tp[3] = { corr_tgt_xyz[c * 3 + 0], corr_tgt_xyz[c * 3 + 1], corr_tgt_xyz[c * 3 + 2] };
        float xp[3]; d_apply_rigid(T, sp, xp);
        if (d_squared_distance3(xp, tp) <= thresh2) ++inliers;
    }

    out_valid[h] = 1u;
    out_transform[h] = T;
    out_inlier_count[h] = inliers;
}

void launch_ransac_hypotheses(int nc, const float* d_corr_src_xyz, const float* d_corr_tgt_xyz,
                              uint32_t global_seed, int k,
                              uint8_t* d_out_valid, Rigid3* d_out_transform, int32_t* d_out_inlier_count)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(k, block);
    ransac_hypotheses_kernel<<<grid, block>>>(nc, d_corr_src_xyz, d_corr_tgt_xyz, global_seed, k,
                                              d_out_valid, d_out_transform, d_out_inlier_count);
    CUDA_CHECK_LAST_ERROR("ransac_hypotheses_kernel launch");
}

// ===========================================================================
// STAGE 6 — the point-to-plane ICP handoff (02.06 lineage, cited, compact
// reimplementation — see kernels.cuh's file header for the scope contrast).
// ===========================================================================

__global__ void transform_cloud_kernel(int n, const float* __restrict__ src_xyz, Rigid3 T,
                                       float* __restrict__ out_xyz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float p[3] = { src_xyz[i * 3 + 0], src_xyz[i * 3 + 1], src_xyz[i * 3 + 2] };
    float o[3]; d_apply_rigid(T, p, o);
    out_xyz[i * 3 + 0] = o[0]; out_xyz[i * 3 + 1] = o[1]; out_xyz[i * 3 + 2] = o[2];
}

void launch_transform_cloud(int n, const float* d_src_xyz, Rigid3 T, float* d_out_xyz)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    transform_cloud_kernel<<<grid, block>>>(n, d_src_xyz, T, d_out_xyz);
    CUDA_CHECK_LAST_ERROR("transform_cloud_kernel launch");
}

// icp_correspondences_kernel — brute-force nearest TARGET point for every
// (already transformed) SOURCE point, gated by max_dist_m (02.06's
// launch_find_correspondences, cited: identical algorithm, this project's
// own point counts). corr_idx[s] = -1 means "no target point within
// max_dist_m" (rejected, THEORY.md "The algorithm").
__global__ void icp_correspondences_kernel(int n_src, const float* __restrict__ cur_xyz,
                                           int n_tgt, const float* __restrict__ tgt_xyz,
                                           float max_dist_m,
                                           int32_t* __restrict__ out_corr_idx,
                                           float* __restrict__ out_corr_dist2)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_src) return;
    const float p[3] = { cur_xyz[s * 3 + 0], cur_xyz[s * 3 + 1], cur_xyz[s * 3 + 2] };

    float best_d2 = 3.0e38f;
    int best_t = -1;
    for (int t = 0; t < n_tgt; ++t) {
        const float q[3] = { tgt_xyz[t * 3 + 0], tgt_xyz[t * 3 + 1], tgt_xyz[t * 3 + 2] };
        const float d2 = d_squared_distance3(p, q);
        if (d2 < best_d2) { best_d2 = d2; best_t = t; }
    }
    const float gate2 = max_dist_m * max_dist_m;
    out_corr_idx[s] = (best_d2 <= gate2) ? best_t : -1;
    out_corr_dist2[s] = best_d2;
}

void launch_icp_correspondences(int n_src, const float* d_cur_xyz, int n_tgt, const float* d_tgt_xyz,
                                float max_dist_m, int32_t* d_out_corr_idx, float* d_out_corr_dist2)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n_src, block);
    icp_correspondences_kernel<<<grid, block>>>(n_src, d_cur_xyz, n_tgt, d_tgt_xyz, max_dist_m,
                                                d_out_corr_idx, d_out_corr_dist2);
    CUDA_CHECK_LAST_ERROR("icp_correspondences_kernel launch");
}

// icp_accumulate_kernel — one thread per SOURCE point: build its point-to-
// plane Jacobian row J = [x_cur x n_tgt ; n_tgt] (THEORY.md "The math"
// derives this from the small-angle SE(3) perturbation x_new ~= x_cur +
// w x x_cur + v, giving de/dw = x_cur x n and de/dv = n for residual
// e = n.(x_cur - q_tgt)), and atomicAdd its contribution (J^T*J's upper
// triangle, 21 entries via hidx, plus J^T*r's 6 entries) into the SHARED
// 27-double accumulator every thread in the launch writes into.
//
// WHY ATOMICS HERE (the direct GPU-mapping contrast with STAGES 2/3's
// per-thread-private histograms, as promised in kernels.cuh's file
// header): unlike a per-point FPFH row, this accumulator has EXACTLY ONE
// logical destination shared by every one of the n_src threads — there is
// no way to give each thread a private copy of "the answer" because the
// 6x6 system IS the sum over all points. THEORY.md "The GPU mapping"
// weighs this against 02.06's shared-memory BLOCK-TREE reduction (partial-
// sum per block, finished on the host) and documents the choice: at this
// project's n_src (a few thousand, not 02.06's much larger runs), atomic
// contention on 27 double-precision locations is cheap relative to the
// code size and conceptual overhead a full block reduction would add for
// what is here a small POLISH step, not the primary teaching payload.
// ---------------------------------------------------------------------------
__global__ void icp_accumulate_kernel(int n_src, const float* __restrict__ cur_xyz,
                                      const float* __restrict__ tgt_xyz, const float* __restrict__ tgt_normal,
                                      const int32_t* __restrict__ corr_idx,
                                      double* __restrict__ accum27)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_src) return;
    const int t = corr_idx[s];
    if (t < 0) return;   // rejected correspondence: contributes nothing

    const float x[3] = { cur_xyz[s * 3 + 0], cur_xyz[s * 3 + 1], cur_xyz[s * 3 + 2] };
    const float q[3] = { tgt_xyz[t * 3 + 0], tgt_xyz[t * 3 + 1], tgt_xyz[t * 3 + 2] };
    const float nrm[3] = { tgt_normal[t * 3 + 0], tgt_normal[t * 3 + 1], tgt_normal[t * 3 + 2] };

    const float diff[3] = { x[0] - q[0], x[1] - q[1], x[2] - q[2] };
    const float residual = nrm[0] * diff[0] + nrm[1] * diff[1] + nrm[2] * diff[2];   // scalar point-to-plane error

    // J = [x cross n ; n] — 6 entries, [wx,wy,wz,vx,vy,vz] order (hidx's
    // documented convention, kernels.cuh, cited from 02.06).
    const float J[6] = {
        x[1] * nrm[2] - x[2] * nrm[1],
        x[2] * nrm[0] - x[0] * nrm[2],
        x[0] * nrm[1] - x[1] * nrm[0],
        nrm[0], nrm[1], nrm[2]
    };

    // Upper-triangle H += J^T*J (21 entries, via the SAME row_start table
    // hidx() encodes — literal here, per this header's "no unqualified-
    // function calls from device code" rule) and g += J^T*residual (6
    // entries), all as double-precision atomicAdd (sm_60+ native support;
    // this repo's sm_75 floor clears that bar).
    const int row_start[6] = { 0, 6, 11, 15, 18, 20 };
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        #pragma unroll
        for (int j = i; j < 6; ++j) {
            const double contrib = static_cast<double>(J[i]) * static_cast<double>(J[j]);
            atomicAdd(&accum27[row_start[i] + (j - i)], contrib);
        }
        atomicAdd(&accum27[21 + i], static_cast<double>(J[i]) * static_cast<double>(residual));
    }
}

void launch_icp_accumulate(int n_src, const float* d_cur_xyz, const float* d_tgt_xyz, const float* d_tgt_normal,
                           const int32_t* d_corr_idx, double* d_accum27)
{
    CUDA_CHECK(cudaMemset(d_accum27, 0, 27 * sizeof(double)));   // caller relaunches per iteration; always start clean
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n_src, block);
    icp_accumulate_kernel<<<grid, block>>>(n_src, d_cur_xyz, d_tgt_xyz, d_tgt_normal, d_corr_idx, d_accum27);
    CUDA_CHECK_LAST_ERROR("icp_accumulate_kernel launch");
}
