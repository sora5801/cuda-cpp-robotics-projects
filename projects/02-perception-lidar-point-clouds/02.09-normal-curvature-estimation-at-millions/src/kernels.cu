// ===========================================================================
// kernels.cu — GPU kernels for project 02.09 (Normal + curvature estimation
//              at millions of points/sec): voxel-hash index build (02.01/
//              02.05 lineage) + the fused per-point KNN/covariance/eigen/
//              normal/curvature/degeneracy pipeline (this project's own).
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the host-side launch wrappers
// that own the grid/block math (CLAUDE.md §6.1 rule 2). Every constant,
// struct, and shared arithmetic helper is defined ONCE in kernels.cuh —
// read that file's long header comment FIRST; it is the map of this one.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR

// Thrust: header-only pieces of the CUDA Toolkit (CLAUDE.md §5). Used for
// exactly the voxel-hash boundary compaction (02.01 Method-B lineage),
// explained at each call site (CLAUDE.md §6.1 rule 6).
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

// is_nonzero — the copy_if predicate for voxel-hash boundary compaction
// (02.01/02.05's identical fix for CUDA 13.3's removed thrust::identity).
struct is_nonzero {
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// ===========================================================================
// Device transcriptions of kernels.cuh's shared plain-inline helpers.
// WHY DUPLICATED: those helpers are unqualified so cl.exe (reference_cpu.cpp)
// can see them too, which makes nvcc treat them as HOST-only and refuse to
// call them from a __global__ kernel (02.01/02.05's identical pattern).
// ===========================================================================

__device__ __forceinline__ int32_t d_voxel_coord(float p, float cell)
{
    return static_cast<int32_t>(floorf(p / cell));
}

__device__ __forceinline__ unsigned long long d_pack_voxel_key(int32_t vx, int32_t vy, int32_t vz)
{
    const uint64_t ux = static_cast<uint64_t>(vx + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uy = static_cast<uint64_t>(vy + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uz = static_cast<uint64_t>(vz + kHashCoordBias) & kHashCoordMask21;
    return ux | (uy << 21) | (uz << 42);
}

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

// d_lower_bound — smallest index in key[0,count) whose value is >= target
// (standard binary search; 02.04/02.05's identical "is this neighbor voxel
// occupied?" query, reimplemented here for this project's own index).
__device__ __forceinline__ int d_lower_bound(const unsigned long long* __restrict__ key,
                                             int count, unsigned long long target)
{
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (key[mid] < target) lo = mid + 1;
        else                   hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// d_jacobi_rotate — apply ONE Jacobi rotation that zeroes A[p][q] (and its
// mirror A[q][p]), updating A in place and accumulating the rotation into V
// (V starts as identity; after every rotation in every sweep, V's COLUMNS
// converge to the eigenvectors of the original matrix — Golub & Van Loan,
// "Matrix Computations" §8.4, cited in THEORY.md). This is THIS project's
// OWN independent transcription of cyclic Jacobi (02.03's jacobi_eigen_3x3
// is the cited precedent for the ALGORITHM CHOICE, not a shared function —
// see reference_cpu.cpp's file header for why VERIFY(eigen) needs two
// separately-typed implementations, not one shared one, to be a real check).
//
// Parameters: A[3][3] — the working symmetric matrix, updated in place.
//             V[3][3] — the accumulated rotation product, updated in place.
//             p, q    — the off-diagonal pair to zero (p < q).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void d_jacobi_rotate(float A[3][3], float V[3][3], int p, int q)
{
    const float apq = A[p][q];
    if (fabsf(apq) < 1.0e-12f) return;   // already ~zero: nothing to rotate

    // Classic stable tan(theta) construction (avoids the ill-conditioned
    // direct cot(2theta) inversion near theta=0 — THEORY.md "Numerical
    // considerations" walks through why): theta measures how "unbalanced"
    // this off-diagonal pair is relative to the diagonal gap.
    const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
    const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    const float c = 1.0f / sqrtf(t * t + 1.0f);   // cos(rotation)
    const float s = t * c;                        // sin(rotation)

    const float app = A[p][p], aqq = A[q][q];
    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0.0f;
    A[q][p] = 0.0f;

    // The THIRD row/column (the index that is neither p nor q) mixes too —
    // a plane rotation touches every entry that shares a row/column with p
    // or q, not just the (p,q) entry itself.
    const int r = 3 - p - q;   // the remaining index: {0,1,2} minus {p,q}
    const float arp = A[r][p], arq = A[r][q];
    A[r][p] = A[p][r] = c * arp - s * arq;
    A[r][q] = A[q][r] = s * arp + c * arq;

    // Accumulate the rotation into V (all three rows mix, columns p,q only).
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float vip = V[i][p], viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
    }
}

// d_jacobi_eigen_3x3 — kJacobiSweeps full sweeps over the 3 off-diagonal
// pairs, then read the (now near-diagonal) A's diagonal as eigenvalues and
// V's columns as eigenvectors, insertion-sorted ascending (n=3: trivial).
__device__ __forceinline__ void d_jacobi_eigen_3x3(const float cov[6], float eigenvalues[3], float eigenvectors[3][3])
{
    float A[3][3] = {
        { cov[0], cov[1], cov[2] },
        { cov[1], cov[3], cov[4] },
        { cov[2], cov[4], cov[5] },
    };
    float V[3][3] = { {1.0f,0.0f,0.0f}, {0.0f,1.0f,0.0f}, {0.0f,0.0f,1.0f} };

    #pragma unroll
    for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
        d_jacobi_rotate(A, V, 0, 1);
        d_jacobi_rotate(A, V, 0, 2);
        d_jacobi_rotate(A, V, 1, 2);
    }

    // Insertion-sort the 3 (eigenvalue, eigenvector-column) pairs ascending.
    float ev[3] = { A[0][0], A[1][1], A[2][2] };
    float vec[3][3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) { vec[i][0] = V[0][i]; vec[i][1] = V[1][i]; vec[i][2] = V[2][i]; }

    #pragma unroll
    for (int i = 1; i < 3; ++i) {
        const float ek = ev[i]; const float vk0 = vec[i][0], vk1 = vec[i][1], vk2 = vec[i][2];
        int j = i - 1;
        while (j >= 0 && ev[j] > ek) {
            ev[j + 1] = ev[j];
            vec[j + 1][0] = vec[j][0]; vec[j + 1][1] = vec[j][1]; vec[j + 1][2] = vec[j][2];
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

// ===========================================================================
// Voxel-hash index build — 02.01 Method-B / 02.05's THE-DOMAIN-CONTRAST
// pipeline, reimplemented compactly (this project's neighbor engine, see
// kernels.cuh's file header for the design decision).
// ===========================================================================

__global__ void compute_hash_keys_kernel(int n, const float* __restrict__ xyz,
                                         float cell, unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t vx = d_voxel_coord(xyz[i * 3 + 0], cell);
    const int32_t vy = d_voxel_coord(xyz[i * 3 + 1], cell);
    const int32_t vz = d_voxel_coord(xyz[i * 3 + 2], cell);
    keys[i] = d_pack_voxel_key(vx, vy, vz);
}

void launch_compute_hash_keys(int n, const float* d_xyz, float cell, unsigned long long* d_keys)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    compute_hash_keys_kernel<<<grid, block>>>(n, d_xyz, cell, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_hash_keys_kernel launch");
}

__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    is_start[i] = (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) ? 1 : 0;
}

__global__ void gather_unique_keys_kernel(int num_voxels, const int* __restrict__ seg_start,
                                          const unsigned long long* __restrict__ keys_sorted,
                                          unsigned long long* __restrict__ unique_key_out)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;
    unique_key_out[v] = keys_sorted[seg_start[v]];
}

int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_scratch,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out)
{
    CUDA_CHECK(cudaMemcpy(d_keys_scratch, d_keys_in, static_cast<size_t>(n) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));

    thrust::device_ptr<unsigned long long> keys_ptr(d_keys_scratch);
    thrust::device_ptr<int> idx_ptr(d_idx_scratch);
    thrust::sequence(idx_ptr, idx_ptr + n);   // idx[i]=i, the identity permutation before sorting

    // thrust::stable_sort_by_key: radix-sorts the 64-bit voxel keys
    // ascending, carrying idx along (02.01's kernels.cu explains what a
    // radix sort computes). STABLE matters here: many points legitimately
    // share one voxel key, and a stable sort keeps their relative order
    // deterministic run to run — the same reason 02.05's hash-baseline sort
    // uses it (unlike THAT project's Stage-2 Morton sort, whose keys are
    // pairwise unique and so needs no stability guarantee at all).
    thrust::stable_sort_by_key(keys_ptr, keys_ptr + n, idx_ptr);

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(n, block);
        mark_boundaries_kernel<<<grid, block>>>(n, d_keys_scratch, d_is_start_scratch);
        CUDA_CHECK_LAST_ERROR("mark_boundaries_kernel launch");
    }

    thrust::device_ptr<int> is_start_ptr(d_is_start_scratch);
    const int num_voxels = thrust::reduce(is_start_ptr, is_start_ptr + n, 0);

    thrust::device_ptr<int> seg_start_ptr(d_seg_start_out);
    thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n),
                    is_start_ptr, seg_start_ptr, is_nonzero());

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(num_voxels, block);
        gather_unique_keys_kernel<<<grid, block>>>(num_voxels, d_seg_start_out, d_keys_scratch, d_unique_key_out);
        CUDA_CHECK_LAST_ERROR("gather_unique_keys_kernel launch");
    }

    return num_voxels;
}

// ===========================================================================
// THE fused per-point pipeline kernel — kernels.cuh's file header STEPS 2-7.
//
// Thread-to-data mapping: thread (blockIdx.x, threadIdx.x) owns exactly one
// QUERY point q = blockIdx.x*blockDim.x + threadIdx.x, and does EVERY stage
// for that point start to finish: neighbor search, covariance, eigensolve,
// normal, curvature, degeneracy. Points are fully independent of each other
// (no barriers, no shared state across threads) — the natural GPU mapping
// once the voxel-hash index is built, and the reason this whole pipeline can
// be ONE kernel instead of five: nothing here needs an intermediate global-
// memory round trip between stages, so fusing them SAVES bandwidth (writing
// and re-reading a per-point neighbor-id array for 1M+ points would cost
// 1M*16*4 bytes = 64 MiB each way for NO benefit, since nothing else ever
// reads it) — the throughput story the catalog promises.
//
// Register-pressure story (measured honestly in THEORY.md "The GPU
// mapping"): this kernel's LIVE per-thread state at its peak includes a
// size-kK max-heap (kK floats + kK int32 = 128 bytes at kK=16), the cached
// neighbor xyz used for the two covariance passes (kK*3 floats = 192
// bytes), the 3x3 Jacobi working matrices A and V (9+9 floats = 72 bytes),
// plus the usual loop/index scratch — comfortably over 100 live 32-bit
// values on an architecture whose register file gives each thread roughly
// 32-64 registers at good occupancy. THEORY.md reports the ACTUAL spill-to-
// local-memory behavior `nvcc --ptxas-options=-v` shows for this kernel and
// what it costs in measured occupancy, rather than asserting a number here.
// ---------------------------------------------------------------------------
__global__ void estimate_normals_kernel(int n, const float* __restrict__ xyz,
                                        const unsigned long long* __restrict__ unique_key, int num_voxels,
                                        const int* __restrict__ seg_start,
                                        const int* __restrict__ idx_sorted, int n_sorted,
                                        float cell,
                                        float sensor_x, float sensor_y, float sensor_z,
                                        float* __restrict__ out_normal,
                                        float* __restrict__ out_eigenvalues,
                                        float* __restrict__ out_curvature,
                                        int32_t* __restrict__ out_degeneracy,
                                        int32_t* __restrict__ out_found,
                                        int32_t* __restrict__ out_neighbor_ids)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const float qp[3] = { xyz[q * 3 + 0], xyz[q * 3 + 1], xyz[q * 3 + 2] };

    // ---- STEP 2: bounded voxel-hash KNN, ring 1 then (if short) ring 2 ----
    // Per-thread bounded max-heap: heap_d2[0]/heap_id[0] is always the
    // WORST (largest dist2, or index-tie-break) of the current best kK
    // candidates — the standard "bounded top-K via a binary max-heap"
    // pattern (02.05's knn_search_bvh_kernel uses the identical shape for
    // its own KNN; reimplemented independently here for this project's
    // grid-stencil source of candidates instead of a tree traversal).
    float heap_d2[kK];
    int32_t heap_id[kK];
    int heap_size = 0;

    const int32_t cvx = d_voxel_coord(qp[0], cell);
    const int32_t cvy = d_voxel_coord(qp[1], cell);
    const int32_t cvz = d_voxel_coord(qp[2], cell);

    #pragma unroll
    for (int ring = 1; ring <= kMaxRing; ++ring) {
        for (int dz = -ring; dz <= ring; ++dz) {
            for (int dy = -ring; dy <= ring; ++dy) {
                for (int dx = -ring; dx <= ring; ++dx) {
                    // Ring 1 scans the FULL (-1..1) cube; ring 2 scans ONLY
                    // the new outer SHELL (max(|d|)==2) so no cell — and
                    // therefore no point — is ever visited twice (kernels.cuh
                    // file header STEP 2: double-counting a point would
                    // silently corrupt both the heap and the covariance).
                    const int cheb = max(abs(dx), max(abs(dy), abs(dz)));
                    if (ring > 1 && cheb != ring) continue;

                    const unsigned long long key = d_pack_voxel_key(cvx + dx, cvy + dy, cvz + dz);
                    const int v = d_lower_bound(unique_key, num_voxels, key);
                    if (v >= num_voxels || unique_key[v] != key) continue;   // neighbor cell unoccupied

                    const int begin = seg_start[v];
                    const int end = (v + 1 < num_voxels) ? seg_start[v + 1] : n_sorted;
                    for (int s = begin; s < end; ++s) {
                        const int pid = idx_sorted[s];
                        const float pp[3] = { xyz[pid * 3 + 0], xyz[pid * 3 + 1], xyz[pid * 3 + 2] };
                        const float d2 = d_squared_distance3(pp, qp);

                        if (heap_size < kK) {
                            // Plain insert: append, sift UP to restore the
                            // max-heap property (root = worst candidate).
                            int c = heap_size++;
                            heap_d2[c] = d2; heap_id[c] = pid;
                            while (c > 0) {
                                const int parent = (c - 1) / 2;
                                if (d_knn_less(heap_d2[parent], heap_id[parent], heap_d2[c], heap_id[c])) {
                                    const float td = heap_d2[parent]; heap_d2[parent] = heap_d2[c]; heap_d2[c] = td;
                                    const int32_t ti = heap_id[parent]; heap_id[parent] = heap_id[c]; heap_id[c] = ti;
                                    c = parent;
                                } else break;
                            }
                        } else if (d_knn_less(d2, pid, heap_d2[0], heap_id[0])) {
                            // Better than the current worst: replace root, sift DOWN.
                            heap_d2[0] = d2; heap_id[0] = pid;
                            int c = 0;
                            while (true) {
                                const int l = 2 * c + 1, r = 2 * c + 2;
                                int worst = c;
                                if (l < kK && d_knn_less(heap_d2[worst], heap_id[worst], heap_d2[l], heap_id[l])) worst = l;
                                if (r < kK && d_knn_less(heap_d2[worst], heap_id[worst], heap_d2[r], heap_id[r])) worst = r;
                                if (worst == c) break;
                                const float td = heap_d2[worst]; heap_d2[worst] = heap_d2[c]; heap_d2[c] = td;
                                const int32_t ti = heap_id[worst]; heap_id[worst] = heap_id[c]; heap_id[c] = ti;
                                c = worst;
                            }
                        }
                    }
                }
            }
        }

        // ---- Safe-radius stopping rule (the correctness-critical fix a
        // naive "stop once heap_size==kK" misses) --------------------------
        // Having found kK CANDIDATES within the scanned rings does NOT by
        // itself prove they are the true kK NEAREST points: the query can
        // sit anywhere inside its own cell, so a point in an UNSCANNED cell
        // just beyond ring `ring` can still be closer than the worst
        // candidate found so far. The provable guarantee (kernels.cuh's
        // file header derives it): after scanning every cell within
        // Chebyshev distance `ring` of the query's cell, EVERY unscanned
        // cell's nearest point is at Euclidean distance >= ring*cell from
        // the query — so the search is safe to stop only once the heap is
        // full AND its worst (largest) distance is already <= ring*cell.
        // Until both hold, there could be a closer point still unscanned,
        // and the loop must widen to the next ring.
        if (heap_size >= kK) {
            const float safe_radius = static_cast<float>(ring) * cell;
            if (heap_d2[0] <= safe_radius * safe_radius) break;   // provably found the true kK nearest: stop
        }
    }

    // Sort the final (<=kK) heap contents ascending by knn_less — a plain
    // insertion sort (kK==16: trivially cheap) so the neighbor list is in
    // the CANONICAL order every CPU/brute-force twin also produces.
    #pragma unroll
    for (int a = 1; a < kK; ++a) {
        if (a >= heap_size) break;
        const float kd = heap_d2[a]; const int32_t ki = heap_id[a];
        int b = a - 1;
        while (b >= 0 && d_knn_less(kd, ki, heap_d2[b], heap_id[b])) {
            heap_d2[b + 1] = heap_d2[b]; heap_id[b + 1] = heap_id[b];
            --b;
        }
        heap_d2[b + 1] = kd; heap_id[b + 1] = ki;
    }

    if (out_neighbor_ids != nullptr) {
        // OPTIONAL write (nullptr during the throughput pass — see the file
        // header): the correctness pass's VERIFY(knn) gate needs this list;
        // production/throughput code never reads it back, so skipping the
        // write there is honest, not a shortcut.
        for (int a = 0; a < kK; ++a) {
            out_neighbor_ids[q * kK + a] = (a < heap_size) ? heap_id[a] : -1;
        }
    }
    out_found[q] = heap_size;

    // ---- STEP 3: mean-shifted covariance (kernels.cuh file header) --------
    // Two passes over the CACHED neighbor positions (gathered once here,
    // reused for both passes — no second global-memory round trip): first
    // the centroid, then the covariance around it. See THEORY.md "Numerical
    // considerations" for why this beats the one-pass E[pp^T]-mean*mean^T
    // formula at real LiDAR ranges.
    float nx[kK], ny[kK], nz[kK];
    #pragma unroll
    for (int a = 0; a < kK; ++a) {
        if (a < heap_size) {
            const int pid = heap_id[a];
            nx[a] = xyz[pid * 3 + 0]; ny[a] = xyz[pid * 3 + 1]; nz[a] = xyz[pid * 3 + 2];
        } else {
            nx[a] = 0.0f; ny[a] = 0.0f; nz[a] = 0.0f;   // unused past heap_size; never read below
        }
    }

    const int m = max(1, heap_size);   // guard div-by-zero for the (should-never-happen) 0-neighbor case
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int a = 0; a < kK; ++a) {
        if (a >= heap_size) break;
        mx += nx[a]; my += ny[a]; mz += nz[a];
    }
    mx /= static_cast<float>(m); my /= static_cast<float>(m); mz /= static_cast<float>(m);

    float cxx = 0.0f, cxy = 0.0f, cxz = 0.0f, cyy = 0.0f, cyz = 0.0f, czz = 0.0f;
    #pragma unroll
    for (int a = 0; a < kK; ++a) {
        if (a >= heap_size) break;
        const float dx = nx[a] - mx, dy = ny[a] - my, dz = nz[a] - mz;
        cxx += dx * dx; cxy += dx * dy; cxz += dx * dz;
        cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
    }
    const float inv_m = 1.0f / static_cast<float>(m);
    const float cov[6] = { cxx * inv_m, cxy * inv_m, cxz * inv_m, cyy * inv_m, cyz * inv_m, czz * inv_m };

    // ---- STEP 4: eigensolve -------------------------------------------------
    float eigenvalues[3]; float eigenvectors[3][3];
    d_jacobi_eigen_3x3(cov, eigenvalues, eigenvectors);

    // ---- STEP 5: normal = smallest-eigenvalue eigenvector, sensor-oriented ---
    float nrm[3] = { eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2] };
    const float view[3] = { sensor_x - qp[0], sensor_y - qp[1], sensor_z - qp[2] };
    const float dotv = nrm[0] * view[0] + nrm[1] * view[1] + nrm[2] * view[2];
    if (dotv < 0.0f) { nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2]; }

    // ---- STEP 6: surface variation (curvature proxy) -----------------------
    const float sum_ev = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    const float curvature = (sum_ev > 1.0e-12f) ? (eigenvalues[0] / sum_ev) : 0.0f;

    // ---- STEP 7: degeneracy flag --------------------------------------------
    int32_t degeneracy = kDegenClean;
    if (heap_size < kK) degeneracy = kDegenIsolated;
    else if (curvature > kCurvatureDegenThreshold) degeneracy = kDegenEdgeCorner;

    // ---- Write outputs -------------------------------------------------------
    out_normal[q * 3 + 0] = nrm[0]; out_normal[q * 3 + 1] = nrm[1]; out_normal[q * 3 + 2] = nrm[2];
    out_eigenvalues[q * 3 + 0] = eigenvalues[0]; out_eigenvalues[q * 3 + 1] = eigenvalues[1]; out_eigenvalues[q * 3 + 2] = eigenvalues[2];
    out_curvature[q] = curvature;
    out_degeneracy[q] = degeneracy;
}

void launch_estimate_normals(int n, const float* d_xyz,
                             const unsigned long long* d_unique_key, int num_voxels,
                             const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                             float cell, float sensor_x, float sensor_y, float sensor_z,
                             float* d_out_normal, float* d_out_eigenvalues, float* d_out_curvature,
                             int32_t* d_out_degeneracy, int32_t* d_out_found, int32_t* d_out_neighbor_ids)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    estimate_normals_kernel<<<grid, block>>>(n, d_xyz, d_unique_key, num_voxels, d_seg_start, d_idx_sorted, n_sorted,
                                             cell, sensor_x, sensor_y, sensor_z,
                                             d_out_normal, d_out_eigenvalues, d_out_curvature,
                                             d_out_degeneracy, d_out_found, d_out_neighbor_ids);
    CUDA_CHECK_LAST_ERROR("estimate_normals_kernel launch");
}
