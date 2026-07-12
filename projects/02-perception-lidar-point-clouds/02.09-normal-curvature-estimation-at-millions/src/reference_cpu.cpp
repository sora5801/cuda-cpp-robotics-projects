// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU references for project 02.09 (Normal +
//                     curvature estimation at millions of points/sec)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) it is the CORRECTNESS ORACLE
// main.cu diffs the GPU result against; (2) it is the TEACHING BASELINE
// that makes the parallelization (and the measured speed-up) legible.
//
// Independence ruling for THIS project (CLAUDE.md §5, the standard text —
// see docs/PROJECT_TEMPLATE/src/reference_cpu.cpp for the full policy)
// --------------------------------------------------------------------
// * Data-layout arithmetic (voxel-key packing, knn_less, squared_distance3)
//   is SHARED, single-sourced in kernels.cuh — a mismatch there is a bug
//   class of its own, not "independence" (see that header's rationale).
// * The SEARCH ALGORITHMS are independently typed, THREE separate ways:
//     1) kernels.cu's estimate_normals_kernel — a sorted-array + binary-
//        search voxel index, streaming candidates through a bounded max-heap.
//     2) estimate_normals_cpu below — an std::unordered_map voxel index (a
//        DIFFERENT data structure entirely), collecting all ring candidates
//        into a std::vector and finishing with std::partial_sort (a
//        DIFFERENT algorithm than a streaming heap).
//     3) estimate_normal_brute_force below — no spatial index AT ALL: an
//        O(n) linear scan over every point, for every query. Shares nothing
//        with either 1 or 2. This is the "closed-form/analytic solution, a
//        physical invariant, or a negative control" tier the independence
//        ruling asks every project to carry — GATE brute_force_anchor in
//        main.cu uses it as a spatial-index-bug-proof anchor over a
//        documented subset (O(n) per query is deliberately never run over
//        the full point set).
// * The EIGENSOLVER (jacobi_eigen_3x3_cpu below) is ALSO independently
//   typed versus kernels.cu's d_jacobi_eigen_3x3: same algorithm FAMILY
//   (cyclic Jacobi, 02.03's cited precedent), but this file derives the
//   rotation angle via the direct theta = 0.5*atan2(2*apq, aqq-app) closed
//   form (the textbook formula) where kernels.cu uses the numerically
//   preferred stable-tan-half-angle construction — two different, both
//   textbook-correct routes to the SAME rotation, so a bug unique to either
//   formula is caught by VERIFY(eigen)'s cross-comparison instead of hiding
//   behind one shared function.
// * The one genuinely SHARED helper below (fit_from_neighbor_ids, static to
//   this file) turns an already-found neighbor id LIST into geometry
//   (mean -> covariance -> eigensolve -> normal -> curvature -> degeneracy).
//   It is shared BETWEEN this file's own two search functions (not with the
//   GPU): both callers hand it a neighbor list found by a DIFFERENT search
//   algorithm, so a bug in EITHER search still shows up as a mismatched
//   neighbor_ids/normal against the GPU; a bug in the FITTING step itself is
//   caught separately by VERIFY(eigen)/VERIFY(normals) against kernels.cu's
//   independently-typed d_jacobi_eigen_3x3 path — the two failure modes
//   (wrong neighbors vs. right neighbors/wrong math) are cross-checked by
//   different comparisons, not conflated into one.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <cmath>

// ---------------------------------------------------------------------------
// jacobi_eigen_3x3_cpu — independent cyclic-Jacobi eigensolver for a
// symmetric 3x3 matrix (see the file header for why this uses the DIRECT
// atan2 rotation-angle formula instead of kernels.cu's stable-tan form).
//
// theta = 0.5 * atan2(2*apq, aqq - app) is the textbook closed form for the
// rotation angle that zeroes A[p][q] (Golub & Van Loan §8.4, cited in
// THEORY.md): it comes directly from setting the OFF-DIAGONAL of the
// rotated 2x2 block to zero and solving for theta with the double-angle
// identity. It is mathematically equivalent to kernels.cu's stable-tan
// construction (both solve the same equation) but numerically distinct
// (atan2 vs. a rational tan formula) — exactly the "two independent, both-
// correct routes" the file header promises.
// ---------------------------------------------------------------------------
static void jacobi_rotate_cpu(float A[3][3], float V[3][3], int p, int q)
{
    const float apq = A[p][q];
    if (std::fabs(apq) < 1.0e-12f) return;

    const float theta = 0.5f * std::atan2(2.0f * apq, A[q][q] - A[p][p]);
    const float c = std::cos(theta);
    const float s = std::sin(theta);

    const float app = A[p][p], aqq = A[q][q];
    // Standard 2x2 rotation of the (p,q) block: [c -s; s c]^T * [[app,apq],[apq,aqq]] * [c -s; s c].
    A[p][p] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
    A[q][q] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
    A[p][q] = 0.0f;
    A[q][p] = 0.0f;

    const int r = 3 - p - q;   // the third index, neither p nor q
    const float arp = A[r][p], arq = A[r][q];
    A[r][p] = A[p][r] = c * arp - s * arq;
    A[r][q] = A[q][r] = s * arp + c * arq;

    for (int i = 0; i < 3; ++i) {
        const float vip = V[i][p], viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
    }
}

void jacobi_eigen_3x3_cpu(const float cov[6], float eigenvalues[3], float eigenvectors[3][3])
{
    float A[3][3] = {
        { cov[0], cov[1], cov[2] },
        { cov[1], cov[3], cov[4] },
        { cov[2], cov[4], cov[5] },
    };
    float V[3][3] = { {1.0f,0.0f,0.0f}, {0.0f,1.0f,0.0f}, {0.0f,0.0f,1.0f} };

    for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
        jacobi_rotate_cpu(A, V, 0, 1);
        jacobi_rotate_cpu(A, V, 0, 2);
        jacobi_rotate_cpu(A, V, 1, 2);
    }

    float ev[3] = { A[0][0], A[1][1], A[2][2] };
    float vec[3][3];
    for (int i = 0; i < 3; ++i) { vec[i][0] = V[0][i]; vec[i][1] = V[1][i]; vec[i][2] = V[2][i]; }

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

    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = ev[i];
        eigenvectors[i][0] = vec[i][0]; eigenvectors[i][1] = vec[i][1]; eigenvectors[i][2] = vec[i][2];
    }
}

// ---------------------------------------------------------------------------
// compute_hash_keys_cpu — the twin of compute_hash_keys_kernel (shared
// voxel-key FORMULA from kernels.cuh; VERIFY-by-comparison, 02.01/02.05's
// identical pattern).
// ---------------------------------------------------------------------------
void compute_hash_keys_cpu(int n, const float* xyz, float cell, unsigned long long* keys_out)
{
    for (int i = 0; i < n; ++i) {
        const int32_t vx = voxel_coord(xyz[i * 3 + 0], cell);
        const int32_t vy = voxel_coord(xyz[i * 3 + 1], cell);
        const int32_t vz = voxel_coord(xyz[i * 3 + 2], cell);
        keys_out[i] = pack_voxel_key(vx, vy, vz);
    }
}

// ---------------------------------------------------------------------------
// build_hash_map_cpu — the independent voxel-hash oracle's data structure:
// std::unordered_map<key, vector<point index>>, built once in O(n).
// Genuinely different from the GPU's sorted-array+binary-search index
// (02.04/02.05's identical "independent data structure" ruling).
// ---------------------------------------------------------------------------
void build_hash_map_cpu(int n, const float* xyz, float cell, HashMapCpu& out_map)
{
    out_map.clear();
    out_map.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int32_t vx = voxel_coord(xyz[i * 3 + 0], cell);
        const int32_t vy = voxel_coord(xyz[i * 3 + 1], cell);
        const int32_t vz = voxel_coord(xyz[i * 3 + 2], cell);
        out_map[pack_voxel_key(vx, vy, vz)].push_back(i);
    }
}

// ---------------------------------------------------------------------------
// fit_from_neighbor_ids — mean-shifted covariance -> eigensolve -> sensor-
// oriented normal -> surface variation -> degeneracy, given an ALREADY-
// DETERMINED neighbor id list. See the file header's last bullet for why
// sharing this step between this file's two search functions is fine.
// ---------------------------------------------------------------------------
static void fit_from_neighbor_ids(const float* xyz, const std::vector<int32_t>& ids,
                                  const float qp[3], float sensor_x, float sensor_y, float sensor_z,
                                  KnnResultCpu& out)
{
    out.neighbor_ids = ids;
    const int m = std::max<int>(1, static_cast<int>(ids.size()));

    // Pass 1: centroid. Pass 2: covariance AROUND the centroid — the
    // mean-shift trick (kernels.cuh file header STEP 3; THEORY.md derives
    // the cancellation this avoids at real LiDAR ranges).
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    for (int32_t id : ids) { mx += xyz[id * 3 + 0]; my += xyz[id * 3 + 1]; mz += xyz[id * 3 + 2]; }
    mx /= static_cast<float>(m); my /= static_cast<float>(m); mz /= static_cast<float>(m);

    float cxx = 0.0f, cxy = 0.0f, cxz = 0.0f, cyy = 0.0f, cyz = 0.0f, czz = 0.0f;
    for (int32_t id : ids) {
        const float dx = xyz[id * 3 + 0] - mx, dy = xyz[id * 3 + 1] - my, dz = xyz[id * 3 + 2] - mz;
        cxx += dx * dx; cxy += dx * dy; cxz += dx * dz;
        cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
    }
    const float inv_m = 1.0f / static_cast<float>(m);
    const float cov[6] = { cxx * inv_m, cxy * inv_m, cxz * inv_m, cyy * inv_m, cyz * inv_m, czz * inv_m };

    float eigenvectors[3][3];
    jacobi_eigen_3x3_cpu(cov, out.eigenvalues, eigenvectors);

    // Normal = smallest-eigenvalue eigenvector, sign-disambiguated toward
    // the sensor (kernels.cuh file header STEP 5) — identical CONVENTION to
    // the GPU kernel, independently typed here.
    float nrm[3] = { eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2] };
    const float view[3] = { sensor_x - qp[0], sensor_y - qp[1], sensor_z - qp[2] };
    const float dotv = nrm[0] * view[0] + nrm[1] * view[1] + nrm[2] * view[2];
    if (dotv < 0.0f) { nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2]; }
    out.normal[0] = nrm[0]; out.normal[1] = nrm[1]; out.normal[2] = nrm[2];

    const float sum_ev = out.eigenvalues[0] + out.eigenvalues[1] + out.eigenvalues[2];
    out.curvature = (sum_ev > 1.0e-12f) ? (out.eigenvalues[0] / sum_ev) : 0.0f;

    out.degeneracy = kDegenClean;
    if (static_cast<int>(ids.size()) < kK) out.degeneracy = kDegenIsolated;
    else if (out.curvature > kCurvatureDegenThreshold) out.degeneracy = kDegenEdgeCorner;
}

// ---------------------------------------------------------------------------
// estimate_normals_cpu — the twin of estimate_normals_kernel: an
// unordered_map voxel index + batch-collect-then-partial_sort KNN (a
// genuinely different algorithm than the GPU's streaming bounded heap over
// a sorted array — see the file header), then the shared fit step.
// ---------------------------------------------------------------------------
void estimate_normals_cpu(int n, const float* xyz, const HashMapCpu& map, float cell,
                          float sensor_x, float sensor_y, float sensor_z,
                          int query_idx, KnnResultCpu& out)
{
    (void)n;
    const float qp[3] = { xyz[query_idx * 3 + 0], xyz[query_idx * 3 + 1], xyz[query_idx * 3 + 2] };
    const int32_t cvx = voxel_coord(qp[0], cell);
    const int32_t cvy = voxel_coord(qp[1], cell);
    const int32_t cvz = voxel_coord(qp[2], cell);

    // Collect ring-1 candidates (the full 3x3x3 cube); widen ring by ring
    // (each new ring a SHELL, never re-scanning an already-visited cell —
    // kernels.cuh STEP 2) until the SAFE-RADIUS stopping rule is provably
    // met: having >= kK candidates is not enough by itself (the query can
    // sit anywhere inside its own cell, so an unscanned cell just beyond
    // the current ring can still hold a closer point) — only once the kK-th
    // smallest candidate distance is <= ring*cell is every unscanned cell
    // PROVABLY farther than every kept candidate (the same argument
    // kernels.cu's estimate_normals_kernel derives; this file's version is
    // independently coded via std::nth_element instead of a running heap
    // comparison, per this file's independence ruling).
    std::vector<std::pair<float, int32_t>> candidates;   // (dist2, id)
    for (int ring = 1; ring <= kMaxRing; ++ring) {
        for (int dz = -ring; dz <= ring; ++dz) {
            for (int dy = -ring; dy <= ring; ++dy) {
                for (int dx = -ring; dx <= ring; ++dx) {
                    const int cheb = std::max(std::abs(dx), std::max(std::abs(dy), std::abs(dz)));
                    if (ring > 1 && cheb != ring) continue;
                    const unsigned long long key = pack_voxel_key(cvx + dx, cvy + dy, cvz + dz);
                    auto it = map.find(key);
                    if (it == map.end()) continue;
                    for (int32_t pid : it->second) {
                        const float pp[3] = { xyz[pid * 3 + 0], xyz[pid * 3 + 1], xyz[pid * 3 + 2] };
                        candidates.emplace_back(squared_distance3(pp, qp), pid);
                    }
                }
            }
        }

        if (static_cast<int>(candidates.size()) >= kK) {
            // Partially order so position kK-1 holds the kK-th smallest
            // distance under the shared tie-break -- cheaper than a full
            // sort, and harmless to the FINAL selection below (partial_sort
            // does not care about candidates' incoming order).
            std::nth_element(candidates.begin(), candidates.begin() + (kK - 1), candidates.end(),
                             [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                                 return knn_less(a.first, a.second, b.first, b.second);
                             });
            const float kth_dist2 = candidates[static_cast<size_t>(kK - 1)].first;
            const float safe_radius = static_cast<float>(ring) * cell;
            if (kth_dist2 <= safe_radius * safe_radius) break;   // provably found the true kK nearest: stop
        }
    }

    // Batch top-K selection via std::partial_sort under the shared knn_less
    // total order — a DIFFERENT algorithm than the GPU's streaming bounded
    // heap (this file's independence ruling, above).
    const int keep = std::min<int>(kK, static_cast<int>(candidates.size()));
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                      [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                          return knn_less(a.first, a.second, b.first, b.second);
                      });

    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(keep));
    for (int i = 0; i < keep; ++i) ids.push_back(candidates[static_cast<size_t>(i)].second);

    fit_from_neighbor_ids(xyz, ids, qp, sensor_x, sensor_y, sensor_z, out);
}

// ---------------------------------------------------------------------------
// estimate_normal_brute_force — the third-tier independent oracle: an O(n)
// linear scan over EVERY point, no spatial index of any kind, then the same
// shared fit-from-neighbor-ids step. Deliberately never run over the whole
// point set (main.cu caps the anchor subset — see kernels.cuh's file header).
// ---------------------------------------------------------------------------
void estimate_normal_brute_force(int n, const float* xyz, float cell,
                                 float sensor_x, float sensor_y, float sensor_z,
                                 int query_idx, KnnResultCpu& out)
{
    (void)cell;   // brute force needs no cell size: it never buckets anything
    const float qp[3] = { xyz[query_idx * 3 + 0], xyz[query_idx * 3 + 1], xyz[query_idx * 3 + 2] };

    std::vector<std::pair<float, int32_t>> all;
    all.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float pp[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
        all.emplace_back(squared_distance3(pp, qp), i);
    }
    const int keep = std::min<int>(kK, static_cast<int>(all.size()));
    std::partial_sort(all.begin(), all.begin() + keep, all.end(),
                      [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                          return knn_less(a.first, a.second, b.first, b.second);
                      });
    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(keep));
    for (int i = 0; i < keep; ++i) ids.push_back(all[static_cast<size_t>(i)].second);

    fit_from_neighbor_ids(xyz, ids, qp, sensor_x, sensor_y, sensor_z, out);
}
