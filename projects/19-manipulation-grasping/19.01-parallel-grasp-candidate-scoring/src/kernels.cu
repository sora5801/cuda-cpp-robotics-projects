// ===========================================================================
// kernels.cu — GPU kernels for project 19.01
//              Parallel grasp-candidate scoring: antipodal sampling over
//              point clouds
//
// Three kernels, run in this order per object (main.cu orchestrates; each
// launcher's full contract is documented in kernels.cuh):
//
//   1. estimate_normals_kernel     — PCA surface normals (02.06's pattern).
//   2. generate_candidates_kernel  — hash-pick p1, ray-search for p2.
//   3. score_candidates_kernel     — friction cone + width + clearance gates.
//
// What is REUSED from 02.06 (cited precisely, not silently copied):
//   * The brute-force k-NN neighbor list (unsorted array + tracked "worst"
//     slot) and the in-register cyclic-Jacobi 3x3 eigensolve, in
//     estimate_normals_kernel below — line-for-line the same ALGORITHM as
//     02.06's estimate_normals_kernel, with exactly one policy change
//     (normal orientation: outward here, inward there — see that kernel's
//     header comment).
//   * The "one thread, one full-cloud scan, UNIFORM loop bound" shape of
//     02.06's find_correspondences_kernel — reused twice here (candidate
//     generation's ray search, and scoring's clearance scan), applied to
//     two different per-thread questions ("what's the best point near my
//     ray" and "is anything obstructing my segment") rather than "what's
//     my nearest neighbor".
//
// What is NEW here beyond 02.06:
//   * A per-candidate STATELESS hash (grasp_hash_u32, kernels.cuh) selects
//     each thread's starting contact point with NO shared state and NO
//     host-generated stream to upload — contrast 08.01's host-generated,
//     uploaded noise array.
//   * No reduction kernel at all: unlike 02.06 (which must SUM thousands of
//     points' contributions into one 6x6 system), this project's per-
//     candidate scores are independent outputs, not partial sums of a
//     shared answer — ranking happens by a HOST sort over the downloaded
//     array (see kernels.cuh's launcher comment and README "Prior art"
//     for the 12.01 NMS parallel).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK_LAST_ERROR (CLAUDE.md §6.1 rule 7)

#include <cfloat>                   // FLT_MAX — "nothing found yet" / "no obstruction yet" sentinels
#include <math_constants.h>         // CUDART_PI_F — precise single-precision pi for rad<->deg conversion

// ===========================================================================
// jacobi_rotate / jacobi_eigen_3x3 — REUSED VERBATIM from project 02.06's
// kernels.cu (same algorithm, same reasoning for choosing Jacobi over the
// closed-form cubic eigenvalue formula: no special case near the repeated-
// eigenvalue inputs a flat local patch produces constantly). See 02.06's
// header comment (../../02-perception-lidar-point-clouds/
// 02.06-icp-point-to-point-point-to-plane-gicp/src/kernels.cu) for the full
// derivation of the rotation-angle formula and the convergence argument;
// duplicated here (not shared via a common header) per this repo's
// self-containment rule — CLAUDE.md §4: every project stays individually
// buildable with no cross-project references at build or run time.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void jacobi_rotate(float A[3][3], float V[3][3], int p, int q)
{
    const float apq = A[p][q];
    if (fabsf(apq) < 1e-12f) return;   // already effectively zero: skip (guards the divide below too)

    const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
    const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    const float c = rsqrtf(t * t + 1.0f);   // cos(rotation angle)
    const float s = t * c;                  // sin(rotation angle)

    const float app = A[p][p], aqq = A[q][q];
    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0.0f;
    A[q][p] = 0.0f;

#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (r == p || r == q) continue;
        const float arp = A[r][p], arq = A[r][q];
        A[r][p] = c * arp - s * arq;  A[p][r] = A[r][p];
        A[r][q] = s * arp + c * arq;  A[q][r] = A[r][q];
    }
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const float vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq;
        V[r][q] = s * vrp + c * vrq;
    }
}

__device__ __forceinline__ void jacobi_eigen_3x3(float A[3][3], float V[3][3])
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            V[i][j] = (i == j) ? 1.0f : 0.0f;

#pragma unroll
    for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
        jacobi_rotate(A, V, 0, 1);
        jacobi_rotate(A, V, 0, 2);
        jacobi_rotate(A, V, 1, 2);
    }
}

// ===========================================================================
// Kernel 1: estimate_normals_kernel — PCA surface normal per cloud point.
// ===========================================================================
// One thread per point j: brute-force scan the cloud for the kPcaK nearest
// neighbors, form their 3x3 covariance, take the eigenvector of the
// SMALLEST eigenvalue (the direction the local neighborhood varies LEAST —
// THEORY.md derives why that is the surface normal). Identical algorithm to
// 02.06's estimate_normals_kernel; see that file's header comment for the
// full memory/complexity discussion (register-heavy: ~64 registers for the
// four kPcaK-sized neighbor arrays, honest occupancy trade for a kernel
// that is arithmetic-per-byte-heavy rather than occupancy-bound).
//
// THE ONE POLICY DIFFERENCE FROM 02.06: orientation. 02.06 orients every
// normal TOWARD its ref_point (a point INSIDE the scanned room, so a wall's
// normal points back into the room the sensor occupies — the natural
// convention for an ENCLOSING scan). This project's clouds are solid,
// CONVEX objects sampled on their OUTER surface, and the grasp geometry
// below is written entirely in terms of "the inward normal is where a
// finger pushes" — the natural convention for a graspED object is OUTWARD
// normals (pointing away from the object, into free space, matching every
// grasping paper THEORY.md cites). So the sign flip below points AWAY from
// ref_point instead of toward it — one inequality flipped, everything else
// unchanged. THEORY.md "Numerical considerations" names this the project's
// worked example of the general inward/outward disambiguation problem.
// ---------------------------------------------------------------------------
__global__ void estimate_normals_kernel(
    const float* __restrict__ xyz,      // [n*3] cloud (m)
    int n,
    float ref0, float ref1, float ref2, // interior reference point (m) — the object's centroid
    float* __restrict__ normals)        // [n*3] OUT: unit normals, oriented OUTWARD
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const float qx = xyz[j * 3 + 0];
    const float qy = xyz[j * 3 + 1];
    const float qz = xyz[j * 3 + 2];

    // Unsorted k-nearest-neighbor list + tracked worst slot — see 02.06's
    // kernels.cu header comment for why this beats a sorted insertion at
    // these sizes (we only ever need "what's currently worst", never rank
    // order).
    float nb_d2[kPcaK], nb_x[kPcaK], nb_y[kPcaK], nb_z[kPcaK];
#pragma unroll
    for (int i = 0; i < kPcaK; ++i) nb_d2[i] = FLT_MAX;
    int   worst    = 0;
    float worst_d2 = FLT_MAX;

    for (int m = 0; m < n; ++m) {
        const float mx = xyz[m * 3 + 0];
        const float my = xyz[m * 3 + 1];
        const float mz = xyz[m * 3 + 2];
        const float dx = mx - qx, dy = my - qy, dz = mz - qz;
        // includes m==j (d2=0): the query point is its own nearest
        // neighbor and stays IN the neighborhood — standard PCA-normal
        // practice (02.06; also PCL's implementation).
        const float d2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));
        if (d2 < worst_d2) {
            nb_d2[worst] = d2; nb_x[worst] = mx; nb_y[worst] = my; nb_z[worst] = mz;
            worst = 0; worst_d2 = nb_d2[0];
#pragma unroll
            for (int i = 1; i < kPcaK; ++i) {
                if (nb_d2[i] > worst_d2) { worst_d2 = nb_d2[i]; worst = i; }
            }
        }
    }

    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
#pragma unroll
    for (int i = 0; i < kPcaK; ++i) { cx += nb_x[i]; cy += nb_y[i]; cz += nb_z[i]; }
    const float inv_k = 1.0f / static_cast<float>(kPcaK);
    cx *= inv_k; cy *= inv_k; cz *= inv_k;

    float cxx = 0.0f, cyy = 0.0f, czz = 0.0f, cxy = 0.0f, cxz = 0.0f, cyz = 0.0f;
#pragma unroll
    for (int i = 0; i < kPcaK; ++i) {
        const float dx = nb_x[i] - cx, dy = nb_y[i] - cy, dz = nb_z[i] - cz;
        cxx += dx * dx; cyy += dy * dy; czz += dz * dz;
        cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
    }

    float A[3][3] = { { cxx, cxy, cxz }, { cxy, cyy, cyz }, { cxz, cyz, czz } };
    float V[3][3];
    jacobi_eigen_3x3(A, V);

    int lo = 0;
    if (A[1][1] < A[lo][lo]) lo = 1;
    if (A[2][2] < A[lo][lo]) lo = 2;
    float nx = V[0][lo], ny = V[1][lo], nz = V[2][lo];

    // Defensive renormalization (~24 chained rotations can leave ~1e-6
    // drift on an orthonormal V — cheap to clean up; every downstream
    // formula in score_candidates_kernel assumes a true unit normal).
    const float inv_len = rsqrtf(fmaf(nz, nz, fmaf(ny, ny, nx * nx)));
    nx *= inv_len; ny *= inv_len; nz *= inv_len;

    // ORIENTATION (the one flip from 02.06 — see this kernel's header
    // comment): point AWAY from ref_point, i.e. OUTWARD from the object's
    // interior. If the normal currently points TOWARD ref_point (positive
    // dot with the to-ref vector), flip it.
    const float to_ref_x = ref0 - qx, to_ref_y = ref1 - qy, to_ref_z = ref2 - qz;
    if (fmaf(nz, to_ref_z, fmaf(ny, to_ref_y, nx * to_ref_x)) > 0.0f) {
        nx = -nx; ny = -ny; nz = -nz;
    }

    normals[j * 3 + 0] = nx;
    normals[j * 3 + 1] = ny;
    normals[j * 3 + 2] = nz;
}

// ===========================================================================
// Kernel 2: generate_candidates_kernel — hash-pick p1, ray-search for p2.
// ===========================================================================
// One thread per candidate k in [0, num_candidates). Exactness matters here
// (kernels.cuh: candidate generation is the one stage required to match the
// CPU oracle index-for-index, not just within tolerance) — every value this
// kernel reads or computes has an EXACT CPU twin in reference_cpu.cpp using
// the identical formulas in the identical order.
//
// THE SEARCH, PRECISELY:
//   idx1 = grasp_hash_u32(seed, k) % n                    (stateless pick)
//   p1, n1 = xyz[idx1], normals[idx1]                      (outward normal)
//   ray direction = -n1                                    (INTO the object —
//                                                            where a finger
//                                                            pushing on p1
//                                                            would travel)
//   for every OTHER cloud point q_j, n_j:
//     d = q_j - p1
//     t = dot(d, -n1)             -- how far along the ray q_j projects
//     reject if t outside [kSearchTMinM, kSearchTMaxM]
//     perp = d - t*(-n1) = d + t*n1   -- the part of d PERPENDICULAR to the ray
//     reject if |perp| > kSearchPerpTolM
//     reject if dot(n1, n_j) > kGenConeCosThreshold   (coarse opposing-normal test)
//     keep the point with SMALLEST |perp|^2 seen so far (strict '<': first
//     minimum wins on an exact tie — the same deterministic tie policy
//     02.06 documents for its correspondence search, needed here for the
//     GPU/CPU exact-match requirement).
//   idx2 = that point's index, or -1 if nothing qualified.
//
// GPU MAPPING NOTE (THEORY.md expands this): every thread's loop bound is
// n, UNIFORM across the whole warp regardless of what any thread finds —
// there is no early exit. This means a warp's target-cloud reads at loop
// step m are a BROADCAST (every lane wants xyz[m*3..]) exactly like 02.06's
// correspondence search, so there is no MEMORY divergence. There IS,
// however, a real WORK-VS-OUTCOME imbalance this project is honest about:
// every thread pays the full O(n) scan whether it finds an excellent
// partner in the first few iterations or finds nothing at all — an early-
// exit version would do less total arithmetic but reintroduce the very
// divergence this design avoids (README Exercise names this trade
// explicitly and asks the learner to measure it).
// ---------------------------------------------------------------------------
__global__ void generate_candidates_kernel(
    const float* __restrict__ xyz,       // [n*3] cloud (m)
    const float* __restrict__ normals,   // [n*3] outward unit normals
    int n,
    unsigned int seed,
    int num_candidates,
    GraspCandidate* __restrict__ candidates)  // [num_candidates] OUT
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_candidates) return;

    // Stateless pick of this candidate's contact point — kernels.cuh's
    // grasp_hash_u32 header comment explains why a hash beats a stream.
    const unsigned int idx1u = grasp_hash_u32(seed, static_cast<unsigned int>(k)) % static_cast<unsigned int>(n);
    const int idx1 = static_cast<int>(idx1u);

    const float p1x = xyz[idx1 * 3 + 0];
    const float p1y = xyz[idx1 * 3 + 1];
    const float p1z = xyz[idx1 * 3 + 2];
    const float n1x = normals[idx1 * 3 + 0];
    const float n1y = normals[idx1 * 3 + 1];
    const float n1z = normals[idx1 * 3 + 2];

    float best_perp2 = FLT_MAX;   // squared perpendicular distance of the best candidate so far
    int   best_j     = -1;        // its index, or -1 if nothing has qualified yet

    for (int j = 0; j < n; ++j) {
        if (j == idx1) continue;   // cannot pair a contact with itself

        const float qx = xyz[j * 3 + 0];
        const float qy = xyz[j * 3 + 1];
        const float qz = xyz[j * 3 + 2];
        const float dx = qx - p1x, dy = qy - p1y, dz = qz - p1z;

        // t = d . (-n1): how far q_j projects along the INWARD ray from p1.
        const float t = -fmaf(dz, n1z, fmaf(dy, n1y, dx * n1x));
        if (t < kSearchTMinM || t > kSearchTMaxM) continue;

        // perp = d - t*(-n1) = d + t*n1: the component of d PERPENDICULAR
        // to the ray. |perp|^2 measures "how close is q_j to the ray",
        // independent of t.
        const float perpx = fmaf(t, n1x, dx);
        const float perpy = fmaf(t, n1y, dy);
        const float perpz = fmaf(t, n1z, dz);
        const float perp2 = fmaf(perpz, perpz, fmaf(perpy, perpy, perpx * perpx));
        if (perp2 > kSearchPerpTolM * kSearchPerpTolM) continue;

        // Coarse opposing-normal prefilter (kernels.cuh: looser than the
        // precise friction-cone test the scoring kernel applies next).
        const float njx = normals[j * 3 + 0];
        const float njy = normals[j * 3 + 1];
        const float njz = normals[j * 3 + 2];
        const float dotn = fmaf(n1z, njz, fmaf(n1y, njy, n1x * njx));
        if (dotn > kGenConeCosThreshold) continue;

        if (perp2 < best_perp2) {   // strict '<': first (i.e. lowest-index) minimum wins on a tie
            best_perp2 = perp2;
            best_j = j;
        }
    }

    candidates[k].idx1 = idx1;
    candidates[k].idx2 = best_j;
}

// ===========================================================================
// Kernel 3: score_candidates_kernel — friction cone, width, clearance.
// ===========================================================================
// One thread per candidate c in [0, num_candidates) — num_candidates here
// may be kNumCandidates (the random batch) or kNumCandidates + a handful of
// hand-picked adversarial entries main.cu appended (kernels.cuh); this
// kernel treats every entry identically regardless of its origin.
//
// THE THREE GATES, PRECISELY (THEORY.md "The math" derives the friction-
// cone test from Coulomb's law and the two-contact force-closure theorem):
//   friction_ok  — theta1, theta2 (the angle between the grasp axis and
//                  each contact's INWARD normal) both <= atan(mu).
//   width_ok     — |p2-p1| in [w_min_m, w_max_m].
//   clearance_ok — a SECOND full-cloud scan: no other cloud point lies
//                  within kClearanceRadiusM of the p1-p2 segment, outside
//                  the kClearanceDeadzoneM dead zones at each end.
//
// Why a second O(n) scan instead of folding clearance into
// generate_candidates_kernel's existing scan? Because the clearance test
// needs the SEGMENT (p1 AND p2, and hence the axis and width) — but p2 is
// only known once THAT kernel's scan has finished picking the best partner.
// Two passes, cleanly separated by pipeline stage (generate, then score),
// is simpler to reason about and verify than a single kernel computing a
// running answer to a question it cannot yet fully state — the same
// "clarity over cleverness" call CLAUDE.md §1 asks for.
// ---------------------------------------------------------------------------
__global__ void score_candidates_kernel(
    const float* __restrict__ xyz,       // [n*3] cloud (m)
    const float* __restrict__ normals,   // [n*3] outward unit normals
    int n,
    const GraspCandidate* __restrict__ candidates,  // [num_candidates]
    int num_candidates,
    float mu,                            // Coulomb friction coefficient (unitless)
    float w_min_m, float w_max_m,        // gripper stroke range (m)
    GraspScore* __restrict__ scores)     // [num_candidates] OUT
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_candidates) return;

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
        const float width2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));
        const float width = sqrtf(width2);
        out.width_m = width;

        // axis: unit vector p1 -> p2. width is bounded below by
        // kSearchTMinM > 0 in every candidate that reached this point (its
        // generator enforced t >= kSearchTMinM), so this division is safe.
        const float inv_w = 1.0f / width;
        const float ax = dx * inv_w, ay = dy * inv_w, az = dz * inv_w;

        // antipodal_cos = dot(n1, -n2) = -dot(n1, n2).
        const float dot_n1n2 = fmaf(n1z, n2z, fmaf(n1y, n2y, n1x * n2x));
        out.antipodal_cos = -dot_n1n2;

        // theta1: angle between axis and INWARD normal at p1 (-n1).
        //   cos(theta1) = dot(axis, -n1) = -dot(axis, n1).
        float cos_t1 = -fmaf(az, n1z, fmaf(ay, n1y, ax * n1x));
        cos_t1 = fminf(1.0f, fmaxf(-1.0f, cos_t1));   // clamp: unit-vector dot can drift a hair past +-1
        out.theta1_deg = acosf(cos_t1) * (180.0f / CUDART_PI_F);

        // theta2: angle between REVERSED axis (p2->p1, i.e. -axis) and
        // INWARD normal at p2 (-n2).  cos(theta2) = dot(-axis,-n2) = dot(axis,n2).
        float cos_t2 = fmaf(az, n2z, fmaf(ay, n2y, ax * n2x));
        cos_t2 = fminf(1.0f, fmaxf(-1.0f, cos_t2));
        out.theta2_deg = acosf(cos_t2) * (180.0f / CUDART_PI_F);

        // --- Gate 1: Coulomb friction cone / two-contact force closure ---
        // (THEORY.md "The math" derives alpha = atan(mu) from Coulomb's law
        // and states the Nguyen 1988 two-contact force-closure theorem this
        // test implements.)
        const float alpha_deg = atanf(mu) * (180.0f / CUDART_PI_F);
        out.friction_ok = (out.theta1_deg <= alpha_deg && out.theta2_deg <= alpha_deg) ? 1 : 0;

        // --- Gate 2: gripper stroke -----------------------------------
        out.width_ok = (width >= w_min_m && width <= w_max_m) ? 1 : 0;

        // --- Gate 3: finger-sweep clearance (second full-cloud scan) ---
        // A point obstructs the grasp if it lies within kClearanceRadiusM
        // of the segment, strictly between the two dead zones — see this
        // kernel's header comment for why this needs its own pass.
        unsigned char clearance_ok = 1;
        for (int j = 0; j < n; ++j) {
            if (j == idx1 || j == idx2) continue;
            const float qx = xyz[j * 3 + 0], qy = xyz[j * 3 + 1], qz = xyz[j * 3 + 2];
            const float rx = qx - p1x, ry = qy - p1y, rz = qz - p1z;
            const float t = fmaf(rz, az, fmaf(ry, ay, rx * ax));   // projection onto the axis, meters from p1
            if (t < kClearanceDeadzoneM || t > (width - kClearanceDeadzoneM)) continue;
            const float perpx = fmaf(-t, ax, rx);
            const float perpy = fmaf(-t, ay, ry);
            const float perpz = fmaf(-t, az, rz);
            const float perp2 = fmaf(perpz, perpz, fmaf(perpy, perpy, perpx * perpx));
            if (perp2 < kClearanceRadiusM * kClearanceRadiusM) { clearance_ok = 0; break; }
        }
        out.clearance_ok = clearance_ok;

        out.feasible = (out.friction_ok && out.width_ok && out.clearance_ok) ? 1 : 0;
        out.score = out.feasible ? out.antipodal_cos : kRejectedScore;
    }

    scores[c] = out;
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================

void launch_estimate_normals(int n, const float* d_xyz, const float ref_point[3], float* d_normals)
{
    if (n <= 0) return;
    const int blocks = blocks_for(n, kThreadsPerBlock);
    estimate_normals_kernel<<<blocks, kThreadsPerBlock>>>(
        d_xyz, n, ref_point[0], ref_point[1], ref_point[2], d_normals);
    CUDA_CHECK_LAST_ERROR("estimate_normals_kernel launch");
}

void launch_generate_candidates(int n, const float* d_xyz, const float* d_normals,
                                unsigned int seed, int num_candidates,
                                GraspCandidate* d_candidates)
{
    if (num_candidates <= 0) return;
    const int blocks = blocks_for(num_candidates, kThreadsPerBlock);
    generate_candidates_kernel<<<blocks, kThreadsPerBlock>>>(
        d_xyz, d_normals, n, seed, num_candidates, d_candidates);
    CUDA_CHECK_LAST_ERROR("generate_candidates_kernel launch");
}

void launch_score_candidates(int n, const float* d_xyz, const float* d_normals,
                             const GraspCandidate* d_candidates, int num_candidates,
                             float mu, float w_min_m, float w_max_m,
                             GraspScore* d_scores)
{
    if (num_candidates <= 0) return;
    const int blocks = blocks_for(num_candidates, kThreadsPerBlock);
    score_candidates_kernel<<<blocks, kThreadsPerBlock>>>(
        d_xyz, d_normals, n, d_candidates, num_candidates, mu, w_min_m, w_max_m, d_scores);
    CUDA_CHECK_LAST_ERROR("score_candidates_kernel launch");
}
