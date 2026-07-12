// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.14
//                     Moving-object segmentation from sequential scans
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) it is the CORRECTNESS ORACLE
// — main.cu runs both paths and asserts agreement within a documented
// tolerance; (2) it is the TEACHING BASELINE — read this file, then
// kernels.cu, to see exactly what parallelization changed (the loop became
// threads; the per-element logic is the same idea, typed fresh).
//
// Independence ruling (see kernels.cuh's file header for the full policy,
// copied from docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's canonical text):
//   * Data-layout contracts (pose algebra, range-image indexing, the
//     encoded-key packing) are single-sourced in kernels.cuh and SHARED here
//     — duplicating quaternion algebra or the (ring,az_bin) formula by hand
//     would be pure transcription, exactly 02.08's precedent for its own
//     shared pose helpers.
//   * The ALGORITHMIC CORE — the reprojection scatter loop, the residual
//     MIN-fusion decision, and the CCL edge-build + union-find — is written
//     TWICE, independently, here in the simplest possible sequential C++.
//   * Sharing the pose algebra alone cannot catch a WRONG residual sign or a
//     wrong fusion rule — so main.cu's sign_semantics and
//     disocclusion_mitigation gates compare the FINAL labels against
//     GROUND TRUTH that never touches this file's functions at all (the
//     required "gate that does not route through the shared code").
//
// Rules for this file: plain C++17, no CUDA headers, no cleverness — clarity
// beats speed here, always (this file is compiled by cl.exe; kernels.cuh's
// __CUDACC__ fence hides device-only declarations from it).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <algorithm>   // std::sort, std::min
#include <cmath>       // std::fabs

// ---------------------------------------------------------------------------
// scatter_current_cpu — INDEPENDENT twin of scatter_current_kernel +
// finalize_current_kernel FUSED into one pass: a plain per-cell
// "running-minimum-range" scan over all input points (02.02/02.12's
// identical CPU-twin technique, cited in kernels.cuh) — no encoding, no
// atomics, order-independent (a min is commutative/associative), so it is
// directly comparable to the GPU's atomicMin race result.
//
// Parameters: n_points/ring/az_bin/range_m/cohort/truth/disocc — the
//             current scan's points and their ground-truth payload (the
//             LAST three are read-only oracle fields, never used by any
//             later algorithm stage — kernels.cuh "Ground truth").
// Outputs (each sized [kNumCells], caller-allocated): range_img (0.0f =
//             no return), cohort_img/truth_img/disocc_img (ground truth,
//             mirrored from the winning point; kCohortNone/-1/0 if empty).
// ---------------------------------------------------------------------------
void scatter_current_cpu(int n_points, const int* ring, const int* az_bin, const float* range_m,
                         const int* cohort, const int* truth, const int* disocc,
                         float* range_img, int* cohort_img, int* truth_img, int* disocc_img)
{
    // best_range/best_idx track, per cell, the nearest point seen so far —
    // the sequential equivalent of the GPU's per-cell atomicMin.
    std::vector<float> best_range(static_cast<size_t>(kNumCells), -1.0f);   // -1 = "no candidate yet"
    std::vector<int> best_idx(static_cast<size_t>(kNumCells), -1);

    for (int i = 0; i < n_points; ++i) {
        const int cell = organized_cell_index(ring[i], az_bin[i]);
        const float r = range_m[i];
        if (best_idx[static_cast<size_t>(cell)] < 0 || r < best_range[static_cast<size_t>(cell)]) {
            best_range[static_cast<size_t>(cell)] = r;
            best_idx[static_cast<size_t>(cell)] = i;
        }
        // Tie-break parity with the GPU: on an EXACT range tie the GPU's
        // encoded key keeps the SMALLER point index (unsigned integer
        // comparison of the low 32 bits). Match that here too so the twin
        // comparison is not merely "close" on the (never actually
        // exercised by this project's continuous synthetic ranges) tie case.
        else if (best_idx[static_cast<size_t>(cell)] >= 0 && r == best_range[static_cast<size_t>(cell)] &&
                 i < best_idx[static_cast<size_t>(cell)]) {
            best_idx[static_cast<size_t>(cell)] = i;
        }
    }

    for (int cell = 0; cell < kNumCells; ++cell) {
        const int idx = best_idx[static_cast<size_t>(cell)];
        if (idx < 0) {
            range_img[cell] = 0.0f;
            cohort_img[cell] = kCohortNone;
            truth_img[cell] = -1;
            disocc_img[cell] = 0;
        } else {
            range_img[cell] = range_m[idx];
            cohort_img[cell] = cohort[idx];
            truth_img[cell] = truth[idx];
            disocc_img[cell] = disocc[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// reproject_scatter_cpu — INDEPENDENT twin of reproject_scatter_kernel +
// finalize_prev_kernel for ONE previous scan: the same per-cell
// running-minimum-range scan, using the SHARED reproject_point_to_current /
// local_point_from_ring_az / cell_for_local_point formulas (the permitted
// "pose/geometry model" sharing exception — this file's header).
// ---------------------------------------------------------------------------
void reproject_scatter_cpu(int n_points, const int* ring, const int* az_bin, const float* range_m,
                           Pose pose_j, Pose pose_cur, float* range_img_prev)
{
    std::vector<float> best_range(static_cast<size_t>(kNumCells), -1.0f);

    for (int i = 0; i < n_points; ++i) {
        const Vec3 p_local_j = local_point_from_ring_az(ring[i], az_bin[i], range_m[i]);
        const Vec3 p_in_current = reproject_point_to_current(pose_j, pose_cur, p_local_j);
        int cell; float range_reproj;
        cell_for_local_point(p_in_current, cell, range_reproj);
        if (best_range[static_cast<size_t>(cell)] < 0.0f || range_reproj < best_range[static_cast<size_t>(cell)])
            best_range[static_cast<size_t>(cell)] = range_reproj;
    }

    for (int cell = 0; cell < kNumCells; ++cell)
        range_img_prev[cell] = (best_range[static_cast<size_t>(cell)] < 0.0f) ? 0.0f : best_range[static_cast<size_t>(cell)];
}

// ---------------------------------------------------------------------------
// residual_fuse_cpu — INDEPENDENT twin of residual_fuse_kernel: the exact
// same MIN-fusion rule (kernels.cu's residual_fuse_kernel comment derives
// WHY min was chosen — this function must reproduce that rule, not merely
// "some reasonable fusion"), typed fresh as two nested sequential loops.
// ---------------------------------------------------------------------------
void residual_fuse_cpu(const float* range_img_cur, const std::vector<const float*>& prev_range_imgs,
                       int window_m, float threshold_m,
                       float* fused_evidence_out, int* sign_out, int* candidate_out)
{
    for (int cell = 0; cell < kNumCells; ++cell) {
        const float r_cur = range_img_cur[cell];
        if (r_cur <= 0.0f) {
            fused_evidence_out[cell] = -1.0f;
            sign_out[cell] = 0;
            candidate_out[cell] = 0;
            continue;
        }
        float min_abs = -1.0f;
        float nearest_signed = 0.0f;
        bool have_nearest = false;
        for (int lag = 0; lag < window_m; ++lag) {
            const float r_prev = prev_range_imgs[static_cast<size_t>(lag)][cell];
            if (r_prev <= 0.0f) continue;
            const float residual = r_cur - r_prev;
            const float abs_residual = std::fabs(residual);
            if (min_abs < 0.0f || abs_residual < min_abs) min_abs = abs_residual;
            if (!have_nearest) { nearest_signed = residual; have_nearest = true; }
        }
        fused_evidence_out[cell] = min_abs;
        sign_out[cell] = have_nearest ? (nearest_signed > 0.0f ? 1 : (nearest_signed < 0.0f ? -1 : 0)) : 0;
        candidate_out[cell] = (min_abs >= 0.0f && min_abs >= threshold_m) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// build_moving_edges_cpu — INDEPENDENT twin of build_moving_edges_kernel: a
// plain double loop (cells x 2 forward neighbors — the SAME rule, typed
// fresh), returned as an ascending-sorted canonical edge vector for
// main.cu's set-equality VERIFY (02.12's identical pattern, cited).
// ---------------------------------------------------------------------------
std::vector<std::pair<int,int>> build_moving_edges_cpu(int num_cells, const int* candidate)
{
    std::vector<std::pair<int,int>> edges;
    for (int cell = 0; cell < num_cells; ++cell) {
        if (!candidate[cell]) continue;
        const int ring = cell / kAzimuthBins;
        const int az = cell % kAzimuthBins;
        if (ring + 1 < kNumBeams) {
            const int nb = organized_cell_index(ring + 1, az);
            if (candidate[nb]) edges.emplace_back(cell, nb);
        }
        {
            const int naz = (az + 1) % kAzimuthBins;
            const int nb = organized_cell_index(ring, naz);
            if (candidate[nb]) edges.emplace_back(cell, nb);
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

// ---------------------------------------------------------------------------
// serial_union_find_cpu — INDEPENDENT sequential union-find (02.04/02.12's
// identical twin, GENERIC over any edge list): classic union-by-min with
// full path compression on every find(). Union-by-min's final partition is
// mathematically order-independent, so this is expected to match the GPU's
// finalized parent[] BIT-EXACT (an integer computation — no float rounding
// anywhere in this function).
// ---------------------------------------------------------------------------
static int uf_find(std::vector<int>& parent, int i)
{
    while (parent[static_cast<size_t>(i)] != i) {
        parent[static_cast<size_t>(i)] = parent[static_cast<size_t>(parent[static_cast<size_t>(i)])];  // path halving
        i = parent[static_cast<size_t>(i)];
    }
    return i;
}

void serial_union_find_cpu(int n, const std::vector<std::pair<int,int>>& edges, std::vector<int>& parent_out)
{
    parent_out.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) parent_out[static_cast<size_t>(i)] = i;

    for (const auto& e : edges) {
        int ru = uf_find(parent_out, e.first);
        int rv = uf_find(parent_out, e.second);
        if (ru == rv) continue;
        if (ru > rv) std::swap(ru, rv);
        parent_out[static_cast<size_t>(rv)] = ru;   // union-by-min, same convention as the GPU sweep
    }
    for (int i = 0; i < n; ++i) parent_out[static_cast<size_t>(i)] = uf_find(parent_out, i);   // finalize to fixpoint
}
