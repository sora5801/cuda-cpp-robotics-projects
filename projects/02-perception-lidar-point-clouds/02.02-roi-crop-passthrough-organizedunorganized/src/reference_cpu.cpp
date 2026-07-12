// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.02
//                     (ROI crop, passthrough, organized<->unorganized
//                     conversion kernels)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) the correctness ORACLE — a
// dead-simple sequential version a reader can verify by eye; (2) the
// TEACHING BASELINE that makes the GPU version's transformation legible.
//
// Independence ruling for THIS project (see docs/PROJECT_TEMPLATE's
// reference_cpu.cpp for the general statement this specializes):
//
//   * SHARED (data-layout contract, kernels.cuh): the geometric predicate
//     formulas (is_passthrough/is_in_box/is_in_frustum/is_fused), the
//     organized-grid indexing (azimuth_bin_of/nearest_ring_of/
//     organized_cell_index), and the 64-bit encoding scheme
//     (pack_range_index et al.). These are DEFINITIONS, not algorithms —
//     duplicating a plane-test formula in two files is a bug class of its
//     own (a typo in one copy silently redefines "inside"), so they are
//     single-sourced and BOTH kernels.cu's device transcriptions AND this
//     file call the exact same kernels.cuh functions (this file, unlike
//     kernels.cu, is compiled by cl.exe and CAN call them directly — no
//     transcription needed here, only on the device side).
//
//   * INDEPENDENT (the actual algorithms below): every compaction here is
//     a single serial "if predicate, push_back" loop — NOT a scan followed
//     by a scatter. This is a genuinely different MECHANISM from the GPU's
//     three-kernel scan-then-scatter pipeline (kernels.cu), so a bug in
//     the GPU's scan (an off-by-one in the up-sweep tree, a dropped
//     block-offset add) manifests as a COUNT or ORDER mismatch against
//     this independent twin — exactly the kind of bug an identically-
//     structured CPU "reference scan" would be blind to. The
//     unorganized->organized twin below uses a plain running-minimum per
//     cell (no atomics, no 64-bit encoding) — see scatter_to_organized_cpu
//     for why this is provably the same answer as the GPU's atomicMin race
//     despite being a completely different mechanism (order-independence
//     of the minimum operator).
//
//   * scan_exclusive_cpu is the SIMPLEST function in this file on purpose
//     (CLAUDE.md "clarity beats speed"): it is the oracle for BOTH GPU
//     scans (hand-rolled Blelloch and Thrust) — main.cu's
//     VERIFY(scan_bitexact) checks all three against each other, element
//     by element, on integer data, so "bit-exact" here is not even a
//     floating-point claim — it is exact equality, full stop.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>       // std::sqrt (range) — std::isnan not needed, is_invalid_point covers it
#include <limits>      // std::numeric_limits<float>::quiet_NaN()
#include <vector>

// ---------------------------------------------------------------------------
// scan_exclusive_cpu — the serial oracle. out_exclusive[i] = sum(in[0..i)).
// A single running accumulator, one pass — this IS the O(n) serial scan
// kernels.cu's "why not the naive serial scan on the GPU" comment
// contrasts against the parallel Blelloch version: same math, one thread.
// ---------------------------------------------------------------------------
void scan_exclusive_cpu(int n, const int* in, int* out_exclusive)
{
    int running = 0;
    for (int i = 0; i < n; ++i) {
        out_exclusive[i] = running;
        running += in[i];
    }
}

// ---------------------------------------------------------------------------
// The five CPU compaction twins. Each is a plain serial filter loop: visit
// points in index order, copy the ones that pass, and (unless the caller
// passes nullptr) record the source index — so the output is, by
// CONSTRUCTION, in ascending original-index order. Comparing this against
// the GPU's scan-based compaction (which THEORY.md "The algorithm" proves
// is ALSO order-preserving) is exactly main.cu's GATE order_preservation:
// two independently-structured algorithms that must agree not just on the
// SET of kept points but on their ORDER.
// ---------------------------------------------------------------------------

int passthrough_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx)
{
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const float z = xyz[i * 3 + 2];
        if (is_passthrough(z)) {
            out_xyz[count * 3 + 0] = xyz[i * 3 + 0];
            out_xyz[count * 3 + 1] = xyz[i * 3 + 1];
            out_xyz[count * 3 + 2] = z;
            if (out_orig_idx) out_orig_idx[count] = i;
            ++count;
        }
    }
    return count;
}

int box_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx)
{
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
        if (is_in_box(x, y, z)) {
            out_xyz[count * 3 + 0] = x;
            out_xyz[count * 3 + 1] = y;
            out_xyz[count * 3 + 2] = z;
            if (out_orig_idx) out_orig_idx[count] = i;
            ++count;
        }
    }
    return count;
}

int frustum_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx)
{
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
        if (is_in_frustum(x, y, z)) {
            out_xyz[count * 3 + 0] = x;
            out_xyz[count * 3 + 1] = y;
            out_xyz[count * 3 + 2] = z;
            if (out_orig_idx) out_orig_idx[count] = i;
            ++count;
        }
    }
    return count;
}

int fused_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx)
{
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
        if (is_fused(x, y, z)) {
            out_xyz[count * 3 + 0] = x;
            out_xyz[count * 3 + 1] = y;
            out_xyz[count * 3 + 2] = z;
            if (out_orig_idx) out_orig_idx[count] = i;
            ++count;
        }
    }
    return count;
}

int valid_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx)
{
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const float x = xyz[i * 3 + 0];
        if (!is_invalid_point(x)) {
            out_xyz[count * 3 + 0] = x;
            out_xyz[count * 3 + 1] = xyz[i * 3 + 1];
            out_xyz[count * 3 + 2] = xyz[i * 3 + 2];
            if (out_orig_idx) out_orig_idx[count] = i;
            ++count;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// scatter_to_organized_cpu — the unorganized->organized CPU twin.
//
// Mechanism: a plain "running minimum per cell", tracked in two parallel
// arrays (best_range/best_idx), visited in a SINGLE pass over the input
// points in index order — no encoding, no atomics, no 64-bit keys. Ties
// (two points landing at the IDENTICAL range in the SAME cell — vanishing
// probability with real float data, but handled explicitly for honesty)
// are broken toward the SMALLER original index, mirroring
// pack_range_index()'s low-32-bit tiebreak on the GPU side.
//
// WHY this is provably the same answer as the GPU's atomicMin race despite
// being a completely different mechanism: the minimum operator is
// COMMUTATIVE and ASSOCIATIVE — min(a,b) = min(b,a), and the minimum of a
// SET does not depend on the order its elements are visited in. The GPU's
// atomicMin race visits points in an unpredictable, hardware-scheduled
// order; this CPU loop visits them in a fixed index order; BOTH compute
// "the minimum (range, index) pair among every point that targeted this
// cell" — the same well-defined SET operation, so they must agree exactly,
// with NO tolerance needed (THEORY.md "How we verify correctness" makes
// this argument in full, contrasting it with 02.01's Method A, where the
// quantity being combined is a SUM — order-dependent under float rounding
// — and therefore needs a measured tolerance instead of bit-exactness).
// ---------------------------------------------------------------------------
OrganizedScatterCpuResult scatter_to_organized_cpu(int n_points, const float* xyz,
                                                   float* organized_xyz_out,
                                                   int* winner_index_out)
{
    std::vector<float> best_range(static_cast<size_t>(kOrganizedCells), -1.0f);  // -1 = "no candidate yet" (range is always >= 0)
    std::vector<int>   best_idx(static_cast<size_t>(kOrganizedCells), -1);
    std::vector<int>   point_cell(static_cast<size_t>(n_points));

    for (int i = 0; i < n_points; ++i) {
        const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
        const int ring = nearest_ring_of(x, y, z);
        const int az   = azimuth_bin_of(x, y);
        const int cell = organized_cell_index(ring, az);
        point_cell[static_cast<size_t>(i)] = cell;

        const float range = std::sqrt(x * x + y * y + z * z);
        const size_t c = static_cast<size_t>(cell);
        // Replace the current best if this point is strictly closer, OR
        // exactly tied and has the smaller index (the pack_range_index
        // tiebreak this mirrors).
        if (best_idx[c] == -1 ||
            range < best_range[c] ||
            (range == best_range[c] && i < best_idx[c])) {
            best_range[c] = range;
            best_idx[c] = i;
        }
    }

    const float nan_f = std::numeric_limits<float>::quiet_NaN();
    int occupied = 0;
    for (int c = 0; c < kOrganizedCells; ++c) {
        const int winner = best_idx[static_cast<size_t>(c)];
        winner_index_out[c] = winner;
        if (winner == -1) {
            organized_xyz_out[c * 3 + 0] = nan_f;
            organized_xyz_out[c * 3 + 1] = nan_f;
            organized_xyz_out[c * 3 + 2] = nan_f;
        } else {
            ++occupied;
            organized_xyz_out[c * 3 + 0] = xyz[winner * 3 + 0];
            organized_xyz_out[c * 3 + 1] = xyz[winner * 3 + 1];
            organized_xyz_out[c * 3 + 2] = xyz[winner * 3 + 2];
        }
    }

    // Same two-traversal reconciliation as the GPU launcher: recount
    // collisions from POINT space (did I win the cell I aimed at?) rather
    // than trusting the algebraic identity n_points - occupied — see
    // kernels.cu's launch_scatter_to_organized comment for why this is a
    // genuine (not tautological) cross-check.
    int collisions = 0;
    for (int i = 0; i < n_points; ++i) {
        if (winner_index_out[point_cell[static_cast<size_t>(i)]] != i) ++collisions;
    }

    return OrganizedScatterCpuResult{ occupied, collisions };
}
