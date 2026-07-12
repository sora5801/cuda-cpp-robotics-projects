// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.11
//                     (Scan Context / ring-descriptor loop-closure search)
//
// Two jobs (CLAUDE.md paragraph 5, and the template's independence ruling
// this file follows exactly — see docs/PROJECT_TEMPLATE/src/reference_cpu.cpp
// for the ruling in full):
//
//   1) CORRECTNESS ORACLE — main.cu's VERIFY stage runs sc_build_cpu /
//      ring_key_cpu / sc_shift_distance_cpu against the GPU kernels on
//      identical inputs and requires agreement within a documented
//      tolerance (exact for the scatter-max, small-tolerance for the
//      shift-distance sum — kernels.cuh's file header explains why the two
//      differ). If GPU and CPU disagree, a bug is proven to exist.
//
//   2) TEACHING BASELINE — reading this file first, then kernels.cu, shows
//      exactly what parallelization changed: the scatter-max becomes an
//      atomic race instead of a safe sequential update; the shift-distance
//      mean becomes a block of 64 threads and a tree reduction instead of
//      one loop. Same math, different concurrency story.
//
// Independence ruling for THIS project (kernels.cuh's own header states it,
// repeated here for the file that actually implements it): the SMALL,
// deterministic, formulaic pieces (ring_index_from_range,
// sector_index_from_xy, column_cosine_distance, ring_key_l1_distance) are
// SHARED — called directly from kernels.cuh, not re-derived — because
// duplicating a four-line index formula would be pure token-for-token
// transcription with no independence value. The AGGREGATION LOOPS below
// (the scatter over points, the reduction over sectors) are INDEPENDENTLY
// reimplemented in the simplest possible sequential C++, with NO structural
// resemblance to kernels.cu's parallel scatter/reduce — this is where a
// real GPU-only bug (wrong thread-to-cell mapping, a race, a reduction
// order bug) would actually surface as a GPU-vs-CPU mismatch.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // shared model constants, layouts, the shared index/distance formulas, signatures

#include <cstring>       // std::memset — zeroing sc_all to the empty sentinel

// ---------------------------------------------------------------------------
// sc_build_cpu — sequential twin of sc_build_kernel's scatter. Every point,
// in order, computes its (ring, sector) cell (via the SHARED formulas) and
// updates that cell's running max with a plain "if bigger, replace" — no
// atomic needed: a single thread touching memory in sequence can never
// race with itself. sc_all must already be filled with kEmptyZ by the
// caller (main.cu mirrors the GPU path's separate init step so both paths
// start from the same documented empty sentinel — kernels.cuh's file
// header, and its derivation of why kEmptyZ, not 0.0f, is the only sentinel
// a running MAX can safely seed from in this project's world).
// ---------------------------------------------------------------------------
void sc_build_cpu(int total_points, const float* xyz, const int32_t* point_scan_id,
                  int n_scans, float* sc_all)
{
    (void)n_scans;   // bounds are the caller's responsibility (mirrors the GPU launcher's contract); kept for a readable signature
    for (int i = 0; i < total_points; ++i) {
        const float x = xyz[i * 3 + 0];
        const float y = xyz[i * 3 + 1];
        const float z = xyz[i * 3 + 2];

        const float range_m = std::sqrt(x * x + y * y);   // planar range — see kernels.cu's identical comment
        const int ring   = ring_index_from_range(range_m);
        const int sector = sector_index_from_xy(x, y);

        const int scan = point_scan_id[i];
        float& cell = sc_all[scan * kScCells + ring * kNumSector + sector];
        if (z > cell) cell = z;   // the whole "atomic" in a single-threaded world: just compare and overwrite
    }
}

// ---------------------------------------------------------------------------
// ring_key_cpu — sequential twin of ring_key_kernel. For every (scan, ring),
// count non-empty sector cells and divide by kNumSector.
// ---------------------------------------------------------------------------
void ring_key_cpu(int n_scans, const float* sc_all, float* ringkey_all)
{
    for (int scan = 0; scan < n_scans; ++scan) {
        for (int ring = 0; ring < kNumRing; ++ring) {
            const float* row = sc_all + scan * kScCells + ring * kNumSector;
            int occupied = 0;
            for (int s = 0; s < kNumSector; ++s)
                if (row[s] > kEmptyZ + 1.0f) ++occupied;
            ringkey_all[scan * kNumRing + ring] = static_cast<float>(occupied) / static_cast<float>(kNumSector);
        }
    }
}

// ---------------------------------------------------------------------------
// sc_shift_distance_cpu — sequential twin of sc_shift_distance_kernel. For
// every candidate and every shift, sum column_cosine_distance() over all
// kNumSector sectors in plain left-to-right order and divide by kNumSector.
// This is a DIFFERENT summation order than the GPU's tree reduction —
// floating point addition is not associative, so the two sums can differ by
// a few ULP even though every column comparison feeding them is identical
// (kernels.cuh's file header flags this; main.cu's VERIFY tolerance for
// this stage is a small ABSOLUTE tolerance, not exact equality, for exactly
// this reason).
// ---------------------------------------------------------------------------
void sc_shift_distance_cpu(const float* sc_query, int num_candidates,
                           const float* sc_candidates, float* out_dist)
{
    for (int c = 0; c < num_candidates; ++c) {
        const float* cand = sc_candidates + static_cast<size_t>(c) * kScCells;
        for (int shift = 0; shift < kNumSector; ++shift) {
            float sum = 0.0f;
            for (int s = 0; s < kNumSector; ++s) {
                const int shifted_col = (s + shift) % kNumSector;
                sum += column_cosine_distance(sc_query, s, cand, shifted_col);
            }
            out_dist[static_cast<size_t>(c) * kNumSector + shift] = sum / static_cast<float>(kNumSector);
        }
    }
}
