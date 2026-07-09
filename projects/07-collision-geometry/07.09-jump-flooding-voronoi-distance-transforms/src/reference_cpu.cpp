// ===========================================================================
// reference_cpu.cpp — EXACT brute-force oracle for project 07.09
//                     Jump-flooding Voronoi/distance transforms
//
// This oracle differs from 33.01/09.01's in one important way, worth
// pausing on: it is not a "same algorithm, serial" twin — it is a DIFFERENT
// (exact, quadratic) algorithm. The GPU runs approximate-but-fast JFA; this
// file computes the true nearest seed for every cell by exhaustive scan.
// Verification is therefore a BOUNDS check against exactness (label
// mismatches ≤ 0.5% of cells, distance error ≤ 2 cells — contract in
// kernels.cuh), not a tolerance on identical math. When the reference
// implements a different algorithm than the kernel, it verifies the
// ALGORITHM'S promise, not just the port — both oracle styles appear
// throughout this repository, and knowing which one you need is part of
// the craft (CLAUDE.md §5).
//
// Everything is integer arithmetic (squared distances in int64), so this
// file's output is bit-identical on every platform — no tolerances anywhere
// on the CPU side.
//
// Cost: O(W·H·N) — 512×512 × 64 seeds ≈ 17M distance evaluations, a
// fraction of a second; the honest quadratic baseline that motivates JFA's
// O(W·H·log) pass structure.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // layouts + this function's signature (int4, kSeedStride)

// ---------------------------------------------------------------------------
// voronoi_exact_cpu — true nearest seed for every cell, by exhaustive scan.
//
// Tie policy: on exactly equal squared distance, the SMALLEST seed id wins —
// deterministic, but deliberately NOT relied on by the comparator (a tie
// cell has two equally-true answers; main.cu counts equal-distance label
// disagreements as agreement).
// ---------------------------------------------------------------------------
void voronoi_exact_cpu(int width, int height,
                       const int* seeds, int n_seeds,
                       int4* cells)
{
    for (int y = 0; y < height; ++y) {           // every cell, row-major...
        for (int x = 0; x < width; ++x) {
            long long best_d2 = 0x7fffffffffffLL; // "infinity" (same sentinel value as the kernel)
            int bx = -1, by = -1, bid = -1;       // best seed found so far

            for (int s = 0; s < n_seeds; ++s) {  // ...scans every seed (the O(N) the GPU avoids)
                const int sx = seeds[s * kSeedStride + 0];
                const int sy = seeds[s * kSeedStride + 1];
                const int id = seeds[s * kSeedStride + 2];
                // Integer squared distance — exact; int64 so no overflow
                // for any realistic grid (dx, dy < 2^15 ⇒ d2 < 2^31).
                const long long d2 = (long long)(x - sx) * (x - sx)
                                   + (long long)(y - sy) * (y - sy);
                // '<' keeps the first (lowest-id) seed on ties because the
                // seed list is scanned in id order — the tie policy above.
                if (d2 < best_d2) { best_d2 = d2; bx = sx; by = sy; bid = id; }
            }
            cells[y * width + x] = { bx, by, bid, 0 };   // same int4 state the GPU produces
        }
    }
}
