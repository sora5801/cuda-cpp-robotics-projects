// ===========================================================================
// kernels.cuh — interface for project 07.09
//               Jump-flooding Voronoi/distance transforms
//               (easy, visual, useful)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (driver), kernels.cu (GPU jump-flooding
// implementation), and reference_cpu.cpp (the EXACT brute-force oracle).
// All shared layouts live here, once (CLAUDE.md §12).
//
// The problem in one line: given N "seed" cells on a W×H grid, compute for
// EVERY cell (a) which seed is nearest (the Voronoi label) and (b) how far
// away it is (the distance transform). Robotics cares because "distance to
// the nearest obstacle" is the clearance field planners and safety monitors
// live on (domains 06/23/31), and Voronoi diagrams are free-space skeletons
// and coverage partitions (05.16, 22.04).
//
// GRID LAYOUT — row-major, cell (x, y) at index y*W + x, x rightward,
// y downward (image convention; stated because robotics code usually works
// in metric frames — the demo treats one cell = one unit ("cell") and
// consumers scale by their map resolution, e.g. 0.05 m/cell on a costmap).
//
// CELL STATE — an int4 per cell, the unit the JFA passes ping-pong:
//     .x = sx   x-coordinate of this cell's current best-known seed
//     .y = sy   y-coordinate of that seed
//     .z = id   that seed's index in the seed list (0..N-1)
//     .w = pad  unused (int4 keeps the load/store a single 16-byte vector
//               transaction — the reason for the padding; see kernels.cu)
// "No seed known yet" is the sentinel (-1, -1, -1, ·).
//
// SEED LIST LAYOUT — three ints per seed: [x, y, id] with id == its index
// (id carried explicitly so files/subsets stay self-describing).
//
// DISTANCES are computed from INTEGER squared distances dx*dx + dy*dy
// (exact in int32 for any grid up to ~2^15 on a side) and only converted to
// float at the very end — this is what makes GPU-vs-CPU comparison exact
// arithmetic rather than a floating-point tolerance story, and results
// bit-identical across machines.
//
// APPROXIMATION CONTRACT — the one thing to internalize: JFA is an
// APPROXIMATE algorithm. It propagates seed hypotheses in O(log max(W,H))
// passes and can mislabel a small fraction of cells (near-tie boundary
// cells), with distance errors that stay tiny. The oracle is EXACT, so
// verification is explicitly a bounds check, not equality:
//     label mismatches ≤ 0.5% of cells  AND  max |d_jfa − d_exact| ≤ 2 cells
// (bounds justified in THEORY.md §verification; ties — two seeds exactly
// equidistant — count as agreement whichever label won).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <vector_types.h>   // int4 — plain-old-data CUDA vector type; safe
                            // for cl.exe too (the header is host-compatible)

// Ints per seed in the flat seed list: [x, y, id].
constexpr int kSeedStride = 3;

// ---------------------------------------------------------------------------
// launch_jfa — run the full jump-flooding algorithm on the GPU.
//
//   width, height : grid dimensions (cells); must be >= 1. Not required to
//                   be powers of two — JFA's step schedule only needs the
//                   starting step to cover the longer side.
//   seeds         : HOST pointer, n_seeds*kSeedStride ints. Seed cells must
//                   lie inside the grid and be mutually distinct (the loader
//                   in main.cu validates; the kernel trusts).
//   n_seeds       : number of seeds (>= 1).
//   d_cells       : DEVICE pointer, width*height int4 OUT — final cell
//                   states (best seed coords + id per cell, layout above).
//
// What it does internally (details in kernels.cu): initialize the grid to
// sentinel + scatter the seeds; then ping-pong JFA passes with step sizes
// step0, step0/2, ..., 1 (step0 = smallest power of two >= max(W,H)/2,
// minimum 1); each pass, every cell gathers the best-known seed from its 8
// step-offset neighbors and itself. The result buffer is left in d_cells.
// Synchronous enough for callers: every launch is checked; the caller's
// subsequent cudaMemcpy orders after all passes (stream 0 ordering).
// ---------------------------------------------------------------------------
void launch_jfa(int width, int height,
                const int* seeds, int n_seeds,
                int4* d_cells);

// ---------------------------------------------------------------------------
// CPU reference (defined in reference_cpu.cpp) — the EXACT oracle.
//
// Brute force: for every cell, scan ALL seeds, keep the true minimum integer
// squared distance; ties broken toward the smallest seed id (a deterministic
// rule the comparator does NOT rely on — equal-distance cells count as
// agreement regardless of which valid label each side picked).
// O(W*H*N) — the honest quadratic cost that makes the JFA's O(W*H*log)
// pass structure worth teaching. Output layout identical to d_cells.
// ---------------------------------------------------------------------------
void voronoi_exact_cpu(int width, int height,
                       const int* seeds, int n_seeds,
                       int4* cells);

#endif // PROJECT_KERNELS_CUH
