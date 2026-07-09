// ===========================================================================
// kernels.cu — GPU implementation for project 07.09
//              Jump-flooding Voronoi/distance transforms
//              (easy, visual, useful)
//
// The big idea
// ------------
// Computing "nearest seed for every cell" exactly costs O(W·H·N) — every
// cell scans every seed. The JUMP FLOODING ALGORITHM (JFA, Rong & Tan 2006)
// gets (almost) the same answer in O(W·H·log max(W,H)) by letting cells
// GOSSIP: each pass, every cell asks 8 neighbors at offset ±step "who is
// YOUR best seed so far?" and adopts any better answer; the step starts at
// half the grid and HALVES each pass. Information about a seed hops
// exponentially far in the first passes, then refines locally:
//
//     step 4:  . . . . S . . . .      a seed's influence jumps 4 cells
//     step 2:  . . S ~ S ~ S . .      then fills at radius 2
//     step 1:  . S S S S S S S .      then every gap at radius 1
//
// This is the repo's first GRID / STENCIL-pattern project (a new pattern
// after 33.01/09.01's thread-per-problem): one thread per CELL, 2-D blocks,
// neighbors gathered at long range, and the classic PING-PONG double buffer
// so a pass reads a consistent snapshot while writing the next one.
//
// Why robotics wants it: replace "seed" with "obstacle cell" and the output
// IS the clearance field local planners and safety monitors consume
// (06.x DWA/costmaps 23.01, speed-and-separation 21.04); keep the labels
// and you have Voronoi regions — free-space skeletons (GVD, 05.16) and
// coverage partitions (22.04). Same kernels, one abstraction swap.
//
// Honesty first: JFA is APPROXIMATE (a cell can settle on a good-but-not-
// best seed near region boundaries when the true seed's information never
// hopped through it). Error rates for scattered seeds are a fraction of a
// percent of cells with tiny distance error — quantified against the EXACT
// CPU oracle every run (bounds contract in kernels.cuh; story in THEORY.md).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry: 2-D blocks of 16×16 = 256 threads.
//
// Why 2-D here when 33.01/09.01 used 1-D? The DATA is 2-D and neighbors are
// 2-D offsets: a 16×16 block maps a square tile of the grid, so blockIdx/
// threadIdx arithmetic reads like grid coordinates (and consecutive
// threadIdx.x along a row keeps row-major loads adjacent — x must be the
// fast axis; swapping x/y here is THE classic coalescing mistake in grid
// kernels). 256 threads/block remains the repo default.
// ---------------------------------------------------------------------------
static constexpr int kTile = 16;

// ===========================================================================
// Pass 0a: reset every cell to the sentinel "no seed known".
// One thread per cell; pure 1-D map (the 2-D structure buys nothing for a
// fill, so index math stays flat and the kernel stays 3 lines).
// ===========================================================================
__global__ void jfa_clear_kernel(int4* __restrict__ cells,  // [W*H] OUT: all sentinel
                                 int total)                 // W*H
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    cells[i] = make_int4(-1, -1, -1, 0);   // (-1,-1,-1) = sentinel (kernels.cuh)
}

// ===========================================================================
// Pass 0b: scatter the seeds — seed s claims its own cell.
// One thread per SEED (tiny launch, typically < one block). The seed list
// lives in plain GLOBAL memory: unlike 09.01's robot model, seed reads here
// are NOT uniform across threads (each thread reads a different seed), so
// __constant__ memory — which serializes divergent reads — would be the
// WRONG tool. Choosing the memory space by ACCESS PATTERN, not by data
// size, is the lesson these two projects teach as a pair.
// ===========================================================================
__global__ void jfa_seed_kernel(const int* __restrict__ seeds,  // [n*kSeedStride] (x,y,id) triples
                                int n_seeds,
                                int4* __restrict__ cells,       // [W*H] seed cells claimed
                                int width)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int x  = seeds[s * kSeedStride + 0];
    const int y  = seeds[s * kSeedStride + 1];
    const int id = seeds[s * kSeedStride + 2];
    // Each seed writes a distinct cell (loader guarantees distinctness), so
    // no two threads collide and no atomics are needed — worth saying,
    // because scatter patterns usually DO raise the race question.
    cells[y * width + x] = make_int4(x, y, id, 0);
}

// ===========================================================================
// The JFA pass: every cell gathers from 9 sample points (itself + 8
// neighbors at offset ±step) and keeps the closest seed seen.
//
// Thread-to-data mapping: thread (x, y) = (blockIdx*blockDim + threadIdx)
// owns cell (x, y); guards handle grids that are not multiples of 16.
//
// READ from `in`, WRITE to `out` — the ping-pong discipline. Within one
// pass every thread reads the same consistent snapshot; writing in place
// would let half-updated neighbors leak into this pass's decisions
// (a data race in the read-modify-write sense, and the reason the launcher
// swaps buffers between passes instead).
//
// Memory behavior (the honest stencil story): the center + small-step reads
// are cache-friendly; the LARGE-step passes (±256 on a 512 grid) touch
// cells far apart, so early passes are effectively random-access and lean
// on L2. That is intrinsic to JFA's "teleporting" information flow — the
// price of O(log) passes. int4 loads keep each access one 16-byte vector
// transaction (the reason cells are int4, kernels.cuh).
// No shared memory: a tile would only cache the ±step ring when step < tile
// size — the last pass or two — for marginal gain; measured honesty over
// speculative tiling (Exercise 4 invites you to try it and measure).
// ===========================================================================
__global__ void jfa_pass_kernel(const int4* __restrict__ in,   // [W*H] snapshot read by all threads
                                int4*       __restrict__ out,  // [W*H] next state written once per cell
                                int width, int height,
                                int step)                      // this pass's jump distance (cells)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's cell, x fast axis
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;                 // ragged edges of the 16×16 tiling

    // Start from my own current best (possibly the sentinel).
    int4 best = in[y * width + x];
    // Best squared distance so far; sentinel = "infinity". Integer squared
    // distances are EXACT (no float rounding in the whole algorithm — what
    // makes GPU-vs-CPU comparison arithmetic-exact; kernels.cuh).
    long long best_d2 = (best.x < 0) ? 0x7fffffffffffLL
        : (long long)(x - best.x) * (x - best.x) + (long long)(y - best.y) * (y - best.y);

    // The 3×3 gather at ±step. Unrolled 9-sample loop; the (0,0) sample
    // re-reads self redundantly but keeps the loop branchless-regular
    // (uniform control flow beats one skipped load).
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx * step;
            const int ny = y + dy * step;
            // Border policy: samples outside the grid are simply skipped
            // (no wrap, no clamp) — Voronoi has no periodic boundary here.
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            const int4 cand = in[ny * width + nx];   // neighbor's best-known seed
            if (cand.x < 0) continue;                // neighbor knows nothing yet

            // Would the NEIGHBOR'S seed be closer to ME than my current best?
            const long long d2 = (long long)(x - cand.x) * (x - cand.x)
                               + (long long)(y - cand.y) * (y - cand.y);
            // Strict '<': on exact ties keep the incumbent. Tie cells are
            // genuinely ambiguous (two valid labels); the comparator treats
            // equal-distance disagreements as agreement (kernels.cuh).
            if (d2 < best_d2) { best_d2 = d2; best = cand; }
        }
    }

    out[y * width + x] = best;    // exactly one coalesced int4 store per cell
}

// ===========================================================================
// Host launcher: allocate the ping-pong partner, run the pass schedule.
// ===========================================================================
void launch_jfa(int width, int height,
                const int* seeds, int n_seeds,
                int4* d_cells)
{
    if (width < 1 || height < 1 || n_seeds < 1 || !seeds || !d_cells) {
        std::fprintf(stderr, "launch_jfa: invalid arguments (w=%d h=%d n=%d)\n",
                     width, height, n_seeds);
        std::exit(EXIT_FAILURE);
    }
    const int total = width * height;

    // Upload the seed list (host → device). Seeds are tiny (kilobytes);
    // the alloc/copy cost is negligible next to the passes.
    int* d_seeds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)n_seeds * kSeedStride * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_seeds, seeds,
                          (size_t)n_seeds * kSeedStride * sizeof(int),
                          cudaMemcpyHostToDevice));

    // The ping-pong partner buffer. d_cells (caller's) is buffer A; we
    // allocate B; passes alternate A→B, B→A; a final copy ensures the
    // RESULT ends in the caller's buffer regardless of pass parity.
    int4* d_pong = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pong, (size_t)total * sizeof(int4)));

    // --- pass 0: clear + scatter seeds into buffer A ------------------------
    {
        const int threads = 256;
        jfa_clear_kernel<<<(total + threads - 1) / threads, threads>>>(d_cells, total);
        CUDA_CHECK_LAST_ERROR("jfa_clear_kernel launch");
        jfa_seed_kernel<<<(n_seeds + threads - 1) / threads, threads>>>(d_seeds, n_seeds, d_cells, width);
        CUDA_CHECK_LAST_ERROR("jfa_seed_kernel launch");
    }

    // --- the JFA schedule: step = P/2, P/4, ..., 1 --------------------------
    // P = smallest power of two >= max(W,H). Starting at P/2 lets the first
    // pass carry seed information across half the grid; log2(P) passes total.
    int longest = width > height ? width : height;
    int step0 = 1;
    while (step0 < longest) step0 <<= 1;   // P
    step0 >>= 1;                           // P/2 (min 1 via the loop condition below)
    if (step0 < 1) step0 = 1;              // 1×1 grids still get one pass

    const dim3 block(kTile, kTile);        // 16×16 tile (see kTile comment)
    const dim3 grid((width + kTile - 1) / kTile,
                    (height + kTile - 1) / kTile);

    int4* in  = d_cells;                   // buffer A holds the seeded state
    int4* out = d_pong;

    // VARIANT NOTE — this is "1+JFA" (Rong & Tan's own refinement): one
    // EXTRA step-1 pass BEFORE the halving schedule. Plain JFA's rare
    // failure mode is a seed whose information never hops through some
    // cell's 9-sample funnel; priming each seed's immediate ring first
    // measurably suppresses those misses for one extra O(W·H) pass. We
    // adopted it after the plain version exceeded this project's 2-cell
    // error bound on the 1024² batch (max error 3.9 cells, 13 cells
    // mislabeled) — the bounds check caught it exactly as designed, and
    // THEORY.md §verification tells the full story.
    jfa_pass_kernel<<<grid, block>>>(in, out, width, height, 1);
    CUDA_CHECK_LAST_ERROR("jfa_pass_kernel launch (1+JFA priming pass)");
    { int4* tmp = in; in = out; out = tmp; }

    for (int step = step0; step >= 1; step >>= 1) {
        jfa_pass_kernel<<<grid, block>>>(in, out, width, height, step);
        CUDA_CHECK_LAST_ERROR("jfa_pass_kernel launch");
        // Swap roles: last pass's output is next pass's snapshot. Pointer
        // swap costs nothing — the buffers trade names, not contents.
        int4* tmp = in; in = out; out = tmp;
    }

    // After the loop the RESULT lives in `in` (the last-written buffer).
    // If that is the pong buffer, copy back into the caller's d_cells.
    // (Device-to-device copies run at full memory bandwidth; ~4 MB here.)
    if (in != d_cells)
        CUDA_CHECK(cudaMemcpy(d_cells, in, (size_t)total * sizeof(int4),
                              cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_pong));
    CUDA_CHECK(cudaFree(d_seeds));
}
