// ===========================================================================
// kernels.cu — GPU kernels for project 01.02
//              Stereo depth: block matching, then Semi-Global Matching (SGM)
//
// Big idea (the whole project in one paragraph)
// -----------------------------------------------
// Stereo matching asks, for every LEFT pixel, "which column of the RIGHT
// image shows the same 3-D point?" That is W*H*D independent comparisons
// (D = 64 candidate disparities) — a textbook MAP, one thread per (pixel,
// disparity) or per pixel-looping-disparity. Two things are NOT
// embarrassingly parallel the same way: turning per-pixel intensities into
// a robust comparable signature (each pixel needs its 7x7 NEIGHBORHOOD —
// a STENCIL), and SGM's path aggregation (each pixel along a path needs its
// PREDECESSOR's result — a sequential recurrence, the one place this file
// is NOT "one thread, one independent answer"). Four kernel families, one
// idea each:
//   1. census_kernel      — stencil:      each thread reads a 7x7 neighborhood
//   2. cost_volume_kernel — map:          each thread does D independent Hamming distances
//   3. sgm_path_kernel    — scan:         each thread marches ONE scanline sequentially
//   4. wta_*/lr_check/median3 — map:      each thread makes one independent decision
//
// All shared layouts, sentinels, and the D-major cost-volume-layout argument
// live in kernels.cuh — read that file's header comment first; it is not
// repeated here.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (a
// line-by-line CPU twin of every kernel below — diff them side by side).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry shared by every "one thread per pixel" kernel in this
// file: a 16x16 2-D block (256 threads — a warp-multiple with plenty of
// blocks per SM at this image size: 384x288 needs 24x18 = 432 blocks, tens
// of times more than an RTX 2080 SUPER's 48 SMs, so the whole GPU stays fed)
// and a grid sized to exactly cover W x H with a ragged-tail guard inside
// each kernel (the classic "ceil-divide the grid, if-guard the thread" idiom
// used throughout this repo, e.g. 07.09's JFA kernels).
// ---------------------------------------------------------------------------
static constexpr int kBlock2D = 16;

static inline dim3 grid2d(int W, int H)
{
    return dim3((W + kBlock2D - 1) / kBlock2D, (H + kBlock2D - 1) / kBlock2D);
}

// ===========================================================================
// 1) CENSUS TRANSFORM — a STENCIL kernel: thread (x, y) reads a (2*half+1)^2
//    neighborhood of the INPUT image and writes ONE uint64_t signature.
//
// Why census and not raw intensity (or SAD/SSD on raw intensity)? Real
// stereo cameras never see IDENTICAL brightness in both views — different
// auto-exposure convergence, vignetting, tiny sensor gain mismatches, even
// dust. SAD/SSD on raw pixels bakes that brightness difference straight
// into the "cost", corrupting every comparison uniformly. Census asks a
// RELATIVE question per pixel pair ("am I brighter than my neighbor?") that
// a uniform brightness/gain offset cannot change — THEORY.md "The problem"
// derives this from the radiometric model in detail; here it is the reason
// the cost volume below can be a plain Hamming distance instead of a
// tolerance-laden intensity difference.
//
// Thread-to-data mapping: thread (bx*16+tx, by*16+ty) owns output pixel
// (x, y) = (blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y).
// Memory: reads up to 49 bytes from GLOBAL memory per thread (no shared-
// memory tiling — a teaching simplification named honestly below), writes
// one 8-byte signature. No shared memory, no atomics, no divergence beyond
// the tail guard and the border branch (which is coherent across whole
// warps near an edge, not scattered — cheap).
// ===========================================================================
__global__ void census_kernel(const unsigned char* __restrict__ img,
                              unsigned long long* __restrict__ census,
                              int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;                          // ragged-tail guard
    const int idx = y * W + x;                              // row-major (kernels.cuh)

    // Border pixels cannot see a full (2*half+1)^2 window — mark them
    // invalid rather than reading out of bounds or silently truncating the
    // window (a truncated window would give border pixels a SYSTEMATICALLY
    // different, weaker signature than interior pixels — a bias, not noise).
    if (x < kCensusHalf || x >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
        census[idx] = kCensusInvalid;
        return;
    }

    const unsigned char center = img[idx];
    unsigned long long bits = 0ULL;   // one bit per neighbor, MSB-first fill order
    int bit = 0;
    // 7x7 = 49 taps minus the center = 48 comparisons -> fits kCensusBits.
    // Unrolled by the compiler (compile-time bounds, kCensusHalf==3); no
    // shared-memory tiling here is a deliberate simplification (documented,
    // not hidden): each thread re-reads its 49-byte neighborhood from
    // global memory independently, so a 16x16 tile of threads re-reads the
    // ~7px overlap between neighbors many times over. THEORY.md "The GPU
    // mapping" quantifies the redundant-traffic cost and names the shared-
    // memory-tiled fix as README Exercise 3 — this project's teaching
    // priority is a census transform a reader can verify BY EYE against
    // reference_cpu.cpp, which a tiled version would obscure.
    for (int wy = -kCensusHalf; wy <= kCensusHalf; ++wy) {
        for (int wx = -kCensusHalf; wx <= kCensusHalf; ++wx) {
            if (wx == 0 && wy == 0) continue;               // the center is compared TO, not with itself
            const unsigned char neighbor = img[(y + wy) * W + (x + wx)];
            // Bit = 1 iff the neighbor is DARKER than the center (strict <
            // — ties clear the bit; an arbitrary but consistent rule shared
            // with the CPU oracle, so GPU and CPU always agree on ties too).
            if (neighbor < center) bits |= (1ULL << bit);
            ++bit;
        }
    }
    census[idx] = bits;
}

void launch_census(const unsigned char* d_img, unsigned long long* d_census, int W, int H)
{
    if (!d_img || !d_census || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_census: invalid arguments (W=%d H=%d)\n", W, H);
        std::exit(EXIT_FAILURE);
    }
    census_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_img, d_census, W, H);
    CUDA_CHECK_LAST_ERROR("census_kernel launch");
}

// ===========================================================================
// 2) COST VOLUME — a MAP kernel: thread (x, y) computes D=64 INDEPENDENT
//    Hamming distances and writes them into the D-major cost volume
//    (layout argued in kernels.cuh — read that first).
//
// popcount64 (device): __popc/__popcll are hardware POPC-instruction
// intrinsics (one instruction on every architecture this repo targets,
// sm_75+) — the "library call" this project uses instead of hand-rolling a
// software popcount ON THE GPU (CLAUDE.md §1's "what would it take to write
// by hand" answer lives on the CPU side: reference_cpu.cpp implements the
// classic SWAR bit-trick popcount by hand, since the host has no single
// POPC instruction guaranteed portable across every build target).
// ===========================================================================
__global__ void cost_volume_kernel(const unsigned long long* __restrict__ census_l,
                                   const unsigned long long* __restrict__ census_r,
                                   unsigned char* __restrict__ cost,
                                   int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix = y * W + x;
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);   // D-major plane stride

    const unsigned long long cl = census_l[pix];
    if (cl == kCensusInvalid) {
        // No signature at this LEFT pixel -> every disparity is unanswerable.
        // Still write the sentinel to all D planes (not just skip) so
        // downstream kernels never read an uninitialized cost — the cost
        // volume is always FULLY defined, just sometimes "no answer".
        for (int d = 0; d < kMaxDisp; ++d) cost[static_cast<size_t>(d) * plane + pix] = kCostInvalid;
        return;
    }

    // Loop over candidate disparities: D independent Hamming distances.
    // Each iteration's WRITE (fixed d, varying x across the warp) lands in
    // one contiguous span under the D-major layout — see kernels.cuh.
    for (int d = 0; d < kMaxDisp; ++d) {
        const int xr = x - d;                                // candidate right column
        unsigned char c;
        if (xr < 0) {
            c = kCostInvalid;                                 // candidate falls off the left edge of the image
        } else {
            const unsigned long long cr = census_r[y * W + xr];
            if (cr == kCensusInvalid) {
                c = kCostInvalid;                              // right pixel has no signature either
            } else {
                // Hamming distance = number of differing bits = popcount(XOR).
                // __popcll: hardware population-count intrinsic on the
                // 64-bit XOR — one SASS instruction; see the file header for
                // why the CPU twin writes this out by hand instead.
                c = static_cast<unsigned char>(__popcll(cl ^ cr));
            }
        }
        cost[static_cast<size_t>(d) * plane + pix] = c;
    }
}

void launch_cost_volume(const unsigned long long* d_census_l,
                        const unsigned long long* d_census_r,
                        unsigned char* d_cost, int W, int H)
{
    if (!d_census_l || !d_census_r || !d_cost || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_cost_volume: invalid arguments (W=%d H=%d)\n", W, H);
        std::exit(EXIT_FAILURE);
    }
    cost_volume_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_census_l, d_census_r, d_cost, W, H);
    CUDA_CHECK_LAST_ERROR("cost_volume_kernel launch");
}

// ===========================================================================
// 3) SGM PATH AGGREGATION — a SCAN kernel: the one kernel in this project
//    where a thread's steps are NOT independent — step t needs step t-1's
//    full D-length result. One thread per SCANLINE (a row for a horizontal
//    path, a column for a vertical path); the thread marches its scanline
//    sequentially, carrying the previous step's D costs in a local array.
//
// The recurrence (THEORY.md "The math" derives WHY this shape, from the
// dynamic-programming relaxation of a 1-D smoothness prior):
//
//     L_r(p, d) = C(p, d)
//                 + min( L_r(p-r, d),                  no jump
//                        L_r(p-r, d-1) + P1,             small jump, one way
//                        L_r(p-r, d+1) + P1,             small jump, other way
//                        min_k L_r(p-r, k) + P2 )         any bigger jump, flat P2
//                 - min_k L_r(p-r, k)                    subtract running min: the
//                                                         standard trick that keeps
//                                                         L_r bounded along an
//                                                         arbitrarily long path
//                                                         instead of drifting up by
//                                                         O(path length) (Hirschmuller
//                                                         2008; THEORY.md derives the
//                                                         bound) — WITHOUT it, int32
//                                                         would eventually not be
//                                                         enough on a long enough path.
//
// main.cu calls this once per direction (4 calls total: L->R, R->L, T->B,
// B->T) into the SAME accumulator d_lsum, which this kernel ADDS into
// (never overwrites) — Lsum(p,d) = sum over the 4 directions of L_r(p,d).
//
// WHY 4 PATHS, NOT 8 (the production number)? Hirschmuller's original SGM
// uses 8 (adding the 4 diagonals) for a tighter approximation of full 2-D
// smoothness — each extra path costs one more full O(W*H*D) pass and one
// more kernel with AWKWARD memory strides (a diagonal path's neighbor is
// neither row- nor column-adjacent in this D-major layout). The 4 axis-
// aligned paths already deliver SGM's headline lesson — comparing this
// project's own BM vs. SGM numbers shows a clear, measured improvement —
// at HALF the passes and with every access pattern already explained by the
// D-major layout above; THEORY.md "Where this sits in the real world" names
// exactly what the missing diagonals buy (mainly: less directional bias at
// diagonal depth edges) and how libSGM/OpenCV implement all 8.
//
// Thread-to-data mapping (both branches share one kernel body — see the
// direction-generic loop below):
//   dx != 0 (horizontal path): thread index = ROW y in [0, H); one thread
//     per row marches x from the START column to the END column stepping
//     by dx (start/end swap with the sign of dx: L->R starts at x=0,
//     R->L starts at x=W-1).
//   dy != 0 (vertical path): thread index = COLUMN x in [0, W); marches y
//     the same way.
// Memory: this kernel's inner per-step loop over d reads/writes a D-major
// STRIDE-(H*W) sequence per thread (D-major favors OTHER kernels' access
// patterns, not this one's per-thread loop — see kernels.cuh); the cross-
// THREAD pattern at a fixed step differs by direction exactly as kernels.cuh
// describes (vertical paths coalesce across threads, horizontal paths pay a
// stride-W cross-thread access — both are far cheaper than the pixel-major
// alternative's stride-(W*D), which is why D-major was still chosen).
// `prev` (the previous step's D-length L_r vector) lives in a per-thread
// LOCAL array — 64 ints = 256 bytes; likely a local-memory (cached, not a
// true register) allocation at this size, a documented perf/clarity
// trade-off in exchange for a kernel body a reader can verify against
// reference_cpu.cpp line by line.
// ===========================================================================
__global__ void sgm_path_kernel(const unsigned char* __restrict__ cost,
                                int* __restrict__ lsum,
                                int W, int H, int P1, int P2, int dx, int dy)
{
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    // Which scanline does THIS thread own, and where does its walk start?
    int line, start, end, step;
    if (dx != 0) {                          // horizontal path: thread = row
        line = blockIdx.x * blockDim.x + threadIdx.x;
        if (line >= H) return;
        start = (dx > 0) ? 0 : (W - 1);
        end   = (dx > 0) ? W : -1;
        step  = dx;
    } else {                                // vertical path: thread = column
        line = blockIdx.x * blockDim.x + threadIdx.x;
        if (line >= W) return;
        start = (dy > 0) ? 0 : (H - 1);
        end   = (dy > 0) ? H : -1;
        step  = dy;
    }

    int prev[kMaxDisp];                     // L_r at the PREVIOUS path step (this thread's scratch)
    bool first = true;

    for (int t = start; t != end; t += step) {
        const int x = (dx != 0) ? t : line;
        const int y = (dx != 0) ? line : t;
        const int pix = y * W + x;

        if (first) {
            // Base case: no predecessor, so L_r(p,d) = C(p,d) exactly — the
            // path's first pixel carries no smoothness information yet.
            for (int d = 0; d < kMaxDisp; ++d) {
                const int c = cost[static_cast<size_t>(d) * plane + pix];
                prev[d] = c;
                lsum[static_cast<size_t>(d) * plane + pix] += c;
            }
            first = false;
            continue;
        }

        // prev_min = min_k L_r(p-r, k) — computed once, reused by every d
        // below (this is what makes the recurrence O(D) per step, not
        // O(D^2): the expensive "any bigger jump" term collapses to one
        // shared scalar instead of a per-d scan).
        int prev_min = prev[0];
        for (int d = 1; d < kMaxDisp; ++d) if (prev[d] < prev_min) prev_min = prev[d];

        int cur[kMaxDisp];
        for (int d = 0; d < kMaxDisp; ++d) {
            const int e0 = prev[d];
            const int e1 = (d > 0)            ? prev[d - 1] + P1 : (prev_min + P2);
            const int e2 = (d < kMaxDisp - 1) ? prev[d + 1] + P1 : (prev_min + P2);
            int m = e0;
            if (e1 < m) m = e1;
            if (e2 < m) m = e2;
            const int e3 = prev_min + P2;
            if (e3 < m) m = e3;
            const int c = cost[static_cast<size_t>(d) * plane + pix];
            cur[d] = c + m - prev_min;         // the running-min subtraction (see the file header)
        }
        for (int d = 0; d < kMaxDisp; ++d) {
            prev[d] = cur[d];
            // Plain (non-atomic) accumulation into the shared d_lsum
            // buffer. This is race-free by construction, not by luck:
            // within ONE kernel launch (one direction), every pixel is
            // touched by EXACTLY ONE thread — its scanline's owner — so no
            // two threads ever write the same address; and the 4 separate
            // launches (one per direction) run in the SAME default stream,
            // which CUDA guarantees executes in issue order, so launch N+1
            // never starts until launch N has fully finished. No atomic is
            // needed for either axis of "safety" here (documented rather
            // than defended with a reflexive atomicAdd — an unnecessary
            // atomic would cost real throughput on a kernel this hot for
            // zero correctness benefit). README Exercise 3 (running the 4
            // directions concurrently on 4 streams) is exactly the point
            // where that trade-off would need revisiting.
            lsum[static_cast<size_t>(d) * plane + pix] += cur[d];
        }
    }
}

void launch_sgm_path(const unsigned char* d_cost, int* d_lsum,
                     int W, int H, int P1, int P2, int dx, int dy)
{
    if (!d_cost || !d_lsum || W < 1 || H < 1 || (dx == 0) == (dy == 0)) {
        std::fprintf(stderr, "launch_sgm_path: invalid arguments (W=%d H=%d dx=%d dy=%d)\n", W, H, dx, dy);
        std::exit(EXIT_FAILURE);
    }
    const int lines = (dx != 0) ? H : W;          // one thread per scanline (see kernel comment)
    const int block = 128;                        // warp-multiple; scanline counts here (288, 384) are modest
    const int grid = (lines + block - 1) / block;
    sgm_path_kernel<<<grid, block>>>(d_cost, d_lsum, W, H, P1, P2, dx, dy);
    CUDA_CHECK_LAST_ERROR("sgm_path_kernel launch");
}

// ===========================================================================
// 4) WINNER-TAKE-ALL — argmin over disparity. Four small, explicit MAP
//    kernels (see kernels.cuh for why not one templated kernel): each
//    thread independently scans its D candidates and keeps the best.
//
// The "_right" kernels compute a RIGHT-referenced disparity map WITHOUT a
// second cost volume, via one identity: the Hamming distance between the
// left signature at column xL and the right signature at column xR = xL-d
// is symmetric in the sense that it equally answers "what is the best xL
// for right-column xR?" — because cost[d, y, xL] was built from
// census_l[xL] and census_r[xL - d] = census_r[xR], the SAME comparison a
// right-referenced cost at (xR, d) would need. So:
//     cost_right(xR, y, d) == cost[d, y, xR + d]     (just re-INDEX, don't recompute)
// This halves the memory this project needs for a full left-right check —
// worth pausing on: the "extra" right-referenced cost volume many
// tutorials build from scratch is redundant information already sitting in
// the left-referenced one, once you notice the index is just a shift.
// ===========================================================================
__global__ void wta_bm_kernel(const unsigned char* __restrict__ cost,
                              unsigned char* __restrict__ disp, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix = y * W + x;
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    if (x < kCensusHalf || x >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
        disp[pix] = kInvalidDisp;              // census-border pixel: no signature was ever computed here
        return;
    }
    int best = 256, best_d = 0;                 // 256 > any uint8_t value, including the 255 sentinel
    for (int d = 0; d < kMaxDisp; ++d) {
        const int c = cost[static_cast<size_t>(d) * plane + pix];
        if (c < best) { best = c; best_d = d; }
    }
    disp[pix] = (best >= kCostInvalid) ? kInvalidDisp : static_cast<unsigned char>(best_d);
}

void launch_wta_bm(const unsigned char* d_cost, unsigned char* d_disp, int W, int H)
{
    if (!d_cost || !d_disp || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_wta_bm: invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }
    wta_bm_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_cost, d_disp, W, H);
    CUDA_CHECK_LAST_ERROR("wta_bm_kernel launch");
}

__global__ void wta_bm_right_kernel(const unsigned char* __restrict__ cost,
                                    unsigned char* __restrict__ disp_r, int W, int H)
{
    const int xr = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (xr >= W || y >= H) return;
    const int pix_r = y * W + xr;
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    if (xr < kCensusHalf || xr >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
        disp_r[pix_r] = kInvalidDisp;
        return;
    }
    int best = 256, best_d = 0;
    for (int d = 0; d < kMaxDisp; ++d) {
        const int xl = xr + d;                  // the symmetric-reuse index shift (see file header)
        if (xl >= W) break;                     // candidate falls off the right edge of the LEFT image
        const int c = cost[static_cast<size_t>(d) * plane + y * W + xl];
        if (c < best) { best = c; best_d = d; }
    }
    disp_r[pix_r] = (best >= kCostInvalid) ? kInvalidDisp : static_cast<unsigned char>(best_d);
}

void launch_wta_bm_right(const unsigned char* d_cost, unsigned char* d_disp_r, int W, int H)
{
    if (!d_cost || !d_disp_r || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_wta_bm_right: invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }
    wta_bm_right_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_cost, d_disp_r, W, H);
    CUDA_CHECK_LAST_ERROR("wta_bm_right_kernel launch");
}

__global__ void wta_sgm_kernel(const int* __restrict__ lsum,
                               unsigned char* __restrict__ disp, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix = y * W + x;
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    if (x < kCensusHalf || x >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
        disp[pix] = kInvalidDisp;
        return;
    }
    long long best = -1;                       // aggregated costs can exceed int range headroom-wise; compare wide
    int best_d = 0;
    for (int d = 0; d < kMaxDisp; ++d) {
        const long long c = lsum[static_cast<size_t>(d) * plane + pix];
        if (best < 0 || c < best) { best = c; best_d = d; }
    }
    disp[pix] = static_cast<unsigned char>(best_d);
}

void launch_wta_sgm(const int* d_lsum, unsigned char* d_disp, int W, int H)
{
    if (!d_lsum || !d_disp || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_wta_sgm: invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }
    wta_sgm_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_lsum, d_disp, W, H);
    CUDA_CHECK_LAST_ERROR("wta_sgm_kernel launch");
}

__global__ void wta_sgm_right_kernel(const int* __restrict__ lsum,
                                     unsigned char* __restrict__ disp_r, int W, int H)
{
    const int xr = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (xr >= W || y >= H) return;
    const int pix_r = y * W + xr;
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    if (xr < kCensusHalf || xr >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
        disp_r[pix_r] = kInvalidDisp;
        return;
    }
    long long best = -1;
    int best_d = 0;
    for (int d = 0; d < kMaxDisp; ++d) {
        const int xl = xr + d;
        if (xl >= W) break;
        const long long c = lsum[static_cast<size_t>(d) * plane + y * W + xl];
        if (best < 0 || c < best) { best = c; best_d = d; }
    }
    disp_r[pix_r] = static_cast<unsigned char>(best_d);
}

void launch_wta_sgm_right(const int* d_lsum, unsigned char* d_disp_r, int W, int H)
{
    if (!d_lsum || !d_disp_r || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_wta_sgm_right: invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }
    wta_sgm_right_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_lsum, d_disp_r, W, H);
    CUDA_CHECK_LAST_ERROR("wta_sgm_right_kernel launch");
}

// ===========================================================================
// 5) LEFT-RIGHT CONSISTENCY CHECK — a MAP kernel that catches the errors
//    WTA cannot see by construction: a wrong match can still have the
//    lowest cost among wrong candidates (repetitive texture, occlusion).
//    Checking that the left and right disparity maps AGREE about the same
//    3-D point is a cheap, purely geometric sanity test that needs no
//    ground truth — THEORY.md "How we verify correctness" explains why
//    this is exactly the kind of check a real robot's stereo node runs
//    online (no ground truth exists at run time).
// ===========================================================================
__global__ void lr_check_kernel(const unsigned char* __restrict__ disp_l,
                                const unsigned char* __restrict__ disp_r,
                                unsigned char* __restrict__ disp_out,
                                int W, int H, int tol)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix = y * W + x;

    const unsigned char dl = disp_l[pix];
    if (dl == kInvalidDisp) { disp_out[pix] = kInvalidDisp; return; }

    const int xr = x - static_cast<int>(dl);
    if (xr < 0 || xr >= W) { disp_out[pix] = kInvalidDisp; return; }   // projects off-frame: unverifiable

    const unsigned char dr = disp_r[y * W + xr];
    if (dr == kInvalidDisp) { disp_out[pix] = kInvalidDisp; return; }

    const int diff = static_cast<int>(dl) - static_cast<int>(dr);
    const int adiff = (diff < 0) ? -diff : diff;
    disp_out[pix] = (adiff <= tol) ? dl : kInvalidDisp;
}

void launch_lr_check(const unsigned char* d_disp_l, const unsigned char* d_disp_r,
                     unsigned char* d_disp_out, int W, int H, int tol)
{
    if (!d_disp_l || !d_disp_r || !d_disp_out || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_lr_check: invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }
    lr_check_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_disp_l, d_disp_r, d_disp_out, W, H, tol);
    CUDA_CHECK_LAST_ERROR("lr_check_kernel launch");
}

// ===========================================================================
// 6) 3x3 MEDIAN FILTER — a MAP + small-STENCIL kernel: SGM's last cleanup
//    step, gathering up to 9 valid neighbors (including the center) and
//    keeping the (lower) median. This is a NOISE filter, not a hole-filler:
//    a kInvalidDisp center stays kInvalidDisp (see kernels.cuh).
// ===========================================================================
__global__ void median3_kernel(const unsigned char* __restrict__ disp_in,
                               unsigned char* __restrict__ disp_out, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix = y * W + x;

    const unsigned char center = disp_in[pix];
    if (center == kInvalidDisp) { disp_out[pix] = kInvalidDisp; return; }

    unsigned char vals[9];
    int n = 0;
    for (int wy = -1; wy <= 1; ++wy) {
        const int ny = y + wy;
        if (ny < 0 || ny >= H) continue;
        for (int wx = -1; wx <= 1; ++wx) {
            const int nx = x + wx;
            if (nx < 0 || nx >= W) continue;
            const unsigned char v = disp_in[ny * W + nx];
            if (v != kInvalidDisp) vals[n++] = v;
        }
    }
    // Insertion sort on <=9 elements: simplest correct sort for a size this
    // small (fewer than 36 compare-swaps worst case) — a sorting NETWORK
    // would be faster still but far less readable; see THEORY.md's general
    // "teaching beats cleverness" stance (CLAUDE.md §1).
    for (int i = 1; i < n; ++i) {
        const unsigned char key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) { vals[j + 1] = vals[j]; --j; }
        vals[j + 1] = key;
    }
    // Lower median for both odd and even counts (index n/2 with integer
    // division): a documented, deterministic tie-break, matched exactly by
    // reference_cpu.cpp so GPU and CPU agree even where n is even.
    disp_out[pix] = vals[n / 2];
}

void launch_median3(const unsigned char* d_disp_in, unsigned char* d_disp_out, int W, int H)
{
    if (!d_disp_in || !d_disp_out || W < 1 || H < 1) {
        std::fprintf(stderr, "launch_median3: invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }
    median3_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_disp_in, d_disp_out, W, H);
    CUDA_CHECK_LAST_ERROR("median3_kernel launch");
}
