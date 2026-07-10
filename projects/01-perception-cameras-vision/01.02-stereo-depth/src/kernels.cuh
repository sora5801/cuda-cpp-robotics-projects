// ===========================================================================
// kernels.cuh — interface for project 01.02
//               Stereo depth: block matching, then Semi-Global Matching (SGM)
//               kernels (teaching progression: census -> cost volume ->
//               winner-take-all block matching -> SGM path aggregation)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver), kernels.cu (the GPU kernels),
// and reference_cpu.cpp (the line-by-line CPU oracle). Everything all three
// must agree on — image layout, the cost-volume MEMORY LAYOUT, disparity
// conventions, and every invalid-value sentinel — is defined HERE, once
// (CLAUDE.md §12), so a mismatch anywhere is a compile-time signature
// mismatch, not a silent runtime bug.
//
// The teaching progression in five lines (THEORY.md derives each step):
//   1. CENSUS TRANSFORM: replace each pixel's raw intensity with a bitstring
//      recording "am I brighter/darker than each of my neighbors?" — robust
//      to the left/right brightness differences real cameras always have.
//   2. COST VOLUME: for every pixel and every candidate disparity, the
//      Hamming distance between the left and right census signatures — one
//      number per (x, y, d), all D=64 of them, for every pixel.
//   3. BLOCK MATCHING (the baseline): per pixel, just take the disparity
//      with the SMALLEST cost — winner-take-all, no smoothness, fast, but
//      noisy and streaky wherever the census signature is ambiguous.
//   4. SEMI-GLOBAL MATCHING: before winner-take-all, AGGREGATE costs along
//      four 1-D paths crossing each pixel (L->R, R->L, T->B, B->T), each
//      penalizing disparity jumps between neighbors — an efficient
//      approximation of a full 2-D smoothness prior that a real GPU can
//      execute as one thread per scanline. SGM's whole point, made visible
//      side by side with BM in this project's demo: SEE the streaks close up.
//
// IMAGE LAYOUT — row-major, pixel (x, y) at index y*W + x, x rightward,
// y downward (image convention, matches every PGM/CSV loader in this repo).
// Both input images are RECTIFIED (epipolar lines are image rows — see
// THEORY.md "the problem"): correspondences for row y live entirely in row y.
//
// DISPARITY CONVENTION — d = xL - xR >= 0 (the left camera sees a point
// d pixels to the right of where the right camera sees it; larger d = nearer
// surface — THEORY.md derives Z = f*B/d from this). Every per-pixel output
// (cost volume, disparity maps) is indexed by the LEFT column x — the LEFT
// image is this project's reference frame throughout.
//
// CENSUS SIGNATURE — uint64_t per pixel, a 7x7 window (kCensusHalf = 3,
// (7*7)-1 = 48 comparison bits used, top 16 bits of the 64 always zero).
// Border pixels (within kCensusHalf of any edge) cannot see a full window;
// their signature is the sentinel kCensusInvalid, and every downstream
// kernel treats that sentinel as "produce no answer here" rather than
// guessing — see kCostInvalid / kInvalidDisp below.
//
// COST VOLUME LAYOUT — the one decision worth arguing about, so it is
// argued here (THEORY.md "The GPU mapping" has the full analysis):
//     cost[d*H*W + y*W + x]                          <-- D-MAJOR (chosen)
// versus the alternative
//     cost[(y*W + x)*kMaxDisp + d]                    <-- pixel-major
// D-major wins for THIS project for two honest reasons:
//   1. The cost-volume-CONSTRUCTION kernel (one thread per pixel, looping
//      d = 0..63 — by far the largest kernel, W*H*D Hamming distances) does
//      W coalesced 1-byte writes per d-iteration under D-major (every thread
//      in a warp writes cost[d*H*W + y*W + x] for consecutive x — one
//      contiguous span); under pixel-major the SAME iteration writes
//      cost[(y*W+x)*64 + d] for consecutive x — a stride-64 scatter, ~64x
//      more memory transactions for the repo's single hottest kernel here.
//   2. It is a genuinely free side effect (not the reason it was chosen,
//      but worth knowing): the SGM VERTICAL paths (one thread per COLUMN,
//      marching y) then also read/write cost[d*H*W + y*W + x] with
//      consecutive threads = consecutive x = consecutive addresses at every
//      fixed (d, y) step — perfectly coalesced, for free. The HORIZONTAL
//      paths (one thread per ROW) do NOT get this for free — consecutive
//      threads (rows y, y+1) are W floats/bytes apart under EITHER layout;
//      D-major's stride-W is still far cheaper than pixel-major's stride-
//      (W*64), so D-major is still the better of two imperfect choices, but
//      the asymmetry is real and stated honestly rather than hidden — a
//      production SGM implementation typically transposes the buffer (or
//      keeps a second, transposed copy) between horizontal and vertical
//      passes to get full coalescing on BOTH; this project keeps ONE buffer
//      for teaching simplicity and names the cost it pays (see kernels.cu).
// Size at the committed sample (384x288, D=64): D*H*W = 7,077,888 bytes of
// uint8_t cost (~6.75 MiB) — small enough to keep entirely resident, large
// enough that the layout choice above is measurable, not academic.
//
// AGGREGATED COST — same D-major indexing, but int32_t (the SGM recurrence
// adds per-step penalties along a path; Hamming distances (0..48) plus the
// classic "subtract the running minimum every step" trick (THEORY.md
// "Numerical considerations") keep every value small, but int32 removes any
// need to reason about it further). Size: D*H*W*4 bytes (~27 MiB at the
// committed sample size).
//
// INVALID-VALUE SENTINELS (all "no meaningful answer here", never a real
// measurement — every consumer checks for these explicitly):
//     kCensusInvalid  (uint64_t, all-ones)  — no full census window here
//     kCostInvalid    (uint8_t,  255)       — Hamming distance is 0..48, so
//                                             255 can never be a real value
//     kInvalidDisp    (uint8_t,  255)       — real disparities are 0..63
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>

// ---------------------------------------------------------------------------
// Shared constants — the single source every kernel, launcher, and CPU
// reference reads (CLAUDE.md §12: "every float* state documents its layout
// in one place"; the same rule applies to every algorithmic constant here).
// ---------------------------------------------------------------------------
constexpr int kMaxDisp = 64;              // D: candidate disparities 0..63
constexpr int kCensusHalf = 3;            // 7x7 census window (half-width)
constexpr int kCensusBits = 48;           // (2*3+1)^2 - 1 comparison bits used

constexpr unsigned long long kCensusInvalid = 0xFFFFFFFFFFFFFFFFULL; // sentinel: no census here
constexpr unsigned char kCostInvalid = 255;   // sentinel: real Hamming distance is 0..48
constexpr unsigned char kInvalidDisp = 255;   // sentinel: real disparity is 0..(kMaxDisp-1)

// SGM smoothness penalties (THEORY.md "The math" derives their semantics):
//   P1 — cost of a SMALL disparity step (|Delta d| == 1) between path
//        neighbors: real surfaces slant/curve, so a *small* jump is normal
//        and only lightly discouraged.
//   P2 — cost of a LARGE disparity step (|Delta d| > 1): usually a real
//        depth discontinuity (an object edge); still allowed, but only when
//        the data term (the Hamming cost) actually supports it. P2 > P1 by
//        design — this asymmetry IS the "semi-" in Semi-Global Matching.
// Tuned empirically for this scene's Hamming-cost scale (0..48) — see
// THEORY.md "How we verify correctness" for the measured BM-vs-SGM effect
// of these exact values.
constexpr int kP1 = 8;
constexpr int kP2 = 48;

constexpr int kLrCheckTolerance = 1;      // max |dispL - dispR| accepted as consistent (disparity levels)

// ---------------------------------------------------------------------------
// launch_census — build the census signature image for one input image.
//   d_img    : DEVICE pointer, W*H uint8_t, row-major intensity image.
//   d_census : DEVICE pointer, W*H uint64_t OUT — signature per pixel
//              (kCensusInvalid at the kCensusHalf-pixel border).
// Launch: one thread per pixel (2D grid) — see kernels.cu for the exact
// block/grid reasoning shared by every "one thread per pixel" kernel below.
// ---------------------------------------------------------------------------
void launch_census(const unsigned char* d_img, unsigned long long* d_census, int W, int H);

// ---------------------------------------------------------------------------
// launch_cost_volume — Hamming-distance cost volume from two census images.
//   d_census_l, d_census_r : DEVICE pointers, W*H uint64_t each (left/right
//                             signatures from launch_census).
//   d_cost                 : DEVICE pointer, kMaxDisp*W*H uint8_t OUT —
//                             D-MAJOR layout cost[d*H*W + y*W + x] (see the
//                             file header for why); kCostInvalid where
//                             either census is invalid or x-d falls outside
//                             the right image's valid-census region.
// Launch: one thread per pixel, looping d = 0..kMaxDisp-1 inside.
// ---------------------------------------------------------------------------
void launch_cost_volume(const unsigned long long* d_census_l,
                        const unsigned long long* d_census_r,
                        unsigned char* d_cost, int W, int H);

// ---------------------------------------------------------------------------
// launch_sgm_path — aggregate the cost volume along ONE of the four 1-D
// SGM paths and ADD the result into d_lsum (accumulation, not overwrite —
// call this once per direction into the SAME buffer; main.cu zeroes d_lsum
// once up front with cudaMemset, then calls this 4 times).
//   d_cost : DEVICE pointer, kMaxDisp*W*H uint8_t (D-major; from launch_cost_volume).
//   d_lsum : DEVICE pointer, kMaxDisp*W*H int32_t IN/OUT — same D-major
//            indexing; this call's path cost is ADDED to whatever is there.
//   P1, P2 : smoothness penalties (see kP1/kP2 above; passed explicitly so
//            README Exercise 2 can retune them without recompiling constants).
//   dx, dy : path direction, exactly one of {dx,dy} nonzero, value +-1:
//            (+1, 0) = left-to-right, (-1, 0) = right-to-left,
//            (0, +1) = top-to-bottom,  (0, -1) = bottom-to-top.
// Launch: one thread per SCANLINE along the path (a row for horizontal
// paths, a column for vertical paths) — see kernels.cu for why this project
// implements 4 paths, not the production-grade 8, and what that trades away.
// ---------------------------------------------------------------------------
void launch_sgm_path(const unsigned char* d_cost, int* d_lsum,
                     int W, int H, int P1, int P2, int dx, int dy);

// ---------------------------------------------------------------------------
// Winner-take-all — argmin over disparity, from EITHER cost representation.
// Four variants because the SOURCE array's element type and the reference
// frame (left vs. right) each change the indexing; keeping them as four
// explicit, readable functions (rather than one templated/branchy one) is a
// deliberate CLAUDE.md §1 "teaching beats cleverness" call — see kernels.cu.
//
//   launch_wta_bm        : argmin over d_cost (uint8_t)  -> LEFT-reference disparity
//   launch_wta_bm_right   : argmin over d_cost (uint8_t)  -> RIGHT-reference disparity
//   launch_wta_sgm        : argmin over d_lsum (int32_t)  -> LEFT-reference disparity
//   launch_wta_sgm_right  : argmin over d_lsum (int32_t)  -> RIGHT-reference disparity
//
// The "_right" variants do NOT recompute a second, right-referenced cost
// volume: they reuse the SAME D-major array via the symmetric-Hamming
// identity cost'(xR, y, d) == cost(xR + d, y, d) (kernels.cu derives this in
// one line) — free left/right consistency checking, no extra memory.
//
// All four write W*H uint8_t disparities (kInvalidDisp at the census-margin
// border, or where every candidate disparity was invalid).
// ---------------------------------------------------------------------------
void launch_wta_bm(const unsigned char* d_cost, unsigned char* d_disp, int W, int H);
void launch_wta_bm_right(const unsigned char* d_cost, unsigned char* d_disp_r, int W, int H);
void launch_wta_sgm(const int* d_lsum, unsigned char* d_disp, int W, int H);
void launch_wta_sgm_right(const int* d_lsum, unsigned char* d_disp_r, int W, int H);

// ---------------------------------------------------------------------------
// launch_lr_check — left-right consistency check (THEORY.md "How we verify
// correctness" explains why this is a correctness filter, not a metric):
//   d_disp_l, d_disp_r : DEVICE pointers, W*H uint8_t (LEFT-ref and
//                         RIGHT-ref disparities, same source e.g. both BM or
//                         both SGM).
//   d_disp_out          : DEVICE pointer, W*H uint8_t OUT — d_disp_l where
//                         |d_disp_r[x - d_disp_l(x,y)] - d_disp_l(x,y)| <=
//                         tol, else kInvalidDisp.
// ---------------------------------------------------------------------------
void launch_lr_check(const unsigned char* d_disp_l, const unsigned char* d_disp_r,
                     unsigned char* d_disp_out, int W, int H, int tol);

// ---------------------------------------------------------------------------
// launch_median3 — 3x3 median filter over a disparity map (SGM's last
// cleanup step; THEORY.md explains why BM deliberately does NOT get one —
// this project wants BM's raw streaks visible for the comparison to mean
// something). Gathers up to 9 in-bounds, non-kInvalidDisp neighbors
// (including the center); kInvalidDisp centers stay kInvalidDisp (no
// inpainting — a median filter smooths noise among agreeing neighbors, it
// does not invent correspondences that were never found).
// ---------------------------------------------------------------------------
void launch_median3(const unsigned char* d_disp_in, unsigned char* d_disp_out, int W, int H);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — line-by-line twins of every launcher
// above, same signatures with HOST pointers and an explicit "_cpu" suffix.
// main.cu's VERIFY stage runs census, the cost volume, ONE aggregation path,
// and the full final disparity through both paths and requires EXACT
// equality — every operation here is integer arithmetic (population count,
// integer min/add), so GPU and CPU are bit-for-bit reproducible; this is
// NOT a floating-point-tolerance project (THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
void census_cpu(const unsigned char* img, unsigned long long* census, int W, int H);
void cost_volume_cpu(const unsigned long long* census_l, const unsigned long long* census_r,
                     unsigned char* cost, int W, int H);
void sgm_path_cpu(const unsigned char* cost, int* lsum, int W, int H, int P1, int P2, int dx, int dy);
void wta_bm_cpu(const unsigned char* cost, unsigned char* disp, int W, int H);
void wta_bm_right_cpu(const unsigned char* cost, unsigned char* disp_r, int W, int H);
void wta_sgm_cpu(const int* lsum, unsigned char* disp, int W, int H);
void wta_sgm_right_cpu(const int* lsum, unsigned char* disp_r, int W, int H);
void lr_check_cpu(const unsigned char* disp_l, const unsigned char* disp_r,
                  unsigned char* disp_out, int W, int H, int tol);
void median3_cpu(const unsigned char* disp_in, unsigned char* disp_out, int W, int H);

#endif // PROJECT_KERNELS_CUH
