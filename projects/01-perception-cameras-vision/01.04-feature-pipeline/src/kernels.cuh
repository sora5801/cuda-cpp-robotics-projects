// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 01.04
//               (Feature pipeline: FAST/Harris detection, ORB descriptors,
//               brute-force Hamming matcher)
//
// Role in the project
// -------------------
// This header is the SINGLE-SOURCED CONTRACT between three translation
// units: kernels.cu (the GPU implementation, nvcc), reference_cpu.cpp (the
// independent CPU oracle, cl.exe), and main.cu (orchestration, nvcc). Per
// the repo's twin-independence ruling (see reference_cpu.cpp's header):
// data LAYOUT (structs, constants, indexing formulas, lookup tables) is
// single-sourced HERE; the ALGORITHMIC CORE (score computation, NMS,
// orientation, descriptor bit-packing, Hamming reduction) is written
// TWICE — once in kernels.cu, independently again in reference_cpu.cpp.
//
// The three-stage pipeline this header describes
// ------------------------------------------------
//   STAGE 1 DETECT   — two corner detectors, taught side by side:
//     FAST-9  (fast_score_kernel)      — all-INTEGER decisions -> the
//                                        GPU/CPU twin is BIT-EXACT.
//     Harris  (sobel_gradient_kernel + harris_response_kernel) — FLOAT
//                                        structure-tensor response -> the
//                                        twin is TOLERANCE-checked.
//     Both feed nms_select_*_kernel (3x3 non-max suppression + compaction);
//     the host then sorts candidates by (score desc, y asc, x asc) and
//     truncates to the top N — a DETERMINISTIC tie-break so the GPU path's
//     final keypoint LIST (not just the raw score array) can be compared
//     bit-for-bit against the CPU path's list (see main.cu "VERIFY").
//
//   STAGE 2 DESCRIBE  — oriented rBRIEF (the "O" and "B" of ORB):
//     orientation_kernel   — intensity centroid -> angle theta (float,
//                             tolerance-checked: atan2 implementations on
//                             GPU vs host libm can differ by a few ULP).
//     describe_kernel      — 256 rotated intensity-pair comparisons packed
//                             into an 8xuint32 bitstring (BIT-EXACT twin).
//
//     How bit-exact descriptors coexist with a tolerant orientation angle
//     (the key design decision of this project, read carefully):
//     real ORB (OpenCV's implementation) does NOT rotate the 256 sample
//     offsets by a continuous angle. It buckets theta into 30 discrete
//     12-degree bins and looks up a PRECOMPUTED rotated-and-rounded offset
//     table per bin (see orientationSteps() / bit_pattern_31_ in OpenCV's
//     orb.cpp) — because re-deriving a rotation matrix from scratch for
//     every one of 256 pairs, per keypoint, per frame, is wasted work when
//     12 degrees of angular resolution is visually indistinguishable for
//     BRIEF's coarse binary comparisons. This project reproduces that
//     design honestly (kOrientBins = 30, matching OpenCV's constant): the
//     table build_rotated_pattern_table() below is SINGLE-SOURCED data
//     (like 01.01's remap LUT), and main.cu quantizes each keypoint's
//     GPU-measured theta into a bin index with orient_to_bin() BEFORE
//     handing that integer bin to both the GPU describe_kernel and the CPU
//     describe_cpu() twin. The two pipelines' continuous theta values may
//     differ by a fraction of a degree (tolerance-checked separately); the
//     bin WIDTH (12 degrees) is ~200x that tolerance, so both pipelines
//     land in the same bin for every keypoint in this project's committed
//     scene (main.cu asserts this explicitly, not just hopes it), and from
//     that point on every input to describe_kernel/describe_cpu() is
//     IDENTICAL integer data — leaving only all-integer pixel comparisons,
//     which are exactly reproducible.
//
//   STAGE 3 MATCH    — brute-force Hamming, all-pairs:
//     hamming_match_kernel — one thread per QUERY descriptor, loops every
//                            TRAIN descriptor, keeps best + second-best
//                            (Hamming) distance via __popc (BIT-EXACT twin:
//                            popcount and integer min-reduction have no
//                            floating point anywhere).
//     Run twice (B-query/A-train, and A-query/B-train) so main.cu can
//     apply BOTH the Lowe ratio test (on the forward direction) and a
//     mutual-consistency cross-check (against the reverse direction).
//
// Why ".cuh"?
// -----------
// The repo convention (CLAUDE.md §12): .cuh headers may contain CUDA-
// specific constructs (__global__, kernel launches) and are included from
// nvcc-compiled .cu files; this header is ALSO included by reference_cpu.cpp
// (compiled by cl.exe, which does not know "__global__"), so every
// device-only declaration is fenced behind #ifdef __CUDACC__. Plain
// functions/structs used by ALL THREE files (the shared data-layout
// contract) live OUTSIDE the fence.
//
// Read this after: main.cu.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t, uint8_t — fixed-width types for bit-exact packing
#include <cmath>     // atan2f/cosf/sinf (host use too — <cmath> is plain C++17, fine for cl.exe)

// ===========================================================================
// Image geometry — SINGLE-SOURCED. scripts/make_synthetic.py renders both
// committed sample images at exactly this resolution; main.cu asserts the
// loaded PGM dimensions match before doing anything else (fail loud, never
// silently truncate — repo convention, see 01.01/01.02's sample loaders).
// ===========================================================================
constexpr int kW = 256;   // image width, px
constexpr int kH = 256;   // image height, px

// ---------------------------------------------------------------------------
// Border margins. Two DIFFERENT margins are used on purpose (a real ORB-SLAM
// design choice, not an oversight):
//   kDetectBorder — the minimum margin the RAW per-pixel score/response
//     kernels need so their own stencil (FAST's radius-3 circle; Harris's
//     radius-1 Sobel + radius-2 box window = 3 total) never reads outside
//     the image. Score/response ARRAYS are computed (and hence VERIFY-
//     comparable) everywhere except this thin rim.
//   kBorder — the (larger) margin a pixel needs to be ELIGIBLE as a
//     KEYPOINT CANDIDATE: NMS only considers centers in [kBorder, W-kBorder)
//     (its 3x3 neighbor reads stay safely inside kDetectBorder), and every
//     accepted keypoint must have a full kOrientPatchRadius disk available
//     for the orientation centroid AND the (rotation-preserves-distance,
//     see describe_kernel's header) rBRIEF sampling — both bounded by
//     kOrientPatchRadius = 15, so kBorder = 16 gives a full pixel of slack
//     even after rounding a rotated offset outward. Real ORB-SLAM applies
//     the identical two-tier idea (a small "edge threshold" for detection,
//     a larger patch-radius margin for description).
// ---------------------------------------------------------------------------
constexpr int kDetectBorder = 3;
constexpr int kBorder       = 16;

// kSobelBorder — the Sobel stencil's OWN minimal border (radius 1: it reads
// a 3x3 neighborhood). Deliberately SMALLER than kDetectBorder: Harris's
// box-sum window (radius kHarrisWinRadius = 2) then reads gx/gy at offsets
// up to +-2 from its center, so it needs gx/gy to be VALID (not zero-
// forced by Sobel's own border) out to (kDetectBorder + kHarrisWinRadius)
// = 3 + 2... the box window's CENTER already only runs out to kDetectBorder
// (3), and its reads extend 2 further, i.e. down to image column
// kDetectBorder - kHarrisWinRadius = 1 — exactly kSobelBorder. If Sobel
// instead zeroed its output out to kDetectBorder (3) like every other
// kernel here, Harris's box sum near the kDetectBorder ring would silently
// mix in zero-forced gradients and under-count — a real bug caught while
// deriving this border arithmetic; the fix is Sobel using the SMALLER,
// individually-correct border it actually needs.
constexpr int kSobelBorder  = 1;

// ===========================================================================
// STAGE 1a: FAST-9 corner detector — the Bresenham circle of 16 pixels at
// radius 3, and the contiguous-arc parameters. See THEORY.md "The
// algorithm" for the ASCII diagram of this circle; kFastCircleX/Y ARE that
// diagram, single-sourced so kernels.cu and reference_cpu.cpp read the
// exact same 16 offsets in the exact same order (index 0 = "north", walking
// clockwise — matching Rosten & Drummond's original numbering).
// ===========================================================================
constexpr int kFastCircleN = 16;                 // points on the Bresenham circle of radius 3
constexpr int kFastArcLen  = 9;                  // FAST-9: need >=9 CONSECUTIVE agreeing points
constexpr int kFastThreshold = 20;                // brightness threshold t, 0..255 intensity units

// (dx, dy) of each of the 16 circle points, index 0 = straight up (north),
// proceeding clockwise — the classic Bresenham circle of radius 3 used by
// every FAST/ORB implementation (OpenCV's pattern32_ table is the same 16
// offsets in a different starting phase; the corner SET found is identical
// either way since the arc test is rotation-invariant around the circle).
//
// Why these three tables are macro-sourced literals, not plain constexpr
// arrays: kernels.cu's device kernels index them with a RUNTIME-shaped
// loop variable (even though #pragma unroll turns it into compile-time
// indices after unrolling, the CUDA language rule is enforced on the
// SYMBOL's storage class before that optimization happens) — a plain
// host-storage-class global array is simply invisible to device code,
// full stop, no exceptions. The fix is NOT to duplicate the literal by
// hand in two places (a drift risk this repo's single-sourcing rule
// exists to prevent) but to define the VALUES once, as a macro, and
// instantiate them into TWO differently-qualified storage locations: the
// plain host arrays below (used by reference_cpu.cpp and any host code)
// and the __constant__-memory device copies kernels.cu declares from the
// SAME macros (see that file's top-of-file constant-memory block).
#define FAST_CIRCLE_X_INIT { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 }
#define FAST_CIRCLE_Y_INIT { -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3 }
constexpr int kFastCircleX[kFastCircleN] = FAST_CIRCLE_X_INIT;
constexpr int kFastCircleY[kFastCircleN] = FAST_CIRCLE_Y_INIT;

// The "high-speed test" quick-reject points: 4 circle indices spaced 90
// degrees apart (north/east/south/west). If fewer than 3 of these 4 pixels
// individually pass the brighter/darker-than-threshold test, NO 9-point
// contiguous arc can possibly exist (a 9-run out of 16 must cover at least
// 3 of any 4 evenly-spaced positions — pigeonhole), so the full 16-point
// test can be skipped. This is the textbook version of the optimization;
// production FAST (and OpenCV's implementation) replaces it with a
// TRAINED DECISION TREE (ID3 over the 16 binary outcomes, learned offline
// from a corpus of images) that reorders and prunes comparisons even more
// aggressively — see THEORY.md "Where this sits in the real world".
#define FAST_QUAD_IDX_INIT { 0, 4, 8, 12 }
constexpr int kFastQuadIdx[4] = FAST_QUAD_IDX_INIT;

// ===========================================================================
// STAGE 1b: Harris corner detector parameters.
// ===========================================================================
constexpr int   kHarrisWinRadius = 2;      // structure-tensor box window: (2*2+1)^2 = 5x5 = 25 taps
constexpr float kHarrisK         = 0.04f;  // the empirical Harris free parameter (Harris & Stephens 1988; 0.04-0.06 is the traditional range)

// ===========================================================================
// STAGE 1c: non-max suppression + top-N selection knobs.
// ===========================================================================
constexpr int kTopNFast   = 300;   // keypoints kept per image after FAST NMS
constexpr int kTopNHarris = 300;   // keypoints kept for the (detection-only) Harris comparison
constexpr int kMaxCandidates = 8192;  // device candidate-buffer capacity (generous headroom over kTopN*; a 256x256 image has at most 224x224 eligible interior pixels)

// ===========================================================================
// STAGE 2: ORB = oriented rBRIEF.
// ===========================================================================
constexpr int kOrientPatchRadius = 15;     // intensity-centroid AND rBRIEF sampling disk radius, px (matches OpenCV ORB's 31x31 = 2*15+1 patch)
constexpr int kOrbNumPairs   = 256;        // rBRIEF descriptor bit count (the ORB paper's default)
constexpr int kOrbDescWords  = kOrbNumPairs / 32;   // 8 x uint32 = 256 bits, packed LSB-first per word (see OrbDescriptor below)
constexpr uint32_t kOrbPatternSeed = 42u;  // xorshift32 seed for the BASE (unrotated) sampling pattern — CLAUDE.md's fixed-seed-42 convention

constexpr int   kOrientBins = 30;                                   // discrete orientation bins — matches OpenCV ORB's actual implementation (see header note above)
constexpr float kPi = 3.14159265358979323846f;
constexpr float kOrientBinWidthRad = (2.0f * kPi) / static_cast<float>(kOrientBins);  // 12 degrees/bin

// ===========================================================================
// STAGE 3: brute-force Hamming matcher.
// ===========================================================================
constexpr float kLoweRatio = 0.80f;  // Lowe's ratio test threshold on (best/second-best) distance.
// Binary Hamming distances are coarsely QUANTIZED (integers 0..256, typical
// good matches cluster in a narrow low band, e.g. 0-40 out of 256 bits) —
// looser than SIFT's classic 0.7-0.75 on continuous L2 distances, because a
// binary descriptor's distance histogram has less dynamic range to work
// with. 0.80 is the commonly-cited practical default for ORB/BRIEF-style
// matching (see THEORY.md "Where this sits in the real world" for the
// OpenCV/ORB-SLAM citation).

constexpr int kMaxHammingDist = 64;   // absolute cap on the BEST distance, in addition to the ratio test.
// Two independent bits per query is not enough: the ratio test alone can
// still accept a match whose best_dist is mediocre in absolute terms, as
// long as the second-best happens to be even worse (a small, sparse train
// set makes this more likely, not less — with few candidates, "runner-up"
// is often a poor match too). A truly-corresponding descriptor pair should
// differ in only a SMALL fraction of its 256 bits (real ORB-SLAM-style
// pipelines commonly cite ~50-64 bits, i.e. ~20-25%, as the practical
// ceiling for a confident match against 256-bit ORB descriptors — well
// under the 128-bit "coin flip" chance level). Both checks are ANDed into
// MatchResult::accepted in main.cu.

// ===========================================================================
// Shared data-layout structs (the "layout contract" — single-sourced).
// ===========================================================================

// A detected corner: integer pixel location + its detector score. `score`
// is stored as float uniformly (both detectors share this struct and the
// host-side sort/top-N code) — for FAST this is always an EXACT integer
// value (FAST scores are small ints, |score| << 2^24, so the float ->
// float round trip loses NO bits; IEEE-754 float32 represents every
// integer up to 2^24 exactly). This lets one sort/select routine serve
// both detectors without templates, while FAST's downstream comparisons
// stay bit-exact (see THEORY.md "Numerical considerations").
struct Keypoint {
    int   x;       // column, px, in [kBorder, kW - kBorder)
    int   y;       // row,    px, in [kBorder, kH - kBorder)
    float score;   // FAST: exact integer corner-strength margin. Harris: float response R.
};

// One (unrotated) rBRIEF sample-pair offset, relative to the keypoint.
// Generated once by build_orb_base_pattern() below; both dx/dy are within
// [-kOrientPatchRadius, kOrientPatchRadius] by construction (rejection
// sampling inside a DISK of that radius, not a square — an isotropic
// pattern, matching the spirit of ORB's learned-but-roughly-Gaussian
// pattern without claiming to reproduce OpenCV's actual trained table;
// see the header note above and THEORY.md for the honest comparison).
struct OrbPatternPair {
    int dx1, dy1;   // first sample point of the pair, px offset from keypoint
    int dx2, dy2;   // second sample point of the pair, px offset from keypoint
};

// A pattern pair AFTER being rotated by one of the kOrientBins discrete
// bin angles and rounded to the nearest integer pixel offset (see the
// header note above — this is exactly OpenCV ORB's precomputed-rotated-
// pattern strategy). One full table has kOrientBins * kOrbNumPairs entries.
struct RotatedOffset {
    int dx1, dy1;
    int dx2, dy2;
};

// A packed 256-bit rBRIEF descriptor: bit k (k = 0..255) lives in
// w[k / 32], bit position (k % 32), value 1 means "I(sample1) < I(sample2)"
// (the ORB paper's tau test), 0 otherwise. LSB-first within each word.
struct OrbDescriptor {
    uint32_t w[kOrbDescWords];
};

// One brute-force match result (query -> best train candidate), plus the
// bookkeeping main.cu needs to explain WHY it was accepted or rejected.
struct MatchResult {
    int  query_idx;     // index into the query keypoint/descriptor array
    int  train_idx;      // index into the train keypoint/descriptor array (best match)
    int  best_dist;      // Hamming distance to the best train match, bits (0..256)
    int  second_dist;    // Hamming distance to the second-best train match
    bool ratio_ok;        // best_dist <= kLoweRatio * second_dist
    bool cross_ok;         // train_idx's own best match (reverse direction) is query_idx
    bool accepted;          // ratio_ok && cross_ok — the final "is this a match" verdict
};

// ===========================================================================
// Shared HOST-ONLY helper functions — plain C++17, no CUDA syntax, so they
// compile identically under cl.exe (reference_cpu.cpp, main.cu) and nvcc
// (main.cu, kernels.cu's host-side table-build call). These build the
// single-sourced DATA (the RNG-derived base pattern, the rotated-table
// geometry, and the angle->bin quantization) that BOTH independently-
// written algorithmic cores (kernels.cu / reference_cpu.cpp) then consume
// identically — see this file's header for why that split is the honest
// one, not a shortcut.
// ===========================================================================

// ---------------------------------------------------------------------------
// xorshift32_next — Marsaglia's xorshift32 PRNG, one step.
//
// Why hand-roll instead of <random>'s std::uniform_real_distribution? Repo
// convention (CLAUDE.md machine facts / project brief): distributions'
// internal algorithms are IMPLEMENTATION-DEFINED per the C++ standard — the
// same seed can produce different sequences on different standard library
// implementations. xorshift32 is ~4 lines of pure unsigned-integer bit
// arithmetic with no implementation-defined behavior at all, so the exact
// same sequence comes out of MSVC's cl.exe, nvcc's host pass, AND (as used
// in scripts/make_synthetic.py) CPython — three different language
// runtimes, one deterministic sequence, given the same seed. That
// reproducibility is the whole point of a "seed 42" convention.
//
// Parameters: state — the generator's 32-bit state, updated in place.
//             MUST be seeded to a NON-ZERO value (xorshift's one dead
//             state: 0 maps to 0 forever).
// Returns: the next pseudo-random 32-bit value.
// ---------------------------------------------------------------------------
inline uint32_t xorshift32_next(uint32_t& state)
{
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// ---------------------------------------------------------------------------
// build_orb_base_pattern — generate the 256 UNROTATED rBRIEF sample-pair
// offsets deterministically from xorshift32(seed = kOrbPatternSeed).
//
// Each of the two points in a pair is drawn by rejection sampling: pick
// (dx, dy) uniformly in the bounding square [-R, R]^2, ACCEPT only if it
// lies inside the radius-R DISK (dx^2 + dy^2 <= R^2) — this keeps the
// pattern isotropic (no bias toward the square's corners, which are
// farther from the keypoint than the sampling radius implies). A pair is
// discarded and redrawn if its two points coincide (a same-point pair
// always compares equal to itself -> a dead, always-zero bit that wastes
// one of the 256 dimensions).
//
// Honesty note (see this file's header): real ORB (Rublee et al. 2011,
// as shipped in OpenCV) does NOT use a random pattern. It uses a FIXED,
// LEARNED 256-pair table chosen offline to (a) have pairwise-uncorrelated
// outcomes across a large training image corpus and (b) have high
// variance (each test should be a near-coin-flip, not always 0 or always
// 1). A seeded-random isotropic pattern is a fair, honest, non-fabricated
// didactic substitute: it demonstrates every mechanical step of ORB
// (rotate, round, sample, compare, pack) faithfully, but its match
// QUALITY on real imagery would be measurably worse than OpenCV's trained
// table. THEORY.md "Where this sits in the real world" says this plainly.
//
// Parameters: pattern — OUT: kOrbNumPairs entries, overwritten.
// Side effects: none beyond writing pattern. Complexity: O(kOrbNumPairs).
// ---------------------------------------------------------------------------
inline void build_orb_base_pattern(OrbPatternPair pattern[kOrbNumPairs])
{
    uint32_t rng = kOrbPatternSeed;
    const int R = kOrientPatchRadius;

    // draw_point_in_disk — rejection-sample one (dx,dy) inside the radius-R
    // disk. xorshift32_next returns a uint32; mapping it to a signed
    // integer in [-R, R] via modulo keeps everything in exact integer
    // arithmetic (no float, so this table is reproducible bit-for-bit
    // across platforms with no rounding concerns at all).
    auto draw_point_in_disk = [&](int& dx, int& dy) {
        for (;;) {
            const uint32_t rx = xorshift32_next(rng);
            const uint32_t ry = xorshift32_next(rng);
            dx = static_cast<int>(rx % static_cast<uint32_t>(2 * R + 1)) - R;  // in [-R, R]
            dy = static_cast<int>(ry % static_cast<uint32_t>(2 * R + 1)) - R;
            if (dx * dx + dy * dy <= R * R) return;   // inside the disk: accept
            // else: outside the disk (a corner of the bounding square) -> redraw
        }
    };

    for (int k = 0; k < kOrbNumPairs; ++k) {
        int dx1, dy1, dx2, dy2;
        for (;;) {
            draw_point_in_disk(dx1, dy1);
            draw_point_in_disk(dx2, dy2);
            if (dx1 != dx2 || dy1 != dy2) break;   // reject a degenerate (always-tied) pair
        }
        pattern[k] = OrbPatternPair{ dx1, dy1, dx2, dy2 };
    }
}

// ---------------------------------------------------------------------------
// build_rotated_pattern_table — for each of the kOrientBins discrete
// orientation bins, rotate every base pair by that bin's representative
// angle and round to the nearest integer pixel offset.
//
// Why round to INTEGER offsets instead of bilinear-sampling the rotated
// (generally fractional) location? Two reasons, both load-bearing:
//   1. It is what OpenCV's real ORB does (see this file's header) — the
//      production implementation already accepts this approximation.
//   2. It keeps the entire describe stage in INTEGER arithmetic: sampling
//      I[y][x] at an integer offset is an exact array read, so the
//      "descriptors BIT-EXACT" requirement (main.cu VERIFY) holds by
//      construction once the (already-quantized, see header note) bin
//      index agrees between the GPU and CPU pipelines — no float
//      comparison, no ULP risk, anywhere in the hot loop.
//
// Bin b's representative angle is simply b * kOrientBinWidthRad (bin 0 =
// 0 degrees, bin 1 = 12 degrees, ...) — this MUST match orient_to_bin()'s
// quantization rule below (both files agree on where bin boundaries fall).
// Rotation uses DOUBLE precision (this function runs ONCE at program
// startup, 30*256 = 7680 pairs total — negligible cost either way) purely
// so the table itself is built as precisely as the host can manage; the
// bit-exactness argument above does not actually depend on this choice
// (any reasonable precision rounds the same way here), but there is no
// reason to leave precision on the table when it is free.
//
// Parameters: base  — the 256 unrotated pairs (from build_orb_base_pattern).
//             table — OUT: kOrientBins x kOrbNumPairs entries, row-major
//                     (table[bin * kOrbNumPairs + k]), overwritten.
// Side effects: none beyond writing table. Complexity: O(kOrientBins * kOrbNumPairs).
// ---------------------------------------------------------------------------
inline void build_rotated_pattern_table(const OrbPatternPair base[kOrbNumPairs],
                                        RotatedOffset table[kOrientBins * kOrbNumPairs])
{
    for (int b = 0; b < kOrientBins; ++b) {
        const double angle = static_cast<double>(b) * static_cast<double>(kOrientBinWidthRad);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        for (int k = 0; k < kOrbNumPairs; ++k) {
            const OrbPatternPair& p = base[k];
            // Standard 2-D rotation matrix [c -s; s c] applied to each of
            // the pair's two offset vectors, then rounded to nearest int
            // (std::lround: round-half-away-from-zero, deterministic).
            const int rx1 = static_cast<int>(std::lround(c * p.dx1 - s * p.dy1));
            const int ry1 = static_cast<int>(std::lround(s * p.dx1 + c * p.dy1));
            const int rx2 = static_cast<int>(std::lround(c * p.dx2 - s * p.dy2));
            const int ry2 = static_cast<int>(std::lround(s * p.dx2 + c * p.dy2));
            table[b * kOrbNumPairs + k] = RotatedOffset{ rx1, ry1, rx2, ry2 };
        }
    }
}

// ---------------------------------------------------------------------------
// orient_to_bin — quantize a continuous orientation angle (radians, any
// range — atan2f's [-pi, pi] included) into a discrete bin index in
// [0, kOrientBins).
//
// Must agree EXACTLY with build_rotated_pattern_table()'s bin->angle
// convention (bin b <-> angle b * kOrientBinWidthRad, i.e. bins cover
// [0, 2*pi) starting at 0): first wrap theta into [0, 2*pi), then divide
// by the bin width and round to the NEAREST bin, then wrap the bin index
// itself modulo kOrientBins (rounding near theta ~ 2*pi can produce bin
// index kOrientBins, which is really bin 0).
//
// Parameters: theta_rad — orientation angle, radians, any finite value.
// Returns: bin index in [0, kOrientBins).
// ---------------------------------------------------------------------------
inline int orient_to_bin(float theta_rad)
{
    float wrapped = std::fmod(theta_rad, 2.0f * kPi);
    if (wrapped < 0.0f) wrapped += 2.0f * kPi;               // now in [0, 2*pi)
    int bin = static_cast<int>(std::lround(wrapped / kOrientBinWidthRad));
    bin %= kOrientBins;                                       // fold a round-up-to-2pi back to bin 0
    if (bin < 0) bin += kOrientBins;                          // defensive; lround/% never actually go negative here
    return bin;
}

// ---------------------------------------------------------------------------
// popcount32_portable — Hamming weight (number of set bits) of a 32-bit
// word, the classic SWAR ("SIMD within a register") bit-trick, no
// intrinsics, no vectorization — exactly the "what would it take to write
// it by hand" answer the repo's no-black-boxes rule (CLAUDE.md §1) asks
// for kernels.cu's use of the hardware __popc() instruction. Used by
// reference_cpu.cpp's Hamming-distance twin; kernels.cu instead calls
// __popc() directly and documents this SAME algorithm as "what the silicon
// is believed to implement" in a comment there.
//
// How it works, in three folds: pair up bits and sum each pair (2-bit
// partial counts) -> pair up nibbles and sum (4-bit partial counts) ->
// pair up bytes and sum (8-bit partial counts) -> the final multiply-and-
// shift horizontally sums the four byte-counts into the top byte in one
// step (a classic trick: multiplying by 0x01010101 adds a value to
// itself shifted by 0, 8, 16, 24 bits, i.e. sums the four byte lanes).
// ---------------------------------------------------------------------------
inline int popcount32_portable(uint32_t v)
{
    v = v - ((v >> 1) & 0x55555555u);                          // every 2 bits -> count of set bits in that pair
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);          // every 4 bits -> count of set bits in that nibble
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;                          // every 8 bits -> count of set bits in that byte
    return static_cast<int>((v * 0x01010101u) >> 24);           // horizontal sum of the 4 byte-lanes, in the top byte
}

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// ===========================================================================
// STAGE 1 kernels — DETECT.
// ===========================================================================

// fast_score_kernel — per-pixel FAST-9 corner score (map/stencil hybrid:
// each thread reads a fixed 16-pixel ring, entirely independent of every
// other thread — no shared memory, no cross-thread communication).
// img: [kW*kH] device image, row-major uint8 grayscale.
// score_out: [kW*kH] device OUT — 0 for non-corners and border pixels
//            (|x-kW/2|... within kDetectBorder of any edge), otherwise the
//            corner-strength margin (>0, integer intensity units).
// Full documentation (the contiguous-arc test, the high-speed reject, the
// score definition) sits with the definition in kernels.cu.
__global__ void fast_score_kernel(const uint8_t* __restrict__ img, int* __restrict__ score_out);

// sobel_gradient_kernel — per-pixel 3x3 Sobel Gx, Gy (a stencil: each
// thread reads its 3x3 neighborhood). Border pixels (within kSobelBorder —
// NOT kDetectBorder; see kSobelBorder's comment for why they differ) get 0.
// gx_out/gy_out: [kW*kH] device OUT, float (exact integers here — see
// kernels.cu's numerics note).
__global__ void sobel_gradient_kernel(const uint8_t* __restrict__ img,
                                      float* __restrict__ gx_out,
                                      float* __restrict__ gy_out);

// harris_response_kernel — per-pixel structure-tensor response R = det(M)
// - k*trace(M)^2, M summed over a (2*kHarrisWinRadius+1)^2 box window of
// (Gx^2, Gy^2, Gx*Gy). gx/gy: [kW*kH] device IN (from sobel_gradient_kernel).
// response_out: [kW*kH] device OUT, float.
__global__ void harris_response_kernel(const float* __restrict__ gx,
                                       const float* __restrict__ gy,
                                       float* __restrict__ response_out);

// nms_select_fast_kernel — 3x3 non-max suppression + atomic compaction over
// an INTEGER score map (FAST). A candidate pixel (x,y) in the eligible
// region [kBorder, kW-kBorder) x [kBorder, kH-kBorder) is kept iff its
// score is > 0 AND strictly greater than all 8 immediate neighbors' scores
// (a tie with a neighbor suppresses BOTH — see kernels.cu for why that is
// the deterministic, twin-reproducible choice). Kept candidates are
// atomically appended (unordered — main.cu sorts afterward) to
// out_x/out_y/out_score, up to max_candidates; *counter is the running
// (possibly over-capacity) count.
// score: [kW*kH] device IN. out_x/out_y/out_score: [max_candidates] device
// OUT (only the first min(*counter, max_candidates) entries are valid).
// counter: single device int, MUST be zeroed by the caller before launch.
__global__ void nms_select_fast_kernel(const int* __restrict__ score,
                                       int* __restrict__ out_x,
                                       int* __restrict__ out_y,
                                       int* __restrict__ out_score,
                                       int* __restrict__ counter,
                                       int max_candidates);

// nms_select_harris_kernel — the FLOAT twin of the kernel above, over the
// Harris response map. thresh: minimum response to be considered at all
// (pre-NMS floor, filters flat/edge-only regions where R <= 0 or is
// negligibly small). Same output contract as nms_select_fast_kernel.
__global__ void nms_select_harris_kernel(const float* __restrict__ response,
                                         float thresh,
                                         int* __restrict__ out_x,
                                         int* __restrict__ out_y,
                                         float* __restrict__ out_score,
                                         int* __restrict__ counter,
                                         int max_candidates);

// ===========================================================================
// STAGE 2 kernels — DESCRIBE.
// ===========================================================================

// orientation_kernel — one thread per keypoint: intensity centroid over a
// radius-kOrientPatchRadius disk, theta = atan2(m01, m10).
// img: [kW*kH] device IN. kp_x/kp_y: [n] device IN (keypoint pixel coords,
// each already >= kBorder from every edge — the caller's contract).
// theta_out: [n] device OUT, radians, atan2f's native [-pi, pi] range.
__global__ void orientation_kernel(const uint8_t* __restrict__ img,
                                   const int* __restrict__ kp_x,
                                   const int* __restrict__ kp_y,
                                   int n,
                                   float* __restrict__ theta_out);

// describe_kernel — one thread per keypoint: 256 rotated intensity-pair
// comparisons -> one packed 256-bit descriptor (see this file's header for
// the full bit-exactness argument).
// img: [kW*kH] device IN. kp_x/kp_y: [n] device IN. bin_idx: [n] device IN
// (already-quantized orientation bins, see orient_to_bin()). table:
// [kOrientBins*kOrbNumPairs] device IN (from build_rotated_pattern_table(),
// uploaded once). desc_out: [n] device OUT.
__global__ void describe_kernel(const uint8_t* __restrict__ img,
                                const int* __restrict__ kp_x,
                                const int* __restrict__ kp_y,
                                const int* __restrict__ bin_idx,
                                int n,
                                const RotatedOffset* __restrict__ table,
                                OrbDescriptor* __restrict__ desc_out);

// ===========================================================================
// STAGE 3 kernel — MATCH.
// ===========================================================================

// hamming_match_kernel — one thread per QUERY descriptor: brute-force scan
// of every TRAIN descriptor, popcount(query XOR train) via __popc, keep
// running best + second-best (Hamming distance, train index).
// query: [nQuery] device IN. train: [nTrain] device IN.
// best1_dist/best1_idx/best2_dist/best2_idx: [nQuery] device OUT.
__global__ void hamming_match_kernel(const OrbDescriptor* __restrict__ query, int nQuery,
                                     const OrbDescriptor* __restrict__ train, int nTrain,
                                     int* __restrict__ best1_dist, int* __restrict__ best1_idx,
                                     int* __restrict__ best2_dist, int* __restrict__ best2_idx);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// Host-callable LAUNCH WRAPPERS — own the grid/block math + post-launch
// error check (CLAUDE.md §6.1 rule 7), visible to any translation unit
// (only their DEFINITIONS, in kernels.cu, need nvcc).
// ===========================================================================
void launch_fast_score(const uint8_t* d_img, int* d_score);
void launch_sobel_gradient(const uint8_t* d_img, float* d_gx, float* d_gy);
void launch_harris_response(const float* d_gx, const float* d_gy, float* d_response);

// launch_nms_select_fast/harris — zero the device counter, launch NMS,
// download and return the candidate count actually written (capped at
// max_candidates; if more candidates existed, main.cu prints a note — see
// kernels.cu). d_out_* are device buffers of length >= max_candidates the
// caller allocated.
int launch_nms_select_fast(const int* d_score, int* d_out_x, int* d_out_y, int* d_out_score, int max_candidates);
int launch_nms_select_harris(const float* d_response, float thresh, int* d_out_x, int* d_out_y, float* d_out_score, int max_candidates);

void launch_orientation(const uint8_t* d_img, const int* d_kp_x, const int* d_kp_y, int n, float* d_theta);
void launch_describe(const uint8_t* d_img, const int* d_kp_x, const int* d_kp_y, const int* d_bin_idx,
                     int n, const RotatedOffset* d_table, OrbDescriptor* d_desc);
void launch_hamming_match(const OrbDescriptor* d_query, int nQuery, const OrbDescriptor* d_train, int nTrain,
                          int* d_best1_dist, int* d_best1_idx, int* d_best2_dist, int* d_best2_idx);

// ===========================================================================
// CPU reference (oracle) declarations — defined in reference_cpu.cpp.
// Declared here so the compiler enforces signature agreement with what
// main.cu calls, exactly like the SAXPY placeholder did for saxpy_cpu.
// Each ALGORITHMICALLY mirrors (independently — see reference_cpu.cpp's
// header) the GPU kernel of the same concept, but as a single-threaded
// host loop over plain arrays, never a CUDA type.
// ===========================================================================
void fast_score_cpu(const uint8_t* img, int* score_out);
int  fast_nms_select_cpu(const int* score, Keypoint* out, int max_out);   // returns count (<= max_out), SORTED (score desc, y asc, x asc), matching the GPU path's post-sort convention

void sobel_gradient_cpu(const uint8_t* img, float* gx_out, float* gy_out);
void harris_response_cpu(const float* gx, const float* gy, float* response_out);

void orientation_cpu(const uint8_t* img, const Keypoint* kps, int n, float* theta_out);
void describe_cpu(const uint8_t* img, const Keypoint* kps, const int* bin_idx, int n,
                  const RotatedOffset* table, OrbDescriptor* desc_out);

void hamming_match_cpu(const OrbDescriptor* query, int nQuery, const OrbDescriptor* train, int nTrain,
                       int* best1_dist, int* best1_idx, int* best2_dist, int* best2_idx);

#endif // PROJECT_KERNELS_CUH
