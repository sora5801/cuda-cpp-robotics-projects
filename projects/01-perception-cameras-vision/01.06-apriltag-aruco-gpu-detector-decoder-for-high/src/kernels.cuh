// ===========================================================================
// kernels.cuh — interface for project 01.06
//               AprilTag / ArUco GPU detector-decoder for high-rate fiducial
//               localization: adaptive threshold -> connected-component
//               labeling -> quad extraction -> DLT homography -> perspective
//               grid decode -> homography-based pose, on a home-grown 32-code
//               6x6 (4x4-payload) fiducial dictionary.
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (driver + gates), kernels.cu (the GPU
// pipeline), and reference_cpu.cpp (the independent CPU oracle). Every
// layout, camera constant, dictionary geometry fact, and sentinel all three
// must agree on lives HERE, once (CLAUDE.md paragraph 12). Style follows
// flagship 01.01 (camera-model single-source-of-truth, HD shared helpers,
// "MUST MATCH ../scripts/make_synthetic.py" cross-references) and reuses the
// iterative label-propagation connected-component pattern taught in 30.01
// (cite: projects/30-field-robotics/30.01-agriculture/src/kernels.cu, stage
// 4) for the dark-blob-finding stage below.
//
// THE PIPELINE, SEVEN STAGES (THEORY.md derives every step in depth):
//   1. ADAPTIVE THRESHOLD    — a separable box filter computes each pixel's
//      LOCAL mean brightness (a "background estimate"); a pixel is FOREGROUND
//      (dark / candidate tag ink) if it reads more than kThreshBiasC below
//      its own local mean. A single GLOBAL threshold would fail the instant
//      the scene has an illumination gradient (this project's synthetic
//      scenes always have one) — see THEORY.md "The problem".
//   2. CONNECTED-COMPONENT LABELING — iterative 4-connected label propagation
//      over the dark mask (identical algorithm and convergence argument to
//      30.01's stage 4; re-derived independently here for THIS project's
//      pixel layout, not included/shared as code).
//   3. QUAD EXTRACTION — per candidate component (a small, CANDIDATE-PARALLEL
//      stage, contrasted with stages 1/2/4's PIXEL-PARALLEL launches — see
//      "The GPU mapping" in THEORY.md): (a) an "extreme-corner" pass finds
//      the 4 pixels that maximize/minimize (x+y) and (x-y) via a packed
//      64-bit atomicMin/Max trick (see pack_corner_key() below) — a cheap
//      teaching approximation to a convex hull; (b) a radial sub-pixel edge
//      search refines each corner along the ray from the component centroid.
//      HONESTLY WEAKER than production (README "Limitations & honesty" and
//      THEORY.md name exactly where): a real AprilTag detector clusters
//      gradient orientations and fits FOUR LINES, then intersects them for a
//      corner that is robust to any component shape and any in-plane
//      rotation; extreme-corner picking degrades whenever a flat edge (not a
//      vertex) is the true extremum of x+y/x-y — which happens for in-plane
//      rotations near +/-45 degrees. This project's synthetic scene therefore
//      deliberately avoids that rotation band (see make_synthetic.py) and
//      says so.
//   4. DLT HOMOGRAPHY — 4 point correspondences (tag-model corners, meters,
//      known by construction <-> the 4 refined image corners) give exactly 8
//      linear equations in 8 unknowns (h33 fixed to 1); one thread solves an
//      8x8 system by Gaussian elimination with partial pivoting, in DOUBLE
//      precision (small-dense-linear-solve-per-thread teaching pattern —
//      cite 33.01, projects/33-foundational-libraries/33.01-*, "batched
//      small-matrix linalg" for the general technique this project applies
//      to homographies instead of robot-arm Jacobians).
//   5. PERSPECTIVE GRID SAMPLING + DECODING — warp-sample the tag's 6x6 cell
//      centers through H, threshold each against the SAME local-mean field
//      used for detection (single source of truth for "what counts as
//      black"), require the 20 border cells all read black (a hard reject —
//      real dictionaries are built the same way), then try the sampled 4x4
//      payload against the dictionary at all 4 in-plane rotations and accept
//      the closest entry within the dictionary's correction capacity.
//   6. POSE FROM HOMOGRAPHY — classical K^-1*H column-normalization
//      decomposition (THEORY.md derives it); production systems refine this
//      with IPPE (Infinitesimal Plane-based Pose Estimation) — named, not
//      implemented, in README "Prior art".
//
// IMAGE LAYOUT — row-major, pixel (x, y) at linear index i = y*W + x, x
// rightward, y downward (matches every PGM in this repo). Camera uses the
// OPTICAL convention (x-right, y-down, z-forward), the same stated exception
// as 01.01/30.01 (SYSTEM_DESIGN.md section 3.2).
//   Gray image      : unsigned char, [H*W], 0..255.
//   Local mean field: float, [H*W], the box-filtered background estimate
//                     used BOTH by detection (stage 1) and decoding (stage 5)
//                     — one field, two consumers, so "what is black" never
//                     drifts between finding a tag and reading it.
//   Mask            : unsigned char, [H*W], 1 = darker than local mean by
//                     more than kThreshBiasC, else 0.
//   Label           : int, [H*W]. kLabelNone = background/never touched by
//                     CCL. A foreground pixel p starts at label[p] = p and
//                     converges to the SMALLEST linear index reachable via
//                     4-connected foreground neighbors (identical convergence
//                     argument to 30.01's kernels.cu — re-derived in this
//                     project's THEORY.md "The algorithm").
//
// TWIN-INDEPENDENCE (per reference_cpu.cpp's file-header ruling): the layout
// constants, structs, and small INDEXING helpers below (bit <-> (r,c),
// 90-degree rotation permutation, corner-key packing) are shared HD code —
// pure data-layout bookkeeping, not "the algorithm under test". The box
// filter, CCL, quad extraction, DLT solve, grid decode, and pose
// decomposition are each typed TWICE — independently — in kernels.cu (GPU)
// and reference_cpu.cpp (CPU). main.cu additionally carries gates that do
// NOT route through this shared bookkeeping at all: the pose gate compares
// against the ANALYTIC ground-truth camera pose recorded by
// scripts/make_synthetic.py (an independent, third implementation of the
// projection math, in Python), and the decode-robustness gate exercises the
// dictionary's Hamming-distance arithmetic directly against known,
// hand-computed flip counts.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe (same trick as
// 01.01's compute_source_pixel / bayer_channel_at). Lets a handful of pure
// data-layout formulas — bit<->(r,c) indexing, the 90-degree rotation
// permutation, corner-key packing — live in exactly one place and be called
// identically from kernels.cu (device) and reference_cpu.cpp (host, compiled
// by cl.exe, which never sees a CUDA header).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// Geometry & camera constants — MUST MATCH ../scripts/make_synthetic.py's
// "MUST MATCH kernels.cuh" block. Single source of truth for image size,
// pinhole intrinsics, and tag physical size (CLAUDE.md paragraph 12).
// ===========================================================================
constexpr int kFullW = 480;   // scene width, px  (all three committed scenes share this size)
constexpr int kFullH = 360;   // scene height, px

// Pinhole intrinsics (see 01.01's identical convention: fx=fy, principal
// point at the exact image center using the "-0.5" pixel-center rule so
// (kFullW-1)/2 lands on the true center pixel). fx=fy=350 px keeps the
// projected tag side in a 55-125 px range across the depth range used by
// scripts/make_synthetic.py (documented there) — big enough to resolve a
// 6x6 grid with several px per cell, small enough that 6 tags fit one frame.
constexpr float kFx = 350.0f;
constexpr float kFy = 350.0f;
constexpr float kCx = (kFullW - 1) * 0.5f;   // = 239.5
constexpr float kCy = (kFullH - 1) * 0.5f;   // = 179.5

constexpr float kTagSizeM = 0.16f;           // outer (border-to-border) tag side, meters
constexpr float kTagHalfM = kTagSizeM * 0.5f;

// ===========================================================================
// Dictionary geometry — a home-grown 32-code family, 6x6 total grid (1-cell
// border ring + 4x4 = 16-bit payload), the same shape as real families this
// project studies toward (README "Prior art"): AprilTag 16h5 (16-bit payload,
// min Hamming distance 5) and ArUco DICT_4X4_50 (4x4 payload + border) are
// both built on exactly this geometry — this project's dictionary is an
// independently GENERATED set of codes (never their published bit tables),
// documented honestly in THEORY.md "The math" / "Where this sits in the real
// world".
// ===========================================================================
constexpr int kGridN       = 6;     // total grid is kGridN x kGridN cells
constexpr int kPayloadN    = 4;     // interior payload is kPayloadN x kPayloadN cells
constexpr int kPayloadBits = kPayloadN * kPayloadN;               // 16
constexpr int kBorderCells = kGridN * kGridN - kPayloadBits;      // 36 - 16 = 20
constexpr int kNumDictCodes = 32;   // committed dictionary size (data/sample/dictionary.bin)

// A cell (r, c) with r, c in [0, kGridN) is BORDER (must render black, and at
// decode time must SAMPLE black, or the candidate is rejected outright) iff
// it touches the outer ring; otherwise it is one of the 16 payload cells.
HD inline bool is_border_cell(int r, int c)
{
    return r == 0 || r == kGridN - 1 || c == 0 || c == kGridN - 1;
}

// payload_bit_index — maps a PAYLOAD cell's grid coordinates (r, c), each in
// [1, kGridN-2] = [1,4], to its bit index k in [0, kPayloadBits). Row-major
// within the 4x4 interior: k = (r-1)*4 + (c-1). This one formula is the
// single source of truth for "which bit is which cell" — shared by the GPU
// grid-sampling kernel, the CPU twin, AND scripts/make_synthetic.py's tag
// renderer (reimplemented there identically in Python; a "MUST MATCH"
// comment marks the mirror).
// ---------------------------------------------------------------------------
HD inline int payload_bit_index(int r, int c)
{
    return (r - 1) * kPayloadN + (c - 1);
}

// rotate_payload_90 — the bit permutation a 4x4 payload undergoes when the
// PHYSICAL tag is rotated 90 degrees clockwise in the image plane before the
// camera reads it. Derived by applying the standard image-rotation index map
// (r,c) -> (c, N-1-r) to every payload cell and reading off where each old
// bit lands. Composed 2x/3x by the caller for 180/270 degrees. This exact
// permutation is what the dictionary GENERATOR uses to compute a code's 4
// rotation variants (for the minimum-Hamming-distance search) and what the
// DECODER uses to try a sampled bit pattern against the dictionary at every
// plausible mounting orientation — both must agree bit-for-bit, which is
// exactly why this is ONE shared function, not two independently-typed ones
// (a data-layout fact, per the twin-independence ruling in
// reference_cpu.cpp's header — not "the algorithm under test").
// ---------------------------------------------------------------------------
HD inline uint16_t rotate_payload_90(uint16_t code)
{
    uint16_t out = 0;
    // Loop in GRID coordinates [1, kPayloadN] (the domain payload_bit_index
    // actually expects — see its own doc comment just above). An earlier
    // draft of this function looped r,c over [0, kPayloadN) — PAYLOAD-LOCAL
    // coordinates — and fed them straight into payload_bit_index(), which
    // silently computed NEGATIVE bit indices and shifted by them (undefined
    // behavior). Converting explicitly between the two coordinate systems
    // (grid = payload-local + 1) at each use is the fix, and is left visible
    // here on purpose as a worked example of the bug class (CLAUDE.md
    // paragraph 6: narrate the thought process, including the one that
    // failed) — see THEORY.md "Numerical considerations" for the full story.
    for (int r = 1; r <= kPayloadN; ++r) {
        for (int c = 1; c <= kPayloadN; ++c) {
            const int old_bit = payload_bit_index(r, c);     // grid coords -> bit index, in [0,15]
            if ((code >> old_bit) & 1u) {
                // Rotate in PAYLOAD-LOCAL coordinates (0-based): (pr,pc) ->
                // (pc, 3-pr) is the standard clockwise-90 image-index map.
                const int pr = r - 1, pc = c - 1;
                const int npr = pc, npc = kPayloadN - 1 - pr;
                out = static_cast<uint16_t>(out | (1u << payload_bit_index(npr + 1, npc + 1)));
            }
        }
    }
    return out;
}

// popcount16 — number of set bits in a 16-bit payload; used both to compute
// Hamming distance (popcount(a^b)) and to detect the two DEGENERATE payloads
// (all-black 0xFFFF, all-white 0x0000) this project deliberately refuses to
// accept even if a dictionary entry were coincidentally close to one (see
// kernels.cu's grid-decode kernel and README "Limitations & honesty" — a
// documented false-positive safeguard, the same spirit as real dictionaries
// never including the all-one/all-zero code).
// ---------------------------------------------------------------------------
HD inline int popcount16(uint16_t v)
{
    int n = 0;
    while (v) { n += (v & 1u); v >>= 1u; }
    return n;
}

// ===========================================================================
// Adaptive threshold & component-filter constants (THEORY.md "The algorithm"
// derives each from the scene's own geometry — not arbitrary):
//   kBoxRadius   — local-mean window half-size. Window side = 2*kBoxRadius+1
//     = 25 px must span several payload cells (cells are ~9-20px depending on
//     tag distance) so the local mean reflects real black/white content, not
//     a window sitting entirely inside one cell.
//   kThreshBiasC — a pixel is foreground if it reads more than this many
//     intensity levels (0..255 scale) below its local mean — the standard
//     "mean minus C" adaptive-threshold rule (THEORY.md derives why C>0 is
//     needed: without it, sensor noise alone flips ~50% of uniform-gray
//     pixels to foreground).
// ---------------------------------------------------------------------------
constexpr int   kBoxRadius   = 12;
constexpr float kThreshBiasC = 6.0f;

// Component filters (host-side, after GPU stats download): reject components
// too small (noise speckle), too large (a merged blob / background texture),
// or with a fill ratio (pixel_count / bbox area) outside a tag-ring-like
// range. THEORY.md "The algorithm" derives the fill-ratio band from the
// grid's own geometry: a bare border ring alone covers 1 - (4/6)^2 = 5/9 ~=
// 0.56 of its bounding box; with some black payload cells mixed in the
// observed ratio commonly reaches 0.6-0.9; a solid disk/blob (this project's
// false-positive distractor) fills close to 0.7-1.0 too, which is why the
// STRICT border-ring bit check in stage 5 — not the fill ratio alone — is
// this project's primary false-positive defense (named honestly in THEORY).
constexpr int   kMinComponentPixels = 250;
constexpr int   kMaxComponentPixels = 30000;
constexpr float kMinFillRatio       = 0.30f;
constexpr float kMaxFillRatio       = 0.98f;
constexpr int   kMinBBoxSidePx      = 24;
constexpr int   kMaxBBoxSidePx      = 220;

// Border-ring tolerance at decode time (kernels.cu's grid_decode_kernel /
// reference_cpu.cpp's grid_decode_one_cpu): a candidate is accepted only if
// AT MOST kMaxBorderErrors of the kBorderCells=20 sampled border cells read
// light instead of black. A STRICT all-20-must-be-black rule was this
// project's first design and measurably too fragile in practice: this
// project's own extreme-corner quad extraction (kernels.cuh's file header
// names the honest weakness) routinely lands the fitted homography a few
// pixels off true, which the corner cells nearest the tag's own corners
// (the LEAST constrained points of the fit, geometrically) feel the most —
// on the committed main scene, legitimately-detected tags measured 3-8 of
// 20 border cells reading wrong under the strict rule, yet their PAYLOAD
// still matched the correct dictionary code within 0-2 bits (comfortably
// inside the correction capacity) every time, because payload cells sit
// CLOSER to the quad's center and are proportionally less disturbed by the
// same homography error. kMaxBorderErrors=9 (up to 45% of the ring) keeps
// this a real, if soft, gate — pure clutter (checkerboard squares, filled
// disks; see the false-positive gate) still needs to coincidentally clear
// this AND the degenerate-payload safeguard AND land within the payload's
// own Hamming ball, which the committed distractor scene never does
// (measured: README "Expected output"). See THEORY.md "Numerical
// considerations" for the full derivation and the honest tradeoff named.
constexpr int kMaxBorderErrors = 9;

constexpr int kLabelNone     = -1;    // CCL sentinel: background / never labeled
constexpr int kMaxCclSweeps  = 512;   // safety cap (30.01 precedent); real scenes converge far sooner
constexpr int kMaxCandidates = 48;    // fixed cap on candidate components per scene (buffer sizing)

// ===========================================================================
// Corner-key packing — the "argmax via packed atomic" trick this project
// uses to find the 4 extreme pixels of a connected component in ONE pixel-
// parallel pass (kernels.cu's component-stats kernel), without a second
// reduction pass. A plain atomicMax/Min on the SCORE alone would tell you the
// extreme VALUE but not WHICH PIXEL achieved it; packing (score, pixel index)
// into one 64-bit integer, with the score in the HIGH bits, makes ordinary
// integer atomicMin/Max on the PACKED value simultaneously an arg-extremum:
// whichever thread holds the true extreme score wins the compare regardless
// of index, and ties break toward the smaller (atomicMin) or larger
// (atomicMax) pixel index — deterministic, if arbitrary, tie-breaking.
//
// kCornerScoreOffset shifts (x+y) and (x-y) — the latter can be NEGATIVE — up
// into a uniformly non-negative range before packing, since the packed key's
// ordering must match the score's ordering bit-for-bit (a two's-complement
// negative score packed into the high bits of an UNSIGNED 64-bit key would
// sort incorrectly). kFullW+kFullH comfortably covers both scores' full
// range with headroom.
// ---------------------------------------------------------------------------
constexpr long long kCornerScoreOffset = kFullW + kFullH;   // = 840

HD inline unsigned long long pack_corner_key(long long score, int pixel_index)
{
    // Shifted score occupies the high 32 bits, pixel index the low 32 bits.
    // Both are small (score+offset < 2000; pixel_index < kFullW*kFullH =
    // 172800), so there is no risk of the two fields colliding.
    const unsigned long long shifted = static_cast<unsigned long long>(score + kCornerScoreOffset);
    return (shifted << 32) | static_cast<unsigned long long>(static_cast<unsigned int>(pixel_index));
}
HD inline int unpack_corner_index(unsigned long long key)
{
    return static_cast<int>(key & 0xFFFFFFFFull);
}

// ===========================================================================
// CandidateComponent — one connected dark component that survived the
// host-side size/fill-ratio/bbox filter, uploaded to the device as a small
// (<= kMaxCandidates) array for the CANDIDATE-PARALLEL stages 3-6. Built on
// the HOST from the pixel-parallel stats kernel's dense [H*W] output arrays
// (same "GPU does O(H*W) pixel work, host does tiny O(#components)
// bookkeeping" split as 30.01 and 08.01).
// ---------------------------------------------------------------------------
struct CandidateComponent {
    int   label;              // canonical label = this component's root pixel index
    int   pixel_count;        // foreground pixel count (post-CCL, pre-quad-fit)
    float centroid_x, centroid_y;             // px, component centroid (sum_x/count, sum_y/count)
    int   bbox_min_x, bbox_max_x;             // inclusive px bounding box
    int   bbox_min_y, bbox_max_y;
    // Raw extreme-corner pixel coordinates, ORDER FIXED here and relied on
    // by every downstream stage: index 0 = argmin(x+y) ("top-left-ish"),
    // 1 = argmax(x-y) ("top-right-ish"), 2 = argmax(x+y) ("bottom-right-ish"),
    // 3 = argmin(x-y) ("bottom-left-ish"). This order need NOT match the
    // tag's true in-plane orientation — stage 5's 4-way rotation trial
    // against the dictionary absorbs whatever 90-degree offset this
    // arbitrary-but-fixed assignment introduces (see kernels.cuh's file
    // header "QUAD EXTRACTION").
    float raw_corner_x[4], raw_corner_y[4];
};

// QuadCorners — stage 3's refined output: the same 4 corners, sub-pixel, in
// the SAME fixed order as CandidateComponent's raw_corner_*, after the
// radial edge search. `valid` is false when the search failed to find a
// clean dark->light crossing (e.g. a corner ray that never leaves the dark
// mask within the search range) — such candidates are dropped before the
// homography stage rather than fed a degenerate quad.
struct QuadCorners {
    float x[4], y[4];
    bool  valid;
};

// Homography — 3x3, row-major, TAG-METERS -> PIXEL homogeneous mapping
// (h[8] IS the free h33 unknown's value AFTER the DLT solve fixes it to 1.0
// by construction — kept explicit here, rather than assumed, so a reader
// never has to remember an implicit convention): pixel_homog = H * (X, Y, 1)
// for a tag-plane point (X, Y, 0) in meters. `valid` is false when the 8x8
// DLT system was singular/ill-conditioned (near-zero pivot during
// elimination — THEORY.md "Numerical considerations").
struct Homography {
    double h[9];
    bool   valid;
};

// Detection — the pipeline's final per-candidate output record. Both the GPU
// path (main.cu, assembled from device arrays) and the CPU oracle
// (reference_cpu.cpp) fill this exact struct so main.cu's verification and
// gates compare like-for-like.
struct Detection {
    int   candidate_index;    // index into the CandidateComponent array this came from
    bool  border_ok;          // at most kMaxBorderErrors of the kBorderCells sampled border
                               // cells read light instead of black (a TOLERANT check — see
                               // kMaxBorderErrors' doc comment for why not a strict AND)
    bool  accepted;           // border_ok && min Hamming distance <= correction capacity
    int   tag_id;             // dictionary index [0,kNumDictCodes) if accepted, else -1
    int   rotation;           // 0..3 quarter-turns that matched, if accepted
    int   hamming_distance;   // to the matched code (or the best code, even if rejected — reported)
    float corners_x[4], corners_y[4];   // refined image corners, same fixed order as QuadCorners
    bool  pose_valid;         // true if homography decomposition succeeded (t_z > 0, r3 orthonormal)
    float R[9];               // row-major 3x3, tag-frame axes expressed in the camera frame
    float t[3];                // meters, tag origin in the camera frame
};

// ===========================================================================
// GPU launch wrappers (kernels.cu). Every wrapper computes its own launch
// geometry, launches, and calls CUDA_CHECK_LAST_ERROR — main.cu never writes
// <<<...>>> syntax directly (same discipline as every flagship in this repo).
// ===========================================================================

// ---- Stage 1: adaptive threshold (PIXEL-parallel; two-pass separable box
// filter, then a threshold map) ---------------------------------------------
// d_gray        : [H*W] uint8 IN, the (blurred, noisy) synthetic scene.
// d_row_sum     : [H*W] float OUT/scratch — horizontal box SUM (not yet
//                 divided by area; the vertical pass finishes the box).
// d_local_mean  : [H*W] float OUT — the finished local mean (background
//                 estimate), reused unmodified by stage 5's decoder.
// d_mask        : [H*W] uint8 OUT — 1 = foreground (candidate tag ink).
void launch_box_sum_h(const unsigned char* d_gray, float* d_row_sum, int W, int H);
void launch_box_sum_v(const float* d_row_sum, float* d_local_mean, int W, int H);
void launch_adaptive_threshold(const unsigned char* d_gray, const float* d_local_mean,
                               unsigned char* d_mask, int W, int H);

// ---- Stage 2: connected-component labeling (PIXEL-parallel, iterative;
// identical algorithm to 30.01's stage 4, re-typed independently here) -----
void launch_ccl_init(const unsigned char* d_mask, int* d_label, int W, int H);
void launch_ccl_propagate_sweep(const unsigned char* d_mask, int* d_label, int W, int H, int* d_changed);

// ---- Component statistics (PIXEL-parallel scatter, feeds the host-side
// candidate filter that produces the small CandidateComponent array) -------
// Dense [H*W]-indexed accumulator arrays, same "index directly by canonical
// label, only <= a few dozen slots ever written" design as 30.01. The four
// d_key_* arrays hold PACKED (score, pixel-index) keys — see
// pack_corner_key() above — one per extremum direction.
// NOTE: sum_x/sum_y accumulate PIXEL COORDINATES (always >= 0), so they are
// declared unsigned long long — atomicAdd has a native unsigned-long-long
// overload, and there is no signedness ambiguity to reason about at the call
// site (CLAUDE.md paragraph 6: name the type choice explicitly).
void launch_component_stats_init(int* d_count, unsigned long long* d_sum_x, unsigned long long* d_sum_y,
                                 int* d_min_x, int* d_max_x, int* d_min_y, int* d_max_y,
                                 unsigned long long* d_key_min_sum, unsigned long long* d_key_max_sum,
                                 unsigned long long* d_key_min_diff, unsigned long long* d_key_max_diff,
                                 int W, int H);
void launch_component_stats_accumulate(const unsigned char* d_mask, const int* d_label,
                                       int* d_count, unsigned long long* d_sum_x, unsigned long long* d_sum_y,
                                       int* d_min_x, int* d_max_x, int* d_min_y, int* d_max_y,
                                       unsigned long long* d_key_min_sum, unsigned long long* d_key_max_sum,
                                       unsigned long long* d_key_min_diff, unsigned long long* d_key_max_diff,
                                       int W, int H);

// ---- Stage 3: quad extraction — corner refinement (CANDIDATE-parallel: one
// thread per candidate, contrast with the pixel-parallel stages above). ----
// d_candidates : [n] device array of CandidateComponent (host-filtered, host
//                built, uploaded once per scene).
// d_gray, d_local_mean : the SAME full-resolution fields stage 1 produced —
//                the radial edge search re-reads real image content.
// d_quads      : [n] device array OUT.
void launch_corner_refine(const CandidateComponent* d_candidates, int n,
                          const unsigned char* d_gray, const float* d_local_mean,
                          int W, int H, QuadCorners* d_quads);

// ---- Stage 4: DLT homography solve (CANDIDATE-parallel; one 8x8 Gaussian
// elimination per thread, double precision — see file header). ------------
void launch_homography_solve(const QuadCorners* d_quads, int n, Homography* d_homographies);

// ---- Stage 5: perspective grid sampling + dictionary decode
// (CANDIDATE-parallel). d_dictionary: [kNumDictCodes] uint16 device array
// (the committed dictionary, uploaded once). correction_capacity is a
// RUNTIME value (loaded from data/sample/dictionary.bin's metadata — the
// dictionary is generated data, not a compile-time constant; see
// scripts/make_synthetic.py). ------------------------------------------------
void launch_grid_decode(const Homography* d_homographies, int n,
                        const unsigned char* d_gray, const float* d_local_mean, int W, int H,
                        const uint16_t* d_dictionary, int num_dict_codes, int correction_capacity,
                        Detection* d_detections);

// ---- Stage 6: pose from homography (CANDIDATE-parallel). Reads
// d_detections[i].accepted (set by stage 5) — pose is computed for every
// candidate with a valid homography regardless of decode acceptance (cheap,
// and useful for debugging a near-miss), but main.cu only ever reports pose
// for ACCEPTED detections. ---------------------------------------------------
void launch_pose_from_homography(const Homography* d_homographies, int n, Detection* d_detections);

// ===========================================================================
// CPU references (reference_cpu.cpp) — INDEPENDENT reimplementations of
// every stage above (see the twin-independence discussion in this file's
// header and in reference_cpu.cpp's own header). Suffixed "_cpu".
// ===========================================================================
void box_sum_h_cpu(const unsigned char* gray, float* row_sum, int W, int H);
void box_sum_v_cpu(const float* row_sum, float* local_mean, int W, int H);
void adaptive_threshold_cpu(const unsigned char* gray, const float* local_mean,
                            unsigned char* mask, int W, int H);

// ccl_union_find_cpu — classic two-pass union-find CCL (4-connectivity),
// canonicalized to "label = minimum linear index in the component" (same
// convention 30.01's CPU twin uses), so it is directly comparable to the
// GPU's label-propagation fixed point.
void ccl_union_find_cpu(const unsigned char* mask, int* label, int W, int H);

// Builds the full CandidateComponent list from a mask+label image via a
// single sequential host-side pass (no atomics needed on the CPU — one
// thread, one component at a time). Returns the number of candidates written
// into out (capped at kMaxCandidates).
int build_candidates_cpu(const unsigned char* mask, const int* label, int W, int H,
                         CandidateComponent* out);

QuadCorners corner_refine_one_cpu(const CandidateComponent& cand,
                                  const unsigned char* gray, const float* local_mean, int W, int H);
Homography homography_solve_one_cpu(const QuadCorners& quad);
Detection grid_decode_one_cpu(int candidate_index, const Homography& H,
                              const unsigned char* gray, const float* local_mean, int W, int Himg,
                              const uint16_t* dictionary, int num_dict_codes, int correction_capacity);
void pose_from_homography_one_cpu(const Homography& H, Detection& det);

#endif // PROJECT_KERNELS_CUH
