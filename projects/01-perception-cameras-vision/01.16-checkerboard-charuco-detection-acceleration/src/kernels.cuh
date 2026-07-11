// ===========================================================================
// kernels.cuh — interface for project 01.16
//               Checkerboard/ChArUco detection acceleration for
//               auto-calibration rigs: a BATCH of B=8 board views processed
//               view-parallel + pixel-parallel on the GPU (saddle-point
//               X-corner response -> NMS -> gradient-orthogonality sub-pixel
//               refinement -> homography-guided ArUco-style marker decode),
//               with the serial grid-walk, DLT homography, and Zhang mini-
//               calibration honestly kept on the host (see "GPU mapping,
//               Amdahl honesty" in THEORY.md).
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (driver + gates), kernels.cu (the GPU
// pipeline), and reference_cpu.cpp (the independent CPU oracle). Every
// board/marker geometry constant and layout struct all three must agree on
// lives HERE, once (CLAUDE.md paragraph 12). Style follows 01.06's
// kernels.cuh (single-sourced contract, "MUST MATCH ../scripts/
// make_synthetic.py" cross-references) — read that file first if you have
// not; this project reuses its PATTERNS (a home-grown small marker
// dictionary, perspective grid sampling + border-cell rejection) but is a
// smaller, simpler pipeline because markers here are never searched for
// blind: they are sampled at a location a homography ALREADY predicts, from
// checkerboard corners this project's own saddle detector already found.
//
// THE PIPELINE, FOUR TWINNED GPU STAGES + THREE SHARED HOST STAGES:
//   GPU 1. SADDLE RESPONSE   — per-pixel Hessian-determinant saddle test
//      (THEORY.md "The math" derives it): an X-corner is a SADDLE of image
//      intensity (one principal curvature positive, one negative), not a
//      Harris "L" corner (both principal curvatures large and SAME sign —
//      contrast with 01.04's structure-tensor corner response, cited by
//      name in THEORY.md). response = max(0, -(Ixx*Iyy - Ixy^2)).
//   GPU 2. NMS               — 2*kNmsRadius+1 window non-max suppression +
//      atomic compaction into a per-view candidate list (same
//      pack-then-compact spirit as 01.04/01.06, reimplemented for this
//      project's batched-view layout).
//   GPU 3. SUB-PIXEL REFINE  — the cornerSubPix idea (THEORY.md derives the
//      2x2 normal-equation system): iterate "every neighborhood gradient is
//      orthogonal to (sample - true_corner)" to convergence. This is the
//      accuracy-critical stage this project measures before/after (README
//      "Expected output").
//   HOST (shared, NOT twinned — see the independence note below): PLAIN
//      GRID ORDERING (order_grid_for_view) walks the refined corners into a
//      provisional (i,j) lattice using ONLY checkerboard geometry (a global
//      seed-and-walk search) — the classic result, carrying the classic
//      180-degree ambiguity (THEORY.md "The problem"), plus a real,
//      documented fragility under combined tilt/rotation/occlusion. THIS
//      PATH IS KEPT ONLY as the "ambiguity_lesson" gate's comparison
//      baseline (README "Expected output") — it is NOT the pipeline's
//      output of record any more (see the next paragraph).
//   GPU 4. MARKER DECODE (GPU-twinned)  — for each of the up to
//      kNumMarkerCodes=24 white-square markers, sample its 5x5 grid through
//      a GIVEN homography at BOTH a marker's assumed board position and its
//      180-degree MIRRORED position (mirror_square()), decode whichever
//      hypothesis clears the border-ring + Hamming-distance test. main.cu
//      runs this once, GPU vs CPU, using the PLAIN homography above, purely
//      to PROVE the decode primitive itself correct (0/192 mismatches,
//      README "Expected output") — independent of which grid-ordering
//      strategy ultimately anchors the pipeline's corners.
//   HOST (shared, NOT twinned): MARKER-FIRST GRID ORDERING
//      (order_grid_marker_first_for_view, THIS project's pipeline output of
//      record) — the production ChArUco strategy (README "Prior art"):
//      decode markers FIRST, independent of any global corner walk. Local
//      2x2 corner quads are found by purely LOCAL nearest-neighbor geometry
//      (no seed-and-walk, no retries budget — THEORY.md "The algorithm"),
//      each candidate quad's own tiny DLT homography is tried against EVERY
//      dictionary code (both axis assignments, both 180-degree reading
//      hypotheses — the local analogue of the plain path's axis-identity /
//      handedness bugs, resolved here by letting the dictionary itself pick
//      the one combination that decodes, rather than a length/handedness
//      heuristic). A clean decode gives that quad's 4 corners an ABSOLUTE
//      (i,j) label directly — no separate "vote across the whole view, then
//      flip everything" step is needed, because every marker anchor is
//      self-sufficient. Corners near no decoded marker are filled in by ONE
//      global homography, refit from every marker-anchored correspondence,
//      predicting + snapping the remainder (same tight-tolerance discipline
//      the plain path uses). 180-degree rotation and occlusion are handled
//      BY CONSTRUCTION: a missing/occluded region simply has no anchor
//      there; visible markers elsewhere still index correctly, with no
//      dependence on a global seed choice or a full-board walk succeeding.
//   HOST (shared): Zhang's mini-calibration (a 6x6 symmetric eigenvalue
//      problem, solved by cyclic Jacobi rotations — THEORY.md "The math"
//      derives the absolute-conic linear system) recovers (fx, fy, cx, cy)
//      from the exactly-ordered views' marker-first homographies.
//
// IMAGE LAYOUT — row-major, pixel (x,y) at linear index i = y*W + x, x
// rightward, y downward. A BATCH of B views is stored as ONE contiguous
// device array of B*H*W bytes: view b's pixel (x,y) is at flat index
// b*(W*H) + y*W + x — the single indexing formula every kernel below uses
// (batch_pixel_index()), so "view-parallel + pixel-parallel" is achieved
// with a single flat grid-stride loop over [0, B*W*H), not a 3-D launch
// (THEORY.md "The GPU mapping" argues why that flattening is the right
// choice here and measures its effect).
//
// TWIN-INDEPENDENCE (per reference_cpu.cpp's file-header ruling, and per
// this project's own brief): saddle response (float tolerance), NMS peak
// set (exact), sub-pixel refinement (tight tolerance), and marker decode
// (exact) are each typed TWICE — independently — in kernels.cu (GPU) and
// reference_cpu.cpp (CPU). Board/marker geometry constants and the small
// pure-bookkeeping HD helpers below (corner_board_xy, mirror_corner,
// marker_payload_bit_index, ...) are shared data-layout code, not
// "algorithm under test" (a disagreement there would be a layout bug, not a
// numerics bug). Grid ordering (BOTH the plain, checkerboard-only path AND
// the marker-first path), DLT, and the Zhang solve are SHARED HOST
// functions (single-sourced in reference_cpu.cpp, called by main.cu on
// whichever corner set it is validating) — not twinned pairwise, because
// they are cheap, serial, host-only bookkeeping over an already-verified
// corner set; main.cu instead validates them with INDEPENDENT gates that
// route through neither copy: grid_ordering / ambiguity_lesson / occlusion
// compare against scripts/make_synthetic.py's own corner ground truth, and
// mini_calibration compares against its own recorded camera intrinsics —
// exactly the "independent gate" tier the ruling requires whenever
// algorithmic code is shared rather than duplicated. Marker-first ordering
// additionally CALLS the already-twinned decode_one_hypothesis_cpu (the
// CPU half of the marker-decode twin) as its own marker-identification
// primitive, at NEW (homography, sample position) arguments the twin
// comparison above never directly exercised — this is safe and does not
// reopen the independence question: decode_one_hypothesis_cpu is pure
// arithmetic (bilinear sampling + a fixed threshold + Hamming distance)
// with no dependence on which homography or sample square it is handed,
// its GPU/CPU agreement was already proven for the arguments the marker-
// decode gate DOES exercise, and grid_ordering's own gate (ground truth,
// never this file's numbers) is exactly the independent check the ruling
// requires for this new call site too.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe (same trick as
// 01.01/01.06). Lets the pure data-layout formulas below live in exactly one
// place and be called identically from kernels.cu (device) and
// reference_cpu.cpp / main.cu (host, compiled by cl.exe).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// Image & batch geometry — MUST MATCH ../scripts/make_synthetic.py's
// identically-named constants (CLAUDE.md paragraph 12).
// ===========================================================================
constexpr int kImgW     = 320;   // view width, px
constexpr int kImgH     = 240;   // view height, px
constexpr int kNumViews = 8;     // the calibration rig batch size B

constexpr int kViewPixels  = kImgW * kImgH;              // pixels per view
constexpr int kBatchPixels = kNumViews * kViewPixels;    // total pixels in the batch

// batch_pixel_index — the ONE formula every kernel below uses to address the
// batched [B*H*W] image array: view b's pixel (x,y) lives at flat index
// b*(W*H) + y*W + x. Single-sourced so kernels.cu's launch and the CPU twin
// never drift on layout (a data-layout fact, not "algorithm under test").
HD inline int batch_pixel_index(int view, int x, int y)
{
    return view * kViewPixels + y * kImgW + x;
}

// ===========================================================================
// Board geometry — MUST MATCH ../scripts/make_synthetic.py's identically-
// named constants and helper functions (is_white_square, corner_board_xy,
// mirror_corner, mirror_square, marker_payload_bit_index,
// marker_is_border_cell, build_marker_id_table).
// ===========================================================================
constexpr int kBoardSquaresX = 8;                      // squares across
constexpr int kBoardSquaresY = 6;                      // squares down
constexpr int kBoardCornersX = kBoardSquaresX - 1;      // 7 -- inner-corner columns
constexpr int kBoardCornersY = kBoardSquaresY - 1;      // 5 -- inner-corner rows
constexpr int kNumCorners    = kBoardCornersX * kBoardCornersY;  // 35
constexpr float kSquareSizeM = 0.030f;                  // physical square side, meters

constexpr int kMarkerGridN       = 5;    // total marker grid is 5x5 cells
constexpr int kMarkerPayloadN    = 3;    // interior payload is 3x3 = 9 bits
constexpr int kMarkerPayloadBits = kMarkerPayloadN * kMarkerPayloadN;                 // 9
constexpr int kMarkerBorderCells = kMarkerGridN * kMarkerGridN - kMarkerPayloadBits;  // 16
constexpr float kMarkerFillFrac  = 0.70f;   // marker footprint, fraction of its square's side
constexpr int kNumMarkerCodes    = 24;      // one code per WHITE square (8x6 board -> exactly 24)

// is_white_square — a square carries a marker iff (bx+by) is odd. MUST
// MATCH make_synthetic.py's is_white_square().
HD inline bool is_white_square(int bx, int by)
{
    return ((bx + by) & 1) == 1;
}

// corner_board_xy — inner corner (i,j) -> board-plane meters. One square of
// margin from the board's own outer edge on every side (only INTERIOR
// vertices are "corners" — the standard checkerboard convention). MUST
// MATCH make_synthetic.py's corner_board_xy().
HD inline void corner_board_xy(int i, int j, float& X, float& Y)
{
    X = static_cast<float>(i + 1) * kSquareSizeM;
    Y = static_cast<float>(j + 1) * kSquareSizeM;
}

// mirror_corner / mirror_square — the board's own 180-degree in-plane
// rotational symmetry (its geometry is IDENTICAL after this relabeling —
// THE source of the classic checkerboard ambiguity; THEORY.md "The
// problem"). MUST MATCH make_synthetic.py's mirror_corner / mirror_square.
HD inline void mirror_corner(int i, int j, int& mi, int& mj)
{
    mi = kBoardCornersX - 1 - i;
    mj = kBoardCornersY - 1 - j;
}
HD inline void mirror_square(int bx, int by, int& mbx, int& mby)
{
    mbx = kBoardSquaresX - 1 - bx;
    mby = kBoardSquaresY - 1 - by;
}

// square_center_board_xy — the board-plane center of white square (bx,by)'s
// marker footprint (its own square's center — the marker is centered in
// its square by construction, make_synthetic.py's board_ink_at()).
HD inline void square_center_board_xy(int bx, int by, float& X, float& Y)
{
    X = (static_cast<float>(bx) + 0.5f) * kSquareSizeM;
    Y = (static_cast<float>(by) + 0.5f) * kSquareSizeM;
}

// marker_payload_bit_index — PAYLOAD-LOCAL coords (pr,pc), each in
// [0, kMarkerPayloadN), -> bit index k = pr*3+pc. MUST MATCH
// make_synthetic.py's marker_payload_bit_index().
HD inline int marker_payload_bit_index(int pr, int pc)
{
    return pr * kMarkerPayloadN + pc;
}

// marker_is_border_cell — GRID-LOCAL coords (r,c), each in [0,kMarkerGridN),
// true iff on the outer ring. MUST MATCH make_synthetic.py's
// marker_is_border_cell().
HD inline bool marker_is_border_cell(int r, int c)
{
    return r == 0 || r == kMarkerGridN - 1 || c == 0 || c == kMarkerGridN - 1;
}

// build_marker_id_table — row-major (by, then bx) enumeration of WHITE
// squares -> sequential marker_id in [0, kNumMarkerCodes). Fills two small
// host-side tables (there are only kBoardSquaresX*kBoardSquaresY = 48
// squares total, so this is negligible one-time bookkeeping, never a hot
// path). MUST MATCH make_synthetic.py's build_marker_id_table() — same
// (by, bx) nested walk order, so marker_id assignment agrees between the
// Python renderer and this C++ pipeline without ever shipping a table file.
//
// Parameters: square_bx_of_id / square_by_of_id — OUT: [kNumMarkerCodes].
//             id_of_square — OUT: [kBoardSquaresX*kBoardSquaresY], -1 for
//             black squares, else the marker_id (indexed bx + by*kBoardSquaresX).
// ---------------------------------------------------------------------------
inline void build_marker_id_table(int square_bx_of_id[kNumMarkerCodes],
                                  int square_by_of_id[kNumMarkerCodes],
                                  int id_of_square[kBoardSquaresX * kBoardSquaresY])
{
    for (int k = 0; k < kBoardSquaresX * kBoardSquaresY; ++k) id_of_square[k] = -1;
    int next_id = 0;
    for (int by = 0; by < kBoardSquaresY; ++by) {
        for (int bx = 0; bx < kBoardSquaresX; ++bx) {
            if (is_white_square(bx, by)) {
                square_bx_of_id[next_id] = bx;
                square_by_of_id[next_id] = by;
                id_of_square[bx + by * kBoardSquaresX] = next_id;
                ++next_id;
            }
        }
    }
}

// square_of_marker_id — the CLOSED-FORM inverse of build_marker_id_table(),
// usable from DEVICE code (no table upload needed). Derivation: this
// project's board has kBoardSquaresX=8 (even), so every row of squares
// contains EXACTLY kBoardSquaresX/2 = 4 white squares (is_white_square()'s
// alternating-parity rule), at bx in {1,3,5,7} when the row's own by is
// even (by+bx must be odd) or bx in {0,2,4,6} when by is odd. Enumerating
// marker_id row-major (by outer, bx inner ascending — the same order
// build_marker_id_table()/make_synthetic.py's build_marker_id_table() walk)
// therefore has a direct formula, verified against build_marker_id_table()
// at program startup by main.cu (never assumed silently — CLAUDE.md §13).
HD inline void square_of_marker_id(int marker_id, int& bx, int& by)
{
    constexpr int kWhitePerRow = kBoardSquaresX / 2;   // = 4
    by = marker_id / kWhitePerRow;
    const int k = marker_id % kWhitePerRow;
    bx = ((by & 1) == 0) ? (1 + 2 * k) : (0 + 2 * k);
}

// ===========================================================================
// Decode threshold — MUST MATCH make_synthetic.py's INK_BLACK / INK_WHITE
// (30.0 / 220.0): the fixed midpoint a sampled marker cell is compared
// against. A FIXED threshold (not 01.06's adaptive local-mean field) is an
// honest, documented simplification: this project samples marker cells at a
// PRECISELY known homography-predicted location (never searches blind for
// them), so — unlike 01.06, which must find unknown-location tags anywhere
// in an unknown-illumination scene — a single global threshold under a
// modest illumination gradient (make_synthetic.py's amplitude=0.10) is
// sufficient here (README "Limitations & honesty" names this trade-off).
// ---------------------------------------------------------------------------
constexpr float kInkMidThreshold = 125.0f;   // (30 + 220) / 2

// ===========================================================================
// Saddle-response & NMS parameters (THEORY.md "The algorithm" derives each).
//   kSaddleStep     — finite-difference half-step (px) for the Hessian
//     estimate. 2, not 1: the image already carries a 5-tap blur + sensor
//     noise (make_synthetic.py), so a coarser second-derivative stencil
//     trades a little localization for much better noise rejection — see
//     THEORY.md "Numerical considerations" for the measured effect.
//   kNmsRadius      — non-max suppression window half-size (px). The window
//     side (2*kNmsRadius+1 = 9) must stay well under the smallest expected
//     corner spacing (~22-31 px, this project's square size in the
//     farthest/closest views) so adjacent corners never suppress each
//     other.
//   kSaddleRespThresh — minimum response to be a CANDIDATE at all (measured
//     and margined from an actual run — see README "Expected output").
// ---------------------------------------------------------------------------
constexpr int   kSaddleStep        = 5;
constexpr int   kNmsRadius         = 6;
constexpr float kSaddleRespThresh  = 400.0f;
// kMinAxisCurvature — a SECOND, independent gate on top of det(Hessian)<0:
// both |Ixx| and |Iyy| must individually exceed this floor. Why: a single
// strong, noisy EDGE (not a corner at all -- e.g. the board's own outer
// silhouette against the background) can have a large second derivative in
// the direction CROSSING the edge but only NOISE-scale curvature ALONG it;
// det<0 alone cannot tell that apart from a true saddle, because det is a
// PRODUCT and a large-times-noise value crosses an det threshold about as
// often as a true corner does. Requiring BOTH axis curvatures to be large
// rejects "edge plus noise" false positives while true X-corners (strong
// curvature in every direction through the pinwheel pattern) sail through.
// THEORY.md "Numerical considerations" shows the measured effect.
constexpr float kMinAxisCurvature  = 60.0f;
// kMaxDiagonalAsymmetry — a THIRD gate, and the one that actually matters
// most in practice (THEORY.md "Numerical considerations" derives why): a
// true X-corner has OPPOSITE quadrants the SAME color (NW~=SE, NE~=SW) --
// that two-color diagonal symmetry is the whole geometric definition of a
// checkerboard saddle. A three-region T-junction (e.g. the board's own
// outer edge, where a uniform background meets two DIFFERENT alternating
// squares) can still produce det(Hessian)<0 with real curvature on both
// axes -- it is NOT symmetric (one diagonal pair matches, the other does
// not) but det() and the per-axis floor alone cannot see that asymmetry.
// Requiring both |NW-SE| and |NE-SW| stay small directly encodes the
// symmetry a real corner has and a T-junction does not.
constexpr float kMaxDiagonalAsymmetry = 60.0f;
constexpr int   kMaxCandidatesPerView = 512;   // device buffer capacity per view (35 expected; texture/noise headroom)

// ===========================================================================
// Sub-pixel refinement parameters (the cornerSubPix idea; THEORY.md derives
// the 2x2 normal-equation system).
//   kRefineWinRadius — the window (px) of neighborhood samples used to build
//     the gradient-orthogonality system at each iteration.
//   kRefineIters     — fixed iteration count (THEORY.md "Numerical
//     considerations" discusses convergence and why a fixed count, not a
//     convergence threshold, keeps the GPU/CPU twin's iteration COUNT
//     identical -- an early-exit-on-convergence version would risk the two
//     platforms stopping after a different number of steps whenever a
//     magnitude comparison lands within float noise of its own threshold).
// ---------------------------------------------------------------------------
constexpr int kRefineWinRadius = 5;
constexpr int kRefineIters     = 5;

// ===========================================================================
// Marker decode tolerance (analogous to 01.06's kMaxBorderErrors, derived
// the same way -- see THEORY.md "Numerical considerations" for this
// project's own measured border-error distribution).
// ---------------------------------------------------------------------------
constexpr int kMaxMarkerBorderErrors = 5;   // out of kMarkerBorderCells = 16

// ===========================================================================
// Grid-ordering parameters (host-only; THEORY.md "The algorithm" walks the
// nearest-neighbor + homography-guided-prediction procedure step by step).
//   kGridMatchTolFactor — a predicted grid point is matched to the nearest
//     ACTUAL detected corner only if within this fraction of the view's own
//     estimated local corner spacing -- scale-adaptive so the same constant
//     works at every view's depth (22-31 px spacing across this project's
//     8 views).
//   kMinCornersForBoard — minimum corner count for main.cu to call a view
//     "a detected board" at all (the negative-control gate's floor: texture
//     clutter may trip a handful of spurious saddle responses, but must
//     never assemble into a 7x5-consistent lattice this large).
// ---------------------------------------------------------------------------
constexpr float kGridMatchTolFactor = 0.45f;
constexpr int   kMinCornersForBoard = 20;

// ===========================================================================
// popcount_u32 — Hamming-weight helper shared by marker decode (GPU and
// CPU); pure bit arithmetic, a data-layout fact like 01.06's popcount16.
// ---------------------------------------------------------------------------
HD inline int popcount_u32(unsigned int v)
{
    int n = 0;
    while (v) { n += (v & 1u); v >>= 1u; }
    return n;
}

// ===========================================================================
// xorshift32_next — Marsaglia's xorshift32 PRNG, one step. Reimplemented
// here (matching make_synthetic.py's XorShift32 and 01.04/01.06's identical
// helper) even though THIS project's C++ side never actually draws random
// numbers at runtime (everything it consumes is the committed, already-
// generated sample) -- kept for any future exercise that wants to jitter
// synthetic inputs from C++ directly (README "Exercises").
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

// ===========================================================================
// Shared data-layout structs.
// ===========================================================================

// RawCandidate — one NMS peak: integer pixel location + response score,
// tagged with which view it came from (the batch is flattened, so every
// downstream stage needs this tag to know which image slice to re-read).
struct RawCandidate {
    int   view;
    int   x, y;       // integer pixel location (the NMS peak, pre-refinement)
    float score;
};

// RefinedCorner — stage 3's output: the same candidate, sub-pixel, after
// gradient-orthogonality iteration. `valid` is false when the 2x2 normal-
// equation system was singular (near-zero determinant -- a flat or
// low-texture neighborhood; THEORY.md "Numerical considerations").
struct RefinedCorner {
    int   view;
    float x, y;
    bool  valid;
};

// Homography — 3x3 row-major, BOARD-METERS -> PIXEL homogeneous mapping:
// pixel_homog = H * (X, Y, 1) for a board-plane point (X, Y, 0) in meters.
// `valid` is false when the DLT normal-equation solve was singular/
// ill-conditioned.
struct Homography {
    double h[9];
    bool   valid;
};

// MarkerDecodeResult — stage 4's per-(view, marker_id) output. Two
// hypotheses are tried per marker (see this file's header): `hyp_mirrored`
// records WHICH one succeeded when `accepted` is true.
struct MarkerDecodeResult {
    int   view;
    int   marker_id;
    bool  border_ok_identity;
    bool  border_ok_mirrored;
    bool  accepted;          // true iff at least one hypothesis cleared the border+Hamming test
    bool  hyp_mirrored;      // which hypothesis was accepted (only meaningful if accepted)
    int   hamming_distance;  // to this marker's own true code, under the ACCEPTED hypothesis
};

// ===========================================================================
// GPU launch wrappers (kernels.cu). main.cu never writes <<<...>>> syntax
// directly (repo-wide discipline).
// ===========================================================================

// ---- Stage 1: saddle response (PIXEL-parallel, batched: one flat
// grid-stride loop over num_views*kViewPixels). ------------------------------
// `num_views` is a RUNTIME batch size (not always kNumViews=8): main.cu
// calls this once on the 8-view calibration rig batch AND once more, as a
// batch-of-1, on the negative-control scene -- the SAME kernel, no special
// casing, because every stage here only ever needs "how many W*H slices are
// in this buffer", never the rig's own fixed size.
// d_gray : [num_views*kViewPixels] uint8 IN, the batched grayscale views.
// d_resp : [num_views*kViewPixels] float OUT, the saddle response (>=0; 0
//          near each view's OWN border, where the Hessian stencil cannot be
//          formed).
void launch_saddle_response(const unsigned char* d_gray, float* d_resp, int num_views);

// ---- Stage 2: NMS + compaction (PIXEL-parallel, batched). -----------------
// d_resp        : [num_views*kViewPixels] float IN.
// d_cand        : [num_views*kMaxCandidatesPerView] RawCandidate OUT.
// d_view_counts : [num_views] int OUT -- raw (possibly over-capacity) count
//                 per view; caller MUST zero this before the launch.
void launch_nms_candidates(const float* d_resp, RawCandidate* d_cand, int* d_view_counts, int num_views);

// ---- Stage 3: sub-pixel refinement (CANDIDATE-parallel: one thread per
// candidate, flattened across every view). ----------------------------------
// d_gray : [kBatchPixels] uint8 IN (the SAME batched images stage 1 read).
// d_cand : [n] RawCandidate IN.
// d_out  : [n] RefinedCorner OUT.
void launch_subpixel_refine(const unsigned char* d_gray, const RawCandidate* d_cand, int n, RefinedCorner* d_out);

// ---- Stage 4: marker decode (one thread per (view, marker_id) pair,
// kNumViews*kNumMarkerCodes threads total -- see this file's header for the
// two-hypothesis trick). -----------------------------------------------------
// d_gray        : [kBatchPixels] uint8 IN.
// d_homography  : [kNumViews] Homography IN (from the shared host DLT solve
//                 -- see this file's header's independence note).
// d_true_codes  : [kNumMarkerCodes] uint16 IN (the committed dictionary).
// correction_capacity : loaded at runtime from marker_dictionary.bin.
// d_results     : [kNumViews*kNumMarkerCodes] MarkerDecodeResult OUT.
void launch_marker_decode(const unsigned char* d_gray, const Homography* d_homography,
                          const uint16_t* d_true_codes, int correction_capacity,
                          MarkerDecodeResult* d_results);

// ===========================================================================
// CPU references (reference_cpu.cpp) — INDEPENDENT reimplementations of the
// four GPU stages above (see this file's header and reference_cpu.cpp's own
// header for the twin-independence ruling). Suffixed "_cpu".
// ===========================================================================
void saddle_response_cpu(const unsigned char* gray, float* resp, int num_views);
// nms_candidates_cpu — returns the TOTAL count written (<= num_views*max_per_view);
// out_view_counts[num_views] (OUT, caller-allocated) receives the RAW
// per-view count (possibly > max_per_view), the same "keep counting past
// capacity" semantics as the GPU path's atomicAdd counter, so main.cu can
// compare per-view counts exactly, not just the clipped totals.
int  nms_candidates_cpu(const float* resp, RawCandidate* out, int max_per_view, int num_views, int* out_view_counts);
RefinedCorner subpixel_refine_one_cpu(const unsigned char* gray, const RawCandidate& cand);
MarkerDecodeResult marker_decode_one_cpu(const unsigned char* gray, const Homography& H,
                                         int view, int marker_id,
                                         const uint16_t* true_codes, int correction_capacity);

// ===========================================================================
// Shared HOST-ONLY functions (single-sourced; see this file's header's
// independence note for why these are NOT twinned). Defined in
// reference_cpu.cpp so both the GPU-corner path and any CPU-corner path in
// main.cu call the identical implementation.
// ===========================================================================

// GridLabel — the (i,j) lattice index assigned to one refined corner by
// order_grid_for_view() below, or {-1,-1} if the corner could not be placed
// into a consistent grid at all (an honest failure, not silently dropped).
struct GridLabel { int i = -1, j = -1; };

// order_grid_for_view — THEORY.md "The algorithm" walks this step by step:
// (1) pick a seed corner (smallest x+y -- "top-left-ish" in image space);
// (2) find its two nearest, most-orthogonal neighbors to estimate the two
//     dominant lattice directions;
// (3) walk each direction from the seed to build a first row and first
//     column, re-estimating direction after every accepted step (handles
//     perspective foreshortening);
// (4) DLT-fit a homography from the first row + column (assuming the seed
//     is canonical (0,0) -- the very assumption that carries the 180-degree
//     ambiguity, see mirror_corner()'s doc comment);
// (5) use that homography to PREDICT every remaining grid point's pixel
//     location and nearest-neighbor-snap it to an actual detected corner.
// This is HOST, SERIAL, and NOT twinned (see this file's header) -- it
// consumes an already-verified corner set and is validated by the
// grid_ordering / ambiguity_lesson / occlusion gates in main.cu instead.
//
// Parameters: cx,cy   — [n] REFINED, valid corner coordinates for ONE view
//                       (any order).
//             out     — OUT: [n] GridLabel, one per input corner, same order.
//             out_hom — OUT: the provisional homography DLT-fit in step (4)
//                       (board-plane meters, ASSUMING out[]'s labeling).
// Returns: number of corners successfully placed into the grid (<= n).
int order_grid_for_view(const float* cx, const float* cy, int n, GridLabel* out, Homography& out_hom);

// order_grid_marker_first_for_view — MARKER-FIRST grid ordering: THIS
// project's pipeline output of record (kernels.cuh's file header and
// THEORY.md "The algorithm" walk it step by step). Unlike
// order_grid_for_view (checkerboard-only, global seed-and-walk, kept only
// as the ambiguity_lesson comparison baseline), this decodes markers FIRST
// and independently of any global corner walk:
//   1. For every refined corner as a candidate "quad seed", find a LOCAL
//      2x2 cluster of corners (nearest neighbor + most-orthogonal-at-
//      comparable-distance neighbor + the predicted 4th/diagonal corner,
//      snapped to an actual detection) — pure local geometry, no chain, no
//      global state, so occlusion or a bad seed ELSEWHERE in the view
//      cannot break it.
//   2. Fit that quad's own tiny DLT homography (board-plane meters
//      (0,0)-(kSquareSizeM,kSquareSizeM) <-> the 4 quad corners) and try
//      EVERY dictionary code against it — both of the two possible axis
//      assignments (which quad neighbor plays the board-X role; local
//      geometry alone cannot tell, unlike the plain path's chain-length
//      cue) and both 180-degree reading hypotheses (identity / mirrored).
//      A clean decode (border-ring + Hamming test, same primitive the
//      marker-decode gate twins) gives an ABSOLUTE (i,j) label to all 4
//      quad corners directly — see reference_cpu.cpp's own header comment
//      at this function's definition for the full label-assignment
//      derivation.
//   3. Corners reached by more than one decoded quad are VOTED (a strict
//      majority wins; a tie is left unanchored rather than guessed).
//   4. Every remaining, unanchored corner is filled in by ONE global
//      homography fit from every marker-anchored correspondence,
//      predicting + snapping the rest (same tight-tolerance discipline as
//      order_grid_for_view's own predict-remaining phase), then refit ONCE
//      more from every final correspondence.
//
// Parameters: cx,cy   — [n] REFINED, valid corner coordinates for ONE view
//                       (any order) — same convention as order_grid_for_view.
//             gray    — [kBatchPixels] uint8, the batched grayscale images
//                       (needed here, unlike order_grid_for_view, because a
//                       marker's identity can only be read from pixels).
//             view    — which view's slice of `gray` this corner set is from.
//             true_codes — [kNumMarkerCodes] uint16, the committed dictionary.
//             correction_capacity — from marker_dictionary.bin.
//             out     — OUT: [n] GridLabel, one per input corner, same order.
//             out_hom — OUT: the FINAL homography (board meters -> pixel),
//                       refit from every placed correspondence.
//             out_quads_decoded, out_anchor_conflicts — OPTIONAL OUT:
//                       diagnostics (how many local quads decoded cleanly,
//                       how many corners saw disagreeing quad proposals) —
//                       pass nullptr to skip.
// Returns: number of corners successfully placed into the grid (<= n); 0 if
// no marker decoded at all in this view (an honest failure, e.g. the
// negative control has no board and thus no markers).
int order_grid_marker_first_for_view(const float* cx, const float* cy, int n,
                                     const unsigned char* gray, int view,
                                     const uint16_t* true_codes, int correction_capacity,
                                     GridLabel* out, Homography& out_hom,
                                     int* out_quads_decoded = nullptr, int* out_anchor_conflicts = nullptr);

// solve_dlt_homography — Hartley-normalized Direct Linear Transform: given
// n >= 4 board-plane <-> pixel correspondences, fit the least-squares
// homography (h33 fixed to 1 in NORMALIZED coordinates, solved via an 8x8
// Gaussian elimination with partial pivoting, then denormalized -- cite
// 33.01's batched-small-dense-solve-per-instance pattern, applied here to a
// homography instead of a robot-arm Jacobian). THEORY.md "Numerical
// considerations" derives why normalization matters (Hartley 1997).
Homography solve_dlt_homography(const float* board_x, const float* board_y,
                                const float* px_x, const float* px_y, int n);

// solve_zhang_calibration — Zhang's absolute-conic linear method (THEORY.md
// "The math" derives it in full): stack 2 constraint rows per homography
// into a 2*kNumViews x 6 design matrix A, form the 6x6 normal matrix A^T A,
// find its SMALLEST eigenvalue's eigenvector via cyclic Jacobi rotations
// (jacobi_eigen_symmetric6 below) -- that eigenvector IS the absolute
// conic's upper-triangular coefficients b = [B11,B12,B22,B13,B23,B33], from
// which the standard closed-form (Zhang 2000, Appendix B) recovers
// (fx, fy, cx, cy, skew).
struct ZhangResult {
    bool  valid = false;
    double fx = 0, fy = 0, cx = 0, cy = 0, skew = 0;
};
ZhangResult solve_zhang_calibration(const Homography* homs, int n);

// jacobi_eigen_symmetric6 — classic cyclic Jacobi eigenvalue algorithm
// (Golub & Van Loan) specialized to N=6: repeatedly zero the largest
// off-diagonal pair via a Givens-like rotation until the matrix is
// (numerically) diagonal. THEORY.md "The GPU mapping" explains why this
// stays on the host (a one-shot 6x6 problem, 8 times total per demo run --
// utterly dominated by the O(B*H*W) pixel-parallel stages above; see the
// measured Amdahl fraction there).
// Parameters: A (IN/OUT, overwritten with the diagonal), eigvecs (OUT,
// columns are the eigenvectors, same order as the diagonal of A on return).
void jacobi_eigen_symmetric6(double A[6][6], double eigvecs[6][6]);

#endif // PROJECT_KERNELS_CUH
