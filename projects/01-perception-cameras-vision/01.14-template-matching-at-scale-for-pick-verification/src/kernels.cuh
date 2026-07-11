// ===========================================================================
// kernels.cuh — single-sourced data contracts + kernel/reference declarations
//               for project 01.14 (Template matching (NCC) at scale for pick
//               verification)
//
// Role in the project
// --------------------
// The ONE place that defines the tray/slot geometry, the template/rotation-
// set layout, the integral-image and score-buffer layouts, and every
// kernel/launcher/CPU-oracle signature shared by kernels.cu (GPU),
// reference_cpu.cpp (CPU oracle), and main.cu (orchestration). Per the
// template's independence ruling (see reference_cpu.cpp's header, and
// project 01.13's precedent): DATA-LAYOUT CONTRACTS (sizes, offsets, the
// template statistics table) are single-sourced and shared here; the
// ALGORITHMIC CORE of every twinned stage (integral-image scan, box query,
// NCC scoring) is written TWICE, independently, in kernels.cu and
// reference_cpu.cpp.
//
// THE STORY IN ONE PARAGRAPH (read THEORY.md for the full derivation): a
// robot picks a part into one of K tray slots; a camera photographs the
// tray; this project checks EVERY slot against the part it was supposed to
// receive using zero-normalized cross-correlation (NCC), searched over a
// small +-8 px translation window and a 5-angle rotation set, computed THREE
// ways on the GPU (naive direct, integer sum-table, sum-table + shared
// memory) to teach the classic "cache the redundant work" acceleration
// ladder — then classifies each slot OK / WRONG_PART / EMPTY and reports a
// verdict table, exactly the sanity check a pick-and-place cell runs after
// every cycle.
//
// TRAY GEOMETRY — 6 columns x 4 rows = 24 slots (K = NUM_SLOTS), each slot a
// WINDOW x WINDOW search region (see SECTION 1) tiled with a fixed border and
// pitch so search windows never overlap. Slot index is ROW-MAJOR:
//     slot = row * NUM_COLS + col,  row in [0,NUM_ROWS), col in [0,NUM_COLS)
//
// TEMPLATE SET — NUM_TYPES (3) machined part silhouettes x NUM_ROT (5)
// pre-rotated angles = NUM_TEMPLATES (15) templates, each TEMPLATE_SIZE x
// TEMPLATE_SIZE uint8 images, laid out template_id = type*NUM_ROT + rot_idx
// (SECTION 2). Rotation set teaches NCC's rotation brittleness (THEORY.md);
// evaluating the FULL set per slot is what recovers a part placed a few
// degrees off nominal.
//
// SCORE VOLUME — every (slot, template, offset) triple gets one NCC score:
// 24 slots * 15 templates * 17*17 offsets = 104,040 evaluations per full
// pass — the "batched, at-scale" launch the catalog bullet names. Three GPU
// kernels compute the SAME volume three different ways (kernels.cu); a CPU
// oracle computes it a fourth, independent way (reference_cpu.cpp); main.cu
// verifies all four agree, then classifies from the verified scores.
//
// Read this after: README.md/THEORY.md (the walkthrough). Read this before:
// kernels.cu, reference_cpu.cpp, main.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>

// HD — the small geometry/layout helper functions below (slot_window_x0,
// ii_index, score_index, ...) are called from BOTH device kernels (kernels.cu)
// and plain host code (main.cu, reference_cpu.cpp compiled by cl.exe, which
// does not know __host__/__device__ at all). Marking them __host__ __device__
// under nvcc, and leaving the qualifier out entirely under cl.exe, is the
// same #ifdef __CUDACC__ trick kernels.cuh's file header describes for
// __global__ declarations — here applied to small pure-arithmetic functions
// instead, so ONE definition serves both compilers and both memory spaces.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// SECTION 1 — tray / slot / search geometry
// ===========================================================================

// TEMPLATE_SIZE: the machined-part silhouette is rendered/rasterized into a
// TEMPLATE_SIZE x TEMPLATE_SIZE uint8 patch (SECTION 2 picks the shapes so
// they fit with a couple of pixels of background border — see
// scripts/make_synthetic.py). Small enough that 104,040 NCC evaluations run
// in milliseconds; large enough that 3 distinct machined silhouettes are
// visually and numerically distinguishable after anti-aliasing + texture.
static constexpr int TEMPLATE_SIZE = 24;

// SEARCH_RADIUS: the catalog bullet's "+-8 px" translation search — pick
// verification KNOWS roughly where the part should be (the robot commanded a
// specific slot pose), so unlike generic scene-wide template matching, the
// search is a small window around the nominal slot center, not the whole
// image (README "Limitations" states this scoping decision explicitly).
static constexpr int SEARCH_RADIUS = 8;
static constexpr int NUM_OFFSETS_1D = 2 * SEARCH_RADIUS + 1;              // 17: dx/dy in [-8,+8]
static constexpr int NUM_OFFSETS = NUM_OFFSETS_1D * NUM_OFFSETS_1D;        // 289 candidate placements

// WINDOW: the slot's search region — big enough that EVERY candidate
// TEMPLATE_SIZE x TEMPLATE_SIZE placement across the full +-SEARCH_RADIUS
// sweep stays inside it, with zero slack (a placement at offset -8 starts at
// window-local 0; at +8 it ends exactly at WINDOW). This is what main.cu's
// verify stage relies on: no placement ever reads outside a slot's window.
static constexpr int WINDOW = TEMPLATE_SIZE + 2 * SEARCH_RADIUS;           // 40

// Tray layout: NUM_COLS x NUM_ROWS slots, each a WINDOW x WINDOW region, tiled
// with a fixed PITCH (window + gap) and an outer BORDER so neighboring
// slots' search windows never overlap (a real tray's slots are physically
// separated; overlapping search regions would let a neighbor's part leak
// into this slot's NCC score, which would be a real, un-taught bug).
static constexpr int NUM_COLS = 6;
static constexpr int NUM_ROWS = 4;
static constexpr int NUM_SLOTS = NUM_COLS * NUM_ROWS;                      // 24 = K
static constexpr int SLOT_GAP = 12;                                        // px between adjacent windows
static constexpr int SLOT_PITCH = WINDOW + SLOT_GAP;                       // 52
static constexpr int BORDER = 12;                                          // px, outer tray margin

static constexpr int IMG_W = 2 * BORDER + (NUM_COLS - 1) * SLOT_PITCH + WINDOW;  // 324
static constexpr int IMG_H = 2 * BORDER + (NUM_ROWS - 1) * SLOT_PITCH + WINDOW;  // 220
static constexpr int IMG_PIXELS = IMG_W * IMG_H;                                 // 71,280

// slot_col/slot_row/slot_window_x0/y0 — the ONE place slot index maps to
// pixel geometry; kernels.cu, reference_cpu.cpp, and main.cu all call these
// instead of re-deriving the arithmetic (a data-layout contract, not an
// "algorithm" — see the file header's independence ruling).
HD static inline int slot_col(int slot) { return slot % NUM_COLS; }
HD static inline int slot_row(int slot) { return slot / NUM_COLS; }
HD static inline int slot_window_x0(int slot) { return BORDER + slot_col(slot) * SLOT_PITCH; }
HD static inline int slot_window_y0(int slot) { return BORDER + slot_row(slot) * SLOT_PITCH; }
// The slot's NOMINAL center in tray-image pixels — where the robot's plan
// says the part should land (offset (0,0) in the search sweep below).
HD static inline float slot_center_x(int slot) { return static_cast<float>(slot_window_x0(slot) + WINDOW / 2); }
HD static inline float slot_center_y(int slot) { return static_cast<float>(slot_window_y0(slot) + WINDOW / 2); }

// ===========================================================================
// SECTION 2 — the template / rotation-set layout
// ===========================================================================

// 3 machined-part silhouettes, rendered/rasterized in scripts/make_synthetic.py
// in the 01.13 visual style (analytic shape + hashed texture + anti-aliased
// supersampling): a corner BRACKET, a toothed GEAR_DISK, and a CONNECTOR_BLOCK
// with two mounting holes — distinct enough silhouettes that NCC should
// separate them cleanly (measured in THEORY.md "How we verify correctness").
static constexpr int NUM_TYPES = 3;
static constexpr int TYPE_BRACKET = 0;
static constexpr int TYPE_GEAR_DISK = 1;
static constexpr int TYPE_CONNECTOR_BLOCK = 2;

// The 5-angle rotation set (degrees) — THE rotation-brittleness lesson
// (THEORY.md derives NCC's translation invariance but NOT rotation
// invariance): a single 0-degree template alone cannot recover a part that
// landed a few degrees off nominal (a real robot placement/orientation
// tolerance), but scoring against ALL 5 pre-rotated templates and keeping the
// best does. Index 2 is exactly 0 degrees — the "single-template" baseline
// the rotation_lesson gate compares against the full 5-angle set.
static constexpr int NUM_ROT = 5;
static constexpr float ROTATION_DEG[NUM_ROT] = { -6.0f, -3.0f, 0.0f, 3.0f, 6.0f };
static constexpr int ROT_ZERO_IDX = 2;   // ROTATION_DEG[2] == 0.0f — the single-angle baseline template

static constexpr int NUM_TEMPLATES = NUM_TYPES * NUM_ROT;                  // 15
static constexpr int TEMPLATE_PIXELS = TEMPLATE_SIZE * TEMPLATE_SIZE;      // 576

// template_id — the ONE place (type, rotation) maps to a flat template index
// into the [NUM_TEMPLATES][TEMPLATE_SIZE][TEMPLATE_SIZE] template array.
HD static inline int template_id(int type, int rot_idx) { return type * NUM_ROT + rot_idx; }
// The single 0-degree template for a given type — the rotation_lesson gate's
// "what a naive single-template matcher would use" baseline.
HD static inline int template_id_single(int type) { return template_id(type, ROT_ZERO_IDX); }

// ===========================================================================
// SECTION 3 — the NCC score volume layout
// ===========================================================================
// Every (slot, template, offset) triple gets ONE float NCC score, in
// [-1, +1] (THEORY.md derives the range from the Cauchy-Schwarz inequality).
// Flat layout, slowest-to-fastest index: slot, template, offset_y, offset_x
// — matching the GPU launch's grid.x=slot, grid.y=template, block=(ox,oy)
// mapping (kernels.cu), so a kernel's own thread indices ARE the layout's
// trailing indices with zero re-derivation.
static constexpr long long SCORE_VOLUME_CELLS =
    static_cast<long long>(NUM_SLOTS) * NUM_TEMPLATES * NUM_OFFSETS_1D * NUM_OFFSETS_1D;  // 104,040

HD static inline long long score_index(int slot, int tmpl, int oy, int ox)
{
    return (((static_cast<long long>(slot) * NUM_TEMPLATES + tmpl) * NUM_OFFSETS_1D) + oy) * NUM_OFFSETS_1D + ox;
}

// ===========================================================================
// SECTION 4 — integral-image layout (the sum-table acceleration structure)
// ===========================================================================
// ONE integral image pair covers the WHOLE tray image (not one per slot —
// every slot's window sits at a different, non-overlapping region of the
// SAME tray image, so a single pair of tables serves every box query in the
// whole 104,040-evaluation volume). Standard PADDED convention (a zero
// border row/col, matching OpenCV's cv::integral): table size
// (IMG_W+1) x (IMG_H+1), II[0][*] = II[*][0] = 0, so a box query never needs
// a bounds special-case (THEORY.md "The GPU mapping" derives the algebra).
//
// TWO tables, TWO integer widths — a load-bearing overflow decision, derived
// in THEORY.md "Numerical considerations" with the exact arithmetic:
//   II_SUM   (running sum of uint8 pixel values):    fits uint32 comfortably
//            (worst case: IMG_PIXELS * 255 ~= 18.2M, 25 bits).
//   II_SUMSQ (running sum of pixel VALUE SQUARED):    does NOT fit uint32 —
//            worst case IMG_PIXELS * 255^2 ~= 4.63e9 > UINT32_MAX (4.29e9).
//            uint64_t is mandatory for the whole-image corner entry.
static constexpr int II_W = IMG_W + 1;
static constexpr int II_H = IMG_H + 1;
static constexpr long long II_CELLS = static_cast<long long>(II_W) * II_H;   // 71,825

HD static inline int ii_index(int row, int col) { return row * II_W + col; }  // row in [0,IMG_H], col in [0,IMG_W]

// Window-statistics buffer: (S_w, S_ww) — the window's raw pixel sum and
// sum-of-squares over one TEMPLATE_SIZE x TEMPLATE_SIZE box — depends ONLY
// on (slot, offset), NOT on which template is being scored (SECTION 6's
// numerator is the only per-template quantity), so this buffer is far
// smaller than the score volume: 24 slots * 289 offsets = 6,936 entries.
// This is also the dedicated bit-exact-integer twin comparison (main.cu's
// VERIFY stage): a box query is pure integer arithmetic, so GPU and CPU
// results must match EXACTLY, not just to float tolerance.
static constexpr long long WINDOW_STATS_CELLS =
    static_cast<long long>(NUM_SLOTS) * NUM_OFFSETS_1D * NUM_OFFSETS_1D;      // 6,936

HD static inline long long window_stats_index(int slot, int oy, int ox)
{
    return (static_cast<long long>(slot) * NUM_OFFSETS_1D + oy) * NUM_OFFSETS_1D + ox;
}

// ===========================================================================
// SECTION 5 — template statistics (a SHARED data contract, computed once)
// ===========================================================================
// S_t[k] = sum of template k's TEMPLATE_PIXELS uint8 values (exact integer).
// S_tt[k] = sum of template k's pixel values SQUARED (exact integer).
// These are per-template CONSTANTS — fixed for the whole program run, and,
// per the SAME reasoning as project 01.13's fixed-point Hough theta table
// (kernels.cuh SECTION 5 there): computed ONCE, HERE, and handed identically
// to both the GPU path (uploaded to constant memory) and the CPU oracle, so
// that any GPU-vs-CPU disagreement in the final NCC score can only come from
// the WINDOW-side computation being verified — not from the two paths
// silently using two different constants for the same template.
//
// int64_t (not int32_t): TEMPLATE_PIXELS=576, so S_tt's worst case (a
// template of solid value 255) is 576*255^2 = 37,454,400 — actually fits
// uint32 alone, but int64_t is used uniformly with the window-side sums
// (SECTION 4's II_SUMSQ) for one consistent, overflow-safe type throughout
// the NCC algebra (THEORY.md works the exact arithmetic for every
// intermediate product, several of which DO need 64 bits — see SECTION 6).
static inline void compute_template_stats(const uint8_t* templates,   // [NUM_TEMPLATES*TEMPLATE_PIXELS]
                                          int64_t* S_t, int64_t* S_tt) // [NUM_TEMPLATES] OUT each
{
    for (int t = 0; t < NUM_TEMPLATES; ++t) {
        const uint8_t* p = templates + static_cast<size_t>(t) * TEMPLATE_PIXELS;
        int64_t sum = 0, sumsq = 0;
        for (int i = 0; i < TEMPLATE_PIXELS; ++i) {
            const int64_t v = p[i];
            sum += v;
            sumsq += v * v;
        }
        S_t[t] = sum;
        S_tt[t] = sumsq;
    }
}

// ===========================================================================
// SECTION 6 — the NCC algebra (documented once; kernels.cu and
// reference_cpu.cpp each implement it independently — see the file header).
// ===========================================================================
// Zero-normalized cross-correlation (ZNCC), derived in THEORY.md "The math"
// from the raw sums S_w = sum(w_i), S_t = sum(t_i), S_ww = sum(w_i^2),
// S_tt = sum(t_i^2), S_wt = sum(w_i*t_i) over the N = TEMPLATE_PIXELS
// paired window/template pixels:
//
//     numerator_unnorm = N*S_wt - S_w*S_t
//     var_w_unnorm      = N*S_ww - S_w*S_w          (>= 0 by Cauchy-Schwarz)
//     var_t_unnorm      = N*S_tt - S_t*S_t           (>= 0, same reason)
//     NCC = numerator_unnorm / sqrt(var_w_unnorm * var_t_unnorm)
//
// Every quantity up to and including numerator_unnorm/var_*_unnorm is EXACT
// INTEGER arithmetic (int64_t; THEORY.md works the overflow bounds) — only
// the final sqrt + divide is floating point. This is why main.cu can verify
// window statistics (S_w, S_ww) BIT-EXACT and only the final NCC ratio needs
// a float tolerance.
static constexpr float NCC_DENOM_EPS = 1e-6f;   // guards a (near-)flat window or template — see THEORY.md

#ifdef __CUDACC__  // ================= device-aware section (nvcc only) ====

// ---- Stage 1: build the whole-tray integral images (2-pass separable scan,
//      the SAME separable-pass idea project 01.13's Gaussian blur uses,
//      applied here to a running SUM instead of a weighted stencil — see
//      kernels.cu for the full derivation and the row/col kernel pair). ----
__global__ void integral_row_scan_kernel(const uint8_t* __restrict__ img,
                                         uint32_t* __restrict__ ii_sum,
                                         uint64_t* __restrict__ ii_sumsq);
__global__ void integral_col_scan_kernel(uint32_t* __restrict__ ii_sum,
                                         uint64_t* __restrict__ ii_sumsq);

// ---- Stage 2: window statistics (S_w, S_ww per slot x offset) via O(1)
//      box queries into the integral images — the dedicated bit-exact-
//      integer twin (SECTION 4). ------------------------------------------
__global__ void window_stats_kernel(const uint32_t* __restrict__ ii_sum,
                                    const uint64_t* __restrict__ ii_sumsq,
                                    uint32_t* __restrict__ ws_sum,
                                    uint64_t* __restrict__ ws_sumsq);

// ---- Stage 3a: NAIVE NCC — every thread recomputes S_w/S_ww itself by
//      directly re-scanning its TEMPLATE_SIZE^2 window (O(T^2) window-stat
//      work PER (slot,template,offset) triple — the redundancy the other
//      two variants remove). ------------------------------------------------
__global__ void ncc_naive_kernel(const uint8_t* __restrict__ img,
                                 const uint8_t* __restrict__ templates,
                                 float* __restrict__ scores);

// ---- Stage 3b: SUM-TABLE NCC — S_w/S_ww come from the O(1) integral-image
//      box query; the numerator S_wt still needs its own O(T^2) direct
//      correlation loop (unavoidable — see THEORY.md). ----------------------
__global__ void ncc_sumtable_kernel(const uint8_t* __restrict__ img,
                                    const uint32_t* __restrict__ ii_sum,
                                    const uint64_t* __restrict__ ii_sumsq,
                                    const uint8_t* __restrict__ templates,
                                    float* __restrict__ scores);

// ---- Stage 3c: SHARED-MEMORY NCC — same O(1) box query as (b), PLUS the
//      block (one (slot,template) pair, 289 offset-threads) cooperatively
//      caches its shared window region and template into shared memory
//      ONCE, so the O(T^2) numerator loop reads on-chip memory instead of
//      re-fetching the same overlapping global-memory bytes 289 times. -----
__global__ void ncc_shared_kernel(const uint8_t* __restrict__ img,
                                  const uint32_t* __restrict__ ii_sum,
                                  const uint64_t* __restrict__ ii_sumsq,
                                  const uint8_t* __restrict__ templates,
                                  float* __restrict__ scores);

#endif // __CUDACC__ ---------------------------------------------------------

// ---- Host launch wrappers (own grid/block math + post-launch error check).
void upload_template_stats(const int64_t* S_t, const int64_t* S_tt);   // -> __constant__ memory, once
void launch_build_integral_images(const uint8_t* d_img, uint32_t* d_ii_sum, uint64_t* d_ii_sumsq);
void launch_window_stats(const uint32_t* d_ii_sum, const uint64_t* d_ii_sumsq,
                         uint32_t* d_ws_sum, uint64_t* d_ws_sumsq);
void launch_ncc_naive(const uint8_t* d_img, const uint8_t* d_templates, float* d_scores);
void launch_ncc_sumtable(const uint8_t* d_img, const uint32_t* d_ii_sum, const uint64_t* d_ii_sumsq,
                         const uint8_t* d_templates, float* d_scores);
void launch_ncc_shared(const uint8_t* d_img, const uint32_t* d_ii_sum, const uint64_t* d_ii_sumsq,
                       const uint8_t* d_templates, float* d_scores);

// ---- CPU reference oracle (reference_cpu.cpp) — INDEPENDENT reimplementation
//      of every twinned stage; see that file's header for the independence
//      ruling and what is/is not twinned. -------------------------------
void build_integral_images_cpu(const uint8_t* img, uint32_t* ii_sum, uint64_t* ii_sumsq);
void window_stats_cpu(const uint32_t* ii_sum, const uint64_t* ii_sumsq,
                      uint32_t* ws_sum, uint64_t* ws_sumsq);
// The CPU oracle scores the WHOLE (slot,template,offset) volume via the SAME
// integral-image + box-query approach as the GPU sum-table/shared variants
// (an independently-typed twin of that structure — see the file header) so
// it is a meaningful correctness oracle for all three GPU kernels at once.
void ncc_scores_cpu(const uint8_t* img, const uint32_t* ii_sum, const uint64_t* ii_sumsq,
                    const uint8_t* templates, const int64_t* S_t, const int64_t* S_tt,
                    float* scores);

#endif // PROJECT_KERNELS_CUH
