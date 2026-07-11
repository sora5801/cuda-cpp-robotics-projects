// ===========================================================================
// kernels.cuh — kernel & reference declarations + the single-sourced data
//               contract for project 01.15 (Background subtraction for
//               fixed-workspace cells)
//
// Role in the project
// --------------------
// This header is the ONE place three per-pixel background models, their
// event-driven test sequence, and their state layouts are defined, so the
// GPU kernels (kernels.cu), the CPU oracle (reference_cpu.cpp), and the
// orchestrator (main.cu) can never silently drift apart (CLAUDE.md §12 —
// "every state vector documents its layout in one place").
//
// Why ".cuh" and the __CUDACC__ fence
// ------------------------------------
// reference_cpu.cpp is compiled by cl.exe, which does not understand
// __global__; kernels.cu and main.cu are compiled by nvcc, which defines the
// __CUDACC__ macro. Device-only declarations are therefore fenced so the
// same header serves both translation units (the repo-wide trick — see
// docs/PROJECT_TEMPLATE/src/kernels.cuh).
//
// THE THREE MODELS (one teaching-focused kernel per concept; see THEORY.md
// "The algorithm" for the full derivation of each):
//   1. FRAME DIFFERENCING   — |I(t) - reference| > threshold. Stateless,
//      no adaptation. The naive baseline, DESIGNED to fail under slow
//      illumination drift (event E3 below).
//   2. RUNNING SINGLE GAUSSIAN — one adaptive (mean, variance) pair per
//      pixel, updated every frame by exponential moving average (EMA).
//      Adapts to drift; cannot represent two legitimately different
//      "background" brightnesses at the same pixel (event E4 below).
//   3. MOG-LITE (K=3)        — a small Gaussian mixture per pixel
//      (Stauffer & Grimson 1999, simplified). Represents up to three
//      recurring appearances per pixel; this is the project's didactic
//      heart (README "The algorithm in brief").
//
// THE DESIGNED SEQUENCE — every geometric/temporal constant a gate needs to
// build ground truth lives below in one place. scripts/make_synthetic.py
// independently TRANSCRIBES the same numeric values (Python cannot #include
// a .cuh file) to render the committed frames; each constant below carries
// a "keep in sync with make_synthetic.py" note, and vice versa in that
// file. This is the SAME kind of cross-language constant duplication
// project 01.04's checkerboard-geometry precedent uses — the values are
// data, not an algorithm, so duplicating them is not the twin-independence
// concern the reference_cpu.cpp ruling addresses (see that file's header).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>   // std::floor for round_half_up() below — host-only math, safe under both cl.exe and nvcc

// ===========================================================================
// SECTION 1 — image & sequence geometry (the data-layout contract, §12)
// ===========================================================================

// Frame size. 128x96, not the catalog-illustrative 240x180: reduced so the
// FULL 160-frame sequence commits at ~1.9 MiB instead of ~6.8 MiB (the
// project brief's two allowed choices — see data/README.md "Size decision"
// for the exact byte math). Every project stays self-contained (CLAUDE.md
// §4): the committed sample IS the whole demo input, no download, no
// Python at run time.
static const int IMG_W = 128;                    // pixels, columns (x)
static const int IMG_H = 96;                      // pixels, rows (y)
static const int IMG_N = IMG_W * IMG_H;            // pixels per frame = 12,288
static const int SEQ_T = 160;                      // frames in the sequence (t = 0..159)

// px_index — row-major flat index, shared by EVERY piece of code that
// touches a per-pixel array (CPU, GPU, and main.cu's gating logic). This is
// a data-layout formula, not an algorithm — sharing it is exactly what the
// reference_cpu.cpp independence ruling calls out as mandatory, not
// optional (duplicating "y*W+x" in three files would be a bug farm, not
// independence). __host__ __device__ so nvcc emits both a host and a device
// version from one definition; cl.exe (host-only) just sees a normal
// inline function because __device__ expands to nothing outside __CUDACC__
// (see the CUDA runtime headers' own convention).
#ifdef __CUDACC__
__host__ __device__
#endif
inline int px_index(int x, int y) { return y * IMG_W + x; }

// ===========================================================================
// SECTION 2 — background regions & the designed event schedule
//
// Every rectangle is [x0, x0+w) x [y0, y0+h) — half-open, so width/height
// are literal pixel counts (no off-by-one at the far edge). All five
// designed events (E1..E5) are described in README "The algorithm in
// brief" and THEORY.md "The problem"; the numbers here are their EXACT
// ground truth, used both to RENDER the sequence (make_synthetic.py) and
// to GATE the three models' output (main.cu) — one set of numbers, two
// consumers, cross-referenced rather than shared at compile time because
// the consumers are in different languages.
// ===========================================================================

// ---- static backdrop (textured, not an event — always present) ----------
// Bench: the work surface the arm/box interact with. Fixtures: small dark
// mounted hardware. "Wall" is everything outside the bench rectangle.
// Mean intensities and the per-region hashed-texture half-range are used
// ONLY by make_synthetic.py (rendering); main.cu never needs them because
// it only ever looks at pixel VALUES it already has, never re-derives them.
static const int BENCH_X = 5,  BENCH_Y = 55, BENCH_W = 118, BENCH_H = 35;
static const int FIX1_X  = 15, FIX1_Y  = 60, FIX1_W  = 20,  FIX1_H  = 8;
static const int FIX2_X  = 95, FIX2_Y  = 62, FIX2_W  = 15,  FIX2_H  = 10;

// ---- E1 / E5: the arm intrusion (two sweeps of the SAME two-link chain) -
// A simplified "articulated" chain: link A (the sweeping rectangle whose
// x position is linearly interpolated across the event's frame range) plus
// link B, a second rectangle rigidly offset from A by (ARM_B_DX, ARM_B_DY).
// Honest simplification (documented again in README "Limitations"): the
// two links translate together rather than rotating about a joint — a true
// revolute chain needs rotated-rectangle rasterization, which would only
// complicate the ground-truth mask without adding to the background-
// subtraction lesson this project teaches. "Two co-moving rectangles" still
// gives an exact, non-convex, per-frame silhouette to test IoU against.
static const int   E1_FRAME_START = 20,  E1_FRAME_END = 50;    // inclusive, 31 frames
static const int   E5_FRAME_START = 130, E5_FRAME_END = 150;   // inclusive, 21 frames — AFTER drift has accumulated
static const int   ARM_A_W = 14, ARM_A_H = 10;
static const int   ARM_B_W = 10, ARM_B_H = 8;
static const int   ARM_B_DX = 16, ARM_B_DY = 6;                // link B's offset from link A's top-left corner
static const int   E1_ARM_Y = 40,  E1_X_START = 10, E1_X_END = 100;  // link A's row and x-sweep for E1
static const int   E5_ARM_Y = 15,  E5_X_START = 30, E5_X_END = 80;   // link A's row and x-sweep for E5 — capped at 80 (not 100) so link B's offset rectangle never reaches the LAMP_X=100 panel (checked by hand: max link-B x = 80+ARM_B_DX+ARM_B_W = 106, but link B's y-band [21,29) never overlaps the lamp's [10,18) either way; the real guard is link A itself, whose y=15 row DOES fall inside the lamp's y-band, so its x must stay clear of [100,110) — 80+14=94 keeps a 6px margin)

// ---- E2: the absorption test — a box PLACED at frame 60 and left there --
static const int E2_FRAME_PLACED = 60;                          // first frame the box exists (persists to SEQ_T-1)
static const int BOX_X = 70, BOX_Y = 70, BOX_W = 18, BOX_H = 14;

// ---- E4: the bimodal lesson — a status lamp blinking every 8 frames -----
// Chosen well clear of every arm/box rectangle above (checked by hand at
// design time; main.cu's gating logic does not need to re-verify this).
static const int LAMP_X = 100, LAMP_Y = 10, LAMP_W = 10, LAMP_H = 8;
static const int LAMP_PERIOD_FRAMES = 8;   // state flips every 8 frames -> a 16-frame full cycle

// ---- E3: the illumination ramp — a uniform +15% linear brightness drift -
// Applied multiplicatively to every pixel INCLUDING the lamp panel, over
// the whole sequence (see THEORY.md "The problem" for the two honest
// physical readings of a uniform ramp: ambient drift or camera auto-gain
// creep). L(t) = 1 + ILLUM_RAMP_FRAC * t / (SEQ_T - 1).
static const float ILLUM_RAMP_FRAC = 0.15f;

// ---- gate frame ranges (event-free windows used to measure false positives)
static const int NOISE_FLOOR_FRAME_LO = 2,   NOISE_FLOOR_FRAME_HI = 15;   // early, drift negligible
static const int DRIFT_LATE_FRAME_LO  = 152, DRIFT_LATE_FRAME_HI  = 159;  // AFTER E5 ends (150): the last, most-drifted event-free window (L(t) = 1.143..1.150)
static const int BIMODAL_FRAME_LO     = 32,  BIMODAL_FRAME_HI     = 159;  // after >=2 lamp cycles have been observed

// is_in_rect — axis-aligned half-open rectangle membership test, shared by
// main.cu (building ground truth for every gate) and, conceptually, by
// make_synthetic.py's identical Python test used at generation time. Pure
// arithmetic, not the system under test — see this file's header.
inline bool is_in_rect(int x, int y, int rx, int ry, int rw, int rh)
{
    return x >= rx && x < rx + rw && y >= ry && y < ry + rh;
}

// round_half_up — floor(v + 0.5). The SAME three floating-point operations
// scripts/make_synthetic.py's Python twin performs (see that script's
// comment for why floor(v+0.5) was chosen over either language's native
// round(): Python's round() is round-half-to-even, C++'s std::lround is
// round-half-away-from-zero — they usually but not PROVABLY agree, whereas
// this formula is bit-identical arithmetic in both languages). Used to
// turn arm_link_a_x()'s exact interpolated position into the same integer
// pixel column the Python renderer drew.
inline int round_half_up(double v) { return static_cast<int>(std::floor(v + 0.5)); }

// arm_link_a_x — link A's left-edge x at frame t during an active arm
// event, linearly interpolated across [x_start, x_end] over the event's
// inclusive frame range, ROUNDED via round_half_up (E5's 50px/20-frame
// sweep passes through half-integer positions; E1's does not, but the
// same formula handles both uniformly).
inline int arm_link_a_x(int t, int frame_start, int frame_end, int x_start, int x_end)
{
    const double span = static_cast<double>(frame_end - frame_start);  // e.g. 30 for E1
    const double frac = (span > 0.0) ? (static_cast<double>(t - frame_start) / span) : 0.0;
    return round_half_up(x_start + frac * (x_end - x_start));
}

// ===========================================================================
// SECTION 3 — model parameters (the ONE place every threshold/rate lives;
// README "The algorithm in brief" and THEORY.md "The math" name and derive
// each of these).
// ===========================================================================

// ---- Model 1: frame differencing -----------------------------------------
// 12.0 (not a larger, noise-safer-looking number): the E3 illumination ramp
// tops out at +15% on a ~140-intensity bench (a ~21-unit drift by the last
// frame), and this project's illumination_drift gate specifically wants
// frame-diff to FAIL under that drift — see THEORY.md "How we verify
// correctness". 12.0 sits comfortably above the noise-only false-positive
// floor (two independent NOISE_SIGMA=3.0 samples differencing gives
// sigma_diff = sqrt(2)*3 = 4.24, so 12.0 is ~2.8 noise-sigma) while sitting
// well BELOW the drift-only signal by the sequence's final quarter —
// exactly the designed failure mode, not a hair-trigger threshold.
static const float FRAME_DIFF_THRESHOLD = 12.0f;   // |I - reference| > this => foreground (intensity units, 0..255 scale)

// ---- Model 2: running single Gaussian -------------------------------------
static const float SG_ALPHA     = 0.08f;  // EMA learning rate for BOTH mean and variance (time constant 1/alpha ~= 12.5 frames)
static const float SG_K_SIGMA   = 2.5f;   // foreground test: |I - mu| > SG_K_SIGMA * sigma
static const float SG_VAR_INIT  = 16.0f;  // initial variance (sigma_init = 4.0 intensity units) — a deliberately generous guess
static const float SG_VAR_FLOOR = 9.0f;   // variance floor (sigma_floor = 3.0) — matches the synthetic sensor noise sigma; see THEORY.md "Numerical considerations"
// SG_VAR_CEIL — a variance CEILING on the stored EMA variance itself (not
// just the sigma used at classification time, which SG_VAR_FLOOR already
// bounds from below). Discovered empirically while building this project
// (see THEORY.md "Numerical considerations" for the full story): the
// BLIND update feeds var[i] = (1-alpha)*var + alpha*diff*diff EVERY frame,
// including the very frame a genuinely new object appears — one huge diff
// (e.g. box-vs-bench, ~69 intensity units) immediately balloons var by
// alpha*diff^2 (here, ~380), which INFLATES sigma, which RAISES the
// k_sigma*sigma detection threshold — a runaway that "absorbs" the new
// object in 1-2 frames via desensitization rather than the intended ~20
// frames of genuine mean convergence, and (worse) makes the model nearly
// blind to the very next real event. Capping storage at SG_VAR_CEIL =
// (6.0)^2 = 36 caps sigma at 6.0 intensity units (2x the noise floor's 3.0)
// — enough headroom for legitimate variance growth, not enough to let one
// outlier neuter the detector. The absorption-time closed form in
// THEORY.md "The math" assumes this ceiling saturates on the very first
// post-event frame (true whenever alpha*diff0^2 > SG_VAR_CEIL, which holds
// for every designed event in this sequence) and therefore treats sigma as
// the CONSTANT sqrt(SG_VAR_CEIL) for the whole foreground phase.
static const float SG_VAR_CEIL  = 36.0f;

// ---- Model 3: MOG-lite (K=3) -----------------------------------------------
static const int   MOG_K              = 3;     // modes per pixel — "lite": Stauffer-Grimson's original paper suggests 3-5
static const float MOG_MATCH_K_SIGMA  = 2.5f;  // match-to-nearest-mode test, same sigma multiple as the single-Gaussian model (fair comparison)
static const float MOG_LR_WEIGHT      = 0.05f; // weight EMA learning rate (slower than the mean/var rate — see THEORY.md)
static const float MOG_LR_PARAM       = 0.08f; // matched mode's mean/var EMA rate — deliberately EQUAL to SG_ALPHA so the two models are apples-to-apples on adaptation speed
static const float MOG_VAR_INIT       = 16.0f; // a freshly-created mode's initial variance (matches SG_VAR_INIT)
static const float MOG_VAR_FLOOR      = 9.0f;  // matches SG_VAR_FLOOR
static const float MOG_W_INIT_NEW     = 0.05f; // a freshly-created (replace-weakest) mode's initial weight
static const float MOG_BG_FRACTION    = 0.8f;  // "T" in Stauffer-Grimson: modes are background until their sorted cumulative weight reaches this

// ===========================================================================
// SECTION 4 — state layout (documented once, obeyed everywhere)
//
//   Single-Gaussian state: two parallel float arrays, mu[IMG_N], var[IMG_N],
//   row-major (px_index above). Plain AoS-of-one-field-each — there is only
//   one mode, so there is no "which layout" question.
//
//   MOG state: THREE float arrays, weight[MOG_K*IMG_N], mean[MOG_K*IMG_N],
//   var[MOG_K*IMG_N], laid out MODE-MAJOR: element (k, pixel) lives at
//   k*IMG_N + pixel — i.e. all of mode 0's pixels first, then all of mode
//   1's, then mode 2's. This is a Structure-of-Arrays choice, and mode-major
//   (not pixel-major-with-3-consecutive-floats) is the one that COALESCES:
//   one GPU thread owns one PIXEL and reads/writes mode k at a fixed
//   k*IMG_N + pixel offset. Adjacent threads (adjacent pixel index) then
//   touch ADJACENT addresses for the SAME mode at the SAME instant — the
//   classic 128-byte-warp coalescing pattern (same argument as the SAXPY
//   placeholder's memory-behavior note, extended to a 3-mode struct). A
//   pixel-major layout (mode 0/1/2 interleaved per pixel) would instead
//   scatter one warp's reads 3 floats apart per thread — still technically
//   coalesced as ONE 96-byte-stride access, but it forces the compiler to
//   issue 3x as many transactions for a full K-mode scan versus 3 clean
//   128-byte reads. See THEORY.md "The GPU mapping" for the full argument.
// ===========================================================================

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// frame_diff_kernel — stateless MAP over every (frame, pixel) pair at once.
// See kernels.cu for the full doc-comment (thread mapping, why one launch
// covers the whole T*N array).
__global__ void frame_diff_kernel(const unsigned char* __restrict__ frames,
                                   const unsigned char* __restrict__ reference,
                                   unsigned char* __restrict__ mask_out,
                                   int n_pixels, int total_elems, float threshold);

// sg_step_kernel — ONE frame's worth of single-Gaussian classify+update,
// one thread per pixel. Called SEQUENTIALLY, once per frame, from the host
// loop in main.cu (state at frame t depends on frame t-1 — this is NOT a
// map over frames, only over pixels within one frame).
__global__ void sg_step_kernel(const unsigned char* __restrict__ frame_t,
                                float* __restrict__ mu,
                                float* __restrict__ var,
                                unsigned char* __restrict__ mask_out,
                                int n_pixels,
                                float alpha, float k_sigma, float var_floor, float var_ceil);

// mog_step_kernel — ONE frame's worth of MOG-lite classify+update, one
// thread per pixel, same sequential-launch discipline as sg_step_kernel.
__global__ void mog_step_kernel(const unsigned char* __restrict__ frame_t,
                                 float* __restrict__ weight,
                                 float* __restrict__ mean,
                                 float* __restrict__ var,
                                 unsigned char* __restrict__ mask_out,
                                 int n_pixels, int mog_k,
                                 float match_k_sigma, float lr_weight, float lr_param,
                                 float var_init, float var_floor,
                                 float w_init_new, float bg_fraction);

// morph_erode_kernel / morph_dilate_kernel — the 3x3, 8-connected,
// zero-padded structuring element used for morphological OPENING (erode
// then dilate), applied to ALL THREE models' raw masks before every gate
// (README "Post-processing"). Deliberately the SAME convention project
// 30.01 (agriculture) ratified for its fruit-mask cleanup stage — see that
// project's kernels.cu for the precedent this project follows rather than
// reinvents. Grid-stride over the WHOLE T*N mask array at once: unlike the
// two state-carrying kernels above, morphology has NO cross-frame
// dependency (frame t's opening only reads frame t's raw mask), so one
// launch cleans every frame in the sequence.
__global__ void morph_erode_kernel(const unsigned char* __restrict__ mask_in,
                                    unsigned char* __restrict__ mask_out,
                                    int w, int h, int total_elems);
__global__ void morph_dilate_kernel(const unsigned char* __restrict__ mask_in,
                                     unsigned char* __restrict__ mask_out,
                                     int w, int h, int total_elems);

#endif // __CUDACC__ --------------------------------------------------------

// ---- host launch wrappers (callable from any translation unit) -----------
void launch_frame_diff(const unsigned char* d_frames, const unsigned char* d_reference,
                        unsigned char* d_mask, int n_pixels, int total_elems, float threshold);
void launch_sg_step(const unsigned char* d_frame_t, float* d_mu, float* d_var,
                     unsigned char* d_mask_out, int n_pixels,
                     float alpha, float k_sigma, float var_floor, float var_ceil);
void launch_mog_step(const unsigned char* d_frame_t, float* d_weight, float* d_mean, float* d_var,
                      unsigned char* d_mask_out, int n_pixels, int mog_k,
                      float match_k_sigma, float lr_weight, float lr_param,
                      float var_init, float var_floor, float w_init_new, float bg_fraction);
void launch_morph_open(const unsigned char* d_mask_raw, unsigned char* d_mask_open,
                        unsigned char* d_scratch, int w, int h, int total_elems);

// ---- CPU reference twins (defined independently in reference_cpu.cpp —
// see that file's header for the "written twice, on purpose" ruling). Same
// signatures modulo device/host pointers so main.cu can call either path
// through nearly-identical code. ----------------------------------------
void frame_diff_cpu(const unsigned char* frames, const unsigned char* reference,
                     unsigned char* mask_out, int n_pixels, int total_elems, float threshold);
void sg_step_cpu(const unsigned char* frame_t, float* mu, float* var,
                  unsigned char* mask_out, int n_pixels,
                  float alpha, float k_sigma, float var_floor, float var_ceil);
void mog_step_cpu(const unsigned char* frame_t, float* weight, float* mean, float* var,
                   unsigned char* mask_out, int n_pixels, int mog_k,
                   float match_k_sigma, float lr_weight, float lr_param,
                   float var_init, float var_floor, float w_init_new, float bg_fraction);
void morph_open_cpu(const unsigned char* mask_raw, unsigned char* mask_open,
                     int w, int h, int total_elems);

#endif // PROJECT_KERNELS_CUH
