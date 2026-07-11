// ===========================================================================
// main.cu — entry point for project 01.15 (Background subtraction for
//           fixed-workspace cells)
//
// Role in the project
// --------------------
// Orchestrates the whole demo: load the committed 160-frame sequence, run
// all three background models (frame differencing, running single
// Gaussian, MOG-lite) on BOTH the CPU oracle and the GPU, clean every raw
// mask with the shared 3x3 morphological open, verify GPU==CPU within
// documented tolerance, then run FIVE independent gates that check the
// MODELS' OWN OUTPUT against ground truth this file derives directly from
// the designed-event schedule (kernels.cuh SECTION 2) — none of which
// route through the GPU-vs-CPU twin comparison (CLAUDE.md's reference_cpu
// ruling: twin agreement proves the parallelization is faithful; it says
// nothing about whether the reference itself is measuring the right
// thing — these gates are that second, independent check).
//
// Pipeline
// --------
//   1. Load data/sample/frames/frame_NNN.pgm x 160 into one flat buffer.
//   2. Run CPU: frame-diff (stateless), single-Gaussian (SEQ_T-1 sequential
//      steps), MOG-lite (SEQ_T-1 sequential steps), then morphological open
//      on all three raw masks.
//   3. Run GPU: identical pipeline, device buffers, one kernel LAUNCH per
//      frame for the two adaptive models (state dependency forces this —
//      see kernels.cu), one launch total for frame-diff and for each
//      morphology stage (no cross-frame dependency there).
//   4. VERIFY: GPU raw masks match CPU raw masks pixel-for-pixel (within a
//      documented rare-tie allowance) and final model state (mu/var,
//      weight/mean/var) agrees within a measured-then-margined tolerance.
//   5. GATE: intrusion_detection, illumination_drift, absorption (the
//      analytic one), bimodal_lesson, noise_floor — see each gate function
//      below for its derivation.
//   6. ARTIFACTS: demo/out/{mask_strip_frameNNN.ppm x3, absorption_curve.csv,
//      fp_rate_timeline.csv, gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", every "VERIFY(...)"/"GATE ...:" verdict line, "ARTIFACT:", and
// "RESULT:" — PASS/FAIL text only, no embedded numbers, so they are
// byte-identical on any GPU. Measured numbers live on "[info]"/"[time]"
// lines, deliberately NOT diffed by demo/run_demo.* (the 01.01/01.04
// convention, followed here too).
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ===========================================================================
// Gate / verify tolerances.
//
//   Measured on the reference machine (RTX 2080 SUPER, sm_75), Release,
//   the committed 160-frame sample (see README "Expected output" /
//   THEORY.md "How we verify correctness" for the full derivation of each
//   number below from these measurements):
//     verify(frame-diff raw mask)      0 / 1,966,080 elements differ (bit-exact: stateless, no accumulation)
//     verify(single-Gaussian raw mask) 0 / 1,955,328 differ; final max|gpu-cpu| mu=0.000000, var=0.000008
//     verify(MOG-lite raw mask)        0 / 1,955,328 differ; final max|gpu-cpu| weight=0.000000, mean=0.000000, var=0.000008
//     gate intrusion_detection         mean IoU: SG=0.6075, MOG=0.5072 (frame-diff=0.6221, reported only)
//     gate illumination_drift          late-window FP rate: SG=0.0172, MOG=0.0000, frame-diff=0.3151
//     gate absorption                  predicted=19 frames, measured=18 frames (SG); MOG measured=2 frames ([info], no closed form)
//     gate bimodal_lesson              lamp-region FP rate: MOG=0.0000, SG=1.0000
//     gate noise_floor                 early-window FP rate: SG=0.0000, MOG=0.0004
//   Every tolerance below is that measurement with a stated margin — never
//   a number invented before the fact (CLAUDE.md §12: never fabricate).
// ===========================================================================

// ---- twin-comparison tolerances -------------------------------------------
static constexpr double kTwinMaskMismatchFrac = 0.0005;  // ceiling on the FRACTION of (frame,pixel) raw-mask disagreements between GPU and CPU, for EACH adaptive model — measured 0.0 on the reference GPU; the ceiling stays non-zero (see "Numerical considerations" in THEORY.md) because float-order divergence over 159 sequential EMA steps COULD flip a classification that lands within ~1 ULP of a threshold on a different compute capability
static constexpr double kTwinStateAbsTol   = 0.01;        // ceiling on max|gpu-cpu| for any single mu/var/weight/mean state value after the full sequence — measured 0.000008, ~1250x margin

// ---- intrusion_detection gate ----------------------------------------------
static constexpr double kIntrusionIoUFloor = 0.35;   // mean IoU (SG, MOG) over all E1+E5 frames must be >= this — measured SG=0.6075, MOG=0.5072

// ---- illumination_drift gate -----------------------------------------------
static constexpr double kDriftFPCeilingAdaptive = 0.03;  // SG, MOG: late-drift FP rate must be <= this — measured SG=0.0172, MOG=0.0000
static constexpr double kDriftFPFloorFrameDiff  = 0.10;  // frame-diff: late-drift FP rate must EXCEED this (the designed failure) — measured 0.3151, 3x the floor

// ---- absorption gate (the analytic one) ------------------------------------
static constexpr int    kAbsorptionConfirmWindow = 5;    // consecutive event-free-of-flicker frames required before declaring "absorbed"
static constexpr double kAbsorptionFactorLo = 0.5;        // measured/predicted must land in [Lo, Hi] — measured 18 / predicted 19 = 0.947, comfortably inside [0.5, 2.0]
static constexpr double kAbsorptionFactorHi = 2.0;

// ---- bimodal_lesson gate ----------------------------------------------------
static constexpr double kBimodalFPCeilingMog = 0.20;  // MOG: FP rate AT THE LAMP must be <= this (it learns both states) — measured 0.0000
static constexpr double kBimodalFPFloorSg    = 0.30;  // single-Gaussian: FP rate AT THE LAMP must EXCEED this (it cannot represent two modes) — measured 1.0000

// ---- noise_floor gate --------------------------------------------------------
static constexpr double kNoiseFloorFPCeiling = 0.02;  // SG, MOG: FP rate in the early, event-free, drift-negligible window must be <= this — measured SG=0.0000, MOG=0.0004

// ===========================================================================
// Minimal, STRICT PGM (P5) reader / PPM (P6) writer — this project only
// ever reads files its own scripts/make_synthetic.py wrote (same discipline
// as 01.04's read_pgm — see that file for the precedent).
// ===========================================================================
static bool read_pgm(const std::string& path, int& W, int& H, std::vector<unsigned char>& data)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    std::string magic;
    in >> magic;
    if (magic != "P5") return false;
    int maxval = 0;
    in >> W >> H >> maxval;
    if (!in || maxval != 255 || W <= 0 || H <= 0) return false;
    in.get();   // the single mandatory whitespace byte after maxval
    data.resize(static_cast<size_t>(W) * static_cast<size_t>(H));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return in.gcount() == static_cast<std::streamsize>(data.size());
}

static bool write_ppm(const std::string& path, int W, int H, const std::vector<unsigned char>& rgb)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P6\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(out);
}

// ===========================================================================
// Sequence loading — 160 committed frames, each individually located via
// util/paths.h's multi-candidate find_data_file() (works from the VS
// layout, both run_demo scripts, and a CMake build alike — see that
// header's file comment).
// ===========================================================================
static bool load_sequence(const std::string& cli_dir, const char* argv0,
                           std::vector<unsigned char>& frames)
{
    frames.resize(static_cast<size_t>(SEQ_T) * static_cast<size_t>(IMG_N));
    for (int t = 0; t < SEQ_T; ++t) {
        char name[48];
        std::snprintf(name, sizeof(name), "frames/frame_%03d.pgm", t);
        const std::string path = find_data_file(cli_dir, argv0, name);
        if (path.empty()) {
            std::fprintf(stderr, "error: could not find data/sample/%s (looked in every candidate directory paths.h knows)\n", name);
            return false;
        }
        int W = 0, H = 0;
        std::vector<unsigned char> data;
        if (!read_pgm(path, W, H, data) || W != IMG_W || H != IMG_H) {
            std::fprintf(stderr, "error: %s failed to parse as a %dx%d P5 PGM\n", path.c_str(), IMG_W, IMG_H);
            return false;
        }
        std::copy(data.begin(), data.end(), frames.begin() + static_cast<long>(static_cast<size_t>(t) * IMG_N));
    }
    return true;
}

// ===========================================================================
// Ground-truth helpers (host-only; independent of the models under test —
// this is the "gate independence" this project's verification story relies
// on, mirroring 01.04's forward_transform() precedent). Every geometric
// number comes from kernels.cuh SECTION 2, the single-sourced contract.
// ===========================================================================

// An axis-aligned rectangle, used only for ground-truth bookkeeping here.
struct Rect { int x, y, w, h; };

// active_arm_rects — the two-link arm's ground-truth footprint at frame t,
// or an empty vector if neither E1 nor E5 is active. E1 and E5 never
// overlap in frame range (kernels.cuh SECTION 2), so at most one event's
// rectangles are ever returned.
static std::vector<Rect> active_arm_rects(int t)
{
    std::vector<Rect> out;
    struct Event { int fs, fe, ay, xs, xe; };
    const Event events[2] = {
        {E1_FRAME_START, E1_FRAME_END, E1_ARM_Y, E1_X_START, E1_X_END},
        {E5_FRAME_START, E5_FRAME_END, E5_ARM_Y, E5_X_START, E5_X_END},
    };
    for (const Event& e : events) {
        if (t < e.fs || t > e.fe) continue;
        const int ax = arm_link_a_x(t, e.fs, e.fe, e.xs, e.xe);
        out.push_back(Rect{ax, e.ay, ARM_A_W, ARM_A_H});
        out.push_back(Rect{ax + ARM_B_DX, e.ay + ARM_B_DY, ARM_B_W, ARM_B_H});
    }
    return out;
}

static inline bool pixel_in_any(int x, int y, const std::vector<Rect>& rects)
{
    for (const Rect& r : rects) if (is_in_rect(x, y, r.x, r.y, r.w, r.h)) return true;
    return false;
}
static inline bool pixel_in_lamp(int x, int y) { return is_in_rect(x, y, LAMP_X, LAMP_Y, LAMP_W, LAMP_H); }
static inline bool pixel_in_box(int x, int y)  { return is_in_rect(x, y, BOX_X, BOX_Y, BOX_W, BOX_H); }

// ===========================================================================
// Model state + CPU/GPU runners. Each runner produces a RAW (pre-morphology)
// 0/1 mask over the whole [SEQ_T * IMG_N] sequence. Frame 0's mask is
// always 0 (background) for the two adaptive models: their state is
// INITIALIZED from frame 0's own pixel values, so classifying frame 0
// against itself is trivially background and carries no information — the
// sequential update loop therefore runs t = 1 .. SEQ_T-1 (see kernels.cuh
// sg_step_kernel / mog_step_kernel doc-comments).
// ===========================================================================

struct SgState { std::vector<float> mu, var; };
struct MogState { std::vector<float> weight, mean, var; };

static void sg_init(const unsigned char* frame0, SgState& st)
{
    st.mu.assign(static_cast<size_t>(IMG_N), 0.0f);
    st.var.assign(static_cast<size_t>(IMG_N), SG_VAR_INIT);
    for (int i = 0; i < IMG_N; ++i) st.mu[static_cast<size_t>(i)] = static_cast<float>(frame0[i]);
}

static void mog_init(const unsigned char* frame0, MogState& st)
{
    const size_t total = static_cast<size_t>(MOG_K) * static_cast<size_t>(IMG_N);
    st.weight.assign(total, 0.0f);
    st.mean.assign(total, 0.0f);
    st.var.assign(total, MOG_VAR_INIT);
    for (int i = 0; i < IMG_N; ++i) {
        // Mode 0 starts as the ENTIRE observed distribution (weight 1); modes
        // 1 and 2 start unused (weight 0) — see kernels.cuh SECTION 4 and
        // mog_step_kernel's doc-comment for how "replace weakest" fills them
        // in as the sequence introduces genuinely new appearances.
        st.weight[static_cast<size_t>(0 * IMG_N + i)] = 1.0f;
        st.mean[static_cast<size_t>(0 * IMG_N + i)]   = static_cast<float>(frame0[i]);
        for (int k = 1; k < MOG_K; ++k) {
            st.mean[static_cast<size_t>(k * IMG_N + i)] = static_cast<float>(frame0[i]);
        }
    }
}

// ---- CPU pipeline -----------------------------------------------------------
static void run_frame_diff_cpu(const std::vector<unsigned char>& frames, std::vector<unsigned char>& mask)
{
    mask.assign(frames.size(), 0);
    frame_diff_cpu(frames.data(), frames.data() /* reference = frame 0 */, mask.data(),
                   IMG_N, static_cast<int>(frames.size()), FRAME_DIFF_THRESHOLD);
}

static void run_sg_cpu(const std::vector<unsigned char>& frames, std::vector<unsigned char>& mask,
                        SgState& st, std::vector<float>& mu_pre_box_snapshot, std::vector<float>& var_pre_box_snapshot)
{
    mask.assign(frames.size(), 0);
    sg_init(frames.data(), st);
    for (int t = 1; t < SEQ_T; ++t) {
        if (t == E2_FRAME_PLACED) {
            // Snapshot state EXACTLY as it enters the box-placement frame —
            // this is the "mu_pre" the absorption gate's closed form needs
            // (see compute_absorption_gate below). Only the box region is
            // needed, but snapshotting the whole array is cheap (IMG_N
            // floats) and keeps this call-site simple.
            mu_pre_box_snapshot = st.mu;
            var_pre_box_snapshot = st.var;
        }
        sg_step_cpu(&frames[static_cast<size_t>(t) * IMG_N], st.mu.data(), st.var.data(),
                    &mask[static_cast<size_t>(t) * IMG_N], IMG_N, SG_ALPHA, SG_K_SIGMA, SG_VAR_FLOOR, SG_VAR_CEIL);
    }
}

static void run_mog_cpu(const std::vector<unsigned char>& frames, std::vector<unsigned char>& mask, MogState& st)
{
    mask.assign(frames.size(), 0);
    mog_init(frames.data(), st);
    for (int t = 1; t < SEQ_T; ++t) {
        mog_step_cpu(&frames[static_cast<size_t>(t) * IMG_N], st.weight.data(), st.mean.data(), st.var.data(),
                     &mask[static_cast<size_t>(t) * IMG_N], IMG_N, MOG_K,
                     MOG_MATCH_K_SIGMA, MOG_LR_WEIGHT, MOG_LR_PARAM, MOG_VAR_INIT, MOG_VAR_FLOOR,
                     MOG_W_INIT_NEW, MOG_BG_FRACTION);
    }
}

// ---- GPU pipeline ------------------------------------------------------------
// All three models share ONE device copy of the frame sequence (d_frames);
// each model owns its own device state + mask buffers. The adaptive models'
// per-frame kernels are launched in a host loop into the DEFAULT stream, so
// CUDA's own in-order-per-stream guarantee serializes frame t's launch
// after frame t-1's — no explicit cudaDeviceSynchronize is needed between
// iterations (see kernels.cu's file header).
struct GpuRun {
    unsigned char* d_frames = nullptr;   // [SEQ_T*IMG_N], uploaded once, read by every model
    unsigned char* d_mask_fd = nullptr;  // [SEQ_T*IMG_N] frame-diff raw mask
    unsigned char* d_mask_sg = nullptr;  // [SEQ_T*IMG_N] single-Gaussian raw mask
    unsigned char* d_mask_mog = nullptr; // [SEQ_T*IMG_N] MOG-lite raw mask
    float* d_sg_mu = nullptr, * d_sg_var = nullptr;                 // [IMG_N]
    float* d_mog_w = nullptr, * d_mog_mean = nullptr, * d_mog_var = nullptr;  // [MOG_K*IMG_N]
};

static void gpu_alloc_and_upload(const std::vector<unsigned char>& frames, GpuRun& g)
{
    const size_t total = frames.size();
    CUDA_CHECK(cudaMalloc(&g.d_frames, total));
    CUDA_CHECK(cudaMalloc(&g.d_mask_fd, total));
    CUDA_CHECK(cudaMalloc(&g.d_mask_sg, total));
    CUDA_CHECK(cudaMalloc(&g.d_mask_mog, total));
    CUDA_CHECK(cudaMalloc(&g.d_sg_mu, static_cast<size_t>(IMG_N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_sg_var, static_cast<size_t>(IMG_N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_mog_w, static_cast<size_t>(MOG_K) * IMG_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_mog_mean, static_cast<size_t>(MOG_K) * IMG_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_mog_var, static_cast<size_t>(MOG_K) * IMG_N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g.d_frames, frames.data(), total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.d_mask_sg, 0, static_cast<size_t>(IMG_N)));   // frame 0's mask = background (see file header)
    CUDA_CHECK(cudaMemset(g.d_mask_mog, 0, static_cast<size_t>(IMG_N)));
}

static void gpu_free(GpuRun& g)
{
    cudaFree(g.d_frames); cudaFree(g.d_mask_fd); cudaFree(g.d_mask_sg); cudaFree(g.d_mask_mog);
    cudaFree(g.d_sg_mu); cudaFree(g.d_sg_var);
    cudaFree(g.d_mog_w); cudaFree(g.d_mog_mean); cudaFree(g.d_mog_var);
}

// ===========================================================================
// Gates. Each returns its headline metric(s) via out-params and a bool
// verdict; main() prints the "GATE <name>: PASS/FAIL" stable line and the
// numbers on an [info] line (README/THEORY.md document each derivation).
// ===========================================================================

// intrusion_detection — mean IoU (prediction vs. arm-rect truth, LAMP
// pixels excluded from both — see kernels.cuh's comment on why the two
// never geometrically overlap, but a stray false positive from the OTHER
// lesson (bimodal) should not silently punish THIS gate) over every E1+E5
// frame. Frame-diff is measured too but never gated — it can legitimately
// detect a real intruder; its designed failure is drift, not intrusion.
static double mean_iou_over_events(const std::vector<unsigned char>& mask_open)
{
    double iou_sum = 0.0;
    int frame_count = 0;
    auto score_range = [&](int fs, int fe) {
        for (int t = fs; t <= fe; ++t) {
            const std::vector<Rect> rects = active_arm_rects(t);
            long inter = 0, uni = 0;
            const unsigned char* frame_mask = &mask_open[static_cast<size_t>(t) * IMG_N];
            for (int y = 0; y < IMG_H; ++y) {
                for (int x = 0; x < IMG_W; ++x) {
                    if (pixel_in_lamp(x, y)) continue;   // excluded from this gate — see doc-comment
                    const bool truth = pixel_in_any(x, y, rects);
                    const bool pred  = frame_mask[px_index(x, y)] != 0;
                    if (truth || pred) ++uni;
                    if (truth && pred) ++inter;
                }
            }
            const double iou = (uni > 0) ? (static_cast<double>(inter) / static_cast<double>(uni)) : 1.0;
            iou_sum += iou;
            ++frame_count;
        }
    };
    score_range(E1_FRAME_START, E1_FRAME_END);
    score_range(E5_FRAME_START, E5_FRAME_END);
    return (frame_count > 0) ? (iou_sum / frame_count) : 0.0;
}

// fp_rate_over_range — the shared false-positive-rate computation used by
// both the illumination_drift and noise_floor gates: fraction of pixels
// classified foreground OUTSIDE every legitimate reason to be foreground
// (the lamp, any active arm rectangle, the box once placed) across a frame
// range. A "clean background" accounting, by construction.
static double fp_rate_over_range(const std::vector<unsigned char>& mask_open, int frame_lo, int frame_hi)
{
    long fg = 0, total = 0;
    for (int t = frame_lo; t <= frame_hi; ++t) {
        const std::vector<Rect> rects = active_arm_rects(t);
        const bool box_here = (t >= E2_FRAME_PLACED);
        const unsigned char* frame_mask = &mask_open[static_cast<size_t>(t) * IMG_N];
        for (int y = 0; y < IMG_H; ++y) {
            for (int x = 0; x < IMG_W; ++x) {
                if (pixel_in_lamp(x, y)) continue;
                if (!rects.empty() && pixel_in_any(x, y, rects)) continue;
                if (box_here && pixel_in_box(x, y)) continue;
                ++total;
                if (frame_mask[px_index(x, y)] != 0) ++fg;
            }
        }
    }
    return (total > 0) ? (static_cast<double>(fg) / static_cast<double>(total)) : 0.0;
}

// bimodal_fp_rate — false-positive rate restricted to the LAMP rectangle
// only, over a frame range. The lamp's blinking is legitimate, pre-learned
// background behavior (it started at frame 0 — there is no "intrusion"
// here), so ANY foreground call inside this rectangle in the steady state
// is, by the scene's own design, a false positive.
static double bimodal_fp_rate(const std::vector<unsigned char>& mask_open, int frame_lo, int frame_hi)
{
    long fg = 0, total = 0;
    for (int t = frame_lo; t <= frame_hi; ++t) {
        const unsigned char* frame_mask = &mask_open[static_cast<size_t>(t) * IMG_N];
        for (int y = LAMP_Y; y < LAMP_Y + LAMP_H; ++y) {
            for (int x = LAMP_X; x < LAMP_X + LAMP_W; ++x) {
                ++total;
                if (frame_mask[px_index(x, y)] != 0) ++fg;
            }
        }
    }
    return (total > 0) ? (static_cast<double>(fg) / static_cast<double>(total)) : 0.0;
}

// box_fg_fraction — fraction of the BOX rectangle's pixels classified
// foreground at frame t, from a model's OPENED mask. Used both by the
// absorption gate (to find "frames until absorbed") and by the
// absorption_curve.csv artifact.
static double box_fg_fraction(const std::vector<unsigned char>& mask_open, int t)
{
    long fg = 0, total = 0;
    const unsigned char* frame_mask = &mask_open[static_cast<size_t>(t) * IMG_N];
    for (int y = BOX_Y; y < BOX_Y + BOX_H; ++y) {
        for (int x = BOX_X; x < BOX_X + BOX_W; ++x) {
            ++total;
            if (frame_mask[px_index(x, y)] != 0) ++fg;
        }
    }
    return (total > 0) ? (static_cast<double>(fg) / static_cast<double>(total)) : 0.0;
}

// frames_until_absorbed — the smallest t' >= E2_FRAME_PLACED such that the
// box region's fg-fraction stays BELOW 0.5 for kAbsorptionConfirmWindow
// CONSECUTIVE frames starting at t' (a short confirm window rules out a
// single lucky noise-driven dip counting as "absorbed"). Returns
// t' - E2_FRAME_PLACED (frames elapsed), or -1 if the box never stabilizes
// below 0.5 anywhere in the remaining sequence.
static int frames_until_absorbed(const std::vector<unsigned char>& mask_open)
{
    for (int t0 = E2_FRAME_PLACED; t0 + kAbsorptionConfirmWindow <= SEQ_T; ++t0) {
        bool all_below = true;
        for (int t = t0; t < t0 + kAbsorptionConfirmWindow; ++t) {
            if (box_fg_fraction(mask_open, t) >= 0.5) { all_below = false; break; }
        }
        if (all_below) return t0 - E2_FRAME_PLACED;
    }
    return -1;
}

// predicted_absorption_frames — the closed-form EMA absorption-time
// derivation from THEORY.md "The math": d_t = d_0 * (1-alpha)^t decays
// below the k*sigma detection threshold at
//     t_abs = ceil( ln(d_0 / (k*sigma)) / (-ln(1-alpha)) )
// using MEASURED quantities (never the scene's hidden ground-truth
// mean — a real background subtractor never gets to see it): d_0 =
// |I_new - mu_pre|, averaged over the box region, where mu_pre is the SG
// model's own mean SNAPSHOTTED the instant before it ever saw the box
// (run_sg_cpu's mu_pre_box_snapshot) and I_new is the box's actual pixel
// value the frame it appears (straight from the loaded data).
//
// sigma is NOT taken from the pre-placement var_pre snapshot: SG_VAR_CEIL
// (kernels.cuh) caps the stored variance, and the box's jump is large
// enough that alpha*d0^2 >> SG_VAR_CEIL — the ceiling SATURATES on the very
// first post-placement update (verified true for this sequence's numbers;
// see kernels.cuh's SG_VAR_CEIL comment) and var effectively STAYS at the
// ceiling for the whole foreground phase, only relaxing once diff shrinks
// enough for the natural EMA value to fall back under it. Treating sigma
// as the CONSTANT sqrt(SG_VAR_CEIL) for the whole phase is therefore the
// honest closed form for THIS project's parameters, not an approximation
// of convenience.
static double predicted_absorption_frames(const std::vector<unsigned char>& frames,
                                           const std::vector<float>& mu_pre)
{
    double sum_I = 0.0, sum_mu = 0.0;
    int n = 0;
    const unsigned char* frame_at_placement = &frames[static_cast<size_t>(E2_FRAME_PLACED) * IMG_N];
    for (int y = BOX_Y; y < BOX_Y + BOX_H; ++y) {
        for (int x = BOX_X; x < BOX_X + BOX_W; ++x) {
            const int idx = px_index(x, y);
            sum_I  += frame_at_placement[idx];
            sum_mu += mu_pre[static_cast<size_t>(idx)];
            ++n;
        }
    }
    const double I_new  = sum_I / n;
    const double mu_pre_avg = sum_mu / n;
    const double sigma   = std::sqrt(static_cast<double>(SG_VAR_CEIL));   // see doc-comment: the ceiling saturates immediately
    const double d0      = std::fabs(I_new - mu_pre_avg);
    const double thresh  = SG_K_SIGMA * sigma;
    if (d0 <= thresh) return 0.0;   // already within threshold at placement -- would never have been flagged
    // Sign note (a genuine numerics lesson — see THEORY.md "Numerical
    // considerations"): ln(1-alpha) is NEGATIVE for 0<alpha<1, so dividing
    // by it directly (as a careless transcription of "ln(d0/thresh) /
    // ln(1-alpha)" might do) yields a NEGATIVE frame count. The correct
    // form divides by -ln(1-alpha), which is positive.
    return std::ceil(std::log(d0 / thresh) / (-std::log(1.0 - SG_ALPHA)));
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    // ---- 0) arguments: optional data-dir override (README/CLAUDE.md §6.1) --
    std::string cli_dir;
    if (argc > 1) cli_dir = argv[1];

    std::printf("[demo] 01.15 background subtraction for fixed-workspace cells: "
                "frame-diff vs running single Gaussian vs MOG-lite (K=3)\n");
    print_device_info();
    std::printf("PROBLEM: %d-frame %dx%d synthetic work-cell sequence, 3 background models, "
                "3x3 morphological open, 5 independent gates\n", SEQ_T, IMG_W, IMG_H);

    // ---- 1) load the committed sequence --------------------------------------
    std::vector<unsigned char> frames;
    if (!load_sequence(cli_dir, argv[0], frames)) return EXIT_FAILURE;
    std::printf("DATA: loaded %d frames (%d bytes) from data/sample/frames/\n", SEQ_T, static_cast<int>(frames.size()));

    const int total_elems = static_cast<int>(frames.size());

    // ---- 2) CPU reference pipeline -------------------------------------------
    CpuTimer cpu_timer;
    cpu_timer.begin();

    std::vector<unsigned char> cpu_raw_fd, cpu_raw_sg, cpu_raw_mog;
    SgState cpu_sg_state;
    MogState cpu_mog_state;
    std::vector<float> mu_pre_box, var_pre_box;   // snapshotted at E2_FRAME_PLACED — see run_sg_cpu

    run_frame_diff_cpu(frames, cpu_raw_fd);
    run_sg_cpu(frames, cpu_raw_sg, cpu_sg_state, mu_pre_box, var_pre_box);
    run_mog_cpu(frames, cpu_raw_mog, cpu_mog_state);

    std::vector<unsigned char> cpu_open_fd(cpu_raw_fd.size()), cpu_open_sg(cpu_raw_sg.size()), cpu_open_mog(cpu_raw_mog.size());
    morph_open_cpu(cpu_raw_fd.data(), cpu_open_fd.data(), IMG_W, IMG_H, total_elems);
    morph_open_cpu(cpu_raw_sg.data(), cpu_open_sg.data(), IMG_W, IMG_H, total_elems);
    morph_open_cpu(cpu_raw_mog.data(), cpu_open_mog.data(), IMG_W, IMG_H, total_elems);

    const double cpu_ms = cpu_timer.end_ms();

    // ---- 3) GPU pipeline -------------------------------------------------------
    GpuRun g;
    gpu_alloc_and_upload(frames, g);

    GpuTimer gpu_timer;
    gpu_timer.begin();

    launch_frame_diff(g.d_frames, g.d_frames, g.d_mask_fd, IMG_N, total_elems, FRAME_DIFF_THRESHOLD);

    // Single-Gaussian: init state on host, upload, then one launch per frame.
    {
        std::vector<float> mu0(static_cast<size_t>(IMG_N)), var0(static_cast<size_t>(IMG_N), SG_VAR_INIT);
        for (int i = 0; i < IMG_N; ++i) mu0[static_cast<size_t>(i)] = static_cast<float>(frames[static_cast<size_t>(i)]);
        CUDA_CHECK(cudaMemcpy(g.d_sg_mu, mu0.data(), mu0.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_sg_var, var0.data(), var0.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int t = 1; t < SEQ_T; ++t) {
            launch_sg_step(g.d_frames + static_cast<size_t>(t) * IMG_N, g.d_sg_mu, g.d_sg_var,
                            g.d_mask_sg + static_cast<size_t>(t) * IMG_N, IMG_N, SG_ALPHA, SG_K_SIGMA, SG_VAR_FLOOR, SG_VAR_CEIL);
        }
    }

    // MOG-lite: init state on host, upload, then one launch per frame.
    {
        const size_t total_modes = static_cast<size_t>(MOG_K) * IMG_N;
        std::vector<float> w0(total_modes, 0.0f), m0(total_modes, 0.0f), v0(total_modes, MOG_VAR_INIT);
        for (int i = 0; i < IMG_N; ++i) {
            w0[static_cast<size_t>(0 * IMG_N + i)] = 1.0f;
            m0[static_cast<size_t>(0 * IMG_N + i)] = static_cast<float>(frames[static_cast<size_t>(i)]);
            for (int k = 1; k < MOG_K; ++k) m0[static_cast<size_t>(k * IMG_N + i)] = static_cast<float>(frames[static_cast<size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpy(g.d_mog_w, w0.data(), w0.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_mog_mean, m0.data(), m0.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_mog_var, v0.data(), v0.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int t = 1; t < SEQ_T; ++t) {
            launch_mog_step(g.d_frames + static_cast<size_t>(t) * IMG_N, g.d_mog_w, g.d_mog_mean, g.d_mog_var,
                             g.d_mask_mog + static_cast<size_t>(t) * IMG_N, IMG_N, MOG_K,
                             MOG_MATCH_K_SIGMA, MOG_LR_WEIGHT, MOG_LR_PARAM, MOG_VAR_INIT, MOG_VAR_FLOOR,
                             MOG_W_INIT_NEW, MOG_BG_FRACTION);
        }
    }

    // Morphology, all three models (one erode+dilate pair per model, each
    // covering the whole sequence in one launch — see kernels.cu).
    unsigned char* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scratch, static_cast<size_t>(total_elems)));
    unsigned char* d_open_fd = nullptr, * d_open_sg = nullptr, * d_open_mog = nullptr;
    CUDA_CHECK(cudaMalloc(&d_open_fd, static_cast<size_t>(total_elems)));
    CUDA_CHECK(cudaMalloc(&d_open_sg, static_cast<size_t>(total_elems)));
    CUDA_CHECK(cudaMalloc(&d_open_mog, static_cast<size_t>(total_elems)));
    launch_morph_open(g.d_mask_fd, d_open_fd, d_scratch, IMG_W, IMG_H, total_elems);
    launch_morph_open(g.d_mask_sg, d_open_sg, d_scratch, IMG_W, IMG_H, total_elems);
    launch_morph_open(g.d_mask_mog, d_open_mog, d_scratch, IMG_W, IMG_H, total_elems);

    const float gpu_ms = gpu_timer.end_ms();   // synchronizes -> everything above has finished

    // ---- copy GPU results back -------------------------------------------------
    std::vector<unsigned char> gpu_raw_fd(static_cast<size_t>(total_elems)), gpu_raw_sg(static_cast<size_t>(total_elems)), gpu_raw_mog(static_cast<size_t>(total_elems));
    std::vector<unsigned char> gpu_open_fd(static_cast<size_t>(total_elems)), gpu_open_sg(static_cast<size_t>(total_elems)), gpu_open_mog(static_cast<size_t>(total_elems));
    CUDA_CHECK(cudaMemcpy(gpu_raw_fd.data(), g.d_mask_fd, static_cast<size_t>(total_elems), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_raw_sg.data(), g.d_mask_sg, static_cast<size_t>(total_elems), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_raw_mog.data(), g.d_mask_mog, static_cast<size_t>(total_elems), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_open_fd.data(), d_open_fd, static_cast<size_t>(total_elems), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_open_sg.data(), d_open_sg, static_cast<size_t>(total_elems), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_open_mog.data(), d_open_mog, static_cast<size_t>(total_elems), cudaMemcpyDeviceToHost));

    std::vector<float> gpu_sg_mu(static_cast<size_t>(IMG_N)), gpu_sg_var(static_cast<size_t>(IMG_N));
    CUDA_CHECK(cudaMemcpy(gpu_sg_mu.data(), g.d_sg_mu, gpu_sg_mu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sg_var.data(), g.d_sg_var, gpu_sg_var.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const size_t total_modes = static_cast<size_t>(MOG_K) * IMG_N;
    std::vector<float> gpu_mog_w(total_modes), gpu_mog_mean(total_modes), gpu_mog_var(total_modes);
    CUDA_CHECK(cudaMemcpy(gpu_mog_w.data(), g.d_mog_w, total_modes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_mog_mean.data(), g.d_mog_mean, total_modes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_mog_var.data(), g.d_mog_var, total_modes * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_scratch); cudaFree(d_open_fd); cudaFree(d_open_sg); cudaFree(d_open_mog);
    gpu_free(g);

    // ---- 4) VERIFY: GPU vs CPU --------------------------------------------------
    auto mask_mismatch_frac = [&](const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
        long mism = 0;
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++mism;
        return static_cast<double>(mism) / static_cast<double>(a.size());
    };
    auto max_abs_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
        float m = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
        return m;
    };

    const double fd_mismatch  = mask_mismatch_frac(cpu_raw_fd, gpu_raw_fd);
    const double sg_mismatch  = mask_mismatch_frac(cpu_raw_sg, gpu_raw_sg);
    const double mog_mismatch = mask_mismatch_frac(cpu_raw_mog, gpu_raw_mog);
    const float sg_mu_diff = max_abs_diff(cpu_sg_state.mu, gpu_sg_mu);
    const float sg_var_diff = max_abs_diff(cpu_sg_state.var, gpu_sg_var);
    const float mog_w_diff = max_abs_diff(cpu_mog_state.weight, gpu_mog_w);
    const float mog_mean_diff = max_abs_diff(cpu_mog_state.mean, gpu_mog_mean);
    const float mog_var_diff = max_abs_diff(cpu_mog_state.var, gpu_mog_var);

    const bool verify_fd  = (fd_mismatch <= kTwinMaskMismatchFrac);
    const bool verify_sg  = (sg_mismatch <= kTwinMaskMismatchFrac) && (sg_mu_diff <= kTwinStateAbsTol) && (sg_var_diff <= kTwinStateAbsTol);
    const bool verify_mog = (mog_mismatch <= kTwinMaskMismatchFrac) && (mog_w_diff <= kTwinStateAbsTol) &&
                             (mog_mean_diff <= kTwinStateAbsTol) && (mog_var_diff <= kTwinStateAbsTol);

    std::printf("[info] verify(frame-diff)   raw mask mismatch fraction = %.6f (%.0f / %d elements)\n",
                fd_mismatch, fd_mismatch * total_elems, total_elems);
    std::printf("[info] verify(single-Gaussian) raw mask mismatch fraction = %.6f; max|gpu-cpu| mu=%.6f var=%.6f\n",
                sg_mismatch, static_cast<double>(sg_mu_diff), static_cast<double>(sg_var_diff));
    std::printf("[info] verify(MOG-lite)     raw mask mismatch fraction = %.6f; max|gpu-cpu| weight=%.6f mean=%.6f var=%.6f\n",
                mog_mismatch, static_cast<double>(mog_w_diff), static_cast<double>(mog_mean_diff), static_cast<double>(mog_var_diff));
    std::printf("VERIFY(frame-diff): %s\n", verify_fd ? "PASS" : "FAIL");
    std::printf("VERIFY(single-Gaussian): %s\n", verify_sg ? "PASS" : "FAIL");
    std::printf("VERIFY(MOG-lite): %s\n", verify_mog ? "PASS" : "FAIL");

    // From here on, GATES read the GPU's OWN opened masks (the shipped
    // path) — the twin check above is what licenses treating GPU and CPU
    // as interchangeable for this purpose.
    const std::vector<unsigned char>& open_fd  = gpu_open_fd;
    const std::vector<unsigned char>& open_sg  = gpu_open_sg;
    const std::vector<unsigned char>& open_mog = gpu_open_mog;

    // ---- 5) GATES ----------------------------------------------------------------
    const double iou_sg  = mean_iou_over_events(open_sg);
    const double iou_mog = mean_iou_over_events(open_mog);
    const double iou_fd  = mean_iou_over_events(open_fd);   // reported only
    const bool gate_intrusion = (iou_sg >= kIntrusionIoUFloor) && (iou_mog >= kIntrusionIoUFloor);
    std::printf("[info] intrusion_detection mean IoU: single-Gaussian=%.4f MOG=%.4f frame-diff=%.4f (reported only)\n",
                iou_sg, iou_mog, iou_fd);
    std::printf("GATE intrusion_detection: %s\n", gate_intrusion ? "PASS" : "FAIL");

    const double drift_fp_sg  = fp_rate_over_range(open_sg, DRIFT_LATE_FRAME_LO, DRIFT_LATE_FRAME_HI);
    const double drift_fp_mog = fp_rate_over_range(open_mog, DRIFT_LATE_FRAME_LO, DRIFT_LATE_FRAME_HI);
    const double drift_fp_fd  = fp_rate_over_range(open_fd, DRIFT_LATE_FRAME_LO, DRIFT_LATE_FRAME_HI);
    const bool gate_drift = (drift_fp_sg <= kDriftFPCeilingAdaptive) && (drift_fp_mog <= kDriftFPCeilingAdaptive)
                           && (drift_fp_fd > kDriftFPFloorFrameDiff);
    std::printf("[info] illumination_drift late-window FP rate: single-Gaussian=%.4f MOG=%.4f frame-diff=%.4f\n",
                drift_fp_sg, drift_fp_mog, drift_fp_fd);
    std::printf("GATE illumination_drift: %s\n", gate_drift ? "PASS" : "FAIL");

    const double predicted = predicted_absorption_frames(frames, mu_pre_box);
    (void)var_pre_box;   // snapshotted for symmetry/diagnostics; the closed form uses SG_VAR_CEIL directly — see predicted_absorption_frames' doc-comment
    const int measured_sg = frames_until_absorbed(open_sg);
    const int measured_mog = frames_until_absorbed(open_mog);
    const bool gate_absorption = (measured_sg >= 0) && (predicted > 0.0)
        && (measured_sg >= kAbsorptionFactorLo * predicted) && (measured_sg <= kAbsorptionFactorHi * predicted);
    std::printf("[info] absorption single-Gaussian: predicted=%.2f frames, measured=%d frames (confirm window %d)\n",
                predicted, measured_sg, kAbsorptionConfirmWindow);
    std::printf("[info] absorption MOG-lite (reported only, no closed form): measured=%d frames\n", measured_mog);
    std::printf("GATE absorption: %s\n", gate_absorption ? "PASS" : "FAIL");

    const double bimodal_fp_mog = bimodal_fp_rate(open_mog, BIMODAL_FRAME_LO, BIMODAL_FRAME_HI);
    const double bimodal_fp_sg  = bimodal_fp_rate(open_sg, BIMODAL_FRAME_LO, BIMODAL_FRAME_HI);
    const bool gate_bimodal = (bimodal_fp_mog <= kBimodalFPCeilingMog) && (bimodal_fp_sg > kBimodalFPFloorSg);
    std::printf("[info] bimodal_lesson lamp-region FP rate: MOG=%.4f single-Gaussian=%.4f\n", bimodal_fp_mog, bimodal_fp_sg);
    std::printf("GATE bimodal_lesson: %s\n", gate_bimodal ? "PASS" : "FAIL");

    const double noise_fp_sg  = fp_rate_over_range(open_sg, NOISE_FLOOR_FRAME_LO, NOISE_FLOOR_FRAME_HI);
    const double noise_fp_mog = fp_rate_over_range(open_mog, NOISE_FLOOR_FRAME_LO, NOISE_FLOOR_FRAME_HI);
    const bool gate_noise_floor = (noise_fp_sg <= kNoiseFloorFPCeiling) && (noise_fp_mog <= kNoiseFloorFPCeiling);
    std::printf("[info] noise_floor early-window FP rate: single-Gaussian=%.4f MOG=%.4f\n", noise_fp_sg, noise_fp_mog);
    std::printf("GATE noise_floor: %s\n", gate_noise_floor ? "PASS" : "FAIL");

    // ---- 6) ARTIFACTS --------------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);

    // mask strips: 3 key frames, 5 panels each (original, truth, frame-diff,
    // single-Gaussian, MOG), concatenated horizontally with 2px black gaps.
    const int strip_frames[3] = {35, 65, 140};
    for (int t : strip_frames) {
        const int gap = 2;
        const int panel_w = IMG_W;
        const int strip_w = panel_w * 5 + gap * 4;
        std::vector<unsigned char> rgb(static_cast<size_t>(strip_w) * IMG_H * 3, 0);
        const std::vector<Rect> rects = active_arm_rects(t);
        const bool box_here = (t >= E2_FRAME_PLACED);
        auto put_px = [&](int panel, int x, int y, unsigned char r, unsigned char gch, unsigned char b) {
            const int gx = panel * (panel_w + gap) + x;
            const size_t o = (static_cast<size_t>(y) * strip_w + gx) * 3;
            rgb[o + 0] = r; rgb[o + 1] = gch; rgb[o + 2] = b;
        };
        const unsigned char* fdm = &open_fd[static_cast<size_t>(t) * IMG_N];
        const unsigned char* sgm = &open_sg[static_cast<size_t>(t) * IMG_N];
        const unsigned char* mgm = &open_mog[static_cast<size_t>(t) * IMG_N];
        const unsigned char* raw = &frames[static_cast<size_t>(t) * IMG_N];
        for (int y = 0; y < IMG_H; ++y) {
            for (int x = 0; x < IMG_W; ++x) {
                const int idx = px_index(x, y);
                const unsigned char gray = raw[idx];
                put_px(0, x, y, gray, gray, gray);
                const bool truth = pixel_in_any(x, y, rects) || (box_here && pixel_in_box(x, y));
                put_px(1, x, y, truth ? 255 : 0, truth ? 255 : 0, truth ? 255 : 0);
                const unsigned char fd_v = fdm[idx] ? 255 : 0;
                put_px(2, x, y, fd_v, fd_v, fd_v);
                const unsigned char sg_v = sgm[idx] ? 255 : 0;
                put_px(3, x, y, sg_v, sg_v, sg_v);
                const unsigned char mg_v = mgm[idx] ? 255 : 0;
                put_px(4, x, y, mg_v, mg_v, mg_v);
            }
        }
        // Buffer sized for the WORST-CASE VS-layout out_dir (a full absolute
        // path several directories deep, e.g. "...\01.15-...\build\x64\
        // Release/../../../demo/out") plus the filename — 128 bytes silently
        // TRUNCATED this exact path during this project's own build/verify
        // pass and std::ofstream happily created a stray file at the
        // truncated (garbage) location one directory outside this project's
        // folder, violating the repo's per-project write boundary (CLAUDE.md
        // §10). 512 bytes is comfortably larger than any real Windows path
        // this demo will ever construct (MAX_PATH is 260); snprintf's
        // return-truncation is defensive, not the primary fix.
        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/mask_strip_frame%03d.ppm", out_dir.c_str(), t);
        if (write_ppm(fname, strip_w, IMG_H, rgb)) {
            // Stable line: a RELATIVE-looking path, never the full resolved
            // out_dir (which is an absolute, machine-specific path — see
            // resolve_out_dir()'s doc-comment — and would make this line
            // impossible to diff against expected_output.txt across
            // machines). Same convention as project 01.04's ARTIFACT lines.
            std::printf("ARTIFACT: demo/out/mask_strip_frame%03d.ppm (panels: original | truth | frame-diff | single-Gaussian | MOG-lite)\n", t);
        }
    }

    // absorption_curve.csv
    {
        const std::string path = out_dir + "/absorption_curve.csv";
        std::ofstream f(path);
        f << "# predicted_absorption_frames_single_gaussian=" << predicted << "\n";
        f << "frame,sg_box_fg_fraction,mog_box_fg_fraction\n";
        const int lo = std::max(0, E2_FRAME_PLACED - 5);
        const int hi = std::min(SEQ_T - 1, E2_FRAME_PLACED + 100);
        for (int t = lo; t <= hi; ++t) {
            f << t << "," << box_fg_fraction(open_sg, t) << "," << box_fg_fraction(open_mog, t) << "\n";
        }
        std::printf("ARTIFACT: demo/out/absorption_curve.csv\n");
    }

    // fp_rate_timeline.csv — per-frame FP rate over the "clean background"
    // pixel set (see fp_rate_over_range's doc-comment), one frame at a time.
    {
        const std::string path = out_dir + "/fp_rate_timeline.csv";
        std::ofstream f(path);
        f << "frame,frame_diff_fp_rate,sg_fp_rate,mog_fp_rate\n";
        for (int t = 0; t < SEQ_T; ++t) {
            f << t << "," << fp_rate_over_range(open_fd, t, t) << ","
              << fp_rate_over_range(open_sg, t, t) << ","
              << fp_rate_over_range(open_mog, t, t) << "\n";
        }
        std::printf("ARTIFACT: demo/out/fp_rate_timeline.csv\n");
    }

    // gates_metrics.csv — one-row-per-gate summary table.
    {
        const std::string path = out_dir + "/gates_metrics.csv";
        std::ofstream f(path);
        f << "gate,metric,value,threshold,verdict\n";
        f << "intrusion_detection,mean_iou_sg," << iou_sg << "," << kIntrusionIoUFloor << "," << (iou_sg >= kIntrusionIoUFloor ? "PASS" : "FAIL") << "\n";
        f << "intrusion_detection,mean_iou_mog," << iou_mog << "," << kIntrusionIoUFloor << "," << (iou_mog >= kIntrusionIoUFloor ? "PASS" : "FAIL") << "\n";
        f << "intrusion_detection,mean_iou_frame_diff_reported_only," << iou_fd << ",n/a,n/a\n";
        f << "illumination_drift,fp_rate_sg," << drift_fp_sg << "," << kDriftFPCeilingAdaptive << "," << (drift_fp_sg <= kDriftFPCeilingAdaptive ? "PASS" : "FAIL") << "\n";
        f << "illumination_drift,fp_rate_mog," << drift_fp_mog << "," << kDriftFPCeilingAdaptive << "," << (drift_fp_mog <= kDriftFPCeilingAdaptive ? "PASS" : "FAIL") << "\n";
        f << "illumination_drift,fp_rate_frame_diff_must_exceed," << drift_fp_fd << "," << kDriftFPFloorFrameDiff << "," << (drift_fp_fd > kDriftFPFloorFrameDiff ? "PASS" : "FAIL") << "\n";
        f << "absorption,predicted_frames_sg," << predicted << ",n/a,n/a\n";
        f << "absorption,measured_frames_sg," << measured_sg << "," << kAbsorptionFactorLo << "-" << kAbsorptionFactorHi << "x predicted," << (gate_absorption ? "PASS" : "FAIL") << "\n";
        f << "absorption,measured_frames_mog_reported_only," << measured_mog << ",n/a,n/a\n";
        f << "bimodal_lesson,fp_rate_mog," << bimodal_fp_mog << "," << kBimodalFPCeilingMog << "," << (bimodal_fp_mog <= kBimodalFPCeilingMog ? "PASS" : "FAIL") << "\n";
        f << "bimodal_lesson,fp_rate_sg_must_exceed," << bimodal_fp_sg << "," << kBimodalFPFloorSg << "," << (bimodal_fp_sg > kBimodalFPFloorSg ? "PASS" : "FAIL") << "\n";
        f << "noise_floor,fp_rate_sg," << noise_fp_sg << "," << kNoiseFloorFPCeiling << "," << (noise_fp_sg <= kNoiseFloorFPCeiling ? "PASS" : "FAIL") << "\n";
        f << "noise_floor,fp_rate_mog," << noise_fp_mog << "," << kNoiseFloorFPCeiling << "," << (noise_fp_mog <= kNoiseFloorFPCeiling ? "PASS" : "FAIL") << "\n";
        std::printf("ARTIFACT: demo/out/gates_metrics.csv\n");
    }

    // ---- 7) report --------------------------------------------------------------
    std::printf("[time] CPU reference (3 models + morphology): %.3f ms\n", cpu_ms);
    std::printf("[time] GPU (3 models + morphology, %d kernel launches): %.3f ms\n",
                1 + 2 * (SEQ_T - 1) + 6, static_cast<double>(gpu_ms));
    if (gpu_ms > 0.0f) {
        std::printf("[time] speed-up (teaching artifact, not a benchmark): %.1fx\n", cpu_ms / static_cast<double>(gpu_ms));
    }

    const bool all_pass = verify_fd && verify_sg && verify_mog
        && gate_intrusion && gate_drift && gate_absorption && gate_bimodal && gate_noise_floor;
    if (all_pass) {
        std::printf("RESULT: PASS (GPU matches CPU reference on all three models; all 5 independent gates pass)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
