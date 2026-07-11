// ===========================================================================
// main.cu — entry point for project 01.14
//           Template matching (NCC) at scale for pick verification
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed synthetic tray scene (data/sample/tray.pgm), the
//      15-template golden reference set (templates.pgm), and per-slot
//      ground truth (truth.csv) — see scripts/make_synthetic.py.
//   2. Build the whole-tray integral images (GPU: 2-pass separable scan;
//      CPU: single-pass recurrence — independent algorithms, same result).
//   3. VERIFY STAGE: integral images and window statistics GPU vs CPU
//      BIT-EXACT (pure integer arithmetic); the full 104,040-evaluation NCC
//      score volume, computed THREE ways on the GPU (naive / sum-table /
//      shared-memory), against an independent CPU oracle within float
//      tolerance.
//   4. GATES (host-side analysis on the VERIFIED score volume — see
//      reference_cpu.cpp's independence ruling for why this stage is not
//      itself twinned): variant_consistency (the 3 GPU kernels agree with
//      each other), classification (every slot's OK/WRONG_PART/EMPTY
//      verdict matches truth), localization (recovered offset accuracy),
//      rotation_lesson (single-template vs 5-angle rotation-set recovery),
//      illumination_robustness (NCC survives the shadow cohort; a plain SSD
//      score does not — the designed NCC-vs-SSD comparison).
//   5. ARTIFACTS: demo/out/{tray_overlay.ppm, score_map_rotated_slot.pgm,
//      score_vs_angle.csv, per_slot_scores.csv, gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY:", "GATE <name>: PASS/FAIL", "ARTIFACT:", "RESULT:" — no
// measured floats or device names, so they are deterministic on any
// machine. Measured numbers live on "[info]"/"[time]" lines (not diffed).
// Change a stable line -> update demo/expected_output.txt in the same change.
//
// Read this after: kernels.cuh (contracts) and kernels.cu (the kernels).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// SECTION A — gate/verification tolerances.
//
// NOT pipeline data contracts (kernels.cuh owns those) — this project's
// ACCEPTANCE CRITERIA, MEASURED against the committed synthetic tray (seed
// 42) and then margined for honest headroom. See THEORY.md "How we verify
// correctness" for the measured numbers each bound was set from.
// ===========================================================================

// GPU-vs-CPU NCC score tolerance. Every intermediate up to the final
// sqrt/divide is EXACT integer arithmetic shared bit-for-bit by both paths
// (kernels.cuh SECTION 6) — the only possible source of disagreement is the
// last floating-point step (device vs host double sqrt/divide). MEASURED on
// this project's committed scene (RTX 2080 SUPER + MSVC 14.51/CUDA 13.3):
// worst observed |GPU-CPU| over all 104,040 evaluations is EXACTLY 0.0 — a
// real, if pleasant, surprise (both toolchains' double sqrt/divide are
// IEEE-754 correctly-rounded here). This bound is real headroom for
// platforms/compilers where that equality is not guaranteed, not a fudge
// factor for an observed gap.
static const float NCC_GPU_CPU_TOL = 5e-4f;

// Cross-GPU-variant agreement — the "three ways of computing the same
// answer must agree" gate. All three read the same inputs and compute the
// same integer sums via different paths (direct rescan / integral-image
// lookup / shared-memory-cached correlation); they should match far tighter
// than the GPU-vs-CPU bound above.
static const float VARIANT_CONSISTENCY_TOL = 1e-6f;

// Classification threshold: an NCC score at/above this is "a real match".
// MEASURED on this project's own committed tray (THEORY.md "How we verify
// correctness" has the full per-cohort score table): well-placed correct
// matches score >= 0.92 even under this scene's texture+noise; the worst
// unrelated (wrong-part/empty) score observed is under 0.45. T_OK is ALSO
// (deliberately) the value the rotation_lesson gate straddles: the ROTATED
// cohort's single-0-degree-template score and its 5-angle-rotation-set score
// were measured to land on either side of ~0.65 with real margin (the 24
// degree test angle was chosen, not guessed, to make that true — see
// scripts/make_synthetic.py's ROTATED-cohort comment and THEORY.md).
static const float T_OK = 0.65f;

// Localization gate: max allowed |recovered - true| offset, in pixels, for
// an OK-verdict slot. The NCC peak is searched on the same INTEGER pixel
// grid the applied offset was rendered on, so an exact match is expected
// except where texture/noise perturbs the argmax by a pixel — margined.
static const float LOCALIZATION_MAX_ERR_PX = 1.5f;

// illumination_robustness: the SSD "would-reject" threshold is measured from
// a NON-shadowed slot of the SAME part type (slot 1, gear_disk) and margined
// — see main() for the measured baseline this multiplier is set against.
static const double SSD_REJECT_MARGIN_MULT = 2.0;

// ===========================================================================
// SECTION B — small host-side helpers: PGM/PPM I/O (no image library —
// CLAUDE.md's "no black boxes": PGM/PPM are ASCII headers + raw samples).
// ===========================================================================
static bool load_pgm(const std::string& path, int expect_w, int expect_h, std::vector<uint8_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string magic;
    f >> magic;
    if (magic != "P5") return false;
    int w = 0, h = 0, maxval = 0;
    f >> w >> h >> maxval;
    f.get();   // the single whitespace byte between header and binary payload
    if (!f || w != expect_w || h != expect_h || maxval != 255) return false;
    out.resize(static_cast<size_t>(w) * h);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    return static_cast<bool>(f);
}

static bool write_pgm(const std::string& path, int w, int h, const std::vector<uint8_t>& px)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
    return static_cast<bool>(f);
}

static bool write_ppm(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(f);
}

static void put_px(std::vector<uint8_t>& rgb, int w, int h, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    const size_t i = (static_cast<size_t>(y) * w + x) * 3;
    rgb[i] = r; rgb[i + 1] = g; rgb[i + 2] = b;
}

static void draw_rect_outline(std::vector<uint8_t>& rgb, int w, int h, int x0, int y0, int x1, int y1,
                              uint8_t r, uint8_t g, uint8_t b)
{
    for (int x = x0; x < x1; ++x) { put_px(rgb, w, h, x, y0, r, g, b); put_px(rgb, w, h, x, y1 - 1, r, g, b); }
    for (int y = y0; y < y1; ++y) { put_px(rgb, w, h, x0, y, r, g, b); put_px(rgb, w, h, x1 - 1, y, r, g, b); }
}

static void draw_cross(std::vector<uint8_t>& rgb, int w, int h, int cx, int cy, uint8_t r, uint8_t g, uint8_t b)
{
    for (int d = -3; d <= 3; ++d) { put_px(rgb, w, h, cx + d, cy, r, g, b); put_px(rgb, w, h, cx, cy + d, r, g, b); }
}

// ===========================================================================
// SECTION C — ground-truth loading (data/sample/truth.csv).
// ===========================================================================
struct SlotTruth {
    int expected_type = 0;
    int actual_type = -1;       // -1 = empty
    float rotation_deg = 0.0f;
    int offset_dx = 0, offset_dy = 0;
    bool shadow = false;
    std::string cohort;
    std::string verdict;        // "OK" / "WRONG_PART" / "EMPTY"
};

static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    return cells;
}

static bool load_truth(const std::string& path, std::vector<SlotTruth>& out)
{
    out.assign(NUM_SLOTS, SlotTruth{});
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    int rows = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto c = split_csv(line);
        if (c.size() < 11) return false;
        const int slot = std::atoi(c[0].c_str());
        if (slot < 0 || slot >= NUM_SLOTS) return false;
        SlotTruth t;
        t.cohort = c[3];
        t.expected_type = std::atoi(c[4].c_str());
        t.actual_type = std::atoi(c[5].c_str());
        t.rotation_deg = std::strtof(c[6].c_str(), nullptr);
        t.offset_dx = std::atoi(c[7].c_str());
        t.offset_dy = std::atoi(c[8].c_str());
        t.shadow = std::atoi(c[9].c_str()) != 0;
        t.verdict = c[10];
        out[static_cast<size_t>(slot)] = t;
        ++rows;
    }
    return rows == NUM_SLOTS;
}

// ===========================================================================
// SECTION D — best-match search over a template range (host-side analysis,
// NOT part of the GPU/CPU twin — see reference_cpu.cpp's independence
// ruling: this consumes the ALREADY-VERIFIED score volume).
// ===========================================================================
struct BestMatch {
    float score = -2.0f;   // NCC in [-1,1]; -2 is an impossible sentinel
    int tmpl = -1;
    int oy = SEARCH_RADIUS, ox = SEARCH_RADIUS;   // offset indices; default = zero offset
};

static BestMatch find_best(const std::vector<float>& scores, int slot, int tmpl_lo, int tmpl_hi)
{
    BestMatch b;
    for (int t = tmpl_lo; t < tmpl_hi; ++t) {
        for (int oy = 0; oy < NUM_OFFSETS_1D; ++oy) {
            for (int ox = 0; ox < NUM_OFFSETS_1D; ++ox) {
                const float s = scores[static_cast<size_t>(score_index(slot, t, oy, ox))];
                if (s > b.score) { b.score = s; b.tmpl = t; b.oy = oy; b.ox = ox; }
            }
        }
    }
    return b;
}

// ===========================================================================
// SECTION E — plain SSD (sum of squared differences), the ILLUSTRATIVE
// non-normalized-matching baseline the illumination_robustness gate
// contrasts against NCC. Deliberately host-only and NOT GPU-accelerated —
// the catalog bullet asks for a designed COMPARISON, not a second production
// matcher; see THEORY.md "Where this sits in the real world".
// ===========================================================================
static int64_t ssd_at(const std::vector<uint8_t>& img, int x0, int y0, const uint8_t* tpl)
{
    int64_t s = 0;
    for (int ty = 0; ty < TEMPLATE_SIZE; ++ty) {
        const uint8_t* row = img.data() + static_cast<size_t>(y0 + ty) * IMG_W + x0;
        const uint8_t* trow = tpl + ty * TEMPLATE_SIZE;
        for (int tx = 0; tx < TEMPLATE_SIZE; ++tx) {
            const int64_t d = static_cast<int64_t>(row[tx]) - trow[tx];
            s += d * d;
        }
    }
    return s;
}

// Best (minimum) SSD over the whole +-SEARCH_RADIUS sweep against ONE
// template — the direct SSD analogue of find_best's NCC maximum search.
static int64_t ssd_best(const std::vector<uint8_t>& img, int slot, const uint8_t* tpl)
{
    int64_t best = INT64_MAX;
    for (int oy = 0; oy < NUM_OFFSETS_1D; ++oy) {
        const int dy = oy - SEARCH_RADIUS;
        for (int ox = 0; ox < NUM_OFFSETS_1D; ++ox) {
            const int dx = ox - SEARCH_RADIUS;
            const int x0 = slot_window_x0(slot) + SEARCH_RADIUS + dx;
            const int y0 = slot_window_y0(slot) + SEARCH_RADIUS + dy;
            const int64_t s = ssd_at(img, x0, y0, tpl);
            if (s < best) best = s;
        }
    }
    return best;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string cli_data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) cli_data_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data data/sample]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Template matching (NCC) at scale for pick verification (project 01.14)\n");
    print_device_info();
    std::printf("PROBLEM: NCC pick verification, tray %dx%d=%d slots, %d types x %d rotations = %d templates, "
                "search +-%d px (%dx%d), score volume %d evals, int64 sum-table + FP32 sqrt/divide\n",
                NUM_COLS, NUM_ROWS, NUM_SLOTS, NUM_TYPES, NUM_ROT, NUM_TEMPLATES, SEARCH_RADIUS,
                NUM_OFFSETS_1D, NUM_OFFSETS_1D, static_cast<int>(SCORE_VOLUME_CELLS));

    // ---- 1) data --------------------------------------------------------
    const std::string tray_path = find_data_file(cli_data_dir, argv[0], "tray.pgm");
    const std::string templates_path = find_data_file(cli_data_dir, argv[0], "templates.pgm");
    const std::string truth_path = find_data_file(cli_data_dir, argv[0], "truth.csv");
    if (tray_path.empty() || templates_path.empty() || truth_path.empty()) {
        std::printf("DATA: NOT FOUND — data/sample/{tray.pgm,templates.pgm,truth.csv} missing "
                    "(run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::vector<uint8_t> h_img, h_templates;
    const bool img_ok = load_pgm(tray_path, IMG_W, IMG_H, h_img);
    const bool tpl_ok = load_pgm(templates_path, TEMPLATE_SIZE, TEMPLATE_SIZE * NUM_TEMPLATES, h_templates);
    std::vector<SlotTruth> truth;
    const bool truth_ok = load_truth(truth_path, truth);
    if (!img_ok || !tpl_ok || !truth_ok) {
        std::printf("DATA: MALFORMED — see file paths above\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }
    std::printf("DATA: loaded synthetic tray.pgm + templates.pgm + truth.csv "
               "(%d slots, 6 designed cohorts: plain/offset/rotated/wrong_part/empty/shadow) [synthetic]\n",
               NUM_SLOTS);
    std::printf("[info] tray: %s | templates: %s | truth: %s\n",
               tray_path.c_str(), templates_path.c_str(), truth_path.c_str());

    // ---- 2) template statistics — the ONE shared computation (kernels.cuh
    // SECTION 5), handed identically to both the GPU (constant memory) and
    // the CPU oracle. ------------------------------------------------------
    std::vector<int64_t> S_t(NUM_TEMPLATES), S_tt(NUM_TEMPLATES);
    compute_template_stats(h_templates.data(), S_t.data(), S_tt.data());
    upload_template_stats(S_t.data(), S_tt.data());

    // ---- 3) device buffers -------------------------------------------------
    uint8_t *d_img = nullptr, *d_templates = nullptr;
    uint32_t *d_ii_sum = nullptr, *d_ws_sum = nullptr;
    uint64_t *d_ii_sumsq = nullptr, *d_ws_sumsq = nullptr;
    float *d_scores_naive = nullptr, *d_scores_sumtable = nullptr, *d_scores_shared = nullptr;

    CUDA_CHECK(cudaMalloc(&d_img, static_cast<size_t>(IMG_PIXELS)));
    CUDA_CHECK(cudaMalloc(&d_templates, static_cast<size_t>(NUM_TEMPLATES) * TEMPLATE_PIXELS));
    CUDA_CHECK(cudaMalloc(&d_ii_sum, sizeof(uint32_t) * static_cast<size_t>(II_CELLS)));
    CUDA_CHECK(cudaMalloc(&d_ii_sumsq, sizeof(uint64_t) * static_cast<size_t>(II_CELLS)));
    CUDA_CHECK(cudaMalloc(&d_ws_sum, sizeof(uint32_t) * static_cast<size_t>(WINDOW_STATS_CELLS)));
    CUDA_CHECK(cudaMalloc(&d_ws_sumsq, sizeof(uint64_t) * static_cast<size_t>(WINDOW_STATS_CELLS)));
    CUDA_CHECK(cudaMalloc(&d_scores_naive, sizeof(float) * static_cast<size_t>(SCORE_VOLUME_CELLS)));
    CUDA_CHECK(cudaMalloc(&d_scores_sumtable, sizeof(float) * static_cast<size_t>(SCORE_VOLUME_CELLS)));
    CUDA_CHECK(cudaMalloc(&d_scores_shared, sizeof(float) * static_cast<size_t>(SCORE_VOLUME_CELLS)));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), h_img.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_templates, h_templates.data(), h_templates.size(), cudaMemcpyHostToDevice));

    // ---- 4) GPU pipeline, each stage individually event-timed --------------
    GpuTimer gt;
    gt.begin(); launch_build_integral_images(d_img, d_ii_sum, d_ii_sumsq); const float ms_integral = gt.end_ms();
    gt.begin(); launch_window_stats(d_ii_sum, d_ii_sumsq, d_ws_sum, d_ws_sumsq); const float ms_wstats = gt.end_ms();
    gt.begin(); launch_ncc_naive(d_img, d_templates, d_scores_naive); const float ms_naive = gt.end_ms();
    gt.begin(); launch_ncc_sumtable(d_img, d_ii_sum, d_ii_sumsq, d_templates, d_scores_sumtable); const float ms_sumtable = gt.end_ms();
    gt.begin(); launch_ncc_shared(d_img, d_ii_sum, d_ii_sumsq, d_templates, d_scores_shared); const float ms_shared = gt.end_ms();

    std::vector<uint32_t> g_ii_sum(static_cast<size_t>(II_CELLS));
    std::vector<uint64_t> g_ii_sumsq(static_cast<size_t>(II_CELLS));
    std::vector<uint32_t> g_ws_sum(static_cast<size_t>(WINDOW_STATS_CELLS));
    std::vector<uint64_t> g_ws_sumsq(static_cast<size_t>(WINDOW_STATS_CELLS));
    std::vector<float> g_scores_naive(static_cast<size_t>(SCORE_VOLUME_CELLS));
    std::vector<float> g_scores_sumtable(static_cast<size_t>(SCORE_VOLUME_CELLS));
    std::vector<float> g_scores_shared(static_cast<size_t>(SCORE_VOLUME_CELLS));
    CUDA_CHECK(cudaMemcpy(g_ii_sum.data(), d_ii_sum, g_ii_sum.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_ii_sumsq.data(), d_ii_sumsq, g_ii_sumsq.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_ws_sum.data(), d_ws_sum, g_ws_sum.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_ws_sumsq.data(), d_ws_sumsq, g_ws_sumsq.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_scores_naive.data(), d_scores_naive, g_scores_naive.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_scores_sumtable.data(), d_scores_sumtable, g_scores_sumtable.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_scores_shared.data(), d_scores_shared, g_scores_shared.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_ii_sum)); CUDA_CHECK(cudaFree(d_ii_sumsq));
    CUDA_CHECK(cudaFree(d_ws_sum)); CUDA_CHECK(cudaFree(d_ws_sumsq));
    CUDA_CHECK(cudaFree(d_scores_naive)); CUDA_CHECK(cudaFree(d_scores_sumtable)); CUDA_CHECK(cudaFree(d_scores_shared));

    // ---- 5) CPU oracle, wall-clock timed as one block -----------------------
    std::vector<uint32_t> c_ii_sum(static_cast<size_t>(II_CELLS));
    std::vector<uint64_t> c_ii_sumsq(static_cast<size_t>(II_CELLS));
    std::vector<uint32_t> c_ws_sum(static_cast<size_t>(WINDOW_STATS_CELLS));
    std::vector<uint64_t> c_ws_sumsq(static_cast<size_t>(WINDOW_STATS_CELLS));
    std::vector<float> c_scores(static_cast<size_t>(SCORE_VOLUME_CELLS));

    CpuTimer ct_integral; ct_integral.begin();
    build_integral_images_cpu(h_img.data(), c_ii_sum.data(), c_ii_sumsq.data());
    const double cpu_ms_integral = ct_integral.end_ms();

    CpuTimer ct_wstats; ct_wstats.begin();
    window_stats_cpu(c_ii_sum.data(), c_ii_sumsq.data(), c_ws_sum.data(), c_ws_sumsq.data());
    const double cpu_ms_wstats = ct_wstats.end_ms();

    CpuTimer ct_ncc; ct_ncc.begin();
    ncc_scores_cpu(h_img.data(), c_ii_sum.data(), c_ii_sumsq.data(), h_templates.data(),
                  S_t.data(), S_tt.data(), c_scores.data());
    const double cpu_ms_ncc = ct_ncc.end_ms();

    // ======================= VERIFY STAGE =====================================
    bool verify_pass = true;

    const bool ii_exact = (g_ii_sum == c_ii_sum) && (g_ii_sumsq == c_ii_sumsq);
    verify_pass = verify_pass && ii_exact;
    std::printf("VERIFY: integral images (sum + sumSq, %lld cells) GPU vs CPU BIT-EXACT %s\n",
               static_cast<long long>(II_CELLS), ii_exact ? "PASS" : "FAIL");

    const bool ws_exact = (g_ws_sum == c_ws_sum) && (g_ws_sumsq == c_ws_sumsq);
    verify_pass = verify_pass && ws_exact;
    std::printf("VERIFY: window statistics (S_w, S_ww; %lld cells) GPU vs CPU BIT-EXACT %s\n",
               static_cast<long long>(WINDOW_STATS_CELLS), ws_exact ? "PASS" : "FAIL");

    auto max_abs_diff = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float worst = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
        return worst;
    };
    const float d_naive = max_abs_diff(g_scores_naive, c_scores);
    const float d_sumtable = max_abs_diff(g_scores_sumtable, c_scores);
    const float d_shared = max_abs_diff(g_scores_shared, c_scores);
    const float worst_gpu_cpu = std::max(d_naive, std::max(d_sumtable, d_shared));
    const bool ncc_tol_pass = worst_gpu_cpu <= NCC_GPU_CPU_TOL;
    verify_pass = verify_pass && ncc_tol_pass;
    std::printf("[info] NCC score worst |GPU-CPU| over %lld evals: naive=%.3e sumtable=%.3e shared=%.3e (tol %.1e)\n",
               static_cast<long long>(SCORE_VOLUME_CELLS), static_cast<double>(d_naive),
               static_cast<double>(d_sumtable), static_cast<double>(d_shared), static_cast<double>(NCC_GPU_CPU_TOL));
    std::printf("VERIFY: NCC scores (naive/sum-table/shared vs CPU oracle) within float tolerance %s\n",
               ncc_tol_pass ? "PASS" : "FAIL");

    std::printf("[time] integral images: GPU %.3f ms | CPU %.2f ms\n", static_cast<double>(ms_integral), cpu_ms_integral);
    std::printf("[time] window stats:    GPU %.3f ms | CPU %.2f ms\n", static_cast<double>(ms_wstats), cpu_ms_wstats);
    std::printf("[time] NCC score volume (%lld evals): naive %.3f ms | sum-table %.3f ms (%.1fx vs naive) | "
               "shared-memory %.3f ms (%.1fx vs naive, %.1fx vs sum-table) | CPU oracle %.1f ms\n",
               static_cast<long long>(SCORE_VOLUME_CELLS),
               static_cast<double>(ms_naive),
               static_cast<double>(ms_sumtable), static_cast<double>(ms_naive) / static_cast<double>(ms_sumtable),
               static_cast<double>(ms_shared),
               static_cast<double>(ms_naive) / static_cast<double>(ms_shared),
               static_cast<double>(ms_sumtable) / static_cast<double>(ms_shared),
               cpu_ms_ncc);

    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting any gate below)\n");
        return 1;
    }

    // ======================= GATE variant_consistency =========================
    // The naive/sum-table/shared-memory kernels must agree with EACH OTHER far
    // tighter than any of them needs to agree with the CPU oracle (same GPU,
    // same FP32/double rounding rules for all three device sqrt() calls).
    auto pairwise_max = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float worst = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
        return worst;
    };
    const float d_ns = pairwise_max(g_scores_naive, g_scores_sumtable);
    const float d_nsh = pairwise_max(g_scores_naive, g_scores_shared);
    const float d_ssh = pairwise_max(g_scores_sumtable, g_scores_shared);
    const float worst_variant = std::max(d_ns, std::max(d_nsh, d_ssh));
    const bool variant_consistency_pass = worst_variant <= VARIANT_CONSISTENCY_TOL;
    std::printf("[info] variant_consistency: worst pairwise |diff| naive-vs-sumtable=%.3e naive-vs-shared=%.3e "
               "sumtable-vs-shared=%.3e (tol %.1e)\n",
               static_cast<double>(d_ns), static_cast<double>(d_nsh), static_cast<double>(d_ssh),
               static_cast<double>(VARIANT_CONSISTENCY_TOL));
    std::printf("GATE variant_consistency: %s\n", variant_consistency_pass ? "PASS" : "FAIL");

    // ======================= CLASSIFICATION ANALYSIS ===========================
    // Uses the SHARED-MEMORY variant's (verified) scores as the canonical
    // volume — the fastest of the three, exactly the one a deployed system
    // would run. main.cu's classification/localization/rotation/illumination
    // logic is host-only downstream analysis, NOT twinned (see
    // reference_cpu.cpp's independence ruling) — its correctness is checked
    // by the independent gates below, against the KNOWN synthetic truth.
    const std::vector<float>& scores = g_scores_shared;

    struct SlotResult {
        std::string verdict;
        BestMatch expected, single, overall;
        int overall_type = -1;
    };
    std::vector<SlotResult> results(NUM_SLOTS);

    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        const int et = truth[static_cast<size_t>(slot)].expected_type;
        SlotResult r;
        r.expected = find_best(scores, slot, et * NUM_ROT, et * NUM_ROT + NUM_ROT);
        r.single = find_best(scores, slot, template_id_single(et), template_id_single(et) + 1);
        r.overall = find_best(scores, slot, 0, NUM_TEMPLATES);
        r.overall_type = r.overall.tmpl / NUM_ROT;

        if (r.expected.score >= T_OK) r.verdict = "OK";
        else if (r.overall.score >= T_OK) r.verdict = "WRONG_PART";
        else r.verdict = "EMPTY";
        results[static_cast<size_t>(slot)] = r;
    }

    // ======================= GATE classification ================================
    bool classification_pass = true;
    int n_ok_truth = 0, n_wrong_truth = 0, n_empty_truth = 0, n_correct = 0;
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        const auto& t = truth[static_cast<size_t>(slot)];
        const auto& r = results[static_cast<size_t>(slot)];
        if (t.verdict == "OK") ++n_ok_truth; else if (t.verdict == "WRONG_PART") ++n_wrong_truth; else ++n_empty_truth;
        if (r.verdict == t.verdict) ++n_correct; else classification_pass = false;
    }
    std::printf("[info] classification: %d/%d slots correctly verdicted (truth: %d OK, %d WRONG_PART, %d EMPTY)\n",
               n_correct, NUM_SLOTS, n_ok_truth, n_wrong_truth, n_empty_truth);
    // Per-slot detail for the 6 designed cohorts (the rest are the "plain"
    // bulk — their aggregate correctness is what the gate above already
    // checked; the full 24-row table is written to per_slot_scores.csv).
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        const auto& t = truth[static_cast<size_t>(slot)];
        if (t.cohort == "plain") continue;
        const auto& r = results[static_cast<size_t>(slot)];
        std::printf("[info]   slot %2d (%-10s) truth=%-11s got=%-11s expected_score=%.3f overall_score=%.3f (type %d)\n",
                   slot, t.cohort.c_str(), t.verdict.c_str(), r.verdict.c_str(),
                   static_cast<double>(r.expected.score), static_cast<double>(r.overall.score), r.overall_type);
    }
    std::printf("GATE classification: %s\n", classification_pass ? "PASS" : "FAIL");

    // ======================= GATE localization ===================================
    // For every slot whose TRUE verdict is OK, the recovered offset (from the
    // expected-type best match) must land within LOCALIZATION_MAX_ERR_PX of
    // the applied truth offset.
    bool localization_pass = true;
    float localization_worst_px = 0.0f;
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        const auto& t = truth[static_cast<size_t>(slot)];
        if (t.verdict != "OK") continue;
        const auto& r = results[static_cast<size_t>(slot)];
        const int rec_dx = r.expected.ox - SEARCH_RADIUS, rec_dy = r.expected.oy - SEARCH_RADIUS;
        const float err = std::sqrt(static_cast<float>((rec_dx - t.offset_dx) * (rec_dx - t.offset_dx)
                                                       + (rec_dy - t.offset_dy) * (rec_dy - t.offset_dy)));
        localization_worst_px = std::max(localization_worst_px, err);
        if (err > LOCALIZATION_MAX_ERR_PX) localization_pass = false;
    }
    std::printf("[info] localization: worst recovered-offset error over all OK slots = %.2f px (max allowed %.1f px)\n",
               static_cast<double>(localization_worst_px), static_cast<double>(LOCALIZATION_MAX_ERR_PX));
    std::printf("GATE localization: %s\n", localization_pass ? "PASS" : "FAIL");

    // ======================= GATE rotation_lesson ================================
    // Find the designed "rotated" cohort slot and its expected type.
    int rot_slot = -1;
    for (int slot = 0; slot < NUM_SLOTS; ++slot)
        if (truth[static_cast<size_t>(slot)].cohort == "rotated") { rot_slot = slot; break; }

    bool rotation_lesson_pass = false;
    float rot_single_score = 0.0f, rot_set_score = 0.0f;
    std::vector<float> rot_curve(NUM_ROT, 0.0f);
    if (rot_slot >= 0) {
        const int et = truth[static_cast<size_t>(rot_slot)].expected_type;
        rot_single_score = results[static_cast<size_t>(rot_slot)].single.score;
        rot_set_score = results[static_cast<size_t>(rot_slot)].expected.score;
        // Full score-vs-angle curve: best-over-offsets score for EACH of the
        // 5 rotation templates individually (the artifact CSV plots this).
        for (int r = 0; r < NUM_ROT; ++r) {
            const BestMatch bm = find_best(scores, rot_slot, template_id(et, r), template_id(et, r) + 1);
            rot_curve[static_cast<size_t>(r)] = bm.score;
        }
        rotation_lesson_pass = (rot_single_score < T_OK) && (rot_set_score >= T_OK);
    }
    std::printf("[info] rotation_lesson: slot %d (true rotation %.1f deg) — single 0-deg template score=%.3f "
               "(must be < %.2f), 5-angle rotation-set score=%.3f (must be >= %.2f)\n",
               rot_slot, static_cast<double>(truth[static_cast<size_t>(std::max(rot_slot, 0))].rotation_deg),
               static_cast<double>(rot_single_score), static_cast<double>(T_OK),
               static_cast<double>(rot_set_score), static_cast<double>(T_OK));
    std::printf("GATE rotation_lesson: %s\n", rotation_lesson_pass ? "PASS" : "FAIL");

    // ======================= GATE illumination_robustness ========================
    // Find the "shadow" cohort slot and a same-TYPE, unshadowed "plain" slot
    // to measure the SSD baseline from (an apples-to-apples comparison).
    int shadow_slot = -1, baseline_slot = -1;
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        const auto& t = truth[static_cast<size_t>(slot)];
        if (t.cohort == "shadow") shadow_slot = slot;
    }
    if (shadow_slot >= 0) {
        const int et = truth[static_cast<size_t>(shadow_slot)].expected_type;
        for (int slot = 0; slot < NUM_SLOTS; ++slot) {
            const auto& t = truth[static_cast<size_t>(slot)];
            if (t.cohort == "plain" && t.expected_type == et) { baseline_slot = slot; break; }
        }
    }
    bool illumination_pass = false;
    double ssd_shadow = 0.0, ssd_baseline_v = 0.0, ssd_threshold = 0.0;
    float ncc_shadow_score = 0.0f;
    if (shadow_slot >= 0 && baseline_slot >= 0) {
        const int et = truth[static_cast<size_t>(shadow_slot)].expected_type;
        const uint8_t* tpl0 = h_templates.data() + static_cast<size_t>(template_id_single(et)) * TEMPLATE_PIXELS;
        ssd_baseline_v = static_cast<double>(ssd_best(h_img, baseline_slot, tpl0));
        ssd_shadow = static_cast<double>(ssd_best(h_img, shadow_slot, tpl0));
        ssd_threshold = ssd_baseline_v * SSD_REJECT_MARGIN_MULT;
        ncc_shadow_score = results[static_cast<size_t>(shadow_slot)].expected.score;
        // The designed comparison: NCC still says OK; plain SSD's best score
        // for the SAME true match has blown well past a threshold that a
        // non-shadowed same-type match satisfies comfortably — i.e. SSD
        // would reject a genuinely correct match that NCC accepts.
        illumination_pass = (ncc_shadow_score >= T_OK) && (ssd_shadow > ssd_threshold);
    }
    std::printf("[info] illumination_robustness: shadow slot %d vs baseline slot %d (both type %d) — "
               "NCC score=%.3f (must be >= %.2f); SSD best=%.0f vs baseline SSD best=%.0f, reject threshold=%.0f "
               "(%.1fx baseline; SSD must EXCEED it)\n",
               shadow_slot, baseline_slot, shadow_slot >= 0 ? truth[static_cast<size_t>(shadow_slot)].expected_type : -1,
               static_cast<double>(ncc_shadow_score), static_cast<double>(T_OK),
               ssd_shadow, ssd_baseline_v, ssd_threshold, SSD_REJECT_MARGIN_MULT);
    std::printf("GATE illumination_robustness: %s\n", illumination_pass ? "PASS" : "FAIL");

    const bool all_gates_pass = variant_consistency_pass && classification_pass && localization_pass
                               && rotation_lesson_pass && illumination_pass;

    // ======================= ARTIFACTS ==========================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifacts_ok = true;

    // tray_overlay.ppm: grayscale tray -> RGB, a colored box per slot (green
    // OK / orange WRONG_PART / blue EMPTY) plus a crosshair at the recovered
    // match position for non-EMPTY verdicts.
    {
        std::vector<uint8_t> rgb(static_cast<size_t>(IMG_W) * IMG_H * 3);
        for (int i = 0; i < IMG_PIXELS; ++i) {
            rgb[static_cast<size_t>(i) * 3] = rgb[static_cast<size_t>(i) * 3 + 1] = rgb[static_cast<size_t>(i) * 3 + 2] = h_img[static_cast<size_t>(i)];
        }
        for (int slot = 0; slot < NUM_SLOTS; ++slot) {
            const auto& r = results[static_cast<size_t>(slot)];
            uint8_t cr = 128, cg = 128, cb = 128;
            if (r.verdict == "OK") { cr = 0; cg = 220; cb = 0; }
            else if (r.verdict == "WRONG_PART") { cr = 255; cg = 140; cb = 0; }
            else { cr = 60; cg = 140; cb = 255; }
            const int x0 = slot_window_x0(slot), y0 = slot_window_y0(slot);
            draw_rect_outline(rgb, IMG_W, IMG_H, x0, y0, x0 + WINDOW, y0 + WINDOW, cr, cg, cb);
            if (r.verdict != "EMPTY") {
                const BestMatch& bm = (r.verdict == "OK") ? r.expected : r.overall;
                const int px = x0 + SEARCH_RADIUS + (bm.ox - SEARCH_RADIUS) + TEMPLATE_SIZE / 2;
                const int py = y0 + SEARCH_RADIUS + (bm.oy - SEARCH_RADIUS) + TEMPLATE_SIZE / 2;
                draw_cross(rgb, IMG_W, IMG_H, px, py, cr, cg, cb);
            }
        }
        artifacts_ok &= write_ppm(out_dir + "/tray_overlay.ppm", IMG_W, IMG_H, rgb);
    }

    // score_map_rotated_slot.pgm: for the rotation-cohort slot, the best-over-
    // rotation-set score at every offset, upscaled for visibility.
    {
        const int scale = 10;
        std::vector<uint8_t> map_img(static_cast<size_t>(NUM_OFFSETS_1D) * scale * NUM_OFFSETS_1D * scale);
        const int map_w = NUM_OFFSETS_1D * scale;
        const int slot_for_map = (rot_slot >= 0) ? rot_slot : 0;
        const int et = truth[static_cast<size_t>(slot_for_map)].expected_type;
        for (int oy = 0; oy < NUM_OFFSETS_1D; ++oy) {
            for (int ox = 0; ox < NUM_OFFSETS_1D; ++ox) {
                float best = -1.0f;
                for (int r = 0; r < NUM_ROT; ++r)
                    best = std::max(best, scores[static_cast<size_t>(score_index(slot_for_map, template_id(et, r), oy, ox))]);
                const uint8_t v = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, (best + 1.0f) * 0.5f)) * 255.0f));
                for (int sy = 0; sy < scale; ++sy)
                    for (int sx = 0; sx < scale; ++sx)
                        map_img[static_cast<size_t>(oy * scale + sy) * map_w + (ox * scale + sx)] = v;
            }
        }
        artifacts_ok &= write_pgm(out_dir + "/score_map_rotated_slot.pgm", map_w, NUM_OFFSETS_1D * scale, map_img);
    }

    // score_vs_angle.csv: the rotation_lesson curve (measured, not asserted
    // beyond the two gate checks above — the CSV is the full picture).
    {
        std::ofstream f(out_dir + "/score_vs_angle.csv");
        artifacts_ok &= f.is_open();
        if (f.is_open()) {
            f << "rotation_template_deg,best_score_over_offsets\n";
            for (int r = 0; r < NUM_ROT; ++r)
                f << ROTATION_DEG[r] << ',' << rot_curve[static_cast<size_t>(r)] << '\n';
        }
    }

    // per_slot_scores.csv: the full per-slot table (all 24 rows).
    {
        std::ofstream f(out_dir + "/per_slot_scores.csv");
        artifacts_ok &= f.is_open();
        if (f.is_open()) {
            f << "slot,cohort,expected_type,truth_verdict,got_verdict,expected_score,single_score,overall_score,"
                 "overall_type,recovered_dx,recovered_dy\n";
            for (int slot = 0; slot < NUM_SLOTS; ++slot) {
                const auto& t = truth[static_cast<size_t>(slot)];
                const auto& r = results[static_cast<size_t>(slot)];
                f << slot << ',' << t.cohort << ',' << t.expected_type << ',' << t.verdict << ',' << r.verdict << ','
                  << r.expected.score << ',' << r.single.score << ',' << r.overall.score << ',' << r.overall_type << ','
                  << (r.expected.ox - SEARCH_RADIUS) << ',' << (r.expected.oy - SEARCH_RADIUS) << '\n';
            }
        }
    }

    // gates_metrics.csv: every gate's measured value(s) and bound(s).
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        artifacts_ok &= f.is_open();
        if (f.is_open()) {
            f << "gate,measured,bound,pass\n";
            f << "variant_consistency,worst_pairwise_diff=" << worst_variant << ",<=" << VARIANT_CONSISTENCY_TOL
              << ',' << (variant_consistency_pass ? 1 : 0) << '\n';
            f << "classification,n/a," << NUM_SLOTS << "/" << NUM_SLOTS << " slots correct,"
              << (classification_pass ? 1 : 0) << '\n';
            f << "localization,worst_err_px=" << localization_worst_px << ",<=" << LOCALIZATION_MAX_ERR_PX << "px,"
              << (localization_pass ? 1 : 0) << '\n';
            f << "rotation_lesson,single=" << rot_single_score << ";set=" << rot_set_score
              << ",single<" << T_OK << ";set>=" << T_OK << ',' << (rotation_lesson_pass ? 1 : 0) << '\n';
            f << "illumination_robustness,ncc=" << ncc_shadow_score << ";ssd=" << ssd_shadow
              << ";ssd_threshold=" << ssd_threshold << ",ncc>=" << T_OK << ";ssd>threshold,"
              << (illumination_pass ? 1 : 0) << '\n';
        }
    }

    if (artifacts_ok)
        std::printf("ARTIFACT: wrote demo/out/{tray_overlay.ppm, score_map_rotated_slot.pgm, score_vs_angle.csv, "
                    "per_slot_scores.csv, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out/ files\n");

    // ======================= RESULT =============================================
    const bool success = verify_pass && all_gates_pass && artifacts_ok;
    if (success)
        std::printf("RESULT: PASS (GPU/CPU verified, all gates passed)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY:/GATE lines above for the failing stage)\n");
    return success ? 0 : 1;
}
