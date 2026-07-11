// ===========================================================================
// main.cu — entry point for project 01.13
//           Canny + Hough line/circle detection for industrial alignment
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed synthetic scene (data/sample/scene.pgm), its
//      negative-control twin (negative_control.pgm), and its ground-truth
//      geometry (truth.csv).
//   2. VERIFY STAGE: run the FULL GPU pipeline and an independently-written
//      CPU pipeline on the scene image, stage by stage, and compare —
//      float tolerance for Gaussian/Sobel/NMS, EXACT integer equality for
//      the hysteresis edge map and the Hough LINE accumulator (the
//      bit-exact headline twin), peak-level tolerance for the Hough CIRCLE
//      accumulator (an honestly-documented exception — see kernels.cu).
//   3. ANALYSIS (host-only, NOT twinned — see reference_cpu.cpp's ruling):
//      extract line/circle peaks from the verified GPU accumulators, then
//      solve the rigid alignment (dx, dy, dtheta) from the detected holes.
//   4. INDEPENDENT GATES: line_recovery, circle_recovery, alignment,
//      edge_quality, hysteresis_lesson (double- vs single-threshold, the
//      designed comparison), and negative_control (run on the part-free
//      image, expect zero detections).
//   5. ARTIFACTS: demo/out/edges.pgm, hough_lines_accum.pgm, overlay.ppm,
//      gates_metrics.csv.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY:", "GATE <name>: PASS/FAIL", "ARTIFACT:", "RESULT:" —
// they carry NO measured floats or device names, so they are deterministic
// on any machine. Measured numbers live on "[info]"/"[time]" lines, which
// are NOT diffed. Change a stable line -> update demo/expected_output.txt.
//
// Read this after: kernels.cuh (contracts) and kernels.cu (the kernels).
// Read this before: nothing — this is the top of the call graph.
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
// These are NOT pipeline data contracts (kernels.cuh owns those) — they are
// this project's ACCEPTANCE CRITERIA, MEASURED against the committed
// synthetic scene (data/sample/scene.pgm, seed 42) and then margined for
// honest headroom. See THEORY.md "How we verify correctness" for the
// measured numbers each bound was set from.
// ===========================================================================
static const float FLOAT_TOL_BLUR = 1e-2f;    // Gaussian blur, GPU vs CPU (float tolerance stage)
static const float FLOAT_TOL_GRAD = 1e-2f;    // Sobel gx/gy
static const float FLOAT_TOL_NMS  = 5e-2f;    // suppressed magnitude (sqrt adds its own rounding)

static const float LINE_MATCH_D_THETA_DEG = 3.0f;   // line_recovery: max angular error accepted
static const float LINE_MATCH_D_RHO_PX    = 3.0f;   // line_recovery: max perpendicular-offset error

static const float CIRCLE_MATCH_D_CENTER_PX = 2.0f; // circle_recovery: max center error

static const float ALIGN_MATCH_D_POS_PX     = 2.0f; // alignment gate: max |solved - truth| in dx, dy
static const float ALIGN_MATCH_D_THETA_DEG  = 1.5f; // alignment gate: max rotation error

// edge_quality: PRECISION is measured per detected PIXEL ("is this detected
// pixel near a true edge?" — a 1-px-wide set, so a tight tol_px band is the
// right comparison). RECALL is measured by SAMPLING the true curves at ~1px
// arc-length spacing and asking "is a detected pixel nearby?" — NOT by
// dividing by the band's pixel AREA, which would structurally cap recall
// near tol_px/curve_width regardless of detector quality (MEASURED: an
// early per-pixel-area version of this gate scored recall 0.33 even though
// the detector was later shown to recover ~99% of the true perimeter length
// — see THEORY.md "How we verify correctness" for the full story).
static const float EDGE_QUALITY_TOL_PX      = 1.5f;
static const float EDGE_QUALITY_MIN_PRECISION = 0.90f;
static const float EDGE_QUALITY_MIN_RECALL    = 0.90f;

static const float HYSTERESIS_LESSON_SAMPLE_RADIUS_PX = 1.5f; // "is there an edge pixel near this sample point"
static const float HYSTERESIS_LESSON_MIN_DOUBLE_FRAC = 0.7f;  // double-threshold must recover AT LEAST this much
static const float HYSTERESIS_LESSON_MAX_SINGLE_FRAC = 0.3f;  // single-threshold must recover AT MOST this much

// extract_circle_peaks windows votes over a small (2*CIRCLE_PEAK_WINDOW+1)^2
// box before picking the plane's maximum — see that function's comment for
// why (sub-pixel rounding scatters a genuine peak's votes across a few
// neighboring cells; MEASURED windowed sums vs. background noise floor are
// documented there too). WINDOW=1 (3x3), not 2: a radius-2 window MEASURED
// a real cross-talk bug on this exact scene — a WRONG-radius vote from a
// DIFFERENT hole's boundary traces a small ring of radius |r_true - r_wrong|
// around THAT hole's true center (see the function's derivation), and this
// scene's nominal radii (6, 8, 10) differ by exactly 2, so a radius-2 window
// scooped that whole false ring into one sum and out-voted the genuine
// (but more spread-out) true peak in a DIFFERENT plane — the extracted
// "r=6" circle silently reported the r=8 hole's location. A radius-1 window
// (9 cells, max reach ~1.4 px) sits safely inside every pairwise radius gap
// here. THEORY.md "Numerical considerations" keeps the full story.
static const int CIRCLE_PEAK_WINDOW = 1;

static const int   LINE_MAX_PEAKS = 8;   // cap on how many line candidates extract_line_peaks returns

// ===========================================================================
// SECTION B — tiny geometry helpers shared by the analytic mask and the
// alignment solve (host-only; see reference_cpu.cpp's ruling on why this
// downstream analysis code is deliberately NOT part of the GPU/CPU twin).
// ===========================================================================

// point_seg_distance — shortest distance from (px,py) to the FINITE segment
// (x0,y0)-(x1,y1); the standard project-and-clamp formula, identical in
// spirit to scripts/make_synthetic.py's Python twin (used there to RENDER
// the scratch mark; used here to VERIFY it was recovered).
static float point_seg_distance(float px, float py, float x0, float y0, float x1, float y1)
{
    const float dx = x1 - x0, dy = y1 - y0;
    const float len_sq = dx * dx + dy * dy;
    float t = (len_sq < 1e-9f) ? 0.0f : ((px - x0) * dx + (py - y0) * dy) / len_sq;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const float ex = x0 + t * dx, ey = y0 + t * dy;
    return std::sqrt((px - ex) * (px - ex) + (py - ey) * (py - ey));
}

// quad_refine — 3-point parabola vertex fit: given 3 EQUALLY-SPACED samples
// (c_lo at bin-1, c at bin 0, c_hi at bin+1), returns the sub-bin offset (in
// [-0.5, 0.5]) of the true continuous peak relative to bin 0. Standard
// closed-form vertex of a parabola through 3 points (THEORY.md derives it);
// the same idea 01.02's stereo disparity sub-pixel refinement uses (cite:
// projects/01-perception-cameras-vision/01.02-stereo-depth/THEORY.md
// "sub-pixel disparity refinement"), applied here to Hough accumulator bins
// instead of a stereo cost curve.
static float quad_refine(float c_lo, float c, float c_hi)
{
    const float denom = c_lo - 2.0f * c + c_hi;
    if (std::fabs(denom) < 1e-6f) return 0.0f;   // flat/degenerate — no reliable sub-bin information
    float off = 0.5f * (c_lo - c_hi) / denom;
    if (off < -0.5f) off = -0.5f;
    if (off > 0.5f) off = 0.5f;
    return off;
}

// gauss_solve4 — solve the 4x4 linear system A*x = b via Gauss-Jordan
// elimination with partial pivoting. Deliberately the simplest correct
// dense solver (no LU decomposition object, no library call) — exactly the
// kind of "small LS" project 33.01 (batched small-matrix linalg) implements
// at GPU-batch scale for hundreds of thousands of such systems at once;
// here we need exactly ONE 4x4 solve per demo run, so a plain host routine
// is the honest teaching choice (cite 33.01 as the production-scale
// version). Returns false if A is (numerically) singular.
static bool gauss_solve4(double A[4][4], double b[4], double x[4])
{
    double M[4][5];
    for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) M[i][j] = A[i][j]; M[i][4] = b[i]; }

    for (int col = 0; col < 4; ++col) {
        int piv = col;
        double best = std::fabs(M[col][col]);
        for (int r = col + 1; r < 4; ++r)
            if (std::fabs(M[r][col]) > best) { best = std::fabs(M[r][col]); piv = r; }
        if (best < 1e-9) return false;               // singular (fewer than 2 independent correspondences)
        if (piv != col) for (int j = 0; j < 5; ++j) std::swap(M[col][j], M[piv][j]);
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double f = M[r][col] / M[col][col];
            for (int j = col; j < 5; ++j) M[r][j] -= f * M[col][j];
        }
    }
    for (int i = 0; i < 4; ++i) x[i] = M[i][4] / M[i][i];
    return true;
}

// ===========================================================================
// SECTION C — ground-truth loading (data/sample/truth.csv).
// ===========================================================================
struct TruthLine { std::string name; float theta, rho, ax, ay, bx, by; };
struct TruthHole { float cx, cy, r; };
struct Truth {
    float dx = 0.0f, dy = 0.0f, dtheta = 0.0f;
    std::vector<TruthLine> lines;
    std::vector<TruthHole> holes;
    float sx0 = 0.0f, sy0 = 0.0f, sx1 = 0.0f, sy1 = 0.0f;
    bool loaded = false;
};

static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    return cells;
}

static Truth load_truth(const std::string& path)
{
    Truth t;
    std::ifstream f(path);
    if (!f.is_open()) return t;

    bool have_transform = false, have_scratch = false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto cells = split_csv(line);
        if (cells.empty()) continue;
        if (cells[0] == "TRANSFORM" && cells.size() >= 4) {
            t.dx = std::strtof(cells[1].c_str(), nullptr);
            t.dy = std::strtof(cells[2].c_str(), nullptr);
            t.dtheta = std::strtof(cells[3].c_str(), nullptr);
            have_transform = true;
        } else if (cells[0] == "LINE" && cells.size() >= 8) {
            TruthLine L;
            L.name = cells[1];
            L.theta = std::strtof(cells[2].c_str(), nullptr);
            L.rho   = std::strtof(cells[3].c_str(), nullptr);
            L.ax = std::strtof(cells[4].c_str(), nullptr);
            L.ay = std::strtof(cells[5].c_str(), nullptr);
            L.bx = std::strtof(cells[6].c_str(), nullptr);
            L.by = std::strtof(cells[7].c_str(), nullptr);
            t.lines.push_back(L);
        } else if (cells[0] == "HOLE" && cells.size() >= 4) {
            TruthHole H;
            H.cx = std::strtof(cells[1].c_str(), nullptr);
            H.cy = std::strtof(cells[2].c_str(), nullptr);
            H.r  = std::strtof(cells[3].c_str(), nullptr);
            t.holes.push_back(H);
        } else if (cells[0] == "SCRATCH" && cells.size() >= 5) {
            t.sx0 = std::strtof(cells[1].c_str(), nullptr);
            t.sy0 = std::strtof(cells[2].c_str(), nullptr);
            t.sx1 = std::strtof(cells[3].c_str(), nullptr);
            t.sy1 = std::strtof(cells[4].c_str(), nullptr);
            have_scratch = true;
        }
    }
    t.loaded = have_transform && have_scratch
             && static_cast<int>(t.lines.size()) == NUM_EDGES
             && static_cast<int>(t.holes.size()) == NUM_HOLES;
    return t;
}

// near_truth_edge — the analytic edge mask the edge_quality gate compares
// the Canny output against: true if (x,y) sits within tol_px of ANY of the
// 4 rendered plate edges (finite segments), the 3 hole boundaries, or the
// engineered scratch segment (which IS a real, if faint, rendered edge).
static bool near_truth_edge(int x, int y, const Truth& t, float tol_px)
{
    const float fx = static_cast<float>(x), fy = static_cast<float>(y);
    for (const auto& L : t.lines)
        if (point_seg_distance(fx, fy, L.ax, L.ay, L.bx, L.by) <= tol_px) return true;
    for (const auto& H : t.holes) {
        const float d = std::fabs(std::sqrt((fx - H.cx) * (fx - H.cx) + (fy - H.cy) * (fy - H.cy)) - H.r);
        if (d <= tol_px) return true;
    }
    if (point_seg_distance(fx, fy, t.sx0, t.sy0, t.sx1, t.sy1) <= tol_px) return true;
    return false;
}

// near_detected_pixel — true if any nonzero pixel of edge_map lies within
// tol_px of the continuous point (x,y). Used by the edge_quality RECALL
// computation (see that gate's comment for why recall is measured by
// sampling the true curves, not by dividing detected-pixel count by
// analytic-mask AREA) and mirrors the same small-window search
// hysteresis_lesson already uses for its own curve-sampling check.
static bool near_detected_pixel(float x, float y, const std::vector<unsigned char>& edge_map, float tol_px)
{
    const int rad = static_cast<int>(std::ceil(tol_px));
    const int xi = static_cast<int>(std::lround(x)), yi = static_cast<int>(std::lround(y));
    for (int oy = -rad; oy <= rad; ++oy) {
        const int yy = yi + oy;
        if (yy < 0 || yy >= IMG_H) continue;
        for (int ox = -rad; ox <= rad; ++ox) {
            const int xx = xi + ox;
            if (xx < 0 || xx >= IMG_W) continue;
            if (std::sqrt(static_cast<float>(ox * ox + oy * oy)) > tol_px) continue;
            if (edge_map[static_cast<size_t>(yy) * IMG_W + xx] != 0) return true;
        }
    }
    return false;
}

// ===========================================================================
// SECTION D — PGM/PPM I/O. No image library (CLAUDE.md's "no black boxes"):
// PGM/PPM are ASCII headers followed by raw samples — trivial and honest.
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
    f.get();   // consume the single whitespace byte between the header and the binary payload
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

// ===========================================================================
// SECTION E — the GPU and CPU pipeline runners.
// ===========================================================================
struct PipelineResult {
    std::vector<float> blurred, gx, gy, suppressed_mag;
    std::vector<unsigned char> state_double, edge_map_double;
    std::vector<unsigned char> state_single, edge_map_single;
    std::vector<int> line_accum;     // HOUGH_LINE_ACCUM_CELLS
    std::vector<int> circle_accum;   // HOUGH_CIRCLE_ACCUM_CELLS
    int hysteresis_sweeps = 0;
    double gpu_ms_total = 0.0;       // sum of every kernel's event-measured time (teaching artifact)
};

// run_gpu_pipeline — every GPU stage, in order, on one host image. Runs the
// double-threshold path (with sweep-counted hysteresis) AND the single-
// threshold comparison path (reusing the SAME suppressed_mag — see
// kernels.cu's classify_threshold_kernel comment on why t_low==t_high
// collapses it to a single-threshold classifier with no propagation
// needed). Allocates and frees its own device buffers — simplicity over
// micro-optimization; this project's 320x240 buffers are a few MB total.
static PipelineResult run_gpu_pipeline(const std::vector<uint8_t>& h_img)
{
    PipelineResult res;
    const int W = IMG_W, H = IMG_H, N = IMG_PIXELS;

    uint8_t* d_img = nullptr;
    float *d_tmp = nullptr, *d_blurred = nullptr, *d_gx = nullptr, *d_gy = nullptr, *d_mag = nullptr;
    unsigned char *d_state_d = nullptr, *d_edge_d = nullptr, *d_state_s = nullptr, *d_edge_s = nullptr;
    int *d_line_accum = nullptr, *d_circle_accum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_img, N));
    CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blurred, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mag, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state_d, N));
    CUDA_CHECK(cudaMalloc(&d_edge_d, N));
    CUDA_CHECK(cudaMalloc(&d_state_s, N));
    CUDA_CHECK(cudaMalloc(&d_edge_s, N));
    CUDA_CHECK(cudaMalloc(&d_line_accum, sizeof(int) * static_cast<size_t>(HOUGH_LINE_ACCUM_CELLS)));
    CUDA_CHECK(cudaMalloc(&d_circle_accum, sizeof(int) * static_cast<size_t>(HOUGH_CIRCLE_ACCUM_CELLS)));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), N, cudaMemcpyHostToDevice));

    GpuTimer gt;
    gt.begin(); launch_gaussian_blur(d_img, W, H, d_tmp, d_blurred); res.gpu_ms_total += gt.end_ms();
    gt.begin(); launch_sobel_gradient(d_blurred, W, H, d_gx, d_gy);  res.gpu_ms_total += gt.end_ms();
    gt.begin(); launch_nms(d_gx, d_gy, W, H, d_mag);                res.gpu_ms_total += gt.end_ms();

    // ---- double-threshold path: classify, then sweep hysteresis to a
    // fixed point (main.cu owns the sweep-until-converged loop and the
    // sweep counter, mirroring 01.06's CCL convergence pattern). ----------
    gt.begin(); launch_classify_threshold(d_mag, W, H, CANNY_T_LOW, CANNY_T_HIGH, d_state_d); res.gpu_ms_total += gt.end_ms();
    int sweeps = 0;
    for (; sweeps < HYSTERESIS_MAX_SWEEPS; ++sweeps) {
        gt.begin();
        const bool changed = launch_hysteresis_sweep(d_state_d, W, H);
        res.gpu_ms_total += gt.end_ms();
        if (!changed) break;
    }
    res.hysteresis_sweeps = sweeps;
    gt.begin(); launch_finalize_edge_map(d_state_d, W, H, d_edge_d); res.gpu_ms_total += gt.end_ms();

    // ---- single-threshold comparison path: t_low == t_high collapses
    // classify_threshold_kernel to strong-or-nothing, so NO hysteresis
    // sweep is needed — the classification IS already the final state
    // (see kernels.cu's Stage 4 comment). ----------------------------------
    gt.begin(); launch_classify_threshold(d_mag, W, H, CANNY_T_HIGH, CANNY_T_HIGH, d_state_s); res.gpu_ms_total += gt.end_ms();
    gt.begin(); launch_finalize_edge_map(d_state_s, W, H, d_edge_s); res.gpu_ms_total += gt.end_ms();

    // ---- Hough voting, both transforms, on the double-threshold edge map
    // (the "real" detector output the industrial measurement uses). --------
    gt.begin(); launch_hough_lines_vote(d_edge_d, W, H, d_line_accum);              res.gpu_ms_total += gt.end_ms();
    gt.begin(); launch_hough_circles_vote(d_edge_d, d_gx, d_gy, W, H, d_circle_accum); res.gpu_ms_total += gt.end_ms();

    // ---- copy everything back for verification/analysis -------------------
    res.blurred.resize(N); res.gx.resize(N); res.gy.resize(N); res.suppressed_mag.resize(N);
    res.state_double.resize(N); res.edge_map_double.resize(N);
    res.state_single.resize(N); res.edge_map_single.resize(N);
    res.line_accum.resize(static_cast<size_t>(HOUGH_LINE_ACCUM_CELLS));
    res.circle_accum.resize(static_cast<size_t>(HOUGH_CIRCLE_ACCUM_CELLS));

    CUDA_CHECK(cudaMemcpy(res.blurred.data(), d_blurred, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.gx.data(), d_gx, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.gy.data(), d_gy, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.suppressed_mag.data(), d_mag, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.state_double.data(), d_state_d, N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.edge_map_double.data(), d_edge_d, N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.state_single.data(), d_state_s, N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.edge_map_single.data(), d_edge_s, N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.line_accum.data(), d_line_accum,
                          sizeof(int) * static_cast<size_t>(HOUGH_LINE_ACCUM_CELLS), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.circle_accum.data(), d_circle_accum,
                          sizeof(int) * static_cast<size_t>(HOUGH_CIRCLE_ACCUM_CELLS), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_gx)); CUDA_CHECK(cudaFree(d_gy)); CUDA_CHECK(cudaFree(d_mag));
    CUDA_CHECK(cudaFree(d_state_d)); CUDA_CHECK(cudaFree(d_edge_d));
    CUDA_CHECK(cudaFree(d_state_s)); CUDA_CHECK(cudaFree(d_edge_s));
    CUDA_CHECK(cudaFree(d_line_accum)); CUDA_CHECK(cudaFree(d_circle_accum));
    return res;
}

// run_cpu_pipeline — the independent oracle, same stage order, called ONLY
// on the scene image (the negative control's correctness was already
// established by the scene's own GPU-vs-CPU agreement — re-running the
// oracle on it would teach nothing new; see README "Expected output").
static PipelineResult run_cpu_pipeline(const std::vector<uint8_t>& h_img,
                                       const int32_t* cos_fixed, const int32_t* sin_fixed)
{
    PipelineResult res;
    const int W = IMG_W, H = IMG_H, N = IMG_PIXELS;

    res.blurred.resize(N); res.gx.resize(N); res.gy.resize(N); res.suppressed_mag.resize(N);
    res.state_double.resize(N); res.edge_map_double.resize(N);
    res.state_single.resize(N); res.edge_map_single.resize(N);
    res.line_accum.resize(static_cast<size_t>(HOUGH_LINE_ACCUM_CELLS));
    res.circle_accum.resize(static_cast<size_t>(HOUGH_CIRCLE_ACCUM_CELLS));

    gaussian_blur_cpu(h_img.data(), W, H, res.blurred.data());
    sobel_gradient_cpu(res.blurred.data(), W, H, res.gx.data(), res.gy.data());
    nms_cpu(res.gx.data(), res.gy.data(), W, H, res.suppressed_mag.data());

    classify_threshold_cpu(res.suppressed_mag.data(), W, H, CANNY_T_LOW, CANNY_T_HIGH, res.state_double.data());
    hysteresis_propagate_cpu(res.state_double.data(), W, H);   // flood-fill to the SAME fixed point
    finalize_edge_map_cpu(res.state_double.data(), W, H, res.edge_map_double.data());

    // Single-threshold path: t_low == t_high, no weak states are ever
    // produced (see kernels.cu Stage 4), so no flood fill is needed.
    classify_threshold_cpu(res.suppressed_mag.data(), W, H, CANNY_T_HIGH, CANNY_T_HIGH, res.state_single.data());
    finalize_edge_map_cpu(res.state_single.data(), W, H, res.edge_map_single.data());

    hough_lines_accum_cpu(res.edge_map_double.data(), W, H, cos_fixed, sin_fixed, res.line_accum.data());
    hough_circles_accum_cpu(res.edge_map_double.data(), res.gx.data(), res.gy.data(), W, H, res.circle_accum.data());
    return res;
}

// ===========================================================================
// SECTION F — peak extraction + alignment solve (host-only analysis; NOT
// part of the GPU/CPU twin — see reference_cpu.cpp's independence ruling
// and the file header note above).
// ===========================================================================
struct LineCand { int t, r, votes; };

static std::vector<LineCand> find_local_maxima_2d(const std::vector<int>& accum, int t_bins, int r_bins, int min_votes)
{
    std::vector<LineCand> cands;
    for (int t = 0; t < t_bins; ++t) {
        for (int r = 0; r < r_bins; ++r) {
            const int v = accum[static_cast<size_t>(t) * r_bins + r];
            if (v < min_votes) continue;
            bool is_max = true;
            for (int dt = -1; dt <= 1 && is_max; ++dt) {
                const int tt = t + dt;
                if (tt < 0 || tt >= t_bins) continue;
                for (int dr = -1; dr <= 1; ++dr) {
                    if (dt == 0 && dr == 0) continue;
                    const int rr = r + dr;
                    if (rr < 0 || rr >= r_bins) continue;
                    if (accum[static_cast<size_t>(tt) * r_bins + rr] > v) { is_max = false; break; }
                }
            }
            if (is_max) cands.push_back({ t, r, v });
        }
    }
    std::sort(cands.begin(), cands.end(), [](const LineCand& a, const LineCand& b) { return a.votes > b.votes; });
    return cands;
}

// extract_line_peaks — local-maximum search (8-neighborhood) over the
// (theta,rho) accumulator, greedy non-max suppression by bin distance (a
// SIMPLIFICATION: it does not wrap theta across the 0/pi boundary, which
// this scene's well-separated ~7deg/~97deg lines never approach — see
// README "Limitations"), then quadratic sub-bin refinement on each axis.
static std::vector<DetectedLine> extract_line_peaks(const std::vector<int>& accum, int min_votes, int max_peaks)
{
    const auto cands = find_local_maxima_2d(accum, HOUGH_THETA_BINS, HOUGH_RHO_BINS, min_votes);
    std::vector<DetectedLine> out;
    std::vector<std::pair<int, int>> chosen_bins;
    const int SUP_T = 5, SUP_R = 8;   // bins; suppression radius around an already-accepted peak

    for (const auto& c : cands) {
        if (static_cast<int>(out.size()) >= max_peaks) break;
        bool too_close = false;
        for (const auto& b : chosen_bins)
            if (std::abs(b.first - c.t) <= SUP_T && std::abs(b.second - c.r) <= SUP_R) { too_close = true; break; }
        if (too_close) continue;

        const int t_lo = std::max(0, c.t - 1), t_hi = std::min(HOUGH_THETA_BINS - 1, c.t + 1);
        const int r_lo = std::max(0, c.r - 1), r_hi = std::min(HOUGH_RHO_BINS - 1, c.r + 1);
        const float dt = quad_refine(static_cast<float>(accum[static_cast<size_t>(t_lo) * HOUGH_RHO_BINS + c.r]),
                                     static_cast<float>(c.votes),
                                     static_cast<float>(accum[static_cast<size_t>(t_hi) * HOUGH_RHO_BINS + c.r]));
        const float dr = quad_refine(static_cast<float>(accum[static_cast<size_t>(c.t) * HOUGH_RHO_BINS + r_lo]),
                                     static_cast<float>(c.votes),
                                     static_cast<float>(accum[static_cast<size_t>(c.t) * HOUGH_RHO_BINS + r_hi]));
        DetectedLine dl;
        dl.theta = (static_cast<float>(c.t) + dt) * HOUGH_THETA_STEP;
        dl.rho   = static_cast<float>(c.r) + dr - static_cast<float>(HOUGH_RHO_MAX);
        dl.votes = c.votes;
        out.push_back(dl);
        chosen_bins.push_back({ c.t, c.r });
    }
    return out;
}

// extract_circle_peaks — ALWAYS returns exactly NUM_HOLES entries, one per
// KNOWN-RADIUS plane, even if its vote count is below
// HOUGH_CIRCLE_PEAK_MIN_VOTES — callers check .votes themselves. This keeps
// the index k (radius, and hence the correspondence to HOLE_LOCAL_X/Y[k])
// unambiguous for the alignment solve, instead of losing it by silently
// dropping low-confidence planes.
//
// WHY A WINDOWED SUM, not the single raw max cell (a real tuning story —
// see THEORY.md "Numerical considerations"): a genuine hole boundary pixel
// votes at (x + r*nx, y + r*ny) using its OWN measured unit gradient
// direction (nx,ny), rounded to the nearest integer cell. Two neighboring
// boundary pixels with slightly different (noise/quantization-perturbed)
// gradient directions round to DIFFERENT neighboring cells even though both
// are "voting for the same true center" — MEASURED on this project's own
// scene: the raw single-cell peak for the r=6 hole was only 14 votes, but
// summing a 3x3 window (CIRCLE_PEAK_WINDOW=1 — see its own comment for why
// not wider) around that same cell recovered 44 (out of ~38 boundary
// pixels, the expected full circumference) — the votes were never lost,
// just scattered across a handful of adjacent cells. Smoothing the
// accumulator before peak-picking is the standard fix (OpenCV's own
// HoughCircles does exactly this — see README "Prior art"); this project
// implements it as a direct windowed sum for maximum readability.
static std::vector<DetectedCircle> extract_circle_peaks(const std::vector<int>& accum)
{
    std::vector<DetectedCircle> out(NUM_HOLES);
    const int W = CIRCLE_PEAK_WINDOW;

    for (int k = 0; k < NUM_HOLES; ++k) {
        const int* plane = &accum[static_cast<size_t>(k) * IMG_W * IMG_H];

        // windowed[y][x] = sum of plane over the (2W+1)x(2W+1) box centered
        // at (x,y) — a direct O(IMG_W*IMG_H*(2W+1)^2) computation (25 cells
        // per output at W=2): simple and plenty fast for a one-shot demo
        // over a 320x240 plane (see README Exercises for the separable-
        // prefix-sum speedup a hot path would want instead).
        std::vector<int> windowed(static_cast<size_t>(IMG_W) * IMG_H, 0);
        for (int y = 0; y < IMG_H; ++y) {
            for (int x = 0; x < IMG_W; ++x) {
                int s = 0;
                for (int oy = -W; oy <= W; ++oy) {
                    const int yy = y + oy;
                    if (yy < 0 || yy >= IMG_H) continue;
                    for (int ox = -W; ox <= W; ++ox) {
                        const int xx = x + ox;
                        if (xx < 0 || xx >= IMG_W) continue;
                        s += plane[yy * IMG_W + xx];
                    }
                }
                windowed[static_cast<size_t>(y) * IMG_W + x] = s;
            }
        }

        int best_v = -1, best_x = 0, best_y = 0;
        for (int y = 0; y < IMG_H; ++y) {
            for (int x = 0; x < IMG_W; ++x) {
                const int v = windowed[static_cast<size_t>(y) * IMG_W + x];
                if (v > best_v) { best_v = v; best_x = x; best_y = y; }
            }
        }
        const int xlo = std::max(0, best_x - 1), xhi = std::min(IMG_W - 1, best_x + 1);
        const int ylo = std::max(0, best_y - 1), yhi = std::min(IMG_H - 1, best_y + 1);
        const float dxr = quad_refine(static_cast<float>(windowed[static_cast<size_t>(best_y) * IMG_W + xlo]),
                                      static_cast<float>(best_v),
                                      static_cast<float>(windowed[static_cast<size_t>(best_y) * IMG_W + xhi]));
        const float dyr = quad_refine(static_cast<float>(windowed[static_cast<size_t>(ylo) * IMG_W + best_x]),
                                      static_cast<float>(best_v),
                                      static_cast<float>(windowed[static_cast<size_t>(yhi) * IMG_W + best_x]));
        out[k].cx = static_cast<float>(best_x) + dxr;
        out[k].cy = static_cast<float>(best_y) + dyr;
        out[k].r  = HOLE_RADIUS[k];
        out[k].votes = best_v;
    }
    return out;
}

// solve_alignment_ls — the industrial measurement itself: given the
// detected hole centers (KNOWN correspondence to HOLE_LOCAL_X/Y by radius
// index — no data-association search needed, unlike generic point-set
// registration), solve the linear system for the rigid transform
// (a,b,tx,ty) with a = cos(dtheta), b = sin(dtheta) via normal equations
// (THEORY.md "The math" derives why q = R(dtheta)*p_local + c_img + t is
// LINEAR in (a,b,tx,ty) even though it is not linear in dtheta itself).
// Only uses circles that cleared HOUGH_CIRCLE_PEAK_MIN_VOTES; needs >= 2 to
// be well-posed (4 unknowns, 2 equations per correspondence).
static AlignmentResult solve_alignment_ls(const std::vector<DetectedCircle>& detected)
{
    AlignmentResult res{ 0.0f, 0.0f, 0.0f, false };
    double AtA[4][4] = {{0}};
    double Atb[4] = {0, 0, 0, 0};
    int n_used = 0;

    for (int k = 0; k < NUM_HOLES; ++k) {
        if (detected[k].votes < HOUGH_CIRCLE_PEAK_MIN_VOTES) continue;
        const double lx = HOLE_LOCAL_X[k], ly = HOLE_LOCAL_Y[k];
        const double qx = static_cast<double>(detected[k].cx) - IMG_CX;
        const double qy = static_cast<double>(detected[k].cy) - IMG_CY;
        // eq_x: a*lx - b*ly + tx = qx   (row over unknowns [a,b,tx,ty])
        // eq_y: b*lx + a*ly + ty = qy
        const double rowx[4] = { lx, -ly, 1.0, 0.0 };
        const double rowy[4] = { ly,  lx, 0.0, 1.0 };
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) AtA[i][j] += rowx[i] * rowx[j] + rowy[i] * rowy[j];
            Atb[i] += rowx[i] * qx + rowy[i] * qy;
        }
        ++n_used;
    }
    if (n_used < 2) return res;   // fewer than 2 correspondences: the 4-unknown system is underdetermined

    double p[4];
    if (!gauss_solve4(AtA, Atb, p)) return res;
    res.dx = static_cast<float>(p[2]);
    res.dy = static_cast<float>(p[3]);
    res.dtheta = static_cast<float>(std::atan2(p[1], p[0]));
    res.solved = true;
    return res;
}

// ===========================================================================
// SECTION G — small drawing helpers for the overlay.ppm artifact (hand-
// rolled, didactic visualizations — not a graphics library; see 01.04's
// main.cu for the same framing on its own overlay output).
// ===========================================================================
static void put_px(std::vector<uint8_t>& rgb, int w, int h, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    const size_t i = (static_cast<size_t>(y) * w + x) * 3;
    rgb[i] = r; rgb[i + 1] = g; rgb[i + 2] = b;
}

// draw_line_full — draws the FULL infinite Hough line (theta,rho) across
// the image by stepping along it in the direction perpendicular to the
// normal (a simple parametric walk, not a generic clipping algorithm).
static void draw_line_full(std::vector<uint8_t>& rgb, int w, int h, float theta, float rho,
                           uint8_t r, uint8_t g, uint8_t b)
{
    const float nx = std::cos(theta), ny = std::sin(theta);   // the line's normal
    const float dx = -ny, dy = nx;                             // direction ALONG the line
    const float px0 = rho * nx, py0 = rho * ny;                // one point on the line (closest to origin)
    const float half_span = static_cast<float>(w + h);         // generous enough to cross the whole image
    const int steps = static_cast<int>(2.0f * half_span);
    for (int s = 0; s <= steps; ++s) {
        const float t = -half_span + static_cast<float>(s);
        const int x = static_cast<int>(std::lround(px0 + t * dx));
        const int y = static_cast<int>(std::lround(py0 + t * dy));
        put_px(rgb, w, h, x, y, r, g, b);
    }
}

static void draw_circle(std::vector<uint8_t>& rgb, int w, int h, float cx, float cy, float radius,
                        uint8_t r, uint8_t g, uint8_t b)
{
    const int steps = std::max(16, static_cast<int>(2.0f * PI_F * radius));
    for (int s = 0; s < steps; ++s) {
        const float a = 2.0f * PI_F * static_cast<float>(s) / static_cast<float>(steps);
        const int x = static_cast<int>(std::lround(cx + radius * std::cos(a)));
        const int y = static_cast<int>(std::lround(cy + radius * std::sin(a)));
        put_px(rgb, w, h, x, y, r, g, b);
    }
}

static void draw_thick_segment(std::vector<uint8_t>& rgb, int w, int h, float x0, float y0, float x1, float y1,
                               uint8_t r, uint8_t g, uint8_t b)
{
    const float dx = x1 - x0, dy = y1 - y0;
    const int steps = std::max(1, static_cast<int>(std::sqrt(dx * dx + dy * dy)) * 2);
    for (int s = 0; s <= steps; ++s) {
        const float t = static_cast<float>(s) / static_cast<float>(steps);
        const int x = static_cast<int>(std::lround(x0 + t * dx));
        const int y = static_cast<int>(std::lround(y0 + t * dy));
        for (int oy = -1; oy <= 1; ++oy)
            for (int ox = -1; ox <= 1; ++ox)
                put_px(rgb, w, h, x + ox, y + oy, r, g, b);
    }
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

    std::printf("[demo] Canny + Hough line/circle detection for industrial alignment (project 01.13)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d synthetic machined plate, Canny (Gaussian 5-tap -> Sobel -> NMS -> "
                "double-threshold hysteresis) -> Hough lines (%d theta bins x %d rho bins) + Hough "
                "circles (%d known radii) -> rigid alignment LS, FP32/int32\n",
                IMG_W, IMG_H, HOUGH_THETA_BINS, HOUGH_RHO_BINS, NUM_HOLES);

    // ---- 1) data ------------------------------------------------------------
    const std::string scene_path = find_data_file(cli_data_dir, argv[0], "scene.pgm");
    const std::string negctrl_path = find_data_file(cli_data_dir, argv[0], "negative_control.pgm");
    const std::string truth_path = find_data_file(cli_data_dir, argv[0], "truth.csv");
    if (scene_path.empty() || negctrl_path.empty() || truth_path.empty()) {
        std::printf("DATA: NOT FOUND — data/sample/{scene.pgm,negative_control.pgm,truth.csv} missing "
                    "(run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::vector<uint8_t> h_scene, h_negctrl;
    const bool scene_ok = load_pgm(scene_path, IMG_W, IMG_H, h_scene);
    const bool negctrl_ok = load_pgm(negctrl_path, IMG_W, IMG_H, h_negctrl);
    const Truth truth = load_truth(truth_path);
    if (!scene_ok || !negctrl_ok || !truth.loaded) {
        std::printf("DATA: MALFORMED — see file paths above\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }
    std::printf("DATA: loaded synthetic scene.pgm + negative_control.pgm + truth.csv "
               "(4 edges, %d holes, 1 hysteresis-lesson scratch mark, applied transform known) [synthetic]\n",
               NUM_HOLES);
    std::printf("[info] scene: %s | negative control: %s | truth: %s\n",
               scene_path.c_str(), negctrl_path.c_str(), truth_path.c_str());
    std::printf("[info] applied truth transform: dx=%.3f px, dy=%.3f px, dtheta=%.3f deg\n",
               static_cast<double>(truth.dx), static_cast<double>(truth.dy),
               static_cast<double>(truth.dtheta) * (180.0 / 3.14159265358979323846));

    // ---- 2) fixed-point Hough theta table (shared data contract; see
    // kernels.cuh SECTION 5) ---------------------------------------------------
    std::vector<int32_t> cos_fixed(HOUGH_THETA_BINS), sin_fixed(HOUGH_THETA_BINS);
    build_hough_theta_table_fixed(cos_fixed.data(), sin_fixed.data());
    upload_hough_constants(cos_fixed.data(), sin_fixed.data());

    // ======================= VERIFY STAGE (scene image) =======================
    const PipelineResult gpu = run_gpu_pipeline(h_scene);
    CpuTimer cpu_wall; cpu_wall.begin();
    const PipelineResult cpu = run_cpu_pipeline(h_scene, cos_fixed.data(), sin_fixed.data());
    const double cpu_ms = cpu_wall.end_ms();

    bool verify_pass = true;

    // -- float-tolerance stages: blurred, gx, gy, suppressed_mag -------------
    auto max_abs_diff = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float worst = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
        return worst;
    };
    const float d_blur = max_abs_diff(gpu.blurred, cpu.blurred);
    const float d_gx = max_abs_diff(gpu.gx, cpu.gx);
    const float d_gy = max_abs_diff(gpu.gy, cpu.gy);
    const float d_nms = max_abs_diff(gpu.suppressed_mag, cpu.suppressed_mag);
    const bool float_stage_pass = (d_blur <= FLOAT_TOL_BLUR) && (d_gx <= FLOAT_TOL_GRAD)
                                && (d_gy <= FLOAT_TOL_GRAD) && (d_nms <= FLOAT_TOL_NMS);
    verify_pass = verify_pass && float_stage_pass;
    std::printf("[info] float-tolerance worst |GPU-CPU|: blur=%.4g (tol %.4g), gx=%.4g, gy=%.4g (tol %.4g), "
               "nms=%.4g (tol %.4g)\n",
               static_cast<double>(d_blur), static_cast<double>(FLOAT_TOL_BLUR),
               static_cast<double>(d_gx), static_cast<double>(d_gy), static_cast<double>(FLOAT_TOL_GRAD),
               static_cast<double>(d_nms), static_cast<double>(FLOAT_TOL_NMS));
    std::printf("VERIFY: gaussian+sobel+nms GPU vs CPU within float tolerance %s\n",
               float_stage_pass ? "PASS" : "FAIL");

    // -- EXACT integer stages: hysteresis edge maps (double AND single) ------
    const bool edge_double_exact = (gpu.edge_map_double == cpu.edge_map_double);
    const bool edge_single_exact = (gpu.edge_map_single == cpu.edge_map_single);
    verify_pass = verify_pass && edge_double_exact && edge_single_exact;
    std::printf("[info] hysteresis fixed point: GPU converged in %d sweeps (sync repeated-sweep scan); "
               "CPU converged via an independent queue-based flood fill to the same state\n",
               gpu.hysteresis_sweeps);
    std::printf("VERIFY: hysteresis edge map (double- and single-threshold) GPU vs CPU EXACT %s\n",
               (edge_double_exact && edge_single_exact) ? "PASS" : "FAIL");

    // -- BIT-EXACT: the Hough LINE accumulator, the headline twin -----------
    const bool line_accum_exact = (gpu.line_accum == cpu.line_accum);
    verify_pass = verify_pass && line_accum_exact;
    std::printf("VERIFY: hough line accumulator (%lld cells, integer atomics) GPU vs CPU BIT-EXACT %s\n",
               static_cast<long long>(HOUGH_LINE_ACCUM_CELLS), line_accum_exact ? "PASS" : "FAIL");

    // -- Hough CIRCLE accumulator: peak-level tolerance (honest exception —
    // see kernels.cu's circle-voting kernel comment) -------------------------
    const auto circ_peaks_gpu = extract_circle_peaks(gpu.circle_accum);
    const auto circ_peaks_cpu = extract_circle_peaks(cpu.circle_accum);
    bool circle_accum_ok = true;
    float circle_peak_worst_px = 0.0f;
    int circle_peak_worst_votes = 0;
    for (int k = 0; k < NUM_HOLES; ++k) {
        const float dpx = std::sqrt((circ_peaks_gpu[k].cx - circ_peaks_cpu[k].cx) * (circ_peaks_gpu[k].cx - circ_peaks_cpu[k].cx)
                                   + (circ_peaks_gpu[k].cy - circ_peaks_cpu[k].cy) * (circ_peaks_gpu[k].cy - circ_peaks_cpu[k].cy));
        const int dvotes = std::abs(circ_peaks_gpu[k].votes - circ_peaks_cpu[k].votes);
        circle_peak_worst_px = std::max(circle_peak_worst_px, dpx);
        circle_peak_worst_votes = std::max(circle_peak_worst_votes, dvotes);
        if (dpx > 1.0f || dvotes > 2) circle_accum_ok = false;
    }
    verify_pass = verify_pass && circle_accum_ok;
    std::printf("[info] hough circle accumulator peak agreement: worst center delta %.3f px, worst vote "
               "delta %d (inherits float tolerance from Sobel gradients, not fixed-point — see kernels.cu)\n",
               static_cast<double>(circle_peak_worst_px), circle_peak_worst_votes);
    std::printf("VERIFY: hough circle accumulator GPU vs CPU peak-level tolerance %s\n",
               circle_accum_ok ? "PASS" : "FAIL");

    std::printf("[time] GPU pipeline (all stages, event-timed): %.3f ms | CPU pipeline (wall clock): %.1f ms | "
               "speed-up (teaching artifact): %.0fx\n",
               gpu.gpu_ms_total, cpu_ms, cpu_ms / (gpu.gpu_ms_total > 0.0 ? gpu.gpu_ms_total : 1.0));

    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting any detection below)\n");
        return 1;
    }

    // ======================= ANALYSIS (host-only, not twinned) ================
    const auto lines = extract_line_peaks(gpu.line_accum, HOUGH_LINE_PEAK_MIN_VOTES, LINE_MAX_PEAKS);
    const auto circles = extract_circle_peaks(gpu.circle_accum);
    const AlignmentResult align = solve_alignment_ls(circles);

    std::printf("[info] detected %zu line peak(s) (min votes %d), %d/%d circle plane(s) above vote threshold\n",
               lines.size(), HOUGH_LINE_PEAK_MIN_VOTES,
               static_cast<int>(std::count_if(circles.begin(), circles.end(),
                   [](const DetectedCircle& c) { return c.votes >= HOUGH_CIRCLE_PEAK_MIN_VOTES; })),
               NUM_HOLES);
    for (const auto& L : lines)
        std::printf("[info]   line: theta=%.2f deg rho=%.2f px votes=%d\n",
                   static_cast<double>(L.theta) * (180.0 / 3.14159265358979323846), static_cast<double>(L.rho), L.votes);
    for (int k = 0; k < NUM_HOLES; ++k)
        std::printf("[info]   circle[r=%.0f]: cx=%.2f cy=%.2f votes=%d\n",
                   static_cast<double>(circles[k].r), static_cast<double>(circles[k].cx),
                   static_cast<double>(circles[k].cy), circles[k].votes);
    if (align.solved)
        std::printf("[info] recovered alignment: dx=%.3f px, dy=%.3f px, dtheta=%.3f deg (truth: dx=%.3f, dy=%.3f, dtheta=%.3f)\n",
                   static_cast<double>(align.dx), static_cast<double>(align.dy),
                   static_cast<double>(align.dtheta) * (180.0 / 3.14159265358979323846),
                   static_cast<double>(truth.dx), static_cast<double>(truth.dy),
                   static_cast<double>(truth.dtheta) * (180.0 / 3.14159265358979323846));
    else
        std::printf("[info] alignment NOT solved (fewer than 2 confident circle correspondences)\n");

    // ======================= GATES =============================================
    bool all_gates_pass = true;

    // -- line_recovery: every truth line matched by a detected peak ----------
    bool line_recovery_pass = true;
    for (const auto& TL : truth.lines) {
        bool matched = false;
        for (const auto& DL : lines) {
            const float dtheta_deg = std::fabs(DL.theta - TL.theta) * (180.0f / PI_F);
            const float drho = std::fabs(DL.rho - TL.rho);
            if (dtheta_deg <= LINE_MATCH_D_THETA_DEG && drho <= LINE_MATCH_D_RHO_PX) { matched = true; break; }
        }
        if (!matched) line_recovery_pass = false;
    }
    all_gates_pass = all_gates_pass && line_recovery_pass;
    std::printf("GATE line_recovery: %s\n", line_recovery_pass ? "PASS" : "FAIL");

    // -- circle_recovery: every truth hole matched by its (same-index)
    // detected circle within center tolerance, at sufficient votes ----------
    bool circle_recovery_pass = true;
    for (int k = 0; k < NUM_HOLES; ++k) {
        const float dpx = std::sqrt((circles[k].cx - truth.holes[k].cx) * (circles[k].cx - truth.holes[k].cx)
                                   + (circles[k].cy - truth.holes[k].cy) * (circles[k].cy - truth.holes[k].cy));
        if (circles[k].votes < HOUGH_CIRCLE_PEAK_MIN_VOTES || dpx > CIRCLE_MATCH_D_CENTER_PX)
            circle_recovery_pass = false;
    }
    all_gates_pass = all_gates_pass && circle_recovery_pass;
    std::printf("GATE circle_recovery: %s\n", circle_recovery_pass ? "PASS" : "FAIL");

    // -- alignment: the business gate — recovered transform vs applied truth -
    bool alignment_pass = align.solved;
    if (alignment_pass) {
        float dtheta_err_deg = std::fabs(align.dtheta - truth.dtheta) * (180.0f / PI_F);
        if (dtheta_err_deg > 180.0f) dtheta_err_deg = 360.0f - dtheta_err_deg;   // wrap, defensive
        alignment_pass = std::fabs(align.dx - truth.dx) <= ALIGN_MATCH_D_POS_PX
                       && std::fabs(align.dy - truth.dy) <= ALIGN_MATCH_D_POS_PX
                       && dtheta_err_deg <= ALIGN_MATCH_D_THETA_DEG;
    }
    all_gates_pass = all_gates_pass && alignment_pass;
    std::printf("GATE alignment: %s\n", alignment_pass ? "PASS" : "FAIL");

    // -- edge_quality: precision (per detected pixel) + recall (per sampled
    // arc-length point of the true curves) of the double-threshold edge map
    // against the analytic (truth-geometry) edge mask — see the tolerance
    // constants' comment above for why these use DIFFERENT denominators. --
    long long det_total = 0, det_near_truth = 0;
    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            if (gpu.edge_map_double[static_cast<size_t>(y) * IMG_W + x] == 0) continue;
            ++det_total;
            if (near_truth_edge(x, y, truth, EDGE_QUALITY_TOL_PX)) ++det_near_truth;
        }
    }
    const float precision = det_total > 0 ? static_cast<float>(det_near_truth) / static_cast<float>(det_total) : 0.0f;

    long long curve_samples = 0, curve_hit = 0;
    auto sample_curve = [&](float x0, float y0, float x1, float y1) {
        const float len = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        const int n = std::max(1, static_cast<int>(len));
        for (int s = 0; s <= n; ++s) {
            const float t = static_cast<float>(s) / static_cast<float>(n);
            ++curve_samples;
            if (near_detected_pixel(x0 + t * (x1 - x0), y0 + t * (y1 - y0), gpu.edge_map_double, EDGE_QUALITY_TOL_PX))
                ++curve_hit;
        }
    };
    for (const auto& L : truth.lines) sample_curve(L.ax, L.ay, L.bx, L.by);
    for (const auto& H : truth.holes) {
        const int n = std::max(8, static_cast<int>(2.0f * PI_F * H.r));
        for (int s = 0; s < n; ++s) {
            const float a = 2.0f * PI_F * static_cast<float>(s) / static_cast<float>(n);
            ++curve_samples;
            if (near_detected_pixel(H.cx + H.r * std::cos(a), H.cy + H.r * std::sin(a),
                                    gpu.edge_map_double, EDGE_QUALITY_TOL_PX))
                ++curve_hit;
        }
    }
    sample_curve(truth.sx0, truth.sy0, truth.sx1, truth.sy1);
    const float recall = curve_samples > 0 ? static_cast<float>(curve_hit) / static_cast<float>(curve_samples) : 0.0f;

    const bool edge_quality_pass = precision >= EDGE_QUALITY_MIN_PRECISION && recall >= EDGE_QUALITY_MIN_RECALL;
    all_gates_pass = all_gates_pass && edge_quality_pass;
    std::printf("[info] edge_quality: precision=%.3f (min %.2f, %lld/%lld detected px near truth) "
               "recall=%.3f (min %.2f, %lld/%lld true-curve samples matched) [tol band %.1f px]\n",
               static_cast<double>(precision), static_cast<double>(EDGE_QUALITY_MIN_PRECISION),
               det_near_truth, det_total,
               static_cast<double>(recall), static_cast<double>(EDGE_QUALITY_MIN_RECALL),
               curve_hit, curve_samples, static_cast<double>(EDGE_QUALITY_TOL_PX));
    std::printf("GATE edge_quality: %s\n", edge_quality_pass ? "PASS" : "FAIL");

    // -- hysteresis_lesson: the designed double- vs single-threshold
    // comparison over the engineered scratch segment --------------------------
    const float scratch_len = std::sqrt((truth.sx1 - truth.sx0) * (truth.sx1 - truth.sx0)
                                       + (truth.sy1 - truth.sy0) * (truth.sy1 - truth.sy0));
    const int n_samples = std::max(2, static_cast<int>(scratch_len));
    int recovered_double = 0, recovered_single = 0;
    for (int s = 0; s <= n_samples; ++s) {
        const float t = static_cast<float>(s) / static_cast<float>(n_samples);
        const float sx = truth.sx0 + t * (truth.sx1 - truth.sx0);
        const float sy = truth.sy0 + t * (truth.sy1 - truth.sy0);
        bool near_double = false, near_single = false;
        const int rad = static_cast<int>(std::ceil(HYSTERESIS_LESSON_SAMPLE_RADIUS_PX));
        for (int oy = -rad; oy <= rad && !(near_double && near_single); ++oy) {
            for (int ox = -rad; ox <= rad; ++ox) {
                const int xx = static_cast<int>(std::lround(sx)) + ox, yy = static_cast<int>(std::lround(sy)) + oy;
                if (xx < 0 || xx >= IMG_W || yy < 0 || yy >= IMG_H) continue;
                if (std::sqrt(static_cast<float>(ox * ox + oy * oy)) > HYSTERESIS_LESSON_SAMPLE_RADIUS_PX) continue;
                const size_t idx = static_cast<size_t>(yy) * IMG_W + xx;
                if (gpu.edge_map_double[idx] != 0) near_double = true;
                if (gpu.edge_map_single[idx] != 0) near_single = true;
            }
        }
        if (near_double) ++recovered_double;
        if (near_single) ++recovered_single;
    }
    const float frac_double = static_cast<float>(recovered_double) / static_cast<float>(n_samples + 1);
    const float frac_single = static_cast<float>(recovered_single) / static_cast<float>(n_samples + 1);
    const bool hysteresis_lesson_pass = frac_double >= HYSTERESIS_LESSON_MIN_DOUBLE_FRAC
                                      && frac_single <= HYSTERESIS_LESSON_MAX_SINGLE_FRAC;
    all_gates_pass = all_gates_pass && hysteresis_lesson_pass;
    std::printf("[info] hysteresis_lesson: scratch mark recovered fraction — double-threshold %.2f (min %.2f), "
               "single-threshold %.2f (max %.2f)\n",
               static_cast<double>(frac_double), static_cast<double>(HYSTERESIS_LESSON_MIN_DOUBLE_FRAC),
               static_cast<double>(frac_single), static_cast<double>(HYSTERESIS_LESSON_MAX_SINGLE_FRAC));
    std::printf("GATE hysteresis_lesson: %s\n", hysteresis_lesson_pass ? "PASS" : "FAIL");

    // ======================= NEGATIVE CONTROL ==================================
    // No CPU twin needed here (correctness already established on the scene
    // image above) — just the GPU path and the same peak extraction.
    const PipelineResult neg = run_gpu_pipeline(h_negctrl);
    const auto neg_lines = extract_line_peaks(neg.line_accum, HOUGH_LINE_PEAK_MIN_VOTES, LINE_MAX_PEAKS);
    const auto neg_circles = extract_circle_peaks(neg.circle_accum);
    const bool neg_circles_clear = std::none_of(neg_circles.begin(), neg_circles.end(),
        [](const DetectedCircle& c) { return c.votes >= HOUGH_CIRCLE_PEAK_MIN_VOTES; });
    const bool negative_control_pass = neg_lines.empty() && neg_circles_clear;
    all_gates_pass = all_gates_pass && negative_control_pass;
    std::printf("[info] negative_control: %zu line peak(s), %d circle plane(s) above vote threshold "
               "(both must be zero)\n", neg_lines.size(),
               static_cast<int>(std::count_if(neg_circles.begin(), neg_circles.end(),
                   [](const DetectedCircle& c) { return c.votes >= HOUGH_CIRCLE_PEAK_MIN_VOTES; })));
    std::printf("GATE negative_control: %s\n", negative_control_pass ? "PASS" : "FAIL");

    // ======================= ARTIFACTS ==========================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifacts_ok = true;

    artifacts_ok &= write_pgm(out_dir + "/edges.pgm", IMG_W, IMG_H, gpu.edge_map_double);

    // Log-stretched line accumulator, laid out (rho across, theta down) so it
    // reads like a classic Hough-space image: each bright horizontal streak
    // is one real line's sinusoid family converging on its peak column.
    {
        int max_v = 1;
        for (int v : gpu.line_accum) max_v = std::max(max_v, v);
        std::vector<uint8_t> accum_img(static_cast<size_t>(HOUGH_THETA_BINS) * HOUGH_RHO_BINS);
        const float log_max = std::log(1.0f + static_cast<float>(max_v));
        for (int t = 0; t < HOUGH_THETA_BINS; ++t)
            for (int r = 0; r < HOUGH_RHO_BINS; ++r) {
                const float v = static_cast<float>(gpu.line_accum[static_cast<size_t>(t) * HOUGH_RHO_BINS + r]);
                const float s = std::log(1.0f + v) / log_max;
                accum_img[static_cast<size_t>(t) * HOUGH_RHO_BINS + r] = static_cast<uint8_t>(std::lround(s * 255.0f));
            }
        artifacts_ok &= write_pgm(out_dir + "/hough_lines_accum.pgm", HOUGH_RHO_BINS, HOUGH_THETA_BINS, accum_img);
    }

    // Overlay: scene (grayscale->RGB) + detected lines (green) + detected
    // circles (red) + the alignment story (yellow: nominal-vs-recovered
    // part outline center marker).
    {
        std::vector<uint8_t> rgb(static_cast<size_t>(IMG_W) * IMG_H * 3);
        for (int i = 0; i < IMG_PIXELS; ++i) { rgb[static_cast<size_t>(i) * 3] = rgb[static_cast<size_t>(i) * 3 + 1]
                                                = rgb[static_cast<size_t>(i) * 3 + 2] = h_scene[i]; }
        for (const auto& L : lines) draw_line_full(rgb, IMG_W, IMG_H, L.theta, L.rho, 0, 255, 0);
        for (int k = 0; k < NUM_HOLES; ++k)
            if (circles[k].votes >= HOUGH_CIRCLE_PEAK_MIN_VOTES)
                draw_circle(rgb, IMG_W, IMG_H, circles[k].cx, circles[k].cy, circles[k].r, 255, 0, 0);
        if (align.solved)
            draw_thick_segment(rgb, IMG_W, IMG_H, IMG_CX, IMG_CY, IMG_CX + align.dx, IMG_CY + align.dy, 255, 255, 0);
        artifacts_ok &= write_ppm(out_dir + "/overlay.ppm", IMG_W, IMG_H, rgb);
    }

    // gates_metrics.csv: every gate's measured value(s) and bound(s), for
    // the learner to inspect or plot outside the demo.
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        if (f.is_open()) {
            f << "gate,measured,bound,pass\n";
            f << "line_recovery,n/a,d_theta<=" << LINE_MATCH_D_THETA_DEG << "deg;d_rho<=" << LINE_MATCH_D_RHO_PX
              << "px," << (line_recovery_pass ? 1 : 0) << "\n";
            f << "circle_recovery,n/a,d_center<=" << CIRCLE_MATCH_D_CENTER_PX << "px,"
              << (circle_recovery_pass ? 1 : 0) << "\n";
            f << "alignment,dx=" << align.dx << ";dy=" << align.dy << ";dtheta_deg="
              << (align.dtheta * 180.0f / PI_F) << ",d_pos<=" << ALIGN_MATCH_D_POS_PX
              << "px;d_theta<=" << ALIGN_MATCH_D_THETA_DEG << "deg," << (alignment_pass ? 1 : 0) << "\n";
            f << "edge_quality,precision=" << precision << ";recall=" << recall
              << ",precision>=" << EDGE_QUALITY_MIN_PRECISION << ";recall>=" << EDGE_QUALITY_MIN_RECALL
              << "," << (edge_quality_pass ? 1 : 0) << "\n";
            f << "hysteresis_lesson,double_frac=" << frac_double << ";single_frac=" << frac_single
              << ",double_frac>=" << HYSTERESIS_LESSON_MIN_DOUBLE_FRAC << ";single_frac<="
              << HYSTERESIS_LESSON_MAX_SINGLE_FRAC << "," << (hysteresis_lesson_pass ? 1 : 0) << "\n";
            f << "negative_control,lines=" << neg_lines.size() << ",lines==0 && circle_planes==0,"
              << (negative_control_pass ? 1 : 0) << "\n";
        } else {
            artifacts_ok = false;
        }
    }

    if (artifacts_ok)
        std::printf("ARTIFACT: wrote demo/out/{edges.pgm, hough_lines_accum.pgm, overlay.ppm, gates_metrics.csv}\n");
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
