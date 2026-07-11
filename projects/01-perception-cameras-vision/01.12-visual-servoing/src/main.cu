// ===========================================================================
// main.cu — entry point for project 01.12
//           Visual servoing: image-Jacobian control loop entirely on GPU
//           (teaching core: eye-in-hand IBVS, batched convergence-basin study)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (K, the
//      basin-map grid side) from data/sample/.
//   2. Build the target/goal geometry (closed form) and upload it to GPU
//      __constant__ memory; generate the K loops' initial poses across the
//      three designed cohorts (kernels.cuh "COHORTS") on the HOST.
//   3. Run the GPU rollout-farm kernel for all THREE controller variants
//      over the SAME initial-pose batch (the didactic comparison), plus a
//      separate structured (dx,dy) GRID batch for the basin-map artifact.
//   4. VERIFY STAGE (the §5 GPU-vs-CPU gate, applied at three grains):
//        a) a single loop's full 400-step trajectory,
//        b) the Jacobian/pseudoinverse linear algebra at 16 sampled poses,
//        c) a 128-loop subset of the full batch's summary statistics.
//   5. INDEPENDENT GATES (control-theory / literature predictions that do
//      not route through either twin — see reference_cpu.cpp's ruling):
//      exponential_decay, convergence_basin, retreat_pathology; plus two
//      [info]-only comparisons: depth_robustness, conditioning_honesty.
//   6. ARTIFACTS: two PPM images and three CSVs into demo/out/.
//
// This project is a CONTROL project living in the vision domain (the
// catalog bullet: "image-Jacobian control loop entirely on GPU") — its
// nearest kin is 08.01 (MPPI): read that project's main.cu first if this is
// your entry point, and note the shared shape (verify stage, then the
// GPU-driven study, then artifacts) applied here to a CLOSED-loop
// convergence-basin study instead of an open-loop single trajectory.
//
// Determinism: every random draw in this program (the cohort RNG) is
// host-generated xorshift32 from a fixed seed (42) — see
// reference_cpu.cpp's generate_batch_init_poses_cpu. The run is
// bit-reproducible on THIS machine; sin/cos/rsqrt implementation
// differences across GPU architectures can flip low bits, so — exactly
// like 08.01 — the STABLE output lines below carry NO measured numbers,
// only PASS/FAIL against thresholds with margins measured and stated
// honestly (THEORY.md §How we verify correctness). Measured numbers live
// on "[info]" lines, deliberately unchecked by demo/run_demo.*.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "GATE <name>:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]"
// unchecked. Change a stable line -> update demo/expected_output.txt in
// the same change.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Verification tolerances — MEASURED then margined (CLAUDE.md §12
// determinism note; see the [info] lines this program prints for the
// actual measured deviations each run). Documented here, all in one place,
// so a reader sees every "how tight is tight enough" decision at a glance.
// ---------------------------------------------------------------------------
static const float kSingleLoopEarlyTol   = 2e-4f;  // early-step (t < kSingleLoopEarlyN) feature abs tolerance
static const int   kSingleLoopEarlyN     = 20;      // steps considered "early" (before compounding grows)
static const float kSingleLoopLateTol    = 5e-3f;  // late-step feature abs tolerance (honest accumulation drift)
static const float kSingleLoopFinalErrRelTol = 0.05f; // final feature-error-norm relative tolerance

// Measured worst deviations over 16 samples x 3 variants (near-180-degree
// retreat-style samples included, where a small Z can inflate A's entries
// into the tens-to-hundreds and a few-ULP FMA-fusion difference between
// nvcc's device code and cl.exe's host code compounds through 4 points'
// worth of accumulation): |dv| ~1.2e-5, |dA| ~6.8e-3, |db| ~3.4e-4. The
// floors below keep ~3x headroom over the measured worst case for A/b
// (looser than v because A/b's absolute magnitude is itself much larger
// near-singular) while still catching an indexing/sign/layout bug, which
// would blow past these by orders of magnitude, not fractions of a percent.
static const float kStepTwinTolV = 1e-3f;   // Jacobian/pinv twin: |v| abs tolerance (v ~ O(0.01-1))
static const float kStepTwinTolA = 2e-2f;   // ...: |A| abs tolerance (A entries ~ O(1-100) near-singular)
static const float kStepTwinTolB = 2e-3f;   // ...: |b| abs tolerance

static const float kBatchStepsTol      = 5.0f;    // batch-stats twin: allowed |steps_gpu-steps_cpu| when both converged
static const float kBatchFinalErrRelTol = 0.02f;  // batch-stats twin: relative tolerance on final_err
static const float kBatchConvergedAgreeFloor = 0.95f; // batch-stats twin: fraction of the 128 that must AGREE on converged flag

// Exponential-decay gate: relative tolerance on the fitted rate vs kLambda.
static const float kDecayRateRelTol = 0.35f;
static const int   kDecayFitWindow  = 60;    // steps considered for the log-linear fit (capped by available rows)

// Convergence-basin gate floors (nominal cohort, percent). Measured at the
// documented nominal region (kNominalPosRange, kNominalAngleMaxDeg):
// true-depth ~98%, fixed-depth ~96-97% — the fixed-depth approximation's
// cost is genuinely SMALL in this modest region (its depth ratio never
// exceeds ~1.4x, well inside the classical local-stability margin;
// THEORY.md discusses where the gap widens). The floors below keep
// meaningful headroom under the measured values without pretending the
// two variants are dramatically different HERE — the retreat_pathology
// gate is where the depth-independent, geometry-driven IBVS failure mode
// actually shows up.
static const float kBasinTrueDepthFloor  = 90.0f;
static const float kBasinFixedDepthFloor = 85.0f;

// Retreat-pathology gate floor (percent of the retreat cohort).
static const float kRetreatDetectFloor = 80.0f;

static const uint32_t kBaseSeed = 42u;

// ---------------------------------------------------------------------------
// Scenario — the committed "task definition": batch size and the basin
// grid resolution. Everything else (target geometry, controller constants,
// cohort ranges) is the taught, tuned setup living as compile-time
// constants in kernels.cuh (the same split 08.01 uses for its scenario).
// Rows: "K,n" and "BASIN_G,g".
// ---------------------------------------------------------------------------
struct Scenario {
    int K = kDefaultK;
    int basinG = kDefaultBasinG;
    bool loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;
    bool haveK = false, haveG = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "K") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short K row\n"); return Scenario{}; }
            sc.K = std::atoi(cell.c_str()); haveK = true;
        } else if (label == "BASIN_G") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short BASIN_G row\n"); return Scenario{}; }
            sc.basinG = std::atoi(cell.c_str()); haveG = true;
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!haveK || !haveG || sc.K < 8 || sc.basinG < 2) {
        std::fprintf(stderr, "scenario: missing/invalid K or BASIN_G\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

// ---------------------------------------------------------------------------
// Small host-side statistics helpers (plain, single-purpose — this is
// report-generation code, not the algorithm under test).
// ---------------------------------------------------------------------------
static float mean_of(const float* v, int lo, int hi)   // mean over [lo,hi)
{
    if (hi <= lo) return 0.0f;
    double s = 0.0;
    for (int i = lo; i < hi; ++i) s += v[i];
    return static_cast<float>(s / (hi - lo));
}

static float median_of_converged_steps(const float* steps, const float* converged, int lo, int hi)
{
    std::vector<float> v;
    for (int i = lo; i < hi; ++i) if (converged[i] > 0.5f) v.push_back(steps[i]);
    if (v.empty()) return -1.0f;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// pearson_correlation — textbook sample correlation coefficient. Used ONLY
// for the [info] "conditioning_honesty" line (never gated — a correlation
// is reported, not passed/failed, because THEORY.md explains this cohort's
// failure mode is geometric, not a conditioning artifact).
static float pearson_correlation(const std::vector<float>& x, const std::vector<float>& y)
{
    const size_t n = x.size();
    if (n < 2) return 0.0f;
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    const double denom = std::sqrt(sxx * syy);
    return denom > 1e-12 ? static_cast<float>(sxy / denom) : 0.0f;
}

// log_linear_fit_rate — least-squares slope of ln(y) vs x, i.e. fit
// y = y0 * exp(rate * x) and return -rate (so a DECAYING y gives a
// POSITIVE returned rate, matching how kLambda is stated). Skips any
// non-positive y (should not occur for a genuine decay curve; defensive).
static float log_linear_fit_rate(const std::vector<float>& xs, const std::vector<float>& ys)
{
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (ys[i] <= 0.0f) continue;
        const double x = xs[i], y = std::log(static_cast<double>(ys[i]));
        sx += x; sy += y; sxx += x * x; sxy += x * y; ++n;
    }
    if (n < 2) return 0.0f;
    const double denom = n * sxx - sx * sx;
    if (std::fabs(denom) < 1e-12) return 0.0f;
    const double slope = (n * sxy - sx * sy) / denom;
    return static_cast<float>(-slope);
}

// ---------------------------------------------------------------------------
// PPM writer (binary P6 — trivial to hand-roll, no library needed for
// something this simple; CLAUDE.md §5 dependency policy: hand-roll before
// reaching for a vendored decoder/encoder). RGB, 8-bit, row-major top-down.
// ---------------------------------------------------------------------------
static bool write_ppm(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P6\n" << w << ' ' << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return f.good();
}

static void set_px(std::vector<unsigned char>& rgb, int w, int h, int x, int y,
                   unsigned char r, unsigned char g, unsigned char b)
{
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    const size_t idx = (static_cast<size_t>(y) * w + x) * 3;
    rgb[idx + 0] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
}

// draw_line — integer Bresenham, the standard hand-rolled line rasterizer
// (no floating point, no library) — plenty for a teaching artifact at a
// few hundred pixels across.
static void draw_line(std::vector<unsigned char>& rgb, int w, int h, int x0, int y0, int x1, int y1,
                      unsigned char r, unsigned char g, unsigned char b)
{
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        set_px(rgb, w, h, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void fill_square(std::vector<unsigned char>& rgb, int w, int h, int cx, int cy, int r,
                        unsigned char cr, unsigned char cg, unsigned char cb)
{
    for (int y = cy - r; y <= cy + r; ++y)
        for (int x = cx - r; x <= cx + r; ++x)
            set_px(rgb, w, h, x, y, cr, cg, cb);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int K = kDefaultK;
    int basinG = kDefaultBasinG;
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--loops")   && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--basin-g")  && i + 1 < argc) basinG = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")     && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--loops K] [--basin-g G] [--data ibvs_scenario.csv]\n"
                "note: non-default K/G changes the PROBLEM/SCENARIO lines; the demo diff will flag it.\n",
                argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Visual servoing: image-Jacobian IBVS control loop entirely on GPU (project 01.12)\n");
    print_device_info();

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_data_file(data_path, argv[0], "ibvs_scenario.csv");
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND - data/sample/ibvs_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED - see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    K = sc.K; basinG = sc.basinG;

    const int n_decay   = static_cast<int>(std::lround(K * kFracDecay));
    const int n_retreat = static_cast<int>(std::lround(K * kFracRetreat));
    const int n_nominal = K - n_decay - n_retreat;
    if (n_nominal < 1 || n_decay < 1 || n_retreat < 1) {
        std::printf("SCENARIO: MALFORMED - K=%d too small for three nonempty cohorts\n", K);
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    std::printf("PROBLEM: batched IBVS convergence-basin study, K=%d loops x T<=%d steps @ dt=%.2f s, "
               "4-point coplanar target, %d controller variants, FP32\n",
               K, kMaxSteps, static_cast<double>(kDt), kVariantCount);
    std::printf("SCENARIO: eye-in-hand IBVS, goal standoff %.2f m, target half-size %.2f m, "
               "cohorts nominal=%d/decay=%d/retreat=%d, basin grid %dx%d [synthetic]\n",
               static_cast<double>(kGoalStandoff), static_cast<double>(kTargetHalfSize),
               n_nominal, n_decay, n_retreat, basinG, basinG);

    // ---- target/goal ----------------------------------------------------------
    float target_pts[12], s_star[8];
    build_target_and_goal_cpu(target_pts, s_star);
    set_target_and_goal(target_pts, s_star);

    // ---- initial poses (host, shared by GPU and CPU paths) --------------------
    std::vector<float> init_poses(static_cast<size_t>(K) * 7);
    generate_batch_init_poses_cpu(K, n_nominal, n_decay, kBaseSeed, init_poses.data());

    // Trace loop indices: 2 nominal, 4 decay, 2 retreat (kernels.cuh
    // kTraceCount=8) — the "small documented subset" for artifacts AND the
    // exponential_decay gate's fit cohort (the 4 decay-slot traces).
    int trace_idx[kTraceCount] = {
        0, n_nominal / 2,
        n_nominal, n_nominal + n_decay / 4, n_nominal + n_decay / 2, n_nominal + 3 * n_decay / 4,
        n_nominal + n_decay, K - 1
    };
    const int kDecaySlotFirst = 2, kDecaySlotCount = 4;   // where the decay traces sit in trace_idx[]

    // ---- device buffers (allocated once, reused per variant) ------------------
    float *d_init_poses = nullptr, *d_out_trace = nullptr;
    int   *d_trace_idx = nullptr;
    float *d_converged = nullptr, *d_steps = nullptr, *d_final_err = nullptr;
    float *d_cond_min = nullptr, *d_zmax = nullptr, *d_featmax = nullptr;
    const size_t traceFloats = static_cast<size_t>(kTraceCount) * (kMaxSteps + 1) * kTraceRowStride;

    CUDA_CHECK(cudaMalloc(&d_init_poses, init_poses.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_init_poses, init_poses.data(), init_poses.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_trace_idx, kTraceCount * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_trace_idx, trace_idx, kTraceCount * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_out_trace, traceFloats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_converged, static_cast<size_t>(K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_steps,      static_cast<size_t>(K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_err,  static_cast<size_t>(K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cond_min,   static_cast<size_t>(K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zmax,       static_cast<size_t>(K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_featmax,    static_cast<size_t>(K) * sizeof(float)));

    // Per-variant host results (kernels.cuh "OUTPUT LAYOUT").
    struct BatchResult {
        std::vector<float> converged, steps, final_err, cond_min, zmax, featmax;
        void resize(int K) {
            converged.resize(K); steps.resize(K); final_err.resize(K);
            cond_min.resize(K); zmax.resize(K); featmax.resize(K);
        }
    };
    BatchResult gpuRes[kVariantCount];
    std::vector<float> gpu_trace(traceFloats, 0.0f);   // only populated for TRUE_DEPTH (see below)

    double gpu_ms_total = 0.0;
    for (int variant = 0; variant < kVariantCount; ++variant) {
        gpuRes[variant].resize(K);
        const bool wantTrace = (variant == kVariantTrueDepth);
        if (wantTrace) CUDA_CHECK(cudaMemset(d_out_trace, 0, traceFloats * sizeof(float)));

        GpuTimer gt; gt.begin();
        launch_ibvs_batch(K, variant, d_init_poses,
                          wantTrace ? d_trace_idx : nullptr, wantTrace ? kTraceCount : 0,
                          d_converged, d_steps, d_final_err, d_cond_min, d_zmax, d_featmax, d_out_trace);
        gpu_ms_total += static_cast<double>(gt.end_ms());

        CUDA_CHECK(cudaMemcpy(gpuRes[variant].converged.data(), d_converged, static_cast<size_t>(K)*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpuRes[variant].steps.data(),     d_steps,     static_cast<size_t>(K)*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpuRes[variant].final_err.data(), d_final_err, static_cast<size_t>(K)*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpuRes[variant].cond_min.data(),  d_cond_min,  static_cast<size_t>(K)*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpuRes[variant].zmax.data(),      d_zmax,      static_cast<size_t>(K)*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpuRes[variant].featmax.data(),   d_featmax,   static_cast<size_t>(K)*sizeof(float), cudaMemcpyDeviceToHost));
        if (wantTrace)
            CUDA_CHECK(cudaMemcpy(gpu_trace.data(), d_out_trace, traceFloats*sizeof(float), cudaMemcpyDeviceToHost));
    }
    std::printf("[time] batch kernel: %.2f ms total across %d variants (K=%d loops, up to %d steps each)\n",
               gpu_ms_total, kVariantCount, K, kMaxSteps);

    // ---- basin-map grid run (true-depth only) ----------------------------------
    const int Kb = basinG * basinG;
    std::vector<float> basin_poses(static_cast<size_t>(Kb) * 7);
    generate_basin_grid_poses_cpu(basinG, basin_poses.data());
    float *d_basin_poses = nullptr, *d_bc = nullptr, *d_bs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_basin_poses, basin_poses.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_basin_poses, basin_poses.data(), basin_poses.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_bc, static_cast<size_t>(Kb) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bs, static_cast<size_t>(Kb) * sizeof(float)));
    float *d_dummy1 = nullptr, *d_dummy2 = nullptr, *d_dummy3 = nullptr, *d_dummy4 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dummy1, static_cast<size_t>(Kb) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dummy2, static_cast<size_t>(Kb) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dummy3, static_cast<size_t>(Kb) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dummy4, static_cast<size_t>(Kb) * sizeof(float)));
    launch_ibvs_batch(Kb, kVariantTrueDepth, d_basin_poses, nullptr, 0,
                      d_bc, d_bs, d_dummy1, d_dummy2, d_dummy3, d_dummy4, nullptr);
    std::vector<float> basin_converged(Kb), basin_steps(Kb);
    CUDA_CHECK(cudaMemcpy(basin_converged.data(), d_bc, static_cast<size_t>(Kb)*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(basin_steps.data(),      d_bs, static_cast<size_t>(Kb)*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_basin_poses)); CUDA_CHECK(cudaFree(d_bc)); CUDA_CHECK(cudaFree(d_bs));
    CUDA_CHECK(cudaFree(d_dummy1)); CUDA_CHECK(cudaFree(d_dummy2)); CUDA_CHECK(cudaFree(d_dummy3)); CUDA_CHECK(cudaFree(d_dummy4));

    bool all_verify_pass = true;

    // =========================================================================
    // VERIFY A — single-loop trajectory twin (loop index 0, TRUE_DEPTH).
    // Reuses the GPU batch's trace slot 0 (trace_idx[0] == 0) against a
    // freshly-run CPU oracle on the SAME single loop. Early steps compared
    // tightly; later steps loosely (honest compounding — see the tolerance
    // constants' header comment).
    // =========================================================================
    {
        std::vector<float> cpu_trace(static_cast<size_t>(kMaxSteps + 1) * kTraceRowStride, 0.0f);
        std::vector<float> cpu_conv(1), cpu_steps(1), cpu_err(1), cpu_cm(1), cpu_zm(1), cpu_fm(1);
        int one_idx[1] = { 0 };
        ibvs_batch_cpu(1, kVariantTrueDepth, &init_poses[0], target_pts, s_star, one_idx, 1,
                      cpu_conv.data(), cpu_steps.data(), cpu_err.data(), cpu_cm.data(), cpu_zm.data(), cpu_fm.data(),
                      cpu_trace.data());

        const bool gpu_conv = gpuRes[kVariantTrueDepth].converged[0] > 0.5f;
        const bool cpu_conv_b = cpu_conv[0] > 0.5f;
        const int gpuValidRows = gpu_conv ? static_cast<int>(gpuRes[kVariantTrueDepth].steps[0]) + 1
                                          : static_cast<int>(gpuRes[kVariantTrueDepth].steps[0]);
        const int cpuValidRows = cpu_conv_b ? static_cast<int>(cpu_steps[0]) + 1 : static_cast<int>(cpu_steps[0]);
        const int cmpRows = std::min(gpuValidRows, cpuValidRows);

        float worstEarly = 0.0f, worstLate = 0.0f;
        for (int t = 0; t < cmpRows; ++t) {
            const float* gr = &gpu_trace[static_cast<size_t>(t) * kTraceRowStride];   // slot 0 == loop 0
            const float* cr = &cpu_trace[static_cast<size_t>(t) * kTraceRowStride];
            float worstRow = 0.0f;
            for (int f = 1; f < kTraceRowStride; ++f) worstRow = std::max(worstRow, std::fabs(gr[f] - cr[f]));
            if (t < kSingleLoopEarlyN) worstEarly = std::max(worstEarly, worstRow);
            else worstLate = std::max(worstLate, worstRow);
        }
        const float gpuFinalErr = gpuRes[kVariantTrueDepth].final_err[0];
        const float cpuFinalErr = cpu_err[0];
        const float finalErrRelDev = std::fabs(gpuFinalErr - cpuFinalErr) / std::max(std::fabs(cpuFinalErr), 1e-6f);

        const bool pass = (gpu_conv == cpu_conv_b) && worstEarly <= kSingleLoopEarlyTol
                        && worstLate <= kSingleLoopLateTol && finalErrRelDev <= kSingleLoopFinalErrRelTol;
        all_verify_pass = all_verify_pass && pass;
        std::printf("[info] single-loop twin: gpu_converged=%d cpu_converged=%d worst_early=%.3e (tol %.1e) "
                   "worst_late=%.3e (tol %.1e) final_err_rel_dev=%.3e (tol %.2f)\n",
                   gpu_conv ? 1 : 0, cpu_conv_b ? 1 : 0, static_cast<double>(worstEarly), static_cast<double>(kSingleLoopEarlyTol),
                   static_cast<double>(worstLate), static_cast<double>(kSingleLoopLateTol),
                   static_cast<double>(finalErrRelDev), static_cast<double>(kSingleLoopFinalErrRelTol));
        std::printf("VERIFY: single-loop trajectory twin %s (GPU matches CPU oracle within documented tolerance)\n",
                   pass ? "PASS" : "FAIL");
    }

    // =========================================================================
    // VERIFY B — Jacobian/pseudoinverse twin at a sampled state set (tight).
    // Samples span: the goal itself, small/large nominal-like offsets, and a
    // near-180-degree retreat-style rotation — the same variety of regimes
    // the batch actually visits, checked at the finest useful grain.
    // =========================================================================
    {
        const int kSamples = 16;
        std::vector<float> samplePoses(static_cast<size_t>(kSamples) * 7);
        {
            uint32_t s = 777u;
            auto uf = [&s]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >> 8) * (1.0f/16777216.0f); };
            for (int i = 0; i < kSamples; ++i) {
                float* p = &samplePoses[static_cast<size_t>(i) * 7];
                if (i == 0) { // exactly the goal pose
                    p[0]=0; p[1]=0; p[2]=-kGoalStandoff; p[3]=1; p[4]=0; p[5]=0; p[6]=0; continue;
                }
                const float range = (i < kSamples/2) ? 0.10f : 0.30f;   // small then large offsets
                p[0] = (2.0f*uf()-1.0f)*range; p[1] = (2.0f*uf()-1.0f)*range; p[2] = -kGoalStandoff + (2.0f*uf()-1.0f)*range;
                float ax=2.0f*uf()-1.0f, ay=2.0f*uf()-1.0f, az=2.0f*uf()-1.0f;
                float angle = (i >= kSamples-3) ? (2.8f + 0.3f*uf()) : (uf()*0.6f);   // last 3: near-180 deg retreat-style
                float n = std::sqrt(ax*ax+ay*ay+az*az); if (n < 1e-6f) n = 1.0f;
                const float half = 0.5f*angle, sn = std::sin(half);
                p[3] = std::cos(half); p[4]=ax/n*sn; p[5]=ay/n*sn; p[6]=az/n*sn;
            }
        }
        float *d_sp = nullptr, *d_v = nullptr, *d_A = nullptr, *d_b = nullptr, *d_e = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sp, samplePoses.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_sp, samplePoses.data(), samplePoses.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(kSamples)*6*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(kSamples)*36*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(kSamples)*6*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_e, static_cast<size_t>(kSamples)*8*sizeof(float)));

        float worst_v = 0.0f, worst_A = 0.0f, worst_b = 0.0f, worst_e = 0.0f;
        for (int variant = 0; variant < kVariantCount; ++variant) {
            launch_ibvs_single_step(kSamples, d_sp, variant, d_v, d_A, d_b, d_e);
            std::vector<float> gv(static_cast<size_t>(kSamples)*6), gA(static_cast<size_t>(kSamples)*36),
                               gb(static_cast<size_t>(kSamples)*6), ge(static_cast<size_t>(kSamples)*8);
            CUDA_CHECK(cudaMemcpy(gv.data(), d_v, gv.size()*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gA.data(), d_A, gA.size()*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gb.data(), d_b, gb.size()*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ge.data(), d_e, ge.size()*sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < kSamples; ++i) {
                float cv[6], cA[36], cb[6], cf[8], cerr, ccp, czm, cfm;
                ibvs_compute_step_cpu(&samplePoses[static_cast<size_t>(i)*7], variant, target_pts, s_star,
                                      cv, cA, cb, cf, &cerr, &ccp, &czm, &cfm);
                float ce[8]; for (int f = 0; f < 8; ++f) ce[f] = cf[f] - s_star[f];
                for (int f = 0; f < 6; ++f)  worst_v = std::max(worst_v, std::fabs(gv[static_cast<size_t>(i)*6+f] - cv[f]));
                for (int f = 0; f < 36; ++f) worst_A = std::max(worst_A, std::fabs(gA[static_cast<size_t>(i)*36+f] - cA[f]));
                for (int f = 0; f < 6; ++f)  worst_b = std::max(worst_b, std::fabs(gb[static_cast<size_t>(i)*6+f] - cb[f]));
                for (int f = 0; f < 8; ++f)  worst_e = std::max(worst_e, std::fabs(ge[static_cast<size_t>(i)*8+f] - ce[f]));
            }
        }
        CUDA_CHECK(cudaFree(d_sp)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_e));

        const bool pass = worst_v <= kStepTwinTolV && worst_A <= kStepTwinTolA && worst_b <= kStepTwinTolB;
        all_verify_pass = all_verify_pass && pass;
        std::printf("[info] jacobian/pinv twin: worst|dv|=%.3e worst|dA|=%.3e worst|db|=%.3e worst|de|=%.3e "
                   "(tol v=%.1e A=%.1e b=%.1e) over %d samples x %d variants\n",
                   static_cast<double>(worst_v), static_cast<double>(worst_A), static_cast<double>(worst_b),
                   static_cast<double>(worst_e), static_cast<double>(kStepTwinTolV), static_cast<double>(kStepTwinTolA),
                   static_cast<double>(kStepTwinTolB), kSamples, kVariantCount);
        std::printf("VERIFY: jacobian/pseudoinverse twin %s (GPU matches CPU oracle within tight tolerance)\n",
                   pass ? "PASS" : "FAIL");
    }

    // =========================================================================
    // VERIFY C — batch statistics twin: a 128-loop subset re-simulated on
    // the CPU, compared against the already-computed GPU batch (TRUE_DEPTH).
    // =========================================================================
    {
        const int kSub = std::min(128, K);
        std::vector<float> c_conv(kSub), c_steps(kSub), c_err(kSub), c_cm(kSub), c_zm(kSub), c_fm(kSub);
        ibvs_batch_cpu(kSub, kVariantTrueDepth, init_poses.data(), target_pts, s_star, nullptr, 0,
                      c_conv.data(), c_steps.data(), c_err.data(), c_cm.data(), c_zm.data(), c_fm.data(), nullptr);

        int agree = 0, stepsChecked = 0, stepsOk = 0;
        float worstErrRel = 0.0f;
        for (int i = 0; i < kSub; ++i) {
            const bool g = gpuRes[kVariantTrueDepth].converged[i] > 0.5f;
            const bool c = c_conv[i] > 0.5f;
            if (g == c) ++agree;
            if (g && c) {
                ++stepsChecked;
                if (std::fabs(gpuRes[kVariantTrueDepth].steps[i] - c_steps[i]) <= kBatchStepsTol) ++stepsOk;
            }
            const float rel = std::fabs(gpuRes[kVariantTrueDepth].final_err[i] - c_err[i])
                            / std::max(std::fabs(c_err[i]), 1e-6f);
            worstErrRel = std::max(worstErrRel, rel);
        }
        const float agreeFrac = static_cast<float>(agree) / static_cast<float>(kSub);
        const bool stepsPass = (stepsChecked == 0) || (static_cast<float>(stepsOk) / stepsChecked >= 0.95f);
        const bool pass = agreeFrac >= kBatchConvergedAgreeFloor && stepsPass && worstErrRel <= kBatchFinalErrRelTol;
        all_verify_pass = all_verify_pass && pass;
        std::printf("[info] batch-stats twin (%d loops): converged-flag agreement=%.1f%% (floor %.0f%%), "
                   "steps-agreement=%d/%d, worst final_err rel dev=%.3e (tol %.2f)\n",
                   kSub, static_cast<double>(agreeFrac*100.0f), static_cast<double>(kBatchConvergedAgreeFloor*100.0f),
                   stepsOk, stepsChecked, static_cast<double>(worstErrRel), static_cast<double>(kBatchFinalErrRelTol));
        std::printf("VERIFY: batch statistics twin %s (128-loop subset matches CPU oracle within documented tolerance)\n",
                   pass ? "PASS" : "FAIL");
    }

    if (!all_verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement in a verify stage — fix before trusting the gates below)\n");
        return 1;
    }

    // =========================================================================
    // GATE exponential_decay — the analytic control-theory prediction: for
    // SMALL, PURE-TRANSLATION initial error under TRUE-DEPTH IBVS, the
    // feature-error norm should decay close to exp(-lambda*t) (THEORY.md
    // derives the local linearization ė ~ -lambda*e and its honest caveat
    // for an 8-feature/6-DOF redundant system). This gate is INDEPENDENT of
    // both twins — it checks the CONTROLLER against a closed-form physics
    // prediction, not against the other implementation (reference_cpu.cpp's
    // ruling).
    // =========================================================================
    bool gate_decay_pass = false;
    float decay_fit_rate = 0.0f;
    {
        // Average error-norm across the kDecaySlotCount decay-cohort traces,
        // over the steps ALL of them actually ran (never read a memset-zero
        // row past a loop's own convergence — see kernels.cuh "TRACE LAYOUT").
        int minValidRows = kMaxSteps + 1;
        for (int s = 0; s < kDecaySlotCount; ++s) {
            const int idx = trace_idx[kDecaySlotFirst + s];
            const bool conv = gpuRes[kVariantTrueDepth].converged[idx] > 0.5f;
            const int steps = static_cast<int>(gpuRes[kVariantTrueDepth].steps[idx]);
            const int validRows = conv ? steps + 1 : steps;
            minValidRows = std::min(minValidRows, validRows);
        }
        const int W = std::min(kDecayFitWindow, minValidRows);
        std::vector<float> ts(W), meanErr(W);
        for (int t = 0; t < W; ++t) {
            double acc = 0.0;
            for (int s = 0; s < kDecaySlotCount; ++s) {
                const float* row = &gpu_trace[static_cast<size_t>(kDecaySlotFirst + s) * (kMaxSteps+1) * kTraceRowStride
                                              + static_cast<size_t>(t) * kTraceRowStride];
                double sq = 0.0;
                for (int f = 0; f < 8; ++f) { const double d = row[4+f] - s_star[f]; sq += d*d; }
                acc += std::sqrt(sq);
            }
            ts[t] = static_cast<float>(t) * kDt;
            meanErr[t] = static_cast<float>(acc / kDecaySlotCount);
        }
        decay_fit_rate = log_linear_fit_rate(ts, meanErr);
        const float relDev = std::fabs(decay_fit_rate - kLambda) / kLambda;
        gate_decay_pass = (W >= 8) && (relDev <= kDecayRateRelTol);

        // Artifact: error_decay.csv
        const std::string out_dir = resolve_out_dir(argv[0]);
        std::ofstream f(out_dir + "/error_decay.csv");
        if (f.is_open()) {
            f << "t_s,mean_err_decay_cohort,exp_fit_lambda=" << kLambda << "\n";
            for (int t = 0; t < W; ++t)
                f << ts[t] << ',' << meanErr[t] << ',' << (meanErr[0] * std::exp(-kLambda * ts[t])) << '\n';
            std::printf("[info] error_decay.csv: %d rows (fit window; may vary by a step or two across GPU architectures)\n", W);
            std::printf("ARTIFACT: wrote demo/out/error_decay.csv\n");
        } else {
            std::printf("ARTIFACT: FAILED to write demo/out/error_decay.csv\n");
        }

        std::printf("[info] exponential_decay: fitted rate=%.4f /s over %d-step window, target lambda=%.4f /s, "
                   "rel dev=%.3f (tol %.2f)\n",
                   static_cast<double>(decay_fit_rate), W, static_cast<double>(kLambda),
                   static_cast<double>(relDev), static_cast<double>(kDecayRateRelTol));
        std::printf("GATE exponential_decay: %s (decay-cohort fit rate within tolerance of lambda=%.2f /s, true-depth)\n",
                   gate_decay_pass ? "PASS" : "FAIL", static_cast<double>(kLambda));
    }

    // =========================================================================
    // GATE convergence_basin — within the NOMINAL cohort, true-depth
    // convergence must clear a high floor; fixed-depth is reported
    // alongside and gated at a LOWER floor (the robustness result made
    // quantitative — the classic claim that fixed-depth IBVS still mostly
    // works, just not as reliably).
    // =========================================================================
    bool gate_basin_pass = false;
    {
        const float trueDepthPct  = mean_of(gpuRes[kVariantTrueDepth].converged.data(), 0, n_nominal) * 100.0f;
        const float fixedDepthPct = mean_of(gpuRes[kVariantFixedDepth].converged.data(), 0, n_nominal) * 100.0f;
        const float desiredJacPct = mean_of(gpuRes[kVariantDesiredJacobian].converged.data(), 0, n_nominal) * 100.0f;
        gate_basin_pass = trueDepthPct >= kBasinTrueDepthFloor && fixedDepthPct >= kBasinFixedDepthFloor;

        const float basinPct = mean_of(basin_converged.data(), 0, Kb) * 100.0f;
        std::printf("[info] convergence_basin (nominal cohort, n=%d): true-depth=%.1f%% (floor %.0f%%), "
                   "fixed-depth=%.1f%% (floor %.0f%%), desired-jacobian=%.1f%%, translation-only basin grid=%.1f%%\n",
                   n_nominal, static_cast<double>(trueDepthPct), static_cast<double>(kBasinTrueDepthFloor),
                   static_cast<double>(fixedDepthPct), static_cast<double>(kBasinFixedDepthFloor),
                   static_cast<double>(desiredJacPct), static_cast<double>(basinPct));
        std::printf("GATE convergence_basin: %s (nominal cohort convergence meets floors for true-depth and fixed-depth)\n",
                   gate_basin_pass ? "PASS" : "FAIL");

        // [info] depth_robustness — median steps-to-converge per variant.
        const float mTrue = median_of_converged_steps(gpuRes[kVariantTrueDepth].steps.data(), gpuRes[kVariantTrueDepth].converged.data(), 0, n_nominal);
        const float mFixed = median_of_converged_steps(gpuRes[kVariantFixedDepth].steps.data(), gpuRes[kVariantFixedDepth].converged.data(), 0, n_nominal);
        const float mDesired = median_of_converged_steps(gpuRes[kVariantDesiredJacobian].steps.data(), gpuRes[kVariantDesiredJacobian].converged.data(), 0, n_nominal);
        std::printf("[info] depth_robustness (median steps-to-converge, nominal cohort): true-depth=%.0f "
                   "fixed-depth=%.0f desired-jacobian=%.0f\n",
                   static_cast<double>(mTrue), static_cast<double>(mFixed), static_cast<double>(mDesired));
    }

    // =========================================================================
    // GATE retreat_pathology — the RETREAT cohort (near-180-deg rotation
    // about the optical axis) MUST exhibit the classic IBVS camera-retreat
    // failure under TRUE-DEPTH IBVS: pinv(L) drives the shortest STRAIGHT
    // LINE in image space, and near a half-turn about the optical axis that
    // straight line is achieved by the camera physically backing away
    // (THEORY.md derives this geometrically) — a failure that is NOT a
    // depth-estimation error (true depth is used here) and NOT (primarily)
    // a conditioning problem (see the conditioning_honesty [info] below).
    // =========================================================================
    bool gate_retreat_pass = false;
    {
        int pathologyCount = 0;
        for (int i = n_nominal + n_decay; i < K; ++i) {
            const bool retreated = gpuRes[kVariantTrueDepth].zmax[i] > kRetreatZMultiple * kGoalStandoff;
            const bool failed = gpuRes[kVariantTrueDepth].converged[i] < 0.5f;
            if (retreated || failed) ++pathologyCount;
        }
        const float detectPct = 100.0f * static_cast<float>(pathologyCount) / static_cast<float>(n_retreat);
        gate_retreat_pass = detectPct >= kRetreatDetectFloor;
        std::printf("[info] retreat_pathology (n=%d, true-depth): detected=%.1f%% (floor %.0f%%; criterion: max depth "
                   "> %.1fx goal standoff OR non-convergence)\n",
                   n_retreat, static_cast<double>(detectPct), static_cast<double>(kRetreatDetectFloor),
                   static_cast<double>(kRetreatZMultiple));
        std::printf("GATE retreat_pathology: %s (near-180 deg cohort exhibits camera retreat at/above floor, true-depth)\n",
                   gate_retreat_pass ? "PASS" : "FAIL");

        // [info] conditioning_honesty — correlation between the worst
        // conditioning proxy encountered and failure, over the WHOLE batch.
        // Expected to be WEAK specifically for the retreat cohort (this is
        // a geometric pathology, not a numerical-singularity one) — a low
        // |r| here is the CORRECT, honest result, not evidence of a bug.
        std::vector<float> cm(gpuRes[kVariantTrueDepth].cond_min.begin(), gpuRes[kVariantTrueDepth].cond_min.end());
        std::vector<float> failFlag(K);
        for (int i = 0; i < K; ++i) failFlag[i] = (gpuRes[kVariantTrueDepth].converged[i] < 0.5f) ? 1.0f : 0.0f;
        const float rWhole = pearson_correlation(cm, failFlag);
        std::vector<float> cmR(cm.begin() + (n_nominal+n_decay), cm.end());
        std::vector<float> ffR(failFlag.begin() + (n_nominal+n_decay), failFlag.end());
        const float rRetreat = pearson_correlation(cmR, ffR);
        std::printf("[info] conditioning_honesty: corr(cond_min, failure) whole-batch r=%.3f, retreat-cohort-only r=%.3f "
                   "(a weak retreat-cohort r is EXPECTED — the pathology is geometric, not a conditioning artifact)\n",
                   static_cast<double>(rWhole), static_cast<double>(rRetreat));
    }

    // =========================================================================
    // ARTIFACT: image_plane_traces.ppm — the kTraceCount loops' feature
    // paths in normalized image coordinates.
    // =========================================================================
    {
        const int W = 480, H = 480;
        std::vector<unsigned char> img(static_cast<size_t>(W)*H*3, 255);   // white background
        const float halfSpan = 0.35f;   // normalized-coordinate range mapped to the canvas
        auto toPx = [&](float x, float y, int& px, int& py) {
            px = static_cast<int>((x + halfSpan) / (2*halfSpan) * (W-1));
            py = static_cast<int>((halfSpan - y) / (2*halfSpan) * (H-1));   // image y grows downward on screen too
        };
        // Axes (light gray) through the image center.
        int cx, cy; toPx(0,0,cx,cy);
        draw_line(img, W, H, 0, cy, W-1, cy, 220,220,220);
        draw_line(img, W, H, cx, 0, cx, H-1, 220,220,220);

        const unsigned char palette[kTraceCount][3] = {
            {200,0,0},{200,100,0},{0,0,200},{0,100,200},{150,0,150},{0,150,150},{100,60,20},{60,60,60}
        };
        for (int slot = 0; slot < kTraceCount; ++slot) {
            const int idx = trace_idx[slot];
            const bool conv = gpuRes[kVariantTrueDepth].converged[idx] > 0.5f;
            const int steps = static_cast<int>(gpuRes[kVariantTrueDepth].steps[idx]);
            const int rows = conv ? steps + 1 : steps;
            const unsigned char* col = palette[slot];
            for (int p = 0; p < 4; ++p) {   // one path per target point
                int prevX=0, prevY=0; bool have=false;
                for (int t = 0; t < rows; ++t) {
                    const float* row = &gpu_trace[static_cast<size_t>(slot)*(kMaxSteps+1)*kTraceRowStride + static_cast<size_t>(t)*kTraceRowStride];
                    int px,py; toPx(row[4+2*p], row[4+2*p+1], px, py);
                    if (have) draw_line(img, W, H, prevX, prevY, px, py, col[0], col[1], col[2]);
                    prevX=px; prevY=py; have=true;
                }
                if (have) fill_square(img, W, H, prevX, prevY, 2, col[0], col[1], col[2]);   // endpoint marker
            }
        }
        // Goal features s* — green squares, drawn LAST so they stay visible
        // even where a converged loop's endpoint lands exactly on the goal.
        for (int i = 0; i < 4; ++i) { int px,py; toPx(s_star[2*i], s_star[2*i+1], px, py); fill_square(img,W,H,px,py,4, 0,160,0); }
        const std::string out_dir = resolve_out_dir(argv[0]);
        if (write_ppm(out_dir + "/image_plane_traces.ppm", W, H, img))
            std::printf("ARTIFACT: wrote demo/out/image_plane_traces.ppm (%d loops traced)\n", kTraceCount);
        else
            std::printf("ARTIFACT: FAILED to write demo/out/image_plane_traces.ppm\n");
    }

    // =========================================================================
    // ARTIFACT: basin_map.ppm — the (dx,dy) grid colored by convergence
    // (green=converged, brightness falls with more steps-to-converge; red =
    // did not converge within budget). Axes documented in the pixel comment.
    // =========================================================================
    {
        const int cell = 6;                       // pixels per grid cell (upscale for visibility)
        const int W = basinG * cell, H = basinG * cell;
        std::vector<unsigned char> img(static_cast<size_t>(W)*H*3, 255);
        for (int iy = 0; iy < basinG; ++iy) {
            for (int ix = 0; ix < basinG; ++ix) {
                const int k = iy * basinG + ix;
                unsigned char r, g, b;
                if (basin_converged[k] > 0.5f) {
                    const float frac = std::min(1.0f, basin_steps[k] / static_cast<float>(kMaxSteps));
                    g = static_cast<unsigned char>(220 - 150*frac); r = static_cast<unsigned char>(40*frac); b = 40;
                } else {
                    r = 200; g = 30; b = 30;
                }
                for (int py = 0; py < cell; ++py)
                    for (int px = 0; px < cell; ++px)
                        set_px(img, W, H, ix*cell+px, (basinG-1-iy)*cell+py, r, g, b);   // iy flipped: +dy drawn upward
            }
        }
        const std::string out_dir = resolve_out_dir(argv[0]);
        if (write_ppm(out_dir + "/basin_map.ppm", W, H, img))
            std::printf("ARTIFACT: wrote demo/out/basin_map.ppm (%dx%d grid; axes: x=dx, y=dy in [-%.2f,+%.2f] m, dz=0, no rotation)\n",
                       basinG, basinG, static_cast<double>(kBasinPosRange), static_cast<double>(kBasinPosRange));
        else
            std::printf("ARTIFACT: FAILED to write demo/out/basin_map.ppm\n");
    }

    // =========================================================================
    // ARTIFACT: batch_stats.csv — per-variant, per-cohort summary.
    // =========================================================================
    {
        const std::string out_dir = resolve_out_dir(argv[0]);
        std::ofstream f(out_dir + "/batch_stats.csv");
        if (f.is_open()) {
            f << "variant,cohort,n,converged_pct,median_steps\n";
            struct Range { const char* name; int lo, hi; };
            const Range ranges[3] = { {"nominal", 0, n_nominal}, {"decay", n_nominal, n_nominal+n_decay}, {"retreat", n_nominal+n_decay, K} };
            for (int variant = 0; variant < kVariantCount; ++variant) {
                for (const auto& r : ranges) {
                    const float pct = mean_of(gpuRes[variant].converged.data(), r.lo, r.hi) * 100.0f;
                    const float med = median_of_converged_steps(gpuRes[variant].steps.data(), gpuRes[variant].converged.data(), r.lo, r.hi);
                    f << variant_name(variant) << ',' << r.name << ',' << (r.hi-r.lo) << ',' << pct << ',' << med << '\n';
                }
            }
            std::printf("ARTIFACT: wrote demo/out/batch_stats.csv (%d variants x 3 cohorts)\n", kVariantCount);
        } else {
            std::printf("ARTIFACT: FAILED to write demo/out/batch_stats.csv\n");
        }
    }

    // =========================================================================
    // ARTIFACT: gates_metrics.csv — flat provenance table of every gate.
    // =========================================================================
    {
        const std::string out_dir = resolve_out_dir(argv[0]);
        std::ofstream f(out_dir + "/gates_metrics.csv");
        if (f.is_open()) {
            f << "gate,measured,threshold,pass\n";
            f << "exponential_decay_rate," << decay_fit_rate << ',' << kLambda << ',' << (gate_decay_pass?1:0) << '\n';
            f << "convergence_basin_true_depth_pct," << mean_of(gpuRes[kVariantTrueDepth].converged.data(),0,n_nominal)*100.0f
              << ',' << kBasinTrueDepthFloor << ',' << (gate_basin_pass?1:0) << '\n';
            f << "convergence_basin_fixed_depth_pct," << mean_of(gpuRes[kVariantFixedDepth].converged.data(),0,n_nominal)*100.0f
              << ',' << kBasinFixedDepthFloor << ',' << (gate_basin_pass?1:0) << '\n';
            std::printf("ARTIFACT: wrote demo/out/gates_metrics.csv\n");
        } else {
            std::printf("ARTIFACT: FAILED to write demo/out/gates_metrics.csv\n");
        }
    }

    CUDA_CHECK(cudaFree(d_init_poses)); CUDA_CHECK(cudaFree(d_trace_idx)); CUDA_CHECK(cudaFree(d_out_trace));
    CUDA_CHECK(cudaFree(d_converged)); CUDA_CHECK(cudaFree(d_steps)); CUDA_CHECK(cudaFree(d_final_err));
    CUDA_CHECK(cudaFree(d_cond_min)); CUDA_CHECK(cudaFree(d_zmax)); CUDA_CHECK(cudaFree(d_featmax));

    const bool success = gate_decay_pass && gate_basin_pass && gate_retreat_pass;
    if (success)
        std::printf("RESULT: PASS (all verification and gates passed: exponential_decay, convergence_basin, retreat_pathology)\n");
    else
        std::printf("RESULT: FAIL (a gate failed - see the [info] lines above for measured values)\n");
    return success ? 0 : 1;
}
