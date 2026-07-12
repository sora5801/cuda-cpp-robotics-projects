// ===========================================================================
// main.cu - entry point for project 02.18
//           Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed sample (data/sample/points.csv): three independent
//      weather captures (SNOW/RAIN/DUST) of the SAME static scene, each with
//      a per-point real/scatterer ground-truth label (used only by gates).
//   2. VERIFY STAGE (the CLAUDE.md paragraph 5 GPU-vs-CPU gate), run on the
//      SNOW scan as representative data (kernels.cuh's file header explains
//      why: the kernels have no weather-specific branching, so one scan
//      certifies all three - the same reasoning project 02.13's file header
//      gives for not re-verifying per scan). THREE filters, each a
//      STATISTIC-then-CLASSIFY pair:
//        (a) SOR   - mean-KNN-distance (tight float tolerance) then classify
//                    against a host-computed threshold (exact, given that
//                    array - 02.13's ledger-then-classify precedent).
//        (b) DROR  - dynamic-radius neighbor count (exact integers) then
//                    classify (exact).
//        (c) LIOR  - fixed-radius neighbor count (exact integers) then
//                    classify (exact).
//   3. SECONDARY: run the now-verified GPU pipeline on ALL THREE scans to
//      produce the masks every gate below grades against - normal use of
//      already-certified kernels, not a second verification pass.
//   4. GATES (12 pass/fail + range-stratified/combined/intensity-dependence
//      [info]): DROR/LIOR precision+recall floors on snow+rain, real-point-
//      preservation floors, the SOR-vs-DROR far-range-failure contrast (both
//      directions asserted), and the dust-plume-core honesty check.
//   5. ARTIFACTS: a raw/DROR-cleaned/LIOR-cleaned top-view triptych of the
//      snow scan, a range-stratified false-removal CSV, and gates_metrics.csv
//      - all under demo/out/.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "VERIFY:", "GATE:", "ARTIFACT:", "RESULT:" - NEVER carrying a measured
// number, only fixed verdicts and fixed descriptive text (08.01's and
// 02.13's identical discipline). Every measured number lives on an
// "[info]"/"[time]" line, deliberately unchecked by demo/run_demo.*. Change
// a stable line -> update demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Gate thresholds - MEASURED-THEN-MARGINED against this project's committed
// sample (README "Expected output" states the actual numbers from the run
// that produced demo/expected_output.txt; THEORY.md "How we verify
// correctness" explains the margining philosophy) - the same discipline
// project 02.13's main.cu documents for its own gate constants.
// ---------------------------------------------------------------------------
static constexpr double kGateDrorRecallSnowFloorPct    = 85.0;
static constexpr double kGateDrorRecallRainFloorPct    = 85.0;
static constexpr double kGateDrorPrecisionSnowFloorPct = 85.0;
static constexpr double kGateDrorPrecisionRainFloorPct = 85.0;
static constexpr double kGateLiorRecallSnowFloorPct    = 75.0;
static constexpr double kGateLiorRecallRainFloorPct    = 75.0;
static constexpr double kGateLiorPrecisionSnowFloorPct = 50.0;
static constexpr double kGateLiorPrecisionRainFloorPct = 60.0;
static constexpr double kGateRealPreserveDrorFloorPct  = 85.0;
static constexpr double kGateRealPreserveLiorFloorPct  = 80.0;
static constexpr double kGateSorFarFalseFloorPct       = 35.0;   // SOR must fail AT LEAST this badly
static constexpr double kGateDrorFarFalseCeilingPct    = 15.0;   // DROR must fail AT MOST this badly

// Dust plume box - MUST MATCH ../scripts/make_synthetic.py's DUST_PLUME_BOX
// (used only by the dust_plume_honesty gate/analysis below, never by the
// filtering kernels themselves - ground truth, kernels.cuh's file header
// discipline).
static constexpr float kPlumeBoxMin[3] = { 3.0f, -6.0f, -1.2f };
static constexpr float kPlumeBoxMax[3] = { 7.0f, 6.0f, 1.5f };

// Intensity-dependence perturbation exponent (main.cu's own analysis, not
// baked into the committed data - see the gate's implementation below for
// the physical story: an uncalibrated per-channel gain that has not fully
// divided out range-dependent falloff, project 02.20's territory).
static constexpr float kIntensityPerturbExponent = 0.6f;
static constexpr float kIntensityPerturbRefRangeM = 10.0f;

// ===========================================================================
// Data loading - points.csv, the committed sample scripts/make_synthetic.py
// writes (that file's module docstring is this format's specification).
// ===========================================================================
struct PointSet {
    std::vector<float> xyz;              // [n*3] interleaved, sensor frame, meters
    std::vector<float> intensity;        // [n], unitless [0,1]
    std::vector<int32_t> is_real;        // [n] GROUND TRUTH: 1 real, 0 scatterer
    std::vector<int32_t> scatterer_type; // [n] GROUND TRUTH: -1/0/1/2 (n/a/snow/rain/dust)
    std::vector<int32_t> surf_cohort;    // [n] GROUND TRUTH: -1/0/1/2/3
    int n = 0;
};

static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) fields.push_back(cell);
    return fields;
}

// load_points - parse points.csv's '#'-prefixed header (asserted against
// kernels.cuh's beam-model constants, the 02.08/02.13-style data/code
// consistency check) and its data rows, splitting them into the three
// per-weather PointSet buckets by the leading 'weather' column.
static bool load_points(const std::string& path, PointSet out[kNumWeatherScans])
{
    std::ifstream in(path);
    if (!in.is_open()) return false;

    int hdr_num_beams = -1, hdr_azimuth_steps = -1;
    float hdr_max_range = -1.0f;
    std::string line;
    long row_count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(1, eq - 1);
            const size_t start = key.find_first_not_of(" \t");
            key = (start == std::string::npos) ? "" : key.substr(start);
            const std::string val = line.substr(eq + 1);
            if (key == "num_beams")          hdr_num_beams = std::atoi(val.c_str());
            else if (key == "azimuth_steps") hdr_azimuth_steps = std::atoi(val.c_str());
            else if (key == "max_range_m")   hdr_max_range = std::strtof(val.c_str(), nullptr);
            continue;
        }
        const auto f = split_csv(line);
        if (f.size() != 8) {
            std::fprintf(stderr, "points.csv: malformed row (expected 8 fields, got %zu): %s\n",
                         f.size(), line.c_str());
            return false;
        }
        const int weather = std::atoi(f[0].c_str());
        if (weather < 0 || weather >= kNumWeatherScans) {
            std::fprintf(stderr, "points.csv: weather id %d out of range\n", weather);
            return false;
        }
        PointSet& ps = out[weather];
        ps.xyz.push_back(std::strtof(f[1].c_str(), nullptr));
        ps.xyz.push_back(std::strtof(f[2].c_str(), nullptr));
        ps.xyz.push_back(std::strtof(f[3].c_str(), nullptr));
        ps.intensity.push_back(std::strtof(f[4].c_str(), nullptr));
        ps.is_real.push_back(std::atoi(f[5].c_str()));
        ps.scatterer_type.push_back(std::atoi(f[6].c_str()));
        ps.surf_cohort.push_back(std::atoi(f[7].c_str()));
        ++row_count;
    }
    for (int w = 0; w < kNumWeatherScans; ++w) out[w].n = static_cast<int>(out[w].is_real.size());

    if (hdr_num_beams != kNumBeams || hdr_azimuth_steps != kAzimuthSteps ||
        std::fabs(hdr_max_range - kMaxRangeM) > 1e-3f) {
        std::fprintf(stderr,
            "points.csv header mismatch: file has num_beams=%d azimuth_steps=%d max_range_m=%.3f; "
            "kernels.cuh expects %d/%d/%.3f - regenerate the sample (scripts/make_synthetic.py) "
            "or update kernels.cuh\n",
            hdr_num_beams, hdr_azimuth_steps, static_cast<double>(hdr_max_range),
            kNumBeams, kAzimuthSteps, static_cast<double>(kMaxRangeM));
        return false;
    }
    return row_count > 0;
}

// ===========================================================================
// PrecisionRecall - the shared confusion-matrix summary every gate below is
// built from. "Outlier" (mask==1) is the POSITIVE class throughout, matching
// each filter's own job description ("flag weather noise for removal").
// ===========================================================================
struct PrecisionRecall {
    long long tp = 0, fp = 0, fn = 0, tn = 0;
    double precision_pct() const { return (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0.0; }
    double recall_pct()    const { return (tp + fn) > 0 ? 100.0 * tp / (tp + fn) : 0.0; }
    double real_preserve_pct() const { return (tn + fp) > 0 ? 100.0 * tn / (tn + fp) : 0.0; }
};

// pr_over - confusion matrix of `mask` (1=removed) against is_real ground
// truth, restricted to the index set `idx` (pass nullptr for "every point").
static PrecisionRecall pr_over(const std::vector<int32_t>& mask, const PointSet& ps,
                               const std::vector<int>* idx = nullptr)
{
    PrecisionRecall r;
    const int n = idx ? static_cast<int>(idx->size()) : ps.n;
    for (int k = 0; k < n; ++k) {
        const int i = idx ? (*idx)[k] : k;
        const bool removed = mask[i] != 0;
        const bool real = ps.is_real[i] != 0;
        if (removed && !real) ++r.tp;
        else if (removed && real) ++r.fp;
        else if (!removed && !real) ++r.fn;
        else ++r.tn;
    }
    return r;
}

// ===========================================================================
// GateResult - one row of the pass/fail summary (also written to
// demo/out/gates_metrics.csv). Mirrors project 02.13's GateResult exactly.
// ===========================================================================
struct GateResult {
    std::string name;
    double measured = 0.0;
    double threshold = 0.0;
    bool pass = false;
    std::string note;
};

// ===========================================================================
// A tiny PPM (binary P6) canvas - the triptych artifact's rendering surface
// (no image library; 02.13's identical choice and reasoning: PPM's format
// IS "write the header, then raw RGB bytes").
// ===========================================================================
struct PpmCanvas {
    int w, h;
    std::vector<unsigned char> rgb;

    PpmCanvas(int w_, int h_) : w(w_), h(h_), rgb(static_cast<size_t>(w_) * static_cast<size_t>(h_) * 3, 0) {}

    void set_px(int x, int y, unsigned char r, unsigned char g, unsigned char b)
    {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 3;
        rgb[idx] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
    }

    void splat(int x, int y, unsigned char r, unsigned char g, unsigned char b)
    {
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                set_px(x + dx, y + dy, r, g, b);
    }

    bool write(const std::string& path) const
    {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
        return f.good();
    }
};

// Scene bounds for the top-view projection (x:[0,46], y:[-40,40] m) - covers
// every real object AND the dust plume/scatterer scatter in this project's
// committed scene (scripts/make_synthetic.py's module docstring).
static constexpr float kSceneXMin = 0.0f, kSceneXMax = 46.0f;
static constexpr float kSceneYMin = -40.0f, kSceneYMax = 40.0f;

static void world_to_panel_px(float x, float y, int panel_x0, int panel_w, int panel_h, int& px, int& py)
{
    const float u = (x - kSceneXMin) / (kSceneXMax - kSceneXMin);
    const float v = (y - kSceneYMin) / (kSceneYMax - kSceneYMin);
    px = panel_x0 + static_cast<int>(u * static_cast<float>(panel_w - 1));
    py = (panel_h - 1) - static_cast<int>(v * static_cast<float>(panel_h - 1));
}

// write_triptych - three top-view panels of the SNOW scan (README "Expected
// output"): RAW (every point, real=light gray, snow scatterer=cyan speckle),
// DROR-CLEANED and LIOR-CLEANED (only RETAINED points; a retained scatterer
// - a false negative - is drawn in orange, honestly shown per 02.13's
// identical convention, not hidden).
static bool write_triptych(const std::string& path, const PointSet& ps,
                           const std::vector<int32_t>& dror_mask, const std::vector<int32_t>& lior_mask)
{
    constexpr int panel_w = 480, panel_h = 480, gutter = 6;
    const int W = panel_w * 3 + gutter * 2;
    PpmCanvas canvas(W, panel_h);
    for (auto& c : canvas.rgb) c = 20;   // near-black background

    const int x0_raw = 0;
    const int x0_dror = panel_w + gutter;
    const int x0_lior = 2 * (panel_w + gutter);

    for (int i = 0; i < ps.n; ++i) {
        const float x = ps.xyz[i * 3 + 0], y = ps.xyz[i * 3 + 1];
        int px, py;

        world_to_panel_px(x, y, x0_raw, panel_w, panel_h, px, py);
        if (ps.is_real[i]) canvas.splat(px, py, 190, 190, 190);        // real: light gray
        else                canvas.splat(px, py, 110, 200, 255);       // snow scatterer: cyan speckle

        if (dror_mask[i] == 0) {   // RETAINED by DROR
            world_to_panel_px(x, y, x0_dror, panel_w, panel_h, px, py);
            if (ps.is_real[i]) canvas.splat(px, py, 190, 190, 190);
            else                canvas.splat(px, py, 255, 140, 0);     // false negative: orange
        }
        if (lior_mask[i] == 0) {   // RETAINED by LIOR
            world_to_panel_px(x, y, x0_lior, panel_w, panel_h, px, py);
            if (ps.is_real[i]) canvas.splat(px, py, 190, 190, 190);
            else                canvas.splat(px, py, 255, 140, 0);
        }
    }
    return canvas.write(path);
}

// ===========================================================================
// GpuScanBuffers - one scan's device allocations, alive for exactly as long
// as that scan is being processed (allocated/freed per call - CLAUDE.md
// paragraph 12's "free memory as soon as done" habit; at this project's
// point counts, cudaMalloc's ~100us overhead is negligible next to the
// O(n^2) kernels it feeds).
// ===========================================================================
struct GpuScanBuffers {
    float* xyz = nullptr;
    float* intensity = nullptr;
    float* mean_dist = nullptr;
    int32_t* dror_count = nullptr;
    int32_t* lior_count = nullptr;
    int32_t* sor_mask = nullptr;
    int32_t* dror_mask = nullptr;
    int32_t* lior_mask = nullptr;

    void alloc(int n)
    {
        CUDA_CHECK(cudaMalloc(&xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&intensity, static_cast<size_t>(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mean_dist, static_cast<size_t>(n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dror_count, static_cast<size_t>(n) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&lior_count, static_cast<size_t>(n) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&sor_mask, static_cast<size_t>(n) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&dror_mask, static_cast<size_t>(n) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&lior_mask, static_cast<size_t>(n) * sizeof(int32_t)));
    }
    void free_all()
    {
        CUDA_CHECK(cudaFree(xyz)); CUDA_CHECK(cudaFree(intensity)); CUDA_CHECK(cudaFree(mean_dist));
        CUDA_CHECK(cudaFree(dror_count)); CUDA_CHECK(cudaFree(lior_count));
        CUDA_CHECK(cudaFree(sor_mask)); CUDA_CHECK(cudaFree(dror_mask)); CUDA_CHECK(cudaFree(lior_mask));
    }
};

// run_gpu_pipeline - the full, ALREADY-VERIFIED three-filter pipeline for
// one scan: upload, six kernel launches (SOR needs a host-side mean/std
// reduction between its two stages - see the inline comment below), copy
// masks back. Used for every scan's SECONDARY gate-driving pass (file
// header point 3) - normal production use of certified kernels, not a
// second verification.
static void run_gpu_pipeline(const PointSet& ps,
                             std::vector<int32_t>& sor_mask,
                             std::vector<int32_t>& dror_mask,
                             std::vector<int32_t>& lior_mask,
                             double& gpu_ms_accum)
{
    const int n = ps.n;
    GpuScanBuffers buf;
    buf.alloc(n);
    CUDA_CHECK(cudaMemcpy(buf.xyz, ps.xyz.data(), static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.intensity, ps.intensity.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer gt;
    gt.begin();
    launch_sor_mean_knn_dist(n, buf.xyz, buf.mean_dist);
    launch_dror_neighbor_count(n, buf.xyz, buf.dror_count);
    launch_lior_neighbor_count(n, buf.xyz, buf.lior_count);
    gpu_ms_accum += static_cast<double>(gt.end_ms());

    // SOR's global mu/sigma reduction: host-side, over the (small, ~1-2K
    // element) mean_dist array copied back - kernels.cuh's file header
    // scoping note explains why this project does not also teach a GPU
    // reduction kernel here (that pattern lives in 08.01/23.01).
    std::vector<float> mean_dist(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(mean_dist.data(), buf.mean_dist, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    double mu_d = 0.0;
    for (float v : mean_dist) mu_d += v;
    mu_d /= n;
    double var_d = 0.0;
    for (float v : mean_dist) { const double d = v - mu_d; var_d += d * d; }
    var_d /= n;
    const float threshold = static_cast<float>(mu_d + kSorStdMult * std::sqrt(var_d));

    GpuTimer gt2;
    gt2.begin();
    launch_sor_classify(n, buf.mean_dist, threshold, buf.sor_mask);
    launch_dror_classify(n, buf.dror_count, buf.dror_mask);
    launch_lior_classify(n, buf.intensity, buf.lior_count, buf.lior_mask);
    gpu_ms_accum += static_cast<double>(gt2.end_ms());

    sor_mask.assign(static_cast<size_t>(n), 0);
    dror_mask.assign(static_cast<size_t>(n), 0);
    lior_mask.assign(static_cast<size_t>(n), 0);
    CUDA_CHECK(cudaMemcpy(sor_mask.data(), buf.sor_mask, static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dror_mask.data(), buf.dror_mask, static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lior_mask.data(), buf.lior_mask, static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyDeviceToHost));

    buf.free_all();
}

// point_in_box - AABB membership test (dust-plume-core gate).
static bool point_in_box(const float p[3], const float lo[3], const float hi[3])
{
    return p[0] >= lo[0] && p[0] <= hi[0] &&
           p[1] >= lo[1] && p[1] <= hi[1] &&
           p[2] >= lo[2] && p[2] <= hi[2];
}

int main(int argc, char** argv)
{
    std::string data_dir_override;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir_override = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data DIR]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Weather filtering: snow/rain/dust outlier removal (DROR/LIOR) - project 02.18\n");
    print_device_info();

    // ---- 0) Data ------------------------------------------------------------
    const std::string points_path = find_data_file(data_dir_override, argv[0], "points.csv");
    if (points_path.empty()) {
        std::printf("[info] points.csv not found under data/sample/ (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::printf("[info] data: %s\n", points_path.c_str());

    PointSet scans[kNumWeatherScans];
    if (!load_points(points_path, scans)) {
        std::printf("RESULT: FAIL (data missing or malformed - see stderr)\n");
        return 1;
    }

    std::printf("PROBLEM: 3 weather scans (SNOW/RAIN/DUST) of the SAME static scene, up to %d beams/scan, "
               "3 filters (SOR/DROR/LIOR), brute-force radius/KNN search\n", kBeamsPerScan);
    for (int w = 0; w < kNumWeatherScans; ++w) {
        int n_real = 0;
        for (int32_t v : scans[w].is_real) n_real += v;
        std::printf("[info] %-5s scan: %d points (%d real, %d scatterer)\n",
                    weather_name(w), scans[w].n, n_real, scans[w].n - n_real);
    }

    // ======================= VERIFY STAGE (on the SNOW scan) ===================
    // kernels.cuh's file header explains why one scan certifies all three:
    // no kernel here branches on which weather scan it is fed.
    bool all_verify_pass = true;
    const PointSet& vps = scans[kWeatherSnow];
    const int vn = vps.n;

    float* d_vxyz = nullptr; float* d_vintensity = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vxyz, static_cast<size_t>(vn) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vintensity, static_cast<size_t>(vn) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vxyz, vps.xyz.data(), static_cast<size_t>(vn) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vintensity, vps.intensity.data(), static_cast<size_t>(vn) * sizeof(float), cudaMemcpyHostToDevice));

    // ---- VERIFY SOR (statistic: tight tolerance; classify: exact given input) ----
    {
        std::vector<float> cpu_mean(static_cast<size_t>(vn));
        CpuTimer ct; ct.begin();
        sor_mean_knn_dist_cpu(vn, vps.xyz.data(), cpu_mean.data());
        const double cpu_ms = ct.end_ms();

        float* d_mean = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mean, static_cast<size_t>(vn) * sizeof(float)));
        GpuTimer gt; gt.begin();
        launch_sor_mean_knn_dist(vn, d_vxyz, d_mean);
        const float gpu_ms = gt.end_ms();
        std::vector<float> gpu_mean(static_cast<size_t>(vn));
        CUDA_CHECK(cudaMemcpy(gpu_mean.data(), d_mean, static_cast<size_t>(vn) * sizeof(float), cudaMemcpyDeviceToHost));

        float worst_abs = 0.0f, worst_rel = 0.0f;
        for (int i = 0; i < vn; ++i) {
            const float d = std::fabs(gpu_mean[static_cast<size_t>(i)] - cpu_mean[static_cast<size_t>(i)]);
            if (d > worst_abs) worst_abs = d;
            const float scale = std::fabs(cpu_mean[static_cast<size_t>(i)]) > 1e-6f ? std::fabs(cpu_mean[static_cast<size_t>(i)]) : 1e-6f;
            const float r = d / scale;
            if (r > worst_rel) worst_rel = r;
        }
        const bool stat_pass = worst_abs <= 1e-3f;   // meters; K=8 sqrt-sum over sub-50m coordinates
        std::printf("[info] SOR mean-KNN-distance: worst |gpu-cpu| = %.3e m (rel %.3e) over %d points; "
                    "CPU %.2f ms | GPU kernel %.3f ms\n",
                    static_cast<double>(worst_abs), static_cast<double>(worst_rel), vn, cpu_ms, static_cast<double>(gpu_ms));
        std::printf("VERIFY: SOR mean-KNN-distance %s (GPU vs CPU, tol 1e-3 m)\n", stat_pass ? "PASS" : "FAIL");
        if (!stat_pass) all_verify_pass = false;

        // Classify stage: SAME (CPU-computed) array fed to both paths, per
        // kernels.cuh's file header - guarantees the comparison below tests
        // the CLASSIFY logic alone, not statistic drift.
        double mu_d = 0.0;
        for (float v : cpu_mean) mu_d += v;
        mu_d /= vn;
        double var_d = 0.0;
        for (float v : cpu_mean) { const double d = v - mu_d; var_d += d * d; }
        var_d /= vn;
        const float threshold = static_cast<float>(mu_d + kSorStdMult * std::sqrt(var_d));

        CUDA_CHECK(cudaMemcpy(d_mean, cpu_mean.data(), static_cast<size_t>(vn) * sizeof(float), cudaMemcpyHostToDevice));
        int32_t* d_mask = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mask, static_cast<size_t>(vn) * sizeof(int32_t)));
        launch_sor_classify(vn, d_mean, threshold, d_mask);
        std::vector<int32_t> gpu_mask(static_cast<size_t>(vn));
        CUDA_CHECK(cudaMemcpy(gpu_mask.data(), d_mask, static_cast<size_t>(vn) * sizeof(int32_t), cudaMemcpyDeviceToHost));

        std::vector<int32_t> cpu_mask(static_cast<size_t>(vn));
        sor_classify_cpu(vn, cpu_mean.data(), threshold, cpu_mask.data());

        int mismatches = 0;
        for (int i = 0; i < vn; ++i) if (gpu_mask[static_cast<size_t>(i)] != cpu_mask[static_cast<size_t>(i)]) ++mismatches;
        std::printf("[info] SOR classify: %d mask mismatch(es) of %d (threshold=%.4f m)\n", mismatches, vn, static_cast<double>(threshold));
        std::printf("VERIFY: SOR classify %s (GPU vs CPU, same input array, exact)\n", mismatches == 0 ? "PASS" : "FAIL");
        if (mismatches != 0) all_verify_pass = false;

        CUDA_CHECK(cudaFree(d_mean)); CUDA_CHECK(cudaFree(d_mask));
    }

    // ---- VERIFY DROR (statistic: exact integer count; classify: exact) --------
    {
        std::vector<int32_t> cpu_count(static_cast<size_t>(vn));
        CpuTimer ct; ct.begin();
        dror_neighbor_count_cpu(vn, vps.xyz.data(), cpu_count.data());
        const double cpu_ms = ct.end_ms();

        int32_t* d_count = nullptr;
        CUDA_CHECK(cudaMalloc(&d_count, static_cast<size_t>(vn) * sizeof(int32_t)));
        GpuTimer gt; gt.begin();
        launch_dror_neighbor_count(vn, d_vxyz, d_count);
        const float gpu_ms = gt.end_ms();
        std::vector<int32_t> gpu_count(static_cast<size_t>(vn));
        CUDA_CHECK(cudaMemcpy(gpu_count.data(), d_count, static_cast<size_t>(vn) * sizeof(int32_t), cudaMemcpyDeviceToHost));

        int mismatches = 0;
        for (int i = 0; i < vn; ++i) if (gpu_count[static_cast<size_t>(i)] != cpu_count[static_cast<size_t>(i)]) ++mismatches;
        std::printf("[info] DROR neighbor count: %d mismatch(es) of %d points; CPU %.2f ms | GPU kernel %.3f ms\n",
                    mismatches, vn, cpu_ms, static_cast<double>(gpu_ms));
        std::printf("VERIFY: DROR neighbor count %s (GPU vs CPU, exact integers)\n", mismatches == 0 ? "PASS" : "FAIL");
        if (mismatches != 0) all_verify_pass = false;

        int32_t* d_mask = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mask, static_cast<size_t>(vn) * sizeof(int32_t)));
        launch_dror_classify(vn, d_count, d_mask);   // GPU classify fed the GPU's OWN (exact-verified) count
        std::vector<int32_t> gpu_mask(static_cast<size_t>(vn));
        CUDA_CHECK(cudaMemcpy(gpu_mask.data(), d_mask, static_cast<size_t>(vn) * sizeof(int32_t), cudaMemcpyDeviceToHost));

        std::vector<int32_t> cpu_mask(static_cast<size_t>(vn));
        dror_classify_cpu(vn, cpu_count.data(), cpu_mask.data());   // CPU classify fed the CPU's OWN count

        int mask_mismatches = 0;
        for (int i = 0; i < vn; ++i) if (gpu_mask[static_cast<size_t>(i)] != cpu_mask[static_cast<size_t>(i)]) ++mask_mismatches;
        std::printf("[info] DROR classify: %d mask mismatch(es) of %d\n", mask_mismatches, vn);
        std::printf("VERIFY: DROR classify %s (GPU vs CPU, given the exact-verified counts)\n", mask_mismatches == 0 ? "PASS" : "FAIL");
        if (mask_mismatches != 0) all_verify_pass = false;

        CUDA_CHECK(cudaFree(d_count)); CUDA_CHECK(cudaFree(d_mask));
    }

    // ---- VERIFY LIOR (statistic: exact integer count; classify: exact) --------
    {
        std::vector<int32_t> cpu_count(static_cast<size_t>(vn));
        CpuTimer ct; ct.begin();
        lior_neighbor_count_cpu(vn, vps.xyz.data(), cpu_count.data());
        const double cpu_ms = ct.end_ms();

        int32_t* d_count = nullptr;
        CUDA_CHECK(cudaMalloc(&d_count, static_cast<size_t>(vn) * sizeof(int32_t)));
        GpuTimer gt; gt.begin();
        launch_lior_neighbor_count(vn, d_vxyz, d_count);
        const float gpu_ms = gt.end_ms();
        std::vector<int32_t> gpu_count(static_cast<size_t>(vn));
        CUDA_CHECK(cudaMemcpy(gpu_count.data(), d_count, static_cast<size_t>(vn) * sizeof(int32_t), cudaMemcpyDeviceToHost));

        int mismatches = 0;
        for (int i = 0; i < vn; ++i) if (gpu_count[static_cast<size_t>(i)] != cpu_count[static_cast<size_t>(i)]) ++mismatches;
        std::printf("[info] LIOR neighbor count: %d mismatch(es) of %d points; CPU %.2f ms | GPU kernel %.3f ms\n",
                    mismatches, vn, cpu_ms, static_cast<double>(gpu_ms));
        std::printf("VERIFY: LIOR neighbor count %s (GPU vs CPU, exact integers)\n", mismatches == 0 ? "PASS" : "FAIL");
        if (mismatches != 0) all_verify_pass = false;

        int32_t* d_mask = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mask, static_cast<size_t>(vn) * sizeof(int32_t)));
        launch_lior_classify(vn, d_vintensity, d_count, d_mask);
        std::vector<int32_t> gpu_mask(static_cast<size_t>(vn));
        CUDA_CHECK(cudaMemcpy(gpu_mask.data(), d_mask, static_cast<size_t>(vn) * sizeof(int32_t), cudaMemcpyDeviceToHost));

        std::vector<int32_t> cpu_mask(static_cast<size_t>(vn));
        lior_classify_cpu(vn, vps.intensity.data(), cpu_count.data(), cpu_mask.data());

        int mask_mismatches = 0;
        for (int i = 0; i < vn; ++i) if (gpu_mask[static_cast<size_t>(i)] != cpu_mask[static_cast<size_t>(i)]) ++mask_mismatches;
        std::printf("[info] LIOR classify: %d mask mismatch(es) of %d\n", mask_mismatches, vn);
        std::printf("VERIFY: LIOR classify %s (GPU vs CPU, given the exact-verified counts + shared intensity)\n",
                    mask_mismatches == 0 ? "PASS" : "FAIL");
        if (mask_mismatches != 0) all_verify_pass = false;

        CUDA_CHECK(cudaFree(d_count)); CUDA_CHECK(cudaFree(d_mask));
    }

    CUDA_CHECK(cudaFree(d_vxyz)); CUDA_CHECK(cudaFree(d_vintensity));

    if (!all_verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement in the verify stage - fix before trusting any gate below)\n");
        return 1;
    }

    // ======================= SECONDARY: run all 3 scans =========================
    // Normal use of the now-certified kernels (file header point 3).
    std::vector<int32_t> sor_mask[kNumWeatherScans], dror_mask[kNumWeatherScans], lior_mask[kNumWeatherScans];
    double total_gpu_ms = 0.0;
    for (int w = 0; w < kNumWeatherScans; ++w)
        run_gpu_pipeline(scans[w], sor_mask[w], dror_mask[w], lior_mask[w], total_gpu_ms);
    std::printf("[time] all 3 scans, all 3 filters (6 kernel launches/scan): %.3f ms total GPU kernel time\n", total_gpu_ms);

    // ======================= GATES ================================================
    std::vector<GateResult> gates;

    auto add_gate = [&](const char* name, double measured, double threshold, bool pass, const char* note) {
        gates.push_back(GateResult{ name, measured, threshold, pass, note });
        std::printf("[info] %s: measured=%.2f threshold=%.2f -> %s (%s)\n",
                    name, measured, threshold, pass ? "PASS" : "FAIL", note);
    };

    // ---- SOR: reported only, no floor (file header / README) -------------------
    for (int w = 0; w < kNumWeatherScans; ++w) {
        const PrecisionRecall r = pr_over(sor_mask[w], scans[w]);
        std::printf("[info] SOR  %-5s: precision=%.1f%% recall=%.1f%% real_preserve=%.1f%% (reported, not gated)\n",
                    weather_name(w), r.precision_pct(), r.recall_pct(), r.real_preserve_pct());
    }

    // ---- DROR / LIOR precision+recall floors on SNOW and RAIN -------------------
    {
        const PrecisionRecall d_snow = pr_over(dror_mask[kWeatherSnow], scans[kWeatherSnow]);
        const PrecisionRecall d_rain = pr_over(dror_mask[kWeatherRain], scans[kWeatherRain]);
        const PrecisionRecall l_snow = pr_over(lior_mask[kWeatherSnow], scans[kWeatherSnow]);
        const PrecisionRecall l_rain = pr_over(lior_mask[kWeatherRain], scans[kWeatherRain]);

        add_gate("dror_recall_snow", d_snow.recall_pct(), kGateDrorRecallSnowFloorPct,
                d_snow.recall_pct() >= kGateDrorRecallSnowFloorPct, "DROR recall removing snow scatterer points");
        add_gate("dror_recall_rain", d_rain.recall_pct(), kGateDrorRecallRainFloorPct,
                d_rain.recall_pct() >= kGateDrorRecallRainFloorPct, "DROR recall removing rain scatterer points");
        add_gate("dror_precision_snow", d_snow.precision_pct(), kGateDrorPrecisionSnowFloorPct,
                d_snow.precision_pct() >= kGateDrorPrecisionSnowFloorPct, "DROR precision on snow scan");
        add_gate("dror_precision_rain", d_rain.precision_pct(), kGateDrorPrecisionRainFloorPct,
                d_rain.precision_pct() >= kGateDrorPrecisionRainFloorPct, "DROR precision on rain scan");
        add_gate("lior_recall_snow", l_snow.recall_pct(), kGateLiorRecallSnowFloorPct,
                l_snow.recall_pct() >= kGateLiorRecallSnowFloorPct, "LIOR recall removing snow scatterer points");
        add_gate("lior_recall_rain", l_rain.recall_pct(), kGateLiorRecallRainFloorPct,
                l_rain.recall_pct() >= kGateLiorRecallRainFloorPct, "LIOR recall removing rain scatterer points");
        add_gate("lior_precision_snow", l_snow.precision_pct(), kGateLiorPrecisionSnowFloorPct,
                l_snow.precision_pct() >= kGateLiorPrecisionSnowFloorPct, "LIOR precision on snow scan");
        add_gate("lior_precision_rain", l_rain.precision_pct(), kGateLiorPrecisionRainFloorPct,
                l_rain.precision_pct() >= kGateLiorPrecisionRainFloorPct, "LIOR precision on rain scan");
    }

    // ---- real_point_preservation: aggregated across all 3 scans ----------------
    {
        long long dror_tn = 0, dror_fp = 0, lior_tn = 0, lior_fp = 0;
        for (int w = 0; w < kNumWeatherScans; ++w) {
            const PrecisionRecall d = pr_over(dror_mask[w], scans[w]);
            const PrecisionRecall l = pr_over(lior_mask[w], scans[w]);
            dror_tn += d.tn; dror_fp += d.fp;
            lior_tn += l.tn; lior_fp += l.fp;
        }
        const double dror_pct = (dror_tn + dror_fp) > 0 ? 100.0 * dror_tn / (dror_tn + dror_fp) : 0.0;
        const double lior_pct = (lior_tn + lior_fp) > 0 ? 100.0 * lior_tn / (lior_tn + lior_fp) : 0.0;
        add_gate("real_point_preservation_dror", dror_pct, kGateRealPreserveDrorFloorPct,
                dror_pct >= kGateRealPreserveDrorFloorPct, "real points correctly KEPT by DROR across all 3 scans combined");
        add_gate("real_point_preservation_lior", lior_pct, kGateRealPreserveLiorFloorPct,
                lior_pct >= kGateRealPreserveLiorFloorPct, "real points correctly KEPT by LIOR across all 3 scans combined");
    }

    // ---- sor_far_range_failure: both directions asserted -----------------------
    {
        long long far_n = 0, sor_bad = 0, dror_bad = 0;
        for (int w = 0; w < kNumWeatherScans; ++w) {
            const PointSet& ps = scans[w];
            for (int i = 0; i < ps.n; ++i) {
                if (!ps.is_real[i]) continue;
                const float p[3] = { ps.xyz[i * 3 + 0], ps.xyz[i * 3 + 1], ps.xyz[i * 3 + 2] };
                if (range3(p) < kRangeFarM) continue;
                ++far_n;
                if (sor_mask[w][i]) ++sor_bad;
                if (dror_mask[w][i]) ++dror_bad;
            }
        }
        const double sor_pct = far_n > 0 ? 100.0 * sor_bad / far_n : 0.0;
        const double dror_pct = far_n > 0 ? 100.0 * dror_bad / far_n : 0.0;
        const bool pass = (sor_pct >= kGateSorFarFalseFloorPct) && (dror_pct <= kGateDrorFarFalseCeilingPct);
        std::printf("[info] sor_far_range_failure: %lld far-range (>= %.0f m) real points; SOR falsely "
                    "removes %.1f%% (floor %.1f%%), DROR falsely removes %.1f%% (ceiling %.1f%%)\n",
                    far_n, static_cast<double>(kRangeFarM), sor_pct, kGateSorFarFalseFloorPct,
                    dror_pct, kGateDrorFarFalseCeilingPct);
        gates.push_back(GateResult{ "sor_far_range_failure", sor_pct, kGateSorFarFalseFloorPct, pass,
                                    "SOR fails badly AND DROR does not, on the same far-range real cohort" });
    }

    // ---- dust_plume_honesty: measured, not performance-gated -------------------
    {
        const PointSet& ps = scans[kWeatherDust];
        std::vector<int> core_idx;
        for (int i = 0; i < ps.n; ++i) {
            const float p[3] = { ps.xyz[i * 3 + 0], ps.xyz[i * 3 + 1], ps.xyz[i * 3 + 2] };
            if (point_in_box(p, kPlumeBoxMin, kPlumeBoxMax)) core_idx.push_back(i);
        }
        const PrecisionRecall d = pr_over(dror_mask[kWeatherDust], ps, &core_idx);
        const PrecisionRecall l = pr_over(lior_mask[kWeatherDust], ps, &core_idx);
        const bool measured_ok = !core_idx.empty();
        std::printf("[info] dust_plume_honesty: %zu points inside the dust plume box; DROR precision=%.1f%% "
                    "recall=%.1f%% | LIOR precision=%.1f%% recall=%.1f%% - measured honestly, NOT floor-gated "
                    "(the designed hard case: a dense enough scatterer field can statistically resemble a "
                    "real surface - see THEORY.md)\n",
                    core_idx.size(), d.precision_pct(), d.recall_pct(), l.precision_pct(), l.recall_pct());
        gates.push_back(GateResult{ "dust_plume_honesty", static_cast<double>(core_idx.size()), 0.0, measured_ok,
                                    "measurement occurred on a non-empty plume-core cohort (no performance floor by design)" });
    }

    // ---- combined DROR+LIOR union/intersection ([info] only) -------------------
    for (int w = 0; w < kNumWeatherScans; ++w) {
        std::vector<int32_t> uni(static_cast<size_t>(scans[w].n)), inter(static_cast<size_t>(scans[w].n));
        for (int i = 0; i < scans[w].n; ++i) {
            uni[i] = (dror_mask[w][i] || lior_mask[w][i]) ? 1 : 0;
            inter[i] = (dror_mask[w][i] && lior_mask[w][i]) ? 1 : 0;
        }
        const PrecisionRecall pu = pr_over(uni, scans[w]);
        const PrecisionRecall pi = pr_over(inter, scans[w]);
        std::printf("[info] combined %-5s: UNION recall=%.1f%% precision=%.1f%% | INTERSECTION recall=%.1f%% "
                    "precision=%.1f%% (production practice: union raises recall, intersection raises precision)\n",
                    weather_name(w), pu.recall_pct(), pu.precision_pct(), pi.recall_pct(), pi.precision_pct());
    }

    // ---- intensity_dependence: LIOR recall, clean vs perturbed intensity -------
    {
        const PointSet& ps = scans[kWeatherSnow];
        std::vector<float> perturbed(static_cast<size_t>(ps.n));
        for (int i = 0; i < ps.n; ++i) {
            const float p[3] = { ps.xyz[i * 3 + 0], ps.xyz[i * 3 + 1], ps.xyz[i * 3 + 2] };
            const float r = range3(p);
            // An uncalibrated per-channel gain that has NOT fully divided out
            // range falloff (project 02.20's territory, stated honestly in
            // README/THEORY): gain = (ref/r)^p brightens near returns and dims
            // far ones relative to the properly-calibrated intensity already
            // in the committed data.
            const float gain = powf(kIntensityPerturbRefRangeM / std::max(r, 1.0f), kIntensityPerturbExponent);
            float v = ps.intensity[i] * gain;
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            perturbed[i] = v;
        }
        std::vector<int32_t> count(static_cast<size_t>(ps.n));
        lior_neighbor_count_cpu(ps.n, ps.xyz.data(), count.data());   // density unaffected by intensity
        std::vector<int32_t> mask_perturbed(static_cast<size_t>(ps.n));
        lior_classify_cpu(ps.n, perturbed.data(), count.data(), mask_perturbed.data());

        const PrecisionRecall clean = pr_over(lior_mask[kWeatherSnow], ps);
        const PrecisionRecall pert = pr_over(mask_perturbed, ps);
        std::printf("[info] intensity_dependence: LIOR recall with CLEAN (as-calibrated) intensity = %.1f%%; "
                    "with a documented miscalibration perturbation (gain=(%.0fm/r)^%.1f, the 02.20 dependency "
                    "quantified) = %.1f%% (delta %.1f pp) - LIOR's decisions move when calibration drifts, "
                    "DROR/SOR (geometry-only) do not\n",
                    clean.recall_pct(), static_cast<double>(kIntensityPerturbRefRangeM),
                    static_cast<double>(kIntensityPerturbExponent), pert.recall_pct(), clean.recall_pct() - pert.recall_pct());
    }

    // ---- range_stratified ([info] + CSV) ----------------------------------------
    struct RangeRow { const char* band; int weather; long long n; double sor_bad, dror_bad, lior_bad; };
    std::vector<RangeRow> range_rows;
    {
        struct Band { const char* name; float lo, hi; };
        const Band bands[3] = { {"near", 0.0f, kRangeNearM}, {"mid", kRangeNearM, kRangeFarM}, {"far", kRangeFarM, 1e9f} };
        for (const Band& b : bands) {
            for (int w = 0; w < kNumWeatherScans; ++w) {
                const PointSet& ps = scans[w];
                long long n = 0, sor_bad = 0, dror_bad = 0, lior_bad = 0;
                for (int i = 0; i < ps.n; ++i) {
                    if (!ps.is_real[i]) continue;
                    const float p[3] = { ps.xyz[i * 3 + 0], ps.xyz[i * 3 + 1], ps.xyz[i * 3 + 2] };
                    const float r = range3(p);
                    if (r < b.lo || r >= b.hi) continue;
                    ++n;
                    if (sor_mask[w][i]) ++sor_bad;
                    if (dror_mask[w][i]) ++dror_bad;
                    if (lior_mask[w][i]) ++lior_bad;
                }
                const double sp = n > 0 ? 100.0 * sor_bad / n : 0.0;
                const double dp = n > 0 ? 100.0 * dror_bad / n : 0.0;
                const double lp = n > 0 ? 100.0 * lior_bad / n : 0.0;
                range_rows.push_back(RangeRow{ b.name, w, n, sp, dp, lp });
                std::printf("[info] range_stratified %-4s %-5s: n=%-4lld real-point false-remove SOR=%.1f%% DROR=%.1f%% LIOR=%.1f%%\n",
                            b.name, weather_name(w), n, sp, dp, lp);
            }
        }
    }

    // ======================= ARTIFACTS ==============================================
    const std::string out_dir = resolve_out_dir(argv[0]);

    const std::string triptych_path = out_dir + "/triptych_snow.ppm";
    const bool triptych_ok = write_triptych(triptych_path, scans[kWeatherSnow], dror_mask[kWeatherSnow], lior_mask[kWeatherSnow]);
    std::printf("ARTIFACT: %s demo/out/triptych_snow.ppm (raw / DROR-cleaned / LIOR-cleaned top view)\n",
                triptych_ok ? "wrote" : "FAILED to write");

    std::string range_csv_path = out_dir + "/range_stratified.csv";
    bool range_csv_ok = false;
    {
        std::ofstream f(range_csv_path);
        if (f.is_open()) {
            f << "band,weather,n,sor_false_remove_pct,dror_false_remove_pct,lior_false_remove_pct\n";
            for (const auto& r : range_rows)
                f << r.band << ',' << weather_name(r.weather) << ',' << r.n << ',' << r.sor_bad << ',' << r.dror_bad << ',' << r.lior_bad << '\n';
            range_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/range_stratified.csv (%zu rows)\n",
                range_csv_ok ? "wrote" : "FAILED to write", range_rows.size());

    std::string gates_csv_path = out_dir + "/gates_metrics.csv";
    bool gates_csv_ok = false;
    {
        std::ofstream f(gates_csv_path);
        if (f.is_open()) {
            f << "gate,measured,threshold,verdict,note\n";
            for (const auto& g : gates)
                f << g.name << ',' << g.measured << ',' << g.threshold << ',' << (g.pass ? "PASS" : "FAIL") << ',' << g.note << '\n';
            gates_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/gates_metrics.csv (%zu gates)\n",
                gates_csv_ok ? "wrote" : "FAILED to write", gates.size());

    // ======================= FINAL VERDICT ===========================================
    for (const auto& g : gates)
        std::printf("GATE: %s %s\n", g.name.c_str(), g.pass ? "PASS" : "FAIL");

    bool all_gates_pass = true;
    for (const auto& g : gates) all_gates_pass = all_gates_pass && g.pass;
    const bool artifacts_ok = triptych_ok && range_csv_ok && gates_csv_ok;
    const bool success = all_verify_pass && all_gates_pass && artifacts_ok;

    if (success)
        std::printf("RESULT: PASS (all verify stages and gates passed; DROR/LIOR clean the weather-noise scans)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above and stderr)\n");
    return success ? 0 : 1;
}
