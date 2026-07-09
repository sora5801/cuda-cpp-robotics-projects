// ===========================================================================
// main.cu — entry point for project 04.01
//           Massive particle filter localization
//           (teaching core: 2-D range-beam Monte Carlo localization)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed synthetic sample
//      (occupancy grid map + 120-step trajectory with noisy odometry and
//      noisy range scans) from data/sample/.
//   2. Initialize K = 100,000 particles around the known start pose
//      (pose TRACKING; global localization is README Exercise 4).
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate): step 0's exact inputs through
//      both paths — GPU predict vs CPU predict (small ABSOLUTE tolerance:
//      the pose update is smooth, only trig/FMA ulps differ), then GPU
//      weight vs CPU weight ON THE SAME POSES (small RELATIVE tolerance:
//      given identical inputs the contraction-safe ray-march visits
//      identical cells, so only the final accumulation rounds differently).
//      Feeding the weight twins identical poses isolates each kernel: the
//      ray-march is DISCONTINUOUS in pose, so chaining the two comparisons
//      would smear the gate (THEORY.md §verification).
//   4. CLOSED LOOP: run the full filter for all 120 steps — GPU predict,
//      GPU weight, host normalize + weighted-mean estimate + systematic
//      resample — logging estimate vs ground truth to
//      demo/out/trajectory_est.csv (the artifact; plot it).
//   5. SUCCESS CHECK: position RMSE of the estimate vs ground truth over
//      the whole run must beat kRmseGateM (0.15 m). Exit 0 only if the
//      verify gate and the success check both hold.
//
// The filter update implemented here (derivation in THEORY.md §the-math):
//      predict:   pose_k += twist(odo + noise_k) * dt          (GPU kernel)
//      weight:    logw_k = -sum_b (z_b - zhat_b(pose_k))^2/(2s^2)  (GPU kernel)
//      normalize: w_k = exp(logw_k - max_j logw_j)             (host, double)
//      estimate:  weighted mean of the cloud (heading via atan2 of mean
//                 sin/cos — the correct mean for a circular quantity)
//      resample:  systematic (one uniform draw, K evenly spaced probes)
// The host side is deliberately kept on the host: it is O(K) trivial
// arithmetic, and seeing normalize/estimate/resample in ~40 lines of plain
// C++ under the kernel calls is worth more didactically than a fused
// reduction kernel (that optimization is README Exercise 5).
//
// Determinism: every random number in the demo flows from xorshift32
// streams seeded from kBaseSeed (kernels.cuh) — the run is bit-reproducible
// on this machine. Across platforms, libm ulp differences (host trig in the
// estimate, double trig in the kernels' float casts) can shift low bits and
// — through the discontinuous resampling — chaotically alter individual
// particles; the RMSE is an aggregate over 100,000 particles x 120 steps
// and is statistically immune to that, which is why the stable verdict is a
// thresholded RMSE with wide margin and no trajectory numbers appear on
// stable lines (THEORY.md §numerics).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SAMPLE:", "VERIFY:",
// "ARTIFACT:", "RESULT:" — "[info]"/"[time]" unchecked. Change a stable
// line => update demo/expected_output.txt in the same commit.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// The occupancy grid map — loaded from data/sample/grid_map.txt.
// Layout contract (kernels.cuh): row-major, occ[iy*w + ix], 0 free / 1
// occupied; cell (ix,iy) covers [ix*res,(ix+1)*res) x [iy*res,(iy+1)*res).
// ---------------------------------------------------------------------------
struct GridMap {
    int w = 0, h = 0;                 // grid dimensions (cells)
    float res = 0.0f;                 // cell size (m)
    float inv_res = 0.0f;             // 1/res (1/m) — precomputed once so the
                                      // kernels convert world->cell with a
                                      // lone exact multiply (0.25 -> 4.0)
    std::vector<unsigned char> occ;   // [h*w] occupancy
    bool loaded = false;
};

// Strict loader: demands WIDTH/HEIGHT/RESOLUTION rows, then a MAP marker,
// then exactly HEIGHT rows of WIDTH '.'/'#' characters. The file stores the
// TOP row first (so it reads like a map with +y up); we flip on load
// (file line j -> iy = h-1-j). NOTE: '#' is a comment ONLY before the MAP
// marker — map rows legitimately start with '#' (border walls).
static GridMap load_map(const std::string& path)
{
    GridMap m;
    std::ifstream in(path);
    if (!in.is_open()) return m;

    std::string line;
    bool in_grid = false;
    int rows_read = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // tolerate CRLF checkouts
        if (!in_grid) {
            if (line.empty() || line[0] == '#') continue;            // header comments
            std::stringstream ss(line);
            std::string label, cell;
            std::getline(ss, label, ',');
            if      (label == "WIDTH")      { std::getline(ss, cell); m.w = std::atoi(cell.c_str()); }
            else if (label == "HEIGHT")     { std::getline(ss, cell); m.h = std::atoi(cell.c_str()); }
            else if (label == "RESOLUTION") { std::getline(ss, cell); m.res = std::strtof(cell.c_str(), nullptr); }
            else if (label == "MAP") {
                if (m.w < 1 || m.h < 1 || m.res <= 0.0f) {
                    std::fprintf(stderr, "map: MAP marker before valid WIDTH/HEIGHT/RESOLUTION\n");
                    return GridMap{};
                }
                m.occ.assign(static_cast<size_t>(m.w) * m.h, 0);
                in_grid = true;
            } else {
                std::fprintf(stderr, "map: unknown header row '%s'\n", label.c_str());
                return GridMap{};
            }
        } else {
            if (rows_read >= m.h) {
                if (line.empty()) continue;                          // tolerate one trailing newline
                std::fprintf(stderr, "map: more grid rows than HEIGHT\n");
                return GridMap{};
            }
            if (static_cast<int>(line.size()) != m.w) {
                std::fprintf(stderr, "map: grid row %d has %zu chars, expected %d\n",
                             rows_read, line.size(), m.w);
                return GridMap{};
            }
            const int iy = m.h - 1 - rows_read;                      // flip: top row first in file
            for (int ix = 0; ix < m.w; ++ix) {
                if (line[static_cast<size_t>(ix)] == '#')      m.occ[static_cast<size_t>(iy) * m.w + ix] = 1;
                else if (line[static_cast<size_t>(ix)] != '.') {
                    std::fprintf(stderr, "map: unexpected char '%c' in grid row %d\n",
                                 line[static_cast<size_t>(ix)], rows_read);
                    return GridMap{};
                }
            }
            ++rows_read;
        }
    }
    if (!in_grid || rows_read != m.h) {
        std::fprintf(stderr, "map: expected %d grid rows, got %d\n", m.h, rows_read);
        return GridMap{};
    }
    m.inv_res = 1.0f / m.res;
    m.loaded = true;
    return m;
}

// ---------------------------------------------------------------------------
// The trajectory log — loaded from data/sample/trajectory_scans.csv.
// Per step t: the ground-truth pose AFTER the step's twist, the noisy
// odometry measurement of that twist, and the kNumBeams noisy ranges taken
// at the post-step true pose (units/frames: data/README.md, kernels.cuh).
// ---------------------------------------------------------------------------
struct Log {
    float init_x = 0, init_y = 0, init_th = 0;    // true start pose (m, m, rad)
    int steps = 0;                                // T
    std::vector<float> gt_x, gt_y, gt_th;         // [T] ground truth (post-step)
    std::vector<float> odo_v, odo_w;              // [T] noisy twist (m/s, rad/s)
    std::vector<float> scan;                      // [T*kNumBeams] ranges (m), scan[t*B + b]
    bool loaded = false;
};

// Strict loader: INIT row, then the STEP header, then rows of exactly
// 7 + kNumBeams comma-separated fields whose step index matches the row
// count — any surprise aborts (a filter fed a half-parsed log would
// "work" and mislocalize, the worst kind of quiet failure).
static Log load_log(const std::string& path)
{
    Log lg;
    std::ifstream in(path);
    if (!in.is_open()) return lg;

    bool have_init = false, have_header = false;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        if (cell == "INIT") {
            float v[3];
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "log: short INIT row\n"); return Log{}; }
                v[i] = std::strtof(cell.c_str(), nullptr);
            }
            lg.init_x = v[0]; lg.init_y = v[1]; lg.init_th = v[2];
            have_init = true;
        } else if (cell == "STEP") {
            have_header = true;                       // column-name row: parsed for presence only
        } else {
            if (!have_init || !have_header) { std::fprintf(stderr, "log: data before INIT/header\n"); return Log{}; }
            if (std::atoi(cell.c_str()) != lg.steps) {
                std::fprintf(stderr, "log: step index %s out of order (expected %d)\n", cell.c_str(), lg.steps);
                return Log{};
            }
            float v[6 + kNumBeams];                    // t_s, gt(3), odo(2), z(16)
            for (int i = 0; i < 6 + kNumBeams; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "log: short row at step %d\n", lg.steps); return Log{}; }
                v[i] = std::strtof(cell.c_str(), nullptr);
            }
            // v[0] is t_s — redundant (= (step+1)*kDt), kept in the file for
            // humans and plotting tools; the filter derives time from kDt.
            lg.gt_x.push_back(v[1]);  lg.gt_y.push_back(v[2]);  lg.gt_th.push_back(v[3]);
            lg.odo_v.push_back(v[4]); lg.odo_w.push_back(v[5]);
            for (int b = 0; b < kNumBeams; ++b) lg.scan.push_back(v[6 + b]);
            ++lg.steps;
        }
    }
    if (!have_init || lg.steps < 1) {
        std::fprintf(stderr, "log: missing INIT or no step rows\n");
        return Log{};
    }
    lg.loaded = true;
    return lg;
}

// ---------------------------------------------------------------------------
// Path helpers — the repo's exe-relative pattern: the exe sits at
// build/x64/<Config>/, so the project root is three levels up.
// ---------------------------------------------------------------------------
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

// Locate the sample directory (must contain grid_map.txt): CLI override
// first, then exe-relative, then CWD-relative fallbacks for IDE launches.
static std::string find_data_dir(const std::string& cli_dir, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_dir.empty()) candidates.push_back(cli_dir);
    candidates.push_back(project_root_from(argv0) + "/data/sample");
    candidates.push_back("data/sample");
    candidates.push_back("../data/sample");
    for (const auto& c : candidates)
        if (std::ifstream(c + "/grid_map.txt").is_open()) return c;
    return "";
}

static bool ensure_dir(const std::string& path)
{
#ifdef _WIN32
    const int r = _mkdir(path.c_str());
#else
    const int r = mkdir(path.c_str(), 0755);
#endif
    return r == 0 || errno == EEXIST;
}

// ---------------------------------------------------------------------------
// wrap_angle — map any angle to (-pi, pi]. THE project's single defined wrap
// point (CLAUDE.md §12): particle headings stay unwrapped (only sin/cos of
// them are consumed); wrapping happens only here, when a heading DIFFERENCE
// is turned into an error number for reporting.
// ---------------------------------------------------------------------------
static float wrap_angle(float a)
{
    const float pi = 3.14159265358979323846f;
    while (a > pi)  a -= 2.0f * pi;
    while (a <= -pi) a += 2.0f * pi;
    return a;
}

// ---------------------------------------------------------------------------
// systematic_resample — the O(K) low-variance resampler (THEORY.md §algorithm).
//
// One uniform draw u in (0, 1/K]; probe the weight CDF at u, u+1/K, u+2/K, …
// Every particle with weight above 1/K survives at least once (multinomial
// resampling cannot promise that), and one draw means one RNG stream to
// document. Params:
//   K       : particle count
//   w, w_sum: UNNORMALIZED weights (double, from the host normalize step)
//   rng     : host RNG state (advanced by exactly one draw)
//   px/py/pth: particle arrays, REPLACED by the resampled cloud
// Complexity: O(K) — i and j each advance monotonically.
// ---------------------------------------------------------------------------
static void systematic_resample(int K, const std::vector<double>& w, double w_sum,
                                uint32_t& rng,
                                std::vector<float>& px, std::vector<float>& py,
                                std::vector<float>& pth)
{
    static std::vector<float> nx, ny, nth;        // reused across steps (no per-step alloc)
    nx.resize(static_cast<size_t>(K));
    ny.resize(static_cast<size_t>(K));
    nth.resize(static_cast<size_t>(K));

    const double step = 1.0 / K;
    const double u0 = static_cast<double>(pf_uniform01_cpu(rng)) * step;  // in (0, 1/K]
    const double inv_sum = 1.0 / w_sum;

    int i = 0;                                     // source particle cursor
    double c = w[0] * inv_sum;                     // running CDF
    for (int j = 0; j < K; ++j) {                  // j-th probe: u0 + j/K
        const double uj = u0 + j * step;
        while (uj > c && i < K - 1) { ++i; c += w[static_cast<size_t>(i)] * inv_sum; }
        nx[static_cast<size_t>(j)] = px[static_cast<size_t>(i)];
        ny[static_cast<size_t>(j)] = py[static_cast<size_t>(i)];
        nth[static_cast<size_t>(j)] = pth[static_cast<size_t>(i)];
    }
    px.swap(nx); py.swap(ny); pth.swap(nth);
}

// ---------------------------------------------------------------------------
// main — verify stage, then the closed loop described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int K = kDefaultK;             // particles (CLI-overridable for experiments)
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--particles") && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")      && i + 1 < argc) data_dir = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--particles K] [--data <dir with grid_map.txt + trajectory_scans.csv>]\n"
                "note: non-default K changes the PROBLEM line; the demo diff will flag it.\n",
                argv[0]);
            return 2;
        }
    }
    if (K < 1) { std::fprintf(stderr, "error: --particles must be >= 1\n"); return 2; }

    std::printf("[demo] massive particle filter localization: range-beam MCL on a 2-D grid (project 04.01)\n");
    print_device_info();

    // ---- sample data ---------------------------------------------------------
    const std::string dir = find_data_dir(data_dir, argv[0]);
    if (dir.empty()) {
        std::printf("SAMPLE: NOT FOUND — data/sample/grid_map.txt missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample missing)\n");
        return 1;
    }
    std::printf("[info] data dir: %s\n", dir.c_str());
    GridMap map = load_map(dir + "/grid_map.txt");
    Log lg = load_log(dir + "/trajectory_scans.csv");
    if (!map.loaded || !lg.loaded) {
        std::printf("SAMPLE: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sample malformed)\n");
        return 1;
    }
    std::printf("PROBLEM: bootstrap particle filter, K=%d particles, B=%d beams/scan, %dx%d grid map @ %.2f m, FP32\n",
                K, kNumBeams, map.w, map.h, static_cast<double>(map.res));
    std::printf("SAMPLE: %d steps @ %.0f Hz, noisy odometry + noisy range scans, ground truth known [synthetic]\n",
                lg.steps, 1.0 / static_cast<double>(kDt));

    // ---- initial particle cloud ----------------------------------------------
    // Pose tracking: Gaussian cloud around the KNOWN start pose (the sample's
    // INIT row). Drawn from the same portable RNG family as everything else;
    // one host stream (seeded kBaseSeed) also serves the resampler below.
    std::vector<float> h_px(static_cast<size_t>(K)), h_py(static_cast<size_t>(K)),
                       h_pth(static_cast<size_t>(K));
    uint32_t h_rng = kBaseSeed;
    for (int k = 0; k < K; ++k) {
        float g0, g1, g2, g3;
        pf_gaussian_pair_cpu(h_rng, g0, g1);
        pf_gaussian_pair_cpu(h_rng, g2, g3);       // g3 unused — half a pair, cheap and simple
        h_px[static_cast<size_t>(k)]  = lg.init_x + kInitSigmaPos * g0;
        h_py[static_cast<size_t>(k)]  = lg.init_y + kInitSigmaPos * g1;
        h_pth[static_cast<size_t>(k)] = lg.init_th + kInitSigmaTh * g2;
    }
    const std::vector<float> h_px0 = h_px, h_py0 = h_py, h_pth0 = h_pth;   // pristine copies for the two runs

    // ---- persistent device buffers -------------------------------------------
    // Allocated ONCE — a 10 Hz filter that reallocates every scan spends its
    // budget in the allocator, not the kernels (same discipline as 08.01).
    const size_t kb = static_cast<size_t>(K) * sizeof(float);
    float *d_px = nullptr, *d_py = nullptr, *d_pth = nullptr, *d_logw = nullptr, *d_scan = nullptr;
    unsigned char* d_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_px, kb));
    CUDA_CHECK(cudaMalloc(&d_py, kb));
    CUDA_CHECK(cudaMalloc(&d_pth, kb));
    CUDA_CHECK(cudaMalloc(&d_logw, kb));
    CUDA_CHECK(cudaMalloc(&d_scan, kNumBeams * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map, map.occ.size()));
    CUDA_CHECK(cudaMemcpy(d_map, map.occ.data(), map.occ.size(), cudaMemcpyHostToDevice));

    // ======================= VERIFY STAGE ====================================
    // Step 0's exact inputs through both paths (the §5 gate), kernel by
    // kernel — the file header explains why the weight twins get IDENTICAL
    // poses instead of chaining through predict.
    //
    // Tolerances, justified:
    //   predict — ABSOLUTE 1e-4 (m / rad): one smooth pose update differs
    //     only by trig-libm ulps and FMA contraction, ~1e-6 at these
    //     magnitudes; 1e-4 is ~100x headroom while a seed/model bug shifts
    //     poses at order 0.1.
    //   weight — RELATIVE 1e-3 (floor 1.0): on identical poses the
    //     contraction-safe ray-march visits identical cells, so log-weights
    //     differ only by the final accumulation's rounding (~1e-6 relative);
    //     an indexing/layout/model bug moves them at order 1.
    {
        CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), kb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), kb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pth, h_pth.data(), kb, cudaMemcpyHostToDevice));

        // -- predict twins ----------------------------------------------------
        launch_pf_predict(K, /*step=*/0, lg.odo_v[0], lg.odo_w[0], d_px, d_py, d_pth);
        std::vector<float> gx(static_cast<size_t>(K)), gy(static_cast<size_t>(K)),
                           gth(static_cast<size_t>(K));
        CUDA_CHECK(cudaMemcpy(gx.data(), d_px, kb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gy.data(), d_py, kb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gth.data(), d_pth, kb, cudaMemcpyDeviceToHost));

        std::vector<float> cx = h_px0, cy = h_py0, cth = h_pth0;   // CPU path from the same cloud
        pf_predict_cpu(K, 0, lg.odo_v[0], lg.odo_w[0], cx.data(), cy.data(), cth.data());

        float worst_pred = 0.0f;
        for (int k = 0; k < K; ++k) {
            const size_t i = static_cast<size_t>(k);
            worst_pred = std::fmax(worst_pred, std::fabs(gx[i] - cx[i]));
            worst_pred = std::fmax(worst_pred, std::fabs(gy[i] - cy[i]));
            worst_pred = std::fmax(worst_pred, std::fabs(gth[i] - cth[i]));
        }
        const bool predict_pass = worst_pred <= 1e-4f;
        std::printf("[info] verify predict: worst absolute pose deviation %.3e (m or rad) over %d particles\n",
                    static_cast<double>(worst_pred), K);

        // -- weight twins, on the SAME (GPU-predicted) poses --------------------
        CUDA_CHECK(cudaMemcpy(d_scan, &lg.scan[0], kNumBeams * sizeof(float), cudaMemcpyHostToDevice));
        GpuTimer gt;
        gt.begin();
        launch_pf_weight(K, d_px, d_py, d_pth, d_map, map.w, map.h, map.inv_res, d_scan, d_logw);
        const float gpu_ms = gt.end_ms();
        std::vector<float> lw_gpu(static_cast<size_t>(K));
        CUDA_CHECK(cudaMemcpy(lw_gpu.data(), d_logw, kb, cudaMemcpyDeviceToHost));

        std::vector<float> lw_cpu(static_cast<size_t>(K));
        CpuTimer ct;
        ct.begin();
        pf_weight_cpu(K, gx.data(), gy.data(), gth.data(),      // identical inputs: the GPU's poses
                      map.occ.data(), map.w, map.h, map.inv_res, &lg.scan[0], lw_cpu.data());
        const double cpu_ms = ct.end_ms();

        bool weight_pass = true;
        float worst_w = 0.0f;
        for (int k = 0; k < K; ++k) {
            const size_t i = static_cast<size_t>(k);
            const float scale = std::fabs(lw_cpu[i]) > 1.0f ? std::fabs(lw_cpu[i]) : 1.0f;
            const float d = std::fabs(lw_gpu[i] - lw_cpu[i]) / scale;
            if (d > worst_w) worst_w = d;
            if (d > 1e-3f) weight_pass = false;
        }
        std::printf("[info] verify weight: worst relative log-likelihood deviation %.3e over %d particles\n",
                    static_cast<double>(worst_w), K);
        std::printf("[time] weight kernel (K=%d, B=%d): CPU %.1f ms | GPU %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                    K, kNumBeams, cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        std::printf("VERIFY: %s (GPU predict/weight match CPU reference within documented tolerances)\n",
                    (predict_pass && weight_pass) ? "PASS" : "FAIL");
        if (!predict_pass || !weight_pass) {
            std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting the filter)\n");
            return 1;
        }
    }

    // ======================= CLOSED LOOP =====================================
    // Restart from the pristine initial cloud so the loop is a clean,
    // self-contained run (the verify stage above consumed step 0 once).
    h_px = h_px0; h_py = h_py0; h_pth = h_pth0;
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), kb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), kb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pth, h_pth.data(), kb, cudaMemcpyHostToDevice));

    std::vector<float> lw(static_cast<size_t>(K));    // downloaded log-weights
    std::vector<double> w(static_cast<size_t>(K));    // normalized-enough weights (see step 3)
    std::vector<float> est;                           // logged rows for the artifact
    est.reserve(static_cast<size_t>(lg.steps) * 4);

    double sq_err_sum = 0.0;      // running sum of squared position errors (m^2)
    double sq_herr_sum = 0.0;     // running sum of squared heading errors (rad^2)
    double ess_min = 1e300;       // worst effective sample size seen
    double ess_sum = 0.0;         // for the mean ESS [info] line
    double loop_gpu_ms = 0.0;     // accumulated predict+weight kernel time

    for (int t = 0; t < lg.steps; ++t) {
        // (1) GPU: push the cloud through the odometry, then score it
        // against this step's scan. One timer brackets both kernels — this
        // pair is the whole per-scan GPU cost of the filter.
        CUDA_CHECK(cudaMemcpy(d_scan, &lg.scan[static_cast<size_t>(t) * kNumBeams],
                              kNumBeams * sizeof(float), cudaMemcpyHostToDevice));
        GpuTimer gt;
        gt.begin();
        launch_pf_predict(K, t, lg.odo_v[static_cast<size_t>(t)], lg.odo_w[static_cast<size_t>(t)],
                          d_px, d_py, d_pth);
        launch_pf_weight(K, d_px, d_py, d_pth, d_map, map.w, map.h, map.inv_res, d_scan, d_logw);
        loop_gpu_ms += static_cast<double>(gt.end_ms());

        // (2) download the cloud + its scores. ~1.6 MB per step at K=1e5 —
        // the didactic price of host-side estimate/resample; Exercise 5
        // moves both onto the GPU and deletes these copies.
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, kb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py.data(), d_py, kb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pth.data(), d_pth, kb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lw.data(), d_logw, kb, cudaMemcpyDeviceToHost));

        // (3) normalize in log space: subtract the max BEFORE exp so the
        // best particle maps to exp(0)=1 and nothing overflows/underflows
        // (16 beams of squared error reach exp(-hundreds) raw). Double
        // accumulators keep 100k tiny weights from vanishing — the same
        // hygiene as 08.01's softmin.
        float lw_max = lw[0];
        for (int k = 1; k < K; ++k) if (lw[static_cast<size_t>(k)] > lw_max) lw_max = lw[static_cast<size_t>(k)];
        double w_sum = 0.0, w_sq_sum = 0.0;
        for (int k = 0; k < K; ++k) {
            const double wk = std::exp(static_cast<double>(lw[static_cast<size_t>(k)] - lw_max));
            w[static_cast<size_t>(k)] = wk;
            w_sum += wk;
            w_sq_sum += wk * wk;
        }
        const double ess = (w_sum * w_sum) / (w_sq_sum > 0.0 ? w_sq_sum : 1.0);  // effective sample size
        if (ess < ess_min) ess_min = ess;
        ess_sum += ess;

        // (4) estimate = weighted mean of the cloud. Heading is CIRCULAR:
        // averaging raw angles fails at the +-pi seam, so average the unit
        // vectors (sin, cos) and take atan2 — the standard circular mean.
        double mx = 0.0, my = 0.0, ms = 0.0, mc = 0.0;
        for (int k = 0; k < K; ++k) {
            const size_t i = static_cast<size_t>(k);
            mx += w[i] * static_cast<double>(h_px[i]);
            my += w[i] * static_cast<double>(h_py[i]);
            ms += w[i] * std::sin(static_cast<double>(h_pth[i]));
            mc += w[i] * std::cos(static_cast<double>(h_pth[i]));
        }
        const float est_x = static_cast<float>(mx / w_sum);
        const float est_y = static_cast<float>(my / w_sum);
        const float est_th = static_cast<float>(std::atan2(ms, mc));   // in (-pi, pi]

        // (5) error vs ground truth (the generator wrote the truth down —
        // the synthetic-data superpower). Heading error goes through the
        // project's single wrap point.
        const size_t ti = static_cast<size_t>(t);
        const float ex = est_x - lg.gt_x[ti];
        const float ey = est_y - lg.gt_y[ti];
        const float eth = wrap_angle(est_th - lg.gt_th[ti]);
        sq_err_sum += static_cast<double>(ex) * ex + static_cast<double>(ey) * ey;
        sq_herr_sum += static_cast<double>(eth) * eth;

        est.push_back(est_x); est.push_back(est_y); est.push_back(est_th);
        est.push_back(std::sqrt(ex * ex + ey * ey));

        // (6) systematic resample (host, O(K)) and re-upload the fresh
        // cloud. Resampling AFTER the estimate: the weighted cloud is the
        // posterior; resampling only re-expresses it with uniform weights.
        systematic_resample(K, w, w_sum, h_rng, h_px, h_py, h_pth);
        CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), kb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), kb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pth, h_pth.data(), kb, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_pth));
    CUDA_CHECK(cudaFree(d_logw));
    CUDA_CHECK(cudaFree(d_scan));
    CUDA_CHECK(cudaFree(d_map));

    const double rmse = std::sqrt(sq_err_sum / lg.steps);
    const double h_rmse = std::sqrt(sq_herr_sum / lg.steps);
    std::printf("[info] closed loop: position RMSE %.4f m, heading RMSE %.4f rad, final position error %.4f m\n",
                rmse, h_rmse, static_cast<double>(est[static_cast<size_t>(lg.steps - 1) * 4 + 3]));
    std::printf("[info] effective sample size: min %.0f, mean %.0f of K=%d (resampled every step)\n",
                ess_min, ess_sum / lg.steps, K);
    std::printf("[time] closed loop: %.2f ms average GPU (predict+weight) per step over %d steps\n",
                loop_gpu_ms / lg.steps, lg.steps);

    // ---- artifact: estimate vs truth, plottable with anything -----------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/trajectory_est.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "step,t_s,gt_x_m,gt_y_m,gt_theta_rad,est_x_m,est_y_m,est_theta_rad,err_pos_m\n";  // units in the header (§12)
            for (int t = 0; t < lg.steps; ++t) {
                const size_t ti = static_cast<size_t>(t);
                const float* r = &est[ti * 4];
                f << t << ',' << (t + 1) * kDt << ','
                  << lg.gt_x[ti] << ',' << lg.gt_y[ti] << ',' << lg.gt_th[ti] << ','
                  << r[0] << ',' << r[1] << ',' << r[2] << ',' << r[3] << '\n';
            }
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/trajectory_est.csv (%d steps)\n", lg.steps);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/trajectory_est.csv\n");

    // ---- success check (the stable verdict) -----------------------------------
    // The filter must TRACK: position RMSE against ground truth below
    // kRmseGateM over the whole run. Measured ~0.03 m on the reference
    // machine — the 0.15 m gate is a wide margin on purpose, so platform
    // ulp differences rippling through resampling cannot flip the verdict
    // (see the determinism note in the file header).
    const bool success = artifact_ok && (rmse < static_cast<double>(kRmseGateM));
    if (success)
        std::printf("RESULT: PASS (closed-loop position RMSE < 0.15 m vs ground truth)\n");
    else
        std::printf("RESULT: FAIL (RMSE gate or artifact failed — see [info] lines)\n");
    return success ? 0 : 1;
}
