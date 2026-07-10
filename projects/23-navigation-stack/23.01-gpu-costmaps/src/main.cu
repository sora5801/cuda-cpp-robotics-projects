// ===========================================================================
// main.cu — entry point for project 23.01
//           GPU costmaps: inflation, raytrace clearing, multi-layer fusion
//           + a DWA local-planner consumer, closed loop
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed synthetic world
//      (data/sample/world_map.pgm) and scenario (data/sample/scenario.csv).
//   2. VERIFY STAGE (the §5 GPU-vs-CPU gate, TWO independent checks):
//        a. one full costmap update cycle (tick 0's scan) — GPU master
//           costmap vs CPU oracle, BYTE-EXACT (every layer is pure integer
//           arithmetic — kernels.cu explains why this is possible here).
//        b. one DWA scoring pass over the tick-0 dynamic window — GPU vs
//           CPU, within a documented relative tolerance (trig-heavy, like
//           08.01's rollout costs).
//   3. CLOSED LOOP: sense (simulated LiDAR) -> GPU costmap update -> GPU
//      DWA scoring -> pick the best admissible (v,w) -> drive the plant one
//      control tick -> repeat, until the goal is reached or the scenario's
//      step cap is hit. Logs the driven path to demo/out/path.csv and the
//      final costmap to demo/out/costmap.pgm.
//   4. SUCCESS CHECK: exit 0 only if both verify stages passed, the goal
//      was reached within the step cap, and the driven path never entered
//      a lethal-cost cell.
//
// Message-shape correspondence (SYSTEM_DESIGN.md §3.6) — named here once so
// every "scan"/"pose"/"costmap" below reads as a real interface, not an ad
// hoc array: the LiDAR scan this file simulates and discretizes corresponds
// to `sensor_msgs/LaserScan`; world_map/master_costmap correspond to
// `nav_msgs/OccupancyGrid`; the chosen (v,w) applied to the plant
// corresponds to `geometry_msgs/Twist` (linear.x = v, angular.z = w).
//
// Determinism: the world, the scan (DDA against a fixed map), and the plant
// are all deterministic given the committed seed-42 map and scenario — no
// RNG anywhere in this file. The whole closed loop is therefore
// bit-reproducible on one machine; the stable output lines below still
// avoid embedding trajectory FLOATS that could drift a ULP across
// architectures (THEORY.md §numerics), reporting counts and PASS/FAIL
// instead, with wide margins.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "MAP:", "SCENARIO:",
// "VERIFY COSTMAP:", "VERIFY DWA:", "ARTIFACT:", "RESULT:" — "[info]"/
// "[time]" unchecked. Change a stable line => update demo/expected_output.txt
// in the same change.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cctype>
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
// Path helpers — same exe-relative resolution as every other project's demo
// (the exe sits at build/x64/<Config>/, three levels below the project root).
// ---------------------------------------------------------------------------
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string dir_of(const std::string& path)
{
    size_t cut = path.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return path.substr(0, cut);
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
// read_pgm — a STRICT P5 (binary grayscale) reader for data/sample/world_map.pgm.
//
// The format is the smallest real image format there is (07.09 writes this
// same format for its artifacts): a three-line ASCII header ("P5",
// "<width> <height>", "<maxval>"), each possibly preceded by '#' comment
// lines, then exactly width*height raw bytes. Strict like every loader in
// this repo (CLAUDE.md §9): wrong magic, dimensions that disagree with the
// kernels.cuh contract, a maxval other than 255, or a short/truncated body
// all fail loudly rather than silently producing a wrong map.
// ---------------------------------------------------------------------------
static bool read_pgm_token(std::istream& in, std::string& tok)
{
    // Skip whitespace and '#'-prefixed comment lines (the PGM "plain
    // header" convention), then read one whitespace-delimited token.
    int c;
    for (;;) {
        c = in.get();
        if (c == EOF) return false;
        if (c == '#') { while (c != '\n' && c != EOF) c = in.get(); continue; }
        if (std::isspace(c)) continue;
        break;
    }
    tok.clear();
    tok.push_back(static_cast<char>(c));
    while (in.peek() != EOF && !std::isspace(in.peek())) tok.push_back(static_cast<char>(in.get()));
    return true;
}

static bool read_pgm(const std::string& path, std::vector<unsigned char>& out,
                     int expect_w, int expect_h)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[2] = { 0, 0 };
    magic[0] = static_cast<char>(in.get());
    magic[1] = static_cast<char>(in.get());
    if (magic[0] != 'P' || magic[1] != '5') {
        std::fprintf(stderr, "world map: not a P5 PGM (bad magic)\n");
        return false;
    }

    std::string tw, th, tmax;
    if (!read_pgm_token(in, tw) || !read_pgm_token(in, th) || !read_pgm_token(in, tmax)) {
        std::fprintf(stderr, "world map: truncated PGM header\n");
        return false;
    }
    const int w = std::atoi(tw.c_str());
    const int h = std::atoi(th.c_str());
    const int maxval = std::atoi(tmax.c_str());
    if (w != expect_w || h != expect_h) {
        std::fprintf(stderr, "world map: size %dx%d does not match the %dx%d contract in kernels.cuh\n",
                     w, h, expect_w, expect_h);
        return false;
    }
    if (maxval != 255) {
        std::fprintf(stderr, "world map: maxval %d != 255 (this reader only accepts 8-bit PGM)\n", maxval);
        return false;
    }
    // Exactly ONE whitespace byte separates the header from the raw binary
    // body per the PGM spec; read_pgm_token already consumed it via peek/get
    // discipline up to (not including) the first body byte, so the stream
    // cursor is already correctly positioned — read the body directly.
    out.resize(static_cast<size_t>(w) * h);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!in) {
        std::fprintf(stderr, "world map: truncated pixel data (expected %d bytes)\n", w * h);
        return false;
    }
    return true;
}

static bool write_pgm(const std::string& path, int width, int height,
                      const std::vector<unsigned char>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()),
              static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// Scenario loading — "MAP,<file>,<w>,<h>,<res>" / "START,<x>,<y>,<theta>" /
// "GOAL,<x>,<y>" / "STEPS,<n>". Strict like 08.01/07.09's loaders: an
// unrecognized row label, a wrong field count, or a MAP row whose w/h/res
// disagree with the kernels.cuh contract all abort with a clear message.
// ---------------------------------------------------------------------------
struct Scenario {
    std::string map_file;
    float start_x = 0.0f, start_y = 0.0f, start_theta = 0.0f;
    float goal_x = 0.0f, goal_y = 0.0f;
    int steps = 0;
    bool loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_map = false, have_start = false, have_goal = false, have_steps = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "MAP") {
            std::vector<std::string> f;
            while (std::getline(ss, cell, ',')) f.push_back(cell);
            if (f.size() != 4) { std::fprintf(stderr, "scenario: MAP row needs 4 fields\n"); return Scenario{}; }
            sc.map_file = f[0];
            const int w = std::atoi(f[1].c_str());
            const int h = std::atoi(f[2].c_str());
            const float res = std::strtof(f[3].c_str(), nullptr);
            if (w != kGridW || h != kGridH || std::fabs(res - kResolutionM) > 1e-6f) {
                std::fprintf(stderr, "scenario: MAP row %dx%d @ %.4f disagrees with the kernels.cuh "
                                     "contract (%dx%d @ %.4f)\n", w, h, static_cast<double>(res),
                             kGridW, kGridH, static_cast<double>(kResolutionM));
                return Scenario{};
            }
            have_map = true;
        } else if (label == "START") {
            float v[3];
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short START row\n"); return Scenario{}; }
                v[i] = std::strtof(cell.c_str(), nullptr);
            }
            sc.start_x = v[0]; sc.start_y = v[1]; sc.start_theta = v[2];
            have_start = true;
        } else if (label == "GOAL") {
            float v[2];
            for (int i = 0; i < 2; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short GOAL row\n"); return Scenario{}; }
                v[i] = std::strtof(cell.c_str(), nullptr);
            }
            sc.goal_x = v[0]; sc.goal_y = v[1];
            have_goal = true;
        } else if (label == "STEPS") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short STEPS row\n"); return Scenario{}; }
            sc.steps = std::atoi(cell.c_str());
            have_steps = true;
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_map || !have_start || !have_goal || !have_steps || sc.steps < 1) {
        std::fprintf(stderr, "scenario: missing MAP/START/GOAL/STEPS row\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

static std::string find_scenario(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/scenario.csv");
    candidates.push_back("data/sample/scenario.csv");
    candidates.push_back("../data/sample/scenario.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// ---------------------------------------------------------------------------
// simulate_lidar_scan — THE PLANT'S SENSOR. Host-only, deterministic:
// kNumBeams beams spread evenly over the full circle, each DDA-stepped in
// small METRIC increments against the TRUE world map (never the costmap —
// see kernels.cuh's file header for why keeping "ground truth" and "what
// the costmap pipeline sees" strictly separate matters). The OUTPUT of this
// function is what actually crosses into the GPU/CPU costmap pipeline: not
// the true map, not even the continuous ranges, but a PRE-DISCRETIZED
// integer endpoint cell per beam (kernels.cu explains why this exactness
// matters for the byte-exact verify gate).
//
// Step size: half a cell (kResolutionM/2) — fine enough that a beam cannot
// tunnel through the map's thinnest wall (2 cells) between samples, coarse
// enough that 360 beams x ~240 steps is a trivial amount of host work,
// timed separately from (and excluded from) the GPU kernel timings below.
// ---------------------------------------------------------------------------
static void simulate_lidar_scan(const std::vector<unsigned char>& world,
                                float robot_x_m, float robot_y_m,
                                int end_ix[kNumBeams], int end_iy[kNumBeams], unsigned char hit[kNumBeams])
{
    const float step_m = kResolutionM * 0.5f;
    const float two_pi = 6.283185307179586f;

    for (int b = 0; b < kNumBeams; ++b) {
        const float angle = (two_pi * static_cast<float>(b)) / static_cast<float>(kNumBeams);
        const float ca = std::cos(angle), sa = std::sin(angle);

        int last_ix = static_cast<int>(std::floor(robot_x_m / kResolutionM));
        int last_iy = static_cast<int>(std::floor(robot_y_m / kResolutionM));
        bool found_hit = false;

        const int n_steps = static_cast<int>(kMaxRangeM / step_m);
        for (int s = 1; s <= n_steps; ++s) {
            const float r = step_m * static_cast<float>(s);
            const float x = robot_x_m + r * ca;
            const float y = robot_y_m + r * sa;
            int ix = static_cast<int>(std::floor(x / kResolutionM));
            int iy = static_cast<int>(std::floor(y / kResolutionM));

            if (ix < 0 || ix >= kGridW || iy < 0 || iy >= kGridH) {
                // Left the mapped area without hitting anything (should not
                // happen inside the committed map's walled perimeter — a
                // defensive clamp, not the expected path).
                ix = ix < 0 ? 0 : (ix >= kGridW ? kGridW - 1 : ix);
                iy = iy < 0 ? 0 : (iy >= kGridH ? kGridH - 1 : iy);
                last_ix = ix; last_iy = iy;
                found_hit = true;   // treat the map edge as a wall — safe default
                break;
            }

            last_ix = ix; last_iy = iy;
            if (world[static_cast<size_t>(iy) * kGridW + ix] == kCostLethal) {
                found_hit = true;
                break;
            }
        }

        end_ix[b] = last_ix;
        end_iy[b] = last_iy;
        hit[b] = found_hit ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// dynamic_window — this tick's admissible (v,w) sampling bounds: whatever
// the robot can reach from (v_prev, w_prev) within one control period given
// its acceleration limits, clamped to the robot's absolute limits. THE
// "Dynamic" in Dynamic Window Approach (THEORY.md §the-math derives it).
// ---------------------------------------------------------------------------
static void dynamic_window(float v_prev, float w_prev,
                           float& v_lo, float& v_hi, float& w_lo, float& w_hi)
{
    const float dv = kAccelV * kDtControl;
    const float dw = kAccelW * kDtControl;
    v_lo = std::max(kVMin, v_prev - dv);
    v_hi = std::min(kVMax, v_prev + dv);
    w_lo = std::max(-kWMax, w_prev - dw);
    w_hi = std::min(kWMax, w_prev + dw);
}

// ---------------------------------------------------------------------------
// argmin_admissible — the host-side reduction over this tick's 4096 DWA
// scores (deliberately kept in plain C++ next to the algorithm, the same
// "keep the blend visible" choice 08.01 makes for its softmin). Returns the
// winning sample index, or -1 if every sample was inadmissible (the
// emergency-brake trigger in the closed loop below).
// ---------------------------------------------------------------------------
static int argmin_admissible(const std::vector<float>& scores)
{
    int best = -1;
    float best_score = kInadmissibleScore;
    for (int k = 0; k < kNumDwaSamples; ++k) {
        if (scores[static_cast<size_t>(k)] < best_score) {
            best_score = scores[static_cast<size_t>(k)];
            best = k;
        }
    }
    return (best_score < kInadmissibleScore) ? best : -1;
}

static void sample_to_vw(int k, float v_lo, float v_hi, float w_lo, float w_hi, float& v, float& w)
{
    const int vi = k / kWSamples;
    const int wi = k % kWSamples;
    v = (kVSamples > 1) ? v_lo + (v_hi - v_lo) * (static_cast<float>(vi) / (kVSamples - 1)) : v_lo;
    w = (kWSamples > 1) ? w_lo + (w_hi - w_lo) * (static_cast<float>(wi) / (kWSamples - 1)) : w_lo;
}

// ---------------------------------------------------------------------------
// main — verify stage, then the closed loop described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] GPU costmaps + DWA local planner: warehouse AMR navigation loop (project 23.01)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d grid @ %.2f m/cell (%.1f x %.1f m), %d-beam LiDAR (max range %.1f m), "
                "DWA %dx%d=%d (v,w) samples, FP32\n",
                kGridW, kGridH, static_cast<double>(kResolutionM),
                static_cast<double>(kGridW * kResolutionM), static_cast<double>(kGridH * kResolutionM),
                kNumBeams, static_cast<double>(kMaxRangeM), kVSamples, kWSamples, kNumDwaSamples);

    // ---- scenario + map ------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    const std::string map_path = dir_of(scenario_path) + "/" + sc.map_file;
    std::printf("[info] map file: %s\n", map_path.c_str());
    std::vector<unsigned char> world;   // the TRUE world — only simulate_lidar_scan ever reads this
    if (!read_pgm(map_path, world, kGridW, kGridH)) {
        std::printf("MAP: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (map malformed)\n");
        return 1;
    }
    long long lethal_cells = 0;
    for (unsigned char c : world) if (c == kCostLethal) ++lethal_cells;
    std::printf("MAP: %dx%d cells, %lld lethal cells (%.1f%% occupied) [synthetic, seed 42]\n",
                kGridW, kGridH, lethal_cells, 100.0 * static_cast<double>(lethal_cells) / kGridTotal);
    std::printf("SCENARIO: start (%.2f, %.2f, %.2f) -> goal (%.2f, %.2f), step cap %d [synthetic, seed 42]\n",
                static_cast<double>(sc.start_x), static_cast<double>(sc.start_y), static_cast<double>(sc.start_theta),
                static_cast<double>(sc.goal_x), static_cast<double>(sc.goal_y), sc.steps);

    const float mission_dist = std::max(0.1f,
        std::sqrt((sc.goal_x - sc.start_x) * (sc.goal_x - sc.start_x) +
                 (sc.goal_y - sc.start_y) * (sc.goal_y - sc.start_y)));

    // ---- persistent device buffers --------------------------------------------
    // Allocated ONCE, outside the control loop (same reasoning as 08.01): the
    // per-tick scan upload inside launch_costmap_update is the deliberate,
    // negligible exception (kernels.cu's launcher comment explains why).
    unsigned char* d_static    = nullptr;
    int*           d_obstacle  = nullptr;
    unsigned char* d_inflation = nullptr;
    unsigned char* d_master    = nullptr;
    float*         d_scores    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_static,    static_cast<size_t>(kGridTotal) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_obstacle,  static_cast<size_t>(kGridTotal) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inflation, static_cast<size_t>(kGridTotal) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_master,    static_cast<size_t>(kGridTotal) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_scores,    static_cast<size_t>(kNumDwaSamples) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_static, world.data(), static_cast<size_t>(kGridTotal) * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    int end_ix[kNumBeams], end_iy[kNumBeams];
    unsigned char hit[kNumBeams];

    // ======================= VERIFY STAGE ====================================
    {
        // (a) One full costmap update cycle, tick 0's exact inputs.
        const int robot_ix = static_cast<int>(std::floor(sc.start_x / kResolutionM));
        const int robot_iy = static_cast<int>(std::floor(sc.start_y / kResolutionM));
        simulate_lidar_scan(world, sc.start_x, sc.start_y, end_ix, end_iy, hit);

        GpuTimer gt;
        gt.begin();
        launch_costmap_update(robot_ix, robot_iy, end_ix, end_iy, hit,
                              d_static, d_obstacle, d_inflation, d_master);
        const float gpu_costmap_ms = gt.end_ms();

        std::vector<unsigned char> gpu_master(static_cast<size_t>(kGridTotal));
        CUDA_CHECK(cudaMemcpy(gpu_master.data(), d_master,
                              static_cast<size_t>(kGridTotal) * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        std::vector<int>           cpu_obstacle(static_cast<size_t>(kGridTotal));
        std::vector<unsigned char> cpu_inflation(static_cast<size_t>(kGridTotal));
        std::vector<unsigned char> cpu_master(static_cast<size_t>(kGridTotal));
        CpuTimer ct;
        ct.begin();
        costmap_update_cpu(robot_ix, robot_iy, end_ix, end_iy, hit,
                           world.data(), cpu_obstacle.data(), cpu_inflation.data(), cpu_master.data());
        const double cpu_costmap_ms = ct.end_ms();

        long long mismatches = 0;
        for (int i = 0; i < kGridTotal; ++i)
            if (gpu_master[static_cast<size_t>(i)] != cpu_master[static_cast<size_t>(i)]) ++mismatches;

        std::printf("[info] verify costmap: %lld/%d cells differ (byte-exact target: 0)\n",
                    mismatches, kGridTotal);
        std::printf("[time] costmap update: CPU %.2f ms | GPU (3 kernels) %.3f ms | speed-up %.0fx "
                    "(teaching artifact)\n", cpu_costmap_ms, static_cast<double>(gpu_costmap_ms),
                    cpu_costmap_ms / (static_cast<double>(gpu_costmap_ms) > 0.0 ? static_cast<double>(gpu_costmap_ms) : 1.0));
        const bool costmap_pass = (mismatches == 0);
        std::printf("VERIFY COSTMAP: %s (GPU master costmap byte-exact vs CPU reference over one full update cycle)\n",
                    costmap_pass ? "PASS" : "FAIL");
        if (!costmap_pass) {
            std::printf("RESULT: FAIL (GPU/CPU costmap disagreement — fix before trusting the planner)\n");
            return 1;
        }

        // (b) One DWA scoring pass over the tick-0 dynamic window.
        float v_lo, v_hi, w_lo, w_hi;
        dynamic_window(0.0f, 0.0f, v_lo, v_hi, w_lo, w_hi);

        std::vector<float> gpu_scores(static_cast<size_t>(kNumDwaSamples));
        GpuTimer gt2;
        gt2.begin();
        launch_dwa_scores(d_master, sc.start_x, sc.start_y, sc.start_theta, sc.goal_x, sc.goal_y,
                          v_lo, v_hi, w_lo, w_hi, mission_dist, d_scores);
        const float gpu_dwa_ms = gt2.end_ms();
        CUDA_CHECK(cudaMemcpy(gpu_scores.data(), d_scores,
                              static_cast<size_t>(kNumDwaSamples) * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> cpu_scores(static_cast<size_t>(kNumDwaSamples));
        CpuTimer ct2;
        ct2.begin();
        dwa_scores_cpu(cpu_master.data(), sc.start_x, sc.start_y, sc.start_theta, sc.goal_x, sc.goal_y,
                       v_lo, v_hi, w_lo, w_hi, mission_dist, cpu_scores.data());
        const double cpu_dwa_ms = ct2.end_ms();

        float worst = 0.0f;
        bool dwa_pass = true;
        for (int k = 0; k < kNumDwaSamples; ++k) {
            const float cv = cpu_scores[static_cast<size_t>(k)];
            const float scale = std::fabs(cv) > 1.0f ? std::fabs(cv) : 1.0f;
            const float d = std::fabs(gpu_scores[static_cast<size_t>(k)] - cv) / scale;
            if (d > worst) worst = d;
            if (d > 1e-3f) dwa_pass = false;
        }
        std::printf("[info] verify dwa: worst relative score deviation %.3e over %d samples\n",
                    static_cast<double>(worst), kNumDwaSamples);
        std::printf("[time] dwa scoring: CPU %.2f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact)\n",
                    cpu_dwa_ms, static_cast<double>(gpu_dwa_ms),
                    cpu_dwa_ms / (static_cast<double>(gpu_dwa_ms) > 0.0 ? static_cast<double>(gpu_dwa_ms) : 1.0));
        std::printf("VERIFY DWA: %s (GPU DWA scores match CPU reference within rel tol 1e-3)\n",
                    dwa_pass ? "PASS" : "FAIL");
        if (!dwa_pass) {
            std::printf("RESULT: FAIL (GPU/CPU DWA scoring disagreement — fix before trusting the planner)\n");
            return 1;
        }
    }

    // ======================= CLOSED LOOP =====================================
    float pose[3] = { sc.start_x, sc.start_y, sc.start_theta };   // the PLANT state ("reality")
    float v_prev = 0.0f, w_prev = 0.0f;

    std::vector<float> traj;                 // logged rows: t,x,y,theta,v,w
    traj.reserve(static_cast<size_t>(sc.steps + 1) * 6);
    traj.push_back(0.0f); traj.push_back(pose[0]); traj.push_back(pose[1]);
    traj.push_back(pose[2]); traj.push_back(0.0f); traj.push_back(0.0f);

    std::vector<unsigned char> master_host(static_cast<size_t>(kGridTotal));
    std::vector<float> scores(static_cast<size_t>(kNumDwaSamples));

    int steps_taken = 0;
    long long lethal_hits = 0;
    long long emergency_brakes = 0;
    bool goal_reached = false;
    double loop_gpu_costmap_ms = 0.0, loop_gpu_dwa_ms = 0.0;

    for (int step = 0; step < sc.steps; ++step) {
        // (1) sense: simulate this tick's LiDAR scan from the CURRENT pose.
        const int robot_ix = static_cast<int>(std::floor(pose[0] / kResolutionM));
        const int robot_iy = static_cast<int>(std::floor(pose[1] / kResolutionM));
        simulate_lidar_scan(world, pose[0], pose[1], end_ix, end_iy, hit);

        // (2) GPU: costmap pipeline (raytrace -> inflation -> fusion).
        GpuTimer gt;
        gt.begin();
        launch_costmap_update(robot_ix, robot_iy, end_ix, end_iy, hit,
                              d_static, d_obstacle, d_inflation, d_master);
        loop_gpu_costmap_ms += static_cast<double>(gt.end_ms());
        CUDA_CHECK(cudaMemcpy(master_host.data(), d_master,
                              static_cast<size_t>(kGridTotal) * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        // (3) the dynamic window for this tick.
        float v_lo, v_hi, w_lo, w_hi;
        dynamic_window(v_prev, w_prev, v_lo, v_hi, w_lo, w_hi);

        // (4) GPU: score every (v,w) sample against this tick's master costmap.
        GpuTimer gt2;
        gt2.begin();
        launch_dwa_scores(d_master, pose[0], pose[1], pose[2], sc.goal_x, sc.goal_y,
                          v_lo, v_hi, w_lo, w_hi, mission_dist, d_scores);
        loop_gpu_dwa_ms += static_cast<double>(gt2.end_ms());
        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores,
                              static_cast<size_t>(kNumDwaSamples) * sizeof(float), cudaMemcpyDeviceToHost));

        // (5) pick the best admissible sample; brake if none exists (the
        // standard DWA safety fallback — should be rare-to-never on this
        // scenario's deliberately generous course, measured honestly below).
        float v_cmd, w_cmd;
        const int best = argmin_admissible(scores);
        if (best >= 0) {
            sample_to_vw(best, v_lo, v_hi, w_lo, w_hi, v_cmd, w_cmd);
        } else {
            v_cmd = std::max(kVMin, v_prev - kAccelV * kDtControl);
            w_cmd = 0.0f;
            ++emergency_brakes;
        }

        // (6) act: drive the plant one control tick.
        diffdrive_step_cpu(pose, v_cmd, w_cmd, kDtControl);
        v_prev = v_cmd; w_prev = w_cmd;
        ++steps_taken;

        // (7) safety re-check: did the ACTUAL driven cell end up lethal?
        // (kDtControl == kDtSub by construction — see kernels.cuh — so this
        // is re-checking the very first substep of the arc DWA already
        // verified admissible; done anyway, on the real driven pose, as an
        // independent assertion rather than trusting the planner's own flag.)
        {
            const int ix = static_cast<int>(std::floor(pose[0] / kResolutionM));
            const int iy = static_cast<int>(std::floor(pose[1] / kResolutionM));
            if (ix < 0 || ix >= kGridW || iy < 0 || iy >= kGridH ||
                master_host[static_cast<size_t>(iy) * kGridW + ix] >= kCostLethal) {
                ++lethal_hits;
            }
        }

        // (8) log + goal check.
        traj.push_back(static_cast<float>(steps_taken) * kDtControl);
        traj.push_back(pose[0]); traj.push_back(pose[1]); traj.push_back(pose[2]);
        traj.push_back(v_cmd); traj.push_back(w_cmd);

        const float dgx = sc.goal_x - pose[0], dgy = sc.goal_y - pose[1];
        if (std::sqrt(dgx * dgx + dgy * dgy) < kGoalTolM) { goal_reached = true; break; }
    }

    CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_obstacle));
    CUDA_CHECK(cudaFree(d_inflation));
    CUDA_CHECK(cudaFree(d_master));
    CUDA_CHECK(cudaFree(d_scores));

    std::printf("[info] final pose: x=%.3f m, y=%.3f m, theta=%.3f rad; steps used: %d/%d; "
                "emergency brakes: %lld\n",
                static_cast<double>(pose[0]), static_cast<double>(pose[1]), static_cast<double>(pose[2]),
                steps_taken, sc.steps, emergency_brakes);
    std::printf("[time] closed loop: costmap %.3f ms/tick, DWA %.3f ms/tick average GPU kernel time over %d ticks\n",
                loop_gpu_costmap_ms / steps_taken, loop_gpu_dwa_ms / steps_taken, steps_taken);

    // ---- artifacts: the driven path and the late-run master costmap ----------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/path.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "t_s,x_m,y_m,theta_rad,v_ms,w_rads\n";   // units in the header row (§12)
            const int rows = steps_taken + 1;
            for (int r = 0; r < rows; ++r) {
                const float* row = &traj[static_cast<size_t>(r) * 6];
                f << row[0] << ',' << row[1] << ',' << row[2] << ',' << row[3] << ',' << row[4] << ',' << row[5] << '\n';
            }
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/path.csv (%d rows)\n", steps_taken + 1);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/path.csv\n");

    bool pgm_ok = artifact_ok && write_pgm(out_dir + "/costmap.pgm", kGridW, kGridH, master_host);
    if (pgm_ok)
        std::printf("ARTIFACT: wrote demo/out/costmap.pgm (%dx%d, master costmap at the final tick)\n",
                    kGridW, kGridH);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/costmap.pgm\n");

    // ---- success check (the stable verdict) ----------------------------------
    const bool success = pgm_ok && goal_reached && (lethal_hits == 0);
    if (success)
        std::printf("RESULT: PASS (goal reached in %d/%d steps; 0 lethal-cell entries along the driven path)\n",
                    steps_taken, sc.steps);
    else
        std::printf("RESULT: FAIL (goal_reached=%s, lethal_hits=%lld — see [info] lines above)\n",
                    goal_reached ? "true" : "false", lethal_hits);
    return success ? 0 : 1;
}
