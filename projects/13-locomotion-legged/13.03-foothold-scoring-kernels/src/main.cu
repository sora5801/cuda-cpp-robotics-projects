// ===========================================================================
// main.cu — entry point for project 13.03
//           Foothold scoring kernels: slope, roughness, edge distance from
//           elevation maps
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed terrain RECIPE from
//      data/sample/terrain_scenario.csv and synthesize the 256x256
//      elevation map + the 1000-query foothold path from it (build_terrain
//      / build_queries below — pure host code, no GPU, no RNG: see the
//      file header in scripts/make_synthetic.py for why).
//   2. VERIFY STAGE (four gates, one per kernel): each of the four kernels
//      in kernels.cu is checked against its CPU twin in reference_cpu.cpp
//      using SHARED, PINNED upstream inputs — the same technique 08.01
//      uses for its single rollout kernel, applied per-stage here because
//      this project chains FOUR kernels and pinning each stage's inputs is
//      what lets each gate isolate exactly one kernel's correctness (see
//      the design note below and THEORY.md §How we verify correctness).
//   3. PIPELINE STAGE: the real, end-to-end, all-GPU run (slope/roughness
//      -> edge distance -> fusion -> selection) that produces this
//      project's actual deliverable — a fused score map and 1000 selected
//      footholds — with no CPU numbers mixed in anywhere.
//   4. ANALYTIC GATES: four checks against the terrain's own KNOWN ground
//      truth (the ramp's constructed angle, the step's known edge, the
//      flat region's near-zero slope, every selection's validity/radius).
//   5. ARTIFACTS: demo/out/foothold_score.pgm (a viewable grayscale score
//      map) and demo/out/selected_footholds.csv (every query's outcome).
//
// Why stage-isolated verification (read this once; it explains VERIFY)
// -----------------------------------------------------------------------
// A naive "run GPU pipeline end to end, run CPU pipeline end to end,
// compare the final scores" check has a real failure mode here: slope and
// roughness differ from their CPU twin by a few ULPs (ordinary float
// rounding — kernels.cu's header explains why), and the edge-distance
// kernel's hazard test is a DISCRETE comparison (slope > limit) against
// that slightly-different slope. A cell exactly on the hazard boundary
// could flip hazard/not-hazard between the two paths, and because
// edge-distance is a discrete nearest-hazard SEARCH, one flipped boundary
// cell can move a downstream cell's answer by much more than the ULP that
// caused it — a correct kernel could then look "wrong" by a wide margin,
// or a real bug could hide behind terrain that never crosses a boundary.
// Both are bad gates. The fix used throughout this file: verify each
// kernel with its CPU twin fed the IDENTICAL upstream arrays (the CPU
// oracle's own prior-stage output, uploaded to the device for the GPU
// kernel-under-test) — no error can accumulate BETWEEN kernels, so each
// gate is a clean statement about ONE kernel. The real, end-to-end
// artifact-producing run (step 3 above) is then a SEPARATE, all-GPU pass
// that never touches the CPU path at all.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" unchecked. Change a
// stable line => update demo/expected_output.txt in the same commit.
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
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

namespace {
constexpr float kTwoPi = 6.283185307179586f;
constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
constexpr float kNaN_() { return std::numeric_limits<float>::quiet_NaN(); }
} // namespace

// ---------------------------------------------------------------------------
// TerrainScenario — the parsed RECIPE (data/sample/terrain_scenario.csv):
// a ripple field, one ramp, one step, N rocks, one hole, and M query path
// segments. See scripts/make_synthetic.py's file header for the full
// rationale (why a recipe, not a committed grid) and docs/SYSTEM_DESIGN.md
// §3.6 for the message-shaped-struct convention this mirrors informally.
// ---------------------------------------------------------------------------
struct Rock { float cx_m, cy_m, h_m, r_m; };
struct PathSeg { float x0_m, y0_m, x1_m, y1_m; int n; };

struct TerrainScenario {
    float ripple_amp_m = 0.0f, ripple_wavelength_m = 1.0f;
    bool  have_ramp = false;
    float ramp_x0_m = 0, ramp_x1_m = 0, ramp_y0_m = 0, ramp_y1_m = 0, ramp_angle_deg = 0;
    bool  have_step = false;
    float step_x0_m = 0, step_x1_m = 0, step_y0_m = 0, step_y1_m = 0, step_edge_x_m = 0, step_height_m = 0;
    std::vector<Rock> rocks;
    bool  have_hole = false;
    float hole_x0_m = 0, hole_x1_m = 0, hole_y0_m = 0, hole_y1_m = 0;
    std::vector<PathSeg> paths;
    bool  loaded = false;
};

// Strict, greppable CSV loader — same discipline as every loader in this
// repo (08.01's load_scenario, 07.09's load_sample): unknown labels, short
// rows, or a missing required section abort the demo rather than silently
// running on a corrupt scenario.
static TerrainScenario load_terrain_scenario(const std::string& path)
{
    TerrainScenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');

        std::vector<float> v;
        auto read_floats = [&](int count) -> bool {
            for (int i = 0; i < count; ++i) {
                if (!std::getline(ss, cell, ',')) return false;
                v.push_back(std::strtof(cell.c_str(), nullptr));
            }
            return true;
        };

        if (label == "RIPPLE") {
            if (!read_floats(2)) { std::fprintf(stderr, "scenario: short RIPPLE row\n"); return TerrainScenario{}; }
            sc.ripple_amp_m = v[0]; sc.ripple_wavelength_m = v[1];
        } else if (label == "RAMP") {
            if (!read_floats(5)) { std::fprintf(stderr, "scenario: short RAMP row\n"); return TerrainScenario{}; }
            sc.ramp_x0_m = v[0]; sc.ramp_x1_m = v[1]; sc.ramp_y0_m = v[2]; sc.ramp_y1_m = v[3];
            sc.ramp_angle_deg = v[4]; sc.have_ramp = true;
        } else if (label == "STEP") {
            if (!read_floats(6)) { std::fprintf(stderr, "scenario: short STEP row\n"); return TerrainScenario{}; }
            sc.step_x0_m = v[0]; sc.step_x1_m = v[1]; sc.step_y0_m = v[2]; sc.step_y1_m = v[3];
            sc.step_edge_x_m = v[4]; sc.step_height_m = v[5]; sc.have_step = true;
        } else if (label == "ROCK") {
            if (!read_floats(4)) { std::fprintf(stderr, "scenario: short ROCK row\n"); return TerrainScenario{}; }
            sc.rocks.push_back({ v[0], v[1], v[2], v[3] });
        } else if (label == "HOLE") {
            if (!read_floats(4)) { std::fprintf(stderr, "scenario: short HOLE row\n"); return TerrainScenario{}; }
            sc.hole_x0_m = v[0]; sc.hole_x1_m = v[1]; sc.hole_y0_m = v[2]; sc.hole_y1_m = v[3];
            sc.have_hole = true;
        } else if (label == "PATH") {
            if (!read_floats(4) || !std::getline(ss, cell, ',')) {
                std::fprintf(stderr, "scenario: short PATH row\n"); return TerrainScenario{};
            }
            PathSeg seg{ v[0], v[1], v[2], v[3], std::atoi(cell.c_str()) };
            if (seg.n < 1) { std::fprintf(stderr, "scenario: PATH row needs n >= 1\n"); return TerrainScenario{}; }
            sc.paths.push_back(seg);
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return TerrainScenario{};
        }
    }

    if (!sc.have_ramp || !sc.have_step || !sc.have_hole || sc.rocks.empty() || sc.paths.empty()) {
        std::fprintf(stderr, "scenario: missing RAMP/STEP/HOLE/ROCK/PATH section(s)\n");
        return TerrainScenario{};
    }
    sc.loaded = true;
    return sc;
}

// ---------------------------------------------------------------------------
// build_terrain — synthesize the kGridW x kGridH height_m grid from the
// recipe. Every feature ADDS on top of the background ripple (a smooth,
// deterministic sin*cos field — see kernels.cuh/THEORY.md for its closed-
// form worst-case slope), confined to its OWN y-band so features never
// blend into each other (scripts/make_synthetic.py's file header documents
// the five-band layout). The RAMP's plateau and the STEP's upper shelf are
// defined to PERSIST for all x past their transition — never reverting to
// baseline — so no feature creates an undocumented second cliff at its
// nominal x1. The HOLE is applied LAST, unconditionally overriding every
// other contribution to NaN: a sensor dropout does not average with the
// ground truth beneath it, it simply erases it (THEORY.md's NaN discipline).
//
// This function runs ONCE, on the host, before either the GPU or the CPU
// pipeline sees any data — it is SETUP, not something being verified (the
// four kernels that consume its output are what VERIFY checks).
// ---------------------------------------------------------------------------
static std::vector<float> build_terrain(const TerrainScenario& sc)
{
    std::vector<float> h(static_cast<size_t>(kGridW) * kGridH);
    const float tan_ramp = std::tan(sc.ramp_angle_deg * kDegToRad);

    for (int row = 0; row < kGridH; ++row) {
        const float y = row * kCellM;
        for (int col = 0; col < kGridW; ++col) {
            const float x = col * kCellM;

            // Background: a smooth, deterministic ripple everywhere (a
            // stand-in for real sensor noise/ground texture with a known
            // analytic worst-case gradient — THEORY.md derives the bound
            // the "flat region" gate checks against).
            float z = sc.ripple_amp_m
                    * std::sin(kTwoPi * x / sc.ripple_wavelength_m)
                    * std::cos(kTwoPi * y / sc.ripple_wavelength_m);

            // RAMP: confined to its y-band; rises linearly across
            // [x0,x1] then PLATEAUS (never reverts) for x > x1.
            if (y >= sc.ramp_y0_m && y <= sc.ramp_y1_m && x >= sc.ramp_x0_m) {
                const float xe = (x <= sc.ramp_x1_m) ? x : sc.ramp_x1_m;
                z += tan_ramp * (xe - sc.ramp_x0_m);
            }

            // STEP: confined to its y-band; a single vertical jump at
            // edge_x, then the upper shelf PLATEAUS for all x beyond it.
            if (y >= sc.step_y0_m && y <= sc.step_y1_m && x >= sc.step_edge_x_m) {
                z += sc.step_height_m;
            }

            // ROCKS: smooth compactly-supported domes,
            // bump(d) = h * (1 - (d/r)^2)^2 for d < r, else 0 — C1 at the
            // rim (zero slope AND zero value at d=r), so a rock never
            // creates its own undocumented cliff at its footprint edge.
            for (const Rock& rk : sc.rocks) {
                const float dx = x - rk.cx_m, dy = y - rk.cy_m;
                const float d = std::sqrt(dx * dx + dy * dy);
                if (d < rk.r_m) {
                    const float t = 1.0f - (d / rk.r_m) * (d / rk.r_m);
                    z += rk.h_m * t * t;
                }
            }

            h[static_cast<size_t>(row) * kGridW + col] = z;
        }
    }

    // HOLE — applied last, unconditionally: a real sensor dropout erases
    // whatever was there, it does not blend with it.
    for (int row = 0; row < kGridH; ++row) {
        const float y = row * kCellM;
        if (y < sc.hole_y0_m || y > sc.hole_y1_m) continue;
        for (int col = 0; col < kGridW; ++col) {
            const float x = col * kCellM;
            if (x < sc.hole_x0_m || x > sc.hole_x1_m) continue;
            h[static_cast<size_t>(row) * kGridW + col] = kNaN_();
        }
    }
    return h;
}

// build_queries — linearly interpolate n points along each PATH segment.
// seg_ids (parallel array, same length) records which segment each query
// came from, purely for the CSV artifact/[info] reporting — it is NOT part
// of the kernel's FootholdQuery record (kernels.cuh keeps that minimal).
static void build_queries(const TerrainScenario& sc,
                          std::vector<FootholdQuery>& queries, std::vector<int>& seg_ids)
{
    queries.clear(); seg_ids.clear();
    for (size_t s = 0; s < sc.paths.size(); ++s) {
        const PathSeg& seg = sc.paths[s];
        for (int i = 0; i < seg.n; ++i) {
            const float t = (seg.n > 1) ? static_cast<float>(i) / static_cast<float>(seg.n - 1) : 0.0f;
            FootholdQuery q;
            q.x_m = seg.x0_m + t * (seg.x1_m - seg.x0_m);
            q.y_m = seg.y0_m + t * (seg.y1_m - seg.y0_m);
            queries.push_back(q);
            seg_ids.push_back(static_cast<int>(s));
        }
    }
}

// ---------------------------------------------------------------------------
// Small device-buffer helpers — every launch wrapper in kernels.cuh takes
// raw device pointers the CALLER owns (the repo's usual style, e.g.
// 08.01's d_u_nom/d_eps); these three functions keep the alloc/copy
// boilerplate in one place instead of repeating it at every call site.
// ---------------------------------------------------------------------------
static float* device_upload(const std::vector<float>& h)
{
    float* d = nullptr;
    const size_t bytes = h.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}
static float* device_alloc(size_t n)
{
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    return d;
}
static std::vector<float> device_download(const float* d, size_t n)
{
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------------------------
// compare_arrays — the per-kernel VERIFY metric: NaN PATTERN must match
// exactly (a hole or a degenerate fit is a structural fact, not a rounding
// question — THEORY.md explains why this is achievable bit-for-bit even
// though the underlying float VALUES are only tolerance-close), and every
// non-NaN pair must agree within an absolute tolerance.
// ---------------------------------------------------------------------------
struct ArrayDiff { float max_abs_diff = 0.0f; long long nan_mismatches = 0; };

static ArrayDiff compare_arrays(const std::vector<float>& gpu, const std::vector<float>& cpu)
{
    ArrayDiff d;
    for (size_t i = 0; i < gpu.size(); ++i) {
        const bool gn = std::isnan(gpu[i]), cn = std::isnan(cpu[i]);
        if (gn != cn) { d.nan_mismatches++; continue; }
        if (gn) continue;   // both NaN: agreement
        const float diff = std::fabs(gpu[i] - cpu[i]);
        if (diff > d.max_abs_diff) d.max_abs_diff = diff;
    }
    return d;
}

// cell_index — round a (x_m,y_m) map-frame point to its containing cell's
// flat index, clamped in-bounds. Used only by the ANALYTIC gates below to
// probe specific, documented terrain coordinates (never by a kernel).
static int cell_index(float x_m, float y_m)
{
    int col = static_cast<int>(std::lround(x_m / kCellM));
    int row = static_cast<int>(std::lround(y_m / kCellM));
    col = std::max(0, std::min(kGridW - 1, col));
    row = std::max(0, std::min(kGridH - 1, row));
    return row * kGridW + col;
}

// region_stats — mean/max slope (deg) and mean score over a rectangular
// map-frame region [x0,x1]x[y0,y1] — the tool the FLAT and RAMP analytic
// gates use to check measured geometry against the recipe's ground truth.
struct RegionStats { float mean_slope_deg = 0.0f, max_slope_deg = 0.0f, mean_score = 0.0f; long long n = 0; };

static RegionStats region_stats(const std::vector<float>& slope_rad, const std::vector<float>& score,
                                float x0, float x1, float y0, float y1)
{
    RegionStats rs;
    const int col0 = std::max(0, static_cast<int>(std::floor(x0 / kCellM)));
    const int col1 = std::min(kGridW, static_cast<int>(std::floor(x1 / kCellM)) + 1);
    const int row0 = std::max(0, static_cast<int>(std::floor(y0 / kCellM)));
    const int row1 = std::min(kGridH, static_cast<int>(std::floor(y1 / kCellM)) + 1);

    double sum_slope = 0.0, sum_score = 0.0;
    float max_slope = 0.0f;
    long long n = 0;
    for (int row = row0; row < row1; ++row) {
        for (int col = col0; col < col1; ++col) {
            const int idx = row * kGridW + col;
            const float s = slope_rad[idx];
            if (!std::isnan(s)) { sum_slope += s; if (s > max_slope) max_slope = s; }
            sum_score += score[idx];
            ++n;
        }
    }
    rs.n = n;
    rs.mean_slope_deg = n ? static_cast<float>(sum_slope / n) * kRadToDeg : 0.0f;
    rs.max_slope_deg = max_slope * kRadToDeg;
    rs.mean_score = n ? static_cast<float>(sum_score / n) : 0.0f;
    return rs;
}

// ---------------------------------------------------------------------------
// Artifacts: a P5 (binary) PGM — the smallest real image format there is —
// for the fused score map, and a CSV for every foothold query's outcome.
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, int width, int height, const std::vector<uint8_t>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
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

static bool write_footholds_csv(const std::string& path,
                                const std::vector<FootholdQuery>& queries,
                                const std::vector<int>& seg_ids,
                                const std::vector<FootholdResult>& results)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "query_index,segment,x_nom_m,y_nom_m,row,col,x_sel_m,y_sel_m,score,valid,dist_m\n";
    for (size_t q = 0; q < queries.size(); ++q) {
        const FootholdResult& r = results[q];
        const float x_sel = (r.row >= 0) ? r.col * kCellM : -1.0f;
        const float y_sel = (r.row >= 0) ? r.row * kCellM : -1.0f;
        f << q << ',' << seg_ids[q] << ','
          << queries[q].x_m << ',' << queries[q].y_m << ','
          << r.row << ',' << r.col << ',' << x_sel << ',' << y_sel << ','
          << r.score << ',' << r.valid << ',' << r.dist_m << '\n';
    }
    return static_cast<bool>(f);
}

static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_scenario(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/terrain_scenario.csv");
    candidates.push_back("data/sample/terrain_scenario.csv");
    candidates.push_back("../data/sample/terrain_scenario.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// Tolerances for the four VERIFY gates. Stage 1 compares two INDEPENDENT
// float computations (GPU vs CPU, same formula, different compilers) so a
// few ULPs of drift through a 25-term sum + 3x3 solve is expected; stages
// 2-4 are fed BIT-IDENTICAL pinned inputs (see the file header), so their
// tolerances exist only to absorb the sqrt/compare ops those stages still
// perform, and are expected to measure far below their ceiling in practice
// ([info] lines print the ACTUAL measured worst case every run).
constexpr float kSlopeTolRad = 1e-4f;      // ~0.006 deg (measured worst case ~1.4e-6 rad)
constexpr float kRoughTolM   = 1e-6f;      // 0.001 mm (measured worst case ~1.5e-8 m)
constexpr float kEdgeTolM    = 1e-4f;
constexpr float kScoreTolAbs = 1e-5f;

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data terrain_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] foothold scoring kernels: slope, roughness, edge distance, fusion + selection (project 13.03)\n");
    print_device_info();

    const float slope_limit_rad = std::atan(kFrictionMu);   // friction-cone derivation: THEORY.md §The problem
    std::printf("PROBLEM: %dx%d elevation map (%.2f m/cell, %.2fx%.2f m), plane-fit window %dx%d cells (%.2f m), "
                "mu=%.2f -> slope limit %.2f deg, edge search %d cells (%.2f m), foothold search disc %.2f m\n",
                kGridW, kGridH, static_cast<double>(kCellM),
                static_cast<double>(kGridW * kCellM), static_cast<double>(kGridH * kCellM),
                2 * kFitRadius + 1, 2 * kFitRadius + 1, static_cast<double>((2 * kFitRadius + 1) * kCellM),
                static_cast<double>(kFrictionMu), static_cast<double>(slope_limit_rad * kRadToDeg),
                kEdgeSearchRadiusCells, static_cast<double>(kEdgeSearchRadiusCells * kCellM),
                static_cast<double>(kFootholdSearchRadiusM));

    // ---- scenario -------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/terrain_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    TerrainScenario sc = load_terrain_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    std::vector<FootholdQuery> queries;
    std::vector<int> seg_ids;
    build_queries(sc, queries, seg_ids);
    const int N = static_cast<int>(queries.size());

    std::printf("SCENARIO: composed terrain (ramp %.1f deg, step %.2f m, %zu rocks, 1 hole) + "
                "%d foothold queries over %zu path segments [synthetic, seed 42]\n",
                static_cast<double>(sc.ramp_angle_deg), static_cast<double>(sc.step_height_m),
                sc.rocks.size(), N, sc.paths.size());

    const std::vector<float> height = build_terrain(sc);
    const size_t WH = static_cast<size_t>(kGridW) * kGridH;
    float* d_height = device_upload(height);

    // ======================= VERIFY STAGE (4 stage-isolated gates) =========
    bool verify_pass = true;

    // ---- Stage 1: slope + roughness (both paths read `height` directly) --
    std::vector<float> cpu_slope(WH), cpu_rough(WH);
    CpuTimer ct1; ct1.begin();
    slope_roughness_cpu(height.data(), cpu_slope.data(), cpu_rough.data());
    const double ct1_ms = ct1.end_ms();

    float* d_slope = device_alloc(WH);
    float* d_rough = device_alloc(WH);
    GpuTimer t1; t1.begin();
    launch_slope_roughness(d_height, d_slope, d_rough);
    const float t1_ms = t1.end_ms();
    std::vector<float> gpu_slope = device_download(d_slope, WH);
    std::vector<float> gpu_rough = device_download(d_rough, WH);

    const ArrayDiff diff_slope = compare_arrays(gpu_slope, cpu_slope);
    const ArrayDiff diff_rough = compare_arrays(gpu_rough, cpu_rough);
    const bool gate1 = diff_slope.nan_mismatches == 0 && diff_rough.nan_mismatches == 0
                     && diff_slope.max_abs_diff <= kSlopeTolRad && diff_rough.max_abs_diff <= kRoughTolM;
    std::printf("[info] stage1 slope/roughness: max|dslope|=%.3e rad, max|drough|=%.3e m, nan mismatches=%lld/%lld\n",
                static_cast<double>(diff_slope.max_abs_diff), static_cast<double>(diff_rough.max_abs_diff),
                diff_slope.nan_mismatches + diff_rough.nan_mismatches, static_cast<long long>(WH));
    std::printf("[time] stage1 slope/roughness (%zu cells): CPU %.2f ms | GPU kernel %.3f ms | "
                "speed-up %.0fx (teaching artifact; kernel only)\n",
                WH, ct1_ms, static_cast<double>(t1_ms), ct1_ms / (static_cast<double>(t1_ms) > 0.0 ? static_cast<double>(t1_ms) : 1.0));
    verify_pass = verify_pass && gate1;

    // ---- Stage 2: edge distance, fed the CPU's slope/roughness on BOTH ---
    // sides (isolates this kernel from stage 1's tiny drift — file header).
    float* d_slope_cpu = device_upload(cpu_slope);
    float* d_rough_cpu = device_upload(cpu_rough);
    std::vector<float> cpu_edge(WH);
    CpuTimer ct2; ct2.begin();
    edge_distance_cpu(height.data(), cpu_slope.data(), cpu_rough.data(), slope_limit_rad, cpu_edge.data());
    const double ct2_ms = ct2.end_ms();

    float* d_edge = device_alloc(WH);
    GpuTimer t2; t2.begin();
    launch_edge_distance(d_height, d_slope_cpu, d_rough_cpu, slope_limit_rad, d_edge);
    const float t2_ms = t2.end_ms();
    std::vector<float> gpu_edge = device_download(d_edge, WH);

    const ArrayDiff diff_edge = compare_arrays(gpu_edge, cpu_edge);
    long long hazard_mismatches = 0;
    for (size_t i = 0; i < WH; ++i)
        if ((gpu_edge[i] == 0.0f) != (cpu_edge[i] == 0.0f)) ++hazard_mismatches;
    const bool gate2 = diff_edge.nan_mismatches == 0 && diff_edge.max_abs_diff <= kEdgeTolM && hazard_mismatches == 0;
    std::printf("[info] stage2 edge distance: max|dedge|=%.3e m, hazard-mask mismatches=%lld\n",
                static_cast<double>(diff_edge.max_abs_diff), hazard_mismatches);
    std::printf("[time] stage2 edge distance (search radius %d cells): CPU %.2f ms | GPU kernel %.3f ms | "
                "speed-up %.0fx (teaching artifact; kernel only)\n",
                kEdgeSearchRadiusCells, ct2_ms, static_cast<double>(t2_ms),
                ct2_ms / (static_cast<double>(t2_ms) > 0.0 ? static_cast<double>(t2_ms) : 1.0));
    verify_pass = verify_pass && gate2;

    // ---- Stage 3: fusion, fed the CPU's slope/roughness/edge on BOTH sides
    float* d_edge_cpu = device_upload(cpu_edge);
    std::vector<float> cpu_score(WH);
    fusion_cpu(height.data(), cpu_slope.data(), cpu_rough.data(), cpu_edge.data(), slope_limit_rad, cpu_score.data());

    float* d_score = device_alloc(WH);
    GpuTimer t3; t3.begin();
    launch_fusion(d_height, d_slope_cpu, d_rough_cpu, d_edge_cpu, slope_limit_rad, d_score);
    const float t3_ms = t3.end_ms();
    std::vector<float> gpu_score = device_download(d_score, WH);

    const ArrayDiff diff_score = compare_arrays(gpu_score, cpu_score);
    const bool gate3 = diff_score.nan_mismatches == 0 && diff_score.max_abs_diff <= kScoreTolAbs;
    std::printf("[info] stage3 fusion: max|dscore|=%.3e, GPU kernel %.3f ms\n",
                static_cast<double>(diff_score.max_abs_diff), static_cast<double>(t3_ms));
    verify_pass = verify_pass && gate3;

    // ---- Stage 4: selection, fed the CPU's fused score on BOTH sides -----
    // Inputs are now bit-identical, so an exact cell-index match is the
    // gate (kernels.cu's tie-break note explains why this is achievable).
    float* d_score_cpu = device_upload(cpu_score);
    FootholdQuery* d_queries = nullptr;
    CUDA_CHECK(cudaMalloc(&d_queries, static_cast<size_t>(N) * sizeof(FootholdQuery)));
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), static_cast<size_t>(N) * sizeof(FootholdQuery), cudaMemcpyHostToDevice));
    FootholdResult* d_results4 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results4, static_cast<size_t>(N) * sizeof(FootholdResult)));

    GpuTimer t4; t4.begin();
    launch_foothold_selection(d_score_cpu, d_queries, N, d_results4);
    const float t4_ms = t4.end_ms();
    std::vector<FootholdResult> gpu_sel(static_cast<size_t>(N));
    CUDA_CHECK(cudaMemcpy(gpu_sel.data(), d_results4, static_cast<size_t>(N) * sizeof(FootholdResult), cudaMemcpyDeviceToHost));

    std::vector<FootholdResult> cpu_sel(static_cast<size_t>(N));
    foothold_selection_cpu(cpu_score.data(), queries.data(), N, cpu_sel.data());

    long long index_mismatches = 0;
    float max_dscore4 = 0.0f;
    for (int q = 0; q < N; ++q) {
        if (gpu_sel[static_cast<size_t>(q)].row != cpu_sel[static_cast<size_t>(q)].row
         || gpu_sel[static_cast<size_t>(q)].col != cpu_sel[static_cast<size_t>(q)].col)
            ++index_mismatches;
        const float d = std::fabs(gpu_sel[static_cast<size_t>(q)].score - cpu_sel[static_cast<size_t>(q)].score);
        if (d > max_dscore4) max_dscore4 = d;
    }
    const bool gate4 = index_mismatches == 0;
    std::printf("[info] stage4 selection: index mismatches=%lld/%d, max|dscore|=%.3e, GPU kernel %.3f ms\n",
                index_mismatches, N, static_cast<double>(max_dscore4), static_cast<double>(t4_ms));
    verify_pass = verify_pass && gate4;

    std::printf("VERIFY: %s (all 4 kernels agree with the CPU reference within documented tolerances)\n",
                verify_pass ? "PASS" : "FAIL");

    // Free the VERIFY-stage-only buffers (the pinned CPU-fed copies).
    CUDA_CHECK(cudaFree(d_slope)); CUDA_CHECK(cudaFree(d_rough));
    CUDA_CHECK(cudaFree(d_slope_cpu)); CUDA_CHECK(cudaFree(d_rough_cpu));
    CUDA_CHECK(cudaFree(d_edge)); CUDA_CHECK(cudaFree(d_edge_cpu));
    CUDA_CHECK(cudaFree(d_score)); CUDA_CHECK(cudaFree(d_score_cpu));
    CUDA_CHECK(cudaFree(d_queries)); CUDA_CHECK(cudaFree(d_results4));

    if (!verify_pass) {
        CUDA_CHECK(cudaFree(d_height));
        std::printf("RESULT: FAIL (GPU/CPU kernel disagreement — see [info] lines above)\n");
        return 1;
    }

    // ======================= PIPELINE STAGE (real, all-GPU run) ============
    // No CPU numbers anywhere below this line — this is the actual GPU
    // pipeline whose output becomes the artifacts and the analytic gates.
    float* d_pslope = device_alloc(WH);
    float* d_prough = device_alloc(WH);
    launch_slope_roughness(d_height, d_pslope, d_prough);

    float* d_pedge = device_alloc(WH);
    launch_edge_distance(d_height, d_pslope, d_prough, slope_limit_rad, d_pedge);

    float* d_pscore = device_alloc(WH);
    launch_fusion(d_height, d_pslope, d_prough, d_pedge, slope_limit_rad, d_pscore);

    FootholdQuery* d_pqueries = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pqueries, static_cast<size_t>(N) * sizeof(FootholdQuery)));
    CUDA_CHECK(cudaMemcpy(d_pqueries, queries.data(), static_cast<size_t>(N) * sizeof(FootholdQuery), cudaMemcpyHostToDevice));
    FootholdResult* d_presults = nullptr;
    CUDA_CHECK(cudaMalloc(&d_presults, static_cast<size_t>(N) * sizeof(FootholdResult)));
    launch_foothold_selection(d_pscore, d_pqueries, N, d_presults);

    const std::vector<float> pipeline_slope = device_download(d_pslope, WH);
    const std::vector<float> pipeline_edge  = device_download(d_pedge, WH);
    const std::vector<float> pipeline_score = device_download(d_pscore, WH);
    std::vector<FootholdResult> pipeline_results(static_cast<size_t>(N));
    CUDA_CHECK(cudaMemcpy(pipeline_results.data(), d_presults, static_cast<size_t>(N) * sizeof(FootholdResult), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_height));
    CUDA_CHECK(cudaFree(d_pslope)); CUDA_CHECK(cudaFree(d_prough));
    CUDA_CHECK(cudaFree(d_pedge)); CUDA_CHECK(cudaFree(d_pscore));
    CUDA_CHECK(cudaFree(d_pqueries)); CUDA_CHECK(cudaFree(d_presults));

    // ======================= ANALYTIC GATES (ground truth is the recipe) ===
    bool analytic_pass = true;

    // Gate A — FLAT CONTROL region [0.20,0.80]x[0.20,0.80]: only the
    // ripple's own tiny gradient is present; slope should sit well under
    // the ripple's closed-form worst case (THEORY.md), and score near 1.
    const RegionStats flat = region_stats(pipeline_slope, pipeline_score, 0.20f, 0.80f, 0.20f, 0.80f);
    constexpr float kFlatSlopeBoundDeg = 3.4f;   // margin above the ripple's ~2.04 deg analytic worst case
    constexpr float kFlatScoreBound = 0.95f;
    const bool gateA = flat.max_slope_deg < kFlatSlopeBoundDeg && flat.mean_score > kFlatScoreBound;
    std::printf("[info] gate A (flat control): max slope=%.3f deg (bound %.1f), mean score=%.4f (bound >%.2f), n=%lld\n",
                static_cast<double>(flat.max_slope_deg), static_cast<double>(kFlatSlopeBoundDeg),
                static_cast<double>(flat.mean_score), static_cast<double>(kFlatScoreBound), flat.n);
    analytic_pass = analytic_pass && gateA;

    // Gate B — RAMP region, margined in from its transition edges by one
    // fit-window radius so the measurement never touches the ramp's own
    // corner kinks: the fitted slope should track the CONSTRUCTED angle.
    const float margin = kFitRadius * kCellM;
    const RegionStats ramp = region_stats(pipeline_slope, pipeline_score,
                                          sc.ramp_x0_m + margin, sc.ramp_x1_m - margin,
                                          sc.ramp_y0_m + margin, sc.ramp_y1_m - margin);
    constexpr float kRampTolDeg = 1.5f;
    const bool gateB = std::fabs(ramp.mean_slope_deg - sc.ramp_angle_deg) <= kRampTolDeg;
    std::printf("[info] gate B (ramp): measured mean slope=%.3f deg (constructed %.2f deg, tol +/-%.1f deg), n=%lld\n",
                static_cast<double>(ramp.mean_slope_deg), static_cast<double>(sc.ramp_angle_deg),
                static_cast<double>(kRampTolDeg), ramp.n);
    analytic_pass = analytic_pass && gateB;

    // Gate C — STEP edge: the cell AT the edge must be hard-vetoed (score
    // exactly 0); a cell comfortably beyond fit-window + edge-search reach
    // from the edge must have its edge-distance SATURATED at the cap.
    const float step_mid_y = 0.5f * (sc.step_y0_m + sc.step_y1_m);
    const int idx_edge = cell_index(sc.step_edge_x_m, step_mid_y);
    const float far_x = sc.step_x0_m + 0.05f;   // comfortably < edge_x - (fit_radius+search_radius)
    const int idx_far = cell_index(far_x, step_mid_y);
    const float cap_m = static_cast<float>(kEdgeSearchRadiusCells) * kCellM;
    const bool gateC = pipeline_score[static_cast<size_t>(idx_edge)] == 0.0f
                     && pipeline_edge[static_cast<size_t>(idx_far)] >= cap_m - 1e-4f;
    std::printf("[info] gate C (step edge): score at edge (%.2f,%.2f)=%.4f (want 0), "
                "edge_distance at far cell (%.2f,%.2f)=%.4f m (want >= cap %.2f m)\n",
                static_cast<double>(sc.step_edge_x_m), static_cast<double>(step_mid_y),
                static_cast<double>(pipeline_score[static_cast<size_t>(idx_edge)]),
                static_cast<double>(far_x), static_cast<double>(step_mid_y),
                static_cast<double>(pipeline_edge[static_cast<size_t>(idx_far)]), static_cast<double>(cap_m));
    analytic_pass = analytic_pass && gateC;

    // Gate D — every one of the 1000 selections: within its search radius
    // (structural — true by construction of the kernel's disc walk, but
    // checked to catch a mapping bug) AND a valid (score >= threshold) cell.
    long long valid_count = 0, radius_ok_count = 0;
    float max_dist = 0.0f;
    for (int q = 0; q < N; ++q) {
        const FootholdResult& r = pipeline_results[static_cast<size_t>(q)];
        if (r.valid) ++valid_count;
        if (r.dist_m <= kFootholdSearchRadiusM + 1e-4f) ++radius_ok_count;
        if (r.dist_m > max_dist) max_dist = r.dist_m;
    }
    const bool gateD = valid_count == N && radius_ok_count == N;
    std::printf("[info] gate D (selection): valid=%lld/%d, within-radius=%lld/%d, max used disc distance=%.4f m (radius %.2f m)\n",
                valid_count, N, radius_ok_count, N, static_cast<double>(max_dist), static_cast<double>(kFootholdSearchRadiusM));
    analytic_pass = analytic_pass && gateD;

    // ======================= ARTIFACTS ======================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        std::vector<uint8_t> gray(WH);
        for (size_t i = 0; i < WH; ++i) {
            float s = pipeline_score[i];
            if (s < 0.0f) s = 0.0f; if (s > 1.0f) s = 1.0f;
            gray[i] = static_cast<uint8_t>(s * 255.0f + 0.5f);
        }
        artifact_ok = write_pgm(out_dir + "/foothold_score.pgm", kGridW, kGridH, gray);
    }
    const bool csv_ok = artifact_ok
        && write_footholds_csv(out_dir + "/selected_footholds.csv", queries, seg_ids, pipeline_results);
    if (csv_ok)
        std::printf("ARTIFACT: wrote demo/out/foothold_score.pgm (%dx%d) and demo/out/selected_footholds.csv (%d rows)\n",
                    kGridW, kGridH, N);
    else
        std::printf("ARTIFACT: FAILED to write demo/out files\n");

    const bool success = analytic_pass && csv_ok;
    if (success)
        std::printf("RESULT: PASS (GPU kernels verified against CPU reference; terrain gates and foothold selection within documented tolerances)\n");
    else
        std::printf("RESULT: FAIL (an analytic gate or artifact write failed — see [info] lines above)\n");
    return success ? 0 : 1;
}
