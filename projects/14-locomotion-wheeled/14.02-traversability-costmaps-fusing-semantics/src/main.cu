// ===========================================================================
// main.cu — entry point for project 14.02
//           Traversability costmaps fusing semantics + geometry
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario RECIPE from
//      data/sample/traversability_scenario.csv and synthesize the 256x256
//      elevation map, the 256x256 semantic-class + confidence maps, and the
//      teaching transect from it (build_elevation / build_semantics /
//      build_transect below — pure host code, no GPU, no RNG: the recipe
//      already carries the one random choice, rock placement, as literal
//      numbers — see scripts/make_synthetic.py's file header).
//   2. VERIFY STAGE (four gates, one per kernel): each of the four kernels
//      in kernels.cu is checked against its CPU twin in reference_cpu.cpp
//      using SHARED, PINNED upstream inputs — 13.03's stage-isolation
//      technique, applied here because this project chains four kernels and
//      pinning each stage's inputs is what lets each gate isolate exactly
//      one kernel's correctness (see the design note below and THEORY.md
//      §How we verify correctness).
//   3. PIPELINE STAGE: the real, end-to-end, all-GPU run (geometric layer ->
//      semantic layer -> fusion -> speed limit) that produces this
//      project's actual deliverable — a fused traversability costmap and a
//      per-cell speed-limit map — with no CPU numbers mixed in anywhere.
//   4. ANALYTIC GATES: four checks against the scenario's own KNOWN ground
//      truth — the two DESIGNED-DISAGREEMENT cases (a geometrically flat
//      region under a water pool must still be vetoed; a geometrically
//      rough vegetation patch must still be rescued to a valid, reduced-
//      speed cell) plus a pure-geometry veto (the ditch) and a pure-geometry
//      accuracy check (the berm's constructed slope angle).
//   5. ARTIFACTS: demo/out/traversability.pgm (fused-cost grayscale map),
//      demo/out/speed_limit.pgm (speed-limit grayscale map), and
//      demo/out/layers.csv (the teaching transect: every layer's value at
//      every sampled point along a path that crosses all six features).
//
// Why stage-isolated verification (read this once; it explains VERIFY)
// -----------------------------------------------------------------------
// A naive "run the GPU pipeline end to end, run the CPU pipeline end to
// end, compare only the final fused cost" check has the same real failure
// mode 13.03's main.cu documents: an early stage's ordinary float rounding
// could flip a cell across a hard-veto boundary, and because fusion_kernel's
// veto is DISCRETE, one flipped boundary cell can make a genuinely correct
// kernel look wrong by a wide margin several stages later — or let a real
// bug hide behind a scenario that never crosses a boundary. This file's
// answer, identical in spirit to 13.03's: feed each kernel-under-test and
// its CPU oracle the IDENTICAL upstream arrays (the CPU oracle's own prior-
// stage output, uploaded to the device for the GPU kernel), so no error can
// accumulate BETWEEN stages and every gate is a clean statement about ONE
// kernel. The real, end-to-end, artifact-producing run (step 3 above) is a
// SEPARATE, all-GPU pass that never touches the CPU path at all.
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
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09/13.03)
#else
#include <sys/stat.h>
#endif

namespace {
constexpr float kTwoPi = 6.283185307179586f;
constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
} // namespace

// ---------------------------------------------------------------------------
// Scenario — the parsed RECIPE (data/sample/traversability_scenario.csv).
// See scripts/make_synthetic.py's file header for the full row-type
// rationale and data/README.md for the field-by-field format.
// ---------------------------------------------------------------------------
struct Berm    { float x0, x1, y0, y1, angle_deg; };
struct Ditch   { float xs, xb0, xb1, xe, y0, y1, depth_m; };
struct Rock    { float cx, cy, h, r; };
struct VegBump { float x0, x1, y0, y1, amplitude_m, wavelength_m; };
struct SemRegion { uint8_t cls; float confidence; float x0, x1, y0, y1; };
struct Waypoint { std::string label; float x, y; };

struct Scenario {
    float ripple_amp_m = 0.0f, ripple_wavelength_m = 1.0f;
    std::vector<Berm> berms;
    std::vector<Ditch> ditches;
    std::vector<Rock> rocks;
    std::vector<VegBump> vegbumps;
    std::vector<SemRegion> semregions;
    bool have_confnoise = false;
    float confnoise_amp = 0.0f, confnoise_wavelength = 1.0f;
    std::vector<Waypoint> waypoints;
    bool loaded = false;
};

// class_from_name — strict string -> CLASS_* lookup; unknown names abort the
// load (the same "no silent corruption" discipline as 13.03's row parser).
static bool class_from_name(const std::string& name, uint8_t* out)
{
    if (name == "DIRT")       { *out = CLASS_DIRT;       return true; }
    if (name == "GRAVEL")     { *out = CLASS_GRAVEL;     return true; }
    if (name == "GRASS")      { *out = CLASS_GRASS;      return true; }
    if (name == "VEGETATION") { *out = CLASS_VEGETATION; return true; }
    if (name == "WATER")      { *out = CLASS_WATER;      return true; }
    if (name == "UNKNOWN")    { *out = CLASS_UNKNOWN;    return true; }
    return false;
}

// Strict, greppable CSV loader — same discipline as every loader in this
// repo (13.03's load_terrain_scenario, 08.01's load_scenario): unknown row
// labels, short rows, or a missing required section abort the demo rather
// than silently running on a corrupt scenario.
static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_ripple = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');

        auto read_floats = [&](int count, std::vector<float>* v) -> bool {
            for (int i = 0; i < count; ++i) {
                if (!std::getline(ss, cell, ',')) return false;
                v->push_back(std::strtof(cell.c_str(), nullptr));
            }
            return true;
        };

        std::vector<float> v;
        if (label == "RIPPLE") {
            if (!read_floats(2, &v)) { std::fprintf(stderr, "scenario: short RIPPLE row\n"); return Scenario{}; }
            sc.ripple_amp_m = v[0]; sc.ripple_wavelength_m = v[1]; have_ripple = true;
        } else if (label == "BERM") {
            if (!read_floats(5, &v)) { std::fprintf(stderr, "scenario: short BERM row\n"); return Scenario{}; }
            sc.berms.push_back({ v[0], v[1], v[2], v[3], v[4] });
        } else if (label == "DITCH") {
            if (!read_floats(7, &v)) { std::fprintf(stderr, "scenario: short DITCH row\n"); return Scenario{}; }
            sc.ditches.push_back({ v[0], v[1], v[2], v[3], v[4], v[5], v[6] });
        } else if (label == "ROCK") {
            if (!read_floats(4, &v)) { std::fprintf(stderr, "scenario: short ROCK row\n"); return Scenario{}; }
            sc.rocks.push_back({ v[0], v[1], v[2], v[3] });
        } else if (label == "VEGBUMP") {
            if (!read_floats(6, &v)) { std::fprintf(stderr, "scenario: short VEGBUMP row\n"); return Scenario{}; }
            sc.vegbumps.push_back({ v[0], v[1], v[2], v[3], v[4], v[5] });
        } else if (label == "SEMREGION") {
            std::string cls_name;
            if (!std::getline(ss, cls_name, ',')) { std::fprintf(stderr, "scenario: short SEMREGION row\n"); return Scenario{}; }
            uint8_t cls;
            if (!class_from_name(cls_name, &cls)) {
                std::fprintf(stderr, "scenario: unknown class name '%s' in SEMREGION\n", cls_name.c_str());
                return Scenario{};
            }
            if (!read_floats(5, &v)) { std::fprintf(stderr, "scenario: short SEMREGION row\n"); return Scenario{}; }
            sc.semregions.push_back({ cls, v[0], v[1], v[2], v[3], v[4] });
        } else if (label == "CONFNOISE") {
            if (!read_floats(2, &v)) { std::fprintf(stderr, "scenario: short CONFNOISE row\n"); return Scenario{}; }
            sc.confnoise_amp = v[0]; sc.confnoise_wavelength = v[1]; sc.have_confnoise = true;
        } else if (label == "WAYPOINT") {
            std::string wp_label;
            if (!std::getline(ss, wp_label, ',')) { std::fprintf(stderr, "scenario: short WAYPOINT row\n"); return Scenario{}; }
            if (!read_floats(2, &v)) { std::fprintf(stderr, "scenario: short WAYPOINT row\n"); return Scenario{}; }
            sc.waypoints.push_back({ wp_label, v[0], v[1] });
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }

    if (!have_ripple || sc.berms.empty() || sc.ditches.empty() || sc.rocks.empty()
     || sc.vegbumps.empty() || sc.semregions.empty() || sc.waypoints.size() < 2) {
        std::fprintf(stderr, "scenario: missing RIPPLE/BERM/DITCH/ROCK/VEGBUMP/SEMREGION/WAYPOINT section(s)\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

// ---------------------------------------------------------------------------
// build_elevation — synthesize the kGridW x kGridH elevation_m grid from the
// recipe. RIPPLE forms the everywhere-background; BERM/DITCH/ROCK/VEGBUMP
// each ADD their contribution within their own documented region — the five
// non-overlapping y-bands (scripts/make_synthetic.py's file header) mean
// this addition never needs to reason about two features overlapping.
//
// This function runs ONCE, on the host, before either the GPU or the CPU
// pipeline sees any data — it is SETUP, not something being verified (the
// four kernels that consume its output are what VERIFY checks).
// ---------------------------------------------------------------------------
static std::vector<float> build_elevation(const Scenario& sc)
{
    std::vector<float> h(static_cast<size_t>(kGridW) * kGridH);

    for (int row = 0; row < kGridH; ++row) {
        const float y = row * kCellM;
        for (int col = 0; col < kGridW; ++col) {
            const float x = col * kCellM;

            // Background: a deterministic traveling-wave ripple, EVERYWHERE.
            // z = A*sin(2*pi*(x+y)/L) has an EXACT closed-form gradient
            // magnitude sqrt(2)*A*(2*pi/L) (THEORY.md §The math derives it
            // in one line — a single cosine argument, unlike a two-variable
            // sin*cos product field, keeps the bound exact and simple).
            float z = sc.ripple_amp_m * std::sin(kTwoPi * (x + y) / sc.ripple_wavelength_m);

            // BERM: ramp-then-plateau ridge, confined to its y-band.
            for (const Berm& bm : sc.berms) {
                if (y < bm.y0 || y > bm.y1 || x < bm.x0) continue;
                const float xe = (x <= bm.x1) ? x : bm.x1;   // plateau past x1, never reverts
                z += std::tan(bm.angle_deg * kDegToRad) * (xe - bm.x0);
            }

            // DITCH: trapezoidal V-shaped depression, confined to its y-band.
            for (const Ditch& dt : sc.ditches) {
                if (y < dt.y0 || y > dt.y1) continue;
                if (x >= dt.xs && x < dt.xb0) {
                    z -= dt.depth_m * (x - dt.xs) / (dt.xb0 - dt.xs);          // descending wall
                } else if (x >= dt.xb0 && x <= dt.xb1) {
                    z -= dt.depth_m;                                          // flat bottom
                } else if (x > dt.xb1 && x <= dt.xe) {
                    z -= dt.depth_m * (1.0f - (x - dt.xb1) / (dt.xe - dt.xb1)); // ascending wall
                }
            }

            // ROCKS: smooth compactly-supported domes, bump(d) = h*(1-(d/r)^2)^2
            // for d<r (else 0) — C1 at the rim, no undocumented cliff at the edge.
            for (const Rock& rk : sc.rocks) {
                const float dx = x - rk.cx, dy = y - rk.cy;
                const float d = std::sqrt(dx * dx + dy * dy);
                if (d < rk.r) {
                    const float t = 1.0f - (d / rk.r) * (d / rk.r);
                    z += rk.h * t * t;
                }
            }

            // VEGBUMP: a SECOND, higher-frequency ripple ADDED on top of the
            // background within its own rectangle — stands in for a noisy
            // LiDAR return off a vegetation canopy (THEORY.md §The problem).
            for (const VegBump& vb : sc.vegbumps) {
                if (x < vb.x0 || x > vb.x1 || y < vb.y0 || y > vb.y1) continue;
                z += vb.amplitude_m * std::sin(kTwoPi * (x + y) / vb.wavelength_m);
            }

            h[static_cast<size_t>(row) * kGridW + col] = z;
        }
    }
    return h;
}

// ---------------------------------------------------------------------------
// build_semantics — synthesize semantic_class[] and confidence[] from the
// recipe. SEMREGION rows PAINT their rectangle in listed order (later rows
// override earlier ones on overlap — the first row is the whole-map
// default); CONFNOISE then adds a smooth, deterministic per-cell jitter to
// every cell's confidence and clamps to [0.05,0.99] (never exactly 0 or 1 —
// a real softmax output is never perfectly certain or perfectly blank).
// ---------------------------------------------------------------------------
static void build_semantics(const Scenario& sc,
                            std::vector<uint8_t>& semantic_class, std::vector<float>& confidence)
{
    const size_t WH = static_cast<size_t>(kGridW) * kGridH;
    semantic_class.assign(WH, CLASS_DIRT);
    confidence.assign(WH, 0.5f);   // overwritten by the first (whole-map) SEMREGION row in practice

    for (int row = 0; row < kGridH; ++row) {
        const float y = row * kCellM;
        for (int col = 0; col < kGridW; ++col) {
            const float x = col * kCellM;
            const size_t idx = static_cast<size_t>(row) * kGridW + col;
            for (const SemRegion& sr : sc.semregions) {
                if (x < sr.x0 || x >= sr.x1 || y < sr.y0 || y >= sr.y1) continue;
                semantic_class[idx] = sr.cls;
                confidence[idx] = sr.confidence;
            }
        }
    }

    if (sc.have_confnoise) {
        for (int row = 0; row < kGridH; ++row) {
            const float y = row * kCellM;
            for (int col = 0; col < kGridW; ++col) {
                const float x = col * kCellM;
                const size_t idx = static_cast<size_t>(row) * kGridW + col;
                const float jitter = sc.confnoise_amp
                    * std::sin(kTwoPi * x / sc.confnoise_wavelength)
                    * std::cos(kTwoPi * y / sc.confnoise_wavelength);
                float c = confidence[idx] + jitter;
                if (c < 0.05f) c = 0.05f;
                if (c > 0.99f) c = 0.99f;
                confidence[idx] = c;
            }
        }
    }
}

// TransectSample — one row of the demo/out/layers.csv teaching artifact.
struct TransectSample {
    std::string leg_label;
    float x_m, y_m;
};

// build_transect — connect consecutive WAYPOINT rows with straight legs,
// kSamplesPerLeg points each (the last leg also emits its final endpoint, so
// the transect visibly ends exactly at the last waypoint).
static std::vector<TransectSample> build_transect(const Scenario& sc)
{
    constexpr int kSamplesPerLeg = 60;
    std::vector<TransectSample> out;
    for (size_t i = 0; i + 1 < sc.waypoints.size(); ++i) {
        const Waypoint& a = sc.waypoints[i];
        const Waypoint& b = sc.waypoints[i + 1];
        const bool last_leg = (i + 2 == sc.waypoints.size());
        const int n = last_leg ? kSamplesPerLeg + 1 : kSamplesPerLeg;   // include the final endpoint once
        for (int k = 0; k < n; ++k) {
            const float t = static_cast<float>(k) / static_cast<float>(kSamplesPerLeg);
            TransectSample s;
            s.leg_label = a.label + "_TO_" + b.label;
            s.x_m = a.x + t * (b.x - a.x);
            s.y_m = a.y + t * (b.y - a.y);
            out.push_back(s);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Small device-buffer helpers — identical role to 13.03's (alloc/copy
// boilerplate kept in one place instead of repeated at every call site).
// ---------------------------------------------------------------------------
template <typename T>
static T* device_upload(const std::vector<T>& h)
{
    T* d = nullptr;
    const size_t bytes = h.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
static T* device_alloc(size_t n)
{
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
    return d;
}
template <typename T>
static std::vector<T> device_download(const T* d, size_t n)
{
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// compare_arrays — the per-kernel VERIFY metric: NaN pattern must match
// exactly (a structural fact, not a rounding question — 13.03's discipline),
// and every non-NaN pair must agree within an absolute tolerance.
struct ArrayDiff { float max_abs_diff = 0.0f; long long nan_mismatches = 0; };

static ArrayDiff compare_arrays(const std::vector<float>& gpu, const std::vector<float>& cpu)
{
    ArrayDiff d;
    for (size_t i = 0; i < gpu.size(); ++i) {
        const bool gn = std::isnan(gpu[i]), cn = std::isnan(cpu[i]);
        if (gn != cn) { d.nan_mismatches++; continue; }
        if (gn) continue;
        const float diff = std::fabs(gpu[i] - cpu[i]);
        if (diff > d.max_abs_diff) d.max_abs_diff = diff;
    }
    return d;
}

// compare_int_exact — exact-match count for the integer veto_reason array
// (a discrete classification, not a float measurement — an exact match is
// the right bar once both paths are fed bit-identical pinned inputs).
static long long compare_int_exact(const std::vector<int32_t>& gpu, const std::vector<int32_t>& cpu)
{
    long long mism = 0;
    for (size_t i = 0; i < gpu.size(); ++i) if (gpu[i] != cpu[i]) ++mism;
    return mism;
}

// cell_index — round a (x_m,y_m) map-frame point to its containing cell's
// flat index, clamped in-bounds. Used only by the ANALYTIC gates below.
static int cell_index(float x_m, float y_m)
{
    int col = static_cast<int>(std::lround(x_m / kCellM));
    int row = static_cast<int>(std::lround(y_m / kCellM));
    col = std::max(0, std::min(kGridW - 1, col));
    row = std::max(0, std::min(kGridH - 1, row));
    return row * kGridW + col;
}

// RegionStats — mean statistics over a rectangular map-frame region, the
// tool every analytic gate below uses to probe a documented sub-area.
struct RegionStats {
    double mean_slope_deg = 0.0, mean_geo_cost = 0.0, mean_fused_cost = 0.0, mean_speed_mps = 0.0;
    long long n = 0, n_veto = 0, n_geo_veto = 0, n_sem_veto = 0, n_valid = 0;
};

static RegionStats region_stats(const std::vector<float>& slope_rad, const std::vector<float>& geo_cost,
                                const std::vector<float>& fused_cost, const std::vector<float>& speed_mps,
                                const std::vector<int32_t>& veto_reason,
                                float x0, float x1, float y0, float y1)
{
    RegionStats rs;
    const int col0 = std::max(0, static_cast<int>(std::floor(x0 / kCellM)));
    const int col1 = std::min(kGridW, static_cast<int>(std::floor(x1 / kCellM)) + 1);
    const int row0 = std::max(0, static_cast<int>(std::floor(y0 / kCellM)));
    const int row1 = std::min(kGridH, static_cast<int>(std::floor(y1 / kCellM)) + 1);

    for (int row = row0; row < row1; ++row) {
        for (int col = col0; col < col1; ++col) {
            const int idx = row * kGridW + col;
            const float s = slope_rad[idx];
            if (!std::isnan(s)) rs.mean_slope_deg += s * kRadToDeg;
            rs.mean_geo_cost += geo_cost[idx];
            rs.mean_fused_cost += fused_cost[idx];
            rs.mean_speed_mps += speed_mps[idx];
            const int32_t reason = veto_reason[idx];
            if (reason != kVetoNone) ++rs.n_veto;
            if (reason & kVetoGeo) ++rs.n_geo_veto;
            if (reason & kVetoSem) ++rs.n_sem_veto;
            if (fused_cost[idx] <= kMaxValidCost) ++rs.n_valid;
            ++rs.n;
        }
    }
    if (rs.n > 0) {
        rs.mean_slope_deg /= rs.n; rs.mean_geo_cost /= rs.n;
        rs.mean_fused_cost /= rs.n; rs.mean_speed_mps /= rs.n;
    }
    return rs;
}

// ---------------------------------------------------------------------------
// Artifacts: two P5 (binary) PGMs and one CSV.
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

static const char* class_name(uint8_t cls)
{
    switch (cls) {
        case CLASS_DIRT: return "DIRT";
        case CLASS_GRAVEL: return "GRAVEL";
        case CLASS_GRASS: return "GRASS";
        case CLASS_VEGETATION: return "VEGETATION";
        case CLASS_WATER: return "WATER";
        case CLASS_UNKNOWN: return "UNKNOWN";
        default: return "?";
    }
}

static bool write_layers_csv(const std::string& path, const std::vector<TransectSample>& transect,
                             const std::vector<float>& elevation_m, const std::vector<float>& slope_rad,
                             const std::vector<float>& step_height_m, const std::vector<float>& roughness_m,
                             const std::vector<float>& geo_cost, const std::vector<uint8_t>& semantic_class,
                             const std::vector<float>& confidence, const std::vector<float>& semantic_cost,
                             const std::vector<float>& fused_cost, const std::vector<int32_t>& veto_reason,
                             const std::vector<float>& speed_mps)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "sample_index,leg,x_m,y_m,elevation_m,slope_deg,step_height_m,roughness_m,geo_cost,"
         "semantic_class,confidence,semantic_cost,fused_cost,veto_reason,speed_limit_mps\n";
    for (size_t i = 0; i < transect.size(); ++i) {
        const int idx = cell_index(transect[i].x_m, transect[i].y_m);
        const float s = slope_rad[idx];
        f << i << ',' << transect[i].leg_label << ','
          << transect[i].x_m << ',' << transect[i].y_m << ','
          << elevation_m[idx] << ',' << (std::isnan(s) ? -1.0f : s * kRadToDeg) << ','
          << step_height_m[idx] << ',' << roughness_m[idx] << ',' << geo_cost[idx] << ','
          << class_name(semantic_class[idx]) << ',' << confidence[idx] << ',' << semantic_cost[idx] << ','
          << fused_cost[idx] << ',' << veto_reason[idx] << ',' << speed_mps[idx] << '\n';
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/traversability_scenario.csv");
    candidates.push_back("data/sample/traversability_scenario.csv");
    candidates.push_back("../data/sample/traversability_scenario.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// Tolerances for the four VERIFY gates. Stage 1 compares two INDEPENDENT
// float32-vs-float64 implementations of the SAME plane-fit formula (GPU
// float32, CPU double — reference_cpu.cpp's file header explains the
// deliberate precision choice), so a real, small numeric gap is expected;
// stages 2-4 are fed BIT-IDENTICAL pinned inputs, so their tolerances exist
// only to absorb the handful of arithmetic ops those stages still perform.
// [info] lines print the ACTUAL measured worst case every run.
constexpr float kSlopeTolRad = 2e-3f;      // ~0.11 deg
constexpr float kStepTolM    = 1e-6f;      // min/max only: expect ~exact
constexpr float kRoughTolM   = 2e-4f;
constexpr float kSemCostTol  = 1e-5f;
constexpr float kGeoCostTol  = 2e-3f;
constexpr float kFusedCostTol = 2e-3f;
constexpr float kSpeedTolMps = 1e-4f;

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data traversability_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] traversability costmaps fusing semantics + geometry (project 14.02)\n");
    print_device_info();

    // ---- Wheeled-vehicle hard-veto limits (THEORY.md §The problem derives
    // both from first principles): slope_limit is the MORE RESTRICTIVE of a
    // Coulomb-friction traction argument and a static-rollover-geometry
    // argument; step_limit is a friction-cone corner-contact argument over
    // the wheel radius. Both are computed once, here, from the single-
    // sourced constants in kernels.cuh, and passed to every kernel/oracle
    // call below — never re-derived or hardcoded a second time.
    const float slope_limit_traction_rad = std::atan(kWheelMu);
    const float slope_limit_rollover_rad = std::atan(kTrackWidthM / (2.0f * kCogHeightM));
    const float slope_limit_rad = std::min(slope_limit_traction_rad, slope_limit_rollover_rad);
    const bool rollover_governs = slope_limit_rollover_rad < slope_limit_traction_rad;
    const float step_limit_m = kWheelRadiusM * (1.0f - std::cos(std::atan(kWheelMu)));

    std::printf("PROBLEM: %dx%d traversability costmap (%.2f m/cell, %.1fx%.1f m), geometric fit window "
                "%dx%d cells (%.2f m), step window %dx%d cells (%.2f m), mu=%.2f -> traction limit %.2f deg, "
                "track/2h=%.2f -> rollover limit %.2f deg -> slope limit %.2f deg (%s-governed), "
                "wheel r=%.2f m -> step limit %.4f m, fusion weights geo=%.2f/sem=%.2f\n",
                kGridW, kGridH, static_cast<double>(kCellM),
                static_cast<double>(kGridW * kCellM), static_cast<double>(kGridH * kCellM),
                2 * kFitRadiusCells + 1, 2 * kFitRadiusCells + 1, static_cast<double>((2 * kFitRadiusCells + 1) * kCellM),
                2 * kStepRadiusCells + 1, 2 * kStepRadiusCells + 1, static_cast<double>((2 * kStepRadiusCells + 1) * kCellM),
                static_cast<double>(kWheelMu), static_cast<double>(slope_limit_traction_rad * kRadToDeg),
                static_cast<double>(kTrackWidthM / (2.0f * kCogHeightM)), static_cast<double>(slope_limit_rollover_rad * kRadToDeg),
                static_cast<double>(slope_limit_rad * kRadToDeg), rollover_governs ? "rollover" : "traction",
                static_cast<double>(kWheelRadiusM), static_cast<double>(step_limit_m),
                static_cast<double>(kWeightGeo), static_cast<double>(kWeightSem));

    // ---- scenario -------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/traversability_scenario.csv missing (run scripts/make_synthetic.py?)\n");
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

    std::printf("SCENARIO: composed terrain (berm %.1f deg, ditch %.2f m deep, %zu rocks, vegetation canopy "
                "bump) + %zu semantic regions (6-class palette) + %zu-waypoint teaching transect [synthetic, seed 42]\n",
                static_cast<double>(sc.berms[0].angle_deg), static_cast<double>(sc.ditches[0].depth_m),
                sc.rocks.size(), sc.semregions.size(), sc.waypoints.size());

    const std::vector<float> elevation = build_elevation(sc);
    std::vector<uint8_t> semantic_class;
    std::vector<float> confidence;
    build_semantics(sc, semantic_class, confidence);
    const std::vector<TransectSample> transect = build_transect(sc);
    const size_t WH = static_cast<size_t>(kGridW) * kGridH;

    float* d_elevation = device_upload(elevation);
    uint8_t* d_semantic_class = device_upload(semantic_class);
    float* d_confidence = device_upload(confidence);

    // ======================= VERIFY STAGE (4 stage-isolated gates) =========
    bool verify_pass = true;

    // ---- Stage 1: geometric layer (both paths read `elevation` directly) --
    std::vector<float> cpu_slope(WH), cpu_step(WH), cpu_rough(WH);
    CpuTimer ct1; ct1.begin();
    geometric_layer_cpu(elevation.data(), cpu_slope.data(), cpu_step.data(), cpu_rough.data());
    const double ct1_ms = ct1.end_ms();

    float* d_slope = device_alloc<float>(WH);
    float* d_step  = device_alloc<float>(WH);
    float* d_rough = device_alloc<float>(WH);
    GpuTimer t1; t1.begin();
    launch_geometric_layer(d_elevation, d_slope, d_step, d_rough);
    const float t1_ms = t1.end_ms();
    std::vector<float> gpu_slope = device_download(d_slope, WH);
    std::vector<float> gpu_step  = device_download(d_step, WH);
    std::vector<float> gpu_rough = device_download(d_rough, WH);

    const ArrayDiff diff_slope = compare_arrays(gpu_slope, cpu_slope);
    const ArrayDiff diff_step  = compare_arrays(gpu_step, cpu_step);
    const ArrayDiff diff_rough = compare_arrays(gpu_rough, cpu_rough);
    const bool gate1 = diff_slope.nan_mismatches == 0 && diff_step.nan_mismatches == 0 && diff_rough.nan_mismatches == 0
                     && diff_slope.max_abs_diff <= kSlopeTolRad && diff_step.max_abs_diff <= kStepTolM
                     && diff_rough.max_abs_diff <= kRoughTolM;
    std::printf("[info] stage1 geometric layer: max|dslope|=%.3e rad, max|dstep|=%.3e m, max|drough|=%.3e m, "
                "nan mismatches=%lld/%lld\n",
                static_cast<double>(diff_slope.max_abs_diff), static_cast<double>(diff_step.max_abs_diff),
                static_cast<double>(diff_rough.max_abs_diff),
                diff_slope.nan_mismatches + diff_step.nan_mismatches + diff_rough.nan_mismatches,
                static_cast<long long>(WH));
    std::printf("[time] stage1 geometric layer (%zu cells): CPU %.2f ms | GPU kernel %.3f ms | "
                "speed-up %.0fx (teaching artifact; kernel only)\n",
                WH, ct1_ms, static_cast<double>(t1_ms), ct1_ms / (static_cast<double>(t1_ms) > 0.0 ? static_cast<double>(t1_ms) : 1.0));
    verify_pass = verify_pass && gate1;

    // ---- Stage 2: semantic layer (both paths read class+confidence directly)
    std::vector<float> cpu_semcost(WH);
    CpuTimer ct2; ct2.begin();
    semantic_layer_cpu(semantic_class.data(), confidence.data(), cpu_semcost.data());
    const double ct2_ms = ct2.end_ms();

    float* d_semcost = device_alloc<float>(WH);
    GpuTimer t2; t2.begin();
    launch_semantic_layer(d_semantic_class, d_confidence, d_semcost);
    const float t2_ms = t2.end_ms();
    std::vector<float> gpu_semcost = device_download(d_semcost, WH);

    const ArrayDiff diff_semcost = compare_arrays(gpu_semcost, cpu_semcost);
    const bool gate2 = diff_semcost.nan_mismatches == 0 && diff_semcost.max_abs_diff <= kSemCostTol;
    std::printf("[info] stage2 semantic layer: max|dsemantic_cost|=%.3e\n", static_cast<double>(diff_semcost.max_abs_diff));
    std::printf("[time] stage2 semantic layer (%zu cells): CPU %.2f ms | GPU kernel %.3f ms\n",
                WH, ct2_ms, static_cast<double>(t2_ms));
    verify_pass = verify_pass && gate2;

    // ---- Stage 3: fusion, fed the CPU's stage1+stage2 outputs on BOTH sides
    float* d_slope_cpu = device_upload(cpu_slope);
    float* d_step_cpu  = device_upload(cpu_step);
    float* d_rough_cpu = device_upload(cpu_rough);
    float* d_semcost_cpu = device_upload(cpu_semcost);
    std::vector<float> cpu_geocost(WH), cpu_fused(WH);
    std::vector<int32_t> cpu_veto(WH);
    CpuTimer ct3; ct3.begin();
    fusion_cpu(cpu_slope.data(), cpu_step.data(), cpu_rough.data(), semantic_class.data(), cpu_semcost.data(),
              slope_limit_rad, step_limit_m, cpu_geocost.data(), cpu_fused.data(), cpu_veto.data());
    const double ct3_ms = ct3.end_ms();

    float* d_geocost = device_alloc<float>(WH);
    float* d_fused = device_alloc<float>(WH);
    int32_t* d_veto = device_alloc<int32_t>(WH);
    GpuTimer t3; t3.begin();
    launch_fusion(d_slope_cpu, d_step_cpu, d_rough_cpu, d_semantic_class, d_semcost_cpu,
                 slope_limit_rad, step_limit_m, d_geocost, d_fused, d_veto);
    const float t3_ms = t3.end_ms();
    std::vector<float> gpu_geocost = device_download(d_geocost, WH);
    std::vector<float> gpu_fused = device_download(d_fused, WH);
    std::vector<int32_t> gpu_veto = device_download(d_veto, WH);

    const ArrayDiff diff_geocost = compare_arrays(gpu_geocost, cpu_geocost);
    const ArrayDiff diff_fused = compare_arrays(gpu_fused, cpu_fused);
    const long long veto_mismatches = compare_int_exact(gpu_veto, cpu_veto);
    const bool gate3 = diff_geocost.nan_mismatches == 0 && diff_fused.nan_mismatches == 0
                     && diff_geocost.max_abs_diff <= kGeoCostTol && diff_fused.max_abs_diff <= kFusedCostTol
                     && veto_mismatches == 0;
    std::printf("[info] stage3 fusion: max|dgeo_cost|=%.3e, max|dfused_cost|=%.3e, veto_reason mismatches=%lld/%zu\n",
                static_cast<double>(diff_geocost.max_abs_diff), static_cast<double>(diff_fused.max_abs_diff),
                veto_mismatches, WH);
    std::printf("[time] stage3 fusion (%zu cells): CPU %.2f ms | GPU kernel %.3f ms\n",
                WH, ct3_ms, static_cast<double>(t3_ms));
    verify_pass = verify_pass && gate3;

    // ---- Stage 4: speed limit, fed the CPU's fused cost on BOTH sides -----
    float* d_fused_cpu = device_upload(cpu_fused);
    std::vector<float> cpu_speed(WH);
    CpuTimer ct4; ct4.begin();
    speed_limit_cpu(cpu_fused.data(), cpu_speed.data());
    const double ct4_ms = ct4.end_ms();

    float* d_speed = device_alloc<float>(WH);
    GpuTimer t4; t4.begin();
    launch_speed_limit(d_fused_cpu, d_speed);
    const float t4_ms = t4.end_ms();
    std::vector<float> gpu_speed = device_download(d_speed, WH);

    const ArrayDiff diff_speed = compare_arrays(gpu_speed, cpu_speed);
    const bool gate4 = diff_speed.nan_mismatches == 0 && diff_speed.max_abs_diff <= kSpeedTolMps;
    std::printf("[info] stage4 speed limit: max|dspeed|=%.3e m/s\n", static_cast<double>(diff_speed.max_abs_diff));
    std::printf("[time] stage4 speed limit (%zu cells): CPU %.2f ms | GPU kernel %.3f ms\n",
                WH, ct4_ms, static_cast<double>(t4_ms));
    verify_pass = verify_pass && gate4;

    std::printf("VERIFY: %s (all 4 kernels agree with the CPU reference within documented tolerances)\n",
                verify_pass ? "PASS" : "FAIL");

    // Free the VERIFY-stage-only buffers (the pinned CPU-fed copies).
    CUDA_CHECK(cudaFree(d_slope)); CUDA_CHECK(cudaFree(d_step)); CUDA_CHECK(cudaFree(d_rough));
    CUDA_CHECK(cudaFree(d_slope_cpu)); CUDA_CHECK(cudaFree(d_step_cpu)); CUDA_CHECK(cudaFree(d_rough_cpu));
    CUDA_CHECK(cudaFree(d_semcost)); CUDA_CHECK(cudaFree(d_semcost_cpu));
    CUDA_CHECK(cudaFree(d_geocost)); CUDA_CHECK(cudaFree(d_fused)); CUDA_CHECK(cudaFree(d_veto));
    CUDA_CHECK(cudaFree(d_fused_cpu)); CUDA_CHECK(cudaFree(d_speed));

    if (!verify_pass) {
        CUDA_CHECK(cudaFree(d_elevation)); CUDA_CHECK(cudaFree(d_semantic_class)); CUDA_CHECK(cudaFree(d_confidence));
        std::printf("RESULT: FAIL (GPU/CPU kernel disagreement — see [info] lines above)\n");
        return 1;
    }

    // ======================= PIPELINE STAGE (real, all-GPU run) ============
    // No CPU numbers anywhere below this line — this is the actual GPU
    // pipeline whose output becomes the artifacts and the analytic gates.
    float* d_pslope = device_alloc<float>(WH);
    float* d_pstep  = device_alloc<float>(WH);
    float* d_prough = device_alloc<float>(WH);
    launch_geometric_layer(d_elevation, d_pslope, d_pstep, d_prough);

    float* d_psemcost = device_alloc<float>(WH);
    launch_semantic_layer(d_semantic_class, d_confidence, d_psemcost);

    float* d_pgeocost = device_alloc<float>(WH);
    float* d_pfused = device_alloc<float>(WH);
    int32_t* d_pveto = device_alloc<int32_t>(WH);
    launch_fusion(d_pslope, d_pstep, d_prough, d_semantic_class, d_psemcost,
                 slope_limit_rad, step_limit_m, d_pgeocost, d_pfused, d_pveto);

    float* d_pspeed = device_alloc<float>(WH);
    launch_speed_limit(d_pfused, d_pspeed);

    const std::vector<float> p_slope = device_download(d_pslope, WH);
    const std::vector<float> p_step  = device_download(d_pstep, WH);
    const std::vector<float> p_rough = device_download(d_prough, WH);
    const std::vector<float> p_semcost = device_download(d_psemcost, WH);
    const std::vector<float> p_geocost = device_download(d_pgeocost, WH);
    const std::vector<float> p_fused = device_download(d_pfused, WH);
    const std::vector<int32_t> p_veto = device_download(d_pveto, WH);
    const std::vector<float> p_speed = device_download(d_pspeed, WH);

    CUDA_CHECK(cudaFree(d_elevation)); CUDA_CHECK(cudaFree(d_semantic_class)); CUDA_CHECK(cudaFree(d_confidence));
    CUDA_CHECK(cudaFree(d_pslope)); CUDA_CHECK(cudaFree(d_pstep)); CUDA_CHECK(cudaFree(d_prough));
    CUDA_CHECK(cudaFree(d_psemcost)); CUDA_CHECK(cudaFree(d_pgeocost));
    CUDA_CHECK(cudaFree(d_pfused)); CUDA_CHECK(cudaFree(d_pveto)); CUDA_CHECK(cudaFree(d_pspeed));

    // ======================= ANALYTIC GATES (designed disagreement cases) ==
    bool analytic_pass = true;
    const float margin = static_cast<float>(kFitRadiusCells) * kCellM;   // 0.30 m — inset from feature edges

    // Gate A — WATER veto DESPITE near-perfect geometry: the pool sits
    // inside a region whose only geometric signal is the gentle background
    // ripple. Geometry alone must look cheap; fusion must still veto every
    // sampled cell — semantics wins on ITS OWN terms.
    const SemRegion& water_region = sc.semregions[5];   // WATER row, scripts/make_synthetic.py order
    const RegionStats waterA = region_stats(p_slope, p_geocost, p_fused, p_speed, p_veto,
                                            water_region.x0 + margin, water_region.x1 - margin,
                                            water_region.y0 + margin, water_region.y1 - margin);
    constexpr double kWaterGeoCostBound = 0.25;   // "near-perfect geometry" — see [info] for the measured value
    const bool gateA = waterA.mean_geo_cost < kWaterGeoCostBound && waterA.n_veto == waterA.n && waterA.n_sem_veto == waterA.n;
    std::printf("[info] gate A (flat-but-water): mean geo_cost=%.4f (bound <%.2f, i.e. geometry alone looks safe), "
                "vetoed=%lld/%lld (want all), semantic-veto=%lld/%lld (want all), mean fused_cost=%.4f (want 1.0000)\n",
                waterA.mean_geo_cost, kWaterGeoCostBound, waterA.n_veto, waterA.n, waterA.n_sem_veto, waterA.n,
                waterA.mean_fused_cost);
    analytic_pass = analytic_pass && gateA;

    // Gate B — VEGETATION rescue DESPITE bad geometry: the canopy-noise
    // patch must show an ELEVATED geo_cost (bad geometry — high roughness
    // from the high-frequency bump) yet the FUSED cost must stay valid
    // (<=kMaxValidCost, i.e. NOT hard-vetoed) and the speed limit must be
    // MEASURABLY reduced below the cruise cap — "rescued, at reduced speed".
    const SemRegion& veg_region = sc.semregions[6];   // VEGETATION row
    const RegionStats vegB = region_stats(p_slope, p_geocost, p_fused, p_speed, p_veto,
                                          veg_region.x0 + margin, veg_region.x1 - margin,
                                          veg_region.y0 + margin, veg_region.y1 - margin);
    constexpr double kVegGeoCostBound = 0.35;     // "meaningfully bad geometry" — measured value in [info]
    const bool gateB = vegB.mean_geo_cost > kVegGeoCostBound && vegB.n_veto == 0
                     && vegB.mean_fused_cost <= static_cast<double>(kMaxValidCost)
                     && vegB.mean_speed_mps < static_cast<double>(kVMaxMps) - 0.05
                     && vegB.mean_speed_mps > 0.5;
    std::printf("[info] gate B (rough-but-vegetation): mean geo_cost=%.4f (bound >%.2f, i.e. geometry alone looks bad), "
                "vetoed=%lld/%lld (want 0), mean fused_cost=%.4f (bound <=%.2f), mean speed_limit=%.3f m/s "
                "(cruise cap %.2f m/s, want visibly reduced but >0.5)\n",
                vegB.mean_geo_cost, kVegGeoCostBound, vegB.n_veto, vegB.n, vegB.mean_fused_cost,
                static_cast<double>(kMaxValidCost), vegB.mean_speed_mps, static_cast<double>(kVMaxMps));
    analytic_pass = analytic_pass && gateB;

    // Teaching comparison (NOT a gate — an honest, measured illustration of
    // THEORY.md's documented alternative fusion rule): what would a MAX-
    // based ("worst channel wins") fusion have produced for this same
    // vegetation patch, computed here post-hoc from the same GPU-produced
    // geo_cost/semantic_cost layers? README/THEORY discuss why this repo
    // ships the weighted blend as the primary rule instead.
    {
        double sum_max = 0.0; long long n = 0, n_would_be_invalid = 0;
        float worst_max_cell = 0.0f;
        const int col0 = std::max(0, static_cast<int>(std::floor((veg_region.x0 + margin) / kCellM)));
        const int col1 = std::min(kGridW, static_cast<int>(std::floor((veg_region.x1 - margin) / kCellM)) + 1);
        const int row0 = std::max(0, static_cast<int>(std::floor((veg_region.y0 + margin) / kCellM)));
        const int row1 = std::min(kGridH, static_cast<int>(std::floor((veg_region.y1 - margin) / kCellM)) + 1);
        for (int row = row0; row < row1; ++row)
            for (int col = col0; col < col1; ++col) {
                const int idx = row * kGridW + col;
                const float max_cost = std::max(p_geocost[idx], p_semcost[idx]);
                sum_max += max_cost;
                if (max_cost > worst_max_cell) worst_max_cell = max_cost;
                if (max_cost > kMaxValidCost) ++n_would_be_invalid;
                ++n;
            }
        const double mean_max_fusion = (n > 0) ? sum_max / n : 0.0;
        std::printf("[info] teaching comparison (not a gate): MAX-fusion on the same vegetation patch — mean "
                    "cost=%.4f, worst single cell=%.4f (vs. this project's weighted-blend mean fused_cost=%.4f, "
                    "worst-cell bound kMaxValidCost=%.2f); %lld/%lld cells MAX-fusion would flag INVALID that this "
                    "project's weighted blend does not (THEORY.md discusses this failure mode: geometry's own "
                    "noise, alone, can dominate a max-rule and flag a genuinely drivable, confidently-classified "
                    "cell invalid — worse the closer any one channel sits to ITS OWN threshold, even when the "
                    "other channel is confidently cheap)\n",
                    mean_max_fusion, static_cast<double>(worst_max_cell), vegB.mean_fused_cost,
                    static_cast<double>(kMaxValidCost), n_would_be_invalid, n);
    }

    // Gate C — DITCH veto REGARDLESS of (cheap) semantics: sampled on the
    // descending WALL (not the flat symmetric bottom — a symmetric V-shape's
    // exact center has near-zero PLANE-FIT slope by construction, since the
    // up-slope on one side and down-slope on the other cancel in a linear
    // fit; THEORY.md §Numerical considerations works this through). The wall
    // is labeled GRAVEL (a cheap 0.10 prior) yet its 45 deg grade must
    // exceed the wheeled-vehicle slope limit, forcing a GEOMETRIC veto
    // independent of how good the semantic reading is.
    const Ditch& ditch = sc.ditches[0];
    const float ditch_wall_x = 0.5f * (ditch.xs + ditch.xb0);   // descending-wall midpoint
    const float ditch_wall_y = 0.5f * (ditch.y0 + ditch.y1);
    const int idx_ditch = cell_index(ditch_wall_x, ditch_wall_y);
    const bool gateC = p_fused[static_cast<size_t>(idx_ditch)] == 1.0f
                     && (p_veto[static_cast<size_t>(idx_ditch)] & kVetoGeo) != 0
                     && semantic_class[static_cast<size_t>(idx_ditch)] != CLASS_WATER;
    std::printf("[info] gate C (ditch veto despite cheap semantics): fused_cost on ditch wall (%.2f,%.2f)=%.4f "
                "(want 1.0000), veto_reason=%d (want geo bit set), semantic class there=%s (want NOT water)\n",
                static_cast<double>(ditch_wall_x), static_cast<double>(ditch_wall_y),
                static_cast<double>(p_fused[static_cast<size_t>(idx_ditch)]),
                p_veto[static_cast<size_t>(idx_ditch)], class_name(semantic_class[static_cast<size_t>(idx_ditch)]));
    analytic_pass = analytic_pass && gateC;

    // Gate D — BERM slope accuracy: measured mean slope over the ramp
    // (margined in from its own transition kinks) must track the RECIPE's
    // constructed angle — a pure geometry-fidelity check, independent of
    // veto/rescue behavior (13.03's ramp gate, reused).
    const Berm& berm = sc.berms[0];
    const RegionStats bermD = region_stats(p_slope, p_geocost, p_fused, p_speed, p_veto,
                                           berm.x0 + margin, berm.x1 - margin,
                                           berm.y0 + 1.0f, berm.y1 - 1.0f);
    constexpr double kBermTolDeg = 1.5;
    const bool gateD = std::fabs(bermD.mean_slope_deg - static_cast<double>(berm.angle_deg)) <= kBermTolDeg;
    std::printf("[info] gate D (berm angle accuracy): measured mean slope=%.3f deg (constructed %.2f deg, "
                "tol +/-%.1f deg), n=%lld\n",
                bermD.mean_slope_deg, static_cast<double>(berm.angle_deg), kBermTolDeg, bermD.n);
    analytic_pass = analytic_pass && gateD;

    // ======================= ARTIFACTS ======================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        // Fused-cost costmap: WHITE (255) = free (cost 0), BLACK (0) = lethal
        // (cost 1) — an intuitive "brighter is safer" convention, documented
        // in demo/README.md (Nav2's own costmap_2d viewers use the opposite
        // convention; 23.01's README notes the same "pick one, document it"
        // choice this project makes independently).
        std::vector<uint8_t> gray_cost(WH);
        std::vector<uint8_t> gray_speed(WH);
        for (size_t i = 0; i < WH; ++i) {
            float c = p_fused[i]; if (c < 0.0f) c = 0.0f; if (c > 1.0f) c = 1.0f;
            gray_cost[i] = static_cast<uint8_t>((1.0f - c) * 255.0f + 0.5f);
            float v = p_speed[i] / kVMaxMps; if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
            gray_speed[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
        artifact_ok = write_pgm(out_dir + "/traversability.pgm", kGridW, kGridH, gray_cost)
                   && write_pgm(out_dir + "/speed_limit.pgm", kGridW, kGridH, gray_speed);
    }
    const bool csv_ok = artifact_ok
        && write_layers_csv(out_dir + "/layers.csv", transect, elevation, p_slope, p_step, p_rough,
                            p_geocost, semantic_class, confidence, p_semcost, p_fused, p_veto, p_speed);
    if (csv_ok)
        std::printf("ARTIFACT: wrote demo/out/traversability.pgm (%dx%d), demo/out/speed_limit.pgm (%dx%d), "
                    "and demo/out/layers.csv (%zu rows)\n",
                    kGridW, kGridH, kGridW, kGridH, transect.size());
    else
        std::printf("ARTIFACT: FAILED to write demo/out files\n");

    const bool success = analytic_pass && csv_ok;
    if (success)
        std::printf("RESULT: PASS (GPU kernels verified against CPU reference; the two designed-disagreement "
                    "cases and both pure-geometry gates hold within documented tolerances)\n");
    else
        std::printf("RESULT: FAIL (an analytic gate or artifact write failed — see [info] lines above)\n");
    return success ? 0 : 1;
}
