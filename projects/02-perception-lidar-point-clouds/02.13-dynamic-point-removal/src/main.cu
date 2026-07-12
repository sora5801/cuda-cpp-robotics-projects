// ===========================================================================
// main.cu — entry point for project 02.13
//           Dynamic point removal (raycast free-space carving)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed sample (data/sample/beams.csv + poses.csv): K=10
//      posed scans, 14,400 beams total, each either a HIT (with ground-truth
//      cohort + dynamic/static label, used ONLY for grading) or a MAX-RANGE
//      miss.
//   2. VERIFY STAGE (the CLAUDE.md §5 GPU-vs-CPU gate, done in THREE parts
//      because this project has three independently-verifiable stages):
//        (a) DDA TRACE EXACT — a documented subset of beams marched on both
//            GPU and CPU; the ORDERED voxel sequences must match exactly.
//        (b) LEDGER EXACT — the full 14,400-beam carve, GPU vs CPU, the
//            three ledger arrays compared element-wise, exactly.
//        (c) CLASSIFICATION EXACT — GPU vs CPU classification against the
//            (already-verified-identical) ledger, labels exact, scores
//            within float headroom.
//   3. SECONDARY ANALYSIS, host-only, using the CPU twins the verify stage
//      just proved bit-exact against the GPU (a 5-scan ledger for the
//      "late leaver" before/after comparison; a per-scan incremental carve
//      for the pedestrian evidence-accumulation artifact) — CLAUDE.md §5's
//      GPU-vs-CPU gate does not require re-verifying a code path already
//      proven identical; re-deriving everything on the GPU a second time
//      would test nothing new.
//   4. GATES: ghost_removal, late_leaver, static_preservation,
//      free_space_consistency, max_range_carving (all pass/fail) plus
//      contention (an [info]-only measurement).
//   5. ARTIFACTS: a top-view triptych (raw map with ghosts / cleaned map /
//      ground-truth-static-only), a per-scan pedestrian evidence CSV, and a
//      gates_metrics.csv summary — all under demo/out/.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "VERIFY:", "GATE:", "ARTIFACT:", "RESULT:" — NEVER carrying a measured
// number (percentages, timings, counts derived from running the algorithm),
// only fixed verdicts and fixed descriptive text (the same discipline
// 08.01's file header states explicitly). Every measured number lives on an
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
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Gate thresholds — MEASURED-THEN-MARGINED against this project's committed
// sample (README "Expected output" states the actual numbers from the run
// that produced demo/expected_output.txt). These are not arbitrary: each
// ceiling/floor sits comfortably on the safe side of what a full run of the
// committed data actually produces, so the gate is meaningful (a real
// regression would cross it) without being fragile (ordinary float-rounding
// noise cannot flip it) — CLAUDE.md §12's "wide margins" discipline.
// ---------------------------------------------------------------------------
static constexpr int   kVerifyTraceSubsetSize      = 48;    // documented DDA-trace subset size
static constexpr float kGhostRemovalFloorPct       = 85.0f; // car-trail points removed, at least
static constexpr float kLateLeaverBeforeCeilingPct = 25.0f; // pedestrian removal rate BEFORE it leaves, at most
static constexpr float kLateLeaverAfterFloorPct    = 70.0f; // pedestrian removal rate AFTER it leaves, at least
static constexpr float kStaticPreservationCeilingPct = 15.0f; // static points falsely removed, at most
static constexpr float kContentionNearM = 3.0f;             // "near sensor" radius for the contention [info] gate (m)
static constexpr float kContentionFarM  = 10.0f;            // "far from sensor" radius (m)

// ===========================================================================
// Data loading — beams.csv / poses.csv, the committed sample scripts/
// make_synthetic.py writes (that file's module docstring is this format's
// specification; see it for the scene the numbers below describe).
// ===========================================================================

// One beam record per array slot (Structure-of-Arrays — kernels.cuh's file
// header "BEAM RECORD LAYOUT"). cohort/truth are GROUND TRUTH: loaded here
// but read ONLY by this file's gates/artifacts below, never by the carving
// or classification algorithm (kernels.cuh's file header "GROUND TRUTH").
struct BeamData {
    std::vector<int> scan_id;
    std::vector<float> dir;             // [n*3] interleaved xyz, unit, world frame
    std::vector<unsigned char> is_hit;  // [n] 1 = real return, 0 = max-range
    std::vector<float> range;           // [n] meters
    std::vector<int> cohort;            // [n] CohortId (ground truth)
    std::vector<int> truth;             // [n] 0=static/1=dynamic/-1=n/a (ground truth)
    int n = 0;
    bool loaded = false;
};

struct PoseData {
    std::vector<float> origin;   // [kNumScans*3] interleaved xyz, world frame (m)
    bool loaded = false;
};

// split_csv — the whole file's CSV tokenizer: split one line on ','. Plain
// and unambitious on purpose (CLAUDE.md §5: no CSV library dependency for a
// format this simple); every caller below knows exactly how many fields to
// expect and treats a mismatch as a hard load failure, never a silent skip.
static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) fields.push_back(cell);
    return fields;
}

// load_beams — parse beams.csv's '#'-prefixed header (asserting it against
// kernels.cuh's beam-model constants — the 02.08-style data/code consistency
// check kernels.cuh's file header promises) and its 14,400 data rows.
static BeamData load_beams(const std::string& path)
{
    BeamData bd;
    std::ifstream in(path);
    if (!in.is_open()) return bd;

    int hdr_num_scans = -1, hdr_num_beams = -1, hdr_azimuth_steps = -1;
    float hdr_max_range = -1.0f;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;   // a prose comment line, not a key=value line
            std::string key = line.substr(1, eq - 1);
            const size_t start = key.find_first_not_of(" \t");
            key = (start == std::string::npos) ? "" : key.substr(start);
            const std::string val = line.substr(eq + 1);
            if (key == "num_scans")       hdr_num_scans = std::atoi(val.c_str());
            else if (key == "num_beams")     hdr_num_beams = std::atoi(val.c_str());
            else if (key == "azimuth_steps") hdr_azimuth_steps = std::atoi(val.c_str());
            else if (key == "max_range_m")   hdr_max_range = std::strtof(val.c_str(), nullptr);
            continue;
        }
        const auto f = split_csv(line);
        if (f.size() != 8) {
            std::fprintf(stderr, "beams.csv: malformed row (expected 8 fields, got %zu): %s\n",
                         f.size(), line.c_str());
            return BeamData{};
        }
        bd.scan_id.push_back(std::atoi(f[0].c_str()));
        bd.dir.push_back(std::strtof(f[1].c_str(), nullptr));
        bd.dir.push_back(std::strtof(f[2].c_str(), nullptr));
        bd.dir.push_back(std::strtof(f[3].c_str(), nullptr));
        bd.is_hit.push_back(static_cast<unsigned char>(std::atoi(f[4].c_str())));
        bd.range.push_back(std::strtof(f[5].c_str(), nullptr));
        bd.cohort.push_back(std::atoi(f[6].c_str()));
        bd.truth.push_back(std::atoi(f[7].c_str()));
    }
    bd.n = static_cast<int>(bd.scan_id.size());

    if (hdr_num_scans != kNumScans || hdr_num_beams != kNumBeams ||
        hdr_azimuth_steps != kAzimuthSteps || std::fabs(hdr_max_range - kMaxRangeM) > 1e-3f) {
        std::fprintf(stderr,
            "beams.csv header mismatch: file has num_scans=%d num_beams=%d azimuth_steps=%d "
            "max_range_m=%.3f; kernels.cuh expects %d/%d/%d/%.3f - regenerate the sample "
            "(scripts/make_synthetic.py) or update kernels.cuh\n",
            hdr_num_scans, hdr_num_beams, hdr_azimuth_steps, static_cast<double>(hdr_max_range),
            kNumScans, kNumBeams, kAzimuthSteps, static_cast<double>(kMaxRangeM));
        return BeamData{};
    }
    if (bd.n != kTotalBeams) {
        std::fprintf(stderr, "beams.csv: expected %d beam rows, found %d\n", kTotalBeams, bd.n);
        return BeamData{};
    }
    bd.loaded = true;
    return bd;
}

// load_poses — parse poses.csv's 10 sensor-position rows. Orientation is
// asserted (not just read) to be identity — this project's documented scope
// cut (kernels.cuh's file header "BEAM RECORD LAYOUT" note); a non-identity
// row is a WARNING, not a hard failure, since it does not stop the pipeline
// from running (it would simply mean the loaded directions are no longer
// truly world-frame, silently wrong in a way worth flagging loudly).
static PoseData load_poses(const std::string& path)
{
    PoseData pd;
    std::ifstream in(path);
    if (!in.is_open()) return pd;

    pd.origin.assign(static_cast<size_t>(kNumScans) * 3, 0.0f);
    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto f = split_csv(line);
        if (f.size() != 9) { std::fprintf(stderr, "poses.csv: malformed row\n"); return PoseData{}; }
        const int sid = std::atoi(f[0].c_str());
        if (sid < 0 || sid >= kNumScans) { std::fprintf(stderr, "poses.csv: scan_id out of range\n"); return PoseData{}; }
        pd.origin[static_cast<size_t>(sid) * 3 + 0] = std::strtof(f[1].c_str(), nullptr);
        pd.origin[static_cast<size_t>(sid) * 3 + 1] = std::strtof(f[2].c_str(), nullptr);
        pd.origin[static_cast<size_t>(sid) * 3 + 2] = std::strtof(f[3].c_str(), nullptr);
        const float qw = std::strtof(f[4].c_str(), nullptr);
        if (std::fabs(qw - 1.0f) > 1e-6f)
            std::fprintf(stderr, "poses.csv: WARNING scan %d has non-identity orientation (qw=%.6f); "
                         "this project assumes identity throughout (see kernels.cuh)\n",
                         sid, static_cast<double>(qw));
        ++count;
    }
    pd.loaded = (count == kNumScans);
    if (!pd.loaded) std::fprintf(stderr, "poses.csv: expected %d poses, found %d\n", kNumScans, count);
    return pd;
}

// beam_point — the world-space point a HIT beam produced, derived (never
// stored twice — kernels.cuh's file header). Used by every host-side
// consumer below (gates, artifacts) that needs an actual xyz.
struct Vec3f { float x, y, z; };
static Vec3f beam_point(const BeamData& bd, const PoseData& pose, int i)
{
    const int sid = bd.scan_id[static_cast<size_t>(i)];
    Vec3f p;
    p.x = pose.origin[static_cast<size_t>(sid) * 3 + 0] + bd.dir[static_cast<size_t>(i) * 3 + 0] * bd.range[static_cast<size_t>(i)];
    p.y = pose.origin[static_cast<size_t>(sid) * 3 + 1] + bd.dir[static_cast<size_t>(i) * 3 + 1] * bd.range[static_cast<size_t>(i)];
    p.z = pose.origin[static_cast<size_t>(sid) * 3 + 2] + bd.dir[static_cast<size_t>(i) * 3 + 2] * bd.range[static_cast<size_t>(i)];
    return p;
}

// ---------------------------------------------------------------------------
// select_trace_subset — the DOCUMENTED subset for the DDA-trace-exact verify
// gate (README "Expected output"): one beam from each interesting category
// (a max-range miss, then a hit on every cohort), padded with a deterministic
// sequential run from the start of the array up to kVerifyTraceSubsetSize.
// Deterministic (depends only on the committed data file, never on RNG or
// launch-time scheduling), so the subset — and therefore the verify gate's
// outcome — is identical on every machine.
// ---------------------------------------------------------------------------
static std::vector<int> select_trace_subset(const BeamData& bd, int target_size)
{
    std::vector<int> idx;
    for (int i = 0; i < bd.n; ++i) if (bd.is_hit[static_cast<size_t>(i)] == 0) { idx.push_back(i); break; }
    const int wanted_cohorts[] = { kCohortWall, kCohortPole, kCohortWallEdge,
                                   kCohortCar, kCohortPedestrian, kCohortGhost };
    for (int c : wanted_cohorts)
        for (int i = 0; i < bd.n; ++i)
            if (bd.is_hit[static_cast<size_t>(i)] == 1 && bd.cohort[static_cast<size_t>(i)] == c) { idx.push_back(i); break; }

    for (int i = 0; i < bd.n && static_cast<int>(idx.size()) < target_size; ++i) {
        if (std::find(idx.begin(), idx.end(), i) == idx.end()) idx.push_back(i);
    }
    if (static_cast<int>(idx.size()) > target_size) idx.resize(static_cast<size_t>(target_size));
    return idx;
}

// ===========================================================================
// GateResult — one row of the pass/fail summary (also written to
// demo/out/gates_metrics.csv). "measured"/"threshold" are doubles purely for
// the CSV artifact and [info] printing; NEVER printed on a stable line
// (file header "Output contract").
// ===========================================================================
struct GateResult {
    std::string name;
    double measured = 0.0;
    double threshold = 0.0;
    bool pass = false;
    std::string note;
};

// ===========================================================================
// A tiny PPM (binary P6) canvas — the triptych artifact's rendering surface.
// No image library (CLAUDE.md §5: PPM's format IS "write the header, then
// raw RGB bytes" — nothing to link, nothing to explain away as a black box).
// ===========================================================================
struct PpmCanvas {
    int w, h;
    std::vector<unsigned char> rgb;   // [w*h*3], row-major, origin top-left

    PpmCanvas(int w_, int h_) : w(w_), h(h_), rgb(static_cast<size_t>(w_) * static_cast<size_t>(h_) * 3, 0) {}

    void set_px(int x, int y, unsigned char r, unsigned char g, unsigned char b)
    {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 3;
        rgb[idx] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
    }

    // A 3x3 splat so a single point is visible at 480x480 resolution over a
    // 32 m span (~15 cm/pixel) — a bare single pixel would be nearly invisible.
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

// world_to_panel_px — map a world (x,y) inside the voxel grid's footprint to
// a pixel inside ONE panel of the triptych (panel_x0 = that panel's left
// edge in the shared canvas). +y maps to "up" on screen (a flip from the
// row-major top-left-origin canvas), the conventional top-view orientation.
static void world_to_panel_px(float x, float y, int panel_x0, int panel_w, int panel_h, int& px, int& py)
{
    const float u = (x - kGridOriginX) / (static_cast<float>(kGridNX) * kVoxelSizeM);   // 0..1 across the grid
    const float v = (y - kGridOriginY) / (static_cast<float>(kGridNY) * kVoxelSizeM);   // 0..1 across the grid
    px = panel_x0 + static_cast<int>(u * static_cast<float>(panel_w - 1));
    py = (panel_h - 1) - static_cast<int>(v * static_cast<float>(panel_h - 1));
}

// ---------------------------------------------------------------------------
// write_triptych — the "money shot" artifact (README "Expected output"):
// three top-view panels side by side.
//   LEFT   (raw)          : every HIT point ever recorded, colored by GROUND
//                           TRUTH (red = truth-dynamic, light gray =
//                           truth-static) — this is where the car's 4-scan
//                           ghost trail and the pedestrian's stand are
//                           plainly visible as red smears.
//   MIDDLE (cleaned)       : only RETAINED points (label == STATIC), colored
//                           by whether that retention was correct (light
//                           gray = truth-static, correctly kept) or a MISS
//                           (orange = truth-dynamic but not removed — a
//                           false negative, honestly shown, not hidden).
//   RIGHT  (truth static)  : only ground-truth-static points, light gray —
//                           the answer key the middle panel is graded against.
// ---------------------------------------------------------------------------
static bool write_triptych(const std::string& path, const BeamData& bd, const PoseData& pose,
                           const std::vector<int>& label)
{
    constexpr int panel_w = 480, panel_h = 480, gutter = 6;
    const int W = panel_w * 3 + gutter * 2;
    PpmCanvas canvas(W, panel_h);
    for (auto& c : canvas.rgb) c = 20;   // near-black background (visible against both point colors)

    const int x0_raw = 0;
    const int x0_clean = panel_w + gutter;
    const int x0_truth = 2 * (panel_w + gutter);

    for (int i = 0; i < bd.n; ++i) {
        if (bd.is_hit[static_cast<size_t>(i)] == 0) continue;   // no point to plot for a max-range beam
        const Vec3f p = beam_point(bd, pose, i);
        int px, py;

        world_to_panel_px(p.x, p.y, x0_raw, panel_w, panel_h, px, py);
        if (bd.truth[static_cast<size_t>(i)] == 1) canvas.splat(px, py, 235, 64, 64);      // truth-dynamic: red
        else                                       canvas.splat(px, py, 190, 190, 190);   // truth-static: light gray

        if (label[static_cast<size_t>(i)] == 0) {   // RETAINED (classified static)
            world_to_panel_px(p.x, p.y, x0_clean, panel_w, panel_h, px, py);
            if (bd.truth[static_cast<size_t>(i)] == 1) canvas.splat(px, py, 255, 165, 0); // false negative: orange
            else                                       canvas.splat(px, py, 190, 190, 190);
        }

        if (bd.truth[static_cast<size_t>(i)] == 0) {
            world_to_panel_px(p.x, p.y, x0_truth, panel_w, panel_h, px, py);
            canvas.splat(px, py, 190, 190, 190);
        }
    }
    return canvas.write(path);
}

// ---------------------------------------------------------------------------
// ensure_out_dir — thin wrapper matching every other project's demo/out/
// artifact convention (util/paths.h's resolve_out_dir()).
// ---------------------------------------------------------------------------

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

    std::printf("[demo] Dynamic point removal (raycast free-space carving) - project 02.13\n");
    print_device_info();
    std::printf("PROBLEM: K=%d posed scans, %d beams (%d beams x %d az x %d scans), "
                "voxel %.2f m, grid %dx%dx%d (%d voxels), max range %.1f m\n",
                kNumScans, kTotalBeams, kNumBeams, kAzimuthSteps, kNumScans,
                static_cast<double>(kVoxelSizeM), kGridNX, kGridNY, kGridNZ, kNumVoxels,
                static_cast<double>(kMaxRangeM));

    // ---- 0) Data --------------------------------------------------------------
    const std::string beams_path = find_data_file(data_dir_override, argv[0], "beams.csv");
    const std::string poses_path = find_data_file(data_dir_override, argv[0], "poses.csv");
    if (beams_path.empty() || poses_path.empty()) {
        std::printf("[info] beams.csv/poses.csv not found under data/sample/ (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::printf("[info] data: %s , %s\n", beams_path.c_str(), poses_path.c_str());

    const BeamData bd = load_beams(beams_path);
    const PoseData pose = load_poses(poses_path);
    if (!bd.loaded || !pose.loaded) {
        std::printf("RESULT: FAIL (data malformed - see stderr)\n");
        return 1;
    }
    const int n_hit_beams = static_cast<int>(std::count(bd.is_hit.begin(), bd.is_hit.end(), 1));
    std::printf("[info] loaded %d beams (%d hits, %d max-range misses), %d poses\n",
                bd.n, n_hit_beams, bd.n - n_hit_beams, kNumScans);

    set_scan_origins(pose.origin.data());

    // ---- upload the FULL beam arrays once; every stage below reuses them ------
    int* d_scan_id = nullptr; float* d_dir = nullptr;
    unsigned char* d_is_hit = nullptr; float* d_range = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_id, static_cast<size_t>(bd.n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dir, static_cast<size_t>(bd.n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_is_hit, static_cast<size_t>(bd.n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_range, static_cast<size_t>(bd.n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scan_id, bd.scan_id.data(), static_cast<size_t>(bd.n) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dir, bd.dir.data(), static_cast<size_t>(bd.n) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_is_hit, bd.is_hit.data(), static_cast<size_t>(bd.n) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_range, bd.range.data(), static_cast<size_t>(bd.n) * sizeof(float), cudaMemcpyHostToDevice));

    bool all_verify_pass = true;

    // ======================= VERIFY 1: DDA TRACE EXACT ==========================
    // A documented subset (select_trace_subset) marched on BOTH paths; the
    // ORDERED voxel sequences must match EXACTLY (kernels.cu's "DETERMINISM"
    // note: no transcendental functions in the march, so bit-exactness is the
    // expected outcome, not generous luck).
    {
        const std::vector<int> subset = select_trace_subset(bd, kVerifyTraceSubsetSize);
        const int ns = static_cast<int>(subset.size());
        std::printf("[info] DDA-trace verify subset: %d beams (documented: one max-range beam + one hit "
                    "per cohort, padded sequentially)\n", ns);

        std::vector<int> sub_scan_id(static_cast<size_t>(ns));
        std::vector<float> sub_dir(static_cast<size_t>(ns) * 3);
        std::vector<unsigned char> sub_is_hit(static_cast<size_t>(ns));
        std::vector<float> sub_range(static_cast<size_t>(ns));
        for (int k = 0; k < ns; ++k) {
            const int i = subset[static_cast<size_t>(k)];
            sub_scan_id[static_cast<size_t>(k)] = bd.scan_id[static_cast<size_t>(i)];
            sub_dir[static_cast<size_t>(k) * 3 + 0] = bd.dir[static_cast<size_t>(i) * 3 + 0];
            sub_dir[static_cast<size_t>(k) * 3 + 1] = bd.dir[static_cast<size_t>(i) * 3 + 1];
            sub_dir[static_cast<size_t>(k) * 3 + 2] = bd.dir[static_cast<size_t>(i) * 3 + 2];
            sub_is_hit[static_cast<size_t>(k)] = bd.is_hit[static_cast<size_t>(i)];
            sub_range[static_cast<size_t>(k)] = bd.range[static_cast<size_t>(i)];
        }

        int* d_sub_scan_id = nullptr; float* d_sub_dir = nullptr;
        unsigned char* d_sub_is_hit = nullptr; float* d_sub_range = nullptr;
        int* d_trace_out = nullptr; int* d_trace_len = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sub_scan_id, static_cast<size_t>(ns) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sub_dir, static_cast<size_t>(ns) * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sub_is_hit, static_cast<size_t>(ns) * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_sub_range, static_cast<size_t>(ns) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_trace_out, static_cast<size_t>(ns) * kMaxDDASteps * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_trace_len, static_cast<size_t>(ns) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_sub_scan_id, sub_scan_id.data(), static_cast<size_t>(ns) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sub_dir, sub_dir.data(), static_cast<size_t>(ns) * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sub_is_hit, sub_is_hit.data(), static_cast<size_t>(ns) * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sub_range, sub_range.data(), static_cast<size_t>(ns) * sizeof(float), cudaMemcpyHostToDevice));

        launch_carve_trace(ns, d_sub_scan_id, d_sub_dir, d_sub_is_hit, d_sub_range, d_trace_out, d_trace_len);

        std::vector<int> gpu_trace(static_cast<size_t>(ns) * kMaxDDASteps);
        std::vector<int> gpu_len(static_cast<size_t>(ns));
        CUDA_CHECK(cudaMemcpy(gpu_trace.data(), d_trace_out, gpu_trace.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_len.data(), d_trace_len, gpu_len.size() * sizeof(int), cudaMemcpyDeviceToHost));

        int mismatches = 0;
        long long total_steps_compared = 0;
        std::vector<int> cpu_trace(kMaxDDASteps);
        for (int k = 0; k < ns; ++k) {
            const int i = subset[static_cast<size_t>(k)];
            const Vec3f o{ pose.origin[static_cast<size_t>(sub_scan_id[static_cast<size_t>(k)]) * 3 + 0],
                          pose.origin[static_cast<size_t>(sub_scan_id[static_cast<size_t>(k)]) * 3 + 1],
                          pose.origin[static_cast<size_t>(sub_scan_id[static_cast<size_t>(k)]) * 3 + 2] };
            const int cpu_len = carve_trace_one_beam_cpu(
                o.x, o.y, o.z,
                sub_dir[static_cast<size_t>(k) * 3 + 0], sub_dir[static_cast<size_t>(k) * 3 + 1], sub_dir[static_cast<size_t>(k) * 3 + 2],
                sub_range[static_cast<size_t>(k)], sub_is_hit[static_cast<size_t>(k)] != 0, cpu_trace.data());
            (void)i;
            if (cpu_len != gpu_len[static_cast<size_t>(k)]) { ++mismatches; continue; }
            for (int s = 0; s < cpu_len; ++s) {
                total_steps_compared++;
                if (cpu_trace[static_cast<size_t>(s)] != gpu_trace[static_cast<size_t>(k) * kMaxDDASteps + s]) { ++mismatches; break; }
            }
        }
        std::printf("[info] DDA trace comparison: %d mismatched beam(s) of %d checked, %lld total voxel "
                    "steps compared\n", mismatches, ns, total_steps_compared);
        std::printf("VERIFY: DDA trace exact %s (documented subset, integer voxel sequences)\n",
                    mismatches == 0 ? "PASS" : "FAIL");
        if (mismatches != 0) all_verify_pass = false;

        CUDA_CHECK(cudaFree(d_sub_scan_id)); CUDA_CHECK(cudaFree(d_sub_dir));
        CUDA_CHECK(cudaFree(d_sub_is_hit)); CUDA_CHECK(cudaFree(d_sub_range));
        CUDA_CHECK(cudaFree(d_trace_out)); CUDA_CHECK(cudaFree(d_trace_len));
    }

    // ======================= VERIFY 2: LEDGER EXACT ==============================
    // The FULL 14,400-beam carve, GPU vs CPU, compared voxel-counter by
    // voxel-counter, EXACT (order-independent integer sums — kernels.cuh's
    // file header "THE LEDGER"). The GPU/CPU timings measured here also feed
    // the [time] report later (file header point 3: no need to re-time a
    // second, redundant full carve).
    std::vector<unsigned int> h_hits(static_cast<size_t>(kNumVoxels), 0u);
    std::vector<unsigned int> h_pass_hit(static_cast<size_t>(kNumVoxels), 0u);
    std::vector<unsigned int> h_pass_max(static_cast<size_t>(kNumVoxels), 0u);
    double carve_gpu_ms = 0.0, carve_cpu_ms = 0.0;
    {
        unsigned int *d_hits = nullptr, *d_pass_hit = nullptr, *d_pass_max = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hits, static_cast<size_t>(kNumVoxels) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_pass_hit, static_cast<size_t>(kNumVoxels) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_pass_max, static_cast<size_t>(kNumVoxels) * sizeof(unsigned int)));
        Ledger ledger{ d_hits, d_pass_hit, d_pass_max };

        launch_ledger_clear(ledger);
        GpuTimer gt; gt.begin();
        launch_carve(bd.n, d_scan_id, d_dir, d_is_hit, d_range, ledger);
        carve_gpu_ms = static_cast<double>(gt.end_ms());

        std::vector<unsigned int> g_hits(static_cast<size_t>(kNumVoxels));
        std::vector<unsigned int> g_pass_hit(static_cast<size_t>(kNumVoxels));
        std::vector<unsigned int> g_pass_max(static_cast<size_t>(kNumVoxels));
        CUDA_CHECK(cudaMemcpy(g_hits.data(), d_hits, g_hits.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(g_pass_hit.data(), d_pass_hit, g_pass_hit.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(g_pass_max.data(), d_pass_max, g_pass_max.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        CpuTimer ct; ct.begin();
        carve_cpu(bd.n, bd.scan_id.data(), bd.dir.data(), bd.is_hit.data(), bd.range.data(),
                 pose.origin.data(), h_hits.data(), h_pass_hit.data(), h_pass_max.data());
        carve_cpu_ms = ct.end_ms();

        long long mismatches = 0;
        for (int v = 0; v < kNumVoxels; ++v) {
            if (g_hits[static_cast<size_t>(v)] != h_hits[static_cast<size_t>(v)]) ++mismatches;
            if (g_pass_hit[static_cast<size_t>(v)] != h_pass_hit[static_cast<size_t>(v)]) ++mismatches;
            if (g_pass_max[static_cast<size_t>(v)] != h_pass_max[static_cast<size_t>(v)]) ++mismatches;
        }
        std::printf("[info] ledger comparison: %lld mismatched counter(s) of %lld (%d voxels x 3 counters)\n",
                    mismatches, static_cast<long long>(kNumVoxels) * 3, kNumVoxels);
        std::printf("[time] carve (full %d-beam set): CPU %.2f ms | GPU kernel %.3f ms | speed-up "
                    "%.0fx (teaching artifact)\n", bd.n, carve_cpu_ms, carve_gpu_ms,
                    carve_gpu_ms > 0.0 ? carve_cpu_ms / carve_gpu_ms : 0.0);
        std::printf("VERIFY: hit/pass ledger exact %s (full-beam-set carve)\n", mismatches == 0 ? "PASS" : "FAIL");
        if (mismatches != 0) all_verify_pass = false;

        CUDA_CHECK(cudaFree(d_hits)); CUDA_CHECK(cudaFree(d_pass_hit)); CUDA_CHECK(cudaFree(d_pass_max));
    }

    // ======================= VERIFY 3: CLASSIFICATION EXACT =====================
    // GPU classify_kernel against a freshly re-uploaded copy of the (already
    // verified identical) ledger, vs classify_cpu against the host copy.
    std::vector<int> label(static_cast<size_t>(bd.n));
    std::vector<float> score(static_cast<size_t>(bd.n));
    {
        unsigned int *d_hits = nullptr, *d_pass_hit = nullptr, *d_pass_max = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hits, static_cast<size_t>(kNumVoxels) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_pass_hit, static_cast<size_t>(kNumVoxels) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_pass_max, static_cast<size_t>(kNumVoxels) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_hits, h_hits.data(), h_hits.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pass_hit, h_pass_hit.data(), h_pass_hit.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pass_max, h_pass_max.data(), h_pass_max.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        Ledger ledger{ d_hits, d_pass_hit, d_pass_max };

        float* d_score = nullptr; int* d_label = nullptr;
        CUDA_CHECK(cudaMalloc(&d_score, static_cast<size_t>(bd.n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_label, static_cast<size_t>(bd.n) * sizeof(int)));
        launch_classify(bd.n, d_scan_id, d_dir, d_is_hit, d_range, ledger, kDynamicThreshold, d_score, d_label);
        CUDA_CHECK(cudaMemcpy(score.data(), d_score, score.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(label.data(), d_label, label.size() * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<float> score_cpu(static_cast<size_t>(bd.n));
        std::vector<int> label_cpu(static_cast<size_t>(bd.n));
        classify_cpu(bd.n, bd.scan_id.data(), bd.dir.data(), bd.is_hit.data(), bd.range.data(),
                    pose.origin.data(), h_hits.data(), h_pass_hit.data(), h_pass_max.data(),
                    kDynamicThreshold, score_cpu.data(), label_cpu.data());

        int label_mismatches = 0;
        float worst_score_dev = 0.0f;
        for (int i = 0; i < bd.n; ++i) {
            if (label[static_cast<size_t>(i)] != label_cpu[static_cast<size_t>(i)]) ++label_mismatches;
            const float dev = std::fabs(score[static_cast<size_t>(i)] - score_cpu[static_cast<size_t>(i)]);
            if (dev > worst_score_dev) worst_score_dev = dev;
        }
        std::printf("[info] classification comparison: %d label mismatch(es) of %d points; worst score "
                    "deviation %.3e\n", label_mismatches, bd.n, static_cast<double>(worst_score_dev));
        const bool classify_pass = (label_mismatches == 0) && (worst_score_dev <= 1e-6f);
        std::printf("VERIFY: classification exact %s (given the verified ledger)\n", classify_pass ? "PASS" : "FAIL");
        if (!classify_pass) all_verify_pass = false;

        CUDA_CHECK(cudaFree(d_hits)); CUDA_CHECK(cudaFree(d_pass_hit)); CUDA_CHECK(cudaFree(d_pass_max));
        CUDA_CHECK(cudaFree(d_score)); CUDA_CHECK(cudaFree(d_label));
    }

    CUDA_CHECK(cudaFree(d_scan_id)); CUDA_CHECK(cudaFree(d_dir));
    CUDA_CHECK(cudaFree(d_is_hit)); CUDA_CHECK(cudaFree(d_range));

    if (!all_verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement in the verify stage - fix before trusting any gate below)\n");
        return 1;
    }

    // ======================= SECONDARY ANALYSIS (host-only) =====================
    // From here on, every computation uses carve_cpu/classify_cpu — the SAME
    // functions VERIFY 2/3 just proved bit-exact against the GPU kernels.
    // Re-deriving a second, independent GPU carve for each secondary ledger
    // below would exercise no code path the verify stage has not already
    // certified (file header point 3).

    // "Late leaver" before-ledger: carve ONLY the first kLateLeaverScans
    // scans (a scan-major array prefix — kernels.cuh's file header).
    std::vector<unsigned int> h_hits5(static_cast<size_t>(kNumVoxels), 0u);
    std::vector<unsigned int> h_pass_hit5(static_cast<size_t>(kNumVoxels), 0u);
    std::vector<unsigned int> h_pass_max5(static_cast<size_t>(kNumVoxels), 0u);
    carve_cpu(kLateLeaverBeamCount, bd.scan_id.data(), bd.dir.data(), bd.is_hit.data(), bd.range.data(),
             pose.origin.data(), h_hits5.data(), h_pass_hit5.data(), h_pass_max5.data());
    std::vector<float> score5(static_cast<size_t>(bd.n));
    std::vector<int> label5(static_cast<size_t>(bd.n));
    classify_cpu(bd.n, bd.scan_id.data(), bd.dir.data(), bd.is_hit.data(), bd.range.data(),
                pose.origin.data(), h_hits5.data(), h_pass_hit5.data(), h_pass_max5.data(),
                kDynamicThreshold, score5.data(), label5.data());

    // Pedestrian "evidence voxel": the voxel containing the CENTROID of every
    // pedestrian ground-truth hit point (a single representative voxel for
    // the per-scan CSV artifact below).
    double ped_sx = 0.0, ped_sy = 0.0, ped_sz = 0.0;
    int ped_count = 0;
    for (int i = 0; i < bd.n; ++i) {
        if (bd.is_hit[static_cast<size_t>(i)] == 1 && bd.cohort[static_cast<size_t>(i)] == kCohortPedestrian) {
            const Vec3f p = beam_point(bd, pose, i);
            ped_sx += p.x; ped_sy += p.y; ped_sz += p.z;
            ++ped_count;
        }
    }
    int ped_voxel = -1;
    if (ped_count > 0) {
        int pvx, pvy, pvz;
        world_to_voxel(static_cast<float>(ped_sx / ped_count), static_cast<float>(ped_sy / ped_count),
                       static_cast<float>(ped_sz / ped_count), pvx, pvy, pvz);
        if (voxel_in_bounds(pvx, pvy, pvz)) ped_voxel = voxel_index(pvx, pvy, pvz);
    }

    // Per-scan incremental carve for the pedestrian evidence-accumulation
    // CSV: carve scan 0, snapshot; carve scan 1 INTO the same accumulator,
    // snapshot; ... — carve_cpu's ledger arguments accumulate in place
    // across calls (it never clears them), so repeated calls on successive
    // scan slices ARE the incremental carve, no special-casing needed.
    std::vector<unsigned int> h_hits_inc(static_cast<size_t>(kNumVoxels), 0u);
    std::vector<unsigned int> h_pass_hit_inc(static_cast<size_t>(kNumVoxels), 0u);
    std::vector<unsigned int> h_pass_max_inc(static_cast<size_t>(kNumVoxels), 0u);
    std::vector<std::array<unsigned int, 3>> ped_evidence_by_scan;   // (hits, pass_hit, pass_maxrange) after each scan
    for (int k = 0; k < kNumScans; ++k) {
        carve_cpu(kBeamsPerScan,
                 bd.scan_id.data() + static_cast<size_t>(k) * kBeamsPerScan,
                 bd.dir.data() + static_cast<size_t>(k) * kBeamsPerScan * 3,
                 bd.is_hit.data() + static_cast<size_t>(k) * kBeamsPerScan,
                 bd.range.data() + static_cast<size_t>(k) * kBeamsPerScan,
                 pose.origin.data(), h_hits_inc.data(), h_pass_hit_inc.data(), h_pass_max_inc.data());
        if (ped_voxel >= 0)
            ped_evidence_by_scan.push_back({ h_hits_inc[static_cast<size_t>(ped_voxel)],
                                             h_pass_hit_inc[static_cast<size_t>(ped_voxel)],
                                             h_pass_max_inc[static_cast<size_t>(ped_voxel)] });
        else
            ped_evidence_by_scan.push_back({ 0u, 0u, 0u });
    }

    // ======================= GATES ================================================
    std::vector<GateResult> gates;

    // ---- ghost_removal: car-trail points classified DYNAMIC (removed) ----------
    {
        int total = 0, removed = 0;
        for (int i = 0; i < bd.n; ++i)
            if (bd.is_hit[static_cast<size_t>(i)] == 1 && bd.cohort[static_cast<size_t>(i)] == kCohortCar) {
                ++total;
                if (label[static_cast<size_t>(i)] == 1) ++removed;
            }
        const double pct = total > 0 ? 100.0 * removed / total : 0.0;
        GateResult g{ "ghost_removal", pct, kGhostRemovalFloorPct, pct >= kGhostRemovalFloorPct,
                     "car-trail points removed" };
        std::printf("[info] ghost_removal: %d/%d car-trail points classified dynamic (%.1f%%), floor %.1f%%\n",
                    removed, total, pct, static_cast<double>(kGhostRemovalFloorPct));
        gates.push_back(g);
    }

    // ---- late_leaver: pedestrian removal rate BEFORE vs AFTER it leaves --------
    {
        int total = 0, before_dyn = 0, after_dyn = 0;
        for (int i = 0; i < bd.n; ++i)
            if (bd.is_hit[static_cast<size_t>(i)] == 1 && bd.cohort[static_cast<size_t>(i)] == kCohortPedestrian) {
                ++total;
                if (label5[static_cast<size_t>(i)] == 1) ++before_dyn;
                if (label[static_cast<size_t>(i)] == 1) ++after_dyn;
            }
        const double before_pct = total > 0 ? 100.0 * before_dyn / total : 0.0;
        const double after_pct = total > 0 ? 100.0 * after_dyn / total : 0.0;
        const bool pass = (before_pct <= kLateLeaverBeforeCeilingPct) && (after_pct >= kLateLeaverAfterFloorPct);
        GateResult g{ "late_leaver", after_pct, kLateLeaverAfterFloorPct, pass,
                     "pedestrian removal rate before/after it leaves" };
        std::printf("[info] late_leaver: pedestrian removal rate BEFORE it leaves (scans 0-%d only) "
                    "= %.1f%% (ceiling %.1f%%); AFTER it leaves (all %d scans) = %.1f%% (floor %.1f%%); "
                    "%d ground-truth pedestrian points\n",
                    kLateLeaverScans - 1, before_pct, static_cast<double>(kLateLeaverBeforeCeilingPct),
                    kNumScans, after_pct, static_cast<double>(kLateLeaverAfterFloorPct), total);
        gates.push_back(g);
    }

    // ---- static_preservation: static (WALL+POLE+WALL_EDGE) points falsely removed ----
    {
        int total = 0, falsely_removed = 0;
        int pole_total = 0, pole_bad = 0, edge_total = 0, edge_bad = 0;
        for (int i = 0; i < bd.n; ++i) {
            if (bd.is_hit[static_cast<size_t>(i)] != 1) continue;
            const int c = bd.cohort[static_cast<size_t>(i)];
            if (c != kCohortWall && c != kCohortPole && c != kCohortWallEdge) continue;
            ++total;
            const bool bad = (label[static_cast<size_t>(i)] == 1);
            if (bad) ++falsely_removed;
            if (c == kCohortPole) { ++pole_total; if (bad) ++pole_bad; }
            if (c == kCohortWallEdge) { ++edge_total; if (bad) ++edge_bad; }
        }
        const double pct = total > 0 ? 100.0 * falsely_removed / total : 0.0;
        const double pole_pct = pole_total > 0 ? 100.0 * pole_bad / pole_total : 0.0;
        const double edge_pct = edge_total > 0 ? 100.0 * edge_bad / edge_total : 0.0;
        GateResult g{ "static_preservation", pct, kStaticPreservationCeilingPct,
                     pct <= kStaticPreservationCeilingPct, "static points falsely removed (overall)" };
        std::printf("[info] static_preservation: %d/%d static points (wall+pole+wall_edge) falsely "
                    "removed (%.1f%%), ceiling %.1f%%\n", falsely_removed, total, pct,
                    static_cast<double>(kStaticPreservationCeilingPct));
        std::printf("[info] static_preservation discretization honesty: thin_pole %d/%d falsely removed "
                    "(%.1f%%); wall_edge %d/%d falsely removed (%.1f%%) - reported, not individually "
                    "gated (grazing-incidence discretization, THEORY.md)\n",
                    pole_bad, pole_total, pole_pct, edge_bad, edge_total, edge_pct);
        gates.push_back(g);
    }

    // ---- free_space_consistency: two exact accounting invariants ----------------
    {
        int violations = 0;
        for (int i = 0; i < bd.n; ++i) {
            if (label[static_cast<size_t>(i)] != 0) continue;   // only RETAINED (static-classified) points
            const Vec3f p = beam_point(bd, pose, i);
            int ix, iy, iz;
            world_to_voxel(p.x, p.y, p.z, ix, iy, iz);
            if (!voxel_in_bounds(ix, iy, iz) || h_hits[static_cast<size_t>(voxel_index(ix, iy, iz))] == 0u)
                ++violations;
        }
        long long total_hits = 0;
        for (int v = 0; v < kNumVoxels; ++v) total_hits += h_hits[static_cast<size_t>(v)];
        const bool ledger_exact = (total_hits == static_cast<long long>(n_hit_beams));
        const bool pass = (violations == 0) && ledger_exact;
        GateResult g{ "free_space_consistency", static_cast<double>(violations), 0.0, pass,
                     "no hits==0 voxel holds a retained point; sum(hits)==hit-beam count" };
        std::printf("[info] free_space_consistency: %d retained-point violation(s); sum(hits)=%lld vs "
                    "hit-beam count=%d (%s)\n", violations, total_hits, n_hit_beams,
                    ledger_exact ? "exact" : "MISMATCH");
        gates.push_back(g);
    }

    // ---- max_range_carving: the isolated ghost, carved ONLY by max-range beams ----
    {
        int ghost_total = 0, ghost_removed = 0;
        unsigned int voxel_pass_hit = 0, voxel_pass_max = 0, voxel_hits = 0;
        for (int i = 0; i < bd.n; ++i) {
            if (bd.is_hit[static_cast<size_t>(i)] == 1 && bd.cohort[static_cast<size_t>(i)] == kCohortGhost) {
                ++ghost_total;
                if (label[static_cast<size_t>(i)] == 1) ++ghost_removed;
                const Vec3f p = beam_point(bd, pose, i);
                int ix, iy, iz;
                world_to_voxel(p.x, p.y, p.z, ix, iy, iz);
                if (voxel_in_bounds(ix, iy, iz)) {
                    const int v = voxel_index(ix, iy, iz);
                    voxel_pass_hit = h_pass_hit[static_cast<size_t>(v)];
                    voxel_pass_max = h_pass_max[static_cast<size_t>(v)];
                    voxel_hits = h_hits[static_cast<size_t>(v)];
                }
            }
        }
        // "Carved (almost) only by max-range beams": the max-range evidence
        // must dominate the hit-beam evidence at this voxel by a wide margin
        // (a strict pass_from_hit == 0 would be fragile against a single
        // grazing hit-beam from an unrelated direction sharing the voxel;
        // README/THEORY document the measured split honestly either way).
        const bool dominated_by_maxrange = voxel_pass_max > 0 && voxel_pass_hit <= voxel_pass_max / 4;
        const bool pass = (ghost_total > 0) && (ghost_removed == ghost_total) && dominated_by_maxrange;
        GateResult g{ "max_range_carving", static_cast<double>(voxel_pass_max), static_cast<double>(voxel_pass_hit),
                     pass, "isolated ghost removed by max-range-beam evidence" };
        std::printf("[info] max_range_carving: %d/%d ghost points classified dynamic; evidence voxel "
                    "pass_from_maxrange=%u, pass_from_hit=%u, hits=%u\n",
                    ghost_removed, ghost_total, voxel_pass_max, voxel_pass_hit, voxel_hits);
        gates.push_back(g);
    }

    // ---- contention ([info] only, no pass/fail) ----------------------------------
    {
        double near_sum = 0.0; long long near_n = 0; unsigned int near_max = 0;
        double far_sum = 0.0; long long far_n = 0;
        for (int iz = 0; iz < kGridNZ; ++iz)
            for (int iy = 0; iy < kGridNY; ++iy)
                for (int ix = 0; ix < kGridNX; ++ix) {
                    const int v = voxel_index(ix, iy, iz);
                    const unsigned int total_pass = h_pass_hit[static_cast<size_t>(v)] + h_pass_max[static_cast<size_t>(v)];
                    if (total_pass == 0u) continue;
                    const float wx = kGridOriginX + (static_cast<float>(ix) + 0.5f) * kVoxelSizeM;
                    const float wy = kGridOriginY + (static_cast<float>(iy) + 0.5f) * kVoxelSizeM;
                    const float wz = kGridOriginZ + (static_cast<float>(iz) + 0.5f) * kVoxelSizeM;
                    float min_d = 1e9f;
                    for (int s = 0; s < kNumScans; ++s) {
                        const float dx = wx - pose.origin[static_cast<size_t>(s) * 3 + 0];
                        const float dy = wy - pose.origin[static_cast<size_t>(s) * 3 + 1];
                        const float dz = wz - pose.origin[static_cast<size_t>(s) * 3 + 2];
                        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (d < min_d) min_d = d;
                    }
                    if (min_d < kContentionNearM) {
                        near_sum += total_pass; ++near_n;
                        if (total_pass > near_max) near_max = total_pass;
                    } else if (min_d > kContentionFarM) {
                        far_sum += total_pass; ++far_n;
                    }
                }
        const double near_avg = near_n > 0 ? near_sum / static_cast<double>(near_n) : 0.0;
        const double far_avg = far_n > 0 ? far_sum / static_cast<double>(far_n) : 0.0;
        std::printf("[info] contention: near-sensor (<%.0fm, %lld voxels) avg passes/voxel=%.1f, peak=%u; "
                    "far (>%.0fm, %lld voxels) avg passes/voxel=%.2f; near/far ratio=%.1fx (atomic "
                    "contention hotspot, measured not asserted)\n",
                    static_cast<double>(kContentionNearM), near_n, near_avg, near_max,
                    static_cast<double>(kContentionFarM), far_n, far_avg,
                    far_avg > 0.0 ? near_avg / far_avg : 0.0);
    }

    // ======================= ARTIFACTS ==============================================
    const std::string out_dir = resolve_out_dir(argv[0]);

    const std::string triptych_path = out_dir + "/triptych.ppm";
    const bool triptych_ok = write_triptych(triptych_path, bd, pose, label);
    std::printf("ARTIFACT: %s demo/out/triptych.ppm (raw / cleaned / truth-static top view)\n",
                triptych_ok ? "wrote" : "FAILED to write");

    std::string ped_csv_path = out_dir + "/pedestrian_evidence.csv";
    bool ped_csv_ok = false;
    {
        std::ofstream f(ped_csv_path);
        if (f.is_open()) {
            f << "scan_id,hits,pass_from_hit,pass_from_maxrange,score\n";
            for (int k = 0; k < kNumScans; ++k) {
                const auto& e = ped_evidence_by_scan[static_cast<size_t>(k)];
                const unsigned int total = e[0] + e[1] + e[2];
                const double sc = total > 0 ? static_cast<double>(e[1] + e[2]) / total : 0.0;
                f << k << ',' << e[0] << ',' << e[1] << ',' << e[2] << ',' << sc << '\n';
            }
            ped_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/pedestrian_evidence.csv (%d scans, voxel evidence accumulation)\n",
                ped_csv_ok ? "wrote" : "FAILED to write", kNumScans);

    std::string gates_csv_path = out_dir + "/gates_metrics.csv";
    bool gates_csv_ok = false;
    {
        std::ofstream f(gates_csv_path);
        if (f.is_open()) {
            f << "gate,measured,threshold,verdict,note\n";
            for (const auto& g : gates)
                f << g.name << ',' << g.measured << ',' << g.threshold << ',' << (g.pass ? "PASS" : "FAIL")
                  << ',' << g.note << '\n';
            gates_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/gates_metrics.csv (%d gates)\n",
                gates_csv_ok ? "wrote" : "FAILED to write", static_cast<int>(gates.size()));

    // ======================= FINAL VERDICT ===========================================
    for (const auto& g : gates)
        std::printf("GATE: %s %s\n", g.name.c_str(), g.pass ? "PASS" : "FAIL");

    bool all_gates_pass = true;
    for (const auto& g : gates) all_gates_pass = all_gates_pass && g.pass;
    const bool artifacts_ok = triptych_ok && ped_csv_ok && gates_csv_ok;
    const bool success = all_verify_pass && all_gates_pass && artifacts_ok;

    if (success)
        std::printf("RESULT: PASS (all verify stages and gates passed; map cleaned of ghosts)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above and stderr)\n");
    return success ? 0 : 1;
}
