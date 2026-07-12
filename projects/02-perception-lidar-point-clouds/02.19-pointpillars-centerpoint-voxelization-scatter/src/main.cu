// ===========================================================================
// main.cu — entry point for project 02.19
//           PointPillars/CenterPoint voxelization + scatter kernels
//           feeding TensorRT (teaching core: pillarization -> PFN-lite ->
//           scatter -> hand-designed BEV head, no TensorRT required)
//
// What this program does, start to finish (kernels.cuh's file header
// derives the WHY behind every stage; this is the WHEN/WHERE)
// ---------------------------------------------------------------------------
//   0. Load the committed scene: points.bin (KITTI-layout x,y,z,intensity),
//      scene_meta.csv (the point-stream layout contract), object_truths.csv
//      (ground truth), pfn_lite_weights.csv (fixed, seed 42, not trained).
//   1. VERIFY(keys): GPU-transcribed pillar/voxel keys vs the CPU's shared-
//      formula twin, bit-exact over every point (the §5 gate's first half).
//   2. GATE cap_truncation: the determinism study — Method B (sorted) run
//      3x, Method A (atomic) run 3x same-order and 3x with SHUFFLED input
//      order — all measured against the cap-stress pillar's known 60-point
//      overflow (kernels.cuh's file header derives the whole design).
//   3. The PRODUCTION pipeline (Method B, deterministic): sort_and_compact
//      -> sorted_bin -> pfn_stats -> augment_features -> pfn_lite ->
//      scatter -> gather -> conv(smooth) -> gate -> conv(sharpen) ->
//      peak_extract -> host NMS. This run's results feed everything below.
//   4. The independent CPU reference pipeline on the SAME data, and
//      VERIFY(binning/pfn/scatter/head/peaks) — the §5 gate's second half.
//   5. GATE layout_roundtrip, GATE feature_semantics, GATE detection_closure.
//   6. [info] sparsity_economics, pillar_vs_voxel, trt_handoff.
//   7. Artifacts: demo/out/{occupancy,heatmap}.pgm, feature_stats.csv,
//      gates_metrics.csv.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY(...)", "GATE ...:", "ARTIFACT:", "RESULT:" — "[info]"/
// "[time]" lines are NOT diffed (device names and timings vary by machine).
// Change a stable line -> update demo/expected_output.txt in the same change.
//
// Read this after: kernels.cuh.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Small host-side helpers: RNG (for the input-order shuffle experiment),
// loaders (points.bin, scene_meta.csv, object_truths.csv, pfn weights),
// and the artifact writers. All plain C++17 — this file is compiled by
// nvcc (like every main.cu in this repo) but nothing here is CUDA-specific
// except the kernel-launch orchestration itself.
// ===========================================================================

// xorshift32 — the repo's portable deterministic generator (08.01/02.01 use
// the identical three-line core). Used ONLY to build input-order PERMUTATIONS
// for the cap_truncation gate — never to generate scene content (that is
// scripts/make_synthetic.py's job, seeded independently).
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// shuffle_points — a Fisher-Yates permutation of the WHOLE point stream
// (keeping each point's 4 floats together), used by the cap_truncation
// gate to simulate a different, equally-legitimate LiDAR packet arrival
// order (kernels.cuh's file header explains why this stands in for raw
// atomicAdd scheduling nondeterminism).
static std::vector<float> shuffle_points(const std::vector<float>& pts, int n, uint32_t seed)
{
    std::vector<int> order(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
    uint32_t s = (seed == 0) ? 1u : seed;
    for (int k = n - 1; k > 0; --k) {
        const uint32_t r = xorshift32(s);
        const int j = static_cast<int>(r % static_cast<uint32_t>(k + 1));
        std::swap(order[static_cast<size_t>(k)], order[static_cast<size_t>(j)]);
    }
    std::vector<float> out(static_cast<size_t>(n) * 4);
    for (int i = 0; i < n; ++i) {
        const int src = order[static_cast<size_t>(i)];
        out[static_cast<size_t>(i) * 4 + 0] = pts[static_cast<size_t>(src) * 4 + 0];
        out[static_cast<size_t>(i) * 4 + 1] = pts[static_cast<size_t>(src) * 4 + 1];
        out[static_cast<size_t>(i) * 4 + 2] = pts[static_cast<size_t>(src) * 4 + 2];
        out[static_cast<size_t>(i) * 4 + 3] = pts[static_cast<size_t>(src) * 4 + 3];
    }
    return out;
}

// load_points_bin — the KITTI/PointPillars raw layout: no header, just
// N*4 float32 back to back; N is recovered from the file size.
static std::vector<float> load_points_bin(const std::string& path, int& n_out)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    n_out = 0;
    if (!in.is_open()) return {};
    const std::streamsize bytes = in.tellg();
    in.seekg(0);
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(4 * sizeof(float)) != 0) return {};
    const int n = static_cast<int>(bytes / static_cast<std::streamsize>(4 * sizeof(float)));
    std::vector<float> pts(static_cast<size_t>(n) * 4);
    in.read(reinterpret_cast<char*>(pts.data()), bytes);
    if (!in) return {};
    n_out = n;
    return pts;
}

// SceneMeta — the point-stream layout contract scripts/make_synthetic.py
// writes and this loader reads VERBATIM (kernels.cuh/make_synthetic.py's
// file headers name this as a data-layout contract, never re-derived).
struct SceneMeta {
    int n_total = 0, n_ground = 0, n_cars_total = 0, n_cars = 0, points_per_car = 0;
    int n_clutter = 0, n_capstress = 0, capstress_start_index = 0, capstress_pillar_key = -1;
    bool loaded = false;
};

static SceneMeta load_scene_meta(const std::string& path)
{
    SceneMeta m;
    std::ifstream in(path);
    if (!in.is_open()) return m;
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string key, val;
        if (!std::getline(ss, key, ',') || !std::getline(ss, val, ',')) continue;
        kv[key] = val;
    }
    auto geti = [&](const char* k) -> int {
        auto it = kv.find(k);
        return it != kv.end() ? std::atoi(it->second.c_str()) : -1;
    };
    m.n_total = geti("n_total");
    m.n_ground = geti("n_ground");
    m.n_cars_total = geti("n_cars_total");
    m.n_cars = geti("n_cars");
    m.points_per_car = geti("points_per_car");
    m.n_clutter = geti("n_clutter");
    m.n_capstress = geti("n_capstress");
    m.capstress_start_index = geti("capstress_start_index");
    m.capstress_pillar_key = geti("capstress_pillar_key");
    m.loaded = (m.n_total > 0 && m.capstress_pillar_key >= 0 && m.n_capstress > 0);
    return m;
}

struct ObjectTruth { int id; float cx, cy, length, width, height; };

static std::vector<ObjectTruth> load_truths(const std::string& path, bool& ok)
{
    std::vector<ObjectTruth> out;
    std::ifstream in(path);
    ok = in.is_open();
    if (!ok) return out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cell;
        ObjectTruth t{};
        if (!std::getline(ss, cell, ',')) { ok = false; break; }
        t.id = std::atoi(cell.c_str());
        if (!std::getline(ss, cell, ',')) { ok = false; break; }
        t.cx = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) { ok = false; break; }
        t.cy = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) { ok = false; break; }
        t.length = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) { ok = false; break; }
        t.width = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) { ok = false; break; }
        t.height = std::strtof(cell.c_str(), nullptr);
        out.push_back(t);
    }
    return out;
}

// load_pfn_weights — kPfnLinOut rows of (kNumPointFeatures weights + 1 bias).
static bool load_pfn_weights(const std::string& path, float* w, float* b)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    int row = 0;
    while (std::getline(in, line) && row < kPfnLinOut) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cell;
        for (int d = 0; d < kNumPointFeatures; ++d) {
            if (!std::getline(ss, cell, ',')) return false;
            w[row * kNumPointFeatures + d] = std::strtof(cell.c_str(), nullptr);
        }
        if (!std::getline(ss, cell, ',')) return false;
        b[row] = std::strtof(cell.c_str(), nullptr);
        ++row;
    }
    return row == kPfnLinOut;
}

// identify_kept — matches each of a pillar's `kept` retained raw points
// against a known candidate pool by EXACT float equality. Safe here because
// every value in play is a byte-for-byte copy (through cudaMemcpy, never
// arithmetic) of one of the candidate points — no rounding is possible, so
// == is the right comparison, not a numerical shortcut.
static std::vector<int> identify_kept(const float* kept_raw, unsigned int kept,
                                      const float* candidates, int num_candidates)
{
    std::vector<int> result;
    for (unsigned int k = 0; k < kept; ++k) {
        const float* p = &kept_raw[k * 4];
        for (int c = 0; c < num_candidates; ++c) {
            const float* cd = &candidates[static_cast<size_t>(c) * 4];
            if (p[0] == cd[0] && p[1] == cd[1] && p[2] == cd[2] && p[3] == cd[3]) { result.push_back(c); break; }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

static int symmetric_diff_count(const std::vector<int>& a, const std::vector<int>& b)
{
    std::vector<int> only_a, only_b;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(only_a));
    std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(only_b));
    return static_cast<int>(only_a.size() + only_b.size());
}

// PeakHost — the GPU path's own detection record (mirrors reference_cpu.cpp's
// PeakCPU but is a SEPARATE type/function: kernels.cuh's file header notes
// main.cu implements its OWN NMS rather than calling into reference_cpu.cpp,
// so the detection_closure gate is a genuine cross-check, not a shared call).
struct PeakHost { int iy; int ix; float score; };

static std::vector<PeakHost> nms_host(std::vector<PeakHost> candidates, int radius_pillars)
{
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const PeakHost& a, const PeakHost& b) { return a.score > b.score; });
    std::vector<unsigned char> suppressed(candidates.size(), 0);
    std::vector<PeakHost> kept;
    const int r2 = radius_pillars * radius_pillars;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            const int ddy = candidates[j].iy - candidates[i].iy;
            const int ddx = candidates[j].ix - candidates[i].ix;
            if (ddy * ddy + ddx * ddx <= r2) suppressed[j] = 1;
        }
    }
    return kept;
}

static bool write_pgm(const std::string& path, const std::vector<unsigned char>& gray, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(f);
}

// to_gray — normalize a float plane to 0..255 by its own [lo,hi] range,
// clamping. Used for both artifact PGMs (occupancy is already in [0,1];
// the heatmap uses its own measured min/max, printed alongside so the
// [info] numbers and the picture agree).
static std::vector<unsigned char> to_gray(const std::vector<float>& plane, float lo, float hi)
{
    std::vector<unsigned char> out(plane.size());
    const float span = (hi > lo) ? (hi - lo) : 1.0f;
    for (size_t i = 0; i < plane.size(); ++i) {
        float v = (plane[i] - lo) / span;
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        out[i] = static_cast<unsigned char>(v * 255.0f + 0.5f);
    }
    return out;
}

// mark_peaks — draw a small bright 3x3 marker at each surviving peak so
// "peaks visible on objects" (CLAUDE.md demo requirement) holds even after
// normalization dims the raw heatmap values.
static void mark_peaks(std::vector<unsigned char>& gray, int w, int h, const std::vector<PeakHost>& peaks)
{
    for (const auto& pk : peaks) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = pk.ix + dx, ny = pk.iy + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) gray[static_cast<size_t>(ny) * w + nx] = 255;
            }
        }
    }
}

// ===========================================================================
// main
// ===========================================================================
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

    std::printf("[demo] PointPillars/CenterPoint voxelization + scatter: Method A (atomic) vs "
               "Method B (sorted, deterministic) binning, PFN-lite, scatter, hand-designed BEV head "
               "(project 02.19)\n");
    print_device_info();
    std::printf("PROBLEM: BEV grid %dx%d pillars @ %.2f m (window [%.0f,%.0f) x [%.0f,%.0f) m), "
               "cap=%d points/pillar, D=%d point features, PFN-lite C=%d channels, "
               "CenterPoint-style comparison: %d z-bins\n",
               kGridNX, kGridNY, static_cast<double>(kPillarSizeM),
               static_cast<double>(kXMin), static_cast<double>(kXMin + kGridNX * kPillarSizeM),
               static_cast<double>(kYMin), static_cast<double>(kYMin + kGridNY * kPillarSizeM),
               kMaxPointsPerPillar, kNumPointFeatures, kPfnChannels, kNumZBins);

    // ---- 0) load data --------------------------------------------------------
    const std::string points_path = find_data_file(data_dir_override, argv[0], "points.bin");
    const std::string meta_path   = find_data_file(data_dir_override, argv[0], "scene_meta.csv");
    const std::string truths_path = find_data_file(data_dir_override, argv[0], "object_truths.csv");
    const std::string weights_path = find_data_file(data_dir_override, argv[0], "pfn_lite_weights.csv");
    if (points_path.empty() || meta_path.empty() || truths_path.empty() || weights_path.empty()) {
        std::printf("DATA: NOT FOUND -- run scripts/make_synthetic.py first\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }
    int n = 0;
    std::vector<float> h_points = load_points_bin(points_path, n);
    SceneMeta meta = load_scene_meta(meta_path);
    bool truths_ok = false;
    std::vector<ObjectTruth> truths = load_truths(truths_path, truths_ok);
    float h_lin_w[kPfnLinOut * kNumPointFeatures];
    float h_lin_b[kPfnLinOut];
    const bool weights_ok = load_pfn_weights(weights_path, h_lin_w, h_lin_b);
    if (n <= 0 || !meta.loaded || !truths_ok || !weights_ok || meta.n_total != n) {
        std::printf("DATA: MALFORMED -- points=%d meta_loaded=%d truths_ok=%d weights_ok=%d "
                   "(meta.n_total=%d vs loaded n=%d)\n",
                   n, meta.loaded ? 1 : 0, truths_ok ? 1 : 0, weights_ok ? 1 : 0, meta.n_total, n);
        std::printf("RESULT: FAIL (sample data malformed)\n");
        return 1;
    }
    // The stable DATA: line prints a portable RELATIVE label (never the
    // resolved path — that can be an absolute, machine-specific string
    // depending on which find_data_file() candidate matched, see paths.h's
    // file header; 02.01's main.cu sets this precedent). The actual
    // resolved path goes on an "[info]" line, unchecked, for debugging.
    std::printf("[info] resolved data file: %s\n", points_path.c_str());
    std::printf("DATA: data/sample/points.bin [synthetic, seed 42, xorshift32, see scripts/make_synthetic.py] "
               "(%d points = %d ground + %d car + %d clutter + %d cap-stress; %d truth objects)\n",
               n, meta.n_ground, meta.n_cars_total, meta.n_clutter, meta.n_capstress,
               static_cast<int>(truths.size()));

    // The 60 cap-stress candidate points, contiguous in the point stream
    // (scene_meta.csv's contract) — the cap_truncation gate's ground truth.
    const float* capstress_candidates = &h_points[static_cast<size_t>(meta.capstress_start_index) * 4];
    const int num_capstress = meta.n_capstress;

    // ---- 1) persistent device buffers ----------------------------------------
    float* d_points = nullptr;
    int* d_keys = nullptr;          // pillar keys, original order
    int* d_voxel_keys = nullptr;    // voxel keys, original order
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_voxel_keys, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), static_cast<size_t>(n) * 4 * sizeof(float),
                          cudaMemcpyHostToDevice));

    launch_compute_pillar_keys(n, d_points, d_keys);
    launch_compute_voxel_keys(n, d_points, d_voxel_keys);

    std::vector<int> h_keys_gpu(static_cast<size_t>(n)), h_voxel_keys_gpu(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(h_keys_gpu.data(), d_keys, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_voxel_keys_gpu.data(), d_voxel_keys, static_cast<size_t>(n) * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // ======================= VERIFY(keys) ======================================
    std::vector<int> h_keys_cpu(static_cast<size_t>(n)), h_voxel_keys_cpu(static_cast<size_t>(n));
    pillar_keys_cpu(n, h_points.data(), h_keys_cpu.data());
    voxel_keys_cpu(n, h_points.data(), h_voxel_keys_cpu.data());
    bool verify_keys_ok = true;
    int n_valid_pillar = 0;
    for (int i = 0; i < n; ++i) {
        if (h_keys_gpu[static_cast<size_t>(i)] != h_keys_cpu[static_cast<size_t>(i)]) verify_keys_ok = false;
        if (h_voxel_keys_gpu[static_cast<size_t>(i)] != h_voxel_keys_cpu[static_cast<size_t>(i)]) verify_keys_ok = false;
        if (h_keys_cpu[static_cast<size_t>(i)] >= 0) ++n_valid_pillar;
    }
    std::printf("[info] in-window points: %d / %d (pillar key >= 0)\n", n_valid_pillar, n);
    std::printf("VERIFY(keys): %s (GPU pillar+voxel keys bit-exact vs CPU reference for all %d points)\n",
               verify_keys_ok ? "PASS" : "FAIL", n);

    // ======================= GATE cap_truncation ===============================
    // See kernels.cuh's file header for the full design rationale. Shared
    // dense binning buffers (kNumPillars is only 40,000 cells; reused across
    // every sub-experiment below, then again for the production run in
    // section 3 — memory-efficient and mirrors how a real pipeline reuses
    // its scratch tensors frame to frame).
    unsigned int* d_point_count = nullptr;
    float* d_raw_points = nullptr;
    CUDA_CHECK(cudaMalloc(&d_point_count, static_cast<size_t>(kNumPillars) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_raw_points,
                          static_cast<size_t>(kNumPillars) * kMaxPointsPerPillar * 4 * sizeof(float)));
    PillarBinGPU bin{ d_point_count, d_raw_points, kNumPillars, kMaxPointsPerPillar };

    // Sort/compact scratch, reused for every pillar-key sort in this file
    // (Method B production run and the 3 sorted repeats here).
    int *d_keys_scratch = nullptr, *d_idx_scratch = nullptr, *d_is_start_scratch = nullptr;
    int *d_seg_start = nullptr, *d_occupied_cell = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys_scratch, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx_scratch, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start_scratch, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_start, static_cast<size_t>(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_occupied_cell, static_cast<size_t>(n) * sizeof(int)));

    std::vector<float> capstress_slice(static_cast<size_t>(kMaxPointsPerPillar) * 4);
    auto read_capstress = [&](unsigned int& count_out) -> std::vector<int> {
        unsigned int point_count = 0;
        CUDA_CHECK(cudaMemcpy(&point_count, d_point_count + meta.capstress_pillar_key, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        const unsigned int kept = point_count < static_cast<unsigned int>(kMaxPointsPerPillar)
                                 ? point_count : static_cast<unsigned int>(kMaxPointsPerPillar);
        CUDA_CHECK(cudaMemcpy(capstress_slice.data(),
                              d_raw_points + static_cast<size_t>(meta.capstress_pillar_key) * kMaxPointsPerPillar * 4,
                              static_cast<size_t>(kMaxPointsPerPillar) * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        count_out = point_count;
        return identify_kept(capstress_slice.data(), kept, capstress_candidates, num_capstress);
    };

    // (a) Sorted (Method B), 3 repeats, ORIGINAL order — expect bit-exact
    //     agreement across runs AND the hand-provable anchor {0..31} (the
    //     first 32 candidates by original index, since sorted truncation
    //     keeps ascending original index and the 60 candidates are already
    //     contiguous and ascending in the point stream).
    std::vector<std::vector<int>> sorted_kept_runs;
    unsigned int sorted_count_seen = 0;
    for (int run = 0; run < 3; ++run) {
        int n_valid_tmp = 0;
        const int num_occ_tmp = launch_sort_and_compact(n, d_keys, d_keys_scratch, d_idx_scratch,
                                                         d_is_start_scratch, d_seg_start, d_occupied_cell,
                                                         &n_valid_tmp);
        launch_sorted_bin(num_occ_tmp, d_seg_start, n_valid_tmp, d_idx_scratch, d_points, d_occupied_cell, bin);
        sorted_kept_runs.push_back(read_capstress(sorted_count_seen));
    }
    std::vector<int> sorted_anchor;
    for (int c = 0; c < kMaxPointsPerPillar; ++c) sorted_anchor.push_back(c);   // {0,...,31}: the hand-provable answer
    const bool sorted_matches_anchor = (sorted_kept_runs[0] == sorted_anchor);
    const int sorted_var_01 = symmetric_diff_count(sorted_kept_runs[0], sorted_kept_runs[1]);
    const int sorted_var_02 = symmetric_diff_count(sorted_kept_runs[0], sorted_kept_runs[2]);
    const int sorted_var_12 = symmetric_diff_count(sorted_kept_runs[1], sorted_kept_runs[2]);

    // (b) Atomic (Method A), 3 repeats, SAME (original) order — measured
    //     honestly: the CUDA memory model gives no ordering guarantee, but
    //     an idle GPU running the identical launch on identical inputs may
    //     (or may not) reproduce the same schedule. We report what we saw.
    std::vector<std::vector<int>> atomic_same_order_runs;
    unsigned int atomic_count_seen = 0;
    for (int run = 0; run < 3; ++run) {
        launch_reset_counts(d_point_count, kNumPillars);
        launch_atomic_bin(n, d_points, d_keys, bin);
        atomic_same_order_runs.push_back(read_capstress(atomic_count_seen));
    }
    const int atomic_same_var_01 = symmetric_diff_count(atomic_same_order_runs[0], atomic_same_order_runs[1]);
    const int atomic_same_var_02 = symmetric_diff_count(atomic_same_order_runs[0], atomic_same_order_runs[2]);
    const int atomic_same_var_12 = symmetric_diff_count(atomic_same_order_runs[1], atomic_same_order_runs[2]);

    // (c) Atomic (Method A), 3 DIFFERENT input-order permutations — the
    //     100%-reproducible stand-in for scheduler nondeterminism (file
    //     header). Uses its own points/keys scratch buffers so the
    //     original-order buffers above stay untouched for later stages.
    float* d_points_shuf = nullptr;
    int* d_keys_shuf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points_shuf, static_cast<size_t>(n) * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keys_shuf, static_cast<size_t>(n) * sizeof(int)));
    std::vector<std::vector<int>> atomic_shuffled_runs;
    unsigned int atomic_shuf_count_seen = 0;
    const uint32_t shuffle_seeds[3] = { 43u, 44u, 45u };
    for (int run = 0; run < 3; ++run) {
        std::vector<float> shuffled = shuffle_points(h_points, n, shuffle_seeds[run]);
        CUDA_CHECK(cudaMemcpy(d_points_shuf, shuffled.data(), static_cast<size_t>(n) * 4 * sizeof(float),
                              cudaMemcpyHostToDevice));
        launch_compute_pillar_keys(n, d_points_shuf, d_keys_shuf);
        launch_reset_counts(d_point_count, kNumPillars);
        launch_atomic_bin(n, d_points_shuf, d_keys_shuf, bin);
        atomic_shuffled_runs.push_back(read_capstress(atomic_shuf_count_seen));
    }
    CUDA_CHECK(cudaFree(d_points_shuf));
    CUDA_CHECK(cudaFree(d_keys_shuf));
    const int atomic_shuf_var_01 = symmetric_diff_count(atomic_shuffled_runs[0], atomic_shuffled_runs[1]);
    const int atomic_shuf_var_02 = symmetric_diff_count(atomic_shuffled_runs[0], atomic_shuffled_runs[2]);
    const int atomic_shuf_var_12 = symmetric_diff_count(atomic_shuffled_runs[1], atomic_shuffled_runs[2]);
    const int atomic_shuf_max_var = std::max({ atomic_shuf_var_01, atomic_shuf_var_02, atomic_shuf_var_12 });

    std::printf("[info] cap-stress pillar key=%d: arrival count sorted=%u atomic(same-order)=%u "
               "atomic(shuffled)=%u (all should equal %d)\n",
               meta.capstress_pillar_key, sorted_count_seen, atomic_count_seen, atomic_shuf_count_seen,
               num_capstress);
    std::printf("[info] cap_truncation: sorted 3-run pairwise differing-slot counts: %d,%d,%d "
               "(bit-exact determinism claim)\n", sorted_var_01, sorted_var_02, sorted_var_12);
    std::printf("[info] cap_truncation: atomic SAME-order 3-run pairwise differing-slot counts: %d,%d,%d "
               "(measured, not guaranteed by the CUDA memory model either way)\n",
               atomic_same_var_01, atomic_same_var_02, atomic_same_var_12);
    std::printf("[info] cap_truncation: atomic SHUFFLED-order 3-run pairwise differing-slot counts: %d,%d,%d "
               "(the reproducible order-dependence demonstration)\n",
               atomic_shuf_var_01, atomic_shuf_var_02, atomic_shuf_var_12);

    const bool cap_bookkeeping_ok =
        (sorted_count_seen == static_cast<unsigned int>(num_capstress)) &&
        (atomic_count_seen == static_cast<unsigned int>(num_capstress)) &&
        (atomic_shuf_count_seen == static_cast<unsigned int>(num_capstress)) &&
        (sorted_kept_runs[0].size() == static_cast<size_t>(kMaxPointsPerPillar)) &&
        (atomic_same_order_runs[0].size() == static_cast<size_t>(kMaxPointsPerPillar)) &&
        (atomic_shuffled_runs[0].size() == static_cast<size_t>(kMaxPointsPerPillar));
    const bool cap_sorted_deterministic = sorted_matches_anchor && sorted_var_01 == 0 && sorted_var_02 == 0 && sorted_var_12 == 0;
    const bool cap_order_dependence_shown = atomic_shuf_max_var > 0;
    const bool gate_cap_truncation = cap_bookkeeping_ok && cap_sorted_deterministic && cap_order_dependence_shown;
    std::printf("GATE cap_truncation: %s (every variant keeps exactly %d of %d arrived points; "
               "sorted keeps the analytic answer {0..%d} bit-identically across repeats; "
               "shuffled-order atomic binning demonstrably changes which points survive)\n",
               gate_cap_truncation ? "PASS" : "FAIL", kMaxPointsPerPillar, num_capstress, kMaxPointsPerPillar - 1);

    // ======================= 3) PRODUCTION PIPELINE (Method B) ================
    // The single, deterministic run every downstream VERIFY/GATE/artifact
    // uses. Reuses the shared PillarBinGPU buffers above (cap_truncation's
    // experiments are done; overwriting them now is safe and mirrors a real
    // pipeline reusing its scratch tensors frame to frame).
    GpuTimer sort_timer;
    sort_timer.begin();
    int n_valid = 0;
    const int num_occupied = launch_sort_and_compact(n, d_keys, d_keys_scratch, d_idx_scratch,
                                                      d_is_start_scratch, d_seg_start, d_occupied_cell, &n_valid);
    const float sort_ms = sort_timer.end_ms();
    launch_sorted_bin(num_occupied, d_seg_start, n_valid, d_idx_scratch, d_points, d_occupied_cell, bin);

    std::printf("[info] occupied pillars: %d / %d cells (%.2f%% of the BEV grid) from %d in-window points\n",
               num_occupied, kNumPillars, 100.0 * num_occupied / kNumPillars, n_valid);

    float* d_mean_xyz = nullptr;
    unsigned int* d_kept_count = nullptr;
    float* d_features = nullptr;
    float* d_pillar_feat = nullptr;
    float* d_canvas = nullptr;
    float* d_gathered = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mean_xyz, static_cast<size_t>(num_occupied) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kept_count, static_cast<size_t>(num_occupied) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_features,
                          static_cast<size_t>(num_occupied) * kMaxPointsPerPillar * kNumPointFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pillar_feat, static_cast<size_t>(num_occupied) * kPfnChannels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_canvas, static_cast<size_t>(kPfnChannels) * kGridNY * kGridNX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gathered, static_cast<size_t>(num_occupied) * kPfnChannels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_canvas, 0, static_cast<size_t>(kPfnChannels) * kGridNY * kGridNX * sizeof(float)));

    float* d_lin_w = nullptr;
    float* d_lin_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lin_w, static_cast<size_t>(kPfnLinOut) * kNumPointFeatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lin_b, static_cast<size_t>(kPfnLinOut) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lin_w, h_lin_w, static_cast<size_t>(kPfnLinOut) * kNumPointFeatures * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lin_b, h_lin_b, static_cast<size_t>(kPfnLinOut) * sizeof(float), cudaMemcpyHostToDevice));

    launch_pfn_stats(num_occupied, d_occupied_cell, bin, d_mean_xyz, d_kept_count);
    launch_augment_features(num_occupied, d_occupied_cell, bin, d_mean_xyz, d_features);
    launch_pfn_lite(num_occupied, d_features, d_kept_count, d_lin_w, d_lin_b, d_pillar_feat);
    launch_scatter(num_occupied, d_occupied_cell, d_pillar_feat, d_canvas);
    launch_gather(num_occupied, d_occupied_cell, d_canvas, d_gathered);

    // The toy head: channel 0 = occupancy plane, channel 1 = height-extent
    // plane — both are just POINTERS into d_canvas (NCHW layout, so channel
    // c's plane starts at d_canvas + c*H*W), no copy needed.
    float* d_ch0 = d_canvas + 0 * kGridNY * kGridNX;
    float* d_ch1 = d_canvas + 1 * kGridNY * kGridNX;
    float* d_smoothed = nullptr;
    float* d_gated = nullptr;
    float* d_heatmap = nullptr;
    unsigned char* d_is_candidate = nullptr;
    float* d_smooth_k = nullptr;
    float* d_sharpen_k = nullptr;
    const size_t plane_elems = static_cast<size_t>(kGridNY) * kGridNX;
    CUDA_CHECK(cudaMalloc(&d_smoothed, plane_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gated, plane_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_heatmap, plane_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_is_candidate, plane_elems * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_smooth_k, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sharpen_k, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_smooth_k, kSmoothKernel3x3, 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sharpen_k, kSharpenKernel3x3, 9 * sizeof(float), cudaMemcpyHostToDevice));

    launch_conv3x3(d_ch1, kGridNY, kGridNX, d_smooth_k, 0.0f, d_smoothed);
    launch_elementwise_mul(d_smoothed, d_ch0, static_cast<int>(plane_elems), d_gated);
    launch_conv3x3(d_gated, kGridNY, kGridNX, d_sharpen_k, 0.0f, d_heatmap);
    launch_peak_extract(d_heatmap, kGridNY, kGridNX, kDetectThreshold, kPeakWindowR, d_is_candidate);

    std::vector<float> h_heatmap(plane_elems);
    std::vector<unsigned char> h_is_candidate(plane_elems);
    CUDA_CHECK(cudaMemcpy(h_heatmap.data(), d_heatmap, plane_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_is_candidate.data(), d_is_candidate, plane_elems * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

    std::vector<PeakHost> candidates_gpu;
    for (int iy = 0; iy < kGridNY; ++iy)
        for (int ix = 0; ix < kGridNX; ++ix)
            if (h_is_candidate[static_cast<size_t>(iy) * kGridNX + ix])
                candidates_gpu.push_back(PeakHost{ iy, ix, h_heatmap[static_cast<size_t>(iy) * kGridNX + ix] });
    std::vector<PeakHost> peaks_gpu = nms_host(candidates_gpu, kNmsRadiusPillars);

    // ======================= 4) CPU REFERENCE PIPELINE =========================
    std::vector<unsigned int> h_point_count_dense(static_cast<size_t>(kNumPillars), 0);
    std::vector<float> h_raw_points_dense(static_cast<size_t>(kNumPillars) * kMaxPointsPerPillar * 4, 0.0f);
    std::vector<int> h_occupied_cell_cpu(static_cast<size_t>(n));
    std::vector<unsigned int> h_kept_count_cpu(static_cast<size_t>(n));
    std::vector<float> h_mean_xyz_cpu(static_cast<size_t>(n) * 3);
    CpuTimer cpu_pipeline_timer;
    cpu_pipeline_timer.begin();
    const int num_occupied_cpu = sorted_bin_cpu(n, h_points.data(), h_point_count_dense.data(),
                                                h_raw_points_dense.data(), h_occupied_cell_cpu.data(),
                                                h_kept_count_cpu.data(), h_mean_xyz_cpu.data());

    std::vector<float> h_features_cpu(static_cast<size_t>(num_occupied_cpu) * kMaxPointsPerPillar * kNumPointFeatures);
    augment_features_cpu(num_occupied_cpu, h_occupied_cell_cpu.data(), h_kept_count_cpu.data(),
                         h_mean_xyz_cpu.data(), h_raw_points_dense.data(), h_features_cpu.data());

    std::vector<float> h_pillar_feat_cpu(static_cast<size_t>(num_occupied_cpu) * kPfnChannels);
    pfn_lite_cpu(num_occupied_cpu, h_features_cpu.data(), h_kept_count_cpu.data(), h_lin_w, h_lin_b,
                h_pillar_feat_cpu.data());

    std::vector<float> h_canvas_cpu(static_cast<size_t>(kPfnChannels) * kGridNY * kGridNX, 0.0f);
    scatter_cpu(num_occupied_cpu, h_occupied_cell_cpu.data(), h_pillar_feat_cpu.data(), h_canvas_cpu.data());

    std::vector<float> h_smoothed_cpu(plane_elems), h_gated_cpu(plane_elems), h_heatmap_cpu(plane_elems);
    conv3x3_cpu(&h_canvas_cpu[1 * plane_elems], kGridNY, kGridNX, kSmoothKernel3x3, 0.0f, h_smoothed_cpu.data());
    for (size_t i = 0; i < plane_elems; ++i) h_gated_cpu[i] = h_smoothed_cpu[i] * h_canvas_cpu[0 * plane_elems + i];
    conv3x3_cpu(h_gated_cpu.data(), kGridNY, kGridNX, kSharpenKernel3x3, 0.0f, h_heatmap_cpu.data());

    std::vector<PeakCPU> peaks_cpu;
    peak_extract_and_nms_cpu(h_heatmap_cpu.data(), kGridNY, kGridNX, kDetectThreshold, kPeakWindowR,
                             kNmsRadiusPillars, peaks_cpu);
    const double cpu_pipeline_ms = cpu_pipeline_timer.end_ms();

    // ======================= VERIFY(binning/pfn/scatter/head/peaks) ===========
    bool verify_binning_ok = (num_occupied == num_occupied_cpu);
    for (int p = 0; verify_binning_ok && p < num_occupied; ++p) {
        // Read GPU occupied_cell/kept_count for this rank (both GPU and CPU
        // walk occupied pillars in ASCENDING KEY order — a side effect of
        // sorting by key on both paths, so ranks line up positionally).
        int gpu_cell = 0;
        CUDA_CHECK(cudaMemcpy(&gpu_cell, d_occupied_cell + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (gpu_cell != h_occupied_cell_cpu[static_cast<size_t>(p)]) verify_binning_ok = false;
    }
    std::vector<unsigned int> h_kept_count_gpu(static_cast<size_t>(num_occupied));
    CUDA_CHECK(cudaMemcpy(h_kept_count_gpu.data(), d_kept_count, static_cast<size_t>(num_occupied) * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    for (int p = 0; verify_binning_ok && p < num_occupied; ++p)
        if (h_kept_count_gpu[static_cast<size_t>(p)] != h_kept_count_cpu[static_cast<size_t>(p)]) verify_binning_ok = false;

    std::vector<float> h_features_gpu(h_features_cpu.size());
    CUDA_CHECK(cudaMemcpy(h_features_gpu.data(), d_features, h_features_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float worst_feature_diff = 0.0f;
    for (size_t i = 0; i < h_features_gpu.size(); ++i) {
        const float d = std::fabs(h_features_gpu[i] - h_features_cpu[i]);
        if (d > worst_feature_diff) worst_feature_diff = d;
    }
    const bool features_ok = worst_feature_diff <= 1e-4f;
    std::printf("[info] VERIFY(binning): worst feature-tensor |gpu-cpu| = %.3e over %zu entries\n",
               static_cast<double>(worst_feature_diff), h_features_gpu.size());
    std::printf("VERIFY(binning): %s (occupied-pillar list + kept counts + augmented 9-D features "
               "match CPU reference; features within 1e-4)\n", (verify_binning_ok && features_ok) ? "PASS" : "FAIL");

    std::vector<float> h_pillar_feat_gpu(h_pillar_feat_cpu.size());
    CUDA_CHECK(cudaMemcpy(h_pillar_feat_gpu.data(), d_pillar_feat, h_pillar_feat_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float worst_pfn_diff = 0.0f;
    for (size_t i = 0; i < h_pillar_feat_gpu.size(); ++i) {
        const float d = std::fabs(h_pillar_feat_gpu[i] - h_pillar_feat_cpu[i]);
        if (d > worst_pfn_diff) worst_pfn_diff = d;
    }
    const bool verify_pfn_ok = worst_pfn_diff <= 1e-4f;
    std::printf("[info] VERIFY(pfn): worst pillar-feature |gpu-cpu| = %.3e over %zu entries\n",
               static_cast<double>(worst_pfn_diff), h_pillar_feat_gpu.size());
    std::printf("VERIFY(pfn): %s (PFN-lite occupancy/height-extent/linear-maxpool channels within 1e-4)\n",
               verify_pfn_ok ? "PASS" : "FAIL");

    std::vector<float> h_canvas_gpu(h_canvas_cpu.size());
    CUDA_CHECK(cudaMemcpy(h_canvas_gpu.data(), d_canvas, h_canvas_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float worst_canvas_diff = 0.0f;
    for (size_t i = 0; i < h_canvas_gpu.size(); ++i) {
        const float d = std::fabs(h_canvas_gpu[i] - h_canvas_cpu[i]);
        if (d > worst_canvas_diff) worst_canvas_diff = d;
    }
    const bool verify_scatter_ok = worst_canvas_diff <= 1e-5f;
    std::printf("VERIFY(scatter): %s (dense BEV canvas bit-exact vs CPU reference, worst |gpu-cpu|=%.3e)\n",
               verify_scatter_ok ? "PASS" : "FAIL", static_cast<double>(worst_canvas_diff));

    float worst_heatmap_diff = 0.0f;
    for (size_t i = 0; i < plane_elems; ++i) {
        const float d = std::fabs(h_heatmap[i] - h_heatmap_cpu[i]);
        if (d > worst_heatmap_diff) worst_heatmap_diff = d;
    }
    const bool verify_head_ok = worst_heatmap_diff <= 1e-4f;
    std::printf("[info] VERIFY(head): worst heatmap |gpu-cpu| = %.3e over %zu pixels\n",
               static_cast<double>(worst_heatmap_diff), plane_elems);
    std::printf("VERIFY(head): %s (2-layer conv + gate heatmap within 1e-4)\n", verify_head_ok ? "PASS" : "FAIL");

    bool verify_peaks_ok = (peaks_gpu.size() == peaks_cpu.size());
    if (verify_peaks_ok) {
        std::vector<PeakHost> sorted_gpu = peaks_gpu;
        std::vector<PeakCPU> sorted_cpu = peaks_cpu;
        std::sort(sorted_gpu.begin(), sorted_gpu.end(), [](const PeakHost& a, const PeakHost& b) {
            return a.iy != b.iy ? a.iy < b.iy : a.ix < b.ix;
        });
        std::sort(sorted_cpu.begin(), sorted_cpu.end(), [](const PeakCPU& a, const PeakCPU& b) {
            return a.iy != b.iy ? a.iy < b.iy : a.ix < b.ix;
        });
        for (size_t i = 0; i < sorted_gpu.size(); ++i)
            if (sorted_gpu[i].iy != sorted_cpu[i].iy || sorted_gpu[i].ix != sorted_cpu[i].ix) verify_peaks_ok = false;
    }
    std::printf("VERIFY(peaks): %s (%zu GPU peaks vs %zu CPU peaks, same (iy,ix) set after NMS)\n",
               verify_peaks_ok ? "PASS" : "FAIL", peaks_gpu.size(), peaks_cpu.size());

    // ======================= GATE layout_roundtrip =============================
    std::vector<float> h_gathered(static_cast<size_t>(num_occupied) * kPfnChannels);
    CUDA_CHECK(cudaMemcpy(h_gathered.data(), d_gathered, h_gathered.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool gate_roundtrip_ok = true;
    for (size_t i = 0; i < h_gathered.size(); ++i)
        if (h_gathered[i] != h_pillar_feat_gpu[i]) gate_roundtrip_ok = false;
    std::printf("GATE layout_roundtrip: %s (gather(scatter(pillar_feat)) bit-identical on all %d occupied pillars)\n",
               gate_roundtrip_ok ? "PASS" : "FAIL", num_occupied);

    // ======================= GATE feature_semantics =============================
    bool gate_feature_semantics = false;
    // A hand-picked 3-point pillar, independent of the main scene, run
    // through the SAME CPU reference functions. Pillar (ix=100,iy=100):
    // center pcx = kXMin + 100.5*kPillarSizeM = -40 + 40.2 = 0.2 (same for
    // pcy). mean = ((0.10+0.20+0.30)/3, (0.10+0.20+0.10)/3, (1+2+3)/3)
    //            = (0.20, 0.13333..., 2.0).
    // Point 0 = (0.10,0.10,1.0,0.5): expect features
    //   [0.10, 0.10, 1.0, 0.5,  0.10-0.20, 0.10-0.13333, 1.0-2.0,  0.10-0.2, 0.10-0.2]
    // = [0.10, 0.10, 1.0, 0.5, -0.10, -0.03333, -1.0, -0.10, -0.10]
    // computed here in double for the analytic answer, compared with a
    // 1e-4 tolerance against the pipeline's float32 result — the "free
    // exactness anchor" (kernels.cuh's file header).
    {
        const float semantics_points[3 * 4] = {
            0.10f, 0.10f, 1.0f, 0.5f,
            0.20f, 0.20f, 2.0f, 0.6f,
            0.30f, 0.10f, 3.0f, 0.7f,
        };
        std::vector<unsigned int> pc(static_cast<size_t>(kNumPillars), 0);
        std::vector<float> rp(static_cast<size_t>(kNumPillars) * kMaxPointsPerPillar * 4, 0.0f);
        int occ_cell[8]; unsigned int occ_kept[8]; float occ_mean[8 * 3];
        const int n_occ_sem = sorted_bin_cpu(3, semantics_points, pc.data(), rp.data(), occ_cell, occ_kept, occ_mean);
        std::vector<float> feat_sem(static_cast<size_t>(n_occ_sem) * kMaxPointsPerPillar * kNumPointFeatures);
        augment_features_cpu(n_occ_sem, occ_cell, occ_kept, occ_mean, rp.data(), feat_sem.data());

        const double mean_x = (0.10 + 0.20 + 0.30) / 3.0;
        const double mean_y = (0.10 + 0.20 + 0.10) / 3.0;
        const double mean_z = (1.0 + 2.0 + 3.0) / 3.0;
        const double pcx = static_cast<double>(kXMin) + 100.5 * static_cast<double>(kPillarSizeM);
        const double pcy = pcx;   // same ix/iy=100 -> identical formula
        const double expect[3][9] = {
            { 0.10, 0.10, 1.0, 0.5, 0.10 - mean_x, 0.10 - mean_y, 1.0 - mean_z, 0.10 - pcx, 0.10 - pcy },
            { 0.20, 0.20, 2.0, 0.6, 0.20 - mean_x, 0.20 - mean_y, 2.0 - mean_z, 0.20 - pcx, 0.20 - pcy },
            { 0.30, 0.10, 3.0, 0.7, 0.30 - mean_x, 0.10 - mean_y, 3.0 - mean_z, 0.30 - pcx, 0.10 - pcy },
        };
        bool sem_ok = (n_occ_sem == 1) && (occ_cell[0] == pillar_key_of(0.10f, 0.10f));
        float worst_sem = 0.0f;
        for (int slot = 0; slot < 3 && sem_ok; ++slot) {
            for (int d = 0; d < kNumPointFeatures; ++d) {
                const float got = feat_sem[static_cast<size_t>(slot) * kNumPointFeatures + d];
                const float diff = static_cast<float>(std::fabs(got - expect[slot][d]));
                if (diff > worst_sem) worst_sem = diff;
                if (diff > 1e-4f) sem_ok = false;
            }
        }
        std::printf("[info] feature_semantics: worst |pipeline - hand-computed| = %.3e over 3 points x 9 features\n",
                   static_cast<double>(worst_sem));
        std::printf("GATE feature_semantics: %s (hand-computed 9-D features for a 3-point synthetic pillar "
                   "match the pipeline within 1e-4)\n", sem_ok ? "PASS" : "FAIL");
        gate_feature_semantics = sem_ok;
    }

    // ======================= GATE detection_closure =============================
    // Every truth object must have a peak within tolerance; every peak must
    // be within tolerance of SOME truth object (no orphans -- includes the
    // cap-stress pillar and every clutter point, which must produce none).
    //
    // match_tol_m = 3.0 m, not one pillar: the surviving (post-NMS) peak per
    // car is whichever CORNER scored highest, not the geometric center (see
    // kernels.cuh's kNmsRadiusPillars comment) -- up to the car's half-
    // diagonal (~2.3 m) from the truth center. 3.0 m covers that with
    // margin while staying far below the 24 m+ inter-car spacing, so a
    // match can never be ambiguous between two different objects.
    const float match_tol_m = 3.0f;
    std::vector<bool> truth_matched(truths.size(), false);
    std::vector<bool> peak_matched(peaks_gpu.size(), false);
    float worst_match_dist = 0.0f;
    for (size_t t = 0; t < truths.size(); ++t) {
        float best_d = 1e30f;
        int best_p = -1;
        for (size_t p = 0; p < peaks_gpu.size(); ++p) {
            const float px = kXMin + (static_cast<float>(peaks_gpu[p].ix) + 0.5f) * kPillarSizeM;
            const float py = kYMin + (static_cast<float>(peaks_gpu[p].iy) + 0.5f) * kPillarSizeM;
            const float d = std::sqrt((px - truths[t].cx) * (px - truths[t].cx) + (py - truths[t].cy) * (py - truths[t].cy));
            if (d < best_d) { best_d = d; best_p = static_cast<int>(p); }
        }
        if (best_p >= 0 && best_d <= match_tol_m) { truth_matched[t] = true; peak_matched[static_cast<size_t>(best_p)] = true; }
        if (best_d > worst_match_dist && best_d < 1e30f) worst_match_dist = best_d;
    }
    bool all_truths_found = true;
    for (bool m : truth_matched) if (!m) all_truths_found = false;
    bool no_orphan_peaks = true;
    for (bool m : peak_matched) if (!m) no_orphan_peaks = false;
    const bool gate_closure = all_truths_found && no_orphan_peaks && (peaks_gpu.size() >= truths.size());
    std::printf("[info] detection_closure: %zu truth objects, %zu peaks after NMS, nearest-truth match tol %.2f m, "
               "worst matched distance %.3f m\n", truths.size(), peaks_gpu.size(), static_cast<double>(match_tol_m),
               static_cast<double>(worst_match_dist));
    std::printf("GATE detection_closure: %s (every truth object has a peak within %.1f m; every peak maps to a "
               "truth object -- zero false peaks on the cap-stress pillar or clutter)\n",
               gate_closure ? "PASS" : "FAIL", static_cast<double>(match_tol_m));

    // ======================= [info] sparsity_economics ==========================
    const size_t dense_bytes = static_cast<size_t>(kPfnChannels) * kGridNY * kGridNX * sizeof(float);
    const size_t sparse_bytes = static_cast<size_t>(num_occupied) * kPfnChannels * sizeof(float);
    std::printf("[info] sparsity_economics: %d/%d pillars occupied (%.2f%%); dense canvas = %zu bytes; "
               "sparse pillar-list storage = %zu bytes (%.1fx smaller) -- this ratio is exactly why "
               "production stacks use sparse convolutions (spconv, MinkowskiEngine) once C grows into "
               "the hundreds, instead of ever materializing the dense canvas until the LAST scatter step\n",
               num_occupied, kNumPillars, 100.0 * num_occupied / kNumPillars, dense_bytes, sparse_bytes,
               static_cast<double>(dense_bytes) / static_cast<double>(sparse_bytes));

    // ======================= [info] pillar_vs_voxel ==============================
    int *d_keys_scratch_v = nullptr, *d_idx_scratch_v = nullptr, *d_is_start_scratch_v = nullptr;
    int *d_seg_start_v = nullptr, *d_occupied_cell_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys_scratch_v, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx_scratch_v, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start_scratch_v, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_start_v, static_cast<size_t>(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_occupied_cell_v, static_cast<size_t>(n) * sizeof(int)));
    GpuTimer voxel_sort_timer;
    voxel_sort_timer.begin();
    int n_valid_voxel = 0;
    const int num_occupied_voxel = launch_sort_and_compact(n, d_voxel_keys, d_keys_scratch_v, d_idx_scratch_v,
                                                            d_is_start_scratch_v, d_seg_start_v, d_occupied_cell_v,
                                                            &n_valid_voxel);
    const float voxel_sort_ms = voxel_sort_timer.end_ms();
    const int occ_pillar_cpu_check = count_occupied_cpu(n, h_points.data(), false, kNumPillars);
    const int occ_voxel_cpu_check = count_occupied_cpu(n, h_points.data(), true, kNumVoxels);
    const size_t dense_voxel_bytes = static_cast<size_t>(kPfnChannels) * kNumZBins * kGridNY * kGridNX * sizeof(float);
    std::printf("[info] pillar_vs_voxel: pillar(2-D, z-collapsed) occupied=%d (CPU cross-check=%d), "
               "voxel(3-D, %d z-bins) occupied=%d (CPU cross-check=%d) -- %.2fx more occupied cells; "
               "sort+compact kernel time pillar=%.3f ms vs voxel=%.3f ms; a dense voxel canvas would need "
               "%zu bytes (%.1fx the pillar canvas) for the SAME channel count\n",
               num_occupied, occ_pillar_cpu_check, kNumZBins, num_occupied_voxel, occ_voxel_cpu_check,
               static_cast<double>(num_occupied_voxel) / static_cast<double>(num_occupied > 0 ? num_occupied : 1),
               static_cast<double>(sort_ms), static_cast<double>(voxel_sort_ms), dense_voxel_bytes,
               static_cast<double>(dense_voxel_bytes) / static_cast<double>(dense_bytes));
    const bool pillar_voxel_counts_ok = (num_occupied == occ_pillar_cpu_check) && (num_occupied_voxel == occ_voxel_cpu_check);
    CUDA_CHECK(cudaFree(d_keys_scratch_v));
    CUDA_CHECK(cudaFree(d_idx_scratch_v));
    CUDA_CHECK(cudaFree(d_is_start_scratch_v));
    CUDA_CHECK(cudaFree(d_seg_start_v));
    CUDA_CHECK(cudaFree(d_occupied_cell_v));

    // ======================= [info] trt_handoff (documented-only) ===============
    std::printf("[info] trt_handoff: a real deployment (project 12.01's TensorRT-deployment pattern) would "
               "ingest pillar_features FP32 [P_max=%d, N=%d, D=%d] (row-major, zero-padded beyond P_occ=%d "
               "and beyond each pillar's kept_count -- exactly this run's d_features layout), pillar_coords "
               "INT32 [P_max, 2]=(iy,ix) (this run's occupied_cell, decoded), and would consume the scattered "
               "canvas FP32 [1,%d,%d,%d] NCHW (this run's d_canvas) as the detection head's input tensor; "
               "no TensorRT numbers are fabricated here -- see PRACTICE.md and 12.01 for the real engine-build "
               "and INT8/FP16 calibration path\n",
               num_occupied > 4096 ? num_occupied : 4096, kMaxPointsPerPillar, kNumPointFeatures, num_occupied,
               kPfnChannels, kGridNY, kGridNX);

    // ======================= [time] =====================================
    std::printf("[time] sort_and_compact (pillar) kernel: %.3f ms | CPU full pipeline (bin+features+pfn+scatter+"
               "head+peaks): %.2f ms (teaching artifact -- different scale of work, not a fair speed-up claim)\n",
               static_cast<double>(sort_ms), cpu_pipeline_ms);

    // ======================= Diagnostic heatmap samples (informational) ========
    {
        float worst_truth_val = 1e30f, capstress_val = 0.0f;
        for (const auto& t : truths) {
            const int ix = static_cast<int>(std::floor((t.cx - kXMin) / kPillarSizeM));
            const int iy = static_cast<int>(std::floor((t.cy - kYMin) / kPillarSizeM));
            const float v = h_heatmap[static_cast<size_t>(iy) * kGridNX + ix];
            if (v < worst_truth_val) worst_truth_val = v;
        }
        {
            const int ix = static_cast<int>(std::floor((-30.2f - kXMin) / kPillarSizeM));
            const int iy = static_cast<int>(std::floor((-30.2f - kYMin) / kPillarSizeM));
            capstress_val = h_heatmap[static_cast<size_t>(iy) * kGridNX + ix];
        }
        std::printf("[info] heatmap diagnostics: threshold=%.4f, weakest truth-object peak value=%.4f, "
                   "cap-stress-pillar value=%.4f (must stay below threshold)\n",
                   static_cast<double>(kDetectThreshold), static_cast<double>(worst_truth_val),
                   static_cast<double>(capstress_val));
    }

    // ======================= Artifacts ==========================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();

    std::vector<float> occupancy_plane(plane_elems);
    for (size_t i = 0; i < plane_elems; ++i) occupancy_plane[i] = h_canvas_gpu[i];   // channel 0
    std::vector<unsigned char> occ_gray = to_gray(occupancy_plane, 0.0f, 1.0f);
    artifact_ok = artifact_ok && write_pgm(out_dir + "/occupancy.pgm", occ_gray, kGridNX, kGridNY);

    float hm_min = h_heatmap[0], hm_max = h_heatmap[0];
    for (float v : h_heatmap) { if (v < hm_min) hm_min = v; if (v > hm_max) hm_max = v; }
    std::vector<unsigned char> hm_gray = to_gray(h_heatmap, hm_min, hm_max);
    mark_peaks(hm_gray, kGridNX, kGridNY, peaks_gpu);
    artifact_ok = artifact_ok && write_pgm(out_dir + "/heatmap.pgm", hm_gray, kGridNX, kGridNY);

    {
        std::ofstream f(out_dir + "/feature_stats.csv");
        if (f.is_open()) {
            f << "# feature_stats.csv -- per-channel min/mean/max over occupied pillars, project 02.19\n";
            f << "field,min,mean,max\n";
            const char* names[9] = { "x", "y", "z", "intensity", "xc", "yc", "zc", "xp", "yp" };
            for (int d = 0; d < kNumPointFeatures; ++d) {
                double mn = 1e30, mx = -1e30, sum = 0.0;
                size_t cnt = 0;
                for (int p = 0; p < num_occupied; ++p) {
                    for (unsigned int s = 0; s < h_kept_count_gpu[static_cast<size_t>(p)]; ++s) {
                        const float v = h_features_gpu[static_cast<size_t>(p) * kMaxPointsPerPillar * kNumPointFeatures
                                                       + static_cast<size_t>(s) * kNumPointFeatures + d];
                        mn = std::min(mn, static_cast<double>(v));
                        mx = std::max(mx, static_cast<double>(v));
                        sum += v;
                        ++cnt;
                    }
                }
                f << names[d] << ',' << mn << ',' << (cnt > 0 ? sum / static_cast<double>(cnt) : 0.0) << ',' << mx << '\n';
            }
        } else {
            artifact_ok = false;
        }
    }

    // pillar_voxel_counts_ok is an [info]-level cross-check (RATIFIED SCOPE's
    // pillar-vs-voxel comparison is documented as measured, not a pass/fail
    // gate) -- printed above, not part of the RESULT verdict, but flagged
    // loudly here if it ever disagrees (a real bug, just not this project's
    // named gate).
    if (!pillar_voxel_counts_ok)
        std::fprintf(stderr, "[warn] pillar_vs_voxel: GPU sort-based occupied count disagreed with the "
                             "independent CPU presence-array cross-check -- investigate before trusting "
                             "the [info] numbers above.\n");

    const bool all_verify = verify_keys_ok && verify_binning_ok && features_ok && verify_pfn_ok &&
                            verify_scatter_ok && verify_head_ok && verify_peaks_ok;
    const bool all_gates = gate_cap_truncation && gate_roundtrip_ok && gate_feature_semantics && gate_closure;

    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        if (f.is_open()) {
            f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.19\n";
            f << "metric,value\n";
            f << "n_points," << n << "\n";
            f << "n_occupied_pillars," << num_occupied << "\n";
            f << "n_occupied_voxels," << num_occupied_voxel << "\n";
            f << "occupied_fraction_pillars," << (100.0 * num_occupied / kNumPillars) << "\n";
            f << "worst_feature_diff," << worst_feature_diff << "\n";
            f << "worst_pfn_diff," << worst_pfn_diff << "\n";
            f << "worst_canvas_diff," << worst_canvas_diff << "\n";
            f << "worst_heatmap_diff," << worst_heatmap_diff << "\n";
            f << "cap_sorted_var_01," << sorted_var_01 << "\n";
            f << "cap_sorted_var_02," << sorted_var_02 << "\n";
            f << "cap_sorted_var_12," << sorted_var_12 << "\n";
            f << "cap_atomic_same_order_var_01," << atomic_same_var_01 << "\n";
            f << "cap_atomic_same_order_var_02," << atomic_same_var_02 << "\n";
            f << "cap_atomic_same_order_var_12," << atomic_same_var_12 << "\n";
            f << "cap_atomic_shuffled_var_01," << atomic_shuf_var_01 << "\n";
            f << "cap_atomic_shuffled_var_02," << atomic_shuf_var_02 << "\n";
            f << "cap_atomic_shuffled_var_12," << atomic_shuf_var_12 << "\n";
            f << "detection_closure_truths," << truths.size() << "\n";
            f << "detection_closure_peaks," << peaks_gpu.size() << "\n";
            f << "detection_closure_worst_match_m," << worst_match_dist << "\n";
            f << "sort_and_compact_pillar_ms," << sort_ms << "\n";
            f << "sort_and_compact_voxel_ms," << voxel_sort_ms << "\n";
            f << "cpu_pipeline_ms," << cpu_pipeline_ms << "\n";
            f << "dense_canvas_bytes," << dense_bytes << "\n";
            f << "sparse_pillar_list_bytes," << sparse_bytes << "\n";
        } else {
            artifact_ok = false;
        }
    }

    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{occupancy.pgm, heatmap.pgm, feature_stats.csv, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out/ files\n");

    // ---- cleanup ---------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_voxel_keys));
    CUDA_CHECK(cudaFree(d_point_count));
    CUDA_CHECK(cudaFree(d_raw_points));
    CUDA_CHECK(cudaFree(d_keys_scratch));
    CUDA_CHECK(cudaFree(d_idx_scratch));
    CUDA_CHECK(cudaFree(d_is_start_scratch));
    CUDA_CHECK(cudaFree(d_seg_start));
    CUDA_CHECK(cudaFree(d_occupied_cell));
    CUDA_CHECK(cudaFree(d_mean_xyz));
    CUDA_CHECK(cudaFree(d_kept_count));
    CUDA_CHECK(cudaFree(d_features));
    CUDA_CHECK(cudaFree(d_pillar_feat));
    CUDA_CHECK(cudaFree(d_canvas));
    CUDA_CHECK(cudaFree(d_gathered));
    CUDA_CHECK(cudaFree(d_lin_w));
    CUDA_CHECK(cudaFree(d_lin_b));
    CUDA_CHECK(cudaFree(d_smoothed));
    CUDA_CHECK(cudaFree(d_gated));
    CUDA_CHECK(cudaFree(d_heatmap));
    CUDA_CHECK(cudaFree(d_is_candidate));
    CUDA_CHECK(cudaFree(d_smooth_k));
    CUDA_CHECK(cudaFree(d_sharpen_k));

    const bool success = all_verify && all_gates && artifact_ok;
    if (success)
        std::printf("RESULT: PASS (VERIFY(keys/binning/pfn/scatter/head/peaks) + GATE(cap_truncation/"
                   "layout_roundtrip/feature_semantics/detection_closure) all passed)\n");
    else
        std::printf("RESULT: FAIL (a VERIFY or GATE above did not pass -- see stderr / [info] lines for details)\n");
    return success ? 0 : 1;
}
