// ===========================================================================
// main.cu — entry point for project 02.09 (Normal + curvature estimation at
//           millions of points/sec)
//
// Role in the project
// -------------------
// Orchestration only: load the committed analytic-surface sample, build the
// voxel-hash index, run the fused GPU pipeline AND its independent CPU
// twins, run every VERIFY/GATE, write demo/out/ artifacts, then replicate
// the sample to >= 1,000,000 points and measure GPU-only throughput (the
// catalog's "millions of points/sec" promise). The kernels themselves live
// in kernels.cu; the CPU oracles live in reference_cpu.cpp; every data-
// layout constant and struct is defined once in kernels.cuh — read that
// file's long header comment FIRST.
//
// Output contract (load-bearing!)
// -------------------------------
// demo/run_demo.ps1 diffs the STABLE lines of this program's stdout against
// demo/expected_output.txt: "[demo]", "PROBLEM:", "DATA:", "VERIFY(...):",
// "GATE ...:", "ARTIFACT:", and "RESULT:" lines. "[info]" and "[time]" lines
// carry real, non-fabricated measurements but are NOT diffed (they vary by
// GPU/run — CLAUDE.md §12). Changing a stable line here requires updating
// demo/expected_output.txt in the same change, and vice versa.
//
// Read this after / before
// -------------------------
// Read this file first for the pipeline shape, then kernels.cuh (the data
// contract), then kernels.cu (the GPU implementation), then
// reference_cpu.cpp (the independence-ruled CPU oracles).
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <limits>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ---------------------------------------------------------------------------
// Sample file layout (written by scripts/make_synthetic.py — see that
// script's write_binary_sample() docstring for the byte-for-byte format).
// ---------------------------------------------------------------------------
static const char* kDataFile = "normals_scan.bin";
static const char kMagic[9] = "NRMLCRV1";   // 8 bytes + NUL for memcmp

// Anchor subset for GATE brute_force_anchor: every kAnchorStride-th point,
// so the O(n) linear-scan oracle runs a bounded number of times (never the
// full point set — kernels.cuh's file header discipline).
static const int kAnchorStride = 20;

// Throughput-pass sizing: replicate the committed sample into a 5x5x5 grid
// of translated+jittered copies (125 copies * 8,400 points/copy =
// 1,050,000 points >= the catalog's 1,000,000-point promise). See
// build_throughput_cloud()'s header comment for the full methodology and
// why it is labeled honestly rather than presented as a "real" 1M-point scan.
static const int kThroughputCopiesPerAxis = 5;
static const float kThroughputGridSpacingM = 60.0f;   // copy-to-copy offset, meters (>> any cohort's few-meter extent)
static const float kThroughputJitterM = 0.01f;         // per-point jitter added to each replica, meters (documented, see below)

// Noise-level vocabulary — MUST match scripts/make_synthetic.py's
// NOISE_NONE/NOISE_LOW/NOISE_HIGH constants field-for-field (data/README.md
// documents the shared contract; kernels.cuh only needs the SURFACE-id
// vocabulary, since noise level is purely a data-generation/gate concept).
constexpr int32_t kNoiseNone = 0, kNoiseLow = 1, kNoiseHigh = 2;

struct SampleHeader {
    int32_t n_points = 0, n_cohorts = 0, k_neighbors = 0;
    float sensor_x = 0.0f, sensor_y = 0.0f, sensor_z = 0.0f;
};

// ---------------------------------------------------------------------------
// load_sample — read the committed binary sample (points, true normals,
// true curvature, grazing-angle cosines, cohort table) per the documented
// format. Returns false (with a message on stderr) on any I/O or format
// problem — never guesses, never fabricates a smaller dataset.
// ---------------------------------------------------------------------------
static bool load_sample(const std::string& path, SampleHeader& hdr, std::vector<Cohort>& cohorts,
                        std::vector<float>& xyz, std::vector<float>& true_normal,
                        std::vector<float>& true_curvature, std::vector<float>& grazing)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "error: cannot open '%s'\n", path.c_str());
        return false;
    }
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) {
        std::fprintf(stderr, "error: '%s' has the wrong magic (expected NRMLCRV1)\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(&hdr.n_points), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.n_cohorts), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.k_neighbors), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.sensor_x), sizeof(float));
    f.read(reinterpret_cast<char*>(&hdr.sensor_y), sizeof(float));
    f.read(reinterpret_cast<char*>(&hdr.sensor_z), sizeof(float));
    if (!f || hdr.n_points <= 0 || hdr.n_cohorts <= 0) {
        std::fprintf(stderr, "error: '%s' header is malformed or truncated\n", path.c_str());
        return false;
    }
    if (hdr.k_neighbors != kK) {
        std::fprintf(stderr, "error: sample k_neighbors=%d does not match kernels.cuh kK=%d\n", hdr.k_neighbors, kK);
        return false;
    }

    cohorts.resize(static_cast<size_t>(hdr.n_cohorts));
    for (auto& c : cohorts) {
        f.read(reinterpret_cast<char*>(&c.surface_id), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&c.noise_level), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&c.start), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&c.count), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&c.param), sizeof(float));
        f.read(reinterpret_cast<char*>(&c.axis_x), sizeof(float));
        f.read(reinterpret_cast<char*>(&c.axis_y), sizeof(float));
        f.read(reinterpret_cast<char*>(&c.axis_z), sizeof(float));
    }

    const size_t n = static_cast<size_t>(hdr.n_points);
    xyz.resize(n * 3);
    f.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(xyz.size() * sizeof(float)));
    true_normal.resize(n * 3);
    f.read(reinterpret_cast<char*>(true_normal.data()), static_cast<std::streamsize>(true_normal.size() * sizeof(float)));
    true_curvature.resize(n);
    f.read(reinterpret_cast<char*>(true_curvature.data()), static_cast<std::streamsize>(true_curvature.size() * sizeof(float)));
    grazing.resize(n);
    f.read(reinterpret_cast<char*>(grazing.data()), static_cast<std::streamsize>(grazing.size() * sizeof(float)));

    if (!f) {
        std::fprintf(stderr, "error: '%s' is truncated\n", path.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// angular_error_deg — the angle, in degrees, between two UNIT vectors:
// acos(clamp(dot, -1, 1)) * 180/pi. Clamping guards the acos domain against
// a dot product that floats a hair past +-1 from rounding (routine for
// nearly-parallel unit vectors) — an unguarded acos there returns NaN, a
// classic float trap this repo's projects document wherever it can bite.
// ---------------------------------------------------------------------------
static double angular_error_deg(const float a[3], const float b[3])
{
    double dot = static_cast<double>(a[0]) * b[0] + static_cast<double>(a[1]) * b[1] + static_cast<double>(a[2]) * b[2];
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    return std::acos(dot) * 180.0 / 3.14159265358979323846;
}

// ---------------------------------------------------------------------------
// put_pixel / write_ppm_topview — hand-rolled binary PPM (P6) writer
// (02.01/02.05 lineage, cited, reimplemented compactly): every point is
// splatted as a single pixel (top-view: world x,y -> pixel column,row) with
// a caller-supplied per-point RGB triple, so ONE function produces all
// three of this project's visual artifacts (normal map, curvature heatmap,
// degeneracy map) just by varying the color array passed in.
// ---------------------------------------------------------------------------
static void put_pixel(std::vector<unsigned char>& px, int width, int height,
                      int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    px[idx + 0] = r; px[idx + 1] = g; px[idx + 2] = b;
}

static void write_ppm_topview(const std::string& path, const std::vector<float>& xyz, int n,
                              const std::vector<unsigned char>& rgb,   // [n*3], caller-computed color per point
                              int width, int height, float xmin, float xmax, float ymin, float ymax)
{
    std::vector<unsigned char> px(static_cast<size_t>(width) * height * 3, 20);   // near-black background
    const float sx = static_cast<float>(width) / (xmax - xmin);
    const float sy = static_cast<float>(height) / (ymax - ymin);
    for (int i = 0; i < n; ++i) {
        const int ox = static_cast<int>((xyz[static_cast<size_t>(i) * 3 + 0] - xmin) * sx);
        const int oy = height - 1 - static_cast<int>((xyz[static_cast<size_t>(i) * 3 + 1] - ymin) * sy);
        // A 2x2 splat: a single pixel per point is nearly invisible at 800x800
        // over a scene several tens of meters across — this keeps every one
        // of the 8,400 committed-sample points visible in the PNG-viewer sense.
        for (int dy = 0; dy <= 1; ++dy)
            for (int dx = 0; dx <= 1; ++dx)
                put_pixel(px, width, height, ox + dx, oy + dy,
                         rgb[static_cast<size_t>(i) * 3 + 0], rgb[static_cast<size_t>(i) * 3 + 1], rgb[static_cast<size_t>(i) * 3 + 2]);
    }
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << ' ' << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
}

// ---------------------------------------------------------------------------
// xorshift32_next / xorshift32_uniform — the same tiny deterministic PRNG
// scripts/make_synthetic.py uses (Marsaglia 2003, 02.01/02.05 lineage,
// cited), reimplemented here in C++ for build_throughput_cloud()'s
// per-replica jitter. Never std::rand (repo convention: fixed-seed,
// portable, bit-reproducible across platforms).
// ---------------------------------------------------------------------------
static uint32_t xorshift32_next(uint32_t& state)
{
    uint32_t x = state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    state = x;
    return x;
}
static float xorshift32_uniform(uint32_t& state, float lo, float hi)
{
    const float u = (xorshift32_next(state) >> 8) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * u;
}

// ---------------------------------------------------------------------------
// build_throughput_cloud — replicate the committed sample into a
// kThroughputCopiesPerAxis^3 grid of translated copies, EACH with a small
// deterministic per-point jitter, to reach >= 1,000,000 points for the
// throughput measurement.
//
// Methodology, stated honestly (CLAUDE.md "never fabricate a benchmark
// claim"): this is NOT a new 1-million-point scan — it is the SAME 8,400
// analytically-generated points, copy-translated by kThroughputGridSpacingM
// (>> any single cohort's few-meter extent, so copies never spatially
// overlap and every copy's local KNN neighborhoods stay meaningful — a
// point's true nearest neighbors are still points from its OWN copy) and
// perturbed by +-kThroughputJitterM of independent jitter per copy (seeded
// per-copy, deterministic) so the GPU is not literally re-reading identical
// cache lines 125 times over, which would flatter memory-bandwidth numbers
// unrealistically. The throughput GATE below measures exactly what it says:
// the fused pipeline's points/sec on a real (if replicated) 1,050,000-point
// GPU workload — labeled, not disguised as a fresh independent scan.
// ---------------------------------------------------------------------------
static void build_throughput_cloud(const std::vector<float>& base_xyz, int base_n, std::vector<float>& out_xyz)
{
    const int copies = kThroughputCopiesPerAxis * kThroughputCopiesPerAxis * kThroughputCopiesPerAxis;
    out_xyz.resize(static_cast<size_t>(copies) * base_n * 3);

    int copy_idx = 0;
    for (int cz = 0; cz < kThroughputCopiesPerAxis; ++cz) {
        for (int cy = 0; cy < kThroughputCopiesPerAxis; ++cy) {
            for (int cx = 0; cx < kThroughputCopiesPerAxis; ++cx, ++copy_idx) {
                const float ox = static_cast<float>(cx) * kThroughputGridSpacingM;
                const float oy = static_cast<float>(cy) * kThroughputGridSpacingM;
                const float oz = static_cast<float>(cz) * kThroughputGridSpacingM;
                uint32_t rng = static_cast<uint32_t>(1000 + copy_idx * 7919);   // distinct, deterministic seed per copy
                if (rng == 0) rng = 1;
                const size_t base_off = static_cast<size_t>(copy_idx) * base_n * 3;
                for (int i = 0; i < base_n; ++i) {
                    const float jx = xorshift32_uniform(rng, -kThroughputJitterM, kThroughputJitterM);
                    const float jy = xorshift32_uniform(rng, -kThroughputJitterM, kThroughputJitterM);
                    const float jz = xorshift32_uniform(rng, -kThroughputJitterM, kThroughputJitterM);
                    out_xyz[base_off + static_cast<size_t>(i) * 3 + 0] = base_xyz[static_cast<size_t>(i) * 3 + 0] + ox + jx;
                    out_xyz[base_off + static_cast<size_t>(i) * 3 + 1] = base_xyz[static_cast<size_t>(i) * 3 + 1] + oy + jy;
                    out_xyz[base_off + static_cast<size_t>(i) * 3 + 2] = base_xyz[static_cast<size_t>(i) * 3 + 2] + oz + jz;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CohortStats — the aggregate numbers every per-cohort GATE/[info] line
// reports: angular error vs. analytic truth, curvature, and degeneracy
// counts, computed over the GPU pipeline's results for one cohort's point
// range [start, start+count).
// ---------------------------------------------------------------------------
struct CohortStats {
    double mean_angle_deg = 0.0, max_angle_deg = 0.0;
    double mean_curvature = 0.0, median_curvature = 0.0;
    int degen_clean = 0, degen_edge = 0, degen_isolated = 0;
    int n = 0;            // points in the cohort (always the full cohort count)
    int n_angle = 0;       // points actually contributing to mean/max_angle_deg (see confident_only below)
};

// compute_cohort_stats — angular error / curvature / degeneracy summary for
// one cohort's point range. `grazing` + `confident_only`: when
// confident_only is true, angular-error statistics (mean/max_angle_deg)
// are computed ONLY over points with |grazing_cos| >= kOrientationGrazingCos
// -- deliberately EXCLUDING the rare grazing-incidence sign-flip outliers
// GATE orientation already characterizes separately (kernels.cuh's file
// header STEP 5), so a plane/sphere GATE measures normal-fit ACCURACY
// (curvature bias + sensor noise) without being dominated by an unrelated
// failure mode. Curvature and degeneracy counts are UNAFFECTED by normal
// sign, so they always use every point in the cohort regardless.
static CohortStats compute_cohort_stats(const Cohort& c, const std::vector<float>& true_normal,
                                        const std::vector<float>& gpu_normal, const std::vector<float>& gpu_curvature,
                                        const std::vector<int32_t>& gpu_degeneracy,
                                        const std::vector<float>& grazing, bool confident_only)
{
    CohortStats s;
    s.n = c.count;
    double sum_angle = 0.0, sum_curv = 0.0;
    std::vector<float> curv_sorted;
    curv_sorted.reserve(static_cast<size_t>(c.count));
    for (int i = 0; i < c.count; ++i) {
        const int idx = c.start + i;
        if (!confident_only || grazing[static_cast<size_t>(idx)] >= kOrientationGrazingCos) {
            const double ang = angular_error_deg(&true_normal[static_cast<size_t>(idx) * 3], &gpu_normal[static_cast<size_t>(idx) * 3]);
            sum_angle += ang;
            if (ang > s.max_angle_deg) s.max_angle_deg = ang;
            ++s.n_angle;
        }
        sum_curv += gpu_curvature[static_cast<size_t>(idx)];
        curv_sorted.push_back(gpu_curvature[static_cast<size_t>(idx)]);
        switch (gpu_degeneracy[static_cast<size_t>(idx)]) {
            case kDegenClean: ++s.degen_clean; break;
            case kDegenEdgeCorner: ++s.degen_edge; break;
            default: ++s.degen_isolated; break;
        }
    }
    s.mean_angle_deg = sum_angle / std::max(1, s.n_angle);
    s.mean_curvature = sum_curv / std::max(1, c.count);
    std::sort(curv_sorted.begin(), curv_sorted.end());
    s.median_curvature = curv_sorted.empty() ? 0.0 : curv_sorted[curv_sorted.size() / 2];
    return s;
}

// find_cohort — the single cohort matching (surface_id, noise_level); every
// surface/noise combination is written exactly once by make_synthetic.py,
// so this always finds exactly one match for a valid combination.
static const Cohort& find_cohort(const std::vector<Cohort>& cohorts, int32_t surface_id, int32_t noise_level)
{
    for (const auto& c : cohorts)
        if (c.surface_id == surface_id && c.noise_level == noise_level) return c;
    static Cohort dummy{};   // unreachable for a valid (surface_id, noise_level) pair from this project's own data
    return dummy;
}

int main(int argc, char** argv)
{
    bool all_ok = true;   // ANDed with every VERIFY/GATE result; drives the final RESULT: line

    // ---- 0) Arguments -------------------------------------------------------
    std::string cli_dir;
    if (argc > 1) cli_dir = argv[1];

    std::printf("[demo] Normal + curvature estimation at millions of points/sec (project 02.09): "
               "voxel-hash KNN -> mean-shifted covariance -> Jacobi eigensolve -> sensor-oriented "
               "normal -> surface-variation curvature -> degeneracy flag\n");
    print_device_info();

    // ---- 1) Load the committed sample ---------------------------------------
    const std::string data_path = find_data_file(cli_dir, argv[0], kDataFile);
    if (data_path.empty()) {
        std::fprintf(stderr, "error: could not find %s under data/sample/ (searched CLI/exe-relative/CWD candidates)\n", kDataFile);
        return EXIT_FAILURE;
    }
    SampleHeader hdr;
    std::vector<Cohort> cohorts;
    std::vector<float> h_xyz, h_true_normal, h_true_curvature, h_grazing;
    if (!load_sample(data_path, hdr, cohorts, h_xyz, h_true_normal, h_true_curvature, h_grazing)) return EXIT_FAILURE;

    const int n = hdr.n_points;
    std::printf("PROBLEM: N=%d points, %d analytic-surface cohorts (plane/sphere/cylinder/edge x "
               "none/low/high noise), K=%d neighbors, cell=%.2f m\n", n, hdr.n_cohorts, kK, kCellSizeM);
    std::printf("DATA: data/sample/%s [synthetic, seed 42, xorshift32, see scripts/make_synthetic.py]\n", kDataFile);

    // ---- 2) Upload the point cloud ------------------------------------------
    float* d_xyz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // =========================================================================
    // Voxel-hash index build — GPU, timed.
    // =========================================================================
    unsigned long long* d_keys = nullptr; unsigned long long* d_keys_scratch = nullptr; unsigned long long* d_unique = nullptr;
    int* d_idx_sorted = nullptr; int* d_is_start = nullptr; int* d_seg_start = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_keys_scratch, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_unique, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_idx_sorted, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_start, static_cast<size_t>(n) * sizeof(int)));

    GpuTimer t_index_gpu; t_index_gpu.begin();
    launch_compute_hash_keys(n, d_xyz, kCellSizeM, d_keys);
    const int num_voxels = launch_build_voxel_index(n, d_keys, d_keys_scratch, d_idx_sorted, d_is_start, d_seg_start, d_unique);
    const float ms_index_gpu = t_index_gpu.end_ms();

    // =========================================================================
    // THE pipeline — GPU, correctness pass (neighbor ids WRITTEN, for VERIFY(knn)).
    // =========================================================================
    float* d_normal = nullptr; float* d_eigenvalues = nullptr; float* d_curvature = nullptr;
    int32_t* d_degeneracy = nullptr; int32_t* d_found = nullptr; int32_t* d_neighbor_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_normal, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eigenvalues, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curvature, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_degeneracy, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_found, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_ids, static_cast<size_t>(n) * kK * sizeof(int32_t)));

    GpuTimer t_pipeline_gpu; t_pipeline_gpu.begin();
    launch_estimate_normals(n, d_xyz, d_unique, num_voxels, d_seg_start, d_idx_sorted, n, kCellSizeM,
                            hdr.sensor_x, hdr.sensor_y, hdr.sensor_z,
                            d_normal, d_eigenvalues, d_curvature, d_degeneracy, d_found, d_neighbor_ids);
    const float ms_pipeline_gpu = t_pipeline_gpu.end_ms();

    std::vector<float> gpu_normal(static_cast<size_t>(n) * 3), gpu_eigenvalues(static_cast<size_t>(n) * 3), gpu_curvature(static_cast<size_t>(n));
    std::vector<int32_t> gpu_degeneracy(static_cast<size_t>(n)), gpu_found(static_cast<size_t>(n)), gpu_neighbor_ids(static_cast<size_t>(n) * kK);
    CUDA_CHECK(cudaMemcpy(gpu_normal.data(), d_normal, gpu_normal.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_eigenvalues.data(), d_eigenvalues, gpu_eigenvalues.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_curvature.data(), d_curvature, gpu_curvature.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_degeneracy.data(), d_degeneracy, gpu_degeneracy.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_found.data(), d_found, gpu_found.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_neighbor_ids.data(), d_neighbor_ids, gpu_neighbor_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // =========================================================================
    // CPU reference — independent voxel-hash-map KNN + covariance + eigen +
    // normal + curvature + degeneracy, one call per point (reference_cpu.cpp).
    // =========================================================================
    HashMapCpu h_hash_map;
    CpuTimer t_hashmap_cpu; t_hashmap_cpu.begin();
    build_hash_map_cpu(n, h_xyz.data(), kCellSizeM, h_hash_map);
    const double ms_hashmap_cpu = t_hashmap_cpu.end_ms();

    std::vector<KnnResultCpu> cpu_results(static_cast<size_t>(n));
    CpuTimer t_pipeline_cpu; t_pipeline_cpu.begin();
    for (int i = 0; i < n; ++i) {
        estimate_normals_cpu(n, h_xyz.data(), h_hash_map, kCellSizeM, hdr.sensor_x, hdr.sensor_y, hdr.sensor_z, i, cpu_results[static_cast<size_t>(i)]);
    }
    const double ms_pipeline_cpu = t_pipeline_cpu.end_ms();

    // ---- VERIFY(knn): neighbor SETS exact, ascending, shared tie-break -----
    bool knn_ok = true;
    for (int i = 0; i < n; ++i) {
        const auto& cpu_ids = cpu_results[static_cast<size_t>(i)].neighbor_ids;
        if (static_cast<int>(cpu_ids.size()) != gpu_found[static_cast<size_t>(i)]) { knn_ok = false; break; }
        for (size_t k = 0; k < cpu_ids.size(); ++k) {
            if (gpu_neighbor_ids[static_cast<size_t>(i) * kK + k] != cpu_ids[k]) { knn_ok = false; break; }
        }
        if (!knn_ok) break;
    }
    all_ok &= knn_ok;
    std::printf("VERIFY(knn): %s (GPU voxel-hash bounded-heap KNN result lists exactly equal the independent "
               "CPU unordered_map + partial_sort twin, ascending under the shared (dist2,index) tie-break, "
               "all %d points)\n", knn_ok ? "PASS" : "FAIL", n);

    // ---- VERIFY(eigen): eigenvalues within a tight float tolerance ---------
    const float kEigenTol = 5.0e-4f;
    float max_eigen_diff = 0.0f;
    for (size_t i = 0; i < gpu_eigenvalues.size(); ++i) {
        const float d = std::fabs(gpu_eigenvalues[i] - cpu_results[i / 3].eigenvalues[i % 3]);
        if (d > max_eigen_diff) max_eigen_diff = d;
    }
    const bool eigen_ok = (max_eigen_diff <= kEigenTol);
    all_ok &= eigen_ok;
    // The measured diff itself lives on an [info] line, NOT in the diffed
    // VERIFY line: it is a real ULP-scale float comparison whose last digit
    // can legitimately differ across GPU architectures (sm_75 vs sm_86 vs
    // sm_89 SASS may contract FMAs differently) even though the PASS/FAIL
    // verdict against a fixed tolerance is architecture-independent -- the
    // same "stable verdict, volatile measurement" split 02.05's own VERIFY
    // lines use (their measurements are exact integer counts instead).
    std::printf("VERIFY(eigen): %s (GPU d_jacobi_eigen_3x3 vs independent CPU jacobi_eigen_3x3_cpu -- two "
               "different rotation-angle formulas, same algorithm family, tol %.1e m^2)\n",
               eigen_ok ? "PASS" : "FAIL", static_cast<double>(kEigenTol));
    std::printf("[info] eigen_diff: max |GPU-CPU| = %.3e m^2\n", static_cast<double>(max_eigen_diff));

    // ---- VERIFY(normals): full-pipeline angular agreement -------------------
    const double kNormalAngleTolDeg = 0.5;   // measured-then-margined, see THEORY.md
    double max_normal_diff_deg = 0.0;
    for (int i = 0; i < n; ++i) {
        const double a = angular_error_deg(&gpu_normal[static_cast<size_t>(i) * 3], cpu_results[static_cast<size_t>(i)].normal);
        if (a > max_normal_diff_deg) max_normal_diff_deg = a;
    }
    const bool normals_ok = (max_normal_diff_deg <= kNormalAngleTolDeg);
    all_ok &= normals_ok;
    std::printf("VERIFY(normals): %s (GPU vs CPU-twin full-pipeline normal agreement, tol %.2f deg)\n",
               normals_ok ? "PASS" : "FAIL", kNormalAngleTolDeg);
    std::printf("[info] normals_diff: max angular |GPU-CPU| = %.4f deg\n", max_normal_diff_deg);

    // ---- VERIFY(curvature) ---------------------------------------------------
    const float kCurvatureTol = 5.0e-4f;
    float max_curv_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(gpu_curvature[static_cast<size_t>(i)] - cpu_results[static_cast<size_t>(i)].curvature);
        if (d > max_curv_diff) max_curv_diff = d;
    }
    const bool curvature_ok = (max_curv_diff <= kCurvatureTol);
    all_ok &= curvature_ok;
    std::printf("VERIFY(curvature): %s (GPU vs CPU-twin surface-variation agreement, tol %.1e)\n",
               curvature_ok ? "PASS" : "FAIL", static_cast<double>(kCurvatureTol));
    std::printf("[info] curvature_diff: max |GPU-CPU| = %.3e\n", static_cast<double>(max_curv_diff));

    // ---- VERIFY(degeneracy): categorical, exact -----------------------------
    int degeneracy_mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (gpu_degeneracy[static_cast<size_t>(i)] != cpu_results[static_cast<size_t>(i)].degeneracy) ++degeneracy_mismatches;
    }
    const bool degeneracy_ok = (degeneracy_mismatches == 0);
    all_ok &= degeneracy_ok;
    std::printf("VERIFY(degeneracy): %s (GPU vs CPU-twin degeneracy-flag agreement, %d/%d mismatches)\n",
               degeneracy_ok ? "PASS" : "FAIL", degeneracy_mismatches, n);

    // =========================================================================
    // GATE brute_force_anchor — the third-tier, hash-free O(n) oracle, over a
    // documented stride subset (kernels.cuh's independence ruling).
    // =========================================================================
    int anchor_checked = 0, anchor_mismatches = 0;
    CpuTimer t_anchor; t_anchor.begin();
    for (int i = 0; i < n; i += kAnchorStride) {
        KnnResultCpu bf;
        estimate_normal_brute_force(n, h_xyz.data(), kCellSizeM, hdr.sensor_x, hdr.sensor_y, hdr.sensor_z, i, bf);
        ++anchor_checked;
        const int found = gpu_found[static_cast<size_t>(i)];
        if (static_cast<int>(bf.neighbor_ids.size()) != found) { ++anchor_mismatches; continue; }
        bool row_ok = true;
        for (int k = 0; k < found; ++k) {
            if (gpu_neighbor_ids[static_cast<size_t>(i) * kK + k] != bf.neighbor_ids[static_cast<size_t>(k)]) { row_ok = false; break; }
        }
        if (!row_ok) ++anchor_mismatches;
    }
    const double ms_anchor = t_anchor.end_ms();
    const bool anchor_ok = (anchor_mismatches == 0);
    all_ok &= anchor_ok;
    std::printf("GATE brute_force_anchor: %s (GPU neighbor sets exactly equal a hash-free O(n) linear-scan "
               "CPU oracle for %d of %d points, stride %d; %d mismatches)\n",
               anchor_ok ? "PASS" : "FAIL", anchor_checked, n, kAnchorStride, anchor_mismatches);

    // =========================================================================
    // GATE plane_normals — angular error vs. analytic truth. The plane's
    // TRUE curvature is exactly 0 everywhere, so noise=none is a genuine
    // near-exact anchor (no curvature-fit bias possible); noise=low/high
    // are measured and bounded. Every stat below EXCLUDES grazing-incidence
    // points (confident_only=true) so this gate measures normal-fit
    // ACCURACY, not the separately-characterized orientation sign-flip
    // failure mode (GATE orientation, kernels.cuh file header STEP 5).
    // =========================================================================
    const double kExactAnchorMeanTolDeg = 0.05, kExactAnchorMaxTolDeg = 0.5;
    const double kNoisyMeanTolDeg = 3.0, kNoisyMaxTolDeg = 8.0;

    const Cohort& plane_c0 = find_cohort(cohorts, kSurfacePlane, kNoiseNone);
    const Cohort& plane_cl = find_cohort(cohorts, kSurfacePlane, kNoiseLow);
    const Cohort& plane_ch = find_cohort(cohorts, kSurfacePlane, kNoiseHigh);
    const CohortStats plane_s0 = compute_cohort_stats(plane_c0, h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
    const CohortStats plane_sl = compute_cohort_stats(plane_cl, h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
    const CohortStats plane_sh = compute_cohort_stats(plane_ch, h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
    const bool plane_ok = (plane_s0.mean_angle_deg <= kExactAnchorMeanTolDeg) && (plane_s0.max_angle_deg <= kExactAnchorMaxTolDeg) &&
                          (plane_sl.mean_angle_deg <= kNoisyMeanTolDeg) && (plane_sl.max_angle_deg <= kNoisyMaxTolDeg) &&
                          (plane_sh.mean_angle_deg <= kNoisyMeanTolDeg) && (plane_sh.max_angle_deg <= kNoisyMaxTolDeg);
    all_ok &= plane_ok;
    std::printf("GATE plane_normals: %s (angular error vs analytic normal, grazing points excluded -- "
               "noise=none mean=%.4f deg max=%.4f deg [near-exact anchor, tol mean<=%.2f max<=%.2f]; "
               "noise=low mean=%.3f max=%.3f deg; noise=high mean=%.3f max=%.3f deg [tol mean<=%.1f max<=%.1f])\n",
               plane_ok ? "PASS" : "FAIL", plane_s0.mean_angle_deg, plane_s0.max_angle_deg, kExactAnchorMeanTolDeg, kExactAnchorMaxTolDeg,
               plane_sl.mean_angle_deg, plane_sl.max_angle_deg, plane_sh.mean_angle_deg, plane_sh.max_angle_deg, kNoisyMeanTolDeg, kNoisyMaxTolDeg);

    // =========================================================================
    // GATE sphere_normals — a CURVED surface has genuine curvature-fit bias
    // even at noise=none: a finite K=16 neighborhood is not an infinitesimal
    // tangent plane, so the smallest-eigenvector normal differs SYSTEMATICALLY
    // from the analytic radial normal by an amount that scales with
    // (neighborhood radius / sphere radius) -- THEORY.md derives and measures
    // this. All three noise cohorts therefore share ONE measured-then-
    // margined bound rather than treating noise=none as exact.
    // =========================================================================
    const double kCurvedMeanTolDeg = 4.0, kCurvedMaxTolDeg = 15.0;
    const Cohort& sph_c0 = find_cohort(cohorts, kSurfaceSphere, kNoiseNone);
    const Cohort& sph_cl = find_cohort(cohorts, kSurfaceSphere, kNoiseLow);
    const Cohort& sph_ch = find_cohort(cohorts, kSurfaceSphere, kNoiseHigh);
    const CohortStats sph_s0 = compute_cohort_stats(sph_c0, h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
    const CohortStats sph_sl = compute_cohort_stats(sph_cl, h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
    const CohortStats sph_sh = compute_cohort_stats(sph_ch, h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
    const bool sphere_ok = (sph_s0.mean_angle_deg <= kCurvedMeanTolDeg) && (sph_s0.max_angle_deg <= kCurvedMaxTolDeg) &&
                           (sph_sl.mean_angle_deg <= kCurvedMeanTolDeg) && (sph_sl.max_angle_deg <= kCurvedMaxTolDeg) &&
                           (sph_sh.mean_angle_deg <= kCurvedMeanTolDeg) && (sph_sh.max_angle_deg <= kCurvedMaxTolDeg);
    all_ok &= sphere_ok;
    std::printf("GATE sphere_normals: %s (angular error vs analytic radial normal, grazing points excluded -- "
               "noise=none mean=%.4f deg max=%.4f deg [curvature-fit bias, NOT sensor noise -- no near-exact "
               "anchor expected here, see THEORY.md]; noise=low mean=%.3f max=%.3f deg; noise=high mean=%.3f "
               "max=%.3f deg [tol mean<=%.1f max<=%.1f])\n",
               sphere_ok ? "PASS" : "FAIL", sph_s0.mean_angle_deg, sph_s0.max_angle_deg,
               sph_sl.mean_angle_deg, sph_sl.max_angle_deg, sph_sh.mean_angle_deg, sph_sh.max_angle_deg,
               kCurvedMeanTolDeg, kCurvedMaxTolDeg);

    // =========================================================================
    // [info] cylinder_normals — per-point angular error vs the analytic radial
    // normal (info-only: a cylinder's normal error mixes with the axis-fit
    // check below, which is the GATED aggregate for this surface).
    // =========================================================================
    {
        const Cohort& c0 = find_cohort(cohorts, kSurfaceCylinder, kNoiseNone);
        const CohortStats s0 = compute_cohort_stats(c0, h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
        std::printf("[info] cylinder_normals: noise=none mean angular error = %.4f deg, max = %.4f deg (vs "
                   "analytic radial normal, grazing points excluded; %d/%d points)\n", s0.mean_angle_deg, s0.max_angle_deg, s0.n_angle, s0.n);
    }

    // =========================================================================
    // GATE cylinder_axis — the free aggregate check (kernels.cuh file header):
    // fit the cylinder axis from the ESTIMATED normals alone (no truth used
    // except for the final comparison), via the smallest-eigenvalue
    // eigenvector of the normals' scatter matrix E[n n^T] -- every cylinder
    // normal is EXACTLY perpendicular to the true axis, so the axis is the
    // direction of near-zero variance in that scatter.
    // =========================================================================
    bool cylinder_axis_ok = false;
    double cylinder_axis_angle_deg = 0.0;
    {
        const Cohort& c0 = find_cohort(cohorts, kSurfaceCylinder, kNoiseNone);
        double sxx = 0, sxy = 0, sxz = 0, syy = 0, syz = 0, szz = 0;
        for (int i = 0; i < c0.count; ++i) {
            const int idx = c0.start + i;
            const double nx = gpu_normal[static_cast<size_t>(idx) * 3 + 0];
            const double ny = gpu_normal[static_cast<size_t>(idx) * 3 + 1];
            const double nz = gpu_normal[static_cast<size_t>(idx) * 3 + 2];
            sxx += nx * nx; sxy += nx * ny; sxz += nx * nz; syy += ny * ny; syz += ny * nz; szz += nz * nz;
        }
        const double inv_m = 1.0 / std::max(1, c0.count);
        const float cov[6] = { static_cast<float>(sxx * inv_m), static_cast<float>(sxy * inv_m), static_cast<float>(sxz * inv_m),
                               static_cast<float>(syy * inv_m), static_cast<float>(syz * inv_m), static_cast<float>(szz * inv_m) };
        float eigenvalues[3]; float eigenvectors[3][3];
        jacobi_eigen_3x3_cpu(cov, eigenvalues, eigenvectors);
        const float fitted_axis[3] = { eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2] };
        const float true_axis[3] = { c0.axis_x, c0.axis_y, c0.axis_z };
        // Axis is sign-ambiguous (both +axis and -axis describe the same
        // line) -- compare via |dot|, not dot.
        double dot = static_cast<double>(fitted_axis[0]) * true_axis[0] + static_cast<double>(fitted_axis[1]) * true_axis[1] +
                    static_cast<double>(fitted_axis[2]) * true_axis[2];
        dot = std::fabs(dot);
        if (dot > 1.0) dot = 1.0;
        cylinder_axis_angle_deg = std::acos(dot) * 180.0 / 3.14159265358979323846;
        const double kAxisTolDeg = 2.0;
        cylinder_axis_ok = (cylinder_axis_angle_deg <= kAxisTolDeg);
        all_ok &= cylinder_axis_ok;
        std::printf("GATE cylinder_axis: %s (axis fitted from %d ESTIMATED normals' scatter-matrix smallest "
                   "eigenvector vs the TRUE cylinder axis: %.4f deg apart, tol %.1f deg)\n",
                   cylinder_axis_ok ? "PASS" : "FAIL", c0.count, cylinder_axis_angle_deg, kAxisTolDeg);
    }

    // =========================================================================
    // GATE orientation — dot(estimated_normal, sensor-p) > 0 required on
    // "confidently viewable" (non-grazing) points; grazing points reported
    // [info] with their measured (lower) success rate (kernels.cuh file
    // header STEP 5, THEORY.md "the grazing-incidence failure mode").
    // =========================================================================
    int confident_total = 0, confident_correct = 0, grazing_total = 0, grazing_correct = 0;
    for (int i = 0; i < n; ++i) {
        const float view[3] = { hdr.sensor_x - h_xyz[static_cast<size_t>(i) * 3 + 0],
                                hdr.sensor_y - h_xyz[static_cast<size_t>(i) * 3 + 1],
                                hdr.sensor_z - h_xyz[static_cast<size_t>(i) * 3 + 2] };
        const float dot = gpu_normal[static_cast<size_t>(i) * 3 + 0] * view[0] + gpu_normal[static_cast<size_t>(i) * 3 + 1] * view[1] +
                          gpu_normal[static_cast<size_t>(i) * 3 + 2] * view[2];
        const bool correct = (dot > 0.0f);
        if (h_grazing[static_cast<size_t>(i)] >= kOrientationGrazingCos) {
            ++confident_total; if (correct) ++confident_correct;
        } else {
            ++grazing_total; if (correct) ++grazing_correct;
        }
    }
    const bool orientation_ok = (confident_total > 0) && (confident_correct == confident_total);
    all_ok &= orientation_ok;
    std::printf("GATE orientation: %s (%d/%d confidently-viewable points (|cos| >= %.2f) correctly face the "
               "sensor; exact-gated)\n", orientation_ok ? "PASS" : "FAIL", confident_correct, confident_total, kOrientationGrazingCos);
    std::printf("[info] orientation_grazing: %d/%d grazing-incidence points (|cos| < %.2f) correctly face the "
               "sensor (%.1f%% -- the honest failure mode: sign-disambiguation degrades near edge-on views)\n",
               grazing_correct, grazing_total, kOrientationGrazingCos,
               grazing_total > 0 ? (100.0 * grazing_correct / grazing_total) : 100.0);

    // =========================================================================
    // GATE curvature_ordering — plane < cylinder < sphere < edge medians, at
    // noise=none (the cleanest test of the geometric ordering claim).
    // =========================================================================
    const CohortStats plane0 = compute_cohort_stats(find_cohort(cohorts, kSurfacePlane, kNoiseNone), h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, false);
    const CohortStats sphere0 = compute_cohort_stats(find_cohort(cohorts, kSurfaceSphere, kNoiseNone), h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, false);
    const CohortStats cyl0 = compute_cohort_stats(find_cohort(cohorts, kSurfaceCylinder, kNoiseNone), h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, false);
    const CohortStats edge0 = compute_cohort_stats(find_cohort(cohorts, kSurfaceEdge, kNoiseNone), h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, false);
    const bool ordering_ok = (plane0.median_curvature < cyl0.median_curvature) &&
                             (cyl0.median_curvature < sphere0.median_curvature) &&
                             (sphere0.median_curvature < edge0.median_curvature);
    all_ok &= ordering_ok;
    std::printf("GATE curvature_ordering: %s (median surface variation, noise=none: plane=%.5f < "
               "cylinder=%.5f < sphere=%.5f < edge=%.5f)\n", ordering_ok ? "PASS" : "FAIL",
               plane0.median_curvature, cyl0.median_curvature, sphere0.median_curvature, edge0.median_curvature);

    // =========================================================================
    // GATE degeneracy_flags — edge cohort flagged at >= a measured floor;
    // plane-interior (noise=none) flagged at <= a measured ceiling. The
    // edge cohort's ridge-adjacent design (scripts/make_synthetic.py's
    // gen_edge docstring) puts most, but honestly not all, of its points
    // within one neighborhood-radius of the discontinuity -- points far
    // from the ridge (large t) sit on a locally FLAT single face and are
    // correctly recognized as clean, which is why this floor is a measured
    // 40%, not a made-up 90% (CLAUDE.md "never fabricate").
    // =========================================================================
    const double kEdgeFloorFrac = 0.40, kPlaneCeilingFrac = 0.05;
    const double edge_flagged_frac = static_cast<double>(edge0.degen_edge) / std::max(1, edge0.n);
    const double plane_flagged_frac = static_cast<double>(plane0.degen_edge) / std::max(1, plane0.n);
    const bool degeneracy_flags_ok = (edge_flagged_frac >= kEdgeFloorFrac) && (plane_flagged_frac <= kPlaneCeilingFrac);
    all_ok &= degeneracy_flags_ok;
    std::printf("GATE degeneracy_flags: %s (edge cohort flagged EDGE_CORNER: %d/%d = %.1f%%, floor %.0f%%; "
               "plane-interior (noise=none) flagged EDGE_CORNER: %d/%d = %.1f%%, ceiling %.0f%%)\n",
               degeneracy_flags_ok ? "PASS" : "FAIL", edge0.degen_edge, edge0.n, 100.0 * edge_flagged_frac, 100.0 * kEdgeFloorFrac,
               plane0.degen_edge, plane0.n, 100.0 * plane_flagged_frac, 100.0 * kPlaneCeilingFrac);

    // =========================================================================
    // [info] noise_scaling — mean angular error and median curvature per
    // (surface, noise) bucket: the K-vs-noise story THEORY.md discusses.
    // =========================================================================
    std::printf("[info] noise_scaling (mean angular error deg / median curvature, by surface x noise level):\n");
    const char* surf_names[4] = { "plane", "sphere", "cylinder", "edge" };
    const int32_t surf_ids[4] = { kSurfacePlane, kSurfaceSphere, kSurfaceCylinder, kSurfaceEdge };
    const int32_t noise_ids[3] = { kNoiseNone, kNoiseLow, kNoiseHigh };
    const char* noise_names[3] = { "none", "low(3mm)", "high(15mm)" };
    for (int si = 0; si < 4; ++si) {
        for (int ni = 0; ni < 3; ++ni) {
            const CohortStats s = compute_cohort_stats(find_cohort(cohorts, surf_ids[si], noise_ids[ni]), h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
            std::printf("[info]   %-8s %-11s mean_angle=%.4f deg (grazing excluded)  median_curvature=%.5f\n",
                       surf_names[si], noise_names[ni], s.mean_angle_deg, s.median_curvature);
        }
    }

    // =========================================================================
    // Throughput pass — GPU ONLY (no CPU twin: correctness is already
    // established above at the committed-sample scale; re-running an O(n)
    // CPU pass over 1M+ points would add runtime without adding evidence).
    // =========================================================================
    std::vector<float> h_xyz_big;
    build_throughput_cloud(h_xyz, n, h_xyz_big);
    const int n_big = static_cast<int>(h_xyz_big.size() / 3);

    float* d_xyz_big = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz_big, static_cast<size_t>(n_big) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xyz_big, h_xyz_big.data(), static_cast<size_t>(n_big) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    unsigned long long* d_keys_big = nullptr; unsigned long long* d_keys_scratch_big = nullptr; unsigned long long* d_unique_big = nullptr;
    int* d_idx_sorted_big = nullptr; int* d_is_start_big = nullptr; int* d_seg_start_big = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys_big, static_cast<size_t>(n_big) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_keys_scratch_big, static_cast<size_t>(n_big) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_unique_big, static_cast<size_t>(n_big) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_idx_sorted_big, static_cast<size_t>(n_big) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start_big, static_cast<size_t>(n_big) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_start_big, static_cast<size_t>(n_big) * sizeof(int)));

    GpuTimer t_index_big; t_index_big.begin();
    launch_compute_hash_keys(n_big, d_xyz_big, kCellSizeM, d_keys_big);
    const int num_voxels_big = launch_build_voxel_index(n_big, d_keys_big, d_keys_scratch_big, d_idx_sorted_big, d_is_start_big, d_seg_start_big, d_unique_big);
    const float ms_index_big = t_index_big.end_ms();

    float* d_normal_big = nullptr; float* d_eigenvalues_big = nullptr; float* d_curvature_big = nullptr;
    int32_t* d_degeneracy_big = nullptr; int32_t* d_found_big = nullptr;
    CUDA_CHECK(cudaMalloc(&d_normal_big, static_cast<size_t>(n_big) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eigenvalues_big, static_cast<size_t>(n_big) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curvature_big, static_cast<size_t>(n_big) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_degeneracy_big, static_cast<size_t>(n_big) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_found_big, static_cast<size_t>(n_big) * sizeof(int32_t)));

    GpuTimer t_pipeline_big; t_pipeline_big.begin();
    // out_neighbor_ids = nullptr: the throughput pass never writes the
    // per-point neighbor-id debug array (kernels.cu's estimate_normals_kernel
    // header explains why -- production code never reads it back either).
    launch_estimate_normals(n_big, d_xyz_big, d_unique_big, num_voxels_big, d_seg_start_big, d_idx_sorted_big, n_big, kCellSizeM,
                            hdr.sensor_x, hdr.sensor_y, hdr.sensor_z,
                            d_normal_big, d_eigenvalues_big, d_curvature_big, d_degeneracy_big, d_found_big, nullptr);
    const float ms_pipeline_big = t_pipeline_big.end_ms();

    const double total_s = (static_cast<double>(ms_index_big) + static_cast<double>(ms_pipeline_big)) / 1000.0;
    const double mpts_per_s_total = (total_s > 0.0) ? (n_big / total_s) / 1.0e6 : 0.0;
    const double mpts_per_s_pipeline_only = (ms_pipeline_big > 0.0f) ? (n_big / (static_cast<double>(ms_pipeline_big) / 1000.0)) / 1.0e6 : 0.0;

    // Measured-then-margined (CLAUDE.md "never fabricate a benchmark claim"):
    // repeated runs on the owner's RTX 2080 SUPER (sm_75) measured 19-21
    // Mpts/s total (index-build+pipeline) on this 1,050,000-point workload;
    // 8.0 Mpts/s is a floor with real margin under that, not a made-up
    // number -- see README/THEORY.md for the actual measured figures.
    const double kThroughputFloorMptsPerS = 8.0;
    const bool throughput_ok = (mpts_per_s_total >= kThroughputFloorMptsPerS);
    all_ok &= throughput_ok;
    // The measured Mpts/s figure is a TIMING measurement -- it varies run to
    // run and machine to machine (CLAUDE.md §12: timings are teaching
    // artifacts, never diffed) -- so it lives on an [info]/[time]-style line
    // below, NOT inside this GATE's stable, diffed text (the mistake this
    // comment guards future edits against: an earlier draft embedded the
    // Mpts/s figure directly in the GATE line, which made demo/run_demo.*
    // fail nondeterministically the moment a rerun measured 19.1 instead of
    // 20.3 Mpts/s -- caught before this project shipped).
    std::printf("GATE throughput: %s (%d points [%d replicated+jittered copies of the %d-point committed "
               "sample -- see build_throughput_cloud()], floor %.1f Mpts/s)\n", throughput_ok ? "PASS" : "FAIL", n_big,
               kThroughputCopiesPerAxis * kThroughputCopiesPerAxis * kThroughputCopiesPerAxis, n, kThroughputFloorMptsPerS);
    std::printf("[info] throughput_measured: index-build+pipeline = %.1f Mpts/s, pipeline-only = %.1f Mpts/s\n",
               mpts_per_s_total, mpts_per_s_pipeline_only);

    // =========================================================================
    // [time] lines.
    // =========================================================================
    std::printf("[time] committed-sample (N=%d): index-build GPU=%.3f ms, pipeline GPU=%.3f ms; hash-map-build "
               "CPU=%.3f ms, pipeline CPU=%.3f ms; brute-force anchor CPU=%.3f ms\n",
               n, static_cast<double>(ms_index_gpu), static_cast<double>(ms_pipeline_gpu), ms_hashmap_cpu, ms_pipeline_cpu, ms_anchor);
    std::printf("[time] throughput workload (N=%d): index-build GPU=%.3f ms, pipeline GPU=%.3f ms, total=%.3f ms\n",
               n_big, static_cast<double>(ms_index_big), static_cast<double>(ms_pipeline_big),
               static_cast<double>(ms_index_big) + static_cast<double>(ms_pipeline_big));

    // =========================================================================
    // Artifacts.
    // =========================================================================
    const std::string out_dir = resolve_out_dir(argv[0]);

    float xmin = std::numeric_limits<float>::infinity(), xmax = -std::numeric_limits<float>::infinity();
    float ymin = std::numeric_limits<float>::infinity(), ymax = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        xmin = std::min(xmin, h_xyz[static_cast<size_t>(i) * 3 + 0]); xmax = std::max(xmax, h_xyz[static_cast<size_t>(i) * 3 + 0]);
        ymin = std::min(ymin, h_xyz[static_cast<size_t>(i) * 3 + 1]); ymax = std::max(ymax, h_xyz[static_cast<size_t>(i) * 3 + 1]);
    }
    const float pad_x = 0.05f * std::max(1.0f, xmax - xmin), pad_y = 0.05f * std::max(1.0f, ymax - ymin);
    xmin -= pad_x; xmax += pad_x; ymin -= pad_y; ymax += pad_y;

    {
        std::vector<unsigned char> rgb(static_cast<size_t>(n) * 3);
        for (int i = 0; i < n; ++i) {
            rgb[static_cast<size_t>(i) * 3 + 0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, (gpu_normal[static_cast<size_t>(i) * 3 + 0] * 0.5f + 0.5f) * 255.0f)));
            rgb[static_cast<size_t>(i) * 3 + 1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, (gpu_normal[static_cast<size_t>(i) * 3 + 1] * 0.5f + 0.5f) * 255.0f)));
            rgb[static_cast<size_t>(i) * 3 + 2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, (gpu_normal[static_cast<size_t>(i) * 3 + 2] * 0.5f + 0.5f) * 255.0f)));
        }
        write_ppm_topview(out_dir + "/normal_map.ppm", h_xyz, n, rgb, 900, 900, xmin, xmax, ymin, ymax);
    }
    {
        const float cmax = 0.15f;   // clamp range for the heatmap, chosen against this project's own measured curvature scale (THEORY.md)
        std::vector<unsigned char> rgb(static_cast<size_t>(n) * 3);
        for (int i = 0; i < n; ++i) {
            float t = gpu_curvature[static_cast<size_t>(i)] / cmax;
            t = std::max(0.0f, std::min(1.0f, t));
            rgb[static_cast<size_t>(i) * 3 + 0] = static_cast<unsigned char>(255.0f * t);
            rgb[static_cast<size_t>(i) * 3 + 1] = 40;
            rgb[static_cast<size_t>(i) * 3 + 2] = static_cast<unsigned char>(255.0f * (1.0f - t));
        }
        write_ppm_topview(out_dir + "/curvature_heatmap.ppm", h_xyz, n, rgb, 900, 900, xmin, xmax, ymin, ymax);
    }
    {
        std::vector<unsigned char> rgb(static_cast<size_t>(n) * 3);
        for (int i = 0; i < n; ++i) {
            unsigned char r = 70, g = 70, b = 80;   // clean: dim gray
            if (gpu_degeneracy[static_cast<size_t>(i)] == kDegenEdgeCorner) { r = 230; g = 60; b = 50; }     // red
            else if (gpu_degeneracy[static_cast<size_t>(i)] == kDegenIsolated) { r = 240; g = 220; b = 40; } // yellow
            rgb[static_cast<size_t>(i) * 3 + 0] = r; rgb[static_cast<size_t>(i) * 3 + 1] = g; rgb[static_cast<size_t>(i) * 3 + 2] = b;
        }
        write_ppm_topview(out_dir + "/degeneracy_map.ppm", h_xyz, n, rgb, 900, 900, xmin, xmax, ymin, ymax);
    }
    {
        std::ofstream f(out_dir + "/per_cohort_errors.csv");
        f << "# per_cohort_errors.csv -- GPU pipeline stats per (surface, noise) cohort, project 02.09\n";
        f << "surface,noise,n,n_angle_confident,mean_angle_deg,max_angle_deg,mean_curvature,median_curvature,degen_clean,degen_edge_corner,degen_isolated\n";
        for (int si = 0; si < 4; ++si) {
            for (int ni = 0; ni < 3; ++ni) {
                const CohortStats s = compute_cohort_stats(find_cohort(cohorts, surf_ids[si], noise_ids[ni]), h_true_normal, gpu_normal, gpu_curvature, gpu_degeneracy, h_grazing, true);
                f << surf_names[si] << ',' << noise_names[ni] << ',' << s.n << ',' << s.n_angle << ',' << s.mean_angle_deg << ',' << s.max_angle_deg << ','
                  << s.mean_curvature << ',' << s.median_curvature << ',' << s.degen_clean << ',' << s.degen_edge << ',' << s.degen_isolated << '\n';
            }
        }
    }
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.09\n";
        f << "metric,value\n";
        f << "n_points," << n << '\n';
        f << "num_voxels," << num_voxels << '\n';
        f << "max_eigen_diff," << max_eigen_diff << '\n';
        f << "max_normal_diff_deg," << max_normal_diff_deg << '\n';
        f << "max_curv_diff," << max_curv_diff << '\n';
        f << "degeneracy_mismatches," << degeneracy_mismatches << '\n';
        f << "anchor_mismatches," << anchor_mismatches << '\n';
        f << "cylinder_axis_angle_deg," << cylinder_axis_angle_deg << '\n';
        f << "orientation_confident_correct," << confident_correct << '\n';
        f << "orientation_confident_total," << confident_total << '\n';
        f << "orientation_grazing_correct," << grazing_correct << '\n';
        f << "orientation_grazing_total," << grazing_total << '\n';
        f << "curvature_median_plane," << plane0.median_curvature << '\n';
        f << "curvature_median_cylinder," << cyl0.median_curvature << '\n';
        f << "curvature_median_sphere," << sphere0.median_curvature << '\n';
        f << "curvature_median_edge," << edge0.median_curvature << '\n';
        f << "edge_flagged_frac," << edge_flagged_frac << '\n';
        f << "plane_flagged_frac," << plane_flagged_frac << '\n';
        f << "throughput_n," << n_big << '\n';
        f << "throughput_mpts_per_s_total," << mpts_per_s_total << '\n';
        f << "throughput_mpts_per_s_pipeline_only," << mpts_per_s_pipeline_only << '\n';
    }
    std::printf("ARTIFACT: wrote demo/out/{normal_map.ppm, curvature_heatmap.ppm, degeneracy_map.ppm, "
               "per_cohort_errors.csv, gates_metrics.csv}\n");

    // ---- Cleanup --------------------------------------------------------------
    cudaFree(d_xyz); cudaFree(d_keys); cudaFree(d_keys_scratch); cudaFree(d_unique);
    cudaFree(d_idx_sorted); cudaFree(d_is_start); cudaFree(d_seg_start);
    cudaFree(d_normal); cudaFree(d_eigenvalues); cudaFree(d_curvature); cudaFree(d_degeneracy); cudaFree(d_found); cudaFree(d_neighbor_ids);
    cudaFree(d_xyz_big); cudaFree(d_keys_big); cudaFree(d_keys_scratch_big); cudaFree(d_unique_big);
    cudaFree(d_idx_sorted_big); cudaFree(d_is_start_big); cudaFree(d_seg_start_big);
    cudaFree(d_normal_big); cudaFree(d_eigenvalues_big); cudaFree(d_curvature_big); cudaFree(d_degeneracy_big); cudaFree(d_found_big);

    // ---- Final verdict ----------------------------------------------------------
    if (all_ok) {
        std::printf("RESULT: PASS (VERIFY(knn/eigen/normals/curvature/degeneracy) + GATE(brute_force_anchor/"
                   "plane_normals/sphere_normals/cylinder_axis/orientation/curvature_ordering/degeneracy_flags/"
                   "throughput) all passed)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see the VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
