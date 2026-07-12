// ===========================================================================
// main.cu — entry point for project 02.04 (Euclidean clustering via GPU
//           union-find / connected components)
//
// Role in the project
// -------------------
// Orchestration: load the committed synthetic non-ground point cloud (with
// its generator-computed single-linkage TRUTH), build the neighbor-edge
// graph, run BOTH clustering algorithms on the GPU (Method A: lock-free
// union-find; Method B: min-label propagation) on the SAME edges, verify
// every stage against an independent CPU reference, gate the designed
// scene's four teaching scenarios (separation, chaining, the long-snake
// convergence pathology, noise filtering), relabel + compute per-cluster
// statistics, and write the demo artifacts. kernels.cu holds the GPU
// kernels; reference_cpu.cpp holds the independent CPU twins; kernels.cuh
// is the shared contract all three agree on.
//
// Output contract (load-bearing! — CLAUDE.md paragraph 6.1 rule, see e.g.
// 02.01's main.cu for the identical convention this file follows)
// -------------------------------------------------------------------------
// demo/run_demo.ps1 diffs the STABLE lines of this program's stdout against
// demo/expected_output.txt. Stable = "[demo]", "PROBLEM:", "DATA:", every
// "VERIFY(...)"/"GATE ...:" verdict, "ARTIFACT:", and "RESULT:" — each is
// either constant or derived ONLY from the fixed committed input file, so
// none varies run to run. "[info]" and "[time]" lines carry machine- or
// run-varying NUMBERS and are deliberately NOT diffed. If a stable line
// here changes, demo/expected_output.txt MUST change in the same edit.
//
// Read this after: kernels.cuh (the interface — read it first this time, it
// walks the whole pipeline and "THE UNION-FIND CHAPTER" end to end). Read
// this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <limits>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ===========================================================================
// Verification tolerances — MEASURED THEN MARGINED (CLAUDE.md paragraph 12).
// On the reference machine (RTX 2080 SUPER, sm_75, Release|x64, the
// committed data/sample/cluster_scene.bin, 1,469 non-ground points, 37 raw
// clusters) an actual run measured:
//     max |GPU stats centroid/AABB - independent CPU (double, sequential,
//          keyed by the already-verified-identical canonical root)| =
//          1.717e-05 m (see demo/out/gates_metrics.csv's max_stats_delta_m
//          after any run, and README "Expected output" for this exact
//          reference-run number).
// float32 atomicAdd accumulates in a scheduler-dependent ORDER (unlike the
// keys/edges/union-find comparisons, which are integer and hence bit-exact
// regardless of order) — the measured 1.7e-5 m is consistent with float32's
// ~1e-7 relative precision at this scene's ~tens-of-meters coordinate scale,
// accumulated over a cluster's largest point count (the snake, 299 points).
// The bound below margins that measurement by ~29x — generous enough to
// absorb a different GPU's atomic scheduling without being so loose it
// would miss a real accumulation bug (the same margin ratio 02.01's
// kToleranceMethodA_m uses for the analogous atomicAdd-order comparison).
// ---------------------------------------------------------------------------
static const float kToleranceStatsM = 5.0e-4f;        // 0.5 mm (~29x the measured 1.717e-5 m)
static const float kContainmentEpsilonM = 1.0e-3f;    // free-invariant epsilon (float rounding headroom)

// ===========================================================================
// Binary sample format — see scripts/make_synthetic.py's write_binary_sample()
// for the authoritative description; also documented in data/README.md.
// Read back with EXPLICIT fixed-width primitive reads (never a raw struct
// fread), the same portability reasoning util/paths.h gives for avoiding
// <filesystem> — no dependency on any compiler's struct-padding rules.
// ===========================================================================
struct SampleHeader {
    int32_t n_ground = 0, n_nonground = 0;
    float   d_m = 0.0f;
    int32_t min_cluster_size = 0;
    int32_t snake_start_idx = 0, snake_count = 0;
    int32_t sep_a_idx = 0, sep_b_idx = 0;
    int32_t chain_a_idx = 0, chain_b_idx = 0;
    int32_t noise_start_idx = 0, noise_count = 0;
};

static bool load_scene(const std::string& path, SampleHeader& hdr,
                       std::vector<float>& ground_xyz,
                       std::vector<float>& nonground_xyz,
                       std::vector<int32_t>& truth_id)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "error: could not open sample file '%s'\n", path.c_str());
        return false;
    }

    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "CLUSTR01", 8) != 0) {
        std::fprintf(stderr, "error: '%s' does not start with the expected CLUSTR01 magic\n", path.c_str());
        return false;
    }

    f.read(reinterpret_cast<char*>(&hdr.n_ground), 4);
    f.read(reinterpret_cast<char*>(&hdr.n_nonground), 4);
    f.read(reinterpret_cast<char*>(&hdr.d_m), 4);
    f.read(reinterpret_cast<char*>(&hdr.min_cluster_size), 4);
    f.read(reinterpret_cast<char*>(&hdr.snake_start_idx), 4);
    f.read(reinterpret_cast<char*>(&hdr.snake_count), 4);
    f.read(reinterpret_cast<char*>(&hdr.sep_a_idx), 4);
    f.read(reinterpret_cast<char*>(&hdr.sep_b_idx), 4);
    f.read(reinterpret_cast<char*>(&hdr.chain_a_idx), 4);
    f.read(reinterpret_cast<char*>(&hdr.chain_b_idx), 4);
    f.read(reinterpret_cast<char*>(&hdr.noise_start_idx), 4);
    f.read(reinterpret_cast<char*>(&hdr.noise_count), 4);
    int32_t reserved = 0;
    f.read(reinterpret_cast<char*>(&reserved), 4);
    if (!f || hdr.n_ground < 0 || hdr.n_nonground <= 0) {
        std::fprintf(stderr, "error: '%s' has a malformed or inconsistent header\n", path.c_str());
        return false;
    }

    // Data/code consistency checks — the same discipline 02.01's kVoxelLeafM
    // assertion follows: the scene was DESIGNED around these two compiled
    // constants (separation/chaining gaps, snake spacing, noise isolation),
    // so a mismatch here means the sample and the pipeline disagree about
    // the geometry every downstream gate assumes.
    if (std::fabs(hdr.d_m - kClusterToleranceM) > 1.0e-6f) {
        std::fprintf(stderr, "error: sample d_m=%.6f does not match kernels.cuh kClusterToleranceM=%.6f "
                             "-- regenerate the sample or update the constant, they must agree\n",
                     static_cast<double>(hdr.d_m), static_cast<double>(kClusterToleranceM));
        return false;
    }
    if (hdr.min_cluster_size != kMinClusterSize) {
        std::fprintf(stderr, "error: sample min_cluster_size=%d does not match kernels.cuh kMinClusterSize=%d\n",
                     hdr.min_cluster_size, kMinClusterSize);
        return false;
    }

    ground_xyz.resize(static_cast<size_t>(hdr.n_ground) * 3);
    if (hdr.n_ground > 0) {
        f.read(reinterpret_cast<char*>(ground_xyz.data()),
              static_cast<std::streamsize>(ground_xyz.size() * sizeof(float)));
    }
    nonground_xyz.resize(static_cast<size_t>(hdr.n_nonground) * 3);
    f.read(reinterpret_cast<char*>(nonground_xyz.data()),
          static_cast<std::streamsize>(nonground_xyz.size() * sizeof(float)));
    truth_id.resize(static_cast<size_t>(hdr.n_nonground));
    f.read(reinterpret_cast<char*>(truth_id.data()),
          static_cast<std::streamsize>(truth_id.size() * sizeof(int32_t)));

    if (!f) {
        std::fprintf(stderr, "error: '%s' is truncated\n", path.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// color_for_cluster — a deterministic, cheap "distinct-enough" pseudo-color
// per cluster id, for the topview artifacts. id < 0 (kNoCluster, or "not a
// clustered point" for the ground layer) renders a dim, unmistakably
// "not a reported object" gray. Otherwise a Knuth multiplicative hash of
// the id scatters bits across a bright [80,255] per-channel range — good
// enough visual separation for a teaching artifact (CLAUDE.md "no black
// boxes": this is intentionally simple enough to explain in three lines,
// not a real perceptual colormap).
// ---------------------------------------------------------------------------
static void color_for_cluster(int id, unsigned char& r, unsigned char& g, unsigned char& b)
{
    if (id < 0) { r = 50; g = 50; b = 50; return; }
    uint32_t h = static_cast<uint32_t>(id) * 2654435761u;   // Knuth's multiplicative hash constant
    h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    r = static_cast<unsigned char>(80 + (h & 0xFFu) % 176u);
    g = static_cast<unsigned char>(80 + ((h >> 8) & 0xFFu) % 176u);
    b = static_cast<unsigned char>(80 + ((h >> 16) & 0xFFu) % 176u);
}

// ---------------------------------------------------------------------------
// write_ppm_topview_colored — a hand-rolled binary PPM (P6) top-view (+z
// looking down), each point drawn as a colored dot. Unlike 02.01's PPM
// writer (a fixed square window centered at the origin), this project's
// scene spans a wide rectangle (see scripts/make_synthetic.py), so the
// world-to-pixel mapping here takes explicit [xmin,xmax] x [ymin,ymax]
// bounds and maps them independently onto [width,height] — a non-uniform
// (non-square-pixel) scale, which is irrelevant for a topological "which
// points share a color" teaching artifact (no distance is measured off the
// image). +y maps to "up" in the image (a flip, since image rows grow
// downward) — the same convention 02.01's writer uses.
// ---------------------------------------------------------------------------
static void write_ppm_topview_colored(const std::string& path,
                                      const float* xyz, const unsigned char* rgb, int n,
                                      int width, int height,
                                      float xmin, float xmax, float ymin, float ymax)
{
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3, 0);
    const float sx = static_cast<float>(width) / (xmax - xmin);
    const float sy = static_cast<float>(height) / (ymax - ymin);

    for (int i = 0; i < n; ++i) {
        const float x = xyz[i * 3 + 0];
        const float y = xyz[i * 3 + 1];
        const int px = static_cast<int>((x - xmin) * sx);
        const int py = height - 1 - static_cast<int>((y - ymin) * sy);
        if (px < 0 || px >= width || py < 0 || py >= height) continue;
        const size_t idx = (static_cast<size_t>(py) * width + px) * 3;
        pixels[idx + 0] = rgb[i * 3 + 0];
        pixels[idx + 1] = rgb[i * 3 + 1];
        pixels[idx + 2] = rgb[i * 3 + 2];
    }

    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << ' ' << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

// ---------------------------------------------------------------------------
// CPU-side stats accumulator (double precision, sequential — the same
// "give the oracle better precision than the thing under test" choice
// 02.01's VoxelAccumD makes for its Method-A oracle) — keyed by CANONICAL
// ROOT VALUE (not dense id), since the root value is what VERIFY(union_find)
// already proved identical between the GPU and CPU union-find results.
// ---------------------------------------------------------------------------
struct StatsAccumD {
    double sx = 0.0, sy = 0.0, sz = 0.0;
    long long count = 0;
    double minx = std::numeric_limits<double>::infinity(), miny = minx, minz = minx;
    double maxx = -std::numeric_limits<double>::infinity(), maxy = maxx, maxz = maxx;
};

int main(int argc, char** argv)
{
    bool all_ok = true;   // ANDed with every VERIFY/GATE result below; drives the final RESULT: line

    std::printf("[demo] Euclidean clustering: GPU lock-free union-find (Method A) vs "
               "GPU min-label propagation (Method B), same edges, same partition, "
               "very different sweep counts (project 02.04)\n");
    print_device_info();

    // ---- 0) Load the committed scene ---------------------------------------
    const std::string data_path = find_data_file("", argv[0], "cluster_scene.bin");
    if (data_path.empty()) {
        std::fprintf(stderr, "error: could not locate data/sample/cluster_scene.bin -- run "
                             "scripts/make_synthetic.py first (see ../data/README.md)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return EXIT_FAILURE;
    }

    SampleHeader hdr;
    std::vector<float> h_ground_xyz, h_xyz;
    std::vector<int32_t> h_truth_id;
    if (!load_scene(data_path, hdr, h_ground_xyz, h_xyz, h_truth_id)) {
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return EXIT_FAILURE;
    }
    const int n = hdr.n_nonground;

    std::printf("PROBLEM: %d ground points (context only, not clustered) + %d non-ground points "
               "(the clustering input), cluster tolerance d=leaf=%.2f m, min_cluster_size=%d\n",
               hdr.n_ground, n, static_cast<double>(kClusterToleranceM), kMinClusterSize);
    std::printf("DATA: data/sample/cluster_scene.bin [synthetic, seed 42, xorshift32, "
               "see scripts/make_synthetic.py]\n");

    // ---- 1) Device allocations ---------------------------------------------
    float* d_xyz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    unsigned long long* d_keys = nullptr;
    unsigned long long* d_keys_scratch = nullptr;
    unsigned long long* d_unique_key = nullptr;
    int* d_idx_sorted = nullptr;
    int* d_is_start = nullptr;
    int* d_seg_start = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_keys_scratch, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_unique_key, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_idx_sorted, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_start, static_cast<size_t>(n) * sizeof(int)));

    const size_t edge_capacity = static_cast<size_t>(n) * static_cast<size_t>(kMaxEdgesPerPoint);
    int* d_edge_u = nullptr; int* d_edge_v = nullptr; int* d_overflow = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_u, edge_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_v, edge_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(int)));

    int* d_parent = nullptr; int* d_label = nullptr; int* d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_parent, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_label, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    int* d_root_scratch = nullptr; int* d_idx_scratch2 = nullptr;
    int* d_is_start2 = nullptr; int* d_scan_scratch = nullptr; int* d_dense_id = nullptr;
    CUDA_CHECK(cudaMalloc(&d_root_scratch, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx_scratch2, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start2, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan_scratch, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dense_id, static_cast<size_t>(n) * sizeof(int)));

    int* d_count = nullptr;
    float* d_sum_x = nullptr; float* d_sum_y = nullptr; float* d_sum_z = nullptr;
    float* d_min_x = nullptr; float* d_min_y = nullptr; float* d_min_z = nullptr;
    float* d_max_x = nullptr; float* d_max_y = nullptr; float* d_max_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_x, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_z, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min_x, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min_y, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min_z, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_x, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_y, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_z, static_cast<size_t>(n) * sizeof(float)));

    // ---- 2) VERIFY(keys): voxel keys, GPU vs CPU shared-formula twin -------
    CpuTimer cpu_timer;
    cpu_timer.begin();
    std::vector<unsigned long long> keys_cpu(static_cast<size_t>(n));
    compute_voxel_keys_cpu(n, h_xyz.data(), kClusterToleranceM, keys_cpu.data());
    const double cpu_keys_ms = cpu_timer.end_ms();

    launch_compute_voxel_keys(n, d_xyz, kClusterToleranceM, d_keys);
    std::vector<unsigned long long> keys_gpu(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(keys_gpu.data(), d_keys, static_cast<size_t>(n) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    int key_mismatches = 0;
    for (int i = 0; i < n; ++i) if (keys_gpu[static_cast<size_t>(i)] != keys_cpu[static_cast<size_t>(i)]) ++key_mismatches;
    const bool verify_keys_ok = (key_mismatches == 0);
    all_ok = all_ok && verify_keys_ok;
    std::printf("VERIFY(keys): %s (GPU voxel keys bit-exact vs CPU reference for all points)\n",
               verify_keys_ok ? "PASS" : "FAIL");
    if (!verify_keys_ok) std::fprintf(stderr, "  %d/%d point keys mismatched\n", key_mismatches, n);

    // ---- 3) Build the neighbor-voxel index, then the edge list -------------
    const int num_voxels = launch_build_voxel_index(n, d_keys, d_keys_scratch, d_idx_sorted,
                                                     d_is_start, d_seg_start, d_unique_key);

    GpuTimer edge_timer; edge_timer.begin();
    const int num_edges_gpu = launch_build_edges(n, d_xyz, d_keys, d_unique_key, num_voxels,
                                                 d_seg_start, d_idx_sorted, n,
                                                 kClusterToleranceM, d_edge_u, d_edge_v,
                                                 static_cast<int>(edge_capacity), d_overflow);
    const float gpu_edges_ms = edge_timer.end_ms();

    int overflow = 0;
    CUDA_CHECK(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    const bool no_overflow = (overflow == 0);

    std::vector<int> edge_u_gpu(static_cast<size_t>(num_edges_gpu));
    std::vector<int> edge_v_gpu(static_cast<size_t>(num_edges_gpu));
    if (num_edges_gpu > 0) {
        CUDA_CHECK(cudaMemcpy(edge_u_gpu.data(), d_edge_u, edge_u_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(edge_v_gpu.data(), d_edge_v, edge_v_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::vector<std::pair<int,int>> edges_gpu(static_cast<size_t>(num_edges_gpu));
    for (int e = 0; e < num_edges_gpu; ++e) edges_gpu[static_cast<size_t>(e)] = { edge_u_gpu[static_cast<size_t>(e)], edge_v_gpu[static_cast<size_t>(e)] };
    std::sort(edges_gpu.begin(), edges_gpu.end());

    // ---- 4) VERIFY(edges): independent CPU edge set, exact set equality ----
    cpu_timer.begin();
    std::vector<std::pair<int,int>> edges_cpu = build_edges_cpu(n, h_xyz.data(), kClusterToleranceM);
    const double cpu_edges_ms = cpu_timer.end_ms();

    const bool verify_edges_ok = no_overflow && (edges_gpu == edges_cpu);
    all_ok = all_ok && verify_edges_ok;
    std::printf("VERIFY(edges): %s (GPU neighbor-edge set exactly equals the independent CPU "
               "unordered_map-based edge set; zero edge-buffer overflows)\n", verify_edges_ok ? "PASS" : "FAIL");
    if (!verify_edges_ok) {
        std::fprintf(stderr, "  GPU edges=%zu CPU edges=%zu overflow=%d\n",
                     edges_gpu.size(), edges_cpu.size(), overflow);
    }
    const int num_edges = num_edges_gpu;   // the edge list BOTH clustering methods consume below

    // ---- 5) Method A: GPU union-find ----------------------------------------
    launch_uf_init(n, d_parent);
    GpuTimer uf_timer; uf_timer.begin();
    int uf_sweeps = 0;
    bool uf_converged = false;
    for (; uf_sweeps < kMaxUfSweeps; ++uf_sweeps) {
        const bool changed = launch_uf_union_sweep(num_edges, d_edge_u, d_edge_v, d_parent, d_changed);
        if (!changed) { uf_converged = true; ++uf_sweeps; break; }   // count the confirming sweep too
    }
    launch_uf_finalize(n, d_parent);
    const float uf_gpu_ms = uf_timer.end_ms();

    std::vector<int> parent_gpu(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(parent_gpu.data(), d_parent, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));

    // ---- 6) Method B: GPU label propagation, the SAME edge list ------------
    launch_lp_init(n, d_label);
    GpuTimer lp_timer; lp_timer.begin();
    int lp_sweeps = 0;
    bool lp_converged = false;
    for (; lp_sweeps < kMaxLpSweeps; ++lp_sweeps) {
        const bool changed = launch_lp_sweep(num_edges, d_edge_u, d_edge_v, d_label, d_changed);
        if (!changed) { lp_converged = true; ++lp_sweeps; break; }
    }
    const float lp_gpu_ms = lp_timer.end_ms();

    std::vector<int> label_gpu(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(label_gpu.data(), d_label, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));

    // ---- 7) CPU serial union-find: the independent correctness anchor ------
    cpu_timer.begin();
    std::vector<int> parent_cpu;
    serial_union_find_cpu(n, edges_cpu, parent_cpu);
    const double cpu_uf_ms = cpu_timer.end_ms();

    int uf_mismatches = 0, lp_mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (parent_gpu[static_cast<size_t>(i)] != parent_cpu[static_cast<size_t>(i)]) ++uf_mismatches;
        if (label_gpu[static_cast<size_t>(i)]  != parent_cpu[static_cast<size_t>(i)]) ++lp_mismatches;
    }
    const bool verify_uf_ok = uf_converged && (uf_mismatches == 0);
    const bool verify_lp_ok = lp_converged && (lp_mismatches == 0);
    all_ok = all_ok && verify_uf_ok && verify_lp_ok;
    std::printf("VERIFY(union_find): %s (GPU lock-free union-find's canonical roots bit-exact vs "
               "independent sequential CPU union-find, for all %d points; converged within %d sweeps)\n",
               verify_uf_ok ? "PASS" : "FAIL", n, kMaxUfSweeps);
    if (!verify_uf_ok) std::fprintf(stderr, "  uf_mismatches=%d converged=%s\n", uf_mismatches, uf_converged ? "yes" : "no");
    std::printf("VERIFY(label_propagation): %s (GPU min-label propagation's converged labels bit-exact "
               "vs the SAME CPU union-find canonical partition, for all %d points; converged within %d sweeps)\n",
               verify_lp_ok ? "PASS" : "FAIL", n, kMaxLpSweeps);
    if (!verify_lp_ok) std::fprintf(stderr, "  lp_mismatches=%d converged=%s\n", lp_mismatches, lp_converged ? "yes" : "no");

    // ---- 8) GATE partition_vs_truth: GPU union-find vs generator truth -----
    int truth_mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (parent_gpu[static_cast<size_t>(i)] != h_truth_id[static_cast<size_t>(i)]) ++truth_mismatches;
    }
    const bool gate_truth_ok = (truth_mismatches == 0);
    all_ok = all_ok && gate_truth_ok;
    std::printf("GATE partition_vs_truth: %s (GPU union-find's canonical partition exactly equals the "
               "generator's single-linkage ground truth, for all %d non-ground points)\n",
               gate_truth_ok ? "PASS" : "FAIL", n);
    if (!gate_truth_ok) std::fprintf(stderr, "  truth_mismatches=%d\n", truth_mismatches);

    // ---- 9) Relabel (compact ids via scan) + per-cluster stats -------------
    const int K = launch_relabel_clusters(n, d_parent, d_root_scratch, d_idx_scratch2,
                                          d_is_start2, d_scan_scratch, d_dense_id);
    std::vector<int> dense_id(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(dense_id.data(), d_dense_id, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));

    launch_stats_init(K, d_count, d_sum_x, d_sum_y, d_sum_z, d_min_x, d_min_y, d_min_z, d_max_x, d_max_y, d_max_z);
    launch_stats_accumulate(n, d_xyz, d_dense_id, d_count, d_sum_x, d_sum_y, d_sum_z,
                            d_min_x, d_min_y, d_min_z, d_max_x, d_max_y, d_max_z);
    launch_stats_finalize(K, d_count, d_sum_x, d_sum_y, d_sum_z);

    std::vector<int> counts(static_cast<size_t>(K));
    std::vector<float> cent_x(static_cast<size_t>(K)), cent_y(static_cast<size_t>(K)), cent_z(static_cast<size_t>(K));
    std::vector<float> min_x(static_cast<size_t>(K)), min_y(static_cast<size_t>(K)), min_z(static_cast<size_t>(K));
    std::vector<float> max_x(static_cast<size_t>(K)), max_y(static_cast<size_t>(K)), max_z(static_cast<size_t>(K));
    CUDA_CHECK(cudaMemcpy(counts.data(), d_count, counts.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cent_x.data(), d_sum_x, cent_x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cent_y.data(), d_sum_y, cent_y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cent_z.data(), d_sum_z, cent_z.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(min_x.data(), d_min_x, min_x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(min_y.data(), d_min_y, min_y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(min_z.data(), d_min_z, min_z.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_x.data(), d_max_x, max_x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_y.data(), d_max_y, max_y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_z.data(), d_max_z, max_z.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 10) GATE stats_integrity: free invariants --------------------------
    long long count_sum = 0;
    for (int k = 0; k < K; ++k) count_sum += counts[static_cast<size_t>(k)];
    bool gate_stats_integrity_ok = (count_sum == n);
    for (int k = 0; k < K; ++k) {
        if (counts[static_cast<size_t>(k)] <= 0) continue;
        const bool inside =
            cent_x[static_cast<size_t>(k)] >= min_x[static_cast<size_t>(k)] - kContainmentEpsilonM &&
            cent_x[static_cast<size_t>(k)] <= max_x[static_cast<size_t>(k)] + kContainmentEpsilonM &&
            cent_y[static_cast<size_t>(k)] >= min_y[static_cast<size_t>(k)] - kContainmentEpsilonM &&
            cent_y[static_cast<size_t>(k)] <= max_y[static_cast<size_t>(k)] + kContainmentEpsilonM &&
            cent_z[static_cast<size_t>(k)] >= min_z[static_cast<size_t>(k)] - kContainmentEpsilonM &&
            cent_z[static_cast<size_t>(k)] <= max_z[static_cast<size_t>(k)] + kContainmentEpsilonM;
        if (!inside) gate_stats_integrity_ok = false;
    }
    all_ok = all_ok && gate_stats_integrity_ok;
    std::printf("GATE stats_integrity: %s (per-cluster counts sum to %d non-ground points exactly; "
               "every centroid lies inside its own cluster's AABB, K=%d raw clusters)\n",
               gate_stats_integrity_ok ? "PASS" : "FAIL", n, K);

    // ---- 11) VERIFY(stats): GPU per-cluster stats vs independent CPU -------
    // Keyed by CANONICAL ROOT (already proven identical GPU vs CPU in step
    // 7) rather than by dense id, so this needs no separate relabeling twin.
    std::unordered_map<int, StatsAccumD> cpu_stats;
    cpu_stats.reserve(static_cast<size_t>(K) * 2);
    for (int i = 0; i < n; ++i) {
        StatsAccumD& a = cpu_stats[parent_cpu[static_cast<size_t>(i)]];
        const double x = h_xyz[static_cast<size_t>(i) * 3 + 0];
        const double y = h_xyz[static_cast<size_t>(i) * 3 + 1];
        const double z = h_xyz[static_cast<size_t>(i) * 3 + 2];
        a.sx += x; a.sy += y; a.sz += z; a.count += 1;
        a.minx = std::min(a.minx, x); a.miny = std::min(a.miny, y); a.minz = std::min(a.minz, z);
        a.maxx = std::max(a.maxx, x); a.maxy = std::max(a.maxy, y); a.maxz = std::max(a.maxz, z);
    }
    std::vector<int> root_of_dense(static_cast<size_t>(K), -1);
    for (int i = 0; i < n; ++i) root_of_dense[static_cast<size_t>(dense_id[static_cast<size_t>(i)])] = parent_gpu[static_cast<size_t>(i)];

    bool verify_stats_ok = true;
    float max_stats_delta = 0.0f;
    for (int k = 0; k < K; ++k) {
        const auto it = cpu_stats.find(root_of_dense[static_cast<size_t>(k)]);
        if (it == cpu_stats.end() || it->second.count != counts[static_cast<size_t>(k)]) { verify_stats_ok = false; continue; }
        const StatsAccumD& a = it->second;
        const double cpu_cx = a.sx / a.count, cpu_cy = a.sy / a.count, cpu_cz = a.sz / a.count;
        const float dcx = std::fabs(static_cast<float>(cpu_cx) - cent_x[static_cast<size_t>(k)]);
        const float dcy = std::fabs(static_cast<float>(cpu_cy) - cent_y[static_cast<size_t>(k)]);
        const float dcz = std::fabs(static_cast<float>(cpu_cz) - cent_z[static_cast<size_t>(k)]);
        const float dminx = std::fabs(static_cast<float>(a.minx) - min_x[static_cast<size_t>(k)]);
        const float dmaxx = std::fabs(static_cast<float>(a.maxx) - max_x[static_cast<size_t>(k)]);
        max_stats_delta = std::max(max_stats_delta, std::max({ dcx, dcy, dcz, dminx, dmaxx }));
        if (dcx > kToleranceStatsM || dcy > kToleranceStatsM || dcz > kToleranceStatsM ||
            dminx > kToleranceStatsM || dmaxx > kToleranceStatsM) {
            verify_stats_ok = false;
        }
    }
    all_ok = all_ok && verify_stats_ok;
    std::printf("VERIFY(stats): %s (GPU per-cluster count/centroid/AABB within documented tolerance of "
               "independent double-precision CPU accumulation, all %d raw clusters; counts exact)\n",
               verify_stats_ok ? "PASS" : "FAIL", K);

    // ---- 12) Min-size filtering: raw K -> final reported clusters ----------
    // Host bookkeeping over K entries (tens, not thousands) — the same
    // "GPU does O(n) work, host does O(#components)" division of labor
    // 02.01/02.03's hypothesis-selection steps use.
    std::vector<int> final_id_of_dense(static_cast<size_t>(K), kNoCluster);
    int K_final = 0;
    for (int k = 0; k < K; ++k) {
        if (counts[static_cast<size_t>(k)] >= kMinClusterSize) final_id_of_dense[static_cast<size_t>(k)] = K_final++;
    }
    std::vector<int> final_cluster_id(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        final_cluster_id[static_cast<size_t>(i)] = final_id_of_dense[static_cast<size_t>(dense_id[static_cast<size_t>(i)])];
    }

    // ---- 13) GATE noise_filtering -------------------------------------------
    bool gate_noise_ok = true;
    for (int k = 0; k < hdr.noise_count; ++k) {
        const int i = hdr.noise_start_idx + k;
        if (i < 0 || i >= n || final_cluster_id[static_cast<size_t>(i)] != kNoCluster) { gate_noise_ok = false; break; }
    }
    all_ok = all_ok && gate_noise_ok;
    std::printf("GATE noise_filtering: %s (all %d scattered-noise points -- each farther than d from "
               "every other point, singleton raw components -- end up unclustered after min-size "
               "filtering, min_cluster_size=%d)\n", gate_noise_ok ? "PASS" : "FAIL", hdr.noise_count, kMinClusterSize);

    // ---- 14) GATE separation_test -------------------------------------------
    const int fid_sep_a = final_cluster_id[static_cast<size_t>(hdr.sep_a_idx)];
    const int fid_sep_b = final_cluster_id[static_cast<size_t>(hdr.sep_b_idx)];
    const bool gate_sep_ok = (fid_sep_a != kNoCluster) && (fid_sep_b != kNoCluster) && (fid_sep_a != fid_sep_b);
    all_ok = all_ok && gate_sep_ok;
    std::printf("GATE separation_test: %s (two objects separated by more than d stay two DISTINCT "
               "clusters -- the resolution test)\n", gate_sep_ok ? "PASS" : "FAIL");

    // ---- 15) GATE chaining_test ----------------------------------------------
    const int fid_chain_a = final_cluster_id[static_cast<size_t>(hdr.chain_a_idx)];
    const int fid_chain_b = final_cluster_id[static_cast<size_t>(hdr.chain_b_idx)];
    const bool gate_chain_ok = (fid_chain_a != kNoCluster) && (fid_chain_a == fid_chain_b);
    all_ok = all_ok && gate_chain_ok;
    std::printf("GATE chaining_test: %s (two objects bridged by a thin points path under d MERGE into "
               "one cluster -- this is single-linkage Euclidean clustering's well-known failure mode, "
               "not a bug: a real system disambiguates it with SEMANTICS -- e.g. a learned classifier or "
               "a max-cluster-extent heuristic -- that this repo's next stage, 12.xx-style semantic "
               "segmentation, would add on top; see THEORY.md \"The problem\")\n", gate_chain_ok ? "PASS" : "FAIL");

    // ---- 16) GATE snake_convergence: the money lesson ------------------------
    // Snake sweep floors are DOCUMENTED CONSTANTS, not measured-then-margined
    // like a float tolerance -- they encode the qualitative claim "label
    // propagation needs asymptotically more sweeps than union-find on a
    // long thin chain", which the scene is specifically built to make TRUE
    // by a wide, robust margin (see scripts/make_synthetic.py's snake
    // parameters and THEORY.md "The algorithm" for the O(D) vs O(log D)
    // derivation this gate is checking a live instance of).
    constexpr int kSnakeLpSweepFloor = 50;    // label propagation MUST need at least this many sweeps
    constexpr int kSnakeUfSweepCeiling = 20;  // union-find MUST converge within this many
    const bool gate_snake_ok = (lp_sweeps >= kSnakeLpSweepFloor) && (uf_sweeps <= kSnakeUfSweepCeiling);
    all_ok = all_ok && gate_snake_ok;
    std::printf("GATE snake_convergence: %s (label_propagation sweeps=%d >= floor %d; "
               "union_find sweeps=%d <= ceiling %d -- both measured on the SAME scene containing the "
               "long-snake pathology, %d points; the complexity gap this project teaches, made visible)\n",
               gate_snake_ok ? "PASS" : "FAIL", lp_sweeps, kSnakeLpSweepFloor,
               uf_sweeps, kSnakeUfSweepCeiling, hdr.snake_count);

    // ---- 17) timing [time] ---------------------------------------------------
    std::printf("[time] CPU compute_voxel_keys:        %.3f ms\n", cpu_keys_ms);
    std::printf("[time] CPU build_edges (independent):  %.3f ms (%zu edges)\n", cpu_edges_ms, edges_cpu.size());
    std::printf("[time] CPU serial_union_find:           %.3f ms\n", cpu_uf_ms);
    std::printf("[time] GPU build_edges:                 %.3f ms (%d edges, %d voxels)\n",
               static_cast<double>(gpu_edges_ms), num_edges, num_voxels);
    std::printf("[time] GPU union_find (Method A):        %.3f ms, %d sweeps\n", static_cast<double>(uf_gpu_ms), uf_sweeps);
    std::printf("[time] GPU label_propagation (Method B): %.3f ms, %d sweeps\n", static_cast<double>(lp_gpu_ms), lp_sweeps);
    long long kept_points = 0;
    for (int k = 0; k < K; ++k) if (counts[static_cast<size_t>(k)] >= kMinClusterSize) kept_points += counts[static_cast<size_t>(k)];
    std::printf("[info] clusters: %d raw connected components -> %d reported after min-size filtering "
               "(%lld points rejected as noise)\n", K, K_final, static_cast<long long>(n) - kept_points);

    // ---- 18) Artifacts ---------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);

    // World bounds for the topview renders: derived from the UNION of
    // ground AND non-ground points (not the ground layer alone) so any
    // feature placed outside the ground rectangle -- the long snake's
    // gentle arc, by construction, see scripts/make_synthetic.py -- still
    // renders in full instead of being silently clipped at the frame edge.
    float xmin = 0.0f, xmax = 1.0f, ymin = 0.0f, ymax = 1.0f;
    bool have_bounds = false;
    for (const std::vector<float>* src : { &h_ground_xyz, &h_xyz }) {
        const size_t count = src->size() / 3;
        for (size_t i = 0; i < count; ++i) {
            const float x = (*src)[i * 3 + 0];
            const float y = (*src)[i * 3 + 1];
            if (!have_bounds) { xmin = xmax = x; ymin = ymax = y; have_bounds = true; }
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
        }
    }
    // A small margin so points exactly on the boundary are not clipped by
    // the strict '<' bound check in write_ppm_topview_colored.
    const float margin = 1.0f;
    xmin -= margin; xmax += margin; ymin -= margin; ymax += margin;
    const int img_w = 1100, img_h = 500;

    // Truth-colored topview: ground dim gray; non-ground colored by TRUTH
    // component id, with truth components smaller than kMinClusterSize
    // shown as noise-gray too (a fair visual comparison against the GPU
    // image below, which is post-filtering by construction).
    {
        std::unordered_map<int,int> truth_size;
        for (int i = 0; i < n; ++i) ++truth_size[h_truth_id[static_cast<size_t>(i)]];
        std::vector<float> all_xyz(h_ground_xyz);
        all_xyz.insert(all_xyz.end(), h_xyz.begin(), h_xyz.end());
        std::vector<unsigned char> rgb(all_xyz.size());
        for (int i = 0; i < hdr.n_ground; ++i) { rgb[static_cast<size_t>(i)*3+0]=60; rgb[static_cast<size_t>(i)*3+1]=60; rgb[static_cast<size_t>(i)*3+2]=60; }
        for (int i = 0; i < n; ++i) {
            const int tid = h_truth_id[static_cast<size_t>(i)];
            const int shown_id = (truth_size[tid] >= kMinClusterSize) ? tid : kNoCluster;
            unsigned char r,g,b; color_for_cluster(shown_id, r, g, b);
            const size_t idx = (static_cast<size_t>(hdr.n_ground) + static_cast<size_t>(i)) * 3;
            rgb[idx+0]=r; rgb[idx+1]=g; rgb[idx+2]=b;
        }
        write_ppm_topview_colored(out_dir + "/topview_truth.ppm", all_xyz.data(), rgb.data(),
                                  hdr.n_ground + n, img_w, img_h, xmin, xmax, ymin, ymax);
    }

    // GPU-result-colored topview: ground dim gray; non-ground colored by
    // FINAL (post-filtering) cluster id from the GPU union-find pipeline.
    {
        std::vector<float> all_xyz(h_ground_xyz);
        all_xyz.insert(all_xyz.end(), h_xyz.begin(), h_xyz.end());
        std::vector<unsigned char> rgb(all_xyz.size());
        for (int i = 0; i < hdr.n_ground; ++i) { rgb[static_cast<size_t>(i)*3+0]=60; rgb[static_cast<size_t>(i)*3+1]=60; rgb[static_cast<size_t>(i)*3+2]=60; }
        for (int i = 0; i < n; ++i) {
            unsigned char r,g,b; color_for_cluster(final_cluster_id[static_cast<size_t>(i)], r, g, b);
            const size_t idx = (static_cast<size_t>(hdr.n_ground) + static_cast<size_t>(i)) * 3;
            rgb[idx+0]=r; rgb[idx+1]=g; rgb[idx+2]=b;
        }
        write_ppm_topview_colored(out_dir + "/topview_gpu_result.ppm", all_xyz.data(), rgb.data(),
                                  hdr.n_ground + n, img_w, img_h, xmin, xmax, ymin, ymax);
    }

    // Snake-highlight topview: everything dim gray except the snake's own
    // points, drawn bright magenta -- the "here is the pathological chain"
    // artifact demo/README.md points a learner at.
    {
        std::vector<unsigned char> rgb(h_xyz.size());
        for (int i = 0; i < n; ++i) { rgb[static_cast<size_t>(i)*3+0]=70; rgb[static_cast<size_t>(i)*3+1]=70; rgb[static_cast<size_t>(i)*3+2]=70; }
        for (int k = 0; k < hdr.snake_count; ++k) {
            const int i = hdr.snake_start_idx + k;
            if (i < 0 || i >= n) continue;
            rgb[static_cast<size_t>(i)*3+0]=255; rgb[static_cast<size_t>(i)*3+1]=0; rgb[static_cast<size_t>(i)*3+2]=255;
        }
        write_ppm_topview_colored(out_dir + "/topview_snake_highlight.ppm", h_xyz.data(), rgb.data(),
                                  n, img_w, img_h, xmin, xmax, ymin, ymax);
    }

    {
        std::ofstream f(out_dir + "/sweep_comparison.csv");
        f << "# sweep_comparison.csv -- union-find vs label-propagation on the same edge graph, project 02.04\n";
        f << "algorithm,sweeps,converged,gpu_ms,cpu_twin_ms\n";
        f << "union_find," << uf_sweeps << ',' << (uf_converged ? 1 : 0) << ',' << uf_gpu_ms << ',' << cpu_uf_ms << '\n';
        f << "label_propagation," << lp_sweeps << ',' << (lp_converged ? 1 : 0) << ',' << lp_gpu_ms << ",\n";
    }
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.04\n";
        f << "metric,value\n";
        f << "n_ground," << hdr.n_ground << '\n';
        f << "n_nonground," << n << '\n';
        f << "cluster_tolerance_d_m," << kClusterToleranceM << '\n';
        f << "min_cluster_size," << kMinClusterSize << '\n';
        f << "num_edges," << num_edges << '\n';
        f << "num_voxels," << num_voxels << '\n';
        f << "edge_overflow_count," << overflow << '\n';
        f << "raw_clusters_K," << K << '\n';
        f << "reported_clusters_K_final," << K_final << '\n';
        f << "uf_sweeps," << uf_sweeps << '\n';
        f << "uf_converged," << (uf_converged ? 1 : 0) << '\n';
        f << "lp_sweeps," << lp_sweeps << '\n';
        f << "lp_converged," << (lp_converged ? 1 : 0) << '\n';
        f << "max_stats_delta_m," << max_stats_delta << '\n';
        f << "cpu_keys_ms," << cpu_keys_ms << '\n';
        f << "cpu_edges_ms," << cpu_edges_ms << '\n';
        f << "cpu_union_find_ms," << cpu_uf_ms << '\n';
        f << "gpu_build_edges_ms," << gpu_edges_ms << '\n';
        f << "gpu_union_find_ms," << uf_gpu_ms << '\n';
        f << "gpu_label_propagation_ms," << lp_gpu_ms << '\n';
    }
    std::printf("ARTIFACT: wrote demo/out/{topview_truth.ppm, topview_gpu_result.ppm, "
               "topview_snake_highlight.ppm, sweep_comparison.csv, gates_metrics.csv}\n");

    // ---- 19) Cleanup -------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_xyz));
    CUDA_CHECK(cudaFree(d_keys)); CUDA_CHECK(cudaFree(d_keys_scratch)); CUDA_CHECK(cudaFree(d_unique_key));
    CUDA_CHECK(cudaFree(d_idx_sorted)); CUDA_CHECK(cudaFree(d_is_start)); CUDA_CHECK(cudaFree(d_seg_start));
    CUDA_CHECK(cudaFree(d_edge_u)); CUDA_CHECK(cudaFree(d_edge_v)); CUDA_CHECK(cudaFree(d_overflow));
    CUDA_CHECK(cudaFree(d_parent)); CUDA_CHECK(cudaFree(d_label)); CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_root_scratch)); CUDA_CHECK(cudaFree(d_idx_scratch2));
    CUDA_CHECK(cudaFree(d_is_start2)); CUDA_CHECK(cudaFree(d_scan_scratch)); CUDA_CHECK(cudaFree(d_dense_id));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_sum_x)); CUDA_CHECK(cudaFree(d_sum_y)); CUDA_CHECK(cudaFree(d_sum_z));
    CUDA_CHECK(cudaFree(d_min_x)); CUDA_CHECK(cudaFree(d_min_y)); CUDA_CHECK(cudaFree(d_min_z));
    CUDA_CHECK(cudaFree(d_max_x)); CUDA_CHECK(cudaFree(d_max_y)); CUDA_CHECK(cudaFree(d_max_z));

    // ---- 20) Verdict ----------------------------------------------------------------
    if (all_ok) {
        std::printf("RESULT: PASS (VERIFY(keys/edges/union_find/label_propagation/stats) + all 6 gates "
                   "passed: partition_vs_truth, stats_integrity, noise_filtering, separation_test, "
                   "chaining_test, snake_convergence)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (a VERIFY or GATE above did not pass -- see stderr for details)\n");
        return EXIT_FAILURE;
    }
}
