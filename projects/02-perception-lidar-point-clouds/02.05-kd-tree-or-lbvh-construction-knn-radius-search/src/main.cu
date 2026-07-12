// ===========================================================================
// main.cu — entry point for project 02.05 (KD-tree or LBVH construction +
//           KNN/radius search on GPU)
//
// Role in the project
// -------------------
// Orchestration only: load the committed sample, drive every build stage
// (Morton -> sort -> Karras radix tree -> AABB propagation) and every query
// path (BVH radius, BVH KNN, voxel-hash radius) on BOTH the GPU and its
// independent CPU twins, run every VERIFY/GATE, print the report, and write
// the demo/out/ artifacts. The kernels themselves live in kernels.cu; the
// CPU oracles live in reference_cpu.cpp; every data-layout constant and
// struct is defined once in kernels.cuh — read that file's long header
// comment FIRST.
//
// Output contract (load-bearing!)
// -------------------------------
// demo/run_demo.ps1 diffs the STABLE lines of this program's stdout against
// demo/expected_output.txt: "[demo]", "PROBLEM:", "DATA:", "VERIFY(...):",
// "GATE ...:", "ARTIFACT:", and "RESULT:" lines. "[info]" and "[time]"
// lines carry real, non-fabricated measurements but are NOT diffed (they
// vary by GPU/run — CLAUDE.md §12). Changing a stable line here requires
// updating demo/expected_output.txt in the same change, and vice versa.
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
#include <chrono>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ---------------------------------------------------------------------------
// Sample file layout (written by scripts/make_synthetic.py — see that
// script's write_binary_sample() docstring for the byte-for-byte format).
// kDataFile / kMagic are the two ends of that contract this file checks.
// ---------------------------------------------------------------------------
static const char* kDataFile = "lbvh_scan.bin";
static const char kMagic[9] = "LBVHSCN1";   // 8 bytes + NUL for strncmp

struct SampleHeader {
    int32_t n_points = 0, n_beam = 0, n_dense = 0, n_sparse = 0;
    float   radius_m = 0.0f;
    int32_t n_queries = 0, idx_dense_query = 0, idx_sparse_query = 0, n_anchor = 0;
};

// ---------------------------------------------------------------------------
// load_sample — read the committed binary sample (points + queries) per the
// documented format. Returns false (with a message on stderr) on any I/O or
// format problem — never guesses, never fabricates a smaller dataset.
// ---------------------------------------------------------------------------
static bool load_sample(const std::string& path, SampleHeader& hdr,
                        std::vector<float>& xyz, std::vector<float>& queries)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "error: cannot open '%s'\n", path.c_str());
        return false;
    }
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) {
        std::fprintf(stderr, "error: '%s' has the wrong magic (expected LBVHSCN1)\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(&hdr.n_points), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.n_beam), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.n_dense), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.n_sparse), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.radius_m), sizeof(float));
    f.read(reinterpret_cast<char*>(&hdr.n_queries), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.idx_dense_query), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.idx_sparse_query), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&hdr.n_anchor), sizeof(int32_t));
    if (!f || hdr.n_points <= 0 || hdr.n_queries <= 0) {
        std::fprintf(stderr, "error: '%s' header is malformed or truncated\n", path.c_str());
        return false;
    }
    if (std::fabs(hdr.radius_m - kRadiusM) > 1e-6f) {
        std::fprintf(stderr, "error: sample radius_m=%.6f does not match kernels.cuh kRadiusM=%.6f\n",
                     hdr.radius_m, kRadiusM);
        return false;
    }

    xyz.resize(static_cast<size_t>(hdr.n_points) * 3);
    f.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(xyz.size() * sizeof(float)));
    queries.resize(static_cast<size_t>(hdr.n_queries) * 3);
    f.read(reinterpret_cast<char*>(queries.data()), static_cast<std::streamsize>(queries.size() * sizeof(float)));

    if (!f) {
        std::fprintf(stderr, "error: '%s' is truncated\n", path.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// compute_scene_aabb — the point cloud's bounding box, PADDED by a small
// fixed margin on every side. Padding matters for exactly one reason: Stage
// 1's Morton quantization saturates coordinates outside [min,max] to the
// boundary cell (kernels.cuh's quantize_axis comment) — harmless for the
// TREE (every point used to build it is, by definition, inside its own
// AABB), but the padding keeps quantization resolution uniform across the
// whole scene rather than compressing everything into a hair's-width of
// the true extent, which the actual `min`/`max` of a finite point sample
// would otherwise do at the sample's own boundary points.
// ---------------------------------------------------------------------------
static SceneAABB compute_scene_aabb(const std::vector<float>& xyz, int n)
{
    SceneAABB aabb;
    for (int a = 0; a < 3; ++a) {
        aabb.min[a] = std::numeric_limits<float>::infinity();
        aabb.max[a] = -std::numeric_limits<float>::infinity();
    }
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) {
            const float v = xyz[static_cast<size_t>(i) * 3 + a];
            aabb.min[a] = std::min(aabb.min[a], v);
            aabb.max[a] = std::max(aabb.max[a], v);
        }
    }
    for (int a = 0; a < 3; ++a) {
        const float pad = 0.05f * std::max(1e-3f, aabb.max[a] - aabb.min[a]);
        aabb.min[a] -= pad;
        aabb.max[a] += pad;
    }
    return aabb;
}

// ---------------------------------------------------------------------------
// check_tree_validity — GATE tree_validity's two free structural
// invariants, checked directly against the GPU-built tree (host-copied):
//   (a) every leaf is reachable from the root EXACTLY once — a DFS from
//       node 0 must visit each of the n leaves exactly once, and never
//       visit an internal node whose subtree does not terminate; a
//       correctly-built binary tree over n leaves has exactly n-1 internal
//       nodes and 2n-1 total, so a full traversal touching every node
//       exactly once is itself part of what this checks (via node_visits).
//   (b) every internal node's AABB contains BOTH children's AABB,
//       component-wise, EXACTLY (no epsilon — see kernels.cuh's VERIFY(aabb)
//       comment: min/max of a fixed point set is order-independent and
//       never rounds, so containment must hold with exact float <=/>=).
// ---------------------------------------------------------------------------
static bool check_tree_validity(const std::vector<LbvhNode>& nodes, int n,
                                int& out_leaf_visits, int& out_internal_visits,
                                bool& out_containment_ok)
{
    out_leaf_visits = 0;
    out_internal_visits = 0;
    out_containment_ok = true;
    if (n < 2) { out_leaf_visits = n; return true; }   // degenerate single-leaf tree

    std::vector<int> node_visits(static_cast<size_t>(2 * n - 1), 0);
    std::vector<int> stack;
    stack.reserve(128);
    stack.push_back(0);

    while (!stack.empty()) {
        const int idx = stack.back();
        stack.pop_back();
        node_visits[idx]++;

        if (is_leaf_node(idx, n)) {
            out_leaf_visits++;
            continue;
        }
        out_internal_visits++;
        const LbvhNode& node = nodes[idx];
        const int lc = node.left, rc = node.right;
        for (int a = 0; a < 3; ++a) {
            if (node.aabb_min[a] > nodes[lc].aabb_min[a] || node.aabb_max[a] < nodes[lc].aabb_max[a] ||
                node.aabb_min[a] > nodes[rc].aabb_min[a] || node.aabb_max[a] < nodes[rc].aabb_max[a]) {
                out_containment_ok = false;
            }
        }
        stack.push_back(lc);
        stack.push_back(rc);
    }

    // "reachable exactly once": every node index must have visits==1 (no
    // node reachable zero times -- unreachable subtree -- or more than
    // once -- a shared/cyclic child, which would corrupt the AABB
    // propagation's second-arrival counting too).
    bool exactly_once = true;
    for (int v : node_visits) if (v != 1) { exactly_once = false; break; }

    return exactly_once && (out_leaf_visits == n) && (out_internal_visits == n - 1);
}

// ---------------------------------------------------------------------------
// write_ppm_topview — hand-rolled binary PPM (P6) writer (02.01/02.04
// lineage, cited). Base point cloud in dim gray; then two OVERLAYS drawn
// on top, each a small block per point so they are visible against ~200k
// background dots: the dense-cluster query's BVH radius-search result
// (red) and the sparse query's BVH KNN result (yellow), plus both query
// locations marked with a small cyan cross. This is the ONE demo artifact
// that makes the density-contrast lesson visible, not just numeric.
// ---------------------------------------------------------------------------
static void put_pixel(std::vector<unsigned char>& px, int width, int height,
                      int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    px[idx + 0] = r; px[idx + 1] = g; px[idx + 2] = b;
}

static void write_ppm_topview(const std::string& path,
                              const std::vector<float>& xyz, int n,
                              const std::vector<int>& dense_result, const float dense_query[3],
                              const std::vector<int>& sparse_result, const float sparse_query[3],
                              int width, int height,
                              float xmin, float xmax, float ymin, float ymax)
{
    std::vector<unsigned char> px(static_cast<size_t>(width) * height * 3, 0);
    const float sx = static_cast<float>(width) / (xmax - xmin);
    const float sy = static_cast<float>(height) / (ymax - ymin);
    auto to_px = [&](float x, float y, int& ox, int& oy) {
        ox = static_cast<int>((x - xmin) * sx);
        oy = height - 1 - static_cast<int>((y - ymin) * sy);
    };

    for (int i = 0; i < n; ++i) {
        int ox, oy;
        to_px(xyz[static_cast<size_t>(i) * 3 + 0], xyz[static_cast<size_t>(i) * 3 + 1], ox, oy);
        put_pixel(px, width, height, ox, oy, 70, 70, 80);
    }

    for (int id : dense_result) {
        int ox, oy;
        to_px(xyz[static_cast<size_t>(id) * 3 + 0], xyz[static_cast<size_t>(id) * 3 + 1], ox, oy);
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
            put_pixel(px, width, height, ox + dx, oy + dy, 230, 60, 50);   // red: dense radius-search hits
    }
    for (int id : sparse_result) {
        int ox, oy;
        to_px(xyz[static_cast<size_t>(id) * 3 + 0], xyz[static_cast<size_t>(id) * 3 + 1], ox, oy);
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
            put_pixel(px, width, height, ox + dx, oy + dy, 240, 220, 40);   // yellow: sparse KNN hits (far away!)
    }

    for (const float* q : { dense_query, sparse_query }) {
        int ox, oy;
        to_px(q[0], q[1], ox, oy);
        for (int t = -4; t <= 4; ++t) {
            put_pixel(px, width, height, ox + t, oy, 40, 230, 230);   // cyan cross
            put_pixel(px, width, height, ox, oy + t, 40, 230, 230);
        }
    }

    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << ' ' << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
}

// ---------------------------------------------------------------------------
// Small stat helpers for the [info] traversal_stats / morton_locality lines.
// ---------------------------------------------------------------------------
static void mean_max(const std::vector<int>& v, double& mean, int& mx)
{
    mean = 0.0; mx = 0;
    if (v.empty()) return;
    long long sum = 0;
    for (int x : v) { sum += x; mx = std::max(mx, x); }
    mean = static_cast<double>(sum) / static_cast<double>(v.size());
}

int main(int argc, char** argv)
{
    bool all_ok = true;   // ANDed with every VERIFY/GATE result; drives the final RESULT: line

    // ---- 0) Arguments -------------------------------------------------------
    // One optional positional argument: an override directory for
    // data/sample/ (learners experimenting with a regenerated sample).
    std::string cli_dir;
    if (argc > 1) cli_dir = argv[1];

    std::printf("[demo] LBVH (Karras radix-tree) construction + KNN/radius search on GPU, "
               "contrasted against fixed-radius voxel hashing (project 02.05)\n");
    print_device_info();

    // ---- 1) Load the committed sample ---------------------------------------
    const std::string data_path = find_data_file(cli_dir, argv[0], kDataFile);
    if (data_path.empty()) {
        std::fprintf(stderr, "error: could not find %s under data/sample/ (searched CLI/exe-relative/CWD candidates)\n", kDataFile);
        return EXIT_FAILURE;
    }
    SampleHeader hdr;
    std::vector<float> h_xyz, h_queries;
    if (!load_sample(data_path, hdr, h_xyz, h_queries)) return EXIT_FAILURE;

    const int n = hdr.n_points;
    const int Q = hdr.n_queries;

    std::printf("PROBLEM: N=%d points (%d beam-scan + %d dense-cluster + %d sparse-region), "
               "%d queries (2 designed + 998 self-query + 1000 grid), K=%d, radius r=%.2f m\n",
               n, hdr.n_beam, hdr.n_dense, hdr.n_sparse, Q, kQueryK, kRadiusM);
    std::printf("DATA: data/sample/%s [synthetic, seed 42, xorshift32, see scripts/make_synthetic.py]\n", kDataFile);

    const SceneAABB aabb = compute_scene_aabb(h_xyz, n);

    // ---- 2) Upload the point cloud + queries --------------------------------
    float* d_xyz = nullptr;
    float* d_queries = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, static_cast<size_t>(Q) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(), static_cast<size_t>(Q) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // =========================================================================
    // STAGE 1 — augmented Morton keys.
    // =========================================================================
    unsigned long long* d_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(n) * sizeof(unsigned long long)));

    GpuTimer t_morton_gpu; t_morton_gpu.begin();
    launch_compute_augmented_keys(n, d_xyz, aabb, d_keys);
    const float ms_morton_gpu = t_morton_gpu.end_ms();

    std::vector<unsigned long long> h_keys_gpu(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(h_keys_gpu.data(), d_keys, static_cast<size_t>(n) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> h_keys_cpu(static_cast<size_t>(n));
    CpuTimer t_morton_cpu; t_morton_cpu.begin();
    compute_augmented_keys_cpu(n, h_xyz.data(), aabb, h_keys_cpu.data());
    const double ms_morton_cpu = t_morton_cpu.end_ms();

    bool morton_ok = true;
    for (int i = 0; i < n; ++i) if (h_keys_gpu[static_cast<size_t>(i)] != h_keys_cpu[static_cast<size_t>(i)]) { morton_ok = false; break; }
    all_ok &= morton_ok;
    std::printf("VERIFY(morton): %s (GPU augmented Morton keys bit-exact vs CPU reference for all %d points)\n",
               morton_ok ? "PASS" : "FAIL", n);

    // =========================================================================
    // STAGE 2 — sort (GPU thrust::sort vs CPU std::sort, both on a COPY).
    // =========================================================================
    unsigned long long* d_sorted_key = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sorted_key, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_sorted_key, d_keys, static_cast<size_t>(n) * sizeof(unsigned long long), cudaMemcpyDeviceToDevice));

    GpuTimer t_sort_gpu; t_sort_gpu.begin();
    launch_sort_augmented_keys(n, d_sorted_key);
    const float ms_sort_gpu = t_sort_gpu.end_ms();

    std::vector<unsigned long long> h_sorted_gpu(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(h_sorted_gpu.data(), d_sorted_key, static_cast<size_t>(n) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> h_sorted_cpu(static_cast<size_t>(n));
    CpuTimer t_sort_cpu; t_sort_cpu.begin();
    sort_keys_cpu(n, h_keys_cpu.data(), h_sorted_cpu.data());
    const double ms_sort_cpu = t_sort_cpu.end_ms();

    const bool sort_ok = (h_sorted_gpu == h_sorted_cpu);
    all_ok &= sort_ok;
    std::printf("VERIFY(sort): %s (GPU thrust::sort order bit-exact vs CPU std::sort twin, all %d points; "
               "augmented keys are pairwise distinct so this is the UNIQUE correct order)\n",
               sort_ok ? "PASS" : "FAIL", n);

    // =========================================================================
    // STAGE 3 — Karras radix-tree construction.
    // =========================================================================
    const int num_nodes = std::max(1, 2 * n - 1);
    LbvhNode* d_nodes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nodes, static_cast<size_t>(num_nodes) * sizeof(LbvhNode)));

    // Every parent slot starts at -1: the root (node 0) is the only node
    // build_radix_tree_kernel never writes as a CHILD, so it is the only
    // node whose parent field survives this initialization unchanged.
    {
        std::vector<LbvhNode> h_init(static_cast<size_t>(num_nodes));
        for (auto& nd : h_init) { nd.left = -1; nd.right = -1; nd.parent = -1; nd.point_idx = -1; }
        CUDA_CHECK(cudaMemcpy(d_nodes, h_init.data(), static_cast<size_t>(num_nodes) * sizeof(LbvhNode), cudaMemcpyHostToDevice));
    }

    GpuTimer t_tree_gpu; t_tree_gpu.begin();
    launch_init_leaves(n, d_xyz, d_sorted_key, d_nodes);
    launch_build_radix_tree(n, d_sorted_key, d_nodes);
    const float ms_tree_gpu = t_tree_gpu.end_ms();

    std::vector<LbvhNode> h_nodes_gpu(static_cast<size_t>(num_nodes));
    CUDA_CHECK(cudaMemcpy(h_nodes_gpu.data(), d_nodes, static_cast<size_t>(num_nodes) * sizeof(LbvhNode), cudaMemcpyDeviceToHost));

    std::vector<LbvhNode> h_nodes_cpu(static_cast<size_t>(num_nodes));
    for (auto& nd : h_nodes_cpu) { nd.left = -1; nd.right = -1; nd.parent = -1; nd.point_idx = -1; }
    CpuTimer t_tree_cpu; t_tree_cpu.begin();
    init_leaves_cpu(n, h_xyz.data(), h_sorted_cpu.data(), h_nodes_cpu.data());
    build_radix_tree_cpu(n, h_sorted_cpu.data(), h_nodes_cpu.data());
    const double ms_tree_cpu = t_tree_cpu.end_ms();

    bool topo_ok = true;
    for (int i = 0; i < num_nodes; ++i) {
        if (h_nodes_gpu[static_cast<size_t>(i)].left   != h_nodes_cpu[static_cast<size_t>(i)].left ||
            h_nodes_gpu[static_cast<size_t>(i)].right  != h_nodes_cpu[static_cast<size_t>(i)].right ||
            h_nodes_gpu[static_cast<size_t>(i)].parent != h_nodes_cpu[static_cast<size_t>(i)].parent ||
            h_nodes_gpu[static_cast<size_t>(i)].point_idx != h_nodes_cpu[static_cast<size_t>(i)].point_idx) {
            topo_ok = false; break;
        }
    }
    all_ok &= topo_ok;
    std::printf("VERIFY(topology): %s (GPU radix-tree left/right/parent/point_idx bit-exact vs independent "
               "CPU Karras-construction twin, all %d nodes: %d internal + %d leaves)\n",
               topo_ok ? "PASS" : "FAIL", num_nodes, std::max(0, n - 1), n);

    // =========================================================================
    // STAGE 4 — bottom-up AABB propagation.
    // =========================================================================
    unsigned int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(std::max(1, n - 1)) * sizeof(unsigned int)));

    GpuTimer t_aabb_gpu; t_aabb_gpu.begin();
    launch_propagate_aabb(n, d_nodes, d_flags);
    const float ms_aabb_gpu = t_aabb_gpu.end_ms();
    CUDA_CHECK(cudaMemcpy(h_nodes_gpu.data(), d_nodes, static_cast<size_t>(num_nodes) * sizeof(LbvhNode), cudaMemcpyDeviceToHost));

    CpuTimer t_aabb_cpu; t_aabb_cpu.begin();
    propagate_aabb_cpu(n, h_nodes_cpu.data());
    const double ms_aabb_cpu = t_aabb_cpu.end_ms();

    bool aabb_ok = true;
    for (int i = 0; i < num_nodes; ++i) {
        for (int a = 0; a < 3; ++a) {
            if (h_nodes_gpu[static_cast<size_t>(i)].aabb_min[a] != h_nodes_cpu[static_cast<size_t>(i)].aabb_min[a] ||
                h_nodes_gpu[static_cast<size_t>(i)].aabb_max[a] != h_nodes_cpu[static_cast<size_t>(i)].aabb_max[a]) {
                aabb_ok = false;
            }
        }
    }
    all_ok &= aabb_ok;
    std::printf("VERIFY(aabb): %s (GPU bottom-up AABB propagation bit-exact vs independent CPU post-order "
               "twin, all %d nodes; exact because float min/max is order-independent -- 02.02 lineage)\n",
               aabb_ok ? "PASS" : "FAIL", num_nodes);

    // GATE tree_validity
    int leaf_visits = 0, internal_visits = 0;
    bool containment_ok = true;
    const bool validity_ok = check_tree_validity(h_nodes_gpu, n, leaf_visits, internal_visits, containment_ok);
    all_ok &= validity_ok;
    std::printf("GATE tree_validity: %s (every leaf reached from root exactly once: %d/%d; every internal "
               "node's AABB exactly contains both children's AABB: %s; %d internal nodes checked)\n",
               validity_ok ? "PASS" : "FAIL", leaf_visits, n, containment_ok ? "yes" : "no", internal_visits);

    // =========================================================================
    // STAGE 5a — BVH radius search, all Q queries, GPU vs independent CPU twin.
    // =========================================================================
    int* d_radius_ids_bvh = nullptr; int* d_radius_count_bvh = nullptr;
    int* d_radius_overflow_bvh = nullptr; int* d_radius_visited_bvh = nullptr; int* d_radius_hwm_bvh = nullptr;
    const size_t radius_buf_elems = static_cast<size_t>(Q) * kMaxRadiusResults;
    CUDA_CHECK(cudaMalloc(&d_radius_ids_bvh, radius_buf_elems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radius_count_bvh, static_cast<size_t>(Q) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radius_overflow_bvh, static_cast<size_t>(Q) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radius_visited_bvh, static_cast<size_t>(Q) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radius_hwm_bvh, static_cast<size_t>(Q) * sizeof(int)));

    GpuTimer t_radius_bvh_gpu; t_radius_bvh_gpu.begin();
    launch_radius_search_bvh(d_nodes, n, d_xyz, d_queries, Q, kRadiusM,
                             d_radius_ids_bvh, d_radius_count_bvh, d_radius_overflow_bvh,
                             d_radius_visited_bvh, d_radius_hwm_bvh);
    const float ms_radius_bvh_gpu = t_radius_bvh_gpu.end_ms();

    std::vector<int> h_radius_ids_bvh(radius_buf_elems);
    std::vector<int> h_radius_count_bvh(static_cast<size_t>(Q)), h_radius_overflow_bvh(static_cast<size_t>(Q));
    std::vector<int> h_radius_visited_bvh(static_cast<size_t>(Q)), h_radius_hwm_bvh(static_cast<size_t>(Q));
    CUDA_CHECK(cudaMemcpy(h_radius_ids_bvh.data(), d_radius_ids_bvh, radius_buf_elems * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radius_count_bvh.data(), d_radius_count_bvh, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radius_overflow_bvh.data(), d_radius_overflow_bvh, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radius_visited_bvh.data(), d_radius_visited_bvh, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radius_hwm_bvh.data(), d_radius_hwm_bvh, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));

    bool radius_bvh_ok = true;
    int radius_overflow_total = 0;
    CpuTimer t_radius_bvh_cpu; t_radius_bvh_cpu.begin();
    for (int q = 0; q < Q; ++q) {
        std::vector<int> cpu_ids; bool cpu_overflow = false;
        radius_search_bvh_cpu(h_nodes_gpu.data(), n, h_xyz.data(), &h_queries[static_cast<size_t>(q) * 3], kRadiusM, cpu_ids, cpu_overflow);
        radius_overflow_total += h_radius_overflow_bvh[static_cast<size_t>(q)];
        const int cnt = h_radius_count_bvh[static_cast<size_t>(q)];
        if (static_cast<int>(cpu_ids.size()) != cnt) { radius_bvh_ok = false; continue; }
        for (int k = 0; k < cnt; ++k) {
            if (h_radius_ids_bvh[static_cast<size_t>(q) * kMaxRadiusResults + k] != cpu_ids[static_cast<size_t>(k)]) { radius_bvh_ok = false; break; }
        }
    }
    const double ms_radius_bvh_cpu = t_radius_bvh_cpu.end_ms();
    radius_bvh_ok = radius_bvh_ok && (radius_overflow_total == 0);
    all_ok &= radius_bvh_ok;
    std::printf("VERIFY(radius_bvh): %s (GPU BVH radius-search result sets exactly equal independent CPU "
               "BVH-traversal twin for all %d queries, r=%.2f m; zero result-buffer overflows)\n",
               radius_bvh_ok ? "PASS" : "FAIL", Q, kRadiusM);

    // =========================================================================
    // STAGE 5b — BVH KNN, all Q queries, GPU vs independent CPU twin.
    // =========================================================================
    int* d_knn_ids = nullptr; float* d_knn_dist2 = nullptr; int* d_knn_found = nullptr;
    int* d_knn_visited = nullptr; int* d_knn_hwm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_knn_ids, static_cast<size_t>(Q) * kQueryK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_knn_dist2, static_cast<size_t>(Q) * kQueryK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_knn_found, static_cast<size_t>(Q) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_knn_visited, static_cast<size_t>(Q) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_knn_hwm, static_cast<size_t>(Q) * sizeof(int)));

    GpuTimer t_knn_gpu; t_knn_gpu.begin();
    launch_knn_search_bvh(d_nodes, n, d_xyz, d_queries, Q, d_knn_ids, d_knn_dist2, d_knn_found, d_knn_visited, d_knn_hwm);
    const float ms_knn_gpu = t_knn_gpu.end_ms();

    std::vector<int> h_knn_ids(static_cast<size_t>(Q) * kQueryK);
    std::vector<float> h_knn_dist2(static_cast<size_t>(Q) * kQueryK);
    std::vector<int> h_knn_found(static_cast<size_t>(Q)), h_knn_visited(static_cast<size_t>(Q)), h_knn_hwm(static_cast<size_t>(Q));
    CUDA_CHECK(cudaMemcpy(h_knn_ids.data(), d_knn_ids, h_knn_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_dist2.data(), d_knn_dist2, h_knn_dist2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_found.data(), d_knn_found, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_visited.data(), d_knn_visited, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_hwm.data(), d_knn_hwm, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));

    bool knn_bvh_ok = true;
    CpuTimer t_knn_cpu; t_knn_cpu.begin();
    for (int q = 0; q < Q; ++q) {
        std::vector<KnnCandidate> cpu_res;
        knn_search_bvh_cpu(h_nodes_gpu.data(), n, h_xyz.data(), &h_queries[static_cast<size_t>(q) * 3], cpu_res);
        if (static_cast<int>(cpu_res.size()) != h_knn_found[static_cast<size_t>(q)]) { knn_bvh_ok = false; continue; }
        for (size_t k = 0; k < cpu_res.size(); ++k) {
            if (h_knn_ids[static_cast<size_t>(q) * kQueryK + k] != cpu_res[k].idx) { knn_bvh_ok = false; break; }
        }
    }
    const double ms_knn_cpu = t_knn_cpu.end_ms();
    all_ok &= knn_bvh_ok;
    std::printf("VERIFY(knn_bvh): %s (GPU BVH K=%d-NN result lists exactly equal independent CPU "
               "BVH-traversal twin under the documented (dist2,index) tie-break, all %d queries)\n",
               knn_bvh_ok ? "PASS" : "FAIL", kQueryK, Q);

    // =========================================================================
    // THE DOMAIN CONTRAST — fixed-radius voxel-hash search, GPU vs CPU twin,
    // then cross-checked against the BVH radius-search results above.
    // =========================================================================
    unsigned long long* d_hash_keys = nullptr;
    unsigned long long* d_hash_keys_scratch = nullptr; unsigned long long* d_hash_unique = nullptr;
    int* d_hash_idx_sorted = nullptr; int* d_hash_is_start = nullptr; int* d_hash_seg_start = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_keys, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hash_keys_scratch, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hash_unique, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hash_idx_sorted, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hash_is_start, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hash_seg_start, static_cast<size_t>(n) * sizeof(int)));

    GpuTimer t_hashbuild_gpu; t_hashbuild_gpu.begin();
    launch_compute_hash_keys(n, d_xyz, kRadiusM, d_hash_keys);
    const int num_voxels = launch_build_voxel_index(n, d_hash_keys, d_hash_keys_scratch, d_hash_idx_sorted,
                                                     d_hash_is_start, d_hash_seg_start, d_hash_unique);
    const float ms_hashbuild_gpu = t_hashbuild_gpu.end_ms();

    int* d_radius_ids_hash = nullptr; int* d_radius_count_hash = nullptr; int* d_radius_overflow_hash = nullptr;
    CUDA_CHECK(cudaMalloc(&d_radius_ids_hash, radius_buf_elems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radius_count_hash, static_cast<size_t>(Q) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radius_overflow_hash, static_cast<size_t>(Q) * sizeof(int)));

    GpuTimer t_radius_hash_gpu; t_radius_hash_gpu.begin();
    launch_radius_search_hash(d_xyz, d_hash_unique, num_voxels, d_hash_seg_start, d_hash_idx_sorted, n,
                              d_queries, Q, kRadiusM, kRadiusM, d_radius_ids_hash, d_radius_count_hash, d_radius_overflow_hash);
    const float ms_radius_hash_gpu = t_radius_hash_gpu.end_ms();

    std::vector<int> h_radius_ids_hash(radius_buf_elems);
    std::vector<int> h_radius_count_hash(static_cast<size_t>(Q)), h_radius_overflow_hash(static_cast<size_t>(Q));
    CUDA_CHECK(cudaMemcpy(h_radius_ids_hash.data(), d_radius_ids_hash, radius_buf_elems * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radius_count_hash.data(), d_radius_count_hash, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radius_overflow_hash.data(), d_radius_overflow_hash, static_cast<size_t>(Q) * sizeof(int), cudaMemcpyDeviceToHost));

    HashMapCpu h_hash_map;
    CpuTimer t_hash_cpu_build; t_hash_cpu_build.begin();
    build_hash_map_cpu(n, h_xyz.data(), kRadiusM, h_hash_map);
    const double ms_hash_cpu_build = t_hash_cpu_build.end_ms();

    bool radius_hash_ok = true;
    int hash_overflow_total = 0;
    bool hash_vs_bvh_ok = true;
    CpuTimer t_radius_hash_cpu; t_radius_hash_cpu.begin();
    for (int q = 0; q < Q; ++q) {
        std::vector<int> cpu_ids;
        radius_search_hash_cpu(h_hash_map, kRadiusM, kRadiusM, h_xyz.data(), &h_queries[static_cast<size_t>(q) * 3], cpu_ids);
        hash_overflow_total += h_radius_overflow_hash[static_cast<size_t>(q)];
        const int cnt = h_radius_count_hash[static_cast<size_t>(q)];
        if (static_cast<int>(cpu_ids.size()) != cnt) { radius_hash_ok = false; }
        else {
            for (int k = 0; k < cnt; ++k) {
                if (h_radius_ids_hash[static_cast<size_t>(q) * kMaxRadiusResults + k] != cpu_ids[static_cast<size_t>(k)]) { radius_hash_ok = false; break; }
            }
        }
        // Cross-check against the BVH radius-search result computed above,
        // for the SAME query and the SAME radius (GATE hash_vs_bvh_agreement).
        const int bvh_cnt = h_radius_count_bvh[static_cast<size_t>(q)];
        if (bvh_cnt != cnt) { hash_vs_bvh_ok = false; }
        else {
            for (int k = 0; k < cnt; ++k) {
                if (h_radius_ids_hash[static_cast<size_t>(q) * kMaxRadiusResults + k] !=
                    h_radius_ids_bvh[static_cast<size_t>(q) * kMaxRadiusResults + k]) { hash_vs_bvh_ok = false; break; }
            }
        }
    }
    const double ms_radius_hash_cpu = t_radius_hash_cpu.end_ms();
    radius_hash_ok = radius_hash_ok && (hash_overflow_total == 0);
    all_ok &= radius_hash_ok;
    all_ok &= hash_vs_bvh_ok;
    std::printf("VERIFY(radius_hash): %s (GPU voxel-hash radius-search result sets exactly equal independent "
               "CPU unordered_map twin for all %d queries; zero overflows; %d occupied voxels)\n",
               radius_hash_ok ? "PASS" : "FAIL", Q, num_voxels);
    std::printf("GATE hash_vs_bvh_agreement: %s (voxel-hash and BVH radius-search return IDENTICAL result "
               "sets for all %d queries at the SAME r=%.2f m)\n", hash_vs_bvh_ok ? "PASS" : "FAIL", Q, kRadiusM);

    // =========================================================================
    // GATE brute_force_anchor — the independent, tree/hash-free O(n*Q) oracle,
    // over the documented n_anchor-query subset (indices [0, n_anchor)).
    // =========================================================================
    bool anchor_ok = true;
    CpuTimer t_anchor; t_anchor.begin();
    for (int q = 0; q < hdr.n_anchor; ++q) {
        const float* qp = &h_queries[static_cast<size_t>(q) * 3];

        std::vector<int> bf_radius;
        radius_search_brute_force(n, h_xyz.data(), qp, kRadiusM, bf_radius);
        const int gpu_cnt = h_radius_count_bvh[static_cast<size_t>(q)];
        if (static_cast<int>(bf_radius.size()) != gpu_cnt) { anchor_ok = false; }
        else {
            for (int k = 0; k < gpu_cnt; ++k)
                if (h_radius_ids_bvh[static_cast<size_t>(q) * kMaxRadiusResults + k] != bf_radius[static_cast<size_t>(k)]) { anchor_ok = false; break; }
        }

        std::vector<KnnCandidate> bf_knn;
        knn_search_brute_force(n, h_xyz.data(), qp, bf_knn);
        if (static_cast<int>(bf_knn.size()) != h_knn_found[static_cast<size_t>(q)]) { anchor_ok = false; }
        else {
            for (size_t k = 0; k < bf_knn.size(); ++k)
                if (h_knn_ids[static_cast<size_t>(q) * kQueryK + k] != bf_knn[k].idx) { anchor_ok = false; break; }
        }
    }
    const double ms_anchor = t_anchor.end_ms();
    all_ok &= anchor_ok;
    std::printf("GATE brute_force_anchor: %s (GPU BVH radius-search AND KNN results exactly equal CPU "
               "brute-force O(N*Q) ground truth for the first %d sampled queries, both query types)\n",
               anchor_ok ? "PASS" : "FAIL", hdr.n_anchor);

    // =========================================================================
    // GATE density_contrast — the two designed queries, read back from the
    // header's fixed indices (idx_dense_query, idx_sparse_query).
    // =========================================================================
    const int iq_dense = hdr.idx_dense_query, iq_sparse = hdr.idx_sparse_query;
    const int dense_radius_count = h_radius_count_bvh[static_cast<size_t>(iq_dense)];
    const int sparse_radius_count = h_radius_count_bvh[static_cast<size_t>(iq_sparse)];
    const int dense_knn_found = h_knn_found[static_cast<size_t>(iq_dense)];
    const int sparse_knn_found = h_knn_found[static_cast<size_t>(iq_sparse)];
    const bool density_ok = (dense_radius_count >= 100) && (sparse_radius_count == 0) &&
                            (dense_knn_found == kQueryK) && (sparse_knn_found == kQueryK);
    all_ok &= density_ok;
    std::printf("GATE density_contrast: %s (fixed-radius search at r=%.2f m returns %d neighbors in the "
               "DENSE cluster query vs %d in the SPARSE far-field query; KNN returns %d/%d neighbors at "
               "BOTH -- the sensor-physics motivation for adaptive search, measured)\n",
               density_ok ? "PASS" : "FAIL", kRadiusM, dense_radius_count, sparse_radius_count,
               dense_knn_found, sparse_knn_found);

    // =========================================================================
    // [info] morton_locality — mean spatial distance of Morton-ADJACENT sorted
    // pairs vs ORIGINAL-scan-order-adjacent pairs (a cheap, deterministic,
    // honest proxy for "unrelated pairs": scan order interleaves beams and
    // revolutions, carrying no privileged spatial relationship of its own).
    // =========================================================================
    double sum_sorted = 0.0, sum_scan = 0.0;
    for (int k = 0; k + 1 < n; ++k) {
        const int pa = decode_point_index(h_sorted_gpu[static_cast<size_t>(k)]);
        const int pb = decode_point_index(h_sorted_gpu[static_cast<size_t>(k + 1)]);
        sum_sorted += std::sqrt(static_cast<double>(squared_distance3(&h_xyz[static_cast<size_t>(pa) * 3], &h_xyz[static_cast<size_t>(pb) * 3])));
        sum_scan += std::sqrt(static_cast<double>(squared_distance3(&h_xyz[static_cast<size_t>(k) * 3], &h_xyz[static_cast<size_t>(k + 1) * 3])));
    }
    const double mean_sorted = sum_sorted / std::max(1, n - 1);
    const double mean_scan = sum_scan / std::max(1, n - 1);
    std::printf("[info] morton_locality: mean distance between Morton-sorted-adjacent pairs = %.4f m vs "
               "original-scan-order-adjacent pairs = %.4f m (%.1fx closer -- quantifying the Z-order "
               "locality claim, n=%d pairs each)\n", mean_sorted, mean_scan,
               (mean_sorted > 1e-9) ? (mean_scan / mean_sorted) : 0.0, n - 1);

    // =========================================================================
    // [info] traversal_stats — mean/max nodes visited, stack high-water mark.
    // =========================================================================
    double mean_v_r; int max_v_r; mean_max(h_radius_visited_bvh, mean_v_r, max_v_r);
    double mean_v_k; int max_v_k; mean_max(h_knn_visited, mean_v_k, max_v_k);
    double mean_h_r; int max_h_r; mean_max(h_radius_hwm_bvh, mean_h_r, max_h_r);
    double mean_h_k; int max_h_k; mean_max(h_knn_hwm, mean_h_k, max_h_k);
    std::printf("[info] traversal_stats: radius-search nodes visited mean=%.1f max=%d, stack high-water "
               "mean=%.1f max=%d; KNN nodes visited mean=%.1f max=%d, stack high-water mean=%.1f max=%d "
               "(proven depth bound=%d, allocated stack=%d)\n",
               mean_v_r, max_v_r, mean_h_r, max_h_r, mean_v_k, max_v_k, mean_h_k, max_h_k, 62, kBvhStackSize);

    // =========================================================================
    // [time] lines — build stages and query throughput, GPU vs CPU (teaching
    // artifacts, never benchmark claims, CLAUDE.md paragraph 12).
    // =========================================================================
    std::printf("[time] build: morton GPU=%.3f ms / CPU=%.3f ms; sort GPU=%.3f ms / CPU=%.3f ms; "
               "radix-tree GPU=%.3f ms / CPU=%.3f ms; aabb-propagate GPU=%.3f ms / CPU=%.3f ms\n",
               static_cast<double>(ms_morton_gpu), ms_morton_cpu, static_cast<double>(ms_sort_gpu), ms_sort_cpu,
               static_cast<double>(ms_tree_gpu), ms_tree_cpu, static_cast<double>(ms_aabb_gpu), ms_aabb_cpu);
    std::printf("[time] hash build: GPU=%.3f ms (keys+sort+compact) / CPU=%.3f ms (unordered_map)\n",
               static_cast<double>(ms_hashbuild_gpu), ms_hash_cpu_build);
    const double qps_radius_bvh_gpu = (ms_radius_bvh_gpu > 0.0f) ? (1000.0 * Q / ms_radius_bvh_gpu) : 0.0;
    const double qps_radius_hash_gpu = (ms_radius_hash_gpu > 0.0f) ? (1000.0 * Q / ms_radius_hash_gpu) : 0.0;
    const double qps_knn_gpu = (ms_knn_gpu > 0.0f) ? (1000.0 * Q / ms_knn_gpu) : 0.0;
    const double qps_radius_bvh_cpu = (ms_radius_bvh_cpu > 0.0) ? (1000.0 * Q / ms_radius_bvh_cpu) : 0.0;
    const double qps_radius_hash_cpu = (ms_radius_hash_cpu > 0.0) ? (1000.0 * Q / ms_radius_hash_cpu) : 0.0;
    const double qps_knn_cpu = (ms_knn_cpu > 0.0) ? (1000.0 * Q / ms_knn_cpu) : 0.0;
    std::printf("[time] query throughput (Q=%d, teaching artifact not a benchmark): "
               "BVH-radius GPU=%.0f q/s / CPU=%.0f q/s; hash-radius GPU=%.0f q/s / CPU=%.0f q/s; "
               "BVH-KNN GPU=%.0f q/s / CPU=%.0f q/s\n",
               Q, qps_radius_bvh_gpu, qps_radius_bvh_cpu, qps_radius_hash_gpu, qps_radius_hash_cpu,
               qps_knn_gpu, qps_knn_cpu);
    std::printf("[time] brute-force anchor (Q=%d, O(N*Q), CPU only): %.1f ms\n", hdr.n_anchor, ms_anchor);

    // =========================================================================
    // Artifacts.
    // =========================================================================
    const std::string out_dir = resolve_out_dir(argv[0]);

    std::vector<int> dense_ids(h_radius_ids_bvh.begin() + static_cast<long>(iq_dense) * kMaxRadiusResults,
                               h_radius_ids_bvh.begin() + static_cast<long>(iq_dense) * kMaxRadiusResults + dense_radius_count);
    std::vector<int> sparse_knn_ids(h_knn_ids.begin() + static_cast<long>(iq_sparse) * kQueryK,
                                    h_knn_ids.begin() + static_cast<long>(iq_sparse) * kQueryK + sparse_knn_found);
    write_ppm_topview(out_dir + "/topview_density_contrast.ppm", h_xyz, n,
                      dense_ids, &h_queries[static_cast<size_t>(iq_dense) * 3],
                      sparse_knn_ids, &h_queries[static_cast<size_t>(iq_sparse) * 3],
                      800, 800, -8.5f, 8.5f, -8.5f, 8.5f);

    {
        std::ofstream f(out_dir + "/traversal_stats.csv");
        f << "# traversal_stats.csv -- per-query BVH traversal cost, project 02.05\n";
        f << "query_idx,radius_nodes_visited,radius_stack_hwm,knn_nodes_visited,knn_stack_hwm\n";
        for (int q = 0; q < Q; ++q) {
            f << q << ',' << h_radius_visited_bvh[static_cast<size_t>(q)] << ',' << h_radius_hwm_bvh[static_cast<size_t>(q)]
              << ',' << h_knn_visited[static_cast<size_t>(q)] << ',' << h_knn_hwm[static_cast<size_t>(q)] << '\n';
        }
    }
    {
        std::ofstream f(out_dir + "/timing.csv");
        f << "# timing.csv -- measured stage/query timings (teaching artifacts, not benchmarks), project 02.05\n";
        f << "stage,gpu_ms,cpu_ms\n";
        f << "morton," << ms_morton_gpu << ',' << ms_morton_cpu << '\n';
        f << "sort," << ms_sort_gpu << ',' << ms_sort_cpu << '\n';
        f << "radix_tree," << ms_tree_gpu << ',' << ms_tree_cpu << '\n';
        f << "aabb_propagate," << ms_aabb_gpu << ',' << ms_aabb_cpu << '\n';
        f << "hash_build," << ms_hashbuild_gpu << ',' << ms_hash_cpu_build << '\n';
        f << "radius_search_bvh_Q" << Q << ',' << ms_radius_bvh_gpu << ',' << ms_radius_bvh_cpu << '\n';
        f << "radius_search_hash_Q" << Q << ',' << ms_radius_hash_gpu << ',' << ms_radius_hash_cpu << '\n';
        f << "knn_search_bvh_Q" << Q << ',' << ms_knn_gpu << ',' << ms_knn_cpu << '\n';
    }
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.05\n";
        f << "metric,value\n";
        f << "n_points," << n << '\n';
        f << "n_queries," << Q << '\n';
        f << "num_voxels_hash," << num_voxels << '\n';
        f << "dense_radius_count," << dense_radius_count << '\n';
        f << "sparse_radius_count," << sparse_radius_count << '\n';
        f << "dense_knn_found," << dense_knn_found << '\n';
        f << "sparse_knn_found," << sparse_knn_found << '\n';
        f << "morton_locality_sorted_m," << mean_sorted << '\n';
        f << "morton_locality_scan_m," << mean_scan << '\n';
        f << "radius_nodes_visited_mean," << mean_v_r << '\n';
        f << "radius_nodes_visited_max," << max_v_r << '\n';
        f << "radius_stack_hwm_max," << max_h_r << '\n';
        f << "knn_nodes_visited_mean," << mean_v_k << '\n';
        f << "knn_nodes_visited_max," << max_v_k << '\n';
        f << "knn_stack_hwm_max," << max_h_k << '\n';
    }
    std::printf("ARTIFACT: wrote demo/out/{topview_density_contrast.ppm, traversal_stats.csv, timing.csv, gates_metrics.csv}\n");

    // ---- Cleanup --------------------------------------------------------------
    cudaFree(d_xyz); cudaFree(d_queries); cudaFree(d_keys); cudaFree(d_sorted_key);
    cudaFree(d_nodes); cudaFree(d_flags);
    cudaFree(d_radius_ids_bvh); cudaFree(d_radius_count_bvh); cudaFree(d_radius_overflow_bvh);
    cudaFree(d_radius_visited_bvh); cudaFree(d_radius_hwm_bvh);
    cudaFree(d_knn_ids); cudaFree(d_knn_dist2); cudaFree(d_knn_found); cudaFree(d_knn_visited); cudaFree(d_knn_hwm);
    cudaFree(d_hash_keys); cudaFree(d_hash_keys_scratch); cudaFree(d_hash_unique);
    cudaFree(d_hash_idx_sorted); cudaFree(d_hash_is_start); cudaFree(d_hash_seg_start);
    cudaFree(d_radius_ids_hash); cudaFree(d_radius_count_hash); cudaFree(d_radius_overflow_hash);

    // ---- Final verdict ----------------------------------------------------------
    if (all_ok) {
        std::printf("RESULT: PASS (VERIFY(morton/sort/topology/aabb/radius_bvh/knn_bvh/radius_hash) + "
                   "GATE(tree_validity/hash_vs_bvh_agreement/brute_force_anchor/density_contrast) all passed)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see the VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
