// ===========================================================================
// main.cu — entry point for project 02.15
//           Point cloud compression (octree/entropy) for fleet uplink
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load two committed clouds (data/sample/): a STRUCTURED map tile
//      (floor + walls + furniture-like boxes — surfaces, a 2-D manifold in
//      3-D space) and a PATHOLOGICAL cube of uniform-random points (no
//      surface at all) — the designed compressibility contrast this
//      project's whole payoff rests on (THEORY.md "The problem").
//   2. VERIFY STAGE (CLAUDE.md §5's GPU-vs-CPU gate, seven parts because
//      this codec has seven independently-verifiable stages — see
//      kernels.cuh's file header for the two-stage pipeline each part
//      belongs to): Morton codes, sort, per-level octree construction,
//      histogram, canonical-Huffman table, GPU encode, and an end-to-end
//      decode round trip — run ONCE, at the canonical depth D=10, on the
//      structured cloud. A failure here aborts before the sweep runs at
//      all (08.01/02.13's identical "fix before trusting anything below"
//      discipline).
//   3. SWEEP STAGE: the SAME already-verified GPU pipeline, run for both
//      clouds across all four swept depths (D=8,9,10,11) — the
//      rate-distortion study this project exists to produce.
//   4. GATES: lossless_roundtrip, distortion_bound, rate_monotonic,
//      entropy_payoff, entropy_bound, timing (all pass/fail, all rows) —
//      plus a fleet_arithmetic [info] block turning the measured
//      compression ratio into an illustrative GB/day uplink saving.
//   5. ARTIFACTS: rd_curve.csv (the whole sweep), occupancy_histogram.csv
//      (structured vs pathological at the canonical depth — the
//      compressibility physics made visible), three PGM top-view renders
//      (the original structured cloud, and its reconstruction at the
//      sweep's coarsest and finest depths), and gates_metrics.csv.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY:", "GATE:", "ARTIFACT:", "RESULT:" — NEVER carrying a
// measured number, only fixed verdicts and fixed descriptive text
// (08.01/02.13/02.14's identical discipline). Every measured number lives
// on an "[info]"/"[time]" line, deliberately unchecked by demo/run_demo.*.
// Change a stable line -> update demo/expected_output.txt in the same change.
//
// Read this after: kernels.cuh (the two-stage pipeline + why decode is
// host-only), then kernels.cu, then reference_cpu.cpp.
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
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Data loading — a tiny binary format, magic 'PCFU0001' + int32 n + n*(x,y,z)
// float32, meters, right-handed map frame (kernels.cuh's file header). See
// scripts/make_synthetic.py and data/README.md for the authoritative format
// table + SHA-256.
// ===========================================================================
struct Cloud {
    std::string name;
    std::vector<float> xyz;   // [n*3], meters, map frame
    int n = 0;
    bool loaded = false;
};

static Cloud load_cloud(const std::string& path, const std::string& name)
{
    Cloud c;
    c.name = name;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return c;

    char magic[8];
    f.read(magic, 8);
    if (!f.good() || std::memcmp(magic, "PCFU0001", 8) != 0) return c;

    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!f.good() || n <= 0) return c;

    c.xyz.resize(static_cast<size_t>(n) * 3);
    f.read(reinterpret_cast<char*>(c.xyz.data()),
          static_cast<std::streamsize>(c.xyz.size() * sizeof(float)));
    if (!f.good()) return c;

    c.n = n;
    c.loaded = true;
    return c;
}

// ---------------------------------------------------------------------------
// compute_cube_aabb — a cloud's bounding box, PADDED TO A CUBE (see
// kernels.cuh's SceneAABB comment for why: a uniform leaf size everywhere
// makes the quantization distortion bound a one-line formula). Run ONCE per
// cloud, on the host, before any device work — a plain O(n) reduction is
// instantaneous at this project's scale, so a GPU reduction kernel would
// add real complexity (a reduction pattern this project doesn't otherwise
// need) for zero measurable benefit (02.05's identical host-side AABB
// precedent).
//
// Padding: 0.5% of the longest raw extent on every side (a floor of 1 cm
// for a degenerate near-zero-extent cloud, which never occurs in this
// project's committed data but is guarded anyway — CLAUDE.md §13 honesty:
// never silently assume input is well-formed). The margin exists so no
// point sits EXACTLY on the [0, 2^D) quantization boundary, where a sub-ULP
// float rounding difference between two equivalent formulas could floor
// into the wrong cell — the same boundary-margin discipline 02.02's
// edge-cohort generator documents for its own predicate thresholds.
// ---------------------------------------------------------------------------
static SceneAABB compute_cube_aabb(const std::vector<float>& xyz, int n)
{
    float lo[3] = { 1.0e30f, 1.0e30f, 1.0e30f };
    float hi[3] = { -1.0e30f, -1.0e30f, -1.0e30f };
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) {
            const float v = xyz[static_cast<size_t>(i) * 3 + static_cast<size_t>(a)];
            if (v < lo[a]) lo[a] = v;
            if (v > hi[a]) hi[a] = v;
        }
    }
    float raw_extent = 0.0f;
    for (int a = 0; a < 3; ++a) raw_extent = std::max(raw_extent, hi[a] - lo[a]);

    float margin = raw_extent * 0.005f;
    if (margin <= 0.0f) margin = 0.01f;
    const float cube_extent = raw_extent + 2.0f * margin;

    SceneAABB aabb;
    for (int a = 0; a < 3; ++a) {
        const float center = 0.5f * (lo[a] + hi[a]);
        aabb.min[a] = center - 0.5f * cube_extent;
        aabb.max[a] = center + 0.5f * cube_extent;
    }
    aabb.extent_m = cube_extent;
    return aabb;
}

// ===========================================================================
// One rate-distortion row + the per-row gate predicates it feeds. See
// kernels.cuh's file header for the two-stage pipeline this row measures.
// ===========================================================================
struct RDRow {
    std::string cohort;
    int    depth = 0;
    int    num_points = 0;
    long long num_nodes = 0;                // M: total octree nodes across all levels (== octree bytes, raw)
    double raw_bits_per_point = 0.0;         // M*8 / n — the geometry stage alone, before entropy coding
    double huffman_bits_per_point = 0.0;     // encoded_bits / n — after entropy coding
    double mean_error_m = 0.0;
    double max_error_m = 0.0;
    double distortion_bound_m = 0.0;         // the analytic leaf half-diagonal bound
    double compression_ratio_vs_xyz32 = 0.0; // (n*12 bytes raw xyz) / (huffman bytes)
    double shannon_entropy_bits = 0.0;       // H of the measured 256-symbol occupancy histogram
    double huffman_avg_bits = 0.0;           // measured average Huffman code length
};

struct PipelineOutput {
    RDRow row;
    bool roundtrip_ok = false;
    bool distortion_ok = false;
    bool entropy_bound_ok = false;
    std::vector<float> leaf_xyz;   // decoded leaf centers (kept for the top-view renders)
    int hist[256] = { 0 };         // the measured occupancy-byte histogram (kept for the histogram artifact)
    double wall_ms = 0.0;          // encode-pipeline wall time (codes..encode; see the timing gate)
};

// ---------------------------------------------------------------------------
// run_depth_pipeline — the WHOLE two-stage codec for one (cohort, depth)
// pair: STAGE 1 (Morton codes -> sort -> per-level octree construction) and
// STAGE 2 (histogram -> canonical Huffman table -> GPU encode), followed by
// decode + distortion + rate/entropy measurement (always) and, when
// do_verify is set, the full GPU-vs-CPU twin comparison the VERIFY stage
// needs (kernels.cuh's declarations name each twin this calls).
//
// Every device buffer is allocated fresh and freed before returning — a
// self-contained function signature over a "fastest possible" one, the same
// trade 02.02's compact_with_flags documents: this function runs 9 times
// total in this demo (1 verify + 8 sweep rows), so the extra allocator
// overhead is immeasurable next to the kernels it measures.
// ---------------------------------------------------------------------------
static PipelineOutput run_depth_pipeline(const std::string& cohort_name,
                                         const std::vector<float>& xyz, int n,
                                         const SceneAABB& aabb, int D,
                                         bool do_verify, bool& verify_all_pass)
{
    PipelineOutput out;
    verify_all_pass = true;   // meaningless unless do_verify, but never left uninitialized

    // ---- device buffers: point cloud + Morton codes -------------------------
    float* d_xyz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xyz, xyz.data(), static_cast<size_t>(n) * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));

    uint64_t* d_codes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_codes, static_cast<size_t>(n) * sizeof(uint64_t)));

    CpuTimer wall;
    wall.begin();   // the ENCODE pipeline's wall clock — see the timing gate; decode is excluded on purpose
                    // (the fleet uplink only needs the ROBOT's encode side to be fast; the cloud decodes
                    // at its leisure — kernels.cuh's file header frames the asymmetry).

    // ---- STAGE 1a: Morton codes ----------------------------------------------
    launch_compute_codes(n, d_xyz, aabb, D, d_codes);

    // Keep the UNSORTED codes (original point order) for the per-point
    // distortion measurement below — pairing code[i] with xyz[i] directly,
    // with no need to consult the sorted array at all.
    std::vector<uint64_t> codes_unsorted(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(codes_unsorted.data(), d_codes, codes_unsorted.size() * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    launch_sort_codes(n, d_codes);   // in place, ascending

    std::vector<uint64_t> codes_sorted(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(codes_sorted.data(), d_codes, codes_sorted.size() * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    // ---- STAGE 1b: per-level octree construction -----------------------------
    int* d_is_start = nullptr;
    int* d_node_id = nullptr;
    uint32_t* d_occ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_is_start, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_id, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_occ, static_cast<size_t>(n) * sizeof(uint32_t)));

    std::vector<uint8_t> occupancy_stream;             // the concatenated geometry code, all levels
    std::vector<int> node_counts(static_cast<size_t>(D));
    occupancy_stream.reserve(static_cast<size_t>(n) * 2);   // a loose guess; grows as needed regardless

    for (int level = 0; level < D; ++level) {
        const OctreeLevelResult r = launch_build_octree_level(
            n, d_codes, D, level, d_is_start, d_node_id, d_occ);
        node_counts[static_cast<size_t>(level)] = r.num_nodes;

        std::vector<uint32_t> words(static_cast<size_t>(r.num_nodes));
        CUDA_CHECK(cudaMemcpy(words.data(), d_occ, words.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        for (const uint32_t w : words) occupancy_stream.push_back(static_cast<uint8_t>(w & 0xFFu));
    }

    const int M = static_cast<int>(occupancy_stream.size());   // total octree node count, this depth

    // ---- STAGE 2: histogram ---------------------------------------------------
    uint8_t* d_symbols = nullptr;
    CUDA_CHECK(cudaMalloc(&d_symbols, static_cast<size_t>(M) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_symbols, occupancy_stream.data(), static_cast<size_t>(M) * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    int* d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    launch_compute_histogram(M, d_symbols, d_hist);
    CUDA_CHECK(cudaMemcpy(out.hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // ---- STAGE 2: canonical Huffman table (host-side; see kernels.cu) --------
    HuffmanTable table;
    build_huffman_table(out.hist, table);

    // ---- STAGE 2: GPU encode ---------------------------------------------------
    const EncodeResult enc = launch_huffman_encode(M, d_symbols, table);

    out.wall_ms = wall.end_ms();

    // ======================= VERIFY (canonical depth only) ===================
    if (do_verify) {
        std::vector<uint64_t> codes_cpu(static_cast<size_t>(n));
        compute_codes_cpu(n, xyz.data(), aabb, D, codes_cpu.data());
        const bool morton_ok = (codes_cpu == codes_unsorted);
        std::printf("VERIFY morton: %s (GPU vs CPU Morton codes, all points, bit-exact)\n",
                    morton_ok ? "PASS" : "FAIL");

        std::vector<uint64_t> codes_sorted_cpu(static_cast<size_t>(n));
        sort_codes_cpu(n, codes_cpu.data(), codes_sorted_cpu.data());
        const bool sort_ok = (codes_sorted_cpu == codes_sorted);
        std::printf("VERIFY sort: %s (GPU thrust::sort vs CPU std::sort, full array, exact)\n",
                    sort_ok ? "PASS" : "FAIL");

        std::vector<std::vector<uint8_t>> occ_cpu;
        build_octree_levels_cpu(n, codes_sorted_cpu.data(), D, occ_cpu);
        bool levels_ok = true;
        {
            size_t pos = 0;
            for (int level = 0; level < D && levels_ok; ++level) {
                const int cnt = node_counts[static_cast<size_t>(level)];
                if (cnt != static_cast<int>(occ_cpu[static_cast<size_t>(level)].size())) {
                    levels_ok = false;
                    break;
                }
                for (int k = 0; k < cnt; ++k) {
                    if (occupancy_stream[pos + static_cast<size_t>(k)] != occ_cpu[static_cast<size_t>(level)][static_cast<size_t>(k)]) {
                        levels_ok = false;
                        break;
                    }
                }
                pos += static_cast<size_t>(cnt);
            }
        }
        std::printf("VERIFY octree_levels: %s (GPU vs CPU node counts + occupancy bytes, every level, exact)\n",
                    levels_ok ? "PASS" : "FAIL");

        int hist_cpu[256];
        compute_histogram_cpu(M, occupancy_stream.data(), hist_cpu);
        const bool hist_ok = (std::memcmp(out.hist, hist_cpu, sizeof(hist_cpu)) == 0);
        std::printf("VERIFY histogram: %s (GPU vs CPU 256-bin occupancy-byte histogram, exact)\n",
                    hist_ok ? "PASS" : "FAIL");

        HuffmanTable table_cpu;
        build_huffman_table_cpu(hist_cpu, table_cpu);
        bool table_ok = true;
        for (int s = 0; s < 256; ++s) {
            if (table.len[s] != table_cpu.len[s] || table.bits[s] != table_cpu.bits[s]) { table_ok = false; break; }
        }
        std::printf("VERIFY huffman_table: %s (independent heap-based vs linear-scan canonical-Huffman builders, all 256 symbols, exact)\n",
                    table_ok ? "PASS" : "FAIL");

        std::vector<uint8_t> packed_cpu;
        long long bits_cpu = 0;
        huffman_encode_cpu(M, occupancy_stream.data(), table, packed_cpu, bits_cpu);
        const bool encode_ok = (bits_cpu == enc.num_bits) && (packed_cpu == enc.packed);
        std::printf("VERIFY encode_bitstream: %s (GPU map+scan+scatter vs CPU serial bit-writer, packed bytes + bit count, exact)\n",
                    encode_ok ? "PASS" : "FAIL");

        std::vector<uint8_t> decoded_symbols;
        huffman_decode_cpu(enc.packed, enc.num_bits, M, table, decoded_symbols);
        std::vector<float> leaf_xyz_decoded;
        octree_decode_leaf_centers(decoded_symbols, D, aabb, leaf_xyz_decoded);
        std::vector<uint64_t> unique_leaf_codes;
        octree_unique_leaf_codes(n, codes_sorted.data(), D, unique_leaf_codes);
        bool roundtrip_ok_v = (decoded_symbols == occupancy_stream) &&
                              (leaf_xyz_decoded.size() == unique_leaf_codes.size() * 3);
        for (size_t k = 0; roundtrip_ok_v && k < unique_leaf_codes.size(); ++k) {
            float gx = 0, gy = 0, gz = 0;
            decode_leaf_center(unique_leaf_codes[k], D, aabb, gx, gy, gz);
            if (leaf_xyz_decoded[k * 3 + 0] != gx || leaf_xyz_decoded[k * 3 + 1] != gy ||
                leaf_xyz_decoded[k * 3 + 2] != gz) {
                roundtrip_ok_v = false;
            }
        }
        std::printf("VERIFY roundtrip: %s (decode(encode(cloud)) reproduces the depth-%d quantized leaf set exactly, in order)\n",
                    roundtrip_ok_v ? "PASS" : "FAIL", D);

        verify_all_pass = morton_ok && sort_ok && levels_ok && hist_ok && table_ok && encode_ok && roundtrip_ok_v;
    }

    // ======================= decode (always) — feeds gates + renders =========
    std::vector<uint8_t> decoded_symbols2;
    huffman_decode_cpu(enc.packed, enc.num_bits, M, table, decoded_symbols2);
    std::vector<float> leaf_xyz;
    octree_decode_leaf_centers(decoded_symbols2, D, aabb, leaf_xyz);
    std::vector<uint64_t> unique_leaf_codes2;
    octree_unique_leaf_codes(n, codes_sorted.data(), D, unique_leaf_codes2);

    bool roundtrip_ok = (decoded_symbols2 == occupancy_stream) &&
                        (leaf_xyz.size() == unique_leaf_codes2.size() * 3);
    for (size_t k = 0; roundtrip_ok && k < unique_leaf_codes2.size(); ++k) {
        float gx = 0, gy = 0, gz = 0;
        decode_leaf_center(unique_leaf_codes2[k], D, aabb, gx, gy, gz);
        if (leaf_xyz[k * 3 + 0] != gx || leaf_xyz[k * 3 + 1] != gy || leaf_xyz[k * 3 + 2] != gz) {
            roundtrip_ok = false;
        }
    }

    // ======================= distortion (always) ==============================
    // Per ORIGINAL point (not just unique leaves): how far is the leaf CENTER
    // its own code maps to from the point's true position — the quantization
    // error a fleet's cloud side actually sees after decoding.
    double sum_err = 0.0, max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        float lx = 0, ly = 0, lz = 0;
        decode_leaf_center(codes_unsorted[static_cast<size_t>(i)], D, aabb, lx, ly, lz);
        const double dx = static_cast<double>(lx) - static_cast<double>(xyz[static_cast<size_t>(i) * 3 + 0]);
        const double dy = static_cast<double>(ly) - static_cast<double>(xyz[static_cast<size_t>(i) * 3 + 1]);
        const double dz = static_cast<double>(lz) - static_cast<double>(xyz[static_cast<size_t>(i) * 3 + 2]);
        const double e = std::sqrt(dx * dx + dy * dy + dz * dz);
        sum_err += e;
        if (e > max_err) max_err = e;
    }
    const double mean_err = sum_err / static_cast<double>(n);
    const double leaf_m = static_cast<double>(aabb.extent_m) / static_cast<double>(1u << D);
    const double bound = leaf_m * 1.7320508075688772 / 2.0;   // sqrt(3) — THEORY.md "The math" derives this
    const double dist_eps = bound * 1.0e-4 + 1.0e-6;           // float-rounding slack, not a correctness fudge
    const bool distortion_ok = (max_err <= bound + dist_eps);

    // ======================= entropy (always) ==================================
    double H = 0.0;
    for (int s = 0; s < 256; ++s) {
        if (out.hist[s] <= 0) continue;
        const double p = static_cast<double>(out.hist[s]) / static_cast<double>(M);
        H -= p * std::log2(p);
    }
    const double huff_avg = static_cast<double>(enc.num_bits) / static_cast<double>(M);
    const double ent_eps = 1.0e-6;
    const bool entropy_bound_ok = (huff_avg >= H - ent_eps) && (huff_avg <= H + 1.0 + ent_eps);

    // ======================= rate / ratio (always) ==============================
    const double raw_bits_per_point = (static_cast<double>(M) * 8.0) / static_cast<double>(n);
    const double huff_bits_per_point = static_cast<double>(enc.num_bits) / static_cast<double>(n);
    const double raw_bytes_xyz32 = static_cast<double>(n) * 12.0;
    const double huff_bytes = static_cast<double>(enc.num_bits) / 8.0;
    const double ratio = raw_bytes_xyz32 / huff_bytes;

    out.row.cohort = cohort_name;
    out.row.depth = D;
    out.row.num_points = n;
    out.row.num_nodes = M;
    out.row.raw_bits_per_point = raw_bits_per_point;
    out.row.huffman_bits_per_point = huff_bits_per_point;
    out.row.mean_error_m = mean_err;
    out.row.max_error_m = max_err;
    out.row.distortion_bound_m = bound;
    out.row.compression_ratio_vs_xyz32 = ratio;
    out.row.shannon_entropy_bits = H;
    out.row.huffman_avg_bits = huff_avg;

    out.roundtrip_ok = roundtrip_ok;
    out.distortion_ok = distortion_ok;
    out.entropy_bound_ok = entropy_bound_ok;
    out.leaf_xyz = std::move(leaf_xyz);

    CUDA_CHECK(cudaFree(d_xyz));
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_is_start));
    CUDA_CHECK(cudaFree(d_node_id));
    CUDA_CHECK(cudaFree(d_occ));
    CUDA_CHECK(cudaFree(d_symbols));
    CUDA_CHECK(cudaFree(d_hist));

    return out;
}

// ===========================================================================
// GateResult — one row of the pass/fail summary (also written to
// demo/out/gates_metrics.csv). NEVER printed with its numbers on a stable
// line (02.14's identical discipline — see the file header).
// ===========================================================================
struct GateResult { std::string name; double measured; double threshold; bool pass; std::string note; };

// ---------------------------------------------------------------------------
// Tiny PGM (P5, grayscale) top-view canvases — no image library needed
// (CLAUDE.md §5: this format IS "write the header, then raw bytes", the
// same choice 02.14's write_pgm makes). Each canvas is a top-down (X-Y)
// splat: one pixel per point/leaf-center, white on black.
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, const std::vector<unsigned char>& gray, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return f.good();
}

static bool render_topview(const std::string& path, const std::vector<float>& xyz, int n,
                           const SceneAABB& aabb, int img_size)
{
    std::vector<unsigned char> img(static_cast<size_t>(img_size) * static_cast<size_t>(img_size), 0);
    for (int i = 0; i < n; ++i) {
        const float x = xyz[static_cast<size_t>(i) * 3 + 0];
        const float y = xyz[static_cast<size_t>(i) * 3 + 1];
        float u = (x - aabb.min[0]) / aabb.extent_m;             // [0,1] across the cube's X extent
        float v = 1.0f - (y - aabb.min[1]) / aabb.extent_m;      // flip so +Y (North-like) renders upward
        if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
        if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
        int px = static_cast<int>(u * static_cast<float>(img_size - 1));
        int py = static_cast<int>(v * static_cast<float>(img_size - 1));
        if (px < 0) px = 0; if (px >= img_size) px = img_size - 1;
        if (py < 0) py = 0; if (py >= img_size) py = img_size - 1;
        img[static_cast<size_t>(py) * static_cast<size_t>(img_size) + static_cast<size_t>(px)] = 255;
    }
    return write_pgm(path, img, img_size, img_size);
}

// ===========================================================================
// main — see the file header for the five stages this orchestrates.
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

    std::printf("[demo] Point cloud compression (octree/entropy) for fleet uplink (project 02.15)\n");
    print_device_info();
    std::printf("PROBLEM: occupancy-octree geometry coding + canonical-Huffman entropy coding, "
                "depth sweep D in {8,9,10,11}, structured map tile vs pathological uniform-random cube\n");

    // ---- 0) Data ----------------------------------------------------------
    const std::string structured_path = find_data_file(data_dir_override, argv[0], "structured_map.bin");
    const std::string pathological_path = find_data_file(data_dir_override, argv[0], "pathological_cube.bin");
    if (structured_path.empty() || pathological_path.empty()) {
        std::printf("DATA: NOT FOUND — data/sample/structured_map.bin or pathological_cube.bin missing "
                    "(run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    Cloud structured = load_cloud(structured_path, "structured");
    Cloud pathological = load_cloud(pathological_path, "pathological");
    if (!structured.loaded || !pathological.loaded) {
        std::printf("DATA: MALFORMED — see data/README.md for the expected binary layout\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }
    std::printf("DATA: structured map tile n=%d points, pathological uniform-random cube n=%d points [synthetic]\n",
                structured.n, pathological.n);

    const SceneAABB aabb_structured = compute_cube_aabb(structured.xyz, structured.n);
    const SceneAABB aabb_pathological = compute_cube_aabb(pathological.xyz, pathological.n);
    std::printf("[info] structured cube: %.3f m side, min=(%.2f,%.2f,%.2f)\n",
                static_cast<double>(aabb_structured.extent_m),
                static_cast<double>(aabb_structured.min[0]), static_cast<double>(aabb_structured.min[1]),
                static_cast<double>(aabb_structured.min[2]));
    std::printf("[info] pathological cube: %.3f m side, min=(%.2f,%.2f,%.2f)\n",
                static_cast<double>(aabb_pathological.extent_m),
                static_cast<double>(aabb_pathological.min[0]), static_cast<double>(aabb_pathological.min[1]),
                static_cast<double>(aabb_pathological.min[2]));

    const int canonical_depth = kSweepDepths[kCanonicalDepthIndex];

    // ======================= VERIFY STAGE =====================================
    std::printf("[info] VERIFY stage: canonical depth D=%d, structured cloud, %d points\n",
                canonical_depth, structured.n);
    bool verify_all_pass = false;
    GpuTimer verify_gt;   // wraps the whole verify call for a visible [time] line
    verify_gt.begin();
    PipelineOutput verify_out = run_depth_pipeline("structured", structured.xyz, structured.n,
                                                   aabb_structured, canonical_depth,
                                                   /*do_verify=*/true, verify_all_pass);
    // GpuTimer.end_ms() also fully synchronizes the device, matching every
    // other project's "time the region, then trust the results" ordering.
    const float verify_region_ms = verify_gt.end_ms();
    std::printf("[time] VERIFY stage total (GPU pipeline + CPU twins + decode): %.1f ms\n",
                static_cast<double>(verify_region_ms));
    if (!verify_all_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement in the verify stage — fix before trusting any gate below)\n");
        return 1;
    }

    // ======================= SWEEP STAGE =======================================
    struct CohortRef { std::string name; const Cloud* cloud; const SceneAABB* aabb; };
    const CohortRef cohorts[2] = {
        { "structured",   &structured,   &aabb_structured },
        { "pathological", &pathological, &aabb_pathological },
    };

    std::vector<PipelineOutput> pipeline_outputs;   // index = cohort_idx*kNumSweepDepths + depth_idx
    pipeline_outputs.reserve(2 * kNumSweepDepths);

    for (int ci = 0; ci < 2; ++ci) {
        for (int di = 0; di < kNumSweepDepths; ++di) {
            const int D = kSweepDepths[di];
            bool dummy = true;
            PipelineOutput o = run_depth_pipeline(cohorts[ci].name, cohorts[ci].cloud->xyz,
                                                  cohorts[ci].cloud->n, *cohorts[ci].aabb, D,
                                                  /*do_verify=*/false, dummy);
            std::printf("[info] sweep %s D=%d: nodes=%lld raw=%.2f bits/pt huffman=%.2f bits/pt "
                        "mean_err=%.4f m max_err=%.4f m bound=%.4f m ratio=%.2fx H=%.3f bits Lavg=%.3f bits\n",
                        cohorts[ci].name.c_str(), D, o.row.num_nodes,
                        o.row.raw_bits_per_point, o.row.huffman_bits_per_point,
                        o.row.mean_error_m, o.row.max_error_m, o.row.distortion_bound_m,
                        o.row.compression_ratio_vs_xyz32, o.row.shannon_entropy_bits, o.row.huffman_avg_bits);
            pipeline_outputs.push_back(std::move(o));
        }
    }

    const PipelineOutput& structured_canon = pipeline_outputs[static_cast<size_t>(0 * kNumSweepDepths + kCanonicalDepthIndex)];
    const PipelineOutput& pathological_canon = pipeline_outputs[static_cast<size_t>(1 * kNumSweepDepths + kCanonicalDepthIndex)];

    // ======================= GATES ==============================================
    std::vector<GateResult> gates;

    // lossless_roundtrip — the correctness anchor: EVERY row, exact.
    {
        int total = 0, passed = 0;
        for (const auto& o : pipeline_outputs) { ++total; if (o.roundtrip_ok) ++passed; }
        GateResult g;
        g.name = "lossless_roundtrip";
        g.measured = passed;
        g.threshold = total;
        g.pass = (passed == total);
        g.note = "decode(encode(cloud)) reproduces the depth-D quantized leaf set exactly, both cohorts, all swept depths";
        gates.push_back(g);
    }

    // distortion_bound — the analytic gate: measured max error <= derived leaf half-diagonal, every row.
    {
        int total = 0, passed = 0;
        double worst_ratio = 0.0;
        for (const auto& o : pipeline_outputs) {
            ++total;
            if (o.distortion_ok) ++passed;
            const double r = o.row.max_error_m / o.row.distortion_bound_m;
            if (r > worst_ratio) worst_ratio = r;
        }
        GateResult g;
        g.name = "distortion_bound";
        g.measured = worst_ratio;
        g.threshold = 1.0;
        g.pass = (passed == total);
        g.note = "measured max reconstruction error <= leaf half-diagonal bound (sqrt(3)/2 * leaf_m), every row";
        gates.push_back(g);
    }

    // rate_monotonic — the nested-prefix property (THEORY.md "The math"):
    // node count non-decreasing and max error non-increasing as D grows,
    // within each cohort.
    {
        bool ok = true;
        for (int ci = 0; ci < 2; ++ci) {
            for (int di = 1; di < kNumSweepDepths; ++di) {
                const auto& prev = pipeline_outputs[static_cast<size_t>(ci * kNumSweepDepths + di - 1)];
                const auto& cur = pipeline_outputs[static_cast<size_t>(ci * kNumSweepDepths + di)];
                if (cur.row.num_nodes < prev.row.num_nodes) ok = false;
                if (cur.row.max_error_m > prev.row.max_error_m + 1.0e-9) ok = false;
            }
        }
        GateResult g;
        g.name = "rate_monotonic";
        g.measured = ok ? 1.0 : 0.0;
        g.threshold = 1.0;
        g.pass = ok;
        g.note = "octree node count non-decreasing and max reconstruction error non-increasing as D grows, both cohorts";
        gates.push_back(g);
    }

    // entropy_payoff — the compressibility physics, asserted at the
    // canonical depth. Two genuinely different effects are in play here,
    // and the gate measures the one that actually carries the "surfaces
    // compress" story (a real finding from this project's own development
    // run, kept honest rather than smoothed over — see THEORY.md "How we
    // verify correctness" for the full account):
    //   (a) STAGE 1 (octree geometry) dominates: a structured surface
    //       needs FAR fewer octree nodes to describe than the same point
    //       count scattered with no surface at all (measured ~2.4x fewer
    //       nodes at D=10 in this project's own run) — this is what
    //       compression_ratio_vs_xyz32 (the END-TO-END ratio, geometry AND
    //       entropy coding combined) captures, and where structured wins
    //       decisively and consistently across every swept depth.
    //   (b) STAGE 2 (the occupancy-byte HISTOGRAM's own skew) does NOT
    //       reliably favor structured data in isolation: a sparse,
    //       scattered cloud's rare non-empty nodes disproportionately show
    //       exactly ONE occupied child (a lone point in an otherwise empty
    //       cell), concentrating its histogram onto the 8 single-bit
    //       symbols and giving it a LOWER measured entropy than a surface,
    //       whose nodes show a wider variety of multi-bit crossing
    //       patterns. Gating on the Huffman/raw ratio ALONE would assert a
    //       false claim; this project measures it honestly (in rd_curve.csv)
    //       without gating a cross-cohort comparison on it.
    // The gate below therefore checks (a) — structured must beat
    // pathological's END-TO-END ratio by a wide, measured margin — plus a
    // sanity check that Huffman still trims a real fraction off structured's
    // own raw octree stream (it does: entropy coding is never worthless,
    // even where it does not favor one cohort over the other).
    {
        const double s_ratio = structured_canon.row.compression_ratio_vs_xyz32;
        const double p_ratio = pathological_canon.row.compression_ratio_vs_xyz32;
        const double s_huff_over_raw = structured_canon.row.huffman_bits_per_point / structured_canon.row.raw_bits_per_point;
        const bool structured_huffman_helps = (s_huff_over_raw <= 0.90);
        const bool structured_beats_pathological_overall = (s_ratio >= p_ratio * 1.5);
        GateResult g;
        g.name = "entropy_payoff";
        g.measured = s_ratio;
        g.threshold = p_ratio;
        g.pass = structured_huffman_helps && structured_beats_pathological_overall;
        g.note = "end-to-end compression ratio vs raw xyz32 at canonical depth D=10: structured must beat pathological by >=1.5x AND Huffman must cut >=10% off structured's own raw octree stream";
        gates.push_back(g);
    }

    // entropy_bound — the Shannon-optimality gate: measured Huffman average
    // length within the classic [H, H+1) band, every row.
    {
        int total = 0, passed = 0;
        for (const auto& o : pipeline_outputs) { ++total; if (o.entropy_bound_ok) ++passed; }
        GateResult g;
        g.name = "entropy_bound";
        g.measured = passed;
        g.threshold = total;
        g.pass = (passed == total);
        g.note = "measured Huffman avg bits/symbol within [H, H+1) of the measured Shannon entropy H, every row";
        gates.push_back(g);
    }

    // timing — canonical-depth structured encode pipeline vs an illustrative
    // fleet-uplink batching budget (wide margin, honestly measured).
    {
        const double budget_ms = 5000.0;   // see PRACTICE.md for the batching-cadence assumption this budget models
        GateResult g;
        g.name = "timing";
        g.measured = structured_canon.wall_ms;
        g.threshold = budget_ms;
        g.pass = (structured_canon.wall_ms <= budget_ms);
        g.note = "canonical-depth (D=10) structured encode pipeline wall time <= illustrative uplink-batching budget";
        gates.push_back(g);
    }

    // ======================= fleet arithmetic [info] ===========================
    {
        const int kFleetRobots = 50;
        const int kTilesPerRobotPerDay = 20;   // an incremental map-delta tile every ~72 min of active mapping
        const double raw_MB = static_cast<double>(structured.n) * 12.0 / (1024.0 * 1024.0);
        const double compressed_MB = raw_MB / structured_canon.row.compression_ratio_vs_xyz32;
        const double raw_GB_day = static_cast<double>(kFleetRobots) * kTilesPerRobotPerDay * raw_MB / 1024.0;
        const double compressed_GB_day = static_cast<double>(kFleetRobots) * kTilesPerRobotPerDay * compressed_MB / 1024.0;
        std::printf("[info] fleet arithmetic (illustrative teaching arithmetic, CLAUDE.md §12 — not a capacity plan): "
                    "%d robots x %d map tiles/robot/day x (%.3f MB raw -> %.3f MB compressed @ measured %.2fx, D=%d) "
                    "= %.2f GB/day raw vs %.2f GB/day compressed (%.2f GB/day saved, %.1f%% reduction)\n",
                    kFleetRobots, kTilesPerRobotPerDay, raw_MB, compressed_MB,
                    structured_canon.row.compression_ratio_vs_xyz32, canonical_depth,
                    raw_GB_day, compressed_GB_day, raw_GB_day - compressed_GB_day,
                    100.0 * (1.0 - compressed_GB_day / raw_GB_day));
    }

    // ======================= ARTIFACTS ===========================================
    const std::string out_dir = resolve_out_dir(argv[0]);

    bool rd_csv_ok = false;
    {
        std::ofstream f(out_dir + "/rd_curve.csv");
        if (f.is_open()) {
            f << std::fixed << std::setprecision(6);
            f << "cohort,depth,num_points,num_nodes,raw_bits_per_point,huffman_bits_per_point,"
                 "mean_error_m,max_error_m,distortion_bound_m,compression_ratio_vs_xyz32,"
                 "shannon_entropy_bits,huffman_avg_bits\n";
            for (const auto& o : pipeline_outputs) {
                const RDRow& r = o.row;
                f << r.cohort << ',' << r.depth << ',' << r.num_points << ',' << r.num_nodes << ','
                  << r.raw_bits_per_point << ',' << r.huffman_bits_per_point << ','
                  << r.mean_error_m << ',' << r.max_error_m << ',' << r.distortion_bound_m << ','
                  << r.compression_ratio_vs_xyz32 << ',' << r.shannon_entropy_bits << ','
                  << r.huffman_avg_bits << '\n';
            }
            rd_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/rd_curve.csv (%d rows: 2 cohorts x %d depths)\n",
                rd_csv_ok ? "wrote" : "FAILED to write", static_cast<int>(pipeline_outputs.size()), kNumSweepDepths);

    bool hist_csv_ok = false;
    {
        std::ofstream f(out_dir + "/occupancy_histogram.csv");
        if (f.is_open()) {
            f << "symbol,count_structured,count_pathological\n";
            for (int s = 0; s < 256; ++s) {
                f << s << ',' << structured_canon.hist[s] << ',' << pathological_canon.hist[s] << '\n';
            }
            hist_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/occupancy_histogram.csv (256 symbols, canonical depth D=%d)\n",
                hist_csv_ok ? "wrote" : "FAILED to write", canonical_depth);

    const int kImgSize = 256;
    const bool render_orig_ok = render_topview(out_dir + "/topview_original.pgm", structured.xyz, structured.n,
                                               aabb_structured, kImgSize);
    const PipelineOutput& structured_d8 = pipeline_outputs[static_cast<size_t>(0 * kNumSweepDepths + 0)];   // kSweepDepths[0]==8
    const PipelineOutput& structured_d11 = pipeline_outputs[static_cast<size_t>(0 * kNumSweepDepths + 3)];  // kSweepDepths[3]==11
    const int n_leaves_d8 = static_cast<int>(structured_d8.leaf_xyz.size() / 3);
    const int n_leaves_d11 = static_cast<int>(structured_d11.leaf_xyz.size() / 3);
    const bool render_d8_ok = render_topview(out_dir + "/topview_recon_d8.pgm", structured_d8.leaf_xyz,
                                             n_leaves_d8, aabb_structured, kImgSize);
    const bool render_d11_ok = render_topview(out_dir + "/topview_recon_d11.pgm", structured_d11.leaf_xyz,
                                              n_leaves_d11, aabb_structured, kImgSize);
    const bool renders_ok = render_orig_ok && render_d8_ok && render_d11_ok;
    std::printf("ARTIFACT: %s demo/out/topview_original.pgm, topview_recon_d8.pgm, topview_recon_d11.pgm "
                "(before/after top-view, %dx%d)\n", renders_ok ? "wrote" : "FAILED to write", kImgSize, kImgSize);

    bool gates_csv_ok = false;
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        if (f.is_open()) {
            f << std::fixed << std::setprecision(6);
            f << "gate,measured,threshold,verdict,note\n";
            for (const auto& g : gates) {
                f << g.name << ',' << g.measured << ',' << g.threshold << ','
                  << (g.pass ? "PASS" : "FAIL") << ",\"" << g.note << "\"\n";
            }
            gates_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/gates_metrics.csv (%d gates)\n",
                gates_csv_ok ? "wrote" : "FAILED to write", static_cast<int>(gates.size()));

    // ======================= FINAL VERDICT ======================================
    for (const auto& g : gates) std::printf("GATE: %s %s\n", g.name.c_str(), g.pass ? "PASS" : "FAIL");

    bool all_gates_pass = true;
    for (const auto& g : gates) all_gates_pass = all_gates_pass && g.pass;
    const bool artifacts_ok = rd_csv_ok && hist_csv_ok && renders_ok && gates_csv_ok;
    const bool success = verify_all_pass && all_gates_pass && artifacts_ok;

    if (success)
        std::printf("RESULT: PASS (verify stage + all gates passed; rate-distortion sweep written)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above and stderr)\n");
    return success ? 0 : 1;
}
