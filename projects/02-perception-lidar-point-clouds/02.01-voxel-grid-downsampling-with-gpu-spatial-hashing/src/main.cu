// ===========================================================================
// main.cu — entry point for project 02.01 (Voxel-grid downsampling with GPU
//           spatial hashing)
//
// Role in the project
// -------------------
// Orchestration: load the committed LiDAR scan, run BOTH downsampling
// methods (Method A: atomic open-addressing hash table; Method B: Thrust
// sort + fixed-order segmented reduction) on the GPU, run their CPU
// oracles, verify everything, quantify the data-structure and determinism
// lessons the catalog bullet asks for, and write the demo artifacts.
// kernels.cu holds the GPU kernels; reference_cpu.cpp holds the CPU twins;
// kernels.cuh is the shared contract all three agree on.
//
// Output contract (load-bearing! — same convention as every project in this
// repo, see docs/COMMENTING_STANDARD.md and e.g. 01.04's main.cu)
// -------------------------------------------------------------------------
// demo/run_demo.ps1 diffs the STABLE lines of this program's stdout against
// demo/expected_output.txt. Stable = "[demo]", "PROBLEM:", "DATA:", every
// "VERIFY(...)"/"GATE ...:" verdict, "ARTIFACT:", and "RESULT:" — every one
// of those lines is either constant or derived ONLY from the fixed
// committed input file (so it never varies run to run). "[info]" and
// "[time]" lines carry machine- or run-varying NUMBERS (GPU name, measured
// tolerances, timings) and are deliberately NOT diffed. If you change a
// stable line here you MUST update demo/expected_output.txt in the same
// change, and vice versa.
//
// Read this after: kernels.cuh (the interface, worth reading first this
// time — it explains the two methods end to end). Read this before:
// kernels.cu, reference_cpu.cpp.
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

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ===========================================================================
// Verification tolerances — MEASURED THEN MARGINED (CLAUDE.md §12), not
// guessed. On the reference machine (RTX 2080 SUPER, sm_75, Release|x64,
// the committed data/sample/lidar_scan.bin, 198,534 points -> 7,132
// occupied voxels) an actual run measured:
//     max |Method A (GPU) - independent CPU hash-map oracle| = 3.34e-06 m
//     max |Method A (GPU) - Method B (GPU)|                  = 4.29e-06 m
// i.e. a few micrometers — consistent with float32's ~1e-7 relative
// precision at these ~1-20 m coordinate magnitudes, accumulated over the
// DENSE adversarial cluster's ~3,000-point voxel (THEORY.md "Numerical
// considerations" derives this bound from first principles). Both
// tolerances below margin that measurement by roughly 30x — generous
// enough to absorb a different GPU architecture's atomic scheduling
// without being so loose it would miss a real accumulation bug.
// ---------------------------------------------------------------------------
static const float kToleranceMethodA_m = 1.0e-4f;      // 100 um (~30x the measured 3.34e-6 m)
static const float kToleranceCrossMethod_m = 1.0e-4f;  // 100 um (~23x the measured 4.29e-6 m)
// centroid_containment: the mean of values individually inside a half-open
// interval is mathematically always inside it; this epsilon exists purely
// to absorb float division/summation rounding at our coordinate scale
// (room extends to ~tens of meters -> float32 ULP there is ~1e-6 m), with
// generous headroom.
static const float kContainmentEpsilon_m = 1.0e-4f;

// ===========================================================================
// Binary sample format — see scripts/make_synthetic.py's write_binary_sample()
// for the authoritative description. Read back with EXPLICIT fixed-width
// primitive reads (never a raw struct fread) so this loader does not depend
// on any compiler's struct-padding rules — the same portability reasoning
// util/paths.h gives for avoiding <filesystem>.
// ===========================================================================
struct SampleHeader {
    int32_t n_total = 0, n_normal = 0, n_dense = 0, n_sparse = 0;
    float   leaf_m = 0.0f, sensor_height_m = 0.0f;
    int32_t num_beams = 0;
};

// ---------------------------------------------------------------------------
// load_point_cloud — read the committed lidar_scan.bin into a flat host xyz
// buffer plus its header fields. Returns false (with a message on stderr)
// on any I/O or format problem — callers must check and fail loudly rather
// than run on garbage data (CLAUDE.md §13 honesty).
// ---------------------------------------------------------------------------
static bool load_point_cloud(const std::string& path, SampleHeader& hdr, std::vector<float>& xyz)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "error: could not open sample file '%s'\n", path.c_str());
        return false;
    }

    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "VXLSCAN1", 8) != 0) {
        std::fprintf(stderr, "error: '%s' does not start with the expected VXLSCAN1 magic\n", path.c_str());
        return false;
    }

    f.read(reinterpret_cast<char*>(&hdr.n_total), 4);
    f.read(reinterpret_cast<char*>(&hdr.n_normal), 4);
    f.read(reinterpret_cast<char*>(&hdr.n_dense), 4);
    f.read(reinterpret_cast<char*>(&hdr.n_sparse), 4);
    f.read(reinterpret_cast<char*>(&hdr.leaf_m), 4);
    f.read(reinterpret_cast<char*>(&hdr.sensor_height_m), 4);
    f.read(reinterpret_cast<char*>(&hdr.num_beams), 4);
    int32_t reserved = 0;
    f.read(reinterpret_cast<char*>(&reserved), 4);
    if (!f || hdr.n_total <= 0 || hdr.n_total != hdr.n_normal + hdr.n_dense + hdr.n_sparse) {
        std::fprintf(stderr, "error: '%s' has a malformed or inconsistent header\n", path.c_str());
        return false;
    }

    // Data/code consistency check: the leaf size baked into the sample by
    // scripts/make_synthetic.py (used to design the adversarial regions)
    // must equal kernels.cuh's compiled-in kVoxelLeafM — if someone changes
    // one without the other, every downstream gate would be comparing
    // against the wrong geometry. Fail loudly instead of silently drifting.
    if (std::fabs(hdr.leaf_m - kVoxelLeafM) > 1.0e-6f) {
        std::fprintf(stderr, "error: sample leaf_m=%.6f does not match kernels.cuh kVoxelLeafM=%.6f "
                             "-- regenerate the sample or update the constant, they must agree\n",
                     static_cast<double>(hdr.leaf_m), static_cast<double>(kVoxelLeafM));
        return false;
    }

    xyz.resize(static_cast<size_t>(hdr.n_total) * 3);
    f.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(xyz.size() * sizeof(float)));
    if (!f) {
        std::fprintf(stderr, "error: '%s' is truncated (expected %d points' worth of xyz data)\n",
                     path.c_str(), hdr.n_total);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_ppm_topview — a minimal, hand-rolled binary PPM (P6) writer: an
// orthographic TOP VIEW (looking down -z) of a point set, ignoring z
// entirely. No image library is used (CLAUDE.md §5 "no black boxes" — PPM's
// header is three text lines and the rest is raw RGB bytes, cheap enough to
// hand-roll and genuinely simpler than linking a PNG library for a teaching
// artifact). Every point is drawn as a single white pixel on a black
// background; points map to the SAME pixel when they are within one
// pixel's world-space footprint of each other, which is deliberate — the
// visual density difference between the "before" and "after" renders IS
// the point of this artifact (demo/README.md explains what to look at).
// ---------------------------------------------------------------------------
static void write_ppm_topview(const std::string& path, const float* xyz, int n,
                              int width, int height, float half_extent_m)
{
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3, 0);  // black background
    const float scale = static_cast<float>(width) / (2.0f * half_extent_m);          // pixels per meter

    for (int i = 0; i < n; ++i) {
        const float x = xyz[i * 3 + 0];
        const float y = xyz[i * 3 + 1];
        const int px = static_cast<int>((x + half_extent_m) * scale);
        const int py = static_cast<int>((half_extent_m - y) * scale);   // flip y: world +y is "up" in the image
        if (px < 0 || px >= width || py < 0 || py >= height) continue;   // outside the view window: skip
        const size_t idx = (static_cast<size_t>(py) * width + px) * 3;
        pixels[idx + 0] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
    }

    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << ' ' << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

// A small host-side voxel summary used for the by-KEY comparisons below
// (Method A vs its CPU twin, Method A vs Method B, and the determinism
// studies) — a float centroid plus its point count, keyed by packed voxel
// key in the unordered_maps this file builds.
struct VoxelSummaryF {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    unsigned int count = 0;
};

static std::unordered_map<unsigned long long, VoxelSummaryF>
to_map(const std::vector<float>& xyz, const std::vector<unsigned int>& count,
      const std::vector<unsigned long long>& key, int num_voxels)
{
    std::unordered_map<unsigned long long, VoxelSummaryF> m;
    m.reserve(static_cast<size_t>(num_voxels) * 2);
    for (int v = 0; v < num_voxels; ++v) {
        VoxelSummaryF s;
        s.x = xyz[v * 3 + 0];
        s.y = xyz[v * 3 + 1];
        s.z = xyz[v * 3 + 2];
        s.count = count[v];
        m.emplace(key[v], s);
    }
    return m;
}

// ===========================================================================
// Method A — run the whole reset -> insert -> compact pipeline once, copy
// results back to host vectors, and time the GPU portion (insert + compact;
// NOT the reset, which is bookkeeping overhead every determinism-study rerun
// pays identically and which would otherwise dilute the "real work" figure).
// ===========================================================================
struct MethodAResult {
    std::vector<float> xyz;
    std::vector<unsigned int> count;
    std::vector<unsigned long long> key;
    unsigned int num_occupied = 0;
    unsigned int overflow = 0;
    float gpu_ms = 0.0f;
};

static MethodAResult run_method_a(int n, const float* d_xyz, const unsigned long long* d_keys,
                                  HashTableGPU table, int* d_probe_len,
                                  unsigned int* d_overflow_count,
                                  float* d_outA_xyz, unsigned int* d_outA_count,
                                  unsigned long long* d_outA_key, unsigned int* d_num_occupied)
{
    MethodAResult r;

    launch_hash_reset(table);   // every fresh attempt needs an empty table (see hash_reset_kernel's comment)
    CUDA_CHECK(cudaMemset(d_overflow_count, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_num_occupied, 0, sizeof(unsigned int)));

    GpuTimer timer;
    timer.begin();
    launch_hash_insert(n, d_xyz, d_keys, table, d_probe_len, d_overflow_count);
    launch_hash_compact(table, d_outA_xyz, d_outA_count, d_outA_key, d_num_occupied);
    r.gpu_ms = timer.end_ms();   // synchronizes -> num_occupied is ready to read back

    CUDA_CHECK(cudaMemcpy(&r.num_occupied, d_num_occupied, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&r.overflow, d_overflow_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    r.xyz.resize(static_cast<size_t>(r.num_occupied) * 3);
    r.count.resize(r.num_occupied);
    r.key.resize(r.num_occupied);
    if (r.num_occupied > 0) {
        CUDA_CHECK(cudaMemcpy(r.xyz.data(), d_outA_xyz, r.xyz.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.count.data(), d_outA_count, r.count.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.key.data(), d_outA_key, r.key.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }
    return r;
}

// ===========================================================================
// Method B — run the sort + segmented-reduce pipeline once and copy results
// back. (No reset needed: launch_sort_based_downsample is self-contained —
// it copies d_keys_in into scratch fresh on every call.)
// ===========================================================================
struct MethodBResult {
    std::vector<float> xyz;
    std::vector<unsigned int> count;
    std::vector<unsigned long long> key;
    int num_voxels = 0;
    float gpu_ms = 0.0f;
};

static MethodBResult run_method_b(int n, const float* d_xyz, const unsigned long long* d_keys,
                                  unsigned long long* d_keys_scratch, int* d_idx_scratch,
                                  int* d_is_start_scratch, int* d_seg_start_out,
                                  float* d_outB_xyz, unsigned int* d_outB_count,
                                  unsigned long long* d_outB_key)
{
    MethodBResult r;

    GpuTimer timer;
    timer.begin();
    r.num_voxels = launch_sort_based_downsample(n, d_xyz, d_keys, d_keys_scratch, d_idx_scratch,
                                                d_is_start_scratch, d_seg_start_out,
                                                d_outB_xyz, d_outB_count, d_outB_key);
    r.gpu_ms = timer.end_ms();

    r.xyz.resize(static_cast<size_t>(r.num_voxels) * 3);
    r.count.resize(static_cast<size_t>(r.num_voxels));
    r.key.resize(static_cast<size_t>(r.num_voxels));
    if (r.num_voxels > 0) {
        CUDA_CHECK(cudaMemcpy(r.xyz.data(), d_outB_xyz, r.xyz.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.count.data(), d_outB_count, r.count.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.key.data(), d_outB_key, r.key.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }
    return r;
}

// ---------------------------------------------------------------------------
// centroid_containment_ok — the geometric invariant gate: the mean of a set
// of points, each individually inside a voxel's half-open AABB, must itself
// lie inside that AABB (a free consequence of convexity — this gate is
// checking the PIPELINE'S bookkeeping, not re-deriving the math). Runs over
// a whole result set; returns the count of violations (0 == PASS) and, via
// out-params, the worst violation seen (for an honest failure message).
// ---------------------------------------------------------------------------
static int centroid_containment_violations(const std::vector<float>& xyz,
                                           const std::vector<unsigned long long>& key,
                                           int num_voxels, float leaf, float eps)
{
    int violations = 0;
    for (int v = 0; v < num_voxels; ++v) {
        int32_t vx, vy, vz;
        unpack_voxel_key(key[static_cast<size_t>(v)], vx, vy, vz);
        const float lo[3] = { vx * leaf, vy * leaf, vz * leaf };
        const float hi[3] = { lo[0] + leaf, lo[1] + leaf, lo[2] + leaf };
        for (int a = 0; a < 3; ++a) {
            const float c = xyz[static_cast<size_t>(v) * 3 + static_cast<size_t>(a)];
            if (c < lo[a] - eps || c > hi[a] + eps) { ++violations; break; }
        }
    }
    return violations;
}

// partition_invariant: per-voxel counts must sum EXACTLY to n (every input
// point mapped to exactly one voxel — no point dropped, none double-counted).
static bool partition_invariant_ok(const std::vector<unsigned int>& count, int n)
{
    long long sum = 0;
    for (unsigned int c : count) sum += c;
    return sum == static_cast<long long>(n);
}

int main(int argc, char** argv)
{
    bool all_ok = true;   // ANDed with every VERIFY/GATE result below; drives the final RESULT: line

    // ---- 0) Identify the demo, the GPU, load data --------------------------
    std::printf("[demo] voxel-grid downsampling: GPU spatial hashing (Method A, atomic) vs "
               "GPU sort-based reduction (Method B, deterministic) (project 02.01)\n");
    print_device_info();

    const std::string data_path = find_data_file("", argv[0], "lidar_scan.bin");
    if (data_path.empty()) {
        std::fprintf(stderr, "error: could not locate data/sample/lidar_scan.bin -- run "
                             "scripts/make_synthetic.py first (see ../data/README.md)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return EXIT_FAILURE;
    }

    SampleHeader hdr;
    std::vector<float> h_xyz;
    if (!load_point_cloud(data_path, hdr, h_xyz)) {
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return EXIT_FAILURE;
    }
    const int n = hdr.n_total;

    // Table capacity: sized from N (not from occupied-voxel count, which we
    // do not know yet) so that even the worst realistic case -- every point
    // landing in its own voxel, occupied == N -- keeps the REALIZED load
    // factor at or under kTargetLoadFactor (see kernels.cuh for the theory;
    // THEORY.md "The math" ties this to open-addressing probe-length bounds).
    const unsigned int capacity = next_pow2(static_cast<unsigned int>(
        std::ceil(static_cast<double>(n) / kTargetLoadFactor)));

    std::printf("PROBLEM: N=%d points (16-beam spinning LiDAR scan = %d, "
               "+ %d-point adversarial dense cluster + %d-point adversarial sparse region), "
               "voxel leaf L=%.2f m, hash table capacity=%u slots\n",
               n, hdr.n_normal, hdr.n_dense, hdr.n_sparse,
               static_cast<double>(kVoxelLeafM), capacity);
    std::printf("DATA: data/sample/lidar_scan.bin [synthetic, seed 42, xorshift32, "
               "see scripts/make_synthetic.py]\n");

    // ---- 1) Device allocations ----------------------------------------------
    float* d_xyz = nullptr;
    unsigned long long* d_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Method A: the hash table + per-point probe-length bookkeeping + output.
    HashTableGPU table{};
    table.capacity = static_cast<int>(capacity);
    CUDA_CHECK(cudaMalloc(&table.keys, static_cast<size_t>(capacity) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&table.sum_x, static_cast<size_t>(capacity) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&table.sum_y, static_cast<size_t>(capacity) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&table.sum_z, static_cast<size_t>(capacity) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&table.count, static_cast<size_t>(capacity) * sizeof(unsigned int)));
    int* d_probe_len = nullptr;
    unsigned int* d_overflow_count = nullptr;
    unsigned int* d_num_occupied = nullptr;
    CUDA_CHECK(cudaMalloc(&d_probe_len, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_overflow_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_num_occupied, sizeof(unsigned int)));
    float* d_outA_xyz = nullptr; unsigned int* d_outA_count = nullptr; unsigned long long* d_outA_key = nullptr;
    CUDA_CHECK(cudaMalloc(&d_outA_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));   // upper bound: occupied <= n
    CUDA_CHECK(cudaMalloc(&d_outA_count, static_cast<size_t>(n) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_outA_key, static_cast<size_t>(n) * sizeof(unsigned long long)));

    // Method B: sort/compaction scratch + output (all sized n -- occupied
    // voxels can never exceed the point count, kernels.cuh documents this).
    unsigned long long* d_keys_scratch = nullptr;
    int* d_idx_scratch = nullptr; int* d_is_start_scratch = nullptr; int* d_seg_start_out = nullptr;
    float* d_outB_xyz = nullptr; unsigned int* d_outB_count = nullptr; unsigned long long* d_outB_key = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys_scratch, static_cast<size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_idx_scratch, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start_scratch, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_start_out, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_outB_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outB_count, static_cast<size_t>(n) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_outB_key, static_cast<size_t>(n) * sizeof(unsigned long long)));

    // ---- 2) Keys: GPU computation + CPU twin + VERIFY(keys) -----------------
    CpuTimer cpu_timer;
    cpu_timer.begin();
    std::vector<unsigned long long> keys_cpu(static_cast<size_t>(n));
    compute_keys_cpu(n, h_xyz.data(), kVoxelLeafM, keys_cpu.data());
    const double cpu_keys_ms = cpu_timer.end_ms();

    launch_compute_keys(n, d_xyz, kVoxelLeafM, d_keys);
    std::vector<unsigned long long> keys_gpu(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(keys_gpu.data(), d_keys, static_cast<size_t>(n) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    int key_mismatches = 0;
    for (int i = 0; i < n; ++i) if (keys_gpu[static_cast<size_t>(i)] != keys_cpu[static_cast<size_t>(i)]) ++key_mismatches;
    const bool verify_keys_ok = (key_mismatches == 0);
    all_ok = all_ok && verify_keys_ok;
    std::printf("VERIFY(keys): %s (GPU voxel keys bit-exact vs CPU reference for all points)\n",
               verify_keys_ok ? "PASS" : "FAIL");
    if (!verify_keys_ok) std::fprintf(stderr, "  %d/%d point keys mismatched\n", key_mismatches, n);

    // ---- 3) Method B: GPU run + CPU bit-exact twin + VERIFY(method_b) -------
    MethodBResult mb = run_method_b(n, d_xyz, d_keys, d_keys_scratch, d_idx_scratch,
                                    d_is_start_scratch, d_seg_start_out,
                                    d_outB_xyz, d_outB_count, d_outB_key);

    cpu_timer.begin();
    std::vector<float> mb_cpu_xyz(static_cast<size_t>(n) * 3);
    std::vector<unsigned int> mb_cpu_count(static_cast<size_t>(n));
    std::vector<unsigned long long> mb_cpu_key(static_cast<size_t>(n));
    const int num_voxels_b_cpu = sort_based_downsample_cpu(n, h_xyz.data(), kVoxelLeafM,
                                                            mb_cpu_xyz.data(), mb_cpu_count.data(), mb_cpu_key.data());
    const double cpu_method_b_ms = cpu_timer.end_ms();

    bool verify_b_ok = (mb.num_voxels == num_voxels_b_cpu);
    int b_mismatches = 0;
    if (verify_b_ok) {
        for (int v = 0; v < mb.num_voxels; ++v) {
            const size_t vs = static_cast<size_t>(v);
            const bool same = (mb.xyz[vs*3+0] == mb_cpu_xyz[vs*3+0]) &&
                              (mb.xyz[vs*3+1] == mb_cpu_xyz[vs*3+1]) &&
                              (mb.xyz[vs*3+2] == mb_cpu_xyz[vs*3+2]) &&
                              (mb.count[vs] == mb_cpu_count[vs]) &&
                              (mb.key[vs] == mb_cpu_key[vs]);
            if (!same) ++b_mismatches;
        }
        verify_b_ok = (b_mismatches == 0);
    }
    all_ok = all_ok && verify_b_ok;
    std::printf("VERIFY(method_b): %s (GPU sort-based centroids/counts/keys bit-exact vs "
               "CPU fixed-order twin)\n", verify_b_ok ? "PASS" : "FAIL");
    if (!verify_b_ok) {
        std::fprintf(stderr, "  num_voxels: GPU=%d CPU=%d, mismatched rows=%d\n",
                     mb.num_voxels, num_voxels_b_cpu, b_mismatches);
    }

    // ---- 4) Method A: GPU run + CPU independent twin + VERIFY(method_a) -----
    MethodAResult ma = run_method_a(n, d_xyz, d_keys, table, d_probe_len, d_overflow_count,
                                    d_outA_xyz, d_outA_count, d_outA_key, d_num_occupied);
    const bool no_overflow_a = (ma.overflow == 0);

    cpu_timer.begin();
    std::unordered_map<unsigned long long, VoxelAccumD> ma_cpu;
    hashmap_downsample_cpu(n, h_xyz.data(), kVoxelLeafM, ma_cpu);
    const double cpu_method_a_ms = cpu_timer.end_ms();

    bool verify_a_ok = no_overflow_a && (ma.num_occupied == ma_cpu.size());
    float max_delta_a = 0.0f;
    if (verify_a_ok) {
        for (unsigned int v = 0; v < ma.num_occupied; ++v) {
            const auto it = ma_cpu.find(ma.key[v]);
            if (it == ma_cpu.end()) { verify_a_ok = false; break; }
            const VoxelAccumD& acc = it->second;
            if (acc.count != ma.count[v]) { verify_a_ok = false; break; }
            const float cx = static_cast<float>(acc.sx / acc.count);
            const float cy = static_cast<float>(acc.sy / acc.count);
            const float cz = static_cast<float>(acc.sz / acc.count);
            const float dx = std::fabs(ma.xyz[v*3+0] - cx);
            const float dy = std::fabs(ma.xyz[v*3+1] - cy);
            const float dz = std::fabs(ma.xyz[v*3+2] - cz);
            max_delta_a = std::max(max_delta_a, std::max(dx, std::max(dy, dz)));
            if (dx > kToleranceMethodA_m || dy > kToleranceMethodA_m || dz > kToleranceMethodA_m) verify_a_ok = false;
        }
    }
    all_ok = all_ok && verify_a_ok;
    std::printf("VERIFY(method_a): %s (GPU atomic-hash centroids within documented tolerance of "
               "independent CPU hash-map twin; occupancy exact; zero hash-table overflows)\n",
               verify_a_ok ? "PASS" : "FAIL");
    if (!verify_a_ok) {
        std::fprintf(stderr, "  occupancy: GPU=%u CPU=%zu, overflow=%u, max delta so far=%.6e m\n",
                     ma.num_occupied, ma_cpu.size(), ma.overflow, static_cast<double>(max_delta_a));
    }

    // ---- 5) GATE cross_method_agreement: Method A vs Method B ---------------
    auto map_b = to_map(mb.xyz, mb.count, mb.key, mb.num_voxels);
    bool gate_cross_ok = (ma.num_occupied == static_cast<unsigned int>(mb.num_voxels));
    float max_delta_cross = 0.0f;
    if (gate_cross_ok) {
        for (unsigned int v = 0; v < ma.num_occupied; ++v) {
            const auto it = map_b.find(ma.key[v]);
            if (it == map_b.end()) { gate_cross_ok = false; break; }
            if (it->second.count != ma.count[v]) { gate_cross_ok = false; break; }
            const float dx = std::fabs(ma.xyz[v*3+0] - it->second.x);
            const float dy = std::fabs(ma.xyz[v*3+1] - it->second.y);
            const float dz = std::fabs(ma.xyz[v*3+2] - it->second.z);
            max_delta_cross = std::max(max_delta_cross, std::max(dx, std::max(dy, dz)));
            if (dx > kToleranceCrossMethod_m || dy > kToleranceCrossMethod_m || dz > kToleranceCrossMethod_m) gate_cross_ok = false;
        }
    }
    all_ok = all_ok && gate_cross_ok;
    std::printf("GATE cross_method_agreement: %s (Method A vs Method B centroids agree within "
               "documented tolerance; occupancy exact)\n", gate_cross_ok ? "PASS" : "FAIL");

    // ---- 6) GATE partition_invariant -----------------------------------------
    const bool gate_partition_ok = partition_invariant_ok(ma.count, n) && partition_invariant_ok(mb.count, n);
    all_ok = all_ok && gate_partition_ok;
    std::printf("GATE partition_invariant: %s (per-voxel counts sum to N for both methods)\n",
               gate_partition_ok ? "PASS" : "FAIL");

    // ---- 7) GATE centroid_containment ----------------------------------------
    const int viol_a = centroid_containment_violations(ma.xyz, ma.key, static_cast<int>(ma.num_occupied), kVoxelLeafM, kContainmentEpsilon_m);
    const int viol_b = centroid_containment_violations(mb.xyz, mb.key, mb.num_voxels, kVoxelLeafM, kContainmentEpsilon_m);
    const bool gate_containment_ok = (viol_a == 0) && (viol_b == 0);
    all_ok = all_ok && gate_containment_ok;
    std::printf("GATE centroid_containment: %s (every centroid lies inside its voxel AABB, both methods)\n",
               gate_containment_ok ? "PASS" : "FAIL");

    // ---- 8) Determinism studies (3 runs each) --------------------------------
    // Method B: fully deterministic by design -- 3 fresh runs must be
    // BYTE-IDENTICAL. Reruns reuse the same device scratch/output buffers
    // (each run overwrites them completely) but are copied back to
    // SEPARATE host vectors for comparison.
    MethodBResult mb2 = run_method_b(n, d_xyz, d_keys, d_keys_scratch, d_idx_scratch,
                                     d_is_start_scratch, d_seg_start_out,
                                     d_outB_xyz, d_outB_count, d_outB_key);
    MethodBResult mb3 = run_method_b(n, d_xyz, d_keys, d_keys_scratch, d_idx_scratch,
                                     d_is_start_scratch, d_seg_start_out,
                                     d_outB_xyz, d_outB_count, d_outB_key);
    bool gate_determinism_b_ok = (mb.num_voxels == mb2.num_voxels) && (mb.num_voxels == mb3.num_voxels);
    if (gate_determinism_b_ok) {
        gate_determinism_b_ok = (mb.xyz == mb2.xyz) && (mb.xyz == mb3.xyz) &&
                                (mb.count == mb2.count) && (mb.count == mb3.count) &&
                                (mb.key == mb2.key) && (mb.key == mb3.key);
    }
    all_ok = all_ok && gate_determinism_b_ok;
    std::printf("GATE determinism_method_b: %s (3 runs bit-identical)\n", gate_determinism_b_ok ? "PASS" : "FAIL");

    // Method A: NOT expected to be bit-identical (float atomicAdd order
    // varies with GPU thread scheduling) -- measure and report the honest
    // number, matched by voxel KEY across runs (the occupied SET is
    // deterministic even though the accumulation order is not -- see
    // kernels.cu's hash_insert_kernel comment).
    MethodAResult ma2 = run_method_a(n, d_xyz, d_keys, table, d_probe_len, d_overflow_count,
                                     d_outA_xyz, d_outA_count, d_outA_key, d_num_occupied);
    MethodAResult ma3 = run_method_a(n, d_xyz, d_keys, table, d_probe_len, d_overflow_count,
                                     d_outA_xyz, d_outA_count, d_outA_key, d_num_occupied);
    auto map_a2 = to_map(ma2.xyz, ma2.count, ma2.key, static_cast<int>(ma2.num_occupied));
    auto map_a3 = to_map(ma3.xyz, ma3.count, ma3.key, static_cast<int>(ma3.num_occupied));
    float max_delta_determinism_a = 0.0f;
    bool determinism_a_occupancy_stable = (ma.num_occupied == ma2.num_occupied) && (ma.num_occupied == ma3.num_occupied);
    for (unsigned int v = 0; v < ma.num_occupied; ++v) {
        for (const auto* m : { &map_a2, &map_a3 }) {
            const auto it = m->find(ma.key[v]);
            if (it == m->end()) { determinism_a_occupancy_stable = false; continue; }
            max_delta_determinism_a = std::max(max_delta_determinism_a, std::fabs(ma.xyz[v*3+0] - it->second.x));
            max_delta_determinism_a = std::max(max_delta_determinism_a, std::fabs(ma.xyz[v*3+1] - it->second.y));
            max_delta_determinism_a = std::max(max_delta_determinism_a, std::fabs(ma.xyz[v*3+2] - it->second.z));
        }
    }
    std::printf("[info] determinism_method_a: occupied-voxel SET stable across 3 runs = %s; "
               "max centroid delta across 3 runs = %.3e m (float atomicAdd order is scheduler-dependent "
               "by design -- THEORY.md \"Numerical considerations\")\n",
               determinism_a_occupancy_stable ? "yes" : "NO", static_cast<double>(max_delta_determinism_a));

    // ---- 9) occupancy_analytics [info] ---------------------------------------
    float bbmin[3] = { h_xyz[0], h_xyz[1], h_xyz[2] };
    float bbmax[3] = { h_xyz[0], h_xyz[1], h_xyz[2] };
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) {
            const float c = h_xyz[static_cast<size_t>(i) * 3 + static_cast<size_t>(a)];
            bbmin[a] = std::min(bbmin[a], c);
            bbmax[a] = std::max(bbmax[a], c);
        }
    }
    const double bbox_volume = static_cast<double>(bbmax[0] - bbmin[0]) *
                               static_cast<double>(bbmax[1] - bbmin[1]) *
                               static_cast<double>(bbmax[2] - bbmin[2]);
    const double leaf_vol = static_cast<double>(kVoxelLeafM) * kVoxelLeafM * kVoxelLeafM;
    const double naive_uniform_estimate = bbox_volume / leaf_vol;
    std::printf("[info] occupancy_analytics: %d occupied voxels vs a naive UNIFORM-VOLUME estimate of "
               "%.0f voxels (bbox %.1f x %.1f x %.1f m / L^3) -- ratio %.4f; real LiDAR returns lie on a "
               "thin 2-D SURFACE, not a filled 3-D volume, which is why occupancy is so much smaller "
               "than the volume estimate (THEORY.md \"The problem\")\n",
               mb.num_voxels, naive_uniform_estimate,
               static_cast<double>(bbmax[0]-bbmin[0]), static_cast<double>(bbmax[1]-bbmin[1]), static_cast<double>(bbmax[2]-bbmin[2]),
               naive_uniform_estimate > 0 ? (static_cast<double>(mb.num_voxels) / naive_uniform_estimate) : 0.0);

    // ---- 10) hash_stats [info] + probe_length_histogram.csv -----------------
    std::vector<int> probe_len(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(probe_len.data(), d_probe_len, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    // NOTE: d_probe_len currently holds ma3's probe lengths (the most recent
    // Method-A run, from the determinism study) -- every run inserts the
    // SAME n points into a table of the SAME capacity, so probe-length
    // STATISTICS are representative of any run; only the float centroid
    // VALUES differ between runs, not the structural probing behavior.
    long long sum_probe_normal = 0, sum_probe_adv = 0;
    int max_probe_normal = 0, max_probe_adv = 0;
    const int n_normal = hdr.n_normal;
    for (int i = 0; i < n; ++i) {
        const int p = probe_len[static_cast<size_t>(i)];
        if (i < n_normal) { sum_probe_normal += p; max_probe_normal = std::max(max_probe_normal, p); }
        else              { sum_probe_adv    += p; max_probe_adv    = std::max(max_probe_adv, p); }
    }
    const int n_adv = n - n_normal;
    const double mean_probe_normal = n_normal > 0 ? static_cast<double>(sum_probe_normal) / n_normal : 0.0;
    const double mean_probe_adv    = n_adv    > 0 ? static_cast<double>(sum_probe_adv)    / n_adv    : 0.0;
    const double load_factor = static_cast<double>(ma.num_occupied) / static_cast<double>(capacity);
    std::printf("[info] hash_stats: load_factor=%.4f (table-wide) | normal scan: mean_probe=%.3f max_probe=%d "
               "(n=%d) | adversarial (dense+sparse): mean_probe=%.3f max_probe=%d (n=%d)\n",
               load_factor, mean_probe_normal, max_probe_normal, n_normal,
               mean_probe_adv, max_probe_adv, n_adv);

    // ---- 11) downsample_quality [info] ---------------------------------------
    // For every ORIGINAL point, its own voxel's Method-B centroid IS the
    // "nearest downsampled representative" -- no separate spatial search
    // needed, since partition_invariant already established the one-point-
    // one-voxel mapping. RMS distance quantifies how much detail a 20 cm
    // voxel discards.
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto it = map_b.find(keys_gpu[static_cast<size_t>(i)]);
        if (it == map_b.end()) continue;  // cannot happen if partition_invariant held, guarded defensively
        const float dx = h_xyz[static_cast<size_t>(i)*3+0] - it->second.x;
        const float dy = h_xyz[static_cast<size_t>(i)*3+1] - it->second.y;
        const float dz = h_xyz[static_cast<size_t>(i)*3+2] - it->second.z;
        sum_sq += static_cast<double>(dx)*dx + static_cast<double>(dy)*dy + static_cast<double>(dz)*dz;
    }
    const double rms_m = std::sqrt(sum_sq / n);
    std::printf("[info] downsample_quality: nearest-neighbor RMS distance original->downsampled = %.4f m "
               "(L/2 = %.4f m back-of-envelope intuition)\n", rms_m, static_cast<double>(kVoxelLeafM) / 2.0);

    // ---- 12) timing [time] ----------------------------------------------------
    std::printf("[time] CPU compute_keys:         %.3f ms\n", cpu_keys_ms);
    std::printf("[time] CPU sort_based (Method B twin): %.3f ms\n", cpu_method_b_ms);
    std::printf("[time] CPU hashmap (Method A twin):    %.3f ms\n", cpu_method_a_ms);
    std::printf("[time] GPU Method A (insert+compact):  %.3f ms\n", static_cast<double>(ma.gpu_ms));
    std::printf("[time] GPU Method B (sort+reduce):     %.3f ms\n", static_cast<double>(mb.gpu_ms));

    // ---- 13) Artifacts ----------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);
    const std::string ppm_orig = out_dir + "/original_topview.ppm";
    const std::string ppm_down = out_dir + "/downsampled_topview.ppm";
    const std::string csv_hist = out_dir + "/probe_length_histogram.csv";
    const std::string csv_gates = out_dir + "/gates_metrics.csv";

    // View window: the scene's room is 16x16 m (+-8 m); pad a little so the
    // walls are not clipped at the image edge.
    const float half_extent_m = 9.0f;
    write_ppm_topview(ppm_orig, h_xyz.data(), n, 640, 640, half_extent_m);
    write_ppm_topview(ppm_down, mb.xyz.data(), mb.num_voxels, 640, 640, half_extent_m);

    {
        // Histogram buckets: 0,1,2,...,19, then "20+" — the vast majority of
        // inserts land in bucket 0 (empty slot on the first try) on a table
        // sized at kTargetLoadFactor=0.5; the tail is what hash_stats above
        // summarizes numerically, this file lets a learner plot the shape.
        const int kBuckets = 21;
        std::vector<long long> hist_normal(kBuckets, 0), hist_adv(kBuckets, 0);
        for (int i = 0; i < n; ++i) {
            int b = probe_len[static_cast<size_t>(i)];
            if (b >= kBuckets - 1) b = kBuckets - 1;
            if (i < n_normal) ++hist_normal[static_cast<size_t>(b)]; else ++hist_adv[static_cast<size_t>(b)];
        }
        std::ofstream f(csv_hist);
        f << "# probe_length_histogram.csv -- Method A atomicCAS linear-probe chain lengths, project 02.01\n";
        f << "# probe_length=20 bucket is \"20 or more\"\n";
        f << "probe_length,count_normal,count_adversarial,count_total\n";
        for (int b = 0; b < kBuckets; ++b) {
            f << b << ',' << hist_normal[static_cast<size_t>(b)] << ',' << hist_adv[static_cast<size_t>(b)]
              << ',' << (hist_normal[static_cast<size_t>(b)] + hist_adv[static_cast<size_t>(b)]) << '\n';
        }
    }
    {
        std::ofstream f(csv_gates);
        f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.01\n";
        f << "metric,value\n";
        f << "n_total," << n << '\n';
        f << "n_normal," << hdr.n_normal << '\n';
        f << "n_dense," << hdr.n_dense << '\n';
        f << "n_sparse," << hdr.n_sparse << '\n';
        f << "leaf_m," << kVoxelLeafM << '\n';
        f << "hash_capacity," << capacity << '\n';
        f << "num_occupied_method_a," << ma.num_occupied << '\n';
        f << "num_voxels_method_b," << mb.num_voxels << '\n';
        f << "overflow_count_method_a," << ma.overflow << '\n';
        f << "max_delta_method_a_vs_cpu_m," << max_delta_a << '\n';
        f << "max_delta_cross_method_m," << max_delta_cross << '\n';
        f << "max_delta_determinism_method_a_m," << max_delta_determinism_a << '\n';
        f << "load_factor," << load_factor << '\n';
        f << "mean_probe_normal," << mean_probe_normal << '\n';
        f << "max_probe_normal," << max_probe_normal << '\n';
        f << "mean_probe_adversarial," << mean_probe_adv << '\n';
        f << "max_probe_adversarial," << max_probe_adv << '\n';
        f << "occupancy_naive_uniform_estimate," << naive_uniform_estimate << '\n';
        f << "downsample_rms_m," << rms_m << '\n';
        f << "cpu_compute_keys_ms," << cpu_keys_ms << '\n';
        f << "cpu_method_b_twin_ms," << cpu_method_b_ms << '\n';
        f << "cpu_method_a_twin_ms," << cpu_method_a_ms << '\n';
        f << "gpu_method_a_ms," << ma.gpu_ms << '\n';
        f << "gpu_method_b_ms," << mb.gpu_ms << '\n';
    }
    std::printf("ARTIFACT: wrote demo/out/{original_topview.ppm, downsampled_topview.ppm, "
               "probe_length_histogram.csv, gates_metrics.csv}\n");

    // ---- 14) Cleanup ------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_xyz));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(table.keys)); CUDA_CHECK(cudaFree(table.sum_x)); CUDA_CHECK(cudaFree(table.sum_y));
    CUDA_CHECK(cudaFree(table.sum_z)); CUDA_CHECK(cudaFree(table.count));
    CUDA_CHECK(cudaFree(d_probe_len)); CUDA_CHECK(cudaFree(d_overflow_count)); CUDA_CHECK(cudaFree(d_num_occupied));
    CUDA_CHECK(cudaFree(d_outA_xyz)); CUDA_CHECK(cudaFree(d_outA_count)); CUDA_CHECK(cudaFree(d_outA_key));
    CUDA_CHECK(cudaFree(d_keys_scratch)); CUDA_CHECK(cudaFree(d_idx_scratch));
    CUDA_CHECK(cudaFree(d_is_start_scratch)); CUDA_CHECK(cudaFree(d_seg_start_out));
    CUDA_CHECK(cudaFree(d_outB_xyz)); CUDA_CHECK(cudaFree(d_outB_count)); CUDA_CHECK(cudaFree(d_outB_key));

    // ---- 15) Verdict --------------------------------------------------------------
    if (all_ok) {
        std::printf("RESULT: PASS (VERIFY(keys/method_b/method_a) + all 4 gates passed: "
                   "cross_method_agreement, partition_invariant, centroid_containment, determinism_method_b)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (a VERIFY or GATE above did not pass -- see stderr for details)\n");
        return EXIT_FAILURE;
    }
}
