// ===========================================================================
// main.cu — entry point for project 02.02 (ROI crop, passthrough,
//           organized<->unorganized conversion kernels)
//
// Role in the project
// -------------------
// Orchestration only: load the synthetic sample, run every kernel pipeline
// (organized<->unorganized, the four predicate compactions, chained vs
// fused, the collision test), compare each GPU result against its
// independent CPU twin, print a VERIFY/GATE verdict per check, write the
// visual artifacts, and report PASS/FAIL. The real teaching content — THE
// SCAN CHAPTER, the predicates, the encoded-atomicMin scatter — lives in
// kernels.cu; reference_cpu.cpp carries the independent oracles.
//
// Output contract (load-bearing — see demo/run_demo.ps1 / .sh)
// -------------------------------------------------------------
// Stable lines ("[demo]", "PROBLEM:", "DATA:", "VERIFY(...):", "GATE ...:",
// "ARTIFACT:", "RESULT:") contain no timings/device names and are diffed
// against demo/expected_output.txt. "[info]"/"[time]" lines are NOT diffed.
//
// Read this after: kernels.cuh.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ===========================================================================
// Sample loading — mirrors scripts/make_synthetic.py's binary layout
// EXACTLY (see that file's module docstring for the authoritative field
// table). Every field is read as an explicit little-endian primitive, never
// a raw struct fread, so the layout is compiler-independent (02.01's
// loader precedent, cited).
// ===========================================================================
struct SampleData {
    std::vector<float> organized_xyz;   // [kOrganizedCells*3], NaN = invalid cell
    std::vector<float> edge_xyz;        // [n_edge*3]
    std::vector<int>   ghost_cell;      // [n_ghost] organized-grid cell index
    std::vector<float> ghost_offset;    // [n_ghost] range offset, meters
    int n_edge = 0;
    int n_ghost = 0;
    int n_organized_valid_header = 0;   // Python's OWN independent tally (cross-checked below)
};

static bool load_sample(const std::string& path, SampleData& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "error: could not open sample file '%s'\n", path.c_str());
        return false;
    }

    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "RCPOU001", 8) != 0) {
        std::fprintf(stderr, "error: '%s' does not start with the expected RCPOU001 magic\n", path.c_str());
        return false;
    }

    int32_t num_beams = 0, azimuth_bins = 0, n_edge = 0, n_ghost = 0, n_valid_hdr = 0, reserved = 0;
    f.read(reinterpret_cast<char*>(&num_beams), 4);
    f.read(reinterpret_cast<char*>(&azimuth_bins), 4);
    f.read(reinterpret_cast<char*>(&n_edge), 4);
    f.read(reinterpret_cast<char*>(&n_ghost), 4);
    f.read(reinterpret_cast<char*>(&n_valid_hdr), 4);
    f.read(reinterpret_cast<char*>(&reserved), 4);
    if (!f) {
        std::fprintf(stderr, "error: '%s' has a truncated header\n", path.c_str());
        return false;
    }
    if (num_beams != kNumBeams || azimuth_bins != kAzimuthBins) {
        std::fprintf(stderr,
            "error: sample grid shape %dx%d does not match kernels.cuh's kNumBeams=%d "
            "kAzimuthBins=%d — regenerate with scripts/make_synthetic.py\n",
            num_beams, azimuth_bins, kNumBeams, kAzimuthBins);
        return false;
    }

    out.organized_xyz.resize(static_cast<size_t>(kOrganizedCells) * 3);
    f.read(reinterpret_cast<char*>(out.organized_xyz.data()),
          static_cast<std::streamsize>(out.organized_xyz.size() * sizeof(float)));

    out.n_edge = n_edge;
    out.edge_xyz.resize(static_cast<size_t>(n_edge) * 3);
    f.read(reinterpret_cast<char*>(out.edge_xyz.data()),
          static_cast<std::streamsize>(out.edge_xyz.size() * sizeof(float)));

    out.n_ghost = n_ghost;
    out.ghost_cell.resize(static_cast<size_t>(n_ghost));
    out.ghost_offset.resize(static_cast<size_t>(n_ghost));
    for (int i = 0; i < n_ghost; ++i) {
        int32_t cell = 0;
        float offset = 0.0f;
        f.read(reinterpret_cast<char*>(&cell), 4);
        f.read(reinterpret_cast<char*>(&offset), 4);
        out.ghost_cell[static_cast<size_t>(i)] = cell;
        out.ghost_offset[static_cast<size_t>(i)] = offset;
    }
    out.n_organized_valid_header = n_valid_hdr;

    if (!f) {
        std::fprintf(stderr, "error: '%s' is truncated (expected full organized grid + edge + ghost sections)\n",
                    path.c_str());
        return false;
    }
    return true;
}

// ===========================================================================
// Small artifact writers — hand-rolled binary PPM (P6, color) / PGM (P5,
// gray) writers, no image library (CLAUDE.md §5 "no black boxes"; the
// two-layer top-view precedent is 02.01's write_ppm_topview, extended here
// with a dim-gray BACKDROP layer plus a bright HIGHLIGHT layer so a crop's
// surviving subset is visible against the full cloud in one image).
// ===========================================================================

static void write_ppm_topview_highlight(const std::string& path,
                                        const float* bg_xyz, int n_bg,
                                        const float* hi_xyz, int n_hi,
                                        int width, int height, float half_extent_m,
                                        unsigned char hr, unsigned char hg, unsigned char hb)
{
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3, 0);  // black background
    const float scale = static_cast<float>(width) / (2.0f * half_extent_m);          // px per meter

    auto plot = [&](float x, float y, unsigned char r, unsigned char g, unsigned char b) {
        const int px = static_cast<int>((x + half_extent_m) * scale);
        const int py = static_cast<int>((half_extent_m - y) * scale);   // flip y: world +y is "up"
        if (px < 0 || px >= width || py < 0 || py >= height) return;
        const size_t idx = (static_cast<size_t>(py) * width + px) * 3;
        pixels[idx + 0] = r; pixels[idx + 1] = g; pixels[idx + 2] = b;
    };

    // Backdrop first (dim gray), highlight second (bright, so it draws on
    // top when a highlighted point shares a pixel with a backdrop point —
    // exactly what "which points survived the crop" wants to show clearly).
    for (int i = 0; i < n_bg; ++i) plot(bg_xyz[i * 3 + 0], bg_xyz[i * 3 + 1], 90, 90, 90);
    for (int i = 0; i < n_hi; ++i) plot(hi_xyz[i * 3 + 0], hi_xyz[i * 3 + 1], hr, hg, hb);

    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << ' ' << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

// write_pgm_occupancy — the organized grid's validity mask as a 1024x16
// grayscale image (white=255=valid cell, black=0=invalid/NaN cell). Viewed
// at native resolution the image is a thin strip; demo/README.md explains
// what a learner should see: horizontal dark bands where whole azimuth
// ranges see open sky over the walls (no ceiling, by scene design), plus a
// scatter of single-cell dropouts (the 5% absorption/glare model).
static void write_pgm_occupancy(const std::string& path, const float* organized_xyz)
{
    std::vector<unsigned char> pixels(static_cast<size_t>(kOrganizedCells), 0);
    for (int c = 0; c < kOrganizedCells; ++c) {
        pixels[static_cast<size_t>(c)] = is_invalid_point(organized_xyz[c * 3 + 0]) ? 0 : 255;
    }
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << kAzimuthBins << ' ' << kNumBeams << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

// ===========================================================================
// run_compact_pair — shared GPU-vs-CPU orchestration for the four named
// predicate compactions (passthrough/box/frustum/fused): run both,
// time both, and precompute every comparison main() reports a verdict on.
// A small helper rather than four near-identical inline blocks in main().
// ===========================================================================
struct CompactPairResult {
    int gpu_count = 0, cpu_count = 0;
    std::vector<float> gpu_xyz, cpu_xyz;
    std::vector<int>   gpu_idx, cpu_idx;
    float  gpu_ms = 0.0f;
    double cpu_ms = 0.0;
    bool count_match = false;             // gpu_count == cpu_count
    bool idx_match = false;               // count_match AND every orig index agrees, position by position
    bool predicate_self_check = false;    // every GPU-kept point independently satisfies the predicate
};

using GpuCompactFn = int (*)(int, const float*, float*, int*);
using CpuCompactFn = int (*)(int, const float*, float*, int*);
using PredicateFn  = bool (*)(float, float, float);

static CompactPairResult run_compact_pair(int n, const float* d_xyz, const float* h_xyz,
                                          GpuCompactFn gpu_fn, CpuCompactFn cpu_fn,
                                          PredicateFn predicate_check)
{
    CompactPairResult r;

    float* d_out_xyz = nullptr;
    int* d_out_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_idx, static_cast<size_t>(n) * sizeof(int)));

    GpuTimer gt; gt.begin();
    r.gpu_count = gpu_fn(n, d_xyz, d_out_xyz, d_out_idx);
    r.gpu_ms = gt.end_ms();

    r.gpu_xyz.resize(static_cast<size_t>(r.gpu_count) * 3);
    r.gpu_idx.resize(static_cast<size_t>(r.gpu_count));
    CUDA_CHECK(cudaMemcpy(r.gpu_xyz.data(), d_out_xyz, r.gpu_xyz.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.gpu_idx.data(), d_out_idx, r.gpu_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out_xyz));
    CUDA_CHECK(cudaFree(d_out_idx));

    std::vector<float> tmp_xyz(static_cast<size_t>(n) * 3);
    std::vector<int> tmp_idx(static_cast<size_t>(n));
    CpuTimer ct; ct.begin();
    r.cpu_count = cpu_fn(n, h_xyz, tmp_xyz.data(), tmp_idx.data());
    r.cpu_ms = ct.end_ms();
    tmp_xyz.resize(static_cast<size_t>(r.cpu_count) * 3);
    tmp_idx.resize(static_cast<size_t>(r.cpu_count));
    r.cpu_xyz = std::move(tmp_xyz);
    r.cpu_idx = std::move(tmp_idx);

    r.count_match = (r.gpu_count == r.cpu_count);
    r.idx_match = r.count_match;
    if (r.idx_match) {
        for (int i = 0; i < r.gpu_count; ++i) {
            if (r.gpu_idx[static_cast<size_t>(i)] != r.cpu_idx[static_cast<size_t>(i)]) { r.idx_match = false; break; }
        }
    }

    r.predicate_self_check = true;
    for (int i = 0; i < r.gpu_count; ++i) {
        const float x = r.gpu_xyz[static_cast<size_t>(i) * 3 + 0];
        const float y = r.gpu_xyz[static_cast<size_t>(i) * 3 + 1];
        const float z = r.gpu_xyz[static_cast<size_t>(i) * 3 + 2];
        if (!predicate_check(x, y, z)) { r.predicate_self_check = false; break; }
    }

    return r;
}

// Predicate-check wrappers matching PredicateFn's (x,y,z) signature — the
// shared kernels.cuh formulas have varying arities (is_passthrough takes
// only z), so each gets a one-line adapter here.
static bool check_passthrough(float x, float y, float z) { (void)x; (void)y; return is_passthrough(z); }
static bool check_box(float x, float y, float z)         { return is_in_box(x, y, z); }
static bool check_frustum(float x, float y, float z)     { return is_in_frustum(x, y, z); }
static bool check_fused(float x, float y, float z)       { return is_fused(x, y, z); }

// strictly_increasing — the structural half of GATE order_preservation:
// independent of any CPU comparison, a compaction that preserves order
// must produce a STRICTLY increasing sequence of original indices (no
// ties are possible — every original index appears at most once).
static bool strictly_increasing(const std::vector<int>& v)
{
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] <= v[i - 1]) return false;
    }
    return true;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    bool all_ok = true;   // ANDed with every VERIFY/GATE below; drives the final RESULT: line

    std::printf("[demo] ROI crop / passthrough / organized<->unorganized conversion kernels: "
               "prefix-scan stream compaction (project 02.02)\n");
    print_device_info();

    // ---- 0) Load the synthetic sample --------------------------------------
    const std::string sample_path = find_data_file("", argv[0], "roi_scan.bin");
    if (sample_path.empty()) {
        std::fprintf(stderr, "error: could not locate data/sample/roi_scan.bin -- run "
                             "scripts/make_synthetic.py\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return EXIT_FAILURE;
    }
    SampleData sample;
    if (!load_sample(sample_path, sample)) {
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return EXIT_FAILURE;
    }

    std::printf("PROBLEM: organized grid %d rings x %d azimuth = %d cells; N_edge=%d, N_ghost=%d\n",
               kNumBeams, kAzimuthBins, kOrganizedCells, sample.n_edge, sample.n_ghost);
    std::printf("DATA: data/sample/roi_scan.bin [synthetic, seed 42, xorshift32, single-revolution "
               "16-beam organized scan + edge cohort + ghost table]\n");

    // ---- 1) Organized -> unorganized: GPU + CPU, VERIFY bit-exact ----------
    float* d_organized = nullptr;
    CUDA_CHECK(cudaMalloc(&d_organized, sample.organized_xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_organized, sample.organized_xyz.data(),
                          sample.organized_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));

    float* d_unorg_xyz = nullptr;
    int* d_unorg_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unorg_xyz, static_cast<size_t>(kOrganizedCells) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_unorg_idx, static_cast<size_t>(kOrganizedCells) * sizeof(int)));

    GpuTimer t_o2u; t_o2u.begin();
    const int n_valid_gpu = launch_valid_compact(kOrganizedCells, d_organized, d_unorg_xyz, d_unorg_idx);
    const float o2u_gpu_ms = t_o2u.end_ms();

    std::vector<float> h_unorg_gpu_xyz(static_cast<size_t>(n_valid_gpu) * 3);
    std::vector<int>   h_unorg_gpu_idx(static_cast<size_t>(n_valid_gpu));
    CUDA_CHECK(cudaMemcpy(h_unorg_gpu_xyz.data(), d_unorg_xyz, h_unorg_gpu_xyz.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_unorg_gpu_idx.data(), d_unorg_idx, h_unorg_gpu_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> h_unorg_cpu_xyz(static_cast<size_t>(kOrganizedCells) * 3);
    std::vector<int>   h_unorg_cpu_idx(static_cast<size_t>(kOrganizedCells));
    CpuTimer t_o2u_cpu; t_o2u_cpu.begin();
    const int n_valid_cpu = valid_compact_cpu(kOrganizedCells, sample.organized_xyz.data(),
                                              h_unorg_cpu_xyz.data(), h_unorg_cpu_idx.data());
    const double o2u_cpu_ms = t_o2u_cpu.end_ms();
    h_unorg_cpu_xyz.resize(static_cast<size_t>(n_valid_cpu) * 3);
    h_unorg_cpu_idx.resize(static_cast<size_t>(n_valid_cpu));

    bool o2u_ok = (n_valid_gpu == n_valid_cpu) && (n_valid_gpu == sample.n_organized_valid_header);
    if (o2u_ok) {
        for (int i = 0; i < n_valid_gpu && o2u_ok; ++i) {
            if (h_unorg_gpu_idx[static_cast<size_t>(i)] != h_unorg_cpu_idx[static_cast<size_t>(i)]) { o2u_ok = false; break; }
            for (int k = 0; k < 3; ++k) {
                if (h_unorg_gpu_xyz[static_cast<size_t>(i) * 3 + k] != h_unorg_cpu_xyz[static_cast<size_t>(i) * 3 + k]) { o2u_ok = false; break; }
            }
        }
    }
    all_ok = all_ok && o2u_ok;
    std::printf("VERIFY(organized_to_unorganized): %s (GPU vs CPU vs Python-generator tally bit-exact, "
               "N_valid=%d/%d cells)\n", o2u_ok ? "PASS" : "FAIL", n_valid_gpu, kOrganizedCells);

    // ---- 2) Build the predicate_test_cloud: CPU-verified organized-valid --
    //         points + the edge cohort (straddling every predicate boundary).
    const int n_predicate = n_valid_cpu + sample.n_edge;
    std::vector<float> h_predicate_xyz(static_cast<size_t>(n_predicate) * 3);
    std::copy(h_unorg_cpu_xyz.begin(), h_unorg_cpu_xyz.end(), h_predicate_xyz.begin());
    std::copy(sample.edge_xyz.begin(), sample.edge_xyz.end(),
             h_predicate_xyz.begin() + static_cast<long>(h_unorg_cpu_xyz.size()));

    float* d_predicate_xyz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_predicate_xyz, h_predicate_xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_predicate_xyz, h_predicate_xyz.data(), h_predicate_xyz.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- 3) Four named compactions: GPU vs CPU -----------------------------
    const CompactPairResult r_pt      = run_compact_pair(n_predicate, d_predicate_xyz, h_predicate_xyz.data(),
                                                          launch_passthrough_compact, passthrough_compact_cpu, check_passthrough);
    const CompactPairResult r_box     = run_compact_pair(n_predicate, d_predicate_xyz, h_predicate_xyz.data(),
                                                          launch_box_compact, box_compact_cpu, check_box);
    const CompactPairResult r_frustum = run_compact_pair(n_predicate, d_predicate_xyz, h_predicate_xyz.data(),
                                                          launch_frustum_compact, frustum_compact_cpu, check_frustum);
    const CompactPairResult r_fused   = run_compact_pair(n_predicate, d_predicate_xyz, h_predicate_xyz.data(),
                                                          launch_fused_compact, fused_compact_cpu, check_fused);

    const bool predicate_correctness_ok =
        r_pt.count_match && r_pt.idx_match && r_pt.predicate_self_check &&
        r_box.count_match && r_box.idx_match && r_box.predicate_self_check &&
        r_frustum.count_match && r_frustum.idx_match && r_frustum.predicate_self_check &&
        r_fused.count_match && r_fused.idx_match && r_fused.predicate_self_check;
    all_ok = all_ok && predicate_correctness_ok;
    std::printf("VERIFY(predicate_correctness): %s (passthrough=%d, box=%d, frustum=%d, fused=%d kept of N=%d; "
               "GPU counts+order match CPU, every kept point independently satisfies its predicate)\n",
               predicate_correctness_ok ? "PASS" : "FAIL",
               r_pt.gpu_count, r_box.gpu_count, r_frustum.gpu_count, r_fused.gpu_count, n_predicate);

    const bool order_preservation_ok =
        strictly_increasing(r_pt.gpu_idx) && strictly_increasing(r_box.gpu_idx) &&
        strictly_increasing(r_frustum.gpu_idx) && strictly_increasing(r_fused.gpu_idx);
    all_ok = all_ok && order_preservation_ok;
    std::printf("GATE order_preservation: %s (compacted original-index sequence strictly increasing, all 4 filters)\n",
               order_preservation_ok ? "PASS" : "FAIL");

    int edge_inside_frustum = 0, edge_outside_frustum = 0;
    for (int i = n_valid_cpu; i < n_predicate; ++i) {
        const float x = h_predicate_xyz[static_cast<size_t>(i) * 3 + 0];
        const float y = h_predicate_xyz[static_cast<size_t>(i) * 3 + 1];
        const float z = h_predicate_xyz[static_cast<size_t>(i) * 3 + 2];
        if (is_in_frustum(x, y, z)) ++edge_inside_frustum; else ++edge_outside_frustum;
    }
    const bool frustum_geometry_ok = r_frustum.count_match && r_frustum.idx_match && r_frustum.predicate_self_check;
    all_ok = all_ok && frustum_geometry_ok;
    std::printf("GATE frustum_geometry: %s (GPU 5-plane frustum crop == CPU brute-force point-in-frustum, "
               "set+order exact; edge cohort: %d inside / %d outside)\n",
               frustum_geometry_ok ? "PASS" : "FAIL", edge_inside_frustum, edge_outside_frustum);

    // ---- 4) Chained (3-pass) vs fused (1-pass): bit-identical? -------------
    float* d_stage1_xyz = nullptr; int* d_stage1_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stage1_xyz, static_cast<size_t>(n_predicate) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stage1_idx, static_cast<size_t>(n_predicate) * sizeof(int)));

    GpuTimer t_chained; t_chained.begin();
    const int n1 = launch_passthrough_compact(n_predicate, d_predicate_xyz, d_stage1_xyz, d_stage1_idx);

    float* d_stage2_xyz = nullptr; int* d_stage2_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stage2_xyz, static_cast<size_t>(n1) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stage2_idx, static_cast<size_t>(n1) * sizeof(int)));
    const int n2 = launch_box_compact(n1, d_stage1_xyz, d_stage2_xyz, d_stage2_idx);

    float* d_stage3_xyz = nullptr; int* d_stage3_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stage3_xyz, static_cast<size_t>(n2) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stage3_idx, static_cast<size_t>(n2) * sizeof(int)));
    const int n3 = launch_frustum_compact(n2, d_stage2_xyz, d_stage3_xyz, d_stage3_idx);
    const float chained_gpu_ms = t_chained.end_ms();

    std::vector<float> h_stage3_xyz(static_cast<size_t>(n3) * 3);
    std::vector<int> h_stage1_idx(static_cast<size_t>(n1)), h_stage2_idx(static_cast<size_t>(n2)), h_stage3_idx(static_cast<size_t>(n3));
    CUDA_CHECK(cudaMemcpy(h_stage3_xyz.data(), d_stage3_xyz, h_stage3_xyz.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stage1_idx.data(), d_stage1_idx, h_stage1_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stage2_idx.data(), d_stage2_idx, h_stage2_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stage3_idx.data(), d_stage3_idx, h_stage3_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Compose the chained pipeline's per-stage-relative indices back into
    // ORIGINAL predicate_test_cloud indices: stage3's index refers into
    // stage2's array, stage2's into stage1's, stage1's into the original.
    std::vector<int> chained_orig_idx(static_cast<size_t>(n3));
    for (int k = 0; k < n3; ++k) {
        const int i2 = h_stage3_idx[static_cast<size_t>(k)];
        const int i1 = h_stage2_idx[static_cast<size_t>(i2)];
        chained_orig_idx[static_cast<size_t>(k)] = h_stage1_idx[static_cast<size_t>(i1)];
    }

    bool fused_vs_chained_ok = (n3 == r_fused.gpu_count);
    for (int k = 0; k < n3 && fused_vs_chained_ok; ++k) {
        if (chained_orig_idx[static_cast<size_t>(k)] != r_fused.gpu_idx[static_cast<size_t>(k)]) { fused_vs_chained_ok = false; break; }
        for (int c = 0; c < 3; ++c) {
            if (h_stage3_xyz[static_cast<size_t>(k) * 3 + c] != r_fused.gpu_xyz[static_cast<size_t>(k) * 3 + c]) { fused_vs_chained_ok = false; break; }
        }
    }
    all_ok = all_ok && fused_vs_chained_ok;
    std::printf("GATE fused_vs_chained: %s (fused single-pass == chained 3-pass, bit-identical set+order+values; "
               "N=%d -> %d -> %d -> %d)\n", fused_vs_chained_ok ? "PASS" : "FAIL", n_predicate, n1, n2, n3);

    // Analytical byte-traffic model (documented, NOT a profiler measurement):
    // each compaction pass reads its input xyz array once in the predicate
    // kernel and once again in the scatter kernel — the dominant term next
    // to the much smaller int flags/scan arrays, which this estimate omits
    // for clarity (see the printed formula for exactly what is counted).
    const long long fused_bytes   = static_cast<long long>(n_predicate) * 3 * 4 * 2;
    const long long chained_bytes = (static_cast<long long>(n_predicate) + n1 + n2) * 3 * 4 * 2;
    std::printf("[info] fused_vs_chained_bytes (analytical: xyz read once by predicate + once by scatter, "
               "per pass; flags/scan arrays omitted): fused=%lld B (1 pass, N=%d), "
               "chained=%lld B (3 passes, N=%d,%d,%d), chained/fused=%.2fx\n",
               fused_bytes, n_predicate, chained_bytes, n_predicate, n1, n2,
               static_cast<double>(chained_bytes) / static_cast<double>(fused_bytes));
    std::printf("[time] chained (3-pass) GPU: %.3f ms | fused (1-pass) GPU: %.3f ms\n",
               static_cast<double>(chained_gpu_ms), static_cast<double>(r_fused.gpu_ms));

    // ---- 5) Roundtrip: organized -> unorganized -> organized ---------------
    float* d_roundtrip_organized = nullptr;
    int* d_roundtrip_winner = nullptr;
    CUDA_CHECK(cudaMalloc(&d_roundtrip_organized, static_cast<size_t>(kOrganizedCells) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_roundtrip_winner, static_cast<size_t>(kOrganizedCells) * sizeof(int)));
    const OrganizedScatterResult rt = launch_scatter_to_organized(n_valid_gpu, d_unorg_xyz,
                                                                   d_roundtrip_organized, d_roundtrip_winner);
    std::vector<float> h_roundtrip_organized(static_cast<size_t>(kOrganizedCells) * 3);
    CUDA_CHECK(cudaMemcpy(h_roundtrip_organized.data(), d_roundtrip_organized,
                          h_roundtrip_organized.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool roundtrip_ok = (rt.collisions == 0) && (rt.occupied == n_valid_gpu);
    for (int c = 0; c < kOrganizedCells && roundtrip_ok; ++c) {
        const bool orig_invalid = is_invalid_point(sample.organized_xyz[static_cast<size_t>(c) * 3 + 0]);
        const bool rt_invalid = is_invalid_point(h_roundtrip_organized[static_cast<size_t>(c) * 3 + 0]);
        if (orig_invalid != rt_invalid) { roundtrip_ok = false; break; }
        if (!orig_invalid) {
            for (int k = 0; k < 3; ++k) {
                if (h_roundtrip_organized[static_cast<size_t>(c) * 3 + k] != sample.organized_xyz[static_cast<size_t>(c) * 3 + k]) {
                    roundtrip_ok = false; break;
                }
            }
        }
    }
    all_ok = all_ok && roundtrip_ok;
    std::printf("GATE roundtrip: %s (organized -> unorganized -> organized identity on all %d valid cells, "
               "0 collisions expected and %d observed, %d empty cells stay empty)\n",
               roundtrip_ok ? "PASS" : "FAIL", n_valid_gpu, rt.collisions, kOrganizedCells - n_valid_gpu);

    // ---- 6) Collision test: organized-valid points + ghost second echoes --
    const int n_collision = n_valid_cpu + sample.n_ghost;
    std::vector<float> h_collision_xyz(static_cast<size_t>(n_collision) * 3);
    std::copy(h_unorg_cpu_xyz.begin(), h_unorg_cpu_xyz.end(), h_collision_xyz.begin());
    for (int g = 0; g < sample.n_ghost; ++g) {
        const int cell = sample.ghost_cell[static_cast<size_t>(g)];
        const float x = sample.organized_xyz[static_cast<size_t>(cell) * 3 + 0];
        const float y = sample.organized_xyz[static_cast<size_t>(cell) * 3 + 1];
        const float z = sample.organized_xyz[static_cast<size_t>(cell) * 3 + 2];
        const float range = std::sqrt(x * x + y * y + z * z);
        const float new_range = range + sample.ghost_offset[static_cast<size_t>(g)];
        const float scale = new_range / range;   // same direction, scaled range -> the "second echo"
        const size_t base = static_cast<size_t>(n_valid_cpu + g) * 3;
        h_collision_xyz[base + 0] = x * scale;
        h_collision_xyz[base + 1] = y * scale;
        h_collision_xyz[base + 2] = z * scale;
    }

    float* d_collision_xyz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_collision_xyz, h_collision_xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_collision_xyz, h_collision_xyz.data(), h_collision_xyz.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    float* d_collision_organized = nullptr;
    int* d_collision_winner = nullptr;
    CUDA_CHECK(cudaMalloc(&d_collision_organized, static_cast<size_t>(kOrganizedCells) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_collision_winner, static_cast<size_t>(kOrganizedCells) * sizeof(int)));

    GpuTimer t_coll; t_coll.begin();
    const OrganizedScatterResult coll_gpu = launch_scatter_to_organized(n_collision, d_collision_xyz,
                                                                        d_collision_organized, d_collision_winner);
    const float coll_gpu_ms = t_coll.end_ms();
    std::vector<int> h_collision_winner_gpu(static_cast<size_t>(kOrganizedCells));
    CUDA_CHECK(cudaMemcpy(h_collision_winner_gpu.data(), d_collision_winner,
                          h_collision_winner_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> h_collision_organized_cpu(static_cast<size_t>(kOrganizedCells) * 3);
    std::vector<int> h_collision_winner_cpu(static_cast<size_t>(kOrganizedCells));
    CpuTimer t_coll_cpu; t_coll_cpu.begin();
    const OrganizedScatterCpuResult coll_cpu = scatter_to_organized_cpu(n_collision, h_collision_xyz.data(),
                                                                        h_collision_organized_cpu.data(),
                                                                        h_collision_winner_cpu.data());
    const double coll_cpu_ms = t_coll_cpu.end_ms();

    bool collision_ok = (coll_gpu.occupied + coll_gpu.collisions == n_collision) &&
                        (coll_cpu.occupied + coll_cpu.collisions == n_collision) &&
                        (coll_gpu.occupied == n_valid_cpu) && (coll_cpu.occupied == n_valid_cpu) &&
                        (coll_gpu.collisions == coll_cpu.collisions);
    for (int c = 0; c < kOrganizedCells && collision_ok; ++c) {
        if (h_collision_winner_gpu[static_cast<size_t>(c)] != h_collision_winner_cpu[static_cast<size_t>(c)]) {
            collision_ok = false;
        }
    }
    all_ok = all_ok && collision_ok;
    std::printf("GATE collision_accounting: %s (valid_in=%d == occupied(%d)+collisions(%d) [GPU], "
               "== occupied(%d)+collisions(%d) [CPU]; GPU winner-per-cell bit-exact vs CPU)\n",
               collision_ok ? "PASS" : "FAIL", n_collision,
               coll_gpu.occupied, coll_gpu.collisions, coll_cpu.occupied, coll_cpu.collisions);

    // ---- 7) VERIFY(scan_bitexact): hand-rolled GPU / Thrust GPU / CPU -----
    //         on two representative flags arrays.
    int* d_flagsA = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flagsA, static_cast<size_t>(kOrganizedCells) * sizeof(int)));
    {
        const int blocks = blocks_for(kOrganizedCells, kThreadsPerBlock);
        valid_predicate_kernel<<<blocks, kThreadsPerBlock>>>(kOrganizedCells, d_organized, d_flagsA);
        CUDA_CHECK_LAST_ERROR("valid_predicate_kernel (scan_bitexact A) launch");
    }
    int* d_flagsB = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flagsB, static_cast<size_t>(n_predicate) * sizeof(int)));
    {
        const int blocks = blocks_for(n_predicate, kThreadsPerBlock);
        fused_predicate_kernel<<<blocks, kThreadsPerBlock>>>(n_predicate, d_predicate_xyz, d_flagsB);
        CUDA_CHECK_LAST_ERROR("fused_predicate_kernel (scan_bitexact B) launch");
    }

    auto scan_triple_agrees = [](int n, const int* d_flags) -> bool {
        std::vector<int> h_flags(static_cast<size_t>(n));
        CUDA_CHECK(cudaMemcpy(h_flags.data(), d_flags, h_flags.size() * sizeof(int), cudaMemcpyDeviceToHost));

        int* d_hand = nullptr; int* d_thr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hand, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_thr, static_cast<size_t>(n) * sizeof(int)));
        launch_scan_blelloch(n, d_flags, d_hand);
        launch_scan_thrust(n, d_flags, d_thr);

        std::vector<int> h_hand(static_cast<size_t>(n)), h_thr(static_cast<size_t>(n)), h_cpu(static_cast<size_t>(n));
        CUDA_CHECK(cudaMemcpy(h_hand.data(), d_hand, h_hand.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_thr.data(), d_thr, h_thr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        scan_exclusive_cpu(n, h_flags.data(), h_cpu.data());

        CUDA_CHECK(cudaFree(d_hand));
        CUDA_CHECK(cudaFree(d_thr));
        return (h_hand == h_thr) && (h_hand == h_cpu);   // std::vector<int>::operator== : element-wise, exact
    };

    const bool scanA_ok = scan_triple_agrees(kOrganizedCells, d_flagsA);
    const bool scanB_ok = scan_triple_agrees(n_predicate, d_flagsB);
    const bool scan_bitexact_ok = scanA_ok && scanB_ok;
    all_ok = all_ok && scan_bitexact_ok;
    std::printf("VERIFY(scan_bitexact): %s (hand-rolled Blelloch GPU scan == Thrust GPU scan == CPU serial "
               "scan, integer-exact, on flag arrays of size %d and %d)\n",
               scan_bitexact_ok ? "PASS" : "FAIL", kOrganizedCells, n_predicate);
    CUDA_CHECK(cudaFree(d_flagsA));
    CUDA_CHECK(cudaFree(d_flagsB));

    // ---- 8) [info] scan_scaling: hand-rolled vs Thrust across 3 sizes ------
    const int scan_sizes[3] = { 2048, 8192, 32768 };
    for (int s : scan_sizes) {
        std::vector<int> h_flags(static_cast<size_t>(s));
        for (int i = 0; i < s; ++i) h_flags[static_cast<size_t>(i)] = (i % 3 == 0) ? 1 : 0;   // reproducible, not tied to real data
        int* d_flags = nullptr; int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(s) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(s) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_flags, h_flags.data(), h_flags.size() * sizeof(int), cudaMemcpyHostToDevice));

        GpuTimer t_hand; t_hand.begin();
        launch_scan_blelloch(s, d_flags, d_out);
        const float hand_ms = t_hand.end_ms();

        GpuTimer t_thr; t_thr.begin();
        launch_scan_thrust(s, d_flags, d_out);
        const float thr_ms = t_thr.end_ms();

        std::printf("[info] scan_scaling: n=%6d  hand-rolled=%.4f ms  thrust=%.4f ms\n",
                   s, static_cast<double>(hand_ms), static_cast<double>(thr_ms));
        CUDA_CHECK(cudaFree(d_flags));
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---- 9) Timings ----------------------------------------------------------
    std::printf("[time] organized_to_unorganized: GPU=%.3f ms CPU=%.3f ms\n",
               static_cast<double>(o2u_gpu_ms), o2u_cpu_ms);
    std::printf("[time] passthrough:  GPU=%.3f ms CPU=%.3f ms\n", static_cast<double>(r_pt.gpu_ms), r_pt.cpu_ms);
    std::printf("[time] box:          GPU=%.3f ms CPU=%.3f ms\n", static_cast<double>(r_box.gpu_ms), r_box.cpu_ms);
    std::printf("[time] frustum:      GPU=%.3f ms CPU=%.3f ms\n", static_cast<double>(r_frustum.gpu_ms), r_frustum.cpu_ms);
    std::printf("[time] fused:        GPU=%.3f ms CPU=%.3f ms\n", static_cast<double>(r_fused.gpu_ms), r_fused.cpu_ms);
    std::printf("[time] collision (unorganized->organized): GPU=%.3f ms CPU=%.3f ms\n",
               static_cast<double>(coll_gpu_ms), coll_cpu_ms);

    // ---- 10) Artifacts ---------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);
    const float half_extent_m = 9.0f;   // room half-extent (8 m) + a small margin, matching 02.01's precedent
    write_ppm_topview_highlight(out_dir + "/full_topview.ppm", nullptr, 0,
                                h_predicate_xyz.data(), n_predicate, 480, 480, half_extent_m, 220, 220, 220);
    write_ppm_topview_highlight(out_dir + "/box_topview.ppm", h_predicate_xyz.data(), n_predicate,
                                r_box.gpu_xyz.data(), r_box.gpu_count, 480, 480, half_extent_m, 0, 255, 0);
    write_ppm_topview_highlight(out_dir + "/frustum_topview.ppm", h_predicate_xyz.data(), n_predicate,
                                r_frustum.gpu_xyz.data(), r_frustum.gpu_count, 480, 480, half_extent_m, 0, 255, 255);
    write_pgm_occupancy(out_dir + "/organized_occupancy_before.pgm", sample.organized_xyz.data());
    write_pgm_occupancy(out_dir + "/organized_occupancy_after.pgm", h_roundtrip_organized.data());

    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.02\n";
        f << "metric,value\n";
        f << "n_organized_cells," << kOrganizedCells << '\n';
        f << "n_valid_gpu," << n_valid_gpu << '\n';
        f << "n_valid_cpu," << n_valid_cpu << '\n';
        f << "n_edge," << sample.n_edge << '\n';
        f << "n_ghost," << sample.n_ghost << '\n';
        f << "n_predicate_test_cloud," << n_predicate << '\n';
        f << "passthrough_kept," << r_pt.gpu_count << '\n';
        f << "box_kept," << r_box.gpu_count << '\n';
        f << "frustum_kept," << r_frustum.gpu_count << '\n';
        f << "fused_kept," << r_fused.gpu_count << '\n';
        f << "chained_stage1_n," << n1 << '\n';
        f << "chained_stage2_n," << n2 << '\n';
        f << "chained_stage3_n," << n3 << '\n';
        f << "fused_bytes_analytical," << fused_bytes << '\n';
        f << "chained_bytes_analytical," << chained_bytes << '\n';
        f << "roundtrip_collisions," << rt.collisions << '\n';
        f << "roundtrip_occupied," << rt.occupied << '\n';
        f << "collision_test_n_in," << n_collision << '\n';
        f << "collision_test_occupied_gpu," << coll_gpu.occupied << '\n';
        f << "collision_test_collisions_gpu," << coll_gpu.collisions << '\n';
        f << "collision_test_occupied_cpu," << coll_cpu.occupied << '\n';
        f << "collision_test_collisions_cpu," << coll_cpu.collisions << '\n';
        f << "edge_inside_frustum," << edge_inside_frustum << '\n';
        f << "edge_outside_frustum," << edge_outside_frustum << '\n';
        f << "time_o2u_gpu_ms," << o2u_gpu_ms << '\n';
        f << "time_o2u_cpu_ms," << o2u_cpu_ms << '\n';
        f << "time_chained_gpu_ms," << chained_gpu_ms << '\n';
        f << "time_fused_gpu_ms," << r_fused.gpu_ms << '\n';
        f << "time_collision_gpu_ms," << coll_gpu_ms << '\n';
        f << "time_collision_cpu_ms," << coll_cpu_ms << '\n';
    }
    std::printf("ARTIFACT: wrote demo/out/{full_topview.ppm, box_topview.ppm, frustum_topview.ppm, "
               "organized_occupancy_before.pgm, organized_occupancy_after.pgm, gates_metrics.csv}\n");

    // ---- 11) Cleanup -------------------------------------------------------
    CUDA_CHECK(cudaFree(d_organized));
    CUDA_CHECK(cudaFree(d_unorg_xyz));
    CUDA_CHECK(cudaFree(d_unorg_idx));
    CUDA_CHECK(cudaFree(d_predicate_xyz));
    CUDA_CHECK(cudaFree(d_stage1_xyz)); CUDA_CHECK(cudaFree(d_stage1_idx));
    CUDA_CHECK(cudaFree(d_stage2_xyz)); CUDA_CHECK(cudaFree(d_stage2_idx));
    CUDA_CHECK(cudaFree(d_stage3_xyz)); CUDA_CHECK(cudaFree(d_stage3_idx));
    CUDA_CHECK(cudaFree(d_roundtrip_organized));
    CUDA_CHECK(cudaFree(d_roundtrip_winner));
    CUDA_CHECK(cudaFree(d_collision_xyz));
    CUDA_CHECK(cudaFree(d_collision_organized));
    CUDA_CHECK(cudaFree(d_collision_winner));

    // ---- 12) Final verdict ---------------------------------------------------
    if (all_ok) {
        std::printf("RESULT: PASS (VERIFY(organized_to_unorganized/predicate_correctness/scan_bitexact) + "
                   "GATE(order_preservation/frustum_geometry/fused_vs_chained/roundtrip/collision_accounting) "
                   "all passed)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (a VERIFY or GATE above did not pass -- see stderr for details)\n");
        return EXIT_FAILURE;
    }
}
