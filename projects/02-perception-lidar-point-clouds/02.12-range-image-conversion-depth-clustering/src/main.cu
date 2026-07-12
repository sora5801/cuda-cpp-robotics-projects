// ===========================================================================
// main.cu — entry point for project 02.12 (Range-image conversion +
//           depth-clustering segmentation)
//
// Role in the project
// -------------------
// Orchestration: load the committed raw LiDAR scan (with per-point ring/
// azimuth-bin and generator-computed truth), convert it to the organized
// range image on the GPU, remove ground with a column-wise angle walk,
// build the depth-clustering (beta-criterion) edge graph, cluster it with
// the generic lock-free GPU union-find, compact the surviving obstacle
// points back to an unorganized cloud, cluster THAT with a voxel-hash
// Euclidean comparison pipeline (the SAME union-find kernels), verify every
// stage against an independent CPU reference, gate the designed scene's
// teaching scenarios, and write the demo artifacts. kernels.cu holds every
// GPU kernel; reference_cpu.cpp holds the independent CPU twins; kernels.cuh
// is the shared contract all three agree on — read it first, it walks the
// whole six-stage pipeline end to end.
//
// Output contract (load-bearing! — CLAUDE.md §6.1 rule, 02.04's identical
// convention)
// -------------------------------------------------------------------------
// demo/run_demo.ps1 diffs the STABLE lines of this program's stdout against
// demo/expected_output.txt. Stable = "[demo]", "PROBLEM:", "DATA:", every
// "VERIFY(...)"/"GATE ...:" verdict, "ARTIFACT:", and "RESULT:" — each is
// either constant or derived ONLY from the fixed committed input file, so
// none varies run to run. "[info]" and "[time]" lines carry machine- or
// run-varying NUMBERS and are deliberately NOT diffed.
//
// Read this after: kernels.cuh.  Read this before: kernels.cu, reference_cpu.cpp.
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
#include <map>
#include <set>
#include <algorithm>
#include <iterator>
#include <limits>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ===========================================================================
// Verification allowances — MEASURED THEN MARGINED (CLAUDE.md §12), the
// same discipline 02.03/02.04 apply to their own boundary-sensitive
// comparisons. Every VERIFY below is EXPECTED to be exactly 0 mismatches
// (the two implementations are independent but compute the same
// definition); a small nonzero allowance only guards against a genuine
// FMA/associativity rounding difference landing a measurement exactly on a
// threshold boundary (kGroundAngleThresholdDeg, kBetaThresholdDeg,
// kEuclideanClusterToleranceM) — see README "Expected output" for the
// actual measured mismatch counts on the reference machine (always 0 in
// practice for this scene: every boundary-relevant cohort was designed
// with a comfortable margin, THEORY.md "Numerical considerations" shows
// the arithmetic).
// ---------------------------------------------------------------------------
static const int kRangeImageAllowedMismatches   = 0;
static const int kGroundRemovalAllowedMismatches = 8;
static const int kDepthEdgesAllowedMismatches    = 8;
static const int kEuclidEdgesAllowedMismatches   = 8;

// Per-truth-object IoU floors for depth clustering's "clean" cohort — the
// four objects that are NOT specifically designed to interact with a
// neighbor (person(1) and wall_behind(2) interact by design, but each
// still forms its OWN clean single cluster once separated by the beta
// criterion; big_box(3) and far_pole(4) are isolated). thin_pole(5) and
// grazing_wall(6) are reported separately (GATE grazing_fragmentation,
// [info] thin_pole) because they are specifically designed NOT to look
// like a single clean cluster.
static const double kCleanIoUFloor = 0.85;

// grazing_wall(6) is specifically designed to FRAGMENT under depth
// clustering (a grazing/shallow-incidence surface — THEORY.md "The math"
// derives why) — this floor asserts the fragmentation actually happens
// (the known weakness demonstrated), not that it stays whole.
static const int kGrazingFragmentationFloor = 3;

// Ground removal precision/recall floor: our ground is a single flat plane
// with a well-conditioned angle test (THEORY.md shows the noise margin) —
// both should be comfortably high.
static const double kGroundPRFloor = 0.95;

// ===========================================================================
// RIMAGE01 binary sample format — see scripts/make_synthetic.py's
// write_binary_sample() for the authoritative description; also documented
// in data/README.md. Explicit fixed-width reads, never a raw struct fread
// (the same portability reasoning util/paths.h gives for avoiding
// <filesystem>).
// ===========================================================================
struct SampleHeader {
    int32_t n_points = 0;
    int32_t num_beams = 0, azimuth_bins = 0;
    float   sensor_height_m = 0.0f;
    float   ground_angle_threshold_deg = 0.0f;
    float   beta_threshold_deg = 0.0f;
    float   euclid_tolerance_m = 0.0f;
    int32_t min_cluster_size_depth = 0, min_cluster_size_euclid = 0;
    int32_t truth_num_objects = 0;
};

struct RawPoint {
    float x, y, z, range_m;
    int32_t ring, az_bin, truth_id;
};

static bool load_scene(const std::string& path, SampleHeader& hdr, std::vector<RawPoint>& points)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "error: could not open sample file '%s'\n", path.c_str());
        return false;
    }
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "RIMAGE01", 8) != 0) {
        std::fprintf(stderr, "error: '%s' does not start with the expected RIMAGE01 magic\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(&hdr.n_points), 4);
    f.read(reinterpret_cast<char*>(&hdr.num_beams), 4);
    f.read(reinterpret_cast<char*>(&hdr.azimuth_bins), 4);
    f.read(reinterpret_cast<char*>(&hdr.sensor_height_m), 4);
    f.read(reinterpret_cast<char*>(&hdr.ground_angle_threshold_deg), 4);
    f.read(reinterpret_cast<char*>(&hdr.beta_threshold_deg), 4);
    f.read(reinterpret_cast<char*>(&hdr.euclid_tolerance_m), 4);
    f.read(reinterpret_cast<char*>(&hdr.min_cluster_size_depth), 4);
    f.read(reinterpret_cast<char*>(&hdr.min_cluster_size_euclid), 4);
    f.read(reinterpret_cast<char*>(&hdr.truth_num_objects), 4);
    int32_t reserved[3];
    f.read(reinterpret_cast<char*>(reserved), 12);
    if (!f || hdr.n_points <= 0) {
        std::fprintf(stderr, "error: '%s' has a malformed header\n", path.c_str());
        return false;
    }

    // Data/code consistency checks (02.01's discipline): the sample was
    // DESIGNED around these compiled constants, so a mismatch means the
    // sample and the pipeline disagree about geometry every downstream
    // gate assumes.
    bool ok = true;
    if (hdr.num_beams != kNumBeams) { std::fprintf(stderr, "error: num_beams mismatch\n"); ok = false; }
    if (hdr.azimuth_bins != kAzimuthBins) { std::fprintf(stderr, "error: azimuth_bins mismatch\n"); ok = false; }
    if (std::fabs(hdr.sensor_height_m - kSensorHeightM) > 1.0e-6f) { std::fprintf(stderr, "error: sensor_height_m mismatch\n"); ok = false; }
    if (std::fabs(hdr.ground_angle_threshold_deg - kGroundAngleThresholdDeg) > 1.0e-6f) { std::fprintf(stderr, "error: ground_angle_threshold_deg mismatch\n"); ok = false; }
    if (std::fabs(hdr.beta_threshold_deg - kBetaThresholdDeg) > 1.0e-6f) { std::fprintf(stderr, "error: beta_threshold_deg mismatch\n"); ok = false; }
    if (std::fabs(hdr.euclid_tolerance_m - kEuclideanClusterToleranceM) > 1.0e-6f) { std::fprintf(stderr, "error: euclid_tolerance_m mismatch\n"); ok = false; }
    if (hdr.min_cluster_size_depth != kMinDepthClusterSize) { std::fprintf(stderr, "error: min_cluster_size_depth mismatch\n"); ok = false; }
    if (hdr.min_cluster_size_euclid != kMinEuclideanClusterSize) { std::fprintf(stderr, "error: min_cluster_size_euclid mismatch\n"); ok = false; }
    if (!ok) return false;

    points.resize(static_cast<size_t>(hdr.n_points));
    for (int32_t i = 0; i < hdr.n_points; ++i) {
        RawPoint& p = points[static_cast<size_t>(i)];
        f.read(reinterpret_cast<char*>(&p.x), 4);
        f.read(reinterpret_cast<char*>(&p.y), 4);
        f.read(reinterpret_cast<char*>(&p.z), 4);
        f.read(reinterpret_cast<char*>(&p.range_m), 4);
        f.read(reinterpret_cast<char*>(&p.ring), 4);
        f.read(reinterpret_cast<char*>(&p.az_bin), 4);
        f.read(reinterpret_cast<char*>(&p.truth_id), 4);
    }
    if (!f) {
        std::fprintf(stderr, "error: '%s' is truncated\n", path.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// PGM/PPM writers — the standard hand-rolled binary formats used throughout
// this repo (01.02/02.01/02.04, cited): "P5\n<W> <H>\n255\n" + raw gray
// bytes, or "P6\n<W> <H>\n255\n" + raw RGB triples.
// ---------------------------------------------------------------------------
static void write_pgm(const std::string& path, int w, int h, const std::vector<unsigned char>& gray)
{
    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << w << ' ' << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
}

static void write_ppm(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb)
{
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << ' ' << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

// color_for_display — a deterministic pseudo-color for a range-image cell,
// covering FOUR display categories with one function so every colored
// artifact (truth/depth/euclid label maps) uses an identical legend:
//   id == kDisplayNoReturn (-3): black   — no beam return in this cell.
//   id == kDisplayGround   (-4): dim blue-gray — ground (removed, not an object).
//   id == kNoCluster       (-1): mid gray — an obstacle cell whose raw
//                                 component was filtered as noise (below
//                                 the min-cluster-size floor) or, for the
//                                 TRUTH map, never used (truth is never -1).
//   id >= 0: a bright, well-separated hashed color (Knuth's multiplicative
//            hash, 02.04's identical technique, cited) — one per distinct
//            cluster/object id.
// ---------------------------------------------------------------------------
constexpr int kDisplayNoReturn = -3;
constexpr int kDisplayGround   = -4;

static void color_for_display(int id, unsigned char& r, unsigned char& g, unsigned char& b)
{
    if (id == kDisplayNoReturn) { r = 0; g = 0; b = 0; return; }
    if (id == kDisplayGround)   { r = 40; g = 40; b = 90; return; }
    if (id == kNoCluster)       { r = 90; g = 90; b = 90; return; }
    uint32_t h = static_cast<uint32_t>(id) * 2654435761u;   // Knuth's multiplicative hash constant
    h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    r = static_cast<unsigned char>(80 + (h & 0xFFu) % 176u);
    g = static_cast<unsigned char>(80 + ((h >> 8) & 0xFFu) % 176u);
    b = static_cast<unsigned char>(80 + ((h >> 16) & 0xFFu) % 176u);
}

// write_range_image_display — renders a per-cell scalar or id array as a
// row-repeated (kDisplayRowScale) raster so the 16-row image is visible at
// a normal viewing size, matching the "the sensor's-eye view" artifact this
// project's README calls the signature visual.
constexpr int kDisplayRowScale = 8;   // 16 rings * 8 = 128 px tall, 1024 px wide

static void write_range_pgm(const std::string& path, const std::vector<float>& range_img)
{
    std::vector<unsigned char> gray(static_cast<size_t>(kAzimuthBins) * kNumBeams * kDisplayRowScale);
    for (int ring = 0; ring < kNumBeams; ++ring) {
        for (int col = 0; col < kAzimuthBins; ++col) {
            const float r = range_img[static_cast<size_t>(organized_cell_index(ring, col))];
            const unsigned char v = static_cast<unsigned char>(
                std::min(255.0f, std::max(0.0f, r / kMaxRangeM * 255.0f)));
            for (int rep = 0; rep < kDisplayRowScale; ++rep) {
                const int out_row = ring * kDisplayRowScale + rep;
                gray[static_cast<size_t>(out_row) * kAzimuthBins + col] = v;
            }
        }
    }
    write_pgm(path, kAzimuthBins, kNumBeams * kDisplayRowScale, gray);
}

static void write_label_ppm(const std::string& path, const std::vector<float>& range_img,
                            const std::vector<int>& ground_or_truth_is_ground, // pass nullptr-equivalent via empty vector when not applicable
                            const std::vector<int>& id_per_cell, bool id_is_truth)
{
    std::vector<unsigned char> rgb(static_cast<size_t>(kAzimuthBins) * kNumBeams * kDisplayRowScale * 3);
    for (int ring = 0; ring < kNumBeams; ++ring) {
        for (int col = 0; col < kAzimuthBins; ++col) {
            const int c = organized_cell_index(ring, col);
            int display_id;
            if (range_img[static_cast<size_t>(c)] <= 0.0f) {
                display_id = kDisplayNoReturn;
            } else if (id_is_truth && id_per_cell[static_cast<size_t>(c)] == 0) {
                display_id = kDisplayGround;   // truth map: object id 0 IS ground
            } else if (!id_is_truth && !ground_or_truth_is_ground.empty() &&
                      ground_or_truth_is_ground[static_cast<size_t>(c)]) {
                display_id = kDisplayGround;   // predicted maps: use the GPU's own ground label
            } else {
                display_id = id_per_cell[static_cast<size_t>(c)];
            }
            unsigned char r, g, b;
            color_for_display(display_id, r, g, b);
            for (int rep = 0; rep < kDisplayRowScale; ++rep) {
                const int out_row = ring * kDisplayRowScale + rep;
                const size_t idx = (static_cast<size_t>(out_row) * kAzimuthBins + col) * 3;
                rgb[idx + 0] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
            }
        }
    }
    write_ppm(path, kAzimuthBins, kNumBeams * kDisplayRowScale, rgb);
}

// ---------------------------------------------------------------------------
// best_iou_per_truth — for every truth object id t present in truth_img
// (values 1..kernels.cuh's named object ids), find the predicted cluster id
// p with the highest IoU = |t cap p| / |t cup p| against pred_id_per_cell.
// A single pass builds truth/pred sizes and the intersection counts; a
// second pass (over the much smaller intersection map) finds each truth
// id's best match. O(num_cells + num_truth_pred_pairs) — trivial at this
// project's kNumCells=16,384 scale.
// ---------------------------------------------------------------------------
struct BestMatch {
    int pred_id = kNoCluster;
    int truth_count = 0, pred_count = 0, intersection = 0;
    double iou = 0.0;
};

static std::map<int, BestMatch> best_iou_per_truth(const std::vector<int>& truth_img,
                                                    const std::vector<int>& pred_id_per_cell,
                                                    int num_cells)
{
    std::unordered_map<int,int> truth_size, pred_size;
    std::map<std::pair<int,int>, int> inter;
    for (int c = 0; c < num_cells; ++c) {
        const int t = truth_img[static_cast<size_t>(c)];
        const int p = pred_id_per_cell[static_cast<size_t>(c)];
        if (t >= 1) ++truth_size[t];
        if (p != kNoCluster) ++pred_size[p];
        if (t >= 1 && p != kNoCluster) ++inter[{t, p}];
    }
    std::map<int, BestMatch> result;
    for (const auto& kv : truth_size) {
        BestMatch bm;
        bm.truth_count = kv.second;
        result[kv.first] = bm;
    }
    for (const auto& ikv : inter) {
        const int t = ikv.first.first, p = ikv.first.second, cnt = ikv.second;
        const int psize = pred_size[p];
        const int tsize = truth_size[t];
        const double iou = static_cast<double>(cnt) / static_cast<double>(tsize + psize - cnt);
        BestMatch& bm = result[t];
        if (iou > bm.iou) { bm.iou = iou; bm.pred_id = p; bm.pred_count = psize; bm.intersection = cnt; }
    }
    return result;
}

// canonical_edges_from_device — copy (edge_u,edge_v) back to host, zip into
// pairs, sort ascending — the standard canonicalization every VERIFY(edges)
// gate in this repo applies before a set-equality comparison (02.04, cited).
static std::vector<std::pair<int,int>> canonical_edges_from_device(const int* d_u, const int* d_v, int count)
{
    std::vector<int> hu(static_cast<size_t>(count)), hv(static_cast<size_t>(count));
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(hu.data(), d_u, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hv.data(), d_v, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::vector<std::pair<int,int>> edges;
    edges.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) edges.emplace_back(hu[static_cast<size_t>(i)], hv[static_cast<size_t>(i)]);
    std::sort(edges.begin(), edges.end());
    return edges;
}

static int64_t symmetric_difference_count(const std::vector<std::pair<int,int>>& a,
                                          const std::vector<std::pair<int,int>>& b)
{
    std::vector<std::pair<int,int>> only_a, only_b;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(only_a));
    std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(only_b));
    return static_cast<int64_t>(only_a.size() + only_b.size());
}

// run_union_find_gpu — the generic "init -> sweep until converged -> finalize"
// driver shared by BOTH graphs this project clusters (depth-image and
// Euclidean-comparison) — the host-side loop main.cu owns per repo
// convention (kernels.cu's launch_uf_* wrappers are single-call primitives;
// the CONVERGENCE LOOP is orchestration, so it lives here, mirroring
// 02.04's identical split).
static void run_union_find_gpu(int n, int num_edges, const int* d_edge_u, const int* d_edge_v,
                                int* d_parent, int* d_changed, int& sweeps_out, bool& converged_out)
{
    launch_uf_init(n, d_parent);
    sweeps_out = 0;
    converged_out = false;
    for (int s = 0; s < kMaxUfSweeps; ++s) {
        const bool changed = launch_uf_union_sweep(num_edges, d_edge_u, d_edge_v, d_parent, d_changed);
        ++sweeps_out;
        if (!changed) { converged_out = true; break; }
    }
    launch_uf_finalize(n, d_parent);
}

int main(int argc, char** argv)
{
    bool all_ok = true;

    std::printf("[demo] Range-image conversion + depth-clustering segmentation (Bogoslavskyi-Stachniss) "
               "vs voxel-hash Euclidean clustering, on the same non-ground points\n");
    print_device_info();

    // ---- 1) Load the committed raw scan ------------------------------------
    const std::string data_path = find_data_file("", argv[0], "range_image_scene.bin");
    if (data_path.empty()) {
        std::fprintf(stderr, "error: could not locate data/sample/range_image_scene.bin (see data/README.md)\n");
        return EXIT_FAILURE;
    }
    SampleHeader hdr;
    std::vector<RawPoint> raw;
    if (!load_scene(data_path, hdr, raw)) return EXIT_FAILURE;
    const int n_points = hdr.n_points;

    std::printf("PROBLEM: %d-beam x %d-azimuth-bin range image, %d raw scan points, "
               "ground_angle<=%.1f deg, beta>=%.1f deg, euclid_d=%.2f m\n",
               kNumBeams, kAzimuthBins, n_points, kGroundAngleThresholdDeg, kBetaThresholdDeg,
               kEuclideanClusterToleranceM);
    // Print only the FILENAME, never the resolved absolute path: find_data_file()
    // returns a path rooted at wherever THIS machine's checkout happens to
    // live, which would make the "DATA:" line vary machine to machine --
    // breaking the stable-line diff contract above. "[info]" carries the
    // full resolved path instead (deliberately NOT diffed).
    std::printf("[info] resolved sample path: %s\n", data_path.c_str());
    std::printf("DATA: loaded range_image_scene.bin (%d points, truth_num_objects=%d)\n",
               n_points, hdr.truth_num_objects);

    // Host SoA copies (device upload needs contiguous per-field arrays).
    std::vector<int>   h_ring(static_cast<size_t>(n_points)), h_az_bin(static_cast<size_t>(n_points));
    std::vector<float> h_range_m(static_cast<size_t>(n_points));
    std::vector<float> h_px(static_cast<size_t>(n_points)), h_py(static_cast<size_t>(n_points)), h_pz(static_cast<size_t>(n_points));
    std::vector<int>   h_ptruth(static_cast<size_t>(n_points));
    for (int i = 0; i < n_points; ++i) {
        const RawPoint& p = raw[static_cast<size_t>(i)];
        h_ring[static_cast<size_t>(i)] = p.ring;
        h_az_bin[static_cast<size_t>(i)] = p.az_bin;
        h_range_m[static_cast<size_t>(i)] = p.range_m;
        h_px[static_cast<size_t>(i)] = p.x; h_py[static_cast<size_t>(i)] = p.y; h_pz[static_cast<size_t>(i)] = p.z;
        h_ptruth[static_cast<size_t>(i)] = p.truth_id;
    }

    int *d_ring = nullptr, *d_az_bin = nullptr, *d_ptruth = nullptr;
    float *d_range_m = nullptr, *d_px = nullptr, *d_py = nullptr, *d_pz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ring, static_cast<size_t>(n_points) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_az_bin, static_cast<size_t>(n_points) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ptruth, static_cast<size_t>(n_points) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_range_m, static_cast<size_t>(n_points) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px, static_cast<size_t>(n_points) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, static_cast<size_t>(n_points) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pz, static_cast<size_t>(n_points) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ring, h_ring.data(), static_cast<size_t>(n_points) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_az_bin, h_az_bin.data(), static_cast<size_t>(n_points) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptruth, h_ptruth.data(), static_cast<size_t>(n_points) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_range_m, h_range_m.data(), static_cast<size_t>(n_points) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), static_cast<size_t>(n_points) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), static_cast<size_t>(n_points) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), static_cast<size_t>(n_points) * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 2) STAGE 1a: unorganized -> organized (range-image conversion) ---
    float *d_range_img = nullptr, *d_xyz_img = nullptr;
    int *d_truth_img = nullptr, *d_winner_idx_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_range_img, static_cast<size_t>(kNumCells) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xyz_img, static_cast<size_t>(kNumCells) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_truth_img, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_winner_idx_img, static_cast<size_t>(kNumCells) * sizeof(int)));

    GpuTimer gt;
    gt.begin();
    launch_scatter_to_organized(n_points, d_ring, d_az_bin, d_range_m, d_px, d_py, d_pz, d_ptruth,
                                d_range_img, d_xyz_img, d_truth_img, d_winner_idx_img);
    const float gpu_scatter_ms = gt.end_ms();

    std::vector<float> h_range_img(static_cast<size_t>(kNumCells)), h_xyz_img(static_cast<size_t>(kNumCells) * 3);
    std::vector<int> h_truth_img(static_cast<size_t>(kNumCells)), h_winner_idx_img(static_cast<size_t>(kNumCells));
    CUDA_CHECK(cudaMemcpy(h_range_img.data(), d_range_img, static_cast<size_t>(kNumCells) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_xyz_img.data(), d_xyz_img, static_cast<size_t>(kNumCells) * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_truth_img.data(), d_truth_img, static_cast<size_t>(kNumCells) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_winner_idx_img.data(), d_winner_idx_img, static_cast<size_t>(kNumCells) * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> cpu_range_img(static_cast<size_t>(kNumCells)), cpu_xyz_img(static_cast<size_t>(kNumCells) * 3);
    std::vector<int> cpu_truth_img(static_cast<size_t>(kNumCells)), cpu_winner_idx_img(static_cast<size_t>(kNumCells));
    CpuTimer ct; ct.begin();
    scatter_to_organized_cpu(n_points, h_ring.data(), h_az_bin.data(), h_range_m.data(),
                             h_px.data(), h_py.data(), h_pz.data(), h_ptruth.data(),
                             cpu_range_img.data(), cpu_xyz_img.data(), cpu_truth_img.data(), cpu_winner_idx_img.data());
    const double cpu_scatter_ms = ct.end_ms();

    int64_t range_mismatches = 0;
    int phantom_wins = 0;
    for (int c = 0; c < kNumCells; ++c) {
        if (h_range_img[static_cast<size_t>(c)] != cpu_range_img[static_cast<size_t>(c)]) ++range_mismatches;
        if (h_truth_img[static_cast<size_t>(c)] != cpu_truth_img[static_cast<size_t>(c)]) ++range_mismatches;
        if (h_truth_img[static_cast<size_t>(c)] == -2) ++phantom_wins;   // PHANTOM_TRUTH_ID: must never win
    }
    const bool range_image_pass = (range_mismatches <= kRangeImageAllowedMismatches) && (phantom_wins == 0);
    all_ok &= range_image_pass;
    std::printf("VERIFY(range_image): %s (%lld mismatch field(s) <= allowance %d; GPU atomicMin scatter "
               "bit-exact vs independent CPU running-minimum scatter)\n",
               range_image_pass ? "PASS" : "FAIL",
               static_cast<long long>(range_mismatches), kRangeImageAllowedMismatches);
    std::printf("GATE collision_resolution: %s (%d synthetic phantom point(s) won an organized cell, "
               "of 2 injected -- nearest-wins atomicMin race verified honest)\n",
               (phantom_wins == 0) ? "PASS" : "FAIL", phantom_wins);
    all_ok &= (phantom_wins == 0);

    // ---- 3) STAGE 2: ground removal ----------------------------------------
    int *d_ground_label = nullptr, *d_obstacle_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ground_label, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_obstacle_mask, static_cast<size_t>(kNumCells) * sizeof(int)));

    gt.begin();
    launch_ground_removal(d_range_img, d_xyz_img, d_ground_label, d_obstacle_mask);
    const float gpu_ground_ms = gt.end_ms();

    std::vector<int> h_ground_label(static_cast<size_t>(kNumCells)), h_obstacle_mask(static_cast<size_t>(kNumCells));
    CUDA_CHECK(cudaMemcpy(h_ground_label.data(), d_ground_label, static_cast<size_t>(kNumCells) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_obstacle_mask.data(), d_obstacle_mask, static_cast<size_t>(kNumCells) * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> cpu_ground_label(static_cast<size_t>(kNumCells)), cpu_obstacle_mask(static_cast<size_t>(kNumCells));
    ct.begin();
    ground_removal_cpu(h_range_img.data(), h_xyz_img.data(), cpu_ground_label.data(), cpu_obstacle_mask.data());
    const double cpu_ground_ms = ct.end_ms();

    int64_t ground_mismatches = 0;
    for (int c = 0; c < kNumCells; ++c) {
        if (h_range_img[static_cast<size_t>(c)] <= 0.0f) continue;   // only meaningful where a return exists
        if (h_ground_label[static_cast<size_t>(c)] != cpu_ground_label[static_cast<size_t>(c)]) ++ground_mismatches;
    }
    const bool ground_verify_pass = ground_mismatches <= kGroundRemovalAllowedMismatches;
    all_ok &= ground_verify_pass;
    std::printf("VERIFY(ground_removal): %s (%lld mismatch(es) <= allowance %d; GPU column-walk vs "
               "independent CPU column-walk)\n",
               ground_verify_pass ? "PASS" : "FAIL",
               static_cast<long long>(ground_mismatches), kGroundRemovalAllowedMismatches);

    // GATE ground_removal: precision/recall of the GPU's ground label vs the
    // generator's TRUTH label (truth_img==0 means "ground" by construction).
    long long tp = 0, fp = 0, fn = 0, tn = 0;
    for (int c = 0; c < kNumCells; ++c) {
        if (h_range_img[static_cast<size_t>(c)] <= 0.0f) continue;
        const bool pred_ground = h_ground_label[static_cast<size_t>(c)] != 0;
        const bool truth_ground = h_truth_img[static_cast<size_t>(c)] == 0;
        if (pred_ground && truth_ground) ++tp;
        else if (pred_ground && !truth_ground) ++fp;
        else if (!pred_ground && truth_ground) ++fn;
        else ++tn;
    }
    const double ground_precision = (tp + fp > 0) ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
    const double ground_recall = (tp + fn > 0) ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
    const bool ground_gate_pass = ground_precision >= kGroundPRFloor && ground_recall >= kGroundPRFloor;
    all_ok &= ground_gate_pass;
    std::printf("GATE ground_removal: %s (precision=%.4f, recall=%.4f, both >= floor %.2f; "
               "TP=%lld FP=%lld FN=%lld TN=%lld; 02.03's full RANSAC/CZM treatment [info] handles terrain "
               "this flat-ground column-walk cannot)\n",
               ground_gate_pass ? "PASS" : "FAIL", ground_precision, ground_recall, kGroundPRFloor,
               tp, fp, fn, tn);

    // ---- 4) STAGE 3: depth-clustering edges (the beta criterion) ----------
    int *d_edge_u_depth = nullptr, *d_edge_v_depth = nullptr;
    const int depth_edge_capacity = kNumCells * 2;
    CUDA_CHECK(cudaMalloc(&d_edge_u_depth, static_cast<size_t>(depth_edge_capacity) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_v_depth, static_cast<size_t>(depth_edge_capacity) * sizeof(int)));

    gt.begin();
    const int num_depth_edges = launch_depth_edges(d_range_img, d_obstacle_mask, d_edge_u_depth, d_edge_v_depth);
    const float gpu_depth_edges_ms = gt.end_ms();

    auto gpu_depth_edges = canonical_edges_from_device(d_edge_u_depth, d_edge_v_depth, num_depth_edges);

    ct.begin();
    auto cpu_depth_edges = depth_edges_cpu(h_range_img.data(), h_obstacle_mask.data());
    const double cpu_depth_edges_ms = ct.end_ms();

    const int64_t depth_edge_diff = symmetric_difference_count(gpu_depth_edges, cpu_depth_edges);
    const bool depth_edges_pass = depth_edge_diff <= kDepthEdgesAllowedMismatches;
    all_ok &= depth_edges_pass;
    std::printf("VERIFY(depth_edges): %s (%lld differing edge(s) <= allowance %d; GPU=%d edges, CPU=%zu edges; "
               "beta >= %.1f deg criterion, GPU device transcription vs independent CPU nested loop)\n",
               depth_edges_pass ? "PASS" : "FAIL", static_cast<long long>(depth_edge_diff), kDepthEdgesAllowedMismatches,
               num_depth_edges, cpu_depth_edges.size(), kBetaThresholdDeg);

    // ---- 5) STAGE 4a: union-find over the depth-image graph (size kNumCells) --
    int *d_parent_depth = nullptr, *d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_parent_depth, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    gt.begin();
    int uf_depth_sweeps = 0; bool uf_depth_converged = false;
    run_union_find_gpu(kNumCells, num_depth_edges, d_edge_u_depth, d_edge_v_depth, d_parent_depth, d_changed,
                       uf_depth_sweeps, uf_depth_converged);
    const float gpu_uf_depth_ms = gt.end_ms();

    std::vector<int> h_parent_depth(static_cast<size_t>(kNumCells));
    CUDA_CHECK(cudaMemcpy(h_parent_depth.data(), d_parent_depth, static_cast<size_t>(kNumCells) * sizeof(int), cudaMemcpyDeviceToHost));

    ct.begin();
    std::vector<int> cpu_parent_depth;
    serial_union_find_cpu(kNumCells, cpu_depth_edges, cpu_parent_depth);
    const double cpu_uf_depth_ms = ct.end_ms();

    int64_t uf_depth_mismatches = 0;
    for (int c = 0; c < kNumCells; ++c)
        if (h_parent_depth[static_cast<size_t>(c)] != cpu_parent_depth[static_cast<size_t>(c)]) ++uf_depth_mismatches;
    const bool uf_depth_pass = (uf_depth_mismatches == 0) && uf_depth_converged;
    all_ok &= uf_depth_pass;
    std::printf("VERIFY(union_find_depth): %s (%lld mismatch(es), bit-exact required; converged=%s in %d sweep(s), "
               "cap %d)\n",
               uf_depth_pass ? "PASS" : "FAIL", static_cast<long long>(uf_depth_mismatches),
               uf_depth_converged ? "yes" : "NO", uf_depth_sweeps, kMaxUfSweeps);

    // ---- 6) Host relabel: canonical roots -> dense depth-cluster ids, -----
    //         min-size filtered (see kernels.cuh's file header, "why this
    //         bookkeeping stays on the host": kNumCells=16,384 is trivial). --
    std::unordered_map<int,int> root_size_depth;
    for (int c = 0; c < kNumCells; ++c) {
        if (h_range_img[static_cast<size_t>(c)] <= 0.0f || !h_obstacle_mask[static_cast<size_t>(c)]) continue;
        ++root_size_depth[h_parent_depth[static_cast<size_t>(c)]];
    }
    std::unordered_map<int,int> root_to_dense_depth;
    {
        // Deterministic dense-id assignment: sort candidate roots ascending
        // (roots are cell indices, so this is just numeric order) and number
        // them in that order — reproducible across runs/machines.
        std::vector<int> roots;
        for (const auto& kv : root_size_depth)
            if (kv.second >= kMinDepthClusterSize) roots.push_back(kv.first);
        std::sort(roots.begin(), roots.end());
        for (size_t k = 0; k < roots.size(); ++k) root_to_dense_depth[roots[k]] = static_cast<int>(k);
    }
    std::vector<int> depth_cluster_id(static_cast<size_t>(kNumCells), kNoCluster);
    for (int c = 0; c < kNumCells; ++c) {
        if (h_range_img[static_cast<size_t>(c)] <= 0.0f || !h_obstacle_mask[static_cast<size_t>(c)]) continue;
        const int root = h_parent_depth[static_cast<size_t>(c)];
        const auto it = root_to_dense_depth.find(root);
        depth_cluster_id[static_cast<size_t>(c)] = (it != root_to_dense_depth.end()) ? it->second : kNoCluster;
    }
    const int depth_num_clusters = static_cast<int>(root_to_dense_depth.size());

    // ---- 7) STAGE 1b: compact obstacle cells -> flat point list -----------
    float* d_ob_xyz = nullptr; int *d_ob_cell_idx = nullptr, *d_ob_truth = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ob_xyz, static_cast<size_t>(kNumCells) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ob_cell_idx, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ob_truth, static_cast<size_t>(kNumCells) * sizeof(int)));

    gt.begin();
    const int M = launch_compact_obstacles(d_range_img, d_xyz_img, d_obstacle_mask, d_truth_img,
                                           d_ob_xyz, d_ob_cell_idx, d_ob_truth);
    const float gpu_compact_ms = gt.end_ms();

    std::vector<int> h_ob_cell_idx(static_cast<size_t>(M)), h_ob_truth(static_cast<size_t>(M));
    CUDA_CHECK(cudaMemcpy(h_ob_cell_idx.data(), d_ob_cell_idx, static_cast<size_t>(M) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ob_truth.data(), d_ob_truth, static_cast<size_t>(M) * sizeof(int), cudaMemcpyDeviceToHost));

    int64_t expected_M = 0;
    for (int c = 0; c < kNumCells; ++c)
        if (h_range_img[static_cast<size_t>(c)] > 0.0f && h_obstacle_mask[static_cast<size_t>(c)]) ++expected_M;
    const bool compaction_pass = (static_cast<int64_t>(M) == expected_M);
    all_ok &= compaction_pass;
    std::printf("GATE compaction_integrity: %s (compacted %d obstacle points; expected %lld from the "
               "organized grid's own ground/obstacle mask)\n",
               compaction_pass ? "PASS" : "FAIL", M, static_cast<long long>(expected_M));

    // ---- 8) STAGE 5: Euclidean comparison clustering (voxel hash) ---------
    unsigned long long *d_keys = nullptr, *d_keys_scratch = nullptr, *d_unique_key = nullptr;
    int *d_idx_sorted = nullptr, *d_is_start = nullptr, *d_seg_start = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(M) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_keys_scratch, static_cast<size_t>(M) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_unique_key, static_cast<size_t>(M) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_idx_sorted, static_cast<size_t>(M) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_start, static_cast<size_t>(M) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_start, static_cast<size_t>(M) * sizeof(int)));

    gt.begin();
    launch_compute_voxel_keys(M, d_ob_xyz, d_keys);
    const float gpu_voxel_keys_ms = gt.end_ms();

    gt.begin();
    const int num_voxels = launch_build_voxel_index(M, d_keys, d_keys_scratch, d_idx_sorted, d_is_start, d_seg_start, d_unique_key);
    const float gpu_voxel_index_ms = gt.end_ms();

    const int euclid_edge_capacity = M * kMaxEdgesPerPointEuclid;
    int *d_edge_u_euclid = nullptr, *d_edge_v_euclid = nullptr, *d_overflow = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_u_euclid, static_cast<size_t>(euclid_edge_capacity) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_v_euclid, static_cast<size_t>(euclid_edge_capacity) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(int)));

    gt.begin();
    const int num_euclid_edges = launch_build_edges_euclid(M, d_ob_xyz, d_keys, d_unique_key, num_voxels,
                                                            d_seg_start, d_idx_sorted, M,
                                                            d_edge_u_euclid, d_edge_v_euclid, euclid_edge_capacity,
                                                            d_overflow);
    const float gpu_euclid_edges_ms = gt.end_ms();

    int h_overflow = 0;
    CUDA_CHECK(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    const bool no_overflow = (h_overflow == 0);
    all_ok &= no_overflow;
    std::printf("GATE euclid_edge_capacity: %s (overflow_count=%d; capacity=%d edges = %d points x %d/point)\n",
               no_overflow ? "PASS" : "FAIL", h_overflow, euclid_edge_capacity, M, kMaxEdgesPerPointEuclid);

    auto gpu_euclid_edges = canonical_edges_from_device(d_edge_u_euclid, d_edge_v_euclid, num_euclid_edges);

    std::vector<float> h_ob_xyz(static_cast<size_t>(M) * 3);
    CUDA_CHECK(cudaMemcpy(h_ob_xyz.data(), d_ob_xyz, static_cast<size_t>(M) * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    ct.begin();
    auto cpu_euclid_edges = build_edges_euclid_cpu(M, h_ob_xyz.data());
    const double cpu_euclid_edges_ms = ct.end_ms();

    const int64_t euclid_edge_diff = symmetric_difference_count(gpu_euclid_edges, cpu_euclid_edges);
    const bool euclid_edges_pass = euclid_edge_diff <= kEuclidEdgesAllowedMismatches;
    all_ok &= euclid_edges_pass;
    std::printf("VERIFY(euclidean_edges): %s (%lld differing edge(s) <= allowance %d; GPU=%d edges, CPU=%zu edges; "
               "voxel-hash+27-cell-stencil vs independent unordered_map neighbor search)\n",
               euclid_edges_pass ? "PASS" : "FAIL", static_cast<long long>(euclid_edge_diff), kEuclidEdgesAllowedMismatches,
               num_euclid_edges, cpu_euclid_edges.size());

    // ---- 9) STAGE 4b: union-find over the Euclidean graph (size M) --------
    int* d_parent_euclid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_parent_euclid, static_cast<size_t>(M) * sizeof(int)));

    gt.begin();
    int uf_euclid_sweeps = 0; bool uf_euclid_converged = false;
    run_union_find_gpu(M, num_euclid_edges, d_edge_u_euclid, d_edge_v_euclid, d_parent_euclid, d_changed,
                       uf_euclid_sweeps, uf_euclid_converged);
    const float gpu_uf_euclid_ms = gt.end_ms();

    std::vector<int> h_parent_euclid(static_cast<size_t>(M));
    CUDA_CHECK(cudaMemcpy(h_parent_euclid.data(), d_parent_euclid, static_cast<size_t>(M) * sizeof(int), cudaMemcpyDeviceToHost));

    ct.begin();
    std::vector<int> cpu_parent_euclid;
    serial_union_find_cpu(M, cpu_euclid_edges, cpu_parent_euclid);
    const double cpu_uf_euclid_ms = ct.end_ms();

    int64_t uf_euclid_mismatches = 0;
    for (int i = 0; i < M; ++i)
        if (h_parent_euclid[static_cast<size_t>(i)] != cpu_parent_euclid[static_cast<size_t>(i)]) ++uf_euclid_mismatches;
    const bool uf_euclid_pass = (uf_euclid_mismatches == 0) && uf_euclid_converged;
    all_ok &= uf_euclid_pass;
    std::printf("VERIFY(union_find_euclid): %s (%lld mismatch(es), bit-exact required; converged=%s in %d sweep(s), "
               "cap %d)\n",
               uf_euclid_pass ? "PASS" : "FAIL", static_cast<long long>(uf_euclid_mismatches),
               uf_euclid_converged ? "yes" : "NO", uf_euclid_sweeps, kMaxUfSweeps);

    // ---- 10) Host relabel: Euclidean canonical roots -> dense ids, --------
    //          min-size filtered, mapped onto the full organized grid so
    //          the same best_iou_per_truth()/artifact code paths as depth
    //          clustering can be reused verbatim. ------------------------
    std::unordered_map<int,int> root_size_euclid;
    for (int i = 0; i < M; ++i) ++root_size_euclid[h_parent_euclid[static_cast<size_t>(i)]];
    std::unordered_map<int,int> root_to_dense_euclid;
    {
        std::vector<int> roots;
        for (const auto& kv : root_size_euclid)
            if (kv.second >= kMinEuclideanClusterSize) roots.push_back(kv.first);
        std::sort(roots.begin(), roots.end());
        for (size_t k = 0; k < roots.size(); ++k) root_to_dense_euclid[roots[k]] = static_cast<int>(k);
    }
    std::vector<int> euclid_cluster_id_pt(static_cast<size_t>(M), kNoCluster);
    for (int i = 0; i < M; ++i) {
        const int root = h_parent_euclid[static_cast<size_t>(i)];
        const auto it = root_to_dense_euclid.find(root);
        euclid_cluster_id_pt[static_cast<size_t>(i)] = (it != root_to_dense_euclid.end()) ? it->second : kNoCluster;
    }
    const int euclid_num_clusters = static_cast<int>(root_to_dense_euclid.size());
    std::vector<int> euclid_cluster_id_cell(static_cast<size_t>(kNumCells), kNoCluster);
    for (int i = 0; i < M; ++i)
        euclid_cluster_id_cell[static_cast<size_t>(h_ob_cell_idx[static_cast<size_t>(i)])] = euclid_cluster_id_pt[static_cast<size_t>(i)];

    // ---- 11) GATE partition_vs_truth (depth clustering vs the "clean" cohort) --
    // truth_for_iou: h_truth_img RESTRICTED to cells that survived ground
    // removal as obstacles. Rationale: a handful of cells right where an
    // object meets the floor are genuinely ambiguous to the column-walk
    // (the object's base is, physically, flush with the ground it stands
    // on -- the SAME honest ambiguity noted in kernels.cuh's ground-removal
    // design comment) and are already scored by their OWN gate
    // (GATE ground_removal, precision/recall vs truth). Leaving those cells
    // in THIS metric's denominator would silently blame the CLUSTERING
    // stage for a ground-removal-stage effect it never had a chance to
    // cluster in the first place -- measured concretely on this scene:
    // big_box's raw truth is 423 cells, 47 of which the column-walk calls
    // ground (its own gate's honest false-positive rate), leaving 376
    // obstacle cells that depth clustering recovers as ONE perfectly pure
    // component (IoU 1.000 against the restricted truth, vs. an
    // artificially deflated 0.889 against the unrestricted one).
    std::vector<int> truth_for_iou(static_cast<size_t>(kNumCells));
    for (int c = 0; c < kNumCells; ++c) {
        truth_for_iou[static_cast<size_t>(c)] =
            (h_range_img[static_cast<size_t>(c)] > 0.0f && h_obstacle_mask[static_cast<size_t>(c)])
                ? h_truth_img[static_cast<size_t>(c)] : -1;
    }
    // person(1)/big_box(3)/far_pole(4) are each a SINGLE, fully-visible
    // object -> a single predicted cluster should cover them almost
    // exactly. wall_behind(2) is deliberately EXCLUDED from this floor: it
    // is a wide panel with the person standing directly in front of its
    // middle, so the person's own silhouette occludes the strip of wall
    // directly behind it -- the wall's TWO VISIBLE FLANKS (left and right
    // of the person) are genuinely disconnected in the range image (no
    // image-adjacent path between them), so a correct depth-clustering
    // result SPLITS wall_behind into two clusters. That is the segmenter
    // doing its job on an occluded object, not an error -- best-SINGLE-
    // cluster IoU is the wrong metric for it, so it is reported as [info]
    // below instead of gated on a single-cluster floor.
    auto depth_iou = best_iou_per_truth(truth_for_iou, depth_cluster_id, kNumCells);
    const int kCleanObjectIds[3] = {1, 3, 4};   // person, big_box, far_pole
    bool partition_gate_pass = true;
    for (int t : kCleanObjectIds) {
        const auto it = depth_iou.find(t);
        const double iou = (it != depth_iou.end()) ? it->second.iou : 0.0;
        if (iou < kCleanIoUFloor) partition_gate_pass = false;
    }
    all_ok &= partition_gate_pass;
    std::printf("GATE partition_vs_truth: %s (best-IoU floor %.2f over the clean, unoccluded cohort "
               "{person,big_box,far_pole}: ", partition_gate_pass ? "PASS" : "FAIL", kCleanIoUFloor);
    for (int t : kCleanObjectIds) {
        const auto it = depth_iou.find(t);
        std::printf("id%d=%.3f ", t, (it != depth_iou.end()) ? it->second.iou : 0.0);
    }
    std::printf(")\n");

    // [info] wall_behind: report how many distinct depth clusters its
    // points fall into (expected: 2, one per visible flank) and what
    // fraction of the wall those clusters jointly, PURELY (no other
    // truth id's points) cover.
    {
        std::set<int> wall_pred_ids;
        int wall_total = 0, wall_covered = 0;
        for (int c = 0; c < kNumCells; ++c) {
            if (h_truth_img[static_cast<size_t>(c)] != 2) continue;
            ++wall_total;
            const int p = depth_cluster_id[static_cast<size_t>(c)];
            if (p != kNoCluster) { wall_pred_ids.insert(p); ++wall_covered; }
        }
        std::printf("[info] wall_behind: split into %zu depth-cluster(s) covering %d/%d cells (%.1f%%) -- "
                   "EXPECTED occlusion split (the person stands directly in front of its middle, "
                   "disconnecting the left/right visible flanks in the range image), not a segmentation error\n",
                   wall_pred_ids.size(), wall_covered, wall_total,
                   wall_total > 0 ? 100.0 * wall_covered / wall_total : 0.0);
    }

    // ---- 12) GATE depth_gap_showcase ---------------------------------------
    // Depth clustering: person(1) and wall_behind(2) must land in DISJOINT
    // sets of valid predicted cluster ids (no shared cluster).
    std::set<int> depth_ids_person, depth_ids_wall;
    for (int c = 0; c < kNumCells; ++c) {
        const int t = h_truth_img[static_cast<size_t>(c)];
        const int p = depth_cluster_id[static_cast<size_t>(c)];
        if (p == kNoCluster) continue;
        if (t == 1) depth_ids_person.insert(p);
        else if (t == 2) depth_ids_wall.insert(p);
    }
    std::vector<int> depth_shared;
    std::set_intersection(depth_ids_person.begin(), depth_ids_person.end(),
                          depth_ids_wall.begin(), depth_ids_wall.end(), std::back_inserter(depth_shared));
    // Euclidean comparison: expect at least one SHARED valid cluster id
    // (i.e. Euclidean merges the two objects because their 3-D gap is
    // smaller than kEuclideanClusterToleranceM).
    std::set<int> euclid_ids_person, euclid_ids_wall;
    for (int c = 0; c < kNumCells; ++c) {
        const int t = h_truth_img[static_cast<size_t>(c)];
        const int p = euclid_cluster_id_cell[static_cast<size_t>(c)];
        if (p == kNoCluster) continue;
        if (t == 1) euclid_ids_person.insert(p);
        else if (t == 2) euclid_ids_wall.insert(p);
    }
    std::vector<int> euclid_shared;
    std::set_intersection(euclid_ids_person.begin(), euclid_ids_person.end(),
                          euclid_ids_wall.begin(), euclid_ids_wall.end(), std::back_inserter(euclid_shared));

    const bool depth_gap_pass = depth_shared.empty() && !euclid_shared.empty();
    all_ok &= depth_gap_pass;
    std::printf("GATE depth_gap_showcase: %s (person/wall_behind visible gap=%.2f m < euclid_d=%.2f m: depth "
               "clustering shares %zu cluster(s) [expect 0 -- separated at any range], Euclidean shares "
               "%zu cluster(s) [expect >=1 -- fixed-radius merge])\n",
               depth_gap_pass ? "PASS" : "FAIL", 0.19, kEuclideanClusterToleranceM,
               depth_shared.size(), euclid_shared.size());

    // ---- 13) GATE grazing_fragmentation ------------------------------------
    std::set<int> depth_ids_grazing, euclid_ids_grazing;
    for (int c = 0; c < kNumCells; ++c) {
        if (h_truth_img[static_cast<size_t>(c)] != 6) continue;
        if (depth_cluster_id[static_cast<size_t>(c)] != kNoCluster) depth_ids_grazing.insert(depth_cluster_id[static_cast<size_t>(c)]);
        if (euclid_cluster_id_cell[static_cast<size_t>(c)] != kNoCluster) euclid_ids_grazing.insert(euclid_cluster_id_cell[static_cast<size_t>(c)]);
    }
    const bool grazing_pass = static_cast<int>(depth_ids_grazing.size()) >= kGrazingFragmentationFloor;
    all_ok &= grazing_pass;
    std::printf("GATE grazing_fragmentation: %s (grazing_wall fragments into %zu depth-cluster(s) "
               "[floor %d -- the beta criterion's known weakness at shallow incidence]; Euclidean "
               "clustering, blind to viewing angle, keeps it in %zu cluster(s) [info])\n",
               grazing_pass ? "PASS" : "FAIL", depth_ids_grazing.size(), kGrazingFragmentationFloor,
               euclid_ids_grazing.size());

    // ---- 14) [info] thin_pole ------------------------------------------------
    std::set<int> depth_ids_pole;
    int pole_raw_size = 0;
    for (int c = 0; c < kNumCells; ++c) {
        if (h_truth_img[static_cast<size_t>(c)] != 5) continue;
        const auto rit = root_size_depth.find(h_parent_depth[static_cast<size_t>(c)]);
        if (rit != root_size_depth.end()) pole_raw_size = std::max(pole_raw_size, rit->second);
        if (depth_cluster_id[static_cast<size_t>(c)] != kNoCluster) depth_ids_pole.insert(depth_cluster_id[static_cast<size_t>(c)]);
    }
    const bool pole_survived = !depth_ids_pole.empty();
    const auto pole_it = depth_iou.find(5);
    const double pole_iou = (pole_it != depth_iou.end()) ? pole_it->second.iou : 0.0;
    if (pole_survived) {
        std::printf("[info] thin_pole: raw component size=%d (min-size floor %d) -> SURVIVES the filter; "
                   "best-IoU=%.3f\n", pole_raw_size, kMinDepthClusterSize, pole_iou);
    } else {
        std::printf("[info] thin_pole: raw component size=%d (min-size floor %d) -> FILTERED OUT by the "
                   "filter (a real but thin object lost to noise rejection -- the honest min-size trade)\n",
                   pole_raw_size, kMinDepthClusterSize);
    }

    // ---- 15) GATE timing_payoff --------------------------------------------
    const double depth_path_ms = static_cast<double>(gpu_ground_ms) + gpu_depth_edges_ms + gpu_uf_depth_ms;
    const double euclid_path_ms = static_cast<double>(gpu_voxel_keys_ms) + gpu_voxel_index_ms + gpu_euclid_edges_ms + gpu_uf_euclid_ms;
    const bool timing_measured = depth_path_ms > 0.0 && euclid_path_ms > 0.0;
    all_ok &= timing_measured;
    // The GATE line itself stays STABLE (no raw ms -- those jitter run to
    // run and machine to machine); the actual numbers are printed on the
    // "[time]" lines below, which are deliberately not diffed.
    std::printf("GATE timing_payoff: %s (both the depth-image path and the voxel-hash Euclidean path "
               "were measured with cudaEvents -- see the '[time] ... path total' lines below for the "
               "actual numbers, a single-shot teaching artifact, not a benchmark claim)\n",
               timing_measured ? "PASS" : "FAIL");

    // ---- 16) Artifacts ------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);

    write_range_pgm(out_dir + "/range_image.pgm", h_range_img);

    {
        std::vector<int> empty_ground;
        write_label_ppm(out_dir + "/truth_labels.ppm", h_range_img, empty_ground, h_truth_img, /*id_is_truth=*/true);
        write_label_ppm(out_dir + "/depth_cluster_labels.ppm", h_range_img, h_ground_label, depth_cluster_id, /*id_is_truth=*/false);
        write_label_ppm(out_dir + "/euclid_cluster_labels.ppm", h_range_img, h_ground_label, euclid_cluster_id_cell, /*id_is_truth=*/false);
    }

    // beta_angle_map.csv — the person/wall depth-gap boundary column pair:
    // find the first azimuth-adjacent (col, col+1-wrapped) pair where truth
    // flips 1(person) -> 2(wall_behind), then dump beta across every ring
    // where both columns have a valid obstacle return.
    {
        int chosen_col = -1, chosen_next = -1;
        for (int col = 0; col < kAzimuthBins && chosen_col < 0; ++col) {
            const int next_col = (col + 1 == kAzimuthBins) ? 0 : col + 1;
            for (int ring = 0; ring < kNumBeams; ++ring) {
                const int c = organized_cell_index(ring, col), nc = organized_cell_index(ring, next_col);
                if (h_truth_img[static_cast<size_t>(c)] == 1 && h_truth_img[static_cast<size_t>(nc)] == 2) {
                    chosen_col = col; chosen_next = next_col; break;
                }
                if (h_truth_img[static_cast<size_t>(c)] == 2 && h_truth_img[static_cast<size_t>(nc)] == 1) {
                    chosen_col = col; chosen_next = next_col; break;
                }
            }
        }
        std::ofstream f(out_dir + "/beta_angle_map.csv");
        f << "# beta_angle_map.csv -- the beta (depth-discontinuity) criterion across one azimuth-adjacent "
             "column pair, project 02.12. columns: ring,range_a_m,range_b_m,beta_deg,connected\n";
        f << "ring,range_a_m,range_b_m,beta_deg,connected\n";
        if (chosen_col >= 0) {
            for (int ring = 0; ring < kNumBeams; ++ring) {
                const int c = organized_cell_index(ring, chosen_col), nc = organized_cell_index(ring, chosen_next);
                const float ra = h_range_img[static_cast<size_t>(c)], rb = h_range_img[static_cast<size_t>(nc)];
                if (ra <= 0.0f || rb <= 0.0f) continue;
                const float r1 = std::max(ra, rb), r2 = std::min(ra, rb);
                const float beta_deg = beta_criterion_rad(r1, r2, kAzimuthStepRad) * (180.0f / kPi);
                f << ring << ',' << ra << ',' << rb << ',' << beta_deg << ',' << (beta_deg >= kBetaThresholdDeg ? 1 : 0) << '\n';
            }
        }
    }

    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.12\n";
        f << "metric,value\n";
        f << "n_points," << n_points << '\n';
        f << "num_cells," << kNumCells << '\n';
        f << "num_obstacle_points_M," << M << '\n';
        f << "num_depth_edges," << num_depth_edges << '\n';
        f << "num_euclid_edges," << num_euclid_edges << '\n';
        f << "depth_num_clusters," << depth_num_clusters << '\n';
        f << "euclid_num_clusters," << euclid_num_clusters << '\n';
        f << "ground_precision," << ground_precision << '\n';
        f << "ground_recall," << ground_recall << '\n';
        for (const auto& kv : depth_iou)
            f << "depth_iou_truth_" << kv.first << ',' << kv.second.iou << '\n';
        f << "depth_gap_depth_shared_clusters," << depth_shared.size() << '\n';
        f << "depth_gap_euclid_shared_clusters," << euclid_shared.size() << '\n';
        f << "grazing_depth_fragments," << depth_ids_grazing.size() << '\n';
        f << "grazing_euclid_fragments," << euclid_ids_grazing.size() << '\n';
        f << "thin_pole_raw_size," << pole_raw_size << '\n';
        f << "thin_pole_survived," << (pole_survived ? 1 : 0) << '\n';
        f << "uf_depth_sweeps," << uf_depth_sweeps << '\n';
        f << "uf_euclid_sweeps," << uf_euclid_sweeps << '\n';
        f << "gpu_scatter_ms," << gpu_scatter_ms << '\n';
        f << "gpu_ground_ms," << gpu_ground_ms << '\n';
        f << "gpu_depth_edges_ms," << gpu_depth_edges_ms << '\n';
        f << "gpu_uf_depth_ms," << gpu_uf_depth_ms << '\n';
        f << "gpu_compact_ms," << gpu_compact_ms << '\n';
        f << "gpu_voxel_keys_ms," << gpu_voxel_keys_ms << '\n';
        f << "gpu_voxel_index_ms," << gpu_voxel_index_ms << '\n';
        f << "gpu_euclid_edges_ms," << gpu_euclid_edges_ms << '\n';
        f << "gpu_uf_euclid_ms," << gpu_uf_euclid_ms << '\n';
        f << "depth_path_total_ms," << depth_path_ms << '\n';
        f << "euclid_path_total_ms," << euclid_path_ms << '\n';
        f << "cpu_scatter_ms," << cpu_scatter_ms << '\n';
        f << "cpu_ground_ms," << cpu_ground_ms << '\n';
        f << "cpu_depth_edges_ms," << cpu_depth_edges_ms << '\n';
        f << "cpu_uf_depth_ms," << cpu_uf_depth_ms << '\n';
        f << "cpu_euclid_edges_ms," << cpu_euclid_edges_ms << '\n';
        f << "cpu_uf_euclid_ms," << cpu_uf_euclid_ms << '\n';
    }

    std::printf("ARTIFACT: wrote demo/out/{range_image.pgm, truth_labels.ppm, depth_cluster_labels.ppm, "
               "euclid_cluster_labels.ppm, beta_angle_map.csv, gates_metrics.csv}\n");

    // ---- 17) Timing summary ([time] lines: NOT diffed) ----------------------
    std::printf("[time] GPU scatter-to-organized: %.4f ms (CPU twin: %.4f ms)\n", static_cast<double>(gpu_scatter_ms), cpu_scatter_ms);
    std::printf("[time] GPU ground removal:       %.4f ms (CPU twin: %.4f ms)\n", static_cast<double>(gpu_ground_ms), cpu_ground_ms);
    std::printf("[time] GPU depth edges:          %.4f ms (CPU twin: %.4f ms)\n", static_cast<double>(gpu_depth_edges_ms), cpu_depth_edges_ms);
    std::printf("[time] GPU union-find (depth):   %.4f ms (CPU twin: %.4f ms)\n", static_cast<double>(gpu_uf_depth_ms), cpu_uf_depth_ms);
    std::printf("[time] GPU compact obstacles:    %.4f ms\n", static_cast<double>(gpu_compact_ms));
    std::printf("[time] GPU voxel keys+index:     %.4f ms\n", static_cast<double>(gpu_voxel_keys_ms) + gpu_voxel_index_ms);
    std::printf("[time] GPU euclid edges:         %.4f ms (CPU twin: %.4f ms)\n", static_cast<double>(gpu_euclid_edges_ms), cpu_euclid_edges_ms);
    std::printf("[time] GPU union-find (euclid):  %.4f ms (CPU twin: %.4f ms)\n", static_cast<double>(gpu_uf_euclid_ms), cpu_uf_euclid_ms);
    std::printf("[time] depth-image path total:   %.4f ms\n", depth_path_ms);
    std::printf("[time] Euclidean path total:      %.4f ms\n", euclid_path_ms);
    if (depth_path_ms > 0.0)
        std::printf("[time] timing ratio (euclid/depth, teaching artifact not a benchmark): %.2fx\n", euclid_path_ms / depth_path_ms);

    // ---- 18) Cleanup ----------------------------------------------------------
    CUDA_CHECK(cudaFree(d_ring)); CUDA_CHECK(cudaFree(d_az_bin)); CUDA_CHECK(cudaFree(d_ptruth));
    CUDA_CHECK(cudaFree(d_range_m)); CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_range_img)); CUDA_CHECK(cudaFree(d_xyz_img)); CUDA_CHECK(cudaFree(d_truth_img)); CUDA_CHECK(cudaFree(d_winner_idx_img));
    CUDA_CHECK(cudaFree(d_ground_label)); CUDA_CHECK(cudaFree(d_obstacle_mask));
    CUDA_CHECK(cudaFree(d_edge_u_depth)); CUDA_CHECK(cudaFree(d_edge_v_depth));
    CUDA_CHECK(cudaFree(d_parent_depth)); CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_ob_xyz)); CUDA_CHECK(cudaFree(d_ob_cell_idx)); CUDA_CHECK(cudaFree(d_ob_truth));
    CUDA_CHECK(cudaFree(d_keys)); CUDA_CHECK(cudaFree(d_keys_scratch)); CUDA_CHECK(cudaFree(d_unique_key));
    CUDA_CHECK(cudaFree(d_idx_sorted)); CUDA_CHECK(cudaFree(d_is_start)); CUDA_CHECK(cudaFree(d_seg_start));
    CUDA_CHECK(cudaFree(d_edge_u_euclid)); CUDA_CHECK(cudaFree(d_edge_v_euclid)); CUDA_CHECK(cudaFree(d_overflow));
    CUDA_CHECK(cudaFree(d_parent_euclid));

    // ---- 19) Verdict ------------------------------------------------------------
    if (all_ok) {
        std::printf("RESULT: PASS (VERIFY(range_image/ground_removal/depth_edges/union_find_depth/"
                   "euclidean_edges/union_find_euclid) + all gates passed: collision_resolution, "
                   "ground_removal, compaction_integrity, euclid_edge_capacity, partition_vs_truth, "
                   "depth_gap_showcase, grazing_fragmentation, timing_payoff)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (a VERIFY or GATE above did not pass -- see stderr for details)\n");
        return EXIT_FAILURE;
    }
}
