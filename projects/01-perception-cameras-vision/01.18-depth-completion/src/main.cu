// ===========================================================================
// main.cu — entry point for project 01.18
//           Depth completion: sparse LiDAR + RGB -> dense depth
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed synthetic sample: rgb.ppm (guidance image),
//      truth_depth.bin (EXACT dense depth, evaluation-only), and
//      lidar_points.csv (raw LiDAR returns, LIDAR frame). Subsample the
//      LiDAR set to the demo's default density (~5%, see PROBLEM line).
//   2. VERIFY STAGE (CLAUDE.md §5): run all four pipeline stages
//      (projection+z-buffer, conductance, diffusion, IDW) on BOTH the GPU
//      kernels and the independent CPU twins, on the SAME inputs, and
//      require element-wise agreement within a documented tolerance.
//   3. EVALUATION GATES: compare the GPU-computed guided (diffusion) and
//      IDW-baseline densified depth fields against the synthetic scene's
//      exact ground truth — overall accuracy, edge quality (the reason
//      the guided method should beat IDW), the texture-trap and
//      camo-edge honesty checks (README/THEORY name these), and input
//      fidelity (Dirichlet anchoring actually holds).
//   4. DENSITY SWEEP ([info] only): re-run the guided pipeline at three
//      LiDAR densities and report whether accuracy improves monotonically.
//   5. ARTIFACTS: write rgb/sparse/completed/truth/error PGMs and a
//      gates_metrics.csv into demo/out/ (CLAUDE.md §6.3).
//
// Output contract: stable lines are "[demo]", "PROBLEM:", "VERIFY:",
// "GATE:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" lines are NOT diffed
// (device names and timings vary by machine). Change a stable line -> update
// demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

// ===========================================================================
// Tiny file-format helpers — PPM/PGM in, PGM/CSV out. Deliberately dumb,
// hand-rolled parsers (no stb_image — CLAUDE.md §5 default dependency
// budget is "CUDA toolkit + C++17 stdlib only", and PPM/PGM's ASCII header
// + flat binary body is trivial to read/write by hand, which is itself the
// point: a learner should be able to read every byte this project touches).
// ===========================================================================

// read_ppm — parse a binary P6 PPM (the exact format write_ppm() in
// scripts/make_synthetic.py writes: "P6\nW H\n255\n" then W*H*3 raw bytes).
// Returns true and fills rgb ([w*h*3] bytes) on success. Rejects any width/
// height other than kImageWidth/kImageHeight — every stage downstream is
// sized for exactly one image, and silently resizing would hide a data
// mismatch instead of failing loudly (CLAUDE.md §13 honesty).
static bool read_ppm(const std::string& path, std::vector<uint8_t>& rgb)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string magic; int w = 0, h = 0, maxval = 0;
    f >> magic >> w >> h >> maxval;
    f.get();   // consume the single whitespace byte between the header and the binary body
    if (magic != "P6" || w != kImageWidth || h != kImageHeight || maxval != 255) return false;
    rgb.resize(static_cast<size_t>(w) * h * 3);
    f.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return f.good() || f.eof();
}

// read_depth_bin — parse the raw float32, row-major depth dump
// scripts/make_synthetic.py's write_depth_bin() produces. Exactly
// kImagePixels floats, no header (the format's ONLY consumer is this
// project, and both writer and reader hardcode the same fixed size).
static bool read_depth_bin(const std::string& path, std::vector<float>& depth)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    depth.resize(static_cast<size_t>(kImagePixels));
    const std::streamsize want = static_cast<std::streamsize>(depth.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(depth.data()), want);
    return f.gcount() == want;   // reject a truncated/oversized file rather than silently accepting it

}

// read_lidar_csv — parse "x,y,z" rows (LIDAR-frame meters), skipping '#'
// comment lines and the "x,y,z" header row scripts/make_synthetic.py writes.
static bool read_lidar_csv(const std::string& path, std::vector<LidarPointF>& pts)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 'x') continue;   // comment or header row
        std::stringstream ss(line);
        std::string cell;
        LidarPointF p{};
        if (!std::getline(ss, cell, ',')) continue; p.x = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) continue; p.y = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) continue; p.z = std::strtof(cell.c_str(), nullptr);
        pts.push_back(p);
    }
    return !pts.empty();
}

static void write_pgm(const std::string& path, const std::vector<uint8_t>& gray)
{
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << kImageWidth << " " << kImageHeight << "\n255\n";
    f.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
}

// depth_to_gray — visualization convention used by EVERY depth PGM this
// project writes: NEAR = BRIGHT. depth in [near_m, far_m] maps LINEARLY to
// gray in [255, 0]; kInvalidDepth (or any depth <= 0) maps to 0 (black) —
// "no data here" reads as darkest, never confusable with "very far".
static std::vector<uint8_t> depth_to_gray(const std::vector<float>& depth, float near_m, float far_m)
{
    std::vector<uint8_t> gray(depth.size());
    for (size_t i = 0; i < depth.size(); ++i) {
        const float d = depth[i];
        if (d <= 0.0f) { gray[i] = 0; continue; }
        float t = (d - near_m) / (far_m - near_m);
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        gray[i] = static_cast<uint8_t>(255.0f * (1.0f - t) + 0.5f);
    }
    return gray;
}

// dilate_for_visibility — splat every valid pixel's gray value into its
// (2*r+1)x(2*r+1) neighborhood. At ~5% density, individual sparse samples
// are near-invisible single pixels in a 160x120 image; this is PURELY a
// visualization aid for sparse_depth_vis.pgm (never fed back into the
// algorithm) so a learner can actually see the scan-line sample pattern.
static std::vector<uint8_t> dilate_for_visibility(const std::vector<uint8_t>& gray, int r)
{
    std::vector<uint8_t> out(gray.size(), 0);
    for (int y = 0; y < kImageHeight; ++y) {
        for (int x = 0; x < kImageWidth; ++x) {
            const uint8_t g = gray[static_cast<size_t>(y) * kImageWidth + x];
            if (g == 0) continue;
            for (int dy = -r; dy <= r; ++dy) {
                const int sy = y + dy;
                if (sy < 0 || sy >= kImageHeight) continue;
                for (int dx = -r; dx <= r; ++dx) {
                    const int sx = x + dx;
                    if (sx < 0 || sx >= kImageWidth) continue;
                    out[static_cast<size_t>(sy) * kImageWidth + sx] = g;
                }
            }
        }
    }
    return out;
}

// error_to_gray — |completed - truth| in meters mapped to [0, kErrVisCapM]
// -> gray [0,255] (BRIGHT = MORE error, the opposite sense of depth_to_gray
// on purpose — an error map and a depth map should never be visually
// confusable). Pixels with no truth are left black (not evaluated).
static constexpr float kErrVisCapM = 3.0f;
static std::vector<uint8_t> error_to_gray(const std::vector<float>& completed,
                                          const std::vector<float>& truth)
{
    std::vector<uint8_t> gray(completed.size(), 0);
    for (size_t i = 0; i < completed.size(); ++i) {
        if (truth[i] == kInvalidDepth) continue;
        const float e = std::fabs(completed[i] - truth[i]);
        const float t = e / kErrVisCapM > 1.0f ? 1.0f : e / kErrVisCapM;
        gray[i] = static_cast<uint8_t>(255.0f * t + 0.5f);
    }
    return gray;
}

// ===========================================================================
// Small evaluation-only host helpers (NOT part of the verified pipeline —
// these compare an ALREADY-COMPUTED densified field against the synthetic
// scene's ground truth, so they are the "independent gate" README/THEORY
// promise on top of the GPU-vs-CPU twin comparisons above).
// ===========================================================================

struct ErrorStats { double mae = 0.0; double rmse = 0.0; int n = 0; };

static ErrorStats compute_error(const std::vector<float>& completed,
                                const std::vector<float>& truth,
                                const std::vector<uint8_t>* mask = nullptr)
{
    ErrorStats s;
    double sum_abs = 0.0, sum_sq = 0.0;
    for (int i = 0; i < kImagePixels; ++i) {
        if (truth[static_cast<size_t>(i)] == kInvalidDepth) continue;
        if (mask && !(*mask)[static_cast<size_t>(i)]) continue;
        const double e = static_cast<double>(completed[static_cast<size_t>(i)]) -
                         static_cast<double>(truth[static_cast<size_t>(i)]);
        sum_abs += std::fabs(e);
        sum_sq  += e * e;
        s.n++;
    }
    if (s.n > 0) { s.mae = sum_abs / s.n; s.rmse = std::sqrt(sum_sq / s.n); }
    return s;
}

// max_channel_diff_host — main.cu's OWN small copy of the same max-abs-
// per-channel-difference measure kernels.cu/reference_cpu.cpp use for
// conductance (kernels.cuh's compute_conductance_kernel doc-comment), used
// here purely for EVALUATION (classifying regions against ground truth) —
// keeping the same formula means "what the algorithm sees as an edge" and
// "what the gates classify as an edge" agree by construction.
static inline float max_channel_diff_host(const std::vector<float>& rgbf, int a_idx, int b_idx)
{
    const float dr = std::fabs(rgbf[static_cast<size_t>(a_idx)]                                    - rgbf[static_cast<size_t>(b_idx)]);
    const float dg = std::fabs(rgbf[static_cast<size_t>(a_idx) + kImagePixels]                     - rgbf[static_cast<size_t>(b_idx) + kImagePixels]);
    const float db = std::fabs(rgbf[static_cast<size_t>(a_idx) + 2 * kImagePixels]                 - rgbf[static_cast<size_t>(b_idx) + 2 * kImagePixels]);
    return std::max(dr, std::max(dg, db));
}

// build_edge_masks — classify every INTERIOR pixel (both forward neighbors
// exist and have valid truth depth) into the regions README/THEORY name,
// using ONLY the ground truth depth and the guidance color image — never
// scene metadata, so the classification is a genuine measurement, not a
// lookup table. See THEORY.md "How we verify correctness" for the
// threshold choices and why detecting these regions this way is robust.
struct EdgeMasks {
    std::vector<uint8_t> boundary;      // real depth edges (any cause)
    std::vector<uint8_t> texture_trap;  // strong RGB edge, flat truth depth
    std::vector<uint8_t> camo_edge;     // real depth edge, near-zero RGB edge
    std::vector<uint8_t> boundary_clean; // boundary AND NOT camo_edge (edge_quality gate uses this)
    std::vector<uint8_t> flat_interior; // flat depth AND ordinary (non-checkerboard) RGB — the "easy" baseline
};

static constexpr float kDepthEdgeThreshM  = 0.30f;   // m, neighbor jump considered a REAL depth discontinuity
static constexpr float kDepthFlatThreshM  = 0.03f;   // m, neighbor jump considered "no discontinuity"
static constexpr float kRgbEdgeThreshHigh = 0.20f;   // normalized [0,1] max-channel jump considered "strong texture"
static constexpr float kRgbEdgeThreshLow  = 0.03f;   // normalized [0,1] max-channel jump considered "near-zero contrast"

static EdgeMasks build_edge_masks(const std::vector<float>& truth, const std::vector<float>& rgbf)
{
    EdgeMasks m;
    m.boundary.assign(kImagePixels, 0);
    m.texture_trap.assign(kImagePixels, 0);
    m.camo_edge.assign(kImagePixels, 0);
    m.boundary_clean.assign(kImagePixels, 0);
    m.flat_interior.assign(kImagePixels, 0);

    for (int y = 0; y < kImageHeight - 1; ++y) {
        for (int x = 0; x < kImageWidth - 1; ++x) {
            const int idx = y * kImageWidth + x;
            const float d0 = truth[static_cast<size_t>(idx)];
            const float dR = truth[static_cast<size_t>(idx) + 1];
            const float dD = truth[static_cast<size_t>(idx) + kImageWidth];
            if (d0 == kInvalidDepth || dR == kInvalidDepth || dD == kInvalidDepth) continue;   // sky-adjacent: skip

            const float depth_grad = std::max(std::fabs(dR - d0), std::fabs(dD - d0));
            const float rgb_grad = std::max(max_channel_diff_host(rgbf, idx, idx + 1),
                                            max_channel_diff_host(rgbf, idx, idx + kImageWidth));

            const bool is_depth_edge = depth_grad > kDepthEdgeThreshM;
            const bool is_depth_flat = depth_grad < kDepthFlatThreshM;
            const bool is_rgb_strong = rgb_grad > kRgbEdgeThreshHigh;
            const bool is_rgb_weak   = rgb_grad < kRgbEdgeThreshLow;

            if (is_depth_edge) m.boundary[static_cast<size_t>(idx)] = 1;
            if (is_depth_flat && is_rgb_strong) m.texture_trap[static_cast<size_t>(idx)] = 1;
            if (is_depth_edge && is_rgb_weak)   m.camo_edge[static_cast<size_t>(idx)] = 1;
            if (is_depth_flat && !is_rgb_strong) m.flat_interior[static_cast<size_t>(idx)] = 1;
        }
    }
    for (int i = 0; i < kImagePixels; ++i)
        m.boundary_clean[static_cast<size_t>(i)] =
            m.boundary[static_cast<size_t>(i)] && !m.camo_edge[static_cast<size_t>(i)];
    return m;
}

static int mask_count(const std::vector<uint8_t>& m)
{
    int c = 0; for (uint8_t v : m) c += v; return c;
}

// sparse_mean — mean depth over the VALID sparse samples, the diffusion
// PDE's initial condition for every pixel with no LiDAR return (see
// launch_diffusion's doc-comment in kernels.cuh for why this beats an
// out-of-range sentinel). Falls back to kMaxDepthM/2 in the degenerate
// case of zero samples (never hit by the committed data, but an honest,
// clearly-labeled fallback beats an uninitialized read).
static float sparse_mean(const std::vector<float>& sparse)
{
    double sum = 0.0; int n = 0;
    for (float v : sparse) if (v != kInvalidDepth) { sum += v; n++; }
    return n > 0 ? static_cast<float>(sum / n) : (kMaxDepthM * 0.5f);
}

// subsample — deterministic index-stride selection (keep every `stride`-th
// point), used both to derive the main demo's default LiDAR density from
// the committed full-density file and to drive the density sweep. No RNG
// involved: same stride, same machine-independent result every time.
static std::vector<LidarPointF> subsample(const std::vector<LidarPointF>& pts, int stride)
{
    std::vector<LidarPointF> out;
    out.reserve(pts.size() / static_cast<size_t>(stride) + 1);
    for (size_t i = 0; i < pts.size(); i += static_cast<size_t>(stride)) out.push_back(pts[i]);
    return out;
}

// ---------------------------------------------------------------------------
// run_pipeline_gpu — the four GPU stages, end to end, for one LiDAR point
// set: project+z-buffer -> decode -> diffusion (guided) -> IDW (baseline).
// Returns the sparse depth map plus both densified fields, all HOST arrays.
// This is the function main() calls both for the primary demo run and for
// each point in the density sweep — one place, one pipeline, no drift.
// ---------------------------------------------------------------------------
struct PipelineResult {
    std::vector<float> sparse;    // [kImagePixels], kInvalidDepth where no LiDAR return landed
    std::vector<float> guided;    // [kImagePixels], anisotropic-diffusion densified
    std::vector<float> idw;       // [kImagePixels], IDW-baseline densified
};

static PipelineResult run_pipeline_gpu(const std::vector<LidarPointF>& pts,
                                       const float* d_rgb)
{
    PipelineResult r;
    r.sparse.resize(static_cast<size_t>(kImagePixels));
    r.guided.resize(static_cast<size_t>(kImagePixels));
    r.idw.resize(static_cast<size_t>(kImagePixels));

    const size_t bytes = static_cast<size_t>(kImagePixels) * sizeof(float);

    LidarPointF* d_pts = nullptr;
    uint32_t* d_encoded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pts, pts.size() * sizeof(LidarPointF)));
    CUDA_CHECK(cudaMalloc(&d_encoded, static_cast<size_t>(kImagePixels) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(LidarPointF), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_encoded, 0xFF, static_cast<size_t>(kImagePixels) * sizeof(uint32_t)));   // 0xFFFFFFFF sentinel

    launch_project_zbuffer(d_pts, static_cast<int>(pts.size()), d_encoded);

    std::vector<uint32_t> encoded(static_cast<size_t>(kImagePixels));
    CUDA_CHECK(cudaMemcpy(encoded.data(), d_encoded, encoded.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kImagePixels; ++i) {
        const uint32_t bits = encoded[static_cast<size_t>(i)];
        float decoded;
        std::memcpy(&decoded, &bits, sizeof(decoded));   // portable bit-reinterpret, no strict-aliasing UB
        r.sparse[static_cast<size_t>(i)] = (bits == 0xFFFFFFFFu) ? kInvalidDepth : decoded;
    }

    float *d_sparse = nullptr, *d_guided = nullptr, *d_idw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sparse, bytes));
    CUDA_CHECK(cudaMalloc(&d_guided, bytes));
    CUDA_CHECK(cudaMalloc(&d_idw, bytes));
    CUDA_CHECK(cudaMemcpy(d_sparse, r.sparse.data(), bytes, cudaMemcpyHostToDevice));

    launch_diffusion(d_sparse, d_rgb, sparse_mean(r.sparse), d_guided);
    launch_idw(d_sparse, d_idw);

    CUDA_CHECK(cudaMemcpy(r.guided.data(), d_guided, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.idw.data(), d_idw, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_encoded));
    CUDA_CHECK(cudaFree(d_sparse));
    CUDA_CHECK(cudaFree(d_guided));
    CUDA_CHECK(cudaFree(d_idw));
    return r;
}

// max_abs_diff — the L-infinity comparison every VERIFY block below uses,
// SKIPPING pixels where either side reports kInvalidDepth (an "empty" pixel
// disagreement is a structural bug, checked separately and more loudly,
// not folded into the same float tolerance as a numeric rounding drift).
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b, int* mismatched_empty)
{
    float worst = 0.0f;
    int mism = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool ea = (a[i] == kInvalidDepth), eb = (b[i] == kInvalidDepth);
        if (ea != eb) { mism++; continue; }
        if (ea && eb) continue;
        const float d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    if (mismatched_empty) *mismatched_empty = mism;
    return worst;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_dir;   // optional CLI override for data/sample/
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data-dir") && i + 1 < argc) data_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data-dir DIR]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] depth completion: sparse LiDAR + RGB -> dense depth (project 01.18)\n");
    print_device_info();

    // ---- 1) load data -------------------------------------------------------
    const std::string rgb_path    = find_data_file(data_dir, argv[0], "rgb.ppm");
    const std::string depth_path  = find_data_file(data_dir, argv[0], "truth_depth.bin");
    const std::string lidar_path  = find_data_file(data_dir, argv[0], "lidar_points.csv");
    if (rgb_path.empty() || depth_path.empty() || lidar_path.empty()) {
        std::printf("PROBLEM: sample data not found under data/sample/ (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }

    std::vector<uint8_t> rgb;
    std::vector<float> truth;
    std::vector<LidarPointF> lidar_full;
    if (!read_ppm(rgb_path, rgb) || !read_depth_bin(depth_path, truth) || !read_lidar_csv(lidar_path, lidar_full)) {
        std::printf("PROBLEM: sample data malformed — see paths above\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }

    // PLANAR normalized-[0,1] color guidance image: rgbf[0..N) = red plane,
    // rgbf[N..2N) = green, rgbf[2N..3N) = blue (N=kImagePixels). Conductance
    // uses full COLOR, not grayscale luminance (kernels.cuh's
    // compute_conductance_kernel doc-comment explains why: two surfaces
    // that differ strongly in hue can land at nearly the same luminance,
    // which would blind a grayscale-only conductance to a real edge).
    std::vector<float> rgbf(static_cast<size_t>(3 * kImagePixels));
    for (int i = 0; i < kImagePixels; ++i) {
        const uint8_t* px = &rgb[static_cast<size_t>(i) * 3];
        rgbf[static_cast<size_t>(i)]                     = px[0] / 255.0f;
        rgbf[static_cast<size_t>(i) + kImagePixels]      = px[1] / 255.0f;
        rgbf[static_cast<size_t>(i) + 2 * kImagePixels]  = px[2] / 255.0f;
    }

    // Main demo density: stride-2 subsample of the committed full set (see
    // scripts/make_synthetic.py's console report for the measured
    // percentages this lands on).
    const std::vector<LidarPointF> lidar_main = subsample(lidar_full, 2);

    const int n_valid_truth = static_cast<int>(std::count_if(
        truth.begin(), truth.end(), [](float d) { return d != kInvalidDepth; }));

    std::printf("PROBLEM: %dx%d image, %d LiDAR beams x subsampled azimuths -> %zu returns "
               "(%.1f%% of pixels have scene truth), diffusion %d iters @ dt=%.2f, IDW radius %d px\n",
               kImageWidth, kImageHeight, 16, lidar_main.size(),
               100.0 * n_valid_truth / kImagePixels, kDiffusionIters,
               static_cast<double>(kDiffusionDt), kIdwRadiusPx);

    // ---- 2) VERIFY STAGE: GPU vs CPU, all four stages ------------------------
    bool verify_pass = true;

    float* d_rgbf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgbf, static_cast<size_t>(3 * kImagePixels) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rgbf, rgbf.data(), static_cast<size_t>(3 * kImagePixels) * sizeof(float), cudaMemcpyHostToDevice));

    LidarPointF* d_pts_main = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pts_main, lidar_main.size() * sizeof(LidarPointF)));
    CUDA_CHECK(cudaMemcpy(d_pts_main, lidar_main.data(), lidar_main.size() * sizeof(LidarPointF), cudaMemcpyHostToDevice));

    // -- 2a) projection + z-buffer --------------------------------------------
    std::vector<float> sparse_gpu, sparse_cpu;
    {
        uint32_t* d_encoded = nullptr;
        CUDA_CHECK(cudaMalloc(&d_encoded, static_cast<size_t>(kImagePixels) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_encoded, 0xFF, static_cast<size_t>(kImagePixels) * sizeof(uint32_t)));

        GpuTimer gt; gt.begin();
        launch_project_zbuffer(d_pts_main, static_cast<int>(lidar_main.size()), d_encoded);
        const float gpu_ms = gt.end_ms();

        std::vector<uint32_t> encoded(static_cast<size_t>(kImagePixels));
        CUDA_CHECK(cudaMemcpy(encoded.data(), d_encoded, encoded.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        sparse_gpu.resize(static_cast<size_t>(kImagePixels));
        for (int i = 0; i < kImagePixels; ++i) {
            const uint32_t bits = encoded[static_cast<size_t>(i)];
            float decoded;
            std::memcpy(&decoded, &bits, sizeof(decoded));
            sparse_gpu[static_cast<size_t>(i)] = (bits == 0xFFFFFFFFu) ? kInvalidDepth : decoded;
        }
        CUDA_CHECK(cudaFree(d_encoded));

        sparse_cpu.resize(static_cast<size_t>(kImagePixels));
        CpuTimer ct; ct.begin();
        project_zbuffer_cpu(lidar_main.data(), static_cast<int>(lidar_main.size()), sparse_cpu.data());
        const double cpu_ms = ct.end_ms();

        int mism = 0;
        const float worst = max_abs_diff(sparse_gpu, sparse_cpu, &mism);
        const bool ok = (worst <= 1e-4f) && (mism == 0);
        verify_pass = verify_pass && ok;
        // Numbers live on "[info]"/"[time]" lines only (NOT diffed): a
        // measured GPU-vs-CPU deviation can, in principle, vary by GPU
        // architecture (FMA contraction, atomic ordering) even when the
        // verdict does not — embedding it in the "VERIFY:" line itself
        // would make demo/expected_output.txt fragile across machines
        // (the same discipline 08.01's main.cu uses; see its VERIFY line).
        std::printf("[time] projection+z-buffer: CPU %.3f ms | GPU %.3f ms\n", cpu_ms, static_cast<double>(gpu_ms));
        std::printf("[info] projection+z-buffer: max |gpu-cpu| depth = %.3e m, %d empty-pixel mismatches\n",
                    static_cast<double>(worst), mism);
        std::printf("VERIFY: projection+z-buffer %s (GPU z-buffer matches CPU reference within tol 1e-4 m, zero empty-pixel mismatches)\n",
                    ok ? "PASS" : "FAIL");
    }

    // -- 2b) conductance --------------------------------------------------------
    {
        float *d_g_right = nullptr, *d_g_down = nullptr;
        const size_t bytes = static_cast<size_t>(kImagePixels) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_g_right, bytes));
        CUDA_CHECK(cudaMalloc(&d_g_down, bytes));

        GpuTimer gt; gt.begin();
        launch_compute_conductance(d_rgbf, d_g_right, d_g_down);
        const float gpu_ms = gt.end_ms();

        std::vector<float> gr_gpu(static_cast<size_t>(kImagePixels)), gd_gpu(static_cast<size_t>(kImagePixels));
        CUDA_CHECK(cudaMemcpy(gr_gpu.data(), d_g_right, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gd_gpu.data(), d_g_down, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_g_right));
        CUDA_CHECK(cudaFree(d_g_down));

        std::vector<float> gr_cpu(static_cast<size_t>(kImagePixels)), gd_cpu(static_cast<size_t>(kImagePixels));
        CpuTimer ct; ct.begin();
        compute_conductance_cpu(rgbf.data(), gr_cpu.data(), gd_cpu.data());
        const double cpu_ms = ct.end_ms();

        float worst = 0.0f;
        for (int i = 0; i < kImagePixels; ++i) {
            worst = std::max(worst, std::fabs(gr_gpu[static_cast<size_t>(i)] - gr_cpu[static_cast<size_t>(i)]));
            worst = std::max(worst, std::fabs(gd_gpu[static_cast<size_t>(i)] - gd_cpu[static_cast<size_t>(i)]));
        }
        const bool ok = worst <= 1e-5f;
        verify_pass = verify_pass && ok;
        std::printf("[time] conductance: CPU %.3f ms | GPU %.3f ms\n", cpu_ms, static_cast<double>(gpu_ms));
        std::printf("[info] conductance: max |gpu-cpu| = %.3e\n", static_cast<double>(worst));
        std::printf("VERIFY: conductance %s (GPU conductance matches CPU reference within tol 1e-5)\n", ok ? "PASS" : "FAIL");
    }

    // -- 2c) diffusion (full kDiffusionIters on both paths) ---------------------
    std::vector<float> guided_gpu, guided_cpu;
    {
        float* d_sparse = nullptr;
        float* d_guided = nullptr;
        const size_t bytes = static_cast<size_t>(kImagePixels) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_sparse, bytes));
        CUDA_CHECK(cudaMalloc(&d_guided, bytes));
        CUDA_CHECK(cudaMemcpy(d_sparse, sparse_gpu.data(), bytes, cudaMemcpyHostToDevice));
        const float unknown_seed = sparse_mean(sparse_gpu);

        GpuTimer gt; gt.begin();
        launch_diffusion(d_sparse, d_rgbf, unknown_seed, d_guided);
        const float gpu_ms = gt.end_ms();

        guided_gpu.resize(static_cast<size_t>(kImagePixels));
        CUDA_CHECK(cudaMemcpy(guided_gpu.data(), d_guided, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_sparse));
        CUDA_CHECK(cudaFree(d_guided));

        guided_cpu.resize(static_cast<size_t>(kImagePixels));
        CpuTimer ct; ct.begin();
        diffusion_densify_cpu(sparse_gpu.data(), rgbf.data(), unknown_seed, guided_cpu.data());
        const double cpu_ms = ct.end_ms();

        int mism = 0;
        const float worst = max_abs_diff(guided_gpu, guided_cpu, &mism);
        // Looser tolerance than the single-shot stages: kDiffusionIters
        // independent forward-Euler steps compound FP rounding differences
        // between the two code paths iteration over iteration (THEORY.md
        // "Numerical considerations" measures and justifies this bound).
        const bool ok = worst <= 5e-2f;
        verify_pass = verify_pass && ok;
        std::printf("[time] diffusion (%d iters): CPU %.2f ms | GPU %.3f ms | speed-up %.0fx (teaching artifact)\n",
                    kDiffusionIters, cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        std::printf("[info] diffusion: max |gpu-cpu| = %.3e m after %d iters\n", static_cast<double>(worst), kDiffusionIters);
        std::printf("VERIFY: diffusion %s (GPU diffusion matches CPU reference within tol 5e-2 m over the full iteration count)\n",
                    ok ? "PASS" : "FAIL");
    }

    // -- 2d) IDW ------------------------------------------------------------------
    std::vector<float> idw_gpu, idw_cpu;
    {
        float* d_sparse = nullptr;
        float* d_idw = nullptr;
        const size_t bytes = static_cast<size_t>(kImagePixels) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_sparse, bytes));
        CUDA_CHECK(cudaMalloc(&d_idw, bytes));
        CUDA_CHECK(cudaMemcpy(d_sparse, sparse_gpu.data(), bytes, cudaMemcpyHostToDevice));

        GpuTimer gt; gt.begin();
        launch_idw(d_sparse, d_idw);
        const float gpu_ms = gt.end_ms();

        idw_gpu.resize(static_cast<size_t>(kImagePixels));
        CUDA_CHECK(cudaMemcpy(idw_gpu.data(), d_idw, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_sparse));
        CUDA_CHECK(cudaFree(d_idw));

        idw_cpu.resize(static_cast<size_t>(kImagePixels));
        CpuTimer ct; ct.begin();
        idw_densify_cpu(sparse_gpu.data(), idw_cpu.data());
        const double cpu_ms = ct.end_ms();

        int mism = 0;
        const float worst = max_abs_diff(idw_gpu, idw_cpu, &mism);
        const bool ok = worst <= 1e-3f;
        verify_pass = verify_pass && ok;
        std::printf("[time] IDW baseline: CPU %.2f ms | GPU %.3f ms | speed-up %.0fx (teaching artifact)\n",
                    cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        std::printf("[info] IDW baseline: max |gpu-cpu| = %.3e m\n", static_cast<double>(worst));
        std::printf("VERIFY: IDW baseline %s (GPU IDW matches CPU reference within tol 1e-3 m)\n", ok ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_pts_main));

    // The forward-Euler stability bound is enforced at COMPILE TIME by
    // kernels.cuh's static_assert (this line would not even build if it
    // failed) — printed here so the demo's stable output records the fact
    // that the check exists and what the numbers are.
    std::printf("STABILITY: dt=%.2f <= CFL-style bound 1/4=0.25 (worst case 4 neighbors, "
               "conductance <= 1.0 each) -> PASS (enforced at compile time, kernels.cuh)\n",
               static_cast<double>(kDiffusionDt));

    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement in the VERIFY stage — see VERIFY lines above)\n");
        return 1;
    }

    // ---- 3) EVALUATION GATES (guided/idw vs. the synthetic scene's truth) ----
    const EdgeMasks masks = build_edge_masks(truth, rgbf);
    const int n_boundary       = mask_count(masks.boundary);
    const int n_trap           = mask_count(masks.texture_trap);
    const int n_camo           = mask_count(masks.camo_edge);
    const int n_boundary_clean = mask_count(masks.boundary_clean);
    const int n_flat           = mask_count(masks.flat_interior);

    const ErrorStats overall_guided = compute_error(guided_gpu, truth);
    const ErrorStats overall_idw    = compute_error(idw_gpu, truth);
    const ErrorStats edge_guided    = compute_error(guided_gpu, truth, &masks.boundary_clean);
    const ErrorStats edge_idw       = compute_error(idw_gpu, truth, &masks.boundary_clean);
    const ErrorStats trap_guided    = compute_error(guided_gpu, truth, &masks.texture_trap);
    const ErrorStats trap_idw       = compute_error(idw_gpu, truth, &masks.texture_trap);
    const ErrorStats camo_guided    = compute_error(guided_gpu, truth, &masks.camo_edge);
    const ErrorStats flat_guided    = compute_error(guided_gpu, truth, &masks.flat_interior);

    std::printf("[info] region pixel counts (of %d interior pixels checked): boundary=%d "
               "boundary_clean(excl. camo)=%d texture_trap=%d camo_edge=%d flat_interior=%d\n",
               (kImageWidth - 1) * (kImageHeight - 1), n_boundary, n_boundary_clean, n_trap, n_camo, n_flat);
    std::printf("[info] overall accuracy: guided MAE=%.4f m RMSE=%.4f m (n=%d) | IDW MAE=%.4f m RMSE=%.4f m (n=%d)\n",
               overall_guided.mae, overall_guided.rmse, overall_guided.n,
               overall_idw.mae, overall_idw.rmse, overall_idw.n);

    bool gates_pass = true;

    // GATE: overall_accuracy — guided must be a genuinely usable dense
    // reconstruction (not merely "beats IDW"): bounded absolute error.
    // kOverallMaeBoundM is set from a MEASURED run (README/THEORY document
    // the number) with headroom, per this project's brief: never an
    // arbitrary target chosen before looking at real data.
    static constexpr double kOverallMaeBoundM = 1.3;
    {
        const bool ok = overall_guided.mae < kOverallMaeBoundM && overall_guided.n > 0;
        gates_pass = gates_pass && ok;
        std::printf("GATE: overall_accuracy %s (guided MAE < bound %.1f m over all valid-truth pixels)\n",
                   ok ? "PASS" : "FAIL", kOverallMaeBoundM);
    }

    // GATE: edge_quality — the reason-to-exist gate. On CLEAN boundaries
    // (real depth edges the RGB image also shows), the edge-aware method
    // must beat the RGB-blind IDW baseline by a MEASURED, then margined,
    // factor (README/THEORY document the measured ratio this threshold was
    // set below).
    static constexpr double kEdgeQualityMinRatio = 1.15;   // guided MAE must be <= idw MAE / this ratio (measured ~1.27x, README/THEORY)
    {
        const bool ok = edge_idw.n > 0 && edge_guided.n > 0 &&
                        edge_guided.mae <= edge_idw.mae / kEdgeQualityMinRatio;
        gates_pass = gates_pass && ok;
        std::printf("[info] edge_quality (clean boundaries, n=%d): guided MAE=%.4f m | IDW MAE=%.4f m | ratio idw/guided=%.2fx\n",
                   edge_guided.n, edge_guided.mae, edge_idw.mae,
                   edge_guided.mae > 0.0 ? edge_idw.mae / edge_guided.mae : 0.0);
        std::printf("GATE: edge_quality %s (guided beats IDW by >= %.1fx at clean depth boundaries)\n",
                   ok ? "PASS" : "FAIL", kEdgeQualityMinRatio);
    }

    // GATE: texture_trap — on the flat, high-contrast checkerboard patch,
    // conductance gating must NOT hallucinate depth structure: guided error
    // there stays close to IDW's (which is RGB-blind and therefore cannot,
    // by construction, be fooled by texture) — a bounded-degradation ratio.
    static constexpr double kTextureTrapMaxRatio = 1.8;   // guided RMSE <= idw RMSE * this ratio (measured ~1.36x, README/THEORY)
    {
        const bool ok = trap_idw.n > 0 && trap_guided.rmse <= trap_idw.rmse * kTextureTrapMaxRatio;
        gates_pass = gates_pass && ok;
        std::printf("[info] texture_trap (n=%d): guided RMSE=%.4f m | IDW RMSE=%.4f m | ratio guided/idw=%.2fx\n",
                   trap_guided.n, trap_guided.rmse, trap_idw.rmse,
                   trap_idw.rmse > 0.0 ? trap_guided.rmse / trap_idw.rmse : 0.0);
        std::printf("GATE: texture_trap %s (guided RMSE <= %.1fx the RGB-blind IDW baseline in the checkerboard patch)\n",
                   ok ? "PASS" : "FAIL", kTextureTrapMaxRatio);
    }

    // GATE: camo_edge honesty — NOT a "guided is good here" claim. The
    // low-contrast depth edge is EXACTLY where "RGB edge implies depth
    // edge" is false, and the demo's job is to PROVE it demonstrates that
    // failure: error at the camo edge must measurably EXCEED the ordinary
    // flat-interior error, by a stated factor.
    static constexpr double kCamoHonestyMinRatio = 2.0;   // camo MAE must be >= flat MAE * this ratio (measured ~3.6x, README/THEORY)
    {
        const bool ok = n_camo > 0 && flat_guided.mae > 0.0 &&
                        camo_guided.mae >= flat_guided.mae * kCamoHonestyMinRatio;
        gates_pass = gates_pass && ok;
        std::printf("[info] camo_edge honesty (n=%d): guided MAE=%.4f m | flat-interior guided MAE=%.4f m | ratio=%.2fx\n",
                   n_camo, camo_guided.mae, flat_guided.mae,
                   flat_guided.mae > 0.0 ? camo_guided.mae / flat_guided.mae : 0.0);
        std::printf("GATE: camo_edge_honesty %s (camo-edge error >= %.1fx flat-interior error — "
                   "the demo must SHOW the prior's failure mode, not hide it)\n",
                   ok ? "PASS" : "FAIL", kCamoHonestyMinRatio);
    }

    // GATE: input_fidelity — Dirichlet anchoring actually holds: at every
    // pixel that HAS a LiDAR sample, both densified fields must reproduce
    // it (near-)exactly, never overwrite it with diffused/interpolated noise.
    {
        float worst_guided = 0.0f, worst_idw = 0.0f;
        int n_anchors = 0;
        for (int i = 0; i < kImagePixels; ++i) {
            if (sparse_gpu[static_cast<size_t>(i)] == kInvalidDepth) continue;
            n_anchors++;
            worst_guided = std::max(worst_guided, std::fabs(guided_gpu[static_cast<size_t>(i)] - sparse_gpu[static_cast<size_t>(i)]));
            worst_idw    = std::max(worst_idw,    std::fabs(idw_gpu[static_cast<size_t>(i)]    - sparse_gpu[static_cast<size_t>(i)]));
        }
        const bool ok = n_anchors > 0 && worst_guided <= 1e-3f && worst_idw <= 1e-3f;
        gates_pass = gates_pass && ok;
        std::printf("[info] input_fidelity: n_anchors=%d, max deviation guided=%.2e m idw=%.2e m\n",
                   n_anchors, static_cast<double>(worst_guided), static_cast<double>(worst_idw));
        std::printf("GATE: input_fidelity %s (both densified fields reproduce every Dirichlet-anchored pixel within tol 1e-3 m)\n",
                   ok ? "PASS" : "FAIL");
    }

    // ---- 4) DENSITY SWEEP ([info] only — not part of the RESULT verdict) -----
    {
        std::printf("[info] density sweep (guided method; strides chosen so the sweep spans roughly a "
                   "2%%-7%% range — see data/README.md for the exact measured densities):\n");
        double prev_mae = -1.0;
        bool monotone = true;
        for (int stride : {5, 2, 1}) {
            const std::vector<LidarPointF> sub = subsample(lidar_full, stride);
            const PipelineResult pr = run_pipeline_gpu(sub, d_rgbf);
            const ErrorStats es = compute_error(pr.guided, truth);
            std::printf("[info]   stride=%d (%zu pts): guided MAE=%.4f m RMSE=%.4f m\n",
                       stride, sub.size(), es.mae, es.rmse);
            if (prev_mae >= 0.0 && es.mae > prev_mae + 1e-9) monotone = false;
            prev_mae = es.mae;
        }
        std::printf("[info] density sweep monotonicity (MAE should not increase as density rises): %s\n",
                   monotone ? "MONOTONE" : "NON-MONOTONE (see [info] lines above)");
    }

    CUDA_CHECK(cudaFree(d_rgbf));

    // ---- 5) ARTIFACTS ----------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifacts_ok = true;

    {
        std::ofstream f(out_dir + "/rgb.ppm", std::ios::binary);
        artifacts_ok = artifacts_ok && f.is_open();
        if (f.is_open()) {
            f << "P6\n" << kImageWidth << " " << kImageHeight << "\n255\n";
            f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
        }
    }
    write_pgm(out_dir + "/sparse_depth_vis.pgm", dilate_for_visibility(depth_to_gray(sparse_gpu, 2.0f, 20.0f), 2));
    write_pgm(out_dir + "/completed_guided.pgm", depth_to_gray(guided_gpu, 2.0f, 20.0f));
    write_pgm(out_dir + "/completed_idw.pgm", depth_to_gray(idw_gpu, 2.0f, 20.0f));
    write_pgm(out_dir + "/truth_depth.pgm", depth_to_gray(truth, 2.0f, 20.0f));
    write_pgm(out_dir + "/error_guided.pgm", error_to_gray(guided_gpu, truth));
    write_pgm(out_dir + "/error_idw.pgm", error_to_gray(idw_gpu, truth));

    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        artifacts_ok = artifacts_ok && f.is_open();
        if (f.is_open()) {
            f << "gate,metric,value,unit\n";
            f << "overall_accuracy,guided_mae," << overall_guided.mae << ",m\n";
            f << "overall_accuracy,guided_rmse," << overall_guided.rmse << ",m\n";
            f << "overall_accuracy,idw_mae," << overall_idw.mae << ",m\n";
            f << "overall_accuracy,idw_rmse," << overall_idw.rmse << ",m\n";
            f << "edge_quality,guided_mae," << edge_guided.mae << ",m\n";
            f << "edge_quality,idw_mae," << edge_idw.mae << ",m\n";
            f << "texture_trap,guided_rmse," << trap_guided.rmse << ",m\n";
            f << "texture_trap,idw_rmse," << trap_idw.rmse << ",m\n";
            f << "camo_edge,guided_mae," << camo_guided.mae << ",m\n";
            f << "camo_edge,flat_interior_guided_mae," << flat_guided.mae << ",m\n";
        }
    }

    if (artifacts_ok)
        std::printf("ARTIFACT: wrote rgb.ppm, sparse_depth_vis.pgm, completed_guided.pgm, completed_idw.pgm, "
                   "truth_depth.pgm, error_guided.pgm, error_idw.pgm, gates_metrics.csv to demo/out/\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more files to demo/out/\n");

    // ---- 6) verdict --------------------------------------------------------
    const bool success = verify_pass && gates_pass && artifacts_ok;
    if (success)
        std::printf("RESULT: PASS (all VERIFY twins agree and all evaluation gates pass — see GATE lines above)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for the failing check)\n");
    return success ? 0 : 1;
}
