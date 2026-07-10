// ===========================================================================
// main.cu — entry point for project 30.01
//           Agriculture, Milestone 1: fruit detection + 3-D localization +
//           ripeness on synthetic orchard RGB-D imagery
//           (BUNDLED PROJECT: see README "Overview" for the six other
//           milestones this catalog bullet names, documented but not
//           implemented here — CLAUDE.md section 2)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Load the committed scene: data/sample/rgb.ppm (640x480 RGB),
//      data/sample/depth.pgm (640x480, 16-bit millimeters), and
//      data/sample/ground_truth.csv (per-fruit exact 3-D truth — used ONLY
//      in the verification stage below, never by the pipeline itself).
//   2. Run the SEVEN-STAGE pipeline (kernels.cuh's file header) TWICE: once
//      on the GPU (kernels.cu) and once on the CPU (reference_cpu.cpp).
//   3. VERIFY: compare the two paths stage by stage — HSV/mask by
//      tolerance, connected-component LABELS by EXACT equality (the
//      argument for why is kernels.cuh's file header), final per-fruit
//      statistics by tolerance.
//   4. GROUND-TRUTH GATES: match the GPU pipeline's detections against the
//      synthetic scene's exact fruit list and check detection rate, false
//      positives, 3-D localization error, radius error, and ripeness rank
//      correlation — each against a threshold DERIVED and DOCUMENTED in
//      THEORY.md, with the actually-measured numbers printed alongside.
//   5. ARTIFACTS: demo/out/detections.pgm (the RGB frame, grayscale, with a
//      ring burned in at every detected fruit) and demo/out/fruit_map.csv
//      (id, 3-D center, radius, ripeness — the seed of the documented
//      Milestone 7 "yield mapping" component; see README).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "DATA:", "VERIFY:",
// "DETECT:", "LOCALIZE:", "RIPENESS:", "ARTIFACT:", "RESULT:" — "[info]"/
// "[time]" lines are NOT diffed (CLAUDE.md's usual convention; see 08.01).
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Verification / gate thresholds — collected here, once, so every number
// this program checks against is visible in one place (mirrors 08.01's
// kernels.cuh constant block; these live in main.cu because they are
// VERIFICATION policy, not pipeline definition — THEORY.md derives each one
// and states the actually-measured margin against it).
// ---------------------------------------------------------------------------
static const float kHsvTol       = 1e-3f;   // max |gpu-cpu| on h(deg)/s/v — plain arithmetic, near-ULP
static const float kMatchGateM   = 0.15f;   // max 3-D distance (m) to accept a GT<->detection match
static const float kLocErrMaxM   = 0.015f;  // 3-D localization error gate (m) — THEORY.md derives the budget
static const float kRadiusErrMaxM = 0.006f; // radius error gate (m)
static const float kMinDetectionRate = 0.80f; // fraction of DETECTABLE (visible_frac>0) GT fruit that must be found
static const int   kMaxFalsePositives = 2;    // detections that match no GT fruit (see README/THEORY: exactly the
                                              // scene's two designed cross-depth merge cases, not sensor noise)
static const double kMinRipenessRankCorr = 0.70; // Spearman rank correlation, matched detectable fruit

// ---------------------------------------------------------------------------
// PNM (PPM/PGM) loaders — the repo's standard image format (01.02 uses the
// 8-bit PGM sibling of these; this project ADDS a 16-bit PGM depth reader
// and an 8-bit color PPM reader, both documented in scripts/make_synthetic.py).
// ---------------------------------------------------------------------------

// skip_ws_and_comments — PNM headers may contain '#' comment lines anywhere
// between the whitespace-separated header tokens (magic/width/height/
// maxval); this generator does not emit any, but a correct reader should
// not choke if one appears (e.g., a learner hand-edits a file to test it).
static void skip_ws_and_comments(std::ifstream& f)
{
    for (;;) {
        const int c = f.peek();
        if (c == '#') { std::string line; std::getline(f, line); continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { f.get(); continue; }
        break;
    }
}

static bool read_pnm_header(std::ifstream& f, std::string& magic, int& w, int& h, int& maxval)
{
    f >> magic;
    skip_ws_and_comments(f); f >> w;
    skip_ws_and_comments(f); f >> h;
    skip_ws_and_comments(f); f >> maxval;
    f.get();   // the single mandatory whitespace byte separating header from binary data (PNM spec)
    return f.good();
}

// read_ppm — 8-bit binary color PPM (P6). Fills rgb as [H*W*3] interleaved
// bytes (kernels.cuh's RGB layout). Returns false (with a stderr message) on
// any format mismatch — a strict loader, so a malformed/foreign file fails
// LOUDLY rather than silently misinterpreting bytes as pixels.
static bool read_ppm(const std::string& path, int expect_w, int expect_h, std::vector<unsigned char>& rgb)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "read_ppm: cannot open %s\n", path.c_str()); return false; }
    std::string magic; int w = 0, h = 0, maxval = 0;
    if (!read_pnm_header(f, magic, w, h, maxval)) { std::fprintf(stderr, "read_ppm: bad header in %s\n", path.c_str()); return false; }
    if (magic != "P6" || maxval != 255 || w != expect_w || h != expect_h) {
        std::fprintf(stderr, "read_ppm: %s is not a %dx%d P6/255 PPM (got %s %dx%d maxval=%d)\n",
                     path.c_str(), expect_w, expect_h, magic.c_str(), w, h, maxval);
        return false;
    }
    rgb.resize(static_cast<size_t>(w) * h * 3);
    f.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return f.good() || f.eof();   // eof right after the last byte is the expected, successful case
}

// read_pgm16 — 16-bit binary gray PGM (P5, maxval 65535), BIG-ENDIAN samples
// (the NetPBM convention — scripts/make_synthetic.py writes it this way).
// Converts millimeters -> METERS as it reads (kernels.cuh's depth-image
// contract is float meters), so every downstream consumer sees SI units.
static bool read_pgm16(const std::string& path, int expect_w, int expect_h, std::vector<float>& depth_m)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "read_pgm16: cannot open %s\n", path.c_str()); return false; }
    std::string magic; int w = 0, h = 0, maxval = 0;
    if (!read_pnm_header(f, magic, w, h, maxval)) { std::fprintf(stderr, "read_pgm16: bad header in %s\n", path.c_str()); return false; }
    if (magic != "P5" || maxval != 65535 || w != expect_w || h != expect_h) {
        std::fprintf(stderr, "read_pgm16: %s is not a %dx%d P5/65535 PGM (got %s %dx%d maxval=%d)\n",
                     path.c_str(), expect_w, expect_h, magic.c_str(), w, h, maxval);
        return false;
    }
    std::vector<unsigned char> raw(static_cast<size_t>(w) * h * 2);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!(f.good() || f.eof())) return false;

    depth_m.resize(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < depth_m.size(); ++i) {
        const unsigned int mm = (static_cast<unsigned int>(raw[i * 2 + 0]) << 8)   // big-endian: MSB first
                               |  static_cast<unsigned int>(raw[i * 2 + 1]);
        depth_m[i] = static_cast<float>(mm) * 0.001f;   // mm -> m
    }
    return true;
}

// ---------------------------------------------------------------------------
// Ground truth — loaded ONLY for the verification gates below; the pipeline
// itself never reads this file (that would be cheating the benchmark).
// ---------------------------------------------------------------------------
struct GtFruit {
    int id = 0;
    float x_m = 0, y_m = 0, z_m = 0, radius_m = 0, ripeness = 0;
    int visible_px = 0;
    float ideal_px = 0, visible_frac = 0;
};

static bool load_ground_truth(const std::string& path, std::vector<GtFruit>& out)
{
    std::ifstream f(path);
    if (!f.is_open()) { std::fprintf(stderr, "load_ground_truth: cannot open %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cell;
        GtFruit g;
        auto next = [&](float& dst) { std::getline(ss, cell, ','); dst = std::strtof(cell.c_str(), nullptr); };
        std::getline(ss, cell, ','); g.id = std::atoi(cell.c_str());
        next(g.x_m); next(g.y_m); next(g.z_m); next(g.radius_m); next(g.ripeness);
        std::getline(ss, cell, ','); g.visible_px = std::atoi(cell.c_str());
        next(g.ideal_px); next(g.visible_frac);
        out.push_back(g);
    }
    return !out.empty();
}

// ---------------------------------------------------------------------------
// Path resolution — mirrors 08.01's find_scenario/project_root_from: try a
// handful of candidate locations relative to the executable and the CWD, so
// the demo works whether launched from run_demo.ps1, Visual Studio's
// debugger, or a learner's own shell.
// ---------------------------------------------------------------------------
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_data_file(const std::string& filename, const std::string& cli_dir, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_dir.empty()) candidates.push_back(cli_dir + "/" + filename);
    candidates.push_back(project_root_from(argv0) + "/data/sample/" + filename);
    candidates.push_back("data/sample/" + filename);
    candidates.push_back("../data/sample/" + filename);
    for (const auto& c : candidates)
        if (std::ifstream(c, std::ios::binary).is_open()) return c;
    return "";
}

static bool ensure_dir(const std::string& path)
{
#ifdef _WIN32
    const int r = _mkdir(path.c_str());
#else
    const int r = mkdir(path.c_str(), 0755);
#endif
    return r == 0 || errno == EEXIST;
}

// ---------------------------------------------------------------------------
// build_detections — turn the dense per-component arrays (Stage 5 of
// kernels.cuh) into the final FruitDetection list. SHARED between the GPU
// path (called on arrays copied back from the device) and the CPU path
// (called directly on reference_cpu.cpp's output) — both sides produce the
// exact same array layout, so this one function is the single place the
// "pixels -> fruit list" logic lives, and main.cu's later comparisons are
// comparing the OUTPUT of two pipelines through the SAME lens.
//
// Scans every pixel for CANONICAL ROOTS (mask[p] && label[p]==p — see
// kernels.cuh's file header for why this identifies exactly one pixel per
// connected component) and applies the kMinComponentPixels noise floor
// (THEORY.md measures what this filters out: the synthetic scene's
// deliberate false-positive glint specks that the morphological opening
// alone did not fully remove).
// ---------------------------------------------------------------------------
static std::vector<FruitDetection> build_detections(
    const std::vector<unsigned char>& mask, const std::vector<int>& label,
    const std::vector<int>& comp_count, const std::vector<int>& comp_sum_x, const std::vector<int>& comp_sum_y,
    const std::vector<int>& comp_min_x, const std::vector<int>& comp_max_x,
    const std::vector<int>& comp_min_y, const std::vector<int>& comp_max_y,
    const std::vector<float>& comp_sum_hue, const std::vector<float>& comp_final_depth,
    int W, int H)
{
    std::vector<FruitDetection> out;
    const int N = W * H;
    for (int p = 0; p < N; ++p) {
        if (!mask[p] || label[p] != p) continue;         // not a canonical root -> not a component
        const int count = comp_count[p];
        if (count < kMinComponentPixels) continue;        // residual noise floor (see comment above)

        FruitDetection d{};
        d.label = p;
        d.pixel_count = count;
        d.centroid_px_x = static_cast<float>(comp_sum_x[p]) / static_cast<float>(count);
        d.centroid_px_y = static_cast<float>(comp_sum_y[p]) / static_cast<float>(count);
        d.bbox_min_x = comp_min_x[p]; d.bbox_max_x = comp_max_x[p];
        d.bbox_min_y = comp_min_y[p]; d.bbox_max_y = comp_max_y[p];

        // AREA-based screen radius: a circle of N pixels has N = pi*r^2, so
        // r = sqrt(N/pi) — THEORY.md "The math" compares this to the
        // alternative bbox-extent estimator and explains why area is more
        // robust to the asymmetric silhouettes partial occlusion produces.
        d.radius_px = std::sqrt(static_cast<float>(count) / 3.14159265358979323846f);

        // comp_final_depth is the robust mean depth of the fruit's VISIBLE
        // SURFACE — not its center. A camera sees the NEAR hemisphere of a
        // sphere, whose depth varies from (Zc - r) at the pole facing the
        // camera to Zc at the silhouette's grazing edge; averaged uniformly
        // over the PROJECTED-AREA (i.e., per-pixel, exactly what this
        // pipeline's pixel-count-weighted mean computes), the visible
        // surface's mean depth sits a full (2/3)*radius NEARER than the
        // sphere's true center (THEORY.md "The math" derives this integral
        // in closed form: mean_h = (2/r^2)*INT[0,r] rho*sqrt(r^2-rho^2)drho
        // = (2/3)r). This is the DOMINANT term in this project's
        // localization error budget — larger than sensor noise or pixel
        // quantization by roughly an order of magnitude — so the pipeline
        // corrects for it explicitly rather than reporting a biased
        // "surface depth" as if it were the fruit's center.
        const float surface_depth_m = comp_final_depth[p];
        const float radius_m_at_surface_depth = d.radius_px * surface_depth_m / kFx;   // first-order radius (bias here is second-order, negligible)
        d.depth_m = surface_depth_m + (2.0f / 3.0f) * radius_m_at_surface_depth;        // CORRECTED to the sphere's center depth

        // Pinhole back-projection (similar triangles — THEORY.md derives
        // this from the SAME ray parametrization make_synthetic.py used to
        // RENDER the scene, so the pipeline is provably inverting the
        // forward model, not an independently-invented formula), evaluated
        // at the CORRECTED center depth above. The "+0.5" matches the
        // PIXEL-CENTER convention both the renderer and this centroid share
        // (a pixel's integer column x represents the ray through x+0.5).
        d.center_z_m = d.depth_m;
        d.center_x_m = (d.centroid_px_x + 0.5f - kCx) * d.depth_m / kFx;
        d.center_y_m = (d.centroid_px_y + 0.5f - kCy) * d.depth_m / kFy;
        d.radius_m = d.radius_px * d.depth_m / kFx;   // recomputed at the corrected depth, self-consistent

        d.mean_hue_deg = comp_sum_hue[p] / static_cast<float>(count);
        // Inverse of make_synthetic.py's hue = 120*(1-ripeness) — see
        // README/THEORY "ripeness-vs-color honesty" for what this DOES and
        // does NOT capture about real fruit ripeness.
        d.ripeness = std::max(0.0f, std::min(1.0f, (120.0f - d.mean_hue_deg) / 120.0f));

        out.push_back(d);
    }
    return out;
}

// ---------------------------------------------------------------------------
// spearman_rank_correlation — Spearman's rho between two equal-length
// vectors: rank-transform each (ties broken by encounter order — with
// floats accumulated from real pixel data, an exact tie is measure-zero
// likely and never observed on the committed scene), then Pearson-correlate
// the RANKS. Used for the ripeness gate (README/THEORY explain why rank
// correlation, not absolute-value agreement, is the honest metric here: the
// hue->ripeness color model is a simplification — THEORY.md "ripeness-vs-
// color honesty" — so what the pipeline can actually promise is getting the
// ORDER of ripeness right, not a calibrated absolute value).
// ---------------------------------------------------------------------------
static double spearman_rank_correlation(const std::vector<float>& a, const std::vector<float>& b)
{
    const size_t n = a.size();
    if (n < 2 || b.size() != n) return 0.0;

    std::vector<size_t> order_a(n), order_b(n);
    for (size_t i = 0; i < n; ++i) { order_a[i] = i; order_b[i] = i; }
    std::sort(order_a.begin(), order_a.end(), [&](size_t i, size_t j) { return a[i] < a[j]; });
    std::sort(order_b.begin(), order_b.end(), [&](size_t i, size_t j) { return b[i] < b[j]; });

    std::vector<double> rank_a(n), rank_b(n);
    for (size_t r = 0; r < n; ++r) { rank_a[order_a[r]] = static_cast<double>(r); rank_b[order_b[r]] = static_cast<double>(r); }

    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < n; ++i) { mean_a += rank_a[i]; mean_b += rank_b[i]; }
    mean_a /= static_cast<double>(n); mean_b /= static_cast<double>(n);

    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = rank_a[i] - mean_a, db = rank_b[i] - mean_b;
        cov += da * db; var_a += da * da; var_b += db * db;
    }
    if (var_a <= 0.0 || var_b <= 0.0) return 0.0;   // degenerate: every rank tied (cannot happen with n>=2 distinct ranks)
    return cov / std::sqrt(var_a * var_b);
}

// ---------------------------------------------------------------------------
// write_detections_pgm — the visual artifact: the RGB frame converted to
// grayscale (standard luma weights, matching how a monochrome preview would
// render it) with a RING burned in around every GPU detection's pixel
// centroid at its estimated screen radius, so a learner can eyeball hits
// and misses directly against the color image.
// ---------------------------------------------------------------------------
static bool write_detections_pgm(const std::string& path, const std::vector<unsigned char>& rgb,
                                 const std::vector<FruitDetection>& dets, int W, int H)
{
    std::vector<unsigned char> gray(static_cast<size_t>(W) * H);
    for (size_t i = 0; i < gray.size(); ++i) {
        // Standard luma (Rec. 601-ish weights) — a monochrome preview a
        // learner could get from any camera's Y channel.
        const float lum = 0.299f * rgb[i * 3 + 0] + 0.587f * rgb[i * 3 + 1] + 0.114f * rgb[i * 3 + 2];
        gray[i] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, lum)));
    }
    for (const auto& d : dets) {
        // Ring: sample enough angles to keep the ring visually continuous
        // (2*pi*r pixels of circumference -> that many samples, floor 24).
        const int steps = std::max(24, static_cast<int>(2.0f * 3.14159265f * d.radius_px));
        for (int s = 0; s < steps; ++s) {
            const float theta = (2.0f * 3.14159265f * s) / static_cast<float>(steps);
            const int px = static_cast<int>(std::lround(d.centroid_px_x + d.radius_px * std::cos(theta)));
            const int py = static_cast<int>(std::lround(d.centroid_px_y + d.radius_px * std::sin(theta)));
            if (px >= 0 && px < W && py >= 0 && py < H) gray[static_cast<size_t>(py) * W + px] = 255;
        }
        // A small dark cross at the centroid so it stands out against a
        // bright ripe-fruit blob (which is often already a light gray).
        const int cx = static_cast<int>(std::lround(d.centroid_px_x));
        const int cy = static_cast<int>(std::lround(d.centroid_px_y));
        for (int t = -2; t <= 2; ++t) {
            if (cx + t >= 0 && cx + t < W && cy >= 0 && cy < H) gray[static_cast<size_t>(cy) * W + (cx + t)] = 0;
            if (cx >= 0 && cx < W && cy + t >= 0 && cy + t < H) gray[static_cast<size_t>(cy + t) * W + cx] = 0;
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return f.good();
}

// write_fruit_map_csv — id, 3-D center, radius, ripeness: the "yield map"
// seed the README/THEORY milestone list points to (Milestone 7).
static bool write_fruit_map_csv(const std::string& path, const std::vector<FruitDetection>& dets)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "# fruit_index,center_x_m,center_y_m,center_z_m,radius_m,ripeness,pixel_count\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        f << (i + 1) << ',' << d.center_x_m << ',' << d.center_y_m << ',' << d.center_z_m << ','
          << d.radius_m << ',' << d.ripeness << ',' << d.pixel_count << '\n';
    }
    return f.good();
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data-dir") && i + 1 < argc) data_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data-dir path/to/data/sample]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Agriculture Milestone 1: fruit detection + 3-D localization + ripeness (project 30.01)\n");
    print_device_info();

    const int W = kImageWidth, H = kImageHeight, N = W * H;
    std::printf("PROBLEM: HSV mask -> morphological opening -> connected-component labeling -> "
               "per-component 3-D localization + ripeness, %dx%d RGB-D, FP32\n", W, H);

    // ---- load data ----------------------------------------------------------
    const std::string rgb_path = find_data_file("rgb.ppm", data_dir, argv[0]);
    const std::string depth_path = find_data_file("depth.pgm", data_dir, argv[0]);
    const std::string gt_path = find_data_file("ground_truth.csv", data_dir, argv[0]);
    if (rgb_path.empty() || depth_path.empty() || gt_path.empty()) {
        std::printf("DATA: NOT FOUND — data/sample/{rgb.ppm,depth.pgm,ground_truth.csv} missing "
                   "(run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::vector<unsigned char> h_rgb;
    std::vector<float> h_depth;
    std::vector<GtFruit> gt_fruits;
    if (!read_ppm(rgb_path, W, H, h_rgb) || !read_pgm16(depth_path, W, H, h_depth) ||
        !load_ground_truth(gt_path, gt_fruits)) {
        std::printf("DATA: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }
    std::printf("DATA: loaded %dx%d RGB-D orchard scene, %zu ground-truth fruit [synthetic]\n",
               W, H, gt_fruits.size());

    // ===================== GPU PIPELINE =======================================
    unsigned char *d_rgb = nullptr, *d_mask = nullptr, *d_mask2 = nullptr;
    float *d_h = nullptr, *d_s = nullptr, *d_v = nullptr, *d_depth = nullptr;
    int *d_label = nullptr, *d_changed = nullptr;
    int *d_count = nullptr, *d_sum_x = nullptr, *d_sum_y = nullptr;
    int *d_min_x = nullptr, *d_max_x = nullptr, *d_min_y = nullptr, *d_max_y = nullptr;
    float *d_sum_hue = nullptr, *d_sum_depth = nullptr, *d_mean_depth = nullptr;
    float *d_sum_depth_in = nullptr, *d_final_depth = nullptr;
    int *d_count_in = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rgb, static_cast<size_t>(N) * 3));
    CUDA_CHECK(cudaMalloc(&d_h, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_mask2, static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_label, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_x, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_x, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_x, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_y, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_y, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_hue, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_depth, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_depth, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_depth_in, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count_in, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_final_depth, static_cast<size_t>(N) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), static_cast<size_t>(N) * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth, h_depth.data(), static_cast<size_t>(N) * sizeof(float), cudaMemcpyHostToDevice));

    // ---- stage group 1: HSV, mask, morphological opening ----------------------
    GpuTimer gt_frontend; gt_frontend.begin();
    launch_rgb_to_hsv(d_rgb, d_h, d_s, d_v, W, H);
    launch_fruit_mask(d_h, d_s, d_v, d_mask, W, H);
    launch_morph_erode(d_mask, d_mask2, W, H);     // mask2 = erode(mask)
    launch_morph_dilate(d_mask2, d_mask, W, H);    // mask  = dilate(mask2)  -> mask now holds the OPENED result
    const float frontend_ms = gt_frontend.end_ms();

    // ---- stage group 2: connected-component labeling (iterate to convergence) --
    GpuTimer gt_ccl; gt_ccl.begin();
    launch_ccl_init(d_mask, d_label, W, H);
    int sweeps = 0, changed_host = 1;
    while (changed_host && sweeps < kMaxCclSweeps) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        launch_ccl_propagate_sweep(d_mask, d_label, W, H, d_changed);
        CUDA_CHECK(cudaMemcpy(&changed_host, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        ++sweeps;
    }
    const float ccl_ms = gt_ccl.end_ms();

    // ---- stage group 3: per-component statistics + robust depth ---------------
    GpuTimer gt_stats; gt_stats.begin();
    launch_component_stats_init(d_count, d_sum_x, d_sum_y, d_min_x, d_max_x, d_min_y, d_max_y,
                                d_sum_hue, d_sum_depth, d_sum_depth_in, d_count_in, W, H);
    launch_component_stats_pass1(d_mask, d_label, d_h, d_depth, d_count, d_sum_x, d_sum_y,
                                 d_min_x, d_max_x, d_min_y, d_max_y, d_sum_hue, d_sum_depth, W, H);
    launch_component_mean_depth(d_count, d_sum_depth, d_mean_depth, W, H);
    launch_component_stats_pass2_inlier(d_mask, d_label, d_depth, d_mean_depth,
                                        d_sum_depth_in, d_count_in, W, H);
    launch_component_finalize_depth(d_mean_depth, d_sum_depth_in, d_count_in, d_final_depth, W, H);
    const float stats_ms = gt_stats.end_ms();

    // ---- copy results back ------------------------------------------------
    std::vector<float> h_gpu(N), s_gpu(N), v_gpu(N);
    std::vector<unsigned char> mask_gpu(N);
    std::vector<int> label_gpu(N);
    std::vector<int> count_gpu(N), sum_x_gpu(N), sum_y_gpu(N);
    std::vector<int> min_x_gpu(N), max_x_gpu(N), min_y_gpu(N), max_y_gpu(N);
    std::vector<float> sum_hue_gpu(N), final_depth_gpu(N);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_h, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s_gpu.data(), d_s, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_gpu.data(), d_v, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mask_gpu.data(), d_mask, static_cast<size_t>(N), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(label_gpu.data(), d_label, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(count_gpu.data(), d_count, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sum_x_gpu.data(), d_sum_x, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sum_y_gpu.data(), d_sum_y, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(min_x_gpu.data(), d_min_x, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_x_gpu.data(), d_max_x, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(min_y_gpu.data(), d_min_y, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_y_gpu.data(), d_max_y, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sum_hue_gpu.data(), d_sum_hue, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(final_depth_gpu.data(), d_final_depth, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_rgb)); CUDA_CHECK(cudaFree(d_h)); CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_depth)); CUDA_CHECK(cudaFree(d_mask)); CUDA_CHECK(cudaFree(d_mask2));
    CUDA_CHECK(cudaFree(d_label)); CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_count)); CUDA_CHECK(cudaFree(d_sum_x)); CUDA_CHECK(cudaFree(d_sum_y));
    CUDA_CHECK(cudaFree(d_min_x)); CUDA_CHECK(cudaFree(d_max_x)); CUDA_CHECK(cudaFree(d_min_y)); CUDA_CHECK(cudaFree(d_max_y));
    CUDA_CHECK(cudaFree(d_sum_hue)); CUDA_CHECK(cudaFree(d_sum_depth)); CUDA_CHECK(cudaFree(d_mean_depth));
    CUDA_CHECK(cudaFree(d_sum_depth_in)); CUDA_CHECK(cudaFree(d_count_in)); CUDA_CHECK(cudaFree(d_final_depth));

    std::vector<FruitDetection> gpu_dets = build_detections(
        mask_gpu, label_gpu, count_gpu, sum_x_gpu, sum_y_gpu, min_x_gpu, max_x_gpu, min_y_gpu, max_y_gpu,
        sum_hue_gpu, final_depth_gpu, W, H);

    std::printf("[time] GPU: front-end (HSV+mask+opening) %.3f ms | CCL (%d sweeps) %.3f ms | "
               "component stats %.3f ms | total %.3f ms\n",
               static_cast<double>(frontend_ms), sweeps, static_cast<double>(ccl_ms),
               static_cast<double>(stats_ms), static_cast<double>(frontend_ms + ccl_ms + stats_ms));

    // ===================== CPU PIPELINE (the oracle) ===========================
    std::vector<float> h_cpu(N), s_cpu(N), v_cpu(N);
    std::vector<unsigned char> mask_cpu(N), mask_cpu_tmp(N);
    std::vector<int> label_cpu(N);
    std::vector<int> count_cpu(N), sum_x_cpu(N), sum_y_cpu(N);
    std::vector<int> min_x_cpu(N), max_x_cpu(N), min_y_cpu(N), max_y_cpu(N);
    std::vector<float> sum_hue_cpu(N), final_depth_cpu(N);

    CpuTimer cpu_timer; cpu_timer.begin();
    rgb_to_hsv_cpu(h_rgb.data(), h_cpu.data(), s_cpu.data(), v_cpu.data(), W, H);
    fruit_mask_cpu(h_cpu.data(), s_cpu.data(), v_cpu.data(), mask_cpu_tmp.data(), W, H);
    morph_erode_cpu(mask_cpu_tmp.data(), mask_cpu.data(), W, H);   // mask_cpu = erode(mask_cpu_tmp)
    morph_dilate_cpu(mask_cpu.data(), mask_cpu_tmp.data(), W, H);  // mask_cpu_tmp = dilate(mask_cpu) = OPENED
    mask_cpu.swap(mask_cpu_tmp);                                   // mask_cpu now holds the opened result
    ccl_union_find_cpu(mask_cpu.data(), label_cpu.data(), W, H);
    component_stats_cpu(mask_cpu.data(), label_cpu.data(), h_cpu.data(), h_depth.data(),
                        count_cpu.data(), sum_x_cpu.data(), sum_y_cpu.data(),
                        min_x_cpu.data(), max_x_cpu.data(), min_y_cpu.data(), max_y_cpu.data(),
                        sum_hue_cpu.data(), final_depth_cpu.data(), W, H);
    const double cpu_ms = cpu_timer.end_ms();

    std::vector<FruitDetection> cpu_dets = build_detections(
        mask_cpu, label_cpu, count_cpu, sum_x_cpu, sum_y_cpu, min_x_cpu, max_x_cpu, min_y_cpu, max_y_cpu,
        sum_hue_cpu, final_depth_cpu, W, H);

    std::printf("[time] CPU reference (all stages, single core): %.1f ms | GPU total %.3f ms | "
               "speed-up (teaching artifact) %.0fx\n",
               cpu_ms, static_cast<double>(frontend_ms + ccl_ms + stats_ms),
               cpu_ms / std::max(0.001, static_cast<double>(frontend_ms + ccl_ms + stats_ms)));

    // ===================== VERIFY: GPU vs CPU ===================================
    float worst_h = 0.0f, worst_s = 0.0f, worst_v = 0.0f;
    for (int i = 0; i < N; ++i) {
        worst_h = std::max(worst_h, std::fabs(h_gpu[i] - h_cpu[i]));
        worst_s = std::max(worst_s, std::fabs(s_gpu[i] - s_cpu[i]));
        worst_v = std::max(worst_v, std::fabs(v_gpu[i] - v_cpu[i]));
    }
    const bool hsv_pass = (worst_h <= kHsvTol * 360.0f) && (worst_s <= kHsvTol) && (worst_v <= kHsvTol);

    int mask_mismatches = 0;
    for (int i = 0; i < N; ++i) if (mask_gpu[i] != mask_cpu[i]) ++mask_mismatches;
    const bool mask_pass = (mask_mismatches == 0);

    // CCL: exact equality required ONLY where both masks agree (a mask
    // mismatch is already reported above; comparing labels where the masks
    // themselves disagree would just double-count that failure).
    long long label_mismatches = 0, label_compared = 0;
    for (int i = 0; i < N; ++i) {
        if (mask_gpu[i] && mask_cpu[i]) {
            ++label_compared;
            if (label_gpu[i] != label_cpu[i]) ++label_mismatches;
        }
    }
    const bool ccl_pass = (label_mismatches == 0);

    const bool detect_count_pass = (gpu_dets.size() == cpu_dets.size());
    float worst_stat_rel = 0.0f;
    if (detect_count_pass) {
        for (size_t i = 0; i < gpu_dets.size(); ++i) {
            const auto& g = gpu_dets[i];
            const auto& c = cpu_dets[i];   // same canonical label order on both sides -> same index
            auto rel = [](float a, float b) { const float scale = std::max(1.0f, std::fabs(b)); return std::fabs(a - b) / scale; };
            worst_stat_rel = std::max({worst_stat_rel, rel(g.center_x_m, c.center_x_m), rel(g.center_y_m, c.center_y_m),
                                       rel(g.center_z_m, c.center_z_m), rel(g.radius_m, c.radius_m),
                                       rel(g.ripeness, c.ripeness)});
        }
    }
    const bool stats_pass = detect_count_pass && (worst_stat_rel <= 1e-2f);

    const bool verify_pass = hsv_pass && mask_pass && ccl_pass && stats_pass;
    std::printf("[info] verify: worst |dH|=%.4f deg |dS|=%.2e |dV|=%.2e | mask mismatches %d/%d | "
               "CCL label mismatches %lld/%lld | detections GPU=%zu CPU=%zu | worst detection-stat rel %.2e\n",
               static_cast<double>(worst_h), static_cast<double>(worst_s), static_cast<double>(worst_v),
               mask_mismatches, N, label_mismatches, label_compared, gpu_dets.size(), cpu_dets.size(),
               static_cast<double>(worst_stat_rel));
    std::printf("VERIFY: %s (HSV tol %.0e, mask exact, CCL labels EXACT after canonicalization, "
               "detection stats rel tol 1e-2)\n", verify_pass ? "PASS" : "FAIL", static_cast<double>(kHsvTol));

    // ===================== GROUND-TRUTH GATES ===================================
    // Greedy nearest-neighbor matching: every (GT, detection) pair within
    // kMatchGateM, sorted by distance ascending, assigned greedily so the
    // closest pairs claim each other first (README/THEORY explain why this
    // is safe here: true matches are separated from wrong-fruit distances
    // by far more than the localization error this pipeline exhibits).
    struct Cand { size_t gi, di; float dist; };
    std::vector<Cand> cands;
    for (size_t gi = 0; gi < gt_fruits.size(); ++gi) {
        for (size_t di = 0; di < gpu_dets.size(); ++di) {
            const float dx = gt_fruits[gi].x_m - gpu_dets[di].center_x_m;
            const float dy = gt_fruits[gi].y_m - gpu_dets[di].center_y_m;
            const float dz = gt_fruits[gi].z_m - gpu_dets[di].center_z_m;
            const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (dist <= kMatchGateM) cands.push_back({gi, di, dist});
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.dist < b.dist; });
    std::vector<char> gt_used(gt_fruits.size(), 0), det_used(gpu_dets.size(), 0);
    std::vector<std::pair<size_t, size_t>> matches;   // (gt index, detection index)
    for (const auto& c : cands) {
        if (gt_used[c.gi] || det_used[c.di]) continue;
        gt_used[c.gi] = 1; det_used[c.di] = 1;
        matches.emplace_back(c.gi, c.di);
    }

    int detectable = 0;
    for (const auto& g : gt_fruits) if (g.visible_frac > 0.0f) ++detectable;
    int matched_detectable = 0;
    float loc_err_sum = 0.0f, loc_err_max = 0.0f, rad_err_sum = 0.0f, rad_err_max = 0.0f;
    std::vector<float> rip_pred, rip_true;
    for (const auto& m : matches) {
        const GtFruit& g = gt_fruits[m.first];
        const FruitDetection& d = gpu_dets[m.second];
        if (g.visible_frac <= 0.0f) continue;   // matched to the fully-occluded case would be a coincidence, not a hit
        ++matched_detectable;
        const float dx = g.x_m - d.center_x_m, dy = g.y_m - d.center_y_m, dz = g.z_m - d.center_z_m;
        const float loc_err = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float rad_err = std::fabs(g.radius_m - d.radius_m);
        loc_err_sum += loc_err; loc_err_max = std::max(loc_err_max, loc_err);
        rad_err_sum += rad_err; rad_err_max = std::max(rad_err_max, rad_err);
        rip_pred.push_back(d.ripeness); rip_true.push_back(g.ripeness);
    }
    const int false_positives = static_cast<int>(gpu_dets.size()) - static_cast<int>(matches.size());
    const float detection_rate = detectable > 0 ? static_cast<float>(matched_detectable) / static_cast<float>(detectable) : 0.0f;
    const float loc_err_mean = matched_detectable > 0 ? loc_err_sum / matched_detectable : 0.0f;
    const float rad_err_mean = matched_detectable > 0 ? rad_err_sum / matched_detectable : 0.0f;
    const double ripeness_rho = spearman_rank_correlation(rip_pred, rip_true);

    const bool detect_gate = (detection_rate >= kMinDetectionRate) && (false_positives <= kMaxFalsePositives);
    const bool loc_gate = (loc_err_max <= kLocErrMaxM);
    const bool radius_gate = (rad_err_max <= kRadiusErrMaxM);
    const bool ripeness_gate = (ripeness_rho >= kMinRipenessRankCorr);

    // NOTE on determinism (why the MEASURED numbers below sit on "[info]"
    // lines, not on the diffed gate lines): comp_sum_hue/comp_sum_depth are
    // built by GPU atomicAdd, and floating-point addition is not
    // associative — the exact bit pattern of a sum can in principle depend
    // on the order concurrent threads happen to execute in. This project's
    // component sizes are small enough (tens to low thousands of pixels)
    // that the measured values have been bit-stable across every run and
    // both configurations on the reference GPU, but CLAUDE.md's honesty
    // rule (and 08.01's precedent) says: do not PROMISE bit-exact floats
    // across arbitrary GPUs. The diffed "DETECT:"/"LOCALIZE:"/"RIPENESS:"
    // lines therefore state only the PASS/FAIL verdict and the fixed
    // thresholds (both are compile-time constants, so those lines truly
    // are stable); the actually-measured numbers are always printed too,
    // on the unchecked "[info]" line immediately above each verdict.
    std::printf("[info] detect: %d/%d detectable fruit found (rate %.2f), %d false positive(s)\n",
               matched_detectable, detectable, static_cast<double>(detection_rate), false_positives);
    std::printf("DETECT: detection rate >= %.2f threshold and false positives <= %d threshold -> %s\n",
               static_cast<double>(kMinDetectionRate), kMaxFalsePositives, detect_gate ? "PASS" : "FAIL");
    std::printf("[info] localize: 3-D center error mean %.4f m max %.4f m; radius error mean %.4f m max %.4f m\n",
               static_cast<double>(loc_err_mean), static_cast<double>(loc_err_max),
               static_cast<double>(rad_err_mean), static_cast<double>(rad_err_max));
    std::printf("LOCALIZE: center error <= %.3f m threshold and radius error <= %.3f m threshold -> %s\n",
               static_cast<double>(kLocErrMaxM), static_cast<double>(kRadiusErrMaxM),
               (loc_gate && radius_gate) ? "PASS" : "FAIL");
    std::printf("[info] ripeness: rank correlation (Spearman) rho=%.3f over %d matched fruit\n",
               ripeness_rho, matched_detectable);
    std::printf("RIPENESS: rank correlation >= %.2f threshold -> %s\n",
               kMinRipenessRankCorr, ripeness_gate ? "PASS" : "FAIL");

    // ===================== ARTIFACTS ===========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    bool wrote_pgm = false, wrote_csv = false;
    if (artifact_ok) {
        wrote_pgm = write_detections_pgm(out_dir + "/detections.pgm", h_rgb, gpu_dets, W, H);
        wrote_csv = write_fruit_map_csv(out_dir + "/fruit_map.csv", gpu_dets);
    }
    if (wrote_pgm) std::printf("ARTIFACT: wrote demo/out/detections.pgm (%zu rings burned in)\n", gpu_dets.size());
    else std::printf("ARTIFACT: FAILED to write demo/out/detections.pgm\n");
    if (wrote_csv) std::printf("ARTIFACT: wrote demo/out/fruit_map.csv (%zu rows)\n", gpu_dets.size());
    else std::printf("ARTIFACT: FAILED to write demo/out/fruit_map.csv\n");

    // ===================== RESULT ===============================================
    const bool success = verify_pass && detect_gate && loc_gate && radius_gate && ripeness_gate
                        && wrote_pgm && wrote_csv;
    if (success)
        std::printf("RESULT: PASS (GPU matches CPU reference; detection/localization/ripeness gates met)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/DETECT/LOCALIZE/RIPENESS lines above)\n");
    return success ? 0 : 1;
}
