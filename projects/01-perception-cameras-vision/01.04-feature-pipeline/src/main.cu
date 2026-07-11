// ===========================================================================
// main.cu — entry point for project 01.04
//           Feature pipeline: FAST/Harris detection, ORB descriptors,
//           brute-force Hamming matching, all GPU-vs-CPU verified and
//           checked against ground-truth-transform gates
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the three committed synthetic scenes (scene_a.pgm, scene_b.pgm
//      — the SAME scene under a known rotation+translation+brightness
//      transform — and neg_scene_c.pgm, an UNRELATED scene) plus build the
//      single-sourced ORB pattern table (see kernels.cuh).
//   2. STAGE 1 DETECT: FAST-9 on A, B, C (GPU + independent CPU twin, BIT-
//      EXACT — score map AND the final sorted keypoint list); Harris on A
//      only (GPU + CPU twin, TOLERANCE — response map), reported against
//      FAST for an overlap metric (not gated: different corner definitions).
//   3. STAGE 2 DESCRIBE: intensity-centroid orientation (GPU + CPU twin,
//      TOLERANCE) on A/B's FAST keypoints, quantized into 30 discrete
//      12-degree bins (see kernels.cuh's header for why), then oriented
//      rBRIEF descriptors (GPU + CPU twin, BIT-EXACT) on A/B/C.
//   4. STAGE 3 MATCH: brute-force Hamming, both directions (GPU + CPU twin,
//      exact), Lowe ratio test + mutual-consistency cross-check -> accepted
//      matches, for BOTH the real pair (A vs B) and the negative control
//      (A vs C).
//   5. FOUR INDEPENDENT GATES (none of which route through a GPU-vs-CPU
//      comparison — each checks something the twins above CANNOT):
//        ground_truth_transform  — accepted A-B matches, mapped through the
//                                   KNOWN transform, must land near the
//                                   real match (retyped independently here).
//        rotation_recovery       — median orientation delta of matched
//                                   pairs must equal the known rotation.
//        repeatability            — fraction of ALL scene-A FAST keypoints
//                                   whose transformed location has a
//                                   scene-B keypoint nearby (bypasses
//                                   descriptors/matching entirely).
//        negative_control         — A vs C matches, scored with the SAME
//                                   ground-truth check: inlier fraction
//                                   must be near zero (C bears no relation
//                                   to the A->B transform).
//      Plus one REPORTED (not gated) metric: harris_vs_fast_overlap.
//   6. ARTIFACTS: demo/out/{keypoints_A.ppm, keypoints_B.ppm, matches.ppm,
//      matches.csv, gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", every "VERIFY(...)"/"GATE ...:" verdict line, "ARTIFACT:", and
// "RESULT:" — all PASS/FAIL with NO embedded numbers (so they are
// byte-identical on every GPU architecture). Measured numbers live on
// "[info]"/"[time]" lines, deliberately NOT diffed by demo/run_demo.*
// (see 01.01/01.02's main.cu for the same convention and its rationale).
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ===========================================================================
// Ground-truth similarity transform, scene_a.pgm -> scene_b.pgm. MUST MATCH
// ../scripts/make_synthetic.py's TRANSFORM_* constants EXACTLY (that
// script's header explains why this repo hardcodes the same numbers twice
// with a cross-reference comment, rather than parsing a shared file — the
// 01.01 precedent for checkerboard geometry constants).
// ===========================================================================
static constexpr double kTransformThetaDeg = 12.0;
static constexpr double kTransformTxPx = 7.0;
static constexpr double kTransformTyPx = -5.0;
static constexpr double kCenterX = (kW - 1) / 2.0;
static constexpr double kCenterY = (kH - 1) / 2.0;

// ===========================================================================
// Gate / verify tolerances.
//
//   Measured on the reference machine (RTX 2080 SUPER, sm_75), Release,
//   scene_a.pgm/scene_b.pgm/neg_scene_c.pgm (committed sample):
//     verify(fast score, A)          max|gpu-cpu| = 0 (bit-exact); 135 keypoints
//     verify(fast score, B)          max|gpu-cpu| = 0 (bit-exact); 173 keypoints
//     verify(fast keypoints, A/B)    identical sorted lists (bit-exact)
//     verify(harris response, A)     max|gpu-cpu| = 5.215e-04 RELATIVE; 129 keypoints (adaptive threshold)
//     verify(orientation, A/B)       max|gpu-cpu| = 0.000000 rad (below float32 printable precision)
//     verify(orb descriptors, A/B)   0 / 34560 and 0 / 44288 bits differ (bit-exact)
//     verify(hamming distances)      0 field mismatches, both directions (bit-exact)
//     match(A,B)                     65 accepted (of 173 query) via ratio+cross-check+distance-cap
//     match(A,C-neg-control)         17 accepted (of 129 query)
//     gate ground_truth_transform    inlier fraction = 0.9231 (60/65) -- see kGtPixelTol's comment for the bimodal error distribution behind this number
//     gate rotation_recovery         median dtheta = 11.5145 deg, |error| = 0.4855 deg (ground truth 12.0 deg)
//     gate repeatability              fraction = 0.6370 (86/135)
//     gate negative_control           inlier fraction = 0.0000 (0/17)
//     [info] harris_vs_fast_overlap  0.5194 (67/129, reported only)
//   See README "Expected output" / THEORY.md "How we verify correctness"
//   for the full derivation of every tolerance below from these numbers.
// ===========================================================================
// Harris response magnitudes on this scene span roughly [0, ~1e13] (det(M)
// is a PRODUCT of two already-large box-summed squared-gradient terms) --
// see max_relative_diff_float()'s comment for why a RELATIVE tolerance
// (not an absolute one) is the honest comparison here, and kHarrisRelFloor
// for why near-zero pixels are excluded from that ratio.
static constexpr double kTolHarrisResponse = 2e-3;        // RELATIVE ceiling, ~3.8x measured 5.215e-4 (float32 box-sum vs double CPU box-sum -- see harris_response_cpu's note)
static constexpr double kHarrisRelFloor = 1.0e6;         // below this magnitude, a pixel's response is negligible (far under any corner threshold) -- relative error there is not meaningful, so the comparison floors the denominator instead of reporting it
static constexpr double kTolOrientationRad = 0.01;       // ceiling; measured GPU-vs-CPU agreement is EXACT (0.000000 to float32 printable precision) on the committed sample -- the tolerance exists for other GPU architectures' atan2f, and is still ~20x smaller than half the 12-deg bin width (0.1047 rad), so bin agreement is never at risk
static constexpr double kHarrisRelThreshold = 0.01;      // Harris pre-NMS floor = 1% of the frame's own peak response (an ADAPTIVE threshold, not a guessed absolute constant -- mirrors OpenCV's goodFeaturesToTrack "qualityLevel" parameter, see THEORY.md)

static constexpr double kGtPixelTol = 5.0;                // "landed near the true point" bound. Measured on the committed sample, the per-match geometric error is sharply BIMODAL: every CORRECT match lands within ~4.1 px (most well under 1 px), every WRONG match lands 84+ px away -- there is no ambiguous middle ground, so this tolerance is not a delicate knob (anywhere from ~5 to ~80 px draws the identical line); 5.0 gives comfortable margin on both sides.
static constexpr double kRotationRecoveryTolDeg = 1.0;    // ceiling, ~2x measured |error| 0.4855 deg
static constexpr double kRepeatabilityRadiusPx = 3.0;     // "nearby" definition for the repeatability gate
static constexpr double kRepeatabilityMinFrac = 0.50;     // floor, ~0.78x measured 0.6370 (comfortable margin below)
static constexpr double kNegControlMaxFrac = 0.10;        // ceiling, comfortably above measured 0.0000
static constexpr double kOverlapRadiusPx = 3.0;           // "nearby" definition for the (reported-only) Harris-vs-FAST overlap metric

// ===========================================================================
// Minimal, STRICT PGM (P5) reader/writer — same discipline as 01.01/01.02's
// readers: only ever reads files this project's own generator wrote.
// ===========================================================================
static bool read_pgm(const std::string& path, int& W, int& H, std::vector<unsigned char>& data)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    std::string magic;
    in >> magic;
    if (magic != "P5") return false;

    auto read_int = [&](int& out) -> bool {
        for (;;) {
            const int c = in.peek();
            if (c == '#') { std::string line; std::getline(in, line); continue; }
            if (c != EOF && std::isspace(c)) { in.get(); continue; }
            break;
        }
        in >> out;
        return static_cast<bool>(in);
    };
    int maxval = 0;
    if (!read_int(W) || !read_int(H) || !read_int(maxval)) return false;
    if (maxval != 255 || W <= 0 || H <= 0) return false;
    in.get();   // the single mandatory whitespace byte after maxval

    data.resize(static_cast<size_t>(W) * static_cast<size_t>(H));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return in.gcount() == static_cast<std::streamsize>(data.size());
}

static bool write_ppm(const std::string& path, int W, int H, const std::vector<unsigned char>& rgb)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P6\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(out);
}

// ===========================================================================
// Ground-truth transform, retyped independently in DOUBLE precision (per
// the roundtrip-gate style established in 01.01's main.cu): a THIRD
// implementation of the SAME formula scripts/make_synthetic.py's
// forward_transform()/inverse_transform() define, deliberately bypassing
// kernels.cuh entirely — this is what lets the ground-truth gates below
// catch a bug that a shared-code twin comparison structurally cannot
// (kernels.cuh's header explains the same principle for the twin-
// independence ruling; this is its "gate independence" analogue).
// ===========================================================================
static void forward_transform(double xa, double ya, double& xb, double& yb)
{
    const double theta = kTransformThetaDeg * (kPi / 180.0);
    const double c = std::cos(theta), s = std::sin(theta);
    const double ux = xa - kCenterX, uy = ya - kCenterY;
    xb = c * ux - s * uy + kCenterX + kTransformTxPx;
    yb = s * ux + c * uy + kCenterY + kTransformTyPx;
}

// wrap_angle_rad — wrap any finite radian angle into (-pi, pi].
static double wrap_angle_rad(double a)
{
    double w = std::fmod(a + kPi, 2.0 * kPi);
    if (w < 0.0) w += 2.0 * kPi;
    return w - kPi;
}

// ===========================================================================
// Sample loading — three committed grayscale PGMs, dimension-checked
// against kW/kH (strict: any mismatch aborts, never silently truncates).
// ===========================================================================
struct Sample {
    std::vector<unsigned char> a, b, c;   // scene_a, scene_b, neg_scene_c — each kW*kH bytes
    bool loaded = false;
};

static Sample load_sample(const std::string& cli_dir, const char* argv0)
{
    Sample s;
    const std::string pa = find_data_file(cli_dir, argv0, "scene_a.pgm");
    const std::string pb = find_data_file(cli_dir, argv0, "scene_b.pgm");
    const std::string pc = find_data_file(cli_dir, argv0, "neg_scene_c.pgm");
    if (pa.empty() || pb.empty() || pc.empty()) {
        std::fprintf(stderr, "sample: one or more of scene_a.pgm/scene_b.pgm/neg_scene_c.pgm not found "
                             "(run scripts/make_synthetic.py?)\n");
        return s;
    }
    int wa, ha, wb, hb, wc, hc;
    if (!read_pgm(pa, wa, ha, s.a)) { std::fprintf(stderr, "sample: failed to read scene_a.pgm\n"); return Sample{}; }
    if (!read_pgm(pb, wb, hb, s.b)) { std::fprintf(stderr, "sample: failed to read scene_b.pgm\n"); return Sample{}; }
    if (!read_pgm(pc, wc, hc, s.c)) { std::fprintf(stderr, "sample: failed to read neg_scene_c.pgm\n"); return Sample{}; }
    if (wa != kW || ha != kH || wb != kW || hb != kH || wc != kW || hc != kH) {
        std::fprintf(stderr, "sample: dimension mismatch -- expected %dx%d everywhere\n", kW, kH);
        return Sample{};
    }
    s.loaded = true;
    return s;
}

// ===========================================================================
// GpuImageFeatures — everything the pipeline knows about ONE image after
// STAGE 1 + STAGE 2 have run on the GPU: its FAST keypoints (sorted,
// GPU-verified) and their ORB descriptors. Bundled into one struct so the
// per-image pipeline (run once for A, once for B, once for C) reads as a
// single function rather than a wall of parallel arrays at the call site.
// ===========================================================================
struct GpuImageFeatures {
    std::vector<Keypoint> kps;             // FAST keypoints, sorted (score desc, y asc, x asc)
    std::vector<float> theta;              // per-keypoint orientation, radians
    std::vector<int> bin;                  // per-keypoint orientation bin, [0, kOrientBins)
    std::vector<OrbDescriptor> desc;       // per-keypoint 256-bit descriptor
};

// max_abs_diff_int / max_abs_diff_float — the two flavors of L-infinity
// comparison used throughout the VERIFY block below.
static long long max_abs_diff_int(const std::vector<int>& a, const std::vector<int>& b)
{
    long long m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<long long>(std::llabs(static_cast<long long>(a[i]) - b[i])));
    return m;
}
static double max_abs_diff_float(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

// max_relative_diff_float — |a-b| / max(floor, |a|, |b|), maxed over the
// array. Harris responses (see harris_response_kernel/harris_response_cpu)
// span an enormous DYNAMIC RANGE on this scene (peak response ~1e13,
// because det(M) involves a PRODUCT of two already-large box-summed
// squared-gradient terms) — a fixed ABSOLUTE tolerance would either be
// wildly loose at small magnitudes or (as an earlier version of this file
// measured directly: max|gpu-cpu| = 2,097,152 against a peak of
// 2.49e13) wildly tight relative to what float32's ~7 significant decimal
// digits can actually promise at that magnitude. A RELATIVE bound is the
// honest comparison for a quantity whose scale varies this much across
// one image; `floor` keeps near-zero pixels (background, far below any
// corner threshold, irrelevant to every downstream decision) from
// reporting a meaningless "infinite" relative error.
static double max_relative_diff_float(const std::vector<float>& a, const std::vector<float>& b, double floor_mag)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double av = static_cast<double>(a[i]), bv = static_cast<double>(b[i]);
        const double denom = std::max(floor_mag, std::max(std::fabs(av), std::fabs(bv)));
        m = std::max(m, std::fabs(av - bv) / denom);
    }
    return m;
}

// count_descriptor_bit_mismatches — total number of DIFFERING bits across
// two equal-length descriptor arrays (0 == bit-exact agreement).
static long long count_descriptor_bit_mismatches(const std::vector<OrbDescriptor>& a, const std::vector<OrbDescriptor>& b)
{
    long long mism = 0;
    for (size_t i = 0; i < a.size(); ++i)
        for (int w = 0; w < kOrbDescWords; ++w)
            mism += popcount32_portable(a[i].w[w] ^ b[i].w[w]);
    return mism;
}

// keypoint_lists_equal — exact (bit-for-bit, since score is an exact
// integer-valued float for FAST) structural equality of two sorted
// keypoint lists — the strong end-to-end check kernels.cuh's header
// promises for FAST ("not just the raw score array").
static bool keypoint_lists_equal(const std::vector<Keypoint>& a, const std::vector<Keypoint>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].score != b[i].score) return false;
    return true;
}

// sort_keypoints_desc — the SAME (score desc, y asc, x asc) convention
// fast_nms_select_cpu() applies internally, applied here to the GPU path's
// unordered atomic-compaction output so the two lists become directly
// comparable (see kernels.cuh's nms_select_fast_kernel header).
static void sort_keypoints_desc(std::vector<Keypoint>& kps)
{
    std::sort(kps.begin(), kps.end(), [](const Keypoint& x, const Keypoint& y) {
        if (x.score != y.score) return x.score > y.score;
        if (x.y != y.y) return x.y < y.y;
        return x.x < y.x;
    });
}

// ===========================================================================
// gates_metrics.csv writer (same shape as 01.01/01.02's CsvRow convention).
// ===========================================================================
struct CsvRow { std::string gate, metric, value, tol, pass; };

static std::string fmt(double v, int prec = 6)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return std::string(buf);
}

static bool write_gates_csv(const std::string& path, const std::vector<CsvRow>& rows)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "gate,metric,value,tolerance,pass\n";
    for (const auto& r : rows) out << r.gate << "," << r.metric << "," << r.value << "," << r.tol << "," << r.pass << "\n";
    return static_cast<bool>(out);
}

// ===========================================================================
// Drawing helpers for the PPM overlay artifacts — grayscale -> RGB
// replicate, plus tiny cross/circle/line primitives. Deliberately simple
// (no anti-aliasing): these are DIDACTIC visualizations, not a graphics
// library, and correctness of the underlying pipeline never depends on
// how they look.
// ===========================================================================
static std::vector<unsigned char> gray_to_rgb(const std::vector<unsigned char>& gray)
{
    std::vector<unsigned char> rgb(gray.size() * 3);
    for (size_t i = 0; i < gray.size(); ++i) { rgb[i * 3 + 0] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = gray[i]; }
    return rgb;
}
static inline void put_px(std::vector<unsigned char>& rgb, int W, int H, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    const size_t i = (static_cast<size_t>(y) * W + x) * 3;
    rgb[i + 0] = r; rgb[i + 1] = g; rgb[i + 2] = b;
}
static void draw_cross(std::vector<unsigned char>& rgb, int W, int H, int cx, int cy, int r,
                       unsigned char cr, unsigned char cg, unsigned char cb)
{
    for (int d = -r; d <= r; ++d) { put_px(rgb, W, H, cx + d, cy, cr, cg, cb); put_px(rgb, W, H, cx, cy + d, cr, cg, cb); }
}
static void draw_circle(std::vector<unsigned char>& rgb, int W, int H, int cx, int cy, int r,
                        unsigned char cr, unsigned char cg, unsigned char cb)
{
    // Midpoint circle, 8-way symmetry — a standard small routine, included
    // here (rather than pulled from a library) per the repo's "no black
    // boxes" spirit for even a purely-cosmetic helper.
    int x = r, y = 0, err = 0;
    while (x >= y) {
        put_px(rgb, W, H, cx + x, cy + y, cr, cg, cb); put_px(rgb, W, H, cx + y, cy + x, cr, cg, cb);
        put_px(rgb, W, H, cx - y, cy + x, cr, cg, cb); put_px(rgb, W, H, cx - x, cy + y, cr, cg, cb);
        put_px(rgb, W, H, cx - x, cy - y, cr, cg, cb); put_px(rgb, W, H, cx - y, cy - x, cr, cg, cb);
        put_px(rgb, W, H, cx + y, cy - x, cr, cg, cb); put_px(rgb, W, H, cx + x, cy - y, cr, cg, cb);
        ++y;
        if (err <= 0) { err += 2 * y + 1; }
        if (err > 0) { --x; err -= 2 * x + 1; }
    }
}
static void draw_line(std::vector<unsigned char>& rgb, int W, int H, int x0, int y0, int x1, int y1,
                      unsigned char cr, unsigned char cg, unsigned char cb)
{
    // Bresenham's line algorithm — the textbook integer-only version.
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        put_px(rgb, W, H, x0, y0, cr, cg, cb);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// ===========================================================================
// main.
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data path/to/data/sample]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] feature pipeline: FAST/Harris detect -> ORB describe -> brute-force Hamming match (project 01.04)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d grayscale scene pair (scene_a.pgm, scene_b.pgm = ground-truth similarity "
               "transform theta=%.1fdeg tx=%.1fpx ty=%.1fpx + brightness offset), plus an unrelated "
               "negative-control scene (neg_scene_c.pgm); FAST-9 t=%d, Harris k=%.2f, ORB 256 pairs / %d "
               "orientation bins, brute-force Hamming matching\n",
               kW, kH, kTransformThetaDeg, kTransformTxPx, kTransformTyPx, kFastThreshold, kHarrisK, kOrientBins);

    // ---- data --------------------------------------------------------------
    Sample sample = load_sample(data_dir, argv[0]);
    if (!sample.loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: %dx%d synthetic scene pair + negative-control scene (checkerboards at multiple "
               "scales/orientations + disks + gradient background) [synthetic, seed 42 / seed 999]\n", kW, kH);

    // ---- single-sourced ORB pattern table (host-built once; see kernels.cuh) ----
    std::vector<OrbPatternPair> base_pattern(kOrbNumPairs);
    build_orb_base_pattern(base_pattern.data());
    std::vector<RotatedOffset> rotated_table(static_cast<size_t>(kOrientBins) * kOrbNumPairs);
    build_rotated_pattern_table(base_pattern.data(), rotated_table.data());

    RotatedOffset* d_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_table, rotated_table.size() * sizeof(RotatedOffset)));
    CUDA_CHECK(cudaMemcpy(d_table, rotated_table.data(), rotated_table.size() * sizeof(RotatedOffset), cudaMemcpyHostToDevice));

    // ---- upload the three images ---------------------------------------------
    const int N = kW * kH;
    uint8_t *d_img_a = nullptr, *d_img_b = nullptr, *d_img_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img_a, N)); CUDA_CHECK(cudaMalloc(&d_img_b, N)); CUDA_CHECK(cudaMalloc(&d_img_c, N));
    CUDA_CHECK(cudaMemcpy(d_img_a, sample.a.data(), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img_b, sample.b.data(), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img_c, sample.c.data(), N, cudaMemcpyHostToDevice));

    bool verify_pass = true;
    std::vector<CsvRow> csv;
    GpuTimer gt; CpuTimer ct;
    float total_gpu_ms = 0.0f;
    double total_cpu_ms = 0.0;

    // =======================================================================
    // STAGE 1 DETECT: FAST-9 on A, B, C (A and B get the full VERIFY
    // battery; C reuses the already-verified kernels with no extra twin).
    // =======================================================================
    auto detect_fast = [&](const uint8_t* d_img, const std::vector<unsigned char>& h_img,
                           const char* label, bool run_verify) -> std::vector<Keypoint> {
        int* d_score = nullptr;
        CUDA_CHECK(cudaMalloc(&d_score, static_cast<size_t>(N) * sizeof(int)));
        gt.begin();
        launch_fast_score(d_img, d_score);
        total_gpu_ms += gt.end_ms();

        std::vector<int> score_gpu(N);
        CUDA_CHECK(cudaMemcpy(score_gpu.data(), d_score, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));

        int* d_out_x = nullptr; int* d_out_y = nullptr; int* d_out_score = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out_x, static_cast<size_t>(kMaxCandidates) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_y, static_cast<size_t>(kMaxCandidates) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_score, static_cast<size_t>(kMaxCandidates) * sizeof(int)));
        gt.begin();
        const int n_cand = launch_nms_select_fast(d_score, d_out_x, d_out_y, d_out_score, kMaxCandidates);
        total_gpu_ms += gt.end_ms();

        std::vector<int> cx(n_cand), cy(n_cand), cs(n_cand);
        if (n_cand > 0) {
            CUDA_CHECK(cudaMemcpy(cx.data(), d_out_x, static_cast<size_t>(n_cand) * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cy.data(), d_out_y, static_cast<size_t>(n_cand) * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cs.data(), d_out_score, static_cast<size_t>(n_cand) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        std::vector<Keypoint> kps_gpu(n_cand);
        for (int i = 0; i < n_cand; ++i) kps_gpu[i] = Keypoint{ cx[i], cy[i], static_cast<float>(cs[i]) };
        sort_keypoints_desc(kps_gpu);
        if (static_cast<int>(kps_gpu.size()) > kTopNFast) kps_gpu.resize(kTopNFast);

        CUDA_CHECK(cudaFree(d_out_x)); CUDA_CHECK(cudaFree(d_out_y)); CUDA_CHECK(cudaFree(d_out_score));
        CUDA_CHECK(cudaFree(d_score));

        if (run_verify) {
            ct.begin();
            std::vector<int> score_cpu(N);
            fast_score_cpu(h_img.data(), score_cpu.data());
            const long long d_score_diff = max_abs_diff_int(score_gpu, score_cpu);
            std::printf("[info] verify(fast score, %s): max|gpu-cpu| = %lld (tol 0, bit-exact)\n", label, d_score_diff);
            if (d_score_diff != 0) verify_pass = false;

            std::vector<Keypoint> kps_cpu(kTopNFast);
            const int n_cpu = fast_nms_select_cpu(score_cpu.data(), kps_cpu.data(), kTopNFast);
            kps_cpu.resize(n_cpu);
            total_cpu_ms += ct.end_ms();

            const bool lists_equal = keypoint_lists_equal(kps_gpu, kps_cpu);
            std::printf("[info] verify(fast keypoints, %s): gpu n=%zu cpu n=%zu, lists equal = %s (bit-exact)\n",
                       label, kps_gpu.size(), kps_cpu.size(), lists_equal ? "yes" : "NO");
            if (!lists_equal) verify_pass = false;
        }

        std::printf("[info] detect(fast, %s): %zu keypoints kept (of %d candidates found)\n", label, kps_gpu.size(), n_cand);
        return kps_gpu;
    };

    std::vector<Keypoint> kps_a = detect_fast(d_img_a, sample.a, "A", /*run_verify=*/true);
    std::vector<Keypoint> kps_b = detect_fast(d_img_b, sample.b, "B", /*run_verify=*/true);
    std::vector<Keypoint> kps_c = detect_fast(d_img_c, sample.c, "C(neg-control)", /*run_verify=*/false);

    std::printf("VERIFY(fast): %s (score maps bit-exact and final sorted keypoint lists bit-exact for A and B)\n",
               verify_pass ? "PASS" : "FAIL");

    // =======================================================================
    // STAGE 1 DETECT: Harris on image A only (detection-only comparison —
    // see kernels.cuh's header for why this project does not chase Harris
    // keypoints through description/matching).
    // =======================================================================
    float *d_gx = nullptr, *d_gy = nullptr, *d_harris = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gx, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_harris, static_cast<size_t>(N) * sizeof(float)));

    gt.begin();
    launch_sobel_gradient(d_img_a, d_gx, d_gy);
    launch_harris_response(d_gx, d_gy, d_harris);
    total_gpu_ms += gt.end_ms();

    std::vector<float> gx_gpu(N), gy_gpu(N), harris_gpu(N);
    CUDA_CHECK(cudaMemcpy(gx_gpu.data(), d_gx, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gy_gpu.data(), d_gy, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(harris_gpu.data(), d_harris, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost));

    bool harris_verify_pass = true;
    {
        ct.begin();
        std::vector<float> gx_cpu(N), gy_cpu(N), harris_cpu(N);
        sobel_gradient_cpu(sample.a.data(), gx_cpu.data(), gy_cpu.data());
        harris_response_cpu(gx_cpu.data(), gy_cpu.data(), harris_cpu.data());
        total_cpu_ms += ct.end_ms();

        const double d_grad = std::max(max_abs_diff_float(gx_gpu, gx_cpu), max_abs_diff_float(gy_gpu, gy_cpu));
        std::printf("[info] verify(sobel gradients, A): max|gpu-cpu| = %.6f (exact-integer-valued floats, expect 0)\n", d_grad);
        if (d_grad > 0.5) harris_verify_pass = false;   // Sobel outputs are EXACT integers (see kernels.cu numerics note) -- any nonzero diff here would be a real bug, not float noise

        const double d_harris_abs = max_abs_diff_float(harris_gpu, harris_cpu);
        const double d_harris_rel = max_relative_diff_float(harris_gpu, harris_cpu, kHarrisRelFloor);
        std::printf("[info] verify(harris response, A): max|gpu-cpu| = %.4f absolute (informational; see below), "
                   "%.3e RELATIVE (tol %.1e, floor %.0f) -- see kernels.cuh's numerics note for why this project "
                   "gates on the RELATIVE metric\n", d_harris_abs, d_harris_rel, kTolHarrisResponse, kHarrisRelFloor);
        if (d_harris_rel > kTolHarrisResponse) harris_verify_pass = false;
    }
    std::printf("VERIFY(harris): %s (Sobel gradients + Harris response match CPU reference within tolerance)\n",
               harris_verify_pass ? "PASS" : "FAIL");
    if (!harris_verify_pass) verify_pass = false;

    // Adaptive pre-NMS floor: 1% of THIS frame's own peak response (see
    // kHarrisRelThreshold's comment) -- computed from the data, not guessed.
    float harris_max = 0.0f;
    for (float v : harris_gpu) harris_max = std::max(harris_max, v);
    const float harris_thresh = static_cast<float>(kHarrisRelThreshold) * harris_max;

    int *d_hx = nullptr, *d_hy = nullptr; float* d_hscore = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hx, static_cast<size_t>(kMaxCandidates) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hy, static_cast<size_t>(kMaxCandidates) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hscore, static_cast<size_t>(kMaxCandidates) * sizeof(float)));
    const int n_harris_cand = launch_nms_select_harris(d_harris, harris_thresh, d_hx, d_hy, d_hscore, kMaxCandidates);
    std::vector<int> hx(n_harris_cand), hy(n_harris_cand);
    std::vector<float> hscore(n_harris_cand);
    if (n_harris_cand > 0) {
        CUDA_CHECK(cudaMemcpy(hx.data(), d_hx, static_cast<size_t>(n_harris_cand) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), d_hy, static_cast<size_t>(n_harris_cand) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hscore.data(), d_hscore, static_cast<size_t>(n_harris_cand) * sizeof(float), cudaMemcpyDeviceToHost));
    }
    std::vector<Keypoint> kps_a_harris(n_harris_cand);
    for (int i = 0; i < n_harris_cand; ++i) kps_a_harris[i] = Keypoint{ hx[i], hy[i], hscore[i] };
    sort_keypoints_desc(kps_a_harris);
    if (static_cast<int>(kps_a_harris.size()) > kTopNHarris) kps_a_harris.resize(kTopNHarris);
    std::printf("[info] detect(harris, A): %zu keypoints kept (of %d candidates found, adaptive threshold = %.2f = %.0f%% of peak %.2f)\n",
               kps_a_harris.size(), n_harris_cand, static_cast<double>(harris_thresh), kHarrisRelThreshold * 100.0, static_cast<double>(harris_max));

    CUDA_CHECK(cudaFree(d_hx)); CUDA_CHECK(cudaFree(d_hy)); CUDA_CHECK(cudaFree(d_hscore));
    CUDA_CHECK(cudaFree(d_gx)); CUDA_CHECK(cudaFree(d_gy)); CUDA_CHECK(cudaFree(d_harris));

    // ---- Harris-vs-FAST overlap: REPORTED ONLY, not gated (see kernels.cuh
    // header + this file's top comment: the two detectors define "corner"
    // differently, so disagreement is expected and not a defect). ----------
    int harris_near_fast = 0;
    for (const auto& hk : kps_a_harris) {
        for (const auto& fk : kps_a) {
            const double dx = hk.x - fk.x, dy = hk.y - fk.y;
            if (dx * dx + dy * dy <= kOverlapRadiusPx * kOverlapRadiusPx) { ++harris_near_fast; break; }
        }
    }
    const double harris_fast_overlap = kps_a_harris.empty() ? 0.0
        : static_cast<double>(harris_near_fast) / static_cast<double>(kps_a_harris.size());
    std::printf("[info] harris_vs_fast_overlap (A, reported only, not gated): %.4f (%d/%zu Harris keypoints "
               "within %.1f px of a FAST keypoint -- different corner definitions, agreement is informative, not required)\n",
               harris_fast_overlap, harris_near_fast, kps_a_harris.size(), kOverlapRadiusPx);
    csv.push_back({ "harris_vs_fast_overlap", "fraction_reported_only", fmt(harris_fast_overlap, 4), "n/a", "n/a" });

    // =======================================================================
    // STAGE 2 DESCRIBE: orientation + ORB descriptors for A, B (verified)
    // and C (reused, unverified — see STAGE 1's C note).
    // =======================================================================
    auto describe = [&](const uint8_t* d_img, const std::vector<unsigned char>& h_img,
                        const std::vector<Keypoint>& kps, const char* label, bool run_verify) -> GpuImageFeatures {
        GpuImageFeatures f;
        const int n = static_cast<int>(kps.size());
        f.kps = kps;
        f.theta.resize(n);
        f.bin.resize(n);
        f.desc.resize(n);
        if (n == 0) return f;

        std::vector<int> kx(n), ky(n);
        for (int i = 0; i < n; ++i) { kx[i] = kps[i].x; ky[i] = kps[i].y; }

        int *d_kx = nullptr, *d_ky = nullptr; float* d_theta = nullptr;
        CUDA_CHECK(cudaMalloc(&d_kx, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ky, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_theta, static_cast<size_t>(n) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kx, kx.data(), static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ky, ky.data(), static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));

        gt.begin();
        launch_orientation(d_img, d_kx, d_ky, n, d_theta);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(f.theta.data(), d_theta, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

        // Quantize the GPU-measured angle into a discrete bin ONCE — this
        // bin index becomes SHARED data fed to both the GPU describe kernel
        // and the CPU describe_cpu() twin below (see kernels.cuh's header
        // for the full bit-exactness argument this decision rests on).
        for (int i = 0; i < n; ++i) f.bin[i] = orient_to_bin(f.theta[i]);

        int* d_bin = nullptr;
        CUDA_CHECK(cudaMalloc(&d_bin, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bin, f.bin.data(), static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));

        OrbDescriptor* d_desc = nullptr;
        CUDA_CHECK(cudaMalloc(&d_desc, static_cast<size_t>(n) * sizeof(OrbDescriptor)));
        gt.begin();
        launch_describe(d_img, d_kx, d_ky, d_bin, n, d_table, d_desc);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(f.desc.data(), d_desc, static_cast<size_t>(n) * sizeof(OrbDescriptor), cudaMemcpyDeviceToHost));

        if (run_verify) {
            ct.begin();
            std::vector<float> theta_cpu(n);
            orientation_cpu(h_img.data(), kps.data(), n, theta_cpu.data());
            double max_ang_diff = 0.0;
            for (int i = 0; i < n; ++i)
                max_ang_diff = std::max(max_ang_diff, std::fabs(wrap_angle_rad(static_cast<double>(f.theta[i]) - theta_cpu[i])));
            std::printf("[info] verify(orientation, %s): max|gpu-cpu| = %.6f rad (tol %.4f rad)\n", label, max_ang_diff, kTolOrientationRad);
            if (max_ang_diff > kTolOrientationRad) verify_pass = false;

            // Confirm the (independently, tolerantly-measured) CPU angle
            // ALSO lands in the same discrete bin as the GPU angle for
            // every keypoint -- an explicit, checked assertion of the
            // "12-degree bin width >> orientation tolerance" argument
            // kernels.cuh's header makes, not just an assumption.
            int bin_mismatches = 0;
            for (int i = 0; i < n; ++i) if (orient_to_bin(theta_cpu[i]) != f.bin[i]) ++bin_mismatches;
            std::printf("[info] verify(orientation bin agreement, %s): %d/%d keypoints landed in a DIFFERENT "
                       "12-degree bin between the GPU and CPU angle (expect 0)\n", label, bin_mismatches, n);
            if (bin_mismatches != 0) verify_pass = false;

            std::vector<OrbDescriptor> desc_cpu(n);
            describe_cpu(h_img.data(), kps.data(), f.bin.data(), n, rotated_table.data(), desc_cpu.data());
            total_cpu_ms += ct.end_ms();
            const long long bit_mism = count_descriptor_bit_mismatches(f.desc, desc_cpu);
            std::printf("[info] verify(orb descriptors, %s): %lld / %lld total bits differ (tol 0, bit-exact)\n",
                       label, bit_mism, static_cast<long long>(n) * kOrbNumPairs);
            if (bit_mism != 0) verify_pass = false;
        }

        CUDA_CHECK(cudaFree(d_kx)); CUDA_CHECK(cudaFree(d_ky)); CUDA_CHECK(cudaFree(d_theta));
        CUDA_CHECK(cudaFree(d_bin)); CUDA_CHECK(cudaFree(d_desc));
        return f;
    };

    GpuImageFeatures feat_a = describe(d_img_a, sample.a, kps_a, "A", /*run_verify=*/true);
    GpuImageFeatures feat_b = describe(d_img_b, sample.b, kps_b, "B", /*run_verify=*/true);
    GpuImageFeatures feat_c = describe(d_img_c, sample.c, kps_c, "C(neg-control)", /*run_verify=*/false);

    std::printf("VERIFY(describe): %s (orientation within tolerance and lands in the same discrete bin; "
               "ORB descriptors bit-exact, for A and B)\n", verify_pass ? "PASS" : "FAIL");

    // =======================================================================
    // STAGE 3 MATCH — a reusable matcher: query vs train, both directions,
    // Lowe ratio test + mutual-consistency cross-check.
    // =======================================================================
    struct MatchSet {
        std::vector<int> fwd_b1d, fwd_b1i, fwd_b2d, fwd_b2i;   // query -> train (forward)
        std::vector<int> rev_b1d, rev_b1i, rev_b2d, rev_b2i;   // train -> query (reverse, for cross-check)
        std::vector<MatchResult> accepted;
    };

    auto match_images = [&](const GpuImageFeatures& query, const GpuImageFeatures& train,
                            const char* label, bool run_verify) -> MatchSet {
        MatchSet m;
        const int nQ = static_cast<int>(query.desc.size());
        const int nT = static_cast<int>(train.desc.size());
        m.fwd_b1d.assign(nQ, 0); m.fwd_b1i.assign(nQ, -1); m.fwd_b2d.assign(nQ, 0); m.fwd_b2i.assign(nQ, -1);
        m.rev_b1d.assign(nT, 0); m.rev_b1i.assign(nT, -1); m.rev_b2d.assign(nT, 0); m.rev_b2i.assign(nT, -1);
        if (nQ == 0 || nT == 0) return m;

        OrbDescriptor *d_q = nullptr, *d_t = nullptr;
        CUDA_CHECK(cudaMalloc(&d_q, static_cast<size_t>(nQ) * sizeof(OrbDescriptor)));
        CUDA_CHECK(cudaMalloc(&d_t, static_cast<size_t>(nT) * sizeof(OrbDescriptor)));
        CUDA_CHECK(cudaMemcpy(d_q, query.desc.data(), static_cast<size_t>(nQ) * sizeof(OrbDescriptor), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_t, train.desc.data(), static_cast<size_t>(nT) * sizeof(OrbDescriptor), cudaMemcpyHostToDevice));

        int *d_fb1d = nullptr, *d_fb1i = nullptr, *d_fb2d = nullptr, *d_fb2i = nullptr;
        int *d_rb1d = nullptr, *d_rb1i = nullptr, *d_rb2d = nullptr, *d_rb2i = nullptr;
        CUDA_CHECK(cudaMalloc(&d_fb1d, static_cast<size_t>(nQ) * sizeof(int))); CUDA_CHECK(cudaMalloc(&d_fb1i, static_cast<size_t>(nQ) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_fb2d, static_cast<size_t>(nQ) * sizeof(int))); CUDA_CHECK(cudaMalloc(&d_fb2i, static_cast<size_t>(nQ) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rb1d, static_cast<size_t>(nT) * sizeof(int))); CUDA_CHECK(cudaMalloc(&d_rb1i, static_cast<size_t>(nT) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rb2d, static_cast<size_t>(nT) * sizeof(int))); CUDA_CHECK(cudaMalloc(&d_rb2i, static_cast<size_t>(nT) * sizeof(int)));

        gt.begin();
        launch_hamming_match(d_q, nQ, d_t, nT, d_fb1d, d_fb1i, d_fb2d, d_fb2i);   // forward: query -> train
        launch_hamming_match(d_t, nT, d_q, nQ, d_rb1d, d_rb1i, d_rb2d, d_rb2i);   // reverse: train -> query (cross-check)
        total_gpu_ms += gt.end_ms();

        CUDA_CHECK(cudaMemcpy(m.fwd_b1d.data(), d_fb1d, static_cast<size_t>(nQ) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.fwd_b1i.data(), d_fb1i, static_cast<size_t>(nQ) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.fwd_b2d.data(), d_fb2d, static_cast<size_t>(nQ) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.fwd_b2i.data(), d_fb2i, static_cast<size_t>(nQ) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.rev_b1d.data(), d_rb1d, static_cast<size_t>(nT) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.rev_b1i.data(), d_rb1i, static_cast<size_t>(nT) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.rev_b2d.data(), d_rb2d, static_cast<size_t>(nT) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.rev_b2i.data(), d_rb2i, static_cast<size_t>(nT) * sizeof(int), cudaMemcpyDeviceToHost));

        if (run_verify) {
            ct.begin();
            std::vector<int> cfb1d(nQ), cfb1i(nQ), cfb2d(nQ), cfb2i(nQ);
            std::vector<int> crb1d(nT), crb1i(nT), crb2d(nT), crb2i(nT);
            hamming_match_cpu(query.desc.data(), nQ, train.desc.data(), nT, cfb1d.data(), cfb1i.data(), cfb2d.data(), cfb2i.data());
            hamming_match_cpu(train.desc.data(), nT, query.desc.data(), nQ, crb1d.data(), crb1i.data(), crb2d.data(), crb2i.data());
            total_cpu_ms += ct.end_ms();

            long long mism = 0;
            for (int i = 0; i < nQ; ++i) mism += (m.fwd_b1d[i] != cfb1d[i]) + (m.fwd_b1i[i] != cfb1i[i]) + (m.fwd_b2d[i] != cfb2d[i]) + (m.fwd_b2i[i] != cfb2i[i]);
            for (int i = 0; i < nT; ++i) mism += (m.rev_b1d[i] != crb1d[i]) + (m.rev_b1i[i] != crb1i[i]) + (m.rev_b2d[i] != crb2d[i]) + (m.rev_b2i[i] != crb2i[i]);
            std::printf("[info] verify(hamming distances, %s): %lld field mismatches across both directions (tol 0, bit-exact)\n", label, mism);
            if (mism != 0) verify_pass = false;
        }

        for (int qi = 0; qi < nQ; ++qi) {
            MatchResult r;
            r.query_idx = qi;
            r.train_idx = m.fwd_b1i[qi];
            r.best_dist = m.fwd_b1d[qi];
            r.second_dist = m.fwd_b2d[qi];
            r.ratio_ok = (r.train_idx >= 0) && (static_cast<double>(r.best_dist) <= kLoweRatio * static_cast<double>(r.second_dist));
            r.cross_ok = (r.train_idx >= 0) && (m.rev_b1i[r.train_idx] == qi);
            const bool dist_ok = (r.train_idx >= 0) && (r.best_dist <= kMaxHammingDist);   // see kernels.cuh's kMaxHammingDist comment
            r.accepted = r.ratio_ok && r.cross_ok && dist_ok;
            if (r.accepted) m.accepted.push_back(r);
        }

        CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_t));
        CUDA_CHECK(cudaFree(d_fb1d)); CUDA_CHECK(cudaFree(d_fb1i)); CUDA_CHECK(cudaFree(d_fb2d)); CUDA_CHECK(cudaFree(d_fb2i));
        CUDA_CHECK(cudaFree(d_rb1d)); CUDA_CHECK(cudaFree(d_rb1i)); CUDA_CHECK(cudaFree(d_rb2d)); CUDA_CHECK(cudaFree(d_rb2i));
        return m;
    };

    MatchSet match_ab = match_images(feat_b, feat_a, "B-query/A-train", /*run_verify=*/true);   // B is "current frame", A is "reference"
    std::printf("VERIFY(hamming): %s\n", verify_pass ? "PASS" : "FAIL");
    std::printf("[info] match(A,B): %zu accepted matches (ratio test tau=%.2f + mutual cross-check) out of %zu query keypoints\n",
               match_ab.accepted.size(), kLoweRatio, feat_b.kps.size());

    MatchSet match_ac = match_images(feat_c, feat_a, "C-query/A-train (negative control)", /*run_verify=*/false);
    std::printf("[info] match(A,C-neg-control): %zu accepted matches out of %zu query keypoints\n",
               match_ac.accepted.size(), feat_c.kps.size());

    // =======================================================================
    // GATE 1: ground_truth_transform — for every accepted A-B match, map
    // the A keypoint through the KNOWN transform and measure pixel error
    // against the actually-matched B keypoint.
    // =======================================================================
    int gt_inliers = 0;
    std::vector<double> gt_errs(match_ab.accepted.size());
    for (size_t i = 0; i < match_ab.accepted.size(); ++i) {
        const MatchResult& r = match_ab.accepted[i];
        const Keypoint& ka = feat_a.kps[static_cast<size_t>(r.train_idx)];
        const Keypoint& kb = feat_b.kps[static_cast<size_t>(r.query_idx)];
        double pbx, pby;
        forward_transform(ka.x, ka.y, pbx, pby);
        const double err = std::hypot(pbx - kb.x, pby - kb.y);
        gt_errs[i] = err;
        if (err <= kGtPixelTol) ++gt_inliers;
    }
    const double gt_inlier_frac = match_ab.accepted.empty() ? 0.0
        : static_cast<double>(gt_inliers) / static_cast<double>(match_ab.accepted.size());
    const bool gate_gt = !match_ab.accepted.empty() && gt_inlier_frac >= 0.90;
    std::printf("GATE ground_truth_transform: %s\n", gate_gt ? "PASS" : "FAIL");
    std::printf("[info] ground_truth_transform: %d/%zu accepted matches land within %.1f px of the KNOWN transform "
               "(tol >= 90%% documented threshold)\n", gt_inliers, match_ab.accepted.size(), kGtPixelTol);
    csv.push_back({ "ground_truth_transform", "inlier_fraction", fmt(gt_inlier_frac, 4), "0.90", gate_gt ? "PASS" : "FAIL" });

    // =======================================================================
    // GATE 2: rotation_recovery — median orientation delta of matched
    // pairs should equal the known rotation angle.
    // =======================================================================
    std::vector<double> dthetas;
    dthetas.reserve(match_ab.accepted.size());
    for (const auto& r : match_ab.accepted) {
        const double ta = static_cast<double>(feat_a.theta[static_cast<size_t>(r.train_idx)]);
        const double tb = static_cast<double>(feat_b.theta[static_cast<size_t>(r.query_idx)]);
        dthetas.push_back(wrap_angle_rad(tb - ta));
    }
    double median_dtheta_deg = 0.0;
    if (!dthetas.empty()) {
        std::vector<double> sorted_dt = dthetas;
        std::sort(sorted_dt.begin(), sorted_dt.end());
        const size_t mid = sorted_dt.size() / 2;
        const double median_rad = (sorted_dt.size() % 2 == 0) ? 0.5 * (sorted_dt[mid - 1] + sorted_dt[mid]) : sorted_dt[mid];
        median_dtheta_deg = median_rad * (180.0 / kPi);
    }
    const double rotation_err_deg = std::fabs(median_dtheta_deg - kTransformThetaDeg);
    const bool gate_rot = !dthetas.empty() && rotation_err_deg <= kRotationRecoveryTolDeg;
    std::printf("GATE rotation_recovery: %s\n", gate_rot ? "PASS" : "FAIL");
    std::printf("[info] rotation_recovery: median matched-pair orientation delta = %.4f deg (ground truth %.1f deg, "
               "|error| = %.4f deg, tol %.2f deg)\n", median_dtheta_deg, kTransformThetaDeg, rotation_err_deg, kRotationRecoveryTolDeg);
    csv.push_back({ "rotation_recovery", "abs_error_deg", fmt(rotation_err_deg, 4), fmt(kRotationRecoveryTolDeg, 2), gate_rot ? "PASS" : "FAIL" });

    // =======================================================================
    // GATE 3: repeatability — of ALL scene-A FAST keypoints (not just
    // matched ones), what fraction have a scene-B FAST keypoint near their
    // ground-truth-transformed location? A PURELY GEOMETRIC check —
    // bypasses descriptors and matching entirely.
    // =======================================================================
    int repeat_hits = 0;
    for (const auto& ka : feat_a.kps) {
        double pbx, pby;
        forward_transform(ka.x, ka.y, pbx, pby);
        bool found = false;
        for (const auto& kb : feat_b.kps) {
            const double dx = pbx - kb.x, dy = pby - kb.y;
            if (dx * dx + dy * dy <= kRepeatabilityRadiusPx * kRepeatabilityRadiusPx) { found = true; break; }
        }
        if (found) ++repeat_hits;
    }
    const double repeatability_frac = feat_a.kps.empty() ? 0.0
        : static_cast<double>(repeat_hits) / static_cast<double>(feat_a.kps.size());
    const bool gate_repeat = repeatability_frac >= kRepeatabilityMinFrac;
    std::printf("GATE repeatability: %s\n", gate_repeat ? "PASS" : "FAIL");
    std::printf("[info] repeatability: %d/%zu scene-A FAST keypoints have a scene-B FAST keypoint within %.1f px "
               "of their ground-truth-transformed location (floor %.2f)\n",
               repeat_hits, feat_a.kps.size(), kRepeatabilityRadiusPx, kRepeatabilityMinFrac);
    csv.push_back({ "repeatability", "fraction", fmt(repeatability_frac, 4), fmt(kRepeatabilityMinFrac, 2), gate_repeat ? "PASS" : "FAIL" });

    // =======================================================================
    // GATE 4: negative_control — A-vs-C accepted matches, scored with the
    // SAME ground-truth-transform check used in Gate 1. C bears no
    // relation to the A->B transform, so a near-zero inlier fraction here
    // is what PROVES Gate 1's high inlier fraction reflects real geometric
    // correspondence rather than the matcher/gate rubber-stamping anything.
    // =======================================================================
    int neg_inliers = 0;
    for (const auto& r : match_ac.accepted) {
        const Keypoint& ka = feat_a.kps[static_cast<size_t>(r.train_idx)];
        const Keypoint& kc = feat_c.kps[static_cast<size_t>(r.query_idx)];
        double pbx, pby;
        forward_transform(ka.x, ka.y, pbx, pby);   // same transform as Gate 1, applied to an UNRELATED image
        const double err = std::hypot(pbx - kc.x, pby - kc.y);
        if (err <= kGtPixelTol) ++neg_inliers;
    }
    const double neg_inlier_frac = match_ac.accepted.empty() ? 0.0
        : static_cast<double>(neg_inliers) / static_cast<double>(match_ac.accepted.size());
    const bool gate_neg = neg_inlier_frac <= kNegControlMaxFrac;
    std::printf("GATE negative_control: %s\n", gate_neg ? "PASS" : "FAIL");
    std::printf("[info] negative_control: %d/%zu A-vs-UNRELATED-scene accepted matches land within %.1f px of the "
               "A->B transform applied to an unrelated image (ceiling %.2f -- near zero proves the matcher/gates are "
               "not self-confirming)\n", neg_inliers, match_ac.accepted.size(), kGtPixelTol, kNegControlMaxFrac);
    csv.push_back({ "negative_control", "inlier_fraction", fmt(neg_inlier_frac, 4), fmt(kNegControlMaxFrac, 2), gate_neg ? "PASS" : "FAIL" });

    std::printf("[time] total GPU kernel time (all stages, all three images): %.3f ms\n", static_cast<double>(total_gpu_ms));
    std::printf("[time] total CPU reference time (all verified stages): %.3f ms\n", total_cpu_ms);

    // =======================================================================
    // ARTIFACTS
    // =======================================================================
    std::vector<unsigned char> vis_a = gray_to_rgb(sample.a);
    for (const auto& k : feat_a.kps) draw_cross(vis_a, kW, kH, k.x, k.y, 3, 0, 255, 0);        // FAST: green cross
    for (const auto& k : kps_a_harris) draw_circle(vis_a, kW, kH, k.x, k.y, 4, 60, 140, 255);  // Harris: blue circle

    std::vector<unsigned char> vis_b = gray_to_rgb(sample.b);
    for (const auto& k : feat_b.kps) draw_cross(vis_b, kW, kH, k.x, k.y, 3, 0, 255, 0);

    // Side-by-side match canvas: A on the left, B on the right (a small gap
    // between them), green lines for ground-truth INLIER matches, red for
    // accepted-but-geometrically-wrong matches — a direct visualization of
    // Gate 1.
    const int gap = 8;
    const int canvasW = kW * 2 + gap;
    std::vector<unsigned char> canvas(static_cast<size_t>(canvasW) * kH * 3, 40);   // dark gray gap/background
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const size_t sa = (static_cast<size_t>(y) * kW + x) * 3;
            const size_t da = (static_cast<size_t>(y) * canvasW + x) * 3;
            canvas[da] = canvas[da + 1] = canvas[da + 2] = sample.a[static_cast<size_t>(y) * kW + x];
            const size_t db = (static_cast<size_t>(y) * canvasW + (kW + gap + x)) * 3;
            canvas[db] = canvas[db + 1] = canvas[db + 2] = sample.b[static_cast<size_t>(y) * kW + x];
            (void)sa;
        }
    }
    for (size_t i = 0; i < match_ab.accepted.size(); ++i) {
        const MatchResult& r = match_ab.accepted[i];
        const Keypoint& ka = feat_a.kps[static_cast<size_t>(r.train_idx)];
        const Keypoint& kb = feat_b.kps[static_cast<size_t>(r.query_idx)];
        const bool inlier = gt_errs[i] <= kGtPixelTol;
        if (inlier) draw_line(canvas, canvasW, kH, ka.x, ka.y, kW + gap + kb.x, kb.y, 0, 220, 0);
        else        draw_line(canvas, canvasW, kH, ka.x, ka.y, kW + gap + kb.x, kb.y, 220, 0, 0);
    }
    for (const auto& k : feat_a.kps) draw_cross(canvas, canvasW, kH, k.x, k.y, 2, 0, 255, 0);
    for (const auto& k : feat_b.kps) draw_cross(canvas, canvasW, kH, kW + gap + k.x, k.y, 2, 0, 255, 0);

    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();
    artifact_ok = artifact_ok
        && write_ppm(out_dir + "/keypoints_A.ppm", kW, kH, vis_a)
        && write_ppm(out_dir + "/keypoints_B.ppm", kW, kH, vis_b)
        && write_ppm(out_dir + "/matches.ppm", canvasW, kH, canvas)
        && write_gates_csv(out_dir + "/gates_metrics.csv", csv);

    if (artifact_ok) {
        std::ofstream mcsv(out_dir + "/matches.csv");
        artifact_ok = static_cast<bool>(mcsv);
        if (artifact_ok) {
            mcsv << "query_b_idx,train_a_idx,bx,by,ax,ay,best_dist,second_dist,ratio_ok,cross_ok,gt_inlier\n";
            for (size_t i = 0; i < match_ab.accepted.size(); ++i) {
                const MatchResult& r = match_ab.accepted[i];
                const Keypoint& ka = feat_a.kps[static_cast<size_t>(r.train_idx)];
                const Keypoint& kb = feat_b.kps[static_cast<size_t>(r.query_idx)];
                const bool inlier = gt_errs[i] <= kGtPixelTol;
                mcsv << r.query_idx << "," << r.train_idx << "," << kb.x << "," << kb.y << "," << ka.x << "," << ka.y
                     << "," << r.best_dist << "," << r.second_dist << "," << (r.ratio_ok ? 1 : 0) << ","
                     << (r.cross_ok ? 1 : 0) << "," << (inlier ? 1 : 0) << "\n";
            }
        }
    }

    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{keypoints_A.ppm, keypoints_B.ppm, matches.ppm, matches.csv, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");

    // ---- cleanup --------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_img_a)); CUDA_CHECK(cudaFree(d_img_b)); CUDA_CHECK(cudaFree(d_img_c));
    CUDA_CHECK(cudaFree(d_table));

    // ---- verdict ----------------------------------------------------------------
    const bool success = verify_pass && gate_gt && gate_rot && gate_repeat && gate_neg && artifact_ok;
    if (success) {
        std::printf("RESULT: PASS (VERIFY(fast/harris/describe/hamming) + all 4 gates passed: "
                   "ground_truth_transform, rotation_recovery, repeatability, negative_control)\n");
    } else {
        std::printf("RESULT: FAIL (a VERIFY or GATE above did not pass -- see the lines above)\n");
    }
    return success ? 0 : 1;
}
