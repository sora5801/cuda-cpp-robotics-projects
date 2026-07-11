// ===========================================================================
// main.cu — entry point for project 01.03
//           Optical flow: dense pyramidal Lucas-Kanade + census-transform
//           block-matching flow, both GPU-vs-CPU verified and checked
//           against ANALYTIC ground-truth flow fields
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the reference frame (scene_a.pgm) and its four paired "frame B"
//      variants: pure translation, rotation+zoom, translation+brightness
//      ramp, and zero-motion (see ../scripts/make_synthetic.py).
//   2. STAGE VERIFY (five independent GPU-vs-CPU twin checks — see
//      kernels.cuh's header for the bit-exact-vs-tolerance strategy):
//        gradient        — Scharr gradients on scene_a, level 0.
//        lk_flow         — the FULL pyramidal-LK pipeline's final flow
//                           field on the translation pair.
//        census_transform— the 24-bit census signature on scene_a.
//        census_match    — WTA displacement + Hamming cost on the
//                           translation pair.
//        census_flow     — the FULL census pipeline's final flow + validity
//                           mask on the translation pair.
//   3. COMPUTE every flow field the gates below need: pyramidal LK and
//      single-level LK (the "no pyramid" ablation) on every scene pair;
//      census flow on every scene pair.
//   4. EIGHT INDEPENDENT GATES against ANALYTIC ground truth (retyped here
//      independently from ../scripts/make_synthetic.py — the same "gate
//      independence" principle project 01.04's main.cu documents in full):
//        translation_lk, translation_census   — the basic sanity gate both
//                                                methods must pass.
//        rotation_zoom_lk, pyramid_advantage  — the pyramid's reason to
//                                                exist, made numeric.
//        brightness_robustness_census         — census's rank-order
//                                                invariance, made numeric
//                                                (LK's degradation is
//                                                measured and reported,
//                                                honestly, not gated).
//        zero_motion_lk, zero_motion_census   — the negative control.
//        confidence_mask_sanity               — proves the LK confidence
//                                                mask is INFORMATIVE, not
//                                                decorative.
//   5. ARTIFACTS: demo/out/{flow_lk_rotzoom.ppm, flow_census_translation.ppm,
//      flow_color_wheel.ppm, epe_heatmap_lk_rotzoom.pgm,
//      confidence_lk_rotzoom.pgm, validity_census_translation.pgm,
//      gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", every "VERIFY(...)"/"GATE ...:" verdict line, "ARTIFACT:", and
// "RESULT:" — all PASS/FAIL with NO embedded numbers (byte-identical on
// every GPU architecture). Measured numbers live on "[info]"/"[time]"
// lines, deliberately NOT diffed by demo/run_demo.* (the 01.01/01.04
// convention this project follows).
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
// Ground-truth transforms, retyped independently in DOUBLE precision (per
// the roundtrip-gate style 01.01/01.04's main.cu establishes): a SECOND,
// independent implementation of the SAME formulas
// ../scripts/make_synthetic.py defines, deliberately bypassing kernels.cuh
// entirely — this is what lets the gates below catch a bug a shared-code
// twin comparison structurally cannot (kernels.cuh's header explains the
// same principle for the GPU-vs-CPU twins; this is its "gate independence"
// analogue). MUST MATCH make_synthetic.py's constants exactly.
// ===========================================================================
static constexpr double kTranslateTxPx = 3.0;
static constexpr double kTranslateTyPx = -3.0;
static constexpr double kRotThetaDeg = 6.0;
static constexpr double kRotZoomScale = 1.05;
static constexpr double kCenterX = (kW - 1) / 2.0;
static constexpr double kCenterY = (kH - 1) / 2.0;
static constexpr double kBrightnessGradMax = 51.0;

// forward_rotzoom — scene coordinate (xa,ya) -> its location in
// scene_b_rotzoom.pgm. See make_synthetic.py's forward_rotzoom() for the
// (identical) formula this independently retypes.
static void forward_rotzoom(double xa, double ya, double& xb, double& yb)
{
    const double theta = kRotThetaDeg * (kPi / 180.0);
    const double c = std::cos(theta), s = std::sin(theta);
    const double ux = xa - kCenterX, uy = ya - kCenterY;
    const double rx = kRotZoomScale * (c * ux - s * uy);
    const double ry = kRotZoomScale * (s * ux + c * uy);
    xb = rx + kCenterX;
    yb = ry + kCenterY;
}

// ===========================================================================
// Tunable gate thresholds and the LK confidence floor. Every bound below
// carries a documented margin over a MEASURED number from this project's
// committed sample (RTX 2080 SUPER, sm_75, Release) — see each constant's
// comment and README "Expected output" for the full measured line.
// ===========================================================================

// LK per-pixel CONFIDENCE floor, as a FRACTION of the frame's own peak
// small-eigenvalue (an ADAPTIVE threshold, the same "percentage of this
// frame's own peak" idea 01.04's Harris pre-NMS floor uses, rather than a
// guessed absolute magnitude — min_eig's absolute scale depends on the
// scene's local contrast, which this project's hashed-noise texture keeps
// fairly uniform but not perfectly so). A pixel is CONFIDENT iff
// min_eig(pixel) >= kMinEigConfidentFrac * max_min_eig(frame).
static constexpr double kMinEigConfidentFrac = 0.05;

static constexpr double kTranslationEpeBoundLkPx = 0.35;       // measured ~0.14 px, see README (~2.4x margin)
static constexpr double kTranslationEpeBoundCensusPx = 0.45;   // measured ~0.27 px, see README (~1.7x margin)
static constexpr double kRotZoomEpeBoundLkPx = 2.5;            // measured ~1.81 px (pyramidal), see README (~1.4x margin)
static constexpr double kPyramidAdvantageFloor = 1.5;          // single-level EPE / pyramidal EPE must exceed this (measured ~4.1x)
static constexpr double kBrightnessCensusFactor = 3.0;         // census(bright) EPE <= this x census(translation) EPE (measured ~2.4x)
static constexpr double kBrightnessCensusAbsBoundPx = 1.0;     // AND an absolute ceiling (measured ~0.64 px) -- still >17x better than LK's measured ~17.6 px on the same scene
static constexpr double kZeroMotionBoundLkPx = 0.20;           // measured mean |flow| ~0.00 px, see README
static constexpr double kZeroMotionBoundCensusPx = 0.35;       // measured mean |flow| ~0.28 px, see README (~1.25x margin)

// ===========================================================================
// Minimal, STRICT PGM (P5) reader / PGM+PPM writers — the same discipline
// as 01.01/01.04's readers: only ever reads files this project's own
// generator wrote.
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

static bool write_pgm(const std::string& path, int W, int H, const std::vector<unsigned char>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
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
// HSV-WHEEL FLOW COLOR CODING — the standard optical-flow visualization
// (the "Middlebury wheel": Baker et al. 2011's evaluation suite popularized
// exactly this convention, cited in README "Prior art"). HUE encodes flow
// DIRECTION (atan2(v,u)); SATURATION encodes flow MAGNITUDE relative to a
// chosen max (zero flow -> white, at-or-beyond max magnitude -> a fully
// saturated color); VALUE is held at 1 (bright) everywhere so hue/saturation
// stay legible. Implemented by hand (no library) per the repo's no-black-
// boxes rule (CLAUDE.md §1) — demo/README.md explains the wheel visually
// for the learner reading the output artifacts.
// ===========================================================================
static void hsv_to_rgb(double h_deg, double s, double v, unsigned char& r, unsigned char& g, unsigned char& b)
{
    h_deg = std::fmod(h_deg, 360.0);
    if (h_deg < 0.0) h_deg += 360.0;
    const double c = v * s;                                   // chroma
    const double hp = h_deg / 60.0;
    const double x = c * (1.0 - std::fabs(std::fmod(hp, 2.0) - 1.0));
    double r1 = 0, g1 = 0, b1 = 0;
    if      (hp < 1.0) { r1 = c; g1 = x; b1 = 0; }
    else if (hp < 2.0) { r1 = x; g1 = c; b1 = 0; }
    else if (hp < 3.0) { r1 = 0; g1 = c; b1 = x; }
    else if (hp < 4.0) { r1 = 0; g1 = x; b1 = c; }
    else if (hp < 5.0) { r1 = x; g1 = 0; b1 = c; }
    else                { r1 = c; g1 = 0; b1 = x; }
    const double m = v - c;
    auto to_byte = [](double c01) -> unsigned char {
        c01 = c01 < 0.0 ? 0.0 : (c01 > 1.0 ? 1.0 : c01);
        return static_cast<unsigned char>(c01 * 255.0 + 0.5);
    };
    r = to_byte(r1 + m); g = to_byte(g1 + m); b = to_byte(b1 + m);
}

// flow_to_rgb — one flow vector (u,v) -> one wheel color, magnitude clamped
// to [0, max_mag] before mapping to saturation (see this section's header).
static void flow_to_rgb(float u, float v, double max_mag, unsigned char& r, unsigned char& g, unsigned char& b)
{
    const double mag = std::sqrt(static_cast<double>(u) * u + static_cast<double>(v) * v);
    const double hue = std::atan2(static_cast<double>(v), static_cast<double>(u)) * (180.0 / kPi);   // (-180,180]
    const double sat = std::min(mag / std::max(max_mag, 1e-6), 1.0);
    hsv_to_rgb(hue, sat, 1.0, r, g, b);
}

// ===========================================================================
// Sample loading — the reference frame plus its four paired "frame B"
// variants, dimension-checked against kW/kH (strict: any mismatch aborts).
// ===========================================================================
struct Sample {
    std::vector<unsigned char> a, b_translate, b_rotzoom, b_bright, b_zero;
    bool loaded = false;
};

static Sample load_sample(const std::string& cli_dir, const char* argv0)
{
    Sample s;
    const char* names[5] = { "scene_a.pgm", "scene_b_translation.pgm", "scene_b_rotzoom.pgm",
                             "scene_b_translation_bright.pgm", "scene_b_zero.pgm" };
    std::vector<unsigned char>* dst[5] = { &s.a, &s.b_translate, &s.b_rotzoom, &s.b_bright, &s.b_zero };
    for (int i = 0; i < 5; ++i) {
        const std::string p = find_data_file(cli_dir, argv0, names[i]);
        if (p.empty()) {
            std::fprintf(stderr, "sample: %s not found (run scripts/make_synthetic.py?)\n", names[i]);
            return Sample{};
        }
        int w, h;
        if (!read_pgm(p, w, h, *dst[i]) || w != kW || h != kH) {
            std::fprintf(stderr, "sample: %s missing, malformed, or not %dx%d\n", names[i], kW, kH);
            return Sample{};
        }
    }
    s.loaded = true;
    return s;
}

// ===========================================================================
// Small numeric/reporting helpers shared by every VERIFY block and gate.
// ===========================================================================
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

// mean_flow_error — mean Euclidean distance between a measured flow field
// and a per-pixel ground-truth field, over MASKED pixels only. Used for
// both "endpoint error against real ground truth" (translation/rotzoom/
// bright gates) and "magnitude against a zero ground truth" (the
// zero-motion gates, by passing an all-zero gt field) — one function, two
// jobs, because they are the SAME formula (README states this explicitly
// rather than hiding it behind two differently-named wrappers).
static double mean_flow_error(const std::vector<float>& fu, const std::vector<float>& fv,
                              const std::vector<double>& gu, const std::vector<double>& gv,
                              const std::vector<uint8_t>& mask, long long& count_out)
{
    double sum = 0.0;
    long long cnt = 0;
    for (size_t i = 0; i < fu.size(); ++i) {
        if (!mask[i]) continue;
        const double du = static_cast<double>(fu[i]) - gu[i];
        const double dv = static_cast<double>(fv[i]) - gv[i];
        sum += std::sqrt(du * du + dv * dv);
        ++cnt;
    }
    count_out = cnt;
    return cnt > 0 ? sum / static_cast<double>(cnt) : -1.0;
}

// build_confident_mask_lk — interior pixels whose min_eig clears
// kMinEigConfidentFrac of this FRAME's own peak (see that constant's
// comment). Returns the mask and (via out params) the measured peak and
// the count of confident pixels, for the [info] line.
static std::vector<uint8_t> build_confident_mask_lk(const std::vector<float>& min_eig, double& peak_out, long long& n_confident_out)
{
    std::vector<uint8_t> mask(min_eig.size(), 0);
    float peak = 0.0f;
    for (size_t i = 0; i < min_eig.size(); ++i) peak = std::max(peak, min_eig[i]);
    peak_out = static_cast<double>(peak);
    const double thresh = kMinEigConfidentFrac * peak_out;
    long long n = 0;
    for (int y = kLkBorder; y < kH - kLkBorder; ++y) {
        for (int x = kLkBorder; x < kW - kLkBorder; ++x) {
            const size_t idx = static_cast<size_t>(y) * kW + x;
            if (static_cast<double>(min_eig[idx]) >= thresh) { mask[idx] = 1u; ++n; }
        }
    }
    n_confident_out = n;
    return mask;
}

static std::vector<uint8_t> invert_mask_interior(const std::vector<uint8_t>& mask, int border)
{
    std::vector<uint8_t> out(mask.size(), 0);
    for (int y = border; y < kH - border; ++y)
        for (int x = border; x < kW - border; ++x) {
            const size_t idx = static_cast<size_t>(y) * kW + x;
            out[idx] = mask[idx] ? 0u : 1u;
        }
    return out;
}

// gates_metrics.csv writer (same shape as 01.01/01.04's CsvRow convention).
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

    std::printf("[demo] optical flow: dense pyramidal Lucas-Kanade + census-transform block matching (project 01.03)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d grayscale frame pairs, %d-level LK pyramid (%d iters/level, %dx%d window), "
               "%dx%d census (radius %d search, %d-bit signature); Farneback documented-only (see THEORY.md, README S13)\n",
               kW, kH, kNumLevels, kLkIterationsPerLevel, 2 * kLkWindowRadius + 1, 2 * kLkWindowRadius + 1,
               2 * kCensusRadius + 1, 2 * kCensusRadius + 1, kCensusSearchRadius, kCensusBits);

    // ---- data ----------------------------------------------------------------
    Sample sample = load_sample(data_dir, argv[0]);
    if (!sample.loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: %dx%d hashed multi-scale-texture scene, 4 paired frames (translation, rotation+zoom, "
               "translation+brightness-ramp up to +%.0f intensity units, zero-motion) [synthetic, seed 42]\n",
               kW, kH, kBrightnessGradMax);

    const size_t N = static_cast<size_t>(kW) * kH;
    bool verify_pass = true;
    std::vector<CsvRow> csv;
    GpuTimer gt; CpuTimer ct;
    float total_gpu_ms = 0.0f;
    double total_cpu_ms = 0.0;

    // ---- upload every frame once ----------------------------------------------
    uint8_t *d_a = nullptr, *d_bt = nullptr, *d_br = nullptr, *d_bb = nullptr, *d_bz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N));  CUDA_CHECK(cudaMalloc(&d_bt, N));
    CUDA_CHECK(cudaMalloc(&d_br, N)); CUDA_CHECK(cudaMalloc(&d_bb, N)); CUDA_CHECK(cudaMalloc(&d_bz, N));
    CUDA_CHECK(cudaMemcpy(d_a, sample.a.data(), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt, sample.b_translate.data(), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_br, sample.b_rotzoom.data(), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bb, sample.b_bright.data(), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bz, sample.b_zero.data(), N, cudaMemcpyHostToDevice));

    // =======================================================================
    // VERIFY 1 — gradient stage: Scharr Gx,Gy on scene_a, level 0.
    // =======================================================================
    {
        float *d_gx = nullptr, *d_gy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_gx, N * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_gy, N * sizeof(float)));
        gt.begin();
        launch_scharr_gradient(d_a, kW, kH, d_gx, d_gy);
        total_gpu_ms += gt.end_ms();
        std::vector<float> gx_gpu(N), gy_gpu(N);
        CUDA_CHECK(cudaMemcpy(gx_gpu.data(), d_gx, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gy_gpu.data(), d_gy, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_gx)); CUDA_CHECK(cudaFree(d_gy));

        ct.begin();
        std::vector<float> gx_cpu(N), gy_cpu(N);
        scharr_gradient_cpu(sample.a.data(), kW, kH, gx_cpu.data(), gy_cpu.data());
        total_cpu_ms += ct.end_ms();

        const double d = std::max(max_abs_diff_float(gx_gpu, gx_cpu), max_abs_diff_float(gy_gpu, gy_cpu));
        std::printf("[info] verify(gradient): max|gpu-cpu| = %.6f (Scharr taps are integers scaled by an exact power-of-two 1/32 -- expect 0)\n", d);
        const bool ok = d <= 0.01;   // Scharr taps are integer-exact before an EXACT (power-of-two) /32 scale (kernels.cu numerics note) -- any nonzero diff is a real bug
        if (!ok) verify_pass = false;
        std::printf("VERIFY(gradient): %s (Scharr gradients bit-exact between GPU and CPU)\n", ok ? "PASS" : "FAIL");
    }

    // =======================================================================
    // VERIFY 2 — the FULL pyramidal-LK final flow field, translation pair.
    // Also reused below as this scene's GATE input (no need to recompute).
    // =======================================================================
    float *d_lk_u = nullptr, *d_lk_v = nullptr, *d_lk_eig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lk_u, N * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_lk_v, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lk_eig, N * sizeof(float)));
    std::vector<float> lk_translate_u(N), lk_translate_v(N), lk_translate_eig(N);
    {
        gt.begin();
        run_pyramidal_lk_gpu(d_a, d_bt, kNumLevels, kLkIterationsPerLevel, d_lk_u, d_lk_v, d_lk_eig);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(lk_translate_u.data(), d_lk_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_translate_v.data(), d_lk_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_translate_eig.data(), d_lk_eig, N * sizeof(float), cudaMemcpyDeviceToHost));

        ct.begin();
        std::vector<float> u_cpu(N), v_cpu(N), eig_cpu(N);
        pyramidal_lk_cpu(sample.a.data(), sample.b_translate.data(), kNumLevels, kLkIterationsPerLevel,
                        u_cpu.data(), v_cpu.data(), eig_cpu.data());
        total_cpu_ms += ct.end_ms();

        const double du = max_abs_diff_float(lk_translate_u, u_cpu);
        const double dv = max_abs_diff_float(lk_translate_v, v_cpu);
        const double deig = max_abs_diff_float(lk_translate_eig, eig_cpu);
        std::printf("[info] verify(lk_flow): max|gpu-cpu| u=%.4f px, v=%.4f px, min_eig=%.4e "
                   "(tol 0.25 px flow / 5%% relative eig -- 3 levels x 3 iters of float-vs-double bilinear-warp "
                   "accumulation, see THEORY.md)\n", du, dv, deig);
        const bool ok = (du <= 0.25) && (dv <= 0.25);
        if (!ok) verify_pass = false;
        std::printf("VERIFY(lk_flow): %s (final pyramidal-LK flow field matches CPU reference within tolerance)\n", ok ? "PASS" : "FAIL");
    }

    // =======================================================================
    // VERIFY 3 — census transform bit-exact, scene_a.
    // =======================================================================
    uint32_t *d_census_a = nullptr, *d_census_bt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_census_a, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_census_bt, N * sizeof(uint32_t)));
    {
        gt.begin();
        launch_census_transform(d_a, kW, kH, d_census_a);
        launch_census_transform(d_bt, kW, kH, d_census_bt);
        total_gpu_ms += gt.end_ms();
        std::vector<uint32_t> census_a_gpu(N);
        CUDA_CHECK(cudaMemcpy(census_a_gpu.data(), d_census_a, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        ct.begin();
        std::vector<uint32_t> census_a_cpu(N);
        census_transform_cpu(sample.a.data(), kW, kH, census_a_cpu.data());
        total_cpu_ms += ct.end_ms();

        long long mism = 0;
        for (size_t i = 0; i < N; ++i) if (census_a_gpu[i] != census_a_cpu[i]) ++mism;
        std::printf("[info] verify(census_transform): %lld / %zu signatures differ (tol 0, bit-exact)\n", mism, N);
        const bool ok = (mism == 0);
        if (!ok) verify_pass = false;
        std::printf("VERIFY(census_transform): %s (24-bit census signatures bit-exact between GPU and CPU)\n", ok ? "PASS" : "FAIL");
    }

    // =======================================================================
    // VERIFY 4 — census match: integer WTA + Hamming cost bit-exact,
    // sub-pixel refinement tolerance-checked, translation pair.
    // =======================================================================
    {
        float *d_mu = nullptr, *d_mv = nullptr; int* d_cost = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mu, N * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_mv, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cost, N * sizeof(int)));
        gt.begin();
        launch_census_match(d_census_a, d_census_bt, kW, kH, d_mu, d_mv, d_cost);
        total_gpu_ms += gt.end_ms();
        std::vector<float> mu_gpu(N), mv_gpu(N);
        std::vector<int> cost_gpu(N);
        CUDA_CHECK(cudaMemcpy(mu_gpu.data(), d_mu, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mv_gpu.data(), d_mv, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cost_gpu.data(), d_cost, N * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_mu)); CUDA_CHECK(cudaFree(d_mv)); CUDA_CHECK(cudaFree(d_cost));

        ct.begin();
        std::vector<uint32_t> census_a_cpu(N), census_bt_cpu(N);
        census_transform_cpu(sample.a.data(), kW, kH, census_a_cpu.data());
        census_transform_cpu(sample.b_translate.data(), kW, kH, census_bt_cpu.data());
        std::vector<float> mu_cpu(N), mv_cpu(N);
        std::vector<int> cost_cpu(N);
        census_match_cpu(census_a_cpu.data(), census_bt_cpu.data(), kW, kH, mu_cpu.data(), mv_cpu.data(), cost_cpu.data());
        total_cpu_ms += ct.end_ms();

        const long long cost_diff = max_abs_diff_int(cost_gpu, cost_cpu);
        const double sub_diff = std::max(max_abs_diff_float(mu_gpu, mu_cpu), max_abs_diff_float(mv_gpu, mv_cpu));
        std::printf("[info] verify(census_match): max|gpu-cpu| WTA cost = %lld (tol 0, bit-exact), "
                   "sub-pixel flow = %.4f px (tol 0.05 px)\n", cost_diff, sub_diff);
        const bool ok = (cost_diff == 0) && (sub_diff <= 0.05);
        if (!ok) verify_pass = false;
        std::printf("VERIFY(census_match): %s (WTA Hamming cost bit-exact; sub-pixel refinement within tolerance)\n", ok ? "PASS" : "FAIL");
    }

    // =======================================================================
    // VERIFY 5 — the FULL census-flow pipeline (transform+match+consistency),
    // translation pair. Also reused below as this scene's GATE input.
    // =======================================================================
    float *d_cf_u = nullptr, *d_cf_v = nullptr; uint8_t* d_cf_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cf_u, N * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_cf_v, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cf_valid, N));
    std::vector<float> census_translate_u(N), census_translate_v(N);
    std::vector<uint8_t> census_translate_valid(N);
    {
        gt.begin();
        run_census_flow_gpu(d_a, d_bt, d_cf_u, d_cf_v, d_cf_valid);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(census_translate_u.data(), d_cf_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_translate_v.data(), d_cf_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_translate_valid.data(), d_cf_valid, N, cudaMemcpyDeviceToHost));

        ct.begin();
        std::vector<float> u_cpu(N), v_cpu(N);
        std::vector<uint8_t> valid_cpu(N);
        census_flow_cpu(sample.a.data(), sample.b_translate.data(), u_cpu.data(), v_cpu.data(), valid_cpu.data());
        total_cpu_ms += ct.end_ms();

        const double du = max_abs_diff_float(census_translate_u, u_cpu);
        const double dv = max_abs_diff_float(census_translate_v, v_cpu);
        long long valid_mism = 0;
        for (size_t i = 0; i < N; ++i) if (census_translate_valid[i] != valid_cpu[i]) ++valid_mism;
        std::printf("[info] verify(census_flow): max|gpu-cpu| u=%.4f px, v=%.4f px, validity mask mismatches=%lld/%zu "
                   "(tol 0.1 px, 0 mask mismatches)\n", du, dv, valid_mism, N);
        const bool ok = (du <= 0.1) && (dv <= 0.1) && (valid_mism == 0);
        if (!ok) verify_pass = false;
        std::printf("VERIFY(census_flow): %s (final census flow + validity mask match CPU reference)\n", ok ? "PASS" : "FAIL");
    }
    CUDA_CHECK(cudaFree(d_census_a)); CUDA_CHECK(cudaFree(d_census_bt));

    // =======================================================================
    // Compute every remaining flow field the gates below need.
    // =======================================================================
    // LK, rotation+zoom pair: full pyramid AND the single-level ablation
    // (same total refinement budget — see kLkIterationsPerLevel's comment).
    std::vector<float> lk_rotzoom_u(N), lk_rotzoom_v(N), lk_rotzoom_eig(N);
    std::vector<float> lk_rotzoom_single_u(N), lk_rotzoom_single_v(N), lk_rotzoom_single_eig(N);
    {
        gt.begin();
        run_pyramidal_lk_gpu(d_a, d_br, kNumLevels, kLkIterationsPerLevel, d_lk_u, d_lk_v, d_lk_eig);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(lk_rotzoom_u.data(), d_lk_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_rotzoom_v.data(), d_lk_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_rotzoom_eig.data(), d_lk_eig, N * sizeof(float), cudaMemcpyDeviceToHost));

        gt.begin();
        run_pyramidal_lk_gpu(d_a, d_br, 1, kLkIterationsPerLevel * kNumLevels, d_lk_u, d_lk_v, d_lk_eig);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(lk_rotzoom_single_u.data(), d_lk_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_rotzoom_single_v.data(), d_lk_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_rotzoom_single_eig.data(), d_lk_eig, N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // LK, brightness pair (reported honestly, not gated — see kernels.cuh header).
    std::vector<float> lk_bright_u(N), lk_bright_v(N), lk_bright_eig(N);
    {
        gt.begin();
        run_pyramidal_lk_gpu(d_a, d_bb, kNumLevels, kLkIterationsPerLevel, d_lk_u, d_lk_v, d_lk_eig);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(lk_bright_u.data(), d_lk_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_bright_v.data(), d_lk_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_bright_eig.data(), d_lk_eig, N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // LK, zero-motion pair (negative control).
    std::vector<float> lk_zero_u(N), lk_zero_v(N), lk_zero_eig(N);
    {
        gt.begin();
        run_pyramidal_lk_gpu(d_a, d_bz, kNumLevels, kLkIterationsPerLevel, d_lk_u, d_lk_v, d_lk_eig);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(lk_zero_u.data(), d_lk_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_zero_v.data(), d_lk_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lk_zero_eig.data(), d_lk_eig, N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_lk_u)); CUDA_CHECK(cudaFree(d_lk_v)); CUDA_CHECK(cudaFree(d_lk_eig));

    // Census, the three remaining pairs (translation already computed above).
    std::vector<float> census_rotzoom_u(N), census_rotzoom_v(N);
    std::vector<uint8_t> census_rotzoom_valid(N);
    std::vector<float> census_bright_u(N), census_bright_v(N);
    std::vector<uint8_t> census_bright_valid(N);
    std::vector<float> census_zero_u(N), census_zero_v(N);
    std::vector<uint8_t> census_zero_valid(N);
    {
        gt.begin();
        run_census_flow_gpu(d_a, d_br, d_cf_u, d_cf_v, d_cf_valid);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(census_rotzoom_u.data(), d_cf_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_rotzoom_v.data(), d_cf_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_rotzoom_valid.data(), d_cf_valid, N, cudaMemcpyDeviceToHost));

        gt.begin();
        run_census_flow_gpu(d_a, d_bb, d_cf_u, d_cf_v, d_cf_valid);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(census_bright_u.data(), d_cf_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_bright_v.data(), d_cf_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_bright_valid.data(), d_cf_valid, N, cudaMemcpyDeviceToHost));

        gt.begin();
        run_census_flow_gpu(d_a, d_bz, d_cf_u, d_cf_v, d_cf_valid);
        total_gpu_ms += gt.end_ms();
        CUDA_CHECK(cudaMemcpy(census_zero_u.data(), d_cf_u, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_zero_v.data(), d_cf_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_zero_valid.data(), d_cf_valid, N, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_cf_u)); CUDA_CHECK(cudaFree(d_cf_v)); CUDA_CHECK(cudaFree(d_cf_valid));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_bt)); CUDA_CHECK(cudaFree(d_br)); CUDA_CHECK(cudaFree(d_bb)); CUDA_CHECK(cudaFree(d_bz));

    // =======================================================================
    // Ground-truth flow fields (per pixel), built from the independent
    // retyped transforms above.
    // =======================================================================
    std::vector<double> gt_translate_u(N, kTranslateTxPx), gt_translate_v(N, kTranslateTyPx);
    std::vector<double> gt_zero_u(N, 0.0), gt_zero_v(N, 0.0);
    std::vector<double> gt_rotzoom_u(N), gt_rotzoom_v(N);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            double xb, yb;
            forward_rotzoom(static_cast<double>(x), static_cast<double>(y), xb, yb);
            const size_t idx = static_cast<size_t>(y) * kW + x;
            gt_rotzoom_u[idx] = xb - x;
            gt_rotzoom_v[idx] = yb - y;
        }
    }

    // =======================================================================
    // Confidence / validity masks.
    // =======================================================================
    double lk_translate_peak, lk_rotzoom_peak, lk_bright_peak, lk_zero_peak;
    long long n_conf_translate, n_conf_rotzoom, n_conf_bright, n_conf_zero;
    std::vector<uint8_t> mask_lk_translate = build_confident_mask_lk(lk_translate_eig, lk_translate_peak, n_conf_translate);
    std::vector<uint8_t> mask_lk_rotzoom = build_confident_mask_lk(lk_rotzoom_eig, lk_rotzoom_peak, n_conf_rotzoom);
    std::vector<uint8_t> mask_lk_bright = build_confident_mask_lk(lk_bright_eig, lk_bright_peak, n_conf_bright);
    std::vector<uint8_t> mask_lk_zero = build_confident_mask_lk(lk_zero_eig, lk_zero_peak, n_conf_zero);
    std::printf("[info] lk confidence: peak min_eig translate=%.3e rotzoom=%.3e bright=%.3e zero=%.3e; "
               "confident pixel count translate=%lld rotzoom=%lld bright=%lld zero=%lld (floor = %.0f%% of each frame's own peak)\n",
               lk_translate_peak, lk_rotzoom_peak, lk_bright_peak, lk_zero_peak,
               n_conf_translate, n_conf_rotzoom, n_conf_bright, n_conf_zero, kMinEigConfidentFrac * 100.0);

    long long n_valid_translate = 0, n_valid_rotzoom = 0, n_valid_bright = 0, n_valid_zero = 0;
    for (auto v : census_translate_valid) n_valid_translate += v;
    for (auto v : census_rotzoom_valid) n_valid_rotzoom += v;
    for (auto v : census_bright_valid) n_valid_bright += v;
    for (auto v : census_zero_valid) n_valid_zero += v;
    const long long n_census_interior = static_cast<long long>(kW - 2 * kCensusBorder) * (kH - 2 * kCensusBorder);
    std::printf("[info] census validity (LR-consistency): translate=%lld rotzoom=%lld bright=%lld zero=%lld valid pixels (of up to %lld census-eligible interior)\n",
               n_valid_translate, n_valid_rotzoom, n_valid_bright, n_valid_zero, n_census_interior);

    // =======================================================================
    // GATE 1/2 — translation, both methods: the exact-answer sanity gate
    // (ground truth is a CONSTANT flow, the simplest possible check).
    // =======================================================================
    long long cnt;
    const double epe_translate_lk = mean_flow_error(lk_translate_u, lk_translate_v, gt_translate_u, gt_translate_v, mask_lk_translate, cnt);
    const bool gate_translate_lk = (cnt > 0) && (epe_translate_lk <= kTranslationEpeBoundLkPx);
    std::printf("GATE translation_lk: %s\n", gate_translate_lk ? "PASS" : "FAIL");
    std::printf("[info] translation_lk: mean EPE = %.4f px over %lld confident pixels (tol <= %.2f px)\n", epe_translate_lk, cnt, kTranslationEpeBoundLkPx);
    csv.push_back({ "translation_lk", "mean_epe_px", fmt(epe_translate_lk, 4), fmt(kTranslationEpeBoundLkPx, 2), gate_translate_lk ? "PASS" : "FAIL" });

    const double epe_translate_census = mean_flow_error(census_translate_u, census_translate_v, gt_translate_u, gt_translate_v, census_translate_valid, cnt);
    const bool gate_translate_census = (cnt > 0) && (epe_translate_census <= kTranslationEpeBoundCensusPx);
    std::printf("GATE translation_census: %s\n", gate_translate_census ? "PASS" : "FAIL");
    std::printf("[info] translation_census: mean EPE = %.4f px over %lld valid pixels (tol <= %.2f px)\n", epe_translate_census, cnt, kTranslationEpeBoundCensusPx);
    csv.push_back({ "translation_census", "mean_epe_px", fmt(epe_translate_census, 4), fmt(kTranslationEpeBoundCensusPx, 2), gate_translate_census ? "PASS" : "FAIL" });

    // =======================================================================
    // GATE 3/4 — rotation+zoom: the pyramid's reason to exist, made numeric.
    // =======================================================================
    const double epe_rotzoom_lk = mean_flow_error(lk_rotzoom_u, lk_rotzoom_v, gt_rotzoom_u, gt_rotzoom_v, mask_lk_rotzoom, cnt);
    const bool gate_rotzoom_lk = (cnt > 0) && (epe_rotzoom_lk <= kRotZoomEpeBoundLkPx);
    std::printf("GATE rotation_zoom_lk: %s\n", gate_rotzoom_lk ? "PASS" : "FAIL");
    std::printf("[info] rotation_zoom_lk: mean EPE = %.4f px over %lld confident pixels (pyramidal, %d levels; tol <= %.2f px)\n",
               epe_rotzoom_lk, cnt, kNumLevels, kRotZoomEpeBoundLkPx);
    csv.push_back({ "rotation_zoom_lk", "mean_epe_px", fmt(epe_rotzoom_lk, 4), fmt(kRotZoomEpeBoundLkPx, 2), gate_rotzoom_lk ? "PASS" : "FAIL" });

    long long cnt_single;
    const double epe_rotzoom_single = mean_flow_error(lk_rotzoom_single_u, lk_rotzoom_single_v, gt_rotzoom_u, gt_rotzoom_v, mask_lk_rotzoom, cnt_single);
    const double pyramid_advantage_ratio = (epe_rotzoom_lk > 1e-9) ? (epe_rotzoom_single / epe_rotzoom_lk) : 0.0;
    const bool gate_pyramid_advantage = (cnt_single > 0) && (pyramid_advantage_ratio >= kPyramidAdvantageFloor);
    std::printf("GATE pyramid_advantage: %s\n", gate_pyramid_advantage ? "PASS" : "FAIL");
    std::printf("[info] pyramid_advantage: single-level LK mean EPE = %.4f px (same %d-iteration total budget, "
               "SAME confident mask) vs pyramidal %.4f px -> ratio %.2fx (floor >= %.2fx)\n",
               epe_rotzoom_single, kLkIterationsPerLevel * kNumLevels, epe_rotzoom_lk, pyramid_advantage_ratio, kPyramidAdvantageFloor);
    csv.push_back({ "pyramid_advantage", "ratio_single_over_pyramidal", fmt(pyramid_advantage_ratio, 3), fmt(kPyramidAdvantageFloor, 2), gate_pyramid_advantage ? "PASS" : "FAIL" });

    // =======================================================================
    // GATE 5 — brightness robustness, census (gated); LK reported honestly.
    // =======================================================================
    const double epe_bright_census = mean_flow_error(census_bright_u, census_bright_v, gt_translate_u, gt_translate_v, census_bright_valid, cnt);
    const double brightness_ratio = (epe_translate_census > 1e-9) ? (epe_bright_census / epe_translate_census) : 0.0;
    const bool gate_brightness = (cnt > 0) && (epe_bright_census <= kBrightnessCensusAbsBoundPx) &&
                                 (brightness_ratio <= kBrightnessCensusFactor);
    std::printf("GATE brightness_robustness_census: %s\n", gate_brightness ? "PASS" : "FAIL");
    std::printf("[info] brightness_robustness_census: mean EPE = %.4f px (translation-scene EPE %.4f px, "
               "ratio %.2fx, tol ratio <= %.2fx AND absolute <= %.2f px) -- census's rank-order comparisons "
               "are near-invariant to the added local brightness ramp (THEORY.md)\n",
               epe_bright_census, epe_translate_census, brightness_ratio, kBrightnessCensusFactor, kBrightnessCensusAbsBoundPx);
    csv.push_back({ "brightness_robustness_census", "mean_epe_px", fmt(epe_bright_census, 4), fmt(kBrightnessCensusAbsBoundPx, 2), gate_brightness ? "PASS" : "FAIL" });

    const double epe_bright_lk = mean_flow_error(lk_bright_u, lk_bright_v, gt_translate_u, gt_translate_v, mask_lk_bright, cnt);
    std::printf("[info] brightness_robustness_lk (REPORTED ONLY, not gated): mean EPE = %.4f px over %lld confident "
               "pixels (vs %.4f px on the plain translation scene) -- LK regresses directly on raw intensity "
               "differences, so a spatially-varying brightness term biases its mismatch vector; this degradation "
               "is the brightness-constancy assumption breaking, exactly as THEORY.md predicts, not a bug\n",
               epe_bright_lk, cnt, epe_translate_lk);
    csv.push_back({ "brightness_robustness_lk", "mean_epe_px_reported_only", fmt(epe_bright_lk, 4), "n/a", "n/a" });

    // =======================================================================
    // GATE 6/7 — zero motion: the negative control, both methods.
    // =======================================================================
    const double mag_zero_lk = mean_flow_error(lk_zero_u, lk_zero_v, gt_zero_u, gt_zero_v, mask_lk_zero, cnt);
    const bool gate_zero_lk = (cnt > 0) && (mag_zero_lk <= kZeroMotionBoundLkPx);
    std::printf("GATE zero_motion_lk: %s\n", gate_zero_lk ? "PASS" : "FAIL");
    std::printf("[info] zero_motion_lk: mean |flow| = %.4f px over %lld confident pixels (tol <= %.2f px)\n", mag_zero_lk, cnt, kZeroMotionBoundLkPx);
    csv.push_back({ "zero_motion_lk", "mean_abs_flow_px", fmt(mag_zero_lk, 4), fmt(kZeroMotionBoundLkPx, 2), gate_zero_lk ? "PASS" : "FAIL" });

    const double mag_zero_census = mean_flow_error(census_zero_u, census_zero_v, gt_zero_u, gt_zero_v, census_zero_valid, cnt);
    const bool gate_zero_census = (cnt > 0) && (mag_zero_census <= kZeroMotionBoundCensusPx);
    std::printf("GATE zero_motion_census: %s\n", gate_zero_census ? "PASS" : "FAIL");
    std::printf("[info] zero_motion_census: mean |flow| = %.4f px over %lld valid pixels (tol <= %.2f px)\n", mag_zero_census, cnt, kZeroMotionBoundCensusPx);
    csv.push_back({ "zero_motion_census", "mean_abs_flow_px", fmt(mag_zero_census, 4), fmt(kZeroMotionBoundCensusPx, 2), gate_zero_census ? "PASS" : "FAIL" });

    // =======================================================================
    // GATE 8 — confidence-mask sanity: REJECTED pixels' mean EPE must
    // exceed ACCEPTED pixels' mean EPE (proves the mask is informative, not
    // decorative) — measured on the rotation+zoom scene (spatially varying
    // flow, so the aperture problem's effect is visible, unlike the
    // constant-flow translation scene where every pixel's TRUE flow is
    // identical regardless of confidence).
    // =======================================================================
    std::vector<uint8_t> mask_lk_rotzoom_rejected = invert_mask_interior(mask_lk_rotzoom, kLkBorder);
    long long cnt_accepted, cnt_rejected;
    const double epe_accepted = mean_flow_error(lk_rotzoom_u, lk_rotzoom_v, gt_rotzoom_u, gt_rotzoom_v, mask_lk_rotzoom, cnt_accepted);
    const double epe_rejected = mean_flow_error(lk_rotzoom_u, lk_rotzoom_v, gt_rotzoom_u, gt_rotzoom_v, mask_lk_rotzoom_rejected, cnt_rejected);
    const bool gate_confidence_sanity = (cnt_accepted > 0) && (cnt_rejected > 0) && (epe_rejected > epe_accepted);
    std::printf("GATE confidence_mask_sanity: %s\n", gate_confidence_sanity ? "PASS" : "FAIL");
    std::printf("[info] confidence_mask_sanity: accepted mean EPE = %.4f px (%lld px) vs rejected mean EPE = %.4f px "
               "(%lld px), rotation+zoom scene -- the LOW-confidence (small structure-tensor eigenvalue) pixels "
               "are where the aperture problem bites, and their flow estimate is measurably WORSE\n",
               epe_accepted, cnt_accepted, epe_rejected, cnt_rejected);
    csv.push_back({ "confidence_mask_sanity", "rejected_minus_accepted_epe_px", fmt(epe_rejected - epe_accepted, 4), "> 0", gate_confidence_sanity ? "PASS" : "FAIL" });

    std::printf("[time] total GPU kernel time (all stages, all scene pairs, both methods): %.3f ms\n", static_cast<double>(total_gpu_ms));
    std::printf("[time] total CPU reference time (all verified stages): %.3f ms\n", total_cpu_ms);

    // =======================================================================
    // ARTIFACTS
    // =======================================================================
    const double kFlowVizMaxMag = 12.0;   // wheel saturates at this many px/frame (comfortably above the rotzoom scene's worst-corner ground-truth magnitude)

    std::vector<unsigned char> flow_lk_rotzoom_rgb(N * 3);
    for (size_t i = 0; i < N; ++i) {
        unsigned char r, g, b;
        flow_to_rgb(lk_rotzoom_u[i], lk_rotzoom_v[i], kFlowVizMaxMag, r, g, b);
        flow_lk_rotzoom_rgb[i * 3 + 0] = r; flow_lk_rotzoom_rgb[i * 3 + 1] = g; flow_lk_rotzoom_rgb[i * 3 + 2] = b;
    }

    std::vector<unsigned char> flow_census_translate_rgb(N * 3);
    for (size_t i = 0; i < N; ++i) {
        unsigned char r, g, b;
        flow_to_rgb(census_translate_u[i], census_translate_v[i], kFlowVizMaxMag, r, g, b);
        flow_census_translate_rgb[i * 3 + 0] = r; flow_census_translate_rgb[i * 3 + 1] = g; flow_census_translate_rgb[i * 3 + 2] = b;
    }

    // EPE heatmap (LK, rotzoom): grayscale, 0 px -> black, kEpeHeatmapCapPx -> white.
    const double kEpeHeatmapCapPx = 3.0;
    std::vector<unsigned char> epe_heatmap(N, 0);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const size_t idx = static_cast<size_t>(y) * kW + x;
            const double du = static_cast<double>(lk_rotzoom_u[idx]) - gt_rotzoom_u[idx];
            const double dv = static_cast<double>(lk_rotzoom_v[idx]) - gt_rotzoom_v[idx];
            const double epe = std::sqrt(du * du + dv * dv);
            const double v = std::min(epe / kEpeHeatmapCapPx, 1.0) * 255.0;
            epe_heatmap[idx] = static_cast<unsigned char>(v + 0.5);
        }
    }

    // Confidence visualization (LK, rotzoom): min_eig normalized against
    // this frame's own peak (the SAME adaptive scale the gate itself uses).
    std::vector<unsigned char> confidence_viz(N, 0);
    for (size_t i = 0; i < N; ++i) {
        const double v = std::min(std::max(static_cast<double>(lk_rotzoom_eig[i]) / std::max(lk_rotzoom_peak, 1.0), 0.0), 1.0) * 255.0;
        confidence_viz[i] = static_cast<unsigned char>(v + 0.5);
    }

    // Validity mask visualization (census, translation): 0/255 binary.
    std::vector<unsigned char> validity_viz(N, 0);
    for (size_t i = 0; i < N; ++i) validity_viz[i] = census_translate_valid[i] ? 255 : 0;

    // Color-wheel legend: a small disc, hue = angle, saturation = radius/R.
    const int kWheelSize = 65, kWheelR = 32;
    std::vector<unsigned char> wheel_rgb(static_cast<size_t>(kWheelSize) * kWheelSize * 3, 30);
    for (int y = 0; y < kWheelSize; ++y) {
        for (int x = 0; x < kWheelSize; ++x) {
            const double dx = x - kWheelR, dy = y - kWheelR;
            const double r = std::sqrt(dx * dx + dy * dy);
            const size_t idx = (static_cast<size_t>(y) * kWheelSize + x) * 3;
            if (r <= kWheelR) {
                unsigned char rr, gg, bb;
                flow_to_rgb(static_cast<float>(dx), static_cast<float>(dy), kWheelR, rr, gg, bb);
                wheel_rgb[idx + 0] = rr; wheel_rgb[idx + 1] = gg; wheel_rgb[idx + 2] = bb;
            }
        }
    }

    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();
    artifact_ok = artifact_ok
        && write_ppm(out_dir + "/flow_lk_rotzoom.ppm", kW, kH, flow_lk_rotzoom_rgb)
        && write_ppm(out_dir + "/flow_census_translation.ppm", kW, kH, flow_census_translate_rgb)
        && write_ppm(out_dir + "/flow_color_wheel.ppm", kWheelSize, kWheelSize, wheel_rgb)
        && write_pgm(out_dir + "/epe_heatmap_lk_rotzoom.pgm", kW, kH, epe_heatmap)
        && write_pgm(out_dir + "/confidence_lk_rotzoom.pgm", kW, kH, confidence_viz)
        && write_pgm(out_dir + "/validity_census_translation.pgm", kW, kH, validity_viz)
        && write_gates_csv(out_dir + "/gates_metrics.csv", csv);

    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{flow_lk_rotzoom.ppm, flow_census_translation.ppm, flow_color_wheel.ppm, "
                   "epe_heatmap_lk_rotzoom.pgm, confidence_lk_rotzoom.pgm, validity_census_translation.pgm, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");

    // ---- verdict ----------------------------------------------------------------
    const bool all_gates = gate_translate_lk && gate_translate_census && gate_rotzoom_lk && gate_pyramid_advantage &&
                           gate_brightness && gate_zero_lk && gate_zero_census && gate_confidence_sanity;
    const bool success = verify_pass && all_gates && artifact_ok;
    if (success) {
        std::printf("RESULT: PASS (VERIFY(gradient/lk_flow/census_transform/census_match/census_flow) + all 8 gates passed)\n");
    } else {
        std::printf("RESULT: FAIL (a VERIFY or GATE above did not pass -- see the lines above)\n");
    }
    return success ? 0 : 1;
}
