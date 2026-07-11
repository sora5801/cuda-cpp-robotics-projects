// ===========================================================================
// main.cu — entry point for project 01.11
//           Low-light denoising (bilateral, non-local means, BM3D-lite)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed clean.pgm (ground truth, never seen by any
//      denoiser) and noisy.pgm (the only input every method sees).
//   2. Run five GPU pipelines: bilateral (naive + tiled), the Gaussian-blur
//      negative control, NLM, and BM3D-lite — AND their four independent
//      CPU twins (naive/tiled share one CPU oracle).
//   3. VERIFY: each GPU output against its CPU oracle (a per-method
//      tolerance), PLUS a dedicated bilateral naive-vs-tiled bit-identical
//      check (the tiling lesson's correctness half).
//   4. Five INDEPENDENT gates against the committed ground truth — never
//      routed through the pipeline being graded (kernels.cuh's Rect
//      constants + predicted_noise_std_dn() are the shared geometry/noise-
//      model CONTRACT, not the algorithm under test):
//        psnr_improvement   — every denoiser beats the noisy PSNR baseline.
//        edge_preservation  — bilateral/NLM/BM3D-lite retain most of the
//                              clean step edge's gradient; the Gaussian
//                              baseline is REQUIRED to fail this one (the
//                              designed negative control, task brief).
//        flat_noise_floor   — residual std in three flat ROIs drops well
//                              below the noisy baseline, every method.
//        method_ordering    — REPORTED honestly (not forced): texture-ROI
//                              PSNR ranks bilateral/NLM/BM3D-lite.
//        noise_model_sanity — measured noisy-frame std in the three flat
//                              ROIs vs. kernels.cuh's analytic prediction.
//   5. Write every artifact demo/README.md describes into demo/out/, print
//      one final PASS/FAIL.
//
// Output contract (load-bearing, CLAUDE.md §12): "[demo]", "PROBLEM:",
// "DATA:", "VERIFY(...)", "GATE ...:", "ARTIFACT:", "RESULT:" lines are
// STABLE and diffed by demo/expected_output.txt; "[info]"/"[time]" lines
// carry the actual measured numbers and are deliberately NOT diffed.
//
// Read this after: kernels.cuh (the contract), kernels.cu (the GPU
// kernels), reference_cpu.cpp (the independent CPU oracles).
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// VERIFY tolerances (GPU vs CPU, per method) and GATE thresholds. Every
// number below was MEASURED on the reference machine (RTX 2080 SUPER,
// sm_75) then margined — never set AT the measured value (the 01.01/01.09
// discipline) — measured values are quoted in each comment; the [info]
// lines this program prints reproduce them on any machine it runs on.
// ===========================================================================

// -- VERIFY: GPU-vs-CPU tolerances, per method (max |gpu-cpu| over all
// pixels, DN units). Bilateral/gaussian/NLM are per-pixel-independent
// stencils with NO cross-thread interaction, so their only disagreement
// source is expf/std::exp implementation ULPs — small. BM3D-lite
// aggregates via GPU atomicAdd (unordered float summation) vs. the CPU's
// fixed-order DOUBLE accumulation (reference_cpu.cpp's header explains the
// choice) — a genuinely different (not just differently-rounded) summation,
// hence the loosest tolerance of the four.
static constexpr double kTolBilateral = 0.05;      // measured ~0.004 DN
static constexpr double kTolGaussian = 0.02;       // measured ~0.001 DN (no range term: fewer expf calls' worth of ULP drift)
static constexpr double kTolNlm = 0.15;            // measured ~0.02 DN (169x more expf calls than bilateral)
static constexpr double kTolBm3d = 3.0;            // measured ~0.6 DN (atomic float vs. double-accumulated oracle)
static constexpr double kTolTiling = 0.0;          // bilateral naive vs. tiled: BIT-IDENTICAL by construction (kernels.cu)

// -- GATE thresholds.
// kMinPsnrImprovementDb is set low enough for the Gaussian baseline to
// still clear it (measured +2.31 dB) — averaging reduces variance almost
// everywhere, so even a filter with NO edge-awareness measurably improves
// whole-image PSNR; that is exactly why edge_preservation (a DIFFERENT,
// complementary gate) exists, and precisely the point README/THEORY make
// about PSNR alone never telling the whole denoising story. Bilateral/NLM/
// BM3D-lite clear this bar by a wide margin (measured +6.3/+12.7/+12.2 dB).
static constexpr double kMinPsnrImprovementDb = 2.0;   // every denoiser must beat noisy PSNR by this much
static constexpr double kEdgePreserveFrac = 0.55;      // bilateral/NLM/BM3D-lite must retain this fraction of clean edge gradient
static constexpr double kFlatNoiseFloorFrac = 0.55;    // denoised flat-ROI std must drop below this fraction of noisy std
static constexpr double kNoiseSanityLo = 0.80;         // measured noisy std / predicted std must fall in this band
static constexpr double kNoiseSanityHi = 1.20;         // (measured ratios: 0.97 / 0.97 / 0.98 across dark/mid/bright)

// Residual-heatmap visualization cap (DN): +-kResidualCap maps to full
// black/white — printed in [info] so the artifact is honestly self-
// describing (01.09's stretch-bounds-printing discipline), never a silent
// rescale.
static constexpr float kResidualCap = 64.0f;

// ===========================================================================
// Minimal, STRICT PGM (P5) reader/writer — the 01.01/01.08/01.09
// convention: only ever reads files this project's own generator wrote;
// any mismatch aborts rather than silently truncating.
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

// dn_to_pgm — round-and-clamp a float DN image (may stray slightly outside
// [0,255] from filter overshoot near a hard edge) into a viewable uint8 PGM.
static std::vector<unsigned char> dn_to_pgm(const std::vector<float>& img)
{
    std::vector<unsigned char> out(img.size());
    for (size_t i = 0; i < img.size(); ++i) {
        float v = img[i];
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        out[i] = static_cast<unsigned char>(v + 0.5f);
    }
    return out;
}

// residual_to_pgm — a SIGNED (denoised - clean) residual heatmap: mid-gray
// (128) is zero error, black is -kResidualCap DN, white is +kResidualCap DN
// (symmetric stretch, ALWAYS the same fixed cap across every artifact so the
// four residual maps are visually comparable to each other — unlike 01.09's
// per-image adaptive stretch, a fixed cap here is the more honest choice
// because "how much noise is left" is exactly what this project compares
// ACROSS methods). Also returns the RMS residual (== the error whose PSNR
// this project reports, by construction: PSNR = 10*log10(255^2/MSE)).
static std::vector<unsigned char> residual_to_pgm(const std::vector<float>& denoised,
                                                   const std::vector<float>& clean,
                                                   double& out_rms)
{
    std::vector<unsigned char> out(denoised.size());
    double acc = 0.0;
    for (size_t i = 0; i < denoised.size(); ++i) {
        const float d = denoised[i] - clean[i];
        acc += static_cast<double>(d) * static_cast<double>(d);
        float v = 128.0f + (d / kResidualCap) * 127.0f;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        out[i] = static_cast<unsigned char>(v + 0.5f);
    }
    out_rms = std::sqrt(acc / static_cast<double>(denoised.size()));
    return out;
}

// ===========================================================================
// Metric helpers shared by VERIFY and every gate below.
// ===========================================================================
static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

// mse_whole / mse_roi — mean squared error in DN^2, whole image or a
// rectangular region of interest (double accumulation: kN=30,000 terms of
// up to 255^2 each could exceed float's clean-integer range, and PSNR is
// exactly the kind of "small errors matter" metric worth the extra bits).
static double mse_whole(const std::vector<float>& a, const std::vector<float>& b)
{
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return acc / static_cast<double>(a.size());
}

static double mse_roi(const std::vector<float>& a, const std::vector<float>& b, int W, const Rect& r)
{
    double acc = 0.0;
    int count = 0;
    for (int y = r.y0; y < r.y1; ++y)
        for (int x = r.x0; x < r.x1; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            acc += d * d;
            ++count;
        }
    return acc / static_cast<double>(count);
}

// psnr_db — the standard 8-bit-peak PSNR: 10*log10(255^2/MSE). A tiny MSE
// floor (1e-8) guards log(0) for the (never expected, but not assumed
// impossible) case of a bit-exact reconstruction.
static double psnr_db(double mse_val)
{
    const double floor_mse = 1e-8;
    if (mse_val < floor_mse) mse_val = floor_mse;
    return 10.0 * std::log10((255.0 * 255.0) / mse_val);
}

// residual_std_in_rect — std of (img - clean_const) over a FLAT rectangle
// (clean_const is that rectangle's known, constant ground-truth value —
// kernels.cuh's kFlat*Dn constants). Equivalent to std(img) within the ROI
// since clean_const is a constant, but written this way so the "this is a
// residual against ground truth" framing is explicit at every call site.
static double residual_std_in_rect(const std::vector<float>& img, int W, const Rect& r, float clean_const)
{
    double acc = 0.0;
    int count = 0;
    for (int y = r.y0; y < r.y1; ++y)
        for (int x = r.x0; x < r.x1; ++x) {
            const double d = static_cast<double>(img[static_cast<size_t>(y) * W + x]) - static_cast<double>(clean_const);
            acc += d * d;
            ++count;
        }
    return std::sqrt(acc / static_cast<double>(count));
}

// erode_rect — shrink a rectangle inward by `margin` pixels on every side.
// Applied to every kFlat* rectangle before measuring residual std (GATEs 3
// and 5 below): the PAINTED patch (48x48, kernels.cuh) is deliberately
// bigger than any method's spatial reach, but a pixel WITHIN kernels.cuh's
// kFlatMeasureMargin of the patch's own border still legitimately blends
// with the surrounding hashed texture — that is correct filter behavior,
// not denoiser failure, and erosion is what keeps it out of the
// measurement (kFlatDark's comment in kernels.cuh tells the story of the
// 160x120/24x24 layout this project tried first, where skipping this step
// silently turned a border-contamination artifact into a gate failure).
static Rect erode_rect(const Rect& r, int margin)
{
    return Rect{ r.x0 + margin, r.x1 - margin, r.y0 + margin, r.y1 - margin };
}

// edge_gradient_mean — mean |I[stepX] - I[stepX-1]| over every row of
// kEdgeRegion: a simple horizontal finite difference straddling the known
// step boundary (kernels.cuh). On the CLEAN image this equals exactly
// kEdgeHiDn - kEdgeLoDn (a hard 1-pixel step, no clamp artifacts — the
// region sits well inside the frame); on a denoised image it measures how
// much of that step SURVIVED the filter.
static double edge_gradient_mean(const std::vector<float>& img, int W)
{
    double acc = 0.0;
    int count = 0;
    for (int y = kEdgeRegion.y0; y < kEdgeRegion.y1; ++y) {
        const float lo = img[static_cast<size_t>(y) * W + (kEdgeStepX - 1)];
        const float hi = img[static_cast<size_t>(y) * W + kEdgeStepX];
        acc += std::fabs(static_cast<double>(hi) - static_cast<double>(lo));
        ++count;
    }
    return acc / static_cast<double>(count);
}

// ===========================================================================
// gates_metrics.csv writer (the 01.01/01.09 shape: one row per measured
// quantity, a machine-readable teaching artifact).
// ===========================================================================
struct CsvRow { std::string gate, metric, value, tol, pass; };

static std::string fmt(double v, int prec = 4)
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

    std::printf("[demo] low-light denoising: bilateral / NLM / BM3D-lite vs a Gaussian-blur "
               "negative control (project 01.11)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d low-light frame, peak signal %.0f e- at code value 255, Poisson "
               "shot + Gaussian read noise (sigma_read=%.1f e-), FP32\n",
               kW, kH, static_cast<double>(kPeakElectrons), static_cast<double>(kReadNoiseE));

    // ---- data ---------------------------------------------------------------
    const std::string clean_path = find_data_file(data_dir, argv[0], "clean.pgm");
    const std::string noisy_path = find_data_file(data_dir, argv[0], "noisy.pgm");
    int w = 0, h = 0;
    std::vector<unsigned char> clean_u8, noisy_u8;
    bool loaded = !clean_path.empty() && !noisy_path.empty()
               && read_pgm(clean_path, w, h, clean_u8) && w == kW && h == kH
               && read_pgm(noisy_path, w, h, noisy_u8) && w == kW && h == kH;
    if (!loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: synthetic hashed multi-scale texture + 3 flat patches + 1 high-contrast "
               "step edge + 1 fine-detail ruling; clean.pgm ground truth + noisy.pgm (exact "
               "Poisson + Gaussian read noise) [synthetic, seed 42]\n");

    std::vector<float> h_clean(kN), h_noisy(kN);
    for (int i = 0; i < kN; ++i) { h_clean[i] = static_cast<float>(clean_u8[i]); h_noisy[i] = static_cast<float>(noisy_u8[i]); }

    // ---- device: upload the noisy frame ONCE, every method reads it --------
    float* d_noisy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_noisy, static_cast<size_t>(kN) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_noisy, h_noisy.data(), static_cast<size_t>(kN) * sizeof(float), cudaMemcpyHostToDevice));

    float *d_bil_naive = nullptr, *d_bil_tiled = nullptr, *d_gauss = nullptr, *d_nlm = nullptr, *d_bm3d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bil_naive, static_cast<size_t>(kN) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bil_tiled, static_cast<size_t>(kN) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gauss, static_cast<size_t>(kN) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nlm, static_cast<size_t>(kN) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bm3d, static_cast<size_t>(kN) * sizeof(float)));

    // ---- GPU pipelines, each timed independently ----------------------------
    GpuTimer gt;
    gt.begin(); launch_bilateral_naive(d_noisy, kW, kH, d_bil_naive); const float ms_bil_naive = gt.end_ms();
    gt.begin(); launch_bilateral_tiled(d_noisy, kW, kH, d_bil_tiled); const float ms_bil_tiled = gt.end_ms();
    gt.begin(); launch_gaussian_blur(d_noisy, kW, kH, d_gauss);       const float ms_gauss = gt.end_ms();
    gt.begin(); launch_nlm(d_noisy, kW, kH, d_nlm);                   const float ms_nlm = gt.end_ms();
    gt.begin(); launch_bm3d_lite(d_noisy, kW, kH, d_bm3d);            const float ms_bm3d = gt.end_ms();

    std::vector<float> h_bil_naive(kN), h_bil_tiled(kN), h_gauss(kN), h_nlm(kN), h_bm3d(kN);
    CUDA_CHECK(cudaMemcpy(h_bil_naive.data(), d_bil_naive, static_cast<size_t>(kN) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bil_tiled.data(), d_bil_tiled, static_cast<size_t>(kN) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gauss.data(), d_gauss, static_cast<size_t>(kN) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nlm.data(), d_nlm, static_cast<size_t>(kN) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bm3d.data(), d_bm3d, static_cast<size_t>(kN) * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("[time] GPU kernels: bilateral_naive %.3f ms | bilateral_tiled %.3f ms "
               "(tiling speed-up %.2fx, teaching artifact) | gaussian %.3f ms | nlm %.3f ms | "
               "bm3d_lite %.3f ms\n",
               static_cast<double>(ms_bil_naive), static_cast<double>(ms_bil_tiled),
               static_cast<double>(ms_bil_naive) / (static_cast<double>(ms_bil_tiled) > 0.0 ? static_cast<double>(ms_bil_tiled) : 1.0),
               static_cast<double>(ms_gauss), static_cast<double>(ms_nlm), static_cast<double>(ms_bm3d));

    // ---- CPU reference oracles, each timed independently --------------------
    std::vector<float> h_bil_cpu(kN), h_gauss_cpu(kN), h_nlm_cpu(kN), h_bm3d_cpu(kN);
    CpuTimer ct;
    ct.begin(); bilateral_cpu(h_noisy.data(), kW, kH, h_bil_cpu.data());     const double cms_bil = ct.end_ms();
    ct.begin(); gaussian_blur_cpu(h_noisy.data(), kW, kH, h_gauss_cpu.data()); const double cms_gauss = ct.end_ms();
    ct.begin(); nlm_cpu(h_noisy.data(), kW, kH, h_nlm_cpu.data());           const double cms_nlm = ct.end_ms();
    ct.begin(); bm3d_lite_cpu(h_noisy.data(), kW, kH, h_bm3d_cpu.data());    const double cms_bm3d = ct.end_ms();
    std::printf("[time] CPU reference (single-thread): bilateral %.1f ms | gaussian %.1f ms | "
               "nlm %.1f ms | bm3d_lite %.1f ms\n", cms_bil, cms_gauss, cms_nlm, cms_bm3d);

    // ======================= VERIFY: GPU vs CPU, per method ===================
    // (diff_* names, deliberately NOT d_* — d_ is this file's device-pointer
    // prefix convention, CLAUDE.md §12, and these are host-side scalars.)
    const double diff_bil = max_abs_diff(h_bil_naive, h_bil_cpu);
    const double diff_gauss = max_abs_diff(h_gauss, h_gauss_cpu);
    const double diff_nlm = max_abs_diff(h_nlm, h_nlm_cpu);
    const double diff_bm3d = max_abs_diff(h_bm3d, h_bm3d_cpu);
    const double diff_tiling = max_abs_diff(h_bil_naive, h_bil_tiled);

    const bool verify_bil = diff_bil <= kTolBilateral;
    const bool verify_gauss = diff_gauss <= kTolGaussian;
    const bool verify_nlm = diff_nlm <= kTolNlm;
    const bool verify_bm3d = diff_bm3d <= kTolBm3d;
    const bool verify_tiling = diff_tiling <= kTolTiling;
    const bool verify_pass = verify_bil && verify_gauss && verify_nlm && verify_bm3d && verify_tiling;

    std::printf("[info] verify(bilateral): max|gpu-cpu|=%.4f DN (tol %.2f)\n", diff_bil, kTolBilateral);
    std::printf("VERIFY(bilateral): %s (GPU naive matches CPU reference within tolerance)\n", verify_bil ? "PASS" : "FAIL");
    std::printf("[info] verify(bilateral_tiling): max|naive-tiled|=%.6f DN (tol %.1f — BIT-IDENTICAL "
               "by construction, kernels.cu)\n", diff_tiling, kTolTiling);
    std::printf("VERIFY(bilateral_tiling): %s (GPU naive and GPU tiled bit-identical)\n", verify_tiling ? "PASS" : "FAIL");
    std::printf("[info] verify(gaussian_baseline): max|gpu-cpu|=%.4f DN (tol %.2f)\n", diff_gauss, kTolGaussian);
    std::printf("VERIFY(gaussian_baseline): %s (GPU matches CPU reference within tolerance)\n", verify_gauss ? "PASS" : "FAIL");
    std::printf("[info] verify(nlm): max|gpu-cpu|=%.4f DN (tol %.2f)\n", diff_nlm, kTolNlm);
    std::printf("VERIFY(nlm): %s (GPU matches CPU reference within tolerance)\n", verify_nlm ? "PASS" : "FAIL");
    std::printf("[info] verify(bm3d_lite): max|gpu-cpu|=%.4f DN (tol %.2f — atomic float vs. "
               "double-accumulated oracle, reference_cpu.cpp)\n", diff_bm3d, kTolBm3d);
    std::printf("VERIFY(bm3d_lite): %s (GPU matches CPU reference within tolerance)\n", verify_bm3d ? "PASS" : "FAIL");

    // ======================= GATE 1: psnr_improvement ==========================
    const double psnr_noisy = psnr_db(mse_whole(h_noisy, h_clean));
    const double psnr_bil = psnr_db(mse_whole(h_bil_naive, h_clean));
    const double psnr_gauss = psnr_db(mse_whole(h_gauss, h_clean));
    const double psnr_nlm = psnr_db(mse_whole(h_nlm, h_clean));
    const double psnr_bm3d = psnr_db(mse_whole(h_bm3d, h_clean));

    const bool imp_bil = (psnr_bil - psnr_noisy) >= kMinPsnrImprovementDb;
    const bool imp_gauss = (psnr_gauss - psnr_noisy) >= kMinPsnrImprovementDb;
    const bool imp_nlm = (psnr_nlm - psnr_noisy) >= kMinPsnrImprovementDb;
    const bool imp_bm3d = (psnr_bm3d - psnr_noisy) >= kMinPsnrImprovementDb;
    const bool gate_psnr = imp_bil && imp_gauss && imp_nlm && imp_bm3d;

    std::printf("GATE psnr_improvement: %s\n", gate_psnr ? "PASS" : "FAIL");
    std::printf("[info] psnr_improvement (whole image, dB; margin required >= %.1f dB over noisy): "
               "noisy=%.2f | bilateral=%.2f (+%.2f) | gaussian=%.2f (+%.2f) | nlm=%.2f (+%.2f) | "
               "bm3d_lite=%.2f (+%.2f)\n", kMinPsnrImprovementDb, psnr_noisy,
               psnr_bil, psnr_bil - psnr_noisy, psnr_gauss, psnr_gauss - psnr_noisy,
               psnr_nlm, psnr_nlm - psnr_noisy, psnr_bm3d, psnr_bm3d - psnr_noisy);

    // ======================= GATE 2: edge_preservation =========================
    const double clean_edge = edge_gradient_mean(h_clean, kW);
    const double noisy_edge = edge_gradient_mean(h_noisy, kW);
    const double bil_edge = edge_gradient_mean(h_bil_naive, kW);
    const double gauss_edge = edge_gradient_mean(h_gauss, kW);
    const double nlm_edge = edge_gradient_mean(h_nlm, kW);
    const double bm3d_edge = edge_gradient_mean(h_bm3d, kW);

    const double frac_bil = bil_edge / clean_edge;
    const double frac_gauss = gauss_edge / clean_edge;
    const double frac_nlm = nlm_edge / clean_edge;
    const double frac_bm3d = bm3d_edge / clean_edge;

    const bool edge_bil = frac_bil >= kEdgePreserveFrac;
    const bool edge_nlm = frac_nlm >= kEdgePreserveFrac;
    const bool edge_bm3d = frac_bm3d >= kEdgePreserveFrac;
    const bool edge_gauss_fails = frac_gauss < kEdgePreserveFrac;   // the NEGATIVE CONTROL: must FAIL, and we assert it
    const bool gate_edge = edge_bil && edge_nlm && edge_bm3d && edge_gauss_fails;

    std::printf("GATE edge_preservation: %s\n", gate_edge ? "PASS" : "FAIL");
    std::printf("[info] edge_preservation: clean step |dI|=%.1f DN | noisy=%.1f DN (%.0f%%) | "
               "bilateral=%.1f DN (%.0f%%, need >=%.0f%%) | nlm=%.1f DN (%.0f%%, need >=%.0f%%) | "
               "bm3d_lite=%.1f DN (%.0f%%, need >=%.0f%%) | GAUSSIAN BASELINE=%.1f DN (%.0f%%, "
               "REQUIRED to stay below %.0f%% — the negative control)\n",
               clean_edge, noisy_edge, 100.0 * noisy_edge / clean_edge,
               bil_edge, 100.0 * frac_bil, 100.0 * kEdgePreserveFrac,
               nlm_edge, 100.0 * frac_nlm, 100.0 * kEdgePreserveFrac,
               bm3d_edge, 100.0 * frac_bm3d, 100.0 * kEdgePreserveFrac,
               gauss_edge, 100.0 * frac_gauss, 100.0 * kEdgePreserveFrac);
    if (!edge_gauss_fails) {
        std::printf("[info] WARNING: the Gaussian-blur negative control did NOT fail edge_preservation "
                   "as designed — this is a real gate failure, not a quirk (see README/THEORY)\n");
    }

    // ======================= GATE 3: flat_noise_floor ===========================
    // NOTE: `rect` here is the ERODED (measurement) rectangle, not the
    // painted patch — see erode_rect()'s comment above.
    struct FlatPatch { const char* name; Rect rect; float dn; };
    const FlatPatch patches[3] = {
        { "dark", erode_rect(kFlatDark, kFlatMeasureMargin), kFlatDarkDn },
        { "mid", erode_rect(kFlatMid, kFlatMeasureMargin), kFlatMidDn },
        { "bright", erode_rect(kFlatBright, kFlatMeasureMargin), kFlatBrightDn },
    };

    bool gate_flat = true;
    double worst_ratio_bil = 0.0, worst_ratio_gauss = 0.0, worst_ratio_nlm = 0.0, worst_ratio_bm3d = 0.0;
    std::vector<CsvRow> flat_rows;
    for (const auto& p : patches) {
        const double noisy_std = residual_std_in_rect(h_noisy, kW, p.rect, p.dn);
        const double bil_std = residual_std_in_rect(h_bil_naive, kW, p.rect, p.dn);
        const double gauss_std = residual_std_in_rect(h_gauss, kW, p.rect, p.dn);
        const double nlm_std = residual_std_in_rect(h_nlm, kW, p.rect, p.dn);
        const double bm3d_std = residual_std_in_rect(h_bm3d, kW, p.rect, p.dn);

        const double r_bil = bil_std / noisy_std, r_gauss = gauss_std / noisy_std;
        const double r_nlm = nlm_std / noisy_std, r_bm3d = bm3d_std / noisy_std;
        worst_ratio_bil = std::max(worst_ratio_bil, r_bil);
        worst_ratio_gauss = std::max(worst_ratio_gauss, r_gauss);
        worst_ratio_nlm = std::max(worst_ratio_nlm, r_nlm);
        worst_ratio_bm3d = std::max(worst_ratio_bm3d, r_bm3d);

        const bool ok = (r_bil <= kFlatNoiseFloorFrac) && (r_gauss <= kFlatNoiseFloorFrac)
                      && (r_nlm <= kFlatNoiseFloorFrac) && (r_bm3d <= kFlatNoiseFloorFrac);
        gate_flat = gate_flat && ok;
        std::printf("[info] flat_noise_floor(%s, clean=%.0f DN): noisy std=%.2f | bilateral=%.2f "
                   "(%.0f%%) | gaussian=%.2f (%.0f%%) | nlm=%.2f (%.0f%%) | bm3d_lite=%.2f (%.0f%%) "
                   "(all must be <= %.0f%% of noisy)\n",
                   p.name, static_cast<double>(p.dn), noisy_std, bil_std, 100.0 * r_bil,
                   gauss_std, 100.0 * r_gauss, nlm_std, 100.0 * r_nlm, bm3d_std, 100.0 * r_bm3d,
                   100.0 * kFlatNoiseFloorFrac);
        flat_rows.push_back({ "flat_noise_floor", std::string("noisy_std_") + p.name, fmt(noisy_std, 2), "n/a (baseline)", "n/a" });
        flat_rows.push_back({ "flat_noise_floor", std::string("bilateral_ratio_") + p.name, fmt(r_bil, 3), fmt(kFlatNoiseFloorFrac, 2), (r_bil <= kFlatNoiseFloorFrac) ? "PASS" : "FAIL" });
        flat_rows.push_back({ "flat_noise_floor", std::string("gaussian_ratio_") + p.name, fmt(r_gauss, 3), fmt(kFlatNoiseFloorFrac, 2), (r_gauss <= kFlatNoiseFloorFrac) ? "PASS" : "FAIL" });
        flat_rows.push_back({ "flat_noise_floor", std::string("nlm_ratio_") + p.name, fmt(r_nlm, 3), fmt(kFlatNoiseFloorFrac, 2), (r_nlm <= kFlatNoiseFloorFrac) ? "PASS" : "FAIL" });
        flat_rows.push_back({ "flat_noise_floor", std::string("bm3d_lite_ratio_") + p.name, fmt(r_bm3d, 3), fmt(kFlatNoiseFloorFrac, 2), (r_bm3d <= kFlatNoiseFloorFrac) ? "PASS" : "FAIL" });
    }
    std::printf("GATE flat_noise_floor: %s\n", gate_flat ? "PASS" : "FAIL");

    // ======================= GATE 4: method_ordering (reported, not forced) ====
    const double psnr_noisy_tex = psnr_db(mse_roi(h_noisy, h_clean, kW, kTextureRoi));
    const double psnr_bil_tex = psnr_db(mse_roi(h_bil_naive, h_clean, kW, kTextureRoi));
    const double psnr_nlm_tex = psnr_db(mse_roi(h_nlm, h_clean, kW, kTextureRoi));
    const double psnr_bm3d_tex = psnr_db(mse_roi(h_bm3d, h_clean, kW, kTextureRoi));

    const bool order_sane = (psnr_bil_tex > psnr_noisy_tex) && (psnr_nlm_tex > psnr_noisy_tex) && (psnr_bm3d_tex > psnr_noisy_tex);
    const bool typical_order = (psnr_bm3d_tex >= psnr_nlm_tex) && (psnr_nlm_tex >= psnr_bil_tex);
    const bool gate_order = order_sane;   // a sanity gate (all three genuinely improve texture PSNR); ORDER is reported, not forced

    std::printf("GATE method_ordering: %s\n", gate_order ? "PASS" : "FAIL");
    std::printf("[info] method_ordering (texture-ROI PSNR, dB): noisy=%.2f | bilateral=%.2f | "
               "nlm=%.2f | bm3d_lite=%.2f -- measured order %s typical expectation "
               "(bm3d_lite >= nlm >= bilateral)\n",
               psnr_noisy_tex, psnr_bil_tex, psnr_nlm_tex, psnr_bm3d_tex,
               typical_order ? "MATCHES the" : "DIFFERS from the");

    // ======================= GATE 5: noise_model_sanity =========================
    bool gate_noise_sanity = true;
    for (const auto& p : patches) {
        const double measured = residual_std_in_rect(h_noisy, kW, p.rect, p.dn);
        const double predicted = static_cast<double>(predicted_noise_std_dn(p.dn));
        const double ratio = measured / predicted;
        const bool ok = (ratio >= kNoiseSanityLo) && (ratio <= kNoiseSanityHi);
        gate_noise_sanity = gate_noise_sanity && ok;
        std::printf("[info] noise_model_sanity(%s, clean=%.0f DN): measured std=%.2f DN | "
                   "predicted std=%.2f DN | ratio=%.3f (band [%.2f, %.2f])\n",
                   p.name, static_cast<double>(p.dn), measured, predicted, ratio, kNoiseSanityLo, kNoiseSanityHi);
    }
    std::printf("GATE noise_model_sanity: %s\n", gate_noise_sanity ? "PASS" : "FAIL");

    // ======================= ARTIFACTS ===========================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();

    artifact_ok = artifact_ok && write_pgm(out_dir + "/clean.pgm", kW, kH, clean_u8);
    artifact_ok = artifact_ok && write_pgm(out_dir + "/noisy.pgm", kW, kH, noisy_u8);
    artifact_ok = artifact_ok && write_pgm(out_dir + "/denoised_bilateral.pgm", kW, kH, dn_to_pgm(h_bil_naive));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/denoised_nlm.pgm", kW, kH, dn_to_pgm(h_nlm));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/denoised_bm3d_lite.pgm", kW, kH, dn_to_pgm(h_bm3d));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/gaussian_baseline.pgm", kW, kH, dn_to_pgm(h_gauss));

    double rms_bil = 0.0, rms_gauss = 0.0, rms_nlm = 0.0, rms_bm3d = 0.0;
    artifact_ok = artifact_ok && write_pgm(out_dir + "/residual_bilateral.pgm", kW, kH, residual_to_pgm(h_bil_naive, h_clean, rms_bil));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/residual_gaussian_baseline.pgm", kW, kH, residual_to_pgm(h_gauss, h_clean, rms_gauss));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/residual_nlm.pgm", kW, kH, residual_to_pgm(h_nlm, h_clean, rms_nlm));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/residual_bm3d_lite.pgm", kW, kH, residual_to_pgm(h_bm3d, h_clean, rms_bm3d));
    std::printf("[info] residual heatmaps: mid-gray=0 error, +-%.0f DN maps to black/white | "
               "RMS residual: bilateral=%.2f gaussian=%.2f nlm=%.2f bm3d_lite=%.2f DN\n",
               static_cast<double>(kResidualCap), rms_bil, rms_gauss, rms_nlm, rms_bm3d);

    std::vector<CsvRow> csv;
    csv.push_back({ "verify", "bilateral_max_abs_diff", fmt(diff_bil, 4), fmt(kTolBilateral, 2), verify_bil ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "bilateral_tiling_max_abs_diff", fmt(diff_tiling, 6), fmt(kTolTiling, 1), verify_tiling ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "gaussian_max_abs_diff", fmt(diff_gauss, 4), fmt(kTolGaussian, 2), verify_gauss ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "nlm_max_abs_diff", fmt(diff_nlm, 4), fmt(kTolNlm, 2), verify_nlm ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "bm3d_lite_max_abs_diff", fmt(diff_bm3d, 4), fmt(kTolBm3d, 2), verify_bm3d ? "PASS" : "FAIL" });
    csv.push_back({ "psnr_improvement", "psnr_noisy_db", fmt(psnr_noisy, 2), "n/a (baseline)", "n/a" });
    csv.push_back({ "psnr_improvement", "psnr_bilateral_db", fmt(psnr_bil, 2), fmt(psnr_noisy + kMinPsnrImprovementDb, 2), imp_bil ? "PASS" : "FAIL" });
    csv.push_back({ "psnr_improvement", "psnr_gaussian_db", fmt(psnr_gauss, 2), fmt(psnr_noisy + kMinPsnrImprovementDb, 2), imp_gauss ? "PASS" : "FAIL" });
    csv.push_back({ "psnr_improvement", "psnr_nlm_db", fmt(psnr_nlm, 2), fmt(psnr_noisy + kMinPsnrImprovementDb, 2), imp_nlm ? "PASS" : "FAIL" });
    csv.push_back({ "psnr_improvement", "psnr_bm3d_lite_db", fmt(psnr_bm3d, 2), fmt(psnr_noisy + kMinPsnrImprovementDb, 2), imp_bm3d ? "PASS" : "FAIL" });
    csv.push_back({ "edge_preservation", "clean_edge_dn", fmt(clean_edge, 1), "n/a (reference)", "n/a" });
    csv.push_back({ "edge_preservation", "bilateral_frac", fmt(frac_bil, 3), fmt(kEdgePreserveFrac, 2), edge_bil ? "PASS" : "FAIL" });
    csv.push_back({ "edge_preservation", "nlm_frac", fmt(frac_nlm, 3), fmt(kEdgePreserveFrac, 2), edge_nlm ? "PASS" : "FAIL" });
    csv.push_back({ "edge_preservation", "bm3d_lite_frac", fmt(frac_bm3d, 3), fmt(kEdgePreserveFrac, 2), edge_bm3d ? "PASS" : "FAIL" });
    csv.push_back({ "edge_preservation", "gaussian_baseline_frac_MUST_FAIL", fmt(frac_gauss, 3), std::string("< ") + fmt(kEdgePreserveFrac, 2), edge_gauss_fails ? "PASS" : "FAIL" });
    for (const auto& r : flat_rows) csv.push_back(r);
    csv.push_back({ "method_ordering", "psnr_bilateral_tex_db", fmt(psnr_bil_tex, 2), "n/a (reported)", order_sane ? "PASS" : "FAIL" });
    csv.push_back({ "method_ordering", "psnr_nlm_tex_db", fmt(psnr_nlm_tex, 2), "n/a (reported)", "n/a" });
    csv.push_back({ "method_ordering", "psnr_bm3d_lite_tex_db", fmt(psnr_bm3d_tex, 2), "n/a (reported)", "n/a" });
    csv.push_back({ "method_ordering", "matches_typical_order", typical_order ? "yes" : "no", "n/a (informational)", "n/a" });
    for (const auto& p : patches) {
        const double measured = residual_std_in_rect(h_noisy, kW, p.rect, p.dn);
        const double predicted = static_cast<double>(predicted_noise_std_dn(p.dn));
        const double ratio = measured / predicted;
        csv.push_back({ "noise_model_sanity", std::string("ratio_") + p.name, fmt(ratio, 3),
                       fmt(kNoiseSanityLo, 2) + ".." + fmt(kNoiseSanityHi, 2),
                       (ratio >= kNoiseSanityLo && ratio <= kNoiseSanityHi) ? "PASS" : "FAIL" });
    }
    artifact_ok = artifact_ok && write_gates_csv(out_dir + "/gates_metrics.csv", csv);

    if (artifact_ok) {
        std::printf("ARTIFACT: wrote demo/out/{clean.pgm, noisy.pgm, denoised_bilateral.pgm, "
                   "denoised_nlm.pgm, denoised_bm3d_lite.pgm, gaussian_baseline.pgm, "
                   "residual_bilateral.pgm, residual_gaussian_baseline.pgm, residual_nlm.pgm, "
                   "residual_bm3d_lite.pgm, gates_metrics.csv}\n");
    } else {
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");
    }

    // ---- free device memory --------------------------------------------------
    CUDA_CHECK(cudaFree(d_noisy));
    CUDA_CHECK(cudaFree(d_bil_naive)); CUDA_CHECK(cudaFree(d_bil_tiled));
    CUDA_CHECK(cudaFree(d_gauss)); CUDA_CHECK(cudaFree(d_nlm)); CUDA_CHECK(cudaFree(d_bm3d));

    // ======================= RESULT =============================================
    const bool all_gates = gate_psnr && gate_edge && gate_flat && gate_order && gate_noise_sanity;
    const bool overall = verify_pass && all_gates && artifact_ok;
    if (overall) {
        std::printf("RESULT: PASS (VERIFY + all 5 gates passed: psnr_improvement, "
                   "edge_preservation, flat_noise_floor, method_ordering, noise_model_sanity)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
