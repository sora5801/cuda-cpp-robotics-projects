// ===========================================================================
// main.cu — entry point for project 01.23
//           Full RAW->RGB ISP: black level -> lens shading -> defect
//           correction -> white balance -> demosaic (MHC + bilinear) ->
//           CCM -> gamma, staged AND fused (stages 1-4), two illuminants
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed synthetic sample: two RAW10-in-uint16 mosaics
//      (D65, tungsten), the noiseless pre-shading sensor-domain ground
//      truth (D65 only), the reference scene rendering, and the runtime
//      defect list (../scripts/make_synthetic.py's outputs).
//   2. Run the STAGED pipeline (four separate kernels) through stages 1-4
//      on BOTH illuminants; compute AWB gains (gray-world AND white-patch)
//      from each; apply gray-world's gain as this project's DEFAULT.
//   3. Run the FUSED kernel (stages 1-4 in one launch) on the D65 mosaic
//      with the SAME gain, for the fused_vs_staged comparison.
//   4. Demosaic (MHC — the production path; bilinear + a SEPARATE MHC pass
//      on the pre-white-balance mosaic — the isolated demosaic-quality
//      gate), CCM, and gamma to produce the final sRGB artifacts: D65
//      (correct AWB), tungsten (correct AWB), and tungsten processed with
//      the WRONG (D65) gains — the negative control.
//   5. VERIFY: every stage's GPU output compared against reference_cpu.cpp's
//      independent twin.
//   6. TEN GATES, each checking something the twin comparison cannot (see
//      kernels.cuh/reference_cpu.cpp's twin-independence ruling):
//        black_level_residual, shading_flatness, defect_recovery,
//        demosaic_psnr, awb_accuracy, awb_red_crop_failure, ccm_color_chart,
//        tungsten_wrong_awb_negative_control, end_to_end_psnr, fused_vs_staged.
//   7. ARTIFACTS: demo/out/{raw_vis_d65.pgm, shading_corrected_d65.pgm,
//      demosaiced_mhc_d65.ppm, demosaiced_bilinear_d65.ppm,
//      white_balanced_d65.ppm, final_d65.ppm, final_tungsten.ppm,
//      final_tungsten_wrong_awb.ppm, chart_crop_d65.ppm, gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", the ten "GATE <name>:" verdict lines, "VERIFY:", "ARTIFACT:", and
// "RESULT:" — every one a PASS/FAIL verdict with NO embedded numbers, so it
// is identical on every GPU architecture. Measured numbers live on
// "[info]"/"[time]" lines, deliberately NOT diffed by demo/run_demo.* (this
// project's floating-point pipeline can differ by a few ULP across
// sm_75/sm_86/sm_89 — THEORY.md "Numerical considerations"). Change a stable
// line => update demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Gate tolerances. Every number below carries either a physical argument
// (documented inline) or a floor/ceiling calibrated from an ACTUAL measured
// run on this project's committed sample (CLAUDE.md section 8: never
// fabricate), with margin so the gate stays robust to legitimate cross-GPU
// float differences (FMA-contraction choices differ by architecture — see
// the output-contract note above). Measured values are recorded in
// THEORY.md "How we verify correctness" and README "Expected output".
// ===========================================================================
static constexpr double kTolStageFloat  = 5e-4;      // GPU-vs-CPU per-pixel float stage tolerance (FMA-class drift)
static constexpr double kTolAwbGain     = 5e-4;      // GPU-vs-CPU AWB gain tolerance
static constexpr double kTolRgbFloat    = 5e-4;      // demosaic/CCM GPU-vs-CPU float tolerance
static constexpr double kTolSrgb8       = 1.5;       // gamma-encoded uint8 GPU-vs-CPU tolerance (rounding)

// -- Physical/statistical gate tolerances -- calibrated with margin once
// this project's ACTUAL measured numbers are known (filled in after the
// first real run; see README "Expected output" for the measured values
// these floors/ceilings sit against).
static constexpr double kBlackLevelResidualTol   = 0.03;   // mean |residual|, normalized [0,1] units
static constexpr double kShadingFlatnessTol      = 0.03;   // |mean(inner)-mean(outer)| radial bands
static constexpr double kShadingAccuracyTol      = 0.03;   // mean |corrected - truth|, normalized units
static constexpr double kDefectRecoveryTol       = 0.05;   // mean |corrected - truth| at defect sites
static constexpr double kDefectFalseCorrectionTol = 1e-5;  // non-defect pixels must be UNCHANGED
static constexpr double kDemosaicPsnrMarginDb    = 1.0;    // MHC PSNR must exceed bilinear by >= this many dB (measured ~1.5 dB)
static constexpr double kAwbAccuracyTolFrac      = 0.26;   // relative gain error vs truth (measured worst ~0.23, tungsten's R/B gains)
static constexpr double kAwbRedCropFailMinFrac   = 0.15;   // red-crop gray-world MUST deviate by more than this (negative control)
static constexpr double kCcmChartMeanTol         = 18.0;   // mean per-patch RGB distance, 0..255 scale
static constexpr double kCcmChartMaxTol          = 40.0;   // max per-patch RGB distance, 0..255 scale
static constexpr double kWrongAwbInflationMinX   = 1.5;    // wrong-gain chart error must be >= this x correct-gain error
static constexpr double kEndToEndPsnrFloorD65      = 24.0; // dB (measured ~26.4)
static constexpr double kEndToEndPsnrFloorTungsten = 21.5; // dB (measured ~23.5; AWB is imperfect -> a real, smaller, honest floor)
static constexpr double kFusedVsStagedTol        = 1e-4;   // fused and staged should nearly bit-match (see kernels.cu header)

// ===========================================================================
// Minimal binary/PGM/PPM/CSV I/O — same "only ever reads files this
// project's own generator wrote" discipline as every sibling project.
// ===========================================================================
static bool read_u16_raw(const std::string& path, int n, std::vector<uint16_t>& out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    out.resize(static_cast<size_t>(n));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(uint16_t)));
    return in.gcount() == static_cast<std::streamsize>(out.size() * sizeof(uint16_t));
}
static bool read_f32_raw(const std::string& path, int n, std::vector<float>& out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    out.resize(static_cast<size_t>(n));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return in.gcount() == static_cast<std::streamsize>(out.size() * sizeof(float));
}
static bool read_ppm(const std::string& path, int& W, int& H, std::vector<unsigned char>& data)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    std::string magic;
    in >> magic;
    if (magic != "P6") return false;
    auto read_int = [&](int& v) -> bool {
        for (;;) {
            const int c = in.peek();
            if (c == '#') { std::string line; std::getline(in, line); continue; }
            if (c != EOF && std::isspace(c)) { in.get(); continue; }
            break;
        }
        in >> v;
        return static_cast<bool>(in);
    };
    int maxval = 0;
    if (!read_int(W) || !read_int(H) || !read_int(maxval)) return false;
    if (maxval != 255 || W <= 0 || H <= 0) return false;
    in.get();
    data.resize(static_cast<size_t>(W) * H * 3);
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
static bool read_defect_csv(const std::string& path, std::vector<int>& xs, std::vector<int>& ys)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    std::getline(in, line);   // header: "x,y,kind"
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string xs_, ys_, kind_;
        if (!std::getline(ss, xs_, ',')) continue;
        if (!std::getline(ss, ys_, ',')) continue;
        std::getline(ss, kind_, ',');   // documentation only, unused by the correction algorithm
        xs.push_back(std::atoi(xs_.c_str()));
        ys.push_back(std::atoi(ys_.c_str()));
    }
    return true;
}

// ===========================================================================
// Small numeric helpers shared by several gates.
// ===========================================================================
static double psnr_f32(const std::vector<float>& a, const std::vector<float>& b, double peak)
{
    double se = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) { const double d = static_cast<double>(a[i]) - b[i]; se += d * d; }
    const double mse = se / static_cast<double>(n);
    if (mse <= 1e-14) return 99.0;
    return 10.0 * std::log10((peak * peak) / mse);
}
static double psnr_u8(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b)
{
    double se = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) { const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]); se += d * d; }
    const double mse = se / static_cast<double>(n);
    if (mse <= 1e-9) return 99.0;
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}
template <typename T>
static double max_abs_diff(const std::vector<T>& a, const std::vector<T>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m) m = d;
    }
    return m;
}
struct RGBd { double r, g, b; };
static RGBd patch_mean_srgb8(const std::vector<unsigned char>& img, int W, int x0, int y0, int size, int margin)
{
    double sr = 0, sg = 0, sb = 0; long long n = 0;
    for (int y = y0 + margin; y < y0 + size - margin; ++y) {
        for (int x = x0 + margin; x < x0 + size - margin; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 3;
            sr += img[i]; sg += img[i + 1]; sb += img[i + 2]; ++n;
        }
    }
    return { sr / n, sg / n, sb / n };
}
static double rgb_dist(RGBd a, double r, double g, double b)
{
    const double dr = a.r - r, dg = a.g - g, db = a.b - b;
    return std::sqrt(dr * dr + dg * dg + db * db);
}
static bool at_defect(int x, int y, const std::vector<int>& dx, const std::vector<int>& dy)
{
    for (size_t k = 0; k < dx.size(); ++k) if (dx[k] == x && dy[k] == y) return true;
    return false;
}

// gates_metrics.csv writer
struct CsvRow { std::string gate, metric, value, tol, pass; };
static std::string fmt(double v, int prec = 6) { char b[64]; std::snprintf(b, sizeof(b), "%.*f", prec, v); return std::string(b); }
static bool write_gates_csv(const std::string& path, const std::vector<CsvRow>& rows)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "gate,metric,value,tolerance,pass\n";
    for (const auto& r : rows) out << r.gate << "," << r.metric << "," << r.value << "," << r.tol << "," << r.pass << "\n";
    return static_cast<bool>(out);
}

// ===========================================================================
// StagedMosaic — device buffers for stages 1-3's output (BL, BL+shading,
// BL+shading+defect) on ONE raw mosaic. A tiny RAII-ish struct so the D65
// and tungsten runs (identical stage sequence) share one code path instead
// of two copy-pasted kernel-launch blocks.
// ===========================================================================
struct StagedMosaic {
    float* d_bl = nullptr;
    float* d_sh = nullptr;
    float* d_dc = nullptr;
};
static StagedMosaic run_stages_1_to_3(const uint16_t* d_raw, int W, int H, int defect_count)
{
    StagedMosaic s;
    const size_t bytes = static_cast<size_t>(W) * H * sizeof(float);
    CUDA_CHECK(cudaMalloc(&s.d_bl, bytes));
    CUDA_CHECK(cudaMalloc(&s.d_sh, bytes));
    CUDA_CHECK(cudaMalloc(&s.d_dc, bytes));
    launch_black_level(d_raw, s.d_bl, W, H);
    launch_lens_shading(s.d_bl, s.d_sh, W, H);
    launch_defect_correct(s.d_sh, s.d_dc, W, H, defect_count);
    return s;
}

// awb_result — gray-world and white-patch gains, both downloaded to host.
struct AwbResult { float gray[3]; float white[3]; };
static AwbResult run_awb(const float* d_dc, int W, int H)
{
    const int n = W * H;
    const int num_blocks = (n + 255) / 256;
    double* d_block_sum3 = nullptr; float* d_block_max3 = nullptr;
    float* d_gray = nullptr; float* d_white = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sum3, static_cast<size_t>(num_blocks) * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_max3, static_cast<size_t>(num_blocks) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_white, 3 * sizeof(float)));
    launch_awb_stats_block(d_dc, W, H, d_block_sum3, d_block_max3, num_blocks);
    launch_awb_finalize(d_block_sum3, d_block_max3, num_blocks, W, H, d_gray, d_white);
    AwbResult r;
    CUDA_CHECK(cudaMemcpy(r.gray, d_gray, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.white, d_white, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_block_sum3)); CUDA_CHECK(cudaFree(d_block_max3));
    CUDA_CHECK(cudaFree(d_gray)); CUDA_CHECK(cudaFree(d_white));
    return r;
}

// finish_pipeline — white balance -> MHC demosaic -> CCM -> gamma, given a
// pre-white-balance mosaic and a gain triple. Returns the final sRGB8 image
// (host vector) plus (optionally) the intermediate white-balanced mosaic
// and linear-RGB CCM output, downloaded for artifacts/gates when requested.
struct FinishedPipeline {
    std::vector<float> wb_mosaic;      // W*H
    std::vector<float> rgb_mhc;        // W*H*3 (post demosaic, pre-CCM)
    std::vector<float> rgb_ccm;        // W*H*3 (post CCM, linear, pre-gamma)
    std::vector<unsigned char> srgb8;  // W*H*3 (final artifact)
    float wb_gpu_ms = 0, demosaic_gpu_ms = 0, ccm_gpu_ms = 0, gamma_gpu_ms = 0;
};
static FinishedPipeline finish_pipeline(const float* d_dc, int W, int H, const float gain[3])
{
    const int n = W * H;
    const size_t bytes1 = static_cast<size_t>(n) * sizeof(float);
    const size_t bytes3 = static_cast<size_t>(n) * 3 * sizeof(float);
    float *d_wb = nullptr, *d_rgb = nullptr, *d_ccm = nullptr;
    unsigned char* d_srgb8 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wb, bytes1));
    CUDA_CHECK(cudaMalloc(&d_rgb, bytes3));
    CUDA_CHECK(cudaMalloc(&d_ccm, bytes3));
    CUDA_CHECK(cudaMalloc(&d_srgb8, static_cast<size_t>(n) * 3));

    GpuTimer t1; t1.begin();
    launch_white_balance(d_dc, d_wb, W, H, gain[0], gain[1], gain[2]);
    const float wb_ms = t1.end_ms();

    GpuTimer t2; t2.begin();
    launch_demosaic_mhc(d_wb, d_rgb, W, H);
    const float dem_ms = t2.end_ms();

    GpuTimer t3; t3.begin();
    launch_ccm_apply(d_rgb, d_ccm, W, H);
    const float ccm_ms = t3.end_ms();

    GpuTimer t4; t4.begin();
    launch_gamma_encode(d_ccm, d_srgb8, W, H);
    const float gam_ms = t4.end_ms();

    FinishedPipeline out;
    out.wb_mosaic.resize(n);
    out.rgb_mhc.resize(static_cast<size_t>(n) * 3);
    out.rgb_ccm.resize(static_cast<size_t>(n) * 3);
    out.srgb8.resize(static_cast<size_t>(n) * 3);
    CUDA_CHECK(cudaMemcpy(out.wb_mosaic.data(), d_wb, bytes1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.rgb_mhc.data(), d_rgb, bytes3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.rgb_ccm.data(), d_ccm, bytes3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.srgb8.data(), d_srgb8, static_cast<size_t>(n) * 3, cudaMemcpyDeviceToHost));
    out.wb_gpu_ms = wb_ms; out.demosaic_gpu_ms = dem_ms; out.ccm_gpu_ms = ccm_ms; out.gamma_gpu_ms = gam_ms;

    CUDA_CHECK(cudaFree(d_wb)); CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_ccm)); CUDA_CHECK(cudaFree(d_srgb8));
    return out;
}

// ===========================================================================
// main.
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--data path/to/data/sample]\n", argv[0]); return 2; }
    }

    std::printf("[demo] full RAW->RGB ISP: black level -> lens shading -> defect correction -> "
               "white balance -> demosaic (MHC + bilinear) -> CCM -> gamma, staged vs fused, "
               "two illuminants (project 01.23)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d RAW10-in-uint16 RGGB mosaic, black=%d white=%d, "
               "shading V(r)=1%+.2fr^2%+.2fr^4, CCM derived from a 3x3 spectral crosstalk matrix, "
               "MHC 5x5 demosaic, D65 + tungsten illuminants\n",
               kRawW, kRawH, kBlackLevel, kWhiteLevel, kShadeA2, kShadeA4);

    // ---- data ---------------------------------------------------------------
    const int n = kRawW * kRawH;
    std::vector<uint16_t> raw_d65, raw_tungsten;
    std::vector<float> true_sensor_rgb_d65;
    std::vector<unsigned char> true_scene_srgb; int tsW = 0, tsH = 0;
    std::vector<int> defect_x, defect_y;

    const std::string p_raw_d65 = find_data_file(data_dir, argv[0], "raw_mosaic_d65.bin");
    const std::string p_raw_tng = find_data_file(data_dir, argv[0], "raw_mosaic_tungsten.bin");
    const std::string p_truth_sensor = find_data_file(data_dir, argv[0], "true_sensor_rgb_d65.bin");
    const std::string p_truth_scene = find_data_file(data_dir, argv[0], "true_scene_srgb.ppm");
    const std::string p_defects = find_data_file(data_dir, argv[0], "defect_list.csv");

    bool ok = !p_raw_d65.empty() && !p_raw_tng.empty() && !p_truth_sensor.empty()
            && !p_truth_scene.empty() && !p_defects.empty();
    ok = ok && read_u16_raw(p_raw_d65, n, raw_d65)
            && read_u16_raw(p_raw_tng, n, raw_tungsten)
            && read_f32_raw(p_truth_sensor, n * 3, true_sensor_rgb_d65)
            && read_ppm(p_truth_scene, tsW, tsH, true_scene_srgb)
            && read_defect_csv(p_defects, defect_x, defect_y);
    ok = ok && tsW == kRawW && tsH == kRawH;

    if (!ok) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    const int defect_count = static_cast<int>(defect_x.size());
    std::printf("DATA: %dx%d synthetic RGGB scene (24-patch chart + AWB card + hashed texture), "
               "%d committed defects, two illuminants [synthetic, seed 42]\n", kRawW, kRawH, defect_count);

    // ---- upload defect list to device constant memory -----------------------
    upload_defect_list(defect_x.data(), defect_y.data(), defect_count);

    // ---- upload raw mosaics --------------------------------------------------
    uint16_t *d_raw_d65 = nullptr, *d_raw_tng = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw_d65, static_cast<size_t>(n) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_raw_tng, static_cast<size_t>(n) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_raw_d65, raw_d65.data(), static_cast<size_t>(n) * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_raw_tng, raw_tungsten.data(), static_cast<size_t>(n) * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // ======================= STAGED stages 1-3, both illuminants ===============
    GpuTimer gt_stage_d65; gt_stage_d65.begin();
    StagedMosaic d65 = run_stages_1_to_3(d_raw_d65, kRawW, kRawH, defect_count);
    const float staged123_d65_ms = gt_stage_d65.end_ms();

    GpuTimer gt_stage_tng; gt_stage_tng.begin();
    StagedMosaic tng = run_stages_1_to_3(d_raw_tng, kRawW, kRawH, defect_count);
    const float staged123_tng_ms = gt_stage_tng.end_ms();

    // ======================= AWB, both illuminants ==============================
    AwbResult awb_d65 = run_awb(d65.d_dc, kRawW, kRawH);
    AwbResult awb_tng = run_awb(tng.d_dc, kRawW, kRawH);
    std::printf("[info] AWB D65:      gray-world gain=(%.4f,%.4f,%.4f) | white-patch gain=(%.4f,%.4f,%.4f) | truth=(%.4f,%.4f,%.4f)\n",
               awb_d65.gray[0], awb_d65.gray[1], awb_d65.gray[2], awb_d65.white[0], awb_d65.white[1], awb_d65.white[2],
               kTrueAwbGainD65[0], kTrueAwbGainD65[1], kTrueAwbGainD65[2]);
    std::printf("[info] AWB tungsten: gray-world gain=(%.4f,%.4f,%.4f) | white-patch gain=(%.4f,%.4f,%.4f) | truth=(%.4f,%.4f,%.4f)\n",
               awb_tng.gray[0], awb_tng.gray[1], awb_tng.gray[2], awb_tng.white[0], awb_tng.white[1], awb_tng.white[2],
               kTrueAwbGainTungsten[0], kTrueAwbGainTungsten[1], kTrueAwbGainTungsten[2]);

    // ======================= WB stage 4 (staged) + FUSED (D65 only) ============
    float* d_wb_d65 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wb_d65, static_cast<size_t>(n) * sizeof(float)));
    launch_white_balance(d65.d_dc, d_wb_d65, kRawW, kRawH, awb_d65.gray[0], awb_d65.gray[1], awb_d65.gray[2]);

    float* d_fused_d65 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fused_d65, static_cast<size_t>(n) * sizeof(float)));
    GpuTimer gt_fused; gt_fused.begin();
    launch_fused_bl_shading_defect_wb(d_raw_d65, d_fused_d65, kRawW, kRawH, defect_count,
                                      awb_d65.gray[0], awb_d65.gray[1], awb_d65.gray[2]);
    const float fused_d65_ms = gt_fused.end_ms();

    std::vector<float> wb_d65_gpu(n), fused_d65_gpu(n);
    CUDA_CHECK(cudaMemcpy(wb_d65_gpu.data(), d_wb_d65, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fused_d65_gpu.data(), d_fused_d65, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

    const double staged_bytes = 4.0 * n * sizeof(float) + n * sizeof(uint16_t);   // 3 intermediate writes + 1 final write + 1 raw read (idealized, no cache reuse)
    const double fused_bytes = 1.0 * n * sizeof(float) + n * sizeof(uint16_t);    // 1 final write + 1 raw read
    std::printf("[time] stages1-4 staged: %.3f ms | fused: %.3f ms\n",
               static_cast<double>(staged123_d65_ms), static_cast<double>(fused_d65_ms));
    std::printf("[info] memory traffic, stages1-4 (derived, idealized no-cache-reuse model): "
               "staged = %.0f bytes | fused = %.0f bytes | savings = %.1f%%\n",
               staged_bytes, fused_bytes, 100.0 * (staged_bytes - fused_bytes) / staged_bytes);

    // ======================= demosaic-quality gate inputs (D65, PRE-WB) ========
    // Isolates demosaic quality from AWB gain error (see kernels.cuh's file
    // header): demosaic the STAGE-3 mosaic directly, compare against
    // true_sensor_rgb_d65 (also pre-WB) instead of the WB-applied mosaic.
    float *d_rgb_bilinear_gate = nullptr, *d_rgb_mhc_gate = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb_bilinear_gate, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb_mhc_gate, static_cast<size_t>(n) * 3 * sizeof(float)));
    launch_demosaic_bilinear(d65.d_dc, d_rgb_bilinear_gate, kRawW, kRawH);
    launch_demosaic_mhc(d65.d_dc, d_rgb_mhc_gate, kRawW, kRawH);
    std::vector<float> rgb_bilinear_gate(static_cast<size_t>(n) * 3), rgb_mhc_gate(static_cast<size_t>(n) * 3);
    CUDA_CHECK(cudaMemcpy(rgb_bilinear_gate.data(), d_rgb_bilinear_gate, rgb_bilinear_gate.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rgb_mhc_gate.data(), d_rgb_mhc_gate, rgb_mhc_gate.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ======================= finish the three production pipelines =============
    FinishedPipeline fin_d65 = finish_pipeline(d65.d_dc, kRawW, kRawH, awb_d65.gray);
    FinishedPipeline fin_tng_correct = finish_pipeline(tng.d_dc, kRawW, kRawH, awb_tng.gray);
    const float wrong_gain[3] = { kTrueAwbGainD65[0], kTrueAwbGainD65[1], kTrueAwbGainD65[2] };   // (1,1,1): tungsten data, D65 gains
    FinishedPipeline fin_tng_wrong = finish_pipeline(tng.d_dc, kRawW, kRawH, wrong_gain);

    // ======================= download stage buffers needed for VERIFY/gates ===
    std::vector<float> bl_d65_gpu(n), sh_d65_gpu(n), dc_d65_gpu(n);
    CUDA_CHECK(cudaMemcpy(bl_d65_gpu.data(), d65.d_bl, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sh_d65_gpu.data(), d65.d_sh, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dc_d65_gpu.data(), d65.d_dc, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

    // ======================= VERIFY: GPU vs CPU, every stage ====================
    bool verify_pass = true;
    CpuTimer cpu_timer; cpu_timer.begin();
    {
        std::vector<float> bl_cpu(n), sh_cpu(n), dc_cpu(n);
        black_level_cpu(raw_d65.data(), bl_cpu.data(), kRawW, kRawH);
        double d = max_abs_diff(bl_d65_gpu, bl_cpu);
        std::printf("[info] verify(black_level): max|gpu-cpu| = %.6f (tol %.4f)\n", d, kTolStageFloat);
        if (d > kTolStageFloat) verify_pass = false;

        lens_shading_cpu(bl_cpu.data(), sh_cpu.data(), kRawW, kRawH);
        d = max_abs_diff(sh_d65_gpu, sh_cpu);
        std::printf("[info] verify(lens_shading): max|gpu-cpu| = %.6f (tol %.4f)\n", d, kTolStageFloat);
        if (d > kTolStageFloat) verify_pass = false;

        defect_correct_cpu(sh_cpu.data(), dc_cpu.data(), kRawW, kRawH, defect_x.data(), defect_y.data(), defect_count);
        d = max_abs_diff(dc_d65_gpu, dc_cpu);
        std::printf("[info] verify(defect_correct): max|gpu-cpu| = %.6f (tol %.4f)\n", d, kTolStageFloat);
        if (d > kTolStageFloat) verify_pass = false;

        double sum3[3]; float max3[3];
        awb_stats_cpu(dc_cpu.data(), kRawW, kRawH, sum3, max3);
        float gray_cpu[3], white_cpu[3];
        awb_gains_from_stats_cpu(sum3, max3, kRawW, kRawH, gray_cpu, white_cpu);
        double dg = std::max({ std::fabs(gray_cpu[0] - awb_d65.gray[0]), std::fabs(gray_cpu[1] - awb_d65.gray[1]), std::fabs(gray_cpu[2] - awb_d65.gray[2]) });
        double dw = std::max({ std::fabs(white_cpu[0] - awb_d65.white[0]), std::fabs(white_cpu[1] - awb_d65.white[1]), std::fabs(white_cpu[2] - awb_d65.white[2]) });
        std::printf("[info] verify(awb gains, D65): max|gpu-cpu| gray=%.6f white=%.6f (tol %.4f)\n", dg, dw, kTolAwbGain);
        if (dg > kTolAwbGain || dw > kTolAwbGain) verify_pass = false;

        std::vector<float> wb_cpu(n);
        white_balance_cpu(dc_cpu.data(), wb_cpu.data(), kRawW, kRawH, awb_d65.gray[0], awb_d65.gray[1], awb_d65.gray[2]);
        d = max_abs_diff(wb_d65_gpu, wb_cpu);
        std::printf("[info] verify(white_balance): max|gpu-cpu| = %.6f (tol %.4f)\n", d, kTolStageFloat);
        if (d > kTolStageFloat) verify_pass = false;

        std::vector<float> fused_cpu(n);
        fused_bl_shading_defect_wb_cpu(raw_d65.data(), fused_cpu.data(), kRawW, kRawH,
                                       defect_x.data(), defect_y.data(), defect_count,
                                       awb_d65.gray[0], awb_d65.gray[1], awb_d65.gray[2]);
        d = max_abs_diff(fused_d65_gpu, fused_cpu);
        std::printf("[info] verify(fused stages1-4): max|gpu-cpu| = %.6f (tol %.4f)\n", d, kTolStageFloat);
        if (d > kTolStageFloat) verify_pass = false;

        std::vector<float> rgb_bilinear_cpu(static_cast<size_t>(n) * 3), rgb_mhc_cpu(static_cast<size_t>(n) * 3);
        demosaic_bilinear_cpu(dc_cpu.data(), rgb_bilinear_cpu.data(), kRawW, kRawH);
        demosaic_mhc_cpu(dc_cpu.data(), rgb_mhc_cpu.data(), kRawW, kRawH);
        d = max_abs_diff(rgb_bilinear_gate, rgb_bilinear_cpu);
        double d2 = max_abs_diff(rgb_mhc_gate, rgb_mhc_cpu);
        std::printf("[info] verify(demosaic bilinear/MHC): max|gpu-cpu| = %.6f / %.6f (tol %.4f)\n", d, d2, kTolRgbFloat);
        if (d > kTolRgbFloat || d2 > kTolRgbFloat) verify_pass = false;

        std::vector<float> ccm_cpu(static_cast<size_t>(n) * 3);
        ccm_apply_cpu(fin_d65.rgb_mhc.data(), ccm_cpu.data(), kRawW, kRawH);
        d = max_abs_diff(fin_d65.rgb_ccm, ccm_cpu);
        std::printf("[info] verify(ccm_apply): max|gpu-cpu| = %.6f (tol %.4f)\n", d, kTolRgbFloat);
        if (d > kTolRgbFloat) verify_pass = false;

        std::vector<unsigned char> srgb8_cpu(static_cast<size_t>(n) * 3);
        gamma_encode_cpu(fin_d65.rgb_ccm.data(), srgb8_cpu.data(), kRawW, kRawH);
        d = max_abs_diff(fin_d65.srgb8, srgb8_cpu);
        std::printf("[info] verify(gamma_encode): max|gpu-cpu| = %.6f (tol %.4f)\n", d, kTolSrgb8);
        if (d > kTolSrgb8) verify_pass = false;
    }
    const double cpu_ms = cpu_timer.end_ms();
    std::printf("[time] full CPU oracle (all stages, D65): %.1f ms\n", cpu_ms);
    std::printf("VERIFY: %s (GPU matches CPU reference within documented per-stage tolerance)\n",
               verify_pass ? "PASS" : "FAIL");

    // ======================= GATES ================================================
    std::vector<CsvRow> csv;

    // -- GATE 1: black_level_residual (D65, non-defect pixels only) --------
    bool gate1;
    {
        double sum_abs = 0.0; long long cnt = 0;
        for (int y = 0; y < kRawH; ++y) for (int x = 0; x < kRawW; ++x) {
            if (at_defect(x, y, defect_x, defect_y)) continue;
            const int i = y * kRawW + x;
            const int ch = phase_to_wb_channel(bayer_phase_at(x, y));
            const float truth_lin = true_sensor_rgb_d65[static_cast<size_t>(i) * 3 + ch];
            float expected = truth_lin * shading_gain_at(x, y);
            expected = std::min(std::max(expected, 0.0f), 1.0f);
            sum_abs += std::fabs(static_cast<double>(bl_d65_gpu[i]) - expected);
            ++cnt;
        }
        const double mean_abs = sum_abs / static_cast<double>(cnt);
        gate1 = mean_abs <= kBlackLevelResidualTol;
        std::printf("GATE black_level_residual: %s\n", gate1 ? "PASS" : "FAIL");
        std::printf("[info] black_level_residual: mean|residual| = %.5f (tol %.3f, n=%lld non-defect px)\n", mean_abs, kBlackLevelResidualTol, cnt);
        csv.push_back({ "black_level_residual", "mean_abs_residual", fmt(mean_abs, 5), fmt(kBlackLevelResidualTol, 3), gate1 ? "PASS" : "FAIL" });
    }

    // -- GATE 2: shading_flatness (D65, inner vs outer radial band) --------
    double flat_inner, flat_outer, shading_accuracy;
    bool gate2;
    {
        double sum_in = 0.0, sum_out = 0.0, sum_acc = 0.0; long long n_in = 0, n_out = 0, n_acc = 0;
        for (int y = 0; y < kRawH; ++y) for (int x = 0; x < kRawW; ++x) {
            if (at_defect(x, y, defect_x, defect_y)) continue;
            const int i = y * kRawW + x;
            const int ch = phase_to_wb_channel(bayer_phase_at(x, y));
            const float truth = true_sensor_rgb_d65[static_cast<size_t>(i) * 3 + ch];
            const double residual = static_cast<double>(sh_d65_gpu[i]) - truth;
            sum_acc += std::fabs(residual); ++n_acc;
            const double dx = x - kShadeCx, dy = y - kShadeCy;
            const double r = std::sqrt(dx * dx + dy * dy) / kShadeRNorm;
            if (r < 0.5) { sum_in += residual; ++n_in; } else { sum_out += residual; ++n_out; }
        }
        flat_inner = sum_in / static_cast<double>(n_in);
        flat_outer = sum_out / static_cast<double>(n_out);
        shading_accuracy = sum_acc / static_cast<double>(n_acc);
        const double flat_gap = std::fabs(flat_inner - flat_outer);
        gate2 = (flat_gap <= kShadingFlatnessTol) && (shading_accuracy <= kShadingAccuracyTol);
        std::printf("GATE shading_flatness: %s\n", gate2 ? "PASS" : "FAIL");
        std::printf("[info] shading_flatness: |mean(inner r<0.5)-mean(outer)| = %.5f (tol %.3f) | mean|residual| overall = %.5f (tol %.3f)\n",
                   flat_gap, kShadingFlatnessTol, shading_accuracy, kShadingAccuracyTol);
        csv.push_back({ "shading_flatness", "inner_outer_gap", fmt(flat_gap, 5), fmt(kShadingFlatnessTol, 3), gate2 ? "PASS" : "FAIL" });
        csv.push_back({ "shading_flatness", "mean_abs_residual", fmt(shading_accuracy, 5), fmt(kShadingAccuracyTol, 3), gate2 ? "PASS" : "FAIL" });
    }

    // -- GATE 3: defect_recovery (recovered value near truth + zero false corrections) --
    bool gate3;
    {
        double sum_abs = 0.0;
        for (int k = 0; k < defect_count; ++k) {
            const int x = defect_x[k], y = defect_y[k];
            const int i = y * kRawW + x;
            const int ch = phase_to_wb_channel(bayer_phase_at(x, y));
            const float truth = true_sensor_rgb_d65[static_cast<size_t>(i) * 3 + ch];
            sum_abs += std::fabs(static_cast<double>(dc_d65_gpu[i]) - truth);
        }
        const double mean_abs = sum_abs / static_cast<double>(defect_count);

        double max_false = 0.0;
        for (int y = 0; y < kRawH; ++y) for (int x = 0; x < kRawW; ++x) {
            if (at_defect(x, y, defect_x, defect_y)) continue;
            const int i = y * kRawW + x;
            max_false = std::max(max_false, std::fabs(static_cast<double>(dc_d65_gpu[i]) - sh_d65_gpu[i]));
        }
        gate3 = (mean_abs <= kDefectRecoveryTol) && (max_false <= kDefectFalseCorrectionTol);
        std::printf("GATE defect_recovery: %s\n", gate3 ? "PASS" : "FAIL");
        std::printf("[info] defect_recovery: mean|corrected-truth| at %d defects = %.5f (tol %.3f) | "
                   "max false-correction on non-defect px = %.2e (tol %.1e)\n",
                   defect_count, mean_abs, kDefectRecoveryTol, max_false, kDefectFalseCorrectionTol);
        csv.push_back({ "defect_recovery", "mean_abs_error_at_defects", fmt(mean_abs, 5), fmt(kDefectRecoveryTol, 3), gate3 ? "PASS" : "FAIL" });
        csv.push_back({ "defect_recovery", "max_false_correction", fmt(max_false, 8), fmt(kDefectFalseCorrectionTol, 6), gate3 ? "PASS" : "FAIL" });
    }

    // -- GATE 4: demosaic_psnr (MHC must beat bilinear by a measured margin) --
    bool gate4;
    double psnr_mhc, psnr_bilinear;
    {
        psnr_mhc = psnr_f32(rgb_mhc_gate, true_sensor_rgb_d65, 1.0);
        psnr_bilinear = psnr_f32(rgb_bilinear_gate, true_sensor_rgb_d65, 1.0);
        const double gap_db = psnr_mhc - psnr_bilinear;
        gate4 = gap_db >= kDemosaicPsnrMarginDb;
        std::printf("GATE demosaic_psnr: %s\n", gate4 ? "PASS" : "FAIL");
        std::printf("[info] demosaic_psnr: MHC = %.3f dB | bilinear = %.3f dB | gap = %.3f dB (must be >= %.2f dB)\n",
                   psnr_mhc, psnr_bilinear, gap_db, kDemosaicPsnrMarginDb);
        csv.push_back({ "demosaic_psnr", "mhc_db", fmt(psnr_mhc, 3), "n/a", "n/a" });
        csv.push_back({ "demosaic_psnr", "bilinear_db", fmt(psnr_bilinear, 3), "n/a", "n/a" });
        csv.push_back({ "demosaic_psnr", "gap_db", fmt(gap_db, 3), fmt(kDemosaicPsnrMarginDb, 2), gate4 ? "PASS" : "FAIL" });
    }

    // -- GATE 5: awb_accuracy (both estimators, both illuminants, vs known truth) --
    bool gate5;
    {
        auto relerr = [](float est, float truth) { return std::fabs(static_cast<double>(est) - truth) / std::max(0.05, static_cast<double>(truth)); };
        double e1 = std::max({ relerr(awb_d65.gray[0], kTrueAwbGainD65[0]), relerr(awb_d65.gray[2], kTrueAwbGainD65[2]) });
        double e2 = std::max({ relerr(awb_d65.white[0], kTrueAwbGainD65[0]), relerr(awb_d65.white[2], kTrueAwbGainD65[2]) });
        double e3 = std::max({ relerr(awb_tng.gray[0], kTrueAwbGainTungsten[0]), relerr(awb_tng.gray[2], kTrueAwbGainTungsten[2]) });
        double e4 = std::max({ relerr(awb_tng.white[0], kTrueAwbGainTungsten[0]), relerr(awb_tng.white[2], kTrueAwbGainTungsten[2]) });
        gate5 = e1 <= kAwbAccuracyTolFrac && e2 <= kAwbAccuracyTolFrac && e3 <= kAwbAccuracyTolFrac && e4 <= kAwbAccuracyTolFrac;
        std::printf("GATE awb_accuracy: %s\n", gate5 ? "PASS" : "FAIL");
        std::printf("[info] awb_accuracy: rel. gain error vs truth -- D65 gray=%.3f white=%.3f | tungsten gray=%.3f white=%.3f (tol %.2f each)\n",
                   e1, e2, e3, e4, kAwbAccuracyTolFrac);
        csv.push_back({ "awb_accuracy", "d65_gray_world_relerr", fmt(e1, 4), fmt(kAwbAccuracyTolFrac, 2), e1 <= kAwbAccuracyTolFrac ? "PASS" : "FAIL" });
        csv.push_back({ "awb_accuracy", "d65_white_patch_relerr", fmt(e2, 4), fmt(kAwbAccuracyTolFrac, 2), e2 <= kAwbAccuracyTolFrac ? "PASS" : "FAIL" });
        csv.push_back({ "awb_accuracy", "tungsten_gray_world_relerr", fmt(e3, 4), fmt(kAwbAccuracyTolFrac, 2), e3 <= kAwbAccuracyTolFrac ? "PASS" : "FAIL" });
        csv.push_back({ "awb_accuracy", "tungsten_white_patch_relerr", fmt(e4, 4), fmt(kAwbAccuracyTolFrac, 2), e4 <= kAwbAccuracyTolFrac ? "PASS" : "FAIL" });
    }

    // -- GATE 6: awb_red_crop_failure (designed negative control) -----------
    bool gate6;
    double crop_gray_r;
    {
        double sum_r = 0.0, sum_g = 0.0; long long cnt_r = 0, cnt_g = 0;
        for (int y = kRedCropY0; y < kRedCropY0 + kRedCropH; ++y) {
            for (int x = kRedCropX0; x < kRedCropX0 + kRedCropW; ++x) {
                const int i = y * kRawW + x;
                const int ch = phase_to_wb_channel(bayer_phase_at(x, y));
                const float v = dc_d65_gpu[i];
                if (ch == 0) { sum_r += v; ++cnt_r; }
                else if (ch == 1) { sum_g += v; ++cnt_g; }
            }
        }
        const double mean_r = sum_r / static_cast<double>(cnt_r);
        const double mean_g = sum_g / static_cast<double>(cnt_g);
        crop_gray_r = mean_g / std::max(1e-6, mean_r);   // gray-world gain the crop ALONE would produce
        const double dev = std::fabs(crop_gray_r - kTrueAwbGainD65[0]);
        gate6 = dev >= kAwbRedCropFailMinFrac;   // MUST deviate -- proving the known failure mode is real
        std::printf("GATE awb_red_crop_failure: %s\n", gate6 ? "PASS" : "FAIL");
        std::printf("[info] awb_red_crop_failure: gray-world R-gain on the red-heavy crop alone = %.4f "
                   "(true D65 gain = %.4f, deviation = %.4f, MUST be >= %.2f -- negative control)\n",
                   crop_gray_r, kTrueAwbGainD65[0], dev, kAwbRedCropFailMinFrac);
        csv.push_back({ "awb_red_crop_failure", "gray_gain_r_deviation", fmt(dev, 4), fmt(kAwbRedCropFailMinFrac, 2), gate6 ? "PASS" : "FAIL" });
    }

    // -- GATE 7: ccm_color_chart (D65, all 24 patches) -----------------------
    bool gate7;
    double chart_mean_err_d65, chart_max_err_d65;
    {
        double sum = 0.0, mx = 0.0;
        for (int r = 0; r < kChartRows; ++r) {
            for (int c = 0; c < kChartCols; ++c) {
                const int idx = r * kChartCols + c;
                const int x0 = kChartX0 + c * (kPatchSize + kPatchGap);
                const int y0 = kChartY0 + r * (kPatchSize + kPatchGap);
                const RGBd m = patch_mean_srgb8(fin_d65.srgb8, kRawW, x0, y0, kPatchSize, 4);
                const double e = rgb_dist(m, kChartRefSrgb8[idx][0], kChartRefSrgb8[idx][1], kChartRefSrgb8[idx][2]);
                sum += e; mx = std::max(mx, e);
            }
        }
        chart_mean_err_d65 = sum / kChartN;
        chart_max_err_d65 = mx;
        gate7 = chart_mean_err_d65 <= kCcmChartMeanTol && chart_max_err_d65 <= kCcmChartMaxTol;
        std::printf("GATE ccm_color_chart: %s\n", gate7 ? "PASS" : "FAIL");
        std::printf("[info] ccm_color_chart (D65): mean patch RGB-distance = %.3f (tol %.2f) | max = %.3f (tol %.2f), 0..255 scale, n=%d patches\n",
                   chart_mean_err_d65, kCcmChartMeanTol, chart_max_err_d65, kCcmChartMaxTol, kChartN);
        csv.push_back({ "ccm_color_chart", "mean_patch_rgb_dist", fmt(chart_mean_err_d65, 3), fmt(kCcmChartMeanTol, 2), gate7 ? "PASS" : "FAIL" });
        csv.push_back({ "ccm_color_chart", "max_patch_rgb_dist", fmt(chart_max_err_d65, 3), fmt(kCcmChartMaxTol, 2), gate7 ? "PASS" : "FAIL" });
    }

    // -- GATE 8: tungsten_wrong_awb_negative_control -------------------------
    bool gate8;
    double chart_err_tng_correct, chart_err_tng_wrong;
    {
        auto chart_mean_err = [&](const std::vector<unsigned char>& img) {
            double sum = 0.0;
            for (int r = 0; r < kChartRows; ++r) for (int c = 0; c < kChartCols; ++c) {
                const int idx = r * kChartCols + c;
                const int x0 = kChartX0 + c * (kPatchSize + kPatchGap);
                const int y0 = kChartY0 + r * (kPatchSize + kPatchGap);
                const RGBd m = patch_mean_srgb8(img, kRawW, x0, y0, kPatchSize, 4);
                sum += rgb_dist(m, kChartRefSrgb8[idx][0], kChartRefSrgb8[idx][1], kChartRefSrgb8[idx][2]);
            }
            return sum / kChartN;
        };
        chart_err_tng_correct = chart_mean_err(fin_tng_correct.srgb8);
        chart_err_tng_wrong = chart_mean_err(fin_tng_wrong.srgb8);
        gate8 = chart_err_tng_wrong >= chart_err_tng_correct * kWrongAwbInflationMinX;
        std::printf("GATE tungsten_wrong_awb_negative_control: %s\n", gate8 ? "PASS" : "FAIL");
        std::printf("[info] tungsten_wrong_awb_negative_control: mean chart error with CORRECT AWB gains = %.3f | "
                   "with WRONG (D65) gains = %.3f | inflation = %.2fx (must be >= %.1fx -- proves the color cast is real)\n",
                   chart_err_tng_correct, chart_err_tng_wrong, chart_err_tng_wrong / std::max(1e-6, chart_err_tng_correct), kWrongAwbInflationMinX);
        csv.push_back({ "tungsten_wrong_awb_negative_control", "chart_err_correct", fmt(chart_err_tng_correct, 3), "n/a", "n/a" });
        csv.push_back({ "tungsten_wrong_awb_negative_control", "chart_err_wrong", fmt(chart_err_tng_wrong, 3), "n/a", "n/a" });
        csv.push_back({ "tungsten_wrong_awb_negative_control", "inflation_x", fmt(chart_err_tng_wrong / std::max(1e-6, chart_err_tng_correct), 3), fmt(kWrongAwbInflationMinX, 2), gate8 ? "PASS" : "FAIL" });
    }

    // -- GATE 9: end_to_end_psnr (D65 and tungsten vs the reference rendering) --
    bool gate9;
    double e2e_psnr_d65, e2e_psnr_tng;
    {
        e2e_psnr_d65 = psnr_u8(fin_d65.srgb8, true_scene_srgb);
        e2e_psnr_tng = psnr_u8(fin_tng_correct.srgb8, true_scene_srgb);
        gate9 = e2e_psnr_d65 >= kEndToEndPsnrFloorD65 && e2e_psnr_tng >= kEndToEndPsnrFloorTungsten;
        std::printf("GATE end_to_end_psnr: %s\n", gate9 ? "PASS" : "FAIL");
        std::printf("[info] end_to_end_psnr: D65 = %.3f dB (floor %.1f) | tungsten = %.3f dB (floor %.1f, lower is expected -- AWB is imperfect)\n",
                   e2e_psnr_d65, kEndToEndPsnrFloorD65, e2e_psnr_tng, kEndToEndPsnrFloorTungsten);
        csv.push_back({ "end_to_end_psnr", "d65_db", fmt(e2e_psnr_d65, 3), fmt(kEndToEndPsnrFloorD65, 1), e2e_psnr_d65 >= kEndToEndPsnrFloorD65 ? "PASS" : "FAIL" });
        csv.push_back({ "end_to_end_psnr", "tungsten_db", fmt(e2e_psnr_tng, 3), fmt(kEndToEndPsnrFloorTungsten, 1), e2e_psnr_tng >= kEndToEndPsnrFloorTungsten ? "PASS" : "FAIL" });
    }

    // -- GATE 10: fused_vs_staged --------------------------------------------
    bool gate10;
    {
        const double d = max_abs_diff(fused_d65_gpu, wb_d65_gpu);
        gate10 = d <= kFusedVsStagedTol;
        std::printf("GATE fused_vs_staged: %s\n", gate10 ? "PASS" : "FAIL");
        std::printf("[info] fused_vs_staged: max|fused-staged| = %.6f (tol %.4f)\n", d, kFusedVsStagedTol);
        csv.push_back({ "fused_vs_staged", "max_abs_diff", fmt(d, 6), fmt(kFusedVsStagedTol, 4), gate10 ? "PASS" : "FAIL" });
    }

    // ======================= ARTIFACTS ============================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    std::vector<unsigned char> raw_vis_d65(n), sh_vis_d65(n);
    for (int i = 0; i < n; ++i) {
        raw_vis_d65[i] = static_cast<unsigned char>(std::min(255, raw_d65[i] >> 2));   // 10-bit -> 8-bit for viewing
        float v = sh_d65_gpu[i] * 255.0f;
        sh_vis_d65[i] = static_cast<unsigned char>(std::min(std::max(v, 0.0f), 255.0f));
    }
    auto to_u8 = [&](const std::vector<float>& lin) {
        std::vector<unsigned char> u8(lin.size());
        for (size_t i = 0; i < lin.size(); ++i) u8[i] = static_cast<unsigned char>(std::min(std::max(lin[i] * 255.0f, 0.0f), 255.0f));
        return u8;
    };
    std::vector<unsigned char> chart_crop(static_cast<size_t>(kChartW) * kChartH * 3);
    for (int y = 0; y < kChartH; ++y)
        for (int x = 0; x < kChartW; ++x)
            for (int c = 0; c < 3; ++c)
                chart_crop[(static_cast<size_t>(y) * kChartW + x) * 3 + c] =
                    fin_d65.srgb8[(static_cast<size_t>(kChartY0 + y) * kRawW + (kChartX0 + x)) * 3 + c];

    bool artifact_ok = !out_dir.empty();
    artifact_ok = artifact_ok
        && write_pgm(out_dir + "/raw_vis_d65.pgm", kRawW, kRawH, raw_vis_d65)
        && write_pgm(out_dir + "/shading_corrected_d65.pgm", kRawW, kRawH, sh_vis_d65)
        && write_ppm(out_dir + "/demosaiced_mhc_d65.ppm", kRawW, kRawH, to_u8(rgb_mhc_gate))
        && write_ppm(out_dir + "/demosaiced_bilinear_d65.ppm", kRawW, kRawH, to_u8(rgb_bilinear_gate))
        && write_ppm(out_dir + "/white_balanced_d65.ppm", kRawW, kRawH, to_u8(fin_d65.rgb_mhc))
        && write_ppm(out_dir + "/final_d65.ppm", kRawW, kRawH, fin_d65.srgb8)
        && write_ppm(out_dir + "/final_tungsten.ppm", kRawW, kRawH, fin_tng_correct.srgb8)
        && write_ppm(out_dir + "/final_tungsten_wrong_awb.ppm", kRawW, kRawH, fin_tng_wrong.srgb8)
        && write_ppm(out_dir + "/chart_crop_d65.ppm", kChartW, kChartH, chart_crop)
        && write_gates_csv(out_dir + "/gates_metrics.csv", csv);
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{raw_vis_d65.pgm, shading_corrected_d65.pgm, demosaiced_mhc_d65.ppm, "
                   "demosaiced_bilinear_d65.ppm, white_balanced_d65.ppm, final_d65.ppm, final_tungsten.ppm, "
                   "final_tungsten_wrong_awb.ppm, chart_crop_d65.ppm, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");

    // ---- cleanup --------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_raw_d65)); CUDA_CHECK(cudaFree(d_raw_tng));
    CUDA_CHECK(cudaFree(d65.d_bl)); CUDA_CHECK(cudaFree(d65.d_sh)); CUDA_CHECK(cudaFree(d65.d_dc));
    CUDA_CHECK(cudaFree(tng.d_bl)); CUDA_CHECK(cudaFree(tng.d_sh)); CUDA_CHECK(cudaFree(tng.d_dc));
    CUDA_CHECK(cudaFree(d_wb_d65)); CUDA_CHECK(cudaFree(d_fused_d65));
    CUDA_CHECK(cudaFree(d_rgb_bilinear_gate)); CUDA_CHECK(cudaFree(d_rgb_mhc_gate));

    // ---- verdict --------------------------------------------------------------
    const bool success = verify_pass && gate1 && gate2 && gate3 && gate4 && gate5
                        && gate6 && gate7 && gate8 && gate9 && gate10 && artifact_ok;
    if (success) {
        std::printf("RESULT: PASS (VERIFY + all 10 gates passed: black_level_residual, shading_flatness, "
                   "defect_recovery, demosaic_psnr, awb_accuracy, awb_red_crop_failure, ccm_color_chart, "
                   "tungsten_wrong_awb_negative_control, end_to_end_psnr, fused_vs_staged)\n");
    } else {
        std::printf("RESULT: FAIL (VERIFY or a gate above did not pass -- see GATE:/VERIFY:/[info] lines)\n");
    }
    return success ? 0 : 1;
}
