// ===========================================================================
// main.cu — entry point for project 01.09 (Photometric/vignetting
//           calibration kernels)
//
// Role in the project
// --------------------
// Orchestration: load the committed dark/flat stacks + test scene, run the
// GPU calibration pipeline (dark-stack mean -> flat-stack mean -> dark-
// subtract -> center-normalize -> radial-bin -> [shared LS fit] -> correct)
// AND the independent CPU reference pipeline, VERIFY agreement stage by
// stage, evaluate six INDEPENDENT verification gates against ground truth
// (never routed through the pipeline being graded), write every artifact
// the README/demo describe, and report one final PASS/FAIL.
//
// The pipeline (THEORY.md "The algorithm" derives every step):
//   dsnu_recovered      = mean(dark_stack)                          [stack_mean]
//   flat_avg            = mean(flat_stack)                          [stack_mean]
//   flat_minus_dsnu     = flat_avg - dsnu_recovered                 [elementwise_sub]
//   center_val          = mean(flat_minus_dsnu over kCenterRoi)     [roi_mean_reduce]
//   gain_nonparametric  = flat_minus_dsnu / center_val              [affine]
//   {bin_sum, bin_count} = radial histogram of gain_nonparametric   [radial_bin]
//   {a2,a4,a6}          = least-squares fit of V(r)=1+a2r^2+a4r^4+a6r^6
//                         to the (binned) nonparametric gain map    [SHARED host solve]
//   scene_corrected     = (scene - dsnu_recovered) / gain_nonparametric   [correction]
//
// Output contract (load-bearing!) — same discipline as every project in
// this repo (see docs/PROJECT_TEMPLATE/src/main.cu's header for the
// general statement): "[demo]", "PROBLEM:", "DATA:", "VERIFY:",
// "GATE ...:", "ARTIFACT:" and "RESULT:" lines are STABLE (no timings, no
// device names, no measured floats) and are diffed against
// demo/expected_output.txt; "[time]" and "[info]" lines carry the actual
// measured numbers and are deliberately NOT diffed.
//
// Read this after: kernels.cuh (the interface), kernels.cu (the GPU
// kernels + the shared LS fit), reference_cpu.cpp (the independent CPU twins).
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
// Ground-truth constants used ONLY by the independent gates below — MUST
// MATCH scripts/make_synthetic.py's FOCAL_EFF_PX / CENTER_OFFSET_*_PX /
// L_FLAT. Per the 01.08 crf_true_g precedent (see that project's main.cu),
// these are re-typed independently here, NEVER shared code with the Python
// generator or with the calibration pipeline being graded — that
// independence is what makes a gate a gate.
// ===========================================================================
static constexpr float kFocalEffPxTrue = 200.0f;
static constexpr float kCenterOffsetXTrue = 3.0f;
static constexpr float kCenterOffsetYTrue = -2.0f;
static constexpr float kLFlatTrue = 180.0f;

// Radius normalization scale for the parametric fit's basis (kernels.cuh
// SECTION 5 derives why normalization is needed at all). ~image half-
// diagonal-ish; only needs to be "the right order of magnitude" for good
// conditioning, not tuned to any specific value.
static constexpr float kRFitNorm = 100.0f;

// Swatch rectangles (x0,x1,y0,y1, half-open) — MUST MATCH
// scripts/make_synthetic.py's SWATCH_* constants (the correction_efficacy
// gate reads these same five regions out of the corrected/uncorrected scene).
struct Rect { int x0, x1, y0, y1; };
static constexpr Rect kSwatchCenter{ 72, 88, 52, 68 };
static constexpr Rect kSwatchTL{ 4, 20, 4, 20 };
static constexpr Rect kSwatchTR{ 140, 156, 4, 20 };
static constexpr Rect kSwatchBL{ 4, 20, 100, 116 };
static constexpr Rect kSwatchBR{ 140, 156, 100, 116 };

// -- GPU-vs-CPU VERIFY tolerances (per stage — "how far can two
// independent float pipelines legitimately drift", not physical bounds;
// THEORY.md "Numerical considerations"). Measured on the reference machine
// (RTX 2080 SUPER, sm_75) then margined, per this repo's calibration
// discipline (01.01/01.08's identical practice). -----------------------
// Every tolerance below was set by MEASURING the actual value on the
// reference machine (RTX 2080 SUPER, sm_75; printed in the [info] lines
// this same run produces) and then adding margin — never set AT the
// measured value (01.01/01.08's identical discipline). The measured
// reference values are quoted in each comment.
static constexpr double kTolStackMean = 5e-5;    // measured ~0 (double-accumulated identically both sides)
static constexpr double kTolSub = 1e-4;          // measured ~0 (one subtraction on top of the above)
static constexpr double kTolCenterVal = 5e-5;    // measured ~0 (scalar: sum over 64 ROI pixels / 64)
static constexpr double kTolGainMap = 5e-5;      // measured ~0 (affine rescale of flat_minus_dsnu)
static constexpr double kTolRadialBinSum = 3e-3; // measured 1.0e-3 (float32 atomicAdd order (GPU) vs sequential (CPU))
static constexpr double kTolCorrection = 2e-4;   // measured ~0 ((I - dsnu) / gain)

// -- Gate tolerances, each a floor/ceiling with margin over a MEASURED
// value (never AT the measured value — 01.01/01.08's identical discipline).
static constexpr double kTolDsnuMeanAbsErr = 0.35;     // measured 0.24 code-value units; DSNU excursion is only +-2
static constexpr double kTolDsnuCorrelation = 0.93;    // measured 0.966 (Pearson r, recovered vs true DSNU pattern)
static constexpr double kTolGainMeanRelErr = 0.005;    // measured 0.0019 mean relative error, nonparametric gain vs true V*PRNU
static constexpr double kTolRadialFitMaxDev = 0.008;   // measured 0.0027; fitted V(r) vs true cos^4 V(r), max |dev| over r=[0,100]px
static constexpr double kTolResidualRatioLo = 0.5;     // measured 1.01; std(residual)/std(prnu_signal) must stay in this band —
static constexpr double kTolResidualRatioHi = 2.0;     // proves the residual IS PRNU-scale, not fit-error-scale
static constexpr double kTolNoiseAvgFactorLo = 0.85;   // measured ratios 1.000x/1.001x of ideal sqrt(N) — very tight fit
static constexpr double kTolNoiseAvgFactorHi = 1.15;
static constexpr double kTolCorrectionEfficacy = 0.02; // measured 0.001; corrected center-vs-corner relative disparity, must be small
static constexpr double kTolFlatness = 0.06;           // measured 0.039; corrected single-flat-frame max relative deviation from its own mean

// ===========================================================================
// Minimal, STRICT PGM (P5) reader/writer — same discipline as 01.01/01.08's
// read_pgm/write_pgm (only ever reads files this project's own generator
// wrote; any mismatch aborts rather than silently truncating).
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

// code01_to_pgm — clamp a value already in CODE-VALUE units (roughly
// [0,255], this project's whole domain — unlike 01.08's [0,1] tone-mapped
// outputs) to uint8 for a viewable PGM artifact: round-to-nearest, hard
// clamp. Used for scene_corrected.pgm / scene_uncorrected.pgm, where the
// values are already near-displayable and need no rescaling.
static std::vector<unsigned char> code_to_pgm(const std::vector<float>& img)
{
    std::vector<unsigned char> out(img.size());
    for (size_t i = 0; i < img.size(); ++i) {
        float v = img[i];
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        out[i] = static_cast<unsigned char>(v + 0.5f);
    }
    return out;
}

// stretch01_to_pgm — ADAPTIVE min-max contrast stretch: maps [lo,hi] -> [0,255].
// Used for dsnu_recovered.pgm, whose true excursion (+-2 code-value units
// around a black-level pedestal) is far too small to be visible without
// stretching — the stretch bounds are ALWAYS printed alongside so the
// artifact is honestly self-describing (never a silent rescale).
static std::vector<unsigned char> stretch_to_pgm(const std::vector<float>& img, float lo, float hi)
{
    std::vector<unsigned char> out(img.size());
    const float range = (hi - lo) > 1e-6f ? (hi - lo) : 1.0f;
    for (size_t i = 0; i < img.size(); ++i) {
        float v = (img[i] - lo) / range * 255.0f;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        out[i] = static_cast<unsigned char>(v + 0.5f);
    }
    return out;
}

// unit_to_pgm — scale a value already conceptually in [0,1] (V(x,y), the
// vignette field, and gain values which sit close to [0,1]) directly by 255
// — no adaptive stretch needed since the domain is already known and fixed.
static std::vector<unsigned char> unit_to_pgm(const std::vector<float>& img)
{
    std::vector<unsigned char> out(img.size());
    for (size_t i = 0; i < img.size(); ++i) {
        float v = img[i] * 255.0f;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        out[i] = static_cast<unsigned char>(v + 0.5f);
    }
    return out;
}

// ===========================================================================
// Sample loading — dark_00..15.pgm, flat_00..15.pgm, scene.pgm, plus the two
// ground-truth binary dumps, all dimension-checked against kW/kH (a strict
// loader: any mismatch aborts rather than silently truncating, per repo
// convention — 01.01/01.08's identical practice).
// ===========================================================================
static bool load_stack(const std::string& prefix, int count,
                       const std::string& cli_dir, const char* argv0,
                       std::vector<float>& out /* count*kN, frame-major */)
{
    out.assign(static_cast<size_t>(count) * kN, 0.0f);
    for (int f = 0; f < count; ++f) {
        char name[64];
        std::snprintf(name, sizeof(name), "%s_%02d.pgm", prefix.c_str(), f);
        const std::string path = find_data_file(cli_dir, argv0, name);
        if (path.empty()) {
            std::fprintf(stderr, "sample: %s not found (run scripts/make_synthetic.py?)\n", name);
            return false;
        }
        int w, h;
        std::vector<unsigned char> raw;
        if (!read_pgm(path, w, h, raw) || w != kW || h != kH) {
            std::fprintf(stderr, "sample: %s missing or wrong size (expected %dx%d)\n", name, kW, kH);
            return false;
        }
        for (int p = 0; p < kN; ++p) out[static_cast<size_t>(f) * kN + p] = static_cast<float>(raw[p]);
    }
    return true;
}

static bool load_float_binary(const std::string& path, int count, std::vector<float>& out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    out.resize(static_cast<size_t>(count));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return in.gcount() == static_cast<std::streamsize>(out.size() * sizeof(float));
}

// ===========================================================================
// Small numeric helpers shared by VERIFY and the gates below.
// ===========================================================================
static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

// pearson_correlation — standard r = cov(a,b) / sqrt(var(a)*var(b)), the
// dsnu_recovery gate's SHAPE-agreement check (a pattern can be recovered
// with the right shape but a biased scale/offset and still be highly
// correlated — complementary to mean_abs_err, which catches scale/offset
// errors correlation alone would miss).
static double pearson_correlation(const std::vector<float>& a, const std::vector<float>& b)
{
    const size_t n = a.size();
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < n; ++i) { mean_a += a[i]; mean_b += b[i]; }
    mean_a /= static_cast<double>(n); mean_b /= static_cast<double>(n);
    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - mean_a, db = b[i] - mean_b;
        cov += da * db; var_a += da * da; var_b += db * db;
    }
    const double denom = std::sqrt(var_a * var_b);
    return denom > 1e-12 ? cov / denom : 0.0;
}

// rms — population root-mean-square of a residual array (the
// noise_averaging gate's per-N noise-std estimator).
static double rms(const std::vector<float>& residual)
{
    double acc = 0.0;
    for (float v : residual) acc += static_cast<double>(v) * static_cast<double>(v);
    return std::sqrt(acc / static_cast<double>(residual.size()));
}

static double mean_in_rect(const std::vector<float>& img, int W, const Rect& r)
{
    double sum = 0.0;
    for (int y = r.y0; y < r.y1; ++y)
        for (int x = r.x0; x < r.x1; ++x)
            sum += static_cast<double>(img[static_cast<size_t>(y) * W + x]);
    return sum / static_cast<double>((r.x1 - r.x0) * (r.y1 - r.y0));
}

// max_rel_dev_from_mean — max_i |img[i]-mean(img)| / mean(img) over the
// WHOLE image. The flatness gate's core metric: "how uniform is this
// image", relative to its own average level.
static double max_rel_dev_from_mean(const std::vector<float>& img)
{
    double mean = 0.0;
    for (float v : img) mean += v;
    mean /= static_cast<double>(img.size());
    double worst = 0.0;
    for (float v : img) worst = std::max(worst, std::fabs(static_cast<double>(v) - mean) / mean);
    return worst;
}

// v_true_of_r — the KNOWN analytic cos^4 vignette, as a pure function of
// radius (pixels), independently re-derived here from FOCAL_EFF_PX (see the
// module header — never shared code with scripts/make_synthetic.py's
// vignette_v(), the same independence 01.08's crf_true_g exercises for its
// own 1-D ground-truth curve).
static double v_true_of_r(double r_px)
{
    const double theta = std::atan2(r_px, static_cast<double>(kFocalEffPxTrue));
    const double c = std::cos(theta);
    return c * c * c * c;
}

// v_fit_of_r — the RECOVERED parametric curve V(r) = 1 + a2*r_n^2 + a4*r_n^4
// + a6*r_n^6, r_n = r/kRFitNorm (kernels.cuh SECTION 5's model).
static double v_fit_of_r(double r_px, double a2, double a4, double a6)
{
    const double rn = r_px / static_cast<double>(kRFitNorm);
    const double rn2 = rn * rn, rn4 = rn2 * rn2, rn6 = rn4 * rn2;
    return 1.0 + a2 * rn2 + a4 * rn4 + a6 * rn6;
}

// ===========================================================================
// gates_metrics.csv writer (same shape as 01.01/01.08's — one row per
// measured quantity, machine-readable teaching artifact).
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

    std::printf("[demo] photometric/vignetting calibration: dark/flat-stack calibration + "
               "parametric radial fit vs ground truth (project 01.09)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d sensor model I=g*L+o, g=V(cos^4 vignette)*PRNU, o=DSNU, "
               "dark-stack N=%d, flat-stack N=%d, FP32\n", kW, kH, kNumDarkFrames, kNumFlatFrames);

    // ---- data ----------------------------------------------------------
    std::vector<float> h_dark_stack, h_flat_stack, h_scene_raw_f, h_dsnu_true, h_gain_true;
    std::vector<unsigned char> h_scene_raw_u8;
    bool loaded = load_stack("dark", kNumDarkFrames, data_dir, argv[0], h_dark_stack)
               && load_stack("flat", kNumFlatFrames, data_dir, argv[0], h_flat_stack);
    if (loaded) {
        const std::string scene_path = find_data_file(data_dir, argv[0], "scene.pgm");
        int w, h;
        loaded = !scene_path.empty() && read_pgm(scene_path, w, h, h_scene_raw_u8) && w == kW && h == kH;
        if (!loaded) std::fprintf(stderr, "sample: scene.pgm not found or wrong size\n");
    }
    if (loaded) {
        const std::string dsnu_path = find_data_file(data_dir, argv[0], "dsnu_true.bin");
        const std::string gain_path = find_data_file(data_dir, argv[0], "gain_true.bin");
        loaded = !dsnu_path.empty() && !gain_path.empty()
              && load_float_binary(dsnu_path, kN, h_dsnu_true)
              && load_float_binary(gain_path, kN, h_gain_true);
        if (!loaded) std::fprintf(stderr, "sample: dsnu_true.bin / gain_true.bin not found or wrong size\n");
    }
    if (!loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    h_scene_raw_f.resize(static_cast<size_t>(kN));
    for (int p = 0; p < kN; ++p) h_scene_raw_f[p] = static_cast<float>(h_scene_raw_u8[p]);
    std::printf("DATA: synthetic camera-calibration rig (%d dark + %d flat frames + 1 natural test "
               "scene with 5 identical-radiance gray-card swatches), decentered cos^4 vignette x "
               "hashed PRNU/DSNU ground truth [synthetic, seed 42]\n", kNumDarkFrames, kNumFlatFrames);

    // ---- device buffers: upload the stacks + scene -------------------------
    float *d_dark_stack = nullptr, *d_flat_stack = nullptr, *d_scene = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dark_stack, static_cast<size_t>(kNumDarkFrames) * kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flat_stack, static_cast<size_t>(kNumFlatFrames) * kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scene, kN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_dark_stack, h_dark_stack.data(),
                          static_cast<size_t>(kNumDarkFrames) * kN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flat_stack, h_flat_stack.data(),
                          static_cast<size_t>(kNumFlatFrames) * kN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scene, h_scene_raw_f.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

    // ---- GPU pipeline (see file header for the full stage list) -----------
    float *d_dsnu_rec = nullptr, *d_flat_avg = nullptr, *d_flat_minus_dsnu = nullptr, *d_gain_np = nullptr;
    float *d_scene_corrected = nullptr, *d_flat0_corrected = nullptr;
    double *d_center_sum = nullptr;
    float *d_bin_sum = nullptr; int *d_bin_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dsnu_rec, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flat_avg, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flat_minus_dsnu, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gain_np, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scene_corrected, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flat0_corrected, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_center_sum, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bin_sum, static_cast<size_t>(kNumRadialBins) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bin_count, static_cast<size_t>(kNumRadialBins) * sizeof(int)));

    GpuTimer gt_all; gt_all.begin();
    launch_stack_mean(d_dark_stack, kNumDarkFrames, kN, d_dsnu_rec);
    launch_stack_mean(d_flat_stack, kNumFlatFrames, kN, d_flat_avg);
    launch_elementwise_sub(d_flat_avg, d_dsnu_rec, kN, d_flat_minus_dsnu);

    CUDA_CHECK(cudaMemset(d_center_sum, 0, sizeof(double)));
    launch_roi_mean_reduce(d_flat_minus_dsnu, kW, kH, kCenterRoiX0, kCenterRoiX1, kCenterRoiY0, kCenterRoiY1,
                           d_center_sum);
    double h_center_sum_gpu = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_center_sum_gpu, d_center_sum, sizeof(double), cudaMemcpyDeviceToHost));
    const int center_count = (kCenterRoiX1 - kCenterRoiX0) * (kCenterRoiY1 - kCenterRoiY0);
    const double center_val_gpu = h_center_sum_gpu / static_cast<double>(center_count);

    launch_affine(d_flat_minus_dsnu, kN, static_cast<float>(1.0 / center_val_gpu), 0.0f, d_gain_np);

    CUDA_CHECK(cudaMemset(d_bin_sum, 0, static_cast<size_t>(kNumRadialBins) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bin_count, 0, static_cast<size_t>(kNumRadialBins) * sizeof(int)));
    launch_radial_bin(d_gain_np, kW, kH, static_cast<float>(kW) / 2.0f, static_cast<float>(kH) / 2.0f,
                      kNumRadialBins, kRadialBinWidthPx, d_bin_sum, d_bin_count);

    launch_correction(d_scene, d_dsnu_rec, d_gain_np, kN, kGainFloor, d_scene_corrected);
    // Frame 0 of the flat stack lives at device offset 0 of d_flat_stack (frame-major layout) —
    // no extra allocation needed, see kernels.cuh SECTION 1.
    launch_correction(d_flat_stack, d_dsnu_rec, d_gain_np, kN, kGainFloor, d_flat0_corrected);
    const float gpu_ms = gt_all.end_ms();

    // ---- download every GPU result needed for VERIFY/gates/artifacts -----
    std::vector<float> h_dsnu_rec_gpu(kN), h_flat_avg_gpu(kN), h_flat_minus_dsnu_gpu(kN), h_gain_np_gpu(kN);
    std::vector<float> h_scene_corrected_gpu(kN), h_flat0_corrected_gpu(kN);
    std::vector<float> h_bin_sum_gpu(static_cast<size_t>(kNumRadialBins));
    std::vector<int> h_bin_count_gpu(static_cast<size_t>(kNumRadialBins));
    CUDA_CHECK(cudaMemcpy(h_dsnu_rec_gpu.data(), d_dsnu_rec, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flat_avg_gpu.data(), d_flat_avg, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flat_minus_dsnu_gpu.data(), d_flat_minus_dsnu, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gain_np_gpu.data(), d_gain_np, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scene_corrected_gpu.data(), d_scene_corrected, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flat0_corrected_gpu.data(), d_flat0_corrected, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bin_sum_gpu.data(), d_bin_sum, static_cast<size_t>(kNumRadialBins) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bin_count_gpu.data(), d_bin_count, static_cast<size_t>(kNumRadialBins) * sizeof(int), cudaMemcpyDeviceToHost));

    // ---- CPU reference oracle: the SAME pipeline, independent primitives -
    CpuTimer cpu_timer; cpu_timer.begin();
    std::vector<float> h_dsnu_rec_cpu(kN), h_flat_avg_cpu(kN), h_flat_minus_dsnu_cpu(kN), h_gain_np_cpu(kN);
    std::vector<float> h_scene_corrected_cpu(kN), h_flat0_corrected_cpu(kN);
    stack_mean_cpu(h_dark_stack.data(), kNumDarkFrames, kN, h_dsnu_rec_cpu.data());
    stack_mean_cpu(h_flat_stack.data(), kNumFlatFrames, kN, h_flat_avg_cpu.data());
    elementwise_sub_cpu(h_flat_avg_cpu.data(), h_dsnu_rec_cpu.data(), kN, h_flat_minus_dsnu_cpu.data());
    const double center_sum_cpu = roi_sum_cpu(h_flat_minus_dsnu_cpu.data(), kW, kH,
                                              kCenterRoiX0, kCenterRoiX1, kCenterRoiY0, kCenterRoiY1);
    const double center_val_cpu = center_sum_cpu / static_cast<double>(center_count);
    affine_cpu(h_flat_minus_dsnu_cpu.data(), kN, static_cast<float>(1.0 / center_val_cpu), 0.0f, h_gain_np_cpu.data());

    std::vector<float> h_bin_sum_cpu(static_cast<size_t>(kNumRadialBins), 0.0f);
    std::vector<int> h_bin_count_cpu(static_cast<size_t>(kNumRadialBins), 0);
    radial_bin_cpu(h_gain_np_cpu.data(), kW, kH, static_cast<float>(kW) / 2.0f, static_cast<float>(kH) / 2.0f,
                  kNumRadialBins, kRadialBinWidthPx, h_bin_sum_cpu.data(), h_bin_count_cpu.data());

    correction_cpu(h_scene_raw_f.data(), h_dsnu_rec_cpu.data(), h_gain_np_cpu.data(), kN, kGainFloor,
                  h_scene_corrected_cpu.data());
    correction_cpu(h_flat_stack.data() /* frame 0 */, h_dsnu_rec_cpu.data(), h_gain_np_cpu.data(), kN, kGainFloor,
                  h_flat0_corrected_cpu.data());
    const double cpu_ms = cpu_timer.end_ms();
    std::printf("[time] GPU pipeline (7 kernel launches): %.3f ms | CPU oracle (independent, "
               "single-thread): %.1f ms\n", static_cast<double>(gpu_ms), cpu_ms);

    // ---- VERIFY: GPU vs CPU, every stage ----------------------------------
    bool verify_pass = true;
    const double d_dsnu = max_abs_diff(h_dsnu_rec_gpu, h_dsnu_rec_cpu);
    const double d_flat_avg_diff = max_abs_diff(h_flat_avg_gpu, h_flat_avg_cpu);
    const double d_sub = max_abs_diff(h_flat_minus_dsnu_gpu, h_flat_minus_dsnu_cpu);
    const double d_center = std::fabs(center_val_gpu - center_val_cpu);
    const double d_gain = max_abs_diff(h_gain_np_gpu, h_gain_np_cpu);
    double d_binsum = 0.0;
    long count_mismatch = 0;
    for (int i = 0; i < kNumRadialBins; ++i) {
        d_binsum = std::max(d_binsum, std::fabs(static_cast<double>(h_bin_sum_gpu[i]) - static_cast<double>(h_bin_sum_cpu[i])));
        if (h_bin_count_gpu[i] != h_bin_count_cpu[i]) ++count_mismatch;
    }
    const double d_scene_corr = max_abs_diff(h_scene_corrected_gpu, h_scene_corrected_cpu);
    const double d_flat0_corr = max_abs_diff(h_flat0_corrected_gpu, h_flat0_corrected_cpu);
    const double d_correction = std::max(d_scene_corr, d_flat0_corr);

    if (d_dsnu > kTolStackMean || d_flat_avg_diff > kTolStackMean) verify_pass = false;
    if (d_sub > kTolSub) verify_pass = false;
    if (d_center > kTolCenterVal) verify_pass = false;
    if (d_gain > kTolGainMap) verify_pass = false;
    if (d_binsum > kTolRadialBinSum) verify_pass = false;
    if (d_correction > kTolCorrection) verify_pass = false;

    std::printf("[info] verify(stack_mean): dark max|gpu-cpu|=%.6f flat max|gpu-cpu|=%.6f (tol %.5f)\n",
               d_dsnu, d_flat_avg_diff, kTolStackMean);
    std::printf("[info] verify(dark_subtract): max|gpu-cpu|=%.6f (tol %.5f)\n", d_sub, kTolSub);
    std::printf("[info] verify(center_normalize): |center_val gpu-cpu|=%.6f (tol %.5f) | gain map "
               "max|gpu-cpu|=%.6f (tol %.5f)\n", d_center, kTolCenterVal, d_gain, kTolGainMap);
    std::printf("[info] verify(radial_bin): max|bin_sum gpu-cpu|=%.6f (tol %.4f) | bin_count "
               "mismatches=%ld/%d\n", d_binsum, kTolRadialBinSum, count_mismatch, kNumRadialBins);
    std::printf("[info] verify(correction): scene max|gpu-cpu|=%.6f flat0 max|gpu-cpu|=%.6f (tol %.5f)\n",
               d_scene_corr, d_flat0_corr, kTolCorrection);
    std::printf("VERIFY: %s (GPU matches CPU reference within documented per-stage tolerance: "
               "stack_mean, dark_subtract, center_normalize, radial_bin, correction)\n",
               verify_pass ? "PASS" : "FAIL");

    // ======================= SHARED LS fit (called ONCE — see kernels.cuh
    //      SECTION 5 for the twin-independence-ruling justification) =======
    std::vector<float> fit_r, fit_mean;
    for (int i = 0; i < kNumRadialBins; ++i) {
        if (h_bin_count_cpu[i] > 0) {
            fit_r.push_back((static_cast<float>(i) + 0.5f) * kRadialBinWidthPx);
            fit_mean.push_back(h_bin_sum_cpu[i] / static_cast<float>(h_bin_count_cpu[i]));
        }
    }
    float a2 = 0.0f, a4 = 0.0f, a6 = 0.0f;
    fit_vignette_radial_ls(fit_r.data(), fit_mean.data(), static_cast<int>(fit_r.size()), kRFitNorm, a2, a4, a6);
    std::printf("[info] radial fit (shared host solve, %zu populated bins): V(r) = 1 + %.4f*(r/%.0f)^2 "
               "+ %.4f*(r/%.0f)^4 + %.4f*(r/%.0f)^6\n", fit_r.size(),
               static_cast<double>(a2), static_cast<double>(kRFitNorm),
               static_cast<double>(a4), static_cast<double>(kRFitNorm),
               static_cast<double>(a6), static_cast<double>(kRFitNorm));

    std::vector<CsvRow> csv;

    // ======================= GATE 1: dsnu_recovery ===========================
    double dsnu_mean_abs_err = 0.0;
    for (int i = 0; i < kN; ++i) dsnu_mean_abs_err += std::fabs(static_cast<double>(h_dsnu_rec_gpu[i]) - static_cast<double>(h_dsnu_true[i]));
    dsnu_mean_abs_err /= static_cast<double>(kN);
    const double dsnu_corr = pearson_correlation(h_dsnu_rec_gpu, h_dsnu_true);
    const bool gate_dsnu = (dsnu_mean_abs_err <= kTolDsnuMeanAbsErr) && (dsnu_corr >= kTolDsnuCorrelation);
    std::printf("GATE dsnu_recovery: %s\n", gate_dsnu ? "PASS" : "FAIL");
    std::printf("[info] dsnu_recovery: mean|recovered-true|=%.4f code-value units (tol %.2f) | "
               "correlation=%.4f (tol >= %.2f)\n", dsnu_mean_abs_err, kTolDsnuMeanAbsErr, dsnu_corr, kTolDsnuCorrelation);
    csv.push_back({ "dsnu_recovery", "mean_abs_err", fmt(dsnu_mean_abs_err, 4), fmt(kTolDsnuMeanAbsErr, 2), gate_dsnu ? "PASS" : "FAIL" });
    csv.push_back({ "dsnu_recovery", "correlation", fmt(dsnu_corr, 4), fmt(kTolDsnuCorrelation, 2), gate_dsnu ? "PASS" : "FAIL" });

    // ======================= GATE 2: gain_recovery ===========================
    double gain_mean_rel_err = 0.0;
    float gain_min = h_gain_np_gpu[0], gain_max = h_gain_np_gpu[0];
    for (int i = 0; i < kN; ++i) {
        gain_mean_rel_err += std::fabs(static_cast<double>(h_gain_np_gpu[i]) - static_cast<double>(h_gain_true[i])) / static_cast<double>(h_gain_true[i]);
        gain_min = std::min(gain_min, h_gain_np_gpu[i]);
        gain_max = std::max(gain_max, h_gain_np_gpu[i]);
    }
    gain_mean_rel_err /= static_cast<double>(kN);
    const bool gate_gain = gain_mean_rel_err <= kTolGainMeanRelErr;
    std::printf("GATE gain_recovery: %s\n", gate_gain ? "PASS" : "FAIL");
    std::printf("[info] gain_recovery: mean relative error = %.5f (tol %.3f) | recovered gain range "
               "[%.4f, %.4f] (gainFloor=%.2f never engaged — see kernels.cu correction_kernel)\n",
               gain_mean_rel_err, kTolGainMeanRelErr, static_cast<double>(gain_min), static_cast<double>(gain_max),
               static_cast<double>(kGainFloor));
    csv.push_back({ "gain_recovery", "mean_rel_err", fmt(gain_mean_rel_err, 5), fmt(kTolGainMeanRelErr, 3), gate_gain ? "PASS" : "FAIL" });

    // ======================= GATE 3: radial_fit ===============================
    // (a) fitted V(r) vs the KNOWN analytic cos^4 curve, as pure functions of
    //     radius — sampled every 1 px from 0..100 (the farthest corner from
    //     the geometric center; kernels.cuh's headroom comment).
    double radial_max_dev = 0.0;
    for (int r_int = 0; r_int <= 100; ++r_int) {
        const double dev = std::fabs(v_fit_of_r(r_int, a2, a4, a6) - v_true_of_r(r_int));
        radial_max_dev = std::max(radial_max_dev, dev);
    }
    // (b) decomposition consistency: residual = gain_nonparametric - V_fit(r)
    //     should have the SAME scale as prnu_signal = gain_true - V_fit(r) —
    //     see main.cu's file-level derivation in the task brief / THEORY.md
    //     "How we verify correctness" for why this is exactly the PRNU
    //     ripple the parametric fit, by construction, cannot capture.
    std::vector<float> residual(kN), prnu_signal(kN);
    for (int p = 0; p < kN; ++p) {
        const int x = p % kW, y = p / kW;
        const double dx = (static_cast<double>(x) + 0.5) - static_cast<double>(kW) / 2.0;
        const double dy = (static_cast<double>(y) + 0.5) - static_cast<double>(kH) / 2.0;
        const double r = std::sqrt(dx * dx + dy * dy);
        const double vfit = v_fit_of_r(r, a2, a4, a6);
        residual[p] = static_cast<float>(static_cast<double>(h_gain_np_gpu[p]) - vfit);
        prnu_signal[p] = static_cast<float>(static_cast<double>(h_gain_true[p]) - vfit);
    }
    const double std_residual = rms(residual);
    const double std_prnu = rms(prnu_signal);
    const double residual_ratio = std_prnu > 1e-9 ? std_residual / std_prnu : 0.0;
    const bool gate_radial_shape = radial_max_dev <= kTolRadialFitMaxDev;
    const bool gate_radial_decomp = (residual_ratio >= kTolResidualRatioLo) && (residual_ratio <= kTolResidualRatioHi);
    const bool gate_radial = gate_radial_shape && gate_radial_decomp;
    std::printf("GATE radial_fit: %s\n", gate_radial ? "PASS" : "FAIL");
    std::printf("[info] radial_fit: max|V_fitted(r)-V_true(r)| over r=[0,100]px = %.5f (tol %.3f) | "
               "residual/PRNU-signal std ratio = %.3f (tol [%.2f,%.2f]) — proves the fit's residual is "
               "PRNU-scale, not fit-error-scale (the decomposition semantics)\n",
               radial_max_dev, kTolRadialFitMaxDev, residual_ratio, kTolResidualRatioLo, kTolResidualRatioHi);
    csv.push_back({ "radial_fit", "max_dev_v", fmt(radial_max_dev, 5), fmt(kTolRadialFitMaxDev, 3), gate_radial_shape ? "PASS" : "FAIL" });
    csv.push_back({ "radial_fit", "residual_ratio", fmt(residual_ratio, 3), fmt(kTolResidualRatioLo, 2) + ".." + fmt(kTolResidualRatioHi, 2), gate_radial_decomp ? "PASS" : "FAIL" });

    // ======================= GATE 4: noise_averaging =========================
    // Independent expected value from ground truth (NOT derived from the
    // pipeline being graded — see kLFlatTrue's header): expected_flat(x,y) =
    // gain_true(x,y)*L_FLAT + dsnu_true(x,y). For N in {1,4,16}, avg_N is the
    // mean of the FIRST N flat frames (stack_mean_cpu reused, parameterized
    // by N — the same function the calibration pipeline itself calls with
    // N=16); std_N is the RMS of (avg_N - expected). THEORY.md "The math"
    // derives the ideal std_1/std_N = sqrt(N) law this gate checks.
    std::vector<float> expected_flat(kN);
    for (int p = 0; p < kN; ++p) expected_flat[p] = h_gain_true[p] * kLFlatTrue + h_dsnu_true[p];

    const int test_n[3] = { 1, 4, 16 };
    double std_n[3];
    for (int k = 0; k < 3; ++k) {
        std::vector<float> avg_n(kN);
        stack_mean_cpu(h_flat_stack.data(), test_n[k], kN, avg_n.data());
        std::vector<float> resid(kN);
        elementwise_sub_cpu(avg_n.data(), expected_flat.data(), kN, resid.data());
        std_n[k] = rms(resid);
    }
    const double ratio4 = std_n[0] / std_n[1];     // ideal sqrt(4) = 2.0
    const double ratio16 = std_n[0] / std_n[2];    // ideal sqrt(16) = 4.0
    const bool gate_noise4 = (ratio4 >= 2.0 * kTolNoiseAvgFactorLo) && (ratio4 <= 2.0 * kTolNoiseAvgFactorHi);
    const bool gate_noise16 = (ratio16 >= 4.0 * kTolNoiseAvgFactorLo) && (ratio16 <= 4.0 * kTolNoiseAvgFactorHi);
    const bool gate_noise = gate_noise4 && gate_noise16;
    std::printf("GATE noise_averaging: %s\n", gate_noise ? "PASS" : "FAIL");
    std::printf("[info] noise_averaging: std(N=1)=%.4f std(N=4)=%.4f std(N=16)=%.4f code-value units | "
               "ratio std1/std4=%.3f (ideal 2.0) ratio std1/std16=%.3f (ideal 4.0), both required within "
               "[%.2f,%.2f]x their ideal\n", std_n[0], std_n[1], std_n[2], ratio4, ratio16,
               kTolNoiseAvgFactorLo, kTolNoiseAvgFactorHi);
    csv.push_back({ "noise_averaging", "std_N1", fmt(std_n[0], 4), "n/a", "n/a" });
    csv.push_back({ "noise_averaging", "std_N4", fmt(std_n[1], 4), "n/a", "n/a" });
    csv.push_back({ "noise_averaging", "std_N16", fmt(std_n[2], 4), "n/a", "n/a" });
    csv.push_back({ "noise_averaging", "ratio_1_over_4", fmt(ratio4, 3), "~2.0", gate_noise4 ? "PASS" : "FAIL" });
    csv.push_back({ "noise_averaging", "ratio_1_over_16", fmt(ratio16, 3), "~4.0", gate_noise16 ? "PASS" : "FAIL" });

    // ======================= GATE 5: correction_efficacy =====================
    // THE reason this project exists: identical-radiance swatches, measured
    // center vs. corner, must AGREE after correction — with the UNCORRECTED
    // disparity reported as the negative-control baseline.
    const double center_corr = mean_in_rect(h_scene_corrected_gpu, kW, kSwatchCenter);
    const double tl_corr = mean_in_rect(h_scene_corrected_gpu, kW, kSwatchTL);
    const double tr_corr = mean_in_rect(h_scene_corrected_gpu, kW, kSwatchTR);
    const double bl_corr = mean_in_rect(h_scene_corrected_gpu, kW, kSwatchBL);
    const double br_corr = mean_in_rect(h_scene_corrected_gpu, kW, kSwatchBR);
    const double avg_corner_corr = (tl_corr + tr_corr + bl_corr + br_corr) / 4.0;
    const double disparity_corrected = std::fabs(center_corr - avg_corner_corr) / center_corr;

    const double center_raw = mean_in_rect(h_scene_raw_f, kW, kSwatchCenter);
    const double tl_raw = mean_in_rect(h_scene_raw_f, kW, kSwatchTL);
    const double tr_raw = mean_in_rect(h_scene_raw_f, kW, kSwatchTR);
    const double bl_raw = mean_in_rect(h_scene_raw_f, kW, kSwatchBL);
    const double br_raw = mean_in_rect(h_scene_raw_f, kW, kSwatchBR);
    const double avg_corner_raw = (tl_raw + tr_raw + bl_raw + br_raw) / 4.0;
    const double disparity_uncorrected = std::fabs(center_raw - avg_corner_raw) / center_raw;

    const bool gate_efficacy = disparity_corrected <= kTolCorrectionEfficacy;
    std::printf("GATE correction_efficacy: %s\n", gate_efficacy ? "PASS" : "FAIL");
    std::printf("[info] correction_efficacy: CORRECTED center-vs-corner relative disparity = %.4f "
               "(tol %.2f) | UNCORRECTED disparity = %.4f (negative-control baseline, NOT gated — the "
               "vignette-induced falloff this project exists to remove)\n",
               disparity_corrected, kTolCorrectionEfficacy, disparity_uncorrected);
    csv.push_back({ "correction_efficacy", "disparity_corrected", fmt(disparity_corrected, 4), fmt(kTolCorrectionEfficacy, 2), gate_efficacy ? "PASS" : "FAIL" });
    csv.push_back({ "correction_efficacy", "disparity_uncorrected_baseline", fmt(disparity_uncorrected, 4), "n/a (reported only)", "n/a" });

    // ======================= GATE 6: flatness =================================
    // Apply the recovered correction to a SINGLE raw flat frame (frame 0 of
    // the calibration stack — a self-consistency check, honestly NOT a held-
    // out generalization test, unlike GATE 5 above; see README "Limitations
    // & honesty"): the corrected result should be far more spatially uniform
    // than the dark-subtracted-only (uncorrected) frame.
    const double flatness_corrected = max_rel_dev_from_mean(h_flat0_corrected_gpu);
    std::vector<float> flat0_dark_sub(kN);
    elementwise_sub_cpu(h_flat_stack.data(), h_dsnu_rec_gpu.data(), kN, flat0_dark_sub.data());
    const double flatness_uncorrected = max_rel_dev_from_mean(flat0_dark_sub);
    const bool gate_flatness = flatness_corrected <= kTolFlatness;
    std::printf("GATE flatness: %s\n", gate_flatness ? "PASS" : "FAIL");
    std::printf("[info] flatness: CORRECTED single-frame max relative deviation from its own mean = "
               "%.4f (tol %.2f) | UNCORRECTED (dark-subtracted only) = %.4f (negative-control baseline, "
               "matches the vignette falloff, NOT gated)\n",
               flatness_corrected, kTolFlatness, flatness_uncorrected);
    csv.push_back({ "flatness", "max_rel_dev_corrected", fmt(flatness_corrected, 4), fmt(kTolFlatness, 2), gate_flatness ? "PASS" : "FAIL" });
    csv.push_back({ "flatness", "max_rel_dev_uncorrected_baseline", fmt(flatness_uncorrected, 4), "n/a (reported only)", "n/a" });

    // ======================= ARTIFACTS =========================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();

    // vignette_true.pgm — the TRUE optical vignette V(x,y), using the TRUE
    // (decentered) optical axis — independently evaluated per pixel, NOT
    // loaded from gain_true.bin (which is V*PRNU combined; this artifact
    // isolates V alone for a clean visual of the cos^4 falloff shape).
    std::vector<float> v_true_map(kN);
    const float cx_true = static_cast<float>(kW) / 2.0f + kCenterOffsetXTrue;
    const float cy_true = static_cast<float>(kH) / 2.0f + kCenterOffsetYTrue;
    for (int p = 0; p < kN; ++p) {
        const int x = p % kW, y = p / kW;
        const double dx = (static_cast<double>(x) + 0.5) - cx_true;
        const double dy = (static_cast<double>(y) + 0.5) - cy_true;
        v_true_map[p] = static_cast<float>(v_true_of_r(std::sqrt(dx * dx + dy * dy)));
    }
    artifact_ok = artifact_ok && write_pgm(out_dir + "/vignette_true.pgm", kW, kH, unit_to_pgm(v_true_map));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/gain_recovered.pgm", kW, kH, unit_to_pgm(h_gain_np_gpu));

    float dsnu_lo = h_dsnu_rec_gpu[0], dsnu_hi = h_dsnu_rec_gpu[0];
    for (float v : h_dsnu_rec_gpu) { dsnu_lo = std::min(dsnu_lo, v); dsnu_hi = std::max(dsnu_hi, v); }
    artifact_ok = artifact_ok && write_pgm(out_dir + "/dsnu_recovered.pgm", kW, kH,
                                           stretch_to_pgm(h_dsnu_rec_gpu, dsnu_lo, dsnu_hi));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/scene_uncorrected.pgm", kW, kH, h_scene_raw_u8);
    artifact_ok = artifact_ok && write_pgm(out_dir + "/scene_corrected.pgm", kW, kH, code_to_pgm(h_scene_corrected_gpu));

    {
        std::ofstream rp(out_dir + "/radial_profile.csv");
        if (rp.is_open()) {
            rp << "r_px,true_v,nonparametric_mean,fitted_v\n";
            for (int i = 0; i < kNumRadialBins; ++i) {
                const double r_center = (static_cast<double>(i) + 0.5) * kRadialBinWidthPx;
                rp << fmt(r_center, 2) << "," << fmt(v_true_of_r(r_center), 5) << ",";
                if (h_bin_count_cpu[i] > 0) rp << fmt(h_bin_sum_cpu[i] / h_bin_count_cpu[i], 5);
                rp << "," << fmt(v_fit_of_r(r_center, a2, a4, a6), 5) << "\n";
            }
        } else {
            artifact_ok = false;
        }
    }
    artifact_ok = artifact_ok && write_gates_csv(out_dir + "/gates_metrics.csv", csv);

    if (artifact_ok) {
        std::printf("ARTIFACT: wrote demo/out/{vignette_true.pgm, gain_recovered.pgm, "
                   "dsnu_recovered.pgm, scene_uncorrected.pgm, scene_corrected.pgm, "
                   "radial_profile.csv, gates_metrics.csv}\n");
    } else {
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");
    }

    // ---- free device memory ------------------------------------------------
    CUDA_CHECK(cudaFree(d_dark_stack)); CUDA_CHECK(cudaFree(d_flat_stack)); CUDA_CHECK(cudaFree(d_scene));
    CUDA_CHECK(cudaFree(d_dsnu_rec)); CUDA_CHECK(cudaFree(d_flat_avg)); CUDA_CHECK(cudaFree(d_flat_minus_dsnu));
    CUDA_CHECK(cudaFree(d_gain_np)); CUDA_CHECK(cudaFree(d_scene_corrected)); CUDA_CHECK(cudaFree(d_flat0_corrected));
    CUDA_CHECK(cudaFree(d_center_sum)); CUDA_CHECK(cudaFree(d_bin_sum)); CUDA_CHECK(cudaFree(d_bin_count));

    // ======================= RESULT =============================================
    const bool all_gates = gate_dsnu && gate_gain && gate_radial && gate_noise && gate_efficacy && gate_flatness;
    const bool overall = verify_pass && all_gates && artifact_ok;
    if (overall) {
        std::printf("RESULT: PASS (VERIFY + all 6 gates passed: dsnu_recovery, gain_recovery, "
                   "radial_fit, noise_averaging, correction_efficacy, flatness)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
