// ===========================================================================
// main.cu — entry point for project 01.22
//           Motion deblurring and super-resolution for inspection zoom
//
// What this program does, start to finish
// -----------------------------------------
//   MILESTONE 1 (deblurring): load the shared truth.pgm + blurred.pgm +
//   psf_truth.csv/psf_mismatch.csv. Run three GPU restorations of the SAME
//   blurred frame (naive inverse filter, Wiener filter, Richardson-Lucy)
//   plus a Wiener-with-wrong-PSF honesty run, each against an independent
//   CPU twin (naive_inverse_cpu/wiener_cpu use a from-scratch radix-2 CPU
//   FFT; richardson_lucy_cpu uses an independent spatial convolution loop —
//   see reference_cpu.cpp's header for the twin-independence ruling this
//   project follows).
//
//   MILESTONE 2 (super-resolution): load 8 low-res frames + their known
//   sub-pixel shifts. Combine by shift-and-add (GPU scatter kernel) onto
//   the 2x grid, refine with iterative back-projection (GPU gather
//   kernels), and compare against bicubic upscaling of a single frame —
//   each against an independent CPU twin.
//
//   Both milestones: VERIFY(gpu vs cpu) per method, INDEPENDENT gates
//   against ground truth (never routed through the shared FFT/bilinear
//   machinery — see reference_cpu.cpp's header), a noise-honesty [info]
//   report tying back to project 01.11 by name, and every artifact
//   demo/README.md describes written to demo/out/.
//
// Output contract (load-bearing, CLAUDE.md §12): "[demo]", "PROBLEM:",
// "DATA:", "VERIFY(...)", "GATE ...:", "ARTIFACT:", "RESULT:" lines are
// STABLE and diffed by demo/expected_output.txt; "[info]"/"[time]" lines
// carry the actual measured numbers and are deliberately NOT diffed.
//
// Read this after: kernels.cuh (the contract), kernels.cu (the GPU
// kernels + cuFFT wrappers), reference_cpu.cpp (the independent CPU oracles).
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ===========================================================================
// VERIFY tolerances (GPU vs CPU, per method) and GATE thresholds. Every
// number below was MEASURED on the reference machine (RTX 2080 SUPER,
// sm_75) then margined — never set AT the measured value (the 01.01/01.09/
// 01.11 discipline) — measured values are quoted in each comment; this
// program's own [info] lines reproduce the measurement on any machine it
// runs on.
// ===========================================================================

// -- VERIFY: GPU-vs-CPU tolerances (max |gpu-cpu| over all pixels, DN
// units, unless noted). naive_inverse gets the LOOSEST tolerance of the
// four milestone-1 methods on purpose: it is a DESIGNED numerical
// instability (kernels.cuh's header) — at a PSF spectral near-zero, even
// the tiny rounding difference between cuFFT (float32) and this project's
// from-scratch CPU FFT (float64) gets multiplied by a huge factor, so
// "the two disagree by a lot at an unstable bin" is EXPECTED, not a bug;
// the naive_inverse_failure GATE below (which checks PSNR, an average over
// all pixels) is the meaningful, stable check for that method.
static constexpr double kTolNaiveInverse = 0.50;    // measured ~0.0013 DN (both FFT implementations are high-precision; the DESIGNED instability amplifies whatever tiny rounding difference remains, so this floor stays well above measurement to absorb cross-GPU-architecture drift)
static constexpr double kTolWiener = 0.05;          // measured ~0.0002 DN (regularized: well-conditioned)
static constexpr double kTolRl = 0.05;              // measured ~0.0003 DN (30 spatial-conv iterations, float32 vs double-CPU)
static constexpr double kTolBicubic = 0.02;         // measured ~0.0000 DN (deterministic gather, no atomics)
static constexpr double kTolShiftAdd = 1.5;         // measured ~0.0000 DN this run (atomic float vs. fixed-order double CPU twin; the 01.11 BM3D-lite precedent keeps this the loosest tolerance since atomic ordering can vary run-to-run)
static constexpr double kTolIbp = 0.10;             // measured ~0.0000 DN (deterministic gather; SAME seed both sides)

// -- GATE thresholds.
static constexpr double kMinWienerImprovementDb = 1.5;   // measured +3.13 dB: wiener PSNR must beat blurred baseline by this much
static constexpr double kMinRlImprovementDb = 1.5;       // measured +2.35 dB: RL PSNR must beat blurred baseline by this much
static constexpr double kMinEdgePreserveFrac = 0.35;     // measured wiener 50%, rl 43%: must retain this fraction of the clean step's gradient
static constexpr double kMinMismatchDegradationDb = 3.0; // measured 7.04 dB: PSF-mismatch Wiener must be at least this much WORSE than correct-PSF Wiener
static constexpr double kMinSrCorrelation = 0.80;        // measured ~0.98: SR must genuinely match the true fine-bar pattern
static constexpr double kMinSrCorrelationMargin = 0.40;  // measured ~0.77: SR's correlation must beat bicubic's by at least this much
static constexpr double kMinSrPsnrMarginDb = 1.5;        // measured +3.36 dB: SR whole-image PSNR must beat bicubic's by at least this much

// ===========================================================================
// Minimal, STRICT PGM (P5) reader/writer — the 01.01/01.09/01.11
// convention: only ever reads files this project's own generator wrote.
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
    in.get();

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

// ===========================================================================
// CSV readers for the PSF kernels and the shift table (scripts/
// make_synthetic.py's format: a "rows,cols" line, then `rows` comma-
// separated lines of `cols` floats each — see that script's write_psf_csv).
// Comment lines ('#') are skipped.
// ===========================================================================
static bool read_next_data_line(std::ifstream& in, std::string& line)
{
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        return true;
    }
    return false;
}

static bool read_psf_csv(const std::string& path, std::vector<float>& out /* [kPsfSize*kPsfSize] */)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    if (!read_next_data_line(in, line)) return false;
    int rows = 0, cols = 0;
    {
        std::istringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ','); rows = std::atoi(tok.c_str());
        std::getline(ss, tok, ','); cols = std::atoi(tok.c_str());
    }
    if (rows != kPsfSize || cols != kPsfSize) return false;
    out.assign(static_cast<size_t>(kPsfSize) * kPsfSize, 0.0f);
    for (int r = 0; r < rows; ++r) {
        if (!read_next_data_line(in, line)) return false;
        std::istringstream ss(line);
        std::string tok;
        for (int c = 0; c < cols; ++c) {
            if (!std::getline(ss, tok, ',')) return false;
            out[static_cast<size_t>(r) * cols + c] = std::strtof(tok.c_str(), nullptr);
        }
    }
    return true;
}

static bool read_shifts_csv(const std::string& path, std::vector<Shift>& out /* [kNumFrames] */)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    // Skip '#' comments, then the "frame,dx_lrpx,dy_lrpx" header row.
    if (!read_next_data_line(in, line)) return false;   // header row (not comment-prefixed)
    out.clear();
    while (read_next_data_line(in, line)) {
        std::istringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');                 // frame index (unused: rows are in order)
        Shift s{};
        std::getline(ss, tok, ','); s.dx_lrpx = std::strtof(tok.c_str(), nullptr);
        std::getline(ss, tok, ','); s.dy_lrpx = std::strtof(tok.c_str(), nullptr);
        out.push_back(s);
    }
    return static_cast<int>(out.size()) == kNumFrames;
}

// ===========================================================================
// build_padded_psf — place a dense kPsfSize x kPsfSize PSF (offset delta =
// (ky-radius, kx-radius) from its own center) into a zero-filled kW x kH
// buffer at the WRAPAROUND position the FFT convolution theorem requires.
//
// Derivation (THEORY.md "The math" walks this in full): this project's
// spatial-domain forward model (matching kernels.cu's convolve_circular_
// kernel AND scripts/make_synthetic.py's circular_convolve, both of which
// this padding must correctly invert) is
//     out[y,x] = sum_{ky,kx} psf[ky,kx] * img[(y+ky-r)%H, (x+kx-r)%W]
// while cuFFT/this file's pointwise multiplication computes the STANDARD
// circular convolution out = img (x) padded, i.e.
//     out[y,x] = sum_{m} img[m] * padded[(y-m)%H, ...].
// Matching the two requires placing tap psf[ky,kx] (offset delta=ky-r) at
// wraparound index (r-ky) mod H — the NEGATIVE of the offset — not +delta.
// (For THIS project's specific PSF, a line segment sampled symmetrically
// about its own center, psf[-delta]==psf[+delta] exactly, so the sign
// would not have mattered here — but the formula below is the general,
// symmetry-independent one, correct for any PSF a future project variant
// might supply.)
// ===========================================================================
static void build_padded_psf(const std::vector<float>& dense, std::vector<float>& padded)
{
    padded.assign(kN, 0.0f);
    for (int ky = 0; ky < kPsfSize; ++ky) {
        const int oy = ((kPsfRadius - ky) % kH + kH) % kH;
        for (int kx = 0; kx < kPsfSize; ++kx) {
            const int ox = ((kPsfRadius - kx) % kW + kW) % kW;
            padded[static_cast<size_t>(oy) * kW + ox] = dense[static_cast<size_t>(ky) * kPsfSize + kx];
        }
    }
}

// build_flipped_psf — the dense kPsfSize x kPsfSize 180-degree rotation of
// `dense`, used by Richardson-Lucy's adjoint (correlation) convolution
// step (kernels.cu's convolve_circular_kernel header explains why RL needs
// this second kernel buffer).
static void build_flipped_psf(const std::vector<float>& dense, std::vector<float>& flipped)
{
    flipped.assign(static_cast<size_t>(kPsfSize) * kPsfSize, 0.0f);
    for (int ky = 0; ky < kPsfSize; ++ky)
        for (int kx = 0; kx < kPsfSize; ++kx)
            flipped[static_cast<size_t>(ky) * kPsfSize + kx] =
                dense[static_cast<size_t>(kPsfSize - 1 - ky) * kPsfSize + (kPsfSize - 1 - kx)];
}

// ===========================================================================
// Metric helpers (the 01.09/01.11 shapes, reused by name where the formula
// is identical — see each comment).
// ===========================================================================
static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

static double mse_whole(const std::vector<float>& a, const std::vector<float>& b)
{
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return acc / static_cast<double>(a.size());
}

static double psnr_db(double mse_val)
{
    const double floor_mse = 1e-8;
    if (mse_val < floor_mse) mse_val = floor_mse;
    return 10.0 * std::log10((255.0 * 255.0) / mse_val);
}

// edge_gradient_mean — REUSES 01.11's exact formula by name (kernels.cuh's
// header cites it): mean |I[stepX] - I[stepX-1]| over every row of
// kEdgeRegion, a horizontal finite difference straddling the known step.
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

// residual_std_in_rect — REUSES 01.11's exact formula by name: std of
// (img - known_const) over a rectangle whose ground-truth value is a
// CONSTANT (kFlatRect is flat at kFlatDn) — the noise-honesty measurement.
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

// bar_contrast — Michelson contrast (max-min)/(max+min) of `img` averaged
// over every ROW of `rect`, restricted to the region's OWN column span. An
// [info]-only metric (see bar_pattern_correlation()'s header for why this
// project does NOT gate on it): a single scalar summarizing "how much
// local amplitude survived", independent of the pattern's mean brightness
// OR its phase/frequency.
static double bar_contrast(const std::vector<float>& img, int W, const Rect& rect)
{
    double acc = 0.0;
    int rows = 0;
    for (int y = rect.y0; y < rect.y1; ++y) {
        float lo = 1.0e9f, hi = -1.0e9f;
        for (int x = rect.x0; x < rect.x1; ++x) {
            const float v = img[static_cast<size_t>(y) * W + x];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        const double denom = static_cast<double>(hi) + static_cast<double>(lo);
        if (denom > 1.0) acc += (static_cast<double>(hi) - static_cast<double>(lo)) / denom;
        ++rows;
    }
    return acc / static_cast<double>(rows);
}

// bar_pattern_correlation — the ACTUAL sr_resolution gate metric: the mean,
// over every ROW of `rect`, of the Pearson correlation coefficient between
// `img`'s row profile and `truth`'s row profile.
//
// WHY correlation and not bar_contrast() above (an honest lesson this
// project's own measurements surfaced, kept here rather than smoothed
// over): the fine bar group's period (kBarPeriodFine=3 truth-px, BELOW the
// low-resolution grid's Nyquist period of 4) means a SINGLE aliased LR
// frame does not simply "blur away" that frequency — Nyquist aliasing
// FOLDS it onto a spurious LOWER apparent frequency (a moire beat
// pattern), which still has substantial local min/max SWING. bicubic
// upscaling of that one aliased frame therefore measures a MISLEADINGLY
// high bar_contrast() (measured ~0.55, not far below SR's ~0.57) despite
// showing the WRONG pattern entirely (measured correlation with the true
// pattern: bicubic ~0.22, essentially uncorrelated; SR ~0.98, a near-exact
// match) — bar_contrast() alone cannot tell "correct high-frequency
// detail" from "aliasing artifact that happens to have similar amplitude",
// exactly the pitfall THEORY.md "The problem" warns about. Correlation
// against the (never-aliased) ground truth is the metric that actually
// answers "did this method resolve the RIGHT pattern" — the honest money
// metric, reported here after bar_contrast() failed to cleanly separate
// the two methods during this project's own tuning pass.
static double bar_pattern_correlation(const std::vector<float>& img, const std::vector<float>& truth,
                                      int W, const Rect& rect)
{
    double acc = 0.0;
    int rows = 0;
    for (int y = rect.y0; y < rect.y1; ++y) {
        double mean_t = 0.0, mean_v = 0.0;
        const int n = rect.x1 - rect.x0;
        for (int x = rect.x0; x < rect.x1; ++x) {
            mean_t += truth[static_cast<size_t>(y) * W + x];
            mean_v += img[static_cast<size_t>(y) * W + x];
        }
        mean_t /= n; mean_v /= n;
        double num = 0.0, dt = 0.0, dv = 0.0;
        for (int x = rect.x0; x < rect.x1; ++x) {
            const double t = static_cast<double>(truth[static_cast<size_t>(y) * W + x]) - mean_t;
            const double v = static_cast<double>(img[static_cast<size_t>(y) * W + x]) - mean_v;
            num += t * v; dt += t * t; dv += v * v;
        }
        if (dt > 1.0e-9 && dv > 1.0e-9) {
            acc += num / (std::sqrt(dt) * std::sqrt(dv));
            ++rows;
        }
    }
    return rows > 0 ? acc / static_cast<double>(rows) : 0.0;
}

// crop_into — copy a W-wide image's [rect] region into a `dst` buffer of
// size (rect.x1-rect.x0) x (rect.y1-rect.y0), used to build the bar-chart
// side-by-side composite artifact below.
static void crop_into(const std::vector<float>& img, int W, const Rect& r, std::vector<float>& dst)
{
    const int cw = r.x1 - r.x0, ch = r.y1 - r.y0;
    dst.assign(static_cast<size_t>(cw) * ch, 0.0f);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x)
            dst[static_cast<size_t>(y) * cw + x] = img[static_cast<size_t>(r.y0 + y) * W + (r.x0 + x)];
}

// ===========================================================================
// gates_metrics.csv writer (the 01.01/01.09/01.11 shape).
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

static bool write_curve_csv(const std::string& path, const char* header, const std::vector<float>& curve)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << header << "\n";
    for (size_t i = 0; i < curve.size(); ++i) out << i << "," << curve[i] << "\n";
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

    std::printf("[demo] motion deblurring (Wiener / naive-inverse / Richardson-Lucy) and multi-frame "
               "super-resolution (shift-and-add + IBP vs bicubic) for inspection zoom (project 01.22)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d inspection scene, line-motion-blur PSF length=%.1fpx angle=%.1fdeg, "
               "%d low-res frames at %dx%d (scale %dx) with known quarter-LR-pixel shifts, FP32\n",
               kW, kH, static_cast<double>(kBlurLengthPx), static_cast<double>(kBlurAngleDeg),
               kNumFrames, kLrW, kLrH, kLrScale);

    // ---- data ---------------------------------------------------------------
    const std::string truth_path = find_data_file(data_dir, argv[0], "truth.pgm");
    const std::string blurred_path = find_data_file(data_dir, argv[0], "blurred.pgm");
    const std::string psf_truth_path = find_data_file(data_dir, argv[0], "psf_truth.csv");
    const std::string psf_mismatch_path = find_data_file(data_dir, argv[0], "psf_mismatch.csv");
    const std::string shifts_path = find_data_file(data_dir, argv[0], "shifts_truth.csv");

    int w = 0, h = 0;
    std::vector<unsigned char> truth_u8, blurred_u8;
    std::vector<float> psf_dense, psf_mismatch_dense;
    std::vector<Shift> shifts;
    std::vector<unsigned char> lr_u8[kNumFrames];
    bool loaded = !truth_path.empty() && !blurred_path.empty()
               && read_pgm(truth_path, w, h, truth_u8) && w == kW && h == kH
               && read_pgm(blurred_path, w, h, blurred_u8) && w == kW && h == kH
               && read_psf_csv(psf_truth_path, psf_dense)
               && read_psf_csv(psf_mismatch_path, psf_mismatch_dense)
               && read_shifts_csv(shifts_path, shifts);
    for (int f = 0; loaded && f < kNumFrames; ++f) {
        const std::string p = find_data_file(data_dir, argv[0], ("lr_frame_" + std::to_string(f) + ".pgm").c_str());
        int lw = 0, lh = 0;
        loaded = loaded && !p.empty() && read_pgm(p, lw, lh, lr_u8[f]) && lw == kLrW && lh == kLrH;
    }
    if (!loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: synthetic inspection scene (flat patch + step edge + 7 dot-matrix glyphs + hashed "
               "texture + 3 bar-chart frequency groups); truth.pgm + blurred.pgm (exact line-PSF "
               "convolution + Gaussian noise) + %d low-res frames on a quarter-pixel shift lattice "
               "[synthetic, seed 42]\n", kNumFrames);

    std::vector<float> h_truth(kN), h_blurred(kN);
    for (int i = 0; i < kN; ++i) { h_truth[i] = static_cast<float>(truth_u8[i]); h_blurred[i] = static_cast<float>(blurred_u8[i]); }
    std::vector<float> h_lr_frames(static_cast<size_t>(kLrFramesN));
    for (int f = 0; f < kNumFrames; ++f)
        for (int i = 0; i < kLrN; ++i)
            h_lr_frames[static_cast<size_t>(f) * kLrN + i] = static_cast<float>(lr_u8[f][i]);

    // ===========================================================================
    // MILESTONE 1 — motion deblurring.
    // ===========================================================================
    std::vector<float> psf_padded, psf_mismatch_padded, psf_flipped;
    build_padded_psf(psf_dense, psf_padded);
    build_padded_psf(psf_mismatch_dense, psf_mismatch_padded);
    build_flipped_psf(psf_dense, psf_flipped);

    // -- device buffers ---------------------------------------------------------
    float *d_blurred = nullptr, *d_psf_padded = nullptr, *d_psf_mismatch_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blurred, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psf_padded, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psf_mismatch_padded, kN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_blurred, h_blurred.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psf_padded, psf_padded.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psf_mismatch_padded, psf_mismatch_padded.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

    ComplexF32 *d_blurred_freq = nullptr, *d_psf_freq = nullptr, *d_psf_mismatch_freq = nullptr;
    ComplexF32 *d_naive_freq = nullptr, *d_wiener_freq = nullptr, *d_wiener_mismatch_freq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blurred_freq, kFreqN * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_psf_freq, kFreqN * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_psf_mismatch_freq, kFreqN * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_naive_freq, kFreqN * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_wiener_freq, kFreqN * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_wiener_mismatch_freq, kFreqN * sizeof(ComplexF32)));

    GpuTimer gt;
    gt.begin();
    launch_fft_forward_r2c(d_blurred, d_blurred_freq);
    launch_fft_forward_r2c(d_psf_padded, d_psf_freq);
    launch_fft_forward_r2c(d_psf_mismatch_padded, d_psf_mismatch_freq);
    launch_naive_inverse(d_blurred_freq, d_psf_freq, d_naive_freq);
    launch_wiener(d_blurred_freq, d_psf_freq, d_wiener_freq, kWienerK);
    launch_wiener(d_blurred_freq, d_psf_mismatch_freq, d_wiener_mismatch_freq, kWienerK);

    float *d_naive = nullptr, *d_wiener = nullptr, *d_wiener_mismatch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_naive, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wiener, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wiener_mismatch, kN * sizeof(float)));
    launch_fft_inverse_c2r(d_naive_freq, d_naive);
    launch_fft_inverse_c2r(d_wiener_freq, d_wiener);
    launch_fft_inverse_c2r(d_wiener_mismatch_freq, d_wiener_mismatch);
    const float kInvN = 1.0f / static_cast<float>(kN);   // cuFFT's C2R is unnormalized (kernels.cu's header)
    launch_scale_real(d_naive, kN, kInvN);
    launch_scale_real(d_wiener, kN, kInvN);
    launch_scale_real(d_wiener_mismatch, kN, kInvN);
    const float ms_fft_stage = gt.end_ms();

    // -- Richardson-Lucy on the GPU: main.cu composes the per-iteration
    // kernels itself (convolve/divide/multiply — no single "launch_rl"
    // wrapper), recording the data-fidelity MSE curve as it goes (a
    // measurement-only, ground-truth-free convergence diagnostic — the
    // SAME quantity reference_cpu.cpp's richardson_lucy_cpu records for
    // its own twin, so VERIFY can additionally sanity-check the curves
    // agree in shape even though only the final estimate is diffed).
    float *d_psf_dense = nullptr, *d_psf_flipped = nullptr;
    CUDA_CHECK(cudaMalloc(&d_psf_dense, static_cast<size_t>(kPsfSize) * kPsfSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psf_flipped, static_cast<size_t>(kPsfSize) * kPsfSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_psf_dense, psf_dense.data(), psf_dense.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psf_flipped, psf_flipped.data(), psf_flipped.size() * sizeof(float), cudaMemcpyHostToDevice));

    float *d_rl_estimate = nullptr, *d_rl_reblur = nullptr, *d_rl_ratio = nullptr, *d_rl_correction = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rl_estimate, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rl_reblur, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rl_ratio, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rl_correction, kN * sizeof(float)));
    // RL's classic starting point: the blurred observation itself (a
    // standard, uninformative initialization — THEORY.md "The algorithm").
    CUDA_CHECK(cudaMemcpy(d_rl_estimate, d_blurred, kN * sizeof(float), cudaMemcpyDeviceToDevice));

    std::vector<float> rl_curve_gpu(kRlIterations);
    gt.begin();
    for (int it = 0; it < kRlIterations; ++it) {
        launch_convolve_circular(d_rl_estimate, d_psf_dense, d_rl_reblur);
        std::vector<float> h_reblur(kN);
        CUDA_CHECK(cudaMemcpy(h_reblur.data(), d_rl_reblur, kN * sizeof(float), cudaMemcpyDeviceToHost));
        double acc = 0.0;
        for (int i = 0; i < kN; ++i) { const double d = static_cast<double>(h_reblur[i]) - static_cast<double>(h_blurred[i]); acc += d * d; }
        rl_curve_gpu[it] = static_cast<float>(acc / static_cast<double>(kN));
        launch_divide_safe(d_blurred, d_rl_reblur, d_rl_ratio, kN, kRlEpsilon);
        launch_convolve_circular(d_rl_ratio, d_psf_flipped, d_rl_correction);
        launch_multiply_inplace(d_rl_estimate, d_rl_correction, kN);
    }
    const float ms_rl = gt.end_ms();

    std::vector<float> h_naive(kN), h_wiener(kN), h_wiener_mismatch(kN), h_rl(kN);
    CUDA_CHECK(cudaMemcpy(h_naive.data(), d_naive, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wiener.data(), d_wiener, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wiener_mismatch.data(), d_wiener_mismatch, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rl.data(), d_rl_estimate, kN * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("[time] GPU milestone 1: fft-stage(fwd x3 + naive + wiener x2 + inv x3 + scale) %.3f ms | "
               "richardson-lucy (%d iters) %.3f ms\n",
               static_cast<double>(ms_fft_stage), kRlIterations, static_cast<double>(ms_rl));

    // -- CPU reference oracles ---------------------------------------------------
    CpuTimer ct;
    std::vector<float> h_naive_cpu(kN), h_wiener_cpu(kN), h_wiener_mismatch_cpu(kN), h_rl_cpu(kN, 0.0f);
    ct.begin(); naive_inverse_cpu(h_blurred.data(), psf_padded.data(), h_naive_cpu.data()); const double cms_naive = ct.end_ms();
    ct.begin(); wiener_cpu(h_blurred.data(), psf_padded.data(), kWienerK, h_wiener_cpu.data()); const double cms_wiener = ct.end_ms();
    ct.begin(); wiener_cpu(h_blurred.data(), psf_mismatch_padded.data(), kWienerK, h_wiener_mismatch_cpu.data()); const double cms_wiener_mm = ct.end_ms();
    std::copy(h_blurred.begin(), h_blurred.end(), h_rl_cpu.begin());   // same RL starting point as the GPU path
    std::vector<float> rl_curve_cpu(kRlIterations);
    ct.begin(); richardson_lucy_cpu(h_blurred.data(), psf_dense.data(), h_rl_cpu.data(), kRlIterations, rl_curve_cpu.data()); const double cms_rl = ct.end_ms();
    std::printf("[time] CPU reference (single-thread): naive_inverse %.1f ms | wiener %.1f ms | "
               "wiener_mismatch %.1f ms | richardson_lucy %.1f ms\n", cms_naive, cms_wiener, cms_wiener_mm, cms_rl);

    // -- VERIFY: GPU vs CPU, milestone 1 -----------------------------------------
    const double diff_naive = max_abs_diff(h_naive, h_naive_cpu);
    const double diff_wiener = max_abs_diff(h_wiener, h_wiener_cpu);
    const double diff_rl = max_abs_diff(h_rl, h_rl_cpu);
    const bool verify_naive = diff_naive <= kTolNaiveInverse;
    const bool verify_wiener = diff_wiener <= kTolWiener;
    const bool verify_rl = diff_rl <= kTolRl;

    std::printf("[info] verify(naive_inverse): max|gpu-cpu|=%.4f DN (tol %.1f -- a DESIGNED instability, "
               "see kernels.cuh; large disagreement at unstable bins is expected, not a bug)\n", diff_naive, kTolNaiveInverse);
    std::printf("VERIFY(naive_inverse): %s (GPU cuFFT path matches independent CPU radix-2 FFT twin within tolerance)\n",
               verify_naive ? "PASS" : "FAIL");
    std::printf("[info] verify(wiener): max|gpu-cpu|=%.4f DN (tol %.2f)\n", diff_wiener, kTolWiener);
    std::printf("VERIFY(wiener): %s (GPU cuFFT path matches independent CPU radix-2 FFT twin within tolerance)\n",
               verify_wiener ? "PASS" : "FAIL");
    std::printf("[info] verify(richardson_lucy): max|gpu-cpu|=%.4f DN (tol %.2f, %d iterations)\n", diff_rl, kTolRl, kRlIterations);
    std::printf("VERIFY(richardson_lucy): %s (GPU spatial convolution matches independent CPU twin within tolerance)\n",
               verify_rl ? "PASS" : "FAIL");

    // -- GATE 1: wiener_recovery + edge preservation -----------------------------
    const double mse_blurred = mse_whole(h_blurred, h_truth);
    const double psnr_blurred = psnr_db(mse_blurred);
    const double psnr_wiener = psnr_db(mse_whole(h_wiener, h_truth));
    const double psnr_rl = psnr_db(mse_whole(h_rl, h_truth));
    const double psnr_naive = psnr_db(mse_whole(h_naive, h_truth));

    const double clean_edge = edge_gradient_mean(h_truth, kW);
    const double blurred_edge_frac = edge_gradient_mean(h_blurred, kW) / clean_edge;
    const double wiener_edge_frac = edge_gradient_mean(h_wiener, kW) / clean_edge;
    const double rl_edge_frac = edge_gradient_mean(h_rl, kW) / clean_edge;

    const bool wiener_psnr_ok = (psnr_wiener - psnr_blurred) >= kMinWienerImprovementDb;
    const bool wiener_edge_ok = wiener_edge_frac >= kMinEdgePreserveFrac;
    const bool gate_wiener = wiener_psnr_ok && wiener_edge_ok;
    std::printf("GATE wiener_recovery: %s\n", gate_wiener ? "PASS" : "FAIL");
    std::printf("[info] wiener_recovery: PSNR blurred=%.2f dB | wiener=%.2f dB (+%.2f, need >=+%.1f) | "
               "edge fraction: clean=%.1f DN | blurred=%.0f%% | wiener=%.0f%% (need >=%.0f%%)\n",
               psnr_blurred, psnr_wiener, psnr_wiener - psnr_blurred, kMinWienerImprovementDb,
               clean_edge, 100.0 * blurred_edge_frac, 100.0 * wiener_edge_frac, 100.0 * kMinEdgePreserveFrac);

    // -- GATE 2: naive_inverse_failure (the designed demonstration) --------------
    const bool gate_naive_fail = psnr_naive < psnr_blurred;   // must be WORSE than doing nothing
    std::printf("GATE naive_inverse_failure: %s\n", gate_naive_fail ? "PASS" : "FAIL");
    std::printf("[info] naive_inverse_failure: PSNR blurred=%.2f dB | naive_inverse=%.2f dB (%.2f dB %s than "
               "blurred -- noise amplification at PSF spectral near-zeros, kernels.cuh Section 2)\n",
               psnr_blurred, psnr_naive, psnr_naive - psnr_blurred, psnr_naive < psnr_blurred ? "WORSE" : "BETTER");

    // -- GATE 3: rl_recovery -------------------------------------------------------
    const bool rl_psnr_ok = (psnr_rl - psnr_blurred) >= kMinRlImprovementDb;
    const bool rl_edge_ok = rl_edge_frac >= kMinEdgePreserveFrac;
    const bool gate_rl = rl_psnr_ok && rl_edge_ok;
    std::printf("GATE rl_recovery: %s\n", gate_rl ? "PASS" : "FAIL");
    std::printf("[info] rl_recovery: PSNR rl=%.2f dB (+%.2f over blurred, need >=+%.1f) | edge fraction "
               "rl=%.0f%% (need >=%.0f%%) | data-fidelity MSE: iter0=%.2f -> iter%d=%.2f\n",
               psnr_rl, psnr_rl - psnr_blurred, kMinRlImprovementDb, 100.0 * rl_edge_frac, 100.0 * kMinEdgePreserveFrac,
               static_cast<double>(rl_curve_gpu.front()), kRlIterations - 1, static_cast<double>(rl_curve_gpu.back()));

    // -- GATE 4: psf_mismatch honesty ----------------------------------------------
    const double psnr_wiener_mismatch = psnr_db(mse_whole(h_wiener_mismatch, h_truth));
    const double mismatch_degradation_db = psnr_wiener - psnr_wiener_mismatch;
    const bool gate_mismatch = mismatch_degradation_db >= kMinMismatchDegradationDb;
    std::printf("GATE psf_mismatch: %s\n", gate_mismatch ? "PASS" : "FAIL");
    std::printf("[info] psf_mismatch: correct-PSF wiener=%.2f dB (angle=%.1fdeg) | wrong-PSF wiener=%.2f dB "
               "(angle=%.1fdeg, off by %.1fdeg) | degradation=%.2f dB (need >=%.1f dB -- non-blind "
               "deconvolution's Achilles heel: an accurate PSF is REQUIRED)\n",
               psnr_wiener, static_cast<double>(kBlurAngleDeg), psnr_wiener_mismatch, static_cast<double>(kMismatchAngleDeg),
               static_cast<double>(kMismatchAngleDeg - kBlurAngleDeg), mismatch_degradation_db, kMinMismatchDegradationDb);

    // ===========================================================================
    // MILESTONE 2 — multi-frame super-resolution.
    // ===========================================================================
    float* d_lr_frames = nullptr;
    Shift* d_shifts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lr_frames, static_cast<size_t>(kLrFramesN) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shifts, static_cast<size_t>(kNumFrames) * sizeof(Shift)));
    CUDA_CHECK(cudaMemcpy(d_lr_frames, h_lr_frames.data(), h_lr_frames.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shifts, shifts.data(), shifts.size() * sizeof(Shift), cudaMemcpyHostToDevice));

    float* d_bicubic = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bicubic, kN * sizeof(float)));
    gt.begin();
    launch_bicubic_upscale(d_lr_frames /* frame 0 sits at offset 0 */, d_bicubic);
    const float ms_bicubic = gt.end_ms();

    float *d_hr_sum = nullptr, *d_hr_weight = nullptr, *d_shift_add = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hr_sum, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hr_weight, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shift_add, kN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hr_sum, 0, kN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hr_weight, 0, kN * sizeof(float)));
    gt.begin();
    launch_shift_and_add(d_lr_frames, d_shifts, d_hr_sum, d_hr_weight);
    launch_finalize_splat(d_hr_sum, d_hr_weight, d_bicubic, d_shift_add);
    const float ms_shift_add = gt.end_ms();

    // IBP refines the shift-and-add result. Started from the SAME buffer
    // whose host copy will also seed the CPU IBP twin below (main.cu's
    // choice, see the file header's VERIFY-isolation note) — so VERIFY(ibp)
    // measures ONLY the forward/back-projection formula's agreement, not a
    // compounded difference from two different shift-and-add starting points.
    float *d_sr_estimate = nullptr, *d_lr_predicted = nullptr, *d_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sr_estimate, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lr_predicted, static_cast<size_t>(kLrFramesN) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, static_cast<size_t>(kLrFramesN) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sr_estimate, d_shift_add, kN * sizeof(float), cudaMemcpyDeviceToDevice));

    std::vector<float> ibp_curve_gpu(kIbpIterations);
    gt.begin();
    for (int it = 0; it < kIbpIterations; ++it) {
        launch_forward_simulate(d_sr_estimate, d_shifts, d_lr_predicted);
        launch_subtract(d_lr_frames, d_lr_predicted, d_residual, kLrFramesN);
        std::vector<float> h_residual(kLrFramesN);
        CUDA_CHECK(cudaMemcpy(h_residual.data(), d_residual, static_cast<size_t>(kLrFramesN) * sizeof(float), cudaMemcpyDeviceToHost));
        double acc_sq = 0.0;
        for (int i = 0; i < kLrFramesN; ++i) acc_sq += static_cast<double>(h_residual[i]) * static_cast<double>(h_residual[i]);
        ibp_curve_gpu[it] = static_cast<float>(std::sqrt(acc_sq / static_cast<double>(kLrFramesN)));
        launch_backproject(d_residual, d_shifts, d_sr_estimate, kIbpStep);
    }
    const float ms_ibp = gt.end_ms();

    std::vector<float> h_bicubic(kN), h_shift_add(kN), h_sr(kN);
    CUDA_CHECK(cudaMemcpy(h_bicubic.data(), d_bicubic, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_shift_add.data(), d_shift_add, kN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sr.data(), d_sr_estimate, kN * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("[time] GPU milestone 2: bicubic %.3f ms | shift-and-add %.3f ms | IBP (%d iters) %.3f ms\n",
               static_cast<double>(ms_bicubic), static_cast<double>(ms_shift_add), kIbpIterations, static_cast<double>(ms_ibp));

    // -- CPU reference oracles, milestone 2 --------------------------------------
    std::vector<float> h_bicubic_cpu(kN), h_shift_add_cpu(kN), h_sr_cpu(kN);
    ct.begin(); bicubic_upscale_cpu(h_lr_frames.data(), h_bicubic_cpu.data()); const double cms_bicubic = ct.end_ms();
    ct.begin(); shift_and_add_cpu(h_lr_frames.data(), shifts.data(), h_shift_add_cpu.data()); const double cms_shift_add = ct.end_ms();
    // IBP CPU twin seeded from the GPU's shift-and-add result (see the
    // isolation note above) so VERIFY(ibp) is not confounded by the
    // (already separately verified, looser-tolerance) shift-and-add step.
    std::copy(h_shift_add.begin(), h_shift_add.end(), h_sr_cpu.begin());
    std::vector<float> ibp_curve_cpu(kIbpIterations);
    ct.begin(); ibp_refine_cpu(h_lr_frames.data(), shifts.data(), h_sr_cpu.data(), kIbpIterations, ibp_curve_cpu.data()); const double cms_ibp = ct.end_ms();
    std::printf("[time] CPU reference (single-thread): bicubic %.1f ms | shift_and_add %.1f ms | ibp %.1f ms\n",
               cms_bicubic, cms_shift_add, cms_ibp);

    // -- VERIFY: GPU vs CPU, milestone 2 -----------------------------------------
    const double diff_bicubic = max_abs_diff(h_bicubic, h_bicubic_cpu);
    const double diff_shift_add = max_abs_diff(h_shift_add, h_shift_add_cpu);
    const double diff_ibp = max_abs_diff(h_sr, h_sr_cpu);
    const bool verify_bicubic = diff_bicubic <= kTolBicubic;
    const bool verify_shift_add = diff_shift_add <= kTolShiftAdd;
    const bool verify_ibp = diff_ibp <= kTolIbp;

    std::printf("[info] verify(bicubic): max|gpu-cpu|=%.4f DN (tol %.2f)\n", diff_bicubic, kTolBicubic);
    std::printf("VERIFY(bicubic): %s (GPU gather kernel matches independent CPU twin within tolerance)\n", verify_bicubic ? "PASS" : "FAIL");
    std::printf("[info] verify(shift_and_add): max|gpu-cpu|=%.4f DN (tol %.1f -- atomic float scatter vs. "
               "fixed-order double CPU twin, the 01.11 BM3D-lite precedent)\n", diff_shift_add, kTolShiftAdd);
    std::printf("VERIFY(shift_and_add): %s (GPU atomic-scatter splat matches independent CPU twin within tolerance)\n",
               verify_shift_add ? "PASS" : "FAIL");
    std::printf("[info] verify(ibp): max|gpu-cpu|=%.4f DN (tol %.2f, %d iterations, SAME seed both sides)\n",
               diff_ibp, kTolIbp, kIbpIterations);
    std::printf("VERIFY(ibp): %s (GPU gather-based forward/back-projection matches independent CPU twin within tolerance)\n",
               verify_ibp ? "PASS" : "FAIL");

    // -- GATE 5: sr_resolution (the money shot) ------------------------------------
    // Both metrics are measured and reported; only CORRELATION gates (see
    // bar_pattern_correlation()'s header for why raw contrast alone proved
    // to be a misleading metric during this project's own tuning pass).
    const double contrast_bicubic_fine = bar_contrast(h_bicubic, kW, kBarFineRegion);
    const double contrast_sr_fine = bar_contrast(h_sr, kW, kBarFineRegion);
    const double corr_bicubic_fine = bar_pattern_correlation(h_bicubic, h_truth, kW, kBarFineRegion);
    const double corr_sr_fine = bar_pattern_correlation(h_sr, h_truth, kW, kBarFineRegion);

    const double psnr_bicubic = psnr_db(mse_whole(h_bicubic, h_truth));
    const double psnr_sr = psnr_db(mse_whole(h_sr, h_truth));

    const bool sr_corr_ok = corr_sr_fine >= kMinSrCorrelation;
    const bool sr_margin_ok = (corr_sr_fine - corr_bicubic_fine) >= kMinSrCorrelationMargin;
    const bool sr_psnr_ok = (psnr_sr - psnr_bicubic) >= kMinSrPsnrMarginDb;
    const bool gate_sr_resolution = sr_corr_ok && sr_margin_ok && sr_psnr_ok;
    std::printf("GATE sr_resolution: %s\n", gate_sr_resolution ? "PASS" : "FAIL");
    std::printf("[info] sr_resolution: fine-bar (period %dpx, below the low-res grid's %dpx Nyquist period) "
               "pattern CORRELATION vs ground truth: bicubic=%.3f | sr=%.3f (need sr>=%.2f AND margin "
               "sr-bicubic>=%.2f) -- for context, raw Michelson CONTRAST (a MISLEADING metric here, see "
               "bar_pattern_correlation()'s comment: aliasing inflates bicubic's apparent amplitude without "
               "recovering the true pattern): bicubic=%.3f | sr=%.3f | whole-image PSNR: bicubic=%.2f dB | "
               "sr=%.2f dB (+%.2f, need >=+%.1f)\n",
               kBarPeriodFine, kBarPeriodMid, corr_bicubic_fine, corr_sr_fine, kMinSrCorrelation, kMinSrCorrelationMargin,
               contrast_bicubic_fine, contrast_sr_fine, psnr_bicubic, psnr_sr, psnr_sr - psnr_bicubic, kMinSrPsnrMarginDb);

    // -- GATE 6: sr_consistency (IBP reprojection error decreases monotonically) --
    int increases = 0;
    for (int it = 1; it < kIbpIterations; ++it)
        if (ibp_curve_gpu[it] > ibp_curve_gpu[it - 1] + 1.0e-4f) ++increases;
    const bool gate_sr_consistency = (ibp_curve_gpu.back() < ibp_curve_gpu.front()) && (increases == 0);
    std::printf("GATE sr_consistency: %s\n", gate_sr_consistency ? "PASS" : "FAIL");
    std::printf("[info] sr_consistency: IBP reprojection RMS iter0=%.4f DN -> iter%d=%.4f DN "
               "(%d/%d iterations increased -- must be 0 for a monotone gate)\n",
               static_cast<double>(ibp_curve_gpu.front()), kIbpIterations - 1, static_cast<double>(ibp_curve_gpu.back()),
               increases, kIbpIterations - 1);

    // -- [info] noise honesty: flat-region residual std, every method, both
    // milestones -- ties to project 01.11's flat_noise_floor gate BY NAME
    // (kernels.cuh's header): restoration trades noise against detail, and
    // this project's "restoration" methods are no exception -- some
    // amplify noise (naive inverse, dramatically), others suppress it
    // (Wiener/RL/SR) at the cost of some sharpness.
    const double flat_std_truth_baseline = residual_std_in_rect(h_truth, kW, kFlatRect, kFlatDn);   // ~0 (truth is exact)
    const double flat_std_blurred = residual_std_in_rect(h_blurred, kW, kFlatRect, kFlatDn);
    const double flat_std_naive = residual_std_in_rect(h_naive, kW, kFlatRect, kFlatDn);
    const double flat_std_wiener = residual_std_in_rect(h_wiener, kW, kFlatRect, kFlatDn);
    const double flat_std_rl = residual_std_in_rect(h_rl, kW, kFlatRect, kFlatDn);
    const double flat_std_bicubic = residual_std_in_rect(h_bicubic, kW, kFlatRect, kFlatDn);
    const double flat_std_sr = residual_std_in_rect(h_sr, kW, kFlatRect, kFlatDn);
    std::printf("[info] noise honesty (flat-region residual std, DN; ties to project 01.11's "
               "flat_noise_floor gate -- EVERY restoration method here also trades noise against "
               "detail, not just denoisers): truth~%.2f | blurred=%.2f | naive_inverse=%.2f "
               "(AMPLIFIED) | wiener=%.2f | richardson_lucy=%.2f | bicubic=%.2f | sr=%.2f\n",
               flat_std_truth_baseline, flat_std_blurred, flat_std_naive, flat_std_wiener, flat_std_rl,
               flat_std_bicubic, flat_std_sr);

    // ===========================================================================
    // ARTIFACTS
    // ===========================================================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();

    artifact_ok = artifact_ok && write_pgm(out_dir + "/truth.pgm", kW, kH, dn_to_pgm(h_truth));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/blurred.pgm", kW, kH, dn_to_pgm(h_blurred));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/wiener.pgm", kW, kH, dn_to_pgm(h_wiener));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/naive_inverse.pgm", kW, kH, dn_to_pgm(h_naive));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/rl.pgm", kW, kH, dn_to_pgm(h_rl));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/lr_frame_0.pgm", kLrW, kLrH, lr_u8[0]);
    artifact_ok = artifact_ok && write_pgm(out_dir + "/bicubic.pgm", kW, kH, dn_to_pgm(h_bicubic));
    artifact_ok = artifact_ok && write_pgm(out_dir + "/sr.pgm", kW, kH, dn_to_pgm(h_sr));

    // Bar-chart side-by-side composite ("the money shot"): truth | bicubic
    // | sr crops of the FINE (aliased) frequency group, separated by a
    // thin black bar, so a learner can see by eye what the contrast-ratio
    // gate measures numerically.
    {
        std::vector<float> c_truth, c_bicubic, c_sr;
        crop_into(h_truth, kW, kBarFineRegion, c_truth);
        crop_into(h_bicubic, kW, kBarFineRegion, c_bicubic);
        crop_into(h_sr, kW, kBarFineRegion, c_sr);
        const int cw = kBarFineRegion.x1 - kBarFineRegion.x0, ch = kBarFineRegion.y1 - kBarFineRegion.y0;
        const int sep = 4;
        const int compW = cw * 3 + sep * 2;
        std::vector<float> comp(static_cast<size_t>(compW) * ch, 0.0f);   // separators stay black (0)
        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                comp[static_cast<size_t>(y) * compW + x] = c_truth[static_cast<size_t>(y) * cw + x];
                comp[static_cast<size_t>(y) * compW + (cw + sep + x)] = c_bicubic[static_cast<size_t>(y) * cw + x];
                comp[static_cast<size_t>(y) * compW + (2 * (cw + sep) + x)] = c_sr[static_cast<size_t>(y) * cw + x];
            }
        }
        artifact_ok = artifact_ok && write_pgm(out_dir + "/bar_chart_comparison.pgm", compW, ch, dn_to_pgm(comp));
    }

    artifact_ok = artifact_ok && write_curve_csv(out_dir + "/rl_convergence.csv",
                                                "iteration,data_fidelity_mse_dn2", rl_curve_gpu);
    artifact_ok = artifact_ok && write_curve_csv(out_dir + "/ibp_convergence.csv",
                                                "iteration,reprojection_rms_dn", ibp_curve_gpu);

    std::vector<CsvRow> csv;
    csv.push_back({ "verify", "naive_inverse_max_abs_diff", fmt(diff_naive, 4), fmt(kTolNaiveInverse, 2), verify_naive ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "wiener_max_abs_diff", fmt(diff_wiener, 4), fmt(kTolWiener, 2), verify_wiener ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "richardson_lucy_max_abs_diff", fmt(diff_rl, 4), fmt(kTolRl, 2), verify_rl ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "bicubic_max_abs_diff", fmt(diff_bicubic, 4), fmt(kTolBicubic, 2), verify_bicubic ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "shift_and_add_max_abs_diff", fmt(diff_shift_add, 4), fmt(kTolShiftAdd, 1), verify_shift_add ? "PASS" : "FAIL" });
    csv.push_back({ "verify", "ibp_max_abs_diff", fmt(diff_ibp, 4), fmt(kTolIbp, 2), verify_ibp ? "PASS" : "FAIL" });
    csv.push_back({ "wiener_recovery", "psnr_blurred_db", fmt(psnr_blurred, 2), "n/a (baseline)", "n/a" });
    csv.push_back({ "wiener_recovery", "psnr_wiener_db", fmt(psnr_wiener, 2), fmt(psnr_blurred + kMinWienerImprovementDb, 2), wiener_psnr_ok ? "PASS" : "FAIL" });
    csv.push_back({ "wiener_recovery", "edge_frac_wiener", fmt(wiener_edge_frac, 3), fmt(kMinEdgePreserveFrac, 2), wiener_edge_ok ? "PASS" : "FAIL" });
    csv.push_back({ "naive_inverse_failure", "psnr_naive_inverse_db", fmt(psnr_naive, 2), std::string("< ") + fmt(psnr_blurred, 2), gate_naive_fail ? "PASS" : "FAIL" });
    csv.push_back({ "rl_recovery", "psnr_rl_db", fmt(psnr_rl, 2), fmt(psnr_blurred + kMinRlImprovementDb, 2), rl_psnr_ok ? "PASS" : "FAIL" });
    csv.push_back({ "rl_recovery", "edge_frac_rl", fmt(rl_edge_frac, 3), fmt(kMinEdgePreserveFrac, 2), rl_edge_ok ? "PASS" : "FAIL" });
    csv.push_back({ "psf_mismatch", "degradation_db", fmt(mismatch_degradation_db, 2), fmt(kMinMismatchDegradationDb, 1), gate_mismatch ? "PASS" : "FAIL" });
    csv.push_back({ "sr_resolution", "fine_correlation_sr", fmt(corr_sr_fine, 3), fmt(kMinSrCorrelation, 2), sr_corr_ok ? "PASS" : "FAIL" });
    csv.push_back({ "sr_resolution", "fine_correlation_margin_sr_minus_bicubic", fmt(corr_sr_fine - corr_bicubic_fine, 3), fmt(kMinSrCorrelationMargin, 2), sr_margin_ok ? "PASS" : "FAIL" });
    csv.push_back({ "sr_resolution", "fine_contrast_bicubic_INFO_ONLY", fmt(contrast_bicubic_fine, 3), "n/a (misleading here, see comment)", "n/a" });
    csv.push_back({ "sr_resolution", "fine_contrast_sr_INFO_ONLY", fmt(contrast_sr_fine, 3), "n/a (misleading here, see comment)", "n/a" });
    csv.push_back({ "sr_resolution", "psnr_margin_db", fmt(psnr_sr - psnr_bicubic, 2), fmt(kMinSrPsnrMarginDb, 2), sr_psnr_ok ? "PASS" : "FAIL" });
    csv.push_back({ "sr_consistency", "ibp_iterations_that_increased", fmt(increases, 0), "0", gate_sr_consistency ? "PASS" : "FAIL" });
    csv.push_back({ "noise_honesty", "flat_std_naive_inverse_dn", fmt(flat_std_naive, 2), "n/a (informational)", "n/a" });
    csv.push_back({ "noise_honesty", "flat_std_wiener_dn", fmt(flat_std_wiener, 2), "n/a (informational)", "n/a" });
    csv.push_back({ "noise_honesty", "flat_std_sr_dn", fmt(flat_std_sr, 2), "n/a (informational)", "n/a" });
    artifact_ok = artifact_ok && write_gates_csv(out_dir + "/gates_metrics.csv", csv);

    if (artifact_ok) {
        std::printf("ARTIFACT: wrote demo/out/{truth.pgm, blurred.pgm, wiener.pgm, naive_inverse.pgm, rl.pgm, "
                   "lr_frame_0.pgm, bicubic.pgm, sr.pgm, bar_chart_comparison.pgm, rl_convergence.csv, "
                   "ibp_convergence.csv, gates_metrics.csv}\n");
    } else {
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");
    }

    // ---- free device memory ----------------------------------------------------
    CUDA_CHECK(cudaFree(d_blurred)); CUDA_CHECK(cudaFree(d_psf_padded)); CUDA_CHECK(cudaFree(d_psf_mismatch_padded));
    CUDA_CHECK(cudaFree(d_blurred_freq)); CUDA_CHECK(cudaFree(d_psf_freq)); CUDA_CHECK(cudaFree(d_psf_mismatch_freq));
    CUDA_CHECK(cudaFree(d_naive_freq)); CUDA_CHECK(cudaFree(d_wiener_freq)); CUDA_CHECK(cudaFree(d_wiener_mismatch_freq));
    CUDA_CHECK(cudaFree(d_naive)); CUDA_CHECK(cudaFree(d_wiener)); CUDA_CHECK(cudaFree(d_wiener_mismatch));
    CUDA_CHECK(cudaFree(d_psf_dense)); CUDA_CHECK(cudaFree(d_psf_flipped));
    CUDA_CHECK(cudaFree(d_rl_estimate)); CUDA_CHECK(cudaFree(d_rl_reblur)); CUDA_CHECK(cudaFree(d_rl_ratio)); CUDA_CHECK(cudaFree(d_rl_correction));
    CUDA_CHECK(cudaFree(d_lr_frames)); CUDA_CHECK(cudaFree(d_shifts));
    CUDA_CHECK(cudaFree(d_bicubic)); CUDA_CHECK(cudaFree(d_hr_sum)); CUDA_CHECK(cudaFree(d_hr_weight)); CUDA_CHECK(cudaFree(d_shift_add));
    CUDA_CHECK(cudaFree(d_sr_estimate)); CUDA_CHECK(cudaFree(d_lr_predicted)); CUDA_CHECK(cudaFree(d_residual));

    // ===========================================================================
    // RESULT
    // ===========================================================================
    const bool verify_pass = verify_naive && verify_wiener && verify_rl && verify_bicubic && verify_shift_add && verify_ibp;
    const bool gates_pass = gate_wiener && gate_naive_fail && gate_rl && gate_mismatch && gate_sr_resolution && gate_sr_consistency;
    const bool overall = verify_pass && gates_pass && artifact_ok;
    if (overall) {
        std::printf("RESULT: PASS (VERIFY x6 + all 6 gates passed: wiener_recovery, naive_inverse_failure, "
                   "rl_recovery, psf_mismatch, sr_resolution, sr_consistency)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
