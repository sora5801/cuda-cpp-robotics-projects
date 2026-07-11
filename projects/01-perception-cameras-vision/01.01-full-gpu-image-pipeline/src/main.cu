// ===========================================================================
// main.cu — entry point for project 01.01
//           Full GPU image pipeline: debayer -> undistort -> rectify ->
//           resize -> normalize, run STAGED and FUSED, both GPU-vs-CPU
//           verified and checked against six independent physical gates
//
// What this program does, start to finish
// -----------------------------------------
//   1. Print the banner + GPU info; load the committed synthetic Bayer
//      scene + ground truth from data/sample/ (bayer_input.pgm,
//      true_rgb.ppm, smooth_mask.pgm — see ../scripts/make_synthetic.py).
//   2. STAGED GPU PIPELINE: debayer -> build remap LUT (once) -> remap
//      (undistort+rectify) -> resize (area-average x2) -> normalize
//      (3-kernel deterministic reduction) — five kernels, each writing a
//      full intermediate image to global memory.
//   3. FUSED GPU PIPELINE: the SAME debayer output and the SAME LUT feed
//      ONE kernel that does undistort+rectify+resize together, then the
//      same normalize kernels — kernels.cu's header derives why this
//      moves fewer bytes; this file measures it and prints both numbers.
//   4. VERIFY STAGE (the CLAUDE.md paragraph 5 GPU-vs-CPU gate): every
//      kernel above is compared, element-wise, against reference_cpu.cpp's
//      INDEPENDENT twin (max-abs-diff, documented per-stage tolerance).
//   5. SIX PHYSICAL GATES, each checking something the twin comparison
//      CANNOT (see kernels.cuh/reference_cpu.cpp's twin-independence
//      ruling — a shared bug would pass VERIFY but fail here):
//        roundtrip                  — forward-distort then independently
//                                      undistort a point grid; must return
//                                      to the start (camera-model self-
//                                      consistency, bypassing kernels.cuh).
//        straightness_rectified     — checkerboard column boundary must
//                                      measure STRAIGHT after rectify.
//        distortion_negative_control— the SAME boundary in the RAW
//                                      (uncorrected) image must measure
//                                      CURVED — proof the distortion being
//                                      corrected is real, not a no-op.
//        color_fidelity             — rectified output vs. true_rgb.ppm's
//                                      ground truth, in smooth regions.
//        resize_conservation        — area-average resize must preserve
//                                      the image's per-channel mean.
//        normalize                  — final tensor must measure mean~0,
//                                      std~1 to documented precision.
//        fused_vs_staged            — the two pipelines' final tensors
//                                      must agree within a tolerance that
//                                      accounts for fusion's single-vs-
//                                      double rounding (kernels.cu's note).
//   6. ARTIFACTS: demo/out/{bayer_input.pgm, debayered.ppm, rectified.ppm,
//      resized.ppm, fused_resized.ppm, normalized_vis.ppm,
//      gates_metrics.csv} — every stage's image, visually inspectable.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY:", the seven "GATE <name>:" verdict lines, "ARTIFACT:",
// and "RESULT:" — every one of these is a PASS/FAIL verdict with NO
// embedded numbers, so it is identical on every GPU architecture. The
// MEASURED numbers behind every verdict (pixel errors, means, timings,
// byte counts) are deliberately printed on separate "[info]"/"[time]"
// lines instead: this project's floating-point pipeline can differ by a
// few ULP across sm_75/sm_86/sm_89 (different FMA-contraction choices —
// THEORY.md "Numerical considerations"), so embedding a measured float in
// a line that demo/run_demo.* diffs verbatim would make the demo non-
// portable across GPUs. "[info]"/"[time]" lines are NOT diffed. Change a
// stable line => update demo/expected_output.txt in the same change.
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
// Gate tolerances — every number below is either a physical/numerical
// argument (documented inline) or a floor/ceiling calibrated from an
// ACTUAL measured run on this project's committed sample (CLAUDE.md
// paragraph 8: "never fabricate" — numbers here are recorded, with the
// measured values, in THEORY.md "How we verify correctness" and README
// "Expected output"), with margin so the gate stays robust to legitimate
// cross-GPU float differences (see the output-contract note above).
//
//   Measured on the reference machine (RTX 2080 SUPER, sm_75), Release:
//     verify(remap/resize/fused)     max|gpu-cpu| = 1.0000        (uint8 units)
//     verify(normalize apply)        max|gpu-cpu| = 0.0153 staged / 0.0150 fused
//     roundtrip                      max pixel error = 0.00000 px
//     straightness_rectified         boundary spread = 0.7395 px
//     distortion_negative_control    boundary spread = 1.3227 px
//     color_fidelity                 smooth-region mean|err| = 0.1463 (edge = 7.8932, ungated)
//     resize_conservation            max_c |mean(pre)-mean(post)| = 0.2059
//     normalize                      |mean| ~ 4.5e-8, |std-1| ~ 4e-6
//     fused_vs_staged                max|fused-staged| = 0.0187
// ===========================================================================

// -- GPU-vs-CPU VERIFY tolerances (per stage; all "how far can two
// independent IEEE-754 float pipelines legitimately drift" bounds, not
// physical bounds) -----------------------------------------------------
static constexpr double kTolUint8 = 1.5;         // debayer/remap/resize/fused outputs, 0..255 scale
static constexpr double kTolLutPx = 2e-3;         // remap LUT (u,v), pixels
static constexpr double kTolNormStat = 5e-3;      // normalize mean/std, 0..255 scale
// normalize-apply tolerance is DERIVED, not guessed: the remap/resize/fused
// stages already tolerate a +-1 uint8 rounding difference between GPU
// (nvcc contracts the bilinear interpolation's "a + (b-a)*t" into one FMA
// on device by default) and CPU (cl.exe does NOT contract multiply-add by
// default without /fp:fast) — a real, expected ~1-ULP-class divergence,
// not a bug (see THEORY.md "Numerical considerations"). Dividing that same
// +-1 unit by the normalize stage's std (empirically ~65 on this project's
// committed scene) propagates to roughly 1/65 =~ 0.015 in normalized-tensor
// units — MEASURED on the reference machine (RTX 2080 SUPER, sm_75) at
// 0.0153 (staged) / 0.0150 (fused); the floor below keeps a comfortable
// ~3x margin above that measured value.
static constexpr double kTolNormApply = 0.05;     // normalized tensor, zero-mean/unit-std scale

// -- Physical gate tolerances, each a floor/ceiling with margin over the
// measured values in the block above (never AT the measured value — see
// 01.02's identical calibration discipline for the precedent) -----------
static constexpr double kRoundtripTolPx = 0.05;          // floor >> measured 0.00000 px (pure numerical-iteration error)
static constexpr double kStraightRectifiedTolPx = 1.2;   // ceiling, ~1.6x measured 0.7395 px
static constexpr double kStraightRawMinPx = 1.0;         // floor, ~25% below measured 1.3227 px (negative control)
static constexpr double kColorFidelityTolMean = 2.0;     // ceiling, ~14x measured 0.1463 (0..255 scale)
static constexpr double kResizeConservationTolMean = 0.75;// ceiling, ~3.6x measured 0.2059 (0..255 scale)
static constexpr double kNormalizeMeanTol = 5e-3;        // ceiling, >> measured ~4.5e-8 (float32 reduction precision)
static constexpr double kNormalizeStdTol = 2e-2;         // ceiling, >> measured ~4e-6
static constexpr double kFusedVsStagedTol = 0.08;        // ceiling, ~4.3x measured 0.0187 (normalized-tensor units)

// -- Checkerboard geometry — MUST MATCH ../scripts/make_synthetic.py's
// CB_X0/CB_Y0/CB_SQUARE/CB_N (the scene's own layout spec). Boundary is
// the line between square column 0 and column 1.
static constexpr int kCbX0 = 32, kCbY0 = 32, kCbSquare = 24, kCbN = 8;
static constexpr int kCbBoundaryX = kCbX0 + kCbSquare * 1;   // = 56

// ===========================================================================
// Minimal, STRICT PGM (P5) / PPM (P6) readers — same discipline as 01.02's
// read_pgm: only ever reads files this project's own generator wrote.
// ===========================================================================
static bool read_pnm(const std::string& path, const char* want_magic, int channels,
                     int& W, int& H, std::vector<unsigned char>& data)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    std::string magic;
    in >> magic;
    if (magic != want_magic) return false;

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

    data.resize(static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(channels));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return in.gcount() == static_cast<std::streamsize>(data.size());
}
static bool read_pgm(const std::string& path, int& W, int& H, std::vector<unsigned char>& data)
{
    return read_pnm(path, "P5", 1, W, H, data);
}
static bool read_ppm(const std::string& path, int& W, int& H, std::vector<unsigned char>& data)
{
    return read_pnm(path, "P6", 3, W, H, data);
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
// Sample loading — the three committed files (see ../scripts/make_synthetic.py's
// file header for what each one is), dimension-checked against kFullW/kFullH
// AND each other: a strict loader, per repo convention — any mismatch aborts
// rather than silently truncating.
// ===========================================================================
struct Sample {
    std::vector<unsigned char> bayer;      // kFullW*kFullH, RGGB mosaic
    std::vector<unsigned char> true_rgb;   // kFullW*kFullH*3, ground truth ideal image
    std::vector<unsigned char> mask;       // kFullW*kFullH, 255 = score for color fidelity
    bool loaded = false;
};

static Sample load_sample(const std::string& cli_dir, const char* argv0)
{
    Sample s;
    const std::string bayer_path = find_data_file(cli_dir, argv0, "bayer_input.pgm");
    const std::string true_path  = find_data_file(cli_dir, argv0, "true_rgb.ppm");
    const std::string mask_path  = find_data_file(cli_dir, argv0, "smooth_mask.pgm");
    if (bayer_path.empty() || true_path.empty() || mask_path.empty()) {
        std::fprintf(stderr, "sample: one or more of bayer_input.pgm/true_rgb.ppm/smooth_mask.pgm not found "
                             "(run scripts/make_synthetic.py?)\n");
        return s;
    }
    int wb, hb, wt, ht, wm, hm;
    if (!read_pgm(bayer_path, wb, hb, s.bayer)) { std::fprintf(stderr, "sample: failed to read bayer_input.pgm\n"); return Sample{}; }
    if (!read_ppm(true_path, wt, ht, s.true_rgb)) { std::fprintf(stderr, "sample: failed to read true_rgb.ppm\n"); return Sample{}; }
    if (!read_pgm(mask_path, wm, hm, s.mask)) { std::fprintf(stderr, "sample: failed to read smooth_mask.pgm\n"); return Sample{}; }
    if (wb != kFullW || hb != kFullH || wt != kFullW || ht != kFullH || wm != kFullW || hm != kFullH) {
        std::fprintf(stderr, "sample: dimension mismatch — expected %dx%d everywhere\n", kFullW, kFullH);
        return Sample{};
    }
    s.loaded = true;
    return s;
}

// ===========================================================================
// max_abs_diff — generic L-infinity comparison, used by every VERIFY
// checkpoint below (uint8 stages implicitly widen to double; the float
// normalize stages compare directly).
// ===========================================================================
template <typename T>
static double max_abs_diff(const std::vector<T>& a, const std::vector<T>& b)
{
    double m = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m) m = d;
    }
    return m;
}
static double max_abs_diff_lut(const std::vector<RemapSample>& a, const std::vector<RemapSample>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, static_cast<double>(std::fabs(a[i].u - b[i].u)));
        m = std::max(m, static_cast<double>(std::fabs(a[i].v - b[i].v)));
    }
    return m;
}

// ===========================================================================
// GATE 1: roundtrip — camera-model self-consistency, BYPASSING kernels.cuh
// entirely (per the twin-independence ruling in kernels.cuh's file header:
// this project must carry at least one check that does not route through
// compute_source_pixel()/distort_forward()). Every formula below is
// hand-retyped in DOUBLE precision, independently of kernels.cu,
// reference_cpu.cpp, AND ../scripts/make_synthetic.py (three independent
// languages/files now agree on this camera model — see that script's
// header). A regular grid of ideal (rectified) pixels is forward-mapped to
// raw pixel coordinates (same math as compute_source_pixel, RETYPED), then
// independently undistorted via fixed-point iteration and un-rotated back
// to an ideal pixel — the round trip must return (very close to) the start.
// ===========================================================================
static double gate_roundtrip()
{
    const int NX = 9, NY = 7;
    const double margin = 24.0;
    const double c = static_cast<double>(kRectCos), s = static_cast<double>(kRectSin);
    double max_err = 0.0;

    for (int iy = 0; iy < NY; ++iy) {
        for (int ix = 0; ix < NX; ++ix) {
            const double xo = margin + (kFullW - 1 - 2.0 * margin) * (static_cast<double>(ix) / (NX - 1));
            const double yo = margin + (kFullH - 1 - 2.0 * margin) * (static_cast<double>(iy) / (NY - 1));

            // ---- FORWARD: ideal (rectified) pixel -> raw pixel ----------
            const double xr = (xo - kCx) / kFx;
            const double yr = (yo - kCy) / kFy;
            const double rx = c * xr - s;             // R_rect_raw^T row0 . (xr,yr,1)
            const double ry = yr;
            const double rz = s * xr + c;              // R_rect_raw^T row2 . (xr,yr,1)
            const double xn = rx / rz, yn = ry / rz;
            const double r2f = xn * xn + yn * yn;
            const double radial = 1.0 + kK1 * r2f + kK2 * r2f * r2f;
            const double xd = xn * radial + 2.0 * kP1 * xn * yn + kP2 * (r2f + 2.0 * xn * xn);
            const double yd = yn * radial + kP1 * (r2f + 2.0 * yn * yn) + 2.0 * kP2 * xn * yn;
            const double u_raw = kFx * xd + kCx;
            const double v_raw = kFy * yd + kCy;

            // ---- INVERSE: raw pixel -> ideal pixel, fixed-point undistort
            const double xdn = (u_raw - kCx) / kFx, ydn = (v_raw - kCy) / kFy;
            double xu = xdn, yu = ydn;
            for (int it = 0; it < 20; ++it) {
                const double rr2 = xu * xu + yu * yu;
                const double icdist = 1.0 / (1.0 + kK1 * rr2 + kK2 * rr2 * rr2);
                const double dx = 2.0 * kP1 * xu * yu + kP2 * (rr2 + 2.0 * xu * xu);
                const double dy = kP1 * (rr2 + 2.0 * yu * yu) + 2.0 * kP2 * xu * yu;
                xu = (xdn - dx) * icdist;
                yu = (ydn - dy) * icdist;
            }
            // un-rotate raw-frame ray -> rectified-frame ray: v_rect = R_rect_raw * v_raw
            const double ox = c * xu + s;
            const double oy = yu;
            const double oz = -s * xu + c;
            const double xi = ox / oz, yi = oy / oz;
            const double xo2 = kFx * xi + kCx;
            const double yo2 = kFy * yi + kCy;

            const double err = std::hypot(xo2 - xo, yo2 - yo);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// ===========================================================================
// GATES 2 & 3: straightness — find where a checkerboard column boundary
// crosses the 50%-gray threshold along a set of scanlines, in a HOST-side,
// from-scratch edge detector (bypasses every kernel and every reference_cpu
// function: this gate exercises the ACTUAL pixel content of the images,
// which is what a geometric-correctness gate must do). Applied to the
// RECTIFIED stage-3 output (should read STRAIGHT: low spread across rows)
// and, as a negative control, to the RAW debayered stage-1 output (should
// read CURVED: distortion has not been removed there — proof the
// distortion being corrected is real, not vacuous).
// ===========================================================================
static bool find_crossing_x(const std::vector<unsigned char>& rgb, int W, int H,
                            int y, int x_lo, int x_hi, double& out_x)
{
    if (y < 0 || y >= H) return false;
    if (x_lo < 0) x_lo = 0;
    if (x_hi > W - 2) x_hi = W - 2;
    for (int x = x_lo; x <= x_hi; ++x) {
        const int a = rgb[(static_cast<size_t>(y) * W + x) * 3 + 0];
        const int b = rgb[(static_cast<size_t>(y) * W + x + 1) * 3 + 0];
        const bool cross = (a > 127) != (b > 127);
        if (cross) {
            const double t = (127.5 - static_cast<double>(a)) / (static_cast<double>(b) - static_cast<double>(a));
            out_x = static_cast<double>(x) + t;
            return true;
        }
    }
    return false;
}

static double straightness_residual(const std::vector<unsigned char>& rgb, int W, int H, int* n_found_out)
{
    std::vector<double> xs;
    const int y_lo = kCbY0 + 12;
    const int y_hi = kCbY0 + kCbSquare * kCbN - 12;
    for (int y = y_lo; y <= y_hi; y += 8) {
        double x;
        if (find_crossing_x(rgb, W, H, y, kCbBoundaryX - 22, kCbBoundaryX + 12, x))
            xs.push_back(x);
    }
    if (n_found_out) *n_found_out = static_cast<int>(xs.size());
    if (xs.empty()) return -1.0;   // sentinel: detector found nothing (reported, gate fails honestly)
    double mean = 0.0;
    for (double v : xs) mean += v;
    mean /= static_cast<double>(xs.size());
    double maxdev = 0.0;
    for (double v : xs) maxdev = std::max(maxdev, std::fabs(v - mean));
    return maxdev;
}

// ===========================================================================
// GATE 4: color_fidelity — mean absolute error of the STAGED rectified
// output against true_rgb.ppm's ground truth, split into smooth_mask==255
// (scored, gated) and smooth_mask==0 (reported, NOT gated — edges are
// EXPECTED to show larger error from bilinear blending across a real
// step edge; hiding that number would be dishonest, gating on it would
// penalize the pipeline for physics it cannot avoid).
// ===========================================================================
struct ColorFidelity { double mean_err_smooth; double mean_err_edge; long long n_smooth; long long n_edge; };

static ColorFidelity color_fidelity(const std::vector<unsigned char>& pipeline_rgb,
                                    const std::vector<unsigned char>& true_rgb,
                                    const std::vector<unsigned char>& mask, int W, int H)
{
    double sum_smooth = 0.0, sum_edge = 0.0;
    long long n_smooth = 0, n_edge = 0;
    const int n = W * H;
    for (int i = 0; i < n; ++i) {
        const double err = (std::fabs(static_cast<double>(pipeline_rgb[i * 3 + 0]) - true_rgb[i * 3 + 0])
                           + std::fabs(static_cast<double>(pipeline_rgb[i * 3 + 1]) - true_rgb[i * 3 + 1])
                           + std::fabs(static_cast<double>(pipeline_rgb[i * 3 + 2]) - true_rgb[i * 3 + 2])) / 3.0;
        if (mask[i] == 255) { sum_smooth += err; ++n_smooth; }
        else                { sum_edge += err;   ++n_edge; }
    }
    ColorFidelity r;
    r.mean_err_smooth = n_smooth > 0 ? sum_smooth / static_cast<double>(n_smooth) : 0.0;
    r.mean_err_edge = n_edge > 0 ? sum_edge / static_cast<double>(n_edge) : 0.0;
    r.n_smooth = n_smooth; r.n_edge = n_edge;
    return r;
}

// ===========================================================================
// GATE 5: resize_conservation — an exact area-average filter must preserve
// the image's per-channel MEAN (every input pixel contributes to exactly
// one output pixel with equal weight — THEORY.md derives this as a
// conservation law, analogous to "the average height of water does not
// change when you pour it into differently-shaped identical-volume
// buckets"). Compares the pre-resize (rectified, full-res) mean to the
// post-resize (resized) mean, per channel, MAX over channels.
// ===========================================================================
static void channel_means(const std::vector<unsigned char>& rgb, int n_pixels, double mean_out[3])
{
    double sum[3] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < n_pixels; ++i)
        for (int c = 0; c < 3; ++c) sum[c] += rgb[static_cast<size_t>(i) * 3 + c];
    for (int c = 0; c < 3; ++c) mean_out[c] = sum[c] / static_cast<double>(n_pixels);
}

// ===========================================================================
// GATE 6: normalize — the final tensor's OWN mean/std, recomputed
// INDEPENDENTLY here (never reusing the GPU's d_mean3/d_std3, which by
// construction the apply kernel used to BUILD the tensor — reusing them
// would make the check circular). A fresh double-accumulation pass over
// the downloaded tensor is the honest way to ask "does this tensor
// actually have the properties normalization promises?".
// ===========================================================================
static void tensor_stats(const std::vector<float>& out, int n_pixels, double mean_out[3], double std_out[3])
{
    double sum[3] = { 0.0, 0.0, 0.0 }, sumsq[3] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < n_pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double v = out[static_cast<size_t>(i) * 3 + c];
            sum[c] += v;
            sumsq[c] += v * v;
        }
    }
    const double n = static_cast<double>(n_pixels);
    for (int c = 0; c < 3; ++c) {
        const double mean = sum[c] / n;
        double var = sumsq[c] / n - mean * mean;
        mean_out[c] = mean;
        std_out[c] = std::sqrt(var > 0.0 ? var : 0.0);
    }
}

// ===========================================================================
// Artifact helpers: visualize the normalized (zero-mean/unit-std float)
// tensor as a viewable 8-bit PPM. This is a DISPLAY-ONLY transform (the
// real normalized output stays float; nothing downstream of normalize
// ever sees this byte image) — map roughly [-3 sigma, +3 sigma] to
// [0, 255] centered at 128, clamped.
// ===========================================================================
static std::vector<unsigned char> normalized_to_vis(const std::vector<float>& out, int n_pixels)
{
    std::vector<unsigned char> vis(static_cast<size_t>(n_pixels) * 3);
    for (int i = 0; i < n_pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = 128.0f + out[static_cast<size_t>(i) * 3 + c] * 42.5f;   // +-3 sigma -> [0.5, 255.5]
            v = std::min(std::max(v, 0.0f), 255.0f);
            vis[static_cast<size_t>(i) * 3 + c] = static_cast<unsigned char>(v + 0.5f);
        }
    }
    return vis;
}

// ===========================================================================
// gates_metrics.csv writer — the per-run, per-metric record (README/THEORY
// describe the gates in prose; this CSV is the machine-readable teaching
// artifact, one row per measured quantity).
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
    for (const auto& r : rows)
        out << r.gate << "," << r.metric << "," << r.value << "," << r.tol << "," << r.pass << "\n";
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

    std::printf("[demo] full GPU image pipeline: debayer -> undistort+rectify -> resize -> normalize, staged vs fused (project 01.01)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d RGGB Bayer input, K=(fx=%.0f,fy=%.0f,cx=%.1f,cy=%.1f), "
               "k1=%.2f k2=%.2f p1=%.4f p2=%.4f, rectify=%.1fdeg about Y, resize x%d area-average\n",
               kFullW, kFullH, kFx, kFy, kCx, kCy, kK1, kK2, kP1, kP2, kRectifyAngleDeg, kResizeFactor);

    // ---- data --------------------------------------------------------------
    Sample sample = load_sample(data_dir, argv[0]);
    if (!sample.loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: %dx%d synthetic RGGB Bayer scene (checkerboard + gradient + 3 disks), "
               "ground truth + smooth_mask [synthetic, seed 42]\n", kFullW, kFullH);

    const int full_n = kFullW * kFullH;
    const int resized_n = kResizedW * kResizedH;
    const int norm_blocks = (resized_n + kNormBlockSize - 1) / kNormBlockSize;

    // ======================= device buffers =====================================
    unsigned char* d_bayer = nullptr;
    unsigned char* d_debayered = nullptr;
    RemapSample* d_lut = nullptr;
    unsigned char* d_rectified = nullptr;          // STAGED stage 2+3 output (full-res)
    unsigned char* d_resized_staged = nullptr;
    unsigned char* d_resized_fused = nullptr;
    double *d_bsum_staged = nullptr, *d_bsumsq_staged = nullptr;
    double *d_bsum_fused = nullptr, *d_bsumsq_fused = nullptr;
    float *d_mean_staged = nullptr, *d_std_staged = nullptr;
    float *d_mean_fused = nullptr, *d_std_fused = nullptr;
    float* d_norm_staged = nullptr;
    float* d_norm_fused = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bayer, full_n));
    CUDA_CHECK(cudaMalloc(&d_debayered, static_cast<size_t>(full_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_lut, static_cast<size_t>(full_n) * sizeof(RemapSample)));
    CUDA_CHECK(cudaMalloc(&d_rectified, static_cast<size_t>(full_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_resized_staged, static_cast<size_t>(resized_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_resized_fused, static_cast<size_t>(resized_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_bsum_staged, static_cast<size_t>(norm_blocks) * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bsumsq_staged, static_cast<size_t>(norm_blocks) * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bsum_fused, static_cast<size_t>(norm_blocks) * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bsumsq_fused, static_cast<size_t>(norm_blocks) * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mean_staged, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_std_staged, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_fused, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_std_fused, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm_staged, static_cast<size_t>(resized_n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm_fused, static_cast<size_t>(resized_n) * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_bayer, sample.bayer.data(), full_n, cudaMemcpyHostToDevice));

    // ======================= shared stage: debayer + LUT ========================
    // Both pipelines consume the SAME debayered image and the SAME LUT — the
    // "debayer stays separate" decision (kernels.cuh's file header).
    GpuTimer gt_shared; gt_shared.begin();
    launch_debayer_rggb(d_bayer, d_debayered, kFullW, kFullH);
    launch_build_remap_lut(d_lut, kFullW, kFullH);
    const float shared_ms = gt_shared.end_ms();

    // ======================= STAGED pipeline =====================================
    GpuTimer gt_remap; gt_remap.begin();
    launch_remap_bilinear(d_debayered, d_lut, d_rectified, kFullW, kFullH);
    const float remap_ms = gt_remap.end_ms();

    GpuTimer gt_resize; gt_resize.begin();
    launch_resize_area2x(d_rectified, d_resized_staged, kFullW, kFullH);
    const float resize_ms = gt_resize.end_ms();

    GpuTimer gt_norm_staged; gt_norm_staged.begin();
    launch_normalize_block_stats(d_resized_staged, kResizedW, kResizedH, d_bsum_staged, d_bsumsq_staged, norm_blocks);
    launch_normalize_finalize(d_bsum_staged, d_bsumsq_staged, norm_blocks, resized_n, d_mean_staged, d_std_staged);
    launch_normalize_apply(d_resized_staged, d_norm_staged, kResizedW, kResizedH, d_mean_staged, d_std_staged);
    const float norm_staged_ms = gt_norm_staged.end_ms();

    const float staged_total_ms = shared_ms + remap_ms + resize_ms + norm_staged_ms;

    // ======================= FUSED pipeline ======================================
    GpuTimer gt_fused; gt_fused.begin();
    launch_fused_undistort_rectify_resize(d_debayered, d_lut, d_resized_fused, kFullW, kFullH);
    const float fused_ms = gt_fused.end_ms();

    GpuTimer gt_norm_fused; gt_norm_fused.begin();
    launch_normalize_block_stats(d_resized_fused, kResizedW, kResizedH, d_bsum_fused, d_bsumsq_fused, norm_blocks);
    launch_normalize_finalize(d_bsum_fused, d_bsumsq_fused, norm_blocks, resized_n, d_mean_fused, d_std_fused);
    launch_normalize_apply(d_resized_fused, d_norm_fused, kResizedW, kResizedH, d_mean_fused, d_std_fused);
    const float norm_fused_ms = gt_norm_fused.end_ms();

    const float fused_total_ms = shared_ms + fused_ms + norm_fused_ms;

    std::printf("[time] staged: shared(debayer+LUT) %.3f ms | remap %.3f ms | resize %.3f ms | normalize %.3f ms | TOTAL %.3f ms\n",
               static_cast<double>(shared_ms), static_cast<double>(remap_ms), static_cast<double>(resize_ms),
               static_cast<double>(norm_staged_ms), static_cast<double>(staged_total_ms));
    std::printf("[time] fused:  shared(debayer+LUT) %.3f ms | fused %.3f ms | normalize %.3f ms | TOTAL %.3f ms\n",
               static_cast<double>(shared_ms), static_cast<double>(fused_ms),
               static_cast<double>(norm_fused_ms), static_cast<double>(fused_total_ms));

    // Analytic memory-traffic accounting (kernels.cu's fused_kernel header
    // derives this formula in full) — an IDEALIZED no-cache-reuse byte
    // count, NOT a profiler measurement (honestly labeled as such).
    const double WHd = static_cast<double>(kFullW) * static_cast<double>(kFullH);
    const double staged_bytes = 18.75 * WHd;
    const double fused_bytes = 12.75 * WHd;
    const double savings_pct = 100.0 * (staged_bytes - fused_bytes) / staged_bytes;
    std::printf("[info] memory traffic, remap+resize only (derived, idealized no-cache-reuse model): "
               "staged = %.0f bytes | fused = %.0f bytes | savings = %.1f%%\n",
               staged_bytes, fused_bytes, savings_pct);

    // ======================= download everything needed for VERIFY/gates =======
    std::vector<unsigned char> debayered_gpu(static_cast<size_t>(full_n) * 3);
    std::vector<RemapSample> lut_gpu(full_n);
    std::vector<unsigned char> rectified_gpu(static_cast<size_t>(full_n) * 3);
    std::vector<unsigned char> resized_staged_gpu(static_cast<size_t>(resized_n) * 3);
    std::vector<unsigned char> resized_fused_gpu(static_cast<size_t>(resized_n) * 3);
    std::vector<float> mean_staged_gpu(3), std_staged_gpu(3), mean_fused_gpu(3), std_fused_gpu(3);
    std::vector<float> norm_staged_gpu(static_cast<size_t>(resized_n) * 3);
    std::vector<float> norm_fused_gpu(static_cast<size_t>(resized_n) * 3);

    CUDA_CHECK(cudaMemcpy(debayered_gpu.data(), d_debayered, debayered_gpu.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lut_gpu.data(), d_lut, static_cast<size_t>(full_n) * sizeof(RemapSample), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rectified_gpu.data(), d_rectified, rectified_gpu.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(resized_staged_gpu.data(), d_resized_staged, resized_staged_gpu.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(resized_fused_gpu.data(), d_resized_fused, resized_fused_gpu.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mean_staged_gpu.data(), d_mean_staged, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(std_staged_gpu.data(), d_std_staged, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mean_fused_gpu.data(), d_mean_fused, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(std_fused_gpu.data(), d_std_fused, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(norm_staged_gpu.data(), d_norm_staged, norm_staged_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(norm_fused_gpu.data(), d_norm_fused, norm_fused_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ======================= VERIFY: GPU vs CPU, every stage ====================
    bool verify_pass = true;
    CpuTimer cpu_timer; cpu_timer.begin();
    {
        std::vector<unsigned char> debayered_cpu(static_cast<size_t>(full_n) * 3);
        debayer_rggb_cpu(sample.bayer.data(), debayered_cpu.data(), kFullW, kFullH);
        const double d_debayer = max_abs_diff(debayered_gpu, debayered_cpu);
        std::printf("[info] verify(debayer): max|gpu-cpu| = %.4f (tol %.2f)\n", d_debayer, kTolUint8);
        if (d_debayer > kTolUint8) verify_pass = false;

        std::vector<RemapSample> lut_cpu(full_n);
        build_remap_lut_cpu(lut_cpu.data(), kFullW, kFullH);
        const double d_lut = max_abs_diff_lut(lut_gpu, lut_cpu);
        std::printf("[info] verify(remap LUT): max|gpu-cpu| = %.6f px (tol %.4f)\n", d_lut, kTolLutPx);
        if (d_lut > kTolLutPx) verify_pass = false;

        std::vector<unsigned char> rectified_cpu(static_cast<size_t>(full_n) * 3);
        remap_bilinear_cpu(debayered_cpu.data(), lut_cpu.data(), rectified_cpu.data(), kFullW, kFullH);
        const double d_remap = max_abs_diff(rectified_gpu, rectified_cpu);
        std::printf("[info] verify(remap/undistort+rectify): max|gpu-cpu| = %.4f (tol %.2f)\n", d_remap, kTolUint8);
        if (d_remap > kTolUint8) verify_pass = false;

        std::vector<unsigned char> resized_cpu(static_cast<size_t>(resized_n) * 3);
        resize_area2x_cpu(rectified_cpu.data(), resized_cpu.data(), kFullW, kFullH);
        const double d_resize = max_abs_diff(resized_staged_gpu, resized_cpu);
        std::printf("[info] verify(resize): max|gpu-cpu| = %.4f (tol %.2f)\n", d_resize, kTolUint8);
        if (d_resize > kTolUint8) verify_pass = false;

        std::vector<unsigned char> fused_cpu(static_cast<size_t>(resized_n) * 3);
        fused_undistort_rectify_resize_cpu(debayered_cpu.data(), lut_cpu.data(), fused_cpu.data(), kFullW, kFullH);
        const double d_fused = max_abs_diff(resized_fused_gpu, fused_cpu);
        std::printf("[info] verify(fused): max|gpu-cpu| = %.4f (tol %.2f)\n", d_fused, kTolUint8);
        if (d_fused > kTolUint8) verify_pass = false;

        double mean3_staged[3], std3_staged[3];
        normalize_stats_cpu(resized_cpu.data(), kResizedW, kResizedH, mean3_staged, std3_staged);
        const double d_mean_s = std::max({ std::fabs(mean3_staged[0] - mean_staged_gpu[0]),
                                           std::fabs(mean3_staged[1] - mean_staged_gpu[1]),
                                           std::fabs(mean3_staged[2] - mean_staged_gpu[2]) });
        const double d_std_s = std::max({ std::fabs(std3_staged[0] - std_staged_gpu[0]),
                                          std::fabs(std3_staged[1] - std_staged_gpu[1]),
                                          std::fabs(std3_staged[2] - std_staged_gpu[2]) });
        std::printf("[info] verify(normalize stats, staged): max|gpu-cpu| mean=%.5f std=%.5f (tol %.4f)\n", d_mean_s, d_std_s, kTolNormStat);
        if (d_mean_s > kTolNormStat || d_std_s > kTolNormStat) verify_pass = false;

        std::vector<float> norm_staged_cpu(static_cast<size_t>(resized_n) * 3);
        normalize_apply_cpu(resized_cpu.data(), norm_staged_cpu.data(), kResizedW, kResizedH, mean3_staged, std3_staged);
        const double d_apply_s = max_abs_diff(norm_staged_gpu, norm_staged_cpu);
        std::printf("[info] verify(normalize apply, staged): max|gpu-cpu| = %.5f (tol %.4f)\n", d_apply_s, kTolNormApply);
        if (d_apply_s > kTolNormApply) verify_pass = false;

        double mean3_fused[3], std3_fused[3];
        normalize_stats_cpu(fused_cpu.data(), kResizedW, kResizedH, mean3_fused, std3_fused);
        const double d_mean_f = std::max({ std::fabs(mean3_fused[0] - mean_fused_gpu[0]),
                                           std::fabs(mean3_fused[1] - mean_fused_gpu[1]),
                                           std::fabs(mean3_fused[2] - mean_fused_gpu[2]) });
        const double d_std_f = std::max({ std::fabs(std3_fused[0] - std_fused_gpu[0]),
                                          std::fabs(std3_fused[1] - std_fused_gpu[1]),
                                          std::fabs(std3_fused[2] - std_fused_gpu[2]) });
        std::printf("[info] verify(normalize stats, fused): max|gpu-cpu| mean=%.5f std=%.5f (tol %.4f)\n", d_mean_f, d_std_f, kTolNormStat);
        if (d_mean_f > kTolNormStat || d_std_f > kTolNormStat) verify_pass = false;

        std::vector<float> norm_fused_cpu(static_cast<size_t>(resized_n) * 3);
        normalize_apply_cpu(fused_cpu.data(), norm_fused_cpu.data(), kResizedW, kResizedH, mean3_fused, std3_fused);
        const double d_apply_f = max_abs_diff(norm_fused_gpu, norm_fused_cpu);
        std::printf("[info] verify(normalize apply, fused): max|gpu-cpu| = %.5f (tol %.4f)\n", d_apply_f, kTolNormApply);
        if (d_apply_f > kTolNormApply) verify_pass = false;
    }
    const double cpu_ms = cpu_timer.end_ms();
    std::printf("[time] full CPU oracle (all stages, both pipelines): %.1f ms\n", cpu_ms);
    std::printf("VERIFY: %s (GPU matches CPU reference within documented per-stage tolerance: "
               "debayer, remap LUT, remap, resize, fused, normalize stats x2, normalize apply x2)\n",
               verify_pass ? "PASS" : "FAIL");

    // ======================= GATES ================================================
    std::vector<CsvRow> csv;

    // -- Gate 1: roundtrip --------------------------------------------------
    const double roundtrip_err = gate_roundtrip();
    const bool gate1 = roundtrip_err <= kRoundtripTolPx;
    std::printf("GATE roundtrip: %s\n", gate1 ? "PASS" : "FAIL");
    std::printf("[info] roundtrip: max pixel error over 63-point grid = %.5f px (tol %.2f)\n", roundtrip_err, kRoundtripTolPx);
    csv.push_back({ "roundtrip", "max_error_px", fmt(roundtrip_err, 5), fmt(kRoundtripTolPx, 2), gate1 ? "PASS" : "FAIL" });

    // -- Gates 2 & 3: straightness -------------------------------------------
    int n_found_rect = 0, n_found_raw = 0;
    const double resid_rect = straightness_residual(rectified_gpu, kFullW, kFullH, &n_found_rect);
    const double resid_raw = straightness_residual(debayered_gpu, kFullW, kFullH, &n_found_raw);
    const bool gate2 = (resid_rect >= 0.0) && (resid_rect <= kStraightRectifiedTolPx);
    const bool gate3 = (resid_raw >= 0.0) && (resid_raw >= kStraightRawMinPx);
    std::printf("GATE straightness_rectified: %s\n", gate2 ? "PASS" : "FAIL");
    std::printf("[info] straightness_rectified: boundary x-spread = %.4f px over %d rows (tol <= %.2f)\n",
               resid_rect, n_found_rect, kStraightRectifiedTolPx);
    std::printf("GATE distortion_negative_control: %s\n", gate3 ? "PASS" : "FAIL");
    std::printf("[info] distortion_negative_control: RAW (uncorrected) boundary x-spread = %.4f px over %d rows "
               "(must be >= %.2f -- proves the distortion being corrected is real)\n",
               resid_raw, n_found_raw, kStraightRawMinPx);
    csv.push_back({ "straightness_rectified", "boundary_spread_px", fmt(resid_rect, 4), fmt(kStraightRectifiedTolPx, 2), gate2 ? "PASS" : "FAIL" });
    csv.push_back({ "distortion_negative_control", "boundary_spread_px", fmt(resid_raw, 4), fmt(kStraightRawMinPx, 2), gate3 ? "PASS" : "FAIL" });

    // -- Gate 4: color fidelity ----------------------------------------------
    const ColorFidelity cf = color_fidelity(rectified_gpu, sample.true_rgb, sample.mask, kFullW, kFullH);
    const bool gate4 = cf.mean_err_smooth <= kColorFidelityTolMean;
    std::printf("GATE color_fidelity: %s\n", gate4 ? "PASS" : "FAIL");
    std::printf("[info] color_fidelity: smooth-region mean|err| = %.4f (tol %.2f, n=%lld) | edge-region mean|err| = %.4f "
               "(reported only, not gated, n=%lld)\n",
               cf.mean_err_smooth, kColorFidelityTolMean, cf.n_smooth, cf.mean_err_edge, cf.n_edge);
    csv.push_back({ "color_fidelity", "mean_abs_err_smooth", fmt(cf.mean_err_smooth, 4), fmt(kColorFidelityTolMean, 2), gate4 ? "PASS" : "FAIL" });
    csv.push_back({ "color_fidelity", "mean_abs_err_edge_reported_only", fmt(cf.mean_err_edge, 4), "n/a", "n/a" });

    // -- Gate 5: resize conservation -----------------------------------------
    double mean_pre[3], mean_post[3];
    channel_means(rectified_gpu, full_n, mean_pre);
    channel_means(resized_staged_gpu, resized_n, mean_post);
    const double resize_dev = std::max({ std::fabs(mean_pre[0] - mean_post[0]),
                                         std::fabs(mean_pre[1] - mean_post[1]),
                                         std::fabs(mean_pre[2] - mean_post[2]) });
    const bool gate5 = resize_dev <= kResizeConservationTolMean;
    std::printf("GATE resize_conservation: %s\n", gate5 ? "PASS" : "FAIL");
    std::printf("[info] resize_conservation: max_c |mean(pre)-mean(post)| = %.4f (tol %.2f) "
               "[pre R,G,B=%.2f,%.2f,%.2f | post R,G,B=%.2f,%.2f,%.2f]\n",
               resize_dev, kResizeConservationTolMean,
               mean_pre[0], mean_pre[1], mean_pre[2], mean_post[0], mean_post[1], mean_post[2]);
    csv.push_back({ "resize_conservation", "max_channel_mean_diff", fmt(resize_dev, 4), fmt(kResizeConservationTolMean, 2), gate5 ? "PASS" : "FAIL" });

    // -- Gate 6: normalize -----------------------------------------------------
    double tensor_mean[3], tensor_std[3];
    tensor_stats(norm_staged_gpu, resized_n, tensor_mean, tensor_std);
    const double max_abs_mean = std::max({ std::fabs(tensor_mean[0]), std::fabs(tensor_mean[1]), std::fabs(tensor_mean[2]) });
    const double max_abs_std_dev = std::max({ std::fabs(tensor_std[0] - 1.0), std::fabs(tensor_std[1] - 1.0), std::fabs(tensor_std[2] - 1.0) });
    const bool gate6 = (max_abs_mean <= kNormalizeMeanTol) && (max_abs_std_dev <= kNormalizeStdTol);
    std::printf("GATE normalize: %s\n", gate6 ? "PASS" : "FAIL");
    std::printf("[info] normalize: |mean| <= %.5f (tol %.4f) | |std-1| <= %.5f (tol %.3f) "
               "[mean R,G,B=%.2e,%.2e,%.2e | std R,G,B=%.5f,%.5f,%.5f]\n",
               max_abs_mean, kNormalizeMeanTol, max_abs_std_dev, kNormalizeStdTol,
               tensor_mean[0], tensor_mean[1], tensor_mean[2], tensor_std[0], tensor_std[1], tensor_std[2]);
    csv.push_back({ "normalize", "max_abs_mean", fmt(max_abs_mean, 6), fmt(kNormalizeMeanTol, 4), gate6 ? "PASS" : "FAIL" });
    csv.push_back({ "normalize", "max_abs_std_minus_1", fmt(max_abs_std_dev, 6), fmt(kNormalizeStdTol, 3), gate6 ? "PASS" : "FAIL" });

    // -- Gate 7: fused vs staged -------------------------------------------------
    const double fvs_diff = max_abs_diff(norm_fused_gpu, norm_staged_gpu);
    const bool gate7 = fvs_diff <= kFusedVsStagedTol;
    std::printf("GATE fused_vs_staged: %s\n", gate7 ? "PASS" : "FAIL");
    std::printf("[info] fused_vs_staged: max|fused-staged| = %.5f normalized-tensor units (tol %.2f; nonzero because "
               "fusion rounds once where staged rounds twice -- see kernels.cu's fused_kernel header)\n",
               fvs_diff, kFusedVsStagedTol);
    csv.push_back({ "fused_vs_staged", "max_abs_diff", fmt(fvs_diff, 5), fmt(kFusedVsStagedTol, 2), gate7 ? "PASS" : "FAIL" });

    // ======================= ARTIFACTS ============================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();
    artifact_ok = artifact_ok
        && write_pgm(out_dir + "/bayer_input.pgm", kFullW, kFullH, sample.bayer)
        && write_ppm(out_dir + "/debayered.ppm", kFullW, kFullH, debayered_gpu)
        && write_ppm(out_dir + "/rectified.ppm", kFullW, kFullH, rectified_gpu)
        && write_ppm(out_dir + "/resized.ppm", kResizedW, kResizedH, resized_staged_gpu)
        && write_ppm(out_dir + "/fused_resized.ppm", kResizedW, kResizedH, resized_fused_gpu)
        && write_ppm(out_dir + "/normalized_vis.ppm", kResizedW, kResizedH, normalized_to_vis(norm_staged_gpu, resized_n))
        && write_gates_csv(out_dir + "/gates_metrics.csv", csv);
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{bayer_input.pgm, debayered.ppm, rectified.ppm, resized.ppm, "
                   "fused_resized.ppm, normalized_vis.ppm, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");

    // ---- cleanup ------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_bayer));       CUDA_CHECK(cudaFree(d_debayered));
    CUDA_CHECK(cudaFree(d_lut));         CUDA_CHECK(cudaFree(d_rectified));
    CUDA_CHECK(cudaFree(d_resized_staged)); CUDA_CHECK(cudaFree(d_resized_fused));
    CUDA_CHECK(cudaFree(d_bsum_staged)); CUDA_CHECK(cudaFree(d_bsumsq_staged));
    CUDA_CHECK(cudaFree(d_bsum_fused));  CUDA_CHECK(cudaFree(d_bsumsq_fused));
    CUDA_CHECK(cudaFree(d_mean_staged)); CUDA_CHECK(cudaFree(d_std_staged));
    CUDA_CHECK(cudaFree(d_mean_fused));  CUDA_CHECK(cudaFree(d_std_fused));
    CUDA_CHECK(cudaFree(d_norm_staged)); CUDA_CHECK(cudaFree(d_norm_fused));

    // ---- verdict --------------------------------------------------------------
    const bool success = verify_pass && gate1 && gate2 && gate3 && gate4 && gate5 && gate6 && gate7 && artifact_ok;
    if (success) {
        std::printf("RESULT: PASS (VERIFY + all 7 gates passed: roundtrip, straightness_rectified, "
                   "distortion_negative_control, color_fidelity, resize_conservation, normalize, fused_vs_staged)\n");
    } else {
        std::printf("RESULT: FAIL (VERIFY or a gate above did not pass -- see GATE:/VERIFY:/[info] lines)\n");
    }
    return success ? 0 : 1;
}
