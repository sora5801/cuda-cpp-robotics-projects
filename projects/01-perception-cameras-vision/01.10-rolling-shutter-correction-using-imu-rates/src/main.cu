// ===========================================================================
// main.cu — entry point for project 01.10 (Rolling-shutter correction using
//           IMU rates): gyro-aided un-warping of a captured rolling-shutter
//           (RS) frame back to the global-shutter (GS) view a camera at the
//           frame's reference instant would have seen, verified end to end.
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed synthetic sample: rs_input.pgm (the captured RS
//      frame), ground_truth_gs.pgm (the analytically-known GS reference
//      image), and TWO 200 Hz gyro traces — gyro_clean.csv and
//      gyro_degraded.csv (the same true motion, but with a realistic gyro
//      bias + noise added — see ../scripts/make_synthetic.py).
//   2. GYRO INTEGRATION (host, once per gyro variant): a small sequential
//      recurrence turns the ~10 sparse gyro samples into a dense quaternion
//      trajectory, then collapses it into one relative quaternion PER
//      OUTPUT ROW (the "row LUT") — see integrate_gyro_to_fine_trajectory()
//      and build_row_lut() below, and kernels.cu's file header for why this
//      step stays on the CPU rather than becoming a second GPU kernel.
//   3. GPU CORRECTION: upload the row LUT (set_row_lut) and run the ONE GPU
//      kernel in this project (launch_rs_correct) — a per-pixel map that
//      resolves the row-time fixed point and bilinearly samples the RS
//      frame. Repeated once for the CLEAN gyro and once for the DEGRADED
//      gyro (step 2 + step 3), the gyro-degradation study this project's
//      catalog bullet asks for.
//   4. VERIFY: every GPU run is compared, element-wise, against
//      reference_cpu.cpp's INDEPENDENT twin (max-abs-diff, documented
//      tolerance) — the standard GPU-vs-CPU gate every project in this
//      repo carries.
//   5. EIGHT INDEPENDENT GATES, each checking something the twin comparison
//      CANNOT (see kernels.cuh/reference_cpu.cpp's twin-independence
//      ruling — a shared bug in the camera-model primitives would pass
//      VERIFY but fail here):
//        quat_integration_analytic — integrate a KNOWN constant angular
//                                    velocity and compare to the CLOSED-
//                                    FORM analytic rotation angle; bypasses
//                                    every camera-model primitive.
//        restoration                — corrected-vs-ground-truth mean abs
//                                    error (clean gyro), in the masked
//                                    "scored" region.
//        restoration_negative_control — the SAME region's error with NO
//                                    correction applied (raw RS frame vs
//                                    ground truth) — must be several-fold
//                                    larger, proving the correction is
//                                    doing real work.
//        straightness_corrected      — the scene's known vertical marker
//                                    line must measure straight (low row-
//                                    to-row spread) in the corrected image.
//        straightness_negative_control — the SAME line in the UNCORRECTED
//                                    RS frame must measure curved/sheared.
//        row_time_convergence        — the fixed-point search's iteration-2
//                                    vs iteration-3 delta, maxed over every
//                                    valid pixel, must be small.
//        gyro_degradation            — correcting with the DEGRADED gyro
//                                    must still beat doing nothing (bias +
//                                    noise make it worse than the clean
//                                    correction, but not worthless — see
//                                    THEORY.md "Where this sits in the real
//                                    world" for why VIO estimates bias
//                                    online instead of trusting raw gyro).
//        valid_coverage              — the fraction of output pixels whose
//                                    resolved source pixel actually lands
//                                    inside the RS frame (rows near the
//                                    top/bottom lose the most — see README
//                                    "Expected output").
//   6. ARTIFACTS: demo/out/{rs_input.pgm, corrected.pgm, ground_truth_gs.pgm,
//      uncorrected_diff.pgm, corrected_diff.pgm, rotation_profile.csv,
//      gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY:", every "GATE <name>:" verdict line, "ARTIFACT:", and
// "RESULT:" — each a PASS/FAIL verdict with NO embedded numbers, so it is
// identical on every GPU architecture. Measured numbers live on separate
// "[info]"/"[time]" lines (NOT diffed by demo/run_demo.*) — this project's
// floating-point pipeline can differ by a few ULP across sm_75/sm_86/sm_89
// (THEORY.md "Numerical considerations"). Change a stable line => update
// demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Gate tolerances — every number below is either a physical/numerical
// argument (documented inline) or a floor/ceiling calibrated from an ACTUAL
// measured run on this project's committed sample (CLAUDE.md paragraph 8:
// "never fabricate"), with margin so the gate stays robust to legitimate
// cross-GPU float differences (see the output-contract note above).
//
//   Measured on the reference machine (RTX 2080 SUPER, sm_75), Release:
//     verify(corrected, clean)     max|gpu-cpu| = 1.0000       (uint8 units)
//     verify(corrected, degraded)  max|gpu-cpu| = 0.0000       (uint8 units)
//     verify(iter_delta, both)     max|gpu-cpu| = 0.00006      (px)
//     verify(valid_mask, both)     mismatched pixels = 0 / 110592
//     quat_integration_analytic    |measured-analytic| = 0.000026 rad
//     restoration (clean)          masked mean|err| = 0.8335   (0..255 scale)
//     restoration_negative_control masked mean|err| = 3.9893   (0..255 scale)
//     straightness_corrected       line spread = 0.5219 px
//     straightness_negative_control line spread = 4.8504 px
//     row_time_convergence         max delta = 0.00221 px
//     gyro_degradation             clean = 0.8335, degraded = 1.3196
//     valid_coverage                98.16 %
// ===========================================================================

// -- GPU-vs-CPU VERIFY tolerances (documented "how far can two independent
// IEEE-754 float pipelines legitimately drift" bounds, not physical bounds).
static constexpr double kTolVerifyUint8 = 2.0;          // corrected image, 0..255 scale
static constexpr double kTolVerifyIterDeltaPx = 0.01;    // iter_delta, px
static constexpr int    kTolVerifyValidMismatch = 8;     // # pixels whose valid flag may disagree (boundary ties)

// -- Physical gate tolerances, each a floor/ceiling with margin over the
// measured values above (never AT the measured value — 01.01's calibration
// discipline).
static constexpr double kTolQuatAnalyticRad = 1e-4;             // ceiling, ~3.8x measured 0.000026 rad
static constexpr double kTolRestorationMean = 3.0;               // ceiling, ~3.6x measured 0.8335 (0..255 scale)
static constexpr double kFloorRestorationBaselineMean = 3.0;     // floor,  ~0.75x measured 3.9893 (negative control)
static constexpr double kTolStraightCorrectedPx = 1.5;           // ceiling, ~2.9x measured 0.5219 px
static constexpr double kFloorStraightRawPx = 3.5;                // floor,  ~0.72x measured 4.8504 px (negative control)
static constexpr double kTolConvergencePx = 0.05;                 // ceiling, ~23x measured 0.00221 px
static constexpr double kFloorValidCoveragePct = 90.0;            // floor,  ~0.92x measured 98.16%
// Degradation gate: the degraded-gyro correction must still improve on
// doing nothing by a real margin, not just "technically less bad" — the
// 0.6 factor demands the degraded result land at or below 60% of the
// uncorrected baseline's error (measured: 1.3196 / 3.9893 = 33%, i.e. the
// degraded correction is still 3x better than nothing at all, comfortably
// inside this bound — see GATE gyro_degradation below for the honest
// framing of WHY it is nonetheless worse than the clean-gyro result).
static constexpr double kGyroDegradationMaxFraction = 0.6;

// ===========================================================================
// GyroSample / FineSample — main.cu-local bookkeeping for the host-side gyro
// integration pipeline (not part of kernels.cuh's GPU/CPU-twin contract —
// see kernels.cu's file header for why this step has no GPU counterpart).
// ===========================================================================
struct GyroSample {
    float t_s;              // frame-relative timestamp, s (see kernels.cuh's float-vs-double note below)
    float wx, wy, wz;        // body/camera-frame angular velocity, rad/s
};

// NOTE on using float (not double) for time here: docs/SYSTEM_DESIGN.md
// paragraph 3.5 asks for monotonic double timestamps in GENERAL robot
// messages; this project deliberately narrows to float because every
// timestamp here is a FRAME-RELATIVE offset bounded to about +-40 ms
// (kFrameT0S..kReadoutTimeS plus the gyro window's small margins) — float32
// gives sub-microsecond resolution at that magnitude, far finer than any
// real IMU's own timing jitter, so double would only add noise to the
// numbers without buying accuracy. Stated here, once, at this API boundary
// (CLAUDE.md paragraph 12's "document the exception" rule).
struct FineSample {
    float t;                 // frame-relative time, s
    Quat q;                  // q_world_cam(t), the integrated orientation at that instant
};

// ===========================================================================
// Minimal, STRICT PGM (P5) reader/writer — same discipline as 01.01's
// read_pgm/write_pgm: only ever reads files this project's own generator
// wrote, aborts rather than guesses on anything malformed.
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

// ===========================================================================
// read_gyro_csv — parse one gyro trace (t_s,wx_rad_s,wy_rad_s,wz_rad_s),
// skipping '#' provenance/comment lines and the one non-numeric header row.
// A row is treated as data the moment its first field parses as a number —
// a simple, honest way to tell the header apart from data without hard-
// coding a line count (see ../scripts/make_synthetic.py's writer for the
// exact format this reads).
// ===========================================================================
static bool read_gyro_csv(const std::string& path, std::vector<GyroSample>& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(ss, field, ',')) fields.push_back(field);
        if (fields.size() != 4) continue;

        char* end = nullptr;
        const double t = std::strtod(fields[0].c_str(), &end);
        if (end == fields[0].c_str()) continue;   // not a number -> this was the header row

        GyroSample s;
        s.t_s = static_cast<float>(t);
        s.wx = static_cast<float>(std::strtod(fields[1].c_str(), nullptr));
        s.wy = static_cast<float>(std::strtod(fields[2].c_str(), nullptr));
        s.wz = static_cast<float>(std::strtod(fields[3].c_str(), nullptr));
        out.push_back(s);
    }
    return out.size() >= 2;   // need at least two samples to form one integration interval
}

// ===========================================================================
// integrate_gyro_to_fine_trajectory — the host-side sequential gyro
// integration (kernels.cu's file header explains why this stays on the
// CPU). Anchors q = IDENTITY at the first gyro sample's own timestamp (an
// arbitrary but self-consistent choice — see kernels.cuh's file header:
// only the RELATIVE rotation between rows and the reference instant is ever
// observed downstream, so the anchor's absolute orientation never matters).
//
// Between each pair of consecutive gyro samples, omega is linearly
// interpolated and each of kIntegrationSubsteps sub-steps uses the
// MIDPOINT-in-time omega estimate (a standard accuracy improvement over
// evaluating omega at a sub-step's start or end — THEORY.md "Numerical
// considerations" quantifies the difference) with quat_integrate_step's
// exact exponential-map update.
// ===========================================================================
static std::vector<FineSample> integrate_gyro_to_fine_trajectory(const std::vector<GyroSample>& gyro)
{
    std::vector<FineSample> fine;
    fine.reserve(static_cast<size_t>(gyro.size()) * static_cast<size_t>(kIntegrationSubsteps) + 1);

    Quat q{ 1.0f, 0.0f, 0.0f, 0.0f };            // integration anchor: identity at gyro[0]'s timestamp
    fine.push_back(FineSample{ gyro[0].t_s, q });

    for (size_t i = 0; i + 1 < gyro.size(); ++i) {
        const GyroSample& a = gyro[i];
        const GyroSample& b = gyro[i + 1];
        const float interval = b.t_s - a.t_s;                       // this gyro interval's duration, s
        const float sub_dt = interval / static_cast<float>(kIntegrationSubsteps);

        for (int s = 1; s <= kIntegrationSubsteps; ++s) {
            // Midpoint-in-time interpolation fraction for THIS sub-step
            // (s-1 -> s), not the sub-step's endpoint — the standard
            // "evaluate the derivative at the midpoint" accuracy trick.
            const float frac_mid = (static_cast<float>(s) - 0.5f) / static_cast<float>(kIntegrationSubsteps);
            const float wx = a.wx + (b.wx - a.wx) * frac_mid;
            const float wy = a.wy + (b.wy - a.wy) * frac_mid;
            const float wz = a.wz + (b.wz - a.wz) * frac_mid;
            q = quat_integrate_step(q, wx, wy, wz, sub_dt);
            fine.push_back(FineSample{ a.t_s + static_cast<float>(s) * sub_dt, q });
        }
    }
    return fine;
}

// ===========================================================================
// interpolate_fine — bracket + linearly interpolate the fine trajectory at
// an arbitrary time t (binary search: fine[] is time-ordered, and this is
// called kImgH times per gyro variant, so an O(log n) lookup is worth the
// extra few lines over a linear scan).
// ===========================================================================
static Quat interpolate_fine(const std::vector<FineSample>& fine, float t)
{
    if (t <= fine.front().t) return fine.front().q;
    if (t >= fine.back().t) return fine.back().q;

    size_t lo = 0, hi = fine.size() - 1;
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2;
        if (fine[mid].t <= t) lo = mid; else hi = mid;
    }
    const float t0 = fine[lo].t, t1 = fine[hi].t;
    const float frac = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;
    const Quat a = fine[lo].q, b = fine[hi].q;
    Quat r{ a.w + (b.w - a.w) * frac, a.x + (b.x - a.x) * frac,
            a.y + (b.y - a.y) * frac, a.z + (b.z - a.z) * frac };
    return quat_normalize(r);
}

// ===========================================================================
// build_row_lut — evaluate q_rel(v) = conj(q_row(v)) (x) q_ref for every
// output row v (kernels.cuh's file header derives this formula). O(kImgH)
// = 288 evaluations; folded into the same cheap host setup as the gyro
// integration above rather than becoming its own GPU kernel (kernels.cu's
// file header explains the design choice for BOTH steps together).
// ===========================================================================
static std::vector<Quat> build_row_lut(const std::vector<FineSample>& fine, Quat q_ref)
{
    std::vector<Quat> lut(static_cast<size_t>(kImgH));
    for (int v = 0; v < kImgH; ++v) {
        const float t_row = kFrameT0S + static_cast<float>(v) * kLineTimeS;
        const Quat q_row = interpolate_fine(fine, t_row);
        lut[static_cast<size_t>(v)] = quat_normalize(quat_mul(quat_conj(q_row), q_ref));
    }
    return lut;
}

// ===========================================================================
// masked_mae — mean absolute error between two grayscale images over the
// "scored" region: excludes a band around the known marker line (column
// kCx) and a thin border strip (kernels.cuh documents both margins), and —
// when a valid_mask is supplied — excludes pixels the fixed-point search
// marked invalid (there is nothing meaningful to score there). Mirrors
// 01.01's color_fidelity gate's smooth-mask reasoning: even a PERFECT
// correction shows more error immediately next to a hard edge, so scoring
// that region would penalize the pipeline for physics it cannot avoid.
// ===========================================================================
struct MaskedError { double mean; long long n; };

static MaskedError masked_mae(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b,
                              const std::vector<unsigned char>* valid_mask, int W, int H)
{
    double sum = 0.0;
    long long n = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            const bool near_line = std::fabs(static_cast<float>(x) - kCx) <= static_cast<float>(kRestorationMaskMarginPx);
            const bool near_border = (x < kBorderMarginPx) || (x >= W - kBorderMarginPx)
                                    || (y < kBorderMarginPx) || (y >= H - kBorderMarginPx);
            const bool is_valid = (valid_mask == nullptr) || ((*valid_mask)[static_cast<size_t>(idx)] != 0);
            if (near_line || near_border || !is_valid) continue;
            sum += std::fabs(static_cast<double>(a[static_cast<size_t>(idx)]) - static_cast<double>(b[static_cast<size_t>(idx)]));
            ++n;
        }
    }
    return MaskedError{ n > 0 ? sum / static_cast<double>(n) : 0.0, n };
}

// ===========================================================================
// find_line_center_x — sub-pixel column of the marker line's CENTER along
// scanline y, searched in [x_lo, x_hi]. The marker is a bright, thin
// vertical stripe on a darker hashed background (../scripts/make_synthetic.py);
// this finds the ENTER crossing (dark->bright) and the EXIT crossing
// (bright->dark), both via linear sub-pixel interpolation (same technique
// 01.01's find_crossing_x uses for its single step edge), and returns their
// midpoint. A HOST-side, from-scratch detector — bypasses every kernel and
// every kernels.cuh primitive on purpose (this gate exercises the ACTUAL
// pixel content of the images, per the twin-independence ruling).
// ===========================================================================
static bool find_line_center_x(const std::vector<unsigned char>& img, int W, int H, int y,
                               double x_lo, double x_hi, double threshold, double& out_x)
{
    if (y < 0 || y >= H) return false;
    const int xl = std::max(0, static_cast<int>(std::floor(x_lo)));
    const int xh = std::min(W - 2, static_cast<int>(std::ceil(x_hi)));
    double x_enter = -1.0, x_exit = -1.0;
    for (int x = xl; x <= xh; ++x) {
        const double a = static_cast<double>(img[static_cast<size_t>(y) * W + x]);
        const double b = static_cast<double>(img[static_cast<size_t>(y) * W + x + 1]);
        if (x_enter < 0.0 && a < threshold && b >= threshold) {
            x_enter = static_cast<double>(x) + (threshold - a) / (b - a);
        } else if (x_enter >= 0.0 && x_exit < 0.0 && a >= threshold && b < threshold) {
            x_exit = static_cast<double>(x) + (a - threshold) / (a - b);
            break;
        }
    }
    if (x_enter < 0.0 || x_exit < 0.0) return false;
    out_x = 0.5 * (x_enter + x_exit);
    return true;
}

// line_spread — the max deviation from the mean line-center-x, sampled
// every 6 rows over a margin-trimmed row range. A LOW spread means the
// line reads as (nearly) vertical; a HIGH spread means it is sheared —
// exactly the "how curved is a known-straight feature" measurement 01.01's
// straightness gate performs, adapted from a step edge to a thin line.
struct SpreadResult { double spread; int n_found; };

static SpreadResult line_spread(const std::vector<unsigned char>& img, int W, int H, double threshold)
{
    std::vector<double> xs;
    const int y_lo = 24, y_hi = H - 24;
    // Search band: +-30 px around kCx comfortably covers both the
    // near-vertical corrected/ground-truth line AND the RAW frame's
    // several-pixel-to-tens-of-pixels shifted line (README "Expected
    // output" reports the measured shift).
    const double x_lo = static_cast<double>(kCx) - 30.0, x_hi = static_cast<double>(kCx) + 30.0;
    for (int y = y_lo; y <= y_hi; y += 6) {
        double x;
        if (find_line_center_x(img, W, H, y, x_lo, x_hi, threshold, x)) xs.push_back(x);
    }
    if (xs.empty()) return SpreadResult{ -1.0, 0 };
    double mean = 0.0;
    for (double v : xs) mean += v;
    mean /= static_cast<double>(xs.size());
    double maxdev = 0.0;
    for (double v : xs) maxdev = std::max(maxdev, std::fabs(v - mean));
    return SpreadResult{ maxdev, static_cast<int>(xs.size()) };
}

// The marker line's crossing threshold: ../scripts/make_synthetic.py draws
// the hashed background in [30,140] and the line itself at 255 — 195 sits
// comfortably in the gap so background texture can never trigger a false
// crossing (MUST MATCH that script's LINE_INTENSITY/background range).
static constexpr double kLineThreshold = 195.0;

// ===========================================================================
// max_abs_diff / valid_mismatch_count — generic GPU-vs-CPU comparators
// shared by every VERIFY checkpoint below.
// ===========================================================================
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

static int valid_mismatch_count(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b)
{
    int n = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++n;
    return n;
}

// ===========================================================================
// gates_metrics.csv writer + float formatter — same shape as 01.01's.
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

static bool write_rotation_profile_csv(const std::string& path, const std::vector<Quat>& row_lut)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "row,t_row_s,angle_rel_deg\n";
    for (int v = 0; v < kImgH; ++v) {
        const float t_row = kFrameT0S + static_cast<float>(v) * kLineTimeS;
        const Quat q = row_lut[static_cast<size_t>(v)];
        // Rotation angle of q_rel(v) from identity: angle = 2*acos(|w|) —
        // |w| (not w) picks the SHORTER of the double-covering pair
        // {q,-q}, both of which represent the identical physical rotation
        // (docs/SYSTEM_DESIGN.md paragraph 3.4's double-cover note).
        const double w_clamped = std::min(1.0, std::max(-1.0, static_cast<double>(std::fabs(q.w))));
        const double angle_deg = 2.0 * std::acos(w_clamped) * (180.0 / 3.14159265358979323846);
        out << v << "," << fmt(static_cast<double>(t_row), 6) << "," << fmt(angle_deg, 5) << "\n";
    }
    return static_cast<bool>(out);
}

// ===========================================================================
// diff_heatmap — |a-b| per pixel as a directly-viewable grayscale image
// (clamped to 0..255 — with this project's error magnitudes, no scaling is
// needed for the difference to be visible; unlike 01.01's normalized_to_vis
// this is display-only in exactly the SAME units as the source images).
// ===========================================================================
static std::vector<unsigned char> diff_heatmap(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b)
{
    std::vector<unsigned char> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        out[i] = static_cast<unsigned char>(std::min(255, d));
    }
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
        else {
            std::fprintf(stderr, "usage: %s [--data path/to/data/sample]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] rolling-shutter correction using IMU rates: gyro-integrated row-homography "
               "un-warping, GPU fixed-point search vs CPU twin, clean vs degraded gyro (project 01.10)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d RS frame, readout=%.1fms (t_line=%.2fus), gyro=%.0fHz, "
               "fixed-point iters=%d, GS reference row=%.1f (t_ref=%.2fms)\n",
               kImgW, kImgH, static_cast<double>(kReadoutTimeS) * 1000.0,
               static_cast<double>(kLineTimeS) * 1e6, static_cast<double>(kGyroRateHz),
               kFixedPointIters, (kImgH - 1) * 0.5, static_cast<double>(kFrameTRefS) * 1000.0);

    // ---- data ---------------------------------------------------------------
    const std::string rs_path = find_data_file(data_dir, argv[0], "rs_input.pgm");
    const std::string gt_path = find_data_file(data_dir, argv[0], "ground_truth_gs.pgm");
    const std::string gyro_clean_path = find_data_file(data_dir, argv[0], "gyro_clean.csv");
    const std::string gyro_degraded_path = find_data_file(data_dir, argv[0], "gyro_degraded.csv");
    if (rs_path.empty() || gt_path.empty() || gyro_clean_path.empty() || gyro_degraded_path.empty()) {
        std::fprintf(stderr, "sample: one or more of rs_input.pgm/ground_truth_gs.pgm/gyro_clean.csv/"
                             "gyro_degraded.csv not found (run scripts/make_synthetic.py?)\n");
        std::printf("DATA: NOT FOUND (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }

    int rs_w = 0, rs_h = 0, gt_w = 0, gt_h = 0;
    std::vector<unsigned char> rs_input, ground_truth;
    std::vector<GyroSample> gyro_clean, gyro_degraded;
    const bool loaded = read_pgm(rs_path, rs_w, rs_h, rs_input)
                      && read_pgm(gt_path, gt_w, gt_h, ground_truth)
                      && read_gyro_csv(gyro_clean_path, gyro_clean)
                      && read_gyro_csv(gyro_degraded_path, gyro_degraded)
                      && rs_w == kImgW && rs_h == kImgH && gt_w == kImgW && gt_h == kImgH;
    if (!loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: %dx%d synthetic RS capture of a hashed multi-scale texture + known vertical "
               "marker line, plus %zu clean / %zu degraded gyro samples [synthetic, seed 42]\n",
               kImgW, kImgH, gyro_clean.size(), gyro_degraded.size());

    const int n_pixels = kImgW * kImgH;

    // ---- GATE: quaternion-integration analytic self-check -------------------
    // Integrate a KNOWN constant angular velocity for a KNOWN duration and
    // compare to the closed-form analytic rotation angle |omega|*dt. This
    // bypasses lerp_row_quat/apply_row_rotation/bilinear_sample_gray
    // entirely — the "at least one gate that does not route through the
    // shared camera-model code" requirement (reference_cpu.cpp's header).
    const float kTestOmegaRadS = 1.234f;      // constant angular velocity about Z, rad/s (arbitrary, nonzero)
    const float kTestDurationS = 0.010f;      // 10 ms test window
    const int kTestSubsteps = 20;             // sub-step the SAME way real integration does
    Quat q_test{ 1.0f, 0.0f, 0.0f, 0.0f };
    for (int s = 0; s < kTestSubsteps; ++s) {
        q_test = quat_integrate_step(q_test, 0.0f, 0.0f, kTestOmegaRadS, kTestDurationS / static_cast<float>(kTestSubsteps));
    }
    const double w_clamped = std::min(1.0, std::max(-1.0, static_cast<double>(std::fabs(q_test.w))));
    const double measured_angle_rad = 2.0 * std::acos(w_clamped);
    const double analytic_angle_rad = static_cast<double>(kTestOmegaRadS) * static_cast<double>(kTestDurationS);
    const double quat_analytic_err = std::fabs(measured_angle_rad - analytic_angle_rad);
    const bool gate_quat_analytic = quat_analytic_err <= kTolQuatAnalyticRad;
    std::printf("GATE quat_integration_analytic: %s\n", gate_quat_analytic ? "PASS" : "FAIL");
    std::printf("[info] quat_integration_analytic: |measured-analytic| = %.8f rad (tol %.4f; measured=%.6f, analytic=%.6f)\n",
               quat_analytic_err, kTolQuatAnalyticRad, measured_angle_rad, analytic_angle_rad);

    // ---- device buffers (allocated once, reused per gyro variant) ----------
    unsigned char* d_rs_frame = nullptr;
    unsigned char* d_corrected = nullptr;
    unsigned char* d_valid = nullptr;
    float* d_iter_delta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rs_frame, static_cast<size_t>(n_pixels)));
    CUDA_CHECK(cudaMalloc(&d_corrected, static_cast<size_t>(n_pixels)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<size_t>(n_pixels)));
    CUDA_CHECK(cudaMalloc(&d_iter_delta, static_cast<size_t>(n_pixels) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rs_frame, rs_input.data(), static_cast<size_t>(n_pixels), cudaMemcpyHostToDevice));

    // ---- uncorrected baseline (negative control for the restoration gate,
    //      computed ONCE — it does not depend on the gyro variant at all) ---
    const MaskedError baseline = masked_mae(rs_input, ground_truth, nullptr, kImgW, kImgH);

    // ---- per-gyro-variant run: build row LUT, run GPU + CPU, verify --------
    struct VariantResult { std::string name; MaskedError restoration; double coverage_pct; };
    std::vector<VariantResult> results;

    // Saved from the CLEAN run for the gates/artifacts that follow the loop
    // (straightness, convergence, artifacts all use the CLEAN correction —
    // the DEGRADED run exists purely for the gyro_degradation study).
    std::vector<unsigned char> corrected_clean, valid_clean;
    std::vector<float> iter_delta_clean;
    std::vector<Quat> row_lut_clean;
    bool verify_pass = true;

    const std::vector<GyroSample>* variant_gyro[2] = { &gyro_clean, &gyro_degraded };
    const char* variant_name[2] = { "clean", "degraded" };

    for (int variant = 0; variant < 2; ++variant) {
        CpuTimer setup_timer; setup_timer.begin();
        const std::vector<FineSample> fine = integrate_gyro_to_fine_trajectory(*variant_gyro[variant]);
        const Quat q_ref = interpolate_fine(fine, kFrameTRefS);
        const std::vector<Quat> row_lut = build_row_lut(fine, q_ref);
        const double setup_ms = setup_timer.end_ms();

        set_row_lut(row_lut.data(), kImgH);

        std::vector<unsigned char> corrected_gpu(static_cast<size_t>(n_pixels));
        std::vector<unsigned char> valid_gpu(static_cast<size_t>(n_pixels));
        std::vector<float> iter_delta_gpu(static_cast<size_t>(n_pixels));

        GpuTimer gpu_timer; gpu_timer.begin();
        launch_rs_correct(d_rs_frame, d_corrected, d_valid, d_iter_delta, kImgW, kImgH);
        const float gpu_ms = gpu_timer.end_ms();

        CUDA_CHECK(cudaMemcpy(corrected_gpu.data(), d_corrected, corrected_gpu.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(valid_gpu.data(), d_valid, valid_gpu.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(iter_delta_gpu.data(), d_iter_delta, iter_delta_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<unsigned char> corrected_cpu(static_cast<size_t>(n_pixels));
        std::vector<unsigned char> valid_cpu(static_cast<size_t>(n_pixels));
        std::vector<float> iter_delta_cpu(static_cast<size_t>(n_pixels));
        CpuTimer cpu_timer; cpu_timer.begin();
        rs_correct_cpu(row_lut.data(), kImgH, rs_input.data(), kImgW, corrected_cpu.data(), valid_cpu.data(), iter_delta_cpu.data());
        const double cpu_ms = cpu_timer.end_ms();

        const double d_corrected_max = max_abs_diff(corrected_gpu, corrected_cpu);
        const double d_iter_max = max_abs_diff(iter_delta_gpu, iter_delta_cpu);
        const int d_valid_mismatch = valid_mismatch_count(valid_gpu, valid_cpu);
        const bool this_verify_pass = (d_corrected_max <= kTolVerifyUint8)
                                     && (d_iter_max <= kTolVerifyIterDeltaPx)
                                     && (d_valid_mismatch <= kTolVerifyValidMismatch);
        verify_pass = verify_pass && this_verify_pass;

        std::printf("[time] gyro=%s: host setup (gyro integration + row LUT) %.3f ms | GPU kernel %.3f ms | CPU twin %.3f ms\n",
                   variant_name[variant], setup_ms, static_cast<double>(gpu_ms), cpu_ms);
        std::printf("[info] verify(corrected, %s): max|gpu-cpu| = %.4f (tol %.2f) | verify(iter_delta): max|gpu-cpu| = %.5f px "
                   "(tol %.2f) | verify(valid_mask): %d mismatched px (tol %d)\n",
                   variant_name[variant], d_corrected_max, kTolVerifyUint8, d_iter_max, kTolVerifyIterDeltaPx,
                   d_valid_mismatch, kTolVerifyValidMismatch);

        const MaskedError restoration = masked_mae(corrected_gpu, ground_truth, &valid_gpu, kImgW, kImgH);
        long long n_valid = 0;
        for (unsigned char v : valid_gpu) if (v) ++n_valid;
        const double coverage_pct = 100.0 * static_cast<double>(n_valid) / static_cast<double>(n_pixels);
        results.push_back(VariantResult{ variant_name[variant], restoration, coverage_pct });

        if (variant == 0) {   // clean — keep everything the post-loop gates/artifacts need
            corrected_clean = corrected_gpu;
            valid_clean = valid_gpu;
            iter_delta_clean = iter_delta_gpu;
            row_lut_clean = row_lut;
        }
    }
    std::printf("VERIFY: %s (GPU matches CPU reference within documented tolerance, both gyro variants: "
               "corrected image, row-time convergence, valid mask)\n", verify_pass ? "PASS" : "FAIL");

    // ---- GATE: row_time_convergence — the max iteration-2-vs-3 delta over
    //      every VALID pixel of the CLEAN run (an invalid pixel's search
    //      path resolved to a source outside the frame, so its convergence
    //      is not meaningful to report), taken directly from the GPU
    //      kernel's own d_iter_delta output saved above -- already cross-
    //      checked against the CPU twin by the VERIFY step, so no further
    //      recomputation is needed here. --------------------------------
    double max_iter_delta = 0.0;
    for (int i = 0; i < n_pixels; ++i)
        if (valid_clean[static_cast<size_t>(i)]) max_iter_delta = std::max(max_iter_delta, static_cast<double>(iter_delta_clean[static_cast<size_t>(i)]));
    const bool gate_convergence = max_iter_delta <= kTolConvergencePx;
    std::printf("GATE row_time_convergence: %s\n", gate_convergence ? "PASS" : "FAIL");
    std::printf("[info] row_time_convergence: max|iter3-iter2| over valid pixels = %.5f px (tol %.2f)\n",
               max_iter_delta, kTolConvergencePx);

    // ---- GATE: restoration + restoration_negative_control ------------------
    const MaskedError& clean_restoration = results[0].restoration;
    const bool gate_restoration = clean_restoration.mean <= kTolRestorationMean;
    const bool gate_restoration_negctrl = baseline.mean >= kFloorRestorationBaselineMean;
    std::printf("GATE restoration: %s\n", gate_restoration ? "PASS" : "FAIL");
    std::printf("[info] restoration: corrected (clean gyro) masked mean|err| = %.4f (tol %.2f, n=%lld, 0..255 scale)\n",
               clean_restoration.mean, kTolRestorationMean, clean_restoration.n);
    std::printf("GATE restoration_negative_control: %s\n", gate_restoration_negctrl ? "PASS" : "FAIL");
    std::printf("[info] restoration_negative_control: UNCORRECTED masked mean|err| = %.4f (must be >= %.2f -- "
               "proves the correction is doing real work, n=%lld)\n",
               baseline.mean, kFloorRestorationBaselineMean, baseline.n);

    // ---- GATE: straightness_corrected + straightness_negative_control ------
    const SpreadResult spread_corrected = line_spread(corrected_clean, kImgW, kImgH, kLineThreshold);
    const SpreadResult spread_raw = line_spread(rs_input, kImgW, kImgH, kLineThreshold);
    const bool gate_straight_corrected = (spread_corrected.spread >= 0.0) && (spread_corrected.spread <= kTolStraightCorrectedPx);
    const bool gate_straight_negctrl = (spread_raw.spread >= 0.0) && (spread_raw.spread >= kFloorStraightRawPx);
    std::printf("GATE straightness_corrected: %s\n", gate_straight_corrected ? "PASS" : "FAIL");
    std::printf("[info] straightness_corrected: marker-line x-spread = %.4f px over %d rows (tol <= %.2f)\n",
               spread_corrected.spread, spread_corrected.n_found, kTolStraightCorrectedPx);
    std::printf("GATE straightness_negative_control: %s\n", gate_straight_negctrl ? "PASS" : "FAIL");
    std::printf("[info] straightness_negative_control: RAW (uncorrected) marker-line x-spread = %.4f px over %d rows "
               "(must be >= %.2f -- proves the RS skew being corrected is real)\n",
               spread_raw.spread, spread_raw.n_found, kFloorStraightRawPx);

    // ---- GATE: gyro_degradation ---------------------------------------------
    const MaskedError& degraded_restoration = results[1].restoration;
    const bool gate_degradation = degraded_restoration.mean <= baseline.mean * kGyroDegradationMaxFraction;
    std::printf("GATE gyro_degradation: %s\n", gate_degradation ? "PASS" : "FAIL");
    std::printf("[info] gyro_degradation: clean-gyro corrected mean|err| = %.4f | degraded-gyro corrected mean|err| = %.4f "
               "| uncorrected baseline = %.4f (degraded must be <= %.0f%% of baseline; degraded is worse than clean because "
               "its bias+noise are never estimated online -- see THEORY.md \"Where this sits in the real world\" on VIO)\n",
               clean_restoration.mean, degraded_restoration.mean, baseline.mean, kGyroDegradationMaxFraction * 100.0);

    // ---- GATE: valid_coverage ------------------------------------------------
    const double coverage_pct = results[0].coverage_pct;
    const bool gate_coverage = coverage_pct >= kFloorValidCoveragePct;
    std::printf("GATE valid_coverage: %s\n", gate_coverage ? "PASS" : "FAIL");
    std::printf("[info] valid_coverage: %.2f%% of output pixels resolved to a source pixel inside the RS frame "
               "(tol >= %.1f%%; the rest sit near the top/bottom rows, whose row-time is furthest from t_ref, "
               "so the row-time search's source column shifts the most -- see README \"Expected output\")\n",
               coverage_pct, kFloorValidCoveragePct);

    // ======================= ARTIFACTS ============================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    const std::vector<unsigned char> uncorrected_diff = diff_heatmap(rs_input, ground_truth);
    const std::vector<unsigned char> corrected_diff = diff_heatmap(corrected_clean, ground_truth);

    std::vector<CsvRow> csv;
    csv.push_back({ "quat_integration_analytic", "abs_err_rad", fmt(quat_analytic_err, 8), fmt(kTolQuatAnalyticRad, 4), gate_quat_analytic ? "PASS" : "FAIL" });
    csv.push_back({ "restoration", "masked_mean_abs_err_clean", fmt(clean_restoration.mean, 4), fmt(kTolRestorationMean, 2), gate_restoration ? "PASS" : "FAIL" });
    csv.push_back({ "restoration_negative_control", "masked_mean_abs_err_uncorrected", fmt(baseline.mean, 4), fmt(kFloorRestorationBaselineMean, 2), gate_restoration_negctrl ? "PASS" : "FAIL" });
    csv.push_back({ "straightness_corrected", "line_spread_px", fmt(spread_corrected.spread, 4), fmt(kTolStraightCorrectedPx, 2), gate_straight_corrected ? "PASS" : "FAIL" });
    csv.push_back({ "straightness_negative_control", "line_spread_px", fmt(spread_raw.spread, 4), fmt(kFloorStraightRawPx, 2), gate_straight_negctrl ? "PASS" : "FAIL" });
    csv.push_back({ "row_time_convergence", "max_iter3_minus_iter2_px", fmt(max_iter_delta, 5), fmt(kTolConvergencePx, 2), gate_convergence ? "PASS" : "FAIL" });
    csv.push_back({ "gyro_degradation", "degraded_masked_mean_abs_err", fmt(degraded_restoration.mean, 4), fmt(baseline.mean * kGyroDegradationMaxFraction, 4), gate_degradation ? "PASS" : "FAIL" });
    csv.push_back({ "valid_coverage", "pct_valid_pixels", fmt(coverage_pct, 2), fmt(kFloorValidCoveragePct, 1), gate_coverage ? "PASS" : "FAIL" });

    bool artifact_ok = !out_dir.empty();
    artifact_ok = artifact_ok
        && write_pgm(out_dir + "/rs_input.pgm", kImgW, kImgH, rs_input)
        && write_pgm(out_dir + "/corrected.pgm", kImgW, kImgH, corrected_clean)
        && write_pgm(out_dir + "/ground_truth_gs.pgm", kImgW, kImgH, ground_truth)
        && write_pgm(out_dir + "/uncorrected_diff.pgm", kImgW, kImgH, uncorrected_diff)
        && write_pgm(out_dir + "/corrected_diff.pgm", kImgW, kImgH, corrected_diff)
        && write_rotation_profile_csv(out_dir + "/rotation_profile.csv", row_lut_clean)
        && write_gates_csv(out_dir + "/gates_metrics.csv", csv);
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{rs_input.pgm, corrected.pgm, ground_truth_gs.pgm, "
                   "uncorrected_diff.pgm, corrected_diff.pgm, rotation_profile.csv, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");

    // ---- cleanup --------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_rs_frame));
    CUDA_CHECK(cudaFree(d_corrected));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_iter_delta));

    // ---- verdict ----------------------------------------------------------------
    const bool success = verify_pass && gate_quat_analytic && gate_restoration && gate_restoration_negctrl
                       && gate_straight_corrected && gate_straight_negctrl && gate_convergence
                       && gate_degradation && gate_coverage && artifact_ok;
    if (success) {
        std::printf("RESULT: PASS (VERIFY + all 8 gates passed: quat_integration_analytic, restoration, "
                   "restoration_negative_control, straightness_corrected, straightness_negative_control, "
                   "row_time_convergence, gyro_degradation, valid_coverage)\n");
    } else {
        std::printf("RESULT: FAIL (VERIFY or a gate above did not pass -- see GATE:/VERIFY:/[info] lines)\n");
    }
    return success ? 0 : 1;
}
