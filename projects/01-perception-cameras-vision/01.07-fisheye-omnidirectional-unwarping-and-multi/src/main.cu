// ===========================================================================
// main.cu — entry point for project 01.07
//           Fisheye/omnidirectional unwarping and multi-camera surround-view
//           stitching: single-camera unwarp (rectilinear + cylindrical) and
//           4-camera bird's-eye-view (BEV) stitching, both GPU-vs-CPU
//           verified and checked against seven independent physical gates
//
// What this program does, start to finish
// -----------------------------------------
//   1. Print the banner + GPU info; load the 4 committed synthetic fisheye
//      camera renders (front/left/right/rear, data/sample/fisheye_*.ppm)
//      and the ground-truth BEV crop (bev_ground_truth.ppm) — see
//      ../scripts/make_synthetic.py.
//   2. HALF 1 (single camera): build the rectilinear LUT and the
//      cylindrical LUT once (purely geometric), then bilinear-remap the
//      FRONT camera's fisheye image through each — two unwarp outputs.
//   3. HALF 2 (surround view): the 4-camera BEV compositor — one kernel,
//      one thread per BEV pixel, a 4-camera loop inside each thread
//      (kernels.cu's bev_compose_kernel).
//   4. VERIFY (the CLAUDE.md §5 GPU-vs-CPU gate): every kernel above is
//      compared, element-wise, against reference_cpu.cpp's INDEPENDENT
//      twin (max-abs-diff, documented per-stage tolerance).
//   5. SEVEN PHYSICAL GATES, each checking something the twin comparison
//      CANNOT (kernels.cuh's twin-independence notes — a shared bug would
//      pass VERIFY but fail here):
//        model_roundtrip           — project->unproject identity over a
//                                    theta grid including angles > 90 deg,
//                                    hand-retyped in double precision,
//                                    bypassing kernels.cuh entirely.
//        straightness_rectilinear  — a world-straight ground-plane edge
//                                    must measure STRAIGHT (small best-fit-
//                                    line residual) after rectilinear unwarp.
//        distortion_negative_control — the SAME edge in the RAW fisheye
//                                    image must measure CURVED (large
//                                    residual) — proof the curvature being
//                                    corrected is real, not a no-op.
//        bev_ground_truth          — the BEV output must match the
//                                    committed ground-truth texture in
//                                    flat, well-covered, seam-free,
//                                    object-free regions.
//        seam_consistency          — in 2-camera overlap regions, the two
//                                    cameras' PRE-blend samples of the SAME
//                                    ground point must agree — proof the
//                                    rig extrinsics + model are self-
//                                    consistent across cameras.
//        flat_ground_assumption    — the teaching negative control: BEV
//                                    error must be LARGE near tall objects
//                                    (the flat-ground assumption visibly
//                                    fails there) while staying small on
//                                    flat ground — the reason production
//                                    BEV systems restrict themselves to
//                                    ground content or fuse in real depth.
//        coverage                  — every BEV pixel inside the rig's
//                                    design radius is seen by >= 1 camera;
//                                    reports the >=2-camera overlap fraction.
//   6. ARTIFACTS: demo/out/{fisheye_front.ppm, rectilinear.ppm,
//      cylindrical.ppm, bev.ppm, coverage_map.pgm, error_heatmap.pgm,
//      gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY:", the seven "GATE <name>:" verdict lines, "ARTIFACT:",
// and "RESULT:" — every one a PASS/FAIL verdict with NO embedded numbers,
// identical on every GPU architecture. MEASURED numbers (pixel errors,
// timings, byte counts) are printed on separate "[info]"/"[time]" lines
// instead (NOT diffed) — same discipline as sibling flagship 01.01.
// Change a stable line => update demo/expected_output.txt in the same change.
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
// ACTUAL measured run on this project's committed sample (CLAUDE.md §8:
// "never fabricate" — measured values are recorded alongside each
// constant, with margin so the gate stays robust to legitimate cross-GPU
// float differences — same discipline as 01.01's tolerance block).
//
//   Measured on the reference machine (RTX 2080 SUPER, sm_75), Release,
//   committed sample (data/sample/, seed 42):
//     verify(rect/cyl LUT)              max|gpu-cpu| = 0.0000305 px
//     verify(rect/cyl remap)            max|gpu-cpu| = 1.0000 (uint8 units)
//     verify(bev compose)               max|gpu-cpu| = 1.0000 (uint8 units)
//     verify(bev coverage)              max|gpu-cpu| = 0 (exact — integer bitmask)
//     model_roundtrip                   max pixel error = 0.0000000 px (closed-form both ways, see kernels.cuh PART 1)
//     straightness_rectilinear          best-fit-line residual = 0.2947 px
//     distortion_negative_control       best-fit-line residual = 60.8862 px
//     bev_ground_truth                  flat-region mean|err| = 10.9495 (0..255 scale)
//     seam_consistency                  mean|A-B| in overlap  = 17.5811 (0..255 scale)
//     flat_ground_assumption            object-region mean|err| = 37.3365 (0..255 scale)
//     coverage                          design-radius coverage = 100.000% | >=2-cam overlap = 63.36%
//
//   Why bev_ground_truth/seam_consistency read "large" (10-18 on a 0..255
//   scale) even with a bug-free pipeline: this rig's fisheye lens has
//   fairly LOW angular resolution (74 px/radian ~= 1.3 px/degree — chosen
//   for a small, fast-to-build committed sample, THEORY.md "The GPU
//   mapping"), so a given ground-plane distance subtends very few fisheye
//   pixels once it is far from a camera or near the edge of its FOV. Thin,
//   high-contrast features (the 15 cm lane stripes, object silhouettes)
//   are the most exposed to this: a 1-2 px bilinear/feather blur at the
//   SOURCE image becomes a visible several-cm position error on the
//   ground. This is a genuine, physically-grounded reconstruction
//   limitation (README "Limitations & honesty" states it plainly), not a
//   bug — VERIFY above already proves the GPU/CPU AGREE on what the rig
//   geometry computes; these two gates test whether that computation is
//   itself ACCURATE against ground truth, and the honest answer on this
//   committed scene is "close, with textured/thin-feature error the
//   dominant term" — exactly the lesson production BEV systems learn too
//   (PRACTICE.md §4: higher-resolution sensors and multi-frame temporal
//   blending are how real systems shrink this further).
// ===========================================================================

static constexpr double kTolLutPx     = 1e-3;   // rect/cyl LUT (u,v), pixels — pure geometry, no iteration anywhere
static constexpr double kTolUint8     = 1.5;    // remap / BEV compose outputs, 0..255 scale (bilinear FMA-contraction drift)
static constexpr double kTolCoverage  = 0.5;    // BEV coverage bitmask — exact integer computation, should be bit-identical

static constexpr double kRoundtripTolPx        = 1e-3;  // ceiling, >> measured 0.0000000 px (pure double-precision trig, closed-form)
static constexpr double kStraightRectTolPx     = 1.0;   // ceiling, ~3.4x measured 0.2947 px
static constexpr double kStraightFisheyeMinPx  = 40.0;  // floor, ~66% of measured 60.8862 px (negative control)
static constexpr double kBevGroundTruthTolMean = 15.0;  // ceiling, ~1.4x measured 10.9495 (0..255 scale; see note above)
static constexpr double kSeamConsistencyTolMean= 24.0;  // ceiling, ~1.4x measured 17.5811 (0..255 scale; see note above)
static constexpr double kFlatGroundMinMeanErr  = 20.0;  // floor, ~54% of measured 37.3365 (negative control)
static constexpr double kCoverageMinFraction   = 0.995; // >= 99.5% of design-radius pixels must have >=1 camera

// Design radius for the coverage gate: the largest ground-plane circle
// (centered on the vehicle) this rig's 4-camera FOV footprint is expected
// to blanket completely — set BELOW kBevRangeM (4.0 m) because the BEV
// crop's own CORNERS are farther from the vehicle center than any edge
// midpoint, and the rig's 92.5-deg-half-FOV cameras, tilted 45 deg down
// from mounts 0.6-1.1 m up, do not reach every corner (measured below).
static constexpr double kBevDesignRadiusM = 3.4;

// ===========================================================================
// Scene-layout constants — MUST MATCH ../scripts/make_synthetic.py's "MUST
// MATCH kernels.cuh / main.cu" block (same cross-referencing discipline as
// 01.01's checkerboard constants). Two groups: (a) the straightness-gate
// boundary edge's approximate pixel footprint in the FRONT camera's
// fisheye and rectilinear views (computed once, offline, from the SAME
// rig+fisheye formulas kernels.cuh defines, by walking the boundary line's
// world-space extent through the model — see THEORY.md "How we verify
// correctness" for the worked numbers); (b) the 3 tall objects' ground
// footprints, needed only for the flat_ground_assumption gate's masking.
// ===========================================================================

// Boundary edge (a straight ground-plane line at world Y = kBoundaryY,
// spanning X in [kBoundaryX0, kBoundaryX1]) separating a light "loading
// zone" from dark asphalt — see make_synthetic.py's file header for the
// full texture description. Search windows below were computed by walking
// this line's endpoints through the FRONT camera's rig extrinsic + the
// fisheye/rectilinear projection formulas (kernels.cuh) at scene-design
// time; they are generous (padded), not exact, because the detector below
// only needs to CONTAIN the true crossing, not predict it precisely.
static constexpr double kBoundaryY = -0.4;
static constexpr double kBoundaryX0 = 2.0, kBoundaryX1 = 6.5;
// Rectilinear search window (kRectW=200 x kRectH=150): the edge crosses
// roughly rows 5-90, columns 100-200 over the visible sub-range.
static constexpr int kRectSearchYLo = 5, kRectSearchYHi = 130;
static constexpr int kRectSearchXLo = 90, kRectSearchXHi = 199;
// Fisheye search window (kFishW=320 x kFishH=240): the edge crosses
// roughly rows 65-175, columns 155-215.
static constexpr int kFishSearchYLo = 60, kFishSearchYHi = 200;
static constexpr int kFishSearchXLo = 140, kFishSearchXHi = 230;
// Crossing threshold: dark asphalt (~44-72 gray) vs light zone (~195 gray)
// — a value comfortably between the two.
static constexpr int kBoundaryThreshold = 130;

// Tall objects (world X, Y, footprint radius meters) — MUST MATCH
// ../scripts/make_synthetic.py's OBJECTS list. Only the ground footprint
// matters here (the flat_ground_assumption gate masks by (X,Y) distance,
// not by height or shape — kernels.cuh's PART 3 header explains WHY any
// tall object breaks the flat-ground assumption regardless of its exact
// geometry). kObjectMaskMarginM pads each footprint to also catch the
// radial "ghost" smear immediately around the object (THEORY.md "The
// problem" derives the smear direction).
struct SceneObject { double x, y, footprint_r; };
static const SceneObject kSceneObjects[3] = {
    { 3.0, 1.2, 0.30 },    // red cylinder
    { -2.5, -1.0, 0.30 },  // blue cylinder
    { 0.3, 1.8, 0.47 },    // yellow box (radius = hypot(0.25,0.4), conservative circular mask)
};
static constexpr double kObjectMaskMarginM = 0.35;
// Seam mask margin, in BEV pixels: pixels within this many pixels of a
// coverage-bitmask CHANGE (a camera boundary) are excluded from the
// bev_ground_truth gate's "flat" measurement (blend imperfections
// concentrate right at a seam, which is seam_consistency's job to gate,
// not bev_ground_truth's).
static constexpr int kSeamMarginPx = 3;

// ===========================================================================
// Minimal, STRICT PGM (P5) / PPM (P6) readers/writers — same discipline as
// 01.01's: only ever reads files this project's own generator wrote.
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
// Sample loading — 4 fisheye views + the BEV ground-truth crop, dimension-
// checked against kernels.cuh's constants (strict: any mismatch aborts
// rather than silently truncating, same as every other project here).
// ===========================================================================
struct Sample {
    std::vector<unsigned char> front, left, right, rear;   // each kFishW*kFishH*3
    std::vector<unsigned char> bev_truth;                  // kBevW*kBevH*3
    bool loaded = false;
};

static Sample load_sample(const std::string& cli_dir, const char* argv0)
{
    Sample s;
    const char* names[5] = { "fisheye_front.ppm", "fisheye_left.ppm", "fisheye_right.ppm",
                             "fisheye_rear.ppm", "bev_ground_truth.ppm" };
    std::string paths[5];
    for (int i = 0; i < 5; ++i) {
        paths[i] = find_data_file(cli_dir, argv0, names[i]);
        if (paths[i].empty()) {
            std::fprintf(stderr, "sample: %s not found (run scripts/make_synthetic.py?)\n", names[i]);
            return s;
        }
    }
    int w, h;
    if (!read_ppm(paths[0], w, h, s.front) || w != kFishW || h != kFishH) { std::fprintf(stderr, "sample: bad fisheye_front.ppm\n"); return Sample{}; }
    if (!read_ppm(paths[1], w, h, s.left)  || w != kFishW || h != kFishH) { std::fprintf(stderr, "sample: bad fisheye_left.ppm\n");  return Sample{}; }
    if (!read_ppm(paths[2], w, h, s.right) || w != kFishW || h != kFishH) { std::fprintf(stderr, "sample: bad fisheye_right.ppm\n"); return Sample{}; }
    if (!read_ppm(paths[3], w, h, s.rear)  || w != kFishW || h != kFishH) { std::fprintf(stderr, "sample: bad fisheye_rear.ppm\n");  return Sample{}; }
    if (!read_ppm(paths[4], w, h, s.bev_truth) || w != kBevW || h != kBevH) { std::fprintf(stderr, "sample: bad bev_ground_truth.ppm\n"); return Sample{}; }
    s.loaded = true;
    return s;
}

// ===========================================================================
// max_abs_diff — generic L-infinity comparison, used by every VERIFY
// checkpoint below (uint8 arrays implicitly widen to double).
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
// GATE 1: model_roundtrip — the equidistant fisheye model's self-
// consistency, BYPASSING kernels.cuh entirely (per the twin-independence
// ruling stated in kernels.cuh's file header: this project must carry at
// least one check that does not route through fisheye_project/
// fisheye_unproject). Every formula below is hand-retyped in DOUBLE
// precision, independently of kernels.cuh AND of
// ../scripts/make_synthetic.py's own Python re-derivation.
//
// Unlike 01.01's roundtrip gate (which must iterate a fixed-point
// undistort because Brown-Conrady has no closed-form inverse), the
// equidistant model's inverse IS closed-form — this gate is a direct,
// non-iterative algebraic check, and its expected error is therefore nÂear
// machine precision, not "however many fixed-point iterations converged".
// The theta grid deliberately includes angles PAST 90 degrees (a regime a
// pinhole/rectilinear model cannot even represent — PART 1's file header)
// to prove the model is well-behaved exactly where it matters most.
// ===========================================================================
static double gate_model_roundtrip()
{
    const double F = static_cast<double>(kFishFx);
    const double CX = static_cast<double>(kFishCx), CY = static_cast<double>(kFishCy);
    const double thetas_deg[] = { 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 92.5, 105.0, 120.0, 135.0, 150.0 };
    const double phis_deg[] = { 0.0, 60.0, 120.0, 180.0, 240.0, 300.0 };
    const double pi = 3.14159265358979323846;
    double max_err = 0.0;

    for (double td : thetas_deg) {
        for (double pd : phis_deg) {
            const double theta = td * pi / 180.0;
            const double phi = pd * pi / 180.0;

            // ---- FORWARD (hand-retyped equidistant projection) --------
            const double r = F * theta;
            const double u0 = CX + r * std::cos(phi);
            const double v0 = CY + r * std::sin(phi);

            // ---- INVERSE (hand-retyped, closed-form -- no iteration) ---
            const double du = u0 - CX, dv = v0 - CY;
            const double r2 = std::hypot(du, dv);
            const double theta2 = r2 / F;
            const double phi2 = std::atan2(dv, du);

            // ---- FORWARD again, to compare in PIXEL space (theta==0
            // makes phi ill-defined but harmless, same argument as
            // kernels.cuh's fisheye_project comment -- r==0 either way) --
            const double r3 = F * theta2;
            const double u1 = CX + r3 * std::cos(phi2);
            const double v1 = CY + r3 * std::sin(phi2);

            const double err = std::hypot(u1 - u0, v1 - v0);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// ===========================================================================
// GATES 2 & 3: straightness — find where the boundary edge crosses the
// kBoundaryThreshold threshold along a set of scanlines (HOST-side, from-
// scratch edge detector — bypasses every kernel and every reference_cpu
// function, exercising the ACTUAL pixel content of the rendered images,
// same "measure the real artifact" discipline as 01.01's checkerboard
// detector). Because the boundary is NOT vertical in image space (it is a
// depth-receding line seen by a tilted camera — see main.cu's file header
// comment on kBoundaryY), straightness is measured as the RESIDUAL from a
// least-squares best-fit LINE (col = a*row + b), not deviation from a
// constant mean — a fair test that correctly rewards "straight but
// sloped" and penalizes genuine curvature.
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
        const bool cross = (a > kBoundaryThreshold) != (b > kBoundaryThreshold);
        if (cross) {
            const double t = (static_cast<double>(kBoundaryThreshold) + 0.5 - a) / (static_cast<double>(b) - a);
            out_x = static_cast<double>(x) + t;
            return true;
        }
    }
    return false;
}

// best_fit_line_residual — least-squares fit col = a*row + b over the
// found (row, col) crossing points; returns the MAX absolute residual
// (a strict, easily-explained bound, same spirit as 01.01's max-deviation
// spread). Returns -1.0 (sentinel) if fewer than 3 points were found — too
// few to fit a meaningful line, reported and gated as a hard failure
// (never silently pass on missing data).
static double best_fit_line_residual(const std::vector<std::pair<double,double>>& pts, int* n_out)
{
    if (n_out) *n_out = static_cast<int>(pts.size());
    if (pts.size() < 3) return -1.0;
    double sum_row = 0.0, sum_col = 0.0, sum_rr = 0.0, sum_rc = 0.0;
    for (const auto& p : pts) {
        sum_row += p.first; sum_col += p.second;
        sum_rr += p.first * p.first; sum_rc += p.first * p.second;
    }
    const double n = static_cast<double>(pts.size());
    const double denom = n * sum_rr - sum_row * sum_row;
    if (std::fabs(denom) < 1e-9) return -1.0;   // degenerate (all rows identical) -- cannot fit a slope
    const double a = (n * sum_rc - sum_row * sum_col) / denom;
    const double b = (sum_col - a * sum_row) / n;
    double max_resid = 0.0;
    for (const auto& p : pts) {
        const double resid = std::fabs(p.second - (a * p.first + b));
        if (resid > max_resid) max_resid = resid;
    }
    return max_resid;
}

static double straightness_residual(const std::vector<unsigned char>& rgb, int W, int H,
                                    int yLo, int yHi, int xLo, int xHi, int* n_found_out)
{
    std::vector<std::pair<double,double>> pts;   // (row, col)
    for (int y = yLo; y <= yHi; y += 2) {
        double x;
        if (find_crossing_x(rgb, W, H, y, xLo, xHi, x))
            pts.push_back({ static_cast<double>(y), x });
    }
    return best_fit_line_residual(pts, n_found_out);
}

// ===========================================================================
// bilinear_sample_rgb_gate — a THIRD, independent bilinear sampler (neither
// kernels.cu's device version nor reference_cpu.cpp's host twin), used only
// by the seam_consistency gate below. Deliberately hand-written here so
// that gate exercises its own arithmetic, not a copy of code already
// exercised by VERIFY.
// ===========================================================================
static void bilinear_sample_rgb_gate(const std::vector<unsigned char>& img, int W, int H,
                                     double u, double v, double out[3])
{
    u = std::min(std::max(u, 0.0), static_cast<double>(W - 1));
    v = std::min(std::max(v, 0.0), static_cast<double>(H - 1));
    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const double fx = u - x0, fy = v - y0;
    for (int c = 0; c < 3; ++c) {
        const double v00 = img[(static_cast<size_t>(y0) * W + x0) * 3 + c];
        const double v10 = img[(static_cast<size_t>(y0) * W + x1) * 3 + c];
        const double v01 = img[(static_cast<size_t>(y1) * W + x0) * 3 + c];
        const double v11 = img[(static_cast<size_t>(y1) * W + x1) * 3 + c];
        const double top = v00 + (v10 - v00) * fx;
        const double bot = v01 + (v11 - v01) * fx;
        out[c] = top + (bot - top) * fy;
    }
}

// ===========================================================================
// GATE: seam_consistency — for every BEV pixel seen by EXACTLY 2 cameras
// (coverage popcount == 2), independently sample BOTH cameras' ACTUAL
// fisheye images at their respective rig-projected points (via the shared
// rig_camera_to_bev_sample() -- kernels.cuh's twin-independence exception
// for rig geometry/data -- and this file's OWN bilinear sampler, never
// kernels.cu's or reference_cpu.cpp's) and measure how well the two
// cameras agree on that ground point's color BEFORE blending. Large
// disagreement would mean the rig extrinsics or the shared model are
// internally inconsistent (e.g. a mount position or tilt typo) -- this is
// exactly the kind of bug the twin/VERIFY comparison is blind to, because
// both kernels.cu and reference_cpu.cpp would reproduce the SAME wrong
// geometry from the SAME kernels.cuh constants.
// ===========================================================================
struct SeamResult { double mean_abs_diff; long long n_pairs; };

static SeamResult gate_seam_consistency(const std::vector<unsigned char>& coverage,
                                        const std::vector<unsigned char>& front,
                                        const std::vector<unsigned char>& left,
                                        const std::vector<unsigned char>& right,
                                        const std::vector<unsigned char>& rear)
{
    const std::vector<unsigned char>* imgs[kNumRigCameras] = { &front, &left, &right, &rear };
    double sum_err = 0.0;
    long long n_pairs = 0;

    for (int yo = 0; yo < kBevH; ++yo) {
        for (int xo = 0; xo < kBevW; ++xo) {
            const unsigned char cov = coverage[static_cast<size_t>(yo) * kBevW + xo];
            // popcount == 2: exactly two bits set (a 2-camera overlap pixel).
            int pc = 0;
            for (int b = 0; b < 4; ++b) if (cov & (1 << b)) ++pc;
            if (pc != 2) continue;

            float X, Y;
            bev_pixel_to_ground(xo, yo, X, Y);

            double colors[2][3];
            int found = 0;
            for (int cam = 0; cam < kNumRigCameras && found < 2; ++cam) {
                if (!(cov & (1 << cam))) continue;
                float u, v, weight;
                if (!rig_camera_to_bev_sample(cam, X, Y, u, v, weight)) continue;   // should always succeed here
                bilinear_sample_rgb_gate(*imgs[cam], kFishW, kFishH,
                                         static_cast<double>(u), static_cast<double>(v), colors[found]);
                ++found;
            }
            if (found != 2) continue;   // defensive: coverage bit says 2, but a boundary re-check failed -- skip, do not fabricate

            const double d = (std::fabs(colors[0][0] - colors[1][0])
                             + std::fabs(colors[0][1] - colors[1][1])
                             + std::fabs(colors[0][2] - colors[1][2])) / 3.0;
            sum_err += d;
            ++n_pairs;
        }
    }
    SeamResult r;
    r.mean_abs_diff = n_pairs > 0 ? sum_err / static_cast<double>(n_pairs) : 0.0;
    r.n_pairs = n_pairs;
    return r;
}

// ===========================================================================
// classify_bev_pixel — shared masking logic for the bev_ground_truth and
// flat_ground_assumption gates: is ground point (X,Y) near a tall object's
// footprint (kSceneObjects, padded by kObjectMaskMarginM)?
// ===========================================================================
static bool near_any_object(double X, double Y)
{
    for (const auto& o : kSceneObjects) {
        const double d = std::hypot(X - o.x, Y - o.y);
        if (d <= o.footprint_r + kObjectMaskMarginM) return true;
    }
    return false;
}

// is_seam_pixel — true if any of the 4-neighborhood pixels (within
// kSeamMarginPx) has a DIFFERENT coverage bitmask than (xo,yo) itself --
// a cheap edge-detector on the coverage map, used to exclude camera-
// boundary pixels from the bev_ground_truth gate's "flat" measurement.
static bool is_seam_pixel(const std::vector<unsigned char>& coverage, int xo, int yo)
{
    const unsigned char self = coverage[static_cast<size_t>(yo) * kBevW + xo];
    for (int dy = -kSeamMarginPx; dy <= kSeamMarginPx; ++dy) {
        for (int dx = -kSeamMarginPx; dx <= kSeamMarginPx; ++dx) {
            const int nx = xo + dx, ny = yo + dy;
            if (nx < 0 || nx >= kBevW || ny < 0 || ny >= kBevH) continue;
            if (coverage[static_cast<size_t>(ny) * kBevW + nx] != self) return true;
        }
    }
    return false;
}

// ===========================================================================
// GATE: bev_ground_truth + GATE: flat_ground_assumption — share one sweep
// over the BEV output vs. the committed ground-truth crop
// (bev_ground_truth.ppm — an orthographic top-down render of the ground
// TEXTURE ONLY, no cameras, no objects; see make_synthetic.py's file
// header: "the BEV ground truth IS the source texture -- exact"). Every
// pixel is classified into exactly one bucket:
//   FLAT   — covered (>=1 camera), not near any object, not near a seam.
//            Expected error: SMALL (bev_ground_truth gate).
//   OBJECT — covered, near a tall object's footprint. Expected error:
//            LARGE (flat_ground_assumption gate's negative control).
//   (seam-margin and uncovered pixels are excluded from BOTH buckets --
//    seam blend behavior is seam_consistency's job, and an uncovered
//    pixel has no BEV color to compare at all.)
// ===========================================================================
struct GroundTruthSweep { double flat_mean_err; long long n_flat; double object_mean_err; long long n_object; };

static GroundTruthSweep sweep_ground_truth(const std::vector<unsigned char>& bev,
                                           const std::vector<unsigned char>& truth,
                                           const std::vector<unsigned char>& coverage)
{
    double sum_flat = 0.0, sum_object = 0.0;
    long long n_flat = 0, n_object = 0;
    for (int yo = 0; yo < kBevH; ++yo) {
        for (int xo = 0; xo < kBevW; ++xo) {
            const int idx = yo * kBevW + xo;
            if (coverage[idx] == 0) continue;              // uncovered: nothing to compare
            if (is_seam_pixel(coverage, xo, yo)) continue;  // seam margin: seam_consistency's job

            float X, Y;
            bev_pixel_to_ground(xo, yo, X, Y);
            const double err = (std::fabs(static_cast<double>(bev[idx * 3 + 0]) - truth[idx * 3 + 0])
                               + std::fabs(static_cast<double>(bev[idx * 3 + 1]) - truth[idx * 3 + 1])
                               + std::fabs(static_cast<double>(bev[idx * 3 + 2]) - truth[idx * 3 + 2])) / 3.0;

            if (near_any_object(static_cast<double>(X), static_cast<double>(Y))) {
                sum_object += err; ++n_object;
            } else {
                sum_flat += err; ++n_flat;
            }
        }
    }
    GroundTruthSweep r;
    r.flat_mean_err = n_flat > 0 ? sum_flat / static_cast<double>(n_flat) : 0.0;
    r.n_flat = n_flat;
    r.object_mean_err = n_object > 0 ? sum_object / static_cast<double>(n_object) : 0.0;
    r.n_object = n_object;
    return r;
}

// ===========================================================================
// GATE: coverage — every BEV pixel within kBevDesignRadiusM of the vehicle
// center must be seen by >= 1 camera; also reports the fraction of pixels
// (within the full BEV crop) seen by >= 2 cameras (informational, not
// gated -- overlap amount is a rig-design choice, not a correctness bound).
// ===========================================================================
struct CoverageResult { double covered_fraction; double overlap2_fraction; long long n_in_radius; };

static CoverageResult gate_coverage(const std::vector<unsigned char>& coverage)
{
    long long n_in_radius = 0, n_covered_in_radius = 0;
    long long n_total = 0, n_overlap2 = 0;
    for (int yo = 0; yo < kBevH; ++yo) {
        for (int xo = 0; xo < kBevW; ++xo) {
            const int idx = yo * kBevW + xo;
            float X, Y;
            bev_pixel_to_ground(xo, yo, X, Y);
            const double radius = std::hypot(static_cast<double>(X), static_cast<double>(Y));

            int pc = 0;
            for (int b = 0; b < 4; ++b) if (coverage[idx] & (1 << b)) ++pc;
            ++n_total;
            if (pc >= 2) ++n_overlap2;

            if (radius <= kBevDesignRadiusM) {
                ++n_in_radius;
                if (pc >= 1) ++n_covered_in_radius;
            }
        }
    }
    CoverageResult r;
    r.covered_fraction = n_in_radius > 0 ? static_cast<double>(n_covered_in_radius) / static_cast<double>(n_in_radius) : 0.0;
    r.overlap2_fraction = n_total > 0 ? static_cast<double>(n_overlap2) / static_cast<double>(n_total) : 0.0;
    r.n_in_radius = n_in_radius;
    return r;
}

// ===========================================================================
// Artifact helper: a viewable error-heatmap PGM from the BEV-vs-truth
// per-pixel mean-abs-channel-error, scaled x2 and clamped for visibility
// (DISPLAY-ONLY scaling -- the raw, unscaled numbers are what every gate
// above actually measures and what gates_metrics.csv records).
// ===========================================================================
static std::vector<unsigned char> make_error_heatmap(const std::vector<unsigned char>& bev,
                                                      const std::vector<unsigned char>& truth,
                                                      const std::vector<unsigned char>& coverage)
{
    std::vector<unsigned char> out(static_cast<size_t>(kBevW) * kBevH, 0);
    for (int i = 0; i < kBevW * kBevH; ++i) {
        if (coverage[i] == 0) { out[i] = 0; continue; }
        const double err = (std::fabs(static_cast<double>(bev[i * 3 + 0]) - truth[i * 3 + 0])
                           + std::fabs(static_cast<double>(bev[i * 3 + 1]) - truth[i * 3 + 1])
                           + std::fabs(static_cast<double>(bev[i * 3 + 2]) - truth[i * 3 + 2])) / 3.0;
        const double scaled = std::min(255.0, err * 2.0);
        out[i] = static_cast<unsigned char>(scaled + 0.5);
    }
    return out;
}

// make_coverage_vis — DISPLAY-ONLY rescale of the raw 4-bit coverage
// bitmask (values 0-15, bit i = camera i — see kCamFront..kCamRear) into
// the full 0-255 range so the PGM is actually visible in an image viewer
// (values 0-15 read as near-black otherwise). Multiplying by 17 maps
// 0->0 and 15->255 while keeping every one of the 16 possible bitmask
// values distinct (17*15=255) — gates_metrics.csv and every gate above
// read the RAW (unscaled) coverage buffer; this function only touches the
// artifact written for human eyes.
static std::vector<unsigned char> make_coverage_vis(const std::vector<unsigned char>& coverage)
{
    std::vector<unsigned char> out(coverage.size());
    for (size_t i = 0; i < coverage.size(); ++i)
        out[i] = static_cast<unsigned char>(coverage[i] * 17);
    return out;
}

// ===========================================================================
// gates_metrics.csv writer — same shape as 01.01's.
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

    std::printf("[demo] fisheye unwarp (rectilinear + cylindrical) and 4-camera BEV surround-view stitching (project 01.07)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d fisheye (equidistant, fx=%.1f, %.0fdeg-class FOV) -> rectilinear %dx%d (%.0fdeg half-FOV) "
               "+ cylindrical %dx%d, and a %d-camera rig -> %dx%d BEV over +-%.1fm\n",
               kFishW, kFishH, static_cast<double>(kFishFx), static_cast<double>(kFishFullFovDeg),
               kRectW, kRectH, 45.0, kCylW, kCylH, kNumRigCameras, kBevW, kBevH, static_cast<double>(kBevRangeM));

    // ---- data --------------------------------------------------------------
    Sample sample = load_sample(data_dir, argv[0]);
    if (!sample.loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: 4 synthetic %dx%d fisheye camera renders (front/left/right/rear, ray-cast, 2x2 supersampled) "
               "+ %dx%d ground-truth BEV crop [synthetic, seed 42]\n", kFishW, kFishH, kBevW, kBevH);
    std::printf("[info] straightness-gate boundary edge: world Y=%.1fm, X in [%.1f,%.1f]m (see ../scripts/make_synthetic.py)\n",
               kBoundaryY, kBoundaryX0, kBoundaryX1);

    const int fish_n = kFishW * kFishH;
    const int rect_n = kRectW * kRectH;
    const int cyl_n = kCylW * kCylH;
    const int bev_n = kBevW * kBevH;

    // ======================= device buffers =====================================
    unsigned char *d_front = nullptr, *d_left = nullptr, *d_right = nullptr, *d_rear = nullptr;
    RemapSample *d_rect_lut = nullptr, *d_cyl_lut = nullptr;
    unsigned char *d_rect_out = nullptr, *d_cyl_out = nullptr;
    unsigned char *d_bev = nullptr, *d_coverage = nullptr;

    CUDA_CHECK(cudaMalloc(&d_front, static_cast<size_t>(fish_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_left, static_cast<size_t>(fish_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_right, static_cast<size_t>(fish_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_rear, static_cast<size_t>(fish_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_rect_lut, static_cast<size_t>(rect_n) * sizeof(RemapSample)));
    CUDA_CHECK(cudaMalloc(&d_cyl_lut, static_cast<size_t>(cyl_n) * sizeof(RemapSample)));
    CUDA_CHECK(cudaMalloc(&d_rect_out, static_cast<size_t>(rect_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_cyl_out, static_cast<size_t>(cyl_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_bev, static_cast<size_t>(bev_n) * 3));
    CUDA_CHECK(cudaMalloc(&d_coverage, static_cast<size_t>(bev_n)));

    CUDA_CHECK(cudaMemcpy(d_front, sample.front.data(), sample.front.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_left, sample.left.data(), sample.left.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, sample.right.data(), sample.right.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rear, sample.rear.data(), sample.rear.size(), cudaMemcpyHostToDevice));

    // ======================= HALF 1: single-camera unwarp ========================
    GpuTimer gt_lut; gt_lut.begin();
    launch_build_rect_lut(d_rect_lut);
    launch_build_cyl_lut(d_cyl_lut);
    const float lut_ms = gt_lut.end_ms();

    GpuTimer gt_unwarp; gt_unwarp.begin();
    launch_remap_bilinear(d_front, d_rect_lut, d_rect_out, kFishW, kFishH, kRectW, kRectH);
    launch_remap_bilinear(d_front, d_cyl_lut, d_cyl_out, kFishW, kFishH, kCylW, kCylH);
    const float unwarp_ms = gt_unwarp.end_ms();

    // ======================= HALF 2: 4-camera BEV compositor =====================
    GpuTimer gt_bev; gt_bev.begin();
    launch_bev_compose(d_front, d_left, d_right, d_rear, d_bev, d_coverage);
    const float bev_ms = gt_bev.end_ms();

    std::printf("[time] LUT build (rect+cyl): %.3f ms | unwarp remap (rect+cyl): %.3f ms | BEV compose: %.3f ms\n",
               static_cast<double>(lut_ms), static_cast<double>(unwarp_ms), static_cast<double>(bev_ms));

    // ======================= download everything needed for VERIFY/gates =======
    std::vector<RemapSample> rect_lut_gpu(rect_n), cyl_lut_gpu(cyl_n);
    std::vector<unsigned char> rect_out_gpu(static_cast<size_t>(rect_n) * 3), cyl_out_gpu(static_cast<size_t>(cyl_n) * 3);
    std::vector<unsigned char> bev_gpu(static_cast<size_t>(bev_n) * 3), coverage_gpu(bev_n);

    CUDA_CHECK(cudaMemcpy(rect_lut_gpu.data(), d_rect_lut, static_cast<size_t>(rect_n) * sizeof(RemapSample), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cyl_lut_gpu.data(), d_cyl_lut, static_cast<size_t>(cyl_n) * sizeof(RemapSample), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rect_out_gpu.data(), d_rect_out, rect_out_gpu.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cyl_out_gpu.data(), d_cyl_out, cyl_out_gpu.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bev_gpu.data(), d_bev, bev_gpu.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(coverage_gpu.data(), d_coverage, coverage_gpu.size(), cudaMemcpyDeviceToHost));

    // ======================= VERIFY: GPU vs CPU, every stage ====================
    bool verify_pass = true;
    CpuTimer cpu_timer; cpu_timer.begin();

    std::vector<RemapSample> rect_lut_cpu(rect_n), cyl_lut_cpu(cyl_n);
    build_rect_lut_cpu(rect_lut_cpu.data());
    build_cyl_lut_cpu(cyl_lut_cpu.data());
    const double d_rect_lut_err = max_abs_diff_lut(rect_lut_gpu, rect_lut_cpu);
    const double d_cyl_lut_err = max_abs_diff_lut(cyl_lut_gpu, cyl_lut_cpu);
    std::printf("[info] verify(rect LUT): max|gpu-cpu| = %.7f px (tol %.4f)\n", d_rect_lut_err, kTolLutPx);
    std::printf("[info] verify(cyl LUT): max|gpu-cpu| = %.7f px (tol %.4f)\n", d_cyl_lut_err, kTolLutPx);
    if (d_rect_lut_err > kTolLutPx || d_cyl_lut_err > kTolLutPx) verify_pass = false;

    std::vector<unsigned char> rect_out_cpu(static_cast<size_t>(rect_n) * 3), cyl_out_cpu(static_cast<size_t>(cyl_n) * 3);
    remap_bilinear_cpu(sample.front.data(), rect_lut_cpu.data(), rect_out_cpu.data(), kFishW, kFishH, kRectW, kRectH);
    remap_bilinear_cpu(sample.front.data(), cyl_lut_cpu.data(), cyl_out_cpu.data(), kFishW, kFishH, kCylW, kCylH);
    const double err_rect_out = max_abs_diff(rect_out_gpu, rect_out_cpu);
    const double err_cyl_out = max_abs_diff(cyl_out_gpu, cyl_out_cpu);
    std::printf("[info] verify(rectilinear remap): max|gpu-cpu| = %.4f (tol %.2f)\n", err_rect_out, kTolUint8);
    std::printf("[info] verify(cylindrical remap): max|gpu-cpu| = %.4f (tol %.2f)\n", err_cyl_out, kTolUint8);
    if (err_rect_out > kTolUint8 || err_cyl_out > kTolUint8) verify_pass = false;

    std::vector<unsigned char> bev_cpu(static_cast<size_t>(bev_n) * 3), coverage_cpu(bev_n);
    bev_compose_cpu(sample.front.data(), sample.left.data(), sample.right.data(), sample.rear.data(),
                    bev_cpu.data(), coverage_cpu.data());
    const double err_bev = max_abs_diff(bev_gpu, bev_cpu);
    const double err_cov = max_abs_diff(coverage_gpu, coverage_cpu);
    std::printf("[info] verify(BEV compose): max|gpu-cpu| = %.4f (tol %.2f)\n", err_bev, kTolUint8);
    std::printf("[info] verify(BEV coverage bitmask): max|gpu-cpu| = %.4f (tol %.2f)\n", err_cov, kTolCoverage);
    if (err_bev > kTolUint8 || err_cov > kTolCoverage) verify_pass = false;

    const double cpu_ms = cpu_timer.end_ms();
    std::printf("[time] full CPU oracle (all stages): %.1f ms\n", cpu_ms);
    std::printf("VERIFY: %s (GPU matches CPU reference within documented per-stage tolerance: "
               "rect LUT, cyl LUT, rectilinear remap, cylindrical remap, BEV compose, BEV coverage)\n",
               verify_pass ? "PASS" : "FAIL");

    // ======================= GATES ================================================
    std::vector<CsvRow> csv;

    // -- Gate 1: model_roundtrip ----------------------------------------------
    const double roundtrip_err = gate_model_roundtrip();
    const bool gate1 = roundtrip_err <= kRoundtripTolPx;
    std::printf("GATE model_roundtrip: %s\n", gate1 ? "PASS" : "FAIL");
    std::printf("[info] model_roundtrip: max pixel error over a theta(incl. >90deg)xphi grid = %.7f px (tol %.4f)\n",
               roundtrip_err, kRoundtripTolPx);
    csv.push_back({ "model_roundtrip", "max_error_px", fmt(roundtrip_err, 7), fmt(kRoundtripTolPx, 4), gate1 ? "PASS" : "FAIL" });

    // -- Gates 2 & 3: straightness ----------------------------------------------
    int n_found_rect = 0, n_found_fish = 0;
    const double resid_rect = straightness_residual(rect_out_gpu, kRectW, kRectH,
                                                     kRectSearchYLo, kRectSearchYHi, kRectSearchXLo, kRectSearchXHi, &n_found_rect);
    const double resid_fish = straightness_residual(sample.front, kFishW, kFishH,
                                                     kFishSearchYLo, kFishSearchYHi, kFishSearchXLo, kFishSearchXHi, &n_found_fish);
    const bool gate2 = (resid_rect >= 0.0) && (resid_rect <= kStraightRectTolPx);
    const bool gate3 = (resid_fish >= 0.0) && (resid_fish >= kStraightFisheyeMinPx);
    std::printf("GATE straightness_rectilinear: %s\n", gate2 ? "PASS" : "FAIL");
    std::printf("[info] straightness_rectilinear: best-fit-line residual = %.4f px over %d rows (tol <= %.2f)\n",
               resid_rect, n_found_rect, kStraightRectTolPx);
    std::printf("GATE distortion_negative_control: %s\n", gate3 ? "PASS" : "FAIL");
    std::printf("[info] distortion_negative_control: RAW fisheye best-fit-line residual = %.4f px over %d rows "
               "(must be >= %.2f -- proves the curvature being corrected is real)\n",
               resid_fish, n_found_fish, kStraightFisheyeMinPx);
    csv.push_back({ "straightness_rectilinear", "best_fit_residual_px", fmt(resid_rect, 4), fmt(kStraightRectTolPx, 2), gate2 ? "PASS" : "FAIL" });
    csv.push_back({ "distortion_negative_control", "best_fit_residual_px", fmt(resid_fish, 4), fmt(kStraightFisheyeMinPx, 2), gate3 ? "PASS" : "FAIL" });

    // -- Gate: bev_ground_truth + flat_ground_assumption (shared sweep) --------
    const GroundTruthSweep sweep = sweep_ground_truth(bev_gpu, sample.bev_truth, coverage_gpu);
    const bool gate4 = sweep.flat_mean_err <= kBevGroundTruthTolMean;
    std::printf("GATE bev_ground_truth: %s\n", gate4 ? "PASS" : "FAIL");
    std::printf("[info] bev_ground_truth: flat-region mean|err| = %.4f (tol %.2f, n=%lld)\n",
               sweep.flat_mean_err, kBevGroundTruthTolMean, sweep.n_flat);
    csv.push_back({ "bev_ground_truth", "flat_region_mean_abs_err", fmt(sweep.flat_mean_err, 4), fmt(kBevGroundTruthTolMean, 2), gate4 ? "PASS" : "FAIL" });

    const bool gate5 = sweep.object_mean_err >= kFlatGroundMinMeanErr;
    const double error_ratio = sweep.flat_mean_err > 1e-6 ? sweep.object_mean_err / sweep.flat_mean_err : 0.0;
    std::printf("GATE flat_ground_assumption: %s\n", gate5 ? "PASS" : "FAIL");
    std::printf("[info] flat_ground_assumption: object-region mean|err| = %.4f (must be >= %.2f -- negative control "
               "proving the flat-ground assumption really breaks near tall objects, n=%lld) | ratio object/flat = %.2fx\n",
               sweep.object_mean_err, kFlatGroundMinMeanErr, sweep.n_object, error_ratio);
    csv.push_back({ "flat_ground_assumption", "object_region_mean_abs_err", fmt(sweep.object_mean_err, 4), fmt(kFlatGroundMinMeanErr, 2), gate5 ? "PASS" : "FAIL" });
    csv.push_back({ "flat_ground_assumption", "error_ratio_object_over_flat", fmt(error_ratio, 2), "n/a", "n/a" });

    // -- Gate: seam_consistency -------------------------------------------------
    const SeamResult seam = gate_seam_consistency(coverage_gpu, sample.front, sample.left, sample.right, sample.rear);
    const bool gate6 = seam.mean_abs_diff <= kSeamConsistencyTolMean;
    std::printf("GATE seam_consistency: %s\n", gate6 ? "PASS" : "FAIL");
    std::printf("[info] seam_consistency: mean|camA-camB| in 2-camera overlap pixels = %.4f (tol %.2f, n_pixel_pairs=%lld)\n",
               seam.mean_abs_diff, kSeamConsistencyTolMean, seam.n_pairs);
    csv.push_back({ "seam_consistency", "mean_abs_diff_overlap", fmt(seam.mean_abs_diff, 4), fmt(kSeamConsistencyTolMean, 2), gate6 ? "PASS" : "FAIL" });

    // -- Gate: coverage -----------------------------------------------------------
    const CoverageResult cov = gate_coverage(coverage_gpu);
    const bool gate7 = cov.covered_fraction >= kCoverageMinFraction;
    std::printf("GATE coverage: %s\n", gate7 ? "PASS" : "FAIL");
    std::printf("[info] coverage: %.3f%% of pixels within the %.1fm design radius have >=1 camera (tol >= %.3f%%, n=%lld) "
               "| >=2-camera overlap = %.2f%% of the full BEV crop\n",
               100.0 * cov.covered_fraction, kBevDesignRadiusM, 100.0 * kCoverageMinFraction, cov.n_in_radius,
               100.0 * cov.overlap2_fraction);
    csv.push_back({ "coverage", "covered_fraction_in_design_radius", fmt(cov.covered_fraction, 5), fmt(kCoverageMinFraction, 5), gate7 ? "PASS" : "FAIL" });
    csv.push_back({ "coverage", "overlap2_fraction_reported_only", fmt(cov.overlap2_fraction, 5), "n/a", "n/a" });

    // ======================= ARTIFACTS ============================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    const std::vector<unsigned char> heatmap = make_error_heatmap(bev_gpu, sample.bev_truth, coverage_gpu);
    bool artifact_ok = !out_dir.empty();
    artifact_ok = artifact_ok
        && write_ppm(out_dir + "/fisheye_front.ppm", kFishW, kFishH, sample.front)
        && write_ppm(out_dir + "/rectilinear.ppm", kRectW, kRectH, rect_out_gpu)
        && write_ppm(out_dir + "/cylindrical.ppm", kCylW, kCylH, cyl_out_gpu)
        && write_ppm(out_dir + "/bev.ppm", kBevW, kBevH, bev_gpu)
        && write_pgm(out_dir + "/coverage_map.pgm", kBevW, kBevH, make_coverage_vis(coverage_gpu))
        && write_pgm(out_dir + "/error_heatmap.pgm", kBevW, kBevH, heatmap)
        && write_gates_csv(out_dir + "/gates_metrics.csv", csv);
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{fisheye_front.ppm, rectilinear.ppm, cylindrical.ppm, bev.ppm, "
                   "coverage_map.pgm, error_heatmap.pgm, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");

    // ---- cleanup ------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_front));  CUDA_CHECK(cudaFree(d_left));
    CUDA_CHECK(cudaFree(d_right));  CUDA_CHECK(cudaFree(d_rear));
    CUDA_CHECK(cudaFree(d_rect_lut)); CUDA_CHECK(cudaFree(d_cyl_lut));
    CUDA_CHECK(cudaFree(d_rect_out)); CUDA_CHECK(cudaFree(d_cyl_out));
    CUDA_CHECK(cudaFree(d_bev)); CUDA_CHECK(cudaFree(d_coverage));

    // ---- verdict --------------------------------------------------------------
    const bool success = verify_pass && gate1 && gate2 && gate3 && gate4 && gate5 && gate6 && gate7 && artifact_ok;
    if (success) {
        std::printf("RESULT: PASS (VERIFY + all 7 gates passed: model_roundtrip, straightness_rectilinear, "
                   "distortion_negative_control, bev_ground_truth, flat_ground_assumption, seam_consistency, coverage)\n");
    } else {
        std::printf("RESULT: FAIL (VERIFY or a gate above did not pass -- see GATE:/VERIFY:/[info] lines)\n");
    }
    return success ? 0 : 1;
}
