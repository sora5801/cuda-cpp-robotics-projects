// ===========================================================================
// main.cu — entry point for project 01.24
//           Transparent/reflective object detection via polarization imaging
//
// What this program does, start to finish
// -----------------------------------------
//   Load the committed DoFP mosaic (mosaic.pgm: matte background + glass
//   pane + glass dome + brushed metal bar — scripts/make_synthetic.py's
//   physics forward model) and run the FIVE-STAGE pipeline kernels.cuh
//   documents on BOTH the GPU (kernels.cu) and an independent CPU oracle
//   (reference_cpu.cpp): demosaic -> Stokes -> DoLP/AoLP -> Malus residual
//   -> {threshold, morphological open, connected-component label, size
//   filter} run TWICE — once on DoLP, once on an intensity-contrast signal
//   built from S0 — to demonstrate this project's whole reason to exist:
//   the glass objects are built with near-zero INTENSITY contrast, so only
//   the polarization channel can find them (both directions asserted below,
//   GATE detection).
//
//   Every stage is VERIFIED gpu-vs-cpu (tight float tolerance for the
//   continuous stages; BIT-EXACT for the detection masks, fed the SAME
//   already-agreeing float signal on both sides — the 01.22 "VERIFY
//   isolation" pattern, this file's header restates it at the call site).
//
//   Independent of that twin comparison, SIX physics/detection GATEs never
//   route through either twin: stokes_accuracy and malus_consistency check
//   internal/ground-truth self-consistency; fresnel_anchor is THE physics
//   gate — main.cu's OWN closed-form Fresnel prediction (kernels.cuh's
//   fresnel_dolp(), called here in C++) against the MEASURED DoLP on the
//   rendered glass pane, closing the loop with scripts/make_synthetic.py's
//   INDEPENDENTLY-CODED Python version of the identical equations;
//   detection checks recall in both directions (DoLP finds the objects,
//   intensity misses the glass); brewster_sweep is a pure closed-form sweep
//   (no rendering at all) checking the peak lands near Brewster's angle;
//   negative_control re-runs the whole pipeline on a matte-only scene and
//   demands zero detections.
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
#include <cstdint>
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
// VERIFY tolerances (GPU vs CPU) and GATE thresholds. Every GATE number
// below was MEASURED with an independent Python (double-precision) re-
// implementation of the exact same pipeline during this project's build
// (the discipline 01.01/01.09/01.11/01.22 all follow: margin OUTWARD from a
// real measurement, never set AT it) — this program's own [info] lines
// reproduce the measurement on whatever machine it actually runs on.
// ===========================================================================

// -- VERIFY: GPU-vs-CPU tolerances (max |gpu-cpu|, per stage). All five
// continuous stages are simple closed-form pointwise formulas (no
// iteration, no reduction) — float32 GPU vs float32 CPU should agree to a
// few ULPs; these tolerances are generous multiples of that expectation.
static constexpr double kTolDemosaic     = 1.0e-2;   // DN
static constexpr double kTolStokes       = 1.0e-2;   // DN (s0/s1/s2, worst of the three)
static constexpr double kTolDolp         = 1.0e-3;   // unitless [0,1]
static constexpr double kTolAolpRad      = 1.0e-3;   // radians (circular distance)
static constexpr double kTolMalusResidual = 1.0e-2;  // DN

// -- GATE thresholds (measured via a from-scratch Python re-implementation
// of this exact pipeline against the committed sample; see this file's
// header and README "Expected output" for the measured numbers this
// program's own run reproduces).
static constexpr double kMaxDolpMaeInterior     = 0.05;   // measured ~0.014
static constexpr double kMaxAolpCircErrDeg      = 5.0;    // measured ~0.82 deg
static constexpr double kMaxMalusResidualMeanAbs = 6.0;   // DN; measured ~3.79
static constexpr double kFresnelAnchorTolDolp   = 0.02;   // measured |diff| ~0.0003
static constexpr double kMinGlassRecallDolp     = 0.85;   // measured ~0.970
static constexpr double kMinMetalRecallDolp     = 0.65;   // measured ~0.820
static constexpr double kMaxGlassRecallIntensity = 0.05;  // measured 0.000 -- THE reason-to-exist assertion
static constexpr double kBrewsterPeakTolDeg     = 2.0;    // sweep step is 1 deg; true peak error ~0.31 deg

// ===========================================================================
// Minimal, STRICT PGM (P5) reader/writer — the 01.01/01.09/01.11/01.22
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

static void write_pgm_from_float(const std::string& path, const std::vector<float>& vals, int W, int H)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return;
    out << "P5\n" << W << " " << H << "\n255\n";
    std::vector<unsigned char> buf(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        float v = vals[i];
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        buf[i] = static_cast<unsigned char>(v + 0.5f);
    }
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

static void write_ppm(const std::string& path, const std::vector<unsigned char>& rgb, int W, int H)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return;
    out << "P6\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

// ===========================================================================
// truth_maps.csv reader — scripts/make_synthetic.py's ground truth: one row
// per pixel, header "x,y,s0_dn,dolp,aolp_deg,label", written in row-major
// (y outer, x inner) order so sequential reads land at index i=y*W+x
// directly (this function double-checks the (x,y) fields anyway — cheap
// insurance against a future generator reordering).
// ===========================================================================
struct TruthMaps {
    std::vector<float> s0, dolp, aolp_deg;
    std::vector<int> label;
};
static bool read_truth_maps_csv(const std::string& path, int W, int H, TruthMaps& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    // Skip '#' comment lines, then the header row.
    while (std::getline(in, line)) { if (!line.empty() && line[0] != '#') break; }
    if (line.rfind("x,y,", 0) != 0) return false;   // sanity: header shape

    const int n = W * H;
    out.s0.assign(n, 0.0f); out.dolp.assign(n, 0.0f); out.aolp_deg.assign(n, 0.0f); out.label.assign(n, 0);
    int count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ','); const int x = std::atoi(tok.c_str());
        std::getline(ss, tok, ','); const int y = std::atoi(tok.c_str());
        std::getline(ss, tok, ','); const float s0 = std::strtof(tok.c_str(), nullptr);
        std::getline(ss, tok, ','); const float dolp = std::strtof(tok.c_str(), nullptr);
        std::getline(ss, tok, ','); const float aolp = std::strtof(tok.c_str(), nullptr);
        std::getline(ss, tok, ','); const int label = std::atoi(tok.c_str());
        if (x < 0 || x >= W || y < 0 || y >= H) return false;
        const int i = y * W + x;
        out.s0[i] = s0; out.dolp[i] = dolp; out.aolp_deg[i] = aolp; out.label[i] = label;
        ++count;
    }
    return count == n;
}

// ===========================================================================
// Small host-side numeric helpers used by the gates below.
// ===========================================================================
static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}
// circular_diff_rad_to_deg — the AoLP wrap-aware distance between two
// angles that both live in [0, pi) (THEORY.md "Numerical considerations"
// teaches this): 179 deg and 1 deg are only 2 deg apart, not 178.
static double circular_diff_rad_to_deg(float a_rad, float b_rad)
{
    double d = std::fabs(static_cast<double>(a_rad) - static_cast<double>(b_rad));
    const double pi = static_cast<double>(kPi);
    if (d > pi / 2.0) d = pi - d;   // wrap: the two representations of "the same axis" are pi apart, not 2*pi
    return d * (180.0 / pi);
}
static double mean_of(const std::vector<float>& v)
{
    double acc = 0.0;
    for (float x : v) acc += static_cast<double>(x);
    return v.empty() ? 0.0 : acc / static_cast<double>(v.size());
}
// is_interior_object_pixel — kernels.cuh's kInteriorMarginPx-eroded object
// footprint test (rects shrunk by the margin on every side; the dome's
// disk shrunk in radius) — used by GATE stokes_accuracy to exclude the
// demosaic-edge-blur ring documented in kernels.cu's demosaic kernel header.
static bool is_interior_object_pixel(int x, int y, int label)
{
    const int m = kInteriorMarginPx;
    if (label == kLabelPane) {
        return x >= kPaneRect.x0 + m && x < kPaneRect.x1 - m && y >= kPaneRect.y0 + m && y < kPaneRect.y1 - m;
    }
    if (label == kLabelMetal) {
        return x >= kMetalRect.x0 + m && x < kMetalRect.x1 - m && y >= kMetalRect.y0 + m && y < kMetalRect.y1 - m;
    }
    if (label == kLabelDome) {
        const float dx = static_cast<float>(x) - kDomeCx, dy = static_cast<float>(y) - kDomeCy;
        return std::sqrt(dx * dx + dy * dy) <= (kDomeRadiusPx - static_cast<float>(m));
    }
    return false;
}

// ===========================================================================
// hsv_to_rgb — textbook HSV->RGB conversion (h in [0,360), s,v in [0,1]),
// used ONLY by the AoLP visualization artifact below. Standard 6-sector
// formula; not part of the pipeline under test (a display convenience, the
// same judgment call 01.21's flow_to_rgb() makes for its own visualization).
// ===========================================================================
static void hsv_to_rgb(float h, float s, float v, unsigned char& r, unsigned char& g, unsigned char& b)
{
    h = std::fmod(h, 360.0f); if (h < 0.0f) h += 360.0f;
    const float c = v * s;
    const float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    const float m = v - c;
    float rp, gp, bp;
    if      (h <  60.0f) { rp = c; gp = x; bp = 0.0f; }
    else if (h < 120.0f) { rp = x; gp = c; bp = 0.0f; }
    else if (h < 180.0f) { rp = 0.0f; gp = c; bp = x; }
    else if (h < 240.0f) { rp = 0.0f; gp = x; bp = c; }
    else if (h < 300.0f) { rp = x; gp = 0.0f; bp = c; }
    else                 { rp = c; gp = 0.0f; bp = x; }
    auto to255 = [](float f) { f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f); return static_cast<unsigned char>(f * 255.0f + 0.5f); };
    r = to255(rp + m); g = to255(gp + m); b = to255(bp + m);
}

// ===========================================================================
// gates_metrics.csv writer (the 01.01/01.09/01.11/01.22 shape).
// ===========================================================================
struct CsvRow { std::string gate, metric, value, tol, pass; };
static std::string fmt(double v, int prec = 4) { char buf[64]; std::snprintf(buf, sizeof(buf), "%.*f", prec, v); return std::string(buf); }
static void write_gates_csv(const std::string& path, const std::vector<CsvRow>& rows)
{
    std::ofstream out(path);
    if (!out.is_open()) return;
    out << "gate,metric,value,tolerance,pass\n";
    for (const auto& r : rows) out << r.gate << "," << r.metric << "," << r.value << "," << r.tol << "," << r.pass << "\n";
}

// ===========================================================================
// PipelineResult — the five continuous-stage outputs (host arrays), plus
// the input mosaic they came from. run_pipeline_gpu/run_pipeline_cpu below
// share this shape so main() can VERIFY field-by-field.
// ===========================================================================
struct PipelineResult {
    std::vector<float> channels4;   // [n*4] interleaved I0,I45,I90,I135
    std::vector<float> s0, s1, s2;  // [n] each
    std::vector<float> dolp, aolp_rad; // [n] each
    std::vector<float> residual;    // [n] the Malus self-consistency residual
};

// run_pipeline_gpu — allocate device buffers, run the four continuous-stage
// kernels (demosaic -> stokes -> dolp/aolp -> malus residual), copy every
// result back to host, free device memory. Self-contained so main() can
// call it twice (main scene, negative control) with zero shared device
// state between calls — a small teaching cost (one extra H2D upload per
// call) in exchange for a much simpler main() (the same tradeoff 01.22's
// per-method cudaMalloc blocks make).
static void run_pipeline_gpu(const std::vector<float>& h_mosaic, int W, int H, PipelineResult& out, double& ms_out)
{
    const int n = W * H;
    float* d_mosaic = nullptr;
    float* d_channels4 = nullptr;
    float *d_s0 = nullptr, *d_s1 = nullptr, *d_s2 = nullptr;
    float *d_dolp = nullptr, *d_aolp = nullptr, *d_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mosaic, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_channels4, static_cast<size_t>(n) * kNumChannels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s0, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s1, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s2, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dolp, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aolp, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mosaic, h_mosaic.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer gt;
    gt.begin();
    launch_demosaic_polarization(d_mosaic, d_channels4, W, H);
    launch_stokes(d_channels4, d_s0, d_s1, d_s2, n);
    launch_dolp_aolp(d_s0, d_s1, d_s2, d_dolp, d_aolp, n);
    launch_malus_residual(d_channels4, d_residual, n);
    ms_out = static_cast<double>(gt.end_ms());

    out.channels4.resize(static_cast<size_t>(n) * kNumChannels);
    out.s0.resize(n); out.s1.resize(n); out.s2.resize(n);
    out.dolp.resize(n); out.aolp_rad.resize(n); out.residual.resize(n);
    CUDA_CHECK(cudaMemcpy(out.channels4.data(), d_channels4, out.channels4.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.s0.data(), d_s0, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.s1.data(), d_s1, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.s2.data(), d_s2, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.dolp.data(), d_dolp, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.aolp_rad.data(), d_aolp, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.residual.data(), d_residual, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_mosaic)); CUDA_CHECK(cudaFree(d_channels4));
    CUDA_CHECK(cudaFree(d_s0)); CUDA_CHECK(cudaFree(d_s1)); CUDA_CHECK(cudaFree(d_s2));
    CUDA_CHECK(cudaFree(d_dolp)); CUDA_CHECK(cudaFree(d_aolp)); CUDA_CHECK(cudaFree(d_residual));
}

// run_pipeline_cpu — the independent CPU oracle twin, same four stages,
// reference_cpu.cpp's functions.
static void run_pipeline_cpu(const std::vector<float>& h_mosaic, int W, int H, PipelineResult& out, double& ms_out)
{
    const int n = W * H;
    out.channels4.resize(static_cast<size_t>(n) * kNumChannels);
    out.s0.resize(n); out.s1.resize(n); out.s2.resize(n);
    out.dolp.resize(n); out.aolp_rad.resize(n); out.residual.resize(n);

    CpuTimer ct;
    ct.begin();
    demosaic_polarization_cpu(h_mosaic.data(), out.channels4.data(), W, H);
    stokes_cpu(out.channels4.data(), out.s0.data(), out.s1.data(), out.s2.data(), n);
    dolp_aolp_cpu(out.s0.data(), out.s1.data(), out.s2.data(), out.dolp.data(), out.aolp_rad.data(), n);
    malus_residual_cpu(out.channels4.data(), out.residual.data(), n);
    ms_out = ct.end_ms();
}

// ===========================================================================
// Detection pipeline — GPU and CPU, run on a caller-supplied HOST signal
// array (dolp or intensity-contrast) so both sides start from IDENTICAL
// bytes (the VERIFY-isolation note in reference_cpu.cpp's header: this is
// what lets VERIFY(detection_*) demand bit-exact agreement).
// ===========================================================================
static void run_detection_gpu(const std::vector<float>& h_signal, float thresh, int W, int H,
                              std::vector<unsigned char>& h_mask_out, int& sweeps_out)
{
    const int n = W * H;
    float* d_signal = nullptr;
    uint8_t* d_mask = nullptr;
    int* d_label = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_label, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_signal, h_signal.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));

    launch_threshold(d_signal, thresh, d_mask, n);
    launch_morphological_open(d_mask, W, H);
    sweeps_out = launch_connected_components(d_mask, d_label, W, H);
    uint8_t* d_mask_final = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mask_final, static_cast<size_t>(n)));
    launch_component_size_filter(d_mask, d_label, kMinComponentSizePx, d_mask_final, n);

    h_mask_out.resize(n);
    CUDA_CHECK(cudaMemcpy(h_mask_out.data(), d_mask_final, static_cast<size_t>(n), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_signal)); CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_label)); CUDA_CHECK(cudaFree(d_mask_final));
}
static void run_detection_cpu(const std::vector<float>& h_signal, float thresh, int W, int H,
                              std::vector<unsigned char>& h_mask_out)
{
    const int n = W * H;
    h_mask_out.assign(n, 0);
    threshold_cpu(h_signal.data(), thresh, h_mask_out.data(), n);
    morphological_open_cpu(h_mask_out.data(), W, H);
    std::vector<int> label(n, -1);
    connected_components_cpu(h_mask_out.data(), label.data(), W, H);
    std::vector<unsigned char> filtered(n, 0);
    component_size_filter_cpu(h_mask_out.data(), label.data(), kMinComponentSizePx, filtered.data(), n);
    h_mask_out = filtered;
}

// recall_over_label — fraction of pixels whose truth label is IN `labels`
// (a small fixed set, e.g. {pane,dome} for "glass") that survive into
// `mask`. The detection GATE's core measurement, both directions.
static double recall_over_labels(const std::vector<unsigned char>& mask, const std::vector<int>& truth_label,
                                 std::initializer_list<int> labels)
{
    long long total = 0, hit = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
        bool match = false;
        for (int l : labels) if (truth_label[i] == l) { match = true; break; }
        if (!match) continue;
        ++total;
        if (mask[i]) ++hit;
    }
    return total > 0 ? static_cast<double>(hit) / static_cast<double>(total) : 0.0;
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

    std::printf("[demo] transparent/reflective object detection via polarization imaging (project 01.24)\n");
    print_device_info();
    std::printf("PROBLEM: %dx%d DoFP polarization mosaic (4 phases: 0/45/90/135 deg), matte background + "
               "glass pane + glass dome + brushed metal bar, FP32\n", kW, kH);

    // ---- data -----------------------------------------------------------
    const std::string mosaic_path = find_data_file(data_dir, argv[0], "mosaic.pgm");
    const std::string negctrl_path = find_data_file(data_dir, argv[0], "mosaic_negctrl.pgm");
    const std::string truth_path = find_data_file(data_dir, argv[0], "truth_maps.csv");

    int w = 0, h = 0, wn = 0, hn = 0;
    std::vector<unsigned char> mosaic_u8, negctrl_u8;
    TruthMaps truth;
    bool loaded = !mosaic_path.empty() && !negctrl_path.empty() && !truth_path.empty()
               && read_pgm(mosaic_path, w, h, mosaic_u8) && w == kW && h == kH
               && read_pgm(negctrl_path, wn, hn, negctrl_u8) && wn == kW && hn == kH
               && read_truth_maps_csv(truth_path, kW, kH, truth);
    if (!loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: synthetic DoFP capture (flat/gradient matte background residual DoLP=%.3f; glass pane "
               "at %.1f deg incidence; curved glass dome (Fresnel per-pixel, Brewster ring); brushed metal "
               "bar (saturating DoLP curve)); negative-control (background-only) scene; per-pixel ground "
               "truth [synthetic, seed 42]\n", static_cast<double>(kBgDolp), static_cast<double>(kPaneThetaDeg));

    std::vector<float> h_mosaic(kN), h_negctrl(kN);
    for (int i = 0; i < kN; ++i) { h_mosaic[i] = static_cast<float>(mosaic_u8[i]); h_negctrl[i] = static_cast<float>(negctrl_u8[i]); }

    // ---- GPU + CPU pipelines on the main scene ---------------------------
    PipelineResult gpu, cpu;
    double ms_gpu = 0.0, ms_cpu = 0.0;
    run_pipeline_gpu(h_mosaic, kW, kH, gpu, ms_gpu);
    run_pipeline_cpu(h_mosaic, kW, kH, cpu, ms_cpu);
    std::printf("[time] GPU pipeline (demosaic+stokes+dolp/aolp+malus_residual): %.3f ms | "
               "CPU reference (single-thread): %.2f ms\n", ms_gpu, ms_cpu);

    // ---- VERIFY: GPU vs CPU, per stage -----------------------------------
    const double diff_demosaic = max_abs_diff(gpu.channels4, cpu.channels4);
    const double diff_s0 = max_abs_diff(gpu.s0, cpu.s0);
    const double diff_s1 = max_abs_diff(gpu.s1, cpu.s1);
    const double diff_s2 = max_abs_diff(gpu.s2, cpu.s2);
    const double diff_stokes = std::max({ diff_s0, diff_s1, diff_s2 });
    const double diff_dolp = max_abs_diff(gpu.dolp, cpu.dolp);
    double diff_aolp_deg = 0.0;
    for (int i = 0; i < kN; ++i) diff_aolp_deg = std::max(diff_aolp_deg, circular_diff_rad_to_deg(gpu.aolp_rad[i], cpu.aolp_rad[i]));
    const double diff_aolp_rad = diff_aolp_deg * (static_cast<double>(kPi) / 180.0);
    const double diff_residual = max_abs_diff(gpu.residual, cpu.residual);

    const bool verify_demosaic = diff_demosaic <= kTolDemosaic;
    const bool verify_stokes = diff_stokes <= kTolStokes;
    const bool verify_dolp = diff_dolp <= kTolDolp;
    const bool verify_aolp = diff_aolp_rad <= kTolAolpRad;
    const bool verify_residual = diff_residual <= kTolMalusResidual;

    std::printf("[info] verify(demosaic): max|gpu-cpu|=%.5f DN (tol %.3f)\n", diff_demosaic, kTolDemosaic);
    std::printf("VERIFY(demosaic): %s (GPU phase-bilinear reconstruction matches independent CPU twin)\n", verify_demosaic ? "PASS" : "FAIL");
    std::printf("[info] verify(stokes): max|gpu-cpu|=%.5f DN (s0=%.5f s1=%.5f s2=%.5f, tol %.3f)\n", diff_stokes, diff_s0, diff_s1, diff_s2, kTolStokes);
    std::printf("VERIFY(stokes): %s (GPU Stokes formulas match independent CPU twin)\n", verify_stokes ? "PASS" : "FAIL");
    std::printf("[info] verify(dolp): max|gpu-cpu|=%.6f (tol %.4f)\n", diff_dolp, kTolDolp);
    std::printf("VERIFY(dolp): %s (GPU DoLP formula matches independent CPU twin)\n", verify_dolp ? "PASS" : "FAIL");
    std::printf("[info] verify(aolp): max circular diff=%.5f deg (tol %.5f deg)\n", diff_aolp_deg, kTolAolpRad * (180.0 / static_cast<double>(kPi)));
    std::printf("VERIFY(aolp): %s (GPU AoLP formula, half-angle wrap included, matches independent CPU twin)\n", verify_aolp ? "PASS" : "FAIL");
    std::printf("[info] verify(malus_residual): max|gpu-cpu|=%.5f DN (tol %.3f)\n", diff_residual, kTolMalusResidual);
    std::printf("VERIFY(malus_residual): %s (GPU self-consistency residual matches independent CPU twin)\n", verify_residual ? "PASS" : "FAIL");

    // ---- detection: DoLP signal, and intensity-contrast signal -----------
    const double mean_s0 = mean_of(gpu.s0);
    std::vector<float> intensity_signal(kN);
    {
        // Built on the GPU (teaching completeness — this is the one stage
        // whose signal is not one of the four continuous-stage outputs
        // above), then copied back for the (bit-exact) CPU detection twin.
        float* d_s0 = nullptr; float* d_intensity = nullptr;
        CUDA_CHECK(cudaMalloc(&d_s0, static_cast<size_t>(kN) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_intensity, static_cast<size_t>(kN) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_s0, gpu.s0.data(), static_cast<size_t>(kN) * sizeof(float), cudaMemcpyHostToDevice));
        launch_abs_diff_scalar(d_s0, static_cast<float>(mean_s0), d_intensity, kN);
        CUDA_CHECK(cudaMemcpy(intensity_signal.data(), d_intensity, static_cast<size_t>(kN) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_s0)); CUDA_CHECK(cudaFree(d_intensity));
    }

    std::vector<unsigned char> mask_dolp_gpu, mask_dolp_cpu, mask_int_gpu, mask_int_cpu;
    int sweeps_dolp = 0, sweeps_int = 0;
    run_detection_gpu(gpu.dolp, kDolpThreshold, kW, kH, mask_dolp_gpu, sweeps_dolp);
    run_detection_cpu(gpu.dolp, kDolpThreshold, kW, kH, mask_dolp_cpu);   // SAME gpu.dolp array feeds both sides
    run_detection_gpu(intensity_signal, kIntensityThreshold, kW, kH, mask_int_gpu, sweeps_int);
    run_detection_cpu(intensity_signal, kIntensityThreshold, kW, kH, mask_int_cpu);

    long long mismatch_dolp = 0, mismatch_int = 0;
    for (int i = 0; i < kN; ++i) { if (mask_dolp_gpu[i] != mask_dolp_cpu[i]) ++mismatch_dolp; if (mask_int_gpu[i] != mask_int_cpu[i]) ++mismatch_int; }
    const bool verify_det_dolp = (mismatch_dolp == 0);
    const bool verify_det_int = (mismatch_int == 0);
    std::printf("[info] verify(detection_dolp): %lld/%d mismatched pixels (CCL converged in %d sweeps)\n", mismatch_dolp, kN, sweeps_dolp);
    std::printf("VERIFY(detection_dolp): %s (GPU threshold+morph+CCL+filter bit-exact vs. independent CPU twin, same input)\n", verify_det_dolp ? "PASS" : "FAIL");
    std::printf("[info] verify(detection_intensity): %lld/%d mismatched pixels (CCL converged in %d sweeps)\n", mismatch_int, kN, sweeps_int);
    std::printf("VERIFY(detection_intensity): %s (GPU threshold+morph+CCL+filter bit-exact vs. independent CPU twin, same input)\n", verify_det_int ? "PASS" : "FAIL");

    // ---- GATE stokes_accuracy: DoLP MAE + AoLP circular MAE vs ground
    // truth, restricted to the eroded object INTERIOR (kInteriorMarginPx
    // excludes the demosaic edge-blur ring, kernels.cu's documented,
    // honest artifact) -----------------------------------------------------
    double dolp_abs_err_sum = 0.0; long long dolp_n = 0;
    double aolp_err_sum = 0.0; long long aolp_n = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int i = y * kW + x;
            const int lbl = truth.label[i];
            if (lbl == kLabelBackground || !is_interior_object_pixel(x, y, lbl)) continue;
            dolp_abs_err_sum += std::fabs(static_cast<double>(gpu.dolp[i]) - static_cast<double>(truth.dolp[i]));
            ++dolp_n;
            if (truth.dolp[i] > kHighDolpFloorForAolpGate) {
                const double measured_deg = static_cast<double>(gpu.aolp_rad[i]) * (180.0 / static_cast<double>(kPi));
                double d = std::fabs(measured_deg - static_cast<double>(truth.aolp_deg[i]));
                d = std::fmod(d, 180.0);
                if (d > 90.0) d = 180.0 - d;   // the same half-angle wrap, expressed in degrees here
                aolp_err_sum += d;
                ++aolp_n;
            }
        }
    }
    const double dolp_mae = dolp_n > 0 ? dolp_abs_err_sum / static_cast<double>(dolp_n) : 0.0;
    const double aolp_mae_deg = aolp_n > 0 ? aolp_err_sum / static_cast<double>(aolp_n) : 0.0;
    const bool gate_stokes_accuracy = (dolp_mae <= kMaxDolpMaeInterior) && (aolp_mae_deg <= kMaxAolpCircErrDeg);
    std::printf("GATE stokes_accuracy: %s\n", gate_stokes_accuracy ? "PASS" : "FAIL");
    std::printf("[info] stokes_accuracy: DoLP MAE (interior, n=%lld) = %.4f (need <= %.2f) | AoLP circular MAE "
               "(high-DoLP interior, n=%lld) = %.3f deg (need <= %.1f deg)\n",
               dolp_n, dolp_mae, kMaxDolpMaeInterior, aolp_n, aolp_mae_deg, kMaxAolpCircErrDeg);

    // ---- GATE malus_consistency: the FREE self-consistency invariant -----
    double resid_abs_sum = 0.0;
    for (float r : gpu.residual) resid_abs_sum += std::fabs(static_cast<double>(r));
    const double resid_mean_abs = resid_abs_sum / static_cast<double>(kN);
    const bool gate_malus = resid_mean_abs <= kMaxMalusResidualMeanAbs;
    std::printf("GATE malus_consistency: %s\n", gate_malus ? "PASS" : "FAIL");
    std::printf("[info] malus_consistency: mean|(I0+I90)-(I45+I135)| = %.4f DN over all %d pixels "
               "(need <= %.1f DN -- 4 measurements, 3 parameters, this is the free 1-DOF residual)\n",
               resid_mean_abs, kN, kMaxMalusResidualMeanAbs);

    // ---- GATE fresnel_anchor: THE physics gate ----------------------------
    double pane_dolp_sum = 0.0; long long pane_n = 0;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x)
            if (is_interior_object_pixel(x, y, kLabelPane)) { pane_dolp_sum += static_cast<double>(gpu.dolp[y * kW + x]); ++pane_n; }
    const double pane_dolp_measured = pane_n > 0 ? pane_dolp_sum / static_cast<double>(pane_n) : 0.0;
    const double pane_dolp_predicted = static_cast<double>(fresnel_dolp(kPaneThetaDeg * (kPi / 180.0f), kNGlass));
    const double fresnel_gap = std::fabs(pane_dolp_measured - pane_dolp_predicted);
    const bool gate_fresnel = fresnel_gap <= kFresnelAnchorTolDolp;
    std::printf("GATE fresnel_anchor: %s\n", gate_fresnel ? "PASS" : "FAIL");
    std::printf("[info] fresnel_anchor: measured DoLP on glass pane (interior, n=%lld) = %.5f | closed-form "
               "Fresnel prediction at theta_i=%.1f deg, n=%.2f: %.5f | |diff| = %.5f (need <= %.3f)\n",
               pane_n, pane_dolp_measured, static_cast<double>(kPaneThetaDeg), static_cast<double>(kNGlass),
               pane_dolp_predicted, fresnel_gap, kFresnelAnchorTolDolp);

    // ---- GATE detection: recall in BOTH directions -------------------------
    const double glass_recall_dolp = recall_over_labels(mask_dolp_gpu, truth.label, { kLabelPane, kLabelDome });
    const double metal_recall_dolp = recall_over_labels(mask_dolp_gpu, truth.label, { kLabelMetal });
    const double glass_recall_intensity = recall_over_labels(mask_int_gpu, truth.label, { kLabelPane, kLabelDome });
    const bool gate_detection = (glass_recall_dolp >= kMinGlassRecallDolp) && (metal_recall_dolp >= kMinMetalRecallDolp)
                              && (glass_recall_intensity <= kMaxGlassRecallIntensity);
    std::printf("GATE detection: %s\n", gate_detection ? "PASS" : "FAIL");
    std::printf("[info] detection: DoLP-based recall: glass=%.1f%% (need >=%.0f%%) metal=%.1f%% (need >=%.0f%%) | "
               "intensity-based glass recall=%.1f%% (need <=%.0f%% -- the reason this project exists: intensity "
               "cannot see the glass, DoLP can)\n",
               100.0 * glass_recall_dolp, 100.0 * kMinGlassRecallDolp, 100.0 * metal_recall_dolp, 100.0 * kMinMetalRecallDolp,
               100.0 * glass_recall_intensity, 100.0 * kMaxGlassRecallIntensity);

    // ---- GATE brewster_sweep: pure closed-form sweep, no rendering --------
    std::vector<std::pair<float, float>> brewster_curve;   // (angle_deg, dolp)
    float peak_angle_deg = 0.0f, peak_dolp = -1.0f;
    for (int deg = 5; deg <= 85; ++deg) {
        const float theta_rad = static_cast<float>(deg) * (kPi / 180.0f);
        const float d = fresnel_dolp(theta_rad, kNGlass);
        brewster_curve.emplace_back(static_cast<float>(deg), d);
        if (d > peak_dolp) { peak_dolp = d; peak_angle_deg = static_cast<float>(deg); }
    }
    const double brewster_true_deg = std::atan(static_cast<double>(kNGlass)) * (180.0 / static_cast<double>(kPi));
    const double brewster_gap_deg = std::fabs(static_cast<double>(peak_angle_deg) - brewster_true_deg);
    const bool gate_brewster = brewster_gap_deg <= kBrewsterPeakTolDeg;
    std::printf("GATE brewster_sweep: %s\n", gate_brewster ? "PASS" : "FAIL");
    std::printf("[info] brewster_sweep: closed-form DoLP(theta) peaks at %.0f deg (DoLP=%.4f) | true Brewster "
               "angle atan(n)=%.2f deg | |diff| = %.2f deg (need <= %.1f deg)\n",
               static_cast<double>(peak_angle_deg), static_cast<double>(peak_dolp), brewster_true_deg, brewster_gap_deg, kBrewsterPeakTolDeg);

    // ---- GATE negative_control: matte-only scene, zero detections --------
    PipelineResult neg;
    double ms_neg = 0.0;
    run_pipeline_gpu(h_negctrl, kW, kH, neg, ms_neg);
    std::vector<unsigned char> mask_neg;
    int sweeps_neg = 0;
    run_detection_gpu(neg.dolp, kDolpThreshold, kW, kH, mask_neg, sweeps_neg);
    long long neg_fg_px = 0; for (unsigned char m : mask_neg) neg_fg_px += m;
    const double neg_mean_dolp = mean_of(neg.dolp);
    const bool gate_negctrl = (neg_fg_px == 0);
    std::printf("GATE negative_control: %s\n", gate_negctrl ? "PASS" : "FAIL");
    std::printf("[info] negative_control: matte-only scene, mean DoLP = %.4f, detected foreground pixels = %lld "
               "(need exactly 0)\n", neg_mean_dolp, neg_fg_px);

    // ---- ARTIFACTS ----------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);

    write_pgm_from_float(out_dir + "/intensity_s0.pgm", gpu.s0, kW, kH);
    std::printf("ARTIFACT: demo/out/intensity_s0.pgm written (S0 -- plain intensity; the glass objects are "
               "INVISIBLE here by construction)\n");

    std::vector<float> dolp_255(kN);
    for (int i = 0; i < kN; ++i) dolp_255[i] = std::min(gpu.dolp[i], 1.0f) * 255.0f;
    write_pgm_from_float(out_dir + "/dolp.pgm", dolp_255, kW, kH);
    std::printf("ARTIFACT: demo/out/dolp.pgm written (DoLP scaled to [0,255] -- the glass objects GLOW here)\n");

    std::vector<unsigned char> aolp_rgb(static_cast<size_t>(kN) * 3);
    for (int i = 0; i < kN; ++i) {
        const float aolp_deg = gpu.aolp_rad[i] * (180.0f / kPi);
        unsigned char r, g, b;
        hsv_to_rgb(2.0f * aolp_deg, 1.0f, std::min(gpu.dolp[i], 1.0f), r, g, b);
        aolp_rgb[static_cast<size_t>(i) * 3 + 0] = r; aolp_rgb[static_cast<size_t>(i) * 3 + 1] = g; aolp_rgb[static_cast<size_t>(i) * 3 + 2] = b;
    }
    write_ppm(out_dir + "/aolp_vis.ppm", aolp_rgb, kW, kH);
    std::printf("ARTIFACT: demo/out/aolp_vis.ppm written (HSV: hue=2*AoLP deg, saturation=1, value=DoLP -- "
               "the dome's radial 'polarization donut' vs. the metal bar's constant hue)\n");

    // Detection overlay: side-by-side (2*kW wide), same S0 grayscale base
    // both halves -- left tinted RED where the INTENSITY-only pipeline
    // fired, right tinted GREEN where the DoLP pipeline fired. Visualizes
    // the GATE detection comparison directly.
    std::vector<unsigned char> overlay(static_cast<size_t>(2 * kW) * kH * 3);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int i = y * kW + x;
            float gray_f = gpu.s0[i]; gray_f = gray_f < 0.0f ? 0.0f : (gray_f > 255.0f ? 255.0f : gray_f);
            const unsigned char gray = static_cast<unsigned char>(gray_f + 0.5f);
            // left half: intensity-only result
            size_t li = (static_cast<size_t>(y) * (2 * kW) + x) * 3;
            if (mask_int_gpu[i]) { overlay[li] = 255; overlay[li + 1] = gray / 3; overlay[li + 2] = gray / 3; }
            else { overlay[li] = gray; overlay[li + 1] = gray; overlay[li + 2] = gray; }
            // right half: DoLP-based result
            size_t ri = (static_cast<size_t>(y) * (2 * kW) + (kW + x)) * 3;
            if (mask_dolp_gpu[i]) { overlay[ri] = gray / 3; overlay[ri + 1] = 255; overlay[ri + 2] = gray / 3; }
            else { overlay[ri] = gray; overlay[ri + 1] = gray; overlay[ri + 2] = gray; }
        }
    }
    write_ppm(out_dir + "/detection_overlay.ppm", overlay, 2 * kW, kH);
    std::printf("ARTIFACT: demo/out/detection_overlay.ppm written (left: intensity-only detection in red -- "
               "misses the glass; right: DoLP-based detection in green -- finds it)\n");

    {
        std::ofstream bc(out_dir + "/brewster_curve.csv");
        bc << "# closed-form Fresnel DoLP(theta_i) for n=" << kNGlass << " -- no rendering, pure physics\n";
        bc << "angle_deg,dolp\n";
        for (auto& p : brewster_curve) bc << fmt(p.first, 1) << "," << fmt(p.second, 6) << "\n";
    }
    std::printf("ARTIFACT: demo/out/brewster_curve.csv written (DoLP vs incidence angle, peak near Brewster's %.1f deg)\n", brewster_true_deg);

    std::vector<CsvRow> gate_rows = {
        { "stokes_accuracy", "dolp_mae_interior", fmt(dolp_mae, 5), fmt(kMaxDolpMaeInterior, 2), gate_stokes_accuracy ? "PASS" : "FAIL" },
        { "stokes_accuracy", "aolp_circ_mae_deg", fmt(aolp_mae_deg, 3), fmt(kMaxAolpCircErrDeg, 1), gate_stokes_accuracy ? "PASS" : "FAIL" },
        { "malus_consistency", "mean_abs_residual_dn", fmt(resid_mean_abs, 4), fmt(kMaxMalusResidualMeanAbs, 1), gate_malus ? "PASS" : "FAIL" },
        { "fresnel_anchor", "abs_diff_dolp", fmt(fresnel_gap, 5), fmt(kFresnelAnchorTolDolp, 3), gate_fresnel ? "PASS" : "FAIL" },
        { "detection", "glass_recall_dolp", fmt(glass_recall_dolp, 4), fmt(kMinGlassRecallDolp, 2) + " (floor)", (glass_recall_dolp >= kMinGlassRecallDolp) ? "PASS" : "FAIL" },
        { "detection", "metal_recall_dolp", fmt(metal_recall_dolp, 4), fmt(kMinMetalRecallDolp, 2) + " (floor)", (metal_recall_dolp >= kMinMetalRecallDolp) ? "PASS" : "FAIL" },
        { "detection", "glass_recall_intensity", fmt(glass_recall_intensity, 4), fmt(kMaxGlassRecallIntensity, 2) + " (ceiling)", (glass_recall_intensity <= kMaxGlassRecallIntensity) ? "PASS" : "FAIL" },
        { "brewster_sweep", "peak_angle_deg", fmt(static_cast<double>(peak_angle_deg), 1), fmt(brewster_true_deg, 2) + " +/- " + fmt(kBrewsterPeakTolDeg, 1), gate_brewster ? "PASS" : "FAIL" },
        { "negative_control", "detected_fg_px", std::to_string(neg_fg_px), "0", gate_negctrl ? "PASS" : "FAIL" },
    };
    write_gates_csv(out_dir + "/gates_metrics.csv", gate_rows);
    std::printf("ARTIFACT: demo/out/gates_metrics.csv written\n");

    // ---- RESULT ---------------------------------------------------------
    const bool all_verify = verify_demosaic && verify_stokes && verify_dolp && verify_aolp && verify_residual && verify_det_dolp && verify_det_int;
    const bool all_gates = gate_stokes_accuracy && gate_malus && gate_fresnel && gate_detection && gate_brewster && gate_negctrl;
    const bool pass = all_verify && all_gates;
    if (pass) {
        std::printf("RESULT: PASS (all VERIFY checks and all GATEs passed)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (%s%s)\n", all_verify ? "" : "a VERIFY check failed ", all_gates ? "" : "a GATE failed");
        return EXIT_FAILURE;
    }
}
