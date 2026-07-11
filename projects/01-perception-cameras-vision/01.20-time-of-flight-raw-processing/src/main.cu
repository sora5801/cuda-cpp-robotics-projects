// ===========================================================================
// main.cu — entry point for project 01.20
//           Time-of-flight raw processing: phase unwrapping, flying-pixel
//           removal (continuous-wave indirect ToF)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Load the committed 8-frame tap stack (2 frequencies x 4 taps) and
//      ground truth (true depth, surface id, true flying-pixel label) from
//      data/sample/, asserting params.csv matches the compiled kernels.cuh
//      contract (catches a Python/C++ drift loudly instead of silently).
//   2. Run all SIX pipeline stages (extract phase/amplitude -> single-freq
//      depth -> dual-freq unwrap -> confidence mask -> flying-pixel detect
//      -> back-project) on BOTH the GPU and an independent CPU oracle, each
//      as its own end-to-end cascade, and VERIFY agreement stage by stage.
//   3. Run NINE independent gates against the synthetic ground truth (not
//      against each other): phase_extraction, offset_invariance,
//      aliasing_demo, unwrap_recovery, flying_pixel, three reconstruction
//      gates (plane/sphere/step), and dark_cohort. Print one [info]
//      noise_scaling diagnostic (not gated — see its own comment).
//   4. Write nine labeled artifacts to demo/out/ and print a final
//      PASS/FAIL RESULT.
//
// Output contract (load-bearing!) — same as every project in this repo:
// "[demo]"/"[info]"/"[time]" lines are NOT diffed (device names and timings
// vary by machine and run); "PROBLEM:", "SCENARIO:", "VERIFY:", "GATE:",
// "ARTIFACT:", "RESULT:" lines ARE diffed verbatim by demo/run_demo.*
// against demo/expected_output.txt. Measured floating-point numbers
// therefore live on "[info]" lines; the paired "VERIFY:"/"GATE:" line
// carries only the PASS/FAIL verdict and the fixed tolerance/floor (never
// the measured value) — 01.19's pattern, applied to six stages and nine
// gates instead of five and eight. Change a stable line here => update
// demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <utility>

// ---------------------------------------------------------------------------
// Verification tolerances (GPU-vs-CPU, two INDEPENDENT end-to-end cascades —
// see the "Independence ruling" restated in reference_cpu.cpp). Every one of
// these is a FIXED CONSTANT printed on the stable "VERIFY:" line (never the
// measured value — see the file header). Chosen with margin over what this
// project's committed sample actually measures (recorded in data/README.md
// "How the sample was tuned").
// ---------------------------------------------------------------------------
static const float kTolPhaseRad          = 1.0e-3f;   // extract_phase_amplitude: atan2f ULP drift
static const float kTolAmplitudeCounts   = 5.0e-2f;   // extract_phase_amplitude: sqrtf ULP drift
static const float kTolSingleFreqDepthM  = 1.0e-3f;   // single_freq_depth: inherits Stage-1 phase drift
static const float kTolUnwrapDepthM      = 2.0e-3f;   // dual_freq_unwrap: depth, on wrap-agreeing pixels only
static const int   kMaxWrapCountMismatch = 30;         // dual_freq_unwrap: near-boundary wrap-decision ties
static const int   kMaxConfidenceMismatch= 30;         // confidence_mask: near-floor amplitude ties
static const int   kMaxFlyingMismatch    = 30;         // flying_pixel_detect: inherits upstream near-ties
static const float kTolXyzM              = 4.0e-3f;   // backproject: on final-valid-agreeing pixels only

// Every floor/bound below is a "measured-then-margined gate" (repo
// convention, e.g. 01.19): tuned against ACTUAL numbers measured on the
// committed sample (data/README.md "How the sample was tuned"), then set
// with visible margin so the gate means something without being brittle to
// this project's ordinary run-to-run determinism (no cuRAND anywhere; the
// sample's noise is baked into the committed PGMs — CLAUDE.md §12).
static const float kGatePhaseExtractionMeanBoundRad = 0.05f;   // phase_extraction: mean |dphi| bound (measured ~0.029 rad)
static const float kGateOffsetInvarianceTolRad      = 1.0e-4f; // offset_invariance: max |dphi| bound (measured ~0)
static const float kGateAliasingFractionFloor       = 0.90f;   // aliasing_demo: fraction of far-background pixels grossly wrong (measured 100%)
static const float kGateUnwrapMeanErrBoundM         = 0.025f;  // unwrap_recovery: mean |err| bound, m (measured ~13 mm)
static const float kGateUnwrapWrapCorrectFloor      = 0.95f;   // unwrap_recovery: wrap-count correctness rate floor (measured ~98%)
static const float kGateFlyingPrecisionFloor        = 0.85f;   // flying_pixel: precision floor (measured ~100%)
static const float kGateFlyingRecallFloor           = 0.50f;   // flying_pixel: recall floor (measured ~62%)
static const float kGatePlaneRmsBoundM              = 0.030f;  // reconstruction_plane: RMS residual bound, m (measured ~14 mm)
static const float kGateSphereRadiusPctBound        = 8.0f;    // reconstruction_sphere: |err|/truth bound, % (measured ~3.3%)
static const float kGateStepHeightPctBound          = 2.0f;    // reconstruction_step: |err|/truth bound, % (measured ~0.01%)
static const float kGateDarkCohortRejectFloor       = 0.95f;   // dark_cohort: cohort rejection-rate floor (measured 100%)
static const float kGateDarkCohortSurvivorBoundM    = 0.06f;   // dark_cohort: surviving-pixel depth sanity bound, m

// ===========================================================================
// Small host-side utilities. All plain C++17 (no <filesystem> — see
// util/paths.h "Why not std::filesystem"); no CUDA here.
// ===========================================================================

// ---- PGM (P5) reader/writer -------------------------------------------------
struct PgmImage {
    int w = 0, h = 0;
    std::vector<unsigned char> data;   // row-major, [h*w]
    bool ok = false;
};
static PgmImage read_pgm(const std::string& path)
{
    PgmImage img;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return img;
    std::string magic;
    f >> magic;
    if (magic != "P5") return img;
    int maxval = 0;
    f >> img.w >> img.h >> maxval;
    if (!f || img.w <= 0 || img.h <= 0 || maxval != 255) return img;
    f.get();   // consume the single mandatory whitespace byte before raster data (PGM P5 format)
    img.data.resize(static_cast<size_t>(img.w) * static_cast<size_t>(img.h));
    f.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
    img.ok = !f.fail();
    return img;
}
static bool write_pgm(const std::string& path, const std::vector<unsigned char>& data, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(f);
}
// PPM (P6): used for the two orthographic profile renders (RGB container,
// R=G=B grayscale content — the artifact spec names it a PPM, 01.19's
// precedent).
static bool write_ppm_gray(const std::string& path, const std::vector<unsigned char>& gray, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P6\n" << w << " " << h << "\n255\n";
    std::vector<unsigned char> rgb(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    for (size_t i = 0; i < gray.size(); ++i) { rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = gray[i]; }
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(f);
}

// ---- raw ground-truth binaries ---------------------------------------------
static bool read_f32_array(const std::string& path, std::vector<float>& out, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return !f.fail();
}
static bool read_u8_array(const std::string& path, std::vector<unsigned char>& out, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count));
    return !f.fail();
}

// ---- params.csv: key,value pairs (comment lines start with '#') -----------
static bool load_params_csv(const std::string& path, std::map<std::string, double>& kv)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string key, val;
        if (!std::getline(ss, key, ',')) continue;
        if (!std::getline(ss, val, ',')) continue;
        kv[key] = std::strtod(val.c_str(), nullptr);
    }
    return true;
}
static double kv_get(const std::map<std::string, double>& kv, const char* key, double fallback)
{
    auto it = kv.find(key);
    return (it == kv.end()) ? fallback : it->second;
}

// assert_params_match — the honesty gate against Python/C++ drift (file
// header): every sensor/scene constant params.csv records must equal the
// compiled kernels.cuh contract, or the committed data no longer describes
// the sensor this program models. Prints readable mismatches to stderr.
static bool assert_params_match(const std::map<std::string, double>& kv)
{
    bool ok = true;
    auto check_int = [&](const char* key, int expect) {
        const double v = kv_get(kv, key, -1.0e18);
        if (static_cast<long long>(v + (v >= 0 ? 0.5 : -0.5)) != expect) {
            std::fprintf(stderr, "params.csv mismatch: %s = %.6f, expected %d\n", key, v, expect);
            ok = false;
        }
    };
    auto check_float = [&](const char* key, float expect, float tol) {
        const double v = kv_get(kv, key, -1.0e18);
        if (std::fabs(v - static_cast<double>(expect)) > tol) {
            std::fprintf(stderr, "params.csv mismatch: %s = %.6f, expected %.6f\n", key, v, static_cast<double>(expect));
            ok = false;
        }
    };
    check_int("cam_w_px", kCamW);
    check_int("cam_h_px", kCamH);
    check_float("cam_fx_px", kCamFx, 1.0e-3f);
    check_float("cam_fy_px", kCamFy, 1.0e-3f);
    check_float("cam_cx_px", kCamCx, 1.0e-3f);
    check_float("cam_cy_px", kCamCy, 1.0e-3f);
    check_float("freq1_hz", kFreq1Hz, 50.0f);      // float32 ULP at 6e7 is ~4-8; 50 is a safe, still-meaningful margin
    check_float("freq2_hz", kFreq2Hz, 50.0f);
    check_int("num_taps", kNumTaps);
    check_float("max_scene_depth_m", kMaxSceneDepthM, 1.0e-3f);
    return ok;
}

// ---- small dense linear solve (double precision, partial pivoting) --------
// Solves A x = b for an n x n system. Used ONLY by the reconstruction gates
// below for their one-off plane (3x3) and sphere (4x4) normal-equations
// solves — O(1) work done ONCE per gate on a few thousand ALREADY-
// BACKPROJECTED points, not a per-pixel parallel workload, so it deliberately
// stays on the host (01.19's identical reasoning: no exploitable parallelism
// at this scale).
static bool solve_dense(std::vector<std::vector<double>> A, std::vector<double> b, std::vector<double>& x)
{
    const int n = static_cast<int>(A.size());
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(A[col][col]);
        for (int r = col + 1; r < n; ++r) {
            if (std::fabs(A[r][col]) > best) { best = std::fabs(A[r][col]); piv = r; }
        }
        if (best < 1.0e-12) return false;   // singular (or degenerate point set) — caller reports honestly
        std::swap(A[col], A[piv]);
        std::swap(b[col], b[piv]);
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / A[col][col];
            for (int c = col; c < n; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    x.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = b[static_cast<size_t>(i)] / A[static_cast<size_t>(i)][static_cast<size_t>(i)];
    return true;
}

// angdiff — signed shortest angular difference (a-b), wrapped to (-pi,pi].
// Needed because offset_invariance compares two [0,2pi)-convention angles
// that could sit on opposite sides of the 0/2pi seam even when physically
// identical (THEORY.md "Numerical considerations").
static inline float angdiff(float a, float b)
{
    float d = a - b;
    const float twopi = 6.28318530717958647692f;
    while (d > 3.14159265358979323846f) d -= twopi;
    while (d < -3.14159265358979323846f) d += twopi;
    return d;
}

// ===========================================================================
// GPU stage wrappers: allocate device buffers, upload, launch, copy back,
// free. Each returns the KERNEL-ONLY elapsed ms (GpuTimer, cudaEvents — see
// util/timer.cuh for why the CPU clock cannot measure this correctly).
// ===========================================================================
static float gpu_extract_phase_amplitude(const std::vector<float>& taps, std::vector<float>& phase,
                                         std::vector<float>& amplitude, int n)
{
    float* d_taps = nullptr; float* d_phase = nullptr; float* d_amplitude = nullptr;
    CUDA_CHECK(cudaMalloc(&d_taps, taps.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phase, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_amplitude, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_taps, taps.data(), taps.size() * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_extract_phase_amplitude(d_taps, d_phase, d_amplitude, n);
    const float ms = t.end_ms();
    phase.resize(static_cast<size_t>(n)); amplitude.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(phase.data(), d_phase, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(amplitude.data(), d_amplitude, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_taps)); CUDA_CHECK(cudaFree(d_phase)); CUDA_CHECK(cudaFree(d_amplitude));
    return ms;
}

static float gpu_single_freq_depth(const std::vector<float>& phase, float ambig_m, std::vector<float>& depth, int n)
{
    float* d_phase = nullptr; float* d_depth = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phase, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_phase, phase.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_single_freq_depth(d_phase, ambig_m, d_depth, n);
    const float ms = t.end_ms();
    depth.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(depth.data(), d_depth, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_phase)); CUDA_CHECK(cudaFree(d_depth));
    return ms;
}

static float gpu_dual_freq_unwrap(const std::vector<float>& phase1, const std::vector<float>& phase2,
                                  std::vector<float>& depth, std::vector<int>& wrap_count, int n)
{
    float* d_phase1 = nullptr; float* d_phase2 = nullptr; float* d_depth = nullptr; int* d_wrap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phase1, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phase2, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wrap, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_phase1, phase1.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phase2, phase2.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_dual_freq_unwrap(d_phase1, d_phase2, d_depth, d_wrap, n);
    const float ms = t.end_ms();
    depth.resize(static_cast<size_t>(n)); wrap_count.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(depth.data(), d_depth, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(wrap_count.data(), d_wrap, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_phase1)); CUDA_CHECK(cudaFree(d_phase2)); CUDA_CHECK(cudaFree(d_depth)); CUDA_CHECK(cudaFree(d_wrap));
    return ms;
}

static float gpu_confidence_mask(const std::vector<float>& amplitude, float floor_v,
                                 std::vector<unsigned char>& valid, int n)
{
    float* d_amp = nullptr; unsigned char* d_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amp, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_amp, amplitude.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_confidence_mask(d_amp, floor_v, d_valid, n);
    const float ms = t.end_ms();
    valid.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(valid.data(), d_valid, static_cast<size_t>(n) * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_amp)); CUDA_CHECK(cudaFree(d_valid));
    return ms;
}

static float gpu_flying_pixel_detect(const std::vector<float>& depth, const std::vector<float>& amplitude,
                                     const std::vector<unsigned char>& confidence_valid,
                                     std::vector<unsigned char>& flying, int w, int h)
{
    const int n = w * h;
    float* d_depth = nullptr; float* d_amp = nullptr; unsigned char* d_conf = nullptr; unsigned char* d_flying = nullptr;
    CUDA_CHECK(cudaMalloc(&d_depth, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_amp, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conf, static_cast<size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_flying, static_cast<size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_depth, depth.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_amp, amplitude.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conf, confidence_valid.data(), static_cast<size_t>(n) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_flying_pixel_detect(d_depth, d_amp, d_conf, d_flying, w, h);
    const float ms = t.end_ms();
    flying.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(flying.data(), d_flying, static_cast<size_t>(n) * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_depth)); CUDA_CHECK(cudaFree(d_amp)); CUDA_CHECK(cudaFree(d_conf)); CUDA_CHECK(cudaFree(d_flying));
    return ms;
}

static float gpu_backproject(const std::vector<float>& depth, const std::vector<unsigned char>& final_valid,
                             std::vector<float>& xyz, int n)
{
    float* d_depth = nullptr; unsigned char* d_valid = nullptr; float* d_xyz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_depth, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_depth, depth.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valid, final_valid.data(), static_cast<size_t>(n) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_backproject(d_depth, d_valid, d_xyz, n);
    const float ms = t.end_ms();
    xyz.resize(static_cast<size_t>(n) * 3);
    CUDA_CHECK(cudaMemcpy(xyz.data(), d_xyz, static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_depth)); CUDA_CHECK(cudaFree(d_valid)); CUDA_CHECK(cudaFree(d_xyz));
    return ms;
}

// ---------------------------------------------------------------------------
// depth_to_pgm — render a depth map (meters) to a normalized [1,255] PGM;
// 0 (pure black) is reserved for "invalid" (01.19's decoded_column_map
// convention, reused here). depths outside [lo,hi] are clamped for display.
// ---------------------------------------------------------------------------
static std::vector<unsigned char> depth_to_pgm(const std::vector<float>& depth,
                                               const std::vector<unsigned char>& valid,
                                               float lo, float hi, int n)
{
    std::vector<unsigned char> img(static_cast<size_t>(n), 0);
    const float range = std::max(1.0e-6f, hi - lo);
    for (int i = 0; i < n; ++i) {
        if (!valid[static_cast<size_t>(i)]) continue;
        float norm = (depth[static_cast<size_t>(i)] - lo) / range;
        norm = std::min(1.0f, std::max(0.0f, norm));
        const int v = static_cast<int>(std::lround(static_cast<double>(norm) * 254.0)) + 1;   // [1,255]
        img[static_cast<size_t>(i)] = static_cast<unsigned char>(v < 1 ? 1 : (v > 255 ? 255 : v));
    }
    return img;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir_override;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir_override = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data <dir with tap PGMs + truth + params.csv>]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Time-of-flight raw processing: phase unwrapping, flying-pixel removal (project 01.20)\n");
    print_device_info();
    std::printf("PROBLEM: continuous-wave iToF, camera %dx%d px, freq1=%.0f MHz (ambig %.3f m), "
                "freq2=%.0f MHz (ambig %.3f m), %d taps/frequency, FP32\n",
                kCamW, kCamH, static_cast<double>(kFreq1Hz / 1.0e6f), static_cast<double>(kAmbig1M),
                static_cast<double>(kFreq2Hz / 1.0e6f), static_cast<double>(kAmbig2M), kNumTaps);

    const int n = kNPix;
    const int w = kCamW, h = kCamH;

    // ---- load params.csv and assert the compiled contract matches --------
    const std::string params_path = find_data_file(data_dir_override, argv[0], "params.csv");
    if (params_path.empty()) {
        std::printf("SCENARIO: NOT FOUND - data/sample/params.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }
    std::map<std::string, double> params;
    load_params_csv(params_path, params);
    if (!assert_params_match(params)) {
        std::printf("SCENARIO: MALFORMED - params.csv does not match kernels.cuh (see stderr)\n");
        std::printf("RESULT: FAIL (data/code drift)\n");
        return 1;
    }
    std::printf("SCENARIO: synthetic room-scale scene (tilted background wall + sphere + box step-edge), "
                "low-reflectivity dark cohort, seed %d [synthetic]\n",
                static_cast<int>(kv_get(params, "seed", -1.0)));

    // ---- load the 8-frame tap stack (2 frequencies x 4 taps) --------------
    std::vector<float> h_taps1(static_cast<size_t>(kNumTaps) * n);
    std::vector<float> h_taps2(static_cast<size_t>(kNumTaps) * n);
    std::vector<unsigned char> tap_sample_f1_bytes, tap_sample_f2_bytes;   // artifact: tap 0 of each frequency
    for (int freq_idx = 1; freq_idx <= 2; ++freq_idx) {
        std::vector<float>& dst = (freq_idx == 1) ? h_taps1 : h_taps2;
        for (int k = 0; k < kNumTaps; ++k) {
            char fname[32];
            std::snprintf(fname, sizeof(fname), "tof_f%d_tap%d.pgm", freq_idx, k);
            const std::string p = find_data_file(data_dir_override, argv[0], fname);
            PgmImage img = p.empty() ? PgmImage{} : read_pgm(p);
            if (!img.ok || img.w != w || img.h != h) {
                std::printf("SCENARIO: NOT FOUND - data/sample/%s missing or malformed\n", fname);
                std::printf("RESULT: FAIL (sample data missing)\n");
                return 1;
            }
            for (int pix = 0; pix < n; ++pix) dst[static_cast<size_t>(k) * n + pix] = static_cast<float>(img.data[static_cast<size_t>(pix)]);
            if (k == 0) { if (freq_idx == 1) tap_sample_f1_bytes = img.data; else tap_sample_f2_bytes = img.data; }
        }
    }

    // ---- load ground truth --------------------------------------------------
    std::vector<float> truth_depth;
    std::vector<unsigned char> truth_surf, truth_flying;
    {
        const std::string pd = find_data_file(data_dir_override, argv[0], "truth_depth.bin");
        const std::string ps = find_data_file(data_dir_override, argv[0], "truth_surface.bin");
        const std::string pf = find_data_file(data_dir_override, argv[0], "truth_flying.bin");
        if (pd.empty() || ps.empty() || pf.empty() ||
            !read_f32_array(pd, truth_depth, static_cast<size_t>(n)) ||
            !read_u8_array(ps, truth_surf, static_cast<size_t>(n)) ||
            !read_u8_array(pf, truth_flying, static_cast<size_t>(n))) {
            std::printf("SCENARIO: NOT FOUND - data/sample truth_*.bin missing or malformed\n");
            std::printf("RESULT: FAIL (sample data missing)\n");
            return 1;
        }
    }

    bool all_ok = true;   // ANDed with every VERIFY/GATE below -> final RESULT

    // ===================================================================
    // Two INDEPENDENT end-to-end cascades: GPU stage1->6 feeding GPU
    // stage2->6, and CPU stage1->6 feeding CPU stage2->6 — never GPU
    // output feeding a CPU stage or vice versa (01.19's twin-pipeline
    // pattern). Compared stage by stage below.
    // ===================================================================

    // ======================= STAGE 1: EXTRACT PHASE/AMPLITUDE ================
    std::vector<float> gpu_phase1, gpu_amp1, gpu_phase2, gpu_amp2;
    std::vector<float> cpu_phase1, cpu_amp1, cpu_phase2, cpu_amp2;
    const float s1_gpu_ms_a = gpu_extract_phase_amplitude(h_taps1, gpu_phase1, gpu_amp1, n);
    const float s1_gpu_ms_b = gpu_extract_phase_amplitude(h_taps2, gpu_phase2, gpu_amp2, n);
    cpu_phase1.resize(static_cast<size_t>(n)); cpu_amp1.resize(static_cast<size_t>(n));
    cpu_phase2.resize(static_cast<size_t>(n)); cpu_amp2.resize(static_cast<size_t>(n));
    CpuTimer ct1; ct1.begin();
    extract_phase_amplitude_cpu(h_taps1.data(), cpu_phase1.data(), cpu_amp1.data(), n);
    extract_phase_amplitude_cpu(h_taps2.data(), cpu_phase2.data(), cpu_amp2.data(), n);
    const double s1_cpu_ms = ct1.end_ms();
    float worst_phase = 0.0f, worst_amp = 0.0f;
    for (int i = 0; i < n; ++i) {
        worst_phase = std::max(worst_phase, std::fabs(angdiff(gpu_phase1[static_cast<size_t>(i)], cpu_phase1[static_cast<size_t>(i)])));
        worst_phase = std::max(worst_phase, std::fabs(angdiff(gpu_phase2[static_cast<size_t>(i)], cpu_phase2[static_cast<size_t>(i)])));
        worst_amp = std::max(worst_amp, std::fabs(gpu_amp1[static_cast<size_t>(i)] - cpu_amp1[static_cast<size_t>(i)]));
        worst_amp = std::max(worst_amp, std::fabs(gpu_amp2[static_cast<size_t>(i)] - cpu_amp2[static_cast<size_t>(i)]));
    }
    std::printf("[info] extract_phase_amplitude: worst |dphi|=%.3e rad, worst |damplitude|=%.3e counts (both frequencies)\n",
                static_cast<double>(worst_phase), static_cast<double>(worst_amp));
    const bool s1_pass = (worst_phase <= kTolPhaseRad) && (worst_amp <= kTolAmplitudeCounts);
    all_ok &= s1_pass;
    std::printf("VERIFY: extract_phase_amplitude %s (GPU vs CPU within tol %.1e rad / %.1e counts, both frequencies)\n",
                s1_pass ? "PASS" : "FAIL", static_cast<double>(kTolPhaseRad), static_cast<double>(kTolAmplitudeCounts));

    // ======================= STAGE 2: SINGLE-FREQUENCY DEPTH ==================
    std::vector<float> gpu_depth_wrapped, cpu_depth_wrapped;
    const float s2_gpu_ms = gpu_single_freq_depth(gpu_phase1, kAmbig1M, gpu_depth_wrapped, n);
    cpu_depth_wrapped.resize(static_cast<size_t>(n));
    CpuTimer ct2; ct2.begin();
    single_freq_depth_cpu(cpu_phase1.data(), kAmbig1M, cpu_depth_wrapped.data(), n);
    const double s2_cpu_ms = ct2.end_ms();
    float worst_wrapped = 0.0f;
    for (int i = 0; i < n; ++i)
        worst_wrapped = std::max(worst_wrapped, std::fabs(gpu_depth_wrapped[static_cast<size_t>(i)] - cpu_depth_wrapped[static_cast<size_t>(i)]));
    std::printf("[info] single_freq_depth: worst |ddepth|=%.3e m\n", static_cast<double>(worst_wrapped));
    const bool s2_pass = worst_wrapped <= kTolSingleFreqDepthM;
    all_ok &= s2_pass;
    std::printf("VERIFY: single_freq_depth %s (GPU vs CPU within tol %.1e m)\n", s2_pass ? "PASS" : "FAIL", static_cast<double>(kTolSingleFreqDepthM));

    // ======================= STAGE 3: DUAL-FREQUENCY UNWRAP ===================
    std::vector<float> gpu_depth_unwrap, cpu_depth_unwrap;
    std::vector<int> gpu_wrap, cpu_wrap;
    const float s3_gpu_ms = gpu_dual_freq_unwrap(gpu_phase1, gpu_phase2, gpu_depth_unwrap, gpu_wrap, n);
    cpu_depth_unwrap.resize(static_cast<size_t>(n)); cpu_wrap.resize(static_cast<size_t>(n));
    CpuTimer ct3; ct3.begin();
    dual_freq_unwrap_cpu(cpu_phase1.data(), cpu_phase2.data(), cpu_depth_unwrap.data(), cpu_wrap.data(), n);
    const double s3_cpu_ms = ct3.end_ms();
    int wrap_mismatches = 0; float worst_unwrap = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (gpu_wrap[static_cast<size_t>(i)] != cpu_wrap[static_cast<size_t>(i)]) { ++wrap_mismatches; continue; }
        worst_unwrap = std::max(worst_unwrap, std::fabs(gpu_depth_unwrap[static_cast<size_t>(i)] - cpu_depth_unwrap[static_cast<size_t>(i)]));
    }
    std::printf("[info] dual_freq_unwrap: %d/%d wrap-count mismatches, worst |ddepth|=%.3e m (agreeing pixels)\n",
                wrap_mismatches, n, static_cast<double>(worst_unwrap));
    const bool s3_pass = (wrap_mismatches <= kMaxWrapCountMismatch) && (worst_unwrap <= kTolUnwrapDepthM);
    all_ok &= s3_pass;
    std::printf("VERIFY: dual_freq_unwrap %s (wrap-count near-exact, depth within tol %.1e m on agreeing pixels)\n",
                s3_pass ? "PASS" : "FAIL", static_cast<double>(kTolUnwrapDepthM));

    // ======================= STAGE 4: CONFIDENCE MASK ==========================
    std::vector<unsigned char> gpu_conf, cpu_conf;
    const float s4_gpu_ms = gpu_confidence_mask(gpu_amp1, kDefaultAmplitudeFloor, gpu_conf, n);
    cpu_conf.resize(static_cast<size_t>(n));
    CpuTimer ct4; ct4.begin();
    confidence_mask_cpu(cpu_amp1.data(), kDefaultAmplitudeFloor, cpu_conf.data(), n);
    const double s4_cpu_ms = ct4.end_ms();
    int conf_mismatches = 0;
    for (int i = 0; i < n; ++i) if (gpu_conf[static_cast<size_t>(i)] != cpu_conf[static_cast<size_t>(i)]) ++conf_mismatches;
    std::printf("[info] confidence_mask: %d/%d GPU-vs-CPU flag mismatches\n", conf_mismatches, n);
    const bool s4_pass = conf_mismatches <= kMaxConfidenceMismatch;
    all_ok &= s4_pass;
    std::printf("VERIFY: confidence_mask %s (flag mismatches <= allowance %d)\n", s4_pass ? "PASS" : "FAIL", kMaxConfidenceMismatch);

    // ======================= STAGE 5: FLYING-PIXEL DETECT ======================
    std::vector<unsigned char> gpu_flying, cpu_flying;
    const float s5_gpu_ms = gpu_flying_pixel_detect(gpu_depth_unwrap, gpu_amp1, gpu_conf, gpu_flying, w, h);
    cpu_flying.resize(static_cast<size_t>(n));
    CpuTimer ct5; ct5.begin();
    flying_pixel_detect_cpu(cpu_depth_unwrap.data(), cpu_amp1.data(), cpu_conf.data(), cpu_flying.data(), w, h);
    const double s5_cpu_ms = ct5.end_ms();
    int flying_mismatches = 0;
    for (int i = 0; i < n; ++i) if (gpu_flying[static_cast<size_t>(i)] != cpu_flying[static_cast<size_t>(i)]) ++flying_mismatches;
    std::printf("[info] flying_pixel_detect: %d/%d GPU-vs-CPU flag mismatches\n", flying_mismatches, n);
    const bool s5_pass = flying_mismatches <= kMaxFlyingMismatch;
    all_ok &= s5_pass;
    std::printf("VERIFY: flying_pixel_detect %s (flag mismatches <= allowance %d)\n", s5_pass ? "PASS" : "FAIL", kMaxFlyingMismatch);

    // ======================= STAGE 6: BACK-PROJECTION ===========================
    std::vector<unsigned char> gpu_final_valid(static_cast<size_t>(n)), cpu_final_valid(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        gpu_final_valid[static_cast<size_t>(i)] = (gpu_conf[static_cast<size_t>(i)] && !gpu_flying[static_cast<size_t>(i)]) ? 1 : 0;
        cpu_final_valid[static_cast<size_t>(i)] = (cpu_conf[static_cast<size_t>(i)] && !cpu_flying[static_cast<size_t>(i)]) ? 1 : 0;
    }
    std::vector<float> gpu_xyz, cpu_xyz;
    const float s6_gpu_ms = gpu_backproject(gpu_depth_unwrap, gpu_final_valid, gpu_xyz, n);
    cpu_xyz.resize(static_cast<size_t>(n) * 3);
    CpuTimer ct6; ct6.begin();
    backproject_cpu(cpu_depth_unwrap.data(), cpu_final_valid.data(), cpu_xyz.data(), n);
    const double s6_cpu_ms = ct6.end_ms();
    int final_valid_mismatches = 0; float worst_xyz = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (gpu_final_valid[static_cast<size_t>(i)] != cpu_final_valid[static_cast<size_t>(i)]) { ++final_valid_mismatches; continue; }
        if (gpu_final_valid[static_cast<size_t>(i)]) {
            for (int c = 0; c < 3; ++c)
                worst_xyz = std::max(worst_xyz, std::fabs(gpu_xyz[static_cast<size_t>(i) * 3 + c] - cpu_xyz[static_cast<size_t>(i) * 3 + c]));
        }
    }
    std::printf("[info] backproject: %d/%d final-valid mismatches, worst |dxyz|=%.3e m (agreeing pixels)\n",
                final_valid_mismatches, n, static_cast<double>(worst_xyz));
    const bool s6_pass = (final_valid_mismatches <= kMaxFlyingMismatch) && (worst_xyz <= kTolXyzM);
    all_ok &= s6_pass;
    std::printf("VERIFY: backproject %s (final-valid near-exact, xyz within tol %.1e m)\n", s6_pass ? "PASS" : "FAIL", static_cast<double>(kTolXyzM));

    std::printf("[time] full pipeline (6 stages, n=%d px): CPU %.2f ms | GPU kernels %.3f ms | speed-up (teaching artifact) %.0fx\n",
                n, s1_cpu_ms + s2_cpu_ms + s3_cpu_ms + s4_cpu_ms + s5_cpu_ms + s6_cpu_ms,
                static_cast<double>(s1_gpu_ms_a + s1_gpu_ms_b + s2_gpu_ms + s3_gpu_ms + s4_gpu_ms + s5_gpu_ms + s6_gpu_ms),
                (s1_cpu_ms + s2_cpu_ms + s3_cpu_ms + s4_cpu_ms + s5_cpu_ms + s6_cpu_ms) /
                    std::max(1e-6, static_cast<double>(s1_gpu_ms_a + s1_gpu_ms_b + s2_gpu_ms + s3_gpu_ms + s4_gpu_ms + s5_gpu_ms + s6_gpu_ms)));

    // =========================================================================
    // GATES — every one checks decoded/reconstructed GPU results against the
    // SYNTHETIC GROUND TRUTH (a third, independent codebase — Python), not
    // against the CPU/GPU twins above (reference_cpu.cpp's header explains
    // why this independence matters).
    // =========================================================================

    // ---- GATE: phase_extraction ------------------------------------------------
    // Recovered freq1 phase vs the ANALYTIC truth phase (from truth_depth,
    // wrapped by kAmbig1M), scored on CLEAN pixels only (not a truth flying
    // pixel, and confidence-valid) — the population a real pipeline would
    // trust as "a single surface, well lit".
    double phase_extraction_mean = 0.0;
    {
        const float two_pi = 6.28318530717958647692f;
        double sum_err = 0.0; long cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (truth_flying[static_cast<size_t>(i)]) continue;
            if (gpu_conf[static_cast<size_t>(i)] == 0) continue;
            float frac = truth_depth[static_cast<size_t>(i)] / kAmbig1M;
            frac -= std::floor(frac);
            const float true_phase = frac * two_pi;
            const float d = std::fabs(angdiff(gpu_phase1[static_cast<size_t>(i)], true_phase));
            sum_err += d; ++cnt;
        }
        phase_extraction_mean = cnt > 0 ? sum_err / cnt : 1.0e9;
        std::printf("[info] phase_extraction: mean |dphi| = %.4f rad over %ld clean confident pixels\n", phase_extraction_mean, cnt);
        const bool gate = phase_extraction_mean <= static_cast<double>(kGatePhaseExtractionMeanBoundRad);
        all_ok &= gate;
        std::printf("GATE: phase_extraction %s (mean |dphi| vs analytic truth <= bound %.2f rad, clean pixels)\n",
                    gate ? "PASS" : "FAIL", static_cast<double>(kGatePhaseExtractionMeanBoundRad));
    }

    // ---- GATE: offset_invariance ------------------------------------------------
    // Add a constant offset to ALL FOUR freq1 taps (simulating extra ambient
    // IR) and re-decode: C3-C1 and C0-C2 are UNCHANGED by a uniform additive
    // offset (reference_cpu.cpp Stage 1 derives this), so phase should come
    // back bit-for-bit (up to float rounding) identical.
    static const float kOffsetTestCounts = 35.0f;
    float worst_offset_dphi = 0.0f;
    {
        std::vector<float> taps_offset(h_taps1.size());
        for (size_t i = 0; i < h_taps1.size(); ++i) taps_offset[i] = h_taps1[i] + kOffsetTestCounts;
        std::vector<float> phase_offset, amp_offset;
        gpu_extract_phase_amplitude(taps_offset, phase_offset, amp_offset, n);
        long cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (gpu_conf[static_cast<size_t>(i)] == 0) continue;
            const float d = std::fabs(angdiff(phase_offset[static_cast<size_t>(i)], gpu_phase1[static_cast<size_t>(i)]));
            worst_offset_dphi = std::max(worst_offset_dphi, d);
            ++cnt;
        }
        std::printf("[info] offset_invariance: max |dphi| = %.3e rad after adding %.0f counts to all 4 taps (n=%ld confident pixels)\n",
                    static_cast<double>(worst_offset_dphi), static_cast<double>(kOffsetTestCounts), cnt);
    }
    {
        const bool gate = worst_offset_dphi <= kGateOffsetInvarianceTolRad;
        all_ok &= gate;
        std::printf("GATE: offset_invariance %s (max |dphi| <= tol %.1e rad)\n", gate ? "PASS" : "FAIL", static_cast<double>(kGateOffsetInvarianceTolRad));
    }

    // ---- GATE: aliasing_demo ---------------------------------------------------
    // The designed aliasing demonstration: every background pixel whose true
    // depth exceeds kAmbig1M should show a GROSS error (>= half the
    // ambiguity range) in the naive single-frequency depth — proving the
    // wraparound problem genuinely exists on this scene before Stage 3 fixes
    // it.
    long alias_pop = 0, alias_gross = 0;
    for (int i = 0; i < n; ++i) {
        if (truth_surf[static_cast<size_t>(i)] != kSurfBackground) continue;
        if (truth_depth[static_cast<size_t>(i)] <= kAmbig1M) continue;
        ++alias_pop;
        const float err = std::fabs(gpu_depth_wrapped[static_cast<size_t>(i)] - truth_depth[static_cast<size_t>(i)]);
        if (err >= 0.5f * kAmbig1M) ++alias_gross;
    }
    const double alias_rate = alias_pop > 0 ? static_cast<double>(alias_gross) / alias_pop : 0.0;
    std::printf("[info] aliasing_demo: %ld/%ld far-background pixels (depth > %.3f m) show gross wrap error (>= %.3f m)\n",
                alias_gross, alias_pop, static_cast<double>(kAmbig1M), 0.5 * static_cast<double>(kAmbig1M));
    {
        const bool gate = alias_rate >= static_cast<double>(kGateAliasingFractionFloor);
        all_ok &= gate;
        std::printf("GATE: aliasing_demo %s (gross-wrap-error fraction >= floor %.0f%%, far-background pixels)\n",
                    gate ? "PASS" : "FAIL", 100.0 * static_cast<double>(kGateAliasingFractionFloor));
    }

    // ---- GATE: unwrap_recovery --------------------------------------------------
    double mean_unwrap_err = 0.0, wrap_correct_rate = 0.0;
    {
        double sum_unwrap_err = 0.0; long unwrap_n = 0; long unwrap_correct = 0;
        for (int i = 0; i < n; ++i) {
            if (gpu_conf[static_cast<size_t>(i)] == 0) continue;
            ++unwrap_n;
            const double err = std::fabs(static_cast<double>(gpu_depth_unwrap[static_cast<size_t>(i)]) - static_cast<double>(truth_depth[static_cast<size_t>(i)]));
            sum_unwrap_err += err;
            const int true_n1 = static_cast<int>(truth_depth[static_cast<size_t>(i)] / kAmbig1M);
            if (gpu_wrap[static_cast<size_t>(i)] == true_n1) ++unwrap_correct;
        }
        mean_unwrap_err = unwrap_n > 0 ? sum_unwrap_err / unwrap_n : 1.0e9;
        wrap_correct_rate = unwrap_n > 0 ? static_cast<double>(unwrap_correct) / unwrap_n : 0.0;
        std::printf("[info] unwrap_recovery: mean |err| = %.2f mm, wrap-count correct on %.2f%% of %ld confident pixels\n",
                    mean_unwrap_err * 1000.0, 100.0 * wrap_correct_rate, unwrap_n);
    }
    {
        const bool gate = (mean_unwrap_err <= static_cast<double>(kGateUnwrapMeanErrBoundM)) &&
                          (wrap_correct_rate >= static_cast<double>(kGateUnwrapWrapCorrectFloor));
        all_ok &= gate;
        std::printf("GATE: unwrap_recovery %s (mean |err| <= bound %.0f mm AND wrap-count correct rate >= floor %.0f%%)\n",
                    gate ? "PASS" : "FAIL", 1000.0 * static_cast<double>(kGateUnwrapMeanErrBoundM), 100.0 * static_cast<double>(kGateUnwrapWrapCorrectFloor));
    }

    // ---- GATE: flying_pixel ------------------------------------------------------
    // Precision/recall against make_synthetic.py's INDEPENDENT truth_flying
    // label (computed from sub-ray surface membership the detector never
    // sees — see that script's file header).
    long fp_tp = 0, fp_fp = 0, fp_fn = 0;
    for (int i = 0; i < n; ++i) {
        const bool flagged = gpu_flying[static_cast<size_t>(i)] != 0;
        const bool truth = truth_flying[static_cast<size_t>(i)] != 0;
        if (flagged && truth) ++fp_tp;
        else if (flagged && !truth) ++fp_fp;
        else if (!flagged && truth) ++fp_fn;
    }
    const double fp_precision = (fp_tp + fp_fp) > 0 ? static_cast<double>(fp_tp) / (fp_tp + fp_fp) : 0.0;
    const double fp_recall    = (fp_tp + fp_fn) > 0 ? static_cast<double>(fp_tp) / (fp_tp + fp_fn) : 0.0;
    std::printf("[info] flying_pixel: tp=%ld fp=%ld fn=%ld, precision=%.1f%% recall=%.1f%% (vs %ld true flying pixels)\n",
                fp_tp, fp_fp, fp_fn, 100.0 * fp_precision, 100.0 * fp_recall, fp_tp + fp_fn);
    {
        const bool gate = (fp_precision >= static_cast<double>(kGateFlyingPrecisionFloor)) &&
                          (fp_recall >= static_cast<double>(kGateFlyingRecallFloor));
        all_ok &= gate;
        std::printf("GATE: flying_pixel %s (precision >= floor %.0f%% AND recall >= floor %.0f%%, vs independent truth_flying)\n",
                    gate ? "PASS" : "FAIL", 100.0 * static_cast<double>(kGateFlyingPrecisionFloor), 100.0 * static_cast<double>(kGateFlyingRecallFloor));
    }

    // ---- [info] noise_scaling ------------------------------------------------
    // NOT a PASS/FAIL gate (deliberately — see the catalog task's own
    // "[info]" label): buckets clean, non-flying pixels by freq1 amplitude
    // and reports the empirical WRAPPED-depth-error std-dev per bucket,
    // testing the qualitative prediction of THEORY.md's derived
    // sigma_d ~ c/(4*pi*f) * sigma_phi/B law: higher amplitude => tighter
    // depth noise.
    //
    // WRAPAROUND CARE: single-frequency depth lives on a CIRCLE of
    // circumference kAmbig1M (depth 0 and depth kAmbig1M-epsilon are
    // physically adjacent, one phase-wrap apart). A pixel whose true
    // wrapped depth sits near that seam can show a huge NAIVE linear
    // difference (measured minus truth) from a TINY phase-noise draw that
    // merely pushed it across the seam — the same "opposite sides of the
    // 0/2pi seam" hazard angdiff() guards against for angles, applied here
    // to a depth expressed in meters. Wrapping the difference into
    // (-kAmbig1M/2, kAmbig1M/2] before accumulating statistics is what
    // makes this diagnostic measure PHASE noise, not seam artifacts.
    {
        struct Bucket { float lo, hi; double sum = 0.0, sumsq = 0.0; long cnt = 0; };
        Bucket buckets[3] = {
            {kDefaultAmplitudeFloor, 35.0f, 0, 0, 0},
            {35.0f, 50.0f, 0, 0, 0},
            {50.0f, 1.0e9f, 0, 0, 0},
        };
        for (int i = 0; i < n; ++i) {
            if (truth_flying[static_cast<size_t>(i)]) continue;
            const float a = gpu_amp1[static_cast<size_t>(i)];
            if (a < kDefaultAmplitudeFloor) continue;
            float frac = truth_depth[static_cast<size_t>(i)] / kAmbig1M;
            frac -= std::floor(frac);
            const float wrapped_truth = frac * kAmbig1M;
            double err = static_cast<double>(gpu_depth_wrapped[static_cast<size_t>(i)]) - static_cast<double>(wrapped_truth);
            // Wrap into (-kAmbig1M/2, kAmbig1M/2] — see the wraparound-care
            // comment above this block.
            while (err > 0.5 * static_cast<double>(kAmbig1M)) err -= static_cast<double>(kAmbig1M);
            while (err < -0.5 * static_cast<double>(kAmbig1M)) err += static_cast<double>(kAmbig1M);
            for (auto& b : buckets) {
                if (a >= b.lo && a < b.hi) { b.sum += err; b.sumsq += err * err; ++b.cnt; break; }
            }
        }
        for (const auto& b : buckets) {
            if (b.cnt < 5) continue;
            const double mean = b.sum / b.cnt;
            const double var = std::max(0.0, b.sumsq / b.cnt - mean * mean);
            const double sd = std::sqrt(var);
            std::printf("[info] noise_scaling: amplitude in [%.0f,%.0f) counts, n=%ld, depth-error std = %.2f mm "
                        "(THEORY.md predicts std shrinks as amplitude grows)\n",
                        static_cast<double>(b.lo), static_cast<double>(std::min(b.hi, 255.0f)), b.cnt, sd * 1000.0);
        }
    }

    // ---- reconstruction gates: bucket the (verified) GPU point cloud by
    // TRUTH surface label (grading only — never fed back into decoding) ----
    struct Pt { float x, y, z; };
    std::vector<Pt> bg_pts, sph_pts, box_pts;
    for (int i = 0; i < n; ++i) {
        if (!gpu_final_valid[static_cast<size_t>(i)]) continue;
        const Pt p{ gpu_xyz[static_cast<size_t>(i) * 3 + 0], gpu_xyz[static_cast<size_t>(i) * 3 + 1], gpu_xyz[static_cast<size_t>(i) * 3 + 2] };
        const unsigned char s = truth_surf[static_cast<size_t>(i)];
        if (s == kSurfBackground) bg_pts.push_back(p);
        else if (s == kSurfSphere) sph_pts.push_back(p);
        else if (s == kSurfBox) box_pts.push_back(p);
    }

    // ---- GATE: reconstruction_plane --------------------------------------------
    double plane_a = 0.0, plane_b = 0.0, plane_c = 0.0, plane_rms = 1.0e9;
    bool plane_fit_ok = false;
    {
        double Sxx = 0, Sxy = 0, Syy = 0, Sx = 0, Sy = 0, Sz = 0, Sxz = 0, Syz = 0, S1 = 0;
        for (const Pt& p : bg_pts) {
            const double X = p.x, Y = p.y, Z = p.z;
            Sxx += X * X; Sxy += X * Y; Syy += Y * Y; Sx += X; Sy += Y; Sz += Z; Sxz += X * Z; Syz += Y * Z; S1 += 1.0;
        }
        std::vector<std::vector<double>> A = { {S1, Sx, Sy}, {Sx, Sxx, Sxy}, {Sy, Sxy, Syy} };
        std::vector<double> b = { Sz, Sxz, Syz }, sol;
        plane_fit_ok = !bg_pts.empty() && solve_dense(A, b, sol);
        if (plane_fit_ok) {
            plane_a = sol[0]; plane_b = sol[1]; plane_c = sol[2];
            double ss = 0.0;
            for (const Pt& p : bg_pts) { const double r = p.z - (plane_a + plane_b * p.x + plane_c * p.y); ss += r * r; }
            plane_rms = std::sqrt(ss / bg_pts.size());
        }
    }
    std::printf("[info] reconstruction_plane: fitted Z = %.4f + %.4f*X + %.4f*Y, RMS residual = %.2f mm over %zu background points\n",
                plane_a, plane_b, plane_c, plane_rms * 1000.0, bg_pts.size());
    const bool gate_plane = plane_fit_ok && (plane_rms <= static_cast<double>(kGatePlaneRmsBoundM));
    all_ok &= gate_plane;
    std::printf("GATE: reconstruction_plane %s (RMS residual <= bound %.0f mm)\n",
                gate_plane ? "PASS" : "FAIL", 1000.0 * static_cast<double>(kGatePlaneRmsBoundM));

    // ---- GATE: reconstruction_sphere -------------------------------------------
    double sphere_radius_fit = -1.0;
    bool sphere_fit_ok = false;
    {
        std::vector<std::vector<double>> A(4, std::vector<double>(4, 0.0));
        std::vector<double> b(4, 0.0);
        for (const Pt& p : sph_pts) {
            const double row[4] = { 2.0 * p.x, 2.0 * p.y, 2.0 * p.z, 1.0 };
            const double rhs = static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y + static_cast<double>(p.z) * p.z;
            for (int r = 0; r < 4; ++r) { for (int c = 0; c < 4; ++c) A[static_cast<size_t>(r)][static_cast<size_t>(c)] += row[r] * row[c]; b[static_cast<size_t>(r)] += row[r] * rhs; }
        }
        std::vector<double> sol;
        sphere_fit_ok = !sph_pts.empty() && solve_dense(A, b, sol);
        if (sphere_fit_ok) {
            const double cx = sol[0], cy = sol[1], cz = sol[2], k = sol[3];
            const double r2 = k + cx * cx + cy * cy + cz * cz;
            sphere_fit_ok = r2 > 0.0;
            if (sphere_fit_ok) sphere_radius_fit = std::sqrt(r2);
        }
    }
    const double sphere_radius_truth = kv_get(params, "sphere_radius_m_truth", -1.0);
    const double sphere_pct_err = (sphere_fit_ok && sphere_radius_truth > 0.0)
        ? 100.0 * std::fabs(sphere_radius_fit - sphere_radius_truth) / sphere_radius_truth : 1.0e9;
    std::printf("[info] reconstruction_sphere: fitted radius = %.4f m (truth %.4f m), error %.2f%% over %zu points\n",
                sphere_radius_fit, sphere_radius_truth, sphere_pct_err, sph_pts.size());
    const bool gate_sphere = sphere_fit_ok && (sphere_pct_err <= static_cast<double>(kGateSphereRadiusPctBound));
    all_ok &= gate_sphere;
    std::printf("GATE: reconstruction_sphere %s (radius error <= bound %.0f%%)\n",
                gate_sphere ? "PASS" : "FAIL", static_cast<double>(kGateSphereRadiusPctBound));

    // ---- GATE: reconstruction_step ---------------------------------------------
    double step_height_fit = -1.0e9, step_pct_err = 1.0e9;
    {
        const double xc = 0.5 * (kv_get(params, "box_x_min_m", 0.0) + kv_get(params, "box_x_max_m", 0.0));
        const double yc = 0.5 * (kv_get(params, "box_y_min_m", 0.0) + kv_get(params, "box_y_max_m", 0.0));
        const double bg_z_at_box = plane_a + plane_b * xc + plane_c * yc;
        double sum_z = 0.0;
        for (const Pt& p : box_pts) sum_z += p.z;
        const double mean_box_z = box_pts.empty() ? 0.0 : sum_z / box_pts.size();
        step_height_fit = bg_z_at_box - mean_box_z;
        const double step_truth = kv_get(params, "step_height_m_truth", -1.0);
        step_pct_err = (step_truth > 0.0 && !box_pts.empty()) ? 100.0 * std::fabs(step_height_fit - step_truth) / step_truth : 1.0e9;
        std::printf("[info] reconstruction_step: recovered step height = %.4f m (truth %.4f m), error %.2f%% over %zu box points\n",
                    step_height_fit, step_truth, step_pct_err, box_pts.size());
    }
    const bool gate_step = plane_fit_ok && !box_pts.empty() && (step_pct_err <= static_cast<double>(kGateStepHeightPctBound));
    all_ok &= gate_step;
    std::printf("GATE: reconstruction_step %s (step-height error <= bound %.0f%%)\n",
                gate_step ? "PASS" : "FAIL", static_cast<double>(kGateStepHeightPctBound));

    // ---- GATE: dark_cohort -------------------------------------------------------
    // The cohort: background-surface pixels inside the committed low-
    // reflectivity patch footprint (params.csv, written by make_synthetic.py).
    // Requires the confidence mask to reject an overwhelming majority AND
    // that any survivors are not silently hallucinating wrong depth.
    const int dark_row_min = static_cast<int>(kv_get(params, "dark_patch_row_min", 0.0));
    const int dark_row_max = static_cast<int>(kv_get(params, "dark_patch_row_max", 0.0));
    const int dark_col_min = static_cast<int>(kv_get(params, "dark_patch_col_min", 0.0));
    const int dark_col_max = static_cast<int>(kv_get(params, "dark_patch_col_max", 0.0));
    long dark_cohort_n = 0, dark_rejected_n = 0, dark_survivor_violations = 0;
    for (int row = dark_row_min; row < dark_row_max; ++row) {
        for (int col = dark_col_min; col < dark_col_max; ++col) {
            const int i = row * w + col;
            if (truth_surf[static_cast<size_t>(i)] != kSurfBackground) continue;
            ++dark_cohort_n;
            if (gpu_conf[static_cast<size_t>(i)] == 0) { ++dark_rejected_n; continue; }
            if (gpu_final_valid[static_cast<size_t>(i)]) {
                const float dz = std::fabs(gpu_xyz[static_cast<size_t>(i) * 3 + 2] - truth_depth[static_cast<size_t>(i)]);
                if (dz > kGateDarkCohortSurvivorBoundM) ++dark_survivor_violations;
            }
        }
    }
    const double dark_reject_rate = dark_cohort_n > 0 ? static_cast<double>(dark_rejected_n) / dark_cohort_n : 0.0;
    std::printf("[info] dark_cohort: rejected %ld/%ld cohort pixels (%.1f%%) by amplitude mask; "
                "%ld survivor(s) exceeded the %.0f mm depth-sanity bound\n",
                dark_rejected_n, dark_cohort_n, 100.0 * dark_reject_rate,
                dark_survivor_violations, 1000.0 * static_cast<double>(kGateDarkCohortSurvivorBoundM));
    const bool gate_dark = (dark_reject_rate >= static_cast<double>(kGateDarkCohortRejectFloor)) && (dark_survivor_violations == 0);
    all_ok &= gate_dark;
    std::printf("GATE: dark_cohort %s (cohort rejection rate >= floor %.0f%% AND zero survivor depth-sanity violations)\n",
                gate_dark ? "PASS" : "FAIL", 100.0 * static_cast<double>(kGateDarkCohortRejectFloor));

    // =========================================================================
    // ARTIFACTS
    // =========================================================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifacts_ok = true;

    artifacts_ok &= write_pgm(out_dir + "/tap_sample_f1.pgm", tap_sample_f1_bytes, w, h);
    std::printf("ARTIFACT: wrote demo/out/tap_sample_f1.pgm (freq1 %.0f MHz, tap 0 / 0 deg)\n", static_cast<double>(kFreq1Hz / 1.0e6f));

    artifacts_ok &= write_pgm(out_dir + "/tap_sample_f2.pgm", tap_sample_f2_bytes, w, h);
    std::printf("ARTIFACT: wrote demo/out/tap_sample_f2.pgm (freq2 %.0f MHz, tap 0 / 0 deg)\n", static_cast<double>(kFreq2Hz / 1.0e6f));

    {
        std::vector<unsigned char> all_valid_wrapped(static_cast<size_t>(n), 1);   // display every pixel's wrapped depth, valid or not
        auto img = depth_to_pgm(gpu_depth_wrapped, all_valid_wrapped, 0.0f, kAmbig1M, n);
        artifacts_ok &= write_pgm(out_dir + "/wrapped_depth_f1.pgm", img, w, h);
        std::printf("ARTIFACT: wrote demo/out/wrapped_depth_f1.pgm (single-frequency depth, freq1 -- the aliasing visual: repeating bands on the far wall)\n");
    }
    {
        std::vector<unsigned char> all_valid(static_cast<size_t>(n), 1);
        auto img = depth_to_pgm(gpu_depth_unwrap, all_valid, 0.0f, kMaxSceneDepthM, n);
        artifacts_ok &= write_pgm(out_dir + "/unwrapped_depth.pgm", img, w, h);
        std::printf("ARTIFACT: wrote demo/out/unwrapped_depth.pgm (dual-frequency unwrapped depth, full scene range)\n");
    }
    {
        std::vector<unsigned char> mask(static_cast<size_t>(n), 0);
        for (int i = 0; i < n; ++i) mask[static_cast<size_t>(i)] = gpu_flying[static_cast<size_t>(i)] ? 255 : 0;
        artifacts_ok &= write_pgm(out_dir + "/flying_pixel_mask.pgm", mask, w, h);
        std::printf("ARTIFACT: wrote demo/out/flying_pixel_mask.pgm (white = flagged flying pixel)\n");
    }

    long cloud_points = 0;
    {
        std::ofstream f(out_dir + "/point_cloud.csv");
        artifacts_ok &= f.is_open();
        if (f.is_open()) {
            // surface_id_truth: from the SYNTHETIC ground truth, included only
            // to color/label this visualization artifact — never consumed by
            // the decode pipeline itself (README "Limitations"). This cloud is
            // AFTER flying-pixel removal (gpu_final_valid) -- the "clean" cloud.
            f << "x_m,y_m,z_m,surface_id_truth\n";
            for (int i = 0; i < n; ++i) {
                if (!gpu_final_valid[static_cast<size_t>(i)]) continue;
                f << gpu_xyz[static_cast<size_t>(i) * 3 + 0] << ',' << gpu_xyz[static_cast<size_t>(i) * 3 + 1] << ','
                  << gpu_xyz[static_cast<size_t>(i) * 3 + 2] << ',' << static_cast<int>(truth_surf[static_cast<size_t>(i)]) << '\n';
                ++cloud_points;
            }
        }
    }
    std::printf("ARTIFACT: wrote demo/out/point_cloud.csv (%ld points, after flying-pixel removal)\n", cloud_points);

    // Two orthographic X-Z profile renders (X horizontal, Z vertical/inverted
    // so CLOSER surfaces draw HIGHER -- 01.19's convention): BEFORE flying-
    // pixel removal (confidence-valid only -- flying pixels visibly "hang"
    // between the foreground objects and the far wall) and AFTER (final_valid
    // -- the same scatter with the flying pixels gone). The "money shot" pair.
    auto render_profile = [&](const std::vector<unsigned char>& mask) {
        const int pw = 240, ph = 160;
        std::vector<unsigned char> img(static_cast<size_t>(pw) * ph, 0);
        float xmin = 1e9f, xmax = -1e9f, zmin = 1e9f, zmax = -1e9f;
        for (int i = 0; i < n; ++i) {
            if (!mask[static_cast<size_t>(i)]) continue;
            const int col = i % w;
            const float dx = (static_cast<float>(col) + 0.5f - kCamCx) / kCamFx;
            const float z = gpu_depth_unwrap[static_cast<size_t>(i)];
            const float x = dx * z;
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }
        const float xr = std::max(1e-6f, xmax - xmin), zr = std::max(1e-6f, zmax - zmin);
        for (int i = 0; i < n; ++i) {
            if (!mask[static_cast<size_t>(i)]) continue;
            const int col = i % w;
            const float dx = (static_cast<float>(col) + 0.5f - kCamCx) / kCamFx;
            const float z = gpu_depth_unwrap[static_cast<size_t>(i)];
            const float x = dx * z;
            int px = static_cast<int>((x - xmin) / xr * (pw - 1));
            int py = static_cast<int>((z - zmin) / zr * (ph - 1));   // closer (smaller Z) -> smaller row (up, "in front")
            px = px < 0 ? 0 : (px >= pw ? pw - 1 : px);
            py = py < 0 ? 0 : (py >= ph ? ph - 1 : py);
            img[static_cast<size_t>(py) * pw + px] = 255;
        }
        return std::make_pair(img, std::make_pair(pw, ph));
    };
    {
        auto result = render_profile(gpu_conf);   // BEFORE removal: confidence-valid only (flying pixels still present)
        artifacts_ok &= write_ppm_gray(out_dir + "/profile_view_before.ppm", result.first, result.second.first, result.second.second);
        std::printf("ARTIFACT: wrote demo/out/profile_view_before.ppm (X-Z profile BEFORE flying-pixel removal -- look for points hanging between the objects and the wall)\n");
    }
    {
        auto result = render_profile(gpu_final_valid);   // AFTER removal
        artifacts_ok &= write_ppm_gray(out_dir + "/profile_view_after.ppm", result.first, result.second.first, result.second.second);
        std::printf("ARTIFACT: wrote demo/out/profile_view_after.ppm (X-Z profile AFTER flying-pixel removal -- the same scene, cleaned up)\n");
    }

    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        artifacts_ok &= f.is_open();
        if (f.is_open()) {
            f << "name,value,verdict\n";
            f << "phase_extraction_mean_err_rad," << phase_extraction_mean << "," << (phase_extraction_mean <= kGatePhaseExtractionMeanBoundRad ? "PASS" : "FAIL") << "\n";
            f << "offset_invariance_max_dphi_rad," << worst_offset_dphi << "," << (worst_offset_dphi <= kGateOffsetInvarianceTolRad ? "PASS" : "FAIL") << "\n";
            f << "aliasing_demo_gross_rate," << alias_rate << "," << (alias_rate >= kGateAliasingFractionFloor ? "PASS" : "FAIL") << "\n";
            f << "unwrap_recovery_mean_err_m," << mean_unwrap_err << ",\n";
            f << "unwrap_recovery_wrap_correct_rate," << wrap_correct_rate << ","
              << ((mean_unwrap_err <= kGateUnwrapMeanErrBoundM && wrap_correct_rate >= kGateUnwrapWrapCorrectFloor) ? "PASS" : "FAIL") << "\n";
            f << "flying_pixel_precision," << fp_precision << ",\n";
            f << "flying_pixel_recall," << fp_recall << "," << ((fp_precision >= kGateFlyingPrecisionFloor && fp_recall >= kGateFlyingRecallFloor) ? "PASS" : "FAIL") << "\n";
            f << "reconstruction_plane_rms_m," << plane_rms << "," << (gate_plane ? "PASS" : "FAIL") << "\n";
            f << "reconstruction_sphere_radius_m," << sphere_radius_fit << "," << (gate_sphere ? "PASS" : "FAIL") << "\n";
            f << "reconstruction_step_height_m," << step_height_fit << "," << (gate_step ? "PASS" : "FAIL") << "\n";
            f << "dark_cohort_reject_rate," << dark_reject_rate << "," << (gate_dark ? "PASS" : "FAIL") << "\n";
        }
    }
    std::printf("ARTIFACT: wrote demo/out/gates_metrics.csv (11 rows)\n");

    all_ok &= artifacts_ok;
    if (!artifacts_ok) std::printf("[info] one or more artifact writes failed - see ARTIFACT lines above\n");

    if (all_ok)
        std::printf("RESULT: PASS (all 6 GPU-vs-CPU verifications and 9 independent gates passed)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for which check failed)\n");
    return all_ok ? 0 : 1;
}
