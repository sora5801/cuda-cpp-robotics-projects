// ===========================================================================
// main.cu — entry point for project 01.19
//           Structured-light decoding (Gray code, phase shift) for 3D
//           scanners (Gray-code + phase-shift HYBRID scanner)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Load the committed pattern stack (20 PGMs: 7 Gray-code bit planes x
//      {direct,inverse}, 4 phase-shift steps, white/black references) and
//      ground truth (continuous projector column, depth, surface id) from
//      data/sample/, asserting params.csv matches the compiled kernels.cuh
//      contract (catches a Python/C++ drift loudly instead of silently).
//   2. Run all FIVE pipeline stages (gray decode -> phase decode -> hybrid
//      combine -> triangulate -> the gray-vs-binary boundary stress test) on
//      BOTH the GPU and an independent CPU oracle, and VERIFY agreement.
//   3. Run EIGHT independent gates that check the decoded/reconstructed
//      results against the synthetic ground truth (not against each other —
//      see reference_cpu.cpp's file header for why that independence
//      matters): gray_decode, hybrid_subpixel, gray_vs_binary,
//      phase_ambient_invariance, three reconstruction gates
//      (plane/sphere/step), and dark_stripe_honesty.
//   4. Write six labeled artifacts to demo/out/ (two sample patterns, the
//      decoded column map, the confidence map, the point cloud, an
//      orthographic profile render, and a metrics CSV) and print a final
//      PASS/FAIL RESULT.
//
// Output contract (load-bearing!) — same as every project in this repo:
// "[demo]"/"[info]"/"[time]" lines are NOT diffed (device names and timings
// vary by machine and run); "PROBLEM:", "SCENARIO:", "VERIFY:", "GATE:",
// "ARTIFACT:", "RESULT:" lines ARE diffed verbatim by demo/run_demo.*
// against demo/expected_output.txt. Measured floating-point numbers
// therefore live on "[info]" lines; the paired "VERIFY:"/"GATE:" line
// carries only the PASS/FAIL verdict and the fixed tolerance/floor (never
// the measured value) — exactly 08.01's pattern, applied to eight gates
// instead of one. Change a stable line here => update
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

// ---------------------------------------------------------------------------
// Verification tolerances and gate floors/bounds. Every one of these is a
// FIXED CONSTANT printed in the stable "VERIFY:"/"GATE:" lines (never the
// measured value — see the file header). Chosen with margin below/above
// what this project's committed sample actually measures (documented in
// data/README.md "How the sample was tuned" and in THEORY.md "How we verify
// correctness") — the repo's "measured-then-margined gate" convention.
// ---------------------------------------------------------------------------
static const float  kTolPhaseRad          = 1.0e-3f;   // phase_decode GPU-vs-CPU (atan2f/sqrtf ULP drift)
static const float  kTolConfidence        = 5.0e-2f;   // phase_decode confidence GPU-vs-CPU (counts)
static const float  kTolHybridCol         = 2.0e-3f;   // hybrid_combine GPU-vs-CPU (projector columns)
static const float  kTolXyzM              = 2.0e-3f;   // triangulate GPU-vs-CPU (meters)
static const int    kMaxValidFlagDrift    = 8;         // triangulate valid-flag near-exact allowance

// Every floor/bound below is a "measured-then-margined gate" (repo
// convention, e.g. project 01.17): tuned against ACTUAL numbers measured on
// the committed sample (recorded in data/README.md "How the sample was
// tuned"), then set with visible margin so the gate is meaningful (not
// vacuous) but not brittle to the ordinary run-to-run determinism this
// project already guarantees (fixed seeds, no cuRAND — CLAUDE.md §12).
static const float  kGateGrayExactFloor        = 0.90f;   // gray_decode: exact-match rate, confident pixels (measured ~97.8%)
static const float  kGateHybridMeanBoundCols   = 0.15f;   // hybrid_subpixel: mean |err| bound, columns (measured ~0.071)
static const float  kGateHybridImproveFloor    = 2.5f;    // hybrid_subpixel: improvement over Gray-only (measured ~3.8x)
static const float  kGateBinaryFactorFloor     = 5.0f;    // gray_vs_binary: binary_rate >= floor * gray_rate (measured ~31x)
static const float  kGateBinaryRateFloor       = 0.03f;   // gray_vs_binary: binary_rate itself must be real (measured ~7.9%)
static const float  kAmbientTestOffsetCounts   = 40.0f;   // phase_ambient_invariance: added to all 4 frames
static const float  kGateAmbientTolRad         = 1.0e-4f; // phase_ambient_invariance: max |delta phi| bound (measured ~0)
static const float  kGatePlaneRmsBoundM        = 0.010f;  // reconstruction_plane: RMS residual bound, m (measured ~3.8 mm)
static const float  kGateSphereRadiusPctBound  = 4.0f;    // reconstruction_sphere: |err|/truth bound, % (measured ~1.4%)
static const float  kGateStepHeightPctBound    = 3.0f;    // reconstruction_step: |err|/truth bound, % (measured ~0.45%)
static const float  kGateDarkStripeRejectFloor = 0.95f;   // dark_stripe_honesty: cohort rejection-rate floor (measured ~99.1%)
static const float  kGateDarkStripeSurvivorBoundM = 0.06f;// dark_stripe_honesty: surviving-pixel depth sanity (m)

// ===========================================================================
// Small host-side utilities. All plain C++17 (no <filesystem> — see
// util/paths.h "Why not std::filesystem"); no CUDA here.
// ===========================================================================

// xorshift32 + Box-Muller: the repo's portable deterministic RNG, used here
// ONLY to draw the boundary-stress test's probe positions and per-bit noise
// (the main pipeline's noise is baked into the committed PGMs — see
// scripts/make_synthetic.py). Same algorithm as 08.01's main.cu.
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
static inline float uniform01(uint32_t& state)      // (0,1] — never 0, safe for log()
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}
static inline float draw_gaussian(uint32_t& state, float sigma)
{
    const double u1 = static_cast<double>(uniform01(state));
    const double u2 = static_cast<double>(uniform01(state));
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    return sigma * static_cast<float>(z);
}

// ---- PGM (P5) reader --------------------------------------------------------
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
// PPM (P6): used ONLY for the orthographic profile render (demo/out/
// profile_view.ppm) — an RGB file with R=G=B (grayscale content, PPM
// container) purely because the artifact spec names it a PPM.
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
// header): every scanner/code constant params.csv records must equal the
// compiled kernels.cuh contract, or the committed data no longer describes
// the scanner this program models. Prints readable mismatches to stderr.
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
    check_int("proj_cols", kProjCols);
    check_float("proj_fx_px", kProjFx, 1.0e-3f);
    check_float("proj_cx_px", kProjCx, 1.0e-3f);
    check_float("baseline_m", kBaselineM, 1.0e-4f);
    check_int("gray_bits", kGrayBits);
    check_int("phase_steps", kPhaseSteps);
    check_float("phase_period_cols", kPhasePeriodCols, 1.0e-4f);
    return ok;
}

// ---- small dense linear solve (double precision, partial pivoting) --------
// Solves A x = b for an n x n system. Used ONLY by the reconstruction gates
// below for their one-off plane (3x3) and sphere (4x4) normal-equations
// solves — O(1) work done ONCE per gate on a few thousand ALREADY-
// TRIANGULATED points, not a per-pixel parallel workload, so it deliberately
// stays on the host: there is no exploitable parallelism at this scale
// (THEORY.md "The GPU mapping" makes this explicit — not everything needs a
// kernel, and forcing one here would teach nothing).
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
// Needed because phase_ambient_invariance compares two [0,2pi)-convention
// angles that could sit on opposite sides of the 0/2pi seam even when
// physically identical (THEORY.md "Numerical considerations").
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
// Written as free functions (not inlined into main()) so the SAME wrapper
// serves both the primary pipeline run and the phase_ambient_invariance
// gate's second, perturbed-input call to phase decode.
// ===========================================================================
static float gpu_gray_decode(const std::vector<float>& direct, const std::vector<float>& inverse,
                             std::vector<int>& gray_col, int n)
{
    float* d_direct = nullptr; float* d_inverse = nullptr; int* d_gray_col = nullptr;
    CUDA_CHECK(cudaMalloc(&d_direct, direct.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inverse, inverse.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray_col, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_direct, direct.data(), direct.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse, inverse.data(), inverse.size() * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_gray_decode(d_direct, d_inverse, d_gray_col, n);
    const float ms = t.end_ms();
    gray_col.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(gray_col.data(), d_gray_col, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_direct)); CUDA_CHECK(cudaFree(d_inverse)); CUDA_CHECK(cudaFree(d_gray_col));
    return ms;
}

static float gpu_phase_decode(const std::vector<float>& phase, std::vector<float>& phase_out,
                              std::vector<float>& confidence, int n)
{
    float* d_phase = nullptr; float* d_phase_out = nullptr; float* d_confidence = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phase, phase.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phase_out, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_confidence, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_phase, phase.data(), phase.size() * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_phase_decode(d_phase, d_phase_out, d_confidence, n);
    const float ms = t.end_ms();
    phase_out.resize(static_cast<size_t>(n)); confidence.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(phase_out.data(), d_phase_out, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(confidence.data(), d_confidence, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_phase)); CUDA_CHECK(cudaFree(d_phase_out)); CUDA_CHECK(cudaFree(d_confidence));
    return ms;
}

static float gpu_hybrid_combine(const std::vector<int>& gray_col, const std::vector<float>& phase,
                                const std::vector<float>& confidence, float floor,
                                std::vector<float>& hybrid_col, std::vector<unsigned char>& valid, int n)
{
    int* d_gray_col = nullptr; float* d_phase = nullptr; float* d_confidence = nullptr;
    float* d_hybrid_col = nullptr; unsigned char* d_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gray_col, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_phase, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_confidence, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hybrid_col, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_gray_col, gray_col.data(), static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phase, phase.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_confidence, confidence.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_hybrid_combine(d_gray_col, d_phase, d_confidence, floor, d_hybrid_col, d_valid, n);
    const float ms = t.end_ms();
    hybrid_col.resize(static_cast<size_t>(n)); valid.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(hybrid_col.data(), d_hybrid_col, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(valid.data(), d_valid, static_cast<size_t>(n) * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_gray_col)); CUDA_CHECK(cudaFree(d_phase)); CUDA_CHECK(cudaFree(d_confidence));
    CUDA_CHECK(cudaFree(d_hybrid_col)); CUDA_CHECK(cudaFree(d_valid));
    return ms;
}

static float gpu_triangulate(const std::vector<float>& hybrid_col, const std::vector<unsigned char>& valid,
                             std::vector<float>& xyz, std::vector<unsigned char>& point_valid, int n)
{
    float* d_hybrid_col = nullptr; unsigned char* d_valid = nullptr;
    float* d_xyz = nullptr; unsigned char* d_point_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hybrid_col, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_point_valid, static_cast<size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_hybrid_col, hybrid_col.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valid, valid.data(), static_cast<size_t>(n) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_triangulate(d_hybrid_col, d_valid, d_xyz, d_point_valid, n);
    const float ms = t.end_ms();
    xyz.resize(static_cast<size_t>(n) * 3); point_valid.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(xyz.data(), d_xyz, static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(point_valid.data(), d_point_valid, static_cast<size_t>(n) * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hybrid_col)); CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_xyz)); CUDA_CHECK(cudaFree(d_point_valid));
    return ms;
}

static float gpu_boundary_stress(const std::vector<float>& true_x, const std::vector<float>& noise,
                                 std::vector<int>& decoded_gray, std::vector<int>& decoded_binary, int n)
{
    float* d_true_x = nullptr; float* d_noise = nullptr;
    int* d_decoded_gray = nullptr; int* d_decoded_binary = nullptr;
    CUDA_CHECK(cudaMalloc(&d_true_x, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_noise, noise.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_decoded_gray, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_decoded_binary, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_true_x, true_x.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_noise, noise.data(), noise.size() * sizeof(float), cudaMemcpyHostToDevice));
    GpuTimer t; t.begin();
    launch_boundary_stress(d_true_x, d_noise, d_decoded_gray, d_decoded_binary, n);
    const float ms = t.end_ms();
    decoded_gray.resize(static_cast<size_t>(n)); decoded_binary.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(decoded_gray.data(), d_decoded_gray, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(decoded_binary.data(), d_decoded_binary, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_true_x)); CUDA_CHECK(cudaFree(d_noise));
    CUDA_CHECK(cudaFree(d_decoded_gray)); CUDA_CHECK(cudaFree(d_decoded_binary));
    return ms;
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
            std::fprintf(stderr, "usage: %s [--data <dir with pattern PGMs + truth + params.csv>]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Structured-light decoding: Gray code + phase shift hybrid scanner (project 01.19)\n");
    print_device_info();
    std::printf("PROBLEM: structured-light scan, camera %dx%d px, projector %d cols "
                "(Gray N=%d + %d-step phase, period %.0f cols), baseline %.3f m, FP32\n",
                kCamW, kCamH, kProjCols, kGrayBits, kPhaseSteps,
                static_cast<double>(kPhasePeriodCols), static_cast<double>(kBaselineM));

    const int n = kNPix;

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
    std::printf("SCENARIO: synthetic scene (tilted background plane + sphere + box step-edge), "
                "dark-albedo stripe cohort, seed %d [synthetic]\n",
                static_cast<int>(kv_get(params, "seed", -1.0)));

    // ---- load the 20-pattern stack -----------------------------------------
    std::vector<float> h_gray_direct(static_cast<size_t>(kGrayBits) * n);
    std::vector<float> h_gray_inverse(static_cast<size_t>(kGrayBits) * n);
    std::vector<unsigned char> gray_direct_pgm_bytes;   // kept for the sample artifact (bit 3)
    for (int bit = 0; bit < kGrayBits; ++bit) {
        char fname[64];
        std::snprintf(fname, sizeof(fname), "gray_b%d_direct.pgm", bit);
        const std::string p = find_data_file(data_dir_override, argv[0], fname);
        PgmImage img = p.empty() ? PgmImage{} : read_pgm(p);
        if (!img.ok || img.w != kCamW || img.h != kCamH) {
            std::printf("SCENARIO: NOT FOUND - data/sample/%s missing or malformed\n", fname);
            std::printf("RESULT: FAIL (sample data missing)\n");
            return 1;
        }
        for (int pix = 0; pix < n; ++pix) h_gray_direct[static_cast<size_t>(bit) * n + pix] = static_cast<float>(img.data[static_cast<size_t>(pix)]);
        if (bit == 3) gray_direct_pgm_bytes = img.data;   // the demo's labeled "one Gray pattern" artifact

        std::snprintf(fname, sizeof(fname), "gray_b%d_inverse.pgm", bit);
        const std::string pi = find_data_file(data_dir_override, argv[0], fname);
        PgmImage imgi = pi.empty() ? PgmImage{} : read_pgm(pi);
        if (!imgi.ok || imgi.w != kCamW || imgi.h != kCamH) {
            std::printf("SCENARIO: NOT FOUND - data/sample/%s missing or malformed\n", fname);
            std::printf("RESULT: FAIL (sample data missing)\n");
            return 1;
        }
        for (int pix = 0; pix < n; ++pix) h_gray_inverse[static_cast<size_t>(bit) * n + pix] = static_cast<float>(imgi.data[static_cast<size_t>(pix)]);
    }

    std::vector<float> h_phase(static_cast<size_t>(kPhaseSteps) * n);
    std::vector<unsigned char> phase_s0_pgm_bytes;
    for (int k = 0; k < kPhaseSteps; ++k) {
        char fname[32];
        std::snprintf(fname, sizeof(fname), "phase_s%d.pgm", k);
        const std::string p = find_data_file(data_dir_override, argv[0], fname);
        PgmImage img = p.empty() ? PgmImage{} : read_pgm(p);
        if (!img.ok || img.w != kCamW || img.h != kCamH) {
            std::printf("SCENARIO: NOT FOUND - data/sample/%s missing or malformed\n", fname);
            std::printf("RESULT: FAIL (sample data missing)\n");
            return 1;
        }
        for (int pix = 0; pix < n; ++pix) h_phase[static_cast<size_t>(k) * n + pix] = static_cast<float>(img.data[static_cast<size_t>(pix)]);
        if (k == 0) phase_s0_pgm_bytes = img.data;
    }

    // ref_white/ref_black: a real per-pixel DYNAMIC RANGE diagnostic (the
    // real-world calibration role make_synthetic.py's file header and
    // THEORY.md describe) — printed, not gated.
    {
        const std::string pw = find_data_file(data_dir_override, argv[0], "ref_white.pgm");
        const std::string pb = find_data_file(data_dir_override, argv[0], "ref_black.pgm");
        PgmImage white = pw.empty() ? PgmImage{} : read_pgm(pw);
        PgmImage black = pb.empty() ? PgmImage{} : read_pgm(pb);
        if (!white.ok || !black.ok || white.w != kCamW || white.h != kCamH || black.w != kCamW || black.h != kCamH) {
            std::printf("SCENARIO: NOT FOUND - data/sample/ref_white.pgm or ref_black.pgm missing or malformed\n");
            std::printf("RESULT: FAIL (sample data missing)\n");
            return 1;
        }
        double sum = 0.0; int cnt = 0;
        for (int pix = 0; pix < n; ++pix) {
            const int d = static_cast<int>(white.data[static_cast<size_t>(pix)]) - static_cast<int>(black.data[static_cast<size_t>(pix)]);
            if (d > 0) { sum += d; ++cnt; }
        }
        std::printf("[info] reference frames: mean dynamic range (white-black) = %.1f counts over %d pixels "
                    "(real scanners use this pair to calibrate per-pixel adaptive thresholds; see THEORY.md)\n",
                    cnt > 0 ? sum / cnt : 0.0, cnt);
    }

    // ---- load ground truth --------------------------------------------------
    std::vector<float> truth_col, truth_depth;
    std::vector<unsigned char> truth_surf;
    {
        const std::string pc = find_data_file(data_dir_override, argv[0], "truth_column.bin");
        const std::string pd = find_data_file(data_dir_override, argv[0], "truth_depth.bin");
        const std::string ps = find_data_file(data_dir_override, argv[0], "truth_surface.bin");
        if (pc.empty() || pd.empty() || ps.empty() ||
            !read_f32_array(pc, truth_col, static_cast<size_t>(n)) ||
            !read_f32_array(pd, truth_depth, static_cast<size_t>(n)) ||
            !read_u8_array(ps, truth_surf, static_cast<size_t>(n))) {
            std::printf("SCENARIO: NOT FOUND - data/sample truth_*.bin missing or malformed\n");
            std::printf("RESULT: FAIL (sample data missing)\n");
            return 1;
        }
    }

    bool all_ok = true;   // ANDed with every VERIFY/GATE below -> final RESULT

    // ======================= STAGE 1: GRAY DECODE ============================
    std::vector<int> gray_col_gpu, gray_col_cpu;
    const float gray_gpu_ms = gpu_gray_decode(h_gray_direct, h_gray_inverse, gray_col_gpu, n);
    gray_col_cpu.resize(static_cast<size_t>(n));
    CpuTimer ct1; ct1.begin();
    gray_decode_cpu(h_gray_direct.data(), h_gray_inverse.data(), gray_col_cpu.data(), n);
    const double gray_cpu_ms = ct1.end_ms();
    int gray_mismatches = 0;
    for (int i = 0; i < n; ++i) if (gray_col_gpu[static_cast<size_t>(i)] != gray_col_cpu[static_cast<size_t>(i)]) ++gray_mismatches;
    std::printf("[info] gray_decode: %d/%d GPU-vs-CPU mismatches\n", gray_mismatches, n);
    const bool gray_verify_pass = (gray_mismatches == 0);
    all_ok &= gray_verify_pass;
    std::printf("VERIFY: gray_decode %s (GPU vs CPU exact integer match)\n", gray_verify_pass ? "PASS" : "FAIL");

    // ======================= STAGE 2: PHASE DECODE ============================
    std::vector<float> phase_out_gpu, confidence_gpu, phase_out_cpu, confidence_cpu;
    const float phase_gpu_ms = gpu_phase_decode(h_phase, phase_out_gpu, confidence_gpu, n);
    phase_out_cpu.resize(static_cast<size_t>(n)); confidence_cpu.resize(static_cast<size_t>(n));
    CpuTimer ct2; ct2.begin();
    phase_decode_cpu(h_phase.data(), phase_out_cpu.data(), confidence_cpu.data(), n);
    const double phase_cpu_ms = ct2.end_ms();
    float worst_phase = 0.0f, worst_conf = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float dphi = std::fabs(angdiff(phase_out_gpu[static_cast<size_t>(i)], phase_out_cpu[static_cast<size_t>(i)]));
        const float dconf = std::fabs(confidence_gpu[static_cast<size_t>(i)] - confidence_cpu[static_cast<size_t>(i)]);
        if (dphi > worst_phase) worst_phase = dphi;
        if (dconf > worst_conf) worst_conf = dconf;
    }
    std::printf("[info] phase_decode: worst |dphi|=%.3e rad, worst |dconfidence|=%.3e counts\n",
                static_cast<double>(worst_phase), static_cast<double>(worst_conf));
    const bool phase_verify_pass = (worst_phase <= kTolPhaseRad) && (worst_conf <= kTolConfidence);
    all_ok &= phase_verify_pass;
    std::printf("VERIFY: phase_decode %s (GPU vs CPU within tol %.1e rad / %.1e counts)\n",
                phase_verify_pass ? "PASS" : "FAIL", static_cast<double>(kTolPhaseRad), static_cast<double>(kTolConfidence));

    // ======================= STAGE 3: HYBRID COMBINE ==========================
    std::vector<float> hybrid_col_gpu, hybrid_col_cpu;
    std::vector<unsigned char> valid_gpu, valid_cpu;
    const float hybrid_gpu_ms = gpu_hybrid_combine(gray_col_gpu, phase_out_gpu, confidence_gpu,
                                                   kDefaultConfidenceFloor, hybrid_col_gpu, valid_gpu, n);
    hybrid_col_cpu.resize(static_cast<size_t>(n)); valid_cpu.resize(static_cast<size_t>(n));
    CpuTimer ct3; ct3.begin();
    hybrid_combine_cpu(gray_col_cpu.data(), phase_out_cpu.data(), confidence_cpu.data(),
                       kDefaultConfidenceFloor, hybrid_col_cpu.data(), valid_cpu.data(), n);
    const double hybrid_cpu_ms = ct3.end_ms();
    int valid_mismatches = 0; float worst_hybrid = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (valid_gpu[static_cast<size_t>(i)] != valid_cpu[static_cast<size_t>(i)]) { ++valid_mismatches; continue; }
        if (valid_gpu[static_cast<size_t>(i)]) {
            const float d = std::fabs(hybrid_col_gpu[static_cast<size_t>(i)] - hybrid_col_cpu[static_cast<size_t>(i)]);
            if (d > worst_hybrid) worst_hybrid = d;
        }
    }
    std::printf("[info] hybrid_combine: %d/%d valid-flag mismatches, worst |dcol|=%.3e columns (matching-flag pixels)\n",
                valid_mismatches, n, static_cast<double>(worst_hybrid));
    const bool hybrid_verify_pass = (valid_mismatches == 0) && (worst_hybrid <= kTolHybridCol);
    all_ok &= hybrid_verify_pass;
    std::printf("VERIFY: hybrid_combine %s (valid-flag exact match, column within tol %.1e cols)\n",
                hybrid_verify_pass ? "PASS" : "FAIL", static_cast<double>(kTolHybridCol));

    // ======================= STAGE 4: TRIANGULATE =============================
    std::vector<float> xyz_gpu, xyz_cpu;
    std::vector<unsigned char> point_valid_gpu, point_valid_cpu;
    const float tri_gpu_ms = gpu_triangulate(hybrid_col_gpu, valid_gpu, xyz_gpu, point_valid_gpu, n);
    xyz_cpu.resize(static_cast<size_t>(n) * 3); point_valid_cpu.resize(static_cast<size_t>(n));
    CpuTimer ct4; ct4.begin();
    triangulate_cpu(hybrid_col_cpu.data(), valid_cpu.data(), xyz_cpu.data(), point_valid_cpu.data(), n);
    const double tri_cpu_ms = ct4.end_ms();
    int point_valid_mismatches = 0; float worst_xyz = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (point_valid_gpu[static_cast<size_t>(i)] != point_valid_cpu[static_cast<size_t>(i)]) { ++point_valid_mismatches; continue; }
        if (point_valid_gpu[static_cast<size_t>(i)]) {
            for (int c = 0; c < 3; ++c) {
                const float d = std::fabs(xyz_gpu[static_cast<size_t>(i) * 3 + c] - xyz_cpu[static_cast<size_t>(i) * 3 + c]);
                if (d > worst_xyz) worst_xyz = d;
            }
        }
    }
    std::printf("[info] triangulate: %d/%d point-valid mismatches (ULP-boundary allowance %d), worst |dxyz|=%.3e m\n",
                point_valid_mismatches, n, kMaxValidFlagDrift, static_cast<double>(worst_xyz));
    const bool tri_verify_pass = (point_valid_mismatches <= kMaxValidFlagDrift) && (worst_xyz <= kTolXyzM);
    all_ok &= tri_verify_pass;
    std::printf("VERIFY: triangulate %s (point-valid near-exact, xyz within tol %.1e m)\n",
                tri_verify_pass ? "PASS" : "FAIL", static_cast<double>(kTolXyzM));

    std::printf("[time] main pipeline (gray+phase+hybrid+triangulate, n=%d px): "
                "CPU %.2f ms | GPU kernels %.3f ms | speed-up (teaching artifact) %.0fx\n",
                n, gray_cpu_ms + phase_cpu_ms + hybrid_cpu_ms + tri_cpu_ms,
                static_cast<double>(gray_gpu_ms + phase_gpu_ms + hybrid_gpu_ms + tri_gpu_ms),
                (gray_cpu_ms + phase_cpu_ms + hybrid_cpu_ms + tri_cpu_ms) /
                    std::max(1e-6, static_cast<double>(gray_gpu_ms + phase_gpu_ms + hybrid_gpu_ms + tri_gpu_ms)));

    // ======================= STAGE 5: BOUNDARY STRESS =========================
    const int M = kBoundarySamples;
    std::vector<float> bx_true(static_cast<size_t>(M));
    std::vector<float> bx_noise(static_cast<size_t>(M) * 2 * kGrayBits);
    {
        uint32_t rng = 42u;   // repo canonical seed (CLAUDE.md §12); mixed immediately by xorshift32
        for (int i = 0; i < M; ++i) {
            bx_true[static_cast<size_t>(i)] = uniform01(rng) * static_cast<float>(kProjCols - 1);
            for (int k = 0; k < 2 * kGrayBits; ++k)
                bx_noise[static_cast<size_t>(i) * 2 * kGrayBits + static_cast<size_t>(k)] = draw_gaussian(rng, kBoundaryNoiseSigma);
        }
    }
    std::vector<int> dec_gray_gpu, dec_bin_gpu, dec_gray_cpu, dec_bin_cpu;
    const float bx_gpu_ms = gpu_boundary_stress(bx_true, bx_noise, dec_gray_gpu, dec_bin_gpu, M);
    dec_gray_cpu.resize(static_cast<size_t>(M)); dec_bin_cpu.resize(static_cast<size_t>(M));
    CpuTimer ct5; ct5.begin();
    boundary_stress_cpu(bx_true.data(), bx_noise.data(), dec_gray_cpu.data(), dec_bin_cpu.data(), M);
    const double bx_cpu_ms = ct5.end_ms();
    int bx_mismatches = 0;
    for (int i = 0; i < M; ++i)
        if (dec_gray_gpu[static_cast<size_t>(i)] != dec_gray_cpu[static_cast<size_t>(i)] ||
            dec_bin_gpu[static_cast<size_t>(i)] != dec_bin_cpu[static_cast<size_t>(i)]) ++bx_mismatches;
    std::printf("[info] boundary_stress: %d/%d GPU-vs-CPU decode mismatches\n", bx_mismatches, M);
    const bool bx_verify_pass = (bx_mismatches == 0);
    all_ok &= bx_verify_pass;
    std::printf("VERIFY: boundary_stress %s (GPU vs CPU exact integer match)\n", bx_verify_pass ? "PASS" : "FAIL");
    std::printf("[time] boundary stress (n=%d probes): CPU %.2f ms | GPU kernel %.3f ms | speed-up (teaching artifact) %.0fx\n",
                M, bx_cpu_ms, static_cast<double>(bx_gpu_ms), bx_cpu_ms / std::max(1e-6, static_cast<double>(bx_gpu_ms)));

    // =========================================================================
    // GATES — every one checks decoded/reconstructed results against the
    // SYNTHETIC GROUND TRUTH (a third, independent codebase — Python), not
    // against the CPU/GPU twins above. All numbers computed from the (now
    // GPU-vs-CPU-verified) GPU results. See reference_cpu.cpp's header for
    // why this independence matters.
    // =========================================================================

    // ---- GATE: gray_decode ---------------------------------------------------
    // exact-match rate against round(truth column), CONFIDENT pixels only
    // (confidence >= floor AND inside the projector's illuminated range) —
    // the population this pipeline actually trusts and forwards downstream.
    long gray_confident_n = 0, gray_confident_exact = 0;
    for (int i = 0; i < n; ++i) {
        if (truth_col[static_cast<size_t>(i)] < 0.0f) continue;                       // outside projector FOV
        if (confidence_gpu[static_cast<size_t>(i)] < kDefaultConfidenceFloor) continue; // masked, see hybrid_combine
        ++gray_confident_n;
        const int truth_round = static_cast<int>(std::lround(static_cast<double>(truth_col[static_cast<size_t>(i)])));
        if (gray_col_gpu[static_cast<size_t>(i)] == truth_round) ++gray_confident_exact;
    }
    const double gray_exact_rate = gray_confident_n > 0 ? static_cast<double>(gray_confident_exact) / gray_confident_n : 0.0;
    std::printf("[info] gray_decode: exact-match rate %.2f%% over %ld confident pixels\n",
                100.0 * gray_exact_rate, gray_confident_n);
    const bool gate_gray = gray_exact_rate >= static_cast<double>(kGateGrayExactFloor);
    all_ok &= gate_gray;
    std::printf("GATE: gray_decode %s (exact-match rate >= floor %.0f%%, confident pixels)\n",
                gate_gray ? "PASS" : "FAIL", 100.0 * static_cast<double>(kGateGrayExactFloor));

    // ---- GATE: hybrid_subpixel ------------------------------------------------
    double sum_err_hybrid = 0.0, sum_err_gray = 0.0; long hyb_n = 0;
    for (int i = 0; i < n; ++i) {
        if (truth_col[static_cast<size_t>(i)] < 0.0f) continue;
        if (!valid_gpu[static_cast<size_t>(i)]) continue;   // masked pixels carry no hybrid answer to score
        ++hyb_n;
        sum_err_hybrid += std::fabs(static_cast<double>(hybrid_col_gpu[static_cast<size_t>(i)]) - static_cast<double>(truth_col[static_cast<size_t>(i)]));
        sum_err_gray   += std::fabs(static_cast<double>(gray_col_gpu[static_cast<size_t>(i)])   - static_cast<double>(truth_col[static_cast<size_t>(i)]));
    }
    const double mean_err_hybrid = hyb_n > 0 ? sum_err_hybrid / hyb_n : 1.0e9;
    const double mean_err_gray   = hyb_n > 0 ? sum_err_gray   / hyb_n : 0.0;
    const double improvement = mean_err_hybrid > 1.0e-9 ? mean_err_gray / mean_err_hybrid : 0.0;
    std::printf("[info] hybrid_subpixel: mean |err| hybrid=%.4f cols, gray-only=%.4f cols, improvement factor=%.2fx (n=%ld)\n",
                mean_err_hybrid, mean_err_gray, improvement, hyb_n);
    const bool gate_hybrid = (mean_err_hybrid <= static_cast<double>(kGateHybridMeanBoundCols)) &&
                            (improvement >= static_cast<double>(kGateHybridImproveFloor));
    all_ok &= gate_hybrid;
    std::printf("GATE: hybrid_subpixel %s (mean |err| <= bound %.2f cols AND improvement over Gray-only >= floor %.1fx)\n",
                gate_hybrid ? "PASS" : "FAIL", static_cast<double>(kGateHybridMeanBoundCols), static_cast<double>(kGateHybridImproveFloor));

    // ---- GATE: gray_vs_binary --------------------------------------------------
    long err_gray_bx = 0, err_bin_bx = 0;
    for (int i = 0; i < M; ++i) {
        const int truth_round = static_cast<int>(std::lround(static_cast<double>(bx_true[static_cast<size_t>(i)])));
        if (std::abs(dec_gray_gpu[static_cast<size_t>(i)] - truth_round) > 1) ++err_gray_bx;
        if (std::abs(dec_bin_gpu[static_cast<size_t>(i)]  - truth_round) > 1) ++err_bin_bx;
    }
    const double rate_gray_bx = static_cast<double>(err_gray_bx) / M;
    const double rate_bin_bx  = static_cast<double>(err_bin_bx) / M;
    const double gvb_factor = rate_bin_bx / std::max(rate_gray_bx, 1.0e-6);
    std::printf("[info] gray_vs_binary: boundary error rate gray=%.2f%% binary=%.2f%% (factor %.1fx, n=%d probes, noise sigma %.2f)\n",
                100.0 * rate_gray_bx, 100.0 * rate_bin_bx, gvb_factor, M, static_cast<double>(kBoundaryNoiseSigma));
    const bool gate_gvb = (rate_bin_bx >= static_cast<double>(kGateBinaryFactorFloor) * std::max(rate_gray_bx, 1.0e-6)) &&
                         (rate_bin_bx >= static_cast<double>(kGateBinaryRateFloor));
    all_ok &= gate_gvb;
    std::printf("GATE: gray_vs_binary %s (binary error rate >= floor %.0fx gray's, and >= %.0f%% absolute)\n",
                gate_gvb ? "PASS" : "FAIL", static_cast<double>(kGateBinaryFactorFloor), 100.0 * static_cast<double>(kGateBinaryRateFloor));

    // ---- GATE: phase_ambient_invariance ----------------------------------------
    // Add a constant offset to ALL FOUR phase frames (simulating extra
    // ambient light) and re-decode: I1-I3 and I0-I2 are UNCHANGED by a
    // uniform additive offset (reference_cpu.cpp Stage 2 derives this), so
    // phi should come back bit-for-bit (up to float rounding) identical.
    std::vector<float> h_phase_ambient(h_phase.size());
    for (size_t i = 0; i < h_phase.size(); ++i) h_phase_ambient[i] = h_phase[i] + kAmbientTestOffsetCounts;
    std::vector<float> phase_out_ambient, confidence_ambient;
    gpu_phase_decode(h_phase_ambient, phase_out_ambient, confidence_ambient, n);
    float worst_ambient_dphi = 0.0f; long ambient_n = 0;
    for (int i = 0; i < n; ++i) {
        if (confidence_gpu[static_cast<size_t>(i)] < kDefaultConfidenceFloor) continue;   // score the trusted population
        const float d = std::fabs(angdiff(phase_out_ambient[static_cast<size_t>(i)], phase_out_gpu[static_cast<size_t>(i)]));
        if (d > worst_ambient_dphi) worst_ambient_dphi = d;
        ++ambient_n;
    }
    std::printf("[info] phase_ambient_invariance: max |dphi| = %.3e rad after adding %.0f counts to all 4 frames (n=%ld confident pixels)\n",
                static_cast<double>(worst_ambient_dphi), static_cast<double>(kAmbientTestOffsetCounts), ambient_n);
    const bool gate_ambient = worst_ambient_dphi <= kGateAmbientTolRad;
    all_ok &= gate_ambient;
    std::printf("GATE: phase_ambient_invariance %s (max |dphi| <= tol %.1e rad)\n",
                gate_ambient ? "PASS" : "FAIL", static_cast<double>(kGateAmbientTolRad));

    // ---- reconstruction gates: bucket the (verified) GPU point cloud by
    // TRUTH surface label (grading only — never fed back into decoding) ----
    struct Pt { float x, y, z; };
    std::vector<Pt> bg_pts, sph_pts, box_pts;
    for (int i = 0; i < n; ++i) {
        if (!point_valid_gpu[static_cast<size_t>(i)]) continue;
        const Pt p{ xyz_gpu[static_cast<size_t>(i) * 3 + 0], xyz_gpu[static_cast<size_t>(i) * 3 + 1], xyz_gpu[static_cast<size_t>(i) * 3 + 2] };
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

    // ---- GATE: dark_stripe_honesty ---------------------------------------------
    // The cohort: background-surface, in-projector-FOV pixels inside the
    // committed low-albedo stripe footprint (params.csv, written by
    // make_synthetic.py). "Detected, not hidden" (catalog bullet): we
    // require the confidence mask to REJECT most of this cohort, AND that
    // any survivors are not silently hallucinating wrong depth.
    const int stripe_row_min = static_cast<int>(kv_get(params, "dark_stripe_row_min", 0.0));
    const int stripe_row_max = static_cast<int>(kv_get(params, "dark_stripe_row_max", 0.0));
    const int stripe_col_min = static_cast<int>(kv_get(params, "dark_stripe_col_min", 0.0));
    const int stripe_col_max = static_cast<int>(kv_get(params, "dark_stripe_col_max", 0.0));
    long stripe_cohort_n = 0, stripe_rejected_n = 0, stripe_survivor_violations = 0;
    for (int row = stripe_row_min; row < stripe_row_max; ++row) {
        for (int col = stripe_col_min; col < stripe_col_max; ++col) {
            const int i = row * kCamW + col;
            if (truth_surf[static_cast<size_t>(i)] != kSurfBackground) continue;
            if (truth_col[static_cast<size_t>(i)] < 0.0f) continue;   // outside projector FOV: not this gate's concern
            ++stripe_cohort_n;
            if (confidence_gpu[static_cast<size_t>(i)] < kDefaultConfidenceFloor) {
                ++stripe_rejected_n;
                continue;
            }
            // Survivor: must not hallucinate a wildly wrong depth.
            if (point_valid_gpu[static_cast<size_t>(i)]) {
                const float dz = std::fabs(xyz_gpu[static_cast<size_t>(i) * 3 + 2] - truth_depth[static_cast<size_t>(i)]);
                if (dz > kGateDarkStripeSurvivorBoundM) ++stripe_survivor_violations;
            }
        }
    }
    const double stripe_reject_rate = stripe_cohort_n > 0 ? static_cast<double>(stripe_rejected_n) / stripe_cohort_n : 0.0;
    std::printf("[info] dark_stripe_honesty: rejected %ld/%ld cohort pixels (%.1f%%) by confidence mask; "
                "%ld survivor(s) exceeded the %.0f mm depth-sanity bound\n",
                stripe_rejected_n, stripe_cohort_n, 100.0 * stripe_reject_rate,
                stripe_survivor_violations, 1000.0 * static_cast<double>(kGateDarkStripeSurvivorBoundM));
    const bool gate_dark = (stripe_reject_rate >= static_cast<double>(kGateDarkStripeRejectFloor)) && (stripe_survivor_violations == 0);
    all_ok &= gate_dark;
    std::printf("GATE: dark_stripe_honesty %s (cohort rejection rate >= floor %.0f%% AND zero survivor depth-sanity violations)\n",
                gate_dark ? "PASS" : "FAIL", 100.0 * static_cast<double>(kGateDarkStripeRejectFloor));

    // =========================================================================
    // ARTIFACTS
    // =========================================================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifacts_ok = true;

    artifacts_ok &= write_pgm(out_dir + "/gray_pattern_sample.pgm", gray_direct_pgm_bytes, kCamW, kCamH);
    std::printf("ARTIFACT: wrote demo/out/gray_pattern_sample.pgm (Gray bit 3, direct illumination)\n");

    artifacts_ok &= write_pgm(out_dir + "/phase_pattern_sample.pgm", phase_s0_pgm_bytes, kCamW, kCamH);
    std::printf("ARTIFACT: wrote demo/out/phase_pattern_sample.pgm (phase step 0/4)\n");

    {
        std::vector<unsigned char> col_map(static_cast<size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            if (!valid_gpu[static_cast<size_t>(i)]) continue;   // stays 0 (black) = invalid, documented in demo/README.md
            const float norm = hybrid_col_gpu[static_cast<size_t>(i)] / static_cast<float>(kProjCols - 1);
            const int v = static_cast<int>(std::lround(static_cast<double>(norm) * 255.0));
            col_map[static_cast<size_t>(i)] = static_cast<unsigned char>(v < 1 ? 1 : (v > 255 ? 255 : v));   // clamp to [1,255]: 0 is reserved for "invalid"
        }
        artifacts_ok &= write_pgm(out_dir + "/decoded_column_map.pgm", col_map, kCamW, kCamH);
        std::printf("ARTIFACT: wrote demo/out/decoded_column_map.pgm (normalized hybrid column, 0=invalid)\n");
    }
    {
        std::vector<unsigned char> conf_map(static_cast<size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            const float v = confidence_gpu[static_cast<size_t>(i)];
            conf_map[static_cast<size_t>(i)] = static_cast<unsigned char>(v < 0.0f ? 0 : (v > 255.0f ? 255 : v));
        }
        artifacts_ok &= write_pgm(out_dir + "/confidence_map.pgm", conf_map, kCamW, kCamH);
        std::printf("ARTIFACT: wrote demo/out/confidence_map.pgm (modulation amplitude, intensity counts)\n");
    }

    long cloud_points = 0;
    {
        std::ofstream f(out_dir + "/point_cloud.csv");
        artifacts_ok &= f.is_open();
        if (f.is_open()) {
            // surface_id_truth: from the SYNTHETIC ground truth, included only
            // to color/label this visualization artifact — never consumed by
            // the decode/triangulation pipeline itself (README "Limitations").
            f << "x_m,y_m,z_m,surface_id_truth\n";
            for (int i = 0; i < n; ++i) {
                if (!point_valid_gpu[static_cast<size_t>(i)]) continue;
                f << xyz_gpu[static_cast<size_t>(i) * 3 + 0] << ',' << xyz_gpu[static_cast<size_t>(i) * 3 + 1] << ','
                  << xyz_gpu[static_cast<size_t>(i) * 3 + 2] << ',' << static_cast<int>(truth_surf[static_cast<size_t>(i)]) << '\n';
                ++cloud_points;
            }
        }
    }
    std::printf("ARTIFACT: wrote demo/out/point_cloud.csv (%ld points)\n", cloud_points);

    // Orthographic profile render (X horizontal, Z vertical/inverted so
    // CLOSER surfaces draw HIGHER — the sphere and box literally look like
    // bumps/steps poking up out of the background band; demo/README.md
    // walks a learner through reading this image).
    {
        const int pw = 240, ph = 160;
        std::vector<unsigned char> img(static_cast<size_t>(pw) * ph, 0);
        float xmin = 1e9f, xmax = -1e9f, zmin = 1e9f, zmax = -1e9f;
        for (int i = 0; i < n; ++i) {
            if (!point_valid_gpu[static_cast<size_t>(i)]) continue;
            xmin = std::min(xmin, xyz_gpu[static_cast<size_t>(i) * 3 + 0]);
            xmax = std::max(xmax, xyz_gpu[static_cast<size_t>(i) * 3 + 0]);
            zmin = std::min(zmin, xyz_gpu[static_cast<size_t>(i) * 3 + 2]);
            zmax = std::max(zmax, xyz_gpu[static_cast<size_t>(i) * 3 + 2]);
        }
        const float xr = std::max(1e-6f, xmax - xmin), zr = std::max(1e-6f, zmax - zmin);
        for (int i = 0; i < n; ++i) {
            if (!point_valid_gpu[static_cast<size_t>(i)]) continue;
            const float X = xyz_gpu[static_cast<size_t>(i) * 3 + 0], Z = xyz_gpu[static_cast<size_t>(i) * 3 + 2];
            int px = static_cast<int>((X - xmin) / xr * (pw - 1));
            int py = static_cast<int>((Z - zmin) / zr * (ph - 1));   // larger Z (farther) -> larger row (down);
                                                                       // smaller Z (closer: sphere/box) -> smaller row (up, a "bump")
            px = px < 0 ? 0 : (px >= pw ? pw - 1 : px);
            py = py < 0 ? 0 : (py >= ph ? ph - 1 : py);
            img[static_cast<size_t>(py) * pw + px] = 255;
        }
        artifacts_ok &= write_ppm_gray(out_dir + "/profile_view.ppm", img, pw, ph);
        std::printf("ARTIFACT: wrote demo/out/profile_view.ppm (orthographic X-Z profile: sphere bump + box step)\n");
    }

    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        artifacts_ok &= f.is_open();
        if (f.is_open()) {
            f << "name,value,verdict\n";
            f << "gray_decode_exact_match_rate," << gray_exact_rate << "," << (gate_gray ? "PASS" : "FAIL") << "\n";
            f << "hybrid_subpixel_mean_err_cols," << mean_err_hybrid << ",\n";
            f << "hybrid_subpixel_improvement_factor," << improvement << "," << (gate_hybrid ? "PASS" : "FAIL") << "\n";
            f << "gray_vs_binary_gray_rate," << rate_gray_bx << ",\n";
            f << "gray_vs_binary_binary_rate," << rate_bin_bx << "," << (gate_gvb ? "PASS" : "FAIL") << "\n";
            f << "phase_ambient_invariance_max_dphi_rad," << worst_ambient_dphi << "," << (gate_ambient ? "PASS" : "FAIL") << "\n";
            f << "reconstruction_plane_rms_m," << plane_rms << "," << (gate_plane ? "PASS" : "FAIL") << "\n";
            f << "reconstruction_sphere_radius_m," << sphere_radius_fit << "," << (gate_sphere ? "PASS" : "FAIL") << "\n";
            f << "reconstruction_step_height_m," << step_height_fit << "," << (gate_step ? "PASS" : "FAIL") << "\n";
            f << "dark_stripe_reject_rate," << stripe_reject_rate << "," << (gate_dark ? "PASS" : "FAIL") << "\n";
        }
    }
    std::printf("ARTIFACT: wrote demo/out/gates_metrics.csv (10 rows)\n");

    all_ok &= artifacts_ok;
    if (!artifacts_ok) std::printf("[info] one or more artifact writes failed - see ARTIFACT lines above\n");

    if (all_ok)
        std::printf("RESULT: PASS (all 5 GPU-vs-CPU verifications and 8 independent gates passed)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for which check failed)\n");
    return all_ok ? 0 : 1;
}
