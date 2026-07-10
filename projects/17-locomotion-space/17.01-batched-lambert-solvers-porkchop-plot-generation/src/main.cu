// ===========================================================================
// main.cu — entry point for project 17.01
//           Batched Lambert solvers + porkchop plot generation
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (two orbit
//      radii, the epoch window, the accepted time-of-flight band, and the
//      grid resolution) from data/sample/.
//   2. SOLVE: run the batched Lambert solver over the WHOLE grid_n*grid_n
//      grid on the GPU kernel AND the CPU oracle (this project has no
//      closed loop — the porkchop grid IS the one-shot computation, unlike
//      08.01's per-tick controller).
//   3. VERIFY (the §5 GPU-vs-CPU gate): every cell's status code must match
//      between the two paths, and OK cells' delta-v must agree within a
//      documented relative tolerance.
//   4. NaN-policy check: the fraction of ATTEMPTED cells (short-way,
//      valid-TOF cells that were not simply excluded by scope) that ended
//      up near-singular or non-converged must stay small (kernels.cuh's
//      NaN policy; the measured fraction is printed on an [info] line).
//   5. ANALYTIC check — VERIFICATION AGAINST PURE MATHEMATICS: for two
//      coplanar circular orbits, the delta-v-optimal transfer is PROVABLY
//      the Hohmann transfer (a textbook vis-viva result, re-derived in
//      THEORY.md, independent of everything above). The grid's own minimum
//      delta-v cell must land within a documented small window at-or-above
//      the closed-form Hohmann delta-v, and its time-of-flight must land
//      near the Hohmann half-period — a discretized search can only
//      APPROACH the continuous optimum, never beat it (THEORY.md derives
//      why, and why the two checks are genuinely independent: nothing in
//      step 5 depends on the Lambert solver being correct in the way steps
//      3-4 already checked — a Lambert bug that still produced SOME
//      self-consistent low-delta-v cell would still fail this gate if that
//      cell were not near the true physics).
//   6. ARTIFACTS: demo/out/porkchop.pgm (log-scaled delta-v, the classic
//      picture) and demo/out/minimum.csv (the winning cell vs. the Hohmann
//      ground truth) — CLAUDE.md §6.3, this project's result is inherently
//      visual.
//   7. Exit 0 only if VERIFY, the NaN-policy check, and ANALYTIC all pass.
//
// Output contract: stable lines are "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "NAN POLICY:", "ANALYTIC:", "ARTIFACT:", "RESULT:";
// "[info]"/"[time]" lines are unchecked. Change a stable line -> update
// demo/expected_output.txt in the same commit.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu files — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Scenario loading — the committed "task definition": both orbit radii, the
// shared epoch window, the accepted time-of-flight band, and the grid
// resolution. Strict, like every loader in this repo: unknown labels,
// short rows, or a missing required row abort rather than guessing.
// Rows: "LABEL,value" (six required rows; mu is NOT one of them — it is
// the canonical-units axiom mu=1, not scenario data — kernels.cuh).
// ---------------------------------------------------------------------------
struct ScenarioFile {
    LambertScenario sc{};
    bool loaded = false;
};

static ScenarioFile load_scenario(const std::string& path)
{
    ScenarioFile out;
    std::ifstream in(path);
    if (!in.is_open()) return out;

    bool have[6] = { false, false, false, false, false, false };
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (!std::getline(ss, cell, ',')) {
            std::fprintf(stderr, "scenario: short row for label '%s'\n", label.c_str());
            return ScenarioFile{};
        }
        if (label == "R1_AU")           { out.sc.r1_au      = std::strtof(cell.c_str(), nullptr); have[0] = true; }
        else if (label == "R2_AU")      { out.sc.r2_au      = std::strtof(cell.c_str(), nullptr); have[1] = true; }
        else if (label == "WINDOW_TU")  { out.sc.window_tu  = std::strtof(cell.c_str(), nullptr); have[2] = true; }
        else if (label == "MIN_TOF_TU") { out.sc.min_tof_tu = std::strtof(cell.c_str(), nullptr); have[3] = true; }
        else if (label == "MAX_TOF_TU") { out.sc.max_tof_tu = std::strtof(cell.c_str(), nullptr); have[4] = true; }
        else if (label == "GRID_N")     { out.sc.grid_n     = std::atoi(cell.c_str());            have[5] = true; }
        else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return ScenarioFile{};
        }
    }
    for (bool h : have) if (!h) {
        std::fprintf(stderr, "scenario: missing a required row (need R1_AU, R2_AU, WINDOW_TU, "
                             "MIN_TOF_TU, MAX_TOF_TU, GRID_N)\n");
        return ScenarioFile{};
    }
    // Sanity checks — a malformed scenario should abort loudly, not solve
    // silently over a nonsensical grid.
    if (!(out.sc.r1_au > 0.0f) || !(out.sc.r2_au > 0.0f) || !(out.sc.window_tu > 0.0f)
        || !(out.sc.min_tof_tu >= 0.0f) || !(out.sc.max_tof_tu > out.sc.min_tof_tu)
        || out.sc.grid_n < 2) {
        std::fprintf(stderr, "scenario: values fail sanity checks (radii/window must be positive, "
                             "0 <= min_tof < max_tof, grid_n >= 2)\n");
        return ScenarioFile{};
    }
    out.loaded = true;
    return out;
}

static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_scenario(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/lambert_scenario.csv");
    candidates.push_back("data/sample/lambert_scenario.csv");
    candidates.push_back("../data/sample/lambert_scenario.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

static bool ensure_dir(const std::string& path)
{
#ifdef _WIN32
    const int r = _mkdir(path.c_str());
#else
    const int r = mkdir(path.c_str(), 0755);
#endif
    return r == 0 || errno == EEXIST;
}

// ---------------------------------------------------------------------------
// Hohmann ground truth — the CLOSED-FORM optimum this project verifies
// against (THEORY.md derives every line from vis-viva). Deliberately a
// THIRD, independent code path: it never calls the Lambert solver, so it
// cannot share a bug with steps 2-4 above (main.cu file header).
// ---------------------------------------------------------------------------
struct HohmannGroundTruth {
    double a_lu;        // transfer ellipse semi-major axis, LU
    double tof_tu;       // half the transfer ellipse's period, TU
    double delta_v;      // total impulsive delta-v, LU/TU
};

static HohmannGroundTruth hohmann_ground_truth(double r1, double r2)
{
    HohmannGroundTruth h{};
    // A local double-precision pi: kernels.cuh's kPi is FP32 (the Lambert
    // solver's own working precision); this ground-truth path is
    // deliberately double so it never inherits the GPU/CPU kernel's own
    // rounding — an independent check needs independent arithmetic.
    const double pi = 3.14159265358979323846;
    h.a_lu = 0.5 * (r1 + r2);
    h.tof_tu = pi * std::pow(h.a_lu, 1.5);            // mu = 1: T = 2*pi*a^1.5, Hohmann = T/2
    const double v1c = std::sqrt(1.0 / r1);            // circular speed at r1 (mu = 1)
    const double v2c = std::sqrt(1.0 / r2);            // circular speed at r2
    const double vp  = std::sqrt(2.0 / r1 - 1.0 / h.a_lu);   // transfer-orbit speed at perihelion (r1)
    const double va  = std::sqrt(2.0 / r2 - 1.0 / h.a_lu);   // transfer-orbit speed at aphelion  (r2)
    h.delta_v = std::fabs(vp - v1c) + std::fabs(v2c - va);
    return h;
}

// ---------------------------------------------------------------------------
// PGM artifact — the porkchop picture. P5 (binary grayscale), no library
// needed (07.09's idiom). Row j = one arrival epoch, row 0 at the TOP of
// the image (standard top-down raster order — documented here because a
// textbook porkchop plot usually draws arrival date increasing UPWARD;
// this PGM is the mirror image of that convention along the vertical axis,
// a labeling choice only, not a computation difference — README exercise
// territory to reorient it).
//
//   status == kStatusOk cells: LOG-scaled delta-v, mapped so LOWER delta-v
//     (better transfers) render BRIGHTER — pixel in [40, 255], never 0, so
//     "valid but expensive" is always visually distinct from "no data".
//   every other status: pixel 0 (pure black) — the sentinel for "excluded"
//     (masked time-of-flight, long-way scope exclusion, near-singular, or
//     non-converged) — the black regions ARE the porkchop's characteristic
//     boundary shape (THEORY.md §the-math explains why it looks like that).
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, int width, int height, const std::vector<uint8_t>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

static bool write_porkchop_pgm(const std::string& path, int n,
                               const std::vector<float>& deltav, const std::vector<int>& status)
{
    const size_t total = static_cast<size_t>(n) * n;
    double dv_min = 1e300, dv_max = -1e300;
    for (size_t idx = 0; idx < total; ++idx) {
        if (status[idx] != kStatusOk) continue;
        const double dv = static_cast<double>(deltav[idx]);
        if (dv < dv_min) dv_min = dv;
        if (dv > dv_max) dv_max = dv;
    }
    if (dv_min > dv_max) return false;   // no OK cells at all — nothing to draw

    const double log_lo = std::log(dv_min);
    const double log_hi = std::log(dv_max > dv_min ? dv_max : dv_min * 1.0001 + 1e-9);
    std::vector<uint8_t> gray(total, 0);   // 0 = excluded sentinel (comment above)
    for (size_t idx = 0; idx < total; ++idx) {
        if (status[idx] != kStatusOk) continue;
        double norm = (std::log(static_cast<double>(deltav[idx])) - log_lo) / (log_hi - log_lo);
        norm = norm < 0.0 ? 0.0 : (norm > 1.0 ? 1.0 : norm);
        // Bright = cheap (low delta-v): pixel = 255 at norm=0, pixel = 40 at norm=1.
        gray[idx] = static_cast<uint8_t>(255.0 - norm * (255.0 - 40.0) + 0.5);
    }
    return write_pgm(path, n, n, gray);
}

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data lambert_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Batched Lambert solvers + porkchop plot generation (project 17.01)\n");
    print_device_info();

    // ---- scenario -------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/lambert_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    ScenarioFile sf = load_scenario(scenario_path);
    if (!sf.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    const LambertScenario sc = sf.sc;
    const int N = sc.grid_n;
    const size_t total = static_cast<size_t>(N) * N;

    std::printf("PROBLEM: batched universal-variable Lambert solve, %dx%d departure x arrival epoch "
               "grid, canonical units (mu=1, 1 LU=1 AU, 1 TU=58.132441 days), FP32\n", N, N);
    std::printf("SCENARIO: Earth-like r1=%.3f LU, Mars-like r2=%.3f LU, window=[0,%.1f) TU, "
               "TOF band (%.1f, %.1f) TU [synthetic, coplanar circular orbits]\n",
               static_cast<double>(sc.r1_au), static_cast<double>(sc.r2_au),
               static_cast<double>(sc.window_tu), static_cast<double>(sc.min_tof_tu),
               static_cast<double>(sc.max_tof_tu));

    // ---- device buffers + the one-shot batched solve ---------------------
    float* d_deltav = nullptr;
    int*   d_status = nullptr;
    CUDA_CHECK(cudaMalloc(&d_deltav, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_status, total * sizeof(int)));

    GpuTimer gt;
    gt.begin();
    launch_lambert_grid(sc, d_deltav, d_status);
    const float gpu_ms = gt.end_ms();

    std::vector<float> deltav_gpu(total);
    std::vector<int>   status_gpu(total);
    CUDA_CHECK(cudaMemcpy(deltav_gpu.data(), d_deltav, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(status_gpu.data(), d_status, total * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_deltav));
    CUDA_CHECK(cudaFree(d_status));

    std::vector<float> deltav_cpu(total);
    std::vector<int>   status_cpu(total);
    CpuTimer ct;
    ct.begin();
    lambert_grid_cpu(sc, deltav_cpu.data(), status_cpu.data());
    const double cpu_ms = ct.end_ms();

    std::printf("[time] full grid (%dx%d cells): CPU sequential %.1f ms | GPU kernel %.3f ms | "
               "speed-up %.0fx (teaching artifact; kernel only)\n",
               N, N, cpu_ms, static_cast<double>(gpu_ms),
               cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));

    // ======================= VERIFY STAGE ==================================
    // Every cell's status must match; OK cells' delta-v must agree within
    // rel tol 1e-3 (the §5 gate — same headroom-over-measured-ulp-noise
    // reasoning as 08.01/09.01: identical algorithm, only sinf/cosf/rsqrtf
    // vs std:: spellings differ between the two paths).
    long long status_mismatches = 0;
    long long dv_compared = 0;
    float worst_rel = 0.0f;
    for (size_t idx = 0; idx < total; ++idx) {
        if (status_gpu[idx] != status_cpu[idx]) { status_mismatches++; continue; }
        if (status_gpu[idx] != kStatusOk) continue;
        const float cg = deltav_gpu[idx], cc = deltav_cpu[idx];
        const float scale = std::fabs(cc) > 1.0f ? std::fabs(cc) : 1.0f;
        const float rel = std::fabs(cg - cc) / scale;
        if (rel > worst_rel) worst_rel = rel;
        dv_compared++;
    }
    // Bound justified in THEORY.md §how-we-verify-correctness: the eps-
    // singular classification boundary is a hard threshold on a
    // continuously-varying angle, so a handful of cells RIGHT on that
    // boundary can legitimately classify differently between GPU and CPU
    // due to ulp-level sinf/cosf/atan2f differences — this is not a bug,
    // it is what comparing two independently-rounded float pipelines
    // against a hard threshold always costs. The bound is generous headroom
    // over the measured count (see the [info] line).
    const long long kMaxStatusMismatches = 200;   // << 0.1% of 262144 cells
    const bool status_ok = status_mismatches <= kMaxStatusMismatches;
    const bool dv_ok = worst_rel <= 1e-3f;
    const bool verify_pass = status_ok && dv_ok;
    std::printf("[info] verify: %lld/%zu status mismatches (bound %lld), worst relative delta-v "
               "deviation %.3e over %lld matched OK cells\n",
               status_mismatches, total, kMaxStatusMismatches,
               static_cast<double>(worst_rel), dv_compared);
    std::printf("VERIFY: %s (GPU cell status/delta-v match CPU reference within documented tolerance)\n",
               verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting the porkchop plot)\n");
        return 1;
    }

    // ======================= NaN POLICY STAGE ===============================
    // Use the GPU's own results from here on (already verified equal to the
    // CPU oracle above) — this project's delivered product.
    long long n_masked = 0, n_long = 0, n_singular = 0, n_nonconv = 0, n_ok = 0;
    for (size_t idx = 0; idx < total; ++idx) {
        switch (status_gpu[idx]) {
            case kStatusOk:           n_ok++;       break;
            case kStatusMaskedTof:    n_masked++;   break;
            case kStatusLongWay:      n_long++;     break;
            case kStatusNearSingular: n_singular++; break;
            case kStatusNonConverged: n_nonconv++;  break;
            default: break;
        }
    }
    const long long attempted = n_ok + n_singular + n_nonconv;   // short-way, valid-TOF cells
    const double nan_fraction = attempted > 0
        ? static_cast<double>(n_singular + n_nonconv) / static_cast<double>(attempted) : 0.0;
    // Bound justified in THEORY.md: the near-singular ring is ~2*kEpsSingularRad
    // wide out of a pi-wide short-way angle range (~2.2% expected, measured
    // below); non-convergence should be ~0 given the validated bracket.
    // Generous headroom over that expectation:
    const double kMaxNanFraction = 0.06;
    const bool nan_ok = nan_fraction <= kMaxNanFraction;
    std::printf("[info] cell census: %lld masked-TOF, %lld long-way (scope), %lld near-singular, "
               "%lld non-converged, %lld ok — of %zu total; attempted (short-way, valid TOF) = %lld\n",
               n_masked, n_long, n_singular, n_nonconv, n_ok, total, attempted);
    std::printf("[info] NaN-policy fraction (near-singular + non-converged)/attempted = %.4f (bound %.2f)\n",
               nan_fraction, kMaxNanFraction);
    std::printf("NAN POLICY: %s (degenerate/non-converged share of attempted cells stays within the "
               "documented bound)\n", nan_ok ? "PASS" : "FAIL");

    // ======================= ANALYTIC STAGE =================================
    // The pure-mathematics check (file header) — completely independent of
    // the Lambert solver's own internal consistency checked above.
    const HohmannGroundTruth hz = hohmann_ground_truth(static_cast<double>(sc.r1_au),
                                                        static_cast<double>(sc.r2_au));

    long long best_idx = -1;
    float best_dv = 0.0f;
    for (size_t idx = 0; idx < total; ++idx) {
        if (status_gpu[idx] != kStatusOk) continue;
        if (best_idx < 0 || deltav_gpu[idx] < best_dv) { best_idx = static_cast<long long>(idx); best_dv = deltav_gpu[idx]; }
    }

    bool analytic_ok = false;
    double dv_gap = 0.0, tof_gap = 0.0, t1_min = 0.0, t2_min = 0.0, tof_min = 0.0;
    if (best_idx >= 0) {
        const int i_min = static_cast<int>(best_idx % N);
        const int j_min = static_cast<int>(best_idx / N);
        const double dt = static_cast<double>(sc.window_tu) / N;
        t1_min = i_min * dt;
        t2_min = j_min * dt;
        tof_min = t2_min - t1_min;
        dv_gap = static_cast<double>(best_dv) - hz.delta_v;
        tof_gap = tof_min - hz.tof_tu;
        // A discretized search can only APPROACH the continuous optimum from
        // above (THEORY.md): a small negative slack absorbs FP32 rounding,
        // a generous positive bound absorbs the grid's finite resolution
        // (measured gap documented on the [info] line below).
        const double kDvGapLo = -2e-4, kDvGapHi = 0.01;      // LU/TU (~5% of the Hohmann value)
        const double kTofGapAbsBound = 1.0;                   // TU (~22% of the Hohmann TOF)
        analytic_ok = (dv_gap >= kDvGapLo) && (dv_gap <= kDvGapHi) && (std::fabs(tof_gap) <= kTofGapAbsBound);
    }
    std::printf("[info] Hohmann ground truth (closed form, vis-viva): a=%.6f LU, TOF=%.6f TU, "
               "delta_v=%.6f LU/TU\n", hz.a_lu, hz.tof_tu, hz.delta_v);
    if (best_idx >= 0) {
        std::printf("[info] grid minimum: t1=%.4f TU, t2=%.4f TU, TOF=%.4f TU, delta_v=%.6f LU/TU "
                   "(gap vs Hohmann: delta_v %+0.6f LU/TU, TOF %+0.4f TU)\n",
                   t1_min, t2_min, tof_min, static_cast<double>(best_dv), dv_gap, tof_gap);
    } else {
        std::printf("[info] grid minimum: NO OK CELLS FOUND\n");
    }
    std::printf("ANALYTIC: %s (grid minimum delta-v/TOF land within the documented window of the "
               "closed-form Hohmann optimum)\n", analytic_ok ? "PASS" : "FAIL");

    // ======================= ARTIFACTS ======================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) artifact_ok = write_porkchop_pgm(out_dir + "/porkchop.pgm", N, deltav_gpu, status_gpu);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/minimum.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "field,value,units\n";
            f << "t1_departure," << t1_min << ",TU\n";
            f << "t2_arrival," << t2_min << ",TU\n";
            f << "tof," << tof_min << ",TU\n";
            f << "delta_v_grid," << static_cast<double>(best_dv) << ",LU/TU\n";
            f << "delta_v_hohmann," << hz.delta_v << ",LU/TU\n";
            f << "tof_hohmann," << hz.tof_tu << ",TU\n";
            f << "delta_v_gap," << dv_gap << ",LU/TU\n";
            f << "tof_gap," << tof_gap << ",TU\n";
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/porkchop.pgm and demo/out/minimum.csv (%dx%d)\n", N, N);
    else {
        std::printf("ARTIFACT: FAILED to write demo/out files\n");
    }

    // ---- verdict ------------------------------------------------------------
    const bool success = verify_pass && nan_ok && analytic_ok && artifact_ok;
    if (success)
        std::printf("RESULT: PASS (GPU/CPU agree; NaN policy within bound; grid minimum tracks the "
                   "closed-form Hohmann optimum; artifacts written)\n");
    else
        std::printf("RESULT: FAIL (see the gate lines above)\n");
    return success ? 0 : 1;
}
