// ===========================================================================
// main.cu — entry point for project 34.03
//           Ergodic control: spectral multiscale coverage (SMC)
//           (teaching core: single 2-D first-order agent, K=32x32 modes,
//           reduced-scope [R&D] implementation — README §Limitations)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Build the target density phi(x) on a trapezoidal grid (two Gaussian
//      hotspots + a uniform washout, normalized to integrate to 1).
//   2. TRANSFORM gate: compute its 1024 Fourier coefficients phi_k TWO
//      independent ways — the GPU DCT-via-cuFFT pipeline, and a direct,
//      no-FFT CPU cosine projection — and require they agree tightly.
//   3. VERIFY gate (the §5 GPU-vs-CPU gate): drive a short window of the
//      SMC control law, feeding the identical position sequence through
//      both the GPU per-mode kernel and its CPU twin, and require the
//      per-mode c_k/Bx/By agree.
//   4. CLOSED LOOP: run the real 60 s / 6000-step SMC controller from a
//      fresh state, logging the trajectory and the ergodic metric.
//   5. ERGODICITY gate: the metric must decrease by a documented factor and
//      be windowed-monotone (transient upticks allowed and measured).
//   6. COVERAGE gate: the fraction of run-time spent in each hotspot's
//      basin must approach that basin's TARGET probability mass.
//   7. NEGATIVE CONTROL: replay the identical-length run with a
//      lawnmower (boustrophedon) sweep instead of SMC control, and require
//      its final ergodic metric is worse by a documented factor — proof
//      the controller is doing something, not just that any dense path
//      would pass.
//   8. Write four artifacts (trajectory.csv, ergodic_metric.csv,
//      target_phi.pgm, empirical_coverage.pgm) and print the final verdict.
//
// Determinism: NOTHING in this project is randomized — no RNG anywhere.
// Every run on a given machine reproduces bit-for-bit; across DIFFERENT GPU
// architectures, cuFFT/device-libm rounding can differ in the last few
// bits (THEORY.md §numerics), which is why stable output lines carry
// PASS/FAIL verdicts and never the raw measured numbers themselves —
// exactly the discipline 08.01 uses for its noise-sensitive verdicts.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "SCENARIO:", the five gate lines ("TRANSFORM:", "VERIFY:", "ERGODICITY:",
// "COVERAGE:", "NEGATIVE-CONTROL:"), "ARTIFACT:", and "RESULT:". "[info]"/
// "[time]" lines are NOT diffed. Change a stable line -> update
// demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// lambda_weight — host copy of the Sobolev weight Lambda_k = (1+||k||^2)^-1.5
// used wherever main.cu needs to score a c_k/phi_k pair (epsilon logging,
// gate checks) WITHOUT going through a full smc_step call. A third,
// intentional duplicate of kernels.cu/reference_cpu.cpp's identical
// one-liner — the whole formula is cheap enough that duplicating it beats
// threading a shared header function through three different call sites
// with three different (device/host/host) linkage needs (kernels.cuh's
// file header explains why this repo does not fight that boundary).
// ---------------------------------------------------------------------------
static double lambda_weight(int k1, int k2)
{
    const double kk = static_cast<double>(k1 * k1 + k2 * k2);
    return 1.0 / ((1.0 + kk) * std::sqrt(1.0 + kk));
}

// ergodic_metric — epsilon = sum_k Lambda_k * (c_k - phi_k)^2, the scalar
// this whole project drives toward zero. O(kNumModes) = O(1024), negligible.
static double ergodic_metric(const std::vector<double>& c, const std::vector<double>& phi_k)
{
    double eps = 0.0;
    for (int k1 = 0; k1 < kK; ++k1)
        for (int k2 = 0; k2 < kK; ++k2) {
            const int idx = k1 * kK + k2;
            const double diff = c[static_cast<size_t>(idx)] - phi_k[static_cast<size_t>(idx)];
            eps += lambda_weight(k1, k2) * diff * diff;
        }
    return eps;
}

// ---------------------------------------------------------------------------
// build_phi_grid — evaluate target_phi_shape on an N x N trapezoidal grid
// and NORMALIZE it so the trapezoidal integral over [0,1]^2 is exactly 1
// (required: phi_(0,0) must equal f_(0,0)=1 exactly, kernels.cuh's file
// header explains why). Returns the normalization constant Z used, so
// callers that re-evaluate target_phi_shape at OTHER resolutions (the
// visualization grid) can apply the identical scaling.
// ---------------------------------------------------------------------------
static double build_phi_grid(std::vector<double>& grid)
{
    const int N = kPhiGridN;
    const double h = 1.0 / static_cast<double>(N - 1);
    grid.resize(static_cast<size_t>(N) * N);

    // Pass 1: raw shape + trapezoidal integral (for normalization).
    double raw_integral = 0.0;
    for (int n = 0; n < N; ++n) {
        const double wn = (n == 0 || n == N - 1) ? 0.5 : 1.0;
        const double x1 = static_cast<double>(n) * h;
        for (int m = 0; m < N; ++m) {
            const double wm = (m == 0 || m == N - 1) ? 0.5 : 1.0;
            const double x2 = static_cast<double>(m) * h;
            const double v = target_phi_shape(x1, x2);
            grid[static_cast<size_t>(n) * N + m] = v;
            raw_integral += wn * wm * v;
        }
    }
    raw_integral *= h * h;

    // Pass 2: normalize in place so the grid integrates to exactly 1.
    for (double& v : grid) v /= raw_integral;
    return raw_integral;
}

// basin_mass — numerically integrate the (already normalized) phi grid over
// a disk of the given radius around (mux,muy), same trapezoidal weights the
// phi_k computation uses — the TARGET mass the COVERAGE gate compares
// against (README §Expected output; never a hand-typed number).
static double basin_mass(const std::vector<double>& grid, double mux, double muy, double radius)
{
    const int N = kPhiGridN;
    const double h = 1.0 / static_cast<double>(N - 1);
    double total = 0.0;
    for (int n = 0; n < N; ++n) {
        const double wn = (n == 0 || n == N - 1) ? 0.5 : 1.0;
        const double x1 = static_cast<double>(n) * h;
        for (int m = 0; m < N; ++m) {
            const double wm = (m == 0 || m == N - 1) ? 0.5 : 1.0;
            const double x2 = static_cast<double>(m) * h;
            const double dx = x1 - mux, dy = x2 - muy;
            if (dx * dx + dy * dy <= radius * radius)
                total += wn * wm * grid[static_cast<size_t>(n) * N + m];
        }
    }
    return total * h * h;
}

// ---------------------------------------------------------------------------
// lawnmower_position — the NEGATIVE CONTROL trajectory: a deterministic
// boustrophedon ("mow the lawn") sweep of kLawnRows horizontal rows, back
// and forth, IGNORING phi entirely. A pure function of the 1-based step
// index and the total step count — no dynamics, no controller, because a
// raster pattern's whole point is that it does not react to anything.
// THEORY.md §the problem derives why this is the natural "naive baseline"
// for non-uniform coverage (equal time everywhere, whatever phi says).
// ---------------------------------------------------------------------------
static constexpr int kLawnRows = 8;

static void lawnmower_position(int step_1based, int n_steps, double out[2])
{
    const double p = static_cast<double>(step_1based - 1) / static_cast<double>(n_steps - 1);   // in [0,1]
    double rowf = p * static_cast<double>(kLawnRows);
    int row = static_cast<int>(rowf);
    if (row >= kLawnRows) row = kLawnRows - 1;
    const double u = rowf - static_cast<double>(row);          // in [0,1): position along this row
    out[0] = (row % 2 == 0) ? u : 1.0 - u;                      // alternate sweep direction per row
    out[1] = (static_cast<double>(row) + 0.5) / static_cast<double>(kLawnRows);   // row centerline
}

// ---------------------------------------------------------------------------
// write_pgm — a plain ASCII (P2) portable graymap: human-diffable, no
// endianness or text/binary newline surprises on Windows. Every project
// that writes an image artifact in this repo documents its own writer;
// this one is deliberately the simplest possible (03.01's is the other
// worked example, also ASCII P2-style plain text).
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, int w, int h, const std::vector<unsigned char>& gray)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "P2\n" << w << ' ' << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        f << static_cast<int>(gray[static_cast<size_t>(i)]);
        f << ((i % w == w - 1) ? '\n' : ' ');
    }
    return f.good();
}

// ---------------------------------------------------------------------------
// Scenario loading — the committed "task definition" (CLAUDE.md §8: a tiny
// synthetic sample, no downloads). Rows: "X0,x1,x2" and "STEPS,n" — the
// target-density and controller CONSTANTS are documented in the same file
// as comments (and regenerated by scripts/make_synthetic.py) but the single
// source of truth actually compiled into the program is kernels.cuh, per
// this repo's "one place, never two" convention (CLAUDE.md §12).
// ---------------------------------------------------------------------------
struct Scenario {
    double x0[2] = { kX0_1, kX0_2 };
    int steps = kNSteps;
    bool loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_x0 = false, have_steps = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "X0") {
            for (int i = 0; i < 2; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short X0 row\n"); return Scenario{}; }
                sc.x0[i] = std::strtod(cell.c_str(), nullptr);
            }
            have_x0 = true;
        } else if (label == "STEPS") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short STEPS row\n"); return Scenario{}; }
            sc.steps = std::atoi(cell.c_str());
            have_steps = true;
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_x0 || !have_steps || sc.steps < kVerifyWindow + 2) {
        std::fprintf(stderr, "scenario: missing X0/STEPS, or STEPS too small\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/ergodic_scenario.csv");
    candidates.push_back("data/sample/ergodic_scenario.csv");
    candidates.push_back("../data/sample/ergodic_scenario.csv");
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

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data ergodic_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Ergodic control: spectral multiscale coverage (project 34.03)\n");
    print_device_info();
    std::printf("PROBLEM: SMC ergodic coverage, K=%dx%d=%d modes, T=%.1f s @ dt=%.3f s (%d steps), "
                "agent speed budget %.2f units/s, FP64\n",
                kK, kK, kNumModes, kTTotal, kDt, kNSteps, kVmax);

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/ergodic_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    std::printf("SCENARIO: start at (%.2f,%.2f); target = 2 Gaussian hotspots + uniform washout; %d steps [synthetic]\n",
                sc.x0[0], sc.x0[1], sc.steps);

    bool all_pass = true;

    // ---- 1) Build + normalize the target density grid -----------------------
    std::vector<double> phi_grid;
    const double phi_norm_z = build_phi_grid(phi_grid);   // raw-shape integral; target_phi_shape(x)/phi_norm_z == the true normalized density phi(x)
    const double mass1 = basin_mass(phi_grid, kMu1X, kMu1Y, kBasinRadius);
    const double mass2 = basin_mass(phi_grid, kMu2X, kMu2Y, kBasinRadius);
    std::printf("[info] target basin masses (radius %.2f): hotspot1=%.4f hotspot2=%.4f (numerically integrated, not hand-typed)\n",
                kBasinRadius, mass1, mass2);

    // ---- persistent device buffers (allocated ONCE, freed at the very end) --
    double *d_phi_grid = nullptr, *d_phi_k = nullptr, *d_S = nullptr;
    double *d_c = nullptr, *d_Bx = nullptr, *d_By = nullptr;
    const size_t grid_bytes = phi_grid.size() * sizeof(double);
    const size_t modes_bytes = static_cast<size_t>(kNumModes) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_phi_grid, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi_k, modes_bytes));
    CUDA_CHECK(cudaMalloc(&d_S, modes_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, modes_bytes));
    CUDA_CHECK(cudaMalloc(&d_Bx, modes_bytes));
    CUDA_CHECK(cudaMalloc(&d_By, modes_bytes));
    CUDA_CHECK(cudaMemcpy(d_phi_grid, phi_grid.data(), grid_bytes, cudaMemcpyHostToDevice));

    // ======================= TRANSFORM GATE ===================================
    // phi_k via the GPU DCT-via-cuFFT pipeline vs. an independent, no-FFT CPU
    // direct cosine projection — this project's transform-correctness check
    // (THEORY.md §How we verify correctness item 1).
    std::vector<double> phi_k_gpu(static_cast<size_t>(kNumModes));
    std::vector<double> phi_k_cpu(static_cast<size_t>(kNumModes));
    {
        GpuTimer gt;
        gt.begin();
        launch_build_phi_k(d_phi_grid, d_phi_k);
        const float gpu_ms = gt.end_ms();
        CUDA_CHECK(cudaMemcpy(phi_k_gpu.data(), d_phi_k, modes_bytes, cudaMemcpyDeviceToHost));

        CpuTimer ct;
        ct.begin();
        phi_k_direct_cpu(phi_grid.data(), phi_k_cpu.data());
        const double cpu_ms = ct.end_ms();

        double worst = 0.0;
        for (int i = 0; i < kNumModes; ++i) {
            const double scale = std::fabs(phi_k_cpu[static_cast<size_t>(i)]) > 1e-6
                                ? std::fabs(phi_k_cpu[static_cast<size_t>(i)]) : 1e-6;
            const double d = std::fabs(phi_k_gpu[static_cast<size_t>(i)] - phi_k_cpu[static_cast<size_t>(i)]) / scale;
            if (d > worst) worst = d;
        }
        const bool pass = worst <= 1e-6;
        all_pass = all_pass && pass;
        std::printf("[info] transform: worst relative |phi_k_gpu - phi_k_cpu| = %.3e over %d modes\n", worst, kNumModes);
        std::printf("[time] phi_k: CPU direct-sum %.2f ms | GPU DCT-via-cuFFT %.3f ms (both ONE-SHOT setup calls)\n",
                    cpu_ms, static_cast<double>(gpu_ms));
        std::printf("TRANSFORM: %s (DCT-via-cuFFT phi_k matches the independent no-FFT CPU cosine projection, rel tol 1e-6)\n",
                    pass ? "PASS" : "FAIL");
    }

    // ======================= VERIFY GATE (the §5 gate) ========================
    // Drive ONE trajectory for kVerifyWindow steps using the GPU control law;
    // at every step, feed the SAME position into the CPU twin (its own,
    // independent running-sum state) and compare c_k/Bx/By.
    {
        CUDA_CHECK(cudaMemset(d_S, 0, modes_bytes));
        std::vector<double> S_cpu(static_cast<size_t>(kNumModes), 0.0);
        std::vector<double> c_gpu(static_cast<size_t>(kNumModes)), Bx_gpu(static_cast<size_t>(kNumModes)), By_gpu(static_cast<size_t>(kNumModes));
        std::vector<double> c_cpu(static_cast<size_t>(kNumModes)), Bx_cpu(static_cast<size_t>(kNumModes)), By_cpu(static_cast<size_t>(kNumModes));

        double x[2] = { sc.x0[0], sc.x0[1] };
        double worst = 0.0;
        double gpu_ms_total = 0.0;

        for (int step = 1; step <= kVerifyWindow; ++step) {
            GpuTimer gt;
            gt.begin();
            launch_smc_step(x[0], x[1], d_phi_k, d_S, step, d_c, d_Bx, d_By);
            gpu_ms_total += static_cast<double>(gt.end_ms());
            CUDA_CHECK(cudaMemcpy(c_gpu.data(), d_c, modes_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(Bx_gpu.data(), d_Bx, modes_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(By_gpu.data(), d_By, modes_bytes, cudaMemcpyDeviceToHost));

            smc_step_cpu(x[0], x[1], phi_k_cpu.data(), S_cpu.data(), step, c_cpu.data(), Bx_cpu.data(), By_cpu.data());

            for (int i = 0; i < kNumModes; ++i) {
                auto rel = [](double a, double b) {
                    const double scale = std::fabs(b) > 1e-6 ? std::fabs(b) : 1e-6;
                    return std::fabs(a - b) / scale;
                };
                worst = std::max({ worst, rel(c_gpu[static_cast<size_t>(i)], c_cpu[static_cast<size_t>(i)]),
                                   rel(Bx_gpu[static_cast<size_t>(i)], Bx_cpu[static_cast<size_t>(i)]),
                                   rel(By_gpu[static_cast<size_t>(i)], By_cpu[static_cast<size_t>(i)]) });
            }

            // Advance the SHARED trajectory using the GPU path's own control —
            // both paths are then compared again next step on the new x.
            double Bx = 0.0, By = 0.0;
            for (int i = 0; i < kNumModes; ++i) { Bx += Bx_gpu[static_cast<size_t>(i)]; By += By_gpu[static_cast<size_t>(i)]; }
            const double Bnorm = std::sqrt(Bx * Bx + By * By);
            const double u1 = -kVmax * Bx / (Bnorm + kBEps);
            const double u2 = -kVmax * By / (Bnorm + kBEps);
            integrate_agent_cpu(x, u1, u2, kDt);
        }

        const bool pass = worst <= 1e-6;
        all_pass = all_pass && pass;
        std::printf("[info] verify: worst relative deviation %.3e over %d steps x %d modes x {c,Bx,By}\n",
                    worst, kVerifyWindow, kNumModes);
        std::printf("[time] SMC step (K=%d): GPU kernel avg %.4f ms/step over %d steps\n",
                    kNumModes, gpu_ms_total / kVerifyWindow, kVerifyWindow);
        std::printf("VERIFY: %s (GPU per-mode update matches CPU reference over a %d-step window, rel tol 1e-6)\n",
                    pass ? "PASS" : "FAIL", kVerifyWindow);
        if (!pass) {
            std::printf("RESULT: FAIL (GPU/CPU SMC-step disagreement — fix before trusting the controller)\n");
            return 1;
        }
    }

    // ======================= CLOSED LOOP (the real run) =======================
    CUDA_CHECK(cudaMemset(d_S, 0, modes_bytes));
    double x[2] = { sc.x0[0], sc.x0[1] };

    std::vector<double> traj;                          // t,x1,x2,u1,u2 rows
    traj.reserve(static_cast<size_t>(sc.steps) * 5);
    std::vector<double> eps_series;                     // epsilon(t), one per step
    eps_series.reserve(static_cast<size_t>(sc.steps));
    std::vector<double> c_step(static_cast<size_t>(kNumModes)), Bx_step(static_cast<size_t>(kNumModes)), By_step(static_cast<size_t>(kNumModes));

    long long in_basin1 = 0, in_basin2 = 0;
    std::vector<unsigned int> vis_hist(static_cast<size_t>(kVisGrid) * kVisGrid, 0u);
    double loop_gpu_ms = 0.0;

    for (int step = 1; step <= sc.steps; ++step) {
        GpuTimer gt;
        gt.begin();
        launch_smc_step(x[0], x[1], d_phi_k, d_S, step, d_c, d_Bx, d_By);
        loop_gpu_ms += static_cast<double>(gt.end_ms());
        CUDA_CHECK(cudaMemcpy(c_step.data(), d_c, modes_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Bx_step.data(), d_Bx, modes_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(By_step.data(), d_By, modes_bytes, cudaMemcpyDeviceToHost));

        double Bx = 0.0, By = 0.0;
        for (int i = 0; i < kNumModes; ++i) { Bx += Bx_step[static_cast<size_t>(i)]; By += By_step[static_cast<size_t>(i)]; }
        const double Bnorm = std::sqrt(Bx * Bx + By * By);
        const double u1 = -kVmax * Bx / (Bnorm + kBEps);
        const double u2 = -kVmax * By / (Bnorm + kBEps);

        // Log the SAMPLE position (x BEFORE this step's move — the position
        // that produced this step's f_k(x) contribution) and the applied control.
        const double t_s = static_cast<double>(step - 1) * kDt;
        traj.push_back(t_s); traj.push_back(x[0]); traj.push_back(x[1]); traj.push_back(u1); traj.push_back(u2);
        eps_series.push_back(ergodic_metric(c_step, phi_k_cpu));

        const double d1x = x[0] - kMu1X, d1y = x[1] - kMu1Y;
        if (d1x * d1x + d1y * d1y <= kBasinRadius * kBasinRadius) ++in_basin1;
        const double d2x = x[0] - kMu2X, d2y = x[1] - kMu2Y;
        if (d2x * d2x + d2y * d2y <= kBasinRadius * kBasinRadius) ++in_basin2;

        int vc = static_cast<int>(x[0] * kVisGrid); if (vc >= kVisGrid) vc = kVisGrid - 1; if (vc < 0) vc = 0;
        int vr = static_cast<int>(x[1] * kVisGrid); if (vr >= kVisGrid) vr = kVisGrid - 1; if (vr < 0) vr = 0;
        vis_hist[static_cast<size_t>(vr) * kVisGrid + vc]++;

        integrate_agent_cpu(x, u1, u2, kDt);
    }

    const double eps_final = eps_series.back();
    const double frac1 = static_cast<double>(in_basin1) / static_cast<double>(sc.steps);
    const double frac2 = static_cast<double>(in_basin2) / static_cast<double>(sc.steps);
    std::printf("[info] final agent position: (%.4f, %.4f)\n", x[0], x[1]);
    std::printf("[info] final ergodic metric epsilon = %.6e\n", eps_final);
    std::printf("[info] basin time-fractions: hotspot1=%.4f (target %.4f) hotspot2=%.4f (target %.4f)\n",
                frac1, mass1, frac2, mass2);
    std::printf("[time] closed loop: %.4f ms average GPU SMC-step kernel per control step over %d steps\n",
                loop_gpu_ms / sc.steps, sc.steps);

    // ======================= ERGODICITY GATE ==================================
    // (i) DECREASE: the metric must fall by at least a documented factor from
    //     early in the run to the end. (ii) WINDOWED MONOTONICITY: split the
    //     run into fixed windows, average epsilon per window, and require
    //     that the window-mean sequence is not allowed to trend UP by more
    //     than a small documented slack — transient single-window upticks are
    //     tolerated (SMC's bang-bang law is not a smooth descent), a sustained
    //     climb is not.
    bool ergodicity_pass;
    {
        const int n_windows = 6;
        const int wlen = sc.steps / n_windows;
        std::vector<double> wmean(static_cast<size_t>(n_windows), 0.0);
        for (int w = 0; w < n_windows; ++w) {
            double sum = 0.0;
            for (int i = w * wlen; i < (w + 1) * wlen; ++i) sum += eps_series[static_cast<size_t>(i)];
            wmean[static_cast<size_t>(w)] = sum / wlen;
        }
        const double decrease_factor = wmean.front() / std::max(wmean.back(), 1e-300);
        int worst_upticks = 0;
        double worst_uptick_ratio = 0.0;
        for (int w = 1; w < n_windows; ++w) {
            if (wmean[static_cast<size_t>(w)] > wmean[static_cast<size_t>(w - 1)]) {
                ++worst_upticks;
                worst_uptick_ratio = std::max(worst_uptick_ratio,
                                              wmean[static_cast<size_t>(w)] / wmean[static_cast<size_t>(w - 1)] - 1.0);
            }
        }
        std::printf("[info] ergodicity: window-mean epsilon decreased by %.1fx (window1=%.4e -> window%d=%.4e); "
                    "%d/%d window-to-window upticks, worst +%.1f%%\n",
                    decrease_factor, wmean.front(), n_windows, wmean.back(),
                    worst_upticks, n_windows - 1, worst_uptick_ratio * 100.0);
        // Thresholds carry wide, MEASURED margin (see README §Expected output
        // for the actual numbers observed on the reference machine).
        ergodicity_pass = (decrease_factor >= 5.0) && (worst_upticks <= 1) && (worst_uptick_ratio <= 0.25);
        all_pass = all_pass && ergodicity_pass;
        std::printf("ERGODICITY: %s (metric decreased >= 5x window-to-window, at most one windowed uptick <= 25%%)\n",
                    ergodicity_pass ? "PASS" : "FAIL");
    }

    // ======================= COVERAGE GATE =====================================
    // Tolerance chosen with real, measured margin (README §Expected output
    // quotes the actual numbers): on the reference machine the observed
    // deviation is ~0.003-0.007, roughly 7-15x inside this bound.
    const double tol_basin = 0.05;
    const bool cov1 = std::fabs(frac1 - mass1) <= tol_basin;
    const bool cov2 = std::fabs(frac2 - mass2) <= tol_basin;
    const bool coverage_pass = cov1 && cov2;
    all_pass = all_pass && coverage_pass;
    std::printf("COVERAGE: %s (time-fraction in each hotspot basin within %.2f absolute of its numerically-integrated target mass)\n",
                coverage_pass ? "PASS" : "FAIL", tol_basin);

    // ======================= NEGATIVE CONTROL ==================================
    // Same length, same phi_k, a lawnmower sweep instead of SMC control —
    // computed entirely on the CPU twin (a comparison baseline, not the
    // taught GPU pattern; no new kernel needed — THEORY.md explains).
    double eps_lawn_final;
    {
        std::vector<double> S_lawn(static_cast<size_t>(kNumModes), 0.0);
        std::vector<double> c_lawn(static_cast<size_t>(kNumModes)), Bx_lawn(static_cast<size_t>(kNumModes)), By_lawn(static_cast<size_t>(kNumModes));
        double xl[2];
        for (int step = 1; step <= sc.steps; ++step) {
            lawnmower_position(step, sc.steps, xl);
            smc_step_cpu(xl[0], xl[1], phi_k_cpu.data(), S_lawn.data(), step, c_lawn.data(), Bx_lawn.data(), By_lawn.data());
        }
        eps_lawn_final = ergodic_metric(c_lawn, phi_k_cpu);
    }
    const double neg_factor = eps_lawn_final / std::max(eps_final, 1e-300);
    const bool negative_control_pass = neg_factor >= 3.0;
    all_pass = all_pass && negative_control_pass;
    std::printf("[info] negative control: lawnmower final epsilon = %.6e (%.1fx the SMC controller's %.6e)\n",
                eps_lawn_final, neg_factor, eps_final);
    std::printf("NEGATIVE-CONTROL: %s (lawnmower sweep's final ergodic metric >= 3x worse than SMC's)\n",
                negative_control_pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_phi_grid));
    CUDA_CHECK(cudaFree(d_phi_k));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_Bx));
    CUDA_CHECK(cudaFree(d_By));

    // ---- artifacts ------------------------------------------------------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);

    if (artifact_ok) {
        std::ofstream f(out_dir + "/trajectory.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "t_s,x1,x2,u1,u2\n";   // domain-normalized [0,1]^2 coordinates; units (see data/README.md)
            for (int s = 0; s < sc.steps; ++s) {
                const double* r = &traj[static_cast<size_t>(s) * 5];
                f << r[0] << ',' << r[1] << ',' << r[2] << ',' << r[3] << ',' << r[4] << '\n';
            }
        }
    }
    if (artifact_ok) {
        std::ofstream f(out_dir + "/ergodic_metric.csv");
        if (f.is_open()) {
            f << "t_s,epsilon\n";
            for (int s = 0; s < sc.steps; ++s)
                f << (static_cast<double>(s) * kDt) << ',' << eps_series[static_cast<size_t>(s)] << '\n';
        } else artifact_ok = false;
    }

    // target_phi.pgm + empirical_coverage.pgm: side-by-side comparison
    // payload. Each image is scaled to ITS OWN max (independent
    // normalization), not a shared absolute scale: the smooth continuous
    // target and the sparse ~10-samples/cell empirical histogram have very
    // different peak-to-mean ratios (a raw visit count is a spikier
    // estimator than a smooth density), so a shared scale would either
    // wash out the target or saturate the histogram to a wall of white.
    // Independent normalization sacrifices absolute-brightness
    // comparability to preserve what the artifact is actually FOR — seeing
    // whether the visited SHAPE (two blobs at the right places, in the
    // right relative proportion) matches the target's shape. The precise,
    // shared-scale, absolute-mass comparison is the numerical COVERAGE gate
    // above, not this picture (documented in demo/README.md).
    if (artifact_ok) {
        std::vector<double> target_vals(static_cast<size_t>(kVisGrid) * kVisGrid);
        double target_max = 0.0;
        for (int r = 0; r < kVisGrid; ++r) {
            const double x2 = (static_cast<double>(r) + 0.5) / kVisGrid;
            for (int c = 0; c < kVisGrid; ++c) {
                const double x1 = (static_cast<double>(c) + 0.5) / kVisGrid;
                const double v = target_phi_shape(x1, x2) / phi_norm_z;   // the TRUE normalized density (matches phi_grid's scale)
                target_vals[static_cast<size_t>(r) * kVisGrid + c] = v;
                target_max = std::max(target_max, v);
            }
        }
        std::vector<unsigned char> target_gray(target_vals.size());
        for (size_t i = 0; i < target_vals.size(); ++i)
            target_gray[i] = static_cast<unsigned char>(std::min(255.0, std::round(255.0 * target_vals[i] / target_max)));
        artifact_ok = write_pgm(out_dir + "/target_phi.pgm", kVisGrid, kVisGrid, target_gray);

        if (artifact_ok) {
            unsigned int emp_max = 1;   // avoid /0 if the trajectory somehow never visited a cell twice
            for (unsigned int v : vis_hist) emp_max = std::max(emp_max, v);
            std::vector<unsigned char> emp_gray(vis_hist.size());
            for (size_t i = 0; i < vis_hist.size(); ++i)
                emp_gray[i] = static_cast<unsigned char>(
                    std::min(255.0, std::round(255.0 * static_cast<double>(vis_hist[i]) / static_cast<double>(emp_max))));
            artifact_ok = write_pgm(out_dir + "/empirical_coverage.pgm", kVisGrid, kVisGrid, emp_gray);
        }
    }

    if (artifact_ok) {
        std::printf("ARTIFACT: wrote demo/out/trajectory.csv, demo/out/ergodic_metric.csv, "
                    "demo/out/target_phi.pgm, demo/out/empirical_coverage.pgm (%d steps)\n", sc.steps);
    } else {
        std::printf("ARTIFACT: FAILED to write one or more demo/out/ files\n");
    }

    const bool success = all_pass && artifact_ok;
    if (success)
        std::printf("RESULT: PASS (transform + verify + ergodicity + coverage + negative-control all hold)\n");
    else
        std::printf("RESULT: FAIL (one or more gates failed — see the gate lines above)\n");
    return success ? 0 : 1;
}
