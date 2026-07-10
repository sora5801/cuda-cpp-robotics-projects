// ===========================================================================
// main.cu — entry point for project 31.01
//           Hamilton-Jacobi reachability: level-set grid solvers
//           (teaching core: double-integrator backward reachable tube)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (grid,
//      domain, |u| bound, target level, horizon) from data/sample/.
//   2. Build the INITIAL LEVEL FUNCTION on the host: l(x,v) = T*(x,v) - t0,
//      where T* is the closed-form minimum time to the origin. Its zero
//      sublevel set {T* <= t0} is the target set — chosen as a min-time
//      sublevel set (not a box) precisely so the answer at EVERY horizon
//      is known exactly: the tube at horizon T is {T* <= t0 + T} (§verify
//      in THEORY.md derives this one-liner).
//   3. Solve the HJ tube PDE on the GPU (n CFL-limited sweeps) AND on the
//      CPU twin, from the same initial field.
//   4. VERIFY STAGE (the §5 GPU-vs-CPU gate): max |V_gpu - V_cpu| over all
//      cells must be within kTwinTol.
//   5. ANALYTIC STAGE (the reason this project is special): classify every
//      cell with the closed-form solution and require the level-set
//      classification V_gpu <= 0 to agree EXACTLY everywhere outside a
//      kBandCells-wide band around the analytic boundary (a first-order
//      scheme cannot localize a front more finely than a couple of cells —
//      the band is documented honesty, not a fudge).
//   6. ARTIFACTS: demo/out/value_function.pgm (the value field as an
//      image — the reachable set is the dark region) and
//      demo/out/brs_boundary.csv (the numeric front, plottable).
//   7. RESULT: PASS only if verify + analytic + artifacts all hold.
//
// Determinism: there is NO RNG anywhere in this project — the scenario is
// constants and the PDE is deterministic. FP32 results are bit-identical
// run-to-run on one machine; across platforms, compiler FMA-contraction
// choices may differ in last ulps, so no stable output line carries a
// floating-point field value — only PASS/FAIL against wide-margin
// tolerances (THEORY.md §numerics).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "ANALYTIC:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" lines
// are unchecked. Change a stable line => update demo/expected_output.txt
// in the same commit.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>                 // EEXIST for the mkdir helper
#include <cmath>                  // std::ceil, std::fabs, std::fmax
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — nvcc chokes; see 07.09)
#else
#include <sys/stat.h>             // mkdir (POSIX twin)
#endif

// ---------------------------------------------------------------------------
// Scenario loading — the committed "problem definition": the grid, the
// dynamics bound, the target level, and the horizon. Everything the PDE
// needs, nothing hardcoded; this is the project's data/sample content and
// follows the repo's strict-loader discipline (a corrupt scenario aborts —
// it never quietly solves the wrong problem, which for a SAFETY tool would
// be the worst possible failure mode).
// Rows: "GRID,nx,nv" "XDOM,xmin,xmax" "VDOM,vmin,vmax" "UMAX,u"
//       "TTARGET,t0" "HORIZON,T" — all six required, order free.
// ---------------------------------------------------------------------------
struct Scenario {
    int    nx = 0, nv = 0;              // grid cells in x and v
    double xmin = 0.0, xmax = 0.0;      // position domain (m)
    double vmin = 0.0, vmax = 0.0;      // velocity domain (m/s)
    double umax = 0.0;                  // acceleration bound (m/s^2)
    double t_target = 0.0;              // target level t0 (s): target = {T* <= t0}
    double horizon = 0.0;               // reachability horizon T (s)
    bool   loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have[6] = { false, false, false, false, false, false };   // GRID..HORIZON seen?
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;              // comment/provenance lines
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');

        // Pull up to two numeric cells; each row type checks what it needs.
        double a = 0.0, b = 0.0;
        int got = 0;
        for (int k = 0; k < 2 && std::getline(ss, cell, ','); ++k) {
            (k == 0 ? a : b) = std::strtod(cell.c_str(), nullptr);
            ++got;
        }

        if      (label == "GRID"    && got == 2) { sc.nx = (int)a; sc.nv = (int)b; have[0] = true; }
        else if (label == "XDOM"    && got == 2) { sc.xmin = a; sc.xmax = b;       have[1] = true; }
        else if (label == "VDOM"    && got == 2) { sc.vmin = a; sc.vmax = b;       have[2] = true; }
        else if (label == "UMAX"    && got >= 1) { sc.umax = a;                    have[3] = true; }
        else if (label == "TTARGET" && got >= 1) { sc.t_target = a;                have[4] = true; }
        else if (label == "HORIZON" && got >= 1) { sc.horizon = a;                 have[5] = true; }
        else {
            std::fprintf(stderr, "scenario: bad row '%s'\n", line.c_str());
            return Scenario{};
        }
    }
    for (bool h : have)
        if (!h) { std::fprintf(stderr, "scenario: a required row is missing\n"); return Scenario{}; }

    // Semantic validation: the numbers must describe a solvable problem.
    if (sc.nx < 16 || sc.nv < 16 || sc.xmax <= sc.xmin || sc.vmax <= sc.vmin ||
        sc.umax <= 0.0 || sc.t_target <= 0.0 || sc.horizon <= 0.0) {
        std::fprintf(stderr, "scenario: values out of range\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

// Path helpers (same exe-relative resolution as the sibling flagships: the
// exe sits at build/x64/<Config>/, three levels below the project root).
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/double_integrator_scenario.csv");
    candidates.push_back("data/sample/double_integrator_scenario.csv");
    candidates.push_back("../data/sample/double_integrator_scenario.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// Create one directory level; succeeding OR already-existing both count as
// success (demo/out sits directly under the existing demo/, so one level
// is all we ever need — the 07.09/08.01 pattern).
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
// PGM artifact — the smallest real image format there is (P5: one header
// trio, then raw bytes), viewable anywhere, zero libraries (07.09's choice).
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, int width, int height,
                      const std::vector<uint8_t>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()),
              static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// main — the pipeline described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data double_integrator_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Hamilton-Jacobi reachability: double-integrator backward reachable tube (project 31.01)\n");
    print_device_info();

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/double_integrator_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    const Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    // ---- derived solver parameters (double precision, computed ONCE) ---------
    // Node-centered grid: cell 0 on xmin, cell nx-1 on xmax (kernels.cuh).
    const double dx = (sc.xmax - sc.xmin) / (sc.nx - 1);   // m per cell
    const double dv = (sc.vmax - sc.vmin) / (sc.nv - 1);   // m/s per cell

    // CFL: information speed is |v| in x (bounded by the domain edge) and
    // umax in v. n_sweeps = ceil(T / dt_max) then dt = T/n lands the last
    // sweep EXACTLY on the horizon; the 1e-9 slack keeps last-ulp rounding
    // from ceil-ing an integer ratio up a whole extra sweep.
    const double a_max = std::fmax(std::fabs(sc.vmin), std::fabs(sc.vmax));  // max |v| (m/s)
    const double rate  = a_max / dx + sc.umax / dv;                          // 1/s per unit CFL
    const int n_sweeps = (int)std::ceil(sc.horizon * rate / (double)kCfl - 1e-9);
    const double dt    = sc.horizon / n_sweeps;                              // s per sweep

    std::printf("PROBLEM: HJ reachability tube, %dx%d grid over x [%g,%g] m x v [%g,%g] m/s, |u| <= %g m/s^2, horizon T=%g s, FP32\n",
                sc.nx, sc.nv, sc.xmin, sc.xmax, sc.vmin, sc.vmax, sc.umax, sc.horizon);
    std::printf("SCENARIO: target = {min-time-to-origin <= %g s}; %d sweeps @ dt=%.3f ms (local Lax-Friedrichs + freezing, CFL %.1f) [synthetic]\n",
                sc.t_target, n_sweeps, dt * 1e3, (double)kCfl);

    const HjGrid g = { sc.nx, sc.nv,
                       (float)sc.xmin, (float)dx,
                       (float)sc.vmin, (float)dv,
                       (float)sc.umax, (float)dt };
    const size_t total = (size_t)sc.nx * sc.nv;

    // ---- initial condition + analytic cache ----------------------------------
    // l(x,v) = T*(x,v) - t0: zero exactly on the target boundary, negative
    // inside. Computed ONCE on the host in double (the oracle's precision),
    // narrowed to FP32, and fed IDENTICALLY to both solver paths — so any
    // GPU/CPU disagreement later is the solvers', never the setup's.
    // The same T* values also serve the analytic stage below, so they are
    // kept (in double) for the whole run: 256x256 doubles = 512 KiB.
    std::vector<double> tstar(total);            // T*(x_i, v_j), seconds
    std::vector<float>  l0(total);               // the initial level function
    for (int j = 0; j < sc.nv; ++j) {
        const double vj = sc.vmin + j * dv;
        for (int i = 0; i < sc.nx; ++i) {
            const double xi = sc.xmin + i * dx;
            const double t = min_time_to_origin(xi, vj, sc.umax);
            tstar[(size_t)j * sc.nx + i] = t;
            l0[(size_t)j * sc.nx + i] = (float)(t - sc.t_target);
        }
    }

    // ---- GPU solve ------------------------------------------------------------
    std::vector<float> v_gpu(total);
    float* d_V = nullptr;
    CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_V, l0.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer gt;                                 // events fence the whole sweep loop
    gt.begin();
    launch_hj_solve(g, n_sweeps, d_V);
    const float gpu_ms = gt.end_ms();            // includes the launcher's internal alloc/copies
    CUDA_CHECK(cudaMemcpy(v_gpu.data(), d_V, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_V));

    // ---- CPU twin solve --------------------------------------------------------
    std::vector<float> v_cpu(l0);                // same start; solved in place
    CpuTimer ct;
    ct.begin();
    hj_solve_cpu(g, n_sweeps, v_cpu.data());
    const double cpu_ms = ct.end_ms();

    std::printf("[time] %d sweeps on %dx%d: CPU %.1f ms (%.3f ms/sweep) | GPU %.2f ms (%.4f ms/sweep, incl. internal alloc) | speed-up %.0fx (teaching artifact)\n",
                n_sweeps, sc.nx, sc.nv, cpu_ms, cpu_ms / n_sweeps,
                (double)gpu_ms, (double)gpu_ms / n_sweeps,
                cpu_ms / ((double)gpu_ms > 0.0 ? (double)gpu_ms : 1.0));

    // ======================= VERIFY STAGE (twin) ================================
    // Same FP32 arithmetic, same order, both paths — only compiler FMA
    // contraction differs, so the fields must agree almost bitwise. The
    // tolerance story lives with kTwinTol in kernels.cuh.
    float worst = 0.0f;
    for (size_t k = 0; k < total; ++k) {
        const float d = std::fabs(v_gpu[k] - v_cpu[k]);
        if (d > worst) worst = d;
    }
    const bool verify_pass = (worst <= kTwinTol);
    std::printf("[info] verify: worst |V_gpu - V_cpu| = %.3e over %zu cells\n", (double)worst, total);
    std::printf("VERIFY: %s (GPU value field matches CPU twin within max |dV| tol 1e-3)\n",
                verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU twin disagreement — fix before trusting anything else)\n");
        return 1;
    }

    // ======================= ANALYTIC STAGE (mathematics) =======================
    // The exact answer at horizon T: the tube is {T* <= t0 + T}, because
    // "can reach {T* <= t0} within T" is literally "T* <= t0 + T" (minimum-
    // time dynamic programming; THEORY.md §verify). Even the full exact
    // VALUE function is known: V_exact = max(T* - t0 - T, -t0) — reported
    // on an [info] line so the scheme's dissipation error is visible, not
    // just its classification.
    const double tau_f = sc.t_target + sc.horizon;   // analytic boundary level (s)

    std::vector<uint8_t> cls_exact(total);           // 1 = inside the true tube
    double sup_err = 0.0;                            // sup |V_num - V_exact|
    size_t n_inside_num = 0;                         // numeric tube size (cells)
    for (size_t k = 0; k < total; ++k) {
        cls_exact[k] = (tstar[k] <= tau_f) ? 1u : 0u;
        const double v_exact = std::fmax(tstar[k] - tau_f, -sc.t_target);
        const double e = std::fabs((double)v_gpu[k] - v_exact);
        if (e > sup_err) sup_err = e;
        if (v_gpu[k] <= 0.0f) ++n_inside_num;
    }

    // The excused band: cells whose ANALYTIC class changes somewhere within
    // Chebyshev radius kBandCells. Defined from the exact solution only —
    // the numeric answer cannot vote on where its own errors are excused.
    size_t n_band = 0, n_wrong_out = 0, n_wrong_in = 0;
    for (int j = 0; j < sc.nv; ++j) {
        for (int i = 0; i < sc.nx; ++i) {
            const size_t k = (size_t)j * sc.nx + i;
            bool band = false;
            for (int dj = -kBandCells; dj <= kBandCells && !band; ++dj) {
                const int jj = j + dj;
                if (jj < 0 || jj >= sc.nv) continue;
                for (int di = -kBandCells; di <= kBandCells; ++di) {
                    const int ii = i + di;
                    if (ii < 0 || ii >= sc.nx) continue;
                    if (cls_exact[(size_t)jj * sc.nx + ii] != cls_exact[k]) { band = true; break; }
                }
            }
            const bool agree = ((v_gpu[k] <= 0.0f) ? 1u : 0u) == cls_exact[k];
            if (band) { ++n_band; if (!agree) ++n_wrong_in; }
            else if (!agree) ++n_wrong_out;   // outside the band: must be zero
        }
    }
    const bool analytic_pass = (n_wrong_out == 0);
    std::printf("[info] analytic: numeric tube %zu cells (%.1f%% of grid); boundary band (%d cells wide) holds %zu cells; "
                "disagreements outside band %zu, inside band %zu\n",
                n_inside_num, 100.0 * (double)n_inside_num / (double)total,
                kBandCells, n_band, n_wrong_out, n_wrong_in);
    std::printf("[info] analytic: sup |V_num - V_exact| = %.4f s (first-order LF dissipation — THEORY.md quantifies it)\n",
                sup_err);
    std::printf("ANALYTIC: %s (grid classification matches the closed-form bang-bang solution everywhere outside a %d-cell boundary band)\n",
                analytic_pass ? "PASS" : "FAIL", kBandCells);

    // ======================= ARTIFACTS ==========================================
    // The result is inherently visual — ship it (CLAUDE.md §6.3). demo/out/
    // is git-ignored run-time scratch, regenerated every run. Paths print
    // relative on the stable line (machine-neutral), absolute on [info].
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);

    if (artifact_ok) {
        // (a) The value field as an 8-bit image. Linear map of V onto gray:
        // the reachable tube is the DARK region, the target's deep interior
        // (V = -t0) is black, far unreachable states are white. Row j=0
        // (v = vmin) is the TOP image row — flip vertically when plotting
        // if you want v to point up.
        float lo = v_gpu[0], hi = v_gpu[0];
        for (size_t k = 1; k < total; ++k) {
            if (v_gpu[k] < lo) lo = v_gpu[k];
            if (v_gpu[k] > hi) hi = v_gpu[k];
        }
        const float span = (hi > lo) ? (hi - lo) : 1.0f;
        std::vector<uint8_t> gray(total);
        for (size_t k = 0; k < total; ++k)
            gray[k] = (uint8_t)(255.0f * (v_gpu[k] - lo) / span + 0.5f);
        artifact_ok = write_pgm(out_dir + "/value_function.pgm", sc.nx, sc.nv, gray);
    }
    if (artifact_ok) {
        // (b) The numeric front as CSV: every inside cell with an outside
        // face neighbor, with the analytic minimum time alongside — plot
        // t_min_s along the boundary and watch it hug t0 + T (that near-
        // constant column IS the verification, visible to the eye).
        std::ofstream f(out_dir + "/brs_boundary.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "x_m,v_mps,t_min_s\n";           // units in the header row (§12)
            for (int j = 0; j < sc.nv; ++j) {
                for (int i = 0; i < sc.nx; ++i) {
                    const size_t k = (size_t)j * sc.nx + i;
                    if (v_gpu[k] > 0.0f) continue;           // outside — not front
                    const bool edge =                        // any face neighbor outside?
                        (i > 0         && v_gpu[k - 1]           > 0.0f) ||
                        (i < sc.nx - 1 && v_gpu[k + 1]           > 0.0f) ||
                        (j > 0         && v_gpu[k - (size_t)sc.nx] > 0.0f) ||
                        (j < sc.nv - 1 && v_gpu[k + (size_t)sc.nx] > 0.0f);
                    if (!edge) continue;
                    f << (sc.xmin + i * dx) << ',' << (sc.vmin + j * dv)
                      << ',' << tstar[k] << '\n';
                }
            }
            artifact_ok = static_cast<bool>(f);
        }
    }
    if (artifact_ok) {
        std::printf("[info] artifact dir: %s\n", out_dir.c_str());
        std::printf("ARTIFACT: wrote demo/out/value_function.pgm and demo/out/brs_boundary.csv (%dx%d grid)\n",
                    sc.nx, sc.nv);
    } else {
        std::printf("ARTIFACT: FAILED to write demo/out files\n");
    }

    // ---- verdict ---------------------------------------------------------------
    const bool all_pass = verify_pass && analytic_pass && artifact_ok;
    if (all_pass)
        std::printf("RESULT: PASS (backward reachable tube verified against the CPU twin and the analytic minimum-time solution)\n");
    else
        std::printf("RESULT: FAIL (a verification stage failed — see the lines above)\n");
    return all_pass ? 0 : 1;
}
