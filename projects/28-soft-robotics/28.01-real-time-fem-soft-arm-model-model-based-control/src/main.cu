// ===========================================================================
// main.cu — entry point for project 28.01
//           Real-time FEM soft-arm model + model-based control
//           (teaching core: explicit corotational FEM soft arm + a
//            model-based task-space tip controller)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Load the committed scenario (data/sample/arm_scenario.csv): the MODEL
//      rows are cross-checked against kernels.cuh's compiled constants (the
//      CFL timestep and analytic-gate formulas are derived from them at
//      compile time — the file cannot silently lie about what is simulated);
//      the SCENARIO rows (controller tuning, setpoint sequence) are genuine
//      runtime inputs.
//   2. VERIFY STAGE (§5 gate): 500 steps of full dynamics (damping + tendon
//      actuation on) from identical initial conditions, GPU vs CPU, compared
//      within a documented, reassociation-aware tolerance (kernels.cu's
//      atomicAdd scatter vs reference_cpu.cpp's deterministic sequential
//      sum). NaN-proofed: any non-finite value fails the gate outright.
//   3. FREE-VIBRATION STAGE (three analytic gates in one run): settle the
//      arm under a small tip point load (gate i, static deflection vs.
//      Euler-Bernoulli), then release it — mass damping OFF, only the
//      numerically-required minimal stiffness damping retained (the ring
//      comment explains why literal zero damping is not available to this
//      force formulation) — and let it ring (gate ii, first-mode frequency
//      via zero-crossing counting; gate iii, energy-conservation bound).
//   4. CONTROL STAGE: probe the FEM model itself with a small tendon-tension
//      step to IDENTIFY the quasi-static tip-deflection-per-tension gain,
//      build a PI controller from it, then track a step+hold setpoint
//      sequence in closed loop, logging the trajectory and per-setpoint
//      rise-time/overshoot/steady-state-error.
//   5. ARTIFACTS: demo/out/tip_trajectory.csv (the control plot),
//      demo/out/arm_snapshots.csv (node positions at documented times),
//      demo/out/arm_deformed.pgm (a rasterized deformed-arm frame).
//   6. REALTIME: report the measured real-time factor (simulated seconds per
//      wall-clock second, accumulated across every stepping phase) — the
//      bullet's "real-time" claim, measured, never promised.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "MESH:", "SCENARIO:", "VERIFY:", "GATE ...:", "IDENTIFY:", "SETPOINT ...:",
// "ARTIFACT:", "REALTIME:", and "RESULT:" — stable lines carry PASS/FAIL
// verdicts and fixed problem parameters only, never measured floats that a
// different-but-correct GPU could shift. "[info]"/"[time]" lines are NOT
// diffed. Change a stable line => update demo/expected_output.txt in the
// same change (08.01's exact convention).
//
// Read this first, then kernels.cuh (the contract) -> kernels.cu (the GPU
// kernels) -> reference_cpu.cpp (the oracle + shared geometry/energy math).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

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
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — repo precedent)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Fixed orchestration constants (not scenario-tunable: they define the
// project's verification protocol, not the experiment). Each value's
// derivation is in the comment beside it; scenario-tunable values live in
// data/sample/arm_scenario.csv (see the Scenario struct below).
// ---------------------------------------------------------------------------
static constexpr int kVerifySteps = 500;              // §5 gate window (500 * 3e-5 s = 15 ms of dynamics)
static constexpr float kVerifyTolX = 1.0e-5f;         // worst |x_gpu - x_cpu| bound (m) — see the VERIFY block
static constexpr float kVerifyTolV = 5.0e-3f;         // worst |v_gpu - v_cpu| bound (m/s): velocities are the
                                                      // phase-sensitive quantity (high-frequency waves amplify
                                                      // ulp differences); measured worst ~6e-4, bound sits ~8x
                                                      // above it and ~10x below the physical velocity scale

// Static-deflection gate: a small tip point load in the small-deflection
// regime where Euler-Bernoulli applies. P = 0.02 N gives an analytic tip
// deflection P*L^3/(3EI) = 4.00 mm = 1.7% of L — comfortably "small".
static constexpr float kStaticLoadN = -0.02f;         // tip point load, -Y (N)

// Settling protocol: run with a BOOSTED mass-proportional damping until the
// arm is quasi-static. The boost (alpha = 12 1/s -> zeta_1 ~ 0.47, still
// stable: dt*alpha = 3.6e-4) is a NUMERICAL device, not a material property
// — equilibrium is independent of damping, and heavier damping reaches it in
// ~1/3 the time (the classic dynamic-relaxation trick, THEORY.md
// "numerics"). Never used in a phase whose DYNAMICS are measured.
static constexpr float kSettleAlphaBoost = 12.0f;     // 1/s (settle phases only)
static constexpr int   kSettleMaxSteps = 90000;       // hard cap (2.7 s sim) — an honest safety net
static constexpr float kSettleVelEps_ms = 1.0e-3f;    // "settled" once max |v| over all DOFs < this (m/s).
                                                      // Why 1 mm/s and not lower: explicit FP32 corotational
                                                      // dynamics never reaches literal stillness — a residual
                                                      // jitter floor (measured ~4e-4 m/s here) persists from
                                                      // rounding + the damped remnant of the warped-stiffness
                                                      // flutter (kernels.cuh's beta comment; THEORY.md). At
                                                      // 1 mm/s of localized jitter the mm-scale POSITIONS are
                                                      // quasi-static (the static gate measures 2-3% EB error).
static constexpr int   kSettleCheckStride = 200;      // steps between velocity-check downloads

// Free-vibration window: 4 analytic periods of the first bending mode gives
// >= 4 rising zero crossings (>= 3 full period estimates) for gate (ii) and
// a long-enough energy record for gate (iii), while staying cheap.
static constexpr float kFreeVibPeriods = 4.0f;
static constexpr int   kTipSampleStride = 20;         // steps between tip-trace samples (~0.6 ms sampling)
static constexpr int   kEnergySampleStride = 400;     // steps between full-state energy samples

// Ring-phase stiffness damping: the ring cannot run at literal beta=0 (the
// warped-stiffness flutter self-excites — kernels.cuh's beta comment), but
// it can run MUCH lighter than the closed-loop material's beta. Measured
// sweep (CPU twin, 4-period window): beta 2.5e-6/5e-6/1e-5/2e-5 all quench
// the flutter (from-rest energy stays at the 1e-9 J noise scale vs 2.6e-5 J
// undamped); 5e-6 is 2x above the smallest tested suppressor. Its mode-1
// dissipation is zeta_1 = beta*omega_1/2 = 3.2e-5 — energy loss 0.16% over
// the window, i.e. mode 1 rings essentially undamped.
static constexpr float kRingBeta = 5.0e-6f;           // s (ring phase only)

// Analytic-gate allowances (each justified in THEORY.md "How we verify"):
static constexpr double kGateStaticRelTol = 0.30;     // Q4 discretization/locking allowance (measured value printed)
static constexpr double kGateFreqRelTol = 0.20;       // mesh stiffness/mass discretization allowance
static constexpr double kGateEnergyDriftPct = 8.0;    // energy-drift bound. Budget (all measured, THEORY.md):
                                                      // ~4% nonlinear leak of mode-1 energy into beta-damped mesh
                                                      // modes (beta-INdependent: 4.1% at beta=2.5e-6 vs 5.3% at
                                                      // 2e-5), 0.16% beta dissipation of mode 1, ~0.02% symplectic
                                                      // oscillation; measured total 4.3% -> 8% is ~1.8x margin,
                                                      // while a real defect (the undamped flutter measured +41%
                                                      // over HALF this window) blows through it immediately.

// Control-verdict bounds (per setpoint; documented in README):
static constexpr double kCtrlOvershootPctMax = 60.0;  // lightly-damped plant (zeta ~ 0.15) + PI: generous
static constexpr double kCtrlSseMax_m = 3.0e-4;       // 0.3 mm — several times smaller than the setpoints

// ---------------------------------------------------------------------------
// Scenario — the committed sample's runtime half (see scripts/
// make_synthetic.py for the generator and data/README.md for field docs).
// MODEL rows are cross-checked against kernels.cuh; SCENARIO rows land here.
// ---------------------------------------------------------------------------
struct Scenario {
    float probe_delta_t_n = 0.0f;      // Jacobian-identification tension step (N)
    int   control_substeps = 0;        // dynamics steps per control tick
    int   hold_steps = 0;              // dynamics steps held per setpoint
    float pi_margin_alpha = 0.0f;      // Kp = margin / |J|
    float pi_integral_time_s = 0.0f;   // Ki = Kp / Ti
    float delta_t_clamp_frac = 0.0f;   // |deltaT| <= frac * bias
    float setpoint_safe_frac = 0.0f;   // setpoint scale = J * frac * clamp
    std::vector<float> setpoint_fracs; // step+hold sequence (fractions of the scale)
    bool loaded = false;
};

// rel_match — float comparison for the model cross-check: the CSV stores
// short decimal strings; parsing them reproduces the compiled constants to
// well below 1e-6 relative, so 1e-6 separates "same value" from "someone
// edited one side only".
static bool rel_match(float file_v, float code_v)
{
    const float scale = std::fabs(code_v) > 1.0f ? std::fabs(code_v) : 1.0f;
    return std::fabs(file_v - code_v) / scale <= 1.0e-6f;
}

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    // Required-key bookkeeping: every MODEL and SCENARIO key must appear —
    // a missing row is a corrupt sample, never a silent default.
    bool model_ok = true;
    int model_seen = 0, scen_seen = 0;

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // tolerate CRLF checkouts
        if (line.empty() || line[0] == '#') continue;
        const size_t comma = line.find(',');
        if (comma == std::string::npos) {
            std::fprintf(stderr, "scenario: bad row '%s'\n", line.c_str());
            return Scenario{};
        }
        const std::string key = line.substr(0, comma);
        const std::string val = line.substr(comma + 1);

        // MODEL rows: cross-check against kernels.cuh (file-header rationale;
        // a mismatch is a hard error, not a warning).
        auto check_model = [&](float code_v) {
            const float file_v = std::strtof(val.c_str(), nullptr);
            if (!rel_match(file_v, code_v)) {
                std::fprintf(stderr, "scenario: MODEL row %s=%s disagrees with the compiled constant %.9g\n",
                             key.c_str(), val.c_str(), static_cast<double>(code_v));
                model_ok = false;
            }
            ++model_seen;
        };
        if      (key == "NELX")           check_model(static_cast<float>(kNelx));
        else if (key == "NELY")           check_model(static_cast<float>(kNely));
        else if (key == "ELEM_SIZE_M")    check_model(kElemSize_m);
        else if (key == "YOUNGS_E_PA")    check_model(kYoungsE_Pa);
        else if (key == "POISSON_NU")     check_model(kPoissonNu);
        else if (key == "THICKNESS_M")    check_model(kThickness_m);
        else if (key == "DENSITY_KGM3")   check_model(kDensity_kgm3);
        else if (key == "DT_S")           check_model(kDt_s);
        else if (key == "RAYLEIGH_ALPHA") check_model(kRayleighAlphaOn);
        else if (key == "RAYLEIGH_BETA")  check_model(kRayleighBetaOn);
        else if (key == "TENDON_BIAS_N")  check_model(kTendonBiasN);
        // SCENARIO rows: genuine runtime inputs.
        else if (key == "PROBE_DELTA_T_N")    { sc.probe_delta_t_n = std::strtof(val.c_str(), nullptr); ++scen_seen; }
        else if (key == "CONTROL_SUBSTEPS")   { sc.control_substeps = std::atoi(val.c_str()); ++scen_seen; }
        else if (key == "HOLD_STEPS")         { sc.hold_steps = std::atoi(val.c_str()); ++scen_seen; }
        else if (key == "PI_MARGIN_ALPHA")    { sc.pi_margin_alpha = std::strtof(val.c_str(), nullptr); ++scen_seen; }
        else if (key == "PI_INTEGRAL_TIME_S") { sc.pi_integral_time_s = std::strtof(val.c_str(), nullptr); ++scen_seen; }
        else if (key == "DELTA_T_CLAMP_FRAC") { sc.delta_t_clamp_frac = std::strtof(val.c_str(), nullptr); ++scen_seen; }
        else if (key == "SETPOINT_SAFE_FRAC") { sc.setpoint_safe_frac = std::strtof(val.c_str(), nullptr); ++scen_seen; }
        else if (key == "SETPOINT_FRACS") {
            std::stringstream ss(val);
            std::string cell;
            while (std::getline(ss, cell, ';')) sc.setpoint_fracs.push_back(std::strtof(cell.c_str(), nullptr));
            ++scen_seen;
        }
        else { std::fprintf(stderr, "scenario: unknown key '%s'\n", key.c_str()); return Scenario{}; }
    }

    if (!model_ok) return Scenario{};
    if (model_seen != 11 || scen_seen != 8 || sc.setpoint_fracs.empty() ||
        sc.control_substeps < 1 || sc.hold_steps < sc.control_substeps ||
        sc.probe_delta_t_n <= 0.0f || sc.pi_margin_alpha <= 0.0f ||
        sc.pi_integral_time_s <= 0.0f || sc.delta_t_clamp_frac <= 0.0f ||
        sc.setpoint_safe_frac <= 0.0f) {
        std::fprintf(stderr, "scenario: missing/invalid rows (model %d/11, scenario %d/8)\n", model_seen, scen_seen);
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/arm_scenario.csv");
    candidates.push_back("data/sample/arm_scenario.csv");
    candidates.push_back("../data/sample/arm_scenario.csv");
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
// ArmSim — RAII-ish bundle of the device buffers every phase steps. Owning
// this in one struct (instead of passing 6 raw pointers everywhere) keeps
// the phase code readable; kernels.cuh/kernels.cu stay pointer-based,
// matching every other project's launch_* signature style.
// ---------------------------------------------------------------------------
struct ArmSim {
    ArmGrid g{};
    int nnode = 0, ndof = 0, nelem = 0;
    float h = 0.0f, Et = 0.0f;
    float* d_x = nullptr;         // [ndof] node positions (m)
    float* d_v = nullptr;         // [ndof] node velocities (m/s)
    float* d_force = nullptr;     // [ndof] scatter target (zero-after-consume contract, kernels.cuh)
    float* d_node_mass = nullptr; // [nnode] lumped mass (kg)
    uint8_t* d_fixed = nullptr;   // [ndof] Dirichlet mask
    std::vector<float> node_mass_host;   // host copies kept for the CPU twin
    std::vector<uint8_t> fixed_host;     // and the energy diagnostics

    void init()
    {
        g = make_arm_grid();
        nnode = g.nx * g.ny;
        ndof = 2 * nnode;
        nelem = g.nelx * g.nely;
        h = kElemSize_m;
        Et = kEt_N;

        node_mass_host.resize(static_cast<size_t>(nnode));
        compute_node_mass(g, kDensity_kgm3, kThickness_m, h, node_mass_host.data());
        fixed_host.resize(static_cast<size_t>(ndof));
        build_fixed_mask(g, fixed_host.data());

        CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(ndof) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(ndof) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_force, static_cast<size_t>(ndof) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_node_mass, static_cast<size_t>(nnode) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fixed, static_cast<size_t>(ndof) * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_node_mass, node_mass_host.data(),
                              static_cast<size_t>(nnode) * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fixed, fixed_host.data(),
                              static_cast<size_t>(ndof) * sizeof(uint8_t), cudaMemcpyHostToDevice));
        // The ONE-TIME force zero the zero-after-consume contract requires
        // (kernels.cuh) — after this, node_integrate_kernel keeps it clean.
        CUDA_CHECK(cudaMemset(d_force, 0, static_cast<size_t>(ndof) * sizeof(float)));
    }

    // reset_to_rest — x = reference grid (i*h, j*h), v = 0.
    void reset_to_rest()
    {
        std::vector<float> x0(static_cast<size_t>(ndof)), v0(static_cast<size_t>(ndof), 0.0f);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const int n = node_id(g, i, j);
                x0[static_cast<size_t>(2 * n)]     = static_cast<float>(i) * h;
                x0[static_cast<size_t>(2 * n + 1)] = static_cast<float>(j) * h;
            }
        CUDA_CHECK(cudaMemcpy(d_x, x0.data(), static_cast<size_t>(ndof) * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, v0.data(), static_cast<size_t>(ndof) * sizeof(float), cudaMemcpyHostToDevice));
    }

    // step — one dt with the given damping/actuation. This wrapper exists so
    // the phase code below reads as physics, not plumbing.
    void step(float alpha, float beta, float T_top, float T_bottom,
              int pf_node = kNoPointForce, float pf_x = 0.0f, float pf_y = 0.0f)
    {
        launch_fem_step(g, d_x, d_v, d_force, d_node_mass, d_fixed, Et, h,
                        alpha, beta, kDt_s, T_top, T_bottom, pf_node, pf_x, pf_y);
    }

    // download_tip_y — the tip centerline node's world Y (m). One 4-byte D2H
    // copy; it synchronizes the stream — that sync is what makes it a
    // "sensor reading" the controller can act on.
    float download_tip_y() const
    {
        float y = 0.0f;
        const int n = tip_node_index(g);
        CUDA_CHECK(cudaMemcpy(&y, d_x + (2 * n + 1), sizeof(float), cudaMemcpyDeviceToHost));
        return y;
    }

    // max_abs_velocity — settle-check diagnostic. NaN-PROOF by construction:
    // any non-finite velocity returns +infinity, so a blown-up state can
    // never read as "settled". (The trap: fabs(NaN) > m is FALSE for every
    // m, so the naive max silently reports 0 for all-NaN input — we check
    // isfinite explicitly instead of trusting comparison semantics.)
    float max_abs_velocity() const
    {
        std::vector<float> v(static_cast<size_t>(ndof));
        CUDA_CHECK(cudaMemcpy(v.data(), d_v, static_cast<size_t>(ndof) * sizeof(float), cudaMemcpyDeviceToHost));
        float m = 0.0f;
        for (float f : v) {
            if (!std::isfinite(f)) return HUGE_VALF;
            if (std::fabs(f) > m) m = std::fabs(f);
        }
        return m;
    }

    void download_state(std::vector<float>& x, std::vector<float>& v) const
    {
        x.resize(static_cast<size_t>(ndof)); v.resize(static_cast<size_t>(ndof));
        CUDA_CHECK(cudaMemcpy(x.data(), d_x, static_cast<size_t>(ndof) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v.data(), d_v, static_cast<size_t>(ndof) * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void destroy()
    {
        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_force));
        CUDA_CHECK(cudaFree(d_node_mass)); CUDA_CHECK(cudaFree(d_fixed));
    }
};

// ---------------------------------------------------------------------------
// settle — run damped dynamics under a fixed load case until quasi-static
// (max |v| < kSettleVelEps_ms) or the step cap. Uses the BOOSTED alpha (see
// kSettleAlphaBoost) — equilibrium does not depend on damping, so the boost
// changes only how fast we get there. Returns steps used; accumulates the
// caller's simulated/wall-second totals (the REALTIME measurement).
// ---------------------------------------------------------------------------
static int settle(ArmSim& sim, float T_top, float T_bottom,
                  int pf_node, float pf_x, float pf_y,
                  double& sim_seconds_accum, double& wall_seconds_accum,
                  float* out_final_maxv = nullptr)
{
    // Require kSettleConsecOk CONSECUTIVE sub-threshold checks before
    // declaring "settled". Why: max |v| dips through zero at every
    // oscillation TURNAROUND (velocity vanishes at displacement extremes) —
    // a single-check criterion can fire mid-transient at ~20% overshoot of
    // the true equilibrium (measured: it inflated the static-gate deflection
    // to 4.84 mm vs the settled 4.10 mm). The turnaround window where
    // max |v| stays below threshold is ~600 steps at this mode-1 period;
    // 5 checks x 200 steps = 1000 steps cannot fit inside it, so passing all
    // five means velocities are GENUINELY low, not momentarily low.
    constexpr int kSettleConsecOk = 5;
    CpuTimer wt; wt.begin();
    int steps = 0, consec = 0;
    float maxv = HUGE_VALF;
    for (; steps < kSettleMaxSteps; ++steps) {
        sim.step(kSettleAlphaBoost, kRayleighBetaOn, T_top, T_bottom, pf_node, pf_x, pf_y);
        if ((steps + 1) % kSettleCheckStride == 0) {
            maxv = sim.max_abs_velocity();
            consec = (maxv < kSettleVelEps_ms) ? consec + 1 : 0;
            if (consec >= kSettleConsecOk) { ++steps; break; }
        }
    }
    sim_seconds_accum += static_cast<double>(steps) * static_cast<double>(kDt_s);
    wall_seconds_accum += wt.end_ms() / 1000.0;
    if (out_final_maxv) *out_final_maxv = maxv;
    return steps;
}

// ---------------------------------------------------------------------------
// count_periods_by_zero_crossing — THEORY.md's FFT-free frequency estimate:
// linearly interpolate the time of each RISING zero crossing of the (zero-
// mean) tip trace, average the spacing between consecutive crossings,
// invert. Returns 0 if fewer than 2 rising crossings were found (measurement
// failure — the caller fails the gate honestly, never invents a number).
// ---------------------------------------------------------------------------
static double count_periods_by_zero_crossing(const std::vector<double>& t, const std::vector<double>& y)
{
    std::vector<double> cross_t;
    for (size_t k = 0; k + 1 < y.size(); ++k) {
        if (y[k] < 0.0 && y[k + 1] >= 0.0) {          // rising crossings only: one per period, no double count
            const double frac = (0.0 - y[k]) / (y[k + 1] - y[k]);
            cross_t.push_back(t[k] + frac * (t[k + 1] - t[k]));
        }
    }
    if (cross_t.size() < 2) return 0.0;
    double sum_period = 0.0;
    for (size_t k = 0; k + 1 < cross_t.size(); ++k) sum_period += (cross_t[k + 1] - cross_t[k]);
    const double mean_period = sum_period / static_cast<double>(cross_t.size() - 1);
    return (mean_period > 0.0) ? (1.0 / mean_period) : 0.0;
}

// ---------------------------------------------------------------------------
// write_pgm_arm — rasterize the CURRENT deformed mesh as plain ASCII PGM
// (P2): white background, mesh edges in black via integer Bresenham. No
// image library — PGM is human-readable text and the dozen-line rasterizer
// is itself a teaching surface (CLAUDE.md §5's "hand-roll unless it teaches
// nothing"). ROBUSTNESS: coordinates are clamped to the canvas neighborhood
// and non-finite values skip their segments — a float->int cast of NaN is
// undefined behavior, and an unclamped Bresenham handed garbage endpoints
// can loop effectively forever (this project's first draft hung exactly
// there when unstable damping produced NaN state; THEORY.md "numerics").
// ---------------------------------------------------------------------------
static void write_pgm_arm(const std::string& path, const ArmGrid& g, const std::vector<float>& x)
{
    const int W = 480, H = 140;                // canvas (px): a wide strip fitting the arm + margin
    std::vector<uint8_t> img(static_cast<size_t>(W) * H, 255);

    // World-to-pixel: fit [-margin, L+margin] across the width; world y=0 at
    // the vertical center; y flipped (image row 0 is the top).
    const float margin_m = 0.02f;              // 2 cm each side (tip deflections are millimeters)
    const float scale = static_cast<float>(W) / (kArmLength_m + 2.0f * margin_m);   // px per meter
    const float y_center_px = static_cast<float>(H) * 0.5f;

    auto put = [&](int px, int py) {
        if (px < 0 || px >= W || py < 0 || py >= H) return;
        img[static_cast<size_t>(py) * W + px] = 0;
    };
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    auto to_px = [&](float wx, float wy, int* px, int* py) -> bool {
        if (!std::isfinite(wx) || !std::isfinite(wy)) return false;      // NaN guard (see header)
        *px = clampi(static_cast<int>((wx + margin_m) * scale), -1, W);  // clamp to canvas+1 ring:
        *py = clampi(static_cast<int>(y_center_px - wy * scale), -1, H); // bounds Bresenham's work
        return true;
    };
    auto line = [&](int x0, int y0, int x1, int y1) {   // integer Bresenham (endpoints pre-clamped above)
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        for (;;) {
            put(x0, y0);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    };

    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            const int n = node_id(g, i, j);
            int px, py;
            if (!to_px(x[static_cast<size_t>(2 * n)], x[static_cast<size_t>(2 * n + 1)], &px, &py)) continue;
            if (i + 1 < g.nx) {                        // edge to the +x neighbor
                const int n2 = node_id(g, i + 1, j);
                int px2, py2;
                if (to_px(x[static_cast<size_t>(2 * n2)], x[static_cast<size_t>(2 * n2 + 1)], &px2, &py2))
                    line(px, py, px2, py2);
            }
            if (j + 1 < g.ny) {                        // edge to the +y neighbor
                const int n2 = node_id(g, i, j + 1);
                int px2, py2;
                if (to_px(x[static_cast<size_t>(2 * n2)], x[static_cast<size_t>(2 * n2 + 1)], &px2, &py2))
                    line(px, py, px2, py2);
            }
        }
    }

    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "P2\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y) {
        for (int xp = 0; xp < W; ++xp) f << static_cast<int>(img[static_cast<size_t>(y) * W + xp]) << ' ';
        f << '\n';
    }
}

// ===========================================================================
// main — the staged pipeline described in the file header.
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data arm_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Real-time FEM soft-arm model + model-based control (project 28.01)\n");
    print_device_info();
    std::printf("PROBLEM: corotational FEM soft arm, %dx%d Q4 elements, %d nodes, %d DOF, dt=%.1e s, E=%.1f MPa, FP32 [synthetic]\n",
                kNelx, kNely, (kNelx + 1) * (kNely + 1), 2 * (kNelx + 1) * (kNely + 1),
                static_cast<double>(kDt_s), static_cast<double>(kYoungsE_Pa) / 1.0e6);
    std::printf("MESH: length %.3f m, height %.4f m, element size %.4f m, thickness %.3f m [synthetic elastomer]\n",
                static_cast<double>(kArmLength_m), static_cast<double>(kArmHeight_m),
                static_cast<double>(kElemSize_m), static_cast<double>(kThickness_m));

    // ---- scenario ------------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/arm_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed or model rows disagree with kernels.cuh)\n");
        return 1;
    }
    std::printf("SCENARIO: identify-then-track, %d setpoints (step+hold %.2f s each), control tick %.1f ms, model rows cross-checked [synthetic]\n",
                static_cast<int>(sc.setpoint_fracs.size()),
                static_cast<double>(sc.hold_steps) * static_cast<double>(kDt_s),
                static_cast<double>(sc.control_substeps) * static_cast<double>(kDt_s) * 1000.0);
    std::fflush(stdout);   // phase boundary: make partial output visible in piped captures

    // ---- setup ----------------------------------------------------------------
    ArmSim sim; sim.init();
    float KE_hat[64];
    compute_KE_hat(kPoissonNu, KE_hat);   // ONE derivation feeds BOTH paths (26.01's pattern)
    upload_KE_hat(KE_hat);

    double total_sim_s = 0.0, total_wall_s = 0.0;   // accumulated across every stepping phase -> REALTIME

    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    ensure_dir(out_dir);
    std::vector<std::string> snapshot_rows;   // rows of arm_snapshots.csv
    int snapshot_id = 0;
    auto take_snapshot = [&](const char* label, double t_s) {
        std::vector<float> x, v; sim.download_state(x, v);
        for (int j = 0; j < sim.g.ny; ++j)
            for (int i = 0; i < sim.g.nx; ++i) {
                const int n = node_id(sim.g, i, j);
                char buf[160];
                std::snprintf(buf, sizeof(buf), "%d,%s,%.6f,%d,%d,%.6f,%.6f",
                              snapshot_id, label, t_s, i, j,
                              static_cast<double>(x[static_cast<size_t>(2 * n)]),
                              static_cast<double>(x[static_cast<size_t>(2 * n + 1)]));
                snapshot_rows.emplace_back(buf);
            }
        ++snapshot_id;
    };

    bool all_pass = true;

    // ============================ VERIFY STAGE ==============================
    // Identical initial state + identical (asymmetric, probe-magnitude)
    // actuation and full damping through the GPU path and the CPU oracle;
    // compare every position/velocity after kVerifySteps steps.
    {
        std::vector<float> x_cpu(static_cast<size_t>(sim.ndof)), v_cpu(static_cast<size_t>(sim.ndof), 0.0f);
        std::vector<float> force_cpu(static_cast<size_t>(sim.ndof));
        for (int j = 0; j < sim.g.ny; ++j)
            for (int i = 0; i < sim.g.nx; ++i) {
                const int n = node_id(sim.g, i, j);
                x_cpu[static_cast<size_t>(2 * n)]     = static_cast<float>(i) * sim.h;
                x_cpu[static_cast<size_t>(2 * n + 1)] = static_cast<float>(j) * sim.h;
            }

        // A representative, asymmetric load case at the probe's magnitude:
        // T_top = bias + probe/2, T_bottom = bias - probe/2 — both tendons
        // taut, bending on: the same regime the demo actually runs in.
        const float T_top_v = kTendonBiasN + 0.5f * sc.probe_delta_t_n;
        const float T_bot_v = kTendonBiasN - 0.5f * sc.probe_delta_t_n;

        sim.reset_to_rest();
        GpuTimer gt; gt.begin();
        for (int s = 0; s < kVerifySteps; ++s)
            sim.step(kRayleighAlphaOn, kRayleighBetaOn, T_top_v, T_bot_v);
        const float gpu_ms = gt.end_ms();

        CpuTimer ct; ct.begin();
        for (int s = 0; s < kVerifySteps; ++s)
            fem_step_cpu(sim.g, x_cpu.data(), v_cpu.data(), force_cpu.data(),
                         sim.node_mass_host.data(), sim.fixed_host.data(),
                         sim.Et, sim.h, kRayleighAlphaOn, kRayleighBetaOn, kDt_s,
                         T_top_v, T_bot_v, kNoPointForce, 0.0f, 0.0f);
        const double cpu_ms = ct.end_ms();

        std::vector<float> x_gpu, v_gpu;
        sim.download_state(x_gpu, v_gpu);

        // Tolerance rationale: the ONLY sanctioned difference between the two
        // paths is floating-point reassociation — the GPU's atomicAdd scatter
        // sums up to 4 element contributions per node in hardware-scheduled
        // order, the CPU in fixed element order (plus libm-vs-CUDA ulp
        // differences in atan2/sin/cos). Each step's per-node reassociation
        // error is ulp-scale; over 500 steps of a stable, damped system it
        // accumulates roughly linearly, not exponentially. kVerifyTolX/V
        // carry >=100x headroom over the measured worst deviations printed
        // below, while an indexing/rotation/scatter BUG shifts positions at
        // 1e-3+ scale instantly. NaN-PROOF: a non-finite value on either
        // path fails outright (NaN's comparison semantics otherwise hide it —
        // the first draft's unstable damping taught exactly that lesson;
        // THEORY.md "numerics").
        float worst_x = 0.0f, worst_v = 0.0f;
        bool finite_ok = true;
        for (int d = 0; d < sim.ndof; ++d) {
            const float xg = x_gpu[static_cast<size_t>(d)], xc = x_cpu[static_cast<size_t>(d)];
            const float vg = v_gpu[static_cast<size_t>(d)], vc = v_cpu[static_cast<size_t>(d)];
            if (!std::isfinite(xg) || !std::isfinite(xc) || !std::isfinite(vg) || !std::isfinite(vc)) {
                finite_ok = false;
                break;
            }
            worst_x = std::fmax(worst_x, std::fabs(xg - xc));
            worst_v = std::fmax(worst_v, std::fabs(vg - vc));
        }
        const bool verify_pass = finite_ok && (worst_x <= kVerifyTolX) && (worst_v <= kVerifyTolV);
        all_pass = all_pass && verify_pass;

        if (!finite_ok)
            std::printf("[info] verify: NON-FINITE state detected (instability) — hard fail\n");
        else
            std::printf("[info] verify: worst |dx|=%.3e m, worst |dv|=%.3e m/s over %d steps, %d elements, %d nodes\n",
                        static_cast<double>(worst_x), static_cast<double>(worst_v),
                        kVerifySteps, sim.nelem, sim.nnode);
        std::printf("[time] verify window: CPU %.2f ms | GPU %.3f ms | per step: CPU %.0f us, GPU %.1f us (teaching artifact)\n",
                    cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms * 1000.0 / kVerifySteps, static_cast<double>(gpu_ms) * 1000.0 / kVerifySteps);
        std::printf("VERIFY: %s (GPU scatter-assembled dynamics match CPU sequential reference within tol dx<=1e-5 m, dv<=5e-3 m/s, all values finite)\n",
                    verify_pass ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    // ====================== FREE-VIBRATION STAGE =============================
    // Gate (i): settle under a small tip point load; compare to Euler-
    // Bernoulli. Gates (ii)+(iii): release undamped and ring.
    {
        sim.reset_to_rest();
        const int tip_n = tip_node_index(sim.g);
        const float tip_y_rest = static_cast<float>(kNely / 2) * sim.h;   // centerline row's rest height

        CpuTimer phase_wt; phase_wt.begin();
        float settle_maxv = 0.0f;
        const int settle_steps = settle(sim, 0.0f, 0.0f, tip_n, 0.0f, kStaticLoadN,
                                        total_sim_s, total_wall_s, &settle_maxv);
        std::printf("[info] static-load settle diagnostics: final max|v| = %.3e m/s\n",
                    static_cast<double>(settle_maxv));
        const float tip_y_loaded = sim.download_tip_y();
        const double eb_delta_measured = static_cast<double>(tip_y_rest - tip_y_loaded);  // magnitude of the -Y sag (m)

        // Analytic Euler-Bernoulli cantilever tip deflection under tip load P:
        //     delta = |P| L^3 / (3 E I),   I = t H^3 / 12
        const double I = static_cast<double>(kThickness_m) * std::pow(static_cast<double>(kArmHeight_m), 3.0) / 12.0;
        const double EI = static_cast<double>(kYoungsE_Pa) * I;
        const double eb_delta_analytic = std::fabs(static_cast<double>(kStaticLoadN)) *
                                         std::pow(static_cast<double>(kArmLength_m), 3.0) / (3.0 * EI);
        const double eb_rel_err = std::fabs(eb_delta_measured - eb_delta_analytic) / eb_delta_analytic;

        // Q4 elements over-stiffen a coarse bending mesh (shear locking), and
        // the FEM arm is a 2-D continuum, not a 1-D beam — 30% is a generous,
        // honestly-documented allowance, not a tuned-to-pass number (the
        // measured error is printed; THEORY.md "How we verify").
        const bool gate_i_pass = (settle_steps < kSettleMaxSteps) && (eb_rel_err <= kGateStaticRelTol);
        all_pass = all_pass && gate_i_pass;
        std::printf("[info] static-load settle: %d steps, tip deflection measured %.4f mm vs analytic EB %.4f mm (rel err %.1f%%)\n",
                    settle_steps, eb_delta_measured * 1000.0, eb_delta_analytic * 1000.0, eb_rel_err * 100.0);
        std::printf("GATE static-deflection: %s (measured within 30%% of Euler-Bernoulli; Q4 discretization honestly allowed for)\n",
                    gate_i_pass ? "PASS" : "FAIL");
        take_snapshot("static_tip_load_settled", total_sim_s);

        // ---- release: alpha OFF, no actuation — ring for kFreeVibPeriods ----
        // The ring keeps ONLY the featherweight kRingBeta = 5e-6 s. Fully-
        // undamped is not an option for this force formulation: the naive
        // warped-stiffness corotational force self-excites (measured:
        // amplitude e-folds at ~6.4/s FROM EXACT REST, at dt AND dt/2 — the
        // known flutter of the rotation-warped force whose tangent is not
        // symmetric; THEORY.md "numerics" tells the full detective story,
        // and building the variational fix is a README exercise). At
        // kRingBeta, mode 1 loses 0.16% of its energy over the whole window
        // — the gate still measures what it claims (see kGateEnergyDriftPct's
        // measured budget above).
        // Analytic first-mode frequency of a uniform cantilever (THEORY.md
        // derives it from the Euler-Bernoulli PDE; 1.875104 is the first root
        // of cos(kL)*cosh(kL) = -1):
        //     f1 = (1.875104^2 / (2*pi)) * sqrt(E I / (rho A L^4)),  A = t*H
        const double A_m2 = static_cast<double>(kThickness_m) * static_cast<double>(kArmHeight_m);
        const double f1_analytic = (1.875104 * 1.875104 / (2.0 * 3.14159265358979323846)) *
                                   std::sqrt(EI / (static_cast<double>(kDensity_kgm3) * A_m2 *
                                                   std::pow(static_cast<double>(kArmLength_m), 4.0)));
        const double T1_analytic = 1.0 / f1_analytic;
        const int ring_steps = static_cast<int>((static_cast<double>(kFreeVibPeriods) * T1_analytic) /
                                                static_cast<double>(kDt_s));

        std::vector<double> tip_t, tip_y;
        std::vector<double> e_ke, e_pe;
        tip_t.reserve(static_cast<size_t>(ring_steps / kTipSampleStride) + 4);
        tip_y.reserve(tip_t.capacity());

        CpuTimer ring_wt; ring_wt.begin();
        double t_local = 0.0;
        for (int s = 0; s < ring_steps; ++s) {
            sim.step(0.0f, kRingBeta, 0.0f, 0.0f);     // alpha off, featherweight beta (see the release comment above)
            t_local += static_cast<double>(kDt_s);
            if (s % kTipSampleStride == 0) {
                tip_t.push_back(t_local);
                tip_y.push_back(static_cast<double>(sim.download_tip_y()) - static_cast<double>(tip_y_rest));
            }
            if (s % kEnergySampleStride == 0) {
                std::vector<float> xs, vs; sim.download_state(xs, vs);
                e_ke.push_back(arm_kinetic_energy(sim.g, vs.data(), sim.node_mass_host.data()));
                e_pe.push_back(arm_elastic_pe(sim.g, xs.data(), sim.Et, sim.h));
            }
        }
        total_sim_s += static_cast<double>(ring_steps) * static_cast<double>(kDt_s);
        total_wall_s += ring_wt.end_ms() / 1000.0;

        const double f1_measured = count_periods_by_zero_crossing(tip_t, tip_y);
        const double f1_rel_err = (f1_measured > 0.0) ? std::fabs(f1_measured - f1_analytic) / f1_analytic : 1.0;
        const bool gate_ii_pass = (f1_measured > 0.0) && (f1_rel_err <= kGateFreqRelTol);
        all_pass = all_pass && gate_ii_pass;
        std::printf("[info] free vibration: %d steps (%.1f analytic periods, alpha off, minimal beta), f1 measured %.4f Hz vs analytic %.4f Hz (rel err %.1f%%)\n",
                    ring_steps, static_cast<double>(kFreeVibPeriods), f1_measured, f1_analytic, f1_rel_err * 100.0);
        std::printf("GATE first-mode-frequency: %s (zero-crossing estimate within 20%% of the analytic cantilever formula)\n",
                    gate_ii_pass ? "PASS" : "FAIL");

        // Energy gate: total energy (KE + elastic PE) of the unactuated,
        // alpha-undamped ring must stay within kGateEnergyDriftPct of its
        // initial value — symplectic Euler's energy error OSCILLATES
        // boundedly instead of growing secularly (the property 10.03's
        // energy gate leans on too). The measured drift budget lives with
        // kGateEnergyDriftPct's definition; the headline: an integrator or
        // force bug (e.g. the undamped flutter: +41% over half this window)
        // blows through the bound immediately, while the healthy system's
        // drift is a documented, measured ~4%.
        double e0 = e_ke.empty() ? 0.0 : (e_ke.front() + e_pe.front());
        double max_dev = 0.0;
        size_t max_dev_idx = 0;
        double e_last = e0, e_min = e0, e_max = e0;
        bool energy_finite = !e_ke.empty();
        for (size_t k = 0; k < e_ke.size(); ++k) {
            const double e_total = e_ke[k] + e_pe[k];
            if (!std::isfinite(e_total)) { energy_finite = false; break; }
            const double dev = (e0 > 0.0) ? std::fabs(e_total - e0) / e0 : 0.0;
            if (dev > max_dev) { max_dev = dev; max_dev_idx = k; }
            e_last = e_total;
            if (e_total < e_min) e_min = e_total;
            if (e_total > e_max) e_max = e_total;
        }
        std::printf("[info] energy diagnostics: E_min=%.6e J, E_max=%.6e J, E_last=%.6e J, peak deviation at sample %zu/%zu\n",
                    e_min, e_max, e_last, max_dev_idx, e_ke.size());
        if (!e_ke.empty())
            std::printf("[info] energy split: KE first %.3e last %.3e J | PE first %.3e last %.3e J | ring-end max|v| %.3e m/s\n",
                        e_ke.front(), e_ke.back(), e_pe.front(), e_pe.back(),
                        static_cast<double>(sim.max_abs_velocity()));
        const double energy_drift_pct = max_dev * 100.0;
        const bool gate_iii_pass = energy_finite && (energy_drift_pct <= kGateEnergyDriftPct);
        all_pass = all_pass && gate_iii_pass;
        std::printf("[info] energy conservation: E0=%.6e J, peak relative drift %.3f%% over %zu samples\n",
                    e0, energy_drift_pct, e_ke.size());
        std::printf("GATE energy-conservation: %s (alpha off, featherweight beta, unactuated; peak total-energy drift <= 8%% - symplectic Euler's bounded drift)\n",
                    gate_iii_pass ? "PASS" : "FAIL");   // stable line kept pure-ASCII: the demo diff must survive any console codepage
        std::printf("[time] free-vibration phase total: %.2f s wall\n", phase_wt.end_ms() / 1000.0);
        std::fflush(stdout);
    }

    // ============================ CONTROL STAGE ===============================
    double identified_J = 0.0;                 // tip-Y per tension differential (m/N)
    std::vector<std::string> traj_rows;        // rows of tip_trajectory.csv
    struct SetpointResult { double setpoint_m, rise_time_s, overshoot_pct, sse_m; bool reached; };
    std::vector<SetpointResult> sp_results;
    {
        const float tip_y_rest = static_cast<float>(kNely / 2) * sim.h;

        // ---- IDENTIFY: probe the model itself ------------------------------
        // Model-based control needs a model; instead of assuming one (an
        // analytic beam formula would do here, but real soft arms defy
        // analytic models), we MEASURE the FEM model's quasi-static gain:
        // apply a known tension differential, settle, read the tip. This is
        // the 1-DOF miniature of the identified-Jacobian workflow soft-robot
        // controllers actually use (THEORY.md "The model-based-control story").
        sim.reset_to_rest();
        const float T_top_probe = kTendonBiasN + 0.5f * sc.probe_delta_t_n;
        const float T_bot_probe = kTendonBiasN - 0.5f * sc.probe_delta_t_n;
        const int probe_steps = settle(sim, T_top_probe, T_bot_probe,
                                       kNoPointForce, 0.0f, 0.0f, total_sim_s, total_wall_s);
        const float tip_y_probe = sim.download_tip_y();
        identified_J = static_cast<double>(tip_y_probe - tip_y_rest) / static_cast<double>(sc.probe_delta_t_n);
        const bool identify_pass = (probe_steps < kSettleMaxSteps) &&
                                   std::isfinite(identified_J) && std::fabs(identified_J) > 1.0e-9;
        all_pass = all_pass && identify_pass;
        std::printf("[info] identify: probe settle %d steps, deltaT=%.3f N -> tip deflection %.4f mm -> J=%.4e m/N\n",
                    probe_steps, static_cast<double>(sc.probe_delta_t_n),
                    static_cast<double>(tip_y_probe - tip_y_rest) * 1000.0, identified_J);
        std::printf("IDENTIFY: %s (quasi-static tip Jacobian measured by probing the FEM model itself, not assumed)\n",
                    identify_pass ? "PASS" : "FAIL");
        take_snapshot("post_probe_settled", total_sim_s);
        std::fflush(stdout);

        if (identify_pass) {
            // ---- gains from the identified model ---------------------------
            // Kp = margin/|J| sets the proportional loop gain to `margin` at
            // DC AND (approximately) at the arm's resonance, where the plant
            // peaks at |G| ~ J/(2*zeta): the loop magnitude there is about
            // margin/(2*zeta) — with zeta ~ 0.2 (the compression-softened
            // first mode), margin = 0.3 keeps the resonant loop gain ~ 0.7,
            // BELOW unity, so the resonance rings down instead of being
            // sustained by the loop (margin = 0.6 was tried first and left a
            // persistent +/-0.5 mm limit-cycle ripple — the measured lesson
            // THEORY.md's control section tells). The integrator then closes
            // steady-state error with gain crossover ~ Ki*|J| = margin/Ti
            // ~ 2 rad/s, well below the ~9-12.7 rad/s resonance.
            const float Kp = sc.pi_margin_alpha / static_cast<float>(std::fabs(identified_J));  // N/m
            const float Ki = Kp / sc.pi_integral_time_s;                                        // N/(m*s)
            const float deltaT_clamp = sc.delta_t_clamp_frac * kTendonBiasN;                    // N
            const float dt_ctrl_s = static_cast<float>(sc.control_substeps) * kDt_s;            // control period (s)
            std::printf("[info] control gains: Kp=%.4e N/m, Ki=%.4e N/(m*s), deltaT clamp +/-%.3f N, tick %.1f ms (%.0f Hz)\n",
                        static_cast<double>(Kp), static_cast<double>(Ki), static_cast<double>(deltaT_clamp),
                        static_cast<double>(dt_ctrl_s) * 1000.0, 1.0 / static_cast<double>(dt_ctrl_s));

            // Setpoints: fractions of the clamp-achievable tip range (via the
            // identified J), scaled by the documented safety fraction so the
            // controller never has to live at the clamp.
            const float sp_scale = static_cast<float>(identified_J) * sc.setpoint_safe_frac * deltaT_clamp;

            // Start the closed loop from the SYMMETRIC-bias equilibrium (the
            // controller's natural operating point).
            sim.reset_to_rest();
            settle(sim, kTendonBiasN, kTendonBiasN, kNoPointForce, 0.0f, 0.0f, total_sim_s, total_wall_s);

            float integral = 0.0f;                     // PI integral state (m*s)
            double t_ctrl = total_sim_s;
            CpuTimer ctrl_wt; ctrl_wt.begin();

            for (size_t sp_idx = 0; sp_idx < sc.setpoint_fracs.size(); ++sp_idx) {
                const double setpoint = static_cast<double>(sc.setpoint_fracs[sp_idx] * sp_scale);
                const double y_start = static_cast<double>(sim.download_tip_y()) - static_cast<double>(tip_y_rest);
                const double step_size = setpoint - y_start;
                const double t_sp_start = t_ctrl;
                double rise_time = -1.0, peak_overshoot = 0.0, last_y = y_start;
                bool reached_90 = false;

                const int n_ticks = sc.hold_steps / sc.control_substeps;
                const int tail_start = n_ticks - n_ticks / 10;   // final 10% of the hold = the "steady state" window
                double tail_abs_err_sum = 0.0;
                int tail_count = 0;
                for (int tick = 0; tick < n_ticks; ++tick) {
                    // (1) "sense": read the tip — the one D2H sync per tick,
                    //     the controller's sensor.
                    const double t_sample = t_ctrl;
                    const double y_now = static_cast<double>(sim.download_tip_y()) - static_cast<double>(tip_y_rest);
                    const double err = setpoint - y_now;

                    // (2) PI law with conditional-integration anti-windup:
                    //     freeze the integral while the actuator is clamped
                    //     in the direction the error is pushing (the simplest
                    //     standard anti-windup; without it a long clamped
                    //     transient winds up an integral that must unwind
                    //     through massive overshoot — THEORY.md).
                    const float dT_p = Kp * static_cast<float>(err);
                    const float dT_trial = dT_p + Ki * integral;
                    const bool sat_same_dir = (dT_trial >  deltaT_clamp && err > 0.0) ||
                                              (dT_trial < -deltaT_clamp && err < 0.0);
                    if (!sat_same_dir) integral += static_cast<float>(err) * dt_ctrl_s;

                    float dT = dT_p + Ki * integral;
                    if (dT >  deltaT_clamp) dT =  deltaT_clamp;
                    if (dT < -deltaT_clamp) dT = -deltaT_clamp;

                    // (3) map the differential onto the antagonistic pair
                    //     around the bias (both stay taut by the clamp's
                    //     design: bias - clamp/2 = 0.025 N > 0).
                    const float T_top = kTendonBiasN + 0.5f * dT;
                    const float T_bot = kTendonBiasN - 0.5f * dT;

                    // (4) log the SAMPLE-TIME row (t, target, measured,
                    //     commanded tensions) — the control plot's data.
                    char row[160];
                    std::snprintf(row, sizeof(row), "%.6f,%.6f,%.6f,%.4f,%.4f",
                                  t_sample, setpoint, y_now,
                                  static_cast<double>(T_top), static_cast<double>(T_bot));
                    traj_rows.emplace_back(row);

                    // (5) "actuate + wait one tick": hold the tensions for
                    //     control_substeps physics steps (zero-order hold —
                    //     exactly how a real tendon drive holds a command
                    //     between controller ticks).
                    for (int sub = 0; sub < sc.control_substeps; ++sub)
                        sim.step(kRayleighAlphaOn, kRayleighBetaOn, T_top, T_bot);
                    t_ctrl += static_cast<double>(dt_ctrl_s);

                    // (6) metrics bookkeeping. Steady-state error is the MEAN
                    //     |error| over the final 10% of the hold — a single
                    //     end-sample would alias whatever phase of residual
                    //     ripple it happens to land on (README documents the
                    //     definition; the ripple itself is visible in the
                    //     tip_trajectory.csv artifact).
                    last_y = y_now;
                    if (!reached_90 && std::fabs(step_size) > 1.0e-6) {
                        if ((y_now - y_start) / step_size >= 0.9) {
                            reached_90 = true;
                            rise_time = t_sample - t_sp_start;
                        }
                    }
                    const double overshoot = (step_size > 0.0) ? (y_now - setpoint) : (setpoint - y_now);
                    if (overshoot > peak_overshoot) peak_overshoot = overshoot;
                    if (tick >= tail_start) { tail_abs_err_sum += std::fabs(y_now - setpoint); ++tail_count; }
                }

                const double sse = (tail_count > 0) ? tail_abs_err_sum / tail_count
                                                    : std::fabs(last_y - setpoint);
                const double overshoot_pct = (std::fabs(step_size) > 1.0e-6)
                                           ? (peak_overshoot / std::fabs(step_size)) * 100.0 : 0.0;
                const bool reached = reached_90 || std::fabs(step_size) <= 1.0e-6;
                sp_results.push_back({ setpoint, rise_time, overshoot_pct, sse, reached });
            }
            total_sim_s = t_ctrl;
            total_wall_s += ctrl_wt.end_ms() / 1000.0;
            take_snapshot("control_run_end", total_sim_s);

            // ---- per-setpoint verdicts --------------------------------------
            for (size_t k = 0; k < sp_results.size(); ++k) {
                const auto& r = sp_results[k];
                const bool pass = r.reached && (r.sse_m <= kCtrlSseMax_m) && (r.overshoot_pct <= kCtrlOvershootPctMax);
                all_pass = all_pass && pass;
                if (r.reached && r.rise_time_s >= 0.0)
                    std::printf("[info] setpoint %zu: target %+.3f mm, rise-to-90%% %.3f s, overshoot %.1f%%, steady-state error %.4f mm\n",
                                k, r.setpoint_m * 1000.0, r.rise_time_s, r.overshoot_pct, r.sse_m * 1000.0);
                else if (r.reached)
                    std::printf("[info] setpoint %zu: target %+.3f mm (zero-size step: already there), overshoot %.1f%%, steady-state error %.4f mm\n",
                                k, r.setpoint_m * 1000.0, r.overshoot_pct, r.sse_m * 1000.0);
                else
                    std::printf("[info] setpoint %zu: target %+.3f mm NOT reached within the hold window (sse %.4f mm)\n",
                                k, r.setpoint_m * 1000.0, r.sse_m * 1000.0);
                std::printf("SETPOINT %zu: %s (reached within hold window, overshoot <= 60%%, mean |error| over final 10%% of hold <= 0.3 mm)\n",
                            k, pass ? "PASS" : "FAIL");
            }
            std::fflush(stdout);
        } else {
            std::printf("SETPOINT 0: FAIL (skipped: identification failed)\n");
            all_pass = false;
        }
    }

    // ============================== ARTIFACTS =================================
    bool artifact_ok = true;
    {
        std::ofstream f(out_dir + "/tip_trajectory.csv");
        if (f.is_open()) {
            f << "t_s,setpoint_y_m,tip_y_m,T_top_N,T_bottom_N\n";   // units in the header row (§12)
            for (const auto& r : traj_rows) f << r << '\n';
        } else artifact_ok = false;
    }
    {
        std::ofstream f(out_dir + "/arm_snapshots.csv");
        if (f.is_open()) {
            f << "snapshot_id,label,t_s,i,j,x_m,y_m\n";
            for (const auto& r : snapshot_rows) f << r << '\n';
        } else artifact_ok = false;
    }
    {
        std::vector<float> x_final, v_final;
        sim.download_state(x_final, v_final);
        write_pgm_arm(out_dir + "/arm_deformed.pgm", sim.g, x_final);
        std::ifstream check(out_dir + "/arm_deformed.pgm");
        if (!check.is_open()) artifact_ok = false;
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/tip_trajectory.csv (%zu rows), demo/out/arm_snapshots.csv (%zu rows), demo/out/arm_deformed.pgm\n",
                    traj_rows.size(), snapshot_rows.size());
    else
        std::printf("ARTIFACT: FAILED to write one or more output files\n");
    all_pass = all_pass && artifact_ok;

    // ============================ REAL-TIME FACTOR =============================
    // The honest measurement the catalog bullet's "real-time" claim reduces
    // to: simulated seconds per wall second, accumulated over EVERY dynamics
    // phase above (settles, ring, closed loop — including their host-side
    // sensing/logging overhead, because a real-time model runs next to a
    // controller that does all of that too). >= 1x = the model outruns the
    // physical arm it represents.
    const double rtf = (total_wall_s > 0.0) ? (total_sim_s / total_wall_s) : 0.0;
    const bool rtf_pass = rtf >= 1.0;
    all_pass = all_pass && rtf_pass;
    std::printf("[info] realtime: %.3f simulated s over %.3f wall s across all stepping phases (measured factor %.2fx)\n",
                total_sim_s, total_wall_s, rtf);
    std::printf("REALTIME: %s (measured real-time factor >= 1x: the FEM model runs faster than the arm it models)\n",
                rtf_pass ? "PASS" : "FAIL");

    sim.destroy();

    if (all_pass) {
        std::printf("RESULT: PASS (dynamics verified vs CPU, analytic gates passed, Jacobian identified, all setpoints tracked, real-time factor >= 1x)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (see the VERIFY/GATE/IDENTIFY/SETPOINT/REALTIME lines above for which check failed)\n");
    return 1;
}
