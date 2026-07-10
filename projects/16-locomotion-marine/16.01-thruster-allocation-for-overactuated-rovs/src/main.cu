// ===========================================================================
// main.cu — entry point for project 16.01
//           Thruster allocation for overactuated ROVs (batched QP)
//
// What this program does, start to finish
// -----------------------------------------
//   0. SETUP (once): build the 6x8 allocation matrix B from the vehicle's
//      thruster geometry (kernels.cuh), form the QP's Hessian H and BtW2,
//      estimate the step size 1/L by host power iteration, upload H/BtW2 to
//      GPU __constant__ memory.
//   1. Load the committed wrench batch (data/sample/wrench_batch.csv) — a
//      synthetic docking-under-current wrench trajectory.
//   2. VERIFY STAGE (the §5 GPU-vs-CPU gate): allocate the WHOLE batch on
//      the GPU kernel AND the CPU oracle from identical inputs; require
//      element-wise agreement.
//   3. OPTIMALITY GATES — the mathematical ground truth, not just "GPU
//      matches CPU":
//        3a. classify each solved wrench as SATURATED (some thruster at its
//            limit) or UNSATURATED;
//        3b. UNSATURATED rows must match the closed-form damped weighted
//            PSEUDOINVERSE allocation (a direct 8x8 Cholesky solve);
//        3c. SATURATED rows must have a near-zero projected-gradient (KKT)
//            residual at the returned point;
//        3d. the QP OBJECTIVE must be monotone non-increasing over
//            iterations on the project's motivating example wrench.
//   4. ARTIFACT — write demo/out/allocation.csv: commanded vs. achieved
//      wrench and all 8 thruster forces, per row of the batch.
//   5. FAILURE ANALYSIS — re-run the SAME wrench batch nine times: once
//      nominal, then once per thruster with that thruster's box forced to
//      [0,0] (a seized/dead thruster). Measure achievable-wrench
//      degradation per configuration; write demo/out/failure_analysis.csv.
//   6. REPORT: PASS only if every gate above holds and both artifacts wrote.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:", "VERIFY:",
// "GATE-PSEUDOINV:", "GATE-KKT:", "GATE-MONOTONE:", "ARTIFACT:", "RESULT:" —
// "[info]"/"[time]" lines are NOT diffed. Change a stable line -> update
// demo/expected_output.txt in the same change.
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
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Verification tolerances — every number below is justified in THEORY.md
// "how we verify correctness" and re-measured (not assumed) by the [info]
// lines this program prints; these constants are the PASS/FAIL thresholds,
// deliberately set with headroom over the measured worst case so ordinary
// FP32/GPU-architecture noise cannot flip the verdict.
// ---------------------------------------------------------------------------
static const float kVerifyTolN      = 5e-3f;  // GPU-vs-CPU max |force| deviation (N); both
                                              // paths run the IDENTICAL fixed-iteration FP32
                                              // algorithm, so agreement should be near-exact —
                                              // the tolerance only absorbs FMA-contraction and
                                              // instruction-scheduling order differences between
                                              // nvcc and cl.exe (THEORY.md "numerical considerations").
static const float kPseudoinvTolN   = 0.05f;  // unsaturated QP solution vs. closed-form damped
                                              // pseudoinverse, max |force| deviation (N) — mostly
                                              // FP32 accumulation over kPgdIters sequential steps.
static const float kKktTol          = 0.05f;  // saturated-row projected-gradient (KKT) residual
                                              // norm, in force units (N) — see THEORY.md for the
                                              // exact definition and the measured worst case.
static const float kSaturationSlackN = 0.05f; // a thruster counts as "at its limit" if
                                              // |u_i| >= u_max_i - this slack (N).
static const float kDegradedThresh  = 0.05f;  // failure-analysis: a wrench counts as
                                              // "significantly degraded" if its relative
                                              // tracking error exceeds this fraction.

// ---------------------------------------------------------------------------
// load_wrench_batch — strict CSV loader for the committed wrench trajectory.
// Format (written by scripts/make_synthetic.py, documented in data/README.md):
//   '#'-prefixed lines are comments; one header row "t_s,Fx_N,Fy_N,Fz_N,
//   Mx_Nm,My_Nm,Mz_Nm"; then one row per sample. Any malformed row aborts
//   loading loudly (fail-fast beats silently truncating a batch — CLAUDE.md
//   §13) rather than returning a partial, silently-wrong result.
// ---------------------------------------------------------------------------
struct WrenchBatch {
    std::vector<float> t_s;     // [count] sample timestamps (s) — for the artifact only
    std::vector<float> tau;     // [count*kNDof] flattened wrenches, kernels.cuh layout
    int count = 0;
    bool loaded = false;
};

static WrenchBatch load_wrench_batch(const std::string& path)
{
    WrenchBatch wb;
    std::ifstream in(path);
    if (!in.is_open()) return wb;

    std::string line;
    bool seen_header = false;
    while (std::getline(in, line)) {
        // Strip a trailing '\r' (files may have been checked out with CRLF).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (!seen_header) {
            // The header row is fixed and known; skip it by content, not by
            // position, so blank/comment lines before it are harmless.
            if (line == "t_s,Fx_N,Fy_N,Fz_N,Mx_Nm,My_Nm,Mz_Nm") { seen_header = true; continue; }
            std::fprintf(stderr, "wrench batch: expected header row, got '%s'\n", line.c_str());
            return WrenchBatch{};
        }
        std::stringstream ss(line);
        std::string cell;
        float vals[7];
        for (int i = 0; i < 7; ++i) {
            if (!std::getline(ss, cell, ',')) {
                std::fprintf(stderr, "wrench batch: short row '%s'\n", line.c_str());
                return WrenchBatch{};
            }
            vals[i] = std::strtof(cell.c_str(), nullptr);
        }
        wb.t_s.push_back(vals[0]);
        for (int d = 0; d < kNDof; ++d) wb.tau.push_back(vals[1 + d]);
        wb.count++;
    }
    if (!seen_header || wb.count < 1) {
        std::fprintf(stderr, "wrench batch: missing header or no data rows\n");
        return WrenchBatch{};
    }
    wb.loaded = true;
    return wb;
}

static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_wrench_batch(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/wrench_batch.csv");
    candidates.push_back("data/sample/wrench_batch.csv");
    candidates.push_back("../data/sample/wrench_batch.csv");
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
// matvec_Bu — achieved wrench B*u for one problem (host-side, plain loop;
// used throughout for artifacts/gates — never the hot batch path, so clarity
// wins outright over any micro-optimization).
// ---------------------------------------------------------------------------
static void matvec_Bu(const float* B, const float* u, float* Bu /* [kNDof] */)
{
    for (int d = 0; d < kNDof; ++d) {
        float acc = 0.0f;
        for (int i = 0; i < kNThr; ++i) acc += B[d * kNThr + i] * u[i];
        Bu[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// main — setup, verify, optimality gates, artifacts, failure analysis.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data wrench_batch.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] thruster allocation for overactuated ROVs: batched box-constrained QP (project 16.01)\n");
    print_device_info();

    // ======================= 0) SETUP (once) =================================
    // Build B from the vehicle geometry (kernels.cuh), form the QP matrices,
    // and estimate the projected-gradient step size by host power iteration
    // — all deterministic, no RNG, run once before any batch is allocated.
    float B[kNDof * kNThr];
    build_allocation_matrix(B);

    float H[kNThr * kNThr], BtW2[kNThr * kNDof], Q[kNThr * kNThr];
    build_qp_matrices(B, kWeight, kEpsReg, H, BtW2, Q);

    const float lambda_max_H = power_iteration_lambda_max(H, kNThr, kPowerIters);
    const float step = 1.0f / lambda_max_H;   // 1/L: the Lipschitz-safe step (THEORY.md "the math")

    upload_allocation_constants(H, BtW2);

    std::printf("PROBLEM: batched box-constrained QP allocation, %d thrusters -> %d DOF wrench, "
                "eps=%.2f, PGD iters=%d, step=1/L=%.6f (L=%.4f, host power iteration x%d), FP32\n",
                kNThr, kNDof, static_cast<double>(kEpsReg), kPgdIters,
                static_cast<double>(step), static_cast<double>(lambda_max_H), kPowerIters);

    // ======================= 1) load the wrench batch ========================
    const std::string wb_path = find_wrench_batch(data_path, argv[0]);
    if (wb_path.empty()) {
        std::printf("SCENARIO: NOT FOUND - data/sample/wrench_batch.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (wrench batch missing)\n");
        return 1;
    }
    std::printf("[info] wrench batch file: %s\n", wb_path.c_str());
    WrenchBatch wb = load_wrench_batch(wb_path);
    if (!wb.loaded) {
        std::printf("SCENARIO: MALFORMED - see stderr\n");
        std::printf("RESULT: FAIL (wrench batch malformed)\n");
        return 1;
    }
    const int K = wb.count;
    std::printf("SCENARIO: synthetic docking-under-current wrench trajectory, %d samples [synthetic]\n", K);

    // Nominal per-thruster limits: every problem, every thruster, +-40 N.
    std::vector<float> umax_nominal(static_cast<size_t>(K) * kNThr, kUMaxNominal);

    // ======================= persistent device buffers =======================
    float *d_tau = nullptr, *d_umax = nullptr, *d_u = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, static_cast<size_t>(K) * kNDof * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_umax, static_cast<size_t>(K) * kNThr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u, static_cast<size_t>(K) * kNThr * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_tau, wb.tau.data(), static_cast<size_t>(K) * kNDof * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_umax, umax_nominal.data(), static_cast<size_t>(K) * kNThr * sizeof(float), cudaMemcpyHostToDevice));

    // ======================= 2) VERIFY STAGE (§5 gate) ========================
    std::vector<float> u_gpu(static_cast<size_t>(K) * kNThr);
    std::vector<float> u_cpu(static_cast<size_t>(K) * kNThr);
    {
        GpuTimer gt;
        gt.begin();
        launch_thruster_allocation(K, d_tau, d_umax, d_u, step);
        const float gpu_ms = gt.end_ms();
        CUDA_CHECK(cudaMemcpy(u_gpu.data(), d_u, static_cast<size_t>(K) * kNThr * sizeof(float), cudaMemcpyDeviceToHost));

        CpuTimer ct;
        ct.begin();
        thruster_allocate_cpu(K, wb.tau.data(), umax_nominal.data(), H, BtW2, step, kPgdIters, u_cpu.data());
        const double cpu_ms = ct.end_ms();

        float worst = 0.0f;
        for (size_t i = 0; i < u_gpu.size(); ++i) {
            const float d = std::fabs(u_gpu[i] - u_cpu[i]);
            if (d > worst) worst = d;
        }
        const bool verify_pass = worst <= kVerifyTolN;
        std::printf("[info] verify: worst |GPU-CPU| thruster-force deviation %.3e N over %d problems x %d thrusters\n",
                    static_cast<double>(worst), K, kNThr);
        std::printf("[time] batch allocation (K=%d, iters=%d): CPU %.2f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                    K, kPgdIters, cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        std::printf("VERIFY: %s (GPU allocation matches CPU reference within tol %.0e N)\n",
                    verify_pass ? "PASS" : "FAIL", static_cast<double>(kVerifyTolN));
        if (!verify_pass) {
            std::printf("RESULT: FAIL (GPU/CPU allocation disagreement — fix before trusting the gates below)\n");
            return 1;
        }
    }
    // From here on, u_gpu (== u_cpu within tolerance) is THE solution used by
    // every downstream gate and artifact — no need to run either path again.
    const std::vector<float>& u = u_gpu;

    // ======================= 3a) classify saturated vs. unsaturated ==========
    std::vector<bool> saturated(K, false);
    int n_saturated = 0;
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < kNThr; ++i) {
            if (std::fabs(u[static_cast<size_t>(k) * kNThr + i]) >= kUMaxNominal - kSaturationSlackN) {
                saturated[k] = true;
                break;
            }
        }
        if (saturated[k]) ++n_saturated;
    }
    std::printf("[info] batch composition: %d/%d wrenches saturate at least one thruster, %d fully unsaturated\n",
                n_saturated, K, K - n_saturated);

    // ======================= 3b) GATE-PSEUDOINV (unsaturated rows) ===========
    // Ground truth: the UNCONSTRAINED minimizer of this exact QP is the
    // closed-form damped weighted pseudoinverse x* = Q^-1 (B^T W^2 tau)
    // (THEORY.md "the math"). Any unsaturated QP solution — one that never
    // touched a box face — solves the SAME unconstrained problem, so the two
    // must agree.
    {
        float worst = 0.0f;
        int checked = 0;
        for (int k = 0; k < K; ++k) {
            if (saturated[k]) continue;
            const float* t = &wb.tau[static_cast<size_t>(k) * kNDof];
            float b[kNThr];
            for (int i = 0; i < kNThr; ++i) {
                float acc = 0.0f;
                for (int d = 0; d < kNDof; ++d) acc += BtW2[i * kNDof + d] * t[d];
                b[i] = acc;
            }
            float x_pinv[kNThr];
            cholesky_solve_spd(Q, b, x_pinv, kNThr);
            for (int i = 0; i < kNThr; ++i) {
                const float dev = std::fabs(u[static_cast<size_t>(k) * kNThr + i] - x_pinv[i]);
                if (dev > worst) worst = dev;
            }
            ++checked;
        }
        const bool pass = worst <= kPseudoinvTolN;
        std::printf("[info] gate-pseudoinv: worst |QP - damped pseudoinverse| deviation %.3e N over %d unsaturated wrenches\n",
                    static_cast<double>(worst), checked);
        std::printf("GATE-PSEUDOINV: %s (unsaturated QP solutions match the closed-form damped weighted pseudoinverse within tol %.2f N)\n",
                    pass ? "PASS" : "FAIL", static_cast<double>(kPseudoinvTolN));
        if (!pass) { std::printf("RESULT: FAIL (pseudoinverse optimality gate failed)\n"); return 1; }
    }

    // ======================= 3c) GATE-KKT (saturated rows) ===================
    // Ground truth: at a TRUE constrained optimum, the point is a fixed point
    // of its own projected-gradient step (the box-QP's KKT condition). The
    // "projected gradient" u - clip(u - step*grad, box) must vanish.
    {
        float worst = 0.0f;
        int checked = 0;
        for (int k = 0; k < K; ++k) {
            if (!saturated[k]) continue;
            const float* t = &wb.tau[static_cast<size_t>(k) * kNDof];
            const float* uk = &u[static_cast<size_t>(k) * kNThr];
            float g[kNThr];
            for (int i = 0; i < kNThr; ++i) {
                float acc = 0.0f;
                for (int d = 0; d < kNDof; ++d) acc += BtW2[i * kNDof + d] * t[d];
                g[i] = 2.0f * acc;
            }
            float grad[kNThr];
            for (int i = 0; i < kNThr; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < kNThr; ++j) acc += H[i * kNThr + j] * uk[j];
                grad[i] = acc - g[i];
            }
            float res_sq = 0.0f;
            for (int i = 0; i < kNThr; ++i) {
                float cand = uk[i] - step * grad[i];
                cand = cand < -kUMaxNominal ? -kUMaxNominal : (cand > kUMaxNominal ? kUMaxNominal : cand);
                const float d = uk[i] - cand;
                res_sq += d * d;
            }
            const float res = std::sqrt(res_sq);
            if (res > worst) worst = res;
            ++checked;
        }
        const bool pass = worst <= kKktTol;
        std::printf("[info] gate-kkt: worst projected-gradient (KKT) residual %.3e N over %d saturated wrenches\n",
                    static_cast<double>(worst), checked);
        std::printf("GATE-KKT: %s (saturated QP solutions satisfy the box-KKT fixed point within tol %.2f N)\n",
                    pass ? "PASS" : "FAIL", static_cast<double>(kKktTol));
        if (!pass) { std::printf("RESULT: FAIL (KKT optimality gate failed)\n"); return 1; }
    }

    // ======================= 3d) GATE-MONOTONE (objective descent) ===========
    // THEORY.md's motivating "naive pseudoinverse clipping" worked example,
    // re-run here with per-iteration tracing (reference_cpu.cpp). The QP
    // objective J(u_k) must never increase, by the projected-gradient
    // descent lemma (step <= 1/L) — a small float-noise slack (kMonotoneSlack)
    // absorbs FP32 rounding without masking a real ascent.
    {
        std::vector<float> J_trace(kPgdIters + 1), residual_trace(kPgdIters + 1);
        float umax8[kNThr];
        for (int i = 0; i < kNThr; ++i) umax8[i] = kUMaxNominal;
        thruster_allocate_trace_cpu(kMotivatingWrench, umax8, B, kWeight, H, BtW2,
                                    kEpsReg, step, kPgdIters, J_trace.data(), residual_trace.data());

        // Slack justification (measured, not guessed): once J converges (here,
        // by ~iteration 35 of 500) it sits at J~1319 and every later step
        // recomputes the SAME 8x8 matvec from a numerically-converged u — pure
        // FP32 rounding noise on a value of that magnitude is ~J*2^-23 ~ 1.6e-4,
        // and the measured worst observed uptick at the reference GPU/toolchain
        // was 3.7e-4 (3 such ticks across 500 iterations, all AFTER convergence,
        // all at the float-epsilon scale for J~1300 — never a real ascent early
        // in the run, where J is still dropping by single/double digits per
        // step). kMonotoneSlack carries ~3x headroom over that measurement.
        const float kMonotoneSlack = 1e-3f;
        bool j_monotone = true, r_monotone = true;
        int j_violations = 0;
        for (int it = 1; it <= kPgdIters; ++it) {
            if (J_trace[it] > J_trace[it - 1] + kMonotoneSlack) { j_monotone = false; ++j_violations; }
            if (residual_trace[it] > residual_trace[it - 1] + kMonotoneSlack) r_monotone = false;
        }
        std::printf("[info] gate-monotone: objective J(u) went %.4f (start) -> %.6f (final) over %d iterations, "
                    "residual ||Bu-tau|| went %.4f -> %.6f; residual ALSO monotone this run: %s\n",
                    static_cast<double>(J_trace.front()), static_cast<double>(J_trace.back()), kPgdIters,
                    static_cast<double>(residual_trace.front()), static_cast<double>(residual_trace.back()),
                    r_monotone ? "yes" : "no (not guaranteed — see THEORY.md)");
        std::printf("GATE-MONOTONE: %s (QP objective J(u_k) is non-increasing over all %d iterations, %d violation(s))\n",
                    j_monotone ? "PASS" : "FAIL", kPgdIters, j_violations);
        if (!j_monotone) { std::printf("RESULT: FAIL (monotone-descent gate failed)\n"); return 1; }
    }

    // ======================= 4) ARTIFACT: allocation.csv =====================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/allocation.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "t_s,Fx_cmd_N,Fy_cmd_N,Fz_cmd_N,Mx_cmd_Nm,My_cmd_Nm,Mz_cmd_Nm,"
                 "Fx_ach_N,Fy_ach_N,Fz_ach_N,Mx_ach_Nm,My_ach_Nm,Mz_ach_Nm,"
                 "u0_N,u1_N,u2_N,u3_N,u4_N,u5_N,u6_N,u7_N,saturated\n";
            for (int k = 0; k < K; ++k) {
                const float* t = &wb.tau[static_cast<size_t>(k) * kNDof];
                const float* uk = &u[static_cast<size_t>(k) * kNThr];
                float ach[kNDof];
                matvec_Bu(B, uk, ach);
                f << wb.t_s[static_cast<size_t>(k)];
                for (int d = 0; d < kNDof; ++d) f << ',' << t[d];
                for (int d = 0; d < kNDof; ++d) f << ',' << ach[d];
                for (int i = 0; i < kNThr; ++i) f << ',' << uk[i];
                f << ',' << (saturated[k] ? 1 : 0) << '\n';
            }
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/allocation.csv (%d rows)\n", K);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/allocation.csv\n");

    // ======================= 5) FAILURE ANALYSIS ==============================
    // Re-allocate the SAME demanding batch nine times: nominal, then once per
    // thruster forced to [0,0] (seized/dead). This is the fault-tolerance
    // sweep a real ROV operator cares about: how much does losing thruster i
    // degrade the vehicle's ability to produce the commanded wrenches it
    // actually needs (THEORY.md "where this sits in the real world")?
    //
    // IMPORTANT metric choice: relative tracking error ||Bu-tau||/||tau|| is
    // NOT near-zero even in the NOMINAL (no-failure) configuration — this
    // vehicle's B matrix has two comparatively weak-authority directions
    // (THEORY.md "numerical considerations" reports their exact singular
    // values), and eps=0.1's damping trades tracking accuracy for
    // conditioning/speed along exactly those directions (the same
    // "damped-least-squares bias" every damped-pseudoinverse controller
    // accepts — 09.05's Levenberg-Marquardt IK faces the identical trade
    // near kinematic singularities). Comparing an ABSOLUTE error threshold
    // across configurations would therefore flag rows that were never
    // healthy to begin with. The metric that actually isolates a FAILURE's
    // effect is the DELTA against the nominal baseline, per row:
    //     degradation_k = rel_err_k(this config) - rel_err_k(nominal)
    // "significantly degraded" = degradation_k > kDegradedThresh (a real
    // worsening caused by the failure, not by the vehicle's baseline
    // geometry/damping bias).
    // ---------------------------------------------------------------------
    struct FailureStats {
        std::string label;
        float mean_rel_err = 0.0f, max_rel_err = 0.0f;          // absolute (vs. tau), for context
        float mean_degradation = 0.0f, max_degradation = 0.0f;  // delta vs. nominal, per row
        int n_significantly_worse = 0;   // rows whose degradation exceeds kDegradedThresh
    };
    std::vector<FailureStats> fail_stats;
    std::vector<float> rel_err_nominal(K, 0.0f);   // filled by cfg==0, read by cfg>0

    std::vector<float> umax_cfg(static_cast<size_t>(K) * kNThr);
    std::vector<float> u_cfg(static_cast<size_t>(K) * kNThr);
    for (int cfg = 0; cfg <= kNThr; ++cfg) {
        // cfg==0: nominal (no failure). cfg==i (i=1..8): thruster (i-1) locked to 0.
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < kNThr; ++i)
                umax_cfg[static_cast<size_t>(k) * kNThr + i] =
                    (cfg != 0 && i == cfg - 1) ? 0.0f : kUMaxNominal;

        CUDA_CHECK(cudaMemcpy(d_umax, umax_cfg.data(), static_cast<size_t>(K) * kNThr * sizeof(float), cudaMemcpyHostToDevice));
        launch_thruster_allocation(K, d_tau, d_umax, d_u, step);
        CUDA_CHECK(cudaMemcpy(u_cfg.data(), d_u, static_cast<size_t>(K) * kNThr * sizeof(float), cudaMemcpyDeviceToHost));

        FailureStats fs;
        fs.label = (cfg == 0) ? "nominal" : ("thruster_" + std::to_string(cfg - 1) + "_failed");
        double sum_rel = 0.0, sum_deg = 0.0;
        for (int k = 0; k < K; ++k) {
            const float* t = &wb.tau[static_cast<size_t>(k) * kNDof];
            const float* uk = &u_cfg[static_cast<size_t>(k) * kNThr];
            float ach[kNDof];
            matvec_Bu(B, uk, ach);
            float err_sq = 0.0f, tau_sq = 0.0f;
            for (int d = 0; d < kNDof; ++d) {
                const float e = ach[d] - t[d];
                err_sq += e * e;
                tau_sq += t[d] * t[d];
            }
            // Relative error against ||tau||, floored so near-zero commands
            // (station-keeping calm moments) do not blow the ratio up.
            const float tau_norm = std::sqrt(tau_sq);
            const float rel = std::sqrt(err_sq) / (tau_norm > 1.0f ? tau_norm : 1.0f);
            sum_rel += rel;
            if (rel > fs.max_rel_err) fs.max_rel_err = rel;

            if (cfg == 0) {
                rel_err_nominal[static_cast<size_t>(k)] = rel;   // establish the baseline
            } else {
                const float deg = rel - rel_err_nominal[static_cast<size_t>(k)];
                sum_deg += deg;
                if (deg > fs.max_degradation) fs.max_degradation = deg;
                if (deg > kDegradedThresh) fs.n_significantly_worse++;
            }
        }
        fs.mean_rel_err = static_cast<float>(sum_rel / K);
        fs.mean_degradation = (cfg == 0) ? 0.0f : static_cast<float>(sum_deg / K);
        std::printf("[info] failure-analysis %-22s mean_rel_err=%.4f max_rel_err=%.4f  "
                    "mean_degradation_vs_nominal=%+.4f max_degradation=%+.4f significantly_worse(>%.0fpp)=%d/%d\n",
                    fs.label.c_str(), static_cast<double>(fs.mean_rel_err), static_cast<double>(fs.max_rel_err),
                    static_cast<double>(fs.mean_degradation), static_cast<double>(fs.max_degradation),
                    static_cast<double>(kDegradedThresh) * 100.0, fs.n_significantly_worse, K);
        fail_stats.push_back(fs);
    }

    bool fa_artifact_ok = ensure_dir(out_dir);
    if (fa_artifact_ok) {
        std::ofstream f(out_dir + "/failure_analysis.csv");
        fa_artifact_ok = f.is_open();
        if (fa_artifact_ok) {
            f << "config,mean_rel_err,max_rel_err,mean_degradation_vs_nominal,max_degradation_vs_nominal,"
                 "n_significantly_worse,n_total,degradation_threshold\n";
            for (const auto& fs : fail_stats)
                f << fs.label << ',' << fs.mean_rel_err << ',' << fs.max_rel_err << ','
                  << fs.mean_degradation << ',' << fs.max_degradation << ','
                  << fs.n_significantly_worse << ',' << K << ',' << kDegradedThresh << '\n';
        }
    }
    if (fa_artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/failure_analysis.csv (%d configurations)\n", static_cast<int>(fail_stats.size()));
    else
        std::printf("ARTIFACT: FAILED to write demo/out/failure_analysis.csv\n");

    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_umax));
    CUDA_CHECK(cudaFree(d_u));

    // ======================= 6) final verdict =================================
    const bool success = artifact_ok && fa_artifact_ok;
    if (success)
        std::printf("RESULT: PASS (GPU/CPU agree; pseudoinverse, KKT, and monotone-descent gates all hold; both artifacts written)\n");
    else
        std::printf("RESULT: FAIL (one or more artifacts failed to write — see ARTIFACT lines above)\n");
    return success ? 0 : 1;
}
