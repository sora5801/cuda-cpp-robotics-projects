// ===========================================================================
// main.cu — entry point for project 27.04
//           Composite layup optimization + Tsai-Wu failure envelope sweeps
//
// What this program does, start to finish
// ---------------------------------------
//   1. Load the committed scenario (lamina material + Tsai-Wu strengths,
//      the 4-angle stacking alphabet, and the MIXED/ALIGNED load-case sets)
//      from data/sample/.
//   2. SWEEP STAGE (the GPU content, part 1): launch one kernel per case
//      set — kNLayups=256 layups x n_cases load cases each, one GPU thread
//      per (layup, case) pair — scoring every stack sequence's first-ply-
//      failure load factor.
//   3. Reduce (host) each layup's WORST case across its set -> rank layups
//      -> report the best (and how many exactly-tied stack SEQUENCES share
//      that score — an honest CLT finding, not a bug; kernels.cuh's file
//      header explains why).
//   4. ENVELOPE STAGE (the GPU content, part 2): launch the envelope kernel
//      twice — once for the MIXED-set winner, once for the documented
//      [0/90/0/90]s cross-ply baseline — each a 128x128 (Nx,Ny) grid.
//   5. VERIFY STAGE (the §5 GPU-vs-CPU gate): recompute EVERY sweep and
//      envelope point from scratch on the CPU oracle and require full-array
//      agreement within a documented tolerance.
//   6. FOUR ANALYTIC GATES, each checked against a closed-form or physical
//      prediction (reference_cpu.cpp's shared physics, called directly):
//      (i) a single 0-degree ply under pure Nx/-Nx must fail at EXACTLY
//      Xt*t / Xc*t; (ii) an isotropic-degenerate material must produce a
//      direction-independent envelope; (iii) CLT sanity — the cross-ply
//      laminate's A11=A22 and Qbar(0)=Q exactly; (iv) the failure factor
//      must scale exactly inversely with load magnitude (homogeneity).
//   7. Write five artifacts: the layup ranking (CSV), and for each of the
//      two envelopes a PGM heatmap plus a CSV point cloud of the
//      factor=1 ("unit-factor") contour — the classic Tsai-Wu envelope
//      boundary in load space.
//   8. Exit 0 only if VERIFY + all four GATE_* checks + every artifact
//      write succeed.
//
// Determinism: every computation in this program is a PURE function of its
// inputs — no RNG anywhere (CLAUDE.md §8: "prefer deterministic no-noise").
// The GPU and CPU paths share their entire physics (kernels.cuh's HD inline
// functions), so the only possible GPU-vs-CPU divergence is sinf/cosf/
// sqrtf's independently-rounded host vs. device implementations — a single
// PASS through a short function chain (no chained-timestep drift like
// 08.01/18.01), so the tolerance this project needs is far tighter than
// theirs (THEORY.md §numerical considerations measures and explains it).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:", "VERIFY:",
// "GATE_*:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" lines are NOT diffed
// (machine-specific numbers). Change a stable line => update
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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09/08.01/18.01)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Verification tolerances. Every one is a RELATIVE tolerance (floored
// against 1.0 where the compared quantity could legitimately be near zero)
// unless documented otherwise. Values below carry the margin measured on
// the reference machine (RTX 2080 SUPER, CUDA 13.3, Release|x64) — see each
// gate's [info] line for the actual measured number and THEORY.md
// §numerical considerations for why each bound is sized the way it is.
// ---------------------------------------------------------------------------
static const float kVerifyRelTol      = 1.0e-3f;  // GPU vs CPU: sweep + envelope, full-array
static const float kGateSinglePlyTol  = 1.0e-3f;  // gate (i): single-ply closed form
static const float kGateIsoTol        = 2.0e-3f;  // gate (ii): isotropic envelope direction spread
static const float kGateCltSanityTol  = 1.0e-5f;  // gate (iii): CLT symmetry identities
static const float kGateHomogTol      = 1.0e-3f;  // gate (iv): load-scaling homogeneity
static const float kTieRelTol         = 1.0e-4f;  // "exactly tied" layup-score reporting threshold

// The documented [0/90/0/90]s cross-ply baseline — the classic aerospace-
// skin reference laminate this project compares its search winner against.
// Hardcoded (not decoded from the alphabet) so it never depends on alphabet
// ORDERING — it names a laminate, not an index into a data file.
static const float kCrossAngles[kNPlies] = { 0.0f, 90.0f, 0.0f, 90.0f, 90.0f, 0.0f, 90.0f, 0.0f };

// ---------------------------------------------------------------------------
// Scenario — the committed "task definition" (CLAUDE.md §8): the lamina
// material, its Tsai-Wu strengths, the envelope grid's half-span, the
// 4-angle stacking alphabet, and the two load-case sets (MIXED, ALIGNED).
// Every field is REQUIRED by the strict loader below.
// ---------------------------------------------------------------------------
struct Scenario {
    Lamina mat{};
    float n_env_max_npm = 0.0f;
    AngleAlphabet alpha{};
    std::vector<LoadCase> mixed;
    std::vector<LoadCase> aligned;
    bool loaded = false;
};

static bool parse_float(const std::string& s, float& out)
{
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtof(s.c_str(), &end);
    return end != s.c_str();
}

// ---------------------------------------------------------------------------
// load_scenario — strict CSV loader (CLAUDE.md §12 discipline, matching
// 08.01/18.01's load_scenario): every scalar field is required; LOAD_MIXED/
// LOAD_ALIGNED rows are read in FILE ORDER and must exactly match the
// declared N_MIXED_CASES/N_ALIGNED_CASES counts. An unknown label, a short
// row, or a count mismatch aborts the demo rather than silently running a
// different experiment than the PROBLEM:/SCENARIO: lines claim.
// ---------------------------------------------------------------------------
static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_e1=false, have_e2=false, have_g12=false, have_nu12=false, have_t=false;
    bool have_xt=false, have_xc=false, have_yt=false, have_yc=false, have_s12=false;
    bool have_envmax=false, have_alpha=false, have_nmix=false, have_nalign=false;
    int n_mixed_declared = -1, n_aligned_declared = -1;

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // tolerate CRLF
        if (line.empty() || line[0] == '#') continue;

        std::vector<std::string> tok;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) tok.push_back(cell);
        if (tok.empty()) continue;
        const std::string& label = tok[0];

        auto need = [&](size_t n) { return tok.size() >= n; };
        float v0=0, v1=0, v2=0, v3=0;

        if (label == "E1_GPA")        { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.E1_pa  = v0 * 1.0e9f; have_e1=true; }
        else if (label == "E2_GPA")   { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.E2_pa  = v0 * 1.0e9f; have_e2=true; }
        else if (label == "G12_GPA")  { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.G12_pa = v0 * 1.0e9f; have_g12=true; }
        else if (label == "NU12")     { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.nu12   = v0; have_nu12=true; }
        else if (label == "T_PLY_MM") { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.t_ply_m = v0 * 1.0e-3f; have_t=true; }
        else if (label == "XT_MPA")   { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.Xt_pa  = v0 * 1.0e6f; have_xt=true; }
        else if (label == "XC_MPA")   { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.Xc_pa  = v0 * 1.0e6f; have_xc=true; }
        else if (label == "YT_MPA")   { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.Yt_pa  = v0 * 1.0e6f; have_yt=true; }
        else if (label == "YC_MPA")   { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.Yc_pa  = v0 * 1.0e6f; have_yc=true; }
        else if (label == "S12_MPA")  { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.mat.S12_pa = v0 * 1.0e6f; have_s12=true; }
        else if (label == "N_ENV_MAX_NM") { if (!need(2) || !parse_float(tok[1], v0)) goto bad; sc.n_env_max_npm = v0; have_envmax=true; }
        else if (label == "ANGLE_ALPHABET_DEG") {
            if (!need(5) || !parse_float(tok[1],v0) || !parse_float(tok[2],v1) ||
                !parse_float(tok[3],v2) || !parse_float(tok[4],v3)) goto bad;
            sc.alpha.deg[0]=v0; sc.alpha.deg[1]=v1; sc.alpha.deg[2]=v2; sc.alpha.deg[3]=v3;
            have_alpha = true;
        }
        else if (label == "N_MIXED_CASES") {
            if (!need(2)) goto bad;
            n_mixed_declared = std::atoi(tok[1].c_str());
            if (n_mixed_declared < 1) goto bad;
            sc.mixed.reserve(static_cast<size_t>(n_mixed_declared));
            have_nmix = true;
        }
        else if (label == "LOAD_MIXED") {
            if (!need(4) || !parse_float(tok[1],v0) || !parse_float(tok[2],v1) || !parse_float(tok[3],v2)) goto bad;
            sc.mixed.push_back(LoadCase{ v0, v1, v2 });
        }
        else if (label == "N_ALIGNED_CASES") {
            if (!need(2)) goto bad;
            n_aligned_declared = std::atoi(tok[1].c_str());
            if (n_aligned_declared < 1) goto bad;
            sc.aligned.reserve(static_cast<size_t>(n_aligned_declared));
            have_nalign = true;
        }
        else if (label == "LOAD_ALIGNED") {
            if (!need(4) || !parse_float(tok[1],v0) || !parse_float(tok[2],v1) || !parse_float(tok[3],v2)) goto bad;
            sc.aligned.push_back(LoadCase{ v0, v1, v2 });
        }
        else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
        continue;
    bad:
        std::fprintf(stderr, "scenario: malformed row for label '%s'\n", label.c_str());
        return Scenario{};
    }

    if (!(have_e1 && have_e2 && have_g12 && have_nu12 && have_t && have_xt && have_xc &&
          have_yt && have_yc && have_s12 && have_envmax && have_alpha && have_nmix && have_nalign)) {
        std::fprintf(stderr, "scenario: missing one or more required scalar fields\n");
        return Scenario{};
    }
    if (static_cast<int>(sc.mixed.size()) != n_mixed_declared) {
        std::fprintf(stderr, "scenario: N_MIXED_CASES=%d but found %zu LOAD_MIXED rows\n",
                     n_mixed_declared, sc.mixed.size());
        return Scenario{};
    }
    if (static_cast<int>(sc.aligned.size()) != n_aligned_declared) {
        std::fprintf(stderr, "scenario: N_ALIGNED_CASES=%d but found %zu LOAD_ALIGNED rows\n",
                     n_aligned_declared, sc.aligned.size());
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/laminate_scenario.csv");
    candidates.push_back("data/sample/laminate_scenario.csv");
    candidates.push_back("../data/sample/laminate_scenario.csv");
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
// format_angles — "0/45/-45/90/90/-45/45/0"-style label for a decoded
// 8-ply stack — used in [info] lines and the layup_ranking.csv artifact.
// Whole-degree formatting is safe here: the alphabet is always whole
// degrees (0/45/-45/90), never fractional.
// ---------------------------------------------------------------------------
static std::string format_angles(const float angles_deg[kNPlies])
{
    std::string s;
    for (int k = 0; k < kNPlies; ++k) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.0f", static_cast<double>(angles_deg[k]));
        s += buf;
        if (k + 1 < kNPlies) s += '/';
    }
    return s;
}

// ---------------------------------------------------------------------------
// write_pgm_p2 — minimal ASCII (P2) PGM grayscale writer (the 18.01 pattern:
// no libraries, no binary-endianness questions, openable by any viewer).
// A FIXED [0, kEnvFactorClamp] scale (not per-image min/max) is used so the
// best-layup and cross-ply PGMs are DIRECTLY brightness-comparable — the
// caller passes the pre-clamped field, this function only linearly maps it.
// ---------------------------------------------------------------------------
static bool write_pgm_p2(const std::string& path, const std::vector<float>& values,
                         const std::string& label, float n_max_npm)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    const float vmin = 0.0f, vmax = kEnvFactorClamp;
    const int gray_at_one = static_cast<int>(((1.0f - vmin) / (vmax - vmin)) * 255.0f + 0.5f);
    f << "P2\n";
    f << "# 27.04 Tsai-Wu failure envelope: " << label << "\n";
    f << "# Nx in [-" << n_max_npm << ", " << n_max_npm << "] N/m (columns, left->right), "
      << "Ny in [-" << n_max_npm << ", " << n_max_npm << "] N/m (rows, top->bottom)\n";
    f << "# gray = clamp(failure_load_factor, 0, " << kEnvFactorClamp << ") linearly mapped to [0,255]; "
      << "brighter = more load margin. The Tsai-Wu envelope boundary (factor=1) sits at gray="
      << gray_at_one << " (see the _contour.csv sibling file for its exact point cloud).\n";
    f << kEnvGridN << ' ' << kEnvGridN << '\n';
    f << 255 << '\n';
    for (int i = 0; i < kEnvGridN; ++i) {
        for (int j = 0; j < kEnvGridN; ++j) {
            const float v = values[static_cast<size_t>(i) * kEnvGridN + j];
            int gray = static_cast<int>(((v - vmin) / (vmax - vmin)) * 255.0f + 0.5f);
            gray = std::max(0, std::min(255, gray));
            f << gray << (j + 1 < kEnvGridN ? ' ' : '\n');
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// extract_unit_contour — a lightweight (edge-crossing, not full marching-
// squares-with-connectivity) extraction of the factor=1 level set: for
// every horizontal and vertical grid EDGE whose two endpoints straddle
// 1.0, linearly interpolate the crossing point's (Nx,Ny) and emit it. The
// result is an unordered point CLOUD tracing the classic Tsai-Wu failure
// envelope boundary in load space — sufficient for plotting/inspection
// without the added bookkeeping of stitching cells into connected
// polylines (an honest scope choice, documented here and in README
// §Limitations).
// ---------------------------------------------------------------------------
static void extract_unit_contour(const std::vector<float>& field, float n_max_npm,
                                 std::vector<std::pair<float, float>>& pts)
{
    auto val = [&](int i, int j) { return field[static_cast<size_t>(i) * kEnvGridN + j] - 1.0f; };

    for (int i = 0; i < kEnvGridN; ++i) {
        for (int j = 0; j < kEnvGridN; ++j) {
            const float v = val(i, j);

            if (j + 1 < kEnvGridN) {                       // horizontal edge (i,j)-(i,j+1)
                const float v2 = val(i, j + 1);
                if ((v < 0.0f) != (v2 < 0.0f)) {
                    const float t = v / (v - v2);           // fraction along the edge where value crosses 0
                    LoadCase a, b;
                    envelope_grid_point(i, j, n_max_npm, a);
                    envelope_grid_point(i, j + 1, n_max_npm, b);
                    pts.emplace_back(a.Nx_npm + t * (b.Nx_npm - a.Nx_npm), a.Ny_npm);
                }
            }
            if (i + 1 < kEnvGridN) {                       // vertical edge (i,j)-(i+1,j)
                const float v2 = val(i + 1, j);
                if ((v < 0.0f) != (v2 < 0.0f)) {
                    const float t = v / (v - v2);
                    LoadCase a, b;
                    envelope_grid_point(i, j, n_max_npm, a);
                    envelope_grid_point(i + 1, j, n_max_npm, b);
                    pts.emplace_back(a.Nx_npm, a.Ny_npm + t * (b.Ny_npm - a.Ny_npm));
                }
            }
        }
    }
}

static bool write_contour_csv(const std::string& path, const std::vector<std::pair<float,float>>& pts)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "Nx_Npm,Ny_Npm\n";
    for (const auto& p : pts) f << p.first << ',' << p.second << '\n';
    return true;
}

// ---------------------------------------------------------------------------
// rank_layups — reduce a (kNLayups x n_cases) factor array into a per-layup
// WORST-CASE score (the design metric: "how much margin does this layup
// have against the WORST load in this set"), then return layup indices
// sorted by score, BEST first. This host-side O(kNLayups*n_cases) reduction
// stays on the host deliberately (trivial arithmetic — the same "keep the
// small blend in plain C++" call 08.01 makes for its softmin).
// ---------------------------------------------------------------------------
static void rank_layups(const std::vector<float>& factor, int n_cases,
                        std::vector<float>& score_out, std::vector<int>& order_out)
{
    score_out.assign(kNLayups, 0.0f);
    for (int layup = 0; layup < kNLayups; ++layup) {
        float worst = factor[static_cast<size_t>(layup) * n_cases];
        for (int c = 1; c < n_cases; ++c)
            worst = std::min(worst, factor[static_cast<size_t>(layup) * n_cases + c]);
        score_out[static_cast<size_t>(layup)] = worst;
    }
    order_out.resize(kNLayups);
    for (int i = 0; i < kNLayups; ++i) order_out[static_cast<size_t>(i)] = i;
    std::sort(order_out.begin(), order_out.end(),
             [&](int a, int b) { return score_out[static_cast<size_t>(a)] > score_out[static_cast<size_t>(b)]; });
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data laminate_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Composite layup optimization + Tsai-Wu failure envelope sweeps (project 27.04)\n");
    print_device_info();

    // ---- scenario -------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/laminate_scenario.csv missing (run scripts/make_synthetic.py?)\n");
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
    const int n_mixed = static_cast<int>(sc.mixed.size());
    const int n_aligned = static_cast<int>(sc.aligned.size());

    std::printf("PROBLEM: CLT + Tsai-Wu first-ply-failure sweep, %d symmetric 8-ply layups "
               "(alphabet {%.0f,%.0f,%.0f,%.0f} deg, %d^%d), MIXED=%d + ALIGNED=%d load cases, "
               "envelope %dx%d, FP32\n",
               kNLayups, static_cast<double>(sc.alpha.deg[0]), static_cast<double>(sc.alpha.deg[1]),
               static_cast<double>(sc.alpha.deg[2]), static_cast<double>(sc.alpha.deg[3]),
               kNAngleAlphabet, kNIndepPlies, n_mixed, n_aligned, kEnvGridN, kEnvGridN);
    std::printf("SCENARIO: E1=%.1f E2=%.1f G12=%.1f GPa nu12=%.2f, ply %.3f mm x %d = %.3f mm, "
               "Xt=%.0f Xc=%.0f Yt=%.0f Yc=%.0f S12=%.0f MPa [synthetic]\n",
               static_cast<double>(sc.mat.E1_pa / 1.0e9f), static_cast<double>(sc.mat.E2_pa / 1.0e9f),
               static_cast<double>(sc.mat.G12_pa / 1.0e9f), static_cast<double>(sc.mat.nu12),
               static_cast<double>(sc.mat.t_ply_m * 1.0e3f), kNPlies,
               static_cast<double>(sc.mat.t_ply_m * 1.0e3f * kNPlies),
               static_cast<double>(sc.mat.Xt_pa / 1.0e6f), static_cast<double>(sc.mat.Xc_pa / 1.0e6f),
               static_cast<double>(sc.mat.Yt_pa / 1.0e6f), static_cast<double>(sc.mat.Yc_pa / 1.0e6f),
               static_cast<double>(sc.mat.S12_pa / 1.0e6f));

    bool overall_pass = true;

    // ======================= STAGE 1: LAYUP SWEEP (GPU) =====================
    std::vector<float> h_factor_mixed(static_cast<size_t>(kNLayups) * n_mixed);
    std::vector<float> h_factor_aligned(static_cast<size_t>(kNLayups) * n_aligned);
    double sweep_gpu_ms = 0.0;
    {
        LoadCase *d_mixed = nullptr, *d_aligned = nullptr;
        float *d_factor_mixed = nullptr, *d_factor_aligned = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mixed, sizeof(LoadCase) * n_mixed));
        CUDA_CHECK(cudaMalloc(&d_aligned, sizeof(LoadCase) * n_aligned));
        CUDA_CHECK(cudaMalloc(&d_factor_mixed, sizeof(float) * kNLayups * n_mixed));
        CUDA_CHECK(cudaMalloc(&d_factor_aligned, sizeof(float) * kNLayups * n_aligned));
        CUDA_CHECK(cudaMemcpy(d_mixed, sc.mixed.data(), sizeof(LoadCase) * n_mixed, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_aligned, sc.aligned.data(), sizeof(LoadCase) * n_aligned, cudaMemcpyHostToDevice));

        GpuTimer gt;
        gt.begin();
        launch_layup_sweep(sc.mat, sc.alpha, d_mixed, n_mixed, d_factor_mixed);
        launch_layup_sweep(sc.mat, sc.alpha, d_aligned, n_aligned, d_factor_aligned);
        sweep_gpu_ms = static_cast<double>(gt.end_ms());

        CUDA_CHECK(cudaMemcpy(h_factor_mixed.data(), d_factor_mixed, sizeof(float) * kNLayups * n_mixed, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_factor_aligned.data(), d_factor_aligned, sizeof(float) * kNLayups * n_aligned, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_mixed)); CUDA_CHECK(cudaFree(d_aligned));
        CUDA_CHECK(cudaFree(d_factor_mixed)); CUDA_CHECK(cudaFree(d_factor_aligned));
    }

    // Rank both sets (host reduction — see rank_layups' comment).
    std::vector<float> score_mixed, score_aligned;
    std::vector<int> order_mixed, order_aligned;
    rank_layups(h_factor_mixed, n_mixed, score_mixed, order_mixed);
    rank_layups(h_factor_aligned, n_aligned, score_aligned, order_aligned);

    const int best_mixed_id = order_mixed[0];
    const int best_aligned_id = order_aligned[0];
    const float best_mixed_score = score_mixed[static_cast<size_t>(best_mixed_id)];
    const float best_aligned_score = score_aligned[static_cast<size_t>(best_aligned_id)];

    float best_mixed_angles[kNPlies], best_aligned_angles[kNPlies];
    decode_layup(best_mixed_id, sc.alpha.deg, best_mixed_angles);
    decode_layup(best_aligned_id, sc.alpha.deg, best_aligned_angles);

    int mixed_ties = 0, aligned_ties = 0;
    for (float s : score_mixed) if (std::fabs(s - best_mixed_score) <= kTieRelTol * std::fabs(best_mixed_score)) ++mixed_ties;
    for (float s : score_aligned) if (std::fabs(s - best_aligned_score) <= kTieRelTol * std::fabs(best_aligned_score)) ++aligned_ties;

    std::printf("[time] layup sweep: GPU kernel %.3f ms for %d (layup x case) evaluations\n",
               sweep_gpu_ms, kNLayups * (n_mixed + n_aligned));
    std::printf("[info] MIXED winner: layup_id=%d [%s], worst-case factor=%.4f "
               "(%d/%d layups tied within %.0e rel — membrane-only ties are a CLT identity, THEORY.md explains)\n",
               best_mixed_id, format_angles(best_mixed_angles).c_str(), static_cast<double>(best_mixed_score),
               mixed_ties, kNLayups, static_cast<double>(kTieRelTol));
    std::printf("[info] ALIGNED winner: layup_id=%d [%s], worst-case factor=%.4f (%d/%d layups tied within %.0e rel)\n",
               best_aligned_id, format_angles(best_aligned_angles).c_str(), static_cast<double>(best_aligned_score),
               aligned_ties, kNLayups, static_cast<double>(kTieRelTol));

    // ======================= STAGE 2: ENVELOPE (GPU) =========================
    std::vector<float> h_env_best(static_cast<size_t>(kEnvGridN) * kEnvGridN);
    std::vector<float> h_env_cross(static_cast<size_t>(kEnvGridN) * kEnvGridN);
    double env_gpu_ms = 0.0;
    {
        Layup8 best_layup{}, cross_layup{};
        for (int k = 0; k < kNPlies; ++k) { best_layup.deg[k] = best_mixed_angles[k]; cross_layup.deg[k] = kCrossAngles[k]; }

        float *d_env_best = nullptr, *d_env_cross = nullptr;
        const size_t env_bytes = sizeof(float) * kEnvGridN * kEnvGridN;
        CUDA_CHECK(cudaMalloc(&d_env_best, env_bytes));
        CUDA_CHECK(cudaMalloc(&d_env_cross, env_bytes));

        GpuTimer gt;
        gt.begin();
        launch_envelope(sc.mat, best_layup, sc.n_env_max_npm, d_env_best);
        launch_envelope(sc.mat, cross_layup, sc.n_env_max_npm, d_env_cross);
        env_gpu_ms = static_cast<double>(gt.end_ms());

        CUDA_CHECK(cudaMemcpy(h_env_best.data(), d_env_best, env_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_env_cross.data(), d_env_cross, env_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_env_best)); CUDA_CHECK(cudaFree(d_env_cross));
    }
    std::printf("[time] envelope sweep: GPU kernel %.3f ms for 2 x %dx%d grids\n",
               env_gpu_ms, kEnvGridN, kEnvGridN);

    // ======================= STAGE 3: VERIFY (§5 gate) ========================
    // Full-array CPU recomputation of every GPU result produced above — this
    // project's problem sizes are small enough (<= kEnvGridN^2 points) that a
    // full oracle, not a stride-sampled spot check, is cheap (THEORY.md
    // §how-we-verify justifies the tolerance below).
    {
        std::vector<float> cpu_mixed(h_factor_mixed.size()), cpu_aligned(h_factor_aligned.size());
        std::vector<float> cpu_env_best(h_env_best.size()), cpu_env_cross(h_env_cross.size());

        Layup8 best_layup{}, cross_layup{};
        for (int k = 0; k < kNPlies; ++k) { best_layup.deg[k] = best_mixed_angles[k]; cross_layup.deg[k] = kCrossAngles[k]; }

        CpuTimer ct;
        ct.begin();
        layup_sweep_cpu(sc.mat, sc.alpha, sc.mixed.data(), n_mixed, cpu_mixed.data());
        layup_sweep_cpu(sc.mat, sc.alpha, sc.aligned.data(), n_aligned, cpu_aligned.data());
        envelope_cpu(sc.mat, best_layup, sc.n_env_max_npm, cpu_env_best.data());
        envelope_cpu(sc.mat, cross_layup, sc.n_env_max_npm, cpu_env_cross.data());
        const double cpu_ms = ct.end_ms();

        auto worst_rel = [](const std::vector<float>& a, const std::vector<float>& b) {
            float worst = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                const float scale = std::fabs(b[i]) > 1.0f ? std::fabs(b[i]) : 1.0f;
                worst = std::max(worst, std::fabs(a[i] - b[i]) / scale);
            }
            return worst;
        };
        const float w_mixed = worst_rel(h_factor_mixed, cpu_mixed);
        const float w_aligned = worst_rel(h_factor_aligned, cpu_aligned);
        const float w_env_best = worst_rel(h_env_best, cpu_env_best);
        const float w_env_cross = worst_rel(h_env_cross, cpu_env_cross);
        const float worst = std::max(std::max(w_mixed, w_aligned), std::max(w_env_best, w_env_cross));

        std::printf("[time] verify: CPU oracle %.2f ms for %zu points (sweep + envelope, full recompute)\n",
                   cpu_ms, h_factor_mixed.size() + h_factor_aligned.size() + h_env_best.size() + h_env_cross.size());
        std::printf("[info] verify: worst relative deviation — mixed sweep %.3e, aligned sweep %.3e, "
                   "best envelope %.3e, cross envelope %.3e (tol %.1e)\n",
                   static_cast<double>(w_mixed), static_cast<double>(w_aligned),
                   static_cast<double>(w_env_best), static_cast<double>(w_env_cross),
                   static_cast<double>(kVerifyRelTol));

        const bool verify_pass = worst <= kVerifyRelTol;
        std::printf("VERIFY: %s (GPU sweep+envelope match the full CPU oracle within rel tol %.1e)\n",
                   verify_pass ? "PASS" : "FAIL", static_cast<double>(kVerifyRelTol));
        overall_pass = overall_pass && verify_pass;
    }

    // ======================= STAGE 4: ANALYTIC GATES ==========================

    // ---- Gate (i): single 0-degree ply under pure +-Nx: closed-form failure
    // load. F1,F11 are CALIBRATED so sigma1=Xt and sigma1=-Xc are EXACT roots
    // of the 1D Tsai-Wu quadratic (THEORY.md §the-math proves it algebraically)
    // — for a SINGLE ply the laminate-axis stress is Nx/t exactly (no A-matrix
    // solve needed: there is only one material through the thickness), so the
    // failure load is EXACTLY Xt*t (tension) / Xc*t (compression).
    bool gate_single_ply_pass;
    {
        const float angle0[1] = { 0.0f };
        const LoadCase lc_tension  = { 1.0f, 0.0f, 0.0f };
        const LoadCase lc_compress = { -1.0f, 0.0f, 0.0f };
        const float lam_t = laminate_failure_factor(sc.mat, angle0, 1, lc_tension);
        const float lam_c = laminate_failure_factor(sc.mat, angle0, 1, lc_compress);
        const float expect_t = sc.mat.Xt_pa * sc.mat.t_ply_m;
        const float expect_c = sc.mat.Xc_pa * sc.mat.t_ply_m;
        const float rel_t = std::fabs(lam_t - expect_t) / expect_t;
        const float rel_c = std::fabs(lam_c - expect_c) / expect_c;
        gate_single_ply_pass = (rel_t <= kGateSinglePlyTol) && (rel_c <= kGateSinglePlyTol);
        std::printf("[info] gate single-ply: tension lambda=%.1f N/m (expect Xt*t=%.1f, rel %.3e), "
                   "compression lambda=%.1f N/m (expect Xc*t=%.1f, rel %.3e), tol %.1e\n",
                   static_cast<double>(lam_t), static_cast<double>(expect_t), static_cast<double>(rel_t),
                   static_cast<double>(lam_c), static_cast<double>(expect_c), static_cast<double>(rel_c),
                   static_cast<double>(kGateSinglePlyTol));
        std::printf("GATE_SINGLE_PLY_CLOSED_FORM: %s (a single 0-deg ply under pure Nx must fail at exactly "
                   "Xt*t in tension, Xc*t in compression)\n", gate_single_ply_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_single_ply_pass;
    }

    // ---- Gate (ii): isotropic-degenerate material -> direction-independent
    // failure strength. E1=E2=kIsoE_Pa with G12=E/(2*(1+nu)) makes Qbar(theta)
    // ROTATION-INVARIANT (an isotropic elastic tensor); equal strengths in
    // every direction PLUS S12=F0/sqrt(3) (THEORY.md derives this from
    // requiring F66=3*F11) additionally makes the TSAI-WU quadratic form
    // invariant under a coordinate rotation of a FIXED stress state.
    //
    // THE SUBTLE POINT THIS GATE IS CAREFUL ABOUT (THEORY.md §the-math walks
    // it in full): sweeping the LOAD DIRECTION around the (Nx,Ny) circle
    // with Nxy HELD AT ZERO does NOT trace a rotation of one physical stress
    // state — Nx=Ny (phi=45 deg) is EQUAL-BIAXIAL tension, a fundamentally
    // different loading condition from Nx-only (phi=0) UNIAXIAL tension, and
    // Tsai-Wu's own F12 interaction term (needed to keep the envelope closed
    // — kernels.cuh's tsaiwu_F comment) makes even a fully isotropic
    // material's failure envelope in that (Nx,Ny,Nxy=0) SLICE an ellipse,
    // not a circle. The mathematically correct invariance test instead
    // rotates a FIXED uniaxial stress resultant of magnitude S through
    // orientation angle phi via the standard 2D stress-transformation
    // identities themselves:
    //     Nx(phi)  = S*cos^2(phi)         Ny(phi) = S*sin^2(phi)
    //     Nxy(phi) = S*sin(phi)*cos(phi)
    // (equivalently: apply a fixed uniaxial load to a ply rotated by phi —
    // the same relative rotation either way). This traces genuine rotations
    // of one stress tensor, so the predicted result — failure factor
    // constant across phi — is now the honest, correct claim; sweeping
    // instead over 180 deg (not 360) avoids re-testing the same physical
    // stress state twice (a uniaxial stress tensor is pi-periodic in its
    // orientation angle).
    bool gate_iso_pass;
    {
        Lamina iso{};
        iso.E1_pa = iso.E2_pa = kIsoE_Pa;
        iso.G12_pa = kIsoE_Pa / (2.0f * (1.0f + kIsoNu));   // the isotropic relation being tested
        iso.nu12 = kIsoNu;
        iso.t_ply_m = sc.mat.t_ply_m;                        // thickness cancels in the RELATIVE spread check
        iso.Xt_pa = iso.Xc_pa = iso.Yt_pa = iso.Yc_pa = kIsoF0_Pa;
        iso.S12_pa = kIsoF0_Pa / std::sqrt(3.0f);             // THEORY.md derives this from F66=3*F11

        const float angle0[1] = { 0.0f };                    // the ply orientation is fixed; the LOAD rotates
        float lam_min = 1.0e30f, lam_max = 0.0f, lam_sum = 0.0f;
        for (int d = 0; d < kIsoGateDirs; ++d) {
            const float phi = kPi * static_cast<float>(d) / static_cast<float>(kIsoGateDirs);   // [0, pi)
            const float c = std::cos(phi), s = std::sin(phi);
            const LoadCase lc = { c * c, s * s, s * c };      // a rotated unit-magnitude UNIAXIAL stress resultant
            const float lam = laminate_failure_factor(iso, angle0, 1, lc);
            lam_min = std::min(lam_min, lam);
            lam_max = std::max(lam_max, lam);
            lam_sum += lam;
        }
        const float lam_mean = lam_sum / static_cast<float>(kIsoGateDirs);
        const float spread = (lam_max - lam_min) / lam_mean;
        gate_iso_pass = spread <= kGateIsoTol;
        std::printf("[info] gate isotropic: %d rotated-uniaxial-load directions, failure strength "
                   "min=%.4f max=%.4f mean=%.4f N/m, relative spread=%.3e (tol %.1e)\n",
                   kIsoGateDirs, static_cast<double>(lam_min), static_cast<double>(lam_max),
                   static_cast<double>(lam_mean), static_cast<double>(spread), static_cast<double>(kGateIsoTol));
        std::printf("GATE_ISOTROPIC_ENVELOPE: %s (an isotropic-degenerate material's failure strength "
                   "must not depend on the applied uniaxial load's orientation)\n", gate_iso_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_iso_pass;
    }

    // ---- Gate (iii): CLT sanity — the cross-ply laminate's A11=A22 (by
    // symmetry: equal counts of 0-deg and 90-deg plies swap Q11<->Q22
    // between the two sums) and Qbar(theta=0) reproduces Q exactly (c=1,s=0
    // kills every s-bearing term in transform_Qbar — kernels.cuh's comment
    // on that function names this exact check).
    bool gate_clt_pass;
    {
        float A11, A12, A16, A22, A26, A66;
        assemble_A(sc.mat, kCrossAngles, kNPlies, A11, A12, A16, A22, A26, A66);
        const float rel_A = std::fabs(A11 - A22) / A11;

        float Q11, Q12, Q22, Q66;
        lamina_Q(sc.mat, Q11, Q12, Q22, Q66);
        float Qb11, Qb12, Qb16, Qb22, Qb26, Qb66;
        transform_Qbar(Q11, Q12, Q22, Q66, 0.0f, Qb11, Qb12, Qb16, Qb22, Qb26, Qb66);
        const float rel_Q11 = std::fabs(Qb11 - Q11) / Q11;
        const float rel_Q12 = std::fabs(Qb12 - Q12) / Q12;
        const float rel_Q22 = std::fabs(Qb22 - Q22) / Q22;
        const float rel_Q66 = std::fabs(Qb66 - Q66) / Q66;
        const float rel_Q16 = std::fabs(Qb16) / Q11;   // Qb16 should be EXACTLY 0 at theta=0 — measured relative to Q11's scale
        const float rel_Q26 = std::fabs(Qb26) / Q11;
        const float worst_Q = std::max({ rel_Q11, rel_Q12, rel_Q22, rel_Q66, rel_Q16, rel_Q26 });

        gate_clt_pass = (rel_A <= kGateCltSanityTol) && (worst_Q <= kGateCltSanityTol);
        std::printf("[info] gate CLT-sanity: cross-ply A11=%.4e A22=%.4e (rel %.3e); "
                   "Qbar(0) vs Q worst relative deviation %.3e (tol %.1e)\n",
                   static_cast<double>(A11), static_cast<double>(A22), static_cast<double>(rel_A),
                   static_cast<double>(worst_Q), static_cast<double>(kGateCltSanityTol));
        std::printf("GATE_CLT_SANITY: %s ([0/90/0/90]s must give A11=A22 exactly; Qbar(theta=0) must "
                   "reproduce Q exactly)\n", gate_clt_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_clt_pass;
    }

    // ---- Gate (iv): load-scaling homogeneity. Stress is LINEAR in applied
    // load (Hooke's law all the way through CLT), so the failure load FACTOR
    // for a load k*N must equal (1/k) times the factor for N — checked on a
    // handful of (layup, case, k) samples spanning the sweep's actual data.
    bool gate_homog_pass;
    {
        struct Sample { const float* angles; LoadCase base; float k; };
        const Sample samples[] = {
            { best_mixed_angles,   sc.mixed[0],           2.0f },
            { best_mixed_angles,   sc.mixed[0],           0.5f },
            { best_aligned_angles, sc.aligned[0],         3.0f },
            { kCrossAngles,        sc.mixed[static_cast<size_t>(n_mixed / 2)], 4.0f },
        };
        float worst_rel = 0.0f;
        for (const Sample& s : samples) {
            const float lam_base = laminate_failure_factor(sc.mat, s.angles, kNPlies, s.base);
            const LoadCase scaled = { s.base.Nx_npm * s.k, s.base.Ny_npm * s.k, s.base.Nxy_npm * s.k };
            const float lam_scaled = laminate_failure_factor(sc.mat, s.angles, kNPlies, scaled);
            const float expect = lam_base / s.k;
            const float rel = std::fabs(lam_scaled - expect) / expect;
            worst_rel = std::max(worst_rel, rel);
        }
        gate_homog_pass = worst_rel <= kGateHomogTol;
        std::printf("[info] gate homogeneity: %zu (layup,case,k) samples, worst relative deviation from "
                   "factor(k*N) = factor(N)/k is %.3e (tol %.1e)\n",
                   sizeof(samples) / sizeof(samples[0]), static_cast<double>(worst_rel),
                   static_cast<double>(kGateHomogTol));
        std::printf("GATE_LOAD_HOMOGENEITY: %s (the failure load factor must scale exactly inversely "
                   "with load magnitude)\n", gate_homog_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_homog_pass;
    }

    // ======================= ARTIFACTS =========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    const bool dir_ok = ensure_dir(out_dir);

    // ---- layup_ranking.csv: top 10 layups per case set ------------------------
    bool rank_ok = dir_ok;
    if (dir_ok) {
        std::ofstream f(out_dir + "/layup_ranking.csv");
        rank_ok = f.is_open();
        if (rank_ok) {
            f << "rank,case_set,layup_id,angles_deg,score_factor\n";
            const int top_n = 10;
            for (int r = 0; r < top_n && r < kNLayups; ++r) {
                const int lid = order_mixed[static_cast<size_t>(r)];
                float ang[kNPlies]; decode_layup(lid, sc.alpha.deg, ang);
                f << (r + 1) << ",MIXED," << lid << ',' << format_angles(ang) << ','
                  << score_mixed[static_cast<size_t>(lid)] << '\n';
            }
            for (int r = 0; r < top_n && r < kNLayups; ++r) {
                const int lid = order_aligned[static_cast<size_t>(r)];
                float ang[kNPlies]; decode_layup(lid, sc.alpha.deg, ang);
                f << (r + 1) << ",ALIGNED," << lid << ',' << format_angles(ang) << ','
                  << score_aligned[static_cast<size_t>(lid)] << '\n';
            }
        }
    }
    std::printf(rank_ok ? "ARTIFACT: wrote demo/out/layup_ranking.csv (20 rows)\n"
                        : "ARTIFACT: FAILED to write demo/out/layup_ranking.csv\n");

    // ---- envelope PGMs + unit-factor contour CSVs ------------------------------
    const bool pgm_best_ok = dir_ok && write_pgm_p2(out_dir + "/envelope_best.pgm", h_env_best,
                                                    "MIXED-set winner layup_id=" + std::to_string(best_mixed_id),
                                                    sc.n_env_max_npm);
    std::printf(pgm_best_ok ? "ARTIFACT: wrote demo/out/envelope_best.pgm (%dx%d)\n"
                            : "ARTIFACT: FAILED to write demo/out/envelope_best.pgm\n", kEnvGridN, kEnvGridN);

    const bool pgm_cross_ok = dir_ok && write_pgm_p2(out_dir + "/envelope_cross.pgm", h_env_cross,
                                                     "[0/90/0/90]s cross-ply baseline", sc.n_env_max_npm);
    std::printf(pgm_cross_ok ? "ARTIFACT: wrote demo/out/envelope_cross.pgm (%dx%d)\n"
                             : "ARTIFACT: FAILED to write demo/out/envelope_cross.pgm\n", kEnvGridN, kEnvGridN);

    std::vector<std::pair<float,float>> contour_best, contour_cross;
    extract_unit_contour(h_env_best, sc.n_env_max_npm, contour_best);
    extract_unit_contour(h_env_cross, sc.n_env_max_npm, contour_cross);
    const bool contour_best_ok = dir_ok && write_contour_csv(out_dir + "/envelope_best_contour.csv", contour_best);
    const bool contour_cross_ok = dir_ok && write_contour_csv(out_dir + "/envelope_cross_contour.csv", contour_cross);
    // The exact POINT COUNT depends on which grid edges straddle factor=1 —
    // a computed result that could shift by a handful of points on a
    // different GPU architecture's independently-rounded sinf/cosf/sqrtf
    // (THEORY.md §numerics). Unlike the PGM/ranking artifact lines above
    // (whose counts are fixed GRID SHAPE, never data-dependent), the count
    // therefore stays on an unchecked "[info]" line — only the checked
    // ARTIFACT line (file written or not) is a stable diff target.
    std::printf("[info] contour point counts: best=%zu, cross=%zu (edge-crossing extraction; see extract_unit_contour)\n",
               contour_best.size(), contour_cross.size());
    std::printf(contour_best_ok ? "ARTIFACT: wrote demo/out/envelope_best_contour.csv\n"
                                : "ARTIFACT: FAILED to write demo/out/envelope_best_contour.csv\n");
    std::printf(contour_cross_ok ? "ARTIFACT: wrote demo/out/envelope_cross_contour.csv\n"
                                 : "ARTIFACT: FAILED to write demo/out/envelope_cross_contour.csv\n");

    overall_pass = overall_pass && rank_ok && pgm_best_ok && pgm_cross_ok && contour_best_ok && contour_cross_ok;

    // ======================= VERDICT ===========================================
    if (overall_pass)
        std::printf("RESULT: PASS (VERIFY + GATE_SINGLE_PLY_CLOSED_FORM + GATE_ISOTROPIC_ENVELOPE + "
                   "GATE_CLT_SANITY + GATE_LOAD_HOMOGENEITY all passed)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE_*/ARTIFACT lines above)\n");
    return overall_pass ? 0 : 1;
}
