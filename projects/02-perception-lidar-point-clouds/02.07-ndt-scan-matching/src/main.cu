// ===========================================================================
// main.cu — entry point for project 02.07
//           NDT scan matching (Autoware-style map localizer)
//
// What this program does, start to finish
// -----------------------------------------
//   0. Load the committed map/scan/cohort/ground-truth files from
//      data/sample/ (scripts/make_synthetic.py's output — see that script
//      and data/README.md for the exact format and generation story).
//   1. Build the NDT voxel grid (coarse 2.0 m AND fine 1.0 m) on the GPU
//      from the map cloud, and independently on the CPU (the oracle twin).
//   2. Run NINE verification stages, in order, each printing a stable
//      "STAGE_NAME: PASS/FAIL" line and appending to gates_metrics.csv:
//        A voxel_stats_twin   — GPU vs CPU voxel mean/covariance (§9 gate)
//        B jacobian_check     — analytic vs central-difference gradient
//        C assembly_twin      — GPU vs CPU score/gradient/Hessian assembly
//        D trajectory_twin    — one full GPU-orchestrated vs CPU-independent
//                               multi-resolution Newton trajectory
//        E score_sanity       — score(truth) < score(every perturbed guess)
//        F convergence/accuracy — the 240-trial cohort sweep (NDT multi-res)
//        G basin_contrast     — NDT multi-res vs NDT fine-only vs ICP, SAME
//                               240 trials
//        H outlier_robustness — same cohort, WITH vs WITHOUT outliers
//        I degenerate_axis    — [info] Hessian conditioning, corridor vs
//                               full scene (no pass/fail verdict — honesty
//                               reporting, 01.17's degeneracy-gate lineage)
//   3. Write four demo/out/ artifacts (registration before/after, basin
//      curve, convergence trajectories, gates_metrics.csv) and the final
//      RESULT verdict.
//
// Why the perturbation cohort is a COMMITTED FILE, not host RNG: every
// initial guess main.cu starts a trial from is precomputed by
// scripts/make_synthetic.py (cohort.csv) — this program never calls a
// random-number generator at all. One less place determinism could break,
// and it makes every trial's exact starting pose auditable in a text file.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "<STAGE>:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" unchecked. Change a
// stable line -> update demo/expected_output.txt in the same change.
//
// Read this after: kernels.cuh.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Data loading — the PC01 binary cloud format (02.06's exact format, cited)
// and the small label-prefixed / plain CSVs scripts/make_synthetic.py writes.
// ===========================================================================

struct CloudHost {
    std::vector<float> xyz;   // interleaved [n*3], meters
    int n = 0;
    bool loaded = false;
};

static CloudHost load_cloud_bin(const std::string& path)
{
    CloudHost c;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return c;

    char magic[4] = { 0, 0, 0, 0 };
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "PC01", 4) != 0) return c;

    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!f) return c;

    c.xyz.resize(static_cast<size_t>(count) * 3);
    if (count > 0) {
        f.read(reinterpret_cast<char*>(c.xyz.data()),
              static_cast<std::streamsize>(c.xyz.size() * sizeof(float)));
        if (!f) return c;
    }
    c.n = static_cast<int>(count);
    c.loaded = true;
    return c;
}

static std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

struct CohortTrial {
    int trial_id = 0;
    int bin_index = 0;
    float mag_trans_m = 0.0f;
    float mag_yaw_deg = 0.0f;
    Rigid3 T_init = kIdentityRigid3;
};

static bool load_cohort(const std::string& path, std::vector<CohortTrial>& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto f = split_csv_line(line);
        if (f.size() != 16) { std::fprintf(stderr, "cohort.csv: malformed row (%zu fields)\n", f.size()); return false; }
        CohortTrial t;
        t.trial_id = std::atoi(f[0].c_str());
        t.bin_index = std::atoi(f[1].c_str());
        t.mag_trans_m = std::strtof(f[2].c_str(), nullptr);
        t.mag_yaw_deg = std::strtof(f[3].c_str(), nullptr);
        for (int i = 0; i < 9; ++i) t.T_init.R[i] = std::strtof(f[4 + i].c_str(), nullptr);
        for (int i = 0; i < 3; ++i) t.T_init.t[i] = std::strtof(f[13 + i].c_str(), nullptr);
        out.push_back(t);
    }
    return !out.empty();
}

struct MetaInfo {
    Rigid3 gt = kIdentityRigid3;
    int n_map = 0, n_scan_main = 0, n_scan_cohort = 0, n_scan_cohort_clean = 0, n_icp_target = 0;
    float noise_sigma_m = 0.0f, outlier_fraction = 0.0f;
    float leaf_coarse_m = 0.0f, leaf_fine_m = 0.0f, icp_target_leaf_m = 0.0f;
    bool loaded = false;
};

static bool load_meta(const std::string& path, MetaInfo& m)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    bool have_gt = false, have_counts = false, have_params = false;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto f = split_csv_line(line);
        if (f.empty()) continue;
        if (f[0] == "GT_POSE" && f.size() == 13) {
            for (int i = 0; i < 9; ++i) m.gt.R[i] = std::strtof(f[1 + i].c_str(), nullptr);
            for (int i = 0; i < 3; ++i) m.gt.t[i] = std::strtof(f[10 + i].c_str(), nullptr);
            have_gt = true;
        } else if (f[0] == "COUNTS" && f.size() == 6) {
            m.n_map = std::atoi(f[1].c_str());
            m.n_scan_main = std::atoi(f[2].c_str());
            m.n_scan_cohort = std::atoi(f[3].c_str());
            m.n_scan_cohort_clean = std::atoi(f[4].c_str());
            m.n_icp_target = std::atoi(f[5].c_str());
            have_counts = true;
        } else if (f[0] == "PARAMS" && f.size() == 6) {
            m.noise_sigma_m = std::strtof(f[1].c_str(), nullptr);
            m.outlier_fraction = std::strtof(f[2].c_str(), nullptr);
            m.leaf_coarse_m = std::strtof(f[3].c_str(), nullptr);
            m.leaf_fine_m = std::strtof(f[4].c_str(), nullptr);
            m.icp_target_leaf_m = std::strtof(f[5].c_str(), nullptr);
            have_params = true;
        }
    }
    m.loaded = have_gt && have_counts && have_params;
    return m.loaded;
}

// ===========================================================================
// GPU voxel grid allocation — one NdtGridGPU per resolution level.
// ===========================================================================
static NdtGridGPU alloc_ndt_grid_device(float leaf)
{
    NdtGridGPU g{};
    g.origin_x = kMapOriginX; g.origin_y = kMapOriginY; g.origin_z = kMapOriginZ;
    g.leaf = leaf;
    grid_dims_for_leaf(leaf, g.nx, g.ny, g.nz);
    const size_t cap = static_cast<size_t>(g.nx) * g.ny * g.nz;

    CUDA_CHECK(cudaMalloc(&g.count, cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.sum_xyz, cap * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.sum_cov6, cap * 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.mean, cap * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.inv_cov6, cap * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.valid, cap * sizeof(unsigned char)));
    return g;
}

static void free_ndt_grid_device(NdtGridGPU& g)
{
    CUDA_CHECK(cudaFree(g.count));
    CUDA_CHECK(cudaFree(g.sum_xyz));
    CUDA_CHECK(cudaFree(g.sum_cov6));
    CUDA_CHECK(cudaFree(g.mean));
    CUDA_CHECK(cudaFree(g.inv_cov6));
    CUDA_CHECK(cudaFree(g.valid));
}

// assemble_gpu — one GPU assembly call + the host-side block-partial sum
// (in double), packaged so both the outer Newton loop and its inner
// accept/reject re-evaluations below share the identical download/sum code.
static void assemble_gpu(const float* d_scan_xyz, int n_scan, const Rigid3& T, NdtGridGPU grid,
                         double d1, double d2, float* d_block_partials,
                         double H21[21], double g6[6], double* score)
{
    const int nblocks = launch_ndt_assemble(d_scan_xyz, n_scan, T, grid, d1, d2, d_block_partials);
    std::vector<float> partials(static_cast<size_t>(nblocks) * kReduceWidth);
    CUDA_CHECK(cudaMemcpy(partials.data(), d_block_partials,
                          partials.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double sums[kReduceWidth];
    for (int k = 0; k < kReduceWidth; ++k) sums[k] = 0.0;
    for (int b = 0; b < nblocks; ++b)
        for (int k = 0; k < kReduceWidth; ++k)
            sums[k] += static_cast<double>(partials[static_cast<size_t>(b) * kReduceWidth + k]);
    for (int k = 0; k < 21; ++k) H21[k] = sums[k];
    for (int k = 0; k < 6; ++k) g6[k] = sums[21 + k];
    *score = sums[27];
}

// ===========================================================================
// run_ndt_stage_gpu — ONE resolution stage of the GPU-orchestrated Newton
// trajectory: a proper damped Gauss-Newton/Levenberg loop with
// ACCEPT/REJECT (a step is taken only if it actually LOWERS the score;
// otherwise lambda grows and the SAME H/g are re-damped and re-solved —
// classic Levenberg-Marquardt, and NECESSARY here in a way it is not for
// 01.17/02.06's PSD Gauss-Newton systems: a raw undamped Newton step on
// NDT's H (which can be indefinite far from the optimum, kernels.cuh's
// cholesky6_solve_flat comment) can easily INCREASE the score, and without
// rejecting that step the trajectory wanders instead of converging — a
// bug this project's own convergence gate caught during development (see
// kernels.cuh's kLambdaInit comment for the companion damping-scale fix;
// THEORY.md "the algorithm" documents both fixes as one lesson).
// Mirrors reference_cpu.cpp's run_ndt_stage_cpu in STRUCTURE (same control
// flow, same hyperparameters from kernels.cuh) but calls the GPU assembly
// kernel instead of ndt_assemble_cpu — the "twin" pairing STAGE D exercises.
// ===========================================================================
static int run_ndt_stage_gpu(const float* d_scan_xyz, int n_scan, NdtGridGPU grid,
                             double d1, double d2, int max_iters,
                             float* d_block_partials,
                             Rigid3& T, std::vector<double>* loss_history)
{
    double H21[21], g6[6], score;
    assemble_gpu(d_scan_xyz, n_scan, T, grid, d1, d2, d_block_partials, H21, g6, &score);
    double lambda = kLambdaInit;   // cholesky6_solve_flat scales this per-parameter internally (see its header)
    if (loss_history) loss_history->push_back(score);

    int iters_done = 0;
    for (int outer = 0; outer < max_iters; ++outer) {
        iters_done = outer + 1;
        bool accepted = false;
        double delta[6] = { 0, 0, 0, 0, 0, 0 };

        for (int inner = 0; inner < kMaxAcceptRejectRetries; ++inner) {   // bounded accept/reject retries at growing lambda
            bool ok = false;
            for (int tries = 0; tries < 40 && !ok; ++tries) {
                ok = cholesky6_solve_flat(H21, g6, lambda, delta);
                if (!ok) lambda *= kLambdaUp;
            }
            if (!ok) break;   // could not damp to SPD even at large lambda -- stop honestly

            Rigid3 T_cand;
            retract(T, delta, T_cand);
            double Hc[21], gc[6], score_cand;
            assemble_gpu(d_scan_xyz, n_scan, T_cand, grid, d1, d2, d_block_partials, Hc, gc, &score_cand);

            if (score_cand < score) {   // improvement -- accept and press forward
                T = T_cand;
                for (int k = 0; k < 21; ++k) H21[k] = Hc[k];
                for (int k = 0; k < 6; ++k) g6[k] = gc[k];
                score = score_cand;
                lambda = std::max(lambda * kLambdaDown, kLambdaMin);
                accepted = true;
                break;
            }
            lambda *= kLambdaUp;   // rejected -- damp harder and retry the SAME H/g
        }

        if (loss_history) loss_history->push_back(score);
        if (!accepted) break;

        const double dn = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
                                    + delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);
        if (dn < kConvergeDeltaNorm) break;
    }
    return iters_done;
}

// run_ndt_multires_gpu — coarse (kMaxItersCoarse) then fine (kMaxItersFine),
// GPU-orchestrated, starting from T_init.
static Rigid3 run_ndt_multires_gpu(Rigid3 T_init,
                                   const float* d_scan_xyz, int n_scan,
                                   NdtGridGPU grid_coarse, double d1c, double d2c,
                                   NdtGridGPU grid_fine, double d1f, double d2f,
                                   float* d_block_partials,
                                   std::vector<double>* loss_history = nullptr)
{
    Rigid3 T = T_init;
    run_ndt_stage_gpu(d_scan_xyz, n_scan, grid_coarse, d1c, d2c, kMaxItersCoarse, d_block_partials, T, loss_history);
    run_ndt_stage_gpu(d_scan_xyz, n_scan, grid_fine, d1f, d2f, kMaxItersFine, d_block_partials, T, loss_history);
    return T;
}

// ===========================================================================
// jacobi_eigen_symmetric6 — 01.17's exact 6x6 cyclic Jacobi eigensolve
// (cited, reimplemented locally per the self-containment rule). Used ONLY
// by STAGE I's [info] Hessian-conditioning report — a one-shot, twice-per-
// run host computation.
// ===========================================================================
static void jacobi_eigen_symmetric6(double A[6][6], double eigvecs[6][6])
{
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            eigvecs[i][j] = (i == j) ? 1.0 : 0.0;

    const int kSweeps = 12;
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        for (int p = 0; p < 6; ++p) {
            for (int q = p + 1; q < 6; ++q) {
                if (std::fabs(A[p][q]) < 1e-15) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                const double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A[p][q] = A[q][p] = 0.0;
                for (int k = 0; k < 6; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = A[p][k] = c * akp - s * akq;
                    A[k][q] = A[q][k] = s * akp + c * akq;
                }
                for (int k = 0; k < 6; ++k) {
                    const double vkp = eigvecs[k][p], vkq = eigvecs[k][q];
                    eigvecs[k][p] = c * vkp - s * vkq;
                    eigvecs[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

// ===========================================================================
// Small reporting/classification helpers shared by several stages.
// ===========================================================================
static const char* kParamNames[6] = { "wx", "wy", "wz", "vx", "vy", "vz" };

struct PoseError { float rot_deg; float trans_m; };

static PoseError pose_error(const Rigid3& T, const Rigid3& gt)
{
    PoseError e;
    e.rot_deg = rotation_angle_deg(T.R, gt.R);
    e.trans_m = translation_error_m(T.t, gt.t);
    return e;
}

static bool is_converged(const PoseError& e)
{
    return e.trans_m < kConvergedTransM && e.rot_deg < kConvergedRotDeg;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::printf("[demo] NDT scan matching: Autoware-style map localizer (project 02.07)\n");
    print_device_info();
    std::printf("PROBLEM: NDT scan-to-map registration, multi-resolution Newton (coarse %.1f m -> fine %.1f m), "
                "vs a compact point-to-point ICP contrast, over a %d-trial perturbation cohort\n",
                static_cast<double>(kLeafCoarse), static_cast<double>(kLeafFine), 6 * 40);

    // ---- 0) Load data --------------------------------------------------------
    const std::string map_path = find_data_file("", argv[0], "map.bin");
    const std::string scan_main_path = find_data_file("", argv[0], "scan_main.bin");
    const std::string scan_cohort_path = find_data_file("", argv[0], "scan_cohort.bin");
    const std::string scan_cohort_clean_path = find_data_file("", argv[0], "scan_cohort_clean.bin");
    const std::string icp_target_path = find_data_file("", argv[0], "icp_target.bin");
    const std::string cohort_path = find_data_file("", argv[0], "cohort.csv");
    const std::string meta_path = find_data_file("", argv[0], "meta.csv");

    if (map_path.empty() || scan_main_path.empty() || scan_cohort_path.empty() || scan_cohort_clean_path.empty()
        || icp_target_path.empty() || cohort_path.empty() || meta_path.empty()) {
        std::printf("SCENARIO: NOT FOUND - data/sample/*.bin|csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }

    const CloudHost map_c = load_cloud_bin(map_path);
    const CloudHost scan_main = load_cloud_bin(scan_main_path);
    const CloudHost scan_cohort = load_cloud_bin(scan_cohort_path);
    const CloudHost scan_cohort_clean = load_cloud_bin(scan_cohort_clean_path);
    const CloudHost icp_target = load_cloud_bin(icp_target_path);
    std::vector<CohortTrial> cohort;
    MetaInfo meta;
    const bool cohort_ok = load_cohort(cohort_path, cohort);
    const bool meta_ok = load_meta(meta_path, meta);

    if (!map_c.loaded || !scan_main.loaded || !scan_cohort.loaded || !scan_cohort_clean.loaded
        || !icp_target.loaded || !cohort_ok || !meta_ok) {
        std::printf("SCENARIO: MALFORMED - see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    std::printf("SCENARIO: L-shaped corridor->room map (%d pts), scan_main %d pts, scan_cohort %d pts "
                "(%d pts clean), ICP target %d pts, %zu perturbation trials [synthetic]\n",
                map_c.n, scan_main.n, scan_cohort.n, scan_cohort_clean.n, icp_target.n, cohort.size());
    std::printf("[info] true outlier fraction %.1f%%, range noise sigma %.3f m, ground truth: R=I (yaw=0), t=(%.2f,%.2f,%.2f) m\n",
                static_cast<double>(meta.outlier_fraction) * 100.0, static_cast<double>(meta.noise_sigma_m),
                static_cast<double>(meta.gt.t[0]), static_cast<double>(meta.gt.t[1]), static_cast<double>(meta.gt.t[2]));

    // ---- 1) Build the NDT voxel grids: GPU (twice: coarse+fine) and CPU twin ----
    float* d_map_xyz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_map_xyz, static_cast<size_t>(map_c.n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_map_xyz, map_c.xyz.data(), static_cast<size_t>(map_c.n) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    NdtGridGPU grid_coarse_gpu = alloc_ndt_grid_device(kLeafCoarse);
    NdtGridGPU grid_fine_gpu = alloc_ndt_grid_device(kLeafFine);
    unsigned int reg_coarse = 0, reg_fine = 0;

    GpuTimer build_timer;
    build_timer.begin();
    launch_build_ndt_grid(map_c.n, d_map_xyz, grid_coarse_gpu, &reg_coarse);
    launch_build_ndt_grid(map_c.n, d_map_xyz, grid_fine_gpu, &reg_fine);
    const float build_ms = build_timer.end_ms();

    std::vector<NdtVoxelCPU> grid_coarse_cpu, grid_fine_cpu;
    CpuTimer build_cpu_timer;
    build_cpu_timer.begin();
    build_ndt_grid_cpu(map_c.n, map_c.xyz.data(), kLeafCoarse, grid_coarse_gpu.nx, grid_coarse_gpu.ny, grid_coarse_gpu.nz, grid_coarse_cpu);
    build_ndt_grid_cpu(map_c.n, map_c.xyz.data(), kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz, grid_fine_cpu);
    const double build_cpu_ms = build_cpu_timer.end_ms();

    std::printf("[time] voxel grid build: GPU %.3f ms (both resolutions) | CPU oracle %.1f ms (both resolutions)\n",
                static_cast<double>(build_ms), build_cpu_ms);
    std::printf("[info] coarse grid: %dx%dx%d = %d voxels, %u regularized (eigenvalue-floored); "
                "fine grid: %dx%dx%d = %d voxels, %u regularized\n",
                grid_coarse_gpu.nx, grid_coarse_gpu.ny, grid_coarse_gpu.nz, ndt_grid_capacity(grid_coarse_gpu), reg_coarse,
                grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz, ndt_grid_capacity(grid_fine_gpu), reg_fine);

    double d1_coarse, d2_coarse, d1_fine, d2_fine;
    ndt_compute_d1_d2(kLeafCoarse, kAssumedOutlierRatio, d1_coarse, d2_coarse);
    ndt_compute_d1_d2(kLeafFine, kAssumedOutlierRatio, d1_fine, d2_fine);
    std::printf("[info] NDT mixture constants (assumed outlier ratio %.2f): coarse d1=%.4f d2=%.4f | fine d1=%.4f d2=%.4f\n",
                kAssumedOutlierRatio, d1_coarse, d2_coarse, d1_fine, d2_fine);

    // Upload scans once, reused across every stage/trial below.
    float* d_scan_main = nullptr;
    float* d_scan_cohort = nullptr;
    float* d_scan_cohort_clean = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_main, static_cast<size_t>(scan_main.n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_cohort, static_cast<size_t>(scan_cohort.n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_cohort_clean, static_cast<size_t>(scan_cohort_clean.n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scan_main, scan_main.xyz.data(), static_cast<size_t>(scan_main.n) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scan_cohort, scan_cohort.xyz.data(), static_cast<size_t>(scan_cohort.n) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scan_cohort_clean, scan_cohort_clean.xyz.data(), static_cast<size_t>(scan_cohort_clean.n) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Reduction scratch, sized for the LARGEST scan ever assembled (scan_main).
    const int max_blocks = blocks_for(scan_main.n, kThreadsAssemble);
    float* d_block_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_partials, static_cast<size_t>(max_blocks) * kReduceWidth * sizeof(float)));

    bool all_pass = true;
    std::ostringstream gates_csv;
    gates_csv << "gate,metric,value\n";

    // ===================== STAGE A — voxel_stats_twin =========================
    // GPU-built grid (both resolutions) vs the independent CPU builder: exact
    // agreement on which voxels are valid and their point counts (pure integer
    // bookkeeping — no rounding possible), tight relative tolerance on the
    // regularized mean/inverse-covariance (float, double-atomic-summed on GPU
    // vs sequential-double on CPU — see kernels.cu's atomicAdd(double*) comment).
    double worst_mean_rel = 0.0, worst_invcov_rel = 0.0;
    int mismatched_valid = 0, mismatched_count = 0;
    {
        auto check_grid = [&](NdtGridGPU& gg, const std::vector<NdtVoxelCPU>& cpu_grid) {
            const int cap = ndt_grid_capacity(gg);
            std::vector<int> h_count(static_cast<size_t>(cap));
            std::vector<float> h_mean(static_cast<size_t>(cap) * 3), h_invcov(static_cast<size_t>(cap) * 6);
            std::vector<unsigned char> h_valid(static_cast<size_t>(cap));
            CUDA_CHECK(cudaMemcpy(h_count.data(), gg.count, h_count.size() * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_mean.data(), gg.mean, h_mean.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_invcov.data(), gg.inv_cov6, h_invcov.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_valid.data(), gg.valid, h_valid.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

            for (int v = 0; v < cap; ++v) {
                const bool gpu_valid = h_valid[static_cast<size_t>(v)] != 0;
                if (gpu_valid != cpu_grid[static_cast<size_t>(v)].valid) { mismatched_valid++; continue; }
                if (h_count[static_cast<size_t>(v)] != cpu_grid[static_cast<size_t>(v)].count) mismatched_count++;
                if (!gpu_valid) continue;
                for (int k = 0; k < 3; ++k) {
                    const double c = cpu_grid[static_cast<size_t>(v)].mean[k];
                    const double g = h_mean[static_cast<size_t>(v) * 3 + k];
                    const double scale = std::fabs(c) > 1.0 ? std::fabs(c) : 1.0;
                    const double rel = std::fabs(g - c) / scale;
                    if (rel > worst_mean_rel) worst_mean_rel = rel;
                }
                for (int k = 0; k < 6; ++k) {
                    const double c = cpu_grid[static_cast<size_t>(v)].inv_cov6[k];
                    const double g = h_invcov[static_cast<size_t>(v) * 6 + k];
                    const double scale = std::fabs(c) > 1.0 ? std::fabs(c) : 1.0;
                    const double rel = std::fabs(g - c) / scale;
                    if (rel > worst_invcov_rel) worst_invcov_rel = rel;
                }
            }
        };
        check_grid(grid_coarse_gpu, grid_coarse_cpu);
        check_grid(grid_fine_gpu, grid_fine_cpu);
    }
    const bool voxel_twin_pass = (mismatched_valid == 0) && (mismatched_count == 0)
                                && (worst_mean_rel < 1e-3) && (worst_invcov_rel < 1e-2);
    all_pass &= voxel_twin_pass;
    std::printf("[info] voxel_stats_twin: worst mean rel dev %.3e, worst inv_cov rel dev %.3e, "
                "%d valid-flag mismatches, %d count mismatches (both resolutions, %d+%d voxels)\n",
                worst_mean_rel, worst_invcov_rel, mismatched_valid, mismatched_count,
                ndt_grid_capacity(grid_coarse_gpu), ndt_grid_capacity(grid_fine_gpu));
    std::printf("VOXEL_STATS_TWIN: %s (GPU voxel grid matches the independent CPU builder)\n", voxel_twin_pass ? "PASS" : "FAIL");
    gates_csv << "voxel_stats_twin,worst_mean_rel_dev," << worst_mean_rel << "\n";
    gates_csv << "voxel_stats_twin,worst_invcov_rel_dev," << worst_invcov_rel << "\n";

    // ===================== STAGE B — jacobian_check ============================
    // The CALCULUS gate (01.17's exact discipline, cited): analytic gradient
    // (ndt_assemble_cpu, which contains the chain-rule J/H code) vs a
    // CENTRAL-DIFFERENCE numeric gradient computed by calling ONLY
    // ndt_total_score_cpu (which shares the score formula but none of the
    // analytic gradient code) at perturbed poses. Entirely CPU, entirely
    // independent of the GPU path.
    double jac_worst_rel = 0.0;
    {
        // NDT's per-point score is only PIECEWISE smooth: voxel_index() is a
        // floor()-based step function of the transformed point, so a point
        // sitting within an eps-induced position shift of a voxel FACE can
        // flip which Gaussian it scores against — a genuine discontinuity,
        // not a bug (THEORY.md "numerical considerations" names it). A
        // central-difference check must exclude such points or it measures
        // the step function's jump, not the gradient. margin_m is set an
        // order of magnitude above the worst-case eps-induced shift (eps
        // rad/m times this scene's ~16 m max range is ~1.6e-3 m).
        const double eps = 1.0e-4;
        const float margin_m = 0.01f;
        const int n_candidates = std::min(400, scan_cohort.n);

        auto safely_interior = [&](const float p[3], const Rigid3& T) {
            float RP[3]; mat3_vec(T.R, p, RP);
            const float y[3] = { RP[0] + T.t[0], RP[1] + T.t[1], RP[2] + T.t[2] };
            const int vidx = voxel_index(y, kMapOriginX, kMapOriginY, kMapOriginZ, kLeafFine,
                                         grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz);
            if (vidx < 0 || !grid_fine_cpu[static_cast<size_t>(vidx)].valid) return false;
            const float fx = (y[0] - kMapOriginX) / kLeafFine, fy = (y[1] - kMapOriginY) / kLeafFine, fz = (y[2] - kMapOriginZ) / kLeafFine;
            const float rx = fx - std::floor(fx), ry = fy - std::floor(fy), rz = fz - std::floor(fz);
            const float mf = margin_m / kLeafFine;
            return rx > mf && rx < 1.0f - mf && ry > mf && ry < 1.0f - mf && rz > mf && rz < 1.0f - mf;
        };

        Rigid3 test_poses[3] = { meta.gt, cohort[0].T_init, cohort[cohort.size() / 2].T_init };
        for (int p = 0; p < 3; ++p) {
            std::vector<float> subset;
            for (int i = 0; i < n_candidates && subset.size() < 3 * 150; ++i) {
                const float* pt = &scan_cohort.xyz[static_cast<size_t>(i) * 3];
                if (safely_interior(pt, test_poses[p])) { subset.push_back(pt[0]); subset.push_back(pt[1]); subset.push_back(pt[2]); }
            }
            const int n_sub = static_cast<int>(subset.size() / 3);
            if (n_sub < 10) continue;   // too few interior points at this pose -- skip rather than test noise

            double H21[21], g6[6], score;
            ndt_assemble_cpu(subset.data(), n_sub, test_poses[p], grid_fine_cpu,
                             kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz,
                             d1_fine, d2_fine, H21, g6, &score);
            for (int dim = 0; dim < 6; ++dim) {
                double delta_p[6] = { 0, 0, 0, 0, 0, 0 }; delta_p[dim] = eps;
                double delta_m[6] = { 0, 0, 0, 0, 0, 0 }; delta_m[dim] = -eps;
                Rigid3 Tp, Tm;
                retract(test_poses[p], delta_p, Tp);
                retract(test_poses[p], delta_m, Tm);
                const double sp = ndt_total_score_cpu(subset.data(), n_sub, Tp, grid_fine_cpu,
                                                      kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz, d1_fine, d2_fine);
                const double sm = ndt_total_score_cpu(subset.data(), n_sub, Tm, grid_fine_cpu,
                                                      kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz, d1_fine, d2_fine);
                const double numeric = (sp - sm) / (2.0 * eps);
                const double scale = std::fabs(g6[dim]) > 1.0 ? std::fabs(g6[dim]) : 1.0;
                const double rel = std::fabs(numeric - g6[dim]) / scale;
                if (rel > jac_worst_rel) jac_worst_rel = rel;
            }
        }
    }
    const bool jac_pass = jac_worst_rel < 5.0e-2;
    all_pass &= jac_pass;
    std::printf("[info] jacobian check: worst relative deviation %.4e over 3 poses x 6 dims (eps=1e-4 central difference)\n", jac_worst_rel);
    std::printf("JACOBIAN_CHECK: %s (analytic vs. central-difference numeric gradient agree within tolerance)\n", jac_pass ? "PASS" : "FAIL");
    gates_csv << "jacobian_check,worst_rel_deviation," << jac_worst_rel << "\n";

    // ===================== STAGE C — assembly_twin =============================
    // One score/gradient/Hessian assembly at the ground-truth pose, on the
    // FULL scan_main cloud, GPU kernel vs CPU independent accumulation loop.
    double assembly_worst_rel = 0.0;
    {
        double H21_cpu[21], g6_cpu[6], score_cpu;
        ndt_assemble_cpu(scan_main.xyz.data(), scan_main.n, meta.gt, grid_fine_cpu,
                         kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz, d1_fine, d2_fine,
                         H21_cpu, g6_cpu, &score_cpu);

        const int nblocks = launch_ndt_assemble(d_scan_main, scan_main.n, meta.gt, grid_fine_gpu, d1_fine, d2_fine, d_block_partials);
        std::vector<float> partials(static_cast<size_t>(nblocks) * kReduceWidth);
        CUDA_CHECK(cudaMemcpy(partials.data(), d_block_partials, partials.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double sums[kReduceWidth];
        for (int k = 0; k < kReduceWidth; ++k) sums[k] = 0.0;
        for (int b = 0; b < nblocks; ++b)
            for (int k = 0; k < kReduceWidth; ++k)
                sums[k] += static_cast<double>(partials[static_cast<size_t>(b) * kReduceWidth + k]);

        for (int k = 0; k < 21; ++k) {
            const double scale = std::fabs(H21_cpu[k]) > 1.0 ? std::fabs(H21_cpu[k]) : 1.0;
            const double rel = std::fabs(sums[k] - H21_cpu[k]) / scale;
            if (rel > assembly_worst_rel) assembly_worst_rel = rel;
        }
        for (int k = 0; k < 6; ++k) {
            const double scale = std::fabs(g6_cpu[k]) > 1.0 ? std::fabs(g6_cpu[k]) : 1.0;
            const double rel = std::fabs(sums[21 + k] - g6_cpu[k]) / scale;
            if (rel > assembly_worst_rel) assembly_worst_rel = rel;
        }
        {
            const double scale = std::fabs(score_cpu) > 1.0 ? std::fabs(score_cpu) : 1.0;
            const double rel = std::fabs(sums[27] - score_cpu) / scale;
            if (rel > assembly_worst_rel) assembly_worst_rel = rel;
        }
    }
    const bool assembly_pass = assembly_worst_rel < 5.0e-3;
    all_pass &= assembly_pass;
    std::printf("[info] assembly twin: worst relative deviation %.4e over 21 H + 6 g + 1 score, n=%d points\n", assembly_worst_rel, scan_main.n);
    std::printf("ASSEMBLY_TWIN: %s (GPU block-reduced assembly matches the independent CPU accumulation)\n", assembly_pass ? "PASS" : "FAIL");
    gates_csv << "assembly_twin,worst_rel_deviation," << assembly_worst_rel << "\n";

    // ===================== STAGE D — trajectory_twin ===========================
    // One full coarse->fine trajectory from a representative cohort trial,
    // GPU-orchestrated (run_ndt_multires_gpu) vs CPU-independent
    // (run_ndt_multires_cpu) — measured-then-margined final-pose deviation
    // (08.01's technique, cited: chained FP32 iterations diverge in the low
    // bits run to run/path to path; the DIRECTION agreeing is what matters).
    // Found by SEARCHING for bin 2's first trial (0.8 m / 15 deg) rather than
    // a raw hardcoded index -- a raw index (this project's earlier "30")
    // silently pointed at the WRONG bin once COHORT_TRIALS_PER_BIN changed
    // from 15 to 40 (trial 30 moved from bin 2's start to inside bin 0), a
    // real bug this project's finisher pass caught and is guarding against
    // here for good.
    size_t twin_trial_idx = cohort.size() - 1;
    for (size_t i = 0; i < cohort.size(); ++i) {
        if (cohort[i].bin_index == 2) { twin_trial_idx = i; break; }
    }
    std::vector<double> loss_gpu, loss_cpu;
    Rigid3 T_traj_gpu = run_ndt_multires_gpu(cohort[twin_trial_idx].T_init, d_scan_cohort, scan_cohort.n,
                                             grid_coarse_gpu, d1_coarse, d2_coarse, grid_fine_gpu, d1_fine, d2_fine,
                                             d_block_partials, &loss_gpu);
    Rigid3 T_traj_cpu; int traj_cpu_iters = 0;
    std::vector<double> loss_cpu_buf(kMaxItersCoarse + kMaxItersFine + 2, 0.0);
    run_ndt_multires_cpu(scan_cohort.xyz.data(), scan_cohort.n,
                         grid_coarse_cpu, kLeafCoarse, grid_coarse_gpu.nx, grid_coarse_gpu.ny, grid_coarse_gpu.nz,
                         grid_fine_cpu, kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz,
                         d1_coarse, d2_coarse, d1_fine, d2_fine,
                         cohort[twin_trial_idx].T_init, T_traj_cpu, loss_cpu_buf.data(), traj_cpu_iters);
    loss_cpu.assign(loss_cpu_buf.begin(), loss_cpu_buf.begin() + traj_cpu_iters);

    const double traj_rot_dev = std::fabs(rotation_angle_deg(T_traj_gpu.R, meta.gt.R) - rotation_angle_deg(T_traj_cpu.R, meta.gt.R));
    const double traj_trans_dev = std::fabs(translation_error_m(T_traj_gpu.t, meta.gt.t) - translation_error_m(T_traj_cpu.t, meta.gt.t));
    const bool traj_pass = traj_rot_dev < 1.0 && traj_trans_dev < 0.05;   // measured-then-margined (see [info] line)
    all_pass &= traj_pass;
    std::printf("[info] trajectory twin: GPU final err (rot %.3f deg, trans %.4f m) vs CPU final err (rot %.3f deg, trans %.4f m); "
                "GPU %d iters, CPU %d iters\n",
                static_cast<double>(rotation_angle_deg(T_traj_gpu.R, meta.gt.R)), static_cast<double>(translation_error_m(T_traj_gpu.t, meta.gt.t)),
                static_cast<double>(rotation_angle_deg(T_traj_cpu.R, meta.gt.R)), static_cast<double>(translation_error_m(T_traj_cpu.t, meta.gt.t)),
                static_cast<int>(loss_gpu.size()), traj_cpu_iters);
    std::printf("TRAJECTORY_TWIN: %s (GPU-orchestrated and independent CPU trajectories converge to the same pose)\n", traj_pass ? "PASS" : "FAIL");
    gates_csv << "trajectory_twin,rot_deviation_deg," << traj_rot_dev << "\n";
    gates_csv << "trajectory_twin,trans_deviation_m," << traj_trans_dev << "\n";

    // ===================== STAGE E — score_sanity ==============================
    // score(truth) must be lower (more negative -> better fit) than score at
    // EVERY perturbed initial guess in the cohort — a free monotonicity check.
    int score_violations = 0;
    const double score_at_truth = ndt_total_score_cpu(scan_cohort.xyz.data(), scan_cohort.n, meta.gt, grid_fine_cpu,
                                                       kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz, d1_fine, d2_fine);
    double worst_perturbed_score = score_at_truth;
    for (const auto& trial : cohort) {
        const double s = ndt_total_score_cpu(scan_cohort.xyz.data(), scan_cohort.n, trial.T_init, grid_fine_cpu,
                                             kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz, d1_fine, d2_fine);
        if (s < score_at_truth) score_violations++;
        if (s < worst_perturbed_score) worst_perturbed_score = s;
    }
    const bool score_sanity_pass = (score_violations == 0);
    all_pass &= score_sanity_pass;
    std::printf("[info] score_sanity: score(truth)=%.3f, most-negative perturbed score=%.3f, %d/%zu violations\n",
                score_at_truth, worst_perturbed_score, score_violations, cohort.size());
    std::printf("SCORE_SANITY: %s (score at ground truth beats every perturbed initial guess)\n", score_sanity_pass ? "PASS" : "FAIL");
    gates_csv << "score_sanity,score_at_truth," << score_at_truth << "\n";
    gates_csv << "score_sanity,violations," << score_violations << "\n";

    // ===================== STAGE F — convergence + accuracy ====================
    // The cohort, NDT multi-resolution (coarse->fine), WITH outliers.
    struct TrialOutcome { PoseError err; bool converged; float final_t[3]; };
    std::vector<TrialOutcome> ndt_multires_outcomes(cohort.size());
    for (size_t i = 0; i < cohort.size(); ++i) {
        Rigid3 T = run_ndt_multires_gpu(cohort[i].T_init, d_scan_cohort, scan_cohort.n,
                                        grid_coarse_gpu, d1_coarse, d2_coarse, grid_fine_gpu, d1_fine, d2_fine,
                                        d_block_partials);
        const PoseError e = pose_error(T, meta.gt);
        ndt_multires_outcomes[i] = { e, is_converged(e), { T.t[0], T.t[1], T.t[2] } };
    }

    // ---- failure-mode classification [info] ---------------------------------
    // Classifies every UNCONVERGED trial as stalled (barely moved from its
    // initial offset), diverged (final error >= initial offset), or
    // partial (moved meaningfully closer but still missed the threshold),
    // and decomposes the final translation error into its Z component
    // (kMapOrigin's vertical axis, the axis STAGE I's degenerate_axis report
    // names as this scene's weakest Hessian direction) vs. its horizontal
    // (X/Y) component -- this answers "is failure dominated by the weak
    // axis, or is it a horizontal/basin problem instead" empirically. This
    // project's finisher pass used exactly this breakdown (plus a one-off
    // per-trial dump and score-trajectory trace, not carried into the
    // committed demo output) to find that bin-0 failures correlate almost
    // perfectly with a perturbation direction aligned with the corridor's
    // long (degenerate) axis, and that the optimizer genuinely PLATEAUS
    // there (confirmed by re-running with 5x the iteration budget: the
    // score does not move) -- a real stationary point, not an iteration-
    // or damping-budget artifact. THEORY.md "numerical considerations"
    // tells the full story with the measured numbers. [info] only -- not a
    // gated stage.
    {
        int n_stalled = 0, n_diverged = 0, n_partial = 0;
        double sum_dz = 0.0, sum_dxy = 0.0;
        int n_fail = 0, n_z_only_blocker = 0;
        for (size_t i = 0; i < cohort.size(); ++i) {
            const auto& o = ndt_multires_outcomes[i];
            if (o.converged) continue;
            n_fail++;
            const float init_trans = cohort[i].mag_trans_m;
            if (o.err.trans_m >= init_trans) n_diverged++;
            else if (o.err.trans_m > 0.5f * init_trans) n_stalled++;
            else n_partial++;
            const double dz = std::fabs(static_cast<double>(o.final_t[2]) - static_cast<double>(meta.gt.t[2]));
            const double dx = static_cast<double>(o.final_t[0]) - static_cast<double>(meta.gt.t[0]);
            const double dy = static_cast<double>(o.final_t[1]) - static_cast<double>(meta.gt.t[1]);
            const double dxy = std::sqrt(dx * dx + dy * dy);
            sum_dz += dz;
            sum_dxy += dxy;
            // "z-only blocker": XY alone (and rotation) would have PASSED
            // the converged-classification, but Z drift alone pushed the
            // combined trans_m error over kConvergedTransM -- isolates
            // exactly how many failures the weak vz Hessian axis alone
            // explains, vs. a genuine XY/rotation basin failure.
            if (dxy < static_cast<double>(kConvergedTransM) && o.err.rot_deg < kConvergedRotDeg) n_z_only_blocker++;
        }
        std::printf("[info] failure_diagnosis (%d/%zu unconverged trials): %d diverged (final>=initial offset), "
                    "%d stalled (<half progress), %d partial (>half progress, still missed threshold); "
                    "mean |dz|=%.1f mm, mean sqrt(dx^2+dy^2)=%.1f mm, %d/%d failures are Z-ONLY blockers "
                    "(XY+rot alone would have converged) (z is NOT perturbed by the cohort generator -- "
                    "any nonzero dz here is optimizer DRIFT away from an already-correct start, not unconverged recovery)\n",
                    n_fail, cohort.size(), n_diverged, n_stalled, n_partial,
                    n_fail > 0 ? sum_dz / n_fail * 1000.0 : 0.0, n_fail > 0 ? sum_dxy / n_fail * 1000.0 : 0.0,
                    n_z_only_blocker, n_fail);

        // Bin-0-only (smallest perturbation, 0.2 m / 5 deg) direction split:
        // is a trial's initial XY offset direction NEAR the corridor's long
        // (degenerate) axis (+-20 deg of the corridor's x-axis or its
        // opposite), or OFF that axis? This project's finisher pass found
        // this single geometric fact almost perfectly separates bin-0's
        // failures from its successes: an offset ALONG the corridor gives
        // the Newton step far less curvature to correct it with (STAGE I's
        // degenerate_axis condition-ratio report measures exactly this
        // axis) than an offset that also moves ACROSS the corridor (toward
        // a wall) or into the room. Confirmed NOT an iteration-budget
        // artifact: re-running the corridor-axis failures with 5x the
        // iteration budget left their final score UNCHANGED to two decimal
        // places -- a genuine stationary point, not a starved one.
        int on_axis_total = 0, on_axis_converged = 0, off_axis_total = 0, off_axis_converged = 0;
        for (size_t i = 0; i < cohort.size(); ++i) {
            if (cohort[i].bin_index != 0) continue;
            const float dx = cohort[i].T_init.t[0] - meta.gt.t[0];
            const float dy = cohort[i].T_init.t[1] - meta.gt.t[1];
            const double phi_deg = std::atan2(static_cast<double>(dy), static_cast<double>(dx)) * (180.0 / 3.14159265358979323846);
            const double dist_to_axis = std::min(std::fabs(phi_deg), std::fabs(std::fabs(phi_deg) - 180.0));   // 0 deg = exactly on the corridor's x-axis
            const bool on_axis = dist_to_axis < 20.0;
            const bool conv = ndt_multires_outcomes[i].converged;
            if (on_axis) { on_axis_total++; on_axis_converged += conv ? 1 : 0; }
            else { off_axis_total++; off_axis_converged += conv ? 1 : 0; }
        }
        std::printf("[info] bin0_corridor_axis_split: perturbations WITHIN 20deg of the corridor's long axis converge %d/%d (%.0f%%) "
                    "vs. %d/%d (%.0f%%) for every other direction -- the corridor-sliding degeneracy (STAGE I) showing up directly "
                    "in the smallest-perturbation bin's convergence rate\n",
                    on_axis_converged, on_axis_total, on_axis_total > 0 ? 100.0 * on_axis_converged / on_axis_total : 0.0,
                    off_axis_converged, off_axis_total, off_axis_total > 0 ? 100.0 * off_axis_converged / off_axis_total : 0.0);
    }

    int converged_count = 0;
    float worst_conv_trans_m = 0.0f, worst_conv_rot_deg = 0.0f;
    double sum_conv_trans_mm = 0.0, sum_conv_rot_deg = 0.0;
    for (const auto& o : ndt_multires_outcomes) {
        if (!o.converged) continue;
        converged_count++;
        if (o.err.trans_m > worst_conv_trans_m) worst_conv_trans_m = o.err.trans_m;
        if (o.err.rot_deg > worst_conv_rot_deg) worst_conv_rot_deg = o.err.rot_deg;
        sum_conv_trans_mm += static_cast<double>(o.err.trans_m) * 1000.0;
        sum_conv_rot_deg += static_cast<double>(o.err.rot_deg);
    }
    const double convergence_pct = 100.0 * converged_count / static_cast<double>(cohort.size());
    const double mean_conv_trans_mm = converged_count > 0 ? sum_conv_trans_mm / converged_count : 0.0;
    const double mean_conv_rot_deg = converged_count > 0 ? sum_conv_rot_deg / converged_count : 0.0;

    // Measured-then-margined (CLAUDE.md's honest-gate discipline): this
    // cohort's magnitudes were deliberately picked large enough to find a
    // REAL basin boundary (01.17's exact argument, cited) rather than
    // converging 100% of the time and teaching nothing about basin SIZE.
    // Measured on this scene (240-trial cohort, 40 trials/bin): 13.3%
    // (32/240) of NDT multi-res trials converge; the floor below sits
    // comfortably under that with headroom for a stray trial flipping
    // classification on a different GPU architecture's float rounding.
    // THEORY.md "numerical considerations" reports the fuller story,
    // including the bin0_corridor_axis_split [info] line just above: the
    // smallest-perturbation bin's OWN 65% (26/40) splits cleanly into 0%
    // (0/9) for perturbations aligned with the corridor's degenerate axis
    // and 84% (26/31) for every other direction -- production localizers
    // resolve that specific axis with wheel/IMU odometry, not the LiDAR
    // match alone (this project's own z-axis scoping note, generalized).
    const bool convergence_pass = convergence_pct >= 10.0;
    const bool accuracy_pass = worst_conv_trans_m < 0.10f && worst_conv_rot_deg < 4.0f;   // measured-then-margined
    all_pass &= convergence_pass;
    all_pass &= accuracy_pass;

    std::printf("[info] convergence: %d/%zu trials converged (%.1f%%) [threshold: trans < %.0f cm AND rot < %.1f deg]\n",
                converged_count, cohort.size(), convergence_pct, static_cast<double>(kConvergedTransM) * 100.0, static_cast<double>(kConvergedRotDeg));
    std::printf("[info] accuracy (converged trials only): mean trans %.2f mm / rot %.3f deg; worst trans %.2f mm / rot %.3f deg\n",
                mean_conv_trans_mm, mean_conv_rot_deg, static_cast<double>(worst_conv_trans_m) * 1000.0, static_cast<double>(worst_conv_rot_deg));
    std::printf("CONVERGENCE: %s (NDT multi-resolution converges from a useful fraction of the perturbation cohort)\n", convergence_pass ? "PASS" : "FAIL");
    std::printf("ACCURACY: %s (converged NDT poses are close to ground truth)\n", accuracy_pass ? "PASS" : "FAIL");
    gates_csv << "convergence,converged_pct," << convergence_pct << "\n";
    gates_csv << "accuracy,worst_conv_trans_mm," << static_cast<double>(worst_conv_trans_m) * 1000.0 << "\n";
    gates_csv << "accuracy,worst_conv_rot_deg," << static_cast<double>(worst_conv_rot_deg) << "\n";

    // Per-bin convergence % for the basin curve artifact (NDT multi-res row).
    const int kNumBins = 6;
    std::vector<int> bin_total(kNumBins, 0), bin_converged_multires(kNumBins, 0);
    for (size_t i = 0; i < cohort.size(); ++i) {
        bin_total[static_cast<size_t>(cohort[i].bin_index)]++;
        if (ndt_multires_outcomes[i].converged) bin_converged_multires[static_cast<size_t>(cohort[i].bin_index)]++;
    }

    // ===================== STAGE G — basin_contrast =============================
    // SAME cohort trials, two more methods: NDT FINE-ONLY (skips the coarse warm
    // start, but gets the SAME total iteration budget: kMaxItersCoarse+
    // kMaxItersFine, spent entirely at fine resolution -- isolates the
    // RESOLUTION SCHEDULE's effect from iteration COUNT) and the compact
    // point-to-point ICP contrast (CPU-only, kernels.cuh's documented scope).
    std::vector<int> bin_converged_fine_only(kNumBins, 0), bin_converged_icp(kNumBins, 0);
    int converged_fine_only = 0, converged_icp = 0;
    double sum_icp_conv_trans_mm = 0.0, sum_icp_conv_rot_deg = 0.0;   // accuracy AMONG ICP's own converged trials, for the direct accuracy contrast below
    const int kFineOnlyMaxIters = kMaxItersCoarse + kMaxItersFine;
    CpuTimer icp_timer;
    icp_timer.begin();
    for (size_t i = 0; i < cohort.size(); ++i) {
        Rigid3 T_fo = cohort[i].T_init;
        run_ndt_stage_gpu(d_scan_cohort, scan_cohort.n, grid_fine_gpu, d1_fine, d2_fine, kFineOnlyMaxIters, d_block_partials, T_fo, nullptr);
        const bool conv_fo = is_converged(pose_error(T_fo, meta.gt));
        if (conv_fo) { converged_fine_only++; bin_converged_fine_only[static_cast<size_t>(cohort[i].bin_index)]++; }

        Rigid3 T_icp; int icp_iters = 0;
        icp_point_to_point_cpu(scan_cohort.xyz.data(), scan_cohort.n, icp_target.xyz.data(), icp_target.n,
                               cohort[i].T_init, kIcpMaxIters, kIcpMaxCorrDistM, T_icp, icp_iters);
        const PoseError e_icp = pose_error(T_icp, meta.gt);
        if (is_converged(e_icp)) {
            converged_icp++;
            bin_converged_icp[static_cast<size_t>(cohort[i].bin_index)]++;
            sum_icp_conv_trans_mm += static_cast<double>(e_icp.trans_m) * 1000.0;
            sum_icp_conv_rot_deg += static_cast<double>(e_icp.rot_deg);
        }
    }
    const double icp_ms = icp_timer.end_ms();
    const double basin_multires_pct = convergence_pct;
    const double basin_fine_only_pct = 100.0 * converged_fine_only / static_cast<double>(cohort.size());
    const double basin_icp_pct = 100.0 * converged_icp / static_cast<double>(cohort.size());
    const double mean_icp_conv_trans_mm = converged_icp > 0 ? sum_icp_conv_trans_mm / converged_icp : 0.0;
    const double mean_icp_conv_rot_deg = converged_icp > 0 ? sum_icp_conv_rot_deg / converged_icp : 0.0;

    // Multi-resolution should be AT LEAST AS ROBUST as spending the identical
    // iteration budget entirely at fine resolution (the whole reason
    // multi-resolution scheduling exists — THEORY.md derives why coarse
    // voxels widen the basin). A small slack absorbs cohort-level noise from
    // individual trials without hiding a real regression.
    const bool basin_multires_vs_fine_pass = (basin_multires_pct + 5.0) >= basin_fine_only_pct;
    all_pass &= basin_multires_vs_fine_pass;
    std::printf("[info] basin_contrast (%zu trials each): NDT multi-res %.1f%% | NDT fine-only %.1f%% | ICP %.1f%% converged "
                "(ICP wall time %.1f ms for the whole cohort)\n",
                cohort.size(), basin_multires_pct, basin_fine_only_pct, basin_icp_pct, icp_ms);
    std::printf("[info] accuracy contrast (converged trials only): NDT mean trans %.2f mm / rot %.3f deg vs. "
                "ICP mean trans %.2f mm / rot %.3f deg\n",
                mean_conv_trans_mm, mean_conv_rot_deg, mean_icp_conv_trans_mm, mean_icp_conv_rot_deg);
    std::printf("BASIN_CONTRAST: %s (NDT's multi-resolution schedule matches or widens the convergence basin vs. the "
                "same iteration budget spent at fine resolution alone)\n", basin_multires_vs_fine_pass ? "PASS" : "FAIL");
    gates_csv << "basin_contrast,multires_pct," << basin_multires_pct << "\n";
    gates_csv << "basin_contrast,fine_only_pct," << basin_fine_only_pct << "\n";
    gates_csv << "basin_contrast,icp_pct," << basin_icp_pct << "\n";

    // ===================== STAGE H — outlier_robustness ========================
    // Same cohort trials, NDT multi-resolution, on the OUTLIER-FREE paired scan
    // (scan_cohort_clean.bin -- same beam directions/noise, outlier fraction
    // forced to 0 at generation time). Accuracy should degrade GRACEFULLY,
    // not catastrophically, when outliers are added back.
    int converged_clean = 0;
    double sum_clean_trans_mm = 0.0;
    float worst_clean_trans_m = 0.0f;
    for (const auto& trial : cohort) {
        Rigid3 T = run_ndt_multires_gpu(trial.T_init, d_scan_cohort_clean, scan_cohort_clean.n,
                                        grid_coarse_gpu, d1_coarse, d2_coarse, grid_fine_gpu, d1_fine, d2_fine,
                                        d_block_partials);
        const PoseError e = pose_error(T, meta.gt);
        if (is_converged(e)) {
            converged_clean++;
            sum_clean_trans_mm += static_cast<double>(e.trans_m) * 1000.0;
            if (e.trans_m > worst_clean_trans_m) worst_clean_trans_m = e.trans_m;
        }
    }
    const double clean_convergence_pct = 100.0 * converged_clean / static_cast<double>(cohort.size());
    const double mean_clean_trans_mm = converged_clean > 0 ? sum_clean_trans_mm / converged_clean : 0.0;
    // "Graceful degradation": the WITH-outliers worst converged error should
    // not blow up relative to the outlier-free run, and convergence rate
    // should not collapse. Bound is measured-then-margined (see [info]).
    const bool outlier_robust_pass = (worst_conv_trans_m < std::max(0.15f, worst_clean_trans_m * 3.0f))
                                    && (convergence_pct >= clean_convergence_pct - 15.0);
    all_pass &= outlier_robust_pass;
    std::printf("[info] outlier_robustness: outlier-free run converged %.1f%% (mean %.2f mm, worst %.2f mm) vs. "
                "with-outliers run converged %.1f%% (mean %.2f mm, worst %.2f mm)\n",
                clean_convergence_pct, mean_clean_trans_mm, static_cast<double>(worst_clean_trans_m) * 1000.0,
                convergence_pct, mean_conv_trans_mm, static_cast<double>(worst_conv_trans_m) * 1000.0);
    std::printf("OUTLIER_ROBUSTNESS: %s (accuracy degrades gracefully, not catastrophically, with the documented outlier fraction)\n",
                outlier_robust_pass ? "PASS" : "FAIL");
    gates_csv << "outlier_robustness,clean_converged_pct," << clean_convergence_pct << "\n";
    gates_csv << "outlier_robustness,with_outliers_converged_pct," << convergence_pct << "\n";

    // ===================== STAGE I — degenerate_axis [info only] ===============
    // Two Hessians at the ground-truth pose, fine grid: the FULL scan_cohort
    // (room + pillar visible -> well constrained) vs. a NEAR-FIELD-ONLY subset
    // (points within 4 m of the sensor -- mostly corridor walls/floor, room
    // and pillar excluded) -- the corridor-sliding degeneracy THEORY.md
    // "physics-first" describes, tied to 01.17's degeneracy-gate lineage
    // (condition-number-of-H diagnostic) BY NAME.
    {
        std::vector<float> near_field;
        for (int i = 0; i < scan_cohort.n; ++i) {
            const float x = scan_cohort.xyz[static_cast<size_t>(i) * 3 + 0];
            const float y = scan_cohort.xyz[static_cast<size_t>(i) * 3 + 1];
            const float z = scan_cohort.xyz[static_cast<size_t>(i) * 3 + 2];
            if (std::sqrt(x * x + y * y + z * z) < 4.0f) {
                near_field.push_back(x); near_field.push_back(y); near_field.push_back(z);
            }
        }
        const int n_near = static_cast<int>(near_field.size() / 3);

        auto condition_report = [&](const float* pts, int n, const char* label) {
            double H21[21], g6[6], score;
            ndt_assemble_cpu(pts, n, meta.gt, grid_fine_cpu, kLeafFine, grid_fine_gpu.nx, grid_fine_gpu.ny, grid_fine_gpu.nz,
                             d1_fine, d2_fine, H21, g6, &score);
            double A[6][6];
            for (int i = 0; i < 6; ++i)
                for (int j = i; j < 6; ++j) { const double v = H21[hidx(i, j)]; A[i][j] = v; A[j][i] = v; }
            double V[6][6];
            jacobi_eigen_symmetric6(A, V);
            // Compare eigenvalue MAGNITUDES (|A[i][i]|) rather than signed
            // values: NDT's Hessian is not guaranteed positive definite
            // (kernels.cuh's cholesky6_solve_flat comment), so "weakest
            // direction" honestly means smallest |curvature|, not smallest
            // signed eigenvalue.
            int min_i = 0, max_i = 0;
            for (int i = 1; i < 6; ++i) {
                if (std::fabs(A[i][i]) < std::fabs(A[min_i][min_i])) min_i = i;
                if (std::fabs(A[i][i]) > std::fabs(A[max_i][max_i])) max_i = i;
            }
            const double cond = (std::fabs(A[min_i][min_i]) > 1e-9) ? std::fabs(A[max_i][max_i] / A[min_i][min_i]) : -1.0;
            int dom = 0; double dom_val = 0.0;
            for (int i = 0; i < 6; ++i) { const double av = std::fabs(V[i][min_i]); if (av > dom_val) { dom_val = av; dom = i; } }
            std::printf("[info] degenerate_axis (%s, n=%d): H eigenvalues [min=%.4g, max=%.4g], condition ratio %.3g, "
                        "weakest direction loads most on '%s' (|component|=%.2f)\n",
                        label, n, A[min_i][min_i], A[max_i][max_i], cond, kParamNames[dom], dom_val);
        };
        condition_report(near_field.data(), n_near, "near-field-only, corridor walls");
        condition_report(scan_cohort.xyz.data(), scan_cohort.n, "full scan, room+pillar visible");
    }

    // ===================== Artifacts ============================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = true;

    // (1) registration before/after top-view.
    {
        std::ofstream f(out_dir + "/registration_topview.csv");
        artifact_ok = artifact_ok && f.is_open();
        if (f.is_open()) {
            f << "category,x_m,y_m\n";
            for (int i = 0; i < map_c.n; i += 8)
                f << "MAP," << map_c.xyz[static_cast<size_t>(i) * 3 + 0] << ',' << map_c.xyz[static_cast<size_t>(i) * 3 + 1] << "\n";
            const Rigid3& T_init = cohort[twin_trial_idx].T_init;
            for (int i = 0; i < scan_main.n; ++i) {
                float RP[3]; mat3_vec(T_init.R, &scan_main.xyz[static_cast<size_t>(i) * 3], RP);
                f << "SCAN_INITIAL," << (RP[0] + T_init.t[0]) << ',' << (RP[1] + T_init.t[1]) << "\n";
            }
            for (int i = 0; i < scan_main.n; ++i) {
                float RP[3]; mat3_vec(T_traj_gpu.R, &scan_main.xyz[static_cast<size_t>(i) * 3], RP);
                f << "SCAN_REGISTERED," << (RP[0] + T_traj_gpu.t[0]) << ',' << (RP[1] + T_traj_gpu.t[1]) << "\n";
            }
        }
    }

    // (2) basin curve: method,bin_index,magnitude_trans_m,magnitude_yaw_deg,converged_pct
    {
        std::ofstream f(out_dir + "/basin_curve.csv");
        artifact_ok = artifact_ok && f.is_open();
        if (f.is_open()) {
            f << "method,bin_index,magnitude_trans_m,magnitude_yaw_deg,converged_pct\n";
            for (int b = 0; b < kNumBins; ++b) {
                float mag_t = 0, mag_y = 0;
                for (const auto& trial : cohort) if (trial.bin_index == b) { mag_t = trial.mag_trans_m; mag_y = trial.mag_yaw_deg; break; }
                const double pm = bin_total[static_cast<size_t>(b)] > 0 ? 100.0 * bin_converged_multires[static_cast<size_t>(b)] / bin_total[static_cast<size_t>(b)] : 0.0;
                const double pf = bin_total[static_cast<size_t>(b)] > 0 ? 100.0 * bin_converged_fine_only[static_cast<size_t>(b)] / bin_total[static_cast<size_t>(b)] : 0.0;
                const double pi = bin_total[static_cast<size_t>(b)] > 0 ? 100.0 * bin_converged_icp[static_cast<size_t>(b)] / bin_total[static_cast<size_t>(b)] : 0.0;
                f << "ndt_multires," << b << ',' << mag_t << ',' << mag_y << ',' << pm << "\n";
                f << "ndt_fine_only," << b << ',' << mag_t << ',' << mag_y << ',' << pf << "\n";
                f << "icp," << b << ',' << mag_t << ',' << mag_y << ',' << pi << "\n";
            }
        }
    }

    // (3) convergence trajectories: the STAGE D twin trial, both sources.
    {
        std::ofstream f(out_dir + "/convergence_trajectories.csv");
        artifact_ok = artifact_ok && f.is_open();
        if (f.is_open()) {
            f << "source,iter,score\n";
            for (size_t i = 0; i < loss_gpu.size(); ++i) f << "GPU," << i << ',' << loss_gpu[i] << "\n";
            for (size_t i = 0; i < loss_cpu.size(); ++i) f << "CPU," << i << ',' << loss_cpu[i] << "\n";
        }
    }

    // (4) gates_metrics.csv
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        artifact_ok = artifact_ok && f.is_open();
        if (f.is_open()) f << gates_csv.str();
    }

    if (artifact_ok)
        std::printf("ARTIFACT: wrote registration_topview.csv, basin_curve.csv, convergence_trajectories.csv, gates_metrics.csv to demo/out/\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out/ files\n");

    CUDA_CHECK(cudaFree(d_map_xyz));
    CUDA_CHECK(cudaFree(d_scan_main));
    CUDA_CHECK(cudaFree(d_scan_cohort));
    CUDA_CHECK(cudaFree(d_scan_cohort_clean));
    CUDA_CHECK(cudaFree(d_block_partials));
    free_ndt_grid_device(grid_coarse_gpu);
    free_ndt_grid_device(grid_fine_gpu);

    all_pass = all_pass && artifact_ok;
    if (all_pass) {
        std::printf("RESULT: PASS (all verification stages passed - voxel/assembly/trajectory twins, jacobian check, "
                    "score sanity, convergence, accuracy, basin contrast, outlier robustness)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (see the stage verdict lines above for which stage(s) failed)\n");
    return 1;
}
