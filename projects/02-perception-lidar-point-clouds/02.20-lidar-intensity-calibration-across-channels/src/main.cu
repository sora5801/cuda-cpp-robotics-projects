// ===========================================================================
// main.cu - entry point for project 02.20 (LiDAR intensity calibration
//           across channels)
//
// Role in the project
// --------------------
// Orchestration: load the two committed scans (scan_primary.csv,
// scan_degenerate.csv) + ground-truth gains, run the GPU self-calibration
// pipeline (point_features -> bin_accumulate -> assemble_ls -> [shared
// solve] -> apply_gain) AND the independent CPU reference pipeline on the
// primary scan, VERIFY agreement stage by stage, run the degenerate scan
// through the same pipeline for the observability gate, evaluate four
// INDEPENDENT gates against ground truth, write every artifact the
// README/demo describe, and report one final PASS/FAIL.
//
// The pipeline (kernels.cuh derives every stage's math in full):
//   point_features   = per-point forward-model inversion + voxel key   [MAP]
//   bin_accumulate   = per-(voxel,channel) log-intensity mean          [SCATTER-REDUCE]
//   assemble_ls      = per-shared-voxel centering-projector -> 16x16 A,b [SCATTER-REDUCE]
//   solve_channel_gains = SHARED host-only 16x16 solve (called ONCE)   [micro-solve]
//   apply_gain       = divide the recovered gain back out              [MAP]
//
// Output contract (load-bearing!) - same discipline as every project in this
// repo: "[demo]", "PROBLEM:", "DATA:", "VERIFY:", "GATE ...:", "ARTIFACT:"
// and "RESULT:" lines are STABLE (no timings, no device names, no measured
// floats) and are diffed against demo/expected_output.txt; "[time]" and
// "[info]" lines carry the actual measured numbers and are deliberately NOT
// diffed.
//
// Read this after: kernels.cuh (the interface), kernels.cu (the GPU kernels
// + the shared solve), reference_cpu.cpp (the independent CPU twins).
// ===========================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

// ===========================================================================
// Debug/visualization constants retyped from ../scripts/make_synthetic.py
// (MUST MATCH its AZIMUTH_MIN_DEG/AZIMUTH_STEP_DEG/AZIMUTH_STEPS/
// DEGENERATE_CHANNEL - main.cu never feeds these INTO the calibration
// algorithm; they exist only to lay out the range-image PPM artifact and to
// name which channel the unobservable_channel gate expects to be flagged,
// per the 01.09 "ground-truth constants retyped independently" precedent).
// Azimuth is recoverable EXACTLY from a point's own (x,y) via atan2(y,x) -
// the beam model's spherical convention makes elevation cancel out of that
// ratio (see kernels.cuh SECTION 6 / make_synthetic.py's beam_direction) -
// so no azimuth column needs to be stored in the CSV at all.
// ===========================================================================
static constexpr float kAzMinDeg = -40.0f;
static constexpr float kAzStepDeg = 1.0f;
static constexpr int   kAzSteps = 81;
static constexpr int32_t kDegenerateChannelExpected = 15;

// ===========================================================================
// GPU-vs-CPU VERIFY tolerances (per stage - "how far can two independent
// float pipelines legitimately drift", not physical bounds; THEORY.md
// "Numerical considerations"). Measured on the reference machine (RTX 2080
// SUPER, sm_75) then margined, per this repo's calibration discipline
// (01.09/02.18's identical practice) - see the comment at each constant for
// the measured value this tolerance was set against.
// ===========================================================================
static constexpr double kTolLogIntensity = 1.0e-4;   // measured 0.0 (identical shared formula, both FP32)
static constexpr double kTolVoxelIdxMismatch = 0;     // exact: deterministic integer arithmetic
static constexpr double kTolSumLog = 3.0e-3;          // measured ~1e-6 (atomicAdd order (GPU) vs sequential double-accum (CPU))
static constexpr double kTolCountMismatch = 0;        // exact: integer counts
static constexpr double kTolAssembleAB = 3.0e-3;      // measured ~3e-6 (same atomic-order story, one level up)
static constexpr double kTolApplyGain = 1.0e-4;       // measured 0.0 (one division on top of already-verified inputs)

// ===========================================================================
// Gate tolerances - each set by MEASURING the actual value on the reference
// machine (RTX 2080 SUPER, sm_75) then adding margin, never AT the measured
// value (01.09/02.18's identical discipline). Measured values are quoted in
// each comment; see README "Expected output" for the full measured table.
// ===========================================================================
static constexpr double kTolGainRecoveryMaxRelErr = 0.09;      // measured 0.058 (worst of 16 observable channels)
static constexpr double kTolConsistencyAfterCv = 0.12;         // measured 0.0435 (avg coeff-of-variation, shared voxels)
static constexpr double kTolConsistencyCollapseFactor = 2.5;   // measured ~5.5x collapse (before/after avg CV)
static constexpr double kTolMultiMaterialDeltaRelErr = 0.08;   // measured 0.044 (all-materials vs wall_near-only solve, common channels)
static constexpr double kTolRangeProfileMaxDev = 0.35;         // [info] only - measured ~0.20

// ===========================================================================
// ScanData - one loaded CSV scan (channel,x,y,z,intensity,surf_id,R_true).
// surf_id/R_true are GROUND TRUTH (kernels.cuh "GROUND TRUTH"): used only
// below by gates/artifacts, never passed into the calibration kernels.
// ===========================================================================
struct ScanData {
    int n = 0;
    std::vector<int32_t> channel;      // [n]
    std::vector<float> xyz;            // [n*3]
    std::vector<float> intensity;      // [n] raw measured
    std::vector<int32_t> surf_id;      // [n] GROUND TRUTH
    std::vector<float> R_true;         // [n] GROUND TRUTH
    bool loaded = false;
};

static ScanData load_scan(const std::string& path)
{
    ScanData d;
    std::ifstream in(path);
    if (!in.is_open()) return d;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> f;
        while (std::getline(ss, cell, ',')) f.push_back(cell);
        if (f.size() != 7) { std::fprintf(stderr, "scan: malformed row (expected 7 fields)\n"); return ScanData{}; }
        d.channel.push_back(std::atoi(f[0].c_str()));
        d.xyz.push_back(std::strtof(f[1].c_str(), nullptr));
        d.xyz.push_back(std::strtof(f[2].c_str(), nullptr));
        d.xyz.push_back(std::strtof(f[3].c_str(), nullptr));
        d.intensity.push_back(std::strtof(f[4].c_str(), nullptr));
        d.surf_id.push_back(std::atoi(f[5].c_str()));
        d.R_true.push_back(std::strtof(f[6].c_str(), nullptr));
        ++d.n;
    }
    d.loaded = d.n > 0;
    return d;
}

static bool load_gains_true(const std::string& path, std::vector<float>& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    out.assign(kNumBeams, 0.0f);
    std::vector<bool> seen(kNumBeams, false);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string a, v;
        if (!std::getline(ss, a, ',') || !std::getline(ss, v, ',')) continue;
        const int ch = std::atoi(a.c_str());
        if (ch < 0 || ch >= kNumBeams) return false;
        out[static_cast<size_t>(ch)] = std::strtof(v.c_str(), nullptr);
        seen[static_cast<size_t>(ch)] = true;
    }
    for (bool s : seen) if (!s) return false;
    return true;
}

// ---------------------------------------------------------------------------
// compute_grid_bounds - the dense voxel grid's extent, derived from a scan's
// OWN point cloud (kernels.cuh SECTION 4: a data-layout parameter, computed
// once, host-side, then handed to BOTH the GPU and CPU pipelines - not part
// of either "algorithm" twin, exactly like 01.09's kW/kH problem geometry).
// A 1-voxel margin on every side keeps flat_voxel_index() from ever
// returning -1 for a point this SAME loop already saw.
// ---------------------------------------------------------------------------
static GridBounds compute_grid_bounds(int n, const std::vector<float>& xyz, float leaf)
{
    int32_t ix_min = INT32_MAX, iy_min = INT32_MAX, iz_min = INT32_MAX;
    int32_t ix_max = INT32_MIN, iy_max = INT32_MIN, iz_max = INT32_MIN;
    for (int i = 0; i < n; ++i) {
        const int32_t ix = voxel_coord(xyz[static_cast<size_t>(i) * 3 + 0], leaf);
        const int32_t iy = voxel_coord(xyz[static_cast<size_t>(i) * 3 + 1], leaf);
        const int32_t iz = voxel_coord(xyz[static_cast<size_t>(i) * 3 + 2], leaf);
        ix_min = std::min(ix_min, ix); ix_max = std::max(ix_max, ix);
        iy_min = std::min(iy_min, iy); iy_max = std::max(iy_max, iy);
        iz_min = std::min(iz_min, iz); iz_max = std::max(iz_max, iz);
    }
    GridBounds g;
    g.ix_min = ix_min - 1; g.iy_min = iy_min - 1; g.iz_min = iz_min - 1;   // 1-voxel margin
    g.nx = (ix_max - ix_min) + 3;
    g.ny = (iy_max - iy_min) + 3;
    g.nz = (iz_max - iz_min) + 3;
    g.leaf = leaf;
    return g;
}

// ===========================================================================
// GPU pipeline runner - allocates device buffers, launches the three GPU
// kernels in sequence, downloads every result the VERIFY stage and the
// gates need. Called once per scan (main.cu calls it for BOTH scans).
// ===========================================================================
struct PipelineOutGpu {
    std::vector<float> log_intensity;      // [n]
    std::vector<int32_t> voxel_idx;        // [n]
    std::vector<float> sum_log;            // [numVoxels*kNumBeams]
    std::vector<int32_t> count;            // [numVoxels*kNumBeams]
    std::vector<int32_t> voxel_family;     // [numVoxels]
    std::vector<float> A;                  // [kNumBeams*kNumBeams]
    std::vector<float> b;                  // [kNumBeams]
    int32_t shared_voxel_count = 0;
    float gpu_ms = 0.0f;
};

static PipelineOutGpu run_gpu_pipeline(int n, const std::vector<int32_t>& h_channel,
                                        const std::vector<float>& h_xyz,
                                        const std::vector<float>& h_intensity,
                                        const GridBounds& grid)
{
    const int numVoxels = grid.nx * grid.ny * grid.nz;
    PipelineOutGpu out;
    out.log_intensity.resize(static_cast<size_t>(n));
    out.voxel_idx.resize(static_cast<size_t>(n));
    out.sum_log.assign(static_cast<size_t>(numVoxels) * kNumBeams, 0.0f);
    out.count.assign(static_cast<size_t>(numVoxels) * kNumBeams, 0);
    out.voxel_family.assign(static_cast<size_t>(numVoxels), kFamilyUnknown);
    out.A.assign(static_cast<size_t>(kNumBeams) * kNumBeams, 0.0f);
    out.b.assign(static_cast<size_t>(kNumBeams), 0.0f);

    int32_t *d_channel = nullptr; float *d_xyz = nullptr, *d_intensity = nullptr;
    float *d_log_intensity = nullptr; int32_t *d_voxel_idx = nullptr;
    float *d_sum_log = nullptr; int32_t *d_count = nullptr, *d_voxel_family = nullptr;
    float *d_A = nullptr, *d_b = nullptr; int32_t *d_shared_voxel_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_channel, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_intensity, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_log_intensity, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_voxel_idx, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sum_log, static_cast<size_t>(numVoxels) * kNumBeams * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count, static_cast<size_t>(numVoxels) * kNumBeams * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_voxel_family, static_cast<size_t>(numVoxels) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(kNumBeams) * kNumBeams * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(kNumBeams) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shared_voxel_count, sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_channel, h_channel.data(), static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensity, h_intensity.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum_log, 0, static_cast<size_t>(numVoxels) * kNumBeams * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_count, 0, static_cast<size_t>(numVoxels) * kNumBeams * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_voxel_family, 0xFF, static_cast<size_t>(numVoxels) * sizeof(int32_t)));   // -1 = kFamilyUnknown
    CUDA_CHECK(cudaMemset(d_A, 0, static_cast<size_t>(kNumBeams) * kNumBeams * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, static_cast<size_t>(kNumBeams) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_shared_voxel_count, 0, sizeof(int32_t)));

    GpuTimer gt; gt.begin();
    launch_point_features(n, d_xyz, d_intensity, grid, d_log_intensity, d_voxel_idx);
    launch_bin_accumulate(n, d_channel, d_xyz, d_log_intensity, d_voxel_idx, numVoxels, d_sum_log, d_count, d_voxel_family);
    launch_assemble_ls(numVoxels, d_sum_log, d_count, d_A, d_b, d_shared_voxel_count);
    out.gpu_ms = gt.end_ms();

    CUDA_CHECK(cudaMemcpy(out.log_intensity.data(), d_log_intensity, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.voxel_idx.data(), d_voxel_idx, static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.sum_log.data(), d_sum_log, static_cast<size_t>(numVoxels) * kNumBeams * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.count.data(), d_count, static_cast<size_t>(numVoxels) * kNumBeams * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.voxel_family.data(), d_voxel_family, static_cast<size_t>(numVoxels) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.A.data(), d_A, static_cast<size_t>(kNumBeams) * kNumBeams * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.b.data(), d_b, static_cast<size_t>(kNumBeams) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&out.shared_voxel_count, d_shared_voxel_count, sizeof(int32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_channel)); CUDA_CHECK(cudaFree(d_xyz)); CUDA_CHECK(cudaFree(d_intensity));
    CUDA_CHECK(cudaFree(d_log_intensity)); CUDA_CHECK(cudaFree(d_voxel_idx));
    CUDA_CHECK(cudaFree(d_sum_log)); CUDA_CHECK(cudaFree(d_count)); CUDA_CHECK(cudaFree(d_voxel_family));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_shared_voxel_count));
    return out;
}

// ===========================================================================
// CPU pipeline runner - the independent oracle (reference_cpu.cpp's three
// twins), double-precision throughout (the "give the oracle better
// precision" asymmetry - reference_cpu.cpp's file header).
// ===========================================================================
struct PipelineOutCpu {
    std::vector<float> log_intensity;
    std::vector<int32_t> voxel_idx;
    std::vector<double> sum_log_d;
    std::vector<int32_t> count;
    std::vector<double> A_d;
    std::vector<double> b_d;
    int32_t shared_voxel_count = 0;
    double cpu_ms = 0.0;
};

static PipelineOutCpu run_cpu_pipeline(int n, const std::vector<int32_t>& h_channel,
                                        const std::vector<float>& h_xyz,
                                        const std::vector<float>& h_intensity,
                                        const GridBounds& grid)
{
    const int numVoxels = grid.nx * grid.ny * grid.nz;
    PipelineOutCpu out;
    out.log_intensity.resize(static_cast<size_t>(n));
    out.voxel_idx.resize(static_cast<size_t>(n));
    out.sum_log_d.assign(static_cast<size_t>(numVoxels) * kNumBeams, 0.0);
    out.count.assign(static_cast<size_t>(numVoxels) * kNumBeams, 0);
    out.A_d.assign(static_cast<size_t>(kNumBeams) * kNumBeams, 0.0);
    out.b_d.assign(static_cast<size_t>(kNumBeams), 0.0);

    CpuTimer ct; ct.begin();
    point_features_cpu(n, h_xyz.data(), h_intensity.data(), grid, out.log_intensity.data(), out.voxel_idx.data());
    bin_accumulate_cpu(n, h_channel.data(), out.log_intensity.data(), out.voxel_idx.data(), numVoxels,
                        out.sum_log_d.data(), out.count.data());
    assemble_ls_cpu(numVoxels, out.sum_log_d.data(), out.count.data(), out.A_d.data(), out.b_d.data(),
                     &out.shared_voxel_count);
    out.cpu_ms = ct.end_ms();
    return out;
}

// ===========================================================================
// Small numeric helpers shared by VERIFY and the gates below.
// ===========================================================================
static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}
static long count_int_mismatches(const std::vector<int32_t>& a, const std::vector<int32_t>& b)
{
    long c = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++c;
    return c;
}
static double max_abs_diff_float_double(const std::vector<float>& a, const std::vector<double>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

// ---------------------------------------------------------------------------
// SharedVoxelObs - one shared voxel's compact per-channel mean-log-intensity
// record, extracted ONCE from the (already GPU-vs-CPU-verified) per-(voxel,
// channel) statistics. Every gate below that needs to RE-SOLVE on a subset
// or resampling of the shared-voxel population (multi_material_robustness,
// the noise_floor bootstrap) works from this compact list rather than
// re-scanning all n points - and every one of those re-solves calls
// kernels.cuh SECTION 5's SHARED channel_ls_accumulate() directly: this is
// gate-side bookkeeping built ON TOP of an already-verified pipeline, not a
// second implementation of the pipeline itself (CLAUDE.md's independence
// ruling applies to the PIPELINE twins, not to derived gate analyses).
// ---------------------------------------------------------------------------
struct SharedVoxelObs {
    int32_t chans[kNumBeams];
    float y[kNumBeams];
    int k = 0;
    int32_t pure_surf = -2;   // GROUND TRUTH: the single surf_id if every point in this voxel shares
                               // one (gate-only bookkeeping), -1 if mixed, -2 if never set
};

static std::vector<SharedVoxelObs> build_shared_voxel_list(int numVoxels,
                                                            const std::vector<double>& sum_log_d,
                                                            const std::vector<int32_t>& count,
                                                            const std::vector<int32_t>& voxel_pure_surf)
{
    std::vector<SharedVoxelObs> out;
    for (int v = 0; v < numVoxels; ++v) {
        SharedVoxelObs o;
        for (int c = 0; c < kNumBeams; ++c) {
            const int32_t cnt = count[static_cast<size_t>(v) * kNumBeams + c];
            if (cnt > 0) {
                o.chans[o.k] = c;
                o.y[o.k] = static_cast<float>(sum_log_d[static_cast<size_t>(v) * kNumBeams + c] / cnt);
                ++o.k;
            }
        }
        if (o.k >= 2) {
            o.pure_surf = voxel_pure_surf[static_cast<size_t>(v)];
            out.push_back(o);
        }
    }
    return out;
}

// build_voxel_pure_surf - GROUND-TRUTH-ONLY bookkeeping (gates/artifacts,
// never the algorithm - kernels.cuh "GROUND TRUTH"): does every point that
// landed in voxel v share ONE surf_id? Used by multi_material_robustness to
// isolate a genuinely single-material shared-voxel subset for comparison.
static std::vector<int32_t> build_voxel_pure_surf(int n, const std::vector<int32_t>& voxel_idx,
                                                   const std::vector<int32_t>& surf_id, int numVoxels)
{
    std::vector<int32_t> pure(static_cast<size_t>(numVoxels), -2);
    for (int i = 0; i < n; ++i) {
        const int32_t v = voxel_idx[static_cast<size_t>(i)];
        if (v < 0 || v >= numVoxels) continue;
        int32_t& p = pure[static_cast<size_t>(v)];
        if (p == -2) p = surf_id[static_cast<size_t>(i)];
        else if (p != surf_id[static_cast<size_t>(i)]) p = -1;
    }
    return pure;
}

// solve_from_list - assemble A/b from a LIST of shared-voxel observations
// (any subset/resampling of build_shared_voxel_list's output) and solve.
static int solve_from_list(const std::vector<SharedVoxelObs>& list, const std::vector<int>& indices,
                            float* out_log_gain, int32_t* out_observable)
{
    std::vector<float> A(static_cast<size_t>(kNumBeams) * kNumBeams, 0.0f), b(static_cast<size_t>(kNumBeams), 0.0f);
    for (int idx : indices) {
        const SharedVoxelObs& o = list[static_cast<size_t>(idx)];
        channel_ls_accumulate(o.chans, o.y, o.k, A.data(), b.data());
    }
    return solve_channel_gains(A.data(), b.data(), out_log_gain, out_observable);
}

// rebin_shared_voxels - GATE-ONLY re-binning of a RESTRICTED point subset at
// a CUSTOM voxel leaf (GATE 3 below): a single-material cohort alone has far
// fewer points than the full multi-material scan, so the primary pipeline's
// kVoxelLeafM (0.5 m) leaves its own channel graph poorly connected - this
// project's build MEASURED that directly (wall_near alone: only 6/16
// channels reach the largest component at 0.5 m; README/THEORY.md report
// the full swept table). Using a LARGER leaf for this one restricted
// re-binning (still calling the SAME shared voxel_coord() formula,
// kernels.cuh SECTION 4) is the honest fix, not a second voxel-grid
// algorithm: a coarser grid trades spatial resolution for the observation
// COUNT a sparser point population needs to stay connected - exactly the
// resolution-vs-connectivity tradeoff THEORY.md's swept comparison
// discusses in the header this function's caller cites. Uses point_features'
// ALREADY-COMPUTED log_intensity (independent of voxel size) - only the
// GROUPING changes, not the per-point forward-model inversion.
static std::vector<SharedVoxelObs> rebin_shared_voxels(const std::vector<int>& point_indices,
                                                        const std::vector<float>& xyz,
                                                        const std::vector<int32_t>& channel,
                                                        const std::vector<float>& log_intensity,
                                                        const std::vector<int32_t>& surf_id,
                                                        float leaf)
{
    struct VoxelAccum { float sum_log[kNumBeams] = { 0.0f }; int32_t count[kNumBeams] = { 0 }; };
    std::map<std::array<int32_t, 3>, VoxelAccum> table;
    std::map<std::array<int32_t, 3>, int32_t> pure;
    for (int i : point_indices) {
        const std::array<int32_t, 3> key = {
            voxel_coord(xyz[static_cast<size_t>(i) * 3 + 0], leaf),
            voxel_coord(xyz[static_cast<size_t>(i) * 3 + 1], leaf),
            voxel_coord(xyz[static_cast<size_t>(i) * 3 + 2], leaf)
        };
        VoxelAccum& acc = table[key];
        const int32_t ch = channel[static_cast<size_t>(i)];
        acc.sum_log[ch] += log_intensity[static_cast<size_t>(i)];
        acc.count[ch] += 1;
        auto it = pure.find(key);
        if (it == pure.end()) pure[key] = surf_id[static_cast<size_t>(i)];
        else if (it->second != surf_id[static_cast<size_t>(i)]) it->second = -1;
    }
    std::vector<SharedVoxelObs> out;
    for (const auto& kv : table) {
        SharedVoxelObs o;
        for (int c = 0; c < kNumBeams; ++c) {
            if (kv.second.count[c] > 0) {
                o.chans[o.k] = c;
                o.y[o.k] = kv.second.sum_log[c] / static_cast<float>(kv.second.count[c]);
                ++o.k;
            }
        }
        if (o.k >= 2) { o.pure_surf = pure.at(kv.first); out.push_back(o); }
    }
    return out;
}

// gain_recovery_max_rel_err - recovered vs ground-truth per-channel gain,
// AFTER gauge alignment (kernels.cuh SECTION 5: only RELATIVE gains are
// observable - both sides are re-centered to zero mean log-gain over the
// OBSERVABLE channels before comparing, THEORY.md "The math"). Returns the
// worst per-channel relative error, or -1.0 if no channel is observable.
static double gain_recovery_max_rel_err(const float* log_gain, const int32_t* observable,
                                         const std::vector<float>& true_gain,
                                         std::vector<double>* per_channel_out = nullptr)
{
    int m = 0; double mean_rec = 0.0, mean_true = 0.0;
    for (int c = 0; c < kNumBeams; ++c) {
        if (!observable[c]) continue;
        mean_rec += log_gain[c];
        mean_true += std::log(static_cast<double>(true_gain[static_cast<size_t>(c)]));
        ++m;
    }
    if (m == 0) return -1.0;
    mean_rec /= m; mean_true /= m;
    double worst = 0.0;
    for (int c = 0; c < kNumBeams; ++c) {
        if (!observable[c]) continue;
        const double rec_aligned = std::exp(static_cast<double>(log_gain[c]) - mean_rec);
        const double true_aligned = std::exp(std::log(static_cast<double>(true_gain[static_cast<size_t>(c)])) - mean_true);
        const double rel = std::fabs(rec_aligned - true_aligned) / true_aligned;
        if (per_channel_out) per_channel_out->push_back(rel);
        if (rel > worst) worst = rel;
    }
    return worst;
}

// compare_solves_over_intersection - a FAIR head-to-head between two
// independently-solved gain vectors (e.g. GATE 3's all-materials vs
// single-material-only solves below), restricted to the channels BOTH
// solves actually flagged observable. Comparing "error over 16 channels"
// against "error over 6 channels" (a real trap this project's build hit
// empirically - see THEORY.md "Numerical considerations") is not
// apples-to-apples: this function gauge-aligns EACH solve's log-gains using
// the MEAN OVER THE INTERSECTION ONLY (not each solve's own, differently-
// sized, observable set), so both worst-case errors below are measured on
// literally the same channel set. Returns the intersection size (0 if the
// two solves share no observable channel at all).
static int compare_solves_over_intersection(const float* log_gain_a, const int32_t* observable_a,
                                             const float* log_gain_b, const int32_t* observable_b,
                                             const std::vector<float>& true_gain,
                                             double& out_err_a, double& out_err_b)
{
    std::vector<int> inter;
    for (int c = 0; c < kNumBeams; ++c) if (observable_a[c] && observable_b[c]) inter.push_back(c);
    if (inter.empty()) { out_err_a = -1.0; out_err_b = -1.0; return 0; }

    double mean_a = 0.0, mean_b = 0.0, mean_true = 0.0;
    for (int c : inter) {
        mean_a += log_gain_a[c];
        mean_b += log_gain_b[c];
        mean_true += std::log(static_cast<double>(true_gain[static_cast<size_t>(c)]));
    }
    const double n = static_cast<double>(inter.size());
    mean_a /= n; mean_b /= n; mean_true /= n;

    double worst_a = 0.0, worst_b = 0.0;
    for (int c : inter) {
        const double true_aligned = std::exp(std::log(static_cast<double>(true_gain[static_cast<size_t>(c)])) - mean_true);
        const double a_aligned = std::exp(static_cast<double>(log_gain_a[c]) - mean_a);
        const double b_aligned = std::exp(static_cast<double>(log_gain_b[c]) - mean_b);
        worst_a = std::max(worst_a, std::fabs(a_aligned - true_aligned) / true_aligned);
        worst_b = std::max(worst_b, std::fabs(b_aligned - true_aligned) / true_aligned);
    }
    out_err_a = worst_a; out_err_b = worst_b;
    return static_cast<int>(inter.size());
}

// ===========================================================================
// Minimal PPM (P6, binary RGB) writer - same "strict, own-generator-only"
// discipline as 01.09's write_pgm, extended to color so misses/background
// can be distinguished from real dim returns (README "Expected output").
// ===========================================================================
static bool write_ppm(const std::string& path, int W, int H, const std::vector<unsigned char>& rgb)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P6\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(out);
}

// ===========================================================================
// gates_metrics.csv writer (same shape as 01.09/02.18's).
// ===========================================================================
struct CsvRow { std::string gate, metric, value, tol, pass; };
static std::string fmt(double v, int prec = 6)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return std::string(buf);
}
static bool write_gates_csv(const std::string& path, const std::vector<CsvRow>& rows)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "gate,metric,value,tolerance,pass\n";
    for (const auto& r : rows) out << r.gate << "," << r.metric << "," << r.value << "," << r.tol << "," << r.pass << "\n";
    return static_cast<bool>(out);
}

// Deterministic xorshift32 (repo convention, CLAUDE.md paragraph 12) - used
// ONLY by the noise_floor bootstrap below (gate-side, not the algorithm).
static inline uint32_t xorshift32(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

// ===========================================================================
// main.
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--data path/to/data/sample]\n", argv[0]); return 2; }
    }

    std::printf("[demo] LiDAR intensity calibration across channels: self-calibrating 16 beam "
               "gains from shared-voxel observations, no reflectance targets (project 02.20)\n");
    print_device_info();
    std::printf("PROBLEM: %d-channel spinning LiDAR, forward model I=g[ch]*R*f(r)*cos(theta)+noise, "
               "self-calibration via a shared-voxel graph least squares, FP32\n", kNumBeams);

    // ---- data --------------------------------------------------------------
    const std::string primary_path = find_data_file(data_dir, argv[0], "scan_primary.csv");
    const std::string degenerate_path = find_data_file(data_dir, argv[0], "scan_degenerate.csv");
    const std::string gains_path = find_data_file(data_dir, argv[0], "gains_true.csv");
    ScanData primary = primary_path.empty() ? ScanData{} : load_scan(primary_path);
    ScanData degenerate = degenerate_path.empty() ? ScanData{} : load_scan(degenerate_path);
    std::vector<float> true_gain;
    const bool gains_ok = !gains_path.empty() && load_gains_true(gains_path, true_gain);
    if (!primary.loaded || !degenerate.loaded || !gains_ok) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("DATA: synthetic structured scene (ground/wall_near/panel/wall_far[/isolated_target]) "
               "ray-cast per channel; primary scan n=%d, degenerate scan n=%d [synthetic, seed 42]\n",
               primary.n, degenerate.n);

    // ---- primary-scan grid + GPU/CPU pipelines ------------------------------
    const GridBounds grid = compute_grid_bounds(primary.n, primary.xyz, kVoxelLeafM);
    const int numVoxels = grid.nx * grid.ny * grid.nz;
    std::printf("[info] voxel grid: %dx%dx%d = %d voxels, leaf=%.2f m\n", grid.nx, grid.ny, grid.nz, numVoxels,
               static_cast<double>(grid.leaf));

    PipelineOutGpu gpu = run_gpu_pipeline(primary.n, primary.channel, primary.xyz, primary.intensity, grid);
    PipelineOutCpu cpu = run_cpu_pipeline(primary.n, primary.channel, primary.xyz, primary.intensity, grid);
    std::printf("[time] GPU pipeline (3 kernel launches, primary scan): %.3f ms | CPU oracle (independent, "
               "double precision): %.1f ms\n", static_cast<double>(gpu.gpu_ms), cpu.cpu_ms);

    // ---- VERIFY: GPU vs CPU, every stage ------------------------------------
    bool verify_pass = true;
    const double d_feat = max_abs_diff(gpu.log_intensity, cpu.log_intensity);
    const long mism_vox = count_int_mismatches(gpu.voxel_idx, cpu.voxel_idx);
    const double d_sumlog = max_abs_diff_float_double(gpu.sum_log, cpu.sum_log_d);
    const long mism_count = count_int_mismatches(gpu.count, cpu.count);
    const double d_ab_A = max_abs_diff_float_double(gpu.A, cpu.A_d);
    const double d_ab_b = max_abs_diff_float_double(gpu.b, cpu.b_d);
    const double d_ab = std::max(d_ab_A, d_ab_b);

    if (d_feat > kTolLogIntensity) verify_pass = false;
    if (mism_vox > kTolVoxelIdxMismatch) verify_pass = false;
    if (d_sumlog > kTolSumLog) verify_pass = false;
    if (mism_count > kTolCountMismatch) verify_pass = false;
    if (d_ab > kTolAssembleAB) verify_pass = false;
    if (gpu.shared_voxel_count != cpu.shared_voxel_count) verify_pass = false;

    std::printf("[info] verify(features): max|gpu-cpu| log_intensity = %.6f (tol %.4f)\n", d_feat, kTolLogIntensity);
    std::printf("[info] verify(binning): voxel_idx mismatches = %ld/%d (tol %.0f)\n", mism_vox, primary.n, kTolVoxelIdxMismatch);
    std::printf("[info] verify(accumulate): max|gpu-cpu| sum_log = %.6f (tol %.4f) | count mismatches = %ld (tol %.0f)\n",
               d_sumlog, kTolSumLog, mism_count, kTolCountMismatch);
    std::printf("[info] verify(assemble): max|gpu-cpu| A,b = %.6f (tol %.4f) | shared_voxel_count gpu=%d cpu=%d\n",
               d_ab, kTolAssembleAB, gpu.shared_voxel_count, cpu.shared_voxel_count);
    std::printf("VERIFY: %s (GPU matches CPU reference within documented per-stage tolerance: features, "
               "binning, accumulate, assemble)\n", verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU pipeline disagreement - fix before trusting the calibration)\n");
        return 1;
    }

    // ======================= SHARED SOLVE (called ONCE - kernels.cu SECTION
    //      8's 01.09-SECTION-5 precedent) - from the higher-precision CPU
    //      assembly, cast to float ==========================================
    std::vector<float> A_solve(static_cast<size_t>(kNumBeams) * kNumBeams), b_solve(static_cast<size_t>(kNumBeams));
    for (size_t i = 0; i < A_solve.size(); ++i) A_solve[i] = static_cast<float>(cpu.A_d[i]);
    for (size_t i = 0; i < b_solve.size(); ++i) b_solve[i] = static_cast<float>(cpu.b_d[i]);
    float log_gain[kNumBeams]; int32_t observable[kNumBeams];
    const int num_observable = solve_channel_gains(A_solve.data(), b_solve.data(), log_gain, observable);

    std::vector<float> gain(kNumBeams, 1.0f);   // unobservable channels: gain 1.0 = "no correction applied", never a guess
    int missing_list_count = 0;
    for (int c = 0; c < kNumBeams; ++c) if (observable[c]) gain[static_cast<size_t>(c)] = std::exp(log_gain[c]); else ++missing_list_count;
    std::printf("[info] channel graph (primary scan): %d/%d channels observable, %d shared voxels (>= 2 "
               "channels each) - observability is graph connectivity (THEORY.md \"The math\")\n",
               num_observable, kNumBeams, gpu.shared_voxel_count);

    // ---- apply_gain: GPU vs CPU (a small extra VERIFY, on the recovered
    //      correction itself), operating on the RANGE-AND-INCIDENCE-
    //      COMPENSATED intensity exp(log_intensity) (kernels.cuh SECTION 6 -
    //      main.cu's file header explains why THIS quantity, not raw
    //      intensity, is what "apply the channel gain" divides) -----------
    std::vector<float> range_compensated(static_cast<size_t>(primary.n));
    for (int i = 0; i < primary.n; ++i) range_compensated[static_cast<size_t>(i)] = std::exp(cpu.log_intensity[static_cast<size_t>(i)]);

    float *d_rc = nullptr, *d_gain = nullptr, *d_corrected = nullptr; int32_t *d_ch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rc, static_cast<size_t>(primary.n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gain, static_cast<size_t>(kNumBeams) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corrected, static_cast<size_t>(primary.n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ch, static_cast<size_t>(primary.n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_rc, range_compensated.data(), static_cast<size_t>(primary.n) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gain, gain.data(), static_cast<size_t>(kNumBeams) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ch, primary.channel.data(), static_cast<size_t>(primary.n) * sizeof(int32_t), cudaMemcpyHostToDevice));
    launch_apply_gain(primary.n, d_rc, d_ch, d_gain, d_corrected);
    std::vector<float> corrected_gpu(static_cast<size_t>(primary.n));
    CUDA_CHECK(cudaMemcpy(corrected_gpu.data(), d_corrected, static_cast<size_t>(primary.n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rc)); CUDA_CHECK(cudaFree(d_gain)); CUDA_CHECK(cudaFree(d_corrected)); CUDA_CHECK(cudaFree(d_ch));

    std::vector<float> corrected_cpu(static_cast<size_t>(primary.n));
    apply_gain_cpu(primary.n, range_compensated.data(), primary.channel.data(), gain.data(), corrected_cpu.data());
    const double d_applygain = max_abs_diff(corrected_gpu, corrected_cpu);
    if (d_applygain > kTolApplyGain) {
        std::printf("[info] verify(apply_gain): max|gpu-cpu| = %.6f (tol %.4f) - FAIL\n", d_applygain, kTolApplyGain);
        std::printf("RESULT: FAIL (apply_gain GPU/CPU disagreement)\n");
        return 1;
    }
    std::printf("[info] verify(apply_gain): max|gpu-cpu| = %.6f (tol %.4f) - PASS\n", d_applygain, kTolApplyGain);

    std::vector<CsvRow> csv;

    // ======================= GATE 1: gain_recovery ============================
    std::vector<double> per_channel_err;
    const double gain_worst = gain_recovery_max_rel_err(log_gain, observable, true_gain, &per_channel_err);
    const bool gate_gain = (gain_worst >= 0.0) && (gain_worst <= kTolGainRecoveryMaxRelErr);
    std::printf("GATE gain_recovery: %s\n", gate_gain ? "PASS" : "FAIL");
    std::printf("[info] gain_recovery: worst per-channel relative error (gauge-aligned) = %.4f (tol %.2f) "
               "over %d observable channels\n", gain_worst, kTolGainRecoveryMaxRelErr, num_observable);
    for (int c = 0; c < kNumBeams; ++c) {
        if (!observable[c]) continue;
        std::printf("[info]   ch%02d: recovered gain=%.4f true gain=%.4f\n", c,
                   static_cast<double>(gain[static_cast<size_t>(c)]), static_cast<double>(true_gain[static_cast<size_t>(c)]));
    }
    csv.push_back({ "gain_recovery", "worst_rel_err", fmt(gain_worst, 4), fmt(kTolGainRecoveryMaxRelErr, 2), gate_gain ? "PASS" : "FAIL" });

    // ======================= GATE 2: consistency_improvement ==================
    // The reason this project exists: per-voxel cross-channel spread BEFORE
    // vs AFTER calibration (main.cu file header derives the exp(log) domain
    // this operates in) - averaged over every shared voxel.
    std::vector<int32_t> pure_surf_primary = build_voxel_pure_surf(primary.n, cpu.voxel_idx, primary.surf_id, numVoxels);
    std::vector<SharedVoxelObs> shared_list = build_shared_voxel_list(numVoxels, cpu.sum_log_d, cpu.count, pure_surf_primary);

    double sum_cv_before = 0.0, sum_cv_after = 0.0;
    for (const auto& o : shared_list) {
        double vals_before[kNumBeams], vals_after[kNumBeams];
        double sum_b = 0.0, sum_a = 0.0;
        for (int i = 0; i < o.k; ++i) {
            vals_before[i] = std::exp(static_cast<double>(o.y[i]));
            vals_after[i] = vals_before[i] / static_cast<double>(gain[static_cast<size_t>(o.chans[i])]);
            sum_b += vals_before[i]; sum_a += vals_after[i];
        }
        const double mean_b = sum_b / o.k, mean_a = sum_a / o.k;
        double var_b = 0.0, var_a = 0.0;
        for (int i = 0; i < o.k; ++i) {
            var_b += (vals_before[i] - mean_b) * (vals_before[i] - mean_b);
            var_a += (vals_after[i] - mean_a) * (vals_after[i] - mean_a);
        }
        var_b /= o.k; var_a /= o.k;
        sum_cv_before += std::sqrt(var_b) / mean_b;
        sum_cv_after += std::sqrt(var_a) / mean_a;
    }
    const double avg_cv_before = shared_list.empty() ? 0.0 : sum_cv_before / static_cast<double>(shared_list.size());
    const double avg_cv_after = shared_list.empty() ? 0.0 : sum_cv_after / static_cast<double>(shared_list.size());
    const double collapse_factor = avg_cv_after > 1e-9 ? avg_cv_before / avg_cv_after : 1.0e9;
    const bool gate_consistency = (avg_cv_after <= kTolConsistencyAfterCv) && (collapse_factor >= kTolConsistencyCollapseFactor);

    // The 02.18/LIOR demand signal, closed: threshold the wall_far cohort's
    // range-compensated intensity (README "System context" names 02.18's
    // kLiorIntensityThresh=0.05 as the conceptual sibling this recreates in
    // this project's own units) BEFORE vs AFTER gain correction and count
    // classification flips - the exact failure mode 02.18's
    // intensity_dependence [info] line quantified as a -5.9pp recall drop.
    std::vector<double> lior_before, lior_after;
    for (int i = 0; i < primary.n; ++i) {
        if (primary.surf_id[static_cast<size_t>(i)] != kGtWallFar) continue;
        const double before = std::exp(static_cast<double>(cpu.log_intensity[static_cast<size_t>(i)]));
        const double after = before / static_cast<double>(gain[static_cast<size_t>(primary.channel[static_cast<size_t>(i)])]);
        lior_before.push_back(before); lior_after.push_back(after);
    }
    double lior_thresh = 0.0;
    if (!lior_after.empty()) {
        std::vector<double> sorted_after = lior_after;
        std::sort(sorted_after.begin(), sorted_after.end());
        lior_thresh = sorted_after[sorted_after.size() / 2];   // median of the AFTER (calibrated) distribution
    }
    int lior_flips = 0;
    for (size_t i = 0; i < lior_before.size(); ++i)
        if ((lior_before[i] >= lior_thresh) != (lior_after[i] >= lior_thresh)) ++lior_flips;

    std::printf("GATE consistency_improvement: %s\n", gate_consistency ? "PASS" : "FAIL");
    std::printf("[info] consistency_improvement: avg cross-channel coeff-of-variation over %zu shared voxels: "
               "BEFORE=%.4f AFTER=%.4f (tol <= %.2f) | collapse factor=%.2fx (tol >= %.1fx)\n",
               shared_list.size(), avg_cv_before, avg_cv_after, kTolConsistencyAfterCv, collapse_factor, kTolConsistencyCollapseFactor);
    std::printf("[info] LIOR-style decision-flip demo (wall_far cohort, n=%zu, threshold=%.4f, the project "
               "02.18 demand signal): %d/%zu points flip keep/reject classification after calibration "
               "(02.18's own LIOR measured a -5.9pp recall drop from this exact miscalibration failure mode)\n",
               lior_before.size(), lior_thresh, lior_flips, lior_before.size());
    csv.push_back({ "consistency_improvement", "avg_cv_before", fmt(avg_cv_before, 4), "n/a (reported only)", "n/a" });
    csv.push_back({ "consistency_improvement", "avg_cv_after", fmt(avg_cv_after, 4), fmt(kTolConsistencyAfterCv, 2), gate_consistency ? "PASS" : "FAIL" });
    csv.push_back({ "consistency_improvement", "collapse_factor", fmt(collapse_factor, 2), fmt(kTolConsistencyCollapseFactor, 1), gate_consistency ? "PASS" : "FAIL" });
    csv.push_back({ "consistency_improvement", "lior_decision_flips", std::to_string(lior_flips), "n/a (reported only)", "n/a" });

    // ======================= GATE 3: multi_material_robustness ================
    // Does the multi-material shared-voxel population (ground+wall_near+
    // panel+wall_far, i.e. the ACTUAL 0.5 m-leaf solve above) recover gains
    // materially WORSE than a solve restricted to a single, pure material
    // (wall_near-only)? If the shared-voxel currency truly cancels R, it
    // must not.
    //
    // A single material has far fewer points than the full scan, so
    // wall_near-ONLY at the primary 0.5 m leaf leaves its own channel graph
    // poorly connected (measured while building this project: only 6/16
    // channels reach the largest component - README/THEORY.md's swept
    // table). rebin_shared_voxels() re-bins the SAME wall_near points at a
    // LARGER, measured-then-margined leaf (kSingleMaterialLeafM) chosen as
    // the smallest leaf at which wall_near ALONE reconnects all 16 channels
    // - a resolution-for-connectivity tradeoff honestly documented, not a
    // second voxel-grid algorithm (rebin_shared_voxels calls the SAME
    // shared voxel_coord() formula, kernels.cuh SECTION 4).
    constexpr float kSingleMaterialLeafM = 1.3f;   // measured: 0.5/0.7/0.9/1.1 m give only 6/10/12/14 of 16
    std::vector<int> wall_near_point_idx;
    for (int i = 0; i < primary.n; ++i) if (primary.surf_id[static_cast<size_t>(i)] == kGtWallNear) wall_near_point_idx.push_back(i);
    std::vector<SharedVoxelObs> single_material_list = rebin_shared_voxels(
        wall_near_point_idx, primary.xyz, primary.channel, cpu.log_intensity, primary.surf_id, kSingleMaterialLeafM);
    std::vector<int> idx_single_all(single_material_list.size());
    for (size_t i = 0; i < single_material_list.size(); ++i) idx_single_all[i] = static_cast<int>(i);

    std::vector<int> idx_all(shared_list.size());
    for (size_t i = 0; i < shared_list.size(); ++i) idx_all[i] = static_cast<int>(i);

    float log_gain_multi[kNumBeams], log_gain_single[kNumBeams];
    int32_t obs_multi[kNumBeams], obs_single[kNumBeams];
    solve_from_list(shared_list, idx_all, log_gain_multi, obs_multi);           // == the primary solve above (sanity)
    const int m_single = solve_from_list(single_material_list, idx_single_all, log_gain_single, obs_single);
    double err_multi = -1.0, err_single = -1.0;
    const int inter_size = compare_solves_over_intersection(log_gain_multi, obs_multi, log_gain_single, obs_single,
                                                              true_gain, err_multi, err_single);
    const double mm_delta = err_multi - err_single;   // positive = multi-material solve is WORSE, over the SAME channels
    const bool gate_multi_material = (inter_size >= kNumBeams / 2) && (mm_delta <= kTolMultiMaterialDeltaRelErr);
    std::printf("GATE multi_material_robustness: %s\n", gate_multi_material ? "PASS" : "FAIL");
    std::printf("[info] multi_material_robustness: comparing on the %d channels BOTH solves flag observable - "
               "ALL-materials solve (leaf=%.2f m, n=%zu voxels, %d/%d observable) worst rel err = %.4f | "
               "wall_near-ONLY solve (leaf=%.2f m, n=%zu voxels, %d/%d observable) worst rel err = %.4f | "
               "delta = %.4f (tol <= %.2f)\n",
               inter_size, static_cast<double>(grid.leaf), idx_all.size(), num_observable, kNumBeams, err_multi,
               static_cast<double>(kSingleMaterialLeafM), single_material_list.size(), m_single, kNumBeams,
               err_single, mm_delta, kTolMultiMaterialDeltaRelErr);
    csv.push_back({ "multi_material_robustness", "err_all_materials_on_intersection", fmt(err_multi, 4), "n/a (reported only)", "n/a" });
    csv.push_back({ "multi_material_robustness", "err_single_material_on_intersection", fmt(err_single, 4), "n/a (reported only)", "n/a" });
    csv.push_back({ "multi_material_robustness", "intersection_size", std::to_string(inter_size), std::to_string(kNumBeams / 2) + "+", gate_multi_material ? "PASS" : "FAIL" });
    csv.push_back({ "multi_material_robustness", "delta", fmt(mm_delta, 4), fmt(kTolMultiMaterialDeltaRelErr, 2), gate_multi_material ? "PASS" : "FAIL" });

    // ======================= GATE 4: unobservable_channel =====================
    // Run the SAME pipeline on the DEGENERATE scan (GPU path - already
    // proven equivalent to the CPU oracle above on the primary scan; no
    // kernel branches on which scan it is fed). Assert the retargeted
    // channel is flagged UNOBSERVABLE, never assigned a hallucinated gain.
    const GridBounds grid_deg = compute_grid_bounds(degenerate.n, degenerate.xyz, kVoxelLeafM);
    PipelineOutGpu gpu_deg = run_gpu_pipeline(degenerate.n, degenerate.channel, degenerate.xyz, degenerate.intensity, grid_deg);
    float log_gain_deg[kNumBeams]; int32_t observable_deg[kNumBeams];
    const int num_observable_deg = solve_channel_gains(gpu_deg.A.data(), gpu_deg.b.data(), log_gain_deg, observable_deg);
    const bool degenerate_flagged = (observable_deg[kDegenerateChannelExpected] == 0);
    int other_observable_count = 0;
    for (int c = 0; c < kNumBeams; ++c) if (c != kDegenerateChannelExpected && observable_deg[c]) ++other_observable_count;
    const bool gate_unobservable = degenerate_flagged && (other_observable_count == kNumBeams - 1);
    std::printf("GATE unobservable_channel: %s\n", gate_unobservable ? "PASS" : "FAIL");
    std::printf("[info] unobservable_channel: degenerate scan, channel %d retargeted off-scene - flagged "
               "unobservable = %s (expected true) | other %d/%d channels observable (expected %d/%d) | "
               "%d shared voxels\n", kDegenerateChannelExpected, degenerate_flagged ? "true" : "false",
               other_observable_count, kNumBeams - 1, kNumBeams - 1, kNumBeams - 1, gpu_deg.shared_voxel_count);
    csv.push_back({ "unobservable_channel", "flagged_channel_15_unobservable", degenerate_flagged ? "1" : "0", "1", gate_unobservable ? "PASS" : "FAIL" });
    csv.push_back({ "unobservable_channel", "other_channels_observable", std::to_string(other_observable_count), std::to_string(kNumBeams - 1), gate_unobservable ? "PASS" : "FAIL" });
    (void)num_observable_deg;

    // ======================= [info] range_profile (bonus milestone) ==========
    // Nonparametric f(r) shape recovery: divide out the RECOVERED gain and
    // the KNOWN (ground-truth, grading-only - kernels.cuh "GROUND TRUTH")
    // reflectivity from every point's raw intensity, bin by measured range,
    // normalize each curve to its nearest-to-kRangePlateauM bin, and compare
    // shapes - the 01.09 radial_fit gate's parametric-vs-nonparametric
    // duality, recast for range instead of radius.
    struct RangeBin { double sum_recovered = 0.0, sum_true = 0.0; int count = 0; double r_lo, r_hi; };
    constexpr int kRangeBins = 12;
    constexpr double kRangeBinLoM = 4.0, kRangeBinHiM = 40.0;
    std::vector<RangeBin> range_bins(kRangeBins);
    for (int i = 0; i < kRangeBins; ++i) {
        range_bins[static_cast<size_t>(i)].r_lo = kRangeBinLoM + i * (kRangeBinHiM - kRangeBinLoM) / kRangeBins;
        range_bins[static_cast<size_t>(i)].r_hi = kRangeBinLoM + (i + 1) * (kRangeBinHiM - kRangeBinLoM) / kRangeBins;
    }
    for (int i = 0; i < primary.n; ++i) {
        const float* p = &primary.xyz[static_cast<size_t>(i) * 3];
        const double r = std::sqrt(static_cast<double>(p[0]) * p[0] + static_cast<double>(p[1]) * p[1] + static_cast<double>(p[2]) * p[2]);
        for (auto& rb : range_bins) {
            if (r >= rb.r_lo && r < rb.r_hi) {
                const double g = static_cast<double>(gain[static_cast<size_t>(primary.channel[static_cast<size_t>(i)])]);
                const double R = static_cast<double>(primary.R_true[static_cast<size_t>(i)]);
                if (g > 1e-9 && R > 1e-9) {
                    rb.sum_recovered += static_cast<double>(primary.intensity[static_cast<size_t>(i)]) / (g * R);
                    rb.sum_true += range_falloff(static_cast<float>(r));
                    rb.count++;
                }
                break;
            }
        }
    }
    int ref_bin = -1;
    for (int i = 0; i < kRangeBins; ++i) if (range_bins[static_cast<size_t>(i)].count > 0) { ref_bin = i; break; }
    double range_profile_max_dev = -1.0;
    std::vector<double> recovered_norm(kRangeBins, 0.0), true_norm(kRangeBins, 0.0);
    if (ref_bin >= 0) {
        const double ref_recovered = range_bins[static_cast<size_t>(ref_bin)].sum_recovered / range_bins[static_cast<size_t>(ref_bin)].count;
        const double ref_true = range_bins[static_cast<size_t>(ref_bin)].sum_true / range_bins[static_cast<size_t>(ref_bin)].count;
        range_profile_max_dev = 0.0;
        for (int i = 0; i < kRangeBins; ++i) {
            if (range_bins[static_cast<size_t>(i)].count == 0) continue;
            recovered_norm[static_cast<size_t>(i)] = (range_bins[static_cast<size_t>(i)].sum_recovered / range_bins[static_cast<size_t>(i)].count) / ref_recovered;
            true_norm[static_cast<size_t>(i)] = (range_bins[static_cast<size_t>(i)].sum_true / range_bins[static_cast<size_t>(i)].count) / ref_true;
            range_profile_max_dev = std::max(range_profile_max_dev, std::fabs(recovered_norm[static_cast<size_t>(i)] - true_norm[static_cast<size_t>(i)]));
        }
    }
    int range_bins_populated = 0;
    for (const auto& rb : range_bins) if (rb.count > 0) ++range_bins_populated;
    std::printf("[info] range_profile (bonus milestone, NOT gated): nonparametric f(r) shape recovery vs "
               "the generator's true curve, normalized to the nearest-to-plateau populated bin - max "
               "deviation = %.4f (informational bound %.2f) over %d/%d populated range bins in [%.0f,%.0f) m\n",
               range_profile_max_dev, kTolRangeProfileMaxDev, range_bins_populated, kRangeBins, kRangeBinLoM, kRangeBinHiM);

    // ======================= [info] noise_floor (bootstrap precision) ========
    // How precise is the gain estimate GIVEN this one committed, noisy
    // scan? Resample the shared-voxel population WITH replacement (xorshift32,
    // seed 42) K times, re-solve each time, report the per-channel std of the
    // resulting recovered gains - an honest bound on estimate precision, not
    // a correctness check (CLAUDE.md paragraph 12: bootstrap std, not a gate).
    constexpr int kBootstrapK = 8;
    std::vector<std::vector<float>> boot_gains(kNumBeams);
    uint32_t rng_state = 42u;
    for (int rep = 0; rep < kBootstrapK; ++rep) {
        std::vector<int> resample(shared_list.size());
        for (size_t i = 0; i < shared_list.size(); ++i)
            resample[i] = static_cast<int>(xorshift32(rng_state) % shared_list.size());
        float lg[kNumBeams]; int32_t ob[kNumBeams];
        solve_from_list(shared_list, resample, lg, ob);
        for (int c = 0; c < kNumBeams; ++c) if (ob[c]) boot_gains[static_cast<size_t>(c)].push_back(std::exp(lg[c]));
    }
    double mean_boot_std = 0.0; int channels_with_boot = 0;
    for (int c = 0; c < kNumBeams; ++c) {
        const auto& v = boot_gains[static_cast<size_t>(c)];
        if (v.size() < 2) continue;
        double mean = 0.0; for (float g : v) mean += g; mean /= v.size();
        double var = 0.0; for (float g : v) var += (g - mean) * (g - mean); var /= v.size();
        mean_boot_std += std::sqrt(var);
        ++channels_with_boot;
    }
    if (channels_with_boot > 0) mean_boot_std /= channels_with_boot;
    std::printf("[info] noise_floor (bootstrap precision, NOT gated): mean per-channel gain std across %d "
               "voxel-resampled re-solves = %.4f (honesty about estimate precision given this one scan's "
               "noise realization, THEORY.md \"How we verify correctness\")\n", kBootstrapK, mean_boot_std);

    // ======================= ARTIFACTS =========================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();

    // intensity_hist_{before,after}.csv - the visual: per-channel histograms
    // of the range-and-incidence-compensated intensity (kernels.cuh SECTION
    // 3's exp(log_intensity)), restricted to the three reflective surfaces
    // (ground excluded - its grazing-incidence returns are near the noise
    // floor, kernels.cuh SECTION 3's kIntensityFloor guard, and not part of
    // the shared-voxel calibration graph anyway).
    {
        constexpr int kHistBins = 20;
        constexpr double kHistLo = 0.0, kHistHi = 1.3;
        std::vector<std::vector<long>> hist_before(kNumBeams, std::vector<long>(kHistBins, 0));
        std::vector<std::vector<long>> hist_after(kNumBeams, std::vector<long>(kHistBins, 0));
        for (int i = 0; i < primary.n; ++i) {
            if (primary.surf_id[static_cast<size_t>(i)] == kGtGround) continue;
            const int32_t ch = primary.channel[static_cast<size_t>(i)];
            const double before = std::exp(static_cast<double>(cpu.log_intensity[static_cast<size_t>(i)]));
            const double after = before / static_cast<double>(gain[static_cast<size_t>(ch)]);
            int bb = static_cast<int>((before - kHistLo) / (kHistHi - kHistLo) * kHistBins);
            int ba = static_cast<int>((after - kHistLo) / (kHistHi - kHistLo) * kHistBins);
            bb = std::max(0, std::min(kHistBins - 1, bb));
            ba = std::max(0, std::min(kHistBins - 1, ba));
            hist_before[static_cast<size_t>(ch)][static_cast<size_t>(bb)]++;
            hist_after[static_cast<size_t>(ch)][static_cast<size_t>(ba)]++;
        }
        auto write_hist = [&](const std::string& path, const std::vector<std::vector<long>>& hist) -> bool {
            std::ofstream f(path);
            if (!f.is_open()) return false;
            f << "bin_center";
            for (int c = 0; c < kNumBeams; ++c) f << ",ch" << (c < 10 ? "0" : "") << c;
            f << "\n";
            for (int bidx = 0; bidx < kHistBins; ++bidx) {
                const double center = kHistLo + (bidx + 0.5) * (kHistHi - kHistLo) / kHistBins;
                f << fmt(center, 4);
                for (int c = 0; c < kNumBeams; ++c) f << "," << hist[static_cast<size_t>(c)][static_cast<size_t>(bidx)];
                f << "\n";
            }
            return static_cast<bool>(f);
        };
        artifact_ok = artifact_ok && write_hist(out_dir + "/intensity_hist_before.csv", hist_before);
        artifact_ok = artifact_ok && write_hist(out_dir + "/intensity_hist_after.csv", hist_after);
    }

    // range_profile.csv
    {
        std::ofstream f(out_dir + "/range_profile.csv");
        if (f.is_open()) {
            f << "r_lo_m,r_hi_m,count,recovered_normalized,true_normalized\n";
            for (int i = 0; i < kRangeBins; ++i) {
                const auto& rb = range_bins[static_cast<size_t>(i)];
                f << fmt(rb.r_lo, 2) << "," << fmt(rb.r_hi, 2) << "," << rb.count << ",";
                if (rb.count > 0) f << fmt(recovered_norm[static_cast<size_t>(i)], 5) << "," << fmt(true_norm[static_cast<size_t>(i)], 5);
                else f << "," ;
                f << "\n";
            }
        } else {
            artifact_ok = false;
        }
    }

    // range_image_{before,after}.ppm - rows=channel, cols=azimuth (recovered
    // EXACTLY from xyz via atan2(y,x) - main.cu file header); missing
    // (channel,azimuth) cells (no return) get a distinct background color.
    {
        auto write_range_image = [&](const std::string& path, bool after_calibration) -> bool {
            std::vector<float> img(static_cast<size_t>(kNumBeams) * kAzSteps, -1.0f);   // -1 = no return
            for (int i = 0; i < primary.n; ++i) {
                const int32_t ch = primary.channel[static_cast<size_t>(i)];
                const float* p = &primary.xyz[static_cast<size_t>(i) * 3];
                const double az_deg = std::atan2(static_cast<double>(p[1]), static_cast<double>(p[0])) * 180.0 / 3.14159265358979323846;
                int col = static_cast<int>(std::lround((az_deg - kAzMinDeg) / kAzStepDeg));
                if (col < 0 || col >= kAzSteps) continue;
                double v = std::exp(static_cast<double>(cpu.log_intensity[static_cast<size_t>(i)]));
                if (after_calibration) v /= static_cast<double>(gain[static_cast<size_t>(ch)]);
                img[static_cast<size_t>(ch) * kAzSteps + col] = static_cast<float>(v);
            }
            float lo = 1e30f, hi = -1e30f;
            for (float v : img) if (v >= 0.0f) { lo = std::min(lo, v); hi = std::max(hi, v); }
            const float range = (hi - lo) > 1e-6f ? (hi - lo) : 1.0f;
            constexpr int colScale = 6, rowScale = 18;
            const int W = kAzSteps * colScale, H = kNumBeams * rowScale;
            std::vector<unsigned char> rgb(static_cast<size_t>(W) * H * 3, 0);
            for (int row = 0; row < kNumBeams; ++row) {
                for (int col = 0; col < kAzSteps; ++col) {
                    const float v = img[static_cast<size_t>(row) * kAzSteps + col];
                    unsigned char r, g, b;
                    if (v < 0.0f) { r = 0; g = 0; b = 80; }   // background: no return
                    else {
                        float t = (v - lo) / range;
                        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                        const unsigned char grey = static_cast<unsigned char>(40.0f + t * 215.0f);
                        r = g = b = grey;
                    }
                    for (int ry = 0; ry < rowScale; ++ry) {
                        for (int rx = 0; rx < colScale; ++rx) {
                            const int px = col * colScale + rx, py = row * rowScale + ry;
                            const size_t off = (static_cast<size_t>(py) * W + px) * 3;
                            rgb[off + 0] = r; rgb[off + 1] = g; rgb[off + 2] = b;
                        }
                    }
                }
            }
            return write_ppm(path, W, H, rgb);
        };
        artifact_ok = artifact_ok && write_range_image(out_dir + "/range_image_before.ppm", false);
        artifact_ok = artifact_ok && write_range_image(out_dir + "/range_image_after.ppm", true);
    }

    artifact_ok = artifact_ok && write_gates_csv(out_dir + "/gates_metrics.csv", csv);

    if (artifact_ok) {
        std::printf("ARTIFACT: wrote demo/out/{intensity_hist_before.csv, intensity_hist_after.csv, "
                   "range_profile.csv, range_image_before.ppm, range_image_after.ppm, gates_metrics.csv}\n");
    } else {
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");
    }

    // ======================= RESULT =============================================
    const bool all_gates = gate_gain && gate_consistency && gate_multi_material && gate_unobservable;
    const bool overall = verify_pass && all_gates && artifact_ok;
    if (overall) {
        std::printf("RESULT: PASS (VERIFY + all 4 gates passed: gain_recovery, consistency_improvement, "
                   "multi_material_robustness, unobservable_channel)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
