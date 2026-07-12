// ===========================================================================
// main.cu — entry point for project 02.14
//           Moving-object segmentation from sequential scans (online MOS)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed sample (data/sample/scans.csv + poses.csv): a
//      5-scan window (current + M=4 previous), each point already carrying
//      its native (ring, az_bin, range) plus ground truth used ONLY below.
//   2. VERIFY STAGE (the CLAUDE.md §5 GPU-vs-CPU gate, in FOUR parts because
//      this project has four independently-verifiable stages):
//        (a) CURRENT-SCAN ORGANIZE exact (no trig involved — a pure
//            nearest-wins selection over stored (ring,az_bin,range)).
//        (b) REPROJECTION close (each of the 4 previous scans; involves
//            trig, so GPU vs CPU is compared with a small, documented
//            tolerance rather than demanded bit-exact).
//        (c) RESIDUAL FUSION exact (fed the SAME already-verified range
//            images on both paths — 02.13's "verify against a freshly
//            re-uploaded copy of the already-verified data" pattern).
//        (d) CCL exact (edge set + union-find, 02.12's identical two-part
//            verify for its own union-find stage).
//   3. SECONDARY ANALYSIS, host-only (02.13's identical policy: a code path
//      already proven bit-exact against the GPU is not re-verified a
//      second time): the window-size study (M=1,2,4) and the
//      pre-CCL-vs-post-CCL disocclusion comparison both reuse the CPU
//      twins the verify stage just certified.
//   4. GATES: mover_detection, sign_semantics, static_precision,
//      disocclusion_mitigation, timing (all pass/fail) plus
//      temporal_boundary and window_size ([info]-only measurements).
//   5. ARTIFACTS: a residual-image PGM (movers glow), a label-vs-truth PPM
//      (a confusion map), a disocclusion-band PPM, a per-cohort metrics CSV,
//      and a gates_metrics.csv summary — all under demo/out/.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "VERIFY:", "GATE:", "ARTIFACT:", "RESULT:" — NEVER carrying a measured
// number (percentages, timings, counts derived from running the algorithm),
// only fixed verdicts and fixed descriptive text (08.01/02.13's identical
// discipline). Every measured number lives on an "[info]"/"[time]" line,
// deliberately unchecked by demo/run_demo.*. Change a stable line -> update
// demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Gate thresholds — MEASURED-THEN-MARGINED against this project's committed
// sample (README "Expected output" states the actual numbers from the run
// that produced demo/expected_output.txt; CLAUDE.md §12 "wide margins").
// ---------------------------------------------------------------------------
static constexpr float kReprojectionToleranceM   = 5.0e-3f;  // GPU-vs-CPU reprojected range agreement (stage b)
static constexpr float kReprojectionMismatchFracCeiling = 0.02f; // fraction of populated cells allowed outside tol

// Measured on the committed sample (this file's header): crossing_car
// 98.5%, oncoming_car 87.9%, receding_car 100.0%, overall IoU 74.4%.
static constexpr float kCrossingRecallFloorPct   = 90.0f;
static constexpr float kOncomingRecallFloorPct   = 75.0f;
static constexpr float kRecedingRecallFloorPct   = 90.0f;
static constexpr float kOverallIouFloorPct       = 65.0f;

// Measured: oncoming 87.9% negative, receding 100.0% positive.
static constexpr float kSignConsistencyFloorPct  = 80.0f;  // fraction of a cohort showing the DERIVED sign

// Measured: 0.0% (WALL+POLE combined).
static constexpr float kStaticFalsePositiveCeilingPct = 5.0f;   // WALL+POLE combined

// Measured: M=1 26.4% FP vs M=4 0.0% FP on the disocclusion band (a ratio
// the code below reports as a capped large number, never a literal
// infinity — see kDisoccMitigationRatioCap).
static constexpr float kDisoccMitigationRatioFloor = 3.0f;  // FP-rate(M=1) / FP-rate(M=4) on the disocclusion band
static constexpr double kDisoccMitigationRatioCap = 50.0;   // reported ratio ceiling when the M=4 rate is exactly 0%

static constexpr double kTimingBudgetMs = 50.0;  // one full MOS pass must clear a 20 Hz (50 ms) per-scan budget

// ===========================================================================
// Data loading — scans.csv / poses.csv, the committed sample
// scripts/make_synthetic.py writes (that script's module docstring is this
// format's specification).
// ===========================================================================

struct ScanBucket {
    std::vector<int> ring, az_bin;
    std::vector<float> range_m;
    std::vector<int> cohort, truth, disocc;   // ground truth — read only by gates/artifacts below
    int n = 0;
};

struct ScansData {
    ScanBucket scan[kNumScansWindow];
    bool loaded = false;
};

struct PoseData {
    Pose pose[kNumScansWindow];
    bool loaded = false;
};

static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) fields.push_back(cell);
    return fields;
}

// load_scans — parse scans.csv's '#'-prefixed header (asserted against
// kernels.cuh's constants — the 02.08/02.13-style data/code consistency
// check) and its data rows, bucketed by scan_id.
static ScansData load_scans(const std::string& path)
{
    ScansData sd;
    std::ifstream in(path);
    if (!in.is_open()) return sd;

    int hdr_num_scans = -1, hdr_num_beams = -1, hdr_az_bins = -1;
    float hdr_max_range = -1.0f;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(1, eq - 1);
            const size_t start = key.find_first_not_of(" \t");
            key = (start == std::string::npos) ? "" : key.substr(start);
            const std::string val = line.substr(eq + 1);
            if (key == "num_scans_window")  hdr_num_scans = std::atoi(val.c_str());
            else if (key == "num_beams")    hdr_num_beams = std::atoi(val.c_str());
            else if (key == "azimuth_bins") hdr_az_bins = std::atoi(val.c_str());
            else if (key == "max_range_m")  hdr_max_range = std::strtof(val.c_str(), nullptr);
            continue;
        }
        const auto f = split_csv(line);
        if (f.size() != 7) {
            std::fprintf(stderr, "scans.csv: malformed row (expected 7 fields, got %zu): %s\n",
                         f.size(), line.c_str());
            return ScansData{};
        }
        const int sid = std::atoi(f[0].c_str());
        if (sid < 0 || sid >= kNumScansWindow) {
            std::fprintf(stderr, "scans.csv: scan_id %d out of range\n", sid);
            return ScansData{};
        }
        ScanBucket& b = sd.scan[sid];
        b.ring.push_back(std::atoi(f[1].c_str()));
        b.az_bin.push_back(std::atoi(f[2].c_str()));
        b.range_m.push_back(std::strtof(f[3].c_str(), nullptr));
        b.cohort.push_back(std::atoi(f[4].c_str()));
        b.truth.push_back(std::atoi(f[5].c_str()));
        b.disocc.push_back(std::atoi(f[6].c_str()));
    }
    for (int s = 0; s < kNumScansWindow; ++s) sd.scan[s].n = static_cast<int>(sd.scan[s].ring.size());

    if (hdr_num_scans != kNumScansWindow || hdr_num_beams != kNumBeams || hdr_az_bins != kAzimuthBins ||
        std::fabs(hdr_max_range - kMaxRangeM) > 1e-3f) {
        std::fprintf(stderr,
            "scans.csv header mismatch: file has num_scans_window=%d num_beams=%d azimuth_bins=%d "
            "max_range_m=%.3f; kernels.cuh expects %d/%d/%d/%.3f - regenerate the sample "
            "(scripts/make_synthetic.py) or update kernels.cuh\n",
            hdr_num_scans, hdr_num_beams, hdr_az_bins, static_cast<double>(hdr_max_range),
            kNumScansWindow, kNumBeams, kAzimuthBins, static_cast<double>(kMaxRangeM));
        return ScansData{};
    }
    sd.loaded = true;
    return sd;
}

// load_poses — parse poses.csv's kNumScansWindow pose rows.
static PoseData load_poses(const std::string& path)
{
    PoseData pd;
    std::ifstream in(path);
    if (!in.is_open()) return pd;

    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto f = split_csv(line);
        if (f.size() != 9) { std::fprintf(stderr, "poses.csv: malformed row\n"); return PoseData{}; }
        const int sid = std::atoi(f[0].c_str());
        if (sid < 0 || sid >= kNumScansWindow) { std::fprintf(stderr, "poses.csv: scan_id out of range\n"); return PoseData{}; }
        Pose& p = pd.pose[sid];
        p.p = Vec3{ std::strtof(f[1].c_str(), nullptr), std::strtof(f[2].c_str(), nullptr), std::strtof(f[3].c_str(), nullptr) };
        p.q = Quat{ std::strtof(f[4].c_str(), nullptr), std::strtof(f[5].c_str(), nullptr),
                    std::strtof(f[6].c_str(), nullptr), std::strtof(f[7].c_str(), nullptr) };
        ++count;
    }
    pd.loaded = (count == kNumScansWindow);
    if (!pd.loaded) std::fprintf(stderr, "poses.csv: expected %d poses, found %d\n", kNumScansWindow, count);
    return pd;
}

// ===========================================================================
// GPU stage helpers — each owns its own device allocation/free so main()
// stays a readable sequence of "run stage, get host result" calls.
// ===========================================================================

struct CurrentRangeImage {
    std::vector<float> range_img;
    std::vector<int> cohort_img, truth_img, disocc_img;
};

// gpu_organize_current — stage 1 on the GPU: upload the current scan's
// points, scatter+finalize, download the range image + ground-truth payload.
// gpu_ms accumulates this call's kernel time (both launches).
static CurrentRangeImage gpu_organize_current(const ScanBucket& cur, double& gpu_ms)
{
    int *d_ring = nullptr, *d_az = nullptr, *d_cohort = nullptr, *d_truth = nullptr, *d_disocc = nullptr;
    float* d_range = nullptr;
    const size_t np = static_cast<size_t>(cur.n);
    CUDA_CHECK(cudaMalloc(&d_ring, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_az, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_range, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cohort, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_truth, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_disocc, np * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ring, cur.ring.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_az, cur.az_bin.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_range, cur.range_m.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cohort, cur.cohort.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_truth, cur.truth.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_disocc, cur.disocc.data(), np * sizeof(int), cudaMemcpyHostToDevice));

    unsigned long long* d_enc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_enc, static_cast<size_t>(kNumCells) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_enc, 0xFF, static_cast<size_t>(kNumCells) * sizeof(unsigned long long)));

    float* d_range_img = nullptr; int *d_cohort_img = nullptr, *d_truth_img = nullptr, *d_disocc_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_range_img, static_cast<size_t>(kNumCells) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cohort_img, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_truth_img, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_disocc_img, static_cast<size_t>(kNumCells) * sizeof(int)));

    GpuTimer gt; gt.begin();
    launch_scatter_current(cur.n, d_ring, d_az, d_range, d_enc);
    launch_finalize_current(kNumCells, d_enc, d_range, d_cohort, d_truth, d_disocc,
                            d_range_img, d_cohort_img, d_truth_img, d_disocc_img);
    gpu_ms += static_cast<double>(gt.end_ms());

    CurrentRangeImage out;
    out.range_img.resize(static_cast<size_t>(kNumCells));
    out.cohort_img.resize(static_cast<size_t>(kNumCells));
    out.truth_img.resize(static_cast<size_t>(kNumCells));
    out.disocc_img.resize(static_cast<size_t>(kNumCells));
    CUDA_CHECK(cudaMemcpy(out.range_img.data(), d_range_img, out.range_img.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.cohort_img.data(), d_cohort_img, out.cohort_img.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.truth_img.data(), d_truth_img, out.truth_img.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.disocc_img.data(), d_disocc_img, out.disocc_img.size() * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ring)); CUDA_CHECK(cudaFree(d_az)); CUDA_CHECK(cudaFree(d_range));
    CUDA_CHECK(cudaFree(d_cohort)); CUDA_CHECK(cudaFree(d_truth)); CUDA_CHECK(cudaFree(d_disocc));
    CUDA_CHECK(cudaFree(d_enc));
    CUDA_CHECK(cudaFree(d_range_img)); CUDA_CHECK(cudaFree(d_cohort_img));
    CUDA_CHECK(cudaFree(d_truth_img)); CUDA_CHECK(cudaFree(d_disocc_img));
    return out;
}

static CurrentRangeImage cpu_organize_current(const ScanBucket& cur)
{
    CurrentRangeImage out;
    out.range_img.resize(static_cast<size_t>(kNumCells));
    out.cohort_img.resize(static_cast<size_t>(kNumCells));
    out.truth_img.resize(static_cast<size_t>(kNumCells));
    out.disocc_img.resize(static_cast<size_t>(kNumCells));
    scatter_current_cpu(cur.n, cur.ring.data(), cur.az_bin.data(), cur.range_m.data(),
                        cur.cohort.data(), cur.truth.data(), cur.disocc.data(),
                        out.range_img.data(), out.cohort_img.data(), out.truth_img.data(), out.disocc_img.data());
    return out;
}

// gpu_reproject — stage 2 on the GPU for ONE previous scan: upload, launch,
// download the reprojected range image. Returns a DEVICE pointer the caller
// owns (kept alive for the residual-fuse stage) in *out_device_ptr, plus the
// host copy for the CPU-side comparisons below.
static std::vector<float> gpu_reproject(const ScanBucket& prevb, Pose pose_j, Pose pose_cur,
                                        double& gpu_ms, float** out_device_ptr)
{
    int *d_ring = nullptr, *d_az = nullptr; float* d_range = nullptr;
    const size_t np = static_cast<size_t>(prevb.n);
    CUDA_CHECK(cudaMalloc(&d_ring, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_az, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_range, np * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ring, prevb.ring.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_az, prevb.az_bin.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_range, prevb.range_m.data(), np * sizeof(float), cudaMemcpyHostToDevice));

    unsigned long long* d_enc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_enc, static_cast<size_t>(kNumCells) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_enc, 0xFF, static_cast<size_t>(kNumCells) * sizeof(unsigned long long)));

    float* d_range_img_prev = nullptr;
    CUDA_CHECK(cudaMalloc(&d_range_img_prev, static_cast<size_t>(kNumCells) * sizeof(float)));

    GpuTimer gt; gt.begin();
    launch_reproject_scatter(prevb.n, d_ring, d_az, d_range, pose_j, pose_cur, d_enc);
    launch_finalize_prev(kNumCells, d_enc, d_range_img_prev);
    gpu_ms += static_cast<double>(gt.end_ms());

    std::vector<float> host(static_cast<size_t>(kNumCells));
    CUDA_CHECK(cudaMemcpy(host.data(), d_range_img_prev, host.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ring)); CUDA_CHECK(cudaFree(d_az)); CUDA_CHECK(cudaFree(d_range));
    CUDA_CHECK(cudaFree(d_enc));
    *out_device_ptr = d_range_img_prev;   // caller now owns this device buffer
    return host;
}

static std::vector<float> cpu_reproject(const ScanBucket& prevb, Pose pose_j, Pose pose_cur)
{
    std::vector<float> host(static_cast<size_t>(kNumCells));
    reproject_scatter_cpu(prevb.n, prevb.ring.data(), prevb.az_bin.data(), prevb.range_m.data(),
                          pose_j, pose_cur, host.data());
    return host;
}

struct FusedResult {
    std::vector<float> evidence;
    std::vector<int> sign;
    std::vector<int> candidate;
};

// gpu_residual_fuse — stage 3-4 on the GPU, given DEVICE range-image
// pointers for the current scan and up to kMaxWindowM previous scans
// (nearest-lag first). Builds the small device array-of-pointers
// residual_fuse_kernel expects (kernels.cuh's declaration comment).
static FusedResult gpu_residual_fuse(const float* d_range_cur, const std::vector<float*>& d_prev_ptrs,
                                     int window_m, double& gpu_ms)
{
    const float** d_ptrs_array = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptrs_array), static_cast<size_t>(window_m) * sizeof(const float*)));
    std::vector<const float*> host_ptrs(d_prev_ptrs.begin(), d_prev_ptrs.begin() + window_m);
    CUDA_CHECK(cudaMemcpy(d_ptrs_array, host_ptrs.data(), host_ptrs.size() * sizeof(const float*), cudaMemcpyHostToDevice));

    float* d_evidence = nullptr; int *d_sign = nullptr, *d_candidate = nullptr;
    CUDA_CHECK(cudaMalloc(&d_evidence, static_cast<size_t>(kNumCells) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sign, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate, static_cast<size_t>(kNumCells) * sizeof(int)));

    GpuTimer gt; gt.begin();
    launch_residual_fuse(d_range_cur, d_ptrs_array, window_m, kDynamicThresholdM, d_evidence, d_sign, d_candidate);
    gpu_ms += static_cast<double>(gt.end_ms());

    FusedResult out;
    out.evidence.resize(static_cast<size_t>(kNumCells));
    out.sign.resize(static_cast<size_t>(kNumCells));
    out.candidate.resize(static_cast<size_t>(kNumCells));
    CUDA_CHECK(cudaMemcpy(out.evidence.data(), d_evidence, out.evidence.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.sign.data(), d_sign, out.sign.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.candidate.data(), d_candidate, out.candidate.size() * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ptrs_array));
    CUDA_CHECK(cudaFree(d_evidence)); CUDA_CHECK(cudaFree(d_sign)); CUDA_CHECK(cudaFree(d_candidate));
    return out;
}

static FusedResult cpu_residual_fuse(const std::vector<float>& range_cur,
                                     const std::vector<std::vector<float>>& prev_imgs, int window_m)
{
    std::vector<const float*> ptrs;
    for (int lag = 0; lag < window_m; ++lag) ptrs.push_back(prev_imgs[static_cast<size_t>(lag)].data());
    FusedResult out;
    out.evidence.resize(static_cast<size_t>(kNumCells));
    out.sign.resize(static_cast<size_t>(kNumCells));
    out.candidate.resize(static_cast<size_t>(kNumCells));
    residual_fuse_cpu(range_cur.data(), ptrs, window_m, kDynamicThresholdM,
                      out.evidence.data(), out.sign.data(), out.candidate.data());
    return out;
}

// gpu_ccl — stage 5 on the GPU: build moving-adjacency edges, run the
// generic lock-free union-find to convergence, finalize. Returns the raw
// edge list (host, for the verify gate) and the finalized parent array
// (host) — main() derives component sizes and the final label from these.
struct CclRaw {
    std::vector<std::pair<int,int>> edges;   // as REPORTED by the GPU path (unsorted, insertion order)
    std::vector<int> parent;                 // finalized, size kNumCells
    int sweeps = 0;
};

static CclRaw gpu_ccl(const std::vector<int>& candidate, double& gpu_ms)
{
    int* d_candidate = nullptr;
    CUDA_CHECK(cudaMalloc(&d_candidate, static_cast<size_t>(kNumCells) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_candidate, candidate.data(), candidate.size() * sizeof(int), cudaMemcpyHostToDevice));

    int *d_edge_u = nullptr, *d_edge_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_u, static_cast<size_t>(kNumCells) * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_v, static_cast<size_t>(kNumCells) * 2 * sizeof(int)));

    GpuTimer gt; gt.begin();
    const int num_edges = launch_build_moving_edges(kNumCells, d_candidate, d_edge_u, d_edge_v);

    int* d_parent = nullptr;
    CUDA_CHECK(cudaMalloc(&d_parent, static_cast<size_t>(kNumCells) * sizeof(int)));
    launch_uf_init(kNumCells, d_parent);
    int* d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    int sweeps = 0;
    for (; sweeps < kMaxUfSweeps; ++sweeps) {
        const bool changed = launch_uf_union_sweep(num_edges, d_edge_u, d_edge_v, d_parent, d_changed);
        if (!changed) break;
    }
    launch_uf_finalize(kNumCells, d_parent);
    gpu_ms += static_cast<double>(gt.end_ms());

    CclRaw out;
    out.sweeps = sweeps;
    out.edges.resize(static_cast<size_t>(num_edges));
    std::vector<int> eu(static_cast<size_t>(num_edges)), ev(static_cast<size_t>(num_edges));
    CUDA_CHECK(cudaMemcpy(eu.data(), d_edge_u, eu.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ev.data(), d_edge_v, ev.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int e = 0; e < num_edges; ++e) out.edges[static_cast<size_t>(e)] = { eu[static_cast<size_t>(e)], ev[static_cast<size_t>(e)] };

    out.parent.resize(static_cast<size_t>(kNumCells));
    CUDA_CHECK(cudaMemcpy(out.parent.data(), d_parent, out.parent.size() * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_candidate)); CUDA_CHECK(cudaFree(d_edge_u)); CUDA_CHECK(cudaFree(d_edge_v));
    CUDA_CHECK(cudaFree(d_parent)); CUDA_CHECK(cudaFree(d_changed));
    return out;
}

// final_labels_from_parent — component-size count + min-size filter, done
// on the HOST (02.12's identical scoping decision: this bookkeeping touches
// at most kNumCells=5,760 elements, trivially small next to the GPU stages
// above — kernels.cuh file header point about 02.12's relabel/stats split).
static std::vector<int> final_labels_from_parent(const std::vector<int>& candidate, const std::vector<int>& parent)
{
    std::vector<int> comp_size(static_cast<size_t>(kNumCells), 0);
    for (int c = 0; c < kNumCells; ++c)
        if (candidate[static_cast<size_t>(c)]) ++comp_size[static_cast<size_t>(parent[static_cast<size_t>(c)])];

    std::vector<int> label(static_cast<size_t>(kNumCells), 0);
    for (int c = 0; c < kNumCells; ++c) {
        if (!candidate[static_cast<size_t>(c)]) continue;
        const int root = parent[static_cast<size_t>(c)];
        label[static_cast<size_t>(c)] = (comp_size[static_cast<size_t>(root)] >= kMinMovingClusterSize) ? 1 : 0;
    }
    return label;
}

// cpu_ccl_labels — the full CPU-only stage 5 pipeline (edges + serial
// union-find + component filter), used by the host-only secondary analysis
// (window-size study) so it never needs a second GPU round-trip.
static std::vector<int> cpu_ccl_labels(const std::vector<int>& candidate)
{
    const auto edges = build_moving_edges_cpu(kNumCells, candidate.data());
    std::vector<int> parent;
    serial_union_find_cpu(kNumCells, edges, parent);
    return final_labels_from_parent(candidate, parent);
}

// ===========================================================================
// GateResult — one row of the pass/fail summary (also written to
// demo/out/gates_metrics.csv). NEVER printed with its numbers on a stable
// line (file header "Output contract").
// ===========================================================================
struct GateResult { std::string name; double measured; double threshold; bool pass; std::string note; };

// ===========================================================================
// Tiny PGM (P5, grayscale) / PPM (P6, RGB) canvases — no image library
// (CLAUDE.md §5: these formats ARE "write the header, then raw bytes").
// Each range-image cell is rendered as an UP_SCALE_X x UP_SCALE_Y block so
// the demo's 360x16 range image is actually visible (a bare 1px/cell image
// would be a sliver).
// ===========================================================================
constexpr int kUpscaleX = 3;
constexpr int kUpscaleY = 9;
constexpr int kImgW = kAzimuthBins * kUpscaleX;
constexpr int kImgH = kNumBeams * kUpscaleY;

// cell_to_block — top-left pixel of cell (ring,az)'s block. Ring 0 (the
// BOTTOM beam) is drawn at the BOTTOM of the image (row flip), the
// conventional "up is up" orientation for a range image.
static void cell_to_block(int ring, int az, int& x0, int& y0)
{
    x0 = az * kUpscaleX;
    y0 = (kNumBeams - 1 - ring) * kUpscaleY;
}

static bool write_pgm(const std::string& path, const std::vector<unsigned char>& gray)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << kImgW << " " << kImgH << "\n255\n";
    f.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return f.good();
}

static bool write_ppm(const std::string& path, const std::vector<unsigned char>& rgb)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P6\n" << kImgW << " " << kImgH << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return f.good();
}

static void fill_block_gray(std::vector<unsigned char>& img, int ring, int az, unsigned char v)
{
    int x0, y0; cell_to_block(ring, az, x0, y0);
    for (int dy = 0; dy < kUpscaleY; ++dy)
        for (int dx = 0; dx < kUpscaleX; ++dx)
            img[static_cast<size_t>((y0 + dy) * kImgW + (x0 + dx))] = v;
}

static void fill_block_rgb(std::vector<unsigned char>& img, int ring, int az, unsigned char r, unsigned char g, unsigned char b)
{
    int x0, y0; cell_to_block(ring, az, x0, y0);
    for (int dy = 0; dy < kUpscaleY; ++dy)
        for (int dx = 0; dx < kUpscaleX; ++dx) {
            const size_t idx = (static_cast<size_t>(y0 + dy) * kImgW + (x0 + dx)) * 3;
            img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
        }
}

int main(int argc, char** argv)
{
    std::string data_dir_override;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir_override = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data DIR]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Moving-object segmentation from sequential scans (online MOS) - project 02.14\n");
    print_device_info();
    std::printf("PROBLEM: online MOS, current scan + M=%d previous scans, range image %d beams x %d az "
                "(%d cells), max range %.1f m, residual threshold %.2f m\n",
                kMaxWindowM, kNumBeams, kAzimuthBins, kNumCells,
                static_cast<double>(kMaxRangeM), static_cast<double>(kDynamicThresholdM));

    // ---- 0) Data --------------------------------------------------------------
    const std::string scans_path = find_data_file(data_dir_override, argv[0], "scans.csv");
    const std::string poses_path = find_data_file(data_dir_override, argv[0], "poses.csv");
    if (scans_path.empty() || poses_path.empty()) {
        std::printf("[info] scans.csv/poses.csv not found under data/sample/ (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    const ScansData sd = load_scans(scans_path);
    const PoseData pd = load_poses(poses_path);
    if (!sd.loaded || !pd.loaded) {
        std::printf("RESULT: FAIL (data malformed - see stderr)\n");
        return 1;
    }
    int total_points = 0;
    for (int s = 0; s < kNumScansWindow; ++s) total_points += sd.scan[s].n;
    std::printf("[info] loaded %d scan points across %d scans (current-scan count: %d)\n",
                total_points, kNumScansWindow, sd.scan[kCurrentScanIdx].n);

    const ScanBucket& cur = sd.scan[kCurrentScanIdx];
    const Pose pose_cur = pd.pose[kCurrentScanIdx];

    bool all_verify_pass = true;
    double total_gpu_ms = 0.0;   // accumulates the CANONICAL M=4 pipeline's kernel time (the timing gate below)

    // ======================= VERIFY (a): CURRENT-SCAN ORGANIZE EXACT ============
    CurrentRangeImage cur_gpu = gpu_organize_current(cur, total_gpu_ms);
    {
        const CurrentRangeImage cur_cpu_ref = cpu_organize_current(cur);
        int mismatches = 0;
        for (int c = 0; c < kNumCells; ++c) {
            if (cur_gpu.range_img[static_cast<size_t>(c)] != cur_cpu_ref.range_img[static_cast<size_t>(c)] ||
                cur_gpu.cohort_img[static_cast<size_t>(c)] != cur_cpu_ref.cohort_img[static_cast<size_t>(c)] ||
                cur_gpu.truth_img[static_cast<size_t>(c)] != cur_cpu_ref.truth_img[static_cast<size_t>(c)] ||
                cur_gpu.disocc_img[static_cast<size_t>(c)] != cur_cpu_ref.disocc_img[static_cast<size_t>(c)])
                ++mismatches;
        }
        std::printf("[info] current-scan organize: %d cell mismatch(es) of %d (bit-exact expected - no "
                    "trig in this stage)\n", mismatches, kNumCells);
        std::printf("VERIFY: current-scan range image exact %s\n", mismatches == 0 ? "PASS" : "FAIL");
        if (mismatches != 0) all_verify_pass = false;
    }

    // ======================= VERIFY (b): REPROJECTION CLOSE ======================
    // Reproject each of the kMaxWindowM previous scans (kPrevScanIdx order,
    // nearest lag first). GPU device pointers are kept alive in d_prev_ptrs
    // for the residual-fuse stage below.
    std::vector<std::vector<float>> prev_host(kMaxWindowM);
    std::vector<float*> d_prev_ptrs(kMaxWindowM, nullptr);
    {
        int total_mismatch_cells = 0;
        int total_populated_cells = 0;
        float worst_dev = 0.0f;
        for (int lag = 0; lag < kMaxWindowM; ++lag) {
            const int scan_idx = kPrevScanIdx[lag];
            const ScanBucket& prevb = sd.scan[scan_idx];
            const Pose pose_j = pd.pose[scan_idx];

            prev_host[static_cast<size_t>(lag)] = gpu_reproject(prevb, pose_j, pose_cur, total_gpu_ms, &d_prev_ptrs[static_cast<size_t>(lag)]);
            const std::vector<float> cpu_img = cpu_reproject(prevb, pose_j, pose_cur);

            for (int c = 0; c < kNumCells; ++c) {
                const float g = prev_host[static_cast<size_t>(lag)][static_cast<size_t>(c)];
                const float h = cpu_img[static_cast<size_t>(c)];
                if (g <= 0.0f && h <= 0.0f) continue;
                ++total_populated_cells;
                const float dev = std::fabs(g - h);
                if (dev > worst_dev) worst_dev = dev;
                if (dev > kReprojectionToleranceM || (g <= 0.0f) != (h <= 0.0f)) ++total_mismatch_cells;
            }
        }
        const double mismatch_frac = total_populated_cells > 0
            ? static_cast<double>(total_mismatch_cells) / static_cast<double>(total_populated_cells) : 0.0;
        std::printf("[info] reprojection comparison: %d/%d populated cells outside tol %.4f m "
                    "(%.2f%%), worst deviation %.4e m (GPU sinf/cosf/asinf/atan2f vs host libm ULP "
                    "differences can rarely flip a nearest-ring/az_bin snap - THEORY.md 'Numerical "
                    "considerations')\n",
                    total_mismatch_cells, total_populated_cells, static_cast<double>(kReprojectionToleranceM),
                    100.0 * mismatch_frac, static_cast<double>(worst_dev));
        const bool pass = mismatch_frac <= static_cast<double>(kReprojectionMismatchFracCeiling);
        std::printf("VERIFY: reprojected previous-scan range images close %s (tol %.4f m, mismatch "
                    "fraction ceiling %.1f%%)\n", pass ? "PASS" : "FAIL",
                    static_cast<double>(kReprojectionToleranceM), 100.0 * static_cast<double>(kReprojectionMismatchFracCeiling));
        if (!pass) all_verify_pass = false;
    }

    // ======================= VERIFY (c): RESIDUAL FUSION EXACT (M=4) =============
    // Fed the SAME already-verified GPU range images on BOTH paths, so this
    // isolates the fusion ARITHMETIC alone (no trig anywhere in this stage —
    // 02.13's "verify against a freshly re-uploaded copy of the already-
    // verified data" pattern, cited in this file's header).
    float* d_range_cur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_range_cur, static_cast<size_t>(kNumCells) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_range_cur, cur_gpu.range_img.data(), static_cast<size_t>(kNumCells) * sizeof(float), cudaMemcpyHostToDevice));

    const FusedResult fused_gpu4 = gpu_residual_fuse(d_range_cur, d_prev_ptrs, kMaxWindowM, total_gpu_ms);
    {
        const FusedResult fused_cpu4 = cpu_residual_fuse(cur_gpu.range_img, prev_host, kMaxWindowM);
        int mismatches = 0;
        for (int c = 0; c < kNumCells; ++c) {
            if (fused_gpu4.evidence[static_cast<size_t>(c)] != fused_cpu4.evidence[static_cast<size_t>(c)] ||
                fused_gpu4.sign[static_cast<size_t>(c)] != fused_cpu4.sign[static_cast<size_t>(c)] ||
                fused_gpu4.candidate[static_cast<size_t>(c)] != fused_cpu4.candidate[static_cast<size_t>(c)])
                ++mismatches;
        }
        std::printf("[info] residual fusion (M=%d): %d cell mismatch(es) of %d (bit-exact expected - "
                    "pure arithmetic on identical inputs)\n", kMaxWindowM, mismatches, kNumCells);
        std::printf("VERIFY: residual fusion exact %s\n", mismatches == 0 ? "PASS" : "FAIL");
        if (mismatches != 0) all_verify_pass = false;
    }

    // ======================= VERIFY (d): CCL EXACT ================================
    // gpu_ccl() owns its own device allocation/upload internally (it takes
    // the host candidate vector directly) — see its definition above.
    const CclRaw ccl_gpu = gpu_ccl(fused_gpu4.candidate, total_gpu_ms);
    std::vector<int> label_m4;
    {
        // (a) edge set: canonicalize the GPU's (unsorted) edges the SAME way
        // the CPU twin does, then compare as sorted vectors (02.12's
        // identical set-equality VERIFY pattern).
        std::vector<std::pair<int,int>> gpu_edges = ccl_gpu.edges;
        std::sort(gpu_edges.begin(), gpu_edges.end());
        const auto cpu_edges = build_moving_edges_cpu(kNumCells, fused_gpu4.candidate.data());
        const bool edges_match = (gpu_edges == cpu_edges);
        std::printf("[info] CCL edges: GPU %zu, CPU %zu, %s after canonicalizing\n",
                    gpu_edges.size(), cpu_edges.size(), edges_match ? "identical" : "DIFFERENT");

        // (b) union-find partition: GPU's finalized parent[] vs a serial CPU
        // union-find over the SAME (CPU-built) edge list - both should
        // converge to the identical partition (order-independent union-by-
        // min, 02.04/02.12's identical proof, cited in kernels.cu).
        std::vector<int> cpu_parent;
        serial_union_find_cpu(kNumCells, cpu_edges, cpu_parent);
        int parent_mismatches = 0;
        for (int c = 0; c < kNumCells; ++c)
            if (ccl_gpu.parent[static_cast<size_t>(c)] != cpu_parent[static_cast<size_t>(c)]) ++parent_mismatches;
        std::printf("[info] CCL union-find: %d root mismatch(es) of %d cells, GPU converged in %d sweep(s)\n",
                    parent_mismatches, kNumCells, ccl_gpu.sweeps);

        const bool ccl_pass = edges_match && (parent_mismatches == 0);
        std::printf("VERIFY: CCL (edges + union-find) exact %s\n", ccl_pass ? "PASS" : "FAIL");
        if (!ccl_pass) all_verify_pass = false;

        label_m4 = final_labels_from_parent(fused_gpu4.candidate, ccl_gpu.parent);
    }

    if (!all_verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement in the verify stage - fix before trusting any gate below)\n");
        return 1;
    }

    // ======================= SECONDARY ANALYSIS (host-only) ======================
    // From here on, every computation uses the CPU twins the verify stage
    // just certified bit-exact-or-tolerance-matched against the GPU
    // (02.13's identical policy: re-deriving a second GPU pass would
    // exercise no code path the verify stage has not already proven).
    const FusedResult fused_cpu1 = cpu_residual_fuse(cur_gpu.range_img, prev_host, 1);
    const FusedResult fused_cpu2 = cpu_residual_fuse(cur_gpu.range_img, prev_host, 2);
    const std::vector<int> label_m1 = cpu_ccl_labels(fused_cpu1.candidate);
    const std::vector<int> label_m2 = cpu_ccl_labels(fused_cpu2.candidate);

    // ======================= GATES =================================================
    std::vector<GateResult> gates;

    // ---- mover_detection: per-cohort recall (M=4, post-CCL) + overall IoU -------
    {
        struct { int cohort; const char* name; float floor_pct; } cohorts[] = {
            { kCohortCrossingCar, "crossing_car", kCrossingRecallFloorPct },
            { kCohortOncomingCar, "oncoming_car", kOncomingRecallFloorPct },
            { kCohortRecedingCar, "receding_car", kRecedingRecallFloorPct },
        };
        bool pass = true;
        for (const auto& co : cohorts) {
            int total = 0, hit = 0;
            for (int c = 0; c < kNumCells; ++c)
                if (cur_gpu.cohort_img[static_cast<size_t>(c)] == co.cohort) {
                    ++total;
                    if (label_m4[static_cast<size_t>(c)] == 1) ++hit;
                }
            const double pct = total > 0 ? 100.0 * hit / total : 0.0;
            std::printf("[info] mover_detection %s: %d/%d points recalled (%.1f%%), floor %.1f%%\n",
                        co.name, hit, total, pct, static_cast<double>(co.floor_pct));
            if (pct < co.floor_pct) pass = false;
        }
        int tp = 0, fp = 0, fn = 0;
        for (int c = 0; c < kNumCells; ++c) {
            if (cur_gpu.range_img[static_cast<size_t>(c)] <= 0.0f) continue;
            const bool truth_dyn = cur_gpu.truth_img[static_cast<size_t>(c)] == 1;
            const bool pred_dyn = label_m4[static_cast<size_t>(c)] == 1;
            if (truth_dyn && pred_dyn) ++tp;
            else if (!truth_dyn && pred_dyn) ++fp;
            else if (truth_dyn && !pred_dyn) ++fn;
        }
        const double iou_pct = (tp + fp + fn) > 0 ? 100.0 * tp / (tp + fp + fn) : 0.0;
        std::printf("[info] mover_detection overall: IoU %.1f%% (tp=%d fp=%d fn=%d), floor %.1f%%\n",
                    iou_pct, tp, fp, fn, static_cast<double>(kOverallIouFloorPct));
        if (iou_pct < kOverallIouFloorPct) pass = false;
        GateResult g{ "mover_detection", iou_pct, static_cast<double>(kOverallIouFloorPct), pass,
                     "per-cohort recall (crossing/oncoming/receding) + overall IoU" };
        gates.push_back(g);
    }

    // ---- sign_semantics: oncoming -> negative, receding -> positive (M=4) -------
    {
        auto sign_fraction = [&](int cohort, int expected_sign) -> std::pair<int,double> {
            int total = 0, matching = 0;
            for (int c = 0; c < kNumCells; ++c)
                if (cur_gpu.cohort_img[static_cast<size_t>(c)] == cohort) {
                    ++total;
                    if (fused_gpu4.sign[static_cast<size_t>(c)] == expected_sign) ++matching;
                }
            return { total, total > 0 ? 100.0 * matching / total : 0.0 };
        };
        // NOTE: plain std::pair .first/.second access, not a structured
        // binding — nvcc's host front end (EDG) in this toolkit version
        // mis-parses "const auto [a, b] = expr;" in a few contexts inside
        // .cu files (it can confuse the [] with a trailing-return-type
        // arrow it expects nearby), so this project avoids C++17
        // structured bindings anywhere inside main.cu on principle.
        const std::pair<int,double> on = sign_fraction(kCohortOncomingCar, -1);
        const std::pair<int,double> re = sign_fraction(kCohortRecedingCar, +1);
        const int on_total = on.first; const double on_pct = on.second;
        const int re_total = re.first; const double re_pct = re.second;
        std::printf("[info] sign_semantics oncoming_car: %.1f%% of %d points show the DERIVED negative "
                    "residual (arrival: current closer than reprojected previous), floor %.1f%%\n",
                    on_pct, on_total, static_cast<double>(kSignConsistencyFloorPct));
        std::printf("[info] sign_semantics receding_car: %.1f%% of %d points show the DERIVED positive "
                    "residual (departure: current farther than reprojected previous), floor %.1f%%\n",
                    re_pct, re_total, static_cast<double>(kSignConsistencyFloorPct));
        const bool pass = (on_pct >= kSignConsistencyFloorPct) && (re_pct >= kSignConsistencyFloorPct);
        GateResult g{ "sign_semantics", std::min(on_pct, re_pct), static_cast<double>(kSignConsistencyFloorPct),
                     pass, "oncoming shows negative residual, receding shows positive (the two-sided derivation)" };
        gates.push_back(g);
    }

    // ---- static_precision: WALL+POLE false-positive rate (M=4) ------------------
    {
        int total = 0, falsely_moving = 0;
        int wall_total = 0, wall_bad = 0, pole_total = 0, pole_bad = 0;
        int disocc_total = 0, disocc_bad = 0, clean_wall_total = 0, clean_wall_bad = 0;
        for (int c = 0; c < kNumCells; ++c) {
            const int co = cur_gpu.cohort_img[static_cast<size_t>(c)];
            if (co != kCohortWall && co != kCohortPole) continue;
            ++total;
            const bool bad = (label_m4[static_cast<size_t>(c)] == 1);
            if (bad) ++falsely_moving;
            if (co == kCohortWall) {
                ++wall_total; if (bad) ++wall_bad;
                if (cur_gpu.disocc_img[static_cast<size_t>(c)] == 1) { ++disocc_total; if (bad) ++disocc_bad; }
                else { ++clean_wall_total; if (bad) ++clean_wall_bad; }
            } else { ++pole_total; if (bad) ++pole_bad; }
        }
        const double pct = total > 0 ? 100.0 * falsely_moving / total : 0.0;
        const bool pass = pct <= kStaticFalsePositiveCeilingPct;
        std::printf("[info] static_precision: %d/%d static (WALL+POLE) points falsely labeled moving "
                    "(%.1f%%), ceiling %.1f%%\n", falsely_moving, total, pct, static_cast<double>(kStaticFalsePositiveCeilingPct));
        std::printf("[info] static_precision honesty split: wall(clean) %d/%d (%.1f%%); "
                    "wall(disocclusion-band) %d/%d (%.1f%%); pole(thin, discretization) %d/%d (%.1f%%) - "
                    "reported, not individually gated (reprojection-quantization honesty, THEORY.md)\n",
                    clean_wall_bad, clean_wall_total, clean_wall_total > 0 ? 100.0 * clean_wall_bad / clean_wall_total : 0.0,
                    disocc_bad, disocc_total, disocc_total > 0 ? 100.0 * disocc_bad / disocc_total : 0.0,
                    pole_bad, pole_total, pole_total > 0 ? 100.0 * pole_bad / pole_total : 0.0);
        GateResult g{ "static_precision", pct, static_cast<double>(kStaticFalsePositiveCeilingPct), pass,
                     "WALL+POLE points falsely labeled moving" };
        gates.push_back(g);
    }

    // ---- disocclusion_mitigation: FP rate on the wall's disocclusion band, ------
    // M=1 (no consistency check) vs M=4 (the mitigation), measured on the
    // PRE-CCL candidate labels (isolating the fusion mechanism itself from
    // the CCL cleanup's own, separate denoising effect).
    {
        auto disocc_fp_rate = [&](const std::vector<int>& candidate) -> std::pair<int,double> {
            int total = 0, bad = 0;
            for (int c = 0; c < kNumCells; ++c) {
                if (cur_gpu.cohort_img[static_cast<size_t>(c)] != kCohortWall) continue;
                if (cur_gpu.disocc_img[static_cast<size_t>(c)] != 1) continue;
                ++total;
                if (candidate[static_cast<size_t>(c)] == 1) ++bad;
            }
            return { total, total > 0 ? 100.0 * bad / total : 0.0 };
        };
        const std::pair<int,double> d1 = disocc_fp_rate(fused_cpu1.candidate);
        const std::pair<int,double> d4 = disocc_fp_rate(fused_gpu4.candidate);
        const int n1 = d1.first; const double fp1 = d1.second;
        const int n4 = d4.first; const double fp4 = d4.second;
        // Avoid a divide-by-zero blow-up if the mitigated rate is exactly
        // zero: report the ratio capped at a large, honestly-labeled value
        // rather than an actual infinity.
        const double ratio = fp4 > 0.0 ? fp1 / fp4 : (fp1 > 0.0 ? kDisoccMitigationRatioCap : 1.0);
        std::printf("[info] disocclusion_mitigation: WALL disocclusion-band false-positive rate WITHOUT "
                    "the multi-scan check (M=1) = %.1f%% of %d points; WITH it (M=4) = %.1f%% of %d "
                    "points; improvement ratio %.2fx, floor %.2fx\n",
                    fp1, n1, fp4, n4, ratio, static_cast<double>(kDisoccMitigationRatioFloor));
        const bool pass = ratio >= static_cast<double>(kDisoccMitigationRatioFloor);
        GateResult g{ "disocclusion_mitigation", ratio, static_cast<double>(kDisoccMitigationRatioFloor), pass,
                     "WALL disocclusion-band FP-rate improvement, M=1 (no check) vs M=4 (mitigated)" };
        gates.push_back(g);
    }

    // ---- timing: one full canonical M=4 MOS pass vs a 20 Hz (50 ms) budget ------
    {
        const bool pass = total_gpu_ms < kTimingBudgetMs;
        std::printf("[time] full MOS pass (organize current + %d reprojections + residual fuse M=%d + "
                    "CCL): %.3f ms total GPU kernel time, budget %.1f ms (20 Hz)\n",
                    kMaxWindowM, kMaxWindowM, total_gpu_ms, kTimingBudgetMs);
        GateResult g{ "timing", total_gpu_ms, kTimingBudgetMs, pass,
                     "full MOS pass GPU kernel time vs 20 Hz per-scan budget" };
        gates.push_back(g);
    }

    // ---- temporal_boundary ([info] only): STOPPED_CAR recall at M=1,2,4 ---------
    {
        auto recall_for = [&](const std::vector<int>& label) -> std::pair<int,double> {
            int total = 0, hit = 0;
            for (int c = 0; c < kNumCells; ++c)
                if (cur_gpu.cohort_img[static_cast<size_t>(c)] == kCohortStoppedCar) {
                    ++total;
                    if (label[static_cast<size_t>(c)] == 1) ++hit;
                }
            return { total, total > 0 ? 100.0 * hit / total : 0.0 };
        };
        const std::pair<int,double> t1 = recall_for(label_m1);
        const std::pair<int,double> t2 = recall_for(label_m2);
        const std::pair<int,double> t4 = recall_for(label_m4);
        const int n1 = t1.first; const double r1 = t1.second;
        const int n2 = t2.first; const double r2 = t2.second;
        const int n4 = t4.first; const double r4 = t4.second;
        std::printf("[info] temporal_boundary (stopped_car, %d ground-truth points, moving for scans "
                    "0-3 then held stationary between the last previous scan and now): recall M=1 "
                    "%.1f%%, M=2 %.1f%%, M=4 %.1f%% - MIN-fusion always includes the freshest (lag-1) "
                    "comparison, which already reads near-zero for a just-stopped object, so recall "
                    "here is NOT expected to improve with larger M (kernels.cu's residual_fuse_kernel "
                    "comment derives this as a property of MIN fusion, not a bug; THEORY.md 'Numerical "
                    "considerations')\n", n4, r1, r2, r4);
        (void)n1; (void)n2;   // n1/n2/n4 are all the SAME ground-truth count (cohort size never changes across M); n4 is printed above
    }

    // ---- window_size ([info] only): overall recall/precision/IoU at M=1,2,4 -----
    {
        // A named POD instead of std::tuple + structured bindings (this
        // project avoids structured bindings anywhere in main.cu — see the
        // sign_semantics gate's comment for why).
        struct Stats { double recall, precision, iou; };
        auto overall_stats = [&](const std::vector<int>& label) -> Stats {
            int tp = 0, fp = 0, fn = 0;
            for (int c = 0; c < kNumCells; ++c) {
                if (cur_gpu.range_img[static_cast<size_t>(c)] <= 0.0f) continue;
                const bool truth_dyn = cur_gpu.truth_img[static_cast<size_t>(c)] == 1;
                const bool pred_dyn = label[static_cast<size_t>(c)] == 1;
                if (truth_dyn && pred_dyn) ++tp;
                else if (!truth_dyn && pred_dyn) ++fp;
                else if (truth_dyn && !pred_dyn) ++fn;
            }
            Stats s;
            s.recall = (tp + fn) > 0 ? 100.0 * tp / (tp + fn) : 0.0;
            s.precision = (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0.0;
            s.iou = (tp + fp + fn) > 0 ? 100.0 * tp / (tp + fp + fn) : 0.0;
            return s;
        };
        const Stats s1 = overall_stats(label_m1);
        const Stats s2 = overall_stats(label_m2);
        const Stats s4 = overall_stats(label_m4);
        const double r1 = s1.recall, p1 = s1.precision, i1 = s1.iou;
        const double r2 = s2.recall, p2 = s2.precision, i2 = s2.iou;
        const double r4 = s4.recall, p4 = s4.precision, i4 = s4.iou;
        std::printf("[info] window_size study (overall, all dynamic cohorts combined): M=1 recall "
                    "%.1f%% precision %.1f%% IoU %.1f%%; M=2 recall %.1f%% precision %.1f%% IoU %.1f%%; "
                    "M=4 recall %.1f%% precision %.1f%% IoU %.1f%% - more history generally trades a "
                    "little recall for a lot of precision (MIN-fusion's disocclusion resistance, "
                    "kernels.cu), at the cost of buffering M scans of latency\n",
                    r1, p1, i1, r2, p2, i2, r4, p4, i4);
    }

    // ======================= ARTIFACTS ==============================================
    const std::string out_dir = resolve_out_dir(argv[0]);

    // residual_image.pgm — fused |residual| (M=4), the signature visual:
    // movers glow. vis_max chosen as a multiple of the threshold so the
    // decision boundary sits at a recognizable mid-gray, not washed out.
    bool residual_ok = false;
    {
        const float vis_max = 4.0f * kDynamicThresholdM;
        std::vector<unsigned char> gray(static_cast<size_t>(kImgW) * static_cast<size_t>(kImgH), 0);
        for (int c = 0; c < kNumCells; ++c) {
            const int ring = c / kAzimuthBins, az = c % kAzimuthBins;
            const float ev = fused_gpu4.evidence[static_cast<size_t>(c)];
            unsigned char v = 0;
            if (ev >= 0.0f) {
                float norm = ev / vis_max;
                if (norm > 1.0f) norm = 1.0f;
                v = static_cast<unsigned char>(norm * 255.0f);
            }
            fill_block_gray(gray, ring, az, v);
        }
        residual_ok = write_pgm(out_dir + "/residual_image.pgm", gray);
    }
    std::printf("ARTIFACT: %s demo/out/residual_image.pgm (fused |residual|, M=%d, movers glow)\n",
                residual_ok ? "wrote" : "FAILED to write", kMaxWindowM);

    // label_vs_truth.ppm — confusion map: TP green, FP red, FN orange, TN gray.
    bool label_ok = false;
    {
        std::vector<unsigned char> rgb(static_cast<size_t>(kImgW) * static_cast<size_t>(kImgH) * 3, 20);
        for (int c = 0; c < kNumCells; ++c) {
            if (cur_gpu.range_img[static_cast<size_t>(c)] <= 0.0f) continue;
            const int ring = c / kAzimuthBins, az = c % kAzimuthBins;
            const bool truth_dyn = cur_gpu.truth_img[static_cast<size_t>(c)] == 1;
            const bool pred_dyn = label_m4[static_cast<size_t>(c)] == 1;
            if (truth_dyn && pred_dyn) fill_block_rgb(rgb, ring, az, 40, 200, 90);
            else if (!truth_dyn && pred_dyn) fill_block_rgb(rgb, ring, az, 230, 60, 60);
            else if (truth_dyn && !pred_dyn) fill_block_rgb(rgb, ring, az, 255, 165, 0);
            else fill_block_rgb(rgb, ring, az, 170, 170, 170);
        }
        label_ok = write_ppm(out_dir + "/label_vs_truth.ppm", rgb);
    }
    std::printf("ARTIFACT: %s demo/out/label_vs_truth.ppm (confusion map: TP green / FP red / FN "
                "orange / TN gray)\n", label_ok ? "wrote" : "FAILED to write");

    // disocclusion_band.ppm — WALL cells: disocclusion-band magenta, clean
    // wall gray; everything else dim.
    bool disocc_ok = false;
    {
        std::vector<unsigned char> rgb(static_cast<size_t>(kImgW) * static_cast<size_t>(kImgH) * 3, 20);
        for (int c = 0; c < kNumCells; ++c) {
            if (cur_gpu.range_img[static_cast<size_t>(c)] <= 0.0f) continue;
            const int ring = c / kAzimuthBins, az = c % kAzimuthBins;
            if (cur_gpu.cohort_img[static_cast<size_t>(c)] == kCohortWall) {
                if (cur_gpu.disocc_img[static_cast<size_t>(c)] == 1) fill_block_rgb(rgb, ring, az, 230, 60, 220);
                else fill_block_rgb(rgb, ring, az, 140, 140, 140);
            } else {
                fill_block_rgb(rgb, ring, az, 55, 55, 85);
            }
        }
        disocc_ok = write_ppm(out_dir + "/disocclusion_band.ppm", rgb);
    }
    std::printf("ARTIFACT: %s demo/out/disocclusion_band.ppm (WALL disocclusion-band cells "
                "highlighted magenta)\n", disocc_ok ? "wrote" : "FAILED to write");

    // per_cohort_metrics.csv
    bool cohort_csv_ok = false;
    {
        std::ofstream f(out_dir + "/per_cohort_metrics.csv");
        cohort_csv_ok = f.is_open();
        if (cohort_csv_ok) {
            f << "cohort,role,total_points,flagged_moving,pct_flagged_moving\n";
            const int all_cohorts[] = { kCohortWall, kCohortPole, kCohortCrossingCar,
                                        kCohortOncomingCar, kCohortRecedingCar, kCohortStoppedCar };
            for (int co : all_cohorts) {
                int total = 0, flagged = 0;
                for (int c = 0; c < kNumCells; ++c)
                    if (cur_gpu.cohort_img[static_cast<size_t>(c)] == co) {
                        ++total;
                        if (label_m4[static_cast<size_t>(c)] == 1) ++flagged;
                    }
                const double pct = total > 0 ? 100.0 * flagged / total : 0.0;
                f << cohort_name(co) << ',' << (cohort_is_dynamic(co) ? "dynamic" : "static") << ','
                  << total << ',' << flagged << ',' << pct << '\n';
            }
            cohort_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/per_cohort_metrics.csv (6 cohorts, M=%d)\n",
                cohort_csv_ok ? "wrote" : "FAILED to write", kMaxWindowM);

    std::string gates_csv_path = out_dir + "/gates_metrics.csv";
    bool gates_csv_ok = false;
    {
        std::ofstream f(gates_csv_path);
        if (f.is_open()) {
            f << "gate,measured,threshold,verdict,note\n";
            for (const auto& g : gates)
                f << g.name << ',' << g.measured << ',' << g.threshold << ',' << (g.pass ? "PASS" : "FAIL")
                  << ',' << g.note << '\n';
            gates_csv_ok = f.good();
        }
    }
    std::printf("ARTIFACT: %s demo/out/gates_metrics.csv (%d gates)\n",
                gates_csv_ok ? "wrote" : "FAILED to write", static_cast<int>(gates.size()));

    // ======================= FINAL VERDICT ===========================================
    for (const auto& g : gates)
        std::printf("GATE: %s %s\n", g.name.c_str(), g.pass ? "PASS" : "FAIL");

    bool all_gates_pass = true;
    for (const auto& g : gates) all_gates_pass = all_gates_pass && g.pass;
    const bool artifacts_ok = residual_ok && label_ok && disocc_ok && cohort_csv_ok && gates_csv_ok;
    const bool success = all_verify_pass && all_gates_pass && artifacts_ok;

    // Free the device range-image buffers kept alive since stage (b).
    for (float* p : d_prev_ptrs) if (p) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(d_range_cur));

    if (success)
        std::printf("RESULT: PASS (all verify stages and gates passed; current-scan movers segmented)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above and stderr)\n");
    return success ? 0 : 1;
}
