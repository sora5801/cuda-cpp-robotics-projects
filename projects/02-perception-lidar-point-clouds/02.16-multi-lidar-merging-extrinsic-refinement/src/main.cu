// ===========================================================================
// main.cu — entry point for project 02.16 (Multi-LiDAR merging + extrinsic
//           refinement)
//
// What this program does, start to finish
// ----------------------------------------
// Loads two committed cohorts (data/sample/aligned.csv, drifted.csv — the
// SAME "yard" scene scanned by the SAME 3-LiDAR rig, differing only in
// whether LEFT/RIGHT's mounts carry their documented drift — see
// kernels.cuh's file header for the full rig/geometry story), then runs
// TEN independent verification stages, each printing a stable PASS/FAIL
// line plus "[info]" line(s) with the measured number(s) behind it:
//
//   TRANSFORM_TWIN     — the trivial merge primitive, GPU vs CPU.
//   PLANE_FIT_TWIN      — GPU-accumulated (atomics + host eigensolve) vs
//                         CPU-independent (own loop, own eigensolve) plane
//                         fit, across every valid surface.
//   DRIFT_DETECTION     — plane-pair residuals: the ALIGNED rig's LEFT/RIGHT-
//                         vs-MAIN residuals set the threshold; the DRIFTED
//                         rig must exceed it (the observable drift signal).
//   ASSEMBLY_TWIN       — one-shot GPU normal-equation assembly vs CPU.
//   TRAJECTORY_TWIN     — a full 20-iteration LM trajectory, GPU-orchestrated
//                         vs CPU-independent.
//   RECOVERY_LEFT / RECOVERY_RIGHT — best-available refined extrinsic vs the
//                         TRUE (ground-truth) drifted extrinsic — the
//                         project's headline number.
//   VALIDATION_LOOP     — after refinement, plane-pair residuals return
//                         under the aligned-rig threshold (the closed loop).
//   OBSERVABILITY       — the Hessian condition-number proxy contrast: all
//                         three (mutually orthogonal) zones vs wall_front
//                         alone (01.17's coplanar-pose degeneracy lesson,
//                         recast for LiDAR planes).
//   ZERO_DRIFT_CONTROL  — refining the ALIGNED (undrifted) rig must not
//                         self-inflict a drift correction of its own.
//   DEDUP_ACCOUNTING    — GPU vs CPU voxel-grid dedup of the merged cloud:
//                         EXACT agreement on which points are kept.
//
// Every stage's exact measured numbers are also written to
// demo/out/gates_metrics.csv; per-zone plane residuals to
// demo/out/plane_residuals.csv; and the "money shot" — a top-down view of
// the merged DRIFTED-rig cloud, colored by sensor, BEFORE vs AFTER
// refinement — to demo/out/topview_before.ppm / topview_after.ppm.
//
// Output contract: stable lines "[demo]", "PROBLEM:", every "*_TWIN:" /
// "DRIFT_DETECTION:" / "RECOVERY_*:" / "VALIDATION_LOOP:" / "OBSERVABILITY:"
// / "ZERO_DRIFT_CONTROL:" / "DEDUP_ACCOUNTING:" verdict line, "ARTIFACT:",
// "RESULT:" — "[info]"/"[time]" lines are NOT diffed (every measured float
// and the GPU name live there; FMA/libm rounding differs by GPU
// architecture even though the algorithm is deterministic — 01.17/08.01's
// identical discipline, cited).
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// ===========================================================================
// Gate tolerances — every one is "measured, then margined" (08.01/01.17's
// phrase, cited): the project was run on the committed sample, the ACTUAL
// worst-case deviation was read off the printed [info] lines, and the
// threshold below was set with documented headroom above that measurement.
// ===========================================================================
// Every constant below is the result of ONE actual run on the committed
// sample (RTX 2080 SUPER, sm_75, Release|x64) — the exact [info] line is
// quoted in each comment. Tolerances then add documented headroom, mostly
// 2-15x, EXCEPT where the measured deviation printed as exactly "0.0000":
// several rotation comparisons here (recovery, trajectory twin, zero-drift
// control) converge SO precisely (hundreds of points, mm-level noise, a
// well-conditioned 6x6 solve) that the true angular deviation falls below
// FP32's representable resolution near identity (~1e-4 deg: 1-cos(theta)
// underflows FP32 epsilon well before then, so rotation_angle_deg's acos
// clamp returns exactly 0) — THEORY.md "Numerical considerations" derives
// this bound. Those tolerances are kept modestly above zero rather than
// driven to zero themselves, both for cross-GPU-architecture headroom
// (FP32 FMA reduction order differs by SM architecture) and honesty: "the
// measured value is below our instrument's resolution" is a different,
// weaker claim than "the tolerance is exactly 0".
constexpr float  kTransformTwinTolM        = 1.0e-5f;   // basic map, GPU vs CPU; measured worst 9.54e-07 m, ~10x headroom
constexpr float  kPlaneFitAngleTolDeg      = 0.05f;     // GPU-decoded vs CPU-independent plane normal; measured worst 1.978e-2 deg, ~2.5x headroom
constexpr float  kPlaneFitCentroidTolM     = 2.0e-5f;   // measured worst 4.79e-06 m, ~4x headroom
constexpr double kAssemblyRelTol           = 5.0e-5;    // one-shot H/g/cost, GPU (float tree-reduce) vs CPU (double sum); measured worst 6.83e-06, ~7x headroom
constexpr float  kTrajectoryRotTolDeg      = 0.01f;     // final-pose GPU-orchestrated vs CPU-independent, 20 chained iterations; measured worst 0.0 deg (below FP32 resolution — see block comment)
constexpr float  kTrajectoryTransTolM      = 1.0e-5f;   // measured worst 2.38e-07 m, ~42x headroom
constexpr float  kDriftDetectionAngleTolDeg = 0.10f;    // ALIGNED-rig worst plane-pair angle residual (detection: 1.978e-2 deg, ~5x headroom) AND the post-refinement VALIDATION_LOOP floor (5.234e-2 deg, ~1.9x headroom) both fit under this one threshold
constexpr float  kDriftDetectionOffsetTolM  = 0.0015f;  // measured aligned-detection worst 2.762e-4 m (~5.4x headroom); post-refinement validation worst is far smaller (6.4e-06 m)
constexpr float  kRecoveryRotTolDeg        = 0.02f;     // refined vs TRUE drifted extrinsic (the headline number); measured worst 0.0 deg (below FP32 resolution — see block comment)
constexpr float  kRecoveryTransTolM        = 0.0015f;   // measured worst (RIGHT) 3.117e-4 m, ~4.8x headroom
constexpr double kObservabilityFactor      = 1.0e4;     // degenerate (wall_front-only) condition number must exceed the full-zone-set condition number by this factor; measured ratio ~9.77e9 -- the threshold is kept far below the measurement (a meaningful "must be dramatic" bar, not a razor-thin one) rather than driven up to match it, since the exact ratio is itself noise-sensitive
constexpr float  kZeroDriftRotTolDeg       = 0.01f;     // refining the ALIGNED rig must not move the extrinsic more than this; measured worst 0.0 deg (below FP32 resolution — see block comment)
constexpr float  kZeroDriftTransTolM       = 0.001f;    // measured worst (LEFT) 3.426e-4 m, ~2.9x headroom

// ===========================================================================
// Cohort — one loaded data/sample/*.csv file: N rows of
// (sensor_id, surface_id, x, y, z), points in their OWNING sensor's raw
// frame (kernels.cuh/scripts/make_synthetic.py's shared layout).
// ===========================================================================
struct Cohort {
    std::vector<int32_t> sensor_id;
    std::vector<int32_t> surface_id;
    std::vector<float>   xyz;   // [n*3]
    int n = 0;
    bool loaded = false;
};

// load_cohort — strict CSV loader: '#'-comment lines skipped, every other
// line must be exactly 5 comma-separated fields (08.01/01.17's "fail
// loudly on malformed input" discipline, cited).
static Cohort load_cohort(const std::string& path)
{
    Cohort c;
    if (path.empty()) return c;
    std::ifstream in(path);
    if (!in.is_open()) return c;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> parts;
        while (std::getline(ss, cell, ',')) parts.push_back(cell);
        if (parts.size() != 5) {
            std::fprintf(stderr, "cohort: malformed row (want 5 fields, got %zu): %s\n", parts.size(), line.c_str());
            return Cohort{};
        }
        c.sensor_id.push_back(static_cast<int32_t>(std::strtol(parts[0].c_str(), nullptr, 10)));
        c.surface_id.push_back(static_cast<int32_t>(std::strtol(parts[1].c_str(), nullptr, 10)));
        c.xyz.push_back(std::strtof(parts[2].c_str(), nullptr));
        c.xyz.push_back(std::strtof(parts[3].c_str(), nullptr));
        c.xyz.push_back(std::strtof(parts[4].c_str(), nullptr));
    }
    c.n = static_cast<int>(c.sensor_id.size());
    c.loaded = c.n > 0;
    return c;
}

// extract_sensor — pull one sensor's rows out of a Cohort, preserving order.
static void extract_sensor(const Cohort& c, int32_t sensor_id, std::vector<float>& xyz, std::vector<int32_t>& surface_id)
{
    xyz.clear();
    surface_id.clear();
    for (int i = 0; i < c.n; ++i) {
        if (c.sensor_id[i] != sensor_id) continue;
        xyz.push_back(c.xyz[i * 3 + 0]);
        xyz.push_back(c.xyz[i * 3 + 1]);
        xyz.push_back(c.xyz[i * 3 + 2]);
        surface_id.push_back(c.surface_id[i]);
    }
}

// ===========================================================================
// Device-orchestration helpers. Each function owns its OWN device
// allocations (upload -> kernel(s) -> download -> free) rather than
// threading persistent device pointers through every stage below — a
// deliberate simplicity-over-micro-optimization choice at this project's
// scale (a few thousand points, a few dozen calls total per demo run —
// CLAUDE.md §1's "teaching clarity over micro-optimization" default,
// contrasted explicitly with 01.17/08.01's tighter "allocate once outside
// the loop" discipline, which pays for itself only at those flagships'
// larger per-run call counts).
// ===========================================================================

// gpu_transform — upload xyz, run transform_points_kernel, download, free.
static std::vector<float> gpu_transform(const std::vector<float>& xyz, Rigid3 T)
{
    const int n = static_cast<int>(xyz.size() / 3);
    std::vector<float> out(xyz.size(), 0.0f);
    if (n == 0) return out;

    float *d_in = nullptr, *d_out = nullptr;
    const size_t bytes = xyz.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, xyz.data(), bytes, cudaMemcpyHostToDevice));
    launch_transform_points(n, d_in, T, d_out);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return out;
}

// gpu_transform_multi — the actual MERGE step: each point picks its own
// transform via T_per_sensor[sensor_id[i]].
static std::vector<float> gpu_transform_multi(const std::vector<float>& xyz, const std::vector<int32_t>& sensor_id,
                                              const Rigid3 T_per_sensor[kNumSensors])
{
    const int n = static_cast<int>(sensor_id.size());
    std::vector<float> out(xyz.size(), 0.0f);
    if (n == 0) return out;

    float *d_in = nullptr, *d_out = nullptr;
    int32_t* d_sid = nullptr;
    Rigid3* d_T = nullptr;
    const size_t bytes = xyz.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_sid, sensor_id.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_T, kNumSensors * sizeof(Rigid3)));
    CUDA_CHECK(cudaMemcpy(d_in, xyz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sid, sensor_id.data(), sensor_id.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, T_per_sensor, kNumSensors * sizeof(Rigid3), cudaMemcpyHostToDevice));
    launch_transform_points_multi(n, d_in, d_sid, d_T, d_out);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sid));
    CUDA_CHECK(cudaFree(d_T));
    return out;
}

// gpu_fit_planes — the GPU plane-fit path: two accumulate kernels
// (centroid, then mean-shifted covariance), decoded on the host via
// kernels.cuh's jacobi_eigen_3x3 + the shared reference-point orientation
// convention (kernels.cuh's Plane doc). Points must ALREADY be in whatever
// frame the caller wants the resulting planes expressed in.
static void gpu_fit_planes(const std::vector<float>& xyz, const std::vector<int32_t>& surface_id, Plane out[kNumSurfaces])
{
    for (int s = 0; s < kNumSurfaces; ++s) out[s] = kInvalidPlane;
    const int n = static_cast<int>(surface_id.size());
    if (n == 0) return;

    float *d_xyz = nullptr, *d_sums = nullptr, *d_centroids = nullptr, *d_cov = nullptr;
    int32_t *d_surf = nullptr, *d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_surf, surface_id.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sums, kNumSurfaces * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, kNumSurfaces * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_xyz, xyz.data(), xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_surf, surface_id.data(), surface_id.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    launch_accumulate_centroid(n, d_xyz, d_surf, d_sums, d_counts);
    float h_sums[kNumSurfaces * 3];
    int32_t h_counts[kNumSurfaces];
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums, sizeof(h_sums), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts, sizeof(h_counts), cudaMemcpyDeviceToHost));

    float h_centroids[kNumSurfaces * 3] = {};
    for (int s = 0; s < kNumSurfaces; ++s) {
        if (h_counts[s] < kMinPlanePoints) continue;
        h_centroids[s * 3 + 0] = h_sums[s * 3 + 0] / static_cast<float>(h_counts[s]);
        h_centroids[s * 3 + 1] = h_sums[s * 3 + 1] / static_cast<float>(h_counts[s]);
        h_centroids[s * 3 + 2] = h_sums[s * 3 + 2] / static_cast<float>(h_counts[s]);
    }

    CUDA_CHECK(cudaMalloc(&d_centroids, sizeof(h_centroids)));
    CUDA_CHECK(cudaMalloc(&d_cov, kNumSurfaces * 6 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, sizeof(h_centroids), cudaMemcpyHostToDevice));
    launch_accumulate_covariance(n, d_xyz, d_surf, d_centroids, d_cov);
    float h_cov[kNumSurfaces * 6];
    CUDA_CHECK(cudaMemcpy(h_cov, d_cov, sizeof(h_cov), cudaMemcpyDeviceToHost));

    for (int s = 0; s < kNumSurfaces; ++s) {
        if (h_counts[s] < kMinPlanePoints) continue;
        float cov_avg[6];
        for (int k = 0; k < 6; ++k) cov_avg[k] = h_cov[s * 6 + k] / static_cast<float>(h_counts[s]);
        float eigenvalues[3];
        float eigenvectors[3][3];
        jacobi_eigen_3x3(cov_avg, eigenvalues, eigenvectors);   // ascending -> [0] is the plane normal

        float normal[3] = { eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2] };
        const float cx = h_centroids[s * 3 + 0], cy = h_centroids[s * 3 + 1], cz = h_centroids[s * 3 + 2];
        const float ref_dot = normal[0] * (kPlaneOrientRef[0] - cx) +
                              normal[1] * (kPlaneOrientRef[1] - cy) +
                              normal[2] * (kPlaneOrientRef[2] - cz);
        if (ref_dot < 0.0f) { normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; }

        out[s].normal[0] = normal[0]; out[s].normal[1] = normal[1]; out[s].normal[2] = normal[2];
        out[s].centroid[0] = cx; out[s].centroid[1] = cy; out[s].centroid[2] = cz;
        out[s].valid = 1;
        out[s].count = h_counts[s];
    }

    CUDA_CHECK(cudaFree(d_xyz)); CUDA_CHECK(cudaFree(d_surf));
    CUDA_CHECK(cudaFree(d_sums)); CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_centroids)); CUDA_CHECK(cudaFree(d_cov));
}

// gpu_assemble_once — one call to assemble_point_to_plane_kernel: upload,
// launch, download the per-block partials, sum them in double on the host
// (01.17/02.06's "GPU partial reduce, host finishes it" split, cited).
static void gpu_assemble_once(const std::vector<float>& p_src, const std::vector<int32_t>& surface_id,
                              Rigid3 T, const Plane target_planes[kNumSurfaces], uint32_t zone_mask,
                              double H21[21], double g6[6], double& cost)
{
    for (int a = 0; a < 21; ++a) H21[a] = 0.0;
    for (int a = 0; a < 6; ++a) g6[a] = 0.0;
    cost = 0.0;
    const int n = static_cast<int>(surface_id.size());
    if (n == 0) return;

    float* d_p = nullptr;
    int32_t* d_surf = nullptr;
    Plane* d_planes = nullptr;
    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p, p_src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_surf, surface_id.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_planes, kNumSurfaces * sizeof(Plane)));
    CUDA_CHECK(cudaMemcpy(d_p, p_src.data(), p_src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_surf, surface_id.data(), surface_id.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_planes, target_planes, kNumSurfaces * sizeof(Plane), cudaMemcpyHostToDevice));

    const int max_blocks = blocks_for(n, kThreadsReduce);
    CUDA_CHECK(cudaMalloc(&d_partials, static_cast<size_t>(max_blocks) * kReduceWidth * sizeof(float)));

    const int num_blocks = launch_assemble_point_to_plane(d_p, d_surf, n, T, d_planes, zone_mask, d_partials);
    std::vector<float> h_partials(static_cast<size_t>(num_blocks) * kReduceWidth);
    CUDA_CHECK(cudaMemcpy(h_partials.data(), d_partials, h_partials.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; ++b) {
        const float* row = &h_partials[static_cast<size_t>(b) * kReduceWidth];
        for (int a = 0; a < 21; ++a) H21[a] += static_cast<double>(row[a]);
        for (int a = 0; a < 6; ++a) g6[a] += static_cast<double>(row[21 + a]);
        cost += static_cast<double>(row[27]);
    }

    CUDA_CHECK(cudaFree(d_p)); CUDA_CHECK(cudaFree(d_surf));
    CUDA_CHECK(cudaFree(d_planes)); CUDA_CHECK(cudaFree(d_partials));
}

// run_refinement_lm_gpu — the host-orchestrated LM loop: every assembly
// (both "at current T" and "at the candidate T" accept/reject needs) goes
// through gpu_assemble_once. Mirrors 01.17's run_lm_gpu control flow
// exactly (Marquardt damping, accept/reject, convergence check).
static void run_refinement_lm_gpu(const std::vector<float>& p_src, const std::vector<int32_t>& surface_id,
                                  const Plane target_planes[kNumSurfaces], uint32_t zone_mask,
                                  Rigid3 T_init, int max_iters,
                                  Rigid3& out_T, std::vector<double>& loss_history,
                                  double out_H21[21], double out_g6[6])
{
    Rigid3 T = T_init;
    double lambda = kLambdaInit;
    double H21[21], g6[6], cost;
    gpu_assemble_once(p_src, surface_id, T, target_planes, zone_mask, H21, g6, cost);
    loss_history.clear();
    loss_history.push_back(cost);

    for (int it = 0; it < max_iters; ++it) {
        double delta[6];
        bool ok = false;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
            ok = cholesky6_solve(H21, g6, lambda, delta);
            if (!ok) lambda *= kLambdaUp;
        }
        if (!ok) break;

        Rigid3 T_cand;
        retract(T, delta, T_cand);
        double H21n[21], g6n[6], cost_new;
        gpu_assemble_once(p_src, surface_id, T_cand, target_planes, zone_mask, H21n, g6n, cost_new);

        const double delta_norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] +
                                            delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);

        if (cost_new < cost) {
            const double rel_change = std::fabs(cost - cost_new) / (cost_new + 1.0e-12);
            T = T_cand;
            cost = cost_new;
            std::memcpy(H21, H21n, sizeof(H21));
            std::memcpy(g6, g6n, sizeof(g6));
            lambda *= kLambdaDown;
            if (lambda < kLambdaMin) lambda = kLambdaMin;
            loss_history.push_back(cost);
            if (delta_norm < kConvergeDeltaNorm || rel_change < kConvergeCostRel) break;
        } else {
            lambda *= kLambdaUp;
            loss_history.push_back(cost);
        }
    }

    out_T = T;
    std::memcpy(out_H21, H21, sizeof(H21));
    std::memcpy(out_g6, g6, sizeof(g6));
}

// gpu_dedup — upload xyz, run the voxel-grid dedup pipeline, download the
// kept (representative) original indices.
static std::vector<int32_t> gpu_dedup(const std::vector<float>& xyz, float cell)
{
    const int n = static_cast<int>(xyz.size() / 3);
    std::vector<int32_t> kept;
    if (n == 0) return kept;

    float* d_xyz = nullptr;
    int32_t* d_rep = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rep, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_xyz, xyz.data(), xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int num_unique = launch_dedup_voxel_grid(n, d_xyz, cell, d_rep);
    kept.resize(static_cast<size_t>(num_unique));
    if (num_unique > 0) CUDA_CHECK(cudaMemcpy(kept.data(), d_rep, kept.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_xyz));
    CUDA_CHECK(cudaFree(d_rep));
    return kept;
}

// ===========================================================================
// Small reporting / geometry helpers.
// ===========================================================================

// plane_pair_residual — the drift OBSERVABLE this whole project is built
// on (THEORY.md "The math" derives the linearized sensitivity in full):
// two fitted planes that SHOULD be the same physical surface, expressed in
// a common frame. angle_deg: the geodesic angle between their (consistently
// oriented — kernels.cuh's Plane doc) unit normals. offset_m: the
// perpendicular distance from plane B's centroid to plane A's own plane —
// a signed-then-absolute point-to-plane distance, BLIND to any purely
// in-plane offset between the two centroids (the same "a single plane
// constrains 3 of 6 DOF" fact 02.06's THEORY.md names, cited).
static void plane_pair_residual(const Plane& a, const Plane& b, float& angle_deg, float& offset_m)
{
    float dot = a.normal[0] * b.normal[0] + a.normal[1] * b.normal[1] + a.normal[2] * b.normal[2];
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    angle_deg = std::acos(dot) * (180.0f / 3.14159265358979323846f);

    const float dx = b.centroid[0] - a.centroid[0];
    const float dy = b.centroid[1] - a.centroid[1];
    const float dz = b.centroid[2] - a.centroid[2];
    offset_m = std::fabs(a.normal[0] * dx + a.normal[1] * dy + a.normal[2] * dz);
}

// condition_number — max/min diagonal eigenvalue of the (packed
// upper-triangle) 6x6 Hessian, via 01.17/01.16's jacobi_eigen_symmetric6
// (cited). Used only by the OBSERVABILITY gate.
static double condition_number(const double H21[21])
{
    double A[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) { A[i][j] = H21[hidx(i, j)]; A[j][i] = A[i][j]; }
    double eigvecs[6][6];
    jacobi_eigen_symmetric6(A, eigvecs);
    double lo = 1.0e300, hi = -1.0e300;
    for (int i = 0; i < 6; ++i) { const double v = A[i][i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo < 1.0e-9) lo = 1.0e-9;
    return hi / lo;
}

// write_ppm — a minimal hand-rolled P6 (binary RGB) writer (01.17's exact
// "hand-roll a five-line format" reasoning, cited: no vendored image lib).
static bool write_ppm(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P6\n" << w << ' ' << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return f.good();
}

// TopView — a fixed top-down (x,y) projection shared by both merged-cloud
// artifacts, so "before" and "after" are directly, pixel-for-pixel
// comparable (the money shot's whole point).
struct TopView { int w, h; float x_min, x_max, y_min, y_max; };
constexpr TopView kTopView{ 520, 520, -13.0f, 13.0f, -13.0f, 13.0f };

// kZoomView — a tight inset on wall_front's central (all-three-sensors)
// patch. The full-scene view above is honest but the drift this project
// recovers (a few CENTIMETERS) is genuinely SUB-PIXEL against a 26-METER
// scene at 520x520 -- so the "ghosting visible then gone" money shot needs
// its own window. x and y use DELIBERATELY DIFFERENT scales here (1 m of x
// spans the same pixels as 6 m of y): the interesting effect is the along-
// normal (x) separation between sensors' copies of the SAME wall, so we
// zoom hard on x and leave y wide enough to show the wall extending past
// the frame — an honest zoom (axes labeled in demo/README.md), not a
// distorted claim about the scene's true proportions.
constexpr TopView kZoomView{ 520, 520, 9.5f, 10.5f, -3.0f, 3.0f };

static void plot_point(std::vector<unsigned char>& rgb, const TopView& tv, float x, float y,
                       unsigned char r, unsigned char g, unsigned char b)
{
    // Base-frame x maps to the image's HORIZONTAL axis (+x forward = right)
    // and y maps to the VERTICAL axis, FLIPPED so image row 0 (the top) is
    // +y_max (the vehicle's LEFT) — a bird's-eye view with forward pointing
    // right and the vehicle's own left at the top of the frame. Readers
    // comparing this to kernels.cuh's separate ASCII rig diagram (drawn with
    // forward pointing up, a different but equally valid orientation choice
    // for THAT sketch) should use the axis labels, not assume the two
    // pictures share one orientation.
    const int px = static_cast<int>((x - tv.x_min) / (tv.x_max - tv.x_min) * tv.w);
    const int py = static_cast<int>((tv.y_max - y) / (tv.y_max - tv.y_min) * tv.h);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = px + dx, yy = py + dy;
            if (xx < 0 || xx >= tv.w || yy < 0 || yy >= tv.h) continue;
            const size_t idx = (static_cast<size_t>(yy) * tv.w + xx) * 3;
            rgb[idx + 0] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
        }
    }
}

static void write_topview(const std::string& path, const TopView& tv, const std::vector<float>& xyz, const std::vector<int32_t>& sensor_id)
{
    std::vector<unsigned char> rgb(static_cast<size_t>(tv.w) * tv.h * 3, 12);   // dark background
    const int n = static_cast<int>(sensor_id.size());
    for (int i = 0; i < n; ++i) {
        unsigned char r = 200, g = 200, b = 200;   // MAIN: light gray
        if (sensor_id[i] == kSensorLeft)  { r = 235; g = 70;  b = 70; }    // LEFT: red
        if (sensor_id[i] == kSensorRight) { r = 70;  g = 130; b = 235; }   // RIGHT: blue
        plot_point(rgb, tv, xyz[i * 3 + 0], xyz[i * 3 + 1], r, g, b);
    }
    write_ppm(path, tv.w, tv.h, rgb);
}

int main(int /*argc*/, char** argv)
{
    bool all_pass = true;
    std::ostringstream gates_csv;
    gates_csv << "stage,metric,value\n";

    std::printf("[demo] multi-lidar-merging-extrinsic-refinement -- project 02.16\n");
    print_device_info();

    // ---- 0) Load both cohorts ---------------------------------------------
    const std::string aligned_path = find_data_file("", argv[0], "aligned.csv");
    const std::string drifted_path = find_data_file("", argv[0], "drifted.csv");
    if (aligned_path.empty() || drifted_path.empty()) {
        std::fprintf(stderr, "could not locate data/sample/aligned.csv or drifted.csv "
                             "(run scripts/make_synthetic.py, or check your working directory)\n");
        return EXIT_FAILURE;
    }
    const Cohort aligned = load_cohort(aligned_path);
    const Cohort drifted = load_cohort(drifted_path);
    if (!aligned.loaded || !drifted.loaded) {
        std::fprintf(stderr, "failed to parse one or both cohort CSVs\n");
        return EXIT_FAILURE;
    }
    std::printf("PROBLEM: multi-LiDAR merge + extrinsic refinement, aligned_points=%d drifted_points=%d\n",
               aligned.n, drifted.n);

    // Wall-clock over the WHOLE pipeline below (every GPU stage plus its CPU
    // twin, ~a few dozen kernel launches and host round-trips): a single
    // cudaEvent GpuTimer measures ONE kernel's stream-order timeline, not a
    // sequence of independent host-orchestrated calls like this project's,
    // so CpuTimer (wall clock) is the honest choice here — util/timer.cuh's
    // header makes exactly this distinction.
    CpuTimer total_timer;
    total_timer.begin();

    const std::string out_dir = resolve_out_dir(argv[0]);

    // Per-sensor subsets, both cohorts (kernels.cuh's ZONE SETS section
    // documents which surfaces each subset actually contains).
    std::vector<float> al_main_xyz, al_left_xyz, al_right_xyz;
    std::vector<int32_t> al_main_surf, al_left_surf, al_right_surf;
    extract_sensor(aligned, kSensorMain, al_main_xyz, al_main_surf);
    extract_sensor(aligned, kSensorLeft, al_left_xyz, al_left_surf);
    extract_sensor(aligned, kSensorRight, al_right_xyz, al_right_surf);

    std::vector<float> dr_main_xyz, dr_left_xyz, dr_right_xyz;
    std::vector<int32_t> dr_main_surf, dr_left_surf, dr_right_surf;
    extract_sensor(drifted, kSensorMain, dr_main_xyz, dr_main_surf);
    extract_sensor(drifted, kSensorLeft, dr_left_xyz, dr_left_surf);
    extract_sensor(drifted, kSensorRight, dr_right_xyz, dr_right_surf);

    const Rigid3 T_main_nom = nominal_extrinsic(kSensorMain);
    const Rigid3 T_left_nom = nominal_extrinsic(kSensorLeft);
    const Rigid3 T_right_nom = nominal_extrinsic(kSensorRight);
    const Rigid3 T_left_true = true_extrinsic(kSensorLeft);     // meaningful only for the drifted cohort
    const Rigid3 T_right_true = true_extrinsic(kSensorRight);

    // =======================================================================
    // STAGE A — TRANSFORM_TWIN: the trivial merge primitive, GPU vs CPU.
    // =======================================================================
    {
        std::vector<float> gpu_out = gpu_transform(dr_left_xyz, T_left_nom);
        std::vector<float> cpu_out(dr_left_xyz.size());
        transform_points_cpu(static_cast<int>(dr_left_surf.size()), dr_left_xyz.data(), T_left_nom, cpu_out.data());
        float worst = 0.0f;
        for (size_t i = 0; i < gpu_out.size(); ++i) worst = std::max(worst, std::fabs(gpu_out[i] - cpu_out[i]));
        const bool pass = worst <= kTransformTwinTolM;
        std::printf("[info] transform twin: n=%zu worst |gpu-cpu| = %.6e m (tol %.3g m)\n",
                   dr_left_surf.size(), static_cast<double>(worst), static_cast<double>(kTransformTwinTolM));
        std::printf("TRANSFORM_TWIN: %s (GPU transform kernel matches the CPU reference)\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
        gates_csv << "transform_twin,worst_abs_diff_m," << worst << "\n";
    }

    // =======================================================================
    // STAGE B — fit every plane this project needs, GPU-driven.
    // MAIN never drifts (true == nominal in BOTH cohorts), so "MAIN's plane
    // in base frame" only ever needs T_main_nom.
    // =======================================================================
    Plane main_planes_aligned[kNumSurfaces], main_planes_drifted[kNumSurfaces];
    Plane left_planes_aligned_nom[kNumSurfaces], right_planes_aligned_nom[kNumSurfaces];
    Plane left_planes_drifted_nom[kNumSurfaces], right_planes_drifted_nom[kNumSurfaces];

    gpu_fit_planes(gpu_transform(al_main_xyz, T_main_nom), al_main_surf, main_planes_aligned);
    gpu_fit_planes(gpu_transform(dr_main_xyz, T_main_nom), dr_main_surf, main_planes_drifted);
    gpu_fit_planes(gpu_transform(al_left_xyz, T_left_nom), al_left_surf, left_planes_aligned_nom);
    gpu_fit_planes(gpu_transform(al_right_xyz, T_right_nom), al_right_surf, right_planes_aligned_nom);
    gpu_fit_planes(gpu_transform(dr_left_xyz, T_left_nom), dr_left_surf, left_planes_drifted_nom);
    gpu_fit_planes(gpu_transform(dr_right_xyz, T_right_nom), dr_right_surf, right_planes_drifted_nom);

    // =======================================================================
    // STAGE C — PLANE_FIT_TWIN: compare the GPU path above (for one
    // representative input — LEFT, drifted cohort, nominal extrinsic)
    // against the CPU-independent fit_planes_cpu on the IDENTICAL transformed
    // points, across every valid surface.
    // =======================================================================
    {
        std::vector<float> transformed = gpu_transform(dr_left_xyz, T_left_nom);   // same points BOTH paths fit
        Plane cpu_planes[kNumSurfaces];
        fit_planes_cpu(static_cast<int>(dr_left_surf.size()), transformed.data(), dr_left_surf.data(), cpu_planes);

        float worst_angle = 0.0f, worst_centroid = 0.0f;
        int compared = 0;
        for (int s = 0; s < kNumSurfaces; ++s) {
            if (!left_planes_drifted_nom[s].valid || !cpu_planes[s].valid) continue;
            ++compared;
            if (left_planes_drifted_nom[s].count != cpu_planes[s].count) worst_angle = 1.0e9f;  // count mismatch is an automatic, loud fail
            float angle_deg, offset_m;
            plane_pair_residual(left_planes_drifted_nom[s], cpu_planes[s], angle_deg, offset_m);
            worst_angle = std::max(worst_angle, angle_deg);
            const float dx = left_planes_drifted_nom[s].centroid[0] - cpu_planes[s].centroid[0];
            const float dy = left_planes_drifted_nom[s].centroid[1] - cpu_planes[s].centroid[1];
            const float dz = left_planes_drifted_nom[s].centroid[2] - cpu_planes[s].centroid[2];
            worst_centroid = std::max(worst_centroid, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        const bool pass = (compared > 0) && (worst_angle <= kPlaneFitAngleTolDeg) && (worst_centroid <= kPlaneFitCentroidTolM);
        std::printf("[info] plane fit twin: %d surfaces compared, worst normal angle %.4e deg (tol %.3g), worst centroid dev %.4e m (tol %.3g)\n",
                   compared, static_cast<double>(worst_angle), static_cast<double>(kPlaneFitAngleTolDeg),
                   static_cast<double>(worst_centroid), static_cast<double>(kPlaneFitCentroidTolM));
        std::printf("PLANE_FIT_TWIN: %s (GPU-decoded vs CPU-independent plane fit agree)\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
        gates_csv << "plane_fit_twin,worst_angle_deg," << worst_angle << "\n";
        gates_csv << "plane_fit_twin,worst_centroid_dev_m," << worst_centroid << "\n";
    }

    // =======================================================================
    // STAGE D — DRIFT_DETECTION: plane-pair residuals, aligned vs drifted.
    // Zone sets: LEFT uses {ground, wall_front, wall_left}, RIGHT uses
    // {ground, wall_front, wall_right} — kernels.cuh's "ZONE SETS".
    // =======================================================================
    const int32_t kLeftZones[3]  = { kSurfaceGround, kSurfaceWallFront, kSurfaceWallLeft };
    const int32_t kRightZones[3] = { kSurfaceGround, kSurfaceWallFront, kSurfaceWallRight };
    const char* kZoneNames[kNumSurfaces] = { "ground", "wall_front", "wall_left", "wall_right", "wall_rear", "pole" };

    std::ostringstream plane_residuals_csv;
    plane_residuals_csv << "stage,cohort,sensor,surface,angle_deg,offset_m\n";

    float aligned_worst_angle = 0.0f, aligned_worst_offset = 0.0f;
    float drifted_left_worst_angle = 0.0f, drifted_left_worst_offset = 0.0f;
    float drifted_right_worst_angle = 0.0f, drifted_right_worst_offset = 0.0f;

    for (int k = 0; k < 3; ++k) {
        const int32_t s = kLeftZones[k];
        if (main_planes_aligned[s].valid && left_planes_aligned_nom[s].valid) {
            float a, o; plane_pair_residual(main_planes_aligned[s], left_planes_aligned_nom[s], a, o);
            aligned_worst_angle = std::max(aligned_worst_angle, a);
            aligned_worst_offset = std::max(aligned_worst_offset, o);
            plane_residuals_csv << "detection,aligned,left," << kZoneNames[s] << "," << a << "," << o << "\n";
        }
        if (main_planes_drifted[s].valid && left_planes_drifted_nom[s].valid) {
            float a, o; plane_pair_residual(main_planes_drifted[s], left_planes_drifted_nom[s], a, o);
            drifted_left_worst_angle = std::max(drifted_left_worst_angle, a);
            drifted_left_worst_offset = std::max(drifted_left_worst_offset, o);
            plane_residuals_csv << "detection,drifted,left," << kZoneNames[s] << "," << a << "," << o << "\n";
        }
    }
    for (int k = 0; k < 3; ++k) {
        const int32_t s = kRightZones[k];
        if (main_planes_aligned[s].valid && right_planes_aligned_nom[s].valid) {
            float a, o; plane_pair_residual(main_planes_aligned[s], right_planes_aligned_nom[s], a, o);
            aligned_worst_angle = std::max(aligned_worst_angle, a);
            aligned_worst_offset = std::max(aligned_worst_offset, o);
            plane_residuals_csv << "detection,aligned,right," << kZoneNames[s] << "," << a << "," << o << "\n";
        }
        if (main_planes_drifted[s].valid && right_planes_drifted_nom[s].valid) {
            float a, o; plane_pair_residual(main_planes_drifted[s], right_planes_drifted_nom[s], a, o);
            drifted_right_worst_angle = std::max(drifted_right_worst_angle, a);
            drifted_right_worst_offset = std::max(drifted_right_worst_offset, o);
            plane_residuals_csv << "detection,drifted,right," << kZoneNames[s] << "," << a << "," << o << "\n";
        }
    }

    const bool aligned_below = (aligned_worst_angle <= kDriftDetectionAngleTolDeg) && (aligned_worst_offset <= kDriftDetectionOffsetTolM);
    const bool drifted_left_above = (drifted_left_worst_angle > kDriftDetectionAngleTolDeg) || (drifted_left_worst_offset > kDriftDetectionOffsetTolM);
    const bool drifted_right_above = (drifted_right_worst_angle > kDriftDetectionAngleTolDeg) || (drifted_right_worst_offset > kDriftDetectionOffsetTolM);
    const bool detection_pass = aligned_below && drifted_left_above && drifted_right_above;
    std::printf("[info] drift detection: aligned worst angle=%.4f deg offset=%.4f mm (threshold %.3g deg / %.3g mm); "
               "drifted LEFT worst angle=%.4f deg offset=%.4f mm; drifted RIGHT worst angle=%.4f deg offset=%.4f mm\n",
               static_cast<double>(aligned_worst_angle), static_cast<double>(aligned_worst_offset) * 1000.0,
               static_cast<double>(kDriftDetectionAngleTolDeg), static_cast<double>(kDriftDetectionOffsetTolM) * 1000.0,
               static_cast<double>(drifted_left_worst_angle), static_cast<double>(drifted_left_worst_offset) * 1000.0,
               static_cast<double>(drifted_right_worst_angle), static_cast<double>(drifted_right_worst_offset) * 1000.0);
    std::printf("DRIFT_DETECTION: %s (aligned rig stays under the threshold; drifted rig's LEFT and RIGHT both exceed it)\n",
               detection_pass ? "PASS" : "FAIL");
    all_pass &= detection_pass;
    gates_csv << "drift_detection,aligned_worst_angle_deg," << aligned_worst_angle << "\n";
    gates_csv << "drift_detection,drifted_left_worst_angle_deg," << drifted_left_worst_angle << "\n";
    gates_csv << "drift_detection,drifted_right_worst_angle_deg," << drifted_right_worst_angle << "\n";

    // =======================================================================
    // STAGE E — ASSEMBLY_TWIN + TRAJECTORY_TWIN (LEFT, drifted cohort, full
    // zone mask, starting from the nominal extrinsic).
    // =======================================================================
    {
        double H21_gpu[21], g6_gpu[6], cost_gpu;
        gpu_assemble_once(dr_left_xyz, dr_left_surf, T_left_nom, main_planes_drifted, kZoneMaskLeftFull, H21_gpu, g6_gpu, cost_gpu);
        double H21_cpu[21], g6_cpu[6], cost_cpu;
        assemble_point_to_plane_cpu(dr_left_xyz.data(), dr_left_surf.data(), static_cast<int>(dr_left_surf.size()),
                                    T_left_nom, main_planes_drifted, kZoneMaskLeftFull, H21_cpu, g6_cpu, &cost_cpu);
        double worst_rel = 0.0;
        for (int a = 0; a < 21; ++a) { const double scale = std::fabs(H21_cpu[a]) > 1.0 ? std::fabs(H21_cpu[a]) : 1.0; worst_rel = std::max(worst_rel, std::fabs(H21_gpu[a] - H21_cpu[a]) / scale); }
        for (int a = 0; a < 6; ++a) { const double scale = std::fabs(g6_cpu[a]) > 1.0 ? std::fabs(g6_cpu[a]) : 1.0; worst_rel = std::max(worst_rel, std::fabs(g6_gpu[a] - g6_cpu[a]) / scale); }
        const double cost_scale = std::fabs(cost_cpu) > 1.0 ? std::fabs(cost_cpu) : 1.0;
        worst_rel = std::max(worst_rel, std::fabs(cost_gpu - cost_cpu) / cost_scale);
        const bool assembly_pass = worst_rel <= kAssemblyRelTol;
        std::printf("[info] assembly twin: worst relative deviation (H/g/cost) = %.4e (tol %.3g)\n", worst_rel, kAssemblyRelTol);
        std::printf("ASSEMBLY_TWIN: %s (one-shot GPU normal-equation assembly matches the CPU oracle)\n", assembly_pass ? "PASS" : "FAIL");
        all_pass &= assembly_pass;
        gates_csv << "assembly_twin,worst_rel_dev," << worst_rel << "\n";
    }

    Rigid3 T_left_refined, T_right_refined;
    double H21_left_final[21], g6_left_final[6];
    double H21_right_final[21], g6_right_final[6];
    std::vector<double> loss_left_gpu, loss_right_gpu;
    {
        run_refinement_lm_gpu(dr_left_xyz, dr_left_surf, main_planes_drifted, kZoneMaskLeftFull, T_left_nom, kMaxLmIters,
                              T_left_refined, loss_left_gpu, H21_left_final, g6_left_final);

        std::vector<double> loss_left_cpu(kMaxLmIters + 2, 0.0);
        int num_iters_cpu = 0;
        Rigid3 T_left_cpu_traj;
        run_refinement_lm_cpu(dr_left_xyz.data(), dr_left_surf.data(), static_cast<int>(dr_left_surf.size()),
                              main_planes_drifted, kZoneMaskLeftFull, T_left_nom, kMaxLmIters,
                              T_left_cpu_traj, loss_left_cpu.data(), num_iters_cpu);

        const float rot_dev = rotation_angle_deg(T_left_refined.R, T_left_cpu_traj.R);
        const float trans_dev = translation_error_m(T_left_refined.t, T_left_cpu_traj.t);
        const bool trajectory_pass = (rot_dev <= kTrajectoryRotTolDeg) && (trans_dev <= kTrajectoryTransTolM);
        std::printf("[info] trajectory twin (LEFT, full zone set, %d GPU iters vs %d CPU iters): "
                   "final-pose deviation %.4e deg rotation, %.4e m translation (tol %.3g deg, %.3g m)\n",
                   static_cast<int>(loss_left_gpu.size()) - 1, num_iters_cpu - 1,
                   static_cast<double>(rot_dev), static_cast<double>(trans_dev),
                   static_cast<double>(kTrajectoryRotTolDeg), static_cast<double>(kTrajectoryTransTolM));
        std::printf("TRAJECTORY_TWIN: %s (GPU-orchestrated LM trajectory matches the CPU-independent trajectory)\n", trajectory_pass ? "PASS" : "FAIL");
        all_pass &= trajectory_pass;
        gates_csv << "trajectory_twin,worst_rot_dev_deg," << rot_dev << "\n";
        gates_csv << "trajectory_twin,worst_trans_dev_m," << trans_dev << "\n";
    }

    // RIGHT's refinement (needed for RECOVERY_RIGHT and the loop-consistency
    // check below; no separate twin gate — LEFT's twin above already
    // exercises the identical machinery).
    run_refinement_lm_gpu(dr_right_xyz, dr_right_surf, main_planes_drifted, kZoneMaskRightFull, T_right_nom, kMaxLmIters,
                          T_right_refined, loss_right_gpu, H21_right_final, g6_right_final);

    // =======================================================================
    // STAGE F — RECOVERY_LEFT / RECOVERY_RIGHT: refined vs TRUE drifted
    // extrinsic — the project's headline number.
    // =======================================================================
    {
        const float left_rot_err = rotation_angle_deg(T_left_refined.R, T_left_true.R);
        const float left_trans_err = translation_error_m(T_left_refined.t, T_left_true.t);
        const bool left_pass = (left_rot_err <= kRecoveryRotTolDeg) && (left_trans_err <= kRecoveryTransTolM);
        std::printf("[info] recovery LEFT: refined vs TRUE drifted extrinsic: rotation error %.4f deg, translation error %.4f mm (tol %.3g deg, %.3g mm)\n",
                   static_cast<double>(left_rot_err), static_cast<double>(left_trans_err) * 1000.0,
                   static_cast<double>(kRecoveryRotTolDeg), static_cast<double>(kRecoveryTransTolM) * 1000.0);
        std::printf("RECOVERY_LEFT: %s (LEFT's refined T_main_left recovers the true mounting drift)\n", left_pass ? "PASS" : "FAIL");
        all_pass &= left_pass;
        gates_csv << "recovery_left,rot_err_deg," << left_rot_err << "\n";
        gates_csv << "recovery_left,trans_err_m," << left_trans_err << "\n";

        const float right_rot_err = rotation_angle_deg(T_right_refined.R, T_right_true.R);
        const float right_trans_err = translation_error_m(T_right_refined.t, T_right_true.t);
        const bool right_pass = (right_rot_err <= kRecoveryRotTolDeg) && (right_trans_err <= kRecoveryTransTolM);
        std::printf("[info] recovery RIGHT: refined vs TRUE drifted extrinsic: rotation error %.4f deg, translation error %.4f mm (tol %.3g deg, %.3g mm)\n",
                   static_cast<double>(right_rot_err), static_cast<double>(right_trans_err) * 1000.0,
                   static_cast<double>(kRecoveryRotTolDeg), static_cast<double>(kRecoveryTransTolM) * 1000.0);
        std::printf("RECOVERY_RIGHT: %s (RIGHT's refined T_main_right recovers the true mounting drift)\n", right_pass ? "PASS" : "FAIL");
        all_pass &= right_pass;
        gates_csv << "recovery_right,rot_err_deg," << right_rot_err << "\n";
        gates_csv << "recovery_right,trans_err_m," << right_trans_err << "\n";
    }

    // =======================================================================
    // STAGE G — VALIDATION_LOOP: refit LEFT/RIGHT's planes using the
    // REFINED extrinsic; the residual against MAIN's (fixed, trusted) plane
    // must now fall back under the aligned-rig threshold (the closed loop).
    // =======================================================================
    {
        Plane left_planes_refined[kNumSurfaces], right_planes_refined[kNumSurfaces];
        gpu_fit_planes(gpu_transform(dr_left_xyz, T_left_refined), dr_left_surf, left_planes_refined);
        gpu_fit_planes(gpu_transform(dr_right_xyz, T_right_refined), dr_right_surf, right_planes_refined);

        float worst_angle = 0.0f, worst_offset = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const int32_t s = kLeftZones[k];
            if (main_planes_drifted[s].valid && left_planes_refined[s].valid) {
                float a, o; plane_pair_residual(main_planes_drifted[s], left_planes_refined[s], a, o);
                worst_angle = std::max(worst_angle, a);
                worst_offset = std::max(worst_offset, o);
                plane_residuals_csv << "validation,drifted,left," << kZoneNames[s] << "," << a << "," << o << "\n";
            }
        }
        for (int k = 0; k < 3; ++k) {
            const int32_t s = kRightZones[k];
            if (main_planes_drifted[s].valid && right_planes_refined[s].valid) {
                float a, o; plane_pair_residual(main_planes_drifted[s], right_planes_refined[s], a, o);
                worst_angle = std::max(worst_angle, a);
                worst_offset = std::max(worst_offset, o);
                plane_residuals_csv << "validation,drifted,right," << kZoneNames[s] << "," << a << "," << o << "\n";
            }
        }
        const bool validation_pass = (worst_angle <= kDriftDetectionAngleTolDeg) && (worst_offset <= kDriftDetectionOffsetTolM);
        std::printf("[info] validation loop: post-refinement worst plane-pair residual angle=%.4f deg offset=%.4f mm (aligned-rig threshold %.3g deg / %.3g mm)\n",
                   static_cast<double>(worst_angle), static_cast<double>(worst_offset) * 1000.0,
                   static_cast<double>(kDriftDetectionAngleTolDeg), static_cast<double>(kDriftDetectionOffsetTolM) * 1000.0);
        std::printf("VALIDATION_LOOP: %s (post-refinement residuals fall back under the aligned-rig threshold)\n", validation_pass ? "PASS" : "FAIL");
        all_pass &= validation_pass;
        gates_csv << "validation_loop,worst_angle_deg," << worst_angle << "\n";
        gates_csv << "validation_loop,worst_offset_m," << worst_offset << "\n";
    }

    // =======================================================================
    // STAGE H — OBSERVABILITY: the Hessian condition-number contrast.
    // Full zone set (ground+wall_front+wall_left, three MUTUALLY ORTHOGONAL
    // normals) vs wall_front ALONE (one normal direction) — 01.17's
    // coplanar-pose degeneracy lesson, recast for LiDAR planes (THEORY.md
    // derives why one plane leaves 3 of 6 DOF unobserved).
    // =======================================================================
    {
        double H21_degen[21], g6_degen[6], cost_degen;
        gpu_assemble_once(dr_left_xyz, dr_left_surf, T_left_refined, main_planes_drifted, kZoneMaskWallFrontOnly, H21_degen, g6_degen, cost_degen);
        const double cond_full = condition_number(H21_left_final);
        const double cond_degen = condition_number(H21_degen);
        const double ratio = cond_full > 1.0e-9 ? (cond_degen / cond_full) : 0.0;
        const bool observability_pass = ratio > kObservabilityFactor;
        std::printf("[info] observability: J^T J condition-number proxy, full zone set (ground+wall_front+wall_left) = %.4e, "
                   "wall_front-only = %.4e (ratio %.2fx, tol > %.1fx)\n",
                   cond_full, cond_degen, ratio, kObservabilityFactor);
        std::printf("OBSERVABILITY: %s (a single planar zone leaves the refinement Hessian far worse conditioned than three orthogonal zones)\n",
                   observability_pass ? "PASS" : "FAIL");
        all_pass &= observability_pass;
        gates_csv << "observability,condition_full," << cond_full << "\n";
        gates_csv << "observability,condition_degenerate," << cond_degen << "\n";
        gates_csv << "observability,ratio," << ratio << "\n";
    }

    // =======================================================================
    // STAGE I — ZERO_DRIFT_CONTROL: refining the ALIGNED (undrifted) rig
    // must not self-inflict a correction of its own.
    // =======================================================================
    {
        Rigid3 T_left_zero, T_right_zero;
        std::vector<double> loss_lz, loss_rz;
        double H21z[21], g6z[6];
        run_refinement_lm_gpu(al_left_xyz, al_left_surf, main_planes_aligned, kZoneMaskLeftFull, T_left_nom, kMaxLmIters,
                              T_left_zero, loss_lz, H21z, g6z);
        run_refinement_lm_gpu(al_right_xyz, al_right_surf, main_planes_aligned, kZoneMaskRightFull, T_right_nom, kMaxLmIters,
                              T_right_zero, loss_rz, H21z, g6z);

        const float left_rot = rotation_angle_deg(T_left_zero.R, T_left_nom.R);
        const float left_trans = translation_error_m(T_left_zero.t, T_left_nom.t);
        const float right_rot = rotation_angle_deg(T_right_zero.R, T_right_nom.R);
        const float right_trans = translation_error_m(T_right_zero.t, T_right_nom.t);
        const float worst_rot = std::max(left_rot, right_rot);
        const float worst_trans = std::max(left_trans, right_trans);
        const bool zero_drift_pass = (worst_rot <= kZeroDriftRotTolDeg) && (worst_trans <= kZeroDriftTransTolM);
        std::printf("[info] zero-drift control: refining the ALIGNED rig moved LEFT by %.5f deg / %.5f mm, RIGHT by %.5f deg / %.5f mm (tol %.3g deg / %.3g mm)\n",
                   static_cast<double>(left_rot), static_cast<double>(left_trans) * 1000.0,
                   static_cast<double>(right_rot), static_cast<double>(right_trans) * 1000.0,
                   static_cast<double>(kZeroDriftRotTolDeg), static_cast<double>(kZeroDriftTransTolM) * 1000.0);
        std::printf("ZERO_DRIFT_CONTROL: %s (no self-inflicted drift when the rig was never mis-mounted)\n", zero_drift_pass ? "PASS" : "FAIL");
        all_pass &= zero_drift_pass;
        gates_csv << "zero_drift_control,worst_rot_deg," << worst_rot << "\n";
        gates_csv << "zero_drift_control,worst_trans_m," << worst_trans << "\n";
    }

    // =======================================================================
    // STAGE J — loop consistency [info only, per the project brief]:
    // T_left_right via MAIN's composed refinements vs a DIRECT LEFT-RIGHT
    // refinement over the zones they share (ground, wall_front — kernels.cuh's
    // kZoneMaskLeftRightDirect). The full graph-consistency treatment of
    // "N independent pairwise estimates that must agree" is pose-graph
    // optimization (05.xx's job); this is the two-sensor version of the
    // question, stated didactically, not solved in general.
    // =======================================================================
    {
        Plane left_planes_own_frame[kNumSurfaces];
        gpu_fit_planes(dr_left_xyz, dr_left_surf, left_planes_own_frame);   // LEFT's own raw frame, no transform

        const Rigid3 T_init_direct = rigid3_compose(rigid3_inverse(T_left_nom), T_right_nom);
        Rigid3 T_right_from_left_direct;
        std::vector<double> loss_direct;
        double H21d[21], g6d[6];
        run_refinement_lm_gpu(dr_right_xyz, dr_right_surf, left_planes_own_frame, kZoneMaskLeftRightDirect, T_init_direct, kMaxLmIters,
                              T_right_from_left_direct, loss_direct, H21d, g6d);

        const Rigid3 T_right_from_left_via_main = rigid3_compose(rigid3_inverse(T_left_refined), T_right_refined);
        const float rot_dev = rotation_angle_deg(T_right_from_left_direct.R, T_right_from_left_via_main.R);
        const float trans_dev = translation_error_m(T_right_from_left_direct.t, T_right_from_left_via_main.t);
        std::printf("[info] loop consistency: T_left_right via main-composed refinements vs a DIRECT left-right refinement "
                   "(shared zones: ground, wall_front) deviate by %.4f deg / %.4f mm -- not gated (small-sample direct "
                   "estimate; see THEORY.md \"Where this sits in the real world\" for the full pose-graph treatment)\n",
                   static_cast<double>(rot_dev), static_cast<double>(trans_dev) * 1000.0);
        gates_csv << "loop_consistency,rot_dev_deg," << rot_dev << "\n";
        gates_csv << "loop_consistency,trans_dev_m," << trans_dev << "\n";
    }

    // =======================================================================
    // STAGE K — MERGE + DEDUP + ARTIFACTS. Merges the FULL drifted-cohort
    // cloud (all three sensors) with NOMINAL extrinsics (the "before"
    // picture the fleet's calibration currently believes) and with the
    // REFINED extrinsics (the "after" picture) -- the ghosting the money
    // shot is built to show. DEDUP_ACCOUNTING runs on the "before" cloud,
    // where duplicate/ghost points are the most pronounced.
    // =======================================================================
    const Rigid3 T_before[kNumSensors] = { T_main_nom, T_left_nom, T_right_nom };
    const Rigid3 T_after[kNumSensors]  = { T_main_nom, T_left_refined, T_right_refined };

    std::vector<float> merged_before = gpu_transform_multi(drifted.xyz, drifted.sensor_id, T_before);
    std::vector<float> merged_after  = gpu_transform_multi(drifted.xyz, drifted.sensor_id, T_after);

    write_topview(out_dir + "/topview_before.ppm", kTopView, merged_before, drifted.sensor_id);
    write_topview(out_dir + "/topview_after.ppm", kTopView, merged_after, drifted.sensor_id);
    std::printf("ARTIFACT: wrote demo/out/topview_before.ppm and demo/out/topview_after.ppm (%dx%d, "
               "gray=main red=left blue=right -- the WHOLE-SCENE view; see topview_zoom_* for the "
               "ghosting itself, which is centimeter-scale against this 26 m scene)\n",
               kTopView.w, kTopView.h);

    write_topview(out_dir + "/topview_zoom_before.ppm", kZoomView, merged_before, drifted.sensor_id);
    write_topview(out_dir + "/topview_zoom_after.ppm", kZoomView, merged_after, drifted.sensor_id);
    std::printf("ARTIFACT: wrote demo/out/topview_zoom_before.ppm and demo/out/topview_zoom_after.ppm (%dx%d, "
               "a zoomed inset on wall_front's shared patch, x in [%.1f,%.1f] m -- the MONEY SHOT: LEFT/RIGHT's "
               "red/blue dots visibly separated from MAIN's gray line before refinement, coincident after)\n",
               kZoomView.w, kZoomView.h, static_cast<double>(kZoomView.x_min), static_cast<double>(kZoomView.x_max));

    {
        std::vector<int32_t> kept_gpu = gpu_dedup(merged_before, kDedupCellM);
        std::vector<int32_t> kept_cpu(static_cast<size_t>(drifted.n));
        const int kept_cpu_n = dedup_voxel_grid_cpu(drifted.n, merged_before.data(), kDedupCellM, kept_cpu.data());
        kept_cpu.resize(static_cast<size_t>(kept_cpu_n));

        std::vector<int32_t> a = kept_gpu, b = kept_cpu;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        const bool sets_match = (a == b);
        const bool accounting_ok = (static_cast<int>(kept_gpu.size()) + (drifted.n - static_cast<int>(kept_gpu.size())) == drifted.n);
        const bool dedup_pass = sets_match && accounting_ok;

        const double inflation = kept_gpu.empty() ? 0.0 : static_cast<double>(drifted.n) / static_cast<double>(kept_gpu.size());
        std::printf("[info] dedup accounting: merged_before total=%d, GPU kept=%zu, CPU kept=%zu, kept-index sets %s; "
                   "removed=%d; raw/deduped inflation ratio = %.3fx (cell=%.3g m)\n",
                   drifted.n, kept_gpu.size(), kept_cpu.size(), sets_match ? "match" : "DIFFER",
                   drifted.n - static_cast<int>(kept_gpu.size()), inflation, static_cast<double>(kDedupCellM));
        std::printf("DEDUP_ACCOUNTING: %s (GPU and CPU voxel-grid dedup agree exactly on which points survive; kept+removed==total)\n",
                   dedup_pass ? "PASS" : "FAIL");
        all_pass &= dedup_pass;
        gates_csv << "dedup_accounting,total," << drifted.n << "\n";
        gates_csv << "dedup_accounting,kept_gpu," << kept_gpu.size() << "\n";
        gates_csv << "dedup_accounting,inflation_ratio," << inflation << "\n";
    }

    // ---- write the remaining artifacts --------------------------------------
    {
        std::ofstream f(out_dir + "/plane_residuals.csv");
        if (f.is_open()) f << plane_residuals_csv.str();
    }
    std::printf("ARTIFACT: wrote demo/out/plane_residuals.csv\n");
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        if (f.is_open()) f << gates_csv.str();
    }
    std::printf("ARTIFACT: wrote demo/out/gates_metrics.csv\n");

    // "[time]" is NOT diffed (wall-clock varies by machine and GPU). This
    // covers every stage above: ~4 plane fits, ~8 full 20-iteration LM
    // solves (each a GPU kernel launch per iteration, round-tripping
    // through gpu_assemble_once's own alloc/upload/download/free — see
    // that function's header for why this project accepts that overhead
    // for simplicity at its point-count scale), and the dedup pipeline.
    std::printf("[time] total pipeline wall-clock (every GPU/CPU stage after the CSV load through the final artifact write): %.1f ms\n", total_timer.end_ms());

    if (all_pass) {
        std::printf("RESULT: PASS (all verification stages passed -- transform/plane-fit/assembly/trajectory twins, "
                   "drift detection, both recoveries, validation loop, observability, zero-drift control, dedup accounting)\n");
        return EXIT_SUCCESS;
    }
    std::printf("RESULT: FAIL (see the stage verdict lines above for which stage(s) failed)\n");
    return EXIT_FAILURE;
}
