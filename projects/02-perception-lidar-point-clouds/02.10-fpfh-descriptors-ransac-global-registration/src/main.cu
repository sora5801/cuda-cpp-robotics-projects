// ===========================================================================
// main.cu — entry point for project 02.10 (FPFH descriptors + RANSAC global
//           registration)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed sample: 3 (source,target) scan pairs of the SAME
//      room, seen from two poses related by a known 140deg/8m transform
//      (data/README.md; scripts/make_synthetic.py).
//   2. Compute FPFH descriptors (normals -> SPFH -> FPFH, STAGES 1-3, GPU)
//      for every cloud, all 3 pairs.
//   3. VERIFY STAGE (the CLAUDE.md paragraph 5 GPU-vs-CPU gate, run on
//      pair1's source cloud + pair1's matching/RANSAC/ICP iteration 0):
//      KNN, normals, SPFH, FPFH, descriptor matching, RANSAC hypothesis
//      generation, the RANSAC refit, and the ICP point-to-plane system.
//   4. For every pair: match descriptors (STAGE 4), run the RANSAC
//      hypothesis farm + refit (STAGE 5/6), then a few point-to-plane ICP
//      iterations (STAGE 6 handoff).
//   5. INDEPENDENT GATES (the project's real teaching payoff — none of
//      these compare GPU against CPU; all compare against GROUND TRUTH or
//      a documented invariant): descriptor_invariance, registration_
//      recovery, icp_negative_control, ransac_formula, refinement_payoff,
//      prescreen_efficiency [info], low_overlap [info].
//   6. Write artifacts (demo/out/): before/after top-view PPMs, a
//      descriptor-distance-separability CSV, and gates_metrics.csv.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY(...):", "GATE ...:", "ARTIFACT:", "RESULT:" — "[info]"
// and "[time]" lines are NOT diffed (measured numbers and device names vary
// run to run / machine to machine). Change a stable line => update
// demo/expected_output.txt in the same change, and vice versa.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>

// ===========================================================================
// Tunable / documented constants — verification tolerances and gate
// thresholds. Every threshold below is measured-then-margined against an
// actual run (CLAUDE.md: "success thresholds carry wide margins" — 08.01's
// phrase, reused repo-wide) — see the [info] lines the demo prints beside
// each gate for how far under threshold the real numbers land.
// ===========================================================================

// GPU-vs-CPU VERIFY tolerances (the correctness-oracle tier).
static constexpr float kVerifyNormalDotMin      = 0.98f;  // per-point normal agreement: dot(gpu,cpu) >= this counts as "agree"
static constexpr float kVerifyNormalPassFrac    = 0.95f;  // fraction of points that must agree (STAGE 1's independent eigensolves can legitimately disagree at near-isotropic covariances)
static constexpr float kVerifySpfhAbsTol        = 0.05f;  // per-bin absolute tolerance (histograms are in [0,1]-ish ranges)
static constexpr float kVerifySpfhPassFrac      = 0.98f;
static constexpr float kVerifyFpfhAbsTol        = 0.05f;
static constexpr float kVerifyFpfhPassFrac      = 0.98f;
static constexpr float kVerifyMatchAgreeFrac    = 0.95f;  // fraction of source points whose (matched,best_idx) agree GPU vs CPU
static constexpr float kVerifyRansacAgreeFrac   = 0.95f;  // fraction of hypotheses whose (valid,inlier_count) agree GPU vs CPU
static constexpr float kVerifyRefitRotTolDeg    = 1.0f;   // real (float, shared-fn) refit vs INDEPENDENT double oracle
static constexpr float kVerifyRefitTransTolM    = 0.03f;
static constexpr double kVerifyIcpSystemRelTol  = 5.0e-3; // relative tol on the 27-entry H/g accumulator, GPU vs CPU

// Independent GATE thresholds (the ground-truth / invariant tier).
static constexpr double kGateDescInvarianceMeanL1 = 0.35;   // mean L1 distance, SAME physical point, source-frame vs target-frame FPFH
static constexpr float  kGateRegRotDeg    = 3.0f;    // registration_recovery (pair1): recovered-vs-true rotation error
static constexpr float  kGateRegTransM    = 0.20f;   // recovered-vs-true translation error
static constexpr float  kGateNegControlRotDeg  = 25.0f;  // icp_negative_control: from-identity ICP error must EXCEED this (proving failure)
static constexpr float  kGateNegControlTransM  = 1.0f;

// Scene / pipeline constants.
static constexpr uint32_t kRansacSeed = 1234567u;   // fixed seed: deterministic hypothesis farm (CLAUDE.md paragraph 12)
static constexpr float kPiDeg = 3.14159265358979323846f;

// ===========================================================================
// Small SE(3) helpers — quaternion (w,x,y,z), the repo order (CLAUDE.md
// paragraph 12). 02.06's identical machinery (Quat/quat_mul/axisangle_to_
// quat/quat_to_matrix/solve_6x6_spd), cited and reimplemented compactly for
// this project's own point-to-plane ICP handoff (STAGE 6).
// ---------------------------------------------------------------------------
struct Quat { float w, x, y, z; };   // unit quaternion, repo (w,x,y,z) order — kept normalized after every update

// quat_normalize — rescale q to unit length. Guards the zero-quaternion edge
// case (returns q unchanged rather than dividing by zero) — should never
// fire in practice (every caller feeds it an already-near-unit input) but
// costs nothing to check and turns a NaN-propagation bug into a no-op.
static Quat quat_normalize(Quat q)
{
    const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    const float inv = (n > 0.0f) ? (1.0f / n) : 1.0f;
    return { q.w * inv, q.x * inv, q.y * inv, q.z * inv };
}

// quat_mul — Hamilton product a*b: composes two rotations, applying b FIRST
// then a (i.e. the combined rotation is "a after b"). run_icp_gpu uses this
// to LEFT-multiply each iteration's small-angle increment onto the running
// estimate: q_est <- q_delta * q_est (02.06's convention, cited).
static Quat quat_mul(Quat a, Quat b)
{
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

// quat_to_matrix — the standard quaternion -> row-major 3x3 rotation matrix
// formula (every kernel below wants R, not q, since it needs R*p+t per
// point — kernels.cuh's Rigid3 comment explains why kernels see the derived
// matrix rather than the quaternion itself).
static void quat_to_matrix(Quat q, float R[9])
{
    const float w = q.w, x = q.x, y = q.y, z = q.z;
    R[0] = 1.0f - 2.0f * (y * y + z * z);  R[1] = 2.0f * (x * y - w * z);         R[2] = 2.0f * (x * z + w * y);
    R[3] = 2.0f * (x * y + w * z);         R[4] = 1.0f - 2.0f * (x * x + z * z);  R[5] = 2.0f * (y * z - w * x);
    R[6] = 2.0f * (x * z - w * y);         R[7] = 2.0f * (y * z + w * x);         R[8] = 1.0f - 2.0f * (x * x + y * y);
}

// Shepperd's method: robust rotation matrix -> quaternion (needed to seed
// q_est from the RANSAC-supplied initial R at the start of run_icp_gpu).
static Quat matrix_to_quat(const float R[9])
{
    const float m00 = R[0], m01 = R[1], m02 = R[2];
    const float m10 = R[3], m11 = R[4], m12 = R[5];
    const float m20 = R[6], m21 = R[7], m22 = R[8];
    const float tr = m00 + m11 + m22;
    Quat q;
    if (tr > 0.0f) {
        const float S = std::sqrt(tr + 1.0f) * 2.0f;
        q = { 0.25f * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S };
    } else if (m00 > m11 && m00 > m22) {
        const float S = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q = { (m21 - m12) / S, 0.25f * S, (m01 + m10) / S, (m02 + m20) / S };
    } else if (m11 > m22) {
        const float S = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q = { (m02 - m20) / S, (m01 + m10) / S, 0.25f * S, (m12 + m21) / S };
    } else {
        const float S = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q = { (m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25f * S };
    }
    return quat_normalize(q);
}

// axisangle_to_quat — the SE(3) exponential map's rotation half, EXACT (not
// a first-order approximation) via Rodrigues' formula in half-angle form:
// q = (cos(theta/2), sin(theta/2)*axis). theta = |w| can be non-tiny on an
// early ICP iteration (this project's RANSAC handoff can still leave a
// fraction of a degree of residual rotation), so the exact trig form is
// used whenever theta clears the near-zero guard; below that, a first-order
// fallback avoids a 0/0 division (dividing by theta when theta~=0).
static Quat axisangle_to_quat(float wx, float wy, float wz)
{
    const float theta = std::sqrt(wx * wx + wy * wy + wz * wz);
    if (theta < 1e-8f) return quat_normalize(Quat{ 1.0f, wx * 0.5f, wy * 0.5f, wz * 0.5f });
    const float half = 0.5f * theta;
    const float s = std::sin(half) / theta;   // sin(half)/theta, not sin(half)/|axis| -- axis = w/theta, so this is sin(half)*axis in one division
    return Quat{ std::cos(half), wx * s, wy * s, wz * s };
}

// rotation_angle_deg_from_matrix — the angle of a rotation MATRIX via the
// standard trace identity trace(R) = 1 + 2*cos(theta). Used by
// rotation_error_deg below to turn a RELATIVE rotation (recovered vs. true,
// or hypothesis vs. oracle) into one scalar degrees figure every GATE
// prints.
static float rotation_angle_deg_from_matrix(const float R[9])
{
    float c = (R[0] + R[4] + R[8] - 1.0f) * 0.5f;
    if (c > 1.0f) c = 1.0f;      // clamp against float rounding pushing |c| a hair past 1 (acos would return NaN otherwise)
    if (c < -1.0f) c = -1.0f;
    return std::acos(c) * (180.0f / kPiDeg);
}

// mat3_mul — plain 3x3 row-major matrix product C = A*B (the textbook
// triple loop; n=3 is far too small to matter for anything fancier).
static void mat3_mul(const float A[9], const float B[9], float C[9])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = acc;
        }
}

// mat3_transpose — At = A^T. For a ROTATION matrix this equals the inverse
// (orthonormality: R^T*R = I), which is exactly how rotation_error_deg uses
// it below (Rb^T undoes Rb's rotation).
static void mat3_transpose(const float A[9], float At[9])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            At[i * 3 + j] = A[j * 3 + i];
}

// rotation_error_deg / translation_error_m — the ground-truth pose-error
// metric every independent GATE below uses: angle of (R_a * R_b^T) via the
// trace identity, and plain Euclidean distance of the translations.
static float rotation_error_deg(const float Ra[9], const float Rb[9])
{
    float Rbt[9]; mat3_transpose(Rb, Rbt);
    float Re[9]; mat3_mul(Ra, Rbt, Re);
    return rotation_angle_deg_from_matrix(Re);
}
static float translation_error_m(const float ta[3], const float tb[3])
{
    return std::sqrt(squared_distance3(ta, tb));
}

// solve_6x6_spd / expand_h — 33.01/02.06-style Cholesky solve for the 6x6
// point-to-plane normal system, ported compactly (cited, not copy-pasted
// verbatim — same three steps: factorize, forward-solve, back-solve).
// solve_6x6_spd — Cholesky solve of H*x=rhs for a 6x6 SYMMETRIC POSITIVE-
// DEFINITE (or near-SPD, hence the damping) system: factorize H+damping*I =
// L*L^T (lower-triangular, positive diagonal), forward-solve L*y=rhs,
// back-solve L^T*x=y. `damping` (kIcpDampingLM) is added to the diagonal
// BEFORE factorizing — the classic Levenberg-Marquardt "H + lambda*I"
// regularization against a near-singular system late in convergence (few
// correspondences left, or a nearly-degenerate point-to-plane geometry).
// Returns false (and fills x with NaN, never a silently-wrong number) if
// any pivot goes non-positive — an unmissable signal downstream that the
// solve failed, rather than a plausible-looking garbage answer.
static bool solve_6x6_spd(const double H[36], const double rhs[6], double damping, double x[6])
{
    double m[36];   // H's lower triangle on entry -> L on exit (in-place factorization)
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            m[i * 6 + j] = (j <= i) ? H[i * 6 + j] : 0.0;
    for (int i = 0; i < 6; ++i) m[i * 6 + i] += damping;

    bool spd = true;
    for (int j = 0; j < 6; ++j) {
        // Standard Cholesky column-by-column: subtract off the contribution
        // of already-factored columns before taking the sqrt of the pivot.
        double diag = m[j * 6 + j];
        for (int p = 0; p < j; ++p) diag -= m[j * 6 + p] * m[j * 6 + p];
        spd = spd && (diag > 0.0);   // ONE non-positive pivot anywhere fails the whole solve
        const double ljj = std::sqrt(diag > 0.0 ? diag : 0.0);
        m[j * 6 + j] = ljj;
        const double inv_ljj = (ljj > 0.0) ? (1.0 / ljj) : 0.0;
        for (int i = j + 1; i < 6; ++i) {
            double s = m[i * 6 + j];
            for (int p = 0; p < j; ++p) s -= m[i * 6 + p] * m[j * 6 + p];
            m[i * 6 + j] = s * inv_ljj;
        }
    }
    double v[6];
    for (int i = 0; i < 6; ++i) v[i] = rhs[i];
    for (int i = 0; i < 6; ++i) {   // forward: L*y = rhs
        double s = v[i];
        for (int p = 0; p < i; ++p) s -= m[i * 6 + p] * v[p];
        v[i] = s / m[i * 6 + i];
    }
    for (int i = 5; i >= 0; --i) {   // back: L^T*x = y
        double s = v[i];
        for (int p = i + 1; p < 6; ++p) s -= m[p * 6 + i] * v[p];
        v[i] = s / m[i * 6 + i];
    }
    for (int i = 0; i < 6; ++i) x[i] = spd ? v[i] : std::numeric_limits<double>::quiet_NaN();
    return spd;
}

// expand_h — unpack the 21-entry upper-triangle (kernels.cuh's hidx layout,
// [wx,wy,wz,vx,vy,vz] order) into the full symmetric 6x6 solve_6x6_spd
// wants. The lower-left block is JUST the transpose of the upper-right
// block by construction (H = J^T*J is always symmetric), so this is a pure
// copy, not a computation.
static void expand_h(const double H21[21], double Hfull[36])
{
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) {
            const double val = H21[hidx(i, j)];
            Hfull[i * 6 + j] = val;
            Hfull[j * 6 + i] = val;
        }
}

// ---------------------------------------------------------------------------
// Data loading — the binary cloud format (scripts/make_synthetic.py's
// module docstring documents the byte layout field-for-field; data/README.md
// mirrors it) and pairs_meta.csv.
// ---------------------------------------------------------------------------
struct CloudHost {
    std::vector<float> xyz;
    std::vector<int32_t> world_idx;
    int n = 0;
};

static bool load_cloud_bin(const std::string& path, CloudHost& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    char magic[9];
    f.read(magic, 9);
    if (f.gcount() != 9 || std::memcmp(magic, "FPFHPAIR1", 9) != 0) return false;
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n <= 0) return false;
    out.n = n;
    out.xyz.assign(static_cast<size_t>(n) * 3, 0.0f);
    const std::streamsize want_xyz = static_cast<std::streamsize>(out.xyz.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(out.xyz.data()), want_xyz);
    if (f.gcount() != want_xyz) return false;
    out.world_idx.assign(static_cast<size_t>(n), 0);
    const std::streamsize want_idx = static_cast<std::streamsize>(out.world_idx.size() * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(out.world_idx.data()), want_idx);
    return f.gcount() == want_idx;
}

struct PairMeta {
    std::string name;
    int n_source = 0, n_target = 0;
    float overlap_fraction = 0.0f;
    float noise_sigma_m = 0.0f;
    float t_true[3] = { 0, 0, 0 };
    float q_true[4] = { 1, 0, 0, 0 };   // (w,x,y,z)
    float relative_yaw_deg = 0.0f;
    float relative_trans_m = 0.0f;
};

static std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

static bool load_pairs_meta(const std::string& path, std::vector<PairMeta>& out)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    bool header_skipped = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (!header_skipped) { header_skipped = true; continue; }   // skip the column-name header row
        const auto cells = split_csv_line(line);
        if (cells.size() < 14) return false;
        PairMeta m;
        m.name = cells[0];
        m.n_source = std::atoi(cells[1].c_str());
        m.n_target = std::atoi(cells[2].c_str());
        m.overlap_fraction = std::strtof(cells[3].c_str(), nullptr);
        m.noise_sigma_m = std::strtof(cells[4].c_str(), nullptr);
        m.t_true[0] = std::strtof(cells[5].c_str(), nullptr);
        m.t_true[1] = std::strtof(cells[6].c_str(), nullptr);
        m.t_true[2] = std::strtof(cells[7].c_str(), nullptr);
        m.q_true[0] = std::strtof(cells[8].c_str(), nullptr);
        m.q_true[1] = std::strtof(cells[9].c_str(), nullptr);
        m.q_true[2] = std::strtof(cells[10].c_str(), nullptr);
        m.q_true[3] = std::strtof(cells[11].c_str(), nullptr);
        m.relative_yaw_deg = std::strtof(cells[12].c_str(), nullptr);
        m.relative_trans_m = std::strtof(cells[13].c_str(), nullptr);
        out.push_back(m);
    }
    return !out.empty();
}

// ---------------------------------------------------------------------------
// centroid3 — mean of an interleaved xyz cloud (host, double accumulation
// for precision — the SAME "orient toward an interior reference point"
// input STAGE 1's normal kernel needs; 02.06's identical centroid role,
// cited, computed on the HOST here rather than a GPU reduction: n is a few
// thousand, a single sequential pass is instant and needs no kernel).
// ---------------------------------------------------------------------------
static void centroid3(const std::vector<float>& xyz, int n, float out[3])
{
    double sx = 0, sy = 0, sz = 0;
    for (int i = 0; i < n; ++i) { sx += xyz[i * 3 + 0]; sy += xyz[i * 3 + 1]; sz += xyz[i * 3 + 2]; }
    out[0] = static_cast<float>(sx / n); out[1] = static_cast<float>(sy / n); out[2] = static_cast<float>(sz / n);
}

// ===========================================================================
// STAGE 1-3 — descriptor pipeline: upload xyz once, run KNN -> normals ->
// SPFH -> FPFH on the GPU, download every intermediate array to host. Data
// sizes here (a few thousand points x <= 33 floats) make repeated H2D/D2H
// transfers between stages irrelevant to runtime and keep every stage
// function fully self-contained (CLAUDE.md's "narrate the thought process":
// the alternative — threading device pointers through every caller — buys
// speed this project's scale does not need, at a real readability cost).
// ---------------------------------------------------------------------------
struct Descriptors {
    std::vector<int32_t> neighbor_ids;
    std::vector<float> neighbor_dist;
    std::vector<float> normal;
    std::vector<float> spfh;
    std::vector<float> fpfh;
};

static Descriptors compute_descriptors_gpu(int n, const std::vector<float>& h_xyz, const float ref[3])
{
    Descriptors d;
    d.neighbor_ids.assign(static_cast<size_t>(n) * kFpfhK, 0);
    d.neighbor_dist.assign(static_cast<size_t>(n) * kFpfhK, 0.0f);
    d.normal.assign(static_cast<size_t>(n) * 3, 0.0f);
    d.spfh.assign(static_cast<size_t>(n) * kFpfhDim, 0.0f);
    d.fpfh.assign(static_cast<size_t>(n) * kFpfhDim, 0.0f);

    const size_t bytes_xyz = static_cast<size_t>(n) * 3 * sizeof(float);
    float* d_xyz = nullptr; int32_t* d_ids = nullptr; float* d_dist = nullptr;
    float* d_normal = nullptr; float* d_spfh = nullptr; float* d_fpfh = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, bytes_xyz));
    CUDA_CHECK(cudaMalloc(&d_ids, static_cast<size_t>(n) * kFpfhK * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_dist, static_cast<size_t>(n) * kFpfhK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normal, bytes_xyz));
    CUDA_CHECK(cudaMalloc(&d_spfh, static_cast<size_t>(n) * kFpfhDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fpfh, static_cast<size_t>(n) * kFpfhDim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), bytes_xyz, cudaMemcpyHostToDevice));

    launch_knn_search(n, d_xyz, d_ids, d_dist);
    launch_estimate_normals(n, d_xyz, d_ids, ref[0], ref[1], ref[2], d_normal);
    launch_compute_spfh(n, d_xyz, d_normal, d_ids, d_spfh);
    launch_compute_fpfh(n, d_spfh, d_ids, d_dist, d_fpfh);

    CUDA_CHECK(cudaMemcpy(d.neighbor_ids.data(), d_ids, d.neighbor_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d.neighbor_dist.data(), d_dist, d.neighbor_dist.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d.normal.data(), d_normal, d.normal.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d.spfh.data(), d_spfh, d.spfh.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d.fpfh.data(), d_fpfh, d.fpfh.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_xyz)); CUDA_CHECK(cudaFree(d_ids)); CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_normal)); CUDA_CHECK(cudaFree(d_spfh)); CUDA_CHECK(cudaFree(d_fpfh));
    return d;
}

// STAGE 4 — descriptor matching (GPU), self-contained the same way.
struct MatchResult {
    std::vector<uint8_t> matched;
    std::vector<int32_t> best_idx;
    std::vector<float> dist1_sq, dist2_sq;
};

static MatchResult match_gpu(int n_src, const std::vector<float>& fpfh_src, int n_tgt, const std::vector<float>& fpfh_tgt)
{
    MatchResult r;
    r.matched.assign(static_cast<size_t>(n_src), 0);
    r.best_idx.assign(static_cast<size_t>(n_src), -1);
    r.dist1_sq.assign(static_cast<size_t>(n_src), 0.0f);
    r.dist2_sq.assign(static_cast<size_t>(n_src), 0.0f);

    float* d_src = nullptr; float* d_tgt = nullptr;
    uint8_t* d_matched = nullptr; int32_t* d_best = nullptr; float* d_d1 = nullptr; float* d_d2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, fpfh_src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt, fpfh_tgt.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_matched, static_cast<size_t>(n_src) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_best, static_cast<size_t>(n_src) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_d1, static_cast<size_t>(n_src) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d2, static_cast<size_t>(n_src) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, fpfh_src.data(), fpfh_src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt, fpfh_tgt.data(), fpfh_tgt.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_match_correspondences(n_src, d_src, n_tgt, d_tgt, d_matched, d_best, d_d1, d_d2);

    CUDA_CHECK(cudaMemcpy(r.matched.data(), d_matched, r.matched.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.best_idx.data(), d_best, r.best_idx.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.dist1_sq.data(), d_d1, r.dist1_sq.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.dist2_sq.data(), d_d2, r.dist2_sq.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_tgt));
    CUDA_CHECK(cudaFree(d_matched)); CUDA_CHECK(cudaFree(d_best)); CUDA_CHECK(cudaFree(d_d1)); CUDA_CHECK(cudaFree(d_d2));
    return r;
}

// STAGE 5 — the RANSAC hypothesis farm (GPU).
struct RansacResult {
    std::vector<uint8_t> valid;
    std::vector<Rigid3> transform;
    std::vector<int32_t> inlier_count;
};

static RansacResult ransac_gpu(int nc, const std::vector<float>& corr_src_xyz, const std::vector<float>& corr_tgt_xyz,
                               uint32_t seed, int k)
{
    RansacResult r;
    r.valid.assign(static_cast<size_t>(k), 0);
    r.transform.assign(static_cast<size_t>(k), Rigid3{});
    r.inlier_count.assign(static_cast<size_t>(k), 0);

    float* d_src = nullptr; float* d_tgt = nullptr;
    uint8_t* d_valid = nullptr; Rigid3* d_T = nullptr; int32_t* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, corr_src_xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt, corr_tgt_xyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<size_t>(k) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_T, static_cast<size_t>(k) * sizeof(Rigid3)));
    CUDA_CHECK(cudaMalloc(&d_count, static_cast<size_t>(k) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_src, corr_src_xyz.data(), corr_src_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt, corr_tgt_xyz.data(), corr_tgt_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_ransac_hypotheses(nc, d_src, d_tgt, seed, k, d_valid, d_T, d_count);

    CUDA_CHECK(cudaMemcpy(r.valid.data(), d_valid, r.valid.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.transform.data(), d_T, r.transform.size() * sizeof(Rigid3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.inlier_count.data(), d_count, r.inlier_count.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_tgt));
    CUDA_CHECK(cudaFree(d_valid)); CUDA_CHECK(cudaFree(d_T)); CUDA_CHECK(cudaFree(d_count));
    return r;
}

// select_best_hypothesis — argmax inlier count over VALID hypotheses, ties
// toward the lowest index (02.03's identical deterministic tie-break).
static int select_best_hypothesis(const std::vector<uint8_t>& valid, const std::vector<int32_t>& count)
{
    int best = -1, best_count = -1;
    for (size_t i = 0; i < valid.size(); ++i) {
        if (!valid[i]) continue;
        if (count[i] > best_count) { best_count = count[i]; best = static_cast<int>(i); }
    }
    return best;
}

// STAGE 6 — point-to-plane ICP handoff (GPU per-iteration kernels, host
// SE(3) bookkeeping — 02.06's identical split, cited).
struct IcpRunResult {
    Rigid3 T_final{};
    int iters_run = 0;
    double accum27_iter0[27] = { 0 };
    std::vector<int32_t> corr_idx_iter0;
    std::vector<float> cur_xyz_iter0;
};

static IcpRunResult run_icp_gpu(Rigid3 T_init, int n_src, const std::vector<float>& h_src_xyz,
                                int n_tgt, const std::vector<float>& h_tgt_xyz, const std::vector<float>& h_tgt_normal,
                                bool capture_iter0)
{
    IcpRunResult out;
    Quat q_est = matrix_to_quat(T_init.R);
    float t_est[3] = { T_init.t[0], T_init.t[1], T_init.t[2] };

    float* d_src_xyz = nullptr; float* d_tgt_xyz = nullptr; float* d_tgt_normal = nullptr;
    float* d_cur_xyz = nullptr; int32_t* d_corr_idx = nullptr; float* d_corr_dist2 = nullptr; double* d_accum27 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src_xyz, static_cast<size_t>(n_src) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_xyz, static_cast<size_t>(n_tgt) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_normal, static_cast<size_t>(n_tgt) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cur_xyz, static_cast<size_t>(n_src) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr_idx, static_cast<size_t>(n_src) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_corr_dist2, static_cast<size_t>(n_src) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accum27, 27 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_src_xyz, h_src_xyz.data(), static_cast<size_t>(n_src) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_xyz, h_tgt_xyz.data(), static_cast<size_t>(n_tgt) * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_normal, h_tgt_normal.data(), static_cast<size_t>(n_tgt) * 3 * sizeof(float), cudaMemcpyHostToDevice));

    for (int iter = 0; iter < kIcpMaxIters; ++iter) {
        Rigid3 T; quat_to_matrix(q_est, T.R); T.t[0] = t_est[0]; T.t[1] = t_est[1]; T.t[2] = t_est[2];

        launch_transform_cloud(n_src, d_src_xyz, T, d_cur_xyz);
        launch_icp_correspondences(n_src, d_cur_xyz, n_tgt, d_tgt_xyz, kIcpMaxCorrDistM, d_corr_idx, d_corr_dist2);
        launch_icp_accumulate(n_src, d_cur_xyz, d_tgt_xyz, d_tgt_normal, d_corr_idx, d_accum27);

        double accum27[27];
        CUDA_CHECK(cudaMemcpy(accum27, d_accum27, 27 * sizeof(double), cudaMemcpyDeviceToHost));

        if (iter == 0 && capture_iter0) {
            std::memcpy(out.accum27_iter0, accum27, sizeof(accum27));
            out.corr_idx_iter0.assign(static_cast<size_t>(n_src), -1);
            CUDA_CHECK(cudaMemcpy(out.corr_idx_iter0.data(), d_corr_idx, static_cast<size_t>(n_src) * sizeof(int32_t), cudaMemcpyDeviceToHost));
            out.cur_xyz_iter0.assign(static_cast<size_t>(n_src) * 3, 0.0f);
            CUDA_CHECK(cudaMemcpy(out.cur_xyz_iter0.data(), d_cur_xyz, out.cur_xyz_iter0.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Gauss-Newton step: minimize sum_i (J_i.delta + e_i)^2 over the
        // linearized residuals accumulated into H=sum(J^T J), g=sum(J^T e)
        // by icp_accumulate_kernel (kernels.cu's e = point-to-plane
        // residual). The normal equations are H*delta = -g (THEORY.md "The
        // math" derives this from setting d/d(delta) of the sum-of-squares
        // to zero) — so the RHS passed to the solver is g NEGATED.
        double H21[21], g6[6];
        for (int i = 0; i < 21; ++i) H21[i] = accum27[i];
        for (int i = 0; i < 6; ++i) g6[i] = -accum27[21 + i];   // RHS = -g
        double Hfull[36]; expand_h(H21, Hfull);
        double delta[6];
        const bool spd = solve_6x6_spd(Hfull, g6, kIcpDampingLM, delta);
        out.iters_run = iter + 1;
        if (!spd) break;   // singular/degenerate system (e.g. too few correspondences): stop, report what we have

        const float wx = static_cast<float>(delta[0]), wy = static_cast<float>(delta[1]), wz = static_cast<float>(delta[2]);
        const float vx = static_cast<float>(delta[3]), vy = static_cast<float>(delta[4]), vz = static_cast<float>(delta[5]);

        // SE(3) update — 02.06's convention, cited and reused UNCHANGED for
        // correctness (THEORY.md derives the linearization): R_est <-
        // Exp(w)*R_est (LEFT-composed, exact via the quaternion exponential
        // map — no small-angle approximation needed even for a large first
        // step); t_est <- t_est + v (an additive translation update that is
        // a documented first-order simplification 02.06 also makes — later
        // iterations correct any residual O(|w|*|t|) error this leaves).
        const Quat q_delta = axisangle_to_quat(wx, wy, wz);
        q_est = quat_normalize(quat_mul(q_delta, q_est));
        t_est[0] += vx; t_est[1] += vy; t_est[2] += vz;

        const float rot_delta_deg = std::sqrt(wx * wx + wy * wy + wz * wz) * (180.0f / kPiDeg);
        const float trans_delta_m = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (rot_delta_deg < kIcpConvRotDeg && trans_delta_m < kIcpConvTransM) break;
    }

    CUDA_CHECK(cudaFree(d_src_xyz)); CUDA_CHECK(cudaFree(d_tgt_xyz)); CUDA_CHECK(cudaFree(d_tgt_normal));
    CUDA_CHECK(cudaFree(d_cur_xyz)); CUDA_CHECK(cudaFree(d_corr_idx)); CUDA_CHECK(cudaFree(d_corr_dist2));
    CUDA_CHECK(cudaFree(d_accum27));

    quat_to_matrix(q_est, out.T_final.R);
    out.T_final.t[0] = t_est[0]; out.T_final.t[1] = t_est[1]; out.T_final.t[2] = t_est[2];
    return out;
}

// ---------------------------------------------------------------------------
// PPM (P6) writer — the repo's "no external image library" convention
// (02.03/02.09's identical choice): a raw, fully commented raster writer.
// ---------------------------------------------------------------------------
static void write_ppm(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

// rasterize_topview — plot two point sets (x,y projected, z ignored) into
// one W x H image with a shared world-to-pixel scale, each set in its own
// color, each point drawn as a small filled square (so isolated points
// stay visible at this resolution). The classic before/after registration
// visualization (PCL/Open3D tutorials all show this exact plot — README
// "Prior art").
// ---------------------------------------------------------------------------
static void rasterize_topview(const std::string& path, int w, int h,
                              const std::vector<float>& pts_a, int n_a, unsigned char ca[3],
                              const std::vector<float>& pts_b, int n_b, unsigned char cb[3])
{
    float min_x = 3.0e38f, max_x = -3.0e38f, min_y = 3.0e38f, max_y = -3.0e38f;
    auto scan = [&](const std::vector<float>& p, int n) {
        for (int i = 0; i < n; ++i) {
            const float x = p[i * 3 + 0], y = p[i * 3 + 1];
            min_x = std::min(min_x, x); max_x = std::max(max_x, x);
            min_y = std::min(min_y, y); max_y = std::max(max_y, y);
        }
    };
    scan(pts_a, n_a); scan(pts_b, n_b);
    const float pad = 1.0f;
    min_x -= pad; max_x += pad; min_y -= pad; max_y += pad;
    const float span_x = std::max(1e-3f, max_x - min_x);
    const float span_y = std::max(1e-3f, max_y - min_y);
    const float scale = std::min(static_cast<float>(w - 1) / span_x, static_cast<float>(h - 1) / span_y);

    // Rasterize each cloud into its OWN boolean coverage mask first (rather
    // than painting directly into one shared image) so we can tell apart
    // "only A landed here", "only B landed here", and "BOTH landed here" —
    // the last case is exactly what a SUCCESSFUL alignment produces at high
    // density (corresponding points differ by only ~1 cm of sensor noise,
    // a fraction of one pixel at this image's world-to-pixel scale), and a
    // naive "paint B over A" approach would silently HIDE that overlap
    // behind whichever cloud is drawn last — the opposite of what a
    // before/after registration plot exists to show a learner.
    std::vector<uint8_t> hit_a(static_cast<size_t>(w) * h, 0);
    std::vector<uint8_t> hit_b(static_cast<size_t>(w) * h, 0);
    auto mark = [&](const std::vector<float>& p, int n, std::vector<uint8_t>& mask) {
        for (int i = 0; i < n; ++i) {
            const int px = static_cast<int>((p[i * 3 + 0] - min_x) * scale);
            const int py = h - 1 - static_cast<int>((p[i * 3 + 1] - min_y) * scale);   // flip Y: image row 0 is TOP
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = px + dx, y = py + dy;
                    if (x < 0 || x >= w || y < 0 || y >= h) continue;
                    mask[static_cast<size_t>(y) * w + x] = 1;
                }
        }
    };
    mark(pts_a, n_a, hit_a);
    mark(pts_b, n_b, hit_b);

    // Compose the final image from the two masks: A-only -> ca, B-only ->
    // cb, BOTH -> the average of ca/cb (this project's red+blue choice
    // averages to a visible purple/magenta — a THIRD color that exists
    // nowhere else in the image, so "purple" unambiguously means "the two
    // clouds agree here").
    std::vector<unsigned char> img(static_cast<size_t>(w) * h * 3, 250);   // near-white background
    for (size_t o = 0; o < hit_a.size(); ++o) {
        const bool a = hit_a[o] != 0, b = hit_b[o] != 0;
        if (a && b) {
            img[o * 3 + 0] = static_cast<unsigned char>((ca[0] + cb[0]) / 2);
            img[o * 3 + 1] = static_cast<unsigned char>((ca[1] + cb[1]) / 2);
            img[o * 3 + 2] = static_cast<unsigned char>((ca[2] + cb[2]) / 2);
        } else if (a) {
            img[o * 3 + 0] = ca[0]; img[o * 3 + 1] = ca[1]; img[o * 3 + 2] = ca[2];
        } else if (b) {
            img[o * 3 + 0] = cb[0]; img[o * 3 + 1] = cb[1]; img[o * 3 + 2] = cb[2];
        }
    }
    write_ppm(path, w, h, img);
}

// ===========================================================================
// descriptor_invariance — THE property gate: for every physically identical
// point present in BOTH clouds (identified by the GROUND-TRUTH world_idx,
// never by the algorithm's own matching), compare FPFH computed
// INDEPENDENTLY in each cloud's own local frame. THEORY.md "The math"
// proves this should be small; this function MEASURES it.
// ---------------------------------------------------------------------------
struct InvarianceStat { int n_pairs = 0; double mean_l1 = 0.0; double max_l1 = 0.0; };

static InvarianceStat descriptor_invariance(const std::vector<int32_t>& src_world_idx, const std::vector<float>& fpfh_src,
                                            const std::vector<int32_t>& tgt_world_idx, const std::vector<float>& fpfh_tgt)
{
    std::unordered_map<int32_t, int32_t> tgt_map;
    tgt_map.reserve(tgt_world_idx.size() * 2);
    for (size_t j = 0; j < tgt_world_idx.size(); ++j) tgt_map[tgt_world_idx[j]] = static_cast<int32_t>(j);

    InvarianceStat s;
    double sum = 0.0;
    for (size_t i = 0; i < src_world_idx.size(); ++i) {
        const auto it = tgt_map.find(src_world_idx[i]);
        if (it == tgt_map.end()) continue;
        const int32_t j = it->second;
        double l1 = 0.0;
        for (int b = 0; b < kFpfhDim; ++b)
            l1 += std::fabs(static_cast<double>(fpfh_src[i * kFpfhDim + b]) - static_cast<double>(fpfh_tgt[static_cast<size_t>(j) * kFpfhDim + b]));
        sum += l1;
        s.max_l1 = std::max(s.max_l1, l1);
        s.n_pairs++;
    }
    s.mean_l1 = (s.n_pairs > 0) ? (sum / s.n_pairs) : 0.0;
    return s;
}

// ===========================================================================
// A tiny CSV metrics ledger — every VERIFY/GATE/[info] number this demo
// prints also lands in demo/out/gates_metrics.csv (02.03/02.09's identical
// artifact, cited), the machine-readable twin of the stdout report.
// ---------------------------------------------------------------------------
struct MetricRow { std::string gate, metric, value, threshold, status; };
static std::vector<MetricRow> g_metrics;
static void record_metric(const std::string& gate, const std::string& metric, double value,
                          const std::string& threshold, const std::string& status)
{
    std::ostringstream vs; vs.precision(6); vs << value;
    g_metrics.push_back({ gate, metric, vs.str(), threshold, status });
}

int main(int argc, char** argv)
{
    bool all_ok = true;   // ANDed with every VERIFY/GATE result; drives the final RESULT: line

    std::printf("[demo] 02.10 FPFH descriptors + RANSAC global registration\n");
    print_device_info();

    // ---- 1) Load data -------------------------------------------------------
    const std::string meta_path = find_data_file("", argv[0], "pairs_meta.csv");
    if (meta_path.empty()) {
        std::fprintf(stderr, "ERROR: could not locate data/sample/pairs_meta.csv (run scripts/make_synthetic.py)\n");
        return EXIT_FAILURE;
    }
    std::vector<PairMeta> metas;
    if (!load_pairs_meta(meta_path, metas) || metas.size() < 3) {
        std::fprintf(stderr, "ERROR: failed to parse %s\n", meta_path.c_str());
        return EXIT_FAILURE;
    }
    const std::string data_dir = meta_path.substr(0, meta_path.find_last_of("/\\"));

    std::vector<CloudHost> sources(metas.size()), targets(metas.size());
    for (size_t p = 0; p < metas.size(); ++p) {
        const std::string sp = find_data_file(data_dir, argv[0], (metas[p].name + "_source.bin").c_str());
        const std::string tp = find_data_file(data_dir, argv[0], (metas[p].name + "_target.bin").c_str());
        if (sp.empty() || tp.empty() || !load_cloud_bin(sp, sources[p]) || !load_cloud_bin(tp, targets[p])) {
            std::fprintf(stderr, "ERROR: failed to load cloud pair '%s'\n", metas[p].name.c_str());
            return EXIT_FAILURE;
        }
    }

    std::printf("PROBLEM: 3 scan pairs of one synthetic room (floor+4 walls+crate+pillar); "
                "TRUE relative pose 140.0 deg yaw, 8.000 m translation (data/README.md)\n");
    for (size_t p = 0; p < metas.size(); ++p) {
        std::printf("DATA: %s n_source=%d n_target=%d overlap=%.1f%% noise_sigma=%.3fm\n",
                    metas[p].name.c_str(), metas[p].n_source, metas[p].n_target,
                    metas[p].overlap_fraction * 100.0f, metas[p].noise_sigma_m);
    }

    // ---- 2) Descriptors for every cloud, every pair --------------------------
    std::vector<Descriptors> src_desc(metas.size()), tgt_desc(metas.size());
    for (size_t p = 0; p < metas.size(); ++p) {
        float ref_s[3]; centroid3(sources[p].xyz, sources[p].n, ref_s);
        float ref_t[3]; centroid3(targets[p].xyz, targets[p].n, ref_t);
        src_desc[p] = compute_descriptors_gpu(sources[p].n, sources[p].xyz, ref_s);
        tgt_desc[p] = compute_descriptors_gpu(targets[p].n, targets[p].xyz, ref_t);
    }
    std::printf("[info] descriptors computed for %zu pairs (STAGES 1-3: KNN(k=%d) + normals + SPFH + FPFH, dim=%d)\n",
               metas.size(), kFpfhK, kFpfhDim);

    // ===========================================================================
    // VERIFY STAGE — GPU vs independent CPU twins, run on pair1 (the
    // "headline" noisy/60%-overlap pair)'s SOURCE cloud + its matching /
    // RANSAC / ICP-iteration-0 systems.
    // ===========================================================================
    const int VP = 1;   // pair index used for VERIFY

    // VERIFY(knn)
    {
        std::vector<int32_t> cpu_ids(static_cast<size_t>(sources[VP].n) * kFpfhK);
        std::vector<float> cpu_dist(static_cast<size_t>(sources[VP].n) * kFpfhK);
        knn_search_cpu(sources[VP].n, sources[VP].xyz.data(), cpu_ids.data(), cpu_dist.data());
        int mismatches = 0;
        for (size_t i = 0; i < cpu_ids.size(); ++i) if (cpu_ids[i] != src_desc[VP].neighbor_ids[i]) ++mismatches;
        const bool ok = (mismatches == 0);
        all_ok = all_ok && ok;
        std::printf("VERIFY(knn): %s (GPU brute-force KNN neighbor lists vs independent CPU twin, pair%d source, exact-match required)\n",
                   ok ? "PASS" : "FAIL", VP);
        std::printf("[info] VERIFY(knn) measured: %d mismatches / %d entries\n", mismatches, static_cast<int>(cpu_ids.size()));
        record_metric("VERIFY(knn)", "mismatches", mismatches, "0", ok ? "PASS" : "FAIL");
    }

    // VERIFY(normals)
    {
        float ref[3]; centroid3(sources[VP].xyz, sources[VP].n, ref);
        std::vector<float> cpu_normal(static_cast<size_t>(sources[VP].n) * 3);
        estimate_normals_cpu(sources[VP].n, sources[VP].xyz.data(), src_desc[VP].neighbor_ids.data(), ref[0], ref[1], ref[2], cpu_normal.data());
        int agree = 0;
        for (int i = 0; i < sources[VP].n; ++i) {
            const float* a = &src_desc[VP].normal[static_cast<size_t>(i) * 3];
            const float* b = &cpu_normal[static_cast<size_t>(i) * 3];
            const float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            if (dot >= kVerifyNormalDotMin) ++agree;
        }
        const float frac = static_cast<float>(agree) / static_cast<float>(sources[VP].n);
        const bool ok = (frac >= kVerifyNormalPassFrac);
        all_ok = all_ok && ok;
        std::printf("VERIFY(normals): %s (GPU vs independent-CPU-eigensolve normal agreement, pair%d source, "
                   "need >=%.0f%% of points with dot>=%.2f)\n", ok ? "PASS" : "FAIL", VP, kVerifyNormalPassFrac * 100.0, kVerifyNormalDotMin);
        std::printf("[info] VERIFY(normals) measured: %.1f%% agree\n", frac * 100.0);
        record_metric("VERIFY(normals)", "agree_frac", frac, ">=0.95", ok ? "PASS" : "FAIL");
    }

    // VERIFY(spfh) and VERIFY(fpfh)
    {
        std::vector<float> cpu_spfh(static_cast<size_t>(sources[VP].n) * kFpfhDim);
        compute_spfh_cpu(sources[VP].n, sources[VP].xyz.data(), src_desc[VP].normal.data(), src_desc[VP].neighbor_ids.data(), cpu_spfh.data());
        int agree_spfh = 0, total_spfh = sources[VP].n * kFpfhDim;
        for (size_t i = 0; i < cpu_spfh.size(); ++i)
            if (std::fabs(cpu_spfh[i] - src_desc[VP].spfh[i]) <= kVerifySpfhAbsTol) ++agree_spfh;
        const float frac_spfh = static_cast<float>(agree_spfh) / static_cast<float>(total_spfh);
        const bool ok_spfh = (frac_spfh >= kVerifySpfhPassFrac);
        all_ok = all_ok && ok_spfh;
        std::printf("VERIFY(spfh): %s (GPU vs CPU-twin SPFH histogram bins, pair%d source, need >=%.0f%% of bins within tol %.2f)\n",
                   ok_spfh ? "PASS" : "FAIL", VP, kVerifySpfhPassFrac * 100.0, kVerifySpfhAbsTol);
        std::printf("[info] VERIFY(spfh) measured: %.1f%% within tolerance\n", frac_spfh * 100.0);
        record_metric("VERIFY(spfh)", "agree_frac", frac_spfh, ">=0.98", ok_spfh ? "PASS" : "FAIL");

        std::vector<float> cpu_fpfh(static_cast<size_t>(sources[VP].n) * kFpfhDim);
        compute_fpfh_cpu(sources[VP].n, cpu_spfh.data(), src_desc[VP].neighbor_ids.data(), src_desc[VP].neighbor_dist.data(), cpu_fpfh.data());
        int agree_fpfh = 0, total_fpfh = sources[VP].n * kFpfhDim;
        for (size_t i = 0; i < cpu_fpfh.size(); ++i)
            if (std::fabs(cpu_fpfh[i] - src_desc[VP].fpfh[i]) <= kVerifyFpfhAbsTol) ++agree_fpfh;
        const float frac_fpfh = static_cast<float>(agree_fpfh) / static_cast<float>(total_fpfh);
        const bool ok_fpfh = (frac_fpfh >= kVerifyFpfhPassFrac);
        all_ok = all_ok && ok_fpfh;
        std::printf("VERIFY(fpfh): %s (GPU vs CPU-twin FPFH descriptor bins, pair%d source, need >=%.0f%% of bins within tol %.2f)\n",
                   ok_fpfh ? "PASS" : "FAIL", VP, kVerifyFpfhPassFrac * 100.0, kVerifyFpfhAbsTol);
        std::printf("[info] VERIFY(fpfh) measured: %.1f%% within tolerance\n", frac_fpfh * 100.0);
        record_metric("VERIFY(fpfh)", "agree_frac", frac_fpfh, ">=0.98", ok_fpfh ? "PASS" : "FAIL");
    }

    // ---- STAGE 4/5/6 for every pair (needed both for VERIFY(match/ransac/icp) below and for the independent gates) ----
    struct PairRun {
        MatchResult match;
        std::vector<float> corr_src_xyz, corr_tgt_xyz;
        int nc = 0;
        RansacResult ransac;
        int best_hyp = -1;
        Rigid3 T_ransac{};
        int ransac_inliers = 0;
        Rigid3 T_refit{};
        IcpRunResult icp;
    };
    std::vector<PairRun> runs(metas.size());

    for (size_t p = 0; p < metas.size(); ++p) {
        PairRun& run = runs[p];
        run.match = match_gpu(sources[p].n, src_desc[p].fpfh, targets[p].n, tgt_desc[p].fpfh);

        for (int i = 0; i < sources[p].n; ++i) {
            if (!run.match.matched[i]) continue;
            const int j = run.match.best_idx[i];
            run.corr_src_xyz.push_back(sources[p].xyz[i * 3 + 0]);
            run.corr_src_xyz.push_back(sources[p].xyz[i * 3 + 1]);
            run.corr_src_xyz.push_back(sources[p].xyz[i * 3 + 2]);
            run.corr_tgt_xyz.push_back(targets[p].xyz[j * 3 + 0]);
            run.corr_tgt_xyz.push_back(targets[p].xyz[j * 3 + 1]);
            run.corr_tgt_xyz.push_back(targets[p].xyz[j * 3 + 2]);
            run.nc++;
        }

        if (run.nc >= 3) {
            run.ransac = ransac_gpu(run.nc, run.corr_src_xyz, run.corr_tgt_xyz, kRansacSeed, kRansacK);
            run.best_hyp = select_best_hypothesis(run.ransac.valid, run.ransac.inlier_count);
        }

        if (run.best_hyp >= 0) {
            run.T_ransac = run.ransac.transform[static_cast<size_t>(run.best_hyp)];
            run.ransac_inliers = run.ransac.inlier_count[static_cast<size_t>(run.best_hyp)];

            // Gather the best hypothesis's INLIER correspondences and refit
            // (STAGE 6's "best transform + inlier refit" — the shared
            // rigid_fit_horn, called DIRECTLY on the host: an O(1) sequential
            // linear-algebra step over a few hundred points is not worth a
            // kernel launch, mirroring 08.01's host-side softmin blend and
            // 02.06's host-side 6x6 solve — GPU work only where genuine
            // parallelism exists, kernels.cuh's file header states this choice).
            std::vector<float> inl_src, inl_tgt;
            const float thresh2 = kRansacInlierThresholdM * kRansacInlierThresholdM;
            for (int c = 0; c < run.nc; ++c) {
                const float sp3[3] = { run.corr_src_xyz[c * 3 + 0], run.corr_src_xyz[c * 3 + 1], run.corr_src_xyz[c * 3 + 2] };
                const float tp3[3] = { run.corr_tgt_xyz[c * 3 + 0], run.corr_tgt_xyz[c * 3 + 1], run.corr_tgt_xyz[c * 3 + 2] };
                float xp[3]; apply_rigid(run.T_ransac, sp3, xp);
                if (squared_distance3(xp, tp3) <= thresh2) {
                    inl_src.push_back(sp3[0]); inl_src.push_back(sp3[1]); inl_src.push_back(sp3[2]);
                    inl_tgt.push_back(tp3[0]); inl_tgt.push_back(tp3[1]); inl_tgt.push_back(tp3[2]);
                }
            }
            run.T_refit = run.T_ransac;   // fallback if the refit is somehow degenerate
            rigid_fit_horn(static_cast<int>(inl_src.size() / 3), inl_src.data(), inl_tgt.data(), run.T_refit.R, run.T_refit.t);

            // STAGE 6 handoff: a few point-to-plane ICP iterations from the refit.
            run.icp = run_icp_gpu(run.T_refit, sources[p].n, sources[p].xyz, targets[p].n, targets[p].xyz, tgt_desc[p].normal,
                                  /*capture_iter0=*/(p == static_cast<size_t>(VP)));
        }
    }

    // VERIFY(match)
    {
        MatchResult cpu_match;
        cpu_match.matched.assign(static_cast<size_t>(sources[VP].n), 0);
        cpu_match.best_idx.assign(static_cast<size_t>(sources[VP].n), -1);
        cpu_match.dist1_sq.assign(static_cast<size_t>(sources[VP].n), 0.0f);
        cpu_match.dist2_sq.assign(static_cast<size_t>(sources[VP].n), 0.0f);
        match_correspondences_cpu(sources[VP].n, src_desc[VP].fpfh.data(), targets[VP].n, tgt_desc[VP].fpfh.data(),
                                  cpu_match.matched.data(), cpu_match.best_idx.data(), cpu_match.dist1_sq.data(), cpu_match.dist2_sq.data());
        int agree = 0;
        for (int i = 0; i < sources[VP].n; ++i)
            if (cpu_match.matched[i] == runs[VP].match.matched[i] && cpu_match.best_idx[i] == runs[VP].match.best_idx[i]) ++agree;
        const float frac = static_cast<float>(agree) / static_cast<float>(sources[VP].n);
        const bool ok = (frac >= kVerifyMatchAgreeFrac);
        all_ok = all_ok && ok;
        std::printf("VERIFY(match): %s (GPU vs independent-CPU descriptor matching, pair%d, need >=%.0f%% agree)\n",
                   ok ? "PASS" : "FAIL", VP, kVerifyMatchAgreeFrac * 100.0);
        std::printf("[info] VERIFY(match) measured: %.1f%% agree\n", frac * 100.0);
        record_metric("VERIFY(match)", "agree_frac", frac, ">=0.95", ok ? "PASS" : "FAIL");
    }

    // VERIFY(ransac) — bit-exact-checkable hypothesis generation (shared
    // rigid_fit_horn called directly by the CPU twin) + independently-
    // looped scoring.
    if (runs[VP].nc >= 3) {
        RansacResult cpu_r;
        cpu_r.valid.assign(static_cast<size_t>(kRansacK), 0);
        cpu_r.transform.assign(static_cast<size_t>(kRansacK), Rigid3{});
        cpu_r.inlier_count.assign(static_cast<size_t>(kRansacK), 0);
        ransac_hypotheses_cpu(runs[VP].nc, runs[VP].corr_src_xyz.data(), runs[VP].corr_tgt_xyz.data(), kRansacSeed, kRansacK,
                              cpu_r.valid.data(), cpu_r.transform.data(), cpu_r.inlier_count.data());
        int agree = 0;
        for (int h = 0; h < kRansacK; ++h)
            if (cpu_r.valid[h] == runs[VP].ransac.valid[h] && cpu_r.inlier_count[h] == runs[VP].ransac.inlier_count[h]) ++agree;
        const float frac = static_cast<float>(agree) / static_cast<float>(kRansacK);
        const bool ok = (frac >= kVerifyRansacAgreeFrac);
        all_ok = all_ok && ok;
        std::printf("VERIFY(ransac): %s (GPU vs independent-CPU hypothesis farm, pair%d, %d hypotheses, need >=%.0f%% "
                   "agree (valid+inlier_count))\n", ok ? "PASS" : "FAIL", VP, kRansacK, kVerifyRansacAgreeFrac * 100.0);
        std::printf("[info] VERIFY(ransac) measured: %.1f%% agree\n", frac * 100.0);
        record_metric("VERIFY(ransac)", "agree_frac", frac, ">=0.95", ok ? "PASS" : "FAIL");
    }

    // VERIFY(ransac_refit) — the real (float, shared rigid_fit_horn) refit
    // vs. the FULLY INDEPENDENT double-precision oracle (reference_cpu.cpp's
    // ransac_refit_cpu — its own Jacobi 4x4, its own accumulation).
    if (runs[VP].best_hyp >= 0) {
        std::vector<float> inl_src, inl_tgt;
        const float thresh2 = kRansacInlierThresholdM * kRansacInlierThresholdM;
        for (int c = 0; c < runs[VP].nc; ++c) {
            const float sp3[3] = { runs[VP].corr_src_xyz[c * 3 + 0], runs[VP].corr_src_xyz[c * 3 + 1], runs[VP].corr_src_xyz[c * 3 + 2] };
            const float tp3[3] = { runs[VP].corr_tgt_xyz[c * 3 + 0], runs[VP].corr_tgt_xyz[c * 3 + 1], runs[VP].corr_tgt_xyz[c * 3 + 2] };
            float xp[3]; apply_rigid(runs[VP].T_ransac, sp3, xp);
            if (squared_distance3(xp, tp3) <= thresh2) {
                inl_src.push_back(sp3[0]); inl_src.push_back(sp3[1]); inl_src.push_back(sp3[2]);
                inl_tgt.push_back(tp3[0]); inl_tgt.push_back(tp3[1]); inl_tgt.push_back(tp3[2]);
            }
        }
        float R_oracle[9], t_oracle[3];
        ransac_refit_cpu(static_cast<int>(inl_src.size() / 3), inl_src.data(), inl_tgt.data(), R_oracle, t_oracle);
        const float rot_err = rotation_error_deg(runs[VP].T_refit.R, R_oracle);
        const float trans_err = translation_error_m(runs[VP].T_refit.t, t_oracle);
        const bool ok = (rot_err <= kVerifyRefitRotTolDeg) && (trans_err <= kVerifyRefitTransTolM);
        all_ok = all_ok && ok;
        std::printf("VERIFY(ransac_refit): %s (real float refit vs independent double-precision oracle, pair%d, "
                    "tol %.1fdeg/%.2fm)\n", ok ? "PASS" : "FAIL", VP, kVerifyRefitRotTolDeg, kVerifyRefitTransTolM);
        std::printf("[info] VERIFY(ransac_refit) measured: rot_err=%.4f deg, trans_err=%.4f m\n", rot_err, trans_err);
        record_metric("VERIFY(ransac_refit)", "rot_err_deg", rot_err, "<=1.0", ok ? "PASS" : "FAIL");
    }

    // VERIFY(icp_system) — GPU (float atomics) vs CPU (sequential double)
    // point-to-plane accumulator, iteration 0 of pair1's RANSAC->ICP handoff.
    if (runs[VP].best_hyp >= 0 && !runs[VP].icp.corr_idx_iter0.empty()) {
        double cpu_accum[27];
        icp_accumulate_cpu(sources[VP].n, runs[VP].icp.cur_xyz_iter0.data(), targets[VP].xyz.data(), tgt_desc[VP].normal.data(),
                           runs[VP].icp.corr_idx_iter0.data(), cpu_accum);
        double max_rel = 0.0;
        for (int i = 0; i < 27; ++i) {
            const double denom = std::max(1.0, std::fabs(cpu_accum[i]));
            const double rel = std::fabs(cpu_accum[i] - runs[VP].icp.accum27_iter0[i]) / denom;
            max_rel = std::max(max_rel, rel);
        }
        const bool ok = (max_rel <= kVerifyIcpSystemRelTol);
        all_ok = all_ok && ok;
        std::printf("VERIFY(icp_system): %s (GPU-atomic vs CPU-sequential-double point-to-plane accumulator, pair%d "
                   "iter0, tol %.2e relative)\n", ok ? "PASS" : "FAIL", VP, kVerifyIcpSystemRelTol);
        std::printf("[info] VERIFY(icp_system) measured: max relative diff %.2e\n", max_rel);
        record_metric("VERIFY(icp_system)", "max_rel_diff", max_rel, "<=5e-3", ok ? "PASS" : "FAIL");
    }

    // ===========================================================================
    // INDEPENDENT GATES — ground truth / invariants, never GPU-vs-CPU.
    // ===========================================================================

    // GATE descriptor_invariance (pair1, primary) + [info] pair0 (clean).
    {
        const auto stat1 = descriptor_invariance(sources[VP].world_idx, src_desc[VP].fpfh, targets[VP].world_idx, tgt_desc[VP].fpfh);
        const bool ok = (stat1.n_pairs > 0) && (stat1.mean_l1 <= kGateDescInvarianceMeanL1);
        all_ok = all_ok && ok;
        std::printf("GATE descriptor_invariance: %s (pair%d, ground-truth-identical point pairs, need mean "
                   "L1(FPFH_src,FPFH_tgt) <= %.2f -- the pose-invariance property Rusu et al.'s Darboux triplet "
                   "promises, MEASURED)\n", ok ? "PASS" : "FAIL", VP, kGateDescInvarianceMeanL1);
        std::printf("[info] descriptor_invariance pair%d (noisy): %d pairs, mean L1 = %.4f, max L1 = %.4f\n",
                   VP, stat1.n_pairs, stat1.mean_l1, stat1.max_l1);
        record_metric("GATE descriptor_invariance", "mean_l1", stat1.mean_l1, "<=0.35", ok ? "PASS" : "FAIL");

        const auto stat0 = descriptor_invariance(sources[0].world_idx, src_desc[0].fpfh, targets[0].world_idx, tgt_desc[0].fpfh);
        std::printf("[info] descriptor_invariance pair0 (clean): %d pairs, mean L1 = %.4f, max L1 = %.4f\n",
                   stat0.n_pairs, stat0.mean_l1, stat0.max_l1);
    }

    // GATE registration_recovery (pair1, headline) + [info] pair0 (clean).
    float reg_rot_err = 0.0f, reg_trans_err = 0.0f;
    {
        float R_true[9]; Quat q_true{ metas[VP].q_true[0], metas[VP].q_true[1], metas[VP].q_true[2], metas[VP].q_true[3] };
        quat_to_matrix(q_true, R_true);
        reg_rot_err = rotation_error_deg(runs[VP].icp.T_final.R, R_true);
        reg_trans_err = translation_error_m(runs[VP].icp.T_final.t, metas[VP].t_true);
        const bool ok = (runs[VP].best_hyp >= 0) && (reg_rot_err <= kGateRegRotDeg) && (reg_trans_err <= kGateRegTransM);
        all_ok = all_ok && ok;
        std::printf("GATE registration_recovery: %s (pair%d, recovered-vs-true pose needs rot_err<=%.1fdeg AND "
                   "trans_err<=%.2fm -- global registration recovers a 140deg/8m transform with NO initial guess)\n",
                   ok ? "PASS" : "FAIL", VP, kGateRegRotDeg, kGateRegTransM);
        std::printf("[info] registration_recovery pair%d measured: rot_err=%.4f deg, trans_err=%.4f m\n", VP, reg_rot_err, reg_trans_err);
        record_metric("GATE registration_recovery", "rot_err_deg", reg_rot_err, "<=3.0", ok ? "PASS" : "FAIL");
        record_metric("GATE registration_recovery", "trans_err_m", reg_trans_err, "<=0.20", ok ? "PASS" : "FAIL");

        if (runs[0].best_hyp >= 0) {
            float R_true0[9]; Quat q_true0{ metas[0].q_true[0], metas[0].q_true[1], metas[0].q_true[2], metas[0].q_true[3] };
            quat_to_matrix(q_true0, R_true0);
            const float rot0 = rotation_error_deg(runs[0].icp.T_final.R, R_true0);
            const float trans0 = translation_error_m(runs[0].icp.T_final.t, metas[0].t_true);
            std::printf("[info] registration_recovery pair0 (clean): rot_err=%.4f deg, trans_err=%.4f m\n", rot0, trans0);
        }
    }

    // GATE icp_negative_control (pair1): ICP FROM IDENTITY, no RANSAC.
    {
        Rigid3 T_identity{}; T_identity.R[0] = T_identity.R[4] = T_identity.R[8] = 1.0f;
        const IcpRunResult neg = run_icp_gpu(T_identity, sources[VP].n, sources[VP].xyz, targets[VP].n, targets[VP].xyz,
                                            tgt_desc[VP].normal, false);
        float R_true[9]; Quat q_true{ metas[VP].q_true[0], metas[VP].q_true[1], metas[VP].q_true[2], metas[VP].q_true[3] };
        quat_to_matrix(q_true, R_true);
        const float rot_err = rotation_error_deg(neg.T_final.R, R_true);
        const float trans_err = translation_error_m(neg.T_final.t, metas[VP].t_true);
        const bool ok = (rot_err >= kGateNegControlRotDeg) || (trans_err >= kGateNegControlTransM);
        all_ok = all_ok && ok;
        std::printf("GATE icp_negative_control: %s (pair%d, ICP-FROM-IDENTITY (no RANSAC) residual error must EXCEED "
                   "%.0f deg or %.1f m, proving local ICP alone cannot recover this 140deg/8m transform: global "
                   "registration earns its keep)\n", ok ? "PASS" : "FAIL", VP, kGateNegControlRotDeg, kGateNegControlTransM);
        std::printf("[info] icp_negative_control pair%d measured: rot_err=%.2f deg, trans_err=%.2f m\n", VP, rot_err, trans_err);
        record_metric("GATE icp_negative_control", "rot_err_deg", rot_err, ">=25.0 (OR)", ok ? "PASS" : "FAIL");
    }

    // GATE ransac_formula (pair1): 02.03's analytic RANSAC-iteration-count
    // check, re-derived for the 3-CORRESPONDENCE sample case.
    {
        const double w = (runs[VP].nc > 0) ? (static_cast<double>(runs[VP].ransac_inliers) / runs[VP].nc) : 0.0;
        const double w_clamped = std::min(0.999, std::max(1.0e-6, w));
        const double k_required = std::log(1.0 - kRansacTargetSuccessProb) / std::log(1.0 - w_clamped * w_clamped * w_clamped);
        const bool ok = (runs[VP].nc >= 3) && (k_required <= static_cast<double>(kRansacK));
        all_ok = all_ok && ok;
        std::printf("GATE ransac_formula: %s (pair%d, classical k=log(1-p)/log(1-w^3) [w = measured correspondence "
                   "inlier ratio] must clear budget kRansacK=%d for p=%.3f)\n", ok ? "PASS" : "FAIL", VP, kRansacK, kRansacTargetSuccessProb);
        std::printf("[info] ransac_formula pair%d measured: w=%.3f (%d/%d), k_required=%.0f\n",
                   VP, w, runs[VP].ransac_inliers, runs[VP].nc, k_required);
        record_metric("GATE ransac_formula", "k_required", k_required, "<=8192", ok ? "PASS" : "FAIL");
    }

    // [info] refinement_payoff (pair1) — the ratified scope explicitly
    // allows this as "[info] or gated"; reported here honestly rather than
    // gated, because this project's synthetic correspondences are already
    // near-exact (same physical world point, only iid sensor noise), so
    // the MANY-POINT INLIER REFIT (STAGE 6's "best transform + inlier
    // refit", a Horn least-squares fit already averaging over hundreds of
    // near-perfect correspondences) can legitimately leave little room for
    // ICP to improve further — occasionally ICP's fixed-radius NEAREST-
    // POINT correspondence search even lands on a slightly different
    // (nearest, not same-physical) target point than the refit's exact
    // correspondence, adding a hair of quantization-like error (THEORY.md
    // "Where this sits in the real world" discusses when real-world
    // correspondence sets are sparser/noisier and ICP's own payoff grows).
    // The MEANINGFUL, always-true comparison is against the MINIMAL
    // 3-point hypothesis alone (T_ransac, before inlier refit) -- both the
    // refit AND ICP should improve on that noisy 3-point estimate.
    {
        float R_true[9]; Quat q_true{ metas[VP].q_true[0], metas[VP].q_true[1], metas[VP].q_true[2], metas[VP].q_true[3] };
        quat_to_matrix(q_true, R_true);
        const float rot_hyp3 = rotation_error_deg(runs[VP].T_ransac.R, R_true);
        const float trans_hyp3 = translation_error_m(runs[VP].T_ransac.t, metas[VP].t_true);
        const float rot_refit = rotation_error_deg(runs[VP].T_refit.R, R_true);
        const float trans_refit = translation_error_m(runs[VP].T_refit.t, metas[VP].t_true);
        const float rot_icp = reg_rot_err;
        const float trans_icp = reg_trans_err;
        std::printf("[info] refinement_payoff pair%d: minimal-3pt-hypothesis rot=%.3fdeg/trans=%.3fm -> inlier-refit "
                   "rot=%.3fdeg/trans=%.3fm -> +ICP rot=%.3fdeg/trans=%.3fm (the global-then-local doctrine: RANSAC's "
                   "inlier refit is the big improvement here; ICP's marginal contribution is small/mixed because this "
                   "scene's correspondences are already near-exact -- see THEORY.md)\n",
                   VP, rot_hyp3, trans_hyp3, rot_refit, trans_refit, rot_icp, trans_icp);
        record_metric("[info] refinement_payoff", "rot_hyp3_deg", rot_hyp3, "n/a", "info");
        record_metric("[info] refinement_payoff", "rot_refit_deg", rot_refit, "n/a", "info");
        record_metric("[info] refinement_payoff", "rot_icp_deg", rot_icp, "n/a", "info");
    }

    // [info] prescreen_efficiency (pair1): fraction of drawn triplets the
    // edge-length prescreen rejected before ever attempting a Horn fit.
    {
        int invalid = 0;
        for (int h = 0; h < kRansacK; ++h) if (!runs[VP].ransac.valid[h]) ++invalid;
        const float frac = static_cast<float>(invalid) / static_cast<float>(kRansacK);
        std::printf("[info] prescreen_efficiency pair%d: %d/%d (%.1f%%) hypotheses rejected before/at the fit step "
                    "(degenerate draw or edge-length prescreen failure across %d retry attempts each)\n",
                    VP, invalid, kRansacK, frac * 100.0, kRansacMaxTripletAttempts);
        record_metric("[info] prescreen_efficiency", "rejected_frac", frac, "n/a", "info");
    }

    // [info]/honesty low_overlap (pair2, the 30%-overlap stress cohort):
    // REPORTED, never required to succeed.
    {
        float R_true[9]; Quat q_true{ metas[2].q_true[0], metas[2].q_true[1], metas[2].q_true[2], metas[2].q_true[3] };
        quat_to_matrix(q_true, R_true);
        if (runs[2].best_hyp >= 0) {
            const float rot_err = rotation_error_deg(runs[2].icp.T_final.R, R_true);
            const float trans_err = translation_error_m(runs[2].icp.T_final.t, metas[2].t_true);
            const bool succeeded = (rot_err <= kGateRegRotDeg) && (trans_err <= kGateRegTransM);
            std::printf("[info] low_overlap pair2 (%.1f%% overlap): registration %s -- rot_err=%.3f deg, trans_err=%.3f m "
                       "(honesty: NOT gated -- low overlap starves the correspondence set and may legitimately fail)\n",
                       metas[2].overlap_fraction * 100.0f, succeeded ? "SUCCEEDED" : "FAILED", rot_err, trans_err);
            record_metric("[info] low_overlap", "rot_err_deg", rot_err, "n/a (reported only)", succeeded ? "info-pass" : "info-fail");
        } else {
            std::printf("[info] low_overlap pair2 (%.1f%% overlap): registration FAILED -- fewer than 3 matched "
                       "correspondences survived the ratio test (honesty: NOT gated)\n", metas[2].overlap_fraction * 100.0f);
            record_metric("[info] low_overlap", "outcome", 0.0, "n/a (reported only)", "info-fail");
        }
    }

    // ---- Artifacts ------------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);

    unsigned char col_src[3] = { 220, 60, 60 };
    unsigned char col_tgt[3] = { 40, 110, 220 };
    rasterize_topview(out_dir + "/topview_before.ppm", 640, 640,
                      sources[VP].xyz, sources[VP].n, col_src, targets[VP].xyz, targets[VP].n, col_tgt);
    // "after" shows the ALIGNED source (final T applied) against the target —
    // computed explicitly here (cheap: n_src points, one host-side loop).
    {
        std::vector<float> aligned(static_cast<size_t>(sources[VP].n) * 3);
        for (int i = 0; i < sources[VP].n; ++i) {
            const float p[3] = { sources[VP].xyz[i * 3 + 0], sources[VP].xyz[i * 3 + 1], sources[VP].xyz[i * 3 + 2] };
            float o[3]; apply_rigid(runs[VP].icp.T_final, p, o);
            aligned[i * 3 + 0] = o[0]; aligned[i * 3 + 1] = o[1]; aligned[i * 3 + 2] = o[2];
        }
        rasterize_topview(out_dir + "/topview_after.ppm", 640, 640, aligned, sources[VP].n, col_src, targets[VP].xyz, targets[VP].n, col_tgt);
    }

    // descriptor_distance_histogram.csv — matched (ground-truth-correct) vs
    // random source/target pairs' FPFH L2 distance (pair1) — the
    // separability visual behind GATE descriptor_invariance / the ratio test.
    {
        std::ofstream f(out_dir + "/descriptor_distance_histogram.csv");
        f << "# descriptor_distance_histogram.csv -- FPFH L2 distance, matched (ground-truth) vs random pairs, pair1\n";
        f << "kind,distance\n";
        std::unordered_map<int32_t, int32_t> tgt_map;
        for (size_t j = 0; j < targets[VP].world_idx.size(); ++j) tgt_map[targets[VP].world_idx[static_cast<size_t>(j)]] = static_cast<int32_t>(j);
        int written_matched = 0;
        for (int i = 0; i < sources[VP].n && written_matched < 400; ++i) {
            const auto it = tgt_map.find(sources[VP].world_idx[static_cast<size_t>(i)]);
            if (it == tgt_map.end()) continue;
            const int j = it->second;
            double d2 = 0.0;
            for (int b = 0; b < kFpfhDim; ++b) {
                const double diff = src_desc[VP].fpfh[static_cast<size_t>(i) * kFpfhDim + b] - tgt_desc[VP].fpfh[static_cast<size_t>(j) * kFpfhDim + b];
                d2 += diff * diff;
            }
            f << "matched," << std::sqrt(d2) << "\n";
            ++written_matched;
        }
        uint32_t rng_state = 777u;
        auto next_u32 = [&]() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; };
        for (int r = 0; r < written_matched; ++r) {
            const int i = static_cast<int>(next_u32() % static_cast<uint32_t>(sources[VP].n));
            const int j = static_cast<int>(next_u32() % static_cast<uint32_t>(targets[VP].n));
            double d2 = 0.0;
            for (int b = 0; b < kFpfhDim; ++b) {
                const double diff = src_desc[VP].fpfh[static_cast<size_t>(i) * kFpfhDim + b] - tgt_desc[VP].fpfh[static_cast<size_t>(j) * kFpfhDim + b];
                d2 += diff * diff;
            }
            f << "random," << std::sqrt(d2) << "\n";
        }
        std::printf("ARTIFACT: wrote demo/out/descriptor_distance_histogram.csv (%d matched + %d random pairs, pair1)\n",
                    written_matched, written_matched);
    }

    // gates_metrics.csv
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        f << "# gates_metrics.csv -- measured numbers behind every VERIFY/GATE/[info] line, project 02.10\n";
        f << "gate,metric,value,threshold,status\n";
        for (const auto& r : g_metrics) f << r.gate << "," << r.metric << "," << r.value << "," << r.threshold << "," << r.status << "\n";
    }

    std::printf("ARTIFACT: wrote demo/out/{topview_before.ppm, topview_after.ppm, descriptor_distance_histogram.csv, gates_metrics.csv}\n");

    // ---- Timing (teaching artifact, not a benchmark) ---------------------------
    std::printf("[time] pipeline covers %zu pairs x (descriptors + match + RANSAC(%d hyps) + up to %d ICP iters)\n",
               metas.size(), kRansacK, kIcpMaxIters);

    if (all_ok) {
        std::printf("RESULT: PASS (VERIFY(knn/normals/spfh/fpfh/match/ransac/ransac_refit/icp_system) + "
                    "GATE(descriptor_invariance/registration_recovery/icp_negative_control/ransac_formula) all PASS)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see the VERIFY/GATE lines above for which check failed)\n");
        return EXIT_FAILURE;
    }
}
