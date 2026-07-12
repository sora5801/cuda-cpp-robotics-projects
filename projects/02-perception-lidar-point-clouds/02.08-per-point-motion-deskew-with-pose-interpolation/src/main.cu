// ===========================================================================
// main.cu — entry point for project 02.08
//           Per-point motion deskew with pose interpolation
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed sample (4 motion cohorts, each a paired
//      skewed/instantaneous-truth point set + a dense trajectory — see
//      ../scripts/make_synthetic.py).
//   2. VERIFY stage (the CLAUDE.md §5 GPU-vs-CPU gate): for every cohort and
//      both interpolation regimes, run the deskew kernel AND its CPU twin on
//      identical inputs and require per-point agreement within a tight
//      tolerance.
//   3. SLERP_CORRECTNESS: a small, data-independent unit-test-style check of
//      quat_slerp itself — a >90-degree quaternion pair, its geodesic angle,
//      and the double-cover sign-flip invariance.
//   4. Deskew every cohort with BOTH interpolation regimes (dense ~200 Hz-
//      equivalent samples, sparse 2-sample start/end) on the GPU, and
//      compare every result against the cohort's analytic instantaneous
//      truth — the correctness oracle ../scripts/make_synthetic.py alone
//      knows exactly (main.cu never sees the continuous trajectory, only
//      its discretization — see kernels.cuh "TRAJECTORY LAYOUT").
//   5. Four independent gates beyond the twin comparison (per the
//      independence ruling in reference_cpu.cpp): IDENTITY_CONTROL,
//      RESTORATION, SAMPLING_LESSON, DOWNSTREAM_PAYOFF.
//   6. Write the demo/out/ artifacts (triptych image, per-cohort error CSVs,
//      a gate-metrics CSV) and report PASS/FAIL.
//
// Output contract (load-bearing!)
// -------------------------------
// demo/run_demo.ps1 diffs the STABLE lines of this program's stdout against
// demo/expected_output.txt: "[demo]", "PROBLEM:", "DATA:", the six gate
// verdict lines, "ARTIFACT:" lines, and "RESULT:". These lines never carry
// a run-varying FLOAT (only integers baked into the committed sample file,
// which never change) — every measured number lives on an "[info]" line,
// which is NOT diffed (same discipline 08.01/01.10 use, for the same
// reason: FMA/intrinsic differences across GPU architectures could
// otherwise flip a byte in a "stable" line on someone else's machine).
// "[time]" lines are timings, also not diffed.
//
// Read this first, then kernels.cuh (the interface + the deskew derivation)
// then kernels.cu (the kernel) then reference_cpu.cpp (the oracle).
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
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// A local, double-precision pi — kept separate from any CUDA math-library
// constant so this host-only orchestration code has no device-header
// dependency beyond kernels.cuh (M_PI is not guaranteed by <cmath> under
// MSVC without a feature-test macro; defining our own is the portable,
// no-surprises choice used elsewhere in this repo, e.g. kernels.cuh's kPi
// pattern in sibling projects).
// ---------------------------------------------------------------------------
static const double kPiD = 3.14159265358979323846;

// ===========================================================================
// Gate tolerances — every bound below carries a generous margin over the
// ACTUAL measured numbers this project's algorithm produces (documented in
// THEORY.md "How we verify correctness" and printed on this program's
// [info] lines every run), following the repo-wide discipline that a
// PASS/FAIL threshold should never sit close to the achieved behavior
// (CLAUDE.md §12 determinism note; 08.01's THEORY.md makes the same case).
// ===========================================================================

// GPU-vs-CPU twin tolerance (VERIFY stage). Both paths call the IDENTICAL
// shared HD deskew_one_point (kernels.cuh) — divergence should be at the
// level of nvcc-device vs cl.exe/nvcc-host sinf/cosf/acosf implementation
// differences, i.e. a few ULP on values of order 1-10 m. 1e-4 m (0.1 mm) is
// enormous headroom above that.
static const float kVerifyTolM = 1.0e-4f;

// IDENTITY_CONTROL (cohort 3, stationary): every trajectory sample is
// IDENTICAL, so the relative transform is the identity everywhere — a
// correct deskew changes nothing but float rounding. 2 mm is generous.
static const float kIdentityToleranceM = 2.0e-3f;

// RESTORATION (cohorts 0,1,2 = straight, arc, wiggle): per-cohort bounds on
// the DENSE-regime deskewed error against instantaneous truth. Measured
// (double-precision Python prototype, THEORY.md quotes the from-this-binary
// numbers): straight ~0 m, arc ~1.9e-4 m mean / 2.8e-4 m max, wiggle
// ~0.096 m mean / 0.281 m max — bounds below sit far above all three.
static const float kRestorationMeanBoundM[3] = { 0.02f, 0.02f, 0.5f };   // [straight, arc, wiggle]
static const float kRestorationMaxBoundM[3]  = { 0.05f, 0.05f, 1.0f };
// Negative-control sanity floor: the UNDESKEWED baseline must show REAL
// distortion (measured means: 0.75 / 2.32 / 1.94 m) — this catches a
// generator regression that accidentally made the cohorts non-distorting.
static const float kMinUndeskewedBaselineM = 0.3f;

// SAMPLING_LESSON:
//   (i) wiggle cohort: sparse-regime mean error / dense-regime mean error
//       must clear this floor (measured ratio: ~19.2x).
//   (ii) straight cohort: dense and sparse regimes must AGREE (constant
//        velocity makes 2-sample interpolation exact) — both means under
//        an absolute bound, and their difference under a tighter one.
static const float kSamplingRatioFloor    = 5.0f;
static const float kConsistencyAbsBoundM  = 0.02f;
static const float kConsistencyDiffBoundM = 0.01f;

// SLERP_CORRECTNESS: geodesic-angle progression error (radians) and
// sign-flip dot-product error, both measured in double precision from a
// >90-degree test pair (see test_slerp_correctness below).
static const double kSlerpAngleTolRad = 2.0e-3;
static const double kSlerpSignDotTol  = 2.0e-3;

// DOWNSTREAM_PAYOFF: wall plane-fit RMS thickness (straight cohort). A
// naively-stacked skewed scan measurably THICKENS the wall (measured:
// ~0.256 m RMS over the sweep's 1.5 m approach) while the deskewed scan
// TIGHTENS it back down (measured: ~0.0055 m, matching the analytic truth).
static const double kWallSkewedRmsMinM      = 0.05;   // skewed must show REAL thickening
static const double kWallDeskewedRmsMaxM    = 0.03;   // deskewed must be TIGHT
static const double kWallImprovementFactor  = 3.0;    // skewed_rms must exceed this * deskewed_rms

// Azimuth/elevation window selecting "the +x wall" ray indices for the
// downstream-payoff gate (see select_wall_indices below) — a modest cone
// aimed straight down +x, restricted to near-horizontal beams so every
// selected ray reliably clears the room and hits the wall (not the floor,
// not open sky over the wall top — see ../scripts/make_synthetic.py's
// scene geometry, room half-width 8 m, wall top 1.5 m above the sensor).
static const float kWallAzimuthWindowDeg = 8.0f;
static const float kWallElevMaxDeg       = 4.0f;

// ===========================================================================
// Binary-sample data model + loader — the format ../scripts/make_synthetic.py
// writes (see that script's write_binary_sample docstring for the exact
// byte layout this mirrors field-for-field, with EXPLICIT primitive reads,
// never a raw struct fread — CLAUDE.md §12, same discipline 02.01 uses).
// ===========================================================================

// One motion cohort: a paired skewed/truth point set + the dense trajectory
// that produced it. See kernels.cuh's file header for every array's layout.
struct Cohort {
    int id = -1;
    int n_points = 0;
    std::vector<float> dense_traj;   // [kDenseSamples * kTrajStride]
    std::vector<float> t;            // [n_points] firing time, s
    std::vector<int>   beam_id;      // [n_points] 0..kNumBeams-1
    std::vector<float> xyz_local;    // [n_points*3] SKEWED, sensor-local-at-t frame, m
    std::vector<float> xyz_truth;    // [n_points*3] ground truth in the reference frame, m
};

struct Sample {
    bool loaded = false;
    int num_beams = 0, azimuth_steps = 0, n_dense_samples = 0;
    float sweep_duration_s = 0.0f, room_half_m = 0.0f;
    Cohort cohorts[kNumCohorts];
};

static bool read_i32(std::ifstream& f, int32_t& v)
{
    f.read(reinterpret_cast<char*>(&v), sizeof(int32_t));
    return static_cast<bool>(f);
}
static bool read_f32(std::ifstream& f, float& v)
{
    f.read(reinterpret_cast<char*>(&v), sizeof(float));
    return static_cast<bool>(f);
}
static bool read_f32_array(std::ifstream& f, std::vector<float>& v, size_t n)
{
    v.resize(n);
    if (n == 0) return true;
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
}

// load_sample — read the committed deskew_scan.bin (see the format comment
// above). Validates the file header against kernels.cuh's compiled-in
// constants (num_beams/azimuth_steps/n_dense_samples) — a data/code
// consistency check, same discipline 02.01's kVoxelLeafM assertion uses —
// so a stale committed sample (regenerated with different constants than
// the code was built against) fails LOUDLY here, not silently downstream.
static Sample load_sample(const std::string& path)
{
    Sample s;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return s;

    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "DESKEW01", 8) != 0) {
        std::fprintf(stderr, "load_sample: bad magic in %s\n", path.c_str());
        return Sample{};
    }

    int32_t num_cohorts = 0;
    if (!read_i32(f, num_cohorts) || !read_i32(f, s.num_beams) ||
        !read_i32(f, s.azimuth_steps) || !read_i32(f, s.n_dense_samples) ||
        !read_f32(f, s.sweep_duration_s) || !read_f32(f, s.room_half_m)) {
        std::fprintf(stderr, "load_sample: truncated header in %s\n", path.c_str());
        return Sample{};
    }
    if (num_cohorts != kNumCohorts || s.num_beams != kNumBeams ||
        s.azimuth_steps != kAzimuthSteps || s.n_dense_samples != kDenseSamples) {
        std::fprintf(stderr,
            "load_sample: file header (%d cohorts, %d beams, %d az steps, %d dense samples) "
            "does not match kernels.cuh's compiled-in constants (%d, %d, %d, %d) -- "
            "regenerate with: python scripts/make_synthetic.py\n",
            num_cohorts, s.num_beams, s.azimuth_steps, s.n_dense_samples,
            kNumCohorts, kNumBeams, kAzimuthSteps, kDenseSamples);
        return Sample{};
    }

    for (int c = 0; c < kNumCohorts; ++c) {
        Cohort& co = s.cohorts[c];
        int32_t cohort_id = 0;
        if (!read_i32(f, cohort_id) || !read_i32(f, co.n_points)) {
            std::fprintf(stderr, "load_sample: truncated cohort %d header\n", c);
            return Sample{};
        }
        if (cohort_id != c) {
            std::fprintf(stderr, "load_sample: cohort order mismatch (expected %d, got %d)\n", c, cohort_id);
            return Sample{};
        }
        co.id = cohort_id;

        if (!read_f32_array(f, co.dense_traj, static_cast<size_t>(kDenseSamples) * kTrajStride)) {
            std::fprintf(stderr, "load_sample: truncated dense trajectory for cohort %d\n", c);
            return Sample{};
        }

        co.t.resize(static_cast<size_t>(co.n_points));
        co.beam_id.resize(static_cast<size_t>(co.n_points));
        co.xyz_local.resize(static_cast<size_t>(co.n_points) * 3);
        for (int i = 0; i < co.n_points; ++i) {
            float t_val = 0.0f, x = 0.0f, y = 0.0f, z = 0.0f;
            int32_t beam = 0;
            if (!read_f32(f, t_val) || !read_i32(f, beam) ||
                !read_f32(f, x) || !read_f32(f, y) || !read_f32(f, z)) {
                std::fprintf(stderr, "load_sample: truncated point record %d in cohort %d\n", i, c);
                return Sample{};
            }
            co.t[static_cast<size_t>(i)] = t_val;
            co.beam_id[static_cast<size_t>(i)] = beam;
            co.xyz_local[static_cast<size_t>(i) * 3 + 0] = x;
            co.xyz_local[static_cast<size_t>(i) * 3 + 1] = y;
            co.xyz_local[static_cast<size_t>(i) * 3 + 2] = z;
        }

        if (!read_f32_array(f, co.xyz_truth, static_cast<size_t>(co.n_points) * 3)) {
            std::fprintf(stderr, "load_sample: truncated truth block for cohort %d\n", c);
            return Sample{};
        }
    }

    s.loaded = true;
    return s;
}

// sparse_trajectory_of — derive the 2-sample SPARSE regime from a cohort's
// dense trajectory by taking just the first and last samples (kernels.cuh
// "TRAJECTORY LAYOUT": one honest source of ground truth, two ways of
// subsampling it — no separate sparse array is ever generated or stored).
static std::vector<float> sparse_trajectory_of(const std::vector<float>& dense_traj)
{
    std::vector<float> sparse(static_cast<size_t>(kSparseSamples) * kTrajStride);
    std::copy(dense_traj.begin(), dense_traj.begin() + kTrajStride, sparse.begin());
    std::copy(dense_traj.end() - kTrajStride, dense_traj.end(), sparse.begin() + kTrajStride);
    return sparse;
}

// ===========================================================================
// Small numeric helpers shared by every gate below.
// ===========================================================================

struct ErrStats { double mean = 0.0, max_v = 0.0; };

// compute_error_stats — mean and max Euclidean per-point distance between
// two [n*3] point arrays (used for BOTH the VERIFY twin comparison and
// every restoration/sampling error measurement — the metric is the same,
// only the two arrays being compared change).
static ErrStats compute_error_stats(const std::vector<float>& a, const std::vector<float>& b, int n_points)
{
    ErrStats s;
    double sum = 0.0;
    for (int i = 0; i < n_points; ++i) {
        const double dx = static_cast<double>(a[static_cast<size_t>(i) * 3 + 0]) - b[static_cast<size_t>(i) * 3 + 0];
        const double dy = static_cast<double>(a[static_cast<size_t>(i) * 3 + 1]) - b[static_cast<size_t>(i) * 3 + 1];
        const double dz = static_cast<double>(a[static_cast<size_t>(i) * 3 + 2]) - b[static_cast<size_t>(i) * 3 + 2];
        const double e = std::sqrt(dx * dx + dy * dy + dz * dz);
        sum += e;
        if (e > s.max_v) s.max_v = e;
    }
    s.mean = (n_points > 0) ? (sum / n_points) : 0.0;
    return s;
}

// per_point_errors — like compute_error_stats but keeps every point's
// individual error (feeds the errors_<cohort>.csv artifact).
static std::vector<double> per_point_errors(const std::vector<float>& a, const std::vector<float>& b, int n_points)
{
    std::vector<double> out(static_cast<size_t>(n_points));
    for (int i = 0; i < n_points; ++i) {
        const double dx = static_cast<double>(a[static_cast<size_t>(i) * 3 + 0]) - b[static_cast<size_t>(i) * 3 + 0];
        const double dy = static_cast<double>(a[static_cast<size_t>(i) * 3 + 1]) - b[static_cast<size_t>(i) * 3 + 1];
        const double dz = static_cast<double>(a[static_cast<size_t>(i) * 3 + 2]) - b[static_cast<size_t>(i) * 3 + 2];
        out[static_cast<size_t>(i)] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return out;
}

// ---------------------------------------------------------------------------
// plane_fit_rms — RMS point-to-best-fit-plane distance (meters) for a set
// of near-planar points, via PCA: the SMALLEST eigenvalue of the point
// set's covariance matrix is the variance along the least-spread axis (the
// least-squares plane's normal direction for a near-planar cloud); its
// square root is exactly the RMS perpendicular distance to that plane
// (02.03 ground-segmentation's PCA lineage, cited — this is the same
// closed-form idea applied to a wall instead of a ground plane).
//
// The smallest eigenvalue of a symmetric 3x3 matrix is computed via Smith's
// (1961) closed-form TRIGONOMETRIC solution of the characteristic cubic —
// no iterative eigensolver, no external linear-algebra library (CLAUDE.md
// §1: no black boxes for something this small), exact up to double
// rounding. See THEORY.md "How we verify correctness" for the formula.
// ---------------------------------------------------------------------------
static double plane_fit_rms(const std::vector<Vec3>& pts)
{
    const size_t n = pts.size();
    if (n < 3) return 0.0;

    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (const Vec3& p : pts) { cx += p.x; cy += p.y; cz += p.z; }
    cx /= static_cast<double>(n); cy /= static_cast<double>(n); cz /= static_cast<double>(n);

    double cxx = 0.0, cyy = 0.0, czz = 0.0, cxy = 0.0, cxz = 0.0, cyz = 0.0;
    for (const Vec3& p : pts) {
        const double dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
        cxx += dx * dx; cyy += dy * dy; czz += dz * dz;
        cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
    }
    const double nd = static_cast<double>(n);
    cxx /= nd; cyy /= nd; czz /= nd; cxy /= nd; cxz /= nd; cyz /= nd;

    const double p1 = cxy * cxy + cxz * cxz + cyz * cyz;
    if (p1 < 1e-18) {
        // Already diagonal (a degenerate/perfectly-axis-aligned point set):
        // eigenvalues are just the diagonal entries.
        const double m = std::min({ cxx, cyy, czz });
        return std::sqrt(std::max(0.0, m));
    }
    const double trace = cxx + cyy + czz;
    const double q = trace / 3.0;
    const double p2 = (cxx - q) * (cxx - q) + (cyy - q) * (cyy - q) + (czz - q) * (czz - q) + 2.0 * p1;
    const double p = std::sqrt(p2 / 6.0);
    // B = (1/p) * (C - q*I) — the "normalized" matrix Smith's method solves.
    const double bxx = (cxx - q) / p, byy = (cyy - q) / p, bzz = (czz - q) / p;
    const double bxy = cxy / p, bxz = cxz / p, byz = cyz / p;
    const double detB = bxx * (byy * bzz - byz * byz) - bxy * (bxy * bzz - byz * bxz) + bxz * (bxy * byz - byy * bxz);
    double r = detB / 2.0;
    r = r < -1.0 ? -1.0 : (r > 1.0 ? 1.0 : r);   // guard acos domain against double rounding
    const double phi = std::acos(r) / 3.0;
    const double eig_max = q + 2.0 * p * std::cos(phi);
    const double eig_min = q + 2.0 * p * std::cos(phi + 2.0 * kPiD / 3.0);
    const double eig_mid = 3.0 * q - eig_max - eig_min;
    const double smallest = std::min({ eig_max, eig_mid, eig_min });
    return std::sqrt(std::max(0.0, smallest));
}

// select_wall_indices — ray indices whose LOCAL firing direction points
// within kWallAzimuthWindowDeg of straight +x and whose beam is nearly
// horizontal (|elevation| <= kWallElevMaxDeg) — a cone reliably aimed at
// the scene's +x wall for the straight cohort (constant identity
// orientation, so local +x IS world +x throughout — see the constant below
// where this is called). Azimuth is recovered from the firing time alone
// (az = 2*pi*t/kSweepDurationS, wrapped to (-pi,pi]) because
// ../scripts/make_synthetic.py fires azimuth-major: t and az are in exact
// bijection within one sweep — no separate azimuth field is stored.
static std::vector<int> select_wall_indices(const Cohort& c)
{
    std::vector<int> idx;
    const double window_rad = kWallAzimuthWindowDeg * kPiD / 180.0;
    for (int i = 0; i < c.n_points; ++i) {
        double az = 2.0 * kPiD * (static_cast<double>(c.t[static_cast<size_t>(i)]) / kSweepDurationS);
        if (az > kPiD) az -= 2.0 * kPiD;
        const float elev = kBeamElevDeg[c.beam_id[static_cast<size_t>(i)]];
        if (std::fabs(az) <= window_rad && std::fabs(elev) <= kWallElevMaxDeg) {
            idx.push_back(i);
        }
    }
    return idx;
}

// ===========================================================================
// GPU / CPU deskew run wrappers — the canonical 5-step CUDA sequence
// (allocate -> H2D -> launch -> D2H -> free) spelled out once here, reused
// for every (cohort, regime) pair the VERIFY and gate stages need.
// ===========================================================================

static std::vector<float> run_deskew_gpu(const Cohort& c, const std::vector<float>& traj,
                                         int n_samples, Pose ref_pose, float& out_kernel_ms)
{
    std::vector<float> out(static_cast<size_t>(c.n_points) * 3, 0.0f);
    out_kernel_ms = 0.0f;
    if (c.n_points == 0) return out;

    set_trajectory(traj.data(), n_samples);   // upload THIS regime's samples to __constant__ memory

    const size_t np = static_cast<size_t>(c.n_points);
    float *d_t = nullptr, *d_xyz_local = nullptr, *d_xyz_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_t, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xyz_local, np * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xyz_out, np * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_t, c.t.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xyz_local, c.xyz_local.data(), np * 3 * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer gt;
    gt.begin();
    launch_deskew(c.n_points, d_t, d_xyz_local, n_samples, ref_pose, d_xyz_out);
    out_kernel_ms = gt.end_ms();

    CUDA_CHECK(cudaMemcpy(out.data(), d_xyz_out, np * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_xyz_local));
    CUDA_CHECK(cudaFree(d_xyz_out));
    return out;
}

static std::vector<float> run_deskew_cpu_wrap(const Cohort& c, const std::vector<float>& traj,
                                              int n_samples, Pose ref_pose, double& out_ms)
{
    std::vector<float> out(static_cast<size_t>(c.n_points) * 3, 0.0f);
    CpuTimer ct;
    ct.begin();
    deskew_cpu(c.n_points, c.t.data(), c.xyz_local.data(), traj.data(), n_samples, ref_pose, out.data());
    out_ms = ct.end_ms();
    return out;
}

// ref_pose_of — the platform's interpolated pose at the reference instant
// t_ref = kSweepDurationS, computed ONCE per cohort from the dense
// trajectory (shared verbatim by both regimes — see kernels.cuh's
// interpolate_pose: at t_ref, which sits exactly on the trajectory's LAST
// sample by construction, both the dense and sparse arrays bracket to the
// identical final segment with alpha=1, so either array would give the
// SAME answer; using the dense one is simply the single source of truth).
static Pose ref_pose_of(const Cohort& c)
{
    Vec3 p_ref; Quat q_ref;
    interpolate_pose(c.dense_traj.data(), kDenseSamples, kSweepDurationS, p_ref, q_ref);
    return Pose{ p_ref, q_ref };
}

// ===========================================================================
// SLERP_CORRECTNESS — a data-independent unit-test-style check of
// quat_slerp itself (kernels.cuh), exercising exactly the two classic
// gotchas that header's comment documents: the geodesic-angle argument, and
// double-cover sign-flip invariance. Uses a test pair MORE than 90 degrees
// apart (the task's explicit ask — a pair inside 90 degrees would never
// exercise the "long way around" risk double-cover handling guards against).
// ===========================================================================
struct SlerpTestResult { bool pass = false; double angle_err_rad = 0.0; double sign_err = 0.0; };

static SlerpTestResult test_slerp_correctness()
{
    // q1 = identity. q2 = a kTestAngleDeg-degree rotation about a GENERAL
    // (not axis-aligned) axis (1,1,1)/sqrt(3) — a stronger test than a pure
    // z-rotation, since no quaternion component is zero by construction.
    const double kTestAngleDeg = 100.0;   // > 90 degrees, as required
    const float axis_inv_len = 1.0f / std::sqrt(3.0f);
    const float half = static_cast<float>(kTestAngleDeg * kPiD / 180.0) * 0.5f;
    const Quat q1{ 1.0f, 0.0f, 0.0f, 0.0f };
    const Quat q2{ std::cos(half), axis_inv_len * std::sin(half),
                   axis_inv_len * std::sin(half), axis_inv_len * std::sin(half) };

    const float test_ts[] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

    // (a) Geodesic-angle progression: slerp(q1,q2,t) must sit EXACTLY
    // t * (total angle) along the geodesic from q1 — the "constant angular
    // velocity" property the file-header comment names.
    double worst_angle_err = 0.0;
    for (float t : test_ts) {
        const Quat r = quat_slerp(q1, q2, t);
        double dot = static_cast<double>(quat_dot(q1, r));
        dot = dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot);
        const double angle = 2.0 * std::acos(dot);
        const double expected = static_cast<double>(t) * (kTestAngleDeg * kPiD / 180.0);
        worst_angle_err = std::max(worst_angle_err, std::fabs(angle - expected));
    }

    // (b) Sign-flip (double-cover) invariance: slerp(q1,-q2,t) must
    // represent the SAME rotation as slerp(q1,q2,t) for every t, because
    // -q2 IS q2 as a rotation — only quat_slerp's internal canonicalization
    // (the "if dot<0, negate" branch) makes this true; a naive SLERP
    // without it would swing the long way for one of the two calls.
    const Quat q2_neg{ -q2.w, -q2.x, -q2.y, -q2.z };
    double worst_sign_err = 0.0;
    for (float t : test_ts) {
        const Quat ra = quat_slerp(q1, q2, t);
        const Quat rb = quat_slerp(q1, q2_neg, t);
        const double dot = std::fabs(static_cast<double>(quat_dot(ra, rb)));
        worst_sign_err = std::max(worst_sign_err, std::fabs(1.0 - dot));
    }

    SlerpTestResult r;
    r.angle_err_rad = worst_angle_err;
    r.sign_err = worst_sign_err;
    r.pass = (worst_angle_err <= kSlerpAngleTolRad) && (worst_sign_err <= kSlerpSignDotTol);
    return r;
}

// ===========================================================================
// Artifact writers — demo/out/. All three kept intentionally simple
// (CLAUDE.md §6.3: a PNG/CSV/OBJ artifact for anything visual; PGM is a
// hand-rolled image format needing no library, in the same "no black
// boxes" spirit as this project's other self-written primitives).
// ===========================================================================

// write_pgm_triptych — a single grayscale PGM (P5) image: three top-view
// (x,y) scatter panels side by side — truth | skewed | deskewed(dense) —
// for ONE representative cohort, sharing a common world-space window so the
// three panels are directly comparable. This is the "money shot": a
// straight wall smeared/warped in the SKEWED panel snapping back flat in
// the DESKEWED panel (README "Expected output").
static bool write_pgm_triptych(const std::string& path,
                               const std::vector<float>& truth_xy_n3,
                               const std::vector<float>& skewed_xy_n3,
                               const std::vector<float>& deskewed_xy_n3,
                               int n_points)
{
    const int panel_w = 220, panel_h = 220, gap = 6;
    const int W = panel_w * 3 + gap * 2;
    const int H = panel_h;
    std::vector<unsigned char> img(static_cast<size_t>(W) * H, 20);   // dark background

    // Shared world-space window: the union extent of all three panels, plus
    // a small margin, so a fixed-size wall/room feature lines up across
    // panels at the SAME pixel scale (an unfair per-panel autoscale would
    // hide exactly the "thickening" effect this artifact exists to show).
    float xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f;
    auto expand = [&](const std::vector<float>& a) {
        for (int i = 0; i < n_points; ++i) {
            const float x = a[static_cast<size_t>(i) * 3 + 0];
            const float y = a[static_cast<size_t>(i) * 3 + 1];
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
        }
    };
    expand(truth_xy_n3); expand(skewed_xy_n3); expand(deskewed_xy_n3);
    if (xmax <= xmin) xmax = xmin + 1.0f;
    if (ymax <= ymin) ymax = ymin + 1.0f;
    const float margin_x = 0.05f * (xmax - xmin), margin_y = 0.05f * (ymax - ymin);
    xmin -= margin_x; xmax += margin_x; ymin -= margin_y; ymax += margin_y;

    // Fill the two 1-px divider gaps a slightly brighter gray so the eye
    // reads three distinct panels rather than one smeared image.
    for (int gseg = 0; gseg < 2; ++gseg)
        for (int gx = 0; gx < gap; ++gx)
            for (int y = 0; y < H; ++y)
                img[static_cast<size_t>(y) * W + (panel_w * (gseg + 1) + gap * gseg + gx)] = 60;

    auto splat = [&](const std::vector<float>& a, int panel_index) {
        const int x0 = panel_index * (panel_w + gap);
        for (int i = 0; i < n_points; ++i) {
            const float x = a[static_cast<size_t>(i) * 3 + 0];
            const float y = a[static_cast<size_t>(i) * 3 + 1];
            int col = static_cast<int>((x - xmin) / (xmax - xmin) * (panel_w - 1));
            // Row 0 = image TOP; larger y (world "left") drawn HIGHER, the
            // usual top-view/north-up map convention.
            int row = static_cast<int>((ymax - y) / (ymax - ymin) * (panel_h - 1));
            if (col < 0 || col >= panel_w || row < 0 || row >= panel_h) continue;
            img[static_cast<size_t>(row) * W + (x0 + col)] = 255;
        }
    };
    splat(truth_xy_n3, 0);
    splat(skewed_xy_n3, 1);
    splat(deskewed_xy_n3, 2);

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
    return static_cast<bool>(f);
}

// write_errors_csv — per-point error breakdown for one cohort: raw
// (undeskewed), dense-regime deskewed, and sparse-regime deskewed error
// against instantaneous truth. Feeds README Exercise "plot the error
// distribution"; also the raw material behind every mean/max this program
// prints on its [info] lines.
static bool write_errors_csv(const std::string& path, const Cohort& c,
                             const std::vector<double>& err_undeskewed,
                             const std::vector<double>& err_dense,
                             const std::vector<double>& err_sparse)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "point_index,t_s,beam_id,err_skewed_m,err_deskew_dense_m,err_deskew_sparse_m\n";
    for (int i = 0; i < c.n_points; ++i) {
        f << i << ',' << c.t[static_cast<size_t>(i)] << ',' << c.beam_id[static_cast<size_t>(i)] << ','
          << err_undeskewed[static_cast<size_t>(i)] << ','
          << err_dense[static_cast<size_t>(i)] << ','
          << err_sparse[static_cast<size_t>(i)] << '\n';
    }
    return static_cast<bool>(f);
}

// GateRow — one line of the gates_metrics.csv summary artifact.
struct GateRow { std::string metric, cohort, unit; double value; };

static bool write_gates_csv(const std::string& path, const std::vector<GateRow>& rows)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "metric,cohort,value,unit\n";
    for (const auto& r : rows)
        f << r.metric << ',' << r.cohort << ',' << r.value << ',' << r.unit << '\n';
    return static_cast<bool>(f);
}

// ===========================================================================
// main — see the file header for the six-stage pipeline this orchestrates.
// ===========================================================================
int main(int argc, char** argv)
{
    // ---- 0) Arguments: an optional --data override, same pattern 08.01's
    //         scenario loader uses (README documents the default path). ----
    std::string cli_data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data-dir") && i + 1 < argc) {
            cli_data_dir = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [--data-dir DIR]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] per-point motion deskew with pose interpolation (project 02.08)\n");
    print_device_info();
    std::printf("PROBLEM: deskew points from 4 motion cohorts (straight/arc/wiggle/stationary) "
                "via LERP+SLERP pose interpolation, sweep %.0f ms, %d beams x %d azimuth steps\n",
                static_cast<double>(kSweepDurationS) * 1000.0, kNumBeams, kAzimuthSteps);

    // ---- 1) Load the committed sample --------------------------------------
    const std::string data_path = find_data_file(cli_data_dir, argv[0], "deskew_scan.bin");
    if (data_path.empty()) {
        std::printf("DATA: NOT FOUND — data/sample/deskew_scan.bin missing "
                    "(run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }
    Sample sample = load_sample(data_path);
    if (!sample.loaded) {
        std::printf("DATA: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sample data malformed)\n");
        return 1;
    }
    int total_points = 0;
    for (int c = 0; c < kNumCohorts; ++c) total_points += sample.cohorts[c].n_points;
    std::printf("DATA: loaded data/sample/deskew_scan.bin (synthetic, %d cohorts, %d points total) [synthetic]\n",
                kNumCohorts, total_points);
    for (int c = 0; c < kNumCohorts; ++c)
        std::printf("[info] cohort '%s': %d points\n", cohort_name(c), sample.cohorts[c].n_points);

    // Precompute, per cohort: the sparse trajectory and the reference pose
    // (shared by both regimes — see ref_pose_of's comment).
    std::vector<float> sparse_traj[kNumCohorts];
    Pose ref_pose[kNumCohorts];
    for (int c = 0; c < kNumCohorts; ++c) {
        sparse_traj[c] = sparse_trajectory_of(sample.cohorts[c].dense_traj);
        ref_pose[c] = ref_pose_of(sample.cohorts[c]);
    }

    // ======================= 2) VERIFY stage =================================
    // GPU vs CPU twin, every cohort x both regimes (8 runs). Both paths call
    // the IDENTICAL shared HD deskew_one_point (kernels.cuh) — see
    // reference_cpu.cpp's file header for what this DOES and does NOT prove.
    bool verify_pass = true;
    double verify_worst_m = 0.0;
    double total_cpu_ms = 0.0, total_gpu_ms = 0.0;
    for (int c = 0; c < kNumCohorts; ++c) {
        const Cohort& co = sample.cohorts[c];
        for (int regime = 0; regime < 2; ++regime) {
            const std::vector<float>& traj = (regime == 0) ? co.dense_traj : sparse_traj[c];
            const int n_samples = (regime == 0) ? kDenseSamples : kSparseSamples;

            float gpu_ms = 0.0f;
            std::vector<float> gpu_out = run_deskew_gpu(co, traj, n_samples, ref_pose[c], gpu_ms);
            double cpu_ms = 0.0;
            std::vector<float> cpu_out = run_deskew_cpu_wrap(co, traj, n_samples, ref_pose[c], cpu_ms);
            total_gpu_ms += gpu_ms;
            total_cpu_ms += cpu_ms;

            const ErrStats twin = compute_error_stats(gpu_out, cpu_out, co.n_points);
            verify_worst_m = std::max(verify_worst_m, twin.max_v);
            if (twin.max_v > kVerifyTolM) verify_pass = false;
        }
    }
    std::printf("[info] VERIFY: worst GPU-vs-CPU per-point deviation %.3e m over %d cohorts x 2 regimes "
                "(tol %.1e m)\n", verify_worst_m, kNumCohorts, static_cast<double>(kVerifyTolM));
    std::printf("[time] twin comparison: CPU total %.2f ms | GPU kernel total %.3f ms (8 launches, tiny N each)\n",
                total_cpu_ms, total_gpu_ms);
    std::printf("VERIFY: %s (GPU deskew matches CPU reference within tol %.1e m, all cohorts x both regimes)\n",
                verify_pass ? "PASS" : "FAIL", static_cast<double>(kVerifyTolM));

    // ======================= 3) SLERP_CORRECTNESS =============================
    const SlerpTestResult slerp_test = test_slerp_correctness();
    std::printf("[info] SLERP test: geodesic-angle error %.3e rad, sign-flip dot error %.3e "
                "(tol %.1e rad, %.1e)\n",
                slerp_test.angle_err_rad, slerp_test.sign_err, kSlerpAngleTolRad, kSlerpSignDotTol);
    std::printf("SLERP_CORRECTNESS: %s (geodesic angle progression + double-cover sign-flip invariance within tolerance)\n",
                slerp_test.pass ? "PASS" : "FAIL");

    // ======= 4) Deskew every cohort, both regimes, on the GPU (the =======
    // ======= "production" results every remaining gate measures)   =======
    std::vector<float> deskewed_dense[kNumCohorts];
    std::vector<float> deskewed_sparse[kNumCohorts];
    float dense_kernel_ms[kNumCohorts] = {};
    float sparse_kernel_ms[kNumCohorts] = {};
    for (int c = 0; c < kNumCohorts; ++c) {
        const Cohort& co = sample.cohorts[c];
        deskewed_dense[c] = run_deskew_gpu(co, co.dense_traj, kDenseSamples, ref_pose[c], dense_kernel_ms[c]);
        deskewed_sparse[c] = run_deskew_gpu(co, sparse_traj[c], kSparseSamples, ref_pose[c], sparse_kernel_ms[c]);
    }

    // Per-cohort error stats against instantaneous truth: undeskewed
    // (negative control), dense-regime deskewed, sparse-regime deskewed.
    ErrStats err_undeskewed[kNumCohorts], err_dense[kNumCohorts], err_sparse[kNumCohorts];
    std::vector<double> pp_undeskewed[kNumCohorts], pp_dense[kNumCohorts], pp_sparse[kNumCohorts];
    for (int c = 0; c < kNumCohorts; ++c) {
        const Cohort& co = sample.cohorts[c];
        err_undeskewed[c] = compute_error_stats(co.xyz_local, co.xyz_truth, co.n_points);
        err_dense[c]      = compute_error_stats(deskewed_dense[c], co.xyz_truth, co.n_points);
        err_sparse[c]     = compute_error_stats(deskewed_sparse[c], co.xyz_truth, co.n_points);
        pp_undeskewed[c]  = per_point_errors(co.xyz_local, co.xyz_truth, co.n_points);
        pp_dense[c]       = per_point_errors(deskewed_dense[c], co.xyz_truth, co.n_points);
        pp_sparse[c]      = per_point_errors(deskewed_sparse[c], co.xyz_truth, co.n_points);

        std::printf("[info] cohort '%s': undeskewed mean=%.4f max=%.4f m | dense mean=%.6f max=%.6f m | "
                    "sparse mean=%.6f max=%.6f m\n",
                    cohort_name(c), err_undeskewed[c].mean, err_undeskewed[c].max_v,
                    err_dense[c].mean, err_dense[c].max_v, err_sparse[c].mean, err_sparse[c].max_v);
    }

    // ======================= 5a) IDENTITY_CONTROL (cohort 3) =================
    const bool identity_pass = (err_dense[kCohortStationary].max_v <= kIdentityToleranceM) &&
                               (err_sparse[kCohortStationary].max_v <= kIdentityToleranceM);
    std::printf("[info] IDENTITY_CONTROL: stationary max deskewed-vs-truth error %.3e m (dense), "
                "%.3e m (sparse), tol %.1e m\n",
                err_dense[kCohortStationary].max_v, err_sparse[kCohortStationary].max_v,
                static_cast<double>(kIdentityToleranceM));
    std::printf("IDENTITY_CONTROL: %s (stationary cohort: deskew is a no-op within tolerance)\n",
                identity_pass ? "PASS" : "FAIL");

    // ======================= 5b) RESTORATION (cohorts 0,1,2) =================
    bool restoration_pass = true;
    for (int c = 0; c < 3; ++c) {
        const bool cohort_ok =
            (err_dense[c].mean <= kRestorationMeanBoundM[c]) &&
            (err_dense[c].max_v <= kRestorationMaxBoundM[c]) &&
            (err_undeskewed[c].mean >= kMinUndeskewedBaselineM);
        if (!cohort_ok) restoration_pass = false;
        std::printf("[info] RESTORATION '%s': dense mean %.6f m <= bound %.3f m ? %s | "
                    "undeskewed baseline %.4f m >= floor %.2f m ? %s\n",
                    cohort_name(c), err_dense[c].mean, static_cast<double>(kRestorationMeanBoundM[c]),
                    (err_dense[c].mean <= kRestorationMeanBoundM[c]) ? "yes" : "NO",
                    err_undeskewed[c].mean, static_cast<double>(kMinUndeskewedBaselineM),
                    (err_undeskewed[c].mean >= kMinUndeskewedBaselineM) ? "yes" : "NO");
    }
    std::printf("RESTORATION: %s (dense-sampled deskew recovers instantaneous truth on all 3 moving "
                "cohorts; undeskewed baseline order-of-magnitude worse)\n",
                restoration_pass ? "PASS" : "FAIL");

    // ======================= 5c) SAMPLING_LESSON ==============================
    // (i) wiggle: sparse/dense ratio must clear the floor.
    const double wiggle_ratio = (err_dense[kCohortWiggle].mean > 1e-12)
                                ? (err_sparse[kCohortWiggle].mean / err_dense[kCohortWiggle].mean)
                                : 0.0;
    const bool sampling_ratio_pass = wiggle_ratio >= kSamplingRatioFloor;
    // (ii) straight: dense and sparse regimes must AGREE (both tiny, and
    // close to each other) — constant velocity makes 2-sample interpolation
    // exact, so this is a consistency check between the two code paths.
    const bool sampling_consistency_pass =
        (err_dense[kCohortStraight].mean <= kConsistencyAbsBoundM) &&
        (err_sparse[kCohortStraight].mean <= kConsistencyAbsBoundM) &&
        (std::fabs(err_dense[kCohortStraight].mean - err_sparse[kCohortStraight].mean) <= kConsistencyDiffBoundM);
    const bool sampling_pass = sampling_ratio_pass && sampling_consistency_pass;
    std::printf("[info] SAMPLING_LESSON: wiggle sparse/dense mean-error ratio = %.2fx (floor %.1fx) | "
                "straight dense=%.6f m sparse=%.6f m diff=%.6f m (bounds %.3f / %.3f m)\n",
                wiggle_ratio, static_cast<double>(kSamplingRatioFloor),
                err_dense[kCohortStraight].mean, err_sparse[kCohortStraight].mean,
                std::fabs(err_dense[kCohortStraight].mean - err_sparse[kCohortStraight].mean),
                static_cast<double>(kConsistencyAbsBoundM), static_cast<double>(kConsistencyDiffBoundM));
    std::printf("SAMPLING_LESSON: %s (wiggle cohort: sparse/dense error ratio clears the floor; "
                "straight cohort: dense and sparse regimes agree, as constant velocity predicts)\n",
                sampling_pass ? "PASS" : "FAIL");

    // ======================= 5d) DOWNSTREAM_PAYOFF ============================
    // Wall plane-fit RMS thickness on the straight cohort (identity
    // orientation, so local +x = world +x throughout the sweep — the
    // cleanest cohort for this measurement; see select_wall_indices).
    const Cohort& straight = sample.cohorts[kCohortStraight];
    const std::vector<int> wall_idx = select_wall_indices(straight);
    auto gather = [&](const std::vector<float>& xyz) {
        std::vector<Vec3> pts;
        pts.reserve(wall_idx.size());
        for (int i : wall_idx)
            pts.push_back(Vec3{ xyz[static_cast<size_t>(i) * 3 + 0],
                                xyz[static_cast<size_t>(i) * 3 + 1],
                                xyz[static_cast<size_t>(i) * 3 + 2] });
        return pts;
    };
    const double wall_skewed_rms = plane_fit_rms(gather(straight.xyz_local));
    const double wall_deskewed_rms = plane_fit_rms(gather(deskewed_dense[kCohortStraight]));
    const double wall_truth_rms = plane_fit_rms(gather(straight.xyz_truth));
    const bool downstream_pass =
        (wall_idx.size() >= 8) &&
        (wall_skewed_rms >= kWallSkewedRmsMinM) &&
        (wall_deskewed_rms <= kWallDeskewedRmsMaxM) &&
        (wall_skewed_rms >= kWallImprovementFactor * wall_deskewed_rms);
    std::printf("[info] DOWNSTREAM_PAYOFF: wall plane-fit RMS (n=%zu points) — skewed=%.4f m, "
                "deskewed=%.4f m, instantaneous truth=%.4f m (a wall the naive scan smears to %.1fx "
                "the deskewed thickness)\n",
                wall_idx.size(), wall_skewed_rms, wall_deskewed_rms, wall_truth_rms,
                (wall_deskewed_rms > 1e-9) ? (wall_skewed_rms / wall_deskewed_rms) : 0.0);
    std::printf("DOWNSTREAM_PAYOFF: %s (wall plane-fit RMS tightens after deskew, straight cohort)\n",
                downstream_pass ? "PASS" : "FAIL");

    // ======================= 6) Timing summary ================================
    double total_deskew_ms = 0.0;
    for (int c = 0; c < kNumCohorts; ++c) total_deskew_ms += dense_kernel_ms[c] + sparse_kernel_ms[c];
    std::printf("[time] full deskew pass (all %d cohorts, both regimes, %d points total): "
                "%.3f ms GPU kernel time -- well inside the 50-100 ms budget a 10-20 Hz spinning "
                "LiDAR allows per sweep (SYSTEM_DESIGN item 1; README System context)\n",
                kNumCohorts, total_points, total_deskew_ms);

    // ======================= 7) Artifacts ======================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifacts_ok = true;

    const bool triptych_ok = write_pgm_triptych(
        out_dir + "/triptych_wiggle.pgm",
        sample.cohorts[kCohortWiggle].xyz_truth,
        sample.cohorts[kCohortWiggle].xyz_local,
        deskewed_dense[kCohortWiggle],
        sample.cohorts[kCohortWiggle].n_points);
    artifacts_ok = artifacts_ok && triptych_ok;
    std::printf(triptych_ok ? "ARTIFACT: wrote demo/out/triptych_wiggle.pgm\n"
                            : "ARTIFACT: FAILED to write demo/out/triptych_wiggle.pgm\n");

    std::vector<GateRow> gate_rows;
    for (int c = 0; c < kNumCohorts; ++c) {
        gate_rows.push_back({ "undeskewed_mean", cohort_name(c), "m", err_undeskewed[c].mean });
        gate_rows.push_back({ "undeskewed_max",  cohort_name(c), "m", err_undeskewed[c].max_v });
        gate_rows.push_back({ "deskewed_dense_mean",  cohort_name(c), "m", err_dense[c].mean });
        gate_rows.push_back({ "deskewed_dense_max",   cohort_name(c), "m", err_dense[c].max_v });
        gate_rows.push_back({ "deskewed_sparse_mean", cohort_name(c), "m", err_sparse[c].mean });
        gate_rows.push_back({ "deskewed_sparse_max",  cohort_name(c), "m", err_sparse[c].max_v });
    }
    gate_rows.push_back({ "verify_worst_deviation", "all", "m", verify_worst_m });
    gate_rows.push_back({ "slerp_angle_error", "n/a", "rad", slerp_test.angle_err_rad });
    gate_rows.push_back({ "slerp_sign_flip_error", "n/a", "unitless", slerp_test.sign_err });
    gate_rows.push_back({ "wiggle_sparse_dense_ratio", "wiggle", "unitless", wiggle_ratio });
    gate_rows.push_back({ "wall_rms_skewed", "straight", "m", wall_skewed_rms });
    gate_rows.push_back({ "wall_rms_deskewed", "straight", "m", wall_deskewed_rms });
    gate_rows.push_back({ "wall_rms_truth", "straight", "m", wall_truth_rms });
    gate_rows.push_back({ "total_deskew_kernel_ms", "all", "ms", total_deskew_ms });
    const bool gates_csv_ok = write_gates_csv(out_dir + "/gates_metrics.csv", gate_rows);
    artifacts_ok = artifacts_ok && gates_csv_ok;
    std::printf(gates_csv_ok ? "ARTIFACT: wrote demo/out/gates_metrics.csv\n"
                             : "ARTIFACT: FAILED to write demo/out/gates_metrics.csv\n");

    for (int c = 0; c < kNumCohorts; ++c) {
        const std::string path = out_dir + "/errors_" + cohort_name(c) + ".csv";
        const bool ok = write_errors_csv(path, sample.cohorts[c], pp_undeskewed[c], pp_dense[c], pp_sparse[c]);
        artifacts_ok = artifacts_ok && ok;
        std::printf(ok ? "ARTIFACT: wrote demo/out/errors_%s.csv\n"
                       : "ARTIFACT: FAILED to write demo/out/errors_%s.csv\n", cohort_name(c));
    }

    // ======================= 8) Final verdict ==================================
    const bool all_pass = verify_pass && slerp_test.pass && identity_pass && restoration_pass &&
                          sampling_pass && downstream_pass && artifacts_ok;
    if (all_pass) {
        std::printf("RESULT: PASS (all gates passed)\n");
        return EXIT_SUCCESS;
    }
    std::printf("RESULT: FAIL (see the gate lines above for which check failed)\n");
    return EXIT_FAILURE;
}
