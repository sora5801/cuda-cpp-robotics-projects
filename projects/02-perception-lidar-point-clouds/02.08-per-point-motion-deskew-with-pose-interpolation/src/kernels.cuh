// ===========================================================================
// kernels.cuh — interface for project 02.08
//               Per-point motion deskew with pose interpolation
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates), kernels.cu (the
// one GPU kernel), reference_cpu.cpp (its independent CPU twin), and
// ../scripts/make_synthetic.py (whose "MUST MATCH kernels.cuh" comments
// mirror the geometry/timing constants below). Everything all four must
// agree on — the beam/sweep timing model, the trajectory-sample layout, and
// the deskew math itself — is defined HERE, once (CLAUDE.md §12).
//
// THE PROBLEM in one paragraph (THEORY.md "The problem" derives every step)
// ----------------------------------------------------------------------------
// A spinning LiDAR does not capture a scene instantaneously: it fires
// kNumBeams beams at kAzimuthSteps azimuth steps across one sweep of
// duration kSweepDurationS (~100 ms — a realistic 10 Hz spin rate). Point i
// is measured in the sensor's OWN, instantaneous body frame AT ITS FIRING
// TIME t_i = az_step_i / kAzimuthSteps * kSweepDurationS — NOT in one shared
// frame. If the platform is moving, naively stacking every point's raw local
// coordinates together (exactly what a driver's packet does with no motion
// compensation) smears the scan: a flat wall becomes a slanted or thickened
// blur (README "Expected output" measures this).
//
// THE FIX in one paragraph (THEORY.md "The algorithm" / "The math")
// ----------------------------------------------------------------------------
// Given the platform's pose trajectory over the sweep — position + unit
// quaternion, sampled at either a DENSE (~200 Hz-equivalent) or SPARSE
// (2-sample: start/end only) rate — every point is re-expressed in the
// frame of a single REFERENCE instant t_ref = kSweepDurationS (sweep END;
// see "Why sweep END" below). This is a two-step per-point job, fully
// data-parallel (embarrassingly so — one thread per point, zero interaction
// between threads):
//   1. INTERPOLATE the platform's pose at the point's own firing time t_i
//      (position LERP + orientation SLERP between the two bracketing
//      trajectory samples — interpolate_pose() below) AND, once per cohort,
//      at t_ref (main.cu computes this ONE TIME and passes it to every
//      thread — the "ref_pose" parameter, a uniform/broadcast read).
//   2. RE-PROJECT the point's local coordinates into the reference frame via
//      the RIGID transform between the two interpolated poses
//      (deskew_one_point() below).
//
// Why sweep END, not sweep START, as the reference instant? Two reasons this
// project's downstream consumers care about (README "System context"): (a)
// it is the instant closest to "now" when the scan finishes arriving, so a
// planner reacting to this scan is reacting to the freshest ego-pose
// estimate available; (b) it matches this repo's and most real drivers'
// convention of stamping a scan with its END time. THEORY.md "The math"
// gives the one-line reason the CHOICE of t_ref does not change how well
// deskew works, only which frame the output lands in.
//
// QUATERNION CONVENTION — repo order (w, x, y, z), UNIT-NORM, "child
// (platform body) expressed in parent (world)" (T_parent_child convention,
// docs/SYSTEM_DESIGN.md §3.3/3.4; SAME order 09.01's forward-kinematics
// model and 01.10's gyro-integrated trajectory use — cite both: 09.01 for
// the convention itself, 01.10 for "a CONTINUOUS trajectory of it, sampled
// and interpolated" being the exact shape of THIS project's problem, one
// level up — 01.10 interpolates a rotation-only trajectory by ROW (image
// rows standing in for time); this project interpolates a full SE(3)
// trajectory by POINT TIMESTAMP, and needs a real SLERP (not 01.10's
// LERP-is-good-enough-for-adjacent-rows shortcut) because trajectory
// SAMPLES here can be tens of milliseconds apart, not one line-time.
//
// TRAJECTORY LAYOUT — a flat float array, kTrajStride (=8) floats per
// sample, samples in ASCENDING time order:
//     traj[i*8 + 0]         = t_s        sample timestamp, seconds (sweep-relative)
//     traj[i*8 + 1..3]      = px,py,pz   platform position, meters, WORLD frame
//     traj[i*8 + 4..7]      = qw,qx,qy,qz platform orientation, unit quaternion
// TWO REGIMES read this SAME layout with a different sample COUNT: the
// DENSE regime passes all kDenseSamples samples; the SPARSE regime passes
// just samples [0] and [kDenseSamples-1] (main.cu extracts these — no
// separate sparse array is ever generated or stored; one honest source of
// ground truth, two ways of subsampling it — see main.cu "Two regimes,
// one array"). interpolate_pose() below is IDENTICAL code for both; only
// the array length n_samples changes what "bracketing samples" means.
//
// POINT LAYOUT (main.cu's per-cohort host/device buffers):
//     t_points[i]        = firing time, seconds (sweep-relative)
//     xyz_local[i*3+0..2] = SKEWED point, meters, sensor-local frame AT t_points[i]
//     xyz_out[i*3+0..2]   = DESKEWED point, meters, sensor-local frame AT t_ref (OUT)
//
// Why this header is CUDA-qualifier-free where possible, HD elsewhere
// ---------------------------------------------------------------------
// Every quaternion/vector/interpolation primitive below is declared HD
// ("__host__ __device__" under nvcc, nothing under cl.exe — same macro
// 01.10/09.01/18.01 use). These are SHARED, token-for-token, by
// kernels.cu's kernel and reference_cpu.cpp's twin: per the independence
// ruling in reference_cpu.cpp's file header, duplicating quaternion algebra
// by hand would be pure transcription (a "camera/pose model" exception,
// exactly 01.10's precedent). The OUTER per-point LOOP — reading input,
// calling deskew_one_point, writing output — is typed TWICE, independently,
// in kernels.cu (__global__, thread-per-point) and reference_cpu.cpp
// (serial for-loop). What makes this honest despite the shared math: this
// project's verification gates (main.cu) compare against ANALYTIC ground
// truth from ../scripts/make_synthetic.py — a continuous trajectory neither
// this header nor the twins have ever seen — not merely GPU-vs-CPU
// agreement (THEORY.md "How we verify correctness" spells this out).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>   // sqrtf/sinf/cosf/acosf/floorf — used by the HD helpers below

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. Same trick as
// 01.10/09.01/18.01 (see that file-header discussion for the full rationale):
// lets kernels.cu's kernel and reference_cpu.cpp's twin call the IDENTICAL
// compiled-twice primitive without either translation unit seeing CUDA
// keywords it cannot parse.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// Beam model + sweep timing — MUST MATCH ../scripts/make_synthetic.py's
// module-level constants of the same name (main.cu asserts the file
// header's num_beams/azimuth_steps/n_dense_samples against these at load
// time — a data/code consistency check, not a coincidence, same discipline
// 02.01's kVoxelLeafM assertion uses).
// ===========================================================================

// 16-beam elevation table, -15..+15 deg in 2 deg steps — the SAME model
// 02.01 cites from 01.18's THEORY.md, reused verbatim here (see that
// project's data/README.md for the original derivation). Declared as a
// plain array (not __constant__): it is read only on the HOST, by main.cu's
// wall-point selector for the downstream-payoff gate (README "Expected
// output") — the deskew kernel itself never needs elevation, only the
// per-point timestamp and local xyz.
constexpr int kNumBeams = 16;
constexpr float kBeamElevDeg[kNumBeams] = {
    -15.0f, -13.0f, -11.0f, -9.0f, -7.0f, -5.0f, -3.0f, -1.0f,
      1.0f,   3.0f,   5.0f,  7.0f,  9.0f, 11.0f, 13.0f, 15.0f
};

constexpr int   kAzimuthSteps    = 360;    // 1 deg/step single-sweep resolution (module docstring, make_synthetic.py)
constexpr float kSweepDurationS  = 0.100f; // 100 ms per revolution (10 Hz spin rate)

// Number of cohorts and their FIXED order in the committed sample file
// (../scripts/make_synthetic.py's write_binary_sample / main.cu's loader
// both walk this exact order — never re-sort by anything).
constexpr int kNumCohorts = 4;
enum CohortId : int { kCohortStraight = 0, kCohortArc = 1, kCohortWiggle = 2, kCohortStationary = 3 };
inline const char* cohort_name(int id)
{
    static const char* kNames[kNumCohorts] = { "straight", "arc", "wiggle", "stationary" };
    return (id >= 0 && id < kNumCohorts) ? kNames[id] : "?";
}

// Dense trajectory sample count: a 200 Hz-equivalent discretization of the
// sweep (21 samples at 5 ms spacing span [0, 100] ms inclusive of both
// ends — make_synthetic.py's DENSE_SAMPLES). The SPARSE regime is derived
// from this SAME array by main.cu (first + last sample only) — see the
// file header's "TRAJECTORY LAYOUT" note.
constexpr int kDenseSamples = 21;
constexpr int kSparseSamples = 2;

// Floats per trajectory sample: t_s, px,py,pz, qw,qx,qy,qz (file header).
constexpr int kTrajStride = 8;

// Upper bound on trajectory sample count a single launch/upload can carry —
// sized with headroom over kDenseSamples so a learner experimenting with a
// denser regime (README Exercise) does not silently truncate. Bounds the
// __constant__ memory buffer kernels.cu declares (kMaxTrajSamples*8 floats
// = 1024 bytes — trivially inside the 64 KiB constant-memory budget).
constexpr int kMaxTrajSamples = 64;

// ---------------------------------------------------------------------------
// Vec3 / Quat — plain structs, no vendor math library (CLAUDE.md §1: no
// black boxes). Quat is REPO ORDER (w,x,y,z) — see the file header.
// ---------------------------------------------------------------------------
struct Vec3 { float x, y, z; };
struct Quat { float w, x, y, z; };

// A rigid pose: WORLD position (m) + orientation quaternion. Small POD
// (28 bytes), passed BY VALUE to the kernel — the same "small POD by value"
// reasoning 02.06's Rigid3 / 08.01's x0 comment gives: cheaper than a
// pointer indirection for something this small, and every thread reads the
// SAME ref_pose (a broadcast, not a divergent per-thread load).
struct Pose { Vec3 p; Quat q; };

// ===========================================================================
// Vec3 helpers — HD, shared verbatim by the GPU kernel and the CPU twin.
// ===========================================================================
HD inline Vec3 vec3_add(Vec3 a, Vec3 b) { return Vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
HD inline Vec3 vec3_sub(Vec3 a, Vec3 b) { return Vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
HD inline Vec3 vec3_scale(Vec3 a, float s) { return Vec3{ a.x * s, a.y * s, a.z * s }; }
HD inline Vec3 vec3_cross(Vec3 a, Vec3 b)
{
    return Vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
// Position interpolation: for CONSTANT platform velocity over a sample
// interval, this is EXACT (no approximation at all) — the textbook reason
// translation only needs LERP while rotation needs the machinery below
// (THEORY.md "The problem" derives why: translation is a FLAT vector
// space, rotation is a CURVED manifold — see quat_slerp's comment).
HD inline Vec3 vec3_lerp(Vec3 a, Vec3 b, float t) { return vec3_add(a, vec3_scale(vec3_sub(b, a), t)); }

// ===========================================================================
// Quaternion helpers — HD, shared verbatim. Textbook formulas written out
// term-by-term (CLAUDE.md §1: no black-box quaternion class).
// ===========================================================================

// quat_dot — the 4-D dot product; also cos(half the geodesic angle) between
// two UNIT quaternions when both point "the same way" (see quat_slerp).
HD inline float quat_dot(Quat a, Quat b) { return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; }

// quat_normalize — renormalize to unit length. Called after every multiply
// (chained products drift off the unit sphere from float rounding — the
// same "quaternion normalization drift" hazard CLAUDE.md §12 names and
// 01.10's quat_normalize comment discusses at length).
HD inline Quat quat_normalize(Quat q)
{
    const float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n < 1e-20f) { Quat id{ 1.0f, 0.0f, 0.0f, 0.0f }; return id; }   // degenerate guard: identity, never divide by ~0
    const float inv_n = 1.0f / n;
    return Quat{ q.w * inv_n, q.x * inv_n, q.y * inv_n, q.z * inv_n };
}

// quat_mul — Hamilton product a (x) b, repo order (w,x,y,z). Composing
// rotations: applying b THEN a to a vector equals applying quat_mul(a,b).
HD inline Quat quat_mul(Quat a, Quat b)
{
    return Quat{
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

// quat_conj — (w,-x,-y,-z); for a UNIT quaternion this is also the inverse:
// R(conj(q)) = R(q)^-1 = R(q)^T (same identity 01.10's quat_conj comment
// cites). Used constantly below to go "from world into a body frame".
HD inline Quat quat_conj(Quat q) { return Quat{ q.w, -q.x, -q.y, -q.z }; }

// quat_rotate — rotate vector v by unit quaternion q via the optimized
// "sandwich product" v' = q*v*conj(q), expanded algebraically (substitute v
// as the pure quaternion (0,v) and simplify using q = w + u; the standard
// derivation appears in any robotics quaternion reference, e.g. Sola 2017
// "Quaternion kinematics for the error-state KF" eq. 55-56). Two cross
// products + a few FMAs — cheaper than two full quaternion multiplies for
// the identical result, and the formula used both here and in
// ../scripts/make_synthetic.py's Python mirror (kept independent — see that
// script's file header — but implementing the SAME textbook identity).
HD inline Vec3 quat_rotate(Quat q, Vec3 v)
{
    const Vec3 u{ q.x, q.y, q.z };                       // the quaternion's vector part (rotation axis * sin(theta/2))
    const Vec3 t = vec3_scale(vec3_cross(u, v), 2.0f);   // t = 2 * (u x v)
    return vec3_add(vec3_add(v, vec3_scale(t, q.w)), vec3_cross(u, t));  // v' = v + w*t + (u x t)
}

// ---------------------------------------------------------------------------
// quat_slerp — Spherical Linear intERPolation between unit quaternions a
// and b, at parameter t in [0,1]. THIS is the function that makes rotation
// interpolation fundamentally different from position interpolation
// (vec3_lerp above): unit quaternions live on the 4-D unit SPHERE (a curved
// manifold), and LERP-then-normalize takes the CHORD across that sphere
// instead of the geodesic ARC along it — for small angular gaps between
// samples the two are nearly indistinguishable (the small-angle fallback
// below exploits exactly this), but for the tens-of-degrees gaps this
// project's SPARSE (2-sample) regime can span, the difference is the whole
// lesson (README "Expected output", THEORY.md "The math" derives the
// geodesic argument in full).
//
// TWO CLASSIC GOTCHAS, both handled here, both load-bearing:
//
//  1) DOUBLE COVER (q and -q represent the IDENTICAL rotation — the unit
//     quaternions are a 2-to-1 cover of SO(3)). If quat_dot(a,b) < 0, a and
//     b sit on OPPOSITE hemispheres of the 4-sphere even though they may
//     represent nearby rotations; interpolating the raw 4-D arc between
//     them would swing through the LONG way around (more than 180 degrees
//     of physical rotation) — visibly wrong for any trajectory with a
//     rotation near +/-180 degrees. Negating b (b and -b are the SAME
//     rotation) flips it onto a's hemisphere without changing what it
//     represents, guaranteeing the SHORTEST geodesic path. main.cu's
//     SLERP_CORRECTNESS gate exercises this explicitly with a designed
//     >90-degree pair and checks the sign-flip invariance directly.
//
//  2) NEAR-PARALLEL / SMALL-ANGLE DIVISION: the exact formula below divides
//     by sin(theta), theta = acos(dot(a,b)). As a and b converge (dot -> 1),
//     sin(theta) -> 0 and the formula becomes an unstable 0/0 (and is
//     outright undefined at dot == 1, the common case of two IDENTICAL
//     consecutive trajectory samples or a stationary platform — this
//     project's cohort 3!). Below kParallelThreshold, this function falls
//     back to LINEAR interpolation of the four components, renormalized.
//     This is not a hack: sin(x) ~= x for small x is the first-order Taylor
//     expansion, so as theta -> 0 the exact weights
//     sin((1-t)*theta)/sin(theta) -> (1-t) and sin(t*theta)/sin(theta) -> t
//     — i.e. the LERP fallback is the Taylor-series LIMIT of the exact
//     SLERP formula, not an approximation of a different thing. THEORY.md
//     "Numerical considerations" bounds the crossover error.
// ---------------------------------------------------------------------------
HD inline Quat quat_slerp(Quat a, Quat b, float t)
{
    float d = quat_dot(a, b);

    // Gotcha 1: double cover. Flip b onto a's hemisphere (same rotation,
    // shortest path) — see the function comment above.
    if (d < 0.0f) { b = Quat{ -b.w, -b.x, -b.y, -b.z }; d = -d; }

    // Gotcha 2: near-parallel fallback (the Taylor-limit argument above).
    // 0.9995 corresponds to a geodesic angle of ~1.8 degrees between
    // samples — comfortably above float rounding noise, comfortably below
    // any angular gap this project's trajectories produce between adjacent
    // DENSE samples in the non-degenerate cohorts (measured in THEORY.md
    // "Numerical considerations"), so this branch is exercised mainly by
    // cohort 3 (stationary: every sample pair is IDENTICAL, dot == 1
    // exactly) and by any two adjacent dense samples in slow stretches.
    constexpr float kParallelThreshold = 0.9995f;
    if (d > kParallelThreshold) {
        Quat r{ a.w + (b.w - a.w) * t, a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t };
        return quat_normalize(r);
    }

    // Exact SLERP: interpolate ALONG THE GEODESIC at constant angular
    // velocity. This is the property this project's ARC cohort (constant
    // yaw RATE) exploits to make 2-sample SLERP reproduce the true
    // continuous orientation EXACTLY — a constant-angular-velocity rotation
    // traces the identical geodesic SLERP interpolates (README ties this to
    // the measured dense-vs-sparse orientation agreement on that cohort).
    const float d_clamped = d < 1.0f ? (d > -1.0f ? d : -1.0f) : 1.0f;  // acosf is undefined outside [-1,1]; float rounding can push d there
    const float theta = acosf(d_clamped);
    const float sin_theta = sinf(theta);
    const float wa = sinf((1.0f - t) * theta) / sin_theta;
    const float wb = sinf(t * theta) / sin_theta;
    Quat r{ wa * a.w + wb * b.w, wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z };
    return quat_normalize(r);
}

// ---------------------------------------------------------------------------
// find_bracket_index — the largest sample index i such that
// traj[i*kTrajStride] <= t < traj[(i+1)*kTrajStride], i.e. which two
// consecutive trajectory samples bracket query time t. Binary search over n
// samples (n is 2 for the SPARSE regime, kDenseSamples=21 for DENSE — TINY
// either way; kernels.cu's launch-configuration comment explains why this
// never dominates runtime: at n=21, binary search costs ~5 comparisons per
// point versus ~10 for a linear scan — both are noise next to the memory
// traffic of even one point's load/store. The teaching value is the
// PATTERN: a real production deskew node ingesting a 200 Hz pose stream
// over a longer window (n in the hundreds) is exactly where O(log n) binary
// search earns its keep over a linear scan — see THEORY.md "The GPU
// mapping" and README Exercise for the "grow n and re-profile" experiment.
//
// t is CLAMPED into the trajectory's covered span before searching: every
// real point's timestamp lies inside [traj[0], traj[(n-1)*stride]] by
// construction (the sweep never fires a beam before it starts or after it
// ends), so the clamp is a defensive guard, not this function's normal path.
// ---------------------------------------------------------------------------
HD inline int find_bracket_index(const float* traj, int n, float t)
{
    if (t <= traj[0]) return 0;
    if (t >= traj[(n - 1) * kTrajStride]) return n - 2;
    int lo = 0, hi = n - 1;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (traj[mid * kTrajStride] <= t) lo = mid; else hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// interpolate_pose — the pose of the platform at query time t, estimated
// from n DISCRETE trajectory samples (either regime — see the file header's
// "TRAJECTORY LAYOUT" note; this function does not know or care which).
//
// Parameters: traj — [n*kTrajStride] samples in ascending-time order (the
//             layout documented at the top of this file).
//             n    — sample count (>= 2; kSparseSamples or kDenseSamples in
//                     every caller in this project).
//             t    — query time, seconds (sweep-relative).
// Returns (by reference): p — LERP'd position, q — SLERP'd orientation.
//
// alpha (the interpolation fraction) is clamped to [0,1] as a second,
// belt-and-suspenders guard beyond find_bracket_index's own clamp — cheap,
// and it makes this function's contract ("always returns a value ON the
// segment between the two bracketing samples, never an extrapolation")
// independently true even if a future caller feeds it an out-of-range t.
// ---------------------------------------------------------------------------
HD inline void interpolate_pose(const float* traj, int n, float t, Vec3& p, Quat& q)
{
    const int i = find_bracket_index(traj, n, t);
    const float t0 = traj[i * kTrajStride];
    const float t1 = traj[(i + 1) * kTrajStride];
    float alpha = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;
    alpha = alpha < 0.0f ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);

    const float* s0 = traj + i * kTrajStride;
    const float* s1 = traj + (i + 1) * kTrajStride;
    const Vec3 p0{ s0[1], s0[2], s0[3] };
    const Vec3 p1{ s1[1], s1[2], s1[3] };
    const Quat q0{ s0[4], s0[5], s0[6], s0[7] };
    const Quat q1{ s1[4], s1[5], s1[6], s1[7] };

    p = vec3_lerp(p0, p1, alpha);
    q = quat_slerp(q0, q1, alpha);
}

// ---------------------------------------------------------------------------
// deskew_one_point — THE algorithm (everything above is machinery this
// function assembles). Re-express a point measured at time t_point, in the
// sensor's OWN local frame at that instant, into the REFERENCE frame
// described by ref_pose (main.cu computes ref_pose ONCE per cohort, via
// interpolate_pose at t = kSweepDurationS, and passes it to every thread).
//
// THE DERIVATION (THEORY.md "The math" walks this in full):
//   Let P_local be the point in the sensor's frame at firing time t_i, and
//   (p_i, q_i) = interpolate_pose(t_i) the platform's estimated pose then.
//   The point's WORLD position is:
//       P_world = p_i + R(q_i) * P_local
//   Re-expressing P_world in the reference sensor's frame (p_ref, q_ref):
//       P_ref = R(q_ref)^-1 * (P_world - p_ref)
//             = R(conj(q_ref)) * (p_i - p_ref)  +  R(conj(q_ref)) * R(q_i) * P_local
//             = t_rel                            +  R(q_rel) * P_local
//   where, using R(a)*R(b) = R(a (x) b) for the Hamilton product with this
//   repo's convention:
//       q_rel = conj(q_ref) (x) q_i        (the relative ROTATION from ref to i)
//       t_rel = R(conj(q_ref)) * (p_i - p_ref)   (the relative TRANSLATION, expressed in the ref frame)
//   This is the exact rigid transform "project the point's own instant into
//   the reference instant" — the per-point re-projection the catalog bullet
//   asks for, and it is a PURE MAP: every point's output depends only on
//   its own t_point/P_local and the two interpolated poses, never on any
//   other point (kernels.cu "The GPU mapping" leans on exactly this).
//
// Parameters: traj, n_samples — the trajectory (either regime).
//             ref_pose        — pose at t_ref, precomputed ONCE by the caller.
//             t_point         — this point's firing time, seconds.
//             p_local         — this point's SKEWED (raw, local-at-t_point) coordinates, meters.
// Returns: the point's coordinates in the reference frame, meters.
// ---------------------------------------------------------------------------
HD inline Vec3 deskew_one_point(const float* traj, int n_samples, Pose ref_pose,
                                float t_point, Vec3 p_local)
{
    Vec3 p_i; Quat q_i;
    interpolate_pose(traj, n_samples, t_point, p_i, q_i);

    const Quat q_ref_conj = quat_conj(ref_pose.q);
    const Quat q_rel = quat_normalize(quat_mul(q_ref_conj, q_i));
    const Vec3 t_rel = quat_rotate(q_ref_conj, vec3_sub(p_i, ref_pose.p));

    return vec3_add(quat_rotate(q_rel, p_local), t_rel);
}

// ===========================================================================
// GPU kernel declaration — nvcc-only (cl.exe, compiling reference_cpu.cpp,
// has never heard of __global__ and must never see this).
// ===========================================================================
#ifdef __CUDACC__

// deskew_kernel — one thread per point (the pure-map mapping the derivation
// above promises). Full launch-configuration reasoning lives with the
// definition in kernels.cu.
//   t_points  : [n_points] device floats, firing time (s) per point.
//   xyz_local : [n_points*3] device floats, SKEWED local coordinates (m).
//   n_samples : trajectory sample count for THIS launch (kSparseSamples or
//               kDenseSamples — the caller selects the regime by choosing
//               which array it uploaded via set_trajectory(), below).
//   ref_pose  : the precomputed pose at t_ref (uniform across all threads —
//               a broadcast read, the middle of the spectrum between
//               09.01's __constant__ model and 07.09's divergent reads,
//               same note 08.01's u_nom comment makes for its own uniform
//               parameter).
//   xyz_out   : [n_points*3] device floats OUT, deskewed reference-frame coordinates (m).
__global__ void deskew_kernel(int n_points, const float* __restrict__ t_points,
                              const float* __restrict__ xyz_local, int n_samples,
                              Pose ref_pose, float* __restrict__ xyz_out);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// set_trajectory — upload one regime's trajectory samples to GPU
// __constant__ memory (kernels.cu owns the __constant__ symbol; see that
// file for why constant memory is the right home for this tiny, everyone-
// reads-the-same-bytes array). Must be called before launch_deskew for the
// SAME regime; main.cu calls this ONCE PER (cohort, regime) PAIR — mirrors
// 01.10's set_row_lut() "upload once per variant, launch many" shape,
// including the reason it re-uploads BETWEEN variants: this project's whole
// point is comparing what changes when the UPLOADED trajectory itself
// changes resolution.
//
//   host_traj — [n*kTrajStride] host floats, the layout documented at the
//               top of this file. n MUST be in [2, kMaxTrajSamples] (checked,
//               aborts loudly otherwise — a silent overflow would corrupt
//               whatever __constant__ memory follows it).
// ---------------------------------------------------------------------------
void set_trajectory(const float* host_traj, int n);

// ---------------------------------------------------------------------------
// launch_deskew — host wrapper owning the grid/block math + post-launch
// error check (CLAUDE.md §6.1 rule 7). Must be called AFTER set_trajectory
// for the desired regime; d_t_points/d_xyz_local/d_xyz_out are DEVICE
// pointers the caller allocated (main.cu).
// ---------------------------------------------------------------------------
void launch_deskew(int n_points, const float* d_t_points, const float* d_xyz_local,
                   int n_samples, Pose ref_pose, float* d_xyz_out);

// ---------------------------------------------------------------------------
// deskew_cpu — the CPU correctness oracle (defined in reference_cpu.cpp).
// Same math (via the shared HD deskew_one_point above), independent
// surrounding loop, plain single-threaded C++. traj/t_points/xyz_local/
// xyz_out are all HOST pointers; traj is NOT read from __constant__ memory
// (no constant memory on a CPU — the same difference 09.01's kernels.cuh
// comment names for its own CPU twin).
// ---------------------------------------------------------------------------
void deskew_cpu(int n_points, const float* t_points, const float* xyz_local,
                const float* traj, int n_samples, Pose ref_pose, float* xyz_out);

#endif // PROJECT_KERNELS_CUH
