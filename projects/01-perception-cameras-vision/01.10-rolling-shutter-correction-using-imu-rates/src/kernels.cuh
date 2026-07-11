// ===========================================================================
// kernels.cuh — interface for project 01.10
//               Rolling-shutter correction using IMU rates: gyro-aided,
//               pure-rotation row-homography un-warping of a CMOS rolling-
//               shutter (RS) frame back to what a global-shutter (GS) camera
//               would have captured at the frame's reference instant.
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (orchestration + gyro integration + gates),
// kernels.cu (the one GPU kernel), and reference_cpu.cpp (its CPU twin).
// Every camera constant, timing constant, and data layout all three must
// agree on lives HERE, once (CLAUDE.md paragraph 12). ../scripts/make_synthetic.py
// mirrors the geometry/timing constants (never the gyro-noise ones — those
// are generator-only, see that script) under "MUST MATCH kernels.cuh"
// comments; a mismatch there is a bug in the synthetic data, not in this file.
//
// The physical picture in one paragraph (THEORY.md derives every step)
// ----------------------------------------------------------------------
// A CMOS rolling-shutter sensor does not expose all rows at once: row v
// starts its exposure at t(v) = kFrameT0S + v*kLineTimeS, one row-time later
// than row v-1 (THEORY.md "The problem" explains WHY, from the ADC-sharing
// architecture of a real sensor). If the camera rotates while the frame is
// being read out, each row sees the world through a SLIGHTLY different
// orientation — a straight edge in the world comes out sheared, wobbled, or
// "jello"-wobbled in the captured image. This project undoes exactly that,
// using ONLY body-frame angular-rate (gyro) samples — no translation is
// modeled (PURE ROTATION only; see "Limitations & honesty" in README.md for
// when that assumption breaks down: near-field parallax).
//
// CAMERA MODEL — the CAMERA-OPTICAL frame exception (CLAUDE.md paragraph 3.2
// permits this, same exception 01.01 states at this exact API boundary):
// z-forward (down the optical axis), x-right, y-down (row 0 = image top).
// Pixel (x, y) has normalized ray direction ((x-cx)/fx, (y-cy)/fy, 1).
//
//   K = (fx, fy, cx, cy)   — pixels; ONE shared intrinsic matrix, no lens
//                            distortion modeled (out of scope; see 01.01 for
//                            that lesson) — this project is about TIME, not
//                            optics.
//
// ORIENTATION STATE — a quaternion trajectory q_world_cam(t), REPO ORDER
// (w, x, y, z), UNIT-NORM, "child (camera) expressed in parent (world)"
// (T_parent_child convention, docs/SYSTEM_DESIGN.md paragraph 3.3/3.4; same
// order 09.01's forward-kinematics model uses — cite that project for the
// convention, this project for how a CONTINUOUS trajectory of it is built
// from angular-RATE samples rather than joint angles). The gyro measures
// BODY-frame angular velocity omega(t) = (wx, wy, wz), rad/s, in the
// camera's OWN instantaneous axes (a real MEMS gyro's native output frame);
// the standard body-rate quaternion kinematic equation is
//
//     dq/dt = 0.5 * q (x) [0, omega]                (x) = quaternion product
//
// integrated here via the EXPONENTIAL MAP per sub-step (exact for a
// piecewise-constant omega over the sub-step — see quat_integrate_step()
// below and THEORY.md "The math" for the small-angle-vs-exact comparison).
//
// THE ROW HOMOGRAPHY — pure-rotation, K*R*K^-1 form (derived in full in
// THEORY.md "The math"; this is the SAME construction 01.01's
// compute_source_pixel() uses for its rectifying rotation, generalized from
// one FIXED rotation to one rotation PER ROW). Given the GS reference
// instant t_ref (this project's choice: the frame's MIDPOINT,
// kFrameTRefS = kFrameT0S + 0.5*(kImgH-1)*kLineTimeS — exactly the exposure
// time of the middle row, so it lines up with kCy = (kImgH-1)/2 by
// construction) and reference orientation q_ref = q_world_cam(t_ref):
// for OUTPUT (reference/GS) pixel (xo, yo), the pixel it corresponds to in
// the RAW rolling-shutter frame, IF that raw pixel's row is exposed at row
// v, is found by rotating the reference ray into that row's camera frame:
//
//     q_rel(v) = conj(q_world_cam(t(v))) (x) q_ref      (repo quaternion order)
//     ray_row  = R(q_rel(v)) * K^-1 [xo, yo, 1]^T
//     [xs, ys] = K * ray_row / ray_row.z
//
// But v itself is UNKNOWN until we know which row (xs, ys) lands in — the
// classic chicken-and-egg of rolling-shutter geometry. This project solves
// it with a 3-ITERATION FIXED-POINT SEARCH (README "The algorithm in
// brief", THEORY.md "The algorithm" derives the contraction argument):
// guess v = yo, look up q_rel(v), compute (xs, ys), let the new guess be
// v = ys, repeat. main.cu's convergence gate MEASURES how much the guess
// still moves between the 2nd and 3rd iteration and gates it small.
//
// LUT-vs-recompute (the GPU-mapping lesson, 01.01 cited above): re-running
// the SEQUENTIAL gyro integration once per PIXEL would be both wrong
// (redundant, W*H times the necessary work) and pointless (every pixel
// would recompute the IDENTICAL trajectory) — so the trajectory is
// integrated ONCE, on the HOST (main.cu; small and inherently sequential —
// see that file's header for why host, not GPU), collapsed into one
// quaternion PER OUTPUT ROW (kImgH of them — tiny), and every pixel-thread
// then does O(1) LUT lookups. The one GPU kernel in this project
// (rs_correct_kernel, kernels.cu) is a pure MAP over the kImgW*kImgH output
// pixels: every pixel's 3-iteration search + bilinear sample is completely
// independent of every other pixel's — see kernels.cu for the launch
// reasoning and the constant-memory broadcast argument.
//
// DATA LAYOUTS
// ------------
//   IMAGE   (rs_input / ground_truth_gs / corrected), kImgW x kImgH:
//     uint8_t[y*kImgW + x] — one grayscale byte per pixel (this project has
//     no color content to teach; geometry is the whole lesson).
//   ROW LUT: Quat[kImgH] — q_rel(v) for output row v = 0..kImgH-1, built by
//     main.cu (host) and uploaded once per gyro variant via set_row_lut().
//   GyroSample (main.cu-local, not declared here): one CSV row —
//     t_s (s, frame-relative), wx/wy/wz (rad/s, camera/body frame).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>     // sqrtf/sinf/cosf/floorf/fabsf — used by the HD helpers below

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. Same trick as
// 01.01/09.01/18.01: the quaternion and camera-model primitives below are
// textually IDENTICAL on both sides of the host/device boundary
// (kernels.cu's kernel and reference_cpu.cpp's twin both call them
// unchanged) — see reference_cpu.cpp's file header for the twin-
// independence ruling this satisfies (these are "camera model", the
// permitted shared exception; the per-pixel LOOP STRUCTURE around them is
// typed twice, independently).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Geometry / timing constants — MUST MATCH ../scripts/make_synthetic.py's
// "MUST MATCH kernels.cuh" block. kImgW/kImgH mirror 01.01's 384x288 (a
// familiar, fast-to-build-verify resolution for this repo).
// ---------------------------------------------------------------------------
constexpr int kImgW = 384;                   // sensor/output width, px
constexpr int kImgH = 288;                   // sensor/output height, px (= number of readout rows)

constexpr float kFx = 380.0f;                // focal length, px
constexpr float kFy = 380.0f;                // focal length, px
constexpr float kCx = (kImgW - 1) * 0.5f;    // principal point x, px = 191.5
constexpr float kCy = (kImgH - 1) * 0.5f;    // principal point y, px = 143.5 (= reference ROW, see below)

// Sensor timing (THEORY.md "The problem" derives WHY a real CMOS sensor has
// a nonzero readout time). 25 ms is a realistic full-frame rolling-shutter
// readout for a consumer/embedded CMOS sensor at moderate resolution
// (compare: a global-shutter or short-readout sensor would be under 5 ms).
constexpr float kReadoutTimeS = 0.025f;                              // 25 ms, full-frame readout
constexpr float kLineTimeS = kReadoutTimeS / static_cast<float>(kImgH);  // ~86.8 us/row
constexpr float kFrameT0S = 0.0f;                                    // row 0's exposure start, frame-relative
// GS reference instant: the MIDDLE row's exposure time (README "The
// algorithm in brief"). Deliberately computed from the ROW index (not as
// kFrameT0S + 0.5*kReadoutTimeS) so it lines up EXACTLY with kCy = (kImgH-1)/2
// — the reference view is, by construction, "what row kCy's own camera
// orientation saw" (main.cu's design note expands on this).
constexpr float kFrameTRefS = kFrameT0S + 0.5f * static_cast<float>(kImgH - 1) * kLineTimeS;

// Gyro sampling (the IMU's own rate — main.cu reads the CSV at this
// nominal spacing; the file itself carries the exact timestamps, this
// constant is documentation + the fine-integration sub-step budget).
constexpr float kGyroRateHz = 200.0f;
constexpr float kGyroDtS = 1.0f / kGyroRateHz;               // 5 ms nominal spacing

// Fixed-point row-time search — THEORY.md "The algorithm" derives why 3
// iterations converges far below one pixel for this project's rotation
// rates; main.cu's convergence gate MEASURES the actual 2nd-vs-3rd-iteration
// delta rather than trusting this number blindly.
constexpr int kFixedPointIters = 3;

// Sub-steps of quaternion integration PER gyro interval, used by main.cu's
// host-side integrate_gyro_to_fine_trajectory(). 32 keeps each fine step's
// duration (kGyroDtS/32 ~ 156 us) far shorter than the fastest jitter
// component in the synthetic profile, so the piecewise-linear-omega
// assumption between gyro samples (THEORY.md "Numerical considerations")
// stays accurate; the whole window is under 1,000 sequential steps, trivial
// on a CPU (justifying "host, not GPU" — see main.cu's header).
constexpr int kIntegrationSubsteps = 32;

// Restoration-gate masking (main.cu): pixels within this many px of the
// scene's known vertical marker line (column kCx, see
// ../scripts/make_synthetic.py) are excluded from the PRIMARY restoration
// score and reported separately — same "smooth mask" idea as 01.01's
// color-fidelity gate, for the same reason: even a PERFECT correction shows
// larger error immediately next to a hard edge (sub-pixel/bilinear blur),
// so scoring that region would penalize the pipeline for physics it cannot
// avoid. Also excludes a thin border strip (bilinear sampling near the
// frame edge is more sensitive to the fixed-point search's convergence).
constexpr int kRestorationMaskMarginPx = 8;   // half-width around column kCx, px
constexpr int kBorderMarginPx = 4;            // strip excluded at every image edge, px

// ---------------------------------------------------------------------------
// Quat — a unit quaternion, REPO ORDER (w, x, y, z) (docs/SYSTEM_DESIGN.md
// paragraph 3.4; same order 09.01's forward-kinematics model uses — cite
// that project's kernels.cuh for the convention). A plain struct (not any
// vendor quaternion type) keeps every operation below visible, no black box
// (CLAUDE.md paragraph 1).
// ---------------------------------------------------------------------------
struct Quat {
    float w, x, y, z;
};

// ---------------------------------------------------------------------------
// quat_normalize — renormalize a quaternion to unit length.
//
// WHY this exists as its own step, called after every multiply/integrate:
// chained quaternion multiplications accumulate floating-point rounding
// error and drift off the unit sphere (THEORY.md "Numerical considerations"
// — the same "quaternion normalization drift" hazard CLAUDE.md paragraph 12
// names explicitly). Renormalizing after every step is this project's fix;
// the alternative (renormalize occasionally) trades accuracy for a few
// saved sqrt() calls that are not worth it at this project's data size.
// ---------------------------------------------------------------------------
HD inline Quat quat_normalize(Quat q)
{
    const float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    // Guard the (never-expected-but-cheap-to-guard) zero-quaternion case:
    // return the identity rather than divide by zero.
    if (n < 1e-20f) { Quat id{ 1.0f, 0.0f, 0.0f, 0.0f }; return id; }
    const float inv_n = 1.0f / n;
    Quat r{ q.w * inv_n, q.x * inv_n, q.y * inv_n, q.z * inv_n };
    return r;
}

// ---------------------------------------------------------------------------
// quat_mul — Hamilton quaternion product a (x) b, REPO ORDER (w,x,y,z).
//
// Standard bilinear form; written out term-by-term (no black-box vendor
// quaternion class, CLAUDE.md paragraph 1) so a reader can check it against
// any quaternion-algebra reference by eye.
// ---------------------------------------------------------------------------
HD inline Quat quat_mul(Quat a, Quat b)
{
    Quat r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return r;
}

// ---------------------------------------------------------------------------
// quat_conj — the conjugate (w,-x,-y,-z), which for a UNIT quaternion is
// also its inverse: R(conj(q)) = R(q)^T (same rotation, opposite direction —
// exactly the forward/inverse relationship 01.01's R_rect_raw / R_rect_raw^T
// pair uses, generalized here from a single fixed rotation to every
// q_world_cam(t) in the trajectory).
// ---------------------------------------------------------------------------
HD inline Quat quat_conj(Quat q)
{
    Quat r{ q.w, -q.x, -q.y, -q.z };
    return r;
}

// ---------------------------------------------------------------------------
// quat_integrate_step — advance a body-frame orientation quaternion by one
// EXPONENTIAL-MAP step under a CONSTANT angular velocity omega=(wx,wy,wz)
// (rad/s, body/camera frame) held for duration dt (s).
//
// Why exponential-map instead of first-order Euler (qdot = 0.5*q(x)[0,omega],
// q_next = normalize(q + qdot*dt))? The exponential map is EXACT for a
// truly constant omega over the step (it integrates the ODE in closed form
// via the rotation's axis-angle representation); first-order Euler is only
// a linear approximation of the same closed form and accumulates a larger
// per-step error for the same dt (THEORY.md "The math" derives both and
// compares their error order — O(dt^3) here vs O(dt^2) for Euler, per
// step). Since main.cu calls this kIntegrationSubsteps times per gyro
// interval specifically to keep omega "nearly constant" per sub-step, the
// exponential map's extra correctness essentially comes for free (one
// extra sin/cos, negligible next to the sequential dependency chain's own
// cost).
//
// Parameters:
//   q            — current unit quaternion, q_world_cam(t)
//   wx, wy, wz   — body-frame angular velocity, rad/s (assumed CONSTANT
//                  over this call's dt — main.cu supplies dt small enough,
//                  and the caller's linear interpolation between gyro
//                  samples supplies the value, that this holds well)
//   dt           — sub-step duration, s (> 0)
// Returns: q_world_cam(t + dt), renormalized.
// ---------------------------------------------------------------------------
HD inline Quat quat_integrate_step(Quat q, float wx, float wy, float wz, float dt)
{
    const float wmag = sqrtf(wx * wx + wy * wy + wz * wz);   // |omega|, rad/s
    Quat dq;                                                  // the BODY-FRAME delta rotation over dt
    if (wmag < 1e-9f) {
        // Degenerate (near-zero) angular rate: the exact exponential map's
        // sin(theta)/wmag term is a 0/0 in the limit; the well-known small-
        // angle fallback (dq ~= [1, 0.5*omega*dt]) avoids the division and
        // is accurate to O(dt^3) here anyway since wmag*dt is tiny.
        dq = Quat{ 1.0f, 0.5f * wx * dt, 0.5f * wy * dt, 0.5f * wz * dt };
        dq = quat_normalize(dq);
    } else {
        const float half_angle = 0.5f * wmag * dt;     // half the rotation angle swept this step
        const float s = sinf(half_angle) / wmag;        // sin(half_angle)/|omega|; axis*sin = omega*s
        dq.w = cosf(half_angle);
        dq.x = wx * s;
        dq.y = wy * s;
        dq.z = wz * s;
    }
    // Right-multiply: dq is expressed in the BODY frame AT TIME t (the
    // frame q itself defines), matching the dq/dt = 0.5*q(x)[0,omega]
    // kinematic equation's convention (file header). Left-multiplying
    // would instead treat omega as a WORLD-frame rate — a real, easy-to-
    // make sign/order bug this project deliberately avoids by never
    // writing it the other way.
    return quat_normalize(quat_mul(q, dq));
}

// ---------------------------------------------------------------------------
// quat_to_mat3 — unit quaternion -> 3x3 rotation matrix, ROW-MAJOR R[9]
// such that for a column vector v, v' = R*v (the standard "active rotation"
// convention; R[0..2] is row 0, etc.). Textbook formula, written out so
// nothing here is a black box (CLAUDE.md paragraph 1).
// ---------------------------------------------------------------------------
HD inline void quat_to_mat3(Quat q, float R[9])
{
    const float w = q.w, x = q.x, y = q.y, z = q.z;
    R[0] = 1.0f - 2.0f * (y * y + z * z);  R[1] = 2.0f * (x * y - z * w);        R[2] = 2.0f * (x * z + y * w);
    R[3] = 2.0f * (x * y + z * w);         R[4] = 1.0f - 2.0f * (x * x + z * z); R[5] = 2.0f * (y * z - x * w);
    R[6] = 2.0f * (x * z - y * w);         R[7] = 2.0f * (y * z + x * w);        R[8] = 1.0f - 2.0f * (x * x + y * y);
}

// ---------------------------------------------------------------------------
// lerp_row_quat — fractional-row lookup into a per-row quaternion LUT
// (either the device __constant__ array, passed by pointer, or the host
// row_lut vector's .data() — this function does not care which).
//
// WHY linear interpolation is good enough here: the fixed-point search
// (kernels.cu's kernel / reference_cpu.cpp's twin) evaluates this at a
// fractional row v_guess, but the LUT only has one quaternion PER INTEGER
// row. Row-to-row, q_rel changes by only kLineTimeS worth of rotation
// (~87 microseconds of the profile's motion — a tiny angle, THEORY.md
// "Numerical considerations" bounds it), so linearly blending adjacent
// rows' (w,x,y,z) components and renormalizing is accurate to well beyond
// this project's other error sources (a cheap LERP, not a full SLERP —
// the two are numerically indistinguishable for angles this small).
//
// Parameters: lut — [H] quaternions, row_lut[v] = q_rel for INTEGER row v.
//             H   — row count (kImgH in every caller).
//             v   — fractional row to sample; CLAMPED into [0, H-1] so the
//                   fixed-point search cannot walk the lookup out of
//                   bounds even mid-iteration (main.cu's convergence gate
//                   reports how far v actually moves, so a clamp here is
//                   never silently hiding a runaway iteration).
// ---------------------------------------------------------------------------
HD inline Quat lerp_row_quat(const Quat* lut, int H, float v)
{
    float vc = v;
    if (vc < 0.0f) vc = 0.0f;
    if (vc > static_cast<float>(H - 1)) vc = static_cast<float>(H - 1);
    const int v0 = static_cast<int>(floorf(vc));
    const int v1 = (v0 + 1 < H) ? (v0 + 1) : v0;   // clamp at the last row
    const float t = vc - static_cast<float>(v0);
    const Quat a = lut[v0];
    const Quat b = lut[v1];
    Quat r{ a.w + (b.w - a.w) * t, a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t };
    return quat_normalize(r);
}

// ---------------------------------------------------------------------------
// apply_row_rotation — the pure-rotation row homography, decomposed into
// its three geometric steps (file header derives the formula; this is
// 01.01's compute_source_pixel() pattern, generalized from ONE fixed
// rectifying rotation to a PER-ROW relative rotation R):
//   1. output pixel -> normalized ray via K^-1.
//   2. rotate by R (row-major, v' = R*v — see quat_to_mat3 above).
//   3. project the rotated ray back to pixel coordinates via K.
//
// Parameters:
//   R      — [9] row-major rotation matrix, q_rel(v) converted by the
//            caller (quat_to_mat3) — kept as a raw matrix here (not a
//            quaternion) so this hot inner-loop function is pure
//            multiply-adds, no trig, no quaternion algebra.
//   xo, yo — OUTPUT (reference/GS) pixel coordinates.
//   xs, ys — OUT: the corresponding RAW rolling-shutter pixel coordinates
//            (may land outside [0,kImgW)x[0,kImgH) near the image border —
//            callers check via bilinear_sample_gray's validity flag).
// ---------------------------------------------------------------------------
HD inline void apply_row_rotation(const float R[9], float xo, float yo, float& xs, float& ys)
{
    const float rx = (xo - kCx) / kFx;   // step 1: output pixel -> normalized ray, x
    const float ry = (yo - kCy) / kFy;   // step 1: output pixel -> normalized ray, y  (z = 1, implicit)

    // step 2: rotate the ray by R (row-major 3x3 * column vector (rx,ry,1)).
    const float sx = R[0] * rx + R[1] * ry + R[2];
    const float sy = R[3] * rx + R[4] * ry + R[5];
    const float sz = R[6] * rx + R[7] * ry + R[8];

    // step 3: perspective-divide and re-apply K. sz stays close to 1 for
    // this project's small jitter angles (a few degrees at most — see
    // ../scripts/make_synthetic.py's rotation profile), so this division
    // never approaches the near-zero-sz singularity a wide-angle rotation
    // could hit (THEORY.md "Numerical considerations").
    const float inv_sz = 1.0f / sz;
    xs = kFx * sx * inv_sz + kCx;
    ys = kFy * sy * inv_sz + kCy;
}

// ---------------------------------------------------------------------------
// bilinear_sample_gray — sample an 8-bit grayscale image at a fractional
// (x, y), returning the interpolated intensity AND whether the sample
// point actually lands inside the image (no clamp-to-edge here, unlike
// 01.01's remap — CLAMPING would silently manufacture a plausible-looking
// but PHYSICALLY WRONG pixel for a row the rolling-shutter sensor never
// captured at that column; this project instead reports INVALID and lets
// main.cu's coverage/gates account for it honestly).
//
// Parameters: img — [W*H] row-major grayscale. x, y — sample point.
//             valid — OUT: 1 if (x,y) in [0,W-1]x[0,H-1], else 0 (and the
//                     returned intensity is a meaningless 0.0f).
// ---------------------------------------------------------------------------
HD inline float bilinear_sample_gray(const unsigned char* img, int W, int H,
                                     float x, float y, int& valid)
{
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(W - 1) || y > static_cast<float>(H - 1)) {
        valid = 0;
        return 0.0f;
    }
    valid = 1;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = (x0 + 1 < W) ? (x0 + 1) : x0;
    const int y1 = (y0 + 1 < H) ? (y0 + 1) : y0;
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);

    const float v00 = static_cast<float>(img[static_cast<size_t>(y0) * W + x0]);
    const float v10 = static_cast<float>(img[static_cast<size_t>(y0) * W + x1]);
    const float v01 = static_cast<float>(img[static_cast<size_t>(y1) * W + x0]);
    const float v11 = static_cast<float>(img[static_cast<size_t>(y1) * W + x1]);

    const float top = v00 + (v10 - v00) * fx;
    const float bot = v01 + (v11 - v01) * fx;
    return top + (bot - top) * fy;
}

// ---------------------------------------------------------------------------
// GPU launch wrappers (kernels.cu). main.cu never touches <<<...>>> syntax
// directly (same discipline as every other project in this repo).
// ---------------------------------------------------------------------------

// set_row_lut — upload one gyro variant's per-row relative-quaternion LUT
// to GPU __constant__ memory (kernels.cu owns the __constant__ symbol; see
// that file for why __constant__ and not a plain device buffer). Must be
// called before launch_rs_correct for the SAME gyro variant; main.cu calls
// this once per variant (clean, then degraded) before re-running the
// kernel — mirrors 09.01's set_robot_model() "upload once, launch many"
// shape, except here the whole point is that we DO re-upload between runs.
//
//   host_lut — [H] host quaternions, H MUST equal kImgH (checked, aborts
//              loudly otherwise — a silent H mismatch would corrupt every
//              row lookup past the shorter length).
void set_row_lut(const Quat* host_lut, int H);

// launch_rs_correct — the ONE GPU kernel in this project: for every OUTPUT
// (reference/GS) pixel, run the kFixedPointIters-iteration row-time search
// (file header) against the currently-uploaded row LUT, then bilinearly
// sample the raw rolling-shutter frame at the resolved source pixel.
//
//   d_rs_frame    — DEVICE [kImgW*kImgH] uint8, the captured RS frame.
//   d_corrected   — DEVICE [kImgW*kImgH] uint8 OUT, the reconstructed GS-
//                   reference-view image (0 where invalid — see d_valid).
//   d_valid_mask  — DEVICE [kImgW*kImgH] uint8 OUT, 1 = source pixel landed
//                   inside the RS frame, 0 = it did not (near-border rows
//                   most often — see README "Expected output").
//   d_iter_delta  — DEVICE [kImgW*kImgH] float OUT, |row_guess after the
//                   FINAL iteration - row_guess after the PREVIOUS one| —
//                   main.cu's convergence gate takes the max over this.
//   W, H          — must equal kImgW, kImgH (every caller in this project
//                   passes exactly those constants; kept as parameters,
//                   not hardcoded, purely so the kernel signature reads
//                   like ordinary image-processing code).
void launch_rs_correct(const unsigned char* d_rs_frame, unsigned char* d_corrected,
                       unsigned char* d_valid_mask, float* d_iter_delta, int W, int H);

// ---------------------------------------------------------------------------
// CPU reference (reference_cpu.cpp) — INDEPENDENT reimplementation of
// launch_rs_correct's kernel (per the twin-independence ruling in that
// file's header): same math (via the shared HD "camera model" primitives
// above), the per-pixel loop nest typed a second time by hand.
//
//   row_lut — [H] host quaternions (the SAME array main.cu built and
//             passed to set_row_lut for this gyro variant — not re-derived
//             here; gyro integration has no GPU counterpart to twin against,
//             see main.cu's header for why).
// ---------------------------------------------------------------------------
void rs_correct_cpu(const Quat* row_lut, int H,
                    const unsigned char* rs_frame, int W,
                    unsigned char* corrected, unsigned char* valid_mask, float* iter_delta);

#endif // PROJECT_KERNELS_CUH
