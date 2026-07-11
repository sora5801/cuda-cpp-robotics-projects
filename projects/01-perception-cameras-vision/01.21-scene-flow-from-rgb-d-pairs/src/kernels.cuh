// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 01.21
//               (Scene flow from RGB-D pairs: 2-D flow lifted to 3-D by
//               depth, a dominant rigid ego-motion robustly fitted and
//               removed, and the residual segmenting an independently
//               moving object)
//
// Role in the project
// -------------------
// SINGLE-SOURCED CONTRACT between kernels.cu (GPU, nvcc), reference_cpu.cpp
// (independent CPU oracle, cl.exe) and main.cu (orchestration, nvcc). Per
// the repo's twin-independence ruling (docs/PROJECT_TEMPLATE/src/
// reference_cpu.cpp's header, reproduced by every project's reference_cpu.cpp
// file comment): data LAYOUT (image geometry, camera intrinsics, the point
// layout, the 16-scalar covariance-reduction record, tolerance-adjacent
// constants) is single-sourced HERE; the ALGORITHMIC CORE of every per-pixel
// stage (gradients, LK solve, back-projection, residuals, the reduction's
// per-point contribution, thresholding, morphology) is written TWICE — once
// in kernels.cu, independently again in reference_cpu.cpp. The ONE exception
// is build_rigid_from_covariance16() below (Horn's closed-form quaternion
// solve): it is a small, textbook, NON-approximating linear-algebra routine
// that both paths must agree on bit-for-bit to even ask "did the reduction
// upstream agree" — duplicating it would be pure transcription, exactly the
// case CLAUDE.md's twin-independence ruling exempts (02.06's cholesky6_solve
// and 01.17's cholesky6_solve are the same exemption for their own normal-
// equation solves, cited). Because this routine is SHARED, it earns its own
// INDEPENDENT verification gate that never routes through it being "the same
// on both sides": the ego_motion gate compares the RECOVERED transform
// against the scene's known, closed-form ground truth (R_gt/t_gt below),
// not against a second implementation of the solve.
//
// THE PIPELINE (five stages — README "The algorithm in brief" names each)
// -------------------------------------------------------------------------
//   1. 2-D OPTICAL FLOW — dense 2-level pyramidal Lucas-Kanade (a compact
//      re-implementation of project 01.03's Milestone 1; cited throughout,
//      see run_pyramidal_lk_gpu). TWO levels (not 01.03's three) are enough
//      at THIS project's scene scale: the largest frame-to-frame
//      displacement in the committed sample is small (THEORY.md "The
//      algorithm" reports the measured maximum), so one halving already
//      brings it inside the 5x5 window's capture range at the coarse level.
//   2. 3-D LIFTING — back-project pixel x using frame0's depth D0(x) -> P1;
//      back-project the flow-shifted location x+u using frame1's depth
//      D1(x+u), BILINEAR-SAMPLED (sub-pixel flow needs sub-pixel depth) with
//      a DEPTH-CONSISTENCY GUARD: sampling straddles a depth discontinuity
//      fabricates a physically meaningless blended depth (a real edge, not
//      noise), so the guard rejects any sample whose 4 bilinear taps
//      disagree by more than kDepthEdgeGuardM (THEORY.md derives the bound
//      from the noise model below). F = P2 - P1 is the RAW ("uncompensated")
//      3-D scene flow: P1 lives in camera0's frame, P2 in camera1's frame —
//      see the ground-truth block below for exactly what this mixture means
//      physically and why the next stage can still make sense of it.
//   3. EGO-MOTION — robustly fit ONE rigid transform T=(R,t) to the whole
//      (P1 -> P2) field via iteratively reweighted Horn/Kabsch alignment
//      (kIrlsIterations rounds; Tukey-biweight downweighting from a
//      median-absolute-residual robust scale — THEORY.md justifies IRLS
//      over a RANSAC-lite alternative). Iteration 0 (uniform weights) is
//      kept as the NAIVE baseline the ego_motion gate reports alongside the
//      robust result — the whole point of robustness only shows up as a
//      COMPARISON.
//   4. RESIDUAL SEGMENTATION — r = |T(P1) - P2| per pixel; threshold this
//      against a residual-noise-DERIVED bound (kSegThresholdKSigma * the
//      depth-noise-propagated residual sigma, computed at RUNTIME from the
//      loaded scene's own measured depth — see main.cu), then a 3x3
//      morphological OPEN (erode-then-dilate, the same idiom project 30.01
//      uses for blob cleanup, cited) removes salt-and-pepper false
//      positives from the noise floor, and THEN a CONNECTED-COMPONENT SIZE
//      FILTER (iterative label propagation, 01.06/30.01's pattern, cited;
//      kMinComponentSizePx below) removes components no larger than the
//      bare noise floor the opening operator itself can produce.
//   5. OBJECT MOTION — a ROBUST (IRLS+Tukey), FIXED-ROTATION offset
//      estimate restricted to the segmented mask's correspondences: R is
//      held at the already-accurate R_robust (the scene's object motion is
//      translation-only, see c_gt below) and only the translation offset is
//      robustly averaged — see main.cu's Milestone-5 comment for why a free
//      6-DOF Horn fit on this small, spatially narrow point set was
//      ill-conditioned enough to recover a near-opposite-direction answer,
//      and how fixing the rotation repairs it. Reported [info] against the
//      scene's known object-motion offset (c_gt below); not yet gated (the
//      recovered magnitude still under-shoots truth — see main.cu).
//
// CAMERA MODEL — pinhole, PIXEL-CENTER convention: for INTEGER pixel (px,py)
// or CONTINUOUS flow-shifted position (px,py), the back-projection at depth
// d is P = ( (px+0.5-cx)/fx * d, (py+0.5-cy)/fy * d, d ) — camera OPTICAL
// frame (z-forward depth axis, x-right, y-down; the same convention 01.18
// uses, cited). This MUST match scripts/make_synthetic.py's ray formula
// exactly (camera_ray_cam_frame()) or every recovered 3-D point carries a
// systematic sub-pixel bias — see that script's module docstring.
//
// GROUND TRUTH (manually synchronized with scripts/make_synthetic.py —
// change one, change both; that script PRINTS these exact numbers so they
// can be copy-pasted here rather than hand-derived a second time)
// -------------------------------------------------------------------------
// The synthetic pair is rendered from camera0 (identity orientation) and
// camera1 (rotated by R_EGO, translated by T_EGO — the camera's OWN known
// motion, WORLD/BODY frame x-forward/y-left/z-up), with one box additionally
// translating by T_OBJ (world frame, rotation-free). scripts/make_synthetic.
// py's module docstring derives, from first principles, that
//   T_gt = (R_gt, t_gt) = (M R_EGO^T M^T,  M (-R_EGO^T T_EGO))
// — CONJUGATED by M, the fixed body-to-camera-OPTICAL permutation
// (body_to_cam() in that script; same convention as 01.18's kTCameraLidar) —
// is EXACTLY the transform stage 3 above is trying to recover from the
// (P1 -> P2) field for STATIC points, because P1/P2 live in the OPTICAL
// frame while R_EGO/T_EGO are naturally expressed in BODY frame; skipping
// the M-conjugation silently recovers a rotation about the wrong axis (this
// was root-caused during this project's own build — see that script's
// docstring for the full story and the independent-Horn-fit check that
// caught it). Recovering camera ego-motion and recovering T_gt are the same
// problem up to this one matrix inverse (plus the M-conjugation). For the
// MOVING object's points, P2 = T_gt(P1) + c_gt with
//   c_gt = M (R_EGO^T T_OBJ)
// a CONSTANT offset — the reason residual segmentation works at all (every
// static pixel's post-T_gt residual is ~sensor noise; every object pixel's
// is ~c_gt, a fixed nonzero vector) and exactly what the object_motion gate
// compares its own recovered (T_obj.t - T_ego.t) against.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint8_t/uint32_t — images, masks, validity flags
#include <cmath>     // sqrtf/fabsf — used by host-side shared helpers below

// ===========================================================================
// Image / camera constants — SINGLE-SOURCED. scripts/make_synthetic.py
// renders every committed frame at exactly this resolution and intrinsics;
// main.cu asserts the loaded files' byte counts match before doing anything
// else (fail loud, never silently truncate — repo convention).
//
// 128x96 (4:3) halves EXACTLY once (128->64, 96->48) — enough for this
// project's 2-level pyramid with no rounding-induced size drift between
// what downsample_area2x_kernel produces and what level 1 expects (the
// same exact-power-of-two argument 01.03's kernels.cuh makes, cited).
// ===========================================================================
constexpr int   kW = 128;                 // level-0 (full-res) image width, px
constexpr int   kH = 96;                  // level-0 (full-res) image height, px
constexpr int   kPixels = kW * kH;        // 12288 — every full-res per-pixel array's length

constexpr float kFx = 118.0f;             // px focal length x (~57.0 deg horizontal FOV)
constexpr float kFy = 116.0f;             // px focal length y (~45.2 deg vertical FOV)
constexpr float kCx = 64.0f;              // px principal point x (image-center)
constexpr float kCy = 48.0f;              // px principal point y

constexpr float kMaxDepthM = 16.0f;       // sensor max range, m (eye-safety/SNR-limited in a real device)
constexpr float kInvalidDepth = -1.0f;    // sentinel: "no return here" (sky) or "not yet computed"

// kMaxRayBoundFactor — max over the WHOLE image of |ray_dir(px,py)| where
// ray_dir = ((px+0.5-cx)/fx, (py+0.5-cy)/fy, 1) — i.e. how much longer the
// back-projection Jacobian dP/dd can be at the image CORNER than at the
// principal point (where it is exactly 1). Used by main.cu to turn a per-
// pixel DEPTH noise sigma into a conservative bound on 3-D POSITION noise
// (THEORY.md "Numerical considerations" derives dP/dd = ray_dir(px,py), a
// first-order argument valid because the noise is small relative to depth).
// Computed once by hand from the corner pixel (0,0) — the corner farthest
// from the principal point in NORMALIZED terms is whichever image edge is
// farther from cx/cy in pixel units divided by focal length; for this
// project's kCx=64 (image half-width 64, so BOTH edges are equidistant) and
// kCy=48 (both edges equidistant too), the bound is realized at every
// corner alike: sqrt((64/118)^2 + (48/116)^2 + 1) ~= 1.2104.
constexpr float kMaxRayBoundFactor = 1.2104f;

// ===========================================================================
// MILESTONE 1 constants — 2-level pyramidal Lucas-Kanade (compact
// re-implementation of 01.03's Milestone 1; cited throughout kernels.cu).
// ===========================================================================
constexpr int kNumLevels = 2;             // level 0 (128x96), level 1 (64x48) — see file header for why 2 suffices here
inline int level_w(int level) { return kW >> level; }
inline int level_h(int level) { return kH >> level; }

constexpr int kLkWindowRadius = 2;        // 5x5 structure-tensor / mismatch window (01.03's default, cited)
constexpr int kGradBorder = 1;            // Scharr's own 3x3 stencil border
constexpr int kLkBorder = kGradBorder + kLkWindowRadius;   // = 3 (01.03's two-tier border argument, cited)
constexpr int kLkIterationsPerLevel = 5;  // warp-resolve-update rounds per level (one more than 01.03's 3: fewer
                                          // levels here means each level must close a larger relative motion)
constexpr float kLkMaxStepPerIterPx = 4.0f;   // per-iteration increment clamp (01.03's safeguard against a
                                              // near-singular/early-iteration overshoot, cited)
constexpr float kLkDetEpsilon = 1.0f;         // structure-tensor degeneracy floor (01.03's convention, cited)

// kMinConfidenceForLift — the structure tensor's small eigenvalue (the
// aperture-problem confidence 01.03 introduces, cited) must clear this floor
// before a pixel's flow is trusted enough to lift to 3-D. Textured surfaces
// in this project's synthetic scene (see scripts/make_synthetic.py's
// value_noise3) commonly reach several hundred; this floor is set well
// below that so only genuinely flat/degenerate pixels (border, sky-adjacent
// speckle) are excluded — THEORY.md reports the measured confidence
// distribution and the fraction this floor actually rejects.
constexpr float kMinConfidenceForLift = 4.0f;

// ===========================================================================
// MILESTONE 2 constants — 3-D lifting and the depth-consistency guard.
// ===========================================================================

// Depth sensor noise model (mirrors scripts/make_synthetic.py's
// DEPTH_NOISE_A_M / DEPTH_NOISE_B exactly — manually synchronized, see file
// header): sigma_z(z) = kDepthNoiseAM + kDepthNoiseB * z^2, a disparity-
// quantization model (THEORY.md "Numerical considerations" derives it: a
// stereo/structured-light depth sensor's range resolution is limited by a
// roughly CONSTANT disparity quantization, and z = f*B/disparity makes
// dz/d(disparity) grow as z^2 — the textbook reason RGB-D noise is small
// near the sensor and grows quadratically with range).
constexpr float kDepthNoiseAM = 0.0015f;   // m — sensor floor (electronics/quantization)
constexpr float kDepthNoiseB  = 0.00015f;  // 1/m — range^2 coefficient (kept modest so the noise floor
                                           // stays well under the ~0.3 m object-motion signal even at
                                           // this scene's ~9 m max depth — see kSegThresholdKSigma)

// kDepthEdgeGuardM — the lift stage rejects a bilinear depth sample whose 4
// integer taps disagree by more than this. Sized WELL above plausible noise
// spread (4 taps, worst-case sensor sigma at this scene's max depth 16 m is
// kDepthNoiseAM + kDepthNoiseB*16^2 = 0.156 m -> a 4-tap spread on the order
// of a few sigma, ~0.3-0.4 m in the worst case at extreme range) yet WELL
// below this scene's real depth discontinuities (box-to-background jumps
// are metres, not centimetres) — THEORY.md "Numerical considerations"
// reports the measured guard-rejection fraction so this claim is checked,
// not just asserted. NOTE the guard is deliberately conservative toward
// FALSE REJECTION (missing a few valid near-noise-floor pixels) rather than
// false acceptance (fabricating a blended depth across a real edge) — an
// explicit, documented asymmetry.
constexpr float kDepthEdgeGuardM = 0.35f;

// ===========================================================================
// MILESTONE 3 constants — robust ego-motion fit (IRLS + Horn/Kabsch).
// ===========================================================================
constexpr int kCovarWidth = 16;    // 1 (sum_w) + 3 (sum_w*P1) + 3 (sum_w*P2) + 9 (sum_w*P1(a)*P2(b)) — see build_rigid_from_covariance16
constexpr int kThreadsReduce = 128;   // block size for the covariance reduction (02.06's kThreadsReduce default, cited)

constexpr int kIrlsIterations = 8;       // robust-fit rounds; iteration 0 (uniform weights) is the NAIVE baseline.
                                         // Reused as-is (no separate constant) by Milestone 5's object-motion
                                         // fixed-rotation IRLS loop (main.cu) — the same round count and the
                                         // same Tukey/MAD machinery, just a 3-DOF location estimate instead of
                                         // a 6-DOF rigid fit.
constexpr float kTukeyC = 4.685f;        // standard Tukey biweight tuning constant (95% Gaussian efficiency)
constexpr float kMadToSigma = 1.4826f;   // MAD -> Gaussian-sigma conversion constant (textbook value)
constexpr int kHornPowerIterations = 60; // shifted power-iteration rounds for the dominant eigenvector (see
                                         // build_rigid_from_covariance16 — generous overkill for a 4x4 matrix,
                                         // the same "textbook sweep count, quantified" spirit as 02.06's Jacobi sweeps)

// ===========================================================================
// MILESTONE 4 constants — residual segmentation.
// ===========================================================================
// kSegThresholdKSigma — the residual-magnitude threshold used to call a
// pixel "moving" is kSegThresholdKSigma robust-sigmas above THIS run's own
// MEASURED residual spread (the MAD-based robust scale main.cu's IRLS loop
// already computes every round, reused here — see main.cu's Milestone-4
// comment). THEORY.md "Numerical considerations" derives the theoretical
// depth-noise-propagated PREDICTION of that spread's order of magnitude
// (kDepthNoiseAM/kDepthNoiseB through kMaxRayBoundFactor) as the physical
// reasoning for why thresholding a residual makes sense and roughly what
// scale to expect — and this project's own build measured that the pure
// depth-noise prediction under-counts a real, non-negligible contributor
// (2-D flow position uncertainty propagated through back-projection, which
// the depth-only model does not include), so the OPERATIONAL threshold
// grounds itself in the measured spread directly, cross-checked against
// (not simply equal to) the theoretical prediction — main.cu's
// noise_derivation [info] line prints both, and THEORY.md documents the gap
// honestly instead of silently absorbing it into an unexplained fudge factor.
constexpr float kSegThresholdKSigma = 3.0f;

// ===========================================================================
// MILESTONE 4b constants — post-morphology CONNECTED-COMPONENT size filter.
//
// WHY this stage exists (added after this project's own build measured the
// morphological-open mask's precision honestly and found it wanting — see
// THEORY.md "Numerical considerations" for the full before/after numbers):
// the 3x3 morphological OPEN (Milestone 4) already removes true SALT-AND-
// PEPPER single-pixel speckle, but this scene's dominant false-positive
// source turned out to be a spatially-COHERENT blob — a disocclusion-
// boundary artifact roughly the size of the object itself, immediately
// adjacent to it — not scattered noise. A connected-component size floor
// cannot discriminate a coherent wrong-blob from a coherent right-blob by
// SIZE alone (both survive); what it CAN honestly do is remove the residue
// that is smaller than anything the opening operator itself can vouch for.
// This is the same iterative label-propagation CCL pattern 01.06's Stage 2
// and 30.01's Stage 4 use (cited; independently re-typed here per this
// project's own mask/label layout — CLAUDE.md's "deliberate, documented
// duplication" cross-project rule, not the intra-project twin-independence
// ruling below, which is about GPU-vs-CPU within THIS project).
// ---------------------------------------------------------------------------
constexpr int kLabelNone = -1;    // CCL sentinel: "not a foreground pixel", 01.06/30.01's convention (cited)

// kMaxCclSweeps — safety cap on the label-propagation convergence loop
// (main.cu owns the loop, 01.06's main.cu shape, cited). Every sweep can
// only ever DECREASE a label (atomicMin), so the loop provably terminates in
// at most the diameter (in 4-connected hops) of the largest component —
// bounded by max(kW,kH)=128 in the worst pathological case (one component
// snaking across the whole image). 256 = 2x that bound, the same "generous
// overkill, quantified" margin build_rigid_from_covariance16's
// kHornPowerIterations documents (cited); real components here converge in
// a handful of sweeps (THEORY.md reports the measured sweep count).
constexpr int kMaxCclSweeps = 256;

// kMinComponentSizePx — after the 3x3 morphological OPEN, a component made
// of exactly ONE erosion-surviving pixel dilates back into a solid 3x3=9-
// pixel block: the THEORETICAL MINIMUM non-empty component the opening
// operator can produce (see erode3x3_kernel/dilate3x3_kernel). Such a
// component carries no more evidence than the bare noise floor the opening
// was already built to catch — it is one isolated erosion survivor and
// nothing else. kMinComponentSizePx is set one dilated-block ABOVE that
// floor (>9, i.e. >=10 survives): a component must contain at least a
// SECOND independently-surviving erosion pixel (or a genuinely larger raw
// blob) to be kept. This bound is derived from the MORPHOLOGICAL OPERATOR'S
// OWN MATH — not fit to this scene's object size, which the pipeline does
// not know at runtime and must not peek at (main.cu only reads truth_mask
// for GRADING, never for a pipeline decision). THEORY.md "Numerical
// considerations" reports the measured, honest effect: real, but modest
// (precision improves; IoU is roughly flat) — because the dominant false-
// positive blob described above is LARGER than this floor, same as the
// true object's own largest fragment, so size alone cannot reject it. A
// documented, characterized limitation, not hidden (see also README
// "Limitations & honesty").
constexpr int kMinComponentSizePx = 10;

// ===========================================================================
// blocks_for — ceil(count/threads), the repo-wide idiom (02.06/08.01,
// cited). Declared here (not just in kernels.cu) because main.cu must
// compute the IDENTICAL block count to size the block-partial download
// buffer for launch_weighted_covariance_reduce.
// ---------------------------------------------------------------------------
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ---------------------------------------------------------------------------
// Rigid3 — a rigid-body transform, IDENTICAL shape to 01.18's/02.06's
// convention: P_dst = R * P_src + t, R stored ROW-MAJOR (R[r*3+c]). A pure
// data-layout struct — safe to include from both nvcc and cl.exe.
// ---------------------------------------------------------------------------
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation
    float t[3];   // translation, meters
};

// ===========================================================================
// clamp_f / clampi — tiny shared numeric helpers (01.03's convention, cited:
// plain enough that writing them twice would be pure transcription, so they
// are shared without needing an extra verification gate — CLAUDE.md's twin-
// independence ruling's "pure token-for-token transcription" exemption).
// ===========================================================================
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
CUDA_HOSTDEV inline float clamp_f(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
CUDA_HOSTDEV inline int   clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ===========================================================================
// apply_rigid — P_out = T.R * P_in + T.t. Used by both main.cu (residual
// gating, artifact generation, ground-truth comparison) and
// reference_cpu.cpp (its own independent residual twin ALSO calls this —
// permitted: this is a 6-FLOP matrix-vector product, pure transcription if
// duplicated, exactly CLAUDE.md's exemption; the residual KERNEL's own
// per-point orchestration is still written twice, only this inner multiply
// is shared).
// ===========================================================================
CUDA_HOSTDEV inline void apply_rigid(const Rigid3& T, const float p_in[3], float p_out[3])
{
    p_out[0] = T.R[0] * p_in[0] + T.R[1] * p_in[1] + T.R[2] * p_in[2] + T.t[0];
    p_out[1] = T.R[3] * p_in[0] + T.R[4] * p_in[1] + T.R[5] * p_in[2] + T.t[1];
    p_out[2] = T.R[6] * p_in[0] + T.R[7] * p_in[1] + T.R[8] * p_in[2] + T.t[2];
}

// ===========================================================================
// build_rigid_from_covariance16 — Horn's (1987) closed-form absolute-
// orientation solve, quaternion form, via a SHIFTED POWER ITERATION for the
// dominant eigenvector (SVD-free — the catalog's requested "small closed-
// form" method; cited from 33.01's batched small-matrix-linalg spirit and
// 02.06's Jacobi-eigensolve precedent, though this project's 4x4 uses power
// iteration rather than Jacobi sweeps — see "why power iteration" below).
//
// SHARED host routine (see this file's header for the twin-independence
// ruling this satisfies): used identically by main.cu (after the GPU
// reduction) and reference_cpu.cpp (after its own from-scratch CPU
// accumulation) — the INDEPENDENT gate that keeps this honest is main.cu's
// ego_motion check against the scene's known R_gt/t_gt, never a second
// implementation of this function.
//
// Input: c[16], the accumulated (possibly weighted) sums in this order:
//   c[0]      = sum_w
//   c[1..3]   = sum_w * P1  (x,y,z)
//   c[4..6]   = sum_w * P2  (x,y,z)
//   c[7..15]  = sum_w * P1(a)*P2(b), row-major a outer b inner:
//               (xx,xy,xz, yx,yy,yz, zx,zy,zz)
// (exactly what weighted_covariance_reduce_kernel/_cpu accumulate — see
// their headers). DOUBLE precision: the GPU path reduces in FP32 across a
// block-tree then main.cu sums the block partials in double (02.06's
// "reduce in float, finish in double" convention, cited); the CPU path
// accumulates directly in double throughout.
//
// Output: *out (only meaningful if this returns true). Returns false when
// sum_w is too small to trust (near-zero valid/weighted correspondences) —
// callers must check and fail loudly rather than silently use a garbage
// identity transform.
//
// The algorithm (THEORY.md "The math" derives every step):
//   1. Centroids mu1 = c[1..3]/sum_w, mu2 = c[4..6]/sum_w.
//   2. Cross-covariance S[a][b] = c[7+3a+b]/sum_w - mu1[a]*mu2[b] (the
//      "E[XY] - E[X]E[Y]" identity — lets the reduction accumulate RAW
//      products in ONE pass with no prior knowledge of the centroids,
//      exactly the packed-accumulator idea 01.17's 28-scalar reduction
//      teaches, cited).
//   3. Build Horn's 4x4 symmetric "key matrix" N from S (see kernels.cu's
//      definition for the exact entries) — its trace is IDENTICALLY ZERO
//      (THEORY.md proves this), so the target eigenvalue (associated with
//      the optimal rotation quaternion) is not always the one of LARGEST
//      MAGNITUDE, which is what plain power iteration converges to.
//   4. WHY POWER ITERATION (not Jacobi): a full Jacobi eigensolve computes
//      ALL 4 eigenvalues/eigenvectors of a 4x4 matrix for the price of
//      needing only the largest — power iteration is the cheaper, more
//      transparent tool for "just the dominant eigenvector", PROVIDED the
//      target is actually dominant. Gershgorin's circle theorem bounds N's
//      spectral radius by `shift` = the sum of |every entry|; adding
//      shift*I to N shifts EVERY eigenvalue by the same +shift without
//      touching any eigenvector, which (since shift bounds the most
//      negative eigenvalue's magnitude too) makes the ORIGINAL largest
//      eigenvalue also the LARGEST-MAGNITUDE eigenvalue of the shifted
//      matrix — guaranteeing power iteration converges to the right vector
//      regardless of N's original sign pattern (THEORY.md works a worked
//      example; this is the textbook "power iteration + spectral shift"
//      fix, not a novel trick).
//   5. kHornPowerIterations rounds of v <- normalize((N+shift*I) v),
//      starting from (0.5,0.5,0.5,0.5) (an arbitrary vector with no zero
//      component along any eigenvector, so it can never accidentally start
//      orthogonal to the target — THEORY.md's numerics note).
//   6. q=(w,x,y,z)=v is the optimal ROTATION quaternion (up to the harmless
//      sign ambiguity q ~ -q); convert to a rotation matrix by the standard
//      formula, then t = mu2 - R*mu1.
// ---------------------------------------------------------------------------
inline bool build_rigid_from_covariance16(const double c[16], Rigid3* out)
{
    const double sum_w = c[0];
    if (sum_w < 1e-6) return false;   // degenerate: essentially no weighted correspondences

    const double mu1[3] = { c[1] / sum_w, c[2] / sum_w, c[3] / sum_w };
    const double mu2[3] = { c[4] / sum_w, c[5] / sum_w, c[6] / sum_w };

    double S[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            S[a][b] = c[7 + 3 * a + b] / sum_w - mu1[a] * mu2[b];

    // Horn's 4x4 "key matrix" N (row-major flat below), built from S — see
    // step 3 above; entries reproduced verbatim from Horn (1987) eq. 4-6.
    double N[4][4];
    N[0][0] = S[0][0] + S[1][1] + S[2][2];
    N[0][1] = N[1][0] = S[1][2] - S[2][1];
    N[0][2] = N[2][0] = S[2][0] - S[0][2];
    N[0][3] = N[3][0] = S[0][1] - S[1][0];
    N[1][1] = S[0][0] - S[1][1] - S[2][2];
    N[1][2] = N[2][1] = S[0][1] + S[1][0];
    N[1][3] = N[3][1] = S[2][0] + S[0][2];
    N[2][2] = -S[0][0] + S[1][1] - S[2][2];
    N[2][3] = N[3][2] = S[1][2] + S[2][1];
    N[3][3] = -S[0][0] - S[1][1] + S[2][2];

    // Gershgorin-safe spectral-radius bound (step 4): sum of |every entry|.
    double shift = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            shift += (N[i][j] < 0.0 ? -N[i][j] : N[i][j]);
    for (int i = 0; i < 4; ++i) N[i][i] += shift;

    // Shifted power iteration (step 5).
    double v[4] = { 0.5, 0.5, 0.5, 0.5 };
    for (int it = 0; it < kHornPowerIterations; ++it) {
        double nv[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                nv[i] += N[i][j] * v[j];
        double norm = std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2] + nv[3] * nv[3]);
        if (norm < 1e-12) norm = 1e-12;   // degenerate (all-zero N + shift==0): guard the divide, result is meaningless anyway
        for (int i = 0; i < 4; ++i) v[i] = nv[i] / norm;
    }

    const double qw = v[0], qx = v[1], qy = v[2], qz = v[3];   // step 6

    Rigid3 T{};
    T.R[0] = static_cast<float>(1.0 - 2.0 * (qy * qy + qz * qz));
    T.R[1] = static_cast<float>(2.0 * (qx * qy - qw * qz));
    T.R[2] = static_cast<float>(2.0 * (qx * qz + qw * qy));
    T.R[3] = static_cast<float>(2.0 * (qx * qy + qw * qz));
    T.R[4] = static_cast<float>(1.0 - 2.0 * (qx * qx + qz * qz));
    T.R[5] = static_cast<float>(2.0 * (qy * qz - qw * qx));
    T.R[6] = static_cast<float>(2.0 * (qx * qz - qw * qy));
    T.R[7] = static_cast<float>(2.0 * (qy * qz + qw * qx));
    T.R[8] = static_cast<float>(1.0 - 2.0 * (qx * qx + qy * qy));

    T.t[0] = static_cast<float>(mu2[0] - (T.R[0] * mu1[0] + T.R[1] * mu1[1] + T.R[2] * mu1[2]));
    T.t[1] = static_cast<float>(mu2[1] - (T.R[3] * mu1[0] + T.R[4] * mu1[1] + T.R[5] * mu1[2]));
    T.t[2] = static_cast<float>(mu2[2] - (T.R[6] * mu1[0] + T.R[7] * mu1[1] + T.R[8] * mu1[2]));

    *out = T;
    return true;
}

// ---------------------------------------------------------------------------
// tukey_biweight — the robust weight applied to a residual MAGNITUDE r
// (not the standard per-axis vector formulation — THEORY.md "Numerical
// considerations" names this a documented simplification for a scalar
// point-to-point residual) given a robust scale `s` (the caller computes
// s = kMadToSigma * median(|r|) over currently-eligible points every IRLS
// round — see main.cu's robust-fit loop): weight = (1-u^2)^2 for |u|<1,
// u = r/(kTukeyC*s), else 0 (a hard cutoff — points more than ~4.7 robust
// sigmas from the current fit are excluded entirely, not just downweighted).
// SHARED for the same "pure formula, would be pure transcription" reason as
// clamp_f/apply_rigid above.
// ---------------------------------------------------------------------------
CUDA_HOSTDEV inline float tukey_biweight(float r, float s, float c)
{
    if (s < 1e-9f) return 1.0f;   // degenerate scale (near-perfect fit): do not divide by ~0
    const float u = r / (c * s);
    if (u <= -1.0f || u >= 1.0f) return 0.0f;
    const float t = 1.0f - u * u;
    return t * t;
}

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// ===========================================================================
// MILESTONE 1 kernels — 2-level pyramidal Lucas-Kanade (01.03's Milestone 1,
// cited; full per-kernel documentation sits with the DEFINITION in
// kernels.cu, matching the repo's "header fixes the signature, kernels.cu
// carries the essay" convention).
// ===========================================================================
__global__ void downsample_area2x_kernel(const uint8_t* __restrict__ in, int inW, int inH,
                                         uint8_t* __restrict__ out);
__global__ void scharr_gradient_kernel(const uint8_t* __restrict__ img, int W, int H,
                                       float* __restrict__ gx_out, float* __restrict__ gy_out);
__global__ void structure_tensor_kernel(const float* __restrict__ gx, const float* __restrict__ gy,
                                        int W, int H,
                                        float* __restrict__ sxx_out, float* __restrict__ syy_out,
                                        float* __restrict__ sxy_out, float* __restrict__ min_eig_out);
__global__ void lk_iterate_kernel(const uint8_t* __restrict__ img0, const uint8_t* __restrict__ img1,
                                  int W, int H,
                                  const float* __restrict__ gx, const float* __restrict__ gy,
                                  const float* __restrict__ sxx, const float* __restrict__ syy,
                                  const float* __restrict__ sxy,
                                  float* __restrict__ flow_u, float* __restrict__ flow_v);
__global__ void upsample_flow_kernel(const float* __restrict__ coarse_u, const float* __restrict__ coarse_v,
                                     int coarseW, int coarseH,
                                     float* __restrict__ fine_u, float* __restrict__ fine_v,
                                     int fineW, int fineH);

// ===========================================================================
// MILESTONE 2 kernel — 3-D lifting with the depth-consistency guard.
// flow_u/flow_v/confidence: [kPixels] device IN (confidence = structure
// tensor min eigenvalue at level 0, from MILESTONE 1). d0/d1: [kPixels]
// device IN, depth maps (frame0/frame1, kInvalidDepth sentinel). P1_out/
// P2_out: [3*kPixels] device OUT, interleaved (x,y,z) meters (02.06's point-
// cloud layout convention, cited); meaningful ONLY where valid_out[i]!=0.
// valid_out: [kPixels] device OUT, 1 iff D0(x) is valid AND the flow
// confidence clears kMinConfidenceForLift AND the depth-consistency guard
// at x+flow passes (see file header stage 2).
// ---------------------------------------------------------------------------
__global__ void lift_scene_flow_kernel(const float* __restrict__ flow_u, const float* __restrict__ flow_v,
                                       const float* __restrict__ confidence,
                                       const float* __restrict__ d0, const float* __restrict__ d1,
                                       float* __restrict__ P1_out, float* __restrict__ P2_out,
                                       uint8_t* __restrict__ valid_out);

// ===========================================================================
// MILESTONE 3 kernels — residual + weighted covariance reduction (one round
// of the IRLS loop main.cu orchestrates; see file header stage 3).
// P1/P2: [3*n] device IN. T: the CURRENT rigid estimate, passed BY VALUE
// (02.06's convention for a per-iteration-changing small struct, cited: a
// kernel PARAMETER is read by every thread with the same broadcast
// efficiency as __constant__ memory, with none of the cudaMemcpyToSymbol
// upload boilerplate a per-iteration value would otherwise need).
// residual_vec_out: [3*n] device OUT (0 where valid[i]==0). residual_mag_out:
// [n] device OUT (0 where valid[i]==0).
// ---------------------------------------------------------------------------
__global__ void compute_residuals_kernel(int n, const float* __restrict__ P1, const float* __restrict__ P2,
                                         const uint8_t* __restrict__ valid, Rigid3 T,
                                         float* __restrict__ residual_vec_out,
                                         float* __restrict__ residual_mag_out);

// weighted_covariance_reduce_kernel — one thread per point, block-level
// shared-memory tree reduction of that block's kCovarWidth=16-scalar partial
// sum (the SAME "GPU partial reduce, host finishes it" split 02.06's ICP
// and 01.17's calibration use, cited). weight: [n] device IN (0 excludes a
// point entirely — both geometric invalidity and IRLS downweighting route
// through this ONE array, see main.cu's loop). block_partials: DEVICE OUT,
// blocks_for(n,kThreadsReduce)*kCovarWidth floats — block b's record lives
// at block_partials[b*kCovarWidth .. +16).
// ---------------------------------------------------------------------------
__global__ void weighted_covariance_reduce_kernel(int n, const float* __restrict__ P1, const float* __restrict__ P2,
                                                   const float* __restrict__ weight,
                                                   float* __restrict__ block_partials);

// ===========================================================================
// MILESTONE 4 kernels — residual segmentation.
// ===========================================================================
// threshold_mask_kernel — mask[i] = 1 iff valid[i] AND residual_mag[i] >
// threshold_m, else 0. n: [kPixels].
__global__ void threshold_mask_kernel(int n, const float* __restrict__ residual_mag,
                                      const uint8_t* __restrict__ valid, float threshold_m,
                                      uint8_t* __restrict__ mask_out);

// erode3x3_kernel / dilate3x3_kernel — binary 3x3 morphological primitives,
// one thread per pixel (the same map/stencil shape as every image kernel in
// this repo). Out-of-bounds neighbors read as 0 (erosion shrinks near the
// border, dilation cannot grow past it — a documented, honest boundary
// condition, THEORY.md notes the alternative of wrap/clamp).
__global__ void erode3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out);
__global__ void dilate3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out);

// ===========================================================================
// MILESTONE 4b kernels — connected-component labeling + size filter
// (kernels.cuh's Milestone-4b constants block derives kMinComponentSizePx;
// see that comment for why this stage exists).
// ===========================================================================

// ccl_init_kernel — label[i] = i (its own linear pixel index) iff mask[i],
// else kLabelNone. n: [kW*kH]. The seed every foreground pixel's label
// converges DOWNWARD from (01.06's ccl_init_kernel, cited).
__global__ void ccl_init_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label, int n);

// ccl_propagate_sweep_kernel — ONE sweep of 4-connected label propagation:
// every foreground pixel takes the MINIMUM label among itself and its four
// foreground neighbors (atomicMin — label only ever decreases, so this
// converges to a unique fixed point regardless of scheduling; kernels.cuh's
// kMaxCclSweeps comment proves the bound). changed: [1] device OUT, set
// nonzero (atomicOr) if ANY pixel's label decreased this sweep — main.cu's
// loop reads this back each round to detect convergence (01.06's main.cu
// loop shape, cited).
__global__ void ccl_propagate_sweep_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label,
                                           int W, int H, int* __restrict__ changed);

// component_size_count_kernel — ATOMIC SCATTER: every foreground pixel adds
// 1 to size_out[label[i]] (01.06/30.01's "dense accumulator keyed by
// canonical label" pattern, cited). size_out: [n] device OUT, MUST be
// zeroed by the caller first. After this kernel, size_out[L] for a
// canonical label L (a pixel index with label[L]==L) holds that
// component's total pixel count.
__global__ void component_size_count_kernel(const uint8_t* __restrict__ mask, const int* __restrict__ label,
                                            int* __restrict__ size_out, int n);

// component_filter_kernel — mask_out[i] = mask_in[i] AND
// size[label[i]] >= min_size, else 0. A pure per-pixel MAP reading the
// size_out component_size_count_kernel just scattered (n: [kW*kH]).
__global__ void component_filter_kernel(const uint8_t* __restrict__ mask_in, const int* __restrict__ label,
                                        const int* __restrict__ size, int min_size,
                                        uint8_t* __restrict__ mask_out, int n);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// Host-callable LAUNCH WRAPPERS — own the grid/block math + post-launch
// error check (CLAUDE.md §6.1 rule 7), visible to any translation unit.
// ===========================================================================
void launch_downsample_area2x(const uint8_t* d_in, int inW, int inH, uint8_t* d_out);
void launch_scharr_gradient(const uint8_t* d_img, int W, int H, float* d_gx, float* d_gy);
void launch_structure_tensor(const float* d_gx, const float* d_gy, int W, int H,
                             float* d_sxx, float* d_syy, float* d_sxy, float* d_min_eig);
void launch_lk_iterate(const uint8_t* d_img0, const uint8_t* d_img1, int W, int H,
                       const float* d_gx, const float* d_gy,
                       const float* d_sxx, const float* d_syy, const float* d_sxy,
                       float* d_flow_u, float* d_flow_v);
void launch_upsample_flow(const float* d_coarse_u, const float* d_coarse_v, int coarseW, int coarseH,
                          float* d_fine_u, float* d_fine_v, int fineW, int fineH);

// run_pyramidal_lk_gpu — the FULL Milestone-1 orchestration (builds the
// 2-level pyramid, coarse-to-fine loop calling the launch_ wrappers above —
// 01.03's run_pyramidal_lk_gpu shape, cited). d_img0_full/d_img1_full:
// [kPixels] device IN, grayscale level-0 frames. d_flow_u_out/d_flow_v_out/
// d_min_eig_out: [kPixels] device OUT, final level-0 flow + confidence.
void run_pyramidal_lk_gpu(const uint8_t* d_img0_full, const uint8_t* d_img1_full,
                          float* d_flow_u_out, float* d_flow_v_out, float* d_min_eig_out);

void launch_lift_scene_flow(const float* d_flow_u, const float* d_flow_v, const float* d_confidence,
                            const float* d_d0, const float* d_d1,
                            float* d_P1_out, float* d_P2_out, uint8_t* d_valid_out);

void launch_compute_residuals(int n, const float* d_P1, const float* d_P2, const uint8_t* d_valid, Rigid3 T,
                              float* d_residual_vec_out, float* d_residual_mag_out);

// launch_weighted_covariance_reduce — returns the number of blocks launched
// (== blocks_for(n,kThreadsReduce)) so the caller knows how many
// kCovarWidth-wide rows to download from d_block_partials and sum (02.06's
// convention, cited).
int launch_weighted_covariance_reduce(int n, const float* d_P1, const float* d_P2, const float* d_weight,
                                      float* d_block_partials);

void launch_threshold_mask(int n, const float* d_residual_mag, const uint8_t* d_valid, float threshold_m,
                           uint8_t* d_mask_out);

// launch_morphological_open — erode3x3 then dilate3x3 (the 07.09/01.18
// "launcher owns the whole ping-pong schedule" convention, cited).
// d_mask_inout: [kPixels] device IN/OUT. Allocates and frees its own scratch.
void launch_morphological_open(uint8_t* d_mask_inout);

// launch_ccl_init / launch_ccl_propagate_sweep — thin launch wrappers around
// the two Milestone-4b kernels above (01.06's split, cited): main.cu owns
// the SWEEP LOOP itself (reads d_changed back each round to detect
// convergence — the same per-iteration host/device round trip shape as this
// project's OWN Milestone-3 IRLS loop, and 01.06's main.cu CCL loop, both
// cited), not a self-contained launcher, because the loop's TERMINATION is
// itself part of what main.cu reports ([info] sweep count).
void launch_ccl_init(const uint8_t* d_mask, int* d_label, int W, int H);
void launch_ccl_propagate_sweep(const uint8_t* d_mask, int* d_label, int W, int H, int* d_changed);

// launch_component_size_filter — component_size_count_kernel then
// component_filter_kernel back to back (the erode+dilate "launcher owns the
// whole two-kernel schedule" convention, cited): allocates+zeros+frees its
// own [kPixels]-sized size scratch. d_mask_in: [n] device IN (post-
// morphology). d_label: [n] device IN (canonical CCL labels). d_mask_out:
// [n] device OUT (may alias d_mask_in for in-place filtering).
void launch_component_size_filter(const uint8_t* d_mask_in, const int* d_label, int min_size_px,
                                  uint8_t* d_mask_out, int n);

// ===========================================================================
// CPU reference (oracle) declarations — defined in reference_cpu.cpp.
// Per-stage twins mirror each kernel of the same name above (independently
// written — see this file's header). All pointers below are HOST pointers.
// ===========================================================================
void downsample_area2x_cpu(const uint8_t* in, int inW, int inH, uint8_t* out);
void scharr_gradient_cpu(const uint8_t* img, int W, int H, float* gx_out, float* gy_out);
void structure_tensor_cpu(const float* gx, const float* gy, int W, int H,
                          float* sxx_out, float* syy_out, float* sxy_out, float* min_eig_out);
void lk_iterate_cpu(const uint8_t* img0, const uint8_t* img1, int W, int H,
                    const float* gx, const float* gy,
                    const float* sxx, const float* syy, const float* sxy,
                    float* flow_u, float* flow_v);
void upsample_flow_cpu(const float* coarse_u, const float* coarse_v, int coarseW, int coarseH,
                       float* fine_u, float* fine_v, int fineW, int fineH);
void pyramidal_lk_cpu(const uint8_t* img0_full, const uint8_t* img1_full,
                      float* flow_u_out, float* flow_v_out, float* min_eig_out);

void lift_scene_flow_cpu(const float* flow_u, const float* flow_v, const float* confidence,
                         const float* d0, const float* d1,
                         float* P1_out, float* P2_out, uint8_t* valid_out);

void compute_residuals_cpu(int n, const float* P1, const float* P2, const uint8_t* valid, const Rigid3& T,
                           float* residual_vec_out, float* residual_mag_out);

// weighted_covariance_accumulate_cpu — the CPU twin of
// weighted_covariance_reduce_kernel: a DIRECT double-precision accumulation
// (no block-partial step needed — sequential code has no reduction tree to
// stage, 02.06's build_normal_system_cpu does the identical simplification,
// cited). out16: [16] HOST OUT, the SAME layout build_rigid_from_covariance16
// consumes.
void weighted_covariance_accumulate_cpu(int n, const float* P1, const float* P2, const float* weight,
                                        double out16[16]);

void threshold_mask_cpu(int n, const float* residual_mag, const uint8_t* valid, float threshold_m,
                        uint8_t* mask_out);
void erode3x3_cpu(const uint8_t* in, int W, int H, uint8_t* out);
void dilate3x3_cpu(const uint8_t* in, int W, int H, uint8_t* out);
void morphological_open_cpu(uint8_t* mask_inout);

// connected_components_cpu — classic Rosenfeld two-pass UNION-FIND (01.06's
// ccl_union_find_cpu, cited): a DELIBERATELY different algorithm from the
// GPU's iterative label propagation, converging to the SAME canonical
// (minimum-linear-index) labeling — see kernels.cu's ccl_propagate_sweep_
// kernel header for the fixed-point argument that makes this a meaningful
// bit-exact comparison rather than a foregone conclusion.
void connected_components_cpu(const uint8_t* mask, int* label, int W, int H);

// component_size_filter_cpu — the CPU twin of launch_component_size_filter:
// a direct two-pass count-then-filter (no atomics needed — a single
// sequential pass can just increment, 01.06's build_candidates_cpu does the
// identical simplification for its own atomic-scatter GPU counterpart,
// cited).
void component_size_filter_cpu(const uint8_t* mask_in, const int* label, int min_size_px,
                               uint8_t* mask_out, int n);

#endif // PROJECT_KERNELS_CUH
