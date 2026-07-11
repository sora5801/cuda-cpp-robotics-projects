// ===========================================================================
// kernels.cuh — interface for project 01.01
//               Full GPU image pipeline: debayer -> undistort -> rectify ->
//               resize -> normalize, staged AND fused, zero CPU copies
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver), kernels.cu (the GPU kernels),
// and reference_cpu.cpp (the independent CPU oracle). Every layout, camera
// constant, and sentinel all three must agree on lives HERE, once
// (CLAUDE.md paragraph 12) — a disagreement anywhere becomes a compile-time
// signature mismatch, not a silent runtime bug. See sibling flagship 01.02
// (stereo depth) for the same discipline applied to a cost volume; this file
// follows its documentation style closely.
//
// The five-stage pipeline in one paragraph (THEORY.md derives every step)
// -------------------------------------------------------------------------
// A real camera ISP hands perception a raw Bayer mosaic; five classical
// stages turn that into an ML-ready tensor:
//   1. DEBAYER    — bilinear demosaic: 1 raw sample/pixel -> 3 (R,G,B).
//   2+3. UNDISTORT+RECTIFY — ONE inverse-mapped remap: every output pixel
//        asks "where in the raw image did my light come from?", so every
//        output pixel gets an answer with NO holes (THEORY.md derives why
//        forward mapping would leave gaps and inverse mapping cannot).
//   4. RESIZE     — exact 2x area-average downscale (anti-aliased by
//        construction: a box filter matched to the decimation factor).
//   5. NORMALIZE  — per-channel affine to zero-mean/unit-std: the standard
//        "tensor the neural net expects" step every ML camera pipeline ends
//        with (this is the ONLY stage that leaves uint8 for float).
// This project builds the middle three (2+3+4) TWICE: once STAGED (one
// kernel per stage, each writing a full intermediate image to GLOBAL
// memory) and once FUSED (undistort+rectify+resize collapsed into a single
// kernel that never materializes the intermediate full-resolution image) —
// the kernel-fusion lesson that is this project's centerpiece (README
// "The algorithm in brief", THEORY.md "The GPU mapping").
//
// CAMERA MODEL — the single source of truth every stage after debayer reads
// -----------------------------------------------------------------------
// Convention (CLAUDE.md paragraph 3.2's stated exception): this file uses
// the CAMERA-OPTICAL frame, not the repo's default x-forward/y-left/z-up
// body frame — z-forward (down the optical axis), x-right, y-down (image
// convention: row 0 is the top of the image), stated here at this API
// boundary as required. Pixel (x, y) has NORMALIZED coordinates
// (x_n, y_n) = ((x - cx)/fx, (y - cy)/fy) with implicit z_n = 1.
//
// This project models ONE physical camera with a raw (distorted,
// mechanically un-rectified) sensor and asks the pipeline to recover the
// image an IDEAL, undistorted, canonically-mounted camera would have taken.
// Two simplifications are named honestly (README "Limitations & honesty"):
// the raw and rectified cameras share ONE intrinsic matrix K (no focal-
// length/principal-point change, only distortion removal + a small
// rotation), and resolution is unchanged by undistort/rectify (only the
// RESIZE stage changes resolution).
//
//   K = (fx, fy, cx, cy)                         — pixels, shared by both
//                                                   the raw and rectified
//                                                   cameras (see above).
//   Brown-Conrady distortion (k1, k2 radial; p1, p2 tangential) — maps an
//     IDEAL (undistorted) normalized point (xu, yu) to the DISTORTED
//     normalized point actually recorded by the raw sensor (THEORY.md "The
//     problem" derives this from the physics of a real lens):
//         r2 = xu*xu + yu*yu
//         radial = 1 + k1*r2 + k2*r2*r2
//         xd = xu*radial + 2*p1*xu*yu       + p2*(r2 + 2*xu*xu)
//         yd = yu*radial + p1*(r2 + 2*yu*yu) + 2*p2*xu*yu
//     This formula is a closed-form FORWARD map (ideal -> distorted); there
//     is no closed-form inverse (THEORY.md "Numerical considerations") —
//     the reason the remap LUT below is built by walking OUTPUT pixels
//     forward through distortion, never by inverting it.
//   R_rect_raw — a small RECTIFYING ROTATION (T_parent_child convention,
//     CLAUDE.md paragraph 12: parent = rectified frame, child = raw frame):
//     a unit vector expressed in the raw camera's own frame is re-expressed
//     in the rectified frame by v_rect = R_rect_raw * v_raw. Physically this
//     models a sensor mounted kRectifyAngleDeg off from the camera's
//     nominal boresight (a real, if small, manufacturing/mounting
//     tolerance) that "rectify" removes — 2 degrees about the camera's Y
//     (image-down) axis, i.e. a small yaw-like misalignment, chosen to be
//     GENUINE (not identity) so rectification visibly does something (see
//     the straightness gate in main.cu). R_rect_raw is a pure rotation
//     about Y:
//         R_rect_raw = [ cos(t)   0   sin(t) ]
//                      [   0      1     0    ]
//                      [-sin(t)   0   cos(t) ]
//     Going the OTHER way (a rectified-frame ray -> its raw-frame
//     representation, which is what the remap LUT needs) uses the inverse,
//     which for a rotation matrix is just the transpose: R_rect_raw^T.
//
// compute_source_pixel() below is the ONE function that turns an OUTPUT
// (rectified) pixel into the (u, v) coordinate to sample in the raw,
// DEBAYERED image — shared HOST+DEVICE code (kernels.cu's LUT kernel and
// reference_cpu.cpp's twin both call it). Per this repo's twin-
// independence ruling (see reference_cpu.cpp's header), a shared
// camera-model formula is permitted (it is data, not "the algorithm under
// test" — cf. 13.03's dynamics-model precedent) PROVIDED the project also
// carries a verification gate that does NOT route through it: main.cu's
// "roundtrip" gate hand-retypes both the forward projection and an
// independent fixed-point undistort, never calling this function.
//
// IMAGE LAYOUTS (row-major throughout; every stage after debayer keeps the
// SAME interleaved-channel convention so no stage has to reshuffle memory):
//   BAYER   (raw sensor, kFullW x kFullH): uint8_t[y*W + x] — ONE byte per
//     pixel, RGGB mosaic (see bayer_channel_at() below).
//   RGB     (debayer / remap / resize outputs): uint8_t[(y*W + x)*3 + c],
//     c in {0=R, 1=G, 2=B} — identical to a binary PPM (P6) pixel, so
//     writing an artifact is a direct memcpy (see main.cu write_ppm()).
//   LUT     (the undistort+rectify remap table, ALWAYS at kFullW x kFullH —
//     the fused kernel reads four full-resolution LUT entries per resized
//     output pixel; see "The GPU mapping" in THEORY.md): RemapSample[y*W+x].
//   NORMALIZED (final tensor, kResizedW x kResizedH): float[(y*W+x)*3 + c],
//     zero mean / unit std PER CHANNEL over the whole image (not per-pixel).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>     // sinf/cosf are NOT used here (angle is precomputed, see
                     // kRectCos/kRectSin below) but sqrt/fabs etc. appear in
                     // kernels.cu / reference_cpu.cpp through this same header

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. Same trick
// used across this repo (e.g. 18.01 snake robots, 10.03 massively parallel
// robot sim) for a function that MUST be textually identical on both sides
// of the host/device boundary: kernels.cu's remap-LUT kernel (nvcc) and
// reference_cpu.cpp's CPU twin (plain cl.exe, never sees a CUDA header)
// both call compute_source_pixel()/distort_forward()/bayer_channel_at()
// UNCHANGED. Sharing the source removes "did I transcribe the formula
// correctly twice?" as a possible explanation for a GPU-vs-CPU mismatch —
// what is left over is genuinely the numerics (THEORY.md "Numerical
// considerations"), which is what the VERIFY gate exists to catch.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Geometry constants — MUST match ../scripts/make_synthetic.py's "MUST
// MATCH kernels.cuh" block (CLAUDE.md paragraph 12: every layout documented
// once, cross-referenced everywhere). Full resolution mirrors sibling
// flagship 01.02's 384x288 (fast to build-verify, small committed sample,
// still large enough that the fusion memory-traffic argument below is a
// measurable, not academic, effect).
// ---------------------------------------------------------------------------
constexpr int kFullW = 384;                  // raw/debayer/remap width, px
constexpr int kFullH = 288;                  // raw/debayer/remap height, px
constexpr int kResizeFactor = 2;             // exact integer downscale (area-average, stage 4)
constexpr int kResizedW = kFullW / kResizeFactor;   // 192
constexpr int kResizedH = kFullH / kResizeFactor;   // 144

// ---------------------------------------------------------------------------
// Camera model constants — the single source of truth for the whole
// pipeline (also mirrored, with a "MUST MATCH" comment, in
// ../scripts/make_synthetic.py, which needs the SAME numbers to author a
// raw image that this pipeline can correctly recover).
//
// fx=fy=380, principal point at the exact image center (kFullW/2 - 0.5,
// kFullH/2 - 0.5 — the "-0.5" because pixel (0,0)'s CENTER is at continuous
// coordinate (0,0), so the image spans [-0.5, W-0.5); the true center pixel
// coordinate is (W-1)/2). k1/k2/p1/p2 give a moderate, clearly visible
// barrel distortion (about 8% inward pixel shift at the image corners —
// measured in README "Expected output") — enough to be pedagogically
// obvious without breaking the fixed-point undistort iteration's
// convergence (THEORY.md "Numerical considerations" derives the convergence
// radius). kRectifyAngleDeg is intentionally NONZERO (2 degrees) so
// rectification is a genuine, testable correction, not a no-op.
// ---------------------------------------------------------------------------
constexpr float kFx = 380.0f;                // focal length, px (shared by raw and rectified K — see file header)
constexpr float kFy = 380.0f;                // focal length, px
constexpr float kCx = (kFullW - 1) * 0.5f;   // principal point x, px = 191.5
constexpr float kCy = (kFullH - 1) * 0.5f;   // principal point y, px = 143.5

constexpr float kK1 = -0.22f;                // Brown-Conrady radial term 1 (dimensionless)
constexpr float kK2 =  0.06f;                // Brown-Conrady radial term 2 (dimensionless)
constexpr float kP1 =  0.0010f;              // Brown-Conrady tangential term 1 (dimensionless)
constexpr float kP2 = -0.0008f;              // Brown-Conrady tangential term 2 (dimensionless)

constexpr float kRectifyAngleDeg = 2.0f;     // rectifying rotation about the camera Y axis, degrees
// cos/sin of kRectifyAngleDeg, precomputed by hand (double-precision calc,
// rounded to float here) so every translation unit — host and device —
// links against bit-identical values instead of two independently rounded
// runtime cosf()/sinf() calls (which CAN differ in the last bit between
// host libm and device intrinsics — see THEORY.md "Numerical considerations").
constexpr float kRectCos = 0.9993908270f;    // cos(2 deg)
constexpr float kRectSin = 0.0348994967f;    // sin(2 deg)

// ---------------------------------------------------------------------------
// RemapSample — one LUT entry: the (u, v) floating-point coordinate, in the
// FULL-RESOLUTION debayered RGB image, that a given OUTPUT (rectified)
// pixel should bilinearly sample. A plain project-local struct (not CUDA's
// float2) keeps the contract obviously portable to a reader who has not
// yet met <vector_types.h> — no black-box types (CLAUDE.md paragraph 1).
// ---------------------------------------------------------------------------
struct RemapSample {
    float u;   // source column in the debayered image, kFullW x kFullH, may be fractional or out-of-range
    float v;   // source row    in the debayered image, kFullW x kFullH, may be fractional or out-of-range
};

// ---------------------------------------------------------------------------
// bayer_channel_at — which color channel the RGGB mosaic places at raw
// pixel (x, y). RGGB tiling (repeats every 2x2 pixels, x increasing
// rightward):
//
//     row y=0 (even): R G R G R G ...
//     row y=1 (odd):  G B G B G B ...
//     row y=2 (even): R G R G R G ...   (repeats)
//
// Returns 0=R, 1=G, 2=B — used by BOTH the debayer kernel and its CPU twin
// (independently re-typed logic overall, but this tiny piece — "which
// physical filter sits over which photosite" — is exactly the kind of
// fixed hardware fact both twins are allowed and expected to share, same
// spirit as the layout constants above it).
// ---------------------------------------------------------------------------
HD inline int bayer_channel_at(int x, int y)
{
    const bool even_row = (y & 1) == 0;
    const bool even_col = (x & 1) == 0;
    if (even_row && even_col)  return 0;   // R
    if (!even_row && !even_col) return 2;  // B
    return 1;                              // G (the two remaining cases)
}

// ---------------------------------------------------------------------------
// distort_forward — Brown-Conrady radial+tangential distortion, IDEAL
// (undistorted) normalized coords -> DISTORTED normalized coords. Pure
// closed-form math (see the file header for the formula and its physical
// reading); no iteration, no branches on convergence. Shared because this
// exact five-line formula IS the camera model being taught, not "the
// algorithm under test" (reference_cpu.cpp's independence ruling allows
// exactly this kind of shared formula) — main.cu's roundtrip gate
// independently re-derives it for the one place that must NOT trust this
// copy.
// ---------------------------------------------------------------------------
HD inline void distort_forward(float xu, float yu, float& xd, float& yd)
{
    const float r2 = xu * xu + yu * yu;
    const float radial = 1.0f + kK1 * r2 + kK2 * r2 * r2;
    xd = xu * radial + 2.0f * kP1 * xu * yu + kP2 * (r2 + 2.0f * xu * xu);
    yd = yu * radial + kP1 * (r2 + 2.0f * yu * yu) + 2.0f * kP2 * xu * yu;
}

// ---------------------------------------------------------------------------
// compute_source_pixel — the heart of the camera model: given an OUTPUT
// (rectified, undistorted) pixel coordinate, return the (u, v) coordinate
// to bilinearly sample in the RAW, DEBAYERED image. This is an INVERSE
// mapping in the imaging sense (output -> input) built entirely from
// FORWARD formulas (rotation-by-transpose, then distort_forward) — no
// iteration anywhere, which is exactly why remap is done this direction
// (THEORY.md "The algorithm" derives the alternative — forward-warping the
// raw image pixel by pixel — and why it leaves holes).
//
// Steps (each documented at the file-header level; here just the code):
//   1. Pixel -> ideal normalized ray in the RECTIFIED frame.
//   2. Rotate into the RAW frame: ray_raw = R_rect_raw^T * ray_rect (the
//      transpose because we go rectified -> raw, the opposite direction
//      R_rect_raw is named for; see file header).
//   3. Perspective-divide by the rotated ray's z to get the normalized
//      point an IDEAL raw-frame camera would have recorded.
//   4. distort_forward(): what the REAL (distorted) raw sensor recorded.
//   5. Apply K to land in raw pixel coordinates.
//
// Parameters: xo, yo — output pixel coordinates (kFullW x kFullH grid;
//   note undistort+rectify does NOT change resolution, only resize does).
// Returns: RemapSample{u, v} — may lie outside [0, kFullW) x [0, kFullH)
//   near the image border (rotation + distortion can push the sampling
//   point slightly out of frame); callers clamp when sampling (see
//   bilinear_sample_rgb in kernels.cu / reference_cpu.cpp).
// ---------------------------------------------------------------------------
HD inline RemapSample compute_source_pixel(int xo, int yo)
{
    // Step 1: output pixel -> ideal normalized ray, RECTIFIED frame.
    const float xr = (static_cast<float>(xo) - kCx) / kFx;
    const float yr = (static_cast<float>(yo) - kCy) / kFy;
    // ray_rect = (xr, yr, 1)

    // Step 2: rotate into the RAW frame using R_rect_raw^T (see file header
    // for the matrix and why the transpose is the correct direction here).
    // R_rect_raw^T = [ c, 0,-s ]   applied to (xr, yr, 1):
    //                [ 0, 1, 0 ]
    //                [ s, 0, c ]
    const float rx = kRectCos * xr - kRectSin;          // c*xr + 0*yr - s*1
    const float ry = yr;                                // 0*xr + 1*yr + 0*1
    const float rz = kRectSin * xr + kRectCos;          // s*xr + 0*yr + c*1

    // Step 3: perspective-divide -> ideal normalized point in the RAW
    // camera's own (undistorted) frame. rz stays very close to 1 for the
    // small kRectifyAngleDeg used here, so this division is always
    // well-conditioned (no near-zero-rz singularity at this rotation size).
    const float xn_raw = rx / rz;
    const float yn_raw = ry / rz;

    // Step 4: the physical lens distorts that ideal ray onto the sensor.
    float xd, yd;
    distort_forward(xn_raw, yn_raw, xd, yd);

    // Step 5: normalized distorted coords -> raw PIXEL coords via K.
    RemapSample s;
    s.u = kFx * xd + kCx;
    s.v = kFy * yd + kCy;
    return s;
}

// ---------------------------------------------------------------------------
// Normalize-stage constants — the two-pass, deterministic (no atomics)
// mean/std reduction (THEORY.md "Numerical considerations" and CLAUDE.md
// paragraph 12 both call for this to be documented explicitly):
//
//   kNormBlockSize — threads per block for the block-level partial-sum
//     kernel; also the tree-reduction width in shared memory.
//   kNormEps — variance floor before sqrt(), guarding a hypothetical
//     perfectly-flat channel (never hit on the committed sample, but a
//     divide-by-zero in production image code is a real, reported failure
//     mode worth guarding against explicitly rather than by accident).
// ---------------------------------------------------------------------------
constexpr int kNormBlockSize = 256;
constexpr float kNormEps = 1e-8f;

// ---------------------------------------------------------------------------
// GPU launch wrappers (kernels.cu). Every wrapper: computes its own launch
// geometry, launches, and calls CUDA_CHECK_LAST_ERROR — main.cu never
// touches <<<...>>> syntax directly (CLAUDE.md paragraph 6's "narrate the
// thought process" applied at the API-boundary level: main.cu reads as
// orchestration, kernels.cu explains the parallelism).
// ---------------------------------------------------------------------------

// Stage 1 — DEBAYER. d_bayer: kFullW*kFullH uint8_t (RGGB). d_rgb OUT:
// kFullW*kFullH*3 uint8_t (interleaved). Shared, byte-identical, by BOTH
// the staged and fused pipelines (file header: "debayer stays separate").
void launch_debayer_rggb(const unsigned char* d_bayer, unsigned char* d_rgb, int W, int H);

// Precompute the remap LUT ONCE (purely geometric — depends only on the
// camera model, never on image content) at full resolution; both the
// staged remap kernel and the fused kernel read the SAME buffer.
void launch_build_remap_lut(RemapSample* d_lut, int W, int H);

// STAGED stage 2+3 — undistort+rectify as one bilinear gather from the
// LUT, materializing a full kFullW x kFullH RGB image in global memory.
void launch_remap_bilinear(const unsigned char* d_rgb_in, const RemapSample* d_lut,
                           unsigned char* d_rgb_out, int W, int H);

// STAGED stage 4 — exact kResizeFactor x area-average downscale.
// d_rgb_in: Wf x Hf (full res). d_rgb_out: (Wf/kResizeFactor) x (Hf/kResizeFactor).
void launch_resize_area2x(const unsigned char* d_rgb_in, unsigned char* d_rgb_out, int Wf, int Hf);

// FUSED stages 2+3+4 — ONE kernel, one thread per RESIZED output pixel:
// for each of the kResizeFactor^2 full-resolution sub-pixels the resize
// would have averaged, look up its LUT entry and bilinear-sample the
// debayered image DIRECTLY, average in registers, write once. The
// full-resolution remapped image is NEVER written to global memory — see
// THEORY.md "The GPU mapping" for the derived memory-traffic savings this
// buys, and main.cu for the measured kernel-time comparison.
// d_rgb_in: Wf x Hf debayered image. d_lut_fullres: Wf x Hf (same LUT the
// staged path uses). d_rgb_out: (Wf/kResizeFactor) x (Hf/kResizeFactor).
void launch_fused_undistort_rectify_resize(const unsigned char* d_rgb_in,
                                           const RemapSample* d_lut_fullres,
                                           unsigned char* d_rgb_out, int Wf, int Hf);

// Stage 5a — per-block partial sums/sum-of-squares (double precision; see
// THEORY.md for why float32 accumulation over tens of thousands of terms
// is the wrong choice here). d_rgb: W*H*3 uint8_t. d_block_sum3 /
// d_block_sumsq3 OUT: num_blocks*3 double each, layout [block*3 + channel].
void launch_normalize_block_stats(const unsigned char* d_rgb, int W, int H,
                                  double* d_block_sum3, double* d_block_sumsq3,
                                  int num_blocks);

// Stage 5b — finalize: a SINGLE-THREAD kernel that walks the (small,
// typically < 200) block partials in a fixed sequential order and derives
// per-channel mean/std. Deliberately NOT a second parallel reduction or an
// atomicAdd-based one — see THEORY.md "Numerical considerations" for why a
// fixed, non-atomic summation order is this project's determinism choice.
// d_mean3 / d_std3 OUT: 3 floats each (R, G, B).
void launch_normalize_finalize(const double* d_block_sum3, const double* d_block_sumsq3,
                               int num_blocks, long long n_pixels,
                               float* d_mean3, float* d_std3);

// Stage 5c — apply the affine map out[c] = (in[c] - mean[c]) / std[c] to
// every pixel. d_rgb: W*H*3 uint8_t IN. d_out OUT: W*H*3 float.
void launch_normalize_apply(const unsigned char* d_rgb, float* d_out, int W, int H,
                            const float* d_mean3, const float* d_std3);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — INDEPENDENT reimplementations (per
// the project template's twin-independence ruling — see that file's
// header) of every launcher above, same math, plain nested loops, "_cpu"
// suffix. compute_source_pixel()/distort_forward()/bayer_channel_at() above
// are the deliberate, documented exception (shared camera-model formulas);
// the bilinear sampling, the debayer neighborhood logic, the fused-kernel
// re-derivation, and the normalize reduction are all typed a second time
// here, independently, so the GPU-vs-CPU comparison in main.cu is not
// blind to bugs in the geometry or the arithmetic.
// ---------------------------------------------------------------------------
void debayer_rggb_cpu(const unsigned char* bayer, unsigned char* rgb, int W, int H);
void build_remap_lut_cpu(RemapSample* lut, int W, int H);
void remap_bilinear_cpu(const unsigned char* rgb_in, const RemapSample* lut,
                        unsigned char* rgb_out, int W, int H);
void resize_area2x_cpu(const unsigned char* rgb_in, unsigned char* rgb_out, int Wf, int Hf);
void fused_undistort_rectify_resize_cpu(const unsigned char* rgb_in, const RemapSample* lut_fullres,
                                        unsigned char* rgb_out, int Wf, int Hf);
// Single-pass double-accumulation mean/var (the CPU needs none of the
// GPU's block-then-finalize dance — one sequential loop is already
// deterministic; THEORY.md compares the two strategies).
void normalize_stats_cpu(const unsigned char* rgb, int W, int H, double mean3[3], double std3[3]);
void normalize_apply_cpu(const unsigned char* rgb, float* out, int W, int H,
                         const double mean3[3], const double std3[3]);

#endif // PROJECT_KERNELS_CUH
