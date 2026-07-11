// ===========================================================================
// kernels.cuh — the single-sourced contract for project 01.19
//               Structured-light decoding (Gray code, phase shift) for 3D
//               scanners (Gray-code + phase-shift HYBRID scanner)
//
// Role in the project
// -------------------
// Every number and layout that main.cu, kernels.cu, reference_cpu.cpp, AND
// scripts/make_synthetic.py must agree on lives HERE, once (CLAUDE.md §12).
// The Python generator cannot literally #include this header, so its
// docstring names this file and repeats the numbers by hand — main.cu reads
// the committed data/sample/params.csv and ASSERTS its W/H/columns match the
// constants below, so a drift between the two would fail loudly on first run
// instead of silently corrupting the demo (see "assert_params_match" in
// main.cu).
//
// THE SCANNER MODEL (taught fully in THEORY.md "The math")
// ----------------------------------------------------------
// A structured-light scanner is a CAMERA (a normal pinhole imager) plus a
// PROJECTOR treated as an INVERSE CAMERA: instead of measuring which pixel a
// ray of light arrives at, a projector pixel EMITS a ray of light. The same
// pinhole equations that map a 3-D point to a camera pixel (project 01.17's
// camera model, cited here) map a 3-D point to a projector pixel — just run
// backwards. This project only encodes the projector's COLUMN axis (vertical
// stripe patterns), so we only ever need the projector's column intrinsics
// (fxp, cxp) — never its rows. A fixed projector column therefore does not
// pick out a single ray, it picks out an entire PLANE of rays (every row at
// that column) — the key geometric fact triangulation below exploits.
//
// Camera frame IS the world frame in this project (T_world_camera = I): all
// 3-D points, and the projector's own pose, are expressed directly in the
// camera's frame. The projector sits at a fixed baseline along the camera's
// +X axis with IDENTITY rotation (kRcp below) — a "parallel-axis" rig, the
// simplest legitimate scanner geometry and structurally identical to a
// stereo camera pair with one camera replaced by a projector (hence why the
// depth-precision relation derived in THEORY.md is the familiar stereo
// dz ~ z^2/(f*b) * d_disparity formula). kRcp is carried as a full 3x3
// matrix (not hard-coded away) so the ray-plane-intersection code in
// triangulate_kernel is the GENERAL formula — change kRcp/kTcp to a verged
// rig and the same code still triangulates correctly (README Exercise).
//
// TWO CODES, THEN A HYBRID (the catalog bullet, taught in full in THEORY.md)
// ----------------------------------------------------------------------------
//   Scheme A — GRAY CODE (binary temporal coding): kGrayBits patterns, each
//     captured TWICE (once direct, once photometrically inverted) so a
//     per-pixel threshold never has to be calibrated — see "gray_decode_
//     kernel" in kernels.cu. Decodes an ABSOLUTE, INTEGER projector column
//     in [0, kProjCols). Quantization floor: +-0.5 columns.
//   Scheme B — PHASE SHIFT (sinusoidal coding): kPhaseSteps patterns of a
//     sinusoid with period kPhasePeriodCols columns. Decodes a SUB-PIXEL
//     column WITHIN one period via atan2 — but wrapped: it cannot tell WHICH
//     of the kProjCols/kPhasePeriodCols periods a pixel is in.
//   HYBRID: Gray resolves the period (coarse, absolute, integer); phase
//     resolves the position within the period (fine, sub-pixel, wrapped).
//     hybrid_combine_kernel is where the two get glued together — see its
//     header comment in kernels.cu for the exact period-snapping rule.
//
// PIXEL-CENTER CONVENTION (shared with make_synthetic.py, load-bearing)
// -----------------------------------------------------------------------
// Every camera ray direction in this project is built from PIXEL CENTERS:
//     dx = (col + 0.5 - kCamCx) / kCamFx,  dy = (row + 0.5 - kCamCy) / kCamFy
// The synthetic generator computes its ground-truth camera rays the exact
// same way. Get this 0.5 wrong in only one of the two languages and every
// gate below fails with a suspicious, uniform sub-pixel bias — a classic,
// specifically robotics-flavoured off-by-half-pixel bug (see THEORY.md
// "Numerical considerations").
//
// PATTERN-STACK MEMORY LAYOUT (the GPU-mapping decision, argued in THEORY.md
// "The GPU mapping" and in kernels.cu's kernel header comments)
// -----------------------------------------------------------------------
// All captured pattern frames are stored FLAT and PATTERN-MAJOR:
//     pattern_stack[p * kNPix + pix]      pix = row * kCamW + col
// i.e. every pattern is one contiguous [kCamH x kCamW] image, and the
// patterns are laid out one after another. Every decode kernel below is a
// per-PIXEL thread that loops over its handful of patterns (7 Gray bits, or
// 4 phase steps) — that pattern loop is the kernel's INNER loop, and because
// the layout is pattern-major, every iteration of that shared inner loop has
// the WHOLE WARP reading kNPix-apart-but-mutually-CONSECUTIVE addresses
// (thread t reads pattern_stack[p*kNPix + pix_of_t], and pix_of_t is
// consecutive across the warp) — one coalesced 128-byte transaction per
// iteration, every iteration. The alternative (pixel-major: all patterns for
// one pixel adjacent) would make that same inner-loop step read one float
// per warp per 4-byte-wide cache line touched — the exact coalescing lesson
// 08.01's transposed noise array teaches, applied here to images instead of
// rollouts (cited in kernels.cu).
//
// WHY NO ATOMICS ANYWHERE IN THIS PROJECT (contrast cited in THEORY.md)
// -----------------------------------------------------------------------
// Every kernel below is a pure per-pixel MAP: output[pix] is a function of
// input[*, pix] for THAT SAME pix only — no thread ever reads or writes
// another thread's pixel, so there is nothing to arbitrate. Contrast with
// 01.18 (depth completion), whose scatter/gather fill-in genuinely needs
// atomics because multiple source pixels can contribute to one destination
// pixel. Structured light's per-pixel independence, top to bottom of the
// pipeline, is what makes it such a clean teaching example of the "map"
// pattern at scale.
//
// Read this after: README.md + THEORY.md. Read this before: kernels.cu,
// reference_cpu.cpp, main.cu (in that order — kernels.cu documents the
// per-kernel math in full; this header only fixes the shared numbers).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Camera intrinsics (pinhole, no distortion — the calibration this project
// ASSUMES was already done by a project like 01.16/01.17; see README "System
// context"). Units: pixels. W/H: pixel counts.
// ---------------------------------------------------------------------------
constexpr int   kCamW  = 200;                    // image width (columns), px
constexpr int   kCamH  = 150;                    // image height (rows), px
constexpr int   kNPix  = kCamW * kCamH;          // 30000 pixels total, one thread each
constexpr float kCamFx = 220.0f;                 // camera focal length, x (px)
constexpr float kCamFy = 220.0f;                 // camera focal length, y (px)
constexpr float kCamCx = 100.0f;                 // camera principal point, x (px) = W/2
constexpr float kCamCy = 75.0f;                  // camera principal point, y (px) = H/2

// ---------------------------------------------------------------------------
// Projector intrinsics — COLUMN AXIS ONLY (see file header: this project
// never encodes projector rows, so fyp/cyp simply do not exist in this
// model). kProjCols is the projector's encoded column resolution; Gray code
// needs ceil(log2(kProjCols)) bits, and 2^kGrayBits == kProjCols EXACTLY
// here (128 = 2^7) — a deliberate choice so Gray code alone already resolves
// every column with zero wasted codewords (THEORY.md "The algorithm").
// ---------------------------------------------------------------------------
constexpr int   kProjCols = 128;                 // Wp: projector columns encoded
constexpr float kProjFx   = 180.0f;              // projector focal length, column axis (px)
constexpr float kProjCx   = 64.0f;               // projector principal column = Wp/2 (px)

// ---------------------------------------------------------------------------
// Extrinsics: projector pose EXPRESSED IN THE CAMERA FRAME (T_camera_projector
// in the repo's T_parent_child convention — "projector expressed in camera").
// kRcp is row-major 3x3 (kRcp[3*r+c]); kTcp is the projector's optical center
// in camera coordinates (meters). IDENTITY rotation + pure-X translation is
// the "parallel-axis" rig this project's synthetic scanner uses (file header
// explains why); triangulate_kernel nonetheless uses the FULL R,t so the
// code stays correct if a learner changes these (README Exercise).
// ---------------------------------------------------------------------------
constexpr float kBaselineM = 0.12f;              // projector-camera baseline along +X (m)
constexpr float kRcp[9] = { 1.0f, 0.0f, 0.0f,    // T_camera_projector rotation, row-major,
                            0.0f, 1.0f, 0.0f,    // identity here: parallel optical axes
                            0.0f, 0.0f, 1.0f };
constexpr float kTcp[3] = { kBaselineM, 0.0f, 0.0f };  // projector center in camera frame (m)

// ---------------------------------------------------------------------------
// Temporal code parameters (THEORY.md "The math" derives both fully).
// ---------------------------------------------------------------------------
constexpr int   kGrayBits        = 7;            // N: Gray-code bit planes (2^7 = kProjCols)
constexpr int   kPhaseSteps      = 4;             // 4-step phase shift (minimum-robust; 3 is the
                                                   // theoretical minimum — THEORY.md derives why 4)
constexpr float kPhasePeriodCols = 8.0f;          // P: projector columns per sinusoid period
constexpr int   kNumPeriods      = 16;            // kProjCols / kPhasePeriodCols (must divide evenly)

// Pattern-stack pixel counts for each stack (used by main.cu's loader and by
// both decode kernels' launch-config math): kGrayBits DIRECT + kGrayBits
// INVERSE frames for Gray, kPhaseSteps frames for phase.
constexpr int kNumGrayFrames  = kGrayBits;        // per direct/inverse stack
constexpr int kNumPhaseFrames = kPhaseSteps;

// ---------------------------------------------------------------------------
// Confidence / validity.
//
// kDefaultConfidenceFloor: the modulation-amplitude threshold below which a
// pixel's phase (and hence hybrid column) is UNTRUSTED and masked out of the
// point cloud (THEORY.md "Numerical considerations": atan2's conditioning
// collapses as B -> 0 — the angle of a near-zero vector is dominated by
// noise). Units: intensity counts on the project's [0,255] camera scale —
// the same units make_synthetic.py's rendered PGMs are quantized to.
//
// Value TUNED (data/README.md "How the sample was tuned" records the sweep)
// against TWO competing failure modes measured on the committed sample:
//   * false-reject of GOOD surfaces (background/sphere/box) — stays under
//     ~2.5% at this floor;
//   * false-ACCEPT of the dark-albedo stripe cohort — at a looser floor
//     (e.g. 15) a small tail of dark-stripe pixels survives the modulation
//     check yet still carries a WRONG Gray-decoded period (a bit-threshold
//     error under low SNR, independent of the phase modulation this floor
//     actually measures) and reconstructs a grossly wrong depth — a real,
//     if narrow, blind spot of confidence-by-modulation-alone (documented
//     honestly in THEORY.md "Numerical considerations" and README
//     "Limitations" rather than hidden). Raising the floor to 25 pushes the
//     dark-stripe cohort's survivor rate down to where this residual risk
//     measures zero violations on the committed sample (see the
//     dark_stripe_honesty gate in main.cu) while keeping every legitimate
//     surface's false-reject rate in the low single digits.
// ---------------------------------------------------------------------------
constexpr float kDefaultConfidenceFloor = 25.0f;

// ---------------------------------------------------------------------------
// Boundary stress test parameters (README/THEORY "gray_vs_binary lesson";
// kernels.cu's boundary_stress_kernel). kBoundaryNoiseSigma is in the SAME
// [0,1]-normalized "ideal bit" units the stress test's own tiny 1-D forward
// model uses (see the kernel's header comment) — NOT the same units as the
// main pipeline's intensity-count noise (NOISE_SIGMA in make_synthetic.py);
// this is a deliberately separate, analytically controlled experiment
// measuring a property of the CODES, not of this project's specific camera
// noise level (kernels.cu explains the distinction).
// ---------------------------------------------------------------------------
constexpr int   kBoundarySamples     = 20000;    // number of synthetic 1-D probe positions
constexpr float kBoundaryNoiseSigma  = 0.15f;    // per-bit analog noise std-dev, [0,1] units

// Sentinel values: printed/stored where "no valid answer" must still occupy
// a slot in a dense array (never NaN in artifacts — NaN complicates CSV/diff
// tooling and hides silently in sums; an out-of-range sentinel is loud).
constexpr float kInvalidColumnF = -1.0f;          // sentinel for float column outputs
constexpr int   kInvalidColumnI = -1;              // sentinel for integer column outputs

// Ground-truth surface labels (written by make_synthetic.py into
// truth_surface.bin, read back by main.cu ONLY to score the reconstruction
// gates — never consumed by the decode/triangulation pipeline itself, which
// has no notion of "which analytic surface" produced a pixel; see README
// "Limitations" for why using truth labels only for grading, not decoding,
// is not circular).
constexpr unsigned char kSurfNone       = 0;       // no analytic surface / outside all footprints
constexpr unsigned char kSurfBackground = 1;       // the tilted background plane
constexpr unsigned char kSurfSphere     = 2;       // the sphere
constexpr unsigned char kSurfBox        = 3;       // the box's fronto-parallel top face

// ===========================================================================
// GPU kernel declarations. Fenced behind __CUDACC__ so reference_cpu.cpp
// (compiled by cl.exe, which does not know __global__) can still include
// this header for the shared constants and launcher/CPU-oracle prototypes
// below (same trick documented in the template's kernels.cuh).
// ===========================================================================
#ifdef __CUDACC__

// Stage 1 — Gray-code bit extraction + Gray-to-binary decode. One thread per
// camera pixel. Declared here; fully documented with the definition in
// kernels.cu (purpose, launch config, memory spaces, thread mapping).
__global__ void gray_decode_kernel(const float* __restrict__ d_direct,     // [kGrayBits*kNPix]
                                   const float* __restrict__ d_inverse,   // [kGrayBits*kNPix]
                                   int*         __restrict__ d_gray_col,  // [kNPix] OUT
                                   int n);

// Stage 2 — 4-step phase-shift atan2 decode: wrapped phase + modulation
// (confidence). One thread per camera pixel.
__global__ void phase_decode_kernel(const float* __restrict__ d_phase,       // [kPhaseSteps*kNPix]
                                    float*       __restrict__ d_phase_out,  // [kNPix] OUT, radians [0,2pi)
                                    float*       __restrict__ d_confidence, // [kNPix] OUT, intensity counts
                                    int n);

// Stage 3 — hybrid combine: Gray-resolved period + phase-refined sub-pixel
// offset, with confidence masking. One thread per camera pixel.
__global__ void hybrid_combine_kernel(const int*   __restrict__ d_gray_col,    // [kNPix]
                                      const float* __restrict__ d_phase,      // [kNPix] radians
                                      const float* __restrict__ d_confidence, // [kNPix]
                                      float confidence_floor,
                                      float*         __restrict__ d_hybrid_col, // [kNPix] OUT
                                      unsigned char* __restrict__ d_valid,      // [kNPix] OUT (0/1)
                                      int n);

// Stage 4 — ray/projector-plane triangulation. One thread per camera pixel.
__global__ void triangulate_kernel(const float*         __restrict__ d_hybrid_col, // [kNPix]
                                   const unsigned char* __restrict__ d_valid,      // [kNPix]
                                   float*         __restrict__ d_xyz,        // [kNPix*3] OUT (x,y,z), m
                                   unsigned char* __restrict__ d_point_valid,// [kNPix] OUT
                                   int n);

// Stage 5 — the Gray-vs-plain-binary boundary stress test (README/THEORY
// "gray_vs_binary lesson"). One thread per synthetic boundary-crossing
// sample; reuses the SAME bit-decision physics as gray_decode_kernel but on
// a purpose-built 1-D sweep instead of the 2-D scene (kernels.cu explains
// why this is still a legitimate, non-cherry-picked measurement).
//
// Noise is pre-generated ON THE HOST (xorshift32, same pattern as 08.01's
// per-tick MPPI noise upload) and passed in as d_noise rather than a per-
// sample seed: both this kernel and its CPU twin then do nothing but float
// add/compare/threshold on IDENTICAL inputs, so their outputs are bit-
// reproducible without relying on device and host transcendental libraries
// (sinf/cosf/logf) agreeing to the last bit — a real hazard this project
// deliberately designs around (THEORY.md "Numerical considerations").
// d_noise layout per sample i: d_noise[i*2*kGrayBits + k] for k in
// [0,kGrayBits) is the Gray-path bit-k noise draw; k in [kGrayBits,2*kGrayBits)
// is the Binary-path bit-(k-kGrayBits) noise draw.
__global__ void boundary_stress_kernel(const float* __restrict__ d_true_x, // [n] true column position
                                       const float* __restrict__ d_noise, // [n*2*kGrayBits] pre-drawn noise
                                       int* __restrict__ d_decoded_gray,   // [n] OUT
                                       int* __restrict__ d_decoded_binary, // [n] OUT
                                       int n);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Host launch wrappers (defined in kernels.cu; callable from any translation
// unit — only their DEFINITIONS need nvcc). Each owns its grid/block math
// and the post-launch CUDA_CHECK_LAST_ERROR call (CLAUDE.md §6.1 rule 7).
// ---------------------------------------------------------------------------
void launch_gray_decode(const float* d_direct, const float* d_inverse, int* d_gray_col, int n);
void launch_phase_decode(const float* d_phase, float* d_phase_out, float* d_confidence, int n);
void launch_hybrid_combine(const int* d_gray_col, const float* d_phase, const float* d_confidence,
                           float confidence_floor, float* d_hybrid_col, unsigned char* d_valid, int n);
void launch_triangulate(const float* d_hybrid_col, const unsigned char* d_valid,
                        float* d_xyz, unsigned char* d_point_valid, int n);
void launch_boundary_stress(const float* d_true_x, const float* d_noise,
                            int* d_decoded_gray, int* d_decoded_binary, int n);

// ---------------------------------------------------------------------------
// CPU oracle twins (reference_cpu.cpp). Per the template's independence
// ruling: these are INDEPENDENTLY-written plain-C++ implementations of the
// same math, not thin wrappers around shared device code — see
// reference_cpu.cpp's file header for the full ruling and why it matters.
// Signatures mirror the launchers exactly so main.cu calls both uniformly.
// ---------------------------------------------------------------------------
void gray_decode_cpu(const float* direct, const float* inverse, int* gray_col, int n);
void phase_decode_cpu(const float* phase, float* phase_out, float* confidence, int n);
void hybrid_combine_cpu(const int* gray_col, const float* phase, const float* confidence,
                        float confidence_floor, float* hybrid_col, unsigned char* valid, int n);
void triangulate_cpu(const float* hybrid_col, const unsigned char* valid,
                     float* xyz, unsigned char* point_valid, int n);
void boundary_stress_cpu(const float* true_x, const float* noise,
                         int* decoded_gray, int* decoded_binary, int n);

#endif // PROJECT_KERNELS_CUH
