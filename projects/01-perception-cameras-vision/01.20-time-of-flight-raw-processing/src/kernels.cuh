// ===========================================================================
// kernels.cuh — the single-sourced contract for project 01.20
//               Time-of-flight raw processing: phase unwrapping, flying-pixel
//               removal (continuous-wave INDIRECT time-of-flight, "iToF")
//
// Role in the project
// -------------------
// Every number and layout that main.cu, kernels.cu, reference_cpu.cpp, AND
// scripts/make_synthetic.py must agree on lives HERE, once (CLAUDE.md §12).
// The Python generator cannot literally #include this header, so its
// docstring names this file and repeats the numbers by hand — main.cu reads
// the committed data/sample/params.csv and ASSERTS its values match the
// constants below (assert_params_match in main.cu), so a Python/C++ drift
// fails loudly on the very first run instead of silently corrupting the demo.
//
// THE SENSOR MODEL (taught fully in THEORY.md "The problem" / "The math")
// --------------------------------------------------------------------------
// A continuous-wave indirect-ToF (iToF) camera illuminates the scene with an
// amplitude-modulated infrared source (modulation frequency `f`, e.g. an LED
// or VCSEL array driven by a sine/square wave) and, PER PIXEL, cross-
// correlates the returning light against four internally-generated reference
// signals shifted by 0, 90, 180, 270 degrees relative to the modulation drive
// ("4-tap" or "4-phase" demodulation — the sensor-level analogue of a lock-in
// amplifier, implemented once per pixel in the pixel's own charge-storage
// wells). Each tap integrates a correlation value `C_k`, k in {0,1,2,3}; this
// project's forward model (kernels.cu / reference_cpu.cpp / make_synthetic.py
// all share the SAME formula, defined once here in prose):
//
//     C_k(phi) = A + B * cos(phi + k*pi/2)          k = 0, 1, 2, 3
//
// where `phi` is the phase delay the light picked up on its round trip
// (0 <= phi < 2*pi, WRAPPED — the whole reason unwrapping is this project's
// subject), `A` is an ambient/DC offset (background IR + a fixed pedestal),
// and `B >= 0` is the modulation amplitude — this project's per-pixel
// CONFIDENCE signal, exactly as in project 01.19's phase-shift stage (same
// arithmetic, different physical carrier: 01.19 modulates a PROJECTOR
// COLUMN in space; this project modulates ILLUMINATION INTENSITY in TIME).
// Expanding the four taps and differencing (THEORY.md "The math" derives
// this from the correlation/mixer integral):
//
//     phase     = atan2(C3 - C1, C0 - C2)                    in [0, 2*pi)
//     amplitude = 0.5 * sqrt((C3-C1)^2 + (C0-C2)^2)           = B
//
// exactly the 01.19 pattern (`phase_decode_kernel`'s atan2/sqrt pair) with a
// different tap-offset sign convention (kernels.cu documents the convention
// choice); the ambient term `A` cancels EXACTLY in both differences, the
// same "offset invariance" property 01.19's phase_ambient_invariance gate
// exploits — reused here as this project's own offset_invariance gate.
//
// DEPTH FROM PHASE, AND THE AMBIGUITY (kernels.cu Stage 2/3; THEORY.md "The
// math" derives this from "distance = (speed of light * time of flight)/2"):
//
//     distance(phi) = (c * phi) / (4 * pi * f)      ambiguity range D = c/(2*f)
//
// A single frequency's phase only resolves distance MODULO D — the
// "designed aliasing demonstration" this project's aliasing_demo gate
// measures directly on the committed scene's far background.
//
// TWO FREQUENCIES, THEN UNWRAPPING (the catalog bullet's "phase unwrapping")
// --------------------------------------------------------------------------
// This project uses TWO modulation frequencies, `kFreq1Hz` (60 MHz, fine,
// SHORT ambiguity range `kAmbig1M` = c/(2*60MHz) ~= 2.50 m) and `kFreq2Hz`
// (20 MHz, coarse, LONG ambiguity range `kAmbig2M` = c/(2*20MHz) ~= 7.49 m).
// The ratio is EXACTLY 3 (`kFreq1Hz / kFreq2Hz == 3.0`), a deliberate design
// choice (kernels.cuh "Scene depth budget" below) so `kAmbig2M` is itself an
// exact integer multiple of `kAmbig1M` — the classic CRT-style ("Chinese
// Remainder Theorem"-flavoured) dual-frequency unwrap: freq2's phase alone
// already gives an UNAMBIGUOUS (if noisier) coarse depth over the WHOLE
// scene depth budget (kMaxSceneDepthM < kAmbig2M by construction), while
// freq1's phase gives a MUCH more precise (per the noise-scaling law
// THEORY.md derives, sigma_d ~ 1/f) but WRAPPED fine depth. Unwrapping
// (`dual_freq_unwrap_kernel`) searches the small set of integer "wrap
// counts" `n1` that make freq1's candidate depth AGREE with freq2's coarse
// depth, then reports the fine (freq1) depth at the winning wrap count —
// exactly 01.19's "Gray resolves the period, phase refines it" hybrid
// pattern, with 01.19's (Gray code, phase-shift) pair replaced by
// (low-frequency CW, high-frequency CW).
//
// FLYING PIXELS (the catalog bullet's other half; kernels.cu Stage 5 and
// make_synthetic.py both derive this in full) arise because a real ToF
// PIXEL, not a ray, receives light: at a depth-discontinuity edge (a
// foreground silhouette against a background), the pixel's active area
// straddles BOTH surfaces and its four correlation taps integrate the AREA-
// WEIGHTED SUM of two returns. Because tap values are LINEAR in incident
// correlated power, this sum is a genuine COMPLEX PHASOR ADDITION
// (`w*B_fg*exp(i*phi_fg) + (1-w)*B_bg*exp(i*phi_bg)`, THEORY.md derives the
// tap-to-phasor equivalence) — the decoded phase of that SUM is, in
// general, at NEITHER surface's true depth and not their weighted AVERAGE
// either; the pixel appears to "fly" in space between the two surfaces (or
// occasionally beyond either, when the phasors partially cancel). Detecting
// this (`flying_pixel_detect_kernel`) uses two independent, physically
// motivated signals: a local depth-DISCONTINUITY test (a flying pixel sits
// between two very different depths) and an amplitude-RATIO test (phasor
// addition of two out-of-phase returns is, by the triangle inequality,
// generally WEAKER than either constituent alone — a physical tell no
// single-surface return produces).
//
// PIXEL-CENTER CONVENTION, MEMORY LAYOUT, ATOMICS: identical to 01.19 (see
// that project's kernels.cuh for the full arguments; not re-derived here).
// Camera rays use `dx=(col+0.5-cx)/fx, dy=(row+0.5-cy)/fy`; the tap stack is
// FLAT, PATTERN-MAJOR (`taps[k*n + pix]`) for the same coalescing reason;
// every kernel here except `flying_pixel_detect_kernel` is a pure per-pixel
// MAP with no atomics. `flying_pixel_detect_kernel` is this project's ONE
// genuine STENCIL kernel (a 3x3 neighborhood GATHER, still race-free: each
// thread only ever WRITES its own output pixel) — the one new GPU-mapping
// idea 01.19 does not need (that project is a pure map top to bottom;
// THEORY.md "The GPU mapping" contrasts the two explicitly).
//
// Read this after: README.md + THEORY.md. Read this before: kernels.cu,
// reference_cpu.cpp, main.cu (in that order — kernels.cu documents the
// per-kernel math in full; this header only fixes the shared numbers).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Camera intrinsics (pinhole, no distortion — this project ASSUMES a prior
// calibration, exactly like 01.19; see README "System context"). Units:
// pixels. A small QQVGA-class resolution: real iToF sensors (Kinect-v2-era
// PMD/Melexis-class parts, Azure Kinect's NFOV mode) run from roughly
// 240x180 up to 1024x1024 — this project's `kCamW x kCamH` is a deliberately
// small teaching size (kernels.cuh "why small" below), not a claim about
// real sensor resolution.
// ---------------------------------------------------------------------------
constexpr int   kCamW  = 160;                    // image width (columns), px
constexpr int   kCamH  = 120;                    // image height (rows), px
constexpr int   kNPix  = kCamW * kCamH;          // 19200 pixels, one thread each (map kernels)
constexpr float kCamFx = 150.0f;                 // camera focal length, x (px) -> ~56 deg horizontal FOV
constexpr float kCamFy = 150.0f;                 // camera focal length, y (px)
constexpr float kCamCx = 80.0f;                  // camera principal point, x (px) = W/2
constexpr float kCamCy = 60.0f;                  // camera principal point, y (px) = H/2

// ---------------------------------------------------------------------------
// Physical constant. Exact value (SI, m/s) — CODATA-exact by definition of
// the metre; used verbatim everywhere depth is computed from phase, so any
// drift between kernels.cu/reference_cpu.cpp/make_synthetic.py would show up
// immediately as a gate failure rather than a silently-wrong constant.
// ---------------------------------------------------------------------------
constexpr float kSpeedOfLightMps = 299792458.0f;

// ---------------------------------------------------------------------------
// Modulation frequencies (THEORY.md "The math" derives the ambiguity-range
// formula D = c/(2f)). kFreq1Hz is the FINE, high-frequency channel: short
// ambiguity range, but per the noise-scaling law (sigma_d ~ c/(4*pi*f) *
// sigma_phi/B) proportionally BETTER depth precision. kFreq2Hz is the
// COARSE, low-frequency channel: long ambiguity range, worse raw precision,
// used ONLY to resolve which wrap of kFreq1Hz is correct.
//
// THE RATIO IS EXACTLY 3 (a deliberate scene-design choice, not a real-
// hardware constraint): kAmbig2M = 3 * kAmbig1M exactly, so kFreq2Hz's own
// ambiguity range already covers the ENTIRE scene depth budget
// (kMaxSceneDepthM, below) with room to spare — freq2 alone is therefore
// UNAMBIGUOUS (if noisy) everywhere in this scene, and the dual-frequency
// unwrap collapses to "search the (small, exactly `kMaxWraps1`-sized) set of
// integer wraps of freq1 that agree with freq2's unambiguous coarse depth" —
// the classic CRT-style consistency search, taught in full generality in
// kernels.cu (the search loop also considers `kMaxWraps2` candidate wraps of
// freq2 itself, even though this scene's design makes freq2's own wrap
// count always 0 — the code does not hard-code that fact, so it stays
// correct if a learner changes the scene to exceed kAmbig2M, README
// Exercise).
// ---------------------------------------------------------------------------
constexpr float kFreq1Hz = 60.0e6f;              // fine channel, 60 MHz
constexpr float kFreq2Hz = 20.0e6f;              // coarse channel, 20 MHz (kFreq1Hz / kFreq2Hz == 3 exactly)

// Ambiguity ranges, D = c/(2f), computed once so every file uses IDENTICAL
// floats (no separate hand-typed approximation to drift): kAmbig1M ~= 2.498
// m, kAmbig2M ~= 7.495 m (both computed at compile time by the constexpr
// evaluator — no runtime cost, and no place for a Python/C++ mismatch to
// hide, unlike a hand-copied decimal would invite).
constexpr float kAmbig1M = kSpeedOfLightMps / (2.0f * kFreq1Hz);
constexpr float kAmbig2M = kSpeedOfLightMps / (2.0f * kFreq2Hz);

constexpr int kNumTaps = 4;                      // 4-tap (0/90/180/270 deg) demodulation, this project's scheme

// ---------------------------------------------------------------------------
// Scene depth budget and wrap-count search bounds (THEORY.md "The math"
// "Unwrapping as an integer consistency search"). kMaxSceneDepthM is a
// documented UPPER BOUND on any truth depth make_synthetic.py ever renders
// (checked by that script's own diagnostic printout) — it is what makes
// kMaxWraps2 == 1 valid (freq2 never wraps in this scene) and bounds the
// dual_freq_unwrap_kernel search to a handful of candidates, not an
// unbounded loop.
// ---------------------------------------------------------------------------
constexpr float kMaxSceneDepthM = 6.0f;          // documented ceiling on any rendered truth depth (m)
// ceil(kMaxSceneDepthM / kAmbig1M): with kAmbig1M ~= 2.498 m, this is 3 —
// freq1 can wrap up to twice within the scene's depth budget, so 3 candidate
// wrap counts (0, 1, 2) are searched.
constexpr int kMaxWraps1 = 3;
// ceil(kMaxSceneDepthM / kAmbig2M): with kAmbig2M ~= 7.495 m > kMaxSceneDepthM,
// this is 1 — freq2 NEVER wraps in this scene (by the ratio-3 design above).
constexpr int kMaxWraps2 = 1;

// ---------------------------------------------------------------------------
// Confidence / validity. kDefaultAmplitudeFloor: the freq1 modulation-
// amplitude threshold below which a pixel's phase (hence depth) is
// UNTRUSTED and masked (THEORY.md "Numerical considerations": atan2's
// conditioning collapses as B -> 0, the identical argument 01.19 makes for
// its own confidence floor). Units: intensity counts on this project's
// [0,255] sensor scale (data/README.md "How the sample was tuned" records
// the sweep that picked this value against the dark cohort / clean-surface
// trade-off, mirroring 01.19's kDefaultConfidenceFloor tuning log).
// ---------------------------------------------------------------------------
constexpr float kDefaultAmplitudeFloor = 18.0f;

// ---------------------------------------------------------------------------
// Flying-pixel detection thresholds (kernels.cu Stage 5 derives the physics;
// data/README.md "How the sample was tuned" records the precision/recall
// sweep that picked these two numbers against the committed scene's
// designed mixed-edge cohort).
//   kFlyingDepthJumpM      — local-neighborhood depth-discontinuity test:
//                            a flying pixel sits inside a neighborhood whose
//                            valid-neighbor depth RANGE (max-min) exceeds
//                            this (m). On its own this ALSO flags plenty of
//                            perfectly clean pixels that merely sit next to
//                            a real depth step (their neighbor set spans the
//                            step) — that over-triggering is deliberate and
//                            expected; the second test below is what narrows
//                            the flag down to the pixel that is ITSELF a
//                            mixed return.
//   kFlyingAmplitudeRatio  — amplitude-ratio test: a mixed-return pixel's
//                            OWN amplitude, divided by the strongest valid
//                            neighbor's amplitude, falls below this ratio
//                            (destructive phasor interference — see
//                            kernels.cuh file header). A clean pixel's
//                            amplitude tracks its own surface's albedo and
//                            is NOT systematically weaker than its
//                            neighbors', even next to a sharp depth step.
// A pixel is flagged FLYING only when BOTH tests fire (kernels.cu Stage 5).
// ---------------------------------------------------------------------------
constexpr float kFlyingDepthJumpM     = 0.30f;
constexpr float kFlyingAmplitudeRatio = 0.60f;
constexpr int   kFlyingMinValidNeighbors = 3;    // need at least this many valid neighbors to judge at all

// Sentinel values (never NaN in artifacts — CLAUDE.md convention, matches
// 01.19's kInvalidColumnF): an out-of-range float / int is loud in CSV/diff
// tooling, NaN silently poisons sums.
constexpr float kInvalidDepthM = -1.0f;          // sentinel: no trustworthy depth at this pixel
constexpr int   kInvalidWrapCount = -1;          // sentinel: no wrap-count decision was made

// Ground-truth surface labels (written by make_synthetic.py into
// truth_surface.bin; used ONLY to score the reconstruction/flying-pixel
// gates in main.cu — never consumed by the decode pipeline itself, exactly
// 01.19's "grading labels are not decoding inputs" rule).
constexpr unsigned char kSurfNone       = 0;       // no analytic surface / outside camera's useful range
constexpr unsigned char kSurfBackground = 1;       // the tilted background plane
constexpr unsigned char kSurfSphere     = 2;       // the sphere
constexpr unsigned char kSurfBox        = 3;       // the box's fronto-parallel top face

// ===========================================================================
// GPU kernel declarations. Fenced behind __CUDACC__ so reference_cpu.cpp
// (compiled by cl.exe, which does not know __global__) can still include
// this header for the shared constants and launcher/CPU-oracle prototypes
// below (same trick documented in the template's kernels.cuh and used
// throughout 01.19).
// ===========================================================================
#ifdef __CUDACC__

// Stage 1 — 4-tap correlation -> phase + amplitude. One thread per camera
// pixel; called ONCE PER FREQUENCY by main.cu (the kernel itself is
// frequency-agnostic — it only ever sees four already-captured tap frames).
__global__ void extract_phase_amplitude_kernel(const float* __restrict__ d_taps,       // [4*n] tap-major
                                               float*       __restrict__ d_phase,      // [n] OUT, radians [0,2pi)
                                               float*       __restrict__ d_amplitude,  // [n] OUT, intensity counts
                                               int n);

// Stage 2 — single-frequency depth: depth = phase/(2*pi) * ambiguity_range.
// One thread per camera pixel. Deliberately trivial (the whole point: this
// is the WRAPPED depth the aliasing_demo gate measures directly).
__global__ void single_freq_depth_kernel(const float* __restrict__ d_phase,   // [n] radians [0,2pi)
                                         float ambiguity_range_m,
                                         float*       __restrict__ d_depth,   // [n] OUT, meters, in [0, ambiguity_range_m)
                                         int n);

// Stage 3 — dual-frequency unwrap: the CRT-style integer-wrap consistency
// search (kernels.cuh "Two frequencies, then unwrapping"). One thread per
// camera pixel.
__global__ void dual_freq_unwrap_kernel(const float* __restrict__ d_phase1,       // [n] fine channel, radians
                                        const float* __restrict__ d_phase2,       // [n] coarse channel, radians
                                        float*         __restrict__ d_depth,      // [n] OUT, meters (unwrapped, fine precision)
                                        int*           __restrict__ d_wrap_count, // [n] OUT, winning n1 in [0,kMaxWraps1)
                                        int n);

// Stage 4 — confidence mask: amplitude-floor threshold. One thread per pixel.
__global__ void confidence_mask_kernel(const float* __restrict__ d_amplitude,  // [n] freq1 amplitude, counts
                                       float amplitude_floor,
                                       unsigned char* __restrict__ d_valid,    // [n] OUT (0/1)
                                       int n);

// Stage 5 — flying-pixel detection: THIS PROJECT'S ONE STENCIL KERNEL. One
// thread per camera pixel, gathering its (up to) 8 immediate neighbors.
__global__ void flying_pixel_detect_kernel(const float*         __restrict__ d_depth,      // [n] unwrapped depth, m
                                           const float*         __restrict__ d_amplitude,  // [n] freq1 amplitude, counts
                                           const unsigned char* __restrict__ d_confidence_valid, // [n] Stage-4 output
                                           unsigned char* __restrict__ d_flying,            // [n] OUT (0/1)
                                           int w, int h);

// Stage 6 — pinhole back-projection to a metric point cloud. One thread per
// camera pixel; final validity = confidence AND NOT flying (main.cu combines
// the two masks before calling this kernel — see main.cu Stage 6).
__global__ void backproject_kernel(const float*         __restrict__ d_depth,        // [n] unwrapped depth, m
                                   const unsigned char* __restrict__ d_final_valid,  // [n] combined mask
                                   float*         __restrict__ d_xyz,                // [n*3] OUT (x,y,z), m
                                   int n);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Host launch wrappers (defined in kernels.cu; callable from any translation
// unit — only their DEFINITIONS need nvcc). Each owns its grid/block math
// and the post-launch CUDA_CHECK_LAST_ERROR call (CLAUDE.md §6.1 rule 7).
// ---------------------------------------------------------------------------
void launch_extract_phase_amplitude(const float* d_taps, float* d_phase, float* d_amplitude, int n);
void launch_single_freq_depth(const float* d_phase, float ambiguity_range_m, float* d_depth, int n);
void launch_dual_freq_unwrap(const float* d_phase1, const float* d_phase2,
                             float* d_depth, int* d_wrap_count, int n);
void launch_confidence_mask(const float* d_amplitude, float amplitude_floor, unsigned char* d_valid, int n);
void launch_flying_pixel_detect(const float* d_depth, const float* d_amplitude,
                                const unsigned char* d_confidence_valid, unsigned char* d_flying, int w, int h);
void launch_backproject(const float* d_depth, const unsigned char* d_final_valid, float* d_xyz, int n);

// ---------------------------------------------------------------------------
// CPU oracle twins (reference_cpu.cpp). Per the template's independence
// ruling: these are INDEPENDENTLY-written plain-C++ implementations of the
// same math, not thin wrappers around shared device code — see
// reference_cpu.cpp's file header for the full ruling and why it matters.
// Signatures mirror the launchers exactly so main.cu calls both uniformly.
// ---------------------------------------------------------------------------
void extract_phase_amplitude_cpu(const float* taps, float* phase, float* amplitude, int n);
void single_freq_depth_cpu(const float* phase, float ambiguity_range_m, float* depth, int n);
void dual_freq_unwrap_cpu(const float* phase1, const float* phase2, float* depth, int* wrap_count, int n);
void confidence_mask_cpu(const float* amplitude, float amplitude_floor, unsigned char* valid, int n);
void flying_pixel_detect_cpu(const float* depth, const float* amplitude, const unsigned char* confidence_valid,
                             unsigned char* flying, int w, int h);
void backproject_cpu(const float* depth, const unsigned char* final_valid, float* xyz, int n);

#endif // PROJECT_KERNELS_CUH
