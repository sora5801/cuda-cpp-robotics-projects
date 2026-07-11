// ===========================================================================
// kernels.cuh — interface & data contract for project 01.24
//               Transparent/reflective object detection via polarization
//               imaging
//
// Role in the project
// --------------------
// The single-sourced contract between main.cu (orchestration + gates),
// kernels.cu (the GPU kernels), reference_cpu.cpp (the independent CPU
// oracle twins) and scripts/make_synthetic.py (the physics forward-model
// scene generator). Every geometry constant, sensor-mosaic convention, and
// scene-object parameter that more than one of those files must agree on
// lives HERE, once (CLAUDE.md §12) — make_synthetic.py mirrors every number
// below with "MUST MATCH kernels.cuh" comments, because Python cannot
// #include a .cuh file.
//
// RATIFIED SCOPE (catalog bullet "Transparent/reflective object detection
// via polarization imaging") — a division-of-focal-plane (DoFP) polarization
// camera (kinship: project 01.23's Bayer-mosaic DoFP-for-color ISP; a DoFP
// polarization sensor is the SAME idea with polarizer angles instead of RGB
// filters) sees a scene with a matte (largely unpolarized) background and
// three specular objects intensity alone cannot separate from that
// background: a flat GLASS PANE, a curved GLASS DOME, and a brushed METAL
// bar. The pipeline: (1) per-angle bilinear DEMOSAIC (01.23 kinship,
// generalized from 3 Bayer phases to 4 polarizer phases); (2) STOKES
// parameter estimation from the 4 recovered angle channels; (3) DEGREE and
// ANGLE of linear polarization (DoLP/AoLP); (4) a FREE self-consistency
// check (the Malus-law residual — 4 measurements, 3 parameters, 1 DOF);
// (5) DoLP-threshold + morphological-open + connected-component detection
// (01.21/01.06 kinship, cited in kernels.cu) — run TWICE, once on DoLP and
// once on plain intensity (S0), to demonstrate the reason this project
// exists: the glass objects are built to have near-zero INTENSITY contrast
// (see make_synthetic.py) so only the polarization channel can find them.
//
// TWIN-INDEPENDENCE RULING applied here (restated from the template and
// project 01.22's kernels.cuh, whose language this project follows
// verbatim): the DATA-LAYOUT contracts below (canvas geometry, the DoFP
// phase/channel mapping, scene-object rectangles/circle, and the
// PhaseSample coordinate-footprint helper) are a shared PROBLEM-DEFINITION
// contract, not an algorithm under test — sharing them is the repo's rule,
// not an exception (duplicating an index formula is not independence, it is
// a second hiding place for the same bug). The ALGORITHMIC core of every
// stage — the demosaic 4-corner weighted blend, the Stokes formulas, the
// DoLP/AoLP formulas, the Malus residual, connected-component labeling
// (GPU sweep-propagation vs. CPU union-find, genuinely different
// algorithms, the 01.21 precedent) — is written TWICE, independently, in
// kernels.cu and reference_cpu.cpp. Independent GATEs in main.cu (never
// routed through either twin) compare against the synthetic ground truth
// and against an independently-coded closed-form Fresnel prediction — see
// main.cu's file header.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>   // sqrtf/atan2f/asinf/sinf/cosf — used by the HD helpers below

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe (the 01.01/
// 01.11/01.22 precedent). Used ONLY for small textbook data-CONTRACT helpers
// (coordinate/footprint arithmetic) — never for the algorithmic cores, which
// this project deliberately writes twice (see the twin-independence ruling
// above).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// clampi / clampf — the one-line clamp used all over this repo (01.21's
// clampi, re-typed fresh; clampf is its float twin, needed for weights).
HD inline int   clampi(int v, int lo, int hi)     { return v < lo ? lo : (v > hi ? hi : v); }
HD inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// kPi — a project-local pi constant (repo convention, e.g. 21.04/35.01):
// MSVC's <cmath> only defines M_PI when _USE_MATH_DEFINES is set before the
// FIRST include of <cmath> anywhere in the translation unit — fragile across
// header include order — so this repo just states the literal once instead.
constexpr float kPi = 3.14159265358979323846f;

// ===========================================================================
// SECTION 1 — canvas geometry. MUST MATCH scripts/make_synthetic.py's
// "MUST MATCH kernels.cuh" Section 1 block.
//
// kW=kH=128: both EVEN (required — the DoFP mosaic tiles in 2x2 super-pixels,
// so an odd dimension would leave a half super-pixel at the border with no
// defined phase). Square is a readability choice only, not a requirement.
// ===========================================================================
constexpr int kW = 128;              // canvas width, px
constexpr int kH = 128;              // canvas height, px
constexpr int kN = kW * kH;          // pixel count (16,384)

// ===========================================================================
// SECTION 2 — the DoFP (division-of-focal-plane) polarization mosaic.
//
// A REAL DoFP polarization sensor (the illustrative real part: Sony
// IMX250MZR-class — PRACTICE.md §2 dates and sources this) etches a linear
// wire-grid polarizer directly over each photosite, four orientations
// arranged in a repeating 2x2 super-pixel — exactly the geometric idea
// project 01.23's Bayer color-filter-array mosaic uses for R/G/G/B, with
// polarizer angles standing in for color filters. Every pixel therefore
// records only ONE of four measurements at its location:
//
//     I(theta) = S0/2 + (S1/2)*cos(2*theta) + (S2/2)*sin(2*theta)     (*)
//
// — Malus's law for a linear polarizer at transmission axis angle `theta`
// viewing partially-polarized light with Stokes parameters (S0,S1,S2)
// (THEORY.md "The math" derives (*) from the polarizer's intensity
// transmission and shows why unpolarized light contributes S0/2 regardless
// of theta). This project's illustrative super-pixel layout (verify against
// a real datasheet before relying on the exact orientation — PRACTICE.md
// §2):
//
//     row y%2==0:   [ 90 deg | 45 deg ]   (x%2==0 | x%2==1)
//     row y%2==1:   [135 deg |  0 deg ]   (x%2==0 | x%2==1)
//
// kChannelAngleDeg indexes the FOUR "virtual full-resolution" channels this
// project reconstructs (channel c holds the theta=kChannelAngleDeg[c]
// measurement, demosaiced to every pixel); dofp_channel_for_phase() and
// dofp_phase_for_channel() are the two directions of the phase<->channel
// map, shared data-contract arithmetic both kernels.cu and
// reference_cpu.cpp index with (never duplicated as two independently
// hand-written tables that could silently disagree).
// ===========================================================================
constexpr int kNumChannels = 4;                                   // 0,45,90,135 deg
constexpr float kChannelAngleDeg[kNumChannels] = { 0.0f, 45.0f, 90.0f, 135.0f };

// dofp_channel_for_phase — given a pixel's own super-pixel phase (px,py in
// {0,1}), return which of the 4 channels (0..3, indexing kChannelAngleDeg)
// that physical pixel actually measures. This IS the mosaic layout drawn
// above, expressed as code instead of prose.
HD inline int dofp_channel_for_phase(int px, int py)
{
    if (px == 0 && py == 0) return 2;   // 90 deg
    if (px == 1 && py == 0) return 1;   // 45 deg
    if (px == 0 && py == 1) return 3;   // 135 deg
    return 0;                            // px==1 && py==1 -> 0 deg
}

// dofp_phase_for_channel — the INVERSE map: which super-pixel phase (out_px,
// out_py, each 0 or 1) carries channel c's own direct measurement. Every
// demosaic call needs this to find "where are MY neighbors of channel c".
HD inline void dofp_phase_for_channel(int c, int& out_px, int& out_py)
{
    switch (c) {
        case 0: out_px = 1; out_py = 1; return;   // 0 deg   at (1,1)
        case 1: out_px = 1; out_py = 0; return;   // 45 deg  at (1,0)
        case 2: out_px = 0; out_py = 0; return;   // 90 deg  at (0,0)
        default: out_px = 0; out_py = 1; return;  // 135 deg at (0,1)
    }
}

// ---------------------------------------------------------------------------
// PhaseSample — the shared FOOTPRINT arithmetic for demosaicing one target
// channel at one pixel: which four same-phase mosaic samples bracket (x,y),
// and the bilinear weights between them. This is the 01.22 "BilinearSample /
// bilinear_sample_at" precedent applied to a phase-subsampled (spacing-2,
// not spacing-1) grid — SHARED as a data-contract coordinate helper because
// a wrong index here is not "independence", it is a second place for the
// same indexing bug to hide (kernels.cuh header, twin-independence ruling).
// The actual 4-corner WEIGHTED BLEND that consumes these weights is written
// independently in kernels.cu's kernel body and reference_cpu.cpp's
// function body — that is where this project's twin independence lives.
//
// nearest_lower_phase_coord/nearest_upper_phase_coord below derive x0/x1 (or
// y0/y1): the phase-c samples along one axis sit on a spacing-2 lattice
// offset by the target phase; we want the pair that BRACKETS the query
// coordinate, clamped to stay in [0,size).
// ---------------------------------------------------------------------------
HD inline int nearest_lower_phase_coord(int v, int phase)
{
    // Coordinates with (v - phase) even carry this phase. If v itself does
    // not, the nearest LOWER one is v-1 (spacing is 2, so the two parities
    // alternate every unit step).
    int v0 = (((v - phase) % 2) == 0) ? v : v - 1;
    if (v0 < 0) v0 += 2;   // border: v=0,phase=1 -> v0=-1 -> shift to the only valid neighbor (=1)
    return v0;
}
HD inline int nearest_upper_phase_coord(int v0, int size)
{
    int v1 = v0 + 2;
    if (v1 > size - 1) v1 = v0;   // no further neighbor at the border -> collapse (weight goes to 0, see below)
    return v1;
}

struct PhaseSample { int x0, y0, x1, y1; float wx, wy; };

HD inline PhaseSample phase_sample_at(int x, int y, int target_px, int target_py, int W, int H)
{
    PhaseSample s;
    s.x0 = nearest_lower_phase_coord(x, target_px);
    s.y0 = nearest_lower_phase_coord(y, target_py);
    s.x1 = nearest_upper_phase_coord(s.x0, W);
    s.y1 = nearest_upper_phase_coord(s.y0, H);
    // A collapsed pair (x1==x0, e.g. right at a border with no further
    // same-phase neighbor) must contribute weight 0 to the "far" corner —
    // otherwise the 4-corner blend would double-count the single available
    // sample. Guarding division by (x1-x0)==0 the same way.
    s.wx = (s.x1 != s.x0) ? clampf(static_cast<float>(x - s.x0) / static_cast<float>(s.x1 - s.x0), 0.0f, 1.0f) : 0.0f;
    s.wy = (s.y1 != s.y0) ? clampf(static_cast<float>(y - s.y0) / static_cast<float>(s.y1 - s.y0), 0.0f, 1.0f) : 0.0f;
    return s;
}

// ===========================================================================
// SECTION 3 — the physics forward model's scene geometry. MUST MATCH
// scripts/make_synthetic.py's Section 3 block EXACTLY (both files render/
// interpret the SAME three objects against the SAME background).
//
// All three objects are chosen so ONE dimension (an incidence angle) drives
// their DoLP via genuinely different physical models (THEORY.md derives
// each): the pane and dome via the REAL Fresnel equations for a dielectric
// (glass, n=1.5); the metal bar via a documented PHENOMENOLOGICAL curve
// (real conductors need complex-refractive-index Fresnel equations — a
// deliberately scoped-out simplification, stated honestly in THEORY.md
// "Where this sits in the real world").
// ===========================================================================
struct Rect { int x0, x1, y0, y1; };   // half-open [x0,x1) x [y0,y1), pixel space

constexpr float kNGlass = 1.5f;             // glass refractive index (typical soda-lime, THEORY.md)

// -- Object 1: flat glass pane. A single incidence angle -> a single DoLP
// over the whole rect (THEORY.md derives DoLP from Rs/Rp via the Fresnel
// equations) — this is the fresnel_anchor gate's target (main.cu): the ONE
// place in this project where "what DoLP should a real glass surface show"
// is checked against the closed-form physics, not just against the
// generator's own (independently-coded) rendering of that same physics.
constexpr Rect kPaneRect{ 14, 54, 14, 82 };      // 40 x 68 px, left side of the canvas
constexpr float kPaneThetaDeg = 35.0f;           // incidence angle, degrees (below Brewster's ~56.3 deg)
constexpr float kPaneAolpDeg = 90.0f;            // vertical s-polarization (plane of incidence is horizontal)

// -- Object 2: curved glass dome (a hemisphere/dome viewed fronto-parallel,
// orthographic approximation). Local incidence angle varies with radial
// distance r from the dome's image-plane center: theta_i(r) = asin(r/R)
// (basic sphere-under-orthographic-view geometry — THEORY.md derives this).
// DoLP therefore RISES from 0 (center, normal incidence) to 1.0 at the
// "Brewster ring" (r/R = sin(56.31 deg) = 0.832) and back toward 0 at the
// silhouette (r=R, grazing) — the real "polarization donut" pattern seen in
// photographs of specular spheres under polarization imaging. AoLP is
// RADIAL + 90 deg (s-polarization is perpendicular to the local plane of
// incidence, which contains the view axis and the radially-tilted normal).
constexpr float kDomeCx = 92.0f;                 // dome center, px (image x)
constexpr float kDomeCy = 40.0f;                 // dome center, px (image y)
constexpr float kDomeRadiusPx = 24.0f;           // dome silhouette radius, px

// -- Object 3: brushed metal bar, curvature in y only (a horizontal
// cylinder's front-facing generator strip). theta_local(y) = asin(dy/R)
// exactly like the dome's radial formula but 1-D. UNLIKE glass, DoLP rises
// MONOTONICALLY with |theta_local| toward a SATURATING ceiling below 1.0
// (real conductors never reach Rp=0 — there is no Brewster ZERO for a
// metal — THEORY.md "Where this sits in the real world" names the complex-
// refractive-index formula this curve stands in for) — the "different
// documented signature" the README promises. AoLP is CONSTANT (0 deg,
// horizontal s-polarization: a horizontal-axis cylinder's local normal
// always lies in the same vertical plane of incidence, wherever you are
// along the curve) — a real, teachable contrast with the dome's RADIALLY
// VARYING AoLP: shape-from-polarization literature uses exactly this
// distinction (constant vs. radial AoLP) to tell cylindrical from
// spherical curvature apart (THEORY.md).
constexpr Rect kMetalRect{ 14, 114, 92, 120 };   // 100 x 28 px, bottom band
constexpr float kMetalDolpMax = 0.55f;           // saturating ceiling (measured curve peak, see THEORY.md)
constexpr float kMetalSat = 0.15f;               // saturation-curve shape constant (sin^2(theta)/(k+sin^2(theta)))
constexpr float kMetalS0Dn = 195.0f;             // metal is DELIBERATELY brighter than background (see Section 4) —
                                                  // unlike glass, a shiny metal part IS usually visible in plain
                                                  // intensity too; only the GLASS objects are built to be invisible.

// -- Background (everywhere not covered by an object): a smooth, LOW
// residual-DoLP matte surface (no real matte surface depolarizes light
// PERFECTLY — a few percent residual DoLP from micro-facet Fresnel
// reflections is realistic, THEORY.md) plus a gentle horizontal brightness
// gradient (uneven lighting, a realistic nuisance a real background-
// subtraction detector must tolerate — PRACTICE.md §3 "exposure discipline").
constexpr float kBgS0Base = 130.0f;              // DN, canvas-center brightness
constexpr float kBgS0GradAmpX = 12.0f;           // DN, +/- half-amplitude of the horizontal gradient
constexpr float kBgDolp = 0.018f;                // residual DoLP, matte background (1.8%)
constexpr float kBgAolpDeg = 45.0f;              // arbitrary but fixed (irrelevant at this tiny a DoLP)

constexpr float kNoiseStdDn = 2.2f;              // additive per-channel sensor noise, DN rms (both scenes)

// ===========================================================================
// SECTION 4 — detection pipeline constants. MUST MATCH
// scripts/make_synthetic.py's Section 4 block (the generator reports these
// numbers in params.csv so a learner can see, offline, what the demo will
// threshold at).
// ===========================================================================
constexpr float kDolpThreshold = 0.10f;          // DoLP foreground threshold (background ~0.02-0.05 incl. noise)
constexpr float kIntensityThreshold = 25.0f;     // |S0 - mean(S0)| DN threshold for the intensity-only baseline
constexpr int   kMinComponentSizePx = 40;        // post-morphology connected-component size floor
constexpr int   kMaxCclSweeps = 256;             // convergence safety cap (01.21 precedent: max(kW,kH)=128 worst case)

// Interior erosion margin for the accuracy gates: demosaic/threshold pixels
// within this many px of an object's true boundary are excluded from
// stokes_accuracy's MAE (edge blur from bilinear demosaic and from
// morphological opening is an honestly-documented artifact, not a bug —
// THEORY.md "Numerical considerations").
constexpr int kInteriorMarginPx = 3;
// AoLP is only meaningful where the signal is strong (THEORY.md "Numerical
// considerations" derives why AoLP is ill-conditioned as DoLP -> 0); the
// aolp accuracy gate restricts to truth-DoLP above this floor.
constexpr float kHighDolpFloorForAolpGate = 0.15f;

// Truth-label convention written by make_synthetic.py into truth_maps.csv
// and read back by main.cu — one small integer per pixel identifying which
// object (if any) generated that pixel, used by every gate that needs to
// know "which pixels are glass" / "which are metal" / "which are neither".
constexpr int kLabelBackground = 0;
constexpr int kLabelPane = 1;
constexpr int kLabelDome = 2;
constexpr int kLabelMetal = 3;

// ===========================================================================
// SECTION 5 — Fresnel physics helper (SHARED data-contract arithmetic: the
// textbook equations themselves, not an "algorithm" with a design choice to
// verify — like Section 2's PhaseSample, duplicating the equations would
// only create two places for an algebra slip to hide). main.cu's
// fresnel_anchor and brewster_sweep gates call this from C++; the
// PYTHON generator implements the SAME equations independently, in a
// different language, as its own ground-truth renderer — THAT cross-
// language duplication is the genuine independence this project's central
// physics claim rests on ("the generator and the analyzer meet at the
// physics", README "Expected output").
//
// fresnel_dolp — the degree of linear polarization of light SPECULARLY
// reflected off a dielectric interface (refractive index `n`, medium 1 is
// air, n1=1) at incidence angle `theta_i_rad`, assuming UNPOLARIZED
// incident illumination (THEORY.md "The math" derives this from the
// Fresnel equations): Rs/Rp are the power reflectances for the two
// polarization components; DoLP = (Rs-Rp)/(Rs+Rp) is always in [0,1] since
// Rs >= Rp >= 0 for external reflection at every angle in [0,90) deg.
// ===========================================================================
HD inline void fresnel_reflectances(float theta_i_rad, float n, float& Rs, float& Rp)
{
    const float cos_i = cosf(theta_i_rad);
    // Snell's law: sin(theta_t) = sin(theta_i) / n (n1=1, going INTO the
    // denser medium). clampf guards a hair of float drift at theta_i=90 deg.
    const float sin_t = clampf(sinf(theta_i_rad) / n, -1.0f, 1.0f);
    const float cos_t = sqrtf(1.0f - sin_t * sin_t);
    const float rs = (cos_i - n * cos_t) / (cos_i + n * cos_t);
    const float rp = (n * cos_i - cos_t) / (n * cos_i + cos_t);
    Rs = rs * rs;
    Rp = rp * rp;
}
HD inline float fresnel_dolp(float theta_i_rad, float n)
{
    float Rs, Rp;
    fresnel_reflectances(theta_i_rad, n, Rs, Rp);
    const float denom = Rs + Rp;
    return (denom > 1.0e-8f) ? (Rs - Rp) / denom : 0.0f;   // denom==0 only at a degenerate n; guarded, never hit here
}

// ===========================================================================
// SECTION 6 — device-only kernel declarations (nvcc only; fenced so
// reference_cpu.cpp, compiled by cl.exe, never sees a __global__ signature).
// Full documentation (thread mapping, memory spaces, numerics) sits with
// each DEFINITION in kernels.cu; one-line summaries here.
// ===========================================================================
#ifdef __CUDACC__

// -- Stage 1: demosaic. One thread per OUTPUT pixel, writes all 4 channels
// (a pure map with a 4-wide output; own channel is a direct copy, the other
// 3 are phase-bilinear interpolated — see kernels.cu).
__global__ void demosaic_polarization_kernel(const float* __restrict__ mosaic,
                                             float* __restrict__ channels4, // [kN*4], interleaved per pixel
                                             int W, int H);

// -- Stage 2: Stokes parameters from the 4 demosaiced channels. Pure map.
__global__ void stokes_kernel(const float* __restrict__ channels4,
                              float* __restrict__ s0, float* __restrict__ s1, float* __restrict__ s2, int n);

// -- Stage 3: DoLP / AoLP from Stokes parameters. Pure map.
__global__ void dolp_aolp_kernel(const float* __restrict__ s0, const float* __restrict__ s1,
                                 const float* __restrict__ s2,
                                 float* __restrict__ dolp, float* __restrict__ aolp_rad, int n);

// -- Stage 4: the free Malus self-consistency residual. Pure map.
__global__ void malus_residual_kernel(const float* __restrict__ channels4, float* __restrict__ residual, int n);

// -- Stage 5 support: |signal - ref_scalar| (builds the intensity-contrast
// detection signal from S0 and its image-mean). Pure map.
__global__ void abs_diff_scalar_kernel(const float* __restrict__ signal, float ref_scalar,
                                       float* __restrict__ out, int n);

// -- Stage 5: threshold -> binary mask. Pure map.
__global__ void threshold_kernel(const float* __restrict__ signal, float thresh,
                                 uint8_t* __restrict__ mask_out, int n);

// -- Stage 5: 3x3 binary erode / dilate (morphological open = erode then
// dilate — 01.21's cited pattern, re-typed fresh here).
__global__ void erode3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out);
__global__ void dilate3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out);

// -- Stage 5: connected-component labeling by iterative label propagation
// (01.21/01.06's cited pattern) + size-filtered output mask.
__global__ void ccl_init_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label, int n);
__global__ void ccl_propagate_sweep_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label,
                                           int W, int H, int* __restrict__ changed);
__global__ void component_size_count_kernel(const uint8_t* __restrict__ mask, const int* __restrict__ label,
                                            int* __restrict__ size_out, int n);
__global__ void component_filter_kernel(const uint8_t* __restrict__ mask_in, const int* __restrict__ label,
                                        const int* __restrict__ size, int min_size,
                                        uint8_t* __restrict__ mask_out, int n);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// SECTION 7 — host-callable launch wrappers (every translation unit sees
// these; only their DEFINITIONS in kernels.cu require nvcc). Each owns its
// grid/block math and the mandatory post-launch error check.
// ===========================================================================
void launch_demosaic_polarization(const float* d_mosaic, float* d_channels4, int W, int H);
void launch_stokes(const float* d_channels4, float* d_s0, float* d_s1, float* d_s2, int n);
void launch_dolp_aolp(const float* d_s0, const float* d_s1, const float* d_s2,
                      float* d_dolp, float* d_aolp_rad, int n);
void launch_malus_residual(const float* d_channels4, float* d_residual, int n);
void launch_abs_diff_scalar(const float* d_signal, float ref_scalar, float* d_out, int n);
void launch_threshold(const float* d_signal, float thresh, uint8_t* d_mask_out, int n);
void launch_morphological_open(uint8_t* d_mask_inout, int W, int H);
// launch_connected_components — runs ccl_init then repeated propagate
// sweeps (host-side convergence loop, mirroring main.cu's own copy of this
// loop for the CPU path) until no pixel's label changes or kMaxCclSweeps is
// hit; returns the number of sweeps actually used (an [info] diagnostic).
int  launch_connected_components(const uint8_t* d_mask, int* d_label, int W, int H);
void launch_component_size_filter(const uint8_t* d_mask_in, const int* d_label, int min_size_px,
                                  uint8_t* d_mask_out, int n);

// ===========================================================================
// SECTION 8 — the CPU reference oracle (reference_cpu.cpp). Declared here so
// main.cu and reference_cpu.cpp agree on every signature at COMPILE time.
// ===========================================================================
void demosaic_polarization_cpu(const float* mosaic, float* channels4, int W, int H);
void stokes_cpu(const float* channels4, float* s0, float* s1, float* s2, int n);
void dolp_aolp_cpu(const float* s0, const float* s1, const float* s2, float* dolp, float* aolp_rad, int n);
void malus_residual_cpu(const float* channels4, float* residual, int n);
void abs_diff_scalar_cpu(const float* signal, float ref_scalar, float* out, int n);
void threshold_cpu(const float* signal, float thresh, uint8_t* mask_out, int n);
void morphological_open_cpu(uint8_t* mask_inout, int W, int H);
void connected_components_cpu(const uint8_t* mask, int* label, int W, int H);
void component_size_filter_cpu(const uint8_t* mask_in, const int* label, int min_size_px,
                               uint8_t* mask_out, int n);

#endif // PROJECT_KERNELS_CUH
