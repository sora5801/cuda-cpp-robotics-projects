// ===========================================================================
// kernels.cuh — interface & sensor-model contract for project 20.01
//               GelSight/DIGIT processing: contact patch, shear field via
//               optical flow, slip detection in real time
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver AND the synthetic-sensor
// renderer — see its file header for why rendering lives there), kernels.cu
// (the GPU pipeline), and reference_cpu.cpp (the line-by-line CPU oracle).
// Every constant that both the renderer and the pipeline must agree on —
// image size, the marker grid, the contact physics, the detection/tracking
// thresholds — is defined HERE, ONCE (CLAUDE.md §12), so main.cu's ground
// truth and kernels.cu's measurements are provably talking about the same
// sensor and the same scene.
//
// The pipeline in five lines (THEORY.md derives every step):
//   1. CONTACT PATCH: |frame - baseline| > threshold, then a binary
//      morphological OPEN (erode -> dilate) to kill speckle -> patch area
//      + centroid via a reduction.
//   2. MARKER DETECTION: for each of the gel's printed marker dots (known
//      rest positions — see "MARKER GRID" below), search a small window of
//      THIS frame for the darkest pixel (a local-minimum / blob search).
//   3. TRACKING: displacement = detected position - rest position; a
//      per-marker validity check (was the window's minimum dark enough to
//      be a real marker, not background?) and an "is this marker inside the
//      current contact patch?" flag sampled straight from the mask.
//   4. SLIP SCORE (host, small — see main.cu): fit ONE rigid 2-D transform
//      (rotation + translation) to every in-contact marker's displacement;
//      the fraction whose RESIDUAL from that fit exceeds a threshold is the
//      slip score.
//   5. DECLARE SLIP when the slip score crosses a documented bound.
// Steps 1-3 are the GPU's job (one thread per pixel / per marker — the
// classic map and small-N-search patterns); step 4 is deliberately kept on
// the host (same call this repo made for MPPI's softmin blend in 08.01 —
// O(num markers) trivial arithmetic that would gain nothing from a kernel).
//
// ===========================================================================
// THE SYNTHETIC SENSOR MODEL (every number below is part of the taught,
// fixed scene — main.cu's renderer and every ground-truth gate read these,
// not ad hoc literals; see THEORY.md "The problem" for the physics that
// justifies each formula this header only STATES).
// ===========================================================================
//
// IMAGE LAYOUT — row-major uint8 grayscale, pixel (x,y) at index y*W+x, x
// rightward, y downward (the repo's universal image convention — matches
// 01.02/07.09). W=320, H=240: small enough for a sub-millisecond pipeline,
// large enough that the marker grid below has real spatial resolution.
//
// MARKER GRID — GelSight/DIGIT sensors print a regular grid of dark dots on
// the gel's inner surface (THEORY.md explains why: dense dot markers are
// what actually make a *shear field* measurable at all). We lay out
// kMarkerNx x kMarkerNy = kNumMarkers dots on a fixed lattice, spacing
// kMarkerSpacingPx, inset kMarkerMarginPx from every edge. INVARIANT (relied
// on by kernels.cu and reference_cpu.cpp to skip bounds-checking in the
// common case, and asserted once in main.cu at startup): margin > search
// radius, so a marker's detection SEARCH WINDOW never needs clamping at the
// image border. Marker i (i = row*kMarkerNx + col) has REST position
//     rest.x = kMarkerMarginPx + col*kMarkerSpacingPx
//     rest.y = kMarkerMarginPx + row*kMarkerSpacingPx
// "Rest" = the undeformed position with no contact anywhere — the reference
// every displacement in this project is measured against (not frame-to-
// frame, which would drift; THEORY.md "The algorithm" explains the choice).
//
// CONTACT PHYSICS — a rigid sphere (radius kSphereRadiusMm) presses into the
// flat gel by up to kIndentDepthMaxMm. Hertzian small-deflection contact
// theory (Johnson, *Contact Mechanics*, 1985, ch. 3) gives the contact
// radius a = sqrt(R*delta) for mutual approach delta — the TRUE, mechanical
// footprint radius; THEORY.md derives it and also derives the *visible*
// (thresholded) radius, which is smaller because our shading model's
// darkening genuinely reaches zero exactly at r=a (see below) — an honest,
// documented measurement bias the ground-truth gate corrects for rather
// than hides.
//
// SHADING MODEL (the "intensity-proxy indentation" this project's README
// names as its v1 scope) — within the contact radius, local depth follows a
// paraboloid consistent with the Hertz boundary (depth=delta at the center,
// 0 at r=a):
//     depth(r) = delta * (1 - (r/a)^2),   r <= a
// and pixel intensity darkens LINEARLY with local depth,
//     darkening(r) = kShadeGainPerMm * depth(r)     [gray levels]
// This is a stand-in for real photometric-stereo reconstruction (three
// colored lights + gradient integration -> a full depth map, what
// production GelSight/DIGIT actually do) — deliberately simplified here to
// one scalar "how indented is this pixel," documented, not hidden
// (THEORY.md "Where this sits in the real world").
//
// SHEAR / STICK-SLIP MODEL — the object then translates by kShearTotalPx
// (a rigid, KNOWN commanded motion) while pressed at constant depth. Every
// marker under contact would move exactly with that command if friction
// never broke ("full stick"). It does not: this project uses the
// Cattaneo-Mindlin partial-slip picture (Johnson ch. 7) — a central STICK
// zone of radius c <= a stays glued to the object; the ANNULUS c < r <= a
// slips, its motion reduced toward a small residual (kStickResidualFrac,
// kinetic friction still dragging it a little). The stick radius shrinks
// with a normalized load fraction s in [0,1] (Cattaneo-Mindlin's T/(mu*N)):
//     c(s) = a * (1 - s)^(1/3)
// s ramps 0 -> 1 over the SLIP phase (THEORY.md "The math" derives both
// formulas and states plainly where this project's use of them departs from
// the full time-dependent friction physics).
//
// SEQUENCE / PHASES — one fixed, deterministic 100-frame demo run:
//     BASELINE (kNBaseline)  : no contact at all — the reference frame AND
//                               a true-negative test for the contact kernel.
//     PRESS    (kNPress)     : indentation ramps 0 -> kIndentDepthMaxMm,
//                               then holds. No shear yet (commanded shear is
//                               0 throughout, so every marker's modeled
//                               displacement is 0 automatically — no special
//                               casing needed anywhere in the pipeline).
//     SHEAR    (kNShear)     : the object translates by kShearTotalPx
//                               (ramps, then holds); contact stays fully
//                               stuck throughout (s=0) by construction.
//     SLIP     (kNSlip)      : object position HELD at kShearTotalPx; the
//                               load fraction s ramps 0->1, so the stick
//                               radius c(s) shrinks and the annulus grows.
// kBaselineStart/kPressStart/kShearStart/kSlipStart are the GLOBAL frame
// index each phase begins at — the single source every gate in main.cu
// reads instead of re-deriving phase boundaries ad hoc.
//
// Read this after: main.cu (the renderer + orchestration).
// Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>

// ---------------------------------------------------------------------------
// Image geometry.
// ---------------------------------------------------------------------------
constexpr int kImgW = 320;          // px
constexpr int kImgH = 240;          // px
constexpr float kPxPerMm = 20.0f;   // pixel-space <-> physical gel-surface scale (px/mm)

// ---------------------------------------------------------------------------
// Marker grid (rest / undeformed positions — see file header).
// ---------------------------------------------------------------------------
constexpr int kMarkerSpacingPx = 18;   // lattice spacing (px)
constexpr int kMarkerMarginPx  = 9;    // inset from every edge (px) — MUST exceed kSearchRadiusPx
constexpr int kMarkerNx = (kImgW - 2 * kMarkerMarginPx) / kMarkerSpacingPx + 1;  // = 17
constexpr int kMarkerNy = (kImgH - 2 * kMarkerMarginPx) / kMarkerSpacingPx + 1;  // = 13
constexpr int kNumMarkers = kMarkerNx * kMarkerNy;                              // = 221

// ---------------------------------------------------------------------------
// Gel appearance (baseline gray levels + the printed marker dots).
// ---------------------------------------------------------------------------
constexpr int kGelBaselineGray   = 180;   // undeformed gel surface, no contact (0..255)
constexpr int kMarkerDarkGray    = 60;    // marker-dot center gray level (dark ink on the gel)
constexpr float kMarkerRadiusPx  = 4.0f;  // visible marker-dot radius (px)
constexpr int kTextureNoiseAmplitude = 4; // +/- gray-level deterministic per-pixel texture noise

// ---------------------------------------------------------------------------
// Contact / indentation physics (Hertzian sphere; see file header + THEORY.md).
// ---------------------------------------------------------------------------
constexpr float kSphereRadiusMm    = 5.0f;   // indenter sphere radius R (mm)
constexpr float kIndentDepthMaxMm  = 1.2f;   // max mutual-approach depth delta (mm)
constexpr float kShadeGainPerMm    = 70.0f;  // shading darkening per mm of local depth at r=0 (gray/mm)
constexpr int kContactMaskThreshold = 8;     // |frame-baseline| >= this => "in contact" candidate pixel (gray levels)

// ---------------------------------------------------------------------------
// Shear / stick-slip (Cattaneo-Mindlin partial-slip annulus; file header).
// ---------------------------------------------------------------------------
constexpr float kShearTotalPx        = 5.0f;   // commanded rigid shear translation, held during SLIP (px, +x)
constexpr float kStickResidualFrac   = 0.15f;  // residual motion fraction left in the slipping annulus (unitless)

// ---------------------------------------------------------------------------
// Marker detection / tracking.
// ---------------------------------------------------------------------------
constexpr int kSearchRadiusPx        = 8;    // half-width of each marker's per-frame search window (px)
constexpr int kMarkerDetectThreshold = 140;  // window minimum must be darker than this to count as "found" (gray level)

// ---------------------------------------------------------------------------
// Slip declaration.
// ---------------------------------------------------------------------------
constexpr float kResidualSlipThresholdPx  = 1.1f;  // |measured - rigid-fit prediction| beyond this counts as "slipping" (px)
constexpr float kSlipScoreDeclareThreshold = 0.5f; // slip_score >= this => declare SLIP for that frame (unitless fraction)

// ---------------------------------------------------------------------------
// Sequence / phases (see file header for the physical story of each).
// ---------------------------------------------------------------------------
constexpr int kNBaseline  = 6;
constexpr int kNPressRamp = 16;
constexpr int kNPressHold = 8;
constexpr int kNPress     = kNPressRamp + kNPressHold;   // 24
constexpr int kNShearRamp = 18;
constexpr int kNShearHold = 12;
constexpr int kNShear     = kNShearRamp + kNShearHold;   // 30
constexpr int kNSlip      = 40;
constexpr int kNumFrames  = kNBaseline + kNPress + kNShear + kNSlip;  // 100

constexpr int kBaselineStart = 0;
constexpr int kPressStart    = kBaselineStart + kNBaseline;  // 6
constexpr int kShearStart    = kPressStart + kNPress;        // 30
constexpr int kSlipStart     = kShearStart + kNShear;        // 60

// Rest-frame indenter/contact center (px) — the sphere presses straight down
// here before any shear; THEORY.md "the math" gives the closed forms for
// contact radius a(t) and stick radius c(t) that main.cu evaluates per frame.
constexpr float kContactCenterX = 160.0f;
constexpr float kContactCenterY = 120.0f;

// ---------------------------------------------------------------------------
// Vec2f — a 2-D pixel-space point/vector (px). Used for marker rest/detected
// positions and displacements throughout the pipeline. Plain aggregate (no
// constructor) so it is trivially usable from BOTH nvcc device code and
// plain host C++ (reference_cpu.cpp) with the same layout.
// ---------------------------------------------------------------------------
struct Vec2f {
    float x;  // px, +right
    float y;  // px, +down (image convention)
};

// ===========================================================================
// GPU launchers (defined in kernels.cu). Every pointer prefixed d_ is a
// DEVICE pointer the caller allocated; every launcher owns its own grid/block
// math and the mandatory post-launch error check (CLAUDE.md §6.1 rule 7).
// ===========================================================================

// launch_contact_mask — |frame - baseline| >= threshold -> mask=255, else 0.
//   d_frame, d_baseline : DEVICE, W*H uint8 (current frame, no-contact reference).
//   d_mask               : DEVICE, W*H uint8 OUT — RAW (pre-morphology) mask.
// Launch: one thread per pixel (map).
void launch_contact_mask(const unsigned char* d_frame, const unsigned char* d_baseline,
                          unsigned char* d_mask, int W, int H, int threshold);

// launch_erode3 / launch_dilate3 — binary morphological erosion / dilation
// with a 3x3 (8-connected) structuring element. Composed erode-then-dilate
// by the CALLER = a morphological OPEN (kills small speckle without eating
// the patch's real interior — THEORY.md "The algorithm" derives why open,
// not close, is the right operator for THIS noise). Out-of-bounds neighbors
// read as 0 ("not in contact") — the safe assumption for both operators.
//   d_in, d_out : DEVICE, W*H uint8 (0 or 255). Must NOT alias.
// Launch: one thread per pixel (stencil, radius 1).
void launch_erode3(const unsigned char* d_in, unsigned char* d_out, int W, int H);
void launch_dilate3(const unsigned char* d_in, unsigned char* d_out, int W, int H);

// launch_patch_stats — contact-patch area and (unnormalized) centroid sums.
//   d_mask                     : DEVICE, W*H uint8 (0 or 255; the FINAL,
//                                 opened mask).
//   d_area, d_sumx, d_sumy     : DEVICE, single unsigned long long each,
//                                 OUT — accumulated via atomicAdd. Caller
//                                 MUST cudaMemset all three to 0 before this
//                                 call (this function only adds).
// Launch: one thread per pixel (map + atomic reduction — see kernels.cu for
// why a naive per-thread atomicAdd, not a shared-memory partial reduction,
// is the honest choice at this image size; THEORY.md "The GPU mapping").
void launch_patch_stats(const unsigned char* d_mask, int W, int H,
                         unsigned long long* d_area,
                         unsigned long long* d_sumx,
                         unsigned long long* d_sumy);

// launch_detect_markers — per-marker local-minimum search (THEORY.md "The
// algorithm" explains why search-near-rest beats whole-image blob detection
// here).
//   d_frame       : DEVICE, W*H uint8 — the current frame.
//   d_rest_pos    : DEVICE, num_markers Vec2f — each marker's REST position.
//   d_detected_pos: DEVICE, num_markers Vec2f OUT — the darkest pixel found
//                    in marker i's search window this frame (integer-valued
//                    coordinates stored as float; see README "Exercises" for
//                    sub-pixel refinement).
//   d_min_intensity: DEVICE, num_markers int OUT — that pixel's gray value
//                    (0..255), consumed by launch_track_markers's validity
//                    check.
// Launch: one thread per marker (kNumMarkers threads total — see kernels.cu
// for why this is the honest, brute-force-appropriate mapping at this N).
void launch_detect_markers(const unsigned char* d_frame, const Vec2f* d_rest_pos,
                            int num_markers, int W, int H, int search_radius,
                            Vec2f* d_detected_pos, int* d_min_intensity);

// launch_track_markers — displacement, validity, and contact-patch
// membership per marker.
//   d_detected_pos, d_min_intensity, d_rest_pos : as produced above.
//   d_mask         : DEVICE, W*H uint8 — the FINAL opened contact mask
//                    (sampled at each marker's REST pixel to decide
//                    d_in_contact — see THEORY.md for why rest, not
//                    detected, position is the right sample point).
//   d_displacement : DEVICE, num_markers Vec2f OUT — detected - rest (px).
//   d_valid        : DEVICE, num_markers uint8 OUT — 1 if the window's
//                    minimum was dark enough (< detect_threshold) to trust,
//                    else 0 (no marker found near this rest position).
//   d_in_contact   : DEVICE, num_markers uint8 OUT — 1 if this marker's REST
//                    pixel lies inside the current contact mask.
// Launch: one thread per marker.
void launch_track_markers(const Vec2f* d_detected_pos, const int* d_min_intensity,
                           const Vec2f* d_rest_pos, const unsigned char* d_mask,
                           int num_markers, int W, int H, int detect_threshold,
                           Vec2f* d_displacement, unsigned char* d_valid,
                           unsigned char* d_in_contact);

// ===========================================================================
// CPU references (reference_cpu.cpp) — line-by-line twins of every launcher
// above, HOST pointers, "_cpu" suffix. main.cu's VERIFY stage runs BOTH
// paths on EVERY frame of the 100-frame sequence and requires EXACT
// equality (every operation here is integer/threshold arithmetic on a
// shared uint8 input frame — no floating-point rounding anywhere in this
// list, so there is no tolerance to justify; THEORY.md "How we verify
// correctness").
// ===========================================================================
void contact_mask_cpu(const unsigned char* frame, const unsigned char* baseline,
                      unsigned char* mask, int W, int H, int threshold);
void erode3_cpu(const unsigned char* in, unsigned char* out, int W, int H);
void dilate3_cpu(const unsigned char* in, unsigned char* out, int W, int H);
void patch_stats_cpu(const unsigned char* mask, int W, int H,
                     unsigned long long* area,
                     unsigned long long* sumx,
                     unsigned long long* sumy);
void detect_markers_cpu(const unsigned char* frame, const Vec2f* rest_pos,
                        int num_markers, int W, int H, int search_radius,
                        Vec2f* detected_pos, int* min_intensity);
void track_markers_cpu(const Vec2f* detected_pos, const int* min_intensity,
                       const Vec2f* rest_pos, const unsigned char* mask,
                       int num_markers, int W, int H, int detect_threshold,
                       Vec2f* displacement, unsigned char* valid,
                       unsigned char* in_contact);

#endif // PROJECT_KERNELS_CUH
