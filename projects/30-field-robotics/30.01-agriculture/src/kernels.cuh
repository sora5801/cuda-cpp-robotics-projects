// ===========================================================================
// kernels.cuh — interface for project 30.01
//               Agriculture, Milestone 1: fruit detection + 3-D localization
//               + ripeness on synthetic orchard RGB-D imagery
//               (BUNDLED PROJECT — see README "Overview" for the six other
//               documented-only milestones: weed/crop segmentation, spray
//               targeting, row following, canopy volume, under-canopy nav,
//               yield mapping)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver + verification + ground-truth
// gates), kernels.cu (the GPU pipeline), and reference_cpu.cpp (the CPU
// oracle). Every layout, threshold, and sentinel all three must agree on is
// defined HERE, once (CLAUDE.md section 12).
//
// The pipeline in seven lines (THEORY.md derives every step):
//   1. RGB -> HSV per pixel: separate a fruit's COLOR (hue) from lighting
//      (value) — the whole reason this classical pipeline can work at all.
//   2. Per-pixel fruit-likelihood MASK: hue/saturation/value gates.
//   3. Morphological OPENING (erode, dilate): remove small false-positive
//      speckle without eating real fruit blobs (cross-ref 20.01's contact
//      mask cleanup — same operation, different sensor).
//   4. CONNECTED-COMPONENT LABELING: group mask pixels into per-fruit blobs
//      via iterative LABEL PROPAGATION (the ratified teaching CCL algorithm
//      — cross-ref 02.04's Euclidean clustering via GPU union-find for the
//      point-cloud sibling of this exact problem).
//   5. PER-COMPONENT STATISTICS: pixel count, bounding box, centroid, mean
//      hue, mean depth — all via atomics keyed by each pixel's canonical
//      label.
//   6. 3-D LOCALIZATION: back-project the pixel centroid through the pinhole
//      camera model using a ROBUST (inlier-band) depth estimate.
//   7. RIPENESS: map each component's mean hue to a ripeness scalar.
//
// IMAGE LAYOUT — row-major, pixel (x, y) at linear index i = y*W + x, x
// rightward, y downward (image convention, matches every PGM/PPM in this
// repo). Camera looks down +Z in the OPTICAL convention (x-right, y-down,
// z-forward) — SYSTEM_DESIGN.md section 3.2's documented exception to the
// repo's default x-forward/y-left/z-up body frame; every function below that
// touches 3-D points states this explicitly.
//
//   RGB   image: unsigned char, [H*W*3], interleaved: rgb[(y*W+x)*3 + c],
//                c = 0 (R), 1 (G), 2 (B), each 0..255.
//   Depth image: float, [H*W], METERS (converted on load from the committed
//                16-bit-millimeter PGM — see main.cu's loader). A depth of
//                0.0f never legitimately occurs in this scene (the
//                background plane sits at 4.2-5.0 m) so it is never treated
//                as a sentinel; every pixel has a real depth reading.
//   HSV   image: three parallel float arrays h[], s[], v[], each [H*W].
//                h in DEGREES [0,360), s and v in [0,1] — the standard
//                cylindrical-coordinate convention (THEORY.md derives the
//                conversion from RGB).
//   Mask       : unsigned char, [H*W], 0 or 1 ("fruit-likely" per pixel).
//   Label      : int, [H*W]. kLabelNone (-1) = background/non-fruit pixel,
//                NEVER touched by CCL. A foreground pixel p is initialized
//                to label[p] = p (ITS OWN LINEAR INDEX) and, after the
//                propagation kernel converges, holds the CANONICAL label of
//                its connected component: label[p] == the SMALLEST linear
//                index of any pixel reachable from p via 4-connected
//                foreground neighbors. This choice (rather than an arbitrary
//                "next free integer" counter) is what makes the CPU/GPU
//                cross-check in main.cu an EXACT integer comparison instead
//                of a tolerance — see "verification" below and THEORY.md
//                "How we verify correctness".
//
// PER-COMPONENT STATISTIC ARRAYS — the deliberate design choice this project
// teaches (THEORY.md "The GPU mapping" has the full argument): rather than
// compacting labels into 0..K-1 on the GPU (a stream-compaction kernel that
// would teach a DIFFERENT lesson), every comp_* array below is DENSE, sized
// [H*W], and indexed DIRECTLY by the raw canonical label value (which is
// always a valid pixel linear index by construction). Only the <= a few
// dozen slots at canonical-root indices are ever written; the rest sit at
// their initial value and are simply never read. main.cu's host-side
// extraction step (one O(H*W) scan: "is pixel p a canonical root, i.e.
// mask[p] && label[p]==p?") is the ONLY place compaction happens — cheap,
// once, on the host, mirroring 08.01's "GPU does the O(H*W) pixel work, host
// does the tiny O(#components) bookkeeping" division of labor.
//
// ROBUST DEPTH ESTIMATION — two-pass per component (THEORY.md "The math"
// derives the error budget this feeds):
//   pass 1: mean depth over ALL of the component's pixels (comp_mean_depth).
//   pass 2: re-average using only pixels within +/- kInlierSigmaMul standard
//           deviations of that mean, where the "standard deviation" is the
//           SENSOR'S OWN documented noise model sigma_z(Z) = kDepthNoiseK *
//           Z^2 (identical formula to scripts/make_synthetic.py's forward
//           noise injection — a realistic assumption: a fielded system
//           calibrates its depth sensor's noise curve once and reuses it,
//           exactly as this pipeline does). This trims occasional
//           mixed-depth pixels at a blob's silhouette edge (foliage/fruit
//           mixing at the boundary, or a neighboring occluder's depth
//           bleeding in) without the cost of a true sorting-based median.
//
// VERIFICATION STRATEGY (main.cu; full argument in THEORY.md):
//   HSV, mask       : tolerance-based (H/S/V are plain arithmetic — no
//                     trig — so GPU/CPU divergence is sub-ULP; a tiny
//                     tolerance absorbs FP order-of-operations differences).
//   CCL labels      : EXACT integer equality. Label propagation's fixed
//                     point is UNIQUE regardless of update schedule (it is a
//                     bounded, monotonically-decreasing relaxation — the
//                     same argument that proves Bellman-Ford correct with
//                     all-zero edge weights); the CPU oracle uses a
//                     DIFFERENT algorithm (union-find) and then
//                     CANONICALIZES its labels to the same "min linear
//                     index per component" convention, so both sides must
//                     land on the identical label image if either is
//                     correct — a strong, cheap correctness check.
//   Detection stats : small relative tolerance (sums accumulate in a
//                     different ORDER on GPU atomics vs. CPU sequential
//                     loops, so bit-exactness is not expected — CLAUDE.md's
//                     usual floating-point honesty).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>

// ---------------------------------------------------------------------------
// Camera & scene constants — MUST match scripts/make_synthetic.py's module
// constants of the same name (that script's header states the reverse
// cross-reference). Single source of truth for the pinhole intrinsics every
// back-projection in this project uses.
// ---------------------------------------------------------------------------
constexpr int   kImageWidth  = 640;     // px
constexpr int   kImageHeight = 480;     // px
constexpr float kFx = 525.0f;           // px — focal length, x (Kinect-v1/TUM-RGBD's documented value)
constexpr float kFy = 525.0f;           // px — focal length, y (== kFx: square pixels, no skew)
constexpr float kCx = 320.0f;           // px — principal point x (image center)
constexpr float kCy = 240.0f;           // px — principal point y (image center)

// Depth-sensor noise model: sigma_z(Z) = kDepthNoiseK * Z^2 (meters), a
// simplified structured-light quadratic-in-range noise curve (loosely after
// Khoshelham & Elberink 2012's Kinect-v1 characterization). Used TWICE: by
// the synthetic generator (forward, to CORRUPT depth) and by this pipeline's
// robust estimator (inverse, to REJECT outliers) — see the file header.
constexpr float kDepthNoiseK = 0.0015f; // 1/m — must match make_synthetic.py

// ---------------------------------------------------------------------------
// Fruit-mask HSV gates (THEORY.md "The algorithm" derives and justifies
// each threshold from the actual color separation measured in the committed
// scene — these are not arbitrary):
//   hue   < kHueMaxDeg  — fruit ripeness 0.35..1.0 maps to hue 0..78 deg
//                         (see make_synthetic.py); foliage sits at
//                         100..140 deg. kHueMaxDeg=85 keeps a comfortable
//                         15-degree margin on the foliage side while
//                         admitting the full fruit range with headroom.
//   sat   > kSatMin     — the PRIMARY discriminator against dark branch
//                         strokes: fruit saturation is 0.65..0.90 (shading
//                         and jitter included) vs. branches at 0.40..0.50.
//                         kSatMin=0.55 sits cleanly between the two.
//   value > kValMin     — a backstop against near-black shadow/branch
//                         pixels; fruit value never drops below the
//                         AMBIENT_FLOOR-driven ~0.27 even on a fully
//                         shadowed side (make_synthetic.py), so kValMin=0.22
//                         has margin without excluding real shadowed fruit.
// ---------------------------------------------------------------------------
constexpr float kHueMaxDeg = 85.0f;
constexpr float kSatMin    = 0.55f;
constexpr float kValMin    = 0.22f;

// Robust depth estimator: accept a pixel as an "inlier" if its depth is
// within kInlierSigmaMul sensor-noise standard deviations of the
// component's pass-1 mean depth (a documented +/- 3-sigma band keeps ~99.7%
// of true in-fruit sensor noise while rejecting boundary/mixed-depth pixels
// and any stray occluder bleed-through).
constexpr float kInlierSigmaMul = 3.0f;

// Minimum pixel count for a connected component to be reported as a
// detection (rather than residual noise the morphological opening did not
// fully remove — THEORY.md "How we verify correctness" measures exactly
// what this threshold catches: the synthetic scene's deliberate false-
// positive "glint" specks).
constexpr int kMinComponentPixels = 30;

// Sentinel: a background / non-fruit pixel's label. Never a valid linear
// pixel index (those are always >= 0), so it can never collide with a real
// canonical label.
constexpr int kLabelNone = -1;

// Safety cap on CCL propagation sweeps (main.cu loops calling
// launch_ccl_propagate_sweep until it reports no change, or this many
// sweeps — whichever comes first; THEORY.md "Numerical considerations"
// measures the actual convergence sweep count on the committed scene, which
// is far below this cap).
constexpr int kMaxCclSweeps = 512;

// ---------------------------------------------------------------------------
// FruitDetection — the pipeline's final output record, ONE per detected
// connected component, built on the HOST from the per-component arrays
// below (both main.cu's GPU path and reference_cpu.cpp's CPU path fill this
// exact struct, so main.cu's verification step compares like-for-like).
// ---------------------------------------------------------------------------
struct FruitDetection {
    int   label;           // canonical label (= this component's root pixel's linear index)
    int   pixel_count;     // foreground pixels in the component (post-opening, pre-size-filter)
    float centroid_px_x;   // pixel-space centroid x (sum_x / count), px
    float centroid_px_y;   // pixel-space centroid y (sum_y / count), px
    int   bbox_min_x, bbox_max_x;   // inclusive pixel bounding box
    int   bbox_min_y, bbox_max_y;
    float radius_px;       // AREA-based screen radius estimate: sqrt(count / pi) — see THEORY.md
    float depth_m;         // robust (inlier-band) depth estimate, meters, camera frame
    float center_x_m;      // back-projected 3-D center, camera frame, meters
    float center_y_m;
    float center_z_m;      // == depth_m (repeated for a self-contained 3-D point)
    float radius_m;        // back-projected 3-D radius, meters
    float mean_hue_deg;    // mean hue over the component's pixels, degrees
    float ripeness;        // mapped from mean_hue_deg, in [0,1] (0=least ripe seen, 1=reddest)
};

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// Kernel prototypes are declared for documentation purposes at each
// launch_* wrapper below (CLAUDE.md's header-carries-summary,
// definition-carries-essay convention); the __global__ functions themselves
// are file-local to kernels.cu (no other translation unit calls them
// directly — only through the host wrappers declared below).

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Stage 1 — launch_rgb_to_hsv: per-pixel RGB -> HSV.
//   d_rgb        : DEVICE pointer, [H*W*3] uint8, interleaved input.
//   d_h, d_s, d_v: DEVICE pointers, [H*W] float OUT — degrees / [0,1] / [0,1].
// Launch: one thread per pixel (map). See kernels.cu for the derivation of
// the conversion and why it separates ripeness color from shading.
// ---------------------------------------------------------------------------
void launch_rgb_to_hsv(const unsigned char* d_rgb,
                       float* d_h, float* d_s, float* d_v,
                       int W, int H);

// ---------------------------------------------------------------------------
// Stage 2 — launch_fruit_mask: per-pixel HSV gate -> binary mask.
//   d_h, d_s, d_v : DEVICE pointers, [H*W] float (from launch_rgb_to_hsv).
//   d_mask        : DEVICE pointer, [H*W] uint8 OUT — 0 or 1.
// ---------------------------------------------------------------------------
void launch_fruit_mask(const float* d_h, const float* d_s, const float* d_v,
                       unsigned char* d_mask, int W, int H);

// ---------------------------------------------------------------------------
// Stage 3 — morphological opening (erode then dilate), 3x3 FULL SQUARE
// structuring element (8-neighborhood) — deliberately DIFFERENT connectivity
// from CCL's 4-neighborhood (kernels.cu explains why both choices are made).
//   d_mask_in / d_mask_out : DEVICE pointers, [H*W] uint8. Never the same
//                            buffer (each pixel's output depends on its
//                            neighbors' INPUT values — in-place would race).
// ---------------------------------------------------------------------------
void launch_morph_erode(const unsigned char* d_mask_in, unsigned char* d_mask_out, int W, int H);
void launch_morph_dilate(const unsigned char* d_mask_in, unsigned char* d_mask_out, int W, int H);

// ---------------------------------------------------------------------------
// Stage 4 — connected-component labeling by iterative label propagation.
//   launch_ccl_init          : d_mask -> d_label. Foreground pixel p gets
//                               label[p] = p (its own linear index);
//                               background gets kLabelNone.
//   launch_ccl_propagate_sweep : ONE relaxation sweep. For every foreground
//                               pixel p, label[p] = min(label[p], min over
//                               its 4-connected foreground neighbors'
//                               CURRENT label) via atomicMin (Gauss-Seidel-
//                               style: neighbors already updated THIS sweep
//                               are seen immediately, which converges faster
//                               than a strict ping-pong Jacobi scheme, and
//                               is still order-independent at the fixed
//                               point — see the file header). Sets
//                               *d_changed = 1 (device int; host resets it
//                               to 0 before each call) iff any pixel's label
//                               actually decreased this sweep. Call
//                               repeatedly (main.cu) until it reports no
//                               change or kMaxCclSweeps is reached.
// ---------------------------------------------------------------------------
void launch_ccl_init(const unsigned char* d_mask, int* d_label, int W, int H);
void launch_ccl_propagate_sweep(const unsigned char* d_mask, int* d_label, int W, int H, int* d_changed);

// ---------------------------------------------------------------------------
// Stage 5 — per-component statistics, keyed by CANONICAL label value
// (directly, as a dense array index — see the file header "PER-COMPONENT
// STATISTIC ARRAYS"). All comp_* arrays are DEVICE pointers, [H*W] each.
//
//   launch_component_stats_init : zero/reset every comp_* array to its
//     identity element (0 for sums/counts, +W/-1/+H/-1 for the bbox
//     min/max accumulators — see kernels.cu for why bbox needs a dedicated
//     init rather than cudaMemset).
//   launch_component_stats_pass1 : for every foreground pixel p, atomically
//     accumulate into comp_count, comp_sum_x, comp_sum_y, comp_min_x,
//     comp_max_x, comp_min_y, comp_max_y, comp_sum_hue, comp_sum_depth — all
//     indexed at [label[p]].
//   launch_component_mean_depth : elementwise comp_mean_depth[i] =
//     comp_sum_depth[i]/comp_count[i] where comp_count[i]>0, else 0 (a MAP,
//     not an atomic step — every index is written by exactly one "owner"
//     thread, so no race is possible here even though the index space is
//     the same dense [H*W] layout).
//   launch_component_stats_pass2_inlier : for every foreground pixel p,
//     using comp_mean_depth[label[p]] and the sensor noise model
//     (kDepthNoiseK, kInlierSigmaMul), atomically accumulate
//     comp_sum_depth_inlier / comp_count_inlier at [label[p]] IF the
//     pixel's own depth is within the inlier band.
//   launch_component_finalize_depth : elementwise comp_final_depth[i] =
//     inlier mean if comp_count_inlier[i]>0, else falls back to
//     comp_mean_depth[i] (a component too small/noisy for any inlier is
//     rare but must not divide by zero).
// ---------------------------------------------------------------------------
void launch_component_stats_init(int* comp_count, int* comp_sum_x, int* comp_sum_y,
                                 int* comp_min_x, int* comp_max_x,
                                 int* comp_min_y, int* comp_max_y,
                                 float* comp_sum_hue, float* comp_sum_depth,
                                 float* comp_sum_depth_inlier, int* comp_count_inlier,
                                 int W, int H);

void launch_component_stats_pass1(const unsigned char* d_mask, const int* d_label,
                                  const float* d_h, const float* d_depth,
                                  int* comp_count, int* comp_sum_x, int* comp_sum_y,
                                  int* comp_min_x, int* comp_max_x,
                                  int* comp_min_y, int* comp_max_y,
                                  float* comp_sum_hue, float* comp_sum_depth,
                                  int W, int H);

void launch_component_mean_depth(const int* comp_count, const float* comp_sum_depth,
                                 float* comp_mean_depth, int W, int H);

void launch_component_stats_pass2_inlier(const unsigned char* d_mask, const int* d_label,
                                         const float* d_depth, const float* comp_mean_depth,
                                         float* comp_sum_depth_inlier, int* comp_count_inlier,
                                         int W, int H);

void launch_component_finalize_depth(const float* comp_mean_depth,
                                     const float* comp_sum_depth_inlier, const int* comp_count_inlier,
                                     float* comp_final_depth, int W, int H);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — oracle twins of the pipeline above.
// The mask/HSV stages are line-by-line twins (same formulas, same
// thresholds); the CCL stage is DELIBERATELY a different algorithm
// (union-find, not label propagation) — see the file header "VERIFICATION
// STRATEGY". build_detections_cpu / build_detections_gpu_side (the latter
// called from main.cu after copying GPU buffers back) both assemble the
// final FruitDetection vector from the same per-component arrays, so the
// comparison logic in main.cu is shared, not duplicated per side.
// ---------------------------------------------------------------------------
void rgb_to_hsv_cpu(const unsigned char* rgb, float* h, float* s, float* v, int W, int H);
void fruit_mask_cpu(const float* h, const float* s, const float* v, unsigned char* mask, int W, int H);
void morph_erode_cpu(const unsigned char* mask_in, unsigned char* mask_out, int W, int H);
void morph_dilate_cpu(const unsigned char* mask_in, unsigned char* mask_out, int W, int H);

// ccl_union_find_cpu — classic Rosenfeld two-pass union-find CCL (4-
// connectivity, matching the GPU's connectivity choice), THEN a
// canonicalization pass that relabels every foreground pixel to the MINIMUM
// linear index within its component — the exact convention the GPU's label
// propagation converges to on its own, making the two directly comparable
// (see the file header). label[] is the OUTPUT, same layout as the GPU's.
void ccl_union_find_cpu(const unsigned char* mask, int* label, int W, int H);

// Per-component statistics, sequential accumulation over the SAME dense
// [H*W]-indexed array layout the GPU uses (so main.cu's array-level
// comparison is meaningful) — see kernels.cuh's Stage 5 comment for the
// field semantics; this single CPU function performs both passes (mean,
// then inlier re-accumulation) since there is no parallelism to stage here.
void component_stats_cpu(const unsigned char* mask, const int* label,
                         const float* h, const float* depth,
                         int* comp_count, int* comp_sum_x, int* comp_sum_y,
                         int* comp_min_x, int* comp_max_x,
                         int* comp_min_y, int* comp_max_y,
                         float* comp_sum_hue, float* comp_final_depth,
                         int W, int H);

#endif // PROJECT_KERNELS_CUH
