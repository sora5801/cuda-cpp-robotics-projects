// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 01.05
//               (SIFT on GPU: Gaussian scale space, DoG extrema, warp-level
//               orientation histograms, warp-level 128-D descriptors,
//               brute-force L2 matching. SURF is documented-only — see
//               THEORY.md "Where this sits in the real world.")
//
// Role in the project
// -------------------
// This header is the SINGLE-SOURCED CONTRACT between kernels.cu (the GPU
// implementation, nvcc), reference_cpu.cpp (the independent CPU oracle,
// cl.exe) and main.cu (orchestration, nvcc). Per the repo's twin-
// independence ruling (see reference_cpu.cpp's header, and
// docs/PROJECT_TEMPLATE/src/reference_cpu.cpp for the repo-wide version):
// data LAYOUT (structs, constants, indexing formulas, the 1-D Gaussian
// weight tables) is single-sourced HERE; the ALGORITHMIC CORE (the
// convolution loop, the extrema test, the sub-pixel solve, the histogram
// accumulation, the matcher) is written TWICE — once in kernels.cu,
// independently again in reference_cpu.cpp.
//
// Why the Gaussian weight TABLE is shared data, not duplicated algorithm
// (a deliberate design choice, read carefully — the precedent is 01.04's
// build_orb_base_pattern()): the weights themselves are a closed-form
// function of sigma with no meaningful "second implementation" (both sides
// would just retype exp(-x^2/2sigma^2) and normalize — a drift risk, not
// independence). What DOES get written twice is the CONVOLUTION LOOP that
// consumes those weights (kernels.cu's gaussian_blur_*_kernel vs
// reference_cpu.cpp's gaussian_blur_cpu) — that loop is where an indexing
// bug would actually hide, and that is where the twin comparison earns its
// keep. Sharing the weights also makes the blur-stage VERIFY comparison
// cleaner: any GPU-vs-CPU difference is then attributable ONLY to
// summation order / FMA fusion, not to two different roundings of the
// Gaussian itself — see main.cu's "VERIFY(scale space)" for the measured
// numbers this isolation buys us.
//
// The five-stage pipeline this header describes
// -----------------------------------------------
//   STAGE 1 SCALE SPACE — build_gaussian_kernel_1d() (shared weight table)
//     + gaussian_blur_h_kernel/gaussian_blur_v_kernel (separable 2-pass
//     convolution, teaching SEPARABILITY: an NxN Gaussian stencil costs
//     O(N^2) taps/pixel; two 1-D passes cost O(2N) — see THEORY.md) build a
//     kNumOctaves x kImagesPerOctave pyramid of increasingly-blurred images;
//     dog_subtract_kernel differences adjacent levels into kDogPerOctave
//     Difference-of-Gaussian images per octave (DoG approximates the
//     scale-normalized Laplacian-of-Gaussian — THEORY.md "The problem").
//
//   STAGE 2 DOG EXTREMA — dog_extrema_candidates_kernel: a 3x3x3 stencil
//     (own DoG layer's 8 neighbors + the 9 pixels directly above and below
//     in scale) with atomic compaction, exactly the NMS-and-append pattern
//     01.04's nms_select_fast_kernel established for a 2-D neighborhood,
//     extended to 3-D (space x space x scale).
//
//   STAGE 3 REFINE — refine_keypoint_kernel: one thread per raw candidate,
//     an ITERATIVE 3x3 linear solve (the quadratic Taylor fit of D around
//     the candidate — see 33.01 for the batched-small-matrix-solve pattern
//     this is a single-instance, per-thread version of), plus the classic
//     Lowe contrast-threshold + principal-curvature (Hessian trace^2/det)
//     edge rejection.
//
//   STAGE 4 ORIENTATION — orientation_kernel: ONE WARP PER KEYPOINT (the
//     catalog's ratified GPU-mapping hook). 32 lanes cooperatively build a
//     36-bin gradient-magnitude histogram over a Gaussian-weighted patch,
//     each lane accumulating a PRIVATE partial histogram, then a
//     __shfl_down_sync TREE REDUCTION folds the 32 lanes' partials into
//     one final histogram — the centerpiece "warp-level reduction" lesson
//     this project exists to teach (full derivation, and the naive
//     shared-atomic alternative it beats, sit with the kernel definition).
//
//   STAGE 5 DESCRIBE — describe_kernel: the SAME one-warp-per-keypoint
//     mapping and the SAME local-accumulate-then-shuffle-reduce pattern,
//     scaled up to a 128-bin (4x4 spatial cells x 8 orientation bins)
//     trilinear-interpolated histogram — the classic SIFT descriptor.
//
//   STAGE 6 MATCH — match_l2_kernel: brute-force squared-L2, one thread per
//     query descriptor (mirrors 01.04's hamming_match_kernel, but every
//     comparison is now 128 float subtracts+multiplies instead of 8
//     uint32 XOR+POPCs — main.cu measures and reports the real cost of
//     that, the "float L2 world vs binary Hamming world" contrast the
//     project brief asks for).
//
// Why ".cuh"?
// -----------
// The repo convention (CLAUDE.md §12): .cuh headers may contain CUDA-
// specific constructs (__global__, kernel launches) and are included from
// nvcc-compiled .cu files; this header is ALSO included by reference_cpu.cpp
// (compiled by cl.exe, which does not know "__global__"), so every
// device-only declaration is fenced behind #ifdef __CUDACC__. Plain
// functions/structs used by ALL THREE files (the shared data-layout
// contract) live OUTSIDE the fence.
//
// Read this after: main.cu.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint8_t — raw grayscale pixel storage
#include <cmath>     // expf/sqrtf/atan2f etc. (host use too — <cmath> is plain C++17, fine for cl.exe)

// ===========================================================================
// Image / octave geometry — SINGLE-SOURCED. scripts/make_synthetic.py
// renders both committed sample images at exactly kBaseW x kBaseH; main.cu
// asserts the loaded PGM dimensions match before doing anything else (fail
// loud, never silently truncate — repo convention, see 01.01/01.02/01.04's
// sample loaders).
//
// octave_w(o)/octave_h(o) below are the SHARED formula every stage uses to
// find an octave's resolution: kBaseW >> o (integer halving per octave,
// matching the classic Lowe pyramid — see THEORY.md "The algorithm").
// ===========================================================================
constexpr int kBaseW = 256;   // octave-0 image width, px
constexpr int kBaseH = 256;   // octave-0 image height, px

constexpr float kPi = 3.14159265358979323846f;   // used host AND device side (angle wrap, bin-width math)

// ---------------------------------------------------------------------------
// SCOPING DECISION (documented honestly — README §13 restates this): Lowe's
// paper and most production SIFT implementations use s=3 intervals (6
// images/octave, 5 DoG, 3 extrema layers) over 4+ octaves of a 2x-upsampled
// image. This project uses s=2 intervals (5 images/octave, 4 DoG, 2 extrema
// layers) over 2 octaves of the ORIGINAL (not upsampled) resolution — a
// smaller but algorithmically IDENTICAL pipeline, chosen so a 256x256
// teaching image produces a legible number of keypoints (tens, not
// thousands) with a build/run time appropriate for a demo that must run on
// every learner's machine in milliseconds, while still exercising every
// stage of real SIFT: multi-octave AND multi-interval scale sampling, DoG
// extrema in 3x3x3, sub-pixel/sub-scale refinement, orientation histograms,
// and the 128-D descriptor. s=2 is a valid (if non-optimal, per Lowe's own
// empirical study) choice of the SAME formula, not a different algorithm.
// ---------------------------------------------------------------------------
constexpr int kNumOctaves      = 2;   // octave 0: kBaseW x kBaseH; octave 1: half resolution
constexpr int kIntervals       = 2;   // Lowe's "s": interior (extrema-searchable) DoG layers per octave
constexpr int kImagesPerOctave = kIntervals + 3;   // 5 Gaussian-blurred images per octave (indices 0..4)
constexpr int kDogPerOctave    = kIntervals + 2;   // 4 Difference-of-Gaussian images per octave (indices 0..3)
constexpr int kFirstExtremaLayer = 1;              // interior DoG layer indices searched: [1, kIntervals] inclusive
constexpr int kLastExtremaLayer  = kIntervals;      // == 2 -- DoG layers 1 and 2 are searched (each needs a layer above AND below)

// octave_w/octave_h — the shared "how big is octave o" formula. Host+device
// (inline, no CUDA syntax) so every stage (GPU launch-geometry math, CPU
// twin, main.cu's host orchestration) computes IDENTICAL sizes.
inline int octave_w(int o) { return kBaseW >> o; }
inline int octave_h(int o) { return kBaseH >> o; }

// ===========================================================================
// STAGE 1: Gaussian scale space parameters (Lowe 1999/2004's classic
// values). Images are normalized to FLOAT [0,1] on load (see main.cu) so
// these thresholds match the literature's convention directly, with no
// extra 255x scaling to carry through every formula.
// ===========================================================================
constexpr float kSigma0            = 1.6f;   // per-Lowe: the "characteristic" blur of octave 0, level 0
constexpr float kSigmaInputAssumed = 0.5f;   // per-Lowe: assumed pre-existing blur of the INPUT image (camera/sensor/anti-aliasing) before any processing
constexpr float kIntervalScale     = 1.41421356237309504880f;  // 2^(1/kIntervals) = 2^(1/2) = sqrt(2): the per-LEVEL blur-growth ratio (see THEORY.md "The math" for why a GEOMETRIC ratio, not arithmetic)

// A blur step never needs more taps than this many pixels either side of
// center (see build_gaussian_kernel_1d()'s "3-sigma rule" comment). Sized
// generously above the largest sigma_diff this project's octave/interval
// schedule ever produces (measured ~4.53 px at kIntervals=2 -- see
// kernels.cuh's derivation in THEORY.md "The math"); a fixed cap keeps the
// weight-table buffers plain stack arrays instead of heap allocations.
constexpr int kMaxGaussRadius = 24;
constexpr int kMaxGaussTaps   = 2 * kMaxGaussRadius + 1;

// ===========================================================================
// STAGE 2/3: DoG extrema detection + sub-pixel/sub-scale refinement.
// ===========================================================================
// Border margin (px, in the SEARCHING octave's own downsampled grid): the
// 3x3x3 extrema stencil reads +-1 px in x/y (needs 1), the Hessian edge
// test reads +-1 px again (needs 1, same ring), and refine_keypoint_kernel
// may RE-CENTER the search up to kMaxRefineIters times, each step moving at
// most 1 px (its offset-rounding rule, see kernels.cu) -- so an accepted,
// refined keypoint can drift up to kMaxRefineIters px from where it was
// first found. kExtremaBorder=5 gives a full pixel of slack beyond that
// worst case (1 stencil px + up to 3 realistic drift steps, rounded up).
constexpr int kExtremaBorder = 5;

constexpr float kContrastThreshold = 0.03f;  // Lowe's default minimum |D| (on [0,1]-normalized intensity) to even consider a candidate a keypoint
constexpr float kEdgeRatioR        = 10.0f;  // Lowe's default principal-curvature ratio cap (rejects candidates that are edge-like, not corner/blob-like)
constexpr int   kMaxRefineIters    = 5;      // Lowe's classic bound on the re-centering loop
constexpr float kRefineConvergeTol = 0.5f;   // |offset| below this in x, y AND scale simultaneously => converged, no re-centering needed
constexpr int   kMaxDogCandidates  = 4096;   // device candidate-buffer capacity (generous headroom -- see main.cu's measured counts)
constexpr int   kMaxKeypoints      = 1024;   // capacity for ACCEPTED (post-refine, post-contrast, post-edge) keypoints

// ===========================================================================
// STAGE 4: orientation assignment. "One warp per keypoint" is THE ratified
// GPU mapping for this project (catalog hook: warp-level reductions) --
// see orientation_kernel's header in kernels.cu for the full argument.
// ===========================================================================
constexpr int kOriHistBins        = 36;    // 10 degrees/bin, Lowe's classic resolution
constexpr float kOriSigmaFactor   = 1.5f;  // the Gaussian weighting window's sigma = kOriSigmaFactor * (keypoint's OWN octave-local sigma)
constexpr float kOriRadiusFactor  = 3.0f;  // patch sampling radius = round(kOriRadiusFactor * weighting sigma) -- ~99.7% of the Gaussian weight's mass
constexpr float kOriPeakRatio     = 0.8f;  // secondary-peak spawn threshold: any OTHER local peak >= 0.8 * the max bin spawns an additional keypoint at that orientation (Lowe's default)
constexpr int   kMaxOrientedPerKeypoint = 4;   // cap: 1 primary + up to 3 secondary orientations per geometric keypoint
constexpr int   kMaxOrientedKeypoints   = kMaxKeypoints * kMaxOrientedPerKeypoint;
constexpr int   kWarpSize = 32;   // every CUDA-capable GPU to date: 32 lanes/warp. Used as the fixed blockDim for the one-warp-per-keypoint kernels.

// ===========================================================================
// STAGE 5: the 128-D descriptor. 4x4 spatial cells x 8 orientation bins
// (Lowe's classic "d=4, 8 bins" configuration), trilinear soft binning.
// ===========================================================================
constexpr int kDescGridSize = 4;                                   // d: the descriptor's spatial grid is d x d cells
constexpr int kDescOriBins  = 8;                                   // orientation bins per cell (45 degrees/bin)
constexpr int kDescDims     = kDescGridSize * kDescGridSize * kDescOriBins;  // 4*4*8 = 128 -- THE "128-D SIFT descriptor" the catalog names
constexpr float kDescScaleFactor = 3.0f;   // SIFT_DESCR_SCL_FCTR (Lowe/OpenCV): one descriptor "cell" spans this many sigma_oct
constexpr float kDescClipValue   = 0.2f;   // post-L2-normalize component clip (illumination-nonlinearity robustness -- see THEORY.md "Numerical considerations")
constexpr int   kDescMaxRadius   = 64;     // defensive cap on the descriptor sampling window radius (see describe_kernel) -- the formula-derived radius stays well under this for every keypoint this project's scale range produces (measured max ~40px, see main.cu); the cap exists so a future constant change cannot silently explode per-keypoint work

// ===========================================================================
// STAGE 6: brute-force squared-L2 matching (the float-descriptor analogue
// of 01.04's Hamming matcher).
// ===========================================================================
constexpr float kLoweRatioSift = 0.92f;   // see main.cu's derivation comment for why this project's measured value departs from Lowe's classic 0.75/OpenCV's 0.7-0.8 literature range
// kMaxL2DistSq -- absolute cap on the BEST squared-L2 distance, mirroring
// 01.04's kMaxHammingDist belt-and-suspenders role: the ratio test alone
// can still accept a mediocre match if the runner-up happens to be worse
// still. Value is MEASURED-then-margined on the committed sample -- see
// main.cu's derivation comment beside its use.
constexpr float kMaxL2DistSq = 1.6f;
// kMinL2DistSq -- a FLOOR on the BEST squared-L2 distance, the honest,
// measured-necessary companion to kMaxL2DistSq (see main.cu's "How we
// verify correctness" derivation and THEORY.md's "Where this sits in the
// real world" for the full story): this project's synthetic geometric
// scene content (checkerboard right-angle corners) turns out to have a
// small number of near-symmetric CANONICAL shapes, so a small minority of
// UNRELATED keypoint pairs land at an almost implausibly small distance
// (measured as low as ~0.02-0.03 -- FAR below every genuinely-
// corresponding pair's measured range of ~0.3-0.98) purely by geometric
// coincidence, not genuine correspondence. A real photograph's richer
// texture would never produce this "too good to be true" signature; this
// floor rejects it explicitly, named and measured rather than silently
// tolerated.
constexpr float kMinL2DistSq = 0.15f;

// ===========================================================================
// Shared data-layout structs (the "layout contract" -- single-sourced).
// ===========================================================================

// A raw DoG extrema candidate: an integer (octave, layer, x, y) BEFORE
// sub-pixel/sub-scale refinement. (x, y) are in OCTAVE-LOCAL pixel
// coordinates (i.e. relative to that octave's own, possibly-downsampled,
// grid) -- refine_keypoint_kernel/refine_keypoint_cpu map to sub-pixel and
// to full-image coordinates (see SiftKeypoint below).
struct DogCandidate {
    int octave;   // 0..kNumOctaves-1
    int layer;    // kFirstExtremaLayer..kLastExtremaLayer (the DoG index the 3x3x3 test centered on)
    int x, y;     // integer pixel location, octave-local, in [kExtremaBorder, octave_w(o)-kExtremaBorder)
};

// A REFINED, ACCEPTED keypoint -- the output of STAGE 3. Every field the
// rest of the pipeline (orientation, description, gates) needs, gathered
// in one place so downstream code reads as "the keypoint", not five
// parallel arrays (mirrors 01.04's Keypoint struct's role).
struct SiftKeypoint {
    int   octave;        // 0..kNumOctaves-1 -- which pyramid octave this keypoint was found in
    int   layer;          // kFirstExtremaLayer..kLastExtremaLayer -- the (integer) DoG layer index searched
    float x_oct, y_oct;    // refined SUB-PIXEL position, OCTAVE-LOCAL pixel coordinates (that octave's own grid)
    float ds;               // refined SUB-SCALE offset along the interval axis, typically in (-0.5, 0.5) after acceptance
    float x_img, y_img;      // refined position mapped to ORIGINAL image pixel coordinates: x_oct * 2^octave (see kernels.cu)
    float sigma_oct;          // absolute Gaussian sigma AT THIS OCTAVE's own pixel grid = kSigma0 * kIntervalScale^(layer+ds) -- the scale used by orientation/description, which operate ON that octave's image
    float sigma_img;           // absolute sigma in ORIGINAL-image pixel units = sigma_oct * 2^octave -- THE scale used for the scale-recovery gate and for drawing scale circles
    float contrast;              // refined |D_hat| value at the sub-pixel optimum (diagnostic; also the input to the contrast re-check)
};

// One assigned orientation for a geometric keypoint. Lowe's "secondary
// peaks spawn additional keypoints" rule (see orientation_kernel) means the
// SAME SiftKeypoint can appear more than once here, each copy carrying a
// DIFFERENT theta -- exactly how real SIFT (and OpenCV's implementation)
// handles multi-orientation keypoints: descriptors and matches are always
// per-ORIENTED-keypoint, not per-geometric-keypoint.
struct OrientedKeypoint {
    SiftKeypoint kp;   // the geometric keypoint this orientation belongs to
    float theta;        // dominant gradient orientation, RADIANS, wrapped to [0, 2*pi) (see orient conventions note below)
};

// The 128-D float descriptor. Plain array, not compressed/quantized: this
// project stays in "SIFT's float L2 world" throughout, the deliberate
// contrast with 01.04's packed-bit "Hamming world" the project brief asks
// for (see match_l2_kernel's header for the measured cost of that choice).
struct SiftDescriptor {
    float v[kDescDims];
};

// One brute-force match result (query -> best train candidate), plus the
// bookkeeping main.cu needs to explain WHY it was accepted or rejected --
// the float-L2 analogue of 01.04's MatchResult.
struct SiftMatchResult {
    int   query_idx;      // index into the query oriented-keypoint/descriptor array
    int   train_idx;       // index into the train oriented-keypoint/descriptor array (best match)
    float best_dist_sq;     // squared L2 distance to the best train match
    float second_dist_sq;    // squared L2 distance to the second-best train match
    bool  ratio_ok;            // best_dist_sq <= kLoweRatioSift^2 * second_dist_sq (ratio test on SQUARED distances -- see match_l2_kernel)
    bool  cross_ok;              // train_idx's own best match (reverse direction) is query_idx (mutual/cross-consistency check)
    bool  accepted;                // ratio_ok && cross_ok && (best_dist_sq <= kMaxL2DistSq) -- the final "is this a match" verdict
};

// ===========================================================================
// Shared HOST-ONLY helper: the 1-D Gaussian weight-table builder (see this
// file's header for why the WEIGHTS are shared data while the CONVOLUTION
// LOOP that consumes them is written twice, independently, in kernels.cu
// and reference_cpu.cpp).
// ===========================================================================

// ---------------------------------------------------------------------------
// build_gaussian_kernel_1d — fill a normalized 1-D Gaussian tap table for a
// given sigma, using the standard "3-sigma rule" radius (>=99.7% of the
// continuous Gaussian's mass lies within +-3 sigma, so truncating there and
// RE-NORMALIZING the discrete taps to sum to exactly 1 is an accepted,
// negligible-error approximation -- see THEORY.md "Numerical
// considerations" for the truncation-error argument).
//
// Parameters:
//   sigma    — blur standard deviation, pixels, > 0.
//   weights  — OUT: [kMaxGaussTaps] taps, ONLY weights[0..2*radius] are
//              written and meaningful; weights[radius] is the center tap.
//              Computed in DOUBLE precision (this function runs a handful
//              of times per octave, not per-pixel -- negligible cost --
//              purely so the table itself is as precise as the host can
//              make it) then cast down to float for the taps consumed by
//              both the GPU upload and the CPU twin.
//   radius   — OUT: the tap radius actually used, <= kMaxGaussRadius.
// Side effects: none beyond writing weights/radius. Complexity: O(radius).
// ---------------------------------------------------------------------------
inline void build_gaussian_kernel_1d(float sigma, float weights[kMaxGaussTaps], int& radius)
{
    // ceil(3*sigma), clamped into [1, kMaxGaussRadius] -- radius 0 (a
    // delta function) would be a degenerate "blur", and kMaxGaussRadius is
    // this project's fixed buffer capacity (see its derivation comment).
    int r = static_cast<int>(std::ceil(3.0 * static_cast<double>(sigma)));
    if (r < 1) r = 1;
    if (r > kMaxGaussRadius) r = kMaxGaussRadius;
    radius = r;

    const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma);
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        // The (unnormalized) Gaussian density; the 1/sqrt(2*pi*sigma^2)
        // prefactor is IRRELEVANT here because we renormalize by `sum`
        // below anyway -- computing it would be extra float ops that
        // cancel out, so we skip it (a small "why we didn't write the
        // textbook formula verbatim" teaching note).
        const double w = std::exp(-(static_cast<double>(i) * i) / (2.0 * s2));
        weights[i + r] = static_cast<float>(w);   // temporarily unnormalized
        sum += w;
    }
    // Renormalize so the DISCRETE, TRUNCATED taps sum to exactly 1 --
    // otherwise the truncated tails' missing mass would slightly darken
    // every blurred image (a real, if tiny, bug in naive Gaussian-blur
    // implementations that skip this step).
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int i = 0; i <= 2 * r; ++i) weights[i] *= inv_sum;
}

// ---------------------------------------------------------------------------
// sigma_at — the absolute Gaussian sigma of pyramid level `interval` (a
// possibly-fractional REAL number, since refined keypoints carry a
// fractional `ds`), OCTAVE-LOCAL (i.e. relative to that octave's own pixel
// grid -- multiply by 2^octave for original-image units, see
// SiftKeypoint::sigma_img). This is THE formula behind kSigma0's geometric
// growth (see THEORY.md "The math" for the derivation of why sigma follows
// a GEOMETRIC, not arithmetic, progression across levels).
// ---------------------------------------------------------------------------
// __host__ __device__ (nvcc only -- guarded so cl.exe, which never defines
// these keywords, still compiles this header when reference_cpu.cpp
// includes it): this tiny, purely-arithmetic formula is used by
// refine_keypoint_kernel (device code), main.cu's host orchestration, AND
// reference_cpu.cpp's independent refine twin -- unlike the ALGORITHMIC
// cores this project keeps deliberately duplicated (see this file's
// header), a one-line closed-form formula has no meaningful "second
// implementation" to write (both would just retype the same power law),
// so it is single-sourced DATA/MATH, exactly like build_gaussian_kernel_1d.
#ifdef __CUDACC__
__host__ __device__
#endif
inline float sigma_at(float interval)
{
    return kSigma0 * powf(kIntervalScale, interval);   // powf (not std::pow): guaranteed device-callable under nvcc, and identical on the host for float arguments
}

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// ===========================================================================
// STAGE 1 kernels — Gaussian pyramid + Difference-of-Gaussian.
// ===========================================================================

// gaussian_blur_h_kernel / gaussian_blur_v_kernel — the two passes of a
// SEPARABLE 2-D Gaussian convolution (see this file's header + THEORY.md
// "The GPU mapping" for the O(N^2) -> O(2N) tap-count argument). Each
// thread owns one OUTPUT pixel and reads `2*radius+1` taps along ONE axis
// from `src`, writing one value to `dst`. Border handling: CLAMP-TO-EDGE
// (reads outside [0,W) or [0,H) repeat the nearest edge pixel) -- the
// standard choice for image blurring (as opposed to zero-padding, which
// would darken every border and corrupt DoG values near the image edge
// exactly where kExtremaBorder is trying to stay safely inside anyway).
// src/dst: [W*H] device float images (row-major); weights: [2*radius+1]
// device float taps (from build_gaussian_kernel_1d(), uploaded once per
// call site by the launch wrapper).
__global__ void gaussian_blur_h_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       int W, int H, const float* __restrict__ weights, int radius);
__global__ void gaussian_blur_v_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       int W, int H, const float* __restrict__ weights, int radius);

// dog_subtract_kernel — a pure MAP: dst[i] = a[i] - b[i], one thread per
// pixel, no stencil, no shared memory (the simplest possible kernel in
// this file, included so the DoG stage is not silently folded into the
// blur kernels -- it is its own algorithmic step, per THEORY.md's
// DoG-approximates-LoG argument).
__global__ void dog_subtract_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                    float* __restrict__ dst, int W, int H);

// downsample2x_kernel — the between-octave step: a pure MAP, one thread
// per OUTPUT pixel, dst[y][x] = src[2y][2x] (nearest-neighbor decimation,
// no anti-alias pre-filter of its own -- see kernels.cu for why the
// SOURCE image, already blurred to 2*kSigma0 by the pyramid construction
// that precedes this call, has already done that job). src: [srcW*srcH]
// device IN; dst: [(srcW/2)*(srcH/2)] device OUT.
__global__ void downsample2x_kernel(const float* __restrict__ src, int srcW, int srcH,
                                    float* __restrict__ dst);

// ===========================================================================
// STAGE 2 kernel — DoG extrema detection.
// ===========================================================================

// dog_extrema_candidates_kernel — one thread per octave-interior pixel: a
// 3x3x3 STENCIL (own layer's 8 neighbors + the 9 pixels directly above and
//9 directly below in scale = 26 neighbors) local-extremum test, gated by
// the contrast pre-filter, with ATOMIC COMPACTION into the candidate list
// (the exact NMS-and-append pattern 01.04's nms_select_fast_kernel
// established, extended from a 2-D neighborhood to 3-D). dog_below/
// dog_center/dog_above are three [W*H] device DoG images from the SAME
// octave, ADJACENT DoG indices (center = the layer being tested). Full
// documentation (the strict-inequality tie rule, contrast threshold) sits
// with the definition in kernels.cu.
__global__ void dog_extrema_candidates_kernel(const float* __restrict__ dog_below,
                                              const float* __restrict__ dog_center,
                                              const float* __restrict__ dog_above,
                                              int W, int H, int octave, int layer,
                                              DogCandidate* __restrict__ out,
                                              int* __restrict__ counter, int max_candidates);

// ===========================================================================
// STAGE 3 kernel — sub-pixel/sub-scale refinement + contrast/edge accept.
// ===========================================================================

// refine_keypoint_kernel — one thread per RAW candidate: the iterative
// quadratic-Taylor sub-pixel/sub-scale solve (a small 3x3 linear system
// per iteration, see kernels.cu for the full derivation and the 33.01
// cross-reference) plus Lowe's contrast-threshold and principal-curvature
// edge tests. dog: ALL kDogPerOctave device DoG images for THIS octave,
// concatenated layer-major (dog + layer*W*H is DoG layer `layer`).
// candidates: [n] device IN. out: [n] device OUT (one SiftKeypoint per
// candidate; only entries with accepted[i]==true are meaningful).
// accepted: [n] device OUT (bool-as-int: 1 if the candidate survived
// refinement + contrast + edge tests, 0 otherwise -- main.cu compacts).
__global__ void refine_keypoint_kernel(const float* __restrict__ dog, int W, int H,
                                       const DogCandidate* __restrict__ candidates, int n,
                                       SiftKeypoint* __restrict__ out, int* __restrict__ accepted);

// ===========================================================================
// STAGE 4 kernel — orientation assignment. ONE WARP (32 threads) PER
// KEYPOINT -- see this kernel's definition in kernels.cu for the complete
// warp-shuffle-reduction lesson this project is built around.
// ===========================================================================

// orientation_kernel — blockDim.x MUST be kWarpSize (32); one BLOCK per
// keypoint (gridDim.x = n). ALL keypoints in one launch must share the
// SAME octave (main.cu launches this once PER OCTAVE — see that file):
// gauss_oct is THAT octave's FULL Gaussian pyramid, all kImagesPerOctave
// images concatenated layer-major ([W*H] each, image `i` at
// gauss_oct + i*W*H) — the kernel selects image kps[blockIdx.x].layer
// internally, because different keypoints in the SAME octave can carry
// DIFFERENT layer indices (1 or 2), so no single fixed image pointer
// would serve every block.
//
// Output layout — FIXED SLOTS, not an atomic-compacted list (a deliberate
// CONTRAST with the DoG-extrema kernel's atomic compaction above): every
// keypoint's fan-out is bounded by the SAME constant (kMaxOrientedPerKeypoint),
// known before the launch, so block `kp_idx` can be given its OWN private
// sub-range out[kp_idx*kMaxOrientedPerKeypoint .. +kMaxOrientedPerKeypoint)
// and fill it with a purely LOCAL counter — no cross-block atomic, no
// launch-order-dependent output position, and (the property this project's
// VERIFY step actually needs) a fully DETERMINISTIC output order that lines
// up index-for-index with orientation_cpu()'s natural sequential order.
// Atomics were the right tool for DoG extrema (an UNBOUNDED, data-dependent
// number of producers writing to one shared list); fixed slots are the
// right tool here (a BOUNDED, statically-known fan-out per producer) — see
// THEORY.md "GPU mapping" for this contrast, spelled out.
//
// kps: [n] device IN. out: [n * kMaxOrientedPerKeypoint] device OUT (fixed
// capacity; only the first out_spawn_count[i] entries of keypoint i's
// sub-range are meaningful). out_spawn_count: [n] device OUT, each in
// [1, kMaxOrientedPerKeypoint] (a keypoint with zero gradient signal in
// its patch writes 0 — see the kernel body's "flat patch" note).
__global__ void orientation_kernel(const float* __restrict__ gauss_oct, int W, int H,
                                   const SiftKeypoint* __restrict__ kps, int n,
                                   OrientedKeypoint* __restrict__ out, int* __restrict__ out_spawn_count);

// ===========================================================================
// STAGE 5 kernel — 128-D descriptor. Same one-warp-per-keypoint mapping.
// ===========================================================================

// describe_kernel — blockDim.x MUST be kWarpSize (32); one BLOCK per
// ORIENTED keypoint (gridDim.x = n). gauss_oct: same per-octave, full-
// pyramid convention as orientation_kernel (all keypoints in one launch
// share an octave; the kernel selects image kps[i].kp.layer internally).
// kps: [n] device IN (oriented keypoints -- theta already assigned).
// desc_out: [n] device OUT.
__global__ void describe_kernel(const float* __restrict__ gauss_oct, int W, int H,
                                const OrientedKeypoint* __restrict__ kps, int n,
                                SiftDescriptor* __restrict__ desc_out);

// ===========================================================================
// STAGE 6 kernel — brute-force squared-L2 matching.
// ===========================================================================

// match_l2_kernel — one thread per QUERY descriptor, brute-force scan of
// every TRAIN descriptor, running best + second-best (squared L2, index).
// query: [nQuery] device IN. train: [nTrain] device IN.
__global__ void match_l2_kernel(const SiftDescriptor* __restrict__ query, int nQuery,
                                const SiftDescriptor* __restrict__ train, int nTrain,
                                float* __restrict__ best1_dist_sq, int* __restrict__ best1_idx,
                                float* __restrict__ best2_dist_sq, int* __restrict__ best2_idx);

#endif // __CUDACC__ --------------------------------------------------------

// ===========================================================================
// Host-callable LAUNCH WRAPPERS — own the grid/block math, the ephemeral
// device weight/counter buffers, and the post-launch error check
// (CLAUDE.md §6.1 rule 7), visible to any translation unit (only their
// DEFINITIONS, in kernels.cu, need nvcc).
// ===========================================================================
void launch_gaussian_blur(const float* d_src, float* d_dst, int W, int H, float sigma, float* d_tmp);
void launch_dog_subtract(const float* d_a, const float* d_b, float* d_dst, int W, int H);
void launch_downsample2x(const float* d_src, int srcW, int srcH, float* d_dst);

int  launch_dog_extrema(const float* d_dog_below, const float* d_dog_center, const float* d_dog_above,
                        int W, int H, int octave, int layer,
                        DogCandidate* d_out, int max_candidates);

// launch_refine_keypoints — launches refine_keypoint_kernel into caller-
// owned, pre-sized [n] device buffers d_out (SiftKeypoint) and d_accepted
// (int, 1/0). Void return: unlike the atomic-compaction kernels, refine
// has a KNOWN, fixed output size (n, one slot per input candidate, some
// unused) — main.cu downloads both buffers and does the host-side
// accept-filter itself (see that file's refine_octave()), the same
// "caller compacts a fixed-size download" shape launch_orientation uses.
void launch_refine_keypoints(const float* d_dog, int W, int H,
                             const DogCandidate* d_candidates, int n,
                             SiftKeypoint* d_out, int* d_accepted);

// launch_orientation — launches orientation_kernel into the caller-owned,
// pre-sized [n*kMaxOrientedPerKeypoint] device buffer d_out and [n] device
// buffer d_spawn_count (see the kernel's fixed-slot layout note above).
// Returns nothing: unlike the atomic-compaction kernels, there is no
// launch-time overflow possible here (every block's fan-out is capped in
// the kernel itself), so there is nothing for the wrapper to check beyond
// the launch error CUDA_CHECK_LAST_ERROR already covers. Compaction into a
// contiguous, ordered host-side list is main.cu's job (a small, honest
// host loop over out_spawn_count — see that file's compact_oriented()).
void launch_orientation(const float* d_gauss_oct, int W, int H, const SiftKeypoint* d_kps, int n,
                        OrientedKeypoint* d_out, int* d_spawn_count);

void launch_describe(const float* d_gauss_oct, int W, int H, const OrientedKeypoint* d_kps, int n,
                     SiftDescriptor* d_desc);

void launch_match_l2(const SiftDescriptor* d_query, int nQuery, const SiftDescriptor* d_train, int nTrain,
                     float* d_best1_dist_sq, int* d_best1_idx, float* d_best2_dist_sq, int* d_best2_idx);

// ===========================================================================
// CPU reference (oracle) declarations — defined in reference_cpu.cpp.
// Declared here so the compiler enforces signature agreement with what
// main.cu calls, exactly like 01.04's precedent. Each ALGORITHMICALLY
// mirrors (independently — see reference_cpu.cpp's header) the GPU kernel
// of the same concept, but as a single-threaded host loop over plain
// arrays, never a CUDA type.
// ===========================================================================
void gaussian_blur_cpu(const float* src, float* dst, int W, int H, const float* weights, int radius);
void dog_subtract_cpu(const float* a, const float* b, float* dst, int W, int H);
void downsample2x_cpu(const float* src, int srcW, int srcH, float* dst);

int  dog_extrema_cpu(const float* dog_below, const float* dog_center, const float* dog_above,
                     int W, int H, int octave, int layer, DogCandidate* out, int max_candidates);

int  refine_keypoints_cpu(const float* dog, int W, int H,
                          const DogCandidate* candidates, int n, SiftKeypoint* out);

int  orientation_cpu(const float* gauss_oct, int W, int H, const SiftKeypoint* kps, int n,
                     OrientedKeypoint* out, int out_capacity);

void describe_cpu(const float* gauss_oct, int W, int H, const OrientedKeypoint* kps, int n, SiftDescriptor* desc_out);

void match_l2_cpu(const SiftDescriptor* query, int nQuery, const SiftDescriptor* train, int nTrain,
                  float* best1_dist_sq, int* best1_idx, float* best2_dist_sq, int* best2_idx);

#endif // PROJECT_KERNELS_CUH
