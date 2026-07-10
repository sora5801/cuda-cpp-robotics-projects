// ===========================================================================
// kernels.cu — GPU kernels for project 30.01
//              Agriculture, Milestone 1: fruit detection + 3-D localization
//              + ripeness
//
// Ten kernels, each a small, single-concept teaching unit (CLAUDE.md section
// 6.2): a color-space MAP (RGB->HSV), a gate MAP (mask), two STENCILs
// (morphological erode/dilate), an iterative STENCIL+ATOMIC relaxation
// (connected-component label propagation), and five small MAP/ATOMIC-SCATTER
// kernels that turn a labeled pixel image into per-fruit statistics.
//
// Launch geometry, used by every kernel below: ONE THREAD PER PIXEL, a 1-D
// grid over the flat linear index i = y*W + x, block = 256, grid =
// ceil(W*H/256), with a simple `if (i >= N) return;` tail guard — NOT a
// grid-stride loop. Unlike a truly open-ended problem size, this project's N
// = W*H is fixed at 307,200 by the committed scene; a single generous launch
// covers it with no meaningful difference in occupancy or code clarity, so
// the extra stride-loop machinery (SAXPY's pattern, still the right choice
// there) would only add ceremony here. Neighbor-indexed kernels (morphology,
// CCL) recover (x, y) from i via x = i % W, y = i / W — a standard, cheap
// technique that avoids a 2-D launch just to get 2-D neighbor math.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cmath>

// ===========================================================================
// Stage 1 — RGB -> HSV
// ===========================================================================

// ---------------------------------------------------------------------------
// rgb_to_hsv_kernel — the standard max/min/chroma HSV conversion, DELIBERATELY
// TRIG-FREE (no atan2f anywhere) — every branch below is +, -, *, / and
// comparisons, which IEEE-754 guarantees compute IDENTICALLY on host and
// device (same rounding, no transcendental-function ULP differences). That
// is what lets main.cu's VERIFY stage compare GPU and CPU HSV with an
// extremely tight tolerance (THEORY.md "Numerical considerations").
//
// WHY HSV, physically: a pixel's RGB triple confounds two independent
// things — the SURFACE'S COLOR (what wavelengths it reflects) and the
// LIGHTING (how much of that color reaches the sensor). A fruit's ripening
// color (green -> yellow -> orange -> red) is a change in the surface's
// reflectance spectrum; the Lambertian shading across its lit/shadowed
// sides is a change in incident light intensity. RGB mixes both into three
// correlated numbers; HSV SEPARATES them: hue (H) tracks the reflectance
// spectrum (ripeness) almost independently of shading, while value (V)
// absorbs almost all of the shading variation. THEORY.md "The math" derives
// the conversion and proves this separation formally.
//
// Parameters:
//   rgb        : [H*W*3] uint8 device pointer, interleaved (R,G,B) input.
//   h, s, v    : [H*W] float device pointers OUT — degrees / [0,1] / [0,1].
// Thread mapping: thread i owns pixel i (i = y*W + x); a pure per-pixel MAP,
// no neighbor reads, so no launch-geometry subtlety beyond the tail guard.
// ---------------------------------------------------------------------------
__global__ void rgb_to_hsv_kernel(const unsigned char* __restrict__ rgb,
                                  float* __restrict__ h,
                                  float* __restrict__ s,
                                  float* __restrict__ v,
                                  int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Normalize the byte triple to [0,1] float — the natural domain for the
    // rest of the conversion (and for every downstream consumer of s, v).
    const float r = rgb[i * 3 + 0] * (1.0f / 255.0f);
    const float g = rgb[i * 3 + 1] * (1.0f / 255.0f);
    const float b = rgb[i * 3 + 2] * (1.0f / 255.0f);

    const float cmax = fmaxf(r, fmaxf(g, b));
    const float cmin = fminf(r, fminf(g, b));
    const float delta = cmax - cmin;    // "chroma": 0 for a perfectly gray pixel

    // Hue: which of the six 60-degree sectors we are in, found by WHICH
    // channel is the max, then a linear (not trig) formula within the
    // sector — the classic hexcone construction. delta==0 (achromatic gray,
    // e.g. deep shadow) leaves hue at 0 by convention; it is meaningless
    // there and the fruit-mask kernel's saturation/value gates exclude such
    // pixels anyway, so an arbitrary hue for them causes no harm downstream.
    float hue_deg = 0.0f;
    if (delta > 1e-6f) {
        if (cmax == r)      hue_deg = 60.0f * fmodf((g - b) / delta, 6.0f);
        else if (cmax == g) hue_deg = 60.0f * ((b - r) / delta + 2.0f);
        else                hue_deg = 60.0f * ((r - g) / delta + 4.0f);
        if (hue_deg < 0.0f) hue_deg += 360.0f;   // fmodf can return negative for negative input
    }

    const float sat = (cmax > 1e-6f) ? (delta / cmax) : 0.0f;   // saturation: chroma relative to brightness
    const float val = cmax;                                     // value: the brightest channel

    h[i] = hue_deg;
    s[i] = sat;
    v[i] = val;
}

void launch_rgb_to_hsv(const unsigned char* d_rgb, float* d_h, float* d_s, float* d_v, int W, int H)
{
    const int N = W * H;
    const int block = 256;
    const int grid = (N + block - 1) / block;
    rgb_to_hsv_kernel<<<grid, block>>>(d_rgb, d_h, d_s, d_v, N);
    CUDA_CHECK_LAST_ERROR("rgb_to_hsv_kernel launch");
}

// ===========================================================================
// Stage 2 — fruit-likelihood mask
// ===========================================================================

// ---------------------------------------------------------------------------
// fruit_mask_kernel — the three-gate AND described in kernels.cuh: hue below
// kHueMaxDeg (fruit-colored, not foliage-green), saturation above kSatMin
// (vivid, not a dull brown branch), value above kValMin (not near-black
// shadow). All three must hold — a single channel cannot separate every
// confusable class in this scene (THEORY.md "The algorithm" walks the
// specific branch-vs-fruit and foliage-vs-fruit confusions this design
// resolves, and states plainly which confusion it does NOT resolve:
// green-on-green unripe fruit, scoped out of this milestone's sample scene).
// ---------------------------------------------------------------------------
__global__ void fruit_mask_kernel(const float* __restrict__ h,
                                  const float* __restrict__ s,
                                  const float* __restrict__ v,
                                  unsigned char* __restrict__ mask,
                                  int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const bool fruit_like = (h[i] < kHueMaxDeg) && (s[i] > kSatMin) && (v[i] > kValMin);
    mask[i] = fruit_like ? 1u : 0u;
}

void launch_fruit_mask(const float* d_h, const float* d_s, const float* d_v,
                       unsigned char* d_mask, int W, int H)
{
    const int N = W * H;
    const int block = 256;
    const int grid = (N + block - 1) / block;
    fruit_mask_kernel<<<grid, block>>>(d_h, d_s, d_v, d_mask, N);
    CUDA_CHECK_LAST_ERROR("fruit_mask_kernel launch");
}

// ===========================================================================
// Stage 3 — morphological opening (erode, then dilate)
// ===========================================================================

// ---------------------------------------------------------------------------
// morph_erode_kernel / morph_dilate_kernel — 3x3 FULL SQUARE structuring
// element (8-neighborhood: the pixel and its 4 edge- + 4 corner-neighbors).
// This is a STENCIL, not a map: each output pixel reads 9 input pixels.
//
//   erode : output 1 only if ALL 9 neighbors are 1 (mask shrinks by ~1px
//           at every boundary — a lone 1-2px speck with fewer than 9
//           foreground neighbors anywhere is wiped out entirely).
//   dilate: output 1 if ANY of the 9 neighbors is 1 (mask regrows by ~1px).
//
// OPENING = erode THEN dilate: small speckle that erosion fully deletes
// never comes back (dilate only regrows what erosion left standing), while
// a real fruit blob (tens to hundreds of pixels across) loses a ~1px rim to
// erosion and gets essentially all of it back from dilation. main.cu runs
// this ONCE (not iterated) — a single opening pass is enough to clear this
// scene's synthetic glint specks (measured in THEORY.md); iterating for a
// more aggressive cleanup is README Exercise territory.
//
// WHY 8-CONNECTIVITY HERE BUT 4-CONNECTIVITY FOR CCL BELOW (a deliberate,
// stated asymmetry, not an oversight): morphology's job is to judge a
// pixel's LOCAL neighborhood densely in every direction including diagonals
// (a diagonal-only speck is still speckle); CCL's job is to decide whether
// two pixels belong to the SAME OBJECT, where 4-connectivity is the
// simpler, more conservative (never over-merges through a single diagonal
// touch) teaching choice — cross-reference 02.04's Euclidean clustering,
// which faces the analogous 3-D radius-vs-connectivity choice for point
// clouds. THEORY.md "The algorithm" discusses 8-connected CCL as a
// documented alternative (and README Exercise).
//
// Boundary handling: pixels outside the image are treated as 0 (background)
// — the simplest, standard "zero-padding" convention; it can only make a
// border pixel's erosion stricter (never spuriously pass), which is the
// safe direction for a mask meant to exclude, not include, uncertain pixels.
// ---------------------------------------------------------------------------
__global__ void morph_erode_kernel(const unsigned char* __restrict__ mask_in,
                                   unsigned char* __restrict__ mask_out,
                                   int W, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;   // recover 2-D coords from the flat index (file header)

    unsigned char all_set = 1u;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            // Out-of-bounds reads as 0 (zero-padding — see the comment above).
            const unsigned char nv = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                    ? mask_in[ny * W + nx] : 0u;
            all_set &= nv;   // erosion survives only if EVERY one of the 9 is set
        }
    }
    mask_out[i] = all_set;
}

__global__ void morph_dilate_kernel(const unsigned char* __restrict__ mask_in,
                                    unsigned char* __restrict__ mask_out,
                                    int W, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    unsigned char any_set = 0u;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            const unsigned char nv = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                    ? mask_in[ny * W + nx] : 0u;
            any_set |= nv;   // dilation fires if ANY of the 9 is set
        }
    }
    mask_out[i] = any_set;
}

void launch_morph_erode(const unsigned char* d_mask_in, unsigned char* d_mask_out, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    morph_erode_kernel<<<grid, block>>>(d_mask_in, d_mask_out, W, H);
    CUDA_CHECK_LAST_ERROR("morph_erode_kernel launch");
}

void launch_morph_dilate(const unsigned char* d_mask_in, unsigned char* d_mask_out, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    morph_dilate_kernel<<<grid, block>>>(d_mask_in, d_mask_out, W, H);
    CUDA_CHECK_LAST_ERROR("morph_dilate_kernel launch");
}

// ===========================================================================
// Stage 4 — connected-component labeling via iterative label propagation
// ===========================================================================

// ---------------------------------------------------------------------------
// ccl_init_kernel — label[p] = p (own linear index) for every foreground
// pixel, kLabelNone for every background pixel. This is the "everyone
// starts as their own component" initial condition the propagation kernel
// then relaxes toward the true per-component minimum (kernels.cuh's file
// header proves the fixed point is unique).
// ---------------------------------------------------------------------------
__global__ void ccl_init_kernel(const unsigned char* __restrict__ mask,
                                int* __restrict__ label, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    label[i] = mask[i] ? i : kLabelNone;
}

void launch_ccl_init(const unsigned char* d_mask, int* d_label, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    ccl_init_kernel<<<grid, block>>>(d_mask, d_label, N);
    CUDA_CHECK_LAST_ERROR("ccl_init_kernel launch");
}

// ---------------------------------------------------------------------------
// ccl_propagate_sweep_kernel — ONE relaxation sweep of
//     label[p] <- min( label[p], min over 4-connected foreground
//                       neighbors q of label[q] )
// implemented via atomicMin so concurrent threads racing to update the SAME
// pixel (impossible here — each thread owns a distinct p) or reading a
// neighbor mid-update (very possible — that neighbor's OWN thread may be
// updating it in this very sweep) never corrupts memory: atomicMin is a
// correct, linearizable read-modify-write regardless of interleaving.
//
// CONVERGENCE ARGUMENT (the load-bearing claim this kernel rests on, proven
// carefully in THEORY.md "The algorithm" — sketched here because it is WHY
// main.cu's later exact-equality check against the CPU union-find oracle is
// justified, not just hoped for):
//   1. Every label only ever DECREASES (atomicMin never increases a value)
//      and is bounded below by 0 (indices are non-negative) — a
//      monotonically decreasing, bounded sequence per pixel MUST converge
//      in finitely many sweeps to a fixed point where no update fires.
//   2. At any fixed point, label[p] <= label[q] for every foreground
//      neighbor q of p (else the update would still fire) — so along ANY
//      foreground-connected PATH from p, labels are non-increasing, which
//      means every pixel in a connected component ends up <= the smallest
//      INITIAL label anywhere in that component (which is that component's
//      own minimum linear index, since propagation can reach it by
//      following the path from p to that very pixel).
//   3. No label can go BELOW the component's true minimum initial label
//      either — atomicMin never invents values, it only propagates ones
//      that were already present somewhere in the component.
//   Together: the UNIQUE fixed point of this relaxation is
//   label[p] = min{ q : q connected to p } q, for every foreground pixel p —
//   exactly the "canonical root = minimum linear index in the component"
//   claim in kernels.cuh, independent of the SCHEDULE (which threads ran in
//   which sweep, in which order) that got there. This is the same argument
//   that proves Bellman-Ford correct for shortest paths with all-zero edge
//   weights — CCL-by-propagation IS that algorithm, specialized.
//
// Cost, honestly: convergence takes as many sweeps as the component's
// GRAPH DIAMETER (the longest shortest 4-connected path across it) — for a
// compact blob a few dozen pixels wide, a few dozen sweeps; THEORY.md
// measures the actual count on the committed scene. A CPU union-find
// (reference_cpu.cpp) converges in one raster pass plus near-O(1)-amortized
// path compression — asymptotically better, and exactly why it is the
// simpler CPU oracle here rather than the GPU's teaching algorithm (see
// kernels.cuh's file header and README "Prior art": production GPU CCL
// libraries typically implement union-find variants, not naive
// propagation, for exactly this reason — 02.04 is that project).
// ---------------------------------------------------------------------------
__global__ void ccl_propagate_sweep_kernel(const unsigned char* __restrict__ mask,
                                           int* __restrict__ label,
                                           int W, int H, int* __restrict__ changed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (!mask[i]) return;               // background pixels never hold a label to relax

    const int x = i % W, y = i / W;
    int best = label[i];

    // 4-connected neighbors only (up/down/left/right) — see the file header
    // "WHY 8-CONNECTIVITY HERE BUT 4-CONNECTIVITY FOR CCL" above.
    if (x > 0     && mask[i - 1])     best = min(best, label[i - 1]);
    if (x < W - 1 && mask[i + 1])     best = min(best, label[i + 1]);
    if (y > 0     && mask[i - W])     best = min(best, label[i - W]);
    if (y < H - 1 && mask[i + W])     best = min(best, label[i + W]);

    if (best < label[i]) {
        // atomicMin is technically unnecessary for correctness here (this
        // thread is the ONLY writer of label[i]), but it IS the operation
        // that makes the READS of neighbor labels above safe to combine
        // with a same-sweep write elsewhere: it guarantees every write to
        // any label[] slot is a proper min-reduction, so no interleaving of
        // this kernel's own threads can ever leave a slot holding a value
        // that is not a true min of some subset of contributions — the
        // property the convergence argument above depends on.
        atomicMin(&label[i], best);
        atomicOr(changed, 1);   // tell the host: at least one label moved this sweep
    }
}

void launch_ccl_propagate_sweep(const unsigned char* d_mask, int* d_label, int W, int H, int* d_changed)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    ccl_propagate_sweep_kernel<<<grid, block>>>(d_mask, d_label, W, H, d_changed);
    CUDA_CHECK_LAST_ERROR("ccl_propagate_sweep_kernel launch");
}

// ===========================================================================
// Stage 5 — per-component statistics (atomics keyed by canonical label)
// ===========================================================================

// ---------------------------------------------------------------------------
// component_stats_init_kernel — reset every comp_* array to its accumulator
// IDENTITY element. Sums/counts start at 0 (cudaMemset would work for those
// alone) but the bbox min/max accumulators need NON-ZERO identities
// (min starts at +W or +H so the first real atomicMin always wins; max
// starts at -1 so the first real atomicMax always wins) — that is the one
// reason this is a dedicated kernel rather than a cudaMemset call.
// ---------------------------------------------------------------------------
__global__ void component_stats_init_kernel(int* __restrict__ comp_count,
                                            int* __restrict__ comp_sum_x, int* __restrict__ comp_sum_y,
                                            int* __restrict__ comp_min_x, int* __restrict__ comp_max_x,
                                            int* __restrict__ comp_min_y, int* __restrict__ comp_max_y,
                                            float* __restrict__ comp_sum_hue, float* __restrict__ comp_sum_depth,
                                            float* __restrict__ comp_sum_depth_inlier, int* __restrict__ comp_count_inlier,
                                            int W, int H, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    comp_count[i] = 0;
    comp_sum_x[i] = 0;
    comp_sum_y[i] = 0;
    comp_min_x[i] = W;    // no real x can reach W (valid x is 0..W-1) -> first atomicMin always wins
    comp_max_x[i] = -1;   // no real x can be negative -> first atomicMax always wins
    comp_min_y[i] = H;
    comp_max_y[i] = -1;
    comp_sum_hue[i] = 0.0f;
    comp_sum_depth[i] = 0.0f;
    comp_sum_depth_inlier[i] = 0.0f;
    comp_count_inlier[i] = 0;
}

void launch_component_stats_init(int* comp_count, int* comp_sum_x, int* comp_sum_y,
                                 int* comp_min_x, int* comp_max_x,
                                 int* comp_min_y, int* comp_max_y,
                                 float* comp_sum_hue, float* comp_sum_depth,
                                 float* comp_sum_depth_inlier, int* comp_count_inlier,
                                 int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    component_stats_init_kernel<<<grid, block>>>(comp_count, comp_sum_x, comp_sum_y,
                                                  comp_min_x, comp_max_x, comp_min_y, comp_max_y,
                                                  comp_sum_hue, comp_sum_depth,
                                                  comp_sum_depth_inlier, comp_count_inlier, W, H, N);
    CUDA_CHECK_LAST_ERROR("component_stats_init_kernel launch");
}

// ---------------------------------------------------------------------------
// component_stats_pass1_kernel — the ATOMIC SCATTER at the heart of Stage 5:
// every foreground pixel contributes one small write to NINE per-component
// accumulators, all indexed by its OWN canonical label (a dense [H*W]-sized
// array — see kernels.cuh's file header for why compaction is unnecessary
// here). Many pixels (all of one fruit's blob) write to the SAME index
// concurrently — this is exactly the pattern atomics exist for; a plain
// (non-atomic) += here would silently drop updates whenever two threads
// happened to read-modify-write the same slot at once (a classic, silent
// GPU correctness bug — CLAUDE.md section 6.1 rule 3 says name it, so: this
// is that bug, and atomics are the fix).
// ---------------------------------------------------------------------------
__global__ void component_stats_pass1_kernel(const unsigned char* __restrict__ mask,
                                             const int* __restrict__ label,
                                             const float* __restrict__ h,
                                             const float* __restrict__ depth,
                                             int* __restrict__ comp_count,
                                             int* __restrict__ comp_sum_x, int* __restrict__ comp_sum_y,
                                             int* __restrict__ comp_min_x, int* __restrict__ comp_max_x,
                                             int* __restrict__ comp_min_y, int* __restrict__ comp_max_y,
                                             float* __restrict__ comp_sum_hue, float* __restrict__ comp_sum_depth,
                                             int W, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (!mask[i]) return;

    const int x = i % W, y = i / W;
    const int L = label[i];   // this pixel's canonical component index — the scatter target

    atomicAdd(&comp_count[L], 1);
    atomicAdd(&comp_sum_x[L], x);
    atomicAdd(&comp_sum_y[L], y);
    atomicMin(&comp_min_x[L], x);
    atomicMax(&comp_max_x[L], x);
    atomicMin(&comp_min_y[L], y);
    atomicMax(&comp_max_y[L], y);
    atomicAdd(&comp_sum_hue[L], h[i]);
    atomicAdd(&comp_sum_depth[L], depth[i]);
}

void launch_component_stats_pass1(const unsigned char* d_mask, const int* d_label,
                                  const float* d_h, const float* d_depth,
                                  int* comp_count, int* comp_sum_x, int* comp_sum_y,
                                  int* comp_min_x, int* comp_max_x,
                                  int* comp_min_y, int* comp_max_y,
                                  float* comp_sum_hue, float* comp_sum_depth,
                                  int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    component_stats_pass1_kernel<<<grid, block>>>(d_mask, d_label, d_h, d_depth,
                                                   comp_count, comp_sum_x, comp_sum_y,
                                                   comp_min_x, comp_max_x, comp_min_y, comp_max_y,
                                                   comp_sum_hue, comp_sum_depth, W, H);
    CUDA_CHECK_LAST_ERROR("component_stats_pass1_kernel launch");
}

// ---------------------------------------------------------------------------
// component_mean_depth_kernel — elementwise MAP (not a scatter): every index
// i is read and written by exactly one thread, so this is race-free without
// atomics even though it shares the same dense [H*W] index space as the
// atomic kernels around it. comp_count[i]==0 at almost every index (only
// canonical-root indices are ever nonzero) — guarded to avoid a division by
// zero; the result there is simply never read downstream.
// ---------------------------------------------------------------------------
__global__ void component_mean_depth_kernel(const int* __restrict__ comp_count,
                                            const float* __restrict__ comp_sum_depth,
                                            float* __restrict__ comp_mean_depth, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int c = comp_count[i];
    comp_mean_depth[i] = (c > 0) ? (comp_sum_depth[i] / static_cast<float>(c)) : 0.0f;
}

void launch_component_mean_depth(const int* comp_count, const float* comp_sum_depth,
                                 float* comp_mean_depth, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    component_mean_depth_kernel<<<grid, block>>>(comp_count, comp_sum_depth, comp_mean_depth, N);
    CUDA_CHECK_LAST_ERROR("component_mean_depth_kernel launch");
}

// ---------------------------------------------------------------------------
// component_stats_pass2_inlier_kernel — the ROBUST re-accumulation pass
// (kernels.cuh "ROBUST DEPTH ESTIMATION"): each foreground pixel looks up
// its OWN component's pass-1 mean depth, computes the sensor-noise-derived
// inlier band around it, and — only if its own depth falls inside that
// band — contributes to a second atomic sum/count. Pixels straddling a
// blob's silhouette (mixed fruit/foliage depth) or sitting near an
// occluding neighbor are exactly the ones a single mean would be dragged
// around by; this pass trims them out using a PHYSICALLY DERIVED band
// (the sensor's own noise curve), not an arbitrary percentile.
// ---------------------------------------------------------------------------
__global__ void component_stats_pass2_inlier_kernel(const unsigned char* __restrict__ mask,
                                                     const int* __restrict__ label,
                                                     const float* __restrict__ depth,
                                                     const float* __restrict__ comp_mean_depth,
                                                     float* __restrict__ comp_sum_depth_inlier,
                                                     int* __restrict__ comp_count_inlier,
                                                     int W, int H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (!mask[i]) return;

    const int L = label[i];
    const float mean_z = comp_mean_depth[L];
    // Sensor noise model, evaluated at the component's own mean depth (the
    // best available estimate of the true range) — same formula the
    // synthetic generator used to CREATE the noise (kernels.cuh header).
    const float sigma_z = kDepthNoiseK * mean_z * mean_z;
    const float band = kInlierSigmaMul * sigma_z;

    if (fabsf(depth[i] - mean_z) <= band) {
        atomicAdd(&comp_sum_depth_inlier[L], depth[i]);
        atomicAdd(&comp_count_inlier[L], 1);
    }
}

void launch_component_stats_pass2_inlier(const unsigned char* d_mask, const int* d_label,
                                         const float* d_depth, const float* comp_mean_depth,
                                         float* comp_sum_depth_inlier, int* comp_count_inlier,
                                         int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    component_stats_pass2_inlier_kernel<<<grid, block>>>(d_mask, d_label, d_depth, comp_mean_depth,
                                                          comp_sum_depth_inlier, comp_count_inlier, W, H);
    CUDA_CHECK_LAST_ERROR("component_stats_pass2_inlier_kernel launch");
}

// ---------------------------------------------------------------------------
// component_finalize_depth_kernel — elementwise MAP, same race-free
// reasoning as component_mean_depth_kernel: the FINAL depth estimate this
// project reports is the inlier-band mean, with a fallback to the pass-1
// mean for the rare component whose inlier band (a real, if unlikely,
// possibility for a very small or very noisy blob) catches zero pixels.
// ---------------------------------------------------------------------------
__global__ void component_finalize_depth_kernel(const float* __restrict__ comp_mean_depth,
                                                 const float* __restrict__ comp_sum_depth_inlier,
                                                 const int* __restrict__ comp_count_inlier,
                                                 float* __restrict__ comp_final_depth, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int ci = comp_count_inlier[i];
    comp_final_depth[i] = (ci > 0) ? (comp_sum_depth_inlier[i] / static_cast<float>(ci))
                                    : comp_mean_depth[i];
}

void launch_component_finalize_depth(const float* comp_mean_depth,
                                     const float* comp_sum_depth_inlier, const int* comp_count_inlier,
                                     float* comp_final_depth, int W, int H)
{
    const int N = W * H, block = 256, grid = (N + block - 1) / block;
    component_finalize_depth_kernel<<<grid, block>>>(comp_mean_depth, comp_sum_depth_inlier,
                                                      comp_count_inlier, comp_final_depth, N);
    CUDA_CHECK_LAST_ERROR("component_finalize_depth_kernel launch");
}
