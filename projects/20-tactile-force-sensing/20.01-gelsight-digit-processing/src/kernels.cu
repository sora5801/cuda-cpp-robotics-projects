// ===========================================================================
// kernels.cu — GPU kernels for project 20.01
//              GelSight/DIGIT processing: contact patch, shear field via
//              optical flow, slip detection in real time
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, with the small host-side launch
// wrappers that own the grid/block math right next to each kernel (repo
// convention — the launch-configuration reasoning sits beside the code it
// configures). Five kernels, three teaching patterns:
//   contact_mask_kernel              -> MAP            (one thread, one pixel)
//   erode3_kernel / dilate3_kernel   -> STENCIL         (3x3 neighborhood)
//   patch_stats_kernel               -> MAP + ATOMIC REDUCTION
//   detect_markers_kernel            -> one thread per MARKER, small search
//   track_markers_kernel             -> MAP over markers (trivial per-thread work)
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp
// (the line-by-line CPU twin of every kernel below).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ---------------------------------------------------------------------------
// Shared 2-D launch geometry for every "one thread per pixel" kernel below
// (contact_mask, erode3, dilate3, patch_stats). 16x16 = 256 threads/block:
// a warp-multiple, small enough that even a 320x240 image needs only
// ceil(320/16)*ceil(240/16) = 20*15 = 300 blocks — plenty to fill an RTX
// 2080 SUPER's 46 SMs many times over even at this tiny resolution.
// ---------------------------------------------------------------------------
static constexpr int kBlockX = 16;
static constexpr int kBlockY = 16;

// ---------------------------------------------------------------------------
// contact_mask_kernel — MAP: mask[p] = 255 if |frame[p]-baseline[p]| >=
// threshold else 0, independently per pixel. THE textbook map: no thread
// reads any other thread's output, so this is the simplest possible GPU
// kernel — one thread, one pixel, no shared memory, no synchronization.
//
// Thread-to-data mapping: thread (bx*16+tx, by*16+ty) owns pixel (x,y) with
// x = bx*16+tx, y = by*16+ty; the last row/column of blocks is ragged (320
// and 240 are both multiples of 16 here, so THIS image never rags, but the
// bounds check stays — a kernel that only works for exact multiples is a
// kernel waiting to crash on the next resolution someone tries).
//
// Memory: frame/baseline reads and the mask write are all COALESCED —
// consecutive threadIdx.x -> consecutive x -> consecutive addresses, one
// 128-byte transaction per warp on each array. Nothing is reused between
// threads, so (as in 08.01/01.02's map kernels) shared memory would add
// complexity for zero benefit here.
// ---------------------------------------------------------------------------
__global__ void contact_mask_kernel(const unsigned char* __restrict__ frame,
                                    const unsigned char* __restrict__ baseline,
                                    unsigned char* __restrict__ mask,
                                    int W, int H, int threshold)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;                    // guard the (here, exact) ragged edge

    const int idx = y * W + x;                        // row-major pixel index (repo image convention)
    // abs-diff in int: frame/baseline are uint8, so the subtraction must
    // widen first or (a-b) underflows for a<b when both are unsigned char.
    const int diff = static_cast<int>(frame[idx]) - static_cast<int>(baseline[idx]);
    const int adiff = diff < 0 ? -diff : diff;
    mask[idx] = (adiff >= threshold) ? 255 : 0;
}

void launch_contact_mask(const unsigned char* d_frame, const unsigned char* d_baseline,
                          unsigned char* d_mask, int W, int H, int threshold)
{
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((W + kBlockX - 1) / kBlockX, (H + kBlockY - 1) / kBlockY);
    contact_mask_kernel<<<grid, block>>>(d_frame, d_baseline, d_mask, W, H, threshold);
    CUDA_CHECK_LAST_ERROR("contact_mask_kernel launch");
}

// ---------------------------------------------------------------------------
// erode3_kernel / dilate3_kernel — STENCIL: binary morphological erosion and
// dilation over an 8-connected 3x3 neighborhood (the classic "min/max filter
// on a 0/255 image" formulation — erosion is a MIN filter, dilation is a MAX
// filter, exactly as their grayscale cousins are, just on a two-level
// image). Composed erode-then-dilate by the CALLER = a morphological OPEN:
// erosion needs a full 3x3 of 255s to keep a pixel lit, so it extinguishes
// any blob narrower than 3px in any direction (the speckle this project's
// threshold noise produces); dilation then regrows every SURVIVING blob
// back to its original size (approximately — corners lost by erosion are
// NOT perfectly restored, a documented, harmless imprecision at 3x3 scale;
// THEORY.md "Numerical considerations").
//
// Why open and not close (dilate-then-erode)? Close would FILL small gaps —
// useful for holes inside a solid contact blob, not for removing speckle
// outside one. This project's noise (isolated pixels crossing threshold by
// chance) is exactly the "small bright spots on a dark background" case
// open is built for.
//
// Thread-to-data mapping: identical to contact_mask_kernel — one thread per
// output pixel, reads its own 3x3 neighborhood. Out-of-bounds neighbors read
// as 0 (border pixels can never have a full lit neighborhood, so erosion
// naturally clears the image edge — a harmless, documented boundary effect
// at this project's scale, since the marker grid margin already keeps every
// marker well clear of the image border).
//
// Memory: each output pixel reads 9 INPUT pixels, but neighboring output
// threads' 3x3 windows overlap heavily (each input pixel is read by up to 9
// different threads) — the textbook case FOR a shared-memory tile (load a
// (16+2)x(16+2) halo once per block, reuse it 9x per thread). This project
// keeps the naive global-memory version for teaching clarity at a 320x240
// image (erosion+dilation together measure well under a millisecond here —
// see [time] lines); 07.09's jump-flooding kernels are this repo's worked
// example of the shared-memory-tile version of the same idea, for a project
// where the tile actually pays for itself.
// ---------------------------------------------------------------------------
__global__ void erode3_kernel(const unsigned char* __restrict__ in,
                              unsigned char* __restrict__ out,
                              int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Erosion: the output is lit ONLY if every one of the 9 neighbors
    // (including the center) is lit. min_val starts at 255 (lit) and any
    // 0 neighbor drags it to 0 — the min-filter formulation of erosion.
    unsigned char min_val = 255;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            // Out-of-bounds reads as 0 ("not lit") — see header comment.
            const unsigned char v = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                   ? in[ny * W + nx] : 0;
            if (v < min_val) min_val = v;
        }
    }
    out[y * W + x] = min_val;
}

__global__ void dilate3_kernel(const unsigned char* __restrict__ in,
                               unsigned char* __restrict__ out,
                               int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Dilation: the output is lit if ANY of the 9 neighbors is lit — the
    // max-filter formulation. max_val starts at 0 and any lit neighbor
    // pulls it to 255.
    unsigned char max_val = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            const unsigned char v = (nx >= 0 && nx < W && ny >= 0 && ny < H)
                                   ? in[ny * W + nx] : 0;
            if (v > max_val) max_val = v;
        }
    }
    out[y * W + x] = max_val;
}

void launch_erode3(const unsigned char* d_in, unsigned char* d_out, int W, int H)
{
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((W + kBlockX - 1) / kBlockX, (H + kBlockY - 1) / kBlockY);
    erode3_kernel<<<grid, block>>>(d_in, d_out, W, H);
    CUDA_CHECK_LAST_ERROR("erode3_kernel launch");
}

void launch_dilate3(const unsigned char* d_in, unsigned char* d_out, int W, int H)
{
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((W + kBlockX - 1) / kBlockX, (H + kBlockY - 1) / kBlockY);
    dilate3_kernel<<<grid, block>>>(d_in, d_out, W, H);
    CUDA_CHECK_LAST_ERROR("dilate3_kernel launch");
}

// ---------------------------------------------------------------------------
// patch_stats_kernel — MAP + ATOMIC REDUCTION: every lit mask pixel adds 1
// to a running area count and its own (x,y) to running centroid sums.
//
// Design choice, stated honestly: a "real" reduction kernel would first sum
// within each block using shared memory (one atomicAdd per BLOCK instead of
// per THREAD — 256x fewer atomics), the pattern 33.01/09.01 teach properly.
// This kernel skips that: at 320x240 = 76,800 pixels, a naive per-thread
// atomicAdd to a global counter is already sub-microsecond work on any
// current GPU (measured in [time] below), and the shared-memory version
// would ADD real code (a block-reduce tree, __syncthreads, three separate
// shared arrays) that teaches nothing new at THIS project's scale — this
// repo's "teaching beats cleverness" call (CLAUDE.md §1), made explicitly
// rather than silently. If you scale this pipeline to megapixel images,
// switch to the block-reduce pattern first (README Exercise).
//
// Parameters: d_area/d_sumx/d_sumy are single-element accumulators the
// CALLER must cudaMemset to 0 before this launch — this kernel only adds.
// unsigned long long (not int) headroom: even at a hypothetical 4K-frame
// scale, sumx/sumy (coordinate * pixel count) cannot overflow 64 bits.
// ---------------------------------------------------------------------------
__global__ void patch_stats_kernel(const unsigned char* __restrict__ mask,
                                   int W, int H,
                                   unsigned long long* __restrict__ area,
                                   unsigned long long* __restrict__ sumx,
                                   unsigned long long* __restrict__ sumy)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    if (mask[y * W + x] != 0) {
        // atomicAdd on unsigned long long is native from compute capability
        // 2.0 onward — safe on every architecture this repo targets
        // (sm_75/86/89). Three independent atomics; they contend with other
        // threads writing the SAME address, not with each other (different
        // addresses), so this is "many threads, one counter each" — the
        // simplest atomic pattern there is.
        atomicAdd(area, 1ULL);
        atomicAdd(sumx, static_cast<unsigned long long>(x));
        atomicAdd(sumy, static_cast<unsigned long long>(y));
    }
}

void launch_patch_stats(const unsigned char* d_mask, int W, int H,
                         unsigned long long* d_area,
                         unsigned long long* d_sumx,
                         unsigned long long* d_sumy)
{
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((W + kBlockX - 1) / kBlockX, (H + kBlockY - 1) / kBlockY);
    patch_stats_kernel<<<grid, block>>>(d_mask, W, H, d_area, d_sumx, d_sumy);
    CUDA_CHECK_LAST_ERROR("patch_stats_kernel launch");
}

// ---------------------------------------------------------------------------
// detect_markers_kernel — one thread PER MARKER (kNumMarkers = 221 threads
// total, not one per pixel): each thread searches a (2R+1)x(2R+1) window of
// THIS frame centered on ITS marker's REST position for the darkest pixel —
// a local-minimum / blob search scoped to where the marker is EXPECTED to
// be, not a whole-image search.
//
// Why search near rest instead of a whole-image blob detector? Two honest
// reasons (THEORY.md "The algorithm" has the full argument):
//   1. It is what real marker trackers do: with the sensor's marker layout
//      known from calibration/manufacture, searching a small neighborhood
//      of the last known (or rest) position is strictly cheaper than
//      re-detecting every blob in the image from scratch every frame, and
//      it sidesteps the data-association problem (whole-image detection
//      returns an UNORDERED candidate list that still has to be matched to
//      marker IDENTITIES — this project's scope note in kernels.cuh/README
//      names that as the piece a real system's assignment step would add).
//   2. kMarkerMarginPx (9px) > kSearchRadiusPx (8px) by construction (see
//      kernels.cuh), so every window is FULLY IN-BOUNDS — no clamping, no
//      divergent branches at the image border, for every one of the 221
//      threads, every frame. A whole-image detector would not get this for
//      free.
//
// Determinism (load-bearing for the exact GPU/CPU verify gate): the window
// is scanned in a FIXED order (dy outer, dx inner, both ascending) and a
// candidate only replaces the running best on a STRICT less-than — ties
// keep the FIRST (smallest dy, then smallest dx) pixel. reference_cpu.cpp
// scans in the identical order, so GPU and CPU agree pixel-for-pixel even
// on a perfectly flat minimum (which this project's marker profile avoids
// in practice, but the tie-break makes the kernel correct regardless).
//
// Thread-to-data mapping: thread i owns marker i (i = blockIdx.x*blockDim.x
// + threadIdx.x); a 1-D grid, since markers have no 2-D neighbor structure
// this kernel cares about (unlike the pixel kernels above).
// ---------------------------------------------------------------------------
__global__ void detect_markers_kernel(const unsigned char* __restrict__ frame,
                                      const Vec2f* __restrict__ rest_pos,
                                      int num_markers, int W, int H, int search_radius,
                                      Vec2f* __restrict__ detected_pos,
                                      int* __restrict__ min_intensity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_markers) return;

    // Rest position is stored as float (Vec2f) but is always an exact
    // integer lattice point (kernels.cuh's formula) — round-to-nearest via
    // +0.5f/truncation is exact here, not an approximation.
    const int cx = static_cast<int>(rest_pos[i].x + 0.5f);
    const int cy = static_cast<int>(rest_pos[i].y + 0.5f);

    int best_val = 256;              // sentinel: any real pixel (0..255) beats this on first comparison
    int best_x = cx, best_y = cy;    // falls back to the rest position itself if nothing ever wins (never
                                      // happens given kMarkerDetectThreshold, but a safe default costs nothing)
    for (int dy = -search_radius; dy <= search_radius; ++dy) {
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            const int nx = cx + dx, ny = cy + dy;
            // The margin > search_radius invariant (kernels.cuh) means this
            // branch is always true in practice; kept as a defensive guard
            // so the kernel is correct even if a caller violates the
            // invariant (e.g. via a hand-edited scenario), per CLAUDE.md
            // §1 "no black boxes" — silent out-of-bounds reads are worse
            // than one redundant comparison.
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            const int v = static_cast<int>(frame[ny * W + nx]);
            if (v < best_val) {          // STRICT less-than: first-found wins ties (see header comment)
                best_val = v;
                best_x = nx;
                best_y = ny;
            }
        }
    }

    detected_pos[i].x = static_cast<float>(best_x);
    detected_pos[i].y = static_cast<float>(best_y);
    min_intensity[i] = best_val;
}

void launch_detect_markers(const unsigned char* d_frame, const Vec2f* d_rest_pos,
                            int num_markers, int W, int H, int search_radius,
                            Vec2f* d_detected_pos, int* d_min_intensity)
{
    const int block = 128;   // 221 markers -> 2 blocks; block size is not performance-critical at this N
    const int grid = (num_markers + block - 1) / block;
    detect_markers_kernel<<<grid, block>>>(d_frame, d_rest_pos, num_markers, W, H,
                                           search_radius, d_detected_pos, d_min_intensity);
    CUDA_CHECK_LAST_ERROR("detect_markers_kernel launch");
}

// ---------------------------------------------------------------------------
// track_markers_kernel — one thread per marker; three trivial, independent
// per-marker computations (another MAP, this time over markers instead of
// pixels):
//   displacement = detected - rest                          (shear field)
//   valid        = (min_intensity < detect_threshold)        (a real marker
//                   was found near rest, not just background texture)
//   in_contact   = mask[round(rest)] != 0                    (is this
//                   marker's undeformed location inside today's contact
//                   patch? sampled from the ALREADY-COMPUTED, ALREADY-
//                   OPENED mask — this is why track_markers runs after the
//                   contact-mask stage in main.cu's per-frame pipeline)
//
// Why sample the mask at REST position, not DETECTED position? A marker
// that has slipped just outside the visible contact boundary is still the
// same physical patch of gel that WAS under the object — "is this marker
// part of the touched patch" is a question about the marker's IDENTITY
// (which rest cell it is), not about where it happens to be drawn this
// frame. Sampling at detected position would flicker a marker in/out of the
// slip-scoring set as its own displacement moves it across the mask
// boundary — exactly the kind of self-referential noise a stable slip
// score must avoid (THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
__global__ void track_markers_kernel(const Vec2f* __restrict__ detected_pos,
                                     const int* __restrict__ min_intensity,
                                     const Vec2f* __restrict__ rest_pos,
                                     const unsigned char* __restrict__ mask,
                                     int num_markers, int W, int H, int detect_threshold,
                                     Vec2f* __restrict__ displacement,
                                     unsigned char* __restrict__ valid,
                                     unsigned char* __restrict__ in_contact)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_markers) return;

    displacement[i].x = detected_pos[i].x - rest_pos[i].x;
    displacement[i].y = detected_pos[i].y - rest_pos[i].y;
    valid[i] = (min_intensity[i] < detect_threshold) ? 1 : 0;

    const int rx = static_cast<int>(rest_pos[i].x + 0.5f);
    const int ry = static_cast<int>(rest_pos[i].y + 0.5f);
    const bool inb = (rx >= 0 && rx < W && ry >= 0 && ry < H);
    in_contact[i] = (inb && mask[ry * W + rx] != 0) ? 1 : 0;
}

void launch_track_markers(const Vec2f* d_detected_pos, const int* d_min_intensity,
                           const Vec2f* d_rest_pos, const unsigned char* d_mask,
                           int num_markers, int W, int H, int detect_threshold,
                           Vec2f* d_displacement, unsigned char* d_valid,
                           unsigned char* d_in_contact)
{
    const int block = 128;
    const int grid = (num_markers + block - 1) / block;
    track_markers_kernel<<<grid, block>>>(d_detected_pos, d_min_intensity, d_rest_pos, d_mask,
                                          num_markers, W, H, detect_threshold,
                                          d_displacement, d_valid, d_in_contact);
    CUDA_CHECK_LAST_ERROR("track_markers_kernel launch");
}
