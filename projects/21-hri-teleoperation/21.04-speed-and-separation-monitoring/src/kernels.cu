// ===========================================================================
// kernels.cu — GPU kernels for project 21.04
//              Speed-and-separation monitoring (didactic, NOT certified --
//              see kernels.cuh's header comment for the full caveat)
//
// Three kernels, three GPU patterns, one shared geometry toolkit:
//
//   render_classify_kernel   -- MAP.        one thread per pixel: render
//                                            the top-down depth image and
//                                            classify BACKGROUND/ROBOT/HUMAN.
//   human_min_distance_kernel -- MAP + REDUCE. one thread scans a stride of
//                                            pixels for HUMAN ones, keeping
//                                            its own running minimum, then a
//                                            shared-memory tree reduction
//                                            collapses each block to one
//                                            (distance, capsule-id) pair.
//   dense_distance_field_kernel -- MAP.     one thread per pixel: the same
//                                            point-capsule distance function
//                                            as above, applied everywhere
//                                            (not just HUMAN pixels) for the
//                                            visual clearance-field artifact.
//
// Why __constant__ memory for the capsules
// -----------------------------------------
// At most 10 capsules (8 robot + 2 human) exist at any time -- tiny,
// read-only, and read by EVERY thread in EVERY kernel launch this frame.
// That is exactly what __constant__ memory is for: a small, cached,
// broadcast-optimized read-only region (64 KB budget on every CUDA GPU;
// our capsules use well under 1 KB). Reading d_robot_capsules[k] from 10
// different threads in a warp costs about the same as one thread reading
// it once -- the constant cache serves the whole warp from one fetch. This
// sits at the "read-only, same-address-for-every-thread" end of the memory
// spectrum 08.01's THEORY.md names (09.01's per-launch __constant__ model
// parameters -> 08.01's uniform global reads -> 07.09's divergent global
// reads); here we are firmly in 09.01's camp, and for the same reason: tiny
// data, read by everyone, unchanged for the whole kernel.
//
// Read this after: kernels.cuh (the geometry/scenario contract) and
// main.cu (the orchestration that calls these launchers every frame).
// Read this beside: reference_cpu.cpp (the independent oracle twin).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cfloat>     // FLT_MAX -- the "no candidate yet" sentinel
#include <cstdio>
#include <cstdlib>
#include <vector>     // host-side scratch for the reduction's tiny final finish

// ---------------------------------------------------------------------------
// GPU-resident scene state: this frame's capsules, uploaded fresh every
// frame by upload_capsules() (a few hundred bytes -- negligible next to a
// 40,000-pixel kernel launch, the same "tiny per-tick upload" trade 08.01
// makes for its 16-byte plant state).
// ---------------------------------------------------------------------------
__constant__ Capsule d_robot_capsules[kNumRobotCapsules];
__constant__ Capsule d_human_capsules[kNumHumanCapsules];

// ===========================================================================
// capsule_top_at — the top-down renderer's core: does this capsule cover
// world point (x,y), and if so, what is the HIGHEST z of its surface there?
//
// Derivation (THEORY.md "The math" walks this in full): a capsule is every
// point within `radius` of segment A->B. Looking straight down at (x,y),
// the capsule is visible there iff the horizontal distance from (x,y) to
// the segment's relevant axis point is <= radius, and the visible height is
// that axis point's z PLUS the vertical "rise" of the swept sphere at that
// horizontal offset, sqrt(radius^2 - d^2) (Pythagoras: a sphere of the
// given radius centered on the axis point).
//
// This is EXACT (not an approximation) because kernels.cuh SECTION 1
// guarantees every capsule here is purely horizontal or purely vertical:
//   HORIZONTAL (az == bz): z is constant along the whole axis, so "the
//     axis point nearest (x,y) horizontally" is unambiguous -- the standard
//     2-D point-to-segment projection, clamped to the segment (the
//     Minkowski-sum footprint is the classic two-semicircle-capped
//     rectangle, a "stadium").
//   VERTICAL (ax==bx, ay==by): every axis point projects to the SAME
//     (ax,ay), so the footprint is a plain disc of radius `radius`, and the
//     highest point of the swept sphere sits above B (the authored-higher
//     endpoint -- kernels.cuh's stated convention).
// A tilted-axis capsule would need a harder joint (t, z) optimization
// because the nearest-in-xy axis point and the highest-z axis point are no
// longer the same point -- see THEORY.md for the sketch; this project's
// scene generator never produces one, by construction.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool capsule_top_at(float x, float y,
                                               const Capsule& c, float* z_top)
{
    const float r2 = c.radius * c.radius;

    if (c.kind == 1) {
        // VERTICAL: the whole axis projects to one point (c.ax, c.ay).
        const float dx = x - c.ax;
        const float dy = y - c.ay;
        const float d2 = dx * dx + dy * dy;
        if (d2 > r2) return false;                 // outside the disc footprint
        *z_top = c.bz + sqrtf(r2 - d2);            // top hemisphere above B (the high end)
        return true;
    } else {
        // HORIZONTAL: standard clamped point-to-segment projection in 2-D
        // (z does not participate -- it is constant along the axis).
        const float abx = c.bx - c.ax;
        const float aby = c.by - c.ay;
        const float apx = x - c.ax;
        const float apy = y - c.ay;
        const float ab2 = abx * abx + aby * aby;
        // ab2 > 0 always for a real link (A != B); the guard only protects
        // a degenerate zero-length capsule (a plain sphere) from a 0/0.
        float t = (ab2 > 1e-12f) ? (apx * abx + apy * aby) / ab2 : 0.0f;
        t = fminf(fmaxf(t, 0.0f), 1.0f);            // clamp onto the segment
        const float cx = c.ax + t * abx;
        const float cy = c.ay + t * aby;
        const float dx = x - cx;
        const float dy = y - cy;
        const float d2 = dx * dx + dy * dy;
        if (d2 > r2) return false;                  // outside the stadium footprint
        *z_top = c.az + sqrtf(r2 - d2);             // z is constant along this axis
        return true;
    }
}

// ===========================================================================
// point_capsule_distance — the point-capsule distance the catalog bullet
// asks to be derived (THEORY.md "The math" gives the full argument).
//
// A capsule's surface is every point at EXACTLY `radius` from the nearest
// point of segment A->B; the distance from an external point P to the
// capsule is therefore (distance from P to the segment) - radius, floored
// at 0 once P is inside the capsule.
//
// Distance from P to a 3-D segment: minimize |P - (A + t*(B-A))|^2 over
// t in [0,1]. This is a convex (upward) parabola in t (the leading
// coefficient is |AB|^2 >= 0), so its unconstrained minimizer
//     t* = dot(P-A, AB) / dot(AB, AB)
// (found by setting the derivative to zero) is a genuine global minimum;
// clamping t* into [0,1] is valid precisely because the objective is
// convex on the whole real line, so the constrained minimum over [0,1] is
// either the unconstrained minimum (if it already lands in range) or the
// nearer endpoint (if it does not) -- exactly what the clamp computes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float point_capsule_distance(float x, float y, float z,
                                                        const Capsule& c)
{
    const float abx = c.bx - c.ax, aby = c.by - c.ay, abz = c.bz - c.az;
    const float apx = x - c.ax,   apy = y - c.ay,   apz = z - c.az;
    const float ab2 = abx * abx + aby * aby + abz * abz;
    float t = (ab2 > 1e-12f) ? (apx * abx + apy * aby + apz * abz) / ab2 : 0.0f;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    const float cx = c.ax + t * abx, cy = c.ay + t * aby, cz = c.az + t * abz;
    const float dx = x - cx, dy = y - cy, dz = z - cz;
    const float d = sqrtf(dx * dx + dy * dy + dz * dz);
    return fmaxf(d - c.radius, 0.0f);
}

// nearest_robot_capsule_distance — loop the (small, fixed, #pragma-unrolled)
// robot capsule list and return the minimum point_capsule_distance plus
// which capsule achieved it. Shared by human_min_distance_kernel (only
// HUMAN pixels call this) and dense_distance_field_kernel (every pixel).
__device__ __forceinline__ float nearest_robot_capsule_distance(float x, float y, float z,
                                                                int* cap_id)
{
    float best = FLT_MAX;
    int best_id = -1;
#pragma unroll
    for (int k = 0; k < kNumRobotCapsules; ++k) {
        const float d = point_capsule_distance(x, y, z, d_robot_capsules[k]);
        if (d < best) { best = d; best_id = k; }
    }
    *cap_id = best_id;
    return best;
}

// pixel_to_world — shared pixel-center-to-world-XY mapping (kernels.cuh
// SECTION 2's pitch/origin), used by all three kernels below.
__device__ __forceinline__ void pixel_to_world(int px, int py, float* x, float* y)
{
    *x = kCellMinX + (static_cast<float>(px) + 0.5f) * kPixelSizeX;
    *y = kCellMinY + (static_cast<float>(py) + 0.5f) * kPixelSizeY;
}

// ===========================================================================
// render_classify_kernel — stage 1 (+ fused reconstruction inputs).
//
// Thread-to-data mapping: a GRID-STRIDE loop over the kNumPixels linear
// pixel index (the SAXPY-placeholder idiom, reused deliberately -- it is
// correct for any pixel count and lets the launcher pick the grid size for
// occupancy, kernels.cuh's contract notwithstanding). Each thread's own
// pixels are fully independent of every other thread's -- a pure map, no
// shared memory, no atomics.
//
// Per pixel: loop the 8 robot capsules (kernels.cuh's fixed, small,
// #pragma-unrolled list) to find the highest robot surface at (x,y), then
// the 2 human capsules likewise; the taller of the two (or the floor, z=0,
// if neither covers this pixel) is what the camera "sees". Classification
// then asks one question: does the sensed height match what the robot's
// OWN known pose predicts (within kSelfFilterEps)? If yes: ROBOT (the
// self-filter). If the sensed height is above the floor but NOT explained
// by the robot: HUMAN. Otherwise: BACKGROUND.
//
// Memory: d_robot_capsules/d_human_capsules -- __constant__, broadcast to
// every thread (see this file's header comment); d_depth/d_label -- global,
// one coalesced write per thread per pixel (consecutive threads write
// consecutive linear indices, since consecutive `i` map to consecutive
// `px` at fixed `py` -- coalesced, the 33.01/08.01 lesson applied again).
// ===========================================================================
__global__ void render_classify_kernel(float* __restrict__ depth,
                                       uint8_t* __restrict__ label)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (; i < kNumPixels; i += stride) {
        const int px = i % kImageW;
        const int py = i / kImageW;
        float x, y;
        pixel_to_world(px, py, &x, &y);

        // Robot pass: the tallest robot surface at (x,y), if any.
        float robot_top = 0.0f;
        bool robot_hit = false;
#pragma unroll
        for (int k = 0; k < kNumRobotCapsules; ++k) {
            float zt;
            if (capsule_top_at(x, y, d_robot_capsules[k], &zt)) {
                if (!robot_hit || zt > robot_top) { robot_top = zt; robot_hit = true; }
            }
        }
        // Human pass: the tallest human surface at (x,y), if any.
        float human_top = 0.0f;
        bool human_hit = false;
#pragma unroll
        for (int k = 0; k < kNumHumanCapsules; ++k) {
            float zt;
            if (capsule_top_at(x, y, d_human_capsules[k], &zt)) {
                if (!human_hit || zt > human_top) { human_top = zt; human_hit = true; }
            }
        }

        // What the camera actually sees: the taller of {robot, human,
        // floor}. This is where OCCLUSION lives -- if the robot is taller
        // than the human at this (x,y), the human is invisible here, an
        // honest limitation of a single overhead camera discussed in
        // README "Limitations & honesty" and THEORY.md.
        float surface_z = 0.0f;                       // floor default
        if (robot_hit) surface_z = robot_top;
        if (human_hit && human_top > surface_z) surface_z = human_top;

        depth[i] = kCamHeight - surface_z;

        uint8_t lbl;
        if (surface_z <= kFloorEps) {
            lbl = static_cast<uint8_t>(PixelLabel::BACKGROUND);
        } else if (robot_hit && surface_z <= robot_top + kSelfFilterEps) {
            // What we see here matches the robot's OWN known pose -- the
            // self-filter. Real systems need exactly this tolerance band
            // because the commanded/known pose and the true pose never
            // match exactly (PRACTICE.md §1); ours is exact only because
            // this scene is synthetic and the "known pose" IS the true
            // pose -- a documented simplification (README "Limitations").
            lbl = static_cast<uint8_t>(PixelLabel::ROBOT);
        } else {
            lbl = static_cast<uint8_t>(PixelLabel::HUMAN);
        }
        label[i] = lbl;
    }
}

// ===========================================================================
// human_min_distance_kernel — stage 3: MAP (per-pixel candidate distance)
// + REDUCE (block-level shared-memory tree reduction to a per-block min).
//
// "One thread per human point" is realized the same way 33.01's
// grid-stride map realizes "one thread per element": every thread in the
// grid strides over the WHOLE pixel array, and a pixel that is not HUMAN
// costs one label read and nothing else (an early skip inside the loop
// body, not a separate kernel/branch). The alternative -- first stream-
// compacting HUMAN pixels into a dense point list (e.g. via atomicAdd on a
// counter, or Thrust's copy_if) -- avoids visiting BACKGROUND/ROBOT pixels
// a second time, at the cost of a materialized point buffer and, for the
// atomic version, a NON-DETERMINISTIC point ORDER (harmless here, since
// the downstream reduction is a MIN -- order-invariant -- but it would
// matter for anything order-sensitive, e.g. ICP). README Exercises name
// both alternatives; this project keeps the simpler grid-stride-and-guard
// form because the label image already makes "which pixels are human"
// a single coalesced read, and 40,000 pixels is cheap to fully scan.
//
// The reduction itself is the CANONICAL shared-memory tree pattern (the
// one taught in every CUDA reduction tutorial and the basis of CUB's
// BlockReduce): each thread first folds its OWN strided pixels down to one
// local (distance, capsule_id) pair (this is what makes a grid-stride
// reduction efficient -- most of the reduction work happens with zero
// synchronization, purely in registers), then a power-of-two tree of
// __syncthreads()-separated comparisons folds the block's blockDim.x
// local pairs down to one. Distance-MIN is COMMUTATIVE AND ASSOCIATIVE
// EXACTLY in IEEE-754 (unlike a SUM, whose reassociation changes rounding)
// -- so, unlike a parallel sum, this reduction's result does not depend on
// the order pixels are combined in, and the GPU result matches the CPU
// oracle's sequential scan far more tightly than a summed reduction would
// (THEORY.md "Numerical considerations" makes this point explicitly).
//
// Grid finishes on the HOST: this kernel writes one (distance, capsule_id)
// pair per BLOCK (a few hundred floats/ints, at most), and
// launch_human_min_distance() below does the final tiny linear scan on the
// host -- the same "host finishes a trivially small reduction" choice
// 08.01 makes for its softmin weights, for the same reason: the whole
// algorithm stays readable in one place, and the cost is negligible next
// to the 40,000-thread kernel that did the real work.
// ===========================================================================
constexpr int kReduceBlockSize = 256;   // warp multiple, standard repo default

__global__ void human_min_distance_kernel(const float* __restrict__ depth,
                                          const uint8_t* __restrict__ label,
                                          float* __restrict__ block_min_dist,
                                          int* __restrict__ block_min_id)
{
    __shared__ float s_dist[kReduceBlockSize];
    __shared__ int   s_id[kReduceBlockSize];

    const int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Phase 1 (registers only): this thread's own running minimum over its
    // strided share of the pixel array.
    float best_d = FLT_MAX;
    int best_id = -1;
    for (; i < kNumPixels; i += stride) {
        if (label[i] == static_cast<uint8_t>(PixelLabel::HUMAN)) {
            const int px = i % kImageW;
            const int py = i / kImageW;
            float x, y;
            pixel_to_world(px, py, &x, &y);
            const float z = kCamHeight - depth[i];   // reconstruct THIS pixel's 3-D point
            int cid;
            const float d = nearest_robot_capsule_distance(x, y, z, &cid);
            if (d < best_d) { best_d = d; best_id = cid; }
        }
    }
    s_dist[tid] = best_d;
    s_id[tid]   = best_id;
    __syncthreads();

    // Phase 2 (shared memory): the classic power-of-two tree reduction.
    // Each surviving half compares itself against the half being folded
    // away and keeps the smaller pair; blockDim.x is a compile-time-chosen
    // power of two (kReduceBlockSize = 256) so the loop terminates exactly
    // at s == 0 with no ragged remainder to special-case.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_dist[tid + s] < s_dist[tid]) {
            s_dist[tid] = s_dist[tid + s];
            s_id[tid]   = s_id[tid + s];
        }
        __syncthreads();   // every thread must see this round's result before the next
    }

    if (tid == 0) {
        block_min_dist[blockIdx.x] = s_dist[0];
        block_min_id[blockIdx.x]   = s_id[0];
    }
}

// reduce_num_blocks — grid size for human_min_distance_kernel: enough
// blocks to cover every pixel once with no stride at all (kNumPixels /
// kReduceBlockSize, rounded up), capped at 256 blocks (this repo's usual
// "plenty to fill any current GPU's SMs many times over" ceiling -- see
// 08.01/07.09's identical reasoning); the grid-stride loop inside the
// kernel absorbs anything the cap leaves uncovered.
int reduce_num_blocks()
{
    int blocks = (kNumPixels + kReduceBlockSize - 1) / kReduceBlockSize;
    if (blocks > 256) blocks = 256;
    return blocks;
}

// ===========================================================================
// dense_distance_field_kernel — the bullet's dense variant: EVERY pixel's
// distance to the nearest robot capsule, reusing nearest_robot_capsule_
// distance verbatim. A pure map (no reduction) -- one thread's output never
// depends on any other thread's, so this kernel is the simplest of the
// three despite doing the same per-pixel arithmetic as stage 3's inner
// loop. Run for exactly ONE frame per demo (main.cu picks the measured
// closest-approach frame) -- the visual clearance-field artifact, not a
// per-tick cost.
// ===========================================================================
__global__ void dense_distance_field_kernel(const float* __restrict__ depth,
                                            float* __restrict__ field)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (; i < kNumPixels; i += stride) {
        const int px = i % kImageW;
        const int py = i / kImageW;
        float x, y;
        pixel_to_world(px, py, &x, &y);
        const float z = kCamHeight - depth[i];
        int cid;
        field[i] = nearest_robot_capsule_distance(x, y, z, &cid);
    }
}

// ===========================================================================
// Host launchers (declared in kernels.cuh) — own the grid/block math and
// the mandatory post-launch error check (CLAUDE.md §6.1 rule 7), same
// shape as every other project's launch_*() wrapper.
// ===========================================================================
namespace {
// grid_for — the repo-standard "ceil(n/block), capped at 4096" grid-stride
// launch geometry (08.01/07.09's identical formula), factored once here
// since all three kernels share it.
int grid_for(int n, int block)
{
    int g = (n + block - 1) / block;
    if (g > 4096) g = 4096;
    return g;
}
} // namespace

void upload_capsules(const Capsule robot[kNumRobotCapsules],
                     const Capsule human[kNumHumanCapsules])
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_robot_capsules, robot,
                                  sizeof(Capsule) * kNumRobotCapsules));
    CUDA_CHECK(cudaMemcpyToSymbol(d_human_capsules, human,
                                  sizeof(Capsule) * kNumHumanCapsules));
}

void launch_render_classify(float* d_depth, uint8_t* d_label)
{
    const int block = 256;
    const int grid = grid_for(kNumPixels, block);
    render_classify_kernel<<<grid, block>>>(d_depth, d_label);
    CUDA_CHECK_LAST_ERROR("render_classify_kernel launch");
}

void launch_human_min_distance(const float* d_depth, const uint8_t* d_label,
                               float* d_block_mins, int* d_block_ids,
                               float* out_dmin, int* out_closest_capsule)
{
    const int blocks = reduce_num_blocks();
    human_min_distance_kernel<<<blocks, kReduceBlockSize>>>(
        d_depth, d_label, d_block_mins, d_block_ids);
    CUDA_CHECK_LAST_ERROR("human_min_distance_kernel launch");

    // The tiny host-side finish (see this file's header comment): download
    // at most 256 (distance, id) pairs and scan them linearly.
    std::vector<float> h_mins(static_cast<size_t>(blocks));
    std::vector<int>   h_ids(static_cast<size_t>(blocks));
    CUDA_CHECK(cudaMemcpy(h_mins.data(), d_block_mins, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ids.data(), d_block_ids, blocks * sizeof(int),
                          cudaMemcpyDeviceToHost));

    float best = FLT_MAX;
    int best_id = -1;
    for (int b = 0; b < blocks; ++b) {
        if (h_mins[static_cast<size_t>(b)] < best) {
            best = h_mins[static_cast<size_t>(b)];
            best_id = h_ids[static_cast<size_t>(b)];
        }
    }
    *out_dmin = best;
    *out_closest_capsule = best_id;
}

void launch_dense_distance_field(const float* d_depth, float* d_field)
{
    const int block = 256;
    const int grid = grid_for(kNumPixels, block);
    dense_distance_field_kernel<<<grid, block>>>(d_depth, d_field);
    CUDA_CHECK_LAST_ERROR("dense_distance_field_kernel launch");
}
