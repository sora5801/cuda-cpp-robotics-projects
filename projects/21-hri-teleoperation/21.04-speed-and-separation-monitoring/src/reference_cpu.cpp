// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 21.04
//                     Speed-and-separation monitoring (didactic, NOT
//                     certified -- see kernels.cuh's header comment)
//
// Independent, single-threaded, line-by-line-twin reimplementations of the
// three GPU kernels in kernels.cu (CLAUDE.md §5): render_classify_cpu,
// human_min_distance_cpu, dense_distance_field_cpu. main.cu's VERIFY stage
// runs both paths on identical capsule geometry and requires agreement --
// EXACT for the integer pixel labels, tight FP32 tolerance for depth/
// distance values (the two paths use the same formulas, so the only
// expected divergence is FP32 rounding-order/intrinsic differences, not a
// missing-feature or indexing bug).
//
// Why the near-duplication of kernels.cu's device functions: this file is
// the CORRECTNESS ORACLE (a reader must be able to trust it BY EYE), and an
// oracle that secretly calls the code it is supposed to be checking proves
// nothing. Diffing capsule_top_at_cpu against kernels.cu's capsule_top_at
// shows they are twins (same math, sqrtf vs std::sqrt, fminf/fmaxf vs plain
// ternary clamps) -- the same "deliberate, documented duplication" 08.01's
// cartpole_deriv/rk4_step/stage_cost make.
//
// This file is compiled by the HOST compiler (cl.exe), never nvcc -- no
// CUDA headers, no __device__, no cleverness (CLAUDE.md §5).
//
// Read this after: kernels.cu -- then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cfloat>    // FLT_MAX
#include <cmath>     // std::sqrt

// ---------------------------------------------------------------------------
// Host twins of kernels.cu's __device__ geometry functions. `static`:
// file-local, exactly like kernels.cu keeps its __device__ helpers
// undeclared in the shared header -- neither is part of the project's
// cross-file contract, only the three top-level oracle functions are
// (declared in kernels.cuh SECTION 8).
// ---------------------------------------------------------------------------

static bool capsule_top_at_cpu(float x, float y, const Capsule& c, float* z_top)
{
    const float r2 = c.radius * c.radius;
    if (c.kind == 1) {
        const float dx = x - c.ax, dy = y - c.ay;
        const float d2 = dx * dx + dy * dy;
        if (d2 > r2) return false;
        *z_top = c.bz + std::sqrt(r2 - d2);
        return true;
    } else {
        const float abx = c.bx - c.ax, aby = c.by - c.ay;
        const float apx = x - c.ax,   apy = y - c.ay;
        const float ab2 = abx * abx + aby * aby;
        float t = (ab2 > 1e-12f) ? (apx * abx + apy * aby) / ab2 : 0.0f;
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);   // clamp, spelled the std:: way
        const float cx = c.ax + t * abx, cy = c.ay + t * aby;
        const float dx = x - cx, dy = y - cy;
        const float d2 = dx * dx + dy * dy;
        if (d2 > r2) return false;
        *z_top = c.az + std::sqrt(r2 - d2);
        return true;
    }
}

static float point_capsule_distance_cpu(float x, float y, float z, const Capsule& c)
{
    const float abx = c.bx - c.ax, aby = c.by - c.ay, abz = c.bz - c.az;
    const float apx = x - c.ax,   apy = y - c.ay,   apz = z - c.az;
    const float ab2 = abx * abx + aby * aby + abz * abz;
    float t = (ab2 > 1e-12f) ? (apx * abx + apy * aby + apz * abz) / ab2 : 0.0f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const float cx = c.ax + t * abx, cy = c.ay + t * aby, cz = c.az + t * abz;
    const float dx = x - cx, dy = y - cy, dz = z - cz;
    const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
    const float out = d - c.radius;
    return out > 0.0f ? out : 0.0f;
}

static float nearest_robot_capsule_distance_cpu(float x, float y, float z,
                                                const Capsule robot[kNumRobotCapsules],
                                                int* cap_id)
{
    float best = FLT_MAX;
    int best_id = -1;
    for (int k = 0; k < kNumRobotCapsules; ++k) {
        const float d = point_capsule_distance_cpu(x, y, z, robot[k]);
        if (d < best) { best = d; best_id = k; }
    }
    *cap_id = best_id;
    return best;
}

static void pixel_to_world_cpu(int px, int py, float* x, float* y)
{
    *x = kCellMinX + (static_cast<float>(px) + 0.5f) * kPixelSizeX;
    *y = kCellMinY + (static_cast<float>(py) + 0.5f) * kPixelSizeY;
}

// ---------------------------------------------------------------------------
// render_classify_cpu — the oracle twin of render_classify_kernel. Two
// nested loops (py, px) instead of a grid-stride loop, but the SAME
// per-pixel logic in the SAME order (robot pass, then human pass, then the
// three-way classification) -- a reader can walk this function and know
// exactly what the kernel computes, thread-by-thread, without CUDA syntax
// in the way.
// ---------------------------------------------------------------------------
void render_classify_cpu(const Capsule robot[kNumRobotCapsules],
                         const Capsule human[kNumHumanCapsules],
                         float* depth, uint8_t* label)
{
    for (int py = 0; py < kImageH; ++py) {
        for (int px = 0; px < kImageW; ++px) {
            const int i = py * kImageW + px;
            float x, y;
            pixel_to_world_cpu(px, py, &x, &y);

            float robot_top = 0.0f;
            bool robot_hit = false;
            for (int k = 0; k < kNumRobotCapsules; ++k) {
                float zt;
                if (capsule_top_at_cpu(x, y, robot[k], &zt)) {
                    if (!robot_hit || zt > robot_top) { robot_top = zt; robot_hit = true; }
                }
            }
            float human_top = 0.0f;
            bool human_hit = false;
            for (int k = 0; k < kNumHumanCapsules; ++k) {
                float zt;
                if (capsule_top_at_cpu(x, y, human[k], &zt)) {
                    if (!human_hit || zt > human_top) { human_top = zt; human_hit = true; }
                }
            }

            float surface_z = 0.0f;
            if (robot_hit) surface_z = robot_top;
            if (human_hit && human_top > surface_z) surface_z = human_top;

            depth[i] = kCamHeight - surface_z;

            uint8_t lbl;
            if (surface_z <= kFloorEps) {
                lbl = static_cast<uint8_t>(PixelLabel::BACKGROUND);
            } else if (robot_hit && surface_z <= robot_top + kSelfFilterEps) {
                lbl = static_cast<uint8_t>(PixelLabel::ROBOT);
            } else {
                lbl = static_cast<uint8_t>(PixelLabel::HUMAN);
            }
            label[i] = lbl;
        }
    }
}

// ---------------------------------------------------------------------------
// human_min_distance_cpu — the oracle twin of human_min_distance_kernel's
// MAP+REDUCE, collapsed to a single sequential scan (no blocks, no shared
// memory -- a CPU has neither). Because point-capsule distance MIN is
// exactly commutative/associative, this sequential scan and the GPU's
// tree-then-host-scan reduction compute the SAME real number up to FP32
// rounding, not merely a statistically similar one (THEORY.md "Numerical
// considerations" makes this the centerpiece of why the VERIFY tolerance
// here can be so much tighter than a summed reduction would allow).
// ---------------------------------------------------------------------------
void human_min_distance_cpu(const float* depth, const uint8_t* label,
                            const Capsule robot[kNumRobotCapsules],
                            float* out_dmin, int* out_closest_capsule)
{
    float best = FLT_MAX;
    int best_id = -1;
    for (int i = 0; i < kNumPixels; ++i) {
        if (label[i] == static_cast<uint8_t>(PixelLabel::HUMAN)) {
            const int px = i % kImageW;
            const int py = i / kImageW;
            float x, y;
            pixel_to_world_cpu(px, py, &x, &y);
            const float z = kCamHeight - depth[i];
            int cid;
            const float d = nearest_robot_capsule_distance_cpu(x, y, z, robot, &cid);
            if (d < best) { best = d; best_id = cid; }
        }
    }
    *out_dmin = best;
    *out_closest_capsule = best_id;
}

// ---------------------------------------------------------------------------
// dense_distance_field_cpu — the oracle twin of dense_distance_field_kernel:
// every pixel's distance to the nearest robot capsule, no HUMAN-label guard.
// ---------------------------------------------------------------------------
void dense_distance_field_cpu(const float* depth,
                              const Capsule robot[kNumRobotCapsules],
                              float* field)
{
    for (int i = 0; i < kNumPixels; ++i) {
        const int px = i % kImageW;
        const int py = i / kImageW;
        float x, y;
        pixel_to_world_cpu(px, py, &x, &y);
        const float z = kCamHeight - depth[i];
        int cid;
        field[i] = nearest_robot_capsule_distance_cpu(x, y, z, robot, &cid);
    }
}
