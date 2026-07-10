// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 11.01
//                     GPU LiDAR simulator: hand-built BVH raycasting + beam
//                     divergence, intensity, dropout noise
//
// One job in this project (declared in kernels.cuh): lidar_raycast_cpu, the
// ORACLE twin of kernels.cu's GPU kernel — identical BVH traversal,
// identical Möller–Trumbore, identical divergence/radiometry/dropout math,
// sequential over beams, plain single-threaded C++, no CUDA headers
// anywhere. main.cu runs this against the GPU kernel on the full demo frame
// (the §5 GPU-vs-CPU gate) AND runs main.cu's ANALYTIC verification gates
// (ground-plane range, inverse-square intensity, dropout-rate statistics)
// EXCLUSIVELY through this function — see kernels.cuh's long comment on
// lidar_raycast_cpu for why that is sufficient (short version: the §5 gate
// already proves the GPU kernel matches THIS function; validating the
// physics model against one simpler, easier-to-audit implementation avoids
// a proliferation of tiny one-off kernel launches).
//
// Every function below is a deliberate line-by-line twin of the
// corresponding __device__ function in kernels.cu — diff the two files to
// see exactly what parallelization changed (spoiler: nothing but the
// __device__ qualifiers and F3 x/y/z vs the CUDA-provided float3 — this file
// defines its OWN small F3, byte-for-byte the same shape, to stay entirely
// free of any CUDA header, CLAUDE.md §5's "reference_cpu.cpp never depends
// on nvcc" rule).
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong, and
// then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // shared layouts, constants (kBvhStackSize, kPiD), and this file's prototype

#include <cmath>         // std::sin, std::cos, std::sqrt, std::log, std::fabs
#include <cstdint>       // uint32_t for the per-beam RNG stream

// ---------------------------------------------------------------------------
// F3 — the host twin of kernels.cu's F3. Same shape, same field names, no
// __device__ anywhere — plain host inline functions the optimizer happily
// inlines just as aggressively as nvcc inlines __forceinline__ device code.
// ---------------------------------------------------------------------------
struct F3 { float x, y, z; };

static inline F3 f3(float x, float y, float z) { return F3{ x, y, z }; }
static inline F3 f3_add(F3 a, F3 b) { return F3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline F3 f3_sub(F3 a, F3 b) { return F3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline F3 f3_scale(F3 a, float s) { return F3{ a.x * s, a.y * s, a.z * s }; }
static inline float f3_dot(F3 a, F3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline F3 f3_cross(F3 a, F3 b)
{
    return F3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static inline F3 f3_normalize(F3 a)
{
    const float inv_len = 1.0f / std::sqrt(f3_dot(a, a));   // see kernels.cu's F3 note: never zero at any call site
    return f3_scale(a, inv_len);
}

// aabb_hit — see kernels.cu's header comment for the full derivation (the
// IEEE-754 divide-by-zero-is-safe argument applies identically on the host:
// standard C++ float division follows IEEE 754, 1.0f/0.0f is +inf, not UB).
static inline bool aabb_hit(const BvhNode& node, F3 origin, F3 inv_dir, float tmin, float tmax)
{
    float t1 = (node.aabb_min[0] - origin.x) * inv_dir.x;
    float t2 = (node.aabb_max[0] - origin.x) * inv_dir.x;
    float lo = std::fmin(t1, t2), hi = std::fmax(t1, t2);

    t1 = (node.aabb_min[1] - origin.y) * inv_dir.y;
    t2 = (node.aabb_max[1] - origin.y) * inv_dir.y;
    lo = std::fmax(lo, std::fmin(t1, t2));
    hi = std::fmin(hi, std::fmax(t1, t2));

    t1 = (node.aabb_min[2] - origin.z) * inv_dir.z;
    t2 = (node.aabb_max[2] - origin.z) * inv_dir.z;
    lo = std::fmax(lo, std::fmin(t1, t2));
    hi = std::fmin(hi, std::fmax(t1, t2));

    lo = std::fmax(lo, tmin);
    hi = std::fmin(hi, tmax);
    return hi >= lo;
}

// moller_trumbore — see kernels.cu's header comment for the derivation.
static inline bool moller_trumbore(const Triangle& tri, F3 origin, F3 dir,
                                   float tmin, float tmax, float* out_t)
{
    const float kEps = 1e-8f;

    F3 v0 = f3(tri.v0[0], tri.v0[1], tri.v0[2]);
    F3 e1 = f3_sub(f3(tri.v1[0], tri.v1[1], tri.v1[2]), v0);
    F3 e2 = f3_sub(f3(tri.v2[0], tri.v2[1], tri.v2[2]), v0);

    F3 pvec = f3_cross(dir, e2);
    const float det = f3_dot(e1, pvec);
    if (std::fabs(det) < kEps) return false;
    const float inv_det = 1.0f / det;

    F3 tvec = f3_sub(origin, v0);
    const float u = f3_dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    F3 qvec = f3_cross(tvec, e1);
    const float v = f3_dot(dir, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    const float t = f3_dot(e2, qvec) * inv_det;
    if (t < tmin || t > tmax) return false;

    *out_t = t;
    return true;
}

// intersect_bvh — see kernels.cu's header comment for the depth-bound
// argument that makes the fixed-size stack correct (not just "probably
// enough"). Same small-stack traversal, sequential, one triangle scan per
// leaf, both children pushed unordered.
static inline bool intersect_bvh(const BvhNode* nodes, const int* tri_indices, const Triangle* tris,
                                 F3 origin, F3 dir, float tmin, float tmax,
                                 float* out_t, int* out_tri)
{
    const F3 inv_dir = f3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    int stack[kBvhStackSize];
    int sp = 0;
    stack[sp++] = 0;

    float best_t = tmax;
    int best_tri = -1;

    while (sp > 0) {
        const int node_idx = stack[--sp];
        const BvhNode node = nodes[node_idx];
        if (!aabb_hit(node, origin, inv_dir, tmin, best_t)) continue;

        if (node.tri_count > 0) {
            for (int i = 0; i < node.tri_count; ++i) {
                const int ti = tri_indices[node.left_first + i];
                float t;
                if (moller_trumbore(tris[ti], origin, dir, tmin, best_t, &t)) {
                    best_t = t;
                    best_tri = ti;
                }
            }
        } else {
            if (sp + 2 <= kBvhStackSize) {
                stack[sp++] = node.left_first;
                stack[sp++] = node.left_first + 1;
            }
        }
    }

    if (best_tri < 0) return false;
    *out_t = best_t;
    *out_tri = best_tri;
    return true;
}

// make_basis — see kernels.cu's header comment.
static inline void make_basis(F3 n, F3* u, F3* v)
{
    const F3 helper = (std::fabs(n.x) < 0.9f) ? f3(1.0f, 0.0f, 0.0f) : f3(0.0f, 1.0f, 0.0f);
    *u = f3_normalize(f3_cross(helper, n));
    *v = f3_cross(n, *u);
}

// ---------------------------------------------------------------------------
// Per-beam RNG twin — same xorshift32 + Box–Muller algorithm as kernels.cu,
// same double-precision transcendental step, same per-beam seeding. This is
// the SAME generator 08.01 uses (reseeded per beam instead of per tick),
// duplicated here in plain C++ so the host oracle needs no CUDA headers.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static inline float uniform01(uint32_t& state)
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

static inline float gaussian(uint32_t& state, float sigma)
{
    const double u1 = static_cast<double>(uniform01(state));
    const double u2 = static_cast<double>(uniform01(state));
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPiD * u2);
    return sigma * static_cast<float>(z);
}

static inline uint32_t per_beam_seed(uint32_t base_seed, int beam_idx)
{
    uint32_t s = base_seed + 1000003u * static_cast<uint32_t>(beam_idx + 1);
    if (s == 0u) s = 1u;
    return s;
}

// ===========================================================================
// cpu_simulate_beam — the host twin of kernels.cu's simulate_beam(). Every
// comment explaining WHY a step is written the way it is lives in
// kernels.cu; this file only re-states WHERE the two differ (nowhere, in
// the math — see the file header).
// ===========================================================================
static void cpu_simulate_beam(int beam_idx,
                              const Triangle* tris, const Material* materials,
                              const BvhNode* nodes, const int* tri_indices,
                              const SensorConfig& cfg, const SensorPose& pose,
                              float* out_range, float* out_intensity,
                              int* out_hit, int* out_dropped)
{
    const int channel = beam_idx / cfg.azimuth_steps;
    const int az_idx = beam_idx % cfg.azimuth_steps;

    const double frac = (cfg.channels > 1)
        ? static_cast<double>(channel) / static_cast<double>(cfg.channels - 1)
        : 0.0;
    const double elevation = static_cast<double>(cfg.elevation_min_rad)
        + frac * (static_cast<double>(cfg.elevation_max_rad) - static_cast<double>(cfg.elevation_min_rad));
    const double azimuth = static_cast<double>(cfg.azimuth_start_rad)
        + static_cast<double>(az_idx) * (2.0 * kPiD / static_cast<double>(cfg.azimuth_steps));

    const double ce = std::cos(elevation), se = std::sin(elevation);
    const double ca = std::cos(azimuth), sa = std::sin(azimuth);
    const F3 dir_sensor = f3(static_cast<float>(ce * ca), static_cast<float>(ce * sa), static_cast<float>(se));

    const F3 origin_world = f3(pose.t[0], pose.t[1], pose.t[2]);
    const F3 dir_center = f3(
        pose.R[0] * dir_sensor.x + pose.R[1] * dir_sensor.y + pose.R[2] * dir_sensor.z,
        pose.R[3] * dir_sensor.x + pose.R[4] * dir_sensor.y + pose.R[5] * dir_sensor.z,
        pose.R[6] * dir_sensor.x + pose.R[7] * dir_sensor.y + pose.R[8] * dir_sensor.z);

    float best_t = cfg.range_max_m;
    int best_tri = -1;
    F3 best_dir = dir_center;

    float t;
    int ti;
    if (intersect_bvh(nodes, tri_indices, tris, origin_world, dir_center, cfg.range_min_m, best_t, &t, &ti)) {
        best_t = t; best_tri = ti; best_dir = dir_center;
    }

    if (cfg.subray_count > 0) {
        F3 u, v;
        make_basis(dir_center, &u, &v);
        const double half = static_cast<double>(cfg.divergence_half_angle_rad);
        const double ch = std::cos(half), sh = std::sin(half);
        for (int k = 0; k < cfg.subray_count; ++k) {
            const double phi = static_cast<double>(k) * (2.0 * kPiD / static_cast<double>(cfg.subray_count));
            const float ox = static_cast<float>(std::cos(phi) * sh);
            const float oy = static_cast<float>(std::sin(phi) * sh);
            const F3 sub_dir = f3_normalize(f3_add(
                f3_scale(dir_center, static_cast<float>(ch)),
                f3_add(f3_scale(u, ox), f3_scale(v, oy))));
            if (intersect_bvh(nodes, tri_indices, tris, origin_world, sub_dir, cfg.range_min_m, best_t, &t, &ti)) {
                best_t = t; best_tri = ti; best_dir = sub_dir;
            }
        }
    }

    if (best_tri < 0) {
        *out_hit = 0; *out_range = 0.0f; *out_intensity = 0.0f; *out_dropped = 0;
        return;
    }
    *out_hit = 1;

    const Triangle tri = tris[best_tri];
    const F3 v0 = f3(tri.v0[0], tri.v0[1], tri.v0[2]);
    const F3 v1 = f3(tri.v1[0], tri.v1[1], tri.v1[2]);
    const F3 v2 = f3(tri.v2[0], tri.v2[1], tri.v2[2]);
    const F3 n = f3_normalize(f3_cross(f3_sub(v1, v0), f3_sub(v2, v0)));
    const float cos_theta = std::fabs(f3_dot(n, best_dir));
    const float range_raw = best_t;
    const float albedo = materials[tri.material_id].albedo;
    float intensity = cfg.intensity_gain * albedo * cos_theta / (range_raw * range_raw);
    intensity = std::fmin(std::fmax(intensity, 0.0f), 1.0f);

    uint32_t rng = per_beam_seed(cfg.seed, beam_idx);
    const float u_dropout = uniform01(rng);
    const float sigma = cfg.range_noise_base_m + cfg.range_noise_per_m * range_raw;
    const float noise = gaussian(rng, sigma);

    float p_drop = cfg.dropout_base
                 + cfg.dropout_range_coeff * (range_raw / cfg.range_max_m)
                 + cfg.dropout_incidence_coeff * (1.0f - cos_theta);
    p_drop = std::fmin(std::fmax(p_drop, 0.0f), 1.0f);

    *out_dropped = (u_dropout < p_drop) ? 1 : 0;
    *out_range = range_raw + noise;
    *out_intensity = intensity;
}

// ---------------------------------------------------------------------------
// lidar_raycast_cpu — sequential over beams, otherwise identical to the GPU
// kernel's per-thread work (kernels.cuh's declared contract).
// ---------------------------------------------------------------------------
void lidar_raycast_cpu(const Triangle* tris, const Material* materials,
                       const BvhNode* nodes, const int* tri_indices,
                       const SensorConfig& cfg, const SensorPose& pose,
                       float* range, float* intensity,
                       int* hit, int* dropped)
{
    const int num_beams = cfg.channels * cfg.azimuth_steps;
    for (int beam = 0; beam < num_beams; ++beam) {
        cpu_simulate_beam(beam, tris, materials, nodes, tri_indices, cfg, pose,
                          &range[beam], &intensity[beam], &hit[beam], &dropped[beam]);
    }
}
