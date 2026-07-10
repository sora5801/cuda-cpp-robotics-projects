// ===========================================================================
// kernels.cu — GPU implementation for project 11.01
//              GPU LiDAR simulator: hand-built BVH raycasting + beam
//              divergence, intensity, dropout noise
//
// The big idea
// ------------
// One GPU thread per BEAM. Each thread walks the SAME flattened BVH — but
// down a DIFFERENT path, because every beam points a different direction.
// That is the project's central new idea beyond the repo's usual
// thread-per-problem shape (33.01/09.01/08.01/02.06): those kernels read
// FLAT arrays where neighboring threads touch neighboring memory; this one
// reads a TREE, where neighboring threads (same warp) can walk completely
// different root-to-leaf paths. The result — DIVERGENT TRAVERSAL — is
// measured and explained in THEORY.md "The GPU mapping"; this file's job is
// to make the traversal itself simple enough to read start to finish.
//
// What is NEW here beyond every earlier project:
//   * a TREE walk with a small fixed-size per-thread STACK (kBvhStackSize,
//     kernels.cuh) instead of a flat loop or a closed-form update;
//   * Möller–Trumbore ray/triangle intersection (derived in THEORY.md "The
//     math"), the first true 3-D geometry primitive in the repo's control/
//     estimation projects seen so far;
//   * three independent per-beam EFFECT MODELS (divergence, radiometry,
//     dropout+noise) layered on top of the geometric raycast — each one
//     small, but their composition is the whole point of the catalog bullet.
//
// All model constants and layouts come from kernels.cuh — the single source
// shared with the CPU oracle; every function below is a deliberate
// line-by-line twin of the corresponding function in reference_cpu.cpp
// (CLAUDE.md §4's documented-duplication rule, applied here to functions,
// not just whole files, exactly as 08.01/02.06/05.01 do for their dynamics/
// transform/integration functions).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// F3 — a minimal float3 stand-in local to THIS file (kept private, not in
// kernels.cuh: the header shares LAYOUTS across the host/device boundary,
// not math helpers — CLAUDE.md §4's duplication rule again). reference_cpu.cpp
// defines an identical-looking struct of its own; only the __device__
// qualifiers differ between the two files.
// ---------------------------------------------------------------------------
struct F3 { float x, y, z; };

__device__ __forceinline__ F3 f3(float x, float y, float z) { return F3{ x, y, z }; }
__device__ __forceinline__ F3 f3_add(F3 a, F3 b) { return F3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
__device__ __forceinline__ F3 f3_sub(F3 a, F3 b) { return F3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
__device__ __forceinline__ F3 f3_scale(F3 a, float s) { return F3{ a.x * s, a.y * s, a.z * s }; }
__device__ __forceinline__ float f3_dot(F3 a, F3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ F3 f3_cross(F3 a, F3 b)
{
    return F3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
__device__ __forceinline__ F3 f3_normalize(F3 a)
{
    // No degenerate-length guard: every vector normalized below is either a
    // unit direction built from sin/cos (never zero) or a cross product of
    // two non-parallel edges/basis vectors (never zero for a valid,
    // non-degenerate triangle/frame) — see the call sites.
    const float inv_len = 1.0f / sqrtf(f3_dot(a, a));
    return f3_scale(a, inv_len);
}

// ---------------------------------------------------------------------------
// aabb_hit — the classic "slab test": does ray (origin, dir) enter this
// node's box within [tmin, tmax]? inv_dir is 1/dir, computed ONCE per ray
// by the caller (three divisions instead of six).
//
// Division-by-zero safety: if dir.x == 0 (a ray exactly parallel to the x
// slab), inv_dir.x is IEEE-754 +-infinity, NOT undefined behavior — this
// project never passes --use_fast_math (CLAUDE.md §5), so the compiler
// preserves standard IEEE division semantics on both host and device. A
// slab with zero-width extent along that axis then correctly reduces to
// "must lie exactly on the plane", which +-inf's arithmetic handles for
// free — the classic branch-free AABB test relies on exactly this
// (THEORY.md "Numerical considerations" names it explicitly).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool aabb_hit(const BvhNode& node, F3 origin, F3 inv_dir,
                                         float tmin, float tmax)
{
    float t1 = (node.aabb_min[0] - origin.x) * inv_dir.x;
    float t2 = (node.aabb_max[0] - origin.x) * inv_dir.x;
    float lo = fminf(t1, t2), hi = fmaxf(t1, t2);

    t1 = (node.aabb_min[1] - origin.y) * inv_dir.y;
    t2 = (node.aabb_max[1] - origin.y) * inv_dir.y;
    lo = fmaxf(lo, fminf(t1, t2));
    hi = fminf(hi, fmaxf(t1, t2));

    t1 = (node.aabb_min[2] - origin.z) * inv_dir.z;
    t2 = (node.aabb_max[2] - origin.z) * inv_dir.z;
    lo = fmaxf(lo, fminf(t1, t2));
    hi = fminf(hi, fmaxf(t1, t2));

    lo = fmaxf(lo, tmin);
    hi = fminf(hi, tmax);
    return hi >= lo;
}

// ---------------------------------------------------------------------------
// moller_trumbore — ray/triangle intersection (Möller & Trumbore, 1997).
//
// THEORY.md "The math" derives this from first principles; the short
// version: express the hit point in the triangle's own (u, v) edge basis,
//     P(t) = origin + t*dir = v0 + u*e1 + v*e2      (e1 = v1-v0, e2 = v2-v0)
// which is 3 equations (x,y,z) in 3 unknowns (t,u,v) — solved here via
// Cramer's rule, restructured (the textbook trick) so every determinant is
// a single triple product computed with one cross and one dot, and the
// early-out tests (u, v, u+v, t ranges) can each reject BEFORE the next
// determinant is even computed.
//
// Returns true and writes *out_t iff the ray hits the triangle with
// tmin <= t <= tmax (barycentric u,v in [0,1], u+v<=1 — inside the
// triangle, not just its plane).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool moller_trumbore(const Triangle& tri, F3 origin, F3 dir,
                                                 float tmin, float tmax, float* out_t)
{
    const float kEps = 1e-8f;   // guards the parallel-ray / degenerate-triangle case

    F3 v0 = f3(tri.v0[0], tri.v0[1], tri.v0[2]);
    F3 e1 = f3_sub(f3(tri.v1[0], tri.v1[1], tri.v1[2]), v0);
    F3 e2 = f3_sub(f3(tri.v2[0], tri.v2[1], tri.v2[2]), v0);

    F3 pvec = f3_cross(dir, e2);
    const float det = f3_dot(e1, pvec);
    // |det| ~ 0 means dir lies in the triangle's plane (or the triangle is
    // degenerate) — no well-defined single intersection point either way.
    // NOT culling det<0 (back-face culling) on purpose: this project's
    // intensity model uses |cos(incidence)| and does not care which side of
    // the mesh a ray approaches from (kernels.cuh's winding-order note).
    if (fabsf(det) < kEps) return false;
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

// ---------------------------------------------------------------------------
// intersect_bvh — find the NEAREST triangle hit along (origin, dir) within
// [tmin, tmax]. SMALL-STACK traversal (not stackless): kernels.cuh's
// median-split-by-COUNT build guarantees tree depth <=
// ceil(log2(num_triangles / kBvhLeafSize)) regardless of scene geometry —
// for this project's ~2,264-triangle warehouse and kBvhLeafSize=4 that is
// ceil(log2(566)) = 10; kBvhStackSize=64 is therefore >5x headroom, not a
// "probably enough" guess (THEORY.md "The algorithm" proves the bound).
// A skip-pointer / "threaded" stackless scheme is a real alternative (used
// in several production BVH traversers) that trades this array for two
// extra ints per node; README Exercise names it as a follow-up — the small
// stack is simpler to read correctly on a first pass, which is this
// project's priority (CLAUDE.md §1).
//
// Interior nodes push BOTH children without ordering by hit distance (an
// optimization production traversers make — visit the nearer child first so
// the farther one can be culled sooner by a tightened best_t; omitted here
// for readability, named in README Exercises).
//
// Every thread in a warp can be at a DIFFERENT stack depth, testing a
// DIFFERENT node, deciding a DIFFERENT branch — this is DIVERGENT
// TRAVERSAL, the reason this kernel cannot reach SAXPY-style full-warp
// efficiency no matter how it is tuned (THEORY.md "The GPU mapping"
// measures the real cost on this project's scene).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool intersect_bvh(const BvhNode* __restrict__ nodes,
                                              const int* __restrict__ tri_indices,
                                              const Triangle* __restrict__ tris,
                                              F3 origin, F3 dir,
                                              float tmin, float tmax,
                                              float* out_t, int* out_tri)
{
    const F3 inv_dir = f3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    int stack[kBvhStackSize];   // PER-THREAD local array — lives in registers
                                // while small enough, spills to local memory
                                // (still per-thread, cached) if not; either
                                // way it is never shared across threads.
    int sp = 0;
    stack[sp++] = 0;            // root is always node 0 (main.cu's builder contract)

    float best_t = tmax;
    int best_tri = -1;

    while (sp > 0) {
        const int node_idx = stack[--sp];
        const BvhNode node = nodes[node_idx];   // one node fetch per pop — see
                                                 // the header comment on why
                                                 // this cannot be coalesced
                                                 // across threads in general
        if (!aabb_hit(node, origin, inv_dir, tmin, best_t)) continue;   // whole subtree culled

        if (node.tri_count > 0) {
            // Leaf: tri_indices[left_first .. left_first+tri_count) is a
            // CONTIGUOUS run (the build reorders triangles into leaves), so
            // this loop is a short, cache-friendly scan, not a scatter.
            for (int i = 0; i < node.tri_count; ++i) {
                const int ti = tri_indices[node.left_first + i];
                float t;
                if (moller_trumbore(tris[ti], origin, dir, tmin, best_t, &t)) {
                    best_t = t;
                    best_tri = ti;
                }
            }
        } else {
            // Interior: both children were allocated together at build time
            // (kernels.cuh), so "the other child" is always +1.
            if (sp + 2 <= kBvhStackSize) {
                stack[sp++] = node.left_first;
                stack[sp++] = node.left_first + 1;
            }
            // The depth guarantee above means this guard should never trip
            // for this project's scene; it stays as a defensive bound so a
            // future scene edit fails SAFE (an early, wrong-but-harmless
            // stop) instead of corrupting the stack array.
        }
    }

    if (best_tri < 0) return false;
    *out_t = best_t;
    *out_tri = best_tri;
    return true;
}

// ---------------------------------------------------------------------------
// make_basis — build an orthonormal (u, v) perpendicular to unit vector n,
// for parametrizing a small cone of directions around n (the divergence
// subrays below). Standard "pick whichever world axis is LEAST aligned with
// n" trick: using the axis most nearly parallel to n would make cross()
// numerically ill-conditioned (near-zero result); |n.x| < 0.9 is a cheap,
// generous test that a unit vector can be nearly-aligned with at most one
// axis at a time.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void make_basis(F3 n, F3* u, F3* v)
{
    const F3 helper = (fabsf(n.x) < 0.9f) ? f3(1.0f, 0.0f, 0.0f) : f3(0.0f, 1.0f, 0.0f);
    *u = f3_normalize(f3_cross(helper, n));
    *v = f3_cross(n, *u);   // n, *u already orthonormal unit vectors -> *v is unit, no renormalize needed
}

// ---------------------------------------------------------------------------
// Per-beam deterministic RNG: the repo's xorshift32 + Box–Muller generator
// (identical algorithm to 08.01's, reseeded per BEAM instead of per control
// tick — see per_beam_seed below). Kept in double for the transcendental
// step, exactly 08.01's reasoning: the cheap way to keep FP32 tails
// well-behaved, and it keeps this stream close to reference_cpu.cpp's
// std::-based twin (THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int xorshift32(unsigned int& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ float uniform01(unsigned int& state)   // (0,1], never 0 — safe for log()
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

__device__ __forceinline__ float gaussian(unsigned int& state, float sigma)
{
    const double u1 = static_cast<double>(uniform01(state));
    const double u2 = static_cast<double>(uniform01(state));
    const double z = sqrt(-2.0 * log(u1)) * cos(2.0 * kPiD * u2);
    return sigma * static_cast<float>(z);
}

// per_beam_seed — mix the config's base seed with the beam index (odd
// multiplier -> full-period stream separation, 08.01's exact reasoning,
// applied per-beam instead of per-tick: every beam gets its own independent
// stream, so a dropped neighbor cannot correlate with this beam's draw).
__device__ __forceinline__ unsigned int per_beam_seed(unsigned int base_seed, int beam_idx)
{
    unsigned int s = base_seed + 1000003u * static_cast<unsigned int>(beam_idx + 1);
    if (s == 0u) s = 1u;   // xorshift32 is degenerate (stays 0 forever) at seed 0
    return s;
}

// ===========================================================================
// simulate_beam — the whole per-beam pipeline: direction -> raycast (+
// divergence subrays) -> radiometry -> dropout + range noise. A deliberate
// line-by-line twin of reference_cpu.cpp's cpu_simulate_beam(); diff the two
// to see exactly what stayed the same (all the math) and what changed
// (nothing but qualifiers — the whole point of the comparison in main.cu's
// VERIFY stage).
//
// Beam-direction math runs in DOUBLE (elevation/azimuth/cone trig), cast to
// float only at the end: this project computes each beam's direction ONCE
// (not chained over many integration steps, unlike 08.01's RK4), so the
// cost is negligible, and using double here is what makes the GPU and CPU
// paths agree to near bit-identical precision — directly serving the §5
// verification gate this project's demo runs (same reasoning 09.01/08.01
// apply to their own angle-sensitive spots).
// ===========================================================================
__device__ void simulate_beam(int beam_idx,
                              const Triangle* __restrict__ tris,
                              const Material* __restrict__ materials,
                              const BvhNode* __restrict__ nodes,
                              const int* __restrict__ tri_indices,
                              SensorConfig cfg, SensorPose pose,
                              float* out_range, float* out_intensity,
                              int* out_hit, int* out_dropped)
{
    // ---- 1) this beam's (channel, azimuth) -> direction in the SENSOR frame
    const int channel = beam_idx / cfg.azimuth_steps;
    const int az_idx = beam_idx % cfg.azimuth_steps;

    const double frac = (cfg.channels > 1)
        ? static_cast<double>(channel) / static_cast<double>(cfg.channels - 1)
        : 0.0;
    const double elevation = static_cast<double>(cfg.elevation_min_rad)
        + frac * (static_cast<double>(cfg.elevation_max_rad) - static_cast<double>(cfg.elevation_min_rad));
    const double azimuth = static_cast<double>(cfg.azimuth_start_rad)
        + static_cast<double>(az_idx) * (2.0 * kPiD / static_cast<double>(cfg.azimuth_steps));

    const double ce = cos(elevation), se = sin(elevation);
    const double ca = cos(azimuth), sa = sin(azimuth);
    // Sensor frame: x-forward, y-left, z-up (SYSTEM_DESIGN.md §3.2's body
    // convention, adopted here as the sensor's own convention too). At
    // azimuth=0 the beam points +x; sweeping azimuth turns it toward +y.
    const F3 dir_sensor = f3(static_cast<float>(ce * ca), static_cast<float>(ce * sa), static_cast<float>(se));

    // ---- 2) rotate into WORLD (the mesh's frame): dir_world = R * dir_sensor
    const F3 origin_world = f3(pose.t[0], pose.t[1], pose.t[2]);
    const F3 dir_center = f3(
        pose.R[0] * dir_sensor.x + pose.R[1] * dir_sensor.y + pose.R[2] * dir_sensor.z,
        pose.R[3] * dir_sensor.x + pose.R[4] * dir_sensor.y + pose.R[5] * dir_sensor.z,
        pose.R[6] * dir_sensor.x + pose.R[7] * dir_sensor.y + pose.R[8] * dir_sensor.z);
    // R is orthonormal and dir_sensor is unit length (cos^2+sin^2=1 by
    // construction), so dir_center is unit length too — no renormalize.

    // ---- 3) cast the central ray, then (if enabled) subray_count jittered
    // rays evenly spaced around the divergence cone; keep the NEAREST hit
    // among all of them (kernels.cuh's documented divergence approximation).
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
        const double ch = cos(half), sh = sin(half);
        for (int k = 0; k < cfg.subray_count; ++k) {
            const double phi = static_cast<double>(k) * (2.0 * kPiD / static_cast<double>(cfg.subray_count));
            const float ox = static_cast<float>(cos(phi) * sh);
            const float oy = static_cast<float>(sin(phi) * sh);
            const F3 sub_dir = f3_normalize(f3_add(
                f3_scale(dir_center, static_cast<float>(ch)),
                f3_add(f3_scale(u, ox), f3_scale(v, oy))));
            if (intersect_bvh(nodes, tri_indices, tris, origin_world, sub_dir, cfg.range_min_m, best_t, &t, &ti)) {
                best_t = t; best_tri = ti; best_dir = sub_dir;
            }
        }
    }

    if (best_tri < 0) {
        // No ray in the footprint hit anything within range — an honest
        // "no return", exactly what a real LiDAR reports for sky/open space.
        *out_hit = 0; *out_range = 0.0f; *out_intensity = 0.0f; *out_dropped = 0;
        return;
    }
    *out_hit = 1;

    // ---- 4) radiometry: Lambertian intensity from the WINNING ray's hit
    // (THEORY.md "The problem" derives the cos/R^2 form from solid-angle
    // radiometry). |cos| makes the result independent of triangle winding —
    // this is where that kernels.cuh design choice pays off.
    const Triangle tri = tris[best_tri];
    const F3 v0 = f3(tri.v0[0], tri.v0[1], tri.v0[2]);
    const F3 v1 = f3(tri.v1[0], tri.v1[1], tri.v1[2]);
    const F3 v2 = f3(tri.v2[0], tri.v2[1], tri.v2[2]);
    const F3 n = f3_normalize(f3_cross(f3_sub(v1, v0), f3_sub(v2, v0)));
    const float cos_theta = fabsf(f3_dot(n, best_dir));
    const float range_raw = best_t;
    const float albedo = materials[tri.material_id].albedo;
    float intensity = cfg.intensity_gain * albedo * cos_theta / (range_raw * range_raw);
    intensity = fminf(fmaxf(intensity, 0.0f), 1.0f);   // sensor saturation (documented in kernels.cuh)

    // ---- 5) dropout + additive range noise, from THIS beam's own private
    // RNG stream (3 draws: 1 for the dropout decision, 2 inside gaussian()).
    unsigned int rng = per_beam_seed(cfg.seed, beam_idx);
    const float u_dropout = uniform01(rng);
    const float sigma = cfg.range_noise_base_m + cfg.range_noise_per_m * range_raw;
    const float noise = gaussian(rng, sigma);

    float p_drop = cfg.dropout_base
                 + cfg.dropout_range_coeff * (range_raw / cfg.range_max_m)
                 + cfg.dropout_incidence_coeff * (1.0f - cos_theta);
    p_drop = fminf(fmaxf(p_drop, 0.0f), 1.0f);

    *out_dropped = (u_dropout < p_drop) ? 1 : 0;
    *out_range = range_raw + noise;   // noisy range is recorded even when
                                      // dropped, for transparency/debugging —
                                      // main.cu's compaction excludes
                                      // dropped beams from the point cloud
                                      // regardless (kernels.cuh's contract).
    *out_intensity = intensity;
}

// ===========================================================================
// lidar_raycast_kernel — one thread per BEAM. Thread-to-data mapping:
// beam = blockIdx.x*blockDim.x + threadIdx.x owns beam index `beam`
// (kernels.cuh's channel-major indexing). Grid: ceil(num_beams/256) x 256
// (repo default; ragged tail guarded) — see launch_lidar_raycast below.
//
// Memory spaces: mesh/BVH arrays are read-only GLOBAL memory (no shared
// memory, no constant memory — TWO independent reasons, not one: the full
// working set (triangles + nodes + the index permutation, ~130 KB on this
// project's scene) does not fit __constant__'s 64 KB budget regardless; and
// even a piece that DID fit would not benefit, because __constant__'s speed
// comes from BROADCAST reads — every thread, same address, same cycle
// (08.01's u_nom[t] is the textbook case) — while divergent traversal means
// different threads are almost never reading the same node at the same
// time (THEORY.md "The GPU mapping" expands this). The
// per-thread traversal stack lives in thread-local storage (registers,
// spilling to cached local memory only if the compiler runs out of
// registers). Final writes to out_range/out_intensity/out_hit/out_dropped
// ARE coalesced: consecutive beam indices in a warp write consecutive
// addresses, even though the WORK that produced those values took
// completely different paths through the tree (THEORY.md "The GPU mapping"
// discusses this "coalesced output, divergent middle" shape).
// ===========================================================================
__global__ void lidar_raycast_kernel(const Triangle* __restrict__ tris,
                                     const Material* __restrict__ materials,
                                     const BvhNode* __restrict__ nodes,
                                     const int* __restrict__ tri_indices,
                                     SensorConfig cfg, SensorPose pose,
                                     float* __restrict__ out_range,
                                     float* __restrict__ out_intensity,
                                     int* __restrict__ out_hit,
                                     int* __restrict__ out_dropped,
                                     int num_beams)
{
    const int beam = blockIdx.x * blockDim.x + threadIdx.x;
    if (beam >= num_beams) return;   // ragged-tail guard

    simulate_beam(beam, tris, materials, nodes, tri_indices, cfg, pose,
                 &out_range[beam], &out_intensity[beam], &out_hit[beam], &out_dropped[beam]);
}

// ===========================================================================
// Host launcher (declared in kernels.cuh).
// ===========================================================================
void launch_lidar_raycast(const Triangle* d_tris, const Material* d_materials,
                          const BvhNode* d_nodes, const int* d_tri_indices,
                          SensorConfig cfg, SensorPose pose,
                          float* d_range, float* d_intensity,
                          int* d_hit, int* d_dropped)
{
    const int num_beams = cfg.channels * cfg.azimuth_steps;
    if (num_beams < 1 || !d_tris || !d_materials || !d_nodes || !d_tri_indices ||
        !d_range || !d_intensity || !d_hit || !d_dropped) {
        std::fprintf(stderr, "launch_lidar_raycast: invalid arguments (num_beams=%d)\n", num_beams);
        std::exit(EXIT_FAILURE);
    }

    const int threads = 256;                          // repo default geometry
    const int blocks = (num_beams + threads - 1) / threads;
    lidar_raycast_kernel<<<blocks, threads>>>(d_tris, d_materials, d_nodes, d_tri_indices,
                                              cfg, pose, d_range, d_intensity, d_hit, d_dropped,
                                              num_beams);
    CUDA_CHECK_LAST_ERROR("lidar_raycast_kernel launch");
}
