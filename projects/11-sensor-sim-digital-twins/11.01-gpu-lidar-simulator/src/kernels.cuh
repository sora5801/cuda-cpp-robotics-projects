// ===========================================================================
// kernels.cuh — interface for project 11.01
//               GPU LiDAR simulator: hand-built BVH raycasting + beam
//               divergence, intensity, dropout noise
//               (teaching core: a spinning-scanner LiDAR over a synthetic
//               triangle-mesh warehouse, with a HAND-BUILT BVH — build your
//               own before touching OptiX, CLAUDE.md §5)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (loads the mesh, BUILDS the BVH on the HOST,
// uploads everything, drives verification + artifacts), kernels.cu (the GPU
// raycasting kernel), and reference_cpu.cpp (the CPU oracle twin of that
// kernel). Everything all three must agree on — the triangle/material/BVH
// node layouts, the sensor config/pose structs, and the per-beam output
// layout — is defined HERE, once (CLAUDE.md §12: state layouts are
// single-sourced).
//
// The pipeline in six lines (THEORY.md derives every step properly):
//   1. main.cu loads a triangle mesh + per-triangle materials (host).
//   2. main.cu BUILDS a median-split BVH over the mesh (host, ONCE — GPU
//      BVH construction is project 07.03's dedicated subject; this project
//      is about the raycasting and the sensor-noise models, so the tree
//      itself is built with a simple, correct host algorithm and uploaded
//      as flat, read-only arrays. See "THE BVH" below.).
//   3. One GPU thread per BEAM (channels x azimuth_steps, e.g. 32x1024 =
//      32,768 beams/frame) generates that beam's direction from the
//      spinning-scanner model, casts 1 (or 1+divergence subrays) ray(s)
//      through the BVH, and finds the nearest triangle hit via
//      Möller–Trumbore intersection.
//   4. From the winning hit: incidence angle + material albedo -> a
//      Lambertian intensity (radiometry, THEORY.md "The problem").
//   5. A per-beam deterministic RNG stream applies range-dependent dropout
//      and additive range noise (THEORY.md "Numerical considerations").
//   6. main.cu (host) compacts the surviving beams into a PointCloud-shaped
//      buffer (xyz + intensity + ring) matching SYSTEM_DESIGN.md §3.6 — this
//      project's output is Chain A's very first stage:
//      [11.01 GPU LiDAR simulator] -> [02.06 ICP] -> [05.01 TSDF fusion] ...
// Step 3 is >99% of the arithmetic and is embarrassingly parallel across
// beams — the same thread-per-problem shape as 33.01/09.01/08.01/02.06, now
// applied to a tree-structured (not flat-array) memory-access pattern: the
// project's central NEW GPU concept is DIVERGENT TRAVERSAL (THEORY.md "The
// GPU mapping" measures and explains it).
//
// THE MESH — a triangle soup with a parallel per-triangle material table:
//   Triangle.{v0,v1,v2} : world-frame vertex positions (m), any winding
//                         (intensity uses |cos(incidence)|, so winding is
//                         NOT load-bearing here — a deliberate simplification
//                         that keeps the synthetic-scene generator simple;
//                         see kernels.cu for exactly where the fabs() lives).
//   Triangle.material_id : index into the materials[] table (bounds-checked
//                         once at load time in main.cu, never in the hot
//                         kernel loop).
//
// THE BVH — flattened, GPU-resident, built ONCE on the host at load time
// (main.cu's build_bvh_median_split(); THEORY.md "The algorithm" walks the
// construction). Layout follows the well-known "children allocated in
// pairs" scheme (Jacco Bikker's "How to Build a BVH" tutorial series;
// README §Prior art):
//
//   BvhNode.aabb_min/aabb_max : the node's axis-aligned bounding box (m),
//                               tight around every triangle in its subtree.
//   BvhNode.tri_count == 0   -> INTERIOR node: its two children live at
//                               nodes[left_first] and nodes[left_first + 1]
//                               (always adjacent — both children of a split
//                               are allocated together, so "the sibling" is
//                               always "+1", no extra pointer needed).
//   BvhNode.tri_count  > 0   -> LEAF node: its triangles are
//                               tri_indices[left_first .. left_first +
//                               tri_count) — a CONTIGUOUS slice of the
//                               PERMUTED triangle-index array the build
//                               produces (leaf triangles are physically
//                               reordered into contiguous runs during the
//                               build so a leaf is one coalesced read, not a
//                               scatter/gather).
//
//   Why median-split BY TRIANGLE COUNT (not by spatial midpoint)? Splitting
//   each node's triangle set exactly in half at every level GUARANTEES tree
//   depth <= ceil(log2(N / kBvhLeafSize)) regardless of how the geometry is
//   arranged in space (a spatial-midpoint split can degenerate to O(N) depth
//   on clustered geometry; a count-based median split cannot — THEORY.md
//   proves this). That guarantee is what makes the traversal's FIXED,
//   SMALL stack (kBvhStackSize below) provably sufficient rather than a
//   "probably enough" guess — the whole reason this project can use a small
//   stack instead of a heap-allocated one or a stackless skip-pointer
//   scheme (both viable alternatives; THEORY.md "The GPU mapping" names the
//   trade this project did NOT take and why).
//
// SENSOR MODEL — a spinning mechanical LiDAR (THEORY.md "The problem" covers
// the real physics: pulsed time-of-flight, a firing sequence of CHANNELS
// fixed-elevation lasers, one full sweep = AZIMUTH_STEPS shots per rotation).
// BEAM INDEXING (the project's other one-place contract):
//     beam = channel * azimuth_steps + azimuth_idx,   channel-major
//     num_beams = channels * azimuth_steps  (demo default: 32*1024 = 32768,
//     the exact figure the catalog bullet's example cites)
// Per beam: elevation is linearly interpolated across [elevation_min_rad,
// elevation_max_rad] by channel index (channel 0 = elevation_min); azimuth
// steps evenly around a full 2*pi starting at azimuth_start_rad. Direction
// is built in the SENSOR frame (repo body convention: x-forward, y-left,
// z-up, SYSTEM_DESIGN.md §3.2) and rotated into world by the sensor pose.
//
// THE THREE EFFECT MODELS (the catalog bullet's second half; each is a
// documented, honestly-scoped simplification — THEORY.md gives the real
// physics each one stands in for):
//   (a) BEAM DIVERGENCE — a real beam is a narrow CONE, not a ray; this
//       project casts the central ray plus `subray_count` extra rays
//       jittered around that cone (evenly spaced in azimuth at the
//       configured half-angle) and reports the NEAREST hit among all of
//       them — "the closest surface in the footprint usually dominates the
//       return" (THEORY.md derives why, and states plainly what real
//       edge-blur / mixed-pixel returns this does NOT reproduce).
//   (b) INTENSITY — Lambertian: intensity = intensity_gain * albedo *
//       |cos(incidence)| / range^2, clamped to [0,1] (sensor saturation).
//       THEORY.md derives the cos/R^2 form from solid-angle radiometry.
//   (c) DROPOUT + RANGE NOISE — a per-beam deterministic RNG stream (the
//       repo's xorshift32 + Box–Muller, exactly 08.01's generator, reseeded
//       per BEAM instead of per control tick) decides (i) whether the beam
//       is dropped, with probability rising with range and grazing
//       incidence (THEORY.md derives the SNR argument), and (ii) an
//       additive Gaussian range perturbation with a range-growing sigma.
//
// OUTPUT LAYOUT (SoA, one entry per BEAM — NOT compacted; main.cu does the
// compaction into the final PointCloud on the HOST, deliberately: it is
// O(num_beams) trivial bookkeeping, and keeping it off the GPU means the
// kernel's per-beam output is directly, exactly comparable against the CPU
// oracle beam-by-beam — the same "simple host bookkeeping stays on the
// host" choice 08.01 makes for its softmin blend):
//     d_range[beam]     : float, meters. The (possibly noisy) range along
//                         the WINNING ray if hit==1, else 0 (meaningless).
//     d_intensity[beam] : float, [0,1]. 0 if hit==0.
//     d_hit[beam]       : int, 1 if any of the 1+subray_count rays hit
//                         geometry within [range_min_m, range_max_m].
//     d_dropped[beam]   : int, 1 if the dropout draw removed this beam
//                         (only meaningful where hit==1; a beam that never
//                         hit anything cannot be "dropped" — it was never a
//                         return in the first place).
// A beam belongs in the final point cloud iff hit==1 AND dropped==0.
// main.cu's compaction recomputes each surviving beam's SENSOR-FRAME
// direction (cheap, pure trig, no branching — see main.cu's packing step)
// and multiplies by d_range to get the point's xyz, exactly how a real
// spinning LiDAR driver reports points in its own "lidar" frame
// (SYSTEM_DESIGN.md §3.6's PointCloud.header.frame_id), independent of
// where the sensor happens to sit in the world.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Triangle — one mesh triangle. Plain float triples (not a Vec3 helper
// type): the repo convention for structs shared across the host/nvcc
// boundary (see 02.06's Rigid3, 05.01's PoseRt) is plain arrays with math
// helpers duplicated as twins in kernels.cu/reference_cpu.cpp, never
// __host__ __device__ shared methods — CLAUDE.md §4's deliberate-duplication
// rule applied to functions, not just whole files.
// ---------------------------------------------------------------------------
struct Triangle {
    float v0[3], v1[3], v2[3];   // world-frame vertex positions, meters
    int   material_id;           // index into the materials[] table
};

// Material — one Lambertian reflectance property. THEORY.md "The problem"
// derives why a single scalar is the whole radiometric story for a
// monostatic (co-located transmitter/receiver) LiDAR return.
struct Material {
    float albedo;   // dimensionless reflectance, (0, 1]
};

// ---------------------------------------------------------------------------
// BvhNode — one flattened BVH node (see the long header comment above for
// the "children allocated in pairs" scheme). 32 bytes: two 12-byte AABB
// corners plus two 4-byte ints — chosen to keep every node read a single
// coalesced-friendly cache-line-aligned-ish fetch (THEORY.md "The GPU
// mapping" discusses the honest limits of that claim under divergent
// traversal, where different threads are never reading the SAME node at
// the SAME time in the first place).
// ---------------------------------------------------------------------------
struct BvhNode {
    float aabb_min[3];
    float aabb_max[3];
    int   left_first;   // interior: index of the LEFT child (right = +1).
                         // leaf: start index into tri_indices[].
    int   tri_count;    // 0 => interior node. >0 => leaf; this many
                         // triangles starting at tri_indices[left_first].
};

// ---------------------------------------------------------------------------
// SensorConfig — the spinning-scanner pattern plus all three effect models,
// passed BY VALUE into the kernel launcher (a few dozen bytes; the same
// "kernel parameter bank" broadcast-read choice 02.06's Rigid3 makes, cheaper
// than a device upload for something this small and read-only per launch).
// Angles are RADIANS here (the loader in main.cu converts from the
// human-readable degrees/milliradians in data/sample/sensor_config.csv).
// ---------------------------------------------------------------------------
struct SensorConfig {
    int   channels;                    // vertical beam rows
    int   azimuth_steps;                // beams per full rotation (columns)
    float elevation_min_rad;            // channel 0's elevation (+up from horizontal)
    float elevation_max_rad;            // channel (channels-1)'s elevation
    float azimuth_start_rad;            // azimuth of column 0
    float range_min_m;                  // beams reporting a hit closer than this are treated as no-return (near-field blanking, real sensors have one too)
    float range_max_m;                  // maximum unambiguous range; also the raycast's tmax and the dropout model's range-normalization scale
    float divergence_half_angle_rad;    // beam footprint half-angle (subray cone)
    int   subray_count;                 // extra jittered rays beyond the central one (0 = divergence modeling off)
    float intensity_gain;               // k_sensor radiometric scale (dimensionless; THEORY.md "The problem")
    float range_noise_base_m;           // sigma(R) = base + per_m * R
    float range_noise_per_m;
    float dropout_base;                 // p = clamp(base + range_coeff*(R/range_max_m) + incidence_coeff*(1-|cos_theta|), 0, 1)
    float dropout_range_coeff;
    float dropout_incidence_coeff;
    unsigned int seed;                  // base seed for the per-beam RNG stream (see kernels.cu's per_beam_seed())
};

// SensorPose — T_world_sensor (SYSTEM_DESIGN.md §3.3: "sensor expressed in
// world"). A point in the sensor frame is R*p + t in world.
struct SensorPose {
    float R[9];   // row-major 3x3 rotation
    float t[3];   // translation, meters, world frame
};

// ---------------------------------------------------------------------------
// Build-time / traversal-time constants — shared by the HOST BVH builder
// (main.cu), the GPU traversal kernel (kernels.cu), and the CPU oracle
// (reference_cpu.cpp), because the depth GUARANTEE that makes the fixed
// stack size safe depends on the SAME leaf-size constant the builder used
// (see the long header comment's median-split argument).
// ---------------------------------------------------------------------------
constexpr int kBvhLeafSize  = 4;    // max triangles per leaf
constexpr int kBvhStackSize = 64;   // traversal stack depth bound; see
                                    // kernels.cu's intersect_bvh() header
                                    // comment for the depth arithmetic this
                                    // is sized against (>5x headroom).

// kPiD — double-precision pi, shared by every beam-direction/cone-sampling
// trig call in kernels.cu, reference_cpu.cpp, AND main.cu (the packing step
// that recomputes each surviving beam's direction — kernels.cuh's output-
// layout comment). One constant, three consumers, zero drift risk. A plain
// file-scope `constexpr double` compiles fine on both sides of the
// __CUDACC__ boundary (the compiler folds it into an immediate at every use
// site, exactly like kMassCart/kGravity do in 08.01's kernels.cuh) — no
// __device__ qualifier needed for a compile-time scalar constant.
constexpr double kPiD = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// launch_lidar_raycast — simulate one full frame (cfg.channels *
// cfg.azimuth_steps beams) against the mesh/BVH on the GPU.
//
//   d_tris, d_materials       : DEVICE pointers, the mesh (never written).
//   d_nodes                   : DEVICE pointer, the flattened BVH (never written).
//   d_tri_indices             : DEVICE pointer, the BVH's permuted triangle
//                               index array (never written) — see the BvhNode
//                               leaf-layout comment above.
//   cfg, pose                 : the scan pattern/effects and the sensor's
//                               world pose for this frame (by value).
//   d_range, d_intensity      : DEVICE pointers, num_beams floats OUT (SoA
//                               per-beam layout above).
//   d_hit, d_dropped          : DEVICE pointers, num_beams ints OUT.
//
// Launch: one thread per BEAM, 256-thread blocks (grid math + reasoning
// lives with the kernel in kernels.cu). num_beams = cfg.channels *
// cfg.azimuth_steps is derived from cfg, not passed separately — one
// fewer place for a caller to get out of sync with the config it passed.
// ---------------------------------------------------------------------------
void launch_lidar_raycast(const Triangle* d_tris, const Material* d_materials,
                          const BvhNode* d_nodes, const int* d_tri_indices,
                          SensorConfig cfg, SensorPose pose,
                          float* d_range, float* d_intensity,
                          int* d_hit, int* d_dropped);

// ---------------------------------------------------------------------------
// lidar_raycast_cpu — the oracle twin of the kernel above: identical BVH
// traversal, identical Möller–Trumbore, identical divergence/radiometry/
// dropout math, sequential over beams, plain C++. main.cu runs this against
// the GPU kernel on the full demo frame and requires agreement within
// documented tolerances (the §5 GPU-vs-CPU gate); main.cu's ANALYTIC
// verification gates (ground-plane range, inverse-square intensity,
// dropout-rate statistics) run EXCLUSIVELY through this function — the
// GPU-vs-CPU gate above already establishes that the kernel computes the
// identical answer, so validating the PHYSICS MODEL itself against this
// single, simpler, easier-to-audit sequential implementation is sufficient
// and avoids a proliferation of tiny one-off kernel launches (documented
// choice, THEORY.md "How we verify correctness").
//
// Same pointers/layouts as launch_lidar_raycast, but HOST pointers.
// ---------------------------------------------------------------------------
void lidar_raycast_cpu(const Triangle* tris, const Material* materials,
                       const BvhNode* nodes, const int* tri_indices,
                       const SensorConfig& cfg, const SensorPose& pose,
                       float* range, float* intensity,
                       int* hit, int* dropped);

#endif // PROJECT_KERNELS_CUH
