// ===========================================================================
// kernels.cuh — interface for project 05.01
//               TSDF fusion (KinectFusion clone) + marching-cubes mesh
//               extraction (teaching core: fuse synthetic depth from KNOWN
//               poses into a dense volume, then pull a mesh out of it)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver: data, depth rendering, verify,
// artifacts), kernels.cu (the two GPU kernels), and reference_cpu.cpp (the
// CPU integration twin + the marching-cubes recount). Everything all three
// must agree on — the volume layout, the scene definition, the camera
// structs, and the TSDF update rule — is defined HERE, once (CLAUDE.md §12).
//
// TSDF fusion in five lines (THEORY.md derives it properly):
//   1. A dense 3-D grid of voxels covers the workspace. Each voxel stores a
//      TRUNCATED SIGNED DISTANCE to the nearest surface (+ = free space in
//      front of the surface, − = behind/inside, clamped to ±1 in units of
//      the truncation distance mu) and a confidence WEIGHT.
//   2. For each depth frame with a KNOWN camera pose, every voxel projects
//      itself into the image, reads the depth there, and measures how far
//      in front of / behind the observed surface it sits (along the ray).
//   3. That per-frame estimate is blended into the voxel by a running
//      weighted average — noisy single-frame measurements average into a
//      clean surface (the whole magic of KinectFusion).
//   4. The surface is the ZERO CROSSING of the fused field.
//   5. Marching cubes walks every 2x2x2 voxel cell, classifies its corners
//      against zero, and emits the triangles of the iso-surface.
// Steps 2–3 are one independent job per VOXEL (128^3 ≈ 2.1 M of them) and
// step 5 is one independent job per CELL (127^3) — the two canonical GPU
// grid patterns this project teaches (map-over-voxels, append-from-cells).
//
// NO TRACKING HERE, deliberately: real KinectFusion estimates each frame's
// pose by ICP against the model. This project takes poses as GIVEN (they
// are part of the committed sample) so the fusion math can be verified
// against exact ground truth; ICP pose tracking is project 02.06.
//
// THE SCENE — analytic, so ground truth is EXACT:
//   a sphere (radius kSphereR, center kSphere{CX,CY,CZ}) floating above an
//   infinite ground plane (the half-space z <= kPlaneZ). Its signed distance
//   function has a closed form:
//        sdf(p) = min( |p - c| - r,  p.z - kPlaneZ )
//   min() of the two SDFs is the exact union SDF wherever the two bodies'
//   truncation bands do not overlap — and the sphere floats kSphereCZ -
//   kSphereR - kPlaneZ = 0.25 m above the plane, more than 2x the 0.12 m
//   truncation, so within the band the closed form is EXACT (THEORY.md
//   §math). Depth frames are RENDERED from this scene by ray casting
//   (closed-form ray/sphere and ray/plane hits, render_depth() in main.cu),
//   so the demo needs no downloads and the fused volume can be compared
//   against sdf(p) itself — real ground truth, not a fixture file.
//
// VOLUME LAYOUT — one flat array, x fastest (the coalescing axis):
//     linear index v = (iz * kVolN + iy) * kVolN + ix
//     voxel CENTER in world:  p = kVolOrigin + (i + 0.5) * kVoxelSize
//   tsdf[v]   : truncated signed distance in UNITS OF mu, in [-1, +1]
//               (+1 = at least mu in front of the surface, 0 = on it,
//               -1 = at least mu behind it). Multiply by kTruncation for
//               meters. Initialized to +1 ("far"), meaningless while
//               weight == 0.
//   weight[v] : accumulated observation count, 0 = never observed (the
//               validity flag marching cubes checks), capped at kMaxWeight
//               so old frames cannot outvote fresh ones forever.
//   Adjacent threads process adjacent ix → adjacent addresses: both arrays
//   are read and written fully coalesced (the 33.01 layout lesson).
//
// FRAMES & POSES — camera optical convention (x-right, y-down, z-forward —
// the domain standard for cameras; stated per CLAUDE.md §12 because it
// differs from the body x-forward/z-up convention). Poses in the sample
// file are T_world_cam ("camera expressed in world"); the integration
// kernel wants the inverse, T_cam_world, and main.cu inverts on the host
// once per frame (a 3x3 transpose — cheap, and it keeps the kernel free of
// per-thread redundant work).
//
// Depth images are z-DEPTH (meters along the optical axis, NOT along the
// ray) — the convention real RGB-D sensors ship, and the reason the
// projective-TSDF update below reads "depth - z_cam" (THEORY.md §algorithm
// discusses the bias this approximation carries).
//
// DETERMINISM CONTRACT: the integration kernel spells every multiply-add as
// an explicit fmaf() and the CPU twin uses std::fmaf identically, so both
// paths execute the same IEEE-754 operations in the same order and the
// projected pixel — an int, where an ulp could flip a rounding decision
// near a pixel boundary and change WHICH depth sample a voxel reads — is
// bit-identical on both paths. The VERIFY tolerance (1e-5) is therefore
// pure headroom against compiler surprises, not an expected error budget
// (THEORY.md §numerics). Marching cubes appends triangles with atomicAdd:
// the triangle SET and COUNT are deterministic, the buffer ORDER is not —
// documented honestly (README §Limitations).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Volume parameters — shared verbatim by the GPU kernels, the CPU twin, the
// ground-truth check, and the artifact writers (one source of truth; a
// mismatch here would silently fuse into the wrong grid).
// ---------------------------------------------------------------------------
constexpr int   kVolN       = 128;      // voxels per axis → kVolN^3 ≈ 2.1 M voxels
constexpr float kVoxelSize  = 0.02f;    // voxel edge (m) → the cube spans 2.56 m
constexpr float kVolOriginX = -1.28f;   // world position of the volume's min
constexpr float kVolOriginY = -1.28f;   // corner (m). Chosen so the cube is
constexpr float kVolOriginZ = -0.20f;   // centered on the scene in x/y and
                                        // keeps the ground plane (z=0) inside.
constexpr float kTruncation = 0.12f;    // mu (m): the half-width of the band
                                        // around the surface where distances
                                        // are trusted — 6 voxels, the classic
                                        // "a few voxels" KinectFusion choice
                                        // (THEORY.md §algorithm: why truncate)
constexpr float kMaxWeight  = 64.0f;    // weight cap: keeps the average
                                        // adaptive (new frames always carry
                                        // at least 1/(kMaxWeight+1) influence)

// ---------------------------------------------------------------------------
// The analytic scene (meters, world frame: right-handed, z up). Used by the
// depth renderer and the ground-truth check in main.cu; documented here
// because it is part of the project's one-place contract.
// ---------------------------------------------------------------------------
constexpr float kSphereCX = 0.0f;       // sphere center (m)
constexpr float kSphereCY = 0.0f;
constexpr float kSphereCZ = 0.75f;
constexpr float kSphereR  = 0.5f;       // sphere radius (m)
constexpr float kPlaneZ   = 0.0f;       // ground plane height: solid is z <= kPlaneZ
// Clearance between the two truncation bands: kSphereCZ - kSphereR - kPlaneZ
// = 0.25 m > 2 * kTruncation → min() of the two SDFs is exact in the band.

// ---------------------------------------------------------------------------
// Marching-cubes output capacity. The demo scene produces ~50 k triangles
// (sphere ~3.1 m^2 + visible plane ~6.6 m^2 of surface, ~2 triangles per
// surface cell of (0.02 m)^2); 4x headroom means the cap can only be hit by
// a real bug, and main.cu still checks the counter against it explicitly.
// ---------------------------------------------------------------------------
constexpr int kMaxTriangles = 200000;

// ---------------------------------------------------------------------------
// Intrinsics — the pinhole camera model (units: pixels), loaded from the
// committed sample (data/sample/camera_path.csv, CAM row).
//   Projection: u = fx * x/z + cx,  v = fy * y/z + cy   (camera optical
//   frame, x-right / y-down / z-forward; u right, v down; the pixel grid is
//   [0..width-1] x [0..height-1] with integer coordinates at pixel CENTERS).
// Passed to kernels BY VALUE (a handful of scalars — cheaper and simpler
// than a device allocation, and constant across all threads of a launch).
// ---------------------------------------------------------------------------
struct Intrinsics {
    int   width;    // image width  (px)
    int   height;   // image height (px)
    float fx, fy;   // focal lengths (px)
    float cx, cy;   // principal point (px)
};

// ---------------------------------------------------------------------------
// PoseRt — a rigid transform as an explicit rotation matrix + translation:
//     p_out = R * p_in + t
// Named by its direction at every use site (T_world_cam vs T_cam_world —
// CLAUDE.md §12 T_parent_child notation). The 3x3 is ROW-MAJOR:
// r[3*row+col]. We pass matrices, not quaternions, into kernels so the
// per-voxel transform is 9 fmaf's with zero per-thread setup; quaternions
// live only in the data file (compact, always normalizable) and are
// converted once on the host (main.cu quat_to_rot()).
// ---------------------------------------------------------------------------
struct PoseRt {
    float r[9];     // rotation, row-major
    float t[3];     // translation (m)
};

// ---------------------------------------------------------------------------
// launch_volume_clear — reset the volume to "never observed".
//
//   d_tsdf   : DEVICE pointer, kVolN^3 floats OUT — all set to +1.0 ("far").
//   d_weight : DEVICE pointer, kVolN^3 floats OUT — all set to 0.0 (invalid).
//
// Launch: one thread per voxel, 256-thread blocks (grid math with the
// kernel). Used before the verify pass AND before the full fusion so both
// start from an identical, defined state.
// ---------------------------------------------------------------------------
void launch_volume_clear(float* d_tsdf, float* d_weight);

// ---------------------------------------------------------------------------
// launch_tsdf_integrate — fuse ONE depth frame into the volume (the
// KinectFusion update; call once per frame, any order).
//
//   d_depth     : DEVICE pointer, K.width*K.height floats — z-depth in
//                 meters, row-major, v*width+u; <= 0 marks "no return"
//                 (sky pixels) and is skipped.
//   K           : intrinsics (by value; see struct).
//   T_cam_world : world -> camera transform (by value): p_cam = R*p_world + t.
//   d_tsdf      : DEVICE pointer, kVolN^3 floats IN/OUT — running TSDF.
//   d_weight    : DEVICE pointer, kVolN^3 floats IN/OUT — running weights.
//
// Launch: one thread per VOXEL (kVolN^3 total), 256-thread blocks. Each
// thread owns voxel v = blockIdx.x*blockDim.x + threadIdx.x and touches
// only tsdf[v]/weight[v] — no atomics, no races, fully coalesced.
// ---------------------------------------------------------------------------
void launch_tsdf_integrate(const float* d_depth, Intrinsics K,
                           PoseRt T_cam_world,
                           float* d_tsdf, float* d_weight);

// ---------------------------------------------------------------------------
// launch_marching_cubes — extract the zero iso-surface as triangles.
//
//   d_tsdf      : DEVICE pointer, kVolN^3 floats — the fused TSDF.
//   d_weight    : DEVICE pointer, kVolN^3 floats — cells with ANY corner of
//                 weight 0 are skipped (never-observed space has no surface).
//   max_tris    : capacity of d_tri_verts in triangles (kMaxTriangles).
//   d_tri_verts : DEVICE pointer, max_tris*9 floats OUT — packed triangles,
//                 [t*9 .. t*9+8] = (x0,y0,z0, x1,y1,z1, x2,y2,z2) in meters,
//                 world frame.
//   d_tri_count : DEVICE pointer, 1 int IN/OUT — must be 0 on entry; on exit
//                 the TOTAL number of triangles the surface generated. May
//                 exceed max_tris (the check is the caller's contract —
//                 vertices beyond the cap are simply not stored).
//
// Launch: one thread per CELL ((kVolN-1)^3 total), 256-thread blocks;
// thread c owns the cell with min corner (ix,iy,iz) decoded from c. Emission
// uses the atomic-append pattern: atomicAdd on d_tri_count reserves a slot
// per triangle (kernels.cu header discusses this vs. the two-pass
// prefix-sum alternative production libraries use).
// ---------------------------------------------------------------------------
void launch_marching_cubes(const float* d_tsdf, const float* d_weight,
                           int max_tris, float* d_tri_verts, int* d_tri_count);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp).
// ---------------------------------------------------------------------------

// tsdf_integrate_cpu — the oracle twin of the integration kernel: same
// projection, same fmaf's in the same order, sequential over voxels.
// main.cu fuses a 4-frame subset through both paths into separate volumes
// and requires agreement within abs tol 1e-5 (the §5 GPU-vs-CPU gate for
// this project). Arrays are HOST pointers with the same layout as above.
void tsdf_integrate_cpu(const float* depth, Intrinsics K,
                        PoseRt T_cam_world,
                        float* tsdf, float* weight);

// marching_cubes_count_cpu — classify every cell of a (host-side) volume
// exactly as the kernel does and return the total triangle count WITHOUT
// emitting geometry. Because classification compares the same float values
// the same way, the count must equal the GPU counter EXACTLY — an
// order-independent check on the whole marching-cubes pass (THEORY.md
// §verification).
long long marching_cubes_count_cpu(const float* tsdf, const float* weight);

#endif // PROJECT_KERNELS_CUH
