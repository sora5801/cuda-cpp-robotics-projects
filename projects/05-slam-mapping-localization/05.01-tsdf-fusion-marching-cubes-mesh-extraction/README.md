# 05.01 — TSDF fusion (KinectFusion clone) + marching-cubes mesh extraction

**Difficulty:** ★ beginner · **Domain:** 5. SLAM, Mapping & Localization

> Catalog bullet (source of truth, verbatim): `★ TSDF fusion (KinectFusion clone) + marching-cubes mesh extraction`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**Dense 3-D mapping asks every voxel in a volume the same question, 24 times over: "camera, how far
in front of or behind the surface you just saw am I?"** This project builds KinectFusion's classic
answer end to end: fuse a sequence of depth images, taken from **known** camera poses, into a dense
**truncated signed distance field (TSDF)** on a 128³ voxel grid — then pull an explicit triangle mesh
out of that field with **marching cubes**. The scene is an analytic sphere floating above a ground
plane, chosen because its exact signed distance function has a closed form: depth frames are rendered
in-code by ray casting that scene, so the demo needs no downloads, and the fused volume can be checked
against *real, exact ground truth* rather than a stored fixture. Everything the catalog bullet names is
implemented in full: TSDF fusion AND marching-cubes extraction, both on the GPU, both verified against
CPU twins. **Not implemented, by design:** pose *tracking* (ICP) — poses are taken as given, exactly as
the catalog's neighbor project `02.06` (GPU ICP) supplies them in a real pipeline; see
[Limitations & honesty](#limitations--honesty).

## What this computes & why the GPU helps

Two independent GPU passes, each the canonical mapping for its data structure:

- **TSDF integration** — for each of 24 depth frames, every one of the volume's ~2.1 million voxels
  projects itself into that frame's image, reads the depth there, and folds a truncated distance
  estimate into its own running weighted average. Voxels never touch each other: a pure **map**, one
  thread per voxel, the same shape as `33.01`'s foundational pattern applied to a 3-D grid instead of a
  batch.
- **Marching cubes** — every one of the volume's ~2.0 million *cells* (a cell is 8 neighboring voxels)
  asks "does the zero-crossing surface pass through me, and where?", looks its 8-corner in/out pattern
  up in a precomputed 256-case table, and **appends** however many triangles (0–5) that case produces
  to a shared output buffer via `atomicAdd`. This is this project's newest pattern: a variable-length,
  thread-parallel **append**, not a fixed-size map.
- **Measured reality** (RTX 2080 SUPER, 2026-07-09): the 128³-voxel integration kernel takes
  ~0.07–0.3 ms per frame vs ~55–75 ms for the plain-C++ CPU twin (roughly 200–700×, single-shot,
  kernel-only); the marching-cubes pass over 127³ cells takes ~0.8 ms. Fusing all 24 frames and
  extracting the mesh is comfortably sub-10-ms total GPU time — the reason dense reconstruction runs
  online on real robots at all.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** **state estimation / world model** — specifically the *mapping* half (SLAM's "M");
  it sits right after localization has supplied a pose and right before the map becomes an obstacle
  field for planning (SYSTEM_DESIGN §1: `PERCEPTION → STATE ESTIMATION/WORLD MODEL → PREDICTION`).
- **Upstream inputs:** a depth `Image` per frame (message shape: `sensor_msgs/Image`-style, meters,
  row-major — SYSTEM_DESIGN §3.6) and that frame's `T_world_cam` pose. In this teaching scope the poses
  are given (the committed sample); on a real robot they come from `02.06` (GPU ICP registration) or an
  equivalent visual/LiDAR odometry front end — SYSTEM_DESIGN's own composition map (§4.1, Chain A) names
  this exact handoff: `[02.06 ICP registration] → T_map_base → [05.01 TSDF fusion]`.
- **Downstream consumers:** the fused volume feeds obstacle/distance fields for planning — SYSTEM_DESIGN
  Chain A continues `[05.01 TSDF fusion] → voxel grid (10–20 Hz) → [07.09 jump-flooding distance field]
  → [06.05 STOMP] → [08.01 MPPI controller]`, with `[31.01 HJ reachability]` watching the state the whole
  time. The extracted mesh itself is also a direct product: a viewable/exportable 3-D reconstruction of
  the workspace (this project's `demo/out/mesh.obj`). The catalog's very next bullet in this domain —
  voxel-hashed large-scale TSDF with ESDF generation, nvblox-style — is the production-scale version of
  exactly this block, generating the Euclidean signed-distance field planners actually consume.
- **Rate / latency budget:** mapping runs at the LiDAR/depth-sensor rate, **10–20 Hz**, with a
  <50–100 ms per-scan budget (SYSTEM_DESIGN §1.1); the warehouse-AMR reference block diagram (§2.1)
  lists "map update 10–20 Hz" explicitly. This demo's ~0.1–0.3 ms/frame integration kernel and ~0.8 ms
  mesh pass sit two to three orders of magnitude inside that budget — real headroom for the larger
  voxel-hashed volumes production systems actually run.
- **Reference robot(s):** the **warehouse AMR** (§2.1) most directly — SYSTEM_DESIGN names `05.01` by ID
  in its own worked example (Chain A, §4.1) as the AMR's mapping stage. The **manipulator work cell**
  (§2.2) also commonly grows exactly this block between vision and grasp/motion planning: real work
  cells often fuse wrist-camera depth into a small TSDF/ESDF volume of the bin so collision-aware
  motion planning (`06`/`07`) has a dense, up-to-date obstacle field, even though §2.2's compact diagram
  shows only the single-frame stereo path.
- **In production:** Newcombe et al.'s **KinectFusion** (2011) is the algorithm this project reimplements
  didactically; **nvblox** (NVIDIA, voxel-hashed TSDF + ESDF, ships in Isaac) and **Voxblox** (ETH,
  ESDF-focused incremental mapping) are its production, large-scale, unbounded-volume descendants;
  **Open3D**'s `ScalableTSDFVolume` and **InfiniTAM**/**VDBFusion** are widely used research/open-source
  implementations. All of them add exactly what this teaching core omits: pose tracking, voxel hashing
  for unbounded scenes, and incidence/confidence-aware fusion (see Limitations).
- **Owning team:** controls & autonomy (SYSTEM_DESIGN §5.1: "state estimation, SLAM, planning, control —
  domains 04, 05, 06, ..."), adjacent to perception (who supplies the depth stream) and simulation/tools
  (who owns the sensor models this project's ray-cast renderer stands in for).

## The algorithm in brief

- **TSDF measurement update** — project each voxel into the current frame, read the depth there,
  compute the *projective* signed distance `d − z_cam` along the optical axis, truncate to `±μ`, and
  fold it into a running weighted average, weight capped. → [THEORY.md](THEORY.md) §The algorithm.
- **Depth rendering by closed-form ray casting** — this project's "sensor": every pixel's ray is
  intersected analytically with the sphere (quadratic) and the plane (linear), so rendered depth is
  exact to floating-point rounding and ground truth is the scene's own SDF. → THEORY §How we verify.
- **Marching cubes** — classify each cell's 8 corners against the zero level set, look the pattern up
  in the 256-case table (`src/mc_tables.h`, in `__constant__` memory), linearly interpolate vertices on
  the sign-changing edges, and atomically append triangles. → THEORY §The GPU mapping.
- **No tracking** — poses are read from the committed sample, not estimated; `02.06` (GPU ICP) is where
  that problem lives in this repo. → [Limitations & honesty](#limitations--honesty).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/tsdf-fusion-marching-cubes-mesh-extraction.sln`](build/tsdf-fusion-marching-cubes-mesh-extraction.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/tsdf-fusion-marching-cubes-mesh-extraction.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
marching-cubes case table (`src/mc_tables.h`) is plain data compiled directly into the binary, not an
external dependency.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including how to
view the **mesh and slice artifacts**.

## Data

The committed sample is a **camera path**, not depth recordings: `data/sample/camera_path.csv`
(~2.4 KiB, synthetic, no RNG) — pinhole intrinsics plus 24 poses on a circle orbiting the scene. Depth
images themselves are **rendered inside the demo at run time** by closed-form ray casting of the
analytic scene defined in [`src/kernels.cuh`](src/kernels.cuh) (a sphere over a ground plane) — so the
committed data stays tiny, the demo needs zero downloads, and the fusion result can be checked against
the scene's own exact SDF instead of a stored fixture. No public dataset applies (a real depth stream
cannot be reduced to a closed-form scene); `scripts/download_data.ps1` is an honest no-op. Details and
full field documentation: [`data/README.md`](data/README.md).

## Expected output

Eight stable lines — banner, `PROBLEM:`, `SAMPLE:`, `VERIFY:`, `GROUND TRUTH:`, `MESH:`, two `ARTIFACT:`
lines, and `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Three independent verifications, all measured
on the reference machine (RTX 2080 SUPER, 2026-07-09), none guessed:

1. **The §5 GPU-vs-CPU gate (`VERIFY`):** fuse the first 4 frames through the GPU kernel and the CPU
   twin into separate volumes; every voxel must agree within abs tol `1e-5`. Measured: **bit-identical**
   (worst deviation `0.0e+00`) — both paths spell every multiply-add as an explicit `fmaf`, so they
   execute the same IEEE-754 operations in the same order (THEORY.md §Numerical considerations).
2. **Ground truth (`GROUND TRUTH`):** the fully-fused TSDF compared against the analytic scene's exact
   SDF, in two shells around the surface. Measured surface-shell (within half a voxel of the true
   surface) mean error **1.47e-2 m**, max **1.14e-1 m**; the max lives in a small (<1%), physically
   explained tail near where the sphere is closest to the plane, where this project's fixed-elevation
   camera orbit gives every view the same shallow, near-grazing incidence angle — the classic
   projective-TSDF bias that constant per-frame weighting does not average away (THEORY.md explains the
   mechanism and shows the error histogram). Bounds (`0.13 m` max, `0.02 m` mean) carry documented
   headroom over these measurements.
3. **Mesh checks (`MESH`):** the GPU triangle count must equal an order-independent CPU recount
   *exactly* (measured: **54822 == 54822**), sit inside a wide sanity range `[40000, 100000]`, and every
   emitted vertex must land on the analytic surface within `0.02 m` (measured max **1.04e-2 m**).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the volume layout, the analytic scene definition, the camera
   structs, and the determinism contract: this project's one-place contracts.
2. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: load poses → render depth by ray
   casting → verify → fuse all frames → check against ground truth → extract the mesh → write artifacts.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the integration oracle (a line-by-line twin of the
   kernel) and the marching-cubes triangle-count oracle.
4. [`src/kernels.cu`](src/kernels.cu) — the heart: the voxel-parallel TSDF update and the cell-parallel,
   atomic-append marching-cubes kernel. The single most interesting thing: the `__constant__`-memory
   case table and how one warp's divergent-but-narrow table lookups still run fast.
5. [`src/mc_tables.h`](src/mc_tables.h) — the 256-case marching-cubes triangulation table, with
   provenance and the corner/edge numbering diagram every other file assumes.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Newcombe, Izadi et al. (2011), "KinectFusion: Real-Time Dense Surface Mapping and Tracking"** — the
  paper this project reimplements the fusion half of (didactically; no ICP tracking here).
- **nvblox** (NVIDIA) — voxel-hashed TSDF + ESDF generation for unbounded scenes, running inside Isaac;
  this project's fixed 128³ dense grid is nvblox's simplest possible special case.
- **Voxblox** (ETH Zurich) — incremental TSDF/ESDF mapping built specifically for fast planner queries;
  compare its ESDF propagation with `07.09`'s jump-flooding distance transform downstream of this project.
- **Open3D** (`ScalableTSDFVolume`) — a widely used open-source integration + mesh-extraction pipeline;
  a good reference implementation to diff this project's simplified update rule against.
- **Lorensen & Cline (1987), "Marching Cubes"** — the original algorithm; the 256-case table in
  `src/mc_tables.h` traces directly back to it (via Paul Bourke's widely reproduced reference tables).
- **InfiniTAM / VDBFusion** — other open research/production dense-mapping codebases worth reading once
  this project's mental model is solid.

## Exercises

1. **Plot the artifacts:** open `demo/out/mesh.obj` in any 3-D viewer (Blender, MeshLab, even a
   Three.js loader) and `demo/out/tsdf_slice.pgm` in any image viewer. Identify the sphere, the plane,
   and the black "shadow" of never-observed space directly beneath the sphere.
2. **Vary the camera path:** edit `scripts/make_synthetic.py` to add elevation variation (e.g., alternate
   high and low orbits) instead of a single constant-height circle, regenerate the sample, and rerun.
   Measure whether the ground-truth error tail (see Expected output) shrinks — it should, because
   varying incidence angle is exactly what averages the projective-TSDF bias away.
3. **Weight by incidence:** change the per-frame observation weight in `tsdf_integrate_kernel` (and its
   CPU twin) from a constant `1.0` to something proportional to the surface-normal/ray-incidence cosine
   (you will need the analytic surface normal, which this scene's closed form provides), and measure the
   ground-truth error improvement.
4. **Two-pass marching cubes:** replace the atomic-append emission with the classic two-pass
   count-then-scan (an exclusive prefix sum over per-cell triangle counts, then a deterministic-order
   write) and confirm the triangle SET is unchanged while the buffer ORDER becomes stable run to run.
5. **Weld the mesh:** `write_obj` emits an un-indexed mesh (each triangle carries its own 3 vertices).
   Add a spatial hash to dedupe near-identical vertices into a proper indexed OBJ and measure the file
   size reduction.

## Limitations & honesty

- **No pose tracking.** Real KinectFusion estimates each frame's pose by ICP against the model being
  built; this project takes poses as given (the committed sample) so the fusion math itself can be
  checked against exact analytic ground truth. Pose tracking is `02.06` (GPU ICP registration); chaining
  the two is exactly SYSTEM_DESIGN's Chain A.
- **Constant-weight fusion — a real, measured bias.** Every observation counts for weight 1 regardless
  of viewing angle; the ground-truth check's headline number (max error ~0.11 m, a near-full truncation
  band) is a genuine consequence of this choice at grazing incidence, not a bug — see `main.cu`'s
  `GROUND-TRUTH CHECK` comment and THEORY.md §Numerical considerations for the full mechanism and how
  production systems avoid it.
- **Fixed dense volume, not voxel hashing.** The 128³ grid (2.56 m cube) is allocated in full up front;
  production systems (nvblox, Voxblox) hash sparse voxel blocks so the map can cover an unbounded scene
  without a fixed bounding box. That extension is real engineering, not more physics, and is out of
  scope here (see the next catalog bullet's project for it).
- **Atomic-append triangle order is nondeterministic.** The triangle *set* and *count* are exact,
  deterministic invariants (checked every run against a CPU recount); the *order* triangles land in the
  output buffer depends on which GPU thread's `atomicAdd` wins the race, and is **not** checked or
  claimed to be stable. THEORY.md explains why this is a set/count invariant, not a byte-for-byte one.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **Sim-validated only (CLAUDE.md §1):** every input here is synthetic (a closed-form scene, ray-cast
  depth, and a scripted camera path); nothing was run against a real depth sensor or a real robot's
  pose estimate, and no claim of readiness for real hardware is made. A real deployment would sit behind
  the full testing ladder in [`PRACTICE.md`](PRACTICE.md) §3.
