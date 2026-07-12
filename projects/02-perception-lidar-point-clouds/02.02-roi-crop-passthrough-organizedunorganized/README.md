# 02.02 — ROI crop, passthrough, organized↔unorganized conversion kernels

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `ROI crop, passthrough, organized↔unorganized conversion kernels`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project is a **stream-compaction masterclass** built around the six "glue" kernels that sit
between every LiDAR driver and every consumer in a real perception pipeline: a **passthrough** filter
(axis-range), a **box ROI crop** (AABB), a **frustum crop** (keep only what a camera can see — the
sensor-fusion use case), the **chained-vs-fused** comparison of applying three filters as three passes
vs. one, and the two **organized↔unorganized conversion** directions every LiDAR pipeline needs at
least once. All six reduce to ONE underlying primitive — a parallel exclusive prefix sum ("scan") —
implemented **by hand** (a two-level, work-efficient Blelloch scan) and cross-checked bit-exact
against `thrust::exclusive_scan` and a plain CPU serial scan. A learner who finishes this project
understands exactly how stream compaction works on a GPU, from the up-sweep/down-sweep tree to the
scatter that consumes it, and has seen the same primitive applied to six genuinely different problems.
Every component in the catalog bullet is implemented (no documented-only pieces): passthrough, box
ROI, organized→unorganized, and unorganized→organized are all real, gated kernels; the frustum crop
and the chained-vs-fused comparison are this project's own additions on top of the bullet, included
because they are the natural next lesson once compaction exists (see "The algorithm in brief").

> **Template placeholder notice.** As scaffolded, `src/` contained a tiny SAXPY placeholder to
> validate the toolchain. It has been fully replaced by the real implementation described below.

## What this computes & why the GPU helps

Every kernel here is either a **map** (predicate evaluation, order-preserving scatter — one thread,
one point, fully independent) or the **scan** pattern (the exclusive prefix sum that turns keep/drop
flags into destination addresses) — with one **scatter-with-collision-resolution** kernel
(unorganized→organized) that races threads against each other via `atomicMin` on a 64-bit encoded key.
The bottleneck being parallelized is exactly the one every point-cloud pipeline pays over and over:
deciding, independently per point, whether to keep it, and then packing survivors into a dense array
without serializing the whole operation on a single thread. A naive GPU compaction (a single thread
walking the array and appending) would throw away all but one thread's worth of parallelism; the
scan-based approach here keeps every stage — predicate, scan, scatter — fully parallel, `O(n)` total
work with only `O(log n)` parallel depth for the scan stage (THEORY.md "The math" derives both
bounds).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception, immediately downstream of the LiDAR driver and upstream of
  essentially every other perception/mapping/planning consumer that touches a point cloud — the
  "glue" layer, not an algorithm layer.
- **Upstream inputs:** the LiDAR driver's raw scan (a `PointCloud`-shaped organized grid,
  SYSTEM_DESIGN.md §3.6), and, upstream of THAT, project **02.20**'s per-channel intensity
  calibration (a real driver applies its own calibration before this project's kernels ever see the
  data).
- **Downstream consumers:** *literally everything else in this domain.* Named explicitly: **02.01**
  (voxel-grid downsampling — consumes an already-cropped/unorganized cloud so it never wastes work
  downsampling points nobody wants), **02.03** (ground segmentation), **02.04** (Euclidean
  clustering), **02.06** (ICP registration) all consume this project's UNORGANIZED output;
  range-image algorithms like **02.12** (depth-clustering on a range image) consume the ORGANIZED
  form directly, which is exactly why the unorganized→organized direction exists. The **frustum
  crop** specifically feeds camera-LiDAR sensor fusion: **02.17** (LiDAR-camera projection/coloring)
  and the z-buffer occlusion technique in **01.18** (depth completion) — both need "only the points
  this camera can see," precisely this project's frustum predicate.
- **Rate / latency budget:** a spinning LiDAR reports at 10-20 Hz (SYSTEM_DESIGN.md §1); these
  kernels must be **nearly free** against that budget, not a bottleneck. Measured on the owner's
  RTX 2080 SUPER (sm_75), a single-revolution scan of ~12,400 points: passthrough/box/frustum/fused
  compaction each run in ~0.12-0.2 ms; organized→unorganized ~1.3-1.6 ms; the unorganized→organized
  collision scatter ~0.13-0.3 ms (exact figures vary run to run — see `[time]` lines in the demo
  output, a teaching artifact never a benchmark claim). At worst-case totals under ~3 ms, this
  pipeline consumes well under 10% of a 20 Hz (50 ms) scan-to-scan budget even run END TO END with no
  overlap — the "nearly free" claim, backed by a real measurement, not an assumption.
- **Reference robot(s):** the **warehouse AMR** (ROI-cropping the scan to the vehicle's stopping
  distance before obstacle detection) and the **autonomous-vehicle stack** (frustum-cropping LiDAR
  points to each mounted camera's field of view for fusion) both use every kernel in this project;
  the **quadrotor** reference robot would use the same passthrough/box primitives at a smaller scale.
- **In production:** these exact predicate shapes, hand-tuned per consumer and versioned alongside
  sensor calibration (PRACTICE.md §3), plus a vehicle-body self-hit mask this project's clean
  synthetic scene never needed (also PRACTICE.md §3) — replacing this project's illustrative bounds
  with real, measured ones.
- **Owning team:** perception infrastructure — the team that keeps the "glue" between the driver and
  every algorithm team's consumer fast, correct, and shared, distinct from the teams owning the
  algorithms that consume its output (PRACTICE.md §4).

## The algorithm in brief

- **Exclusive prefix scan** (the star): a hand-rolled two-level, work-efficient (Blelloch) scan —
  block-level up-sweep/down-sweep in shared memory, then a second-level scan of the per-block totals,
  then a broadcast pass — cross-checked bit-exact against `thrust::exclusive_scan` (which wraps CUB's
  single-pass decoupled-look-back scan) and a plain CPU serial scan. See
  [`THEORY.md` — "The math" and "The GPU mapping"](THEORY.md#the-math).
- **Predicate → scan → scatter compaction**, applied to three ROI predicates (passthrough, box,
  frustum) plus their conjunction (fused) plus the organized grid's NaN-validity predicate
  (organized→unorganized). See [`THEORY.md` — "The algorithm"](THEORY.md#the-algorithm).
- **Chained vs. fused filtering**: the same three predicates as three sequential compaction passes vs.
  one conjoined pass — bit-identical output, measurably different memory traffic. See
  [`THEORY.md` — "The algorithm"](THEORY.md#the-algorithm).
- **Camera-frustum crop**: a five-plane (near + four image-edge) test derived from pinhole camera
  intrinsics and a LiDAR↔camera extrinsic reused from project 01.18. See
  [`THEORY.md` — "The math"](THEORY.md#the-math).
- **Unorganized → organized scatter with a 64-bit encoded-atomicMin nearest-wins collision policy**,
  extending project 01.18's uint-encoded atomicMin z-buffer trick from 32 to 64 bits. See
  [`THEORY.md` — "The math"](THEORY.md#the-math) and [`kernels.cuh`](src/kernels.cuh).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/roi-crop-passthrough-organizedunorganized.sln`](build/roi-crop-passthrough-organizedunorganized.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration (or `Debug|x64` — both build clean, zero warnings).
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/roi-crop-passthrough-organizedunorganized.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependency: **Thrust** (header-only, part of the CUDA Toolkit — no separate library to
install). `build/*.vcxproj` carries the exact `/Zc:preprocessor` / `/Zc:__cplusplus` / `-std=c++17`
flags Thrust's CCCL headers require under this toolchain, copied verbatim from project 02.01's
ratified precedent (cited in that project's `.vcxproj` and in this project's `kernels.cu`). No other
dependency beyond the CUDA toolkit + C++17 standard library.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample (`data/sample/roi_scan.bin`, ~194 KiB) is entirely **synthetic** (CLAUDE.md §8
default): a single-revolution 16-beam organized LiDAR scan of a virtual warehouse room (scene and
raycaster reused from project 02.01, cited in `scripts/make_synthetic.py`), plus a 39-point "edge
cohort" straddling every predicate boundary this project tests, plus a 200-entry "ghost" second-echo
table for the deterministic collision test. Generated with a fixed xorshift32 seed (42) — regenerate
with `python scripts/make_synthetic.py`. Full field-by-field format, SHA-256 checksum, and an honest
account of a boundary-rounding bug this project's own gates caught during development live in
[`data/README.md`](data/README.md).

## Expected output

`RESULT: PASS` means every one of the following agreed (see `demo/expected_output.txt` for the exact
stable lines, and `THEORY.md` "How we verify correctness" for why no comparison in this project needs
a floating-point tolerance — every check is either an exact integer or a byte-exact float copy):

- `VERIFY(organized_to_unorganized)` — GPU vs. CPU vs. the Python generator's own independent tally,
  three-way bit-exact.
- `VERIFY(predicate_correctness)` — passthrough/box/frustum/fused GPU counts and order match CPU,
  and every GPU-kept point independently re-satisfies its own predicate.
- `VERIFY(scan_bitexact)` — hand-rolled Blelloch GPU scan, `thrust::exclusive_scan`, and a CPU serial
  scan agree, integer-exact, on two representative flag arrays.
- `GATE order_preservation`, `GATE frustum_geometry`, `GATE fused_vs_chained`, `GATE roundtrip`,
  `GATE collision_accounting` — see [`demo/README.md`](demo/README.md) for what each one checks.

The demo also writes visual/data artifacts to `demo/out/` (top-view point-cloud renders, an organized-
grid occupancy image before/after the round trip, and a CSV of every measured number) — see
`demo/README.md`.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **start here.** The whole project's contract: data layout,
   predicate formulas, the encoding scheme, every kernel/launcher declaration, heavily commented.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels. Read "THE SCAN CHAPTER" comment block first
   (the didactic heart of this project — the two-level Blelloch scan, bank-conflict honesty included),
   then the predicate kernels, then the unorganized→organized scatter.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracles; read its file
   header for exactly what is shared with the GPU path (data-layout formulas) versus independently
   structured (every algorithm).
4. [`src/main.cu`](src/main.cu) — orchestration: loads the sample, runs every stage in order, prints
   every `VERIFY`/`GATE` line, writes the artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h` (data-file/artifact-dir
   resolution — copied fresh from `docs/PROJECT_TEMPLATE`).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **PCL** (`pcl::CropBoxFilter`, `pcl::PassThrough`, `pcl::ExtractIndices`) — the open-source baseline
  for exactly this project's passthrough/box crop, CPU-only.
- **Thrust / CUB** (`thrust::exclusive_scan`, `cub::DeviceScan::ExclusiveSum`,
  `cub::DeviceSelect::Flagged`) — the production single-pass, decoupled-look-back scan and
  predicate-compaction primitives this project's hand-rolled scan is contrasted against.
- **Blelloch (1990), "Prefix Sums and Their Applications"** — the original work-efficient scan
  algorithm this project implements by hand; Harris/Sengupta/Owens, *GPU Gems 3* ch. 39, "Parallel
  Prefix Sum (Scan) with CUDA" — the canonical CUDA implementation writeup, including the
  bank-conflict-avoiding padding trick this project deliberately omits (and explains why).
- **`sensor_msgs/PointCloud2`** (ROS 2) — the real-world message shape this project's organized/
  unorganized distinction and `is_dense` discussion map onto directly.
- **NVIDIA Isaac ROS / cuPCL** — the GPU-accelerated production equivalents of PCL's CPU filters,
  and the closest shipped analog of this project's frustum-crop sensor-fusion use case.
- **Project 01.18** (depth completion) — the source of the camera↔LiDAR extrinsic and the
  uint-encoded-atomicMin z-buffer trick this project extends to 64 bits; **project 02.01** (voxel-grid
  downsampling) — the source of the scene/raycaster this project's synthetic data reuses, and the
  `atomicAdd`-based compaction contrasted against this project's scan-based one in THEORY.md.

## Exercises

1. Implement the bank-conflict-avoiding padded offset (`idx + (idx >> 5)`) in
   `blelloch_block_scan_kernel` and measure the speedup against `[info] scan_scaling`'s current
   numbers — THEORY.md "The GPU mapping" gives you the formula.
2. Add a **far plane** to the frustum test (turning it into a proper six-plane view frustum) and add a
   far-plane edge-cohort point pair to `scripts/make_synthetic.py` to test it.
3. Implement a THIRD scan level so `launch_scan_blelloch` can handle arrays larger than
   `kScanElemsPerBlock^2` (262,144 elements) — kernels.cu's scaling-limit comment tells you exactly
   where the current implementation would need to grow.
4. Add a vehicle-body self-hit mask (PRACTICE.md §3) as a seventh predicate kernel — a fixed
   per-(ring, azimuth)-cell boolean array ANDed into `valid_predicate_kernel`.
5. Replace the analytical `fused_vs_chained_bytes` estimate in `main.cu` with a real Nsight Compute
   memory-traffic measurement and compare the two.

## Limitations & honesty

- The frustum crop deliberately omits a **far plane** — passthrough/box already bound range, and a
  camera has no natural far plane the way a rasterizer's clip volume does (THEORY.md "The math"
  states this scoping choice explicitly).
- The organized→unorganized dropout model (5% independent per-return) is a **lumped stand-in** for
  several physically distinct phenomena (absorption, specular deflection, max-range falloff, sun
  noise — THEORY.md "The problem") — a real sensor's dropout rate is angle-, material-, and
  range-dependent, not a flat constant.
- The synthetic scene has **no per-ring dropout correlation** — a real degrading slip-ring/optical
  rotary joint tends to fail per-ring, not uniformly across rings (PRACTICE.md §1); this project's
  model does not capture that failure mode.
- No vehicle-body self-hit masking is modeled (PRACTICE.md §3's "unglamorous production reality") —
  the synthetic scene has no robot chassis for a real LiDAR to see part of.
- The bank-conflict-avoiding padded offset for the hand-rolled scan is **documented but not
  implemented** (THEORY.md "The GPU mapping" explains the trade and gives the formula; Exercise 1
  above is exactly this).
- This project's output could feed a real robot's obstacle-avoidance or fusion pipeline — everything
  here is **sim-validated only, not safety-certified** (CLAUDE.md §1); a wrong ROI bound silently
  hides real obstacles from whatever consumes this project's output, and PRACTICE.md §3's safe-testing
  ladder must be followed in full before any of this code is trusted near a moving real robot.
