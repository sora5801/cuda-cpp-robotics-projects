# 02.06 — ICP: point-to-point → point-to-plane → GICP, all batched

**Difficulty:** ★ beginner · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `★ ICP: point-to-point → point-to-plane → GICP, all batched`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**ICP (Iterative Closest Point) answers the question every mobile robot asks dozens of times a
second: "I have two point clouds of the same place — what rigid motion aligns them?"** This project
builds that loop entirely on the GPU and teaches it as a *progression*: **point-to-point** ICP (match
nearest points, minimize Euclidean distance) first, then **point-to-plane** ICP (match the same way,
but minimize distance along the matched surface's normal instead), run side by side on the same
synthetic scene so the learner *measures* — not just reads about — why point-to-plane is the
practitioner's default on structured environments. **GICP**, the catalog bullet's third rung
(covariance-to-covariance matching), is taught in full in [`THEORY.md`](THEORY.md) and documented as a
milestone rather than implemented (see [Limitations & honesty](#limitations--honesty) for the exact
scoping and CLAUDE.md §2/§13's reduced-scope rule for bundled catalog bullets).

The demo runs both variants on **two synthetic pairs**: a ~30,000-point "room" (floor + two walls
meeting at a corner + a box) and a small 5,000-point pair with a differently-tilted rotation axis, each
transformed by a **known** ground-truth rigid motion (5–10° rotation, 0.2–0.4 m translation) plus
independent sensor noise. Four closed-loop registrations run total (2 pairs × 2 variants); every one is
checked against the committed ground truth, and the measured result is exactly the one the catalog
promises: on the wall-dominated main pair, point-to-plane converges in **6 iterations** versus
point-to-point's **48** — an 8× difference, measured, not asserted (see the `[info]` lines in
[`demo/expected_output.txt`](demo/expected_output.txt)'s companion run, and
[`demo/out/convergence.csv`](demo/README.md) for the full curve).

## What this computes & why the GPU helps

Every ICP iteration does three GPU-shaped jobs: (1) for every source point, find its nearest neighbor
in the target cloud — a **map + per-thread search**; (2) turn every valid match into a small
least-squares contribution and sum ~30,000 of them into one 6×6 linear system — a **reduction**; (3)
(once per target cloud, not per iteration) estimate a surface normal at every target point from its
16 nearest neighbors via PCA — another **map + per-thread search**, followed by an in-register
eigensolve.

- **Pattern 1 — brute-force nearest-neighbor search (map + search):** one thread per source point,
  scanning the whole target cloud; embarrassingly parallel across points, honestly O(N·M) (a KD-tree
  is project 02.05's job — see [Exercises](#exercises)).
- **Pattern 2 — two-stage reduction:** per-point 27-scalar contributions (a 6×6 symmetric matrix's 21
  unique entries + a 6-vector) reduced within each GPU block via shared memory, then finished with a
  small host-side sum — this project's central NEW GPU concept beyond the repo's usual
  thread-per-problem shape (see 33.01/09.01/08.01), because here many independent threads must combine
  into *one* shared answer.
- **Pattern 3 — PCA normal estimation:** the same brute-force-search shape as pattern 1, feeding an
  in-register 3×3 Jacobi eigensolve per thread — the repo's second appearance of "thread = one small
  linear-algebra problem" (33.01 taught the first, for a *solve*; this one is an *eigendecomposition*).
- **Measured reality (RTX 2080 SUPER):** on the 30,000-point main pair, one full ICP iteration (map +
  search + reduce) costs ~4–5 ms of GPU kernel time; point-to-plane's 6-iteration convergence finishes
  a full registration in ~24–29 ms of kernel time versus point-to-point's ~234–240 ms for 48 iterations
  — the iteration-count win *is* the wall-clock win (`[info]` lines in the demo output).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the perception → state-estimation boundary — ICP consumes a perception-layer
  point cloud and PRODUCES a relative pose, the raw material state estimation turns into a filtered
  trajectory (SYSTEM_DESIGN §1's `PERCEPTION` → `STATE ESTIMATION / WORLD MODEL` arrow).
- **Upstream inputs:** a deskewed, already-motion-corrected `PointCloud` (SYSTEM_DESIGN §3.6's
  message shape) — the job of project 02.08 (motion-distortion correction); this project assumes a
  clean scan and documents that assumption in [Limitations & honesty](#limitations--honesty).
- **Downstream consumers:** odometry and mapping — the recovered `T_target_source` feeds a pose graph
  or a running map estimate (05.09 LIO-style odometry consumes exactly this kind of scan-to-scan/
  scan-to-map result; 05.01 TSDF fusion consumes the resulting pose to place new depth into a voxel
  volume). SYSTEM_DESIGN §4.1 **Chain A** names this project explicitly:
  `[11.01 GPU LiDAR simulator] -> [02.06 ICP registration] -> [05.01 TSDF fusion] -> ...`
- **Rate / latency budget:** LiDAR → perception/mapping runs at **10–20 Hz**, budget **< 100 ms** per
  scan (SYSTEM_DESIGN item 1's rate table) — this project's measured per-iteration cost (~4–5 ms on
  30,000 points) times point-to-plane's ~6-iteration convergence (~25–30 ms total) comfortably fits
  that budget with room to spare; point-to-point's ~48-iteration convergence (~235 ms) would not.
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN §2.1 — LiDAR-based localization against
  a site map is exactly this loop) and the **autonomous-vehicle stack** (§2.5 — LiDAR odometry/
  localization against an HD map), both of which name domain 02 explicitly in their block diagrams.
- **In production:** a KD-tree- or voxel-grid-accelerated correspondence search (PCL, Open3D, KISS-ICP)
  replaces this project's brute-force O(N·M) scan; the linear-algebra core (build a normal system,
  solve, update) is otherwise the same shape this project implements.
- **Owning team:** perception (the correspondence search and normal estimation) handing off to
  SLAM/state-estimation (consuming the pose) — SYSTEM_DESIGN item 5's org map; adjacent to the mapping
  team that owns 05.01/05.09.

## The algorithm in brief

- **Correspondence search** — brute-force GPU nearest-neighbor, one thread per source point, with a
  distance-gate rejection. → [THEORY.md](THEORY.md) §The algorithm.
- **Point-to-point linearized ICP** — Gauss-Newton on SE(3): residual `r = x - q`, closed-form 6×6
  contribution per point. → THEORY §The math.
- **Point-to-plane linearized ICP** (Low 2004) — residual `e = n·(x - q)` (scalar, projected onto the
  matched surface normal); converges faster on structured/planar scenes because it does not fight the
  "free slide" along a flat surface. → THEORY §The algorithm.
- **PCA surface normals** — per-target-point covariance of its 16 nearest neighbors, eigendecomposed
  via an in-register cyclic Jacobi sweep; the smallest eigenvalue's eigenvector is the normal. →
  THEORY §The math.
- **The reduction** — per-point 27-scalar contributions (21 unique H entries + 6 g entries), block-
  reduced via shared memory, finished on the host in double precision. → THEORY §The GPU mapping.
- **The solve + update** — a 33.01-style 6×6 Cholesky solve (host, double), folded into the running
  pose estimate via a quaternion-composed SE(3) update. → THEORY §Numerical considerations.
- **GICP** (covariance-to-covariance matching, the catalog bullet's third rung) — derived in full in
  THEORY §Where this sits in the real world, shipped as a documented milestone (see
  [Limitations & honesty](#limitations--honesty)).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/icp-point-to-point-point-to-plane-gicp.sln`](build/icp-point-to-point-point-to-plane-gicp.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/icp-point-to-point-point-to-plane-gicp.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
6×6 Cholesky solve and the 3×3 Jacobi eigensolve are hand-rolled (no cuSOLVER); README §Prior art names
where a production stack would reach for a library instead.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including the
**two plotting artifacts** (`aligned.csv`, `convergence.csv`).

## Data

The committed sample is **synthetic by necessity**, not just by repo default: ICP needs a *known*
ground-truth transform to be checkable at all, and no public LiDAR dataset ships that at the precision
this project's gate demands (and KITTI/nuScenes-class datasets carry non-redistribution licenses besides
— `scripts/download_data.ps1`/`.sh` explain the decision and are honest no-ops). Two pairs, ~836 KiB
total, in a small documented binary format (`data/README.md`): a 30,000-point "room" (floor + 2 walls +
box) and a small 5,000-point second pair with a differently-tilted ground-truth rotation. Regenerate
with `python scripts/make_synthetic.py --seed 42` (byte-identical; checksums in `data/README.md`).
Details and the byte-exact format: [`data/README.md`](data/README.md).

## Expected output

Fifteen stable lines — banner, `PROBLEM:`, two `SCENARIO:` lines, three `VERIFY:` lines, five `CHECK:`
lines, two `ARTIFACT:` lines, `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt); every `[info]` line above them carries the
actual measured numbers (not diffed — see `src/main.cu`'s "Output contract"). Three independent
verifications:

1. **The §5 GPU-vs-CPU gate (VERIFY stage, 3 checks):** correspondence indices match the CPU oracle
   **exactly** (measured: 0/30,000 mismatches); both normal-system variants' 27 reduced scalars agree
   with the CPU oracle within relative tolerance 1e-3 (measured worst: 1.5e-07 point-to-point, 3.3e-08
   point-to-plane).
2. **The ground-truth pose gate (CHECK, 4 checks):** every one of the 4 (pair, variant) runs must
   recover the committed ground-truth pose within 1.00° rotation and 0.050 m translation. Measured
   worst case: pair1 point-to-point at 0.119° / 0.007 m — both roughly 7–8× under threshold.
3. **The taught superiority (CHECK, 1 check):** point-to-plane must converge in fewer iterations than
   point-to-point on the wall-dominated main pair. Measured: 6 vs. 48 iterations (pair0), 5 vs. 32
   (pair1) — an 8× and 6.4× difference.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the project's single source of truth: point-cloud layout,
   the correspondence record, the SE(3) pose representation, and the 27-scalar reduction layout that
   every other file depends on. Read this FIRST.
2. [`src/main.cu`](src/main.cu) — the whole ICP loop in plain sight: transform → correspond → reduce →
   solve → update, four times over (2 pairs × 2 variants); plus the VERIFY stage, the ground-truth
   gate, and the two artifact writers. The SE(3)/quaternion math and the 33.01-style Cholesky solve
   live here (kept on the host, next to the loop that calls them — the same choice 08.01 makes for its
   softmin blend).
3. [`src/kernels.cu`](src/kernels.cu) — the four GPU kernels: `transform_cloud_kernel` (the simplest
   map), `find_correspondences_kernel` (brute-force search), `estimate_normals_kernel` (PCA + Jacobi),
   and `build_normal_system_kernel` — the single most interesting thing in this project: the two-stage
   shared-memory reduction that turns 30,000 independent per-point answers into one 6×6 system.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the line-by-line CPU oracle twin of every kernel
   above; diff it against `kernels.cu` to see exactly what parallelization changed.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Besl & McKay (1992), "A Method for Registration of 3-D Shapes"** — the original point-to-point ICP
  paper; this project's point-to-point variant uses a linearized Gauss-Newton solve instead of their
  closed-form SVD/quaternion solution (THEORY.md explains the trade), but solves the same problem.
- **Chen & Medioni (1992) and Low (2004), "Linear Least-Squares Optimization for Point-to-Plane ICP
  Surface Registration"** — the point-to-plane linearization this project implements almost verbatim;
  Low's technical report is the cleanest derivation available and this project's THEORY.md credits it
  by name at every formula it borrows.
- **Segal, Haehnel & Thrun (2009), "Generalized-ICP"** — the GICP paper: covariance-to-covariance
  matching, this project's documented-only third rung (THEORY.md §Where this sits in the real world
  derives its cost function in full).
- **PCL (`pcl::IterativeClosestPoint`, `pcl::NormalEstimation`)** — the production C++ library
  implementing all three variants (plus GICP) with KD-tree correspondence search; study its API shape
  and its `TransformationEstimationPointToPlaneLLS` class specifically (the direct ancestor of this
  project's point-to-plane kernel).
- **Open3D (`registration_icp`)** — a modern, Python-friendly ICP implementation with GPU tensor
  backends; compare its correspondence-search strategy (KD-tree, voxel-hashed) against this project's
  brute force.
- **KISS-ICP (Vizzo et al., 2023)** — a deliberately simple, robust point-to-point LiDAR odometry
  pipeline; a good next read once this project's variants feel familiar — it shows how far a *simple*
  ICP variant gets with careful engineering around it (adaptive thresholds, motion prediction).
- **FAST-GICP** — a real-time GPU/multi-threaded GICP implementation; the production answer to this
  project's documented-only GICP rung.

## Exercises

1. **Plot the artifacts:** `demo/out/convergence.csv` → RMS vs. iteration, one line per variant. Watch
   point-to-plane hit its floor in a handful of iterations while point-to-point visibly takes longer to
   get there. Then plot `demo/out/aligned.csv` (both `cloud` values as a 3D scatter) and see the
   registered room.
2. **Build the KD-tree:** replace `find_correspondences_kernel`'s O(N·M) brute-force scan with a
   KD-tree or a spatial hash grid (a CPU-built, GPU-traversed tree is the simplest starting point).
   Measure the speedup on the 30,000-point pair and compare against 02.05's approach.
3. **Break the correspondence gate:** set `kDefaultMaxCorrDist` far too small (e.g. 0.05 m) and watch
   both variants fail to converge — explain why from the geometry (how large can the rotation-induced
   displacement be for a point 4 m from the rotation axis, at 7°?).
4. **Implement GICP:** THEORY.md derives the full covariance-to-covariance cost function. Extend
   `IcpMode` with a third value and a third `build_normal_system_kernel` branch; you will need a
   per-source-point covariance too (not just per-target-point), doubling the normal-estimation work.
5. **Fuse the reduction:** move `main.cu`'s host-side block-partial sum onto the GPU (a small second
   kernel with one block) — the same optimization 08.01's README names for its softmin blend. Measure
   the host-download bytes saved per iteration.

## Limitations & honesty

- **GICP is documented, not implemented** — the catalog bullet's third rung. The reduced-scope
  decision (CLAUDE.md §2/§13): implement the taught *progression* (point-to-point → point-to-plane)
  completely, and teach GICP's covariance-to-covariance formulation in full in THEORY.md's "Where this
  sits in the real world" section (what it adds over point-to-plane, its cost function, and its own
  GPU-mapping story) rather than shipping a reduced/toy version of a third algorithm.
- **Brute-force correspondence search, not a KD-tree** — deliberately: this project teaches the
  reduction pattern and the linearized normal-equation math; a spatial acceleration structure is
  project 02.05's dedicated subject and this project's README Exercise 2.
- **The input is a clean, already-deskewed scan** — real spinning LiDAR data is motion-distorted
  within a single sweep; this project assumes that correction already happened (02.08's job) and its
  synthetic data has no motion distortion to correct.
- **Isotropic synthetic noise, not per-beam range noise** — THEORY.md "The problem" is explicit about
  this simplification: real LiDAR noise is 1-D along the beam; this project's synthetic scene generator
  (no per-beam raycasting) applies isotropic 3-D Gaussian jitter instead. 11.01's GPU LiDAR simulator
  is where per-beam noise modeling belongs.
- **Point-to-point uses a linearized Gauss-Newton solve, not Besl & McKay's closed-form SVD** — a
  deliberate unification: both variants share exactly the same 6×6-normal-system machinery (THEORY.md
  explains why this is correct and what it costs relative to the closed form).
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), kernel-only where
  labeled; never a benchmark claim (CLAUDE.md §12).
- **Sim-validated only (CLAUDE.md §1):** this project's output is a *pose* that a downstream mapping or
  planning stack would act on. Everything here ran only against synthetic data; nothing is
  safety-certified, and any real-sensor deployment would need the full validation described in
  [`PRACTICE.md`](PRACTICE.md) §3.
