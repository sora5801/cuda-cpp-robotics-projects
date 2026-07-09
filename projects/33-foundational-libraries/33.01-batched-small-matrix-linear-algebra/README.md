# 33.01 — Batched small-matrix linear algebra (3×3, 4×4, 6×6 — the robotics sizes)

**Difficulty:** ★ beginner · **Domain:** 33. Foundational GPU Libraries (Build-Your-Own)

> Catalog bullet (source of truth, verbatim): `★ Batched small-matrix linear algebra (3×3, 4×4, 6×6 — the robotics sizes)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds the smallest useful piece of a GPU robotics stack: a library that multiplies and
solves **hundreds of thousands of tiny matrices at once** — 3×3, 4×4, and 6×6, because those are the
sizes robotics actually lives in (rotations, homogeneous transforms, spatial/mass matrices). It
implements two operations end to end: batched matrix multiply `C_k = A_k·B_k` and batched Cholesky
solve `A_k·x_k = b_k` for symmetric positive-definite systems. After studying it you will understand
the **thread-per-problem batch pattern** — one GPU thread owns one whole matrix problem, solved
entirely in registers — which is the same pattern the flagship projects for forward kinematics
(09.01), MPPI control (08.01), and particle filters (04.01) are built on. The demo verifies every
GPU result against a plain-C++ CPU oracle and prints a PASS/FAIL verdict plus honest timing lines.

## What this computes & why the GPU helps

A 6×6 Cholesky factorization is ~100 flops — far too small to parallelize *internally* (it wouldn't
even fill one 32-thread warp). The bottleneck in robotics is never one tiny matrix; it is the
**batch**: a mass matrix per sampled configuration, a covariance per particle, a normal-equation
solve per IK seed — 10⁴–10⁶ independent problems per control or planning cycle. That independence is
the parallelism:

- **Pattern:** batched map / batched solve — one thread = one matrix problem, no inter-thread
  communication at all (no shared memory, no atomics, no synchronization).
- **Memory story:** each thread stages its matrices into **registers** (compile-time sizes + full
  unrolling make that possible) and touches global memory the minimum number of times.
- **Contrast worth learning:** a *large*-matrix GEMM is the opposite regime — massive data reuse
  across threads, shared-memory tiling, tensor cores. Knowing which regime you are in is half of
  GPU engineering; [THEORY.md](THEORY.md) §The GPU mapping develops both sides.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** cross-cutting **infrastructure/foundational library** (domain 33) — it has no
  single layer of its own; it sits *underneath* estimation, planning, and control, the way BLAS sits
  underneath scientific code.
- **Upstream inputs:** batches of small matrices produced by other modules — covariance blocks from
  estimators (04.01 particle filter, 04.02 EKF banks), mass matrices/Jacobians from batched dynamics
  (09.01, 09.03), damped normal equations `JᵀJ+λI` from batched IK (09.05), per-rollout dynamics
  inside sampling controllers (08.01 MPPI). Shape: contiguous row-major float arrays, matrix k at
  offset `k·n·n` (the layout contract in [`src/kernels.cuh`](src/kernels.cuh)).
- **Downstream consumers:** the same modules, one step later in their pipelines — solved
  accelerations to integrators, whitened residuals to weight updates, IK steps to seed ranking.
- **Rate / latency budget:** whatever loop it is embedded in — realistically a 100–400 Hz estimator
  tick (≤2.5 ms total) or a 10–50 Hz planner cycle (≤20 ms), per SYSTEM_DESIGN item 1. The measured
  kernel times here (hundreds of microseconds for 10⁵-problem batches on an RTX 2080 SUPER) fit
  those budgets with room to spare — which is exactly why the pattern is production-relevant.
- **Reference robot(s):** the **6-DoF manipulator work cell** (batched IK + mass-matrix solves) and
  the **quadruped** (whole-body control solves, sampling controllers) exercise it most directly;
  SYSTEM_DESIGN's composition chains route both through 09.x/08.x, which reuse this pattern.
- **In production:** `cublasSgemmBatched` / `cublasSgemmStridedBatched` for the multiplies,
  `cusolverDnSpotrfBatched`+`potrsBatched` for the solves, MAGMA's batched kernels, or — very often —
  the operation is *fused into a bigger kernel* exactly the way this repo's 08.01/09.01 do, which is
  why hand-rolling it is worth learning.
- **Owning team:** controls/autonomy or the simulation & tools team, whichever owns the GPU compute
  foundations at a robotics company (SYSTEM_DESIGN item 5); such kernels are typically maintained as
  an internal "robotics math" library.

## The algorithm in brief

- **Batched matrix multiply** — textbook row·column product, fully unrolled at compile-time sizes
  3/4/6; explicit FMA accumulation. → [THEORY.md](THEORY.md) §The algorithm.
- **Batched Cholesky solve** — column Cholesky `A = L·Lᵀ` (no pivoting needed for SPD), then forward
  substitution `L·y = b` and back substitution `Lᵀ·x = y`, all in registers; a non-positive pivot
  marks the system non-SPD and fills that solution with NaN (loud, propagating failure). →
  [THEORY.md](THEORY.md) §The algorithm, §Numerical considerations.
- **Runtime→compile-time dispatch** — callers pass `n ∈ {3,4,6}`; a switch instantiates the right
  template so matrices live in registers. Unsupported sizes abort loudly, never fall back silently.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/batched-small-matrix-linear-algebra.sln`](build/batched-small-matrix-linear-algebra.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/batched-small-matrix-linear-algebra.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA toolkit runtime + C++17 standard library
only. cuBLAS/cuSOLVER are deliberately *not* linked; they are the comparison point, not a dependency.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Fully **synthetic** (labeled so everywhere): `data/sample/smallmat_sample.csv` (~59 KiB, committed)
holds 64/32/16 matmul input pairs at n=3/4/6 plus 32 SPD 6×6 systems, generated by
[`scripts/make_synthetic.py`](scripts/make_synthetic.py) with seed 42 — byte-identical on
regeneration. SPD matrices are built as `A = G·Gᵀ + n·I`, which guarantees positive definiteness and
single-digit condition numbers. No public dataset applies to random test matrices, so
`scripts/download_data.ps1` is an honest no-op. Format, field meanings, and the SHA-256 checksum:
[`data/README.md`](data/README.md).

## Expected output

A passing run prints six stable lines — banner, `PROBLEM:`, `SAMPLE:`, `SAMPLE RESULT: PASS`,
`BATCH:`, and `RESULT: PASS` — which [`demo/expected_output.txt`](demo/expected_output.txt) checks as
a subset diff (machine-dependent `[info]`/`[time]` lines are deliberately unchecked). Verification:
every problem is computed twice — GPU kernels vs the single-threaded oracle in
[`src/reference_cpu.cpp`](src/reference_cpu.cpp) — and compared element-wise within documented
tolerances: **1e-5 absolute** for matmul, **1e-4 relative (with a max(1,|x|) floor)** for solves;
NaN-vs-NaN counts as agreement (shared non-SPD policy), one-sided NaN is an automatic failure. On the
reference machine the worst observed deviations were ~2.4e-07 (matmul) and ~2.2e-08 (solve) — two
orders inside tolerance. No artifact file is written; the result of a correctness check is text.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: arguments, sample loading, deterministic batch
   generation, CPU reference, GPU path, tolerance checks, timing, the output contract.
2. [`src/kernels.cuh`](src/kernels.cuh) — the interface, and the **one place** the batch memory
   layout is defined.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plain-C++ correctness oracle, written as a
   line-by-line twin of the kernels.
4. [`src/kernels.cu`](src/kernels.cu) — the heart: both kernels plus the runtime→template dispatch.
   The most interesting single thing in the project is `batched_cholesky_solve_kernel` — a complete
   factorize-and-solve living entirely in one thread's registers.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and why they are copied, not shared.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **cuBLAS batched GEMM** (`cublasSgemmBatched`, `...StridedBatched`) — the production answer to this
  project's matmul; compare its pointer-array vs strided APIs with our layout contract.
- **cuSOLVER batched Cholesky** (`cusolverDnSpotrfBatched`/`potrsBatched`) — production batched SPD
  solves; note it reports per-matrix info flags where we use the NaN policy.
- **MAGMA** (batched routines) — the academic reference for small-size batched linalg on GPUs; its
  papers document the layout and register/shared trade-offs we discuss in THEORY.md.
- **CUTLASS** — NVIDIA's open GEMM template library; its grouped/batched kernels show what the
  coalescing-optimal version of this project looks like.
- **Eigen** (with fixed-size matrices) — the CPU baseline robotics actually uses; our
  `reference_cpu.cpp` is deliberately what Eigen would do without the expression templates.
- **Featherstone, *Rigid Body Dynamics Algorithms*** — where 6×6 spatial matrices come from; the
  bridge to projects 09.01/09.03.

## Exercises

1. **Provoke the NaN path:** hand-edit one `S6` row in the sample CSV to be non-SPD (e.g. negate a
   diagonal entry), rerun, and watch the failure stay loud — then explain why zeros would have been
   dangerous.
2. **Add batched 3×3 symmetric eigendecomposition** (closed-form, for point-cloud covariance
   normals) following the same thread-per-problem pattern — the natural bridge to project 02.09.
3. **Convert the layout to SoA** (structure-of-arrays) and measure the coalescing gain in Nsight
   Compute; compare against the matrix-contiguous numbers and write up which regime wins and why.
4. **Break the conditioning:** generate SPD matrices as `G·Gᵀ + ε·I` with ε → 1e-6 and watch FP32
   solves degrade; add an FP64 variant and compare (this is the kinematic-singularity story).
5. **Tune `kThreadsPerBlock`** (64/128/256/512) and chart occupancy vs runtime in Nsight — then
   explain why the answer barely moves for this arithmetic-per-thread profile.

## Limitations & honesty

- **FP32 only** — right for well-conditioned teaching inputs; production estimators/dynamics often
  need FP64 or mixed precision near singularities (THEORY.md §Numerical considerations).
- **Sizes fixed to 3/4/6** — the point of the project; a general-n batched library is what
  cuBLAS/MAGMA are for.
- **Matrix-contiguous layout is coalescing-suboptimal, on purpose** — it is the natural robotics
  layout and the honest starting point; the SoA fix is Exercise 3, not hidden.
- **Single stream, no transfer overlap, single-shot timings** — the demo teaches kernels, not
  pipeline engineering (domain 32 owns that); speed-up lines are teaching artifacts from one run on
  one machine (kernel-only vs one CPU core) and vary with GPU clocks.
- **Sample and batch data are synthetic** and labeled so; there is no real-robot data here at all.
- **No motion, no safety surface:** this is library code that commands no hardware, so the repo's
  sim-validated-only caveat applies only transitively — via consumers like 08.01, which carry it
  themselves.
