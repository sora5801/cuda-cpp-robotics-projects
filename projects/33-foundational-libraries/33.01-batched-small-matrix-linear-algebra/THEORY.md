# 33.01 — Batched small-matrix linear algebra (3×3, 4×4, 6×6 — the robotics sizes): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

This project is **purely computational** — no photons, no torques, no contact forces are modeled
here. Per the repo contract, we therefore teach the physics of its nearest physical carriers: the
places in a robot where 3×3, 4×4, and 6×6 matrices are *born*. They are not arbitrary sizes; they
are the dimensions of rigid-body geometry and dynamics (conventions per
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md): SI units, right-handed frames,
`T_parent_child` transform naming):

- **3×3** — rotation matrices `R` (members of SO(3), the group of orientations; orthonormal,
  det = +1); **inertia tensors** `I` (kg·m², from the mass distribution of a physical link: how hard
  it is to angularly accelerate about each axis — a machined aluminum arm link literally *is* this
  matrix, integrated over its geometry); **covariances** of 3-D quantities (m², from sensor noise —
  a stereo camera's depth uncertainty, a LiDAR point's position uncertainty).
- **4×4** — homogeneous transforms `T_parent_child = [R p; 0 1]` (rotation + translation in meters):
  the algebra of "where is the gripper in the base frame", composed down every kinematic chain.
- **6×6** — **spatial** quantities that couple rotation and translation: the spatial inertia of a
  link, the adjoint transforms that move twists/wrenches between frames, and above all the
  **joint-space mass matrix** `M(q)` of a 6-DoF manipulator — the map from joint accelerations to
  joint torques (N·m per rad/s²). `M(q)` is symmetric positive definite for any physical mechanism
  (kinetic energy `½q̇ᵀM q̇` is positive for any motion — physics *guarantees* our SPD assumption for
  this carrier), and it changes with configuration `q`, so it must be recomputed and re-solved
  constantly.

**Where the batches come from — the engineering constraint.** A modern robotics stack rarely wants
*one* of these problems solved; it wants *populations* of them per cycle, inside hard rate budgets
(SYSTEM_DESIGN item 1):

| Consumer | Problems per cycle | Budget |
|---|---|---|
| Sampling controller (08.01 MPPI) | one dynamics evaluation per rollout step: 10⁴–10⁵ | 10–50 Hz loop |
| Batched IK with restarts (09.05) | one `JᵀJ+λI` solve per seed per iteration: 10³–10⁵ | interactive / planner tick |
| Particle filter (04.01) | one covariance/likelihood computation per particle: 10⁵–10⁶ | 100–400 Hz estimator |
| Whole-body control | mass-matrix solves | 0.5–1 kHz loop |

A single CPU core doing 6×6 solves one at a time burns the entire 2.5 ms estimator tick on ~10⁵
problems (measured on the reference machine: ~13 ms for 10⁵ solves). The GPU does the same batch in
~0.2 ms — not by solving any one matrix faster, but by solving *all of them at once*. Converting
"many tiny problems" into "one wide problem" is the foundational skill this project teaches.

## The math

**Problem 1 — batched multiply.** Given batches `{A_k}, {B_k}`, `k = 0..K−1`, each `n×n` (n ∈
{3,4,6}), compute `C_k = A_k·B_k`, i.e. `c_ij = Σ_p a_ip·b_pj`. Cost: `n³` multiply-adds per matrix.
All quantities here are dimensionless test values in [−1,1); real callers attach units (composing
transforms: dimensionless; transforming inertia: kg·m²).

**Problem 2 — batched SPD solve.** Given SPD `{A_k}` and right-hand sides `{b_k}`, solve
`A_k·x_k = b_k`. A symmetric matrix is **positive definite** iff `vᵀAv > 0` for all `v ≠ 0` —
physically, kinetic energy is positive (mass matrices), variances are positive (covariances). Every
SPD matrix has a unique **Cholesky factorization** `A = L·Lᵀ` with `L` lower-triangular and
`l_jj > 0`:

```
l_jj = sqrt( a_jj − Σ_{p<j} l_jp² )            (diagonal)
l_ij = ( a_ij − Σ_{p<j} l_ip·l_jp ) / l_jj     (below diagonal, i > j)
```

Solving then splits into two triangular solves: `L·y = b` (forward substitution), `Lᵀ·x = y` (back
substitution). Cost: `n³/3 + O(n²)` for the factorization plus `2n²` for the solves — half the flops
of LU, with **no pivoting needed**: in exact arithmetic SPD guarantees every pivot `l_jj² > 0`
(each is `vᵀAv` for some nonzero `v`). The positivity test *is* the SPD diagnostic — if it fails,
the input was not SPD (or was numerically singular at the working precision).

**Conditioning.** The relative error of a linear solve amplifies by the condition number
`κ(A) = λ_max/λ_min`. Our generators build `A = G·Gᵀ + n·I` with `G` entries in [−1,1): `G·Gᵀ` is
positive semi-definite, so every eigenvalue of `A` is ≥ n, while `λ_max ≤ n + ‖G‖² ≤ n + n²` — hence
`κ(A)` stays single-digit and FP32 keeps ~6 significant digits through the solve. This is a
deliberate *teaching* choice: it makes GPU-vs-CPU disagreement attributable to code, not to
amplified rounding. The ill-conditioned regime is Exercise 4.

## The algorithm

Per problem `k` (each fully independent of every other — the crucial structural fact):

1. **Multiply:** three nested loops over (i, j, p); `n³` fused multiply-adds; write `C_k`.
2. **Solve:** (a) column-by-column Cholesky as in the equations above, overwriting a working copy of
   `A_k`'s lower triangle with `L` in place (entry (i,j) of `A` is never needed after `l_ij` exists);
   (b) forward substitution; (c) back substitution — the working vector goes `b → y → x` in place.
   A non-positive pivot sets a flag; the affected `x_k` is filled with NaN at the end.

**Complexity.** Serial: `O(K·n³)` — a CPU core visits every problem in turn. Parallel: the *work* is
the same `O(K·n³)`, but the *span* (critical path) is one problem, `O(n³)` ≈ 100 flops. With `K`
hardware threads the batch finishes in the time of one tiny problem plus memory traffic — the
textbook definition of an embarrassingly parallel workload. There are no data structures beyond flat
arrays: matrix `k` lives at offset `k·n·n`, row-major (defined once in
[`src/kernels.cuh`](src/kernels.cuh)).

## The GPU mapping

**Thread-to-data mapping: one thread = one whole problem.**

```
global thread id  k = blockIdx.x * blockDim.x + threadIdx.x
        thread k  owns  A_k, B_k → C_k   (or A_k, b_k → x_k)
grid: ceil(K/256) blocks × 256 threads;  tail threads exit at the k >= count guard
```

Why not split one matrix across threads? A 6×6 Cholesky is ~100 dependent flops — less work than
one warp's width, dominated by *sequential* dependencies (each column needs the previous ones). At
these sizes, intra-matrix parallelism buys synchronization costs and nothing else. (At n ≈ 32–128
the answer changes: MAGMA uses a warp per matrix; at n ≥ 10³ full tiled GEMM takes over. Knowing
the crossover regimes is the transferable lesson.)

**Registers as the working memory.** The kernels are C++ templates on `N`, so `float m[N*N]` has a
compile-time size and `#pragma unroll` turns every loop into straight-line code where **every array
index is a literal**. That is the precondition for the compiler to place the array in registers —
the register file is not runtime-indexable, so any dynamically-indexed array would spill to "local"
memory, which physically lives in global DRAM (a silent 100× latency cliff; the single most common
performance bug in this pattern). Cost at N=6: ~48 registers/thread — heavy, so fewer warps fit per
SM (lower **occupancy**), but each thread carries so much arithmetic per byte loaded that latency
stays hidden. Occupancy is a means, not a score to maximize.

**Why no shared memory.** Shared memory earns its keep when threads *reuse each other's data*
(GEMM tiling). Here thread `k`'s matrices are touched by thread `k` alone — zero cross-thread reuse
— so shared memory would add copying for nothing. Its absence is a design statement, not an
omission.

**Coalescing — the honest weakness.** With matrix-contiguous layout, at a given instruction the 32
threads of a warp load elements `36·k + c` for 32 different `k` — addresses 144 bytes apart, so the
warp touches ~32 different 128-byte segments instead of 4: poorly coalesced. The fix real libraries
use is an SoA/interleaved layout (element `c` of all matrices contiguous), restoring perfectly
coalesced loads at the price of a transpose at the API boundary. We keep matrix-contiguous because
it is the layout robotics code naturally has, and measuring the SoA gain yourself is Exercise 3.

**The dispatch trick.** Callers have a runtime `n`; registers need a compile-time `N`. The host
launcher bridges with `switch (n) { case 3: kernel<3><<<...>>>; ... }` — three instantiations exist
in the binary, anything else aborts loudly (no silent slow fallback). No CUDA library calls are
used anywhere — that is the point of domain 33; the library equivalents are named in
§Where-this-sits and README §Prior art.

## Numerical considerations

- **FP32 rounding & FMA contraction.** The GPU accumulates with explicit `fmaf` (one rounding per
  multiply-add); the CPU oracle leaves contraction to the host compiler. Same math, legitimately
  different rounding — so bit-equality is the *wrong* test and a tolerance is the right one. The
  measured evidence: worst matmul deviation 2.4e-07 abs, worst solve deviation 2.2e-08 rel on the
  reference machine — pure rounding noise, two orders below the tolerances.
- **The SPD pivot test** doubles as input validation: `diag ≤ 0` ⇒ not SPD at FP32. Policy: fill
  that `x_k` with **NaN** and keep computing the rest of the batch. NaN propagates through any
  downstream arithmetic, so a consumer cannot mistake a failed solve for an answer — zeros would
  look plausible and be dangerous. Both implementations share the policy, so the comparator can
  treat NaN-vs-NaN as agreement and one-sided NaN as a real divergence.
- **Ill-conditioning is the robotics hazard.** Near kinematic singularities `JᵀJ` becomes
  near-singular and FP32 solves lose most of their digits; mass matrices of chains with wide mass
  ratios similarly. Production remedies: damping (`+λI`, Levenberg–Marquardt — note our generator's
  `+n·I` is the same trick worn as a guarantee), FP64 promotion for the factorization, or iterative
  refinement. This demo deliberately stays well-conditioned; Exercise 4 breaks it on purpose.
- **Determinism.** Inputs come from a hand-rolled xorshift32 (`(bits >> 8) * 2⁻²⁴`, exact in FP32)
  rather than `std::uniform_real_distribution`, whose float output is *not* specified identically
  across standard libraries — the committed expected output must reproduce on Windows *and* Linux.
  The kernels themselves are deterministic (no atomics, no reduction reordering, fixed schedules do
  not affect results because threads share nothing).
- **Not applicable here** (stated per contract): angle wrapping, quaternion drift, stiff ODEs —
  none arise; they belong to the consumers (09.x, 08.x) of this library.

## How we verify correctness

Two-stage, both stages GPU-vs-CPU on identical inputs
([`src/reference_cpu.cpp`](src/reference_cpu.cpp) is written as a line-by-line twin of the kernels —
diff them to see that only the parallel plumbing differs):

1. **Sample stage** — the committed synthetic CSV (64/32/16 matmul pairs, 32 SPD systems; seed 42;
   strict loader that aborts on any malformed row). Small enough to eyeball, committed so the check
   runs offline on every clone.
2. **Batch stage** — 200,000 pairs per size + 100,000 solves, regenerated deterministically in
   memory each run. Large enough to exercise many blocks and the ragged tail (the sample counts are
   deliberately smaller than one 256-thread block, covering the other extreme).

**Tolerances and what each catches.** Matmul: `|Δ| ≤ 1e-5` absolute (outputs bounded by 6 for
[−1,1) inputs; genuine indexing/mapping bugs produce errors of order 1, so 100× rounding headroom
masks nothing). Solve: `|Δ| ≤ 1e-4 · max(1, |x_cpu|)` per element — relative because solution
magnitudes vary, floored so near-zero entries do not demand impossible absolute precision. One-sided
NaN fails instantly (policy divergence); both-NaN agrees. Edge cases exercised: ragged tail (sample
counts), empty-batch no-op (guarded in the launchers), non-SPD path (unit-exercisable via
Exercise 1). The demo exits nonzero on any failure, and `demo/expected_output.txt` pins the stable
output lines.

## Where this sits in the real world

- **cuBLAS** `Sgemm(Strided)Batched` and **cuSOLVER** `Spotrf/SpotrsBatched` are the production
  versions of exactly these two operations — same thread-per-problem idea at small sizes, plus the
  interleaved layouts, pointer-array APIs, and per-matrix `info` flags (where we chose NaN). If your
  batch is large, standalone, and standard-shaped: use them, don't hand-roll.
- **MAGMA**'s batched-BLAS papers (Dongarra et al.) document the size regimes: thread-per-matrix →
  warp-per-matrix → block-per-matrix as n grows; this project implements the first regime and tells
  you where it ends.
- **CUTLASS** grouped/batched GEMM shows the coalescing-optimal, tensor-core-capable endpoint of
  Exercise 3.
- **The fusion argument** — the reason hand-rolling stays relevant: production robotics kernels
  rarely call a library per 6×6 solve; they *fuse* the solve into a bigger kernel (a whole dynamics
  step per thread in Isaac-Gym-style simulators; a whole rollout per thread in MPPI — see 08.01 and
  10.03). You cannot fuse what you cannot write. This project is the writing lesson.
- On CPU, **Eigen** with fixed-size types is the same unrolled-registers idea compiled for one core
  — the mental model transfers directly.
