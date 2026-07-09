# Push note — 2026-07-08-01: flagship 33.01 small-matrix linalg

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Phase 1 begins: the repository's **first finished project**. Flagship 33.01 — batched small-matrix
linear algebra (3×3/4×4/6×6) — is built to the full Definition of Done: real kernels replacing the
scaffold placeholder, a line-by-line CPU oracle, a committed synthetic sample, a passing offline
demo, and complete README/THEORY/PRACTICE documentation. It matters because it teaches the single
most reused GPU idiom in this repo — **one thread owns one whole matrix problem, solved in
registers** — the pattern the upcoming FK (09.01), jump-flooding (07.09), and MPPI (08.01)
flagships build on. It also battle-tested the template end to end: the standards survived contact
with a real implementation.

## What changed

- **[projects/33-foundational-libraries/33.01-batched-small-matrix-linear-algebra/](../projects/33-foundational-libraries/33.01-batched-small-matrix-linear-algebra/)** —
  complete: [`src/kernels.cu`](../projects/33-foundational-libraries/33.01-batched-small-matrix-linear-algebra/src/kernels.cu)
  (batched matmul + batched SPD Cholesky solve, templates on N, runtime dispatch),
  [`src/reference_cpu.cpp`](../projects/33-foundational-libraries/33.01-batched-small-matrix-linear-algebra/src/reference_cpu.cpp)
  (oracle twin), [`src/main.cu`](../projects/33-foundational-libraries/33.01-batched-small-matrix-linear-algebra/src/main.cu)
  (two-stage verification + output contract), synthetic sample (~59 KiB CSV, seed 42) +
  [`scripts/make_synthetic.py`](../projects/33-foundational-libraries/33.01-batched-small-matrix-linear-algebra/scripts/make_synthetic.py),
  full README / THEORY / PRACTICE, data & demo READMEs, resolved scaffold markers throughout.
- **[docs/STATUS.md](../docs/STATUS.md)** — 33.01 → `done` (1/505).
- **[docs/PROJECT_TEMPLATE/build/TEMPLATE.vcxproj](../docs/PROJECT_TEMPLATE/build/TEMPLATE.vcxproj)** —
  template fix found during battle-testing: Debug config now sets only `-G` (device debug), because
  nvcc warns that `-G` overrides `-lineinfo`; Release keeps `-lineinfo`. This satisfies the §9
  zero-warnings gate while keeping Nsight fully usable. **Note:** CLAUDE.md §5's wording ("Debug
  enables `-G` and `-lineinfo`") is internally conflicting with §9 zero-warnings; per §13 this is
  surfaced here rather than silently patched — the template now implements the sensible reading.

## New projects (didactic blurbs)

**33.01 — Batched small-matrix linear algebra** (★ beginner, domain 33, flagship). Teaches the
thread-per-problem batch pattern: robotics needs 10⁵ tiny solves per cycle, not one big one, so each
GPU thread factorizes and solves its own 6×6 SPD system entirely in registers (compile-time sizes +
full unrolling make register placement possible). Sits underneath estimation/planning/control the
way BLAS sits under scientific code. The single most interesting thing to look at:
`batched_cholesky_solve_kernel` in `src/kernels.cu` — a complete factorize→forward→back-substitute
pipeline in one thread's registers, with the honest coalescing discussion right above it.

## How to build & run

```powershell
# from the project folder
projects\33-foundational-libraries\33.01-batched-small-matrix-linear-algebra\demo\run_demo.ps1
# or: open build\batched-small-matrix-linear-algebra.sln in VS 2026, Release|x64, build, run the exe
```

## What to study here

Read in this order: project `README.md` (Overview → System context) → `THEORY.md` §The problem
(where 3×3/4×4/6×6 matrices physically come from in a robot) → `src/main.cu` (the output contract
and verification design) → `src/reference_cpu.cpp` → `src/kernels.cu` (the heart). Exercises to
try first: provoke the non-SPD NaN path (README Exercise 1), then the SoA layout conversion with
Nsight measurement (Exercise 3).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-08):

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings** (after the template
  `-G`/`-lineinfo` fix above).
- `demo/run_demo.ps1` passes end to end: all 6 stable lines matched `demo/expected_output.txt`,
  exit 0. Worst GPU-vs-CPU deviations: **2.4e-07 abs** (matmul, tol 1e-5), **2.2e-08 rel** (solve,
  tol 1e-4).
- Observed kernel-only timings (single-shot teaching artifacts, not benchmarks; GPU-clock variance
  across runs was large): matmul n=6 ×200k ≈ 1.0–2.3 ms vs CPU ≈ 29 ms; Cholesky solve n=6 ×100k
  ≈ 0.20 ms vs CPU ≈ 13 ms.
- `tools/verify_project.py`: **all structural gates PASS** (layout, 13 README sections, THEORY 7,
  PRACTICE 4, no scaffold markers, comment density, expected output).

## Known limitations / TODOs

- FP32 only; sizes fixed to {3,4,6}; matrix-contiguous layout is deliberately coalescing-suboptimal
  (taught, with the SoA fix as an exercise); single-stream, no transfer overlap — all documented in
  the project's README §Limitations.
- **Session note:** parallel worker dispatch hit the Claude session limit mid-batch; 33.01's docs
  were written by the lead directly. The remaining three foundation flagships (09.01, 07.09, 08.01)
  are unstarted and follow next.

## Next push preview

Foundation flagships continue in sequence: 09.01 batched forward kinematics + Jacobians, then 07.09
jump-flooding SDF/Voronoi, then 08.01 MPPI — after which worker batches take over for the remaining
32 flagships (CLAUDE.md §11).
