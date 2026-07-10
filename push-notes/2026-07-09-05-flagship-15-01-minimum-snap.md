# Push note — 2026-07-09-05: flagship 15.01 minimum snap

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 15.01 — **batched minimum-snap trajectory optimization** — is done: 10,000 waypoint sets
turned into smooth quadrotor-style trajectories in ~11 ms of GPU time, one thread solving one
set's 64-unknown constrained polynomial system end to end. Verification is two-tier and the second
tier is the star: beyond the GPU-vs-CPU gate, every one of the 10,000 solutions is re-checked in
double precision **against the constraint definitions themselves** (waypoint interpolation,
endpoint derivatives, interior continuity) by code that deliberately shares nothing with the
assembly path — a mathematical audit, not an echo. THEORY.md is honest about the formulation: this
is the fully-determined closed-form spline (the constraint count exactly consumes the degrees of
freedom), explicitly distinguished from Mellinger–Kumar's free-derivative QP, which is documented
as the production path.

## What changed

- **[projects/15-locomotion-aerial/15.01-minimum-snap-trajectory-optimization-batched/](../projects/15-locomotion-aerial/15.01-minimum-snap-trajectory-optimization-batched/)** —
  complete: constraint-assembly + in-thread Gaussian-elimination kernel (32×32 per axis, partial
  pivoting, per-thread local memory — the honest contrast to 33.01's register regime), CPU twin,
  independent double-precision constraint audit, dense-sampled trajectory CSV + slalom-path PGM
  artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 15.01 → `done` (**10/505**).

## New projects (didactic blurbs)

**15.01 — Minimum-snap batches** (★ beginner, domain 15, flagship). Teaches why quadrotors
minimize *snap* of all things: differential flatness — position's fourth derivative maps directly
to rotor commands, so smooth snap means feasible motor inputs (THEORY derives it physics-first).
The GPU lesson is the memory-tier ladder: at 32×32, per-thread systems no longer fit registers and
live in local memory — measured, documented, and still a 5× win at batch scale because the
problems are independent. The single most interesting thing to look at: the constraint-audit stage
in `src/main.cu` — verification against definitions rather than implementations.

## How to build & run

```powershell
projects\15-locomotion-aerial\15.01-minimum-snap-trajectory-optimization-batched\demo\run_demo.ps1
# then plot demo\out\trajectory.csv or open demo\out\slalom_path.pgm
```

## What to study here

`THEORY.md` §The problem (differential flatness — the physics that makes this objective exist) →
§Numerical considerations (why high-order polynomial bases are ill-conditioned and normalized time
fixes it) → `src/kernels.cu`. First exercise: unequal segment times (the time-allocation problem
the README points at).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 7 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** worst relative coefficient deviation 6.4e-04 over 10,000 sets × 2 axes
  (tol 5e-3 — FP32 Gaussian elimination on 32×32 systems; headroom measured, not assumed).
- **Constraint audit (all 10,000 sets, double precision, independent code path):** worst
  interpolation error 5.1e-05 m (tol 1e-3), worst endpoint derivative 2.3e-04 (tol 1e-2), worst
  continuity jump 5.0e-04 relative (tol 1e-2); snap costs finite and positive throughout.
- Timing (teaching artifact): K=10,000 in ≈ 11.4 ms GPU vs ≈ 56 ms CPU.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- 2-D axes, fixed equal segment times, fully-determined spline (no free-derivative optimization) —
  each documented with its production counterpart (mav_trajectory_generation, PX4) in README/THEORY.

## Next push preview

Batch 1b continues: 17.01 batched Lambert solvers + porkchop plots (with a Hohmann-transfer
analytic check), then 23.01 GPU costmaps + DWA to close the batch.
