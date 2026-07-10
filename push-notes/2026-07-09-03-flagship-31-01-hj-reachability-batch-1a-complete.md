# Push note — 2026-07-09-03: flagship 31.01 hj reachability batch 1a complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 31.01 — **Hamilton–Jacobi reachability** — is done, and with it **worker batch 1a is
fully merged** (04.01, 05.01, 22.01, 31.01; 8/505 overall, 8 of 36 flagships). The project
computes the backward reachable tube of a double integrator by solving the HJ PDE with a local
Lax–Friedrichs level-set sweep on the GPU, and verifies it two ways: against a CPU twin
(1.7e-05 worst deviation) and against **pure mathematics** — the closed-form bang-bang
minimum-time solution, which the grid classification matches everywhere outside a measured 3-cell
boundary band. The finisher's debugging journey is the didactic highlight of the batch: what
looked like a solver bug was proven — by falsifying four hypotheses experimentally — to be
textbook first-order Lax–Friedrichs dissipation compounding over long horizons; the fix was a
measured re-scope of the horizon (1.5 s → 0.4 s) with the mechanism quantified in THEORY.md, not
a widened check. Per CLAUDE.md §8, the project states plainly that it computes safety-style
metrics didactically and is **not** a certified implementation.

## What changed

- **[projects/31-safety-verification/31.01-hamilton-jacobi-reachability/](../projects/31-safety-verification/31.01-hamilton-jacobi-reachability/)** —
  complete: level-set sweep kernel (local Lax–Friedrichs numerical Hamiltonian, upwinding, CFL,
  ping-pong buffers), CPU twin, analytic bang-bang oracle, value-function PGM + BRS boundary CSV
  artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 31.01 → `done` (**8/505**); batch-1a rows all closed.

## New projects (didactic blurbs)

**31.01 — HJ reachability** (★ beginner, domain 31, flagship). Teaches why reachability *is* the
safety question (stopping distance, inevitable-collision states), how dynamic programming becomes
a PDE, and how a stencil sweep solves it — the same grid/ping-pong pattern as 07.09, now carrying
safety semantics. The verification design is the lesson: a CPU twin catches porting bugs; only the
*analytic* solution catches numerical-scheme artifacts — and it did, in the recorded debugging
story (THEORY.md §Numerical considerations quantifies the dissipation it caught). The single most
interesting thing to look at: `demo/out/value_function.pgm` — the reachable tube's characteristic
bang-bang "S" shape between the switching curves.

## How to build & run

```powershell
projects\31-safety-verification\31.01-hamilton-jacobi-reachability\demo\run_demo.ps1
# then open demo\out\value_function.pgm
```

## What to study here

The batch as a set: 04.01 (particles), 05.01 (volumes), 22.01 (agents+atomics), 31.01 (level
sets) — four different GPU patterns, all gated by two-tier verification. Within 31.01: README →
THEORY §The problem (reachability as the safety question) and §Numerical considerations (the
dissipation story) → `src/kernels.cu`. First exercise: shrink umax and watch the tube collapse.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the finisher's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 7 stable lines matched, exit 0.
- **CPU-twin gate:** worst |V_gpu − V_cpu| = 1.669e-05 over 65,536 cells (tol 1e-3).
- **Analytic gate:** 0 classification disagreements outside the 3-cell boundary band vs the
  closed-form bang-bang minimum-time solution (band width measured, not assumed: 2 cells fails,
  3 passes — recorded in the code comments); sup |V_num − V_exact| = 0.38 s of first-order LF
  dissipation, quantified and taught rather than hidden.
- Timing (teaching artifact): 109 sweeps on 256² ≈ 1.0 ms GPU vs ≈ 50 ms CPU.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- 2-D double integrator at a 0.4 s horizon (first-order scheme; the honest cost of Lax–Friedrichs
  — higher-order WENO/TVD-RK schemes and longer horizons are documented as the production path,
  alongside hj_reachability/OptimizedDP). The curse of dimensionality beyond ~4-D is stated, with
  the tensor-decomposition frontier pointed at (catalog 34.06).
- Batch 1a process record: an 8-worker and a 4-worker parallel batch were both killed by account
  session limits mid-build; the surviving pattern is sequential finisher agents completing
  checkpointed partial work, independently re-gated by the lead. Batch 1b (06.05, 15.01, 17.01,
  23.01) proceeds one builder at a time.

## Next push preview

Batch 1b begins: 06.05 STOMP (parallel noisy-rollout trajectory optimization — MPPI's planning
cousin), then 15.01 minimum-snap batches, 17.01 Lambert/porkchop, 23.01 GPU costmaps + DWA.
