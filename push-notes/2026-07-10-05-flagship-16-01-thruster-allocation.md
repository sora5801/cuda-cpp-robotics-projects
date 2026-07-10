# Push note — 2026-07-10-05: flagship 16.01 thruster allocation

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 16.01 — **thruster allocation for overactuated ROVs (batched QP)** — opens batch 1e. An
8-thruster, 6-DOF ROV's control allocation is solved as a batch of box-constrained QPs, one
projected-gradient solve per GPU thread, with the step size derived from the quadratic's Lipschitz
constant. The motivating example is the lesson in one line: when thrusters saturate, the naive
pseudoinverse-then-clip answer keeps only **7.6%** of the commanded surge and points **14.2°**
off-axis, while the QP keeps **70.5%** at **4.1°** — clip-after-pseudoinverse doesn't just lose
authority, it changes the wrench *direction*. Verification is four independent mathematical
gates, and the demo's second stage is the analysis real ROV operators run: an 8-way
thruster-failure Monte Carlo quantifying achievable-wrench degradation per lost thruster.

## What changed

- **[projects/16-locomotion-marine/16.01-thruster-allocation-for-overactuated-rovs/](../projects/16-locomotion-marine/16.01-thruster-allocation-for-overactuated-rovs/)** —
  complete: PGD QP kernel (thread-per-problem, constant-memory H, 500 fixed iterations —
  divergence-free by design), CPU twin, pseudoinverse/KKT/monotonicity gates, failure Monte
  Carlo, allocation + failure-analysis artifacts, full README / THEORY / PRACTICE. Body frame
  uses the Fossen/SNAME marine convention (x-fwd, y-stbd, z-down) — a documented §12 domain-
  standard deviation.
- **[docs/STATUS.md](../docs/STATUS.md)** — 16.01 → `done` (**21/505**).

## New projects (didactic blurbs)

**16.01 — Thruster allocation QP** (intermediate, domain 16, flagship). Why overactuation exists
(failure tolerance + station-keeping), how geometry becomes an allocation matrix (rows =
[direction; moment arm × direction]), why saturation breaks the pseudoinverse, and how box-KKT
conditions make a solution *checkable*. The damping-bias story is derived rather than hidden: the
regularizer trades a measured 13–30% authority bias along the vehicle's weakest eigen-directions
for conditioning, quantified per-DOF in THEORY. The single most interesting thing to look at:
`demo/out/failure_analysis.csv` — horizontal-thruster loss costs 3.7–11 points of tracking,
vertical loss barely 1–2, exactly as the geometry predicts.

## How to build & run

```powershell
projects\16-locomotion-marine\16.01-thruster-allocation-for-overactuated-rovs\demo\run_demo.ps1
# then read demo\out\allocation.csv and failure_analysis.csv
```

## What to study here

`THEORY.md` §The problem (the pseudoinverse-clip failure case — the whole motivation) → §The math
(box-KKT + the Lipschitz step derivation) → `src/kernels.cu`. First exercise: re-weight W to
prioritize heave over surge and watch the allocation redistribute.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 10 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** worst deviation 2.7e-05 N over 500×8 forces.
- **Optimality gates:** unsaturated solutions match the closed-form damped pseudoinverse at
  6.1e-05 N (472 wrenches); saturated solutions satisfy the box-KKT fixed point at 5.7e-06 N
  (28 wrenches); objective monotone over all 500 iterations, 0 violations (slack justified by the
  proximal-descent lemma against measured FP32 rounding).
- **Failure Monte Carlo:** per-thruster degradation quantified vs the nominal baseline
  (H-group 3.7–11.0 pp mean, V-group 1.1–2.2 pp — matching the wrench batch's composition).
- Timing (teaching artifact): 500 QPs × 500 iterations ≈ 0.17 ms GPU vs ≈ 5.8 ms CPU.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Teaching solver (PGD; interior-point/active-set documented as production), force-level
  allocation (the T = k|n|n prop map and wake interactions documented as the next layer down),
  uniform weighting in the committed scenario.

## Next push preview

18.01 snake-robot serpenoid gait sweeps, then 19.01 antipodal grasp scoring and 20.01 GelSight
tactile processing close batch 1e.
