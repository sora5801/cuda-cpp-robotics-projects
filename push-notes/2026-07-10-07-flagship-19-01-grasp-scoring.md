# Push note — 2026-07-10-07: flagship 19.01 grasp scoring

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 19.01 — **parallel antipodal grasp-candidate scoring** — is done: 4,096 two-finger grasp
candidates per object generated and scored on the GPU (~2 ms/object, comfortably inside the
bin-picking perception budget), on three analytic objects whose good grasps are *geometric
theorems* — a box, a cylinder, and a sphere. That data design makes the gates real: every top-10
grasp must be a provably-valid antipodal pair (box widths land within a millimeter of the true
40/60 mm faces), the box's 100 mm pairing is correctly killed by the gripper-stroke gate (614
candidates), and **12/12 deliberately adversarial adjacent-face candidates are rejected by the
friction-cone gate** — the negative control that separates a scorer from a rubber stamp. Force
closure for two-contact antipodal grasps is derived from Coulomb cones in THEORY.md (Nguyen 1988
lineage), and the pipeline reuses 02.06's PCA-normals machinery with the inward/outward
disambiguation policy documented.

## What changed

- **[projects/19-manipulation-grasping/19.01-parallel-grasp-candidate-scoring/](../projects/19-manipulation-grasping/19.01-parallel-grasp-candidate-scoring/)** —
  complete: normals (k-NN PCA + Jacobi, cited from 02.06), stateless counter-hash candidate
  generation + antipodal ray search, friction-cone/width/clearance scoring, host ranking, CPU
  twin per stage, analytic + adversarial gates, grasp + visualization CSV artifacts, full
  README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 19.01 → `done` (**23/505**).

## New projects (didactic blurbs)

**19.01 — Antipodal grasp scoring** (★ beginner, domain 19, flagship). Contact mechanics from
first principles: Coulomb friction becomes a cone, two opposing cones become force closure, and
force closure becomes a per-candidate GPU test. The load-imbalance honesty of divergent ray
searches is the GPU lesson. The single most interesting thing to look at: the box's gate
composition — geometrically perfect antipodal pairs correctly rejected because the *gripper*
cannot open 100 mm; grasping is about the hand, not just the object.

## How to build & run

```powershell
projects\19-manipulation-grasping\19.01-parallel-grasp-candidate-scoring\demo\run_demo.ps1
# then plot demo\out\grasp_cloud.csv (recipe in demo\README.md)
```

## What to study here

`THEORY.md` §The math (the force-closure derivation — the heart) → `src/kernels.cu` (generation →
scoring) → the adversarial-candidate design in `main.cu`. First exercise: halve μ and watch the
cylinder's feasible set shrink from "everything diametral" toward "only near-perpendicular
approaches".

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 16 stable lines matched, exit 0.
- **GPU-vs-CPU gates:** normals worst 0.034° (tol 0.5°); candidate generation exact (0/4,096);
  scoring 6.0e-08 rel with 0 feasibility-flag mismatches.
- **Analytic gates:** top-10 grasps valid on all three objects (box 39.4–60.5 mm vs 40/60 true;
  cylinder 49.2–50.6 vs 50; sphere 59.4–60.9 vs 60; antipodal scores ≥ 0.9999); box z-axis
  pairing rejected by width (614 candidates); **adversarial negative control 12/12 rejected**.
- Timing (teaching artifact): ≈ 1–2 ms GPU per object across all three stages.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Two-finger antipodal only (ε-quality/wrench-space metrics documented as the full version —
  19.02), slab-approximated finger clearance, synthetic analytic objects by design (verification
  needs known-good grasps; real-cloud noise robustness is an exercise).

## Next push preview

20.01 GelSight tactile processing closes batch 1e; then batch 1f (21.04 speed-and-separation,
24.01 magnetostatic FEA, 25.01 battery electro-thermal, 26.01 topology optimization).
