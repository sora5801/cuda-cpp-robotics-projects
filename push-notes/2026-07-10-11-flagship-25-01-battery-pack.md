# Push note — 2026-07-10-11: flagship 25.01 battery pack

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 25.01 — **Li-ion electrochemistry + 3-D pack thermal + cooling-design sweeps** — is done:
a single-particle-model cell (spherical finite-volume diffusion, closed-form Butler–Volmer,
synthetic-labeled OCV curves) electro-thermally coupled through Arrhenius factors into a 24-cell
anisotropic 3-D pack, driven by an AMR duty-cycle mission, swept over 12 cooling designs in
batched kernel launches. Three analytic gates hold — including the elegant one: the constant-flux
sphere's quasi-steady c_surf − c_avg matches the closed form jR/5D, with the FV discretization
bias *measured by a shell-count convergence study* (10→23%, 20→12%, 40→6%) and the tolerance set
from that measurement. The sweep produced a genuine, unengineered engineering finding:
bottom-plate cooling is conduction-limited by the cells' weak axial conductivity and barely
responds to h, while side-plate cooling scales strongly with h but pays in cell-to-cell spread —
a real peak-temperature-versus-balance trade quantified in the design table. SPMe/P2D ship as the
documented ladder per §2/§13 (README §13 states the scoping).

## What changed

- **[projects/25-power-energy/25.01-li-ion-electrochemical-solver-on-gpu-3d-pack/](../projects/25-power-energy/25.01-li-ion-electrochemical-solver-on-gpu-3d-pack/)** —
  complete: spherical-FV electrochem kernel + batched FTCS thermal kernel (design axis batched
  like 24.01), Arrhenius coupling, CPU twins (exact agreement), three analytic gates + per-design
  energy conservation, temps/design-table/slice artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 25.01 → `done` (**27/505**).

## New projects (didactic blurbs)

**25.01 — Battery electro-thermal** (★ beginner, domain 25, flagship). Why lithium intercalation
is a diffusion problem, where the SPM sits on the model ladder (and where it breaks — high C-rates
push you to SPMe/P2D, documented with their math), why pack *spread* matters more than pack peak
(aging divergence), and how anisotropic cell conductivity decides your cooling architecture before
you pick a single part. The single most interesting thing to look at: `demo/out/design_sweep.csv`
— six bottom-plate designs landing within 0.01 K of each other (conduction-limited) next to
side-plate designs fanning out with h.

## How to build & run

```powershell
projects\25-power-energy\25.01-li-ion-electrochemical-solver-on-gpu-3d-pack\demo\run_demo.ps1
# then read demo\out\design_sweep.csv and plot demo\out\pack_temps.csv
```

## What to study here

`THEORY.md` §The problem (the intercalation-diffusion story and the model ladder) → the jR/5D
gate derivation and its convergence study → `src/kernels.cu`. First exercise: double the mission's
C-rate and watch the SPM's isothermal assumption start to creak — then read the SPMe section to
see what fixes it.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 11 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** exact agreement (0.0 deviation) on both solvers over 200 verification steps.
- **Analytic gates:** diffusion jR/5D at 12.2% vs the measured 20-shell bias (~12%, tol 15% —
  convergence-study calibrated); coulomb counting 0.35% (tol 1%); thermal P = hAΔT identity 3.5%
  (tol 5%); per-design energy balance worst 0.70% (tol 2%).
- **Design sweep:** best peak-T design h=500 side-plate (307.83 K, but 4.20 K spread); most
  balanced h=10 bottom-plate (0.047 K spread) — the trade documented, not averaged away.
- The builder disclosed an honest parameter-tuning pass (R_ohm 0.04→0.10 Ω to make the cooling
  comparison didactically visible; rerun and re-measured, never back-filled).
- Timing (teaching artifact): 12 designs × 12,000 steps ≈ 0.91 s GPU kernel time.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- SPM tier only (SPMe/P2D documented with math as the ladder), shared pack current, synthetic
  teaching parameters labeled as such (never real-cell data), 24-cell scale (the GPU's real
  payoff at 1000-cell packs and Monte Carlo aging is argued honestly in THEORY).

## Next push preview

26.01 topology optimization (SIMP) closes batch 1f; then batch 1g (27.04 composite layup, 28.01
FEM soft arm, 29.05 ultrasound beamforming, 30.01 agriculture bundle).
