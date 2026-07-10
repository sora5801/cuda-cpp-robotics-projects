# Data — 28.01 Real-time FEM soft-arm model + model-based control (GPU SOFA-style)

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** (where one genuinely teaches more) are fetched by `../scripts/download_data.ps1`
  / `.sh` — idempotent, with source URL, expected size, and checksum documented below. **Respect every
  license**; registration-gated or no-redistribution datasets (KITTI, nuScenes) are pointed at, never
  mirrored.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

This project's "data" is not recordings — it is the **scenario definition** of a fully synthetic
soft arm: the material/mesh model constants (cross-checked at startup against the compiled constants
in [`../src/kernels.cuh`](../src/kernels.cuh) — the file cannot silently disagree with the code) and
the runtime controller scenario (probe magnitude, PI tuning, setpoint sequence). There is no physical
arm and no datasheet material: the "elastomer" is a synthetic teaching material chosen so the CFL
timestep, first-mode frequency, and buckling budget land at legible values (each derivation lives in
`kernels.cuh`). No public dataset applies — the project's ground truth is **analytic** (Euler-Bernoulli
statics, cantilever mode frequencies), which is exactly what the demo's gates check against;
`../scripts/download_data.ps1` / `.sh` are therefore honest no-ops.

| Property | Value |
|----------|-------|
| Kind | Synthetic (scenario definition; constants, no RNG) |
| Generator / source | `python ../scripts/make_synthetic.py` (writes `sample/arm_scenario.csv`) |
| License | Synthetic — the repo MIT license applies |
| Size (committed) | `arm_scenario.csv`: 2,457 bytes |
| Checksum (SHA-256) | `502e1632e7ba2c9008a38a9aa73dc2847da2ca92f4efb79266bf521b3d87286a` |
| Regenerate with | `python ../scripts/make_synthetic.py` (deterministic; no seed needed — the file is constants; the script prints the SHA-256 to verify) |

## Fields / format

`sample/arm_scenario.csv` is a `key,value` file (UTF-8, LF line endings for a stable checksum;
`#`-prefixed lines are comments). Two row classes, split exactly as
[`../src/main.cu`](../src/main.cu)'s loader treats them:

### MODEL rows — must match `../src/kernels.cuh` (loader aborts on mismatch)

| Key | Value | Units | Meaning (mirrored `kernels.cuh` constant) |
|-----|-------|-------|--------------------------------------------|
| `NELX` | 120 | — | elements along the arm length, x (`kNelx`) |
| `NELY` | 12 | — | elements through the arm height, y (`kNely`) |
| `ELEM_SIZE_M` | 0.002 | m | square element side h (`kElemSize_m`) |
| `YOUNGS_E_PA` | 1000000 | Pa | Young's modulus, synthetic elastomer-class (`kYoungsE_Pa`) |
| `POISSON_NU` | 0.4 | — | Poisson's ratio; 0.40 not 0.49 — Q4 locking, THEORY.md (`kPoissonNu`) |
| `THICKNESS_M` | 0.02 | m | out-of-plane depth (`kThickness_m`) |
| `DENSITY_KGM3` | 1100 | kg/m³ | mass density, synthetic silicone-like (`kDensity_kgm3`) |
| `DT_S` | 3e-05 | s | integration timestep, CFL-derived (`kDt_s`) |
| `RAYLEIGH_ALPHA` | 3.8 | 1/s | mass-proportional damping, ζ₁ ≈ 0.149 (`kRayleighAlphaOn`) |
| `RAYLEIGH_BETA` | 2e-05 | s | stiffness-proportional damping, high-mode/flutter damper (`kRayleighBetaOn`) |
| `TENDON_BIAS_N` | 0.25 | N | co-contraction pretension per tendon; 2×bias ≈ 51% of the buckling load (`kTendonBiasN`) |

### SCENARIO rows — genuine runtime inputs (edit + rerun to experiment, no rebuild)

| Key | Value | Units | Meaning |
|-----|-------|-------|---------|
| `PROBE_DELTA_T_N` | 0.18 | N | tension differential used to identify the tip Jacobian |
| `CONTROL_SUBSTEPS` | 100 | steps | dynamics steps per control tick (3 ms → ~333 Hz) |
| `HOLD_STEPS` | 66000 | steps | dynamics steps held per setpoint (~2.0 s ≈ 4 first-mode periods) |
| `PI_MARGIN_ALPHA` | 0.3 | — | Kp = margin/\|J\|; resonant loop gain ≈ margin/(2ζ) ≈ 0.7 < 1 |
| `PI_INTEGRAL_TIME_S` | 0.15 | s | Ki = Kp/Ti; integrator crossover ≈ margin/Ti = 2 rad/s |
| `DELTA_T_CLAMP_FRAC` | 1.8 | — | \|ΔT\| ≤ frac × bias (bias − clamp/2 = 0.025 N > 0: both tendons stay taut) |
| `SETPOINT_SAFE_FRAC` | 0.65 | — | setpoint scale = J × frac × clamp (targets stay inside the identified range) |
| `SETPOINT_FRACS` | 0.6;-0.6;0.3;0 | — | step+hold sequence, fractions of the scale (up, down, half up, return to center) |

All geometry is 2-D in the arm's body plane: x along the arm (base at x = 0, cantilevered), y through
its height, right-handed with z out of plane; SI units throughout (CLAUDE.md §12). The frame and the
mesh/DOF layout are defined once in `../src/kernels.cuh` and honored by every file.

> **Changing the model:** MODEL rows exist so the committed sample fully describes what was simulated —
> but the CFL timestep and analytic-gate formulas are *derived from these values at compile time*
> (`kernels.cuh` documents each derivation). To change the model, edit `kernels.cuh`, re-derive
> `DT_S`/damping per its comments, mirror the values in `../scripts/make_synthetic.py`, regenerate,
> and update the checksum above. The loader's cross-check makes a half-done edit fail loudly instead
> of silently simulating something else.
