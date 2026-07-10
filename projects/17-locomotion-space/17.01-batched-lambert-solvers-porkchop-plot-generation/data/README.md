# Data — 17.01 Batched Lambert solvers + porkchop plot generation

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

A Lambert solver's "dataset" is a **scenario**, not recordings: two orbit radii, a shared epoch
window, an accepted time-of-flight band, and a grid resolution. Both bodies are SYNTHETIC coplanar
circular heliocentric orbits — no ephemeris, no SPICE kernel, nothing to download (`scripts/download_data.ps1`
is an honest no-op; see its DECISION note for why real ephemerides are a documented, out-of-scope next
step, not a missing feature).

| Property | Value |
|----------|-------|
| Kind | **Synthetic** scenario (six constants — no RNG involved) |
| File | `sample/lambert_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: r1=1.000 AU, r2=1.524 AU, window=28.0 TU, TOF band=(0.5,14.0) TU, grid=512×512) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 771 bytes |
| Checksum (SHA-256) | `9a5965d6d8e4d743ec26bb2c73e815416e8e9c1dee85736e9550af63062141a0` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness) |

### Fields / format

Plain-text CSV; `#` lines are comments. Six required rows, `LABEL,value` (loader: `load_scenario()`
in [`../src/main.cu`](../src/main.cu); layout authority: [`../src/kernels.cuh`](../src/kernels.cuh)):

| Field | Units | Meaning |
|-------|-------|---------|
| `R1_AU` | AU (= 1 canonical length unit, LU) | body 1 ("Earth-like") circular heliocentric orbit radius |
| `R2_AU` | AU | body 2 ("Mars-like") circular heliocentric orbit radius — 1.524 AU is Mars' real semi-major axis, used here only as a *size*, not as a real ephemeris (both orbits are idealized circles, not Mars' actual ~9.3%-eccentric ellipse) |
| `WINDOW_TU` | TU (canonical time unit, 1 TU = 58.132441 mean solar days — derived in `kernels.cuh`) | BOTH the departure-epoch and arrival-epoch axes span `[0, WINDOW_TU)` |
| `MIN_TOF_TU` | TU | time-of-flight floor: transfers faster than this are masked (structural exclusion, not a solver failure) |
| `MAX_TOF_TU` | TU | time-of-flight ceiling: transfers slower than this are masked |
| `GRID_N` | cells (int) | grid resolution per axis — `GRID_N × GRID_N` total cells |

`mu` (the Sun's gravitational parameter) is **not** a row in this file: canonical units *define*
`mu = 1` — it is an axiom of the unit system, not scenario data (`kernels.cuh`'s file header derives
the full SI conversion table from that axiom). Both bodies are placed at orbital phase angle 0 (the
+x heliocentric axis) at canonical time `t = 0` — a modeling convention documented in `kernels.cuh` and
worked out precisely in `THEORY.md` §the-math (including exactly where in the grid the Hohmann-optimal
alignment recurs, given that convention).

The loader is strict: unknown labels, short rows, a missing required row, or values failing the
sanity checks (radii/window positive, `0 <= MIN_TOF_TU < MAX_TOF_TU`, `GRID_N >= 2`) abort the demo.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
