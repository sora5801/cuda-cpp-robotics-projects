# Data — 34.03 Ergodic control: spectral multiscale coverage (FFT-based — very GPU-friendly)

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

There is no real-world "correct" information density to download for this project — the target
`phi(x)` is a constructed teaching scenario (two Gaussian hotspots + a uniform washout), and the
committed sample is a **scenario file**, not a recording, in the same spirit as 08.01's
`cartpole_scenario.csv`.

| Property | Value |
|----------|-------|
| Kind | Synthetic (the only option — CLAUDE.md §8 default, and no public dataset applies here) |
| Generator / source | `../scripts/make_synthetic.py` (no RNG — every value is a fixed constant) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | ~0.6 KiB (`ergodic_scenario.csv`) |
| Checksum | `ergodic_scenario.csv` SHA-256: `8e5742d862127aae966be88e6565b56f34ecbf7dbfd8579518ddf055b6484d7b` |
| Regenerate with | `python ../scripts/make_synthetic.py` (deterministic; `--x0 X1 X2 --steps N` to override) |

### Fields / format

`data/sample/ergodic_scenario.csv` — `#`-prefixed lines are provenance/documentation comments
(including a full restatement of the target-density and controller constants, for a human reading
the file — the values ACTUALLY compiled into the program are `src/kernels.cuh`'s `constexpr`
declarations, the single source of truth, CLAUDE.md §12). Two functional rows `src/main.cu`'s
`load_scenario()` parses:

| Row | Fields | Meaning |
|-----|--------|---------|
| `X0,x1,x2` | `x1`, `x2` — floats in `[0,1]` | Agent start position, domain-normalized workspace coordinates (unitless; see `PRACTICE.md` §2 for scaling to real meters). |
| `STEPS,n` | `n` — int | Number of `dt=0.01 s` control steps in the closed-loop run (`n=6000` -> 60 s). Must exceed `kVerifyWindow+2=52` (see `src/kernels.cuh`). |

The target-density shape (`kMu1*/kSigma1/kW1`, `kMu2*/kSigma2/kW2`, `kWBg`), the mode count `K`,
and the agent's speed budget `kVmax` are documented in the CSV's comment header for provenance but
are **not** parsed at runtime — changing them means editing `src/kernels.cuh` and rebuilding (the
repo's "one place, never two" rule for anything that affects the algorithm's correctness proofs,
CLAUDE.md §12).
