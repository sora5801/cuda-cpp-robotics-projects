# Data — 18.01 Snake robots: serpenoid gait sweeps coupled to granular sim

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

This project's "data" is a **task definition**, not a recording: the snake's geometry, the ground's
anisotropic friction coefficients, and the 3-D gait-sweep grid the demo searches. There is no sensor
recording, no public dataset, and no randomness anywhere in this project's physics — every simulation
is a pure, deterministic function of its gait parameters (CLAUDE.md §8: "seed only if noise is used;
prefer deterministic no-noise" — this project needs no seed at all).

| Property | Value |
|----------|-------|
| Kind | Synthetic — a task/scenario definition, not a recording |
| Generator / source | `python scripts/make_synthetic.py` (no `--seed` flag exists: the generator is deterministic by construction) |
| License | Synthetic — the repo's MIT license (`../../../LICENSE`) applies |
| Size (committed) | `data/sample/snake_scenario.csv` — 889 bytes |
| Checksum | SHA-256 `4dcb2c43766a5f55b2b2e53af6fbf9a72fc0d08e18e9e76544992fad4e506ce9` |
| Regenerate with | `python scripts/make_synthetic.py` (writes the committed defaults; pass `--n-amp`/`--n-beta`/`--n-omega`/`--mu-t`/`--mu-n`/etc. to explore other scenarios) |

### Fields / format

`data/sample/snake_scenario.csv` is a strict `LABEL,value` CSV (one row per field, `#`-prefixed
comment lines ignored, every field REQUIRED — `src/main.cu`'s `load_scenario()` rejects the file if any
field is missing or unrecognized). All 17 fields, in the order the committed file writes them:

| Label | Meaning | Units / range |
|---|---|---|
| `LINK_LEN_M` | per-link rigid-rod length | m (committed: 0.10 — 12 links -> 1.20 m body) |
| `LINK_MASS_KG` | per-link mass | kg (committed: 0.15 — 12 links -> 1.80 kg total) |
| `GRAVITY` | gravitational acceleration, sets each link's normal load `m*g` | m/s^2 (committed: 9.81) |
| `MU_T` | TANGENTIAL (along-link) Coulomb friction coefficient — LOW | unitless (committed: 0.10) |
| `MU_N` | NORMAL (across-link) Coulomb friction coefficient — HIGH | unitless (committed: 0.70; `MU_N/MU_T = 7.0` is the anisotropy ratio THEORY.md derives as necessary for propulsion) |
| `N_AMP` | number of amplitude grid points | int, >= 2 (committed: 32) |
| `AMP_MIN_R`, `AMP_MAX_R` | amplitude `A` sweep range | rad, inclusive both ends (committed: 0.05 to 1.05, i.e. ~2.9-60.2 deg) |
| `N_BETA` | number of inter-joint phase-offset grid points | int, >= 2 (committed: 32) |
| `BETA_MIN_R`, `BETA_MAX_R` | phase offset `beta` sweep range | rad, inclusive both ends (committed: 0.10 to 3.00) |
| `N_OMEGA` | number of temporal-frequency grid points | int, >= 1 (committed: 8) |
| `OMEGA_MIN_R`, `OMEGA_MAX_R` | temporal frequency `omega` sweep range | rad/s, inclusive both ends (committed: 1.0 to 6.0, i.e. periods 6.28 s down to 1.05 s) |
| `T_SIM_S` | simulated duration per gait | s (committed: 8.0) |
| `DT_S` | integration step | s (committed: 0.001, i.e. 1 ms -> 8,000 steps/gait) |
| `TURN_GAMMA_R` | turning-bias test offset for `GATE_TURNING_BIAS` | rad (committed: 0.15, ~8.6 deg) |

`N_AMP * N_BETA * N_OMEGA` (derived, not stored) gives the sweep width `G` — 8,192 for the committed
scenario. All angles are in the world top-down frame (SYSTEM_DESIGN §3.2's `x`-forward/`y`-left/`z`-up
convention specialized to flat ground): `yaw=0` points along `+x`, CCW positive.
