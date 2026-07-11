# Data — 36.03 Lattice-robot kinematics batches

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

There is no public dataset for synthetic lattice-robot configurations — there is nothing to download,
and `../scripts/download_data.ps1`/`.sh` are honest no-ops that say so. The committed sample is not a
dataset of configurations at all — it is a tiny **generator parameter file**, the same "scenario, not
recordings" choice project 08.01 makes for its cart-pole start state:

| Property | Value |
|----------|-------|
| Kind | Synthetic — a generator PARAMETER file, not recorded/pre-built configurations |
| Generator | `../scripts/make_synthetic.py` (writes the parameter file); the actual K=4096-configuration batch is regenerated at RUNTIME by `../src/main.cu`'s seeded-accretion `generate_config()`, deterministically from the `SEED` below, every time the demo runs |
| License | Synthetic — the repo's MIT license applies; no external data or license involved |
| Size (committed) | `lattice_scenario.csv`: 9 lines, ~450 bytes |
| Checksum | SHA-256 of `sample/lattice_scenario.csv` (seed 42, the committed default): `4f39f8daa85f63cdeaaea93036b1dfea9682f4ac87b220730a7b517694531e97` |
| Regenerate with | `python make_synthetic.py --k 4096 --seed 42 --corrupt-frac 0.10 --vignette-steps 600` (the defaults — running with no arguments reproduces the exact committed file byte-for-byte) |

### Fields / format

`sample/lattice_scenario.csv` — four labelled rows (plus `#`-prefixed provenance comments the loader
skips), each `LABEL,value`:

| Field | Type | Units / range | Meaning |
|---|---|---|---|
| `K` | int | configurations, > 0 | Batch size — how many lattice configurations `main.cu` generates and analyzes per run. Committed value: 4096. |
| `SEED` | uint32 | any 32-bit value | Base seed for the repo-standard xorshift32 generator; `main.cu` derives one per-configuration seed from this value (`seed + 1000003 * (k+1)`, the same per-item seed-mixing 08.01 uses for its per-tick noise), so the ENTIRE batch — every module position, every corruption choice — is a pure deterministic function of this one number. Committed value: 42. |
| `CORRUPT_FRAC` | float | `[0, 1)` | Fraction of `K` deliberately corrupted as negative controls (half duplicate-position defects, half severed-connectivity defects — see `../src/main.cu`'s `corrupt_duplicate`/`corrupt_disconnect` and `../THEORY.md` "How we verify correctness"). Committed value: 0.10 (→ 410 corrupted, 205+205, 3686 clean at K=4096). |
| `VIGNETTE_MAX_STEPS` | int | steps, > 0 | Step budget CAP for the greedy reconfiguration vignette (`../src/main.cu` §8) — a ceiling, not a target the greedy is guaranteed to need; the committed demo run actually converges to a local optimum in 127 steps, well under this cap. Committed value: 600. |

No lattice coordinates, module positions, or move sequences are stored anywhere under `data/` — they
are all **generated, not recorded**, every time `../src/main.cu` runs (CLAUDE.md §8's synthetic-first
rule applied as literally as this repository gets: the "data" is a recipe, not a result).

Everything the demo writes AS A RESULT of running (the batch statistics, the vignette's frame-by-frame
positions, the rendered configuration image) is a git-ignored ARTIFACT under `../demo/out/`, never
committed here — see `../demo/README.md`.
