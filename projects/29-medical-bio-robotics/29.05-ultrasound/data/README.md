# Data — 29.05 Ultrasound: GPU beamforming, elastography, image-based servoing

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

**EDUCATIONAL / SYNTHETIC ONLY.** Every file here is numerically generated. Nothing under `data/`
is, or is derived from, patient data, and nothing in this project makes a diagnostic or
therapeutic claim (CLAUDE.md §1, §8, and the medical-robotics rule in CLAUDE.md §8).

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

| Property | Value |
|----------|-------|
| Kind | Synthetic (default per CLAUDE.md §8) — a numerical point-scatterer phantom, never patient data |
| Generator / source | `../scripts/make_synthetic.py --seed 42` (deterministic; the default invocation) |
| License | Synthetic — repository MIT license applies |
| Size (committed) | `array_params.csv` ~0.4 KiB; `phantom.csv` ~750 KiB (19,409 scatterer rows) |
| Checksum (SHA-256) | `array_params.csv`: `5e4bec6ef96e808e0cfcf6c46f8c1c8c6d70fd33bcf21e433d75d556022e083c` |
| | `phantom.csv`: `c0b0294f9f084d770e4ad71ae955e19f502d4e6237da01665f9caf8c5a398270` |
| Regenerate with | `python scripts/make_synthetic.py --seed 42` (from the project root) |

The channel data itself — the simulated per-element RF traces a real transducer array would
record from this phantom — is **not** a committed file: `src/main.cu`'s `simulate_channel_data()`
generates it in-demo, deterministically, from the phantom + the array/pulse physics in
`src/kernels.cuh` (README "Data" explains why: it is the project's synthetic *sensor*, not
committed ground truth, and regenerating it is instant). No public ultrasound RF dataset applies
here — there is no license that permits redistributing real patient RF data, so
`scripts/download_data.ps1`/`.sh` are honest no-ops (they print that this project is
synthetic-only and exit 0, the same uniform-callable-by-CI shape every project's download script
follows).

### Fields / format

**`data/sample/array_params.csv`** — one data row (after `#`-prefixed comment/provenance lines),
cross-checked field-by-field against `src/kernels.cuh`'s constexpr constants at load time
(`main.cu`'s `check_array_params()` — the same pattern project 03.01 uses for `radar_params.csv`):
a mismatch is a loud `SCENARIO: MISMATCH` failure, not a silently wrong demo.

| Column | Type | Units / meaning |
|---|---|---|
| `num_elements` | int | Array element count (64) |
| `pitch_m` | float | Element-to-element pitch (m) |
| `fc_hz` | float | Transducer/pulse center frequency (Hz) |
| `fs_hz` | float | Channel-data sampling rate (Hz) |
| `c_mps` | float | Assumed speed of sound (m/s) — the "tissue convention" 1540 m/s |
| `fnumber` | float | Receive f-number (dimensionless) |
| `incl_x_m`, `incl_z_m` | float | Inclusion disk center, lateral/axial (m), array/image frame |
| `incl_radius_m` | float | Inclusion disk radius (m) |

**`data/sample/phantom.csv`** — one row per point scatterer, columns `kind,x_m,z_m,amp_rel`:

| Column | Type | Units / meaning |
|---|---|---|
| `kind` | string | `wire` \| `inclusion` \| `speckle` — see below |
| `x_m` | float | Scatterer lateral position (m), array/image frame: 0 = array center, +x toward higher element index |
| `z_m` | float | Scatterer depth (m), array/image frame: 0 = the array face (imaging starts at `kImageZMinM` = 10 mm) |
| `amp_rel` | float | Relative reflectivity (unitless, teaching scale — **not** a calibrated acoustic backscatter coefficient) |

**Phantom contents** (19,409 rows total; THEORY.md "The problem" explains the physics each part
teaches):

- **9 `wire` rows** — a "+" cross pattern: a horizontal row of 5 wires at z = 20 mm (x = −8, −4, 0,
  4, 8 mm) and a vertical column of 5 wires at x = 0 mm (z = 12, 16, 20, 24, 28 mm — sharing the
  (0, 20 mm) center point with the horizontal row), `amp_rel` = 15.0 — deliberately far above
  background so each wire's mainlobe unambiguously dominates local speckle interference (a real
  QA-phantom wire target is a highly reflective monofilament for the same reason — PRACTICE.md
  §1–2). The center wire (0, 20 mm) is the "isolated" wire `main.cu`'s RESOLUTION gate measures a
  point-spread function from (nearest neighbor 4 mm away, far beyond the ~0.3–0.5 mm resolution).
- **1,400 `inclusion` rows** — uniform-random points inside a 2.5 mm-radius disk centered at
  (−6 mm, 15 mm), `amp_rel` = 1.4 (1.4x background) at roughly double the background scatterer
  density — a high-scattering region `main.cu`'s CONTRAST gate measures against background.
- **18,000 `speckle` rows** — uniform-random points over the imaged field of view (inset from
  `src/kernels.cuh`'s `kImageXMinM/XMaxM/ZMinM/ZMaxM` bounds), EXCLUDING the inclusion disk,
  `amp_rel` = 1.0 (identical for every speckle scatterer — deliberately: THEORY.md "The problem"
  explains why identical-amplitude, randomly-positioned coherent reflectors is the textbook setup
  that produces Rayleigh-distributed speckle, not decorative noise).

All positions are generated deterministically from `random.Random(42)` in a fixed order (wires,
then inclusion, then background) so the same command always produces byte-identical output
(CLAUDE.md §12).
