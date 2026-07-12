# Data — 02.18 Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)

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

| Property | Value |
|----------|-------|
| Kind | Synthetic (default, CLAUDE.md §8) — no public dataset used. See `scripts/download_data.ps1`/`.sh` for the specific reason CADC/WADS (the real weather-LiDAR datasets that would apply here) are pointed at, not mirrored. |
| Generator / source | `../scripts/make_synthetic.py` — a real 16-beam spinning-LiDAR ray-cast against a closed-form structured scene (ground plane, two walls, a car), overlaid with three independent Beer-Lambert-extinction atmospheres (SNOW/RAIN/DUST). Ground truth (real vs. scatterer, and which scatterer type) is exact by construction — see that script's module docstring for the full physics derivation. |
| License | Synthetic — repo MIT license applies (CLAUDE.md §8). |
| Size (committed) | `points.csv`: 161,318 bytes (157.5 KiB). Well under the 50 MB ceiling. |
| Checksum (SHA-256) | `points.csv`: `8de9251722a202b77267646c0072d53e5ec213f6c91d535a3ee49301e28cd289` |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (the default — the committed file IS this exact invocation's output). |

### Size math

`points.csv` has 3 weather scans of up to `kBeamsPerScan = 16 * 100 = 1,600` beams each (fewer rows
than beams: a beam that neither hits real structure nor scatters off a particle produces no point at
all — "clean miss", see the generator's tallies) — measured at 1,074 (snow) + 1,164 (rain) + 1,218
(dust) = **3,456 data rows**, plus a 15-line `#`-prefixed header. Each row is
`weather,x,y,z,intensity,is_real,scatterer_type,surf_cohort` at roughly 45-47 bytes/row (6-decimal
floats, small integers, newline) → ~157 KB, matching the measured size.

### Fields / format

**`points.csv`** — one row per LiDAR return (real surface OR airborne scatterer), `#`-prefixed header
lines first (parsed and asserted by `../src/main.cu` against `../src/kernels.cuh`'s beam-model
constants — `num_beams`, `azimuth_steps`, `max_range_m`), then data rows:

| Column | Type | Units / frame | Meaning |
|--------|------|----------------|---------|
| `weather` | int (0/1/2) | — | Which of the three independent weather captures this point belongs to: 0=SNOW, 1=RAIN, 2=DUST. All three ray-cast the SAME static real structure (see the generator's module docstring for why the scene is held fixed — a deliberate, documented scope cut, README "Limitations"). |
| `x,y,z` | float | meters, SENSOR frame (sensor at the origin) | The point's 3-D position. Range `r = sqrt(x^2+y^2+z^2)` is *derived*, never stored (`kernels.cuh`'s "POINT-RECORD LAYOUT" note). |
| `intensity` | float | unitless, [0,1] | Real points: Lambertian reflectance `rho_cohort * cos(incidence)` + sensor noise. Scatterer points: `rho_type * (particle cross-section / beam footprint area)` + sensor noise — see `make_synthetic.py`'s module docstring for the full partial-beam-interception derivation. |
| `is_real` | int (0/1) | — | **Ground truth**, used only by `main.cu`'s gates/artifacts, never by the filtering kernels: 1 = real surface return, 0 = airborne scatterer return. |
| `scatterer_type` | int (-1/0/1/2) | — | **Ground truth**: -1 = n/a (a real point), 0 = SNOW, 1 = RAIN, 2 = DUST. |
| `surf_cohort` | int (-1/0/1/2/3) | — | **Ground truth**, real points only: -1 = n/a (a scatterer point), 0 = GROUND, 1 = WALL_NEAR (~8 m), 2 = WALL_FAR (~32 m), 3 = CAR (~18-22 m). |

Per-weather point/cohort tallies actually observed in the committed file (from the generator's own
stdout when it wrote this file):

| Weather | Points | Real | Scatterer | Real cohorts (ground / wall_near / wall_far / car) |
|---------|--------|------|-----------|------------------------------------------------------|
| SNOW | 1,074 | 939 | 135 | 573 / 293 / 49 / 24 |
| RAIN | 1,164 | 898 | 266 | 551 / 285 / 41 / 21 |
| DUST | 1,218 | 566 | 652 | 356 / 170 / 29 / 11 |

The DUST scan's real-point count is lower than SNOW/RAIN's: the dust plume sits close to the sensor
(3-7 m, `kernels.cuh`'s file header / `make_synthetic.py`'s `DUST_PLUME_BOX`) astride every beam's
path, so at this project's tuned density (documented in `make_synthetic.py`'s `DUST_DENSITY_PER_M3`
comment) a meaningful fraction of beams that would otherwise reach real structure scatter off dust
first — attenuation honesty, not a bug (README "Limitations").
