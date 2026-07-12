# Data — 02.20 LiDAR intensity calibration across channels

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** (where one genuinely teaches more) are fetched by `../scripts/download_data.ps1`
  / `.sh` — idempotent, with source URL, expected size, and checksum documented below. **Respect every
  license**; registration-gated or no-redistribution datasets are pointed at, never mirrored.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | Synthetic (default, CLAUDE.md §8) — no public dataset used. See [`../scripts/download_data.ps1`](../scripts/download_data.ps1)/`.sh` for why: recovering exact per-channel gain ground truth from a real LiDAR requires a laboratory reflectance-target rig this repo cannot reproduce, and no public dataset publishes the true per-channel gain it used to synthesize a scan. |
| Generator / source | [`../scripts/make_synthetic.py`](../scripts/make_synthetic.py) — a real 16-beam spinning-LiDAR ray-cast against a closed-form structured scene (ground plane, two vertical walls, a small high-reflectivity test panel), overlaid with a documented per-channel-gain × reflectivity × range-falloff × incidence-angle forward model. Ground truth (per-channel gain, per-point surface id and reflectivity) is exact by construction — see that script's module docstring for the full derivation. |
| License | Synthetic — repo MIT license applies (CLAUDE.md §8). |
| Size (committed) | `scan_primary.csv`: 53,404 bytes. `scan_degenerate.csv`: 54,676 bytes. `gains_true.csv`: 438 bytes. Total ~108 KB — well under the 50 MB ceiling. |
| Checksum (SHA-256) | `scan_primary.csv`: `3674838c049397d8e22e844d450649eb1113eede2bd6e725ecc3f57f49fdfa50` |
| | `scan_degenerate.csv`: `72340b2ac4a232731ca9eab5211ed75a35d29dc073ba9eca757b1aabf326914e` |
| | `gains_true.csv`: `72d38311d8cc725dea851b9b4e30b90dba38ab2caadecc0aa15b469dfcc04078` |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (the default — the committed files ARE this exact invocation's output). |

### Size math

`scan_primary.csv` ray-casts 16 channels × 81 azimuth steps = 1,296 candidate beams; a beam that hits
neither a real surface nor open sky within `kMaxRangeM` produces no row ("clean miss") — measured at
**1,032 data rows** (ground=172, wall_near=496, panel=56, wall_far=308), plus a 17-line `#`-prefixed
header. Each row is `channel,x,y,z,intensity,surf_id,R_true` at roughly 50-52 bytes/row (6-decimal
floats, small integers, newline) → ~53 KB, matching the measured size. `scan_degenerate.csv` ray-casts
the identical 1,296-beam grid with channel 15 retargeted upward (file header below); it additionally
hits the isolated target for every one of its 81 azimuth steps, so it has 24 more rows than a
straightforward re-run of the primary geometry would (1,056 total: ground=172, wall_near=465,
panel=52, wall_far=286, isolated_target=81) — a few points move between wall_near/panel/wall_far
because the SAME xorshift32 stream continues (not reset) between the two scans (matching project
02.18's identical two-scans-share-one-stream convention), so per-point noise draws differ slightly
scan to scan even where the ray geometry is unchanged.

### Fields / format

**`scan_primary.csv` / `scan_degenerate.csv`** — one row per LiDAR return, `#`-prefixed header lines
first (documenting the beam model, scene, and the exact regenerate command — parsed by nothing in
`main.cu`; this project's calibration algorithm re-derives its own geometric model from
`kernels.cuh`'s scene constants rather than parsing the header, the same "single-sourced constants,
not a parsed contract" choice project 01.09 makes for its own problem geometry), then data rows:

| Column | Type | Units / frame | Meaning |
|--------|------|----------------|---------|
| `channel` | int, [0,16) | — | Which of the 16 beams produced this return. |
| `x,y,z` | float | meters, SENSOR frame (sensor at the origin, x-forward/y-left/z-up) | The point's 3-D position. Range `r = sqrt(x^2+y^2+z^2)` and azimuth `atan2(y,x)` are *derived*, never stored (`kernels.cuh`'s point-record-layout note; `main.cu`'s file header derives why azimuth is recoverable exactly). |
| `intensity` | float | unitless, >= 0 | The RAW measured intensity: `g[channel] * R_true * f(r) * cos(theta) + noise` (`kernels.cuh` SECTION 3 / `make_synthetic.py`'s module docstring derive every term). |
| `surf_id` | int, {0,1,2,3,4} | — | **Ground truth**, used only by `main.cu`'s gates/artifacts, never by the calibration kernels: 0=GROUND, 1=WALL_NEAR (~8 m), 2=PANEL (~8 m, brighter test patch), 3=WALL_FAR (~20 m), 4=ISOLATED_TARGET (degenerate scan only). |
| `R_true` | float | unitless, (0,1] | **Ground truth**: the struck surface's true reflectivity (ground=0.22, wall_near=0.55, panel=0.85, wall_far=0.35, isolated_target=0.45). |

**`gains_true.csv`** — 16 rows, `channel,true_gain`. **Ground truth**, used only by `main.cu`'s
`gain_recovery` gate (and every gate built on top of it); the calibration algorithm never opens this
file. True gains span **0.60–1.40** (illustrative "detector aging / laser power / alignment variance"
magnitudes, dated 2026-07-12 — see `make_synthetic.py`'s `TRUE_GAINS` comment; deliberately
non-monotonic, since real per-channel drift is not a clean ramp).

### Per-surface / per-channel tallies actually observed in the committed files

(from the generator's own stdout when it wrote these files)

| Scan | Points | ground | wall_near | panel | wall_far | isolated_target |
|------|--------|--------|-----------|-------|----------|------------------|
| primary | 1,032 | 172 | 496 | 56 | 308 | 0 |
| degenerate | 1,056 | 172 | 465 | 52 | 286 | 81 |

Per-channel point counts (81 azimuth steps possible per channel): channels 0–4 (the steepest downward
elevations) register all 81 — every azimuth step hits *something*, usually the ground; channels 5–15
register 57 in the primary scan — the shallower elevations produce clean misses (open sky) at the
azimuth extremes, past every surface's rectangular extent (README "Limitations" states this as
expected geometry, the same "some beams see nothing" honesty project 02.18's generator states for its
own scene). Channel 15 registers 57 points on `wall_near`/`wall_far`/etc. in the **primary** scan (a
normal channel) but a full 81 points on `isolated_target` **only** in the **degenerate** scan, where its
elevation is retargeted skyward — it shares zero voxels with any other channel there by construction
(`main.cu`'s `unobservable_channel` gate).
