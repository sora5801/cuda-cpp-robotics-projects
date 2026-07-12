# Data — 02.13 Dynamic point removal (raycast free-space carving)

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
| Kind | Synthetic (default, CLAUDE.md §8) — no public dataset used. |
| Generator / source | `../scripts/make_synthetic.py` — closed-form ray/box and ray/cylinder intersection against an analytic scene (a wall, a thin pole, a car, a pedestrian, and an isolated "ghost" crate), so ground truth is exact, not approximated. |
| License | Synthetic — repo MIT license applies (CLAUDE.md §8). |
| Size (committed) | `beams.csv`: 1,603,043 bytes (1,565.5 KiB). `poses.csv`: 709 bytes. Well under the 50 MB ceiling. |
| Checksum (SHA-256) | `beams.csv`: `d145b11ecaa0f6cd14e6a7ef109fb5283b826e4c27f24e7f51709cc69aeb7903`<br>`poses.csv`: `eb0dec0c8cf37e30b9e5815776f2599c1d7f72bd39ecf45c142a56f3e1776c03` |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (the default — the committed files ARE this exact invocation's output). |

### Size math

`beams.csv` has `K * kNumBeams * kAzimuthSteps = 10 * 16 * 180 = 28,800` data rows plus a small
`#`-prefixed header, each row `scan_id,dir_x,dir_y,dir_z,is_hit,range_m,cohort,truth_dynamic` at
roughly 56 bytes/row (8 decimals per float, 3 floats, 4 small integers, newline) → ~1.6 MB, matching
the measured size. `poses.csv` has exactly `K = 10` rows.

### Fields / format

**`beams.csv`** — one row per LiDAR beam, `#`-prefixed header lines first (parsed and asserted by
`../src/main.cu` against `../src/kernels.cuh`'s beam-model constants — see `num_scans`,
`num_beams`, `azimuth_steps`, `max_range_m` in the header), then data rows:

| Column | Type | Units / frame | Meaning |
|--------|------|----------------|---------|
| `scan_id` | int | — | Which of the 10 scans this beam belongs to (0-indexed); indexes `poses.csv`. |
| `dir_x,dir_y,dir_z` | float | unitless, WORLD frame, unit vector | Beam direction (sensor orientation is identity throughout — see the script's module docstring — so this is already world-frame). |
| `is_hit` | int (0/1) | — | 1 = real return, 0 = max-range (no return within 20 m). |
| `range_m` | float | meters | The (Gaussian-noise-perturbed, sigma 2 cm) measured range if `is_hit`; EXACTLY `max_range_m` (20.0) otherwise. |
| `cohort` | int | — | Ground truth object id: 0=WALL, 1=POLE, 2=WALL_EDGE, 3=CAR, 4=PEDESTRIAN, 5=GHOST, -1=NONE (a miss). **Used only by gates/artifacts — never by the carving/classification algorithm.** |
| `truth_dynamic` | int (0/1/-1) | — | Ground truth: 1 if the cohort is CAR/PEDESTRIAN/GHOST (something that moved), 0 if WALL/POLE/WALL_EDGE (permanent structure), -1 for a miss (not applicable). |

**`poses.csv`** — one row per scan, `#`-prefixed header first:

| Column | Type | Units / frame | Meaning |
|--------|------|----------------|---------|
| `scan_id` | int | — | 0-indexed scan number. |
| `px,py,pz` | float | meters, world frame | Sensor position (`T_world_lidar`'s translation). |
| `qw,qx,qy,qz` | float | unit quaternion, repo order (w,x,y,z) | Sensor orientation — always identity `(1,0,0,0)` in this project's scenario (README "Limitations"). |
| `t_s` | float | seconds, monotonic | Illustrative scan timestamp (2 Hz mapping-session cadence; PRACTICE.md discusses real cadence policy). |

A hit point's world-space position is never stored a third time — every consumer derives it as
`P = origin(scan_id) + dir * range_m` (`kernels.cuh`'s file header names this the single-sourced
formula).
