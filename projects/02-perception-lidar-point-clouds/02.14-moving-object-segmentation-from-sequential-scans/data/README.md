# Data — 02.14 Moving-object segmentation from sequential scans

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
| Generator / source | `../scripts/make_synthetic.py` — closed-form ray/box and ray/cylinder intersection (Kay & Kajiya 1986) against an analytic scene (a wall, a thin pole, and four differently-moving cars), so ground truth — including the per-point `disocclusion_band` flag — is exact, not approximated. |
| License | Synthetic — repo MIT license applies (CLAUDE.md §8). |
| Size (committed) | `scans.csv`: 229,123 bytes (223.8 KiB). `poses.csv`: 446 bytes. Well under the 50 MB ceiling. |
| Checksum (SHA-256) | `scans.csv`: `be4c082af5b03b6c03d0b42f360981aec7547f97d62fa421b81078f6398fb35e`<br>`poses.csv`: `948e19819ca361191bcf6119b76258eb3867f190a90c115ee056863ff4f3dcb2` |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (the default — the committed files ARE this exact invocation's output). |

### Size math

`scans.csv` has one row per BEAM THAT RETURNED A HIT, across `kNumScansWindow=5` scans of
`kNumBeams=16 * kAzimuthBins=360 = 5,760` beams each (28,800 beams fired total; this run's hit rate is
`8,869 / 28,800 ≈ 30.8%` — most beams either exceed `kMaxRangeM=30 m` or point into open sky/ground with
no modeled surface). Each row (`scan_id,ring,az_bin,range_m,cohort,truth_dynamic,disocclusion_band`) is
roughly 26 bytes (8-decimal range, five small integers, newline) → `8,869 * 26 ≈ 231 KB`, matching the
measured size. `poses.csv` has exactly `kNumScansWindow = 5` rows.

### Fields / format

**`scans.csv`** — one row per LiDAR beam that returned a hit (a beam with no return in
`kMaxRangeM=30 m` is simply absent — the organized range image's default "no data" state, `#`-prefixed
header lines first (parsed and asserted by `../src/main.cu` against `../src/kernels.cuh`'s beam-model
constants — see `num_scans_window`, `num_beams`, `azimuth_bins`, `max_range_m` in the header), then data
rows:

| Column | Type | Units / frame | Meaning |
|--------|------|----------------|---------|
| `scan_id` | int | — | Which of the 5 window scans this point belongs to (0-indexed; 4 = the CURRENT scan, 0-3 = previous scans, oldest first); indexes `poses.csv`. |
| `ring` | int | — | Elevation beam index (0-15; elevation = `-15 + 2*ring` degrees — `kernels.cuh`'s `kBeamElevMinDeg`/`kBeamElevStepDeg`). |
| `az_bin` | int | — | Azimuth column index (0-359; azimuth = `az_bin` degrees, CCW from local +x — `kernels.cuh`'s `kAzimuthStepDeg`). |
| `range_m` | float | meters | The (Gaussian-noise-perturbed, sigma 2 cm) measured range along this beam, in the sensor's OWN local frame at capture time. The point's local xyz is derived once from `(ring, az_bin, range_m)` — `kernels.cuh`'s `local_point_from_ring_az()` — never stored a second time. |
| `cohort` | int | — | Ground truth object id: 0=WALL, 1=POLE, 2=CROSSING_CAR, 3=ONCOMING_CAR, 4=RECEDING_CAR, 5=STOPPED_CAR. **Used only by main.cu's gates/artifacts — never by the reprojection/residual/CCL algorithm.** |
| `truth_dynamic` | int (0/1) | — | Ground truth: 1 if the cohort is one of the four car cohorts (something that can move), 0 if WALL/POLE (permanent structure). |
| `disocclusion_band` | int (0/1) | — | Ground truth, meaningful ONLY for `cohort=WALL` hits in the CURRENT scan (`scan_id=4`): 1 iff CROSSING_CAR's presence/absence in front of this exact world point TOGGLED somewhere across the 5-scan window (computed by re-ray-casting from every scan's own sensor position — the script's `wall_point_occluded_by_crossing_car()`). This is the exact definition of "this wall point sits in the disocclusion band" the `disocclusion_mitigation` gate reads. |

**`poses.csv`** — one row per scan, `#`-prefixed header first:

| Column | Type | Units / frame | Meaning |
|--------|------|----------------|---------|
| `scan_id` | int | — | 0-indexed scan number (0-4). |
| `px,py,pz` | float | meters, world frame | Sensor position (`T_world_lidar`'s translation) — a straight line along +x at `SENSOR_SPEED_MS=1.0` m/s. |
| `qw,qx,qy,qz` | float | unit quaternion, repo order (w,x,y,z) | Sensor orientation — always identity `(1,0,0,0)` in this project's scenario (README "Limitations"; the reprojection algebra itself is fully general — see `kernels.cuh`'s `reproject_point_to_current`). |
| `t_s` | float | seconds, monotonic | Scan timestamp, `scan_id * DT_S` (`DT_S=0.3 s` — the window's buffered sampling interval; README "Limitations" distinguishes this from the algorithm's own 10-20 Hz per-call compute budget, which `main.cu`'s `timing` gate measures separately). |

A hit point's world-space position is never stored a third time — every consumer derives it as
`origin(scan_id) + R(q) * local_point_from_ring_az(ring, az_bin, range_m)` (`kernels.cuh`'s shared
formula).
