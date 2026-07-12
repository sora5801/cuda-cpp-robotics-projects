# Data — 02.16 Multi-LiDAR merging + extrinsic refinement

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

**Synthetic by necessity, not just by repo default.** Extrinsic refinement needs a KNOWN
ground-truth mounting drift to be verifiable at all — no recorded multi-LiDAR fleet log ships with
an answer key for "how far did this sensor actually drift from its factory pose". `../scripts/
make_synthetic.py` builds a small structured "yard" (a ground plane plus four walls at three
mutually orthogonal orientations, plus decorative poles — see `../kernels.cuh` and `../THEORY.md`
for the geometry and why it was chosen), works out which of three rig-mounted LiDARs (MAIN, a
360° roof unit; LEFT and RIGHT, front-corner units) captures each world point from FOV/range alone,
and writes that capture TWICE — once with every sensor at its exact nominal (as-designed) mounting
pose (`aligned.csv`, the control), and once with LEFT and RIGHT carrying a small, documented,
per-sensor-different mounting drift (`drifted.csv`, the patient) — plus independent Gaussian LiDAR
range noise on every point, both cohorts.

| Property | Value |
|----------|-------|
| Kind | Synthetic (repo default, CLAUDE.md §8) — **and structurally required here**, see above |
| Generator | `python ../scripts/make_synthetic.py --seed 42` (the default; deterministic, stdlib-only, no numpy) |
| License | Synthetic — repo MIT license applies; no external data, no redistribution concerns |
| Size (committed) | 2 files, ~518 KiB total (well under the repo's 50 MB ceiling) |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (byte-identical output — verify against the checksums below) |

### Committed files and checksums

Regenerate and diff against these SHA-256 hashes to confirm byte-for-byte reproducibility (measured
on the files actually committed to this repo):

| File | Size (bytes) | Rows | SHA-256 |
|------|--------------:|-----:|---------|
| `sample/aligned.csv` | 265810 | 8138 (3 header + 8135 data) | `92c974b88d60aa4e35d01a05dee29d9656fe9a0f7351222c4202960a10b3abeb` |
| `sample/drifted.csv` | 265957 | 8138 (3 header + 8135 data) | `7e95b4be61e494fe3cac9958e647ae5b16232c58825aab9ad41c2e6ae1c6eafb` |

**Size math:** 8135 data rows/file × (5 comma-separated int/float fields, `%.6f` precision) ≈ 32.6
bytes/row average → ~265 KB/file, matching the committed sizes. Per-sensor row counts (identical
between the two cohorts — see "Two cohorts, one world grid" below): MAIN 4261 (ground 1681,
wall_front 410, wall_left 680, wall_right 680, wall_rear 410, pole 400), LEFT 1937 (ground 737,
wall_front 250, wall_left 680, wall_rear 70, pole 200), RIGHT 1937 (mirror of LEFT). 4261+1937+1937
= 8135.

### Fields / format

Plain CSV, `#`-prefixed comment header (3 lines: format description, sensor/surface id legend,
per-cohort point count + noise sigma), then one row per `(sensor, point)`:

```
sensor_id,surface_id,x,y,z
```

| Field | Type | Meaning |
|-------|------|---------|
| `sensor_id` | int, 0/1/2 | `0`=MAIN (roof, 360°), `1`=LEFT (front-left corner), `2`=RIGHT (front-right corner) — `../kernels.cuh`'s `kSensorMain`/`kSensorLeft`/`kSensorRight`. |
| `surface_id` | int, 0-5 | `0`=ground, `1`=wall_front, `2`=wall_left, `3`=wall_right, `4`=wall_rear, `5`=pole (decorative; excluded from every plane fit) — `../kernels.cuh`'s `kSurface*` constants. |
| `x,y,z` | float, meters | The point's position in **its own sensor's raw frame** (NOT base/vehicle frame — `main.cu` transforms into base frame itself, once per cohort per candidate extrinsic, using `nominal_extrinsic()`/the refined estimate as appropriate). Right-handed, SI (CLAUDE.md §12). Includes independent Gaussian range noise, sigma = 6 mm (`NOISE_SIGMA_M` in `../scripts/make_synthetic.py`), applied along the sensor-to-point direction (a 1-D range-noise model, not isotropic Cartesian jitter — `../THEORY.md` "Numerical considerations" discusses the simplification). |

### Two cohorts, one world grid

Both files are generated from the exact SAME deterministic world-point grid and the exact SAME
per-point sensor-visibility decision (FOV/range membership, evaluated once from each sensor's
NOMINAL mount position — a documented simplification `../kernels.cuh`'s file header explains).
Only the coordinates differ, because each cohort expresses a captured point through a DIFFERENT
"true" extrinsic for LEFT/RIGHT: `aligned.csv` uses the nominal extrinsic for every sensor (drift =
zero); `drifted.csv` composes each drifted sensor's nominal extrinsic with its own small, documented
rotation+translation drift (`../scripts/make_synthetic.py`'s `DRIFT` table, mirrored in
`../kernels.cuh`'s `kDrift`). This means any difference a learner sees between the two cohorts'
merged clouds is caused ONLY by the drift itself — never by an accidental change in which points got
sampled or assigned to which sensor.

### Ground truth (not stored in the CSVs — derived from documented constants instead)

Neither file stores extrinsics: both the nominal and the true (drifted) per-sensor extrinsics are
pure functions of a handful of documented constants (mount position, mount yaw, drift
rotation/translation), duplicated — not shared, since Python and C++ cannot share a header —
between `../scripts/make_synthetic.py`'s `MOUNT`/`DRIFT` tables and `../kernels.cuh`'s
`kNominalMount`/`kDrift` tables, with a "must match" comment at each. `main.cu` re-derives both the
nominal extrinsic (used for merging-before and as the refinement's initial guess) and the true
extrinsic (used ONLY as the answer key for the RECOVERY gates) from `../kernels.cuh`'s constants —
never from the CSV.
