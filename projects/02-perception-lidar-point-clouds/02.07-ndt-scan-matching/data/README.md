# Data — 02.07 NDT scan matching (Autoware-style map localizer)

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
| Kind | Synthetic (default) — no public dataset used. NDT needs a map, a scan, and an EXACT ground-truth pose between them to be verifiable at all; no real LiDAR recording ships an answer key, so synthesis is necessary here, not just the repo default. |
| Generator / source | `../scripts/make_synthetic.py` — an L-shaped corridor-into-a-room scene, a 16-beam (VLP-16-like) analytic raycaster, and a deterministic xorshift32 RNG (08.01/01.17's exact algorithm, reimplemented in Python). |
| License | Synthetic — repo MIT license applies. |
| Size (committed) | ~630 KB total (see the table below; well under the 50 MB ceiling). |
| Regenerate with | `python make_synthetic.py` (defaults: `--seed 42`, writes to `data/sample/`) — bit-for-bit reproducible. |

### Files, sizes, and checksums (SHA-256, generated with `--seed 42`)

| File | Size (bytes) | SHA-256 |
|------|--------------:|---------|
| `map.bin` | 480,008 | `cd9dfe5256e0720f41061a809bd10e3ff198101bb948fea65bf41b018c1f58b2` |
| `scan_main.bin` | 61,472 | `a5aa2b8a678615e5b8d11ea38aa1b9de87e5dd9d2bdc49f8dba6375b462537ba` |
| `scan_cohort.bin` | 15,236 | `b2b8c123c67478c95c8b056c00b24084afa084061e13c3d5906e4a41df4ddc91` |
| `scan_cohort_clean.bin` | 15,152 | `1ef4195fca094781f4cecaa1c94f0cdb875e0974b6de53ff2c99a5a498c192d1` |
| `icp_target.bin` | 8,696 | `cf0a789f5214f1b3e306afa446d3a50ece731e906891e539e8cd695e45355b3e` |
| `cohort.csv` | 43,302 | `d19a16aea32f7cd9a901694af61308d56cb41eafdd0a6e223c44e58f730dbe52` |
| `meta.csv` | 645 | `a4a51431a7a3849875e2be6f6f235fda6f552088bb1f4d0e83ccd9a14372654b` |

`scan_main.bin`/`scan_cohort.bin` changed from an earlier committed version of this table: a
finisher pass fixed a real RNG stream-desync bug in `generate_scan()` (see this project's THEORY.md
"Numerical considerations" — the outlier "wrong-depth" draw used to silently consume an extra draw
from the SAME stream every inlier's range noise reads from). `scan_cohort_clean.bin` is BYTE-FOR-
BYTE unchanged by that fix (its outlier branch is never taken at `outlier_fraction=0.0`, so it
never drew from the affected code path either before or after) — a nice independent confirmation
that the fix only touches what it should. `cohort.csv` changed because `COHORT_TRIALS_PER_BIN` was
raised from 15 to 40 (240 total trials, up from 90) for a statistically trustworthier basin
measurement, also documented there.

Re-running `python make_synthetic.py --seed 42` reproduces every byte above exactly (verify with
`sha256sum data/sample/*`).

### The scene

An L-shaped building, all axis-aligned primitives (SI units, right-handed, `x` forward / `y` left /
`z` up — CLAUDE.md §12): a **corridor** (`x` in [0,10] m, `y` in [-1.5,1.5] m — 3 m wide,
deliberately narrow so the corridor-sliding degeneracy the catalog bullet and THEORY.md discuss is
actually present in the data) opening into a **room** (`x` in [10,16] m, `y` in [-4,4] m) with a
**pillar** obstacle (center (13, 0, 0.5) m, half-extents (0.3, 0.3, 0.5) m). Wall height 2.5 m; no
ceiling (an intentional, documented simplification — see `main.cu`/THEORY.md).

### `map.bin`, `scan_main.bin`, `scan_cohort.bin`, `scan_cohort_clean.bin`, `icp_target.bin` — binary point-cloud format

02.06's exact "PC01" format (cited):

```
offset 0, 4 bytes  : ASCII magic "PC01"
offset 4, 4 bytes  : uint32 little-endian point count N
offset 8, N*12 bytes: N * (float32 x, float32 y, float32 z), little-endian, meters
```

- `map.bin` — 40,000 points, uniform-on-area sampled across the scene (MAP frame).
- `scan_main.bin` — 5,122 points, a full-resolution (16 channels × 360 azimuth steps = 5,760 beams
  cast) simulated scan from the TRUE sensor pose, WITH the documented outlier fraction. Points are
  in the **SENSOR-LOCAL frame** (origin at the sensor; apply `T_map_scan` from `meta.csv` to reach
  the MAP frame). Used for the assembly/voxel twin gates and the before/after artifact.
- `scan_cohort.bin` — 1,269 points, a reduced-resolution (90 azimuth steps = 1,440 beams cast) scan
  from the SAME true pose, WITH outliers — used for the many-trial convergence/accuracy/basin gates
  (kept smaller so 240 trials × two algorithms finishes in seconds).
- `scan_cohort_clean.bin` — 1,262 points, the SAME beam directions and range-noise draws as
  `scan_cohort.bin` but the outlier fraction forced to 0 at generation time (RNG streams kept
  aligned beam-for-beam — see the script's `generate_scan()` docstring) — the paired baseline for
  the `outlier_robustness` gate.
- `icp_target.bin` — 724 points, a simple voxel-average downsample of `map.bin` (leaf 0.5 m,
  Python-only, independent of the C++ NDT voxel machinery) — the compact ICP contrast's target
  cloud (`kernels.cuh`'s `icp_point_to_point_cpu`).

Sensor model: 16 channels, ±15° elevation (VLP-16-like), range gated [0.3, 16.0] m, range noise
σ=0.02 m (Gaussian), TRUE injected outlier fraction 5% (a "wrong-depth return" model: the beam
direction is correct but the reported range is a spurious uniform draw over the valid range — a
dynamic-object/multipath stand-in). Ground truth sensor pose: `R = I` (yaw = 0, facing +x down the
corridor), `t = (5.0, 0.0, 1.2)` m — standing mid-corridor at a plausible robot-mounted LiDAR
height. The optimizer is never told `R = I`; every cohort trial perturbs full 6-DOF away from it.

### `cohort.csv` — the perturbation cohort

```
# columns: trial_id,bin_index,magnitude_trans_m,magnitude_yaw_deg,
#          r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz
```

240 rows (6 magnitude bins × 40 trials/bin — raised from 15/bin during a finisher pass so the
smallest bin's measured convergence rate is statistically trustworthy, not one small sample's luck;
THEORY.md "Numerical considerations" tells the story). `r00..r22,tx,ty,tz` is the INITIAL GUESS `T_map_scan`
(row-major rotation, translation in meters) every registration trial in `main.cu` starts from — a
random-direction translation offset at the bin's magnitude, composed with a random-sign yaw offset
at the bin's paired magnitude, applied as a LEFT/world-frame perturbation of the ground truth (the
same convention `kernels.cuh`'s `retract()` uses). Magnitude bins: translation
{0.2, 0.5, 0.8, 1.2, 1.6, 2.0} m, yaw {5, 10, 15, 20, 25, 30}° (paired 1:1 by index) — deliberately
large enough at the top end to find a real basin boundary (01.17's exact "otherwise the sweep
teaches nothing about basin size" argument, cited), not just to converge every time.

### `meta.csv` — ground truth and generation parameters

```
# GT_POSE row: r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz  -- T_map_scan
# COUNTS row: n_map,n_scan_main,n_scan_cohort,n_scan_cohort_clean,n_icp_target
# PARAMS row: range_noise_sigma_m,true_outlier_fraction,leaf_coarse_m,leaf_fine_m,icp_target_leaf_m
```

Three label-prefixed rows (08.01's row-label CSV convention, cited), parsed by `main.cu`'s
`load_meta()`.
