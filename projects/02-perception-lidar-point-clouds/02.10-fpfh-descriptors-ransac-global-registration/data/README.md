# Data — 02.10 FPFH descriptors + RANSAC global registration

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
| Kind | 100% synthetic — one room (floor + 4 walls + a box crate + a cylindrical pillar), scanned from two known sensor poses. |
| Generator / source | `../scripts/make_synthetic.py --seed 42` (xorshift32, stdlib-only Python — CLAUDE.md §8/§12). |
| License | Synthetic — repo MIT license applies; no external data, no redistribution concerns. |
| Size (committed) | 6 point-cloud `.bin` files + 1 `pairs_meta.csv`, **240,375 bytes total** (~235 KiB) — see the table below. |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (reproduces every byte below exactly). |

### The scene

A 20 m x 20 m room (floor `z=0`, 4 walls to `z=3.0` m), a box crate (1.8 x 1.8 x 1.2 m, at world
`(3.0,-3.0)`) and a cylindrical pillar (radius 0.45 m, world `(-4.0,4.0)`), sampled on a jittered grid
(`GRID_SPACING_M=0.42`, `JITTER_FRAC=0.35`) — **3,768 world points total**. Two sensor poses see
overlapping subsets of this world set:

- **pose A** (the SOURCE frame for every pair): world position `(-2.0,-2.0,1.3)` m, yaw 0 deg, max
  range 13.0 m.
- **pose B** (the TARGET frame): world position `(4.553,2.589,1.3)` m (pose A + 8.000 m at heading 35
  deg), yaw **140.0 deg** — the RATIFIED negative-control relative pose, far outside any local
  method's convergence basin (see `icp_negative_control` in `../src/main.cu` and `../THEORY.md`).

A world point is "seen" by a pose if it is within range AND its analytic outward normal faces the
sensor (`dot(normal, direction-to-sensor) > 0.02`) — range + back-face visibility, **not** a full
ray-caster (`../README.md` "Limitations & honesty" states this simplification's cost honestly).

### The three committed pairs

| Pair | n_source | n_target | Overlap (measured) | Noise sigma | Role |
|------|---------:|---------:|--------------------:|------------:|------|
| `pair0` | 3158 | 2649 | 62.0% | 0.000 m (clean) | `[info]` clean-cohort contrast |
| `pair1` | 3158 | 2649 | 62.0% | 0.010 m | **headline / VERIFY pair** — `registration_recovery`, `icp_negative_control`, `ransac_formula` all gate on this pair |
| `pair2` | 3158 | 1496 | 33.5% | 0.010 m | low-overlap stress cohort — `[info]`-only, honestly reported, never required to succeed |

Overlap = `|visible(A) ∩ visible(B)| / |visible(A) ∪ visible(B)|` over the shared world-point set —
**measured** by `make_synthetic.py` from the actual visibility masks, not assumed (the ~60%/~30%
targets in the catalog brief were hit by tuning `MAX_RANGE_STRESS_M`, not by fabricating the number).
Noise: Gaussian, along each point's own true normal, drawn **independently** per scan (a real sensor's
noise on the same physical point differs shot to shot).

### Checksums & byte-size math (regenerate and `sha256sum` to verify)

Binary layout (see `../scripts/make_synthetic.py`'s `write_cloud_bin`): `9` bytes magic `"FPFHPAIR1"`
+ `4` bytes `int32 n` + `n*3` `float32` xyz + `n` `int32` world_idx = `13 + 16*n` bytes.

| File | n | Bytes (`13+16n`) | SHA-256 |
|------|--:|------------------:|---------|
| `pair0_source.bin` | 3158 | 50541 | `5458c88bcd421dc9b069ca8048179aef7c0f9bceb493e5d1235d943baff9aff8` |
| `pair0_target.bin` | 2649 | 42397 | `518ca644a8f4c2afc87f2472f9df0d97207bfb369c40d493195efb1cce793d79` |
| `pair1_source.bin` | 3158 | 50541 | `6fdb266ff1ea76cc5d765c20757badaf4a2d57e8aed50977b12ceeb0dbe0b444` |
| `pair1_target.bin` | 2649 | 42397 | `cadad98fb3dc24a0230b0239bede4f162be935c793755b82a731fd52d89ccb12` |
| `pair2_source.bin` | 3158 | 50541 | `e83192566caf5ef064fb769d46e43c2494ecd11440c63447901aedff23e6583a` |
| `pair2_target.bin` | 1496 | 23949 | `77683fb583fe6fb13f2d9765d4a97f8d49dfd950e5b0d7d5e77aa0ed67489465` |
| `pairs_meta.csv` | — | 439 | `fb91f48b7f80aa7a059bd06919c7f0bd21f0ccdc83bfe7958ebbd4f5ed0e9414` |

(`pair0_source.bin` and `pair2_source.bin` are byte-identical to `pair1_source.bin`'s *n* and *xyz
generation order* but NOT identical bytes — each pair's noise draw consumes RNG state independently,
so even pair0's "clean" source and pair1's "noisy" source differ; verify with the checksums above, not
by assuming reuse.)

### Fields / format

`pairs_meta.csv` columns: `pair,n_source,n_target,overlap_fraction,noise_sigma_m,tx,ty,tz,qw,qx,qy,qz,`
`relative_yaw_deg,relative_trans_m`. `(tx,ty,tz)` (meters) and `(qw,qx,qy,qz)` (unit quaternion, repo
order, CLAUDE.md §12) together are the **TRUE** `T_target_source`: apply `R(q)*p_source + t` to land a
source-frame point in the target frame — exactly what `registration_recovery`/`icp_negative_control`
grade the recovered pose against.

Each `<pair>_source.bin` / `<pair>_target.bin`: `xyz` (meters, that scan's OWN sensor-local frame,
+x forward/+y left/+z up) and `world_idx` (int32, the index into the shared 3,768-point WORLD set —
the **ground-truth correspondence key**: the same `world_idx` in both a pair's source and target file
names the identical physical point, used by `descriptor_invariance` to compare FPFH computed
independently in each local frame, with zero dependence on the algorithm's own matching).
