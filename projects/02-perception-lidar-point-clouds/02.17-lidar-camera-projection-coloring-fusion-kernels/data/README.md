# Data — 02.17 LiDAR-camera projection/coloring fusion kernels

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

Grading a fusion pipeline honestly needs an answer key no real sensor pair can provide: the EXACT true
surface color every LiDAR return hit, and whether that 3-D point is genuinely visible from the camera's
own (different) origin. A real rig can only ever tell you "what pixel did I land on" — never "was that
the right pixel." Synthetic data is therefore not merely convenient here; it is the *only* way to get an
honest evaluation. `scripts/make_synthetic.py` ray-casts a small scene from two physically separate
origins (a pinhole camera and a 2-D-scanning LiDAR mounted 0.30 m above and 0.05 m behind it — the
identical rig geometry 01.18/02.02 use) and writes:

| Property | Value |
|----------|-------|
| Kind | Synthetic (repo default, CLAUDE.md §8) |
| Generator | `python ../scripts/make_synthetic.py --seed 42 --el-count 31 --az-count 120` (the committed defaults) |
| License | Synthetic — repo MIT license applies (CLAUDE.md §8) |
| Size (committed) | `rgb.ppm` 56.3 KiB, `lidar_points.csv` 190.3 KiB — ~247 KiB total |
| Checksum (SHA-256) | `rgb.ppm`: `f00c7d67ad4083ac774b8752445bcc935fe850cda154f62c671284b6b0f96052`<br>`lidar_points.csv`: `3a270309aa7f7c272f7146c1baef48cf7d69d2a47e7a36f7661eeab438000579` |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (writes into `sample/`) |

No public dataset applies (see `../scripts/download_data.ps1`'s honest no-op and its header comment for
why): even KITTI/nuScenes-class camera-LiDAR datasets do not ship per-point "was this occluded from the
*camera's* viewpoint" ground truth — that oracle only exists because the scene is ray-traced twice, once
per sensor, in this script.

### The scene (what the numbers below describe)

A ground plane plus four flat-colored (unshaded — see `../README.md` "Limitations" for why) boxes:

| Surface | Color (R,G,B) | Role |
|---|---|---|
| `ground` | (100,100,100) gray | floor plane, z=0 |
| `occluder` | (200,60,60) **red** | 4 m out, top at z=1.6 m — hides part of `background` from the camera only (see below) |
| `background` | (60,160,70) **green** | 12 m out — the occlusion cohort's TRUE color |
| `blue_box` | (60,90,200) blue | ordinary, unoccluded color-boundary test subject |
| `yellow_box` | (230,210,60) yellow | ordinary, unoccluded color-boundary test subject |

**The occlusion geometry, worked (THEORY.md derives it in full):** the LiDAR sits 0.30 m *higher* than
the camera (z=1.80 m vs. z=1.50 m). Similar-triangles geometry through the occluder's top edge (z=1.6 m,
4 m out) shows the LiDAR's line of sight clears the occluder for any `background` point above z=1.2 m,
while the camera's line of sight only clears it above z=1.8 m. Background points with z in (1.2, 1.8) m
are therefore genuinely visible to the LiDAR but hidden from the camera — real parallax occlusion from
the sensor baseline, not a synthetic label. `scripts/make_synthetic.py` confirms this independently, per
point, by casting a *second* ray from the camera's own origin toward each LiDAR hit.

**Measured on the committed sample** (seed 42, 31 elevations × 120 azimuths):

- 3,368 total LiDAR returns; per-surface hit counts: ground 883, occluder 612, blue_box 195,
  yellow_box 127, background 1,551.
- 138/3,368 = 4.10% are ground-truth-occluded (`visible=0`) — the occlusion cohort.
- 3,053/19,200 = 15.90% of camera pixels receive at least one projected LiDAR return (z-buffer-deduped).

### Fields / format

**`rgb.ppm`** — binary P6 PPM, 160×120, the camera's RGB rendering (2×2 supersampled for anti-aliasing).
The ONLY color source the C++ pipeline ever reads.

**`lidar_points.csv`** — one row per LiDAR return, comma-separated, `#`-prefixed comment header:

| Column | Type | Units / frame | Meaning |
|---|---|---|---|
| `x,y,z` | float | meters, LiDAR frame (x-forward/y-left/z-up, origin-translated only) | the return's position — the **only** columns `src/kernels.cuh`'s `LidarPointF` and every kernel/CPU-twin consume |
| `true_r,true_g,true_b` | float | [0,255] | the EXACT flat surface color the LiDAR ray hit — **evaluation-only** ground truth, never touches a kernel |
| `visible` | int | 0/1 | 1 iff a second ray cast from the CAMERA's own origin toward this exact 3-D point reaches it unobstructed — **evaluation-only** ground truth |
| `surface` | string | — | human-readable object name (see the scene table above) — read by no C++ code, informational only |

`main.cu` parses `x,y,z` into `LidarPointF` (the pipeline's only input) and `true_r/true_g/true_b/visible`
into a parallel `Truth` array that **never** reaches a kernel or a `reference_cpu.cpp` function — keeping
it outside both verified code paths is what makes `main.cu`'s evaluation gates an *independent* check
(CLAUDE.md's "independent gate" discipline, following 01.18's precedent).

Range noise: LiDAR returns carry ±0.02 m (1σ) Gaussian range noise (illustrative MEMS/ToF-class sensor,
applied via the repo's portable xorshift32 generator — never Python's `random` module, so the file is
bit-reproducible across machines).

> **Self-containment note (CLAUDE.md §4):** this project's extrinsic (`kTCameraLidar` in
> `src/kernels.cuh`) and camera intrinsics are numerically IDENTICAL to 01.18/02.02's own constants (the
> same illustrative rig) — cited, not re-derived, and not referenced across project folders at build or
> run time; every number this project needs is compiled into its own `src/kernels.cuh`.
