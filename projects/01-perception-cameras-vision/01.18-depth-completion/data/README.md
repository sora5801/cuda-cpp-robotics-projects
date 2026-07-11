# Data — 01.18 Depth completion: sparse LiDAR + RGB → dense depth

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

This project needs something no real sensor can provide: **exact, dense, per-pixel ground-truth
depth** to grade a *completion* algorithm against. That is only obtainable from a synthetic scene, so
`../scripts/make_synthetic.py` ray-casts a small 3-D scene (ground plane, three boxes, a cylinder, and
two purpose-built test surfaces — see below) from two different sensor origins and writes three files.

| Property | Value |
|----------|-------|
| Kind | Synthetic (repo default, CLAUDE.md §8) — nothing here is real sensor data |
| Generator | `../scripts/make_synthetic.py` (ray-caster; technique follows 01.07's renderer, reimplemented independently) |
| License | Synthetic — this repository's MIT license applies to the generator and its output |
| Size (committed) | `rgb.ppm` 56 KB, `truth_depth.bin` 75 KB, `lidar_points.csv` 55 KB (~186 KB total) |
| Regenerate with | `python make_synthetic.py --seed 42 --az-count 280` (run from `scripts/`) |
| Determinism | xorshift32 only (no `random` module) — bit-identical output on every platform for the same seed |

### SHA-256 checksums (committed `data/sample/` files)

```
6da4383cc47f8ca63ec4be2ececb90c2aa2a56f4a9e6b23777ec49de6b312d09  rgb.ppm
dc5873a1f767d6fa0368ea74b89375c7438e6db4bfeccff9cae395d40d927dd0  truth_depth.bin
b47ae15e2f4d3c1aa50f90d9f4d72a93afb365a341ca7c679c171d4057817476  lidar_points.csv
```

Recompute with `sha256sum data/sample/*` (or `Get-FileHash` on Windows) after regenerating to confirm
byte-identical output.

## The scene

Body/world frame: x-forward, y-left, z-up (SYSTEM_DESIGN.md convention), origin on the ground under the
rig. A pinhole camera sits at `(0, 0, 1.5 m)` looking dead level along +x; a 16-beam LiDAR sits at
`(-0.05, 0, 1.80 m)` — 0.30 m above and 0.05 m behind the camera, an illustrative roof-LiDAR-over-
windshield-camera rig (PRACTICE.md §1). The extrinsic this implies, `kTCameraLidar` in `src/kernels.cuh`,
is a **fixed, already-solved** constant (a clean axis-permutation rotation plus that translation) — the
kind of number project 01.17's calibration optimizer would hand you; this project *consumes* it rather
than solving for it.

Objects, all at different depths from the camera (didactic role in parentheses; exact placement and
colors are in `OBJECTS` in `make_synthetic.py`):

- **near_box / mid_box / far_box** — ordinary Lambertian-shaded boxes at ~3.6 m / ~7.5 m / ~13.4 m,
  the baseline "does depth completion work at all" subjects.
- **pole** — a vertical cylinder (~9.75 m), a *curved* depth edge rather than an axis-aligned one.
- **trap_board** — the **texture trap**: a small (0.7 m × 0.7 m), perfectly flat, unlit board painted
  with a high-contrast checkerboard (10 cm cells). Every pixel on its face shares the SAME true depth
  despite huge RGB gradients — the case where "RGB edge implies depth edge" is a *false* prior.
- **camo_near / camo_far** — the **camo edge**: two flat, unlit, *identically* colored boxes (RGB
  `(130,130,130)`, no shading) at different depths (~4.5 m and ~7–11 m). Their shared silhouette is a
  REAL depth discontinuity with near-zero RGB contrast — the case where the prior *fails to fire*.

## Fields / format

### `rgb.ppm` — the camera's guidance image

Binary PPM (P6): `"P6\n160 120\n255\n"` header, then 160×120×3 raw `uint8` bytes, row-major, RGB
order. 2×2-supersampled for anti-aliasing. This is the ONLY color information the algorithm sees —
`truth_depth.bin` is evaluation-only and never read by `src/kernels.cu` / `reference_cpu.cpp`.

### `truth_depth.bin` — exact dense ground truth (evaluation-only)

Raw binary, no header: 160×120 `float32`, row-major, **meters**, camera-frame **z-depth** (`Pcam.z`,
the pinhole/z-buffer convention — NOT Euclidean range from the camera). One exact ray per pixel
(*not* supersampled — blending two objects' depths at a silhouette would be physically meaningless).
`-1.0` marks a pixel whose ray hit nothing (sky) or exceeded `kMaxDepthM` (20 m) — the same
`kInvalidDepth` sentinel `src/kernels.cuh` defines for the sparse map. Width/height are the compile-time
constants `kImageWidth`/`kImageHeight` in `src/kernels.cuh` — **manually kept in sync** with this
script's `IMG_W`/`IMG_H` (no shared source between Python and CUDA C++ in this repo; a comment in both
files flags the contract).

### `lidar_points.csv` — raw LiDAR returns (algorithm input)

CSV, `#`-comment header rows, then columns `x,y,z` — one row per beam return, **meters, in the LiDAR's
own frame** (the same x-forward/y-left/z-up convention the body frame uses, just translated to the
LiDAR's mount point — see `src/kernels.cuh`'s `kTCameraLidar`). Generated by ray-casting a 16-beam
elevation fan (`-15°` to `+15°`, VLP-16-like) × 280 azimuth samples spanning `±32°` around forward, from
the LiDAR's own origin — genuine multi-sensor parallax, not a reprojection of the camera's depth map.
Range noise: independent zero-mean Gaussian, **σ = 0.02 m** (2 cm, illustrative MEMS/ToF-class LiDAR),
drawn with the script's own xorshift32 + Box–Muller generator.

**Measured densities** (after projecting into the 160×120 image with `src/kernels.cuh`'s z-buffer
formula, deduplicating collisions — the exact number the C++ pipeline sees, printed by
`make_synthetic.py` on every regeneration):

| Subset | LiDAR points | Covered pixels | Density |
|--------|-------------:|----------------:|--------:|
| Full committed set (stride 1) | 1890 | 1150 / 19200 | **5.99%** |
| Main demo default (stride 2) | 945 | 839 / 19200 | **4.37%** |
| Density-sweep low end (stride 5) | 378 | 332 / 19200 | **1.73%** |

`src/main.cu` derives the main demo's ~4.37% density and the density-sweep's ~1.73%/4.37%/5.99% points
by deterministic index-stride subsampling of this ONE committed file — no second file needed, and the
strides are documented in `src/main.cu`'s `subsample()` calls.
