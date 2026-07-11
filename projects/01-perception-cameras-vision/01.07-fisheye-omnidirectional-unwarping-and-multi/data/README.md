# Data — 01.07 Fisheye/omnidirectional unwarping and multi-camera surround-view stitching

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
| Kind | Synthetic (default per CLAUDE.md §8) — 4 analytically ray-cast fisheye camera renders + an exact orthographic ground-truth BEV crop |
| Generator / source | `../scripts/make_synthetic.py` (fixed seed 42, xorshift32-based value noise, no CLI args for the committed sample) |
| License | Synthetic — repo MIT license applies; no third-party data involved |
| Size (committed) | 6 files, 1,229,524 bytes total (~1.2 MiB) — well under the 50 MB ceiling |
| Checksum (SHA-256) | see table below |
| Regenerate with | `python ../scripts/make_synthetic.py` (from `scripts/`; writes into `sample/`; ~13 s on the reference machine) |

| File | Bytes | SHA-256 |
|------|-------|---------|
| `sample/fisheye_front.ppm` | 230,415 | `c4fe284e5c622e9643e2a167339393404a7104f0467278866b982c895ca76307` |
| `sample/fisheye_left.ppm` | 230,415 | `0d71b1e1b0a732e4aae12121cae6e1cbd84f4473a95dbcd0cdb03c724eea76ab` |
| `sample/fisheye_right.ppm` | 230,415 | `9579aa0a3a962dd4bbeafd9a0a9d7cfa5d719f236c0c846a838424e0144e96cc` |
| `sample/fisheye_rear.ppm` | 230,415 | `9592f6a808f0ad5f54b7447f63eaac3590ff7714bf7cfca47f4e79c72989154e` |
| `sample/bev_ground_truth.ppm` | 307,215 | `a796f77efe1e305ea75bcc4b5d01ff17d28378ae7831c958ff500df5b96000b0` |
| `sample/rig_extrinsics.csv` | 649 | `5392c67c56d18a15a49f0be0a892d14fd23d4f0868dbd76a7017bd42569c2165` |

### Fields / format

All 4 `fisheye_*.ppm` files are `kFishW x kFishH` = 320x240 px (matching `../src/kernels.cuh`'s
`kFishW`/`kFishH` — the single source of truth every file below cross-references); `bev_ground_truth.ppm`
is `kBevW x kBevH` = 320x320 px. None carry SI units directly (they are 2-D pixel-color arrays), but
every pixel's GEOMETRIC meaning does: fisheye pixels follow the camera-optical frame stated in
`kernels.cuh`'s file header (z-forward, x-right, y-down); `bev_ground_truth.ppm`'s pixel grid follows
`bev_pixel_to_ground()`'s vehicle-frame (x-forward, y-left, z-up, SI meters) mapping.

- **`fisheye_{front,left,right,rear}.ppm`** (PPM, P6, 8-bit RGB, 3 interleaved bytes/pixel,
  `(y*W+x)*3+c`). One image per rig camera (`../src/kernels.cuh`'s `kCamFront..kCamRear` order —
  same order this project loads/indexes them everywhere). Each is an analytic RAY-CAST render (2x2
  supersampled) of the synthetic parking-lot ground plane + 3 tall objects through that camera's
  equidistant fisheye model and mount pose — see `make_synthetic.py`'s file header for the exact
  algorithm. Pixels outside the lens's 92.5-degree design half-FOV are rendered pure black (0,0,0) —
  the vignette, physics not a rendering shortcut (`kernels.cuh`'s `kFishFx` comment).
- **`bev_ground_truth.ppm`** (PPM, P6, 8-bit RGB, same layout). The GROUND TRUTH: an orthographic
  top-down render of the ground TEXTURE ONLY — no cameras, no ray casting, no objects, no occlusion.
  Directly comparable to the pipeline's stitched BEV output (`demo/out/bev.ppm`) in flat, well-covered,
  seam-free, object-free regions — that comparison is `main.cu`'s `bev_ground_truth` gate; the SAME
  file, restricted to near-object regions, is `flat_ground_assumption`'s negative-control comparison.
- **`rig_extrinsics.csv`** — documentation/provenance mirror of the 4 rig cameras' mount positions and
  `R_cam_vehicle` rotation matrices, independently computed in Python (`make_synthetic.py`'s
  `build_camera_matrix()`) from the same nominal-basis + 45-degree-tilt construction `kernels.cuh`
  hardcodes in C++ — two languages agreeing is this project's cross-check that the hardcoded floats are
  not a transcription slip. **Not read by the C++ program** (its own copy is compiled in, per
  `kernels.cuh`'s `rig_extrinsic_for()`); this file exists purely so a reader can inspect the rig
  numbers without cross-referencing two languages by eye.

### Camera model + rig (authoritative numbers; MUST match `../src/kernels.cuh`)

Fisheye (equidistant, `r = f*theta`): `fx = 74.0 px/rad`, principal point at the exact image center
(`cx=159.5, cy=119.5`), 185-degree-class design FOV (`kFishValidHalfFovRad = 92.5 deg` half-angle).
Rig: 4 cameras (front/left/right/rear), each tilted 45 degrees down from horizontal; mounts at
`(2.0, 0.0, 0.6)`, `(0.0, 1.0, 1.1)`, `(0.0, -1.0, 1.1)`, `(-2.0, 0.0, 0.6)` meters (vehicle frame). See
`kernels.cuh`'s PART 1/PART 3 headers for the full derivation and `THEORY.md` for the physics behind
every term.

### The synthetic scene (fully specified in `make_synthetic.py`'s file header)

Dark asphalt (value-noise mottled, xorshift32-hashed, 0.4 m cells) + 2 dashed white lane lines
(Y = ±2.2 m) + a straight boundary edge (Y = -0.4 m, X in [2.0, 6.5] m — the straightness gate's
target) separating a flat light "loading zone" from the asphalt + 3 tall objects (2 cylinders, 1 box —
the flat-ground-assumption gate's targets).
