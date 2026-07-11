# Data — 01.17 Camera-LiDAR / camera-camera extrinsic calibration (batched reprojection-error optimization)

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

**Kind:** 100% synthetic — no public dataset applies (see `../scripts/download_data.ps1`'s "decision"
comment for why: real calibration datasets ship the ANSWER, not the raw correspondences a learner could
re-derive it from). Every ground-truth extrinsic and every correspondence is exact by construction; the
demo adds its own sensor noise at runtime (see `../src/kernels.cuh`'s "CORRESPONDENCE LAYOUT" note).

**Generator:** `../scripts/make_synthetic.py` (pure Python 3.12 stdlib, no numpy — an independent
reimplementation of the same pinhole-projection/Rodrigues math the C++ side uses, deliberately, per
`../src/reference_cpu.cpp`'s file header). **Regenerate with:** `python make_synthetic.py --seed 42`
(the default; every number below is from this exact invocation).

| File | Kind | Size | SHA-256 |
|------|------|------|---------|
| `sample/cam_lidar_diverse.csv` | synthetic — repo MIT license applies | 3673 bytes | `2683c55df210424e66459722a9c2b9505d7d05130de38b68e5e3bad7e0610905`* |
| `sample/cam_lidar_coplanar.csv` | synthetic — repo MIT license applies | 3670 bytes | `427b65111a3224f72f5e653a0e091e9491ae0325498e578ce7c0cdb02aaf6467`* |
| `sample/cam_cam_diverse.csv` | synthetic — repo MIT license applies | 3681 bytes | `13ae874676d9b7704d2dbebcca213dc87f8e0cd853eab1dcd852858e4898e36f`* |
| `sample/cam_background.pgm` | synthetic — repo MIT license applies | 19215 bytes | `7aa0d457fcba84d30ffdc90451430cdab1a9303eb57aeab438a2794728726622`* |

*(SHA-256 as computed on the reference machine at generation time; recompute after regenerating —
`sha256sum sample/*` — to confirm a byte-identical regeneration.)

Total committed size: ~30 KB, well under the repo's tiny-sample ceiling.

### Fields / format — `cam_lidar_diverse.csv`, `cam_lidar_coplanar.csv`, `cam_cam_diverse.csv`

Every correspondence file shares one format: a `#`-comment header (provenance + regeneration command),
then three labeled rows, then 48 (`kNumViews=12 * kPointsPerView=4`, `src/kernels.cuh`) correspondence
rows:

| Row label | Fields | Units / frame | Meaning |
|-----------|--------|----------------|---------|
| `OMEGA_GT` | `wx,wy,wz` | rad, so(3) axis-angle log-rotation | `R_gt = Exp(OMEGA_GT)` (Rodrigues — `so3_exp` in `src/kernels.cuh`); the TRUE rotation of `T_dest_src` |
| `T_GT` | `tx,ty,tz` | m, dest frame | the TRUE translation of `T_dest_src` |
| `INTRINSICS` | `fx,fy,cx,cy` | px | the dest camera's pinhole intrinsics — MUST match `src/kernels.cuh`'s `kFx/kFy/kCx/kCy` exactly (a manually-maintained invariant across the Python/C++ language boundary; both sides comment this) |
| `CORR` (x48) | `view,point,px_src_m,py_src_m,pz_src_m,u_true_px,v_true_px` | `view`/`point`: unitless indices (0-11, 0-3); `p_src_*`: m, SOURCE sensor frame (lidar frame for the two camera-LiDAR files, camera-1 frame for the camera-camera file); `uv_true_*`: px, EXACT (noise-free) pixel the point projects to under `(OMEGA_GT, T_GT, INTRINSICS)` | one correspondence: a known 3-D point and its exact projected pixel |

`p_src` for the camera-LiDAR files is a fiducial's TRUE position as the LiDAR would measure it (before
`main.cu` adds range/angular noise); for the camera-camera file it is a board point's position in
camera 1's frame (treated as EXACT — see README "Limitations" for why camera-camera carries no
source-point noise). `T_dest_src` follows SYSTEM_DESIGN.md §3.3 naming: `T_camera_lidar` for the two
camera-LiDAR files, `T_camera2_camera1` for the camera-camera file.

### Fields / format — `cam_background.pgm`

A 160x120 (matching `kImageWidth`/`kImageHeight` exactly) binary grayscale (P5) PGM: a synthetic
vertical-gradient "floor/wall" backdrop with a faint grid and seeded speckle, used only as the overlay
artifact's background (README "Expected output"). No calibration content is encoded in this image; it
exists purely so the "money shot" artifact has a plausible scene to draw reprojected points onto.
