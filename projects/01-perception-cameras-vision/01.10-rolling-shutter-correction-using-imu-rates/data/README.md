# Data — 01.10 Rolling-shutter correction using IMU rates

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
| Kind | 100% synthetic — no public dataset applies (see "Why no public dataset" below) |
| Generator / source | `../scripts/make_synthetic.py` (`python make_synthetic.py --seed 42`) |
| License | synthetic — repo MIT license applies |
| Size (committed) | 5 files, ~218 KB total (see checksum table) |
| Checksum | SHA-256, per file, below |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (the default; matches the committed files exactly) |

### Why no public dataset

Rolling-shutter correction needs analytically-known ground truth: the pixel-exact image an ideal
GLOBAL-SHUTTER camera would have captured at the same instant as a given rolling-shutter frame, plus the
camera's exact continuous orientation trajectory during readout. No real camera+IMU recording can provide
this after the fact — there is no way to "un-shutter" a real sensor's already-captured frame to recover
what it would have looked like without rolling-shutter artifacts. Synthetic authorship is not just this
repo's default policy here; it is the only way to get real ground truth for this problem at all
(`../scripts/make_synthetic.py`'s file header expands on this).

### Committed files (`sample/`)

| File | SHA-256 | Bytes |
|------|---------|-------|
| `rs_input.pgm` | `c69fcac7a50badce8e003c4eae2c7ab66d1ab8c9172ea3a38b5e034d183a2153` | 110,607 |
| `ground_truth_gs.pgm` | `e8e41485b7570f5657013396e724fa95ee399a6e53303a0e12c9472986a5164e` | 110,607 |
| `gyro_clean.csv` | `ef05692cac601bfc7a49e5a6714cb7becd9247798a0be1842d1e6f82a7723ba5` | 684 |
| `gyro_degraded.csv` | `d00f6c3b621ea70e03d4d06332394d85a1c4de90d6c4ca1406ff38fa75eab3ac` | 763 |
| `params.csv` | `d489ed54f5c00ddc98508710d71aa18748aee8dfb0807bc491ec3b44f348821d` | 754 |

### Fields / format

- **`rs_input.pgm`** — 384x288, 8-bit grayscale (PGM P5), row-major `uint8[y*384+x]`. The CAPTURED
  rolling-shutter frame: row `v` shows the scene as it was oriented at `t(v) = v * kLineTimeS` seconds
  into the frame's 25 ms readout (frame-relative time, `t0 = 0`). This is the pipeline's ONLY image
  input.
- **`ground_truth_gs.pgm`** — 384x288, 8-bit grayscale (PGM P5), same layout. The scene as an ideal
  GLOBAL-SHUTTER camera would have captured it at `t_ref` (the frame's middle row's own exposure time,
  ≈12.46 ms in) — the restoration gate's ground truth. Never fed to the correction pipeline; used only
  for scoring.
- **`gyro_clean.csv`** / **`gyro_degraded.csv`** — ~10 rows each, columns `t_s` (frame-relative seconds,
  float), `wx_rad_s`, `wy_rad_s`, `wz_rad_s` (body/camera-frame angular velocity, rad/s, camera-optical
  axes — x-right/y-down/z-forward, the exception CLAUDE.md §3.2 permits camera optics, stated at this API
  boundary). `gyro_clean.csv` is the TRUE angular velocity sampled at 200 Hz with no noise;
  `gyro_degraded.csv` is the SAME sample times with a constant +8 deg/s bias and ±0.18 rad/s uniform
  noise added per axis per sample (illustrative, not a calibrated IMU noise model — see
  `../scripts/make_synthetic.py`'s comment on `GYRO_BIAS_DPS`/`GYRO_NOISE_HALF_WIDTH_RAD_S`). Both cover
  a window slightly wider than the frame's own readout window so `../src/main.cu`'s trajectory
  interpolation never needs to extrapolate.
- **`params.csv`** — key/value provenance record of every constant the generator used (geometry, timing,
  the true rotation profile's amplitude/frequency/phase per axis, the degraded-gyro error-model
  parameters, and the seed). NOT read by `../src/main.cu` (which uses its own baked `kernels.cuh`
  constants — this file exists purely so a human can confirm the two agree, and so the exact generation
  parameters travel with the data).

All angles/rates are SI (radians, radians/second); all timestamps are frame-relative seconds
(`kernels.cuh`'s file header states the float-vs-double time convention this project uses, a documented
narrowing of docs/SYSTEM_DESIGN.md §3.5's general "double" timestamp rule).
