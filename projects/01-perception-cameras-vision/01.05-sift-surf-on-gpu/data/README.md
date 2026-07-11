# Data — 01.05 SIFT/SURF on GPU (harder, warp-level reductions)

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
| Kind | Synthetic (repo default, CLAUDE.md §8) — no public dataset applies (see `../scripts/download_data.ps1`'s honest no-op) |
| Generator / source | `python ../scripts/make_synthetic.py` |
| License | Synthetic — repo MIT license applies |
| Size (committed) | 4 files, ~197 KB total (three 256x256 8-bit PGMs, ~64 KB each, plus a small CSV) |
| Checksum (SHA-256) | see table below |
| Regenerate with | `python ../scripts/make_synthetic.py` (fixed seeds 42 and 999 — see module header) |

| File | SHA-256 |
|------|---------|
| `sample/scene_a.pgm` | `ad51adc4d158e60fd3fbdd6fabbc817b62a0b2196e7142fa5ceba8088820cf57` |
| `sample/scene_b.pgm` | `af5b1e5c980cb3e58a7e12a9ea6b23f57bd7db3ca555a62dac05d290cd603ffd` |
| `sample/neg_scene_c.pgm` | `ab8b97993a1e2bd7a37da3a443f7b6ada4f7e0961b1bd0dd769acd4fdaec3967` |
| `sample/transform.csv` | `9cd7a6e9a166ef4c5a28248020594dc1f773e567c43e3134f537b01e93874c76` |

### What each file is

- **`scene_a.pgm`** — 256x256, 8-bit grayscale (PGM P5), identity pose. The reference view: a hashed
  multi-scale checkerboard-and-disk scene (seed 42), analytically rendered with 4x4 supersampled
  anti-aliasing (see `make_synthetic.py`'s module header for why AA matters for DoG extrema stability).
- **`scene_b.pgm`** — 256x256, 8-bit grayscale, the **SAME** scene under a KNOWN ground-truth similarity
  transform (1.5x zoom + 20deg rotation + translation (10, -8) px + a +15 intensity-unit brightness
  offset). This is SIFT's whole reason for existing over 01.04's rotation-and-translation-only pair: a
  REAL scale change. Rendered by inverse-transforming every output pixel back into scene A's analytic
  scene function (never by warping A's raster), so the transform is EXACT ground truth, not approximate.
- **`neg_scene_c.pgm`** — 256x256, 8-bit grayscale, an UNRELATED scene (different layout, seed 999),
  identity pose + the same brightness offset. Used only as the negative control in the
  `negative_control` gate — proof the matcher is not self-confirming.
- **`transform.csv`** — human-readable copy of the ground-truth transform parameters. The AUTHORITATIVE
  copy (the one every gate actually checks against) is hardcoded, cross-referenced, in
  `../src/main.cu`'s `kTransform*` constants — this CSV exists so a learner can see the numbers without
  reading C++ (the 01.01/01.04 precedent for this split).

### Fields / format

- **PGM (P5)**: binary grayscale, `maxval` 255, one byte per pixel, row-major, origin at the TOP-LEFT
  (row 0 = top row of the image, matching every raster-image convention in this repo). Intensity units:
  0-255 "digital number" (no photometric calibration claimed — see PRACTICE.md). `../src/main.cu`
  normalizes to float `[0, 1]` before the SIFT pipeline runs (Lowe's classic thresholds, e.g.
  `kContrastThreshold=0.03`, are defined on that scale).
- **`transform.csv`** columns: `field` (name), `value` (float, printed to 6 decimals), `units`. Fields:
  `theta_deg` (rotation, degrees CCW — see `../src/main.cu`'s sign-convention note on GATE 2 for the
  row-axis subtlety this creates), `scale` (unitless zoom factor), `tx_px`/`ty_px` (translation, pixels),
  `brightness_offset` (intensity units, 0-255 scale), `center_x`/`center_y` (the rotation/scale pivot,
  pixels), `width`/`height` (pixels).

No SI units/frames apply here (this is 2-D image-plane data, not a 3-D robot state) — pixel coordinates
follow the plain raster convention (x = column, y = row, both zero-indexed, row increasing downward).
