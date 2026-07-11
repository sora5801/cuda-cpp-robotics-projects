# Data — 01.01 Full GPU image pipeline: debayer → undistort → rectify → resize → normalize, zero CPU copies

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
| Kind | Synthetic (default per CLAUDE.md §8) — a fully analytic RGGB Bayer scene with dense, exact ground truth |
| Generator / source | `../scripts/make_synthetic.py` (fixed seed 42, no CLI args for the committed sample) |
| License | Synthetic — repo MIT license applies; no third-party data involved |
| Size (committed) | 3 files, 553,005 bytes total (~540 KiB) — well under the 50 MB ceiling |
| Checksum (SHA-256) | see table below |
| Regenerate with | `python ../scripts/make_synthetic.py` (from `scripts/`; writes into `sample/`) |

| File | Bytes | SHA-256 |
|------|-------|---------|
| `sample/bayer_input.pgm` | 110,607 | `aa63fd2f8a68dfae6eaa1ed56f67d4e2db6bb0241de240f10f68ea2e455214f5` |
| `sample/true_rgb.ppm` | 331,791 | `42dcc1876f5a8b523d087fbbd4028fd548bf21bf2a79cb859fb04dac079a43a5` |
| `sample/smooth_mask.pgm` | 110,607 | `d00aa437ec1dc9f35729a2b08ee4bab245342a77172b114a8eddaa2323e36a9f` |

### Fields / format

All three files are `kFullW x kFullH` = 384x288 px (matching `kernels.cuh`'s `kFullW`/`kFullH` — the
single source of truth every file below cross-references). None carry SI units in the CLAUDE.md §3.1
sense (they are 2-D pixel arrays, not physical-quantity vectors); pixel coordinates follow this
project's stated camera-optical frame exception (`kernels.cuh`'s file header: z-forward, x-right,
y-down, row 0 at the top).

- **`bayer_input.pgm`** (PGM, P5, 8-bit grayscale, 1 byte/pixel). The RGGB Bayer mosaic — the
  pipeline's ONLY input. Byte at `(x,y)` is whichever of R/G/B `bayer_channel_at(x,y)`
  (`kernels.cuh`) says the mosaic measures there. Built by inverse-warping `true_rgb.ppm` through the
  camera model (rotate + Brown-Conrady distort — the physical direction a real lens actually bends
  light) and mosaicking — see `make_synthetic.py`'s file header for the exact algorithm.
- **`true_rgb.ppm`** (PPM, P6, 8-bit RGB, 3 interleaved bytes/pixel, `(y*W+x)*3+c`). The GROUND
  TRUTH: the scene as an ideal, undistorted, rectified camera would see it — a checkerboard (x,y in
  `[32,224)`, 24 px squares, for the straightness gate), a smooth low-frequency RGB gradient, and 3
  flat-color disks. Directly comparable to the pipeline's rectified-stage output (`demo/out/rectified.ppm`)
  — that comparison is main.cu's `color_fidelity` gate.
- **`smooth_mask.pgm`** (PGM, P5, 8-bit grayscale, 1 byte/pixel). 255 = this pixel is >= 6 px from the
  image border, the checkerboard rectangle, and every disk boundary (safe to score for color
  fidelity — a few pixels of legitimate interpolation error cannot hide a real bug there); 0 = near an
  edge (reported by the color-fidelity gate, never gated — edges are EXPECTED to show larger blend
  error). 54,371 / 110,592 px (49.2%) are mask-valid on the committed scene.

### Camera model (authoritative numbers; MUST match `../src/kernels.cuh`)

`fx=fy=380 px`, `cx=191.5, cy=143.5 px` (exact image center), Brown-Conrady `k1=-0.22, k2=0.06,
p1=0.0010, p2=-0.0008`, rectifying rotation `2 deg` about the camera's Y (image-down) axis. See
`kernels.cuh`'s file header for the full camera-model derivation and `THEORY.md` for the physics
behind each term.
