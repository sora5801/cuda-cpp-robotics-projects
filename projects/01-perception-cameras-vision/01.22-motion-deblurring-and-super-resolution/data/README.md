# Data — 01.22 Motion deblurring and super-resolution for inspection zoom

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
| Kind | 100% synthetic, generated in Python (stdlib only — no numpy, no PIL). |
| Generator | `../scripts/make_synthetic.py` |
| License | Synthetic — this repository's MIT license applies (no external data, no third-party rights). |
| Size (committed) | ~72 KB total (`sample/` folder, all 15 data files + this folder's own README). |
| Regenerate with | `python ../scripts/make_synthetic.py` (fixed seed 42; byte-identical output every run). |

`../scripts/download_data.ps1` / `.sh` intentionally do nothing meaningful here: there is no public
dataset this project draws from — every pixel is synthesized so the exact ground truth (the
un-blurred scene, the exact PSF, the exact sub-pixel registration of every low-res frame) is known,
which a real photograph never gives you. This is the same synthetic-first reasoning 01.10's and
01.11's data folders document.

## Files and their fields

| File | Format | Shape | What it is |
|------|--------|-------|------------|
| `truth.pgm` | PGM (P5), 8-bit grayscale | 128x128 | The ground-truth inspection scene — flat patch, high-contrast step edge, 7 hand-drawn dot-matrix glyphs, hashed texture, 3 bar-chart frequency groups. NEVER seen by any restoration method; the sole basis of every gate's PSNR/correlation measurement. |
| `blurred.pgm` | PGM (P5) | 128x128 | `truth.pgm` circularly convolved with `psf_truth.csv`'s line PSF, plus additive Gaussian sensor noise (std 3.0 DN). Milestone 1's only input. |
| `psf_truth.csv` | CSV, `size,size` header + `size` rows of `size` floats | 15x15 | The exact motion-blur kernel used to make `blurred.pgm`: a line segment, length 9.0px, angle 20.0deg from +x, rasterized with bilinear anti-aliasing, sums to 1.0 exactly. |
| `psf_mismatch.csv` | Same format as `psf_truth.csv` | 15x15 | The SAME line length at a DELIBERATELY WRONG angle (45.0deg, 25deg off) — the PSF-mismatch honesty test's input; never used to generate `blurred.pgm`. |
| `lr_frame_0.pgm` … `lr_frame_7.pgm` | PGM (P5) | 64x64 each | 8 low-resolution captures of the SAME scene, each a genuinely aliased (box-filtered, not merely blurred) sample at a known sub-pixel shift, plus independent Gaussian sensor noise (std 2.0 DN) per frame. Milestone 2's only input. |
| `shifts_truth.csv` | CSV, header + 8 rows | 8 rows | `frame,dx_lrpx,dy_lrpx` — the EXACT sub-pixel registration (LR-pixel units) each `lr_frame_*.pgm` was rendered at; a quarter-LR-pixel lattice. This project studies NON-BLIND super-resolution (known shifts); registration estimation is documented-only (README "Limitations"). |
| `params.csv` | CSV, `parameter,value` rows | 13 rows | Every generation parameter in one place (seed, blur length/angle/mismatch-angle/noise, LR scale/count/noise, supersample factor) — the same numbers `kernels.cuh` states as compile-time constants (a MUST-MATCH contract, see that file's header). |

### Units and conventions

All pixel values are DN (digital number / code value), 8-bit range [0, 255], matching the repo's
image convention (01.01/01.09/01.11). Angles are degrees, measured counter-clockwise from the
image's +x (right) axis. Shifts in `shifts_truth.csv` are in **LR-pixel units**; multiply by the
super-resolution scale factor (2) to convert to truth/HR-pixel units — `kernels.cuh` Section 3
documents this conversion at every call site that needs it.

## How this was generated (the honest anti-aliasing method)

`make_synthetic.py` renders every scene feature (bars, edges, glyphs, texture) BLOCK-CONSTANT per
truth pixel — every feature in this scene is a "hard" step pattern, so no sub-truth-pixel gradient
is lost by this choice (that script's header proves the construction is exact). It then
nearest-upsamples the truth image by 4x into an oversized (margin-padded) canvas, extracts a
WINDOW at each frame's known supersample-pixel offset, and box-downsamples that window 8x total to
LR resolution — the same area-integration a real sensor pixel performs. Because a shift offset is
usually NOT a multiple of the 4x upsample factor, this extraction genuinely blends across truth-pixel
boundaries: the resulting low-res frames are honestly, physically aliased captures, not merely
blurred copies of the truth image — exactly the property the `sr_resolution` gate (`../src/main.cu`)
depends on to tell single-frame bicubic upscaling (which cannot un-alias) apart from multi-frame
super-resolution (which can, up to the finer grid's own Nyquist limit).

## Checksums (SHA-256, computed on the committed files)

```
56a648ceb54a91217ae92236f8cb6b50204289bf788dce3b3762513eaeb18809  truth.pgm
eb60aca6b71d3956b3f86ca32a13c2106ac3a38c10c45a8bcc5b9c13ffb6e9cd  blurred.pgm
e17061105db10ecd3a520531146bd1bdb8436c7e1333fc1f2aac70f8b4b083fa  psf_truth.csv
53109657936e264a34af0f0b4b8b359955caf96dbc479c814e5a5d92d1528adf  psf_mismatch.csv
e07a6a66691d302fcc178d8f50b917872aa8da48791ae510d1ee8706c02d9622  shifts_truth.csv
2e35ac66bfee000123c9081b6c7991bfdcf4adc7a3592e8a94f52a3bc4335aa7  params.csv
caee3c56ddd0aaf7e302449f3468db27e23f54b29f28295c2e4ecdd122617a2e  lr_frame_0.pgm
5f5e81d0c8b924051845d4b5aa95de48978b327592f7d8f1b8d8a683c88022a7  lr_frame_1.pgm
d242f7e861c9c7a10f9f98665e7ee7acd80d4485643e70d375938da969b9c3df  lr_frame_2.pgm
99753eccdbb79337dccd7c9249a7eec30f0b041efc94148edcf5e61c9bdc8d49  lr_frame_3.pgm
c61fcd6718e876f10028f66e33e03972597b1cb29648c2eab312d9278d8e0d90  lr_frame_4.pgm
d7967dbb6914fd399716522455b70f75fd562eabdf0848e028080c627de6e456  lr_frame_5.pgm
617efeff487274c990a128c8aa5d1b5c15aba880580799502f978f5b427f1fd2  lr_frame_6.pgm
3c32a27712da7430704e48f61896edf700766c7b7618ee1f4715b15b55ffac9a  lr_frame_7.pgm
```

(Regenerate and verify with `python ../scripts/make_synthetic.py && sha256sum sample/*.pgm sample/*.csv` —
the seed is fixed, so a clean regeneration reproduces these hashes exactly.)

## Size math

`truth.pgm`/`blurred.pgm`: 128x128 = 16,384 pixel bytes + a 15-byte P5 header (`P5\n128 128\n255\n`)
= 16,399 bytes each, matching the checksummed files above exactly. `lr_frame_*.pgm`: 64x64 = 4,096
pixel bytes + a 13-byte header (`P5\n64 64\n255\n`) = 4,109 bytes each x 8 frames = 32,872 bytes.
CSVs total under 6.5 KB. **Committed sample total: ~72 KB** — three orders of magnitude under the
repo's 50 MB ceiling (CLAUDE.md §8).
