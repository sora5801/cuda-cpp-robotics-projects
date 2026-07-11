# Data — 01.03 Optical flow: pyramidal Lucas-Kanade, Farneback, census-transform flow

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

Optical flow needs **pixel-dense** ground truth — every one of `kW*kH` pixels needs a known true
displacement, not just a handful of keypoints — so no redistribution-friendly public dataset applies
here (Middlebury, MPI-Sintel, and KITTI's flow benchmarks are research/non-commercial or
registration-gated; `../scripts/download_data.ps1`/`.sh` are documented, honest no-ops). An analytic
scene function evaluated at an inverse-transformed coordinate gives EXACT, closed-form ground truth for
free — the same technique project 01.04's `make_synthetic.py` uses for its checkerboard scene, applied
here to a continuous "hashed multi-scale texture" field instead (see `../scripts/make_synthetic.py`'s
module header for why a checkerboard specifically fails DENSE correspondence).

| Property | Value |
|----------|-------|
| Kind | Synthetic (default; no public dataset applies — see `../scripts/download_data.ps1`/`.sh`, documented no-op stubs) |
| Generator / source | `../scripts/make_synthetic.py` (no arguments — writes directly into this folder) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | 6 files, ~114 KB total (five 19,215-byte PGMs + a 1,058-byte CSV) |
| Checksum | SHA-256, see table below |
| Regenerate with | `python ../scripts/make_synthetic.py` (deterministic: byte-identical output every run) |

### Files

| File | Format | Bytes | SHA-256 |
|------|--------|-------|---------|
| `sample/scene_a.pgm` | PGM (P5), 160x120, 8-bit grayscale | 19,215 | `3923cd5cea1f84ca95ea777dcf1f718d36630c2de4fd800ff1e7847c9986ba57`\* |
| `sample/scene_b_translation.pgm` | PGM (P5), 160x120, 8-bit grayscale | 19,215 | `478c5aee8eafa2ab1c8d7f4b529c0185edbeefe21fadfaf2506dd5a4ba8a68c4`\* |
| `sample/scene_b_rotzoom.pgm` | PGM (P5), 160x120, 8-bit grayscale | 19,215 | `348d992cd8413f56706ddf9d465d19a6f787a21cdb95df4d9d99c8aec067edf4`\* |
| `sample/scene_b_translation_bright.pgm` | PGM (P5), 160x120, 8-bit grayscale | 19,215 | `2632a3b407222daa39a440cd8524d243a6812f187fbee019049f20dfb141aaae`\* |
| `sample/scene_b_zero.pgm` | PGM (P5), 160x120, 8-bit grayscale | 19,215 | `3923cd5cea1f84ca95ea777dcf1f718d36630c2de4fd800ff1e7847c9986ba57`\* (byte-identical to `scene_a.pgm` — see below) |
| `sample/ground_truth.csv` | CSV, provenance | 1,058 | `ba2565d2dfee488c51d03d1ce3c1836b228a5e6fc3a7218e92debd5bb04c41b8`\* |

\* Every hash above is 64 hex characters (SHA-256); wrapped by the Markdown renderer, not shortened.
Recompute with `python -c "import hashlib;print(hashlib.sha256(open('FILE','rb').read()).hexdigest())"`.

### What each file is — the four ground-truth pairs

Every pair shares `scene_a.pgm` as its reference ("frame A"); only "frame B" differs:

- **(a) `scene_b_translation.pgm`** — frame A translated by a KNOWN constant `(tx_px, ty_px) = (3.0,
  -3.0)`. Ground truth: `flow(x,y) = (3.0, -3.0)` at **every** pixel — "the exact gate" (README
  "Expected output"). `../src/main.cu`'s `kTranslateTxPx`/`kTranslateTyPx` are the authoritative,
  cross-referenced copy of these numbers.
- **(b) `scene_b_rotzoom.pgm`** — frame A rotated `6.0` degrees and scaled `1.05x` about the image
  center. Ground truth is an AFFINE field that varies smoothly across the frame (`../src/main.cu`'s
  `forward_rotzoom()`) — the scene that makes pyramidal coarse-to-fine initialization earn its keep
  (see the `pyramid_advantage` gate).
- **(c) `scene_b_translation_bright.pgm`** — `scene_b_translation.pgm` (identical geometry, identical
  ground-truth flow) plus a smooth HORIZONTAL brightness ramp, `0` at the left edge rising to `+51`
  intensity units (`~20%` of the `0..255` range) at the right edge. Isolates brightness robustness as
  the only new variable between pairs (a) and (c) — the census milestone's selling point.
- **(d) `scene_b_zero.pgm`** — byte-identical to `scene_a.pgm` (confirm via the matching SHA-256 above).
  Ground truth: `flow = (0, 0)` everywhere — the zero-motion negative control.

### Rendering notes (why these PGMs look the way they do)

- **Hashed multi-scale texture, not a checkerboard.** `scene_a.pgm`'s content is a sum of four OCTAVES
  of hash-based value noise (32, 16, 8, 4 px cells, smoothstep-interpolated) — continuous, non-periodic,
  and textured at multiple scales. `../scripts/make_synthetic.py`'s module header explains why a
  checkerboard (project 01.04's scene) specifically fails DENSE optical flow: strict two-tone
  alternation is locally self-similar at every cell corner, and periodic content aliases under motion
  (a shift by one period looks identical to zero motion). The OCTAVES' relative weights were tuned
  empirically — see `../THEORY.md` "How we verify correctness" for the measured before/after numbers
  that motivated giving the finest (4 px) octave real weight instead of a token amount.
- **Anti-aliased**: every pixel is a 3x3-supersampled average (9 sub-samples), the same "integrate over
  the pixel's footprint" technique project 01.04's `render_image()` uses.
- **Exact, closed-form ground truth**: frame B is rendered by evaluating the SAME continuous scene
  function at the INVERSE-transformed coordinate of every output pixel — never by warping frame A's
  raster (which would blur/interpolate and make the "ground truth" approximate, not exact).
- All randomness (the per-lattice-point hash) comes from a hand-rolled **xorshift32** generator (seed
  42), the same 4-line algorithm used throughout this repo's Python generators (e.g. 01.04's
  `XorShift32`) — never Python's `random` module (implementation-defined internals).

### Fields / format

**PGM (P5) files** — binary grayscale, one byte per pixel, row-major, top-left origin, 160x120 (`kW`,
`kH` in `../src/kernels.cuh`). Pixel value is intensity in `[0, 255]`, unitless (a synthetic scene, not
a radiometric measurement). No SI units apply — these are not physical camera captures.

**`ground_truth.csv`** columns: `field, value, units`.

| field | value | units |
|-------|-------|-------|
| `width` / `height` | 160 / 120 | pixels |
| `translate_tx_px` / `translate_ty_px` | 3.0 / -3.0 | pixels — constant flow for pair (a)/(c) |
| `rot_theta_deg` | 6.0 | degrees, counter-clockwise, about the image center — pair (b) |
| `rot_zoom_scale` | 1.05 | unitless scale factor, about the image center — pair (b) |
| `center_x` / `center_y` | 79.5 / 59.5 | pixels ( = (160-1)/2, (120-1)/2 ) |
| `brightness_grad_max` | 51.0 | intensity units, 0..255 scale — pair (c)'s horizontal ramp ceiling |

This file is **documentation/provenance only** — `main.cu` does not parse it; the authoritative numbers
are hardcoded (with a cross-reference comment) in `main.cu`'s `k*` constants, following the same-repo
precedent set by projects 01.01/01.04 (CLAUDE.md's single-sourcing spirit applied to small, human-checked
numeric constants rather than a machine-parsed file).
