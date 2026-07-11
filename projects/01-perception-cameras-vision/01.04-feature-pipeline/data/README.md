# Data — 01.04 Feature pipeline: FAST/Harris detection, ORB descriptors, brute-force Hamming matcher

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
| Kind | Synthetic (default; no public dataset applies — see `../scripts/download_data.ps1/.sh`, which is a documented no-op stub for this project) |
| Generator / source | `../scripts/make_synthetic.py` (no arguments — writes directly into this folder) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | 4 files, ~197 KB total (three 65,551-byte PGMs + a 548-byte CSV) |
| Checksum | SHA-256, see table below |
| Regenerate with | `python ../scripts/make_synthetic.py` (deterministic: byte-identical output every run) |

### Files

| File | Format | Bytes | SHA-256 |
|------|--------|-------|---------|
| `sample/scene_a.pgm` | PGM (P5), 256x256, 8-bit grayscale | 65,551 | `53849030538039ceb0296122e5c65593883a637e3892a45a1cec66b35d048a05`\* |
| `sample/scene_b.pgm` | PGM (P5), 256x256, 8-bit grayscale | 65,551 | `127ae7c5f4379600135b07efe6b627342993a6e97524148a284427c0c2bc79fb`\* |
| `sample/neg_scene_c.pgm` | PGM (P5), 256x256, 8-bit grayscale | 65,551 | `bf4685fe0ddc5cafc401f885aa921cfce10926b1352a7c203224da4876d034ac`\* |
| `sample/transform.csv` | CSV, 8 key/value rows | 548 | `0b2b51165f556ff44ac4605f82167bcdf679a8dff71db3884289ea90b0c40b20`\* |

\* Every hash above is 64 hex characters (SHA-256); wrapped by the Markdown renderer, not shortened.
Recompute with `python -c "import hashlib;print(hashlib.sha256(open('FILE','rb').read()).hexdigest())"`.

### What each file is

- **`scene_a.pgm`** — the reference view: an analytically-rendered scene (5 checkerboard patches at
  different scales, orientations, and per-cell hashed colors, 4 filled disks, a smooth gradient
  background) at the identity pose. This is the "previous frame" / map keyframe in the pipeline's
  vocabulary (`main.cu`'s `sample.a`).
- **`scene_b.pgm`** — the **exact same scene**, rendered under a KNOWN similarity transform (in-plane
  rotation 12.0 degrees about the image center, translation (+7.0, -5.0) px, plus a +18 intensity
  brightness offset) — the "current frame". `../src/main.cu`'s `kTransform*` constants are the
  authoritative copy of this transform (cross-referenced from `../scripts/make_synthetic.py`); every
  ground-truth gate in `main.cu` is checked against exactly these numbers.
- **`neg_scene_c.pgm`** — a **different** scene (different checkerboard layout, different disk
  placement, different background gradient, generated from an unrelated seed) used ONLY as the
  negative control for the matching gates (`main.cu`'s `gate_negative_control` — see THEORY.md "How we
  verify correctness"). It bears no geometric relationship to `scene_a.pgm`/`scene_b.pgm`.
- **`transform.csv`** — the ground-truth transform parameters in human-readable form (see "Fields"
  below). This file is **documentation/provenance only** — `main.cu` does not parse it; the
  authoritative numbers are hardcoded (with a cross-reference comment) in `main.cu`, following the
  same-repo precedent set by project 01.01's checkerboard-geometry constants (CLAUDE.md's
  single-sourcing spirit applied to small, human-checked numeric constants rather than a
  machine-parsed file).

### Rendering notes (why these PGMs look the way they do)

- **Anti-aliased**: every pixel is a 4x4-supersampled average (16 sub-samples), not a single hard-edged
  point sample — see `make_synthetic.py`'s `render_image()` header for why this matters for feature
  **repeatability** specifically (an earlier, non-anti-aliased version of this scene measured
  single-digit-percent repeatability; THEORY.md "How we verify correctness" records the numbers).
- **Checkerboard cells are individually, deterministically hash-colored** (5-level grayscale palette),
  not simple black/white alternation — a strict two-tone checkerboard is locally IDENTICAL at every
  interior corner (the reason checkerboards are used for camera *calibration*'s specialized corner
  detectors, and a well-known trap for generic feature *matching*, whose entire premise is that a
  local patch looks distinctive). See `make_synthetic.py`'s `cell_color()`.
- All randomness (shape jitter, per-cell colors) comes from a hand-rolled **xorshift32** generator
  (seed 42 for `scene_a.pgm`/`scene_b.pgm`, seed 999 for `neg_scene_c.pgm`) — the SAME 4-line algorithm
  `../src/kernels.cuh` implements in C++ for the ORB sampling pattern, not Python's `random` module
  (implementation-defined internals) — see `make_synthetic.py`'s module header.

### Fields / format

**PGM (P5) files** — binary grayscale, one byte per pixel, row-major, top-left origin, 256x256 (`kW`,
`kH` in `../src/kernels.cuh`). Pixel value is intensity in `[0, 255]`, unitless (a synthetic scene, not
a radiometric measurement). No SI units apply — these are not physical camera captures.

**`transform.csv`** columns: `field, value, units`.

| field | value | units |
|-------|-------|-------|
| `theta_deg` | 12.0 | degrees, counter-clockwise, about the image center |
| `tx_px` | 7.0 | pixels |
| `ty_px` | -5.0 | pixels |
| `brightness_offset` | 18.0 | intensity units, 0..255 scale, added to `scene_b.pgm`/`neg_scene_c.pgm` |
| `center_x` | 127.5 | pixels ( = (256-1)/2 ) |
| `center_y` | 127.5 | pixels |
| `width` | 256 | pixels |
| `height` | 256 | pixels |

Transform convention: `forward(xa, ya) = R(theta_deg) * (xa - center, ya - center) + (center, center) +
(tx_px, ty_px)` — see `../src/main.cu`'s `forward_transform()` (an independent, double-precision
retyping of `../scripts/make_synthetic.py`'s `forward_transform()`, deliberately bypassing
`kernels.cuh` — the same "gate independence" principle `kernels.cuh`'s header explains for the
GPU-vs-CPU twins).
