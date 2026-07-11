# Data — 01.15 Background subtraction for fixed-workspace cells

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
| Kind | Synthetic (100% — no public dataset applies; see `../scripts/download_data.ps1`'s header for why) |
| Generator / source | `python ../scripts/make_synthetic.py` |
| License | Synthetic — repo MIT license applies (CLAUDE.md §8) |
| Size (committed) | 160 files, 1,968,320 bytes total (1.88 MiB) — see "Size decision" below |
| Checksum | Deterministic from `SEED = 42`; three representative files below (regenerate and `sha256sum` any file to verify byte-for-byte reproducibility) |
| Regenerate with | `python make_synthetic.py` (writes into this folder) or `--out-dir <path>` to inspect elsewhere |

| File | SHA-256 |
|------|---------|
| `sample/frames/frame_000.pgm` | `e4cb765a0bc59ccbfa0b76fe1f3eab4fee65b58e843ee8d99498b84d3a6b3a3c` |
| `sample/frames/frame_060.pgm` | `3d867221dad1a0e151d59b865290cb4ae41b9094b0f0474da93fdb88fcef4cb8` |
| `sample/frames/frame_159.pgm` | `8ddd1348b49c73cc11c16a79107c8ecb4d4486e2847cae5ea1b1d1bebddccb78` |

### Size decision (documented per the project brief)

The catalog's illustrative frame size (240x180) would commit `240*180 + 14`-byte PGMs (the 14-byte
P5 header: `"P5\n240 180\n255\n"`) x 160 frames = **~6.86 MiB** for the full designed sequence — far
above every neighboring sample in this repository (most are tens to low hundreds of KiB; the
biggest, 30.01's bundled multi-milestone sample, is 1.5 MiB). Two ways to shrink that were
considered:

1. **Commit fewer frames.** Rejected: the five designed events (README "The algorithm in brief")
   need their full frame ranges (E1 spans 31 frames, E3's drift needs the whole 160-frame span to
   reach +15%) — thinning the sequence would silently narrow the very phenomena this project teaches.
2. **Commit smaller frames.** Chosen: 128x96 instead of 240x180. Each PGM is
   `128*96 + 14 = 12,302` bytes; 160 of them total **1,968,320 bytes (1.88 MiB)** — verified by `du`
   on the actual committed folder. Small enough to sit comfortably inside this repo's "tiny sample"
   norm while keeping the full, undiminished 160-frame event schedule.

The demo therefore runs **fully offline**: `data/sample/frames/frame_000.pgm .. frame_159.pgm` are
committed directly; `main.cu` never invokes Python and never downloads anything (CLAUDE.md §4/§8).

### Fields / format

Each `frame_NNN.pgm` (`NNN` = zero-padded 3-digit frame index, `000`..`159`) is a **binary PGM (P5)**,
128x96, grayscale, 8-bit (`maxval 255`) — the ASCII header `P5\n128 96\n255\n` followed by
`128*96 = 12,288` raw bytes, one per pixel, row-major (row `y`'s 128 bytes precede row `y+1`'s). Each
byte is a synthetic camera intensity in `[0, 255]`, **unitless** (no radiometric calibration is
modeled — see THEORY.md "Numerical considerations"). There is no explicit timestamp field: frame
index `t` **is** the timeline, at an implied 30 Hz (README "System context" / THEORY.md).

The exact scene — a fixed-camera work-cell backdrop (bench + two fixtures) plus five DESIGNED
events (E1/E5 an intruding two-link arm, E2 a permanently-placed box, E3 a uniform +15% illumination
ramp, E4 a blinking status lamp) — is defined by the constants in
[`../src/kernels.cuh`](../src/kernels.cuh) SECTION 2 (the single-sourced, machine-readable ground
truth `main.cu`'s gates read directly) and independently transcribed in
[`../scripts/make_synthetic.py`](../scripts/make_synthetic.py) (the renderer — Python cannot
`#include` a `.cuh` file, so the same numbers are hardcoded a second time there, with a
cross-referencing comment in both files; see `reference_cpu.cpp`'s header for why duplicating DATA
this way is not the twin-independence concern CLAUDE.md's verification ruling addresses).
No per-frame ground-truth mask file is committed: because every event's geometry is a closed-form
function of frame index, `main.cu` recomputes exact ground truth on demand from those same constants
— there is nothing to store.
