# Data — 02.02 ROI crop, passthrough, organized↔unorganized conversion kernels

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
| Kind | Synthetic (repo default). A single-revolution 16-beam spinning-LiDAR organized scan of a virtual warehouse room, plus a handcrafted predicate-boundary "edge cohort" and a "ghost second-echo" collision table. |
| Generator | `../scripts/make_synthetic.py`, xorshift32 RNG, seed 42 |
| License | Synthetic — repo MIT license applies (no external data, no redistribution concerns) |
| Size (committed) | `sample/roi_scan.bin` — 198,708 bytes (~194 KiB) |
| Checksum | SHA-256 `c7aee5416d1edcd546a0bbb630c8af8792c438e33f2a6ddb368c5c90b611b95b` |
| Regenerate with | `python ../scripts/make_synthetic.py` (writes `sample/roi_scan.bin`; add `--seed N` to experiment, but the checked-in sample and `demo/expected_output.txt` assume the default seed 42) |

The scene, the 16-beam elevation table, and the raycaster are reused near-verbatim from
[`02.01`](../../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/scripts/make_synthetic.py)'s
`make_synthetic.py` (cited in this project's own generator's module docstring) — the same virtual
16 m x 16 m room (floor + four open-topped walls + three obstacle boxes), sensor at the origin,
+x forward / +y left / +z up (SYSTEM_DESIGN.md §3.2).

### Binary format: `roi_scan.bin`

All fields little-endian, written with explicit `struct.pack` formats (never a raw struct dump), so
the layout is compiler-independent. `../src/main.cu`'s `load_sample()` reads the identical sequence.

| Offset | Bytes | Field | Meaning |
|--------|-------|-------|---------|
| 0 | 8 | magic | `b'RCPOU001'` (no null terminator) |
| 8 | 4 | `int32` | `NUM_BEAMS` = 16 — must equal `kernels.cuh`'s `kNumBeams` |
| 12 | 4 | `int32` | `AZIMUTH_BINS` = 1024 — must equal `kernels.cuh`'s `kAzimuthBins` |
| 16 | 4 | `int32` | `N_EDGE` — edge-cohort point count (39 in the committed sample) |
| 20 | 4 | `int32` | `N_GHOST` — ghost-duplicate count (200 in the committed sample) |
| 24 | 4 | `int32` | `N_ORGANIZED_VALID` — the Python generator's own independent tally of valid organized cells (12,361 in the committed sample); `main.cu` cross-checks this against both the GPU and CPU organized-to-unorganized counts as a third, differently-timed, differently-languaged tally |
| 28 | 4 | `int32` | reserved (0) |
| 32 | `NUM_BEAMS*AZIMUTH_BINS*3*4` = 196,608 | `float32[]` | the **organized grid**, ring-major (`cell = ring*AZIMUTH_BINS + azimuth`), meters, `lidar` sensor frame (x-forward/y-left/z-up). An invalid cell (geometric miss or the independent 5% absorption/glare dropout) stores three IEEE-754 NaN floats. |
| 32+196608 | `N_EDGE*3*4` = 468 | `float32[]` | the **edge cohort**: points placed at ±1e-4 (meters or "pixels", per boundary type) around every passthrough/box/frustum predicate threshold in `kernels.cuh`. NOT part of the organized grid — no beam direction produced these. |
| 32+196608+468 | `N_GHOST*8` = 1,600 | `(int32 cell_index, float32 range_offset_m)[]` | the **ghost table**: each entry names an already-VALID organized cell and a range offset (meters, may be negative); `main.cu` derives each ghost point's xyz at load time as `(that cell's unit direction) * (that cell's range + range_offset_m)` — a second echo along the same ray, feeding `GATE collision_accounting`. |

Total: `8 + 24 + 196608 + 468 + 1600 = 198708` bytes — matches the committed file size exactly.

### Fields / format — semantics

- **Units & frame:** meters, `lidar` sensor frame, right-handed, x-forward/y-left/z-up
  (SYSTEM_DESIGN.md §3.2). No orientation/timestamp fields — a single static revolution.
- **NaN convention:** `kernels.cuh`'s `is_invalid_point(x)` tests `x != x`; by construction every
  invalid organized cell has all three coordinates NaN together, so testing `x` alone is sufficient
  (documented once, relied on everywhere).
- **Dropout model:** every organized cell is invalid for one of two independent reasons — a
  geometric miss (the ray escapes the room; the scene has no ceiling, so many upward-tilted beams
  see open sky) or the independent 5% absorption/glare dropout applied to genuine geometric hits.
  Both end up as the same NaN sentinel (a real driver cannot distinguish them either) — THEORY.md
  "The problem" teaches the physics behind each.
- **Predicate bounds the edge cohort straddles:** `kPassthroughZMin/Max`, `kBoxMin/Max`,
  `kFrustumNearM` + the four image-edge planes derived from `kFx/kFy/kCx/kCy/kImgW/kImgH` — all
  defined once in `../src/kernels.cuh` and mirrored (by value, with a "must match" comment) in
  `make_synthetic.py`.

### A note on the azimuth-bin-center fix (an honest record)

The first generated sample cast rays at each azimuth bin's **lower edge**
(`az = 2*pi*az_step/AZIMUTH_BINS`). That placed every ray's angle exactly on the `floor()` decision
boundary `azimuth_bin_of()` reconstructs from, so a sub-ULP rounding difference between the
generation-time `cos`/`sin` and the reconstruction-time `atan2` round-trip flipped roughly **half**
of all reconstructed azimuth bins down by one — caught by this project's own `GATE roundtrip` during
development (see `scripts/make_synthetic.py`'s `build_organized_grid()` comment for the full story).
The fix — casting at the bin **center** (`az = 2*pi*(az_step+0.5)/AZIMUTH_BINS`) — gives a full
half-bin-width of floating-point margin and is committed in the current generator and sample.
