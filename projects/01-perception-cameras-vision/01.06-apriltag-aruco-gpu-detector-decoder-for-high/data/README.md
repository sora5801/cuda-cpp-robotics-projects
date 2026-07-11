# Data — 01.06 AprilTag / ArUco GPU detector-decoder for high-rate fiducial localization

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
| Kind | Synthetic (default per CLAUDE.md §8) — a home-grown fiducial dictionary plus three rendered scenes, all with dense, exact ground truth |
| Generator / source | `../scripts/make_synthetic.py` (fixed seed 42, xorshift32, stdlib-only Python; no CLI args for the committed sample) |
| License | Synthetic — repo MIT license applies; no third-party data or bit tables involved (the dictionary is independently generated, never AprilTag's or ArUco's published codes — see `THEORY.md` "Where this sits in the real world") |
| Size (committed) | 7 files, 521,337 bytes total (~509 KiB) — well under the 50 MB ceiling |
| Checksum (SHA-256) | see table below |
| Regenerate with | `python ../scripts/make_synthetic.py` (from `scripts/`; writes into `sample/`; ~45 s, dominated by the dictionary's greedy Hamming-distance search) |

| File | Bytes | SHA-256 |
|------|-------|---------|
| `sample/dictionary.bin` | 84 | `7f65713dbd7f7129371ceccbe816682e1fc39396e2e6b3c2971e7643f8e42ec4` |
| `sample/dictionary.csv` | 920 | `481e958cbbb6dd53dc406d6bf8463879c94f4b35b6e7e02f1d45d79ec1e8c883` |
| `sample/scene_main.pgm` | 172,815 | `394d06537fdf24a4774b7117825772768f05c44e188d64189cd0fd5f8404eba5` |
| `sample/scene_main_ground_truth.csv` | 1,415 | `71316b0fd53b28c36a96364b51a0eb19ea32544e863d82af45e0939f2fd47277` |
| `sample/scene_distractor.pgm` | 172,815 | `7ef4b7a26d0aa52199766261f6a95be0d8549dfa61402ea6a20f4dbfdc2d3ddb` |
| `sample/scene_robustness.pgm` | 172,815 | `160b436120551614fe7d43ba1ae52baaa165bf6b4d16e5271d6b5aa779f74578` |
| `sample/scene_robustness_ground_truth.csv` | 473 | `dd01474d5d362b41e82a1c4707b4f68198daa4f44d3c953c2d6d57f3f5004fb2` |

### Fields / format

All three `.pgm` scenes are `kFullW x kFullH` = 480x360 px, 8-bit grayscale (PGM P5) — the single
source of truth every file below cross-references is `../src/kernels.cuh`'s `kFullW`/`kFullH`/`kFx`/
`kFy`/`kCx`/`kCy`/`kTagSizeM`/`kGridN` constants, each mirrored in `make_synthetic.py` with a "MUST
MATCH" comment. Pixel coordinates follow this project's stated camera-optical frame exception
(`kernels.cuh`'s file header: z-forward, x-right, y-down, row 0 at the top).

- **`dictionary.bin`** — the 32-code fiducial dictionary (binary, little-endian): 5x `int32` header
  `(num_codes=32, bits_per_code=16, grid_n=6, min_distance=5, correction_capacity=2)` followed by
  `num_codes` x `uint16` codes (each a 16-bit 4x4 payload, row-major, `payload_bit_index(r,c) =
  (r-1)*4+(c-1)` for grid coordinates `r,c` in `[1,4]` — see `kernels.cuh`). **Measured**, not assumed:
  the search targeted distance 6 and backed off automatically to the largest achievable distance —
  achieved minimum Hamming distance 5 (matching real family AprilTag 16h5's own distance, though this
  project's 32 codes are independently generated, never AprilTag's bit table), giving correction
  capacity `floor((5-1)/2) = 2` bits.
- **`dictionary.csv`** — the same 32 codes, human-readable (`index, code_hex, code_binary_16bit`).
- **`scene_main.pgm`** — 6 tags rendered under FULL PERSPECTIVE (independently randomized depth
  0.60-1.00 m, yaw/pitch tilt up to +/-25 deg, in-plane roll up to +/-38 deg — deliberately kept away
  from the +/-45 deg band where this project's extreme-corner quad extraction degrades, see
  `kernels.cuh`'s file header), plus an illumination gradient (+/-18%), a 5-tap Gaussian blur, and
  additive sensor noise (sigma=4 intensity levels). `scene_main_ground_truth.csv` records, per tag:
  `tag_index, dict_id`, the 4 tag-model corners' PIXEL positions (`corner{0..3}_{x,y}`, in the fixed
  TL/TR/BR/BL model order — see `kernels.cuh`'s `Homography`/`Detection` doc comments), the 3x3
  rotation matrix `R{00..22}` (tag axes in the camera frame, row-major) and translation `t{0,1,2}`
  (meters, camera frame) — the exact `(R, t)` used to render the tag, independently re-derived in
  `make_synthetic.py`'s own Python projection code (never calling into `kernels.cuh`/`kernels.cu`) —
  this is the analytic ground truth `main.cu`'s corner-accuracy and pose gates check against.
- **`scene_distractor.pgm`** — **no tags**: an 8x8 checkerboard block (18 px squares — smaller than
  `kMinBBoxSidePx`, filtered by size alone) plus 5 filled disks (3 small, sub-threshold; 2 large,
  sized to PASS the size/fill-ratio filters and reach the decoder, where the degenerate all-black-
  payload safeguard rejects them — see `kernels.cuh`'s `popcount16` doc comment). No ground-truth CSV
  — the only claim this scene makes is "zero accepted detections" (`main.cu`'s `false_positive` gate).
- **`scene_robustness.pgm`** — 4 tags, front-parallel-ish (roll only, up to +/-15 deg, fixed depth
  0.65 m), each rendered with a deliberately CORRUPTED payload: 2 tags at exactly
  `correction_capacity` (2) flipped bits (must still decode to the true ID — guaranteed by the
  dictionary's own minimum-distance construction, any bit choice works), 2 tags at
  `correction_capacity + 1` (3) flipped bits, with the specific flipped bits chosen (
  `find_isolated_flip_mask()` in `make_synthetic.py`) so the corrupted pattern is ALSO farther than
  `correction_capacity` from every OTHER dictionary code — otherwise a beyond-capacity corruption can
  (correctly, per coding theory) land within another code's correction ball by chance, which would
  demonstrate mis-decoding rather than this gate's intended lesson (clean rejection). Ground truth:
  `scene_robustness_ground_truth.csv` records `tag_index, true_dict_id, num_flips, expected_outcome`
  (`accept`/`reject`) plus the 4 corner pixel positions (same format as `scene_main`, used only for
  matching a detection to its ground-truth row by position).

### Camera & dictionary geometry (authoritative numbers; MUST match `../src/kernels.cuh`)

`fx=fy=350 px`, `cx=239.5, cy=179.5 px` (exact image center), tag physical size `0.16 m` (outer
border-to-border), 6x6 total grid (1-cell border ring + 4x4 = 16-bit payload). See `kernels.cuh`'s
file header for the full camera-model and dictionary-geometry derivation and `THEORY.md` for the
physics and coding-theory behind each choice.
