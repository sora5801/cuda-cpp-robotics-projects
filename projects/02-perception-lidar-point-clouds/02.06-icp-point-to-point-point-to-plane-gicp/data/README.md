# Data — 02.06 ICP: point-to-point → point-to-plane → GICP, all batched

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

**Synthetic by necessity, not just by repo default.** ICP needs a KNOWN ground-truth transform to be
verifiable at all — a recorded LiDAR scan never comes with an answer key. `../scripts/make_synthetic.py`
builds a small structured "room" (a floor and two walls meeting at a corner, plus a box — deliberately
**wall-dominated** so point-to-plane ICP's faster convergence on planar scenes, the property this project
teaches, is actually exercised), samples two independent point clouds off its surfaces (mimicking two
different LiDAR scans of the same static scene), and applies a **known** rigid transform plus independent
Gaussian sensor noise to the second cloud. See [`../THEORY.md`](../THEORY.md) "The problem" for why
`NOISE_SIGMA_M = 0.005` m is a physically reasonable range-noise magnitude.

| Property | Value |
|----------|-------|
| Kind | Synthetic (repo default, CLAUDE.md §8) — **and structurally required here**, see above |
| Generator | `python ../scripts/make_synthetic.py --seed 42` (the default; deterministic, stdlib-only) |
| License | Synthetic — repo MIT license applies; no external data, no redistribution concerns |
| Size (committed) | 5 files, ~836 KiB total (well under the 2 MiB budget and the repo's 50 MB ceiling) |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` (byte-identical output — verify against the checksums below) |

### Committed files and checksums

Regenerate and diff against these SHA-256 hashes to confirm byte-for-byte reproducibility (measured on
the files actually committed to this repo):

| File | Size (bytes) | SHA-256 |
|------|--------------:|---------|
| `sample/pair0_source.bin` | 360008 | `ce5b6b1471c15d7d188a7522ad49236275cc38e52ee44965bf003f13d3e4b14b` |
| `sample/pair0_target.bin` | 360008 | `83ecc9da77f1aabb40daf512e36f1fcea26171430a34a77d237703f2165384de` |
| `sample/pair1_source.bin` |  60008 | `b0c929907c29ded23c4d4c66c5a63338ce0ca4deadadbc0c28c7e14c6acca2ae` |
| `sample/pair1_target.bin` |  60008 | `875c71c2c03f21a057853130cdaa3bcf0c1898a3d2b0a43d4f4d3984fc89805c` |
| `sample/pairs_meta.csv`   |    690 | `bc9f09894d889043d799feb6394915f8037c6b246c46d863860aea3ba33df474` |

### Fields / format

**Point-cloud binary format** (`pair*_source.bin`, `pair*_target.bin`) — a tiny custom binary chosen
over CSV (as CLAUDE.md §8's "CSV/binary" allows) specifically to demonstrate binary point-cloud I/O, the
format real LiDAR drivers use. Byte-exact layout, little-endian throughout:

```
offset  size        field
------  ----------  --------------------------------------------------------
0       4 bytes     ASCII magic "PC01" (format id + version 1)
4       4 bytes     uint32 little-endian point count N
8       N * 12 bytes  N * (float32 x, float32 y, float32 z), little-endian, meters
```

Coordinates are in the CLOUD'S OWN frame: `*_source.bin` is the CANONICAL (untransformed) scene frame;
`*_target.bin` is already in the "scanned" frame — i.e. the canonical scene sampled independently, then
rotated/translated by the pair's ground-truth transform (below), then given fresh sensor noise. Applying
that ground-truth transform to the source cloud is what should land it on the target cloud (ICP's job).
`main.cu`'s `load_cloud_bin()` reads this format; `scripts/make_synthetic.py`'s `write_cloud_bin()`
writes it — the two are the format's single source of truth alongside this document.

**`pairs_meta.csv`** — one `PAIR` row per demo pair, comma-separated, `#`-prefixed comment header (same
strict-loader discipline as 08.01's scenario file: unknown labels or short rows fail loudly, never
silently). Columns, in order:

| Column | Type | Meaning |
|--------|------|---------|
| `PAIR` | literal | row-type label (only row type this file contains) |
| `name` | string | pair identifier (`pair0`, `pair1`) — also the `.bin` filename prefix |
| `source_file`, `target_file` | string | filenames of the two clouds, relative to this directory |
| `n_source`, `n_target` | int | point counts — cross-checked against each `.bin` file's own header at load time |
| `qw,qx,qy,qz` | float | **ground-truth** unit quaternion (w,x,y,z) of `T_target_source` (CLAUDE.md §12 order) |
| `tx_m,ty_m,tz_m` | float | **ground-truth** translation (m) of `T_target_source` |
| `noise_sigma_m` | float | the isotropic Gaussian position-noise sigma used for BOTH clouds (m) — see THEORY.md |

`T_target_source` (09.01's naming convention: "source cloud's frame, expressed in the target cloud's
frame") is the transform that, applied to the source cloud, best aligns it with the target cloud — the
value ICP is trying to recover. `main.cu`'s ground-truth gate compares its recovered estimate against
these exact numbers.

### The two pairs

| Pair | Points/cloud | Rotation | \|translation\| | Purpose |
|------|--------------:|----------|-----------------:|---------|
| `pair0` | 30000 | 7.0° about +z (pure yaw) | 0.314 m | **Main pair** — the wall-dominated scene the point-to-plane-vs-point-to-point iteration-count comparison runs on. |
| `pair1` | 5000  | 9.0° about a tilted axis (0.2, 0.1, 1.0) | 0.381 m | **Small second pair** — proves the pipeline is not accidentally specialized to pure yaw, at 1/6th the point budget. |

Both ground-truth rotations and translations fall inside the brief's 5–10° / 0.2–0.4 m ranges. Both
clouds in both pairs carry independent `NOISE_SIGMA_M = 0.005` m Gaussian position noise (THEORY.md "The
problem" explains why this magnitude is a reasonable stand-in for LiDAR range noise).
