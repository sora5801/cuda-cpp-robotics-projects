# Data — 02.12 Range-image conversion + depth-clustering segmentation

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
| Kind | Synthetic (default, CLAUDE.md §8) — a ray-cast 16-beam LiDAR scan with exact per-point ground truth |
| Generator / source | `../scripts/make_synthetic.py` (xorshift32 PRNG, fixed seed 42; stdlib-only, no numpy) |
| License | Synthetic — repo MIT license applies; no external data, no redistribution concerns |
| Size (committed) | `data/sample/range_image_scene.bin` — 222,856 bytes (217.6 KiB) |
| Checksum | SHA-256 `7603c7ff59573ad20d76cf6cbc0a3c28c2e833acdf5176076595c9c61b5f419` |
| Regenerate with | `python scripts/make_synthetic.py` (no arguments — the seed and every scene constant are fixed in the script) |

### What the scene is

A synthetic 16-beam spinning LiDAR (the same beam table as 02.01/02.02, `-15..+15` degrees in 2-degree
steps, 1024 azimuth bins) stands at the origin, `SENSOR_HEIGHT_M = 1.5` m above a flat ground plane, and
sees six objects placed in four non-interacting azimuth sectors — see `../scripts/make_synthetic.py`'s
module docstring for the full sector map and the exact geometry (and *why* each number is what it is —
several were re-derived after measuring an earlier attempt's actual nearest-point distances, not merely
assumed). In one sentence each:

- **person** (truth id 1) + **wall_behind** (id 2) — a narrow person-sized box standing directly in
  front of a wide, flat panel. The MEASURED nearest visible-point gap between them (brute-force
  nearest-neighbor search over the generated points, accounting for the person's own occlusion shadow)
  is **0.19 m** — smaller than `EUCLID_TOLERANCE_M` (0.40 m) but the objects are still two DIFFERENT
  physical surfaces at different ranges. The depth-gap showcase (README "The comparison").
- **big_box** (id 3) + **far_pole** (id 4) — a large near obstacle with a small, distant object peeking
  out one column beside its angular footprint. The large-near/small-far occlusion-boundary pair.
- **thin_pole** (id 5) — an isolated 0.10 m x 0.10 m post, alone in its own sector. The min-cluster-size
  vs. real-thin-object trade.
- **grazing_wall** (id 6) — a 13 m long flat panel running almost along the sensor's own line of sight
  (viewed at a shallow, grazing incidence that gets more extreme toward its far end). The beta
  criterion's known weakness, MEASURED to fragment into 13 depth clusters of size >= 5 (see
  `demo/out/gates_metrics.csv` after any run).

Ground is a flat, infinite plane at `z = -1.5 m`; any ray that hits neither an object nor the ground
within `MAX_RANGE_M = 18 m` legitimately returns nothing (open sky), exactly as a real LiDAR reports no
return when nothing is in range. Range values include a small (`sigma = 3 mm`) Gaussian range-axis
noise, sized (THEORY.md "Numerical considerations" shows the arithmetic) to stay well clear of flipping
the ground-removal angle test's decision on this scene's geometry.

### Fields / format (RIMAGE01)

A flat list of the scan's VALID returns only, each already carrying its native (ring, azimuth-bin) —
exactly what a real spinning-LiDAR driver reports per packet (see `../PRACTICE.md` section 1). Nothing
is pre-organized into the range-image grid; that conversion is the GPU's own first pipeline stage
(`src/kernels.cu`'s `scatter_encode_kernel` / `finalize_organized_kernel`), exercised and verified
against this file, not baked into it.

All multi-byte fields little-endian (`<` struct format), matching the host machine's native byte order.

**Header** (60 bytes):

| Offset | Bytes | Field | Type | Meaning |
|--------|-------|-------|------|---------|
| 0 | 8 | magic | `char[8]` | literal `"RIMAGE01"` |
| 8 | 4 | `n_points` | `int32` | number of point records that follow |
| 12 | 4 | `num_beams` | `int32` | 16 — must match `kernels.cuh`'s `kNumBeams` |
| 16 | 4 | `azimuth_bins` | `int32` | 1024 — must match `kernels.cuh`'s `kAzimuthBins` |
| 20 | 4 | `sensor_height_m` | `float32` | LiDAR mount height above ground, meters |
| 24 | 4 | `ground_angle_threshold_deg` | `float32` | ground-removal flatness threshold, degrees |
| 28 | 4 | `beta_threshold_deg` | `float32` | depth-clustering beta-criterion threshold, degrees |
| 32 | 4 | `euclid_tolerance_m` | `float32` | Euclidean-comparison cluster tolerance `d`, meters |
| 36 | 4 | `min_cluster_size_depth` | `int32` | depth-cluster min-size noise floor |
| 40 | 4 | `min_cluster_size_euclid` | `int32` | Euclidean-cluster min-size noise floor |
| 44 | 4 | `truth_num_objects` | `int32` | 6 — the number of NAMED (non-ground) truth objects |
| 48 | 12 | reserved | `int32[3]` | zero-filled, reserved for future header growth |

`src/main.cu`'s `load_scene()` asserts every one of these (except `n_points`/reserved) against the
compiled `kernels.cuh` constants and fails loudly on any mismatch (02.01's data/code consistency
discipline) — the sample and the compiled pipeline were designed around each other.

**Point records** (28 bytes each, `n_points` of them, immediately after the header):

| Offset | Bytes | Field | Type | Meaning |
|--------|-------|-------|------|---------|
| +0 | 4 | `x` | `float32` | meters, LiDAR sensor frame (+x forward) |
| +4 | 4 | `y` | `float32` | meters, sensor frame (+y left) |
| +8 | 4 | `z` | `float32` | meters, sensor frame (+z up) |
| +12 | 4 | `range_m` | `float32` | meters; the noisy range this point was cast at (`x/y/z` are `range_m` times the ray's unit direction) |
| +16 | 4 | `ring` | `int32` | beam index, `0..15` (ring 0 = most negative elevation, -15 deg) |
| +20 | 4 | `az_bin` | `int32` | azimuth-bin index, `0..1023` |
| +24 | 4 | `truth_id` | `int32` | `0` = ground, `1..6` = the named objects above, `-2` = a synthetic COLLISION-TEST phantom point (see below) — never `-1` (that sentinel is reserved for "empty organized-grid cell", which only exists after GPU/CPU conversion, not in this raw point list) |

Total file size = `60 + n_points * 28` bytes = `60 + 7957 * 28` = **222,856 bytes**, matching the
checksum above exactly.

**The two collision-test phantoms** (`truth_id == -2`, the last two records in the file): every genuine
ray-cast point already owns a unique `(ring, az_bin)` cell, so the organized-grid scatter never
naturally collides on this scene. To exercise (and gate, via `GATE collision_resolution` in
`src/main.cu`) the nearest-wins `atomicMin` collision machinery honestly, the generator appends two
extra points that deliberately TARGET an already-used cell with a range 5 m FARTHER than the real
point there — a stand-in for a multipath/ghost return. A correct scatter must never let one win.
