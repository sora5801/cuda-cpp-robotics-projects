# Data — 02.11 Scan Context / ring-descriptor loop-closure search

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
| Kind | 100% synthetic — a hand-authored trajectory through a procedurally-varied synthetic "town" |
| Generator | `python ../scripts/make_synthetic.py --seed 42` |
| License | Synthetic — this repository's MIT license applies (no external data, no external license) |
| Size (committed) | `world.csv` 592 B · `trajectory.csv` 7.3 KB · `loop_pairs.csv` 965 B · `scans.bin` 2.47 MB — 2.5 MB total |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42 --target-keyframes 120` (yields 160 keyframes at the odd-`n` anchor-symmetry fix — see the "why 160, not ~120" note below) |

**Why no public dataset.** See `../scripts/download_data.ps1`'s header for the full reasoning: Scan
Context loop-closure evaluation needs LABELED REVISIT GROUND TRUTH (which earlier keyframe is the same
physical place, from which relative heading, with which lateral offset — and which keyframes are
genuinely new places that must NOT match anything). No small, cleanly-licensed public LiDAR dataset ships
that labeling out of the box; the well-known loop-closure benchmarks that do (KITTI odometry sequences
with community-contributed loop pairs, NCLT) are large, non-commercially licensed, and still require the
same kind of hand curation this project's synthetic world provides directly, with the added benefit of
INDEPENDENT COHORTS (same-heading / rotated / laterally-offset / negative) chosen by design rather than
mined after the fact.

### Checksums (SHA-256, computed on the committed files)

```
2516525018c5d8d1b070a9d2a920de47786ef97081eec75436428380df9ccddc  world.csv
5a07371c41d3fa1b679cd1c90491b5257710e07eafa84b695b10952736d29861  trajectory.csv
543b18ead78fa96bde67d75b54a767fffab5862f9e7d94cfc6bbaa513356c2b5  loop_pairs.csv
de0aba72f7d207e116ac3160bc035a6fffdad3d63e718f233604294a7b3f186e  scans.bin
```

Verify with `sha256sum -c` (Linux/macOS/Git-Bash) or `Get-FileHash -Algorithm SHA256` (PowerShell).

### The world: a 4x3-station "town"

`scripts/make_synthetic.py` lays out 12 street intersections ("stations") on a 25 m grid (4 columns x 3
rows) and fills each of the 6 interior blocks with 1–2 buildings whose footprint/height are drawn from a
seeded xorshift32 stream — even-indexed blocks get one large building, odd-indexed blocks get two smaller
ones side by side, so neighboring blocks look structurally different (the raw material Scan Context needs
to tell places apart — THEORY.md "the aliasing problem"). A hand-authored ROUTE (documented in full in
`make_synthetic.py`'s module docstring and the `ROUTE` list itself) walks this station graph in six
labeled phases that deliberately produce, by construction:

- **8 same-heading revisits** — the SAME street, same direction, much later in the trajectory.
- **4 rotated revisits** — the SAME street, ~180° reversed heading (the rotation-invariance showcase).
- **4 lateral-offset revisits** — the SAME street, same heading, shifted sideways by 0.6 m, 1.3 m, or
  2.6 m (two magnitudes tested twice) — the translation-sensitivity honesty cohort.
- **8 genuinely new places** — station-graph edges visited exactly once, never revisited (the negative
  cohort a correct system must never "recognize").

**Why 160 keyframes, not ~120.** Each route segment is sampled into an ODD number of keyframes so its
midpoint keyframe (the "anchor" used for curated pairs) lands at exactly `t=0.5` regardless of which
direction the segment is traversed — required so a forward pass and a reversed pass of the same physical
edge share the identical anchor POSITION (see `make_synthetic.py`'s `sample_keyframes()` docstring for the
~6 m position-mismatch bug this fix resolves, and THEORY.md's "numerical considerations" for the full
story). Forcing every segment's sample count to the next odd integer rounds `target-keyframes=120` up to
160 keyframes across the 32 segments — the honest, measured count, not a target hit exactly.

### `world.csv` — building footprints (world frame, meters)

| Column | Type | Meaning |
|---|---|---|
| `x0_m,y0_m,x1_m,y1_m` | float | Axis-aligned footprint corners, world frame, meters |
| `height_m` | float | Building height above ground, meters |

9 buildings total (6 blocks: 3 single-building, 3 double-building).

### `trajectory.csv` — one row per keyframe (world frame, meters/radians)

| Column | Type | Meaning |
|---|---|---|
| `idx` | int | Keyframe index, 0..159 — the SAME index scans.bin and loop_pairs.csv use |
| `x_m,y_m` | float | Sensor position, world frame, meters |
| `heading_rad` | float | Sensor forward-direction yaw, world frame, radians, right-handed (0 = +x) |
| `seg_index` | int | Which of the 32 ROUTE segments this keyframe belongs to |
| `from_id,to_id` | int | The route segment's station endpoints (station id = `row*4+col`) |
| `offset_m` | float | The segment's authored lateral offset, meters (0 unless a phase-5 offset pass) |
| `is_anchor` | 0/1 | 1 iff this is the segment's t=0.5 midpoint keyframe (the one loop_pairs.csv references) |

### `loop_pairs.csv` — curated ground-truth revisit pairs

| Column | Type | Meaning |
|---|---|---|
| `query_idx,match_idx` | int | Keyframe indices; query is always LATER in time than match |
| `cohort` | string | `same_heading` \| `rotated` \| `lateral_offset` |
| `relative_yaw_true_deg` | float | True `heading[query] - heading[match]`, wrapped to (-180,180] |
| `lateral_offset_m` | float | True perpendicular position difference between the two anchors, meters |

A keyframe that is an anchor (`trajectory.csv`'s `is_anchor=1`) but appears in NEITHER column of this
file is a **negative-cohort** example — a genuinely new place with no true revisit anywhere in its valid
(temporally-gapped) candidate range. `main.cu` derives the negative set this way rather than from a
separate file.

### `scans.bin` — every keyframe's point cloud, SENSOR FRAME, meters

Binary, little-endian, no padding:

```
offset 0   : char[8]      magic = "SCANCTX1"
offset 8   : int32        num_scans (= 160)
then, for scan s = 0 .. num_scans-1, back to back:
    int32        n_points_s
    float32[3] x n_points_s   interleaved (x,y,z), SENSOR FRAME, meters
```

Points are in the SENSOR's own frame at the moment of that keyframe (origin at the sensor, +x forward
along `heading_rad`, +y left, +z up — the sensor sits `SENSOR_HEIGHT_M=1.6` m above the ground it rides
on, so a ground return reads `z≈-1.6`, never `z≈0`) — this is deliberately the EGOCENTRIC frame Scan
Context bins directly (no pose lookup needed at read time). Average 1351 points/scan (min–max varies with
how much building wall is in view). Generated by ray-casting a 16-channel, 120-azimuth-step LiDAR model
(elevation -18°..+12°, max range 40 m) against the ground plane and every building's vertical walls —
`scripts/make_synthetic.py`'s `simulate_scan()` documents the exact model and its simplifications
(rooftops are not modeled — a documented limitation, see `../README.md` "Limitations & honesty").
