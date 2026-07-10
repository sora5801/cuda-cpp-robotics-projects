# Data — 13.03 Foothold scoring kernels: slope, roughness, edge distance from elevation maps

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
| Kind | **Synthetic** terrain RECIPE — not a recorded elevation map, and not the map grid itself (see below) |
| File | `sample/terrain_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (rock-placement RNG seed 42; every other feature is exact deterministic geometry) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | ~1.9 KiB |
| Checksum (SHA-256) | `4b34a030e9044ca840e1773350568b738e8dd7a6e29ed7fa55064e58fe9041c8` |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` — byte-identical (Python's Mersenne Twister is deterministic per-seed) |

### Why a recipe, not a committed grid

This project's real input is a 256x256 (65536-cell) elevation map — but committing that grid as a
file would cost ~256 KiB of mostly-redundant floats for a demo this size, and grows quadratically for
anything bigger. Instead, `sample/terrain_scenario.csv` commits the small, human-readable RECIPE that
`../src/main.cu`'s `build_terrain()` turns into the grid at start-up, every run, byte-for-byte
identically (no run-time randomness at all — the recipe already contains the one random choice, where
the 16 rocks sit, as literal numbers). This is the same "scenario, not recordings" pattern project
08.01 uses for its cart-pole start state.

### Fields / format

Plain-text CSV; `#` lines are comments. Six row types (loader: `load_terrain_scenario()` in
[`../src/main.cu`](../src/main.cu); grid constants and algorithm thresholds live in
[`../src/kernels.cuh`](../src/kernels.cuh), NOT in this file — CLAUDE.md §8's distinction between data
and the tuned, taught setup):

| Row | Fields (all lengths in **meters**, angles in **degrees**) | Meaning |
|-----|------------------------------------------------------------|---------|
| `RIPPLE` | `amplitude_m, wavelength_m` | background sensor-noise-like ripple, applied everywhere: `z += amplitude*sin(2*pi*x/wavelength)*cos(2*pi*y/wavelength)` |
| `RAMP` | `x0,x1,y0,y1, angle_deg` | a linear ramp confined to the y-band `[y0,y1]`, rising from 0 at `x0` to `tan(angle)*(x1-x0)` at `x1`, then **plateauing** (not reverting) for `x>x1` |
| `STEP` | `x0,x1,y0,y1, edge_x,height_m` | a vertical step confined to the y-band `[y0,y1]`: 0 for `x<edge_x`, `height_m` (persisting) for `x>=edge_x`; `x0,x1` are the documented region of interest, not part of the height formula |
| `ROCK` | `cx,cy,h,r` | one smooth dome, `bump(d)=h*(1-(d/r)^2)^2` for `d<r` (else 0); 16 rows |
| `HOLE` | `x0,x1,y0,y1` | a rectangular NaN block (sensor dropout), applied LAST — overrides every other contribution |
| `PATH` | `x0,y0,x1,y1, n` | one straight foothold-query segment: `n` queries linearly interpolated from `(x0,y0)` to `(x1,y1)`; 5 rows, 200 points each = 1000 queries total |

The five named features sit in five non-overlapping y-bands across the 5.12x5.12 m map (a `>=0.20 m`
gap between every pair, wider than the plane-fit window plus the edge-distance search reach, so no
analytic gate ever sees two features blended together): **Band A** `y in [0.20,0.80]` flat control,
**Band B** `y in [1.00,1.60]` ramp, **Band C** `y in [1.90,2.50]` step, **Band D** `y in [2.90,4.00]`
rock field, **Band E** `y in [4.30,4.70]` hole vicinity. `scripts/make_synthetic.py`'s file header has
the full layout rationale.

The loader is strict: unknown labels, short rows, or a missing `RAMP`/`STEP`/`HOLE`/`ROCK`/`PATH`
section abort the demo.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
