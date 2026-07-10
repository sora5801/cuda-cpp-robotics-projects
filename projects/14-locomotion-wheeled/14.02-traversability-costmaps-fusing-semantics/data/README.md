# Data — 14.02 Traversability costmaps fusing semantics + geometry

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
| Kind | **Synthetic** scenario RECIPE — not a recorded elevation/semantic map, and not the two 65,536-cell grids themselves (see below) |
| File | `sample/traversability_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (rock-placement RNG seed 42; every other feature is exact deterministic geometry/painting) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 2546 bytes (~2.5 KiB) |
| Checksum (SHA-256) | `a593c1f6c9e44319db9e38288a0ccdf8bfb76c1cdcd4b408d3b07c4a17cec762` |
| Regenerate with | `python ../scripts/make_synthetic.py --seed 42` — byte-identical (Python's Mersenne Twister is deterministic per-seed) |

### Why a recipe, not two committed grids

This project's real input is TWO co-registered 256x256 (65,536-cell) grids — a float elevation map and
a `uint8_t` class map + float confidence map — but committing those grids as files would cost
~330 KiB of mostly-redundant bytes for a demo this size. Instead, `sample/traversability_scenario.csv`
commits the small, human-readable RECIPE that `../src/main.cu`'s `build_elevation()` /
`build_semantics()` / `build_transect()` turn into the three arrays and the teaching-transect sample
list at start-up, every run, byte-for-byte identically (no run-time randomness at all — the recipe
already contains the one random choice, where the 10 rock-patch domes sit, as literal numbers). This is
the same "scenario, not recordings" pattern project 13.03 uses for its terrain recipe and 08.01 uses for
its cart-pole start state.

### Fields / format

Plain-text CSV; `#` lines are comments. Nine row types (loader: `load_scenario()` in
[`../src/main.cu`](../src/main.cu); grid constants and algorithm thresholds live in
[`../src/kernels.cuh`](../src/kernels.cuh), NOT in this file — CLAUDE.md §8's distinction between data
and the tuned, taught setup):

| Row | Fields (lengths in **meters**, angles in **degrees**) | Meaning |
|-----|--------------------------------------------------------|---------|
| `RIPPLE` | `amplitude_m, wavelength_m` | background rolling-terrain ripple, applied everywhere: `z += A*sin(2*pi*(x+y)/L)` — an exact-closed-form-gradient traveling wave, not real sensor noise |
| `BERM` | `x0,x1,y0,y1, angle_deg` | a linear ramp confined to the y-band `[y0,y1]`, rising from 0 at `x0` to `tan(angle)*(x1-x0)` at `x1`, then **plateauing** for `x>x1` (13.03's RAMP formula, reused) |
| `DITCH` | `xs,xb0,xb1,xe,y0,y1, depth_m` | a trapezoidal V-shaped depression confined to `[y0,y1]`: linear descent `[xs,xb0]` from 0 to `-depth_m`, flat bottom `[xb0,xb1]` at `-depth_m`, linear ascent `[xb1,xe]` back to 0 |
| `ROCK` | `cx,cy,h,r` | one smooth dome, `bump(d)=h*(1-(d/r)^2)^2` for `d<r` (else 0); 10 rows, seeded RNG placement |
| `VEGBUMP` | `x0,x1,y0,y1, amplitude_m,wavelength_m` | a SECOND, higher-frequency ripple ADDED on top of `RIPPLE`, confined to one rectangle — stands in for noisy LiDAR returns off a vegetation canopy (THEORY.md §The problem) |
| `SEMREGION` | `class_name,confidence, x0,x1,y0,y1` | paints one rectangle's semantic class + BASE confidence; applied IN ORDER, later rows override earlier ones on overlap; the first row covers the whole map as the default background; `class_name` is one of `DIRT`/`GRAVEL`/`GRASS`/`VEGETATION`/`WATER`/`UNKNOWN` |
| `CONFNOISE` | `amplitude,wavelength` | smooth deterministic per-cell confidence jitter added everywhere after painting, then clamped to `[0.05,0.99]`: `conf += amplitude*sin(2*pi*x/L)*cos(2*pi*y/L)` |
| `WAYPOINT` | `label,x,y` | one point on the teaching TRANSECT polyline `demo/out/layers.csv` samples; consecutive waypoints are connected by straight legs, 60 samples/leg |

The six named terrain features sit in six non-overlapping y-bands across the 25.6x25.6 m map (a
`>=0.3 m` gap between every pair, wider than the geometric kernel's widest window, so no analytic gate
in `main.cu` ever sees two DIFFERENT features blended together): **Band 1** `y in [0.5,4.3)` control
(background ripple only), **Band 2** `y in [4.6,8.4)` berm (18 deg, `x in [9.0,10.0]`), **Band 3**
`y in [8.7,12.5)` ditch (0.5 m deep, 45 deg walls), **Band 4** `y in [12.8,16.6)` rock patch (10 domes,
`x in [1,24]`), **Band 5** `y in [16.9,20.7)` water (pool `x in [18,20]`, grass elsewhere in-band),
**Band 6** `y in [21.0,24.8)` vegetation (high-frequency canopy bump, `x in [10,16]`). The six
`SEMREGION` rows (after the whole-map `DIRT` default) paint, in order: `GRAVEL` over the berm+ditch
bands, `GRAVEL` over the rock-patch band, a small `UNKNOWN` flavor patch in the control band, `GRASS`
in part of the water band, `WATER` over the pool, and `VEGETATION` over the canopy-bump patch — using
all six palette classes at least once. `scripts/make_synthetic.py`'s file header has the full layout
rationale.

The loader is strict: unknown row labels, short rows, an unrecognized `class_name`, or a missing
`RIPPLE`/`BERM`/`DITCH`/`ROCK`/`VEGBUMP`/`SEMREGION`/`WAYPOINT` section aborts the demo.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
