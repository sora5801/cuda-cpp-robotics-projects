# Data — 23.01 GPU costmaps: inflation, raytrace clearing, multi-layer fusion

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more — idempotent, documented, license-respecting. **This project needs none**: a costmap
  pipeline needs an occupancy grid and a start/goal pair, both of which are exactly the kind of thing
  this repo can synthesize with full ground truth (and, unlike a real building, VERIFY solvable
  before ever writing the file — see the generator's own printed check).
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** (100% generated) — a 2-D occupancy grid + a navigation scenario |
| Files | `sample/world_map.pgm`, `sample/scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: seed 42, 256x256 grid @ 0.05 m/cell, step cap 500) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | `world_map.pgm` ~64.0 KiB, `scenario.csv` ~0.3 KiB |
| Checksum (SHA-256) | `world_map.pgm`: `1d00b83c1c9e96f079e124fdd513374bd776e27aaf320dd0bc084262f886f86e` <br> `scenario.csv`: `c9074ab6e7f83e4a11f5767bba263718065f13a6fb6fac84a53ce8c75a652889` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical for the default seed 42 |

Measured, not asserted: the committed map has 6171 lethal cells (9.4% occupied) and the generator's
own breadth-first search confirms a start-to-goal path exists at the robot's inscribed-radius
clearance before the file is ever written (its printed `BFS solvability check PASSED` line is the
receipt — see `../scripts/make_synthetic.py`).

### Fields / format

**`world_map.pgm`** — a binary P5 PGM (the smallest real image format there is; 07.09 and this
project's own `demo/out/costmap.pgm` artifact use the identical format, so one PGM reader/viewer
serves the whole repo). Three-line ASCII header (`P5`, `<width> <height>`, `255`), then exactly
`width*height` raw bytes, row-major, `(x,y)` at index `y*width+x` (x rightward, y downward — image
convention; the same layout `../src/kernels.cuh` documents for every device-side layer buffer).

| Byte value | Meaning |
|---|---|
| `0` | free (traversable) |
| `254` | lethal (an obstacle cell — the map's border walls plus four interior pillars) |

No other byte values appear in the committed map (the generator only ever draws these two; the
inflation gradient the demo computes at runtime is never part of the STATIC map file). Units: cells
— 1 cell = 0.05 m (`kResolutionM` in `../src/kernels.cuh`); the whole grid is 256x256 cells = 12.8 m
x 12.8 m.

**`scenario.csv`** — plain-text CSV, `#` lines are comments, one labeled row per field:

```
MAP,<filename>,<width_cells>,<height_cells>,<resolution_m>
START,<x_m>,<y_m>,<theta_rad>
GOAL,<x_m>,<y_m>
STEPS,<control_tick_cap>
```

| Field | Type / range | Meaning |
|-------|--------------|---------|
| `MAP` filename | string | The occupancy-grid file, resolved relative to `scenario.csv`'s own directory |
| `MAP` width/height/resolution | int, int, float | Must match `kGridW`/`kGridH`/`kResolutionM` in `../src/kernels.cuh` exactly — the loader in `../src/main.cu` refuses a mismatch |
| `START` x, y, theta | float (m), float (m), float (rad, `(-pi,pi]`) | The robot's starting pose, world frame |
| `GOAL` x, y | float (m), float (m) | The goal position (no goal heading — DWA scores position + heading-toward-goal, not a fixed final orientation) |
| `STEPS` | int, `>= 1` | The closed loop's control-tick cap (at `kDtControl = 0.1 s`, 500 steps = 50 s of simulated time) |

The loader is strict, matching every other project's convention: an unrecognized row label, a short
row, or a `MAP` row whose geometry disagrees with the compiled-in contract all abort the demo with a
clear message — corrupt or stale data can never quietly pass.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
