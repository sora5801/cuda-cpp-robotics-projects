# Data — 31.01 Hamilton-Jacobi reachability: level-set grid solvers (stencil ops — GPU-perfect)

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

A reachability solver's "dataset" is a **problem definition**, not recordings — the state-space
grid, the acceleration bound, the target level, and the horizon. Everything else (the initial level
function, every PDE sweep, the analytic oracle) is computed *inside* the demo from that scenario
plus closed-form mathematics (`min_time_to_origin` in `src/reference_cpu.cpp`); correctness comes
from the CPU twin plus that closed form, never from stored ground truth. No public dataset applies
— `scripts/download_data.ps1`/`.sh` are honest no-ops (see the `DECISION` comment at the top of
each).

| Property | Value |
|----------|-------|
| Kind | **Synthetic** scenario (a problem definition — no RNG involved; a scenario is constants) |
| File | `sample/double_integrator_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults match the committed file exactly) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 819 bytes |
| Checksum (SHA-256) | `c67e5ef90b0e6a7657cdd7abe768a3c2d2736330a2ef898a56598bb25f717052` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness) |

### Fields / format

Plain-text CSV; `#` lines are comments/provenance. Six required rows (loader: `load_scenario()` in
[`../src/main.cu`](../src/main.cu); layout authority: [`../src/kernels.cuh`](../src/kernels.cuh)),
order-free:

| Row | Fields | Units, frame | Meaning |
|-----|--------|---------------|---------|
| `GRID,nx,nv` | cell counts | dimensionless, `nx` >= 16, `nv` >= 16 | state-space grid cells; `x` (position) is the fast/contiguous axis, `v` (velocity) the slow axis (§12 layout convention — see `kernels.cuh`) |
| `XDOM,xmin,xmax` | position domain | m | node-centered: cell `i=0` sits ON `xmin`, cell `nx-1` ON `xmax` |
| `VDOM,vmin,vmax` | velocity domain | m/s | same node-centered convention along `v` |
| `UMAX,u` | acceleration bound | m/s^2 | `\|u\| <= umax`, the double integrator's only dynamics parameter |
| `TTARGET,t0` | target level | s | target set `= {(x,v) : min-time-to-origin(x,v) <= t0}` |
| `HORIZON,T` | reachability horizon | s | how far backward in time the tube is grown; total elapsed-time budget from any tube state to the origin is `t0 + T` |

**The committed scenario:** `GRID,256,256` · `XDOM,-3,3` m · `VDOM,-2,2` m/s · `UMAX,0.8` m/s^2 ·
`TTARGET,0.6` s · `HORIZON,0.4` s (total budget `t0+T = 1.0` s).

**Why `HORIZON` is 0.4 s and not the larger number a casual reading of the catalog bullet might
suggest:** this value was **measured**, not guessed. The committed grid resolution (`256x256`)
combined with this project's Lax-Friedrichs numerical scheme accumulates boundary-classification
error roughly in proportion to the number of explicit sweeps (`THEORY.md §Numerical
considerations` has the full measured table). A float64 reference model of the exact per-cell
update shows `HORIZON=0.4` (109 sweeps) needs an excused boundary band of 2 grid cells; the
compiled FP32 build needs one more cell of real-rounding margin on top of that idealized number —
`kBandCells=3` in `src/kernels.cuh` is the smallest value that actually passes on this scenario
(confirmed by rebuilding at 2 and watching it fail; README Exercise 3). At a longer horizon like
1.5 s the same scheme needs roughly a 13-cell band, which would undercut the whole point of
checking against closed-form mathematics rather than a fudge factor. `scripts/make_synthetic.py`'s
docstring and inline comments carry the same reasoning, and its `--horizon` flag lets you reproduce
the horizon-vs-band measurement yourself (README Exercise 2).

The loader is strict: unknown row labels, missing required rows, or values outside a sane range
(e.g. `xmax <= xmin`, `umax <= 0`) abort the demo with an explicit message — a reachability tool
must never silently solve the wrong problem.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
