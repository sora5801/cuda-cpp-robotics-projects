# Data — 15.01 Minimum-snap trajectory optimization batched over waypoint sets

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more. **This project needs none** — a minimum-snap batch's input is a *list of (x,y)
  waypoints*, not recordings; the demo's random 9,995-set fill and the CPU/constraint verification
  are all generated/computed in-demo from documented fixed seeds and formulas.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic**, hand-designed waypoint sets (5 named path shapes — not drawn from a distribution) |
| File | `sample/waypoint_sets.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (fixed constants; see the script docstring for why `--seed` has no effect on these 5 sets) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 820 bytes |
| Checksum (SHA-256) | `9dcb1dd2fc5c6980ef60a0ca47ca38d2d247d1ab5a30bd8527b54df3b910bfb8` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness) |

### Fields / format

Plain-text CSV; `#` lines are comments; LF line endings. Two row types (loader: `load_sample_sets()`
in [`../src/main.cu`](../src/main.cu); waypoint-count authority: [`../src/kernels.cuh`](../src/kernels.cuh)'s
`kNumWaypoints`):

**`SET,<name>`** — opens a new waypoint set; `<name>` is a plain identifier (no commas). The set
named **`slalom` is required** — `main.cu` aborts loudly if it is missing, because the demo
dense-samples that exact set into the `demo/out/trajectory.csv` / `demo/out/slalom_path.pgm`
artifact.

**`WP,<x_m>,<y_m>`** — exactly `kNumWaypoints` (5) of these must follow each `SET` row, in flight
order, SI units (meters), world frame (x-forward, y-left — [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) §3.2).

The five committed sets and what each teaches:

| Name | Shape | Why it is here |
|------|-------|-----------------|
| `straight_line` | 5 colinear points | Sanity case: near-zero curvature: does the solver behave on the simplest possible input (and does the demo's "cost must be positive" check still hold — see THEORY.md, the rest-to-rest boundary conditions force a nonzero snap profile even here)? |
| `right_angle` | one sharp corner | A single abrupt direction change at the middle waypoint. |
| `slalom` | zig-zag | **Required** — the demo's dense-sampled artifact; visually the clearest "pumped, curving flight" shape. |
| `s_curve` | smooth reversing curve | Opposite of `right_angle`'s sharp corner — a gentle direction reversal. |
| `big_loop` | returns near its start | Four segments trace a closed-ish quadrilateral. |

Everything else the demo consumes is generated or computed at run time from documented fixed
parameters:

- **The other ~9,995 waypoint sets** of the default 10,000-set batch: seeded xorshift32
  (`generate_random_waypoint_set` in `../src/main.cu`), points drawn uniformly from a
  `±kBoxHalfExtentM` (4 m) box with a minimum `kMinSpacingM` (0.75 m) between consecutive waypoints
  (rejection sampling).
- **The 32×32 constraint system per axis, per set**: assembled from the fixed algebraic rules in
  [`../src/kernels.cuh`](../src/kernels.cuh) (position, endpoint-derivative, and interior-continuity
  rows) — no data file involved; the matrix is the SAME for every set (only the right-hand side
  differs), because it depends only on the constraint structure, never on the waypoint values.
- **The verification tolerances and the analytic snap-cost formula** are compile-time constants in
  `../src/main.cu`, documented there and in `THEORY.md` — part of the taught, tuned setup, not data.

The loader is strict: unknown row labels, a `WP` row before any `SET` row, a `SET` with the wrong
number of `WP` rows, or a missing `slalom` set all abort the demo with a clear message.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
