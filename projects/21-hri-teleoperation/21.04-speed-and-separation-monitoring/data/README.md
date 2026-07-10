# Data — 21.04 Speed-and-separation monitoring: depth streams → minimum-distance fields at frame rate (ISO/TS 15066 helper)

> Didactic implementation — NOT a certified safety function. See [`../src/kernels.cuh`](../src/kernels.cuh)
> for the full caveat.

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

There is no depth-camera recording anywhere in this project — every depth frame, robot pose, and
human capsule position is synthesized in closed form, in-code, at run time (`src/main.cu`'s
`build_scene()`). What is committed here is the tiny **scenario description** that seeds that
synthesis: how many frames the sequence runs, at what rate, and where the human's walk starts and
turns around. Everything else the pipeline needs (the robot's link lengths and joint sweep, the
camera/cell geometry, the ISO/TS-15066-style formula parameters) is the compile-time "model"
documented once in [`../src/kernels.cuh`](../src/kernels.cuh) — the same synthetic-first,
data-vs-model split project 08.01 uses for its cart-pole scenario.

| Property | Value |
|----------|-------|
| Kind | Synthetic (default, CLAUDE.md §8) — no public dataset applies to a synthetic SSM teaching pipeline |
| Generator / source | `python ../scripts/make_synthetic.py` (no RNG anywhere — every field is a literal, deterministic constant) |
| License | Synthetic — this repository's MIT license applies |
| Size (committed) | `ssm_scenario.csv`, 613 bytes |
| Checksum | `ssm_scenario.csv`: `139b6d8a3b716ba7b2c0417a271287e1bd84dc3aa6610e54661de629ef87dcc1` (SHA-256) |
| Regenerate with | `python ../scripts/make_synthetic.py --frames 240 --rate 30.0 --human-start -1.6 0.9 --human-closest 0.7 0.9` (the committed defaults) |

### Fields / format (`data/sample/ssm_scenario.csv`)

Row-labeled CSV, one field per row (the same strict-loader idiom as project 08.01's
`cartpole_scenario.csv`); `#`-prefixed lines are comments. All coordinates are in the **cell
frame**: meters, right-handed, x/y in the flat floor plane, z up, origin at the robot's base
(`docs/SYSTEM_DESIGN.md` §3.1 interface conventions).

| Row label | Fields | Units / range | Meaning |
|---|---|---|---|
| `FRAMES` | `n` | integer, > 1 | Total frames in the demo sequence (240 in the committed sample = 8.0 s at 30 Hz) |
| `RATE_HZ` | `hz` | float, > 0 | Depth-camera frame rate (30 Hz — `docs/SYSTEM_DESIGN.md` §1.1's 30–60 Hz camera band) |
| `HUMAN_START` | `x_m, y_m` | meters, cell frame | The human's walk start/end point (the raised-cosine path's `s=0` endpoint) |
| `HUMAN_CLOSEST` | `x_m, y_m` | meters, cell frame | The walk's turnaround point (`s=1`, the human's closest approach to the robot) |

No robot geometry, camera intrinsics, or SSM formula parameters live in this file by design — they
are the project's compile-time "model" (`kernels.cuh` SECTIONS 2/3/4/6), not scenario data;
`scripts/make_synthetic.py`'s own docstring explains the split and warns that changing the
scenario's start/closest points is safe to try but may change *when* (or whether) the SSM state
machine transitions — `main.cu`'s own analytic gates report that honestly on every run, they never
silently pass a broken scenario.

> **What `data/sample/README.md` calls "the tiny committed sample" here is a task description, not a
> recording** — exactly the same distinction 08.01 draws for its cart-pole scenario. The demo's
> actual visual/tabular artifacts (`demo/out/distance_field.pgm`, `demo/out/ssm_timeline.csv`) are
> generated fresh every run from this scenario plus the compiled model, never committed (they are
> run-time scratch, git-ignored — see `demo/README.md`).
