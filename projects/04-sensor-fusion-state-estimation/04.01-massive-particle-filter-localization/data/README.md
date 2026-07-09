# Data — 04.01 Massive particle filter localization (10⁵–10⁶ particles, GPU likelihoods + resampling)

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more. **This project needs none** — the classic candidates (TUM RGB-D, EuRoC, KITTI)
  carry the wrong sensor for a planar range-fan MCL teaching core, and the closed-loop RMSE gate
  needs *exact* ground truth, which only synthesis provides for free. The scripts are honest
  no-ops with the decision documented in their headers.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** world + run: occupancy-grid map (fixed geometry, no RNG) and a 120-step trajectory with seeded noisy odometry + noisy range scans, ground truth included |
| Files | `sample/grid_map.txt` and `sample/trajectory_scans.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: seed 42, 120 steps) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | ~4.8 KiB (map) + ~20.3 KiB (trajectory/scans) |
| Checksum (SHA-256) | `grid_map.txt`: `7703961420ba3773232dbff5794e99730a08e19a991f81b116d7cb9f98c90190` · `trajectory_scans.csv`: `8ec991c8a2d9951d146aba2be894c07fe6f613369b65bff686717df847440392` |
| Regenerate with | `python ../scripts/make_synthetic.py` (seed 42). Fixed-precision floats + pinned LF endings make regeneration byte-stable in practice; in principle a last-ulp libm difference on another platform could flip a 6th decimal, so **the committed copy is canonical** — an honest note, not a hedge. |

### Frames & conventions (shared with `../src/kernels.cuh` — the C++ authority)

World frame: origin at the map's lower-left corner, **x right, y up**, right-handed; heading
`theta` measured CCW from +x, radians, **unwrapped** in the log (it grows to 2π over the loop).
All units SI (m, m/s, rad, rad/s, s).

### `sample/grid_map.txt` — the occupancy grid

| Field | Meaning |
|-------|---------|
| `WIDTH,64` / `HEIGHT,64` | grid dimensions (cells) |
| `RESOLUTION,0.25` | cell size (m) → a 16 m × 16 m world |
| `MAP` + 64 rows | `'.'` free / `'#'` occupied; cell `(ix,iy)` covers `[ix·res,(ix+1)·res) × [iy·res,(iy+1)·res)` |

Rows are written **top row first** (`iy = HEIGHT−1`) so the file reads like a map with +y up;
the loader flips (file line *j* → `iy = HEIGHT−1−j`). Geometry: a 1-cell border wall plus five
rectangular obstacles (four outer blocks for beam structure, one 0.5 m pillar inside the robot's
loop). `#` lines are comments **only before** the `MAP` marker — wall rows legitimately start
with `#`.

### `sample/trajectory_scans.csv` — the run

One `INIT` row, one `STEP` header row, then 120 data rows (`#` lines are comments):

| Field(s) | Units | Meaning |
|----------|-------|---------|
| `INIT,x,y,theta` | m, m, rad | the TRUE start pose; the filter seeds its particle cloud around it (pose tracking) |
| `STEP` (col 1) | — | step index 0..119, validated in order by the strict loader |
| `t_s` | s | time after the step = `(step+1)·0.1` (redundant; for humans and plots) |
| `gt_x_m, gt_y_m, gt_theta_rad` | m, m, rad | ground-truth pose **after** applying the step's true twist (Euler-integrated unicycle, same update order as the predict kernel) |
| `odo_v_ms, odo_w_rads` | m/s, rad/s | the NOISY odometry measurement of that twist: true command + N(0, 0.05) each |
| `z00_m..z15_m` | m | 16 NOISY ranges from the post-step true pose: true fixed-step raycast + N(0, 0.10), clamped to [0.125, 8.0]. Beam *b* points at world angle `gt_theta + (−π + b·π/8)`: z00 rear, z04 right, z08 forward, z12 left |

The generator ray-marches with the **same fixed-step algorithm** the kernels use (0.125 m steps,
8 m max), so the measurement model and the filter's expected-range model agree by construction;
Python marches in doubles where the kernels march in FP32 — the sub-millimeter discrepancy just
looks like extra sensor noise, well inside the filter's assumed σ_z = 0.15 m. The generator also
asserts the true path keeps ≥ 0.45 m clearance from every occupied cell.

Everything else the demo consumes is generated at run time from documented fixed seeds
(kernels.cuh `kBaseSeed = 42`): the initial particle cloud, the per-particle prediction noise
(counter-based in-kernel xorshift32), and the resampler's uniform draws. Filter hyperparameters
(noise sigmas, beam count, gate) are compile-time constants in `../src/kernels.cuh` — part of the
*taught, tuned* setup, not data.

The loaders are strict: unknown labels, short rows, wrong row lengths, or out-of-order step
indices abort the demo (`load_map`/`load_log` in [`../src/main.cu`](../src/main.cu)).

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
