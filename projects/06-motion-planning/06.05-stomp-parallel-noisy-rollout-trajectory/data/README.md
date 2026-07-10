# Data — 06.05 STOMP: parallel noisy-rollout trajectory optimization (born for GPU)

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more. **This project needs none** — a planner's input is a *scenario* (a map, a start, a
  goal, and obstacles), not recordings; the obstacle-cost field, the exploration noise, and the
  rollouts are generated in-demo from the scenario and fixed seeds, and correctness comes from the CPU
  scoring oracle plus the closed-loop collision / cost-reduction verdict.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** scenario (map + start + goal + obstacles; the demo inflates the cost field from it at load time) |
| File | `sample/obstacle_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: 3 obstacles, seed 42) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 566 bytes |
| Checksum (SHA-256) | `fb0877e85147abbc281bf98081519620041afc1d2c6f32bdc8deae5dded0108f` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (seed 42; LF line endings) |

### Fields / format

Plain-text CSV, LF line endings; `#` lines are comments. Four row types (loader: `load_scenario()` in
[`../src/main.cu`](../src/main.cu); layout/cost conventions: [`../src/kernels.cuh`](../src/kernels.cuh)).
All positions are in the world frame, right-handed, **+x right / +y up, origin at the map's
lower-left corner**, SI units (metres):

| Row | Fields | Meaning |
|-----|--------|---------|
| `MAP,w,h` | width, height (m) | world size. The demo assumes a **square** map (one cell size for x and y); the committed scenario is 10 × 10 m. |
| `START,x,y` | position (m) | the **fixed** start of the trajectory (committed: 1, 1) — never perturbed or optimized. |
| `GOAL,x,y` | position (m) | the **fixed** goal (committed: 9, 9). |
| `OBST,cx,cy,r` | centre x, centre y, radius (m) | one circular obstacle; the row repeats (committed: 3, straddling the start→goal diagonal so the straight-line initialization collides). |

Everything else the demo consumes is generated at run time from documented fixed seeds: the obstacle-cost
**field** (distance-based inflation of the circles onto a 256×256 grid — `build_cost_field()` in
`../src/main.cu`), the smoothing matrix **M** (from `R = AᵀA`), and the exploration noise (xorshift32 +
Box–Muller, base seed 42, smoothed by `M`, fresh stream per iteration). STOMP hyperparameters and cost
weights are compile-time constants in `../src/kernels.cuh` — part of the *taught, tuned* setup, not data.

The loader is strict: unknown labels, short rows, a missing `MAP`/`START`/`GOAL`, or zero obstacles
abort the demo.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
