# Data — 01.12 Visual servoing: image-Jacobian control loop entirely on GPU

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

Like project 08.01's MPPI controller, a CONTROLLER's "dataset" is a task SIZE, not recordings: how
many independent closed IBVS loops to simulate. Everything else — the target geometry, the goal
camera pose, the K loops' initial poses across the three designed cohorts, and the basin-map grid —
is generated **inside the demo executable** from documented compile-time constants
(`src/kernels.cuh`) and a fixed xorshift32 seed (`src/reference_cpu.cpp`
`generate_batch_init_poses_cpu`), and verified against the independent CPU oracle (see
`THEORY.md` "How we verify correctness"). No recordings, no camera captures, no fiducial images —
this project studies the **control law**, not the perception front end that would feed it on a real
robot (see README "System context": upstream 01.04/01.06 would produce these 4 image points from a
real camera).

| Property | Value |
|----------|-------|
| Kind | **Synthetic** scenario (the task definition — no RNG involved; a scenario is constants) |
| File | `sample/ibvs_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: K=4096, BASIN_G=64) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | ~0.7 KiB |
| Checksum (SHA-256) | `4565db5c94a6853d6dea42a1acff2c939c71884619d400440a761d75a8ed6e3a` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness) |

### Fields / format

Plain-text CSV; `#` lines are comments. Two row types (loader: `load_scenario()` in
[`../src/main.cu`](../src/main.cu)):

| Field | Meaning |
|-------|---------|
| `K,n` | number of independent closed IBVS loops in the main batch (default 4096) |
| `BASIN_G,g` | basin-map grid side; the grid has `g*g` translation-only loops (default 64, so 4096 grid points) |

Everything else this project consumes is generated at run time from documented fixed constants and
seeds (all in `src/kernels.cuh`, cross-referenced from `THEORY.md`):

- **Target geometry & goal pose** — a 4-point coplanar square (`kTargetHalfSize` = 0.06 m half-side)
  fixed in the world frame; the goal camera stands off `kGoalStandoff` = 0.5 m, fronto-parallel
  (identity orientation). Closed-form, no RNG: `build_target_and_goal_cpu()`.
- **Initial poses** — the K loops split across three designed cohorts (nominal / decay / retreat,
  see `kernels.cuh` "COHORTS"), drawn from a **host-generated xorshift32** stream, base seed **42**,
  one independent stream per loop index (the same per-index seed-mixing formula 08.01 uses for its
  per-tick noise). `generate_batch_init_poses_cpu()` — SI units (m, rad) throughout, documented at
  the point of use.
- **Controller constants** (λ, damping μ, dt, step budget, convergence threshold) — compile-time
  constants in `kernels.cuh`; part of the *taught, tuned* setup, not data (mirrors 08.01's MPPI
  hyperparameters).

The loader is strict: unknown labels, short rows, or a missing `K`/`BASIN_G` abort the demo.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
