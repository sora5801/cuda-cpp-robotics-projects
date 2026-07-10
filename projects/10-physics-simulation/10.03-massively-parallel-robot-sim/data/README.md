# Data — 10.03 Massively parallel robot sim (Isaac-Gym-style: one robot, 10,000 environments)

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

This project's "dataset" is a **farm scenario** — how many environments, how long to run them, the
domain-randomization envelope, and the fixed balance-controller gains — not recordings. Every
per-environment mass/length draw, every initial angle, and every mid-run episode reset is generated
**inside the demo, on the GPU**, from the scenario's `SEED` field (via the shared xorshift32 stream in
[`../src/kernels.cuh`](../src/kernels.cuh)); correctness comes from the CPU-oracle comparison (the §5
gate) plus two physics-invariant checks (farm-level finiteness/reset-count bounds, undriven energy
conservation), not from stored ground truth.

| Property | Value |
|----------|-------|
| Kind | **Synthetic** scenario (the task definition — no RNG involved; a scenario is constants) |
| File | `sample/farm_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: N=10000, T_FARM=1000) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 1029 bytes (~1.0 KiB) |
| Checksum (SHA-256) | `24ca6f31dae9caa3be6a752dfb7a1e47c59cfaba29dd91364a68c1bf8a531e98` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness) |

### Fields / format

Plain-text CSV; `#` lines are comments. Every row is `LABEL,value` (loader: `load_scenario()` in
[`../src/main.cu`](../src/main.cu); the strict loader aborts on an unknown label, a short row, or a
missing required field — the committed file exercises every field, so nothing here is "just a
default"). Layout authority for all model constants not listed here (nominal mass/length, gravity,
actuator limit, fail/balanced angle thresholds, physics timestep): [`../src/kernels.cuh`](../src/kernels.cuh).

| Field | Units | Meaning |
|-------|-------|---------|
| `N` | count | environments in the full farm (committed: 10000) |
| `T_FARM` | ticks | ticks to run the full farm for, at `dt=0.02 s` (committed: 1000 = 20 s of simulated time per environment) |
| `EPISODE_CAP` | ticks | ticks per episode before a **successful** (cap) reset (committed: 200 = 4 s) |
| `SEED` | — | base RNG seed; every environment's own stream is derived from this + its index (`env_seed()` in kernels.cuh) |
| `DR_MASS_CART` | fraction | ± domain-randomization range on `mass_cart` around the 1.0 kg nominal (committed: 0.20 = ±20%) |
| `DR_MASS_POLE` | fraction | ± domain-randomization range on `mass_pole` around the 0.1 kg nominal (committed: 0.30 = ±30%) |
| `DR_LEN` | fraction | ± domain-randomization range on `pole_half_len` around the 0.5 m nominal (committed: 0.15 = ±15%) |
| `THETA0_RANGE` | rad | ± range the initial pole angle is drawn from at **every** reset (committed: 0.15 rad ≈ 8.6°) |
| `KX`, `KXD`, `KTH`, `KTHD` | N/m, N·s/m, N/rad, N·s/rad | fixed linear state-feedback balance-controller gains (committed: 12.0, 14.0, 73.0, 19.0 — pole-placement derivation in [`../THEORY.md`](../THEORY.md)) |

The domain-randomization ranges and the initial-angle range are two **different kinds** of randomness
— the former is drawn ONCE per environment at farm init and held fixed for the whole run (it models
per-unit manufacturing/loading uncertainty); the latter is redrawn at **every** reset, including
mid-run ones (it models "where does the robot happen to start this episode"). THEORY.md's §domain
randomization explains why the distinction matters.

Everything else the demo consumes is generated at run time from the `SEED` field: the per-environment
`xorshift32` streams (kernels.cuh), the domain-randomized `mass_cart`/`mass_pole`/`pole_half_len`
draws, and every initial-angle draw at reset. The energy-conservation experiment
([`../src/reference_cpu.cpp`](../src/reference_cpu.cpp)) uses none of this file — it is a fixed,
compile-time diagnostic (nominal mass/length, `theta0 = 0.5 rad`, `kEnergySteps = 1000` — see
kernels.cuh) with no per-run configuration.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
