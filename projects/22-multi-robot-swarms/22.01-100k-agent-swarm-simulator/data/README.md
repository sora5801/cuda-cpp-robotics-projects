# Data — 22.01 100k-agent swarm simulator: flocking, pheromone grids, stigmergy

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

A swarm simulator's "dataset" is a **scenario** — how many agents, how long to simulate, and which
seed to spawn them from — not recordings. This project needs no public dataset: agent state is
generated in-demo from the seed below, and correctness comes from the lockstep CPU oracle plus the
end-of-run flock statistics (`THEORY.md` §How we verify correctness), not from stored ground truth.

| Property | Value |
|----------|-------|
| Kind | **Synthetic** scenario (constants — no RNG runs in the generator itself; a scenario is just numbers) |
| File | `sample/swarm_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: N=100000, STEPS=300, SEED=42) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 457 bytes |
| Checksum (SHA-256) | `41f482d88f904a98797e7a0a274f062e9b0bb092cedf39c3540641a801885bb5` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness in the generator) |

### Fields / format

Plain-text CSV; `#` lines are comments. Three `label,value` rows (loader: `load_scenario()` in
[`../src/main.cu`](../src/main.cu); every constant's meaning and units are the single source of truth
in [`../src/kernels.cuh`](../src/kernels.cuh)):

| Field | Type | Meaning |
|-------|------|---------|
| `N,<n>` | int | Agent count. Committed value: 100,000 (the catalog's headline swarm size). |
| `STEPS,<t>` | int | Simulation steps to run at `dt = kDt = 0.05 s` (20 Hz). Committed value: 300 steps = 15 s of simulated swarm time. |
| `SEED,<s>` | uint32 | Spawn seed for the demo's deterministic xorshift32 generator (uniform positions inside the wall margin, uniform random headings, `|v| = 1 m/s`). Committed value: 42. |

Everything else the demo consumes is generated at run time, deterministically, from that seed: agent
positions/velocities (`spawn_agents()` in `main.cu`), and the pheromone field (starts at exactly zero —
built entirely by the agents as they run). All physical constants (arena size, interaction radii, rule
gains, pheromone diffusion/decay, integration timestep) are compile-time SI constants in
`../src/kernels.cuh` — they are the taught, tuned setup, not data, and both the GPU path and the CPU
oracle read the identical values.

The loader is strict: unknown row labels, short rows, or a missing `N`/`STEPS`/`SEED` abort the demo
loudly rather than silently falling back to a default (`load_scenario()` in `main.cu`).

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
