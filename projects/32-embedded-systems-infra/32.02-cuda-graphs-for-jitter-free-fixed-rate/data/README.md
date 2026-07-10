# Data — 32.02 CUDA Graphs for jitter-free fixed-rate perception-control loops

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

This project's "data" is a **configuration**, not a recording: the tick pipeline's starting state
and the two RNG seeds that make 2000×3 ticks of synthetic sensor readings and MPPI exploration
noise reproducible byte-for-byte across every run and every mode. The bulky per-tick arrays
themselves (`sensor_raw[512]`, `eps[16×512]` every tick) are generated **in memory** from this
seed, deterministically, by `src/main.cu`'s `tick_inputs()` — nothing bulky needs to be committed.

| Property | Value |
|----------|-------|
| Kind | Synthetic configuration (no public dataset applies — this is a latency-measurement workload, not sensor data) |
| Generator / source | `../scripts/make_synthetic.py` (defaults match the committed file exactly) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | `sample/tick_scenario.csv`, ~0.6 KiB |
| Checksum | SHA-256 `844198cd9f8f1d13f66d3a9dc27f35643cbc391c3a2634f355eca2f939323df7` (recompute with `sha256sum sample/tick_scenario.csv` after any regeneration) |
| Regenerate with | `python ../scripts/make_synthetic.py` (defaults) or with explicit flags: `python ../scripts/make_synthetic.py --measured-ticks 2000 --warmup-ticks 50 --hz 250.0 --seed-eps 1234567 --seed-sensor 7654321` |

### Fields / format

`sample/tick_scenario.csv` — six labeled rows, parsed by `src/main.cu`'s `load_scenario()`
(any unrecognized label is a hard error, the same strict-loader discipline as 08.01's scenario file):

| Label | Fields | Meaning |
|---|---|---|
| `X0` | `p_m, pdot_ms, theta_rad, thdot_rads` | the tick pipeline's initial state estimate — kernels.cuh's 08.01-derived cart-pole layout (§12 unit/frame convention: SI, radians), reused here as a representative small state vector, NOT a claim about any real plant |
| `SEED_EPS` | one `uint32` | xorshift32 seed for the MPPI exploration-noise stream (`tick_inputs()`) |
| `SEED_SENSOR` | one `uint32` | xorshift32 seed for the synthetic sensor-reading stream (independent of `SEED_EPS`, so noise and "sensor" bits never correlate) |
| `MEASURED_TICKS` | one `int` | ticks measured per mode (2000 in the committed sample) |
| `WARMUP_TICKS` | one `int` | unpaced ticks run and discarded before measurement starts, per mode (50 in the committed sample — absorbs first-launch JIT/module-load cost) |
| `PACING_HZ` | one `double` | the fixed software pacing rate every mode targets (250.0 Hz in the committed sample — README "System context" argues this choice) |

Changing `MEASURED_TICKS`/`WARMUP_TICKS`/`PACING_HZ` and regenerating the file changes the demo's
`PROBLEM:` line — the same documented contract 08.01's `--rollouts` note uses for its own scenario
parameter (`demo/expected_output.txt` would need updating to match).
