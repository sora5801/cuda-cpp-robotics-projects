# Data — 08.01 MPPI controller — the canonical GPU controller: cart-pole → quadrotor → AGV → off-road racer

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more. **This project needs none** — a controller's input is a *scenario*, not recordings;
  noise, rollouts, and the simulated plant are generated in-demo from fixed seeds, and correctness
  comes from the CPU rollout oracle plus the closed-loop success check.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** scenario (the task definition — no RNG involved; a scenario is constants) |
| File | `sample/cartpole_scenario.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: θ₀ = π, 400 steps) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | ~0.4 KiB |
| Checksum (SHA-256) | `d31eea71550299c279fb812b26d3ceabf91e36e930a7683694514f0e23d73bcc` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness) |

### Fields / format

Plain-text CSV; `#` lines are comments. Two row types (loader: `load_scenario()` in
[`../src/main.cu`](../src/main.cu); state layout authority: [`../src/kernels.cuh`](../src/kernels.cuh)):

**`X0,p,pdot,theta,thetadot`** — the plant's start state, SI units:

| Field | Units | Meaning |
|-------|-------|---------|
| `p` | m | cart position (+right; 0 = track center) |
| `pdot` | m/s | cart velocity |
| `theta` | rad | pole angle, **0 = upright, π = hanging straight down** (the committed scenario starts at exactly π — the classic swing-up benchmark) |
| `thetadot` | rad/s | pole angular velocity |

**`STEPS,n`** — closed-loop control steps at 50 Hz (committed: 400 = 8 s — enough for a ~2–3 s
swing-up plus a long balancing tail the success check can be strict about).

Everything else the demo consumes is generated at run time from documented fixed seeds:
exploration noise (xorshift32 + Box–Muller, base seed 42, fresh stream per control step) and the
simulated plant itself (the RK4 stepper in `../src/reference_cpu.cpp`). MPPI hyperparameters and
cost weights are compile-time constants in `../src/kernels.cuh` — they are part of the *taught,
tuned* setup, not data.

The loader is strict: unknown labels, short rows, or a missing `X0`/`STEPS` abort the demo.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
