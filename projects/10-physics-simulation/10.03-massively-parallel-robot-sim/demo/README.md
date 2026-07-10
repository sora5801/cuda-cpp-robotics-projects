# Demo — 10.03 Massively parallel robot sim (Isaac-Gym-style: one robot, 10,000 environments)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A GPU training farm actually running.** The demo builds a 10,000-environment cart-pole farm —
one robot model, ten thousand independent copies, each with its own domain-randomized mass/length and
its own episode clock — and steps the whole farm 1,000 ticks (20 s of simulated time per environment)
in **one GPU kernel launch**, at an aggregate throughput typically in the **several-billion
env-steps/second** range on the reference machine (an RTX 2080 SUPER; single-shot, teaching artifact,
not a benchmark claim — see `[time]` lines). Watch three things happen:

1. **VERIFY** — a 256-environment subset is stepped 220 ticks on both the GPU kernels and a
   sequential CPU oracle, from identical seeds. The two must agree — not just "close": the episode
   reset (an integer step-counter hitting the 200-step cap) is bit-for-bit deterministic on both
   paths, so `reset_count` is required to match **exactly**, while the floating-point state is allowed
   a documented tolerance (measured worst case: ~5e-7, see `[info]` lines).
2. **FARM** — the full 10,000-environment run: every environment's state must stay finite (no NaN/Inf
   — the GPU never silently produces garbage), and every environment's `reset_count` must land in a
   documented, largely *provable* range (episode_cap=200 and T=1000 ticks means every environment
   resets **at least** 5 times by construction — no dynamics needed to prove that).
3. **ENERGY** — a completely separate, single-trajectory diagnostic: an undriven, frictionless
   cart-pole should conserve energy exactly; RK4 doesn't, quite, and this stage measures exactly how
   much it doesn't — the integrator's own truncation error, made visible as a number.

**This demo writes two artifacts** (git-ignored, regenerated each run):

- `out/env_metrics.csv` — per-environment `mass_cart_kg, mass_pole_kg, pole_half_len_m,
  steps_balanced, reset_count, balanced_fraction` for the first 1,000 environments. Plot
  `balanced_fraction` against `mass_pole_kg` and you can *see* domain randomization's effect on
  control quality.
- `out/energy_drift.csv` — `step, t_s, energy_j, drift_rel` for the 1,001-sample undriven trajectory.
  Plot `energy_j` vs `t_s`: it should look like a flat line at the scale of the swing itself, and a
  very slightly wobbling one at the scale of `drift_rel`.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured deviations/throughput/energy numbers — vary by machine. | No. |
| `PROBLEM:`  | The exact problem instance (N, T, dt). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The domain-randomization envelope and episode cap. | Yes — stable. |
| `[time]`    | CPU/GPU timings and the aggregate env-steps/second figure — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:`   | `PASS`/`FAIL` of the §5 GPU-vs-CPU gate on the 256-env subset (tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `FARM:`     | `PASS`/`FAIL` of the full-farm finiteness + reset-count-range gates. | Yes — stable. |
| `ENERGY:`   | `PASS`/`FAIL` of the undriven energy-conservation gate. | Yes — stable. |
| `ARTIFACT:` | Confirms each CSV was written, with its row count. | Yes — stable (two lines). |
| `RESULT:`   | Overall `PASS`/`FAIL` — `PASS` only if VERIFY, FARM, and ENERGY all pass. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU state disagreed with the CPU oracle, or `reset_count` did not match
  exactly — a real bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`;
  both call the SAME shared per-env functions in `../src/kernels.cuh`, so a mismatch usually means the
  kernel's register bookkeeping (entry read / exit write) diverged from the CPU loop's, not the physics.
- **`FARM: FAIL`:** either a NaN/Inf appeared (a real numerical bug — the farm should never produce
  one) or `reset_count` left the documented range (the controller became meaningfully less robust
  across the domain-randomization envelope than measured — see THEORY.md's calibration).
- **`ENERGY: FAIL`:** the RK4 integrator drifted more than the documented, measured-calibrated bound —
  check that `cartpole_deriv`/`rk4_step` in `kernels.cuh` were not accidentally changed.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
