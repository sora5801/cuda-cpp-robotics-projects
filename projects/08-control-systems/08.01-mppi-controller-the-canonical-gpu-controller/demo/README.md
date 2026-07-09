# Demo — 08.01 MPPI controller — the canonical GPU controller: cart-pole → quadrotor → AGV → off-road racer

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A GPU controller actually controlling something.** The demo runs MPPI closed-loop on a simulated
force-limited cart-pole: from hanging straight down, the controller pumps energy into the pole
(the force limit makes a direct pull-up impossible — watch `u_N` saturate at ±10 N in the log),
swings it up in ~2–3 s, catches it, and balances it upright for the rest of the 8-second run.
Every 20 ms tick, the GPU simulates **4096 candidate futures of 50 steps each** (~0.3 ms of kernel
time), and the host blends them with softmin weights into the plan whose first action is applied.

Two checks gate the verdict:

1. **VERIFY** — the §5 GPU-vs-CPU gate: iteration 0's 4096 rollout costs computed on both paths
   must agree within rel 1e-3 (measured deviation ~2e-7).
2. **RESULT** — the control check: the pole must hold |θ| < 0.2 rad for every one of the final 100
   steps (it typically balances for the final ~290).

**This demo writes an artifact**: `out/trajectory.csv` (git-ignored, regenerated each run) with
columns `t_s, p_m, pdot_ms, theta_rad, thdot_rads, u_N` — plot θ vs t with any tool and you can
*see* the energy-pumping oscillations, the swing-up, and the catch. That plot is the picture this
project exists to produce.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name and compute capability — varies by machine. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, parameters). | Yes — stable (demo runs with no args). |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU check (tolerance documented in `../src/main.cu` and `THEORY.md`). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
