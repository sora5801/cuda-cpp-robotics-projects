# Demo — 06.05 STOMP: parallel noisy-rollout trajectory optimization (born for GPU)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A GPU planner actually routing around obstacles.** The demo builds an obstacle-cost field from the
committed scenario (a 10 × 10 m map with 3 circular obstacles straddling the start→goal diagonal),
initializes the trajectory as a straight line — which drives **through** the obstacles — and runs
STOMP: each iteration it samples **1024 smooth-noise variations** of the path, scores them all on the
GPU (~0.6 ms of kernel time), and updates *each of the 64 waypoints separately* by a softmin blend.
In ~16 iterations the path bends smoothly around the obstacle cluster and becomes collision-free.

Two checks gate the verdict:

1. **VERIFY** — the §5 GPU-vs-CPU gate: iteration 0's 1024 rollout costs computed on both paths must
   agree within rel 1e-3 (measured worst on the reference machine ~2.2e-07; exactly 0 in a Debug build).
2. **RESULT** — the end-to-end verdict: the final path must be collision-free with margin (max field
   value along it below 25, i.e. ≥0.30 m clear of every obstacle) **and** its total cost must fall below
   5% of the straight-line value. Measured: fully clear (max field 0.000, ≥0.6 m clearance) and the
   collision cost eliminated (total 591.2 → 0.0013).

**This demo writes two artifacts** into `out/` (git-ignored, regenerated each run):

- `out/costfield.pgm` — a 256×256 grayscale image of the cost field (light = free space, dark =
  obstacles) with the **final path burned in as a black line**. Open it in any image viewer: you can
  *see* the route curve around the dark obstacle blobs. This is the picture the project exists to produce.
- `out/trajectory.csv` — the final path as `idx,x_m,y_m` (66 points: start, 64 waypoints, goal). Plot
  x vs y with any tool to overlay it on the obstacle positions from the scenario CSV.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, verify deviation, convergence and cost numbers — vary by machine. | No. |
| `PROBLEM:`  | The exact problem instance (K, N, iterations, cost). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The loaded task: start, goal, obstacle count, map size. | Yes — stable (fixed committed scenario). |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `VERIFY:`   | `PASS`/`FAIL` of the §5 GPU-vs-CPU scoring gate (rel tol 1e-3; documented in `../src/main.cu` and `THEORY.md`). Nonzero exit on `FAIL`. | Yes — stable. |
| `ARTIFACT:` | Confirms the trajectory CSV + cost-field PGM were written. | Yes — stable (fixed sizes). |
| `RESULT:`   | The end-to-end verdict: final trajectory collision-free with margin **and** total cost reduced. Nonzero exit on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU scoring disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **`RESULT: FAIL`:** the planner did not reach a collision-free, cost-reduced path — inspect the
  `[info]` cost/field lines and `out/costfield.pgm`.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
