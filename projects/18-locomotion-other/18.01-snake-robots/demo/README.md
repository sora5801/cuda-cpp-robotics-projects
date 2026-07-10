# Demo — 18.01 Snake robots: serpenoid gait sweeps coupled to granular sim

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo sweeps **8,192 candidate snake gaits** (32 amplitudes x 32 phase offsets x 8 temporal
frequencies) on the GPU — one thread per gait, each integrating an independent 8-second, 8,000-step
trajectory of a 12-link snake dragging on anisotropic-friction ground — finds the fastest one, spot-
checks 32 of the GPU's own results against a from-scratch CPU recomputation, and runs four physics
gates that each test a specific prediction from `THEORY.md` (zero amplitude, isotropic friction,
turning bias, and the amplitude speed-ridge). It writes three artifacts:

- **`demo/out/sweep_surface.csv`** — the full 32x32 `(amplitude, phase offset)` -> `(speed,
  straightness, cost-of-transport)` grid at the best-discovered temporal frequency. **This is the
  plotting payload** — load it into any spreadsheet or plotting tool and look for the speed ridge
  THEORY.md derives (a single interior peak along the amplitude axis, at every phase offset).
- **`demo/out/sweep_surface.pgm`** — the same speed grid as a viewable grayscale heatmap (ASCII P2
  PGM — openable in any image viewer, or readable by eye as a text file). Amplitude increases
  downward, phase offset increases rightward; brighter = faster.
- **`demo/out/best_gait_path.csv`** — the winning gait's head trajectory `(t, x, y, yaw)`, ~400 rows
  sampled every 20 ms across the 8 s run. Plot `x` vs `y` to see the actual path traced; plot `yaw` vs
  `t` to see how much the heading drifts even with zero turning bias (README §Limitations discusses
  this honestly — the fastest gait is not the straightest one).

Look at the `[info] best gait:` line for the winning `(A, beta, omega)` and its measured
speed/straightness/cost-of-transport, and the four `[info] gate ...:` lines for the exact measured
numbers behind each `GATE_*:` verdict.

## Interpreting the artifacts

- **`sweep_surface.csv`/`.pgm`**: look for a bright/high-speed BAND rather than a single point — the
  ridge is a whole curve of good `(A, beta)` combinations, not one isolated peak; and look for how the
  ridge's location shifts (or doesn't) across the 32x32 grid.
- **`best_gait_path.csv`**: the path is generally NOT a straight line (measured straightness ~0.75 on
  the reference machine) — this is real, measured physics (the gait that goes fastest is not
  necessarily the gait that goes straightest), not a bug; see README §Limitations and Exercise 3 for
  how to search for a straighter gait instead.

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
