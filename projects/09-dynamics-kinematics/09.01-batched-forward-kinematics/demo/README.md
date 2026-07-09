# Demo — 09.01 Batched forward kinematics (10⁵ configurations — the foundation for everything above)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**Batched forward kinematics** — the pose of a 6-DoF arm's end-effector computed for many joint
configurations at once, one GPU thread per configuration. The demo:

1. loads the committed synthetic sample (`../data/sample/fk_sample.csv`: the arm model + 64
   configurations), validates and uploads the model to GPU **constant memory** once, then computes
   every pose on the GPU (`../src/kernels.cu`) *and* on the single-threaded CPU oracle
   (`../src/reference_cpu.cpp`), comparing within tolerance (position 1e-4 m; quaternion 1e-4 per
   component **after hemisphere alignment** — q and −q are the same rotation);
2. regenerates 200,000 configurations in memory (seed 42, angles in (−π, π]), verifies GPU == CPU
   on all of them, and times both paths.

What to notice: the worst deviations sit near **1e-7** (pure FP32 + trig-implementation rounding,
two-plus orders inside tolerance), and the speed-up comes from batch parallelism alone — the FK
chain itself is inherently sequential (joint *j* needs joint *j−1*'s frame); parallelism lives
*across* configurations, never along the chain. No artifact file is written — poses are verified,
not visualized (the visual FK story belongs to consumers like reachability maps, 09.06).

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
