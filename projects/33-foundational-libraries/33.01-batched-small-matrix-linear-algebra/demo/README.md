# Demo — 33.01 Batched small-matrix linear algebra (3×3, 4×4, 6×6 — the robotics sizes)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The **thread-per-problem batch pattern** — the single most reused GPU idiom in this repository. The
demo runs two stages:

1. **Sample stage** — loads the committed synthetic sample (`../data/sample/smallmat_sample.csv`:
   64/32/16 matmul pairs at *n*=3/4/6 and 32 SPD 6×6 systems), computes every problem on the GPU
   (`../src/kernels.cu`) **and** on the single-threaded CPU oracle (`../src/reference_cpu.cpp`), and
   compares within documented tolerances (matmul: 1e-5 absolute; Cholesky solve: 1e-4 relative).
2. **Batch stage** — regenerates 200,000 matmul pairs *per size* and 100,000 SPD solves in memory
   from a fixed seed, verifies GPU == CPU on **all** of it, and times the *n*=6 kernels against the
   CPU loop.

What to notice in the output: the worst sample deviations sit near **1e-7** — two orders below
tolerance — which is FP32 rounding plus FMA-contraction difference, *not* error (THEORY.md
§Numerical considerations tells that story); and the speed-up comes purely from batch parallelism —
one thread per matrix, no shared memory, no tuning — which is the whole lesson of the project.

No artifact file is written: the result of a linear-algebra check is a verdict, not an image, so
the demo's output is its text (the repo writes PNG/CSV artifacts only where results are inherently
visual, per CLAUDE.md §6.3).

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
