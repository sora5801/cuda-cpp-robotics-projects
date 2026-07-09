# Demo — 07.09 Jump-flooding Voronoi/distance transforms (easy, visual, useful)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**The jump-flooding algorithm** — Voronoi labels + a distance field for every cell of a grid in
O(log) gather passes, versus the exact-but-quadratic CPU scan. Two stages (512×512 sample from the
committed seed file; 1024×1024 batch with 128 in-memory seeds), each checked against the **exact**
CPU oracle under JFA's documented approximation bounds: label mismatches ≤ 0.5% of cells, max
distance error ≤ 2 cells (mismatch counts are deterministic — the whole pipeline is integer
arithmetic).

**This demo writes images** (the result is inherently visual, CLAUDE.md §6.3): after a run, open

- `out/voronoi.pgm` — the Voronoi regions, one gray tone per seed label;
- `out/distance.pgm` — the distance transform (dark at seeds, bright far away): read it as the
  **clearance field** a local planner or safety monitor would consume, with seeds as obstacles.

(`demo/out/` is git-ignored run-time scratch; the images regenerate every run. Any image viewer
opens PGM; even VS Code previews it with common extensions.)

What to notice in the output: the sample stage typically matches the exact field **perfectly**, and
the million-cell batch disagrees on only a handful of cells with sub-cell distance error — that
gap between "approximate in theory" and "near-exact in practice" (and the 1+JFA variant that buys
it) is the project's central lesson (THEORY.md §How we verify correctness).

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
