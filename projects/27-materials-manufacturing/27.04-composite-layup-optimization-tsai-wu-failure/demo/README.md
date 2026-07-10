# Demo — 27.04 Composite layup optimization + Tsai-Wu failure envelope sweeps

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs the full pipeline described in [`../README.md`](../README.md): a GPU sweep scores
all 256 symmetric 8-ply layups (from the `{0,45,-45,90}` alphabet) against two load-case sets
(`MIXED`: 16 combined `Nx/Ny/Nxy` directions; `ALIGNED`: pure `+/-Nx`), ranks them by worst-case
Tsai-Wu first-ply-failure load factor, then maps the full `(Nx,Ny)` failure envelope for both the
`MIXED`-set winner and the documented `[0/90/0/90]s` cross-ply baseline.

**What to notice:**
- The `MIXED` winner (`[info] MIXED winner: ...`) is always a permutation of one-ply-each-angle — a
  quasi-isotropic-like stack — and the `ALIGNED` winner is the all-0-degree stack, by a wide margin.
  Both are visible directly in `demo/out/layup_ranking.csv`'s top-10 rows.
- `demo/out/envelope_best.pgm` / `envelope_cross.pgm` are viewable grayscale images (any viewer
  that reads plain PGM, or open in a text editor — it is ASCII) of the failure-load-factor field
  over the `(Nx,Ny)` plane; brighter = more margin. `envelope_best_contour.csv` /
  `envelope_cross_contour.csv` are point clouds of the exact factor=1 boundary — the classic
  Tsai-Wu failure envelope — plot them as a scatter to see the boundary shape (see README Exercises
  for a plotting suggestion).
- Four `GATE_*:` lines are closed-form / physical checks independent of the sweep itself — see
  `THEORY.md` §How we verify correctness for what each one proves and why.

Both PGMs use the SAME fixed `[0, 5]` factor-to-gray scale (documented in each file's own comment
header) so the two images are directly brightness-comparable.

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
