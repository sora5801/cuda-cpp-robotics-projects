# Demo — 02.04 Euclidean clustering via GPU union-find / connected components

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

It loads the committed scene (180,000 ground + 1,469 non-ground points,
[`../data/README.md`](../data/README.md)), builds the neighbor-edge graph once, then runs **two
independent GPU clustering algorithms on the identical edges** — a lock-free union-find (Method A) and
iterative min-label propagation (Method B) — verifies every stage against an independent CPU reference,
checks the final partition against the generator's own single-linkage ground truth, gates four designed
scenarios (a separation test, a chaining test, a long-diameter "snake" convergence test, and a
noise-filtering test), and writes five artifacts to `out/` (git-ignored, regenerated every run):

| Artifact | What it shows |
|----------|----------------|
| `topview_truth.ppm` | Top-view (looking down -z) of the whole scene, non-ground points colored by GROUND-TRUTH cluster id (ground rendered dim gray, context only). |
| `topview_gpu_result.ppm` | The same view, non-ground points colored by the GPU union-find pipeline's FINAL (post-filtering) cluster id — compare side by side with the truth image; they should look identical. |
| `topview_snake_highlight.ppm` | Everything dim gray except the long-diameter "snake" chain, highlighted in bright magenta — the chain THEORY.md's O(diameter) argument is about. |
| `sweep_comparison.csv` | Union-find vs. label-propagation: sweep count, converged flag, GPU ms, CPU-twin ms (union-find only — label propagation has no CPU sweep-count twin, only a partition twin). |
| `gates_metrics.csv` | Every measured number behind every `VERIFY`/`GATE`/`[info]` line — edge counts, sweep counts, timings, the stats tolerance's measured delta. |

The money number: on the committed scene, **union-find converges in 2 sweeps (~0.2-0.4 ms)**; **label
propagation needs 299 sweeps (~13-14 ms)** on the exact same edge list, producing the exact same final
partition — the O(log D) vs. O(D) complexity gap THEORY.md derives, made visible as a stopwatch
difference, not just an asymptotic claim.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|------------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and the raw/reported cluster-count summary line — printed `[info]` (not diffed) for consistency with every other measured-number line, even though it happens to be stable given the fixed sample. | No. |
| `PROBLEM:`  | The exact problem instance (ground/non-ground point counts, `d`, `min_cluster_size`) — fixed by the committed sample, so stable. | Yes — stable. |
| `DATA:`     | Which sample file, and that it is synthetic. | Yes — stable. |
| `VERIFY(...)` | GPU-vs-independent-CPU agreement for keys, edges, union-find, label propagation, and per-cluster stats. | Yes — stable (PASS/FAIL text only, no numbers). |
| `GATE ...:` | The six gates: `partition_vs_truth`, `stats_integrity`, `noise_filtering`, `separation_test`, `chaining_test`, `snake_convergence`. The `snake_convergence` line embeds the measured sweep counts (299 / 2) directly — confirmed stable across repeated runs before being locked into `expected_output.txt`. | Yes — stable. |
| `[time]`    | CPU/GPU timings — a **teaching artifact, never a benchmark claim** (single-shot; see `../THEORY.md`). | No. |
| `ARTIFACT:` | Which files were written to `out/`. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict; the program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list). If the failure mentions `CCCL`, `/Zc:preprocessor`,
  or `__cplusplus`, see the detailed comment in `../build/*.vcxproj`'s `CudaCompile` sections — a known
  CUDA 13.3 + Thrust + MSVC interaction, already worked around there.
- **A `VERIFY` or `GATE` line prints `FAIL`:** a real bug — the program prints extra detail to stderr.
  Start in `../src/kernels.cu` (the stage that failed) and compare against `../src/reference_cpu.cpp`'s
  matching twin.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together (`../src/main.cu`'s "Output contract" comment explains the rule).
