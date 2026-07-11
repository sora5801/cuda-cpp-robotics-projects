# Demo — 36.03 Lattice-robot kinematics batches

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo generates a K=4096-configuration batch of 24-module sliding-cube lattice robots (seeded
accretion, 10% deliberately corrupted as negative controls), runs four GPU stage kernels over the
whole batch (validity, connectivity, articulation points, legal-move enumeration), checks the GPU
result against a CPU oracle **bit-exact** (this project is all-integer — no tolerance anywhere), checks
that every injected corruption is caught with zero false alarms, cross-checks a 128-configuration
subset against two independently-coded brute-force oracles, and finally runs a **reconfiguration
vignette**: one 24-module straight line, greedily folded into a compact blob via real, legality-checked
moves, with every intermediate state independently re-verified valid and connected.

Three artifacts land in `out/` (git-ignored — regenerate by rerunning the demo):

| File | What it is |
|---|---|
| `out/batch_stats.csv` | One row per configuration: `config_id,label,valid,connected,num_articulation,num_legal_moves`. `label` is `clean`/`duplicate`/`disconnect` (ground truth from the batch generator). Plot a histogram of `num_legal_moves` or `num_articulation` over the 3686 clean configurations. |
| `out/vignette_frames.csv` | Long-format `step,module,x,y,z` — every module's position at every one of the vignette's 128 recorded steps (step 0 = the starting line). The animation payload: group by `step`, scatter-plot the 24 `(x,y,z)` points, and step through frames to watch the line fold into a blob. |
| `out/config_render.pgm` | An ASCII PGM (`P2` — open it in any image viewer, or read it as text) top-down (XY, z ignored) projection of batch configuration 0 (guaranteed clean). Any occupied `(x,y)` column (any z) paints one `6x6`-pixel white block; row 0 of the image is the MAXIMUM y (image convention: y increases upward, so the top row is the "far" end of the shape). |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, scenario file path, mismatch counts, vignette details — varies by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (K, M, all-integer). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The seed and the corruption split (corrupted/clean counts). | Yes — stable. |
| `[time]`    | Batch-generation, GPU-kernel, and CPU-reference timings — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `VERIFY:`   | GPU-vs-CPU **bit-exact** agreement over all four stages, all K configurations. | Yes — stable. |
| `CORRUPTION-GATE:` | Every injected duplicate/disconnect config caught, zero false alarms on clean configs. | Yes — stable. |
| `ARTICULATION-BRUTEFORCE:` / `MOVE-PRECONDITION-BRUTEFORCE:` | The two independent brute-force cross-checks on a 128-configuration subset. | Yes — stable. |
| `ARTIFACT:` | Confirms each of the three files above was written (with a row/step count). | Yes — stable (three such lines). |
| `VIGNETTE:` | The greedy reconfiguration's PASS/FAIL verdict (moves executed, Phi before/after). | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every gate above must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle **bit-exact** — a real indexing or
  logic bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp` (they must be
  line-by-line twins).
- **`CORRUPTION-GATE: FAIL` or a `*-BRUTEFORCE: FAIL`:** the fast algorithm and an independent oracle
  disagree, or a designed negative control was not caught — see `../THEORY.md` "How we verify
  correctness" for what each layer is designed to catch.
- **`VIGNETTE: FAIL`:** the greedy reconfiguration made zero moves, or an intermediate state failed
  its own re-verification (which would `exit()` loudly with a distinct stderr message before this line
  ever prints) — see `../src/main.cu` §8.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
