# Demo — 01.16 Checkerboard/ChArUco detection acceleration for auto-calibration rigs

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Runs the full pipeline — saddle-point X-corner response → NMS → sub-pixel refinement → marker-first
grid ordering → Zhang mini-calibration — on the committed 8-view ChArUco calibration rig batch and a
board-free negative-control scene. It:

1. Runs stages 1–3 (response, NMS, sub-pixel refine) on **both** the GPU and an independent CPU
   oracle, on the rig batch **and** the negative control, and verifies they agree (`VERIFY:` line).
2. Orders each view's refined corners into a `7x5` `(i,j)` lattice TWO ways (host, shared code): the
   RETIRED plain-checkerboard walk (kept only as the ambiguity-lesson comparison baseline), and
   MARKER-FIRST ordering — THE pipeline's output of record — which decodes markers independent of
   any global corner walk and anchors their surrounding corners with an absolute `(i,j)` directly.
3. Runs marker decode on **both** GPU and CPU, fed the plain-checkerboard homography, and verifies
   agreement — proving the decode primitive itself correct, independent of which ordering strategy
   the pipeline ultimately uses.
4. Runs Zhang's method on the views whose marker-first ordering came out exact, recovering
   `(fx, fy, cx, cy)`.
5. Checks six independent gates — `corner_accuracy`, `grid_ordering`, `ambiguity_lesson`,
   `occlusion`, `mini_calibration`, `negative_control` — each compared against
   `scripts/make_synthetic.py`'s own recorded ground truth, never against the pipeline's own
   intermediate values.

**Artifacts** (written to `demo/out/`, all git-ignored, regenerated every run):

| File | What it is |
|------|------------|
| `corners_overlay.ppm` | Detected corners + `(i,j)` labels overlaid on view00 (clean baseline) and view06 (the 180-degree ambiguity view), side by side. Green cross + yellow label = a labeled corner; red cross = detected but unlabeled. |
| `refinement_error.csv` | Per matched corner: raw NMS peak position, sub-pixel refined position, and the error against ground truth for both — the before/after refinement comparison `README.md` "Expected output" quotes. |
| `zhang_results.csv` | True vs. recovered `fx, fy, cx, cy, skew`. |
| `gates_metrics.csv` | Every gate's measured value, tolerance, and PASS/FAIL verdict. |

See `README.md` "Expected output" for the actual measured numbers this project's committed sample
produces on the reference machine, and "Limitations & honesty" for exactly which of the 8 views
achieve exact corner ordering and why the others do not.

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
