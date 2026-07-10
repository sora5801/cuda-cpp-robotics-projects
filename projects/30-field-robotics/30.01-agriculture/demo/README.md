# Demo — 30.01 Agriculture: fruit detection + 3D localization + ripeness; weed-vs-crop segmentation at frame rate; per-plant spray targeting; crop-row following; canopy volume from LiDAR; under-canopy navigation; yield mapping

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A complete classical fruit-perception pipeline, GPU vs. CPU, checked against exact 3-D ground
truth.** The demo loads the committed synthetic orchard scene (`rgb.ppm` + `depth.pgm`), runs the
seven-stage pipeline (HSV -> mask -> morphological opening -> connected-component labeling ->
per-component statistics -> robust 3-D localization -> ripeness) on BOTH the GPU and a plain-C++ CPU
oracle, and checks four independent things:

1. **VERIFY** — GPU matches CPU at every stage: HSV/mask by tight tolerance, connected-component
   LABELS by **exact** integer equality (see `../THEORY.md` for why that is a fair, achievable bar
   here), final per-fruit statistics by a small relative tolerance.
2. **DETECT** — the fraction of ground-truth fruit actually found, and how many detections do not
   correspond to any real fruit (honestly including the scene's two designed cross-depth merge cases
   as counted misses/extras — see `../data/README.md`).
3. **LOCALIZE** — how close the predicted 3-D centers and radii are to the true ones (after the
   pipeline's derived surface-to-center depth correction — `../THEORY.md` "The math").
4. **RIPENESS** — rank correlation between predicted and true ripeness (rank, not absolute value —
   `../THEORY.md` explains why that is the honest metric for a hue-only color model).

**This demo writes two artifacts** (git-ignored, regenerated each run):

- `out/detections.pgm` — the scene rendered to grayscale with a ring burned in around every GPU
  detection, for a direct eyeball check against `../data/sample/rgb.ppm`.
- `out/fruit_map.csv` — id, 3-D center, radius, ripeness, pixel count per detection: the seed of the
  documented Milestone 7 "yield mapping" component (README "Overview").

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and the actually-MEASURED numbers behind each gate (detection rate, localization error, ripeness correlation) — varies slightly by machine (see `../src/main.cu`'s "NOTE on determinism"). | No. |
| `PROBLEM:` / `DATA:` | The exact problem instance and data loaded. | Yes — stable. |
| `[time]`    | Per-stage GPU kernel ms, CPU reference ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:` / `DETECT:` / `LOCALIZE:` / `RIPENESS:` | PASS/FAIL verdict against a fixed, documented threshold (never the measured number itself — see `[info]` above). | Yes — stable. |
| `ARTIFACT:` | Confirms the two output files were written. | Yes — stable. |
| `RESULT:`   | Overall verdict: every gate above must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

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
