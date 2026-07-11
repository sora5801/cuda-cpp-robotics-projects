# Demo — 01.24 Transparent/reflective object detection via polarization imaging

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads the committed DoFP polarization mosaic (`../data/sample/mosaic.pgm`: a matte
background plus a glass pane, a curved glass dome, and a brushed metal bar), runs the five-stage
pipeline (demosaic -> Stokes -> DoLP/AoLP -> Malus self-consistency residual -> detection) on BOTH the
GPU and an independent CPU oracle, VERIFIES every stage agrees, then runs six physics/detection GATEs
that never touch either twin — most importantly `fresnel_anchor` (does the MEASURED polarization on the
glass pane match the closed-form Fresnel prediction?) and `detection` (does DoLP-based detection find
the glass **and** does plain-intensity detection MISS it — the reason this project exists).

Six artifacts land in `out/` (git-ignored, regenerated every run):

| Artifact | What it shows |
|----------|----------------|
| `intensity_s0.pgm` | Plain intensity (S0). The glass pane and dome are **invisible** here — built that way on purpose. |
| `dolp.pgm` | Degree of linear polarization, scaled to `[0,255]`. The glass objects **glow**; so does the metal bar, with a visibly different pattern. |
| `aolp_vis.ppm` | Angle of linear polarization as color (hue = 2 x AoLP, saturation = 1, brightness = DoLP). The dome shows a radial "polarization donut"; the metal bar shows one constant hue. |
| `detection_overlay.ppm` | Side-by-side: left half = intensity-only detection (red) — misses the glass; right half = DoLP-based detection (green) — finds it. Same S0 grayscale base both halves. |
| `brewster_curve.csv` | Closed-form Fresnel DoLP vs. incidence angle, 5-85 deg — the peak sits at Brewster's angle, `atan(1.5) = 56.31 deg`. |
| `gates_metrics.csv` | Every GATE's measured value, tolerance, and PASS/FAIL, in one table. |

**What to notice:** open `intensity_s0.pgm` and `dolp.pgm` side by side — the pane and dome are
essentially indistinguishable from their surroundings in the first and unmistakable in the second.
That contrast is the entire didactic point of this project.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number (VERIFY diffs, gate metrics) — varies by machine/run. | No. |
| `PROBLEM:` / `DATA:` | The exact problem instance and what the sample data is. | Yes — stable (demo runs with no args). |
| `[time]`    | GPU pipeline ms and CPU reference ms — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY(stage):` | `PASS`/`FAIL` — does the GPU stage agree with its independent CPU twin within the documented tolerance (`../src/main.cu`)? | Yes — stable. |
| `GATE name:` | `PASS`/`FAIL` — does the pipeline's OUTPUT agree with physics/ground truth (never routed through either twin)? | Yes — stable. |
| `ARTIFACT:` | Which file was written to `out/` and a one-line caption. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every VERIFY and every GATE must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

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
