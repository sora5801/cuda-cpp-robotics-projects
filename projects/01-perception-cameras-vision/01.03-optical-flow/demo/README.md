# Demo — 01.03 Optical flow: pyramidal Lucas-Kanade, Farneback, census-transform flow

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The two implemented milestones of this project's catalog bullet, taught end to end on four
ground-truth-known synthetic frame pairs (Farneback is documented-only — see `../THEORY.md` and README
§13):

1. **DENSE PYRAMIDAL LUCAS-KANADE** — a 3-level image pyramid, per-level Scharr gradients, a 5x5
   structure-tensor + mismatch-vector solve with 3 warped-resampling iterations per level, and
   coarse-to-fine flow-field propagation. Reports a per-pixel CONFIDENCE (the structure tensor's small
   eigenvalue — the aperture problem, made numeric).
2. **CENSUS-TRANSFORM BLOCK-MATCHING FLOW** — a 5x5 (24-bit) census signature per pixel, brute-force
   Hamming block matching over a 13x13 search window with parabolic sub-pixel refinement, and a
   forward/backward (left-right) consistency check producing a validity mask.
3. **VERIFY** five independent GPU-vs-CPU stages (gradient, the full LK pipeline, census transform,
   census match, the full census pipeline) — bit-exact where the math is all-integer, tolerance-checked
   where it is float (see `../THEORY.md`).
4. **GATE** the result against EIGHT independent checks against ANALYTIC ground truth: the basic
   translation sanity gate for both methods, the pyramid's reason to exist (rotation+zoom accuracy AND
   a measured pyramid-vs-no-pyramid advantage ratio), census's brightness-robustness selling point (LK's
   degradation on the same scene is measured and reported honestly, not gated), the zero-motion negative
   control for both methods, and a confidence-mask sanity check (rejected pixels really are worse).

**Artifacts written to `out/`** (git-ignored; regenerated every run):

| File | What it shows |
|------|----------------|
| `flow_lk_rotzoom.ppm` | Dense pyramidal-LK flow on the rotation+zoom pair, HSV-wheel color coded (see below) — the spatially-varying field the pyramid was built to recover. |
| `flow_census_translation.ppm` | Census-transform flow on the translation pair, HSV-wheel color coded — should look like a near-uniform color patch (constant flow), the qualitative signature of "the exact gate". |
| `flow_color_wheel.ppm` | A small (65x65) legend disc: decode any of the above two images by eye. |
| `epe_heatmap_lk_rotzoom.pgm` | Per-pixel endpoint error (LK vs. the analytic rotation+zoom ground truth), grayscale, black=0 px to white=3 px (capped). |
| `confidence_lk_rotzoom.pgm` | The structure tensor's small eigenvalue, normalized against this frame's own peak — black=low confidence (aperture problem), white=high confidence (real 2-D texture). Compare against the heatmap above: low-confidence regions should visibly correlate with high-error regions (the `confidence_mask_sanity` gate makes this numeric). |
| `validity_census_translation.pgm` | The census left-right consistency mask on the translation pair — white=passed, black=failed (occlusion-truncated border, or a genuinely ambiguous local match). |
| `gates_metrics.csv` | One row per measured gate/metric, machine-readable (mirrors the `[info]`/`GATE` lines below). |

### Reading the HSV flow wheel

Both `flow_*.ppm` artifacts and the legend `flow_color_wheel.ppm` use the standard "Middlebury wheel"
optical-flow color convention (see `../README.md` "Prior art"): for a flow vector `(u, v)`,

- **HUE** (color) encodes the flow's DIRECTION — `atan2(v, u)`, wrapped around the color circle. Red
  points right, cyan points left, and so on around the wheel — check `flow_color_wheel.ppm` for the
  exact mapping at your monitor's color rendering.
- **SATURATION** encodes the flow's MAGNITUDE, relative to a fixed cap (12 px/frame here — comfortably
  above this scene's worst-corner ground truth): zero flow is WHITE (fully desaturated), and magnitude
  at-or-beyond the cap is a fully saturated color.
- **VALUE** (brightness) is held at 1 everywhere, so hue and saturation stay legible without a
  brightness gradient fighting for attention.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number (confidence/validity counts, verify diffs, gate statistics) — varies by machine/GPU architecture. | No. |
| `PROBLEM:`  | The exact problem instance (frame size, pyramid depth, census parameters). | Yes — stable (demo runs with no args). |
| `DATA:`     | Confirms the synthetic sample loaded. | Yes — stable. |
| `VERIFY(...): ` | PASS/FAIL verdict that a GPU stage matches its independent CPU reference — never embeds a number (cross-GPU-portability rule). | Yes — stable. |
| `GATE <name>:` | PASS/FAIL verdict of one of the eight independent, ground-truth-based gates. | Yes — stable. |
| `[time]`    | GPU kernel time and CPU reference time — a **teaching artifact, never a benchmark claim** (single-shot; see `../src/util/timer.cuh`). | No. |
| `ARTIFACT:` | Confirms the `out/` files were written. | Yes — stable. |
| `RESULT:`   | The overall PASS/FAIL verdict (every VERIFY and every GATE must pass). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, per-stage `[info]` detail) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `VERIFY(...)` line FAILs:** the GPU result disagreed with the CPU oracle beyond its documented
  tolerance — a real bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`
  for the stage named in the failing line (`gradient`/`lk_flow`/`census_transform`/`census_match`/
  `census_flow`).
- **A `GATE` line FAILs but every `VERIFY` passes:** the GPU and CPU agree with each other, but the
  RESULT doesn't match physical reality — look at the relevant artifact (the heatmap for an LK gate,
  the validity mask for a census gate) and `../src/main.cu`'s gate implementation; this is the class of
  bug a twin comparison structurally cannot catch (see `../THEORY.md` "How we verify correctness").
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
