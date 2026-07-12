# Demo — 02.03 Ground segmentation: RANSAC plane fit; Patchwork++-style GPU port

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

It loads the committed 161,836-point synthetic scene ([`../data/README.md`](../data/README.md) — a flat
segment, an 8° ramp, a raised plateau, six standing obstacles, and a floating canopy overhang, every
point exactly labeled ground/not-ground by construction), then:

1. **Milestone 1 (RANSAC)** — fits ONE global plane, twice: once to the **whole scene** (used to
   demonstrate the designed failure), once to a **near-field flat-only crop** (RANSAC's home turf, used
   to validate the algorithm and audit the classical iteration-count formula).
2. **Milestone 2 (CZM)** — the Patchwork++-style concentric-zone model: 160 small local patches, each
   independently fit, with a region-growing rule that lets the recovered ground follow the ramp/plateau.
3. Verifies every GPU stage against an independent CPU twin (8 `VERIFY(...)` lines).
4. Scores every stage against the scene's exact ground truth with **6 independent gates**.
5. Writes four artifacts to `out/` (git-ignored; regenerated every run).

| Artifact | What it shows |
|----------|----------------|
| `topview_truth_ransac_czm.ppm` | Three panels side by side (truth \| RANSAC \| CZM), looking down −z, colored by classification (green = ground, red = not-ground, magenta = canopy in the truth panel only). **The money shot**: the RANSAC panel shows the forward ramp/plateau corridor turned almost entirely red (misclassified) while the CZM panel matches the truth panel's green. |
| `sideview_truth_ransac_czm.ppm` | The same three panels from the side (x,z; z exaggerated ~8x so the ramp's 0.56 m rise is visible) — the ramp and plateau's *profile*, and where each method draws the ground/obstacle line. |
| `czm_patch_stats.csv` | Every one of the 160 patches' zone/sector/ring, pass/fail verdict, point counts, RMS residual, uprightness angle, and whether it used the region-growing height prior. |
| `gates_metrics.csv` | Every measured number behind the stdout report, machine-readable. |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name (varies by machine), and the `slope_accuracy` diagnostic (deterministic given the fixed sample, but informational — not gated). | No. |
| `PROBLEM:`  | The exact problem instance (point counts by category, RANSAC K/threshold, CZM patch geometry) — fixed by the committed sample. | Yes — stable. |
| `DATA:`     | Which sample file, and that it is synthetic. | Yes — stable. |
| `VERIFY(...)` | GPU-vs-CPU-twin agreement, one line per pipeline stage (hypothesis generation, evaluation, refinement — each run twice, full-scene and flat-only — plus patch assignment and patch fit/classify). See `THEORY.md` "How we verify correctness" for which are bit-exact-with-tolerance vs. genuinely independent-and-margined. | Yes — stable (PASS/FAIL text and measured mismatch counts, all deterministic on the fixed sample). |
| `GATE ...:` | The six independent gates: `ransac_flat`, `ransac_formula`, `single_plane_failure`, `czm_recovery`, `overhang`, `obstacle_rejection`. | Yes — stable. |
| `[time]`    | CPU/GPU timings — a **teaching artifact, never a benchmark claim** (single-shot; see `../THEORY.md`). | No. |
| `ARTIFACT:` | Which files were written to `out/`. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict; the program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list). If the failure mentions `CCCL`,
  `/Zc:preprocessor`, or `__cplusplus`, see the detailed comment in `../build/*.vcxproj`'s
  `CudaCompile` sections — a known CUDA 13.3 + Thrust + MSVC interaction, already worked around there
  (this project uses Thrust for the CZM's patch sort + boundary search).
- **A `VERIFY` line prints `FAIL`:** a real bug in the GPU/CPU agreement for that specific pipeline
  stage. Start in `../src/kernels.cu` (the kernel named in the line) and compare against
  `../src/reference_cpu.cpp`'s matching twin — `kernels.cuh`'s file header explains which twins are
  meant to be near-bit-exact vs. independently-derived-and-tolerant.
- **A `GATE` line prints `FAIL`:** the algorithm itself under- or over-performed a measured-then-margined
  floor on the fixed scene — a real regression, not a flaky threshold (nothing here is stochastic at
  the gate level; the RNG seeds and the sample data are both fixed). Check `demo/out/gates_metrics.csv`
  for the exact numbers and `../src/main.cu`'s gate constants for the floor each one compares against.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together (`../src/main.cu`'s "Output contract" comment explains the rule).
