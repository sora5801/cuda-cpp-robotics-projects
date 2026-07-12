# Demo — 02.18 Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Three independent weather captures (SNOW, RAIN, DUST) of the *same* static scene — a ground plane,
two walls (one at ~8 m, one at ~32 m), and a parked car — each ray-cast by a real 16-beam spinning
LiDAR model and overlaid with a physically-derived Beer-Lambert scatterer field (`../scripts/
make_synthetic.py`). Three filters are run against every scan: SOR (the generic statistical-outlier
baseline), DROR (Dynamic Radius Outlier Removal), and LIOR (Low-Intensity Outlier Removal). The demo:

1. **Verifies** all three filters' GPU kernels against independently-typed CPU twins on the SNOW
   scan (representative — no kernel branches on which weather scan it is fed): SOR's mean-K-nearest
   distance (tight float tolerance) and DROR's/LIOR's neighbor counts (exact integers), each followed
   by an exact classify-stage check.
2. **Runs 12 gates** against ground truth loaded from `data/sample/points.csv` (never seen by the
   filtering kernels themselves): DROR/LIOR precision+recall floors on the snow and rain cohorts, the
   real-point-preservation floors, and the headline `sor_far_range_failure` gate — SOR must fail
   badly (>= 35% false-removal) on far-range REAL points while DROR must not (<= 15%), the SAME
   cohort, both directions asserted in one gate.
3. **Measures, without gating, the honest hard case**: `dust_plume_honesty` reports DROR's and
   LIOR's precision/recall inside the dense dust-plume core — by design, this project does not
   assert a performance floor there (see THEORY.md for why a sufficiently dense scatterer field can
   statistically resemble a real surface).
4. **Writes three artifacts** to `demo/out/` (git-ignored, regenerated every run):
   - `triptych_snow.ppm` — three top-view panels side by side. LEFT is every point in the snow scan,
     colored by ground truth (light gray = real surface, cyan = snow scatterer speckle) — you can
     see the near wall, far wall, and car as solid gray shapes, with cyan snow speckle scattered
     everywhere including right on top of them. MIDDLE is DROR's cleaned result (only retained
     points; a retained snowflake — a false negative — is shown honestly in orange, not hidden).
     RIGHT is LIOR's cleaned result, same convention. Open the PPM with any image viewer that reads
     the format (GIMP, IrfanView, `pillow`'s `Image.open`), or convert it
     (`magick triptych_snow.ppm triptych_snow.png`).
   - `range_stratified.csv` — near/mid/far real-point false-removal rates for all three filters, all
     three weather scans: the density story made visible (SOR's false-removal rate climbs sharply at
     far range; DROR's stays low across every band).
   - `gates_metrics.csv` — every gate's measured value, threshold, and verdict, for the record.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, data paths, and every MEASURED number (precision/recall percentages, neighbor counts, range-stratified rates) — informative but not diffed, per the output contract in `../src/main.cu`. | No. |
| `PROBLEM:`  | The exact problem instance (scan count, beam budget, filter list). | Yes — stable (demo runs with no args). |
| `VERIFY:`   | GPU-vs-CPU agreement verdicts for each filter's statistic and classify stages. | Yes — stable. |
| `[time]`    | CPU vs. GPU kernel timing — a **teaching artifact, never a benchmark claim**. | No. |
| `GATE:`     | Pass/fail verdict for each of the 12 gates (measured numbers live on the paired `[info]` line just above). | Yes — stable. |
| `ARTIFACT:` | Confirms each of the three files above was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL`. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, every `[info]` measurement) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `VERIFY:` line fails:** a filter's GPU kernel disagreed with its CPU oracle — a real bug. Start
  in `../src/kernels.cu` and compare the matching function side by side against
  `../src/reference_cpu.cpp`.
- **A `GATE:` line fails but every `VERIFY:` line passes:** the filters are internally consistent but
  the measured behavior on this scene crossed a threshold — read the paired `[info]` line for the
  actual number, and see README "Expected output" / THEORY.md for what each gate's threshold means
  and how it was set.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
