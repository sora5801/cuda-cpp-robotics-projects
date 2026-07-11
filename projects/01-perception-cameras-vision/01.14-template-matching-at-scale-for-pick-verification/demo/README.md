# Demo — 01.14 Template matching (NCC) at scale for pick verification

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo checks a synthetic 24-slot pick-and-place tray against 15 templates (3 machined-part shapes x
5 pre-rotated angles) using zero-normalized cross-correlation (NCC), searched over a +-8 px window per
slot — 104,040 individual NCC evaluations, computed three GPU ways (naive re-scan, integer sum-table,
sum-table + shared memory) and cross-checked against an independent CPU oracle. It then classifies
every slot **OK** / **WRONG_PART** / **EMPTY**, recovers the placement offset of matched parts, and
runs 5 independent gates that each teach one lesson: `variant_consistency` (the 3 GPU kernels agree),
`classification` (every one of the 24 slots is correctly verdicted), `localization` (the recovered
offset matches the applied one), `rotation_lesson` (a single fixed-angle template score falls below the
confidence threshold on the rotated slot; the 5-angle rotation SET recovers it), and
`illumination_robustness` (NCC still confidently matches the shadowed slot; a plain SSD score for the
same match would reject it — the designed NCC-vs-SSD comparison).

**Artifacts written to `demo/out/`:**

| File | What it shows |
|------|----------------|
| `tray_overlay.ppm` | The tray, with a colored box per slot (green=OK, orange=WRONG_PART, blue=EMPTY) and a crosshair at the recovered match position. |
| `score_map_rotated_slot.pgm` | The best-over-rotation-set NCC score at every searched offset, for the `rotated` cohort slot — a visual "how confident was the search at each candidate position" map. |
| `score_vs_angle.csv` | The measured NCC score vs. rotation-template angle curve for the `rotated` cohort slot — the rotation_lesson gate's full picture, not just its two asserted points. |
| `per_slot_scores.csv` | Every slot's cohort, truth verdict, computed verdict, and all measured scores. |
| `gates_metrics.csv` | Every gate's measured value(s), bound(s), and pass/fail. |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, per-slot detail, and measured numbers (scores, timings) — varies by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, parameters). | Yes — stable (demo runs with no args). |
| `DATA:`     | Confirms the tray/templates/truth sample loaded and describes the 6 designed cohorts. | Yes — stable. |
| `VERIFY:`   | GPU-vs-CPU agreement, per twinned stage (integral images, window statistics, NCC scores). | Yes — stable. |
| `[time]`    | Per-stage GPU/CPU timings and the naive-vs-sum-table-vs-shared speed-up figures — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `GATE <name>:` | `PASS`/`FAIL` for each of the 5 independent gates (see "What the demo demonstrates" above). | Yes — stable. |
| `ARTIFACT:` | Confirms the 5 `demo/out/` files were written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the WHOLE run (verify stage + every gate + artifacts). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** either a `VERIFY:` line failed (the GPU result disagreed with the CPU oracle — a
  real bug; start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`) or a `GATE`
  line failed (the verified scores were classified/localized/compared incorrectly — start in
  `../src/main.cu`'s classification/gate logic, and check `demo/out/per_slot_scores.csv` for the exact
  slot and score that disagreed with `data/sample/truth.csv`).
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
