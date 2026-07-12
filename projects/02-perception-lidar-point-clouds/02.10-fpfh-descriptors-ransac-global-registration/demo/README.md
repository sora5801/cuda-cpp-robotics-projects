# Demo — 02.10 FPFH descriptors + RANSAC global registration

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads three synthetic (source, target) LiDAR scan pairs of one room, related by a known
140-degree/8-meter transform (`data/README.md`), computes FPFH descriptors for every cloud on the GPU,
runs the full pipeline (descriptor matching → RANSAC hypothesis farm → inlier refit → point-to-plane
ICP polish) on each pair, then reports:

- **`VERIFY(...)` lines** — GPU vs. independent-CPU agreement for every stage (KNN, normals, SPFH,
  FPFH, matching, RANSAC hypotheses, the RANSAC refit, the ICP linear system).
- **`GATE ...` lines** — the independent, ground-truth checks that are this project's real teaching
  payoff: `descriptor_invariance` (FPFH computed independently in each scan's own frame agrees for the
  SAME physical point — pose-invariance, measured), `registration_recovery` (the recovered pose matches
  the TRUE 140deg/8m transform, from a cold start), `icp_negative_control` (local ICP alone, run from
  identity with no RANSAC, provably FAILS at this relative pose — proving global registration earns its
  keep), and `ransac_formula` (the classical RANSAC iteration-count formula, checked against the
  measured correspondence inlier ratio).
- **`[info]` lines** for every measured number behind those verdicts (percentages, angular/translation
  errors, the low-overlap stress cohort's honestly-reported result) — not diffed, so the demo's checked
  contract survives running on a different GPU architecture.

**Artifacts** written to `demo/out/`: `topview_before.ppm`/`topview_after.ppm` (the classic
before/after registration top-view — source in red, target in blue, **purple where both clouds land
on the same pixel** (the visual signature of correct alignment: corresponding points differ by only
~1 cm of sensor noise, far under one pixel at this image's scale, so successful registration turns
large red/blue regions into purple), before = raw unaligned local frames, after = the final recovered
alignment), `descriptor_distance_histogram.csv` (FPFH L2 distance
for ground-truth-matched vs. random point pairs — the separability visual behind the ratio test and
`descriptor_invariance`), and `gates_metrics.csv` (every measured number in machine-readable form).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number (percentages, errors, counts) — deliberately not diffed so the contract survives a different GPU architecture's atomic-reduction order. | No. |
| `PROBLEM:` / `DATA:` | The problem instance and the three committed scan pairs' sizes/overlap/noise. | Yes — stable (demo runs with no args, data is committed). |
| `VERIFY(...):` | `PASS`/`FAIL` — GPU vs. independent-CPU agreement for one pipeline stage. | Yes — stable (verdict + compile-time thresholds only; the measured agreement percentage is a companion `[info]` line). |
| `GATE ...:` | `PASS`/`FAIL` — an independent ground-truth/invariant check (never GPU-vs-CPU). | Yes — stable, same split as `VERIFY`. |
| `ARTIFACT:` | Which files were written to `demo/out/`. | Yes — stable. |
| `[time]`    | Pipeline timing — a **teaching artifact, never a benchmark claim**. | No. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict (every `VERIFY` and gated `GATE` must pass). The program exits nonzero on `FAIL`. | Yes — stable. |

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
