# Demo — 02.01 Voxel-grid downsampling with GPU spatial hashing

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

It loads the committed 198,534-point synthetic LiDAR scan
([`../data/README.md`](../data/README.md)), downsamples it to a 20 cm voxel grid **two independent
ways** — Method A (an atomic open-addressing GPU hash table) and Method B (a Thrust sort + fixed-order
segmented reduction) — checks each against its own CPU reference, cross-checks the two methods against
each other, runs four geometric/bookkeeping gates, quantifies the hash table's probe-length behavior
and Method A/B's very different determinism stories with **three repeated runs of each method**, and
writes four artifacts to `out/` (git-ignored; regenerated every run):

| Artifact | What it shows |
|----------|----------------|
| `original_topview.ppm` | Orthographic top-view (looking down −z) of all 198,534 input points — a dense, near-saturated cloud of white dots on black. |
| `downsampled_topview.ppm` | The same view of the 7,132 Method-B centroids — visibly far sparser; the room's walls, floor pattern, and the three boxes are still legible, but the near-field oversampling is gone. |
| `probe_length_histogram.csv` | How many atomicCAS linear-probe steps each of the 198,534 inserts needed, split by "normal scan" vs "adversarial" point index range. |
| `gates_metrics.csv` | Every measured number behind the stdout report — tolerances, deltas, load factor, RMS distance, timings — in one machine-readable file. |

The two PPMs are the "before/after" visual the catalog bullet's teaching goal asks for: open them
side by side and the point-count reduction (198,534 → 7,132, a 27.8× reduction on this sample) is
immediately visible as dot density, not just a printed number.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and measured-but-not-gated numbers (determinism_method_a's delta, occupancy_analytics, hash_stats, downsample_quality) — vary by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (point counts, voxel leaf, hash capacity) — fixed by the committed sample, so stable. | Yes — stable. |
| `DATA:`     | Which sample file, and that it is synthetic. | Yes — stable. |
| `VERIFY(...)` | GPU-vs-CPU-twin agreement for keys, Method B (bit-exact), and Method A (tolerance-based). | Yes — stable (PASS/FAIL text only, no numbers). |
| `GATE ...:` | The four independent gates: `cross_method_agreement`, `partition_invariant`, `centroid_containment`, `determinism_method_b`. | Yes — stable. |
| `[time]`    | CPU/GPU timings — a **teaching artifact, never a benchmark claim** (single-shot; see `../THEORY.md`). | No. |
| `ARTIFACT:` | Which files were written to `out/`. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict; the program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, the `[info]`-prefixed measured numbers) are allowed. `#`-prefixed lines in that
file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list). If the failure mentions `CCCL`, `/Zc:preprocessor`,
  or `__cplusplus`, see the detailed comment in `../build/*.vcxproj`'s `CudaCompile` sections — it is a
  known CUDA 13.3 + Thrust + MSVC interaction, already worked around there.
- **A `VERIFY` or `GATE` line prints `FAIL`:** a real bug — the program prints extra detail to stderr.
  Start in `../src/kernels.cu` (the method that failed) and compare against `../src/reference_cpu.cpp`'s
  matching twin.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together (`../src/main.cu`'s "Output contract" comment explains the rule).
