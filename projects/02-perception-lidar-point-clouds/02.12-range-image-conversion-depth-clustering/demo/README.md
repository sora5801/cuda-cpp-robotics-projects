# Demo — 02.12 Range-image conversion + depth-clustering segmentation

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Loads the committed raw LiDAR scan (`../data/sample/range_image_scene.bin`), converts it to the
organized range image on the GPU, removes ground with a column-wise angle walk, builds the
depth-clustering (Bogoslavskyi-Stachniss beta-criterion) edge graph directly on the image grid — no
3-D neighbor search — clusters it with a generic lock-free GPU union-find, compacts the surviving
obstacle points back to an unorganized cloud, clusters THAT with a voxel-hash Euclidean comparison
pipeline (the SAME union-find kernels), and verifies every stage against an independent CPU reference.

Every pipeline stage prints a `VERIFY(...)` line (GPU vs. an independently-written CPU twin) or a
`GATE ...:` line (a measured, designed teaching assertion about the scene) — see
[`../README.md`](../README.md) "Expected output" for what each one means and the exact numbers a
reference run produced. The learner should notice, in order:

1. `VERIFY(range_image)` — the range-image conversion round trip is bit-exact.
2. `GATE ground_removal` — the flat-ground column walk recovers ground with high precision/recall.
3. `GATE partition_vs_truth` — depth clustering recovers the isolated objects (person, big_box,
   far_pole) almost exactly.
4. `GATE depth_gap_showcase` — **the headline result**: the person standing 0.19 m in front of a wall
   stays a SEPARATE cluster under depth clustering (any range), but MERGES under fixed-radius
   Euclidean clustering (its 0.40 m tolerance is bigger than the gap).
5. `GATE grazing_fragmentation` — **the honest weakness**: a long wall viewed at shallow (grazing)
   incidence fragments into >= 3 depth clusters even though it is one object — the beta criterion's
   documented failure mode, measured, not asserted.
6. `GATE timing_payoff` — both clustering paths' GPU time is measured (`[time]` lines below it show
   the actual numbers); the image-native path has no neighbor search to pay for.

**Artifacts** written to `demo/out/` (see that folder after any run): `range_image.pgm` (the sensor's-
eye view — the signature visual, grayscale = range), `truth_labels.ppm` / `depth_cluster_labels.ppm` /
`euclid_cluster_labels.ppm` (colored label maps — compare the three side by side), `beta_angle_map.csv`
(the beta criterion's actual value at every ring of one azimuth-adjacent column pair straddling the
person/wall boundary), and `gates_metrics.csv` (every measured number behind every line above).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, the resolved sample path, and two informational (but deterministic) reports (`wall_behind` occlusion split, `thin_pole` min-size trade) — printed for a human to read, deliberately not part of the diff contract. | No. |
| `PROBLEM:` / `DATA:` | The exact problem instance and the loaded sample's point count. | Yes — stable (demo runs with no args). |
| `VERIFY(...)` | GPU result vs. an independently-written CPU twin, for one pipeline stage. | Yes — stable. |
| `GATE ...:` | A measured, designed teaching assertion about the committed scene. | Yes — stable (the pass/fail verdict and any embedded COUNTS; embedded timings are deliberately excluded from gate lines — see `timing_payoff`). |
| `[time]`    | GPU (cudaEvent) and CPU wall-clock milliseconds per stage — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs; see `../THEORY.md`). | No. |
| `ARTIFACT:` | Which files were written to `demo/out/`. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every `VERIFY`/`GATE` above must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

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
