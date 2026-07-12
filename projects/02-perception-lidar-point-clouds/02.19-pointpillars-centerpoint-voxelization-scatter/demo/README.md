# Demo — 02.19 PointPillars/CenterPoint voxelization + scatter kernels feeding TensorRT

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The full pillarization → PFN-lite → scatter → toy-detection-head pipeline, end to end, on a synthetic
80×80 m bird's-eye-view scene (6 hollow-box "cars" + ground + clutter + one deliberately pillar-cap-
overflowing cluster — `../data/README.md`). No TensorRT is required or used (`../README.md` §Build) —
the demo proves the CUDA voxelization/scatter kernels are correct and closes the loop with a small
hand-designed head so a learner sees an end-to-end result without needing the TensorRT SDK installed.

In order, the demo:

1. **VERIFY(keys)** — GPU-transcribed pillar/voxel keys vs. the CPU's shared-formula twin, bit-exact.
2. **GATE cap_truncation** — the determinism study: sorted (Method B) binning run 3×, atomic (Method A)
   binning run 3× same-order and 3× with shuffled input order, all measured against the cap-stress
   pillar's known 60-point overflow. See `../src/kernels.cuh`'s file header for why this is the
   project's central lesson.
3. The **production pipeline** (Method B, deterministic): sort+compact → binning → per-pillar stats →
   9-D feature augmentation → the fixed PFN-lite → scatter into a dense `[6,200,200]` canvas → a
   roundtrip gather → two hand-designed 3×3 conv passes + an occupancy gate → peak extraction + NMS.
4. The **independent CPU reference** pipeline on the same data, and `VERIFY(binning/pfn/scatter/head/
   peaks)` — the GPU-vs-CPU §5 gate, stage by stage.
5. **GATE layout_roundtrip**, **GATE feature_semantics** (a hand-computed 3-point pillar, the "free
   exactness anchor"), **GATE detection_closure** (every truth car found, zero false peaks).
6. `[info]` **sparsity_economics**, **pillar_vs_voxel** (the PointPillars-vs-CenterPoint memory/time
   trade), and **trt_handoff** (the exact tensor shapes a TensorRT engine — project
   [`12.01`](../../../12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/README.md) — would
   ingest, documented-only, no fabricated numbers).

**Artifacts** (written to `out/`, git-ignored):

| File | What it shows |
|------|----------------|
| `occupancy.pgm` | The BEV occupancy channel (channel 0 of the scattered canvas) — a grayscale top-down map; the 6 cars and the isolated cap-stress pillar are visible as bright clusters. |
| `heatmap.pgm` | The final detection head response, with a small bright 3×3 marker drawn at every surviving peak (post-NMS) — the 6 detections should sit on/near the 6 cars. |
| `feature_stats.csv` | Per-channel (x,y,z,intensity,xc,yc,zc,xp,yp) min/mean/max over every occupied pillar's kept points — a sanity check on the augmented feature tensor's scale. |
| `gates_metrics.csv` | Every measured number behind every VERIFY/GATE/`[info]` line above, in one place. |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, resolved data path, and measured diagnostic numbers (including the atomic-binning nondeterminism variance, which is *supposed* to vary run to run — see `../src/kernels.cuh`). | No. |
| `PROBLEM:` / `DATA:` | The exact problem instance and committed sample identity. | Yes — stable. |
| `VERIFY(...)` | A GPU-vs-CPU agreement check for one pipeline stage. | Yes — stable (the `PASS`/`FAIL` text; the `[info]` line with the measured diff is not). |
| `GATE ...:` | An independent correctness/behavioral check that does not merely re-run the same code twice. | Yes — stable. |
| `[time]`    | CPU vs. GPU timings — a **teaching artifact, never a benchmark claim** (different scale of work between the two paths; single-shot numbers). | No. |
| `ARTIFACT:` | Confirms the `out/` files were written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines (timings,
device info, the nondeterminism-measurement `[info]` lines) are allowed and expected. `#`-prefixed lines
in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** a `VERIFY` or `GATE` line above did not pass — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp` for the failing stage; the
  `[info]` line immediately above each `VERIFY`/`GATE` line prints the measured number that decided it.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
