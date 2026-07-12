# Demo — 02.05 KD-tree or LBVH construction + KNN/radius search on GPU

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

It loads the committed point cloud (199,404 points, [`../data/README.md`](../data/README.md)),
builds a **linear BVH from scratch every run** — Morton codes → sort → Karras's parallel radix-tree
construction → bottom-up AABB propagation — verifies every build stage against an independent CPU
twin, then runs 2,000 radius-search and K-NN (K=8) queries through the tree, cross-checks them against
an independent fixed-radius voxel-hash baseline (the technique 02.01/02.04 already teach in this
domain) AND against a tree-free O(N·Q) brute-force oracle, gates a designed density-contrast scenario,
and writes four artifacts to `out/` (git-ignored, regenerated every run):

| Artifact | What it shows |
|----------|----------------|
| `topview_density_contrast.ppm` | Top-view (looking down −z) of the whole point cloud in dim gray, with the DENSE-cluster query's radius-search result highlighted in red (~1,200 points, tightly packed) and the SPARSE-region query's KNN result highlighted in yellow (8 points, scattered far away) — both query locations marked with a cyan cross. The one image that makes "radius search finds nothing here, KNN still finds 8" visible, not just numeric. |
| `traversal_stats.csv` | Per-query BVH traversal cost: nodes visited and stack high-water mark, for both radius search and KNN, all 2,000 queries. |
| `timing.csv` | Every measured build-stage and query-throughput timing, GPU and CPU. |
| `gates_metrics.csv` | Every measured number behind every `VERIFY`/`GATE`/`[info]` line. |

The money numbers on the committed data: a radius search at `r = 0.5 m` centered on the dense cluster
returns **1,215 neighbors**; the SAME radius centered in the sparse region returns **0**; KNN (K=8)
returns exactly **8** at both locations, just much farther away at the sparse one. On GPU query
throughput, the voxel-hash baseline (~19,400 q/s) modestly beats the BVH radius search (~11,600 q/s) at
this uniform small radius — the case fixed-radius hashing was built for — while the BVH is the *only*
one of the two that can answer a KNN query at all (~70,500 q/s). Exact figures vary by GPU/run and are
printed as `[time]`/`[info]` lines, never diffed.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|------------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, Morton-locality measurement, traversal-stats summary. | No. |
| `PROBLEM:`  | The exact problem instance (point/query counts, K, radius) — fixed by the committed sample, so stable. | Yes — stable. |
| `DATA:`     | Which sample file, and that it is synthetic. | Yes — stable. |
| `VERIFY(...)` | GPU-vs-independent-CPU agreement for Morton codes, sort order, radix-tree topology, AABBs, BVH radius search, BVH KNN, and voxel-hash radius search. | Yes — stable (PASS/FAIL text only). |
| `GATE ...:` | `tree_validity` (structural invariants), `hash_vs_bvh_agreement` (the two techniques agree), `brute_force_anchor` (the independent, tree-free O(N·Q) oracle), `density_contrast` (the designed dense-vs-sparse scenario). | Yes — stable. |
| `[time]`    | CPU/GPU build and query timings — a **teaching artifact, never a benchmark claim** (single-shot; see `../THEORY.md`). | No. |
| `ARTIFACT:` | Which files were written to `out/`. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict; the program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list). If the failure mentions `CCCL`,
  `/Zc:preprocessor`, or `__cplusplus`, see the detailed comment in `../build/*.vcxproj`'s
  `CudaCompile` sections — a known CUDA 13.3 + Thrust + MSVC interaction, already worked around there.
- **A `VERIFY` or `GATE` line prints `FAIL`:** a real bug — the program prints extra detail to stderr.
  Start in `../src/kernels.cu` (the stage that failed) and compare against `../src/reference_cpu.cpp`'s
  matching twin. If `VERIFY(aabb)` specifically fails, re-read `propagate_aabb_kernel`'s
  `__threadfence()` comment in `kernels.cu` — a missing fence there is exactly the bug this project
  shipped with during development, caught by `GATE brute_force_anchor`.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together (`../src/main.cu`'s "Output contract" comment explains the rule).
