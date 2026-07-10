# Demo — 05.01 TSDF fusion (KinectFusion clone) + marching-cubes mesh extraction

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A dense 3-D reconstruction, fused from scratch and turned into a mesh, in well under a second of GPU
time.** The demo renders 24 synthetic depth frames of a sphere floating above a ground plane (by
closed-form ray casting — no downloads, no stored images), fuses all of them into a 128³-voxel truncated
signed distance field on the GPU, checks the result against the scene's *exact* analytic signed distance
function (real ground truth, not a fixture), and extracts an explicit triangle mesh from the fused field
with marching cubes. Watch for:

- **`VERIFY: PASS`** — the GPU integration kernel and its plain-C++ CPU twin agree on a 4-frame subset
  to `0.0` (bit-identical; both paths execute the same IEEE-754 operations in the same order).
- **`GROUND TRUTH: PASS`** — the fully fused field matches the scene's real SDF within documented
  bounds. The `[info]` lines right above it print the measured mean/max error in two shells around the
  surface — read them to see the actual numbers this project is checking, not just the verdict.
- **`MESH: PASS`** — the GPU's triangle count matches an independent CPU recount *exactly*, sits in a
  wide sanity range, and every vertex lands on the analytic surface within 2 cm.

**This demo writes two artifacts** into `out/` (git-ignored, regenerated every run):

- **`mesh.obj`** — the extracted iso-surface, in plain Wavefront OBJ (open it in Blender, MeshLab, or
  any 3-D viewer). Un-indexed (every triangle carries its own 3 vertices) — README Exercise 5 adds
  vertex welding.
- **`tsdf_slice.pgm`** — a vertical (x–z) slice through the fused volume at the sphere's center, as an
  8-bit grayscale image (any image viewer opens PGM, or convert with any tool). **Black** = never
  observed by any camera; **dark → light** = inside the surface → free space; the **boundary** between
  dark and light *is* the zero crossing — the surface itself. You should see the sphere's circular
  cross-section, the flat line of the ground plane, and a black wedge of never-observed space in the
  sphere's "shadow" directly beneath it (no camera pose in this orbit can see underneath the sphere).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, sample-file path, and every numeric diagnostic (error statistics, mesh triangle counts, final voxel counts) — varies by machine and, for the error statistics, is the actual measured evidence behind each `PASS`. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, parameters). | Yes — stable (demo runs with no args). |
| `SAMPLE:`   | The committed camera path's shape (frame count, synthetic label). | Yes — stable. |
| `[time]`    | Render time, CPU/GPU integration time, marching-cubes kernel time — **teaching artifacts, never benchmark claims** (single-shot, one machine, kernel-only where labeled). | No. |
| `VERIFY:` / `GROUND TRUTH:` / `MESH:` | The three independent checks THEORY.md's "How we verify correctness" explains. | Yes — stable. |
| `ARTIFACT:` | Confirms `mesh.obj` / `tsdf_slice.pgm` were written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

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
