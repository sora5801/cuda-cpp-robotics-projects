# Demo — 11.01 GPU LiDAR simulator: BVH raycasting + beam divergence, intensity, dropout noise

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A GPU LiDAR actually scanning a room.** The demo loads a 2,264-triangle synthetic warehouse (floor,
walls, shelving, crates), builds a BVH over it on the host, and spins a simulated 32-channel LiDAR
through a full 360° sweep (1,024 azimuth steps -> 32,768 beams) from a mounting point at the room's
center — every beam casts a central ray plus 4 divergence-cone subrays through the BVH via
Möller-Trumbore, computes Lambertian intensity from the winning hit, then rolls dropout and range-noise
dice from its own deterministic RNG stream. On the RTX 2080 SUPER reference machine the whole frame
costs ~1 ms of GPU kernel time (~230 ms on one CPU core running the identical algorithm sequentially)
and returns from about 71% of beams (the rest miss over the walls into open sky — an honest consequence
of an unroofed synthetic room, not a bug).

Five checks gate the verdict, in addition to the standard §5 GPU-vs-CPU gate:

1. **VERIFY** — every beam's hit/dropped decision must match the CPU oracle EXACTLY; intensity within
   rel tol 1e-3; range within rel tol 2e-2 (looser than the repo's usual 1e-3 — see
   [`../THEORY.md`](../THEORY.md) "Numerical considerations" for exactly why a handful of beams near
   geometric silhouette edges need that extra headroom, and the measured count that justifies it).
2. **Ground-plane range gate** — a beam aimed at the open floor must return the closed-form range
   `h / sin(|elevation|)` (measured relative error: ~8e-8).
3. **Inverse-square intensity gate** — normal-incidence intensity at range R vs 2R must ratio exactly
   4:1 (measured: 4.000000).
4. **Dropout statistics gate** — the empirical dropout rate over 20,000 i.i.d. beams must match the
   configured probability within a 5-sigma binomial bound.
5. **Frame-level sanity gates** — hit fraction and mean range of the full demo frame must land inside
   documented, MEASURED bounds.

**This demo writes two artifacts** (both git-ignored, regenerated each run):
`out/cloud.csv` — one row per surviving (hit, not dropped) beam: `x_m,y_m,z_m,intensity,ring`, in the
SENSOR frame — and `out/range_image.pgm` — a 1024x32 grayscale range image (closer = brighter, no-return
= black), the LiDAR's own native "picture" (organized point clouds ARE images; see THEORY.md). Plot
`cloud.csv` as a 3-D scatter and you can see the room the simulator built: floor, walls, shelving racks,
crates, and the shadow each object casts in the scan.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number (verify deviations, gate values, frame stats, point counts) — varies by machine/run. | No. |
| `SCENE:`    | Triangle/vertex/material counts loaded from `data/sample/`. | Yes — stable (fixed input). |
| `BVH:`      | Node/leaf/depth counts from the host build (deterministic given the fixed scene). | Yes — stable. |
| `PROBLEM:`  | The exact problem instance (beam count, triangle count). | Yes — stable. |
| `[time]`    | BVH build ms, CPU/GPU frame ms, and a speed-up figure — **teaching artifacts, never benchmark claims** (single-shot, kernel-only vs. one CPU core). | No. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of the §5 GPU-vs-CPU raycast comparison (tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `CHECK:`    | Five independent verdicts: ground-plane range, inverse-square intensity, dropout statistics, frame hit fraction, frame mean range. | Yes — stable (five lines). |
| `ARTIFACT:` | Confirms an output file was written (exact point counts, which vary with hit/dropout decisions, live on the paired `[info]` line instead — see `../src/main.cu`'s "output contract"). | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every check above must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, measured numbers) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL` or any `CHECK: ... -> FAIL`:** a real disagreement — the GPU kernel and the CPU
  oracle disagree beyond documented tolerance, or the physics model itself failed an analytic check.
  Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp` (they are meant to be
  line-by-line twins — diff them); for a physics-gate failure, re-read `../THEORY.md` "How we verify
  correctness" for what each gate assumes about the scene/sensor geometry.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
