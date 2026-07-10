# Demo — 26.01 Topology optimization (SIMP/level-set) on GPU for lightweight links and brackets — flagship design project

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Six pipeline stages, in order, printed as they run (see `../src/main.cu`'s file header for the
full narration):

1. **Derive** the element stiffness matrix `KE_hat` (2x2 Gauss quadrature, computed at startup —
   no magic numbers) and the density filter's weight table; upload both to GPU constant memory.
2. **VERIFY** — one full SIMP inner iteration (matrix-free CG FEA solve + sensitivity + filter) on
   a small intermediate-density problem, run through both the GPU kernels and the plain-C++ CPU
   oracle; the two must agree within a documented tolerance (the repo's standard §5 gate).
3. **PATCH** — a solid rectangular strip under uniform tension must reproduce the *exact*
   closed-form linear displacement field — the standard FEM correctness check.
4. **BEAM** — a solid cantilever's tip deflection under a point load must match Euler-Bernoulli /
   Timoshenko beam theory within a documented allowance (Q4 element shear locking, honestly named).
5. **MBB** — the classic half-MBB beam: 80 outer SIMP iterations of GPU FEA + sensitivity
   filtering + host Optimality-Criteria updates, writing `demo/out/topology_mbb.pgm` — the famous
   diagonal-strut result, visible on sight if you view the image.
6. **BRACKET** — a robot L-bracket load case (bolted along the top, loaded at the bottom-right
   foot by a motor flange), writing `demo/out/topology_bracket.pgm` — material concentrates along
   a diagonal strut that routes *around* the reentrant corner, exactly as topology-optimization
   theory predicts.

**Artifacts** (git-ignored scratch, written every run): `demo/out/topology_mbb.pgm`,
`demo/out/topology_bracket.pgm` — the two designs, viewable in any image tool that reads PGM
(black = solid material, white = void) — and `demo/out/convergence.csv` (columns
`case,iter,compliance_J,volfrac`, both cases appended) for plotting the optimization trajectory
(README Exercise 1).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, CG iteration counts, measured errors/margins — varies by machine/GPU. | No. |
| `PROBLEM:`  | The exact problem instance (discretization, solver, precision). | Yes — stable. |
| `SCENARIO:` | The MBB/bracket mesh size, material, and volume target loaded from `data/sample/`. | Yes — stable. |
| `VERIFY:` / `PATCH:` / `BEAM:` / `MBB:` / `BRACKET:` | `PASS`/`FAIL` verdict of each of the five independent gates described above. | Yes — stable. |
| `ARTIFACT:` | Confirms the PGM/CSV files were written. | Yes — stable. |
| `RESULT:`   | The aggregate verdict — `PASS` only if every gate above passed. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, measured margins) are allowed. `#`-prefixed lines in that file are comments.
Total run time on the reference machine (RTX 2080 SUPER): **~8 seconds** — two 80-iteration SIMP
optimizations plus three verification stages, comfortably inside the project's documented budget.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** find which of the five `<STAGE>: FAIL` lines fired — `VERIFY` points at a
  GPU-vs-CPU disagreement (start in `../src/kernels.cu`, compare against
  `../src/reference_cpu.cpp`); `PATCH`/`BEAM` point at the solver itself (a real correctness bug,
  since both compare against closed-form physics, not a tolerance-only twin); `MBB`/`BRACKET` point
  at the optimizer (`oc_update`/`run_simp` in `../src/main.cu`).
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
