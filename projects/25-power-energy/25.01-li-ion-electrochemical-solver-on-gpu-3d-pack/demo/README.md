# Demo — 25.01 Li-ion electrochemical (SPM) solver + 3D pack thermal simulation + cooling-design sweeps

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Nine stages, in order (full narrative in `../src/main.cu`'s file header):

1. **VERIFY** — one design, 200 mission steps (20 s), the GPU electrochemistry+thermal kernels checked
   against their plain-C++ CPU twins from *identical* per-step inputs.
2. **ANALYTIC_DIFFUSION / ANALYTIC_COULOMB** — a standalone single-particle, constant-flux diffusion
   fixture checked against two closed forms: the quasi-steady surface-minus-average concentration
   `j·R/(5D)`, and exact charge conservation (integrated applied flux = measured mole change).
3. **ANALYTIC_THERMAL** — a standalone uniformly-heated, single-cooling-face pack fixture run to steady
   state, checked against the *exact* energy-balance identity `P = h·A·(T_face − T_coolant)`.
4. **THE SWEEP** — the actual point: all 12 cooling designs (6 convective coefficients × {bottom, side}
   cold plate) driven through the same 20-minute AMR duty-cycle mission, batched into one sequence of
   GPU kernel launches, each design's 24 cells diverging only through their own thermal history.
5. **PHYSICS** — per-design thermal-grid energy conservation (generated − convected − stored ≈ 0),
   checked for every one of the 12 designs.
6. **ARTIFACTS** — three files written to `out/` (git-ignored; regenerated every run):
   - `pack_temps.csv` — per-cell temperature history (24 columns × 2 designs) for the best- and
     worst-peak-temperature designs, decimated to ~200 samples.
   - `design_sweep.csv` — the 12-design comparison table: peak temperature, cell-to-cell spread, and
     end-of-mission voltage spread, per design.
   - `pack_slice.pgm` — a mid-pack (z = 8) temperature slice of the worst design at its own peak step.

**The finding worth reading the numbers for:** because the pack's through-plane conductivity `kz` is
deliberately much lower than its in-plane `kx`/`ky` (a real wound/stacked cell's anisotropy), **bottom**
cooling is internal-conduction-limited — raising `h` from 10 to 500 W/(m²K) barely changes peak
temperature — while **side** cooling is boundary-limited and responds strongly to `h`. The side design
with the highest `h` reaches the *lowest* peak temperature but the *largest* cell-to-cell spread — a
genuine, non-obvious design trade-off between peak-temperature control and pack balance. `design_sweep.csv`
has the numbers; THEORY.md "Where this sits in the real world" discusses the trade-off.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, CFL margins, and every measured number (gate values, design results). | No. |
| `PROBLEM:` / `SCENARIO:` | The exact problem instance. | Yes — stable (demo runs with no args). |
| `[time]`    | CPU/GPU timings and speed-up figures — **teaching artifacts, never benchmark claims**. | No. |
| `VERIFY:` / `ANALYTIC_DIFFUSION:` / `ANALYTIC_COULOMB:` / `ANALYTIC_THERMAL:` / `SWEEP:` / `PHYSICS:` / `ARTIFACT:` | Stage verdicts. | Yes — stable (PASS/FAIL text only, no numbers). |
| `RESULT:`   | Overall `PASS`/`FAIL`. The program exits nonzero on `FAIL`. | Yes — stable. |

No stable line carries a raw floating-point number (CLAUDE.md §12 determinism discipline extended to
FP32 PDE state across platforms — see `src/main.cu`'s "Determinism" note): every measured value lives
on an `[info]`/`[time]` line, and every checked line is a PASS/FAIL verdict with the tolerance named in
the text. The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines are allowed.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).
- **`VERIFY: FAIL`:** the GPU kernels disagree with their CPU twins — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **An `ANALYTIC_*` or `PHYSICS` gate fails:** the solvers agree with each other but not with the
  underlying physics/math — check `../src/main.cu`'s fixture functions against THEORY.md's derivations.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — `expected_output.txt` and `main.cu` drifted apart; fix them together.
