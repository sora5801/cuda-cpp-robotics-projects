# Demo — 24.01 2D magnetostatic FEA solver on GPU → motor torque-ripple/cogging parameter sweeps

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

One executable, five stages, in order:

1. **VERIFY** — solves one representative motor variant (a non-trivial rotor angle) on the GPU
   batched red-black SOR solver AND on the CPU twin, and requires the two vector-potential fields
   to agree within a documented tolerance (measured worst case: `2.948e-07` Wb/m).
2. **ANALYTIC_AMPERE** — solves a uniform-current annulus in air (no motor geometry) and checks the
   computed azimuthal B field against Ampere's law's closed form in three regions: zero in the bore,
   growing inside the annulus, decaying as 1/r outside.
3. **ANALYTIC_INTERFACE** — solves a straight air/iron interface driven by a current strip and
   checks that the field's NORMAL component is continuous across the interface — the direct
   correctness check on the solver's harmonic-mean face-averaging.
4. **THE SWEEP** — for each of 5 magnet pole-arc fractions, batches all 24 rotor-angle solves for
   that arc fraction into ONE kernel-launch sequence (the project's central GPU lesson), computes
   cogging torque via the Maxwell stress tensor at every angle, and reports which arc fraction
   minimizes PEAK cogging torque — the actual motor-design question the catalog bullet asks.
5. **PHYSICS** — every cogging waveform must integrate to ~zero net torque over the sampled period
   (no net work from cogging) and repeat after one magnet pole pitch (checked with an independent
   solve, not just inferred from symmetry).

**Artifacts written to `out/` (git-ignored, regenerated every run):**

- `field_magnitude.pgm` — |B| over the full 256x256 cross-section for the recommended (minimum-
  cogging) design at a non-trivial rotor angle — "the classic motor-field picture": bright rings at
  the magnets and stator teeth, darker in the air gap and slot openings. View with any PGM-capable
  viewer (GIMP, IrfanView) or convert with ImageMagick (`magick field_magnitude.pgm field.png`).
- `cogging_waveforms.csv` — rotor angle (degrees) vs. torque (N·m/m), one column per swept arc
  fraction — the design plot. Plot it: the arc fraction with the smallest peak-to-peak swing is the
  one `[info] design result:` names.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, scenario path, and every MEASURED number (verify tolerance achieved, Ampere/interface errors, per-arc-fraction peak/mean torque, the recommended design). Varies by machine/build (THEORY.md "Numerical considerations" explains why — FP32 rounding-order differences across compilers/optimization levels). | No. |
| `PROBLEM:` / `SCENARIO:` | The exact problem instance (grid, geometry, materials, sweep plan) — entirely determined by the committed scenario CSV, so stable across machines. | Yes — stable. |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:` / `ANALYTIC_AMPERE:` / `ANALYTIC_INTERFACE:` / `PHYSICS:` | `PASS`/`FAIL` verdicts — the words only, never the underlying measured number (which can shift by a few ulps across builds; THEORY.md quantifies the measured spread). | Yes — stable. |
| `SWEEP:`    | Confirms the sweep ran to completion (arc-fraction count x rotor-angle count). | Yes — stable. |
| `ARTIFACT:` | Confirms both output files were written. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict — `PASS` only if every stage above passed. The program exits nonzero on `FAIL`. | Yes — stable. |

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
