# Demo — 29.05 Ultrasound: GPU beamforming, elastography, image-based servoing

**EDUCATIONAL / SYNTHETIC ONLY** — every scatterer imaged below is numerically generated; nothing
here is patient data, and nothing this demo prints or writes is a diagnostic or therapeutic claim.

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads a synthetic point-scatterer phantom (`data/sample/phantom.csv`: 9 wire targets in
a "+" cross pattern, a high-scattering inclusion, and 18,000 background speckle scatterers),
simulates the channel data a 64-element linear array would record from it, then reconstructs a
B-mode image with the GPU delay-and-sum beamforming pipeline (`src/kernels.cu`) — verified against
a plain-C++ CPU oracle at every stage, and against four physics-derived ground-truth checks (wire
localization, measured-vs-derived resolution, inclusion contrast, and a delay-formula sanity
check). See `../THEORY.md` and `../README.md` "Expected output" for what each check means and the
measured numbers behind it.

**Artifacts written to `out/` (git-ignored; regenerated every run):**

- **`out/bmode.pgm`** — the B-mode image itself: a binary PGM (any image viewer that reads PGM,
  or ImageMagick/PIL, opens it directly), rows = depth (10–30 mm, shallow at top), columns =
  lateral position (−9.6 to 9.6 mm). Each localized wire target is marked with a small bright
  crosshair. Look for: the "+" cross pattern of wire targets, a subtly brighter patch in the
  upper-left where the inclusion sits, and speckle texture filling the rest of the field — THIS is
  the "ultrasound image" the project's catalog bullet promises.
- **`out/psf_profile.csv`** — the isolated center wire's lateral and axial dB profiles (columns
  `axis,position_mm,db_relative`; `axis` is `lateral` or `axial`). Plot `db_relative` vs.
  `position_mm` filtered by `axis` to see the point-spread function directly and the −6 dB
  crossings the RESOLUTION gate measures (README Exercise 1).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, derived-resolution estimate, per-stage measured deviations, per-wire localization detail, measured resolution/contrast/delay numbers — all vary by machine/GPU architecture or are simply too detailed for a stable line. | No. |
| `PROBLEM:`  | The exact problem instance (array/pulse/grid parameters — compile-time fixed). | Yes — stable. |
| `SCENARIO:` | The loaded phantom's scatterer counts (deterministic given the fixed seed). | Yes — stable. |
| `[time]`    | Channel-data synthesis, GPU pipeline, and CPU reference timings, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `VERIFY:`   | PASS/FAIL verdict of the three-stage GPU-vs-CPU beamforming comparison (tolerances documented in `../src/main.cu` and `../THEORY.md`). | Yes — stable. |
| `LOCALIZATION:` | PASS/FAIL: every wire target found within one resolution cell of its true position. | Yes — stable. |
| `RESOLUTION:` | PASS/FAIL: measured PSF width matches the derived formulas within a documented factor. | Yes — stable. |
| `CONTRAST:` | PASS/FAIL: the inclusion is measurably brighter than background speckle. | Yes — stable. |
| `DELAY_CHECK:` | PASS/FAIL: the DAS delay formula matches an independent closed-form re-derivation. | Yes — stable. |
| `ARTIFACT:` | Confirms `bmode.pgm`/`psf_profile.csv` were written. | Yes — stable. |
| `RESULT:`   | Overall PASS/FAIL — the program exits nonzero on FAIL. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, per-wire detail) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU beamforming result disagreed with the CPU oracle at one of the three
  stages — a real bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`
  stage by stage (DAS, then envelope, then log compress).
- **`LOCALIZATION:`/`RESOLUTION:`/`CONTRAST:`/`DELAY_CHECK: FAIL`:** the beamformer runs and agrees
  with its own CPU oracle, but the *physics* is wrong — check `../src/kernels.cuh`'s array/pulse
  constants and the delay/apodization formulas in `das_kernel` against `../THEORY.md` "The math".
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
