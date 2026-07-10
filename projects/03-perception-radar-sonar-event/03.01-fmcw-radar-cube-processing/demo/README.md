# Demo — 03.01 FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs the **complete FMCW radar cube pipeline** on a synthetic 6-target scene: synthesize the
raw ADC cube -> Hann-windowed range FFT -> Hann-windowed Doppler FFT -> fftshift -> noncoherent antenna
integration -> **both** CA-CFAR and OS-CFAR detection -> per-detection zero-padded angle FFT. It then
runs the identical pipeline on the CPU (a hand-rolled O(N^2) DFT oracle) and checks the two
range-Doppler maps agree; checks every injected target was found (within a resolution-cell tolerance)
by OS-CFAR; and checks that CA-CFAR — as designed — **misses** the scene's one closely-spaced weak
target while OS-CFAR finds it, the project's headline comparison (see `../THEORY.md` "The algorithm").

**Artifacts written to `demo/out/` (git-ignored; regenerated every run):**

- `range_doppler.pgm` — a binary PGM (viewable in most image tools, or Python `PIL.Image.open`) of the
  GPU's log-magnitude range-Doppler map. Rows are range bins (near range at the top), columns are
  Doppler bins (receding velocity at the left, approaching at the right — this project's sign
  convention). The 6 targets appear as bright spots; OS-CFAR's detections are marked with a small
  bright cross for visibility.
- `detections.csv` — every detection from BOTH detectors (tagged `CA`/`OS`), with its estimated
  range/velocity/azimuth/power and, where it matched a ground-truth target, the exact per-field error.
  A detection with no match (`matched_target_idx = -1`) is a false alarm.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, per-target match details, exact detection/false-alarm counts, worst VERIFY deviation — these vary in the low digits across GPU architectures (cuFFT's internal algorithm differs sm_75 vs sm_86 vs sm_89) even though the demo is fully deterministic on any ONE machine. | No. |
| `PROBLEM:` / `RESOLUTION:` | The radar configuration and its derived resolution/ambiguity limits — fixed at compile time, identical on every machine. | Yes — stable. |
| `SCENARIO:` | How many targets were loaded and from where. | Yes — stable. |
| `[time]`    | GPU pipeline ms, CPU O(N^2) oracle ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `VERIFY:`   | `PASS`/`FAIL`: does the GPU range-Doppler map match the CPU O(N^2) DFT oracle within tolerance (`../src/main.cu`, `../THEORY.md`)? | Yes — stable (qualitative; the exact deviation number is `[info]`). |
| `GROUND_TRUTH:` | `PASS`/`FAIL`: did OS-CFAR find every injected target within its documented tolerance, with false alarms inside the bound? | Yes — stable (qualitative). |
| `CFAR_COMPARE:` | `PASS`/`FAIL`: does the CA-vs-OS masking comparison show its documented shape on the close-target pair? | Yes — stable (qualitative). |
| `ARTIFACT:` | Confirms `range_doppler.pgm`/`detections.csv` were written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` verdict (all four gates above). The program exits nonzero on `FAIL`. | Yes — stable. |

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
