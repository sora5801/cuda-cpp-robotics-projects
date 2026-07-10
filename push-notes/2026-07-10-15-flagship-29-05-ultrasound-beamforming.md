# Push note — 2026-07-10-15: flagship 29.05 ultrasound beamforming

> Push-note per CLAUDE.md §7.1 — written **before** and included **in** the push, so the
> repository always explains its own latest state.

## Summary

Flagship 29.05 — **ultrasound: GPU delay-and-sum B-mode beamforming** — is done (**31/505; 31 of
36 flagships**). A 64-element plane-wave acquisition of a fully synthetic phantom (9 wire targets,
a scattering inclusion, 18,000 speckle scatterers) is beamformed pixel-parallel on the GPU
(≈1 ms vs ≈112 ms CPU), demodulated, and log-compressed into a genuine B-mode image. The project's
spine is resolution physics: derive λ·F# lateral and pulse-length/2 axial resolution, then
*measure* them from the point-spread function — every wire localizes within one derived resolution
cell (worst 0.18 mm), and the measured/derived ratios are physically attributed (Hann mainlobe
widening), not hand-waved. Speckle is taught as what it is — deterministic interference, not
noise — and its grain size matches the measured resolution. Medical framing per §8 throughout:
educational only, synthetic data only, no diagnostic claims; elastography and image-based servoing
ship as the bundle's documented milestones (README §13).

## What changed

- **[projects/29-medical-bio-robotics/29.05-ultrasound/](../projects/29-medical-bio-robotics/29.05-ultrasound/)** —
  complete: DAS kernel (exact two-way delays, linear interpolation, f-number-limited Hann
  aperture), quadrature demodulation + constant-memory FIR envelope, log compression, CPU twin,
  five gate families, B-mode PGM + PSF-profile artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 29.05 → `done` (**31/505**).

## New projects (didactic blurbs)

**29.05 — Ultrasound beamforming** (★ beginner, domain 29, flagship). How sound becomes an image:
pulse-echo physics, geometric delays as the entire "lens", why interpolation quality matters, why
apodization trades resolution for sidelobes, and why plane-wave transmit is the modern ultrafast
choice (the robotics tie: kHz frame rates are what image-based servoing needs — the documented
milestone 3). Two honest engineering stories kept: a tolerance design bug near signal nulls
(absolute vs relative-with-floor — the right call derived), and wire amplitudes re-tuned after
*measured* speckle-interference localization failures. The single most interesting thing to look
at: `demo/out/bmode.pgm` — a real ultrasound image from first principles.

## How to build & run

```powershell
projects\29-medical-bio-robotics\29.05-ultrasound\demo\run_demo.ps1
# then open demo\out\bmode.pgm and plot demo\out\psf_profile.csv
```

## What to study here

`THEORY.md` §The problem (speckle as interference — read this first) → the resolution derivations
→ `src/kernels.cu` (the DAS kernel's delay geometry). First exercise: drop the f-number to 1.0
and watch lateral resolution improve while sidelobe clutter grows — the aperture trade, felt.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 10 stable lines matched, exit 0.
- **GPU-vs-CPU gates (three stages):** RF 1.5e-3, envelope 3.0e-4, dB 0.013 (absolute tolerances,
  null-crossing rationale documented).
- **Physics gates:** all 9 wires localized ≤ 0.18 mm (tol = one resolution cell, 0.462 mm);
  measured PSF widths within the documented factor of derived formulas with the ratio explained;
  inclusion contrast margin 7.6 dB (min 2.0); delay formula matches an independent closed-form
  re-derivation to 7e-5 ns.
- Timing (teaching artifact): full GPU pipeline ≈ 1.0 ms vs ≈ 112 ms CPU.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Single plane-wave transmit (compounding documented), 2-D linear array, synthetic phantoms only
  (per §8 — never patient data), elastography + servoing documented milestones. IEC 62304/FDA
  orientation in PRACTICE §4, educational-only restated.

## Next push preview

30.01 the agriculture bundle (milestone 1: fruit detection + 3-D localization) closes batch 1g;
then batch 1h (32.02, 34.03, 35.01, 36.03) completes all 36 flagships.
