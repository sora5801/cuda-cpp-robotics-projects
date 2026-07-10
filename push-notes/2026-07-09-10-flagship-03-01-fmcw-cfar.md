# Push note — 2026-07-09-10: flagship 03.01 fmcw cfar

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 03.01 — **FMCW radar cube processing** — is done: a synthetic 77 GHz radar cube
(256 samples × 128 chirps × 8 antennas) processed through the classic pipeline — windowed range
FFT, Doppler FFT, noncoherent integration, CA- and OS-CFAR detection, FFT angle estimation — with
every injected target recovered within centimeters/centimeters-per-second/fractions-of-a-degree.
This is the repository's first project using a CUDA toolkit library (**cuFFT**), and it earns the
§5 exception the honest way: every plan and call is explained per §6.1 rule 6, including a proof
in the comments that the Doppler pass genuinely cannot be one advanced-layout call in the chosen
cube layout. The CFAR lesson is measured, not asserted: on a deliberately engineered close target
pair, **CA-CFAR is masked by the strong neighbor (5/6 detected) while OS-CFAR resolves it (6/6)**.
A real bug found en route is kept as a teaching artifact: linearly-seeded xorshift noise was
measurably non-white (400+ phantom detections) until an avalanche hash fixed the seeding —
documented in THEORY.md.

## What changed

- **[projects/03-perception-radar-sonar-event/03.01-fmcw-radar-cube-processing/](../projects/03-perception-radar-sonar-event/03.01-fmcw-radar-cube-processing/)** —
  complete: cube synthesis kernel, Hann window, cuFFT range/Doppler/angle stages, integration,
  CA/OS-CFAR kernels, hand-rolled O(N²) DFT CPU oracle, RD-map + detections artifacts, full
  README / THEORY / PRACTICE; `cufft.lib` linked in both configs with the required XML comment.
- **[docs/STATUS.md](../docs/STATUS.md)** — 03.01 → `done` (**15/505**).

## New projects (didactic blurbs)

**03.01 — FMCW radar cube + CFAR** (★ beginner, domain 03, flagship). Teaches the unification at
the heart of FMCW radar: range, velocity, and angle are the *same* phase measurement at three time
scales (within a chirp, across chirps, across antennas) — derived once, then read straight out of
three FFTs. CFAR theory follows: why fixed thresholds fail, how cell-averaging adapts, and where
ordered statistics beat it (target masking — demonstrated). GPU content: batched cuFFT layouts as
taught material, per-cell CFAR stencils. The single most interesting thing to look at:
`demo/out/range_doppler.pgm` — six bright targets in a live noise floor, including the close pair
that separates the two CFAR variants.

## How to build & run

```powershell
projects\03-perception-radar-sonar-event\03.01-fmcw-radar-cube-processing\demo\run_demo.ps1
# then open demo\out\range_doppler.pgm and demo\out\detections.csv
```

## What to study here

`THEORY.md` §The problem (the three-timescale phase unification — read this even if you skip the
rest) → §The algorithm (CFAR derivations) → `src/kernels.cu` (the cuFFT layout commentary).
First exercise: drop the Hann window and watch a strong target's sidelobes bury the weak one —
the window's purpose, felt.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors and zero new warnings** (the one
  Debug LNK4099 is the pre-existing template-wide PDB race already root-caused in 01.02's push —
  second confirmed sighting, queued for the §11 standards retrospective).
- `demo/run_demo.ps1` passes end to end: all 9 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** worst relative RD-map power deviation 0.55% vs the hand-rolled O(N²) DFT
  oracle (tol 5% — window + FFT-vs-DFT rounding, documented).
- **Ground-truth gates:** OS-CFAR 6/6 targets matched, 0 false alarms (bound 8); worst errors
  0.055 m range / 0.137 m/s velocity / 0.95° azimuth (tolerances 0.50 / 0.30 / 3.0).
- **CFAR comparison gate:** CA-CFAR masked on the close pair's weak target; OS-CFAR resolves it —
  the designed demonstration, confirmed.
- Timing (teaching artifact): full GPU pipeline ≈ 32 ms vs CPU oracle ≈ 422 ms.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Decoupled range-Doppler phase model (coupling bias quantified in THEORY), point targets, no
  clutter model beyond noise — imaging radar and MIMO virtual arrays documented as the production
  step. Radar spectrum regulation covered as didactic orientation only.

## Next push preview

10.03 massively-parallel robot sim closes batch 1c — the Isaac-Gym-style 10,000-environment
pattern.
