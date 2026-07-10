# Data — 03.01 FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1` / `.sh` where one genuinely
  teaches more. **This project needs none** — see "Why no public dataset" below.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** — a radar configuration record + a fixed ground-truth target list (no recordings; the raw ADC cube is synthesized in code, never written to disk) |
| Files | `sample/radar_params.csv`, `sample/targets.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 459 bytes + 775 bytes = 1.2 KiB total |
| Checksum (SHA-256) | `radar_params.csv`: `29a239592ce8bfd4f11c76de4fc288029b0d252312cfac522a26a40e3da69708` |
| | `targets.csv`: `4b8dfa1b919770078d9bb21431f60adf4fc15ac185a3a5a9d534d2c38aa77429` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness; both files are constants) |

### Why no public dataset

Real automotive/robotics radar ADC-level cube recordings are essentially never published raw: radar
OEMs treat the raw cube as proprietary (it is the input to their own signal-processing IP), and the
public radar datasets that DO exist (nuScenes' radar channel, RADIal, and similar) ship
already-processed detection lists or range-Doppler spectrograms — not the `Ns x Nc x Na` complex
baseband cube this project's pipeline is built to teach from end to end. A synthetic cube with EXACT,
documented ground truth (every target's true range/velocity/azimuth known to machine precision) also
teaches the verification story (README "Expected output") better than any recording could: there is
no ambiguity about what the "correct" detection list should be. `../scripts/download_data.ps1`/`.sh`
are therefore honest, permanent no-ops — see their file headers for the same reasoning.

### Fields / format

Both files are plain-text CSV; `#`-prefixed lines are comments. Loaders: `load_radar_params()` and
`load_targets()` in [`../src/main.cu`](../src/main.cu); the authoritative constants both are checked
against live in [`../src/kernels.cuh`](../src/kernels.cuh).

**`radar_params.csv`** — one data row, columns `fc_hz,bandwidth_hz,chirp_dur_s,ns,nc,na`:

| Field | Units | Meaning | Committed value |
|-------|-------|---------|------------------|
| `fc_hz` | Hz | carrier (center) frequency | 77,000,000,000 (77 GHz) |
| `bandwidth_hz` | Hz | chirp sweep bandwidth `B` | 300,000,000 (300 MHz) |
| `chirp_dur_s` | s | active chirp sweep duration `Tc` | 0.00005 (50 us) |
| `ns` | count | ADC samples per chirp (fast-time / range axis length) | 256 |
| `nc` | count | chirps per frame (slow-time / Doppler axis length) | 128 |
| `na` | count | virtual receive antennas, half-wavelength ULA (angle axis length) | 8 |

**This file is not a runtime configuration input.** `ns`/`nc`/`na`/`fc_hz`/`bandwidth_hz`/
`chirp_dur_s` size COMPILE-TIME structures in `src/kernels.cuh` (CFAR's fixed-size per-thread local
arrays, cuFFT plan lengths) — they cannot be changed by editing this file alone. `main.cu` loads it
purely to CROSS-CHECK against the values the running binary was actually built with (within a 1e-5
relative tolerance — `fc_hz`'s value is not exactly representable in FP32, see `src/main.cu`'s
`check_radar_params` comment) and aborts loudly on any mismatch, so a stale committed file can never
silently produce a wrong demo.

**`targets.csv`** — one row per target, columns `range_m,vel_mps,az_deg,amp`:

| Field | Units, frame | Meaning | Valid range |
|-------|--------------|---------|--------------|
| `range_m` | meters | true target range | `0 <= range_m < kRangeMaxM` (~127.9 m) |
| `vel_mps` | meters/second | true radial velocity. **Sign convention: POSITIVE = APPROACHING** the radar (closing, range decreasing) | `-kVelMaxMps < vel_mps < kVelMaxMps` (~+/-19.47 m/s) |
| `az_deg` | degrees | true azimuth, 0 = boresight (broadside), the array's steering-angle convention (THEORY.md "The math") | roughly `-90 < az_deg < 90` |
| `amp` | unitless | reflection amplitude — an RCS-ISH teaching scale, **not** a calibrated radar-cross-section or power (see README "Limitations & honesty") | `> 0`, this scene's values span 0.15-1.0 |

The 6 committed rows (seed conceptually "42" — see `../scripts/make_synthetic.py`'s docstring for the
exact design story) were chosen to:

1. Span the unambiguous range/velocity/azimuth envelope with realistic, round numbers.
2. Include one weak, far target (target 3: 80 m, amplitude 0.3) as a detection-sensitivity check.
3. Include a **close pair** (targets 5 and 6: 60.0 m / +6.0 m/s / amplitude 1.0, and 61.5 m / +6.9 m/s
   / amplitude 0.15 — 3 range bins and 3 Doppler bins apart, a 6.7x amplitude ratio) that reliably
   demonstrates CA-CFAR's masking weakness while OS-CFAR still resolves the weaker target — this
   project's headline, measured comparison (THEORY.md "The algorithm", README "Overview").

The loader is strict: unknown row shapes, short rows, or a missing/empty target list abort the demo
rather than silently running a different scenario than intended.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
