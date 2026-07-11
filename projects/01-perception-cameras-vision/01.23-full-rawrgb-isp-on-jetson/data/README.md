# Data — 01.23 Full RAW→RGB ISP on Jetson (Argus + custom CUDA stages)

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md section 8).

## The rules (repo-wide, CLAUDE.md section 8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads.
- **Public datasets** are fetched by `../scripts/download_data.ps1` / `.sh` where one genuinely
  teaches more. **None applies here** — see below.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | 100% synthetic — a hand-authored scene rendered through a documented forward sensor model |
| Generator | `../scripts/make_synthetic.py` (stdlib-only Python, xorshift32, seed 42) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | 6 files, ≈365 KB total (see checksum table below) |
| Regenerate with | `python ../scripts/make_synthetic.py` (deterministic — regenerates byte-identical output) |

**Why no public dataset applies.** A public RAW10 sensor dump would give a real Bayer mosaic but
**none** of the per-stage ground truth this project's ten gates need: the exact pre-noise sensor
value at every pixel, the exact illuminant gain, the exact defect-pixel locations, the exact
spectral crosstalk matrix. Synthesizing the scene is the *only* way to get all of that, exactly,
for every stage — real RAW captures are exactly what real ISP tuning teams use *unit tests against
a known target chart* for, which is precisely this project's approach in miniature. `../scripts/download_data.ps1`
and `.sh` are therefore honest no-ops.

## Files in `sample/`

| File | What it is | Size | SHA-256 |
|------|------------|------|---------|
| `raw_mosaic_d65.bin` | 160×120 RAW10-in-uint16 RGGB mosaic, D65 illuminant, little-endian, no header | 38,400 B | `bf1ca376343c4505236c0d008ac768853b5302e57c4955e0cfa5166372777335` |
| `raw_mosaic_tungsten.bin` | Same, tungsten illuminant | 38,400 B | `34b5fdd9f25a71244f841860f0f35785605d0543e5fb7007d39709b503df6b31` |
| `true_sensor_rgb_d65.bin` | 160×120×3 float32 (little-endian), noiseless PRE-shading sensor-domain RGB under D65 — the demosaic-quality ground truth | 230,400 B | `5b396ead26b30f93c6af6894490c0b1b90b4ef8118e0d717c68956726189236d` |
| `true_scene_srgb.ppm` | 160×120×3 uint8 PPM (P6): `gamma_encode(scene reflectance)` — the illuminant-independent end-to-end reference rendering | 57,615 B | `e90b1dc28727ec86bd84ef37882025cacf7f40645bc3fb8681961dd42b79ad58` |
| `defect_list.csv` | 16 rows of `x,y,kind` — the committed factory defect map, loaded at RUNTIME (not compiled in) | 308 B | `24072412a18cfa7e87d3b10a95c40fd72a0d6c77c4ad69058804f285429b0aa3` |
| `params.txt` | Human-readable mirror of every generation constant + the defect list, for auditing without opening `kernels.cuh` | 970 B | `b94007c60407eaf6fe02b490abb5431f6ae2918d472614426741529f12950cb4` |

Regenerate the SHA-256 table yourself with `sha256sum data/sample/*` (or `Get-FileHash` on Windows)
after running the generator — it is fully deterministic (fixed seed, fixed draw order, documented
in `../scripts/make_synthetic.py`'s file header), so the bytes above should reproduce exactly.

## Fields / format

- **RAW mosaics** (`raw_mosaic_*.bin`): `uint16_t[160*120]`, row-major, one 10-bit code (0–1023,
  RAW10-in-uint16 convention — see `../src/kernels.cuh` section 0) per pixel. RGGB tiling: pixel
  `(x,y)` is R if `(x,y)` both even, B if both odd, G otherwise (two green sub-lattices, Gr/Gb — see
  `bayer_phase_at()`). No header; dimensions are compile-time constants (`kRawW=160`, `kRawH=120`)
  shared by the generator and the pipeline.
- **`true_sensor_rgb_d65.bin`**: `float[160*120*3]`, row-major, channel-interleaved (R,G,B per
  pixel), little-endian. Units: dimensionless sensor-domain signal in `[0,1]` (illuminant gain +
  spectral crosstalk applied to the scene reflectance; no shading, no noise, no quantization).
- **`true_scene_srgb.ppm`**: standard binary PPM (P6), 8-bit sRGB-encoded, identical layout to every
  other project's PPM artifacts in this repo.
- **`defect_list.csv`**: header `x,y,kind`; `x,y` are raw-pixel integer coordinates (0-indexed);
  `kind` is one of `stuck_high` / `stuck_low` / `stuck_mid` (documentation only — the correction
  algorithm uses locations, not kinds, exactly like a real defect map).
- **`params.txt`**: plain text, one constant per line, plus the defect list restated — no fixed
  schema, meant for human eyes.

## Provenance & honesty notes

- The 24-patch color chart's reference sRGB8 values (`kChartRefSrgb8` in `kernels.cuh`) are
  **illustrative**, loosely modeled on the *families* of patches a classic X-Rite ColorChecker
  carries (skin tones, primaries/secondaries, a grayscale ramp) for pedagogical familiarity — they
  are **not** the certified X-Rite colorimetric values, which require a licensed spectral
  measurement this project does not have.
- The spectral crosstalk matrix, illuminant gains, shading polynomial, and noise model are
  documented, physically-motivated, **hand-chosen** numbers (see `kernels.cuh` section 2 and
  `THEORY.md`) — not measurements of any real sensor. They are dated and labeled as such wherever
  they appear.
- All timing numbers reported by the demo are measured on the owner's machine (RTX 2080 SUPER,
  sm_75) — see `README.md` "Expected output" — never fabricated (CLAUDE.md section 8).
