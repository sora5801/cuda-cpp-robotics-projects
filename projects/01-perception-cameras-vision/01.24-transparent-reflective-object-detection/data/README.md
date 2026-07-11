# Data — 01.24 Transparent/reflective object detection via polarization imaging

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** (where one genuinely teaches more) are fetched by `../scripts/download_data.ps1`
  / `.sh` — idempotent, with source URL, expected size, and checksum documented below. **Respect every
  license**; registration-gated or no-redistribution datasets (KITTI, nuScenes) are pointed at, never
  mirrored.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | 100% synthetic, generated in Python (stdlib only — no numpy, no PIL). |
| Generator | `../scripts/make_synthetic.py` — the physics forward model (Malus's law + the closed-form Fresnel equations), independently re-derived in Python from the C++/CUDA equations in `../src/kernels.cuh` (see that script's module docstring for why the duplication is deliberate). |
| License | Synthetic — this repository's MIT license applies (no external data, no third-party rights). |
| Size (committed) | ~611 KB total (`sample/` folder; `truth_maps.csv` alone is ~563 KiB — one row per pixel, kept because it is the ground truth every physics/accuracy GATE in `../src/main.cu` checks against). |
| Regenerate with | `python ../scripts/make_synthetic.py` (fixed seed 42; byte-identical output every run). |

`../scripts/download_data.ps1` / `.sh` intentionally do nothing meaningful here: there is no public
polarization dataset this project draws from, and a synthetic scene is what makes the `fresnel_anchor`
gate possible in the first place — a real photograph never comes with per-pixel TRUE DoLP/AoLP labeled
against a first-principles physics model, only a synthetic one can (the same synthetic-first reasoning
01.10's/01.11's/01.22's data folders document).

## Files and their fields

| File | Format | Shape | What it is |
|------|--------|-------|------------|
| `mosaic.pgm` | PGM (P5), 8-bit grayscale | 128x128 | The DoFP sensor's actual capture of the main scene (matte background + glass pane + glass dome + brushed metal bar): one polarizer-angle reading per pixel (`kernels.cuh` Section 2's 2x2 phase mosaic), rendered via Malus's law from the true per-pixel Stokes parameters, plus additive Gaussian sensor noise (std 2.2 DN), clamped to 8-bit DN. The ONLY thing the pipeline (`../src/main.cu`) ever loads as its "sensor" input. |
| `mosaic_negctrl.pgm` | PGM (P5) | 128x128 | The SAME background model alone (no objects at all), rendered with an INDEPENDENT noise draw — the `negative_control` gate's input; the pipeline must report zero detections here. |
| `truth_maps.csv` | CSV, header + 16,384 rows | 1 row/pixel | `x,y,s0_dn,dolp,aolp_deg,label` — the noise-free GROUND TRUTH at every pixel of the MAIN scene: true Stokes S0 (DN), true DoLP (unitless [0,1]), true AoLP (degrees, `[0,180)` convention), and an object label (`0`=background, `1`=glass pane, `2`=glass dome, `3`=metal bar). **Never fed into the detection pipeline** — only `main.cu`'s GATEs (`stokes_accuracy`, `fresnel_anchor`, `detection`) read it, to check the pipeline's OUTPUT against physics ground truth. |
| `params.csv` | CSV, `parameter,value` rows | 23 rows | Every generation parameter in one place (seed, canvas size, n_glass, the closed-form Brewster angle, pane/dome/metal/background parameters, noise std, and the detection thresholds/size floor) — the same numbers `kernels.cuh` states as compile-time constants (a MUST-MATCH contract, see that file's header), included so a learner can inspect the exact scene recipe without reading Python. |

### Units and conventions

Pixel values (`mosaic*.pgm`, `s0_dn`) are DN (digital number / code value), 8-bit range `[0, 255]`
after the sensor-noise + ADC-quantization step, matching the repo's image convention (01.01/01.09/
01.11/01.22). DoLP is unitless in `[0, 1]`. AoLP is in **degrees**, convention `[0, 180)` — a linear
polarizer's transmission axis at `theta` and `theta+180` is physically the same axis (Malus's law is
`pi`-periodic, not `2*pi`; `THEORY.md` "The math" derives the half-angle wrap this convention encodes).
The DoFP super-pixel phase layout (which physical pixel measures which of the four angles) is documented
once, as code, in `kernels.cuh` Section 2 — this data folder never restates it as a second source of truth.

## How this was generated (the physics forward model)

`make_synthetic.py` computes, PER PIXEL, the true `(S0, DoLP, AoLP)` from first principles: a smooth
matte background (small residual DoLP + a gentle brightness gradient); a flat glass pane at one
documented incidence angle (the REAL Fresnel equations, `n=1.5`); a curved glass dome whose LOCAL
incidence angle varies with radius (`theta_i(r) = asin(r/R)`, orthographic-sphere geometry) — producing
the real "Brewster ring" polarization cameras photograph on specular spheres; and a brushed metal bar
with a documented phenomenological (non-Fresnel — real metals need a complex refractive index, stated
honestly in `THEORY.md`) saturating DoLP curve. The two glass objects' S0 is set to MATCH the local
background exactly ("by construction") — only their DoLP/AoLP differs, which is the whole point of this
project (README "System context"). Malus's law then renders the four polarizer-angle intensities at
every pixel, adds noise, and keeps only the ONE channel each physical pixel's own super-pixel phase
measures — the real DoFP sensor's spatial multiplexing.

## Checksums (SHA-256, computed on the committed files)

```
9a117b87b2903faa513fe0cf93c7de386a5340457cb21f55b11f6e0a5d551019  mosaic.pgm
a57a91256e3b44186f7b95a29e6b210abbc07a120c00c851da6357bed10aabdb  mosaic_negctrl.pgm
a660f1b82eb8de0d00966d560832177cbd35c3f3195e2b9597dadd598c166013  truth_maps.csv
0b24cc617a61cf9ab9dcef1a454c8a1c9efef8d64b98071b9d592e9351f01590  params.csv
```

(Regenerate and verify with `python ../scripts/make_synthetic.py && sha256sum sample/*.pgm sample/*.csv` —
the seed is fixed, so a clean regeneration reproduces these hashes exactly.)

## Size math

`mosaic.pgm`/`mosaic_negctrl.pgm`: 128x128 = 16,384 pixel bytes + a 15-byte P5 header
(`P5\n128 128\n255\n`) = 16,399 bytes each. `truth_maps.csv`: a 2-line comment/header preamble plus
16,384 data rows of `x,y,s0_dn,dolp,aolp_deg,label` (roughly 35 bytes/row) = ~576 KB — by far the
largest file here, kept anyway because it is the ground truth every physics gate needs and 576 KB is
three orders of magnitude under the repo's 50 MB ceiling (CLAUDE.md §8). `params.csv`: under 1 KB.
**Committed sample total: ~611 KB.**
