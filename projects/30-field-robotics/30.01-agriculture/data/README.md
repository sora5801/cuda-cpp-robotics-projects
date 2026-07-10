# Data — 30.01 Agriculture: fruit detection + 3D localization + ripeness; weed-vs-crop segmentation at frame rate; per-plant spray targeting; crop-row following; canopy volume from LiDAR; under-canopy navigation; yield mapping

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
| Kind | **Synthetic** orchard RGB-D scene with exact, analytic 3-D ground truth |
| Generator | `python ../scripts/make_synthetic.py` (defaults: 640x480, seed 42, 25 fruit) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed, 3 files) | 1,537,795 bytes (~1.47 MiB) total |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no RNG state, only the fixed integer hash / `random.Random(42)` — see the script) |

### Why synthetic, not a public fruit-detection dataset (the DECISION)

Public orchard-imagery datasets exist (MinneApple, Fuji-SfM, various Kaggle fruit-detection sets) and
are excellent for the **2-D, deep-learning** literature this project's classical pipeline is explicitly
positioned as the didactic baseline for (README "Prior art & further reading"). None of them fit
*this* project's need, though, for one specific, load-bearing reason: **they carry 2-D bounding-box or
segmentation labels, not exact 3-D fruit centers/radii/ripeness** — and this project's whole point is
3-D localization. Nobody hand-measures a real orchard photograph's true metric fruit positions to
sub-centimeter accuracy; synthesizing the scene directly in 3-D and rendering the camera's view of it
is the only way to get EXACT ground truth to verify against (the same reasoning 01.02's stereo project
documents for depth ground truth). `../scripts/download_data.ps1` / `.sh` state this decision too.

### Fields / format

Three files, all documented in full in `../scripts/make_synthetic.py`'s header comment (layout
authority: [`../src/kernels.cuh`](../src/kernels.cuh)); loader: `read_ppm`/`read_pgm16`/
`load_ground_truth` in [`../src/main.cu`](../src/main.cu).

| File | SHA-256 | Meaning |
|------|---------|---------|
| `sample/rgb.ppm` | `fb8d1b38c689a68e933338d240fc31d06bc0f9e577b8633c0acc9bfcba998db8` | 640x480, 8-bit RGB (PPM P6). The camera's color view of the scene. |
| `sample/depth.pgm` | `0ba3e1dae1666f1071cbf4cc43201ea7d2aaf965a7a6cfaedb74ac20c6a964aa` | 640x480, 16-bit gray (PGM P5, **big-endian**), depth in **millimeters** — converted to float meters on load. Includes realistic quadratic-in-range sensor noise, sigma_z(Z) = 0.0015*Z^2 m (see THEORY.md). |
| `sample/ground_truth.csv` | `9e20c1eacb1b7e439b03420f584badf0d0c6b8a56062ca0b8bfe658f7095335d` | One row per fruit: `fruit_id,cx_m,cy_m,cz_m,radius_m,ripeness,visible_px,ideal_px_est,visible_frac` — the EXACT scene truth. Read ONLY by `main.cu`'s verification/gate stage; the pipeline itself never touches this file. |

**Camera** — pinhole, `fx=fy=525` px (the Kinect-v1/TUM-RGBD focal length — a real, documented anchor,
not invented), principal point `(cx,cy)=(320,240)` (image center). Camera looks down `+Z` in the
OPTICAL convention (x-right, y-down, z-forward — SYSTEM_DESIGN.md section 3.2's documented exception
to the repo default frame). These four numbers are shared, single-sourced constants in
[`../src/kernels.cuh`](../src/kernels.cuh) and `../scripts/make_synthetic.py`.

**The scene**: **25 fruit** (spherical, radius 2.8-4.8 cm) at depths 1-4 m, ripeness sampled uniformly
in `[0.35, 1.0]` (hue `0..78` degrees — scoped away from the green-on-green ambiguity a fully-unripe
fruit at hue 120 degrees would create against the foliage background; see THEORY.md "ripeness-vs-color
honesty" and README "Limitations"). Placement is random (seed 42) within a frustum sized to produce
SOME genuine screen-space overlap — this is a **designed occlusion scene**, not an accident: the
committed sample contains

- **one fully-occluded fruit** (0% visible — entirely hidden behind a nearer fruit; no algorithm could
  find it from this view, and it is honestly excluded from the demo's detection-rate denominator), and
- **two designed cross-depth merge cases** — pairs of fruit at noticeably different depths (roughly
  1.2 m and 0.5 m apart in Z) whose projected disks nonetheless touch/overlap in the 2-D image, which
  this project's color-and-connectivity pipeline (by design — it does not split components on depth
  discontinuities; that is documented future-milestone territory) merges into ONE detected blob each.

Foliage background: 2-octave procedural value noise (hue 100-140 degrees — comfortably clear of every
fruit's hue range), plus dark branch strokes and a handful of tiny fruit-colored "glint" specks
(deliberate false-positive bait for the mask stage, cleaned up by morphological opening and/or the
component size floor — see THEORY.md). Full derivation and every constant: the generator script itself.

The loader is strict: wrong magic/maxval/dimensions on either image, or an empty ground-truth file,
abort the demo with a clear message rather than silently misinterpreting bytes.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
