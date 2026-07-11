# Data — 01.13 Canny + Hough line/circle detection for industrial alignment

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

Why synthetic is not a compromise here (CLAUDE.md §8's default, and the honest right call): an
industrial-alignment demo needs EXACT ground truth for the applied fixture offset, the 4 edge lines,
and the 3 hole centers/radii — a real photograph cannot give you that without a calibrated
metrology rig, while a renderer with a known transform gives it for free, to sub-pixel precision, by
construction. `scripts/download_data.ps1` / `.sh` are therefore honest no-ops for this project.

| Property | Value |
|----------|-------|
| Kind | Synthetic (repo default) |
| Generator | `../scripts/make_synthetic.py --seed 42 --dx 8.0 --dy -5.0 --dtheta-deg 7.0` |
| License | Synthetic — repo MIT license applies (no external data, no restrictions) |
| Size (committed) | ~154 KB total (two 75 KB PGMs + a 1 KB CSV) |
| Checksum (SHA-256) | `scene.pgm`: `34a62eccb7f19225be7110dd67cd81bdc7c3a63258204cf98d4c0c704ec33501` |
| | `negative_control.pgm`: `984b4ef2eba093da34c5b2bd2d6f9b5aa384e452e8794ec4204bb9edc65bb837` |
| | `truth.csv`: `c9154c5fef6a9b4263c1b123ba6bf574634069064dd7784c92a3a8651641851d` |
| Regenerate with | `python ../scripts/make_synthetic.py` (defaults reproduce the committed files exactly) |

Recompute any checksum with `python -c "import hashlib;print(hashlib.sha256(open('FILE','rb').read()).hexdigest())"`.

### What the scene depicts

A 320x240, 8-bit grayscale image of a rectangular machined plate (like a laser-cut mounting bracket)
photographed from directly above under bright, slightly vignetted lighting: brushed-metal texture,
3 drilled alignment holes of distinct known radii (6, 8, 10 px), and a deliberately shallow "scratch"
mark on the top edge used to demonstrate single- vs. double-threshold hysteresis. The plate is rendered
under a KNOWN in-plane offset+rotation (`dx=8.0 px, dy=-5.0 px, dtheta=7.0 deg`) relative to image
center — exactly the kind of error a fixture, conveyor, or robot pick would introduce on a real line.
`negative_control.pgm` is the same background/texture/vignette/noise with NO plate at all, used to
prove the detector reports nothing when there is nothing to detect.

### Fields / format

**`scene.pgm` / `negative_control.pgm`** — binary PGM (P5): a 3-line ASCII header (`P5\n320 240\n255\n`)
followed by 76,800 raw uint8 grayscale samples, row-major, `idx = y*320+x`. No units/frame beyond pixel
coordinates (`x` right, `y` down) — see `../src/kernels.cuh` SECTION 1.

**`truth.csv`** — plain CSV, `#`-prefixed header comments, one row type per line:

| Row | Columns | Meaning |
|-----|---------|---------|
| `TRANSFORM` | `dx_px, dy_px, dtheta_rad` | The applied rigid offset — what the alignment gate checks the pipeline's *recovered* transform against. |
| `LINE` | `name, theta_rad, rho_px, ax, ay, bx, by` | One of the 4 plate edges in Hough form (`x*cos(theta)+y*sin(theta)=rho`, `theta` in `[0,pi)`) PLUS its two rasterized corner endpoints (image frame, px) for the finite-segment analytic edge mask. |
| `HOLE` | `cx_px, cy_px, r_px` | One drilled hole's image-frame center and its KNOWN nominal radius (rows appear in the same order as `HOLE_LOCAL_X/Y/RADIUS` in `../src/kernels.cuh`). |
| `SCRATCH` | `x0, y0, x1, y1` | The engineered weak-but-connected scratch mark's leading-edge endpoints (image frame, px) — what the `hysteresis_lesson` gate samples. |

All positions/lengths are in pixels; this teaching project does not model a camera's mm-per-pixel
intrinsic calibration (see `../README.md` "Limitations & honesty" and `../PRACTICE.md` §3 for how a
real station gets millimeters out of a pipeline like this one).
