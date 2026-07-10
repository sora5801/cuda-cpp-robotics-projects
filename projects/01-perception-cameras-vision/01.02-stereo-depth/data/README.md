# Data — 01.02 Stereo depth: block matching, then Semi-Global Matching (SGM) kernels

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
| Kind | **Synthetic** rectified stereo pair with dense, analytic ground truth |
| Generator | `python ../scripts/make_synthetic.py` (defaults: 384x288, seed 42) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed, 4 files) | 442,428 bytes (~432 KiB) total |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no RNG state, only the fixed integer hash — see the script) |

### Why synthetic, not Middlebury/KITTI (the DECISION)

Public stereo benchmarks exist and are excellent — but none fit this project's constraints without
either a license problem or a redistribution problem, so v1 ships synthetic and documents the
alternative honestly rather than silently working around it:

- **Middlebury Stereo** (2001–2021 datasets) — free for research use, but the license terms are
  per-dataset and do not grant blanket redistribution rights inside a public MIT-licensed repository;
  the *fair* path is linking learners to the official site, not re-hosting extracted frames.
- **KITTI Stereo** — explicitly **non-commercial, no-redistribution** (CLAUDE.md §8 names KITTI as a
  standing example of this). Even a small extracted crop cannot be committed here.
- Both also give SPARSE ground truth (Middlebury: structured-light scans with real gaps at
  reflective/textureless surfaces; KITTI: LiDAR projected into the image, sparse by construction) —
  which would force the ground-truth gate below to work around missing values before it could even
  start comparing methods, muddying the BM-vs-SGM story this project exists to tell.

A synthesized scene with an EXACT, ANALYTIC, fully-dense disparity field (this project authors the
scene directly in disparity space, then physically-correct z-buffer forward-warping derives the right
image and the occlusion mask — see `../scripts/make_synthetic.py`) sidesteps all three problems at
once: zero license risk, zero redistribution risk, and 100% dense ground truth with EXACTLY known
occlusion, not an approximation of it. `../scripts/download_data.ps1` / `.sh` state this decision too
and point at the two datasets' official pages for learners who want to try real photographs next
(README "Exercises").

### Fields / format

Four plain binary PGM (P5, 8-bit grayscale) files, 384x288 each — the smallest real image format
there is, viewable in any image tool with no libraries (`../src/main.cu`'s `read_pgm`/`write_pgm` are
~15 lines each). Loader: `load_sample()` in [`../src/main.cu`](../src/main.cu); layout authority
(image indexing, disparity convention, every constant below): [`../src/kernels.cuh`](../src/kernels.cuh).

| File | SHA-256 | Meaning |
|------|---------|---------|
| `sample/left.pgm` | `2cf11cb62781fc32fbcc3fb7e0848b7e963991f646ea74b217a0a6738851ff0e` | The reference image — this project's disparity output is indexed by ITS columns throughout. |
| `sample/right.pgm` | `5cf59e83caecd5a1b8bcf6295f0329f33530a9fa7e66b213f1e12ab3e8d98f89` | The matching image — produced by exact z-buffer forward-warping of `left.pgm` (occlusion falls out of the warp, not a heuristic). |
| `sample/gt_disparity.pgm` | `58d7a4997deccd37067deafce13150a72f0e98626189a01b5f39ec8e85930ae1` | Ground-truth disparity, **scaled by 4** (`kGtDispScale` in `../src/kernels.cuh` / `DISP_SCALE` in `../scripts/make_synthetic.py`): pixel value = true disparity (0..63) × 4, so every one of the 64 disparity levels stays visually distinct (max 63×4=252) without leaving byte range. |
| `sample/gt_valid.pgm` | `c7efb151e1d6dac0974a745f3cf0b6483cfcef65527c6aac0f267efbae1cc038` | Scoring mask: **255** = this LEFT pixel has a genuine, unoccluded, in-frame correspondence in `right.pgm` and is used by the ground-truth gate; **0** = occluded, projects off-frame, or inside the 3-px census border margin — never scored, because no window-based stereo method could produce a meaningful answer there regardless of ground truth. |

**The scene** (fully specified in `../scripts/make_synthetic.py`'s header comment, reproduced here for
quick reference): a textured ground plane whose disparity varies smoothly with image ROW ONLY (4 px at
the top/far, 18 px at the bottom/near — the physically correct behavior for a flat ground plane under
a forward-looking camera, derived in `../THEORY.md`), plus three textured fronto-parallel rectangles at
constant disparities 26, 40, and 52 px — two of which partially overlap (a nearer one occluding a
farther one) so the scene exercises BOTH object-vs-background and object-vs-object occlusion. Texture
is 2-octave deterministic value noise, coarse-frequency-dominated on purpose (see the script's
`texture_byte` docstring): real matching ambiguity — the kind SGM's smoothness prior is built to
resolve — needs locally-similar-looking patches, not maximally-distinctive ones. All content traces
back to one documented base seed, 42, with per-layer offsets 43/44/45.

**Ground-truth coverage**: 95,448 of 110,592 pixels (86.3%) are GT-valid (unoccluded, outside the
census border) — printed by `make_synthetic.py` on every regeneration, and matches the "N GT-valid
pixels" denominator in the demo's `BM:`/`SGM:` output lines exactly.

The loader (`load_sample` in `../src/main.cu`) is strict: any of the four files missing, dimension
mismatches between them, or a scene too small for the 3-px census margin plus D=64 disparities all
abort the demo rather than silently degrading.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
