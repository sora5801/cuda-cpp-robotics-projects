# Data — 12.01 TensorRT deployment with custom CUDA pre/post kernels: NMS, argmax decode, keypoint extraction

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored; unused here).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more. **This project needs none** — see "Why no public dataset" below.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic**, three paired files: fixed network weights, a test image, and its ground truth |
| Generator | `python ../scripts/make_synthetic.py` (default: seed 42 — the committed sample) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | `weights.bin` 460 B, `test_scene.ppm` 19,213 B, `ground_truth.csv` 390 B (≈19.7 KiB total) |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (fixed seed, no external entropy) |

### Checksums (SHA-256)

| File | SHA-256 |
|------|---------|
| `sample/weights.bin` | `28aece9a079102c17e956d0e04c91f97b06a5a3ff857a00ed07210ac5c11dd51` |
| `sample/test_scene.ppm` | `d89b049cd99441ccafafeb6d628cf6a2ffb0815b5b32437e0524ec372d60aaeb` |
| `sample/ground_truth.csv` | `a4f8a8588e3bbaffe5185d7b9954d022e485871dce475ddb6557a14e6e7ed9d3` |

## Why no public dataset (and no trained model)

This project teaches **deployment plumbing** — custom CUDA pre/post kernels around an inference
core — not object-detection accuracy or training (that is project 12.06's job; see README
"Limitations & honesty"). A real trained detector's weights are not reproducible from a seed and
are usually large (megabytes to hundreds of megabytes) — exactly the kind of asset CLAUDE.md §8
says stays out of a tiny committed sample. Using a small, **hand-designed, deterministic** network
instead sidesteps both problems: `weights.bin` is 460 bytes, the forward pass is checkable by hand
(THEORY.md works two full worked examples), and the demo is byte-reproducible everywhere with zero
downloads and zero training.

## Fields / format

### `sample/weights.bin` — the fixed "network" (460 bytes)

Byte-exact binary format, little-endian; the canonical documentation (and the only place this
layout is allowed to change) is [`../src/kernels.cuh`](../src/kernels.cuh) SECTION 3:

| Offset | Bytes | Content |
|--------|-------|---------|
| 0 | 8 | magic `"RCWTPK01"` (ASCII, not NUL-terminated) |
| 8 | 4 | `uint32` format_version (currently `1`) |
| 12 | 216 | `conv1_w[2][3][3][3]` float32 (54 values) |
| 228 | 8 | `conv1_b[2]` float32 |
| 236 | 144 | `conv2_w[2][2][3][3]` float32 (36 values) |
| 380 | 8 | `conv2_b[2]` float32 |
| 388 | 48 | `head_w[6][2][1][1]` float32 (12 values) |
| 436 | 24 | `head_b[6]` float32 |

Total 460 bytes = 12-byte header + 112 float32 values. `../src/main.cu`'s `load_weight_blob()` reads
each array with its own `ifstream::read()` call, in this exact order — never one bulk struct-sized
read — so the format never depends on any compiler's struct-packing behavior.

### `sample/test_scene.ppm` — the test image (80×80 RGB, binary PPM/P6)

A standard, library-free binary PPM: ASCII header `"P6\n80 80\n255\n"` followed by
80×80×3 = 19,200 raw bytes, row-major, interleaved RGB (HWC). Contents: a mid-gray (128,128,128)
background with a small, seeded (seed 42), per-pixel-per-channel dither of amplitude ±3 (so the
scene is not unrealistically flat), and three solid rectangles — "red" (240,50,50) or "blue"
(50,50,240) — placed at the coordinates in `ground_truth.csv`. All units: pixels, `(0,0)` = top-left,
x right, y down (image-space convention, not the repo's default robot body frame — CLAUDE.md §3.2
allows this for camera/image data with the convention stated, as here).

### `sample/ground_truth.csv` — the known objects (paired 1:1 with `test_scene.ppm`)

Plain-text CSV; `#` lines are comments. One row type:

**`OBJ,class_id,x0,y0,w,h`** — a placed rectangle, SOURCE-image pixel coordinates (int):

| Field | Meaning |
|-------|---------|
| `class_id` | `0` = red, `1` = blue (the network's 2-class table — `../src/kernels.cuh`) |
| `x0,y0` | top-left corner, pixels |
| `w,h` | width, height, pixels (committed sample: always 15×15) |

`../src/main.cu`'s `load_ground_truth()` is a strict loader: an unknown row label, a short row, or
an out-of-range `class_id` aborts the demo rather than silently misreading the scene.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
