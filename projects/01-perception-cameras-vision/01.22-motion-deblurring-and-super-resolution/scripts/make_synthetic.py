#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.22
(Motion deblurring and super-resolution for inspection zoom).

Stdlib-only, deterministic (xorshift32, seed 42 — no random/numpy, per
CLAUDE.md paragraph 12: this repo standardizes on a hand-rolled xorshift32
PRNG rather than Python's Mersenne-Twister `random` module, so the SAME
generator logic can be ported verbatim to a future CUDA-side generator
without a library-RNG mismatch).

What this script builds (all under ../data/sample/, SYNTHETIC, labeled
everywhere it appears — CLAUDE.md paragraph 8):
  1. A single kW x kH "inspection scene" truth image: a flat patch, a
     high-contrast step edge, a row of hand-drawn dot-matrix glyphs, a
     deterministic hashed texture patch, and three bar-chart frequency
     groups (see kernels.cuh Section 4 for the exact rectangles — this
     script's constants MUST MATCH that file, marked at each block below).
  2. MILESTONE 1 data: a motion-blur PSF (psf_truth.csv) rasterized from a
     documented line length + angle, a MISMATCHED PSF at a wrong angle
     (psf_mismatch.csv, the honesty test), and blurred.pgm = the truth
     image circularly convolved with psf_truth plus additive Gaussian
     sensor noise.
  3. MILESTONE 2 data: kNumFrames low-resolution frames, each a genuinely
     ALIASED sub-pixel-shifted sampling of the SAME scene (lr_frame_*.pgm)
     plus shifts_truth.csv (the known registration every frame was rendered
     at) — the "honest way to make aliased LR frames" the task brief asks
     for: render at 4x supersample resolution ONCE, extract a shifted
     window per frame, then box-downsample 8x total to LR resolution — the
     same area-integration a real sensor's pixel performs, so every LR
     frame is a physically plausible (and genuinely alias-prone) capture,
     not merely a blurred copy of the truth image.

Because every truth-image FEATURE below is drawn BLOCK-CONSTANT per truth
pixel (bars/edges/glyphs/texture are all "hard" step patterns; no feature
needs sub-truth-pixel gradients), rendering at 4x supersample resolution
and nearest-neighbor-replicating each truth pixel into its 4x4 supersample
block is EXACT: box-downsampling that replicated block by 4 recovers the
original truth pixel exactly (mean of four identical values), and — the
part that matters for milestone 2 — extracting a window shifted by a
NON-multiple-of-4 supersample offset (this project shifts at 2, 4, or 6
supersample px = half/one/one-and-a-half truth pixels) genuinely blends
across truth-pixel boundaries, producing real, physically-motivated
anti-aliasing/aliasing in the resulting low-res frame. No interpolation
library, no numpy — plain nested loops over integers.

Usage:
    python make_synthetic.py                  # writes into ../data/sample/
    python make_synthetic.py --out DIR         # custom output directory
"""

import argparse
import csv
import math
from pathlib import Path

# ===========================================================================
# SECTION 1 — problem geometry. MUST MATCH kernels.cuh Section 1.
# ===========================================================================
K_W = 128
K_H = 128
K_N = K_W * K_H

# MUST MATCH kernels.cuh Section 2 (the PSF).
K_PSF_SIZE = 15
K_PSF_RADIUS = K_PSF_SIZE // 2
K_BLUR_LENGTH_PX = 9.0
K_BLUR_ANGLE_DEG = 20.0
K_MISMATCH_ANGLE_DEG = K_BLUR_ANGLE_DEG + 25.0
K_BLUR_NOISE_STD_DN = 3.0

# MUST MATCH kernels.cuh Section 3 (super-resolution geometry).
K_LR_SCALE = 2
K_LR_W = K_W // K_LR_SCALE
K_LR_H = K_H // K_LR_SCALE
K_NUM_FRAMES = 8
# Additive per-frame sensor noise for the LR captures — Python-generator-only
# (main.cu never needs this number: it measures noise empirically from the
# committed frames rather than trusting a re-stated constant, the same
# "independently re-derive, don't just trust the generator" discipline
# 01.11's noise_model_sanity gate uses).
K_LR_NOISE_STD_DN = 2.0

# Supersample factor and the truth-pixel margin drawn around the core scene
# so every LR frame's shifted extraction window (max shift 1.5 truth px)
# stays safely inside rendered content (never reads past the oversized
# canvas). S=4 supersample subdivisions per truth pixel; MARGIN_TRUTH_PX=4
# truth pixels of margin on every side (=16 supersample px, comfortably
# more than the largest shift, 6 supersample px — see kernels.cuh Section 3).
K_SUPERSAMPLE = 4
K_MARGIN_TRUTH_PX = 4

# MUST MATCH kernels.cuh Section 4 (shared scene-layout rectangles).
RECT_FLAT = (8, 44, 8, 40)          # x0,x1,y0,y1 — 36x32, value K_FLAT_DN
K_FLAT_DN = 128.0

RECT_EDGE = (52, 120, 8, 40)        # 68x32
K_EDGE_STEP_X = 86
K_EDGE_LO_DN = 24.0
K_EDGE_HI_DN = 220.0

RECT_GLYPHS = (8, 120, 44, 68)      # 112x24
K_GLYPH_CELL_PX = 2
K_GLYPH_PITCH_PX = 14
K_GLYPH_LO_DN = 20.0
K_GLYPH_HI_DN = 235.0

RECT_TEXTURE = (8, 120, 72, 96)     # 112x24
K_TEXTURE_BLOCK_PX = 4
K_TEXTURE_LEVELS = (60.0, 130.0, 200.0)

RECT_BAR_COARSE = (8, 40, 100, 120)     # period 8
RECT_BAR_MID = (44, 76, 100, 120)       # period 4
RECT_BAR_FINE = (80, 116, 100, 120)     # period 3
K_BAR_PERIOD_COARSE = 8
K_BAR_PERIOD_MID = 4
K_BAR_PERIOD_FINE = 3
K_BAR_LO_DN = 30.0
K_BAR_HI_DN = 225.0

DEFAULT_SEED = 42

# Seven hand-drawn 5x7 dot-matrix glyphs — a small bespoke bitmap set, NOT a
# real font (README/THEORY state this honestly). '1' = stroke (rendered at
# K_GLYPH_HI_DN), '0' = background (K_GLYPH_LO_DN).
GLYPHS = [
    ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],  # E
    ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],  # L
    ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],  # T
    ["00100", "00100", "11111", "00100", "00100", "00100", "00100"],  # +
    ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],  # H
    ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],  # 3
    ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],  # I
]


# ===========================================================================
# SECTION 2 — xorshift32 PRNG (repo-standard; no random/numpy). Also
# provides a Box-Muller Gaussian draw (cached second sample, the classic
# two-draws-per-pair technique) for the sensor-noise steps.
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9  # xorshift's state must never be exactly 0 (it would stay 0 forever)
        self._spare = None  # cached second Box-Muller sample

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def next_float01(self) -> float:
        """Uniform in [0, 1) — 32-bit resolution is plenty for 8-bit imagery."""
        return self.next_u32() / 4294967296.0

    def next_gaussian(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Standard Box-Muller transform, one cached sample per pair of
        uniform draws — the textbook way to turn two uniforms into two
        independent standard-normal samples without a library normal()."""
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return mean + std * v
        u1 = max(self.next_float01(), 1e-12)  # guard log(0)
        u2 = self.next_float01()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        self._spare = z1
        return mean + std * z0


def block_hash_level(bx: int, by: int, seed: int, levels) -> float:
    """Deterministic per-block hash (NOT part of the sequential RNG stream —
    a block's texture level must not depend on how many other draws
    happened before it) picking one of `levels`. A cheap integer mix
    (large-prime multiplies + xorshift32 avalanche) is enough for a
    cosmetic hashed-texture pattern; it is NOT used anywhere numerically
    load-bearing."""
    h = (seed * 2654435761 + bx * 2246822519 + by * 3266489917) & 0xFFFFFFFF
    rng = Xorshift32(h)
    idx = rng.next_u32() % len(levels)
    return levels[idx]


# ===========================================================================
# SECTION 3 — truth-image scene rasterization (block-constant per truth px).
# ===========================================================================
def make_truth_scene() -> list:
    """Return a K_H x K_W list-of-lists of float DN values: the shared
    ground-truth inspection scene both milestones study."""
    img = [[K_FLAT_DN for _ in range(K_W)] for _ in range(K_H)]  # background = the flat patch's own value

    # -- edge --------------------------------------------------------------
    x0, x1, y0, y1 = RECT_EDGE
    for y in range(y0, y1):
        for x in range(x0, x1):
            img[y][x] = K_EDGE_LO_DN if x < K_EDGE_STEP_X else K_EDGE_HI_DN

    # -- glyph row -----------------------------------------------------------
    gx0, gx1, gy0, gy1 = RECT_GLYPHS
    for y in range(gy0, gy1):
        for x in range(gx0, gx1):
            img[y][x] = K_GLYPH_LO_DN  # glyph-row background
    glyph_w = 5 * K_GLYPH_CELL_PX
    glyph_h = 7 * K_GLYPH_CELL_PX
    n_glyphs = len(GLYPHS)
    span = (n_glyphs - 1) * K_GLYPH_PITCH_PX + glyph_w
    start_x = gx0 + max(0, ((gx1 - gx0) - span) // 2)
    start_y = gy0 + max(0, ((gy1 - gy0) - glyph_h) // 2)
    for gi, bitmap in enumerate(GLYPHS):
        ox = start_x + gi * K_GLYPH_PITCH_PX
        for row, bits in enumerate(bitmap):
            for col, bit in enumerate(bits):
                if bit != "1":
                    continue
                px0 = ox + col * K_GLYPH_CELL_PX
                py0 = start_y + row * K_GLYPH_CELL_PX
                for dy in range(K_GLYPH_CELL_PX):
                    for dx in range(K_GLYPH_CELL_PX):
                        img[py0 + dy][px0 + dx] = K_GLYPH_HI_DN

    # -- hashed texture ------------------------------------------------------
    tx0, tx1, ty0, ty1 = RECT_TEXTURE
    nb_x = (tx1 - tx0) // K_TEXTURE_BLOCK_PX
    nb_y = (ty1 - ty0) // K_TEXTURE_BLOCK_PX
    for by in range(nb_y):
        for bx in range(nb_x):
            level = block_hash_level(bx, by, DEFAULT_SEED, K_TEXTURE_LEVELS)
            for dy in range(K_TEXTURE_BLOCK_PX):
                for dx in range(K_TEXTURE_BLOCK_PX):
                    img[ty0 + by * K_TEXTURE_BLOCK_PX + dy][tx0 + bx * K_TEXTURE_BLOCK_PX + dx] = level

    # -- bar-chart frequency groups -------------------------------------------
    for rect, period in ((RECT_BAR_COARSE, K_BAR_PERIOD_COARSE),
                         (RECT_BAR_MID, K_BAR_PERIOD_MID),
                         (RECT_BAR_FINE, K_BAR_PERIOD_FINE)):
        bx0, bx1, by0, by1 = rect
        for y in range(by0, by1):
            for x in range(bx0, bx1):
                phase = (x - bx0) % period
                img[y][x] = K_BAR_LO_DN if phase < period / 2.0 else K_BAR_HI_DN

    return img


# ===========================================================================
# SECTION 4 — motion-blur PSF rasterization.
#
# A camera translating at constant velocity during a GLOBAL-shutter
# exposure integrates the scene over a straight line segment of length
# K_BLUR_LENGTH_PX at angle K_BLUR_ANGLE_DEG (THEORY.md derives this from
# the exposure integral, specializing project 01.10's per-row rolling-
# shutter integral to a single global-shutter window). We rasterize that
# CONTINUOUS line into a discrete K_PSF_SIZE x K_PSF_SIZE kernel by
# sampling the segment densely (500 points) and BILINEARLY SPLATTING each
# sample's unit energy into the 4 nearest kernel cells — the same
# splat-with-bilinear-weights idea kernels.cu's shift_and_add_kernel uses,
# applied here once, offline, in Python. The result sums to 1.0 (energy-
# preserving) after a final exact normalization.
# ===========================================================================
def rasterize_line_psf(length_px: float, angle_deg: float, size: int) -> list:
    radius = size // 2
    kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    theta = math.radians(angle_deg)
    dxu, dyu = math.cos(theta), math.sin(theta)  # unit vector along the motion direction

    n_samples = 500  # dense enough that consecutive samples land < 1 supersample-equivalent apart
    total_weight = 0.0
    for i in range(n_samples):
        t = -length_px / 2.0 + length_px * (i / (n_samples - 1))  # t in [-L/2, +L/2]
        cx = radius + t * dxu   # continuous kernel-local x (center = radius)
        cy = radius + t * dyu
        x0 = math.floor(cx)
        y0 = math.floor(cy)
        fx = cx - x0
        fy = cy - y0
        w = 1.0 / n_samples
        for (ix, iy, wgt) in ((x0, y0, (1 - fx) * (1 - fy)),
                              (x0 + 1, y0, fx * (1 - fy)),
                              (x0, y0 + 1, (1 - fx) * fy),
                              (x0 + 1, y0 + 1, fx * fy)):
            if 0 <= ix < size and 0 <= iy < size:
                kernel[iy][ix] += w * wgt
                total_weight += w * wgt

    # Exact energy normalization: divide by whatever weight actually landed
    # inside the kernel window (should be ~1.0 already since L/2=4.5 <
    # radius=7 keeps every sample comfortably inside bounds; this line
    # guards floating-point drift rather than a real clipping loss).
    for y in range(size):
        for x in range(size):
            kernel[y][x] /= total_weight
    return kernel


def circular_convolve(img, psf, W, H, psf_size, psf_radius):
    """out[y][x] = sum psf[ky][kx] * img[(y+ky-r) mod H][(x+kx-r) mod W] —
    MUST MATCH kernels.cu's convolve_circular_kernel / reference_cpu.cpp's
    convolve_circular formula EXACTLY (the shared spatial-domain blur model
    every deconvolution method in this project inverts). Only the NONZERO
    taps are walked (a line PSF touches a small fraction of the dense
    K_PSF_SIZE^2 grid), which keeps this pure-Python convolution fast."""
    taps = [(ky, kx, psf[ky][kx]) for ky in range(psf_size) for kx in range(psf_size)
            if abs(psf[ky][kx]) > 1e-9]
    out = [[0.0 for _ in range(W)] for _ in range(H)]
    for y in range(H):
        for x in range(W):
            acc = 0.0
            for ky, kx, wgt in taps:
                sy = (y + ky - psf_radius) % H
                sx = (x + kx - psf_radius) % W
                acc += wgt * img[sy][sx]
            out[y][x] = acc
    return out


# ===========================================================================
# SECTION 5 — the oversized, nearest-upsampled supersample canvas (Section
# header explains why nearest-upsampling a block-constant truth image is
# EXACT for this project's synthetic content) and per-frame LR extraction.
# ===========================================================================
def make_oversampled_canvas(truth):
    """Pad `truth` with K_MARGIN_TRUTH_PX truth-pixels of background on
    every side, then nearest-upsample by K_SUPERSAMPLE. Returns the
    (ext_h*S) x (ext_w*S) supersample canvas plus the margin size in
    supersample units (both needed by extract_lr_frame below)."""
    ext_w = K_W + 2 * K_MARGIN_TRUTH_PX
    ext_h = K_H + 2 * K_MARGIN_TRUTH_PX
    ext = [[K_FLAT_DN for _ in range(ext_w)] for _ in range(ext_h)]
    for y in range(K_H):
        for x in range(K_W):
            ext[K_MARGIN_TRUTH_PX + y][K_MARGIN_TRUTH_PX + x] = truth[y][x]

    S = K_SUPERSAMPLE
    ss_w, ss_h = ext_w * S, ext_h * S
    canvas = [[0.0 for _ in range(ss_w)] for _ in range(ss_h)]
    for ey in range(ext_h):
        row_val = ext[ey]
        for ex in range(ext_w):
            v = row_val[ex]
            for dy in range(S):
                crow = canvas[ey * S + dy]
                for dx in range(S):
                    crow[ex * S + dx] = v
    margin_ss = K_MARGIN_TRUTH_PX * S
    return canvas, margin_ss


def extract_lr_frame(canvas, margin_ss, dx_ss: int, dy_ss: int):
    """Extract the K_W*S x K_H*S window at supersample offset (dx_ss,dy_ss)
    from the oversized canvas, then box-downsample by ratio =
    K_SUPERSAMPLE*K_LR_SCALE (8) to produce one K_LR_H x K_LR_W low-
    resolution frame — the SAME area-integration a real sensor pixel
    performs (THEORY.md "The problem"), so the frame is a physically
    honest (and genuinely alias-prone) capture."""
    ratio = K_SUPERSAMPLE * K_LR_SCALE  # 8 supersample px per LR px
    x0 = margin_ss + dx_ss
    y0 = margin_ss + dy_ss
    lr = [[0.0 for _ in range(K_LR_W)] for _ in range(K_LR_H)]
    inv_area = 1.0 / (ratio * ratio)
    for v in range(K_LR_H):
        sy0 = y0 + v * ratio
        for u in range(K_LR_W):
            sx0 = x0 + u * ratio
            acc = 0.0
            for dy in range(ratio):
                row = canvas[sy0 + dy]
                for dx in range(ratio):
                    acc += row[sx0 + dx]
            lr[v][u] = acc * inv_area
    return lr


# ===========================================================================
# SECTION 6 — file I/O: PGM (P5), PSF/shift/param CSVs.
# ===========================================================================
def write_pgm(path: Path, img, W: int, H: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{W} {H}\n255\n".encode("ascii"))
        buf = bytearray(W * H)
        i = 0
        for y in range(H):
            row = img[y]
            for x in range(W):
                v = row[x]
                v = 0.0 if v < 0.0 else (255.0 if v > 255.0 else v)
                buf[i] = int(v + 0.5)
                i += 1
        f.write(bytes(buf))


def write_psf_csv(path: Path, kernel, size: int, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {comment}\n")
        f.write(f"# SYNTHETIC data — generated by scripts/make_synthetic.py for project 01.22\n")
        f.write(f"{size},{size}\n")
        w = csv.writer(f)
        for y in range(size):
            w.writerow([f"{kernel[y][x]:.9f}" for x in range(size)])


def write_shifts_csv(path: Path, shifts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data — generated by scripts/make_synthetic.py for project 01.22\n")
        f.write("# shifts are GROUND TRUTH sub-pixel registration, in LR-pixel units (see kernels.cuh Section 3)\n")
        w = csv.writer(f)
        w.writerow(["frame", "dx_lrpx", "dy_lrpx"])
        for i, (dx, dy) in enumerate(shifts):
            w.writerow([i, f"{dx:.4f}", f"{dy:.4f}"])


def write_params_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("seed", DEFAULT_SEED),
        ("truth_w_px", K_W), ("truth_h_px", K_H),
        ("blur_length_px", K_BLUR_LENGTH_PX), ("blur_angle_deg", K_BLUR_ANGLE_DEG),
        ("blur_mismatch_angle_deg", K_MISMATCH_ANGLE_DEG),
        ("blur_noise_std_dn", K_BLUR_NOISE_STD_DN),
        ("psf_size", K_PSF_SIZE),
        ("lr_scale", K_LR_SCALE), ("lr_w_px", K_LR_W), ("lr_h_px", K_LR_H),
        ("num_frames", K_NUM_FRAMES), ("lr_noise_std_dn", K_LR_NOISE_STD_DN),
        ("supersample_factor", K_SUPERSAMPLE),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data generation parameters — project 01.22 (regenerate: python make_synthetic.py)\n")
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in rows:
            w.writerow([k, v])


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default_out, help="output directory (default: ../data/sample)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (default 42)")
    args = ap.parse_args()
    out_dir: Path = args.out

    print(f"[make_synthetic] building the shared inspection scene ({K_W}x{K_H})...")
    truth = make_truth_scene()
    write_pgm(out_dir / "truth.pgm", truth, K_W, K_H)

    # ---- milestone 1: PSF + blurred frame ----------------------------------
    print("[make_synthetic] rasterizing motion-blur PSFs (truth + mismatch angle)...")
    psf_truth = rasterize_line_psf(K_BLUR_LENGTH_PX, K_BLUR_ANGLE_DEG, K_PSF_SIZE)
    psf_mismatch = rasterize_line_psf(K_BLUR_LENGTH_PX, K_MISMATCH_ANGLE_DEG, K_PSF_SIZE)
    write_psf_csv(out_dir / "psf_truth.csv", psf_truth, K_PSF_SIZE,
                 f"line PSF, length={K_BLUR_LENGTH_PX}px angle={K_BLUR_ANGLE_DEG}deg, sums to 1.0")
    write_psf_csv(out_dir / "psf_mismatch.csv", psf_mismatch, K_PSF_SIZE,
                 f"MISMATCHED line PSF (wrong angle, PSF-mismatch honesty test), "
                 f"length={K_BLUR_LENGTH_PX}px angle={K_MISMATCH_ANGLE_DEG}deg, sums to 1.0")

    print("[make_synthetic] circularly convolving truth with psf_truth + adding sensor noise...")
    blurred = circular_convolve(truth, psf_truth, K_W, K_H, K_PSF_SIZE, K_PSF_RADIUS)
    rng = Xorshift32(args.seed)
    for y in range(K_H):
        for x in range(K_W):
            blurred[y][x] += rng.next_gaussian(0.0, K_BLUR_NOISE_STD_DN)
    write_pgm(out_dir / "blurred.pgm", blurred, K_W, K_H)

    # ---- milestone 2: supersampled canvas + shifted, noisy LR frames -------
    print("[make_synthetic] building the 4x supersampled canvas for low-res frame extraction...")
    canvas, margin_ss = make_oversampled_canvas(truth)

    # 8 quarter-LR-pixel shifts (documented lattice, task brief): diverse
    # coverage of the unit LR cell so the 2x HR grid receives dense,
    # varied sub-pixel phase information from shift-and-add (milestone 2's
    # first stage). dx/dy in LR-pixel units; ratio (8 supersample px per
    # LR px) makes dx*8/dy*8 exactly integer for every quarter-pixel value.
    shifts_lrpx = [
        (0.00, 0.00), (0.50, 0.00), (0.00, 0.50), (0.50, 0.50),
        (0.25, 0.75), (0.75, 0.25), (0.75, 0.75), (0.25, 0.25),
    ]
    assert len(shifts_lrpx) == K_NUM_FRAMES
    write_shifts_csv(out_dir / "shifts_truth.csv", shifts_lrpx)

    lr_rng = Xorshift32(args.seed + 1)  # a DIFFERENT stream than the blur noise, so the two noise sources are independent
    for i, (dx_lr, dy_lr) in enumerate(shifts_lrpx):
        print(f"[make_synthetic] extracting LR frame {i} (shift dx={dx_lr:.2f}, dy={dy_lr:.2f} LR-px)...")
        dx_ss = int(round(dx_lr * K_LR_SCALE * K_SUPERSAMPLE))  # exact integer (multiples of 0.25 LR-px * 8 = multiples of 2)
        dy_ss = int(round(dy_lr * K_LR_SCALE * K_SUPERSAMPLE))
        lr = extract_lr_frame(canvas, margin_ss, dx_ss, dy_ss)
        for v in range(K_LR_H):
            for u in range(K_LR_W):
                lr[v][u] += lr_rng.next_gaussian(0.0, K_LR_NOISE_STD_DN)
        write_pgm(out_dir / f"lr_frame_{i}.pgm", lr, K_LR_W, K_LR_H)

    write_params_csv(out_dir / "params.csv")

    print(f"[make_synthetic] done. Wrote truth.pgm, blurred.pgm, psf_truth.csv, psf_mismatch.csv, "
         f"{K_NUM_FRAMES} lr_frame_*.pgm, shifts_truth.csv, params.csv to {out_dir}  (all SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
