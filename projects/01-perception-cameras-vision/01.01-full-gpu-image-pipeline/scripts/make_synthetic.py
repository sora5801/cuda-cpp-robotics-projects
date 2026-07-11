#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 01.01
(Full GPU image pipeline: debayer -> undistort -> rectify -> resize -> normalize).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
This project needs a raw Bayer image with a known-correct, dense ground
truth for "what the ideal, undistorted, rectified camera would have seen"
— exactly the kind of thing a real camera rig cannot give you for free
(you would need a calibration target, a checkerboard-detection tool, and
still only get SPARSE corner ground truth) and synthetic authorship gives
for free and EXACTLY.

The generation direction is the physically honest one, and it is the
EXACT INVERSE of what the pipeline undoes (the whole point of an
end-to-end synthetic test): we author the scene directly in the IDEAL,
RECTIFIED camera's pixel grid (true_rgb.ppm — a checkerboard for the
straightness gate, a smooth low-frequency gradient plus three flat-color
disks for the color-fidelity gate), then WARP it backward through the
camera model — rotate into the raw camera's frame, apply Brown-Conrady
distortion, mosaic to RGGB — to produce the raw Bayer sensor image the
pipeline actually consumes (bayer_input.pgm). Un-warping THAT is exactly
main.cu's job; if generator and pipeline agree on the camera model (they
share every numeric constant below, cross-checked by the "MUST MATCH"
comments), the pipeline's rectified output should closely reproduce
true_rgb.ppm — main.cu's color-fidelity gate measures exactly how closely.

Because there is no closed-form inverse of Brown-Conrady distortion (see
../src/kernels.cuh's file header), turning a RAW pixel into the IDEAL
pixel it should show requires an iterative fixed-point undistort — this
script implements it independently in Python (raw_pixel_to_ideal() below),
deliberately mirroring, but never importing, the C++ camera model. This
is the generator's OWN math, not test code, but it plays the same
"independent reimplementation" role as reference_cpu.cpp does for the
kernels: three separate parties (this script, kernels.cu, reference_cpu.cpp)
each compute the SAME physical camera model from the SAME five numbers
(fx, fy, cx, cy, k1, k2, p1, p2, rectify angle) — agreement across three
independent implementations is strong evidence the physics, not just the
code, is right.

What it writes (into ../data/sample/, well under the CLAUDE.md paragraph 8
budget):

    bayer_input.pgm   FULL_W x FULL_H, 8-bit grayscale (PGM P5) — the RGGB
                       Bayer mosaic the pipeline reads as its ONLY input.
    true_rgb.ppm       FULL_W x FULL_H, 8-bit RGB (PPM P6) — the ideal,
                       undistorted, rectified scene, ground truth for the
                       color-fidelity gate (main.cu compares the pipeline's
                       rectified-stage output against this directly).
    smooth_mask.pgm    FULL_W x FULL_H, 8-bit grayscale (PGM P5) — 255 =
                       this pixel is far enough from any analytic edge
                       (checkerboard, disk boundary, image border) that a
                       few pixels of geometric/interpolation error cannot
                       hide a real color mismatch; 0 = near an edge, where
                       even a PERFECT pipeline shows some blend error, so
                       these pixels are reported separately, not gated.

The scene (fully specified here so this file IS the scene's spec; THEORY.md
and main.cu's straightness-gate comment cross-reference these numbers):

    Checkerboard  — x in [32, 224), y in [32, 224), CB_N=8 x CB_N=8 squares
                    of CB_SQUARE=24 px, alternating CB_WHITE/CB_BLACK
                    (grayscale, R=G=B, for an unambiguous 50%-crossing edge
                    detector in main.cu's straightness gate).
    Gradient bg   — everywhere else: a smooth, LOW spatial-frequency
                    3-channel cosine field (see gradient_rgb()) — smooth on
                    purpose, so a handful of pixels of remap/resize error
                    cannot swing the color-fidelity gate's mean.
    3 flat disks  — solid colors, drawn over the gradient (never over the
                    checkerboard — geometry keeps them disjoint), giving
                    visual variety and three more flat-color regions the
                    color-fidelity gate can score confidently away from
                    their boundaries.

Usage:
    python make_synthetic.py                  # the committed 384x288 scene
    python make_synthetic.py --width 256 --height 192   # experiments; do not commit
"""

import argparse
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Camera model constants — MUST MATCH ../src/kernels.cuh exactly (that file
# is the single source of truth for the C++ side; this comment is the
# cross-reference CLAUDE.md paragraph 12 asks every duplicated constant to
# carry). Unlike kernels.cuh (float32, hand-rounded trig constants for
# host/device bit-identity — see that file's comment), this script uses
# Python's native double-precision math.cos/math.sin; the ~1e-7 difference
# from kernels.cuh's rounded float32 constants is many orders of magnitude
# below anything that matters at pixel resolution.
# ---------------------------------------------------------------------------
FULL_W = 384                       # must match kFullW
FULL_H = 288                       # must match kFullH
FX = 380.0                         # must match kFx
FY = 380.0                         # must match kFy
CX = (FULL_W - 1) * 0.5            # must match kCx = 191.5
CY = (FULL_H - 1) * 0.5            # must match kCy = 143.5
K1, K2 = -0.22, 0.06                # must match kK1, kK2
P1, P2 = 0.0010, -0.0008            # must match kP1, kP2
RECT_ANGLE_DEG = 2.0                # must match kRectifyAngleDeg
RECT_COS = math.cos(math.radians(RECT_ANGLE_DEG))
RECT_SIN = math.sin(math.radians(RECT_ANGLE_DEG))

UNDISTORT_ITERS = 30                # fixed-point undistort iterations (see raw_pixel_to_ideal)

# ---------------------------------------------------------------------------
# Scene layout constants — MUST MATCH main.cu's straightness-gate and
# color-fidelity-gate constants (search main.cu for "MUST MATCH
# scripts/make_synthetic.py"). Values chosen so the checkerboard, the three
# disks, and the border margin are all mutually disjoint by construction
# (verified by inspection, not by a runtime check — this is scene
# AUTHORSHIP, not a general-purpose layout solver).
# ---------------------------------------------------------------------------
CB_X0, CB_Y0 = 32, 32               # checkerboard top-left corner, px (ideal/rectified frame)
CB_SQUARE = 24                      # checkerboard square size, px
CB_N = 8                            # checkerboard is CB_N x CB_N squares -> 192x192 px, x/y in [32,224)
CB_WHITE, CB_BLACK = 235, 20        # grayscale square intensities (R=G=B), max-contrast for edge detection

# (center_x, center_y, radius_px, (R,G,B)) — all outside the checkerboard
# rect [32,224)x[32,224) by construction (x=300 disks sit right of it;
# the x=110 disk sits below it, y in [227,283) vs checkerboard's y<224).
DISKS = [
    (300, 80, 35, (220, 60, 60)),    # red disk, upper right
    (300, 200, 32, (60, 200, 90)),   # green disk, mid right
    (110, 255, 28, (70, 90, 220)),   # blue disk, lower left (below the checkerboard)
]

MASK_MARGIN = 6                     # px excluded from smooth_mask near any edge (border/checkerboard/disk boundary)


# ---------------------------------------------------------------------------
# bayer_channel_at — Python mirror of kernels.cuh's bayer_channel_at(): RGGB
# tiling, 0=R, 1=G, 2=B. Deliberately re-typed (not imported — this is a
# different language) so the mosaic step below is an independent
# statement of the same hardware fact the C++ side relies on.
# ---------------------------------------------------------------------------
def bayer_channel_at(x: int, y: int) -> int:
    even_row = (y % 2) == 0
    even_col = (x % 2) == 0
    if even_row and even_col:
        return 0   # R
    if (not even_row) and (not even_col):
        return 2   # B
    return 1       # G


# ---------------------------------------------------------------------------
# gradient_rgb — the smooth background field. Low spatial frequency by
# construction (periods are a large fraction of the image size), so
# neighboring pixels are nearly identical — exactly the property the
# color-fidelity gate needs to be insensitive to a few pixels of remap
# geometric error while still being sensitive to a genuine pipeline bug
# (a wrong channel order, a systematic bias, a badly miscalibrated k1).
# Returns a (R, G, B) tuple of floats in roughly [68, 188].
# ---------------------------------------------------------------------------
def gradient_rgb(x: float, y: float):
    r = 128.0 + 60.0 * math.cos(2.0 * math.pi * x / (1.3 * FULL_W))
    g = 128.0 + 60.0 * math.cos(2.0 * math.pi * y / (1.3 * FULL_H) + 1.0)
    b = 128.0 + 60.0 * math.sin(2.0 * math.pi * (x + y) / (1.7 * (FULL_W + FULL_H)))
    return r, g, b


def in_checkerboard(x: int, y: int) -> bool:
    return CB_X0 <= x < CB_X0 + CB_SQUARE * CB_N and CB_Y0 <= y < CB_Y0 + CB_SQUARE * CB_N


def checkerboard_color(x: int, y: int):
    ix = (x - CB_X0) // CB_SQUARE
    iy = (y - CB_Y0) // CB_SQUARE
    v = CB_WHITE if ((ix + iy) % 2 == 0) else CB_BLACK
    return float(v), float(v), float(v)


def disk_color_at(x: float, y: float):
    """Return the (R,G,B) of the first disk containing (x, y), or None."""
    for (dcx, dcy, r, color) in DISKS:
        dx, dy = x - dcx, y - dcy
        if dx * dx + dy * dy <= r * r:
            return float(color[0]), float(color[1]), float(color[2])
    return None


def scene_at(x: int, y: int):
    """The TRUE (ideal, rectified-frame) color at INTEGER pixel (x, y) —
    the generative rule true_rgb.ppm is a direct raster of. Checkerboard
    wins inside its rect; disks win outside it; the gradient fills
    everything else — a simple painter's-algorithm composite, safe because
    the three regions are disjoint by construction (see the constants
    above)."""
    if in_checkerboard(x, y):
        return checkerboard_color(x, y)
    d = disk_color_at(float(x), float(y))
    if d is not None:
        return d
    return gradient_rgb(float(x), float(y))


def is_smooth(x: int, y: int) -> bool:
    """smooth_mask predicate: True iff (x,y) is far (>= MASK_MARGIN px)
    from the image border, the checkerboard rectangle, AND every disk
    boundary. See the file header for why this matters to the
    color-fidelity gate."""
    if x < MASK_MARGIN or x >= FULL_W - MASK_MARGIN or y < MASK_MARGIN or y >= FULL_H - MASK_MARGIN:
        return False
    cb_x0, cb_x1 = CB_X0 - MASK_MARGIN, CB_X0 + CB_SQUARE * CB_N + MASK_MARGIN
    cb_y0, cb_y1 = CB_Y0 - MASK_MARGIN, CB_Y0 + CB_SQUARE * CB_N + MASK_MARGIN
    if cb_x0 <= x < cb_x1 and cb_y0 <= y < cb_y1:
        return False
    for (dcx, dcy, r, _color) in DISKS:
        dist = math.hypot(x - dcx, y - dcy)
        if abs(dist - r) <= MASK_MARGIN:
            return False
    return True


# ---------------------------------------------------------------------------
# bilinear_sample_true — sample the RASTERIZED true_rgb grid (a Python list
# of (R,G,B) floats, row-major) at a fractional coordinate, clamp-to-edge —
# the SAME boundary policy ../src/kernels.cu's bilinear_sample_rgb() uses,
# so a raw pixel whose inverse-mapped ideal coordinate lands just outside
# the frame is handled identically by the generator and the pipeline that
# later re-inverts it.
# ---------------------------------------------------------------------------
def bilinear_sample_true(true_grid, W: int, H: int, x: float, y: float):
    x = min(max(x, 0.0), float(W - 1))
    y = min(max(y, 0.0), float(H - 1))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    fx = x - x0
    fy = y - y0
    r00, g00, b00 = true_grid[y0 * W + x0]
    r10, g10, b10 = true_grid[y0 * W + x1]
    r01, g01, b01 = true_grid[y1 * W + x0]
    r11, g11, b11 = true_grid[y1 * W + x1]
    r = (r00 + (r10 - r00) * fx) * (1 - fy) + (r01 + (r11 - r01) * fx) * fy
    g = (g00 + (g10 - g00) * fx) * (1 - fy) + (g01 + (g11 - g01) * fx) * fy
    b = (b00 + (b10 - b00) * fx) * (1 - fy) + (b01 + (b11 - b01) * fx) * fy
    return r, g, b


# ---------------------------------------------------------------------------
# raw_pixel_to_ideal — the generator's core: given a RAW SENSOR pixel
# coordinate (u, v), find the coordinate in the IDEAL/RECTIFIED true_rgb
# grid whose light a real lens would have bent onto that raw pixel. This is
# the INVERSE of ../src/kernels.cuh's compute_source_pixel() (which goes
# ideal -> raw); since Brown-Conrady distortion has no closed-form inverse,
# this uses the classic fixed-point iteration (the same one OpenCV's
# undistortPoints uses internally, and the same one main.cu's INDEPENDENT
# roundtrip gate re-derives in C++ — three independent implementations of
# the same fixed-point scheme, cross-checking the physics from three
# directions).
#
# Steps:
#   1. Raw pixel -> distorted normalized coords (apply K^-1).
#   2. Fixed-point iterate to the UNDISTORTED normalized coords in the raw
#      camera's own frame (Newton-free "Gauss-Seidel" scheme: assume the
#      distortion is small, use the distorted point as the seed, and
#      repeatedly re-solve holding the higher-order terms fixed).
#   3. Rotate that ray into the RECTIFIED frame with R_rect_raw (the
#      FORWARD rotation this time — see kernels.cuh's file header for why
#      going raw -> rect uses R, not R^T).
#   4. Perspective-divide and apply K to land in ideal PIXEL coordinates.
# ---------------------------------------------------------------------------
def raw_pixel_to_ideal(u: float, v: float):
    xd = (u - CX) / FX
    yd = (v - CY) / FY

    xu, yu = xd, yd     # seed the iteration at the distorted point itself
    for _ in range(UNDISTORT_ITERS):
        r2 = xu * xu + yu * yu
        icdist = 1.0 / (1.0 + K1 * r2 + K2 * r2 * r2)
        delta_x = 2.0 * P1 * xu * yu + P2 * (r2 + 2.0 * xu * xu)
        delta_y = P1 * (r2 + 2.0 * yu * yu) + 2.0 * P2 * xu * yu
        xu = (xd - delta_x) * icdist
        yu = (yd - delta_y) * icdist

    # Rotate raw-frame ray -> rectified-frame ray: v_rect = R_rect_raw * v_raw
    #   R_rect_raw = [ c, 0, s ]      applied to (xu, yu, 1):
    #                [ 0, 1, 0 ]
    #                [-s, 0, c ]
    c, s = RECT_COS, RECT_SIN
    ox = c * xu + s
    oy = yu
    oz = -s * xu + c

    xi = ox / oz
    yi = oy / oz
    return FX * xi + CX, FY * yi + CY


# ---------------------------------------------------------------------------
# gen_scene — build true_rgb, smooth_mask (both rasterized directly from
# scene_at()/is_smooth() over the IDEAL grid), then bayer_input by
# INVERSE-warping every raw pixel back through the camera model and
# mosaicking (see the file header for the full pipeline of this function).
# ---------------------------------------------------------------------------
def gen_scene(width: int, height: int):
    assert width == FULL_W and height == FULL_H, (
        "this generator's camera model constants (fx,fy,cx,cy,k1,k2,...) are "
        "tuned for the committed 384x288 scene; a different size needs its "
        "own constants to stay physically consistent — experiment via "
        "--width/--height only with that caveat in mind, and never commit "
        "a non-default size (see main())")

    # ---- true_rgb + smooth_mask: direct rasterization over the ideal grid ----
    true_grid = [None] * (width * height)   # list of (R,G,B) floats, row-major
    true_rgb_bytes = bytearray(width * height * 3)
    mask_bytes = bytearray(width * height)
    for y in range(height):
        row = y * width
        for x in range(width):
            r, g, b = scene_at(x, y)
            true_grid[row + x] = (r, g, b)
            o = (row + x) * 3
            true_rgb_bytes[o + 0] = int(round(min(max(r, 0.0), 255.0)))
            true_rgb_bytes[o + 1] = int(round(min(max(g, 0.0), 255.0)))
            true_rgb_bytes[o + 2] = int(round(min(max(b, 0.0), 255.0)))
            mask_bytes[row + x] = 255 if is_smooth(x, y) else 0

    # ---- bayer_input: inverse-warp + mosaic every RAW pixel ----
    bayer_bytes = bytearray(width * height)
    for v in range(height):
        row = v * width
        for u in range(width):
            xi, yi = raw_pixel_to_ideal(float(u), float(v))
            r, g, b = bilinear_sample_true(true_grid, width, height, xi, yi)
            ch = bayer_channel_at(u, v)
            val = (r, g, b)[ch]
            bayer_bytes[row + u] = int(round(min(max(val, 0.0), 255.0)))

    return bytes(true_rgb_bytes), bytes(mask_bytes), bytes(bayer_bytes)


def write_pgm(path: Path, width: int, height: int, data: bytes) -> None:
    """8-bit binary PGM (P5) — one grayscale byte per pixel. Same format
    used throughout this repo (e.g. 01.02's write_pgm); no library needed
    to read or write it."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(data)


def write_ppm(path: Path, width: int, height: int, data: bytes) -> None:
    """8-bit binary PPM (P6) — three interleaved RGB bytes per pixel,
    IDENTICAL to this project's internal RGB layout (see kernels.cuh's
    IMAGE LAYOUTS section) — writing/reading a PPM is a direct memcpy."""
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=FULL_W, help=f"image width in px (default {FULL_W})")
    ap.add_argument("--height", type=int, default=FULL_H, help=f"image height in px (default {FULL_H})")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default ../data/sample)")
    args = ap.parse_args()

    true_rgb, mask, bayer = gen_scene(args.width, args.height)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_pgm(args.out_dir / "bayer_input.pgm", args.width, args.height, bayer)
    write_ppm(args.out_dir / "true_rgb.ppm", args.width, args.height, true_rgb)
    write_pgm(args.out_dir / "smooth_mask.pgm", args.width, args.height, mask)

    n_smooth = sum(1 for b in mask if b)
    total = args.width * args.height
    total_bytes = sum((args.out_dir / n).stat().st_size
                      for n in ("bayer_input.pgm", "true_rgb.ppm", "smooth_mask.pgm"))
    print(f"wrote {args.out_dir} : {args.width}x{args.height} RGGB Bayer scene "
          f"({total_bytes} bytes total across 3 files) - labeled SYNTHETIC")
    print(f"note: {n_smooth}/{total} pixels ({100.0*n_smooth/total:.1f}%) are smooth_mask-valid "
          f"(>= {MASK_MARGIN}px from the border, the checkerboard, and every disk boundary)")
    if args.width != FULL_W or args.height != FULL_H:
        print("note: non-default size - fine for experiments, do NOT commit these files "
              "(the camera model constants above are tuned for the committed size)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
