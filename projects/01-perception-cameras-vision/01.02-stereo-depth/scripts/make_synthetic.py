#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 01.02
(Stereo depth: block matching, then Semi-Global Matching (SGM) kernels).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
Stereo evaluation needs a RECTIFIED image pair with known-correct, DENSE
ground-truth disparity — exactly the kind of thing that is easy to fabricate
perfectly and impossible to get for free from a real camera (real rigs need
a laser scanner or a second sensing modality just to get ground truth, and
even then it is sparse). So we author the scene directly in DISPARITY SPACE:
a textured ground plane receding into the distance plus a few textured
fronto-parallel rectangles nearer the camera, at different constant
disparities. The right image is produced by physically-correct forward
(scatter) warping of the left image with a per-pixel z-buffer, so occlusion
falls out for free and is exactly known — no photogrammetry, no external
tools, fully reproducible from one fixed seed.

What it writes (into ../data/sample/, ~<300 KiB total, well under the 1 MiB
budget in CLAUDE.md paragraph 8):

    left.pgm         384x288, 8-bit grayscale (PGM P5) — the reference image
    right.pgm        384x288, 8-bit grayscale (PGM P5) — the matching image
    gt_disparity.pgm 384x288, 8-bit grayscale — ground-truth disparity * 4
                      (raw disparity is 0..63; *4 keeps every level visually
                      distinct in a viewer without leaving byte range: 63*4=252)
    gt_valid.pgm     384x288, 8-bit grayscale — 255 = this left pixel has a
                      genuine, unoccluded, in-frame correspondence in
                      right.pgm and should be SCORED; 0 = occluded / off-frame
                      / inside the census border margin (see kernels.cuh
                      kCensusHalf) — never scored, because no window-based
                      stereo method could ever produce a meaningful answer
                      there, ground-truth or not.

The scene (fully described here so the file IS the scene's specification;
kernels.cuh's header comment cross-references these numbers):

    Ground plane   — disparity varies smoothly with ROW only, d_bg(y):
                     4 px (far, top of frame) .. 18 px (near, bottom) — a
                     physically honest simplification (see THEORY.md): a
                     flat ground plane viewed by a forward-looking camera
                     has depth that is a function of image row alone, so
                     disparity = f*B/Z is too. This also makes the
                     occlusion-fill math below exact instead of iterative.
    Rectangle A    — x[40,150) y[40,150),   disparity 26 (nearer than bg)
    Rectangle B    — x[110,230) y[120,230), disparity 40 (nearer than A —
                     the two overlap in x[110,150)*y[120,150); B wins there,
                     producing an object-vs-object occlusion edge, not just
                     object-vs-background)
    Rectangle C    — x[260,350) y[60,180),  disparity 52 (nearest, isolated)

All disparities stay under kMaxDisp=64 (D) with headroom to spare. Texture
is 2-octave deterministic value noise (a lattice-hash + smoothstep + bilinear
interpolation — the "write it by hand" version of Perlin-style noise),
seeded from a single base seed 42 (background) with per-layer offsets
43/44/45 (rectangles) — everything traces back to one documented seed.

Usage:
    python make_synthetic.py                 # the committed 384x288 scene
    python make_synthetic.py --width 256 --height 192   # experiments; do not commit
"""

import argparse
import math
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene constants — MUST match the disparity range assumed by kernels.cuh
# (kMaxDisp = 64, i.e. valid disparities are 0..63). Kept as module constants
# (not CLI flags) because they are part of the taught, fixed scene, not
# something a learner should casually vary and still expect the committed
# gt_valid.pgm / expected_output.txt to apply.
# ---------------------------------------------------------------------------
BASE_SEED = 42                 # the repo-documented seed; layers derive from it
D_MAX = 64                     # must match kMaxDisp in ../src/kernels.cuh
DISP_SCALE = 4                 # gt_disparity.pgm stores disparity * DISP_SCALE
CENSUS_HALF = 3                # must match kCensusHalf in ../src/kernels.cuh
                                # (7x7 census window, half-width 3) — pixels
                                # within this margin of any edge can never
                                # produce a census signature, so gt_valid
                                # excludes them from scoring (see file header)

BG_FAR_DISP = 4                # background disparity at the top row (far)
BG_NEAR_DISP = 18              # background disparity at the bottom row (near)

# (x0, x1, y0, y1, disparity, seed, noise_scale_px) — drawn in this order,
# FARTHEST first, so a nearer rectangle painted later correctly overwrites a
# farther one in any overlap (a one-line "painter's algorithm" z-sort,
# because the list is already sorted by ascending disparity == descending depth).
RECTANGLES = [
    (40, 150, 40, 150, 26, BASE_SEED + 1, 26.0),   # Rectangle A
    (110, 230, 120, 230, 40, BASE_SEED + 2, 24.0), # Rectangle B (overlaps A; nearer)
    (260, 350, 60, 180, 52, BASE_SEED + 3, 22.0),  # Rectangle C (isolated, nearest)
]


# ---------------------------------------------------------------------------
# Deterministic integer hash -> float in [0, 1). This is the ONLY source of
# randomness in the whole generator, and it is a pure function of (ix, iy,
# seed) — no global RNG state, so the image is reproducible from these three
# integers alone regardless of iteration order (unlike random.Random, whose
# output depends on call ORDER — a subtle reproducibility trap this avoids).
# The constants are the classic large-odd-prime "multiplicative hash" mix
# (same family as Squirrel3/PCG-style integer hashes); it is not
# cryptographic, just decorrelated enough for texture.
# ---------------------------------------------------------------------------
def _hash01(ix: int, iy: int, seed: int) -> float:
    h = (ix * 374761393 + iy * 668265263 + seed * 2654435761) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / float(1 << 24)   # top 24 bits -> [0,1)


def _smoothstep(t: float) -> float:
    """Hermite smoothing t*t*(3-2t): the standard trick that makes lattice
    value-noise C1-continuous instead of showing grid-aligned creases —
    without it, bilinear interpolation of the raw hash values produces
    visible diamond artifacts at integer lattice coordinates."""
    return t * t * (3.0 - 2.0 * t)


def _value_noise(x: float, y: float, seed: int, scale: float) -> float:
    """Single-octave value noise at continuous image coordinates (x, y).

    Lays a lattice of `scale`-pixel cells over the image, hashes a
    pseudo-random value at each lattice CORNER, and bilinearly interpolates
    (with smoothstep-warped weights) between the 4 corners surrounding
    (x/scale, y/scale). Returns a value in [0, 1]. This is "write your own
    Perlin noise" in about 10 lines — the teaching point being that texture
    synthesis is just hashing + interpolation, no library required.
    """
    fx, fy = x / scale, y / scale
    ix, iy = math.floor(fx), math.floor(fy)
    tx, ty = _smoothstep(fx - ix), _smoothstep(fy - iy)
    v00 = _hash01(ix, iy, seed)
    v10 = _hash01(ix + 1, iy, seed)
    v01 = _hash01(ix, iy + 1, seed)
    v11 = _hash01(ix + 1, iy + 1, seed)
    a = v00 + (v10 - v00) * tx
    b = v01 + (v11 - v01) * tx
    return a + (b - a) * ty


def texture_byte(x: float, y: float, seed: int, scale: float) -> int:
    """2-octave fractal value noise -> an 8-bit pixel intensity.

    Octave 1 (frequency 1/scale, weight 0.8) supplies a COARSE, low-
    frequency pattern; octave 2 (frequency 2/scale, a different seed
    offset, weight 0.2) adds a little fine detail on top. The mix is
    deliberately COARSE-dominated: within a single 7x7 census window (or
    even within a 64-px disparity search range), neighboring patches often
    look nearly identical — real matching AMBIGUITY, the classic failure
    mode of pure window matching that this project exists to make visible
    (THEORY.md "why census/why SGM"). A texture with strong high-frequency
    detail at every pixel (an earlier version of this generator) makes
    window matching too easy to be a fair test — every window is locally
    unique, and even a naive per-pixel winner-take-all rarely gets
    confused. The [0,1] sum is mapped to [20, 235] — inside the byte range,
    never pure black/white, so no region saturates and every census
    comparison stays meaningful.
    """
    n1 = _value_noise(x, y, seed, scale)
    n2 = _value_noise(x, y, seed + 1000, scale * 0.5)
    v = 0.8 * n1 + 0.2 * n2                      # in [0, 1]
    return int(round(20 + v * (235 - 20)))


# ---------------------------------------------------------------------------
# Scene evaluation: for ANY (x, y) in the LEFT image, what disparity and what
# left-image texture intensity is there? Pure functions of (x, y) — this is
# what makes exact occlusion-fill possible below (see gen_scene's docstring).
# ---------------------------------------------------------------------------
def bg_disparity(y: int, height: int) -> int:
    """Ground-plane disparity at row y: linear in y, far (small d) at the
    top, near (large d) at the bottom — see the file header for the physical
    reading (a flat ground plane's depth is a function of image row alone)."""
    t = y / float(height - 1)
    return int(round(BG_FAR_DISP + (BG_NEAR_DISP - BG_FAR_DISP) * t))


def scene_at(x: int, y: int, width: int, height: int):
    """Return (disparity, texture_byte) for LEFT-image pixel (x, y).

    Starts from the background, then overwrites with any rectangle that
    contains (x, y), in far-to-near order (RECTANGLES is pre-sorted by
    ascending disparity) — the last, nearest match wins, which is exactly
    the correct compositing rule (nearer surfaces occlude farther ones).
    """
    d = bg_disparity(y, height)
    tex = texture_byte(x, y, BASE_SEED, 28.0)     # background: coarse, seed 42
    for (x0, x1, y0, y1, disp, seed, scale) in RECTANGLES:
        if x0 <= x < x1 and y0 <= y < y1:
            d = disp
            tex = texture_byte(x, y, seed, scale)
    return d, tex


# ---------------------------------------------------------------------------
# gen_scene — build left.pgm, right.pgm, gt_disparity, gt_valid.
#
# The right image is built by FORWARD (scatter) warping every left pixel to
# its right-image column xR = x - d, resolved with a per-column Z-BUFFER: if
# two left pixels map to the same xR (a foreground object shifts "past" the
# background it sits in front of), the one with the LARGER disparity (closer
# to the camera) wins — exactly the physical rule "the nearer surface
# occludes the farther one". This single pass gives us, for free and
# exactly:
#   (a) the right image's pixel content,
#   (b) which LEFT pixels are OCCLUDED in the right view (they did not win
#       the z-buffer at their target column, or their target column fell
#       outside the frame) — these become gt_valid = 0,
#   (c) DISOCCLUSION holes in the right image (columns no left pixel ever
#       claimed — background revealed in the right view that a foreground
#       object hides in the left view). Because the background's disparity
#       depends on row y ONLY (see bg_disparity), the hole at (xR, y) can be
#       filled EXACTLY, no iteration needed: the background pixel that would
#       appear there is at left-column x = xR + bg_disparity(y).
# ---------------------------------------------------------------------------
def gen_scene(width: int, height: int):
    left = bytearray(width * height)
    gt_disp = bytearray(width * height)
    gt_valid = bytearray(width * height)
    right = bytearray(width * height)

    for y in range(height):
        row_off = y * width

        # Pass 1: evaluate every LEFT column once (scene_at is a pure
        # function, so this doubles as both the left image AND the source
        # data the scatter pass below reads — computed once, used twice).
        row_disp = [0] * width
        row_tex = [0] * width
        for x in range(width):
            d, tex = scene_at(x, y, width, height)
            row_disp[x] = d
            row_tex[x] = tex
            left[row_off + x] = tex

        # Pass 2: scatter to the right image with a z-buffer over disparity.
        # win_d[xR]   = the highest disparity (nearest surface) that has
        #               claimed target column xR so far (-1 = unclaimed).
        # win_tex[xR] = that winner's texture intensity.
        win_d = [-1] * width
        win_tex = [0] * width
        for x in range(width):
            d = row_disp[x]
            xr = x - d
            if 0 <= xr < width and d > win_d[xr]:
                win_d[xr] = d
                win_tex[xr] = row_tex[x]

        # Pass 3: fill disocclusion holes in the right image exactly, using
        # the row-only background disparity function (see docstring above).
        for xr in range(width):
            if win_d[xr] < 0:
                x_bg = xr + bg_disparity(y, height)
                win_tex[xr] = texture_byte(x_bg, y, BASE_SEED, 14.0)
            right[row_off + xr] = win_tex[xr]

        # Pass 4: ground truth for the LEFT frame — a left pixel is VALID
        # (scoreable) iff its own disparity is the one that actually won the
        # z-buffer at its target column (i.e. it is genuinely visible in the
        # right image) AND it sits outside the census border margin.
        border = CENSUS_HALF
        for x in range(width):
            d = row_disp[x]
            xr = x - d
            visible = (0 <= xr < width) and (win_d[xr] == d)
            in_margin = (x < border or x >= width - border
                         or y < border or y >= height - border)
            gt_disp[row_off + x] = min(255, d * DISP_SCALE)
            gt_valid[row_off + x] = 0 if (not visible or in_margin) else 255

    return left, right, gt_disp, gt_valid


def write_pgm(path: Path, width: int, height: int, data: bytes) -> None:
    """Write an 8-bit binary PGM (P5) — the smallest real image format there
    is: one ASCII header line trio, then raw bytes. Viewable in any image
    tool (GIMP, IrfanView, VS Code image preview); needs no libraries to
    read or write, which is why this repo uses it for every image artifact
    (see also 07.09's write_pgm in src/main.cu — same format, same reasoning)."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=384, help="image width in px (default 384)")
    ap.add_argument("--height", type=int, default=288, help="image height in px (default 288)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default ../data/sample)")
    args = ap.parse_args()
    if args.width < 16 or args.height < 16:
        ap.error("--width/--height must be >= 16 (census window + max disparity need margin)")

    left, right, gt_disp, gt_valid = gen_scene(args.width, args.height)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_pgm(args.out_dir / "left.pgm", args.width, args.height, left)
    write_pgm(args.out_dir / "right.pgm", args.width, args.height, right)
    write_pgm(args.out_dir / "gt_disparity.pgm", args.width, args.height, gt_disp)
    write_pgm(args.out_dir / "gt_valid.pgm", args.width, args.height, gt_valid)

    n_valid = sum(1 for b in gt_valid if b)
    total = args.width * args.height
    total_bytes = sum((args.out_dir / n).stat().st_size
                      for n in ("left.pgm", "right.pgm", "gt_disparity.pgm", "gt_valid.pgm"))
    print(f"wrote {args.out_dir} : {args.width}x{args.height} stereo pair "
          f"({total_bytes} bytes total across 4 PGMs) - labeled SYNTHETIC")
    print(f"note: {n_valid}/{total} pixels ({100.0*n_valid/total:.1f}%) are GT-valid "
          f"(unoccluded, outside the {CENSUS_HALF}px census border margin)")
    if args.width != 384 or args.height != 288:
        print("note: non-default size - fine for experiments, do NOT commit these files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
