#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.03
(Optical flow: dense pyramidal Lucas-Kanade + census-transform block-
matching flow).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
Optical flow needs a scene with FULL, PIXEL-DENSE ground truth — every one
of kW*kH pixels needs a known true displacement, not just a handful of
corners (project 01.04's feature-matching ground truth only needed to be
correct AT detected keypoints). An analytic scene function evaluated at an
INVERSE-transformed coordinate (the same technique 01.04's module header
explains at length) gives that for free and exactly: frame B is rendered by
evaluating the SAME scene function at the inverse-transformed location of
every output pixel, so the true displacement field is the FORWARD transform
evaluated in closed form — never approximated, never blurred by warping a
raster.

Why "hashed multi-scale texture", not a checkerboard (a lesson this project
inherits directly from 01.04's THEORY.md and this repo's own history)
------------------------------------------------------------------------
01.04's THEORY.md "The problem" names the reason checkerboards are a trap
for CORRESPONDENCE algorithms: a strict two-tone alternating checkerboard
is locally IDENTICAL at every interior corner (rotate 90 degrees and any
corner looks like any other) — exactly the SELF-SIMILARITY that breaks
matching, whether sparse (01.04's fix: per-cell hashed, non-alternating
colors) or DENSE, which is this project's much stricter requirement: dense
optical flow needs a strong, non-repeating, roughly UNIFORM gradient signal
at nearly every pixel (not just at cell corners), or the aperture problem
(THEORY.md) leaves huge flat regions with no recoverable motion at all, and
periodic texture creates ALIASED, ambiguous correspondences (a periodic
pattern shifted by one period looks identical to zero motion — the textbook
motion-aliasing failure mode). The fix used here: a smooth, NON-PERIODIC,
multi-OCTAVE value-noise field built by hashing a coarse lattice at several
scales (32, 16, 8, 4 px cells) and blending with smoothstep interpolation
— continuous and differentiable (so it can be evaluated at ANY fractional
scene coordinate for exact-ground-truth rendering, exactly like 01.04's
checkerboard function), never repeating, and textured at every scale a 5x5
LK/census window could examine (the finest 4 px cell size is deliberately
SMALLER than the 5x5 window itself — see OCTAVES' comment below for why
that specific relationship matters for the census milestone).

What this script writes (into ../data/sample/, all committed, all tiny)
-------------------------------------------------------------------------
  scene_a.pgm                   — 160x120 grayscale reference frame ("frame
                                   A"), shared by all four pairs below.
  scene_b_translation.pgm       — frame A translated by a KNOWN constant
                                   (tx_px, ty_px) — pair (a): flow is the
                                   SAME (tx_px, ty_px) at every pixel.
  scene_b_rotzoom.pgm           — frame A rotated theta_deg and scaled
                                   zoom_scale about the image center — pair
                                   (b): flow varies smoothly across the
                                   frame (an affine field).
  scene_b_translation_bright.pgm — EXACTLY scene_b_translation.pgm plus a
                                   smooth horizontal brightness ramp (up to
                                   +brightness_grad_max, ~20% of full
                                   scale) — pair (c): the SAME ground-truth
                                   flow as pair (a), isolating brightness
                                   robustness as the only new variable.
  scene_b_zero.pgm              — byte-identical to scene_a.pgm — pair (d):
                                   the zero-motion negative control.
  ground_truth.csv              — every transform parameter above, human-
                                   readable provenance. THE AUTHORITATIVE
                                   copy is hardcoded (cross-referenced) in
                                   ../src/main.cu's k* constants, following
                                   01.01/01.04's single-sourcing precedent
                                   (CLAUDE.md's "small human-checked
                                   constants" spirit, not a machine-parsed
                                   file).

Determinism (CLAUDE.md paragraph 12 / project brief): every random choice
below is drawn from a hand-rolled xorshift32 generator — the SAME 4-line
algorithm ../src/kernels.cuh implements in C++ (popcount32_portable's
sibling constants aside, this script's xorshift32_once() is the identical
recurrence used throughout this repo's Python generators, e.g. 01.04's
XorShift32) — never Python's random module (Mersenne Twister is spec'd but
this repo standardizes on one hand-rolled generator everywhere for the
cross-language teaching parallel — see 01.04's module header).

Usage
-----
    python make_synthetic.py                     # writes the committed default sample
    python make_synthetic.py --out-dir /tmp/x     # regenerate elsewhere for inspection
"""

import argparse
import csv
import math
from pathlib import Path

# ===========================================================================
# Image geometry — MUST MATCH kW/kH in ../src/kernels.cuh.
# ===========================================================================
W = 160
H = 120

# ===========================================================================
# Ground-truth transforms. MUST MATCH the k* constants in ../src/main.cu
# (cross-referenced there too) — every gate in main.cu checks its measured
# flow against exactly these numbers.
# ===========================================================================
TRANSLATE_TX_PX = 3.0
TRANSLATE_TY_PX = -3.0

ROT_THETA_DEG = 6.0
ROT_ZOOM_SCALE = 1.05                 # > 1 = the scene "zooms in" from A to B
CENTER_X = (W - 1) / 2.0
CENTER_Y = (H - 1) / 2.0

BRIGHTNESS_GRAD_MAX = 51.0            # 0.20 * 255 — a HORIZONTAL ramp, 0 at the left edge to this at the right edge


# ---------------------------------------------------------------------------
# xorshift32_once — one xorshift32 step from a given (non-zero) seed. Byte-
# for-byte the SAME recurrence as ../src/kernels.cuh's popcount-adjacent
# generators / 01.04's XorShift32.next_u32() (see this module's header for
# why a hand-rolled generator is used everywhere in this repo instead of a
# language-standard-library RNG).
# ---------------------------------------------------------------------------
def xorshift32_once(seed: int) -> int:
    x = seed & 0xFFFFFFFF
    if x == 0:
        x = 1                          # xorshift32's one dead state — never let a caller pass 0 through
    x = (x ^ (x << 13)) & 0xFFFFFFFF
    x = (x ^ (x >> 17)) & 0xFFFFFFFF
    x = (x ^ (x << 5)) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def hash01(ix: int, iy: int, seed: int) -> float:
    """Deterministically hash one integer lattice point (ix, iy) at a given
    octave `seed` to a float in [0, 1) — one xorshift32 draw from a freshly
    combined seed, the SAME pattern 01.04's cell_color() uses (combine the
    coordinates and a salt into one seed, take ONE draw from a fresh
    stream) rather than a running/shared generator: this makes hash01 a
    PURE function of (ix, iy, seed) — evaluate it a million times in any
    order, from any octave, and the answer for a given lattice point never
    changes, which is exactly the property a re-evaluatable analytic scene
    function needs (module header)."""
    combined = (seed * 1000003 + ix * 92821 + iy * 68917) & 0xFFFFFFFF
    return xorshift32_once(combined) / 4294967296.0   # 2^32 -> [0, 1)


def smoothstep(t: float) -> float:
    """The classic Perlin smoothstep 3t^2-2t^3: zero first derivative at
    t=0 and t=1, so adjacent lattice CELLS meet with continuous slope (no
    visible seams) — this is what makes the noise field differentiable
    almost everywhere, which matters because LK's gradient-based
    linearization (THEORY.md "The math") implicitly assumes the image is
    locally smooth."""
    return t * t * (3.0 - 2.0 * t)


def value_noise(x: float, y: float, cell_size: float, seed: int) -> float:
    """One OCTAVE of 2-D value noise: hash the 4 lattice points surrounding
    continuous coordinate (x, y) at the given cell_size, then bilinearly
    blend them with smoothstep-eased weights. Returns a value in [0, 1)."""
    gx = x / cell_size
    gy = y / cell_size
    ix0 = math.floor(gx)
    iy0 = math.floor(gy)
    fx = gx - ix0
    fy = gy - iy0

    h00 = hash01(ix0, iy0, seed)
    h10 = hash01(ix0 + 1, iy0, seed)
    h01 = hash01(ix0, iy0 + 1, seed)
    h11 = hash01(ix0 + 1, iy0 + 1, seed)

    sx = smoothstep(fx)
    sy = smoothstep(fy)
    top = h00 + (h10 - h00) * sx
    bot = h01 + (h11 - h01) * sx
    return top + (bot - top) * sy


# OCTAVES — (cell_size_px, weight, seed_offset). Weights sum to 1.0 so the
# blended result stays in [0,1). Four scales (32, 16, 8, 4 px cells) so
# every one of the LK pyramid's three levels (full/half/quarter res) and
# the 5x5 LK/census windows see genuine multi-SCALE structure, not just
# one dominant frequency — the "multi-scale" half of this project's
# "hashed multi-scale texture" brief.
#
# Weight balance (root-caused empirically while building this project —
# see THEORY.md "How we verify correctness" for the before/after numbers):
# an EARLIER version of this scene weighted the COARSE octaves much more
# heavily (45%/28%/17%/10%), which is exactly what dense LK's smooth-
# gradient assumption wants, but starved the finest scale (a 5x5 census
# window sitting inside a near-linear coarse gradient sees almost the SAME
# rank-order pattern at every nearby candidate shift, so the block matcher
# had no genuine local information to disambiguate a precise integer
# displacement — many candidates tied at the minimum Hamming cost, and the
# winner-take-all tie-break landed essentially at random within the search
# window). The FIX is not "abandon multi-scale" but REBALANCE it: the
# finest cell size (4 px) is now smaller than the 5x5 census/LK window
# itself, guaranteeing real local contrast inside every window, while the
# 8 px octave carries the most weight as the "workhorse" scale genuinely
# resolvable by both a 5x5 window AND a 13x13 (2*6+1) search neighborhood.
OCTAVES = [
    (32.0, 0.20, 0),
    (16.0, 0.25, 1_000_003),
    (8.0, 0.30, 2_000_003),
    (4.0, 0.25, 3_000_003),
]


def scene_value(x: float, y: float) -> float:
    """Evaluate the analytic scene at continuous (x, y): sum the four
    octaves above, map [0,1) -> an intensity range comfortably inside
    [0,255] (leaving headroom so this project's later brightness-gradient
    scene (c) cannot clip at the bright end — see BRIGHTNESS_GRAD_MAX)."""
    total = 0.0
    for cell_size, weight, seed_offset in OCTAVES:
        total += weight * value_noise(x, y, cell_size, 42 + seed_offset)
    return 25.0 + 200.0 * total   # range approximately [25, 225]


def forward_translate(xa: float, ya: float):
    """Scene coordinate (xa,ya) -> its location in scene_b_translation.pgm.
    Retyped independently (double, plain C++) in ../src/main.cu's
    forward_translate() — the SAME "gate independence" principle 01.04's
    module header documents (cited there in full)."""
    return xa + TRANSLATE_TX_PX, ya + TRANSLATE_TY_PX


def inverse_translate(xb: float, yb: float):
    """Exact inverse of forward_translate — used to RENDER scene_b_translation
    (module header: evaluate the scene at the inverse-mapped coordinate, not
    by warping a raster, so the ground truth is exact, not approximate)."""
    return xb - TRANSLATE_TX_PX, yb - TRANSLATE_TY_PX


def forward_rotzoom(xa: float, ya: float):
    """Scene coordinate (xa,ya) -> its location in scene_b_rotzoom.pgm:
    rotate by ROT_THETA_DEG and scale by ROT_ZOOM_SCALE about the image
    center. Retyped independently in ../src/main.cu (see forward_translate's
    comment for why that independent retyping matters)."""
    theta = math.radians(ROT_THETA_DEG)
    c, s = math.cos(theta), math.sin(theta)
    ux, uy = xa - CENTER_X, ya - CENTER_Y
    rx = ROT_ZOOM_SCALE * (c * ux - s * uy)
    ry = ROT_ZOOM_SCALE * (s * ux + c * uy)
    return rx + CENTER_X, ry + CENTER_Y


def inverse_rotzoom(xb: float, yb: float):
    """Exact inverse of forward_rotzoom (R(theta)^-1 = R(-theta), scale^-1 =
    1/scale) — used to RENDER scene_b_rotzoom.pgm."""
    ux = (xb - CENTER_X) / ROT_ZOOM_SCALE
    uy = (yb - CENTER_Y) / ROT_ZOOM_SCALE
    theta = math.radians(-ROT_THETA_DEG)
    c, s = math.cos(theta), math.sin(theta)
    xa = c * ux - s * uy + CENTER_X
    ya = s * ux + c * uy + CENTER_Y
    return xa, ya


SUPERSAMPLE = 3   # NxN sub-samples averaged per output pixel (anti-aliasing,
                  # same technique/rationale as 01.04's render_image() — the
                  # finest noise octave (5px cells) is close enough to the
                  # pixel grid's own Nyquist limit that single-point sampling
                  # would alias; 3x3=9 sub-samples is enough to tame it while
                  # keeping generation time reasonable at 160x120x3 renders).
_SS_OFFSETS = [(-0.5 + (i + 0.5) / SUPERSAMPLE) for i in range(SUPERSAMPLE)]


def render_frame(inverse_transform) -> list:
    """Rasterize one WxH grayscale frame, anti-aliased by averaging
    SUPERSAMPLE x SUPERSAMPLE sub-pixel samples per output pixel (a box
    filter over the pixel's footprint — see 01.04's render_image() header
    for the full "camera sensor integrates over a pixel's area" argument
    this project cites rather than re-deriving).

    inverse_transform(px, py) -> (xa, ya): maps an OUTPUT pixel's continuous
    coordinate back to the SCENE coordinate to sample (identity for
    scene_a.pgm — pass a lambda that returns its input unchanged).

    Returns a flat list of W*H floats in [0,255] (NOT yet quantized to
    bytes — callers add any post-hoc brightness term before quantizing, so
    quantization happens exactly ONCE per pixel, matching 01.04's "round
    once" numerics discipline)."""
    out = [0.0] * (W * H)
    n_samples = SUPERSAMPLE * SUPERSAMPLE
    for py in range(H):
        for px in range(W):
            total = 0.0
            for oy in _SS_OFFSETS:
                for ox in _SS_OFFSETS:
                    xa, ya = inverse_transform(px + ox, py + oy)
                    total += scene_value(xa, ya)
            out[py * W + px] = total / n_samples
    return out


def quantize(values: list) -> bytes:
    """Clip to [0,255] and round to the nearest byte — the ONE rounding
    step every rendered frame goes through (module header)."""
    out = bytearray(len(values))
    for i, v in enumerate(values):
        c = 0.0 if v < 0.0 else (255.0 if v > 255.0 else v)
        out[i] = int(c + 0.5)
    return bytes(out)


def write_pgm(path: Path, w: int, h: int, data: bytes) -> None:
    """Write a binary PGM (P5) file — the exact format ../src/main.cu's
    read_pgm() reads (see that function's header for the format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(data)


def write_ground_truth_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground-truth transform parameters for project 01.03\n")
        f.write("# generated by scripts/make_synthetic.py -- AUTHORITATIVE copy is\n")
        f.write("# hardcoded (cross-referenced) in ../src/main.cu's k* constants\n")
        f.write("# pair (a) translation:      flow(x,y) = (tx_px, ty_px) EVERYWHERE\n")
        f.write("# pair (b) rotation+zoom:    forward(xa,ya) = center + scale*R(theta_deg)*(xa-center,ya-center); flow = forward(xa,ya) - (xa,ya)\n")
        f.write("# pair (c) translation+bright: SAME flow as pair (a); frame B additionally ramped +brightness_grad_max left->right\n")
        f.write("# pair (d) zero-motion:      flow = (0, 0) EVERYWHERE (scene_b_zero.pgm is byte-identical to scene_a.pgm)\n")
        writer = csv.writer(f)
        writer.writerow(["field", "value", "units"])
        writer.writerow(["width", f"{W}", "pixels"])
        writer.writerow(["height", f"{H}", "pixels"])
        writer.writerow(["translate_tx_px", f"{TRANSLATE_TX_PX:.6f}", "pixels"])
        writer.writerow(["translate_ty_px", f"{TRANSLATE_TY_PX:.6f}", "pixels"])
        writer.writerow(["rot_theta_deg", f"{ROT_THETA_DEG:.6f}", "degrees, counter-clockwise, about the image center"])
        writer.writerow(["rot_zoom_scale", f"{ROT_ZOOM_SCALE:.6f}", "unitless scale factor, about the image center"])
        writer.writerow(["center_x", f"{CENTER_X:.6f}", "pixels"])
        writer.writerow(["center_y", f"{CENTER_Y:.6f}", "pixels"])
        writer.writerow(["brightness_grad_max", f"{BRIGHTNESS_GRAD_MAX:.6f}", "intensity units, 0..255 scale, horizontal ramp added to scene_b_translation_bright.pgm"])
    print(f"[make_synthetic] wrote {path} (ground-truth transform parameters, SYNTHETIC)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic optical-flow sample pairs (+ ground-truth CSV) "
                    "for project 01.03 (Optical flow).")
    parser.add_argument("--out-dir", type=Path, default=default_out,
                        help="output directory (default: ../data/sample)")
    args = parser.parse_args()

    print("[make_synthetic] rendering scene_a.pgm (identity view) ...")
    frame_a = render_frame(lambda x, y: (x, y))
    write_pgm(args.out_dir / "scene_a.pgm", W, H, quantize(frame_a))
    print(f"[make_synthetic] wrote {args.out_dir / 'scene_a.pgm'} ({W}x{H}, SYNTHETIC, seed=42, hashed multi-scale texture)")

    print("[make_synthetic] rendering scene_b_translation.pgm (pure translation) ...")
    frame_translate = render_frame(inverse_translate)
    write_pgm(args.out_dir / "scene_b_translation.pgm", W, H, quantize(frame_translate))
    print(f"[make_synthetic] wrote {args.out_dir / 'scene_b_translation.pgm'} "
          f"(tx={TRANSLATE_TX_PX}px ty={TRANSLATE_TY_PX}px, flow constant everywhere)")

    print("[make_synthetic] rendering scene_b_rotzoom.pgm (rotation + zoom about center) ...")
    frame_rotzoom = render_frame(inverse_rotzoom)
    write_pgm(args.out_dir / "scene_b_rotzoom.pgm", W, H, quantize(frame_rotzoom))
    print(f"[make_synthetic] wrote {args.out_dir / 'scene_b_rotzoom.pgm'} "
          f"(theta={ROT_THETA_DEG}deg scale={ROT_ZOOM_SCALE}, flow varies spatially)")

    # scene_b_translation_bright.pgm — the ALREADY-RENDERED translation frame
    # (pre-quantization float values) plus a smooth horizontal brightness
    # ramp, quantized ONCE (module header's "one rounding step" discipline).
    # This is the SAME ground-truth flow as pair (a): only the brightness
    # changes, isolating that as the one new variable (main.cu's
    # brightness-robustness gate).
    print("[make_synthetic] deriving scene_b_translation_bright.pgm (+brightness ramp) ...")
    bright = [0.0] * (W * H)
    for py in range(H):
        for px in range(W):
            ramp = BRIGHTNESS_GRAD_MAX * (px / (W - 1))
            bright[py * W + px] = frame_translate[py * W + px] + ramp
    write_pgm(args.out_dir / "scene_b_translation_bright.pgm", W, H, quantize(bright))
    print(f"[make_synthetic] wrote {args.out_dir / 'scene_b_translation_bright.pgm'} "
          f"(same flow as scene_b_translation.pgm, +0..{BRIGHTNESS_GRAD_MAX:.0f} horizontal brightness ramp)")

    # scene_b_zero.pgm — byte-identical to scene_a.pgm: the zero-motion
    # negative control (main.cu loads both independently rather than
    # special-casing "reuse scene_a.pgm twice", so every pair's data files
    # are self-describing and symmetric).
    print("[make_synthetic] writing scene_b_zero.pgm (byte-identical to scene_a.pgm) ...")
    write_pgm(args.out_dir / "scene_b_zero.pgm", W, H, quantize(frame_a))
    print(f"[make_synthetic] wrote {args.out_dir / 'scene_b_zero.pgm'} (zero-motion negative control)")

    write_ground_truth_csv(args.out_dir / "ground_truth.csv")


if __name__ == "__main__":
    main()
