#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.05
(SIFT on GPU: Gaussian scale space, DoG extrema, warp-level orientation/
descriptor histograms, brute-force L2 matching).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project's whole selling point over its sibling 01.04 (feature-pipeline,
FAST/ORB — single-scale) is SCALE INVARIANCE, so the synthetic pair needs a
SECOND view under a KNOWN SIMILARITY TRANSFORM that includes a REAL zoom, not
just rotation+translation. As in 01.04, image B is rendered by evaluating the
scene function at the INVERSE-transformed coordinate of every output pixel
(never by warping image A's raster, which would blur/interpolate and muddy
the ground truth) — so the transform between A and B is exact by
construction.

What this script writes (into ../data/sample/, all committed, all tiny)
-------------------------------------------------------------------------
  scene_a.pgm      — 256x256 grayscale, the reference view (identity pose).
  scene_b.pgm      — 256x256 grayscale, the SAME scene under a known
                      similarity transform (1.5x ZOOM + 20deg rotation +
                      translation) plus a brightness offset.
  neg_scene_c.pgm  — 256x256 grayscale, a DIFFERENT scene (different
                      layout, different seed) — the negative control.
  transform.csv    — the ground-truth transform parameters, human-readable
                      provenance. THE AUTHORITATIVE copy of these numbers is
                      hardcoded (cross-referenced) in ../src/main.cu's
                      kTransform* constants (the 01.04/01.01 precedent).

Scale-diverse scene content (the lesson 01.04's self-similarity trap taught
this project, applied AND extended)
-----------------------------------------------------------------------------
01.04's THEORY.md records a real, empirically-found bug: a strict two-tone
alternating checkerboard is locally IDENTICAL at every interior corner
(rotate 90 degrees and any corner looks like any other), which is exactly
the property camera-calibration checkerboards rely on but generic feature
MATCHING is defeated by. The fix that project shipped — a small palette of
5 well-separated grayscale levels, pseudo-randomly (not alternately)
assigned per CELL via a seeded hash — is reused here VERBATIM (see
cell_color() below, credited). This project ADDS a second, SIFT-specific
requirement on top: checkerboard patches and disks at SEVERAL DIFFERENT
PHYSICAL SIZES across the canvas, so the Gaussian/DoG pyramid's different
octaves and intervals each have something in their own characteristic
scale band to detect — a scene built entirely from one feature size would
under-exercise the very multi-scale machinery this project teaches.

Determinism (CLAUDE.md paragraph 12 / project brief): every random choice
below is drawn from a hand-rolled xorshift32 generator — the SAME 4-line
algorithm ../src/kernels.cuh implements in C++ (see build_orb_base_pattern's
precedent in 01.04, xorshift32_next there). Implementation-defined library
RNGs are avoided so the same seed produces the same bytes in Python, cl.exe,
and nvcc's host pass alike.

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
# xorshift32 — Marsaglia's 32-bit xorshift PRNG, one step. Byte-for-byte the
# same recurrence as kernels.cuh's xorshift32-style helpers (see 01.04's
# precedent and its comment for why a hand-rolled generator is used at all).
# ===========================================================================
class XorShift32:
    def __init__(self, seed: int):
        assert seed != 0, "xorshift32 has one dead state: seed must be non-zero"
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        x = self.state
        x = (x ^ (x << 13)) & 0xFFFFFFFF
        x = (x ^ (x >> 17)) & 0xFFFFFFFF
        x = (x ^ (x << 5)) & 0xFFFFFFFF
        self.state = x
        return x

    def uniform(self, lo: float, hi: float) -> float:
        """Deterministic float in [lo, hi), derived from one xorshift32 draw."""
        u = self.next_u32() / 4294967296.0   # 2^32 -> [0, 1)
        return lo + u * (hi - lo)


# ===========================================================================
# Image geometry — MUST MATCH kBaseW/kBaseH in ../src/kernels.cuh.
# ===========================================================================
W = 256
H = 256

# ===========================================================================
# Ground-truth SIMILARITY transform, scene A -> scene B. MUST MATCH the
# kTransform* constants in ../src/main.cu (cross-referenced there too).
#
#   forward(xa, ya) = scale * R(theta) * (xa - cx, ya - cy) + (cx, cy) + (tx, ty)
#
# The 1.5x SCALE factor is this project's entire reason for existing over
# 01.04's rotation+translation-only transform: it is the one thing a
# single-scale detector (FAST, or SIFT with only one octave/interval)
# cannot recover, and the one thing SIFT's whole multi-octave, multi-
# interval, sub-scale-refined pipeline is built to survive.
# ===========================================================================
TRANSFORM_THETA_DEG = 20.0
TRANSFORM_SCALE = 1.5
TRANSFORM_TX_PX = 10.0
TRANSFORM_TY_PX = -8.0
TRANSFORM_BRIGHTNESS_OFFSET = 15   # added to every scene-B (and neg-scene-C) pixel, then clipped [0,255]
CENTER_X = (W - 1) / 2.0
CENTER_Y = (H - 1) / 2.0


def forward_transform(xa: float, ya: float):
    """Map a scene-A coordinate to its scene-B coordinate under the ground-
    truth transform (see module header). Retyped independently in
    ../src/main.cu's forward_transform() — TWO independent implementations
    of one formula agreeing is an informal cross-check this project relies
    on the same way 01.04's precedent does."""
    theta = math.radians(TRANSFORM_THETA_DEG)
    c, s = math.cos(theta), math.sin(theta)
    ux, uy = xa - CENTER_X, ya - CENTER_Y
    rx = c * ux - s * uy
    ry = s * ux + c * uy
    return TRANSFORM_SCALE * rx + CENTER_X + TRANSFORM_TX_PX, TRANSFORM_SCALE * ry + CENTER_Y + TRANSFORM_TY_PX


def inverse_transform(xb: float, yb: float):
    """The exact inverse of forward_transform — used to RENDER scene B: for
    each output pixel in B, find the scene-A coordinate that maps there,
    and sample the analytic scene function there (see module header)."""
    theta = math.radians(TRANSFORM_THETA_DEG)
    c, s = math.cos(theta), math.sin(theta)
    ux = (xb - TRANSFORM_TX_PX - CENTER_X) / TRANSFORM_SCALE
    uy = (yb - TRANSFORM_TY_PX - CENTER_Y) / TRANSFORM_SCALE
    xa = c * ux + s * uy + CENTER_X       # R(-theta) = R(theta)^T
    ya = -s * ux + c * uy + CENTER_Y
    return xa, ya


# ===========================================================================
# Scene construction: a background gradient + filled disks + rotated,
# hashed-palette checkerboard patches, all defined as ANALYTIC functions of
# continuous (x, y) so they evaluate cleanly at the fractional coordinates
# inverse_transform() produces for scene B (see module header).
# ===========================================================================

def build_scene_params(seed: int, anchors_checker, anchors_disk, bg_base, bg_kx, bg_ky):
    """Generate one scene's shape parameters from a seeded xorshift32 stream,
    jittering hand-placed anchor points (guarantees coverage + border
    clearance while the RNG still supplies generated, not hand-tuned,
    jitter — the same design 01.04 established)."""
    rng = XorShift32(seed)

    disks = []
    for (cx0, cy0, r0, inten0) in anchors_disk:
        cx = cx0 + rng.uniform(-6.0, 6.0)
        cy = cy0 + rng.uniform(-6.0, 6.0)
        r = r0 + rng.uniform(-2.0, 2.0)
        inten = min(245.0, max(10.0, inten0 + rng.uniform(-15.0, 15.0)))
        disks.append({"cx": cx, "cy": cy, "r": r, "inten": inten})

    checkers = []
    for patch_id, (cx0, cy0, ang0, sq0, half0) in enumerate(anchors_checker):
        cx = cx0 + rng.uniform(-6.0, 6.0)
        cy = cy0 + rng.uniform(-6.0, 6.0)
        angle_deg = ang0 + rng.uniform(-6.0, 6.0)
        square = sq0 + rng.uniform(-0.8, 0.8)
        half_extent = half0 + rng.uniform(-2.0, 2.0)
        # patch_salt folds this scene's seed into the per-cell hash (see
        # cell_color() below) so scene A/B (seed 42) and scene C (seed 999)
        # get independently hashed cell colors.
        checkers.append({"cx": cx, "cy": cy, "angle_rad": math.radians(angle_deg),
                          "square": square, "half": half_extent,
                          "patch_salt": (seed * 1000003 + patch_id * 97) & 0xFFFFFFFF})

    return {"bg": (bg_base, bg_kx, bg_ky), "disks": disks, "checkers": checkers}


# cell_color — a SECOND, SIFT-specific self-similarity lesson, layered on
# top of 01.04's original one. 01.04's THEORY.md records the base finding:
# a strict two-tone alternating checkerboard is locally IDENTICAL at every
# interior corner (defeats generic feature MATCHING), fixed there with a
# small discrete 5-level palette hashed per CELL. That fix is necessary but
# turned out NOT SUFFICIENT for SIFT specifically (root-caused empirically
# while building this project: an earlier 5-level-palette version of this
# scene measured only 2/75 query keypoints clearing the Lowe ratio test,
# even though 30/75 had a stable MUTUAL nearest neighbor — i.e., detection
# and orientation were fine, but descriptors were not DISCRIMINATIVE
# enough). The reason is specific to SIFT's descriptor: it is a
# L2-NORMALIZED gradient-orientation HISTOGRAM, which discards absolute
# contrast/magnitude by construction, leaving only the SHAPE of the local
# edge-junction structure. A checkerboard corner's shape is dominated by
# "two roughly-perpendicular step edges" regardless of which two of a
# small palette's values happen to be on each side — so even with 5^4=625
# combinatorially distinct 2x2 corner colorings, MANY of them produce
# statistically SIMILAR normalized histograms once rotation is cancelled
# out. ORB/Hamming (01.04) is far less sensitive to this because its
# rBRIEF bits are raw greater/less comparisons at many individual sample
# PAIRS across the patch, not a smoothed, normalized magnitude histogram.
# The fix: draw each cell's intensity from a CONTINUOUS, effectively
# unique-per-cell range (not a small discrete set) — every corner's local
# intensity RATIOS become distinct, breaking the shape-level repetition
# while still avoiding the original two-tone-alternation trap (colors are
# still independent per cell, never alternating by parity).
def cell_color(patch_salt: int, col: int, row: int) -> float:
    """Deterministically hash one checkerboard CELL to a CONTINUOUS
    intensity in [20, 235] via a freshly-seeded xorshift32 stream (see the
    comment above for why continuous, not a small discrete palette, is
    what SIFT's normalized-histogram descriptor needs)."""
    seed = (patch_salt + (col * 92821) + (row * 68917)) & 0xFFFFFFFF
    if seed == 0:
        seed = 1   # xorshift32's one dead state
    return XorShift32(seed).uniform(20.0, 235.0)


# A THIRD idea, tried and honestly rejected during development (see
# THEORY.md "How we verify correctness" for the measured before/after):
# adding fine per-pixel micro-texture noise (a synthetic stand-in for
# sensor grain) to break the remaining "generic right-angle corner"
# descriptor self-similarity. Measured effect: WORSE, not better — the
# noise is resampled at different sub-pixel phase between scene A's direct
# sampling and scene B's inverse-transformed sampling, so it perturbs a
# TRUE match's two descriptors independently (hurting real correspondences)
# while barely denting the coincidental similarity between unrelated
# corners (which is a SHAPE-level, not fine-texture-level, resemblance).
# Left undone here, honestly, rather than shipped as a fix that measured
# worse than not fixing it at all.


def render_scene(params, x: float, y: float) -> float:
    """Evaluate the analytic scene at continuous (x, y): background, then
    disks, then checkerboards on top (painter's algorithm — checkerboards
    are the primary feature-detector target, so they are never occluded),
    then micro-texture noise (see the comment above) added everywhere.
    Returns intensity in [0, 255] BEFORE any brightness offset."""
    base, kx, ky = params["bg"]
    color = base + kx * x + ky * y

    for d in params["disks"]:
        dx, dy = x - d["cx"], y - d["cy"]
        if dx * dx + dy * dy <= d["r"] * d["r"]:
            color = d["inten"]

    for cb in params["checkers"]:
        dx, dy = x - cb["cx"], y - cb["cy"]
        c, s = math.cos(cb["angle_rad"]), math.sin(cb["angle_rad"])
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        if abs(lx) <= cb["half"] and abs(ly) <= cb["half"]:
            col = math.floor(lx / cb["square"])
            row = math.floor(ly / cb["square"])
            color = cell_color(cb["patch_salt"], col, row)

    return color


SUPERSAMPLE = 4   # NxN sub-samples averaged per output pixel — the synthetic
# stand-in for a camera's point-spread function (THEORY.md "The problem").
# Without this anti-aliasing, DoG extrema locations near a shape boundary
# are hypersensitive to which side of a hard step an integer pixel falls
# on — a sub-pixel viewpoint shift (exactly what a 1.5x zoom + 20deg
# rotation causes) can shift or destroy a keypoint. 01.04's THEORY.md
# records the same lesson for FAST/Harris; it applies here unchanged.
_SS_OFFSETS = [(-0.5 + (i + 0.5) / SUPERSAMPLE) for i in range(SUPERSAMPLE)]


def render_image(params, transform_view: bool, brightness_offset: float) -> bytes:
    """Rasterize one WxH grayscale image, anti-aliased by averaging
    SUPERSAMPLE x SUPERSAMPLE sub-pixel samples per output pixel (see
    module-level SUPERSAMPLE comment). transform_view=False -> scene A
    (identity). transform_view=True -> scene B (or the negative-control
    image): each sub-sample is mapped through inverse_transform()
    INDIVIDUALLY (the anti-aliasing footprint is the OUTPUT pixel's
    footprint, projected back into the scene — the physically correct
    order), then brightness_offset is added and the averaged result
    clipped. Returns W*H raw bytes, row-major (PGM P5 payload)."""
    out = bytearray(W * H)
    n_samples = SUPERSAMPLE * SUPERSAMPLE
    for py in range(H):
        for px in range(W):
            total = 0.0
            for oy in _SS_OFFSETS:
                for ox in _SS_OFFSETS:
                    sx, sy = px + ox, py + oy
                    if transform_view:
                        xa, ya = inverse_transform(sx, sy)
                    else:
                        xa, ya = sx, sy
                    total += render_scene(params, xa, ya)
            v = (total / n_samples) + brightness_offset
            v = 0.0 if v < 0.0 else (255.0 if v > 255.0 else v)
            out[py * W + px] = int(v + 0.5)
    return bytes(out)


def write_pgm(path: Path, w: int, h: int, data: bytes) -> None:
    """Write a binary PGM (P5) file — the exact format ../src/main.cu's
    read_pgm() reads (see that function's header for the format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(data)


# ===========================================================================
# Scene A/B shape layout — SCALE-DIVERSE on purpose (see module header):
# large checkerboard patches for the coarse octave/interval levels, plus
# SMALL checkerboard patches and disks sized for the fine levels, spread
# across the canvas with at least kExtremaBorder-plus-refinement clearance
# from every edge.
# ===========================================================================
def build_scene_a_params():
    anchors_checker = [
        # (cx, cy, angle_deg, square_px, half_extent_px) -- LARGE patches:
        # sigma_img for octave-1 keypoints ranges roughly [3.8, 12.8] px
        # (see kernels.cuh's sigma_at()), so a feature that should attract
        # a COARSE-scale keypoint needs cell/patch structure on that order.
        (66, 64, 8.0, 26.0, 56.0),
        (190, 62, 33.0, 30.0, 54.0),
        (128, 190, -20.0, 28.0, 58.0),
        # SMALL patches: sigma_img for octave-0 keypoints ranges roughly
        # [1.9, 6.4] px, so these need finer cell structure.
        (40, 200, 15.0, 8.0, 20.0),
        (216, 196, -40.0, 7.0, 18.0),
        (128, 46, 50.0, 9.0, 22.0),
    ]
    anchors_disk = [
        # (cx, cy, radius_px, intensity_0_255) -- a spread of disk sizes,
        # same coarse/fine-scale reasoning as the checkerboard patches.
        (128, 128, 22.0, 60.0),     # large central disk -- coarse scale
        (46, 128, 7.0, 205.0),      # small disk -- fine scale
        (208, 132, 9.0, 90.0),
        (150, 216, 6.0, 190.0),
    ]
    # Background gradient: smooth (zero second derivative away from shape
    # edges), so DoG correctly finds almost no extrema in open background.
    return build_scene_params(seed=42, anchors_checker=anchors_checker, anchors_disk=anchors_disk,
                              bg_base=75.0, bg_kx=55.0 / (W - 1), bg_ky=25.0 / (H - 1))


def build_scene_c_params():
    """A DIFFERENT scene (different layout, different seed=999) for the
    negative-control gate — see main.cu's negative_control gate."""
    anchors_checker = [
        (188, 68, 42.0, 27.0, 52.0),
        (64, 190, -25.0, 29.0, 56.0),
        (128, 40, 62.0, 26.0, 50.0),
        (208, 210, 12.0, 8.0, 19.0),
        (48, 46, -35.0, 9.0, 21.0),
        (128, 210, 70.0, 7.0, 17.0),
    ]
    anchors_disk = [
        (200, 200, 20.0, 210.0),
        (56, 128, 8.0, 65.0),
        (150, 60, 7.0, 150.0),
        (196, 40, 21.0, 45.0),
    ]
    return build_scene_params(seed=999, anchors_checker=anchors_checker, anchors_disk=anchors_disk,
                              bg_base=150.0, bg_kx=-45.0 / (W - 1), bg_ky=-35.0 / (H - 1))


def write_transform_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground-truth transform, scene_a.pgm -> scene_b.pgm\n")
        f.write("# generated by scripts/make_synthetic.py -- AUTHORITATIVE copy is\n")
        f.write("# hardcoded (cross-referenced) in ../src/main.cu's kTransform* constants\n")
        f.write("# forward(xa,ya) = scale * R(theta_deg) * (xa-cx, ya-cy) + (cx,cy) + (tx_px, ty_px)\n")
        writer = csv.writer(f)
        writer.writerow(["field", "value", "units"])
        writer.writerow(["theta_deg", f"{TRANSFORM_THETA_DEG:.6f}", "degrees, counter-clockwise"])
        writer.writerow(["scale", f"{TRANSFORM_SCALE:.6f}", "unitless (1.5 = 1.5x zoom)"])
        writer.writerow(["tx_px", f"{TRANSFORM_TX_PX:.6f}", "pixels"])
        writer.writerow(["ty_px", f"{TRANSFORM_TY_PX:.6f}", "pixels"])
        writer.writerow(["brightness_offset", f"{TRANSFORM_BRIGHTNESS_OFFSET:.6f}", "intensity units, 0..255 scale"])
        writer.writerow(["center_x", f"{CENTER_X:.6f}", "pixels"])
        writer.writerow(["center_y", f"{CENTER_Y:.6f}", "pixels"])
        writer.writerow(["width", f"{W}", "pixels"])
        writer.writerow(["height", f"{H}", "pixels"])
    print(f"[make_synthetic] wrote {path} (ground-truth transform, SYNTHETIC)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic scene_a/scene_b/neg_scene_c PGM pair (+ ground-truth "
                    "transform CSV) for project 01.05 (SIFT on GPU).")
    parser.add_argument("--out-dir", type=Path, default=default_out,
                        help="output directory (default: ../data/sample)")
    args = parser.parse_args()

    params_a = build_scene_a_params()
    params_c = build_scene_c_params()

    scene_a = render_image(params_a, transform_view=False, brightness_offset=0.0)
    scene_b = render_image(params_a, transform_view=True, brightness_offset=TRANSFORM_BRIGHTNESS_OFFSET)
    scene_c = render_image(params_c, transform_view=False, brightness_offset=TRANSFORM_BRIGHTNESS_OFFSET)

    write_pgm(args.out_dir / "scene_a.pgm", W, H, scene_a)
    print(f"[make_synthetic] wrote {args.out_dir / 'scene_a.pgm'} ({W}x{H}, SYNTHETIC, seed=42)")
    write_pgm(args.out_dir / "scene_b.pgm", W, H, scene_b)
    print(f"[make_synthetic] wrote {args.out_dir / 'scene_b.pgm'} ({W}x{H}, SYNTHETIC, seed=42, "
          f"transformed scale={TRANSFORM_SCALE} theta={TRANSFORM_THETA_DEG}deg tx={TRANSFORM_TX_PX}px "
          f"ty={TRANSFORM_TY_PX}px brightness+={TRANSFORM_BRIGHTNESS_OFFSET})")
    write_pgm(args.out_dir / "neg_scene_c.pgm", W, H, scene_c)
    print(f"[make_synthetic] wrote {args.out_dir / 'neg_scene_c.pgm'} ({W}x{H}, SYNTHETIC, seed=999, "
          f"UNRELATED scene -- negative control)")

    write_transform_csv(args.out_dir / "transform.csv")


if __name__ == "__main__":
    main()
