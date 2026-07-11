#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.04
(Feature pipeline: FAST/Harris detection, ORB descriptors, brute-force
Hamming matcher).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
Robotics data can almost always be synthesized with full ground truth; this
project needs THREE things a real photograph could never give us honestly:
(1) a scene rich enough in corner structure to exercise two different
detectors, (2) a SECOND view of the exact same scene under a PRECISELY
KNOWN rigid transform (no calibration/registration error to argue about),
and (3) a scene GUARANTEED unrelated to (1) for the negative-control gate.
An analytic, hand-specified scene function gives all three for free: image
B is rendered by evaluating the SAME scene function at the inverse-
transformed coordinate of every output pixel (not by warping image A's
raster, which would blur/interpolate and muddy the ground truth), so the
transform between A and B is exact by construction, not approximate.

What this script writes (into ../data/sample/, all committed, all tiny)
-------------------------------------------------------------------------
  scene_a.pgm      — 256x256 grayscale, the reference view (identity pose).
  scene_b.pgm      — 256x256 grayscale, the SAME scene under a known
                      similarity transform (rotation + translation) plus a
                      brightness offset (to honestly exercise the "FAST/
                      Harris/ORB claim brightness-offset invariance" story).
  neg_scene_c.pgm  — 256x256 grayscale, a DIFFERENT scene (different
                      layout, different seed) used only as the negative
                      control in main.cu's matching gate — proof the
                      matcher is not self-confirming.
  transform.csv    — the ground-truth transform parameters, human-
                      readable provenance. THE AUTHORITATIVE COPY of these
                      numbers is hardcoded (with a cross-referencing
                      comment) in ../src/main.cu, exactly the precedent
                      01.01's checkerboard-geometry constants set (CLAUDE.md
                      "single-sourced" spirit applied to human-checked
                      constants rather than a machine-parsed file) — this
                      CSV exists so a learner can see the numbers without
                      reading C++, and so a re-run of this script prints a
                      value a learner can diff against main.cu by eye.

Determinism (CLAUDE.md paragraph 12 / this project's brief): every random
choice below is drawn from a hand-rolled xorshift32 generator — the EXACT
same 4-line algorithm ../src/kernels.cuh implements in C++ for the ORB
sampling pattern (see xorshift32_next() there). Implementation-defined
library RNGs (Python's Mersenne Twister, C++'s <random> distributions) are
avoided so the SAME algorithm, in two languages, is visibly doing the
project's "seeded, reproducible randomness" job — a deliberate teaching
parallel, not merely a rule.

Usage
-----
    python make_synthetic.py                     # writes the committed default sample
    python make_synthetic.py --out-dir /tmp/x     # regenerate elsewhere for inspection
"""

import argparse
import csv
import math
import struct
from pathlib import Path

# ===========================================================================
# xorshift32 — Marsaglia's 32-bit xorshift PRNG, one step. Byte-for-byte the
# same recurrence as kernels.cuh's xorshift32_next() (C++) — see that
# function's comment for why a hand-rolled generator is used at all instead
# of the standard library's (implementation-defined) distributions.
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
# Image geometry — MUST MATCH kW/kH in ../src/kernels.cuh.
# ===========================================================================
W = 256
H = 256

# ===========================================================================
# Ground-truth similarity transform, scene A -> scene B. MUST MATCH the
# kTransform* constants in ../src/main.cu (cross-referenced there too) —
# this is the single most important shared number in the whole project:
# every gate in main.cu that claims "the matcher recovered the true motion"
# is checked against exactly this transform.
#
#   forward(xa, ya) = R(theta) * (xa - cx, ya - cy) + (cx, cy) + (tx, ty)
#
# applied about the image CENTER (cx, cy), then translated. Chosen to be
# large enough to meaningfully exercise rotation/translation invariance
# claims (12 degrees is well beyond anti-aliasing noise) but small enough
# that the two views still share the bulk of the same scene content (no
# transform-induced content leaving the frame at the CENTER, where most of
# the checkerboard texture lives).
# ===========================================================================
TRANSFORM_THETA_DEG = 12.0
TRANSFORM_TX_PX = 7.0
TRANSFORM_TY_PX = -5.0
TRANSFORM_BRIGHTNESS_OFFSET = 18   # added to every scene-B (and neg-scene-C) pixel, then clipped [0,255]
CENTER_X = (W - 1) / 2.0
CENTER_Y = (H - 1) / 2.0


def forward_transform(xa: float, ya: float):
    """Map a scene-A coordinate to its scene-B coordinate under the ground-
    truth transform (see module header). Retyped independently (double
    precision, plain Python floats) in main.cu's gate_ground_truth_transform()
    and gate_repeatability() — THREE independent implementations of this one
    formula (Python here, and two C++ call sites) all agreeing is itself a
    cross-check the project relies on informally while developing (any typo
    here would show up as a gate failure, not a silent wrong answer)."""
    theta = math.radians(TRANSFORM_THETA_DEG)
    c, s = math.cos(theta), math.sin(theta)
    ux, uy = xa - CENTER_X, ya - CENTER_Y
    rx = c * ux - s * uy
    ry = s * ux + c * uy
    return rx + CENTER_X + TRANSFORM_TX_PX, ry + CENTER_Y + TRANSFORM_TY_PX


def inverse_transform(xb: float, yb: float):
    """The exact inverse of forward_transform — used to RENDER scene B (for
    each output pixel in B, find the scene-A coordinate that maps there, and
    sample the analytic scene function there — see module header for why
    this, rather than warping A's raster, is the honest way to synthesize
    an exact-ground-truth second view)."""
    theta = math.radians(TRANSFORM_THETA_DEG)
    c, s = math.cos(theta), math.sin(theta)
    ux = xb - TRANSFORM_TX_PX - CENTER_X
    uy = yb - TRANSFORM_TY_PX - CENTER_Y
    xa = c * ux + s * uy + CENTER_X       # R(-theta) = R(theta)^T
    ya = -s * ux + c * uy + CENTER_Y
    return xa, ya


# ===========================================================================
# Scene construction: a background gradient + filled disks + rotated
# checkerboard patches, all defined as ANALYTIC functions of continuous
# (x, y) so they can be evaluated at the fractional coordinates
# inverse_transform() produces for scene B, with no interpolation/blur.
# ===========================================================================

def build_scene_params(seed: int, anchors_checker, anchors_disk, bg_base, bg_kx, bg_ky):
    """Generate one scene's shape parameters from a seeded xorshift32 stream,
    jittering a set of hand-placed anchor points (see call sites below for
    WHY hand-placed anchors: they guarantee good coverage and enough margin
    from the image border for kBorder=16-eligible keypoints, while the RNG
    still supplies the "generated, not hand-tuned pixel-by-pixel" jitter
    CLAUDE.md's synthetic-data convention asks for).

    Returns a dict: {'bg': (base, kx, ky), 'disks': [...], 'checkers': [...]}.
    """
    rng = XorShift32(seed)

    disks = []
    for (cx0, cy0, r0, inten0) in anchors_disk:
        cx = cx0 + rng.uniform(-6.0, 6.0)
        cy = cy0 + rng.uniform(-6.0, 6.0)
        r = r0 + rng.uniform(-3.0, 3.0)
        inten = min(245.0, max(10.0, inten0 + rng.uniform(-15.0, 15.0)))
        disks.append({"cx": cx, "cy": cy, "r": r, "inten": inten})

    checkers = []
    for patch_id, (cx0, cy0, ang0, sq0, half0) in enumerate(anchors_checker):
        cx = cx0 + rng.uniform(-8.0, 8.0)
        cy = cy0 + rng.uniform(-8.0, 8.0)
        angle_deg = ang0 + rng.uniform(-6.0, 6.0)
        square = sq0 + rng.uniform(-1.0, 1.0)
        half_extent = half0 + rng.uniform(-3.0, 3.0)
        # patch_salt: folds this scene's seed into the per-cell hash (see
        # cell_color() below) so scene A/B (seed 42) and scene C (seed 999)
        # get INDEPENDENTLY hashed cell colors even where an anchor position
        # coincidentally lines up — not load-bearing for correctness, just
        # keeps the two scenes' checker colorings from ever accidentally
        # correlating.
        checkers.append({"cx": cx, "cy": cy, "angle_rad": math.radians(angle_deg),
                          "square": square, "half": half_extent,
                          "patch_salt": (seed * 1000003 + patch_id * 97) & 0xFFFFFFFF})

    return {"bg": (bg_base, bg_kx, bg_ky), "disks": disks, "checkers": checkers}


# CELL_PALETTE — the discrete intensities a checkerboard CELL may be
# colored (see cell_color() below). Deliberately NOT just {light, dark}:
# a strict two-tone alternating checkerboard is locally IDENTICAL at every
# interior corner (rotate 90 degrees and any corner looks like any other —
# a well-known reason checkerboards are used for CAMERA CALIBRATION's
# specialized corner detectors, and a well-known TRAP for generic feature
# MATCHING, whose whole premise is that a local patch's appearance is
# distinctive). Five well-separated grayscale levels, pseudo-randomly
# assigned per cell (not alternating), give FAST/Harris the same strong
# corners at cell boundaries while giving ORB's descriptor something
# non-repetitive to actually discriminate between corners with — this
# was root-caused empirically (an earlier alternating-parity version of
# this scene measured only 8/15 = 53% ground-truth-transform inliers;
# see THEORY.md "How we verify correctness" for the before/after numbers).
CELL_PALETTE = [25.0, 75.0, 130.0, 180.0, 230.0]


def cell_color(patch_salt: int, col: int, row: int) -> float:
    """Deterministically hash one checkerboard CELL (identified by its
    integer column/row in the patch's own local frame) to one of
    CELL_PALETTE's five intensities, via a freshly-SEEDED xorshift32 stream
    (the same generator this whole script uses elsewhere — see module
    header) rather than a different hash family: combine patch_salt/col/row
    into one seed, draw ONE value from a fresh stream keyed on that seed.
    This is deterministic (same inputs -> same color, every run) and
    intentionally NON-periodic (no two nearby cells are guaranteed to
    differ OR to repeat with any regular period)."""
    seed = (patch_salt + (col * 92821) + (row * 68917)) & 0xFFFFFFFF
    if seed == 0:
        seed = 1   # xorshift32's one dead state — see XorShift32.__init__
    idx = XorShift32(seed).next_u32() % len(CELL_PALETTE)
    return CELL_PALETTE[idx]


def render_scene(params, x: float, y: float) -> float:
    """Evaluate the analytic scene at continuous (x, y), painter's-algorithm
    style: background, then disks, then checkerboards on top (checkerboards
    drawn last so their high-frequency corners are never occluded — they
    are the primary feature-detector target). Returns intensity in
    [0, 255] BEFORE any brightness offset (the caller adds that, if any)."""
    base, kx, ky = params["bg"]
    color = base + kx * x + ky * y

    for d in params["disks"]:
        dx, dy = x - d["cx"], y - d["cy"]
        if dx * dx + dy * dy <= d["r"] * d["r"]:
            color = d["inten"]

    for cb in params["checkers"]:
        dx, dy = x - cb["cx"], y - cb["cy"]
        c, s = math.cos(cb["angle_rad"]), math.sin(cb["angle_rad"])
        # Rotate the world offset INTO the checkerboard's own local frame
        # (inverse rotation: apply R(-angle) = R(angle)^T to (dx,dy)).
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        if abs(lx) <= cb["half"] and abs(ly) <= cb["half"]:
            col = math.floor(lx / cb["square"])
            row = math.floor(ly / cb["square"])
            color = cell_color(cb["patch_salt"], col, row)

    return color


SUPERSAMPLE = 4   # NxN sub-samples averaged per output pixel — see below

# Sub-pixel OFFSETS for a SUPERSAMPLE x SUPERSAMPLE regular grid within one
# output pixel, e.g. for SUPERSAMPLE=4: (-0.375,-0.125,0.125,0.375) in both
# axes (the standard "N evenly-spaced sample centers covering [-0.5,0.5)"
# formula). Precomputed once, shared by every pixel of every image.
_SS_OFFSETS = [(-0.5 + (i + 0.5) / SUPERSAMPLE) for i in range(SUPERSAMPLE)]


def render_image(params, transform_view: bool, brightness_offset: float) -> bytes:
    """Rasterize one WxH grayscale image, ANTI-ALIASED by averaging
    SUPERSAMPLE x SUPERSAMPLE sub-pixel samples per output pixel (a box
    filter over the pixel's footprint — the same "integrate incident light
    over the sensor pixel's area" idea a real camera's optics+sensor
    perform physically; THEORY.md's "problem" section discusses this as
    the synthetic stand-in for a camera's point-spread function).

    Why this matters for THIS project specifically: without anti-aliasing,
    every shape boundary is a single-pixel-sharp step, and FAST/Harris
    corner LOCATIONS near such a step are hypersensitive to exactly which
    side of the boundary each integer pixel happens to fall on — a sub-
    pixel difference in viewpoint (exactly what the 12-degree scene_b.pgm
    rotation causes) can shift the detected corner by several pixels or
    make it vanish/appear entirely. A softened (but still sharp-CORNERED,
    since only ~1 pixel of blur is added) boundary is far more repeatable
    under resampling — this was root-caused empirically: an earlier
    non-anti-aliased version of this scene measured single-digit-percent
    repeatability; THEORY.md "How we verify correctness" records the
    before/after numbers.

    transform_view=False -> scene A: sample near each integer pixel (px,py)
        directly (identity pose, no offset).
    transform_view=True  -> scene B (or the negative-control image): for
        each output pixel (px, py), each SUB-sample is mapped through
        inverse_transform() individually (so the anti-aliasing footprint
        is correctly the OUTPUT pixel's footprint, projected back into the
        scene — the physically correct order of operations), THEN
        brightness_offset is added and the AVERAGED result is clipped.

    Returns W*H raw bytes, row-major, one byte per pixel (PGM P5 payload).
    """
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
# Scene A/B shape layout — five checkerboard patches at different scales
# and orientations, four disks, a smooth gradient background. Anchor
# points are spread across the five "zones" of the 256x256 canvas (four
# quadrant-ish positions + the center) with at least kBorder=16px of
# clearance from every edge even after the +-8px jitter above, so corner-
# rich texture is never wasted on ineligible border pixels.
# ===========================================================================
def build_scene_a_params():
    # Square sizes are deliberately LARGE relative to the image (16-24 px,
    # a handful of cells per patch instead of a dense grid): a corner's
    # local FAST/Harris neighborhood (radius <= kOrientPatchRadius = 15 px)
    # then sits mostly INSIDE one dominant cell on either side of the true
    # boundary, so the corner survives the sub-pixel resampling a 12-degree
    # rotation causes (see main.cu's repeatability gate). An earlier,
    # denser version of this scene (9-16 px squares) measured only
    # 9/49 = 18% repeatability; THEORY.md "How we verify correctness"
    # records the before/after numbers from this exact change.
    anchors_checker = [
        # (cx, cy, angle_deg, square_px, half_extent_px)
        (68, 66, 0.0, 18.0, 46.0),
        (188, 64, 25.0, 20.0, 44.0),
        (64, 190, -35.0, 16.0, 42.0),
        (192, 192, 55.0, 22.0, 48.0),
        (128, 128, -15.0, 17.0, 30.0),
    ]
    anchors_disk = [
        # (cx, cy, radius_px, intensity_0_255)
        (128, 42, 16.0, 60.0),
        (42, 128, 14.0, 200.0),
        (213, 128, 18.0, 95.0),
        (128, 213, 15.0, 175.0),
    ]
    # Background gradient: 70 (top-left-ish) ramping up to ~160 toward the
    # bottom-right — smooth (zero second derivative away from shape edges),
    # so FAST/Harris correctly find almost no corners in open background.
    return build_scene_params(seed=42, anchors_checker=anchors_checker, anchors_disk=anchors_disk,
                              bg_base=70.0, bg_kx=60.0 / (W - 1), bg_ky=30.0 / (H - 1))


def build_scene_c_params():
    """A DIFFERENT scene (different anchor layout, different seed=999) for
    the negative-control gate — see main.cu's gate_negative_control()."""
    anchors_checker = [
        # Same large-square rationale as build_scene_a_params() above.
        (188, 70, 40.0, 19.0, 42.0),
        (68, 190, -20.0, 16.0, 46.0),
        (128, 48, 60.0, 21.0, 34.0),
        (128, 205, -50.0, 15.0, 40.0),
        (48, 48, 10.0, 18.0, 34.0),
    ]
    anchors_disk = [
        (200, 200, 17.0, 210.0),
        (200, 56, 13.0, 70.0),
        (56, 128, 19.0, 150.0),
        (150, 150, 12.0, 40.0),
    ]
    # A different gradient direction/range (high top-right -> low bottom-
    # left) so the two scenes are not even statistically similar in bulk
    # brightness layout, on top of the entirely different shape placement.
    return build_scene_params(seed=999, anchors_checker=anchors_checker, anchors_disk=anchors_disk,
                              bg_base=150.0, bg_kx=-50.0 / (W - 1), bg_ky=-40.0 / (H - 1))


def write_transform_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground-truth transform, scene_a.pgm -> scene_b.pgm\n")
        f.write("# generated by scripts/make_synthetic.py -- AUTHORITATIVE copy is\n")
        f.write("# hardcoded (cross-referenced) in ../src/main.cu's kTransform* constants\n")
        f.write("# forward(xa,ya) = R(theta_deg) * (xa-cx, ya-cy) + (cx,cy) + (tx_px, ty_px)\n")
        writer = csv.writer(f)
        writer.writerow(["field", "value", "units"])
        writer.writerow(["theta_deg", f"{TRANSFORM_THETA_DEG:.6f}", "degrees, counter-clockwise"])
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
                    "transform CSV) for project 01.04 (Feature pipeline).")
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
          f"transformed theta={TRANSFORM_THETA_DEG}deg tx={TRANSFORM_TX_PX}px ty={TRANSFORM_TY_PX}px "
          f"brightness+={TRANSFORM_BRIGHTNESS_OFFSET})")
    write_pgm(args.out_dir / "neg_scene_c.pgm", W, H, scene_c)
    print(f"[make_synthetic] wrote {args.out_dir / 'neg_scene_c.pgm'} ({W}x{H}, SYNTHETIC, seed=999, "
          f"UNRELATED scene -- negative control)")

    write_transform_csv(args.out_dir / "transform.csv")


if __name__ == "__main__":
    main()
