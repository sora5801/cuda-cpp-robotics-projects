#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.14
(Template matching (NCC) at scale for pick verification).

Renders a 6x4 (24-slot) pick-and-place TRAY photographed after a robot pick
cycle, plus a 15-template "golden reference" set (3 machined part shapes x 5
pre-rotated angles), the way a real vision station's teach-in step would
capture them. The pipeline under test (src/) checks every tray slot against
its expected template using NCC, searched over a small translation window
and the rotation set; this script manufactures both the pixels AND the
ground truth (per-slot expected/actual type, applied offset/rotation/shadow,
and the correct verdict) they are checked against.

GEOMETRY CONTRACT (load-bearing): every constant in the "geometry contract"
section below MUST match src/kernels.cuh SECTIONS 1-2 exactly — Python cannot
#include a C++ header, so this is the one place in the project where that
duplication is unavoidable (kernels.cuh says so too; if you change one,
change both). The visual/rendering style (XorShift32, hashed per-pixel
texture, NxN supersampled anti-aliasing) deliberately follows project
01.13's generator (cite:
projects/01-perception-cameras-vision/01.13-canny-hough-line-circle-detection-for-industrial/scripts/make_synthetic.py)
— re-derived independently here for this project's tray/slot/part scene,
per CLAUDE.md's cross-project "deliberate, documented duplication" norm.

Usage
-----
    python make_synthetic.py                      # default: seed 42
    python make_synthetic.py --seed 42 --out-dir ../data/sample
"""

import argparse
import hashlib
import math
from pathlib import Path

# ===========================================================================
# XorShift32 — the repo's portable deterministic PRNG (CLAUDE.md's "no
# std::uniform_real_distribution" rule, mirrored here in Python: the stdlib
# `random` module is NOT used so the byte-for-byte contract matches every
# other project's C++ noise generator in spirit, seed in / bytes out).
# ===========================================================================
class XorShift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1  # xorshift32's one dead state: 0 maps to 0 forever

    def next_u32(self) -> int:
        x = self.state
        x = (x ^ (x << 13)) & 0xFFFFFFFF
        x = (x ^ (x >> 17)) & 0xFFFFFFFF
        x = (x ^ (x << 5)) & 0xFFFFFFFF
        self.state = x
        return x

    def uniform(self, lo: float, hi: float) -> float:
        u = self.next_u32() / 4294967296.0  # 2^32 -> [0, 1)
        return lo + u * (hi - lo)


def _mix_seed(base: int, a: int, b: int) -> int:
    """Deterministically fold (base, a, b) into one xorshift32 seed — the
    same large-odd-multiplier hashing idiom as 01.13's/01.04's generators,
    re-derived here so this project's texture/noise are reproducible without
    importing another project's script (CLAUDE.md's self-containment rule)."""
    return (base + a * 92821 + b * 68917) & 0xFFFFFFFF


def clamp_u8(v: float) -> int:
    if v < 0.0:
        v = 0.0
    elif v > 255.0:
        v = 255.0
    return int(v + 0.5)


# ===========================================================================
# GEOMETRY CONTRACT — MUST MATCH src/kernels.cuh SECTIONS 1-2 exactly.
# ===========================================================================
TEMPLATE_SIZE = 24
SEARCH_RADIUS = 8
WINDOW = TEMPLATE_SIZE + 2 * SEARCH_RADIUS          # 40

NUM_COLS, NUM_ROWS = 6, 4
NUM_SLOTS = NUM_COLS * NUM_ROWS                     # 24 = K
SLOT_GAP = 12
SLOT_PITCH = WINDOW + SLOT_GAP                      # 52
BORDER = 12

IMG_W = 2 * BORDER + (NUM_COLS - 1) * SLOT_PITCH + WINDOW   # 324
IMG_H = 2 * BORDER + (NUM_ROWS - 1) * SLOT_PITCH + WINDOW   # 220

NUM_TYPES = 3
TYPE_BRACKET, TYPE_GEAR_DISK, TYPE_CONNECTOR_BLOCK = 0, 1, 2
NUM_ROT = 5
ROTATION_DEG = [-6.0, -3.0, 0.0, 3.0, 6.0]
NUM_TEMPLATES = NUM_TYPES * NUM_ROT                 # 15


def slot_window_x0(slot: int) -> int:
    return BORDER + (slot % NUM_COLS) * SLOT_PITCH


def slot_window_y0(slot: int) -> int:
    return BORDER + (slot // NUM_COLS) * SLOT_PITCH


# ===========================================================================
# Rendering parameters (MEASURED against this scene by src/main.cu's
# classification thresholds — see THEORY.md "How we verify correctness" for
# the actual score numbers these were tuned against).
# ===========================================================================
BG_GRAY = 90.0        # tray surface, mean gray level
PART_GRAY = 195.0     # machined part top face, mean gray level (bright: lit, painted/anodized)
TEXTURE_AMP = 4.0      # hashed per-pixel texture jitter (+- gray levels)
NOISE_AMP = 3.0        # zero-mean sensor noise, added per OUTPUT pixel (+- gray levels)
SUPERSAMPLE = 3         # NxN sub-samples per pixel for shape-boundary anti-aliasing

# The SHADOWED cohort's illumination gradient (README/THEORY "why NCC, not
# SSD"): a smooth multiplicative dimming across the slot's search window,
# from SHADOW_HI (near-full brightness) at one corner to SHADOW_LO (heavily
# darkened) at the opposite corner — like a fixture arm's cast shadow
# creeping across one tray position. Deliberately kept close to a GLOBAL
# scale change (nearly-affine across the small window) so NCC's brightness-
# affine invariance (THEORY.md "The math") cleanly survives it while a plain
# SSD match score (no normalization at all) visibly degrades — the designed
# comparison the illumination_robustness gate checks.
SHADOW_HI = 0.85
SHADOW_LO = 0.50

DEFAULT_SEED = 42

_SS_OFFSETS = [(-0.5 + (i + 0.5) / SUPERSAMPLE) for i in range(SUPERSAMPLE)]


def shape_membership(ptype: int, lx: float, ly: float) -> bool:
    """Analytic silhouette test in the part's OWN local frame (origin at the
    part's nominal center, +x right, +y down — matching image convention,
    the same choice project 01.13's plate geometry makes). Three distinct
    machined-part silhouettes, in the "one bright face, drawn over a darker
    tray" style of 01.13's plate/holes:

      TYPE_BRACKET (0): an L-shaped corner bracket — the union of a vertical
        arm and a horizontal arm (two rectangles).
      TYPE_GEAR_DISK (1): a hub circle plus 6 evenly-spaced rectangular
        teeth (a stylized gear).
      TYPE_CONNECTOR_BLOCK (2): a rectangle with two circular mounting holes
        removed (like a small connector housing)."""
    if ptype == TYPE_BRACKET:
        vertical_arm = (-10.0 <= lx <= -4.0) and (-10.0 <= ly <= 10.0)
        horizontal_arm = (-10.0 <= lx <= 10.0) and (4.0 <= ly <= 10.0)
        return vertical_arm or horizontal_arm

    if ptype == TYPE_GEAR_DISK:
        if math.hypot(lx, ly) <= 7.0:
            return True
        # 6 teeth at 60-degree spacing: rotate the query point INTO each
        # tooth's own local frame (tooth 0 points along +x) and test a small
        # rectangle spanning the hub-to-tip radius — the same to_local
        # "un-rotate the query, keep the shape axis-aligned" idea used below
        # for the part's own overall rotation, applied per-tooth here.
        for k in range(6):
            a = math.radians(k * 60.0)
            c, s = math.cos(-a), math.sin(-a)
            rx = lx * c - ly * s
            ry = lx * s + ly * c
            if 7.0 <= rx <= 10.0 and -1.5 <= ry <= 1.5:
                return True
        return False

    # TYPE_CONNECTOR_BLOCK
    if not (-10.0 <= lx <= 10.0 and -7.0 <= ly <= 7.0):
        return False
    if math.hypot(lx + 5.0, ly) <= 2.5:
        return False
    if math.hypot(lx - 5.0, ly) <= 2.5:
        return False
    return True


def to_local(px: float, py: float, cx: float, cy: float, rot_deg: float):
    """Map a PATCH-pixel-space point (px,py) into the part's own LOCAL,
    axis-aligned frame: translate so the part's center is the origin, then
    rotate by -rot_deg (inverse of "the part is rotated by rot_deg") so
    shape_membership can test against the shape's un-rotated definition —
    the same to_local pattern 01.13's generator uses for its plate."""
    rx, ry = px - cx, py - cy
    a = math.radians(-rot_deg)
    c, s = math.cos(a), math.sin(a)
    lx = rx * c - ry * s
    ly = rx * s + ry * c
    return lx, ly


def render_template(ptype: int, rot_deg: float) -> bytes:
    """Render ONE clean 'golden reference' template: shape silhouette only
    (BG_GRAY / PART_GRAY, anti-aliased), NO texture and NO sensor noise —
    standing in for a CAD-derived or single best-shot master image, exactly
    what a real teach-in step would capture. Centered in a TEMPLATE_SIZE x
    TEMPLATE_SIZE canvas."""
    size = TEMPLATE_SIZE
    cx = cy = size / 2.0
    out = bytearray(size * size)
    for py in range(size):
        for px in range(size):
            hits = 0
            for oy in _SS_OFFSETS:
                for ox in _SS_OFFSETS:
                    lx, ly = to_local(px + ox, py + oy, cx, cy, rot_deg)
                    if shape_membership(ptype, lx, ly):
                        hits += 1
            frac = hits / (SUPERSAMPLE * SUPERSAMPLE)
            v = BG_GRAY + (PART_GRAY - BG_GRAY) * frac
            out[py * size + px] = clamp_u8(v)
    return bytes(out)


def render_tray(cohorts: list, seed: int) -> bytes:
    """Render the full IMG_W x IMG_H tray scene. Every pixel gets hashed
    texture + sensor noise (keyed by its ABSOLUTE tray coordinate, so a
    slot's part and the surrounding background share one continuous texture
    field — no seam at a silhouette edge); slots with an actual part
    additionally get the shape's brightness blended in (anti-aliased), an
    optional positional offset/rotation, and an optional illumination gradient
    (SHADOW_HI/LO) for the shadowed cohort. EMPTY slots are left as plain
    textured/noisy background — nothing is drawn there at all."""
    canvas = bytearray(IMG_W * IMG_H)

    # Pass 1: background everywhere (also the FINAL value for empty slots
    # and the tray's borders/gaps — nothing overwrites it there).
    for y in range(IMG_H):
        for x in range(IMG_W):
            tv = XorShift32(_mix_seed(seed + 1, y, x)).uniform(-TEXTURE_AMP, TEXTURE_AMP)
            nv = XorShift32(_mix_seed(seed + 2, y, x)).uniform(-NOISE_AMP, NOISE_AMP)
            canvas[y * IMG_W + x] = clamp_u8(BG_GRAY + tv + nv)

    # Pass 2: paint each non-empty slot's part into its WINDOW region,
    # recomputing (not blending with) the background so the SAME texture/
    # noise hash is reused at every pixel regardless of shape coverage —
    # continuity by construction, not by special-casing frac==0.
    for slot, info in enumerate(cohorts):
        if info["actual_type"] is None:
            continue   # EMPTY cohort: leave the plain background from pass 1
        wx0, wy0 = slot_window_x0(slot), slot_window_y0(slot)
        cx = WINDOW / 2.0 + info["offset_dx"]
        cy = WINDOW / 2.0 + info["offset_dy"]
        for wy in range(WINDOW):
            for wx in range(WINDOW):
                hits = 0
                for oy in _SS_OFFSETS:
                    for ox in _SS_OFFSETS:
                        lx, ly = to_local(wx + ox, wy + oy, cx, cy, info["rotation_deg"])
                        if shape_membership(info["actual_type"], lx, ly):
                            hits += 1
                frac = hits / (SUPERSAMPLE * SUPERSAMPLE)
                base = BG_GRAY + (PART_GRAY - BG_GRAY) * frac

                ax, ay = wx0 + wx, wy0 + wy   # absolute tray coordinates
                tv = XorShift32(_mix_seed(seed + 1, ay, ax)).uniform(-TEXTURE_AMP, TEXTURE_AMP)
                v = base + tv

                if info["shadow"]:
                    # Smooth diagonal multiplicative dimming across the
                    # WINDOW-local coordinate (wx,wy in [0,WINDOW)): t=0 at
                    # the top-left corner, t=1 at the bottom-right — see the
                    # module-level SHADOW_HI/LO comment for why this is kept
                    # close to affine (a real, but honestly moderate, cast
                    # shadow), the case NCC is invariant to and raw SSD is not.
                    t = (wx + wy) / (2.0 * (WINDOW - 1))
                    factor = SHADOW_HI + (SHADOW_LO - SHADOW_HI) * t
                    v *= factor

                nv = XorShift32(_mix_seed(seed + 2, ay, ax)).uniform(-NOISE_AMP, NOISE_AMP)
                v += nv
                canvas[ay * IMG_W + ax] = clamp_u8(v)

    return bytes(canvas)


def build_cohorts() -> list:
    """The tray's designed layout (README/THEORY cite this table directly):
    a fixed 'recipe' expected_type = slot % NUM_TYPES (cycling
    bracket/gear/connector across all 24 slots, the realistic case of a tray
    that expects a repeating kit of parts), with 5 slots deliberately
    deviating from a perfect pick to exercise every gate:

        slot 3 — OFFSET cohort:   correct part (bracket), correct rotation,
                                   but placed (dx=+5, dy=-6) px off nominal —
                                   within the +-8 px search range
                                   (localization gate).
        slot 6 — ROTATED cohort:  correct part (bracket — MEASURED to be the
                                   most rotation-SENSITIVE of the 3 shapes,
                                   its mass concentrated far from the
                                   rotation center; see THEORY.md "How we
                                   verify correctness" for the measured
                                   falloff curve for all 3 shapes), correct
                                   position, but placed at 24 degrees — a
                                   deliberately LARGE angle relative to the
                                   +-6 degree rotation set, chosen (measured,
                                   not guessed) so that a single 0-degree
                                   template's score and the 5-angle set's
                                   best score straddle the classification
                                   threshold T_OK with real margin on both
                                   sides (rotation_lesson gate).
        slot 4 — EMPTY:           nothing was placed at all (a dropped pick).
        slot 5 — WRONG_PART:      a bracket sits where a connector block was
                                   expected.
        slot 7 — SHADOWED cohort: correct part (gear disk), correct
                                   position, but under the illumination
                                   gradient described above
                                   (illumination_robustness gate).

    Every other slot is a plain, well-placed, unshadowed correct pick — the
    realistic bulk of a healthy tray, and the classification gate's breadth
    check (zero misclassifications across ALL 24 slots, not just the 5
    designed edge cases)."""
    cohorts = []
    for slot in range(NUM_SLOTS):
        expected_type = slot % NUM_TYPES
        cohorts.append({
            "expected_type": expected_type,
            "actual_type": expected_type,
            "rotation_deg": 0.0,
            "offset_dx": 0,
            "offset_dy": 0,
            "shadow": False,
            "verdict": "OK",
            "cohort": "plain",
        })

    # slot 3 (expected type 0, bracket): OFFSET cohort.
    cohorts[3].update(offset_dx=5, offset_dy=-6, cohort="offset")
    # slot 6 (expected type 0, bracket): ROTATED cohort — 24 degrees, see the
    # docstring above for why this angle and this shape were chosen.
    cohorts[6].update(rotation_deg=24.0, cohort="rotated")
    # slot 5 (expected type 2, connector block): WRONG_PART — a bracket
    # (type 0) sits here instead.
    cohorts[5].update(actual_type=TYPE_BRACKET, verdict="WRONG_PART", cohort="wrong_part")
    # slot 4 (expected type 1, gear disk): EMPTY — nothing placed.
    cohorts[4].update(actual_type=None, verdict="EMPTY", cohort="empty")
    # slot 7 (expected type 1, gear disk): SHADOWED cohort.
    cohorts[7].update(shadow=True, cohort="shadow")
    return cohorts


def write_pgm(path: Path, w: int, h: int, pixels: bytes) -> None:
    """Write a binary PGM (P5) grayscale image — no libraries, no black box
    (CLAUDE.md's "no black boxes"): a short ASCII header followed by raw
    8-bit samples, row-major. src/main.cu's loader reads this exact format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(pixels)


def write_truth_csv(path: Path, cohorts: list) -> None:
    """Ground-truth CSV src/main.cu loads for every gate: one row per slot,
    the expected/actual part type and pose, and the correct verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground truth — generated by scripts/make_synthetic.py for project 01.14\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {DEFAULT_SEED}\n")
        f.write("# columns: slot,row,col,cohort,expected_type,actual_type(-1=empty),"
                "rotation_deg,offset_dx_px,offset_dy_px,shadow(0/1),verdict\n")
        for slot, info in enumerate(cohorts):
            actual = -1 if info["actual_type"] is None else info["actual_type"]
            f.write(f"{slot},{slot // NUM_COLS},{slot % NUM_COLS},{info['cohort']},"
                    f"{info['expected_type']},{actual},{info['rotation_deg']:.3f},"
                    f"{info['offset_dx']},{info['offset_dy']},{1 if info['shadow'] else 0},"
                    f"{info['verdict']}\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", type=Path, default=default_out)
    args = ap.parse_args()

    tray_path = args.out_dir / "tray.pgm"
    templates_path = args.out_dir / "templates.pgm"
    truth_path = args.out_dir / "truth.csv"

    cohorts = build_cohorts()

    print(f"[make_synthetic] rendering tray.pgm ({IMG_W}x{IMG_H}, {NUM_SLOTS} slots, seed={args.seed})...")
    tray_pixels = render_tray(cohorts, args.seed)
    write_pgm(tray_path, IMG_W, IMG_H, tray_pixels)

    print(f"[make_synthetic] rendering templates.pgm ({NUM_TEMPLATES} templates, "
          f"{TEMPLATE_SIZE}x{TEMPLATE_SIZE} each, stacked vertically)...")
    # Stacked VERTICALLY, template_id = type*NUM_ROT + rot_idx in order, so
    # the raw PGM bytes ARE exactly the flat [NUM_TEMPLATES][SIZE][SIZE]
    # array src/kernels.cuh expects — no reshuffling needed on the C++ side.
    template_rows = bytearray()
    for ptype in range(NUM_TYPES):
        for rot_deg in ROTATION_DEG:
            template_rows += render_template(ptype, rot_deg)
    write_pgm(templates_path, TEMPLATE_SIZE, TEMPLATE_SIZE * NUM_TEMPLATES, bytes(template_rows))

    print("[make_synthetic] writing truth.csv...")
    write_truth_csv(truth_path, cohorts)

    for p in (tray_path, templates_path, truth_path):
        print(f"[make_synthetic] wrote {p}  sha256={sha256_of(p)}")


if __name__ == "__main__":
    main()
