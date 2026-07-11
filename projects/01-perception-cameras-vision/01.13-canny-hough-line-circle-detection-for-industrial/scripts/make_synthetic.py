#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.13
(Canny + Hough line/circle detection for industrial alignment).

Renders a machined rectangular plate (4 straight edges, 3 drilled alignment
holes of known distinct radii) under a KNOWN in-plane offset+rotation, the
way a fixture, conveyor, or robot pick error would present it to a camera on
a real inspection line. The pipeline under test (src/) recovers the plate's
edges, holes, and that applied (dx, dy, dtheta) purely from the rendered
pixels; this script is what manufactures both the pixels AND the ground
truth they are checked against.

Why pure Python stdlib, no numpy (CLAUDE.md paragraph 8 + this project's
brief): reproducibility across any learner's machine with zero extra
installs. Anti-aliasing follows the SAME analytic-scene + N x N supersample
pattern project 01.04's generator uses (cite:
projects/01-perception-cameras-vision/01.04-feature-pipeline/scripts/make_synthetic.py) —
re-derived independently here for this project's plate/hole/scratch scene,
per CLAUDE.md's cross-project "deliberate, documented duplication" norm.

GEOMETRY CONTRACT (load-bearing): every constant in the "geometry" section
below MUST match src/kernels.cuh SECTION 2/3 exactly — Python cannot #include
a C++ header, so this is the one place in the project where that duplication
is unavoidable (kernels.cuh says so too; if you change one, change both).

Usage
-----
    python make_synthetic.py                      # defaults: seed 42, dx=8, dy=-5, dtheta=7 deg
    python make_synthetic.py --seed 42 --dx 8 --dy -5 --dtheta-deg 7 --out-dir ../data/sample
"""

import argparse
import hashlib
import math
from pathlib import Path

# ===========================================================================
# XorShift32 — the repo's portable deterministic PRNG (identical algorithm
# to src/kernels.cuh's implicit C++ generator convention and to every other
# project's make_synthetic.py, e.g. 01.04's). Same seed => same bytes, on
# any machine, forever — the whole point of "synthetic, not downloaded."
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
    """Deterministically fold (base, a, b) into one xorshift32 seed — same
    hashing idiom as 01.04's cell_color() (large odd-ish multipliers spread
    small integer coordinates across the 32-bit seed space)."""
    return (base + a * 92821 + b * 68917) & 0xFFFFFFFF


# ===========================================================================
# GEOMETRY CONTRACT — MUST MATCH src/kernels.cuh SECTIONS 1-3 exactly.
# ===========================================================================
IMG_W, IMG_H = 320, 240
IMG_CX, IMG_CY = 160.0, 120.0          # image position of the plate's NOMINAL (untransformed) center

PART_HALF_W, PART_HALF_H = 70.0, 45.0  # plate half-extents, local frame (px)

# (local_x, local_y, radius) — index order MUST match kernels.cuh's
# HOLE_LOCAL_X/Y/RADIUS (radii 6, 8, 10 respectively — the KNOWN nominal set).
HOLES_LOCAL = [
    (45.0, -15.0, 6.0),
    (-40.0, -20.0, 8.0),
    (5.0, 30.0, 10.0),
]

# The engineered weak-but-connected scratch mark (kernels.cuh's
# SCRATCH_LOCAL_*): starts ON the top edge, runs 25 px into the interior.
# Rendered as a ONE-SIDED band (a shallow shaded strip), not a thin two-
# edged groove: an early version used a thin "valley" (both walls within a
# couple pixels of each other) and MEASURED that its two opposing gradients
# partially cancelled under the 5-tap Gaussian blur, leaving a gradient
# magnitude too small to ever cross even T_LOW — a real lesson about narrow
# features and blur radius (THEORY.md "Numerical considerations"). A single-
# sided step (like a shallow anodizing/finish-line boundary) has ONE clean
# gradient, scaling linearly with contrast the way a normal edge does.
SCRATCH_LOCAL = (0.0, -PART_HALF_H, 0.0, -PART_HALF_H + 25.0)   # the LEADING edge — what the gate checks
SCRATCH_WIDTH = 8.0   # px, local frame — the band's extent to the +local-x side of the leading edge

# ===========================================================================
# Rendering parameters — MEASURED against this scene by src/main.cu's Canny
# thresholds (kernels.cuh CANNY_T_LOW/T_HIGH = 20/55 on Sobel-scaled
# gradient magnitude); chosen so that: (a) the plate/hole boundaries clear
# T_HIGH comfortably, (b) the scratch mark's own gradient lands BETWEEN
# T_LOW and T_HIGH (recoverable only by hysteresis propagation from the top
# edge it touches — the whole point of the hysteresis_lesson gate), and
# (c) background brushed-metal texture + sensor noise never crosses T_LOW
# by more than an isolated pixel or two (the negative_control gate).
# ===========================================================================
BG_GRAY = 95.0        # table/fixture background, mean gray level
PLATE_GRAY = 200.0    # machined plate top face, mean gray level (bright: it is lit and polished)
HOLE_GRAY = 45.0      # drilled hole interior (shadowed — genuinely darker, not just "different")
SCRATCH_DELTA = 40.0  # plate_gray minus this = the shallow scribe band's gray level (LOW contrast,
                      # by design, but NOT thin — see SCRATCH_WIDTH above; MEASURED to land the
                      # band's own gradient between T_LOW and T_HIGH after blur, see THEORY.md)
# The first SCRATCH_SEED_LEN px of the band (right where it meets the top
# edge) use a MUCH deeper "seed" contrast instead of the shallow one. Why:
# a bare T-junction (a weak edge meeting a strong one at 90 degrees) MEASURED
# a genuine Gaussian-blur corner-rounding artifact — the gradient dips below
# even T_LOW for a pixel or two exactly at the junction, breaking 8-
# connectivity so hysteresis never seeds into the weak chain at all
# (THEORY.md "Numerical considerations" reproduces the failed attempt: a
# uniform-contrast band measured ~0.1 recovered fraction for BOTH double-
# AND single-threshold, i.e. no propagation ever happened). A short deep
# "seed" segment guarantees an unbroken STRONG bridge into the weak chain —
# physically like a scribe mark that starts as a firm score and fades —
# so double-threshold hysteresis can propagate the REST of the mark from it.
SCRATCH_SEED_LEN = 3.0     # px, local frame — kept short so it barely affects the single-threshold fraction
SCRATCH_SEED_DELTA = 110.0 # px contrast in the seed zone (deep — comparable to the hole contrast)

VIGNETTE_STRENGTH = 0.18   # multiplicative darkening at the image corners (simple lens-falloff stand-in)

PLATE_BAND_AMP = 4.0   # brushed-metal streaks: per-LOCAL-row jitter (streaks run along local +x)
PLATE_FINE_AMP = 2.0   # brushed-metal fine per-pixel grain, on top of the streak banding
BG_BAND_AMP = 3.0      # background surface texture: same idea, in IMAGE coordinates (a fixed table, not the moving part)
BG_FINE_AMP = 2.0

NOISE_AMP = 3.0         # zero-mean uniform sensor noise, +/- this many gray levels, added per OUTPUT pixel
SUPERSAMPLE = 3          # NxN sub-samples per pixel for shape-boundary anti-aliasing (see render_image)

DEFAULT_SEED = 42
DEFAULT_DX = 8.0
DEFAULT_DY = -5.0
DEFAULT_DTHETA_DEG = 7.0


def to_local(x: float, y: float, dx: float, dy: float, dtheta: float):
    """Inverse of the forward transform image = R(dtheta)*local + (IMG_CX+dx,
    IMG_CY+dy): map an IMAGE-frame point back into the plate's LOCAL frame.
    Used to test shape membership (is this point inside the plate rectangle?
    inside a hole? near the scratch segment?) in the frame those shapes are
    naturally axis-aligned in."""
    cx, cy = IMG_CX + dx, IMG_CY + dy
    rx, ry = x - cx, y - cy
    c, s = math.cos(-dtheta), math.sin(-dtheta)
    lx = rx * c - ry * s
    ly = rx * s + ry * c
    return lx, ly


def shape_gray(x: float, y: float, dx: float, dy: float, dtheta: float, draw_part: bool) -> float:
    """Evaluate the scene's BASE gray level (no texture/vignette/noise yet)
    at a continuous point, painter's-algorithm style: background, then the
    plate rectangle, then holes (darker, drawn over the plate), then the
    scratch mark (drawn over the plate, never inside a hole by construction
    — see the geometry contract's placement). draw_part=False renders the
    NEGATIVE CONTROL scene: background/texture/vignette/noise only, no part
    at all — see the negative_control gate this feeds."""
    if draw_part:
        lx, ly = to_local(x, y, dx, dy, dtheta)
        if -PART_HALF_W <= lx <= PART_HALF_W and -PART_HALF_H <= ly <= PART_HALF_H:
            for hlx, hly, hr in HOLES_LOCAL:
                if math.hypot(lx - hlx, ly - hly) <= hr:
                    return HOLE_GRAY
            # one-sided band: local x in [0, SCRATCH_WIDTH], local y spanning
            # the scratch's length (SCRATCH_LOCAL's y0..y1) — see SCRATCH_LOCAL's
            # comment for why this replaced a thin two-edged groove. The first
            # SCRATCH_SEED_LEN px use the deep "seed" contrast (see its
            # comment above) so hysteresis has an unbroken strong bridge to
            # propagate from; the remaining length is the LOW, weak contrast.
            sy0, sy1 = SCRATCH_LOCAL[1], SCRATCH_LOCAL[3]
            if 0.0 <= lx <= SCRATCH_WIDTH and sy0 <= ly <= sy1:
                if ly <= sy0 + SCRATCH_SEED_LEN:
                    return PLATE_GRAY - SCRATCH_SEED_DELTA
                return PLATE_GRAY - SCRATCH_DELTA
            return PLATE_GRAY
    return BG_GRAY


def vignette_factor(x: float, y: float) -> float:
    """Smooth radial darkening toward the corners — the synthetic stand-in
    for real lens/sensor falloff (THEORY.md's "problem" section discusses
    the physical cause). max_dist normalizes so the factor reaches exactly
    (1 - VIGNETTE_STRENGTH) at the image corners, 1.0 at the center."""
    max_dist = math.hypot(IMG_CX, IMG_CY)
    d = math.hypot(x - IMG_CX, y - IMG_CY) / max_dist
    return 1.0 - VIGNETTE_STRENGTH * (d * d)


def texture_value(x: int, y: int, dx: float, dy: float, dtheta: float, draw_part: bool, salt: int) -> float:
    """Deterministic hashed texture, added AFTER shape anti-aliasing (at
    output-pixel resolution, not supersampled — a deliberate simplification:
    real sensor grain is a per-pixel phenomenon, while shape EDGES are what
    genuinely need sub-pixel anti-aliasing; see README "Limitations").

    Two components, matching how brushed metal actually looks: a coarse
    per-STREAK band (nearly constant along the brushing direction, jumps
    between streaks) plus fine per-pixel grain on top. The plate's streaks
    run along its own LOCAL +x axis (so they rotate WITH the part, like a
    real machined surface would); the background's streaks run in fixed
    IMAGE coordinates (a stationary table), by design a different axis."""
    if draw_part:
        lx, ly = to_local(float(x), float(y), dx, dy, dtheta)
        if -PART_HALF_W <= lx <= PART_HALF_W and -PART_HALF_H <= ly <= PART_HALF_H:
            band = round(ly)  # streak identity: constant along local x, changes with local y
            fine = round(lx)
            band_v = XorShift32(_mix_seed(salt, band, 0)).uniform(-PLATE_BAND_AMP, PLATE_BAND_AMP)
            fine_v = XorShift32(_mix_seed(salt, band, fine)).uniform(-PLATE_FINE_AMP, PLATE_FINE_AMP)
            return band_v + fine_v
    band = y  # background streaks run in fixed image rows
    fine = x
    band_v = XorShift32(_mix_seed(salt + 777, band, 0)).uniform(-BG_BAND_AMP, BG_BAND_AMP)
    fine_v = XorShift32(_mix_seed(salt + 777, band, fine)).uniform(-BG_FINE_AMP, BG_FINE_AMP)
    return band_v + fine_v


_SS_OFFSETS = [(-0.5 + (i + 0.5) / SUPERSAMPLE) for i in range(SUPERSAMPLE)]


def render_image(dx: float, dy: float, dtheta: float, draw_part: bool, seed: int) -> bytes:
    """Rasterize one IMG_W x IMG_H grayscale frame: shape boundaries
    anti-aliased by SUPERSAMPLE x SUPERSAMPLE box-filtered sampling (the
    same "integrate over the sensor pixel's footprint" idea as 01.04's
    generator — see that project's THEORY.md for the corner-repeatability
    argument this borrows), then texture + vignette + noise applied once per
    OUTPUT pixel (see texture_value's docstring for why that stage is not
    supersampled)."""
    n_samples = SUPERSAMPLE * SUPERSAMPLE
    out = bytearray(IMG_W * IMG_H)
    for y in range(IMG_H):
        for x in range(IMG_W):
            total = 0.0
            for oy in _SS_OFFSETS:
                for ox in _SS_OFFSETS:
                    total += shape_gray(x + ox, y + oy, dx, dy, dtheta, draw_part)
            base = total / n_samples

            v = base * vignette_factor(x, y)
            v += texture_value(x, y, dx, dy, dtheta, draw_part, seed)
            noise = XorShift32(_mix_seed(seed + 31337, y, x)).uniform(-NOISE_AMP, NOISE_AMP)
            v += noise

            v = 0.0 if v < 0.0 else (255.0 if v > 255.0 else v)
            out[y * IMG_W + x] = int(v + 0.5)
    return bytes(out)


def write_pgm(path: Path, w: int, h: int, pixels: bytes) -> None:
    """Write a binary PGM (P5) grayscale image — no libraries, no black box:
    a PGM is a short ASCII header ("P5\\nW H\\n255\\n") followed by raw 8-bit
    samples, row-major. src/main.cu's loader reads this exact format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(pixels)


def compute_truth(dx: float, dy: float, dtheta: float):
    """Compute exact ground-truth line/hole/scratch parameters in the IMAGE
    frame by literally transforming the local-frame geometry — the same
    transform render_image()'s inverse uses, applied forward here, so the
    truth file and the rendered pixels can never silently disagree."""
    c, s = math.cos(dtheta), math.sin(dtheta)

    def fwd(lx, ly):
        x = lx * c - ly * s + IMG_CX + dx
        y = lx * s + ly * c + IMG_CY + dy
        return x, y

    # 4 edges as (name, theta0_local, rho0_local, corner_a_local, corner_b_local).
    edges_local = [
        ("left",   0.0,            -PART_HALF_W, (-PART_HALF_W, -PART_HALF_H), (-PART_HALF_W, PART_HALF_H)),
        ("right",  0.0,             PART_HALF_W, ( PART_HALF_W, -PART_HALF_H), ( PART_HALF_W, PART_HALF_H)),
        ("top",    math.pi / 2.0,  -PART_HALF_H, (-PART_HALF_W, -PART_HALF_H), ( PART_HALF_W, -PART_HALF_H)),
        ("bottom", math.pi / 2.0,   PART_HALF_H, (-PART_HALF_W,  PART_HALF_H), ( PART_HALF_W,  PART_HALF_H)),
    ]
    lines = []
    for name, theta0, rho0_local, ca, cb in edges_local:
        theta = theta0 + dtheta
        # wrap to [0, pi): Hough theta has period pi, and dtheta is small
        # enough here that no wrap is ever actually needed, but do it
        # correctly regardless of the applied rotation's sign/magnitude.
        theta = theta % math.pi
        rho = rho0_local + IMG_CX * math.cos(theta0 + dtheta) + IMG_CY * math.sin(theta0 + dtheta) \
            + dx * math.cos(theta0 + dtheta) + dy * math.sin(theta0 + dtheta)
        ax, ay = fwd(*ca)
        bx, by = fwd(*cb)
        lines.append((name, theta, rho, ax, ay, bx, by))

    holes = []
    for hlx, hly, hr in HOLES_LOCAL:
        hx, hy = fwd(hlx, hly)
        holes.append((hx, hy, hr))

    sx0, sy0 = fwd(SCRATCH_LOCAL[0], SCRATCH_LOCAL[1])
    sx1, sy1 = fwd(SCRATCH_LOCAL[2], SCRATCH_LOCAL[3])

    return lines, holes, (sx0, sy0, sx1, sy1)


def write_truth_csv(path: Path, dx: float, dy: float, dtheta: float) -> None:
    """Write the ground-truth CSV src/main.cu loads for every gate: the
    applied transform, the 4 image-frame lines (with both the (theta,rho)
    Hough form and the two corner endpoints, for the finite-segment
    edge_quality mask), the 3 hole centers/radii, and the scratch segment's
    endpoints (for the hysteresis_lesson gate)."""
    lines, holes, scratch = compute_truth(dx, dy, dtheta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground truth — generated by scripts/make_synthetic.py for project 01.13\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {DEFAULT_SEED} --dx {dx} --dy {dy} --dtheta-deg {math.degrees(dtheta)}\n")
        f.write("# TRANSFORM,dx_px,dy_px,dtheta_rad  (applied rigid offset the alignment solve must recover)\n")
        f.write("# LINE,name,theta_rad,rho_px,ax,ay,bx,by  (Hough form + finite-segment endpoints, image frame)\n")
        f.write("# HOLE,cx,cy,r_px  (image frame)\n")
        f.write("# SCRATCH,x0,y0,x1,y1  (the weak-but-connected engineered segment, image frame)\n")
        f.write(f"TRANSFORM,{dx:.6f},{dy:.6f},{dtheta:.6f}\n")
        for name, theta, rho, ax, ay, bx, by in lines:
            f.write(f"LINE,{name},{theta:.6f},{rho:.6f},{ax:.6f},{ay:.6f},{bx:.6f},{by:.6f}\n")
        for hx, hy, hr in holes:
            f.write(f"HOLE,{hx:.6f},{hy:.6f},{hr:.6f}\n")
        sx0, sy0, sx1, sy1 = scratch
        f.write(f"SCRATCH,{sx0:.6f},{sy0:.6f},{sx1:.6f},{sy1:.6f}\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dx", type=float, default=DEFAULT_DX, help="applied translation, image x (px)")
    ap.add_argument("--dy", type=float, default=DEFAULT_DY, help="applied translation, image y (px)")
    ap.add_argument("--dtheta-deg", type=float, default=DEFAULT_DTHETA_DEG, help="applied rotation (degrees)")
    ap.add_argument("--out-dir", type=Path, default=default_out)
    args = ap.parse_args()

    dtheta = math.radians(args.dtheta_deg)

    scene_path = args.out_dir / "scene.pgm"
    negctrl_path = args.out_dir / "negative_control.pgm"
    truth_path = args.out_dir / "truth.csv"

    print(f"[make_synthetic] rendering scene.pgm ({IMG_W}x{IMG_H}, seed={args.seed}, "
          f"dx={args.dx} px, dy={args.dy} px, dtheta={args.dtheta_deg} deg)...")
    scene_pixels = render_image(args.dx, args.dy, dtheta, draw_part=True, seed=args.seed)
    write_pgm(scene_path, IMG_W, IMG_H, scene_pixels)

    print("[make_synthetic] rendering negative_control.pgm (texture/vignette/noise only, no part)...")
    negctrl_pixels = render_image(args.dx, args.dy, dtheta, draw_part=False, seed=args.seed)
    write_pgm(negctrl_path, IMG_W, IMG_H, negctrl_pixels)

    print("[make_synthetic] writing truth.csv...")
    write_truth_csv(truth_path, args.dx, args.dy, dtheta)

    for p in (scene_path, negctrl_path, truth_path):
        print(f"[make_synthetic] wrote {p}  sha256={sha256_of(p)}")


if __name__ == "__main__":
    main()
