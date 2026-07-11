#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 01.07
(Fisheye/omnidirectional unwarping and multi-camera surround-view stitching).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
This project needs 4 fisheye camera images from KNOWN mount positions over
a ground plane with EXACT ground-truth texture — exactly the kind of thing
a real automotive surround-view rig cannot give you for free (you would
need a physical vehicle, 4 real fisheye cameras, a calibration target, and
still only get an approximate, hand-measured rig geometry). Synthetic
authorship gives all of it for free and EXACTLY: this script IS the world
the pipeline reconstructs, so main.cu's gates can compare the GPU's BEV
output against ground truth that is provably correct by construction.

What it writes (into ../data/sample/, well under the CLAUDE.md §8 budget):

    fisheye_front.ppm    FISH_W x FISH_H, 8-bit RGB (PPM P6) — analytically
    fisheye_left.ppm     ray-cast render of the ground/objects through each
    fisheye_right.ppm    rig camera's equidistant fisheye model (the SAME
    fisheye_rear.ppm     model ../src/kernels.cuh's C++ uses).
    bev_ground_truth.ppm BEV_W x BEV_H, 8-bit RGB (PPM P6) — an ORTHOGRAPHIC
                          top-down render of the ground TEXTURE ONLY (no
                          cameras, no objects, no occlusion): the exact
                          answer a flat-ground BEV reconstruction SHOULD
                          produce everywhere the flat-ground assumption
                          actually holds. main.cu's bev_ground_truth and
                          flat_ground_assumption gates compare the GPU's
                          real (camera-derived) BEV output against this file.
    rig_extrinsics.csv   human/machine-readable mirror of the 4 camera
                          mounts + rotation matrices this script computes
                          (see build_camera_matrix() below) — documentation/
                          provenance only; the C++ side never reads it (its
                          own copy is hand-rounded and hardcoded in
                          kernels.cuh, per that file's header).

The generation direction — RAY CASTING, the physically honest one
-------------------------------------------------------------------
For each fisheye camera, for each PIXEL (2x2 supersampled for anti-
aliasing), this script asks "which ray does this pixel see, and what does
that ray hit first?" — the exact inverse of what the C++ pipeline's BEV
compositor does (which starts from a GROUND POINT and asks "which pixel
shows it?"). Un-doing one direction with the other is exactly what makes
this an end-to-end test: if the renderer and the rig geometry in
kernels.cuh agree (they are computed from the SAME physical description,
independently, in two languages — see build_camera_matrix()'s docstring),
a ground point painted here should reappear, geometrically correctly, in
the BEV output main.cu produces.

The scene (parking-lot style; fully specified here so this file IS the
scene's spec — main.cu's "MUST MATCH ../scripts/make_synthetic.py" block
cross-references every number that matters to a gate):

    Asphalt        — dark gray (58, +-14 via cell-based value noise,
                     0.4 m cells) covering the whole ground plane by default.
    Lane lines     — 2 dashed white lines at Y = +-2.2 m, 15 cm wide,
                     0.6 m on / 0.4 m off along X.
    Boundary edge  — a straight, WORLD-STRAIGHT edge at Y = -0.4 m
                     (X in [2.0, 6.5]) separating a flat light "loading
                     zone" (Y in [-0.4, 2.1]) from the dark asphalt — the
                     feature main.cu's straightness_rectilinear /
                     distortion_negative_control gates measure.
    3 tall objects — 2 cylinders + 1 box (colors below), each standing on
                     the ground, causing the flat_ground_assumption gate's
                     deliberate BEV ghosting.

Usage:
    python make_synthetic.py                  # the committed scene (fixed, seed 42)
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fisheye camera model — MUST MATCH ../src/kernels.cuh's PART 1 constants
# exactly (that file is the single source of truth for the C++ side; this
# comment is the cross-reference CLAUDE.md §12 asks every duplicated
# constant to carry). Python computes theta/phi in native double precision
# (no hand-rounded trig constants needed here — this script is not compiled
# into two translation units that must link against a bit-identical value,
# so the ~1e-9 double-vs-float difference from kernels.cuh's rounded
# float32 literals is many orders of magnitude below anything that shows up
# at pixel resolution — same asymmetric-precision discipline 01.01 uses).
# ---------------------------------------------------------------------------
FISH_W, FISH_H = 320, 240          # must match kFishW, kFishH
FISH_CX = (FISH_W - 1) * 0.5       # must match kFishCx = 159.5
FISH_CY = (FISH_H - 1) * 0.5       # must match kFishCy = 119.5
FISH_FX = 74.0                     # must match kFishFx
VALID_HALF_FOV_RAD = math.radians(92.5)   # must match kFishValidHalfFovRad (92.5 deg)

SEED = 42                          # fixed seed, per CLAUDE.md/task requirement
SUBSAMPLE_OFFSETS = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]  # 2x2 AA grid, in-pixel offsets

# ---------------------------------------------------------------------------
# Rig — MUST MATCH ../src/kernels.cuh's PART 3 kRigCameras[] (the mounts
# and the nominal-basis + 45-deg-tilt construction; that file's comment
# derives the general "x_cam=x0, y_cam=cos(t)*y0-sin(t)*z0, z_cam=
# sin(t)*y0+cos(t)*z0" formula from first principles — this function
# applies the SAME formula, independently, in Python double precision, so
# two languages agree the hardcoded C++ floats are not a transcription
# slip (mirrors 01.01's 3-way camera-model cross-check).
# ---------------------------------------------------------------------------
RIG_TILT_RAD = math.radians(45.0)   # must match kRigTiltDeg

# Vehicle-frame nominal (untilted) bases per facing direction: (x0, y0, z0),
# each a unit vector in (bx, by, bz) — see kernels.cuh's PART 3 header ASCII
# diagram for the physical picture.
NOMINAL_BASES = {
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    "rear":  ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
    "left":  ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}
# Mount positions, vehicle frame, meters — must match kRigCameras[].mount.
MOUNTS = {
    "front": (2.0, 0.0, 0.6),
    "left":  (0.0, 1.0, 1.1),
    "right": (0.0, -1.0, 1.1),
    "rear":  (-2.0, 0.0, 0.6),
}
CAMERA_ORDER = ["front", "left", "right", "rear"]   # must match kCamFront..kCamRear bit order


def build_camera_matrix(face: str, tilt_rad: float):
    """Return R_cam_vehicle as 3 ROWS (x_cam, y_cam, z_cam), each a 3-tuple
    of (bx,by,bz) components — the SAME matrix kernels.cuh hardcodes for
    this camera, computed here from the general tilt formula instead of
    copied. x_cam is the tilt-rotation axis (unchanged); y_cam and z_cam
    rotate together by tilt_rad (kernels.cuh's PART 3 header derives this)."""
    x0, y0, z0 = NOMINAL_BASES[face]
    c, s = math.cos(tilt_rad), math.sin(tilt_rad)
    x_cam = x0
    y_cam = tuple(c * y0[i] - s * z0[i] for i in range(3))
    z_cam = tuple(s * y0[i] + c * z0[i] for i in range(3))
    return [list(x_cam), list(y_cam), list(z_cam)]


def cam_dir_to_vehicle(M, d_cam):
    """Camera-frame ray direction -> vehicle-frame direction: d_vehicle =
    M^T * d_cam (M's ROWS are the camera axes in vehicle components, so
    this is the transpose multiply — kernels.cuh's CameraExtrinsic comment
    explains why the C++ side stores it the OTHER way around (no transpose
    needed there, because it goes vehicle -> camera, the opposite
    direction); this script needs camera -> vehicle for ray casting, so it
    transposes explicitly here.)"""
    return [sum(M[k][i] * d_cam[k] for k in range(3)) for i in range(3)]


def fisheye_unproject(u: float, v: float):
    """Fisheye pixel (u,v) -> unit camera-frame ray direction, equidistant
    model — independent Python re-typing of kernels.cuh's
    fisheye_unproject() (same formula, double precision). Returns None if
    the pixel is outside the design half-FOV (the lens's illuminated
    circle — see kernels.cuh's kFishFx comment on vignetting)."""
    du, dv = u - FISH_CX, v - FISH_CY
    r = math.hypot(du, dv)
    theta = r / FISH_FX
    if theta > VALID_HALF_FOV_RAD:
        return None   # outside the illuminated circle -> vignette (rendered black)
    phi = math.atan2(dv, du)
    s = math.sin(theta)
    return (s * math.cos(phi), s * math.sin(phi), math.cos(theta))


# ===========================================================================
# Ground texture — hash-based VALUE NOISE for the asphalt mottling. Uses the
# xorshift32 bit-mixing rule (x^=x<<13; x^=x>>17; x^=x<<5) as a HASH
# finalizer over a per-cell integer key, not as a sequential PRNG stream —
# a standard, deterministic technique: the same (ix, iy, seed) ALWAYS
# hashes to the same value, with no state to carry between calls, which is
# exactly what a spatial texture function needs (call it for any pixel, in
# any order, and get the same answer — unlike a stream generator, which
# would need to be called in a fixed sequence to be reproducible).
# ===========================================================================

def xorshift32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def cell_hash01(ix: int, iy: int, seed: int) -> float:
    """Deterministic pseudo-random value in [0,1) for integer cell (ix,iy),
    mixed through two xorshift32 rounds so nearby cells (which differ by a
    small amount in the seed key below) still land far apart in hash
    space — avoiding the "obviously periodic" look CLAUDE.md's self-
    similarity lesson (project 01.04) warns against."""
    x = ((ix * 0x1F1F1F1F) ^ (iy * 0x2C1B3C6D) ^ seed) & 0xFFFFFFFF
    x = xorshift32(x)
    x = xorshift32(x)
    return x / 4294967295.0


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


NOISE_CELL_M = 0.4       # value-noise lattice spacing, meters
NOISE_AMPLITUDE = 14.0   # +- gray levels added to the base asphalt color


def value_noise01(x: float, y: float, seed: int) -> float:
    """Bilinearly-interpolated (smoothstepped) value noise in [0,1] at
    continuous ground coordinate (x,y) meters, lattice spacing
    NOISE_CELL_M — the classic "value noise" technique: hash the 4
    surrounding lattice corners, smoothstep-interpolate between them, so
    neighboring ground points get gradually-varying, non-repeating mottling
    instead of a flat color (which would look fake/uniform) or raw
    per-pixel noise (which would look like static, not asphalt)."""
    fx, fy = x / NOISE_CELL_M, y / NOISE_CELL_M
    ix0, iy0 = math.floor(fx), math.floor(fy)
    tx, ty = fx - ix0, fy - iy0
    h00 = cell_hash01(ix0, iy0, seed)
    h10 = cell_hash01(ix0 + 1, iy0, seed)
    h01 = cell_hash01(ix0, iy0 + 1, seed)
    h11 = cell_hash01(ix0 + 1, iy0 + 1, seed)
    sx, sy = smoothstep(tx), smoothstep(ty)
    top = h00 + (h10 - h00) * sx
    bot = h01 + (h11 - h01) * sx
    return top + (bot - top) * sy


# ---------------------------------------------------------------------------
# Scene layout — MUST MATCH ../src/main.cu's "Scene-layout constants" block
# (the boundary edge and the 3 objects; the lane lines are cosmetic and not
# read by any gate, so they carry no C++ mirror).
# ---------------------------------------------------------------------------
ASPHALT_GRAY = 58.0
LANE_YS = (2.2, -2.2)
LANE_HALF_WIDTH_M = 0.075
LANE_DASH_PERIOD_M = 1.0
LANE_DASH_ON_FRAC = 0.6
LANE_COLOR = (225, 225, 225)

BOUNDARY_Y = -0.4               # must match main.cu's kBoundaryY
BOUNDARY_X0, BOUNDARY_X1 = 2.0, 6.5   # must match kBoundaryX0/X1
LIGHT_ZONE_WIDTH_M = 2.5        # light zone spans Y in [BOUNDARY_Y, BOUNDARY_Y + this]
LIGHT_ZONE_COLOR = (195, 190, 178)

# (shape, cx, cy, extra..., height, color) — must match main.cu's kSceneObjects positions/radii.
OBJECTS = [
    {"shape": "cylinder", "cx": 3.0, "cy": 1.2, "r": 0.30, "h": 1.0, "color": (200, 40, 40)},
    {"shape": "cylinder", "cx": -2.5, "cy": -1.0, "r": 0.30, "h": 0.8, "color": (40, 70, 200)},
    {"shape": "box", "cx": 0.3, "cy": 1.8, "hx": 0.25, "hy": 0.4, "h": 1.2, "color": (210, 190, 30)},
]

BEV_W, BEV_H = 320, 320         # must match kBevW, kBevH
BEV_RANGE_M = 4.0               # must match kBevRangeM


def clampi(v: float) -> int:
    return 0 if v < 0 else (255 if v > 255 else int(v))


def ground_color(x: float, y: float):
    """The TRUE ground-plane color at world point (x,y), Z=0. Painter's-
    algorithm composite in priority order: boundary light zone, then lane
    lines, then base asphalt+noise — chosen because the light zone and the
    lane lines never geometrically overlap in this scene (by construction:
    the light zone's Y range [-0.4,2.1] excludes the lane Ys +-2.2), so the
    order does not matter for correctness, only readability."""
    if BOUNDARY_X0 <= x <= BOUNDARY_X1 and BOUNDARY_Y <= y <= BOUNDARY_Y + LIGHT_ZONE_WIDTH_M:
        return LIGHT_ZONE_COLOR
    for ly in LANE_YS:
        if abs(y - ly) <= LANE_HALF_WIDTH_M:
            phase = x % LANE_DASH_PERIOD_M   # Python's % is non-negative for a positive divisor, even for negative x
            if phase < LANE_DASH_PERIOD_M * LANE_DASH_ON_FRAC:
                return LANE_COLOR
    n = value_noise01(x, y, SEED)                      # in [0,1]
    g = clampi(ASPHALT_GRAY + (n - 0.5) * 2.0 * NOISE_AMPLITUDE)
    return (g, g, g)


def shade_color(color, factor: float):
    return tuple(clampi(c * factor) for c in color)


# ===========================================================================
# Ray-object intersection — plain analytic geometry, one function per
# primitive shape (cylinder, box), each returning the nearest positive hit
# distance (and enough info to shade it) or None.
# ===========================================================================

def intersect_ground(O, D):
    """Ray-plane intersection with Z=0 (the ground). Only a ray pointing
    DOWNWARD (D[2] < 0) can hit it, since every camera mount sits above
    the ground (O[2] > 0) — see kernels.cuh's PART 3 header. Returns the
    hit distance t, or None."""
    if D[2] >= -1e-9:
        return None
    t = -O[2] / D[2]
    return t if t > 1e-6 else None


def intersect_cylinder(O, D, obj):
    """Ray-vs-vertical-cylinder (infinite side surface capped at Z in
    [0,h], radius r, footprint center (cx,cy)) — the classic quadratic-in-t
    side-surface test, plus a flat top-cap disk test; returns (t, is_top)
    for whichever is nearer, or None if the ray misses both."""
    cx, cy, r, h = obj["cx"], obj["cy"], obj["r"], obj["h"]
    ox, oy = O[0] - cx, O[1] - cy
    dx, dy, dz = D
    best_t, best_top = None, False

    a = dx * dx + dy * dy
    if a > 1e-12:
        b = 2.0 * (dx * ox + dy * oy)
        c = ox * ox + oy * oy - r * r
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sq = math.sqrt(disc)
            t_lo, t_hi = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)   # t_lo <= t_hi (nearer root first)
            for t in (t_lo, t_hi):
                if t > 1e-6:
                    z = O[2] + t * dz
                    if 0.0 <= z <= h:
                        best_t, best_top = t, False
                        break   # nearer valid root wins; no need to check the farther one

    if abs(dz) > 1e-12:
        t_top = (h - O[2]) / dz
        if t_top > 1e-6:
            px, py = O[0] + t_top * dx - cx, O[1] + t_top * dy - cy
            if px * px + py * py <= r * r and (best_t is None or t_top < best_t):
                best_t, best_top = t_top, True

    return None if best_t is None else (best_t, best_top)


def intersect_box(O, D, obj):
    """Ray-vs-axis-aligned-box (the classic "slab method": intersect the
    ray's parametric interval against each axis's [lo,hi] slab, then
    intersect all three intervals). Returns (t, face) where face names
    which axis the entry point lies on (for simple flat shading), or None."""
    cx, cy, hx, hy, h = obj["cx"], obj["cy"], obj["hx"], obj["hy"], obj["h"]
    lo = (cx - hx, cy - hy, 0.0)
    hi = (cx + hx, cy + hy, h)
    tmin, tmax = -math.inf, math.inf
    face_axis = -1

    for i in range(3):
        o, d = O[i], D[i]
        if abs(d) < 1e-12:
            if o < lo[i] or o > hi[i]:
                return None
            continue
        t1, t2 = (lo[i] - o) / d, (hi[i] - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin, face_axis = t1, i
        if t2 < tmax:
            tmax = t2

    if tmin > tmax or tmax <= 1e-6:
        return None
    t = tmin if tmin > 1e-6 else tmax
    face = "top" if face_axis == 2 else ("x" if face_axis == 0 else "y")
    return t, face


def trace_ray(O, D):
    """The renderer's core: cast one ray from O in direction D, find the
    NEAREST hit among the ground plane and the 3 objects, and return its
    shaded color. If nothing is hit (the ray points at empty sky), returns
    a simple sky gradient. This is the ONE place "what does this ray see?"
    is answered — every fisheye pixel's 4 AA subsamples call this once
    each (see render_camera below)."""
    best_t = math.inf
    best_color = None

    t_ground = intersect_ground(O, D)
    if t_ground is not None and t_ground < best_t:
        best_t = t_ground
        gx, gy = O[0] + t_ground * D[0], O[1] + t_ground * D[1]
        best_color = ground_color(gx, gy)

    for obj in OBJECTS:
        if obj["shape"] == "cylinder":
            hit = intersect_cylinder(O, D, obj)
            if hit is not None:
                t, is_top = hit
                if t < best_t:
                    best_t = t
                    best_color = shade_color(obj["color"], 1.15 if is_top else 0.9)
        else:
            hit = intersect_box(O, D, obj)
            if hit is not None:
                t, face = hit
                if t < best_t:
                    best_t = t
                    best_color = shade_color(obj["color"], 1.15 if face == "top" else 0.85)

    if best_color is not None:
        return best_color

    # Sky: a simple two-tone gradient by the ray's own "up" (Z) component —
    # cosmetic only (no gate reads a sky pixel), included so the vignette-
    # free part of the image outside the ground/object footprint is not an
    # arbitrary flat color.
    up = max(0.0, min(1.0, D[2]))
    horizon, zenith = (180, 195, 205), (120, 150, 190)
    return tuple(int(horizon[i] + (zenith[i] - horizon[i]) * up) for i in range(3))


def render_camera(face: str) -> bytes:
    """Ray-cast the full FISH_W x FISH_H fisheye image for rig camera
    `face`, 2x2 supersampled. Pixels outside the design half-FOV are
    rendered BLACK (the vignette — kernels.cuh's kFishFx comment: a real
    fisheye lens does not illuminate past its design circle, so this is
    physics, not a rendering shortcut)."""
    M = build_camera_matrix(face, RIG_TILT_RAD)
    O = MOUNTS[face]
    img = bytearray(FISH_W * FISH_H * 3)

    for py in range(FISH_H):
        row = py * FISH_W
        for px in range(FISH_W):
            acc = [0.0, 0.0, 0.0]
            for (ox, oy) in SUBSAMPLE_OFFSETS:
                d_cam = fisheye_unproject(px + ox, py + oy)
                if d_cam is None:
                    color = (0, 0, 0)   # vignette
                else:
                    d_veh = cam_dir_to_vehicle(M, d_cam)
                    color = trace_ray(O, d_veh)
                acc[0] += color[0]; acc[1] += color[1]; acc[2] += color[2]
            n = float(len(SUBSAMPLE_OFFSETS))
            o = (row + px) * 3
            img[o + 0] = clampi(acc[0] / n + 0.5)
            img[o + 1] = clampi(acc[1] / n + 0.5)
            img[o + 2] = clampi(acc[2] / n + 0.5)
    return bytes(img)


def bev_pixel_to_ground(xo: int, yo: int):
    """BEV output pixel -> ground point, MUST MATCH kernels.cuh's
    bev_pixel_to_ground() exactly (row 0 = farthest forward, col 0 =
    farthest left — see that function's comment)."""
    y = BEV_RANGE_M - xo * (2.0 * BEV_RANGE_M) / (BEV_W - 1)
    x = BEV_RANGE_M - yo * (2.0 * BEV_RANGE_M) / (BEV_H - 1)
    return x, y


def render_bev_ground_truth() -> bytes:
    """Direct, camera-free, object-free orthographic render of the ground
    TEXTURE at every BEV pixel — no ray casting needed (the answer is just
    ground_color(X,Y) evaluated at the pixel's ground point). This is the
    EXACT answer a perfect flat-ground BEV reconstruction should produce
    (main.cu's file header: "the BEV ground truth IS the source texture")."""
    img = bytearray(BEV_W * BEV_H * 3)
    for yo in range(BEV_H):
        row = yo * BEV_W
        for xo in range(BEV_W):
            x, y = bev_pixel_to_ground(xo, yo)
            r, g, b = ground_color(x, y)
            o = (row + xo) * 3
            img[o + 0], img[o + 1], img[o + 2] = r, g, b
    return bytes(img)


def write_ppm(path: Path, width: int, height: int, data: bytes) -> None:
    """8-bit binary PPM (P6) — three interleaved RGB bytes per pixel,
    identical convention to every other PPM in this repository (e.g.
    01.01's write_ppm) — no library needed to read or write it."""
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(data)


def write_rig_csv(path: Path) -> None:
    """rig_extrinsics.csv — documentation/provenance mirror of the rig this
    script (and, independently, kernels.cuh) constructs: one row per
    camera, mount position + the 9 R_cam_vehicle matrix entries (row-major,
    same layout as kernels.cuh's CameraExtrinsic::m). Not read by the C++
    program (its rig constants are compiled in — see kernels.cuh's PART 3
    header); this file exists so a reader can inspect the rig numbers
    without cross-referencing two languages by eye."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "mount_x_m", "mount_y_m", "mount_z_m", "tilt_deg",
                   "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22"])
        for face in CAMERA_ORDER:
            M = build_camera_matrix(face, RIG_TILT_RAD)
            mount = MOUNTS[face]
            flat = [v for row in M for v in row]
            w.writerow([face, *[f"{v:.6f}" for v in mount], f"{math.degrees(RIG_TILT_RAD):.1f}",
                       *[f"{v:.8f}" for v in flat]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default ../data/sample)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for face in CAMERA_ORDER:
        print(f"rendering fisheye_{face}.ppm ({FISH_W}x{FISH_H}, 2x2 supersampled, ray-cast) ...")
        img = render_camera(face)
        out_path = args.out_dir / f"fisheye_{face}.ppm"
        write_ppm(out_path, FISH_W, FISH_H, img)
        written.append(out_path)

    print(f"rendering bev_ground_truth.ppm ({BEV_W}x{BEV_H}, direct texture evaluation) ...")
    bev_truth = render_bev_ground_truth()
    bev_path = args.out_dir / "bev_ground_truth.ppm"
    write_ppm(bev_path, BEV_W, BEV_H, bev_truth)
    written.append(bev_path)

    csv_path = args.out_dir / "rig_extrinsics.csv"
    write_rig_csv(csv_path)
    written.append(csv_path)

    total_bytes = sum(p.stat().st_size for p in written)
    print(f"wrote {args.out_dir} : {len(written)} files, {total_bytes} bytes total - labeled SYNTHETIC (seed {SEED})")
    for p in written:
        print(f"  {p.name}: {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
