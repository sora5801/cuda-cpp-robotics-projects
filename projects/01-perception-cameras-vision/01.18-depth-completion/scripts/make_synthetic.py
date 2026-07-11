#!/usr/bin/env python3
"""make_synthetic.py -- synthetic sample-data generator for 01.18 (Depth completion:
sparse LiDAR + RGB -> dense depth).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Depth completion needs EXACT dense ground truth to grade against -- something no real
sensor can hand you (a real LiDAR is exactly the sparse signal this project densifies).
Synthesis is therefore not just convenient here, it is the ONLY way to get an honest
answer key. This script ray-casts a small synthetic scene (technique follows project
01.07's ray-cast renderer, cited throughout, reimplemented independently for this
project's own geometry) from TWO different sensor origins -- a pinhole camera and a
spinning multi-beam LiDAR mounted nearby -- producing:

  rgb.ppm          the camera's RGB rendering (2x2 supersampled for anti-aliasing)
  truth_depth.bin  EXACT dense camera-frame depth (Pcam.z), one float32 per pixel,
                   raw row-major binary, kInvalidDepth (-1.0) where the ray hit sky
  lidar_points.csv the LiDAR's raw returns (x,y,z, LIDAR-frame meters), range-noised

All three are consumed by the C++ pipeline: rgb.ppm is the guidance image,
lidar_points.csv is what project_zbuffer_kernel/_cpu project into the camera, and
truth_depth.bin is ONLY used by main.cu's evaluation gates -- never by the algorithm
itself (that would be cheating: the whole point is recovering depth WITHOUT it).

Geometry & camera constants below are a hand-kept mirror of src/kernels.cuh (image
size, intrinsics, the LiDAR extrinsic) -- there is deliberately no shared source
between Python and CUDA C++ in this repo, so a comment here and in kernels.cuh's file
header both flag this as a manually-synchronized contract; change one, change both.

Determinism: this script uses ONLY a from-scratch xorshift32 generator (the same
bit-mixing rule 01.07's ground-texture hash uses, cited) for every random choice
(ground texture noise, LiDAR range noise) -- never Python's `random` module -- so the
output is bit-identical across machines and matches this repo's C++-side convention
of avoiding library-specific RNG implementations (CLAUDE.md machine-facts brief).

Usage
-----
    python make_synthetic.py                      # defaults: seed=42, writes into ../data/sample/
    python make_synthetic.py --seed 42 --az-count 70
"""

import argparse
import csv
import math
import struct
from pathlib import Path

# ===========================================================================
# Camera / image constants -- MUST match src/kernels.cuh exactly (manually
# synchronized, see module docstring).
# ===========================================================================
IMG_W, IMG_H = 160, 120
FX, FY, CX, CY = 154.0, 152.0, 80.0, 60.0
MAX_DEPTH_M = 20.0
INVALID_DEPTH = -1.0

# Camera world (body-frame: x-forward, y-left, z-up) pose: mounted at 1.5 m,
# looking dead level along +x (no tilt) -- see THEORY.md "The math" for why
# zero tilt still lets the ground plane appear in the lower half of the image
# (any camera-frame ray with a downward y-component maps to v > cy).
CAM_POS = (0.0, 0.0, 1.5)

# LiDAR world (body-frame) pose: 0.30 m above and 0.05 m behind the camera --
# an illustrative roof-LiDAR-over-windshield-camera rig (PRACTICE.md SS1).
# This position, together with CAM_POS and the pure axis-permutation rotation
# below, is EXACTLY what produces kTCameraLidar in kernels.cuh (t = R *
# (LIDAR_POS - CAM_POS); see that header's derivation comment).
LIDAR_POS = (-0.05, 0.0, 1.80)

# R_camera_body / R_camera_lidar: converts a body-or-lidar-frame vector to the
# camera's OPTICAL frame (z-forward, x-right, y-down). A pure axis permutation
# (both LiDAR and body share the same x-forward/y-left/z-up convention, so one
# rotation serves both conversions): x_cam = -y_body; y_cam = -z_body; z_cam = x_body.
def body_to_cam(v):
    x, y, z = v
    return (-y, -z, x)


def cam_to_body(v):
    # The inverse permutation (this rotation is its own transpose-applied-once
    # relation since it is a signed permutation matrix; written out directly
    # rather than "cleverly" inverting body_to_cam, for readability):
    #   x_body = z_cam ; y_body = -x_cam ; z_body = -y_cam
    xc, yc, zc = v
    return (zc, -xc, -yc)


# LiDAR beam model: 16 elevation beams spanning +-15 deg (a VLP-16-like
# spinning multi-beam LiDAR, THEORY.md derives the elevation/azimuth scan
# pattern this produces on the image plane). Azimuth sweeps a wedge around
# forward, wider than the camera's own ~56.5 deg horizontal FOV so LiDAR
# returns cover the image right to its edges.
N_BEAMS = 16
ELEV_MIN_DEG, ELEV_MAX_DEG = -15.0, 15.0
AZ_HALF_RANGE_DEG = 32.0          # sweep +-32 deg (camera HFOV is ~56.5 deg, i.e. +-28.25 deg)
LIDAR_RANGE_NOISE_SIGMA_M = 0.02  # 1-sigma range noise, meters (illustrative MEMS/ToF-class LiDAR)

DEFAULT_SEED = 42


# ===========================================================================
# xorshift32 -- the repo's portable deterministic PRNG (same bit-mixing rule
# 01.07's ground-texture hash uses, cited in the module docstring). Operates
# on Python ints masked to 32 bits so results match the C++ uint32_t version
# bit-for-bit on any platform.
# ===========================================================================
MASK32 = 0xFFFFFFFF


def xorshift32(x: int) -> int:
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5) & MASK32
    return x & MASK32


def cell_hash01(ix: int, iy: int, seed: int) -> float:
    """Deterministic hash of an integer grid cell to a float in [0,1), via two
    xorshift32 rounds mixing (ix, iy, seed) -- the same "hash the hash" trick
    01.07 uses so nearby cells land far apart in the output despite nearby
    inputs (cited)."""
    x = (ix * 1103515245 + iy * 12345 + seed * 2654435761) & MASK32
    x = xorshift32(x)
    x = xorshift32(x)
    return (x & 0xFFFFFF) / float(0x1000000)  # top 24 bits -> [0,1)


def uniform01(state: int):
    """One xorshift32 draw -> (new_state, float in (0,1])."""
    state = xorshift32(state)
    v = ((state >> 8) & 0xFFFFFF) / float(0x1000000) + (0.5 / 0x1000000)
    return state, v


def gaussian(state: int, sigma: float):
    """One N(0, sigma^2) draw via Box-Muller, seeded by the running xorshift32
    state (the SAME construction main.cu's noise generator would use in C++ --
    see 08.01's main.cu gaussian() for the precedent this mirrors)."""
    state, u1 = uniform01(state)
    state, u2 = uniform01(state)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return state, sigma * z


# ===========================================================================
# Scene -- ground plane (z=0, hashed asphalt-like texture) plus five objects
# at varied depths (THEORY.md/README name each object's didactic role):
#
#   near_box / mid_box / far_box  -- ordinary Lambertian-shaded boxes, plain
#     "does depth completion work at all" test subjects at three depths.
#   pole                          -- a vertical cylinder, tests a CURVED
#     depth edge (not just axis-aligned box silhouettes).
#   trap_board                    -- the TEXTURE TRAP: a flat, unlit (no
#     shading) board painted with a high-contrast checkerboard. Every pixel
#     on its face is the SAME depth despite huge RGB gradients -- exactly
#     the case where "RGB edge implies depth edge" is a FALSE prior.
#   camo_near / camo_far          -- the CAMO EDGE pair: two flat, unlit,
#     IDENTICALLY colored boxes at different depths. Their shared silhouette
#     is a REAL depth discontinuity with near-zero RGB contrast -- the case
#     where the prior FAILS to fire and diffusion should smear.
#
# "flat_unlit" objects skip Lambertian shading entirely (return their base
# color verbatim) specifically so their RGB contrast is controlled ONLY by
# the paint, never contaminated by a shading gradient across the surface.
# ===========================================================================
GROUND_BASE_GRAY = 92.0
GROUND_NOISE_AMPLITUDE = 14.0
GROUND_NOISE_CELL_M = 0.35

OBJECTS = [
    {"name": "near_box", "type": "box", "center": (4.0, -1.0, 0.4), "half": (0.4, 0.4, 0.4),
     "color": (185, 60, 50), "flat_unlit": False},
    {"name": "mid_box", "type": "box", "center": (8.0, 1.2, 0.6), "half": (0.5, 0.5, 0.6),
     "color": (55, 95, 190), "flat_unlit": False},
    {"name": "far_box", "type": "box", "center": (14.0, -0.5, 0.8), "half": (0.6, 0.6, 0.8),
     "color": (70, 170, 90), "flat_unlit": False},
    {"name": "pole", "type": "cylinder", "center_xy": (10.0, -3.6), "radius": 0.25,
     "z0": 0.0, "z1": 2.5, "color": (205, 150, 40), "flat_unlit": False},
    {"name": "trap_board", "type": "box", "center": (5.98, 1.7, 1.0), "half": (0.02, 0.35, 0.35),
     "color": None, "flat_unlit": True, "checker": True, "checker_cell_m": 0.10,
     "checker_colors": ((235, 235, 235), (18, 18, 18))},
    {"name": "camo_near", "type": "box", "center": (5.0, -1.6, 0.5), "half": (0.5, 0.5, 0.5),
     "color": (130, 130, 130), "flat_unlit": True},
    {"name": "camo_far", "type": "box", "center": (9.0, -1.6, 1.0), "half": (2.0, 1.5, 1.0),
     "color": (130, 130, 130), "flat_unlit": True},
]

LIGHT_DIR = (0.35, 0.30, 0.887)  # pre-normalized-ish directional light, body frame (points TOWARD the light)
_ll = math.sqrt(sum(c * c for c in LIGHT_DIR))
LIGHT_DIR = tuple(c / _ll for c in LIGHT_DIR)
AMBIENT = 0.38
DIFFUSE = 0.62


def ground_color(x: float, y: float):
    """Hashed value-noise asphalt mottling (technique cited from 01.07,
    reimplemented independently): bilinear-blend four hashed grid-corner
    values for a smooth-but-textured base gray."""
    gx, gy = x / GROUND_NOISE_CELL_M, y / GROUND_NOISE_CELL_M
    ix0, iy0 = math.floor(gx), math.floor(gy)
    tx, ty = gx - ix0, gy - iy0
    # smoothstep easing so cell boundaries do not show as visible creases
    sx = tx * tx * (3 - 2 * tx)
    sy = ty * ty * (3 - 2 * ty)
    h00 = cell_hash01(ix0, iy0, 1001)
    h10 = cell_hash01(ix0 + 1, iy0, 1001)
    h01 = cell_hash01(ix0, iy0 + 1, 1001)
    h11 = cell_hash01(ix0 + 1, iy0 + 1, 1001)
    h0 = h00 * (1 - sx) + h10 * sx
    h1 = h01 * (1 - sx) + h11 * sx
    h = h0 * (1 - sy) + h1 * sy
    g = GROUND_BASE_GRAY + (h * 2.0 - 1.0) * GROUND_NOISE_AMPLITUDE
    g = max(0.0, min(255.0, g))
    return (g, g, g)


def shade(color, normal):
    """Lambertian shade: ambient + diffuse * max(0, N.L). `normal` must be a
    unit vector in body frame; `color` is the base (r,g,b) in [0,255]."""
    ndotl = max(0.0, normal[0] * LIGHT_DIR[0] + normal[1] * LIGHT_DIR[1] + normal[2] * LIGHT_DIR[2])
    factor = AMBIENT + DIFFUSE * ndotl
    return tuple(max(0.0, min(255.0, c * factor)) for c in color)


# ---------------------------------------------------------------------------
# Ray/object intersection -- the standard closed-form tests (plane, AABB
# slab test, capped-cylinder quadratic), the same ALGORITHM FAMILY 01.07's
# renderer uses (cited), reimplemented here independently for this project's
# own scene and return convention: every intersect_* returns either None (no
# hit) or (s, normal, base_color) where `s` is the ray parameter along the
# CALLER-SUPPLIED direction D (P_hit = O + s*D) -- callers choose whether D
# is unit-length (LiDAR: s becomes true Euclidean range) or has an implicit
# z_cam=1 (camera: s becomes Pcam.z depth directly, see module docstring).
# ---------------------------------------------------------------------------
def intersect_ground(O, D):
    if D[2] >= -1e-9:
        return None  # ray level or moving up -> never reaches the z=0 ground plane
    s = -O[2] / D[2]
    if s <= 1e-6:
        return None
    hit = (O[0] + s * D[0], O[1] + s * D[1], 0.0)
    return s, (0.0, 0.0, 1.0), ground_color(hit[0], hit[1])


def intersect_box(O, D, obj):
    cx, cy, cz = obj["center"]
    hx, hy, hz = obj["half"]
    lo = (cx - hx, cy - hy, cz - hz)
    hi = (cx + hx, cy + hy, cz + hz)
    tmin, tmax = -1e18, 1e18
    hit_axis_min = -1
    for ax in range(3):
        o, d = O[ax], D[ax]
        if abs(d) < 1e-12:
            if o < lo[ax] or o > hi[ax]:
                return None
            continue
        t0 = (lo[ax] - o) / d
        t1 = (hi[ax] - o) / d
        sign = 1.0
        if t0 > t1:
            t0, t1 = t1, t0
            sign = -1.0
        if t0 > tmin:
            tmin = t0
            hit_axis_min = ax if sign > 0 else -(ax + 1)  # encode which face + which side
        if t1 < tmax:
            tmax = t1
    if tmin > tmax or tmax < 1e-6:
        return None
    s = tmin if tmin > 1e-6 else tmax
    # Face normal from which axis produced tmin (encoded sign above).
    axis = hit_axis_min if hit_axis_min >= 0 else -(hit_axis_min + 1)
    face_sign = 1.0 if hit_axis_min >= 0 else -1.0
    normal = [0.0, 0.0, 0.0]
    normal[axis] = -face_sign if D[axis] > 0 else face_sign
    # (normal points opposite the incoming ray's component along that axis)
    normal[axis] = -1.0 if D[axis] > 0 else 1.0
    color = obj["color"]
    if obj.get("checker"):
        # Checkerboard on the (y,z) plane of the board's local face -- the
        # TEXTURE TRAP: purely a function of hit position, independent of
        # depth (every pixel on this flat face has the SAME true depth).
        px = O[1] + s * D[1]
        pz = O[2] + s * D[2]
        cell = obj["checker_cell_m"]
        parity = (math.floor(px / cell) + math.floor(pz / cell)) % 2
        color = obj["checker_colors"][int(parity)]
    return s, tuple(normal), color


def intersect_cylinder(O, D, obj):
    cx, cy = obj["center_xy"]
    r = obj["radius"]
    ox, oy = O[0] - cx, O[1] - cy
    dx, dy = D[0], D[1]
    a = dx * dx + dy * dy
    best = None
    if a > 1e-12:
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - r * r
        disc = b * b - 4 * a * c
        if disc >= 0.0:
            sq = math.sqrt(disc)
            for s in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
                if s <= 1e-6:
                    continue
                z = O[2] + s * D[2]
                if obj["z0"] <= z <= obj["z1"]:
                    if best is None or s < best:
                        best = s
    # Top cap disk (z = z1): the camera/LiDAR sit below the pole's top on
    # this scene's geometry, but the cap is included for a robust, honest
    # renderer that doesn't silently mis-render off-nominal viewpoints.
    if abs(D[2]) > 1e-12:
        s_cap = (obj["z1"] - O[2]) / D[2]
        if s_cap > 1e-6 and (best is None or s_cap < best):
            hx, hy = O[0] + s_cap * D[0] - cx, O[1] + s_cap * D[1] - cy
            if hx * hx + hy * hy <= r * r:
                best = s_cap
    if best is None:
        return None
    z = O[2] + best * D[2]
    if abs(z - obj["z1"]) < 1e-4:
        normal = (0.0, 0.0, 1.0)
    else:
        nx, ny = O[0] + best * D[0] - cx, O[1] + best * D[1] - cy
        nlen = math.sqrt(nx * nx + ny * ny) or 1.0
        normal = (nx / nlen, ny / nlen, 0.0)
    return best, normal, obj["color"]


def trace_ray(O, D):
    """Cast one ray; return (color(r,g,b) floats 0..255, depth_or_None). depth
    is `s` from the nearest hit (see module docstring for what `s` means for
    the caller's choice of D); None means the ray hit nothing (sky)."""
    best_s = None
    best_color = None
    best_normal = None
    best_flat = False
    for obj in OBJECTS:
        if obj["type"] == "box":
            hit = intersect_box(O, D, obj)
        else:
            hit = intersect_cylinder(O, D, obj)
        if hit is None:
            continue
        s, normal, color = hit
        if best_s is None or s < best_s:
            best_s, best_color, best_normal, best_flat = s, color, normal, obj.get("flat_unlit", False)
    ghit = intersect_ground(O, D)
    if ghit is not None and (best_s is None or ghit[0] < best_s):
        best_s, best_normal, best_color = ghit[0], ghit[1], ghit[2]
        best_flat = False

    if best_s is None:
        # Sky: a simple two-tone vertical gradient by the ray's own "up"
        # (z, body frame) component -- purely decorative background.
        up = max(0.0, min(1.0, D[2] / (abs(D[2]) + 1.0) + 0.5))
        c = (150 + 60 * up, 175 + 55 * up, 225)
        return c, None

    color = best_color if best_flat else shade(best_color, best_normal)
    return color, best_s


def render(seed: int):
    """Ray-cast the full RGB image (2x2 supersampled) and the exact dense
    depth map (single center ray per pixel -- see module docstring for why
    depth must NOT be supersampled/blended)."""
    rgb = bytearray(IMG_W * IMG_H * 3)
    depth = [INVALID_DEPTH] * (IMG_W * IMG_H)

    for v in range(IMG_H):
        for u in range(IMG_W):
            # --- exact center-ray depth --------------------------------------
            xc = (u + 0.5 - CX) / FX
            yc = (v + 0.5 - CY) / FY
            d_cam = (xc, yc, 1.0)          # unnormalized: z_cam component == 1 -> s IS the depth
            d_body = cam_to_body(d_cam)
            _, s = trace_ray(CAM_POS, d_body)
            if s is not None and 0.0 < s <= MAX_DEPTH_M:
                depth[v * IMG_W + u] = s
            # else stays INVALID_DEPTH (sky, or beyond the LiDAR/scene's max range)

            # --- 2x2 supersampled color --------------------------------------
            r_acc = g_acc = b_acc = 0.0
            for sy in (0.25, 0.75):
                for sx in (0.25, 0.75):
                    xc2 = (u + sx - CX) / FX
                    yc2 = (v + sy - CY) / FY
                    d2 = cam_to_body((xc2, yc2, 1.0))
                    color, _ = trace_ray(CAM_POS, d2)
                    r_acc += color[0]; g_acc += color[1]; b_acc += color[2]
            idx = (v * IMG_W + u) * 3
            rgb[idx + 0] = int(max(0, min(255, r_acc / 4.0)))
            rgb[idx + 1] = int(max(0, min(255, g_acc / 4.0)))
            rgb[idx + 2] = int(max(0, min(255, b_acc / 4.0)))

    return rgb, depth


def generate_lidar_points(az_count: int, seed: int):
    """Ray-cast the spinning-LiDAR beam grid (N_BEAMS elevations x az_count
    azimuths spanning +-AZ_HALF_RANGE_DEG) from LIDAR_POS, independently of
    the camera rendering above -- a genuinely different sensor origin, so
    this is real multi-sensor parallax, not a reprojection of the camera's
    own depth map. Returns a list of (x,y,z) LIDAR-FRAME points (range-noised)
    and the raw hit count before any image-plane deduplication."""
    state = seed
    points = []
    elevs = [ELEV_MIN_DEG + i * (ELEV_MAX_DEG - ELEV_MIN_DEG) / (N_BEAMS - 1) for i in range(N_BEAMS)]
    azs = [-AZ_HALF_RANGE_DEG + i * (2 * AZ_HALF_RANGE_DEG) / (az_count - 1) for i in range(az_count)]

    for el_deg in elevs:
        el = math.radians(el_deg)
        for az_deg in azs:
            az = math.radians(az_deg)
            # Unit direction in the LiDAR's own (body-aligned) frame: az=0
            # along +x (forward), az>0 toward +y (left) -- standard math
            # convention right-handed about +z (up), documented in THEORY.md.
            dx = math.cos(el) * math.cos(az)
            dy = math.cos(el) * math.sin(az)
            dz = math.sin(el)
            _, s = trace_ray(LIDAR_POS, (dx, dy, dz))
            if s is None or s > MAX_DEPTH_M:
                continue
            state, noise = gaussian(state, LIDAR_RANGE_NOISE_SIGMA_M)
            r = max(0.01, s + noise)   # range noise; clamp away from exactly zero
            # LiDAR-frame point (frame is body-aligned, only translated from
            # body -- see kernels.cuh kTCameraLidar derivation): the point
            # measured RELATIVE TO THE LIDAR ORIGIN, in body-aligned axes.
            points.append((r * dx, r * dy, r * dz))

    return points


def write_ppm(path: Path, w: int, h: int, rgb: bytearray) -> None:
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(bytes(rgb))


def write_depth_bin(path: Path, depth) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(depth)}f", *depth))


# T_camera_lidar, mirrored from kernels.cuh's kTCameraLidar purely so this
# script can report an honest projected-density number to the console (the
# C++ pipeline is the one actual consumer of the projection at demo time).
T_CAM_LIDAR_R = (0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
T_CAM_LIDAR_T = (0.0, -0.30, -0.05)


def projected_pixel_density(points):
    """Project every LIDAR-frame point into the camera image with the SAME
    formula project_zbuffer_cpu uses (independently re-typed here -- this is
    a reporting utility, not part of the verified pipeline) and return the
    fraction of the 160x120 image that ends up with at least one point
    (after z-buffer dedup, i.e. the ACTUAL pixel density the C++ side will
    see) -- printed so README/data docs quote a measured number, never a
    guess."""
    covered = set()
    R, t = T_CAM_LIDAR_R, T_CAM_LIDAR_T
    for (x, y, z) in points:
        xc = R[0] * x + R[1] * y + R[2] * z + t[0]
        yc = R[3] * x + R[4] * y + R[5] * z + t[1]
        zc = R[6] * x + R[7] * y + R[8] * z + t[2]
        if zc <= 0.0 or zc > MAX_DEPTH_M:
            continue
        u = FX * xc / zc + CX
        v = FY * yc / zc + CY
        px, py = int(math.floor(u + 0.5)), int(math.floor(v + 0.5))
        if 0 <= px < IMG_W and 0 <= py < IMG_H:
            covered.add((px, py))
    return len(covered), len(covered) / float(IMG_W * IMG_H)


def write_lidar_csv(path: Path, points, seed: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC LiDAR returns for project 01.18 -- LIDAR-FRAME meters (x,y,z)\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write(f"# {len(points)} points, range noise sigma={LIDAR_RANGE_NOISE_SIGMA_M} m\n")
        w = csv.writer(f)
        w.writerow(["x", "y", "z"])
        for p in points:
            w.writerow([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--az-count", type=int, default=70,
                    help="azimuth samples per beam, spanning +-%.0f deg (default 70; tunes raw LiDAR density)" % AZ_HALF_RANGE_DEG)
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("[make_synthetic] ray-casting RGB + exact dense depth (160x120, 2x2 supersampled color)...")
    rgb, depth = render(args.seed)
    write_ppm(args.out / "rgb.ppm", IMG_W, IMG_H, rgb)
    write_depth_bin(args.out / "truth_depth.bin", depth)

    n_valid = sum(1 for d in depth if d != INVALID_DEPTH)
    print(f"[make_synthetic] wrote rgb.ppm + truth_depth.bin "
          f"({n_valid}/{len(depth)} = {100.0 * n_valid / len(depth):.1f}% pixels have a scene hit; rest is sky)")

    print(f"[make_synthetic] ray-casting LiDAR beam grid ({N_BEAMS} beams x {args.az_count} azimuths)...")
    points = generate_lidar_points(args.az_count, args.seed)
    write_lidar_csv(args.out / "lidar_points.csv", points, args.seed)
    print(f"[make_synthetic] wrote lidar_points.csv ({len(points)} raw beam returns, seed={args.seed}, labeled SYNTHETIC)")

    n_cov, density = projected_pixel_density(points)
    print(f"[make_synthetic] projected density (full set, z-buffer deduped): "
          f"{n_cov}/{IMG_W * IMG_H} pixels = {100.0 * density:.2f}%")
    for label, stride in (("~50% subsample (main demo default)", 2),
                          ("~20% subsample (density-sweep low end)", 5)):
        sub = points[::stride]
        n_sub, d_sub = projected_pixel_density(sub)
        print(f"[make_synthetic]   {label}: stride={stride}, {len(sub)} pts -> "
              f"{n_sub}/{IMG_W * IMG_H} = {100.0 * d_sub:.2f}%")

    print("[make_synthetic] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
