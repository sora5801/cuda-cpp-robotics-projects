#!/usr/bin/env python3
"""make_synthetic.py -- synthetic sample-data generator for 02.17 (LiDAR-camera
projection/coloring fusion kernels).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Grading a fusion pipeline honestly needs an answer key no real sensor pair can hand
you: for every LiDAR return, the EXACT true surface color it hit, AND whether that
3-D point is actually visible from the CAMERA's (different!) origin. Both of those
come for free from a ray-cast synthetic scene and from NOTHING else -- a real rig can
only ever measure "what pixel did I land on", never "was that the RIGHT pixel". This
script ray-casts a small scene (technique/lineage follows 01.18's renderer, cited
throughout, reimplemented independently for this project's own geometry -- the
occlusion-by-parallax story below is new) from TWO physically separate origins -- a
pinhole camera and a spinning multi-beam LiDAR mounted 0.30 m above and 0.05 m behind
it (the SAME rig geometry 01.18/02.02 use, so this project's kTCameraLidar is
numerically IDENTICAL to theirs -- cited, not re-derived) -- producing:

  rgb.ppm           the camera's RGB rendering (2x2 supersampled for anti-aliasing;
                     this is BOTH the fusion pipeline's color source AND, through its
                     own rendering, the reason occluded LiDAR points sample the WRONG
                     surface -- the image simply has no idea a LiDAR point exists)
  lidar_points.csv  the LiDAR's raw returns: x,y,z (LIDAR-frame meters) plus the
                     EVALUATION-ONLY ground truth columns true_r,true_g,true_b (the
                     exact surface color the LiDAR ray actually hit, [0,255]),
                     visible (1 if a SECOND ray cast from the CAMERA's origin toward
                     this exact 3-D point reaches it unobstructed, else 0 -- the
                     honest, geometry-first definition of "occluded from the
                     camera's point of view"), and surface (a human-readable label).

The C++ pipeline (src/) reads x,y,z as its ONLY input (the same LidarPointF shape
01.18/02.02 use) and rgb.ppm as its ONLY color source; true_r/g/b/visible/surface
never touch the GPU or CPU projection/coloring code paths -- they exist ONLY so
main.cu's evaluation gates can grade the pipeline's output against ground truth the
pipeline itself never saw (CLAUDE.md's "independent gate" discipline, 01.18 section).

THE OCCLUSION STORY THIS SCENE IS BUILT TO TELL (THEORY.md derives the geometry)
---------------------------------------------------------------------------------
A red occluder box (top at z=1.6 m) sits 4 m from the rig; a green background wall
sits 12 m out. Because the LiDAR (z=1.80 m) sits HIGHER than the camera (z=1.50 m),
simple similar-triangles geometry (worked below and in THEORY.md) shows the LiDAR's
line of sight clears the occluder's top edge for any background point above z=1.2 m,
while the CAMERA's line of sight only clears it above z=1.8 m. Background points with
z in (1.2, 1.8) m are therefore genuinely, physically visible to the LiDAR but hidden
from the camera behind the occluder -- real parallax occlusion from the sensor
baseline, not a synthetic label. Naively coloring those points with "whatever pixel
they project to" paints them the OCCLUDER's red instead of their own true green --
exactly the failure src/main.cu's occlusion_correctness gate measures and the
z-buffer visibility pass (kernels.cu) is built to catch.

Determinism: this script uses ONLY a from-scratch xorshift32 generator (same
bit-mixing rule 01.07/01.18 use, cited) for the one random choice it makes (LiDAR
range noise) -- never Python's `random` module.

Usage
-----
    python make_synthetic.py                      # defaults: seed=42, writes into ../data/sample/
    python make_synthetic.py --seed 42 --az-count 160 --el-count 41
"""

import argparse
import csv
import math
import struct
from pathlib import Path

# ===========================================================================
# Camera / image constants -- MUST match src/kernels.cuh exactly (manually
# synchronized across Python and CUDA C++, no shared source -- see kernels.cuh
# file header). IDENTICAL to 01.18/02.02's teaching camera (cited).
# ===========================================================================
IMG_W, IMG_H = 160, 120
FX, FY, CX, CY = 154.0, 152.0, 80.0, 60.0
MAX_DEPTH_M = 20.0
INVALID_DEPTH = -1.0

# Camera world (body-frame: x-forward, y-left, z-up) pose -- IDENTICAL position
# to 01.18's CAM_POS.
CAM_POS = (0.0, 0.0, 1.5)

# LiDAR world (body-frame) pose -- IDENTICAL to 01.18's LIDAR_POS: 0.30 m above
# and 0.05 m behind the camera. This position, together with the axis-permutation
# rotation below, reproduces kTCameraLidar in kernels.cuh EXACTLY (bit-for-bit
# the same constants 01.18/02.02 use) -- cited, not re-derived.
LIDAR_POS = (-0.05, 0.0, 1.80)


def body_to_cam(v):
    # x_cam = -y_body ; y_cam = -z_body ; z_cam = x_body (pure axis permutation,
    # 01.18's derivation, reused verbatim).
    x, y, z = v
    return (-y, -z, x)


def cam_to_body(v):
    xc, yc, zc = v
    return (zc, -xc, -yc)


# T_camera_lidar, mirrored from kernels.cuh's kTCameraLidar (manually
# synchronized contract, see kernels.cuh file header) -- used here only to
# report an honest projected-density number to the console.
T_CAM_LIDAR_R = (0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
T_CAM_LIDAR_T = (0.0, -0.30, -0.05)

DEFAULT_SEED = 42

# ===========================================================================
# xorshift32 -- the repo's portable deterministic PRNG (01.07/01.18's bit-mixing
# rule, cited). Operates on Python ints masked to 32 bits so results match the
# C++ uint32_t version bit-for-bit on any platform.
# ===========================================================================
MASK32 = 0xFFFFFFFF


def xorshift32(x: int) -> int:
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5) & MASK32
    return x & MASK32


def uniform01(state: int):
    state = xorshift32(state)
    v = ((state >> 8) & 0xFFFFFF) / float(0x1000000) + (0.5 / 0x1000000)
    return state, v


def gaussian(state: int, sigma: float):
    state, u1 = uniform01(state)
    state, u2 = uniform01(state)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return state, sigma * z


# ===========================================================================
# Scene -- ground plane + four flat-colored objects. Every object is
# "flat_unlit" (no Lambertian shading): this project's evaluation gates
# compare SAMPLED color against a single per-point TRUE color, and an
# unshaded flat surface has an unambiguous, exact true color everywhere on
# its face -- a Lambertian shading gradient would smear that truth into a
# range, which is 01.18's problem (depth/RGB edge correlation), not this
# project's (README "Limitations" names this simplification explicitly).
#
# THE OCCLUSION GEOMETRY (worked in the module docstring and THEORY.md):
#   occluder  -- RED box, top at z=1.6 m, 4 m out.  Blocks the CAMERA's (but
#                not the higher LiDAR's) view of background points with
#                z in (1.2, 1.8) m -- the occlusion cohort.
#   background-- GREEN wall, 12 m out, spanning the full height range so the
#                cohort band actually exists on it.
#   blue_box / yellow_box -- ordinary, unoccluded color-boundary test
#                subjects (coloring_accuracy + the sensitivity sweep's
#                color-boundary population).
# ===========================================================================
GROUND_COLOR = (100, 100, 100)          # gray asphalt-like floor, flat (no texture -- see module docstring)
OCCLUDER_COLOR = (200, 60, 60)          # RED -- the "wrong color" a naive fusion paints onto hidden background points
BACKGROUND_COLOR = (60, 160, 70)        # GREEN -- the occlusion cohort's TRUE color
BLUE_COLOR = (60, 90, 200)
YELLOW_COLOR = (230, 210, 60)

OBJECTS = [
    {"name": "background", "type": "box", "center": (12.5, 0.0, 2.0), "half": (0.5, 6.0, 4.0), "color": BACKGROUND_COLOR},
    # occluder: x=[3.6,4.4] (top at z=1.6 -- see the occlusion-geometry derivation
    # in the module docstring), y=[-1.6,-0.4] -> azimuth shadow ~[-21.8,-5.7] deg
    # (narrow ON PURPOSE: kept clear of both blue_box's and yellow_box's azimuth
    # footprints below, so those two objects are ordinary, UNoccluded color
    # boundary test subjects -- only "background" points sit in this shadow).
    {"name": "occluder", "type": "box", "center": (4.0, -1.0, 0.8), "half": (0.4, 0.6, 0.8), "color": OCCLUDER_COLOR},
    # blue_box: azimuth ~[21.0, 32.5] deg -- clear of the occluder's shadow.
    {"name": "blue_box", "type": "box", "center": (6.0, 3.0, 0.5), "half": (0.5, 0.5, 0.5), "color": BLUE_COLOR},
    # yellow_box: azimuth ~[3.8, 15.9] deg -- clear of both the occluder's
    # shadow and blue_box's azimuth footprint.
    {"name": "yellow_box", "type": "box", "center": (9.0, 1.6, 0.6), "half": (0.6, 0.6, 0.6), "color": YELLOW_COLOR},
]

# LiDAR beam model. UNLIKE 01.18's fixed 16-beam mechanical table (which this
# project deliberately does NOT reuse -- there is no organized-grid consumer
# here, only the occlusion-cohort statistics), elevation/azimuth density are
# chosen densely enough that the narrow (-2.86 deg, 0 deg) occlusion-cohort
# elevation band (derived below) is sampled by several beams -- a documented
# scope choice named in README "Limitations".
LIDAR_RANGE_NOISE_SIGMA_M = 0.02  # 1-sigma range noise, meters (illustrative MEMS/ToF-class LiDAR)


def ray_color(color):
    return tuple(float(c) for c in color)


# ---------------------------------------------------------------------------
# Ray/object intersection -- axis-aligned-box slab test (the same algorithm
# family 01.18/01.07 use, cited; reimplemented independently for this
# project's flat-color-only scene -- no shading, no checker pattern).
# Returns (s, normal, color) or None. `s` is the ray parameter along the
# CALLER-SUPPLIED direction D: P_hit = O + s*D.
# ---------------------------------------------------------------------------
def intersect_box(O, D, obj):
    cx, cy, cz = obj["center"]
    hx, hy, hz = obj["half"]
    lo = (cx - hx, cy - hy, cz - hz)
    hi = (cx + hx, cy + hy, cz + hz)
    tmin, tmax = -1e18, 1e18
    for ax in range(3):
        o, d = O[ax], D[ax]
        if abs(d) < 1e-12:
            if o < lo[ax] or o > hi[ax]:
                return None
            continue
        t0 = (lo[ax] - o) / d
        t1 = (hi[ax] - o) / d
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > tmin:
            tmin = t0
        if t1 < tmax:
            tmax = t1
    if tmin > tmax or tmax < 1e-6:
        return None
    s = tmin if tmin > 1e-6 else tmax
    return s, (0.0, 0.0, 1.0), ray_color(obj["color"])


def intersect_ground(O, D):
    if D[2] >= -1e-9:
        return None
    s = -O[2] / D[2]
    if s <= 1e-6:
        return None
    return s, (0.0, 0.0, 1.0), ray_color(GROUND_COLOR)


def trace_ray(O, D):
    """Cast one ray; return (color (r,g,b) floats 0..255, surface_name, hit
    distance s_or_None). None distance means the ray hit nothing (sky)."""
    best_s, best_color, best_name = None, None, None
    for obj in OBJECTS:
        hit = intersect_box(O, D, obj)
        if hit is None:
            continue
        s, _n, color = hit
        if best_s is None or s < best_s:
            best_s, best_color, best_name = s, color, obj["name"]
    ghit = intersect_ground(O, D)
    if ghit is not None and (best_s is None or ghit[0] < best_s):
        best_s, best_color, best_name = ghit[0], ghit[2], "ground"

    if best_s is None:
        up = max(0.0, min(1.0, D[2] / (abs(D[2]) + 1.0) + 0.5))
        return (150 + 60 * up, 175 + 55 * up, 225), "sky", None
    return best_color, best_name, best_s


def render_camera(seed: int):
    """Ray-cast the full RGB image (2x2 supersampled)."""
    rgb = bytearray(IMG_W * IMG_H * 3)
    for v in range(IMG_H):
        for u in range(IMG_W):
            r_acc = g_acc = b_acc = 0.0
            for sy in (0.25, 0.75):
                for sx in (0.25, 0.75):
                    xc2 = (u + sx - CX) / FX
                    yc2 = (v + sy - CY) / FY
                    d2 = cam_to_body((xc2, yc2, 1.0))
                    color, _name, _s = trace_ray(CAM_POS, d2)
                    r_acc += color[0]; g_acc += color[1]; b_acc += color[2]
            idx = (v * IMG_W + u) * 3
            rgb[idx + 0] = int(max(0, min(255, r_acc / 4.0)))
            rgb[idx + 1] = int(max(0, min(255, g_acc / 4.0)))
            rgb[idx + 2] = int(max(0, min(255, b_acc / 4.0)))
    return rgb


def is_visible_from_camera(P_world, D_lidar_range: float) -> bool:
    """Ground-truth visibility: cast a SECOND, independent ray from the
    CAMERA's own origin toward the exact 3-D point P_world the LiDAR hit. If
    something else along that ray is measurably closer than P_world itself,
    the camera's own image cannot show P_world's true surface at that pixel
    -- it shows whatever DOES block it (the occluder, in this scene's
    designed cohort). This is the honest, geometry-first definition of
    "occluded from the camera's point of view", independent of any
    approximation the C++ pipeline's z-buffer makes (THEORY.md/README name
    this as the ground-truth oracle the z-buffer's PROXY is graded against).
    """
    dx = P_world[0] - CAM_POS[0]
    dy = P_world[1] - CAM_POS[1]
    dz = P_world[2] - CAM_POS[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-6:
        return True
    ux, uy, uz = dx / dist, dy / dist, dz / dist
    _color, _name, s_hit = trace_ray(CAM_POS, (ux, uy, uz))
    if s_hit is None:
        return False  # camera ray found nothing at all -- treat as not-visible (should not happen for a real surface point)
    # A 5 cm tolerance absorbs float/grazing-angle noise at the exact
    # silhouette edge without weakening the occlusion cohort's signal (the
    # cohort's own height margin is tens of cm, THEORY.md derives the exact
    # figure) -- points genuinely behind the occluder land tens of cm to
    # meters short of s_hit == dist, nowhere near this tolerance.
    return s_hit >= dist - 0.05


def generate_lidar_points(az_count: int, el_count: int, seed: int):
    """Ray-cast the spinning-LiDAR beam grid from LIDAR_POS -- a genuinely
    different sensor origin than the camera, so this is real multi-sensor
    parallax. el_count elevations span ELEV_MIN..ELEV_MAX deg; az_count
    azimuths span +-AZ_HALF_RANGE deg. Returns a list of dicts (one per hit):
    x,y,z (LiDAR-frame meters, range-noised), true_r/g/b (the surface's exact
    flat color, [0,255]), visible (ground-truth camera-visibility flag,
    see is_visible_from_camera), surface (name string)."""
    state = seed
    points = []
    ELEV_MIN_DEG, ELEV_MAX_DEG = -20.0, 10.0   # dense band around the horizon -- see module docstring
    AZ_HALF_RANGE_DEG = 32.0                    # wider than the camera's ~56.5 deg HFOV (matches 01.18's convention)

    elevs = [ELEV_MIN_DEG + i * (ELEV_MAX_DEG - ELEV_MIN_DEG) / (el_count - 1) for i in range(el_count)]
    azs = [-AZ_HALF_RANGE_DEG + i * (2 * AZ_HALF_RANGE_DEG) / (az_count - 1) for i in range(az_count)]

    for el_deg in elevs:
        el = math.radians(el_deg)
        for az_deg in azs:
            az = math.radians(az_deg)
            # Unit direction in the LiDAR's own (body-aligned) frame -- az=0
            # along +x (forward), az>0 toward +y (left), right-handed about +z.
            dx = math.cos(el) * math.cos(az)
            dy = math.cos(el) * math.sin(az)
            dz = math.sin(el)
            color, name, s = trace_ray(LIDAR_POS, (dx, dy, dz))
            if s is None or s > MAX_DEPTH_M:
                continue
            state, noise = gaussian(state, LIDAR_RANGE_NOISE_SIGMA_M)
            r = max(0.01, s + noise)
            # World-frame hit point (BEFORE range noise -- ground truth color
            # and visibility describe the TRUE surface; range noise is a
            # sensor-measurement artifact applied only to the reported x,y,z).
            world_hit = (LIDAR_POS[0] + s * dx, LIDAR_POS[1] + s * dy, LIDAR_POS[2] + s * dz)
            visible = is_visible_from_camera(world_hit, s)
            points.append({
                "x": r * dx, "y": r * dy, "z": r * dz,     # LiDAR-frame (body-aligned, origin-translated only)
                "true_r": color[0], "true_g": color[1], "true_b": color[2],
                "visible": 1 if visible else 0,
                "surface": name,
            })
    return points


def write_ppm(path: Path, w: int, h: int, rgb: bytearray) -> None:
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(bytes(rgb))


def projected_pixel_density(points):
    covered = set()
    R, t = T_CAM_LIDAR_R, T_CAM_LIDAR_T
    for p in points:
        x, y, z = p["x"], p["y"], p["z"]
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
    n_occluded = sum(1 for p in points if p["visible"] == 0)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC LiDAR returns for project 02.17 -- LIDAR-FRAME meters (x,y,z)\n")
        f.write("# true_r/g/b: exact flat surface color the ray hit, [0,255] -- EVALUATION-ONLY ground truth\n")
        f.write("# visible: 1 if unobstructed from the CAMERA's own origin, 0 if occluded -- EVALUATION-ONLY\n")
        f.write("# surface: human-readable object name -- EVALUATION-ONLY\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write(f"# {len(points)} points, {n_occluded} ground-truth-occluded (the occlusion cohort), "
                f"range noise sigma={LIDAR_RANGE_NOISE_SIGMA_M} m\n")
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "true_r", "true_g", "true_b", "visible", "surface"])
        for p in points:
            w.writerow([f"{p['x']:.6f}", f"{p['y']:.6f}", f"{p['z']:.6f}",
                       f"{p['true_r']:.1f}", f"{p['true_g']:.1f}", f"{p['true_b']:.1f}",
                       p["visible"], p["surface"]])


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--az-count", type=int, default=120, help="azimuth samples per beam (default 120)")
    ap.add_argument("--el-count", type=int, default=31, help="elevation beam count (default 31)")
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("[make_synthetic] ray-casting camera RGB image (160x120, 2x2 supersampled)...")
    rgb = render_camera(args.seed)
    write_ppm(args.out / "rgb.ppm", IMG_W, IMG_H, rgb)
    print("[make_synthetic] wrote rgb.ppm")

    print(f"[make_synthetic] ray-casting LiDAR beam grid ({args.el_count} elevations x {args.az_count} azimuths)"
          " + per-point ground-truth visibility (a second ray cast from the camera origin per point)...")
    points = generate_lidar_points(args.az_count, args.el_count, args.seed)
    write_lidar_csv(args.out / "lidar_points.csv", points, args.seed)

    n_occluded = sum(1 for p in points if p["visible"] == 0)
    by_surface = {}
    for p in points:
        by_surface[p["surface"]] = by_surface.get(p["surface"], 0) + 1
    print(f"[make_synthetic] wrote lidar_points.csv ({len(points)} returns, seed={args.seed}, labeled SYNTHETIC)")
    print(f"[make_synthetic] per-surface hit counts: {by_surface}")
    print(f"[make_synthetic] ground-truth-occluded (the occlusion cohort, visible=0): "
          f"{n_occluded}/{len(points)} = {100.0 * n_occluded / len(points):.2f}%")

    n_cov, density = projected_pixel_density(points)
    print(f"[make_synthetic] projected pixel density (z-buffer-deduped): "
          f"{n_cov}/{IMG_W * IMG_H} pixels = {100.0 * density:.2f}%")

    print("[make_synthetic] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
