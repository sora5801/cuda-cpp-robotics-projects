#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.03
(Ground segmentation: RANSAC plane fit; Patchwork++-style GPU port).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project needs a scene that PROVES a single global plane cannot
represent real ground while a patch-local model can — that requires a
designed, controllable, exactly-labeled scene, which only synthesis
provides. No download, license-clean, reproducible bit-for-bit from a fixed
seed (CLAUDE.md paragraph 12): xorshift32, seed 42, stdlib only.

The scene, in the LiDAR sensor's own frame (origin at the sensor, +x
forward, +z up — CLAUDE.md paragraph 12), reuses 02.01's beam-casting
machinery style (cited, not reinvented: the 16-beam elevation table and the
ray-vs-analytic-primitives approach) extended with THREE ground levels
instead of one:

  * FLAT ground at z = -SENSOR_HEIGHT_M, everywhere EXCEPT a forward
    corridor (|y| <= RAMP_Y_HALF_WIDTH_M) between x = RAMP_X_START_M and
    x = PLATEAU_X_END_M — RANSAC's home turf, the majority of the scene.
  * A RAMP inside that corridor: ground rises linearly at RAMP_SLOPE_DEG
    over RAMP_LENGTH_M, from RAMP_X_START_M to RAMP_X_START_M+RAMP_LENGTH_M.
  * A PLATEAU: flat ground again, but RAISED by the ramp's rise, filling
    the rest of the corridor out to PLATEAU_X_END_M — a second ground LEVEL
    a single plane cannot also satisfy alongside the flat segment. Beyond
    the plateau's far edge the ground simply ends (a documented "ledge" —
    rays that would continue past it return nothing, an honest dropout).
  * NO room-bounding walls (an intentional departure from 02.01's enclosed
    room — see "Why no bounding walls" below). Ground otherwise extends to
    MAX_RANGE_M in every direction; beams that clear MAX_RANGE without
    hitting ground, an obstacle, or the canopy simply return nothing.
  * Six obstacles STANDING ON the local ground: two boxes + one pole on the
    flat segment, one box + one pole on the plateau, and one THIN WALL
    SEGMENT (a short retaining-wall-style obstacle, not a room boundary) on
    the flat segment — ground segmentation must reject every one of them
    regardless of which level they stand on.
  * A CANOPY overhang: a disc of points floating well above the flat
    ground (a stand-in for tree branches/an awning) — points that must
    NEVER be classified as ground (main.cu's `overhang` gate: calling
    canopy "ground" would let a path planner route a robot under it as if
    it were drivable surface — the safety-relevant miss this scene is
    designed to catch).

  Why no bounding walls (an honest scoping departure from 02.01)
  ----------------------------------------------------------------
  An early version of this scene DID use 02.01-style room-bounding walls.
  Measuring it exposed a real problem for THIS project's specific teaching
  goal: with a 16-beam elevation table spanning +-15 degrees, roughly half
  the beams point upward and can only ever hit a wall (never the ground);
  at typical room scale those wall returns wildly outnumber ground returns
  (measured: ~118k wall/obstacle points vs. ~43k ground points on the
  walled version of this scene) and, worse, a single wall plane can hold
  MORE points than the flat ground plane — RANSAC would then sometimes
  pick a WALL as its "best" global plane instead of the ground, an
  ambiguity that made `single_plane_failure` (README/THEORY) unreliable
  from run to run. Removing the room-bounding walls (keeping ONE small
  wall-segment OBSTACLE standing on the ground, per the catalog's
  "boxes/cylinders/a wall" list) fixes both problems at once and is the
  more honest choice for a project about GROUND segmentation specifically:
  the point of this scene is the ground/not-ground boundary, not spatial
  containment (02.01 already teaches enclosed-room scanning).

Per-point GROUND TRUTH is exact by construction: every beam-scan return
records which analytic surface it came from (see build_beam_scan below), so
this script writes an exact ground/not-ground label AND a zone id
(0=flat,1=ramp,2=plateau,-1=not-ground) for every point — no estimation,
no noise in the LABEL itself (only in the ray's measured RANGE, matching
real sensor noise).

Usage
-----
    python make_synthetic.py                 # writes the committed sample
    python make_synthetic.py --out DIR/FILE   # experiments; do not commit
"""

import argparse
import math
import struct
import sys
from pathlib import Path

# ===========================================================================
# Deterministic RNG: xorshift32 (stdlib-only — CLAUDE.md paragraph 12), the
# SAME algorithm 02.01's generator and this project's CUDA device code use
# (see kernels.cuh's xorshift32_step / hypothesis_seed for the CUDA side).
# ===========================================================================
class Xorshift32:
    """32-bit xorshift PRNG (Marsaglia 2003). See 02.01's make_synthetic.py
    for the identical implementation this one is copied from verbatim."""

    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        if s == 0:
            s = 1
        self.state = s

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        self.state = x
        return x

    def uniform01(self) -> float:
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform01()


DEFAULT_SEED = 42

# ===========================================================================
# Scene geometry — SHARED CONSTANTS with src/kernels.cuh (same values,
# documented there too; the LEAF_M / kVoxelLeafM precedent from 02.01 for
# why a documented match is sufficient here: these are geometry-DESIGN
# constants, not a safety-critical binary-format field main.cu cross-checks
# at load time).
# ===========================================================================
SENSOR_HEIGHT_M = 1.5          # LiDAR mount height above the BASE floor
RAMP_X_START_M = 4.0           # corridor: ramp begins at this forward range (must match kernels.cuh)
RAMP_LENGTH_M = 4.0            # ramp run length (must match kernels.cuh)
RAMP_Y_HALF_WIDTH_M = 3.5      # corridor half-width (must match kernels.cuh)
RAMP_SLOPE_DEG = 8.0           # ramp grade (must match kernels.cuh)
RAMP_RISE_M = math.tan(math.radians(RAMP_SLOPE_DEG)) * RAMP_LENGTH_M   # ~0.562 m
PLATEAU_Z_M = -SENSOR_HEIGHT_M + RAMP_RISE_M                            # plateau ground height (world z)

PLATEAU_X_END_M = RAMP_X_START_M + RAMP_LENGTH_M + 4.0   # ramp end + 4 m plateau depth -> plateau's far "ledge"
MAX_RANGE_M = 16.0             # sensor spec ceiling — also the FLAT ground's effective footprint (no walls to stop it)

# 16-beam elevation table, cited verbatim from 01.18 via 02.01's precedent.
BEAM_ELEV_DEG = list(range(-15, 16, 2))
NUM_BEAMS = len(BEAM_ELEV_DEG)
assert NUM_BEAMS == 16, "beam table must match the repo's cited 16-beam model exactly"

AZIMUTH_STEPS = 2560           # ~0.1406 deg/step, matching 02.01's single-sweep resolution
REVOLUTIONS = 12               # accumulated sweeps -> the "short dwell submap" scope decision (see 02.01);
                                # higher than 02.01's 6 because roughly half this beam table's elevations
                                # point upward and (with no bounding walls) simply return nothing, so more
                                # sweeps are needed to reach a comparable committed-sample point count
RANGE_NOISE_M = 0.015          # +-15 mm per-return range noise (matches 02.01/11.01's order of magnitude)

# Obstacles standing on the LOCAL ground (flat or plateau — never inside the
# ramp corridor itself, so the ramp's own surface stays clean for the
# slope_accuracy diagnostic). Each tuple: (cx, cy, half_x, half_y, height, base_z).
BOXES = [
    (2.5, -6.0, 0.40, 0.40, 0.80, -SENSOR_HEIGHT_M),   # flat zone
    (-3.0, 5.5, 0.35, 0.35, 1.00, -SENSOR_HEIGHT_M),   # flat zone
    (10.0, 0.0, 0.50, 0.50, 0.90, PLATEAU_Z_M),        # plateau
    (2.0, 4.5, 1.50, 0.15, 1.00, -SENSOR_HEIGHT_M),    # a THIN WALL SEGMENT obstacle (3 m long, 0.3 m thick,
                                                        # 1 m tall) — the catalog's "...a wall", modeled as a
                                                        # long thin box rather than a room-bounding plane (see
                                                        # "Why no bounding walls" above)
]
# Poles (vertical cylinders): (cx, cy, radius, height, base_z).
POLES = [
    (5.5, -7.0, 0.12, 1.20, -SENSOR_HEIGHT_M),         # flat zone (x sits inside the corridor's x-range but
                                                        # y is well outside the corridor's +-3.5 m width)
    (9.0, 2.0, 0.12, 1.00, PLATEAU_Z_M),               # plateau
]

# Canopy overhang: a disc of points floating well above flat ground — a
# stand-in for tree branches/an awning a robot must never treat as ground.
#
# WHY the center sits at r=6.2 m specifically (not just "somewhere flat"):
# this 16-beam elevation table only puts a handful of beams (-15,-13,-11,
# -9,-7 degrees) close enough to vertical to reach the floor within
# MAX_RANGE_M at all -- each such beam paints a THIN RING of ground returns
# at one FIXED range (range = SENSOR_HEIGHT_M / sin|elevation|; a real
# consequence of a flat plane + a straight ray, not a bug), leaving WIDE
# RADIAL GAPS with zero ground coverage between rings (measured ring radii:
# ~5.8, 6.7, 7.9, 9.6, 12.3 m). An early version of this scene centered the
# canopy at r=4 m -- squarely inside the gap before the first ring -- so
# every CZM patch under the canopy had ZERO real ground points to seed
# from, and (correctly, given only canopy data to fit) fit a plane to the
# canopy itself. Centering the canopy across the r=5.8 m ring instead
# guarantees genuine ground points coexist with canopy points in the same
# patches, which is what makes the overhang gate a fair test of "can the
# algorithm tell ground from overhang GIVEN both are visible" rather than
# an artifact of where the beam table happens to sample.
CANOPY_CENTER_XY = (-6.2, 0.0)
CANOPY_RADIUS_M = 1.0
CANOPY_Z_LO_M = 0.3            # world z (sensor frame); ground beneath is flat at z=-1.5 -> 1.8 m clearance
CANOPY_Z_HI_M = 1.6            # -> up to 3.1 m clearance
N_CANOPY = 2500


# ---------------------------------------------------------------------------
# ray_box_intersect — the standard SLAB METHOD for an axis-aligned box
# (cited verbatim from 02.01's make_synthetic.py; see that file's docstring
# for the full derivation).
# ---------------------------------------------------------------------------
def ray_box_intersect(ox, oy, oz, dx, dy, dz, x0, x1, y0, y1, z0, z1):
    t_min, t_max = -math.inf, math.inf
    for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1), (oz, dz, z0, z1)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
        else:
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None
    if t_max < 1e-6:
        return None
    t_hit = t_min if t_min > 1e-6 else t_max
    return t_hit if t_hit > 1e-6 else None


# ---------------------------------------------------------------------------
# ray_cylinder_intersect — ray vs. an infinite-height vertical cylinder
# (axis parallel to z, centered at (cx,cy), radius r), clipped afterward to
# [z0,z1]. Standard quadratic in the ray parameter t, restricted to the
# ray's (x,y) components only (the cylinder's cross-section is a circle
# independent of z): |( ox+dx*t - cx, oy+dy*t - cy )|^2 = r^2.
#
# Returns the smallest POSITIVE t whose hit point's z lies in [z0,z1], or
# None. A vertical pole standing on the ground is this repo's simplest
# real-world stand-in for street furniture (bollards, sign posts, lamp
# posts) — a shape RANSAC/CZM must reject regardless of its footprint size.
# ---------------------------------------------------------------------------
def ray_cylinder_intersect(ox, oy, dx, dy, dz, cx, cy, r, z0, z1):
    a = dx * dx + dy * dy
    if a < 1e-12:
        return None  # ray nearly vertical: cannot cross a vertical cylinder's side wall
    ex, ey = ox - cx, oy - cy
    b = 2.0 * (dx * ex + dy * ey)
    c = ex * ex + ey * ey - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None  # ray misses the infinite cylinder entirely
    sqrt_disc = math.sqrt(disc)
    for t in sorted(((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))):
        if t <= 1e-6:
            continue
        z = dz * t
        if z0 <= z <= z1:
            return t
    return None


# ---------------------------------------------------------------------------
# ray_intersect_ground — test the THREE ground candidate planes (flat,
# ramp, plateau) and return (t, zone_name) for the nearest valid hit, or
# (None, None). Each candidate is an infinite plane restricted to its own
# footprint (see the module docstring's geometry) — exactly the "test every
# candidate surface, keep the nearest IN-BOUNDS one" pattern 02.01 uses for
# its ground+walls+boxes, extended here to THREE ground planes instead of
# one.
# ---------------------------------------------------------------------------
def ray_intersect_ground(dx, dy, dz):
    best_t, best_zone = None, None

    # --- FLAT: z = -SENSOR_HEIGHT_M -----------------------------------------
    # No room bounds any more (see "Why no bounding walls" in the module
    # docstring) — the flat plane extends until the ramp corridor carves it
    # out, or until MAX_RANGE_M truncates it (the shared best_t<=MAX_RANGE_M
    # check in ray_intersect_scene, applied once for every surface family).
    if dz < -1e-9:
        t = -SENSOR_HEIGHT_M / dz
        if t > 1e-6:
            x, y = dx * t, dy * t
            in_flat_region = (x < RAMP_X_START_M) or (abs(y) > RAMP_Y_HALF_WIDTH_M)
            if in_flat_region and (best_t is None or t < best_t):
                best_t, best_zone = t, 'flat'

    # --- RAMP: z = -SENSOR_HEIGHT_M + tan(slope)*(x - RAMP_X_START_M) ------
    tan_theta = math.tan(math.radians(RAMP_SLOPE_DEG))
    denom = dz - tan_theta * dx
    if abs(denom) > 1e-9:
        t = -(SENSOR_HEIGHT_M + tan_theta * RAMP_X_START_M) / denom
        if t > 1e-6:
            x, y = dx * t, dy * t
            if RAMP_X_START_M <= x < RAMP_X_START_M + RAMP_LENGTH_M and abs(y) <= RAMP_Y_HALF_WIDTH_M:
                if best_t is None or t < best_t:
                    best_t, best_zone = t, 'ramp'

    # --- PLATEAU: z = PLATEAU_Z_M (flat, but raised), bounded by the ledge -
    if dz < -1e-9:
        t = PLATEAU_Z_M / dz
        if t > 1e-6:
            x, y = dx * t, dy * t
            if RAMP_X_START_M + RAMP_LENGTH_M <= x <= PLATEAU_X_END_M and abs(y) <= RAMP_Y_HALF_WIDTH_M:
                if best_t is None or t < best_t:
                    best_t, best_zone = t, 'plateau'

    return best_t, best_zone


# ---------------------------------------------------------------------------
# ray_intersect_scene — cast one ray from the sensor origin, test EVERY
# candidate surface family (ground x3, boxes/wall-segment, poles), and
# return (range, hit_type) for the nearest valid hit, or (None, None) — an
# honest "no return" (no walls bound this scene; see the module docstring's
# "Why no bounding walls" — an upward beam, a beam past the plateau's
# ledge, or a beam beyond MAX_RANGE_M simply sees nothing).
# ---------------------------------------------------------------------------
def ray_intersect_scene(dx, dy, dz):
    best_t, best_type = ray_intersect_ground(dx, dy, dz)

    for (cx, cy, hx, hy, height, base_z) in BOXES:
        t = ray_box_intersect(0.0, 0.0, 0.0, dx, dy, dz,
                              cx - hx, cx + hx, cy - hy, cy + hy, base_z, base_z + height)
        if t is not None and (best_t is None or t < best_t):
            best_t, best_type = t, 'box'

    for (cx, cy, r, height, base_z) in POLES:
        t = ray_cylinder_intersect(0.0, 0.0, dx, dy, dz, cx, cy, r, base_z, base_z + height)
        if t is not None and (best_t is None or t < best_t):
            best_t, best_type = t, 'pole'

    if best_t is not None and best_t <= MAX_RANGE_M:
        return best_t, best_type
    return None, None


ZONE_ID_FOR_TYPE = {'flat': 0, 'ramp': 1, 'plateau': 2}   # everything else -> zone_id -1, ground_label 0


def build_beam_scan(rng: Xorshift32):
    """Cast REVOLUTIONS accumulated sweeps of NUM_BEAMS x AZIMUTH_STEPS rays
    against the analytic scene. Returns (points, labels, zones, n_rays_cast)
    where points[i]=(x,y,z), labels[i] in {0,1} (ground/not), zones[i] in
    {-1,0,1,2}."""
    points, labels, zones = [], [], []
    n_rays_cast = 0
    for _rev in range(REVOLUTIONS):
        for el_deg in BEAM_ELEV_DEG:
            el = math.radians(el_deg)
            cel, sel = math.cos(el), math.sin(el)
            for az_step in range(AZIMUTH_STEPS):
                az = 2.0 * math.pi * az_step / AZIMUTH_STEPS
                dx = cel * math.cos(az)
                dy = cel * math.sin(az)
                dz = sel
                n_rays_cast += 1
                t, hit_type = ray_intersect_scene(dx, dy, dz)
                if t is None:
                    continue  # honest dropout, not injected noise
                r = t + rng.uniform(-RANGE_NOISE_M, RANGE_NOISE_M)
                if r <= 1e-6:
                    r = t
                points.append((dx * r, dy * r, dz * r))
                is_ground = 1 if hit_type in ZONE_ID_FOR_TYPE else 0
                labels.append(is_ground)
                zones.append(ZONE_ID_FOR_TYPE.get(hit_type, -1))
    return points, labels, zones, n_rays_cast


def build_canopy(rng: Xorshift32):
    """N_CANOPY points uniformly filling a DISC in (x,y) at CANOPY_CENTER_XY,
    each with an independent random height in [CANOPY_Z_LO_M, CANOPY_Z_HI_M]
    — an overhang appended AFTER the beam scan (not raycast: real canopy
    returns are sparse, porous scatter off many small leaves/branches, not
    one solid raycastable surface — the same "append a designed adversarial
    region" technique 02.01 uses for its dense/sparse clusters)."""
    cx0, cy0 = CANOPY_CENTER_XY
    points = []
    for _ in range(N_CANOPY):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        # sqrt(uniform01) samples a UNIFORM DISC (not a uniform SQUARE mapped
        # to a disc, which would over-concentrate points near the center —
        # the standard disc-sampling identity: for a uniform disc, the CDF of
        # radius r is proportional to r^2, so inverting requires a sqrt).
        radius = CANOPY_RADIUS_M * math.sqrt(rng.uniform01())
        x = cx0 + radius * math.cos(angle)
        y = cy0 + radius * math.sin(angle)
        z = rng.uniform(CANOPY_Z_LO_M, CANOPY_Z_HI_M)
        points.append((x, y, z))
    return points


def write_binary_sample(out_path: Path, points_beam, labels_beam, zones_beam, points_canopy):
    """Write the committed sample as a small fixed binary format:

        bytes  0.. 7  magic       b'GNDSEG01' (8 bytes, no null terminator)
        bytes  8..11  int32       n_total
        bytes 12..15  int32       n_ground_flat
        bytes 16..19  int32       n_ground_ramp
        bytes 20..23  int32       n_ground_plateau
        bytes 24..27  int32       n_nonground_beam  (box/pole/wall-segment returns in the beam scan)
        bytes 28..31  int32       reserved (0)
        bytes 32..35  int32       n_canopy          (appended overhang returns, all non-ground)
        bytes 36..39  float32     sensor_height_m
        bytes 40..43  float32     ramp_slope_deg
        bytes 44..47  int32       reserved (0)
        bytes 48..51  int32       reserved (0)
        bytes 52..    per point, 20 bytes each, in the FIXED ORDER
                       [beam scan points in cast order] then [canopy points]:
            float32 x, float32 y, float32 z   (meters, sensor frame)
            int32   ground_label               (1=ground, 0=not-ground)
            int32   zone_id                     (0=flat,1=ramp,2=plateau,-1=not-ground)

    Every field is written with an EXPLICIT little-endian struct format
    (never a raw fwrite of a C struct), following 02.01's precedent, so the
    layout never depends on any compiler's struct-padding rules.
    """
    n_total = len(points_beam) + len(points_canopy)
    n_flat = sum(1 for z in zones_beam if z == 0)
    n_ramp = sum(1 for z in zones_beam if z == 1)
    n_plateau = sum(1 for z in zones_beam if z == 2)
    n_nonground_beam = sum(1 for z in zones_beam if z == -1)
    n_canopy = len(points_canopy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        f.write(b'GNDSEG01')
        f.write(struct.pack('<i', n_total))
        f.write(struct.pack('<iii', n_flat, n_ramp, n_plateau))
        f.write(struct.pack('<i', n_nonground_beam))
        f.write(struct.pack('<i', 0))                    # reserved
        f.write(struct.pack('<i', n_canopy))
        f.write(struct.pack('<ff', SENSOR_HEIGHT_M, RAMP_SLOPE_DEG))
        f.write(struct.pack('<ii', 0, 0))                 # reserved x2

        for (x, y, z), label, zone in zip(points_beam, labels_beam, zones_beam):
            f.write(struct.pack('<fff', x, y, z))
            f.write(struct.pack('<ii', label, zone))
        for (x, y, z) in points_canopy:
            f.write(struct.pack('<fff', x, y, z))
            f.write(struct.pack('<ii', 0, -1))

    return n_total, n_flat, n_ramp, n_plateau, n_nonground_beam, n_canopy


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / 'data' / 'sample' / 'ground_scan.bin'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out,
                        help='output binary path (default: ../data/sample/ground_scan.bin)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    points_beam, labels_beam, zones_beam, n_rays_cast = build_beam_scan(rng)
    points_canopy = build_canopy(rng)

    n_total, n_flat, n_ramp, n_plateau, n_nonground_beam, n_canopy = write_binary_sample(
        args.out, points_beam, labels_beam, zones_beam, points_canopy)

    n_beam = len(points_beam)
    hit_rate = 100.0 * n_beam / n_rays_cast if n_rays_cast else 0.0
    print(f"[make_synthetic] SYNTHETIC 3-level ground scene (seed={args.seed}): "
          f"{REVOLUTIONS} revolutions x {NUM_BEAMS} beams x {AZIMUTH_STEPS} azimuth steps "
          f"= {n_rays_cast} rays cast, {n_beam} valid returns ({hit_rate:.1f}% hit rate)")
    print(f"[make_synthetic] ground breakdown: flat={n_flat} ramp={n_ramp} plateau={n_plateau} "
          f"(ground total={n_flat+n_ramp+n_plateau})")
    print(f"[make_synthetic] non-ground in beam scan (box/pole/wall-segment): {n_nonground_beam}")
    print(f"[make_synthetic] + {n_canopy} appended CANOPY overhang points "
          f"(disc radius {CANOPY_RADIUS_M} m at {CANOPY_CENTER_XY}, z in "
          f"[{CANOPY_Z_LO_M},{CANOPY_Z_HI_M}] m)")
    print(f"[make_synthetic] wrote {args.out} ({n_total} points total, "
          f"{n_total*20 + 52} bytes, ramp_slope_deg={RAMP_SLOPE_DEG}, labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
