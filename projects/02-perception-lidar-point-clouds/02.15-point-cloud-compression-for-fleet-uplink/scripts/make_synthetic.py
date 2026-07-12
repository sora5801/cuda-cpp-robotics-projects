#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.15
(Point cloud compression (octree/entropy) for fleet uplink).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
This project's whole teaching payoff is a CONTRAST: real robot maps compress
well because surfaces are 2-D manifolds in 3-D space (most of a room's
volume is empty air; the occupied volume concentrates on floor/wall/object
SURFACES), while incompressible geometry (points scattered with no surface
structure at all) does not. Both halves of that contrast are cheap to
synthesize exactly, with a fixed seed, so nothing here is downloaded
(CLAUDE.md paragraph 12: reproducible, license-clean) — and unlike a real
LiDAR scan (which needs a sensor model, beam divergence, occlusion...) a
STATIC MAP TILE is even simpler to synthesize: it is literally "sample
points on a set of known surfaces," no raycasting required.

Two committed clouds, both exactly POINTS_PER_CLOUD points (a fair,
apples-to-apples comparison — same N, same physical room scale, only the
GEOMETRY differs):

  1. structured_map.bin — a small warehouse-room map tile: one floor
     rectangle, four wall rectangles (room interior faces only — a real
     fleet map is an aggregated multi-view reconstruction, not a single
     scan, so there is no self-occlusion to model), and five box-shaped
     obstacles (furniture/pallets), each contributing up to 5 visible faces
     (top + 4 sides; bottom faces sit on the floor and are never sampled).
     Points are allocated PROPORTIONAL TO SURFACE AREA (a constant areal
     point density, the same physical assumption a real dense map scan
     makes) and each carries a small (+-3 mm) jitter along its surface's
     NORMAL direction — enough to avoid a mathematically zero-thickness
     slab (which would make every leaf-quantization comparison in main.cu
     trivially exact for uninteresting reasons) while remaining, honestly,
     an overwhelmingly flat, highly compressible scene.

  2. pathological_cube.bin — POINTS_PER_CLOUD points drawn UNIFORMLY AT
     RANDOM inside a cube of the SAME side length as the structured tile's
     bounding cube (so the two clouds occupy comparable physical volumes
     at comparable point densities) — geometry with NO surface structure
     at all, the designed worst case main.cu's entropy_payoff gate
     measures against.

Room geometry (reused in spirit from 02.01/02.02's virtual warehouse room,
cited, though this project samples SURFACES directly rather than raycasting
a sensor — see the module docstring's "what is new here" note above): a
20 m x 20 m room (ROOM_HALF_M=10 m), floor at z=0, walls WALL_HEIGHT_M=3 m
tall, five axis-aligned obstacle boxes. Frame: right-handed, +z up, meters
— this project's clouds are static MAP TILES (SYSTEM_DESIGN.md's frame
convention applies unchanged; there is no sensor origin to define
forward/left against, so no such convention is claimed here).

Binary sample format (see ../data/README.md for the authoritative field
table + SHA-256):
    bytes 0..7  magic       b'PCFU0001' (8 bytes, no null terminator)
    bytes 8..11 int32       N — point count (POINTS_PER_CLOUD in both files)
    bytes 12..  float32 x N x3   interleaved (x,y,z), meters, map frame

Usage
-----
    python make_synthetic.py                 # writes both committed samples
    python make_synthetic.py --out DIR        # experiments; do not commit
"""

import argparse
import struct
import sys
from pathlib import Path

# ===========================================================================
# Deterministic RNG: xorshift32 (Marsaglia 2003), reused verbatim from
# 02.01/02.02's make_synthetic.py (cited) — the same three-shift/three-XOR
# core this repo's CUDA device code uses (project 11.01, 08.01), so a
# learner reading Python or CUDA C++ across projects sees one RNG algorithm.
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        if s == 0:
            s = 1  # xorshift32 is degenerate (stays 0 forever) at seed 0
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
        """(0,1], never exactly 0 — top 24 bits (float32 significand) + half-ULP bias."""
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform01()


DEFAULT_SEED = 42  # repo convention (CLAUDE.md paragraph 12)
POINTS_PER_CLOUD = 200_000  # both committed clouds, exactly — a fair, apples-to-apples comparison

# ===========================================================================
# Room geometry (meters, map frame, +z up).
# ===========================================================================
ROOM_HALF_M = 10.0     # room spans x,y in [-10, 10]  (20 m x 20 m)
WALL_HEIGHT_M = 3.0     # walls span z in [0, 3]
JITTER_M = 0.003        # +-3 mm per-point jitter along the surface normal (RANGE_NOISE_M-style realism)

# (center_x, center_y, half_x, half_y, height) — five furniture/pallet-like
# obstacle boxes, sized and placed by hand to avoid overlapping each other
# or the walls; illustrative, not derived from any real warehouse layout.
BOXES = [
    (4.0, 3.0, 1.0, 0.6, 1.2),
    (-3.0, -4.0, 0.5, 0.5, 0.8),
    (6.0, -5.0, 1.5, 0.4, 0.5),
    (-6.0, 5.0, 0.8, 0.8, 1.6),
    (0.0, -7.0, 2.0, 0.3, 0.4),
]


# ---------------------------------------------------------------------------
# Surface — a flat rectangular patch parameterized by an origin corner and
# two IN-PLANE axis vectors (u_axis, v_axis, both unit length) spanning
# u_len x v_len meters, plus its outward unit NORMAL (used only for the
# +-JITTER_M realism nudge). area = u_len * v_len drives point allocation
# (see build_structured_cloud): a constant areal point density is the same
# physical assumption a real dense map scan makes.
# ---------------------------------------------------------------------------
class Surface:
    __slots__ = ("origin", "u_axis", "v_axis", "u_len", "v_len", "normal", "area")

    def __init__(self, origin, u_axis, v_axis, u_len, v_len, normal):
        self.origin = origin
        self.u_axis = u_axis
        self.v_axis = v_axis
        self.u_len = u_len
        self.v_len = v_len
        self.normal = normal
        self.area = u_len * v_len

    def sample(self, rng: Xorshift32):
        u = rng.uniform01() * self.u_len
        v = rng.uniform01() * self.v_len
        j = rng.uniform(-JITTER_M, JITTER_M)
        x = self.origin[0] + u * self.u_axis[0] + v * self.v_axis[0] + j * self.normal[0]
        y = self.origin[1] + u * self.u_axis[1] + v * self.v_axis[1] + j * self.normal[1]
        z = self.origin[2] + u * self.u_axis[2] + v * self.v_axis[2] + j * self.normal[2]
        return (x, y, z)


def build_surfaces():
    """The room's 30 flat surfaces: 1 floor + 4 walls + 5 boxes x 5 faces
    each (top + 4 sides; bottom faces sit on the floor and are never
    sampled — a real fleet map is an aggregated reconstruction with no
    self-occlusion to model, but a box's underside is still never observed
    by anything, so omitting it is physically honest, not a shortcut)."""
    R = ROOM_HALF_M
    H = WALL_HEIGHT_M
    surfaces = []

    # Floor: z=0, normal +z.
    surfaces.append(Surface((-R, -R, 0.0), (1, 0, 0), (0, 1, 0), 2 * R, 2 * R, (0, 0, 1)))

    # Four walls (interior faces): normal points INTO the room.
    surfaces.append(Surface((-R, R, 0.0), (1, 0, 0), (0, 0, 1), 2 * R, H, (0, -1, 0)))   # north wall (y=+R)
    surfaces.append(Surface((-R, -R, 0.0), (1, 0, 0), (0, 0, 1), 2 * R, H, (0, 1, 0)))   # south wall (y=-R)
    surfaces.append(Surface((R, -R, 0.0), (0, 1, 0), (0, 0, 1), 2 * R, H, (-1, 0, 0)))   # east wall (x=+R)
    surfaces.append(Surface((-R, -R, 0.0), (0, 1, 0), (0, 0, 1), 2 * R, H, (1, 0, 0)))   # west wall (x=-R)

    # Five boxes, five faces each.
    for (cx, cy, hx, hy, height) in BOXES:
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        surfaces.append(Surface((x0, y0, height), (1, 0, 0), (0, 1, 0), 2 * hx, 2 * hy, (0, 0, 1)))   # top
        surfaces.append(Surface((x1, y0, 0.0), (0, 1, 0), (0, 0, 1), 2 * hy, height, (1, 0, 0)))      # +x side
        surfaces.append(Surface((x0, y0, 0.0), (0, 1, 0), (0, 0, 1), 2 * hy, height, (-1, 0, 0)))     # -x side
        surfaces.append(Surface((x0, y1, 0.0), (1, 0, 0), (0, 0, 1), 2 * hx, height, (0, 1, 0)))      # +y side
        surfaces.append(Surface((x0, y0, 0.0), (1, 0, 0), (0, 0, 1), 2 * hx, height, (0, -1, 0)))     # -y side

    return surfaces


def allocate_point_counts(surfaces, total_points):
    """Proportional-to-area allocation with a deterministic largest-
    remainder correction so the counts sum EXACTLY to total_points (a
    plain floor() of each float share would under-count by a few dozen
    points; largest-remainder-first is the standard, deterministic fix)."""
    total_area = sum(s.area for s in surfaces)
    raw = [total_points * s.area / total_area for s in surfaces]
    counts = [int(r) for r in raw]  # floor
    remainder = total_points - sum(counts)
    # Distribute the leftover points to the surfaces with the largest
    # fractional remainder, ties broken by surface index (deterministic).
    order = sorted(range(len(surfaces)), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for i in range(remainder):
        counts[order[i]] += 1
    return counts


def build_structured_cloud(rng: Xorshift32):
    surfaces = build_surfaces()
    counts = allocate_point_counts(surfaces, POINTS_PER_CLOUD)
    pts = []
    for surf, cnt in zip(surfaces, counts):
        for _ in range(cnt):
            pts.append(surf.sample(rng))
    assert len(pts) == POINTS_PER_CLOUD
    return pts, surfaces, counts


def build_pathological_cloud(rng: Xorshift32, cube_half_m: float):
    """POINTS_PER_CLOUD points, uniformly at random, inside a cube of side
    2*cube_half_m centered at the origin — geometry with NO surface
    structure, the designed worst case for the octree/entropy codec."""
    pts = []
    for _ in range(POINTS_PER_CLOUD):
        x = rng.uniform(-cube_half_m, cube_half_m)
        y = rng.uniform(-cube_half_m, cube_half_m)
        z = rng.uniform(-cube_half_m, cube_half_m)
        pts.append((x, y, z))
    return pts


def write_binary_cloud(out_path: Path, pts):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for (x, y, z) in pts:
        flat.extend((x, y, z))
    with out_path.open('wb') as f:
        f.write(b'PCFU0001')
        f.write(struct.pack('<i', len(pts)))
        f.write(struct.pack(f'<{len(flat)}f', *flat))
    return 8 + 4 + len(flat) * 4


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir.parent / 'data' / 'sample'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out_dir,
                        help='output directory (default: ../data/sample/)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    structured_pts, surfaces, counts = build_structured_cloud(rng)

    # The pathological cube's half-extent is MEASURED from the structured
    # tile's own bounding cube (not hardcoded), so the two clouds always
    # occupy comparable physical volumes even if BOXES/ROOM_HALF_M change.
    xs = [p[0] for p in structured_pts]
    ys = [p[1] for p in structured_pts]
    zs = [p[2] for p in structured_pts]
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    cube_half_m = extent / 2.0

    pathological_pts = build_pathological_cloud(rng, cube_half_m)

    structured_out = args.out / 'structured_map.bin'
    pathological_out = args.out / 'pathological_cube.bin'
    structured_bytes = write_binary_cloud(structured_out, structured_pts)
    pathological_bytes = write_binary_cloud(pathological_out, pathological_pts)

    print(f"[make_synthetic] SYNTHETIC structured map tile (seed={args.seed}): "
          f"{len(structured_pts)} points on {len(surfaces)} surfaces "
          f"(1 floor + 4 walls + {len(BOXES)} boxes x 5 faces), +-{JITTER_M*1000:.0f} mm normal jitter")
    print(f"[make_synthetic] surface point allocation (proportional to area): "
          f"floor={counts[0]}, walls={sum(counts[1:5])}, boxes={sum(counts[5:])}")
    print(f"[make_synthetic] wrote {structured_out} ({structured_bytes} bytes, labeled SYNTHETIC)")
    print(f"[make_synthetic] SYNTHETIC pathological cube (seed={args.seed}): "
          f"{len(pathological_pts)} points, uniform random, cube half-extent={cube_half_m:.3f} m "
          f"(matches the structured tile's own bounding cube)")
    print(f"[make_synthetic] wrote {pathological_out} ({pathological_bytes} bytes, labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
