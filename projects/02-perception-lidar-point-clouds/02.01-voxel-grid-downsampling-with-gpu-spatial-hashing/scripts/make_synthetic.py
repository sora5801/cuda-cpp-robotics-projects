#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.01
(Voxel-grid downsampling with GPU spatial hashing).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project needs a LiDAR-scale point cloud (N ~ 200k-500k points) with
real spatial structure: near-field oversampling, a scene made of surfaces
(not a filled volume), and two deliberately adversarial regions that stress
a hash table in opposite ways. All of that can be synthesized analytically
with exact ground truth, so this script is the sole source of the committed
sample under ../data/sample/ — no download, license-clean, reproducible
bit-for-bit from a fixed seed (CLAUDE.md paragraph 12).

The scene: a 16 m x 16 m room, sensor frame
--------------------------------------------
Everything is generated directly in the LiDAR SENSOR's own frame (origin at
the sensor, +x forward, +z up — CLAUDE.md paragraph 12 right-handed
convention) exactly as raw LiDAR returns arrive on the wire (PRACTICE.md
section 1 describes the real packet format this stands in for); a
downstream localizer would transform these into a world/map frame, which is
out of this project's scope (see 02.06 ICP for exactly that next stage).

  * Ground: an infinite plane at z = -SENSOR_HEIGHT_M, clipped to the room
    footprint |x|,|y| <= ROOM_HALF_M (beyond the walls, "ground" does not
    exist in this scene).
  * Four walls at x = +-ROOM_HALF_M, y = +-ROOM_HALF_M, spanning
    z in [-SENSOR_HEIGHT_M, WALL_TOP_M] (floor to a 3 m physical wall
    height). There is deliberately NO CEILING: any beam whose elevation
    carries it over the wall top before it reaches the wall plane escapes
    to open sky and returns NOTHING — a physically honest dropout pattern
    (real LiDAR reports no return when nothing is in range), not an
    artificial thinning.
  * Three axis-aligned boxes standing on the floor (obstacles) — see BOXES
    below for exact placement/size.

The beam model (cited, not reinvented): 01.18's THEORY.md derives a
16-beam, -15deg..+15deg-in-2deg-steps spinning LiDAR direction formula,

    d = (cos(el) cos(az), cos(el) sin(az), sin(el))

reused here verbatim (NUM_BEAMS / BEAM_ELEV_DEG below reproduce that exact
beam table) — see ../../01-perception-cameras-vision/01.18-depth-completion/THEORY.md
"Why LiDAR is sparse".

Why 6 accumulated revolutions, not one (an honest scoping note)
-----------------------------------------------------------------
A single revolution of a realistic 16-beam unit at ~0.14 deg azimuth
resolution (AZIMUTH_STEPS below) produces roughly 16*2560 =~ 41k candidate
rays per sweep — far short of the N ~ 200k-500k this catalog bullet scopes
the project at, and finer azimuth resolution alone would be UNREALISTIC for
a 16-beam mechanical unit. Instead this script accumulates REVOLUTIONS
independent sweeps of the SAME static scene, each with its own independent
per-return range noise (RANGE_NOISE_M, matching project 11.01's
RANGE_NOISE_BASE_M order of magnitude) — precisely how a real robot builds
a locally-dense "submap" by integrating a short dwell of LiDAR spins before
downsampling it (the exact motivating scenario for voxel-grid downsampling
as LiDAR pipelines' first stage — see README "System context"). This is a
deliberate, documented scope choice (CLAUDE.md section 13), not a claim
that any single real 16-beam sweep contains this many points.

The two adversarial regions (the catalog's "stress hashing" ask)
-------------------------------------------------------------------
Appended AFTER the beam scan, in this fixed order (main.cu and this script
share the resulting index boundaries as a data-layout contract, recorded in
the file header below):

  * DENSE cluster: N_DENSE points crammed into a tiny cube (edge
    2*DENSE_HALF_M = 15 cm, smaller than one 20 cm voxel) — guarantees many
    thousands of points collide into a HANDFUL of voxel keys, stressing the
    hash table's probe-chain length and Method B's segmented-reduction load
    balance (kernels.cu's segmented_reduce_kernel comment names this
    exact cost).
  * SPARSE region: N_SPARSE points scattered on a 1 m grid (5x the voxel
    edge) floating at a fixed height — guarantees each point lands in its
    OWN voxel, exercising the opposite extreme: many nearly-empty probe
    sequences spread across the table's key space.

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
# Deterministic RNG: xorshift32 (stdlib-only, no `random` module — the
# repo's fixed-seed convention, CLAUDE.md paragraph 12). This is the SAME
# algorithm project 11.01's device-side generator uses (see that project's
# kernels.cu "Per-beam deterministic RNG" comment) — reused here as a
# single SEQUENTIAL stream (not per-thread streams: this is a serial,
# one-time offline generator, so stream separation across "threads" is not
# a concern the way it is for 11.01's parallel device kernel).
# ===========================================================================
class Xorshift32:
    """A 32-bit xorshift PRNG (Marsaglia 2003) — three shifts, three XORs,
    full 2^32-1 period for any nonzero seed. Chosen over Python's built-in
    `random` (Mersenne Twister) specifically to match the algorithm this
    repository's CUDA device code uses (kernels.cu's own generators,
    project 11.01's, project 08.01's) — a learner reading across projects
    sees the SAME three-line RNG core everywhere, in Python or CUDA C++.
    """

    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        if s == 0:
            s = 1  # xorshift32 is degenerate (stays 0 forever) at seed 0 — same guard kernels.cu uses
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
        """(0,1], never exactly 0 — mirrors kernels.cu's uniform01(): keep the
        top 24 bits (float32 has a 24-bit significand) and add a half-ULP
        bias so the result never lands exactly on 0.0."""
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform01()


DEFAULT_SEED = 42  # repo convention (CLAUDE.md paragraph 12); no special meaning beyond tradition

# ===========================================================================
# Scene geometry (all meters, LiDAR sensor frame — sensor at the origin).
# Every constant here is ALSO documented in ../data/README.md; the leaf
# size LEAF_M must equal kernels.cuh's kVoxelLeafM exactly — main.cu
# asserts this at load time (a data/code consistency check, not a
# coincidence) — see the file-header write_header() below.
# ===========================================================================
LEAF_M = 0.20              # voxel edge length L (must match kernels.cuh kVoxelLeafM)
SENSOR_HEIGHT_M = 1.5      # sensor mount height above the floor
ROOM_HALF_M = 8.0          # walls at x,y = +-ROOM_HALF_M -> a 16x16 m room
WALL_TOP_M = 1.5           # walls span z in [-SENSOR_HEIGHT_M, WALL_TOP_M] (a 3 m physical wall)
MAX_RANGE_M = 20.0         # sensor spec ceiling (room diagonal ~11.3 m always stays under this)

# Three obstacle boxes standing on the floor: (center_x, center_y, half_x, half_y, height).
# Sizes/positions are illustrative "warehouse clutter" — chosen to sit
# comfortably inside the room and away from each other and the adversarial
# regions below, no further physical meaning.
BOXES = [
    (3.0, 2.0, 0.4, 0.4, 0.8),
    (-2.5, -3.0, 0.3, 0.3, 1.2),
    (5.0, -4.0, 0.5, 0.35, 0.6),
]

# 16-beam elevation table, cited verbatim from 01.18's beam model (module
# docstring above): -15 .. +15 degrees in 2-degree steps -> 16 beams.
BEAM_ELEV_DEG = list(range(-15, 16, 2))
NUM_BEAMS = len(BEAM_ELEV_DEG)
assert NUM_BEAMS == 16, "beam table must match 01.18's cited 16-beam model exactly"

AZIMUTH_STEPS = 2560       # ~0.1406 deg/step — realistic single-sweep 16-beam resolution (module docstring)
REVOLUTIONS = 6            # accumulated sweeps -> the "short dwell submap" scope decision (module docstring)
RANGE_NOISE_M = 0.015      # +-15 mm per-return range noise, matching 11.01's RANGE_NOISE_BASE_M order of magnitude

# Adversarial DENSE cluster: many points, one tiny cube (< 1 voxel edge).
DENSE_CENTER = (6.0, 6.0, -1.2)
DENSE_HALF_M = 0.075       # 15 cm cube edge -- smaller than LEAF_M (0.20 m): mostly 1, at most a few voxels
N_DENSE = 3000

# Adversarial SPARSE region: isolated points on a coarse (1 m) grid, 5x the
# voxel edge apart -- guarantees each lands in its own voxel. Floating at a
# fixed height (not on any modeled surface) -- an unambiguous synthetic
# marker, not meant to resemble a real return.
SPARSE_Z_M = -0.5
SPARSE_GRID_STEP_M = 1.0
SPARSE_GRID_HALF_CELLS = 7   # cells range -7..+7 -> 15x15 = 225 candidate cells
N_SPARSE = 150
SPARSE_EXCLUDE_RADIUS_CELLS = 1  # keep the grid away from the DENSE cluster's own cell


# ---------------------------------------------------------------------------
# ray_box_intersect — the standard SLAB METHOD for an axis-aligned box: for
# each of the 3 axes, compute the ray-parameter interval [t1,t2] during
# which the ray is between that axis's two bounding planes, then intersect
# (max of all t1's, min of all t2's) across axes. If the resulting interval
# is non-empty and lies (at least partly) ahead of the ray origin, the ray
# hits the box at its entry parameter.
#
# Returns: the smallest POSITIVE hit parameter t, or None if the ray misses
# the box or the box is entirely behind the origin.
# ---------------------------------------------------------------------------
def ray_box_intersect(ox, oy, oz, dx, dy, dz, x0, x1, y0, y1, z0, z1):
    t_min, t_max = -math.inf, math.inf
    for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1), (oz, dz, z0, z1)):
        if abs(d) < 1e-12:
            # Ray parallel to this pair of slab planes: either always inside
            # the slab (o within [lo,hi], interval unchanged) or never
            # (outside, entire ray misses the box on this axis alone).
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
        return None  # box is entirely behind the ray origin
    t_hit = t_min if t_min > 1e-6 else t_max  # origin may be inside the box on the near face
    return t_hit if t_hit > 1e-6 else None


# ---------------------------------------------------------------------------
# ray_intersect_scene — cast one ray from the sensor origin and return the
# range to the FIRST surface it hits (ground, a wall, or a box), or None if
# it hits nothing within MAX_RANGE_M (an honest "no return").
#
# All five candidate surface families (ground + 4 walls + boxes) are tested
# independently; the smallest POSITIVE, IN-BOUNDS parameter across all of
# them is the true first hit — exactly the same "take the nearest valid
# candidate" logic a real depth-buffer / raycaster uses (project 11.01's
# BVH raycaster does the analogous search over a full triangle mesh; this
# scene is small enough that per-ray brute force over ~8 primitives is the
# honest, simplest-possible choice here — CLAUDE.md "teaching beats
# cleverness").
# ---------------------------------------------------------------------------
def ray_intersect_scene(dx, dy, dz):
    best_t = None

    # --- ground: z = -SENSOR_HEIGHT_M, clipped to the room footprint -------
    if dz < -1e-9:  # only a DOWNWARD-pointing ray can reach the floor
        t = (-SENSOR_HEIGHT_M - 0.0) / dz
        if t > 1e-6:
            x, y = dx * t, dy * t
            if abs(x) <= ROOM_HALF_M and abs(y) <= ROOM_HALF_M:
                best_t = t if best_t is None else min(best_t, t)

    # --- four walls ----------------------------------------------------------
    R = ROOM_HALF_M
    wall_candidates = []
    if dx > 1e-9:
        wall_candidates.append(((R - 0.0) / dx, 'x'))
    if dx < -1e-9:
        wall_candidates.append(((-R - 0.0) / dx, 'x'))
    if dy > 1e-9:
        wall_candidates.append(((R - 0.0) / dy, 'y'))
    if dy < -1e-9:
        wall_candidates.append(((-R - 0.0) / dy, 'y'))
    for t, axis in wall_candidates:
        if t <= 1e-6:
            continue
        z = dz * t
        if not (-SENSOR_HEIGHT_M <= z <= WALL_TOP_M):
            continue
        if axis == 'x':
            y = dy * t
            if abs(y) <= R:
                best_t = t if best_t is None else min(best_t, t)
        else:
            x = dx * t
            if abs(x) <= R:
                best_t = t if best_t is None else min(best_t, t)

    # --- boxes -----------------------------------------------------------------
    for (cx, cy, hx, hy, height) in BOXES:
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        z0, z1 = -SENSOR_HEIGHT_M, -SENSOR_HEIGHT_M + height
        t = ray_box_intersect(0.0, 0.0, 0.0, dx, dy, dz, x0, x1, y0, y1, z0, z1)
        if t is not None:
            best_t = t if best_t is None else min(best_t, t)

    if best_t is not None and best_t <= MAX_RANGE_M:
        return best_t
    return None  # no surface in range: an honest dropout (e.g. a beam that clears the wall top -- no ceiling)


def build_beam_scan(rng: Xorshift32):
    """Cast REVOLUTIONS accumulated sweeps of NUM_BEAMS x AZIMUTH_STEPS rays
    against the analytic scene, with independent per-return range noise.
    Returns (points, n_rays_cast, n_hits) — points is a flat list of
    (x,y,z) tuples in the fixed revolution-major -> beam-major -> azimuth-
    minor emission order."""
    points = []
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
                t = ray_intersect_scene(dx, dy, dz)
                if t is None:
                    continue  # no return this ray -- physically honest dropout, not injected noise
                r = t + rng.uniform(-RANGE_NOISE_M, RANGE_NOISE_M)
                if r <= 1e-6:
                    r = t  # guard against a pathological near-zero noisy range (never triggered in practice: nearest surface is ~1 m+)
                points.append((dx * r, dy * r, dz * r))
    return points, n_rays_cast


def build_dense_cluster(rng: Xorshift32):
    """N_DENSE points uniformly jittered inside a tiny cube -- the 'many
    points, few voxels' adversarial case (module docstring)."""
    points = []
    cx, cy, cz = DENSE_CENTER
    for _ in range(N_DENSE):
        x = cx + rng.uniform(-DENSE_HALF_M, DENSE_HALF_M)
        y = cy + rng.uniform(-DENSE_HALF_M, DENSE_HALF_M)
        z = cz + rng.uniform(-DENSE_HALF_M, DENSE_HALF_M)
        points.append((x, y, z))
    return points


def build_sparse_region(rng: Xorshift32):
    """N_SPARSE points on a 1 m grid (5x the voxel edge) -- the 'each point
    its own voxel' adversarial case (module docstring). Grid-cell selection
    is shuffled with a deterministic Fisher-Yates draw from the SAME
    xorshift32 stream (never Python's `random.shuffle`, to keep the whole
    file's determinism story to one RNG algorithm)."""
    cells = [(i, j) for i in range(-SPARSE_GRID_HALF_CELLS, SPARSE_GRID_HALF_CELLS + 1)
                    for j in range(-SPARSE_GRID_HALF_CELLS, SPARSE_GRID_HALF_CELLS + 1)]
    # Keep the sparse grid away from the dense cluster's own cell so the two
    # adversarial regions never accidentally overlap into the same voxel.
    dense_cell = (round(DENSE_CENTER[0] / SPARSE_GRID_STEP_M), round(DENSE_CENTER[1] / SPARSE_GRID_STEP_M))
    cells = [c for c in cells
            if abs(c[0] - dense_cell[0]) > SPARSE_EXCLUDE_RADIUS_CELLS
            or abs(c[1] - dense_cell[1]) > SPARSE_EXCLUDE_RADIUS_CELLS]

    # Fisher-Yates shuffle using the shared xorshift32 stream: for k from
    # the end down to 1, swap cells[k] with a uniformly-random cells[0..k].
    for k in range(len(cells) - 1, 0, -1):
        j = int(rng.uniform01() * (k + 1))
        if j > k:
            j = k  # guard the (probability~0, but possible at the float boundary) uniform01()==1.0 case
        cells[k], cells[j] = cells[j], cells[k]

    chosen = cells[:N_SPARSE]
    points = []
    jitter = 0.05 * SPARSE_GRID_STEP_M  # small within-cell jitter, well under the 1 m cell spacing
    for (ci, cj) in chosen:
        x = ci * SPARSE_GRID_STEP_M + rng.uniform(-jitter, jitter)
        y = cj * SPARSE_GRID_STEP_M + rng.uniform(-jitter, jitter)
        z = SPARSE_Z_M + rng.uniform(-jitter, jitter)
        points.append((x, y, z))
    return points


def write_binary_sample(out_path: Path, points_normal, points_dense, points_sparse):
    """Write the committed sample as a small fixed binary format:

        bytes  0.. 7  magic       b'VXLSCAN1' (8 bytes, no null terminator)
        bytes  8..23  int32 x4    n_total, n_normal, n_dense, n_sparse
        bytes 24..31  float32 x2  leaf_m, sensor_height_m
        bytes 32..39  int32 x2    num_beams, reserved(0)
        bytes 40..    float32 x (n_total*3)   xyz, meters, sensor "lidar" frame,
                                               in the fixed order
                                               [0,n_normal) = beam scan,
                                               [n_normal,n_normal+n_dense) = dense cluster,
                                               [..,n_total) = sparse region.

    Every field is written with an EXPLICIT little-endian struct format
    (never a raw fwrite of a C struct) so the layout is independent of any
    compiler's padding/alignment rules -- main.cu's loader reads the same
    explicit sequence of primitives back (see its file-header comment).
    """
    n_normal = len(points_normal)
    n_dense = len(points_dense)
    n_sparse = len(points_sparse)
    n_total = n_normal + n_dense + n_sparse

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        f.write(b'VXLSCAN1')
        f.write(struct.pack('<iiii', n_total, n_normal, n_dense, n_sparse))
        f.write(struct.pack('<ff', LEAF_M, SENSOR_HEIGHT_M))
        f.write(struct.pack('<ii', NUM_BEAMS, 0))
        flat = []
        for (x, y, z) in points_normal:
            flat.extend((x, y, z))
        for (x, y, z) in points_dense:
            flat.extend((x, y, z))
        for (x, y, z) in points_sparse:
            flat.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat)}f', *flat))

    return n_total, n_normal, n_dense, n_sparse


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / 'data' / 'sample' / 'lidar_scan.bin'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out,
                        help='output binary path (default: ../data/sample/lidar_scan.bin)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    points_normal, n_rays_cast = build_beam_scan(rng)
    points_dense = build_dense_cluster(rng)
    points_sparse = build_sparse_region(rng)

    n_total, n_normal, n_dense, n_sparse = write_binary_sample(
        args.out, points_normal, points_dense, points_sparse)

    hit_rate = 100.0 * n_normal / n_rays_cast if n_rays_cast else 0.0
    print(f"[make_synthetic] SYNTHETIC 16-beam spinning-LiDAR scan (seed={args.seed}, "
          f"{REVOLUTIONS} accumulated revolutions x {NUM_BEAMS} beams x {AZIMUTH_STEPS} azimuth steps "
          f"= {n_rays_cast} rays cast, {n_normal} valid returns, {hit_rate:.1f}% hit rate)")
    print(f"[make_synthetic] + {n_dense} adversarial DENSE-cluster points "
          f"(cube edge {2*DENSE_HALF_M*100:.0f} cm at {DENSE_CENTER})")
    print(f"[make_synthetic] + {n_sparse} adversarial SPARSE-region points "
          f"({SPARSE_GRID_STEP_M:.1f} m grid spacing, z={SPARSE_Z_M} m)")
    print(f"[make_synthetic] wrote {args.out} ({n_total} points total, "
          f"{n_total*3*4 + 40} bytes, leaf_m={LEAF_M}, labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
