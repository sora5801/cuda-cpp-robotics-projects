#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.05
(KD-tree or LBVH construction + KNN/radius search on GPU).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project needs a LiDAR-scale point cloud (N ~ 200k points) or a
neighbor-search engine to build a tree over, PLUS a set of query points
designed to make the KNN-vs-fixed-radius engineering tradeoff (this
project's whole teaching point) visible and MEASURABLE, not just assertable.
Both can be synthesized analytically with exact control over density, so
this script is the sole source of the committed sample under
../data/sample/ — no download, license-clean, reproducible bit-for-bit from
a fixed seed.

The point cloud: the same 16-beam spinning-LiDAR room scan project 02.01
pioneered in this domain (cited, reimplemented here — NOT imported, per
CLAUDE.md's self-containment rule: every project stays individually
buildable, so shared generation logic is copied and cited, not linked),
PLUS two regions placed and sized specifically for THIS project's KNN/
radius contrast (different numbers than 02.01's own adversarial regions,
which were tuned for hashing/voxelization instead):

  * A DENSE cluster: thousands of points crammed into a 1.2 m cube. A
    radius search at kRadiusM=0.5 m centered on this cluster returns
    HUNDREDS of neighbors — the "fixed radius is cheap AND correct" case
    fixed-radius voxel hashing was built for.
  * A SPARSE, isolated region: a handful of points 3 m apart, floating in
    open space at least 1 m from any surface. A radius search at the SAME
    r=0.5 m centered in this region returns ZERO neighbors — exactly the
    physically-motivated failure THEORY.md derives from the sensor's
    1/r^2 density falloff (real LiDAR points get sparser with range, so
    ANY fixed radius is eventually wrong somewhere in every scan). KNN
    (K=8), by contrast, ALWAYS returns exactly 8 neighbors here too — just
    farther away — which is the entire point of building a structure (the
    LBVH) that can answer "how far do I need to look" adaptively instead
    of committing to one radius in advance.

The beam model (cited, not reinvented): 01.18's THEORY.md derives a
16-beam, -15deg..+15deg-in-2deg-steps spinning LiDAR direction formula,

    d = (cos(el) cos(az), cos(el) sin(az), sin(el))

reused here verbatim, exactly as project 02.01's make_synthetic.py already
reuses it (see that project's module docstring for the original 01.18
citation) — this script's beam table, room geometry, and ray-casting
functions are a close structural copy of 02.01's generator, RETUNED here
with this project's own dense/sparse region parameters and EXTENDED with
this project's own query-point generation (novel to this project).

Query generation (novel to this project; not present in any cited sibling)
----------------------------------------------------------------------------
2000 total queries, in three FIXED, documented blocks (main.cu and this
script share these index boundaries as a data-layout contract, recorded in
the file header written below):

  [0, 2)     — the two DESIGNED queries the density_contrast gate reads by
               explicit index: query 0 = the dense cluster's exact center
               (guaranteed hundreds of radius-search hits); query 1 = a
               point deliberately placed BETWEEN two sparse-region grid
               cells (>1 m from every sparse point, >1 m from every scanned
               surface, >1 m from the dense cluster) so its TRUE nearest
               neighbors are all well outside kRadiusM — a genuine "radius
               search finds nothing" case, not an accidental one.
  [2, 1000)  — 998 SELF-QUERY points: a deterministic stride sample of the
               point cloud itself (query coordinates == existing point
               coordinates). These exercise the K=0-distance / duplicate-
               key-adjacent edge of both search types.
  [1000, 2000) — 1000 GRID queries: a systematic x/y grid at two heights
               spanning the room, for broad, unbiased spatial coverage
               (traversal-stats and hash-vs-BVH agreement want queries that
               are NOT suspiciously aligned with the point cloud's own
               structure).

The first 1000 queries (indices [0,1000), i.e. the two designed queries
plus the full self-query block) are the documented "Q=1000 sampled
queries" GATE brute_force_anchor in main.cu checks against an O(N*Q)
linear-scan oracle — small enough to run in a few seconds on one CPU core,
large enough to be a real statistical net, and deliberately overlapping the
two hand-designed edge cases so the anchor gate is not just "typical case".

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
# Deterministic RNG: xorshift32 (stdlib-only, no `random` module — the repo's
# fixed-seed convention). Identical algorithm to 02.01/11.01/08.01's device-
# side generators and to 02.01's own make_synthetic.py, reused here verbatim
# (cited, not reinvented) so a learner sees the SAME three-line RNG core in
# every corner of this repository, CUDA C++ or Python alike.
# ===========================================================================
class Xorshift32:
    """32-bit xorshift PRNG (Marsaglia 2003). See 02.01's make_synthetic.py
    for the full rationale (matching this repo's CUDA device-side generator
    family instead of Python's Mersenne-Twister `random` module)."""

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
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform01()


DEFAULT_SEED = 42  # repo convention; no meaning beyond tradition

# ===========================================================================
# Scene geometry (meters, LiDAR sensor frame, sensor at the origin) — a
# structural copy of 02.01's room (see that project's make_synthetic.py for
# the full derivation of every constant below; reused verbatim here since
# this project's teaching point is neighbor SEARCH, not scene design, and a
# realistic scene is a means to that end, not this project's own subject).
# ===========================================================================
SENSOR_HEIGHT_M = 1.5
ROOM_HALF_M = 8.0
WALL_TOP_M = 1.5
MAX_RANGE_M = 20.0

BOXES = [
    (3.0, 2.0, 0.4, 0.4, 0.8),
    (-2.5, -3.0, 0.3, 0.3, 1.2),
    (5.0, -4.0, 0.5, 0.35, 0.6),
]

BEAM_ELEV_DEG = list(range(-15, 16, 2))
NUM_BEAMS = len(BEAM_ELEV_DEG)
assert NUM_BEAMS == 16, "beam table must match 01.18's cited 16-beam model exactly"

AZIMUTH_STEPS = 2560
REVOLUTIONS = 6
RANGE_NOISE_M = 0.015

# ---------------------------------------------------------------------------
# This project's OWN adversarial regions (parameters retuned for the KNN /
# fixed-radius contrast at kRadiusM=0.5 m — see kernels.cuh's kRadiusM;
# LEAF_M below MUST equal it, and main.cu asserts this at load time, the
# same data/code consistency check 02.01 performs for its own leaf size).
# ---------------------------------------------------------------------------
LEAF_M = 0.5  # must equal kernels.cuh's kRadiusM (the voxel-hash baseline's leaf == the query radius, 02.04's proof)

DENSE_CENTER = (-6.0, 6.0, -1.0)
DENSE_HALF_M = 0.60          # 1.2 m cube; a radius-0.5 sphere from the center fits entirely inside it
N_DENSE = 4000                # -> ~1200 points expected within r=0.5 of the center at this density (measured at generation time)

SPARSE_Z_M = -0.5              # 1.0 m above the floor (floor at z=-SENSOR_HEIGHT_M=-1.5): well clear of any surface
SPARSE_GRID_STEP_M = 3.0       # >> 2*kRadiusM: guarantees sparse points are NOT each other's radius-neighbors
SPARSE_GRID_HALF_CELLS = 2     # cells range -2..2 -> 5x5 = 25 candidate cells, safely inside the walls (<=6 m)
N_SPARSE = 20
SPARSE_JITTER_FRAC = 0.05      # within-cell jitter, tiny relative to the 3 m spacing

N_SELF_QUERY = 998
N_GRID_QUERY = 1000
N_QUERIES_TOTAL = 2 + N_SELF_QUERY + N_GRID_QUERY   # 2000
N_BRUTE_FORCE_ANCHOR = 2 + N_SELF_QUERY             # 1000: the two designed + all self-query points


# ---------------------------------------------------------------------------
# Ray-scene intersection — a structural copy of 02.01's ray_box_intersect /
# ray_intersect_scene (cited; see that project's make_synthetic.py for the
# full slab-method derivation). Reproduced here, not imported, per the
# self-containment rule (CLAUDE.md section 4): every project stays
# individually buildable from its OWN scripts/ folder.
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


def ray_intersect_scene(dx, dy, dz):
    best_t = None
    if dz < -1e-9:
        t = (-SENSOR_HEIGHT_M - 0.0) / dz
        if t > 1e-6:
            x, y = dx * t, dy * t
            if abs(x) <= ROOM_HALF_M and abs(y) <= ROOM_HALF_M:
                best_t = t if best_t is None else min(best_t, t)

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

    for (cx, cy, hx, hy, height) in BOXES:
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        z0, z1 = -SENSOR_HEIGHT_M, -SENSOR_HEIGHT_M + height
        t = ray_box_intersect(0.0, 0.0, 0.0, dx, dy, dz, x0, x1, y0, y1, z0, z1)
        if t is not None:
            best_t = t if best_t is None else min(best_t, t)

    if best_t is not None and best_t <= MAX_RANGE_M:
        return best_t
    return None


def build_beam_scan(rng: Xorshift32):
    """Cast REVOLUTIONS accumulated sweeps of NUM_BEAMS x AZIMUTH_STEPS rays
    against the analytic scene (02.01 lineage, cited). Returns (points, n_rays_cast)."""
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
                    continue
                r = t + rng.uniform(-RANGE_NOISE_M, RANGE_NOISE_M)
                if r <= 1e-6:
                    r = t
                points.append((dx * r, dy * r, dz * r))
    return points, n_rays_cast


def build_dense_cluster(rng: Xorshift32):
    """N_DENSE points uniformly jittered inside a small cube — this
    project's OWN 'radius search wins' adversarial region (module docstring)."""
    points = []
    cx, cy, cz = DENSE_CENTER
    for _ in range(N_DENSE):
        x = cx + rng.uniform(-DENSE_HALF_M, DENSE_HALF_M)
        y = cy + rng.uniform(-DENSE_HALF_M, DENSE_HALF_M)
        z = cz + rng.uniform(-DENSE_HALF_M, DENSE_HALF_M)
        points.append((x, y, z))
    return points


def build_sparse_region(rng: Xorshift32):
    """N_SPARSE points on a coarse grid (module docstring's 'radius search
    fails, KNN doesn't' adversarial region). Returns (points, cell_list) —
    cell_list is kept so the query generator can place query 1 BETWEEN
    cells rather than accidentally on top of one."""
    cells = [(i, j) for i in range(-SPARSE_GRID_HALF_CELLS, SPARSE_GRID_HALF_CELLS + 1)
                    for j in range(-SPARSE_GRID_HALF_CELLS, SPARSE_GRID_HALF_CELLS + 1)]
    for k in range(len(cells) - 1, 0, -1):
        jx = int(rng.uniform01() * (k + 1))
        if jx > k:
            jx = k
        cells[k], cells[jx] = cells[jx], cells[k]
    chosen = cells[:N_SPARSE]

    points = []
    jitter = SPARSE_JITTER_FRAC * SPARSE_GRID_STEP_M
    for (ci, cj) in chosen:
        x = ci * SPARSE_GRID_STEP_M + rng.uniform(-jitter, jitter)
        y = cj * SPARSE_GRID_STEP_M + rng.uniform(-jitter, jitter)
        z = SPARSE_Z_M + rng.uniform(-jitter, jitter)
        points.append((x, y, z))
    return points, chosen


def build_queries(rng: Xorshift32, points_normal, points_dense, points_sparse, sparse_cells):
    """Build the fixed 2000-query set: 2 designed + 998 self-query + 1000
    grid (module docstring's three blocks, in this exact order)."""
    all_points = points_normal + points_dense + points_sparse
    n_total = len(all_points)

    queries = []

    # -- Block 1: the two DESIGNED queries -----------------------------------
    queries.append(DENSE_CENTER)   # query 0: dense-cluster center

    # query 1: the "empty" cell exactly BETWEEN four chosen sparse cells —
    # (ci+0.5, cj+0.5) in cell units, scaled by the grid step — guaranteed
    # >= 0.5*sqrt(2)*STEP =~ 2.12 m from the nearest LATTICE point regardless
    # of which cells were chosen, comfortably clear of both jitter and the
    # walls (SPARSE_GRID_HALF_CELLS keeps every cell center within 6 m).
    queries.append((0.5 * SPARSE_GRID_STEP_M, 0.5 * SPARSE_GRID_STEP_M, SPARSE_Z_M))

    # -- Block 2: 998 self-query points (deterministic stride sample) -------
    stride = max(1, n_total // N_SELF_QUERY)
    self_pts = []
    idx = 0
    while len(self_pts) < N_SELF_QUERY and idx < n_total:
        self_pts.append(all_points[idx])
        idx += stride
    while len(self_pts) < N_SELF_QUERY:   # pad (only possible if n_total < N_SELF_QUERY, never true at this scale)
        self_pts.append(all_points[len(self_pts) % n_total])
    queries.extend(self_pts)

    # -- Block 3: 1000 systematic grid queries, two heights ------------------
    nx, ny = 25, 20
    heights = (-0.8, 0.5)   # near-floor and mid-air, so both surface-adjacent and open-space queries are covered
    grid_pts = []
    for h in heights:
        for ix in range(nx):
            x = -7.5 + 15.0 * ix / (nx - 1)
            for iy in range(ny):
                y = -7.5 + 15.0 * iy / (ny - 1)
                grid_pts.append((x, y, h))
    assert len(grid_pts) == N_GRID_QUERY
    queries.extend(grid_pts)

    assert len(queries) == N_QUERIES_TOTAL
    return queries


def write_binary_sample(out_path: Path, points_normal, points_dense, points_sparse, queries):
    """Write the committed sample as a small fixed binary format:

        bytes  0.. 7  magic       b'LBVHSCN1' (8 bytes)
        bytes  8..23  int32 x4    n_points, n_beam, n_dense, n_sparse
        bytes 24..27  float32     radius_m (must equal kernels.cuh kRadiusM)
        bytes 28..43  int32 x4    n_queries, idx_dense_query, idx_sparse_query, n_anchor
        bytes 44..    float32 x (n_points*3)   xyz, meters, sensor frame:
                                                [0,n_beam) beam scan,
                                                [n_beam,n_beam+n_dense) dense cluster,
                                                [..,n_points) sparse region
                      float32 x (n_queries*3)  xyz, meters, sensor frame (query block order per module docstring)

    Every field is written with an EXPLICIT little-endian struct format
    (never a raw fwrite of a C struct), matching 02.01/02.04's precedent.
    """
    n_beam = len(points_normal)
    n_dense = len(points_dense)
    n_sparse = len(points_sparse)
    n_points = n_beam + n_dense + n_sparse
    n_queries = len(queries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        f.write(b'LBVHSCN1')
        f.write(struct.pack('<iiii', n_points, n_beam, n_dense, n_sparse))
        f.write(struct.pack('<f', LEAF_M))
        f.write(struct.pack('<iiii', n_queries, 0, 1, N_BRUTE_FORCE_ANCHOR))  # designed-query indices are fixed: 0, 1

        flat_pts = []
        for (x, y, z) in points_normal:
            flat_pts.extend((x, y, z))
        for (x, y, z) in points_dense:
            flat_pts.extend((x, y, z))
        for (x, y, z) in points_sparse:
            flat_pts.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat_pts)}f', *flat_pts))

        flat_q = []
        for (x, y, z) in queries:
            flat_q.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat_q)}f', *flat_q))

    return n_points, n_beam, n_dense, n_sparse, n_queries


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / 'data' / 'sample' / 'lbvh_scan.bin'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out,
                        help='output binary path (default: ../data/sample/lbvh_scan.bin)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    points_normal, n_rays_cast = build_beam_scan(rng)
    points_dense = build_dense_cluster(rng)
    points_sparse, sparse_cells = build_sparse_region(rng)
    queries = build_queries(rng, points_normal, points_dense, points_sparse, sparse_cells)

    n_points, n_beam, n_dense, n_sparse, n_queries = write_binary_sample(
        args.out, points_normal, points_dense, points_sparse, queries)

    # A quick honesty check on the two designed queries, printed so the
    # console output itself documents the density contrast this data was
    # built to demonstrate (main.cu's GATE density_contrast re-measures this
    # for real via the actual GPU/CPU search paths; this is just a sanity
    # preview using a trivial O(n) scan at generation time).
    def count_within(center, radius, pts):
        r2 = radius * radius
        return sum(1 for (x, y, z) in pts
                   if (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= r2)

    all_pts = points_normal + points_dense + points_sparse
    dense_hits = count_within(queries[0], LEAF_M, all_pts)
    sparse_hits = count_within(queries[1], LEAF_M, all_pts)

    hit_rate = 100.0 * n_beam / n_rays_cast if n_rays_cast else 0.0
    print(f"[make_synthetic] SYNTHETIC 16-beam spinning-LiDAR scan (seed={args.seed}, "
          f"{REVOLUTIONS} revolutions x {NUM_BEAMS} beams x {AZIMUTH_STEPS} az steps "
          f"= {n_rays_cast} rays cast, {n_beam} valid returns, {hit_rate:.1f}% hit rate)")
    print(f"[make_synthetic] + {n_dense} DENSE-cluster points (1.2 m cube at {DENSE_CENTER})")
    print(f"[make_synthetic] + {n_sparse} SPARSE-region points ({SPARSE_GRID_STEP_M:.1f} m grid spacing, z={SPARSE_Z_M} m)")
    print(f"[make_synthetic] {n_queries} queries: 2 designed + {N_SELF_QUERY} self-query + {N_GRID_QUERY} grid "
          f"(brute-force anchor subset = first {N_BRUTE_FORCE_ANCHOR})")
    print(f"[make_synthetic] preview at r={LEAF_M} m (generation-time O(n) scan, NOT the gated measurement): "
          f"dense query -> {dense_hits} points within r; sparse query -> {sparse_hits} points within r")
    print(f"[make_synthetic] wrote {args.out} ({n_points} points + {n_queries} queries, labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
