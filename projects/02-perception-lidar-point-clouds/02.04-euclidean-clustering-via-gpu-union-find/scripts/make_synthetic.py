#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.04
(Euclidean clustering via GPU union-find / connected components).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project needs a scene that makes FOUR specific claims about Euclidean
clustering demonstrably true or false, on purpose:

  (a) two objects farther than d apart MUST stay two clusters (the
      resolution test — SEP_A / SEP_B below),
  (b) two objects bridged by a thin points path closer than d together MUST
      merge into one cluster — the well-known "chaining" failure mode of
      single-linkage clustering, taught here HONESTLY rather than hidden
      (CHAIN_A / CHAIN_B / the bridge points),
  (c) a long, thin, curved chain of points (the "snake") must have a large
      GRAPH diameter, so that label propagation (which needs O(diameter)
      sweeps) and union-find (which needs only O(log diameter), thanks to
      path halving) visibly diverge in sweep count on the SAME data, and
  (d) scattered points, each isolated (farther than d from everything else),
      must be rejected by min-size filtering as noise.

No public LiDAR dataset ships these four properties by construction with
known ground truth, so synthetic generation — with the SAME distance rule
(d) the C++ pipeline uses — is not just the default here, it is the only
way to get an exact, checkable ground truth at all (CLAUDE.md paragraph 8).

The scene, in the LiDAR "sensor"/local frame (x forward, y left, z up,
CLAUDE.md paragraph 12)
--------------------------------------------------------------------------
A GROUND layer (a flat, gently-jittered plane, x in [GX0,GX1], y in
[GY0,GY1]) plays the role of "everything project 02.03 (ground segmentation)
would have already removed" — it is written to the sample file for context
and for the demo's topview artifacts, but it is NEVER fed to the clustering
pipeline (main.cu loads it separately and only visualizes it; see README
"System context" for why this project's real input starts one stage later
than a raw scan). On top of the ground, several NON-GROUND objects stand at
various x,y positions, kept in clearly separated Y-BANDS so unrelated
objects never accidentally touch each other:

  Y in [-9,-5]   : five "documented obstacles" (boxes/poles) at varied,
                   generous separations — read this project's README's
                   "cars/boxes/poles" framing loosely: these are honestly
                   SMALL, filled point blocks (see OBJECT_FILL_SPACING_M's
                   docstring for the scoping decision), not raycast LiDAR
                   returns of full-size vehicles.
  Y in [-2, 2]    : the SEPARATION test pair (SEP_A, SEP_B) — edge-to-edge
                   gap = d + 0.10 m.
  Y in [3, 6]     : the CHAINING test pair (CHAIN_A, CHAIN_B) plus the
                   bridge — point spacing along the bridge = d - 0.10 m.
  Y in [-18,-13]  : the LONG SNAKE — a gently curved arc of closely (but not
                   TOO closely, see the spacing derivation below) spaced
                   points, engineered to have a large GRAPH diameter. Its
                   own Y-band sits clear below the generic obstacles' -9..-5
                   band, with a comfortable gap.
  Y in [12, 14]   : scattered NOISE points, each isolated by construction.

Note: the ground layer's own Y extent (GY0..GY1 below) does NOT need to
bound every non-ground feature — the demo's topview artifacts derive their
world-to-pixel window from the UNION of every rendered point (main.cu), not
from the ground layer alone, so a feature placed slightly outside the
ground rectangle (the snake's arc, by construction) still renders in full.

Ground-truth cluster ids (build_truth_clusters below) are computed HERE,
with the SAME distance rule and the SAME 27-cell-voxel-stencil algorithm the
C++ pipeline implements (a pure-Python spatial hash + union-find, small
enough at this scene's point count to run in well under a second) — see
that function's docstring for why this makes the truth an honest
"single-linkage ground truth", not an independently-invented one.

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
# Deterministic RNG: xorshift32 (stdlib-only — CLAUDE.md paragraph 12's
# fixed-seed convention). Identical algorithm to 02.01/02.03's generators
# and this project's own kernels.cu device code — cited, reused verbatim.
# ===========================================================================
class Xorshift32:
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
# Pipeline-design constants — MUST match src/kernels.cuh exactly (main.cu
# asserts this at load time; the same discipline 02.01's LEAF_M / kVoxelLeafM
# pairing follows).
# ===========================================================================
D_M = 0.40                 # cluster tolerance == voxel leaf (kClusterToleranceM)
MIN_CLUSTER_SIZE = 5       # kMinClusterSize

# ---------------------------------------------------------------------------
# OBJECT_FILL_SPACING_M — the point spacing used to fill every solid object
# (boxes, poles, the chain boxes). Chosen well under D_M so every object is
# UNAMBIGUOUSLY internally connected (a filled 3-D lattice at this spacing
# gives each interior point roughly 30 neighbors within d, THEORY.md "The
# GPU mapping" derives the exact count this drove kMaxEdgesPerPoint from)
# while staying well above D_M/3 so a point's reach stays LOCAL (no
# accidental long-range shortcuts through a dense block).
#
# Scoping decision (README "Limitations" restates this): objects are FILLED
# synthetic point blocks, not ray-cast LiDAR returns. This project's
# teaching focus is the CLUSTERING algorithm, not sensor simulation (that is
# 02.01/11.01/01.18's job, cited); filled blocks give exact, simple control
# over point count, density, and connectivity for the four designed test
# scenarios above, at the honest cost of looking more like a CAD block than
# a real LiDAR return pattern.
# ---------------------------------------------------------------------------
OBJECT_FILL_SPACING_M = 0.15
JITTER_M = 0.01             # small per-point jitter so grids are not perfectly regular

# Ground layer extent and density (see the module docstring's scene map).
GX0, GX1 = -10.0, 50.0
GY0, GY1 = -15.0, 15.0
GROUND_SPACING_M = 0.10
GROUND_Z_M = 0.0


def sample_filled_box(rng, cx, cy, cz, hx, hy, hz, spacing):
    """A solid axis-aligned box, filled with a jittered lattice of points,
    centered at (cx,cy) with its BASE at z=cz (so a 'box standing on the
    floor' has cz=0 and height 2*hz measured upward — see call sites)."""
    pts = []
    nx = max(1, int(round((2 * hx) / spacing)))
    ny = max(1, int(round((2 * hy) / spacing)))
    nz = max(1, int(round((2 * hz) / spacing)))
    for ix in range(nx + 1):
        x = cx - hx + ix * (2 * hx / nx)
        for iy in range(ny + 1):
            y = cy - hy + iy * (2 * hy / ny)
            for iz in range(nz + 1):
                z = cz + iz * (2 * hz / nz)
                pts.append((x + rng.uniform(-JITTER_M, JITTER_M),
                           y + rng.uniform(-JITTER_M, JITTER_M),
                           z + rng.uniform(-JITTER_M, JITTER_M)))
    return pts


def sample_filled_cylinder(rng, cx, cy, radius, height, spacing, cz=0.0):
    """A solid, upright cylinder (a 'pole'), filled with a jittered lattice:
    a square xy-grid clipped to the circle, repeated up the z axis."""
    pts = []
    nz = max(1, int(round(height / spacing)))
    steps = max(1, int(round((2 * radius) / spacing)))
    for iz in range(nz + 1):
        z = cz + iz * (height / nz)
        for ix in range(-steps, steps + 1):
            x = ix * spacing
            for iy in range(-steps, steps + 1):
                y = iy * spacing
                if x * x + y * y > radius * radius:
                    continue
                pts.append((cx + x + rng.uniform(-JITTER_M, JITTER_M),
                           cy + y + rng.uniform(-JITTER_M, JITTER_M),
                           z + rng.uniform(-JITTER_M, JITTER_M)))
    return pts


def sample_chain_line(rng, p0, p1, spacing):
    """A single-file line of jittered points from p0 to p1 (exclusive of the
    endpoints — callers append it BETWEEN two objects, so the objects' own
    surface points are the effective endpoints)."""
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    n = max(1, int(round(length / spacing)))
    pts = []
    for i in range(1, n):
        t = i / n
        x, y, z = p0[0] + dx * t, p0[1] + dy * t, p0[2] + dz * t
        pts.append((x + rng.uniform(-JITTER_M, JITTER_M),
                   y + rng.uniform(-JITTER_M, JITTER_M),
                   z + rng.uniform(-JITTER_M, JITTER_M)))
    return pts


def sample_arc(rng, center, radius, angle0_rad, angle1_rad, z, spacing):
    """The LONG SNAKE: points along a circular arc at height z, spaced
    `spacing` apart ALONG THE ARC (arc length / spacing ~= point count).

    Why radius/angle are chosen the way build_scene() calls this: two
    points at chain-index difference k are separated, for a low-curvature
    arc, by very close to k*spacing in a straight EUCLIDEAN line too (the
    arc is locally almost straight at these radii) -- so choosing spacing
    JUST UNDER d (0.35 m here, d=0.40 m) means 2*spacing=0.70 m > d: only
    IMMEDIATE chain neighbors (k=1) fall within d of each other. The
    resulting connectivity graph is a simple PATH graph (each interior
    point has exactly 2 neighbors), whose diameter equals point_count-1
    EXACTLY -- the largest possible diameter for a given point budget,
    which is exactly what THEORY.md's O(diameter)-sweep argument for label
    propagation needs to be dramatically, cheaply visible (a few hundred
    points, not thousands). See THEORY.md "The algorithm" for the full
    derivation and the chord-distance check confirming non-adjacent points
    never accidentally fall within d of each other at this radius/angle.
    """
    arc_span = radius * (angle1_rad - angle0_rad)
    n = max(2, int(round(arc_span / spacing)))
    pts = []
    for i in range(n):
        a = angle0_rad + (angle1_rad - angle0_rad) * (i / (n - 1))
        x = center[0] + radius * math.cos(a)
        y = center[1] + radius * math.sin(a)
        pts.append((x + rng.uniform(-JITTER_M, JITTER_M),
                   y + rng.uniform(-JITTER_M, JITTER_M),
                   z + rng.uniform(-JITTER_M, JITTER_M)))
    return pts


def sample_ground(rng):
    pts = []
    nx = int(round((GX1 - GX0) / GROUND_SPACING_M))
    ny = int(round((GY1 - GY0) / GROUND_SPACING_M))
    for ix in range(nx):
        x = GX0 + ix * GROUND_SPACING_M
        for iy in range(ny):
            y = GY0 + iy * GROUND_SPACING_M
            z = GROUND_Z_M + rng.uniform(-0.01, 0.01)   # a few mm of floor roughness
            pts.append((x, y, z))
    return pts


def sample_noise(rng, n_target):
    """Scattered points, each isolated by construction: a coarse 2 m grid
    (>> 2*D_M, so no noise point can ever be within d of ANOTHER noise
    point) in a Y-band (see module docstring) that contains no other
    object, so isolation from every OTHER feature is true by placement,
    not by a runtime collision check."""
    pts = []
    x = GX0 + 2.0
    y_lo, y_hi = 12.0, 14.0
    step = 2.0
    while len(pts) < n_target and x < GX1 - 2.0:
        y = y_lo + rng.uniform01() * (y_hi - y_lo)
        z = 0.3 + rng.uniform(-0.05, 0.05)
        pts.append((x, y, z))
        x += step
    return pts


# ===========================================================================
# build_truth_clusters — the generator's OWN single-linkage connected-
# components computation: a spatial hash (cell size = D_M, matching
# kClusterToleranceM exactly) + the classic 27-cell stencil + union-find
# with path compression, run in pure Python. This IS the same algorithm
# (same distance rule, same neighbor rule) the C++/CUDA pipeline implements
# -- so "truth" here honestly means "single-linkage ground truth", not an
# independently-invented labeling (README/THEORY.md state this precisely:
# the C++ pipeline's job is to reproduce THIS partition, not to discover
# some other notion of "object"). Runs in well under a second at this
# scene's non-ground point count (a few thousand), so no further
# optimization is attempted (CLAUDE.md "teaching beats cleverness").
#
# Returns a list of canonical ids, one per point, where the canonical id of
# a component is the MINIMUM point index in it -- the exact convention the
# GPU union-find (union-by-min) and label-propagation (min-label flooding)
# algorithms both converge to (see kernels.cuh "THE UNION-FIND CHAPTER").
# ===========================================================================
def build_truth_clusters(points, d):
    n = len(points)
    cell = d
    grid = {}
    for i, (x, y, z) in enumerate(points):
        key = (math.floor(x / cell), math.floor(y / cell), math.floor(z / cell))
        grid.setdefault(key, []).append(i)

    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    d2 = d * d
    for i, (x, y, z) in enumerate(points):
        cx, cy, cz = math.floor(x / cell), math.floor(y / cell), math.floor(z / cell)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    bucket = grid.get((cx + dx, cy + dy, cz + dz))
                    if not bucket:
                        continue
                    for j in bucket:
                        if j <= i:
                            continue
                        xj, yj, zj = points[j]
                        ddx, ddy, ddz = x - xj, y - yj, z - zj
                        if ddx * ddx + ddy * ddy + ddz * ddz <= d2:
                            union(i, j)

    return [find(i) for i in range(n)]


def build_scene(rng):
    """Assemble the non-ground point list IN ORDER, tracking the index
    ranges/representative indices main.cu's gates need. Returns
    (points, meta) where meta is a dict of the special indices."""
    points = []
    meta = {}

    def append(pts):
        start = len(points)
        points.extend(pts)
        return start, len(pts)

    # -- five "documented obstacles", generously separated (Y in [-9,-5]) --
    append(sample_filled_box(rng, 5.0, -7.0, 0.0, 0.3, 0.3, 0.35, OBJECT_FILL_SPACING_M))
    append(sample_filled_cylinder(rng, 12.0, -7.0, 0.15, 1.4, OBJECT_FILL_SPACING_M))
    append(sample_filled_box(rng, 20.0, -7.0, 0.0, 1.0, 0.5, 0.3, OBJECT_FILL_SPACING_M))
    append(sample_filled_cylinder(rng, 30.0, -7.0, 0.18, 1.6, OBJECT_FILL_SPACING_M))
    append(sample_filled_box(rng, 40.0, -7.0, 0.0, 0.25, 0.25, 0.4, OBJECT_FILL_SPACING_M))

    # -- separation test: two poles, edge-to-edge gap = D_M + 0.10 m -------
    sep_radius = 0.15
    sep_gap = D_M + 0.10
    sep_a_center_x = 15.0
    sep_b_center_x = sep_a_center_x + 2 * sep_radius + sep_gap
    sep_a_start, sep_a_n = append(sample_filled_cylinder(rng, sep_a_center_x, 0.0, sep_radius, 1.2, OBJECT_FILL_SPACING_M))
    sep_b_start, sep_b_n = append(sample_filled_cylinder(rng, sep_b_center_x, 0.0, sep_radius, 1.2, OBJECT_FILL_SPACING_M))
    meta['sep_a_idx'] = sep_a_start                    # any point inside object A
    meta['sep_b_idx'] = sep_b_start                    # any point inside object B

    # -- chaining test: two boxes bridged by a thin path, spacing < D_M ----
    chain_hx = chain_hy = chain_hz = 0.2
    chain_a_center = (10.0, 4.5, 0.0)
    chain_b_center = (13.0, 4.5, 0.0)
    chain_a_start, _ = append(sample_filled_box(rng, *chain_a_center, chain_hx, chain_hy, chain_hz, OBJECT_FILL_SPACING_M))
    chain_b_start, _ = append(sample_filled_box(rng, *chain_b_center, chain_hx, chain_hy, chain_hz, OBJECT_FILL_SPACING_M))
    meta['chain_a_idx'] = chain_a_start
    meta['chain_b_idx'] = chain_b_start
    bridge_spacing = D_M - 0.10
    bridge_p0 = (chain_a_center[0] + chain_hx, chain_a_center[1], chain_hz)
    bridge_p1 = (chain_b_center[0] - chain_hx, chain_b_center[1], chain_hz)
    append(sample_chain_line(rng, bridge_p0, bridge_p1, bridge_spacing))

    # -- the long snake: a gently curved arc, spacing just under D_M -------
    # A LARGE radius (300 m) with a SMALL angular span (20 deg) keeps the
    # curve gentle (a "thin curved wall", not a tight spiral) while reaching
    # arc length ~= 104.7 m; at spacing 0.35 m that is ~300 points, diameter
    # ~300 hops -- comfortable margin above the snake_convergence gate's
    # 50-sweep label-propagation floor (main.cu), while union-find
    # (O(log diameter)) still converges in a handful of sweeps. The center
    # is placed far below the scene (y0 - radius) so the visible arc sits
    # around y ~= -13..-18, clear of every other feature's Y-band (the
    # closest, the five generic obstacles, sit at y=-7). See sample_arc()'s
    # docstring for the chord-distance argument that this radius/span never
    # lets non-adjacent snake points touch each other.
    snake_spacing = D_M - 0.05
    snake_center_y = -13.0
    snake_radius = 300.0
    snake_start, snake_n = append(sample_arc(rng, center=(25.0, snake_center_y - snake_radius), radius=snake_radius,
                                             angle0_rad=math.radians(90.0 - 10.0),
                                             angle1_rad=math.radians(90.0 + 10.0),
                                             z=0.3, spacing=snake_spacing))
    meta['snake_start_idx'] = snake_start
    meta['snake_count'] = snake_n

    # -- scattered, isolated noise points ------------------------------------
    noise_start, noise_n = append(sample_noise(rng, 80))
    meta['noise_start_idx'] = noise_start
    meta['noise_count'] = noise_n

    return points, meta


def write_binary_sample(out_path: Path, ground_pts, nonground_pts, truth_ids, meta):
    """Write the committed sample:

        bytes  0.. 7   magic       b'CLUSTR01'
        bytes  8..11   int32       n_ground
        bytes 12..15   int32       n_nonground
        bytes 16..19   float32     d_m (== kClusterToleranceM)
        bytes 20..23   int32       min_cluster_size (== kMinClusterSize)
        bytes 24..27   int32       snake_start_idx
        bytes 28..31   int32       snake_count
        bytes 32..35   int32       sep_a_idx
        bytes 36..39   int32       sep_b_idx
        bytes 40..43   int32       chain_a_idx
        bytes 44..47   int32       chain_b_idx
        bytes 48..51   int32       noise_start_idx
        bytes 52..55   int32       noise_count
        bytes 56..59   int32       reserved (0)
        ....           float32[n_ground*3]     ground xyz, meters, LOCAL frame (context only, NOT clustered)
        ....           float32[n_nonground*3]  non-ground xyz, meters (the clustering INPUT)
        ....           int32[n_nonground]      truth_cluster_id (canonical = min index in THIS array)

    Every field is written with an EXPLICIT little-endian struct format
    (never a raw fwrite of a C struct), the same portability reasoning
    02.01's write_binary_sample gives, cited rather than repeated in full.
    """
    n_ground = len(ground_pts)
    n_nonground = len(nonground_pts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        f.write(b'CLUSTR01')
        f.write(struct.pack('<ii', n_ground, n_nonground))
        f.write(struct.pack('<f', D_M))
        f.write(struct.pack('<i', MIN_CLUSTER_SIZE))
        f.write(struct.pack('<ii', meta['snake_start_idx'], meta['snake_count']))
        f.write(struct.pack('<ii', meta['sep_a_idx'], meta['sep_b_idx']))
        f.write(struct.pack('<ii', meta['chain_a_idx'], meta['chain_b_idx']))
        f.write(struct.pack('<ii', meta['noise_start_idx'], meta['noise_count']))
        f.write(struct.pack('<i', 0))

        flat_ground = []
        for (x, y, z) in ground_pts:
            flat_ground.extend((x, y, z))
        if flat_ground:
            f.write(struct.pack(f'<{len(flat_ground)}f', *flat_ground))

        flat_ng = []
        for (x, y, z) in nonground_pts:
            flat_ng.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat_ng)}f', *flat_ng))

        f.write(struct.pack(f'<{len(truth_ids)}i', *truth_ids))

    return n_ground, n_nonground


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / 'data' / 'sample' / 'cluster_scene.bin'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out,
                        help='output binary path (default: ../data/sample/cluster_scene.bin)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    ground_pts = sample_ground(rng)
    nonground_pts, meta = build_scene(rng)
    truth_ids = build_truth_clusters(nonground_pts, D_M)

    n_ground, n_nonground = write_binary_sample(args.out, ground_pts, nonground_pts, truth_ids, meta)

    num_true_components = len(set(truth_ids))
    sizes = {}
    for t in truth_ids:
        sizes[t] = sizes.get(t, 0) + 1
    num_reported = sum(1 for s in sizes.values() if s >= MIN_CLUSTER_SIZE)

    print(f"[make_synthetic] SYNTHETIC clustering scene (seed={args.seed}): "
          f"{n_ground} ground points (context only) + {n_nonground} non-ground points "
          f"(the clustering input), d={D_M} m, min_cluster_size={MIN_CLUSTER_SIZE}")
    print(f"[make_synthetic] truth: {num_true_components} raw single-linkage components -> "
          f"{num_reported} would be reported after min-size filtering")
    print(f"[make_synthetic] designed scenarios: separation pair @ idx {meta['sep_a_idx']}/{meta['sep_b_idx']}, "
          f"chaining pair @ idx {meta['chain_a_idx']}/{meta['chain_b_idx']}, "
          f"snake [{meta['snake_start_idx']}, +{meta['snake_count']}), "
          f"noise [{meta['noise_start_idx']}, +{meta['noise_count']})")
    total_bytes = 60 + n_ground * 3 * 4 + n_nonground * 3 * 4 + n_nonground * 4
    print(f"[make_synthetic] wrote {args.out} ({total_bytes} bytes, labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
