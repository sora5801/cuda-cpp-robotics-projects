#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.02
(ROI crop, passthrough, organized<->unorganized conversion kernels).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project needs THREE things, all synthesizable analytically with exact
ground truth: (1) an ORGANIZED LiDAR scan (a real 16-beam spinning sensor's
native ring x azimuth grid, some cells invalid), (2) a handful of points that
sit EXACTLY on the boundary of every predicate this project tests (the "edge
cohort" the worker brief calls for), and (3) a controlled set of "ghost"
second-echo returns that force deterministic collisions when the unorganized
cloud is re-binned into an organized grid. All three are cheap to generate
with a fixed seed, so nothing here is downloaded (CLAUDE.md paragraph 12:
reproducible, license-clean).

The scene: reused verbatim from project 02.01
-----------------------------------------------
The room geometry (16 m x 16 m, one sensor-height floor plane, four walls
with NO ceiling, three axis-aligned obstacle boxes) and the ray_box_intersect
/ ray_intersect_scene analytic raycaster below are REUSED, near-verbatim,
from ../../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/scripts/
make_synthetic.py (cited explicitly at each reused block) — the same virtual
warehouse room recurs across this domain's projects on purpose, so a learner
who has read 02.01 recognizes the scene immediately and can focus on what is
NEW here: the organized (ring x azimuth) grid structure and the ROI/
passthrough/frustum predicates, not a new scene to parse.

The 16-beam elevation table is reused from 01.18's THEORY.md ("Why LiDAR is
sparse", -15..+15 deg in 2 deg steps -> 16 beams), the same table 02.01 also
cites — see BEAM_ELEV_DEG below.

What is NEW in this project versus 02.01 (the organized grid)
-----------------------------------------------------------------
02.01 emits an UNORGANIZED point list (just xyz, no fixed shape). This
project's primary dataset is the OPPOSITE: a single revolution is rasterized
into a fixed-shape grid of NUM_BEAMS rings x AZIMUTH_BINS azimuth steps,
ring-major (flat index = ring*AZIMUTH_BINS + azimuth) — the sensor's NATIVE
geometry, exactly the shape a real spinning-LiDAR driver emits before any
"unorganize" step (THEORY.md "The problem" explains why: neighbor-by-index,
not neighbor-by-search, is the entire point of keeping this shape). A cell
that received no valid return (open sky over a wall top, OR the independent
5% ABSORPTION/GLARE dropout applied on top of every geometric hit — see
DROPOUT_PROB below) is written as three IEEE-754 NaN floats — NaN, not a
sentinel like -1, because NaN is the one float value guaranteed never to
collide with a legitimate coordinate and it self-documents "not a number" at
every print/debug session (documented again in kernels.cuh is_invalid_point).

Layout contract (single-sourced here AND in ../src/kernels.cuh — the two
MUST agree; main.cu asserts NUM_BEAMS/AZIMUTH_BINS from the file header
against the compiled-in constants at load time, the same discipline 02.01
uses for its leaf_m):
    NUM_BEAMS = 16, AZIMUTH_BINS = 1024 (0.3516 deg/step)
    ring elevation table: BEAM_ELEV_DEG (must match kernels.cuh's device
      table entry-for-entry — see kBeamElevDeg in kernels.cuh)
    predicate bounds (passthrough z-range, box AABB, frustum intrinsics +
      extrinsic + near plane) — must match kernels.cuh's kPassthroughZMin/
      Max, kBoxMin/Max, kFx/Fy/Cx/Cy/kFrustumNearM, kTCameraLidar EXACTLY,
      because this script places the "edge cohort" points analytically AT
      those exact boundaries (+-EDGE_EPS) to exercise the <=/>= comparisons
      at the float boundary main.cu's predicate_correctness gate checks.

Binary sample format (see ../data/README.md for the authoritative field
table + SHA-256):
    bytes  0.. 7  magic        b'RCPOU001' (8 bytes, no null terminator)
    bytes  8..31  int32 x6     NUM_BEAMS, AZIMUTH_BINS, N_EDGE, N_GHOST,
                                N_ORGANIZED_VALID, reserved(0)
    organized grid: NUM_BEAMS*AZIMUTH_BINS float32 x3 (x,y,z; NaN if
      invalid), ring-major (index = ring*AZIMUTH_BINS + azimuth), meters,
      "lidar" sensor frame (SYSTEM_DESIGN.md section 3.6 PointCloud/§3.2)
    edge cohort:    N_EDGE   float32 x3 (x,y,z), meters, "lidar" frame —
      NOT part of the organized grid (they do not correspond to any real
      beam direction; they exist purely to straddle predicate boundaries)
    ghost table:    N_GHOST records of (int32 cell_index, float32
      range_offset_m) — cell_index indexes the organized grid above (always
      a VALID cell); main.cu derives each ghost point's xyz at load time as
      (that cell's unit direction) * (that cell's range + range_offset_m) —
      i.e. the SAME ring/azimuth direction, a DIFFERENT range, modeling a
      second echo / multipath return / accumulated-frame duplicate. Deriving
      ghosts from the grid at load time (instead of baking xyz here) keeps
      the ray geometry single-sourced in this one raycaster.

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
# Deterministic RNG: xorshift32 (Marsaglia 2003), reused verbatim from
# 02.01's make_synthetic.py (cited) — the same three-shift/three-XOR core
# this repo's CUDA device code uses (project 11.01, 08.01), so a learner
# reading Python or CUDA C++ across projects sees one RNG algorithm.
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

    def shuffle(self, items: list) -> None:
        """In-place deterministic Fisher-Yates using THIS stream (never
        Python's random.shuffle — one RNG algorithm for the whole file)."""
        for k in range(len(items) - 1, 0, -1):
            j = int(self.uniform01() * (k + 1))
            if j > k:
                j = k  # guard the float-boundary uniform01()==1.0 case
            items[k], items[j] = items[j], items[k]


DEFAULT_SEED = 42  # repo convention (CLAUDE.md paragraph 12)

# ===========================================================================
# Organized-grid shape (MUST match kernels.cuh's kNumBeams / kAzimuthBins).
# ===========================================================================
NUM_BEAMS = 16
AZIMUTH_BINS = 1024  # 360 / 1024 = 0.3516 deg/step

# 16-beam elevation table, cited verbatim from 01.18's THEORY.md / reused
# from 02.01's make_synthetic.py: -15..+15 deg in 2 deg steps.
BEAM_ELEV_DEG = list(range(-15, 16, 2))
assert len(BEAM_ELEV_DEG) == NUM_BEAMS, "beam table must match the cited 16-beam model exactly"

# ===========================================================================
# Scene geometry — REUSED VERBATIM from 02.01's make_synthetic.py (cited in
# the module docstring). Sensor at the origin, +x forward, +z up.
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

RANGE_NOISE_M = 0.015     # +-15 mm per-return range noise (02.01 precedent)
DROPOUT_PROB = 0.05       # independent absorption/glare dropout on top of geometric hits (THEORY.md derives why)

# ===========================================================================
# Predicate bounds — MUST match kernels.cuh's constexpr constants exactly
# (see the module docstring's "layout contract"). Meters, "lidar" frame.
# ===========================================================================
PASSTHROUGH_Z_MIN = -1.0
PASSTHROUGH_Z_MAX = 0.5

BOX_MIN = (-4.0, -4.0, -1.5)
BOX_MAX = (4.0, 4.0, 1.0)

# Frustum camera: intrinsics identical to 01.18's teaching camera (cited),
# extrinsic identical to 01.18's kTCameraLidar (cited): a roof LiDAR looking
# down at a windshield-height camera, R a clean axis permutation (camera-z =
# lidar-x, camera-x = -lidar-y, camera-y = -lidar-z), t=(0,-0.30,-0.05) in
# the CAMERA frame. p_cam = R*p_lidar + t.
FX, FY, CX, CY, IMG_W, IMG_H = 154.0, 152.0, 80.0, 60.0, 160, 120
FRUSTUM_NEAR_M = 0.5
# T_CAMERA_LIDAR translation (camera frame) — see kernels.cuh kTCameraLidar.
T_CAM_T = (0.0, -0.30, -0.05)


def cam_to_lidar(pcx: float, pcy: float, pcz: float):
    """Inverse of kTCameraLidar: given a point in the CAMERA frame, return
    its coordinates in the LIDAR frame. Derived by hand from R^T (R is a
    pure permutation-with-sign, so R^T = R^-1 exactly) and documented in
    THEORY.md "The math" (frustum plane derivation) — reproduced here so
    this script can place points EXACTLY on the frustum's image-plane
    boundaries without needing a matrix library:
        lx = pcz + 0.05
        ly = -pcx
        lz = -pcy - 0.30
    """
    return (pcz + 0.05, -pcx, -pcy - 0.30)


# Adversarial ghost-duplicate count for the unorganized->organized collision
# test (main.cu derives each ghost's xyz from the organized grid + this
# offset at load time — see the file header).
N_GHOST = 200
GHOST_OFFSET_RANGE_M = 0.5   # range_offset_m drawn uniform(-this, +this)

EDGE_EPS = 1.0e-4  # how far off the exact boundary the "just inside/outside" edge points sit (meters/px)


# ---------------------------------------------------------------------------
# ray_box_intersect / ray_intersect_scene — REUSED VERBATIM from 02.01's
# make_synthetic.py (cited in the module docstring): the slab method for an
# AABB, and the "smallest positive in-bounds t across ground+walls+boxes"
# brute-force scene intersection. See that file for the full derivation
# comments; kept terse here to avoid duplicating the same essay twice.
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
        t = (-SENSOR_HEIGHT_M) / dz
        if t > 1e-6:
            x, y = dx * t, dy * t
            if abs(x) <= ROOM_HALF_M and abs(y) <= ROOM_HALF_M:
                best_t = t if best_t is None else min(best_t, t)

    R = ROOM_HALF_M
    wall_candidates = []
    if dx > 1e-9:
        wall_candidates.append(((R) / dx, 'x'))
    if dx < -1e-9:
        wall_candidates.append(((-R) / dx, 'x'))
    if dy > 1e-9:
        wall_candidates.append(((R) / dy, 'y'))
    if dy < -1e-9:
        wall_candidates.append(((-R) / dy, 'y'))
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


def build_organized_grid(rng: Xorshift32):
    """Cast ONE revolution of NUM_BEAMS x AZIMUTH_BINS rays against the
    analytic scene and fill the organized grid, ring-major.

    Returns (grid, n_hits, n_geom_miss, n_dropout) where grid is a flat
    list of length NUM_BEAMS*AZIMUTH_BINS of either (x,y,z) or None (NaN).
    n_geom_miss counts rays that hit nothing in range (open sky); n_dropout
    counts rays that DID hit geometry but were independently discarded by
    the DROPOUT_PROB absorption/glare model — the two invalid REASONS this
    project's THEORY.md teaches separately even though both end up as the
    same NaN sentinel in the grid (a real driver cannot tell them apart
    either — see THEORY.md "The problem").
    """
    grid = [None] * (NUM_BEAMS * AZIMUTH_BINS)
    n_hits = n_geom_miss = n_dropout = 0
    for ring, el_deg in enumerate(BEAM_ELEV_DEG):
        el = math.radians(el_deg)
        cel, sel = math.cos(el), math.sin(el)
        for az_step in range(AZIMUTH_BINS):
            # BIN CENTER, not bin edge: azimuth_bin_of() in kernels.cuh
            # reconstructs a bin index via floor(atan2(y,x)/bin_width), and
            # atan2(sin(t),cos(t)) == t only in EXACT arithmetic — a ray cast
            # at the bin's exact LOWER EDGE sits precisely on the floor()
            # decision boundary, so a sub-ULP round-trip rounding error (the
            # cos/sin/atan2 chain does not invert perfectly in float32/
            # float64) floors into az_step-1 roughly HALF the time. This was
            # caught by this project's own GATE roundtrip during development
            # (~49% azimuth-bin mismatch on reconstruction — an honest record
            # of the bug, not smoothed over: see THEORY.md "Numerical
            # considerations"). Casting at the bin CENTER instead gives a
            # full half-bin-width (~0.003 rad) of margin against sub-ULP
            # noise, and is also more physically honest: a real spinning
            # LiDAR's firing angle for azimuth bin k IS its bin's angular
            # center, not its leading edge.
            az = 2.0 * math.pi * (az_step + 0.5) / AZIMUTH_BINS
            dx = cel * math.cos(az)
            dy = cel * math.sin(az)
            dz = sel
            idx = ring * AZIMUTH_BINS + az_step
            t = ray_intersect_scene(dx, dy, dz)
            if t is None:
                n_geom_miss += 1
                continue  # grid[idx] stays None -> NaN
            if rng.uniform01() < DROPOUT_PROB:
                n_dropout += 1
                continue  # geometry says "hit", sensor model says "lost" -> NaN
            r = t + rng.uniform(-RANGE_NOISE_M, RANGE_NOISE_M)
            if r <= 1e-6:
                r = t
            grid[idx] = (dx * r, dy * r, dz * r)
            n_hits += 1
    return grid, n_hits, n_geom_miss, n_dropout


def build_edge_cohort():
    """Handcrafted points straddling every predicate boundary this project
    tests, at +-EDGE_EPS around the exact threshold — the deliberately
    adversarial "does <= really mean <=" cohort the worker brief calls for.
    Returns a list of (x,y,z) tuples. None of these correspond to a real
    beam direction; they are NOT part of the organized grid.
    """
    pts = []

    # --- passthrough (z) boundary: fixed x,y=2,0 (well inside box+frustum
    # in x/y so this isolates the z-passthrough effect specifically). -----
    x0, y0 = 2.0, 0.0
    for z in (PASSTHROUGH_Z_MIN - EDGE_EPS, PASSTHROUGH_Z_MIN, PASSTHROUGH_Z_MIN + EDGE_EPS,
              PASSTHROUGH_Z_MAX - EDGE_EPS, PASSTHROUGH_Z_MAX, PASSTHROUGH_Z_MAX + EDGE_EPS):
        pts.append((x0, y0, z))

    # --- box AABB boundary: each of the 6 faces, +-EDGE_EPS and exact,
    # holding the other two coordinates at the box CENTER so only the one
    # face under test can flip the predicate. --------------------------------
    box_center = tuple((BOX_MIN[i] + BOX_MAX[i]) * 0.5 for i in range(3))
    for axis in range(3):
        for bound in (BOX_MIN[axis], BOX_MAX[axis]):
            for delta in (-EDGE_EPS, 0.0, EDGE_EPS):
                p = list(box_center)
                p[axis] = bound + delta
                pts.append(tuple(p))

    # --- frustum boundary: near plane + the 4 image-edge planes, expressed
    # in the CAMERA frame then mapped back to the LIDAR frame via
    # cam_to_lidar() (derived above, cited from THEORY.md). ------------------
    z_probe_m = 3.0  # comfortably inside the room, in front of the camera
    # Near plane: straight ahead (pcx=pcy=0), z_cam = near +- EDGE_EPS.
    for pcz in (FRUSTUM_NEAR_M - EDGE_EPS, FRUSTUM_NEAR_M, FRUSTUM_NEAR_M + EDGE_EPS):
        pts.append(cam_to_lidar(0.0, 0.0, pcz))
    # Left edge (u=0) / right edge (u=W-1) / top edge (v=0) / bottom edge
    # (v=H-1): perturb the pixel coordinate by +-EDGE_EPS "pixels" and
    # convert back to a camera-frame (x,y) at z_probe_m via the pinhole
    # inverse x = (u-cx)/fx * z, y = (v-cy)/fy * z.
    for u_edge in (0.0, IMG_W - 1.0):
        for du in (-EDGE_EPS, 0.0, EDGE_EPS):
            u = u_edge + du
            pcx = (u - CX) / FX * z_probe_m
            pcy = 0.0
            pts.append(cam_to_lidar(pcx, pcy, z_probe_m))
    for v_edge in (0.0, IMG_H - 1.0):
        for dv in (-EDGE_EPS, 0.0, EDGE_EPS):
            v = v_edge + dv
            pcy = (v - CY) / FY * z_probe_m
            pcx = 0.0
            pts.append(cam_to_lidar(pcx, pcy, z_probe_m))

    return pts


def build_ghost_table(rng: Xorshift32, grid):
    """Pick N_GHOST distinct VALID cells (deterministic Fisher-Yates over
    the valid-cell index list) and draw each a range_offset_m in
    +-GHOST_OFFSET_RANGE_M, guarded so the resulting range stays positive.
    Returns a list of (cell_index, range_offset_m).
    """
    valid_cells = [i for i, p in enumerate(grid) if p is not None]
    rng.shuffle(valid_cells)
    chosen = valid_cells[:N_GHOST]

    table = []
    for cell in chosen:
        x, y, z = grid[cell]
        rng_m = math.sqrt(x * x + y * y + z * z)
        offset = rng.uniform(-GHOST_OFFSET_RANGE_M, GHOST_OFFSET_RANGE_M)
        if rng_m + offset <= 0.05:
            offset = 0.1  # guard: never produce a non-positive or near-zero ghost range
        table.append((cell, offset))
    return table


def write_binary_sample(out_path: Path, grid, edge_pts, ghost_table):
    n_organized = NUM_BEAMS * AZIMUTH_BINS
    n_valid = sum(1 for p in grid if p is not None)
    n_edge = len(edge_pts)
    n_ghost = len(ghost_table)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nan = float('nan')
    with out_path.open('wb') as f:
        f.write(b'RCPOU001')
        f.write(struct.pack('<iiiiii', NUM_BEAMS, AZIMUTH_BINS, n_edge, n_ghost, n_valid, 0))

        flat = []
        for p in grid:
            if p is None:
                flat.extend((nan, nan, nan))
            else:
                flat.extend(p)
        f.write(struct.pack(f'<{len(flat)}f', *flat))

        flat_edge = []
        for (x, y, z) in edge_pts:
            flat_edge.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat_edge)}f', *flat_edge))

        for (cell, offset) in ghost_table:
            f.write(struct.pack('<if', cell, offset))

    total_bytes = 8 + 24 + n_organized * 3 * 4 + n_edge * 3 * 4 + n_ghost * 8
    return n_organized, n_valid, n_edge, n_ghost, total_bytes


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / 'data' / 'sample' / 'roi_scan.bin'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out,
                        help='output binary path (default: ../data/sample/roi_scan.bin)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    grid, n_hits, n_geom_miss, n_dropout = build_organized_grid(rng)
    edge_pts = build_edge_cohort()
    ghost_table = build_ghost_table(rng, grid)

    n_organized, n_valid, n_edge, n_ghost, total_bytes = write_binary_sample(
        args.out, grid, edge_pts, ghost_table)

    n_rays = NUM_BEAMS * AZIMUTH_BINS
    print(f"[make_synthetic] SYNTHETIC single-revolution 16-beam organized scan (seed={args.seed}, "
          f"{NUM_BEAMS} rings x {AZIMUTH_BINS} azimuth = {n_rays} cells)")
    print(f"[make_synthetic] valid={n_valid} ({100.0*n_valid/n_rays:.1f}%), "
          f"geometric-miss={n_geom_miss} ({100.0*n_geom_miss/n_rays:.1f}%), "
          f"dropout={n_dropout} ({100.0*n_dropout/n_rays:.1f}%, target {100*DROPOUT_PROB:.0f}%)")
    print(f"[make_synthetic] + {n_edge} edge-cohort points straddling passthrough/box/frustum boundaries")
    print(f"[make_synthetic] + {n_ghost} ghost second-echo duplicates (deterministic collision test)")
    print(f"[make_synthetic] wrote {args.out} ({total_bytes} bytes, labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
