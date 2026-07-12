#!/usr/bin/env python3
"""make_synthetic.py -- synthetic trajectory + LiDAR scans for project 02.11
                        (Scan Context / ring-descriptor loop-closure search).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Scan Context is a PLACE-RECOGNITION algorithm: it needs a world with several
visually/geometrically DISTINCT places, a robot trajectory that revisits some
of them (from several relative headings and a couple of lateral offsets) and
drives past several new ones exactly once, and a real 3-D point cloud per
keyframe so the descriptor has actual vertical structure to describe. None of
that exists in any tiny public dataset we could license cleanly and keep
"tiny" (CLAUDE.md paragraph 8), so -- as is typical for robotics -- we
synthesize it, with FULL ground truth (every keyframe's true pose, and a
curated list of true revisit pairs with their cohort label and true relative
yaw). Determinism: every random choice in this file is driven by ONE
hand-rolled xorshift32 generator seeded 42 (CLAUDE.md paragraph 12 -- no
std::uniform_real_distribution / no Python `random` module; the repo's C++
projects use xorshift32 for the same determinism reason and this script
mirrors it so a reader who has seen one has seen both).

The three-stage pipeline this file implements
-----------------------------------------------
  1. WORLD: a small grid of "city blocks" -- 4x3 street intersections
     ("stations") 25 m apart, with a building (or two) filling each of the
     6 interior cells. Building footprint/height are derived from the cell
     index through xorshift32, so neighboring blocks look genuinely
     different -- exactly what Scan Context needs to tell places apart (see
     THEORY.md "the aliasing problem" for why visually-similar places are
     the hard case this world is deliberately built to mostly avoid, with
     one exception discussed there).
  2. ROUTE: a hand-authored walk over the station graph (ROUTE below), built
     from labelled PHASES so the resulting trajectory contains, by
     construction: several places revisited with the SAME heading, several
     revisited HEADING-REVERSED (~180 deg -- the rotation-invariance
     showcase), a few revisited with a small LATERAL OFFSET (a "shifted
     lane" -- the honesty cohort for translation sensitivity), and several
     genuinely NEW places visited exactly once (the negative cohort). Each
     traversal of a station-to-station edge becomes one "segment"; segments
     are sampled into KEYFRAMES at a spacing chosen so the whole route
     yields close to TARGET_KEYFRAMES total (~120, per the project brief).
  3. SCANS: for every keyframe pose, a simple multi-channel LiDAR is
     simulated by ray-casting (ground plane + axis-aligned building walls)
     from the sensor (mounted SENSOR_HEIGHT_M above the ground on the robot)
     across a grid of (azimuth, elevation) directions. Each ray's first hit
     becomes one 3-D point, stored in the SENSOR FRAME (not world frame) --
     Scan Context is an EGOCENTRIC descriptor (THEORY.md derives why), so
     sensor-frame points are exactly what the C++ pipeline expects to bin
     directly into (ring, sector) cells.

Outputs (all under ../data/sample/, all labeled SYNTHETIC):
  world.csv        -- building footprints, for the trajectory-view artifact.
  trajectory.csv   -- one row per keyframe: pose + which route segment it
                       belongs to (so the C++ side can also reconstruct
                       cohorts if it ever wants to; the curated pairs below
                       are the primary ground truth main.cu reads).
  loop_pairs.csv   -- the curated ground-truth revisit pairs: (query
                       keyframe, match keyframe, cohort label, true relative
                       yaw, lateral offset). This is what main.cu's
                       loop_detection / rotation_invariance / lateral_
                       sensitivity / negative_cohort gates score against.
  scans.bin        -- every keyframe's point cloud, SENSOR FRAME, meters,
                       packed as: magic "SCANCTX1" (8 bytes), int32
                       num_scans, then per scan: int32 n_points followed by
                       n_points*3 float32 (x,y,z). data/README.md documents
                       this byte-for-byte.

Usage
-----
    python make_synthetic.py                  # defaults: seed=42, ~120 keyframes
    python make_synthetic.py --seed 42 --target-keyframes 120 --out-dir ../data/sample
"""

import argparse
import csv
import math
import struct
from pathlib import Path

# ===========================================================================
# xorshift32 -- the repo's portable deterministic RNG (Marsaglia 2003). Used
# here for EVERY random choice (building footprints) so the whole world is
# reproducible bit-for-bit from one seed, matching the C++ side's generator
# (main.cu/kernels.cu use none -- this project's C++ has no RNG at all; only
# the DATA is randomized, at generation time, here).
# ===========================================================================
class Xorshift32:
    """A tiny, portable, seedable PRNG -- the exact three-shift/three-XOR
    core used throughout this repo's C++ projects (08.01, 02.03, 02.10, ...),
    reimplemented in Python so this script needs no third-party dependency
    and produces choices that are reproducible independent of Python's own
    `random` module (whose algorithm is not part of this repo's contract)."""

    def __init__(self, seed: int):
        # xorshift32 is undefined at state 0 (it would stay 0 forever) --
        # guard exactly like the C++ call sites do.
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def next_u32(self) -> int:
        s = self.state
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17)
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        self.state = s
        return s

    def uniform(self, lo: float, hi: float) -> float:
        """Deterministic float in [lo, hi)."""
        u = self.next_u32() / 4294967296.0  # 2^32; result in [0, 1)
        return lo + u * (hi - lo)


DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# World geometry constants. STATION_SPACING_M / MAX_RANGE_M are part of the
# data<->pipeline CONTRACT: kernels.cuh's kSensorMaxRangeM (C++) must equal
# MAX_RANGE_M here, or ring binning silently clips real returns.  Both sides
# document the shared value; changing one means changing both (CLAUDE.md
# paragraph 12: one contract, stated in more than one file, kept in sync by
# hand with a cross-reference comment -- there is no build-time check that
# spans Python and CUDA).
# ---------------------------------------------------------------------------
STATION_SPACING_M = 25.0
N_COLS = 4                      # stations per row
N_ROWS = 3                      # rows of stations -> N_COLS*N_ROWS = 12 stations, 17 edges
SENSOR_HEIGHT_M = 1.6           # sensor mount height above ground (a typical AMR/AV roof mount)
MAX_RANGE_M = 40.0              # MUST match kernels.cuh kSensorMaxRangeM
AZIMUTH_STEPS = 120             # 3 deg azimuth resolution for the ray fan
ELEV_MIN_DEG = -18.0            # mostly downward -- a roof-mounted LiDAR looking at a robot-scale world
ELEV_MAX_DEG = 12.0
ELEV_CHANNELS = 16              # a Velodyne-Puck-like channel count (teaching scale, not a product claim)
TARGET_KEYFRAMES = 120          # the brief's "T~120 keyframe scans"
MIN_LOOP_GAP_KF = 15            # MUST match kernels.cuh kMinLoopGapKeyframes (temporal exclusion window)
SAME_OFFSET_TOL_M = 0.3         # |offset_a - offset_b| below this -> "no intentional lateral shift" for cohort tagging


def station_pos(station_id: int):
    """Station id -> (x, y) world position. id = row*N_COLS + col."""
    col = station_id % N_COLS
    row = station_id // N_COLS
    return (col * STATION_SPACING_M, row * STATION_SPACING_M)


# ===========================================================================
# STAGE 1 -- WORLD: buildings filling the 6 interior cells of the station
# grid. Cell (i, j) for i in [0, N_COLS-2], j in [0, N_ROWS-2] sits between
# stations (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1). Even cells get ONE large building
# roughly centered in the cell; odd cells get TWO smaller buildings side by
# side -- deliberately different silhouettes so a Scan Context descriptor
# computed near one cell looks different from one computed near another
# (the whole reason a "town", not a featureless plane, is the test world).
# ===========================================================================
def build_world(rng: Xorshift32):
    buildings = []   # each: dict(x0,y0,x1,y1,h)
    n_cell_cols = N_COLS - 1
    n_cell_rows = N_ROWS - 1
    for j in range(n_cell_rows):
        for i in range(n_cell_cols):
            cell_idx = j * n_cell_cols + i
            cx = (i + 0.5) * STATION_SPACING_M
            cy = (j + 0.5) * STATION_SPACING_M
            if cell_idx % 2 == 0:
                # One large building, jittered off-center so it is not a
                # perfectly symmetric (and therefore rotation-ambiguous) box.
                w = rng.uniform(10.0, 16.0)
                d = rng.uniform(10.0, 16.0)
                h = rng.uniform(5.0, 14.0)
                jx = rng.uniform(-2.0, 2.0)
                jy = rng.uniform(-2.0, 2.0)
                buildings.append({
                    "x0": cx + jx - w / 2.0, "y0": cy + jy - d / 2.0,
                    "x1": cx + jx + w / 2.0, "y1": cy + jy + d / 2.0, "h": h,
                })
            else:
                # Two smaller buildings side by side -- a "warehouse row"
                # silhouette, structurally distinct from the single-block
                # cells so neighboring places do not alias (THEORY.md).
                for k in range(2):
                    w = rng.uniform(5.0, 8.0)
                    d = rng.uniform(8.0, 14.0)
                    h = rng.uniform(3.0, 10.0)
                    ox = (k - 0.5) * (w + 3.0)  # 3 m gap between the pair
                    jy = rng.uniform(-1.5, 1.5)
                    buildings.append({
                        "x0": cx + ox - w / 2.0, "y0": cy + jy - d / 2.0,
                        "x1": cx + ox + w / 2.0, "y1": cy + jy + d / 2.0, "h": h,
                    })
    return buildings


# ===========================================================================
# STAGE 2 -- ROUTE: the hand-authored walk. See the file header for the
# per-phase intent. Each entry is (from_station_id, to_station_id,
# lateral_offset_m). A nonzero offset shifts BOTH endpoints of that segment
# by offset_m perpendicular to the direction of travel ("left of travel"),
# i.e. it is a parallel-shifted straight line -- a different lane through
# the same place, not a different place.
# ===========================================================================
ROUTE = [
    # ---- phase 1: outer perimeter loop, first visit, zero offset --------
    (0, 1, 0.0), (1, 2, 0.0), (2, 3, 0.0), (3, 7, 0.0), (7, 11, 0.0),
    (11, 10, 0.0), (10, 9, 0.0), (9, 8, 0.0), (8, 4, 0.0), (4, 0, 0.0),
    # ---- phase 2: interior cross, first visit, zero offset --------------
    (0, 4, 0.0), (4, 5, 0.0), (5, 6, 0.0), (6, 7, 0.0),
    # ---- phase 3: more interior + reversals of phase-1 edges ------------
    (7, 6, 0.0), (6, 10, 0.0), (10, 11, 0.0), (11, 7, 0.0),
    # ---- phase 4: SAME-HEADING revisits of the phase-1 outer loop -------
    (7, 11, 0.0), (11, 10, 0.0), (10, 9, 0.0), (9, 8, 0.0),
    (8, 4, 0.0), (4, 0, 0.0), (0, 1, 0.0), (1, 2, 0.0),
    # ---- phase 5: LATERAL-OFFSET revisits (three distinct magnitudes) ---
    (1, 2, 1.3), (0, 1, 2.6), (4, 5, 0.6),
    # ---- phase 6: genuinely NEW places, visited exactly once ------------
    (1, 5, 0.0), (5, 9, 0.0), (2, 6, 0.0),
]


def build_segments():
    """ROUTE -> list of segment dicts with world-space endpoints (offset
    already applied) and a direction unit vector, ready for keyframe
    sampling. The perpendicular ("left of travel") used for the offset is
    rotate(direction, +90 deg) -- consistent for any two traversals sharing
    a direction, which is all the offset cohort ever compares (file header)."""
    segments = []
    for (a, b, offset_m) in ROUTE:
        ax, ay = station_pos(a)
        bx, by = station_pos(b)
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length          # unit direction of travel
        px, py = -uy, ux                            # left-of-travel unit perpendicular
        sx0, sy0 = ax + px * offset_m, ay + py * offset_m
        sx1, sy1 = bx + px * offset_m, by + py * offset_m
        segments.append({
            "from_id": a, "to_id": b, "offset_m": offset_m,
            "x0": sx0, "y0": sy0, "x1": sx1, "y1": sy1,
            "length_m": length, "heading_rad": math.atan2(uy, ux),
            "edge_id": frozenset((a, b)),
        })
    return segments


def sample_keyframes(segments, target_keyframes: int):
    """Sample each segment into keyframes at a spacing chosen so the WHOLE
    route yields close to target_keyframes total. Samples sit at the
    midpoints of n equal sub-intervals of each segment (t = (k+0.5)/n,
    k=0..n-1) -- deliberately never exactly at a station, so two segments
    that share a station endpoint never place two keyframes on top of each
    other (see file header: this is what keeps the temporal-gap exclusion
    and the negative cohort clean at intersections).

    n is forced ODD (this is load-bearing, not cosmetic -- an earlier
    version of this function used round()'s raw result and shipped a real
    bug this project's own diagnostic sweep caught): the ANCHOR is sample
    k=n//2, at t=(n//2+0.5)/n. For EVEN n that fraction is NOT 0.5 (e.g.
    n=4 gives t=0.625) -- fine on its own, but it breaks the one property
    two OPPOSITE-direction traversals of the same edge need to agree on:
    position_forward(t) from station A must equal position_reverse(t) from
    station B when both anchors are meant to be "the same physical spot,
    approached from either direction". That equality holds ONLY at t=0.5
    (the true geometric midpoint) -- at t=0.625 the forward anchor sits
    62.5% of the way from A to B while the reverse anchor sits only 37.5%
    of the way, a real ~6 m position mismatch on a 25 m block that showed
    up as an unexpectedly large Scan Context distance for the ROTATED
    cohort specifically. For ODD n, k=n//2=(n-1)/2 exactly, and
    t=((n-1)/2+0.5)/n simplifies to exactly 0.5 -- the fix."""
    total_length = sum(s["length_m"] for s in segments)
    spacing = total_length / float(target_keyframes)

    keyframes = []   # each: dict(idx, x, y, heading_rad, seg_index)
    for seg_idx, seg in enumerate(segments):
        n = max(3, round(seg["length_m"] / spacing))
        if n % 2 == 0:
            n += 1   # force odd -- see docstring: this is what makes the anchor land exactly at t=0.5
        seg["kf_start"] = len(keyframes)
        for k in range(n):
            t = (k + 0.5) / n
            x = seg["x0"] + (seg["x1"] - seg["x0"]) * t
            y = seg["y0"] + (seg["y1"] - seg["y0"]) * t
            keyframes.append({
                "idx": len(keyframes), "x": x, "y": y,
                "heading_rad": seg["heading_rad"], "seg_index": seg_idx,
            })
        seg["kf_end"] = len(keyframes) - 1
        seg["kf_anchor"] = seg["kf_start"] + (n // 2)   # exactly the segment's t=0.5 midpoint sample (see docstring)
    return keyframes


def wrap_deg(d: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d


def build_loop_pairs(segments, keyframes):
    """Group segment traversals by physical edge (edge_id), and for every
    pair of traversals of the SAME edge that are far enough apart in
    keyframe-index (>= MIN_LOOP_GAP_KF -- a query has no business "loop
    closing" against a keyframe from a few seconds ago; that is what
    odometry already knows, not what place recognition is for), emit one
    curated ground-truth row: (query=later anchor, match=earlier anchor,
    cohort, true relative yaw, lateral offset). Cohort tagging compares each
    pair's DIRECTED (from_id,to_id): identical -> same_heading (or
    lateral_offset if the two offsets differ by more than the tolerance);
    reversed -> rotated. Any edge whose every traversal-pair fails the gap
    test (or that was traversed only once) contributes NO pair at all --
    its anchor keyframe(s) become negative-cohort examples by construction
    (built in main.cu from "no curated pair names me")."""
    by_edge = {}
    for i, seg in enumerate(segments):
        by_edge.setdefault(seg["edge_id"], []).append(i)

    pairs = []
    for edge_id, seg_indices in by_edge.items():
        for a in range(len(seg_indices)):
            for b in range(a + 1, len(seg_indices)):
                sa, sb = segments[seg_indices[a]], segments[seg_indices[b]]
                # Order by anchor keyframe index: query = later, match = earlier
                # (loop closure only ever looks into the PAST).
                if sa["kf_anchor"] <= sb["kf_anchor"]:
                    earlier, later = sa, sb
                else:
                    earlier, later = sb, sa
                gap = later["kf_anchor"] - earlier["kf_anchor"]
                if gap < MIN_LOOP_GAP_KF:
                    continue   # too close in time to be a meaningful loop closure (see docstring)

                same_dir = (earlier["from_id"], earlier["to_id"]) == (later["from_id"], later["to_id"])
                reversed_dir = (earlier["from_id"], earlier["to_id"]) == (later["to_id"], later["from_id"])
                assert same_dir or reversed_dir, "edge_id matched but neither same nor reversed direction"

                true_yaw_deg = wrap_deg(math.degrees(later["heading_rad"] - earlier["heading_rad"]))
                offset_delta = abs(later["offset_m"] - earlier["offset_m"])

                if reversed_dir:
                    cohort = "rotated"
                elif offset_delta > SAME_OFFSET_TOL_M:
                    cohort = "lateral_offset"
                else:
                    cohort = "same_heading"

                pairs.append({
                    "query_idx": later["kf_anchor"], "match_idx": earlier["kf_anchor"],
                    "cohort": cohort, "relative_yaw_true_deg": true_yaw_deg,
                    "lateral_offset_m": offset_delta,
                })
    return pairs


# ===========================================================================
# STAGE 3 -- SCANS: ray-cast every keyframe's point cloud.
# ===========================================================================
def ray_aabb_entry_t(ox, oy, dx, dy, x0, y0, x1, y1):
    """Standard slab-method ray/AABB intersection in the XY plane. Returns
    the ray parameter t at which the ray ENTERS the box (or None if it never
    does, or the box is entirely behind the ray). The sensor is guaranteed
    to be outside every building (stations sit on streets), so "entry" is
    always the near intersection tmin, never the far one."""
    if abs(dx) < 1e-9:
        if ox < x0 or ox > x1:
            return None
        tx0, tx1 = -math.inf, math.inf
    else:
        tx0, tx1 = (x0 - ox) / dx, (x1 - ox) / dx
        if tx0 > tx1:
            tx0, tx1 = tx1, tx0
    if abs(dy) < 1e-9:
        if oy < y0 or oy > y1:
            return None
        ty0, ty1 = -math.inf, math.inf
    else:
        ty0, ty1 = (y0 - oy) / dy, (y1 - oy) / dy
        if ty0 > ty1:
            ty0, ty1 = ty1, ty0
    tmin = max(tx0, ty0)
    tmax = min(tx1, ty1)
    if tmin > tmax or tmax < 0.0 or tmin <= 0.0:
        return None
    return tmin


def simulate_scan(x, y, heading_rad, buildings):
    """Cast the (AZIMUTH_STEPS x ELEV_CHANNELS) ray fan from a sensor at
    (x, y, SENSOR_HEIGHT_M) with forward direction heading_rad, against the
    ground plane (z=0) and every building's vertical walls (a wall hit is a
    ray/AABB entry in XY whose z at that t falls within [0, building height]
    -- rooftops are not modeled; a documented simplification, README
    Limitations). Returns a list of (x, y, z) points in the SENSOR FRAME
    (rotated by -heading_rad, translated by -(x,y,SENSOR_HEIGHT_M)) --
    exactly the egocentric frame Scan Context bins directly."""
    points = []
    elevations = [
        math.radians(ELEV_MIN_DEG + (ELEV_MAX_DEG - ELEV_MIN_DEG) * ch / (ELEV_CHANNELS - 1))
        for ch in range(ELEV_CHANNELS)
    ]
    azimuths = [2.0 * math.pi * a / AZIMUTH_STEPS for a in range(AZIMUTH_STEPS)]

    for elev in elevations:
        dz = math.sin(elev)
        horiz = math.cos(elev)
        for az in azimuths:
            world_az = heading_rad + az
            dx = horiz * math.cos(world_az)
            dy = horiz * math.sin(world_az)

            best_t = MAX_RANGE_M
            hit = False

            # Ground plane z=0: sensor sits at SENSOR_HEIGHT_M, so a downward
            # ray (dz<0) reaches it at t = SENSOR_HEIGHT_M / (-dz).
            if dz < -1e-6:
                t_ground = SENSOR_HEIGHT_M / (-dz)
                if 0.0 < t_ground <= best_t:
                    best_t = t_ground
                    hit = True

            for b in buildings:
                t_enter = ray_aabb_entry_t(x, y, dx, dy, b["x0"], b["y0"], b["x1"], b["y1"])
                if t_enter is not None and t_enter < best_t:
                    z_hit = SENSOR_HEIGHT_M + t_enter * dz
                    if 0.0 <= z_hit <= b["h"]:
                        best_t = t_enter
                        hit = True

            if hit and best_t < MAX_RANGE_M:
                wx = x + dx * best_t
                wy = y + dy * best_t
                wz = SENSOR_HEIGHT_M + dz * best_t
                # World -> sensor frame: translate then rotate by -heading.
                rx, ry, rz = wx - x, wy - y, wz - SENSOR_HEIGHT_M
                c, s = math.cos(-heading_rad), math.sin(-heading_rad)
                sx = c * rx - s * ry
                sy = s * rx + c * ry
                points.append((sx, sy, rz))
    return points


# ===========================================================================
# Output writers.
# ===========================================================================
def write_world_csv(path: Path, buildings, seed, n_stations):
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data -- generated by scripts/make_synthetic.py for project 02.11\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write(f"# {n_stations} stations, {STATION_SPACING_M} m grid spacing, {len(buildings)} buildings\n")
        f.write("# columns: x0_m,y0_m,x1_m,y1_m,height_m (axis-aligned footprint, world frame)\n")
        w = csv.writer(f)
        w.writerow(["x0_m", "y0_m", "x1_m", "y1_m", "height_m"])
        for b in buildings:
            w.writerow([f"{b['x0']:.3f}", f"{b['y0']:.3f}", f"{b['x1']:.3f}", f"{b['y1']:.3f}", f"{b['h']:.3f}"])


def write_trajectory_csv(path: Path, keyframes, segments, seed):
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data -- generated by scripts/make_synthetic.py for project 02.11\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write("# columns: idx,x_m,y_m,heading_rad,seg_index,from_id,to_id,offset_m,is_anchor(0/1)\n")
        w = csv.writer(f)
        w.writerow(["idx", "x_m", "y_m", "heading_rad", "seg_index", "from_id", "to_id", "offset_m", "is_anchor"])
        for kf in keyframes:
            seg = segments[kf["seg_index"]]
            is_anchor = 1 if seg["kf_anchor"] == kf["idx"] else 0
            w.writerow([kf["idx"], f"{kf['x']:.4f}", f"{kf['y']:.4f}", f"{kf['heading_rad']:.6f}",
                       kf["seg_index"], seg["from_id"], seg["to_id"], f"{seg['offset_m']:.3f}", is_anchor])


def write_loop_pairs_csv(path: Path, pairs, seed):
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground truth -- generated by scripts/make_synthetic.py for project 02.11\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write("# columns: query_idx,match_idx,cohort,relative_yaw_true_deg,lateral_offset_m\n")
        f.write("# cohort in {same_heading, rotated, lateral_offset}; a query anchor absent from every\n")
        f.write("# row of this file (as query_idx AND match_idx) is a NEGATIVE example (see main.cu).\n")
        w = csv.writer(f)
        w.writerow(["query_idx", "match_idx", "cohort", "relative_yaw_true_deg", "lateral_offset_m"])
        for p in sorted(pairs, key=lambda r: (r["query_idx"], r["match_idx"])):
            w.writerow([p["query_idx"], p["match_idx"], p["cohort"],
                       f"{p['relative_yaw_true_deg']:.3f}", f"{p['lateral_offset_m']:.3f}"])


def write_scans_bin(path: Path, all_points):
    with path.open("wb") as f:
        f.write(b"SCANCTX1")
        f.write(struct.pack("<i", len(all_points)))
        for pts in all_points:
            f.write(struct.pack("<i", len(pts)))
            for (px, py, pz) in pts:
                f.write(struct.pack("<fff", px, py, pz))


def main():
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic trajectory + LiDAR scans for project 02.11.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"xorshift32 seed for the world (default {DEFAULT_SEED})")
    parser.add_argument("--target-keyframes", type=int, default=TARGET_KEYFRAMES,
                        help=f"approximate total keyframe count (default {TARGET_KEYFRAMES})")
    parser.add_argument("--out-dir", type=Path, default=default_out,
                        help="output directory (default: ../data/sample)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = Xorshift32(args.seed)
    buildings = build_world(rng)
    segments = build_segments()
    keyframes = sample_keyframes(segments, args.target_keyframes)
    pairs = build_loop_pairs(segments, keyframes)

    print(f"[make_synthetic] world: {N_COLS}x{N_ROWS} stations, {len(buildings)} buildings (seed={args.seed})")
    print(f"[make_synthetic] route: {len(segments)} segments, {len(keyframes)} keyframes")
    cohort_counts = {}
    for p in pairs:
        cohort_counts[p["cohort"]] = cohort_counts.get(p["cohort"], 0) + 1
    print(f"[make_synthetic] curated revisit pairs: {len(pairs)} total, by cohort: {cohort_counts}")

    print("[make_synthetic] ray-casting scans (this is the slow step; pure-Python, one-time)...")
    all_points = []
    total_pts = 0
    for i, kf in enumerate(keyframes):
        pts = simulate_scan(kf["x"], kf["y"], kf["heading_rad"], buildings)
        all_points.append(pts)
        total_pts += len(pts)
        if (i + 1) % 20 == 0 or i + 1 == len(keyframes):
            print(f"[make_synthetic]   scan {i + 1}/{len(keyframes)} ({len(pts)} points)")
    print(f"[make_synthetic] average points/scan: {total_pts / len(keyframes):.0f}")

    write_world_csv(args.out_dir / "world.csv", buildings, args.seed, N_COLS * N_ROWS)
    write_trajectory_csv(args.out_dir / "trajectory.csv", keyframes, segments, args.seed)
    write_loop_pairs_csv(args.out_dir / "loop_pairs.csv", pairs, args.seed)
    write_scans_bin(args.out_dir / "scans.bin", all_points)

    print(f"[make_synthetic] wrote world.csv, trajectory.csv, loop_pairs.csv, scans.bin to {args.out_dir}")
    print("[make_synthetic] all data labeled SYNTHETIC; see ../data/README.md for the full format + checksums")


if __name__ == "__main__":
    main()
