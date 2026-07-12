#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.13
(Dynamic point removal (raycast free-space carving)).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
Free-space carving needs something no public LiDAR dataset hands you cleanly:
an OBJECT-LEVEL ground truth for which points belong to permanent structure
and which belong to something that moved — plus a designed sequence of scans
where "moved" comes in every interesting flavor (crossed the scene, stood
still and then left, appeared once in open space). This script builds that
scene, ray-casts a real 16-beam spinning LiDAR through it from a moving
platform, and writes the resulting beams (hit or miss) plus per-hit ground
truth to ../data/sample/. No download, no license question, bit-for-bit
reproducible from a fixed seed (42, xorshift32 — CLAUDE.md paragraph 12: no
std::uniform_real_distribution / no Python random module; this project's C++
side does not even need noise at runtime, since the noise is baked into the
committed file once, here).

THE SCENE (world frame: x-forward, y-left, z-up, right-handed; meters).
------------------------------------------------------------------------
A sensor rides a straight track (y=0, z=SENSOR_HEIGHT_M) through a small
"parking lot" of static structure while a car drives across its path and a
pedestrian waits, then leaves. Every object below is an axis-aligned box
(AABB) except the pole, a vertical cylinder — both have closed-form ray
intersections, so ground truth (which object, if any, a beam hits, and at
what exact range) is exact, no approximation, no downloaded mesh:

  STATIC (never moves, present every scan):
    WALL       — a long straight wall at y ~= 8 m, x in [-15, 15] m. The last
                 2 m of it (x in [13, 15]) is tagged WALL_EDGE — a free end
                 where the box terminates — instead of WALL, because a beam
                 grazing past that corner is this project's textbook case for
                 "a ray that geometrically misses the object can still share
                 a voxel with real hits from other scans" (THEORY.md "The
                 problem" derives the geometry of this precisely).
    POLE       — a thin vertical cylinder (radius 4 cm — well under one voxel
                 width at the carving stage's 20 cm voxels), the other
                 textbook discretization victim.
  DYNAMIC (present only in the scans listed; ray-cast as absent otherwise —
  i.e. beams that would have hit it instead sail on to whatever is behind it,
  or to nothing):
    CAR        — drives across the sensor's path in scans 1-4 (0-indexed),
                 at four different positions — the classic "ghost trail":
                 four clusters of points, one per scan, none of them ever
                 seen again.
    PEDESTRIAN — stands at a FIXED position for scans 0-4, then leaves for
                 good. For scans 0-4 nothing ever proves it moved (no beam
                 ever gets a clear shot through where it is standing); only
                 once it is gone do scans 5-9 carve that voxel — the "late
                 leaver", this project's hardest case (THEORY.md "The
                 temporal-evidence argument").
    GHOST      — a small crate present ONLY in scan 0, deliberately placed in
                 open space with nothing behind it along the sensor's later
                 lines of sight. Every later scan's beam toward that spot
                 finds nothing at all (a MAX-RANGE return) — this is the
                 project's designed proof that a beam which returns nothing
                 still carves free space (README/THEORY "max_range_carving").

Ground-truth label per HIT beam: 1 (dynamic) if the object it hit is CAR,
PEDESTRIAN, or GHOST; 0 (static) if WALL, WALL_EDGE, or POLE. This is an
OBJECT-CLASS truth (did this thing ever move?), not a "was it successfully
removed" truth — the whole point of the demo is to check how well raycast
carving recovers this label from evidence alone.

Noise: every hit range gets independent Gaussian noise, sigma
RANGE_NOISE_SIGMA_M (2 cm — a realistic spinning-LiDAR range noise floor),
drawn from the repo's portable xorshift32 + Box-Muller (never Python's
`random` module — CLAUDE.md paragraph 12 "no std::uniform_real_distribution"
extends here to "no hidden Mersenne Twister" for the same determinism reason:
xorshift32 is trivial to reimplement bit-for-bit in any language, which is
exactly what this project's C++ side would need to do if it ever generated
noise itself; here it does not — the noise is baked into the file once).

Usage
-----
    python make_synthetic.py                 # writes the committed sample
    python make_synthetic.py --out DIR        # experiments; do not commit
"""

import argparse
import math
import sys
from pathlib import Path

# ===========================================================================
# Deterministic RNG: xorshift32 (stdlib-only, repo convention, CLAUDE.md
# paragraph 12), seed 42. Used ONLY for per-return range noise below.
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        if s == 0:
            s = 1  # degenerate at seed 0 (stays 0 forever) — same guard used repo-wide
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
        """(0,1], never exactly 0 — safe for log() in Box-Muller below."""
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def gaussian(self, sigma: float) -> float:
        """One N(0, sigma^2) draw via Box-Muller (double precision, matching
        the flagship 08.01's gaussian() helper style: the transcendental step
        done in double so the tails stay well-behaved)."""
        u1 = self.uniform01()
        u2 = self.uniform01()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return sigma * z


DEFAULT_SEED = 42

# ===========================================================================
# Beam model — MUST MATCH src/kernels.cuh's kNumBeams/kBeamElevDeg/
# kAzimuthSteps/kNumScans/kMaxRangeM (main.cu asserts the header line below
# against those constants at load time — the same data/code consistency
# discipline 02.08's kernels.cuh cites from 02.01).
#
# The 16-beam elevation table is the SAME table 02.08/02.01 cite from 01.18's
# derivation (repo convention: -15..+15 deg in 2 deg steps) — reused here
# verbatim rather than invented fresh.
# ===========================================================================
NUM_BEAMS = 16
BEAM_ELEV_DEG = [-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
                  1.0,   3.0,   5.0,   7.0,  9.0, 11.0, 13.0, 15.0]

# Azimuth resolution: 180 steps = 2 deg/step (the repo's other spinning-
# LiDAR project, 02.08, uses 360 steps/1 deg for its deskew-error
# measurement; this project's measurement is about TEMPORAL evidence
# accumulation across scans, not angular resolution, so a documented, coarser
# sweep keeps the committed sample smaller than 02.08's — a scope reduction,
# README "Limitations"). 90 steps (4 deg) was tried first and measured: most
# static WALL voxels then received only ONE hit across all 10 scans (the
# sensor's own translation shifts the wall-intersection point of any FIXED
# beam direction by exactly the sensor's per-scan advance, so consecutive
# scans barely re-observe the same voxel at coarse resolution) — one hit
# is not enough repeat evidence to outweigh ordinary nearby pass traffic,
# and static_preservation measured a much higher false-removal rate than
# 180 steps gives (README "Expected output" states both numbers). This is
# the SAME "more independent observations sharpen a ratio estimate"
# statistics as any occupancy-grid mapping system — cite Elfes 1989.
AZIMUTH_STEPS = 180
NUM_SCANS = 10
MAX_RANGE_M = 20.0
RANGE_NOISE_SIGMA_M = 0.02          # 2 cm — realistic spinning-LiDAR noise floor
SENSOR_HEIGHT_M = 1.2

# Sensor track: straight line along x, y=0, one stop per scan (a simple,
# honest choice — orientation held at IDENTITY throughout, so "sensor frame"
# and "world frame" coincide and every beam direction below is already a
# WORLD direction; see README "Limitations" for why this is a deliberate
# scope cut, not an oversight — the DDA carving algorithm this project
# teaches does not care about platform heading at all, only where each beam
# starts and ends).
SENSOR_X0_M = -12.0
SENSOR_DX_M = 2.0     # advance per scan


def sensor_pos(scan_id: int):
    return (SENSOR_X0_M + SENSOR_DX_M * scan_id, 0.0, SENSOR_HEIGHT_M)


# ===========================================================================
# Scene objects — closed-form ray intersection, so ground truth is EXACT
# (the same "analytic scene, exact ground truth" choice project 05.01 makes
# for its sphere-over-plane, cited here as this domain's precedent for
# "render synthetic sensor data from an analytic scene instead of a mesh").
# ===========================================================================

# Cohort ids — shared with src/kernels.cuh's CohortId-style enum (mirrored,
# independently maintained: this is a data-generation-time label, not a
# runtime data structure the GPU/CPU carving code shares with this script).
COHORT_WALL = 0
COHORT_POLE = 1
COHORT_WALL_EDGE = 2
COHORT_CAR = 3
COHORT_PEDESTRIAN = 4
COHORT_GHOST = 5
COHORT_NONE = -1   # max-range beam: no object, no cohort

# Ground-truth dynamic classes (README/THEORY "Truth: per-point static/
# dynamic labels" — an OBJECT-CLASS truth, see file header).
DYNAMIC_COHORTS = {COHORT_CAR, COHORT_PEDESTRIAN, COHORT_GHOST}


def ray_aabb(origin, dir_, box_min, box_max):
    """Ray/axis-aligned-box intersection (the classic 'slab method').

    Parameters: origin, dir_ (unit) — the ray; box_min, box_max — the box's
    two opposite corners. Returns the smallest t >= 0 at which the ray enters
    the box, or None if it misses (or the box is entirely behind the ray).

    Method: clip the ray's parametric interval [t_near, t_far] against each
    axis's pair of planes in turn; the box is hit iff the three per-axis
    intervals still overlap at the end (t_near <= t_far and t_far >= 0).
    Textbook computer-graphics algorithm (Kay & Kajiya 1986); taught in full,
    independently transcribed here and in the ray/cylinder routine below —
    this script is not something the C++ pipeline imports or shares code
    with, so there is no "independence ruling" to apply, only the ordinary
    obligation to get the closed form right.
    """
    t_near, t_far = -math.inf, math.inf
    for axis in range(3):
        o, d = origin[axis], dir_[axis]
        lo, hi = box_min[axis], box_max[axis]
        if abs(d) < 1e-12:
            # Ray parallel to this axis's slab: inside iff origin is within
            # the slab; otherwise it can never enter the box on this axis.
            if o < lo or o > hi:
                return None
            continue
        t0 = (lo - o) / d
        t1 = (hi - o) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_near = max(t_near, t0)
        t_far = min(t_far, t1)
        if t_near > t_far:
            return None
    if t_far < 0.0:
        return None  # box is entirely behind the ray's origin
    return t_near if t_near >= 0.0 else t_far  # origin may start inside the box


def ray_cylinder(origin, dir_, cx, cy, z_lo, z_hi, radius):
    """Ray/vertical-cylinder intersection (finite height, capped by z_lo/z_hi).

    The infinite cylinder x'^2 + y'^2 = r^2 (x' = x - cx, y' = y - cy) meets
    the ray o + t*d where a*t^2 + b*t + c = 0 with
        a = dx^2 + dy^2,  b = 2*(dx*ox' + dy*oy'),  c = ox'^2 + oy'^2 - r^2
    (the standard 2-D-circle-in-3-D quadratic — z drops out because the
    cylinder is infinite along z until the height cap below clips it).
    Returns the smallest t >= 0 whose z-coordinate also lies in [z_lo, z_hi],
    or None. Degenerate a~=0 (a perfectly vertical ray) never crosses a
    vertical cylinder's SIDE and is correctly reported as a miss.
    """
    ox, oy = origin[0] - cx, origin[1] - cy
    dx, dy = dir_[0], dir_[1]
    a = dx * dx + dy * dy
    if a < 1e-12:
        return None
    b = 2.0 * (dx * ox + dy * oy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    for t in sorted(((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a))):
        if t < 0.0:
            continue
        z = origin[2] + t * dir_[2]
        if z_lo <= z <= z_hi:
            return t
    return None


# ---------------------------------------------------------------------------
# Static objects (present in every scan).
# ---------------------------------------------------------------------------
# Wall height deliberately spans almost the whole vertical grid extent
# (-1.8..3.8 m; the carving grid's own z bounds are -2..4 m, kernels.cuh).
# WHY floor-to-ceiling and not the more "realistic" 0..3 m first tried here:
# with only 16 discrete elevation rings across +/-15 deg, a SHORT wall's own
# TOP and BOTTOM edges clip a large fraction of elevation rings at the
#8-20 m ranges this scene actually uses (a ring that grazes just past the
# top or bottom edge sails on to nothing behind the wall and becomes a
# max-range beam whose near-wall path sits one voxel from a genuine hit —
# precisely the wall_edge grazing story, but happening at literally every
# point along the wall's LENGTH instead of only its INTENTIONAL x-direction
# terminus). Measured: a 0..3 m wall put ~30% of "generic wall" points into
# that accidental edge-contamination regime, which taught the WRONG lesson
# (that ordinary flat static structure is unreliable). A floor-to-ceiling
# wall removes almost all of that accident, isolating the DELIBERATE
# discretization cohorts (POLE, WALL_EDGE below) as the honest carriers of
# the lesson (README "Limitations" documents this design iteration).
WALL_BOX = ((-15.0, 7.8, -1.8), (13.0, 8.2, 3.8))       # generic wall face
WALL_EDGE_BOX = ((13.0, 7.8, -1.8), (15.0, 8.2, 3.8))   # the last 2 m: a free end
POLE_CENTER = (-2.0, 3.0)
POLE_RADIUS_M = 0.04
POLE_Z = (0.0, 3.0)

# ---------------------------------------------------------------------------
# Dynamic objects — per-scan presence and position.
# ---------------------------------------------------------------------------
# Car: crosses the scene at fixed x = -5 m, sweeping y from -9.5 to -5.0 m
# over scans 1..4 (0-indexed) — a box roughly car-sized (4.0 x 1.8 x 1.5 m).
# Kept AT LEAST 5 m from the sensor track (y=0) at every position — closer
# was tried first (down to y=-1.5, only 1.8 m from scan 4's own sensor
# position) and measured to under-remove: that close, a single scan's
# viewing angle spans the car at very fine angular resolution (many small
# hit voxels), but the LATER, more-distant scans that must carve those same
# voxels sample the same physical footprint at a coarser angular resolution
# and do not re-visit every one of them — a near/far resolution mismatch,
# not a bug (README "Limitations" states the measured before/after numbers).
CAR_SCANS = {1: -9.5, 2: -8.0, 3: -6.5, 4: -5.0}
CAR_X = -5.0
CAR_HALF = (2.0, 0.9, 0.75)   # half-extents (m): length, width, height/2

def car_box(y_center):
    cx, cy, cz = CAR_X, y_center, CAR_HALF[2]
    return ((cx - CAR_HALF[0], cy - CAR_HALF[1], cz - CAR_HALF[2]),
            (cx + CAR_HALF[0], cy + CAR_HALF[1], cz + CAR_HALF[2]))

# Pedestrian: fixed at (3, 2), present scans 0..4 inclusive, then gone.
PED_POS = (3.0, 2.0)
PED_HALF = (0.3, 0.3, 0.85)
PED_SCANS = set(range(0, 5))

def ped_box():
    cx, cy, cz = PED_POS[0], PED_POS[1], PED_HALF[2]
    return ((cx - PED_HALF[0], cy - PED_HALF[1], cz - PED_HALF[2]),
            (cx + PED_HALF[0], cy + PED_HALF[1], cz + PED_HALF[2]))

# Ghost crate: present ONLY in scan 0, in open space away from every other
# object so later scans' rays toward it find nothing else in the way.
GHOST_POS = (-10.0, -6.0, 0.0)
GHOST_HALF = (0.3, 0.3, 0.3)
GHOST_SCANS = {0}

def ghost_box():
    cx, cy, cz = GHOST_POS[0], GHOST_POS[1], GHOST_HALF[2]
    return ((cx - GHOST_HALF[0], cy - GHOST_HALF[1], cz - GHOST_HALF[2]),
            (cx + GHOST_HALF[0], cy + GHOST_HALF[1], cz + GHOST_HALF[2]))


def active_objects(scan_id: int):
    """The list of (kind, params, cohort) ray-castable objects for one scan.

    kind is 'box' or 'cyl'; params match ray_aabb/ray_cylinder's signature
    (minus origin/dir_, which the caller supplies per beam). Order does not
    matter — the caller takes the globally nearest hit across the whole list.
    """
    objs = [
        ('box', WALL_BOX, COHORT_WALL),
        ('box', WALL_EDGE_BOX, COHORT_WALL_EDGE),
        ('cyl', (POLE_CENTER[0], POLE_CENTER[1], POLE_Z[0], POLE_Z[1], POLE_RADIUS_M), COHORT_POLE),
    ]
    if scan_id in CAR_SCANS:
        objs.append(('box', car_box(CAR_SCANS[scan_id]), COHORT_CAR))
    if scan_id in PED_SCANS:
        objs.append(('box', ped_box(), COHORT_PEDESTRIAN))
    if scan_id in GHOST_SCANS:
        objs.append(('box', ghost_box(), COHORT_GHOST))
    return objs


def cast_ray(origin, dir_, scan_id: int):
    """Nearest hit across every active object this scan, or None (a miss).

    Returns (t, cohort) for the closest positive intersection, exactly as a
    real LiDAR return would report whichever surface is nearest along the
    beam — occlusion falls out for free from taking the minimum t.
    """
    best_t = None
    best_cohort = COHORT_NONE
    for kind, params, cohort in active_objects(scan_id):
        if kind == 'box':
            t = ray_aabb(origin, dir_, params[0], params[1])
        else:
            t = ray_cylinder(origin, dir_, *params)
        if t is not None and t >= 0.0 and (best_t is None or t < best_t):
            best_t = t
            best_cohort = cohort
    return best_t, best_cohort


def beam_direction(elev_deg: float, az_deg: float):
    """Unit direction for one beam, spherical convention: az measured CCW
    from +x in the xy-plane, elev measured up from the xy-plane — the
    standard right-handed x-forward/y-left/z-up convention (CLAUDE.md
    paragraph 12). Since the sensor's orientation is identity (file header),
    this IS already the world-frame direction — no rotation needed."""
    el = math.radians(elev_deg)
    az = math.radians(az_deg)
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def generate(out_dir: Path, seed: int) -> None:
    rng = Xorshift32(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    beams_path = out_dir / "beams.csv"
    poses_path = out_dir / "poses.csv"

    # Diagnostic tallies (printed at the end — sanity numbers for whoever
    # regenerates the sample, NOT part of the graded demo output).
    cohort_hit_counts = {COHORT_WALL: 0, COHORT_POLE: 0, COHORT_WALL_EDGE: 0,
                         COHORT_CAR: 0, COHORT_PEDESTRIAN: 0, COHORT_GHOST: 0}
    n_hits = 0
    n_miss = 0

    with beams_path.open("w", encoding="utf-8", newline="\n") as bf:
        bf.write("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 02.13\n")
        bf.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        bf.write("# scene: static WALL/WALL_EDGE/POLE + dynamic CAR/PEDESTRIAN/GHOST; see this\n")
        bf.write("#        script's module docstring for the full scene description\n")
        bf.write(f"# num_scans={NUM_SCANS}\n")
        bf.write(f"# num_beams={NUM_BEAMS}\n")
        bf.write(f"# azimuth_steps={AZIMUTH_STEPS}\n")
        bf.write(f"# max_range_m={MAX_RANGE_M}\n")
        bf.write(f"# range_noise_sigma_m={RANGE_NOISE_SIGMA_M}\n")
        bf.write(f"# seed={seed}\n")
        bf.write("# cohort ids: 0=WALL 1=POLE 2=WALL_EDGE 3=CAR(dyn) 4=PEDESTRIAN(dyn) 5=GHOST(dyn) -1=NONE(miss)\n")
        bf.write("# columns: scan_id,dir_x,dir_y,dir_z,is_hit,range_m,cohort,truth_dynamic\n")

        for scan_id in range(NUM_SCANS):
            origin = sensor_pos(scan_id)
            for elev_deg in BEAM_ELEV_DEG:
                for az_i in range(AZIMUTH_STEPS):
                    az_deg = az_i * (360.0 / AZIMUTH_STEPS)
                    dir_ = beam_direction(elev_deg, az_deg)
                    t, cohort = cast_ray(origin, dir_, scan_id)

                    if t is not None and t <= MAX_RANGE_M:
                        # Real return: perturb the measured range with the
                        # sensor's noise floor (never negative — a physical
                        # rangefinder cannot report a negative distance).
                        r_noisy = max(0.01, t + rng.gaussian(RANGE_NOISE_SIGMA_M))
                        truth = 1 if cohort in DYNAMIC_COHORTS else 0
                        bf.write(f"{scan_id},{dir_[0]:.8f},{dir_[1]:.8f},{dir_[2]:.8f},"
                                f"1,{r_noisy:.8f},{cohort},{truth}\n")
                        cohort_hit_counts[cohort] += 1
                        n_hits += 1
                    else:
                        # No return within range: a MAX-RANGE beam — carries
                        # no point, but the carving stage still walks it the
                        # full MAX_RANGE_M to mark free space along the way.
                        bf.write(f"{scan_id},{dir_[0]:.8f},{dir_[1]:.8f},{dir_[2]:.8f},"
                                f"0,{MAX_RANGE_M:.8f},{COHORT_NONE},-1\n")
                        n_miss += 1

    with poses_path.open("w", encoding="utf-8", newline="\n") as pf:
        pf.write("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 02.13\n")
        pf.write("# sensor poses: identity orientation throughout (see module docstring)\n")
        pf.write("# columns: scan_id,px,py,pz,qw,qx,qy,qz,t_s\n")
        for scan_id in range(NUM_SCANS):
            px, py, pz = sensor_pos(scan_id)
            t_s = scan_id * 0.5   # 2 Hz mapping-session cadence (illustrative; README/PRACTICE discuss real cadence)
            pf.write(f"{scan_id},{px:.6f},{py:.6f},{pz:.6f},1.0,0.0,0.0,0.0,{t_s:.3f}\n")

    total = n_hits + n_miss
    print(f"[make_synthetic] wrote {total} beams ({n_hits} hits, {n_miss} max-range misses) "
          f"across {NUM_SCANS} scans to {beams_path}")
    print(f"[make_synthetic] wrote {NUM_SCANS} poses to {poses_path}")
    print("[make_synthetic] per-cohort hit counts:")
    for name, cid in [("WALL", COHORT_WALL), ("POLE", COHORT_POLE), ("WALL_EDGE", COHORT_WALL_EDGE),
                       ("CAR", COHORT_CAR), ("PEDESTRIAN", COHORT_PEDESTRIAN), ("GHOST", COHORT_GHOST)]:
        print(f"    {name:11s}: {cohort_hit_counts[cid]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default: ../data/sample)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"xorshift32 seed (default: {DEFAULT_SEED})")
    args = ap.parse_args()
    generate(args.out, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
