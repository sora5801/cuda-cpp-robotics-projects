#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.14
(Moving-object segmentation from sequential scans).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Online MOS needs a short WINDOW of posed scans (current + M=4 previous) with
an OBJECT-LEVEL ground truth for which points belong to something moving
RIGHT NOW versus permanent structure — and, critically, every interesting
FLAVOR of motion at once: a lateral (crossing) mover, a purely radial
approach, a purely radial departure, and a mover that only just stopped. No
public dataset hands you this decomposition with exact, noise-free ground
truth; this script builds an analytic scene, ray-casts a real 16-beam
spinning LiDAR through it from 5 sensor positions, and writes the resulting
per-scan hit points (plus ground truth) to ../data/sample/. No download, no
license question, bit-for-bit reproducible from a fixed seed (42, xorshift32
— CLAUDE.md paragraph 12: no Python `random` module, no
std::uniform_real_distribution equivalent).

THE SCENE (world frame: x-forward, y-left, z-up, right-handed; meters).
------------------------------------------------------------------------
A sensor drives slowly along +x (SENSOR_SPEED_MS) while the window's 5 scans
are captured DT_S apart (a documented, reduced sampling rate for the
multi-scan comparison buffer — README "Limitations" states the real
per-scan rate the algorithm's own compute must meet separately). Every
object is an axis-aligned box (AABB) except the pole, a vertical cylinder —
both have closed-form ray intersections (Kay & Kajiya 1986), so ground truth
is exact, no approximation:

  STATIC (present every scan, identical position):
    WALL — a tall, long wall at y ~= 15 m (floor-to-ceiling in z, the same
           02.13 lesson cited in this project's THEORY.md: a SHORT wall
           would let ordinary elevation-ring clipping masquerade as the
           deliberate discretization cohorts below). The crossing car (see
           below) passes directly between the sensor and a section of this
           wall, producing this project's disocclusion band.
    POLE — a thin vertical cylinder (radius 5 cm, well under one azimuth
           bin's angular footprint at its range) — the discretization
           honesty cohort (THEORY.md "The problem" derives the aliasing).

  DYNAMIC (present every scan, but at a DIFFERENT position each scan — ray-
  cast fresh per scan_id, so occlusion between movers and statics falls out
  for free from "nearest hit wins"):
    CROSSING_CAR — moves laterally (approximately constant range, sweeping
           azimuth) directly in front of the WALL: the "easy case" (README)
           and simultaneously the source of this project's occlusion /
           disocclusion band on the wall behind it.
    ONCOMING_CAR — moves along a FIXED azimuth, RELATIVE TO THE SENSOR'S OWN
           (moving) position each scan, with DECREASING range: pure radial
           approach — the negative-residual showcase (THEORY.md derives the
           sign).
    RECEDING_CAR — same construction, INCREASING range: pure radial
           departure — the positive-residual showcase.
    STOPPED_CAR — drives (a different position each scan) for scans 0-3,
           then holds its scan-3 position for scan 4 (the CURRENT scan) —
           i.e. it has JUST stopped as of "now". This project's temporal-
           boundary cohort (README "Expected output" / THEORY.md "Numerical
           considerations" measure what MIN-fusion does with a mover that
           just became stationary).

Ground-truth label per CURRENT-SCAN (scan_id=4) hit point: truth_dynamic=1
if the cohort is CROSSING_CAR/ONCOMING_CAR/RECEDING_CAR/STOPPED_CAR, else 0
(WALL/POLE). disocclusion_band is computed ONLY for WALL hits in the current
scan: for each of the 5 scan indices, ray-cast from THAT scan's sensor
position toward this exact world point and test whether CROSSING_CAR's
THEN-position blocks it; disocclusion_band=1 iff that occlusion status is
NOT the same across all 5 scans (the crossing car's passage toggled
visibility of this exact wall point at some point in the window) — the
precise, analytic definition of "this wall point sits in the disocclusion
band" that main.cu's disocclusion_mitigation gate reads (ground truth used
ONLY by gates/artifacts, never by the reprojection/residual/CCL algorithm —
kernels.cuh's "Ground truth" note).

Noise: every stored hit range gets independent Gaussian noise, sigma
RANGE_NOISE_SIGMA_M, drawn from the repo's portable xorshift32 + Box-Muller
generator (never Python's `random` module).

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
# Identical implementation to 02.13's Xorshift32 (retyped fresh here — a
# data-generation-time script, not shared code the C++ pipeline imports).
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
        """One N(0, sigma^2) draw via Box-Muller (double precision — same
        style as 08.01's/02.13's gaussian() helper: the transcendental step
        done in double so the tails stay well-behaved)."""
        u1 = self.uniform01()
        u2 = self.uniform01()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return sigma * z


DEFAULT_SEED = 42

# ===========================================================================
# Beam model + range-image shape — MUST MATCH src/kernels.cuh's kNumBeams/
# kBeamElevMinDeg/kBeamElevStepDeg/kAzimuthBins/kMaxRangeM/kNumScansWindow
# (main.cu asserts the data file's header against those at load time — the
# 02.08/02.13-style data/code consistency discipline).
# ===========================================================================
NUM_BEAMS = 16
BEAM_ELEV_MIN_DEG = -15.0
BEAM_ELEV_STEP_DEG = 2.0            # rings: -15, -13, ..., +15 (16 rings)
AZIMUTH_BINS = 360                  # 1 deg/bin (matches 02.08's sweep resolution, cited)
MAX_RANGE_M = 30.0
RANGE_NOISE_SIGMA_M = 0.02          # 2 cm — the repo's realistic spinning-LiDAR noise floor (02.08/02.13)

NUM_SCANS_WINDOW = 5                # current (index 4) + 4 previous (indices 0-3)
CURRENT_SCAN_IDX = NUM_SCANS_WINDOW - 1

# Sensor trajectory — straight line along +x, fixed y/z, IDENTITY
# orientation throughout (the same documented scope cut 02.13 makes for its
# own sensor track, cited): DT_S is the SAMPLING interval BETWEEN the 5
# buffered window scans, a deliberately reduced rate from the sensor's own
# 10 Hz native scan period, chosen so the window spans enough real time for
# the designed object speeds to clear the residual noise floor with margin
# (README "Limitations" states this honestly; main.cu's timing gate measures
# the ALGORITHM's own per-call latency against the native 10-20 Hz budget,
# a separate question from how far apart the buffered comparison scans are).
SENSOR_SPEED_MS = 1.0
SENSOR_HEIGHT_M = 1.5
DT_S = 0.3


def sensor_pos(scan_id: int):
    t = scan_id * DT_S
    return (SENSOR_SPEED_MS * t, 0.0, SENSOR_HEIGHT_M)


# ===========================================================================
# Scene objects — closed-form ray intersection (Kay & Kajiya 1986), so
# ground truth is EXACT (02.13's identical "analytic scene" choice, cited).
# ===========================================================================

COHORT_WALL = 0
COHORT_POLE = 1
COHORT_CROSSING_CAR = 2
COHORT_ONCOMING_CAR = 3
COHORT_RECEDING_CAR = 4
COHORT_STOPPED_CAR = 5
COHORT_NONE = -1

DYNAMIC_COHORTS = {COHORT_CROSSING_CAR, COHORT_ONCOMING_CAR, COHORT_RECEDING_CAR, COHORT_STOPPED_CAR}


def ray_aabb(origin, dir_, box_min, box_max):
    """Ray/axis-aligned-box intersection (the classic 'slab method',
    identical algorithm to 02.13's ray_aabb — retyped fresh here, a data-
    generation-time script with no shared code path to the C++ pipeline)."""
    t_near, t_far = -math.inf, math.inf
    for axis in range(3):
        o, d = origin[axis], dir_[axis]
        lo, hi = box_min[axis], box_max[axis]
        if abs(d) < 1e-12:
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
        return None
    return t_near if t_near >= 0.0 else t_far


def ray_cylinder(origin, dir_, cx, cy, z_lo, z_hi, radius):
    """Ray/vertical-cylinder intersection (finite height) — identical
    algorithm to 02.13's ray_cylinder, retyped fresh (see that script's
    docstring for the full quadratic derivation)."""
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
# Static objects.
# ---------------------------------------------------------------------------
# Floor-to-ceiling (z in [-3,7], a full 10 m) so the wall comfortably spans
# this scene's vertical FOV at its ~15-16 m range (+-15 deg elevation at
# 15 m subtends about +-4.0 m about the sensor's 1.5 m mount height, i.e.
# world z in about [-2.5, 5.5]) — a SHORT wall would let ordinary ring
# clipping masquerade as the deliberate POLE discretization cohort (02.13's
# identical lesson, cited in THEORY.md "The problem").
WALL_BOX = ((-20.0, 14.8, -3.0), (20.0, 15.2, 7.0))
POLE_CENTER = (5.0, 9.0)
POLE_RADIUS_M = 0.05
POLE_Z = (-1.0, 3.0)

# Crossing car: lateral sweep at fixed y=11 (between the sensor and the
# WALL's y=15 plane), x advancing east each scan — passes in front of a
# section of the wall, producing the occlusion/disocclusion band.
#
# STEP SIZE IS LOAD-BEARING (a measured design choice, not arbitrary): the
# per-scan step (4.0 m) is deliberately LARGER than the car's own footprint
# (CAR_HALF gives a ~3.6-4.0 m silhouette depending on viewing angle). A
# first attempt used a much slower sweep (0.9 m/scan, well under the car's
# own size) and measured two compounding problems: (1) mover_detection
# recall on this cohort was only ~34% — because the car's CURRENT position
# overlapped its OWN position from 1-2 scans back, so for THOSE lags the
# residual was small (car vs car, similar range) and MIN-fusion (which
# takes the SMALLEST |residual| across the window) picked that small value,
# suppressing detection even though OTHER lags correctly showed a huge
# residual (car vs wall); (2) disocclusion_mitigation measured ZERO
# improvement — the car's shadow on the wall persisted across every
# included previous scan (never a one-off event), so there was nothing for
# multi-scan consistency to filter. Widening the step so consecutive
# scans' footprints on the wall are mostly DISJOINT fixes both: the car
# never re-occupies a cell it or its own shadow held one scan ago, and a
# given wall cell is now occluded in only a MINORITY of the 4 previous
# scans — exactly the "one-off blip vs. persistent change" contrast
# main.cu's disocclusion_mitigation gate is designed to measure (THEORY.md
# "The problem" derives this in full).
CROSSING_X = [-8.0, -4.0, 0.0, 4.0, 8.0]
CROSSING_Y = 11.0
CAR_HALF = (2.0, 0.9, 0.75)   # half-extents (m): length, width, height/2 — 02.13's CAR_HALF, cited

def crossing_car_box(scan_id: int):
    cx, cy, cz = CROSSING_X[scan_id], CROSSING_Y, CAR_HALF[2]
    return ((cx - CAR_HALF[0], cy - CAR_HALF[1], cz - CAR_HALF[2]),
            (cx + CAR_HALF[0], cy + CAR_HALF[1], cz + CAR_HALF[2]))


# Oncoming / receding cars: positioned RELATIVE TO THE SENSOR'S OWN (moving)
# position each scan, along a FIXED azimuth, at a range that strictly
# decreases (oncoming) or increases (receding) — an EXACT radial trajectory
# with respect to the sensor at every instant (kernels.cuh's sign-semantics
# derivation assumes exactly this: azimuth/elevation invariant, range-only
# change — THEORY.md "The math" derives why this is the clean showcase).
ONCOMING_AZ_DEG = 200.0
ONCOMING_RANGE_M = [20.0, 19.0, 18.0, 17.0, 16.0]      # strictly decreasing: approaching
RECEDING_AZ_DEG = 340.0
RECEDING_RANGE_M = [8.0, 9.0, 10.0, 11.0, 12.0]        # strictly increasing: receding


def _radial_car_box(scan_id: int, az_deg: float, range_m: float):
    ox, oy, oz = sensor_pos(scan_id)
    az = math.radians(az_deg)
    cx = ox + range_m * math.cos(az)
    cy = oy + range_m * math.sin(az)
    cz = CAR_HALF[2]   # box sits on the ground (z=0) regardless of sensor mount height
    return ((cx - CAR_HALF[0], cy - CAR_HALF[1], cz - CAR_HALF[2]),
            (cx + CAR_HALF[0], cy + CAR_HALF[1], cz + CAR_HALF[2]))

def oncoming_car_box(scan_id: int):
    return _radial_car_box(scan_id, ONCOMING_AZ_DEG, ONCOMING_RANGE_M[scan_id])

def receding_car_box(scan_id: int):
    return _radial_car_box(scan_id, RECEDING_AZ_DEG, RECEDING_RANGE_M[scan_id])


# Stopped car: drives scans 0-3 (a different WORLD position each scan, NOT
# relative to the sensor — an ordinary driving car, unrelated to the
# sensor's own motion), then HOLDS its scan-3 position for scan 4 (current)
# — "just stopped" as of now (module docstring's temporal-boundary cohort).
STOPPED_X = [-10.0, -8.8, -7.6, -6.4, -6.4]   # scan 4 repeats scan 3's position on purpose
STOPPED_Y = -14.0

def stopped_car_box(scan_id: int):
    cx, cy, cz = STOPPED_X[scan_id], STOPPED_Y, CAR_HALF[2]
    return ((cx - CAR_HALF[0], cy - CAR_HALF[1], cz - CAR_HALF[2]),
            (cx + CAR_HALF[0], cy + CAR_HALF[1], cz + CAR_HALF[2]))


def active_objects(scan_id: int):
    """Every ray-castable object for this scan_id (kind, params, cohort)."""
    return [
        ('box', WALL_BOX, COHORT_WALL),
        ('cyl', (POLE_CENTER[0], POLE_CENTER[1], POLE_Z[0], POLE_Z[1], POLE_RADIUS_M), COHORT_POLE),
        ('box', crossing_car_box(scan_id), COHORT_CROSSING_CAR),
        ('box', oncoming_car_box(scan_id), COHORT_ONCOMING_CAR),
        ('box', receding_car_box(scan_id), COHORT_RECEDING_CAR),
        ('box', stopped_car_box(scan_id), COHORT_STOPPED_CAR),
    ]


def cast_ray(origin, dir_, scan_id: int):
    """Nearest hit across every active object this scan, or (None, NONE) —
    occlusion falls out for free from taking the minimum t (02.13's
    identical cast_ray pattern, cited)."""
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
    """Unit beam direction — IDENTICAL formula to src/kernels.cuh's
    beam_dir_local() (both independently state the same textbook spherical
    convention; this script and the C++ pipeline share no code, per the
    self-containment rule, CLAUDE.md paragraph 4)."""
    el = math.radians(elev_deg)
    az = math.radians(az_deg)
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec_norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

def vec_normalize(a):
    n = vec_norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def wall_point_occluded_by_crossing_car(world_point, scan_id: int) -> bool:
    """Ray-cast from scan_id's OWN sensor position toward world_point (an
    EXACT world-space point on the wall) and test whether CROSSING_CAR's
    THEN-position (its box at this scan_id) blocks the line of sight at a
    strictly smaller parameter than the wall point itself. Used ONLY to
    derive the disocclusion_band ground-truth flag (module docstring) —
    never read by the algorithm under test."""
    origin = sensor_pos(scan_id)
    to_point = vec_sub(world_point, origin)
    dist = vec_norm(to_point)
    if dist < 1e-9:
        return False
    dir_ = vec_normalize(to_point)
    box = crossing_car_box(scan_id)
    t_car = ray_aabb(origin, dir_, box[0], box[1])
    return (t_car is not None) and (t_car < dist - 1e-6)


def generate(out_dir: Path, seed: int) -> None:
    rng = Xorshift32(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    scans_path = out_dir / "scans.csv"
    poses_path = out_dir / "poses.csv"

    cohort_names = {COHORT_WALL: "WALL", COHORT_POLE: "POLE", COHORT_CROSSING_CAR: "CROSSING_CAR",
                    COHORT_ONCOMING_CAR: "ONCOMING_CAR", COHORT_RECEDING_CAR: "RECEDING_CAR",
                    COHORT_STOPPED_CAR: "STOPPED_CAR"}
    cohort_hit_counts = {c: 0 for c in cohort_names}
    n_hits = 0
    n_miss = 0
    n_disocclusion_band = 0

    with scans_path.open("w", encoding="utf-8", newline="\n") as sf:
        sf.write("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 02.14\n")
        sf.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        sf.write("# scene: static WALL/POLE + dynamic CROSSING_CAR/ONCOMING_CAR/RECEDING_CAR/STOPPED_CAR;\n")
        sf.write("#        see this script's module docstring for the full scene description\n")
        sf.write(f"# num_scans_window={NUM_SCANS_WINDOW}\n")
        sf.write(f"# num_beams={NUM_BEAMS}\n")
        sf.write(f"# azimuth_bins={AZIMUTH_BINS}\n")
        sf.write(f"# max_range_m={MAX_RANGE_M}\n")
        sf.write(f"# range_noise_sigma_m={RANGE_NOISE_SIGMA_M}\n")
        sf.write(f"# seed={seed}\n")
        sf.write("# cohort ids: 0=WALL 1=POLE 2=CROSSING_CAR(dyn) 3=ONCOMING_CAR(dyn) 4=RECEDING_CAR(dyn)"
                  " 5=STOPPED_CAR(dyn) -1=NONE(miss)\n")
        sf.write("# columns: scan_id,ring,az_bin,range_m,cohort,truth_dynamic,disocclusion_band\n")

        for scan_id in range(NUM_SCANS_WINDOW):
            origin = sensor_pos(scan_id)
            for ring in range(NUM_BEAMS):
                elev_deg = BEAM_ELEV_MIN_DEG + ring * BEAM_ELEV_STEP_DEG
                for az_bin in range(AZIMUTH_BINS):
                    az_deg = az_bin * (360.0 / AZIMUTH_BINS)
                    dir_ = beam_direction(elev_deg, az_deg)
                    t, cohort = cast_ray(origin, dir_, scan_id)

                    if t is not None and t <= MAX_RANGE_M:
                        r_noisy = max(0.01, t + rng.gaussian(RANGE_NOISE_SIGMA_M))
                        truth = 1 if cohort in DYNAMIC_COHORTS else 0

                        disocc = 0
                        if scan_id == CURRENT_SCAN_IDX and cohort == COHORT_WALL:
                            # Ground truth only makes sense on the CURRENT
                            # scan's own wall hits (module docstring): use
                            # the EXACT (noise-free) world point so the
                            # occlusion re-cast below is geometrically clean.
                            world_point = (origin[0] + dir_[0] * t, origin[1] + dir_[1] * t, origin[2] + dir_[2] * t)
                            statuses = [wall_point_occluded_by_crossing_car(world_point, s)
                                        for s in range(NUM_SCANS_WINDOW)]
                            if len(set(statuses)) > 1:
                                disocc = 1
                                n_disocclusion_band += 1

                        sf.write(f"{scan_id},{ring},{az_bin},{r_noisy:.8f},{cohort},{truth},{disocc}\n")
                        cohort_hit_counts[cohort] += 1
                        n_hits += 1
                    else:
                        n_miss += 1
                        # No return: simply not written (organized cell stays
                        # "no data" — kernels.cuh's file header convention).

    with poses_path.open("w", encoding="utf-8", newline="\n") as pf:
        pf.write("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 02.14\n")
        pf.write("# sensor poses: identity orientation throughout (see module docstring)\n")
        pf.write("# columns: scan_id,px,py,pz,qw,qx,qy,qz,t_s\n")
        for scan_id in range(NUM_SCANS_WINDOW):
            px, py, pz = sensor_pos(scan_id)
            t_s = scan_id * DT_S
            pf.write(f"{scan_id},{px:.6f},{py:.6f},{pz:.6f},1.0,0.0,0.0,0.0,{t_s:.3f}\n")

    total = n_hits + n_miss
    print(f"[make_synthetic] wrote {total} beam samples ({n_hits} hits, {n_miss} no-returns) across "
          f"{NUM_SCANS_WINDOW} scans to {scans_path}")
    print(f"[make_synthetic] wrote {NUM_SCANS_WINDOW} poses to {poses_path}")
    print(f"[make_synthetic] disocclusion-band WALL points (current scan): {n_disocclusion_band}")
    print("[make_synthetic] per-cohort hit counts:")
    for cid, name in cohort_names.items():
        print(f"    {name:14s}: {cohort_hit_counts[cid]}")


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
