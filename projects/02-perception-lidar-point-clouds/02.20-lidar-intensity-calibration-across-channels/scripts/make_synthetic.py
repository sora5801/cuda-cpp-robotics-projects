#!/usr/bin/env python3
"""make_synthetic.py - synthetic sample-data generator for 02.20
(LiDAR intensity calibration across channels).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Self-calibrating 16 LiDAR channels' intensity gains needs something no public
dataset hands you: a per-point ground-truth per-CHANNEL gain, per-point
reflectivity, and a known range-falloff curve, so recovery can be graded
against a real answer. This script RENDERS the physics instead: a real
16-beam spinning LiDAR ray-casts a closed-form structured scene (ground
plane, a near wall, a small high-reflectivity test panel beside the near
wall, and a far wall), and a documented per-point intensity forward model
applies each beam's TRUE, hidden channel gain. Every point's surface id,
true reflectivity, and generating channel are exact by construction. No
download, no license question, bit-for-bit reproducible from a fixed seed
(42, xorshift32 - CLAUDE.md paragraph 12: no Python `random` module).

THE FORWARD MODEL (THEORY.md "The math" derives this in full; this is the
one-paragraph version). Measured intensity for a beam of channel `ch`
striking a surface point at range r_m with incidence angle theta from that
surface's normal:

    I = g[ch] * R_surface * f(r_m) * cos(theta) + noise

  * g[ch]      - the per-channel GAIN this project recovers. TRUE_GAINS below
                 span 0.6-1.4 (documented, illustrative "detector aging /
                 alignment variance" magnitude - never fabricated as
                 measured hardware data).
  * R_surface  - the struck surface's Lambertian reflectivity (unitless,
                 documented per surface below): this is the "material"
                 term a real calibration procedure does NOT know per point.
  * f(r_m)     - range-falloff: f(r) = (kRangePlateauM / max(r, kRangePlateauM))^2
                 - a 1/r^2 regime for r > kRangePlateauM, clamped flat inside
                 it (the near-range defocus-plateau shape a real receiver's
                 optics produce; THEORY.md "The problem" derives why). This
                 project's committed geometry stays entirely in the 1/r^2
                 regime (nearest surface ~8 m, well past the 4 m plateau
                 radius) - an honest, stated scope cut (README "Limitations").
  * cos(theta) - Lambertian incidence-angle falloff, theta measured from the
                 struck surface's own analytic normal (all surfaces here are
                 axis-aligned planes, so this is exact, not estimated).
  * noise      - small multiplicative + additive Gaussian sensor noise
                 (kNoiseSigmaMult / kNoiseSigmaAdd below).

THE SCENE (sensor frame: x-forward, y-left, z-up, right-handed, meters,
SI, matching docs/SYSTEM_DESIGN.md's convention; sensor fixed at the origin
- a deliberate scope cut shared with project 02.18's generator: this project
compares ONE static structured scene's self-consistency across channels, not
a moving platform). Four real, closed-form planar surfaces (exact ray
intersection, so ground truth is exact - the same choice 02.13/02.18 make):

    GROUND      - horizontal plane z = -kSensorHeightM (the workhorse of the
                  "far, grazing-incidence, dim" cohort: 1.2 m below the
                  sensor, hit only where nothing taller stands in the way).
    WALL_NEAR   - vertical plane x = kWallNearXM (~8 m ahead): a broad,
                  near-normal-incidence, BRIGHT surface. Because it is wide
                  and every one of the 16 channels' rays sweep across it at
                  a similar range, MANY (voxel, channel-pair) observations
                  land here - this is the project's main calibration
                  currency (THEORY.md "The algorithm").
    PANEL       - a SMALL patch beside WALL_NEAR, same range, DIFFERENT
                  reflectivity (kRPanel, a bright "test-target"-like
                  material) - the designed "multi-material cohort at similar
                  range" project 02.20's multi_material_robustness gate
                  measures (README "Expected output"): if the shared-voxel
                  currency truly cancels R_surface, mixing this patch in
                  must not corrupt the gain solve.
    WALL_FAR    - vertical plane x = kWallFarXM (~20 m ahead): the same
                  physical wall material as WALL_NEAR, now naturally SPARSE
                  cross-channel overlap purely from beam angular spacing at
                  range (the same 1/r footprint lesson project 02.01 teaches
                  from the opposite direction) - contributes fewer, noisier
                  shared voxels, honestly, not hidden.

DEGENERATE SCAN VARIANT (scan_degenerate.csv) - the observability lesson
(THEORY.md "The math": observability as GRAPH connectivity; the 01.17
precedent restated as a graph statement). One channel (kDegenerateChannel,
the highest-elevation beam) is retargeted from its normal elevation to a
steep upward angle that ONLY ever strikes a fifth surface, ISOLATED_TARGET -
a horizontal ceiling-like plane far above every other surface's extent. No
other channel's ray ever reaches it, so this one channel's points share NO
voxel with any other channel: its gain is structurally UNOBSERVABLE from
this scan, and the calibration pipeline's job is to DETECT that (main.cu's
unobservable_channel gate) rather than silently emit a hallucinated gain.

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
# paragraph 12), seed 42. Identical shape to 02.18's/08.01's generator.
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        if s == 0:
            s = 1  # degenerate at seed 0 (stays 0 forever) - same guard used repo-wide
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
        """(0,1], never exactly 0 - safe for log() below."""
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def gaussian(self, sigma: float) -> float:
        """One N(0, sigma^2) draw via Box-Muller (double precision)."""
        u1 = self.uniform01()
        u2 = self.uniform01()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return sigma * z


DEFAULT_SEED = 42

# ===========================================================================
# Beam model - MUST MATCH ../src/kernels.cuh's kNumBeams/kElevMinDeg/
# kElevStepDeg/kAzimuthMinDeg/kAzimuthStepDeg/kAzimuthSteps/kMaxRangeM (main.cu
# asserts the data file's '#'-prefixed header against those, the 02.08/
# 02.13/02.18-style data/code consistency check).
# ===========================================================================
NUM_BEAMS = 16
ELEV_MIN_DEG = -4.0
ELEV_MAX_DEG = 4.0
ELEV_STEP_DEG = (ELEV_MAX_DEG - ELEV_MIN_DEG) / (NUM_BEAMS - 1)   # 15 gaps across 16 beams
ELEV_DEG = [ELEV_MIN_DEG + i * ELEV_STEP_DEG for i in range(NUM_BEAMS)]

AZIMUTH_MIN_DEG = -40.0
AZIMUTH_STEP_DEG = 1.0
AZIMUTH_STEPS = 81             # covers [-40, +40] deg inclusive
MAX_RANGE_M = 45.0

SENSOR_HEIGHT_M = 1.2           # sensor mounted this high above the ground plane

# Degenerate-scan retargeting (file header "DEGENERATE SCAN VARIANT").
DEGENERATE_CHANNEL = 15                    # the top beam (elevation index 15)
DEGENERATE_ELEV_DEG = 45.0                 # steep upward angle, hits ONLY the isolated target

RANGE_NOISE_SIGMA_M = 0.02        # real-surface range noise (02.13/02.18's value)
NOISE_SIGMA_MULT = 0.03           # multiplicative intensity noise (shot-noise-like), fraction of signal
NOISE_SIGMA_ADD = 0.0008          # additive intensity noise floor (read-noise-like)

# ===========================================================================
# Scene geometry - closed-form ray/plane intersection with rectangular
# extents (the same "analytic scene, exact ground truth" choice
# 02.01/02.13/02.18 make). MUST MATCH ../src/kernels.cuh's SECTION "scene
# geometry" constants (main.cu's classify_surface_normal() re-derives normals
# from these SAME numbers, independently of the ground-truth surf_id column
# below - see kernels.cuh's file header for why that is still honest).
# ===========================================================================
GROUND_Z_M = -SENSOR_HEIGHT_M

WALL_NEAR_X_M = 8.0
WALL_NEAR_Y_RANGE = (-2.2, 2.2)     # lateral (left-right) extent, meters
WALL_NEAR_Z_RANGE = (-1.5, 3.0)     # vertical extent, meters

PANEL_X_M = 8.0                      # same range as WALL_NEAR - the multi-material-at-similar-range cohort
PANEL_Y_RANGE = (2.4, 3.0)           # a small patch just beyond the wall's edge
PANEL_Z_RANGE = (-0.5, 2.0)

WALL_FAR_X_M = 20.0
WALL_FAR_Y_RANGE = (-11.0, 11.0)    # deliberately WIDER angular footprint than WALL_NEAR+PANEL so it
WALL_FAR_Z_RANGE = (-3.0, 5.0)      # pokes out beyond their occlusion shadow instead of being hidden (below)

ISOLATED_TARGET_Z_M = 6.0            # a horizontal "ceiling" far above every other surface's extent

SURF_GROUND, SURF_WALL_NEAR, SURF_PANEL, SURF_WALL_FAR, SURF_ISOLATED = 0, 1, 2, 3, 4
SURF_NAMES = {SURF_GROUND: "ground", SURF_WALL_NEAR: "wall_near", SURF_PANEL: "panel",
              SURF_WALL_FAR: "wall_far", SURF_ISOLATED: "isolated_target"}

# Reflectivities rho per surface (unitless, illustrative magnitudes, dated
# 2026-07-12 - real values depend on material/wavelength/finish; verify
# current before relying on any of them beyond teaching). Deliberately span a
# realistic low-to-high range: PANEL is the brightest (a retroreflective-ish
# test-target material) so the multi-material cohort is a REAL contrast, not
# a token one.
R_GROUND = 0.22
R_WALL_NEAR = 0.55
R_PANEL = 0.85
R_WALL_FAR = 0.35
R_ISOLATED = 0.45
R_OF = {SURF_GROUND: R_GROUND, SURF_WALL_NEAR: R_WALL_NEAR, SURF_PANEL: R_PANEL,
        SURF_WALL_FAR: R_WALL_FAR, SURF_ISOLATED: R_ISOLATED}

# ===========================================================================
# TRUE per-channel gains (file header "THE FORWARD MODEL"). Illustrative
# "detector aging / laser power / alignment variance" magnitudes, dated
# 2026-07-12 - deliberately NON-monotonic (real per-channel gain drift is not
# a clean ramp). Spans the documented 0.6-1.4 range. THIS ARRAY IS GROUND
# TRUTH: it is written to gains_true.csv for main.cu's gates ONLY - never
# read by the calibration algorithm itself (kernels.cu/reference_cpu.cpp).
# ===========================================================================
TRUE_GAINS = [
    0.62, 1.18, 0.85, 1.35, 0.70, 1.05, 1.28, 0.90,
    1.40, 0.65, 1.12, 0.78, 0.95, 1.22, 0.60, 1.08,
]
assert len(TRUE_GAINS) == NUM_BEAMS

# Range-falloff model f(r) = (kRangePlateauM / max(r, kRangePlateauM))^2 -
# MUST MATCH ../src/kernels.cuh's kRangePlateauM (file header "THE FORWARD
# MODEL"). f(kRangePlateauM) = 1.0 by construction (a convenient reference
# scale, not a physical unit).
RANGE_PLATEAU_M = 4.0


def range_falloff(r_m: float) -> float:
    """f(r) - the range-falloff model (file header). Flat for r <=
    RANGE_PLATEAU_M (the near-range defocus plateau), 1/r^2 beyond it."""
    r_eff = max(r_m, RANGE_PLATEAU_M)
    return (RANGE_PLATEAU_M / r_eff) ** 2


def ray_plane_x(origin, dir_, x_plane, y_range, z_range):
    """Ray/vertical-plane (x = x_plane) intersection, masked to a rectangular
    [y_range] x [z_range] extent. Returns t >= 0 or None."""
    if abs(dir_[0]) < 1e-12:
        return None
    t = (x_plane - origin[0]) / dir_[0]
    if t < 0.0:
        return None
    y = origin[1] + dir_[1] * t
    z = origin[2] + dir_[2] * t
    if not (y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]):
        return None
    return t


def ray_plane_z(origin, dir_, z_plane):
    """Ray/horizontal-plane (z = z_plane) intersection, unbounded in x,y.
    Returns t >= 0 or None (ray parallel to, or moving away from, the plane)."""
    if abs(dir_[2]) < 1e-12:
        return None
    t = (z_plane - origin[2]) / dir_[2]
    return t if t >= 0.0 else None


def beam_direction(elev_deg: float, az_deg: float):
    """Unit direction, spherical convention (matches 02.13/02.18's
    generator): az measured CCW from +x (forward) in the xy-plane, elev up
    from the xy-plane. x-forward/y-left/z-up (CLAUDE.md paragraph 12)."""
    el = math.radians(elev_deg)
    az = math.radians(az_deg)
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def cast_ray(origin, dir_):
    """Nearest surface hit among all five (file header scene list), or None.
    Returns (t, surf_id, normal). Occlusion falls out for free from the
    minimum-t rule, the same choice 02.13/02.18's cast_real makes."""
    candidates = []

    tg = ray_plane_z(origin, dir_, GROUND_Z_M)
    if tg is not None and tg <= MAX_RANGE_M:
        candidates.append((tg, SURF_GROUND, (0.0, 0.0, 1.0)))

    tn = ray_plane_x(origin, dir_, WALL_NEAR_X_M, WALL_NEAR_Y_RANGE, WALL_NEAR_Z_RANGE)
    if tn is not None and tn <= MAX_RANGE_M:
        candidates.append((tn, SURF_WALL_NEAR, (-1.0, 0.0, 0.0)))

    tp = ray_plane_x(origin, dir_, PANEL_X_M, PANEL_Y_RANGE, PANEL_Z_RANGE)
    if tp is not None and tp <= MAX_RANGE_M:
        candidates.append((tp, SURF_PANEL, (-1.0, 0.0, 0.0)))

    tf = ray_plane_x(origin, dir_, WALL_FAR_X_M, WALL_FAR_Y_RANGE, WALL_FAR_Z_RANGE)
    if tf is not None and tf <= MAX_RANGE_M:
        candidates.append((tf, SURF_WALL_FAR, (-1.0, 0.0, 0.0)))

    ti = ray_plane_z(origin, dir_, ISOLATED_TARGET_Z_M)
    if ti is not None and ti <= MAX_RANGE_M:
        candidates.append((ti, SURF_ISOLATED, (0.0, 0.0, -1.0)))

    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])


def generate_scan(rng: Xorshift32, elev_table, tallies: dict):
    """Ray-cast every (channel, azimuth) beam of ONE scan variant. Returns a
    list of rows (channel, x, y, z, intensity, surf_id, R_true)."""
    origin = (0.0, 0.0, 0.0)   # sensor at the origin (file header scope cut)
    rows = []

    for ch in range(NUM_BEAMS):
        elev_deg = elev_table[ch]
        for az_i in range(AZIMUTH_STEPS):
            az_deg = AZIMUTH_MIN_DEG + az_i * AZIMUTH_STEP_DEG
            dir_ = beam_direction(elev_deg, az_deg)

            hit = cast_ray(origin, dir_)
            if hit is None:
                tallies["miss"] += 1
                continue

            t, surf_id, normal = hit
            r_noisy = max(0.05, t + rng.gaussian(RANGE_NOISE_SIGMA_M))
            x = dir_[0] * r_noisy
            y = dir_[1] * r_noisy
            z = dir_[2] * r_noisy

            cos_theta = abs(dir_[0] * normal[0] + dir_[1] * normal[1] + dir_[2] * normal[2])
            cos_theta = max(cos_theta, 0.02)   # grazing floor: a real receiver never reads exactly 0

            R_true = R_OF[surf_id]
            clean = TRUE_GAINS[ch] * R_true * range_falloff(r_noisy) * cos_theta
            noisy = clean * (1.0 + rng.gaussian(NOISE_SIGMA_MULT)) + rng.gaussian(NOISE_SIGMA_ADD)
            intensity = max(0.0, noisy)

            rows.append((ch, x, y, z, intensity, surf_id, R_true))
            tallies["hit"] += 1
            tallies["surf_" + SURF_NAMES[surf_id]] += 1
            tallies["ch_" + str(ch)] += 1

    return rows


def write_scan_csv(path: Path, rows, seed: int, elev_table, variant_note: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# SYNTHETIC data - generated by scripts/make_synthetic.py for project 02.20\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write(f"# variant: {variant_note}\n")
        f.write("# scene: GROUND/WALL_NEAR/PANEL/WALL_FAR/ISOLATED_TARGET - see this script's module\n")
        f.write("#        docstring for the full forward-model derivation\n")
        f.write(f"# num_beams={NUM_BEAMS}\n")
        f.write(f"# elev_min_deg={ELEV_MIN_DEG}\n")
        f.write(f"# elev_max_deg={ELEV_MAX_DEG}\n")
        f.write(f"# azimuth_min_deg={AZIMUTH_MIN_DEG}\n")
        f.write(f"# azimuth_step_deg={AZIMUTH_STEP_DEG}\n")
        f.write(f"# azimuth_steps={AZIMUTH_STEPS}\n")
        f.write(f"# max_range_m={MAX_RANGE_M}\n")
        f.write(f"# range_plateau_m={RANGE_PLATEAU_M}\n")
        f.write(f"# seed={seed}\n")
        f.write("# surf_id: 0=GROUND 1=WALL_NEAR 2=PANEL 3=WALL_FAR 4=ISOLATED_TARGET\n")
        f.write("# elev_deg_by_channel=" + ",".join(f"{e:.6f}" for e in elev_table) + "\n")
        f.write("# columns: channel,x,y,z,intensity,surf_id,R_true\n")
        for row in rows:
            ch, x, y, z, inten, surf_id, R_true = row
            f.write(f"{ch},{x:.6f},{y:.6f},{z:.6f},{inten:.6f},{surf_id},{R_true:.6f}\n")


def write_gains_csv(path: Path, seed: int) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# SYNTHETIC ground truth - generated by scripts/make_synthetic.py for project 02.20\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write("# GROUND TRUTH ONLY: gates/reporting use this file; the calibration algorithm never reads it.\n")
        f.write("# columns: channel,true_gain\n")
        for ch, g in enumerate(TRUE_GAINS):
            f.write(f"{ch},{g:.6f}\n")


def generate(out_dir: Path, seed: int) -> None:
    rng = Xorshift32(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- primary scan: all 16 channels at their normal elevations ---------
    tallies_primary = {"hit": 0, "miss": 0}
    for s in SURF_NAMES.values():
        tallies_primary["surf_" + s] = 0
    for ch in range(NUM_BEAMS):
        tallies_primary["ch_" + str(ch)] = 0
    rows_primary = generate_scan(rng, ELEV_DEG, tallies_primary)
    write_scan_csv(out_dir / "scan_primary.csv", rows_primary, seed, ELEV_DEG, "primary (all channels nominal)")

    # ---- degenerate scan: channel kDegenerateChannel retargeted upward ----
    elev_degenerate = list(ELEV_DEG)
    elev_degenerate[DEGENERATE_CHANNEL] = DEGENERATE_ELEV_DEG
    tallies_degenerate = {"hit": 0, "miss": 0}
    for s in SURF_NAMES.values():
        tallies_degenerate["surf_" + s] = 0
    for ch in range(NUM_BEAMS):
        tallies_degenerate["ch_" + str(ch)] = 0
    rows_degenerate = generate_scan(rng, elev_degenerate, tallies_degenerate)
    write_scan_csv(out_dir / "scan_degenerate.csv", rows_degenerate, seed, elev_degenerate,
                    f"degenerate (channel {DEGENERATE_CHANNEL} retargeted to {DEGENERATE_ELEV_DEG} deg elevation, "
                    "hits ONLY the isolated target - unobservable-channel gate)")

    write_gains_csv(out_dir / "gains_true.csv", seed)

    print(f"[make_synthetic] wrote {len(rows_primary)} points to scan_primary.csv")
    print(f"  surfaces: " + ", ".join(f"{s}={tallies_primary['surf_' + s]}" for s in SURF_NAMES.values()))
    print(f"  per-channel counts: " + ", ".join(str(tallies_primary["ch_" + str(c)]) for c in range(NUM_BEAMS)))
    print(f"[make_synthetic] wrote {len(rows_degenerate)} points to scan_degenerate.csv")
    print(f"  surfaces: " + ", ".join(f"{s}={tallies_degenerate['surf_' + s]}" for s in SURF_NAMES.values()))
    print(f"  per-channel counts: " + ", ".join(str(tallies_degenerate["ch_" + str(c)]) for c in range(NUM_BEAMS)))
    print(f"[make_synthetic] wrote {NUM_BEAMS} true gains to gains_true.csv (illustrative, span "
          f"{min(TRUE_GAINS):.2f}-{max(TRUE_GAINS):.2f})")


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
