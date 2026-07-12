#!/usr/bin/env python3
"""make_synthetic.py -- synthetic sample-data generator for 02.12
(Range-image conversion + depth-clustering segmentation).

Why this script exists (CLAUDE.md section 8: synthetic-first)
---------------------------------------------------------------
This project needs a 16-beam LiDAR scan (raw, driver-shaped: a flat list of
valid returns, each already carrying its native ring/azimuth-bin, exactly
what a real spinning-LiDAR driver reports per packet -- see
../PRACTICE.md section 1) that is DESIGNED to make several specific,
measurable lessons true: a range-image conversion round trip, a flat ground
plane a column-wise angle walk can remove cleanly, and four object cohorts
that individually exercise the depth-clustering (Bogoslavskyi-Stachniss
beta-criterion) algorithm's strengths and its one well-known weakness. All
of this is synthesized analytically with EXACT ground truth (no download,
license-clean, reproducible bit-for-bit from a fixed seed).

Stdlib only, xorshift32 PRNG (seed 42) -- no numpy, no random module, so the
byte stream is identical on every machine/Python build (CLAUDE.md section 12
determinism rule; the C++ side of this repo carries the identical
"xorshift32, not std::uniform_real_distribution" rule for the same reason).

The scene, in the LiDAR SENSOR's own frame
--------------------------------------------
Origin at the sensor; +x forward, +y left, +z up (CLAUDE.md's body
convention). The sensor is mounted SENSOR_HEIGHT_M above a flat ground plane
at z = -SENSOR_HEIGHT_M (ground removal's column-walk in kernels.cuh takes
this exact height as its virtual first reference point). On top of that
flat ground sit six axis-aligned-box objects, one per "cohort" the project's
gates exercise -- see the per-object comments below for exactly which
lesson each one teaches. Every object is placed in its OWN azimuth sector
(see the sector map below) so the four cohorts do not interact with each
other; only the two members of a designed PAIR are meant to interact.

  Sector map (approximate azimuth, degrees, atan2(y,x) convention):
    A  ~  -25 .. +25   PERSON + WALL_BEHIND       (the depth-gap showcase pair)
    B  ~   78 .. 101   BIG_BOX + FAR_POLE          (large-near / small-far pair)
    C  ~  178 .. 182   THIN_POLE                   (isolated, min-size trade)
    D  ~  290 .. 344   GRAZING_WALL                (the beta criterion's known weakness)

The beam model (cited from 02.01/02.02, reused verbatim: 16 beams, -15..+15
degrees in 2-degree steps) and the ray/scene intersection below are a
minimal, honest ray caster: for every (ring, azimuth-bin) cell, cast one ray
from the sensor origin, intersect it against the ground plane and all six
object boxes (a standard ray/AABB slab test), and keep the NEAREST hit
within MAX_RANGE_M (open-sky/too-far beams get no return, exactly as a real
LiDAR reports nothing when no surface is in range). A small, fixed-sigma
Gaussian range noise (RANGE_NOISE_SIGMA_M) is then added along the same ray
direction -- range-axis noise, matching how a real time-of-flight sensor's
error actually manifests (see ../THEORY.md "Numerical considerations" for
why this noise level was chosen small enough not to perturb the ground-
removal angle test past its threshold, with the arithmetic shown).

Why NOT include a bounding room (contrast 02.01/02.02's walled room)
----------------------------------------------------------------------
Those projects use walls to teach dropout/occlusion at a room boundary.
Here MAX_RANGE_M alone is the honest cutoff -- any ray that does not hit the
ground or an object within range legitimately returns nothing (open sky),
which is exactly the physical situation a range-image pipeline must handle
column-wise (a column can have missing cells at the top). Adding walls
would only clutter the four designed sectors without adding a new lesson.

The two synthetic COLLISION points (exercising scatter_encode_kernel's
nearest-wins atomicMin race, kernels.cuh's stage 1a)
---------------------------------------------------------------------------
Every genuine ray-cast point already owns a unique (ring, az_bin) cell by
construction (one ray per cell), so the organized-grid scatter never
naturally collides in this scene. To exercise (and VERIFY, in main.cu) the
collision-resolution machinery honestly rather than leave it untested, this
script appends exactly two synthetic "phantom" points that deliberately
TARGET an already-used cell with a LARGER (farther) range than the real
return there -- a stand-in for a multipath/ghost return a real receiver
occasionally reports. Their truth_id is the sentinel PHANTOM_TRUTH_ID (-2,
distinct from -1 = "empty cell") so a correct nearest-wins scatter must
never let a phantom become the organized cell's content -- main.cu's
VERIFY(range_image) checks exactly that.

Regenerate with: python scripts/make_synthetic.py
(no arguments; the seed and every scene constant are fixed in this file).
"""

import math
import struct
from pathlib import Path

# ===========================================================================
# xorshift32 PRNG -- stdlib-only, deterministic across platforms (CLAUDE.md
# section 12: fixed seeds, no std::uniform_real_distribution/np.random
# whose bit-for-bit output is implementation-defined across versions).
# ===========================================================================
class Xorshift32:
    """A minimal 32-bit xorshift generator (Marsaglia 2003). Not
    cryptographic, not even great statistically -- but PERFECTLY adequate
    for scattering synthetic sensor noise, and its entire state is one
    32-bit integer, so "seed 42" reproduces the exact same byte stream on
    any machine, forever (the property this project's determinism rule
    actually needs)."""

    def __init__(self, seed: int):
        # xorshift's recurrence has a fixed point at 0 (0 maps to 0
        # forever), so a zero seed would be degenerate; guard it here
        # rather than relying on the caller to know.
        self.state = seed & 0xFFFFFFFF or 0x9E3779B9

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        self.state = x
        return x

    def next_float01(self) -> float:
        """Uniform in [0,1) -- 24 bits of the state scaled into a float,
        mirroring the standard "top bits -> mantissa-width fraction" trick
        used to turn an integer PRNG into a float generator without bias
        toward the low bits (which xorshift's low bit especially can be
        weak in)."""
        return (self.next_u32() >> 8) / float(1 << 24)

    def next_gauss(self) -> float:
        """One standard-normal sample via the Box-Muller transform (uses
        two uniforms per call; the classically simplest way to turn a
        uniform generator into a Gaussian one without a stdlib
        distribution helper -- CLAUDE.md's xorshift32 rule applies to this
        script too, so no `random.gauss`)."""
        u1 = max(self.next_float01(), 1.0e-12)   # avoid log(0)
        u2 = self.next_float01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


RNG = Xorshift32(42)

# ===========================================================================
# Organized-grid shape -- MUST match src/kernels.cuh's kNumBeams/kAzimuthBins
# exactly (documented match, main.cu asserts it against this file's header
# at load time -- 02.01's discipline).
# ===========================================================================
NUM_BEAMS = 16
AZIMUTH_BINS = 1024
BEAM_ELEV_DEG = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
BEAM_ELEV_RAD = [math.radians(d) for d in BEAM_ELEV_DEG]

SENSOR_HEIGHT_M = 1.5           # matches kernels.cuh kSensorHeightM
GROUND_Z = -SENSOR_HEIGHT_M
MAX_RANGE_M = 18.0              # matches kernels.cuh kMaxRangeM
RANGE_NOISE_SIGMA_M = 0.003     # 3 mm -- see module docstring for why this small

GROUND_ANGLE_THRESHOLD_DEG = 10.0   # matches kernels.cuh kGroundAngleThresholdDeg
BETA_THRESHOLD_DEG = 10.0           # matches kernels.cuh kBetaThresholdDeg
EUCLID_TOLERANCE_M = 0.40           # matches kernels.cuh kEuclideanClusterToleranceM
MIN_CLUSTER_SIZE_DEPTH = 5          # matches kernels.cuh kMinDepthClusterSize
MIN_CLUSTER_SIZE_EUCLID = 5         # matches kernels.cuh kMinEuclideanClusterSize

PHANTOM_TRUTH_ID = -2   # sentinel: "should never win a scatter race" (see module docstring)


def polar_to_xy(r_m: float, az_deg: float):
    az = math.radians(az_deg)
    return r_m * math.cos(az), r_m * math.sin(az)


# ---------------------------------------------------------------------------
# The six named objects. Each is an axis-aligned box (bmin, bmax) plus a
# truth id (1..6) and a short name used in this script's own diagnostic
# printout and in ../data/README.md. Positions were chosen with
# polar_to_xy() at a TARGET (range, azimuth) per the sector map above, then
# widened into a box -- see each comment for the specific lesson.
# ---------------------------------------------------------------------------

def _box(cx, cy, half_x, half_y, zmin, zmax):
    return (cx - half_x, cx + half_x, cy - half_y, cy + half_y, zmin, zmax)


_person_cx, _person_cy = polar_to_xy(2.0, 0.0)     # front face 1.95, back face 2.05
_wall_cx, _wall_cy = polar_to_xy(2.20, 0.0)         # front face 2.15 -> 0.10 m face gap from the person
_bigbox_cx, _bigbox_cy = polar_to_xy(4.0, 90.0)
_farpole_cx, _farpole_cy = polar_to_xy(9.0, 81.3)
_thinpole_cx, _thinpole_cy = polar_to_xy(4.0, 180.0)

OBJECTS = [
    # id=1 PERSON: a narrow standing box directly in front of WALL_BEHIND
    # (id=2) -- the DEPTH-GAP SHOWCASE pair.
    #
    # THE OCCLUSION-SHADOW CORRECTION (why the gap below is 0.10 m, not the
    # naively-expected face-to-face distance): a wall point directly behind
    # the person is OCCLUDED by the person itself, so the nearest actually
    # VISIBLE wall point is not the one straight back -- it sits just
    # outside the person's angular silhouette, which widens with range (the
    # person's near-face corner at (x0, half_y) casts a shadow onto the
    # wall's near face at x_wall of half-width half_y * x_wall / x0, a
    # PERSPECTIVE effect from the shared origin). The visible gap is
    # therefore always LARGER than the raw face-to-face distance -- an
    # earlier version of this scene used a 0.30 m face gap and a 0.20 m
    # half-width and measured an ACTUAL nearest visible-point distance of
    # 0.50 m (verified by brute-force nearest-neighbor search over the
    # generated points) -- comfortably ABOVE EUCLID_TOLERANCE_M, silently
    # defeating the showcase. The numbers below were re-derived from that
    # same geometry (half_y=0.10 m, a 0.10 m face gap, person front face at
    # x0=1.95 m) so the MEASURED nearest visible-point distance clears
    # comfortably under 0.40 m -- see THEORY.md "The math" for the full
    # shadow derivation and ../data/README.md for the as-generated number.
    dict(id=1, name="person",
        aabb=_box(_person_cx, _person_cy, 0.05, 0.10, -SENSOR_HEIGHT_M, 0.2)),
    # id=2 WALL_BEHIND: a wide, flat panel directly behind the person,
    # spanning well past the person's silhouette on both sides so the
    # person/wall image-adjacency (and the resulting depth gap) shows up on
    # BOTH flanks.
    #
    # WHY THE WALL'S TOP MUST STAY AT OR BELOW THE PERSON'S TOP (a second,
    # RING-direction occlusion-shadow correction, the vertical twin of the
    # azimuth one documented on PERSON above): if the wall were taller than
    # the person, rings steep enough to clear the person's top would still
    # hit the wall directly above it, at the SAME azimuth column -- a
    # RING-adjacent pair, whose angular step (2 deg) is ~6x the azimuth
    # step (0.35 deg). For the SAME physical range gap, a 6x larger alpha
    # produces a MUCH larger beta (beta grows with alpha for a fixed range
    # ratio -- THEORY.md "The math"), comfortably exceeding
    # BETA_THRESHOLD_DEG even though the azimuth-direction boundary
    # correctly cuts -- an earlier version of this scene used wall top
    # z=0.5 m (vs. person top z=0.2 m) and MEASURED exactly this failure
    # (ring-adjacent beta ~18 deg, well above the 10 deg threshold, wrongly
    # merging person and wall). Because the wall sits FARTHER away, giving
    # it the SAME top z as the person already makes its angular top
    # LOWER than the person's (a farther object subtends a smaller angle
    # for the same height) -- any ring that clears the person's top also
    # clears the wall's, so it sees past both into open sky instead of
    # "jumping" onto the wall. 0.10 m margin below the person's top height
    # is added for a comfortable, MEASURED-clear margin (data/README.md).
    dict(id=2, name="wall_behind",
        aabb=_box(_wall_cx, _wall_cy, 0.05, 1.00, -SENSOR_HEIGHT_M, 0.1)),
    # id=3 BIG_BOX: a large near obstacle (sector B) -- half of the
    # LARGE-NEAR / SMALL-FAR pair with FAR_POLE (id=4).
    dict(id=3, name="big_box",
        aabb=_box(_bigbox_cx, _bigbox_cy, 0.5, 0.5, -SENSOR_HEIGHT_M, 0.1)),
    # id=4 FAR_POLE: a small, distant box tucked just outside BIG_BOX's
    # angular footprint (see the module docstring's azimuth arithmetic) --
    # peeking out beside the near object, the way a background object peeks
    # out from behind a foreground one at an occlusion boundary.
    dict(id=4, name="far_pole",
        aabb=_box(_farpole_cx, _farpole_cy, 0.10, 0.10, -SENSOR_HEIGHT_M, -0.5)),
    # id=5 THIN_POLE: an isolated 0.10 x 0.10 m post (sector C) -- exercises
    # the min-cluster-size vs "real but thin object" trade honestly (README
    # / THEORY.md "Numerical considerations" report whichever way it goes).
    dict(id=5, name="thin_pole",
        aabb=_box(_thinpole_cx, _thinpole_cy, 0.05, 0.05, -SENSOR_HEIGHT_M, -0.3)),
    # id=6 GRAZING_WALL: a long flat panel running mostly ALONG the sensor's
    # line of sight rather than facing it (sector D, placed in the wide-open
    # 185-235 deg range so its az footprint never brushes WALL_BEHIND's own
    # +-25 deg span -- see the az-overlap note below) -- viewed at a shallow
    # (grazing) incidence angle, its own range changes rapidly from column
    # to column purely from geometry: dr/dazimuth = -y*cos(az)/sin^2(az) for
    # a wall at fixed lateral offset y (THEORY.md "The math" derives this in
    # full), which DIVERGES as az -> 0 (viewed edge-on). That is exactly the
    # beta criterion's known failure mode: a single continuous surface
    # FRAGMENTS into several depth clusters near its shallow (far, grazing)
    # end even though it is one object -- the near (more face-on) end stays
    # one connected piece. This is the honest, designed demonstration of the
    # algorithm's documented weakness (measured fragment count in
    # data/README.md), not a bug in this generator. 13 m long (x from -14 to
    # -1) so the far end reaches a genuinely shallow ~4 deg incidence.
    dict(id=6, name="grazing_wall",
        aabb=(-14.0, -1.0, -1.5, -1.3, -SENSOR_HEIGHT_M, 0.7)),
]
OBJECT_BY_ID = {o["id"]: o for o in OBJECTS}
TRUTH_NUM_OBJECTS = len(OBJECTS)


def ray_aabb_hit(origin, d, aabb):
    """Standard ray/axis-aligned-box slab test. origin/d are 3-tuples (d
    need not be normalized here -- callers always pass a unit direction, so
    the returned t IS the range in meters). Returns the entry distance t
    (t>0, i.e. the box is in FRONT of the sensor) or None if the ray misses
    the box, or the box is entirely behind the origin."""
    tmin, tmax = 0.0, MAX_RANGE_M
    bmin = aabb[0], aabb[2], aabb[4]
    bmax = aabb[1], aabb[3], aabb[5]
    for axis in range(3):
        o, dc = origin[axis], d[axis]
        lo, hi = bmin[axis], bmax[axis]
        if abs(dc) < 1.0e-12:
            if o < lo or o > hi:
                return None   # ray parallel to this slab and outside it: no hit on any t
        else:
            t1, t2 = (lo - o) / dc, (hi - o) / dc
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmin > 1.0e-6 else None


def ray_ground_hit(origin, d):
    """Ground is the infinite plane z = GROUND_Z. Only beams looking
    downward (d[2] < 0) can ever reach it going FORWARD along the ray."""
    if d[2] >= -1.0e-9:
        return None
    t = (GROUND_Z - origin[2]) / d[2]
    return t if t > 1.0e-6 else None


def cast_scene(ring: int, az_bin: int):
    """Cast one ray for cell (ring, az_bin); return (t_true, truth_id, d)
    for the NEAREST surface within MAX_RANGE_M, or None for no return."""
    el = BEAM_ELEV_RAD[ring]
    az = az_bin * (2.0 * math.pi / AZIMUTH_BINS)
    d = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    origin = (0.0, 0.0, 0.0)

    best_t, best_id = None, None
    tg = ray_ground_hit(origin, d)
    if tg is not None and tg <= MAX_RANGE_M:
        best_t, best_id = tg, 0
    for obj in OBJECTS:
        t = ray_aabb_hit(origin, d, obj["aabb"])
        if t is not None and t <= MAX_RANGE_M and (best_t is None or t < best_t):
            best_t, best_id = t, obj["id"]
    if best_t is None:
        return None
    return best_t, best_id, d


def build_scan():
    """Cast every (ring, az_bin) ray, apply range noise, and return the
    flat list of valid points as dicts. This IS the raw, driver-shaped scan
    -- see the module docstring for why no organizing happens here (that is
    main.cu's GPU job, exercised against this exact list)."""
    points = []
    per_object_cells = {o["id"]: [] for o in OBJECTS}
    ground_cells = []

    for ring in range(NUM_BEAMS):
        for az_bin in range(AZIMUTH_BINS):
            hit = cast_scene(ring, az_bin)
            if hit is None:
                continue
            t_true, truth_id, d = hit
            noise = RNG.next_gauss() * RANGE_NOISE_SIGMA_M
            t_noisy = max(t_true + noise, 0.05)
            x, y, z = d[0] * t_noisy, d[1] * t_noisy, d[2] * t_noisy
            points.append(dict(x=x, y=y, z=z, range_m=t_noisy,
                                ring=ring, az_bin=az_bin, truth_id=truth_id))
            if truth_id == 0:
                ground_cells.append((ring, az_bin))
            else:
                per_object_cells[truth_id].append((ring, az_bin, t_noisy))

    return points, ground_cells, per_object_cells


def add_collision_phantoms(points, per_object_cells):
    """Append the two synthetic collision-test points described in the
    module docstring: pick one real PERSON cell and one real BIG_BOX cell,
    and add a farther "phantom" return targeting the SAME (ring, az_bin). A
    correct nearest-wins scatter must keep the real (nearer) point in that
    cell."""
    anchors = []
    if per_object_cells[1]:
        anchors.append(per_object_cells[1][len(per_object_cells[1]) // 2])
    if per_object_cells[3]:
        anchors.append(per_object_cells[3][len(per_object_cells[3]) // 2])

    added = 0
    for ring, az_bin, real_range in anchors:
        az = az_bin * (2.0 * math.pi / AZIMUTH_BINS)
        el = BEAM_ELEV_RAD[ring]
        d = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
        phantom_range = min(real_range + 5.0, MAX_RANGE_M - 0.1)   # clearly FARTHER -> must lose the race
        px, py, pz = d[0] * phantom_range, d[1] * phantom_range, d[2] * phantom_range
        points.append(dict(x=px, y=py, z=pz, range_m=phantom_range,
                            ring=ring, az_bin=az_bin, truth_id=PHANTOM_TRUTH_ID))
        added += 1
    return added


def write_binary_sample(out_path: Path, points):
    """RIMAGE01 binary format -- see ../data/README.md for the authoritative
    field-by-field description (this docstring and that file must be kept
    in lockstep, 02.04's convention). Explicit struct.pack calls, never a
    raw struct dump, so the format is portable across compilers (the same
    reasoning util/paths.h gives for avoiding <filesystem>)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"RIMAGE01")
        f.write(struct.pack("<i", len(points)))
        f.write(struct.pack("<i", NUM_BEAMS))
        f.write(struct.pack("<i", AZIMUTH_BINS))
        f.write(struct.pack("<f", SENSOR_HEIGHT_M))
        f.write(struct.pack("<f", GROUND_ANGLE_THRESHOLD_DEG))
        f.write(struct.pack("<f", BETA_THRESHOLD_DEG))
        f.write(struct.pack("<f", EUCLID_TOLERANCE_M))
        f.write(struct.pack("<i", MIN_CLUSTER_SIZE_DEPTH))
        f.write(struct.pack("<i", MIN_CLUSTER_SIZE_EUCLID))
        f.write(struct.pack("<i", TRUTH_NUM_OBJECTS))
        f.write(struct.pack("<iii", 0, 0, 0))   # reserved, for future header growth
        for p in points:
            f.write(struct.pack("<ffff iii",
                                p["x"], p["y"], p["z"], p["range_m"],
                                p["ring"], p["az_bin"], p["truth_id"]))


def main():
    points, ground_cells, per_object_cells = build_scan()
    n_before = len(points)
    n_phantoms = add_collision_phantoms(points, per_object_cells)

    out_path = Path(__file__).resolve().parent.parent / "data" / "sample" / "range_image_scene.bin"
    write_binary_sample(out_path, points)

    # ---- Diagnostic printout (this script's own stdout -- NOT part of the
    # C++ demo's checked output contract; purely so a human regenerating
    # the sample can sanity-check the designed geometry before committing
    # it) -----------------------------------------------------------------
    print(f"[make_synthetic] wrote {out_path}")
    print(f"[make_synthetic] {n_before} ray-cast returns + {n_phantoms} synthetic collision phantoms "
          f"= {len(points)} total points")
    print(f"[make_synthetic] ground cells: {len(ground_cells)}")
    for obj in OBJECTS:
        cells = per_object_cells[obj["id"]]
        if not cells:
            print(f"[make_synthetic]   {obj['name']:12s} (id={obj['id']}): 0 cells -- WARNING: object never hit")
            continue
        az_list = sorted(c[1] for c in cells)
        ranges = [c[2] for c in cells]
        az_deg_lo = az_list[0] * 360.0 / AZIMUTH_BINS
        az_deg_hi = az_list[-1] * 360.0 / AZIMUTH_BINS
        print(f"[make_synthetic]   {obj['name']:12s} (id={obj['id']}): {len(cells):4d} cells, "
              f"az_bin [{az_list[0]:4d},{az_list[-1]:4d}] (~[{az_deg_lo:6.2f},{az_deg_hi:6.2f}] deg), "
              f"range [{min(ranges):5.2f},{max(ranges):5.2f}] m")


if __name__ == "__main__":
    main()
