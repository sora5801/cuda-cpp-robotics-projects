#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.09
(Normal + curvature estimation at millions of points/sec).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Normal/curvature estimation can only be graded honestly against a TRUTH
ENGINE — a surface whose normal and curvature are known in CLOSED FORM at
every point, not eyeballed from a real scan. This script builds exactly
that: four analytic surfaces (a tilted PLANE, a SPHERE, a CYLINDER, and a
sharp two-plane EDGE — the degeneracy cohort) at three noise levels each
(none / low / high), sampled from a single simulated sensor at the origin,
+x forward (the repo's LiDAR-frame convention, CLAUDE.md paragraph 12).
Twelve cohorts, well separated in space so no point's true K-nearest
neighborhood ever straddles two cohorts by accident.

Placement (shared by every cohort type)
----------------------------------------
Each SURFACE TYPE gets one fixed viewing direction `dir` (unit vector from
the sensor), built with the same elevation/azimuth spherical formula
project 01.18 derives for its 16-beam model (cited verbatim, reused here
for camera-independent reasons: it is just "a unit direction from two
angles", the natural building block for any sensor-centric placement) and
project 02.01/02.05 already reuse in this same domain:

    dir(az, el) = (cos(el)*cos(az), cos(el)*sin(az), sin(el))

The three NOISE-LEVEL cohorts of one surface type sit at increasing range
R along that SAME direction (well separated: R steps of 5 m against patch
extents of ~3 m) — a deliberate re-use of the domain's own "sensor sees
several returns along similar bearings at different ranges" character,
not a coincidence.

At each cohort's center C = R*dir, `view_dir = normalize(-C)` is the unit
direction FROM the patch back TOWARD the sensor — the reference every
surface's outward-facing geometry (and its ground-truth ORIENTATION) is
built from. An orthonormal in-patch basis (u, v) perpendicular to
view_dir is built once (`make_perp_basis`) and reused by every surface's
own local geometry.

The four surfaces, and what each teaches (see kernels.cuh / THEORY.md for
how main.cu gates each one):

  * PLANE  — `view_dir` tilted by a fixed (12 deg, 8 deg) two-axis rotation
    so the plane is NOT axis-aligned (a plane normal to +x would silently
    hide a bug that only shows up for a general 3-vector). Exact normal =
    the tilt result (constant over the whole cohort); exact curvature = 0.
  * SPHERE — points sampled on the sensor-facing CAP (half-angle 80 deg
    from `view_dir`) via inverse-CDF sampling of cos(beta) so the areal
    density is UNIFORM across the cap (no pole bunching). Exact normal =
    the outward radial direction; exact curvature = 1/radius (both
    principal curvatures equal on a sphere).
  * CYLINDER — axis = u (perpendicular to `view_dir`); cross-section angle
    beta swept over the visible +-80 deg arc. Exact normal = the radial
    (perpendicular-to-axis) direction; exact curvature = 1/radius (ONE
    principal curvature; the other, along the axis, is exactly 0 — the
    geometric reason this project's curvature_ordering gate expects
    sphere > cylinder at matched radius, see THEORY.md).
  * EDGE — two half-planes meeting at a ridge line (axis u) through C,
    with a `dihedral_deg` interior angle (a CONVEX ridge facing the
    sensor, like the corner of a box). Every point's exact normal is its
    OWN face's normal (nA or nB) — there is no single "the" normal at the
    ridge itself, which is exactly why this cohort has NO valid true
    curvature (written as the sentinel -1.0, see data/README.md) and is
    this project's dedicated DEGENERACY-DETECTION cohort.

Noise model (documented, not hidden): Gaussian, ALONG THE TRUE NORMAL
only, sigma = 0 / 3 mm / 15 mm for the none/low/high cohorts. 15 mm
matches project 02.01/02.05's own cited LiDAR range-noise magnitude
(RANGE_NOISE_M) — a deliberate consistency, not a coincidence. Along-
normal noise (rather than isotropic 3-D jitter) is the standard synthetic
model for GRADING normal estimators (it is exactly the error component a
covariance-based fit is sensitive to; lateral noise barely moves the
smallest eigenvector) and is physically motivated too: a real LiDAR's
per-shot error is dominated by RANGE (line-of-sight) noise, and for a
surface seen close to head-on the line of sight IS close to the normal.

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
# fixed-seed convention, identical algorithm to 02.01/02.05/08.01's device-
# side generators and their own make_synthetic.py scripts, cited not
# reinvented).
# ===========================================================================
class Xorshift32:
    """32-bit xorshift PRNG (Marsaglia 2003). See 02.01/02.05's
    make_synthetic.py for the full rationale (matching this repo's CUDA
    device-side generator family instead of Python's Mersenne-Twister)."""

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

    def gauss(self) -> float:
        """Standard normal via Box-Muller (stdlib math only, no numpy —
        CLAUDE.md's Python-stdlib-only rule). Two independent uniforms in
        (0,1) map to one standard-normal sample; the second Box-Muller
        output is simply discarded (a documented, deliberate 2x waste in
        exchange for the simplest possible correct implementation — this
        script calls it only a few thousand times, so the cost is
        negligible)."""
        u1 = max(self.uniform01(), 1.0e-12)   # guard log(0)
        u2 = self.uniform01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


DEFAULT_SEED = 42  # repo convention; no meaning beyond tradition

# ===========================================================================
# Plain 3-vector helpers (tuples of 3 floats; no numpy — stdlib only).
# ===========================================================================
def v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_scale(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def v_norm(a): return math.sqrt(v_dot(a, a))
def v_normalize(a):
    n = v_norm(a)
    if n < 1.0e-12:
        return (0.0, 0.0, 1.0)   # defensive: never called on a near-zero vector by this script's geometry
    return (a[0] / n, a[1] / n, a[2] / n)


def direction(az_deg: float, el_deg: float):
    """dir(az, el) = (cos(el)cos(az), cos(el)sin(az), sin(el)) — the 16-beam
    spinning-LiDAR direction formula project 01.18 derives (cited, reused
    verbatim by 02.01/02.05's own generators and here again: it is simply
    'a unit vector from two spherical angles', the natural sensor-centric
    building block, not a beam-table lookup in this script)."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def rotate_about_axis(v, k, angle_rad):
    """Rodrigues' rotation formula: rotate vector v by angle_rad (radians)
    about the UNIT axis k (right-hand rule). v_rot = v*cos(a) +
    (k x v)*sin(a) + k*(k.v)*(1-cos(a)) — the standard closed-form rotation
    this script uses to build every tilted/rotated surface direction below
    (the plane's tilt, the edge's two face normals) without a full rotation
    matrix."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    kxv = v_cross(k, v)
    kdv = v_dot(k, v)
    term1 = v_scale(v, c)
    term2 = v_scale(kxv, s)
    term3 = v_scale(k, kdv * (1.0 - c))
    return v_add(v_add(term1, term2), term3)


def make_perp_basis(n):
    """Return an orthonormal (u, v) pair both perpendicular to unit vector
    n — the standard 'pick a non-parallel helper axis, cross twice' trick.
    Picking +x or +y as the helper (whichever is LESS aligned with n) keeps
    the cross product well-conditioned for every n this script ever calls
    it with (no near-degenerate cross products)."""
    helper = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = v_normalize(v_cross(helper, n))
    v = v_cross(n, u)   # already unit length: n and u are orthonormal
    return u, v


# ===========================================================================
# Surface / noise-level vocabulary — MUST match kernels.cuh's
# kSurfacePlane/.../kSurfaceEdge constants field-for-field (data/README.md
# documents the shared contract).
# ===========================================================================
SURF_PLANE, SURF_SPHERE, SURF_CYLINDER, SURF_EDGE = 0, 1, 2, 3
NOISE_NONE, NOISE_LOW, NOISE_HIGH = 0, 1, 2
NOISE_SIGMA_M = {NOISE_NONE: 0.0, NOISE_LOW: 0.003, NOISE_HIGH: 0.015}

N_PER_COHORT = 700          # points per (surface, noise) cohort
R0_M = 14.0                 # nearest cohort's range, meters
R_STEP_M = 5.0               # range step between noise-level siblings, meters
PATCH_HALF_M = 1.5           # plane/edge tangent half-extent, meters (3 m x 3 m patch)
SPHERE_RADIUS_M = 1.0
SPHERE_CAP_HALF_DEG = 80.0
CYLINDER_RADIUS_M = 1.0
CYLINDER_HALF_HEIGHT_M = 1.5
CYLINDER_HALF_ARC_DEG = 80.0
EDGE_DIHEDRAL_DEG = 90.0      # interior angle between the two faces (a right-angle corner -- common, and a strong curvature signal)
EDGE_HALF_LENGTH_M = 1.5      # extent ALONG the ridge line (axis u) -- generous, for spatial variety
EDGE_T_MAX_M = 0.05           # extent AWAY from the ridge, PER FACE -- deliberately narrow (see gen_edge)

# One fixed (az, el) viewing direction per SURFACE TYPE — deliberately not
# axis-aligned (a plane/edge normal to a world axis would hide bugs that
# only a general 3-vector eigenproblem exposes; THEORY.md "Numerical
# considerations" makes this point explicitly).
SURFACE_VIEW_ANGLES = {
    SURF_PLANE:    (20.0, 8.0),
    SURF_SPHERE:   (110.0, -5.0),
    SURF_CYLINDER: (200.0, 12.0),
    SURF_EDGE:     (290.0, -8.0),
}

SENSOR = (0.0, 0.0, 0.0)


def gen_plane(rng: Xorshift32, C, view_dir, u, v, sigma):
    """PLANE cohort: a tilted flat patch, exact normal constant, exact
    curvature 0 everywhere (module docstring)."""
    normal = v_normalize(rotate_about_axis(rotate_about_axis(view_dir, u, math.radians(12.0)), v, math.radians(8.0)))
    a, b = make_perp_basis(normal)   # the plane's OWN tangent frame (perpendicular to the TILTED normal, not to view_dir)

    side = int(math.ceil(math.sqrt(N_PER_COHORT)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N_PER_COHORT:
                break
            # Grid + small jitter: near-uniform density (minimizes the
            # boundary-thinning "isolated" effect a purely random scatter
            # would cause — see kernels.cuh's kCellSizeM sizing comment).
            jx = rng.uniform(-0.4, 0.4)
            jy = rng.uniform(-0.4, 0.4)
            uc = -PATCH_HALF_M + 2.0 * PATCH_HALF_M * (i + 0.5 + jx) / side
            vc = -PATCH_HALF_M + 2.0 * PATCH_HALF_M * (j + 0.5 + jy) / side
            clean = v_add(C, v_add(v_scale(a, uc), v_scale(b, vc)))
            noisy = v_add(clean, v_scale(normal, sigma * rng.gauss()))
            pts.append((noisy, normal, 0.0, clean))
    return pts[:N_PER_COHORT]


def gen_sphere(rng: Xorshift32, C, view_dir, u, v, sigma):
    """SPHERE cohort: the sensor-facing CAP, uniform-area sampled via
    inverse-CDF on cos(beta) (module docstring). Exact normal = outward
    radial; exact curvature = 1/radius."""
    cos_cap = math.cos(math.radians(SPHERE_CAP_HALF_DEG))
    pts = []
    for _ in range(N_PER_COHORT):
        cos_beta = 1.0 - rng.uniform01() * (1.0 - cos_cap)   # uniform in [cos_cap, 1] -> uniform AREA on the cap
        beta = math.acos(max(-1.0, min(1.0, cos_beta)))
        az2 = rng.uniform(0.0, 2.0 * math.pi)
        d = v_normalize(v_add(v_scale(view_dir, math.cos(beta)),
                              v_scale(v_add(v_scale(u, math.cos(az2)), v_scale(v, math.sin(az2))), math.sin(beta))))
        clean = v_add(C, v_scale(d, SPHERE_RADIUS_M))
        noisy = v_add(clean, v_scale(d, sigma * rng.gauss()))
        pts.append((noisy, d, 1.0 / SPHERE_RADIUS_M, clean))
    return pts


def gen_cylinder(rng: Xorshift32, C, view_dir, u, v, sigma):
    """CYLINDER cohort: axis = u, visible +-CYLINDER_HALF_ARC_DEG arc.
    Exact normal = radial (perpendicular to axis); exact curvature =
    1/radius (one principal curvature; the axis direction is flat)."""
    axis = u
    pts = []
    for _ in range(N_PER_COHORT):
        h = rng.uniform(-CYLINDER_HALF_HEIGHT_M, CYLINDER_HALF_HEIGHT_M)
        beta = math.radians(rng.uniform(-CYLINDER_HALF_ARC_DEG, CYLINDER_HALF_ARC_DEG))
        cross_dir = v_normalize(v_add(v_scale(view_dir, math.cos(beta)), v_scale(v, math.sin(beta))))
        clean = v_add(v_add(C, v_scale(cross_dir, CYLINDER_RADIUS_M)), v_scale(axis, h))
        noisy = v_add(clean, v_scale(cross_dir, sigma * rng.gauss()))
        pts.append((noisy, cross_dir, 1.0 / CYLINDER_RADIUS_M, clean))
    return pts, axis


def gen_edge(rng: Xorshift32, C, view_dir, u, v, sigma):
    """EDGE cohort: two half-planes meeting at a ridge (axis u) with
    EDGE_DIHEDRAL_DEG interior angle (module docstring). Every point's
    exact normal is its OWN face's normal; there is NO single true
    curvature at a discontinuity — every point is written with the -1.0
    'undefined' sentinel (data/README.md).

    WHY the perpendicular extent (EDGE_T_MAX_M) is so much smaller than the
    plane cohort's patch half-width: this cohort's entire PURPOSE is to be
    the degeneracy-detection stress test (README/THEORY.md) — a point's
    K-nearest-neighborhood only straddles BOTH faces (the geometry that
    actually inflates surface variation) when the point sits within roughly
    one neighborhood-radius of the ridge. At this project's local point
    spacing (~0.04 m near the ridge), a K=16 neighborhood radius is
    ~0.09-0.11 m — so EDGE_T_MAX_M=0.18 m keeps MOST of the cohort within
    that influence zone, deliberately, the same way 02.05's dense/sparse
    regions are deliberately over-designed to trigger their target behavior
    clearly rather than left to chance."""
    phi_half = math.radians((180.0 - EDGE_DIHEDRAL_DEG) / 2.0)
    axis = u
    nA = v_normalize(v_add(v_scale(view_dir, math.cos(phi_half)), v_scale(v, math.sin(phi_half))))
    nB = v_normalize(v_sub(v_scale(view_dir, math.cos(phi_half)), v_scale(v, math.sin(phi_half))))
    tA = v_normalize(v_add(v_scale(view_dir, -math.sin(phi_half)), v_scale(v, math.cos(phi_half))))
    tB = v_normalize(v_sub(v_scale(view_dir, -math.sin(phi_half)), v_scale(v, math.cos(phi_half))))

    pts = []
    half = N_PER_COHORT // 2
    for count, normal, tangent in ((half, nA, tA), (N_PER_COHORT - half, nB, tB)):
        for _ in range(count):
            h = rng.uniform(-EDGE_HALF_LENGTH_M, EDGE_HALF_LENGTH_M)
            t = rng.uniform(0.0, EDGE_T_MAX_M)   # distance from the ridge line, t=0 AT the ridge
            clean = v_add(v_add(C, v_scale(axis, h)), v_scale(tangent, t))
            noisy = v_add(clean, v_scale(normal, sigma * rng.gauss()))
            pts.append((noisy, normal, -1.0, clean))
    return pts


def build_cohorts(rng: Xorshift32):
    """Build all 12 (surface, noise) cohorts in a FIXED, documented order:
    surface-major (plane, sphere, cylinder, edge), noise-minor (none, low,
    high) within each — main.cu / data/README.md rely on this exact order
    to name cohort indices in printed gate output."""
    all_points = []          # list of (x,y,z)
    all_normals = []         # list of (nx,ny,nz)
    all_curvature = []       # list of float (curvature or -1.0 sentinel)
    all_grazing = []         # list of float |cos(true_normal, sensor_dir)|
    cohorts = []              # list of dict: surface_id, noise_level, start, count, param, axis

    for surface_id in (SURF_PLANE, SURF_SPHERE, SURF_CYLINDER, SURF_EDGE):
        az, el = SURFACE_VIEW_ANGLES[surface_id]
        dir_from_sensor = direction(az, el)
        for noise_level in (NOISE_NONE, NOISE_LOW, NOISE_HIGH):
            R = R0_M + R_STEP_M * noise_level
            C = v_scale(dir_from_sensor, R)
            view_dir = v_normalize(v_sub(SENSOR, C))   # unit direction FROM the patch back TO the sensor
            u, v = make_perp_basis(view_dir)
            sigma = NOISE_SIGMA_M[noise_level]

            axis_out = (0.0, 0.0, 0.0)
            param_out = 0.0
            if surface_id == SURF_PLANE:
                pts = gen_plane(rng, C, view_dir, u, v, sigma)
            elif surface_id == SURF_SPHERE:
                pts = gen_sphere(rng, C, view_dir, u, v, sigma)
                param_out = SPHERE_RADIUS_M
            elif surface_id == SURF_CYLINDER:
                pts, axis_out = gen_cylinder(rng, C, view_dir, u, v, sigma)
                param_out = CYLINDER_RADIUS_M
            else:
                pts = gen_edge(rng, C, view_dir, u, v, sigma)

            start = len(all_points)
            for (p, n, curv, clean) in pts:
                all_points.append(p)
                all_normals.append(n)
                all_curvature.append(curv)
                # Grazing angle computed at the CLEAN (pre-noise) position —
                # the physically meaningful viewing geometry of the analytic
                # surface itself, not perturbed by this script's own added
                # noise (kernels.cuh kOrientationGrazingCos comment).
                sensor_dir = v_normalize(v_sub(SENSOR, clean))
                all_grazing.append(abs(v_dot(n, sensor_dir)))
            cohorts.append({
                'surface_id': surface_id, 'noise_level': noise_level,
                'start': start, 'count': len(pts),
                'param': param_out, 'axis': axis_out,
            })

    return all_points, all_normals, all_curvature, all_grazing, cohorts


def write_binary_sample(out_path: Path, points, normals, curvature, grazing, cohorts):
    """Write the committed sample as a small fixed binary format (no
    library, explicit little-endian struct packs — CLAUDE.md's no-black-
    boxes stance, 02.01/02.05's identical discipline):

        bytes  0.. 7  magic       b'NRMLCRV1' (8 bytes)
        bytes  8..11  int32       n_points
        bytes 12..15  int32       n_cohorts (12)
        bytes 16..19  int32       k_neighbors (must equal kernels.cuh kK)
        bytes 20..31  float32 x3  sensor_x, sensor_y, sensor_z
        bytes 32..    cohort table: n_cohorts * (
                          int32 surface_id, int32 noise_level,
                          int32 start, int32 count,
                          float32 param, float32 axis_x, float32 axis_y, float32 axis_z)
                      = n_cohorts * 32 bytes
        then: float32 x (n_points*3)  xyz            [meters, sensor frame]
              float32 x (n_points*3)  true_normal    [unit vector]
              float32 x (n_points)    true_curvature [1/meters, or -1.0 = undefined at the EDGE cohort]
              float32 x (n_points)    grazing_cos    [|cos(true_normal, direction-to-sensor)|, unitless]

    See data/README.md for the field-by-field description main.cu's
    load_sample() reads against.
    """
    n_points = len(points)
    n_cohorts = len(cohorts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        f.write(b'NRMLCRV1')
        f.write(struct.pack('<iii', n_points, n_cohorts, 16))
        f.write(struct.pack('<fff', *SENSOR))
        for c in cohorts:
            f.write(struct.pack('<iiiiffff', c['surface_id'], c['noise_level'], c['start'], c['count'],
                                c['param'], c['axis'][0], c['axis'][1], c['axis'][2]))

        flat_xyz = []
        for (x, y, z) in points:
            flat_xyz.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat_xyz)}f', *flat_xyz))

        flat_n = []
        for (x, y, z) in normals:
            flat_n.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat_n)}f', *flat_n))

        f.write(struct.pack(f'<{len(curvature)}f', *curvature))
        f.write(struct.pack(f'<{len(grazing)}f', *grazing))

    return n_points, n_cohorts


SURFACE_NAMES = {SURF_PLANE: 'plane', SURF_SPHERE: 'sphere', SURF_CYLINDER: 'cylinder', SURF_EDGE: 'edge'}
NOISE_NAMES = {NOISE_NONE: 'none', NOISE_LOW: 'low(3mm)', NOISE_HIGH: 'high(15mm)'}


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / 'data' / 'sample' / 'normals_scan.bin'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out,
                        help='output binary path (default: ../data/sample/normals_scan.bin)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)
    points, normals, curvature, grazing, cohorts = build_cohorts(rng)
    n_points, n_cohorts = write_binary_sample(args.out, points, normals, curvature, grazing, cohorts)

    print(f"[make_synthetic] SYNTHETIC normal/curvature truth scan (seed={args.seed}): "
          f"{n_cohorts} cohorts x {N_PER_COHORT} points/cohort = {n_points} points total")
    for c in cohorts:
        print(f"[make_synthetic]   cohort surface={SURFACE_NAMES[c['surface_id']]:9s} "
              f"noise={NOISE_NAMES[c['noise_level']]:10s} count={c['count']:4d} "
              f"start={c['start']:5d} param={c['param']:.2f}")
    print(f"[make_synthetic] wrote {args.out} ({n_points} points, labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
