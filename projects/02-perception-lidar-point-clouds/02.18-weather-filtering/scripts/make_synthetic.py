#!/usr/bin/env python3
"""make_synthetic.py - synthetic sample-data generator for 02.18
(Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Weather-filtering research (Charron et al.'s DROR paper, the CADC/WADS snow
datasets) needs something almost no public dataset hands you cleanly: a
per-point ground-truth label of "is this return a real surface, or an
airborne scatterer (and which kind)". Real snow/rain datasets are hand- or
weakly-labeled at best. This script instead RENDERS the physics: a real
structured scene (ground/walls/car) is ray-cast by a 16-beam spinning LiDAR,
and three independent atmospheres (SNOW, RAIN, DUST) are overlaid on the
IDENTICAL scene using a Beer-Lambert-style single-scatter argument, so every
point's real/scatterer/weather-type label is exact by construction. No
download, no license question, bit-for-bit reproducible from a fixed seed
(42, xorshift32 - CLAUDE.md paragraph 12: no Python `random` module).

THE SCENE (world/sensor frame: x-forward, y-left, z-up, right-handed;
meters; sensor fixed at the origin, mounted kSensorHeightM above the ground
plane it looks down at).
------------------------------------------------------------------------
The SAME static structured scene is ray-cast under three different weather
conditions (this is a deliberate, documented simplification: a real fleet
would capture three genuinely different moments in time; here we hold the
scene fixed so every difference between the SNOW/RAIN/DUST runs below is
attributable ONLY to the atmosphere, a clean A/B/C comparison for teaching -
see README "Limitations"). Real, closed-form objects (exact ray intersection,
so ground truth is exact, no approximation - the same choice project 02.13's
generator makes and cites):

    GROUND     - infinite horizontal plane at z = -kSensorHeightM. The
                 workhorse of the near/far real-point density story: a ray
                 at shallow (near-horizontal) elevation grazes the ground at
                 very long range before it intersects, while a steep
                 downward ray hits close by - so the SAME fixed angular
                 sampling grid naturally produces DENSE near-range ground
                 points and SPARSE far-range ones, exactly the 1/r^2
                 footprint-area lesson project 02.01 teaches from the
                 opposite direction (voxel occupancy of a uniform-density
                 cloud), cited here as this project's precedent.
    WALL_NEAR  - a thin vertical box at x ~= 8 m (a solid, well-observed
                 near-range surface).
    WALL_FAR   - the same wall geometry at x ~= 32 m (an equally solid
                 surface, but now naturally sparse purely from beam
                 divergence/angular spacing at range - this project's
                 designed "real point, but far and sparse" cohort, the one
                 SOR is built to fail on; see THEORY.md).
    CAR        - a box roughly car-sized at x ~= 18 m (a third reflectivity
                 class, and a range band in between the two walls).

Every real surface gets a per-point INTENSITY from a documented Lambertian
reflectance model (rho_cohort * cos(theta_incidence), plus small sensor
noise) - the exact model project 02.20 (lidar intensity calibration) studies
in depth; this project treats the RAW, already-range-compensated intensity
as given and states that dependency honestly in THEORY.md/README (02.20 is
this project's forward-looking sibling, not yet built as of this writing).

AIRBORNE SCATTERERS (the weather itself) - three independent single-scatter
Beer-Lambert draws per beam, per weather condition:

    For a beam of direction d and a candidate path length L (the range to
    the real surface it would otherwise hit, or kMaxRangeM if it would
    otherwise see open sky), the probability that AT LEAST ONE particle
    scatters the beam somewhere along [0, L] is the classic extinction law

        p_hit = 1 - exp(-N * sigma * L)

    where N is the particle NUMBER DENSITY (particles / m^3) and sigma is
    ONE particle's geometric cross-section (m^2, sigma = pi * a^2 for a
    particle of radius a) - "N * sigma" is the medium's EXTINCTION
    COEFFICIENT (1/m), the same quantity that governs radar attenuation and
    optical depth in atmospheric physics generally. If a scatter event
    occurs, its RANGE along the beam is drawn from the truncated exponential
    distribution that a Poisson process implies for "the position of the
    FIRST event given that at least one occurred in [0, L]" (inverse-CDF
    sampling below) - not a uniform draw, which would be physically wrong
    (a photon is more likely to be stopped early in a dense medium than
    late).

    SNOW and RAIN fill the WHOLE scene uniformly (L = the beam's full
    candidate path); DUST is a single LOCALIZED PLUME (an axis-aligned box)
    with zero density outside it - only the beam's sub-path INSIDE the box
    contributes to the extinction integral, giving DUST a hard edge and a
    dense core exactly where the plume sits (the "hard case" the DROR/LIOR
    gates measure honestly).

    A scattered beam gets a LOW intensity from a second, independent
    physical argument: PARTIAL BEAM INTERCEPTION. A real LiDAR beam is not
    an infinitely thin ray - it diverges with range, illuminating a disk of
    area pi*(range*divergence/2)^2 by the time it reaches distance `range`
    (kBeamDivergenceRad below). A millimeter-scale particle intercepts only
    a TINY fraction of that disk's power - fraction = sigma / footprint_area
    - and returns only that fraction, scaled by the particle's own backscatter
    reflectance rho_type. Because footprint_area grows as range^2 while a
    solid surface always fills the WHOLE beam (its own intensity model
    range-compensates that growth away), a scatterer's return intensity
    falls off much faster with range than a real surface's - the physical
    reason LIOR's low-intensity signal is a real, not arbitrary, effect.

    ATTENUATION HONESTY: some scattered beams are lost outright (no return
    recorded at all) rather than reporting the weak scatterer echo -
    kPLostGivenScatter per weather type models the fraction of scatter
    events strong enough to fully attenuate/desensitize the receiver before
    a usable return forms. This is also where this project's single-return
    scope cut lives: a real dual-return LiDAR could report BOTH the
    scatterer echo and the (attenuated) surface echo as two separate
    returns; this project's synthetic sensor, and the filters that consume
    it, model single-return only (README "Limitations").

Noise: intensity gets independent Gaussian sensor noise (sigma
kIntensityNoiseSigma); real-surface RANGE gets independent Gaussian noise
(sigma kRangeNoiseSigmaM, the same LiDAR range-noise floor project 02.13
uses). Both drawn from the repo's portable xorshift32 + Box-Muller generator
(never Python's `random` module).

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
# paragraph 12), seed 42.
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
        """One N(0, sigma^2) draw via Box-Muller (double precision, matching
        the flagship 08.01's gaussian() helper style)."""
        u1 = self.uniform01()
        u2 = self.uniform01()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return sigma * z


DEFAULT_SEED = 42

# ===========================================================================
# Beam model - MUST MATCH ../src/kernels.cuh's kNumBeams/kBeamElevDeg/
# kAzimuthMinDeg/kAzimuthStepDeg/kAzimuthSteps/kMaxRangeM (main.cu asserts
# the data file's header against those constants at load time, the 02.08/
# 02.13-style data/code consistency check).
#
# The 16-beam elevation table is the SAME table 02.08/02.09/02.13 cite from
# 01.18's derivation (repo convention: -15..+15 deg in 2 deg steps), reused
# verbatim rather than invented fresh.
# ===========================================================================
NUM_BEAMS = 16
BEAM_ELEV_DEG = [-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
                  1.0,   3.0,   5.0,   7.0,  9.0, 11.0, 13.0, 15.0]

AZIMUTH_MIN_DEG = -50.0     # a forward sector is enough: the scene sits ahead
AZIMUTH_STEP_DEG = 1.0      # 1 deg/step (02.08's convention)
AZIMUTH_STEPS = 100         # covers [-50, +49] deg
MAX_RANGE_M = 45.0          # sensor max usable range (real returns AND the
                             # snow/rain path-integration ceiling for open sky)

BEAMS_PER_SCAN = NUM_BEAMS * AZIMUTH_STEPS   # 1,600

SENSOR_HEIGHT_M = 1.2       # sensor mounted 1.2 m above the ground plane

RANGE_NOISE_SIGMA_M = 0.02        # real-surface range noise (02.13's value)
INTENSITY_NOISE_SIGMA = 0.01      # additive intensity sensor noise, both classes

# ===========================================================================
# Real scene geometry - closed-form ray/plane and ray/box intersection (the
# same "analytic scene, exact ground truth" choice 02.13/05.01 make).
# ===========================================================================
GROUND_Z = -SENSOR_HEIGHT_M
# Each object lives in its OWN azimuth sector (verified numerically against
# the beam grid below) so that nearest-hit occlusion never hides one real
# object entirely behind another - a bug caught while building this project
# (an earlier layout centered all three objects on azimuth 0 and the near
# wall, being both TALL and WIDE, shadowed the car/far-wall completely: 0
# hits on either). WALL_NEAR keeps the CENTER sector (az in roughly
# [-15,+15] deg) as a full-height "near wall"; WALL_FAR takes the LEFT
# sector (az < -18 deg) at long range; CAR is a small target in the RIGHT
# sector (az > +18 deg). Ground-plane occlusion (steep downward rings hit
# the ground before reaching x=8/18/32) is left as-is - that is real,
# expected geometry, not a bug (README "Limitations").
WALL_NEAR_BOX = ((7.9, -2.2, -SENSOR_HEIGHT_M), (8.1, 2.2, 1.8))
WALL_FAR_BOX = ((31.9, -38.5, -SENSOR_HEIGHT_M), (32.1, -10.0, 1.8))
CAR_BOX = ((16.0, 8.5, -SENSOR_HEIGHT_M), (20.0, 11.5, 0.3))

COHORT_GROUND = 0
COHORT_WALL_NEAR = 1
COHORT_WALL_FAR = 2
COHORT_CAR = 3
COHORT_NAMES = {COHORT_GROUND: "ground", COHORT_WALL_NEAR: "wall_near",
                 COHORT_WALL_FAR: "wall_far", COHORT_CAR: "car"}

# Lambertian reflectance rho per cohort (unitless, illustrative magnitudes,
# dated 2026-07-12 - real values depend on material, wavelength, and surface
# finish; verify current before relying on any of them for anything but
# teaching). Deliberately spans a realistic low-to-high range so a threshold
# tuned against it is meaningful, not trivial.
RHO_GROUND = 0.10       # dark asphalt-like surface
RHO_WALL = 0.35         # painted wall/panel
RHO_CAR = 0.55          # metallic paint, closer to normal incidence

# Range-compensation gain: a real LiDAR's intensity channel already divides
# out the 1/r^2 falloff so a given material reads the same intensity at any
# range (the reason a raw "intensity" column is usable at all for LIOR-style
# thresholds); this generator therefore does NOT re-apply a 1/r^2 term to
# real-surface intensity - only cos(incidence) and material rho matter. The
# residual per-channel gain error that motivates project 02.20 is injected
# at EVALUATION time (src/main.cu's intensity_dependence gate), not baked
# into this committed file - see that file's header for why.

# ===========================================================================
# Atmosphere models - Beer-Lambert extinction + partial-beam-interception
# intensity (file header derives both). All constants below are illustrative
# ORDER-OF-MAGNITUDE choices (dated 2026-07-12; real values require Mie
# scattering theory at the sensor's wavelength - THEORY.md "Where this sits
# in the real world"), tuned (like project 02.13's documented "Design
# iteration" section) so the three weather conditions produce comparably
# sized, clearly-separated cohorts for a legible demo - never fabricated,
# always stated as a deliberate scope/tuning choice.
# ===========================================================================
WEATHER_SNOW, WEATHER_RAIN, WEATHER_DUST = 0, 1, 2
WEATHER_NAMES = {WEATHER_SNOW: "snow", WEATHER_RAIN: "rain", WEATHER_DUST: "dust"}

# Beam divergence (full angle, radians) - a typical automotive spinning-LiDAR
# order of magnitude (illustrative, dated). Drives BOTH the DROR dynamic
# search radius (src/kernels.cuh) and the partial-interception intensity
# fraction below - one physical constant, two consumers, single-sourced.
BEAM_DIVERGENCE_RAD = 0.003

def footprint_area_m2(range_m: float) -> float:
    """pi * (range * divergence / 2)^2 - the beam's illuminated disk area at
    `range_m` (file header 'PARTIAL BEAM INTERCEPTION')."""
    radius = range_m * BEAM_DIVERGENCE_RAD * 0.5
    return math.pi * radius * radius

# particle_radius_m, number_density_per_m3, backscatter_rho, p_lost_given_scatter
SNOW_PARTICLE_RADIUS_M = 0.0030
SNOW_DENSITY_PER_M3 = 150.0
SNOW_RHO = 0.09
SNOW_P_LOST = 0.15

RAIN_PARTICLE_RADIUS_M = 0.0010
RAIN_DENSITY_PER_M3 = 2500.0
RAIN_RHO = 0.05
RAIN_P_LOST = 0.10

DUST_PARTICLE_RADIUS_M = 0.0002
# Dense LOCALIZED plume core, not scene-wide. Tuned (measured, not guessed -
# see THEORY.md "Numerical considerations" for the parameter sweep that
# produced this choice) to the density at which the plume's RECORDED-POINT
# spacing starts to rival the filters' own search radii: a lower density
# (e.g. 3-5x smaller) leaves both DROR and LIOR comfortably discriminating
# the plume; a much higher one (5-10x) starts to erase the real structure
# behind the plume entirely (the beam never survives long enough to reach
# it), which teaches attenuation but stops teaching the density-confusion
# lesson this project wants. This value sits in between: dense enough that
# the two filters' *different* search radii (DROR's small range-scaled
# radius at this near range vs LIOR's larger fixed companion radius) start
# to disagree with each other - the designed "hard case" (README/THEORY).
DUST_DENSITY_PER_M3 = 1200000.0
DUST_RHO = 0.12
DUST_P_LOST = 0.05
DUST_PLUME_BOX = ((3.0, -6.0, -SENSOR_HEIGHT_M), (7.0, 6.0, 1.5))   # a low, wide, NEAR
# cloud (3-7 m) rather than out among the real objects - deliberately placed
# so EVERY beam's near-field path crosses it regardless of azimuth sector or
# elevation ring (the steepest downward ring's ground intercept is ~4.6 m,
# still past the plume's near face at 3 m), giving the "localized but
# ubiquitous" plume this project's dust gates need (see the module docstring).

def particle_sigma_m2(radius_m: float) -> float:
    """Geometric cross-section pi*a^2 (file header's 'sigma')."""
    return math.pi * radius_m * radius_m

SNOW_SIGMA = particle_sigma_m2(SNOW_PARTICLE_RADIUS_M)
RAIN_SIGMA = particle_sigma_m2(RAIN_PARTICLE_RADIUS_M)
DUST_SIGMA = particle_sigma_m2(DUST_PARTICLE_RADIUS_M)


def ray_aabb(origin, dir_, box_min, box_max):
    """Ray/axis-aligned-box intersection (the slab method; transcribed
    independently here, same algorithm 02.13's generator cites from Kay &
    Kajiya 1986). Returns (t_enter, t_exit) with t_enter possibly < 0 (ray
    starts inside), or None if the ray misses the box entirely."""
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
    return (t_near, t_far)


def ray_plane_z(origin, dir_, z_plane):
    """Ray/horizontal-plane intersection z = z_plane. Returns t >= 0 or None
    (ray parallel to, or moving away from, the plane)."""
    if abs(dir_[2]) < 1e-12:
        return None
    t = (z_plane - origin[2]) / dir_[2]
    return t if t >= 0.0 else None


def beam_direction(elev_deg: float, az_deg: float):
    """Unit direction, spherical convention (matches 02.13's generator):
    az measured CCW from +x in the xy-plane, elev up from the xy-plane."""
    el = math.radians(elev_deg)
    az = math.radians(az_deg)
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def cast_real(origin, dir_):
    """Nearest REAL-surface hit (ground/wall_near/wall_far/car), or None.

    Returns (t, cohort, normal) for the closest positive intersection within
    MAX_RANGE_M - occlusion falls out for free from taking the minimum t,
    exactly as 02.13's cast_ray does for its own object list.
    """
    best_t, best_cohort, best_n = None, -1, (0.0, 0.0, 1.0)

    tg = ray_plane_z(origin, dir_, GROUND_Z)
    if tg is not None and tg <= MAX_RANGE_M:
        best_t, best_cohort, best_n = tg, COHORT_GROUND, (0.0, 0.0, 1.0)

    for box, cohort in ((WALL_NEAR_BOX, COHORT_WALL_NEAR),
                         (WALL_FAR_BOX, COHORT_WALL_FAR),
                         (CAR_BOX, COHORT_CAR)):
        hit = ray_aabb(origin, dir_, box[0], box[1])
        if hit is None:
            continue
        t_enter, _ = hit
        if t_enter < 0.0 or t_enter > MAX_RANGE_M:
            continue
        if best_t is None or t_enter < best_t:
            best_t, best_cohort, best_n = t_enter, cohort, (-1.0, 0.0, 0.0)

    if best_t is None:
        return None
    return best_t, best_cohort, best_n


def path_length_in_box(origin, dir_, box, t_min, t_max):
    """How much of the ray's [t_min, t_max] parameter interval lies inside
    `box` - the DUST plume's finite extinction path (file header)."""
    hit = ray_aabb(origin, dir_, box[0], box[1])
    if hit is None:
        return 0.0, (0.0, 0.0)
    t_enter, t_exit = hit
    lo = max(t_min, t_enter)
    hi = min(t_max, t_exit)
    return max(0.0, hi - lo), (lo, hi)


def sample_scatter_range(rng: Xorshift32, extinction_per_m: float, path_lo: float, path_hi: float) -> float:
    """Draw the FIRST scatter event's range within [path_lo, path_hi], given
    that at least one event occurred there (file header: truncated
    exponential via inverse-CDF, not a uniform draw).

    For a homogeneous Poisson process with rate `extinction_per_m` over a
    segment of length Ls = path_hi - path_lo, the first-event position s
    (measured from path_lo) has density f(s) proportional to exp(-k*s) on
    [0, Ls]; its CDF is F(s) = (1 - exp(-k*s)) / (1 - exp(-k*Ls)). Inverting
    F(s) = U for U ~ Uniform(0,1] gives the sampler below.
    """
    Ls = path_hi - path_lo
    if Ls <= 0.0:
        return path_lo
    k = extinction_per_m
    if k * Ls < 1e-9:
        # Degenerate near-zero-density segment: fall back to a uniform draw
        # (the exponential and uniform distributions coincide in this limit).
        return path_lo + rng.uniform01() * Ls
    u = rng.uniform01()
    norm = 1.0 - math.exp(-k * Ls)
    s = -math.log(1.0 - u * norm) / k
    return path_lo + min(s, Ls)


def try_scatter(rng: Xorshift32, path_lo: float, path_hi: float,
                 sigma: float, density: float, rho: float, p_lost: float):
    """One weather type's Beer-Lambert draw over [path_lo, path_hi] (file
    header). Returns None (no scatter), 'lost' (scattered but attenuated
    below detection), or (range_m, intensity) for a recorded scatterer point.
    """
    Ls = path_hi - path_lo
    if Ls <= 0.0:
        return None
    extinction_per_m = density * sigma
    p_hit = 1.0 - math.exp(-extinction_per_m * Ls)
    if rng.uniform01() >= p_hit:
        return None  # no particle intercepted this beam over this segment

    if rng.uniform01() < p_lost:
        return "lost"  # attenuation honesty: scattered, but no usable return

    r = sample_scatter_range(rng, extinction_per_m, path_lo, path_hi)
    footprint = footprint_area_m2(max(r, 0.5))   # floor to avoid a divide-by-tiny-range spike
    fraction = min(1.0, sigma / footprint)
    intensity = max(0.0, min(1.0, rho * fraction + rng.gaussian(INTENSITY_NOISE_SIGMA)))
    return (r, intensity)


def real_intensity(cohort: int, dir_, normal) -> float:
    rho = RHO_GROUND if cohort == COHORT_GROUND else (RHO_CAR if cohort == COHORT_CAR else RHO_WALL)
    cos_theta = abs(dir_[0] * normal[0] + dir_[1] * normal[1] + dir_[2] * normal[2])
    cos_theta = max(cos_theta, 0.02)   # grazing floor: a real receiver never reads exactly 0
    return max(0.0, min(1.0, rho * cos_theta))


def generate_scan(rng: Xorshift32, weather: int, rows: list, tallies: dict) -> None:
    """Ray-cast every beam of ONE weather scan; append CSV rows to `rows`."""
    origin = (0.0, 0.0, 0.0)   # sensor at the origin, every scan (file header)

    if weather == WEATHER_SNOW:
        sigma, density, rho, p_lost = SNOW_SIGMA, SNOW_DENSITY_PER_M3, SNOW_RHO, SNOW_P_LOST
    elif weather == WEATHER_RAIN:
        sigma, density, rho, p_lost = RAIN_SIGMA, RAIN_DENSITY_PER_M3, RAIN_RHO, RAIN_P_LOST
    else:
        sigma, density, rho, p_lost = DUST_SIGMA, DUST_DENSITY_PER_M3, DUST_RHO, DUST_P_LOST

    for elev_deg in BEAM_ELEV_DEG:
        for az_i in range(AZIMUTH_STEPS):
            az_deg = AZIMUTH_MIN_DEG + az_i * AZIMUTH_STEP_DEG
            dir_ = beam_direction(elev_deg, az_deg)

            real_hit = cast_real(origin, dir_)
            path_hi_ceiling = real_hit[0] if real_hit is not None else MAX_RANGE_M

            if weather == WEATHER_DUST:
                # Only the sub-path INSIDE the plume box carries extinction.
                seg_len, seg = path_length_in_box(origin, dir_, DUST_PLUME_BOX, 0.0, path_hi_ceiling)
                path_lo, path_hi = (seg if seg_len > 0.0 else (0.0, 0.0))
            else:
                path_lo, path_hi = 0.0, path_hi_ceiling

            scatter = try_scatter(rng, path_lo, path_hi, sigma, density, rho, p_lost)

            if scatter == "lost":
                tallies["lost"] += 1
                continue  # attenuation honesty: no point recorded at all

            if scatter is not None:
                r, inten = scatter
                # A scatter event at range r PRE-EMPTS the real surface (this
                # project's single-return scope cut, file header).
                x = dir_[0] * r
                y = dir_[1] * r
                z = dir_[2] * r
                rows.append((weather, x, y, z, inten, 0, weather, -1))
                tallies["scatter"] += 1
                continue

            if real_hit is None:
                tallies["miss"] += 1
                continue  # open sky, nothing scattered, nothing hit: no return

            t, cohort, normal = real_hit
            r_noisy = max(0.05, t + rng.gaussian(RANGE_NOISE_SIGMA_M))
            inten = real_intensity(cohort, dir_, normal)
            x = dir_[0] * r_noisy
            y = dir_[1] * r_noisy
            z = dir_[2] * r_noisy
            rows.append((weather, x, y, z, inten, 1, -1, cohort))
            tallies["real"] += 1
            tallies["cohort_" + COHORT_NAMES[cohort]] += 1


def generate(out_dir: Path, seed: int) -> None:
    rng = Xorshift32(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = out_dir / "points.csv"

    rows = []
    tallies_by_weather = {}
    for weather in (WEATHER_SNOW, WEATHER_RAIN, WEATHER_DUST):
        tallies = {"real": 0, "scatter": 0, "lost": 0, "miss": 0,
                   "cohort_ground": 0, "cohort_wall_near": 0, "cohort_wall_far": 0, "cohort_car": 0}
        generate_scan(rng, weather, rows, tallies)
        tallies_by_weather[weather] = tallies

    with points_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 02.18\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write("# scene: GROUND/WALL_NEAR/WALL_FAR/CAR real structure, ray-cast under three\n")
        f.write("#        independent atmospheres (SNOW/RAIN/DUST) - see this script's module\n")
        f.write("#        docstring for the full physics derivation\n")
        f.write(f"# num_beams={NUM_BEAMS}\n")
        f.write(f"# azimuth_steps={AZIMUTH_STEPS}\n")
        f.write(f"# azimuth_min_deg={AZIMUTH_MIN_DEG}\n")
        f.write(f"# azimuth_step_deg={AZIMUTH_STEP_DEG}\n")
        f.write(f"# max_range_m={MAX_RANGE_M}\n")
        f.write(f"# seed={seed}\n")
        f.write("# weather ids: 0=SNOW 1=RAIN 2=DUST\n")
        f.write("# surf_cohort ids (is_real==1 only): 0=GROUND 1=WALL_NEAR 2=WALL_FAR 3=CAR ; -1=n/a\n")
        f.write("# scatterer_type (is_real==0 only): 0=SNOW 1=RAIN 2=DUST ; -1=n/a\n")
        f.write("# columns: weather,x,y,z,intensity,is_real,scatterer_type,surf_cohort\n")
        for row in rows:
            weather, x, y, z, inten, is_real, scat_t, cohort = row
            f.write(f"{weather},{x:.6f},{y:.6f},{z:.6f},{inten:.6f},{is_real},{scat_t},{cohort}\n")

    total = len(rows)
    print(f"[make_synthetic] wrote {total} points to {points_path}")
    for weather in (WEATHER_SNOW, WEATHER_RAIN, WEATHER_DUST):
        t = tallies_by_weather[weather]
        n_scan = t["real"] + t["scatter"]
        print(f"[make_synthetic] {WEATHER_NAMES[weather]:5s}: {n_scan} points "
              f"({t['real']} real, {t['scatter']} scatterer, {t['lost']} lost-to-attenuation, "
              f"{t['miss']} clean miss) of {BEAMS_PER_SCAN} beams")
        print(f"    real cohorts: ground={t['cohort_ground']} wall_near={t['cohort_wall_near']} "
              f"wall_far={t['cohort_wall_far']} car={t['cohort_car']}")


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
