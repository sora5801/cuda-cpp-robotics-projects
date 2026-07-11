#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.24
(Transparent/reflective object detection via polarization imaging).

Stdlib-only, deterministic (xorshift32, seed 42 — no random/numpy, per
CLAUDE.md paragraph 12: this repo standardizes on a hand-rolled xorshift32
PRNG rather than Python's Mersenne-Twister `random` module).

THIS SCRIPT IS THE PHYSICS FORWARD MODEL — the didactic heart of the
project (task brief). It does NOT call any shared C++/CUDA code (Python
cannot); it independently re-derives, in Python, the same closed-form
Fresnel equations kernels.cuh's fresnel_reflectances()/fresnel_dolp() state
in C++ — this cross-language duplication is deliberate: the "fresnel_anchor"
gate in ../src/main.cu succeeds only if BOTH independent implementations
agree with each other AND with the rendered pixels, which is a much
stronger claim than "the code agrees with itself".

What this script builds (all under ../data/sample/, SYNTHETIC, labeled
everywhere it appears — CLAUDE.md paragraph 8):
  1. A kW x kH scene: a smooth matte background (small residual DoLP, a
     gentle brightness gradient) plus THREE specular objects:
       - a flat GLASS PANE at a single documented incidence angle — real
         Fresnel equations, n=1.5, DoLP computed PER PIXEL from first
         principles;
       - a curved GLASS DOME (sphere-under-orthographic-view geometry) —
         local incidence angle varies with radius, so DoLP forms the
         classic "Brewster ring" real polarization cameras photograph on
         specular spheres;
       - a brushed METAL bar — a documented PHENOMENOLOGICAL DoLP curve
         (real metals need complex-refractive-index Fresnel equations, out
         of scope here — stated honestly, see THEORY.md).
     The two GLASS objects' mean intensity (S0) is made to MATCH the local
     background exactly (by construction) — intensity alone cannot see
     them; only their DoLP/AoLP differs from the background. This is the
     project's whole reason to exist (README "System context").
  2. Malus's law (kernels.cuh (*)) renders the four polarizer-angle
     intensities I0/I45/I90/I135 at every pixel from that pixel's true
     (S0,S1,S2), adds independent sensor noise, then MOSAICS: each pixel
     keeps only the ONE channel its own super-pixel phase measures (the
     real DoFP sensor's spatial multiplexing) -> mosaic.pgm.
  3. A SECOND, independent noise draw of the BACKGROUND ALONE (no objects
     at all) -> mosaic_negctrl.pgm, the negative_control gate's input.
  4. truth_maps.csv: the noise-free ground truth (S0, DoLP, AoLP, object
     label) at every pixel — main.cu never lets this file feed the
     pipeline, only the GATES that check the pipeline's OUTPUT against it.

Usage:
    python make_synthetic.py                  # writes into ../data/sample/
    python make_synthetic.py --out DIR         # custom output directory
"""

import argparse
import csv
import math
from pathlib import Path

# ===========================================================================
# SECTION 1 — canvas geometry. MUST MATCH kernels.cuh Section 1.
# ===========================================================================
K_W = 128
K_H = 128
K_N = K_W * K_H
DEFAULT_SEED = 42

# ===========================================================================
# SECTION 2 — DoFP phase/channel map. MUST MATCH kernels.cuh Section 2.
# Channel index -> polarizer angle (degrees); phase (px,py) -> channel.
# ===========================================================================
CHANNEL_ANGLE_DEG = [0.0, 45.0, 90.0, 135.0]


def channel_for_phase(px: int, py: int) -> int:
    if px == 0 and py == 0:
        return 2   # 90 deg
    if px == 1 and py == 0:
        return 1   # 45 deg
    if px == 0 and py == 1:
        return 3   # 135 deg
    return 0        # (1,1) -> 0 deg


# ===========================================================================
# SECTION 3 — scene object geometry + physics constants. MUST MATCH
# kernels.cuh Section 3 EXACTLY (both files render/interpret the SAME
# three objects against the SAME background).
# ===========================================================================
N_GLASS = 1.5

# Object 1: flat glass pane.
PANE_RECT = (14, 54, 14, 82)          # x0,x1,y0,y1
PANE_THETA_DEG = 35.0
PANE_AOLP_DEG = 90.0

# Object 2: curved glass dome.
DOME_CX = 92.0
DOME_CY = 40.0
DOME_RADIUS_PX = 24.0

# Object 3: brushed metal bar (curvature in y only).
METAL_RECT = (14, 114, 92, 120)
METAL_DOLP_MAX = 0.55
METAL_SAT = 0.15
METAL_S0_DN = 195.0

# Background.
BG_S0_BASE = 130.0
BG_S0_GRAD_AMP_X = 12.0
BG_DOLP = 0.018
BG_AOLP_DEG = 45.0

NOISE_STD_DN = 2.2

# MUST MATCH kernels.cuh Section 4 (detection pipeline constants) — restated
# here ONLY so params.csv can document, for a human reader, the exact
# numbers the demo will threshold at; this script does not threshold or
# detect anything itself.
DOLP_THRESHOLD = 0.10
INTENSITY_THRESHOLD = 25.0
MIN_COMPONENT_SIZE_PX = 40

LABEL_BACKGROUND = 0
LABEL_PANE = 1
LABEL_DOME = 2
LABEL_METAL = 3


# ===========================================================================
# SECTION 4 — xorshift32 PRNG (repo-standard; no random/numpy). Also
# provides a Box-Muller Gaussian draw (01.22's cached-second-sample idiom).
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9
        self._spare = None

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def next_float01(self) -> float:
        return self.next_u32() / 4294967296.0

    def next_gaussian(self, mean: float = 0.0, std: float = 1.0) -> float:
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return mean + std * v
        u1 = max(self.next_float01(), 1e-12)
        u2 = self.next_float01()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        self._spare = z1
        return mean + std * z0


# ===========================================================================
# SECTION 5 — the Fresnel physics (INDEPENDENT of kernels.cuh's C++ version —
# see this file's module docstring for why that independence matters).
# ===========================================================================
def fresnel_reflectances(theta_i_rad: float, n: float):
    """Return (Rs, Rp): power reflectances for s- and p-polarized light at
    a dielectric interface (medium 1 = air, n1=1), incidence angle
    theta_i_rad, refractive index n (medium 2). THEORY.md derives these two
    equations from Maxwell's boundary conditions; this function states them
    exactly as printed there, independently of kernels.cuh's C++ version."""
    cos_i = math.cos(theta_i_rad)
    sin_t = max(-1.0, min(1.0, math.sin(theta_i_rad) / n))   # Snell's law
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    rs = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    rp = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return rs * rs, rp * rp


def fresnel_dolp(theta_i_rad: float, n: float) -> float:
    """DoLP of unpolarized light specularly reflected at theta_i_rad off a
    dielectric of index n: (Rs-Rp)/(Rs+Rp), always in [0,1] since Rs>=Rp>=0
    for every external-reflection angle in [0,90) deg."""
    Rs, Rp = fresnel_reflectances(theta_i_rad, n)
    denom = Rs + Rp
    return (Rs - Rp) / denom if denom > 1e-12 else 0.0


# ===========================================================================
# SECTION 6 — per-pixel scene physics: for pixel (x,y), return
# (s0_dn, dolp, aolp_deg, label) — the noise-free GROUND TRUTH this script
# both renders FROM (via Malus's law, Section 7) and WRITES OUT (truth_maps.csv)
# for main.cu's independent gates to check its own measurements against.
# ===========================================================================
def scene_physics_at(x: int, y: int):
    # Background model FIRST — every object "sees through" to this same
    # local S0 (the pane/dome's built-in "matched intensity" property).
    bg_s0 = BG_S0_BASE + BG_S0_GRAD_AMP_X * ((x / (K_W - 1)) - 0.5)

    # -- glass pane: axis-aligned rect, ONE incidence angle everywhere in it.
    px0, px1, py0, py1 = PANE_RECT
    if px0 <= x < px1 and py0 <= y < py1:
        dolp = fresnel_dolp(math.radians(PANE_THETA_DEG), N_GLASS)
        return bg_s0, dolp, PANE_AOLP_DEG, LABEL_PANE

    # -- curved glass dome: disk of radius DOME_RADIUS_PX. Local incidence
    # angle theta_i(r) = asin(r/R) (orthographic sphere geometry,
    # THEORY.md); AoLP is RADIAL+90 (s-polarization perpendicular to the
    # local plane of incidence, which contains the radial direction).
    dx, dy = x - DOME_CX, y - DOME_CY
    r = math.hypot(dx, dy)
    if r <= DOME_RADIUS_PX:
        theta_i = math.asin(max(0.0, min(1.0, r / DOME_RADIUS_PX)))
        dolp = fresnel_dolp(theta_i, N_GLASS)
        phi_deg = math.degrees(math.atan2(dy, dx))
        aolp_deg = (phi_deg + 90.0) % 180.0   # wrap into the [0,180) linear-polarization convention
        return bg_s0, dolp, aolp_deg, LABEL_DOME

    # -- brushed metal bar: curvature in y only, y-symmetric about the
    # rect's vertical center. DoLP is a SATURATING curve (never a Brewster
    # zero -- the "different documented signature" from glass, THEORY.md
    # "Where this sits in the real world" names the complex-Fresnel formula
    # this stands in for); AoLP is CONSTANT (0 deg) across the whole bar.
    mx0, mx1, my0, my1 = METAL_RECT
    if mx0 <= x < mx1 and my0 <= y < my1:
        cy = 0.5 * (my0 + my1)
        radius = 0.5 * (my1 - my0)
        theta_local = math.asin(max(-1.0, min(1.0, (y - cy) / radius)))
        s = math.sin(abs(theta_local))
        dolp = METAL_DOLP_MAX * (s * s) / (METAL_SAT + s * s)
        return METAL_S0_DN, dolp, 0.0, LABEL_METAL

    # -- plain matte background: a small residual DoLP (no real matte
    # surface fully depolarizes light) at a fixed (irrelevant, tiny-signal)
    # AoLP.
    return bg_s0, BG_DOLP, BG_AOLP_DEG, LABEL_BACKGROUND


# ===========================================================================
# SECTION 7 — Malus's law rendering + DoFP mosaicking (kernels.cuh (*)):
# from TRUE (S0,S1,S2) at a pixel, render the 4 polarizer-angle intensities,
# add independent sensor noise, keep only the pixel's own phase channel.
# ===========================================================================
def stokes_from_dolp_aolp(s0: float, dolp: float, aolp_deg: float):
    aolp_rad = math.radians(aolp_deg)
    s1 = s0 * dolp * math.cos(2.0 * aolp_rad)
    s2 = s0 * dolp * math.sin(2.0 * aolp_rad)
    return s1, s2


def malus_intensity(s0: float, s1: float, s2: float, theta_deg: float) -> float:
    theta_rad = math.radians(theta_deg)
    return 0.5 * s0 + 0.5 * s1 * math.cos(2.0 * theta_rad) + 0.5 * s2 * math.sin(2.0 * theta_rad)


def render_mosaic(rng: Xorshift32, objects_enabled: bool):
    """Render one kH x kW mosaic (list-of-lists of float DN, PRE-clamp).
    objects_enabled=False renders the background alone (negative control)."""
    mosaic = [[0.0 for _ in range(K_W)] for _ in range(K_H)]
    for y in range(K_H):
        for x in range(K_W):
            if objects_enabled:
                s0, dolp, aolp_deg, _label = scene_physics_at(x, y)
            else:
                s0 = BG_S0_BASE + BG_S0_GRAD_AMP_X * ((x / (K_W - 1)) - 0.5)
                dolp, aolp_deg = BG_DOLP, BG_AOLP_DEG
            s1, s2 = stokes_from_dolp_aolp(s0, dolp, aolp_deg)
            px, py = x & 1, y & 1
            c = channel_for_phase(px, py)
            reading = malus_intensity(s0, s1, s2, CHANNEL_ANGLE_DEG[c])
            reading += rng.next_gaussian(0.0, NOISE_STD_DN)   # one sensor-noise draw per PHYSICAL photosite
            mosaic[y][x] = reading
    return mosaic


# ===========================================================================
# SECTION 8 — file I/O: PGM (P5), truth_maps.csv, params.csv.
# ===========================================================================
def write_pgm(path: Path, img, W: int, H: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{W} {H}\n255\n".encode("ascii"))
        buf = bytearray(W * H)
        i = 0
        for y in range(H):
            row = img[y]
            for x in range(W):
                v = row[x]
                v = 0.0 if v < 0.0 else (255.0 if v > 255.0 else v)
                buf[i] = int(v + 0.5)
                i += 1
        f.write(bytes(buf))


def write_truth_maps_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground truth — generated by scripts/make_synthetic.py for project 01.24\n")
        f.write("# NEVER fed into the detection pipeline; only main.cu's GATEs read this file.\n")
        w = csv.writer(f)
        w.writerow(["x", "y", "s0_dn", "dolp", "aolp_deg", "label"])
        for y in range(K_H):
            for x in range(K_W):
                s0, dolp, aolp_deg, label = scene_physics_at(x, y)
                w.writerow([x, y, f"{s0:.4f}", f"{dolp:.6f}", f"{aolp_deg:.4f}", label])


def write_params_csv(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    brewster_deg = math.degrees(math.atan(N_GLASS))
    rows = [
        ("seed", seed),
        ("canvas_w_px", K_W), ("canvas_h_px", K_H),
        ("n_glass", N_GLASS), ("brewster_angle_deg", f"{brewster_deg:.4f}"),
        ("pane_theta_deg", PANE_THETA_DEG), ("pane_aolp_deg", PANE_AOLP_DEG),
        ("pane_dolp_closed_form", f"{fresnel_dolp(math.radians(PANE_THETA_DEG), N_GLASS):.6f}"),
        ("dome_cx_px", DOME_CX), ("dome_cy_px", DOME_CY), ("dome_radius_px", DOME_RADIUS_PX),
        ("metal_dolp_max", METAL_DOLP_MAX), ("metal_sat", METAL_SAT), ("metal_s0_dn", METAL_S0_DN),
        ("bg_s0_base_dn", BG_S0_BASE), ("bg_s0_grad_amp_x_dn", BG_S0_GRAD_AMP_X),
        ("bg_dolp", BG_DOLP), ("bg_aolp_deg", BG_AOLP_DEG),
        ("noise_std_dn", NOISE_STD_DN),
        ("dolp_threshold", DOLP_THRESHOLD), ("intensity_threshold_dn", INTENSITY_THRESHOLD),
        ("min_component_size_px", MIN_COMPONENT_SIZE_PX),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data generation parameters — project 01.24 (regenerate: python make_synthetic.py)\n")
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in rows:
            w.writerow([k, v])


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default_out, help="output directory (default: ../data/sample)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (default 42)")
    args = ap.parse_args()
    out_dir: Path = args.out

    print(f"[make_synthetic] rendering the main scene ({K_W}x{K_H}: matte background + "
          f"glass pane + glass dome + brushed metal bar)...")
    rng_main = Xorshift32(args.seed)
    mosaic_main = render_mosaic(rng_main, objects_enabled=True)
    write_pgm(out_dir / "mosaic.pgm", mosaic_main, K_W, K_H)

    print("[make_synthetic] rendering the negative-control scene (background only, independent noise draw)...")
    rng_neg = Xorshift32(args.seed + 1)   # a DIFFERENT stream than the main scene's noise
    mosaic_neg = render_mosaic(rng_neg, objects_enabled=False)
    write_pgm(out_dir / "mosaic_negctrl.pgm", mosaic_neg, K_W, K_H)

    print("[make_synthetic] writing per-pixel ground truth (truth_maps.csv, "
          f"{K_N} rows)...")
    write_truth_maps_csv(out_dir / "truth_maps.csv")

    write_params_csv(out_dir / "params.csv", args.seed)

    print(f"[make_synthetic] done. Wrote mosaic.pgm, mosaic_negctrl.pgm, truth_maps.csv, params.csv "
          f"to {out_dir}  (all SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
