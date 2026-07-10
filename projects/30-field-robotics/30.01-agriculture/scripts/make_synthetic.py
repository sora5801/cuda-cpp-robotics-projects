#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 30.01
(Agriculture — Milestone 1: fruit detection + 3-D localization + ripeness).

Why this script exists (CLAUDE.md section 8: synthetic-first)
---------------------------------------------------------------
Real orchard RGB-D data has NO ground truth: nobody hand-measures the 3-D
center, radius, and ripeness of every fruit in a photograph. So, exactly like
project 01.02 (stereo depth), we author the scene directly in 3-D and RENDER
the sensor's view of it — every ground-truth number is exact by construction,
and the whole pipeline (including realistic sensor noise) is reproducible
from one fixed seed with no external tools, no downloads, no cameras.

What it writes (into ../data/sample/, well under the repo's data budget):

    rgb.ppm          640x480, 8-bit RGB (PPM P6)   — the color image
    depth.pgm        640x480, 16-bit gray (PGM P5) — depth in MILLIMETERS,
                     big-endian (the NetPBM 16-bit convention), WITH realistic
                     depth-sensor noise baked in (see DEPTH_NOISE_K below)
    ground_truth.csv one row per fruit: exact 3-D center/radius/ripeness plus
                     a measured OCCLUSION statistic (visible_frac) — see
                     below. This is the ONLY place the true fruit list exists;
                     the C++ pipeline never sees it except in the verification
                     stage of main.cu.

The scene, fully specified here (this file IS the scene's specification —
kernels.cuh's header comment cross-references every constant below):

  CAMERA — pinhole, fx=fy=525 px (the classic Kinect-v1 / TUM-RGBD intrinsic
  value; a well-known real-world anchor, not invented), principal point at
  the image center (320, 240). Camera looks down +Z (OpenCV/optical camera
  convention: x-right, y-down, z-forward — SYSTEM_DESIGN.md section 3.2
  notes this is the documented exception to the repo's default body frame).

  FRUIT — N_FRUIT=25 spheres, radius 2.8-4.8 cm (small-citrus to apple
  scale), placed at depths 1-4 m within the camera frustum. Each fruit's
  RIPENESS in [0.35, 1.0] maps to a HUE in [0, 78] degrees (green-yellow at
  0.35 through orange to full red at 1.0) via hue = 120*(1-ripeness) — see
  "why 0.35, not 0" below. Lambertian shading from an ON-ROBOT ring light
  (light direction ~= -normalize(point): the light effectively rides at the
  camera, physically realistic for an under-canopy robot working in variable
  or low ambient light — PRACTICE.md section 2 discusses this hardware
  choice) gives every fruit a lit side and a shadowed side, so detection must
  survive real shading, not just flat-colored disks.

  WHY RIPENESS STARTS AT 0.35, NOT 0: a fully unripe fruit (ripeness=0) would
  be hue=120 degrees — THE SAME hue as the green foliage background. Hue-only
  classical segmentation genuinely cannot separate green-on-green; this is a
  well-known limitation of color-based fruit detection (see THEORY.md and
  README "Limitations"), not a bug this generator hides. Scoping the sample
  scene to ripeness >= 0.35 (hue <= 78 degrees) keeps Milestone 1's synthetic
  benchmark honest: it tests what the classical HSV pipeline CAN do (spot
  color-separable fruit across a real ripening gradient), and documents, not
  fakes, what it cannot (green-on-green fruit needs texture/shape/NIR or a
  learned detector — production reality, stated plainly in README section 11).

  OCCLUSION — fruits are placed independently at random (seed 42); some
  overlap in the CAMERA's projection by chance, exactly as real fruit
  clusters do. A per-pixel Z-BUFFER resolves the overlap (nearer sphere
  wins), so occlusion is physically exact, not scripted. Each fruit's
  ground-truth row reports visible_frac = (pixels actually won) / (pixels
  its full unoccluded disk would cover) — this is the honest number the
  demo's detection-rate gate is set against (a heavily-occluded fruit that
  merges into its neighbor's blob is not a pipeline bug; main.cu's
  verification and this project's README/THEORY say so explicitly).

  BACKGROUND — procedural green foliage: 2-octave value noise (same
  lattice-hash + smoothstep technique as 01.02's make_synthetic.py) drives
  hue in [100,140] degrees (SATURATED GREEN, well clear of every fruit's hue
  range), saturation in [0.30,0.52], value in [0.25,0.60]. A handful of
  darker, low-saturation BRANCH STROKES (line segments) and a handful of
  tiny FRUIT-COLORED GLINT SPECKS (1-3 px each — meant to look like sun
  glints or a stray red leaf) are stamped on top; the glints are deliberate,
  documented FALSE-POSITIVE bait for the mask stage, which the morphological
  opening kernel (kernels.cu) must remove — see THEORY.md "How we verify
  correctness".

  SENSOR NOISE — depth is corrupted with zero-mean per-pixel Gaussian noise,
  std-dev sigma_z(Z) = DEPTH_NOISE_K * Z^2 (a simplified, documented version
  of the well-known QUADRATIC range noise of structured-light depth sensors;
  see Khoshelham & Elberink 2012 for the Kinect-v1 case this loosely follows)
  — then quantized to whole millimeters. THEORY.md derives the localization
  error budget directly from this formula, and kernels.cu's robust depth
  estimator uses this SAME formula (a realistic assumption: a fielded system
  calibrates its own sensor's noise curve once, then uses it forever).

Usage:
    python make_synthetic.py                  # the committed 640x480 scene
    python make_synthetic.py --seed 7          # a different scene; DO NOT commit
"""

import argparse
import csv
import hashlib
import math
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene constants — MUST match the camera/noise constants in
# ../src/kernels.cuh (that header states the reverse cross-reference). Kept
# as module constants, not CLI flags: they are part of the taught, fixed
# scene, not something a learner should vary and still expect the committed
# ground_truth.csv / expected_output.txt to apply.
# ---------------------------------------------------------------------------
BASE_SEED = 42          # the repo-documented seed; every draw traces back here
WIDTH = 640
HEIGHT = 480
FX = 525.0              # px — classic Kinect-v1/TUM-RGBD focal length (documented real anchor)
FY = 525.0
CX = 320.0              # px — principal point at the image center
CY = 240.0

N_FRUIT = 25
Z_MIN, Z_MAX = 1.0, 4.0             # m — fruit depth range (must match kernels.cuh)
RADIUS_MIN, RADIUS_MAX = 0.028, 0.048  # m — small-citrus to apple scale
RIPENESS_MIN, RIPENESS_MAX = 0.35, 1.0  # see "why 0.35" in the file header

BG_Z_MIN, BG_Z_MAX = 4.2, 5.0       # m — foliage depth, always beyond every fruit (never occludes one)
AMBIENT_FLOOR = 0.30                # V-channel floor on a fruit's shadowed side (never pure black)

DEPTH_NOISE_K = 0.0015  # sigma_z(Z) = DEPTH_NOISE_K * Z^2 (meters); must match kernels.cuh kDepthNoiseK

N_BRANCHES = 10          # dark low-saturation strokes over the foliage
N_GLINTS = 6              # tiny fruit-colored false-positive specks (see file header)


# ---------------------------------------------------------------------------
# Deterministic integer hash -> float in [0, 1) — identical technique to
# 01.02's make_synthetic.py: a pure function of (ix, iy, seed), so the image
# is reproducible regardless of iteration order (unlike a stateful RNG).
# ---------------------------------------------------------------------------
def _hash01(ix: int, iy: int, seed: int) -> float:
    h = (ix * 374761393 + iy * 668265263 + seed * 2654435761) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / float(1 << 24)


def _smoothstep(t: float) -> float:
    """Hermite smoothing t*t*(3-2t) — see 01.02's generator for why this
    removes grid-aligned creases from bilinearly-interpolated lattice noise."""
    return t * t * (3.0 - 2.0 * t)


def _value_noise(x: float, y: float, seed: int, scale: float) -> float:
    """Single-octave value noise at continuous image coords (x, y), in [0,1].
    Same lattice-hash + smoothstep + bilinear technique as 01.02; see that
    file's docstring for the full derivation. Reused here for foliage hue/
    saturation/value/depth texture -- four independent noise "channels" are
    obtained just by passing four different `seed` offsets."""
    fx, fy = x / scale, y / scale
    ix, iy = math.floor(fx), math.floor(fy)
    tx, ty = _smoothstep(fx - ix), _smoothstep(fy - iy)
    v00 = _hash01(ix, iy, seed)
    v10 = _hash01(ix + 1, iy, seed)
    v01 = _hash01(ix, iy + 1, seed)
    v11 = _hash01(ix + 1, iy + 1, seed)
    a = v00 + (v10 - v00) * tx
    b = v01 + (v11 - v01) * tx
    return a + (b - a) * ty


def _fbm2(x: float, y: float, seed: int, scale: float) -> float:
    """2-octave fractal value noise in [0,1] -- coarse octave (weight 0.75)
    plus a finer octave (weight 0.25, half the scale) for a touch of detail,
    same coarse-dominated mixing rationale as 01.02 (foliage should look
    organic without becoming so high-frequency that every patch is locally
    unique -- some ambiguity is realistic and, again, honestly documented)."""
    n1 = _value_noise(x, y, seed, scale)
    n2 = _value_noise(x, y, seed + 1000, scale * 0.5)
    return 0.75 * n1 + 0.25 * n2


# ---------------------------------------------------------------------------
# HSV <-> RGB -- hand-rolled (no colorsys import) so the exact formula is
# visible here and can be compared line-by-line with the CPU/GPU versions in
# reference_cpu.cpp / kernels.cu (CLAUDE.md section 1: no black boxes). H in
# degrees [0,360), S and V in [0,1]. Standard six-sector algorithm.
# ---------------------------------------------------------------------------
def hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    h = h % 360.0
    c = v * s                                  # chroma
    hp = h / 60.0
    x = c * (1.0 - abs((hp % 2.0) - 1.0))
    if 0 <= hp < 1:   r1, g1, b1 = c, x, 0.0
    elif 1 <= hp < 2: r1, g1, b1 = x, c, 0.0
    elif 2 <= hp < 3: r1, g1, b1 = 0.0, c, x
    elif 3 <= hp < 4: r1, g1, b1 = 0.0, x, c
    elif 4 <= hp < 5: r1, g1, b1 = x, 0.0, c
    else:             r1, g1, b1 = c, 0.0, x
    m = v - c
    r, g, b = r1 + m, g1 + m, b1 + m
    clamp255 = lambda t: max(0, min(255, int(round(t * 255.0))))
    return clamp255(r), clamp255(g), clamp255(b)


# ---------------------------------------------------------------------------
# Box-Muller from two deterministic hash-derived uniforms -- same technique
# family as 08.01's host noise generator, but PIXEL-INDEXED instead of
# stream-indexed so depth noise is a pure, reproducible function of (x, y).
# ---------------------------------------------------------------------------
def _gaussian01(ix: int, iy: int, seed: int) -> float:
    u1 = _hash01(ix, iy, seed)
    u1 = u1 if u1 > 1e-7 else 1e-7             # never exactly 0 -> log() safe
    u2 = _hash01(ix, iy, seed + 777)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# Fruit sampling -- positions, radii, ripeness. Uses Python's random module
# (NOT the pixel hash) because these are drawn ONCE, sequentially, to lay out
# the scene -- order-dependence is fine and expected for a fixed-seed
# Random() stream (unlike the per-pixel texture, which must be order-free).
# ---------------------------------------------------------------------------
def sample_fruits(seed: int):
    rng = random.Random(seed)
    fruits = []
    for i in range(N_FRUIT):
        z = rng.uniform(Z_MIN, Z_MAX)
        # Frustum half-extent in METERS at this depth (similar triangles:
        # a point at the edge of the image projects to +-W/2 px, so its
        # camera-space X extent is z*(W/2)/fx -- THEORY.md derives this same
        # relation for the RADIUS back-projection the C++ pipeline performs).
        half_w = z * (WIDTH / 2.0) / FX * 0.815  # tighter than the full frustum -> fruit CLUSTER
        half_h = z * (HEIGHT / 2.0) / FY * 0.695 # (deliberately crowds the scene so some overlap occurs)
        x = rng.uniform(-half_w, half_w)
        y = rng.uniform(-half_h * 0.9, half_h * 0.9)
        radius = rng.uniform(RADIUS_MIN, RADIUS_MAX)
        ripeness = rng.uniform(RIPENESS_MIN, RIPENESS_MAX)
        sat = rng.uniform(0.70, 0.85)
        hue_jitter_seed = BASE_SEED + 200 + i    # per-fruit pixel-noise channel (texture, below)
        fruits.append({
            "id": i + 1, "x": x, "y": y, "z": z, "radius": radius,
            "ripeness": ripeness, "sat": sat, "seed": hue_jitter_seed,
        })
    return fruits


def sample_branches(seed: int):
    """N_BRANCHES random line segments (endpoints in pixel space) with a
    per-segment half-width, representing dark woody structure glimpsed
    through/behind the foliage canopy."""
    rng = random.Random(seed)
    segs = []
    for _ in range(N_BRANCHES):
        x0, y0 = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
        length = rng.uniform(80, 260)
        angle = rng.uniform(0, 2 * math.pi)
        x1, y1 = x0 + length * math.cos(angle), y0 + length * math.sin(angle)
        half_width = rng.uniform(2.5, 5.5)
        segs.append((x0, y0, x1, y1, half_width))
    return segs


def project_fruit(f):
    """Pinhole projection of a fruit's CENTER to pixel coords, plus its
    approximate on-screen radius (fx==fy, so a single scalar suffices):
    u = fx*x/z + cx, v = fy*y/z + cy, r_px = fx*radius/z (THEORY.md derives
    the radius relation as the inverse of the pipeline's back-projection)."""
    if f["z"] <= 0:
        return None, None, None
    u = FX * f["x"] / f["z"] + CX
    v = FY * f["y"] / f["z"] + CY
    r_px = FX * f["radius"] / f["z"]
    return u, v, r_px


def sample_glints(seed: int, fruits):
    """N_GLINTS tiny fruit-colored specks, placed away from every real
    fruit's projected disk so they are unambiguously FALSE positives (not
    just occluded fruit) -- the mask stage will catch them; the morphological
    opening stage must remove them (THEORY.md "How we verify correctness")."""
    rng = random.Random(seed)
    glints = []
    tries = 0
    while len(glints) < N_GLINTS and tries < 5000:
        tries += 1
        gx, gy = rng.uniform(20, WIDTH - 20), rng.uniform(20, HEIGHT - 20)
        far_enough = True
        for f in fruits:
            u, v, r_px = project_fruit(f)
            if u is None:
                continue
            if (gx - u) ** 2 + (gy - v) ** 2 < (r_px + 14.0) ** 2:
                far_enough = False
                break
        if far_enough:
            radius_px = rng.uniform(1.2, 3.0)
            hue = rng.uniform(0.0, 70.0)
            glints.append((gx, gy, radius_px, hue))
    return glints


def point_segment_dist(px, py, x0, y0, x1, y1) -> float:
    """Shortest distance from point (px,py) to segment (x0,y0)-(x1,y1) -- the
    standard projection-and-clamp formula, used to rasterize branch strokes
    without drawing them pixel-by-pixel with a line algorithm."""
    dx, dy = x1 - x0, y1 - y0
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = x0 + t * dx, y0 + t * dy
    return math.hypot(px - cx, py - cy)


# ---------------------------------------------------------------------------
# gen_scene -- the whole rendering pass. Returns (rgb_bytes, depth_mm_u16,
# fruits_with_stats). Structure:
#   1. background layer (foliage + branches + glints) for every pixel, with
#      its own smoothly-varying depth (never occludes a fruit by design);
#   2. per-fruit ray-sphere intersection over each fruit's projected bbox
#      ONLY (not the whole image -- this is what keeps a pure-Python renderer
#      fast: N_FRUIT * (a few thousand px) instead of N_FRUIT * 307200),
#      z-buffered against whatever is currently closest;
#   3. per-fruit visibility stats from the final owner grid;
#   4. per-pixel sensor depth noise, then millimeter quantization.
# ---------------------------------------------------------------------------
def gen_scene(seed: int):
    fruits = sample_fruits(seed)
    branches = sample_branches(seed + 5)
    glints = sample_glints(seed + 7, fruits)

    rgb = bytearray(WIDTH * HEIGHT * 3)
    depth_true = [[0.0] * WIDTH for _ in range(HEIGHT)]   # meters, PRE-noise (z-buffer working array)
    owner = [[0] * WIDTH for _ in range(HEIGHT)]           # 0 = background, else fruit id

    # ---- pass 1: background (foliage + branches + glints), every pixel ----
    for y in range(HEIGHT):
        for x in range(WIDTH):
            hue = 100.0 + 40.0 * _fbm2(x, y, BASE_SEED + 10, 46.0)      # [100,140] deg: saturated green
            sat = 0.30 + 0.22 * _fbm2(x, y, BASE_SEED + 11, 38.0)       # [0.30,0.52]
            val = 0.25 + 0.35 * _fbm2(x, y, BASE_SEED + 12, 34.0)       # [0.25,0.60]
            z_bg = BG_Z_MIN + (BG_Z_MAX - BG_Z_MIN) * _fbm2(x, y, BASE_SEED + 13, 70.0)

            # Branch strokes: hard-threshold onto a dark, low-saturation
            # brown (documented simplification -- no anti-aliased blend).
            on_branch = False
            for (x0, y0, x1, y1, hw) in branches:
                if point_segment_dist(x + 0.5, y + 0.5, x0, y0, x1, y1) <= hw:
                    on_branch = True
                    break
            if on_branch:
                hue = 32.0 + 6.0 * (_hash01(x, y, BASE_SEED + 20) - 0.5)
                sat = 0.40 + 0.10 * _hash01(x, y, BASE_SEED + 21)
                val = 0.14 + 0.08 * _hash01(x, y, BASE_SEED + 22)

            # Glint specks: small fruit-colored disks (see sample_glints doc).
            for (gx, gy, gr, ghue) in glints:
                if (x + 0.5 - gx) ** 2 + (y + 0.5 - gy) ** 2 <= gr * gr:
                    hue, sat, val = ghue, 0.80, 0.85
                    break

            r, g, b = hsv_to_rgb(hue, sat, val)
            idx = (y * WIDTH + x) * 3
            rgb[idx + 0], rgb[idx + 1], rgb[idx + 2] = r, g, b
            depth_true[y][x] = z_bg

    # ---- pass 2: fruits, bbox-restricted ray-sphere intersection ----------
    for f in fruits:
        u_c, v_c, r_px = project_fruit(f)
        if u_c is None:
            continue
        pad = 2.0   # small safety margin so the disk is never clipped by the bbox
        x_lo = max(0, int(math.floor(u_c - r_px - pad)))
        x_hi = min(WIDTH - 1, int(math.ceil(u_c + r_px + pad)))
        y_lo = max(0, int(math.floor(v_c - r_px - pad)))
        y_hi = min(HEIGHT - 1, int(math.ceil(v_c + r_px + pad)))

        cx3, cy3, cz3, r3 = f["x"], f["y"], f["z"], f["radius"]
        for y in range(y_lo, y_hi + 1):
            for x in range(x_lo, x_hi + 1):
                # Ray through pixel center, UNNORMALIZED with z-component 1:
                # a hit at parameter t has point = t*(dx,dy,1), so t IS the
                # hit's depth in meters directly (no extra division needed --
                # THEORY.md "The math" explains this parametrization).
                dxr = (x + 0.5 - CX) / FX
                dyr = (y + 0.5 - CY) / FY
                # Quadratic a*t^2 + b*t + c = 0 for |t*d - C|^2 = r^2:
                a = dxr * dxr + dyr * dyr + 1.0
                bq = -2.0 * (dxr * cx3 + dyr * cy3 + cz3)
                c = cx3 * cx3 + cy3 * cy3 + cz3 * cz3 - r3 * r3
                disc = bq * bq - 4.0 * a * c
                if disc < 0.0:
                    continue    # ray misses the sphere entirely
                sqrt_disc = math.sqrt(disc)
                t = (-bq - sqrt_disc) / (2.0 * a)   # nearer root (front face)
                if t <= 0.0:
                    continue    # sphere is behind the camera
                if t >= depth_true[y][x]:
                    continue    # something already there is nearer -- occluded

                # Shading: point in camera space, outward normal, and an
                # ON-ROBOT ring light (direction ~= -normalize(point): the
                # light rides at the camera -- file header explains why).
                px3, py3, pz3 = t * dxr, t * dyr, t
                nx, ny, nz = (px3 - cx3) / r3, (py3 - cy3) / r3, (pz3 - cz3) / r3
                plen = math.sqrt(px3 * px3 + py3 * py3 + pz3 * pz3)
                lx, ly, lz = -px3 / plen, -py3 / plen, -pz3 / plen
                diffuse = max(0.0, nx * lx + ny * ly + nz * lz)

                tex = _hash01(x, y, f["seed"])                 # per-pixel micro-texture, deterministic
                value = (AMBIENT_FLOOR + (1.0 - AMBIENT_FLOOR) * diffuse) * (0.92 + 0.16 * tex)
                value = max(0.0, min(1.0, value))
                sat = max(0.0, min(1.0, f["sat"] * (0.95 + 0.10 * _hash01(x, y, f["seed"] + 1))))
                hue = 120.0 * (1.0 - f["ripeness"]) + 3.0 * (_hash01(x, y, f["seed"] + 2) - 0.5)

                r8, g8, b8 = hsv_to_rgb(hue, sat, value)
                idx = (y * WIDTH + x) * 3
                rgb[idx + 0], rgb[idx + 1], rgb[idx + 2] = r8, g8, b8
                depth_true[y][x] = t
                owner[y][x] = f["id"]

    # ---- pass 3: per-fruit visibility stats from the final owner grid -----
    visible_px = {f["id"]: 0 for f in fruits}
    for y in range(HEIGHT):
        row = owner[y]
        for x in range(WIDTH):
            fid = row[x]
            if fid:
                visible_px[fid] += 1
    for f in fruits:
        _, _, r_px = project_fruit(f)
        ideal_px = math.pi * r_px * r_px if r_px else 0.0
        f["visible_px"] = visible_px[f["id"]]
        f["ideal_px"] = ideal_px
        f["visible_frac"] = (visible_px[f["id"]] / ideal_px) if ideal_px > 0 else 0.0

    # ---- pass 4: sensor depth noise + millimeter quantization -------------
    depth_mm = bytearray(WIDTH * HEIGHT * 2)   # 16-bit big-endian, per pixel
    for y in range(HEIGHT):
        for x in range(WIDTH):
            z = depth_true[y][x]
            sigma = DEPTH_NOISE_K * z * z
            z_noisy = z + sigma * _gaussian01(x, y, BASE_SEED + 99)
            mm = int(round(max(1.0, z_noisy * 1000.0)))
            mm = max(0, min(65535, mm))
            idx = (y * WIDTH + x) * 2
            depth_mm[idx + 0] = (mm >> 8) & 0xFF   # big-endian (NetPBM 16-bit convention)
            depth_mm[idx + 1] = mm & 0xFF

    return bytes(rgb), bytes(depth_mm), fruits


def write_ppm(path: Path, width: int, height: int, rgb: bytes) -> None:
    """8-bit binary color PPM (P6) -- the color sibling of 01.02's PGM writer:
    one ASCII header trio, then raw interleaved RGB bytes."""
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb)


def write_pgm16(path: Path, width: int, height: int, data16: bytes) -> None:
    """16-bit binary gray PGM (P5, maxval 65535) -- per the NetPBM spec,
    samples wider than one byte are BIG-ENDIAN (most-significant byte
    first); src/main.cu's loader documents and honors this same convention."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n65535\n".encode("ascii"))
        f.write(data16)


def write_ground_truth(path: Path, fruits) -> None:
    with open(path, "w", newline="") as f:
        f.write("# fruit_id,cx_m,cy_m,cz_m,radius_m,ripeness,visible_px,ideal_px_est,visible_frac\n")
        w = csv.writer(f)
        for fr in fruits:
            w.writerow([
                fr["id"], f'{fr["x"]:.6f}', f'{fr["y"]:.6f}', f'{fr["z"]:.6f}',
                f'{fr["radius"]:.6f}', f'{fr["ripeness"]:.6f}',
                fr["visible_px"], f'{fr["ideal_px"]:.2f}', f'{fr["visible_frac"]:.4f}',
            ])


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=BASE_SEED,
                    help=f"scene seed (default {BASE_SEED}, the committed scene)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default ../data/sample)")
    args = ap.parse_args()

    rgb, depth_mm, fruits = gen_scene(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = args.out_dir / "rgb.ppm"
    depth_path = args.out_dir / "depth.pgm"
    gt_path = args.out_dir / "ground_truth.csv"

    write_ppm(rgb_path, WIDTH, HEIGHT, rgb)
    write_pgm16(depth_path, WIDTH, HEIGHT, depth_mm)
    write_ground_truth(gt_path, fruits)

    total_bytes = sum(p.stat().st_size for p in (rgb_path, depth_path, gt_path))
    print(f"wrote {args.out_dir} : {WIDTH}x{HEIGHT} orchard RGB-D scene "
          f"({total_bytes} bytes total) - labeled SYNTHETIC (seed {args.seed})")

    well_visible = sum(1 for f in fruits if f["visible_frac"] >= 0.5)
    heavily_occluded = [f for f in fruits if f["visible_frac"] < 0.5]
    print(f"note: {well_visible}/{len(fruits)} fruits are >=50% visible "
          f"(unoccluded majority of their projected disk)")
    if heavily_occluded:
        ids = ", ".join(f'#{f["id"]}({f["visible_frac"]*100:.0f}%)' for f in heavily_occluded)
        print(f"note: heavily-occluded designed cases: {ids}")
    ripeness_vals = [f["ripeness"] for f in fruits]
    print(f"note: ripeness range in sample: [{min(ripeness_vals):.2f}, {max(ripeness_vals):.2f}]")
    print(f"note: rgb.ppm sha256={sha256_of(rgb_path)}")
    print(f"note: depth.pgm sha256={sha256_of(depth_path)}")
    print(f"note: ground_truth.csv sha256={sha256_of(gt_path)}")
    if args.seed != BASE_SEED:
        print("note: non-default seed - fine for experiments, do NOT commit these files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
