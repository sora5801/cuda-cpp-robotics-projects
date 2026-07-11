#!/usr/bin/env python3
# ===========================================================================
# make_synthetic.py — data generator for project 01.06
#                      AprilTag / ArUco GPU detector-decoder for high-rate
#                      fiducial localization
#
# Two independent jobs, both from scratch, stdlib-only Python, seed 42,
# xorshift32 (CLAUDE.md paragraph 8: synthetic-first; the machine facts brief
# for this project bans numpy AND std::uniform_real_distribution-style
# library RNGs — everything here is one small, auditable PRNG class):
#
#   1. GENERATE THE DICTIONARY — 32 codes, 6x6 grid (1-cell border ring + a
#      4x4 = 16-bit payload), by a seeded greedy search over the full 65536-
#      code space that enforces a MINIMUM PAIRWISE HAMMING DISTANCE across
#      all 4 rotations of every code (including a code against its OWN other
#      rotations — see generate_dictionary()'s docstring for why that matters
#      too). This teaches the coding-theory constraint that makes a fiducial
#      dictionary work at all: real families (AprilTag 16h5, ArUco
#      DICT_4X4_50) are built the SAME way, published, and standardized —
#      this project's codes are independently generated, never their bit
#      tables (README "Prior art" / "Limitations & honesty").
#   2. RENDER THREE SCENES — a full-perspective 6-tag detection scene, a
#      tag-free "distractor" scene (checkerboard + disks, deliberately
#      corner-rich, for the false-positive gate), and a "robustness" scene
#      (4 front-parallel tags with deliberately corrupted payload bits, at
#      and beyond the dictionary's correction capacity) — each with an
#      illumination gradient, a small Gaussian blur, and additive noise
#      (all documented, all synthetic — CLAUDE.md paragraph 8).
#
# EVERY geometry/camera constant below carries a "MUST MATCH kernels.cuh"
# comment — this script and the C++ pipeline must agree bit-for-bit on image
# size, intrinsics, tag size, and grid geometry, or the pipeline will not
# recover the ground truth this script writes (CLAUDE.md paragraph 12).
#
# Outputs, all under ../data/sample/ (committed, tiny, offline-runnable):
#   dictionary.bin, dictionary.csv        — the 32-code dictionary + metadata
#   scene_main.pgm, scene_main_ground_truth.csv
#   scene_distractor.pgm
#   scene_robustness.pgm, scene_robustness_ground_truth.csv
#
# Read this after: kernels.cuh (the contract this script fills in with data).
# ===========================================================================

import math
import os
import struct
import hashlib

# ===========================================================================
# xorshift32 — the ONE random source this entire script uses (seed 42, per
# the project brief). A 32-bit xorshift generator (Marsaglia 2003): three
# shift-xor steps produce a full-period (2^32 - 1) pseudo-random sequence
# from a single 32-bit state word — no tables, no library RNG, trivially
# auditable by reading the three lines below.
# ===========================================================================
class XorShift32:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9   # xorshift's fixed point is 0; never seed with 0

    def next_u32(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def uniform(self):
        """Uniform float in [0, 1)."""
        return self.next_u32() / 4294967296.0

    def uniform_range(self, lo, hi):
        return lo + self.uniform() * (hi - lo)

    def randint(self, lo, hi):
        """Uniform integer in [lo, hi], INCLUSIVE."""
        return lo + int(self.uniform() * (hi - lo + 1))

    def gauss(self, mean=0.0, sigma=1.0):
        """Standard Box-Muller transform: two uniforms -> one standard
        normal sample, scaled/shifted. u1 is floored away from 0 so log()
        never sees exactly zero (a 1-in-2^32 event, guarded anyway)."""
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + sigma * z

    def shuffled_range(self, n):
        """A Fisher-Yates shuffle of range(n), using this generator only —
        the deterministic-but-unpredictable SEARCH ORDER the dictionary
        generator walks the 65536-code space in."""
        order = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.randint(0, i)
            order[i], order[j] = order[j], order[i]
        return order


# ===========================================================================
# Geometry & camera constants — MUST MATCH kernels.cuh's identically-named
# constants (CLAUDE.md paragraph 12: single source of truth, mirrored here
# because C++ and Python cannot literally share a header).
# ===========================================================================
FULL_W = 480           # MUST MATCH kernels.cuh kFullW
FULL_H = 360            # MUST MATCH kernels.cuh kFullH
FX = 350.0              # MUST MATCH kernels.cuh kFx
FY = 350.0              # MUST MATCH kernels.cuh kFy
CX = (FULL_W - 1) * 0.5     # MUST MATCH kernels.cuh kCx = 239.5
CY = (FULL_H - 1) * 0.5     # MUST MATCH kernels.cuh kCy = 179.5

TAG_SIZE_M = 0.16       # MUST MATCH kernels.cuh kTagSizeM
TAG_HALF_M = TAG_SIZE_M * 0.5

GRID_N = 6              # MUST MATCH kernels.cuh kGridN
PAYLOAD_N = 4            # MUST MATCH kernels.cuh kPayloadN
PAYLOAD_BITS = PAYLOAD_N * PAYLOAD_N        # 16
NUM_DICT_CODES = 32      # MUST MATCH kernels.cuh kNumDictCodes

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sample")

# Rendering reflectance levels (0..255, BEFORE illumination gradient / blur /
# noise are applied — see render_tag() / the scene builders below): not pure
# 0/255, so the subsequent gradient multiply and additive noise have
# headroom in both directions without immediately saturating every pixel.
INK_BLACK = 28.0
INK_WHITE = 224.0
BACKGROUND_GRAY = 196.0


# ===========================================================================
# Part 1 — the dictionary: grid-geometry helpers (MUST MATCH kernels.cuh's
# is_border_cell / payload_bit_index / rotate_payload_90 — same formulas,
# reimplemented in Python since the two languages cannot share a header).
# ===========================================================================
def is_border_cell(r, c):
    """r, c are GRID coordinates in [0, GRID_N). MUST MATCH kernels.cuh's
    is_border_cell()."""
    return r == 0 or r == GRID_N - 1 or c == 0 or c == GRID_N - 1


def payload_bit_index(pr, pc):
    """pr, pc are PAYLOAD-LOCAL coordinates in [0, PAYLOAD_N) (i.e. already
    grid coords minus 1). MUST MATCH kernels.cuh's payload_bit_index(r, c)
    evaluated at r=pr+1, c=pc+1: (r-1)*4+(c-1) = pr*4+pc."""
    return pr * PAYLOAD_N + pc


def rotate90(code):
    """Bit permutation for a 90-degree clockwise in-plane rotation of the
    4x4 payload. MUST MATCH kernels.cuh's rotate_payload_90(): for every set
    bit at payload-local (pr, pc), the rotated code sets the bit at
    (pc, PAYLOAD_N-1-pr) — the standard clockwise image-index rotation."""
    out = 0
    for pr in range(PAYLOAD_N):
        for pc in range(PAYLOAD_N):
            if (code >> payload_bit_index(pr, pc)) & 1:
                npr, npc = pc, PAYLOAD_N - 1 - pr
                out |= 1 << payload_bit_index(npr, npc)
    return out


def all_rotations(code):
    r0 = code
    r1 = rotate90(r0)
    r2 = rotate90(r1)
    r3 = rotate90(r2)
    return (r0, r1, r2, r3)


def hamming(a, b):
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# generate_dictionary — seeded greedy search for NUM_DICT_CODES 16-bit codes
# with a target minimum Hamming distance, walking the FULL 65536-code space
# in a deterministic-but-shuffled order (XorShift32.shuffled_range) so the
# result is reproducible from the seed alone, not from insertion order into
# a hash table or similar accidental non-determinism.
#
# Two rejection rules enforce the SAME distance floor in two different
# senses, both load-bearing for a working dictionary:
#   SELF distance  — a code's own 4 rotations must be pairwise >= target_d
#                    apart. Without this, a tag mounted at 90 degrees could
#                    be misread as EXACTLY the same code at a different
#                    (wrong) rotation — the decoder would report the right
#                    ID but the wrong orientation, silently.
#   CROSS distance — every rotation of a NEW candidate must be >= target_d
#                    from every rotation of every ALREADY-ACCEPTED code —
#                    the usual "two different tags must not be confusable at
#                    any relative mounting angle" requirement.
# The search tries the LARGEST target_d first. A single greedy accept-in-
# order pass over the 65536-code space is ORDER-DEPENDENT (which codes get
# claimed first blocks off different later candidates), so for each target_d
# this function makes UP TO try_attempts_per_distance independent attempts,
# each with a FRESH shuffle drawn from the SAME continuing rng stream (still
# 100% deterministic from the one seed — CLAUDE.md paragraph 12 — just many
# shuffles deep instead of one) before giving up and backing off to
# target_d - 1. The loop reports whichever (distance, attempt) first reached
# num_codes, and a SEPARATE, independent pass afterward MEASURES the true
# achieved minimum distance over the final set (never assumed — CLAUDE.md
# paragraph 8, "never fabricate": the greedy search's TARGET is an ASK, not
# a proof, so main.py always re-derives the number it prints and writes).
# ---------------------------------------------------------------------------
def generate_dictionary(rng, num_codes=NUM_DICT_CODES, try_from_distance=6, try_down_to=3,
                        try_attempts_per_distance=24):
    def one_attempt(order, target_d):
        acc = []
        acc_rot_variants = []
        for code in order:
            if code == 0 or code == (1 << PAYLOAD_BITS) - 1:
                continue   # never assign the degenerate all-white / all-black code (see kernels.cuh)
            rots = all_rotations(code)

            ok = True
            for a in range(4):
                for b in range(a + 1, 4):
                    if hamming(rots[a], rots[b]) < target_d:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            for v in acc_rot_variants:
                if any(hamming(r, v) < target_d for r in rots):
                    ok = False
                    break
            if not ok:
                continue

            acc.append(code)
            acc_rot_variants.extend(rots)
            if len(acc) >= num_codes:
                break
        return acc

    accepted = []
    achieved_target = None
    for target_d in range(try_from_distance, try_down_to - 1, -1):
        for _attempt in range(try_attempts_per_distance):
            order = rng.shuffled_range(1 << PAYLOAD_BITS)   # next shuffle in the seed-42 stream
            accepted = one_attempt(order, target_d)
            if len(accepted) >= num_codes:
                break
        if len(accepted) >= num_codes:
            achieved_target = target_d
            break

    if achieved_target is None:
        raise RuntimeError(
            "generate_dictionary: could not find {} codes even at distance {} "
            "-- widen try_down_to or try_attempts_per_distance".format(num_codes, try_down_to))

    accepted = accepted[:num_codes]

    # Independent measurement pass (see docstring): the TRUE minimum distance
    # over every (code, rotation) pair in the final set, excluding a
    # rotation-vs-itself comparison (distance 0, meaningless).
    variant_lists = [all_rotations(c) for c in accepted]
    min_dist = 1 << 30
    for i in range(num_codes):
        for j in range(i, num_codes):
            for a in range(4):
                for b in range(4):
                    if i == j and a == b:
                        continue
                    d = hamming(variant_lists[i][a], variant_lists[j][b])
                    if d < min_dist:
                        min_dist = d
    correction_capacity = (min_dist - 1) // 2
    return accepted, min_dist, correction_capacity


# ===========================================================================
# Part 2 — small linear-algebra helpers for the pinhole camera model (pure
# Python, no numpy — 3x3/3-vector operations only, spelled out explicitly).
# ===========================================================================
def mat3_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation_matrix(yaw_rad, pitch_rad, roll_rad):
    """Compose R = Rz(roll) * Rx(pitch) * Ry(yaw) — an arbitrary but fixed
    Euler-angle order used ONLY to generate varied, valid rotation matrices
    for the synthetic scene; the pipeline never assumes this order, it only
    ever sees the resulting 3x3 R (kernels.cu's pose kernel recovers R
    directly from the homography, independent of how R was built here)."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    Rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]
    Rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]
    Ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    return mat3_mul(Rz, mat3_mul(Rx, Ry))


def build_homography(R, t):
    """H = K * [r1 r2 t] (see kernels.cuh's file header for the derivation):
    tag-plane point (X, Y, 0) meters -> pixel homogeneous coordinate.
    Columns of [r1 r2 t]: r1 = R's column 0, r2 = R's column 1, t as given."""
    r1 = (R[0][0], R[1][0], R[2][0])
    r2 = (R[0][1], R[1][1], R[2][1])
    H = [[0.0, 0.0, 0.0] for _ in range(3)]
    H[0][0] = FX * r1[0] + CX * r1[2]; H[0][1] = FX * r2[0] + CX * r2[2]; H[0][2] = FX * t[0] + CX * t[2]
    H[1][0] = FY * r1[1] + CY * r1[2]; H[1][1] = FY * r2[1] + CY * r2[2]; H[1][2] = FY * t[1] + CY * t[2]
    H[2][0] = r1[2];                    H[2][1] = r2[2];                    H[2][2] = t[2]
    return H


def mat3_inv(M):
    """Closed-form 3x3 inverse via the adjugate/cofactor formula — the
    homography here is never near-singular by construction (the tag always
    faces the camera with bounded tilt), so no pivoting is needed."""
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    Hh = -(a * f - c * d)
    I = a * e - b * d
    det = a * A + b * B + c * C
    inv_det = 1.0 / det
    return [[A * inv_det, D * inv_det, G * inv_det],
            [B * inv_det, E * inv_det, Hh * inv_det],
            [C * inv_det, F * inv_det, I * inv_det]]


def apply_h(H, X, Y):
    w = H[2][0] * X + H[2][1] * Y + H[2][2]
    u = (H[0][0] * X + H[0][1] * Y + H[0][2]) / w
    v = (H[1][0] * X + H[1][1] * Y + H[1][2]) / w
    return u, v


# ===========================================================================
# Canvas helpers: a flat list[float] of FULL_W*FULL_H, plus PGM I/O and the
# gradient/blur/noise post-processing every scene shares.
# ===========================================================================
def new_canvas(fill=BACKGROUND_GRAY):
    return [fill] * (FULL_W * FULL_H)


def apply_illumination_gradient(canvas, amplitude=0.18):
    """Linear left-to-right illumination gradient: factor ranges
    [1-amplitude, 1+amplitude] across the image width — the single most
    common reason a GLOBAL threshold fails (THEORY.md "The problem"),
    and exactly what the adaptive threshold's local mean is built to
    tolerate."""
    for y in range(FULL_H):
        row = y * FULL_W
        for x in range(FULL_W):
            factor = 1.0 + amplitude * (2.0 * (x / (FULL_W - 1)) - 1.0)
            canvas[row + x] *= factor


def gaussian_blur_5tap(canvas):
    """Separable 5-tap binomial-approximation-of-Gaussian blur (weights
    [1,4,6,4,1]/16, sigma ~= 1 px) — the same separable-filter idea the GPU
    pipeline's adaptive threshold uses for its box filter (kernels.cu's
    file header), applied here purely for scene realism (mild optical
    defocus / anti-aliasing of the hard-edged tag rendering below)."""
    weights = [1, 4, 6, 4, 1]
    wsum = 16
    tmp = [0.0] * (FULL_W * FULL_H)
    for y in range(FULL_H):
        row = y * FULL_W
        for x in range(FULL_W):
            s = 0.0
            for k in range(-2, 3):
                xx = min(max(x + k, 0), FULL_W - 1)
                s += canvas[row + xx] * weights[k + 2]
            tmp[row + x] = s / wsum
    out = [0.0] * (FULL_W * FULL_H)
    for y in range(FULL_H):
        for x in range(FULL_W):
            s = 0.0
            for k in range(-2, 3):
                yy = min(max(y + k, 0), FULL_H - 1)
                s += tmp[yy * FULL_W + x] * weights[k + 2]
            out[y * FULL_W + x] = s / wsum
    return out


def add_noise_and_quantize(canvas, rng, sigma=4.0):
    out = bytearray(FULL_W * FULL_H)
    for i, v in enumerate(canvas):
        n = v + rng.gauss(0.0, sigma)
        n = 0.0 if n < 0.0 else (255.0 if n > 255.0 else n)
        out[i] = int(n + 0.5)
    return out


def write_pgm(path, w, h, byte_data):
    with open(path, "wb") as f:
        f.write("P5\n{} {}\n255\n".format(w, h).encode("ascii"))
        f.write(bytes(byte_data))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ===========================================================================
# render_tag — rasterize ONE tag onto `canvas` (reflectance values, BEFORE
# gradient/blur/noise) using INVERSE mapping (pixel -> tag coordinate via
# H^-1), the same "ask every output location where its content comes from"
# strategy 01.01's remap stage teaches (no holes, unlike forward-warping the
# tag pixel by pixel). A 2x2 sub-pixel supersample per output pixel gives a
# light anti-aliasing pass on top of the later Gaussian blur.
#
# `code` is the RENDERED payload (which may already have bits flipped from
# the dictionary's true code — see make_robustness_scene() — so this
# function never looks at the dictionary itself, only at the 16-bit pattern
# it is told to draw).
# ---------------------------------------------------------------------------
def render_tag(canvas, H, code):
    Hinv = mat3_inv(H)
    half = TAG_HALF_M
    cell = TAG_SIZE_M / GRID_N

    corners_px = [apply_h(H, X, Y) for X, Y in ((-half, -half), (half, -half), (half, half), (-half, half))]
    xs = [p[0] for p in corners_px]; ys = [p[1] for p in corners_px]
    x_lo = max(0, int(math.floor(min(xs))) - 3)
    x_hi = min(FULL_W - 1, int(math.ceil(max(xs))) + 3)
    y_lo = max(0, int(math.floor(min(ys))) - 3)
    y_hi = min(FULL_H - 1, int(math.ceil(max(ys))) + 3)

    ss_offsets = (0.25, 0.75)
    for y in range(y_lo, y_hi + 1):
        for x in range(x_lo, x_hi + 1):
            acc = 0.0
            hits = 0
            for oy in ss_offsets:
                for ox in ss_offsets:
                    px, py = x + ox, y + oy
                    Xw = Hinv[0][0] * px + Hinv[0][1] * py + Hinv[0][2]
                    Yw = Hinv[1][0] * px + Hinv[1][1] * py + Hinv[1][2]
                    Ww = Hinv[2][0] * px + Hinv[2][1] * py + Hinv[2][2]
                    Xt, Yt = Xw / Ww, Yw / Ww
                    if -half <= Xt < half and -half <= Yt < half:
                        c = int((Xt + half) / cell)
                        r = int((Yt + half) / cell)
                        c = min(max(c, 0), GRID_N - 1)
                        r = min(max(r, 0), GRID_N - 1)
                        if is_border_cell(r, c):
                            black = True
                        else:
                            bit = (code >> payload_bit_index(r - 1, c - 1)) & 1
                            black = (bit == 1)
                        acc += INK_BLACK if black else INK_WHITE
                        hits += 1
            if hits > 0:
                # Blend supersampled tag ink with whatever background value
                # was already there for the (rare) partially-covered pixels
                # right at the tag's silhouette edge.
                frac = hits / 4.0
                canvas[y * FULL_W + x] = acc / hits * frac + canvas[y * FULL_W + x] * (1.0 - frac)


# ===========================================================================
# place_tags_no_overlap — reject-and-resample placement so a scene's tags
# never overlap and always stay a safe margin inside the frame. Returns a
# list of dicts with keys: R, t, tz, H, corners_px, cx, cy, radius.
# ---------------------------------------------------------------------------
def place_tags_no_overlap(rng, n, nominal_centers, jitter_px, depth_range, tilt_deg, roll_deg,
                          margin_px=15.0, sep_margin_px=10.0):
    placed = []
    for idx in range(n):
        cx0, cy0 = nominal_centers[idx]
        ok = False
        for _attempt in range(60):
            cx = cx0 + rng.uniform_range(-jitter_px, jitter_px)
            cy = cy0 + rng.uniform_range(-jitter_px, jitter_px)
            tz = rng.uniform_range(*depth_range)
            yaw = math.radians(rng.uniform_range(-tilt_deg, tilt_deg))
            pitch = math.radians(rng.uniform_range(-tilt_deg, tilt_deg))
            roll = math.radians(rng.uniform_range(-roll_deg, roll_deg))
            R = rotation_matrix(yaw, pitch, roll)
            tx = (cx - CX) * tz / FX
            ty = (cy - CY) * tz / FY
            t = (tx, ty, tz)
            H = build_homography(R, t)
            half = TAG_HALF_M
            corners = [apply_h(H, X, Y) for X, Y in ((-half, -half), (half, -half), (half, half), (-half, half))]
            xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
            if min(xs) < margin_px or max(xs) > FULL_W - margin_px or min(ys) < margin_px or max(ys) > FULL_H - margin_px:
                continue
            radius = max(math.hypot(p[0] - cx, p[1] - cy) for p in corners)
            collide = False
            for other in placed:
                d = math.hypot(cx - other["cx"], cy - other["cy"])
                if d < radius + other["radius"] + sep_margin_px:
                    collide = True
                    break
            if collide:
                continue
            placed.append({"R": R, "t": t, "tz": tz, "H": H, "corners_px": corners,
                          "cx": cx, "cy": cy, "radius": radius})
            ok = True
            break
        if not ok:
            raise RuntimeError("place_tags_no_overlap: could not place tag {} after retries".format(idx))
    return placed


# ===========================================================================
# Scene 1 — scene_main: 6 tags, full perspective (varied depth/scale/tilt/
# roll), illumination gradient + blur + noise.
# ===========================================================================
def make_main_scene(rng, dictionary):
    nominal = [(100, 110), (240, 110), (380, 110), (100, 250), (240, 250), (380, 250)]
    placements = place_tags_no_overlap(rng, 6, nominal, jitter_px=15.0,
                                       depth_range=(0.60, 1.00), tilt_deg=25.0, roll_deg=38.0)
    dict_indices = rng.shuffled_range(len(dictionary))[:6]

    canvas = new_canvas()
    rows = []
    for i, (p, dict_idx) in enumerate(zip(placements, dict_indices)):
        code = dictionary[dict_idx]
        render_tag(canvas, p["H"], code)
        rows.append({
            "tag_index": i, "dict_id": dict_idx,
            "corners": p["corners_px"],
            "R": p["R"], "t": p["t"],
        })

    apply_illumination_gradient(canvas, amplitude=0.18)
    blurred = gaussian_blur_5tap(canvas)
    bytes_out = add_noise_and_quantize(blurred, rng, sigma=4.0)

    write_pgm(os.path.join(OUT_DIR, "scene_main.pgm"), FULL_W, FULL_H, bytes_out)

    with open(os.path.join(OUT_DIR, "scene_main_ground_truth.csv"), "w") as f:
        f.write("tag_index,dict_id,"
               "corner0_x,corner0_y,corner1_x,corner1_y,corner2_x,corner2_y,corner3_x,corner3_y,"
               "R00,R01,R02,R10,R11,R12,R20,R21,R22,t0,t1,t2\n")
        for row in rows:
            corner_fields = []
            for (u, v) in row["corners"]:
                corner_fields.append("{:.4f}".format(u))
                corner_fields.append("{:.4f}".format(v))
            R = row["R"]; t = row["t"]
            R_fields = ["{:.8f}".format(R[i][j]) for i in range(3) for j in range(3)]
            t_fields = ["{:.8f}".format(v) for v in t]
            f.write("{},{},{},{}\n".format(
                row["tag_index"], row["dict_id"],
                ",".join(corner_fields), ",".join(R_fields + t_fields)))
    return rows


# ===========================================================================
# Scene 2 — scene_distractor: NO tags. A checkerboard block (squares smaller
# than kMinBBoxSidePx, so they are filtered by SIZE) plus several filled
# disks, two of them deliberately sized to PASS the size/fill-ratio filters
# — the disks' solid interior triggers the degenerate-payload (all-black)
# safeguard at decode time instead (kernels.cu's grid_decode_kernel doc
# comment) — this scene exercises BOTH false-positive defenses, not just
# the easy one.
# ===========================================================================
def make_distractor_scene(rng):
    canvas = new_canvas()

    cb_x0, cb_y0, cb_square, cb_n = 40, 40, 18, 8
    for gy in range(cb_n):
        for gx in range(cb_n):
            if (gx + gy) % 2 == 0:
                continue   # only paint the "black" squares; the rest stays background
            for y in range(cb_y0 + gy * cb_square, cb_y0 + (gy + 1) * cb_square):
                for x in range(cb_x0 + gx * cb_square, cb_x0 + (gx + 1) * cb_square):
                    canvas[y * FULL_W + x] = INK_BLACK

    small_disks = [(300, 60, 8), (340, 90, 6), (60, 300, 7)]
    large_disks = [(360, 260, 45), (420, 90, 40)]
    for (cx, cy, r) in small_disks + large_disks:
        for y in range(max(0, cy - r - 1), min(FULL_H, cy + r + 2)):
            for x in range(max(0, cx - r - 1), min(FULL_W, cx + r + 2)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    canvas[y * FULL_W + x] = INK_BLACK

    apply_illumination_gradient(canvas, amplitude=0.18)
    blurred = gaussian_blur_5tap(canvas)
    bytes_out = add_noise_and_quantize(blurred, rng, sigma=4.0)
    write_pgm(os.path.join(OUT_DIR, "scene_distractor.pgm"), FULL_W, FULL_H, bytes_out)


# ===========================================================================
# Scene 3 — scene_robustness: 4 front-parallel-ish tags, fixed depth. Two
# rendered with EXACTLY `correction_capacity` payload bits flipped from
# their true dictionary code (must still decode to the true ID); two with
# `correction_capacity + 1` bits flipped (must be REJECTED). Flip positions
# are chosen via a shuffled bit order from `rng`, distinct per tag.
#
# A coding-theory subtlety this function must respect (see
# generate_dictionary()'s docstring for the distance argument in full): at
# EXACTLY `correction_capacity` flips, the corrupted pattern is GUARANTEED
# (by the dictionary's own minimum-distance construction and the triangle
# inequality) to be closer to the TRUE code than to any other — no bit
# choice can break this, so the "accept" tags below use ANY random flip
# selection. At `correction_capacity + 1` flips that guarantee is GONE: the
# triangle inequality only promises every OTHER code is at distance >=
# min_distance - (capacity+1), which for this project's measured min_distance
# (5) equals 2 -- exactly AT the capacity, so an unlucky bit choice can
# coincidentally land the corrupted pattern within the ACCEPT radius of a
# DIFFERENT code (a real, if rare, decode failure mode: production error-
# correcting codes have exactly this property beyond their guaranteed
# radius). find_isolated_flip_mask() below searches for a flip choice that
# avoids this coincidence, so the "reject" tags demonstrate the INTENDED
# lesson (no accept at all) rather than an occasional unlucky mis-accept.
# ===========================================================================
def find_isolated_flip_mask(rng, dictionary, dict_idx, nflip, correction_capacity, max_attempts=500):
    """Return a flip_mask with exactly nflip bits set such that the
    resulting corrupted code's nearest OTHER dictionary code (any rotation)
    is STRICTLY FARTHER than correction_capacity away -- i.e. decoding the
    corrupted pattern is guaranteed to either recover the true code (only
    possible when nflip <= correction_capacity) or REJECT, never silently
    match a different code. Tries up to max_attempts random bit subsets."""
    true_code = dictionary[dict_idx]
    for _ in range(max_attempts):
        bit_order = rng.shuffled_range(PAYLOAD_BITS)[:nflip]
        flip_mask = 0
        for b in bit_order:
            flip_mask |= (1 << b)
        corrupted = true_code ^ flip_mask
        rots = all_rotations(corrupted)
        nearest_other = min(
            hamming(r, dictionary[j])
            for j in range(len(dictionary)) if j != dict_idx
            for r in rots)
        if nearest_other > correction_capacity:
            return flip_mask
    raise RuntimeError("find_isolated_flip_mask: no isolated {}-bit flip found for dict_id={} "
                       "after {} attempts".format(nflip, dict_idx, max_attempts))


def make_robustness_scene(rng, dictionary, correction_capacity):
    nominal = [(150, 110), (350, 110), (150, 250), (350, 250)]
    placements = place_tags_no_overlap(rng, 4, nominal, jitter_px=6.0,
                                       depth_range=(0.65, 0.65), tilt_deg=0.0, roll_deg=15.0,
                                       margin_px=20.0, sep_margin_px=15.0)
    dict_indices = rng.shuffled_range(len(dictionary))[:4]
    flip_counts = [correction_capacity, correction_capacity,
                  correction_capacity + 1, correction_capacity + 1]
    expected = ["accept", "accept", "reject", "reject"]

    canvas = new_canvas()
    rows = []
    for i, (p, dict_idx, nflip, exp) in enumerate(zip(placements, dict_indices, flip_counts, expected)):
        true_code = dictionary[dict_idx]
        flip_mask = find_isolated_flip_mask(rng, dictionary, dict_idx, nflip, correction_capacity)
        rendered_code = true_code ^ flip_mask
        render_tag(canvas, p["H"], rendered_code)
        rows.append({"tag_index": i, "true_dict_id": dict_idx, "num_flips": nflip,
                    "expected_outcome": exp, "corners": p["corners_px"]})

    apply_illumination_gradient(canvas, amplitude=0.12)
    blurred = gaussian_blur_5tap(canvas)
    bytes_out = add_noise_and_quantize(blurred, rng, sigma=3.0)
    write_pgm(os.path.join(OUT_DIR, "scene_robustness.pgm"), FULL_W, FULL_H, bytes_out)

    with open(os.path.join(OUT_DIR, "scene_robustness_ground_truth.csv"), "w") as f:
        f.write("tag_index,true_dict_id,num_flips,expected_outcome,"
               "corner0_x,corner0_y,corner1_x,corner1_y,corner2_x,corner2_y,corner3_x,corner3_y\n")
        for row in rows:
            corner_fields = []
            for (u, v) in row["corners"]:
                corner_fields.append("{:.4f}".format(u))
                corner_fields.append("{:.4f}".format(v))
            f.write("{},{},{},{},{}\n".format(row["tag_index"], row["true_dict_id"],
                                             row["num_flips"], row["expected_outcome"],
                                             ",".join(corner_fields)))
    return rows


# ===========================================================================
# main
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = XorShift32(SEED)

    print("[make_synthetic] generating dictionary (seed={})...".format(SEED))
    dictionary, min_dist, capacity = generate_dictionary(rng)
    print("[make_synthetic] dictionary: {} codes, measured min Hamming distance = {}, "
        "correction capacity floor((d-1)/2) = {}".format(len(dictionary), min_dist, capacity))

    dict_bin_path = os.path.join(OUT_DIR, "dictionary.bin")
    with open(dict_bin_path, "wb") as f:
        f.write(struct.pack("<5i", len(dictionary), PAYLOAD_BITS, GRID_N, min_dist, capacity))
        f.write(struct.pack("<{}H".format(len(dictionary)), *dictionary))

    with open(os.path.join(OUT_DIR, "dictionary.csv"), "w") as f:
        f.write("index,code_hex,code_binary_16bit\n")
        for i, c in enumerate(dictionary):
            f.write("{},0x{:04X},{}\n".format(i, c, format(c, "016b")))

    print("[make_synthetic] rendering scene_main.pgm (6 tags, full perspective)...")
    main_rows = make_main_scene(rng, dictionary)

    print("[make_synthetic] rendering scene_distractor.pgm (checkerboard + disks, no tags)...")
    make_distractor_scene(rng)

    print("[make_synthetic] rendering scene_robustness.pgm (2 at-capacity + 2 beyond-capacity tags)...")
    make_robustness_scene(rng, dictionary, capacity)

    print("[make_synthetic] done. SHA-256 checksums:")
    for name in ("dictionary.bin", "dictionary.csv",
               "scene_main.pgm", "scene_main_ground_truth.csv",
               "scene_distractor.pgm",
               "scene_robustness.pgm", "scene_robustness_ground_truth.csv"):
        path = os.path.join(OUT_DIR, name)
        print("  {}  {}".format(sha256_of(path), name))

    print("[make_synthetic] scene_main tag summary:")
    for row in main_rows:
        print("  tag {}: dict_id={} corner0=({:.1f},{:.1f})".format(
            row["tag_index"], row["dict_id"], row["corners"][0][0], row["corners"][0][1]))


if __name__ == "__main__":
    main()
