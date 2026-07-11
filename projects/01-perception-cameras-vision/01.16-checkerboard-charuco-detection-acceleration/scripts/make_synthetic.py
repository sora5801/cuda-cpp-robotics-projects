#!/usr/bin/env python3
# ===========================================================================
# make_synthetic.py — data generator for project 01.16
#                      Checkerboard/ChArUco detection acceleration for
#                      auto-calibration rigs
#
# Three jobs, all from scratch, stdlib-only Python, seed 42, xorshift32
# (CLAUDE.md paragraph 8: synthetic-first; no numpy, no std::uniform_real_
# distribution-style library RNG anywhere in this repo):
#
#   1. GENERATE THE MARKER DICTIONARY — 24 small ArUco-style codes (one per
#      WHITE square of the board — see board_geometry() below), 5x5 grid
#      (1-cell border ring + a 3x3 = 9-bit payload), by a seeded greedy
#      Hamming-distance search — the SAME coding-theory idea project 01.06
#      teaches for its 6x6/16-bit fiducial family (cite:
#      projects/01-perception-cameras-vision/01.06-.../scripts/make_synthetic.py
#      generate_dictionary()), reimplemented here from scratch at this
#      project's own (smaller) grid geometry. Unlike 01.06, THIS project's
#      markers are never searched for blind — they are sampled at a KNOWN
#      board-plane location through a homography the checkerboard corners
#      already gave us (see ../src/kernels.cu), so the codes only need to
#      distinguish "which white square is this" among 24 possibilities, not
#      survive a full open-world detection search.
#   2. RENDER EIGHT CALIBRATION-RIG VIEWS — a 7x5-inner-corner ChArUco board
#      (8x6 squares) photographed from 8 documented poses: two front-parallel
#      distances, two yaws, a pitch, a roll, one view rotated 180 degrees in
#      its own plane (the "ambiguity" view — see THEORY.md "The problem"),
#      and one view with a synthetic occluder patch covering ~25% of the
#      board (the "occlusion" view). Every view gets a mild illumination
#      gradient, a small Gaussian blur, and additive sensor noise.
#   3. RENDER ONE NEGATIVE-CONTROL IMAGE — textured clutter, no board at all,
#      for the false-positive gate.
#
# EVERY geometry/camera constant below carries a "MUST MATCH kernels.cuh"
# comment. The one constant this script knows that the C++ pipeline NEVER
# gets to see directly is the ground-truth camera intrinsics (fx, fy, cx,
# cy): the whole point of the mini-calibration stage (Zhang's method, see
# ../src/reference_cpu.cpp) is to RECOVER those numbers from image
# observations alone, so they live only in intrinsics_truth.csv, read by
# main.cu strictly for the calibration GATE, never fed into detection.
#
# Outputs, all under ../data/sample/ (committed, tiny, offline-runnable):
#   marker_dictionary.bin, marker_dictionary.csv   — the 24-code dictionary
#   view00.pgm .. view07.pgm                        — the 8 rig views
#   corners_truth.csv                                — every corner, every view
#   poses_truth.csv                                   — R, t per view
#   intrinsics_truth.csv                              — fx, fy, cx, cy (GATE-only)
#   occluder_truth.csv                                — view07's occluder rect
#   negative_control.pgm                              — board-free scene
#
# Read this after: kernels.cuh (the contract this script fills in with data).
# ===========================================================================

import math
import os
import struct
import hashlib

# ===========================================================================
# xorshift32 — the ONE random source this entire script uses (seed 42).
# Marsaglia's 32-bit xorshift: three shift-xor steps, full period 2^32-1,
# from a single 32-bit state word. No tables, no library RNG — the same
# three lines are reimplemented in C++ in kernels.cuh's xorshift32_next()
# so Python and C++ agree bit-for-bit given the same seed (CLAUDE.md §12).
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

    def gauss(self, mean=0.0, sigma=1.0):
        """Box-Muller: two uniforms -> one standard normal sample."""
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + sigma * z

    def shuffled_range(self, n):
        """Fisher-Yates shuffle of range(n) using this generator only."""
        order = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(self.uniform() * (i + 1))
            order[i], order[j] = order[j], order[i]
        return order


# ===========================================================================
# Geometry constants — MUST MATCH kernels.cuh's identically-named constants.
# ===========================================================================
IMG_W = 320             # MUST MATCH kernels.cuh kImgW
IMG_H = 240              # MUST MATCH kernels.cuh kImgH
NUM_VIEWS = 8             # MUST MATCH kernels.cuh kNumViews

SQUARES_X = 8             # MUST MATCH kernels.cuh kBoardSquaresX
SQUARES_Y = 6              # MUST MATCH kernels.cuh kBoardSquaresY
CORNERS_X = SQUARES_X - 1   # 7  -- MUST MATCH kernels.cuh kBoardCornersX
CORNERS_Y = SQUARES_Y - 1    # 5  -- MUST MATCH kernels.cuh kBoardCornersY
SQUARE_SIZE_M = 0.030          # MUST MATCH kernels.cuh kSquareSizeM (3 cm squares)

MARKER_GRID_N = 5                # MUST MATCH kernels.cuh kMarkerGridN
MARKER_PAYLOAD_N = 3              # MUST MATCH kernels.cuh kMarkerPayloadN
MARKER_PAYLOAD_BITS = MARKER_PAYLOAD_N * MARKER_PAYLOAD_N   # 9
MARKER_FILL_FRAC = 0.70            # MUST MATCH kernels.cuh kMarkerFillFrac
NUM_MARKER_CODES = 24               # MUST MATCH kernels.cuh kNumMarkerCodes (= #white squares, see below)

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sample")

# GROUND-TRUTH camera intrinsics. These numbers exist ONLY here and in
# intrinsics_truth.csv -- the C++ pipeline never reads them as input; it
# RECOVERS them (approximately) via Zhang's method and main.cu compares the
# recovered values against this file for the mini_calibration gate. fx != fy
# on purpose (non-square pixels are the general case; recovering both
# independently is part of the lesson).
FX_TRUE = 305.0
FY_TRUE = 295.0
CX_TRUE = (IMG_W - 1) * 0.5   # = 159.5 (pixel-center convention, matches 01.01/01.06)
CY_TRUE = (IMG_H - 1) * 0.5   # = 119.5

# Rendering reflectance levels (0..255, BEFORE gradient/blur/noise). MUST
# MATCH kernels.cuh's kInkMidThreshold (the fixed decode threshold sits at
# the midpoint of these two).
INK_BLACK = 30.0
INK_WHITE = 220.0
# The board's own outer silhouette alternates black/white squares against
# the background: at every such boundary crossing, a THREE-region
# T-junction forms (background above, two different-colored squares
# below), which can satisfy a naive det(Hessian)<0 + per-axis-curvature
# test without being a real two-color diagonal saddle (root-caused
# empirically; THEORY.md "Numerical considerations" derives the geometry
# and why kernels.cuh's kMaxDiagonalAsymmetry gate is the correct, general
# fix -- it does not depend on this background value at all). An ordinary
# mid-gray background is therefore fine; no special-casing needed here.
BACKGROUND_GRAY = 175.0

# Marker ink levels are DELIBERATELY LOWER CONTRAST than the checkerboard's
# own squares (60 vs 190 gray levels) -- same midpoint (125) so ONE decode
# threshold (kernels.cuh kInkMidThreshold) still separates black from white
# cleanly (a 60-level gap against sigma=2.5 sensor noise remains enormously
# significant, so decoding is unaffected), but the *second derivative* a
# marker's own internal bit boundaries can produce scales with contrast
# SQUARED -- roughly (60/190)^2 =~ 0.10x the checkerboard's own corner
# response -- which is what keeps the saddle detector (kernels.cuh's
# kSaddleRespThresh) from mistaking a marker's internal payload pattern for
# a real board corner (THEORY.md "Numerical considerations" measures this).
MARKER_INK_BLACK = 95.0
MARKER_INK_WHITE = 155.0


# ===========================================================================
# Board geometry helpers — MUST MATCH kernels.cuh's identically-named inline
# functions (is_white_square, corner_board_xy, marker_center_board_xy,
# mirror_corner, mirror_square, marker_payload_bit_index).
# ===========================================================================
def is_white_square(bx, by):
    """A square is WHITE (carries a marker) iff (bx+by) is odd. For an 8x6
    board this yields exactly 24 white squares (4 per row x 6 rows) -- which
    is why NUM_MARKER_CODES = 24 exactly, not a round-number coincidence."""
    return ((bx + by) & 1) == 1


def build_marker_id_table():
    """Row-major (by, then bx) enumeration of white squares -> sequential
    marker_id in [0, NUM_MARKER_CODES). MUST MATCH kernels.cuh's
    build_marker_id_table() -- both walk (by, bx) in the same nested order,
    so marker_id assignment agrees between Python and C++ without needing to
    ship a table (CLAUDE.md paragraph 12: single formula, mirrored)."""
    square_of_id = []
    id_of_square = {}
    next_id = 0
    for by in range(SQUARES_Y):
        for bx in range(SQUARES_X):
            if is_white_square(bx, by):
                id_of_square[(bx, by)] = next_id
                square_of_id.append((bx, by))
                next_id += 1
    assert next_id == NUM_MARKER_CODES, "white-square count drifted from NUM_MARKER_CODES"
    return square_of_id, id_of_square


SQUARE_OF_MARKER_ID, MARKER_ID_OF_SQUARE = build_marker_id_table()


def corner_board_xy(i, j):
    """Inner corner (i,j), i in [0,CORNERS_X), j in [0,CORNERS_Y), sits at
    board-plane meters ((i+1)*SQUARE, (j+1)*SQUARE) -- one square margin
    from the board's own outer edge on every side (the standard checkerboard
    convention: only INTERIOR vertices are corners). MUST MATCH kernels.cuh."""
    return ((i + 1) * SQUARE_SIZE_M, (j + 1) * SQUARE_SIZE_M)


def mirror_corner(i, j):
    """The board's own 180-degree in-plane rotational symmetry, expressed on
    inner-corner indices. THE source of the ChArUco ambiguity lesson (see
    THEORY.md): a plain checkerboard's own geometry is IDENTICAL after this
    relabeling, so corner detection alone cannot tell the two apart."""
    return (CORNERS_X - 1 - i, CORNERS_Y - 1 - j)


def mirror_square(bx, by):
    return (SQUARES_X - 1 - bx, SQUARES_Y - 1 - by)


def marker_payload_bit_index(pr, pc):
    """pr, pc are PAYLOAD-LOCAL coords in [0, MARKER_PAYLOAD_N). Row-major:
    k = pr*3+pc. MUST MATCH kernels.cuh's marker_payload_bit_index()."""
    return pr * MARKER_PAYLOAD_N + pc


def marker_is_border_cell(r, c):
    """r,c are GRID-LOCAL coords in [0, MARKER_GRID_N). MUST MATCH
    kernels.cuh's marker_is_border_cell()."""
    return r == 0 or r == MARKER_GRID_N - 1 or c == 0 or c == MARKER_GRID_N - 1


def hamming(a, b):
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# generate_marker_dictionary — seeded greedy search for NUM_MARKER_CODES
# 9-bit codes with a target minimum pairwise Hamming distance. Unlike 01.06,
# this project's markers are read at a KNOWN, homography-predicted location
# (never searched for), so we do NOT need each code to survive an unknown
# in-plane rotation: the board's own 180-degree symmetry (mirror_square) is
# the only "wrong hypothesis" the decoder ever has to reject (see
# ../src/kernels.cu's marker_decode_kernel), so the distance search only
# needs to separate DIFFERENT codes from each other, not a code from its own
# rotations. Reimplemented independently of 01.06's generate_dictionary()
# (different grid size, different rotation requirement) but the same
# shuffled-order greedy-with-backoff STRATEGY, cited honestly.
# ---------------------------------------------------------------------------
def generate_marker_dictionary(rng, num_codes=NUM_MARKER_CODES, try_from_distance=4, try_down_to=2,
                               try_attempts_per_distance=32):
    def one_attempt(order, target_d):
        acc = []
        for code in order:
            if code == 0 or code == (1 << MARKER_PAYLOAD_BITS) - 1:
                continue  # never assign the degenerate all-white / all-black code
            if all(hamming(code, other) >= target_d for other in acc):
                acc.append(code)
                if len(acc) >= num_codes:
                    break
        return acc

    accepted, achieved_target = [], None
    for target_d in range(try_from_distance, try_down_to - 1, -1):
        for _attempt in range(try_attempts_per_distance):
            order = rng.shuffled_range(1 << MARKER_PAYLOAD_BITS)
            accepted = one_attempt(order, target_d)
            if len(accepted) >= num_codes:
                break
        if len(accepted) >= num_codes:
            achieved_target = target_d
            break
    if achieved_target is None:
        raise RuntimeError("generate_marker_dictionary: could not find {} codes".format(num_codes))
    accepted = accepted[:num_codes]

    # Independent measurement pass (never trust the search target -- CLAUDE.md
    # paragraph 8, "never fabricate": always re-derive the number we print).
    min_dist = min(hamming(accepted[i], accepted[j])
                   for i in range(num_codes) for j in range(i + 1, num_codes))
    correction_capacity = (min_dist - 1) // 2
    return accepted, min_dist, correction_capacity


# ===========================================================================
# Small pinhole-camera / 3x3 linear-algebra helpers (pure Python, no numpy).
# ===========================================================================
def mat3_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation_matrix(yaw_rad, pitch_rad, roll_rad):
    """R = Rz(roll) * Rx(pitch) * Ry(yaw) -- an arbitrary but fixed Euler
    order used only to GENERATE varied ground-truth rotations; the pipeline
    never assumes this order (it only ever consumes the resulting 3x3 R)."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    Rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]
    Rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]
    Ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    return mat3_mul(Rz, mat3_mul(Rx, Ry))


def build_homography(R, t):
    """H = K * [r1 r2 t]: board-plane point (X,Y,0) meters -> pixel
    homogeneous coordinate. r1/r2 are R's first two COLUMNS (the board
    plane's own x/y axes expressed in the camera frame)."""
    r1 = (R[0][0], R[1][0], R[2][0])
    r2 = (R[0][1], R[1][1], R[2][1])
    H = [[0.0, 0.0, 0.0] for _ in range(3)]
    H[0][0] = FX_TRUE * r1[0] + CX_TRUE * r1[2]; H[0][1] = FX_TRUE * r2[0] + CX_TRUE * r2[2]; H[0][2] = FX_TRUE * t[0] + CX_TRUE * t[2]
    H[1][0] = FY_TRUE * r1[1] + CY_TRUE * r1[2]; H[1][1] = FY_TRUE * r2[1] + CY_TRUE * r2[2]; H[1][2] = FY_TRUE * t[1] + CY_TRUE * t[2]
    H[2][0] = r1[2];                              H[2][1] = r2[2];                              H[2][2] = t[2]
    return H


def apply_h(H, X, Y):
    w = H[2][0] * X + H[2][1] * Y + H[2][2]
    u = (H[0][0] * X + H[0][1] * Y + H[0][2]) / w
    v = (H[1][0] * X + H[1][1] * Y + H[1][2]) / w
    return u, v


def mat3_inv(M):
    a, b, c = M[0]; d, e, f = M[1]; g, h, i = M[2]
    A = e * i - f * h; B = -(d * i - f * g); C = d * h - e * g
    D = -(b * i - c * h); E = a * i - c * g; F = -(a * h - b * g)
    G = b * f - c * e; Hh = -(a * f - c * d); I = a * e - b * d
    det = a * A + b * B + c * C
    inv_det = 1.0 / det
    return [[A * inv_det, D * inv_det, G * inv_det],
            [B * inv_det, E * inv_det, Hh * inv_det],
            [C * inv_det, F * inv_det, I * inv_det]]


def solve_pose_for_image_center(R, target_u, target_v, z_depth):
    """Choose t so the board's CENTER point (board coords Xc,Yc,0) projects
    to (target_u, target_v) at approximately z_depth camera-frame depth --
    used only to place each synthetic view nicely inside the frame; the
    pipeline recovers R,t from the image, never reads this function."""
    Xc = SQUARES_X * SQUARE_SIZE_M * 0.5
    Yc = SQUARES_Y * SQUARE_SIZE_M * 0.5
    rc = [R[0][0] * Xc + R[0][1] * Yc, R[1][0] * Xc + R[1][1] * Yc, R[2][0] * Xc + R[2][1] * Yc]
    Z_cam = z_depth
    X_cam = (target_u - CX_TRUE) * Z_cam / FX_TRUE
    Y_cam = (target_v - CY_TRUE) * Z_cam / FY_TRUE
    return (X_cam - rc[0], Y_cam - rc[1], Z_cam - rc[2])


# ===========================================================================
# Canvas helpers: PGM I/O, gradient/blur/noise (identical spirit to 01.06's
# scene post-processing, reimplemented here for this project's own canvas
# resolution).
# ===========================================================================
def new_canvas(fill=BACKGROUND_GRAY):
    return [fill] * (IMG_W * IMG_H)


def apply_illumination_gradient(canvas, amplitude=0.10):
    """A gentler gradient than 01.06's (0.10 vs 0.18): this project samples
    marker cells at KNOWN homography-predicted locations rather than
    searching blind for tags, so a fixed mid-level decode threshold (see
    kernels.cuh kInkMidThreshold) only needs to tolerate a modest gradient
    -- a documented, honest scoping choice (README "Limitations")."""
    for y in range(IMG_H):
        row = y * IMG_W
        for x in range(IMG_W):
            factor = 1.0 + amplitude * (2.0 * (x / (IMG_W - 1)) - 1.0)
            canvas[row + x] *= factor


def gaussian_blur_5tap(canvas):
    weights = [1, 4, 6, 4, 1]
    wsum = 16
    tmp = [0.0] * (IMG_W * IMG_H)
    for y in range(IMG_H):
        row = y * IMG_W
        for x in range(IMG_W):
            s = 0.0
            for k in range(-2, 3):
                xx = min(max(x + k, 0), IMG_W - 1)
                s += canvas[row + xx] * weights[k + 2]
            tmp[row + x] = s / wsum
    out = [0.0] * (IMG_W * IMG_H)
    for y in range(IMG_H):
        for x in range(IMG_W):
            s = 0.0
            for k in range(-2, 3):
                yy = min(max(y + k, 0), IMG_H - 1)
                s += tmp[yy * IMG_W + x] * weights[k + 2]
            out[y * IMG_W + x] = s / wsum
    return out


def add_noise_and_quantize(canvas, rng, sigma=2.5):
    out = bytearray(IMG_W * IMG_H)
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
# board_ink_at — the "shader": given a board-plane point (X, Y) in meters,
# return the reflectance value BEFORE gradient/blur/noise. Handles all three
# layers in one place: (1) outside the board -> None (caller keeps
# background), (2) checkerboard squares, (3) the marker rendered inside each
# WHITE square's central MARKER_FILL_FRAC region. `codes` maps marker_id ->
# 9-bit payload actually rendered (normally the dictionary's true code; see
# make_view() for how a view can force one marker to render WRONG so the
# occlusion/negative-control honesty is exercised elsewhere, not here).
# ---------------------------------------------------------------------------
def board_ink_at(X, Y, codes):
    board_w = SQUARES_X * SQUARE_SIZE_M
    board_h = SQUARES_Y * SQUARE_SIZE_M
    if X < 0.0 or X >= board_w or Y < 0.0 or Y >= board_h:
        return None
    bx = min(int(X / SQUARE_SIZE_M), SQUARES_X - 1)
    by = min(int(Y / SQUARE_SIZE_M), SQUARES_Y - 1)

    if not is_white_square(bx, by):
        return INK_BLACK   # black square: solid ink, no marker

    # White square: background-white everywhere except the centered marker.
    lx = X - bx * SQUARE_SIZE_M
    ly = Y - by * SQUARE_SIZE_M
    half = MARKER_FILL_FRAC * SQUARE_SIZE_M * 0.5
    center = SQUARE_SIZE_M * 0.5
    if abs(lx - center) > half or abs(ly - center) > half:
        return INK_WHITE
    # Inside the marker's footprint: map to a MARKER_GRID_N x MARKER_GRID_N cell.
    cell = (2.0 * half) / MARKER_GRID_N
    mc = int((lx - (center - half)) / cell)
    mr = int((ly - (center - half)) / cell)
    mc = min(max(mc, 0), MARKER_GRID_N - 1)
    mr = min(max(mr, 0), MARKER_GRID_N - 1)
    marker_id = MARKER_ID_OF_SQUARE[(bx, by)]
    code = codes[marker_id]
    if marker_is_border_cell(mr, mc):
        return MARKER_INK_BLACK
    bit = (code >> marker_payload_bit_index(mr - 1, mc - 1)) & 1
    return MARKER_INK_BLACK if bit == 1 else MARKER_INK_WHITE


# ---------------------------------------------------------------------------
# render_board — inverse-mapping rasterizer (pixel -> board plane via H^-1),
# 2x2 sub-pixel supersample, same "ask every output pixel where its content
# comes from" strategy as 01.01's remap stage and 01.06's render_tag.
# ---------------------------------------------------------------------------
def render_board(canvas, H, codes):
    Hinv = mat3_inv(H)
    board_w = SQUARES_X * SQUARE_SIZE_M
    board_h = SQUARES_Y * SQUARE_SIZE_M
    corners_px = [apply_h(H, X, Y) for X, Y in ((0, 0), (board_w, 0), (board_w, board_h), (0, board_h))]
    xs = [p[0] for p in corners_px]; ys = [p[1] for p in corners_px]
    x_lo = max(0, int(math.floor(min(xs))) - 2)
    x_hi = min(IMG_W - 1, int(math.ceil(max(xs))) + 2)
    y_lo = max(0, int(math.floor(min(ys))) - 2)
    y_hi = min(IMG_H - 1, int(math.ceil(max(ys))) + 2)

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
                    ink = board_ink_at(Xt, Yt, codes)
                    if ink is not None:
                        acc += ink
                        hits += 1
            if hits > 0:
                frac = hits / 4.0
                canvas[y * IMG_W + x] = acc / hits * frac + canvas[y * IMG_W + x] * (1.0 - frac)


# ===========================================================================
# The 8-view pose table. Each entry: (name, yaw_deg, pitch_deg, roll_deg,
# depth_m). Depths/tilts were chosen (and checked below, see
# assert_corners_in_frame) so the full 7x5 corner grid projects inside the
# 320x240 frame with margin for every view except the deliberately-occluded
# one (view 7), which drops ~25% of corners UNDER a documented occluder
# rectangle, not off-frame.
# ===========================================================================
POSES = [
    ("frontal_near",    0.0,   0.0,   0.0,  0.300),
    ("frontal_far",     3.0,   2.0,   0.0,  0.335),
    ("yaw_left",        -5.0,  0.0,   0.0,  0.300),
    ("yaw_right",        5.0,  0.0,   0.0,  0.300),
    ("pitch_down",       0.0,  5.0,   0.0,  0.300),
    ("rolled",            2.0, -2.0, 26.0,  0.300),
    ("ambiguity_180",    0.0,  0.0, 180.0,  0.300),   # the 180-degree in-plane rotation
    ("occluded",         10.0, -6.0, 10.0,  0.300),   # + occluder rectangle, see below
]


def make_view(rng, view_index, name, yaw_deg, pitch_deg, roll_deg, depth_m, codes):
    R = rotation_matrix(math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
    t = solve_pose_for_image_center(R, CX_TRUE, CY_TRUE, depth_m)
    H = build_homography(R, t)

    canvas = new_canvas()
    render_board(canvas, H, codes)

    # ---- occluder (view 7 only): a flat rectangle, in IMAGE space, sized to
    # cover roughly one quadrant of the board's projected bounding box. ----
    occluder_rect = None
    if name == "occluded":
        board_w = SQUARES_X * SQUARE_SIZE_M
        board_h = SQUARES_Y * SQUARE_SIZE_M
        corners_px = [apply_h(H, X, Y) for X, Y in ((0, 0), (board_w, 0), (board_w, board_h), (0, board_h))]
        xs = [p[0] for p in corners_px]; ys = [p[1] for p in corners_px]
        bb_x0, bb_x1 = min(xs), max(xs)
        bb_y0, bb_y1 = min(ys), max(ys)
        # Bottom-right quadrant of the board's own bounding box: far from the
        # top-left seed the grid-ordering algorithm picks first (kernels.cuh
        # / THEORY.md "The algorithm"), so the seed and first row/col are
        # always found even under this occlusion.
        ox0 = bb_x0 + 0.55 * (bb_x1 - bb_x0)
        oy0 = bb_y0 + 0.55 * (bb_y1 - bb_y0)
        ox1 = bb_x1 + 6.0
        oy1 = bb_y1 + 6.0
        occluder_rect = (ox0, oy0, ox1, oy1)
        ix0, iy0 = max(0, int(ox0)), max(0, int(oy0))
        ix1, iy1 = min(IMG_W - 1, int(ox1)), min(IMG_H - 1, int(oy1))
        OCCLUDER_GRAY = 110.0   # a flat mid-gray patch -- e.g. a cable or mounting bracket (PRACTICE.md)
        for y in range(iy0, iy1 + 1):
            for x in range(ix0, ix1 + 1):
                canvas[y * IMG_W + x] = OCCLUDER_GRAY

    apply_illumination_gradient(canvas, amplitude=0.10)
    blurred = gaussian_blur_5tap(canvas)
    bytes_out = add_noise_and_quantize(blurred, rng, sigma=2.5)
    write_pgm(os.path.join(OUT_DIR, "view{:02d}.pgm".format(view_index)), IMG_W, IMG_H, bytes_out)

    # ---- ground truth: every inner corner's image position + visibility ---
    corner_rows = []
    for j in range(CORNERS_Y):
        for i in range(CORNERS_X):
            X, Y = corner_board_xy(i, j)
            u, v = apply_h(H, X, Y)
            visible = 1
            if occluder_rect is not None:
                ox0, oy0, ox1, oy1 = occluder_rect
                if ox0 - 3.0 <= u <= ox1 + 3.0 and oy0 - 3.0 <= v <= oy1 + 3.0:
                    visible = 0
            if u < -2.0 or u > IMG_W + 2.0 or v < -2.0 or v > IMG_H + 2.0:
                visible = 0
            corner_rows.append((view_index, i, j, u, v, visible))

    return R, t, corner_rows, occluder_rect


def assert_corners_in_frame(all_corner_rows):
    """Sanity check (never silently trust the pose table -- CLAUDE.md
    paragraph 8): every corner not flagged occluder-invisible must still
    land inside the frame with a small margin, for every view except the
    ones that legitimately clip (none, by design -- this is a hard check)."""
    bad = [r for r in all_corner_rows if r[5] == 1 and (r[3] < 1.0 or r[3] > IMG_W - 1.0 or r[4] < 1.0 or r[4] > IMG_H - 1.0)]
    if bad:
        raise RuntimeError("assert_corners_in_frame: {} visible corners fall outside the frame margin: {}"
                          .format(len(bad), bad[:5]))


# ===========================================================================
# Negative control: textured clutter, no board. Random gray blocks + a
# handful of filled disks, gradient + blur + noise -- same false-positive
# spirit as 01.06's scene_distractor, reimplemented at this project's canvas
# size and its own RNG draws.
# ===========================================================================
def make_negative_control(rng):
    canvas = new_canvas()
    for _ in range(14):
        w = int(rng.uniform_range(10, 40))
        h = int(rng.uniform_range(10, 40))
        x0 = int(rng.uniform_range(0, IMG_W - w))
        y0 = int(rng.uniform_range(0, IMG_H - h))
        level = rng.uniform_range(INK_BLACK, INK_WHITE)
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                canvas[y * IMG_W + x] = level
    for _ in range(6):
        cx = int(rng.uniform_range(0, IMG_W))
        cy = int(rng.uniform_range(0, IMG_H))
        r = int(rng.uniform_range(6, 22))
        level = rng.uniform_range(INK_BLACK, INK_WHITE)
        for y in range(max(0, cy - r), min(IMG_H, cy + r + 1)):
            for x in range(max(0, cx - r), min(IMG_W, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    canvas[y * IMG_W + x] = level
    apply_illumination_gradient(canvas, amplitude=0.10)
    blurred = gaussian_blur_5tap(canvas)
    bytes_out = add_noise_and_quantize(blurred, rng, sigma=2.5)
    write_pgm(os.path.join(OUT_DIR, "negative_control.pgm"), IMG_W, IMG_H, bytes_out)


# ===========================================================================
# main
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = XorShift32(SEED)

    print("[make_synthetic] generating {}-code marker dictionary (seed={})...".format(NUM_MARKER_CODES, SEED))
    codes, min_dist, capacity = generate_marker_dictionary(rng)
    print("[make_synthetic] marker dictionary: {} codes, measured min Hamming distance = {}, "
        "correction capacity floor((d-1)/2) = {}".format(len(codes), min_dist, capacity))

    dict_bin_path = os.path.join(OUT_DIR, "marker_dictionary.bin")
    with open(dict_bin_path, "wb") as f:
        f.write(struct.pack("<5i", len(codes), MARKER_PAYLOAD_BITS, MARKER_GRID_N, min_dist, capacity))
        f.write(struct.pack("<{}H".format(len(codes)), *codes))
    with open(os.path.join(OUT_DIR, "marker_dictionary.csv"), "w") as f:
        f.write("marker_id,square_bx,square_by,code_hex,code_binary_9bit\n")
        for mid, code in enumerate(codes):
            bx, by = SQUARE_OF_MARKER_ID[mid]
            f.write("{},{},{},0x{:03X},{}\n".format(mid, bx, by, code, format(code, "09b")))

    print("[make_synthetic] rendering {} views...".format(NUM_VIEWS))
    all_corner_rows = []
    pose_rows = []
    occluder_row = None
    for idx, (name, yaw, pitch, roll, depth) in enumerate(POSES):
        R, t, corner_rows, occ = make_view(rng, idx, name, yaw, pitch, roll, depth, codes)
        all_corner_rows.extend(corner_rows)
        pose_rows.append((idx, name, yaw, pitch, roll, depth, R, t))
        if occ is not None:
            occluder_row = (idx, name) + occ
        n_visible = sum(1 for r in corner_rows if r[5] == 1)
        print("  view{:02d} ({:16s}): yaw={:+6.1f} pitch={:+6.1f} roll={:+6.1f} depth={:.3f}m "
            "-> {}/{} corners visible".format(idx, name, yaw, pitch, roll, depth, n_visible, CORNERS_X * CORNERS_Y))

    assert_corners_in_frame(all_corner_rows)

    with open(os.path.join(OUT_DIR, "corners_truth.csv"), "w") as f:
        f.write("view_index,i,j,x_px,y_px,visible\n")
        for (v, i, j, u, vv, vis) in all_corner_rows:
            f.write("{},{},{},{:.4f},{:.4f},{}\n".format(v, i, j, u, vv, vis))

    with open(os.path.join(OUT_DIR, "poses_truth.csv"), "w") as f:
        f.write("view_index,name,yaw_deg,pitch_deg,roll_deg,depth_m,"
               "R00,R01,R02,R10,R11,R12,R20,R21,R22,t0,t1,t2\n")
        for (idx, name, yaw, pitch, roll, depth, R, t) in pose_rows:
            R_fields = ["{:.8f}".format(R[a][b]) for a in range(3) for b in range(3)]
            t_fields = ["{:.8f}".format(v) for v in t]
            f.write("{},{},{:.2f},{:.2f},{:.2f},{:.4f},{}\n".format(
                idx, name, yaw, pitch, roll, depth, ",".join(R_fields + t_fields)))

    with open(os.path.join(OUT_DIR, "intrinsics_truth.csv"), "w") as f:
        f.write("fx,fy,cx,cy\n")
        f.write("{:.6f},{:.6f},{:.6f},{:.6f}\n".format(FX_TRUE, FY_TRUE, CX_TRUE, CY_TRUE))

    with open(os.path.join(OUT_DIR, "occluder_truth.csv"), "w") as f:
        f.write("view_index,name,x0_px,y0_px,x1_px,y1_px\n")
        if occluder_row is not None:
            idx, name, ox0, oy0, ox1, oy1 = occluder_row
            f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(idx, name, ox0, oy0, ox1, oy1))

    print("[make_synthetic] rendering negative_control.pgm (textured clutter, no board)...")
    make_negative_control(rng)

    print("[make_synthetic] done. SHA-256 checksums:")
    names = ["marker_dictionary.bin", "marker_dictionary.csv", "corners_truth.csv", "poses_truth.csv",
            "intrinsics_truth.csv", "occluder_truth.csv", "negative_control.pgm"]
    names += ["view{:02d}.pgm".format(i) for i in range(NUM_VIEWS)]
    for name in names:
        path = os.path.join(OUT_DIR, name)
        print("  {}  {}".format(sha256_of(path), name))


if __name__ == "__main__":
    main()
