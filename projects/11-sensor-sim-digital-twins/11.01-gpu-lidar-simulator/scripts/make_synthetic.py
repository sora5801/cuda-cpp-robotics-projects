#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 11.01 (GPU LiDAR
simulator: BVH raycasting + beam divergence, intensity, dropout noise).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
A LiDAR simulator's "dataset" is a SCENE plus a SENSOR SPEC, not recordings:
this script builds a small triangle-mesh warehouse (the world the simulated
beams bounce off) and the sensor configuration/pose that will scan it. Both
are pure geometry and constants — no RNG anywhere in this script, so the
output is byte-identical on every machine, every run (CLAUDE.md paragraph 12
determinism). All sensor NOISE (dropout, range jitter) is generated inside
the demo itself from a documented fixed seed (see ../src/main.cu and
../src/kernels.cu) — this script only builds the physical scene and the
sensor's *configuration*, not any simulated measurement.

What it writes (all under ../data/sample/, tiny, plain text)
--------------------------------------------------------------
  warehouse_scene.obj  — the triangle mesh (Wavefront OBJ, `v`/`f` lines
                          only — this repo hand-rolls its own minimal OBJ
                          reader in main.cu rather than link a mesh library,
                          CLAUDE.md paragraph 5's "no black boxes" stance).
  materials.csv         — per-triangle-INDEX-RANGE material assignment (a
                          material table + contiguous [start,end] ranges,
                          not a full per-vertex/per-face OBJ+MTL parser —
                          simple, hand-rolled, and exactly matches the face
                          emission order below, which this script controls).
  sensor_config.csv      — the spinning-LiDAR scan pattern + the three noise
                          models the catalog bullet names (beam divergence,
                          intensity radiometry, dropout).
  sensor_poses.csv       — where the sensor sits in the scene (T_world_sensor,
                          SI units, repo quaternion order w,x,y,z).

The scene: a warehouse-like room
---------------------------------
A 24 m x 24 m concrete floor (subdivided into a 32x32 grid so the mesh has
enough triangles to make a BVH worth building — see ../THEORY.md "The
algorithm" for why a flat single quad would defeat the point of this
project), four 3 m perimeter walls, six steel shelving racks along one side,
and eight cardboard crates scattered on the open floor on the other side.
The +x corridor near the room's center (y ~ 0) is kept deliberately CLEAR of
obstacles: ../src/main.cu's analytic verification gates fire test beams down
that corridor and need a guaranteed, obstruction-free line of sight to the
floor (see ../THEORY.md "How we verify correctness").

Usage
-----
    python make_synthetic.py                 # writes the committed sample
    python make_synthetic.py --out-dir X      # experiments; do not commit
"""

import argparse
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene dimensions (meters). Every constant here is ALSO documented in
# ../data/README.md and cited by ../src/kernels.cuh / ../src/main.cu comments
# where the analytic verification gates depend on specific scene geometry
# (the clear +x corridor, the flat z=0 floor). Change one place, update both.
# ---------------------------------------------------------------------------
GROUND_HALF = 12.0          # floor spans [-12, +12] m in x and y -> 24x24 m
GROUND_DIV = 32              # 32x32 grid cells -> 32*32*2 = 2048 triangles
WALL_HEIGHT = 3.0            # m
WALL_THICK = 0.3             # m
SHELF_SIZE = (1.2, 0.6, 2.2)  # (x, y, z) extents, meters
SHELF_Y = -8.0                # all shelves sit on this y-line (south side)
SHELF_XS = (-10.0, -6.0, -2.0, 2.0, 6.0, 10.0)
CRATE_SIZE = 0.6              # cube edge, meters
# Crates on the north side (y > 0): kept off the y=0 (+x) corridor and off
# the shelving side so the analytic-gate corridor (see module docstring)
# stays clear from the origin out to the east wall.
CRATE_XY = [
    (-9.0, 8.0), (-5.0, 9.0), (-1.0, 7.0), (3.0, 8.0),
    (7.0, 9.0), (10.0, 6.0), (-7.0, 5.0), (5.0, 5.0),
]


class MeshBuilder:
    """Accumulates OBJ vertices/faces and the parallel material-range table.

    OBJ is 1-indexed (Wavefront convention); this class hides that off-by-one
    so the emit_* helpers below can just think in 0-indexed triangle counts.
    Winding order is kept CCW-from-outside as good practice, but the C++
    loader/kernel do NOT depend on it: intensity uses |cos(incidence)|, which
    is sign-independent (documented in ../src/kernels.cuh) — a deliberate
    simplification that makes this generator's geometry forgiving to get
    exactly right.
    """

    def __init__(self):
        self.verts: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []   # 0-indexed vertex ids
        self.material_ranges: list[tuple[int, int, int]] = []  # (start,end,mat_id) inclusive, 0-indexed triangle

    def add_vertex(self, x: float, y: float, z: float) -> int:
        self.verts.append((x, y, z))
        return len(self.verts) - 1   # 0-indexed id

    def add_tri(self, a: int, b: int, c: int) -> None:
        self.faces.append((a, b, c))

    def begin_material_span(self) -> int:
        return len(self.faces)   # first triangle index of the span about to be emitted

    def end_material_span(self, start_tri: int, material_id: int) -> None:
        end_tri = len(self.faces) - 1
        if end_tri >= start_tri:   # guard against a degenerate empty span
            self.material_ranges.append((start_tri, end_tri, material_id))

    def emit_ground(self, half: float, divisions: int) -> None:
        """A `divisions` x `divisions` grid of quads (2 tris each) at z=0,
        spanning [-half,half] in x and y. The grid gives the mesh enough
        SPATIAL VARIATION for a median-split BVH's axis choice to matter
        (a single 2-triangle quad would make every BVH decision trivial)."""
        step = (2.0 * half) / divisions
        # (divisions+1)^2 grid vertices, row-major, y-major then x.
        base = len(self.verts)
        for iy in range(divisions + 1):
            y = -half + iy * step
            for ix in range(divisions + 1):
                x = -half + ix * step
                self.add_vertex(x, y, 0.0)
        for iy in range(divisions):
            for ix in range(divisions):
                # 4 corners of this cell, CCW seen from +z (above).
                v00 = base + iy * (divisions + 1) + ix
                v10 = base + iy * (divisions + 1) + (ix + 1)
                v11 = base + (iy + 1) * (divisions + 1) + (ix + 1)
                v01 = base + (iy + 1) * (divisions + 1) + ix
                self.add_tri(v00, v10, v11)
                self.add_tri(v00, v11, v01)

    def emit_box(self, cx: float, cy: float, z0: float, z1: float,
                 sx: float, sy: float) -> None:
        """An axis-aligned box centered at (cx,cy) in x/y, spanning z0..z1,
        with full x/y extents sx (x) and sy (y). Emits 8 vertices + 12
        triangles (2 per face x 6 faces) — the standard box triangulation."""
        x0, x1 = cx - sx / 2.0, cx + sx / 2.0
        y0, y1 = cy - sy / 2.0, cy + sy / 2.0
        # 8 corners, indexed by (xi, yi, zi) each in {0,1}.
        idx = {}
        for xi, x in enumerate((x0, x1)):
            for yi, y in enumerate((y0, y1)):
                for zi, z in enumerate((z0, z1)):
                    idx[(xi, yi, zi)] = self.add_vertex(x, y, z)

        def quad(a, b, c, d):
            self.add_tri(a, b, c)
            self.add_tri(a, c, d)

        # -x face (x=x0), +x face (x=x1)
        quad(idx[(0, 0, 0)], idx[(0, 0, 1)], idx[(0, 1, 1)], idx[(0, 1, 0)])
        quad(idx[(1, 0, 0)], idx[(1, 1, 0)], idx[(1, 1, 1)], idx[(1, 0, 1)])
        # -y face (y=y0), +y face (y=y1)
        quad(idx[(0, 0, 0)], idx[(1, 0, 0)], idx[(1, 0, 1)], idx[(0, 0, 1)])
        quad(idx[(0, 1, 0)], idx[(0, 1, 1)], idx[(1, 1, 1)], idx[(1, 1, 0)])
        # -z face (bottom, z=z0), +z face (top, z=z1)
        quad(idx[(0, 0, 0)], idx[(0, 1, 0)], idx[(1, 1, 0)], idx[(1, 0, 0)])
        quad(idx[(0, 0, 1)], idx[(1, 0, 1)], idx[(1, 1, 1)], idx[(0, 1, 1)])

    def write_obj(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# warehouse_scene.obj - SYNTHETIC triangle mesh for project 11.01\n")
            f.write("# generated by scripts/make_synthetic.py - NO RNG (pure deterministic geometry)\n")
            f.write(f"# {len(self.verts)} vertices, {len(self.faces)} triangles\n")
            f.write("# face order: ground grid, then 4 walls, then 6 shelves, then 8 crates\n")
            f.write("# (materials.csv's RANGE rows depend on this exact order - do not reorder)\n")
            f.write("# units: meters, world frame (right-handed, z-up, CLAUDE.md paragraph 12)\n")
            for (x, y, z) in self.verts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for (a, b, c) in self.faces:
                # OBJ is 1-indexed.
                f.write(f"f {a + 1} {b + 1} {c + 1}\n")

    def write_materials(self, path: Path, materials: list[tuple[float, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# materials.csv - SYNTHETIC per-triangle-range material assignment, project 11.01\n")
            f.write("# generated by scripts/make_synthetic.py\n")
            f.write("# MATERIAL,id,albedo,name          : one row per material, id = 0..N-1 in order\n")
            f.write("# RANGE,tri_start,tri_end,material_id : inclusive 0-indexed triangle range (face\n")
            f.write("#   order in warehouse_scene.obj) assigned to material_id. Every triangle must be\n")
            f.write("#   covered by exactly one RANGE row (the loader in ../src/main.cu asserts this).\n")
            f.write("# albedo: dimensionless Lambertian reflectance in (0,1] - illustrative values,\n")
            f.write("#   see ../THEORY.md 'The problem' for the radiometry these feed.\n")
            for i, (albedo, name) in enumerate(materials):
                f.write(f"MATERIAL,{i},{albedo:.4f},{name}\n")
            for (start, end, mat_id) in self.material_ranges:
                f.write(f"RANGE,{start},{end},{mat_id}\n")


def build_scene() -> MeshBuilder:
    mb = MeshBuilder()

    # --- ground: material 0 (concrete floor) --------------------------------
    span = mb.begin_material_span()
    mb.emit_ground(GROUND_HALF, GROUND_DIV)
    mb.end_material_span(span, 0)

    # --- perimeter walls: material 1 (painted steel) ------------------------
    span = mb.begin_material_span()
    g = GROUND_HALF
    # North/south walls run along x, centered on y = +-g; east/west along y.
    mb.emit_box(0.0, g, 0.0, WALL_HEIGHT, 2.0 * g + WALL_THICK, WALL_THICK)   # +y wall
    mb.emit_box(0.0, -g, 0.0, WALL_HEIGHT, 2.0 * g + WALL_THICK, WALL_THICK)  # -y wall
    mb.emit_box(g, 0.0, 0.0, WALL_HEIGHT, WALL_THICK, 2.0 * g + WALL_THICK)   # +x wall
    mb.emit_box(-g, 0.0, 0.0, WALL_HEIGHT, WALL_THICK, 2.0 * g + WALL_THICK)  # -x wall
    mb.end_material_span(span, 1)

    # --- shelving racks: material 2 (steel shelving) ------------------------
    span = mb.begin_material_span()
    sx, sy, sz = SHELF_SIZE
    for x in SHELF_XS:
        mb.emit_box(x, SHELF_Y, 0.0, sz, sx, sy)
    mb.end_material_span(span, 2)

    # --- crates: material 3 (cardboard) --------------------------------------
    span = mb.begin_material_span()
    for (cx, cy) in CRATE_XY:
        mb.emit_box(cx, cy, 0.0, CRATE_SIZE, CRATE_SIZE, CRATE_SIZE)
    mb.end_material_span(span, 3)

    return mb


def write_sensor_config(path: Path) -> None:
    """The scan pattern + the three per-beam effect models the catalog bullet
    names. All values are illustrative, order-of-magnitude-real (see
    ../THEORY.md for citations), never a specific product's exact datasheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# sensor_config.csv - SYNTHETIC spinning-LiDAR configuration, project 11.01",
        "# generated by scripts/make_synthetic.py - no RNG (a config is constants)",
        "# angles in degrees/milliradians as labeled; converted to radians by the loader",
        "CHANNELS,32",
        "AZIMUTH_STEPS,1024",
        "ELEVATION_MIN_DEG,-15.0",
        "ELEVATION_MAX_DEG,15.0",
        "AZIMUTH_START_DEG,0.0",
        "RANGE_MIN_M,0.30",
        "RANGE_MAX_M,40.0",
        "DIVERGENCE_HALF_ANGLE_MRAD,1.5",
        "SUBRAY_COUNT,4",
        "INTENSITY_GAIN,5.0",
        "RANGE_NOISE_BASE_M,0.015",
        "RANGE_NOISE_PER_M,0.001",
        "DROPOUT_BASE,0.02",
        "DROPOUT_RANGE_COEFF,0.10",
        "DROPOUT_INCIDENCE_COEFF,0.35",
        "SEED,42",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_sensor_poses(path: Path) -> None:
    """T_world_sensor for each frame: translation (m) + unit quaternion
    (repo order w,x,y,z, CLAUDE.md paragraph 12). Frame 0 is an AMR-style
    LiDAR mounted level at 1.5 m, centered in the room, facing +x (identity
    orientation) - the pose the demo's main frame and the analytic
    verification gates both use (their geometry is derived from this exact
    position; see ../src/main.cu)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# sensor_poses.csv - SYNTHETIC sensor pose(s), project 11.01",
        "# generated by scripts/make_synthetic.py - no RNG (a pose is constants)",
        "# POSE,frame_idx,x,y,z,qw,qx,qy,qz : T_world_sensor (meters; unit quaternion w,x,y,z)",
        "# v1 uses only frame_idx 0 (see ../src/main.cu); the format allows more for future frames.",
        "POSE,0,0.0,0.0,1.5,1.0,0.0,0.0,0.0",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default: ../data/sample)")
    args = ap.parse_args()

    mb = build_scene()
    materials = [
        (0.30, "concrete_floor"),
        (0.55, "painted_steel_wall"),
        (0.65, "steel_shelving"),
        (0.80, "cardboard_crate"),
    ]

    obj_path = args.out_dir / "warehouse_scene.obj"
    mat_path = args.out_dir / "materials.csv"
    cfg_path = args.out_dir / "sensor_config.csv"
    pose_path = args.out_dir / "sensor_poses.csv"

    mb.write_obj(obj_path)
    mb.write_materials(mat_path, materials)
    write_sensor_config(cfg_path)
    write_sensor_poses(pose_path)

    n_tri = len(mb.faces)
    print(f"[make_synthetic] wrote {obj_path} ({len(mb.verts)} vertices, {n_tri} triangles, labeled SYNTHETIC)")
    print(f"[make_synthetic] wrote {mat_path} ({len(materials)} materials, {len(mb.material_ranges)} ranges)")
    print(f"[make_synthetic] wrote {cfg_path}")
    print(f"[make_synthetic] wrote {pose_path}")

    # Sanity check: every triangle must be covered by exactly one material
    # range, contiguously and without gaps - the C++ loader assumes this and
    # aborts loudly if it does not hold (fail here too, at generation time,
    # so a scene-authoring bug is caught before it is ever committed).
    covered = 0
    expected_next = 0
    for (start, end, _mat) in mb.material_ranges:
        if start != expected_next:
            print(f"[make_synthetic] ERROR: material range gap/overlap at triangle {start} "
                  f"(expected {expected_next})", file=sys.stderr)
            return 1
        covered += end - start + 1
        expected_next = end + 1
    if covered != n_tri:
        print(f"[make_synthetic] ERROR: material ranges cover {covered} triangles, "
              f"mesh has {n_tri}", file=sys.stderr)
        return 1
    print(f"[make_synthetic] material-range coverage OK: {covered}/{n_tri} triangles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
