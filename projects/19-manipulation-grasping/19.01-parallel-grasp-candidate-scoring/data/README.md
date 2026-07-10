# Data — 19.01 Parallel grasp-candidate scoring: antipodal sampling over point clouds

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** (where one genuinely teaches more) are fetched by `../scripts/download_data.ps1`
  / `.sh` — idempotent, with source URL, expected size, and checksum documented below. **Respect every
  license**; registration-gated or no-redistribution datasets are pointed at, never mirrored.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

**Why synthetic, and not a public grasp dataset (Cornell Grasp, YCB, GraspNet-1Billion, ...)?** This
project's whole verification strategy (README "Expected output", `THEORY.md` "How we verify
correctness") depends on objects whose good grasps are known **geometrically**, in closed form —
"the box's opposite faces are 60 mm and 40 mm apart, antipodal, and gripper-feasible; the 100 mm axis
is not." No real scanned object cloud carries that closed-form ground truth; a public dataset would
let the demo *look* plausible without ever being *checkable*. Synthetic analytic shapes are the
correct tool here, not a shortcut (CLAUDE.md §8's synthetic-first default, taken to its logical
conclusion for this project).

| Property | Value |
|----------|-------|
| Kind | Synthetic (3 analytic objects: box, cylinder, sphere) |
| Generator | `../scripts/make_synthetic.py`, one seeded `random.Random(42)` stream, order box → cylinder → sphere |
| License | Synthetic — repo MIT license applies (no third-party data, no redistribution question) |
| Size (committed) | `box_cloud.bin` 72,008 B · `cylinder_cloud.bin` 108,008 B · `sphere_cloud.bin` 84,008 B · `objects_meta.csv` 803 B — **260 KiB total** |
| Checksum (SHA-256) | `box_cloud.bin` `a7bba23eee79e0816af5989e9c8be664d01d26e8584c18ba070a2e585a846aa4`<br>`cylinder_cloud.bin` `8ace9aef33acafd33cc2eb214d23caa1158c9f576898a6a13ede8a0e53524347`<br>`sphere_cloud.bin` `9849c05f05571f1373406eda7158eed66010b7cd8e1a69ea493acbcbb05d5e39`<br>`objects_meta.csv` `362f03fec7e306405c1c868984ec14fb0b87c97afd1a91d04874f355f5bb08a2` |
| Regenerate with | `python scripts/make_synthetic.py --seed 42` (the default; byte-identical output on any machine) |

### The three objects

| Object | Geometry (ground truth) | Points | Known-good grasps |
|--------|--------------------------|--------|--------------------|
| **box** | axis-aligned, 60 × 40 × 100 mm full extents, centered at the origin | 6,000 | opposite-face pairs: 60 mm axis, 40 mm axis — **both** gripper-feasible (stroke 10–90 mm). The 100 mm axis is geometrically antipodal but **too wide for the gripper** — a deliberate, checkable negative case (see `src/main.cu`'s box width-gate CHECK line). |
| **cylinder** | radius 25 mm, height 120 mm, **lateral surface only** (no end-cap points — see below) | 9,000 | any diametral pair on the lateral surface (50 mm), at any height and any angle around the axis |
| **sphere** | radius 30 mm | 7,000 | any diametral pair through the center (60 mm) — there is no "wrong axis" on a sphere |

**Why the cylinder has no end-cap points.** A depth camera looking at a can from the side sees the
lateral surface, not the caps; this project's grasp geometry (README "The algorithm in brief",
`THEORY.md`) only needs the lateral surface's diametral pairs, so the caps would add points that
teach nothing about *this* project's antipodal search (and would, if included carelessly, let a
top-to-bottom "grasp" appear that no real two-finger gripper could execute along a smooth cylinder's
axis). Leaving them out is a scoping decision, not an oversight — see README "Limitations & honesty".

### Fields / format

**`<name>_cloud.bin`** — this project's point-cloud binary format (distinct from project 02.06's
"PC01" format — same *layout* convention, but no cross-project file coupling, CLAUDE.md §4
self-containment rule):

| Offset | Type | Meaning |
|--------|------|---------|
| 0 | 4 bytes, ASCII | magic `"GC01"` |
| 4 | `uint32`, little-endian | point count `N` |
| 8 | `N × 3 × float32`, little-endian | interleaved `x0,y0,z0, x1,y1,z1, ...` — **meters**, object-local frame (origin at the object's centroid/geometric center) |

**`objects_meta.csv`** — one row per object:

| Column | Type | Meaning |
|--------|------|---------|
| `name` | string | `box` \| `cylinder` \| `sphere` |
| `file` | string | the matching `<name>_cloud.bin`, relative to this directory |
| `n_points` | int | point count (must match the `.bin` file's own count — `src/main.cu` checks this) |
| `shape` | string | same as `name`, used by `src/main.cu` to select the analytic gate formula |
| `param_a_m`, `param_b_m`, `param_c_m` | float, meters | shape-dependent: **box** = full extents `dim_x,dim_y,dim_z`; **cylinder** = `radius,height,0`; **sphere** = `radius,0,0` |
| `gripper_w_min_m`, `gripper_w_max_m` | float, meters | the modeled parallel-jaw gripper's stroke range (10–90 mm here — `PRACTICE.md` §2 dates and caveats the illustrative hardware this mirrors) |
| `friction_mu` | float, unitless | Coulomb friction coefficient used by the friction-cone gate (0.5 here — `THEORY.md` "The math" derives the gate from it) |

### Noise model

Every point gets two independent zero-mean Gaussian offsets, applied in `scripts/make_synthetic.py`:

- **axial** (along the object's TRUE surface normal at that point) — σ = 0.3 mm, simulating a depth
  camera's range noise.
- **tangential** (in the local surface plane) — σ = 0.15 mm, simulating lateral pixel jitter.

Both are far below the point spacing on every committed cloud (box ≈ 2.0 mm, cylinder ≈ 1.5 mm,
sphere ≈ 1.4 mm — area divided by point count, square-rooted), so the PCA normal estimator
(`src/kernels.cu`'s `estimate_normals_kernel`, k = 16 neighbors) recovers each true surface normal to
within a fraction of a degree (measured: worst GPU-vs-CPU normal deviation 0.034° on the box — see
`demo/README.md`; that number is GPU-vs-CPU agreement, not noise-induced error, but it is the same
order of magnitude the noise model is sized to keep negligible).

> **Placeholder status: none.** This is the project's real, final sample data — not a toolchain
> smoke test. `src/main.cu` reads exactly these three files at startup; there is no in-memory
> fallback generator.
