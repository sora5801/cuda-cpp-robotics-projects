# Data — 05.01 TSDF fusion (KinectFusion clone) + marching-cubes mesh extraction

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** (where one genuinely teaches more) are fetched by `../scripts/download_data.ps1`
  / `.sh` — idempotent, with source URL, expected size, and checksum documented below. **Respect every
  license**; registration-gated or no-redistribution datasets (KITTI, nuScenes) are pointed at, never
  mirrored.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** camera path (the task definition — no RNG involved; a circular orbit + look-at is closed-form trigonometry) |
| File | `sample/camera_path.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: 24 frames, radius 2.0 m, height 1.2 m) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 2408 bytes (~2.4 KiB) |
| Checksum (SHA-256) | `98b98406eb9c2e2d48b9899719ed14d77ac5afaa78739385b161efd96a96ede9` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no randomness) |

**What is *not* in this file, and why:** the 24 depth *images* fused every demo run. Unlike most
sensor-fusion projects in this repo, a depth camera's raw output is not what makes this project's data
"the task" — the **scene** does, and the scene (a sphere over a ground plane, `../src/kernels.cuh`) has
a closed-form signed distance function. Depth frames are rendered **inside the demo, every run**, by
exact closed-form ray casting of that scene from each committed pose (`render_depth()` in
`../src/main.cu`) — no noise, no compression, no storage cost, and (the actual point) a fusion result
that can be checked against the scene's *exact* SDF instead of a recorded fixture. This is the same
synthetic-first spirit as every other project's data (CLAUDE.md §8), taken one step further: even the
"sensor" is synthesized on demand rather than pre-rendered and stored.

### Fields / format

Plain-text CSV; `#` lines are comments. Two row types (loader: `load_sample()` in
[`../src/main.cu`](../src/main.cu); layout authority: [`../src/kernels.cuh`](../src/kernels.cuh)):

**`CAM,width,height,fx,fy,cx,cy`** — pinhole camera intrinsics, one row, pixels:

| Field | Units | Meaning |
|-------|-------|---------|
| `width`, `height` | px | image size (committed: 160 × 120) |
| `fx`, `fy` | px | focal lengths (committed: 120, 120 — chosen so one pixel spans roughly one voxel edge at scene range, matching sampling densities; THEORY.md §numerics) |
| `cx`, `cy` | px | principal point (committed: 79.5, 59.5 — the image center) |

Camera **optical** frame convention (stated explicitly because it differs from the repo's default body
convention — CLAUDE.md §12): **x-right, y-down, z-forward**. Pixel `(u,v)` projects as
`u = fx·x/z + cx`, `v = fy·y/z + cy`.

**`POSE,i,tx,ty,tz,qw,qx,qy,qz`** — one row per frame, in file order (committed: 24 rows, `i` = 0..23):

| Field | Units | Meaning |
|-------|-------|---------|
| `i` | — | frame index (informational; poses are consumed in file order) |
| `tx`, `ty`, `tz` | m | camera position in the **world** frame (world is right-handed, z up) |
| `qw`, `qx`, `qy`, `qz` | — | orientation quaternion, **scalar-first `(w,x,y,z)`** (CLAUDE.md §12), unit norm |

Each `POSE` row is `T_world_cam` — "camera expressed in world" (CLAUDE.md §12 `T_parent_child`
notation): `p_world = T_world_cam · p_cam`. The integration kernel wants the inverse, `T_cam_world`;
`main.cu` inverts once per frame on the host (`invert_pose()`) rather than repeating the inversion per
voxel.

The committed path: 24 poses evenly spaced around a circle of radius 2.0 m at height 1.2 m, each aimed
(via a look-at construction) at a point between the ground plane and the sphere — chosen so every part
of the sphere and the surrounding plane is observed from multiple directions. `../scripts/make_synthetic.py`
documents the look-at quaternion derivation in full.

Everything else the demo consumes is generated at run time: the 24 depth frames (closed-form ray
casting, no storage, no seed needed — see above) and the analytic scene's ground-truth SDF (a pure
function of the scene constants in `../src/kernels.cuh`, evaluated on demand by `scene_sdf()` in
`../src/main.cu`). The loader is strict: unknown row labels, short rows, or a missing `CAM`/`POSE` data
abort the demo.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
