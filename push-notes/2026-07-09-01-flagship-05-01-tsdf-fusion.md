# Push note — 2026-07-09-01: flagship 05.01 tsdf fusion

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 05.01 — **TSDF fusion (KinectFusion-style) + marching-cubes mesh extraction** — is done:
24 synthetic depth frames of an analytic sphere-over-plane scene fused into a 128³ voxel volume on
the GPU, then extracted as a 54,822-triangle OBJ mesh. Two things make this push worth reading
even if you skip the code. First, the scene is *analytic*, so the fused TSDF is checked against a
closed-form ground-truth SDF — real mathematics, not just a CPU twin. Second, the finisher agent
caught the prior worker's **fabricated placeholder verification bounds** (never measured), refused
to keep them, measured reality, explained the physics of the error tail it found — the classic
grazing-incidence bias of projective TSDF, concentrated on the sphere's lower belly where the
constant-elevation camera orbit only ever sees the surface edge-on — and set honest measured
bounds with the mechanism documented in THEORY.md. That is the repository's no-fabrication rule
(§8) enforced in practice.

## What changed

- **[projects/05-slam-mapping-localization/05.01-tsdf-fusion-marching-cubes-mesh-extraction/](../projects/05-slam-mapping-localization/05.01-tsdf-fusion-marching-cubes-mesh-extraction/)** —
  complete: TSDF integration kernel (voxel-parallel projective update), marching-cubes kernel
  (constant-memory edge/tri tables, atomic append), in-code analytic depth renderer, CPU twin,
  ground-truth SDF checks, mesh + TSDF-slice artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 05.01 → `done` (**6/505**).

## New projects (didactic blurbs)

**05.01 — TSDF fusion + marching cubes** (★ beginner, domain 05, flagship). Teaches how dense
mapping actually works: each depth pixel carves truncated signed distance into a voxel grid, and
weighted averaging across views turns noisy frames into a clean implicit surface — then marching
cubes turns the implicit surface into triangles. GPU patterns: voxel-parallel integration,
`__constant__` lookup tables, and the atomic-append output pattern (the first project that must
*count* its variable-size output). The single most interesting thing to look at:
`demo/out/tsdf_slice.pgm` — a cross-section of the volume showing the sphere, the plane, and the
occlusion shadow beneath the sphere, exactly as the theory predicts.

## How to build & run

```powershell
projects\05-slam-mapping-localization\05.01-tsdf-fusion-marching-cubes-mesh-extraction\demo\run_demo.ps1
# then open demo\out\mesh.obj in any 3-D viewer and demo\out\tsdf_slice.pgm in an image viewer
```

## What to study here

Project `README.md` → `THEORY.md` (depth-sensor physics → SDF/TSDF math → projective-vs-true SDF
honesty → the marching-cubes construction) → `src/kernels.cu`. Then look at the ground-truth
`[info]` lines in the demo output and find the grazing-incidence story in THEORY §numerics — the
error tail is a *feature* of this dataset design, taught on purpose. First exercise: add a second
camera ring at a lower elevation and watch the belly error collapse.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the finisher's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 9 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** bit-identical TSDF integration on the 4-frame subset (worst deviation
  exactly 0.0 — the fmaf-everywhere determinism contract held).
- **Analytic ground truth:** surface-shell mean error 1.47e-02 m (max 1.14e-01 m in the documented
  grazing-incidence tail, 0.65% of shell voxels); bounds set to measured values with headroom and
  the mechanism explained — not aspirational numbers.
- **Mesh:** 54,822 triangles, GPU count == CPU recount exactly; vertices on the analytic surface
  within 1.04e-02 m.
- Timings (teaching artifacts): integration ≈ 0.08–0.13 ms/frame GPU vs ≈ 61 ms CPU; marching
  cubes ≈ 1.9 ms over 127³ cells.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Known poses (no tracking — 02.06's job), analytic scene (no real sensor noise model beyond the
  documented additive noise), single 128³ dense volume (voxel hashing is 05.02's job) — all
  documented in README §Limitations.
- 22.01 (swarm) finisher is running; 31.01 (HJ reachability) queued; their interrupted partial
  `src/` remains in-tree, marked `in-progress` in STATUS (disclosed in push-note 2026-07-09-00).

## Next push preview

22.01 100k-agent swarm simulator (flocking + pheromone stigmergy), then 31.01 Hamilton–Jacobi
reachability with its analytic double-integrator ground truth.
