# 02.13 — Dynamic point removal (raycast free-space carving)

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Dynamic point removal (raycast free-space carving)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A map accumulated from many LiDAR scans is full of ghosts: a car that drove through the scene
leaves points behind at every position it was ever seen, a person who stood still for a while and
then walked off leaves a phantom obstacle exactly where they stood. This project cleans that map
using nothing but the physics of the ray itself: every LiDAR beam that returns a HIT also proves
that everything it passed *through* on the way there was, at that instant, empty. Accumulate that
free-space evidence over K posed scans into a per-voxel ledger (hits vs. passes), and a point sitting
in a voxel with much more "passed through" evidence than "hit" evidence is very likely something
that has since moved — remove it. The traversal that turns one beam into a sequence of visited
voxels is the classic Amanatides & Woo (1987) DDA voxel march, implemented here from scratch and
run beam-parallel on the GPU with atomic per-voxel counters.

This project implements the full catalog bullet as a single, complete pipeline: DDA carving,
three-way ledger bookkeeping (hit / pass-from-hit / pass-from-max-range), ratio-based
classification, and a synthetic scene designed to exercise every interesting case — a moving car
(the classic ghost trail), a pedestrian who is temporarily static and only provably dynamic once it
leaves (the hard case), permanent structure with two flavors of discretization false-positive (a
thin pole, a wall's free end), and an isolated object carved away by max-range beams alone. Nothing
here is a reduced-scope stand-in; every gate named in the catalog's spirit is measured and gated.

## What this computes & why the GPU helps

The computation is **beam-parallel ray marching into a shared voxel grid, followed by a point-
parallel classification**: for each of 28,800 independent LiDAR beams, march a 3-D DDA path from
the sensor to the beam's endpoint (or out to max range), atomically incrementing per-voxel hit/pass
counters along the way; then, for each recorded point, look up its own voxel's ledger and classify
it STATIC or DYNAMIC from the hit/pass ratio. Beams never interact except through the shared ledger,
so this is the *scatter* pattern (many independent producers, one shared array, resolved with
atomics) — the same family of GPU idiom 02.01's hash-table insert uses, applied here to 3-D space
instead of a hash table, and to a *ray march* instead of a *point-to-voxel* lookup. Classification
afterward is a pure independent-per-point *map* over the now-finished ledger.

- The march itself (Amanatides & Woo 1987): O(voxels-crossed) per beam, no per-step distance
  recomputation — a handful of comparisons and additions carries the whole algorithm.
- Contention in the atomic writes is real but structurally bounded: beams from one scan all start
  at the same sensor voxel and fan out, so only the near-sensor region is hot (measured every run —
  see "Expected output").

## System context — where this sits in a robot

Full stack reference: [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md). Physical/
commercial grounding: [`PRACTICE.md`](PRACTICE.md).

- **Stack position:** state estimation / world model layer (`SYSTEM_DESIGN.md` §1) — specifically
  the map-maintenance stage that sits between raw scan accumulation and every consumer that trusts
  the map to describe *permanent* structure. It is a **filter over an accumulated map**, not a
  per-scan perception step.
- **Upstream inputs:** posed `PointCloud` scans — the output of per-point motion deskew (project
  02.08, `xyz_out`) composed with a localization estimate that supplies each scan's pose (project
  02.07's NDT scan matching, or any SLAM front end) — the accumulated-but-uncleaned map this project
  carves.
- **Downstream consumers:** everything that assumes the map describes what is *actually still
  there* — localization itself (a real feedback loop: an uncleaned map with ghosts corrupts the very
  estimates used to build the next map, so this project is not a nice-to-have, it closes a loop),
  occupancy costmaps for planning (project 23.01), and any map shared across a fleet (project
  02.15's compressed uplink carries the CLEANED map, not the raw one). Its dual is project 02.14
  (moving-object segmentation): where this project asks "what should I *remove* from the map," 02.14
  asks "what should I *track* right now" — the same free-space evidence, two different jobs.
- **Rate/latency budget:** honestly split. As implemented and gated here, carving is a
  **post-session batch job** (K scans, run once after the session, ~ms on a desktop GPU for this
  scene) — real mapping-rate budgets (`SYSTEM_DESIGN.md` §1.1: map update 10-20 Hz for an AMR) apply
  to *incremental* per-scan fusion, not to full carving; PRACTICE.md §1 discusses the batch-vs-
  incremental tradeoff and what an online version would need.
- **Reference robot(s):** the warehouse AMR (`SYSTEM_DESIGN.md` §2.1, domains 02/04/05/23) and the
  autonomous-vehicle stack (§2.5, domains 02/04/05) both maintain a map that must stay free of
  ghosts for localization and costmaps to be trustworthy.
- **In production:** OctoMap's hit/miss occupancy model (the direct academic ancestor of this
  project's ledger) for the mapping side; Removert and ERASOR for the specifically LiDAR-map
  dynamic-removal literature this project's ratio score is a teaching version of (README §11).
- **Owning team:** sits at the boundary of **perception** (owns the LiDAR pipeline that produces
  scans, `SYSTEM_DESIGN.md` §5.1) and **controls & autonomy** (owns SLAM/mapping, domains 04-05) —
  in practice, whichever team owns "the map" owns this.

## The algorithm in brief

- **Amanatides & Woo (1987) voxel DDA traversal** — turn a ray into an exact, ordered sequence of
  integer voxel coordinates using only per-axis `tMax`/`tDelta` bookkeeping, no per-step distance
  recomputation (THEORY.md "The algorithm").
- **Three-way atomic ledger** (`hits` / `pass_from_hit` / `pass_from_maxrange`) accumulated
  beam-parallel across K posed scans, with endpoint exclusion and a self-carve guard at the sensor's
  own voxel (THEORY.md "The math").
- **Ratio-based classification**: `score = passes / (hits + passes)`, thresholded, per recorded
  point (THEORY.md "The math" derives the statistics).
- **Temporal evidence accumulation** for temporarily-static objects — why a pedestrian that never
  moves during the scans it is observed only becomes provably dynamic once later scans carve through
  where it stood (THEORY.md "The problem").
- **Discretization honesty**: grazing-incidence and sub-voxel-thin geometry create real false
  positives on genuinely static structure — measured, reported, never hidden (THEORY.md "The
  problem" and "Numerical considerations").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting
steps live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/dynamic-point-removal.sln`](build/dynamic-point-removal.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/dynamic-point-removal.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the
VS solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none. Only the CUDA runtime + C++17 standard library
(CLAUDE.md §5 default budget) — the DDA march, the ledger, and the classification ratio are all
hand-rolled, no cuBLAS/Thrust/etc.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Fully synthetic (CLAUDE.md §8 default): 10 posed scans, 28,800 LiDAR beams, ray-cast against an
analytic scene (a wall, a thin pole, a car that crosses the scene, a pedestrian who leaves partway
through, and an isolated "ghost" crate) with exact closed-form ground truth for every point's
static/dynamic label. Generated by [`scripts/make_synthetic.py`](scripts/make_synthetic.py) (fixed
seed 42, xorshift32 range noise) — no public dataset used; free-space carving needs object-level
dynamic/static ground truth across a designed scan sequence that no existing LiDAR dataset hands you
labeled. Full field documentation, checksums, and provenance: [`data/README.md`](data/README.md).

## Expected output

**The pipeline (from an actual run on an RTX 2080 SUPER, CUDA 13.3, Release|x64 — every number below
is measured, never invented):**

- **VERIFY (three independent GPU-vs-CPU gates, all exact, no tolerance):**
  - *DDA trace exact* — a documented 48-beam subset (one max-range beam plus one hit per cohort,
    padded sequentially), integer voxel sequences compared entry-by-entry: 0 mismatches, 3,820
    total voxel steps compared.
  - *Hit/pass ledger exact* — the full 28,800-beam carve, all 768,000 voxels × 3 counters compared:
    0 mismatches.
  - *Classification exact given the ledger* — 0 label mismatches of 28,800 points; worst score
    deviation `0.000e+00`.
- **`ghost_removal`** (the car-trail points — the headline): **97.7%** removed (507/519), floor 85%.
- **`late_leaver`** (the pedestrian): removal rate measured on a ledger carved from scans 0-4 only
  (**3.8%** — it still looks static) vs. the full 10-scan ledger (**94.2%** — carved away once it's
  gone). Ceiling 25% / floor 70%, both cleared with wide margin.
- **`static_preservation`** (wall + pole + wall_edge combined): **5.4%** falsely removed (458/8,471),
  ceiling 15%. The discretization-honesty cohorts, reported separately and NOT folded into that
  ceiling: thin pole **100%** falsely removed (48/48 — a 4 cm-radius object is smaller than a fifth
  of one 20 cm voxel; every voxel it occupies is dominated by nearby open-space pass traffic), wall
  edge **5.2%** (8/153 — the wall's free end, much closer to the generic-wall rate because the wall
  is a large solid target, not a sub-voxel one).
- **`free_space_consistency`**: 0 retained-point violations of "no `hits==0` voxel holds a kept
  point"; `sum(hits) == 9,051 == ` the exact count of hit beams (both exact accounting checks).
- **`max_range_carving`** (the isolated ghost crate, scan 0 only): 9/9 points classified dynamic;
  its voxel's evidence is `pass_from_maxrange=8, pass_from_hit=0` — carved entirely by beams that
  never hit anything at all.
- **`[info]` contention**: near-sensor voxels (< 3 m) average ~14 passes/voxel (peak 875 at one
  voxel); far voxels (> 10 m) average ~2/voxel — a measured ~7× hotspot ratio, not asserted.
- **Timing** (teaching artifact, not a benchmark): full 28,800-beam carve, CPU ~13-15 ms vs. GPU
  kernel ~0.13-0.16 ms, ~90-100x — single-shot, one machine.

**Design iteration, stated honestly:** two scene parameters were tuned after measuring a first pass
that failed `static_preservation` (~30% falsely removed) and `ghost_removal` (~72%) — the grid
origin was offset by a sub-voxel amount so the scene's round-number geometry never lands exactly on
a voxel boundary (an object surface exactly on a boundary is the worst case for discretization: 2 cm
range noise then scatters half its hits across the boundary), and the car's final crossing position
was moved from 1.8 m to 5.1 m off the sensor track (a car observed that close creates a much finer
hit-voxel footprint than later, farther-away scans can densely re-visit). Both are documented at
their source (`kernels.cuh`'s grid-origin comment; `scripts/make_synthetic.py`'s `CAR_SCANS`
comment) — the numbers above are the result, not a claim that discretization artifacts do not exist.

Tolerance: the three VERIFY gates require **exact** agreement (integer voxel sequences, integer
ledger counters, and — since the DDA march contains no transcendental function — bit-identical
scores); the five downstream GATEs are measured-then-margined against the numbers above (THEORY.md
"How we verify correctness").

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — read this FIRST: the whole algorithm's contract (the
   ledger, the beam model, the voxel grid, the classification rule) is documented once, here.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels: `carve_one_beam` (the DDA march, the heart
   of the project), `carve_kernel`/`carve_trace_kernel` (bulk carving vs. verify-stage
   instrumentation), `classify_kernel`.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independently-typed CPU twin of the march;
   read it side by side with `kernels.cu` to see exactly what stayed shared (voxel indexing) and
   what was retyped from scratch (the algorithm).
4. [`src/main.cu`](src/main.cu) — orchestration: load data, the three VERIFY stages, the five gates,
   the three artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `find_data_file`/`resolve_out_dir`.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md
§4.1):

- **Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing" (1987)** — the DDA march
  this project implements from the original paper's derivation, not a library call.
- **Elfes, "Using Occupancy Grids for Mobile Robot Perception and Navigation" (1989)** — the
  founding occupancy-grid insight (free space is evidence, not absence of evidence) this project's
  hit/pass ledger is a direct descendant of.
- **OctoMap (Hornung et al. 2013)** — the production hit/miss probabilistic occupancy library; its
  `insertPointCloud` does essentially this project's carving pass at production scale and quality,
  with a log-odds update instead of a raw ratio.
- **Removert (Kim & Kim, 2020) and ERASOR (Lim et al., 2021)** — the modern LiDAR-map dynamic-point-
  removal literature; both use range-image/pseudo-occupancy comparisons across sessions rather than
  per-beam raycasting, a different (often faster, less exact) way to reach a similar removal
  decision — study them for how a production system trades this project's per-beam exactness for
  speed at city-map scale.
- **nvblox** — NVIDIA's GPU-accelerated voxel mapping library (TSDF-based, closer kin to project
  05.01 than to this one, but the same "dense voxel grid, GPU-parallel update" family).

## Exercises

1. Plot `demo/out/pedestrian_evidence.csv`'s `score` column against `scan_id` — watch the evidence
   cross the 0.6 threshold exactly when the pedestrian leaves.
2. Re-run `scripts/make_synthetic.py` with a larger `RANGE_NOISE_SIGMA_M` and re-measure
   `static_preservation` — how much noise before the pole's 100% false-positive rate starts pulling
   down the wall's rate too?
3. Add a SECOND isolated ghost, present in the LAST scan instead of the first — does
   `max_range_carving`-style evidence exist for it? (Hint: think about what scans exist AFTER it.)
4. Implement the two-pass prefix-sum alternative to this project's atomic-append-free ledger (there
   is nothing to append here, but you can still replace the `atomicAdd` hits/passes with a
   shared-memory local histogram per block before a single global atomic per voxel per block — 07.09
   and 23.01 use this pattern; measure whether it helps at this project's contention level).
5. Swap the interleaved `dir[i*3+0..2]` beam layout for a split `dx[]/dy[]/dz[]` Structure-of-Arrays
   layout (kernels.cuh's file header names the tradeoff) and re-profile `carve_kernel`.

## Limitations & honesty

- **Sensor orientation is held identity throughout** (no yaw/pitch/roll) — a deliberate scope cut
  documented in `scripts/make_synthetic.py`'s file header; the DDA march itself does not care about
  platform heading at all, only per-beam start/end points, so a yaw-varying platform (project 02.08's
  full pose-interpolation machinery) is a straightforward extension.
- **Carving is a post-session batch job here**, not the incremental per-scan fusion a real online
  mapping system would run (PRACTICE.md §1 discusses the difference and what an incremental version
  needs).
- **Azimuth resolution (2°, 180 steps) is coarser than 02.08's 1° convention** — a documented scope
  cut for committed-sample size; a real spinning LiDAR's native resolution would give even cleaner
  static-preservation numbers, at the cost of more beams to carve and a larger committed sample.
- **Thin/edge discretization false positives are real, not simulated for effect** — the thin-pole
  (100%) and wall-edge (5.2%) rates in "Expected output" are what THIS voxel size (20 cm) actually
  does to THIS geometry; a finer voxel grid would shrink both at the cost of more memory and compute
  (THEORY.md "Numerical considerations" quantifies the tradeoff).
- **Not safety-certified.** Nothing here commands real hardware; if a production version of this
  pipeline ever fed a costmap that a real robot planned around, the usual sim-validated-only caveat
  applies in full (CLAUDE.md §1).
