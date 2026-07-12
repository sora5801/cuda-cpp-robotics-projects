# 02.11 — Scan Context / ring-descriptor loop-closure search

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Scan Context / ring-descriptor loop-closure search`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> A false loop closure here would silently corrupt a robot's map; see
> [Limitations & honesty](#limitations--honesty) and [`PRACTICE.md`](PRACTICE.md) §1.

## Overview

This project answers, for a mobile robot, "have I been here before?" — the **place-recognition**
problem SLAM systems call **loop closure**. Every keyframe LiDAR scan is compressed into a **Scan
Context**: a 20-ring x 60-sector polar grid around the sensor, each cell holding the tallest point that
landed there. Two scans of the same physical place — even taken facing opposite directions — turn out
to produce the *same* grid with its columns cyclically shifted, because a yaw rotation is exactly a
column shift in this egocentric, polar representation. Finding the shift that makes two grids look most
alike therefore does two things at once: it scores "how similar are these places" and it hands back a
**free relative-yaw estimate**, no extra work required. Comparing a new scan against every scan the
robot has ever seen is too slow to do the expensive way every time, so a cheap **ring key** (how full is
each ring, a number that does not care about rotation) prefilters a short candidate list before the full
search ever runs — the two-stage search every deployed system uses.

The demo builds these descriptors for a synthetic 160-keyframe trajectory through a small procedurally
varied "town", searches for loop closures, and scores the result against curated ground truth: revisits
from the same heading, revisits with the heading reversed ~180°, revisits shifted a few meters sideways,
and several genuinely new places that must never be mistaken for anywhere else. Every piece named in the
catalog bullet is implemented: the ring x sector descriptor, the ring-key prefilter, and the
column-shift search.

## What this computes & why the GPU helps

Two computations, two different GPU patterns:

- **Building the descriptor is a SCATTER.** Every point independently computes its own (ring, sector)
  cell and races to claim that cell's running maximum height — the natural GPU mapping is one thread
  per POINT (not per cell), each doing an atomic max into a shared 1200-cell output. This is the
  *reduce-by-key* pattern seen throughout the repo (voxel-grid downsampling's spatial hash, 02.01,
  is its closest relative) specialized to a fixed, small key space.
- **Searching for the best shift is a CANDIDATE x SHIFT sweep.** For every candidate scan and every one
  of 60 possible column shifts, the mean cosine distance between 60 column pairs must be computed and
  reduced to one number — a natural **map + block-level reduce**: one GPU block per (candidate, shift)
  pair, one thread per column, ending in a shared-memory tree reduction. This is the project's hot
  loop and the GPU mapping the catalog bullet calls out explicitly.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception / SLAM & mapping — the **place-recognition / loop-closure trigger**
  sub-problem. It runs *alongside* frame-to-frame odometry, not instead of it: odometry answers "where
  am I relative to a moment ago", this project answers "is my CURRENT keyframe the same physical place
  as some keyframe from long ago" — the question that catches accumulated drift before it corrupts the
  map.
- **Upstream inputs:** one `PointCloud` per keyframe (this repo's flattened `sensor_msgs/PointCloud2`
  shape, `docs/SYSTEM_DESIGN.md` §3), already preprocessed by the keyframe front end (voxel-grid
  downsampling and ROI cropping, projects [02.01](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md)/[02.02](../02.02-roi-crop-passthrough-organizedunorganized/README.md),
  and motion-deskewed if the platform is fast, project [02.08](../02.08-per-point-motion-deskew-with-pose-interpolation/README.md)).
- **Downstream consumers:** a positive detection (a candidate `match_idx` + a coarse yaw estimate)
  hands off to **geometric verification** — project [02.10](../02.10-fpfh-descriptors-ransac-global-registration/README.md)'s
  descriptor-matching + RANSAC global registration (or a compact ICP, illustrated by this project's own
  `yaw_handoff` gate) — which either confirms the loop with a real `T_match_query` transform or rejects
  it as an aliasing false alarm. A confirmed loop becomes an edge in the **pose-graph optimizer** (the
  `05.xx` SLAM/mapping domain by name — this repo's SLAM & mapping section), which redistributes
  accumulated drift across the whole trajectory. This project also serves **kidnapped-robot
  recovery/relocalization**: the same search, run with no prior pose at all, against a stored map.
- **Rate/latency budget:** runs once per KEYFRAME, not once per scan — typically 1–2 Hz (keyframes are
  spaced by distance/rotation traveled, not wall-clock time), closer to the `docs/SYSTEM_DESIGN.md`
  §1.1 "global planner (route), 0.1–1 Hz, event-driven" row than to the per-scan perception rate. This
  demo measures its own search cost directly (see "Expected output"): comfortably inside that budget
  even before the GPU is asked to do anything clever.
- **Reference robot(s):** the **warehouse AMR** (§2.1 — a fixed facility with repeating aisles is
  exactly where perceptual aliasing, this project's own honest limitation, bites hardest) and the
  **autonomous-vehicle stack** (§2.5 — loop closure against a prior map is a standard relocalization
  fallback when GPS is degraded).
- **In production:** would be replaced/joined by a learned place-recognition network (NetVLAD,
  OverlapNet) or a bag-of-visual-words system (DBoW2, the descriptor-vocabulary approach ORB-SLAM uses)
  running as a background thread inside the SLAM back-end, feeding the SAME pose-graph edge interface
  this project's output would feed.
- **Owning team:** SLAM / localization, inside the controls & autonomy org
  (`docs/SYSTEM_DESIGN.md` §5.1), working closely with perception (who own the upstream keyframe
  pipeline) and the mapping/pose-graph sub-team who consume its detections.

## The algorithm in brief

- **The Scan Context matrix** — bin every point into a polar (ring, sector) grid around the sensor;
  each cell keeps the MAX height of the points that land there (Kim & Kim, IROS 2018). See
  [`THEORY.md`](THEORY.md) "the math" for the ring/sector formulas and why max-height is the right
  summary statistic.
- **The ring key** — a per-ring occupancy fraction, rotation-invariant by construction, used as a cheap
  L1-distance prefilter over the whole database before the expensive search runs. See `THEORY.md` "the
  math" for the derivation and the `ringkey_prefilter` gate for the measured recall-vs-speed trade.
- **The column-shift search** — for every candidate and every one of 60 possible shifts, the mean
  column-wise cosine distance; the minimum over shifts is "how different are these places" and the
  argmin shift is a free yaw estimate. See `THEORY.md` "the math" for why rotation becomes a shift, and
  the numerics section for the empty-cell masking this computation depends on getting right.
- **The two-stage search** — ring-key prefilter narrows the field, then the full shift search runs only
  on the survivors: the deployed pattern (not exhaustive search), gated end-to-end by
  `ringkey_prefilter` against an exhaustive baseline.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/scan-context-ring-descriptor-loop-closure-search.sln`](build/scan-context-ring-descriptor-loop-closure-search.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/scan-context-ring-descriptor-loop-closure-search.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

No optional dependencies: only the CUDA runtime + C++17 standard library (CLAUDE.md §5 default budget).
Every kernel here — the scatter-max descriptor build, the ring-key reduction, the shift-distance search —
is hand-written; nothing from cuBLAS/cuFFT/Thrust is linked.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

100% synthetic (CLAUDE.md §8): a hand-authored 160-keyframe trajectory through a procedurally-varied
4x3-block "town", with curated ground-truth revisit pairs (same-heading, heading-reversed, laterally
offset) and genuinely new places. Regenerate with `python scripts/make_synthetic.py --seed 42`. Full
field-by-field documentation, provenance, and checksums: [`data/README.md`](data/README.md).

## Expected output

Three GPU-vs-CPU correctness gates and five independent, ground-truth-scored gates, all measured on the
committed sample (Release|x64, RTX 2080 SUPER, CUDA 13.3) — every number below is from an actual run,
never fabricated (CLAUDE.md §8):

| Gate | What it checks | Measured | Floor |
|------|-----------------|----------|-------|
| `VERIFY(scan_context)` | GPU vs. CPU descriptor matrices (scatter-max is order-independent — near-exact) | 99.74% cells exactly equal | ≥ 99.5% |
| `VERIFY(ring_key)` | GPU vs. CPU ring occupancy (a cascade of the above) | 94.91% entries match | ≥ 90% |
| `VERIFY(shift_distance)` | GPU vs. CPU column-shift search, identical inputs | worst abs diff 3.0e-7 | ≤ 2e-4 |
| `GATE loop_detection` | precision/recall on 13 curated positive queries + 8 negatives | recall 0.769, precision 1.000 | recall ≥ 0.70, precision ≥ 0.90 |
| `GATE rotation_invariance` | rotated-cohort recall vs. same-heading recall + recovered-yaw error | both recalls 1.000; mean yaw error 0.0° | recall gap ≤ 0.40; yaw ≤ 24° |
| `GATE lateral_sensitivity` | detection rate reported by lateral-offset magnitude (honesty gate) | 4 offsets (0.6/1.3/2.6/2.6 m) measured and reported | reported at all |
| `GATE negative_cohort` | zero false closures on 8 never-revisited places | 0/8 fired | == 0 (safety-critical, see PRACTICE.md) |
| `GATE ringkey_prefilter` | prefilter recall vs. an exhaustive search | 125/145 = 86.2% | ≥ 80% |

The demo also writes an `[info]`-only illustration (`yaw_handoff`): handing the recovered shift-yaw to a
compact ICP converges it to 0.000 m RMSE in 1 iteration, versus 0.645 m RMSE after 3 iterations starting
from identity — the free yaw estimate is not just a number, it measurably helps the next stage.
Canonical stable-line output: [`demo/expected_output.txt`](demo/expected_output.txt). Artifacts (Scan
Context heatmaps, a trajectory view with detected loops drawn as chords, a full PR-curve sweep,
`gates_metrics.csv`) land in `demo/out/` — see [`demo/README.md`](demo/README.md).

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: load data, build descriptors (GPU+CPU), verify, run the
   full search sweep (exhaustive + prefiltered), score every independent gate, write artifacts.
2. [`src/kernels.cuh`](src/kernels.cuh) — the data-layout contract (Scan Context matrix, ring key,
   empty-cell sentinel) and the small shared formulas (ring/sector binning, column cosine distance).
   Read the file header in full — it carries two real bugs this project's own development caught and
   fixed, and why.
3. [`src/kernels.cu`](src/kernels.cu) — the three GPU kernels: `sc_build_kernel` (the scatter-max),
   `ring_key_kernel` (the occupancy reduce), `sc_shift_distance_kernel` (the candidate x shift search —
   the project's most interesting kernel; read its header comment for the coalescing argument).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle twins.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `find_data_file`/`resolve_out_dir`.

## Prior art & further reading

- **Kim, G. & Kim, A., "Scan Context: Egocentric Spatial Descriptor for Place Recognition within 3D
  Point Cloud Map", IROS 2018** — the method this project reimplements didactically; the original also
  proposes Scan Context++ (a rotation-invariant variant using a discrete Fourier transform of the ring
  key) worth reading once this version's mechanics are clear.
- **PCL / Open3D** — production point-cloud libraries with their own registration/feature pipelines;
  study their descriptor APIs for how a real system packages this as a reusable component.
- **ORB-SLAM2/3 (Mur-Artal et al.)** — the canonical bag-of-visual-words loop-closure detector (DBoW2),
  the image-feature analogue of this project's geometric descriptor; project
  [01.04](../../01-perception-cameras-vision/01.04-feature-pipeline/README.md)'s feature-matching
  lessons (a distinctiveness/ratio test, not raw distance, decides a match) apply here too.
- **OverlapNet (Chen et al., RSS 2020) / LCDNet / NetVLAD** — learned place-recognition, this project's
  natural successor: a CNN over a range image or point cloud replaces the hand-built descriptor while
  keeping the same "produce a candidate + a coarse relative pose" interface.
- **GTSAM / g2o** — the pose-graph optimizers that consume a confirmed loop-closure edge (this
  project's ultimate downstream consumer, by name, in "System context" above).

## Exercises

1. Sweep `kScDistanceThreshold` and reproduce `demo/out/pr_curve.csv` by hand from `gates_metrics.csv` —
   confirm the chosen operating point sits in the middle of the measured gap, not hugging either edge.
2. Implement Scan Context++'s sector key / column-based variant and compare its rotation-invariance
   margin against this project's ring-key prefilter.
3. Replace the brute-force ring-key prefilter (a full L1 scan of every valid candidate) with an actual
   k-d tree over ring keys, and measure whether the *result* changes (it shouldn't) versus how the *cost*
   scales with database size.
4. Widen the world (more blocks, taller/more-varied buildings) and re-measure the negative-cohort
   aliasing gap this project's own diagnostic run found — does more structural variety close it?
5. Feed the recovered shift-yaw into project [02.06](../02.06-icp-point-to-point-point-to-plane-gicp/README.md)'s
   full 6-DOF point-to-plane ICP instead of this project's compact 2D illustration, and measure the
   convergence-speed payoff at loop-closure scale.

## Limitations & honesty

- **Perceptual aliasing is real and measured, not swept under the rug.** This project's synthetic
  world's repeating grid layout produces at least one pair of structurally similar (but physically
  different) streets whose Scan Context distance is smaller than some genuine lateral-offset revisits —
  exactly the "corridors that all look alike" problem named in `THEORY.md`. The chosen operating
  threshold sits safely below every such confound found in the committed sample; a larger or more
  repetitive real building would need a wider margin, a richer descriptor, or geometric verification on
  every candidate (which is what `02.10` is for).
- **The ring-key prefilter is not lossless.** It misses the exhaustive-search best match on roughly 14%
  of queries in the committed sample (the `ringkey_prefilter` gate measures exactly this) — the honest
  cost of the speed it buys, quantified rather than assumed.
- **Rooftops are not ray-cast** in the synthetic world (`scripts/make_synthetic.py`'s `simulate_scan()`
  models ground + vertical walls only) — a documented simplification of the data generator, not of the
  algorithm.
- **Two real bugs surfaced during this project's own development** and are left documented in place
  (not scrubbed from history) because they are genuinely instructive: an empty-cell sentinel that could
  never be beaten by a legitimate (negative) ground return, and a keyframe-sampling asymmetry that put
  forward/reversed revisit anchors ~6 m apart instead of on the same physical spot. See `kernels.cuh`'s
  file header and `scripts/make_synthetic.py`'s `sample_keyframes()` docstring, and `THEORY.md`
  "numerical considerations" for the full story.
- **Sim-validated only.** Nothing here commands real hardware motion, but a detected loop closure would
  feed a pose-graph optimizer that DOES influence a robot's belief about its own position — see
  `PRACTICE.md` §1 for why a false positive here is a safety-relevant failure mode, not just an
  accuracy statistic.
