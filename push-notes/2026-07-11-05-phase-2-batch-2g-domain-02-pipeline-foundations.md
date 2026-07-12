# Push note — 2026-07-11-05: batch 2g — domain 02's pipeline foundations

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2g (**63/505**; domain 02 at 5/20 with the 02.06 flagship) lays the LiDAR pipeline's
foundations in execution order: voxel downsampling (02.01), the glue kernels (02.02), ground
segmentation (02.03), and obstacle clustering (02.04) — the chain 02.01 → 02.03 → 02.04 that
every scan traverses before anything downstream sees an "object". The batch's through-line is
the repo's determinism doctrine maturing into a taxonomy: 02.01 shows determinism achieved by
*ordering* (sort-based reduction, bit-identical across runs) versus surrendered to atomics
(measured at 5.2e-6 m run-to-run); 02.02 gets exactness for free from *order-independent
operators* (min never cares who arrives first — every comparison in the project is exact);
02.04 gets it from *canonical form* (two different algorithms — lock-free union-find and label
propagation — proven to converge to the identical canonical partition). And 02.04 lands the
complexity lesson three CCL projects built toward: on a designed 299-point snake, label
propagation needs 299 sweeps where union-find needs 2. Also in the batch: 02.01 is the repo's
first Thrust project and ratified the CUDA 13.3 CCCL toolchain flags for everyone after it;
02.03 gates the RANSAC iteration-count formula *analytically* and shows single-plane RANSAC
missing 93% of multi-level ground that the Patchwork++-style zone model recovers at 0.954 IoU.

## What changed

- **[projects/02-perception-lidar-point-clouds/02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/](../projects/02-perception-lidar-point-clouds/02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/)** —
  atomicCAS open-addressing hash vs thrust-sort-based reduction, determinism study, probe
  statistics, surface-not-volume occupancy lesson; first Thrust project (CCCL flags ratified in
  the vcxproj with full comments).
- **[projects/02-perception-lidar-point-clouds/02.02-roi-crop-passthrough-organizedunorganized/](../projects/02-perception-lidar-point-clouds/02.02-roi-crop-passthrough-organizedunorganized/)** —
  hand-rolled Blelloch scan == Thrust == CPU (bit-exact), stable compaction with gated order
  preservation, 5-plane frustum crop, organized↔unorganized round trip with exact collision
  bookkeeping.
- **[projects/02-perception-lidar-point-clouds/02.03-ground-segmentation/](../projects/02-perception-lidar-point-clouds/02.03-ground-segmentation/)** —
  GPU RANSAC (K=1024, formula-gated) + Patchwork++-style concentric-zone model on a designed
  flat/ramp/plateau scene with canopy overhang; the single-plane failure asserted at 93.25%.
- **[projects/02-perception-lidar-point-clouds/02.04-euclidean-clustering-via-gpu-union-find/](../projects/02-perception-lidar-point-clouds/02.04-euclidean-clustering-via-gpu-union-find/)** —
  lock-free atomicCAS union-find vs label propagation on identical edges; partition exact vs
  truth; separation + chaining cohorts asserted; the snake convergence gate.
- **[docs/STATUS.md](../docs/STATUS.md)** — 02.01–02.04 → `done` (**63/505**).

## New projects (didactic blurbs)

**02.01 — Voxel downsampling** (★). Why LiDAR density falls as 1/r² and what a centroid keeps
vs loses; the canonical GPU hash-insert loop taught against the sort-reduce alternative; the
honest measurement that float atomics wander a few micrometers run to run — and how ordering
buys the determinism back.

**02.02 — Glue kernels.** The prefix scan as *the* parallel primitive: Blelloch up-sweep and
down-sweep taught phase by phase, then proven bit-exact against Thrust and serial CPU. Where
crops actually come from in production (vehicle-body masks, camera frusta), and why organized
clouds are the sensor's native geometry.

**02.03 — Ground segmentation** (★). RANSAC's probability formula derived and then *checked
against its own run* (w=0.570 → 34 iterations needed, 1,024 run); the designed multi-level
scene where one plane cannot win; zone-model recovery with canopy false positives at exactly
0.00% — with the drive-under-the-tree stake stated.

**02.04 — Euclidean clustering.** Objects are connected components of the d-graph; the
lock-free union step (find roots, CAS the smaller, retry) taught line by line; the chaining
hazard asserted as the known single-linkage failure it is, with semantics named as the
production fix. Money numbers: 2 sweeps vs 299 on the same edges.

## How to build & run

```powershell
projects\02-perception-lidar-point-clouds\02.01-voxel-grid-downsampling-with-gpu-spatial-hashing\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.02-roi-crop-passthrough-organizedunorganized\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.03-ground-segmentation\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.04-euclidean-clustering-via-gpu-union-find\demo\run_demo.ps1
```

## What to study here

Read the four as the single pipeline they are — a scan enters 02.01 and exits 02.04 as objects.
Then study the determinism taxonomy across 02.01 (ordering), 02.02 (order-independent
operators), and 02.04 (canonical form) — three different ways to the same reproducibility goal,
each with its cost. Exercise: chain the actual demos — feed 02.01's downsampled output into
02.03's CZM and measure what leaf size does to plateau recall.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-11), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (12/12, 13/13, 19/19, 17/17);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder.

- **02.01:** Method B bit-identical ×3 runs and bit-exact vs twin; Method A occupancy exact,
  centroids 3.3e-6 m vs independent twin; partition + containment invariants exact.
- **02.02:** three scan implementations bit-exact; all four compactions exact incl. order;
  round trip identity on 12,361 cells; collision bookkeeping 12,561 = 12,361 + 200 exact.
- **02.03:** RANSAC flat 0.084°/5.2 mm, P/R 0.981/1.000; formula check 33.8 ≪ 1,024;
  single-plane failure 93.25% asserted; CZM IoU 0.954; canopy FP 0.00%; obstacles 8.96%
  (ceiling 13%, the curb ambiguity documented).
- **02.04:** union-find + label propagation both bit-exact vs independent CPU union-find and
  vs generator truth; snake 2 vs 299 sweeps (0.25 vs 14.8 ms); noise 28/28 rejected; stats
  counts exact, centroids ≤1.7e-5 m.

## Known limitations / TODOs

- 02.01: hash capacity fixed (resizing documented); 02.02: fused-vs-chained wall-time favors
  chained at this size (bytes favor fused — both reported); 02.03: seed selection simplified vs
  Patchwork++'s percentile rule (stated); 02.04: filled-lattice obstacle sampling rather than
  ray-cast returns (a deliberate, documented scoping — this project teaches clustering, not
  sensor simulation).

## Next push preview

Batch 2h: 02.05 KD-tree/LBVH + KNN search (the neighbor-query engine), 02.07 NDT scan matching,
02.08 per-point motion deskew, 02.09 normal + curvature estimation.
