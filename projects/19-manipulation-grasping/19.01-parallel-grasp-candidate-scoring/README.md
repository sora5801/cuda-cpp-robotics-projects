# 19.01 — Parallel grasp-candidate scoring: antipodal sampling over point clouds

**Difficulty:** ★ beginner · **Domain:** 19. Manipulation & Grasping

> Catalog bullet (source of truth, verbatim): `★ Parallel grasp-candidate scoring: antipodal sampling over point clouds`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**Antipodal grasping asks a geometric question millions of times: "if I close a two-finger gripper
around these two points, does it actually hold the object?"** This project answers that question on
the GPU, at scale. It samples K = 4096 candidate contact-point pairs over a point cloud, and for each
one independently: estimates local surface normals (PCA over k-nearest-neighbors, project 02.06's
pattern, reused and cited), searches along the inward normal ray for a plausible antipodal partner,
and scores the pair against three physically-grounded gates — a Coulomb friction cone (does the
grasp actually resist gravity without slipping?), a gripper stroke limit (can this hardware's
fingers even reach that far?), and an approximate finger-clearance check. The demo runs this pipeline
on three synthetic objects whose good grasps are known **geometrically** — a box, a cylinder, a
sphere — so the ranked output can be checked against closed-form ground truth, not just eyeballed.
Every component named in the catalog bullet ("antipodal sampling over point clouds") is implemented
in full; nothing here is documented-only.

## What this computes & why the GPU helps

Per demo run: 3 objects × (1 normal estimate per point + 4096 candidate ray-searches + 4096 scoring
evaluations), each search/score a brute-force scan over its object's whole cloud (6,000–9,000
points). That is roughly 100–150 million independent floating-point comparisons — every one of them
data-parallel across either points or candidates, with zero communication between them.

- **Pattern:** sampling — one thread per point (normals) or one thread per candidate (generation,
  scoring); no candidate's search or score depends on any other candidate's result. This is the same
  "thread = independent problem" shape as project 08.01's rollouts and 02.06's correspondence search,
  applied here to grasp geometry instead of control rollouts or point registration.
- **Measured reality (RTX 2080 SUPER):** all three kernels combined finish in **1–2 ms per object**
  (`[info]` lines from a real run — normals ~1–2 ms, candidate generation + scoring ~1–2 ms) — for a
  bin-picking cell whose whole perceive→plan→move cycle budgets a few hundred milliseconds (§ System
  context below), grasp-candidate scoring is not the bottleneck.
- **No reduction, unlike 02.06.** ICP (02.06) must SUM thousands of points' contributions into one
  shared 6×6 system — a genuine tree reduction. This project's candidates are independent OUTPUTS,
  not partial sums of one answer, so ranking is a host sort over the downloaded array, not a GPU
  reduction kernel (see `kernels.cuh`'s launcher comments, and the 12.01 NMS parallel in "Prior art").

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **planning** layer, specifically the grasp-planning box in
  SYSTEM_DESIGN.md §2.2's manipulator work-cell diagram — "GRASP PLANNING [19 →]: candidate sampling
  + scoring", sitting between perception (vision) and kinematics (reachability).
- **Upstream inputs:** a segmented object `PointCloud` (SYSTEM_DESIGN.md §3.6) — in a real cell,
  produced by stereo depth (01.02) plus instance segmentation (19.05's bin-picking stack names this
  step explicitly); this demo substitutes three synthetic analytic-object clouds in its place
  (`data/README.md` explains why synthetic beats a public dataset for THIS project's verification
  strategy).
- **Downstream consumers:** a ranked list of grasp poses (contact points + axis + width + score).
  SYSTEM_DESIGN.md §4.2 (Chain B) names the immediate next stage: **19.08, batched-IK grasp
  reachability ranking** — which of these geometrically-good grasps can the arm's kinematics actually
  reach, and with how much joint travel (19.08 itself leans on the batched numerical IK pattern
  09.05 teaches). Reachable grasps then feed **06.07 (cuRobo-style arm motion planning)**, which
  produces the joint-space trajectory that finally reaches the object.
- **Rate / latency budget:** SYSTEM_DESIGN.md §1.1's "Local planner / trajectory replan" row
  (10–50 Hz, 20–100 ms per replan) is the closest generic budget, but the more precise number is
  SYSTEM_DESIGN.md §4.2's own figure for this exact stage: **grasp poses per view, < 100 ms**, inside
  a bin-picking cycle that should complete in a few hundred milliseconds total (§2.2: "the whole
  perceive→plan→move loop should fit in a few hundred milliseconds"). Measured GPU time here
  (1–2 ms per object, all three kernels) leaves that 100 ms budget almost entirely to perception
  (segmentation, pose estimation) and to the downstream IK/motion-planning stages — candidate scoring
  itself is not the pacing item.
- **Reference robot(s):** the **6-DoF manipulator work cell** (SYSTEM_DESIGN.md §2.2) doing
  pick-and-place or bin picking; this project is also the first link in SYSTEM_DESIGN.md §4.2's
  named "Chain B — manipulator pick-and-place" worked example.
- **In production:** a learned grasp-quality network (GPD, Dex-Net's GQ-CNN) typically REPLACES or
  augments hand-built scoring like this project's — but even learned scorers are usually trained
  against, or validated with, an analytic force-closure/antipodal check exactly like the one here
  (see "Prior art" below). Purely geometric antipodal sampling (this project's approach) remains the
  default for simple/known-CAD parts (kitting, machine tending) where a network is overkill.
- **Owning team:** **manipulation** (a sub-team of controls/autonomy, SYSTEM_DESIGN.md §5.1),
  working closely with perception (who provide the segmented cloud) and with the team owning 19.08 /
  06.07 (who consume this project's ranked output).

## The algorithm in brief

- **PCA surface normals** — k=16 nearest neighbors, in-register cyclic-Jacobi eigensolve on the 3×3
  covariance, oriented outward from the object's centroid. Project 02.06's exact pattern, reused and
  cited (`src/kernels.cu`), with one policy flip (outward vs. 02.06's inward) explained in both files.
  → [THEORY.md](THEORY.md) §The GPU mapping.
- **Antipodal candidate search** — hash-pick a contact point, ray-search the whole cloud along its
  inward normal for the nearest "exit point" with an opposing normal. → THEORY.md §The algorithm.
- **Coulomb friction-cone / two-contact force-closure test** — the project's mathematical heart:
  derived from first principles from Coulomb's law, implementing the classical two-point
  force-closure theorem. → THEORY.md §The math.
- **Gripper-width and finger-clearance gates** — turning "geometrically antipodal" into
  "actually graspable by THIS hardware". → THEORY.md §The algorithm.
- **Ranking** — host `std::sort` by score, top-M selection (the same honest "not every step needs a
  GPU kernel" call project 12.01 makes for greedy NMS). → THEORY.md §The GPU mapping.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/parallel-grasp-candidate-scoring.sln`](build/parallel-grasp-candidate-scoring.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/parallel-grasp-candidate-scoring.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
Candidate seeding uses a stateless hash (`grasp_hash_u32`, `src/kernels.cuh`) rather than cuRAND, so
there is no RNG library dependency either (README "Exercises" names on-device cuRAND Philox streams
as the natural next step for the learner who wants to compare against the real thing).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Three synthetic, analytically-known objects — a box (60×40×100 mm), a cylinder (r=25 mm, h=120 mm,
lateral surface only), and a sphere (r=30 mm) — each a noisy point cloud (0.3 mm axial + 0.15 mm
tangential Gaussian noise) generated by `scripts/make_synthetic.py` from a fixed seed (42). Synthetic
is the deliberate choice here, not a default fallen into: this project's verification strategy needs
objects whose good grasps are known **geometrically**, which no public grasp dataset provides in a
directly checkable form — `data/README.md` explains this in full, with checksums, exact byte layout,
and the noise-vs-point-spacing arithmetic. No public dataset applies; `scripts/download_data.ps1` is
an honest no-op.

## Expected output

Sixteen stable lines — banner, `PROBLEM:`, three `SCENARIO:` lines, three `VERIFY:` lines, five
`CHECK:` lines, two `ARTIFACT:` lines, `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Three layers of verification, all measured on
a real run (RTX 2080 SUPER):

1. **The §5 GPU-vs-CPU gate**, on the box object: normals agree within 0.5° (measured worst: 0.034°);
   candidate generation matches **exactly**, index-for-index (measured: 0/4096 mismatches); scoring
   agrees within relative tolerance 1e-3 (measured worst: 5.96e-08).
2. **Analytic gates**, one per object: does the top-10 ranked list actually contain valid,
   gripper-feasible antipodal grasps at the right width and axis? All three objects pass — box grasp
   widths measured 39.4–60.5 mm (ground truth 40/60 mm), cylinder 49.2–50.6 mm (ground truth 50 mm),
   sphere 59.4–60.9 mm (ground truth 60 mm), every one with antipodal-quality score ≥ 0.9999.
3. **Negative controls**, box only: 614 of 4096 random candidates found the geometrically-antipodal
   but 100 mm-wide (gripper max stroke 90 mm) axis pairing and were correctly marked infeasible; all
   12 hand-picked adjacent-face (non-antipodal) candidates were correctly rejected by the
   friction-cone gate.

Two artifacts land in `demo/out/`: `grasps.csv` (top-10 ranked grasps per object, every field) and
`grasp_cloud.csv` (subsampled clouds + top-5 grasp axes, for plotting — `demo/README.md` gives the
plotting recipe).

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: load 3 objects → normals →
   VERIFY (box) → per-object generate/score/rank → analytic gates → artifacts.
2. [`src/kernels.cuh`](src/kernels.cuh) — the point-cloud/candidate/score layouts, every tuning
   constant with its sizing rationale, and `grasp_hash_u32` (the stateless per-candidate seed).
3. [`src/kernels.cu`](src/kernels.cu) — the three kernels. The single most interesting one:
   `generate_candidates_kernel` — read its header comment for the exact ray-proximity search, then
   compare it to `score_candidates_kernel`'s friction-cone math right below it.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plain-C++ correctness oracle; read it next
   to `kernels.cu` to see exactly what "one thread per point/candidate" replaced.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — CLAUDE.md §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Nguyen (1988), "Constructing Force-Closure Grasps"** — the two-contact antipodal force-closure
  theorem `THEORY.md` derives and this project's friction-cone gate implements.
- **Ferrari & Canny (1992), "Planning Optimal Grasps"** — the epsilon-quality metric, the full
  quantitative grasp-quality measure this project's simpler `antipodal_cos` score is a fast proxy
  for (`THEORY.md` "Where this sits in the real world"); 19.02 (grasp-wrench-space computation)
  implements it.
- **GPD (Grasp Pose Detection, ten Pas et al.)** and **Dex-Net / GQ-CNN (Mahler et al.)** — learned
  grasp-quality networks trained (and, for Dex-Net, partly labeled) against exactly this kind of
  analytic antipodal/force-closure scoring; compare their neural scorer to this project's hand-built
  one and see where a learned model earns its keep.
- **GraspIt!** (Miller & Allen) — the classical grasp-planning and force-closure analysis simulator;
  study its contact/wrench-space machinery, the direct ancestor of 19.02's project.
- **cuRobo** (NVIDIA) — the GPU motion-planning library that would sit downstream of this project's
  output in a real cell (see "System context" above and 06.07's project).
- **PCL (Point Cloud Library)**, `pcl::NormalEstimation` — the production version of this project's
  PCA-normals kernel; study its k-d-tree-accelerated neighbor search as the fix for this project's
  brute-force O(n²) normals cost (README "Exercises").

## Exercises

1. **Plot the artifact:** `demo/out/grasp_cloud.csv` → 3D scatter of each object's cloud, with a line
   segment through each top-5 grasp's two contact points. Confirm by eye that the box's lines run
   parallel to a coordinate axis, the cylinder's lines pass near its central axis, and the sphere's
   lines pass near its center.
2. **Loosen the friction cone:** raise `friction_mu` in `data/sample/objects_meta.csv` from 0.5 toward
   1.0 (a very high-friction fingertip) and rerun. Predict, then check, whether any of the 12
   adversarial box candidates start passing — and relate the answer to `THEORY.md`'s friction-cone
   derivation.
3. **Accelerate the search:** both `generate_candidates_kernel` and `score_candidates_kernel`'s
   clearance check are brute-force O(n) scans per thread. Build a uniform spatial grid (a light
   version of 07.x's spatial-acceleration projects) and measure the speedup on the cylinder object
   (9,000 points, the largest cloud here).
4. **On-device seeding:** replace `grasp_hash_u32` with a cuRAND Philox generator (one subsequence
   per candidate) and confirm the ranked top-10 grasps are statistically similar (not identical —
   Philox and this project's triple32 hash are different generators) across many seeds.
5. **Fuse ranking onto the GPU:** replace the host `std::sort` top-M selection with a GPU top-k
   (bitonic top-k or a CUB `DeviceSegmentedRadixSort`) and measure whether it beats the host sort at
   K=4096 — then measure again at K=100,000 and see where the crossover is.

## Limitations & honesty

- **Force closure, not full wrench closure.** The friction-cone gate implements Nguyen's two-contact
  FORCE-closure theorem (can the two contacts resist an arbitrary net FORCE without slipping) — it
  is not the full 6-DOF wrench-closure test (can the grasp also resist an arbitrary TORQUE), which in
  general needs more than two point contacts. Real parallel-jaw grasps rely on additional passive
  stability (soft fingertips, gripper compliance, contact patches rather than points) beyond this
  idealization — `THEORY.md` "Where this sits in the real world" and `PRACTICE.md` §1 say more.
- **`antipodal_cos` is a fast proxy, not the epsilon-quality metric.** Production grasp planners
  (Dex-Net, GraspIt!) compute the Ferrari-Canny epsilon-quality (the radius of the largest ball
  inscribed in the grasp wrench space) — a strictly more informative, strictly more expensive metric.
  19.02 (grasp-wrench-space computation) implements it; this project's simpler score is the honest
  "teaching core" the catalog bullet asks for.
- **Brute-force O(n) searches, not spatially accelerated.** Every kernel here scans its whole cloud;
  a k-d-tree or spatial hash (as PCL/production stacks use) would cut this by orders of magnitude at
  larger cloud sizes — a deliberate clarity-over-cleverness scoping choice (CLAUDE.md §1;
  README "Exercises" 3 asks the learner to build the fix).
- **Convex analytic objects only; the clearance gate is under-exercised.** The finger-clearance check
  (`kernels.cuh`'s `kClearanceRadiusM`/`kClearanceDeadzoneM`) is real code, run on every candidate,
  but on these three CONVEX solids there is (by construction) never a surface point between two
  genuinely antipodal contacts, so it never actually rejects a candidate in this demo's measured
  output. A concave test object (not included here) would exercise it — a natural extension, not a
  bug in the current one.
- **Top-M is not diversity-filtered.** Random sampling can (and does, in this demo's own
  `grasps.csv`) surface near-duplicate or mirror-image grasps in the top-10 — a real system typically
  follows with a pose-clustering/NMS pass (the same idea as 12.01's greedy NMS, applied to grasp
  poses instead of bounding boxes).
- **No real-hardware claim.** This project's output is a ranked list of geometric grasp candidates —
  it commands nothing directly, but it is squarely in the causal chain that would (via 19.08's
  reachability ranking and 06.07's motion planner) eventually command a real gripper to close on a
  real object. Everything here is validated only against synthetic point clouds; no claim is made
  about real sensor noise, real object materials, or real friction coefficients (PRACTICE.md §2
  dates and caveats the illustrative μ=0.5 used throughout). Sim-validated only, not
  safety-certified (CLAUDE.md §1).
