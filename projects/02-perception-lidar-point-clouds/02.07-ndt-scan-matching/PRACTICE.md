# 02.07 — NDT scan matching (Autoware-style map localizer): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Section dated 2026-07-11 — all figures/prices are illustrative and go stale; verify current.

## 1. Building it — construction of the robot/part

NDT scan matching is software-only, but it depends on a physical artifact this project's teaching
core treats as already existing: **the map**. In production this is where most of the unglamorous
engineering effort actually lives.

- **Map building is a survey operation.** A real map is built by driving/walking the environment
  once (or repeatedly) with a calibrated LiDAR + high-accuracy positioning (survey-grade GNSS/RTK
  outdoors, a SLAM back-end with loop closure indoors — 05.xx's domain), then post-processing:
  voxel-downsampling (02.01), removing dynamic objects (02.13/02.14), and often manual QA of
  problem areas (glass, featureless corridors — this project's own corridor scene is exactly the
  kind of stretch a real mapping team flags for extra survey passes or supplemental fiducials).
- **Map MAINTENANCE is the unsung production burden.** A map is a snapshot; the world drifts —
  construction, seasonal foliage, rearranged shelving in a warehouse, a repainted lane. Production
  localization stacks version their maps explicitly (a map ID + timestamp + checksum shipped with
  every vehicle/robot, exactly like this project's `data/README.md` documents a checksum for its
  own committed `map.bin`) and set a RE-SURVEY CADENCE — weeks for a fast-changing warehouse floor,
  months to years for stable outdoor infrastructure, immediately after any known environment
  change (a wall moved, a new aisle added). A stale map is not a "worse" map — past a threshold, it
  is actively dangerous, because NDT will confidently converge to a wrong pose against surfaces
  that no longer exist. Health monitoring (§3 below) is the runtime mitigation; re-survey cadence
  is the operational one.
- **Physical carrier of "the map":** typically a compressed point-cloud or voxel-grid file (`.pcd`,
  a custom binary NDT-cache like this project's `map.bin`, or a tiled format for city-scale
  coverage) distributed to every vehicle/robot in a fleet — a real logistics and versioning problem
  at fleet scale, not just a file on one machine.

## 2. Real hardware — chips, parts, illustrative BOM

*All parts below are **illustrative examples, never endorsements**; verify current pricing/
availability before relying on any of them.*

| Tier | Compute | LiDAR | Notes |
|------|---------|-------|-------|
| Hobby/research | NVIDIA Jetson Orin Nano (~$249, 2024 list) or a laptop RTX GPU | Livox Mid-360 (~$799) or a used Velodyne VLP-16 (this project's 16-channel model) | Enough compute for a 1-10 Hz NDT loop on a room/building-scale map; a laptop GPU (this project's own reference machine, an RTX 2080 SUPER) is overkill but convenient for development. |
| Industrial/AMR | NVIDIA Jetson AGX Orin (~$1,999 dev kit) or an embedded x86 + discrete RTX | Ouster OS1-32/64 or Hesai XT32 (~$4,000-$9,000) | The compute tier a warehouse AMR (this project's reference robot) typically ships; enough headroom to run NDT alongside a costmap (23.xx) and an EKF (04.xx) on one box. |
| Automotive-grade | An automotive SoC (e.g., NVIDIA DRIVE Orin class) with ASIL-rated safety island | Automotive LiDAR (e.g., Hesai AT128, Velodyne/Ouster automotive lines) — hundreds to low thousands of dollars per unit at volume | The AV-stack reference robot's actual tier; NDT here typically runs as one of several localization sources feeding a safety-monitored fusion layer (§4), not the sole source of truth. |

No specialized "NDT chip" exists — the computation is a modest GEMM-adjacent workload (small dense
6x6 solves, point-parallel reductions) that any of the above GPU/SoC tiers handles comfortably; the
LiDAR sensor and its calibration are the dominant cost and engineering effort, not the compute.

## 3. Installation & integration — putting it on a real robot

- **Where it runs:** the same compute box running perception (01.xx/02.xx) and state estimation
  (04.xx) — typically the vehicle's/robot's main autonomy computer, not a dedicated microcontroller;
  NDT's per-tick cost (this project measures low-single-digit milliseconds of GPU time at teaching
  scale) fits comfortably inside a shared 10 Hz budget alongside other perception nodes.
- **OS / real-time constraints:** Linux (Ubuntu, typically) with ROS 2; NDT itself is not
  hard-real-time (a late pose degrades gracefully if the downstream EKF interpolates), but the NODE
  should run on a predictable schedule so the fusion filter's timing model stays valid.
- **The localization stack shape (NDT + EKF + odometry) — the actual production architecture:**

  ```
  [Wheel odometry / IMU] --(high-rate, drifts)-->  [EKF/UKF fusion, 04.xx]  --> pose estimate
  [NDT scan match, 10 Hz] --(low-rate, absolute)-->        ^
  [GNSS, outdoors only]   --(low-rate, absolute)-----------|
  ```

  NDT is NEVER the sole pose source in a real system — it is one absolute-position CORRECTION feeding
  a fusion filter that also integrates high-rate relative motion (wheel odometry, IMU) between NDT
  updates. This project's teaching core produces the "NDT correction" half of that picture only;
  04.01 (massive particle filter localization) and the broader 04.xx domain own the fusion half.
- **ROS 2 node/topic shape:** a `ndt_scan_matcher`-style node subscribing to a downsampled
  `sensor_msgs/PointCloud2` (fed by a 02.01-style voxel filter) and the prior map (loaded once at
  startup), publishing `geometry_msgs/PoseWithCovarianceStamped` at the matcher's rate — the
  covariance ideally reflecting the Hessian's own conditioning (THEORY.md's degeneracy report is a
  first-principles ingredient for that covariance estimate, not just a diagnostic).
- **Initialization sources — how NDT gets a first guess at all.** This project's basin study stands
  in for a question a real system answers ONCE at startup (and again if localization is ever LOST,
  §health-monitoring below): outdoors, GNSS (even a coarse, few-meter fix) seeds the first NDT
  guess; indoors or under GNSS-denied conditions, a PLACE-RECOGNITION descriptor match (02.11's scan
  context ring descriptor, named explicitly — its whole job is producing a coarse initial pose from
  a global scan comparison, no odometry required) or a known start pose (a docking station, a
  charging bay) seeds it instead. Once seeded, NDT only needs to correct SMALL per-tick drift, which
  is exactly why this project's own cohort finds the smallest-perturbation bin easiest (README
  "Expected output") — a real system is normally operating in that regime, not the wide blind search
  this project's basin study deliberately stresses.
- **Health monitoring — score-based localization-loss detection.** A production NDT node watches
  its OWN score/fitness metric (this project's `score` field, or a normalized transformation-
  likelihood metric) tick to tick: a sudden drop (points no longer fitting their voxels well) signals
  either a genuinely lost localization (kidnapped-robot problem), a stale/wrong map region, or heavy
  dynamic-object contamination — any of which should suppress that tick's NDT correction from
  reaching the fusion filter rather than feeding it a confidently-wrong pose. This project's
  `SCORE_SANITY` gate (score at truth beats every perturbed guess) and the `degenerate_axis` report
  are exactly the KIND of signal a real health monitor watches, generalized from "one committed
  scene" to "every tick, forever."
- **Map-format realities.** Real map files are versioned, checksummed (this project's own
  `data/README.md` SHA-256 table is the same discipline at teaching scale), and often TILED (a
  city-scale map split into loadable chunks by geographic cell, not one monolithic file) — this
  project's single small `map.bin` sidesteps tiling entirely, a documented scope reduction.
- **The safe testing ladder** (CLAUDE.md §1's sim-validated-only caveat, restated for this domain):
  simulation (this project's whole demo) → replay against RECORDED real sensor data (compare NDT's
  pose against a higher-fidelity reference trajectory, e.g. RTK-GNSS ground truth) → a static
  vehicle/robot with the localization stack running but NOT connected to motion command generation
  → low-speed, geofenced, human-supervised operation with an E-stop and a safety driver/operator →
  full deployment. This project's teaching core has climbed exactly none of these rungs.

## 4. Business & regulatory context

- **Who needs this:** any mobile robot or vehicle operating relative to a known map — warehouse
  AMRs (Amazon Robotics, Locus Robotics, Fetch/Zebra), autonomous shuttles and robotaxis, and
  Autoware-based AV stacks (the catalog bullet's own naming). Indoor service robots and agricultural
  robots (30.xx) use the same algorithm family at a smaller map scale.
- **Main players:** Autoware Foundation (open-source, the direct namesake of this project's catalog
  bullet), PCL (the reference open-source library), and every AV/AMR company's in-house localization
  stack (most build on one of the two above rather than starting from scratch — build-vs-buy,
  SYSTEM_DESIGN §5.3, usually favors buying/adapting the open-source core and differentiating on
  integration, health monitoring, and map operations).
- **Cost of getting it wrong:** a localization error that is CONFIDENT but wrong (exactly the
  "converged to the wrong local minimum" failure mode this project's own convergence study measures
  honestly, including the cases where NDT does NOT converge) is a safety-relevant failure — a vehicle
  or AMR that believes it is somewhere it is not can violate geofences, collide, or misroute. This is
  why health monitoring (§3) and multi-source fusion (never trusting NDT alone) are load-bearing
  production requirements, not optional polish.
- **Regulatory path (orientation only — see SYSTEM_DESIGN.md item 6's regulatory map for the fuller
  picture, and CLAUDE.md §1: nothing here is safety-certified).** For an AV stack, localization
  integrity feeds directly into the safety case under **ISO 26262** (functional safety) and,
  increasingly, **UL 4600** (safety case methodology for autonomous products) — a localization
  component typically needs a documented failure-mode analysis (what happens when NDT silently
  converges wrong?) and an independent monitor (an ASIL-rated cross-check, not just NDT's own score)
  before it can inform a safety-relevant decision. For an indoor AMR, **ISO 13482** (personal-care/
  service-robot safety) is the more relevant standard, with localization integrity feeding collision-
  avoidance rather than a full drive-by-wire safety case. None of this project's code has undergone
  any such process — it is a teaching implementation of the ALGORITHM a real safety case would need
  to wrap, not a component of one.
- **Where this work lives inside a robotics company** (SYSTEM_DESIGN.md item 5): the
  localization/state-estimation team, a sub-team of controls/autonomy — typical role titles
  "Localization Engineer" or "State Estimation Engineer" — working adjacent to mapping/SLAM
  (who build and maintain the map this project's teaching core treats as a given input, §1), the
  broader perception team (who hand it the filtered scan), and functional-safety/QA (who own the
  monitoring and safety-case work referenced above).
