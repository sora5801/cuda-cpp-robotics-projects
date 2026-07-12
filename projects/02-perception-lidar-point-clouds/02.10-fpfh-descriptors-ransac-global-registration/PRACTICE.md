# 02.10 — FPFH descriptors + RANSAC global registration: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is pure perception software — it has no dedicated mechanical/electrical subsystem of its
own the way an actuator or sensor-mount project would. The physical carrier is the **LiDAR unit(s) and
their mount** that produce the two point clouds being registered: a spinning mechanical LiDAR (Velodyne/
Ouster/Hesai-class, 16-128 channels) or a solid-state/MEMS unit, rigidly bolted to the robot chassis
(or, for the "merge two mapping sessions" use case, to two DIFFERENT robots or the same robot at two
different times). Construction concerns that matter to THIS project's correctness, not just to the
sensor generically: **mounting rigidity** — any flex between the LiDAR and the frame it is supposedly
rigidly attached to injects a small, unmodeled extra transform between "where the robot thinks the
sensor is" and "where the sensor actually is," which shows up as a bias in every downstream registration
result; **thermal drift** of the mount (aluminum brackets expand measurably over a robot's operating
temperature range, tens of microns per degree per 10 cm — usually below this project's centimeter-scale
tolerance floor, but not always, on a long-boom or actuated mount); and **vibration** during motion,
which this project's use case (event-triggered relocalization, not continuous tracking — see §3) mostly
sidesteps by design: registration typically runs when the robot is momentarily stationary or moving
slowly, not mid-stride on a legged platform.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute tier.** This pipeline's real cost driver is the brute-force KNN and descriptor-matching steps
(`O(n^2)` and `O(n_src*n_tgt*33)` — THEORY.md's "The algorithm"), which want a discrete GPU for anything
beyond a few thousand points. Realistic tiers:
- **Research/dev bench**: an x86 workstation with a discrete NVIDIA GPU (this project's own reference
  machine class, RTX-2080-SUPER-and-up) — where this project's demo actually ran.
  Verify current: NVIDIA product pages.
- **Embedded/onboard (AMR-class)**: NVIDIA Jetson Orin NX/AGX (Ampere-class integrated GPU, 8-16 GB
  unified memory) — enough for this project's point-count scale (thousands per scan) comfortably;
  millions-of-points submap registration (02.09's target scale) would want the Orin AGX's larger SM
  count. Verify current: NVIDIA Jetson product pages.
- **Cost-constrained/no-GPU fallback**: a multi-core x86/ARM CPU running PCL's OpenMP-parallel
  `FPFHEstimation`/`SampleConsensusPrerejective` — slower, but a legitimate fallback for a robot whose
  compute budget cannot spare a GPU for an EVENT-DRIVEN (not continuous) computation.

**Sensors.** Any LiDAR whose driver publishes a `sensor_msgs/PointCloud2`-shaped point cloud with
per-point range/position works as this pipeline's input — the algorithm has no dependency on channel
count or scan pattern beyond "enough points to describe local geometry" (this project's own `k=20`
neighborhood floor). Illustrative tiers: hobby/research (Livox Mid-360-class solid-state LiDAR, a few
hundred USD, sparser/noisier returns), industrial/AMR (Ouster OS1/OS0-class 32-128 channel mechanical
LiDAR, low-thousands USD), automotive-grade (Hesai/Velodyne automotive-qualified units, higher cost,
tighter range/noise specs and functional-safety documentation). Verify current: manufacturer datasheets.

**No dedicated actuation/power silicon.** This project produces a POSE (a coordinate transform), not a
motor command — it has no actuator-chain BOM of its own (motor-control MCUs, gate drivers, current-sense
amps are N/A here; see PRACTICE.md of an actuator project, e.g.
[24.01](../../24-actuators-motors/24.01-magnetostatic-fea-sweeps/PRACTICE.md), for that BOM shape).

## 3. Installation & integration — putting it on a real robot

**Where this runs.** On the robot's main perception/autonomy compute (the same box running SLAM,
planning, and perception — not a dedicated microcontroller; this pipeline's KNN/matching/RANSAC steps
are exactly the kind of throughput-hungry, latency-tolerant work a Jetson-class or x86+dGPU box handles,
never a real-time safety MCU). OS: Linux (Ubuntu, the ROS 2 standard target) with the CUDA driver stack;
no hard real-time constraint on THIS component specifically (see the rate discussion below), though it
shares a machine with components that do have one.

**The ROS 2 shape.** A `global_registration` node subscribing to two `sensor_msgs/PointCloud2` topics
(a live scan and a target map/submap, or two independently-recorded sessions' point clouds) and an
optional trigger topic/service (`std_srvs/Trigger`-shaped, or a `relocalize` action) that fires the
pipeline on demand rather than every frame — see the rate discussion in README "System context": this
is fundamentally an EVENT-driven service, not a periodic publisher. Output: a `geometry_msgs/
PoseWithCovarianceStamped` (the recovered `T_target_source`, with a covariance this teaching version does
not compute — a real system would derive one from the RANSAC inlier count and residual spread, or run a
proper Hessian-based uncertainty estimate on the final ICP system) published to whatever consumes it —
typically a pose-graph SLAM back-end as a new edge, or a localization filter as a correction/reset.

**Calibration & bring-up.** The only "calibration" this pipeline itself needs is that its two input point
clouds are ALREADY in a consistent SENSOR-LOCAL frame (standard LiDAR intrinsic calibration, out of
scope here) — it explicitly does NOT need extrinsic calibration to another sensor or a prior pose
estimate, which is the entire point. What DOES need bring-up tuning per real deployment: the RANSAC
inlier threshold (`kRansacInlierThresholdM`, tuned here to this project's synthetic noise level) and the
descriptor-matching ratio-test threshold (`kMatchRatioMax`) both want re-tuning against a real sensor's
actual noise floor and a real environment's actual self-similarity (a warehouse full of identical
pallet racks is FAR more self-similar than this project's one-room synthetic scene, and would likely
need a tighter ratio test and a larger RANSAC budget — `THEORY.md`'s honest discussion of the measured
~10% inlier ratio applies with a vengeance to a real repetitive industrial environment).

**The safe hardware-testing ladder.** Because this component only ever PRODUCES a pose estimate — it
never directly commands actuators — its own testing ladder is shorter than a controller's, but the
CONSUMER of its output (a planner, a localization filter that a motion controller trusts) absolutely
needs the full ladder: simulation (this project's own demo, entirely) → HIL/replay against recorded real
sensor logs (does the recovered pose match a known ground truth from a motion-capture rig or surveyed
markers?) → bench/tethered testing on the real robot with the recovered pose ONLY LOGGED, never acted on
→ current-limited/geofenced testing where the pose feeds a planner with tight velocity/geofence limits
→ free running, only after the false-positive rate (RANSAC converging on a plausible-looking WRONG
transform — a real, non-hypothetical failure mode; see THEORY.md's low-overlap discussion) has been
characterized on real data. Everything in this repository is sim-validated only (CLAUDE.md §1); a real
deployment ladder is sketched here, not walked.

## 4. Business & regulatory context

**Who needs this.** Any mobile robot or AV that must recover from "I don't know where I am" — after
power-on with no stored pose, after a localization fault (wheel slip, GPS dropout, sensor occlusion), or
when merging maps built in separate sessions or by separate robots (multi-robot fleets building a shared
map, cited in `docs/SYSTEM_DESIGN.md` domain 22/swarms). Warehouse AMR fleets (Locus Robotics, 6 River
Systems, Fetch-class platforms), autonomous-vehicle stacks doing HD-map-relative localization without
GPS (parking garages, tunnels), and any SLAM stack's loop-closure back-end (Cartographer, RTAB-Map,
LIO-SAM-class systems) all ship SOME version of this capability, commercial or open-source.

**Main players.** Open-source: PCL and Open3D (both cited in README "Prior art") ship production-grade
implementations of exactly this pipeline shape and are the DE FACTO standard most commercial stacks
build on or benchmark against, rather than reimplementing from scratch. Commercial: SLAM/localization
vendors bundle proprietary variants (often learned-descriptor-based, per THEORY.md's "research frontier"
discussion) as part of a fuller autonomy stack; few companies sell "global registration" as a
standalone product — it is almost always a component inside a larger mapping/localization offering.

**What getting it wrong costs.** A FALSE POSITIVE (RANSAC converging confidently on a wrong transform,
this project's `low_overlap` cohort's honest failure mode writ large) is the dangerous direction: a
robot that believes a wrong pose can drive confidently into a wall, off a loading dock edge, or into
another robot's path — the failure is silent (no error is raised; the system reports high confidence in
a wrong answer) unless the consuming system cross-checks the recovered pose against another source
(odometry consistency, a plausibility gate) before acting on it. A FALSE NEGATIVE (registration honestly
reports it could not align, this project's `icp_negative_control`/`low_overlap` cohorts) is safer but
costs uptime — a fleet robot stuck unable to relocalize is a support ticket, not a safety incident.

**Regulatory path.** This component sits inside the broader autonomy stack's regulatory umbrella rather
than having its own dedicated standard: for a warehouse AMR, the relevant framework is ISO 13482
(personal-care/service robot safety, though most warehouse AMRs are industrial mobile robots governed
more directly by ANSI/RIA R15.08 or the older ISO 3691-4 for driverless industrial trucks); for an AV
stack, ISO 26262 (functional safety) and UL 4600 (safety case framework for autonomous products) —
neither of which certifies "the registration algorithm" in isolation, but both require the fleet
operator to demonstrate the FULL localization pipeline (of which this is one component) meets a
documented safety case, including its failure-mode behavior on exactly the false-positive scenario
above (`docs/SYSTEM_DESIGN.md` item 6's regulatory map, cited). This is orientation, not compliance
guidance.

**Where the work lives in a company.** Localization/mapping team (part of the perception/autonomy
org, `docs/SYSTEM_DESIGN.md` item 5), typically titled "SLAM engineer," "localization engineer," or
"perception engineer (mapping)" — closely adjacent to the broader perception team (who own the
descriptor/feature pipelines this project reuses conventions from, e.g. 01.04/01.05's lineage) and the
planning/autonomy team who CONSUME this component's output pose and must design their own plausibility
checks around it.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
