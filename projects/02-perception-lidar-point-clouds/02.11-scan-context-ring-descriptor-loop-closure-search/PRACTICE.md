# 02.11 — Scan Context / ring-descriptor loop-closure search: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-11. All parts, vendors, and standards named below are **illustrative examples,
> never endorsements**; verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project is algorithm, not hardware — N/A in the literal "what bolts to what" sense — but the
CONSEQUENCE of getting it wrong is a physical, operational failure mode worth building intuition for
before it ever runs on a real robot.

**A false loop closure corrupts the map irreversibly, in the moment it happens.** A pose-graph
optimizer treats a confirmed loop closure as a hard geometric CONSTRAINT: "keyframe 340's pose and
keyframe 12's pose must be related by (approximately) this transform." If that constraint is WRONG — the
detector matched two different, similar-looking corridors — the optimizer will happily bend the ENTIRE
trajectory between those two keyframes to satisfy a false constraint, warping a straight hallway into a
curve, or worse, snapping two genuinely distant parts of the map together on top of each other. Unlike a
single bad odometry reading (which the optimizer can down-weight or the next good measurement can
gradually correct), a false loop-closure edge is typically ADDED PERMANENTLY to the graph and actively
FIGHTS every subsequent correct measurement — this is the "war story shape" every SLAM engineer learns
early: a robot that drove confidently until "the map folded in on itself" at one specific corridor
junction, and the post-mortem trail always leads back to one bad loop-closure edge from hours earlier.
This is exactly why `negative_cohort` — zero false closures on never-revisited places — is gated at a
hard `== 0`, not a floor with slack, while `loop_detection` recall is allowed to be well under 1.0: a
missed loop closure costs you accuracy; a false one costs you the map.

**Threshold tuning is a per-environment, per-sensor, ongoing exercise, not a one-time constant.** This
project's own operating threshold (`kScDistanceThreshold`) was chosen from a diagnostic sweep of ONE
synthetic world (`THEORY.md` "how we verify correctness") — deploying the same descriptor in a
different building (more repetitive, e.g. a warehouse with identical racking every 3 m; or less
repetitive, e.g. an outdoor campus) would need the SAME sweep re-run against that environment's own
positive/negative distance distributions, not a copy-pasted number. Production systems typically log
every detection's distance and shift for weeks after deployment and re-tune the threshold (or retrain a
learned re-ranker) once enough field data accumulates — threshold tuning is a fleet-operations
responsibility, not a one-time engineering sign-off.

## 2. Real hardware — chips, parts, illustrative BOM

This is a perception ALGORITHM, not a new physical sensor — it consumes whatever LiDAR the platform
already carries for mapping/obstacle-avoidance. The relevant hardware question is COMPUTE, not sensing.

| Tier | Compute | Illustrative cost | Notes |
|------|---------|--------------------|-------|
| Hobby/research | Jetson Orin Nano / NUC + no dGPU | $200–500 | This project's whole workload (a few thousand points, a 20x60 matrix, a search over a few hundred candidates) is small enough to run on CPU alone in most single-robot deployments; the GPU path here is a teaching exercise in the RIGHT pattern for when the database and point counts scale up (fleet-scale multi-session mapping, §3 below). |
| Typical AMR/service robot | Jetson Orin NX/AGX (shared with perception/planning) | $700–2000 (module) | Loop closure runs as ONE thread on the SAME SoC that already runs the rest of the perception stack — no dedicated silicon. |
| AV / high-end mapping rig | x86 + discrete RTX-class GPU, or multiple Orin AGX modules | $2,000–10,000+ | Database sizes (hundreds of thousands of keyframes across multi-hour drives) make the GPU shift-search this project teaches genuinely load-bearing, not just illustrative. |

**Sensor side** (already assumed present for other purposes — not procured FOR this project): the LiDAR
this descriptor consumes is whatever the platform's mapping stack uses — hobby-tier (RPLiDAR-class 2-D,
~$100–400, though Scan Context wants the vertical structure a multi-line sensor provides), research-tier
(Livox Mid-360, Ouster OS0/OS1, ~$1,000–6,000), or industrial/AV-tier (Velodyne/Hesai/Ouster 32–128
channel, $5,000–20,000+). More channels and a wider vertical field of view directly improve descriptor
quality (this project's own synthetic sensor — 16 channels, -18° to +12° — is deliberately modest;
`THEORY.md`'s numerics section shows how much of the descriptor's occupied cells come from just the
steepest few channels).

## 3. Installation & integration — putting it on a real robot

**Where this runs:** the SAME compute module that runs the rest of the SLAM/perception stack (§2) — a
background THREAD or low-priority PROCESS, not a hard-real-time loop; it has no business anywhere near
the motor-control MCU tier of `docs/SYSTEM_DESIGN.md` §6.1's hardware diagram. OS: Linux (Ubuntu +
real-time-adjacent kernel is typical, but this workload has no hard deadline — a missed loop-closure
cycle just means the check runs on the next keyframe instead).

**ROS 2 node shape** (`docs/SYSTEM_DESIGN.md` §3.6 message conventions): a `loop_closure_node`
subscribing to `PointCloud2` keyframes (published by the front-end at keyframe rate, not scan rate) and
the current pose estimate; publishing a custom `LoopClosureCandidate` message
(`query_stamp, match_stamp, relative_yaw_estimate, distance_score`) consumed by the pose-graph node
(the `05.xx` SLAM domain). In a real deployment this candidate is NEVER trusted directly — it is always
routed through geometric verification (project [02.10](../02.10-fpfh-descriptors-ransac-global-registration/README.md)'s
lineage, or a compact ICP as this project's own `yaw_handoff` illustrates) before becoming a pose-graph
edge; this project's own `main.cu` never claims otherwise (README "Limitations").

**Calibration/bring-up:** none specific to this algorithm beyond the sensor's own extrinsic calibration
(LiDAR-to-base transform) already required by every other consumer of its point clouds — Scan Context has
no internal parameters that need per-robot tuning beyond the OPERATING THRESHOLD (§1 above), which is an
environment property, not a hardware one.

**The safe testing ladder** (CLAUDE.md §1's caveat applies in full — nothing in this repo is
safety-certified, and a place-recognition system's output can influence a robot's belief about its own
position, which indirectly influences motion):

1. **Simulation** — exactly this project's demo: synthetic world, known ground truth, offline.
2. **Recorded-data replay** — run the SAME pipeline against a rosbag of a real drive with independently
   surveyed keyframe poses (e.g. from a fixed total-station or RTK-GPS reference), still fully offline.
3. **Bench/tethered, map-building disabled** — run live on the robot's real sensor while stationary or
   hand-carried, LOGGING detections but not letting them feed the live pose graph — a human reviews every
   candidate against the recorded video/point-cloud before trusting the pipeline's verdict.
4. **Live, map-building enabled, geometric verification mandatory** — only after step 3 shows a clean
   negative-cohort record on REAL data, and only ever gated behind the verification stage named above,
   never a raw descriptor-distance threshold alone.

## 4. Business & regulatory context

**Who needs this.** Any company shipping a robot that operates over an area larger than one sensor's
range for longer than one battery charge needs SOME loop-closure mechanism — this spans warehouse AMR
makers (Locus Robotics, 6 River Systems, Fetch/Zebra-class players), autonomous-vehicle mapping/
localization stacks (every AV company's HD-map-relative localization has a "did GPS/IMU drift too far,
re-anchor against the map" fallback that is architecturally this exact problem), and any long-duration
field/mapping robot (agricultural, inspection, mining). **Main players:** open-source SLAM stacks
(ORB-SLAM3, RTAB-Map, LIO-SAM/FAST-LIO's loop-closure extensions — several ship a Scan-Context-family
detector directly) and every commercial SLAM vendor building on top of similar building blocks; there is
no separate "loop-closure vendor" market — it is a component inside a SLAM/localization product, not a
standalone product category.

**What getting it wrong costs.** Beyond the map-corruption failure mode (§1): a warehouse AMR whose map
silently warps can start commanding motion based on a WRONG belief about wall/rack positions — a
near-miss or collision liability event, and in a fleet, a corrupted SHARED map (multi-session/multi-robot
merge, project [22.xx](../../22-multi-robot-swarms/README.md) by name) can propagate the error to every
robot that merges against it. Recall/precision are not abstract ML metrics here — precision failures are
the safety-relevant direction (§1), recall failures are an efficiency/uptime cost (a robot that never
closes loops just accumulates drift and eventually needs a manual re-localization).

**Regulatory path.** Loop closure itself is not named in any standard, but the robots it runs on are:
for a warehouse/service AMR sharing space with people, **ISO 13482** (personal-care/service-robot hazard
analysis, `docs/SYSTEM_DESIGN.md` §6.2) is the relevant orientation point — a corrupted map that leads to
unsafe motion falls under the SAME hazard-analysis umbrella as any other localization failure mode. For
an autonomous-vehicle stack using loop closure as a GPS-degraded localization fallback, **ISO 26262**
(functional safety of the E/E system) and **UL 4600** (the safety-case standard for full autonomy)
frame how a localization-integrity failure must be argued and evidenced. Both citations are orientation
only, per `docs/SYSTEM_DESIGN.md` §6.2's label — not a compliance claim.

**Where this work lives in a company.** SLAM/localization, inside controls & autonomy
(`docs/SYSTEM_DESIGN.md` §5.1) — typical adjacent roles: perception engineer (owns the upstream keyframe
pipeline), mapping/SLAM engineer (owns this component and the pose-graph backend), and — once deployed —
fleet-operations/reliability engineering (owns the field monitoring that catches a threshold that has
drifted out of tune, §1). Multi-session/multi-robot map merging (domain
[22](../../22-multi-robot-swarms/README.md)) is this project's natural scaling direction and typically
sits with the same team once a fleet, not a single robot, is being mapped.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice.*
