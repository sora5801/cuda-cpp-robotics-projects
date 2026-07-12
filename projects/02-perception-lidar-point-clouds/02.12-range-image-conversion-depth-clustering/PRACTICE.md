# 02.12 — Range-image conversion + depth-clustering segmentation: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is purely computational (no mechanical part of its own), so this section describes the
physical carrier the code would serve: the **spinning mechanical LiDAR unit** whose driver output this
project's committed sample stands in for, and its **mounting** on a robot.

A mechanical spinning LiDAR is built around a rotating head carrying the laser/detector pairs, spun by
a small brushless motor (often with a slip ring or, in newer designs, inductive/optical power+data
coupling to avoid wear-prone brushes/wires crossing the rotating joint). The housing is a sealed
weatherproof enclosure (commonly IP67 or better for outdoor units) with an optical window — a
cylindrical band of clear polycarbonate the beams fire through, which must stay clean and scratch-free
(dust, mud, or condensation on this window directly degrades range accuracy and can create phantom
near-range returns). The unit mounts to the robot via a rigid bracket at a KNOWN, calibrated height and
orientation — this project's `SENSOR_HEIGHT_M` constant is exactly that calibrated mounting height, and
ground-removal's virtual-reference-point trick (`THEORY.md` "The math") depends on it being accurate;
a bracket that flexes or a mount that is not re-measured after a bump directly degrades ground-removal
quality on the real robot. Vibration isolation (rubber or elastomer mounts) matters because the spinning
head's own imbalance, plus vehicle vibration, both directly add angular jitter to every beam's true
firing direction — a real-world noise source `THEORY.md`'s Gaussian range-noise model does not capture
(it models range-axis noise only, the dominant term for time-of-flight, but a full sensor model would
add angular jitter too).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-12. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute this pipeline would run on:** the range-image conversion, ground removal, and depth
clustering stages are all lightweight (this project's own measured GPU time is well under a millisecond
on an RTX 2080 SUPER for a ~4,000-obstacle-point scene) — well within reach of an embedded GPU tier.

| Tier | Illustrative example | Notes |
|------|----------------------|-------|
| Research desktop | NVIDIA RTX 4070/4080-class discrete GPU, x86 host | This repo's own dev target; massive headroom over what this pipeline needs |
| Embedded/production | NVIDIA Jetson Orin NX/AGX (Ampere-class SM, 8–100+ sparse TOPS) | The realistic target for an actual robot; this pipeline's image-stencil regime (fixed-shape kernels, no spatial-hash memory overhead) is a good fit for Jetson-class memory bandwidth and power budgets, unlike the voxel-hash Euclidean comparison this project benchmarks against |
| MCU-class (not viable for this workload) | — | The depth-clustering algorithm needs a real GPU (or, at minimum, a multi-core CPU with SIMD) for real-time performance at real point counts; a bare MCU cannot run this pipeline at LiDAR frame rate |

**The sensor itself:** illustrative tiers, hobby to industrial:

| Tier | Illustrative example | Interface |
|------|----------------------|-----------|
| Hobby/research, 16-beam class | Velodyne VLP-16 (legacy, still common secondhand), RoboSense RS-LiDAR-16 | Ethernet (UDP packets), PTP or GPS-PPS time sync |
| Research/mid-tier, 32–64-beam | Ouster OS1/OS2, Hesai Pandar series | Ethernet, similar packet model with more beams |
| Industrial/safety-rated (a DIFFERENT category — 2-D safety scanners, not 3-D perception LiDAR) | SICK microScan3, Pilz safety laser scanners | Dedicated safety-rated I/O, not this project's data path at all |

**Networking/compute interconnect:** the driver's packets arrive over Gigabit Ethernet (the near-
universal choice for spinning LiDAR); the perception compute (Jetson-class or x86+dGPU) receives them
via a dedicated NIC or switch port, often on its own VLAN to isolate sensor traffic from the robot's
control-plane network.

## 3. Installation & integration — putting it on a real robot

**Where this code would run:** on the robot's perception compute tier (the "big" compute — Jetson
Orin-class or x86+dGPU — NOT the real-time safety controller or motor MCUs), as one node in a
perception pipeline, typically pinned to run immediately after the LiDAR driver node and before
tracking/costmap nodes.

**OS and real-time constraints:** Linux (Ubuntu, JetPack on Jetson), NOT a hard-real-time OS — this
pipeline's soft ~10–20 Hz / <100 ms budget (`README.md` "System context," `docs/SYSTEM_DESIGN.md` §1.1)
is comfortably met by a normal Linux scheduler with the GPU doing the heavy lifting; no RTOS is needed
at this layer (contrast the motor current loops several layers downstream, which DO need hard
real-time silicon).

**ROS 2 node/topic shape it would take:** a node subscribing to `sensor_msgs/PointCloud2` (or, for a
driver that already publishes range-image-organized data, a custom organized-cloud message) on a topic
like `/lidar/points_raw`, and publishing an obstacle list — e.g. `vision_msgs/Detection3DArray` or a
custom `ClusterArray` message carrying per-cluster centroid/count/AABB — on `/perception/lidar_clusters`,
feeding into tracking (04.xx) and costmap population (23.01's GPU costmaps) downstream, matching this
repo's message-shaped-struct convention (`docs/SYSTEM_DESIGN.md` §3.6).

**Bus/calibration/bring-up:** the LiDAR itself connects via Ethernet (UDP), typically time-synced to the
robot's clock via PTP or a shared GPS-PPS pulse so LiDAR timestamps align with IMU/odometry for sensor
fusion. Extrinsic calibration (the LiDAR's pose relative to the robot base frame) is a one-time
bring-up step (often solved with a calibration target or via a SLAM-based extrinsic refinement, 02.16's
territory) — this project's own sensor-frame convention (origin at the sensor) assumes that extrinsic is
applied downstream, not inside this pipeline.

**The safe hardware-testing ladder** (CLAUDE.md §1 applies in full — this project itself commands no
motion, but anything consuming its output eventually might): simulation (this project's own synthetic
scene, and higher-fidelity LiDAR simulators like 11.01's GPU ray-caster) → hardware-in-the-loop (real
LiDAR driver, simulated robot motion) → bench jig (real LiDAR mounted on a static test rig, verifying
ground-removal/clustering quality against hand-labeled real scans) → tethered/current-limited → free
running, with E-stop and speed/torque limits enforced at every rung once any downstream consumer
actually commands motion. This project's own outputs are never used to command hardware directly in this
repo — everything here is sim-validated only.

## 4. Business & regulatory context

**Who needs this:** any mobile robot or vehicle using 3-D LiDAR for obstacle perception — warehouse
AMRs, delivery robots, agricultural/field robots, and autonomous vehicles (`docs/SYSTEM_DESIGN.md` §2.1,
§2.5's reference robots). Range-image-native processing specifically matters for platforms with tight
compute/power budgets (mobile, battery-powered) where the "no neighbor search" latency and memory-
bandwidth advantage this project measures translates directly into either faster perception cycles or
cheaper compute hardware for the same cycle time.

**Commercial and open-source players:** Autoware Foundation (open-source AV stack with range-image-
style ground filtering and clustering nodes), Apollo (Baidu's open AV stack), and every AV/robotics
company running a LiDAR perception stack internally (most do not open-source their production
segmentation code, though the research it descends from — Bogoslavskyi & Stachniss and its many
follow-ups — is public). On the sensor side: Velodyne/Ouster/Hesai/RoboSense (spinning mechanical
LiDAR) and an active solid-state/MEMS LiDAR market (Livox, Innoviz, and others) whose non-spinning,
non-uniform scan patterns break the clean "ring x azimuth" grid this project assumes — an honest
limitation worth naming (see `README.md` "Limitations & honesty").

**What getting it wrong costs:** a missed or badly-fragmented obstacle cluster in a perception pipeline
can mean a collision (this project's `grazing_fragmentation` gate demonstrates a REAL failure mode: a
long wall or curb, viewed edge-on, can fragment into pieces small enough to fall below a naive
min-cluster-size noise filter and be dropped entirely — a genuinely safety-relevant effect a real system
must mitigate, e.g. with the range-image smoothing pre-pass `THEORY.md` "Where this sits in the real
world" names). A falsely-merged object (this project's Euclidean-comparison "merge" failure mode) can
under-count or mis-localize obstacles the same way.

**Regulatory path (orientation only — see `docs/SYSTEM_DESIGN.md` item 6's regulatory map, cited, not
reproduced here):** for a ground mobile robot, ISO 13482 (personal care robots) or the relevant
industrial-mobile-robot standards; for an AV, ISO 26262 (functional safety) and UL 4600 (safety case for
autonomous products) govern the perception stack this project's output would feed, with LiDAR-based
obstacle detection typically forming part of the safety CASE rather than being independently certified
component-by-component. This project computes no certified metric and makes no compliance claim — it is
a didactic reimplementation of a published research algorithm, not a certified perception component.

**Where this work lives inside a robotics company:** the Perception team (sometimes with a dedicated
"LiDAR perception" sub-team at larger organizations, distinct from camera/vision perception and from
sensor fusion / state estimation) owns this kind of pipeline; adjacent teams include Sensor
Engineering/Calibration (owns the LiDAR mounting, extrinsics, and driver integration this project's
input assumes), Simulation (owns the synthetic-data tooling this project's own generator is a small
example of), and Safety/Functional-Safety (owns the case that this perception stage's failure modes are
acceptably mitigated downstream) — `docs/SYSTEM_DESIGN.md` §5.1's org map.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
