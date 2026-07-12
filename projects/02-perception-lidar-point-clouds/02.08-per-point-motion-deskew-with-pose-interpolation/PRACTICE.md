# 02.08 — Per-point motion deskew with pose interpolation: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is pure software sitting between two physical subsystems — the spinning LiDAR's firing
electronics and the pose source's sensors — so its "construction" is really the construction of the
TIMING PATH those two subsystems must share, honestly, including where it breaks:

- **Where a per-point timestamp really comes from.** A mechanical LiDAR's firmware knows its own motor
  encoder position at every firing instant (that IS the azimuth) and stamps each firing group with a
  timestamp derived from an internal clock. That internal clock must be disciplined to the SAME time
  base the pose source uses, or "per-point time" and "pose sample time" are comparing apples to
  oranges. Production sensors (Velodyne, Ouster, Hesai) support **PTP** (IEEE 1588 Precision Time
  Protocol) over Ethernet specifically to solve this: the sensor's clock is continuously disciplined to
  a shared grandmaster clock on the vehicle's network, giving sub-microsecond cross-sensor timestamp
  agreement — the SAME synchronization problem [`01.18`](../../../01-perception-cameras-vision/01.18-depth-completion/PRACTICE.md)
  and [`01.20`](../../../01-perception-cameras-vision/01.20-time-of-flight-raw-processing/PRACTICE.md)
  discuss for their own depth sensors' hardware timing paths — read those for continuity on how a real
  sensor's timestamp electronics are built and disciplined.
- **What breaks in the field.** Without PTP (many lower-cost or older LiDAR units, and many hobby/
  research rigs), the only synchronization available is the HOST computer's arrival-time stamp on each
  UDP packet — which includes OS scheduling jitter, network stack latency, and USB/Ethernet driver
  buffering, easily tens of milliseconds of uncertainty. Using an arrival-time stamp AS IF it were the
  firing-time this project assumes silently reintroduces the exact distortion deskew exists to remove —
  a subtle, easy-to-miss integration bug, not a code bug.
- **The pose source's own construction.** The trajectory this project consumes is itself built by a
  physical IMU (accelerometer + gyroscope MEMS die) sampled at a fixed rate and fused (typically with
  wheel odometry or a prior LiDAR/visual estimate) into the pose stream — see domain 04's projects for
  that construction. The IMU's OWN timestamp discipline (its sample clock vs. the LiDAR's) is a second
  synchronization boundary, not a given.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| LiDAR unit | Hobby/research: Velodyne VLP-16 / Ouster OS0-32 class (~US$4k–12k); industrial/AV: Hesai Pandar / Ouster OS1-128 class with PTP support (~US$5k–20k+) | Produces the raw per-point returns and (on PTP-capable units) firing timestamps this project consumes |
| Time sync hardware | GPS-disciplined oscillator / PTP grandmaster clock (on-vehicle network switch with PTP support, or a dedicated timing card) | Disciplines the LiDAR's and IMU's clocks to a common time base — the physical prerequisite for "per-point timestamp" to mean anything across sensors |
| Pose-source IMU | Hobby: MPU-6050/ICM-20948-class MEMS IMU (~US$5–30); research/industrial: tactical-grade IMU (e.g. an Analog Devices ADIS16xxx-class part, ~US$500–3k); AV-grade: a full INS with integrated GNSS (~US$5k–50k+) | The raw angular-rate/acceleration stream domain 04's fusion projects turn into the pose trajectory this project interpolates |
| Compute for the deskew kernel | Jetson Orin class (embedded, on-robot) / x86 + discrete RTX (reference machine: RTX 2080 SUPER, this project's measured numbers) | Runs `deskew_kernel` inside the perception pipeline's per-sweep budget |
| Network fabric | Automotive/industrial Ethernet (100BASE-T1 or 1000BASE-T1 for AV-grade; plain Gigabit Ethernet for research rigs), PTP-aware switch | Carries the LiDAR's UDP point stream and (where used) PTP sync traffic |

## 3. Installation & integration — putting it on a real robot

- **Process shape.** On a ROS 2 robot this project's logic runs as a dedicated `PointCloud2` → 
  `PointCloud2` node (or a library call inside the LiDAR driver's own composable-node pipeline):
  subscribes the raw scan (with its per-point `time` field, if the driver populates it — see §1) and
  the pose source's stream, publishes the deskewed cloud downstream. The demo's structure — load once,
  interpolate per point, re-project — maps directly onto that node's per-sweep callback.
- **TF time-travel, honestly.** The idiomatic ROS 2 way to ask "where was the robot at time `t_i`?" is
  `tf2_ros::Buffer::lookupTransform(..., t_i)` against a buffered TF tree — this is EXACTLY the
  `interpolate_pose` this project implements by hand, wrapped in a general-purpose transform library
  that also handles multi-frame chains, extrapolation policies, and buffer-duration limits this
  project's fixed 2-sample-or-21-sample array does not need to. The honest caveat: `lookupTransform`
  can only interpolate WITHIN its buffered time window and FAILS (throws) if `t_i` falls outside it or
  before the buffer has enough history — a real integration must size that buffer to comfortably exceed
  one sweep period, and must decide (and test) what happens when a lookup fails mid-sweep (this
  project's "no data before/after the array — clamp" choice, `kernels.cuh`'s `find_bracket_index`, is
  one defensible default; silently dropping the point is another).
- **Pose-source latency and extrapolation, honestly.** This project assumes the pose trajectory
  COVERING the sweep is already available when deskew runs — in reality, a live estimator's LATEST pose
  sample often lags the LiDAR's most recent firing by the estimator's own processing latency (a few
  milliseconds to tens of milliseconds, filter-dependent). Two honest choices, both used in production:
  (a) delay deskew slightly, buffering the sweep until enough FUTURE pose samples have arrived to
  interpolate (not extrapolate) every point — this project's approach, and the more ACCURATE one; or
  (b) EXTRAPOLATE the trailing points' poses from the latest available samples — lower latency, lower
  accuracy for exactly the points that need it most (the freshest ones). FAST-LIO2-class tightly-coupled
  systems (THEORY.md "Where this sits in the real world") sidestep this tension by using the IMU's raw,
  always-current propagation instead of a filtered pose stream.
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1):**
  1. *Simulation* — this demo's synthetic cohorts, plus a model-mismatch run (feed a WRONG trajectory
     on purpose and confirm the restoration gates catch it).
  2. *Replay* — recorded rosbag data from a real LiDAR + real pose source, offline, comparing
     deskewed-vs-undeskewed registration quality on real geometry (a real building corridor, a real
     vehicle drive) — no robot motion is commanded by this stage.
  3. *On-vehicle, logging only* — run the node on the real robot's compute while it is driven (by a
     human, or by an already-validated stack), log the deskewed output, inspect offline; this project's
     output never commands anything, so this rung carries no motion-safety risk beyond the ordinary
     operation of a robot someone is already, separately, authorized to drive.
  4. *On-vehicle, in the loop* — the deskewed cloud now feeds the LIVE registration/mapping/planning
     stack; standard staged rollout (canary robot, fleet-wide only after validation) applies from here,
     same as any perception-pipeline change (CLAUDE.md §1's real-hardware caveat, restated: nothing in
     this project is safety-certified, and its role feeding localization means a regression here can
     degrade a robot's understanding of where it is, even though it commands no motion directly).
- **Motion-compensation on/off A-B testing discipline.** Because this project's effect is invisible at
  low speed and dramatic at high speed/high yaw-rate (THEORY.md's distortion table), the single most
  useful integration test is a scripted A/B: run the SAME recorded drive through the registration/
  mapping stack twice, once with deskew enabled and once with it forced off, and diff the resulting map
  or trajectory estimate. A real regression (a sign error, a stale trajectory, a timestamp unit
  mismatch) usually shows up as "the A/B comparison stopped showing a difference" or "the deskewed run
  is now WORSE than the raw one" long before it shows up as an obvious crash.

## 4. Business & regulatory context

- **Who needs this.** Any product with a spinning or scanning LiDAR moving faster than a few meters per
  second during the scan — warehouse AMRs above walking pace, delivery robots, autonomous vehicles, and
  legged/wheeled platforms doing dynamic maneuvers. Below that speed (a slow-moving indoor AMR) the
  distortion is small enough that many products ship without dedicated deskew and simply accept the
  error — a legitimate build-vs-skip engineering call this project's own distortion table (THEORY.md)
  gives the reader the tools to make for their own platform.
- **The players.** Every major AV/robotics LiDAR vendor (Velodyne, Ouster, Hesai, Livox) ships firmware
  or driver-level support for per-point timestamps specifically so downstream stacks CAN deskew;
  motion-compensation is a standard, often undocumented, stage inside commercial SLAM/localization
  stacks (Cartographer, LIO-SAM-descended internal tools, and every AV company's proprietary
  localization module). It is rarely sold as a standalone product — it is glue code that lives inside a
  perception or localization team's pipeline, not a market of its own.
- **What getting it wrong costs.** A silently-undistorted (or wrongly-distorted, e.g. a sign error in
  the relative transform) point cloud degrades every downstream consumer's accuracy without necessarily
  crashing anything — the failure mode is a slowly-drifting map, a localization estimate that is
  confidently wrong, or a registration algorithm that converges to the wrong answer under partial
  occlusion. In a warehouse AMR this shows up as navigation drift and near-misses; in an AV stack it is
  the kind of latent perception bug that regulatory post-incident review (see below) specifically looks
  for, because it degrades the SAME localization integrity that safety cases assume is trustworthy.
- **Regulatory.** This project's output feeds localization and mapping, not actuation directly, but
  localization integrity is a load-bearing assumption of nearly every downstream safety case — the
  `docs/SYSTEM_DESIGN.md` item-6 regulatory map's AV entries (ISO 26262 functional safety, UL 4600 for
  autonomous system safety cases) and industrial/service-robot entries (ISO 10218, ISO 13482) all
  ultimately depend on the robot's belief about where it is being accurate; motion deskew is one of the
  many unglamorous correctness steps that belief depends on. This is orientation, not a compliance
  claim — see the SYSTEM_DESIGN map for the actual standards landscape.
- **Owning team.** Perception (SYSTEM_DESIGN item 5) — specifically the driver/sensor-integration
  sub-team that owns the boundary between raw sensor data and the rest of the stack; adjacent teams:
  localization/SLAM (the direct consumer, and often the team that FLAGS a deskew bug by noticing map
  quality regress), and embedded/hardware (owns the PTP/timing infrastructure §1 depends on existing at
  all).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
