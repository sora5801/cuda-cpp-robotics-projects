# 01.18 — Depth completion: sparse LiDAR + RGB → dense depth: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

*Sections dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project's algorithm has no moving parts of its own — but it exists entirely to serve one physical
subsystem: the **rigidly co-mounted camera–LiDAR pair** this project's fixed extrinsic
(`kTCameraLidar`) describes. Getting that pair built and calibrated well is most of what makes depth
completion work at all; a loose or drifting extrinsic corrupts the projection step before any algorithm
gets a chance.

- **Co-mounting.** Camera and LiDAR are typically bolted to a common rigid bracket (machined aluminum
  or a stiff printed/molded polymer for lower-cost platforms) rather than to two separately-flexing
  points on the chassis — any relative flex between the two sensors under vibration or thermal
  expansion shows up directly as projection error, i.e. sparse points landing on the wrong pixels. This
  is the SAME fixturing story [`01.17`](../01.17-camera-lidar-camera-camera-extrinsic-calibration)'s
  PRACTICE.md covers for the calibration rig — this project's mount is the thing 01.17 calibrates.
- **Time synchronization.** A spinning LiDAR and a rolling/global-shutter camera sample the world at
  different instants; without synchronization, a fast-moving object projects to the wrong pixel purely
  from motion during the time offset. Production rigs distribute a common time base — **PTP (IEEE 1588)
  over Ethernet** or a **PPS (pulse-per-second) hardware trigger** wired to both devices — so every
  LiDAR packet and camera frame carries a timestamp on the SAME clock; software-only synchronization
  (matching the nearest timestamps post-hoc) is a common but noisier fallback used when hardware sync
  is unavailable.
- **Lens/FOV matching.** This project's synthetic camera has a horizontal FOV of
  `2·atan(cx/fx) ≈ 56.5°` and vertical `2·atan(cy/fy) ≈ 44.7°`; the LiDAR sweeps a wider azimuth wedge
  (`±32°` used here) specifically so its coverage does not fall short of the camera's edges. On a real
  rig, choosing a camera lens (focal length) to roughly match or exceed the LiDAR's useful angular
  resolution avoids two failure modes: too WIDE a camera FOV wastes LiDAR density that will be spread
  over image area with no LiDAR coverage at all; too NARROW wastes LiDAR returns that fall outside the
  image entirely.
- **What breaks in the field.** Bracket loosening from vibration (the extrinsic silently drifts —
  01.17's recalibration workflow exists for exactly this); LiDAR window/lens contamination (dust, rain,
  bug splatter) that attenuates or blocks specific azimuth/elevation cells, creating LOCALIZED sparsity
  worse than this project's uniform beam model; and thermal defocus of the camera lens shifting the
  effective `fx/fy/cx/cy` slightly from their calibrated values.

## 2. Real hardware — chips, parts, illustrative BOM

The compute this project's algorithm would run on, and the sensors it would consume, span a wide cost
range. Tiers below are illustrative starting points, not recommendations — **verify current
availability/pricing before relying on any of them.**

| Tier | Compute | LiDAR | Camera |
|------|---------|-------|--------|
| Hobby/research | Jetson Orin Nano / NX (a few TOPS, ~10-25 W) | Livox Mid-360 or similar (non-mechanical, ~$500-1000 class) | USB3 global-shutter machine-vision camera (~$150-400) |
| Prosumer/prototype | Jetson AGX Orin (~275 TOPS INT8, ~15-60 W) | 16- or 32-beam mechanical spinning LiDAR (Velodyne/Hesai/RoboSense class, low-$k) | GigE or automotive-grade camera module with hardware trigger |
| Industrial/AV | x86 + discrete RTX-class GPU, redundant compute lanes | 64-128-beam automotive LiDAR (Hesai/Ouster/Velodyne Alpha class, several-$k to tens-of-$k) | Automotive image-sensor camera (e.g. Sony IMX-series) with a dedicated ISP |

The pipeline itself is light enough (README "System context": single-digit-to-low-teens ms on a
desktop RTX 2080 SUPER at 160×120) that at PRODUCTION resolution (1920×1080+) the compute tier matters
far more than it does for this teaching-scale demo — a Jetson-class embedded GPU is the realistic
target for an actual vehicle/AMR, not a desktop discrete card.

**Silicon this project's inputs come from, further upstream:** the LiDAR's own receive chain (avalanche
photodiodes or SPAD arrays, a time-to-digital converter ASIC) and the camera's image sensor + ISP are
each their own hardware story (see the sibling perception projects in domains 01/02/03 for those).
This project starts downstream of both, consuming an already-digitized point cloud and image.

## 3. Installation & integration — putting it on a real robot

- **Where this runs.** On the SAME compute tier that runs the rest of the perception stack — typically
  the vehicle/robot's main perception computer (a Jetson-class SoC or an x86+GPU box), NOT a
  microcontroller; the diffusion PDE's iteration count and the IDW search both need a real GPU to hit
  a 10-20 Hz LiDAR-rate budget at production resolution.
- **OS / real-time.** Ubuntu + ROS 2 is the dominant stack for this kind of perception node; the node
  itself is soft-real-time (missing one cycle produces a stale depth map, not a safety fault by itself —
  contrast a motor current-control loop, which IS hard-real-time), but the WHOLE perception pipeline it
  feeds typically runs under a bounded-latency scheduling policy on a Linux `PREEMPT_RT` kernel or
  equivalent.
- **ROS 2 node/topic shape.** A depth-completion node would subscribe to `sensor_msgs/msg/PointCloud2`
  (LiDAR) and `sensor_msgs/msg/Image` (camera, ideally synchronized via `message_filters`' approximate-
  time policy using the hardware-synchronized timestamps from §1), and publish
  `sensor_msgs/msg/Image` (32-bit float depth, `encoding: "32FC1"`) — this project's own message-shaped
  interface conventions (SYSTEM_DESIGN.md item 3) map directly onto these standard types.
- **Bus/interface.** LiDAR: typically Ethernet (UDP packet stream) for spinning mechanical units, or a
  vendor SPI/Ethernet link for solid-state units. Camera: USB3, GigE Vision, or a MIPI CSI-2 ribbon
  direct to the SoC (Jetson-class boards commonly take camera modules over CSI-2 for lower latency than
  USB).
- **Calibration & bring-up.** Intrinsics first (checkerboard/ChArUco — 01.16), extrinsics second
  (01.17's reprojection-error minimization against fiducials seen by both sensors), THEN this project's
  pipeline — running depth completion against a mis-calibrated extrinsic produces confidently-wrong
  dense depth, which is worse for a downstream planner than an honestly sparse point cloud.
- **The safe testing ladder** (every rung, every project in this repo — CLAUDE.md §1): simulation (this
  project's synthetic scene) → hardware-in-the-loop (real sensor data replayed through the same code,
  no actuation) → bench jig with the sensor pair mounted but the vehicle stationary and any actuation
  E-stopped/current-limited → tethered/low-speed real-world testing → free running. This project's own
  output is a PERCEPTION map, not a command — but if it ever feeds a component that DOES command motion
  (README "Limitations"), that downstream integration inherits this whole ladder from CLAUDE.md §1.

## 4. Business & regulatory context

**Who needs this.** Depth completion (or a learned successor that plays the same role) is used
wherever a cost- or power-constrained sensor suite pairs a dense camera with a sparser depth sensor —
autonomous-vehicle perception stacks (LiDAR is expensive and mechanically sparse relative to camera
resolution), warehouse/logistics AMRs (dense local geometry for costmaps from a cheaper sparse sensor),
and increasingly AR/robotic-manipulation systems pairing RGB with a sparse structured-light or
time-of-flight depth sensor.

**Commercial and open-source players.** LiDAR OEMs (Velodyne/Ouster merger, Hesai, RoboSense, Livox,
Cepton, Innoviz — non-exhaustive, illustrative) build the sensors; AV/robotics perception teams at
companies building self-driving stacks and warehouse-automation platforms build and maintain the
depth-completion (or fused-perception) software, often as an internal, non-public component; the
academic/open-source side (KITTI benchmark entrants, PCL, Open3D) publishes the classical and
early-learned baselines this project's THEORY.md names.

**Cost of getting it wrong.** A dense depth map with confidently-wrong values in exactly the regions
this project studies (texture-fooled or camo-edge-smeared) is arguably WORSE for a downstream
planner/costmap than an honestly sparse signal, because a planner that trusts dense depth may not
independently re-derive "how confident am I here" — a systematic failure mode (e.g. every checkerboard-
patterned surface in a warehouse, every low-contrast object-against-wall edge) could recur across an
entire fleet's operating environment rather than being a one-off sensor glitch, making it a fleet-wide
safety/reliability liability rather than a per-unit hardware fault.

**Regulatory path.** This project sits inside PERCEPTION, one input among many to whatever safety case
a shipping robot's regulatory story is built on (SYSTEM_DESIGN.md item 6's regulatory map, by robot
type): an AV stack's perception-input quality feeds into its ISO 26262 / UL 4600 functional-safety
case; a service robot's into ISO 13482; neither standard certifies a depth-completion ALGORITHM in
isolation — it is validated as part of the whole perception subsystem's safety case, typically via
extensive scenario-based testing including exactly the adversarial cases (texture-heavy surfaces,
low-contrast object boundaries) this project's gates are a miniature, illustrative version of.
Eye-safety classification of the LiDAR ITSELF (IEC 60825-1, THEORY.md "The problem") is a separate,
component-level regulatory requirement the sensor vendor certifies, not something this software project
touches.

**Where this work lives in a company.** Owning team: perception / sensor-fusion (SYSTEM_DESIGN.md item
5), reporting typically into an autonomy/perception engineering org; typical role titles: perception
engineer, computer vision engineer, robotics software engineer. Adjacent teams: the sensor/hardware
team (owns the physical rig this file's §1-2 describe), the mapping/SLAM team (a direct downstream
consumer, README "System context"), and the planning/controls team (a further downstream consumer whose
safety case depends on this component's honesty about its own failure modes).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
