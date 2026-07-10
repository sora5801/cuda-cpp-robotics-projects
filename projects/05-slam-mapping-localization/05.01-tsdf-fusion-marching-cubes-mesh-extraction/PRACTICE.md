# 05.01 — TSDF fusion (KinectFusion clone) + marching-cubes mesh extraction: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is the **depth-camera mount and its rigid link to the compute
platform** — the mapping algorithm itself has no moving parts, but the *sensor* feeding it very much
does, and its construction directly determines the fusion quality this project's ground-truth check
teaches you to reason about:

- **The mount.** A depth camera (structured-light or ToF module) is bolted to a bracket that fixes its
  pose relative to the robot's base frame — on a mobile robot, typically a forward-facing or downward
  tilted mount above the drive base; on a manipulator, a wrist mount (eye-in-hand) or a fixed frame
  mount overlooking the workspace (eye-to-hand). That mounting transform, `T_base_camera`, is exactly
  the `T_world_cam` this project reads from a file — on real hardware it is **calibrated**, not given,
  and any looseness in the bracket (a single loose screw) silently corrupts every fused voxel by the
  same rigid offset until re-calibrated.
- **Vibration and thermal drift.** Depth modules contain a projector/emitter and an image sensor whose
  relative alignment is factory-calibrated; a bracket that flexes under drive-motor vibration, or a
  housing that thermally expands during a multi-hour duty cycle, drifts that internal calibration —
  exactly the kind of systematic bias this project's ground-truth check exists to make visible in
  software (real hardware needs periodic re-calibration to catch it in the field).
- **Cabling and connectors.** USB3 or MIPI CSI-2 links from the sensor to the compute board are
  bandwidth- and length-sensitive (USB3 depth streams commonly exceed reliable passive cable lengths
  around 3–5 m without active extension); a marginal connector produces dropped frames or corrupted
  rows — visible in a real system as depth "holes" this project's synthetic renderer never has to model.
- **What breaks in the field.** IR-emitting depth sensors are sensitive to direct sunlight (the ambient
  IR floods the structured pattern or ToF return) and to condensation/dust on the emitter/receiver
  windows; both degrade coverage long before they cause outright failure, which is why fusing many
  frames (this project's whole premise) is also a *robustness* strategy against per-frame dropout, not
  just a noise-averaging one.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Depth sensor | Hobby/research: Intel RealSense D400/D435-class active stereo, Azure Kinect-class ToF (successor modules); industrial: SICK/Zivid/Photoneo structured-light or ToF units | Produces the per-frame depth `Image` this project consumes (here, rendered synthetically instead) |
| Sensor interface | USB3 (most RGB-D modules) or MIPI CSI-2 (embedded ToF/stereo modules) | Gets the raw depth frame onto the compute platform |
| Compute for fusion | Jetson Orin class (embedded, on-robot) / x86 + RTX (reference machine: RTX 2080 SUPER) | Runs the integration + marching-cubes kernels this project implements |
| Real-time/host CPU | The same box's CPU cores | Pose lookup, depth pre-processing, orchestrating the fusion loop |
| Pose source | Wheel/IMU odometry + `02.06`-style ICP, or a visual-inertial estimator, or (research rigs) an external motion-capture system | Supplies the `T_world_cam` this project takes as given |
| Mount/bracket | Machined or 3D-printed bracket, vibration-damped standoffs on mobile platforms | Fixes the sensor's pose relative to the robot's base frame — the calibration this project's poses stand in for |

The reconstruction-compute story (where the fusion kernel physically runs, GPU SoC vs. discrete card) is
the same tier ladder `33.01` PRACTICE §1 and `08.01` PRACTICE §2 already walk through; it is not repeated
here.

## 3. Installation & integration — putting it on a real robot

- **Process shape (ROS 2).** A real deployment runs this pipeline as a mapping node: subscribes a depth
  `sensor_msgs/msg/Image` (or `PointCloud2`) topic and a `tf2` transform for the camera's pose at each
  frame's timestamp, and republishes the fused result — commonly a mesh (`visualization_msgs/msg/Marker`
  or a custom mesh message) for RViz/operator viewing, and/or an occupancy or ESDF grid
  (`nav_msgs/msg/OccupancyGrid`-style, extended to 3-D — see `05`'s ESDF-generation sibling project) for
  the planning stack. This project's own structs (`Intrinsics`, `PoseRt`, the flat `tsdf`/`weight`
  arrays) are deliberately message-shaped stand-ins for exactly that boundary
  (SYSTEM_DESIGN §3.6) — the demo's `main.cu` load → integrate → extract loop maps one-to-one onto that
  node's per-frame callback plus a periodic (or on-demand) mesh-extraction service call.
- **Real-time constraints, honestly.** Fusion at 10–20 Hz with a sub-millisecond-to-low-single-digit-ms
  GPU kernel (this project's measured 128³ numbers) is comfortably schedulable as a **soft** real-time
  task on a general-purpose Linux host — it is not in the 0.5–1 kHz hard-deadline tier (SYSTEM_DESIGN
  §1.1) that would demand a dedicated real-time core. Marching-cubes extraction is typically run less
  often than integration (on demand for viewing, or at a slower fixed rate like 1–5 Hz), since planners
  usually want a distance FIELD (queryable everywhere), not a re-triangulated mesh, on every tick.
- **Calibration and bring-up, rung by rung (CLAUDE.md §1 testing ladder):**
  1. *Simulation* — this demo, plus stress cases (very oblique camera paths, moving objects, more
     frames than the truncation weight cap can track).
  2. *Intrinsic/extrinsic calibration on the bench* — checkerboard or ChArUco calibration for the depth
     module's intrinsics, then camera-to-robot-base extrinsic calibration (the `T_base_camera` this
     project reads from a file); verify against a known flat target before trusting any fused output.
  3. *Bench/tethered, static scene* — run the real fusion pipeline against a stationary object of known
     dimensions (a calibrated sphere or cube is the direct real-world analog of this project's analytic
     scene) and compare the extracted mesh against the known geometry — literally this project's
     ground-truth check, performed with calipers instead of a closed-form SDF.
  4. *Moving platform, low speed, instrumented* — confirm the fused map stays coherent as the pose
     source (not this project's given poses) starts contributing its own drift and noise.
  5. *Free operation* — only after the map has demonstrated it degrades gracefully (drops confidence,
     not correctness) under realistic pose and depth noise.
- **N/A here:** no fieldbus or sensor driver is implemented in this project — depth arrives as an
  in-memory array (here, synthetically rendered) rather than over USB3/MIPI, and no ROS 2 node wrapper
  exists in `src/`. Stated per contract; the mapping above is the honest translation, not an
  implementation.

## 4. Business & regulatory context

- **Who needs dense mapping:** any robot that must plan motion around, or present a 3-D model of, an
  environment it cannot fully know in advance — warehouse AMRs avoiding dynamic obstacles, manipulator
  work cells reconstructing a bin's contents for collision-aware grasping, inspection robots producing
  as-built 3-D scans, and AR/reconstruction products outside robotics entirely (SYSTEM_DESIGN §5.1: this
  work sits with the controls & autonomy team, domains 04/05/06/07 in the org map).
- **The players.** NVIDIA's nvblox (part of the Isaac stack) and ETH Zurich's open-source Voxblox are
  the reference production/research implementations most directly descended from KinectFusion; Open3D
  ships a widely used integration pipeline; commercial 3-D scanning and industrial-metrology vendors
  (structured-light and laser scanner makers) solve an adjacent, offline-accuracy-focused version of the
  same problem. Build-vs-buy (SYSTEM_DESIGN §5.3): dense mapping is rarely a company's differentiator by
  itself — most teams adopt nvblox/Voxblox-class libraries and spend engineering effort on what
  *consumes* the map (planning, grasp scoring, human-facing visualization) instead, which is exactly the
  "is it your differentiator" litmus test §5.3 describes.
- **Cost of getting it wrong.** A biased or stale map is a silent failure mode: a robot's *planner* may
  behave perfectly correctly against a *map* that no longer matches reality (a moved obstacle the
  truncation-weight cap has not yet "forgotten," or a grazing-incidence bias like this project's
  measured tail making a surface appear centimeters away from where it is). In a manipulator cell this
  can mean a collision the planner thought was clear; in an AMR fleet it can mean stuck or unsafe
  navigation. Mitigations are architectural: conservative inflation of mapped obstacles for planning
  margins, confidence-aware fusion (PRACTICE §1's calibration discipline, THEORY's incidence-weighting
  discussion), and independent safety monitors that do not trust the map alone (`31`, SYSTEM_DESIGN §6.1
  safety chain).
- **Regulatory.** Dense mapping itself is not directly named by a certification standard, but it feeds
  systems that are: on an industrial arm, the reconstructed scene informs collision-avoidance behavior
  in scope for **ISO 10218 / ISO/TS 15066** (SYSTEM_DESIGN §6.2); on a mobile robot sharing space with
  people, mapping quality is one input to the hazard analysis **ISO 13482** expects; on an AV, HD-map
  and localization quality sits inside the **ISO 26262 / UL 4600** safety-case argument. This project's
  domain (05) appears in that table via those downstream consumers, not directly — cite the
  SYSTEM_DESIGN §6.2 orientation map, not this project, for specifics.
- **Owning team.** Controls & autonomy (SYSTEM_DESIGN §5.1) typically owns dense mapping; the closest
  adjacent teams are perception (owns the depth sensor driver and calibration pipeline) and simulation
  & tools (owns the sensor models this project's ray-cast renderer stands in for, and the regression
  test scenes that keep a real mapping stack honest — this project's whole verification strategy is a
  miniature version of exactly that kind of regression scene).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
