# 01.02 — Stereo depth: block matching, then Semi-Global Matching (SGM) kernels: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

The physical carrier this project's algorithm serves is a **rigid stereo camera rig**: two image
sensors, a known, FIXED baseline apart, both hard-mounted to a single rigid structure.

- **The rig.** Two camera modules (sensor + lens + often an IR-cut filter) bolted or bonded to a
  single rigid bar, plate, or PCB — usually machined aluminum or a stiff PCB/composite, chosen
  specifically because it must NOT flex, twist, or thermally expand differently at the two ends. The
  entire theory in `THEORY.md` ("The math") assumes `B` (the baseline) is a known CONSTANT; if the
  physical baseline drifts by even a fraction of a millimeter, every downstream depth estimate is
  silently biased (the `Z²` depth-error law in THEORY.md means this bias GROWS with range).
- **Mounting and tolerances.** Both sensors must be co-planar and rotationally aligned within a small
  fraction of a degree of "rectified" — real rigs cannot achieve this by machining alone, which is WHY
  rectification (README "System context": project 01.01/01.07's job) exists as a SOFTWARE correction on
  top of imperfect hardware alignment; the mechanical rig's job is to keep that misalignment SMALL and
  STABLE enough that one calibration stays valid for a useful length of time.
- **Synchronization wiring.** Both sensors must expose the SAME instant (THEORY.md "The problem"); this
  is usually a hardware trigger line (GPIO or a dedicated sync bus) connecting the two sensor boards, not
  something software timestamps alone can guarantee at the sub-millisecond level a moving scene needs.
- **What breaks in the field.** Thermal cycling flexes the rig and drifts the baseline/rotation
  (recalibration cadence, §3); mechanical shock (a robot bumping into something, being dropped in
  transit) can permanently bend the rig; connector fatigue on the sync/data cables from vibration;
  lens fogging, dust, and rain ingress for outdoor rigs (an IP-rated enclosure is a real line item, not
  an afterthought); and — specific to PASSIVE stereo (this project's family, THEORY.md "Where this
  sits in the real world") — performance degrades gracefully but measurably in low light or on
  textureless surfaces (a blank wall gives every window an identical census signature — THEORY.md's
  "locally ambiguous" story made physical).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

**The baseline/field-of-view physics trade** (governs every rig design decision below, derived from
THEORY.md's `Z = f·B/d` line): a WIDER baseline `B` gives more disparity per unit depth — better
far-range precision (the `Z²` error law scales inversely with `B`) — but ALSO shrinks the overlapping
field of view between the two cameras at close range (the two cameras literally cannot both see
something right in front of the rig if they are far apart) and increases the minimum measurable range
(`Z_min = f·B/D_max`, from THEORY.md — a wider `B` pushes the near limit farther away for the SAME
disparity search range `D`). A manipulator's wrist camera (needs close-range precision, small working
volume) typically uses a NARROW baseline (5–12 cm); an outdoor AMR or AV forward-facing rig (needs
far-range awareness) uses a WIDER baseline (12–50+ cm) or accepts a narrower one and leans on other
sensors (LiDAR, radar) for long range instead.

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Compute for the kernels | Jetson Orin class (embedded, on-robot) / x86 + discrete RTX (reference machine: RTX 2080 SUPER, sm_75) | Runs census + cost volume + SGM every frame |
| Camera modules, hobby | Stereolabs ZED / ZED 2i (integrated stereo + onboard depth), Luxonis OAK-D (onboard depth ASIC), Intel RealSense D4xx family (structured-light-ASSISTED stereo — a hybrid, not purely passive) | ~US$200–500/unit; good for learning and prototyping |
| Camera modules, research | Paired global-shutter industrial cameras (e.g. Basler/FLIR/IDS) on a custom-machined rig with hardware trigger sync | ~US$500–3,000/pair + rig machining; full control over baseline, lens, exposure |
| Camera modules, industrial | Automotive-grade GMSL2 camera pairs (e.g. built around Sony IMX sensors) with a certified, temperature-qualified rig | ~US$1,000–5,000+/pair; the tier that survives outdoor vibration/thermal cycling for years |
| Sync/interconnect | MIPI CSI-2 (short runs, embedded boards), GMSL2/FPD-Link III (long automotive-grade runs with power-over-coax), USB3 (simple, less deterministic timing) | Carries pixel data AND (for CSI/GMSL rigs) the hardware sync trigger |
| Onboard depth ASIC (where present) | Movidius Myriad X (OAK-D family), proprietary depth ASICs in some structured-light modules | An alternative to running THIS project's kernels at all — see §3 |

The bullet's own compute is illustrative of the domain's spread, not a purchasing recommendation:
`../../33-foundational-libraries/` and this project's own README "In production" name the
dedicated-silicon and library alternatives (VPI, libSGM) that replace hand-rolled kernels like this
project's in a shipping product.

## 3. Installation & integration — putting it on a real robot

- **Process shape (ROS 2).** A real deployment is typically two nodes: a camera DRIVER node
  publishing `sensor_msgs/msg/Image` (raw, left + right) and `sensor_msgs/msg/CameraInfo` (intrinsics
  + distortion, from calibration), then a RECTIFICATION node (`image_proc`'s
  `image_proc::RectifyNode`, or a vendor equivalent) producing rectified `Image` pairs, then THIS
  project's kind of node consuming the rectified pair and publishing `stereo_msgs/msg/DisparityImage`
  or a converted `sensor_msgs/msg/PointCloud2` (`Z = f·B/d`, THEORY.md, applied per pixel) for
  downstream consumers (README "Downstream consumers": TSDF fusion, costmaps, grasp planning). This
  project's own structure — persistent device buffers, one shared census+cost-volume pass feeding two
  disparity extractors — maps directly onto "one GPU-backed node, called once per synchronized
  stereo frame".
- **Where it runs, and the real-time picture.** On a manipulator work cell: the industrial PC beside
  the arm controller (SYSTEM_DESIGN §2.2). On a mobile robot: the perception compute tier (Jetson Orin
  class or an onboard x86+dGPU box). The camera→perception budget is 30–60 Hz / <1 frame end-to-end
  (SYSTEM_DESIGN item 1); this project's own SGM pass (~60–80 ms, THEORY.md "The GPU mapping") is an
  HONEST MISS of that budget as a teaching-scale, untiled implementation — a fielded node would need
  the tiling/8-direction/library-grade optimizations named in README "Limitations" before it could sit
  in a hard 30–60 Hz loop; this is a SOFT real-time workload (a slow frame delays the depth estimate,
  it does not crash a control loop the way a missed 0.5–1 kHz whole-body-control tick would,
  SYSTEM_DESIGN item 1's rate table).
- **Calibration is the single biggest gap between this demo and reality.** Every real rig needs
  INTRINSIC calibration (per camera: focal length, principal point, lens distortion — a checkerboard
  or ChArUco-board procedure, `camera_calibration` in ROS 2 or a vendor tool) and EXTRINSIC calibration
  (the two cameras' relative pose, which rectification and `B` in THEORY.md's equations both depend
  on) — usually done ONCE at build time and re-verified on a schedule (§1's "what breaks in the field"
  drives the cadence: a rig that has taken a shock or a large thermal swing gets re-checked before
  being trusted again). This project's synthetic scene sidesteps calibration entirely by AUTHORING an
  already-perfectly-rectified pair (`data/README.md`) — the single largest simplification versus a
  real deployment, named honestly in README "Limitations".
- **The safe hardware-testing ladder (CLAUDE.md §1), applied to a PERCEPTION node specifically:**
  1. *Simulation* — this demo, plus synthetic scenes with deliberately harder texture/occlusion
     (README Exercise 2) and (eventually) real recorded stereo pairs (Exercise 5).
  2. *HIL / recorded data* — replay real rosbags of stereo pairs through the exact node that will run
     on the robot, checking latency and depth-map quality against known scene geometry, with no robot
     motion involved yet.
  3. *Bench, camera live, robot stationary* — the rig mounted on the actual robot, publishing live
     depth, but with NOTHING downstream allowed to command motion from it yet — a human watches the
     depth/point-cloud visualization against the real scene.
  4. *Closed loop* — only once a downstream consumer (grasp planning, obstacle avoidance) is itself
     independently tested and staged per ITS OWN ladder; this project's output is a *depth estimate*,
     not a motion command, but everything that turns it INTO one inherits the full staged-bring-up and
     E-stop discipline of whichever control/planning project consumes it (that project's own PRACTICE.md).
- **N/A here:** no real camera driver, rectification node, or ROS 2 topic publishing is implemented in
  this project — the demo's "camera" is two committed PGM files and its "output" is a function return,
  by the self-containment rule (CLAUDE.md §4). Stated per contract, not left implicit.

## 4. Business & regulatory context

- **Who needs stereo depth.** Any robot that needs 3-D structure without an active emitter: outdoor
  field robotics (sunlight defeats most structured-light/ToF sensors — THEORY.md "Where this sits in
  the real world"), long-range perception on AMRs and AVs, and increasingly manipulation work cells
  that want a cheaper or wider-FOV alternative to a dedicated depth camera. It is also the DEFAULT
  depth source anywhere a robot ALREADY carries stereo cameras for other reasons (visual odometry,
  object detection) and stereo depth comes essentially "for free" as an extra output of the same
  sensor pair.
- **The players.** Camera/module vendors (Stereolabs, Luxonis, Intel RealSense — PRACTICE §2);
  GPU-accelerated SGM libraries (libSGM/fixstars, OpenCV `cuda::StereoSGM`, NVIDIA VPI — README "Prior
  art"); and, increasingly, learned stereo/monocular depth models (RAFT-Stereo and peers, THEORY.md
  "Where this sits in the real world") as a competing or complementary approach. Build-vs-buy: the
  MATCHING ALGORITHM itself (census/SGM or a learned equivalent) is usually BOUGHT (a library or a
  pretrained model) rather than built in-house — what a robotics company typically owns instead is the
  CALIBRATION pipeline, the sensor selection/rig design, and the downstream consumers of the depth
  output (SYSTEM_DESIGN item 5's build-vs-buy framing).
- **Cost of getting it wrong.** A biased or noisy depth estimate propagates directly into whatever
  consumes it: a grasp planner picking the wrong grasp pose (a dropped or crushed part, PRACTICE
  chains through 19.01), an obstacle layer under- or over-estimating clearance (a collision, or an
  overly conservative robot that cannot do its job), or — worst — a safety-relevant distance estimate
  that is wrong in the DANGEROUS direction (THEORY.md's `Z²` depth-error law means this risk GROWS at
  range, precisely where a robot has the least time to react). This is why production stereo depth is
  routinely FUSED with other modalities (LiDAR, radar, active depth) rather than trusted alone for
  safety-critical decisions (README "Limitations" states this explicitly).
- **Regulatory.** Camera-based perception sits under whichever product-level regulatory regime the
  robot itself falls under (SYSTEM_DESIGN item 6's map, not a perception-specific standard): industrial
  arms (ISO 10218 / ISO/TS 15066), service robots (ISO 13482), autonomous vehicles (ISO 26262 / UL
  4600, where a camera-only stereo path would need to be one validated input among several, never the
  sole basis for a safety case). No project in this repository claims certification of any kind
  (CLAUDE.md §1) — this is an orientation map, not compliance guidance.
- **Owning team.** Perception (SYSTEM_DESIGN item 5) — typical titles: perception engineer, computer
  vision engineer. Adjacent teams: embedded/hardware (owns the physical rig and its calibration
  procedure, PRACTICE §1–§3), simulation (owns the synthetic/replay data this kind of node is tested
  against before real hardware), and functional safety (owns the fusion/redundancy policy that keeps
  any single perception modality, including this one, from being a single point of failure).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
