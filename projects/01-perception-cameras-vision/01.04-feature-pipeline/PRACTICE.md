# 01.04 — Feature pipeline: FAST/Harris detection, ORB descriptors, brute-force Hamming matcher: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

The physical carrier this project's algorithm serves is a single, calibrated **monocular camera rig** —
simpler than project 01.02's stereo pair (no baseline to hold rigid), but the SAME underlying camera
module, and the SAME reliance on a stable mount:

- **The camera module.** A lens (fixed- or auto-focus; a feature pipeline generally prefers FIXED focus
  and exposure so a keypoint's appearance stays consistent frame to frame) over an image sensor (a
  global-shutter sensor is strongly preferred — THEORY.md "The problem" discusses why rolling shutter's
  row-by-row exposure skew corrupts corner locations on a fast-moving robot), on a small PCB, in a
  housing that keeps the lens-to-sensor distance and alignment fixed.
- **Mounting and vibration.** The camera must be RIGIDLY fixed to whatever rigid body its pose is being
  tracked relative to (the robot chassis, a drone's frame, a gimbal's output stage). Vibration blurs
  every frame (motion blur destroys the sharp intensity structure both FAST and Harris depend on —
  THEORY.md's aperture-problem discussion), and any UNMODELED flex between the camera and the IMU it is
  typically fused with (SYSTEM_DESIGN.md §2.4's quadrotor VIO block) directly corrupts the extrinsic
  calibration every downstream pose estimate assumes is fixed.
- **What breaks in the field.** Lens fogging/condensation and dust/rain ingress (an IP-rated housing for
  outdoor robots); connector fatigue on the MIPI CSI-2 / USB3 cable from vibration; auto-exposure
  "hunting" under changing lighting, which this project's brightness-offset invariance claim is
  specifically meant to survive (a SLOW, uniform gain change) but a SUDDEN, non-uniform one (headlights,
  a shadow sweeping across the scene) can still starve a frame of usable corners for one or more cycles;
  and — the single most common real-world failure mode — texture-poor scenes (blank walls, uniform
  floors, fog, open sky) simply starving the detector of corners regardless of hardware health, which
  THEORY.md's "engineering constraints" section names as a real, un-fixable-by-better-hardware
  limitation of feature-based approaches generally.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Compute for the kernels | Jetson Orin class (embedded, on-robot, the realistic target for this project's kernels) / x86 + discrete RTX (reference machine used to build this project: RTX 2080 SUPER, sm_75) / MCU-class (too weak for the GPU kernels here, but a real product's FALLBACK CPU implementation, e.g. OpenCV's non-CUDA FAST/ORB, can run on a Cortex-A-class SoC) | Runs FAST/Harris/ORB/matching every frame |
| Camera module, hobby | Raspberry Pi Camera Module 3 (global-shutter variant), OV9281-based global-shutter USB modules, Arducam global-shutter boards | ~US$25–80; good for learning and low-speed prototyping |
| Camera module, research | Machine-vision global-shutter cameras (e.g. Basler dart/ace, FLIR Blackfly S, IDS uEye) on a rigid mount with a hardware trigger input | ~US$300–1,500; full control over exposure/gain/trigger timing, the tier most VIO/VO research platforms actually use |
| Camera module, industrial/automotive | GMSL2-linked automotive-grade global-shutter modules (built around sensors like the Sony IMX or OnSemi AR-series), temperature-qualified housings | ~US$500–3,000+; the tier that survives outdoor vibration/thermal cycling for years, common on production AMRs and AVs |
| Companion IMU (typical pairing, not this project's own scope) | Bosch BMI088/BMI270-class (hobby/research), tactical-grade fiber-optic or MEMS IMUs (industrial VIO) | Fused with feature tracks downstream (SYSTEM_DESIGN §2.4's `STATE ESTIMATION [04]` block) — this project produces the visual half of that fusion's input, not the IMU half |
| Interconnect | MIPI CSI-2 (short runs, embedded boards — the common case for a Jetson-class companion computer), USB3 (simple, less deterministic timing), GMSL2 (long automotive-grade runs) | Carries pixel data from sensor to compute |

The compute tier is illustrative of the domain's spread, not a purchasing recommendation: this
project's own README "In production" section names the dedicated-library alternative (NVIDIA VPI,
OpenCV `cv::cuda`) that replaces hand-rolled kernels like this project's in a shipping product, and the
learned-front-end alternative (SuperPoint-class networks, needing a GPU/NPU inference budget this
project's classical kernels do not) that is displacing hand-crafted pipelines at the research frontier.

## 3. Installation & integration — putting it on a real robot

- **Process shape (ROS 2).** A real deployment is typically two or three nodes: a camera DRIVER node
  publishing `sensor_msgs/msg/Image` (raw or already-rectified) and `sensor_msgs/msg/CameraInfo`
  (intrinsics + distortion, from calibration — project 01.01/01.16's job), then a FEATURE node of
  exactly this project's shape consuming each `Image` and publishing a custom
  `KeyPoints`/`Descriptors` message pair (or, more commonly in practice, feeding directly into an
  in-process visual-odometry node rather than round-tripping through a separate topic, since
  keypoint+descriptor messages for a few hundred points at 30-60 Hz are small but the LATENCY of an
  extra pub/sub hop matters at that rate). Downstream, a visual-odometry/SLAM node (README "Downstream
  consumers": SLAM domain 05) consumes the matched correspondences this project's `matches.csv` stands
  in for.
- **Compute placement: Jetson vs. discrete GPU.** On a SWaP-constrained platform (the quadrotor
  reference robot, SYSTEM_DESIGN §2.4), this project's kernels run on a Jetson-class SoC's integrated
  GPU, sharing silicon and power budget with the flight-control companion computer's other perception
  work — every kernel here was written assuming a SINGLE small image (256x256 in the demo; a VO front
  end typically runs on a similarly modest working resolution, often downsampled from the sensor's
  native VGA/HD, specifically to keep this stage's compute and the state estimator's per-frame budget
  small) rather than a large batch, matching that embedded reality. On a warehouse AMR (SYSTEM_DESIGN
  §2.1) with a discrete dGPU already present for LiDAR/point-cloud work, this project's kernels would
  typically run as one more small workload sharing that GPU, competing for launch slots with the
  point-cloud pipeline rather than needing dedicated silicon.
- **Real-time picture.** Camera→perception is a 30-60 Hz / <1 frame (16-33 ms) budget (SYSTEM_DESIGN
  item 1); this project's entire three-stage pipeline measures ~1.8 ms of GPU kernel time on the
  reference machine for a 256x256 frame (`main.cu`'s `[time]` line) — comfortably inside that budget at
  this resolution, though a fielded system at VGA/HD resolution and with the shared-memory-tiled
  optimizations THEORY.md's "GPU mapping" names as future work would need to re-measure before assuming
  the same headroom holds. This is a SOFT real-time workload in the SYSTEM_DESIGN sense: a slow frame
  delays or drops one visual-odometry update, it does not directly damage hardware the way a missed
  0.5-1 kHz whole-body-control tick would.
- **Calibration and bring-up.** A real deployment needs (a) INTRINSIC calibration (focal length,
  principal point, lens distortion — project 01.16's job, typically a checkerboard/ChArUco procedure
  done once at build time and re-verified on a schedule) and (b) — if this project's output feeds a
  visual-INERTIAL pipeline, as the quadrotor reference robot's does — camera-IMU EXTRINSIC calibration
  (the rigid transform between the two sensors, e.g. via Kalibr or a vendor tool), since VIO's whole
  premise depends on that transform being known accurately. This project's synthetic scene sidesteps
  both entirely (an already-perfect, already-calibrated analytic rendering, `data/README.md`) — the
  single largest simplification versus a real deployment, named honestly in README "Limitations".
- **The safe hardware-testing ladder (CLAUDE.md §1), applied to a PERCEPTION node specifically:**
  1. *Simulation* — this demo, plus synthetic scenes with deliberately harder texture/lighting/motion
     (README Exercises), and eventually replayed real camera footage with independently-known ground
     truth (e.g. a motion-capture-tracked camera rig).
  2. *HIL / recorded data* — replay real rosbags of camera frames through the exact node that will run
     on the robot, checking feature counts, match rates, and (where ground truth exists) pose-recovery
     accuracy, with no robot motion involved yet.
  3. *Bench, camera live, robot stationary* — the camera mounted on the actual robot, publishing live
     features/matches, but with NOTHING downstream allowed to command motion from them yet — a human
     watches the keypoint overlay (this project's `keypoints_A.ppm`-style visualization, on live video)
     against the real scene.
  4. *Closed loop* — only once the downstream consumer (a VO/VIO state estimator, and whatever planner/
     controller consumes ITS output) is itself independently tested and staged per its own ladder; this
     project's output is a set of point correspondences, not a motion command, but everything built on
     top of it inherits the full staged-bring-up and E-stop discipline of the control system it
     eventually feeds.
- **N/A here:** no real camera driver, ROS 2 node, or topic publishing is implemented in this project —
  the demo's "camera" is two committed PGM files and its "output" is a set of files under `demo/out/`,
  by the self-containment rule (CLAUDE.md §4). Stated per contract, not left implicit.

## 4. Business & regulatory context

- **Who needs this capability.** Essentially every robot that moves and carries a camera: drones and
  quadrotors (visual-inertial odometry, often the PRIMARY state estimate indoors/GPS-denied — SYSTEM_
  DESIGN §2.4), AMRs and AVs (visual loop closure and relocalization alongside LiDAR — SYSTEM_DESIGN
  §2.1/§2.5), AR/VR headsets (a close commercial cousin running the identical detect-describe-match
  primitive for 6-DoF head tracking), and robotic manipulation systems doing camera-based tool/object
  tracking. Camera calibration and multi-camera extrinsic estimation (project 01.16/01.17) use the same
  underlying corner-detection primitive for a different purpose (metric calibration rather than motion
  estimation).
- **The players.** Feature-pipeline libraries and SDKs: OpenCV (`cv::ORB`, `cv::cuda`), NVIDIA VPI and
  Isaac ROS's visual-SLAM packages, Google's ARCore and Apple's ARKit (closed-source VIO stacks built on
  this exact class of front end), and the open-source visual-SLAM ecosystem (ORB-SLAM3, VINS-Fusion,
  OpenVINS). Increasingly, learned front ends (SuperPoint/SuperGlue-class models, sometimes shipped as
  part of a vendor's SDK) compete with hand-crafted FAST/ORB pipelines like this project's, particularly
  where compute budget allows a neural-network forward pass per frame. Build-vs-buy: the DETECTOR/
  DESCRIPTOR/MATCHER algorithm itself is usually BOUGHT (a library, SDK, or pretrained model) rather
  than hand-rolled in a shipping product — what a robotics company typically owns instead is sensor
  selection, calibration procedures, the state-estimation/SLAM back end consuming the features, and
  system-level tuning (SYSTEM_DESIGN item 5's build-vs-buy framing).
- **Cost of getting it wrong.** A feature front end that drifts, loses tracking, or accepts too many
  false matches propagates directly into whatever consumes it: a drone's VIO estimate diverging (loss
  of position hold, a crash), an AMR failing to relocalize after a map update (a stalled or lost robot
  requiring human intervention), or an AR/VR headset's tracking "jumping" (at minimum a bad user
  experience; at worst, in an industrial AR overlay context, a misleading overlay). This project's own
  four-gate verification strategy (README "Expected output") — and specifically the NEGATIVE CONTROL
  gate — exists because false-but-confident matches are exactly the failure mode that costs the most
  downstream: a wrong correspondence that LOOKS confident (passes the ratio test, passes the distance
  cap) is more dangerous to a SLAM back end than an honest "no match found".
- **Regulatory.** Camera-based perception sits under whichever product-level regulatory regime the
  robot itself falls under (SYSTEM_DESIGN item 6's map), not a perception-specific standard: drones (FAA
  Part 107 / EASA rules, which govern the VEHICLE's certification, not this algorithm specifically),
  service robots (ISO 13482), autonomous vehicles (ISO 26262 / UL 4600 — a camera-only visual-odometry
  path would need to be one validated, redundant input among several, never a sole basis for a safety
  case). No project in this repository claims certification of any kind (CLAUDE.md §1) — this is an
  orientation map, not compliance guidance.
- **Owning team.** Perception (SYSTEM_DESIGN item 5) — typical titles: perception engineer, computer
  vision engineer, SLAM/state-estimation engineer. Adjacent teams: embedded/hardware (owns the physical
  camera module, mount, and its calibration procedure, PRACTICE §1-§3), simulation (owns the synthetic/
  replay data this kind of node is tested against before real hardware — this project's own
  `scripts/make_synthetic.py` is a miniature version of that role), and controls/autonomy (the eventual
  consumer of whatever state estimate this project's output feeds into).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
