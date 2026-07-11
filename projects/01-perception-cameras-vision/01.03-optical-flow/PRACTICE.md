# 01.03 — Optical flow: pyramidal Lucas-Kanade, Farneback, census-transform flow: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-10. Parts numbers, prices, and vendor names below are **illustrative
> examples, never endorsements**; verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project is pure software, but it exists to feed on ONE specific piece of physical hardware: a
camera assembly, and specifically its **shutter type and lens**, because both directly determine
whether brightness constancy (`THEORY.md`) holds well enough for either algorithm here to work at all.

- **Global vs. rolling shutter.** A rolling-shutter CMOS sensor (the overwhelming majority of
  consumer/webcam-class sensors, because it is cheaper to fabricate) exposes each row microseconds
  after the row above it. A fast-rotating or vibrating robot then captures a single "frame" whose top
  and bottom rows are genuinely different POSES — real geometric skew that neither Lucas-Kanade's
  translation model nor census's block search accounts for. Optical-flow-for-motion-estimation
  hardware (see §2) is built almost exclusively on **global-shutter** sensors for exactly this reason:
  every pixel in a global-shutter frame shares one exposure instant, so a rigid-motion assumption
  between two frames is actually true, not approximately true.
- **Lens and exposure trade-offs.** A wider aperture (lower f-number) shortens the exposure needed for
  a given light level, reducing MOTION BLUR — a smeared frame violates brightness constancy in a
  spatially-varying way no simple model in this project corrects, and both algorithms silently produce
  garbage on a badly-blurred pair rather than failing loudly. A narrower field of view (longer focal
  length) increases the apparent pixel displacement for a given real motion — useful for precision at
  low speed, but shrinks the field over which the pyramid's coarse levels can still find corresponding
  content before it exits the frame entirely.
- **Mounting and vibration.** A camera rigidly bolted to a vibrating airframe (a multirotor's motor
  vibration coupling through the frame) injects high-frequency apparent motion the flow field will
  faithfully report as real — mechanically isolating the camera (a small elastomer damping mount, or a
  gimbal) is a common, low-tech fix upstream of any software flow algorithm; production
  vision-for-control camera modules (see §2) often ship pre-mounted on such isolation.
- **What breaks in the field**: a scratched or dust-fouled lens locally degrades contrast exactly where
  optical flow needs it most (both algorithms report low confidence/validity there, which is the
  CORRECT behavior — a robot's fusion layer should down-weight, not trust, flow from a fouled optic);
  a loose lens mount that shifts focus under vibration changes the effective PSF (point-spread
  function) frame to frame, subtly violating brightness constancy at every edge.

## 2. Real hardware — chips, parts, illustrative BOM

*Verify current before relying on any part number, price, or spec below — this list ages quickly.*

**Compute this project's kernels would run on, by tier:**

| Tier | Example | Where flow compute happens |
|------|---------|------------------------------|
| Hobby / research | Jetson Orin Nano / NX (Jetson class), or a laptop with a discrete RTX-class GPU | The SAME CUDA kernels this project ships, recompiled for the target's compute capability (`sm_87` on Orin vs. this project's `sm_75` floor — see `docs/BUILD_GUIDE.md`). |
| Mid-tier product | Jetson Orin AGX, or x86 + discrete RTX-class GPU (an AMR's compute puck) | Same kernels; more SMs, more headroom for higher resolution/frame rate or additional perception tasks sharing the GPU. |
| Industrial / fixed-function | A GPU with NVIDIA's dedicated optical-flow hardware ACCELERATOR (Turing and later — see `THEORY.md` "Where this sits in the real world", NVIDIA VPI), OR a dedicated vision-processing SoC (e.g., an Ambarella/Movidius-class VPU) | A fixed-function silicon block computes a census-like cost volume with near-zero CUDA-core load, freeing the GPU entirely for other perception tasks — the production answer to "make this project's Milestone 2 free." |

**The camera itself, by tier:**

- **Hobby**: a USB3 global-shutter camera module (e.g., in the spirit of an OV9281-class or
  IMX296-class global-shutter sensor board, ~US$40-120) — inexpensive, but check the datasheet's
  shutter type explicitly; most cheap USB webcams are rolling-shutter and will visibly skew under
  robot motion (§1).
- **Research**: an industrial global-shutter camera (e.g., in the spirit of FLIR/Teledyne Blackfly S
  or Basler ace-class modules, ~US$300-900) with GPIO hardware-trigger input — lets an upstream IMU or
  flight controller time-stamp frames precisely, important for fusing flow with inertial data (see §3).
- **Industrial / flight-certified**: a purpose-built optical-flow sensor MODULE that does the entire
  pipeline (camera + a small onboard processor running LK-family or correlation-based flow + a
  sonar/lidar rangefinder for the metric scale flow alone cannot provide — see §3) as a single
  integrated part, e.g. in the lineage of the PX4Flow project (an open-hardware global-shutter camera +
  gyro + ARM MCU module that outputs `(x-velocity, y-velocity, quality)` over UART/I2C directly to a
  flight controller) — the closest real-world hardware analogue to what this project's `main.cu`
  computes, at a much smaller compute/power budget than a full Jetson.
- **Power/interface**: USB3 (5-10 W typ. for a machine-vision camera) or MIPI CSI-2 (lower power,
  board-level, typical of Jetson-attached cameras) — MIPI is the usual choice when the camera and the
  flow-computing SoC are co-designed on the same carrier board.

## 3. Installation & integration — putting it on a real robot

- **Where it runs.** On an AMR or manipulator work cell, flow compute typically lives on the SAME
  onboard perception computer as the rest of the vision stack (the GPU/Jetson node named in
  `docs/SYSTEM_DESIGN.md`'s reference architectures), sharing the GPU with detection/SLAM/mapping
  kernels. On a small multirotor, it more often lives on a DEDICATED, small, low-power module (a
  PX4Flow-class sensor, §2) physically separate from the main flight-control compute, precisely because
  velocity estimation is safety-critical and low-latency enough to want its own guaranteed compute
  budget, not to compete for GPU time with a mapping or object-detection job.
- **OS / real-time constraints.** A Linux perception node (ROS 2 on Ubuntu, typical Jetson deployment)
  is adequate for flow feeding a STATE ESTIMATOR (which itself runs at 100-400 Hz and can tolerate the
  flow input arriving at camera frame rate, 30-60 Hz, with some jitter — see `docs/SYSTEM_DESIGN.md`
  item 1's rate table). Flow feeding a TIGHT inner control loop directly (rare, but done on some
  optical-flow-only hover controllers) wants a bounded-latency path — the PX4Flow-class module's
  dedicated MCU firmware, not a general-purpose Linux scheduler, is how production systems buy that
  bound.
- **ROS 2 node/topic shape.** A flow node would typically subscribe to `sensor_msgs/Image` (or
  `sensor_msgs/CompressedImage`) from the camera driver, and publish something in the spirit of a dense
  `sensor_msgs/Image` (flow encoded as a 2-channel float image, the common convention) or, for the
  sparse/PX4Flow-style use case, a small custom message carrying `(vx, vy, quality, integration_dt)` —
  matching `docs/SYSTEM_DESIGN.md`'s "message-shaped structs that deliberately resemble ROS 2 types"
  interface convention. This project's own `Keypoint`-free, per-pixel `(float u, float v)` arrays are
  the dense-image-message shape; a downstream consumer wanting sparse velocity would aggregate
  (median-filter) the confident/valid subset, exactly as `main.cu`'s masks already identify.
- **Calibration and bring-up.** Optical flow needs the camera's INTRINSICS (focal length, principal
  point) to convert pixel-domain flow into an ANGULAR rate, and — critically, for velocity in physical
  units (m/s) rather than px/frame — either a known HEIGHT ABOVE THE GROUND (a downward-facing
  navigation flow sensor, the PX4Flow use case, paired with a rangefinder) or independent depth
  (stereo, a depth camera, or LiDAR) to scale pixel flow into metric velocity: `v_metric =
  v_pixels_per_s * height / focal_length_px` for a downward-facing camera over a locally flat surface —
  the flow itself, alone, only ever gives ANGULAR/pixel-domain motion, never absolute scale. Standard
  camera intrinsic calibration (a checkerboard/AprilTag target, `cv::calibrateCamera` or `kalibr`) is a
  bring-up prerequisite, not something this project computes.
- **The safe hardware-testing ladder** (CLAUDE.md §1's repo-wide caveat applies at full strength the
  moment optical flow feeds anything that MOVES a real robot):
  1. **Simulation** — this project's synthetic scenes ARE step one of that ladder: known ground truth,
     zero hardware risk, exactly what this repo's Definition of Done requires before anything else.
  2. **Playback / bench replay** — run the same pipeline against RECORDED real-camera footage
     (rosbag playback) with the robot's actuators disconnected or in a safe simulated loop, checking
     that flow behaves sanely on real sensor noise, real motion blur, and real rolling-shutter skew this
     project's synthetic frames do not have.
  3. **Bench jig / tethered, current-limited** — the camera mounted on the real vehicle, the vehicle
     mechanically restrained (a test stand, a tether) or power-limited, actuators live but motion
     bounded, comparing flow-derived velocity against an independent ground truth (motion capture, a
     second sensor) before trusting it for control.
  4. **Free running, with an E-stop and hard velocity/geofence limits** the entire time — the final
     rung, and the only one where this project's output could plausibly command real, unconstrained
     motion. Nothing in this repository has been run at this rung; everything here is simulation-only
     (CLAUDE.md §1).

## 4. Business & regulatory context

**Who needs this, and in which products.** Optical flow (or its learned RAFT-era descendants) is a
component inside: consumer and prosumer DRONES (hover-assist and low-altitude velocity estimation when
GPS is degraded or absent — the PX4Flow lineage's original market), indoor AMRs and service robots
(visual odometry augmenting or replacing wheel odometry on slick or obstructed floors), ADAS/AV stacks
(scene-flow and moving-object detection contribute to a broader perception stack, though modern AV
stacks lean more heavily on learned, multi-camera, multi-frame methods than the classical single-pair
algorithms taught here), and driver/operator MONITORING systems (eye/gaze and head-pose tracking use
flow-adjacent techniques, framed here — per CLAUDE.md §1 — around collaborative safety rather than
surveillance of individuals).

**Commercial and open-source players.** NVIDIA (VPI's fixed-function optical-flow hardware, §2; also
Isaac ROS's GPU-accelerated perception stack), Intel (RealSense-family depth+vision modules, and
OpenVINO-optimized classical/learned flow), the OpenCV project (the reference open-source
implementation of every classical method this project teaches, `THEORY.md`'s "Where this sits in the
real world"), and PX4/ArduPilot (the dominant open-source flight-control ecosystems that natively
consume PX4Flow-class sensor output) are the main names a learner would encounter building on this
material; the learned-flow frontier (RAFT and successors) lives mostly in research code (official
PyTorch releases) with commercial deployment increasingly folded into larger perception-model stacks
(e.g., autonomous-vehicle companies' proprietary multi-task networks) rather than shipped as standalone
libraries.

**What getting it wrong costs.** A velocity estimate that is wrong but CONFIDENT (no validity/confidence
signal, or one that is ignored downstream) is far more dangerous than one that is wrong and visibly
LOW-confidence — this is precisely why this project spends real effort on the confidence output (LK) and
validity mask (census) rather than just reporting raw flow: a drone's hover controller trusting a
brightness-constancy-violated flow estimate near a specular reflection (THEORY.md) can drift or
oscillate; an AMR trusting flow-derived odometry over a textureless floor (correctly LOW-confidence in
this project's terms) without a confidence-aware fusion layer can silently accumulate large position
error. Liability and recall exposure scale with how much control authority the flow estimate is given
and how well the surrounding system is engineered to detect and reject a bad estimate rather than act
on it blindly.

**Applicable standards / regulatory path** (see `docs/SYSTEM_DESIGN.md` item 6's regulatory map for the
full orientation): a drone using optical flow for hover-assist sits under FAA Part 107 (US) / EASA
rules (EU) as an aircraft component, not a specially-regulated sensor category on its own; an AMR using
flow-augmented odometry in a shared human workspace falls under the SAME ISO 13482 (personal-care/
service robot) or ISO 3691-4 (industrial AMR) frameworks as the rest of the robot's safety case, with
flow-derived velocity being one input among several a functional-safety analysis must characterize
(failure modes, confidence thresholds, fallback behavior) rather than a standard with flow-specific
clauses. This is orientation, not compliance guidance — CLAUDE.md §1, §8.

**Where this work lives inside a robotics company.** Perception (the team, per
`docs/SYSTEM_DESIGN.md` item 5) typically owns classical and learned optical-flow modules, working
closely with controls/autonomy (the consumer of flow-derived velocity/odometry) and, for a
flight-vehicle product specifically, the flight-software/autopilot team that owns the PX4Flow-class
sensor's firmware and fusion logic. Typical role titles: perception engineer, computer vision engineer,
and (for the GPU-kernel-authoring work this project itself is modeled on) GPU/systems engineer
embedded within perception. Adjacent teams: simulation (owns the synthetic-data and validation
pipeline this project's `make_synthetic.py` is a toy version of), and — because a bad flow estimate can
end up commanding motion — functional safety / verification.
