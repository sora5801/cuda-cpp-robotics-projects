# 01.10 — Rolling-shutter correction using IMU rates: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is a pure image-processing algorithm; there is no bespoke mechanical part to build for it.
What DOES need real construction is the **camera-IMU rig** it depends on — how the two sensors this
algorithm fuses are physically mounted together so their measurements refer to a common, known geometric
relationship:

- **Rigid, vibration-resistant mounting.** The camera and IMU must be bolted (not just adhered) to a
  single common rigid structure — any flex between them between capture and calibration silently
  invalidates the calibrated extrinsic transform this algorithm implicitly assumes is identity (see §3).
  On a drone, this usually means both live on (or are rigidly linked to) the same carbon-fiber or
  machined-aluminum camera plate, isolated from the airframe's own vibration by soft mounts tuned to
  damp the propeller's vibration frequency band WITHOUT damping the frequencies the gyro needs to sense
  accurately — a genuine mechanical-design trade-off (over-isolating the IMU makes it blind to real
  airframe motion; under-isolating it lets propeller buzz swamp the useful signal).
- **The lever arm.** Even rigidly mounted, the camera's optical center and the IMU's sensing origin are
  physically some distance apart (millimeters to centimeters). This project's row-homography math
  assumes PURE ROTATION about a single point (§ THEORY.md "The math") — a nonzero lever arm means the
  IMU's measured rotation is technically about the IMU's own origin, not the camera's, introducing a
  small translation-like error during any angular acceleration. Real rigs calibrate and account for this
  lever arm explicitly (Kalibr, §3, estimates it as part of the extrinsic calibration).
- **What breaks in the field**: loose mounting screws (reintroducing the flex problem above after a hard
  landing or drop), connector/cable strain at the IMU (a ribbon or wire harness pulling on the sensor
  board flexes its mount slightly, enough to matter at the sub-degree precision this correction wants),
  and thermal drift of adhesive-mounted sensors (adhesive creeps under sustained vibration + heat,
  slowly changing the extrinsic calibration over the vehicle's service life).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute** this algorithm would run on: it is cheap enough (a few ms per frame at this project's test
resolution, README "Expected output") to run on almost anything with a GPU, but a real robot pipeline
would place it right after image acquisition:
- *Hobby tier*: a Raspberry Pi + a small USB accelerator, or folded into a Jetson Nano-class board's
  ISP/GPU pipeline alongside other perception work.
- *Research tier*: NVIDIA Jetson Orin NX/AGX — enough GPU headroom to run this correction plus the
  downstream feature/flow/SLAM pipeline it feeds, all on one board.
- *Industrial tier*: an x86 + discrete RTX-class GPU industrial PC (fanless, wide-temperature-range),
  common in fixed-installation or larger-vehicle robotics where power/space budgets are looser.

**Sensors**:
- *Rolling-shutter camera*: essentially any consumer or embedded CMOS camera module — global-shutter
  parts exist (see THEORY.md) but cost more per pixel and per frame rate, so most drone/handheld/AR
  cameras are rolling-shutter by default, which is exactly why this correction exists. Illustrative
  interfaces: MIPI CSI-2 (embedded/mobile), USB3 Vision or GigE Vision (industrial), depending on tier.
- *Gyro / IMU*: a 6-DoF (accel+gyro) or 9-DoF (+magnetometer) MEMS IMU sampled at 200 Hz-1 kHz.
  Illustrative parts: Bosch BMI088/BMI270 (consumer/hobby drones), InvenSense/TDK ICM-42688-P (a common
  "VIO-grade" choice in research quadrotors), or an integrated flight-controller IMU (e.g. inside a
  Pixhawk-class autopilot) if the camera pipeline can access its raw angular-rate stream.
- *Hardware time-sync*: a camera with an FSYNC/strobe output (asserted at the start of each frame's
  exposure) wired into the IMU's or flight controller's timestamp-capture input — the hardware answer to
  §3's "which clock is which sample on" problem, preferred over software-only timestamping whenever the
  camera module supports it.

**What this correction does NOT need**: no motor-control silicon, no actuation chain, no power
electronics beyond what the camera/IMU/compute already need — this is a pure sensing/compute pipeline
component, not an actuator.

## 3. Installation & integration — putting it on a real robot

**Where it runs**: on the same compute that receives the camera stream — typically the perception/
companion computer (Jetson-class or x86+dGPU), NOT the flight-controller MCU (too little compute for a
per-pixel GPU kernel, and the FC's job is the 0.5-1 kHz attitude loop, not image processing).

**OS/real-time**: ordinary Linux (Ubuntu/JetPack) is sufficient — this is a soft-real-time perception
step (must finish within one camera frame period, ~16-33 ms at 30-60 Hz), not a hard-real-time control
loop; no RTOS needed.

**ROS 2 node shape**: a node subscribing to `sensor_msgs/Image` (the raw RS frame) and
`sensor_msgs/Imu` (angular_velocity), synchronized with `message_filters::TimeSynchronizer` (or an
approximate-time policy, since the IMU arrives far more often than the camera) buffering the IMU stream
across each frame's readout window, and publishing a corrected `sensor_msgs/Image` for downstream
feature/flow/SLAM nodes — the exact `Image`/`ImuSample` shapes named in README "System context".

**Calibration and bring-up procedure** (what must be measured before this algorithm's numbers mean
anything on real hardware):
1. **Camera intrinsics** (`fx, fy, cx, cy` + lens distortion) — standard checkerboard/AprilGrid
   calibration (see 01.16/01.06 in this repo for the GPU-accelerated versions of the detectors involved).
2. **Camera-IMU extrinsic calibration + line-readout time + time offset** — **Kalibr**
   (github.com/ethz-asl/kalibr) is the standard open-source tool for exactly this: given a recorded
   checkerboard/AprilGrid sequence plus synchronized IMU data, it jointly estimates the camera-IMU
   rotation/translation (the "lever arm", §1), the camera's `t_line` (this project's `kLineTimeS`), AND
   the camera-IMU time offset (this project's "assumes perfect sync" simplification, README
   "Limitations & honesty") — all three numbers this project's `kernels.cuh` otherwise takes as given.
3. **Readout-time measurement, standalone** (if not using Kalibr's joint estimate): record video of a
   fast, KNOWN angular-velocity rotation (e.g. a calibrated turntable, or a bright LED strobing at a
   known frequency swept across the frame) and measure the resulting row-dependent shift to back out
   `t_line` directly — a classic sensor-characterization technique, doable on a bench with no robot at all.

**Safe hardware-testing ladder** (per this repo's motion-of-real-hardware caveat, CLAUDE.md §1):
simulation (this project, exactly as built — offline over synthetic data) → **hardware-in-the-loop**
(replay recorded real camera+IMU logs through the same code, no motion commanded) → **bench jig /
tethered** (mount the camera+IMU rig on a controlled turntable or handheld jig, motion limited and
observed, still no vehicle motion) → **free running** (the correction feeding a live downstream SLAM/
flow pipeline on an actually-flying or actually-moving platform) — with the property that at every rung,
THIS project's own code never commands any actuator; it only ever consumes a camera+IMU stream and emits
a corrected image. Nothing here should be mistaken for a flight-safety-relevant component in its own
right, but a SILENTLY WRONG correction feeding a SLAM/VIO pipeline that DOES command motion is a real,
indirect hazard worth testing through this ladder rather than skipping straight to free flight.

## 4. Business & regulatory context

**Who needs this capability**: any company shipping a camera on a platform that rotates fast enough,
relative to its frame rate, to visibly distort images — **drone manufacturers** (the dominant case:
propeller vibration + aggressive flight maneuvers), **handheld/mobile mapping and 3-D scanning** vendors
(human hand tremor plus deliberate fast sweeps during a scan), and **AR/VR headset** makers (head rotation
during 6-DoF tracking, where RS artifacts directly corrupt the visual-inertial tracking that keeps virtual
content anchored to the real world).

**Main players**: camera-module and IMU vendors (Sony/OmniVision sensors; Bosch/TDK-InvenSense IMUs) sell
the raw hardware; VIO/SLAM software vendors (e.g. companies building on OKVIS/VINS-Fusion-class stacks, or
proprietary equivalents inside drone and AR companies) build the RS-aware estimation THEORY.md describes
as the more common production path; Kalibr and similar open-source tools (§3) are the de facto standard
calibration layer nearly everyone in this space uses or forks from.

**What getting it wrong costs**: uncorrected (or badly corrected) rolling shutter is one of the
best-documented real-world sources of VIO/SLAM pose error on rotating platforms — in a mapping product
this shows up as warped or misaligned 3-D reconstructions (a direct product-quality and customer-refund
cost); in a navigation/obstacle-avoidance stack it degrades the pose estimate that safety margins are
computed from (an indirect safety cost, several layers removed from this correction step itself, but real
— see the safe-testing-ladder note in §3).

**Regulatory path**: this project's output (an image) does not itself trigger a distinct regulatory
category, but the PLATFORMS that need it are heavily regulated — drones under FAA Part 107 (US) / EASA
categories (EU), and any camera/IMU-derived pose feeding an autonomous-navigation decision on a vehicle
falls under that vehicle's own regulatory umbrella (SYSTEM_DESIGN.md §6.2's table: e.g. ISO 26262/UL 4600
for ground AVs, if this kind of correction were ever used in that context). This is an orientation
pointer, not compliance guidance (SYSTEM_DESIGN.md §6.2's label applies here verbatim).

**Where this work lives inside a robotics company**: the **perception** team (SYSTEM_DESIGN.md §5.1),
usually the engineers who own the camera driver and the front end of the SLAM/VIO stack; adjacent teams
are **embedded/firmware** (who own the camera/IMU hardware timestamping this project's "perfect sync"
assumption depends on, §3) and **controls & autonomy** (who own the downstream SLAM/VIO consumer named in
README "System context"). Typical role titles: perception engineer, computer vision engineer, or (at
companies with a dedicated function) SLAM/VIO engineer.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
