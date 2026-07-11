# 01.08 — HDR exposure fusion + tone mapping for outdoor robots: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5-6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is an **outdoor camera module**: the lens/sensor/housing assembly a
robot mounts to actually acquire the bracketed exposures this code processes.

- **Lens & sensor stack.** A machine-vision lens (usually a fixed-focal-length, low-distortion design —
  see 01.01-full-gpu-image-pipeline and 01.07-fisheye-omnidirectional-unwarping for the distortion-
  correction side of this) is mounted, aligned, and epoxy- or set-screw-locked to a sensor PCB. Focus and
  back-focus are set once at assembly and typically locked; a vibrating vehicle platform that lost lock
  would defocus every frame, HDR or not.
- **Housing & sealing.** An outdoor unit needs an IP65/IP67-rated housing: a machined or die-cast
  aluminum or polycarbonate shell, an O-ring or gasket seal at every mating face, a conformally-coated or
  potted PCB, and cable glands (not bare connectors) at every wire entry. Water ingress at any seam is
  the single most common outdoor-camera field failure.
- **Lens hood / anti-flare shading.** This project's synthetic scene puts a sun disk directly in frame —
  a real lens facing that would need a physical hood or baffle to control flare and ghosting (internal
  lens reflections of the bright source), an OPTICAL problem no amount of downstream HDR software fixes;
  see the honesty note in Section 2 below.
- **Thermal path.** A sensor in direct sun can self-heat well above ambient; a metal housing acting as a
  heat sink (thermal pad from sensor PCB to housing wall) keeps dark current and read noise from
  drifting significantly over a bracket's capture window (THEORY.md's "Engineering constraints" section).
- **Mounting & vibration isolation.** Rigidly bolted to a vehicle chassis, a camera inherits every bump
  and vibration mode of the platform; some designs add a compliant (rubber or elastomer) mount between
  camera and chassis specifically to reduce motion blur — directly relevant here, since HDR bracketing's
  multi-frame capture window is *more* motion-sensitive than a single exposure (README "Limitations &
  honesty").
- **What breaks in the field:** seal degradation (UV breaks down rubber gaskets over 1-2 years outdoors,
  eventually leaking), connector corrosion at unsealed joints, lens-hood impact damage, and thermal-cycle
  solder-joint fatigue on PCBs mounted near a hot housing wall.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Compute | Sensor | Notes |
|---|---|---|---|
| Hobby / prototype | Raspberry Pi 5 + a USB3/CSI camera | A rolling-shutter CMOS module (e.g., Sony IMX series, ~$20-60) with manual exposure control exposed via `libcamera`/V4L2 | No native HDR-sensor mode; bracketing done entirely in software, exactly this project's approach |
| Research / mid-tier robotics | NVIDIA Jetson Orin NX/Nano (the GPU tier this repo targets — CLAUDE.md §5) | A global-shutter industrial camera (e.g., a Sony IMX-series-based GigE/USB3 Vision module, ~$200-600) with hardware-triggered exposure bracketing | Global shutter matters more for a moving platform than for HDR per se, but the two compound (rolling-shutter + long-exposure HDR frame = worse motion smear) |
| Industrial / production AV | x86 + discrete GPU, or an automotive SoC (e.g., an ISP-integrated automotive vision chip) | A **sensor-level HDR** part: a dual-conversion-gain or split-pixel/DOL-HDR sensor (e.g., a Sony automotive-grade HDR CMOS) | This is the "buy the dynamic range in silicon" alternative to this project's software bracketing — see THEORY.md "Where this sits in the real world" |

- **Compute:** the demo here runs comfortably on a desktop RTX-class GPU (the repo's reference machine,
  CLAUDE.md); a fielded robot's equivalent tier is Jetson Orin-class (compute) vs. an x86+dGPU box for a
  larger AV-class platform vs. an MCU-class front end for a cost-constrained AMR that offloads HDR/tone
  mapping to a shared perception compute box rather than doing it at the camera.
- **Sensor interface:** MIPI CSI-2 (short-run, board-to-board), or GigE Vision / USB3 Vision (longer
  cable runs, more common on industrial/robotics camera modules where the camera and compute box are
  physically separated).
- **Lens hood / ND options:** a physical neutral-density (ND) filter or a mechanically-actuated iris is a
  hardware-level partial answer to extreme dynamic range (reduce the brightest part of the scene before
  it ever reaches the sensor) — complementary to, not a replacement for, HDR software.

## 3. Installation & integration — putting it on a real robot

- **Where this runs.** On the perception compute tier — the same GPU-equipped box (Jetson-class embedded
  or x86+dGPU, per Section 2) that runs the rest of the vision stack (01.01's debayer pipeline,
  01.04's feature detection), NOT on an MCU. This project's four-exposure, ~1-2 ms-of-compute pipeline
  (`demo/expected_output.txt`'s `[time]` lines) is cheap relative to the ~150 ms of CAPTURE time the
  bracket itself needs (THEORY.md), so the compute budget is not the bottleneck here — the capture
  cadence is.
- **OS / real-time constraints.** Standard Linux (Ubuntu/JetPack) is adequate: this is NOT a hard-real-
  time control loop (docs/SYSTEM_DESIGN.md's kHz control tier), it is a perception pre-processing stage
  feeding downstream vision at whatever cadence the bracket capture itself allows.
- **ROS 2 node shape.** A `hdr_fusion_node` subscribing to a bracket topic (`N` synchronized
  `sensor_msgs/Image` messages, or a custom `ExposureBracket` message bundling them with per-frame
  exposure-time metadata — matching this repo's message-shaped-interface convention,
  `docs/SYSTEM_DESIGN.md` item 3) and publishing one fused/tone-mapped `sensor_msgs/Image` downstream.
  Real systems typically run this as a periodic (not per-video-frame) service — e.g., re-exposing a
  fixed workspace or dock every few seconds — rather than in the main 30-60 Hz perception loop, given the
  capture-time cost above.
- **Calibration & bring-up.** The CRF is a property of the SENSOR + ISP pipeline, not the scene — a real
  deployment calibrates it ONCE (e.g., photographing a bracket of a static, evenly-lit calibration target
  at manufacturing time or first bring-up) and reuses the recovered curve indefinitely, rather than
  re-running `crf_solve_debevec` on every capture as this teaching demo does for clarity. Exposure-time
  synchronization across the bracket (and, for a moving platform, wheel-odometry/IMU timestamps bracketing
  the capture window) matters for any downstream motion-compensation extension (README "Limitations &
  honesty" names this as explicitly out of scope here).
- **The safe hardware-testing ladder.** This project's output is a perception IMAGE, never a motion
  command, so the usual actuator-testing ladder (simulation -> HIL -> bench-jig/tethered -> free-running,
  with E-stop and limits at every rung) is largely **N/A** here in the strict sense — nothing in this
  pipeline drives an actuator. The one practical caution: if a downstream consumer (e.g., a
  vision-based obstacle detector) ever gates a motion decision on this pipeline's output, that
  consumer's own testing ladder must account for HDR fusion's failure modes (motion ghosting in a moving
  scene, an under-covered dynamic-range tail — README "Limitations & honesty") as an input-quality risk,
  same as it would for a raw, unfused camera feed. Per CLAUDE.md §1: everything in this repository is
  sim-validated only, not safety-certified.

## 4. Business & regulatory context

*Didactic orientation, dated 2026-07-10 — not procurement, legal, or compliance advice.*

- **Who needs this.** Any outdoor-operating robot with a camera in its perception stack: delivery/yard
  AMRs, agricultural robots (30.01-crop-detection-and-yield-estimation, named in README), construction/
  field robotics, and autonomous vehicles — anywhere a single auto-exposed frame would lose either the
  shadowed region or the sunlit region of a scene that matters for the task.
- **Commercial and open-source players.** OpenCV (open-source, the reference implementation of every
  algorithm here, THEORY.md); camera/sensor vendors building sensor-level HDR silicon (Sony's automotive
  and industrial DOL-HDR sensor lines being the most cited example in this space); ISP IP vendors
  licensing HDR tone-mapping blocks into automotive/robotics SoCs; and every AV/ADAS company's in-house
  perception team, since image-quality-under-extreme-dynamic-range is a near-universal requirement for
  camera-based autonomy.
- **What getting it wrong costs.** For a delivery or warehouse-yard AMR: a missed obstacle or a
  misread lane/dock marking in a shadowed or sun-glared region translates directly into safety incidents
  or operational downtime (a stopped robot, a support call). For an AV/ADAS system this is a much higher-
  stakes version of the same failure — image-quality-under-adverse-lighting is a named, tested category
  in automotive perception validation.
- **Standards / regulatory path.** For automotive/ADAS cameras specifically: **IEEE P2020** ("Automotive
  Image Quality") is the standards effort most directly on-topic for this project — it defines objective
  image-quality metrics (including dynamic-range and HDR-related measures) for automotive imaging
  systems; cited here as orientation, not as a claim of compliance (this project computes none of
  P2020's specific metrics). More broadly, this work sits under whatever functional-safety framework
  governs the platform it ships in — **ISO 26262** (road vehicles) or **UL 4600** (autonomous vehicles)
  if this pipeline's output ever feeds a safety-relevant perception decision; see
  `docs/SYSTEM_DESIGN.md` item 6 for the full regulatory-landscape map this project's outputs would sit
  under, by robot type.
- **Where this work lives inside a robotics company.** Perception / camera-systems engineering owns
  sensor selection, ISP/HDR tuning, and this code's production equivalent; adjacent teams are
  optics/mechanical (lens, housing, thermal — Section 1), the broader perception team (everything
  downstream that consumes a well-exposed image, `docs/SYSTEM_DESIGN.md` item 5), and — for an
  automotive-grade product — a dedicated image-quality/validation function responsible for exactly the
  kind of adverse-lighting testing this project's `dynamic_range_coverage` gate is a toy version of.
