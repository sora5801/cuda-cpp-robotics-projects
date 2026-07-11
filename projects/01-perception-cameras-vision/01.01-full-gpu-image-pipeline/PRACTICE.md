# 01.01 — Full GPU image pipeline: debayer → undistort → rectify → resize → normalize, zero CPU copies: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

A machine-vision camera module — the physical thing this pipeline's input comes from — is a small,
tightly toleranced electromechanical assembly, typically built in this stack (front to back):

- **Lens barrel**: a small stack of 3-8 glass or molded-plastic elements, held in precise alignment
  by a metal or engineering-plastic barrel. Machine-vision lenses use an **M12 (S-mount)** or
  **C/CS-mount** threaded interface for focus/back-focus adjustment during assembly, then are usually
  LOCKED with a set screw or thread-locking adhesive once focused — a lens that can drift its
  back-focus distance after calibration silently invalidates every `fx/fy/cx/cy` this project's
  camera model assumes fixed.
- **IR-cut filter**: a thin glass filter (often coated directly onto the lens's last element or a
  separate glued disc) blocking near-infrared light, which the Bayer color filters cannot separate
  from visible light and which would otherwise wash out color accuracy — omitted deliberately on
  cameras built for active-IR depth or night vision.
- **Sensor + CFA**: the bare CMOS die with its Bayer filter array (THEORY.md's "physics" section)
  bonded on top, itself soldered or socketed onto a small PCB (the "sensor board").
  **Mounting tolerance here — the barrel's alignment to the sensor board's normal — is EXACTLY the
  physical source of the small rotation `kRectifyAngleDeg` in this project's camera model.**
- **Housing + mount**: an aluminum or plastic enclosure that (a) holds the lens barrel and sensor
  board in fixed relative position, and (b) provides the mechanical interface (screw holes, a
  quick-release plate, or a printed bracket) to the robot chassis. On a real robot this interface's
  own tolerance ADDS to the lens-to-sensor tolerance above — the two are indistinguishable from a
  single image, which is exactly why calibration (measuring the combined effect) rather than
  spec-sheet trust is standard practice.
- **Cabling**: a flex-cable (FPC) or a MIPI CSI-2 ribbon runs from the sensor board to the compute
  module — short (centimeters, for signal integrity) on an integrated module, or a longer shielded
  cable (FAKRA, GMSL) for a remote/vehicle-mounted camera (see §2 below).
- **What breaks in the field**: connector/FPC fatigue from vibration (a common field failure on
  mobile robots — repeated flexing at a cable's fixed bend radius fractures traces over
  months-to-years), condensation inside a poorly sealed housing (fogs the lens or shorts the PCB),
  and — directly relevant to this project — **thermal or mechanical shock shifting the lens's
  back-focus or the housing's mounting angle**, silently invalidating a prior calibration without any
  visible external damage. This is why production robots periodically re-verify calibration rather
  than trusting a one-time factory measurement forever.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute this pipeline would run on** (all three tiers can run this exact algorithm; the choice
trades cost, power, and integration complexity):

| Tier | Example | Where debayer+undistort+rectify+resize actually run | Rough cost (2026, verify current) |
|------|---------|-------------------------------------------------------|-------------------------------------|
| Hobby/research | NVIDIA Jetson Orin Nano dev kit | Jetson's dedicated ISP hardware block (libargus), OR this project's kernels on the onboard GPU | ~$250 |
| Research/prototype | x86 mini-PC + discrete RTX-class GPU (e.g. RTX 4000 Ada, laptop-class) | This project's exact kernels — GPU debayer/remap/resize is common when the ISP hardware's algorithm quality is insufficient (e.g. HDR or algorithmic demosaic tuning) | ~$1,500-4,000 |
| Industrial | Basler/FLIR camera + dedicated frame grabber + industrial PC (Jetson AGX Orin or Xeon+GPU class) | Camera's onboard FPGA/ISP does debayer at the sensor; host GPU/CPU does undistort/rectify/resize/normalize | $3,000-15,000+ per camera system |

**Sensor + lens** (the thing whose physics THEORY.md describes): a **global-shutter** CMOS sensor
(e.g. Sony IMX-family industrial parts, Sensing/OnSemi AR-family) is strongly preferred over rolling
shutter for anything the robot itself is moving (THEORY.md's rolling-shutter honesty note); lens
choice trades field of view against distortion severity — wider FOV lenses have LARGER `|k1|`, giving
this project's undistort stage more real work to do. Interfaces: **MIPI CSI-2** (short, on-board,
common on embedded/Jetson designs — centimeters of cable), **GMSL2** (Maxim/Analog Devices — coax or
shielded twisted pair, tens of meters, the automotive/robotics standard for remote camera mounting
with power-over-cable), or **GigE Vision** (standard Ethernet, tens to 100 meters, common in
industrial machine vision where cable length or PoE convenience matters more than the extra latency).

**Actuation/comms silicon this pipeline talks to**: N/A directly — this pipeline only consumes camera
data and produces an image tensor; it commands nothing. (Downstream consumers — planners, controllers
— are where domain 08/24's actuation-chain hardware becomes relevant.)

## 3. Installation & integration — putting it on a real robot

**Where this code would run**: on the robot's main perception compute (the GPU SoC or x86+dGPU tier
in SYSTEM_DESIGN.md §6.1's hardware diagram) — NOT on a real-time MCU; this pipeline's ~1 ms/frame at
384x288 (measured, README "Expected output") scales to single-digit milliseconds at real sensor
resolutions (1-4 MP), comfortably inside the 16-33 ms camera->perception budget on a soft-real-time
Linux + GPU stack, never a hard-real-time deadline.

**OS / real-time constraints**: Linux (Ubuntu on Jetson, or a robot-specific distro) is standard;
this pipeline has no hard-real-time requirement of its own (a missed frame just delays perception by
one camera period, not a safety fault), so a normal (non-PREEMPT_RT) kernel is typically fine unless
the whole perception stack has tighter jitter requirements.

**ROS 2 node/topic shape** (SYSTEM_DESIGN.md §3.6): this pipeline is naturally one node —
subscribes to a raw `sensor_msgs/msg/Image` (`encoding="bayer_rggb8"`) on a topic like
`/camera/image_raw`, publishes a rectified `sensor_msgs/msg/Image` (`encoding="rgb8"`) on
`/camera/image_rect_color` (matching the real `image_proc`/`image_pipeline` package's topic
convention — see README "Prior art") plus the final normalized tensor on a project-specific topic (or,
more commonly in production, hands the tensor directly to an in-process inference node rather than
serializing it over a topic, to avoid the copy/serialization cost).

**Calibration bring-up procedure** (this is the step that actually MEASURES the
`fx,fy,cx,cy,k1,k2,p1,p2` this project hardcodes): (1) print/mount a checkerboard or AprilTag/ChArUco
calibration target of known physical square size; (2) capture 20-50 images of it from varied
angles/distances filling the frame, using ROS's `camera_calibration` package or Kalibr; (3) the tool
solves for intrinsics + distortion via Zhang's method (THEORY.md "Where this sits in the real world");
(4) verify reprojection error is sub-pixel (typically < 0.5 px RMS) before trusting the result; (5)
store the calibration (a YAML file in ROS convention) alongside the camera's serial number — swapping
a lens or a sensor board invalidates it.

**Safe hardware-testing ladder** (CLAUDE.md §1's caveat applies in full — this repo's code is
sim-validated only): (1) **simulation** — this project's synthetic Bayer scene IS step one of that
ladder; (2) **HIL (hardware-in-the-loop)** — feed the real camera a known calibration target and
compare the pipeline's output against ground truth, exactly what this project's `color_fidelity` and
`straightness_rectified` gates do, just with synthetic data instead of a real sensor; (3) **bench
jig** — mount the real camera on a fixed rig, verify calibration stability across temperature/vibration
before installing on the robot; (4) **on-robot, stationary** — verify the live pipeline output on the
actual robot before any motion; (5) **on-robot, in motion** — only after (1)-(4) pass, and only
because THIS pipeline commands no motion itself, the "free running" rung that matters is whatever
consumes its output (a planner/controller), not this node.

## 4. Business & regulatory context

**Who needs this**: every company building a camera-equipped robot needs SOME version of this
pipeline — it is table-stakes infrastructure, not a product differentiator (SYSTEM_DESIGN.md §5.3's
build-vs-buy framing: most teams buy or adopt an existing ISP/pipeline rather than hand-rolling one,
UNLESS camera quality or a specific algorithmic tweak IS their differentiator, e.g. a company
building night-vision or HDR-heavy perception).

**Commercial and open-source players**: camera/ISP vendors (Sony, OnSemi, OmniVision for sensors;
NVIDIA's libargus/VPI, Qualcomm's Spectra ISP for embedded ISP stacks), machine-vision integrators
(Basler, FLIR/Teledyne, Cognex — sell calibrated camera+software systems, not just sensors), and the
open-source `image_pipeline`/`image_proc` ROS packages that most robotics teams actually use rather
than writing this from scratch.

**What getting it wrong costs**: a wrong or drifted calibration silently corrupts every downstream
measurement — a stereo depth estimate (01.02), a visual-SLAM pose, an object's estimated distance for
a manipulator's grasp — usually WITHOUT an obvious symptom (the image still "looks fine" to a human
reviewing a live feed), making this a classic silent-failure risk. In a shipped product, this
translates to unreliable perception (false obstacle distances, missed grasps, drifted SLAM maps) that
is expensive to diagnose in the field, since the camera itself rarely reports a fault.

**Regulatory path**: this pipeline itself is not directly named in a standard, but it is
FOUNDATIONAL to systems that are: for a mobile robot sharing space with people, perception quality
feeds directly into the hazard analysis ISO 13482 requires; for an autonomous vehicle, camera
calibration integrity is part of the safety case ISO 26262/UL 4600 expect evidence for
(SYSTEM_DESIGN.md §6.2's regulatory map). This section is **orientation, not compliance guidance**.

**Where this work lives in a robotics company**: the perception / camera-systems team
(SYSTEM_DESIGN.md §5.1's "Perception" row, domains 01/02/03/20) typically owns this exact
boundary — often the very FIRST thing a new perception engineer on a robotics team works on, since
every other perception project (feature detection, depth, SLAM) depends on this pipeline being
correct first. Adjacent teams: embedded/firmware (owns the actual sensor driver / MIPI bring-up
below this pipeline), and ML/data (owns the neural network this pipeline's normalized tensor feeds
into).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
