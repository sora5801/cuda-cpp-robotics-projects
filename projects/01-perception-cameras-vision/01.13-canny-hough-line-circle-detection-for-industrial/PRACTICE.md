# 01.13 — Canny + Hough line/circle detection for industrial alignment: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is an **industrial vision STATION**, not a single part — the camera,
lens, lighting, and mounting hardware that would actually produce the kind of image `scripts/make_synthetic.py`
renders. A typical station bolts together from four subsystems:

- **Mechanical mounting.** The camera and lens assemble onto a rigid stand or gantry arm, usually via a
  standard C-mount or S-mount thread, clamped to a extruded-aluminum frame (80/20-style T-slot rail is
  the de facto industry default) so the whole optical path can be re-aimed and locked during
  commissioning, then never touched again — vibration and thermal drift between the camera and the part
  are the enemy of repeatable sub-pixel measurement. The frame typically also carries the lighting
  fixture at a fixed standoff, since lighting GEOMETRY (angle, diffusion, distance) is as much a
  calibrated parameter as the lens focus.
- **The part fixture.** The plate this project detects would sit in (or pass through, on a conveyor) a
  locating fixture — pins, a vee-block, or a vacuum chuck — whose whole PURPOSE is to bound the part's
  position/orientation error to something small enough that this project's alignment pipeline has to
  correct only a few degrees and a few millimeters, not search the whole field of view. The "known
  in-plane offset and rotation" this project recovers IS the fixture's residual tolerance, made visible.
- **Electrical/wiring.** Industrial cameras are usually powered and triggered over the same cable
  (GigE Vision cameras commonly use Power-over-Ethernet or a separate 12/24 V hardwired supply plus a
  shielded I/O cable for the hardware trigger and strobe lines — see §3). Cabling is routed in
  drag-chain or flexible conduit if the camera moves with a gantry axis, and shielded/twisted-pair for
  any signal running near variable-frequency motor drives (a very common source of image noise on a
  real factory floor that this project's synthetic sensor noise only crudely stands in for).
- **What breaks in the field.** Lens focus/aperture rings creep under vibration if not locked with a
  set-screw or thread-locker; connectors work loose from repeated cable flex; lighting LEDs dim over
  their service life (gradually raising the effective noise floor relative to signal — exactly the kind
  of drift a `hysteresis_lesson`-style weak-edge margin exists to tolerate); and any change to standoff
  distance (a bumped camera, a re-tensioned belt) invalidates the pixel-to-mm calibration in §3 until
  it is redone.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute tier.** This exact pipeline (Canny + 2 Hough transforms on a 320x240-to-a-few-MP image, tens
of milliseconds) comfortably fits three very different compute tiers, in ascending cost/capability:

| Tier | Example | Notes |
|------|---------|-------|
| Embedded/edge | NVIDIA Jetson Orin Nano/NX | Runs this pipeline at full-resolution industrial-camera sizes in real time; the usual choice when the vision station is a self-contained smart-camera unit. |
| Desktop/industrial PC | Small-form-factor PC + a discrete RTX-class GPU (or even integrated graphics, for Canny/Hough at this problem size) | This project's own development target (an RTX 2080 SUPER desktop). |
| MCU-class (no GPU at all) | A dedicated FPGA or DSP "smart camera" (e.g., Cognex In-Sight, Keyence CV-X) | Production metrology cameras often run these algorithms on fixed-function silicon inside the camera body itself — no separate PC. |

**Camera & sensor.** A global-shutter (not rolling-shutter — see project 01.10) machine-vision camera,
e.g. a Basler ace or FLIR Blackfly S class, 1-5 MP monochrome sensor (color is rarely needed for pure
geometric alignment and costs contrast/resolution for no benefit here), GigE Vision or USB3 Vision
interface. Hobby tier: a Raspberry Pi HQ Camera or a USB webcam (rolling shutter, no hardware trigger —
workable for a bench demo, not for a moving line). Industrial tier: the Basler/FLIR class above plus a
telecentric or low-distortion fixed-focal-length lens (see §3's calibration note on why lens choice
matters for metrology).

**Lighting.** A ring light or diffuse dome LED illuminator (brightfield, matching this project's scene)
or a backlight panel (for silhouette/through-hole work — see `THEORY.md`'s lighting paragraph),
typically 24 V DC, driven through a dedicated LED lighting controller that also provides the STROBE
signal (see §3).

**Actuation chain silicon (if this feeds a robot, per README's downstream-consumer note).** Out of this
project's own scope, but the manipulator that acts on the recovered alignment would carry its own
motor-control MCU (e.g. an STM32-class part running FOC), gate drivers, current-sense amplifiers, and
absolute encoder ICs — see project domain 24 (actuators & motors) for that chain in depth.

**Rough cost tiers (illustrative, 2026, verify current):** hobby (USB camera + LED strip): under $200.
Research/prototype (machine-vision camera + C-mount lens + basic ring light + a Jetson): roughly
$800-$2,500. Industrial (certified machine-vision camera, telecentric lens, calibrated lighting
controller, smart-camera or industrial PC): $3,000-$15,000+ per station, before integration labor.

## 3. Installation & integration — putting it on a real robot

**Where this code runs.** On the compute tier from §2 — most commonly a small industrial PC or a
Jetson-class module physically mounted near the vision station (not on the robot's own main controller,
to keep the camera's own vibration/thermal environment separate from the robot's).

**ROS 2 shape.** This pipeline would naturally be one ROS 2 node subscribing to `sensor_msgs/Image` (or
this repo's own `Image` message-shape convention, see `docs/SYSTEM_DESIGN.md` item 3) from a camera
driver node, and publishing a `geometry_msgs/Pose2D`-shaped alignment result (or this project's own
`AlignmentResult` struct, message-ified) on a topic like `/part_alignment`, at the trigger rate (not a
continuous stream — see below).

**Trigger/strobe wiring — the real-time-critical part.** A production station does NOT run this
pipeline on a free-running video stream; it HARDWARE-TRIGGERS one exposure per part, synchronized to a
part-present sensor (a photoeye or the PLC's own motion profile), and the camera drives a STROBE output
that fires the light for exactly the exposure window — this is what freezes motion blur on a moving
conveyor without needing an expensive ultra-short exposure at full continuous illumination power.
Trigger and strobe are simple TTL/opto-isolated digital I/O lines, wired directly between the camera, the
light controller, and the PLC or a dedicated I/O module.

**PLC handshake.** The vision PC and the line's PLC talk over a fieldbus — commonly **EtherCAT** or
**PROFINET** in a modern cell, or plain digital I/O (a "part good"/"part bad" relay pair) in a simpler
one; the vision result (accept/reject, or the recovered offset for a robot pick correction) is written
to a shared PLC register or published on the fieldbus for the robot/actuator to consume within the
part's cycle-time budget (README's rate/latency section).

**Calibration to millimeters.** This project works entirely in pixels (README "Limitations"); a real
station calibrates pixel-to-mm and corrects lens distortion using a checkerboard or ChArUco target —
project **01.16**'s checkerboard/ChArUco detection is exactly that calibration step, run once at
commissioning (and re-run any time the camera or lens is disturbed). The resulting intrinsic matrix and
distortion coefficients turn this project's `(dx_px, dy_px, dtheta)` into `(dx_mm, dy_mm, dtheta)`, the
units a robot's motion planner actually needs.

**GigE Vision / GenICam.** Most industrial cameras (including the Basler/FLIR examples in §2) expose a
standardized **GenICam** feature interface over **GigE Vision** or **USB3 Vision** transport — the
standard that lets any compliant vision software (this project's own capture code, if extended to a live
camera instead of `data/sample/`, or a library like Aravis/pylon/Spinnaker) discover and configure
exposure, gain, trigger mode, and strobe timing without camera-specific driver code.

**The safe hardware-testing ladder (CLAUDE.md §1 — sim-validated only, not safety-certified).** If this
pipeline's output ever drives a real robot's pick correction: (1) **simulation** — exactly what this
project's demo does, on synthetic data with known ground truth; (2) **HIL** — replay real camera frames
(or this project's synthetic frames) through the actual vision PC and PLC hardware with the robot
disconnected or in simulation, checking the correction command is sane before any motor moves; (3)
**bench jig / tethered / current-limited** — the robot arm on a bench, motion-limited, E-stop within
reach, verifying the correction physically moves the gripper the right amount on a few known offsets;
(4) **free running** — full-speed production, only after (1)-(3) pass repeatedly and a human has
reviewed the failure modes. E-stop and software travel limits apply at every rung; this repository
provides none of the safety interlocks a real installation requires.

## 4. Business & regulatory context

*Didactic orientation only — **not** procurement, legal, or compliance advice.*

**Who needs this.** Any discrete-parts manufacturer doing assembly, machining, or packaging: automotive
and electronics assembly (checking a bracket or PCB is correctly seated before a robot places the next
part), general machining (verifying a part is correctly clamped before a CNC operation begins), and
logistics/packaging (aligning a label or verifying a box's position on a conveyor). This is one of the
highest-volume, most mature applications of computer vision in industry — often called "machine vision"
specifically to distinguish it from consumer/research computer vision, precisely because of its
metrology and reliability requirements.

**Commercial and open-source players.** Commercial machine-vision leaders include **Cognex**
(In-Sight smart cameras, VisionPro software), **Keyence** (CV-X series), and **MVTec** (Halcon software
library, used by many system integrators rather than sold as a standalone camera). On the open-source
side, **OpenCV** (with its CUDA module) is the dominant toolkit system integrators and researchers build
custom pipelines on top of — exactly the library this project's kernels teach toward
(see [`README.md`](README.md#prior-art--further-reading)).

**What getting it wrong costs.** A missed misalignment can mean a robot crashing its gripper into a
mis-seated part (tooling damage, unplanned downtime), a defective assembly shipping (warranty/recall
cost, and in regulated industries like automotive or medical-device manufacturing, a formal quality
escape investigation), or — at the safety-critical end — a part released into a process it was not
correctly positioned for (e.g., an under-clamped part in a CNC operation). This is why real stations
pair detection accuracy with the **gauge repeatability & reproducibility (Gauge R&R)** discipline below,
not just a single demo run passing.

**Gauge R&R — the acceptance discipline (orientation only).** Before a vision station is trusted for
production accept/reject or robot guidance, it undergoes a **Gauge R&R study**: the SAME part (or a
calibrated reference artifact) is measured repeatedly, by the same station and ideally across shifts/
operators/lighting conditions, and the spread of results is decomposed into repeatability (variation
from the gauge itself) and reproducibility (variation across conditions) against the tolerance the
measurement is meant to enforce. This project's own `alignment` gate (recovered vs. applied truth
within a documented pixel/degree bound, see `THEORY.md`) is a toy, single-run version of exactly that
idea — a real Gauge R&R study runs it dozens of times and demands statistical confidence, not one pass.

**Regulatory path (orientation, citing `docs/SYSTEM_DESIGN.md` item 6).** Machine vision itself is not
typically directly regulated, but the ROBOT or MACHINE it guides is: an industrial arm acting on this
project's alignment output falls under **ISO 10218** (industrial robot safety) and, where a human shares
the workspace, **ISO/TS 15066** (collaborative robot force/speed limits) — the same standards
`docs/SYSTEM_DESIGN.md` item 6 maps for every motion-capable project in this repository. In regulated
manufacturing verticals (automotive, medical device, aerospace), the vision station's measurement
accuracy may also need to be traceable under the industry's own quality system (e.g., IATF 16949 in
automotive) — again, orientation, not a compliance checklist.

**Where this work lives inside a robotics/manufacturing company.** Typically owned by a **machine
vision engineering** or **manufacturing/automation engineering** team (titles: Machine Vision Engineer,
Automation Engineer, Controls Engineer), adjacent to **mechanical/fixture design** (who own the
locating-fixture tolerances this pipeline corrects for), **robotics/controls** (who consume the
alignment output), and **quality engineering** (who own the Gauge R&R acceptance process above) — see
`docs/SYSTEM_DESIGN.md` item 5 for the fuller org map this project's "owning team" line in `README.md`
draws from.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
