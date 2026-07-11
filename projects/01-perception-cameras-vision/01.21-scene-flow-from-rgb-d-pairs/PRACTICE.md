# 01.21 — Scene flow from RGB-D pairs: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

*Section dated 2026-07-11.*

This project is pure algorithm — it consumes an RGB-D stream and produces a moving-object mask; it
has no mechanical parts of its own. The physical carrier the code serves is the RGB-D CAMERA ASSEMBLY
this project's input pair (`data/README.md`) is a stand-in for, and its "construction" is worth
teaching because the SYNC and MOUNTING choices below directly bound this algorithm's accuracy.

An RGB-D module (e.g. a structured-light or active-stereo unit) is typically a single rigid PCB
carrying: two IR-sensitive image sensors (or one IR sensor plus an IR pattern projector) separated
by a fixed baseline (millimeters to a few centimeters — THEORY.md's `z = f*B/d` shows why a LARGER
baseline gives better far-range depth precision at the cost of a larger, heavier module and a larger
near-range "dead zone" where the two views no longer overlap), plus a separate RGB sensor a few
millimeters away. All three sensors are behind their own lens/filter stack, epoxied or screwed to a
common rigid frame (thermal/mechanical stability matters — a baseline that flexes by even tens of
microns under vibration or thermal cycling measurably degrades the disparity-to-depth calibration).
The module mounts to the robot via a rigid bracket (never a compliant/vibration-damped mount for a
depth sensor — any relative motion between the IR emitter/receiver pair during an exposure smears
the pattern and corrupts depth); cabling is a shielded USB3/MIPI-CSI harness, strain-relieved at the
connector (the single most common field failure point for camera modules on a moving robot: repeated
flex fatigues the connector or the ribbon cable's copper traces).

**What breaks in the field:** condensation/fogging on the lens (thermal cycling between a cold
loading dock and a warm building), IR interference from sunlight or another active depth sensor
(structured-light and many ToF units partially blind each other and direct sunlight when their
patterns/emissions overlap spectrally), and connector/cable fatigue from vibration on a mobile base.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

This project's ALGORITHM runs on the robot's main compute; the sensor supplying its input is
separate hardware with its own compute (often a small ISP/depth-processing chip on the camera
module itself, which does the disparity→depth conversion before the metric depth map this project
consumes ever reaches the host).

| Tier | Compute | Depth sensor | Notes |
|------|---------|---------------|-------|
| Hobby/research | Raspberry Pi 4/5 + USB webcam depth module | Intel RealSense D435i (~$300) or an Orbbec Astra | Full RGB-D pipeline at a few hundred dollars; USB3 bandwidth limits resolution/framerate on a Pi. |
| Prosumer/research | Jetson Orin Nano/NX ($200–$700 module) | RealSense D455/D457, or a stereo pair + this repo's own SGM (01.02) | This project's actual GPU kernels (LK, Horn solve) fit comfortably on an Orin-class GPU at real sensor resolution. |
| Industrial/AMR | x86 + discrete RTX-class GPU, or Jetson AGX Orin | Automotive/industrial stereo camera (e.g. an e-con Systems or Basler stereo pair) with a hardened IP-rated enclosure | Industrial units add lens heaters (anti-fog), IP65+ sealing, and locking connectors this project's "what breaks" note above names. |

The algorithm itself needs no exotic silicon: a single CUDA-capable GPU (this repo's floor is
sm_75/Turing) running the kernels in `src/kernels.cu`. At full sensor resolution and frame rate, the
main compute cost this project's THEORY.md flags for future optimization is the per-pixel LK/lift
kernels' memory bandwidth, not any specialized accelerator need.

## 3. Installation & integration — putting it on a real robot

*Section dated 2026-07-11.*

This code would run on the robot's PERCEPTION compute (the Jetson/x86+GPU tier in the table above),
as a node consuming the RGB-D driver's synchronized color+depth topics and publishing a moving-
object mask (or, more usefully downstream, a list of moving-object detections with an estimated
velocity derived from the object-motion fit). In ROS 2 terms: subscribes to
`sensor_msgs/Image` (color) + `sensor_msgs/Image` (depth, or a driver-specific depth encoding) on a
synchronized `message_filters::TimeSynchronizer`-style pairing (CRITICAL — this project's whole
ego-motion/object-motion split assumes both frames of a pair are genuinely simultaneous captures;
timestamp skew between the color and depth streams directly corrupts the 3-D lift, THEORY.md's back-
projection formula has no way to know a depth sample is stale); publishes a mask image plus,
optionally, a `vision_msgs/Detection3DArray`-shaped moving-object list. It requires no realtime OS
guarantee stronger than "keep up with the sensor's frame period" (15–30 Hz — README "System
context") — a soft-realtime Linux perception node is the standard target, not a hard-realtime
control loop.

**Bring-up / calibration:** the RGB-D sensor's own factory (or field) intrinsic/extrinsic
calibration (color-to-depth alignment) must already be correct before this project's pipeline sees
data — a miscalibrated color/depth pair corrupts the RGB texture this project's Lucas-Kanade stage
tracks against the WRONG depth, silently.

**Safe hardware-testing ladder** (CLAUDE.md §1/§8 caveat: everything in this repo is sim-validated
only): (1) simulation — this project's own synthetic pipeline, exactly as shipped; (2)
hardware-in-the-loop — replay a RECORDED real RGB-D bag through the same code with no robot
connected, checking the moving-object mask against a human-labeled ground truth; (3) bench jig —
the camera mounted stationary on a bench, a person or cart moving in front of it, output displayed
but NOT connected to any actuator; (4) tethered/current-limited — mounted on the actual robot,
robot base motion enabled at reduced speed with a physical E-stop within reach, output logged but
still not driving avoidance behavior; (5) free running — only after (1)–(4) demonstrate the mask's
false-positive/false-negative rate is acceptable for the SPECIFIC downstream use (e.g., feeding a
costmap inflation radius) — this project's own measured numbers (README "Expected output": IoU
≈0.20 on a clean synthetic scene) are nowhere near the reliability bar real-world safety-adjacent
use would require, and are presented here strictly as an educational baseline.

## 4. Business & regulatory context

*Section dated 2026-07-11. Didactic orientation only — not procurement, legal, or compliance advice.*

Robots sharing space with people or other moving equipment — warehouse AMRs, hospital delivery
robots, agricultural robots operating alongside workers, collaborative manipulators near a second
machine — all need SOME form of "what around me is moving and where is it going" capability; scene
flow / dynamic-object detection is one classical building block toward it (increasingly supplemented
or replaced by learned detection+tracking in production, per THEORY.md "Where this sits in the real
world"). Commercial players span perception-stack vendors selling this as a component (e.g.,
stereo/depth camera makers bundling SDK-level scene understanding), full-stack AMR companies building
it in-house as a competitive differentiator, and open-source SLAM/perception projects (the
dynamic-SLAM systems THEORY.md names) that ship reference implementations research and smaller
companies build on.

**What getting it wrong costs:** a missed (false-negative) moving obstacle is a collision-risk safety
issue; a hallucinated (false-positive) mover is an operational-efficiency issue (unnecessary stops,
degraded throughput) that erodes trust in the system and, at scale, real cost. Neither failure mode
is acceptable in an uncharacterized state — which is exactly why a real deployment would need the
kind of rigorous, quantified validation (README "Limitations & honesty," this project's own measured
IoU) that this project only gestures toward on a clean synthetic scene.

**Regulatory path** (SYSTEM_DESIGN.md's item-6 regulatory map, cited as orientation only): scene-flow
/ dynamic-object-detection output is PERCEPTION INPUT to a robot's safety functions — it is not
itself a certified safety function, and this repo's rule (CLAUDE.md §8) that "perception input to
safety functions is NOT itself certified" applies directly (the same continuity 21.04's speed-and-
separation-monitor and 01.15's person-perception projects state). Depending on the deployment
category, the applicable framework is ISO 13482 (personal-care/service robots) or ISO 10218 /
ISO/TS 15066 (industrial robots operating near people, e.g. collaborative manipulation) — in every
case, a certified safety function (a physical E-stop, a certified safety-rated lidar scanner with
its own independent hazard-stop logic) sits BETWEEN a perception output like this project's mask and
any actuator, never this project's own uncertified output alone.

**Where this work lives inside a robotics company:** the perception team owns this kind of module,
typically titled Perception Engineer or Computer Vision Engineer; closely adjacent teams are SLAM/
State Estimation (the direct consumer named in README "System context" — dynamic-object rejection
feeds map/pose quality) and Planning/Navigation (the other consumer — a segmented mover's estimated
velocity seeds prediction and local-planner avoidance), with Functional Safety as the team that
would own turning any of this into (or gating it behind) a certified safety function before it
touches a real actuator.
