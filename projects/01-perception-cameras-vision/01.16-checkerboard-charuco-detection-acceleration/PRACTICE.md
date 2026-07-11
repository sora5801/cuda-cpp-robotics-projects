# 01.16 — Checkerboard/ChArUco detection acceleration for auto-calibration rigs: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.

## 1. Building it — construction of the robot/part

The "part" here is the calibration target itself and the rig that presents it — not a robot
subsystem in the usual sense, but a piece of manufacturing/test tooling every camera-equipped robot
depends on at some point in its life.

**The target.** A rigid, flat panel — options span a wide quality range: laminated paper on
foam-core (cheap, adequate for hobbyist work, prone to warping with humidity), an inkjet/laser
print bonded to an aluminum-composite panel (common in small-shop and university labs — good
flatness, moderate cost), or a photolithographically etched chrome-on-glass plate (the precision-
metrology-grade option — sub-micron pattern accuracy, used where calibration accuracy itself is the
product being sold, e.g., machine-vision system integrators). This project's synthetic board
follows the glass/aluminum-composite convention implicitly: `THEORY.md`'s physics-first section
already establishes WHY flatness and print accuracy matter (they set a floor on achievable
calibration accuracy that no downstream algorithm can recover from); a real build should pick a
substrate whose flatness tolerance is well under the sub-pixel corner accuracy this project's own
detector demonstrates (~0.7 px, which at typical `fx ≈ 300 px` and 0.3 m range corresponds to
roughly 0.7 mm of in-plane position error — a useful back-of-envelope conversion for choosing a
substrate).

**Matte finish, not glossy.** A glossy print or laminate creates specular highlights under
directional lighting that can locally saturate the sensor or shift the effective edge location —
real calibration targets are almost always printed or coated matte for exactly this reason. This
project's synthetic renderer sidesteps the issue by construction (a pure Lambertian reflectance
model, no specular term), a documented simplification.

**Thermal stability.** Aluminum-composite and glass substrates are chosen partly for low thermal
expansion — a paper target can visibly cockle under a warm studio light over the minutes a
multi-pose capture sequence takes, introducing exactly the kind of small, systematic geometric
error this project's math has no way to detect or correct.

**Target-size-vs-field-of-view design rule.** A calibration target should fill a substantial
fraction of the frame (this project's synthetic views target roughly 60–85% of the frame width) —
too small and the corner-position noise floor dominates the calibration signal; too large and
perspective distortion at the target's own edges gets severe enough to strain the corner detector
(exactly the tilt-sensitivity `THEORY.md` documents). A rule of thumb from machine-vision practice:
size the target so it fills 50–80% of the frame at the CLOSEST planned working distance, leaving
margin for the farthest.

**Auto-calibration rig construction — two common architectures:**

- **Robot-arm-presented target.** The target is mounted on a fixture; a robot arm (or a simple
  2-axis gimbal) moves it through a programmed sequence of poses in front of a STATIONARY camera —
  this is the architecture this project's own "rig batch" of 8 views most directly models (each
  view = one programmed pose). Repeatable, scriptable, and — because the arm's own forward
  kinematics gives an independent estimate of relative pose between views — a natural way to
  cross-check the vision-based calibration against a second, independent measurement.
- **Static multi-board rig.** Several calibration targets are mounted at fixed, precisely-surveyed
  positions around a work cell (or around a vehicle on a rotating turntable); the CAMERA moves
  (or the vehicle) rather than the target. Common in automotive EOL (end-of-line) stations, where
  many identical units need calibrating quickly and a moving-target robot arm would be slower than
  driving the vehicle itself through a fixed sequence of stations.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Compute | Camera | Target/rig actuation | Rough cost (rig, excl. camera) |
|------|---------|--------|----------------------|-------------------------------|
| Hobby/lab | A laptop or desktop with any CUDA-capable GPU (this project's own reference machine, an RTX 2080 SUPER, is well beyond what this workload needs) | A USB webcam or a Raspberry Pi camera module | Hand-held target, hand-waved poses | Under $50 (printed target only) |
| Small-shop/university | A small form-factor PC or a Jetson Orin Nano dev kit | An industrial GigE/USB3 machine-vision camera (e.g. a Basler ace or FLIR Blackfly class part — 1–5 MP global shutter, since a rolling shutter smears a moving target) | A 6-axis desktop robot arm (e.g. a UR3e class collaborative arm) presenting an aluminum-composite target | Low thousands to tens of thousands of USD (dominated by the arm) |
| Industrial EOL station | An industrial PC with a discrete workstation GPU, often shared across several camera stations on a line | Multiple synchronized industrial cameras (the exact units the production vehicle/robot ships with, calibrated in-situ) | A fixed, precisely surveyed multi-target jig or a servo-driven turntable; PLC-sequenced | Tens of thousands to low hundreds of thousands USD (a full line station, amortized over volume) |

**Chips this project's computation would actually run on in each tier:** any tier's "compute"
column above is a general-purpose CPU/GPU running exactly the kind of code in `src/` — no
specialized silicon is needed for calibration itself (unlike, say, a real-time control loop). The
GPU acceleration this project teaches matters more as camera COUNT and RESOLUTION scale up
(an EOL station calibrating an 8-camera AV sensor suite at once, each at several megapixels, is
where batch-parallel corner detection genuinely saves wall-clock time on a busy line) than for a
single hobbyist camera, where even a CPU-only version would finish in well under a second.

## 3. Installation & integration — putting it on a real robot

**Where this code would run.** Calibration is normally run OFF the robot's own real-time compute —
on a laptop, a service-bay PC, or (for factory EOL) a dedicated station PC — precisely because it
has no real-time constraint (README "System context"). The recovered intrinsics are then written
to a **calibration file** that IS deployed onto the robot's own compute (its perception PC or the
camera's own onboard flash, for cameras that store their own calibration).

**OS and real-time constraints.** None specific to this computation — any general-purpose OS
(Linux is standard in robotics tooling) suffices; no real-time kernel, no deterministic scheduling
needed.

**ROS 2 shape.** The natural output of this pipeline is a `sensor_msgs/CameraInfo` message (ROS 2's
standard camera-calibration message type: `k` for the 3x3 intrinsic matrix this project recovers,
`d` for the distortion coefficients this project's teaching scope skips, `width`/`height`,
`distortion_model`). In practice this is usually persisted as a `.yaml` calibration file (the
`camera_calibration_parsers` package's format) loaded once at each camera driver node's startup,
not published live — calibration is a STARTUP-time configuration input to the perception stack,
not a topic in the steady-state data flow (`docs/SYSTEM_DESIGN.md` interface conventions).

**Calibration-file management and versioning.** A fielded robot fleet needs the SAME discipline
around calibration files that 01.14 names for template management: each file should be versioned
(which rig, which firmware, which calibration date), checksummed, and traceable to the specific
unit it was measured on — swapping a camera or a mount without re-running calibration (or without
updating which file is loaded) is a common, quietly-serious field failure mode.

**Recalibration triggers.** A fielded system should recalibrate (a) on a fixed maintenance
interval (weeks to months, application-dependent), (b) after any mechanical event that could shift
a camera mount (a hard bump above a logged IMU-acceleration threshold, a mount screw torque check
failing, a collision-detection event), and (c) whenever a downstream consumer (e.g., a stereo depth
or SLAM system) reports elevated reprojection error consistent with drift — an automated,
statistically-triggered recalibration flag being the more mature version of a fixed interval.

**Safe hardware-testing ladder.** This project's output never commands motion of real hardware —
it is a pure measurement computed from already-captured images — so the usual simulation → HIL →
bench → free-running ladder does not directly apply to the CALIBRATION COMPUTATION itself. It DOES
apply to the ROBOT ARM (or turntable) that PRESENTS the target, if that presentation is automated:
that motion sequence should be validated in simulation first, then run tethered/current-limited on
the bench, with the arm's own standard E-stop and joint-limit safeguards active throughout, exactly
as any other robot-arm program (CLAUDE.md §1's caveat is genuinely operative for the PRESENTER arm,
even though it is not operative for the calibration math itself).

## 4. Business & regulatory context

*Section dated 2026-07-11. Didactic orientation only — not procurement, legal, or compliance advice.*

**Who needs this.** Any company shipping a camera-equipped robot or vehicle at volume: automotive
OEMs and Tier 1s (every ADAS/autonomy camera needs individual intrinsic + extrinsic calibration at
manufacture, and often periodic recalibration in service), warehouse/logistics robot makers (AMR
fleets with vision-based localization or pick systems), industrial machine-vision integrators
(gauging and inspection systems whose accuracy claims are only as good as their calibration), and
robotics research labs (calibration is table-stakes infrastructure for nearly every vision-based
project in this repo's catalog).

**Commercial and open-source players.** OpenCV's calibration module is the dominant open-source
implementation (used directly or as a dependency by most of the tools below); Kalibr (ETH Zurich,
open source) is the standard for multi-camera/camera-IMU rigs in research robotics; MATLAB's
Camera Calibrator app is common in industrial/academic settings that already license MATLAB;
several machine-vision vendors (Cognex, Keyence, Zivid, and others) ship calibration as a bundled
feature of their own camera/software stacks rather than a standalone product.

**What getting it wrong costs.** A systematically mis-calibrated camera degrades EVERY downstream
perception estimate that consumes it — a stereo depth error, a SLAM drift, a grasp-pose offset — in
ways that are often subtle enough to pass casual testing and only show up as a slow accumulation of
failures in the field. In automotive/ADAS contexts this is a genuine functional-safety concern
(mis-calibration is a plausible contributor to a missed-detection or false-ranging failure mode);
in manufacturing/logistics contexts it shows up as elevated pick-failure or collision rates and the
associated downtime and liability exposure. Traceable, versioned calibration records (section 3
above) exist specifically to make this failure mode diagnosable after the fact, not just to prevent it.

**Regulatory/standards path (orientation, cite `docs/SYSTEM_DESIGN.md` item 6).** Calibration
itself is rarely regulated directly, but it is a load-bearing input to systems that ARE: automotive
perception feeds into ISO 26262 (functional safety) and, for autonomous driving specifically,
UL 4600 process arguments — both of which expect a documented, repeatable, traceable calibration
and verification process as part of the broader safety case, not a one-off engineering step.
Industrial vision-guided robotics falls under the same general machinery/robot-safety umbrella as
the arm or cell it is integrated into (ISO 10218 / ISO 13482 territory, per `docs/SYSTEM_DESIGN.md`
item 6) — calibration accuracy feeds into whatever collision-avoidance or speed-and-separation
monitoring (21.04, this repo) the cell relies on.

**Where this work lives inside a robotics company.** Calibration and manufacturing test
engineering — a distinct discipline from perception R&D, typically reporting into hardware
engineering, quality, or manufacturing operations rather than the autonomy/perception software
team, though it works closely with perception engineers who define accuracy requirements and
consume the resulting calibration files (`docs/SYSTEM_DESIGN.md` item 5). Adjacent roles: test
engineering (designs and maintains the EOL station itself), quality engineering (owns the
traceability and statistical-process-control side), and firmware/embedded engineering (owns how
calibration files are stored on and loaded by the shipping unit).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
