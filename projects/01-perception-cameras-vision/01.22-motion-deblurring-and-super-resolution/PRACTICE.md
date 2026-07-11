# 01.22 — Motion deblurring and super-resolution for inspection zoom: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is software-only, but it exists to serve a specific physical carrier: an **inspection
camera rig**, either a fixed/pan-tilt camera on a static station, or a camera mounted on an
inspection crawler / AMR that moves past the part being inspected (the reference-robot mapping
README "System context" makes). Two physical configurations matter here:

- **The camera-and-lens assembly itself.** A rigid lens barrel (often a fixed-focus or motorized
  zoom lens) mounted to the image sensor's PCB via a lens mount (C-mount/CS-mount for industrial
  cameras, or a custom mount on a robot's own housing), with the sensor's optical axis mechanically
  aligned to the inspection geometry (e.g. perpendicular to a conveyor, or along a pipe-crawler's
  direction of travel). Vibration isolation (rubber grommets, foam mounts) between the camera and
  its moving carrier matters directly to this project's subject: mechanical vibration transmitted
  into the camera during exposure IS additional motion blur, on top of the robot's own
  commanded/intentional motion.
- **The strobe/trigger wiring**, if the installation uses strobed illumination to freeze motion
  instead of (or in addition to) deblurring (see §2/§3 below) — a strobe LED array wired to a
  camera-synchronized trigger output, with its own heatsinking (strobes run high instantaneous
  current) and light-shaping optics (a diffuser or a ring light around the lens) to give even,
  repeatable illumination frame to frame — important because this project's Wiener/Richardson-Lucy
  filters assume a KNOWN, CONSISTENT blur/noise model; inconsistent illumination breaks that
  assumption in practice.
- **What breaks in the field:** lens focus drift from thermal expansion (a metal lens barrel
  changes focal length measurably over a plant's temperature swing), connector/cable fatigue at
  points of repeated flex (a camera on a moving crawler flexes its own cable every cycle),
  vibration loosening the lens-to-mount fasteners over time (introducing a slowly growing,
  unmodeled blur/defocus this project's fixed-PSF assumption would silently mis-correct for).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute.** This project's actual restoration workload (a 128x128-scale demo; a real inspection
frame is typically 2-12 MP) is FFT- and stencil-bound, not tensor-core-bound — a modest discrete GPU
or a Jetson Orin-class SoC comfortably handles it in near-line/batch mode (README "System context").
Illustrative tiers: **hobby** — a Jetson Orin Nano (~$250, 2024 pricing) running the classical
filters this project teaches, offline per-frame; **research/prototyping** — an x86 host + an RTX
40-series desktop GPU (this repo's own reference class), batch-processing a day's captured frames;
**industrial** — a Jetson Orin NX/AGX or an industrial PC with an embedded RTX-class GPU (e.g.
NVIDIA IGX or an industrial MXM/embedded RTX module), integrated into a machine-vision enclosure
with conformal coating and a fanless or filtered-fan thermal design for a factory-floor environment.

**Camera & sensor.** Industrial GigE Vision or USB3 Vision cameras (e.g. Basler ace/boost series,
FLIR Blackfly S, illustrative examples) with a global-shutter CMOS sensor (global shutter matters
directly here: THEORY.md's exposure-integral derivation assumes one shared exposure window — a
rolling-shutter sensor would need 01.10's per-row treatment layered on top, README "System context"
cites that project by name). Sensor interfaces: GigE (PoE-powered, long cable runs, common on fixed
inspection stations), USB3 (shorter runs, common on robot-mounted cameras), or MIPI CSI-2 direct to
a Jetson-class SoC (shortest run, lowest latency, common in compact embedded rigs).

**Motion-metadata sensing.** The known PSF/shift inputs this project's NON-BLIND restoration needs
(README "Limitations") come from real hardware in production: a rotary/linear ENCODER on the
robot's drive or the inspection stage's motion axis (quadrature encoder, read by an MCU or directly
by the camera's trigger logic), and/or an IMU (accelerometer + gyroscope, e.g. an Bosch BMI-series
or InvenSense ICM-series part, illustrative) for angular-velocity-driven blur estimation — the SAME
class of sensor 01.10's rolling-shutter correction depends on, cited by name in README.

**Strobe illumination hardware** (the freeze-motion alternative to deblurring, discussed in §3): a
high-current LED array (e.g. a machine-vision ring or bar light rated for strobe operation, with a
pulse driver capable of sub-100-microsecond pulses at several amps), triggered by the camera's
exposure-active or a dedicated trigger output.

## 3. Installation & integration — putting it on a real robot

**Where this code runs.** As a NEAR-LINE / offline restoration stage (README "System context"), this
project's GPU pipeline runs on the inspection system's onboard compute (a Jetson-class module for an
embedded rig) or on a plant-floor edge server that ingests captured frames over the network — NOT in
the tight real-time loop a 30-60 Hz perception stack needs. A typical ROS 2 shape: a
`sensor_msgs/Image` topic (raw captures) plus a `Twist`/`JointState`-derived motion estimate (per
`docs/SYSTEM_DESIGN.md` §3.6's message-shaped-struct convention) feed an offline/batch restoration
NODE that publishes a restored `sensor_msgs/Image` for the downstream OCR/defect-classification
node to consume — the SAME upstream/downstream shape README "System context" states, made concrete.

**Motion-metadata plumbing (encoder/IMU -> PSF).** This is the direct integration point with 01.10's
continuity story (cited by name, README "System context"): the SAME per-frame angular-velocity /
linear-velocity estimate that project turns into a per-row rolling-shutter correction is, for a
GLOBAL-shutter camera, turned into this project's single-PSF-per-frame `(length, angle)` pair —
`length = |v| * exposure_time`, `angle = atan2(vy, vx)` (THEORY.md "The problem" derives this).
Getting this metadata pipeline WRONG (a stale encoder reading, a miscalibrated camera-to-IMU
extrinsic) is exactly what this project's `psf_mismatch` gate demonstrates the cost of — a real
system needs the SAME timestamp-synchronization discipline `docs/SYSTEM_DESIGN.md` §3.5 states
(monotonic seconds, `double`) between the motion sensor stream and the camera's exposure-start
timestamp.

**Quality gates before OCR.** In a real pipeline, this project's output would feed a QUALITY GATE
before the expensive downstream OCR/classification step runs — e.g. a lightweight sharpness/contrast
check (this project's own `edge_gradient_mean`/`bar_pattern_correlation` measurements are exactly
the KIND of metric such a gate would use) that flags "this frame's restoration confidence is too low
to trust; re-capture or flag for human review" rather than silently feeding a garbage-in reading to
an automated pass/fail decision.

**The safe hardware-testing ladder.** This project commands NO actuator and moves NO robot — it is a
pure image-restoration filter running on committed/captured frames, so the usual
simulation -> HIL -> bench -> free-running ladder (CLAUDE.md §1) does not apply to THIS code in the
way it would to a controller. What DOES apply: validating the restoration pipeline's OUTPUT before
trusting it for any decision that has real consequences (a pass/fail inspection call, a
robot-navigation decision derived from a restored image) — start on synthetic ground truth (as this
project does), then on a captured-with-known-ground-truth bench target (a calibrated resolution
chart, physically moved a known amount), before ever trusting it against real, unverified field
parts. Every project in this repository whose OUTPUT could feed a downstream motion-commanding
system carries this repo's sim-validated-only caveat (CLAUDE.md §1) — this project's restored images
are exactly such an input if a downstream system used them for navigation or grasp planning.

## 4. Business & regulatory context

Who needs this: machine-vision integrators and manufacturers building automated optical inspection
(AOI) systems, metrology equipment makers, and any robotics company whose product reads
serials/labels/defects with a camera that cannot always stop moving to shoot (throughput economics
directly reward imaging WHILE moving — README "System context" states this). Commercial and
open-source players: **OpenCV** (the de facto open-source baseline for classical restoration,
README "Prior art"), machine-vision software vendors (Cognex, Keyence, illustrative examples — AOI
software suites bundling classical + increasingly learned restoration and defect classification),
and camera/lens/lighting integrators (the hardware side of §2) who sell the strobe-vs-deblur
trade-off as a system design choice to their customers.

**What getting it wrong costs.** A missed defect that motion blur or under-resolution hid from an
automated inspection is a **quality escape** — the same failure mode that drives product recalls
and warranty cost in any manufacturing line; a FALSE reject (blur/aliasing artifacts mistaken for a
real defect) costs scrapped-good-parts and line-throughput. Both failure modes are exactly why this
project's gates are framed as measured PSNR/correlation numbers against KNOWN ground truth, not
vibes — a real deployment needs the equivalent discipline against a certified calibration target.

**Regulatory / standards path.** Machine-vision inspection systems do not carry a dedicated
functional-safety standard the way a collaborative robot arm does (`docs/SYSTEM_DESIGN.md` §6.2's
regulatory map: this domain sits closest to general **quality-management-system** territory — ISO
9001-family process discipline — rather than a hazard-focused standard, UNLESS the inspection
system's output also drives a safety-relevant decision, in which case whatever standard governs
THAT decision (e.g. ISO 10218/ISO 13482 for a robot the inspection result might command to move)
applies to the larger system, not to this filtering stage in isolation). **Evidence-grade imagery
honesty:** if a RESTORED (not raw) image is ever used as evidence of a defect, a measurement, or
compliance with a specification, the restoration itself becomes part of the evidentiary CHAIN — a
genuinely important orientation point, not a fabricated obligation: any organization relying on
restored imagery for a documented inspection decision should be able to say what algorithm ran, with
what parameters, and ideally retain the RAW frame alongside the restored one, the same
chain-of-custody discipline any measurement instrument's calibration record would need. This project
states that honestly as an orientation note; it is **not** legal, quality-system, or compliance
advice, and no output of this repository should be treated as a certified measurement instrument's
result (CLAUDE.md §1, §8).

**Where this work lives inside a robotics company.** The perception/inspection specialization within
the Perception team (`docs/SYSTEM_DESIGN.md` §5.1) typically owns this kind of restoration pipeline,
often alongside or handed off to the ML/data team once a learned restoration model replaces the
classical filters here; adjacent teams include controls/autonomy (for the motion-metadata pipeline,
§3 above) and QA/functional safety (for validating restoration quality against a certified target,
§4 above). Typical role titles: Perception Engineer, Computer Vision Engineer, Machine Vision
Applications Engineer (a title specific to the AOI/inspection industry).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
