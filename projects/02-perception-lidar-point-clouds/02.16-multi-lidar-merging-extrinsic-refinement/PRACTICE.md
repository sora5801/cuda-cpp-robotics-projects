# 02.16 — Multi-LiDAR merging + extrinsic refinement: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5-6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections 2-4 dated 2026-07-12. All parts, prices, and named commercial players below are
> **illustrative examples, never endorsements** — verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

**The rig this project models.** One 360° roof-mounted LiDAR plus two front-corner units, bolted to
a vehicle body (`kernels.cuh`'s rig diagram gives the exact positions/angles this project's synthetic
scene uses). Each mounting point is a small mechanical sub-assembly, not just "a bolt":

- **The bracket.** Typically machined or cast aluminum (stiff, light, thermally matched-ish to the
  sensor housing), bolted to a reinforced point in the body structure — NOT to a thin unsupported
  panel, which would let road-vibration amplitude grow at the bracket's own resonant frequency and
  accelerate exactly the fastener-loosening mechanism THEORY.md's "why mounts drift" section
  describes. Roof units often sit on a raised pod/fairing (aerodynamic and keeps the sensor above
  roof-rack/cargo obstructions); corner units mount lower, often integrated into a bumper or fender
  corner where impact risk (curb strikes, low-speed contact) is highest — the SAME location most
  exposed to the "impact and handling" drift mechanism.
- **Fasteners and locking.** Precision mounts use dowel pins or a kinematic mount (three contact
  points) for REPEATABLE remove-and-reinstall positioning (a sensor swapped for service should not
  need a full recalibration if the mount is kinematic and the pins are true), plus thread-locking
  compound or lock washers on the fasteners themselves to resist the vibration-loosening mechanism.
- **Cabling.** Power, Ethernet (or a proprietary high-bandwidth interconnect for point-cloud data),
  and — for multi-LiDAR rigs specifically — a **PTP (Precision Time Protocol, IEEE 1588) or PPS
  (pulse-per-second) sync line** run to every unit from a common time source, so all sensors' scans
  can be deskewed and merged against ONE shared clock (02.08's deskew, assumed upstream of this
  project — see README "Limitations"). Cable runs need strain relief at the connector (vibration
  fatigues a cable at its fixed end first) and routing away from hot exhaust/brake components.
- **Sealing.** Automotive-exterior sensors need an IP67-or-better rated housing and connector
  (dust/water ingress kills electronics), with the mounting bracket's own seal (a gasket or O-ring at
  the bracket-to-body interface) preventing water intrusion INTO the body cavity, not just around the
  sensor itself.
- **Interference between units (the multi-sensor crosstalk honesty).** Multiple mechanically-spinning
  LiDARs firing pulsed lasers into overlapping space can, in principle, have one unit's receiver pick
  up ANOTHER unit's emitted pulse — a genuine cross-talk failure mode, the LiDAR analogue of 01.20's
  documented iToF-camera interference problem (that project's PRACTICE.md §3 names the same three
  mitigations that apply here: frequency/timing diversity per unit, phase-staggered spin
  synchronization between units so their pulses are never simultaneously incident on the same target,
  and pseudo-random pulse-timing dithering). This project's synthetic rig does not model interference
  (each sensor's returns are generated independently, noise-only) — it is named here as the real
  installation concern a from-scratch multi-LiDAR integration must budget for, alongside the
  calibration problem this project actually teaches.
- **What breaks in the field.** Beyond the slow drift THEORY.md describes: connector corrosion at
  the sensor end (the lowest, most exposed cabling point on a corner unit), bracket fatigue cracks
  at stress-concentration points (sharp corners, weld toes) after enough vibration cycles, and lens/
  window scratching or pitting from road debris on low-mounted corner units — all of which are
  MECHANICAL maintenance items distinct from the CALIBRATION drift this project addresses, though a
  cracked or shifted bracket obviously causes calibration drift too (an extreme, sudden version of
  the slow drift this project's gradual model represents).

## 2. Real hardware — chips, parts, illustrative BOM

The actual hardware a multi-LiDAR rig like this project's would run on and talk to:

| Tier | LiDAR units (x3) | Onboard compute | Sync/comms |
|------|-------------------|------------------|-------------|
| **Hobby/research** | Entry mechanical-spinning units (e.g. RPLiDAR/Livox-class, ~$100-1500 each) | Jetson Orin Nano/NX-class SoC | USB/Ethernet, software-timestamped (no hardware PTP) |
| **Prototyping/research fleet** | Automotive-grade mechanical units (e.g. Velodyne/Ouster/Hesai-class 16-128 channel, ~$2000-15000 each) | Jetson AGX Orin or small x86 + entry discrete GPU | Gigabit Ethernet per sensor + a PTP grandmaster clock module |
| **Production AV/industrial** | Automotive-qualified solid-state or hybrid-solid-state units (multiple vendors, cost varies widely, often under NDA) | Automotive-grade GPU compute module (100s of TOPS) + a safety-rated MCU for watchdogs | Automotive Ethernet (100BASE-T1/1000BASE-T1) or a proprietary high-bandwidth bus, IEEE 1588 PTP hardware timestamping in the NIC/switch |

Supporting silicon this project's pipeline would talk to on real hardware: a PTP-capable Ethernet
switch or NIC (hardware timestamping, not software, is what makes multi-sensor deskew accurate at
the microsecond level real fleets need); a small nonvolatile store per sensor (or a fleet-management
backend record) holding the CURRENT believed extrinsic — exactly the value this project's
`nominal_extrinsic()` stands in for, and what a real drift-watchdog would read and, on a validated
correction, write back to.

## 3. Installation & integration — putting it on a real robot

**Where this runs.** The merge step (README "System context": per-scan, 10-20 Hz) runs onboard, on
whichever compute node already ingests raw LiDAR — typically the SAME perception PC/SoC that runs
ground segmentation, clustering, and the rest of domain 02's pipeline (this project's merged
`PointCloud` output is exactly their shared input). The refinement/drift-watchdog step runs on a
completely different clock and, in most real deployments, a different MACHINE — an offboard fleet
backend or a lower-priority background process on the vehicle, since it needs no hard real-time
guarantee (README "System context": event-driven, not per-scan).

**ROS 2 node/topic shape.** A `lidar_merge_node` subscribes to three `sensor_msgs/msg/PointCloud2`
topics (one per sensor, already deskewed by an upstream `lidar_deskew_node` per sensor — 02.08's
job), holds the current believed extrinsics as node parameters (or reads them from `tf2`'s static
transform tree — `T_base_lidar_i` is exactly a `tf2` static transform in ROS 2's own convention),
and publishes one merged `PointCloud2` on a `~/merged` topic at the sensor's own rate. A separate,
much-lower-rate `lidar_calib_watchdog_node` subscribes to the same three topics (or a decimated,
buffered sample of them), runs this project's detect-then-refine logic on a timer (e.g. hourly or
daily) or on a triggered event (a large IMU jolt suggesting a possible impact), and on a validated
correction, updates the `tf2` static transform (or a parameter server entry) that `lidar_merge_node`
reads — closing the loop exactly as this project's `VALIDATION_LOOP` gate checks it in miniature.

**Calibration/bring-up procedure.** Factory bring-up: mount the sensor, run a controlled calibration
routine (checkerboard/target array or a structured bay — see THEORY.md "Where this sits in the real
world"), record the resulting extrinsic as the FACTORY NOMINAL. In-service: the watchdog above
compares live overlap-zone plane residuals against DOCUMENTED, MEASURED thresholds (this project's
`kDriftDetectionAngleTolDeg`/`kDriftDetectionOffsetTolM` — set from an actual run's aligned-rig
noise floor, `src/main.cu`'s tolerance-block comment) — a real deployment would derive the equivalent
numbers from ITS OWN sensor noise characterization, not copy this project's synthetic-data values.

**Recalibration triggers.** (1) Scheduled — a periodic health check, regardless of any detected
anomaly. (2) Threshold-triggered — the drift-detection residual exceeds its documented bound (this
project's `DRIFT_DETECTION` gate, generalized to a continuously-running check). (3) Event-triggered —
a physical event that plausibly moved a bracket (a logged high-g IMU event, a service record noting a
sensor swap or a body-shop repair near a mount point).

**The safe hardware-testing ladder.** Everything in this repository is sim-validated only —
Simulation (this project, entirely) → Hardware-in-the-loop (replaying REAL recorded multi-LiDAR logs
through this project's detect/refine pipeline offline, comparing against an independently surveyed
ground-truth extrinsic) → Bench jig (a physical multi-LiDAR rig on a bench or a stationary vehicle in
a controlled bay, WITH the vehicle immobilized/current-limited if any actuator is in the loop, which
none is for this project's own scope) → Free-running fleet deployment, with the watchdog's proposed
correction gated behind a human review step until its track record earns automatic application. A
sensor extrinsic feeding perception is squarely safety-relevant (a wrong extrinsic silently degrades
every downstream perception/planning decision) — CLAUDE.md §1's real-hardware caveat applies in full;
nothing here has been run on, or is validated for, physical hardware.

## 4. Business & regulatory context

**Who needs this.** Any fleet operator running more than one LiDAR per vehicle at scale — AV
companies (SYSTEM_DESIGN.md §2.5, the primary reference robot here, given "1-5 LiDARs" and "the
heaviest regulatory burden in robotics"), heavy industrial AMR/AGV fleets with multi-sensor coverage
requirements, and any mobile-robot product line built by a company that has stopped hand-calibrating
each unit at the depot because it no longer scales.

**Commercial and open-source players.** Autoware (open-source, the reference implementation this
project's README §11 names for the merge step) and most AV-stack vendors ship SOME form of
multi-sensor extrinsic management; continuous/online calibration specifically is an active commercial
and research space (fleet-calibration-as-a-service offerings, and a substantial published literature
on plane/edge/feature-based online LiDAR calibration this project's THEORY.md names by category
rather than by vendor, since offerings and their claims change quickly — verify current before
citing any specific one).

**What getting it wrong costs.** A silently mis-calibrated sensor degrades EVERY downstream
perception output that fuses its data — phantom obstacles or missed ones at the overlap-zone
boundary, biased occupancy maps, corrupted training-data labels if the mis-merged cloud is logged for
later ML use. In a safety-relevant stack this is a genuine SAFETY-CASE input, not just a data-quality
nuisance: an AV's UL 4600 safety case (SYSTEM_DESIGN.md §6.2) has to argue that perception inputs are
trustworthy, and a calibration-health monitor (this project's `DRIFT_DETECTION`, generalized) is
exactly the kind of runtime evidence such an argument leans on. Getting it wrong costs downtime (a
fleet grounded pending recalibration), and in the worst case, contributes to a missed-detection
safety incident — the reason this class of monitoring increasingly appears as a named requirement
rather than an internal engineering nicety.

**Regulatory path.** Cite SYSTEM_DESIGN.md §6.2: for the AV reference robot, ISO 26262 (functional
safety of the E/E system this calibration pipeline is part of) and UL 4600 (the safety-case
methodology that would need to document HOW calibration health is monitored and what happens when it
fails) are the relevant rows; for a heavy AMR, ISO 13482's hazard-analysis framing covers the same
question at a lower stakes level. Orientation only — not compliance guidance (CLAUDE.md §1, §8).

**Where this work lives in a company.** Cite SYSTEM_DESIGN.md §5.1: the MERGE path is straightforward
**perception** team territory; the REFINEMENT/drift-watchdog path straddles perception and **fleet
operations** (whoever owns telemetry, remote monitoring, and incident response for a deployed fleet)
— a **perception infrastructure / sensor calibration** role is exactly the kind of position that
exists at this intersection in a mid-size-or-larger robotics/AV company, adjacent to both teams and
to **QA & functional safety** (who consume the calibration-health signal as safety-case evidence).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
