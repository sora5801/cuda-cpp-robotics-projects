# 29.05 — Ultrasound: GPU beamforming, elastography, image-based servoing: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
>
> **Educational orientation only.** Nothing below is medical, procurement, legal, or compliance
> advice, and no part number, price, or regulatory statement should be relied on without
> independent verification — see the dated notice in §2 and the closing note.

## 1. Building it — construction of the robot/part

This project's code teaches the **beamformer** — the signal-processing chain — but the code's
nearest *physical* carrier is the **transducer array**, the part that actually touches tissue (or,
for a robot-held probe, the part the robot's end-effector holds).

- **The transducer stack, layer by layer:** a matching layer (impedance-transitions the piezo
  element to tissue, typically a quarter-wavelength-thick engineered composite), the piezoelectric
  or capacitive micromachined (PMUT/CMUT) element array itself (64–256+ elements on a sub-
  millimeter pitch — this project's 0.3 mm pitch is representative of a real linear-array
  research probe), a backing layer (damps ringing so pulses are short — directly sets the
  `kPulseCycles` this project assumes), and an acoustic lens (cylindrical, focuses the elevation
  — the *out-of-plane* axis this project's 2-D model ignores entirely). Each element is
  individually wired to a coaxial or flex-cable conductor running the probe's length to the
  connector — 64+ conductors in a cable a few millimeters thick is a genuine mechanical
  engineering problem (flex fatigue at the strain-relief boot is the classic probe failure mode).
- **Construction realities the beamformer inherits:** element-to-element pitch and position
  tolerance directly become the `x_e` formula's accuracy — a probe manufactured with pitch drift
  images with the same *systematic* delay error a bug in this project's `kElementPitchM` constant
  would cause; a dead or intermittent element (a real, common failure — cable fatigue,
  delamination, connector corrosion) silently degrades that element's contribution to every
  pixel's apodized sum, an effect this project's DAS math would tolerate gracefully (the weighted
  average simply has fewer real elements) but that a real system must detect and calibrate around.
- **What breaks in the field:** the acoustic lens delaminates or degrades from repeated
  disinfectant exposure (probes are cleaned between every patient — a real materials-engineering
  constraint this project's synthetic phantom never has to survive); the cable's strain-relief
  boot cracks from repeated coiling/uncoiling; connector pins corrode. None of this project's
  simulated channel data can represent element failure or miscalibration — a real deployment's
  bring-up (§3) always includes a per-element sensitivity/uniformity check this project skips.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Transducer array | Linear array, 64–256 elements, 0.2–0.4 mm pitch, 2–15 MHz (this project: 64 el., 0.3 mm, 5 MHz) — research: Verasonics-compatible probes; clinical: vendor-proprietary (GE/Philips/Siemens/Canon) | The physical source of `channel_data` — this project SIMULATES what this part would record |
| Analog front end (AFE) | Per-channel low-noise amplifier + time-gain-compensation amp + ADC, e.g. TI AFE58xx-class or similar multi-channel ultrasound AFE ASICs; research systems (Verasonics) expose 64–256 such channels | Turns the piezo voltage into the digitized `channel_data[e][s]` this project starts from |
| Beamforming compute | Research: x86 + discrete GPU (Verasonics Vantage systems ship exactly this — the reference machine this project's DAS pipeline targets, RTX 2080 SUPER class or newer); clinical cart: embedded GPU SoC or FPGA/ASIC beamformer for power/cost/real-time reasons; point-of-care handheld: SoC + ASIC front end | Runs `das_kernel` → `envelope_lowpass_kernel` → `log_compress_kernel` |
| Probe connector & cable | Multi-conductor coax/flex bundle, tens to hundreds of conductors, hot-swappable connector (probe-to-system) | Physical path for the AFE's digitized samples to reach compute |
| Robot arm (if probe-held) | Research: collaborative arm (e.g. a 6-7 DoF cobot in the payload/reach class used for teleoperated scanning research); clinical/investigational: purpose-built robotic ultrasound platforms | Holds and positions the probe — the "upstream" partner named in README "System context" |
| Display / recording | Medical-grade monitor (luminance/calibration standards apply clinically), DICOM-compatible storage | Where a real `bmode.pgm`-equivalent frame would be shown/archived — N/A in this synthetic demo |

Tiering: a **hobby/research** setup is a Verasonics-class programmable system (the platform this
project's software architecture imitates) or an open research platform (e.g. university-built
FPGA/GPU front ends); an **industrial/clinical** system is a vendor cart or handheld with a
closed, regulatory-cleared signal chain end to end — none of which this educational project
touches or claims to replicate at production fidelity.

## 3. Installation & integration — putting it on a real robot

**This project's caveat sits at full strength here: everything is sim-validated only, entirely
synthetic, and nothing below is a license to connect this code to a real probe, a real robot, or
a real patient (CLAUDE.md §1).**

- **Where this code would physically run:** on a real image-guided robotic system, the
  beamforming compute (this project's GPU pipeline) runs on the system's imaging-engine compute
  node — typically the same x86+GPU box the ultrasound engine already uses (Verasonics-class
  research platforms), NOT the robot arm's own real-time joint controller. The two communicate
  over a normal (non-hard-real-time) link: the arm's controller publishes pose (`T_base_probe`,
  this repo's transform convention), the imaging engine publishes frames (and, for milestone 3,
  a feature/error signal) — a `ros2_control`-adjacent split, imaging as a sensor node, arm control
  as its own real-time node (SYSTEM_DESIGN.md §6.1's compute-tier layering).
- **OS / real-time constraints, honestly:** the beamforming pipeline itself (this project's four
  kernels, ~1–2 ms measured) is comfortably soft-real-time on Linux + GPU — the same "10–60 Hz
  layers live on Linux + GPU with soft deadlines" placement SYSTEM_DESIGN.md §1.2 gives every
  perception workload in this repo. A real robot-held probe's *motion* loop (if milestone 3 were
  built) would need the same real-time discipline as this repo's other control projects (08.x):
  a monitored deadline, graceful degradation on a missed frame (hold the last good pose command,
  never extrapolate blindly), and the imaging pipeline never in the arm's hard-real-time path.
- **The ROS 2 shape this would take:** an imaging node publishing a B-mode-frame message (a
  `sensor_msgs/Image`-shaped topic, this project's `db` array reinterpreted as pixel data, with
  the probe's calibrated pose as a `TransformStamped`), consumed by an elastography node
  (milestone 2) and/or a visual-servo node (milestone 3) that publishes `geometry_msgs/Twist`-
  shaped velocity commands to the arm's controller — the same message-shape discipline
  SYSTEM_DESIGN.md §3 asks every project to imagine, even where (as here) no real message is sent.
- **Calibration a real deployment needs that this project skips entirely:** per-element
  sensitivity/uniformity calibration (§1's dead-element problem), a speed-of-sound calibration
  against the actual tissue in the beam path (this project fixes c = 1540 m/s; real tissue varies
  and a wrong assumed c distorts every depth this project's `t_tx`/`r_rx` formulas compute), and —
  for a robot-held probe — a hand-eye calibration between the probe's image frame and the arm's
  flange frame (`T_flange_probe`, a rigid calibration this project's synthetic phantom has no
  need for since the phantom IS defined in the image frame already).
- **The safe hardware-testing ladder (CLAUDE.md §1), rung by rung, for milestone 3 specifically**
  (the only milestone whose output could ever move a robot):
  1. *Simulation* — this project (and its milestone-2/3 extensions) against synthetic phantoms
     only, exactly as shipped.
  2. *HIL* — the imaging + servo pipeline against a simulated arm and a simulated (or recorded,
     de-identified, appropriately licensed) ultrasound stream, deadline monitoring on.
  3. *Bench, current-limited, tethered* — the arm on a bench rig with reduced force/velocity
     limits, imaging a **phantom** (a physical, non-patient test object — the real-world analog of
     this project's synthetic scatterers), E-stop verified working *before* any closed-loop
     authority is granted.
  4. *Any contact with tissue* — out of scope for this educational repository entirely; that step
     belongs to a regulated medical-device development process (§4), never to a student project.
- **N/A here:** no fieldbus, no real transducer driver, and no real robot interface are
  implemented — this project's "hardware" is a phantom description file and a function call into a
  synthetic channel-data generator. Stated per contract.

## 4. Business & regulatory context

- **Who needs this capability:** every clinical ultrasound manufacturer (GE HealthCare, Philips,
  Siemens Healthineers, Canon Medical, Samsung/Butterfly-class point-of-care makers) builds and
  owns a beamforming pipeline much like this project's core; research groups and startups building
  robotic/image-guided intervention systems (teleoperated scanning, robot-assisted biopsy/HIFU
  guidance) build exactly the milestone-3 image-based-servoing layer this project documents but
  does not implement, on top of a beamformer much like this one.
- **The players:** Verasonics is the dominant open, programmable research platform (this project's
  architectural model); the major clinical vendors listed above own closed, vertically-integrated
  stacks from probe silicon through display; k-Wave and Field II are the open-source acoustic
  simulation tools researchers validate new beamforming/elastography methods against before ever
  touching hardware. Build-vs-buy (SYSTEM_DESIGN.md §5.3): a company differentiating on image
  quality or a novel imaging mode owns its beamformer end to end (the way this repo's philosophy
  argues for anything that is a genuine differentiator); a company building a *robotic platform*
  around ultrasound sensing very reasonably buys a certified imaging engine and focuses its
  engineering on the robot and the workflow around it.
- **What getting it wrong costs:** in a diagnostic product, a beamforming or calibration bug that
  distorts geometry (e.g., a wrong assumed speed of sound, silently) can mean a mismeasured
  structure — a real patient-safety consequence, which is exactly why clinical imaging software
  sits inside a formal quality system, not a "ship it, iterate later" software culture. In a
  robotic/interventional product, the stakes compound: a servoing bug (milestone 3) could
  physically mis-position a probe or a needle-guide relative to anatomy, i.e. a control bug with a
  patient in the loop — the reason PRACTICE §3's testing ladder gates every rung on demonstrated
  safety before advancing, and the reason nothing in this educational repository is validated
  beyond simulation.
- **Regulatory reality — the central honesty point of this section.** Medical device software in
  general (and diagnostic ultrasound + any robotic/servoing extension specifically) is regulated
  hardware-adjacent software, not ordinary consumer code:
  - **IEC 62304** ("Medical device software — software life cycle processes") is the standard that
    governs *how* medical device software must be developed, documented, verified, and maintained
    — software safety classification (Class A/B/C by potential harm), risk management tied to
    ISO 14971, configuration management, and a documented verification/validation trail. This
    project's own CLAUDE.md-mandated discipline (a CPU oracle at every stage, documented
    tolerances, ground-truth gates, honest limitations) is a *recognizable, if informal, echo* of
    IEC 62304's verification culture — a deliberate teaching choice, not a claim of compliance.
  - **The wider regulatory map** — IEC 60601 (electrical safety/EMC for medical electrical
    equipment) and the FDA's 510(k)/De Novo/PMA pathways in the US (or the EU MDR elsewhere) — is
    the SYSTEM_DESIGN.md §6.2 orientation table's entry for medical robots, which states plainly:
    *"repo §29 projects are educational only, no clinical claims."* This project is exactly that:
    no part of it — not the beamformer, not the ground-truth gates, not the artifacts — has been
    developed under a quality system, and none of it may be represented as validated for any
    diagnostic, therapeutic, or clinical use.
  - A robot-held probe (milestone 3's territory) additionally intersects **ISO 10218 /
    ISO/TS 15066** (industrial/collaborative robot safety, SYSTEM_DESIGN.md §6.2) wherever the arm
    itself is concerned, layered on top of the medical-device pathway above — two regulatory
    regimes overlapping, a real integration headache real companies budget significant regulatory-
    affairs engineering time for.
- **Owning team:** medical imaging / robotics R&D — titles like imaging systems engineer,
  ultrasound algorithms engineer, robotics/controls engineer (for milestone 3); adjacent teams:
  regulatory/quality (owns the IEC 62304 process and has veto power the moment any of this targets
  a real product), clinical/scientific affairs (owns the evidence a diagnostic claim would need),
  and — for a robotic platform — the same functional-safety team every control project in this
  repository answers to (SYSTEM_DESIGN.md §5.1).

---

*Didactic orientation only — **not** procurement, legal, medical, or compliance advice. Where a
topic truly cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded,
never fabricated.*
