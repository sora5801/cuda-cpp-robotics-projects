# 35.01 — Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

There is no single "robot" here in the usual sense — the physical system this project's code would
serve is a **bench-top electromagnetic manipulation platform**: a coil array built around a stationary
sample workspace, not a machine that moves through the world. Its construction:

- **The coils themselves.** Four (or, in a real OctoMag-class system, eight) air-core electromagnet
  coils, each hundreds of turns of enamelled copper (or, for higher current density, water-cooled
  hollow copper tubing) wound on a bobbin, mounted on a rigid frame around a central sample stage. Coil
  geometry is a real engineering trade-off this project's code abstracts away: more turns raises the
  field per amp but raises inductance (slower current response) and resistance (more heat); fewer turns
  needs more current (bigger, more expensive amplifiers) for the same field.
- **Thermal management.** Air-core coils at hundreds of ampere-turns dissipate real power
  (`P = I_wire^2 * R_coil`) as heat — sustained operation at this project's illustrative 500 ampere-turns
  (e.g. 250 turns at 2 A) needs either duty-cycling (pulse the field, do not hold it continuously) or
  active cooling (water jackets, forced air) in a real build; an under-cooled coil drifts in resistance
  as it heats, which drifts the actual field away from the commanded value — a real closed-loop system
  must either measure and compensate for this or keep well within thermal limits.
- **The sample stage and workspace enclosure.** The workspace this project's field map covers (8×8 mm)
  sits at the geometric center of the coil array — in a real build, typically a transparent sample
  chamber (glass or clear polymer) so an overhead or angled camera can image the beads, sized to keep
  the imaging optics' working distance compatible with the coils' physical bulk around it.
- **Wiring, connectors, and what breaks in the field.** Each coil needs a dedicated pair of
  high-current leads back to its driving amplifier; connector fatigue at high current (loose contacts
  heat up disproportionately, a classic failure mode in any high-current system) and coil-to-coil
  crosstalk (mutual inductance between nearby coils, unmodeled by this project's static Biot-Savart
  treatment) are the two most common real-world gremlins in a system like this.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's model |
|---|---|---|
| Coil current amplifiers | Linear or Class-D audio-derived current amplifiers, or purpose-built lab supplies (research: Kepco/TDK-Lambda bipolar supplies, ~US$1-5k/channel; hobby-adjacent: repurposed motor-driver H-bridges, ~US$50-200/channel, far less precise) | Turn a commanded ampere-turn value into real coil current — this project's `I0_ampere_turns` scenario field IS this amplifier's setpoint, divided by the coil's turn count |
| Current sensing | Hall-effect current sensors or shunt + instrumentation amplifier, one per coil channel | Closed-loop current regulation (this project assumes commanded current = actual current instantaneously — real amplifiers have finite bandwidth and need feedback to hold that assumption at the ms timescale a real control loop would run) |
| Compute for field solve + control | Desktop/workstation GPU (reference machine: RTX 2080 SUPER) for offline field-map precomputation (this project's actual workload); a real-time host CPU (or embedded GPU, Jetson-class, for a compact bench instrument) for any closed-loop extension | Where `biot_savart_basis_kernel`'s one-time solve and (in a closed-loop extension) a per-tick current solve would run |
| Imaging (closed-loop only — this project has none) | Machine-vision camera (research: Basler/FLIR industrial cameras, tens-of-Hz frame rates, ~US$500-3k) or, for in-vivo work, fluoroscopy/ultrasound (medical-grade, an entirely different cost and regulatory tier) | Would provide the MEASURED swarm position a closed-loop controller needs — this project's dynamics model stands in for it |
| Sample stage | Machined aluminum or 3D-printed fixture, transparent sample chamber (glass/PMMA) | Physically positions the workspace at the coil array's geometric center |
| Cooling | Passive heatsinking (hobby/short-duty-cycle) up to active water cooling (research/continuous-duty, ~US$100s-1000s for a small closed-loop chiller) | Keeps coil resistance (and therefore the actual field per commanded current) stable |

The scenario's illustrative `500 ampere-turns` could be realized many ways (250 turns @ 2 A, 100 turns
@ 5 A, ...) — the turn-count/current split is a real design trade-off (§1) this project deliberately
does not resolve, quoting only the physically meaningful ampere-turn product.

## 3. Installation & integration — putting it on a real robot

**This project's output (a current schedule) is, honestly, a data structure consumed by a simulation in
this teaching version — not a signal wired to real amplifiers.** The testing-ladder discipline below is
what a REAL implementation of this pipeline would need before that changed, stated at full strength
per CLAUDE.md §1.

- **Where the code would run.** The field-map precomputation (`biot_savart_basis_kernel` et al.) is a
  natural fit for a desktop/workstation GPU running OFFLINE, once per coil-geometry change — exactly
  this project's actual usage. A CLOSED-LOOP extension's per-tick control solve would need to run on
  whatever computer also ingests the camera feed, at the camera's frame rate — this could be the same
  desktop GPU (if colocated with the coil array) or, for a compact bench instrument, an embedded
  GPU (Jetson-class) or even a CPU-only solve (the per-tick linear-combination math this project's
  `combine_field_kernel` performs is cheap enough that a closed-loop version's control-rate bottleneck
  would be the CAMERA and vision pipeline, not this project's field math).
- **ROS 2 shape (if this were integrated into a broader lab-automation stack):** a `MicroswarmController`
  node subscribing a `PointCloud`-shaped topic (measured bead positions, closed-loop only) and a
  goal/waypoint topic, publishing a `Float64MultiArray`-shaped `coil_currents` topic (4 or 8 ampere-turn
  setpoints) that a driver node forwards to the amplifiers — a small, clean interface, matching
  SYSTEM_DESIGN §3.6's message-shaped-struct convention.
- **Buses / physical signal path:** amplifier setpoints are typically analog (0-10V or 4-20mA current-
  loop) or a simple digital bus (CAN, or even USB-serial for a bench instrument) — nothing as demanding
  as the EtherCAT/CAN-FD real-time buses SYSTEM_DESIGN item 6 names for actuator-chain control, because
  this system's control rate (tens of Hz, camera-limited) is far below a motor current loop's 10-20 kHz.
- **Calibration and bring-up.** Before ANY current flows: (1) verify each coil's field independently
  against a calibrated Gaussmeter/Hall probe at several points — this is the real-world analogue of
  `GATE_ONAXIS`/`GATE_HELMHOLTZ`, done with a physical instrument instead of a closed-form check; (2)
  verify current-to-field linearity across the amplifier's full range (catches saturation the linear
  model assumes away); (3) if closed-loop, calibrate the camera-to-workspace coordinate transform.
- **The safe hardware-testing ladder (CLAUDE.md §1), rung by rung:**
  1. *Simulation* — this project's demo, exactly as shipped.
  2. *HIL* — this project's control/planning code driving a simulated (not real) amplifier model with
     realistic bandwidth/saturation limits, to catch anything the idealized "current = field
     instantaneously" assumption would break.
  3. *Bench, current-limited, no sample* — coils energized at a small fraction of design current, field
     verified with a Gaussmeter, thermal behavior monitored, BEFORE any bead or biological sample is
     anywhere near the workspace.
  4. *Bench, full current, inert test sample* — non-biological superparamagnetic beads in a benign fluid
     (not a living sample), verifying the actual gradient-pulling behavior this project's `GATE_ATTRACT`
     predicts, against ACTUAL beads, for the first time.
  5. *Only then*, and only for a project with an actual medical/biological aim, would any biological
     sample or in-vivo work begin — under an entirely separate regulatory and ethical framework (§4).
- **E-stop / limits:** amplifier-level current limiting (hardware, not software) at every coil, and a
  hard cutoff (relay or amplifier-enable line) reachable independent of whatever computer is running the
  control code — the same "does not depend on this software behaving" principle every other project in
  this repository's safety chain follows.

## 4. Business & regulatory context

- **Who needs this capability.** Two distinct markets, at very different maturity: (1) **research
  instrumentation** — magnetic-tweezer and micromanipulation platforms sold to biophysics/bioengineering
  labs for single-molecule and microparticle experiments (a real, if niche, commercial market today); and
  (2) **medical microrobotics** — targeted drug delivery, minimally-invasive microsurgery, and cell/
  particle sorting for lab-on-chip diagnostics, which is **overwhelmingly still preclinical/research-
  stage**, not a shipping product category, as of this writing. Framing this honestly matters: a learner
  should not come away thinking "magnetic microrobot swarms for drug delivery" is an established product
  line — it is an active, well-funded, and NOT YET clinically deployed research direction.
  ★ Market-maturity characterization dated 2026-07-10 — verify current status before relying on it.
- **The players.** Academic groups (ETH Zürich's Multi-Scale Robotics Lab — OctoMag's origin — and
  numerous peer labs worldwide) drive most of the published research; a handful of research-instrument
  companies sell magnetic-tweezer and micromanipulation hardware to labs; no major medical-device company
  currently ships a magnetic-microrobot-swarm THERAPEUTIC product (individual magnetically-STEERED
  catheters/guidewires for cardiovascular procedures are a related, more mature, and DIFFERENT product
  category — single large steerable device, not a particle swarm).
- **What getting it wrong costs.** For research instrumentation: a miscalibrated field wastes
  experiment time and can produce silently-wrong scientific results (a systematic force-calibration
  error propagates into every downstream measurement). For any future medical application: getting the
  field-to-force model wrong is not an inconvenience but a patient-safety failure mode — under- or
  over-delivering a targeted therapeutic dose, or mis-navigating a device near sensitive tissue —
  exactly why this domain's regulatory bar (below) is so much higher than a research bench instrument's.
- **Regulatory path (SYSTEM_DESIGN item 6's map, applied here):** research instrumentation used only on
  benchtop samples generally faces ordinary lab-equipment safety standards (electrical safety, EMC), not
  a medical regulatory pathway. Anything aimed at human use falls under **IEC 60601** (electrical safety/
  EMC for medical electrical equipment) and an **FDA pathway** (510(k)/De Novo/PMA in the US; EU MDR in
  Europe) — SYSTEM_DESIGN's medical-robots row, cited directly. Given the field's current maturity, any
  real product in this space is realistically YEARS of preclinical validation from a regulatory
  submission; this repository's projects in domain 29 (medical/bio robotics) carry the same rule
  restated here: **educational only, no diagnostic or therapeutic claims.**
- **Owning team, in a company pursuing this:** a dedicated **research/advanced-development** group
  (titles: research scientist, robotics research engineer — the SYSTEM_DESIGN item 5 org map's closest
  fit is "ML/data" or a bespoke research function, since this capability does not map cleanly onto any
  of the more mature teams that map does name); adjacent teams would include controls/autonomy (for any
  closed-loop extension), and — the moment any human-use ambition enters the picture — regulatory/
  compliance and quality/functional-safety, engaged far earlier than a typical mechanical/perception
  project would need them (SYSTEM_DESIGN item 5's product-lifecycle framing: this whole capability is
  still firmly in the "concept/prototype" stage, item 5.2).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
