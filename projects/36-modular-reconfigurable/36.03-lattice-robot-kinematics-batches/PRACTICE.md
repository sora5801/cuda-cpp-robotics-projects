# 36.03 — Lattice-robot kinematics batches: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> **Field-maturity caveat that shapes every section below:** modular self-reconfigurable robotics is a
> **research field with no commercial product today**. Where this repo's other PRACTICE.md files
> describe an established supply chain, this one describes a research lineage — stated honestly, not
> padded to look more mature than it is.

## 1. Building it — construction of the robot/part

The physical carrier this project's abstract "slide" and "corner" moves stand in for is **one lattice
module** — the unit this whole catalog domain is built from. THEORY.md's physics section names the
subsystems; this section is about how one would actually BUILD one:

- **Structure:** a rigid unit-cube (or near-cube) shell, typically 3D-printed or CNC-machined in
  research prototypes, sized to comfortably enclose a battery, a microcontroller, one or more
  actuators, and connector hardware on every face that needs one. M-TRAN-lineage designs use a
  two-part hinged body (two half-cubes connected by a motorized rotational joint) rather than a rigid
  cube, because the ROTATION between two module halves is itself one of the module's moves — a detail
  this project's discrete "corner move" abstracts away entirely (the physical rotation happens WITHIN
  a module pair, not as a free-floating cube pivoting in space).
- **Connectors (the hardware "no black boxes" moment for this project's move preconditions):** real
  lattice-robot connectors have taken many forms across the research literature — mechanical
  hermaphroditic latches (identical on every module, so any face can mate with any face), permanent or
  electropermanent magnets (fast engage, but disengage needs an active current pulse or a mechanical
  release), and hook-and-loop or cam-based latches. Every design must solve: (a) **alignment
  tolerance** — connectors must self-correct a few millimeters of positioning error from imperfect
  actuator motion, usually via a funnel or chamfer geometry; (b) **holding force** — a connector deep
  in a large structure may need to support the weight/load of many modules above it; (c) **release
  reliability** — a connector that engages easily but occasionally fails to RELEASE on command can
  strand a whole reconfiguration sequence.
- **Power and data pass-through:** most lattice-module designs route both power and a communication bus
  (commonly a simple serial/CAN-style bus, or even connector-face contacts) across the SAME mechanical
  faces the latches use, so a module far from any external power/radio connection can still be powered
  and commanded through its neighbors — a real distributed-systems problem this project's
  centralized-view batch kernels do not model (see [`THEORY.md`](THEORY.md) "Where this sits in the
  real world" on distributed control).
- **What breaks in the field (from the published research record, generalized honestly):** connector
  latches wear and misalign after repeated cycles; hinge/actuator gearboxes develop backlash that
  accumulates position error across a long reconfiguration sequence (with no external position
  reference, a module's belief about "which lattice cell am I in" can drift from reality — a real
  localization problem this project's ground-truth-position kernels assume away); battery life limits
  how many moves a module can execute before needing to dock and recharge; and at any nontrivial
  module count, SOME fraction of modules will be non-functional at any given time, which is exactly
  the self-repair open problem named in THEORY.md.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them. This is a RESEARCH
field: most of the "hardware" here is one-off or small-batch research fabrication, not an off-the-shelf
supply chain the way most of this repo's other PRACTICE.md BOMs describe.*

| Piece | Illustrative choices (2026) | Role in a real lattice module |
|---|---|---|
| Per-module compute | Small MCU-class board (e.g. an ARM Cortex-M class microcontroller) — a whole module rarely carries a GPU; this project's batch analysis would run OFF-module, on a base-station computer | Local sensing/actuation control, connector state, bus communication |
| Batch/planning compute | A single Jetson-Orin-class or x86+RTX workstation (reference machine here: RTX 2080 SUPER) | Runs THIS project's kernels — the "brain" that validates candidate shapes and, in a real system, would feed a project-36.01-class planner |
| Actuation | Small geared DC or brushless hobby-servo-class motor per hinge/connector (research designs vary widely; industrial-grade actuators are largely unused in this research-stage field) | Executes the hinge rotation or connector engage/release a real move requires |
| Connectors | Custom mechanical latch, permanent/electropermanent magnet array, or hook-and-loop mechanism (research-specific, rarely a standard catalog part) | The physical realization of "modules are face-adjacent" — this project's `adjacent()` test |
| Comms bus | A simple serial/CAN-style bus across connector-face contacts, or short-range radio per module | Distributes commands and position/state gossip between neighbouring modules |
| Power | Small per-module Li-ion/Li-poly cell + simple regulator, OR shared power bus through connector faces from a central supply | Powers the actuator and MCU; module count above a few dozen makes per-module battery life a real constraint |

The bullet's broader domain (36. Modular & Self-Reconfigurable Robots) spans several sibling projects
whose hardware differs (see 36.04, "Connector/latch contact mechanics simulation," for the connector
physics this table only summarizes) — this table describes ONE representative lattice module, not the
whole domain's hardware diversity.

## 3. Installation & integration — putting it on a real robot

**Honest framing first:** there is no "install this on a real robot" path for this specific project in
the way most of this repo's PRACTICE.md files describe, because this project's OUTPUT is a data
structure (legality flags, move counts, a move sequence) consumed by a PLANNER (project 36.01) that
does not exist in this repo — not a command that drives an actuator. The chain from "this project's
output" to "a physical module moves" has a whole missing link (the planner and the per-module motion
controller) that this repo does not implement. What follows is the honest description of where THIS
project's compute WOULD sit if that chain existed:

- **Where this code would run:** on the base-station / planning computer of a lattice-robot research
  rig — NOT on any individual module's onboard MCU (a module's tiny microcontroller has nowhere near
  the memory or compute to batch-analyze thousands of candidate configurations; it would instead
  receive a single, already-validated "your next move is: slide +x" command from the planner this
  project's output would feed).
- **Real-time constraints:** **none, in the control-loop sense.** This is planning-time analysis run
  between physical moves, which themselves take seconds (connector release, actuator sweep, re-latch,
  verification) — there is no kHz/Hz deadline anywhere in this project's pipeline, a genuine contrast
  with the control-layer projects elsewhere in this repo (08.01's 50 Hz MPPI loop, for instance).
- **The ROS 2 shape this would take, if built out:** a planning NODE that subscribes to a target-shape
  goal (analogous to a `geometry_msgs/PoseArray` describing the desired occupied cells) and the current
  configuration (a custom message resembling this project's position array), runs this project's
  validity/connectivity/articulation/move-enumeration pipeline plus a real search (36.01) internally,
  and publishes a MOVE SEQUENCE — one message per commanded module move — to whatever distributed
  control layer (36.05) or per-module command bus a real system used.
- **Bus/hardware it would command:** in a real system, a low-level command per move would need to reach
  the specific module over whatever bus that hardware uses (connector-face serial contacts, a
  lattice-wide CAN-style bus, or short-range radio — see §2) — outside this project's scope entirely.
- **The safe hardware-testing ladder (CLAUDE.md §1), stated for HONESTY even though this project never
  reaches rung 1:** simulation (this project, and the kind of digital twin project 36.02's
  "Stochastic self-assembly simulation" would provide) → single-module bench testing of a connector's
  engage/release cycle → small multi-module (2–4 module) tethered reconfiguration tests with a hard
  physical E-stop and a human present → scaling module count only after connector reliability is
  characterized. This project's contribution stops at rung 0 (simulation/analysis); everything past
  that is out of scope here and, honestly, largely still research-stage across the field.
- **N/A:** no calibration procedure is described here because no physical actuator is driven by this
  project — stated per contract rather than fabricated.

## 4. Business & regulatory context

- **Who needs this capability:** almost exclusively **academic and industrial RESEARCH labs**
  exploring self-reconfiguring robotics — there is no shipped commercial product category for lattice
  self-reconfiguring robots as of this writing (2026-07-10). Adjacent commercial fields that draw on
  overlapping ideas without being "lattice robots" themselves: modular satellite/spacecraft components
  (fixed docking, not general reconfiguration), warehouse robotics (fixed-form AMRs, not
  self-reconfiguring), and reconfigurable furniture/architecture research (Roombots' stated target
  application) — none of these are lattice-robot PRODUCTS today.
- **The players:** primarily university robotics labs (the M-TRAN lineage from Japan's AIST; the
  Roombots lineage from EPFL; and a broader international research community publishing at venues like
  IROS/ICRA/RSS on lattice and chain-type modular robots) and a handful of industrial research groups
  exploring the space pre-competitively. There is no dominant commercial vendor to name, because there
  is no commercial market segment yet.
- **What getting it wrong costs:** in the RESEARCH context this field actually operates in, "getting it
  wrong" mostly costs research time and hardware — a failed connector or a planning bug in a lab
  prototype is an engineering setback, not (today) a safety or liability event with the stakes this
  repo's other domains carry (compare to 08.01 PRACTICE.md's controller-bug framing). If this field
  ever reaches deployment (e.g., disaster-response rubble-clearing structures, or reconfigurable space
  structures), the safety and reliability bar would need to rise to match those domains' existing
  regulatory frameworks (industrial robots: ISO 10218/ISO TS 15066; space hardware: mission-specific
  qualification) — but that is a FUTURE consideration, not a current regulatory reality for lattice
  robots specifically.
- **Regulatory:** **N/A because no lattice-self-reconfiguring-robot product category exists to
  regulate.** The SYSTEM_DESIGN.md item-6 regulatory orientation map (industrial arms, service robots,
  AVs, medical devices, drones, marine, space/defense) does not have a lattice-robot row — the closest
  analogues (industrial robot safety standards, or the space-hardware qualification pathways drone/
  spacecraft components fall under) would only become relevant if and when this research field
  produces a deployable product, which it has not, as of this writing.
- **Owning team, if this field existed inside a robotics company today:** it would sit inside
  **advanced research / R&D** (titles: research scientist, research engineer — robotics), reporting
  distance from product teams that this repo's other domains (controls/autonomy, perception, etc.) do
  not have — the SYSTEM_DESIGN.md item-5 org map's "simulation & tools" and "ML/data" teams are the
  closest existing categories a lattice-robotics research effort would resemble, since the work is
  overwhelmingly algorithms-and-simulation before it is anything else.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
