# 31.09 — Physics-of-failure prognostics Monte Carlo: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

The mechanical and electrical construction of the subsystem this project belongs to: how it is
physically built, assembled, and manufactured; materials and tolerances; mounting, wiring, connectors,
sealing, shielding; what breaks in the field and why. For abstract/software-only projects, describe
the physical carrier the code would serve and its construction instead.

TODO(scaffold): write the construction section (or "N/A because …" with the honest reason).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-08. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

The actual hardware this would run on and talk to: compute tier (Jetson Orin class vs. x86 + discrete
RTX vs. MCU-class), the sensors and their interfaces, the actuation chain's silicon (motor-control
MCUs, gate drivers, current-sense amps, encoder ICs), comms transceivers, power parts (BMS, DC/DC).
Offer hobby / research / industrial-grade alternatives with rough cost tiers.

TODO(scaffold): write the hardware/BOM section (or "N/A because …").

## 3. Installation & integration — putting it on a real robot

Where this code would physically run (which computer on which robot); OS and real-time constraints;
the ROS 2 node/topic shape it would take; which bus it consumes or commands (CAN-FD, EtherCAT,
Ethernet); sensor/actuator calibration and bring-up procedure; and the safe hardware-testing ladder —
simulation → HIL → bench jig / tethered / current-limited → free running — with E-stop and limits at
every rung. Everything in this repo is sim-validated only and not safety-certified (CLAUDE.md §1).

TODO(scaffold): write the installation & integration section (or "N/A because …").

## 4. Business & regulatory context

Who needs this capability, in which products and markets; the main commercial and open-source players;
what getting it wrong costs (downtime, recalls, liability); the applicable standards / regulatory path
for this domain (cite the SYSTEM_DESIGN.md item-6 regulatory map); and where the work lives inside a
robotics company — owning team, typical role titles, adjacent teams (SYSTEM_DESIGN.md item 5).

TODO(scaffold): write the business & regulatory section (or "N/A because …").

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
