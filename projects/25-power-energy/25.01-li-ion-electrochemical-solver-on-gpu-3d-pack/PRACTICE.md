# 25.01 — Li-ion electrochemical (P2D/SPMe) solver on GPU + 3D pack thermal simulation + cooling-design sweeps: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it.
> It grounds the README "System context" section in the physical and commercial whole, citing
> [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6. Depth scales with
> relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections 2–4 dated 2026-07-10.*

## 1. Building it — construction of the robot/part

This project simulates a **24-cell pack** shaped like a compact AMR traction/hotel-load battery (4×3×2
cells, each roughly 80×60×120 mm in this project's synthetic scenario — a plausible prismatic-cell
footprint). Real construction of a pack like this, in the order it is typically assembled:

- **Cell selection and grading.** Cylindrical (18650/21700/4680), prismatic, or pouch cells are bought
  in bulk and *graded* — capacity- and internal-resistance-matched into groups, because a pack's weakest
  cell limits the whole pack's usable capacity and, per this project's own electro-thermal coupling
  (THEORY.md), its weakest (hottest) cell ages fastest too.
- **Mechanical structure.** Cells sit in a machined or injection-molded cell holder (a "cage") that sets
  their exact spacing — the gap this project's `THERMAL` medium implicitly represents. Prismatic/pouch
  cells commonly need **compression plates** (they swell slightly with cycling; unconstrained swelling
  cracks internal layers).
- **Electrical interconnection — busbars.** Cells are joined in series/parallel groups by welded or
  bolted **busbars** (nickel, copper, or aluminum strips) — laser or ultrasonic welded to cell terminals
  for cylindrical cells, or bolted/welded to tabs for prismatic/pouch. Busbar cross-section is sized for
  the peak current this project's `MISSION` current profile represents (here, up to 20 A per cell) —
  undersized busbars are a real, common source of localized hot spots and exactly the kind of
  `q_cell` this project's thermal solver would need extending to model at busbar resolution.
- **Thermal interface material (TIM) and the cold plate.** A gap-filling thermal pad or potting compound
  sits between the cells and the **cold plate** this project's `h`/`face` design choice represents —
  real TIMs run ~1–5 W/(m·K), typically the dominant thermal resistance between a cell and its coolant
  (a resistance this project's simplified Robin BC folds directly into the single number `h`, rather
  than modeling TIM thickness/conductivity as a separate layer — an explicit scoping choice, extending
  it is a natural follow-on exercise).
- **Sensing and wiring.** Thermistors (typically one per few cells, not one per cell — this project's
  "one temperature per cell" resolution is already finer than most real packs achieve) and cell-voltage
  taps route to the BMS board (§3).
- **Enclosure.** A sealed (often IP65+) metal or polymer housing provides mechanical protection,
  environmental sealing, and — critically for a real product — the **vent path** for a cell that goes
  into thermal runaway, so gas is directed away from the robot and its operator rather than building
  pressure inside a sealed box. This project does not model venting, gas generation, or runaway
  propagation at all (§1 of `CLAUDE.md`'s safety guardrails: this project computes design-time thermal
  metrics only).

## 2. Real hardware — chips, parts, illustrative BOM

*All parts named below are **illustrative examples, never endorsements**; part numbers and prices go
stale — verify current before relying on any of them.*

| Subsystem | Hobby/research tier | Industrial tier |
|---|---|---|
| Cells | 18650/21700 cylindrical (e.g. Samsung/LG/Molicel commodity cells), ~$3–6/cell | Automotive/industrial prismatic or pouch cells (CATL, EVE, Panasonic industrial lines), custom-qualified per application |
| BMS IC (per-cell monitoring — the runtime consumer of this project's electrochemistry, 25.02) | Analog Devices LTC68xx / Texas Instruments BQ769x2 multi-cell monitor/balancer ICs | Automotive-grade functional-safety BMS SoCs (e.g. Infineon/TI ASIL-rated parts) with redundant monitoring |
| Pack-level compute (running a runtime SOC/SOH estimator, 25.02) | An MCU-class part (STM32/similar) on the BMS board itself | A dedicated battery-management ECU, often with a second, independent safety monitor |
| Cooling — bottom/side cold plate (this project's design variable) | Off-the-shelf aluminum liquid cold plate + a small 12 V pump/radiator loop, or a finned aluminum plate for natural/forced-air convection | Custom-machined or brazed cold plate integrated into the pack structure, sized by CFD; liquid loop shared with drivetrain/electronics cooling |
| Thermal interface material | Off-the-shelf silicone gap pad (~3–5 W/(m·K)) | Higher-conductivity gap pads or dispensed thermal putty, qualified for the pack's vibration/thermal-cycling spec |
| Current sensing | Shunt resistor + INA-class amplifier, or a Hall-effect current sensor | Automotive-qualified Hall/fluxgate current sensors, often redundant |
| Contactors/fusing | Automotive 12–48 V DC contactor + fuse | High-voltage-rated contactors with pre-charge circuit, pyro-fuse for crash/fault disconnect |

## 3. Installation & integration — putting it on a real robot

This project itself is a **design-time desktop tool** — it produces a cooling-design recommendation and
a set of electrochemical/thermal parameters *before* a pack is built; it does not run on the robot at
all (README "System context"). What it hands off to, and where THAT lands:

- **Where the runtime analogue lives.** A real-time SOC/SOH estimator using the same SPM equations
  (25.02, named throughout README/THEORY) would run on the BMS's own MCU or the robot's low-level
  compute, at 1–10 Hz, consuming live cell-voltage/current/temperature sensor readings — a completely
  different real-time/resource budget than this project's offline sweep.
- **ROS 2 shape, if this project's OUTPUT were published on a real robot.** Not this project's own
  output (a design table, not a live topic) but its runtime successor would plausibly publish a
  `sensor_msgs/BatteryState`-shaped topic (voltage, current, per-cell temperatures, SOC) at BMS-cycle
  rate; this project's role is choosing the physical pack design that estimator will later run on.
- **Bus/interface a real pack uses.** Cell-monitor ICs typically report over an isolated SPI/I²C chain
  up to the BMS MCU; the BMS then reports to the rest of the robot over **CAN (often CAN-FD)** —
  `SYSTEM_DESIGN.md`'s comms-bus map — publishing pack SOC/health/fault status the way any other
  robot subsystem would.
- **Calibration/bring-up a real pack needs** that this project does not simulate: cell-to-cell
  capacity/impedance matching at assembly, BMS balancing-threshold configuration, current-sensor
  offset/gain calibration, and thermistor placement verification (confirming the physical thermistor
  actually sits where the thermal model assumes it does — a real, common integration bug).
- **The safe hardware-testing ladder** (CLAUDE.md §1 applies fully — nothing in this repository is
  safety-certified, and pack-level testing is inherently higher-consequence than most other projects
  here): simulation (this project, and its P2D/SPMe extensions) → single-cell bench characterization in
  a controlled/vented environment → small parallel/series group testing with active thermal and voltage
  monitoring and hard cutoffs → full pack testing in a rated battery test chamber with fire suppression
  → integration onto the robot with the BMS's own independent safety limits active throughout, never
  bypassed. **This project's output is an input to that process, never a substitute for any rung of it.**

## 4. Business & regulatory context

Who needs this: any robotics company shipping a mobile, battery-powered product — AMRs, legged
platforms, drones, field robots — needs a pack design process that answers exactly the question this
project's sweep answers ("which cooling design keeps the pack balanced, for our actual duty cycle?"),
because getting it wrong costs real money and real safety margin: undersized cooling shortens pack life
(warranty/replacement cost, fleet downtime) and, at the extreme, raises thermal-runaway risk (recalls,
liability, and — for anything that ships internationally — transport restrictions).

**Main players:** cell manufacturers (CATL, LG Energy Solution, Panasonic, EVE, Samsung SDI, and many
smaller/specialty cell makers); pack integrators and BMS suppliers (both in-house teams at robotics
companies and dedicated BMS vendors); simulation tooling vendors (this project's own "prior art" list —
PyBaMM as the open-source standard, COMSOL/ANSYS/Siemens as commercial multiphysics tools).

**Regulatory/standards orientation** (see `SYSTEM_DESIGN.md` item 6 for the full cross-domain map; this
is orientation, **not** compliance guidance):

- **UN 38.3** — the mandatory transport-safety test series (altitude, thermal cycling, vibration, shock,
  external short, impact, overcharge, forced discharge) every lithium cell/pack that ships by air, sea,
  or road must pass; this is a *transport* requirement, separate from product safety certification.
- **UL 1642 / UL 2054** (cell / battery-pack safety, US) and **IEC 62133** (the international analogue)
  — the product-safety standards a commercial pack is typically certified against before sale.
- **UN 38.3 and these safety standards are TESTED on physical hardware** — nothing this project computes
  (a design-time simulation) constitutes or substitutes for that testing; it exists to help an engineer
  arrive at a design worth testing, faster and with better-informed starting parameters.
- For a **medical or safety-critical** robot's power system, additional standards apply per the robot's
  own domain (`SYSTEM_DESIGN.md` item 6) — out of scope for this project's own analysis.

**Where this work lives inside a robotics company:** power & energy engineering, typically reporting
into hardware/mechanical or electrical engineering, adjacent to (but distinct from) the embedded/
firmware team that owns the BMS runtime software (25.02's home) and the controls/autonomy team that
consumes battery state for mission planning (25.05's home, upstream of this project per README "System
context"). Typical role titles: battery systems engineer, thermal engineer, electrochemical modeling
engineer (a role that specifically does the kind of SPM/SPMe/P2D modeling this project teaches a
reduced version of, often using PyBaMM or COMSOL day to day).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice.*
