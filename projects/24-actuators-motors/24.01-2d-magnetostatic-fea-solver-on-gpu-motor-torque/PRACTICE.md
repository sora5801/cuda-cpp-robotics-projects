# 24.01 — 2D magnetostatic FEA solver on GPU → motor torque-ripple/cogging parameter sweeps: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's field solution describes a physical object: a **surface-mount permanent-magnet
brushless motor**. Building the cross-section it models is real, well-understood electromechanical
manufacturing:

- **Rotor:** a stack of thin (~0.35–0.5 mm) electrical-steel LAMINATIONS, punched or laser-cut to the
  rotor-core profile and stacked/bonded (interlocked, welded, or adhesive-bonded) to the target axial
  length — laminating, rather than using a solid steel shaft, breaks up the eddy-current loops a
  spinning field would otherwise induce (THEORY.md's "eddy currents — entirely absent here" limitation
  is exactly the physical effect lamination exists to suppress). The magnets are **surface-mounted**
  onto the rotor core with a high-strength structural adhesive rated for the operating temperature and
  the rotor's centrifugal loading (a magnet flying off a spinning rotor is a real, historically
  documented failure mode), often with a thin RETAINING SLEEVE (carbon fiber or Inconel) wrapped over
  the magnets at higher speeds for extra mechanical security.
- **Stator:** the same lamination stack idea, punched with the tooth/slot profile this project's
  `rasterize_motor()` idealizes as clean angular windows. Copper windings are wound (or inserted as
  pre-formed coils) into the slots, insulated from the iron by a slot liner, then the whole assembly
  is VARNISH-IMPREGNATED (vacuum-pressure-impregnated, VPI) to bond the windings solid, exclude
  moisture, and improve thermal conduction from copper to the iron/frame.
- **Air gap:** the single tightest-toleranced dimension in the whole machine (this project's
  0.001 m default is realistic for a small motor) — set by the rotor and stator bore machining
  tolerances and bearing runout; too tight risks rotor-stator rub under thermal expansion or bearing
  wear, too loose costs torque density and increases cogging sensitivity (a wider gap is LESS sensitive
  to slot geometry, one of the classic torque-vs-smoothness trade-offs this project's sweep touches).
- **What breaks in the field:** demagnetization from over-temperature or an external field spike
  (NdFeB progressively loses remanence above its rated temperature — irreversible past the knee),
  winding insulation breakdown from thermal cycling or vibration (the classic "motor smells burnt,
  shorts a turn" failure), bearing wear widening the air gap over the machine's life, and — if magnet
  bonding was under-specified — magnet delamination at speed.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role relative to this project |
|---|---|---|
| Compute for the field solve | Jetson Orin class (in-the-loop optimization) / x86 + RTX (offline design — reference machine: RTX 2080 SUPER) | Runs THIS solver; not present on the finished robot at all — a design-time tool only |
| Rotor/stator lamination steel | Non-oriented electrical steel, hobby-grade M19/M45-class through research-grade thin-gauge (0.2–0.35 mm) low-loss grades | The `mu_r_iron` material this project models linearly (real grades ship saturating B-H datasheets) |
| Magnets | Hobby: ceramic/ferrite (Br ~0.4 T); research/most robots: sintered NdFeB N35–N52 grade (Br ~1.1–1.4 T, this project's `Br=1.2 T` default); industrial high-temp: SmCo (lower Br, better thermal stability) | The `Br`/`M` source this project's `equivalent_current()` models |
| Magnet wire / windings | Hobby: enamel-coated round copper magnet wire; research/industrial: rectangular or Litz-wire windings for higher fill factor and lower AC loss | Not modeled directly (cogging is a zero-current quantity); feeds the load-torque extension (Exercise, README) |
| Motor-drive silicon (downstream, catalog 24.03) | Control MCU (e.g., an ARM Cortex-M/-R class FOC controller) + gate driver + MOSFET/GaN power stage + current-sense amps | Turns this project's torque/cogging characterization into a current-command strategy (feedforward cogging compensation is a real, deployed technique) |
| Position sensing | Hobby: Hall-effect commutation sensors; research/industrial: magnetic or optical encoder, resolver for harsh environments | Feeds the FOC loop that consumes this project's motor design |

## 3. Installation & integration — putting it on a real robot

**This is a design-time tool, not runtime code — the installation question is "where does the DESIGN
go," not "where does this program run on the robot" (there is no runtime deployment of `main.cu`
itself; the honest N/A below reflects that).**

- **Where the design output goes:** the swept cogging waveform and the chosen magnet arc fraction
  become entries in the motor's DESIGN RECORD — manufacturing drawings (magnet pole-arc dimension),
  and, if cogging-COMPENSATION is used rather than purely mechanical mitigation, a per-angle torque
  correction table loaded into the drive's firmware (catalog 24.03's territory) and indexed by the
  position sensor's reading each control tick.
- **Meeting the drive (24.03) and the encoder:** the motor this project characterizes connects to its
  drive over the phase-winding power leads (three-phase for a typical BLDC/PMSM) and to the position
  sensor over its own interface (Hall sensor digital lines, encoder quadrature/SSI, or resolver
  excitation/sine-cosine) — none of which this project simulates; it produces the FIELD/TORQUE
  characterization those downstream systems are designed against.
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1), for the PHYSICAL motor this project's
  numbers describe (not for this program, which has no hardware-facing rung of its own):**
  1. *Simulation* — this project's field solve and sweep (and, in a production flow, its nonlinear/
     3D/thermal-coupled successors — THEORY.md "Where this sits in the real world").
  2. *Bench characterization* — a built prototype motor on a dynamometer, measuring ACTUAL cogging
     torque (a torque transducer, rotated slowly by hand or a low-speed drive) and comparing against
     this project's predicted waveform SHAPE and relative magnitude — the real validation of every
     simplification THEORY.md documents.
  3. *Drive integration, current-limited* — the motor under FOC control with conservative current
     limits, verifying commutation, direction, and basic torque production before any load.
  4. *Loaded / in-robot* — installed on the actual joint/axis, under the full testing ladder that
     axis's own project documents (e.g., 08.01 PRACTICE §3 for a controller sitting above this motor).
  E-stop and mechanical limits apply at every rung past simulation, per the repo-wide caveat.
- **N/A here:** no ROS 2 node, no real-time constraint, no fieldbus — this project produces a design
  ARTIFACT (numbers and a plot), consumed by engineers and downstream design tools, not by a running
  robot process. Stated honestly per contract, not padded to look like a runtime deployment story it
  is not.

## 4. Business & regulatory context

- **Who needs custom motor-field design:** any robotics company building a proprietary actuator rather
  than buying a catalog motor — most concentrated in legged-robot companies (torque density and
  smoothness are competitive differentiators for dynamic locomotion) and manipulator/cobot makers
  (low-speed smoothness for fine manipulation), per the README "System context" reference-robot
  discussion. Drone and AMR companies more often buy catalog or lightly-customized motors (24.14's
  "actuator selection" territory) rather than commissioning a from-scratch field-solved design.
- **The players:** the commercial motor-FEA tools named in README "Prior art" (FEMM — free; Ansys
  Maxwell, JMAG, Motor-CAD — commercial) are the established build-vs-buy alternative to writing your
  own solver; companies at meaningful actuator-design scale typically license one of these rather than
  maintain an in-house field solver, reserving custom code (much like this project) for teaching,
  rapid early-stage screening, or a specific optimization loop the commercial tool does not expose
  (SYSTEM_DESIGN item 5's build-vs-buy judgment call, applied to this specific domain).
- **What getting it wrong costs:** under-designed cogging shows up as audible noise/vibration
  (a product-quality complaint), degraded low-speed control performance (a functional defect in
  precision applications — manipulation, camera gimbals), and, at the extreme, resonance with a
  structural mode of the machine it is mounted in (a reliability/fatigue issue). None of these are
  safety-certification failures on their own, but they are exactly the kind of "expensive to discover
  late" defect that makes design-time simulation (this project's category) cost-effective versus
  discovering the problem in a built prototype.
  **Sim-validated only, and this matters here specifically: nothing in this project has been checked
  against a built, measured motor** — see [Limitations & honesty](README.md#limitations--honesty) and
  §1's PGM/CSV outputs are teaching artifacts, not a substitute for §3's bench-measurement rung.
- **Regulatory:** electric motors themselves are not typically subject to robot-specific safety
  standards directly, but the ROBOT they actuate is: industrial arms fall under ISO 10218/ISO-TS 15066
  (collaborative force/torque limits — a motor's torque RIPPLE characteristics feed directly into how
  smoothly a compliant/force-controlled arm can meet those limits), service robots under ISO 13482.
  Motor-specific standards that DO apply directly: insulation/thermal class ratings (e.g., IEC 60034
  family) and, if the design is for a vehicle or medical application, the corresponding domain
  standard (SYSTEM_DESIGN item 6's regulatory map). This project computes none of these directly —
  it is an orientation map, not a compliance tool.
- **Owning team:** actuation/hardware engineering (titles: motor design engineer, electric machines
  engineer); adjacent teams: mechanical design (owns the physical lamination stack, magnet bonding,
  bearing/shaft design this field solution's geometry describes — §1), controls/electronics (owns the
  FOC drive and position sensing this project's output feeds — §3), and the systems/actuator-selection
  team (24.14) that decides when a custom design is warranted at all (SYSTEM_DESIGN item 5).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
