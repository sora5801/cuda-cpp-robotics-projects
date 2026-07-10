# 28.01 — Real-time FEM soft-arm model + model-based control (GPU SOFA-style): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> The physical carrier here is the tendon-driven soft arm this project models — this file describes
> how such an arm is actually made, what silicon it runs on, and who pays for it.

## 1. Building it — construction of the robot/part

A tendon-driven soft arm of this project's proportions (a ~24 cm elastomer beam, cable pair,
fixed base) is a thing labs genuinely cast and companies genuinely ship variants of. How it is
physically made:

- **Casting the body.** The arm is molded silicone: a two-part mold (3D-printed or CNC'd), a
  two-component platinum-cure silicone (Shore 00-30 to A-20 class for arms this soft — the ~1 MPa
  modulus range this project's synthetic material imitates), vacuum degassing (bubbles are stress
  concentrators and tear-initiation sites), a cure cycle, demolding. Multi-hardness arms are cast
  in stages — a stiffer spine layer, softer skin — with inter-layer adhesion becoming its own
  failure mode. Fiber or fabric reinforcement (the fiber-reinforced-actuator family) is laid into
  the mold to shape how pressure or tension turns into motion.
- **Tendon routing — where this project's abstraction meets reality.** The model treats a tendon
  as a distributed force along a fiber (kernels.cuh documents the choice). A real tendon is a
  Bowden cable or braided line (Dyneema/Spectra class) running through guides: molded-in channels,
  embedded PTFE tubes, or discrete guide rings bonded along the arm. Every guide adds Coulomb
  friction; the sum makes the tension the motor applies proximally measurably different from what
  the tip region receives — and direction-dependent, which is most of a real arm's **hysteresis**
  (the single biggest thing this model omits). Termination is a crimp or knot in a molded-in
  anchor at the tip; anchors tear out of soft silicone under cycling unless reinforced (a fabric
  patch or stiffer overmolded insert — a classic field failure).
- **The antagonistic pair and pretension.** Two tendons (top/bottom here; three or four spaced
  around the section in 3-D arms) with co-contraction bias, exactly as modeled — and the model's
  buckling budget (THEORY.md: 2×bias ≈ 51% of P_cr = 0.987 N) is a real design constraint:
  over-pretension a slender soft backbone and it buckles or takes a permanent set. Pretension is
  set at assembly (adjustment screws or motor-offset calibration) and drifts as the silicone
  creeps — real arms get re-tensioned.
- **The base.** The cantilever boundary condition is a bonded or clamped root: silicone bonded to
  an aluminum or printed flange (primer + silicone adhesive), or a molded-in flange bolted to the
  mount. The root carries the highest strain and is a dominant tear location; generous fillets at
  the clamp line matter more there than anywhere else.
- **What breaks in the field:** tendon fray at guides and terminations; anchor tear-out; root
  tears; creep and the Mullins effect shifting the tension-to-shape map (the identified Jacobian
  of THEORY.md goes stale — real systems re-identify on a schedule); modulus drift with
  temperature (silicone stiffens noticeably toward 0 °C); abrasion and chemical attack on the
  skin in dirty environments.

The compute enclosure this project's *software* would live in is ordinary embedded-PC/Jetson
practice — the interesting physical construction is all in the arm.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

What a physical instance of this project — the arm, its drives, and the computer running exactly
this code — would consist of, tiered:

| Piece | Illustrative choices (2026) | Role vs this project's code |
|---|---|---|
| Compute for the model | Hobby/research: x86 + NVIDIA RTX (reference machine: RTX 2080 SUPER); embedded: Jetson Orin-class SoC | Runs the two FEM kernels + controller — this exact code. The measured 1.7x real-time factor is the sizing number; an Orin would be slower — re-measure, never assume |
| Tendon actuators | Hobby: 2x smart servos with position/current modes (~US$20–80 each); research: brushless motors + spool, optionally a series-elastic element; industrial: certified servo drives | The plant's `T_top`/`T_bottom`. The code commands *tension*, so a real build needs tension control: current-controlled motors with calibrated spool radius, or closed loop on inline load cells |
| Tension / shape sensing | Inline micro load cells (strain-gauge, ~US$10–50) per tendon; tip: small IMU, magnetic tracker, or external camera + fiducial; research-grade: stretchable strain sensors or fiber Bragg gratings along the body | The demo reads the tip from the model (`download_tip_y`); a physical loop needs a physical tip measurement — the camera route is the usual lab shortcut |
| Motor-control silicon | Hobby servo's integrated MCU; research: motor-driver board (control MCU + gate driver + current-sense amps — SYSTEM_DESIGN item 6's actuation chain) running 10–20 kHz current loops | Turns 333 Hz tension setpoints into winding currents; the kHz loops live HERE, on the drive MCU, never behind the GPU |
| Comms | USB/UART to hobby servos; CAN-FD for research/industrial drives | Carries per-tick tension setpoints (a few bytes at 333 Hz — trivial bandwidth) |
| Power | Bench PSU (hobby); 24/48 V rail + DC/DC (embedded) | Motors at this scale draw single-digit watts; the GPU dwarfs them |
| The arm itself | Cast silicone per §1: ~US$50–200 in materials (mold + silicone + tendons) at hobby/lab scale | The thing `kernels.cuh`'s constants pretend to be — a real build identifies E, ρ, and damping from coupon tests, never the datasheet alone |

Rough system tiers: a hobby bench copy of this demo lands around US$300–800 excluding the PC; a
research rig with proper tension control and motion capture runs US$5–30k; industrial
soft-robotics products (mostly grippers today) are dominated by integration and certification
cost, not BOM.

## 3. Installation & integration — putting it on a real robot

**Sim-validated only (CLAUDE.md §1), and here the caveat has teeth: this project's output is a
pair of tendon tension commands — code of exactly the kind that moves hardware. Nothing below is a
license to actuate; it describes how this class of software is integrated by a team with a safety
process.**

- **Where the code runs.** The FEM model + controller run on the robot's GPU-bearing computer
  (x86+RTX cart PC, or a Jetson on a mobile platform), as a *soft real-time* process. This demo's
  loop (100 physics steps + one sensor read + one control update per 3 ms tick) maps directly onto
  a ROS 2 node: subscribe a tip-pose topic (`geometry_msgs/PoseStamped` from a camera or tracker),
  publish tension setpoints (a small custom `TendonCommand` message), with the FEM stepping inside
  the node between ticks. The hard-real-time work (motor current loops) stays on the drive MCUs
  across CAN-FD/EtherCAT — SYSTEM_DESIGN item 1's division of labor, unchanged by softness.
- **Three integration roles for this exact code:** (a) **digital twin** — run the model alongside
  the arm, compare predicted vs measured tip, alarm on divergence; (b) **model-based gain source**
  — run the IDENTIFY stage against the *hardware* (probe, settle, measure), then run the same PI
  law with hardware-derived gains; (c) **model-in-the-loop preview** — use the
  faster-than-real-time model to sanity-check commands before sending them. The measured real-time
  factor is the admission ticket to all three.
- **Calibration & bring-up:** characterize the material (cast test coupons; measure E and damping
  — datasheets are ±20%); calibrate tendon tension (motor current ↔ measured tension per load
  cell, including friction's direction dependence); calibrate the tip sensor's frame
  (`T_camera_base`, quaternion conventions per CLAUDE.md §12); THEN run the identification probe
  end-to-end. Expect the hardware Jacobian to differ from the FEM's — the gap IS the model-error
  budget, and tracking it over weeks measures the creep rate.
- **The safe hardware-testing ladder (E-stop and limits at every rung):**
  1. *Simulation* — this demo, plus mismatch runs (perturb E ±20%, add a tendon-friction model).
  2. *HIL* — the controller against real drives with tendons disconnected (spool torque only),
     deadline monitoring on.
  3. *Bench, current-limited* — tendons connected; motor current limits cap worst-case tension at
     a fraction of the buckling/tear budget; test the E-stop FIRST (press it; watch it work).
  4. *Free running* — only inside a physical envelope (travel stops, and a tension fuse — a
     deliberate weak link in the tendon path, a genuinely soft-robotic safety device) that does
     not depend on this software behaving.
  A soft arm is forgiving — that is its selling point — but the tensioned cable and the motor are
  not soft: an over-tensioned tendon's failure mode (snap and whip) is an eye hazard, and the
  E-stop must cut the *winch*, not the software.
- **Honest N/A:** this project ships no fieldbus code, no sensor drivers, and no gravity term — a
  real deployment adds all three (gravity is one line in `node_integrate_kernel`; the analytic
  gates would be re-derived around the sagged equilibrium).

## 4. Business & regulatory context

*Didactic orientation, dated 2026 — order-of-magnitude context, not market research.*

- **Who needs this capability.** Soft **grippers** are the commercially-proven wedge of soft
  robotics — food handling, e-commerce picking, delicate-part assembly — where compliance replaces
  expensive force control. Continuum/soft **arms** concentrate in medical robotics (steerable
  catheters and endoscopic platforms are tendon-driven continuum devices in regulatory earnest)
  and confined-space inspection (aeroengine bores, nuclear). Real-time deformable *models*
  specifically are the core of surgical simulation/planning products (SOFA's heritage) and,
  increasingly, of the model-based controllers inside those devices — the slot this project
  teaches.
- **The players.** Commercial: the soft-gripper vendors (the Soft Robotics Inc. lineage; Festo's
  bionic line as the perennial R&D showcase), surgical-robotics platforms with steerable
  instruments (tendon-driven continuum tips), and simulation vendors. Open source: **SOFA** (the
  framework this catalog bullet names, including its soft-robotics plugin), **Elastica**, and
  MuJoCo's deformables, plus the academic toolchains around them. Build-vs-buy: the *simulation
  framework* is usually adopted open-source; the *material characterization and the controller*
  are the in-house crown jewels — exactly the two things this project practices.
- **What getting it wrong costs.** A wrong soft-body model in a surgical device is a
  patient-safety event (misjudged contact force through tissue); in a gripper fleet it is crushed
  product and line downtime; in any tendon-driven product a missed creep/fatigue prediction is a
  warranty recall (tendons and anchors are the wear items). The standing mitigations — digital-twin
  divergence alarms and conservative force budgets — are direct descendants of this project's
  gate-everything habit.
- **Regulatory path (via SYSTEM_DESIGN item 6's orientation map).** Soft grippers on industrial
  cells inherit the industrial-robot framework: ISO 10218, and ISO/TS 15066 for collaborative
  cells — softness *helps* meet TS 15066's pressure/force limits but never waives the analysis.
  Medical continuum devices go the IEC 60601 / FDA route, where simulation used in design falls
  under design-control scrutiny (model-validation evidence; this project's
  analytic-gates-with-measured-margins habit is the kindergarten version of that evidence
  package). Service robots with soft arms fall under ISO 13482. A map of which doors exist — not
  compliance guidance.
- **Where the work lives (SYSTEM_DESIGN item 5).** Owning team: **soft robotics R&D** — in
  practice a simulation/modeling engineer (this project's exact skill set: FEM + GPU + controls)
  embedded with the materials/mechanical group that owns casting and characterization. Adjacent:
  controls (consumes the model), perception (supplies the tip/shape measurements soft robots
  cannot get from encoders), QA/reliability (owns the creep/fatigue rigs that reveal when the
  model went stale). Typical titles: simulation engineer (soft robotics), soft robotics research
  engineer, R&D mechatronics engineer.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
