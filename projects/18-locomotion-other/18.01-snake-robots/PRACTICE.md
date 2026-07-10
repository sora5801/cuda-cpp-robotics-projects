# 18.01 — Snake robots: serpenoid gait sweeps coupled to granular sim: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

A real snake robot realizing this project's output (a winning `A, beta, omega, gamma`) is a chain of
identical **actuator modules**, each one link of this project's `N_LINKS=12` model made physical:

- **The module.** The dominant real-world design is the **serial elastic actuator (SEA) module**: a
  brushless motor + harmonic or cycloidal gearbox + an output-side torque/spring element (the
  "elastic" part — it makes the joint compliant and torque-sensable without a separate force sensor),
  packaged inside a housing that is ALSO the structural link (unlike a typical robot arm, there is no
  separate "link" part — the actuator IS the link, end to end). CMU's modsnake and HEBI Robotics'
  snake products both use this pattern; consecutive modules are typically rotated 90° from each other
  so the snake can bend in TWO planes (this project's model is intentionally planar — a single-plane
  simplification, THEORY.md §Where this sits in the real world).
- **Through-bore wiring.** Every module needs power and a communication bus (CLAUDE.md §12 / SYSTEM_DESIGN
  §6.1's CAN-FD/EtherCAT territory) to pass through to every module further from the head — this is
  usually done with a **hollow-shaft ("through-bore") design**: the motor and gearbox are built around
  a central bore that the power/comms cabling threads through, so the cable bundle survives being
  twisted and flexed at every joint for the robot's whole service life without ever being pinched
  between rotating parts. This is one of the hardest mechanical-design problems specific to snake
  robots — a manipulator arm has 6-7 joints to route cable through; a 12-segment snake has 11, each one
  rotating continuously in normal operation, not just occasionally like a wrist joint.
- **Sealing, for the pipe-inspection use case named in README §System context.** A module meant to
  enter a wet or gas-bearing pipe needs IP67/IP68-class rotary seals at every joint (a rotary seal
  around a continuously-flexing, torque-transmitting shaft is a harder sealing problem than a static
  seal) — and because pipe inspection often means confined, unreachable spaces, field-serviceable
  seals (that can be swapped without full disassembly) matter more than in most robotics products.
- **What breaks in the field:** joint seals wear from repeated flex-and-rotate cycling (the #1 failure
  mode reported in the snake-robot literature for field deployments); through-bore cabling fatigues at
  the same flex points; and modules nearer the head see more duty cycles per mission than tail modules
  in a typical "push forward, wiggle to advance" gait — asymmetric wear across a nominally-identical
  module chain is a real maintenance-planning fact, not a defect.

The GPU compute enclosure story (where THIS project's sweep would run, if run onboard rather than on a
desk) is 33.01 PRACTICE.md §1's, unchanged — this project's compute has no bespoke physical form of its
own; it runs wherever the robot's (or the operator's) general-purpose compute lives (§3 below).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's chain |
|---|---|---|
| Gait-search compute | Any laptop/desktop CPU (offline design-time use), or a Jetson Orin-class module for ONBOARD re-optimization | Where THIS project's sweep actually runs — see §3 |
| Joint actuator module | Research: Dynamixel-class smart servos (~US$100-300/joint) for a teaching/prototype snake; Industrial/research-grade: HEBI Robotics X-series SEA modules (~US$2-4k/joint) with integrated torque sensing and CAN/EtherCAT | Executes the reference `phi_j(t)` this project's sweep produces |
| Joint encoder | Absolute magnetic (e.g. AMS/Broadcom AS5x4x-class ICs) per joint | Closes the servo's own position loop — the "perfect tracking" this project assumes |
| Motor-drive silicon | Integrated into the SEA module: control MCU + gate driver + current-sense amp per joint (SYSTEM_DESIGN §6.1's actuation-chain pattern, repeated 11 times) | Converts the commanded `phi_j` into phase current at each joint's own kHz current loop |
| Comms bus between modules | CAN-FD or a proprietary daisy-chain bus running through the through-bore wiring (§1) | Carries the reference trajectory + telemetry between the head's compute and every joint |
| Head sensor suite (for terrain-adaptive re-sweeping) | Miniature camera + IMU; sometimes a contact/tactile sensor on the head (domain 20) | Feeds the terrain estimate that would pick which pre-swept gait table entry to use (§3) |
| Skin / anisotropic-friction surface | Hobby/prototype: passive caster wheels or ribbed silicone sleeve; Hirose's own robots and most research platforms: rows of small **passive wheels** along the belly (mechanically engineered anisotropy, no biological "scale" to replicate) | The physical realization of `mu_t << mu_n` this project's `link_friction_force()` idealizes |
| Sealing (pipe/field use) | IP67-class rotary shaft seals per joint; fully potted head electronics for gas-tight variants | Required for the pipe-inspection/field use case named in README §System context |

## 3. Installation & integration — putting it on a real robot

- **Where this code would actually run.** The sweep itself is NOT an onboard, real-time process — it
  is design-time or, at most, occasional-online-terrain-re-adaptation compute (README §System context:
  the global-planner band, 0.1-1 Hz or event-driven, never the joint servo loop). Concretely, one of
  three deployment shapes:
  1. **Offline, on an engineer's desk** (this project's own demo): run once per target environment
     (e.g. "this particular pipe diameter and material"), the winning `A, beta, omega` gets baked into
     firmware or a config file the robot loads at boot. This is how most fielded snake robots actually
     work today — gait tables are tuned once, not re-optimized live.
  2. **Onboard, occasional re-sweep**, on the robot's own single-board compute (Jetson Orin class or
     similar) when a head-mounted sensor detects a terrain/friction change — a MUCH smaller sweep than
     this project's 8,192-gait grid (the design-time version can afford to be exhaustive; an onboard
     re-sweep triggered mid-mission would use a coarser grid or a local refinement around the last-known-
     good gait to fit inside whatever time budget the mission tolerates).
  3. **Cloud/base-station**, for a tele-operated or semi-autonomous field robot: the head streams a
     terrain estimate back over a radio link, a sweep runs off-robot, and an updated gait table is
     pushed back — viable when latency to a decision (seconds, not milliseconds) is acceptable, which
     it is for a global-planner-band decision (README §System context).
- **The ROS 2 shape.** A gait-sweep node would NOT be a real-time `ros2_control` controller (that role
  belongs to each joint's own servo firmware) — it publishes a `trajectory_msgs/JointTrajectory`-style
  reference (the message-shaped analogue of `phi_j(t)` sampled at the servo's rate) that a downstream,
  per-joint `ros2_control` hardware interface tracks. The bus between the ROS 2 host and the joint
  chain is CAN-FD or EtherCAT (SYSTEM_DESIGN §6.1) running through the through-bore wiring (§1).
- **Calibration/bring-up specific to a snake:** each joint's zero-angle reference must be calibrated so
  the whole chain agrees on what "straight" means (a small per-joint offset error compounds across 11
  joints into a large head-position error — exactly the kind of error this project's forward-kinematics
  math would silently mis-model if fed uncalibrated angles); and the anisotropic-friction surface (§1-2)
  needs its OWN characterization (measuring effective `mu_t`/`mu_n` on the actual target surface) before
  any sweep's results are trustworthy off the flat, uniform, idealized ground this project assumes.
- **The safe hardware-testing ladder (CLAUDE.md §1 applies at full strength — this project's output
  becomes a joint-position reference stream, the archetype of "could command real hardware motion"):**
  1. *Simulation* — this demo, plus a friction-mismatch sensitivity sweep (perturb `mu_t`/`mu_n` ±30%
     and see how much the winning gait's parameters shift — a cheap, honest way to gauge how much a
     real surface's friction uncertainty matters before trusting a swept result).
  2. *HIL* — the winning gait's `phi_j(t)` streamed to the ACTUAL joint controllers with the robot
     suspended (no ground contact) — verifies the joints CAN track the reference at the required rate
     before any friction physics is involved at all.
  3. *Bench, tethered, current-limited* — one or two joints powered at a fraction of rated torque,
     the rest held rigid, verifying the mechanical chain (§1) before the full gait is attempted.
  4. *Free running* — only after the full chain has independently verified sealing (if wet/gas
     environment), E-stop reachability (a snake robot deep in a pipe or rubble pile may not be
     physically reachable for a manual E-stop — a documented, real operational risk, not a detail),
     and joint torque limits set below whatever would damage the chain.
- **N/A here:** no fieldbus, no ROS 2 node, and no onboard hardware are IMPLEMENTED in this project —
  the demo's "actuator" is a function call into the shared physics; this section states, per contract,
  where those pieces would attach.

## 4. Business & regulatory context

- **Who needs this capability.** Pipe and duct inspection (oil & gas midstream/downstream, municipal
  water and sewer utilities, nuclear facility piping — environments too small, too irregular, or too
  hazardous for a wheeled crawler) is the dominant COMMERCIAL market for snake-like robots today.
  Search-and-rescue (voids in collapsed structures) is the dominant RESEARCH/humanitarian-funded market
  — real deployments exist but the business model differs (agency-funded, not a repeat commercial
  buyer). A distant relative — continuum/snake-arm surgical and endoscopic devices — shares the "long,
  thin, many-joint" body plan and some of the same kinematic ideas, but with entirely different
  actuation (often cable-driven, not modular SEA joints), scale, and regulatory path (medical device,
  not industrial robot — see the table below).
- **The players.** CMU's Biorobotics Lab (Howie Choset's group) is the dominant RESEARCH lineage this
  project's math descends from; HEBI Robotics (a CMU spinout) is the closest commercial descendant,
  selling the modular SEA joints §2 names as a product line, not just snake robots built from them.
  Inspection-robotics companies (pipe-crawler and confined-space-inspection vendors) are the
  commercial buyers/integrators; there is no dominant "snake robot as a category" market leader the way
  there is for, say, quadrupeds (Boston Dynamics/Unitree) — this remains a smaller, more fragmented
  niche. Build-vs-buy: the ACTUATOR MODULES (§2) are almost always bought (HEBI-class or equivalent) —
  very few companies re-engineer serial elastic actuators from scratch; the GAIT/CONTROL SOFTWARE this
  project is a teaching version of is where a company's differentiation and in-house engineering
  investment concentrates (SYSTEM_DESIGN §5.3's build-vs-buy criteria, applied).
- **What getting it wrong costs.** A snake robot deep inside a pipe, duct, or rubble pile that fails
  (mechanically jams, or is commanded a gait that gets it physically stuck) is often NOT easily
  retrievable — unlike a wheeled robot that can usually be walked back out, a stuck snake robot can
  mean a lost asset AND, in an inspection context, an incomplete or aborted inspection with its own
  downstream cost (a missed pipe defect, a re-scheduled outage window). In search-and-rescue, a failed
  robot in a void can complicate rather than help the human rescue effort. These are reliability-
  engineering stakes (SYSTEM_DESIGN §5.1's QA & functional-safety team's territory), not safety-of-
  bystander stakes in most inspection deployments (the robot is usually the only thing in the space).
- **Regulatory.** No dedicated standard exists for "snake robots" as a category. The closest orientation
  points (SYSTEM_DESIGN §6.2's regulatory map): **ISO 13482** (service robots sharing space with people)
  when a snake robot operates near workers rather than inside an unoccupied pipe/void; general
  **machinery directives / workplace-safety law** wherever the deployment site is an active industrial
  facility (the pipe's PARENT facility, e.g. a refinery, carries its own safety regime the robot
  operation must respect — confined-space-entry regulations are a real, binding constraint on how
  inspection missions are planned, independent of anything about the robot itself); and, if the "distant
  relative" continuum-surgical application is ever pursued, **IEC 60601**/FDA pathways (SYSTEM_DESIGN
  §6.2) apply in full — a completely different regulatory world from industrial inspection, named here
  only to draw the boundary clearly.
- **Owning team.** Controls/autonomy engineering owns the gait-search software this project teaches a
  version of (titles: robotics software engineer, controls engineer); mechanical engineering owns the
  actuator-module chain and the anisotropic-friction surface (§1-2); for a company selling INTO
  inspection markets specifically, a "field robotics" or "applications engineering" team (SYSTEM_DESIGN
  §5.1) typically owns the mission-planning layer that decides WHEN to re-sweep and WHICH pre-computed
  gait table entry to load — the direct downstream consumer of this project's output.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
