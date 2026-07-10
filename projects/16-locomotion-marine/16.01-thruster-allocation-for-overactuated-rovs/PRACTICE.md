# 16.01 — Thruster allocation for overactuated ROVs (batched QP): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5-6.
> This project's physical carrier is the **thruster and its mount** — this file's sections 1-2 teach
> that hardware, because it is where a control engineer's allocation output meets water and current.

## 1. Building it — construction of the robot/part

The subsystem this project's output drives is the **thruster and its through-hull mount** — one per
allocation-matrix column. What that looks like, physically, for a small/medium ROV like this project's
synthetic vehicle:

- **The thruster itself.** A brushless motor in a flooded or oil-filled housing (flooding the motor
  cavity with a dielectric fluid equalizes pressure with the surrounding water, avoiding a
  pressure-rated housing at the motor — the common small-ROV-thruster design, e.g. the Blue Robotics
  T-series family) drives a propeller through a shaft seal. The propeller sits inside a **duct/nozzle**
  (a Kort-nozzle-style ring) that both protects it and increases thrust per watt at low speed — the
  duct is why bollard thrust (THEORY.md) is meaningfully asymmetric forward/reverse: the duct's internal
  shape is optimized for one flow direction.
- **Mounting.** Each thruster bolts to a rigid bracket on the vehicle frame, oriented to the exact
  `r_i`/`d_i` this project's `B` matrix assumes (THEORY.md "the math"). Getting that orientation right
  *in the field*, not just on a CAD drawing, is a real bring-up step (§3, calibration) — a bracket
  installed a few degrees off from spec quietly makes the allocation matrix wrong, and nothing in this
  project's software would catch that (README "Limitations": `B` is assumed exact).
- **Wiring and sealing.** Motor phase wires (3, for a BLDC/PMSM thruster) exit the housing through a
  **potted cable gland or a wet-mateable underwater connector** — the single most common ROV field
  failure point, because every mate/unmate cycle and every scrape against a structure risks the seal.
  Cabling routes back through the vehicle frame (often inside a flooded compartment, to avoid another
  set of hull penetrations) to a **thruster driver/ESC** in the vehicle's pressure housing.
- **What breaks in the field:** fouled propellers (fishing line, kelp — jams the prop, draws excess
  current, sometimes trips a protection circuit that looks identical in software to "thruster failed,"
  which is exactly the fault this project's failure-analysis stage models the *consequence* of, not the
  *cause* of); connector corrosion/flooding (a slow ingress that degrades before it fully fails);
  bearing/seal wear from sand or grit ingestion; and, less obviously, **bent brackets** from a hard dock
  or a snag, which change `r_i`/`d_i` without killing the thruster outright — a subtler, uncaught failure
  mode than the "locked to 0" model this project's software uses.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Compute running the allocator | Jetson Orin-class SoC (small ROV) or x86 + embedded GPU (larger work-class ROV with a topside/subsea compute skid) — reference dev machine here: RTX 2080 SUPER | Runs the batched-QP kernel every control tick; at `K=1` (one wrench per tick, the real-time case) the kernel's GPU time is negligible next to launch overhead — the GPU's real payoff here is the *batch* use cases (README), so a fielded single-tick allocator often runs the identical algorithm on a CPU/MCU instead |
| Thruster driver / ESC | Brushless ESC: control MCU + gate driver + MOSFET half-bridges + current sense, e.g. Blue Robotics Basic ESC-class (hobby/research tier, ~US$20-50/thruster) up to industrial subsea-rated thruster controllers (~US$500-2000+/thruster) | Converts this project's force setpoint (after the force->RPM inversion, THEORY.md) into commanded phase currents |
| Thruster | Ducted BLDC thruster, hobby/research tier (~US$150-300, e.g. T200-class, bollard thrust in the tens of Newtons — this project's `u_max=40N` is sized in that range) up to industrial work-class thrusters (hundreds of Newtons, US$1000s) | The physical actuator `u_i` commands |
| Position/heading sensing (feeds the upstream controller, not this project directly) | Depth sensor (pressure transducer), AHRS/IMU, USBL or DVL for position — see domain-16 siblings (16.08 USBL processing) | Produces the state estimate the upstream DP/docking controller (16.09) uses to compute `tau_cmd` in the first place |
| Vehicle bus | CAN-FD or a vendor serial protocol (e.g. Blue Robotics' thruster PWM-over-serial for hobby tier; industrial vehicles increasingly run CAN-FD or EtherCAT to the thruster drivers) | Carries this project's per-thruster setpoints from the allocator to each ESC |
| Power | Battery pack -> BMS -> DC/DC to the thruster bus voltage (commonly 14.8-16V for small ROVs, higher for work-class) | Thrust droops with bus voltage under load — a real-world detail this project's fixed `u_max` does not model (README "Limitations") |

## 3. Installation & integration — putting it on a real robot

**This project's output is a set of thruster force commands — the §1 caveat applies at full strength:
everything here is sim-validated only; nothing below is a license to actuate a real thruster.**

- **Where this runs.** On a ROS 2-based ROV stack, allocation is typically a small, fast node (or a
  library called from inside the controller node) subscribing the commanded wrench (`geometry_msgs/
  Wrench` or an equivalent DP-controller output topic) and publishing per-thruster setpoints — commonly
  `std_msgs/Float32MultiArray` or a vehicle-specific thruster-command message, matching
  SYSTEM_DESIGN.md's message-shaped-struct convention. In this project's own demo the "topic" is a
  function call (`launch_thruster_allocation`) — the mapping to a real node is direct.
- **Real-time constraints, honestly.** A single allocation solve (`kPgdIters=500` fixed iterations,
  `O(8^2)` per iteration) is microseconds of CPU work even without a GPU — the batching story
  (README/THEORY) is about *many* allocations at once, not making one allocation fast enough for a
  real-time tick (it already is). A fielded single-tick allocator is a strong candidate to run on the
  same real-time host CPU as the rest of the control loop rather than round-tripping to a GPU at all;
  this project's GPU kernel earns its place specifically for offline/batch use (planning a whole
  trajectory, or the fault-tolerance sweep) — say this plainly rather than overselling the GPU angle
  for the single-tick case (SYSTEM_DESIGN §1.2's "where the GPU classically sits" discussion applies).
- **Calibration and bring-up.** The allocation matrix `B` is only as correct as the thruster geometry
  fed into it (THEORY.md). Real bring-up: (1) verify each thruster's *sign* — command a small positive
  force to each thruster individually, in air or in a test tank, and confirm the vehicle (or a force
  gauge, for a bench jig) moves/reads in the expected direction, catching a miswired ESC phase or a
  flipped mounting orientation before it ever reaches software; (2) verify `u_max` against the
  thruster's actual measured bollard thrust at the vehicle's real operating voltage, not the
  datasheet's headline number; (3) only then trust `B` and re-derive it (this project's
  `build_allocation_matrix`) from the *as-installed* geometry, not the CAD nominal.
- **The safe testing ladder (CLAUDE.md §1):**
  1. *Simulation* — this project's demo, plus deliberately mis-specified `B`/`u_max` runs (README
     Exercise 5) to see how allocation degrades under a wrong model, exactly the failure mode §1 above
     describes for a bent bracket.
  2. *Bench, thrusters in air, tethered and current-limited* — command small, known wrenches and verify
     each thruster spins the expected direction at the expected relative speed, per the sign-check above.
  3. *Tank/pool, restrained* — the vehicle physically tied off, verify closed-loop station-keeping
     behavior against a real (if small) current before trusting free motion.
  4. *Free running, in a controlled body of water, with a tended tether and a topside E-stop* — only
     after 1-3 pass, and only within an operational envelope (a surface-support boat, a recovery plan)
     that does not depend on the allocator behaving perfectly.
- **N/A here:** no fieldbus, ESC driver, or ROS 2 node is implemented in this project — the demo's
  "actuator" is a CSV artifact recording what force *would* be commanded. Stated per contract.

## 4. Business & regulatory context

- **Who needs this capability.** Any company building or operating ROVs/AUVs with more thrusters than
  DOF — which is nearly all of them, from hobby-tier inspection ROVs (Blue Robotics-class) through
  survey/inspection work-class vehicles (oil & gas platform inspection, offshore wind cable/foundation
  survey, aquaculture net inspection) to research AUVs. Anywhere a vehicle must hold station precisely
  against current while a payload (camera, manipulator, sensor package) does the actual paid work,
  thrust allocation quality is directly a **quality-of-service** feature: better allocation means
  steadier station-keeping means better sensor/manipulator data, and (README/THEORY) means the vehicle
  degrades gracefully rather than catastrophically when a thruster fails mid-mission.
- **The players.** Vehicle makers building their own vertically-integrated stacks (their allocation
  logic is proprietary, close to their real product IP); the open-source ArduSub/BlueOS ecosystem
  (README "Prior art") giving smaller integrators a working reference; classical DP-system vendors
  (the Kongsberg-class systems THEORY.md's "Prior art" section references) serving the larger
  work-class/drilling-support end of the market, where thrust allocation is one module inside a much
  larger, classed and certified DP control system.
- **What getting it wrong costs.** A poorly-tuned or buggy allocator does not usually look like a
  dramatic failure — it looks like degraded station-keeping accuracy, wasted battery/fuel (fighting
  current less efficiently than the vehicle's redundancy allows), or — the scenario this project's
  motivating worked example (THEORY.md) makes concrete — a wrong-direction response to a saturating
  command, which in a tight subsea environment (near a structure, near a manipulator's own work area)
  is a real collision/entanglement risk, not just an efficiency loss. For classed work-class DP vessels,
  thrust allocation failure modes are explicitly part of the safety case (below).
- **Regulatory.** Small/hobby ROVs largely operate outside a dedicated robot-safety regulatory regime;
  larger work-class ROV/DP operations fall under **class-society rules** (e.g., DNV's DP-class notation
  system: DP1/DP2/DP3 describe redundancy requirements, directly relevant to *why* overactuation and
  graceful thruster-failure degradation — this project's whole subject — matter enough to be a
  certification criterion, not just an engineering nicety) and, everywhere an autonomous or remotely
  operated vessel shares water with crewed traffic, **COLREGs** (the international collision-avoidance
  rules) govern right-of-way behavior — see SYSTEM_DESIGN.md §6.2's regulatory map, "Marine/AUV-ASV"
  row. This is an orientation pointer, not compliance guidance (SYSTEM_DESIGN §6's label applies in
  full).
- **Owning team.** Vehicle **controls & autonomy** (SYSTEM_DESIGN §5.1: domains 13-18, including this
  one, are owned by that team) — titles like controls engineer or GNC (guidance/navigation/control)
  engineer; closely adjacent to **electrical/embedded** (who own the ESC/thruster-driver hardware this
  project's output ultimately commands) and, for classed work-class vehicles, **QA & functional safety**
  (who own the DP-class redundancy case this project's failure-analysis stage is a teaching-scale
  version of).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
