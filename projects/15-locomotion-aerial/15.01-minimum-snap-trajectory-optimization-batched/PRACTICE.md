# 15.01 — Minimum-snap trajectory optimization batched over waypoint sets: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's code is pure computation — no sensor, no actuator of its own — so its physical
carrier is the **quadrotor airframe and the companion computer bolted to it** that would run this
trajectory generator, and the flight-controller stack downstream that would fly the result.

- **The airframe.** A small quadrotor (the reference robot, SYSTEM_DESIGN.md §2.4) is a rigid
  central frame (carbon-fiber plate or 3D-printed/machined arms), four motor mounts, landing gear,
  and a stack of PCBs (flight controller, power distribution, companion computer) usually isolated
  from the frame by soft rubber/foam standoffs — **vibration isolation matters directly to this
  project's output**: a high-snap trajectory segment asks for rapid rotor-speed changes, which can
  excite frame resonances; the IMU that the state estimator (upstream of this project,
  SYSTEM_DESIGN.md Chain D) reads is *always* isolated for exactly this reason, and a trajectory
  generator that ignores vehicle dynamics (as this teaching version does — THEORY.md's honesty
  section) can ask for motion that makes vibration worse, not better.
- **Companion-computer mounting.** The GPU/SoC board this kernel would run on sits on its own
  vibration-isolated tray, wired to the flight controller over a serial or Ethernet link (§3 below),
  with its own regulated power rail (motor-driven power buses are electrically noisy; compute and
  power electronics are kept on separate, filtered rails on any careful build).
- **What breaks in the field:** loose motor mounts and worn bearings change the airframe's real
  dynamic limits out from under a fixed-parameter trajectory generator (the max acceleration/angular
  rate the vehicle can ACTUALLY deliver drifts as hardware wears — a mismatch this teaching version,
  which never queries vehicle limits, cannot detect); connector fatigue on the vibrating
  companion-computer-to-flight-controller link is a classic intermittent-fault source; propeller
  damage or asymmetric wear shifts the thrust-to-torque mapping differential flatness assumes is
  known and fixed.

The GPU compute enclosure story (thermal, connectors, the physical box this kernel runs inside) is
33.01 PRACTICE §1's, unchanged — a companion computer is, physically, the same class of small
embedded GPU box discussed there, just flying instead of sitting on a bench.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

Where this project's computation would physically execute on a real quadrotor, tiered:

| Piece | Illustrative choices (2026) | Role in this project's computation |
|---|---|---|
| Companion computer (runs THIS kernel) | Jetson Orin Nano/NX class (hobby/research, ~US$200–700); x86 mini-PC + entry dGPU (research bench rigs); automotive/embedded SoC modules (industrial) | Where the batched 32×32 solve (and, upstream, VIO/state estimation) would run — this project's demo runs on a desktop RTX 2080 SUPER as a stand-in for this tier |
| Flight controller | Pixhawk-class (STM32 Cortex-M7 class MCU running PX4 or ArduPilot firmware; hobby ~US$50–200, industrial/certified variants far higher) | Runs the kHz attitude/rate control loops (SYSTEM_DESIGN.md §1.1) that TRACK the trajectory this project produces — never runs this project's own compute |
| IMU | MEMS 6-axis (hobby, integrated on the FC) up to tactical-grade fiber-optic/MEMS (research/industrial, US$100s–1000s+) | Feeds the upstream state estimator (VIO backend, SYSTEM_DESIGN.md Chain D) whose output seeds this project's start conditions |
| Motors + ESCs | Brushless outrunner + BLHeli/DShot ESC (hobby, ~US$20–60/motor); higher-KV, higher-current industrial variants for larger payloads | The actuation chain a snap-minimizing trajectory is, physically, trying to be gentle to (THEORY.md's differential-flatness derivation) |
| Comms link (companion ↔ FC) | UART/MAVLink (hobby/research standard); uXRCE-DDS over serial or Ethernet (PX4 v1.14+, ROS 2-native) | Carries the position/velocity/acceleration SETPOINTS this project's polynomial, evaluated at the FC's rate, would stream downstream — see §3 |
| GPS/RTK (outdoor flight) | Consumer GNSS (hobby, meter-level); RTK-corrected (research/industrial, cm-level, ~US$200–1000+) | Part of the state estimate this project's start conditions depend on; N/A indoors (VIO/motion-capture substitutes) |

## 3. Installation & integration — putting it on a real robot

**This project's output is a trajectory that would feed a controller commanding real rotors — the
§1 caveat applies at full strength: everything here is sim-validated only; nothing below is a
license to fly anything.**

- **Process shape.** This kernel would run on the companion computer (§2), NOT on the
  resource-constrained flight-controller MCU. The companion computer's node (a ROS 2 node in a real
  stack) would: receive/hold a waypoint set (from a mission planner or operator), run this project's
  batched solve (in practice batch size 1 — one trajectory per replan, not 10,000; this project's
  large batch is a teaching/statistics device — see README §system-context), and then, at the
  flight controller's OWN update rate, evaluate the resulting polynomial (`eval_segment_derivs`'s
  logic, already written and reused unchanged) to produce position/velocity/acceleration setpoints —
  never shipping raw polynomial COEFFICIENTS to the flight controller, which expects a stream of
  numeric setpoints, not a symbolic trajectory.
- **The link and message shape.** MAVLink's `SET_POSITION_TARGET_LOCAL_NED` (or PX4's newer
  uXRCE-DDS / ROS 2 `TrajectorySetpoint` message) is the concrete wire format: position, velocity,
  and acceleration (a subset of this project's `(x, ẋ, ẍ)` output, evaluated at the current time)
  streamed at tens of Hz, consumed by PX4's own internal position controller, which in turn feeds
  the attitude/rate loops (SYSTEM_DESIGN.md §1.1's 0.5–1 kHz band) that this project never touches
  directly.
- **Real-time constraints, honestly.** The REPLAN rate (10–50 Hz, SYSTEM_DESIGN.md §1.1) is soft
  real-time — a missed replan can be covered by continuing to evaluate the PREVIOUS trajectory for
  one more tick (this project's batch has no notion of "previous trajectory" to fall back on; a
  real online replanner would keep one). The setpoint-STREAMING rate to the flight controller is a
  harder deadline (PX4's position controller expects setpoints at a bounded staleness or it holds
  position / triggers a failsafe) — this project's demo evaluates the artifact trajectory offline,
  not against any such deadline.
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1):**
  1. *Simulation* — this demo's batch verification is step zero; a real bring-up continues in PX4
     SITL (software-in-the-loop, Gazebo/jMAVSim) flying the ACTUAL vehicle model against generated
     trajectories, checking feasibility (required thrust/angular rate within the simulated
     airframe's limits) before anything spins.
  2. *HIL* — the trajectory generator and the real flight-controller firmware, against a simulated
     vehicle, on the real companion-computer/FC hardware pair, over the real comms link.
  3. *Bench, props off / tethered* — verify the comms link and setpoint timing with the vehicle
     restrained, propellers removed or the vehicle secured, before any motion is possible.
  4. *Free flight* — ONLY inside a netted cage or controlled test range, RC override / kill switch
     always live and tested, starting from trivial waypoint sets (a short hover-to-hover hop) and
     only working up in aggressiveness after each rung passes cleanly.
- **Calibration.** Accelerometer/gyro biases, motor thrust curves (needed to translate this
  project's implicit "required acceleration" into real throttle commands), and the vehicle's actual
  maximum safe acceleration/angular-rate envelope (feasibility-checking a trajectory against this
  envelope — README Exercise territory, not implemented here) are the gap between this project's
  ideal, dynamics-free polynomial and any real airframe.
- **N/A here:** no ROS 2 node, no MAVLink/DDS wiring, and no PX4 integration is implemented in this
  project — the demo's "flight controller" is a CSV file and a rasterized image. Stated per
  contract.

## 4. Business & regulatory context

- **Who needs this capability.** Aerial-inspection and mapping-drone companies (the dominant
  commercial use of small quadrotors today), drone-delivery operators (route segments become
  waypoint-to-waypoint trajectory-generation problems at scale), cinematography/racing-drone R&D
  (where aggressive, near-limit trajectories are the whole product), agricultural and
  infrastructure-survey fleets, and any research group building an autonomy stack on top of
  PX4/ArduPilot. It is squarely a **local-planning** capability (SYSTEM_DESIGN.md §1) — table stakes
  for autonomous flight, not usually a company's differentiator by itself.
- **The players.** PX4 and ArduPilot are the dominant open-source flight-control ecosystems this
  project's output would feed; ethz-asl's `mav_trajectory_generation` and similar academic libraries
  are the reference implementations of the "real" free-derivative QP (THEORY.md); commercial players
  (Skydio, DJI, Zipline, Wing, and countless smaller inspection/delivery/agriculture companies) build
  proprietary planning stacks on top of, or instead of, these open-source building blocks. Build-vs-
  buy: trajectory generation of this KIND is close to commodity (open-source libraries cover it
  well) — the differentiator is usually the LAYER ABOVE it (mission planning, obstacle-aware
  replanning, fleet coordination) or BELOW it (vehicle-specific feasibility/safety margins), not the
  minimum-snap solve itself (SYSTEM_DESIGN.md §5.3's build-vs-buy criteria).
- **Cost of getting it wrong.** A trajectory that asks for accelerations or angular rates the
  airframe cannot deliver (this project never checks feasibility — THEORY.md's honesty section) can
  cause a loss of control; over people or property, that is a genuine safety incident, not just a
  broken demo. The mitigations are architectural: certified/tested flight-controller firmware that
  enforces its OWN internal limits regardless of what setpoints arrive (PX4's position controller
  saturates and fails safe rather than blindly obeying an infeasible setpoint stream), geofencing
  and RC-override hardware that does not trust the autonomy stack, and the staged bring-up ladder in
  §3.
- **Regulatory.** Small commercial drone operations in the US fall under **FAA Part 107**
  (operational limits: visual line of sight, altitude ceilings, operator certification); the EU's
  **EASA** framework separates open/specific/certified categories by risk. Neither regime
  specifically evaluates a trajectory-generation ALGORITHM — they govern the OPERATION (where, how
  high, over whom, with what oversight) — see `docs/SYSTEM_DESIGN.md` §6.2's regulatory-landscape
  table, row "Drones / UAS," for the fuller orientation. This is didactic orientation, not a
  compliance pathway for any real flight.
- **Owning team.** Controls/autonomy (titles: autonomy engineer, flight-software engineer,
  guidance-and-control engineer) typically owns trajectory generation, adjacent to state estimation
  and the tracking controller it feeds; simulation owns the vehicle model any real feasibility check
  would validate against; flight-test/safety owns the bring-up ladder in §3 and has veto power over
  anything that flies (SYSTEM_DESIGN.md §5.1).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
