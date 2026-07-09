# 08.01 — MPPI controller — the canonical GPU controller: cart-pole → quadrotor → AGV → off-road racer: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> A controller's physical carrier is the ACTUATION CHAIN it commands — this file's sections 1–2
> teach that chain, because it is where a control engineer's software meets amps and heat.

## 1. Building it — construction of the robot/part

The subsystem this project belongs to on a real machine is the **drive chain** its force command
would flow into. For a cart-pole made physical (they exist in every controls lab), and equally for
each axis of the ladder's bigger plants:

- **The chain:** motor → transmission → load, sensed by encoders. A linear cart axis is typically
  a belt or ballscrew driven by a brushless motor; the pole pivot carries only an encoder (that is
  the *point* — the unactuated joint is what makes the problem interesting).
- **Construction realities a controller inherits:** backlash in the transmission (the pole feels
  the cart's force through whatever slack the belt has — mm of dead zone visibly degrade the
  catch), friction the frictionless model ignores (stiction at the pivot eats pumping energy;
  real cart-pole rigs need friction identification or compensation, project 08.15), structural
  compliance (a flexing mount adds an unmodeled oscillator — the classic way lab rigs surprise
  clean simulations), and sensor mounting (the pole encoder's zero must be calibrated to gravity's
  vertical, or "upright" in software leans in reality).
- **What breaks in the field:** belts stretch and skip teeth (the force-to-acceleration map
  drifts), connectors on moving axes fatigue, and E-stop chains get wired *around* prototypes by
  impatient people — the construction-level failure the safety section below exists to prevent.

The GPU compute enclosure story (where this kernel runs) is 33.01 PRACTICE §1's, unchanged.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-08. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

What a physical version of this loop runs on, tiered — the interesting column is where the
20 ms tick's pieces physically execute:

| Piece | Illustrative choices (2026) | Role in this project's loop |
|---|---|---|
| Compute for the rollouts | Jetson Orin class / x86 + RTX (reference machine: RTX 2080 SUPER) | The 0.3 ms rollout kernel per 20 ms tick |
| Real-time host | The same box's CPU (PREEMPT_RT Linux) or a separate MCU | The 50 Hz loop, softmin blend, state I/O |
| Motor-drive silicon | Brushless servo drive: control MCU + gate driver + MOSFET/GaN power stage + current-sense amps | Turns the force command into phase currents at 10–20 kHz current loops (SYSTEM_DESIGN item 6's chain) |
| Motor + transmission | Hobby: NEMA-class brushless + belt (~US$100s); research: integrated servo axes (~US$1–3k); industrial: certified servo systems (~US$3k+/axis) | The plant's `u` |
| Encoders | Incremental/absolute on cart axis AND pole pivot | The plant's `x` — a cart-pole is fully observable with two encoders |
| E-stop chain | Certified relay/safety controller, dual-channel | The layer that must exist BEFORE any experiment (see §3) |

The bullet's ladder shifts the BOM, not the shape: a quadrotor replaces the drive with four ESCs
and adds an IMU/EKF for state; the off-road racer replaces it with throttle/steering actuators and
a GNSS/INS stack. The 20 ms loop and the rollout kernel are the invariants.

## 3. Installation & integration — putting it on a real robot

**This is the project whose output is a force command — the §1 caveat applies at full strength:
everything here is sim-validated only; nothing below is a license to actuate.**

- **Process shape:** on a ROS 2 robot the controller runs as a real-time node (or inside
  `ros2_control` as a custom controller plugin): subscribes the state estimate, publishes effort/
  command messages each tick; the GPU rollout call sits inside the update. The demo's structure —
  persistent device buffers, one kernel per tick, host blend — maps one-to-one.
- **Real-time constraints, honestly:** 50 Hz with a ~1 ms GPU call is comfortably schedulable on
  PREEMPT_RT *as a soft deadline*; the driver's jitter tail means worst-case ticks can spike, so
  fielded designs (a) monitor deadline misses, (b) hold the previous plan on a miss (MPPI degrades
  gracefully — yesterday's shifted plan is still a plan), and (c) keep the kHz current loops on
  the drive's MCU, never behind the GPU (SYSTEM_DESIGN item 1's division of labor; 32.02's CUDA
  Graphs work is the frontier for tightening this).
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1):**
  1. *Simulation* — this demo, plus model-mismatch runs (perturb masses ±20%, add friction).
  2. *HIL* — the controller against a real-time simulated plant on the target computer, deadline
     monitoring on.
  3. *Bench, current-limited* — drive limits set to a fraction of capability, pole removed, then
     tethered; verify the force→current→motion chain and the E-stop *first* (press it; watch it
     work) before any closed-loop authority.
  4. *Free running* — only inside a physical envelope (track end-stops, torque limits in the
     drive) that does not depend on this software behaving.
- **Calibration:** encoder zeros (pole vertical!), force-constant identification (commanded u vs
  measured acceleration), and friction identification — the gap between this demo's ideal plant
  and any real rig lives in those three.
- **N/A here:** no fieldbus is implemented in this project (a real deployment would command the
  drive over EtherCAT/CAN-FD — SYSTEM_DESIGN item 6); the demo's "actuator" is a function call
  into the simulated plant. Stated per contract.

## 4. Business & regulatory context

- **Who needs sampling MPC:** off-road/defense autonomy (MPPI's birthplace), legged-robot
  companies (sampling whole-body controllers and their descendants), racing/performance driving
  programs, agile drones, and increasingly manipulation (sampling over contact-rich futures where
  gradients fail). It is the control-layer capability that turns "GPU on the robot" from a
  perception accessory into an autonomy architecture decision.
- **The players:** research stacks (AutoRally, MuJoCo-MPC), NVIDIA's rollout-based planning
  direction, and every legged/AV company's internal controls team; the gradient-MPC ecosystem
  (acados/OSQP/Drake, plus certified industrial MPC vendors) is the established counterweight.
  Build-vs-buy: controllers this close to actuation are almost always built and owned in-house —
  the model, the cost shaping, and the safety case are the product (SYSTEM_DESIGN item 5).
- **Cost of getting it wrong:** a controller bug is the most expensive kind of software bug a
  robotics company has — it breaks the robot, the payload, or worse, at actuator speed. The
  mitigations are architectural, not aspirational: certified E-stop chains and drive-level limits
  that do not trust this software (items 1/6), independent runtime monitors (31.04 CBF filters,
  31.08 STL monitoring — this project's natural safety siblings), and staged bring-up (§3).
- **Regulatory:** stochastic, GPU-hosted controllers currently sit OUTSIDE what certification
  regimes (ISO 10218/13482 for robots, ISO 26262/UL 4600 for vehicles) know how to bless — which
  is why fielded architectures put certified deterministic layers at the actuation seat and run
  sampling MPC above them, inside monitored envelopes. Consult the SYSTEM_DESIGN item-6
  orientation map; the SOUP framing (33.01 PRACTICE §4) applies to every library underneath this
  controller too.
- **Owning team:** controls/autonomy (titles: controls engineer, robotics software engineer —
  motion control); adjacent: simulation (owns the model this controller trusts), functional safety
  (owns the envelope and has veto power), and embedded (owns the drives and the real-time budget).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
