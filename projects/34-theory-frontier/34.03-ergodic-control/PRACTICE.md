# 34.03 — Ergodic control: spectral multiscale coverage (FFT-based — very GPU-friendly): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> This is a research-stage (`[R&D]`) guidance-layer algorithm — this file is honest about how far
> it is from a fielded product, and grounds each section in the platform that would carry it.

## 1. Building it — construction of the robot/part

This project's natural physical carrier (README §System context) is a **survey/inspection
quadrotor** — the reference robot ergodic coverage fits most directly (SYSTEM_DESIGN §2.4). What
that vehicle's construction actually involves:

- **Airframe.** A carbon-fiber or reinforced-nylon quadrotor frame (250 mm–900 mm motor-to-motor
  class for the survey missions this algorithm targets), four brushless motor/ESC/propeller
  assemblies, a battery bay sized for the mission duration (coverage missions run longer than
  racing/acrobatic flights, so battery volume dominates the airframe budget), and a
  vibration-isolated mount for the flight-controller IMU (vibration is the #1 practical enemy of
  the state estimate this algorithm's `x` depends on).
- **What breaks in the field.** Propeller strikes and imbalance (vibration feeding into the IMU,
  corrupting the state estimate this controller's `x` relies on), connector fatigue on a
  vibrating airframe, GPS multipath/dropout near structures a survey mission is often *specifically
  sent to inspect* (bridges, buildings, crop rows under tree cover), and battery degradation
  shortening the mission time this algorithm's `T` assumes fixed.
- **The compute enclosure.** Wherever the companion computer running this guidance loop lives, it
  needs thermal management (a Jetson-class SoC under a survey drone's plastic/composite shell can
  throttle in direct sun) and vibration isolation of its own — the GPU compute enclosure story is
  the same one 33.01 PRACTICE §1 tells for any GPU-carrying robot, unchanged here.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's loop |
|---|---|---|
| Companion compute | Jetson Orin Nano/NX class (survey-drone SWaP budget); x86+dGPU only for a ground-robot variant (reference machine here: RTX 2080 SUPER) | Runs the SMC guidance loop (`phi_k` build once, ~1,024-mode update per tick) |
| Flight controller | PX4- or ArduPilot-class MCU (STM32-family) | State estimation (VIO/EKF), attitude/velocity control loop, consumes this project's `u` as a velocity/heading setpoint |
| Position sensing | GNSS/RTK (outdoor survey) or VIO + downward-facing sensor (indoor/GPS-denied) | Supplies the state estimate `x` this controller reads every tick |
| Information source | RGB/multispectral camera + onboard lightweight inference, or a pre-loaded map | Produces (or updates) the target density `phi` — this is where 05.15/23.09-style information-gain scoring would run |
| Comms | Telemetry radio (900 MHz/2.4 GHz) or cellular/mesh for BVLOS missions | Mission monitoring, live `phi` updates, operator override |
| Power | LiPo/Li-ion pack sized for mission duration; BMS | The mission-time budget `T` this algorithm's ergodic guarantee is measured over |

Hobby-tier survey drones (~US$1–3k) can run this guidance loop at reduced K/rate on a Jetson-class
board; research-tier platforms add RTK GPS and multispectral sensing (~US$5–15k); industrial
survey platforms (agricultural, infrastructure-inspection) add certified avionics and redundant
compute, pricing well beyond either tier — verify current before relying on any of these numbers.

## 3. Installation & integration — putting it on a real robot

**This project's output (`u`, a velocity/heading command) is a signal that could command real
motion — the §1 caveat applies at full strength: everything here is sim-validated only, and
nothing below is a license to fly or drive.**

- **Process shape.** On a ROS 2 platform this runs as a **guidance-layer node**: subscribes the
  state estimate (`nav_msgs/Odometry`-shaped, matching SYSTEM_DESIGN §3.6's conventions) and an
  information-density topic (an occupancy-grid-shaped message carrying `phi`, refreshed by an
  information-gain front end — 05.15/23.09), and publishes a `geometry_msgs/Twist`-shaped velocity
  setpoint at 10–100 Hz (README §System context) for the flight controller's own inner loop (which
  runs its position/velocity control at a much higher rate — SYSTEM_DESIGN item 1's layering) to
  track.
- **Real-time constraints, honestly.** This algorithm's own per-tick cost (a ~1,024-mode GPU kernel
  plus an O(K) host reduction, measured at well under a millisecond in this demo) is trivially
  schedulable inside a 10–100 Hz guidance loop — the real integration risk is *upstream*: `phi`
  updates arriving late or a state-estimate glitch, both of which this controller has no special
  handling for (it would need a watchdog and a "hold last valid setpoint" fallback in any real
  deployment, the same discipline 08.01 PRACTICE §3 describes for its own guidance-adjacent loop).
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1):**
  1. *Simulation* — this demo, plus a simulated `phi` that changes mid-mission (this project's fixed
     target is the first thing to relax).
  2. *HIL* — the guidance node against a simulated flight controller/vehicle model on the actual
     companion computer, with deadline monitoring on the `phi`-update and state-estimate topics.
  3. *Bench, tethered/current-limited* — the guidance node commanding a REAL flight controller with
     motors disarmed or current-limited, verifying the velocity-setpoint interface end-to-end before
     any authority over real thrust.
  4. *Free running* — only inside a geofenced test area, with the vehicle's own independent
     safety systems (geofence, RTL-on-link-loss, operator override) active and NOT dependent on this
     guidance node behaving correctly.
- **Calibration.** State-estimate calibration (IMU/GPS/VIO alignment) is entirely the flight
  controller's job, not this node's; this project's own tunable surface is small — the speed budget
  `v_max` and the Sobolev exponent `s` (README Exercise 3) — and both are mission-planning choices,
  not hardware calibration.
- **N/A here:** no fieldbus is implemented in this project (a real deployment would publish over
  ROS 2/DDS to the flight controller, not a raw bus like CAN-FD/EtherCAT — those sit *below* the
  flight controller, in its own actuation chain, SYSTEM_DESIGN item 6). Stated per contract.

## 4. Business & regulatory context

- **Who needs this capability.** Survey and inspection operators (agriculture, infrastructure,
  environmental monitoring) whose missions have genuinely non-uniform information value; search-
  and-rescue programs, where "spend time where a person is likely to be" is a near-literal
  restatement of the ergodic-coverage problem statement; and, more speculatively, any autonomous
  platform tasked with monitoring a changing environment rather than a one-shot mapping pass.
- **The players.** This capability is overwhelmingly **research-stage** (the catalog's `[R&D]` tag
  is accurate, not conservative) — academic robotics labs are the primary source of published SMC
  and ergodic-exploration work (README §Prior art); production survey/inspection platforms today
  almost universally ship **frontier-based exploration** or **hand-authored waypoint grids**
  instead, because those are simpler to certify, debug, and explain to a customer. A team adopting
  ergodic control today is making a deliberate research bet, not buying an established capability
  — the build-vs-buy calculus (SYSTEM_DESIGN §5.3) currently has no "buy" option here at all.
- **What getting it wrong costs.** A guidance-layer bug in this class of system typically costs
  **mission quality** (missed detections, wasted flight time, an incomplete survey) rather than
  a safety incident directly — because a well-architected vehicle keeps this node several layers
  above the actuation chain (§3 above) behind an independent geofence/RTL safety system. That
  layering is precisely why fielding a research-stage algorithm like this one at the guidance layer
  is a *much* lower-stakes decision than fielding a research-stage algorithm at the control layer
  (contrast with 08.01 PRACTICE §4's controller, whose bugs are actuator-speed, not mission-quality,
  incidents).
- **Regulatory.** For the quadrotor reference robot: **FAA Part 107** (US commercial small-UAS
  operations) or **EASA**'s open/specific/certified categories (EU) govern the *vehicle's*
  operation regardless of what guidance algorithm it runs — this project's output never bypasses
  those operational limits, it only decides *where within the approved flight area* to spend time.
  For a ground-robot variant sharing space with people, **ISO 13482** (SYSTEM_DESIGN item 6's map)
  would apply to the platform. Consult the SYSTEM_DESIGN item-6 orientation map; this is
  **orientation, not compliance guidance**.
- **Owning team.** Research/autonomy (README §System context) — titles: research engineer, autonomy
  software engineer (exploration/coverage); adjacent teams: the perception/mapping team that would
  own the 05.15/23.09-style information-gain front end this project's `phi` stands in for, and
  flight-controls/embedded, who own the vehicle-level safety systems this guidance layer sits
  strictly above.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
