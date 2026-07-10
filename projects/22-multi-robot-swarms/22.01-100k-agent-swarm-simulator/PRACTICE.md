# 22.01 — 100k-agent swarm simulator: flocking, pheromone grids, stigmergy: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

A "swarm" is not one machine — it is many identical, mass-produced units. That changes what
"construction" means here versus a single-robot project like 08.01's cart-pole rig: the interesting
engineering is less about any one unit's mechanical design and more about the fact that the same
design gets built **N times**, and N is large.

- **The physical carrier, per unit.** The most direct physical match to a "100k-agent" simulated
  swarm is a fleet of small, cheap, mass-producible aerial or ground robots — a Crazyflie-class
  micro-quadrotor (~27–30 g, four brushless-DC or coreless motors on a molded/PCB frame) or a
  Kilobot-class ground robot (vibration-motor locomotion, a few centimeters across, built to cost
  under $20/unit at scale). Both are designed from day one around **manufacturability at fleet
  scale**: injection-molded or laser-cut frames (no hand-fitted parts), pick-and-place-populated PCBAs
  (no per-unit hand assembly), and firmware flashed identically to every unit off one build — the
  opposite design philosophy from a one-off research arm, where hand tuning is normal.
- **The pheromone-analogue hardware, if physically realized.** Real robots cannot easily secrete and
  sense chemical trails cheaply, so a physical stigmergy layer is usually emulated: a downward camera
  reading a projected light pattern on the floor, a low-power radio beacon whose received signal
  strength stands in for "concentration" and naturally attenuates with distance and time, or (in
  ground-robot research) a shared server maintaining a virtual field that robots query over Wi-Fi —
  each is a different physical/electrical carrier for the same diffuse-and-decay abstraction this
  project computes on a grid.
- **What breaks at swarm scale, and why it differs from single-robot failure.** A single dead or
  stuck unit among 100,000 becomes a *static obstacle* the rest of the flock must locally avoid,
  rather than a mission-ending failure — the robustness property decentralization buys. But new
  failure modes appear that a single-robot project never sees: battery degradation is a **fleet-wide
  statistical** problem, not a per-unit event (staggered charging/replacement schedules become a
  logistics problem in themselves); RF/comm congestion scales with N the same way this project's
  brute-force neighbor search does (an all-to-all beacon scheme is the physical-world version of the
  O(N²) algorithm THEORY.md shows is unusable — real swarm comms protocols solve the identical scaling
  problem this project's counting-sort grid solves computationally); and manufacturing defects that
  would be a one-off annoyance in a single robot become a *rate* — a 0.1% first-flight failure rate is
  1 broken robot in a research lab and 100 broken robots in a 100,000-unit fleet.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's model |
|---|---|---|
| Per-agent compute (real swarm) | MCU-class (e.g., STM32-class Cortex-M), not a GPU — each robot runs its *own* local flock step over a handful of sensed neighbors | What `flock_step_kernel`'s math would run on, one robot at a time, in the field |
| This demo's compute | Jetson Orin class / x86 + discrete GPU (reference machine: RTX 2080 SUPER) | Centralized *simulation/design-tool* role — see §3 for why this is explicitly not the deployment architecture |
| Neighbor sensing (real swarm) | Onboard camera + relative-pose estimation, ultra-wideband (UWB) ranging radios, or a swarm-specific optical/IR proximity sensor | Replaces this project's exact global-frame neighbor query with local, noisy, range-limited sensing (README Limitations) |
| Inter-agent / stigmergy comms | Low-power mesh radio (e.g., 2.4 GHz proprietary or Bluetooth Mesh-class) for beacon-based virtual pheromone, or none at all for camera-read physical/projected trails | The physical carrier of the pheromone grid's deposit/sense loop (§1) |
| Airframe / drivetrain | Hobby: Crazyflie-class micro-quadrotor kit (~US$150–250/unit) or a simple two-wheel ground bot; research: Bitcraze Crazyflie 2.1 + Crazyswarm2 infrastructure, e-puck2-class ground robots (~US$1k+/unit); production swarm-drone-show companies' custom airframes at volume (~US$1–3k/unit, order-of-magnitude, heavily volume-dependent) | The `u` a real per-agent controller would command |
| Onboard IMU / state sensing | MEMS IMU (accelerometer + gyro, often integrated with the flight-controller MCU) | Feeds each agent's own local state estimate (the "upstream input" in the README System-context section) |
| Ground-station / fleet compute | The same class of GPU box this demo runs on, or a cloud GPU instance | Design-time simulation (this project), pre-flight verification, and any centralized fleet-monitoring dashboard |
| Per-agent power | Small LiPo cell (hundreds of mAh, flight times of single-digit minutes for micro-quadrotors) | Sets the "how long can 100,000 agents actually fly" question this simulation does not model |

The invariant across every tier: **the compute that runs `flock_step_kernel`'s math in a real
deployment is distributed across N tiny MCUs, not centralized on one GPU.** This project's GPU is
doing the job of a simulation/design environment (or, in a hybrid architecture, a ground station that
might broadcast occasional coarse coordination hints) — never mistake the demo's centralized 100k-wide
kernel launch for how the compute would actually be arranged on a fielded swarm.

## 3. Installation & integration — putting it on a real robot

**Where this code would physically run: nowhere on a single deployed agent, as written.** This
project's GPU kernels compute *all* 100,000 agents' neighbor searches and rule updates in one place —
that is a simulation and design tool (used for tuning gains, validating emergent behavior, and
pre-flight regression testing before a real flight campaign), or a digital twin run alongside a
smaller real sub-fleet, not a deployment target. Each **physical** agent in a real swarm instead runs
its own greatly-scaled-down version of the identical math in `finish_agent()` — a handful of sensed
neighbors, not thousands — on its own onboard MCU, independently and asynchronously from every other
agent.

- **Per-agent ROS 2 shape.** Each robot could run a `swarm_agent` node subscribing to its own state
  estimate (`nav_msgs/Odometry`) and to locally-sensed neighbor state — either a lightweight custom
  `NeighborState[]` message populated from onboard relative sensing, or (in the beacon-based stigmergy
  variant) a low-rate broadcast topic carrying just a scalar "pheromone read here" value — and
  publishing `geometry_msgs/Twist` (or a lower-level rotor/motor command) at the coordination rate
  (this project's 20 Hz is a reasonable target). A separate, much lower-rate **ground-station** node
  could aggregate reported positions into a shared visualization (the density/pheromone heatmaps this
  demo writes are exactly what such a dashboard would show) — but that aggregation is monitoring, not
  a per-tick control dependency, unlike this demo's centralized kernel.
- **Real-time constraints.** Per-agent MCU firmware runs the flock step as ordinary embedded control
  code — not GPU-class parallelism (there is nothing to parallelize across when N ≈ a handful of
  sensed neighbors), so this is a soft-real-time loop on commodity MCU hardware, not the compute
  bottleneck a fielded swarm actually has. The bottleneck is **communication**, not FLOPs: reliable,
  low-latency neighbor sensing/beacon exchange at fleet scale is the harder real-time problem, and it
  is the direct physical analog of the O(N²)-vs-O(N) computational lesson in THEORY.md.
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1) — and it gets an extra rung a
  single-robot project does not need, because of N:**
  1. *Simulation* — this demo (and its natural extensions: noisy/range-limited neighbor sensing,
     per-agent asynchronous timing, communication dropout).
  2. *Small-N HIL* — a handful of real agents (2–5) plus the rest simulated, on the target per-agent
     MCU firmware, exercising the real communication stack.
  3. *Bench / tethered / caged* — a small physical group (single-digit to low tens of units) flown or
     driven in a netted/indoor test volume, geofenced, with a supervisor able to command an immediate
     land/stop for every unit.
  4. *Small free-operating group, outdoors, supervised* — under the regulatory constraints in §4, with
     automatic geofencing and individual return-to-home/land failsafes armed.
  5. *Scale-up* — only after the smaller-N behavior is trusted; scaling N introduces the communication-
     and logistics-scale failure modes named in §1, which a small trial cannot exercise.
- **The "E-stop" of a swarm is architecturally different.** A single robot has one E-stop chain
  (CLAUDE.md §1, SYSTEM_DESIGN §6.1). A swarm has, in effect, **N independent E-stops** — each agent
  needs its own reliable stop/land/return-home failsafe that does not depend on the coordination layer
  (or the other N−1 agents) working correctly, because the coordination layer is exactly the thing
  being tested. Designing "stop everything, safely, even if the swarm software has just misbehaved" at
  fleet scale is a genuinely different safety-architecture problem from the single-robot case, not a
  bigger version of the same one.
- **Calibration.** Per-agent state-estimator calibration (IMU/VIO biases) is identical to any single
  robot's bring-up; what is swarm-specific is **inter-agent relative sensing calibration** (camera or
  UWB ranging must agree closely enough across units that "neighbor within 1 m" means the same thing
  fleet-wide) and, for beacon-based stigmergy, calibrating decay/deposit parameters against the real
  radio or optical channel's actual attenuation — the real-world counterpart of tuning `kDecay` and
  `kDiffuse` in `kernels.cuh`.

## 4. Business & regulatory context

- **Who needs this capability.** Search-and-rescue and disaster-response research (coordinated
  coverage of a collapsed structure or disaster area), precision agriculture (coordinated aerial
  spraying/seeding/scouting over a field), warehouse and logistics fleets (the same coordination-
  algorithm family as this project, at the ground-robot scale SYSTEM_DESIGN's warehouse-AMR reference
  robot describes), infrastructure inspection at scale, and entertainment (drone light shows — though
  those are almost always pre-scripted trajectories replayed in formation for predictability and
  safety, not live emergent flocking; understanding *why* a safety-critical commercial product avoids
  the emergent behavior this project demonstrates is itself a useful lesson). Consistent with
  CLAUDE.md §1, this project is framed around these civilian/collaborative uses; no project in this
  repository is framed for weaponization, and swarm coordination research broadly has substantial
  legitimate civilian application independent of any defense context.
- **The players.** Bitcraze (Crazyflie hardware plus the open-source Crazyswarm/Crazyswarm2 ROS 2
  stack) is the most accessible real multi-quadrotor research platform; indoor swarm-drone-show
  companies (e.g., Verity and similar) productize precise multi-drone choreography; academic groups
  (Penn's GRASP Lab, IRIDIA's stigmergic-robotics lineage, and many others) publish the algorithms this
  project teaches toward. Reciprocal-velocity-obstacle libraries (RVO2 and successors) are the
  open-source backbone many production multi-agent navigation stacks build on, per THEORY.md.
- **Cost of getting it wrong.** A software bug in a single robot's controller breaks one robot. A
  software bug in a **shared** coordination rule — pushed identically to every unit in a fleet — can
  degrade or fail simultaneously across the *entire* swarm, which is a categorically larger blast
  radius than almost any other kind of robotics software bug. This is the direct business argument for
  the staged, small-N-first bring-up ladder in §3: a coordination-layer regression must be caught at
  N = 5, not discovered at N = 100,000 mid-deployment.
- **Regulatory.** Any real quadrotor swarm falls under the **drone/UAS row** of
  SYSTEM_DESIGN.md §6.2's regulatory map: **FAA Part 107** (US commercial small-UAS operations) or
  **EASA**'s open/specific/certified categories (EU) at minimum, and in most jurisdictions operating
  more aircraft than licensed remote pilots requires a specific waiver or falls under an evolving
  "specific category"-style regime — multi-aircraft swarm operations remain a genuine regulatory
  frontier, not a routine operation, as of this writing. Ground-robot swarms (warehouse/agriculture)
  instead sit under workplace-safety and (for shared-space-with-people operation) the service-robot
  standards line (ISO 13482) named in the same table. This is orientation, not a compliance checklist —
  consult current regulations and qualified counsel before any real multi-robot flight or operation.
- **Owning team.** The flocking/coordination algorithm itself is **controls & autonomy** work
  (SYSTEM_DESIGN §5.1); the same org map explicitly names **fleet operations** as the owner of "22
  fleet coordination" once a swarm is actually deployed and running, with **simulation & tools** owning
  the digital-twin environment this project is a teaching-scale version of, and **QA & functional
  safety** owning the staged bring-up ladder in §3 and holding veto power over any scale-up.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
