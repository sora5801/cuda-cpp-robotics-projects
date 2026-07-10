# 13.03 — Foothold scoring kernels: slope, roughness, edge distance from elevation maps: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's software has two physical carriers on a real quadruped: the **perception sensor** that
produces the elevation map this pipeline consumes, and the **foot/leg** whose placement the pipeline's
output ultimately governs.

- **The perception sensor.** A depth camera or small solid-state LiDAR, rigidly mounted to the
  robot's chassis (typically front-facing, angled down toward the ground ahead of the robot) so its
  field of view sweeps the terrain the legs are about to cross. Construction realities: the mount must
  survive the same shock and vibration the whole chassis does at trot/gallop frequencies (loose optics
  smear the depth image exactly when the robot needs it sharpest); the sensor's window needs
  environmental sealing (dust, mud, rain) rated for the deployment environment; and its extrinsic
  calibration (the fixed transform from sensor frame to `base` frame, `T_base_sensor` in
  SYSTEM_DESIGN §3.3's convention) must be re-verified after any mechanical shock, because that
  transform is exactly what turns a depth pixel into the map-frame height this project reads.
- **The foot.** Typically an elastomer (rubber-like) pad over a rigid core, sometimes with an embedded
  contact/force sensor (§2 below). Construction realities that directly matter to this project's `mu`
  parameter: the pad's compound and wear state SET the real friction coefficient — a worn, glazed pad
  has a meaningfully lower `mu` than a fresh one, meaning `THEORY.md`'s friction-cone slope limit is
  not a fixed constant on a real robot, it degrades with foot wear and must be re-characterized
  periodically (a maintenance item, not a one-time calibration).
- **What breaks in the field:** foot pads wear and crack (changing `mu` and adding compliance the rigid
  contact-point model in `THEORY.md` does not capture); the perception sensor's mount loosens over
  thousands of trot cycles; and mud/snow packed onto the sensor window silently degrades range accuracy
  in a way that looks, to this project's kernels, exactly like ordinary sensor noise (the ripple field
  in `THEORY.md`'s terrain model) until it gets bad enough to look like a full hole — a gradual failure
  mode worth monitoring for, not just handling at the two extremes this project models.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Compute for the four kernels | Jetson Orin-class SoC (hobby/mid-tier quadrupeds) or x86 + discrete RTX (reference machine: RTX 2080 SUPER, research-tier); MCU-class is NOT viable — the plane-fit + bounded-search kernels need real GPU parallelism to hit the measured sub-millisecond figures (README) | Runs all four kernels every elevation-map update (10–30 Hz) |
| Depth/range sensor for the input map | Hobby: stereo depth camera (~US$100s, e.g. an Intel RealSense-class module); research: solid-state or spinning LiDAR (~US$1–10k); industrial/all-weather: ruggedized automotive-grade LiDAR (~US$5k+) | Produces the raw range data an elevation mapper (05.05) fuses into `height_m` |
| Foot contact/force sensing | Hobby: none, or a simple binary bump switch (~US$1–10); research: a single-axis force sensor in the foot (~US$50–300); industrial: a full 6-axis force/torque sensor per foot (~US$500–3000+) | NOT consumed by this project's kernels directly, but the ground-truth signal that would VALIDATE this project's predictions against reality — "did the foot actually land where and how firmly the score predicted?" |
| Leg actuation | Hobby: geared BLDC + encoder (~US$50–200/joint); research/industrial: quasi-direct-drive or high-torque-density actuators with integrated torque sensing (~US$500–3000+/joint) | The chain that carries out whatever the downstream gait planner (13.02/13.08) does with this project's `FootholdResult` |
| Compute↔actuator link | CAN-FD (motor commands, ~5 Mbps) or EtherCAT (synced servo chains, sub-ms) — SYSTEM_DESIGN §6.1's actuation-chain bus choices | Two-three hops downstream of this project: score -> foothold -> gait plan -> joint torques -> bus |

The bullet's own scope (slope, roughness, edge distance — the map-scale layer) is deliberately BELOW
any specific actuator or sensor choice: the same four kernels run unmodified whether the legs below
them are hobby-grade or industrial, which is exactly why this project's `kernels.cuh` constants
(friction, thresholds, weights) are the only things a real integration would retune per-robot.

## 3. Installation & integration — putting it on a real robot

- **Where this runs:** the same onboard compute that runs elevation mapping (05.05) and the gait
  planner (13.02/13.08) — typically a Jetson-class SoC or an x86+dGPU box, per SYSTEM_DESIGN §6.1's
  compute-tier diagram, NOT the real-time MCU tier (this pipeline's ~1 ms-scale GPU budget is a soft
  10–50 Hz deadline, not the 0.5–1 kHz hard deadline the whole-body controller downstream of it faces).
- **The ROS 2 node shape this would take:** a node subscribing the elevation map (a `grid_map_msgs`-
  or custom `GridMap`-shaped topic, mirroring this project's `height_m[W*H]` — SYSTEM_DESIGN §3.6) and
  a stream of nominal-foothold requests from the gait planner, publishing `FootholdResult`-shaped
  replies (map-frame coordinates + score + validity) — either as a service call per query or a batched
  topic once per gait-phase update, matching how this demo scores all of a gait cycle's queries in one
  kernel launch rather than one query at a time.
- **Real-time constraints, honestly:** the four kernels' measured GPU time (~1 ms total at this map's
  resolution) is comfortably inside even a 50 Hz soft deadline; the CPU-side terrain/query bookkeeping
  this demo does once at start-up would instead run once per elevation-map update in a real system —
  still cheap relative to the kernel time. As with every GPU-hosted planning layer in this repo
  (SYSTEM_DESIGN §1.1), a missed deadline here should degrade gracefully (reuse last cycle's score map
  for one more gait phase) rather than stall the pipeline behind it.
- **Calibration and bring-up, rung by rung (CLAUDE.md §1):**
  1. *Simulation* — this demo, plus perturbation runs (vary `mu`, the ripple amplitude, the rock field
     density) to stress-test the fusion weights before trusting them on anything that can fall over.
  2. *HIL* — real elevation-mapper output (from a real depth sensor on a static or teleoperated rig)
     piped into these kernels, comparing predicted footholds against a human's judgment of the same
     terrain.
  3. *Bench/tethered* — a single leg, or the whole robot suspended in a gantry, stepping onto a
     controlled test terrain (a physical mockup of a ramp/step/rock patch) with the robot's weight
     supported and an E-stop within reach; verify the foot force/torque sensor's reading at touchdown
     against the fused score's prediction.
  4. *Free running* — only after (1)-(3) hold, on a real but forgiving surface (grass, packed dirt),
     with a spotter and a torque-limited safety envelope, per the repo-wide caveat in CLAUDE.md §1.
- **N/A here:** no fieldbus, ROS 2 middleware, or physical sensor driver is implemented in this
  project — the demo's "sensor" is a synthetic, in-memory terrain (README §Data); a real deployment's
  message-passing and driver layer is exactly what SYSTEM_DESIGN §6.1's comms-bus diagram maps out.

## 4. Business & regulatory context

- **Who needs this capability:** every legged-robot company shipping outdoor or unstructured-terrain
  quadrupeds — inspection robots (oil/gas, utilities, construction sites), research platforms, and the
  emerging security/logistics quadruped market. It is the capability that turns "a quadruped that can
  walk on a flat lab floor" into "a quadruped that can cross a real work site" — the geometric layer
  every legged company's locomotion stack needs somewhere, whether built in-house or licensed.
- **The players:** ANYbotics (ANYmal), Boston Dynamics (Spot), Unitree, and a growing field of
  research and commercial entrants all ship some version of terrain-aware foothold planning; the
  open-source `elevation_mapping`/`grid_map` ecosystem from ETH Zurich (README §Prior art) is the
  closest widely-studied reference implementation of the upstream map this project's kernels consume.
  Build-vs-buy: the geometric scoring layer itself (this project's scope) is common enough across
  robots that teams often start from an open-source base and specialize the fusion weights/thresholds
  per robot and deployment — the differentiator is usually the LAYER ABOVE it (learned terrain
  semantics, joint gait-foothold optimization) more than the geometry itself (SYSTEM_DESIGN §5.3's
  build-vs-buy criteria).
- **Cost of getting it wrong:** a bad foothold call is a fall — at minimum a mission abort and a
  damaged/scuffed robot, at worst a damaged payload (an inspection sensor, a delivered package) or,
  for a robot working near people, a safety incident. Because this layer feeds a downstream controller
  that commands real leg motion, the mitigations are the same architectural ones every actuation-adjacent
  project in this repo relies on: independent runtime monitors (31.x) that do not trust this
  software's score alone, conservative validity thresholds with margin (this project's own
  `kValidThreshold` is a tuned safety margin, not a bare pass/fail line), and staged bring-up (§3).
- **Regulatory:** a quadruped operating in a workplace or public space (inspection, logistics) most
  plausibly falls under **ISO 13482** (service robots — hazard analysis for machines sharing space
  with untrained people), the SYSTEM_DESIGN §6.2 row this project's domain maps onto; an industrial
  quadruped permanently fixtured within a guarded cell would instead look more like ISO 10218's world.
  Neither standard has a specific foothold-scoring-algorithm certification path today — this is
  orientation, not compliance guidance, and the SYSTEM_DESIGN item-6 map is the place to start a real
  regulatory review, not this file.
- **Owning team:** locomotion/controls, inside controls & autonomy (SYSTEM_DESIGN item 5) — the same
  team that owns state estimation (04), elevation mapping (05), and the gait planner this project's
  output feeds; adjacent teams include perception (owns the depth sensor and its calibration, §1-2
  above) and functional safety (owns the runtime monitors and the fall-safe envelope, §4 above).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
