# 14.02 — Traversability costmaps fusing semantics + geometry: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's software has three physical carriers on a real off-road wheeled platform: the
**perception sensors** that produce the elevation and semantic maps this pipeline fuses, the
**compute box** that runs the fusion, and the **wheel/tire/suspension system** whose limits
`kernels.cuh`'s constants are meant to approximate.

- **The perception sensors.** Two DIFFERENT sensing modalities feed this pipeline's two channels, and
  they are usually two DIFFERENT physical devices: a 3D LiDAR or stereo/structured-light depth camera
  (feeding elevation mapping, e.g. project 05.05) and a separate RGB (or RGB+range) camera feeding a
  semantic segmentation network (domain 12/30). Both are rigidly mounted, typically forward-and-down
  facing so their field of view leads the vehicle's direction of travel by at least a few meters —
  this project's `kStopDistM = 2.0 m` bakes in an assumption that the sensing horizon comfortably
  exceeds the stopping distance the speed-limit layer reasons about (THEORY.md §The math). Construction
  realities: off-road vibration and dust/mud ingress are the two dominant field-failure modes for BOTH
  sensor types; the LiDAR/depth sensor's extrinsic calibration (`T_base_sensor`, SYSTEM_DESIGN §3.3) and
  the camera's own calibration must both be re-verified after any mechanical shock, because a
  mis-calibrated extrinsic silently misaligns the two channels this pipeline is fusing — a subtle,
  hard-to-detect failure mode worse than either sensor failing outright (the pipeline would fuse
  geometry from one location with semantics from a slightly different one).
- **The compute box.** Housed in a sealed, vibration-damped enclosure (an off-road platform's compute
  tier takes continuous shock a warehouse AMR's never does), typically with active cooling sized for
  sustained GPU load in an outdoor thermal envelope (direct sun load is a real design constraint an
  indoor AMR's compute box never faces).
- **The wheel/tire/suspension system.** `kernels.cuh`'s `kWheelRadiusM`, `kWheelMu`, `kTrackWidthM`, and
  `kCogHeightM` are all properties of THIS physical system, not tunable software parameters in the
  usual sense — they must be RE-DERIVED whenever the vehicle's tires, payload, or suspension change.
  Construction realities: a worn or under-inflated tire changes `kWheelMu` measurably (a worn, glazed
  tire has meaningfully less grip than a fresh, properly-inflated one — the same maintenance-item
  caveat 13.03's PRACTICE.md raises for its foot pad's friction coefficient); a raised payload or an
  added sensor mast changes `kCogHeightM` directly, and — per THEORY.md's rollover derivation — can flip
  WHICH hard-veto-limit governs (from traction- to rollover-governed) without any change to the terrain
  at all.
- **What breaks in the field:** tires wear and lose the tread pattern that sets their effective `mu`;
  suspension travel (not modeled by this project's rigid-body assumptions at all) absorbs some real
  step heights a rigid-vehicle model like this project's would flag as impassable — a genuine, honest
  gap between this teaching model and a real suspended vehicle's actual capability; and both perception
  sensors degrade gradually with dust/mud accumulation in a way that looks, to this project's kernels,
  like ordinary sensor noise until it crosses into genuine data loss — the same gradual-failure caveat
  13.03's PRACTICE.md raises for its own perception sensor.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Compute for the four kernels | Jetson Orin-class SoC (hobby/mid-tier off-road platforms) or x86 + discrete RTX (reference machine: RTX 2080 SUPER, research-tier); MCU-class is NOT viable — the windowed geometric-layer kernel needs real GPU parallelism to hit the measured sub-millisecond figures (README) | Runs all four kernels every costmap update (5–20 Hz) |
| Elevation-map sensor | Hobby: stereo depth camera (~US$100s); research: solid-state or spinning LiDAR (~US$1–10k); industrial/all-weather: ruggedized automotive-grade LiDAR (~US$5k+) | Produces the raw range data an elevation mapper (05.05) fuses into `elevation_m` |
| Semantic-segmentation sensor + compute | Hobby: a single RGB camera + a small on-device CNN; research: RGB + a mid-size segmentation network (e.g. a DeepLab/SegFormer-class model) run via TensorRT (12.01); industrial: multi-camera coverage + a validated, field-tuned model with a documented confusion matrix per class | Produces `semantic_class`/`confidence` — the segmentation net's own compute is USUALLY the dominant cost in this pipeline's real-world latency budget, not this project's four kernels |
| Tire/wheel | Hobby: off-the-shelf pneumatic or foam-filled RC/UGV tire (~US$10–50); research: purpose-selected off-road tire matched to expected terrain (~US$50–300); industrial: application-specific compound with a characterized, re-measured `mu` (~US$200–1000+) | Sets the REAL `kWheelRadiusM`/`kWheelMu` this project's constants only approximate |
| Suspension | Hobby: none or a simple sprung axle; research: independent suspension with meaningful travel; industrial: actively-damped or semi-active suspension | NOT modeled by this project's rigid-body step-height/slope limits at all — a real vehicle with suspension travel can absorb some step heights this project's teaching model would flag impassable (an honest, stated gap) |
| Compute↔actuator link | CAN-FD (throttle/steer/brake commands, ~5 Mbps) — SYSTEM_DESIGN §6.1's actuation-chain bus choice for a wheeled platform | Downstream of this project: cost/speed-limit -> local planner (14.01) -> vehicle control (08/14) -> drive-by-wire -> bus |

The bullet's own scope (fusing semantics + geometry into a costmap) is deliberately BELOW any specific
sensor or actuator choice: the same four kernels run unmodified whether the sensing suite below them is
hobby-grade or industrial, which is exactly why this project's `kernels.cuh` constants (the wheeled-
vehicle physical parameters, the class-prior costs, the fusion weights) are the things a real
integration would retune per-vehicle and per-deployment-environment.

## 3. Installation & integration — putting it on a real robot

- **Where this runs:** the same onboard compute that runs elevation mapping (05.05) and semantic
  segmentation (12.x) — typically a Jetson-class SoC or an x86+dGPU box, per SYSTEM_DESIGN §6.1's
  compute-tier diagram, NOT the real-time MCU tier (this pipeline's ~1.3 ms-scale GPU budget is a soft
  5–20 Hz deadline, not the 0.5–1 kHz hard deadline a whole-body/vehicle controller downstream of it
  faces).
- **The ROS 2 node shape this would take:** a node subscribing an elevation-map topic (a
  `grid_map_msgs`- or custom `GridMap`-shaped topic, mirroring this project's `elevation_m[W*H]` —
  SYSTEM_DESIGN §3.6) and a semantic-label topic (an `Image`-shaped topic carrying class ids +
  confidence, mirroring this project's `semantic_class[W*H]`/`confidence[W*H]`), time-synchronized
  (both must describe the SAME map instant — a real system needs an explicit sync/registration step
  this demo's single-timestep synthetic scenario does not exercise), publishing a fused-costmap topic
  (`fused_cost[W*H]`, this project's own shape, close enough to `nav_msgs/OccupancyGrid` that a real
  integration would likely quantize it into that exact message type — 23.01's byte-cost `[0,254]`
  convention is the natural target) and a speed-limit-layer topic for the local planner (14.01) to
  subscribe.
- **Real-time constraints, honestly:** the four kernels' measured GPU time (~1.3 ms total at this map's
  resolution) is comfortably inside even a 20 Hz soft deadline; the true bottleneck in a real deployment
  is almost always the SEMANTIC SEGMENTATION network's own inference time (§2's BOM note), not this
  project's fusion kernels. As with every GPU-hosted planning layer in this repo (SYSTEM_DESIGN §1.1), a
  missed deadline here should degrade gracefully (reuse the last cycle's fused costmap for one more
  planning tick, clearly flagged as stale) rather than stall the downstream planner behind it.
- **Calibration and bring-up, rung by rung (CLAUDE.md §1):**
  1. *Simulation* — this demo, plus perturbation runs (vary `kWheelMu`, `kTrackWidthM`/`kCogHeightM`,
     the fusion weights, the class-prior table) to stress-test the two hard-veto limits and the fusion
     rule before trusting them on anything that can drive into water or roll over.
  2. *HIL* — real elevation-map AND real semantic-segmentation output (from a real sensor suite on a
     static or teleoperated rig) piped into these kernels, comparing the fused costmap and speed-limit
     layer against a human operator's own judgment of the same terrain, cell by cell.
  3. *Bench/tethered* — the vehicle on a dyno/rolling-road or a short tethered/current-limited run over
     a controlled test terrain (a physical mockup of a berm, a shallow ditch, a wet patch) with an
     E-stop within reach; verify the vehicle's actual behavior at the commanded speed limit against
     what this project's kernels predicted for that terrain.
  4. *Free running* — only after (1)-(3) hold, starting on a forgiving surface (packed dirt, gravel)
     well clear of any real water hazard, with a spotter and a hard speed cap independent of this
     project's own computed limit, per the repo-wide caveat in CLAUDE.md §1.
- **N/A here:** no fieldbus, ROS 2 middleware, or physical sensor driver is implemented in this
  project — the demo's "sensors" are a synthetic, in-memory scenario (README §Data); a real
  deployment's message-passing, time-synchronization, and driver layer is exactly what SYSTEM_DESIGN
  §6.1's comms-bus diagram maps out.

## 4. Business & regulatory context

- **Who needs this capability:** off-road and agricultural autonomy companies (autonomous tractors,
  orchard/row-crop robots, forestry and mining platforms), outdoor inspection and security robots
  operating on unstructured ground, and the off-road/rural-driving edge of the autonomous-vehicle
  industry. It is the capability that turns "a wheeled robot that can navigate a paved lot" into "a
  wheeled robot that can navigate a field, a construction site, or a disaster-response scene" — the
  fusion layer every off-road autonomy stack needs somewhere, whether built in-house or licensed.
- **The players:** agricultural autonomy companies (row-crop and orchard robotics, autonomous tractor
  retrofits), off-road/defense-adjacent autonomy programs (the DARPA RACER lineage, README §Prior art),
  and a growing field of field-robotics research groups all ship some version of geometric+semantic
  traversability fusion; the open-source `elevation_mapping`/`grid_map` ecosystem (13.03's closest
  reference, increasingly extended with learned semantic layers) is the closest widely-studied
  open foundation the geometric half of this pipeline builds on. Build-vs-buy: the GEOMETRIC layer is
  common enough across off-road robots that teams often start from an open-source elevation-mapping
  base; the SEMANTIC layer (what the segmentation network is actually trained to recognize — crop rows,
  specific hazard types, a particular biome's vegetation) is usually the differentiator and is rarely
  bought off the shelf (SYSTEM_DESIGN §5.3's build-vs-buy criteria).
- **Cost of getting it wrong:** a bad traversability call off-road is not a curb bump — it can be a
  rollover, a vehicle stuck axle-deep in mud or water (an expensive, sometimes dangerous recovery
  operation), or damage to a valuable payload (agricultural equipment, a sensor suite, cargo). Because
  this layer's output feeds a downstream controller that commands real vehicle motion at real speed
  (the speed-limit layer, README §System context), the mitigations are the same architectural ones
  every actuation-adjacent project in this repo relies on: independent runtime monitors (31.x) that do
  not trust this software's cost alone, a HARD veto that no confidence value can soften for the two
  failure modes that matter most (THEORY.md §The two-channel fusion problem), and staged bring-up (§3).
- **Regulatory:** an off-road autonomous platform operating on private/controlled land (agriculture,
  mining, construction) typically falls outside the on-road vehicle regulatory regime entirely, governed
  instead by general workplace-safety law and machinery directives; a platform that ever operates on
  or crosses public roads re-enters **ISO 26262**/**UL 4600** territory (SYSTEM_DESIGN §6.2's AV row);
  a platform working near untrained people (inspection, agriculture with farmworkers present) most
  plausibly falls under **ISO 13482** (SYSTEM_DESIGN §6.2's service-robot row, the same row 13.03's
  PRACTICE.md cites). No standard has a specific traversability-costmap-fusion certification path
  today — this is orientation, not compliance guidance, and the SYSTEM_DESIGN item-6 map is the place
  to start a real regulatory review, not this file.
- **Owning team:** controls & autonomy, specifically the navigation/costmap specialization
  (SYSTEM_DESIGN item 5) — the same team that owns elevation mapping (05) and the local planner/vehicle
  controller (14.01/23/08) this project's output feeds; adjacent teams include perception (owns the
  depth sensor AND the semantic-segmentation model, §1-2 above), ML/data (owns the segmentation
  network's training and validation, including its per-class confusion rates — directly relevant to
  how confidently this project's fusion should trust each class), and functional safety (owns the
  runtime monitors and the independent speed/rollover envelope, §4 above).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
