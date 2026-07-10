# 10.03 — Massively parallel robot sim (Isaac-Gym-style: one robot, 10,000 environments): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project has TWO physical carriers, and each is already taught elsewhere in this repo — this
section names them and points at the right place rather than repeating either:

- **The simulated robot** (the cart-pole itself) is 08.01's territory:
  [`08.01/PRACTICE.md §1`](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/PRACTICE.md#1-building-it--construction-of-the-robotpart)
  covers the drive chain, encoders, and the friction/backlash a real cart-pole rig fights that this
  frictionless model ignores — unchanged by domain randomization (randomizing the SIMULATED mass/length
  does not change what a PHYSICAL cart-pole rig is built from).
- **The physical carrier this project actually adds** is the machine the farm RUNS on: a GPU
  compute enclosure — a workstation tower or a rack server, not a robot at all. That enclosure's
  construction (case, PSU, cooling, PCIe slot mechanicals) is the SAME story
  [`33.01/PRACTICE.md §1`](../../33-foundational-libraries/33.01-batched-small-matrix-linalg/PRACTICE.md)
  tells for every GPU-hosted project in this repo, unchanged here — 08.01's PRACTICE.md cites the same
  section for the same reason. What is genuinely NEW about this project's physical footprint is scale
  and duty cycle, covered in §2 below: a training farm runs GPUs at near-100% utilization for hours to
  days at a time (unlike an on-robot inference GPU's bursty, thermally-constrained duty cycle), which
  is a materially different cooling/power engineering problem than a single desktop GPU doing a demo.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

Where a real domain-randomization training farm like this project's teaching core actually runs,
tiered by scale — the interesting axis here is **cloud vs. workstation**, not sensor/actuator silicon
(this project has no sensors or actuators; it IS the simulator):

| Tier | Illustrative choices (2026) | Role |
|---|---|---|
| Desktop / single-researcher | Workstation-class discrete GPU (reference machine: RTX 2080 SUPER, 8 GB) | This demo's actual scale (10,000 envs, ~1 ms/step) — fine for algorithm development and this project's teaching core |
| Small team / lab | Multi-GPU workstation or a handful of cloud instances (RTX 4090/5090-class or A-series cloud GPUs) | Scaling `N` into the 100k–1M environment range; still single-node |
| Production RL training | Multi-node GPU clusters (H100/H200/B-series-class, NVLink/InfiniBand-connected) | Where Isaac Gym/Lab-scale locomotion policies are actually trained — thousands of GPU-hours per policy is a normal budget |
| Cloud vs. on-prem | Cloud (AWS/GCP/Azure GPU instances, or Lambda/CoreWeave-style GPU clouds) trades capex for opex and elastic scaling; on-prem clusters trade that flexibility for lower marginal cost at sustained high utilization | A build-vs-buy decision (SYSTEM_DESIGN item 5.3) most simulation teams revisit as training volume grows |

**The duty-cycle difference that matters here:** an on-robot GPU (e.g., a Jetson Orin doing perception)
is thermally and power-constrained and runs BURSTY workloads with idle gaps; a training-farm GPU runs
pinned at high utilization for the ENTIRE training run, which is a data-center cooling and power-budget
problem (rack-level airflow, PDU capacity, PUE) that a single workstation's case fans do not have to
solve — the reason production RL training happens in data centers, not on desks, once `N` and `T` grow
past what one card can hold in memory and finish in a workday.

## 3. Installation & integration — putting it on a real robot

**This project's output is a POLICY (or, here, a fixed gain vector) that a LATER project deploys — not
a command that moves hardware directly. That one step of indirection changes the safety story
relative to 08.01/13.03, but does not remove it** — see the sim-to-real honesty note below.

- **Where this runs in a company's infrastructure:** NOT on the robot. This is a `simulation & tools`
  team asset (SYSTEM_DESIGN item 5.1) running on training infrastructure (§2) — typically a CI/training
  cluster that also runs regression sims, HIL rigs (SYSTEM_DESIGN item 5.2's EVT/DVT stage), and
  nightly validation farms. The ONLY artifact that ever reaches the robot is the trained policy's
  weights (or, in this teaching core, a hand-derived gain vector) — a few kilobytes to megabytes,
  deployed like any other software update, not a live connection to this simulator.
- **The ROS 2 shape this feeds, eventually:** the deployed policy runs INSIDE the robot's control node
  (the same `ros2_control`-style shape 08.01/PRACTICE.md §3 describes for MPPI), subscribing
  `sensor_msgs/JointState`-shaped state and publishing effort commands — this project produces the
  WEIGHTS that node loads, not the node itself.
- **Sim-to-real transfer — the honest gap domain randomization narrows but does not close:** training
  (or, here, validating) a controller across a randomized PARAMETER envelope (§THEORY.md's domain
  randomization) makes it robust to the KINDS of uncertainty that were randomized — mass, length,
  within the ranges tested. It does NOT automatically cover: unmodeled dynamics (friction, backlash,
  structural compliance — 08.01/PRACTICE §1's list), sensor noise and latency (this project assumes
  perfect, instantaneous state — no observation model at all), actuator dynamics beyond an ideal force
  clamp (real motors have bandwidth limits, dead zones, and current limits this project's `kUmax` clamp
  does not model), or genuinely novel failure modes outside the randomized envelope entirely. The
  respected practice is the FULL testing ladder, not "trust the randomization and skip to hardware"
  (CLAUDE.md §1's caveat, restated at this project's own strength): **simulation → sim-to-real gap
  analysis / system identification → HIL → bench, current-limited → tethered → free running**, with
  the SAME E-stop and limit discipline 08.01/PRACTICE §3 describes at every physical rung. Nothing in
  this project's output is validated against real hardware; it is exclusively a simulation and
  training-infrastructure teaching core.
- **N/A here:** no fieldbus, no real-time OS constraint, no calibration procedure — this project never
  touches a physical actuator or sensor at all; those all belong to whatever project consumes its
  output (08.01/13.03-style controllers) once deployed.

## 4. Business & regulatory context

- **Who needs this capability:** any robotics company doing more than hand-tuning controllers on one
  physical prototype — which in practice means most legged/manipulation/aerial companies past the
  earliest prototype stage (SYSTEM_DESIGN item 5.2's concept/prototype phase, where "simulation (10,
  11) ... dominate[s]"). Locomotion companies (quadrupeds, humanoids) are the heaviest users: a
  policy trained across thousands of randomized simulated robots is now the STANDARD way modern
  legged robots get their whole-body controllers, not a research curiosity (SYSTEM_DESIGN §2.3's
  "sim→real pattern": the GPU's biggest quadruped contribution is offline).
- **The players:** NVIDIA (Isaac Gym/Isaac Lab, PhysX) is the namesake and the dominant commercial
  platform; Google DeepMind's MuJoCo/MJX and Brax occupy the JAX/XLA-native, auto-differentiable niche;
  most well-resourced robotics companies (legged-robot makers especially) maintain SOME in-house
  simulation capability even when they also license a commercial engine, because the model IS part of
  their product (SYSTEM_DESIGN item 5.3's build-vs-buy: "if customers choose you because of your
  planner/controller, own it").
- **What getting it wrong costs:** a policy over-fit to the simulator (or under-randomized against real
  manufacturing/wear variation) fails in ways that only show up on real hardware — at best a
  disappointing demo, at worst a fall, a collision, or a damaged payload during the free-running rung
  of the testing ladder (§3). The mitigation is architectural, not aspirational: the testing ladder
  itself, plus independent runtime safety monitors (31.x — CBF filters, reachability) that do not
  trust the trained policy's competence, exactly as 08.01/PRACTICE §4 argues for its controller.
- **Regulatory:** simulation and training infrastructure sit OUTSIDE the certification regimes
  themselves (SYSTEM_DESIGN item 6.2's table — ISO 10218/13482, ISO 26262/UL 4600, etc. — govern the
  DEPLOYED robot, not the training pipeline that produced its policy), but a rigorous simulation-based
  validation campaign (documented randomization ranges, documented pass/fail criteria, exactly the
  shape this project's VERIFY/FARM/ENERGY gates model in miniature) is increasingly part of the
  EVIDENCE a safety case (UL 4600's framing especially) cites — "we validated this controller across N
  simulated trials spanning this parameter envelope" is a sentence a functional-safety engineer wants
  to be able to write, and it is only true if the simulation infrastructure that produced it is itself
  trustworthy (SYSTEM_DESIGN item 6's orientation map; this is orientation, not a compliance claim).
- **Owning team:** simulation & tools (SYSTEM_DESIGN item 5.1's mapping: "Sim environments, digital
  twins, CI farms, internal libs" → domains 10, 11, 33, 34 — this project sits squarely at that
  intersection of 10 and 33). Adjacent teams: controls/autonomy (consumes the trained policy, owns the
  deployment-side integration in §3), ML/data (owns the actual training ALGORITHM once 12.06 replaces
  this project's fixed gains with a learned policy), and QA/functional safety (owns the validation
  campaign this simulation infrastructure feeds evidence into).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
