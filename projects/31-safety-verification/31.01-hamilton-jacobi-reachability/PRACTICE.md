# 31.01 — Hamilton-Jacobi reachability: level-set grid solvers (stencil ops — GPU-perfect): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is pure computation — it has no sensor, actuator, or mechanical part of its own. Its
**physical carrier** is whichever robot's safety envelope it computes, and its output (a
precomputed value field) is consumed by a **safety monitor process**, so the relevant "construction"
question is what that monitor's home looks like:

- **The monitor is software, but it sits next to hardware that matters.** A reachability-based
  safety monitor typically runs as a dedicated process (or a partition of one) on the robot's main
  compute, reading the same state estimate the controller reads, with **no write path into the
  actuation chain other than a veto/override signal**. That physical separation — the monitor
  cannot itself command motion, only forbid it — is a construction decision as real as any wiring
  choice: it is what makes the monitor trustworthy even if the controller it watches has a bug.
- **What the monitor watches is built like any other sensed/estimated quantity.** The state
  `(x, v)` this project reaches over is, on a real robot, the output of a state estimator (04.x) —
  itself built from encoders, IMUs, or vision, each with its own mounting, wiring, and calibration
  concerns (see the sibling flagships in domains 01–04 for that construction detail). This project
  assumes a clean `(x,v)` input and is honest that noisy/biased state estimates are a real gap
  between this demo and a fielded monitor (§3).
- **The GPU compute enclosure story** (where the offline solve and the online lookup physically
  run) is 33.01 PRACTICE §1's, unchanged — nothing about reachability computation demands special
  physical construction beyond a standard compute enclosure.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

Reachability has an unusual two-phase hardware story: the **offline solve** (this demo) and the
**online monitor** (the runtime lookup) have very different compute profiles.

| Piece | Illustrative choices (2026) | Role in this project's split |
|---|---|---|
| Offline solve compute | Jetson Orin class / x86 + discrete GPU (reference machine here: RTX 2080 SUPER); could equally run on a build/CI machine, not the robot itself | Runs this demo's ~1–3 ms kernel time once, ahead of deployment, whenever the dynamics bound or target changes |
| Online monitor compute | The robot's main compute (Jetson-class SoC or x86+dGPU) OR, for a small enough grid, even a capable MCU | A single array lookup at the state-estimator's rate (100–400 Hz) — cheap enough that this is rarely the bottleneck |
| Safety controller | Independent, dual-channel safety PLC/controller (SYSTEM_DESIGN §6.1's "SAFETY CONTROLLER" box) | The layer that ultimately gates power to the motors — this project's monitor should feed *into* this layer's decision, never replace it |
| State estimate source | Encoders + IMU (for a `(position, velocity)`-style axis), fused by a 04.x-style estimator | Supplies the `(x, v)` this project's field is queried with |
| E-stop chain | Certified relay/safety controller, dual-channel, hardwired (SYSTEM_DESIGN §6.1) | Must exist and be independently testable BEFORE any experiment that uses this monitor's output for anything (see §3) |

The compute for the *offline* solve is not resource-constrained the way an onboard control loop is
— it can run on a desktop, a build server, or the robot itself between missions. The *online* check
is the piece that must respect the robot's real-time budget, and it is, by construction, nearly
free (a lookup, not a solve).

## 3. Installation & integration — putting it on a real robot

**Everything in this repository is sim-validated only, not safety-certified (CLAUDE.md §1). This
project computes a safety-*style* metric didactically — it is emphatically NOT a certified safety
implementation, and nothing below is a license to gate real actuation with this code.**

- **Where it would run:** the *offline* solve (compute the value field for a given dynamics bound
  and target) runs wherever is convenient — a build machine, a fleet-wide batch job re-run whenever
  the dynamics model or target set changes. The *online* monitor (the array lookup) runs on the
  robot's main compute, alongside — never inside — the controller it watches.
- **ROS 2 shape:** the online monitor is naturally a node that subscribes the state estimate
  (`nav_msgs/Odometry`-shaped, per SYSTEM_DESIGN §3.6), looks up the precomputed field (loaded once
  at startup, e.g. from the PGM/CSV-style artifacts this demo writes, or a denser binary dump),
  and publishes a boolean/severity "inside the safe set" topic that a downstream safety filter or
  the controller's own guard logic subscribes to. It would never itself publish a `Twist`/effort
  command — the veto pattern from §1.
- **Which layer commands the override:** in the CBF-filter pattern (catalog 31.04, Chain D of
  `SYSTEM_DESIGN.md` §4.4), a monitor like this one's output *minimally modifies* the controller's
  commanded action rather than replacing the whole control stack — this project computes the set a
  CBF-style filter would be designed against, not the filter itself.
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1):**
  1. *Simulation* — this demo, plus sensitivity runs (perturb `umax`, the target level, the grid
     resolution) to understand how the computed set moves.
  2. *HIL* — the monitor node against a real-time simulated plant and a simulated state estimator
     with realistic noise, checking that the monitor's veto fires *before* an inevitable-collision
     state is reached, not after.
  3. *Bench, current-limited* — the monitor watching a real but power-limited actuator, verifying
     the veto path (not the actuation path) end to end; confirm the E-stop chain independently,
     first, before trusting any software veto at all.
  4. *Free running* — only inside a physical envelope (limit switches, torque limits) that does not
     depend on this monitor behaving correctly, exactly as SYSTEM_DESIGN §6.1 describes the
     hardwired safety chain as deliberately "boring" and independent of any GPU code.
- **Certified-layer honesty (matching 08.01 PRACTICE §3's framing for its own force-command
  output):** this project's output is a reachability *classification*, one step further from
  actuation than a force command — but a safety monitor whose classification is silently wrong is
  arguably *more* dangerous than a controller that is obviously wrong, because a safety layer is
  trusted precisely when other things fail. Nothing about this demo's `RESULT: PASS` constitutes
  evidence that a downstream safety architecture built on this code would be safe; it is evidence
  only that the numerics match a hand-derivable closed-form answer for one toy plant.
- **N/A here:** no fieldbus, no real sensor, no real actuator is implemented in this project — the
  demo's "state estimate" is a grid of `(x,v)` values generated from the scenario file. Stated per
  contract.

## 4. Business & regulatory context

- **Who needs this capability:** any company shipping mobile or manipulating robots that share
  space with people, valuable equipment, or each other — warehouse AMR fleets, collaborative
  manipulator cells, drones operating near structures or people, autonomous vehicles. Reachability-
  style safety analysis is squarely inside the **QA & functional safety** team's toolkit
  (SYSTEM_DESIGN §5.1's row for domain 31), working alongside controls/autonomy (who own the model
  being checked) — this project's domain (31, Safety, Verification & Testing) *is* the
  safety-standards-adjacent one in this repository's map.
- **The players:** the open-source `hj_reachability`/`OptimizedDP` toolboxes and Mitchell's
  `helperOC` lineage (academic, UC Berkeley/Stanford-adjacent) are the reference implementations;
  commercial functional-safety tooling (safety PLCs, certified motion-monitoring modules from
  industrial-automation vendors) occupies the *certified* layer this project's output would feed
  *into*, never replace. CBF-based runtime filters (31.04) are the more common production choice
  for the *online* half of this problem specifically because they avoid the dense-grid cost this
  project's THEORY.md is honest about.
- **What getting it wrong costs:** a safety monitor with a false "safe" verdict is a near-miss or a
  collision — downtime, equipment damage, injury, and (per SYSTEM_DESIGN §6.2) potential regulatory
  and liability exposure well beyond the cost of the software bug itself. A monitor with a false
  "unsafe" verdict costs performance (unnecessary stops), which is why real deployments tune the
  numerical conservatism (this project's boundary band is exactly that kind of tunable, measured
  conservatism, made visible instead of hidden) rather than treating either error as free.
- **Regulatory:** the applicable standards depend on the host robot type — industrial arms
  (**ISO 10218** / **ISO/TS 15066** for collaborative operation), service/mobile robots
  (**ISO 13482**), autonomous vehicles (**ISO 26262** / **UL 4600**) — see
  `SYSTEM_DESIGN.md` §6.2's orientation table for the full map. None of these standards recognize
  "we ran a CUDA kernel and it matched a closed-form formula" as certification evidence; a real
  safety case would require an independently developed, professionally validated implementation,
  an argued and evidenced safety case (UL 4600's framing), and process rigor (ISO 26262-style
  ASIL development) this didactic project does not attempt and should not be mistaken for.
- **Owning team:** QA & functional safety (titles: functional safety engineer, verification
  engineer) own work like this; adjacent teams are controls & autonomy (own the dynamics model and
  the controller being watched, per §5.1's "second opinion" framing in this project's README),
  simulation & tools (own the offline compute infrastructure the solve runs on), and regulatory/
  compliance (own translating any of this into the standards-map above, if it were ever pursued
  for real).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
