# 02.13 — Dynamic point removal (raycast free-space carving): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is software-only: there is no physical part to build. The "carrier" it serves is the
**map itself** — a data structure and its storage, not a mechanical assembly — so this section
describes how a mapping-session carving pass is physically OPERATED rather than built.

**When carving runs.** Two real deployment patterns, both legitimate, with different engineering
consequences:

- **Post-session batch** (what this project implements): a robot completes a mapping run — a
  warehouse walkthrough, a survey lap — accumulating K posed scans; carving runs ONCE at the end,
  over the whole session, on a workstation or a beefy onboard GPU with time to spare. This is how a
  site's reference map is typically built or refreshed (a scheduled, offline maintenance operation),
  and it is the natural fit for this project's exact, whole-batch VERIFY gates (THEORY.md "How we
  verify correctness").
- **Incremental / online** (what a deployed fleet robot needs day to day): carve each NEW scan's
  free-space evidence into a PERSISTENT ledger as it arrives, re-evaluating affected voxels'
  classification continuously. This needs (a) a ledger that survives across scans in device or host
  memory rather than being rebuilt from scratch, (b) a bound on how many PAST scans' evidence stays
  "live" (an unbounded ledger grows forever; real systems use a sliding window or a decay/log-odds
  scheme — THEORY.md "Where this sits in the real world" names OctoMap's log-odds update as the
  production answer), and (c) a policy for when a voxel's classification is allowed to FLIP back from
  DYNAMIC to STATIC (a spot correctly cleared once a car left could, in principle, become occupied
  again by something permanent — a real system needs a rule for this, and this project's one-shot
  batch classification sidesteps the question entirely by design).

**Storage and versioning of the ledger.** The three-array ledger (`hits`/`pass_from_hit`/
`pass_from_maxrange`, ~9.2 MiB for this project's 768,000-voxel grid) is small enough to keep
resident in GPU memory for a live session, and small enough to checkpoint to disk between sessions
(a flat binary dump of three `uint32_t[kNumVoxels]` arrays, or a sparse encoding for a mostly-empty
real-world grid — this project's dense array is a teaching simplification; PRACTICE.md §3 below
notes the sparse-structure alternative a city-scale map would need). Versioning a map across
carving passes is the same problem as versioning any large binary artifact: a real fleet operation
tags each published map with the session/timestamp that produced it, keeps the last N versions for
rollback (a carving bug that corrupts a map should be recoverable), and treats "publish a new map
version" as a deliberate, auditable event, not a silent overwrite — the map is what every robot's
localization trusts, and a bad map is a fleet-wide outage waiting to happen (§4 below).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-12. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

This project's compute is the carving pass itself; the hardware story is really "what would run
this at fleet scale," split by deployment pattern (§1):

| Tier | Compute | Fits | Illustrative cost (2026, verify current) |
|------|---------|------|-------------------------------------------|
| Hobby / research desktop | A discrete NVIDIA RTX-class GPU (this project's own dev machine: RTX 2080 SUPER, sm_75) in a workstation, batch carving offline | Post-session batch carving for a single robot's map, research/prototyping | $400-1500 for the GPU alone |
| Embedded / single-robot online | Jetson Orin-class SoC (e.g. Orin Nano/NX/AGX tier) onboard the robot, running incremental carving as scans arrive | A single AMR maintaining its own local map online | $200-2000 depending on Orin tier and carrier board |
| Fleet-scale map service | Rack-mounted or cloud GPU instances (e.g. NVIDIA L4/L40S-class or cloud A10/A100 instances), running many robots' sessions' carving in parallel, feeding a shared map store | A warehouse or site-wide fleet with a centralized map service (PRACTICE.md §3/§4) | Cloud GPU instance pricing, highly variable; or a few $5-15k on-prem GPU servers for a mid-size site |

**Sensors this project's input represents:** a real 16-32 beam mechanical spinning LiDAR (e.g. the
Velodyne/Ouster/Hesai/RoboSense product families) or a solid-state/MEMS unit with comparable
angular coverage — this project's 16-beam, 2°-azimuth beam model is a coarser stand-in for any of
these (README "Limitations"). No other sensor hardware is needed for the algorithm itself; a real
deployment ALSO needs the localization stack (wheel odometry + IMU + a SLAM front end) that supplies
each scan's pose, which is out of this project's scope (upstream, per README "System context").

**No actuation, no power electronics, no comms silicon specific to this project** — it is a pure
compute/data-structure workload; PRACTICE.md §3 covers where it runs relative to the robot's other
compute, not new hardware it introduces.

## 3. Installation & integration — putting it on a real robot

**Where this runs.** For the post-session batch pattern, off-robot: a fleet's mapping/IT
infrastructure, run after a mapping session uploads its scans. For the incremental pattern, on the
robot's main compute (the same Jetson-class or x86+dGPU box perception/mapping/planning already
share, `SYSTEM_DESIGN.md` §6.1) — carving is cheap enough (this project's whole 10-scan batch: ~13 ms
CPU, ~0.15 ms GPU kernel) that even continuous online carving is a small fraction of any realistic
20 Hz map-update budget (`SYSTEM_DESIGN.md` §1.1), leaving plenty of headroom for the DENSER,
production-scale grids a real site needs.

**OS / real-time constraints.** Not hard-real-time: this sits in the mapping/perception layer
(`SYSTEM_DESIGN.md` §1, 10-20 Hz territory at the tightest), comfortably served by a standard Linux +
CUDA stack (Ubuntu + JetPack on embedded, or plain Ubuntu + CUDA on a workstation/server) — nothing
here belongs on an MCU or in the safety chain (`SYSTEM_DESIGN.md` §6.1's "no GPU code in this repo
belongs on the safety chain" applies in full).

**ROS 2 node/topic shape.** A `dynamic_point_removal` node would subscribe to accumulated posed
scans (a `sensor_msgs/PointCloud2`-shaped topic per scan, paired with a `geometry_msgs/PoseStamped`
or a TF lookup for its pose — matching `SYSTEM_DESIGN.md` §3's message-shaped-struct convention this
project's own `PointCloud`/pose arrays mirror) and publish a CLEANED `PointCloud2` (or update an
occupancy-grid/octree topic in place, if wired into an OctoMap-style incremental system). The
upstream chain this node sits in, named concretely: raw scans → per-point motion deskew (project
02.08) → pose from localization (project 02.07's NDT scan matching, or any SLAM front end) →
accumulate into a session buffer → **this node carves and classifies** → publish the cleaned static
map (consumed by costmaps, project 23.01, and by fleet map-sharing, project 02.15's compressed
uplink).

**Map-update cadence policy.** A real fleet decides, as an operations policy (not an algorithm
question): how often does a site's reference map get RE-CARVED from fresh sessions? Options span
"nightly batch, during off-shift hours" (simplest, matches this project's batch design) through
"continuous incremental, every robot's every session contributes" (freshest, needs the online
architecture §1 describes) — the tradeoff is map freshness against compute/ops cost and the risk of
propagating a single session's carving error into the shared map every other robot trusts (§4 below
frames this as a cost center).

**The safe testing ladder.** This project computes no motion command and touches no actuator — its
output is a MAP, several steps upstream of any commanded motion (README "System context"). The
relevant caution is therefore not "will this move the robot" but "will a BAD map cause a downstream
planner to do something unsafe" — so the ladder that matters is: (1) simulation — this project's
own synthetic-scene VERIFY/GATE pipeline; (2) offline replay against LOGGED real sessions before
trusting a new map version in production; (3) shadow deployment — run the new carving pipeline
alongside the old map-generation process and diff results before cutover; (4) staged rollout — a
new map version published to a subset of a fleet before all robots switch. Everything here is
sim-validated only (CLAUDE.md §1); a corrupted map feeding a real robot's costmap is exactly the
kind of indirect-but-real safety path that testing ladder exists to catch.

## 4. Business & regulatory context

*Didactic orientation only — not procurement, legal, or compliance advice.*

**Who needs this.** Any fleet operator running LiDAR-based mapping and localization over time in an
environment with moving objects — which is essentially every commercial deployment: warehouse AMR
fleets (`SYSTEM_DESIGN.md` §2.1, forklifts and people share the floor), autonomous-vehicle mapping
(§2.5, HD maps must not encode parked cars as permanent obstacles), and any facility robot that
returns to the same space repeatedly. **Players:** map-quality tooling is largely built in-house by
AMR/AV companies (it is close to their core competency) or supplied as part of a broader SLAM/mapping
stack (e.g. the commercial and open-source mapping components of Nav2's ecosystem, vendor SLAM
stacks bundled with AMR platforms); the underlying academic techniques (OctoMap, Removert, ERASOR —
THEORY.md "Where this sits in the real world") are open-source and widely adopted as starting points
rather than shipped verbatim.

**What getting it wrong costs.** A map polluted with ghosts degrades localization accuracy (the
feedback loop README "System context" names — bad map → bad localization → worse map), causes
costmap-driven planners to treat long-gone obstacles as permanent (wasted routes, or conversely a
STALE map that fails to show a genuinely new permanent obstacle as one), and at fleet scale, a single
bad carving pass propagated to a shared map can degrade every robot that relies on it at once — this
is why **map freshness and correctness is a genuine fleet-operations cost center**
(`SYSTEM_DESIGN.md` §5.2 "fleet operations... lasts 5-10x longer than all other phases and usually
decides profitability"; §5.2 names "map/data pipelines at fleet scale" (project 02.15) explicitly as
this phase's work). Getting it wrong costs operational time (robots re-routing around ghosts, or
missing real new obstacles), not typically a safety incident directly — but a downstream planner
that trusts a stale/wrong map IS a contributing factor a real safety investigation would examine.

**Regulatory path.** This project produces no motion command and is not itself a safety function,
but it feeds systems that are: for a warehouse/service AMR, the relevant standard is **ISO 13482**
(service robot hazard analysis, `SYSTEM_DESIGN.md` §6.2) — a stale or ghost-polluted costmap that
causes a planner to route too close to a person is exactly the kind of hazard that standard's risk
assessment must consider; for an autonomous-vehicle HD map, **ISO 26262** / **UL 4600** govern the
broader safety case that a mapping pipeline's correctness is one input to (§6.2). This project itself
is an orientation example, not a certified component of any such system.

**Where this lives in a company.** Per `SYSTEM_DESIGN.md` §5.1, this work sits at the boundary of
the **perception** team (owns the LiDAR pipeline producing the scans this project consumes) and the
**controls & autonomy** team (owns SLAM/mapping, domains 04-05, and typically owns "the map" as a
data product); adjacent teams include **fleet operations** (who feel map-quality problems first, as
routing failures or re-localization events) and, at larger scale, a dedicated **maps/mapping
infrastructure** sub-team once a company's fleet is large enough to need one. Typical role titles:
mapping engineer, SLAM engineer, perception engineer, robotics software engineer (mapping/
localization focus).
