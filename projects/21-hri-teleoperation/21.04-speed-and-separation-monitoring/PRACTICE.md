# 21.04 — Speed-and-separation monitoring: depth streams → minimum-distance fields at frame rate (ISO/TS 15066 helper): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> **Didactic orientation only — this project's business/regulatory section (§4) is the load-bearing
> one here**, because the entire subject matter of this project is a safety function; §4 is written
> at more depth than §1–2 for exactly that reason.

## 1. Building it — construction of the robot/part

This project's physical carrier is **the collaborative work cell as a whole**, not a single part —
speed-and-separation monitoring is a cell-level function that watches the *relationship* between two
things (a person and a robot), not a component you bolt onto either one. What "building it" means in
practice:

- **The monitored volume.** A real cell has a defined floor footprint and a defined "protective
  zone" boundary (often marked on the floor, sometimes lit). This project's 4 m × 4 m cell and the
  camera's field of view are a miniature, honest stand-in for that footprint — small enough that a
  single overhead camera can plausibly cover it (real cells needing wider coverage tile multiple
  sensors, `PRACTICE.md` §2).
- **Mounting the sensor.** An overhead depth/vision sensor for SSM is typically mounted on a fixed
  truss or bracket above the cell, aimed straight down or at a shallow angle, with a **clear,
  unobstructed** view of the monitored floor area — any occlusion (a shelf, a WIP bin, the robot's
  own pose at full reach) is a blind spot the safety case must account for. This project's
  "robot self-filter" (THEORY.md "The algorithm") and its silhouette-visibility finding
  (THEORY.md "Numerical considerations") are a small, concrete taste of exactly this class of
  real installation problem: what the sensor *cannot* see is not a detail, it is the safety case.
- **Cabling and power.** A real sensor mount needs power and a data link (Ethernet/GigE for most
  industrial depth/vision safety sensors, or a dedicated safety-rated bus for certified scanners —
  §2) run along the truss to a junction box, then to the compute enclosure and the safety controller;
  strain relief and connector sealing matter because these mounts vibrate with the building and get
  bumped by material-handling traffic beneath them.
- **What breaks in the field:** camera mounts drift or get bumped out of calibration (silently
  shrinking the effective monitored zone — a serious failure mode any real installation must detect,
  e.g. via periodic recalibration checks or a certified sensor's built-in diagnostics), dust/debris
  accumulates on a lens or scanner window in industrial environments (degrading range and adding
  noise a real system must budget for in its own `Z_detection`-equivalent term), and cabling on a
  moving gantry or near a robot's reach envelope fatigues and fails exactly where the robot could most
  need the sensor working.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

The central hardware decision this project's whole subject matter turns on is **certified sensor vs.
general-purpose depth camera** — the two categories are not interchangeable, and the gap between them
is exactly the gap between this project (didactic) and a real SSM installation (certified):

| Category | Illustrative examples (2026) | Role | Certification status |
|---|---|---|---|
| Certified safety laser scanners | SICK microScan3 / nanoScan3 class, Pilz PSENscan class | 2D planar protective-field monitoring around a cell perimeter or a mobile base | Certified to relevant IEC/ISO functional-safety categories for the safety function itself |
| Certified 3D safety systems | SICK safety-rated 3D camera/LiDAR product lines, Pilz's 3D camera safety systems | Volumetric SSM-style monitoring of a cell (closer in *concept* to this project's overhead camera) | Certified, with documented diagnostic coverage and failure modes |
| General-purpose depth cameras | Structured-light / ToF / stereo modules (e.g. Intel RealSense class, Azure Kinect-class, industrial stereo modules) | Research, prototyping, non-safety perception (grasp planning, bin picking) — **this project's stand-in sensor** | **Not** safety-rated; using one to gate a real stop decision is exactly the mistake this project's `NOTICE:` line exists to prevent |
| Compute | Jetson Orin class (embedded) or an industrial PC + discrete GPU (reference machine here: RTX 2080 SUPER) | Runs the (non-certified) depth pipeline and SSM computation | Not safety-rated by itself; a real safety FUNCTION runs on hardware qualified for it, per the standard's requirements |
| Safety controller | A certified safety PLC / safety relay module (dual-channel, category-rated) | The layer that actually holds STO (safe torque off) authority | Certified — this is the layer this project's output would only ever *advise*, never replace |
| Comms | GigE/Ethernet for general sensor data; a certified safety bus (or hardwired dual-channel I/O) for anything that reaches the STO input | Data path from sensor(s) to compute, and separately, decision to safety controller | The safety-relevant path must itself be certified/rated, not just "a wire" |

The illustrative cost-tier story: a general-purpose depth camera + a small GPU box is a few hundred
to low thousands of dollars — cheap enough to prototype with, which is exactly this project's
posture. A certified safety-rated 3D monitoring system is a materially larger line item (often
multiples of that), *because* the certification, redundancy, and documented failure-mode analysis
are themselves the product, not the sensor silicon. That price gap is not friction to engineer away;
it is what "safety-rated" is buying.

## 3. Installation & integration — putting it on a real robot

- **Where this code would run:** the cell's perception/compute box (Jetson-class or industrial PC +
  GPU — SYSTEM_DESIGN.md §2.2's manipulator work-cell diagram), as an **advisory** ROS 2 node
  publishing a state topic (`NORMAL`/`REDUCED`/`PROTECTIVE_STOP`, this project's `SsmState`) at the
  camera's own rate. **It never itself asserts STO.** The advisory-vs-safety-rated split is the single
  most important integration fact about this whole project: SYSTEM_DESIGN.md §2.2's cell diagram draws
  this exact block (`HUMAN SAFETY [21 →] speed-and-separation monitoring of the cell`) feeding
  "slow/stop overrides" into joint control as a *separate arrow* from the certified hardwired safety
  chain (§6.1: E-stop buttons → safety relay → motor power cutoff/STO) — the two paths exist in
  parallel on a real cell, and only the second is allowed to be the thing that actually removes power.
  A non-certified advisory node like this one may *request* a slowdown or stop through the ordinary
  control path (and, on a well-designed cell, that request is honored quickly because it is usually
  right) — but the certified chain is what a safety case is allowed to *rely on*, because only it has
  the documented failure-mode analysis and independence a real hazard assessment requires.
- **Real-time constraints:** soft, not hard — this project's own render+classify+reduce pipeline
  measures well under 1 ms of GPU kernel time per synthetic frame, comfortably inside a 30 Hz (33 ms)
  camera period even before accounting for host-side overhead; a real deployment budgets the camera's
  own latency, network transfer, and the safety controller's own scan cycle on top of that, all of
  which THEORY.md's `T_r` term is meant to represent honestly (a monitor that reports its own distance
  correctly but too slowly has quietly enlarged the real-world stopping distance its formula assumes).
- **Calibration and bring-up:** extrinsic calibration of the sensor to the robot's base frame (so
  "the robot's known pose," this project's self-filter input, is actually where the sensor thinks it
  is), intrinsic calibration of the sensor itself, and — for a certified system — a documented
  commissioning and periodic re-validation procedure are all real, non-optional steps this project
  has no analog for (its "known pose" is exactly true by construction, THEORY.md flags this honestly).
- **The safe hardware-testing ladder (CLAUDE.md §1), applied to an SSM installation specifically:**
  1. *Simulation* — this demo: verify the distance/state logic against synthetic, fully-known geometry.
  2. *HIL* — the real depth pipeline, real camera, against a **mannequin or tracked target**, not a
     person, with the robot's motion either simulated or driven at safe, current-limited authority.
  3. *Bench, supervised, person present but the robot's stopping authority independently verified
     first* — confirm the certified safety chain (not this code) actually removes power when
     triggered, before ever trusting any advisory signal near a person.
  4. *Free running* — only once the certified chain has been independently validated and commissioned
     per the applicable standard's procedure, with a human safety expert signing off — never as a
     consequence of this project's code passing its own gates, which prove something narrower
     (THEORY.md "How we verify correctness") than "safe to run near people."
- **N/A here:** no real fieldbus, no real camera driver, and no real safety-controller interface are
  implemented in this project — main.cu's "actuator" is a printed state and a CSV row. Stated per
  contract (CLAUDE.md §4.3).

## 4. Business & regulatory context

**Who needs this capability.** Any manufacturer running collaborative or shared-space cells —
electronics assembly, machine tending, packaging/palletizing, warehouse pick cells with a human
present — needs *some* mechanism satisfying the "keep the person outside the robot's stopping
distance, dynamically" requirement whenever a fixed guard fence is undesirable (throughput, floor
space, or genuinely collaborative work). This capability is also directly relevant to mobile-robot
fleets (AMRs sharing aisles with people) and to service/field robots operating around untrained
bystanders, though the specific standard differs by robot type (below).

**The players.** Certified functional-safety sensor and system vendors (SICK, Pilz, and similar) are
the established commercial answer for industrial cells; robot OEMs increasingly offer integrated
"collaborative mode" options (power-and-force-limiting per-joint, sometimes combined with an SSM-style
external monitor) as a product feature; a smaller research/startup layer explores richer sensing
(multi-camera fusion, learned human tracking) aimed at eventually meeting certification, not at
replacing it. **Build-vs-buy is close to a non-question here**: the certified sensing and the safety
case are almost always *bought* (SYSTEM_DESIGN.md §5.3's build-vs-buy criteria point directly at "can
you debug it at 3 a.m." and "license/support horizon" — a company's own uncertified prototype code,
however good, is never the answer to "what removes power" on a real cell).

**What getting it wrong costs.** This is the sharpest instance of SYSTEM_DESIGN.md §5.3's "a
controller bug is the most expensive kind of software bug a robotics company has" for the whole
repository: a missed stop here is a person-injury event, with the attendant human cost, liability,
regulatory investigation, and (for the company) potential shutdown of the line or the whole
collaborative-operation program pending a fix. A false stop, by contrast, costs only availability
(SYSTEM_DESIGN.md §5.4's unit-economics frame: lost cycle time is a real but bounded cost) — which is
precisely why THEORY.md's false-stop/missed-stop asymmetry argument is not just a numerics footnote;
it is the commercial logic of the whole feature, stated in code.

**Regulatory: orientation, not compliance guidance** (SYSTEM_DESIGN.md §6.2's regulatory-landscape
table, this row specifically): **industrial arms/collaborative cells fall under ISO 10218** (robot and
system integration safety) **and ISO/TS 15066** (the collaborative-operation technical specification
that speed-and-separation monitoring, and this project's illustrative S_p formula, are structurally
inspired by). Both are risk-assessment-driven standards: they require a documented hazard analysis for
the *specific* cell and task, not a one-size-fits-all number, and SSM is one of several permitted
risk-reduction strategies alongside guarding and power-and-force limiting (choosing among them, or
combining them, is itself part of the required risk assessment). **Mobile robots sharing space with
untrained people** fall instead under **ISO 13482** (SYSTEM_DESIGN.md §6.2) — a different standard
with a different hazard model, worth knowing if this project's ideas get reused for an AMR-and-
pedestrian scenario rather than a fixed manipulator cell. Consult the licensed standard texts for
anything resembling real compliance work; this project (and this section) is orientation only.

**Where this work lives inside a robotics company** (SYSTEM_DESIGN.md §5.1's org map): **QA &
functional safety** owns this capability's safety case and its evidence trail — the same team that
owns catalog domain 31 (safety/verification) and HRI safety broadly. **Perception/HRI** engineering
typically owns the sensing pipeline this project's kernels are a teaching model of (the depth
rendering, classification, and distance computation), working *for* the safety case functional safety
defines rather than defining the safety requirement itself. Adjacent: **controls/autonomy** (owns the
joint-control path this project's state would advise into), **electrical/embedded** (owns the
hardwired safety chain and the safety controller's I/O), and **regulatory/compliance** (owns the
audit trail and the relationship with certification bodies). Typical titles: functional safety
engineer, safety systems engineer, perception engineer (HRI-focused), robotics software engineer.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
