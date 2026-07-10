# 27.04 — Composite layup optimization + Tsai-Wu failure envelope sweeps: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's output — a ranked stacking sequence — is a **manufacturing instruction**, so its
physical carrier is the laminate itself, built by hand or machine layer by layer.

- **Prepreg layup.** The dominant aerospace-grade process: each ply arrives as **prepreg** tape or
  fabric — fibers pre-impregnated with a precisely metered, partially-cured ("B-stage") epoxy resin,
  stored frozen (typically below -18°C) to arrest cure until use. A technician (or an automated
  tape-laying / fiber-placement machine on production parts) cuts each ply to shape and places it on
  a tool at the EXACT angle this project's sweep chose, in the EXACT stacking order — a layup
  travdeler (paper or digital) specifies angle, ply number, and orientation tolerance (typically
  `+-2` degrees) for every single ply, because a misplaced ply angle silently invalidates the whole
  CLT calculation this project performs.
- **Debulk.** Every few plies, the stack is vacuum-bagged and pressed briefly (a "debulk") to squeeze
  out trapped air and consolidate the stack before adding more plies — skipping debulks on a thick
  layup is a common source of internal voids.
- **Cure.** The completed, vacuum-bagged stack cures in an **autoclave** (heat + pressure, typically
  120-180°C and several atmospheres, following the resin manufacturer's specified cure cycle) for
  aerospace-grade parts, or in a simple heated oven under vacuum-bag pressure only for lower-
  performance / cost-sensitive parts (a real strength/cost trade: autoclave cure gives lower void
  content and better mechanical properties). Cure locks the resin permanently — there is no
  "re-laying-up" a cured part.
- **What breaks in the field:** delamination (plies separating, often starting from an edge, a
  ply-drop, or an impact — this project's pure-membrane-load, first-ply-failure analysis does not
  predict delamination at all, a real and separate failure mode), moisture ingress degrading the
  resin over years, UV/thermal degradation of exposed resin, and — the failure mode most directly
  tied to THIS project's assumptions — a layup error (wrong angle, wrong ply count, wrong stacking
  order) that makes the as-built part's real `A` matrix silently different from the one this sweep
  scored. This is exactly why coupon testing (§3) exists: to catch the gap between "the layup
  traveler said" and "what was actually built."

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts/processes named below are **illustrative examples, never
endorsements**; part numbers, process names, and prices go stale — verify current before relying on
any of them.*

This project's compute is a one-shot, offline design calculation — the "hardware" question that
matters more than the GPU it ran on is the **manufacturing hardware** the winning layup travels to:

| Piece | Illustrative choices (2026) | Role |
|---|---|---|
| Design compute | Any desktop with a CUDA GPU (reference machine: RTX 2080 SUPER) — this sweep needs no special hardware at all | Runs the ranking + envelope sweep interactively |
| Cutting | Ply cutter (laser or reciprocating-knife CNC cutting table) for prepreg ply shapes | Turns the layup traveler's ply outlines into physical material |
| Layup | Hand layup (hobby/low-rate) through Automated Tape Laying (ATL) / Automated Fiber Placement (AFP) machines (production-rate aerospace) | Places each ply at its specified angle and position |
| Cure | Vacuum-bag + oven (hobby/research) through autoclave (aerospace-grade, ~US$100k-multi-million capital equipment) | Consolidates and cures the resin |
| Inspection | Visual/dye-penetrant (hobby); ultrasonic C-scan or X-ray CT (research/industrial) for internal void/delamination detection | Confirms the built part matches the analysis's assumptions |
| Test | Universal testing machine (tension/compression coupon rig, research-grade ~US$10k-100k+) | Physically validates the strengths this project's `data/sample/` treats as known inputs |

The GPU compute enclosure story (where the design-time sweep itself runs) is 33.01 PRACTICE §1's,
unchanged — this project needs nothing beyond a desktop.

## 3. Installation & integration — putting it on a real robot

This project's output never runs ON a robot — it is consumed by the people and processes that BUILD
one, so "integration" here means integration into an **engineering workflow**, not a runtime stack.

- **Where the loads actually come from.** In a real design chain, `Nx/Ny/Nxy` do not appear from
  nowhere: an upstream structural analysis (a topology-optimization study like **26.01**'s, a flight-
  loads envelope, a hand-calculated worst-case bending moment at a bracket's mounting flange) hands
  this project its load cases as data (README §System context names 26.01 by name as the canonical
  upstream). Garbage load cases in, garbage-ranked layup out — this project's honesty depends
  entirely on the fidelity of whatever produced its `LOAD_MIXED`/`LOAD_ALIGNED` rows.
- **How a layup spec enters a real BOM.** The winning layup (angles, ply count, material callout)
  becomes a line item in the part's manufacturing definition — typically a **ply book** or **layup
  traveler** referencing the exact prepreg material spec (e.g. a specific resin/fiber system's
  qualified allowables, NOT this project's synthetic numbers), the cutting pattern for each ply, and
  the stacking order. That document, not this project's console output, is what a technician or an
  ATL/AFP machine actually follows.
- **Coupon testing — the mandatory bridge from analysis to trust.** No composite structural
  allowable is used in a certified part on the strength of an analysis alone. The real qualification
  ladder: (1) **material-level coupon tests** (simple tension/compression/shear specimens establish
  the `Xt/Xc/Yt/Yc/S12` this project treats as *given inputs* — in a real program these numbers come
  from hundreds of physical tests, statistically reduced to a conservative "B-basis" or "A-basis"
  allowable, not a single measurement), (2) **element-level tests** (small representative laminate
  coupons — exactly this project's kind of layup — loaded to failure to validate the CLT+failure-
  criterion PREDICTION against reality), (3) **sub-component** and (4) **full-scale** structural
  tests. This is the composites-world analogue of CLAUDE.md's sim-to-real testing ladder
  (simulation -> HIL -> bench -> free-running) — a design like this project's sweep NEVER skips
  straight from a spreadsheet/GPU calculation to a flying part.
- **N/A here:** no ROS 2 node, no fieldbus, no real-time constraint applies — this is an offline
  design tool with no runtime footprint on any robot (README §System context states this rate/latency
  honesty directly).

## 4. Business & regulatory context

- **Who needs this capability.** Any robotics program building lightweight structure under known
  loads: drone-airframe manufacturers (spars, skins, arms — the dominant weight-driver on a
  multirotor or fixed-wing UAV), legged-robot limb designers chasing low reflected inertia,
  manipulator-arm builders wanting stiff-but-light forearm links, and — well beyond robotics — the
  aerospace, automotive, and sporting-goods industries that collectively created the tools this
  project's "Prior art" section names.
- **The players.** Commercial structural-sizing tools (HyperSizer, and composite-specific modules of
  ANSYS/Altair/Siemens' broader FEA suites) dominate certified aerospace work; open-source and
  academic tools (e.g. CLT implementations inside general FEA/composites-teaching packages) serve
  research and smaller programs. Build-vs-buy here usually favors BUY for anything safety-critical
  (the certification paper trail behind a commercial tool's validated methods is itself valuable) and
  BUILD/hand-calculation for early-stage, low-volume, or research-stage designs — exactly the niche a
  quick GPU sweep like this project's fills well.
- **What getting it wrong costs.** A composite structural failure is not a "patch and redeploy"
  software bug — it is a physical part that cracks, delaminates, or breaks, potentially in flight or
  under load with a person nearby. The cost spans scrapped material and rework (cheap, if caught at
  the coupon-test stage) through grounded fleets and liability (expensive, if caught after fielding).
  The entire qualification ladder in §3 exists because the cost of being wrong rises steeply the
  later it is discovered.
- **Regulatory orientation (SYSTEM_DESIGN item 6's map, cited, not repeated).** For aerospace
  structures, the relevant framework is airworthiness certification (e.g. FAA/EASA structural
  substantiation requirements, and the composite-specific guidance material — e.g. FAA AC 20-107 —
  that governs how allowables must be derived and how much physical testing is required before an
  analysis like this project's can be trusted for a certified part). For drones specifically,
  SYSTEM_DESIGN's FAA Part 107 / EASA references govern the VEHICLE's operation, not this project's
  structural analysis directly, but a structural failure is exactly the kind of event those
  operational rules exist to prevent the consequences of. This is an **orientation map**, not
  compliance guidance — a real certified composite part's substantiation package is prepared by
  qualified structures engineers against the specific applicable standard, never by running this
  teaching sweep.
- **Owning team:** mechanical/structures engineering — typically a **structures engineer** or
  **composites engineer** role, reporting into or adjacent to the broader mechanical-design team
  (SYSTEM_DESIGN item 5); immediate neighbors are the topology-optimization/FEA team (26.x, upstream
  loads), manufacturing engineering (27.x siblings, downstream process), and materials/quality
  engineering (owns the coupon-test program that validates every number this project treats as an
  input).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
