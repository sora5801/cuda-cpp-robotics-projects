# 17.01 — Batched Lambert solvers + porkchop plot generation: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> This project's "physical carrier" is unusual for the repo: it is **mission-design software that
> never touches a robot directly** — its output is a number (a launch window and a delta-v budget)
> that other teams, months or years later, turn into hardware decisions. Sections 1–3 are honest
> about that indirection rather than inventing a tighter coupling than exists.

## 1. Building it — construction of the robot/part

This project has no moving parts of its own — it is math run on a ground computer, long before any
hardware exists to build. The honest physical carrier is **the spacecraft this trade study eventually
serves**, and the "construction" question that actually applies is: what does the mission-design
process, of which this project is one step, hand off to the people who *do* build hardware?

- **What this project's output constrains, physically.** A chosen (departure epoch, arrival epoch,
  delta-v budget) triple flows into the propulsion subsystem's sizing: total delta-v drives propellant
  mass (via the rocket equation), which drives tank volume, which drives spacecraft structural mass,
  which drives launch-vehicle selection — a chain of construction decisions that starts, in a very
  real sense, at the kind of grid search this project performs. Missing a launch window's low-delta-v
  region by a wide margin can mean tens to hundreds of kilograms of extra propellant mass, rippling
  through the entire spacecraft's structural and thermal design.
- **The ground segment that runs this code.** Unlike almost every other project in this repository,
  this one's "construction" is a **data center or engineering workstation** at a mission-design team's
  facility (JPL, ESOC, a NewSpace company's mission-design group) — server racks or desktop
  workstations with GPUs, not flight hardware. There is no sealing, no vibration qualification, no
  thermal-vacuum testing for this piece; it runs in an office, months to years before launch.
- **What breaks, in this domain, is a PROCESS failure, not a physical one:** a porkchop plot run with
  the wrong launch-vehicle performance curve, a stale ephemeris, or an un-communicated change to the
  delta-v budget between the mission-design team and the systems-engineering team that owns the
  spacecraft's mass budget — the "field failure" here is a coordination failure between teams, not a
  bolt shaking loose.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

Two genuinely different hardware contexts matter here, and conflating them is a common beginner
mistake this section exists to prevent:

| Context | Illustrative hardware (2026) | Role |
|---|---|---|
| **Ground segment — where THIS code runs** | Ordinary engineering workstations/servers: x86 + a discrete GPU (this project's own reference machine, an RTX 2080 SUPER, is entirely representative) or a cloud GPU instance | Mission-design trade studies, porkchop-plot generation, trajectory optimization — everything in this project and its natural successors (17.02+) |
| **Flight segment — where the CHOSEN trajectory is EXECUTED** | Radiation-hardened/tolerant flight computers: e.g. RAD750-class or newer rad-hard PowerPC/SPARC parts, or modern rad-tolerant options built around COTS silicon with error-correcting memory and watchdog architectures; single-digit-Watt power budgets, no GPU | The spacecraft's onboard guidance, navigation & control (GN&C) software that flies the trajectory this project's *class* of tool helped choose — a wildly different silicon world: radiation tolerance, power budget, and decades-long reliability dominate every choice, GPU-class throughput is neither available nor needed |
| **Deep Space Network (or ESA Estrack) ground stations** | Large parabolic dish antennas (NASA DSN's 34 m and 70 m class), ultra-low-noise receivers, atomic clocks for ranging | Tracks the spacecraft after launch, performs orbit determination (measuring where it *actually* is) that later trajectory-correction maneuvers are computed against — downstream of, and independent from, this project's pre-launch trade study |

The contrast matters pedagogically: this project's GPU-heavy batched Lambert solve is squarely
**ground-segment, pre-launch, offline** work (SYSTEM_DESIGN.md's "classical GPU territory" — mapping,
planning, and simulation, scaled from a robot's onboard compute up to a mission's entire design
process); nothing in this project's compute profile applies to the flight computer that would fly a
chosen trajectory, which lives in the rad-hardened, power-starved, real-time-but-not-GPU world
SYSTEM_DESIGN.md §6.1 sketches for actuation-adjacent hardware.

## 3. Installation & integration — putting it on a real robot

**N/A in the usual sense — this is the honest answer, not a placeholder.** This project's code never
runs on a spacecraft, never subscribes or publishes a ROS 2 topic, and never talks to a fieldbus; it
runs once (or iteratively, during trade studies) on a ground engineer's workstation, and its *output*
— a chosen launch window and delta-v budget — is a **document**, not a running process, that flows
into other teams' work:

- **Where the real workflow goes next.** The chosen departure/arrival epochs and delta-v budget become
  inputs to (a) a **trajectory design** team that refines the impulsive Lambert solution into a
  realistic, perturbation-corrected, potentially low-thrust trajectory (this repo's domain-17
  successors, e.g. a low-thrust optimizer), (b) **systems engineering**, who fold the delta-v number
  into the spacecraft's propellant and mass budget, and (c) **launch operations**, who lock in an
  actual launch date within the recommended window subject to launch-vehicle range availability.
- **The closest thing to "bring-up" here** is a mission-design **review process**: independent
  verification of the trade study (often by a second team running a different tool — GMAT vs. an
  in-house tool vs. this project's teaching implementation — and comparing results), sensitivity
  analysis (how much does the answer change if the launch date slips two weeks?), and margin analysis
  (how much delta-v headroom is kept for trajectory-correction maneuvers after real navigation errors
  are discovered post-launch?). There is no simulation -> HIL -> bench -> free-running ladder here
  (CLAUDE.md §1) because there is no actuation in this project's scope at all — the "testing" that
  matters is numerical verification (THEORY.md §how-we-verify) and independent cross-checking against
  other tools, not a hardware bring-up ladder.
- **Once a spacecraft is flying**, the ACTUAL trajectory-correction maneuvers that keep it on the
  planned path are computed by an onboard or ground-based navigation team using real tracking data
  (Deep Space Network ranging and Doppler) — a completely different, closed-loop, safety-relevant
  software system this project's open-loop, pre-launch trade study does not touch.

## 4. Business & regulatory context

- **Who needs this capability.** Every organization that flies anything beyond Earth orbit: national
  space agencies (NASA, ESA, JAXA, ISRO, and peers), and an increasingly capable NewSpace commercial
  sector (interplanetary and cislunar mission companies, lunar landers, deep-space smallsat
  constellations) that increasingly builds or licenses its own mission-design tooling rather than
  relying solely on legacy government software.
- **The players.** GMAT (NASA/AFRL, free and open source) and STK/Astrogator (commercial, AGI/Ansys)
  are the two most widely used professional tools; poliastro and pykep are the open-source
  Python/C++ astrodynamics-library layer increasingly used for rapid trade studies and teaching (this
  project's spiritual open-source neighbors); most flight-proven missions still rely on
  institutionally-validated in-house tools for the final, certified trajectory design, with open tools
  used for early trade studies and cross-checks.
- **What getting it wrong costs.** A launch-window trade study is a MASS-BUDGET decision made
  extremely early, when it is cheap to change and catastrophically expensive to have gotten wrong: a
  delta-v budget set too optimistically (say, from a bug in a Lambert solver that under-reports
  delta-v — precisely the class of bug THEORY.md's three independent verification checks exist to
  catch) can force a late redesign, a missed launch window (costing a full synodic period — for Mars,
  about 26 months — of schedule slip), or a spacecraft that arrives with insufficient propellant
  margin for orbit insertion. This is a "getting it wrong costs mission success, not just money" domain.
- **Regulatory: export control is the standard here, not safety certification.** Space technology sits
  in the one row of SYSTEM_DESIGN.md §6.2's regulatory table that is about *information control* more
  than physical safety: **ITAR** (International Traffic in Arms Regulations, US) and **EAR** (Export
  Administration Regulations, US) — and equivalent regimes in other spacefaring nations — restrict
  sharing certain spacecraft and launch-vehicle technical data, including some trajectory-design and
  guidance software, across national borders, and reach individual engineers directly (a US engineer
  discussing certain trajectory details with a non-US colleague can itself be a regulated "deemed
  export"). This is genuinely different in *character* from the safety-certification rows
  (ISO 10218, ISO 26262, etc.) that dominate this repo's other domains: it is about who is allowed to
  KNOW something, not about whether a machine is safe to be near. **This is orientation only, not legal
  guidance** — real space-technology work requires actual export-control compliance review by
  qualified counsel, not a repository README.
- **Owning team.** Mission design / flight dynamics (titles: mission design engineer, flight dynamics
  engineer, astrodynamicist), inside the broader GN&C/controls-and-autonomy organization
  (SYSTEM_DESIGN.md item 5); tightly adjacent to systems engineering (owns the mass/delta-v budget this
  project's output feeds), propulsion (owns the hardware that must deliver the computed delta-v), and
  mission operations (who eventually executes the chosen trajectory and manages the export-control
  boundary on international collaborations).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
