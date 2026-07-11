# 01.14 — Template matching (NCC) at scale for pick verification: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is a **pick-verification vision station** bolted onto (or immediately
downstream of) a pick-and-place cell — continuing project 01.13's "industrial vision station" framing,
specialized here for the tray/slot geometry this project's demo checks.

- **Camera mounting, directly above the tray.** A fixed overhead camera (not a moving one — repeatability
  of the camera-to-tray geometry is what makes the `+-8` px search window valid at all) mounts on a rigid
  gantry or stand above the tray's staging position, aimed straight down. Any deviation from a true
  overhead nadir view introduces perspective foreshortening that this project's 2-D-only NCC matching
  does not model (README "Limitations & honesty") — the mechanical alignment tolerance of the camera
  mount is therefore a real, if easy to overlook, accuracy input, not just a convenience.
- **Diffuse dome lighting — chosen BECAUSE of, not despite, this project's illumination story.** A
  diffuse dome or large-area diffuse panel light (rather than a single point-source ring light) spreads
  illumination evenly across the whole tray, minimizing the sharp local shadows that `THEORY.md`
  "Numerical considerations" notes NCC does NOT fully compensate for (only a global-ish affine
  brightness change is provably invariant). This is a direct, physical answer to a numerical limitation:
  choosing lighting hardware that keeps the real illumination change closer to the affine regime NCC is
  built to handle, rather than trying to fix it in software after the fact.
- **The tray's own fixturing and repeatability.** Each slot is a molded or machined POCKET (or, for
  softer parts, a foam/rubber nest) with a known nominal position — the tray itself is the mechanical
  reason a `+-8` px search window is sufficient instead of a full-image search: the pocket geometry
  bounds how far off-center a placed part can land. Tray-to-camera repeatability (the tray always
  re-docks at the same station position, via locating pins or a kinematic mount) is what keeps that
  bound valid run after run; a tray that is not consistently re-positioned would need either a larger
  search window or a tray-registration step before per-slot matching (an extension of project 01.13's
  own alignment-solve idea, applied to the tray rather than the part).
- **What breaks in the field.** Dome-light LEDs age and dim unevenly (slowly re-introducing the kind of
  illumination gradient this project's `shadow` cohort models deliberately); tray pockets wear or chip
  over thousands of cycles, loosening the placement tolerance the search window assumes; camera-to-tray
  standoff drifts if the gantry is bumped, changing the effective pixels-per-mm scale this project does
  not calibrate for (see §3).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute tier.** This project's ENTIRE measured workload (104,040 NCC evaluations, all 3 GPU variants)
runs in under a millisecond on a desktop RTX 2080 SUPER — pick verification at this scale is not
compute-bound on any modern tier:

| Tier | Example | Notes |
|------|---------|-------|
| Embedded/edge | NVIDIA Jetson Orin Nano/NX | Comfortably real-time even at industrial-camera (multi-MP) resolutions with a larger tray; the common choice for a smart-camera-style verification station. |
| Desktop/industrial PC | Small-form-factor PC + a discrete RTX-class GPU, or even a modern integrated GPU at this problem size | This project's own development target. |
| MCU-class (no GPU) | A dedicated smart-camera SoC (Cognex In-Sight, Keyence CV-X class) | At tray-image scale, template matching is well within the budget of fixed-function/DSP smart-camera silicon — no separate PC needed for many real installations. |

**Camera & sensor.** A global-shutter machine-vision camera (Basler ace / FLIR Blackfly S class or
similar), 1-5 MP monochrome — color is rarely needed for pure silhouette matching and costs
contrast/resolution for no benefit here — GigE Vision or USB3 Vision interface, matching project 01.13's
own camera framing (continuity: the same station class serves both alignment and verification tasks).

**Lighting.** A diffuse dome LED illuminator (see §1) or a large diffuse area light, 24 V DC, through a
dedicated LED lighting controller with a strobe output (see §3) — the deliberate choice over a harsher
ring or bar light, for the reason §1 explains.

**Rough cost tiers (illustrative, 2026, verify current):** hobby (USB camera + diffuse LED panel): under
$250. Research/prototype (machine-vision camera + C-mount lens + dome light + a Jetson): roughly
$1,000-$2,500. Industrial (certified camera, calibrated dome lighting controller, smart-camera or
industrial PC, integrated with a PLC): $3,000-$12,000+ per station, before integration labor — a similar
range to project 01.13's alignment station, since the hardware bill of materials is largely shared.

## 3. Installation & integration — putting it on a real robot

**Where this code runs.** Same placement as project 01.13's station: a small industrial PC or
Jetson-class module near the camera, physically and electrically separated from the robot's own
controller.

**ROS 2 shape.** One node subscribing to `sensor_msgs/Image` (or this repo's own `Image` message-shape,
`docs/SYSTEM_DESIGN.md` item 3) triggered once per pick-and-place cycle, publishing a per-slot verdict
array (this project's own `SlotResult`-shaped message: verdict, recovered offset/rotation, score) on a
topic like `/tray_verification`.

**GigE trigger chain — synchronized to the ROBOT, not free-running.** The camera is HARDWARE-TRIGGERED
by the cell's PLC (or directly by the robot controller's cycle-complete signal) exactly once per tray
cycle, after the pick/place motion has fully settled — verifying mid-motion would see motion blur and a
meaningless result. The same TTL/opto-isolated trigger + strobe wiring pattern as project 01.13 §3
applies unchanged.

**PLC handshake + reject lane.** The vision result is written back to the PLC over EtherCAT/PROFINET or
plain digital I/O within the cycle's time budget (README "Rate / latency budget"). A REJECT verdict
(any `WRONG_PART` or `EMPTY` slot) typically routes the tray to a REJECT LANE or triggers a re-pick
attempt rather than letting a bad tray continue downstream — the physical consequence of this project's
`classification` gate, made real.

**Template management — the underrated ops problem.** A production verification station's biggest
long-run maintenance burden is rarely the matching ALGORITHM — it is keeping the TEMPLATE SET correct
as parts, tooling, and lighting drift over months of operation:

- **Versioning.** Every template needs a version, a capture date, the exact lighting/camera settings it
  was captured under, and a record of which PART REVISION it represents — swapping in a new part
  revision without updating its template is a classic silent-failure mode (the station keeps "passing"
  parts against a stale reference, or starts rejecting good parts against an obsolete one).
- **Re-teach procedure.** A documented, repeatable capture procedure (same lighting rig, same camera
  settings, ideally an averaged/denoised capture rather than one frame) so a new template is a fair
  comparison basis, not accidentally noisier or differently lit than the live tray images it will be
  matched against.
- **Drift monitoring.** Logging the SCORE DISTRIBUTION over time (not just pass/fail) — the same measured
  discipline this project's own `per_slot_scores.csv` artifact demonstrates at unit-test scale — lets an
  operator notice a slow illumination or tooling drift (scores trending down) before it crosses the
  `T_OK` threshold and starts producing false rejects.
- **Threshold governance.** `T_OK` (this project's classification threshold) is itself a tuned parameter,
  not a law of physics — changing it trades false-accept risk against false-reject risk (see §4). A real
  station documents WHO owns that threshold and under what change-control process it may be adjusted.

**The safe hardware-testing ladder (CLAUDE.md §1 — sim-validated only, not safety-certified).** Same 4
rungs as project 01.13 §3: (1) simulation (this project's own demo, on synthetic data with known ground
truth); (2) HIL (replay real or synthetic camera frames through the actual vision PC and PLC with the
robot disconnected/simulated); (3) bench jig / tethered / current-limited (verify a REJECT verdict
actually diverts a real tray on a slow, limited-motion bench setup); (4) free running, only after (1)-(3)
pass repeatedly. This repository provides none of the safety interlocks a real installation requires.

## 4. Business & regulatory context

*Didactic orientation only — **not** procurement, legal, or compliance advice.*

**Who needs this.** Any assembly or kitting operation using pick-and-place robots or human-robot
collaborative picking: automotive/electronics assembly (verifying the right connector or fastener was
placed), kit assembly and packaging (checking a multi-part kit tray is complete and correct before it
ships downstream), and pharmaceutical/medical-device packaging (verifying blister-pack or tray contents
— continuing project 01.13's "machine vision as the workhorse of manufacturing QA" framing).

**Commercial and open-source players.** The same landscape as project 01.13 §4: **Cognex**, **Keyence**,
and **MVTec Halcon** on the commercial side (many with dedicated "pattern find" or "PatMax" tools that
are the production-grade descendants of this project's NCC search); **OpenCV** on the open-source side
(`cv::matchTemplate`, the library call this project's kernels teach toward).

**The cost of getting it wrong — escapes vs. false rejects, a genuine trade-off.** Every classification
threshold (this project's `T_OK`) sits on a curve between two costs that pull in opposite directions:

- **A missed escape** (a genuinely wrong or missing part classified `OK`) ships a defective kit or
  assembly downstream — at best a warranty/rework cost, at worst (in regulated industries) a formal
  quality-escape investigation and, in safety-relevant assemblies, a real hazard.
- **A false reject** (a genuinely correct pick classified `WRONG_PART`/`EMPTY`) stops or slows the line
  for no real defect, or scraps a good part — a pure throughput/cost loss with no safety benefit.

Raising `T_OK` reduces escapes but increases false rejects, and vice versa — exactly the trade-off this
project's own threshold choice makes concretely (README/`THEORY.md` show the real measured score gap
`T_OK=0.65` sits inside). A real station tunes this threshold against MEASURED score distributions from
production data and the line's own economics (the cost of one escape vs. the cost of one false reject),
not a single demo run's numbers.

**Gauge R&R — the acceptance discipline (orientation only, continuing project 01.13 §4).** Before a
verification station is trusted in production, it undergoes the same Gauge R&R study project 01.13
describes: repeated measurement of known-good and known-bad reference trays, decomposed into
repeatability and reproducibility, checked against the line's tolerance for both escape and false-reject
rates — this project's own `classification` gate (24/24 correct on one synthetic tray) is a toy,
single-run version of exactly that idea.

**Regulatory path (orientation, citing `docs/SYSTEM_DESIGN.md` item 6).** As in project 01.13, the
vision station itself is not typically directly regulated, but the ROBOT/CELL it gates is — **ISO
10218**/**ISO/TS 15066** for an industrial or collaborative arm acting on a REJECT verdict. In regulated
verticals (automotive, medical device, aerospace), the station's own measurement performance (its
escape/false-reject rates) may need to be traceable under the industry's quality system (e.g., IATF
16949, or FDA-adjacent quality-system requirements for medical-device packaging) — orientation, not a
compliance checklist.

**Where this work lives inside a robotics/manufacturing company.** The same **machine vision
engineering** / **manufacturing automation engineering** team project 01.13 names (titles: Machine
Vision Engineer, Automation Engineer), adjacent to **quality engineering** (who own the Gauge R&R
process and the escape/false-reject economics above) and **manufacturing/process engineering** (who own
the tray fixture tolerances this project's search window assumes) — see `docs/SYSTEM_DESIGN.md` item 5
for the fuller org map.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
