# 30.01 — Agriculture: fruit detection + 3D localization + ripeness; weed-vs-crop segmentation at frame rate; per-plant spray targeting; crop-row following; canopy volume from LiDAR; under-canopy navigation; yield mapping: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md section 4.3). It grounds the README "System context" section in the physical and
> commercial whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items
> 5-6. Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-10. All parts, prices, and vendor names below are **illustrative examples,
> never endorsements**; verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project's code has no moving parts of its own — it is a perception computation. Its physical
carrier is the **camera mount and enclosure** on whatever platform carries it (README "System context"
names two: a harvesting manipulator platform and a field AMR/spray robot), so this section describes
that mount and enclosure honestly rather than padding a construction narrative this code does not have.

- **The mount.** An RGB-D camera on an agricultural robot is typically rigidly mounted on a fixed or
  slow-pan bracket — for a harvesting arm, often WRIST- or FRAME-mounted (mirroring SYSTEM_DESIGN section
  2.2's manipulator work cell, "wrist- or frame-mounted stereo/structured-light 3D camera"); for a
  crop-scouting or spray robot, boom- or mast-mounted looking down/forward into the canopy. The mount
  must survive VIBRATION (uneven field terrain, a moving boom) without losing calibration — a loose
  camera mount silently corrupts every downstream 3-D position this project computes, since the whole
  pipeline trusts a FIXED `(fx,fy,cx,cy)` intrinsic model (`src/kernels.cuh`).
- **The enclosure.** Outdoor agricultural equipment lives in dust, water spray (from irrigation or the
  robot's own spray milestone), and temperature swings — an IP65/IP67-rated enclosure for the camera and
  its cabling is standard practice, and the enclosure's front window (glass or optical-grade acrylic)
  must be kept clean: dust or water droplets on that window corrupt hue/saturation exactly where this
  project's mask stage is most sensitive (THEORY.md "The algorithm").
- **What breaks in the field.** Camera cable connectors fatigue from vibration and repeated
  flex (booms and arms move); front windows scratch or fog; and — the field-specific failure this
  project's synthetic scene cannot model — direct sun glare or rain on the lens produces exactly the
  kind of spurious bright/saturated pixels this project's synthetic "glint" specks stand in for
  (THEORY.md "How we verify correctness"), except uncontrolled and far more frequent than six per frame.

## 2. Real hardware — chips, parts, illustrative BOM

What this project's INPUT (an RGB-D frame) would come from, and what the code would run on, tiered:

| Piece | Illustrative choices (2026) | Role in this project |
|---|---|---|
| Compute | Jetson Orin-class (on-robot, SWaP-constrained) / x86 + discrete RTX (reference machine: RTX 2080 SUPER, this project's actual measured numbers) | Runs the ~3-4 ms GPU pipeline per frame |
| RGB-D camera, hobby/research tier | Intel RealSense D400-series (structured-light/stereo IR), OAK-D (stereo + on-board neural inference) | ~US$200-600; indoor-leaning, workable outdoors in shade with tuning |
| RGB-D camera, industrial/outdoor tier | GLOBAL-SHUTTER machine-vision color camera (e.g., a Basler/FLIR GigE unit) genlocked with a separate ToF or structured-light depth head, or a stereo pair of global-shutter cameras computing depth on-device (this repo's own 01.02 SGM pipeline is exactly that computation) | ~US$1-5k+; the honest answer for direct sunlight and a moving platform — see the note below |
| Illumination | On-robot LED ring light or bar, often near-IR-supplemented for low-light/under-canopy work | Modeled by this project's ring-light shading assumption (THEORY.md "The math") |
| Mounting/enclosure hardware | IP65/67 machine-vision housings, vibration-damping camera mounts | Section 1 |

**Why GLOBAL SHUTTER matters here specifically, and why it is not a detail:** a ROLLING-shutter camera
(the kind in nearly every consumer webcam and phone) exposes its rows at slightly different times; on a
moving platform (a robot driving through an orchard row, or a fast-moving harvesting arm) this smears
and skews any object with relative motion — for a stationary CAMERA imaging FRUIT swaying gently in
wind it is a mild nuisance, but for a MOVING platform imaging a row it directly corrupts the pixel
positions this project's back-projection math trusts to be simultaneous (THEORY.md "The math" assumes
every pixel in a frame was captured at the same instant). Machine-vision outdoor cameras are
overwhelmingly global-shutter for exactly this reason, and it is the single most consequential camera
spec choice for this project's real-hardware counterpart — ahead of resolution or even depth accuracy.

**Direct sunlight is also a first-order engineering problem**, independent of shutter type:
structured-light depth sensors (which project their own IR pattern) are frequently WASHED OUT by
sunlight's own strong IR content, which is why the industrial tier above leans toward stereo-computed
depth (two ordinary cameras + a GPU SGM pipeline, no active IR projection to be overwhelmed) for
consistently reliable outdoor daylight operation — see 01.02 (this repo's stereo-depth flagship) for
the depth-computation half of that exact story.

## 3. Installation & integration — putting it on a real robot

- **Where this runs.** On the platform's main perception compute (the Jetson-class SoC or x86+dGPU box
  named in section 2), as a node subscribing the camera's `Image` topics (RGB + depth,
  SYSTEM_DESIGN section 3.6 message shape) and publishing a per-frame fruit-detection list — the ROS 2
  shape would be a custom message array (`FruitDetection[]`, mirroring this project's own
  `src/kernels.cuh` struct) or, if downstream consumers expect standard types, a
  `vision_msgs/Detection3DArray`.
- **Real-time constraints, honestly.** This is a PERCEPTION front end, not a control loop — the 30-60 Hz
  camera-rate budget (SYSTEM_DESIGN section 1.1) is soft, not hard: a missed or late frame means one
  stale detection list, not a robot that falls over (contrast with 08.01's control-loop caveat). The
  harvest-cycle argument in README "System context" gives this project's measured ~3-4 ms pipeline
  roughly two orders of magnitude of headroom against either the camera-rate or the harvest-cycle
  budget, so a simple "run once per incoming frame, publish the latest result" design is more than
  sufficient — no CUDA Graphs or persistent-kernel tricks (32.02's territory) are warranted here.
- **Calibration and bring-up.** Camera intrinsics (`fx,fy,cx,cy` — this project hardcodes the classic
  Kinect-v1 values as a TEACHING anchor; a real camera needs its OWN intrinsics from a standard
  checkerboard/AprilTag calibration, e.g. via ROS 2's `camera_calibration` package) and the camera-to-
  robot extrinsic transform (`T_base_camera`, SYSTEM_DESIGN section 3.3 notation) are the two
  calibration artifacts every downstream consumer of this project's 3-D output depends on; both drift
  with vibration and mount fatigue (section 1) and need periodic re-verification, not a one-time
  factory calibration assumed to hold forever.
- **The safe hardware-testing ladder — but stated honestly for what THIS milestone is.** Milestone 1
  (this project) is a PURE PERCEPTION COMPUTATION: its output is a printed/published list of 3-D
  points, not an actuator command. It therefore does not itself need the full sim -> HIL -> bench ->
  free-running ladder CLAUDE.md section 1 requires for anything that moves hardware — **but the very
  next milestone in this bundle that consumes its output, per-plant spray targeting (Milestone 3),
  absolutely does**, at full strength, the moment its output becomes a nozzle command: simulation first,
  then HIL against a simulated actuator, then a bench/tethered rig with the spray system's flow
  physically disabled or diverted, then free-running only inside a supervised test row — with an
  independent E-stop that does not trust this (or any) perception software. Stating this boundary
  explicitly is the point of this section: it is easy to conflate "the vision worked in the demo" with
  "the system is safe to actuate," and this repository's contract (CLAUDE.md section 1) is that those
  are never the same claim.
- **N/A here:** no fieldbus (CAN-FD/EtherCAT) is implemented in this project — its "output" is a CSV
  artifact and a printed detection list, not a command onto any bus. A real deployment's next milestone
  (spray targeting) would be the first to actually command hardware over one.

## 4. Business & regulatory context

- **Who needs this capability.** Fruit/produce growers (especially high-value, labor-intensive crops:
  apples, citrus, table grapes, berries) facing chronic seasonal labor shortages; agricultural
  equipment makers adding vision to existing sprayers/harvesters; and specialized agtech robotics
  startups building purpose-built harvesting or scouting platforms. Vision-based fruit detection is
  frequently one of the FIRST technical capabilities an agtech startup builds and demonstrates, because
  it is the capability the business case (labor replacement, yield visibility) most directly depends on
  proving — commonly preceding a full autonomy/navigation stack, unlike most other domains in this
  repository where perception is one layer among many built roughly in parallel.
- **The players.** Dedicated harvesting-robot companies (targeting specific high-value crops), precision-
  agriculture divisions of major equipment makers (adding vision/spray-targeting to existing sprayer and
  tractor platforms), and a wide academic/open literature (MinneApple, DeepFruits, and similar —
  README "Prior art") that most commercial systems build on or benchmark against. Build-vs-buy
  (SYSTEM_DESIGN section 5.3): the DETECTION MODEL (whether classical or learned) is very often the
  differentiator a company owns and iterates on continuously as it encounters new cultivars, lighting,
  and canopy structures in the field — closer to "always build" than "buy," even though the SURROUNDING
  camera/GPU/robotics stack may be substantially off-the-shelf.
- **What getting it wrong costs.** A missed or double-counted fruit costs YIELD ACCURACY (Milestone 7)
  or a wasted harvesting-arm cycle (Milestone 1 itself) — an economic cost, not a safety one, AS LONG AS
  the pipeline's output stays perception-only. The moment its output drives an ACTUATOR (Milestone 3,
  per-plant spray targeting; or a harvesting arm's motion), the cost model changes to the same one
  every actuation-adjacent project in this repository carries: damaged crop or equipment, and — because
  agricultural robots often share space with field workers — a genuine safety case, not just an
  economic one (section 3 above states this boundary explicitly).
- **Regulatory and adjacent considerations (orientation only — SYSTEM_DESIGN section 6.2's regulatory
  map does not name agriculture directly; the nearest rows and adjacent concerns):**
  - **Machine safety** for any powered field robot broadly follows the same machinery-directive/
    ISO-guarding family SYSTEM_DESIGN section 6.2 names for industrial and service robots, adapted by
    each jurisdiction's agricultural-equipment rules — an orientation point, not a specific standard
    citation, since this repository makes no certification claim (CLAUDE.md section 1).
  - **Food safety** is a genuinely agriculture-specific adjacency with no analog in most of this
    repository's other domains: any hardware that physically contacts or comes near harvested produce
    (a gripper, a conveyor, a spray nozzle) sits inside a food-safety compliance environment (e.g.,
    sanitary-design and food-contact-material expectations, and — for the spray-targeting milestone
    specifically — pesticide/agrichemical application regulations, which are jurisdiction-specific and
    entirely outside this repository's scope to advise on).
  - **Farm data.** A yield map (Milestone 7) or a per-plant health record is commercially and personally
    sensitive DATA about a specific grower's land and operation — who owns it, who can resell or
    aggregate it across growers, and under what terms, is an active, unsettled commercial question in
    agtech (echoing, in a different domain, the same "who owns the data" question SYSTEM_DESIGN section
    5's business orientation raises generally) — worth a learner's awareness, not something this
    repository takes a position on.
- **Owning team.** Perception (SYSTEM_DESIGN section 5.1) — commonly, in an agtech company, the very
  first dedicated engineering hire behind the founding team, because (as noted above) the detection
  capability itself is frequently the product's initial proof point; adjacent teams: ML/data (if the
  production system uses a learned detector instead of or alongside this project's classical pipeline),
  mechanical/electrical (the camera mount and enclosure, section 1-2), and — the moment any
  spray/harvest milestone is added — controls/autonomy and functional safety, per section 3's
  actuation boundary.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
