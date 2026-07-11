# 01.06 — AprilTag / ArUco GPU detector-decoder for high-rate fiducial localization: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5-6.
>
> *Sections dated 2026-07-10. All parts, prices, and vendor names below are **illustrative examples,
> never endorsements**; verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

Unlike most projects in this repository, this one's PHYSICAL CARRIER is not a subsystem of the robot
at all — it is a small manufactured PART: the printed tag itself, plus the camera that reads it.

- **Fabricating a tag.** A fiducial tag is printed on a laser or high-DPI inkjet printer (a real
  minimum resolution matters — THEORY.md's cell-size math means a `0.16 m` tag with 6 cells per side
  needs each cell resolved cleanly at print time; a home inkjet at 300+ DPI is comfortably sufficient,
  a low-resolution thermal label printer is not) on MATTE paper or matte photo stock — THEORY.md "The
  problem" explains why matte (Lambertian), not glossy, media is load-bearing, not a preference: a
  glossy tag develops a viewing-angle-dependent specular highlight that can wash out part of the
  pattern from some approach angles, exactly the failure this project's pipeline has no way to
  compensate for.
- **Mounting and lamination.** A printed tag flexes and fades in the field; production tags are
  typically laminated (matte laminate — glossy lamination reintroduces the specular problem above) and
  mounted RIGIDLY and FLAT on a stiff backing (foam-core, acrylic, or aluminum composite panel) — this
  project's whole pipeline assumes the tag is PLANAR (THEORY.md "The math": `Z=0` for every tag-frame
  point); a warped or bowed tag violates that assumption and biases the recovered pose in a way no
  amount of image processing can detect or correct.
- **Size/distance design rule.** THEORY.md's projected-pixel-size formula, `side_px = fx * tagSize /
  depth`, is the design tool: for a target minimum readable size (this project's own committed scenes
  never render a tag below ~56 px, empirically the floor for reliable 6x6-cell decoding at this
  pipeline's noise/blur levels) and a known camera `fx` and expected working distance, solve for the
  MINIMUM physical tag size — e.g., at `fx=350 px` (this project's synthetic camera) and a 3 m working
  distance, a reliably-readable tag needs `tagSize >= 56*3/350 ~= 0.48 m` — noticeably larger than a
  desk-distance tag. Getting this wrong (printing a tag sized for the wrong working distance) is the
  single most common real-world fiducial deployment mistake.
- **What breaks in the field:** ink fading/UV bleaching (outdoor mounted tags), physical damage
  (scuffing, tearing, a forklift clipping a warehouse-floor tag), dirt/dust accumulation reducing
  contrast (directly attacks the adaptive-threshold margin THEORY.md's "mean minus bias" rule depends
  on), and condensation or glare under certain lighting — all of which this project's synthetic noise
  model only loosely approximates (README "Limitations & honesty").

## 2. Real hardware — chips, parts, illustrative BOM

| Piece | Illustrative choices (2026) | Role in this project |
|---|---|---|
| Compute | Jetson Orin-class (on-robot, embedded) / x86 + discrete RTX (reference machine: RTX 2080 SUPER, this project's actual measured numbers) | Runs the detector-decoder pipeline; the whole point of "high-rate" in this project's title is that this stays comfortably real-time even on an embedded SoC |
| Camera, hobby/research tier | A USB global-shutter machine-vision camera (e.g., an OV9281-based board, ~US$30-80) or a Raspberry Pi HQ Camera with a global-shutter sensor variant | Enough for desk/lab-distance tag work; rolling-shutter consumer webcams work too at LOW relative motion, with the same caveat as below |
| Camera, industrial tier | A genlocked global-shutter GigE/USB3 machine-vision camera (e.g., a Basler/FLIR ace-class unit), often paired with a fixed-focus lens chosen for the working-distance/tag-size design rule in section 1 | The honest answer once the platform or the tag itself moves at meaningful speed (docking, handheld operator use) |
| Tag substrate | Matte photo paper / laminate on rigid backing (foam-core, acrylic, aluminum composite) — see section 1 | The physical target itself |
| Mounting hardware | Rigid brackets, VHB tape or screws — must keep the tag FLAT (section 1) | Keeps the planarity assumption valid |

**Why GLOBAL SHUTTER matters here specifically:** a rolling-shutter camera exposes its rows at
slightly different times; on a platform or tag in relative motion (a docking AMR, a handheld
teleoperation camera, a tag on a moving part) this skews the tag's apparent shape row-by-row —
directly corrupting the corner positions this project's DLT solve trusts to all belong to the SAME
instant (THEORY.md "The math" implicitly assumes simultaneous capture of all 4 corners). For a
STATIC scene it is a non-issue; the moment either the camera or the tag moves at appreciable speed
relative to the frame period, global shutter becomes the single most consequential camera spec choice
for this project's real-hardware counterpart — the same conclusion 30.01's PRACTICE.md reaches for a
different reason (motion blur vs. row-skew), worth noticing as a recurring theme across this repo's
vision projects.

## 3. Installation & integration — putting it on a real robot

- **Where this runs.** On the platform's perception compute, as a node subscribing a RECTIFIED camera
  `Image` topic (SYSTEM_DESIGN section 3.3 — this project's own README "System context" names 01.01's
  image pipeline as the natural upstream) and publishing one pose per detected tag. The real,
  widely-deployed ROS 2 shape is `apriltag_ros`'s: a node that publishes
  `apriltag_msgs/AprilTagDetectionArray` and/or a `geometry_msgs/PoseStamped` per tag directly into
  `tf2`, as `T_camera_tag` (this repo's stated transform-naming convention, SYSTEM_DESIGN section 3.3)
  — exactly the `(R, t)` this project's `Detection` struct already computes.
- **Real-time constraints.** Unlike 30.01's fruit-detection perception (a soft, camera-rate budget),
  fiducial localization commonly feeds a CLOSED docking or servo loop — README "System context" names
  30-60+ Hz as the target and explains why: a stale tag pose shows up immediately as control error, not
  just a late report. In practice this means running the detector on every incoming frame at the
  camera's native rate, not batching or throttling it the way a slower-changing perception output
  might be.
- **Calibration and bring-up.** Two calibration artifacts, both prerequisites this project's own
  synthetic scenes bypass by construction (they are GENERATED with known-exact intrinsics): (1) camera
  INTRINSICS (`fx,fy,cx,cy` plus lens distortion — a standard checkerboard or, recursively, AprilTag-
  based calibration, e.g. ROS 2's `camera_calibration` package; 01.01's Brown-Conrady model is this
  repo's own worked example of what that distortion correction looks like), required BEFORE this
  project's pose math is meaningful at all (THEORY.md "The math" assumes `K` is known exactly); and (2)
  the camera-to-robot extrinsic `T_base_camera`, needed to turn a `T_camera_tag` pose into a
  robot-frame or world-frame number any planner or controller can use.
- **The safe hardware-testing ladder.** This project's OWN output is a computed pose — a perception
  computation, not an actuator command, so it does not itself require the sim -> HIL -> bench ->
  free-running ladder CLAUDE.md §1 mandates for anything that moves hardware. The moment a docking
  controller, teleoperation arm, or any actuator closes a loop AROUND this project's output, that full
  ladder applies at full strength to THAT consuming system: simulation first (verify the docking
  controller's response to synthetic tag-pose sequences, including this project's own known failure
  modes — a missed detection, an occasional large pose error near the +/-45-degree corner-extraction
  weakness), then HIL, then a tethered/current-limited bench test, then free-running only with an
  independent E-stop that does not trust the vision pipeline. A learner should note explicitly: "the
  tag detector worked in the demo" is never the same claim as "it is safe to let a robot dock on it
  unsupervised."
- **N/A here:** no fieldbus (CAN-FD/EtherCAT) integration — this project's output is a pose, published
  over whatever middleware (ROS 2/TF, or a custom IPC) the consuming controller uses; it never itself
  commands a bus.

## 4. Business & regulatory context

- **Who needs this capability.** Warehouse/logistics AMR fleets (dock and shelf alignment); electronics
  and light-industrial assembly (fixture and part localization, tool-changer verification); AR/VR and
  teleoperation products (overlay anchoring); and robotics R&D broadly, where a cheap, ubiquitous,
  well-understood localization primitive is often the FIRST tool reached for before investing in a
  full visual-SLAM or learned-pose-estimation stack.
- **The players.** AprilTag (academic origin, University of Michigan, now widely embedded via the
  open-source `apriltag` library and ROS's `apriltag_ros`) and ArUco (OpenCV's built-in module) are the
  two dominant OPEN-SOURCE fiducial families most robotics teams reach for directly, at zero licensing
  cost — a notable contrast with much of this repository's other perception domains, where the
  production-grade tooling is commercial or requires substantial in-house engineering. NVIDIA's Isaac
  ROS AprilTag node is the GPU-accelerated production path this project's own architecture (README
  "Prior art") most directly mirrors. Build-vs-buy (SYSTEM_DESIGN section 5.3) is close to "always
  buy" here — the open-source detectors are mature, fast, and free; a company writing its own fiducial
  detector from scratch (as this PROJECT does, for teaching purposes) is rare in production, EXCEPT
  when embedding the detector into custom silicon or an unusually constrained compute budget, where
  understanding the algorithm well enough to hand-optimize it (this project's actual pedagogical goal)
  becomes valuable again.
- **What getting it wrong costs.** A missed detection is usually just a retry (a docking approach that
  aborts and re-attempts); a WRONG pose accepted with high confidence is the dangerous failure mode —
  an AMR docking against a miscalculated pose, or a manipulator approaching a fixture at the wrong
  offset, can cause real equipment damage or, in a shared workspace, a safety incident. This is exactly
  why this project's decode step enforces an error-correction MARGIN (THEORY.md's coding-theory
  section) rather than a bare "closest match" — accepting a low-confidence match is a worse failure
  than rejecting a marginal one and asking for a re-detection.
- **Regulatory and adjacent considerations (orientation only — SYSTEM_DESIGN section 6.2's regulatory
  map does not name fiducial localization directly; the nearest rows and adjacent concerns):**
  - Whatever machine-safety standard applies to the PLATFORM this feeds (ISO 10218/15066 for an
    industrial arm docking against a fixture, ISO 13482-family expectations for a service AMR, per
    SYSTEM_DESIGN section 6.2) governs the SYSTEM this project's pose feeds into — this project's own
    output carries no certification of its own, and the "sim-validated only" caveat in section 3 above
    is the operative constraint at this project's own layer.
  - Positional accuracy claims for anything safety-adjacent (a docking sequence near people, a
    fixture-verification step in a regulated manufacturing process) would, in a real deployment, need
    the same kind of measured-and-margined error characterization this project's own `corner_accuracy`
    and `pose` gates model in miniature — never asserted, always measured against ground truth.
- **Owning team.** Perception/calibration (SYSTEM_DESIGN section 5.1) — README "System context"
  already names this: fiducial tooling is characteristically a bring-up and integration tool as much
  as a shipped feature, commonly owned by whichever team is closest to camera calibration and sensor
  bring-up; adjacent teams include controls/autonomy (the consuming docking or servo loop) and, for a
  teleoperation/AR product, the HRI/UX team (SYSTEM_DESIGN's domain 21).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
