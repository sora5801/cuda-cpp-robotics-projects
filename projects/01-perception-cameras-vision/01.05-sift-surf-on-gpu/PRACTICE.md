# 01.05 — SIFT/SURF on GPU (harder, warp-level reductions): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-10. All parts, prices, and vendor names below are **illustrative examples,
> never endorsements** — verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project is pure software — there is no "part" to assemble. Its PHYSICAL CARRIER is a camera and
the compute module reading its stream, so this section describes THAT construction instead.

A camera module that would feed this pipeline on a real robot is built from: a **lens assembly** (glass
or molded-plastic elements in a barrel, epoxied or threaded to hold focus under vibration), an **image
sensor** (a CMOS die — global-shutter for anything that moves fast enough to alias under rolling
shutter, e.g. a legged robot's foot cameras or a drone's downward-facing one; rolling-shutter is
acceptable and cheaper for a slow-moving AMR), a **sensor PCB** carrying the die, decoupling capacitors,
and a MIPI-CSI or USB3 transceiver, and a **housing** that sets the standoff between lens and sensor
(focus is set once at manufacture and often epoxy-locked) while sealing against dust/moisture (IP54+ for
anything outdoor). Mechanical mounting matters more than it looks: a camera that flexes relative to the
robot's frame under vibration corrupts EVERY downstream geometric assumption this pipeline's gates check
(the scale/rotation recovery this project demonstrates assumes the two views differ ONLY by the robot's
own motion, not by the bracket wobbling) — a rigid, ideally metal, mount directly to the structural
chassis (not to a body panel) is the standard fix. Lens choice trades field of view against the very
"scale change per unit motion" this project is built to survive: a wide FOV lens produces LARGER apparent
scale changes per meter of robot travel (more of this project's demonstrated capability gets exercised
per frame) at the cost of more radial distortion to calibrate out upstream (01.01's domain).

## 2. Real hardware — chips, parts, illustrative BOM

The compute this project's kernels would run on, by tier:

| Tier | Compute | Notes |
|---|---|---|
| Embedded / edge | NVIDIA Jetson Orin Nano/NX (Ampere-class iGPU, shared LPDDR5) | The realistic target for an AMR/drone running SIFT-based relocalization on-device; this project's kernel list (separable blur, 3x3x3 stencil, warp-level reductions) maps directly, though real keypoint counts (hundreds-thousands, see THEORY.md) demand more careful memory-tiling than this teaching implementation uses |
| Desktop / dev workstation | x86 + discrete RTX-class GPU (this project's reference machine: RTX 2080 SUPER, sm_75) | Where this project was built and verified; also the typical OFFLINE map-building host (structure-from-motion, one-time large-scale reconstruction) |
| MCU-class | N/A for this stage | A DoG/SIFT pipeline needs a real GPU or a dedicated vision accelerator; an MCU (STM32-class) would run only the FINAL consumer of matched features (e.g., a pose-graph update), not this pipeline itself |

The camera this pipeline's input would come from, by grade: **hobby** — a USB3 global-shutter module
(e.g. in the class of an OV9281-based board, ~$30-60) with a fixed-focus M12 lens; **research** — an
industrial machine-vision camera (e.g. FLIR/Teledyne Blackfly S class, ~$300-600) with a genlock/trigger
input for multi-sensor synchronization; **industrial/automotive** — an automotive-qualified (AEC-Q100)
image sensor behind a ruggedized, connector-sealed housing, often paired with an ISP ASIC that does
lens-shading/HDR correction in hardware before the frame ever reaches this pipeline's Gaussian-blur
stage. Comms from camera to compute: MIPI CSI-2 (short, board-to-board, lowest latency — the common
choice on Jetson-class carriers), USB3 (longer runs, easy off-the-shelf cameras, adds a few ms of
latency this project's real-time budget in SYSTEM_DESIGN.md item 1 would need to account for), or GigE
Vision (industrial, long cable runs, PoE power-over-the-same-cable convenience).

## 3. Installation & integration — putting it on a real robot

This pipeline would run as its own process/node on the robot's PERCEPTION compute tier (the Jetson-class
or x86+dGPU board named in SYSTEM_DESIGN.md's hardware-architecture item, NOT the safety-rated MCU that
closes the actuator loop). OS: Linux (Ubuntu, JetPack's L4T on Jetson) — CUDA's driver stack and every
tool this repo's `docs/BUILD_GUIDE.md` documents target Linux in production even though this repo's OWN
build target is Windows + VS 2026 for the OWNER's development machine. Real-time constraints: this is a
PERCEPTION-tier component (per SYSTEM_DESIGN.md's canonical stack), typically running at whatever rate
new keyframes are selected (often well under the camera's raw frame rate — e.g., 2-10 Hz for a
relocalization service, not 30-60 Hz), so it does NOT need a hard real-time OS or scheduling class; it
DOES need to not stall the pipeline stage feeding it (bounded worst-case latency, not average-case).

**ROS 2 node/topic shape.** Input: `sensor_msgs/Image` (or `CompressedImage`) on a camera topic, plus
`sensor_msgs/CameraInfo` for intrinsics (needed by whatever calibration step, 01.17, precedes this
pipeline in a real stack). Output: a custom or `vision_msgs`-family message carrying keypoint
`(x, y, octave, scale, orientation)` and descriptor arrays per frame, OR — more typically for the
loop-closure use case this project's README "System context" names — a direct service/action call into
a SLAM back-end's relocalization query (e.g., ORB-SLAM3-style, or a vocabulary-tree place-recognition
service) rather than a raw topic, since matched features alone are rarely the end consumer; the POSE
correction they imply is.

**Calibration and bring-up.** Intrinsic calibration (focal length, principal point, distortion — a
checkerboard-target procedure, 01.01's domain) must precede this pipeline: SIFT's own scale/orientation
math assumes a locally-linear (undistorted) image, and radial distortion directly corrupts the
`sigma_img` values this project's gates check. No further "calibration" is specific to SIFT itself
beyond that, and beyond the synthetic-vs-real tuning story this project's own THEORY.md documents
honestly (the ratio-test threshold and minimum-distance floor were measured and set FOR THIS PROJECT'S
synthetic scene — a real deployment would re-measure both against its OWN operating imagery, exactly the
same way a real vision team tunes a matcher against its own dataset before shipping).

**The safe hardware-testing ladder** (CLAUDE.md §1 applies in full — everything in this repository is
sim-validated only): simulation (this project's synthetic scene pair, and any photorealistic simulator
like Isaac Sim/Gazebo feeding synthetic camera frames with KNOWN ground-truth poses, extending exactly
the "exact ground truth from an analytic transform" trick this project's `make_synthetic.py` uses) →
hardware-in-the-loop (real camera, recorded/replayed bag data, no robot motion) → bench jig (camera
mounted on a controlled slider/turntable so the TRUE relative pose between two captures is known,
letting a real deployment's scale/rotation-recovery gates be checked against physical ground truth the
same way this project's synthetic gates are) → tethered/current-limited operation → free running. This
project's output (matched features, implied relative pose) never directly commands actuation on its own
— it feeds a SLAM/localization back-end that does — but any downstream motion command that traces back
to a bad match (e.g., a false loop closure snapping the map) is exactly the class of failure this
ladder, and the negative-control gate this project ships, exist to catch before it reaches a real motor.

## 4. Business & regulatory context

*Didactic orientation only — not procurement, legal, or compliance advice.*

**Who needs this, and where it lives.** Scale-and-rotation-invariant feature matching is core
infrastructure for: SLAM/localization stacks in mobile robotics (warehouse AMRs re-localizing after a
lift interruption or a map update; consumer robot vacuums building and re-using a floor map);
structure-from-motion and photogrammetry (surveying, film/VFX, 3-D asset capture — largely OFFLINE, batch
use, where SIFT's float-descriptor cost this project measures directly matters less); panorama stitching
and image registration (consumer and industrial imaging products); and AR/VR world tracking. Commercial
and open-source players: Google (VPS / visual positioning), Niantic (AR anchoring), essentially every
commercial SLAM vendor (RTAB-Map, Cartographer-adjacent ecosystems, and proprietary AMR-fleet stacks from
companies like Locus Robotics, 6 River Systems-class vendors) either use SIFT/SURF-family features
directly or a learned successor trained to solve the identical correspondence problem (SuperPoint/DISK,
per THEORY.md). OpenCV (BSD-3 licensed, `cv::SIFT` and `cv::cuda::SIFT`) is the dominant OPEN-SOURCE
implementation any of these teams would start from rather than hand-roll — exactly the "study it, do not
copy it wholesale" relationship this project's README "Prior art" section names.

**Patent history (didactic, not legal advice).** SIFT was covered by US Patent 6,711,293 (Lowe, filed
1999), which EXPIRED in March 2020 — the reason `cv::SIFT` moved from OpenCV's patent-encumbered
`opencv_contrib` module into the main, freely-licensed `opencv` module that same year. SURF was similarly
covered by patents associated with its original authors/KU Leuven; as with SIFT, the core algorithmic
patents in this family have now lapsed in most jurisdictions, though (as with any patent question) a
team shipping a commercial product should have its own counsel confirm current status in every
jurisdiction it operates, rather than relying on a general statement like this one.

**What getting it wrong costs.** A false-positive loop closure (the exact failure mode this project's
`negative_control` gate exists to catch) can silently corrupt a robot's entire map, causing subsequent
navigation to send the robot into a wall or off a ledge it "believes" is somewhere else — a real,
recorded failure class in mobile robotics, not a theoretical one. In a photogrammetry/survey product,
mismatched features corrupt the reconstructed 3-D geometry, a quality problem rather than a safety one.
Neither failure mode is covered by a dedicated robot-safety standard the way, say, collaborative-arm
force limits are (see SYSTEM_DESIGN.md item 6's regulatory map) — perception-stack correctness for
localization is generally addressed through a company's OWN validation/test process and, for
higher-integrity applications (e.g., an AV's localization stack), through the broader functional-safety
frameworks named there (ISO 26262 / UL 4600 territory), not a perception-specific standard.

**Where this work lives inside a robotics company.** Squarely the PERCEPTION team's ownership (per
SYSTEM_DESIGN.md item 5's org map), adjacent to SLAM/mapping and to the simulation/tools team that
would own the synthetic-data-generation approach this project itself demonstrates; typical role titles
include Perception Engineer, Computer Vision Engineer, and SLAM Engineer. Build-vs-buy: most teams BUY
(use OpenCV/a vendor SLAM stack) rather than hand-roll a SIFT implementation from scratch for
production — the "build" case (writing your own, as this project teaches) is almost always for
EDUCATIONAL purposes, a genuinely novel research variant, or a hard real-time/embedded constraint no
off-the-shelf library meets, which is exactly this repository's own framing (CLAUDE.md §1: didactic
study material, a starting point toward real systems, not itself a production dependency).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
