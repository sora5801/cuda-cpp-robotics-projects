# 01.07 — Fisheye/omnidirectional unwarping and multi-camera surround-view stitching: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5-6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

*Section dated 2026-07-10. Illustrative, not a build guide for any specific product.*

A real automotive/AMR fisheye camera module is a small, sealed optical/electronic assembly, not a bare
lens: a stacked LENS GROUP (typically 6-8 glass/plastic elements, the front one strongly concave to
achieve the 150-190-degree FOV class this project models) sits in front of a CMOS image sensor die
(commonly a 1/2.7" to 1/3" optical format for automotive parts) bonded to a small PCB carrying the
sensor's serializer chip. The whole assembly is potted or gasket-sealed into an **IP67-or-better
housing** — mandatory for anything mounted low on a vehicle exterior (front/rear bumper, side mirror
housing) where road spray, pressure washing, and temperature cycling are routine. The front element
usually carries a hydrophobic/oleophobic coating (reduces water beading and fingerprint smudging) and,
in higher-end modules, a small resistive heater bonded near the glass to clear condensation and light
frost — a direct answer to a real, common field failure: a fogged or iced fisheye lens is functionally
blind, and because the FOV is so wide, even a small area of obstruction near the rim (exactly where
this project's vignette sits) can occlude a disproportionate share of useful ground coverage.

Mounting is rigid and PRECISE: this project's whole geometric pipeline assumes a FIXED,
known extrinsic pose per camera (`kernels.cuh`'s hardcoded mounts/tilts) — on a real vehicle this means
a stamped or die-cast metal or reinforced-plastic bracket with a repeatable, low-tolerance mounting
interface (a shift of even 1-2 degrees in tilt visibly displaces the effective FOV boundary, THEORY.md
"The problem"). Common field failure modes: bracket fatigue/loosening from vibration (the mounting
torque spec and thread-locker are not incidental details — a camera that has rotated 3 degrees since
factory calibration silently degrades every downstream BEV pixel), connector/cable chafing at the
body-to-mirror or body-to-bumper flex point (a wiring harness that flexes with every mirror fold or
door-adjacent panel gap eventually fatigues), and lens-surface abrasion from repeated automated
car-wash brush contact on exposed mounts.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Compute | Camera module class | Link/interconnect | Approx. cost tier (per camera + shared compute) |
|---|---|---|---|---|
| Hobby/prototype | A single desktop/laptop GPU (this project's own dev target — an RTX 2080-class card) fed by USB3 fisheye webcams (e.g. a 180-degree USB board-camera module) | Consumer USB fisheye modules, ~2-5 MP, no automotive environmental rating | USB 3.0 (a few meters max, no automotive EMC hardening) | Camera modules ~$20-60 each; compute is the desktop GPU already on hand |
| Research/robotics (AMR) | Jetson Orin-class SoC (integrated GPU shares memory with CPU — no PCIe camera-frame copy needed) | Automotive-adjacent or industrial fisheye modules (e.g. an OV2311/AR0234-class sensor behind a fisheye lens stack), MIPI CSI-2 direct or via a deserializer | MIPI CSI-2 (short runs) or a single GMSL2/FPD-Link-III deserializer hop for runs beyond ~1 m | Camera modules ~$50-200 each; Orin module ~$400-2000 depending on tier |
| Industrial/automotive production | Automotive-grade SoC (e.g. an ISP-equipped ADAS domain controller) or a dedicated surround-view ECU ASIC | Automotive-qualified (AEC-Q100) image sensor + fisheye lens stack in an IP6K9K-rated housing, e.g. an OmniVision OX-series automotive sensor | GMSL2 or FPD-Link III serdes per camera, aggregated at a deserializer hub feeding the domain controller | Camera modules ~$30-80 each at automotive volume (BOM cost, not retail); domain controller ~$100-500 in BOM at volume |

Actuation/power chips are N/A for this project (no actuators — the 4 cameras and their compute are the
whole hardware surface); a real installation's power tree is simple: each camera module draws a few
hundred mA at a low DC rail (often delivered over the serdes coax/twinax pair itself, "power-over-
coax," avoiding a separate power harness run to each mirror/bumper location).

## 3. Installation & integration — putting it on a real robot

Where this code would physically run: the vehicle/AMR's main perception compute (the Jetson-class or
automotive-domain-controller tier from §2 above), NOT an MCU — the BEV compositor's 4-camera-per-pixel
gather is squarely GPU/vision-accelerator work. Real-time constraint: the whole capture-to-display
pipeline (ISP -> unwarp/stitch -> display) targets a full camera frame period (typically 33 ms at 30
Hz) so the driver display never lags perceptibly behind the physical world — this project's own
kernels measure well under 1 ms each on the reference GPU, so at production scale the budget is
consumed by higher resolution and additional processing (photometric blending, ghosting suppression),
not by this algorithm's asymptotic cost.

**ROS 2 shape.** Each camera would publish a raw or ISP-processed `sensor_msgs/Image` (or the
compressed equivalent) on its own topic (`/surround/front/image_raw`, `/surround/left/image_raw`,
etc.), each carrying a `sensor_msgs/CameraInfo` with this project's equidistant intrinsics
(OpenCV's `fisheye` distortion-model convention maps directly onto `kernels.cuh`'s `fx, cx, cy` +
Kannala-Brandt `k1..k4`) and a static `tf2` transform `T_vehicle_cam` matching this project's rig
extrinsics. A surround-view node would subscribe to all 4, publish the stitched BEV as its own
`sensor_msgs/Image` on `/surround/bev`, and (in a production system) also publish the per-pixel
coverage/confidence this project's coverage bitmask represents, so downstream consumers (a costmap
node, a display renderer) know which BEV pixels are single-camera, overlap-blended, or unfilled.

**Bus/link:** the 4 cameras connect via GMSL2 or FPD-Link III serdes (§2) into a deserializer hub,
which lands the 4 raw streams on the domain controller's MIPI CSI-2 input(s) — this project's "4
independent `Image` inputs" model maps directly onto that hub's 4 output streams. **Time-sync (FSYNC):**
because the BEV compositor treats all 4 cameras' pixels as simultaneous samples of one instant, the
cameras must be HARDWARE FRAME-SYNCHRONIZED (a shared FSYNC line from the deserializer hub, common in
GMSL2 systems) — without it, a fast-moving object could appear at inconsistent BEV positions from
different cameras' slightly different capture instants, a real, non-obvious failure mode this
project's static synthetic scene cannot demonstrate.

**Calibration & bring-up procedure** (the step this project hardcodes the ANSWER to):
1. **Intrinsic calibration** — capture a fisheye checkerboard/AprilTag-grid sequence per camera and fit
   the Kannala-Brandt polynomial (Kalibr or OpenCV `fisheye::calibrate`) — replaces this project's
   assumed `fx=74.0, k_i=0`.
2. **Extrinsic calibration** — with the cameras mounted on the actual vehicle, either survey each
   mount's position/orientation by hand (low accuracy, adequate for a coarse BEV) or use a printed
   ground-plane calibration target (a large checkerboard or fiducial-marker mat) visible to 2+
   adjacent cameras at once, solving for each `T_vehicle_cam` that makes their overlapping views agree
   — the real-world analogue of this project's `seam_consistency` gate, run as a CALIBRATION step
   rather than a verification one.
3. **Verification** — drive/roll the vehicle over a known ground pattern and visually confirm the
   stitched BEV shows no visible seam offset or double-imaging.

**Safe testing ladder** (CLAUDE.md §1's caveat applies in full — nothing in this repository is
safety-certified, and this project's output is a perception/visualization artifact, never validated
for closed-loop control): simulation (this project's synthetic scene) -> hardware-in-the-loop with
pre-recorded real fisheye footage and no vehicle motion -> a stationary bench rig with the physical
camera modules and a printed calibration target -> a tethered, current-limited vehicle at a near-
walking pace in a controlled area with an E-stop -> free operation only after the full calibration and
verification procedure above has been independently checked. This project's output should never be
wired directly into a motion controller without every one of those rungs and a great deal more
validation than a synthetic 4-camera scene can provide.

## 4. Business & regulatory context

*Section dated 2026-07-10. Didactic orientation only — not procurement, legal, or compliance advice.*

**Who needs this.** Surround-view/"around-view monitor" systems are now common on mid-to-high-trim
passenger vehicles (originated with Nissan's Around View Monitor and Toyota/Honda/etc. equivalents;
now widespread) and are close to standard on delivery/warehouse AMRs for docking and close-quarters
obstacle awareness — the two reference robots this project's README cites. Commercial players: Tier-1
automotive suppliers (Bosch, Continental, Valeo, Denso) ship full surround-view ECUs and camera
modules as an integrated product; NVIDIA (DriveWorks/VPI, THEORY.md "Prior art") supplies the
GPU-accelerated software stack many of these systems build on; on the AMR side, mobile-robot OEMs
(and their perception-stack vendors) increasingly ship an equivalent capability as a docking/parking
aid, sometimes literally reusing automotive surround-view components.

**What getting it wrong costs.** A miscalibrated or ghosting-prone surround-view system that
mis-represents an obstacle's true position is, at minimum, a driver-trust and UX failure (documented
consumer complaints exist for real production systems around exactly the flat-ground-assumption
ghosting this project measures); at worst, if a driver or an automated system TRUSTS the visualization
for a close-quarters maneuvering decision and the ghosting hides or misplaces a real obstacle, it is a
genuine collision-risk contributor. This is precisely why production systems treat surround-view as a
DRIVER-ASSIST visualization aid, not a certified collision-avoidance sensor, and why any AMR use of an
equivalent capability for actual obstacle avoidance (rather than just a docking-aid display) needs a
fused, depth-aware sensor input this project's single-frame flat-ground BEV does not provide.

**Regulatory landscape (orientation only — see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) item 6 for the full map).** Camera-
based surround-view/mirror-replacement systems intersect **UN Regulation No. 46** (devices for indirect
vision — the regulation under which camera-monitor systems can legally replace physical mirrors in UN
markets) and, in the US, **FMVSS 111** (rear visibility requirements, which for years mandated a
rear-view camera on new passenger vehicles and increasingly intersects camera-based indirect-vision
systems generally). Neither this project nor anything in this repository claims or implements
compliance with either regulation — they are named here purely so a learner knows what real,
binding standards exist for exactly this capability, and where to start looking if building toward a
real product (CLAUDE.md §1: no compliance claims are made anywhere in this repository).

**Where this work lives inside a robotics/automotive company.** Owning team: perception / camera
systems (sometimes a dedicated "surround view" or "around-view" sub-team on larger ADAS/AV programs),
working closely with the ADAS integration team (who consume the BEV for driver-display and, on more
advanced programs, for automated parking features) and with the mechanical/electrical teams who own
camera-module sourcing, mounting, and the serdes/wiring harness (§3). Typical role titles: Perception
Engineer / Computer Vision Engineer (camera calibration, unwarp/stitch algorithms — this project's own
scope), Embedded Systems Engineer (ISP pipeline, serdes/deserializer bring-up), ADAS Systems Integration
Engineer (fusing the BEV into the broader driver-assist feature set). Adjacent teams: mechanical
(camera module housing/mounting), functional safety / QA (surround-view systems on production vehicles
increasingly fall under ISO 26262 functional-safety processes at the system level, even though the
visualization itself is typically not the safety-rated element — SYSTEM_DESIGN.md item 6).
