# 01.24 — Transparent/reflective object detection via polarization imaging: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

A DoFP polarization camera is, mechanically, an ordinary machine-vision camera with one extra
lithography step in the sensor's fabrication: after the photodiode array is built, a layer of
sub-wavelength metal WIRE-GRID polarizers is deposited and patterned directly over the pixels (the same
back-end-of-line process family that deposits a Bayer color-filter array over a color sensor — the
optical stack is: microlens -> polarizer wire grid -> (optional) color filter -> photodiode). Each
polarizer is a set of parallel conductive nanowires spaced far below the wavelength of visible light
(~100-140 nm pitch); light polarized PARALLEL to the wires drives current in them and is absorbed/
reflected, while light polarized PERPENDICULAR passes through largely unimpeded — the wire orientation
IS the polarizer's transmission axis. Four orientations (0/45/90/135 degrees) are patterned in a
repeating 2x2 tile across the array, exactly the geometric idea this project's mosaic model uses.

Downstream of the sensor die, construction is ordinary industrial-camera practice: a C/CS-mount or
board-level lens (a REGULAR lens — the polarization optics live entirely in the sensor, no special glass
needed, unlike a bandpass or polarization-ROTATING filter design), a rigid housing (often aluminum,
sometimes with active cooling for long machine-vision duty cycles), a GigE/USB3/MIPI-CSI interface
connector, and — for an outdoor or wash-down robot deployment — IP-rated sealing at every connector and
seam (PRACTICE.md's usual "what breaks in the field" concern: condensation or dust ingress on the
sensor window degrades polarization measurement MUCH faster than it degrades plain intensity, because a
speck of dust or a moisture film is itself a small unwanted polarizer/depolarizer sitting directly in
the optical path). Mounting is otherwise identical to any other machine-vision camera: a rigid bracket
referencing the robot's own frame, with the extrinsic calibration this project's upstream siblings
(01.17) would establish relative to other sensors on the same platform.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**The sensor itself — the one part that makes this modality possible.** The illustrative real part this
project's mosaic model is styled after is Sony's **IMX250MZR** (and its color sibling **IMX253MZR**,
Polarsens series): a 2/3" CMOS sensor with a 2x2 wire-grid polarizer array (0/45/90/135 degrees)
fabricated over a 5-megapixel-class pixel array, sold integrated into cameras from LUCID Vision Labs
(Phoenix/Triton series), FLIR/Teledyne, and others rather than as a bare die a robotics team would
solder themselves. **Verify current before relying on any of this** — sensor generations and vendor
lineups change.

**Compute tiers** (mirroring the repo's usual three-tier framing):

| Tier | Example | Notes |
|------|---------|-------|
| Hobby/research | A laptop/desktop with any recent NVIDIA GPU (this project's own dev target, an RTX 2080 SUPER, sm_75) + a USB3 polarization camera | Fine for algorithm development and the exact demo this project ships; not weatherproofed or vibration-rated. |
| Embedded/production | Jetson Orin-class SoC (the repo's usual "on-robot GPU" answer) + a GigE or MIPI-CSI polarization camera | Where this pipeline would actually run on a warehouse AMR or a manipulator cell's vision PC — full-resolution demosaic/Stokes/DoLP/AoLP at 30-60 Hz is comfortably within an Orin's perception budget given this project's tiny per-pixel FLOP count. |
| Industrial | A ruggedized machine-vision PC + GigE Vision polarization camera in an IP67 housing | The bin-picking-cell answer — fixed lighting, fixed mount, long duty cycles; matches SYSTEM_DESIGN.md §2.2's manipulator work cell suite. |

**Lens.** A standard fixed-focal-length machine-vision lens (C-mount is common for this sensor format);
no special optical coating is required FOR the polarization measurement itself, though a good
anti-reflective coating reduces internal lens-flare-induced polarization noise the sensor cannot
distinguish from scene signal.

**Illumination (an easily-overlooked but load-bearing part).** Because DoLP/AoLP measurements are
weak-signal for anything but strongly specular surfaces, controlled, diffuse, UNPOLARIZED illumination
(the sensor's own baseline assumption — THEORY.md's "unpolarized light source" premise) matters more for
this modality than for plain intensity imaging: a linearly-polarized light source (some LED ring lights
are, inadvertently, partially polarized) would corrupt every measurement in this project's pipeline in
a way a plain camera would never notice.

## 3. Installation & integration — putting it on a real robot

**Where it runs.** The perception compute this project's pipeline would run on (Jetson Orin-class or an
industrial PC's discrete GPU, per §2 above) — the SAME machine that already runs the rest of the vision
stack (01.01-class ISP, stereo/structured-light depth), since a `TransparentObjectMask` is meant to be
FUSED with, not replace, the existing depth pipeline (README "System context": polarization tells you
*that* something specular is there, not directly *how far*).

**OS and real-time constraints.** Linux (Ubuntu/JetPack on Jetson), soft real-time — this pipeline sits
at the camera-perception boundary (30-60 Hz, SYSTEM_DESIGN.md §1.1), not the hard-real-time control
loop, so a standard (non-RT) kernel with a well-behaved perception process is the normal answer, same as
every other domain-01 camera project in this repo.

**ROS 2 shape.** A DoFP camera driver node publishing the raw mosaic (or, more commonly, the vendor
driver's already-demosaiced 4-plane output) as a custom message resembling `sensor_msgs/Image` but
4-channel (`PolarizationImage`-shaped, per SYSTEM_DESIGN.md §3's message-shaped-struct convention: the
same fields as `Image` — `width`, `height`, `encoding`, `data` — with `encoding` naming the 4-plane
layout); this project's node would subscribe to that topic, publish a `DoLP`/`AoLP` pair (each a plain
single-channel `Image`-shaped topic) and a `TransparentObjectMask` (binary/labeled `Image`-shaped)
downstream to the costmap (14.02) or grasp-candidate (19.01) consumer named in README "System context".

**Exposure discipline — the practical gotcha this project's own numerics section anticipates.** DoLP
needs UNCLIPPED channels: if any of the four polarizer-angle readings SATURATES (hits the sensor's max
DN — 255 in this project's 8-bit convention, or the sensor's native bit depth in a real system), the
Stokes reconstruction is silently wrong in that pixel (a clipped `I(theta)` under-reports its true
value, corrupting `S0`/`S1`/`S2` in a way that does not fail loudly). A real deployment therefore needs
AUTO-EXPOSURE tuned to the WORST-CASE (most specular, closest-to-normal-incidence, i.e. brightest)
channel across the whole frame, not the frame's mean brightness — the opposite of typical plain-camera
auto-exposure, which targets a mid-gray average. This project's synthetic scene is built with headroom
(peak mosaic DN measured well under 255, `data/README.md`) specifically so this real-world caveat does
not silently corrupt the demo; a learner adapting this pipeline to a real sensor should add a saturated-
pixel confidence mask (THEORY.md "Numerical considerations" flags the same need for low-`S0` pixels).

**Bring-up / calibration procedure.** (1) Verify the sensor's actual per-pixel polarizer geometry against
its datasheet (this project's `kernels.cuh` mosaic layout is illustrative, README "Limitations" says so
explicitly) — do NOT assume the 0/45/90/135 assignment this project uses matches a specific real part
without checking. (2) Measure per-pixel polarizer EXTINCTION RATIO non-uniformity (THEORY.md "Where
this sits in the real world") against a known-polarized calibration source (a rotating polarizer in
front of a diffuse unpolarized light works) and build a per-pixel correction LUT — real DoFP sensors
have measurable pixel-to-pixel extinction-ratio variation that, uncorrected, shows up as a fixed-pattern
DoLP/AoLP noise floor. (3) Verify the auto-exposure discipline above against the actual sensor's
saturation point. (4) Validate the whole pipeline against a KNOWN target (a flat glass sample at a
measured incidence angle is a natural choice — exactly this project's `fresnel_anchor` gate, run against
a real photograph instead of a render) before trusting the detector on an unknown scene.

**Safe hardware-testing ladder.** This project's OUTPUT is a detection mask, not itself a motion
command — it does not directly drive an actuator — so the ladder that matters is the one for whatever
CONSUMES the mask: simulation (validate the costmap/grasp-planner behaves sanely given a
`TransparentObjectMask`, using this project's own synthetic scenes as test fixtures) -> hardware-in-the-
loop (real camera, simulated or logged robot motion) -> bench jig / tethered / current-limited motion
near the detected glass/metal object -> free running, with E-stop and software limits active at every
rung. Nothing in this repository is safety-certified (CLAUDE.md §1); a `TransparentObjectMask` feeding
motion planning near a real glass storefront or a real bin-picking cell is exactly the kind of consumer
that needs this full ladder before any free-running use.

## 4. Business & regulatory context

**Who needs this.** Any robot operating around glass or bare/polished metal that its OTHER sensors
cannot reliably see: warehouse/retail AMRs (glass storefronts, interior partition walls, safety
railings, elevator doors — SYSTEM_DESIGN.md §2.1), 6-DoF manipulator bin-picking cells handling glass
containers, mirrored/polished metal parts, or shiny packaging film (§2.2), and — beyond this project's
two named reference robots — automotive perception (windshields, wet-road glare) and recycling/sorting
lines separating glass and metal fractions by exactly the physical cue this project measures.

**Market landscape.** Commercial DoFP sensor/camera vendors: Sony (the sensor silicon), LUCID Vision
Labs, FLIR/Teledyne, and Basler among the integrated-camera makers; on the software/algorithm side, both
classical (this project's lineage — OpenCV's polarization module) and learned (README "Prior art"'s
ClearGrasp-era transparent-object-segmentation research, which several robotics/computer-vision groups
and startups have productized for bin-picking) approaches are commercially active. This remains a
NICHE sensing modality relative to plain RGB or depth cameras — most robots ship without one — used
specifically where glass/metal failure modes are a proven, costly problem for the target application.

**What getting it wrong costs.** A missed glass panel is a COLLISION, not a near-miss: because the
robot's other depth sensors report confident-but-wrong data (THEORY.md "The problem"), a mobile robot
can drive straight into a glass door at full speed with no warning from its primary obstacle sensors,
and a manipulator can crush or drop a glass/metal part its depth camera mis-triangulated. Costs are the
usual robotics-safety-incident costs (property damage, schedule/downtime loss, and — where a person is
nearby — an injury liability event) rather than anything unique to this sensing modality; the modality
exists specifically to REDUCE that risk class, not because a regulation mandates it.

**Regulatory path.** No standard mandates polarization sensing specifically. The relevant frameworks are
the ones any perception input feeding motion decisions falls under, per the domain the robot operates
in (SYSTEM_DESIGN.md §6.2's regulatory map): industrial manipulator work cells fall under **ISO 10218**
/ **ISO/TS 15066** (collaborative-robot force/speed limits) if the cell is collaborative; mobile service
robots fall under **ISO 13482**; where this feeds an automotive perception stack, **ISO 26262** /
**UL 4600**. This project computes no certified metric and makes no compliance claim — it is a
perception INPUT to whatever safety-rated system consumes its output, orientation only, never
compliance guidance.

**Where this work lives in a robotics company.** Perception (SYSTEM_DESIGN.md §5.1's org map), typically
with a "sensing/algorithms research" specialization for a genuinely non-standard modality like this one
(as opposed to the "core perception" team maintaining the primary camera/LiDAR/depth stack) — adjacent
teams: optics/hardware (choosing and integrating the actual sensor, §2 above), the planning/navigation
team consuming the costmap (14.02), and the manipulation team consuming the grasp filter (19.01).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
