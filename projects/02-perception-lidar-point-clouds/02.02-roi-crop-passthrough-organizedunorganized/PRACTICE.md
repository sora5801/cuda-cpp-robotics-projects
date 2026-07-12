# 02.02 — ROI crop, passthrough, organized↔unorganized conversion kernels: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections below dated 2026-07-11.*

## 1. Building it — construction of the robot/part

These kernels are pure software infrastructure — there is no physical part of a robot that "is" a
crop filter. The physical carrier this project's data structure represents, however, is very real:
the **organized grid is the literal shape of a spinning LiDAR's data stream**, and that shape is a
direct consequence of the sensor's mechanical construction.

A mechanical spinning LiDAR (the class this project models: a 16-beam unit in the geometric family of
a Velodyne VLP-16 / Ouster OS0-16 / Hesai XT16) is built around a rotating head carrying a vertical
stack of laser-diode/photodiode pairs (one per "ring"), spun by a small brushless motor at a fixed
rate (typically 5-20 Hz, i.e. 300-1200 RPM) with a slip-ring or optical rotary joint carrying power and
data across the rotating interface (no wires can survive continuous rotation). Every firing cycle, ALL
rings fire together at the current azimuth angle, timed by an encoder on the rotation axis — this is
*exactly* why the organized grid's natural packet layout is what it is (§3 below expands this). The
housing is a sealed, optically-transparent dome (usually polycarbonate) protecting the rotating optics
from dust, rain, and impact while passing the infrared beams; the whole assembly is vibration-mounted
(rotating machinery + vehicle vibration is a real fatigue-life concern) and, on an outdoor robot,
rated to an IP6x/IP67-class ingress standard. What breaks in the field: bearing wear in the rotating
joint (the single moving mechanical part, and the dominant wear item), dome scratching/fouling
(mud, insects, and rain streaks directly cost return rate — this project's synthetic 5% dropout is a
crude stand-in for exactly this), and slip-ring/optical-joint degradation causing intermittent, ring-
specific dropouts (a real-world failure mode this project's per-ring-independent dropout model does
NOT capture, since our model drops points uniformly across rings — an honest scoping gap, noted here
rather than silently implied away).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute this code would run on:** these kernels are cheap (sub-millisecond on an RTX 2080 SUPER at
this project's problem sizes) and would, in production, run on whatever GPU is ALREADY on the robot
for perception — not a dedicated chip. Illustrative tiers:

| Tier | Example | Notes |
|------|---------|-------|
| Embedded/edge (most common for this workload) | NVIDIA Jetson Orin NX/AGX (Ampere-class iGPU, shared with perception/planning) | The realistic target: these kernels are a rounding error next to the rest of the perception stack's GPU budget, so they ride along on whatever compute the LiDAR pipeline already uses. |
| Desktop/dev | x86 host + discrete RTX-class GPU (this project's development target: RTX 2080 SUPER, sm_75) | Used for algorithm development and offline dataset processing, not typically the deployed target. |
| MCU-class (NOT viable for this workload as GPU kernels) | none — a Cortex-M/R-class safety MCU could run a much-simplified CPU passthrough filter, but not this project's GPU pipeline | Named for contrast: the organized-grid scatter/scan machinery genuinely needs a GPU (or, at minimum, SIMD-capable) target to be worth the complexity over a trivial CPU loop at real-time rates. |

**The sensor this data comes from:** an illustrative 16-beam mechanical spinning LiDAR — hobby/
research tier (e.g. a Velodyne VLP-16-class unit, largely discontinued in favor of solid-state/
MEMS units as of this writing but still common in research fleets and used-equipment markets),
industrial tier (Ouster OS0/OS1-class, Hesai XT-series, higher channel counts and better dropout
characteristics than this project's illustrative 16-beam model), interfacing over Ethernet (UDP
packet stream, the near-universal spinning-LiDAR interface) into the compute tier above.

**No actuation, motor-control, or power-electronics silicon is exercised by this project** — N/A
because this project's inputs and outputs are both point-cloud data; nothing here commands hardware.

## 3. Installation & integration — putting it on a real robot

**Where this runs:** on the robot's main perception compute (the same box running the rest of the
LiDAR pipeline — 02.01 downsampling, 02.03 ground segmentation, 02.06 registration), as one stage in
a longer GPU pipeline, NOT a standalone process — the kernels here are cheap enough that launching
them as their own ROS 2 node with its own IPC overhead would likely cost more in message-passing
latency than the kernels themselves take to run. In production this logic is more often a function
call inside a larger perception node than a node of its own.

**ROS 2 shape:** conceptually, a node subscribing to `sensor_msgs/msg/PointCloud2` (the LiDAR
driver's raw output — organized when the driver publishes one, which many real drivers do via the
`height`/`width` fields: `height > 1` signals an organized cloud shaped `height x width`, exactly this
project's ring x azimuth grid) and republishing a cropped/converted `PointCloud2`. The **`is_dense`
field** is `sensor_msgs/PointCloud2`'s honesty flag: `is_dense = false` means "this cloud may contain
NaN/Inf points" — exactly this project's invalid-cell convention. In practice, a meaningful fraction
of real driver output sets `is_dense = true` even when NaN points ARE present (a known, long-standing
sloppiness in parts of the ROS ecosystem this project's authors have observed firsthand across
multiple vendors) — a downstream consumer that trusts the flag instead of checking for NaN itself will
eventually crash on a NaN it was told could not be there. This project's `is_invalid_point()` NaN test
never trusts a flag; it checks the data.

**Zero-copy aspirations:** a real GPU pipeline wants the LiDAR driver to DMA its packet buffer
straight into a device-resident array (GPUDirect, or at minimum a pinned-host-memory staging buffer)
so this project's kernels never wait on a host round trip — this project's own `main.cu` does an
ordinary `cudaMemcpy` from a loaded file, the honest simplification appropriate for a from-a-file
teaching demo; a production integration would replace that one call with a zero-copy ingest path
without touching any kernel in `kernels.cu`.

**ROI configs are per-consumer, not universal — the unglamorous production reality.** A single LiDAR
scan typically feeds several DIFFERENT crops simultaneously, each tuned to its consumer: a
localization stack might want a passthrough excluding ground and overhangs (roughly this project's
`kPassthroughZMin/Max`); an obstacle detector might want the FULL vertical range but a tight box ROI
around the vehicle's stopping distance; a camera-fusion coloring stage (project 02.17) wants the
frustum crop for exactly the mounted camera it feeds, at exactly that camera's calibrated intrinsics —
get any of these bounds wrong (too tight and real obstacles are silently invisible to that consumer;
too loose and the consumer wastes compute or, worse, sees things outside its intended field) and the
failure is silent, not a crash — a genuinely dangerous class of bug in a safety-relevant pipeline. Real
production LiDAR configs are strewn with hand-tuned crop bounds per consumer, versioned alongside
calibration data, and (should be, though this is inconsistently done in practice) covered by
regression tests exactly like this project's edge-cohort boundary test.

**A crop no one enjoys writing: vehicle-body / self-hit masking.** Before ANY of the crops this
project teaches run, virtually every real LiDAR install applies a FIXED per-unit mask removing points
that hit the robot's own chassis, roof rack, or mounting bracket — a LiDAR mounted on a vehicle roof
sees its own hood, mirrors, and roof rails as "obstacles" every single scan unless masked out. This
mask is typically NOT a clean geometric predicate like this project's box/frustum (the vehicle's own
geometry is irregular) — it is a per-unit, per-mount CALIBRATION artifact: a fixed boolean mask over
(ring, azimuth) cells, measured once during installation (park the vehicle, scan a known-empty area,
mark every cell that ALWAYS returns a near-range self-hit) and baked into the driver or the first
perception stage. It is unglamorous, install-specific, and absolutely necessary — the kind of
production detail a pedagogical project like this one, with its clean analytic room-and-boxes scene,
never needs to deal with but every real deployment does on day one. Because it is naturally expressed
as a per-(ring, azimuth)-cell boolean mask, it is architecturally IDENTICAL to this project's organized
`valid_predicate_kernel` — a real system would simply AND a self-hit mask into that same predicate.

**Safe hardware-testing ladder (this project's caveat — CLAUDE.md §1):** everything in this repository
is sim-validated only, never safety-certified. Were this pipeline ever connected to a real sensor and
a real robot: simulation (this project's synthetic scene) -> replay against LOGGED real sensor data
(no live robot motion involved) -> bench-mounted sensor on a static rig, live data flowing through the
FULL pipeline with an operator watching for silently-wrong crops -> only then would this code's output
be trusted as an input to any planning/control stage that could move a real robot, and even then never
without the safety layer(s) SYSTEM_DESIGN.md describes.

## 4. Business & regulatory context

**Who needs this, and why it is a real (if invisible) cost center.** Every company shipping a LiDAR-
equipped robot — warehouse AMRs, autonomous-vehicle stacks, agricultural and construction equipment —
needs exactly this kind of "glue" kernel, dozens of times over, one per consumer's ROI. It is
unglamorous, rarely gets a dedicated team, and is exactly the kind of code that either lives as
scattered ad-hoc filters throughout a codebase or, in a more mature organization, as a small shared
**perception infrastructure** library that every perception team calls into — the team most likely to
own this exact code in a real company is a **perception infrastructure / perception platform** team
(distinct from the teams owning the ALGORITHMS that consume its output — localization, detection,
tracking), reporting into the broader perception/autonomy organization (SYSTEM_DESIGN.md §5's org
map). Getting it WRONG costs nothing dramatic in isolation (a silently mis-cropped point cloud does
not crash a process) but compounds: a localization stack silently missing ground returns because a
passthrough bound was copy-pasted from a different sensor mount, or a camera-fusion crop silently using
a stale calibration after a camera was re-mounted, are the kind of bug that surfaces as an unexplained
accuracy regression weeks later, not a stack trace on day one — exactly why this project insists on
edge-cohort boundary testing and bit-exact GPU/CPU agreement as standard practice, not overkill.

**Commercial and open-source players:** PCL (`CropBox`, `PassThrough`) is the de facto open-source
baseline every robotics engineer has used at least once; NVIDIA's Isaac ROS / cuPCL packages the
GPU-accelerated equivalents for Jetson-class deployments; every major AV company (Waymo, Cruise,
Zoox-class stacks) and LiDAR-equipped AMR vendor maintains an internal, proprietary equivalent tuned
to their exact sensor mounts and consumers — this exact kind of kernel is common enough that it is
rarely a competitive differentiator on its own, but getting it wrong IS a differentiator (in the
undesirable direction).

**Regulatory path:** this project computes no safety-rated output on its own (it is a data-shaping
step, not a safety monitor), so no standard directly certifies IT — but it sits upstream of systems
that ARE regulated: an AMR's obstacle-detection pipeline (ISO 13482 orientation) or an AV's perception
stack (ISO 26262 / UL 4600 orientation, SYSTEM_DESIGN.md §6's regulatory map) depends on this stage
producing a CORRECT, not just plausible-looking, cropped cloud — a wrong ROI bound feeding a safety-
rated obstacle detector is exactly the kind of upstream data-quality bug a functional-safety audit
would want traced back to its source. This is a didactic orientation, not compliance guidance.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
