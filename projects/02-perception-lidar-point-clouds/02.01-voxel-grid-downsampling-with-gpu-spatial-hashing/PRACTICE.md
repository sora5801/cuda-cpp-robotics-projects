# 02.01 — Voxel-grid downsampling with GPU spatial hashing: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

*Section dated 2026-07-11. This project is software-only (no part of it is fabricated), so this section
describes the physical carrier the code serves: how the raw data it consumes actually arrives.*

A spinning mechanical LiDAR (the sensor this project's synthetic scan stands in for) is a rotating head
carrying the laser emitter/receiver pairs (the "beams" — 16 of them in this project's model), spun by a
small motor at a fixed rate (commonly 5–20 Hz), with the rotation angle tracked by an encoder so every
return can be timestamped with a known azimuth. Power and the digitized return stream cross the
rotating/stationary boundary either through a slip ring (older, mechanically simpler, wears over time)
or — in most current units — an inductive/optical rotary coupling with no physical contact at all
(better reliability, higher cost). The unit reports returns as a continuous stream of **UDP packets**,
not a "point cloud file": each packet typically bundles a handful of consecutive azimuth "firing
sequences," each sequence holding one range + intensity reading per beam (a "ring") plus that
sequence's precise timestamp — a driver reassembles a full revolution's packets into one `PointCloud`
message only after the fact (this is also why raw LiDAR data is inherently **motion-distorted**: a
360° sweep at 10 Hz takes 100 ms, during which a moving robot has physically translated/rotated, which
project [`02.08`](../02.08-per-point-motion-deskew-with-pose-interpolation/README.md) corrects — this
project's synthetic scan sidesteps that by construction, ray-casting against a STATIC scene, and says so
in `data/README.md`). **Mounting and field-of-view choices shape density directly**: a LiDAR mounted
higher sees more of the near-field ground at a shallower incidence angle (spreading the same beam count
over more area, i.e. LOWER near-field density but a larger dead zone directly beneath the sensor); a
narrower elevation fan concentrates the same beam count into a smaller vertical slice (higher density in
that slice, less vertical coverage) — exactly the elevation-vs-range trade `THEORY.md`'s beam-geometry
derivation makes quantitative, and exactly why voxel downsampling's leaf size is a per-mounting tuning
knob, not a universal constant (see §3 below).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Example unit class | Beams | Illustrative interface | Rough cost tier (2026, verify current) |
|------|--------------------|-------|--------------------------|------------------------------------------|
| Hobby / research, low channel count | Velodyne VLP-16-class (16-beam, this project's exact beam model) or an RPLIDAR/Livox-class solid-state unit | 16 (mechanical) or a scanning-pattern solid-state equivalent | 100BASE-T1/standard Ethernet UDP + a separate GPS/PPS sync line | low hundreds to low thousands (USD) |
| Mid-tier research / prosumer AMR | Velodyne/Ouster/Hesai 32- or 64-beam mechanical units | 32–64 | Gigabit Ethernet UDP, PTP or PPS+NMEA time sync | low-to-mid thousands |
| Automotive / high-density | Automotive-grade 128-beam mechanical, or automotive solid-state/MEMS/flash units (Hesai AT-class, Innoviz-class, etc.) | 64–128+ | Automotive Ethernet (100/1000BASE-T1), often with a dedicated perception SoC on the sensor side doing partial preprocessing | thousands to tens of thousands, automotive-qualified units command a premium for AEC-Q10x-class reliability |

**Compute this stage would run on:** the perception computer in every SYSTEM_DESIGN §2 reference robot
that carries a spinning LiDAR — a Jetson Orin-class embedded GPU SoC for an AMR or drone-scale platform,
or an x86 + discrete RTX-class GPU for a larger AV/industrial stack (this project's own reference machine,
an RTX 2080 SUPER, sits comfortably in that second tier). **Illustrative silicon this stage's inputs and
outputs touch:** on the sensor side, a laser driver/TIA (transimpedance amplifier) front end per beam, an
FPGA or small SoC doing time-of-flight computation and packetization, and a PHY driving the Ethernet
output; on the compute side, nothing beyond the GPU itself and a standard NIC receiving the UDP stream —
this project's kernel work begins after a driver has already turned packets into a `PointCloud` in host
or device memory.

## 3. Installation & integration — putting it on a real robot

Where this code would run: the SAME perception computer the LiDAR driver node runs on (or a downstream
GPU-equipped node it publishes to) — this stage has no independent real-time deadline of its own beyond
fitting inside the 10–20 Hz scan period, so it does not need a dedicated real-time OS partition the way a
motor current loop would (SYSTEM_DESIGN §1.1). **ROS 2 node shape:** a `voxel_downsample_node` subscribing
to `sensor_msgs/msg/PointCloud2` (the real-world analogue of this project's `PointCloud` struct,
SYSTEM_DESIGN §3.6) published by the LiDAR driver, publishing a downsampled `PointCloud2` on a
`~/points_downsampled` topic (or an equivalent QoS-matched output) — a single-responsibility node exactly
matching README "System context"'s upstream/downstream hand-offs (driver → **this stage** → ICP /
clustering / NDT / mapping). **Leaf-size tuning is a per-consumer discipline, not a single global
constant**: an ICP scan-matcher wants a leaf small enough to preserve the geometric features it aligns on
(too coarse and thin structures like table legs or curb edges vanish into a handful of voxels — see
`THEORY.md`'s centroid-vs-normal discussion); a coarse global-planning costmap can tolerate a much larger
leaf for the same input, trading detail for the neighbor-count savings that make its own downstream
search cheap. Real stacks often run this stage TWICE with two leaf sizes off the same raw scan — a fine
one feeding local, precision-sensitive consumers and a coarse one feeding a global costmap — rather than
picking one compromise value. **Latency budget:** must fit inside the LiDAR's own 10–20 Hz scan period,
comfortably under the 50 ms per-scan perception budget SYSTEM_DESIGN §1 gives the whole
"downsample → ground seg → clustering → deskew" box; this project's own measured GPU times (well under
4 ms at 198,534 points) leave enormous headroom inside that budget even before considering that a real
driver's per-scan point count is usually smaller than this project's accumulated-submap sample. **Safe
testing ladder:** this stage reads sensor data and writes a smaller point cloud — it never commands
actuation directly, so the simulation → HIL → bench → free-running ladder (CLAUDE.md §1) applies to
whichever DOWNSTREAM consumer eventually turns its output into motion, not to this node in isolation;
bring-up here is "does the downsampled cloud look right on a recorded or live bag," checked visually
(this project's own `original_topview.ppm`/`downsampled_topview.ppm` artifacts are exactly that check, in
miniature) before wiring the output to anything that moves.

## 4. Business & regulatory context

*Didactic orientation only — **not** procurement, legal, or compliance advice (SYSTEM_DESIGN.md item 6).
Section dated 2026-07-11.*

Every company shipping a LiDAR-equipped mobile robot, AV, or drone needs SOME form of this stage —
voxel/grid downsampling (or an equivalent, like a learned point-sampling network) is close to universal
in production LiDAR perception stacks, because the near-field-oversampling problem `THEORY.md` derives
is a property of the SENSOR, not of any one company's algorithm choices. The commercial and open-source
landscape spans **sensor vendors** (Velodyne/Ouster/Hesai/Livox-class companies, whose SDKs often ship a
basic downsampling filter alongside the raw driver), **perception-stack/middleware providers** (PCL and
Open3D as the dominant open-source point-cloud libraries this project's README names directly; NVIDIA's
cuPCL/Isaac-class GPU-accelerated perception offerings for anyone who has standardized on NVIDIA compute),
and every **robotics/AV integrator** building its own perception pipeline in-house on top of one of those
foundations. Getting this stage wrong in either direction has a real, quantifiable cost: too coarse a
leaf silently discards the geometric detail a scan-matcher or obstacle-clustering stage needs (a
downstream localization or collision-avoidance FAILURE, potentially safety-critical, that would be very
hard to trace back to "the voxel leaf was too big" without exactly the kind of `downsample_quality`
measurement this project builds in), while too fine a leaf (or skipping this stage entirely) inflates
every downstream algorithm's compute cost, which on an embedded platform can mean missing the very
real-time deadline SYSTEM_DESIGN §1.1 sets. **Regulatory context is inherited from the sensor and from
whatever downstream system this feeds, not created by this stage itself**: the LiDAR unit itself is
subject to **IEC 60825-1** laser eye-safety classification (project
[`01.18`](../../01-perception-cameras-vision/01.18-depth-completion/THEORY.md)'s "Beam divergence and the
eye-safety power ceiling" section derives WHY beam count cannot simply be scaled up for free under a
fixed eye-safety power budget — a constraint this stage's leaf-size choice cannot loosen or tighten, but
must design around, since a sparser sensor means this stage sees fewer points to begin with); whichever
downstream consumer this project's output feeds into (an AV's obstacle-avoidance stack, say) carries its
own applicable standard — **ISO 26262** functional safety and **UL 4600** for autonomous vehicles,
**ISO 13482** for personal-care service robots — per SYSTEM_DESIGN.md item 6's regulatory map. **Where
this work lives inside a robotics company:** the perception team, specifically whoever owns the LiDAR
driver and the first few pipeline stages before hand-off to the localization/mapping and
planning/navigation teams (SYSTEM_DESIGN item 5) — typical adjacent role titles include perception
engineer, robotics software engineer, and (at companies large enough to split the role) a dedicated
"sensor pipeline" or "point-cloud infrastructure" engineer.
