# 23.01 — GPU costmaps: inflation, raytrace clearing, multi-layer fusion: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> This project's physical carrier is the **warehouse AMR** it is scoped around (README's reference
> robot) — sections 1–2 teach that machine's sensing and compute, section 3 how this exact loop
> would be wired onto it, section 4 who buys and regulates it.

## 1. Building it — construction of the robot/part

No part is machined for a costmap or a DWA planner — the physical subsystem this project belongs to
is the **AMR's sensing ring and its onboard compute enclosure**, the same carrier 07.09's PRACTICE.md
teaches (that project computes the clearance field this one's inflation layer is a cousin of):

- **The sensing ring.** A typical warehouse AMR (SYSTEM_DESIGN.md §2.1) carries one or two 2-D LiDARs
  mounted low (ankle height) in cutouts at opposing corners for 360° coverage — the direct physical
  analog of this project's simulated 360-beam scan. Construction realities that shape what the
  raytrace kernel actually receives: scanners need an unobstructed sweep plane (bumper styling is
  negotiated around the optics), mounting brackets must hold alignment through shock (a 1° pitch
  error tilts the whole scan plane centimeters over meters — a phantom wall or a missed real one),
  and lens/window cleanliness is scheduled maintenance (a dusty window looks, to the raytrace kernel,
  exactly like a wall the inflation layer will dutifully swell — the software cannot tell "the sensor
  is dirty" from "there really is an obstacle here").
- **Why the SAFETY scanner is separate hardware, not this pipeline.** Certified safety-rated LiDARs
  (IEC 61496 Type 3 devices) implement their own hardwired protective zones INDEPENDENT of any
  software — the drive stops on a zone violation even if every kernel in this project has a bug. This
  project's costmap serves PLANNING comfort and efficiency ON TOP OF that certified layer, never
  instead of it — the division that lets uncertified, sim-validated-only GPU code like this exist on
  a machine that shares floor space with people at all (§4 revisits this distinction).
- **Compute mounting** (where these kernels physically run): the same enclosure/thermal/EMI story as
  08.01 PRACTICE.md §1's controller compute — deliberate cross-reference rather than repetition; a
  costmap-and-local-planner GPU box lives in the same class of sealed, vibration-isolated compute
  bay as a controller's, just with more DRAM for the grid.

What breaks in the field: dirty optics and bracket misalignment (both discussed above), and — specific
to a MOVING costmap consumer rather than a static field — wheel odometry drift accumulating between
map re-localizations, which shows up as the costmap's static layer appearing to "slide" relative to
what the LiDAR actually sees, degrading exactly the byte-exact static/obstacle agreement this
project's simulated world enjoys by construction.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

The chain that produces this project's inputs and would run its outputs, tiered:

| Piece | Illustrative choices (2026) | Role in this project's loop |
|---|---|---|
| 2-D LiDAR — hobby | 360° triangulation scanners, ~US$100 class, ~12 m range | Produces a scan a student build would feed the raytrace kernel |
| 2-D LiDAR — research | 270°+ ToF scanners, ~US$2–5k class, cm-accuracy | The standard costmap source on lab AMRs — closest real analog to this demo's simulated scan |
| 2-D LiDAR — industrial/SAFETY | Certified safety scanners (IEC 61496 Type 3), ~US$3–8k class | The CERTIFIED protective layer (§1) — separate hardware, separate stop path, never this project's job |
| Depth cameras | Stereo/ToF modules, ~US$300–600 class | Overhang/low-obstacle coverage a flat 2-D LiDAR misses (this project's obstacle layer is 2-D only, stated as scope) |
| Compute for the pipeline | Jetson Orin class SoC, or x86 + entry/mid dGPU (reference machine here: RTX 2080 SUPER) | This project's four kernels: <1 ms measured combined at 256x256/4096 samples |
| Wheel drives + encoders | 2–4 BLDC hub or geared motors with off-the-shelf velocity-loop drives | Consumes the `Twist` this project's DWA emits every tick |
| Fleet/site compute | A facility map server, typically off-robot | Supplies the static layer's source map — not modeled in this desktop demo (data/README.md's synthetic map stands in) |

At this project's grid scale (65,536 cells, ~1 ms per full pipeline pass), any tier above the hobby
LiDAR class comfortably meets the 5-20 Hz costmap / 10-50 Hz local-planner budget on the listed
compute — the bottleneck on a real AMR is far more often the sensing/localization chain than this
GPU work.

## 3. Installation & integration — putting it on a real robot

**This project's output is a velocity command — the README §1/CLAUDE.md §1 caveat applies at full
strength: everything here is sim-validated only; nothing below is a license to actuate.**

- **Process shape (ROS 2).** This maps onto TWO adjacent, well-known Nav2 extension points, not one:
  the costmap pipeline (raytrace + inflation + fusion) is the GPU-accelerated shape of a Nav2
  `costmap_2d` layer plugin (subscribing scan/odometry topics, publishing/maintaining an
  `nav_msgs/OccupancyGrid`), and the DWA scoring kernel is the GPU-accelerated shape of a `DWB`
  (Nav2's DWA-family local planner) critic/scorer running inside the planner server's control loop,
  publishing `geometry_msgs/Twist` (or `TwistStamped`) to the base controller. This project's single
  demo process fuses both roles for teaching clarity; a real deployment keeps them as two Nav2
  plugins so each can be swapped, tuned, or safety-reviewed independently.
- **The certified layer stays OUTSIDE this software, always.** This is the same honest boundary
  07.09's PRACTICE.md draws for its distance field: whatever this pipeline computes is a PLANNING
  comfort/efficiency layer, published alongside — never wired into — the certified safety scanner's
  hardwired protective-field stop path (IEC 61496). A bug in this project's inflation math can, at
  worst, make the robot plan a poor route or stop unnecessarily; it must never be the thing standing
  between the robot and a collision. That job belongs to hardware this project never touches.
- **Real-time constraints, honestly.** Costmap/local-planner ticks are SOFT real-time (5-20 Hz /
  10-50 Hz, SYSTEM_DESIGN.md §1.1); a missed tick degrades navigation smoothness — the robot plans
  against a slightly stale world for one extra tick — not safety, because safety lives in the
  certified layer above. That is precisely why this workload can ride a general-purpose GPU on a
  soft-real-time OS today while the kHz current loops underneath it cannot (33.01 PRACTICE §3's
  reasoning, restated for this layer).
- **Calibration & bring-up.** The costmap is only as good as its inputs: LiDAR-to-`base_link`
  extrinsic calibration, map resolution/origin agreement between the static layer and the live scan,
  and wheel-odometry-to-map alignment (drift here looks exactly like the static/obstacle-layer
  disagreement this project's simulated world never has to face) all need bring-up and verification
  BEFORE trusting the fused costmap. A practical acceptance test used on real deployments: park the
  robot a tape-measured distance from a known wall and compare the costmap's lethal cell against the
  tape — the same spirit as this project's byte-exact GPU-vs-CPU gate, applied to real hardware.
- **The safe hardware-testing ladder** (CLAUDE.md §1):
  1. *Simulation* — this demo, plus adversarial synthetic worlds (README Exercise 3's slalom layout,
     which deliberately trips DWA's local-minima failure mode — better to find that in sim).
  2. *HIL* — the pipeline against a real-time simulated LiDAR/plant on the target compute, with
     deadline monitoring on the 10 Hz tick.
  3. *Bench, tethered/current-limited* — verify the sensor-to-costmap chain visually (does the
     inflation gradient actually appear around a real object placed in front of the scanner?) before
     any wheel authority is granted, and verify the CERTIFIED scanner's independent stop function
     FIRST, separately, with this pipeline's output ignored.
  4. *Free running* — only inside a facility with the certified protective layer active and verified,
     drive limits set conservatively, and a human-operable E-stop in reach.
- **N/A here:** no fieldbus is implemented in this project (a real deployment commands the wheel
  drives over CAN-FD or a vendor's own protocol, SYSTEM_DESIGN.md §6.1); this demo's "actuator" is a
  function call into a simulated differential-drive plant. Stated per contract.

## 4. Business & regulatory context

- **Who needs this capability.** Every mobile-robot company — warehouse/logistics AMR makers, service
  and delivery robot companies, AGV retrofitters — runs a costmap-plus-local-planner pipeline
  continuously, on every unit, for the robot's entire duty cycle; it is as core to the product as the
  drive motors. It is also one of the most-tuned pieces of software in a fleet's lifetime: obstacle
  weights, inflation radii, and velocity limits get adjusted per-site as real-world layouts and
  traffic patterns reveal edge cases the factory tuning missed.
- **The players.** Nav2/ROS 2 (the open-source standard for this exact layer stack — `costmap_2d` +
  `DWB`, this project's direct production analogs, README §11), NVIDIA Isaac (GPU-accelerated
  perception/mapping building blocks a costmap can consume), and every AMR/AGV vendor's in-house
  navigation team (fleet-scale tuning and site-specific behavior are almost always kept in-house —
  SYSTEM_DESIGN.md §5.3's build-vs-buy criteria apply cleanly: navigation quality is usually a
  genuine product differentiator, so teams build/tune it rather than run a stock configuration).
- **Cost of getting it wrong.** An under-inflated costmap (margin too thin) risks a real collision —
  damaged goods, a damaged robot, a safety incident report that can pause an entire fleet pending
  investigation; an over-inflated one (margin too generous, or DWA stuck in a local minimum) makes
  the robot unable to navigate normal aisle widths, directly costing throughput — the same efficiency
  metric that justifies the robot's existence on the floor. The mitigations are architectural, not
  aspirational: the certified protective layer (hard stop, independent of this software, §1/§3), a
  conservative measured inflation radius (this project derives its own from the plant's stopping
  distance, THEORY.md §the-problem), and regression tests that exploit this pipeline's byte-exactness
  (a real deployment can demand bit-identical costmaps across a driver/toolkit update — the strongest
  regression signal software can offer, the same argument 07.09 PRACTICE.md §3 makes).
- **Regulatory.** Driverless industrial trucks and warehouse AMRs are the direct addressees of
  **ISO 3691-4** (safety requirements for driverless industrial trucks); general service robots sharing
  space with untrained people fall under **ISO 13482** (SYSTEM_DESIGN.md §6.2's regulatory map, both
  rows). Neither standard is satisfied by planning software like this project — the protective function
  is delivered by the certified scanner and stop circuits named in §1/§3; this pipeline is
  UNCERTIFIED supporting software running alongside them (the same SOUP — Software Of Unknown
  Provenance — framing 33.01 PRACTICE.md §4 and 07.09 PRACTICE.md §4 both apply to every layer under
  a certified system, this one included).
- **Owning team.** Navigation, inside an autonomy/controls group (titles: navigation engineer,
  motion-planning engineer, robotics software engineer — navigation; SYSTEM_DESIGN.md §5.1);
  adjacent teams: perception (supplies the scans and the prebuilt map this project's static layer
  stands in for), functional safety (owns the certified scanner/stop chain and reviews every claim
  this layer's output makes near it), and fleet operations (the team that actually re-tunes these
  weights per site once the robot is deployed).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
