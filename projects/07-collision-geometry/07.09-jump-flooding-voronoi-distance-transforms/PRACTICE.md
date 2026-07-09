# 07.09 — Jump-flooding Voronoi/distance transforms (easy, visual, useful): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> This is planning-layer software; its physical carrier is the mobile robot whose costmap it
> serves — sections 1–2 teach that carrier's sensing-to-clearance chain.

## 1. Building it — construction of the robot/part

No part is machined for a distance transform — the physical subsystem it belongs to is the **AMR
navigation sensor suite** whose measurements become the obstacle cells this code eats:

- **The sensing ring.** A typical warehouse AMR carries one or two 2-D safety LiDARs mounted low
  (ankle height) in cutouts at opposing corners for 360° coverage, plus depth cameras angled down
  for overhangs and floor debris. Construction realities that shape the data: scanners need an
  unobstructed sweep plane (bumper styling is negotiated around optics), mounting brackets must
  hold alignment through shock (a 1° pitch error tilts the scan plane centimeters over meters),
  and lens/window cleanliness is a scheduled-maintenance item — a dusty window *is* a phantom
  obstacle field, which the costmap then dutifully inflates.
- **Why the safety LiDAR is special hardware:** certified safety scanners (Type 3 devices under
  IEC 61496) implement their own hardwired protective zones independent of software — the robot
  stops even if every line of code here is wrong. The clearance field this project computes serves
  *planning* comfort and efficiency on top of, never instead of, that certified layer — the
  division that lets uncertified GPU code exist on a safety-rated machine at all.
- **Compute mounting** (where this kernel runs): the same enclosure/thermal/EMI story as project
  33.01 PRACTICE §1 — deliberate cross-reference rather than repetition.

What breaks in the field: dirty optics (phantom obstacles), bracket misalignment after collisions
(scan plane tilt → floor readings become walls), reflective/black surfaces (dropouts → missing
obstacles — the reason costmaps decay evidence over time rather than trusting single scans).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-08. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

The chain that produces and consumes this project's field, tiered:

| Piece | Illustrative choices (2026) | Relevance to this project |
|---|---|---|
| 2-D LiDAR — hobby | 360° triangulation scanners of the ~US$100 class (12 m range) | Produces the occupancy grid a student project would feed in |
| 2-D LiDAR — research | 270° ToF scanners, ~US$2–5k class, cm-accuracy | The standard costmap source on lab AMRs |
| 2-D LiDAR — industrial | Certified safety scanners (IEC 61496 Type 3), ~US$3–8k class | Provides the *certified* protective layer; our field is the uncertified planning layer above it |
| Depth cameras | Stereo/ToF modules, ~US$300–600 class | Adds 3-D obstacles projected into the grid |
| Compute | Jetson Orin Nano/AGX class or x86+RTX (the reference machine: RTX 2080 SUPER) | A 1024² field in ~4–6 ms fits any of these; on integrated-memory Jetsons the costmap→GPU copy is free |
| The grid itself | 25–100 m² facility maps at 2.5–10 cm/cell → 10⁵–10⁷ cells | Sets the W×H this kernel must sustain at 5–20 Hz |

No actuation silicon appears in this BOM — the project reads maps and writes fields; motion is a
consumer's business (N/A stated, not padded).

## 3. Installation & integration — putting it on a real robot

- **Process shape:** in a ROS 2 stack this lives inside the costmap pipeline — practically, as a
  GPU-accelerated costmap layer plugin (Nav2's `costmap_2d` layer API) or a standalone node
  subscribing `nav_msgs/OccupancyGrid` (or the raw obstacle buffer) and publishing the clearance
  field for planner nodes. In-process is preferable: a million-cell float field at 20 Hz is
  80 MB/s of message traffic if serialized naively — the classic argument for zero-copy /
  same-process composition (project 32.06's territory).
- **Real-time constraints:** costmap ticks are soft-real-time (5–20 Hz); a missed tick degrades
  planning smoothness, not safety — safety lives in the certified scanner layer (§1). That is
  *why* this workload can ride the GPU today while kHz loops cannot (see 33.01 PRACTICE §3).
- **Calibration & bring-up:** the field is only as good as the grid — bring-up validates the
  sensor→map chain (extrinsic calibration of scanners to `base_link`, map resolution choice,
  occupancy thresholds) before anyone trusts clearance numbers. A practical acceptance test used
  on real deployments: park the robot a tape-measured distance from a wall and compare the
  field's value at the robot's cells against the tape.
- **The testing ladder** (§1 caveat): this code moves nothing, but its consumers do — validate in
  simulation (feed synthetic grids like this demo's), then on logged real maps (rosbag replay),
  then on the vehicle with the certified protective layer active and verified *first*. E-stop
  chains and speed zones belong to that certified layer; N/A to this library beyond the duty not
  to undermine it.
- **Toolkit pinning / regression:** identical story to 33.01 PRACTICE §3 — with one advantage:
  this project's outputs are integer-exact, so a regression test can demand *bit-identical* fields
  across driver updates, the strongest regression signal software can have.

## 4. Business & regulatory context

- **Who needs it:** every mobile-robot company (warehouse AMRs, service robots, sidewalk delivery,
  AGV retrofits) computes clearance fields continuously; drone and manipulator companies use the
  3-D cousins (ESDFs) for the same purpose. Faster fields → denser sampling in local planners →
  smoother motion in clutter — a visible product quality, not an internal nicety.
- **The players:** Nav2/ROS 2 (open standard for AMR stacks), NVIDIA Isaac (nvblox: GPU distance
  fields as a product), scanner vendors (whose certified zones are the safety product), and the
  AMR vendors themselves (fleet-scale costmap tuning is in-house lore). Build-vs-buy: use
  Nav2/nvblox layers where they fit; hand-roll (this project's skill) when the field must fuse
  into custom planners or exotic grids — SYSTEM_DESIGN item 5's criteria apply cleanly.
- **Cost of getting it wrong:** an over-optimistic clearance field (distance too large — exactly
  JFA's error direction, which is why the bound is checked every run) lets a planner clip a rack
  at 1.5 m/s: damaged goods, damaged robot, an incident report that pauses a fleet. The mitigation
  stack: certified protective layer (hard stop), conservative inflation radii (eat the error
  bound), and exact-field regression tests (the §3 bit-identical trick).
- **Regulatory:** service/industrial mobile robots orient to ISO 3691-4 (driverless industrial
  trucks) and ISO 13482 (service robots); the protective function is delivered by certified
  devices (IEC 61496 scanners) and certified stop circuits — planning software like this is
  uncertified supporting software around them (SYSTEM_DESIGN item 6 orientation map; the SOUP
  framing from 33.01 PRACTICE §4 applies verbatim).
- **Owning team:** navigation within an autonomy group (titles: navigation/motion-planning
  engineer, robotics software engineer — mapping); adjacent: perception (feeds the grid),
  functional safety (owns the certified layer and reviews every claim this layer makes near it).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
