# 02.15 — Point cloud compression (octree/entropy) for fleet uplink: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is software: there is no physical part *of this codec* to assemble. What follows
instead is the physical carrier it depends on — the **onboard data pipeline** that produces the
bytes this codec compresses, and the **uplink path** those compressed bytes travel.

- **Where the input comes from.** A real deployment's occupancy-octree input is not raw sensor
  output — it is a *cleaned* local map, the product of upstream perception (domain 01/02) and, on
  a fleet vehicle, specifically **02.13 (dynamic point removal)**: static-map construction removes
  moving objects (people, forklifts, other robots) before anything gets compressed and uploaded, a
  privacy- and quality-relevant step discussed again in §4 below.
- **The physical construction this depends on**, concretely: the onboard compute module (§2) is
  bolted or DIN-rail-mounted inside the robot's electronics bay, connected to its sensors over
  MIPI/USB/GigE (per SYSTEM_DESIGN.md §6.1's sensor-suite block) and to its radio module over
  PCIe/USB/M.2, with the whole assembly needing the same engineering care as any embedded compute
  install: vibration-rated mounting (a warehouse AMR's wheels transmit real shock into the chassis),
  conformal coating or an IP-rated enclosure if the robot operates outdoors or in wet areas, cable
  strain relief at every connector (a loose antenna pigtail is one of the most common field
  failures in fleet radios), and EMI shielding between the high-current motor-drive wiring
  (SYSTEM_DESIGN.md §6.1's actuation chain) and the compute/radio module's sensitive analog
  front-ends — motor PWM switching is a broadband noise source that can desensitize a nearby
  cellular or Wi-Fi radio if the two are not separated or shielded.
- **What breaks in the field, and why.** Antenna connectors (SMA/U.FL) are the single most common
  point of failure in a fleet radio path — vibration works them loose over months of operation,
  degrading signal quality gradually rather than failing outright, which makes it a classic
  "why did uplink bandwidth quietly get worse" support ticket. Storage media (§2) wears out under
  sustained write load if map tiles are logged locally before upload (flash write-endurance is a
  real, finite budget on an embedded SSD/eMMC part — fleet software should log conservatively and
  rotate old tiles, not accumulate them indefinitely).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-12. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

This codec's ENCODE side is cheap enough (measured **7.8 ms** for a 200,000-point tile on an RTX
2080 SUPER — see `demo/expected_output.txt`'s `GATE timing`) that it does not, by itself, dictate
compute-tier choice; it rides on whatever GPU/compute the robot already carries for perception and
planning. The choice below is really "what compute does a mapping-capable AMR/AV carry anyway,"
with this codec as one more (light) consumer of it.

| Tier | Illustrative part | Role here |
|------|--------------------|-----------|
| **Hobby/prototype** | Raspberry Pi 5 + a USB LiDAR (e.g. RPLidar-class) | No real GPU — this codec's CPU reference path (`reference_cpu.cpp`) is the honest fallback; encode would run seconds, not milliseconds, but correctness does not change. |
| **Research/mid-tier** | NVIDIA Jetson Orin NX/AGX (this repo's typical edge-GPU reference) | Runs the full CUDA pipeline taught here; shares GPU time with perception/planning workloads already running onboard. |
| **Industrial fleet AMR** | x86 industrial PC + discrete RTX-class GPU (this project's own dev/test target, an RTX 2080 SUPER) | Comfortable headroom; encode is a negligible fraction of the compute budget next to SLAM/perception. |
| **Radio (uplink path)** | Indoor: 802.11ac/ax Wi-Fi module (e.g. an M.2 Wi-Fi 6 card). Outdoor/wide-area: an LTE/5G IoT modem module (e.g. a Quectel-class M.2/mini-PCIe cellular modem) with a data-plan SIM. | Carries the compressed bytes this codec produces; see §4 for the data-cost arithmetic this compression exists to reduce. |
| **Onboard storage (staging before upload)** | Industrial eMMC (32–128 GB) or a small NVMe SSD, power-loss-protected if available | Buffers map tiles between generation and successful upload — a real robot uploads opportunistically (best signal, off-peak), not necessarily the instant a tile is ready. |

Compute is virtually never the bottleneck for THIS component specifically; the RADIO and the DATA
PLAN are — which is exactly why a compression project earns its place in a fleet stack even on
capable hardware.

## 3. Installation & integration — putting it on a real robot

- **Where this runs.** On the SAME onboard compute that runs perception/mapping/SLAM (not a
  separate box) — the codec consumes a map tile that already exists in that process's memory, so
  running it as a library call inside the mapping node (rather than a separate process with an IPC
  hop) is the natural integration, avoiding an unnecessary serialize/deserialize round trip for
  data that is about to be serialized anyway.
- **ROS 2 node/topic shape** (illustrative — this repo's message-struct convention,
  SYSTEM_DESIGN.md §3.6): the mapping node that already publishes a `PointCloud`-shaped local map
  would additionally publish (or, more realistically, write to a local queue consumed by) a
  `CompressedMapTile` message: `{ stamp: float64, tile_id: uint64, depth: uint8, huffman_table:
  bytes, packed: bytes, num_points: uint32, aabb_min: float32[3], aabb_extent_m: float32 }` — the
  Huffman table travels WITH each tile (this project's canonical-Huffman build is per-tile, not a
  fixed shared codebook, so the decoder needs the table to decode that specific tile; a production
  system MIGHT instead standardize a small set of pre-trained tables per environment class to skip
  shipping a fresh table every time — a real engineering trade this project deliberately keeps
  simple, named honestly).
- **Format/versioning discipline.** A production uplink format needs a version byte and a documented
  schema-evolution policy from day one, because the robot fleet and the cloud decoder are almost
  never on the exact same software version simultaneously during a rolling deployment — an
  unversioned format makes that upgrade window a landmine. This project's own committed sample
  format already carries this discipline in miniature: `data/README.md`'s binary layout opens with
  an explicit 8-byte magic number (`b'PCFU0001'`, the trailing digits a de facto version tag) — a
  real fleet wire format should extend that same explicit-tag pattern to the compressed tile format
  itself (the illustrative `CompressedMapTile` message sketched above would need its own version
  field for the same reason).
- **Decode-side infrastructure.** The cloud/fleet-management side (this project's sibling,
  **05.18 — map streaming/compression for robot fleets**, by name) is where decode, map merging
  across robots, and long-term storage happen; this project teaches the ROBOT-side encode half of
  that pipeline specifically. A production decode service would run the block-wise parallel decode
  variant `THEORY.md` "The GPU mapping" documents (not implemented here) to keep pace with an
  entire fleet's simultaneous uploads.
- **Safe hardware-testing ladder — N/A because** this component never commands an actuator: it
  consumes an already-built map and produces bytes for a radio. It has no motion output, no
  control-loop authority, and no safety-relevant real-time deadline (its only "budget" is the
  uplink batching cadence, an economic constraint, not a safety one — see the `timing` gate). The
  simulation → HIL → bench → free-running ladder (CLAUDE.md §1) applies to the SENSORS and
  ACTUATORS this map ultimately serves, not to this codec itself.

## 4. Business & regulatory context

**Who needs this, and why it is a real line item, not a nice-to-have.** Any company operating more
than a handful of mapping-capable robots hits this problem quickly: SYSTEM_DESIGN.md §5.2 names
"map/data pipelines at fleet scale" as a **fleet-operations**-phase concern — the phase that "lasts
5–10× longer than all the others and usually decides profitability." Warehouse AMR fleets
(hundreds of robots, each remapping its zone regularly as inventory/layout changes), autonomous
survey/mapping vehicles, and any AV mapping fleet building or refreshing HD maps all face the same
math: a raw point cloud is enormous relative to a typical shared wireless or cellular data budget
(THEORY.md's problem statement works the raw-bytes number; PRACTICE.md's own illustrative arithmetic
below works the fleet-scale dollar cost).

**The cellular data economics, worked through with this project's own measured number** (all rates
illustrative, dated 2026-07-12, verify current before relying on them — a business/IoT cellular
data plan commonly prices somewhere in the **$0.50–$3 per GB** range depending on volume tier and
region, sometimes with a monthly per-device base fee on top): this project's own demo measured a
**12.2×** end-to-end compression ratio on a structured 200,000-point map tile at the sweep's
canonical depth (`rd_curve.csv`). A fleet of 50 robots each uploading 20 map tiles a day — the
`[info] fleet arithmetic` line the demo itself prints — moves from **2.24 GB/day raw** to
**0.18 GB/day compressed**, a saving of **~2.05 GB/day**, or roughly **60 GB/month** for that one
fleet. At an illustrative $1/GB cellular data rate, that is on the order of **$60/month** in avoided
data cost for a fleet this size — a small number for 50 robots, but one that scales linearly with
fleet size and tile-upload frequency, and that is BEFORE counting the operational value of shorter
upload windows (a robot on a metered or congested link finishes its uplink faster, freeing radio
time for teleoperation/safety traffic) — the batch-vs-stream policy question every fleet-ops team
eventually asks: upload every tile immediately (freshest data, worst for bursty bandwidth) or batch
and upload during low-traffic windows (cheaper, staler)? This project's compression makes EITHER
policy cheaper, but does not itself decide between them.

**On-robot storage tiers.** A robot that cannot upload immediately (out of coverage, link
saturated) needs to buffer tiles locally — §2's onboard eMMC/SSD tier — and a real fleet-ops
policy needs a retention/eviction rule for that buffer (oldest-first? highest-value-tile-first?) —
an operational decision this project does not make but whose STORAGE COST it directly reduces
(a compressed tile buffer holds far more history in the same flash budget).

**Commercial and open-source players.** Cloud robotics/fleet-management platforms (offered by
major robotics and cloud vendors) commonly include a map-uplink/compression layer as part of their
fleet stack; on the open-source side, Draco (Google) and PCL's octree compression are the
closest widely-used building blocks (THEORY.md "Where this sits in the real world" names both).

**What getting it wrong costs.** Undercompressed uplink either burns real money (metered cellular
data, at fleet scale) or, worse, silently degrades fleet coordination: a robot that cannot get its
map update through in time is operating on stale shared-map information, a real safety-adjacent
quality issue for any fleet that relies on shared occupancy for collision avoidance between
robots — though the compression LOSS itself is what this project's `distortion_bound` gate exists
to keep provably bounded, so "compression breaks the map" is not the failure mode; "the uplink
link degrades and nothing gets through in time" is (a link-budget/scheduling problem this project's
`timing` gate speaks to, but does not solve).

**Data retention and privacy — a one-paragraph orientation.** A LiDAR or camera-derived map, even a
"static" one, can incidentally contain traces of people who were nearby when the underlying scan was
taken — this repo's person-perception framing (CLAUDE.md §1, §8: collaborative safety, never
individual tracking or surveillance) applies here too: the **02.13 (dynamic point removal)**
upstream step this project's README "System context" names is not just a mapping-quality step, it
is also the point where transient, potentially person-associated points are removed BEFORE anything
is compressed, uploaded, or retained — a real privacy-by-design consideration, not an afterthought.
Regions with data-protection regulation (e.g. GDPR-style regimes) treat facility maps that could
reveal personal presence/movement patterns as data warranting a documented retention policy and
purpose limitation; this is **orientation, not legal advice** (SYSTEM_DESIGN.md §6.2's label
applies verbatim here — no specific regulatory regime is cited as applicable, because that
determination depends on jurisdiction, deployment context, and what the fleet actually retains).

**Where this work lives inside a robotics company.** Per SYSTEM_DESIGN.md §5.1's org map: this is
**fleet operations** territory (owning team), consuming input from **perception**
(01/02, the mapping pipeline) and handing off to **simulation & tools / cloud infrastructure**
(the decode-side, map-merging service — 05.18's territory) and **embedded** (the on-robot radio
and storage integration, §2–§3 above). Typical role titles: fleet infrastructure engineer, robotics
systems engineer (uplink/telemetry focus), or — at a company large enough to have one — a dedicated
mapping-infrastructure team straddling perception and cloud engineering. Regulatory/compliance
involvement here is usually about data retention and privacy policy (above), not the safety-
certification standards SYSTEM_DESIGN.md §6.2 catalogs for actuating systems — this component
never actuates anything.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
