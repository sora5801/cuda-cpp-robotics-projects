# 02.05 — KD-tree or LBVH construction + KNN/radius search on GPU: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

*Sections dated 2026-07-11. All parts and figures below are **illustrative examples, never
endorsements**; verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project is pure software — there is no physical part to assemble. The honest physical carrier is
the **perception compute module** the LBVH engine would run on: a GPU-equipped compute board bolted
into a robot's electronics bay, fed by a LiDAR over Ethernet or a proprietary interconnect, and wired
into the robot's compute/power/thermal envelope like any other embedded card (PRACTICE.md §2 below
names the actual hardware tiers).

**Where neighbor queries actually burn a real robot's compute budget — profile-first, not guess-first.**
The single most common mistake a perception team makes with a project like this one is optimizing the
wrong stage. This project's OWN measured numbers make the point: on the reference GPU, the FULL LBVH
rebuild (Morton + sort + radix-tree + AABB, all four stages) costs **~4–5 ms** for a ~200k-point scan —
a small fraction of a 50–100 ms frame budget. The QUERY stage is where the budget actually gets spent,
and how much depends entirely on how many queries a downstream consumer issues: normal estimation
(02.09) or FPFH (02.10) querying EVERY point in the cloud is `O(N)` queries per scan — at this
project's measured ~11,600 BVH-radius queries/sec or ~70,500 BVH-KNN queries/sec, a 200k-point cloud's
full self-query pass costs **~17 ms (radius) to ~3 ms (KNN)** — now a MEANINGFUL fraction of the frame
budget, and the number a real profiling pass (Nsight Systems, not intuition) would actually catch. The
practice discipline this project's own numbers teach: measure BUILD and QUERY separately, because a
consumer's query PATTERN (self-query-every-point vs. a handful of targeted lookups) changes which one
dominates by orders of magnitude — CLAUDE.md's "never fabricate a benchmark claim" extends naturally to
"never guess which stage is slow without measuring."

**Rebuild-vs-refit for streaming scans — a documented trade this project does NOT take.** This project
rebuilds the ENTIRE tree from scratch every scan (README "Limitations"). A production streaming system
processing a continuous LiDAR feed has a real alternative: REFIT — keep the tree's TOPOLOGY (which
points share which subtree) from the previous scan and only recompute AABBs (Stage 4 alone, `~0.3 ms`
measured here) when points move a small amount frame-to-frame (e.g., a slowly-drifting map, or a
tracked object's local neighborhood). Refitting is far cheaper than rebuilding but degrades over time
as topology becomes stale relative to the data's true spatial distribution (a point that moved far
enough may now belong in a very different subtree than its stale topology assigns it) — production
systems typically refit for several frames, then trigger a full rebuild on a schedule or when a
quality metric (e.g., measured AABB "bloat" versus a fresh build) crosses a threshold. This project's
full-rebuild-every-scan choice is the SIMPLE, always-correct baseline; refit is named here, not
implemented, as the natural next engineering step (README Exercise territory).

## 2. Real hardware — chips, parts, illustrative BOM

| Tier | Compute | Example parts (illustrative, verify current) | Notes |
|------|---------|------------------------------------------------|-------|
| Embedded / AMR | Jetson Orin NX / AGX Orin class SoC | NVIDIA Jetson Orin NX 16GB module (~$400–700 module price class) | Shares one SoC's GPU across perception, planning, and control — this project's ~4–5 ms build cost matters more here, where GPU time is contended, than on a discrete-GPU AV stack. |
| Desktop / research | x86 + discrete RTX-class GPU | This project's own reference machine: RTX 2080 SUPER (sm_75, 8 GB) — a 2019-era consumer card, illustrative of the "any CUDA-capable desktop GPU" floor this repo targets (CLAUDE.md §5) | The numbers in `demo/expected_output.txt` were measured here; a newer card (Ada/Blackwell-class) would be faster but the RELATIVE hash-vs-tree story should hold. |
| Automotive / production AV | Multiple discrete automotive-qualified SoCs (e.g., NVIDIA DRIVE-class, Qualcomm Ride-class) | Automotive-grade parts carry AEC-Q100 qualification and multi-year availability guarantees consumer parts do not | An AV's perception stack runs many neighbor-search-consuming stages (02.06, 02.09, 02.10, clustering, tracking) CONCURRENTLY, each issuing its own query load against shared point clouds — the aggregate query rate this project's per-stage numbers only hint at. |
| Data-center (offline map-building, not on-robot) | Multi-GPU server (e.g., 4–8× data-center-class GPUs) | Used for building/refining large-scale maps offline, not for the real-time on-robot path this project targets | Named for completeness; N/A to this project's real-time scope. |

**Sensor side (context only — this project consumes a point cloud, it does not produce one):** the
committed sample's scan pattern (16 beams, 2,560 azimuth steps) mirrors a mid-tier mechanical spinning
LiDAR (e.g., Velodyne/Ouster-class hardware, illustrative price class $2k–$12k depending on channel
count and grade) — see [`11.01`](../../11-sensor-sim-digital-twins/11.01-gpu-lidar-simulator/PRACTICE.md)
for the sensor hardware itself.

## 3. Installation & integration — putting it on a real robot

**Where this runs.** The perception compute node — the SAME machine running 02.01–02.20's other LiDAR
stages, typically a dedicated perception SoC/GPU on an AV or a shared Jetson-class module on an AMR
(SYSTEM_DESIGN.md's reference-robot block diagrams). Real-time constraints favor a Linux RT-patched
kernel or a partitioned/QoS'd stock kernel; this project's own demo has NO real-time guarantees (a
teaching CLI, not a scheduled task) — a production deployment would wrap the build+query pipeline in a
fixed-period ROS 2 timer callback or a dedicated perception-graph node with a hard deadline monitor.

**ROS 2 node/topic shape (illustrative).** A `neighbor_search_node` subscribing to
`sensor_msgs/PointCloud2` (or this repo's internal `PointCloud` message shape, SYSTEM_DESIGN.md §3.6),
publishing... typically NOTHING directly — this is an in-process LIBRARY/SERVICE other nodes call into
(e.g., as a shared component inside a larger perception node, or via a ROS 2 service/action for
cross-process queries), not a topic-publishing node of its own. The interesting integration point is
the QUERY API contract: callers need to know whether they get RADIUS or KNN semantics, at what `K`/`r`,
and whether results are the point's INDEX (fast, requires the caller to already hold the cloud) or
the full XYZ (self-contained, costs a copy) — this project's own `radius_search_bvh` /
`knn_search_bvh` signatures (returning indices into the caller-owned `xyz` array) are exactly this
design choice, made explicit.

**Consumer query patterns, named concretely.** 02.09 (normal/curvature estimation) issues ONE
radius-or-KNN query PER POINT in the cloud — an `O(N)` self-query pass, exactly the workload this
project's own README "Practice §1" profiling numbers analyze. 02.10 (FPFH descriptors) issues query
pairs at TWO different radii per point (a smaller radius for the base normal, a larger one for the
histogram) — double the query volume of 02.09 for the same cloud. 02.06 (ICP) issues one
NEAREST-NEIGHBOR (K=1) query per source point, PER ITERATION (typically 10–50 iterations to converge)
— a very different amortization story: the SAME tree is reused across every iteration, so paying the
~4–5 ms build cost once and then issuing tens of millions of K=1 queries against it is the
dominant real workload ICP would present to this project's engine.

**Memory-budget arithmetic at automotive point rates.** A production automotive LiDAR (or an array of
several) can sustain ~1–2 million points/second aggregate. At `2N−1` nodes of 40 bytes each (this
project's `LbvhNode` layout) for `N = 500,000` points/scan (an illustrative single-frame count for a
higher-resolution automotive unit), the tree alone costs `(2·500,000−1)·40 ≈ 40 MB` — resident for one
scan's lifetime, rebuilt every ~50–100 ms. At 10 Hz, that is a sustained **~400–800 MB/s of GPU memory
CHURN** just for tree allocation/deallocation (not counting the point cloud itself, query buffers, or
any other stage's memory) — a real budget line a systems engineer sizing a perception GPU's memory
bandwidth and allocator strategy would need to account for, and the kind of arithmetic PRACTICE.md
sections exist to make concrete rather than hand-waved.

**Safe hardware-testing ladder.** N/A in the direct sense — this project's output never commands
actuation; it is a read-only query engine. Its INDIRECT safety relevance is entirely about correctness
feeding downstream planning/control stages (a wrong neighbor set silently degrading a normal estimate,
which degrades a ground-plane fit, which degrades an obstacle boundary) — the standard "garbage in,
garbage out" case for any perception infrastructure component. As with every project in this repository:
simulation-validated only, not safety-certified (CLAUDE.md §1); any real deployment sits behind the
downstream stage's own testing ladder (sim → HIL → bench → tethered → free-running), not this
project's own.

## 4. Business & regulatory context

**Who needs this, and why it is an "infrastructure" capability, not a product feature.** Every company
building a LiDAR-based perception stack — AV companies (Waymo-, Cruise-, Zoox-class and their tier-1
supplier ecosystem), warehouse/logistics AMR makers (Locus-, 6River-, Fetch-class), and any research lab
doing point-cloud SLAM or manipulation perception — needs fast neighbor search SOMEWHERE in their
stack. Almost none of them BUILD it as a standalone product: it is infrastructure every perception
feature (registration, segmentation, descriptor extraction, tracking) is quietly built on top of, and
it is invisible until it is too slow — exactly the "unsung layer" framing SYSTEM_DESIGN.md item 5
gives the perception-infrastructure team inside a robotics company's org chart. **Main commercial and
open-source players:** PCL (open-source, CPU, the default almost every robotics team reaches for
first); NVIDIA's cuML/FAISS-adjacent GPU libraries and internal perception SDKs (Isaac Perceptor,
DriveWorks) for GPU-native production paths; FLANN (the library PCL's KdTreeFLANN wraps) as the
long-standing CPU reference implementation.

**What getting it wrong costs.** Silently — not catastrophically. A slow or approximate neighbor-search
engine does not usually crash a robot; it degrades the QUALITY of every downstream stage that depends
on it (worse normals → worse ground-plane fits → worse obstacle boundaries; slower ICP convergence →
looser real-time margins; a missed rebuild-vs-refit staleness bug → a subtly wrong map). The cost shows
up as accumulated perception error and missed real-time deadlines, which show up downstream as
disengagements, false positives/negatives in obstacle detection, or — worst case — a planning stack
acting on stale/wrong geometry. This is precisely the class of "infrastructure debt" that is cheap to
under-invest in early and expensive to untangle once six other teams depend on it.

**Regulatory path.** This project itself has no direct regulatory surface (it produces no actuation
commands and makes no safety claims) — its regulatory relevance is entirely INHERITED from whatever
downstream perception/planning/control stack consumes its output. Per SYSTEM_DESIGN.md item 6's
regulatory map: an AV stack built on top of a neighbor-search engine like this one falls under
ISO 26262 (functional safety) and, in the US, the emerging UL 4600 autonomous-vehicle safety-case
framework; a collaborative service robot or AMR falls under ISO 13482; neither framework certifies a
component like this one in isolation — they certify the SYSTEM, and a component like this one would
need to be shown, as part of that system argument, to meet whatever timing/correctness requirements
the downstream safety case allocates to it. This project makes no such certification claim (CLAUDE.md
§1) — it is an orientation pointer, not compliance guidance.

**Where this work lives inside a robotics company.** Perception infrastructure / "core perception" —
adjacent to, but organizationally distinct from, the feature teams (segmentation, registration,
tracking, mapping) that consume it (SYSTEM_DESIGN.md item 5's org map). Typical role titles: perception
infrastructure engineer, GPU/systems engineer embedded in a perception org, or (at smaller companies)
simply "the perception engineer who happens to own the spatial-index library everyone else imports."
Adjacent teams: simulation/tools (who need the SAME structure for synthetic-data generation and
offline evaluation), and the ML/data team (whose learned point-cloud models often need neighborhood
pooling — the SAME query this project answers, feeding a different consumer).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
