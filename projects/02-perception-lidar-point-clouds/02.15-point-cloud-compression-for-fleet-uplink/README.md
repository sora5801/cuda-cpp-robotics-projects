# 02.15 — Point cloud compression (octree/entropy) for fleet uplink

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `Point cloud compression (octree/entropy) for fleet uplink`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A robot fleet builds maps, and maps are big: a single 200,000-point local map tile stored as raw
`float32 x,y,z` is 2.3 MiB — too big to upload cheaply, over and over, from every robot in a fleet,
over shared Wi-Fi or metered cellular data. This project builds a two-stage **lossy-then-lossless**
point-cloud compressor, entirely on the GPU: Stage 1 quantizes the cloud into an **occupancy
octree** (one byte per tree node, describing which of its 8 children are occupied — the classic
sparse-3-D representation) built via a genuinely elegant composition of this repo's own
scan/compaction primitive, repurposed to LABEL points with their owning tree node instead of
filtering them; Stage 2 measures the octree byte stream's symbol statistics and entropy-codes it
with a **canonical Huffman** code, built once on the host and applied on the GPU via a
map→scan→bit-scatter pipeline. The demo sweeps four octree depths (`D = 8, 9, 10, 11`) on TWO
committed clouds — a structured warehouse-room map tile and a "pathological" cube of uniformly
scattered points with no surface at all — and measures, honestly, exactly how much (and why)
surface-structured geometry compresses better: **12.2× smaller than raw xyz at D=10 for the
structured cloud, vs. 6.2× for the pathological one** (this project's own measured numbers). Every
stage is verified GPU-vs-CPU bit-exact, and the quantization error is checked against an analytic
bound derived from first principles — every piece of the catalog bullet is fully implemented; no
component is documented-only.

## What this computes & why the GPU helps

The computation is a **two-stage codec**: (1) build a sparse occupancy octree from a Morton-sorted
point array — a STRING problem (finding where sorted codes' shared prefixes change) solved with a
boundary-detection MAP, an exclusive SCAN (repurposed from this repo's own stream-compaction
primitive to LABEL points with their tree node instead of filtering them), and an atomic
SCATTER-reduce (OR-ing child bits into each node's occupancy word); (2) entropy-code the resulting
byte stream with canonical Huffman — a code-length MAP, a SCAN turning lengths into bit offsets, and
a bit-level atomic SCATTER packing the compressed stream. Both stages are fundamentally
memory-bandwidth-bound, embarrassingly parallel MAP+SCAN+SCATTER pipelines over hundreds of
thousands of points/bytes — exactly the shape this repo's GPU patterns exist to teach — with the
ONE genuinely serial piece (canonical-Huffman table construction, and later, decode) correctly left
on the host, because forcing a thread-per-something mapping onto an inherently sequential ≤256-symbol
merge or a variable-length-code bit walk would not be parallelism, it would be theater.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** cross-cutting infrastructure — specifically **fleet-operations map
  streaming**, sitting between a robot's local mapping/perception stack and the fleet's cloud/map
  infrastructure (SYSTEM_DESIGN.md §5.2 names this phase explicitly: "map/data pipelines at fleet
  scale (02.15, 05.18)").
- **Upstream inputs:** a cleaned local map tile — a `PointCloud`-shaped message, already filtered of
  transient/moving points by **02.13 (dynamic point removal)**, this project's named upstream
  neighbor (PRACTICE.md §1 and §4 discuss why that filtering matters for both map quality AND
  privacy).
- **Downstream consumers:** the fleet's map-streaming/cloud infrastructure — this project's named
  sibling, **05.18 (map streaming/compression for robot fleets)** — which decodes, merges maps
  across robots, and serves them to dashboards, other robots, or offline audit tools; a fleet
  running multi-robot coordination (domain 22, e.g. 22.01's swarm simulator) also depends on timely
  shared-map updates reaching every robot, one more reason uplink cost matters.
- **Rate / latency budget:** this codec is NOT on a real-time control loop — it has no Hz
  requirement in the SYSTEM_DESIGN.md §1.1 sense. Its real constraint is economic/operational: encode
  must complete comfortably inside the fleet's **uplink batching cadence** (this project's `timing`
  gate checks the canonical-depth pipeline against an illustrative 5-second budget; measured: **7.8
  ms** — encode is never the bottleneck, the radio link is, PRACTICE.md §1 and §4).
- **Reference robot(s):** warehouse **AMR fleets** (SYSTEM_DESIGN.md §2.1) and **autonomous-vehicle
  mapping fleets** (§2.5) — any reference robot that (a) operates in a fleet and (b) builds and
  shares maps.
- **In production:** MPEG's G-PCC / Google's Draco (context-adaptive arithmetic coding instead of
  flat Huffman, temporal delta coding between updates) or PCL's `OctreePointCloudCompression` —
  THEORY.md "Where this sits in the real world" names both and the specific gap between them and
  this teaching version.
- **Owning team:** **fleet operations / mapping infrastructure** — SYSTEM_DESIGN.md §5.1's org map;
  PRACTICE.md §4 works through the adjacent teams (perception upstream, cloud/simulation-and-tools
  downstream) and typical role titles.

## The algorithm in brief

- **Morton encoding + sort** — quantize each point to a `D`-bit-per-axis grid, bit-interleave into
  one integer code, sort ascending (THEORY.md "The math": "the octree as a prefix tree over Morton
  strings").
- **Level-by-level occupancy-octree construction** — boundary-detection + exclusive scan + atomic
  occupancy scatter, one pass per tree depth (THEORY.md "The algorithm", "The GPU mapping").
- **256-symbol histogram** of the resulting occupancy-byte stream (a parallel atomic count).
- **Canonical Huffman coding** — host-side table build (two independently-implemented,
  provably-convergent constructions — THEORY.md "The math"), GPU-side map+scan+bit-scatter encode.
- **Host-side decode** — a bit-trie Huffman walk, then level-by-level octree expansion — deliberately
  serial; THEORY.md "The GPU mapping" derives exactly why no parallel form exists for this stage.
- **Rate-distortion analysis** — an analytic quantization-error bound (derived, not measured) gated
  against every measured reconstruction error, swept across four depths on two designed-contrast
  point clouds.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/point-cloud-compression-for-fleet-uplink.sln`](build/point-cloud-compression-for-fleet-uplink.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/point-cloud-compression-for-fleet-uplink.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies: **Thrust** (header-only, part of the CUDA Toolkit — no separate install; used
for the radix sort over Morton codes and one large-array exclusive scan, both cited by name in
`src/kernels.cu`). No fallback path is needed since Thrust ships with the toolkit itself. No other
dependency beyond the CUDA runtime + C++17 standard library.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Two committed, synthetic clouds — a fair, same-point-count, same-scale contrast pair: a structured
warehouse-room map tile (`data/sample/structured_map.bin`, 200,000 points on 30 flat surfaces) and a
pathological cube of uniformly-scattered points (`data/sample/pathological_cube.bin`, 200,000
points, no surface structure at all). Both are generated by `scripts/make_synthetic.py` (xorshift32,
seed 42) — no public dataset is downloaded; `scripts/download_data.ps1`/`.sh` are honest no-ops (see
their headers for why: no public dataset ships a matched incompressible negative control). Full
binary format, field table, and SHA-256 checksums in [`data/README.md`](data/README.md).

## Expected output

The demo runs a **VERIFY stage** (seven GPU-vs-CPU bit-exact checks at the canonical depth D=10,
structured cloud — Morton codes, sort, per-level octree construction, histogram, canonical Huffman
table, encoded bitstream, and a full decode round trip) followed by a **sweep stage** (both clouds
× four depths) and **six gates** (`lossless_roundtrip`, `distortion_bound`, `rate_monotonic`,
`entropy_payoff`, `entropy_bound`, `timing`). `RESULT: PASS` requires the verify stage, every gate,
and every artifact write to succeed. The canonical stable lines live in
[`demo/expected_output.txt`](demo/expected_output.txt); every measured number (bits/point,
reconstruction error, compression ratio, timings) lives on `[info]`/`[time]` lines and in the CSV
artifacts under `demo/out/` — deliberately not diffed, since exact floats can differ in their last
bit across GPU architectures even when every gate still passes. See `demo/README.md` for the full
line-by-line contract and what each written artifact shows.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — start here: the two-stage pipeline explained in full, the
   shared data-layout formulas (Morton encoding, node/child-index arithmetic, the Huffman table
   format), and why the decoder has no GPU counterpart.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels themselves: Morton codes + Thrust sort
   (section B), the scan chapter (section C — the SAME primitive answering two different
   questions), per-level octree construction (section D — including the node-index off-by-one bug
   this project's own verify gate caught during development), histogram + canonical Huffman table +
   GPU encode (section E).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins AND the project's
   only decoder (a bit-trie Huffman walk + level-by-level octree expansion) — read this side by side
   with `kernels.cu` to see exactly what parallelization changed at each stage.
4. [`src/main.cu`](src/main.cu) — orchestration: load both clouds, run the VERIFY stage, sweep both
   clouds across all four depths, compute every gate, write every artifact.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `find_data_file`/`resolve_out_dir`, and
   why they are copied, not shared.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **PCL's `OctreePointCloudCompression`** — the closest direct relative: occupancy-octree geometry
  coding plus entropy coding, with optional temporal (frame-to-frame) delta encoding this project
  does not implement.
- **MPEG G-PCC (Geometry-based Point Cloud Compression)** — the modern standardized production
  answer; its octree geometry mode is structurally this project's Stage 1, but its entropy stage
  uses context-adaptive binary arithmetic coding (CABAC) instead of flat Huffman — study it for how
  conditioning on parent/neighbor occupancy beats a context-free code.
- **Google/Draco** — an open-source, widely-deployed point-cloud/mesh compressor; study its range-
  coding entropy stage as the natural "remove Huffman's +1-bit slack" next step (README "Exercises").
- **LASzip** — the LAS/LAZ airborne/terrestrial LiDAR compressor; a differently-tuned
  predictive+entropy pipeline for a differently-shaped point cloud (large surveys with per-point
  attributes), worth comparing against this project's map-tile-shaped problem.
- **Huffman (1952)**, "A Method for the Construction of Minimum-Redundancy Codes" — the original
  optimal-prefix-code result this project's entropy stage implements; DEFLATE (RFC 1951) is the
  most widely deployed canonical-Huffman implementation, and this project's canonical bit-assignment
  formula is the same one.
- **Karras (2012)** / this repo's own **02.05** (LBVH construction) — the closest sibling GPU
  pattern: Morton-sort-then-build, applied there to a bounding-volume hierarchy instead of an
  occupancy octree; read 02.05 first if the Morton/sort machinery here feels unfamiliar.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Plot the R-D curve.** Load `demo/out/rd_curve.csv` into a spreadsheet or `matplotlib` and plot
   `huffman_bits_per_point` vs. `max_error_m` for both cohorts — the classic rate-distortion tradeoff
   curve, made from this project's own measured numbers.
2. **Add run-length encoding (RLE) as a third alternative** and measure it against Huffman on the
   SAME occupancy stream — THEORY.md "Where this sits in the real world" predicts RLE will do
   poorly here (occupancy bytes rarely repeat identically many times in a row, even in structured
   data); verify that prediction with real code instead of taking the theory's word for it.
3. **Shared-memory-privatized histogram.** `compute_histogram_kernel` (kernels.cu) goes straight to
   global-memory atomics; implement the classic per-block shared-memory histogram + merge
   optimization THEORY.md "The GPU mapping" names, and measure the speedup at this project's own
   symbol counts (hint: it may be small at THIS scale — that is itself a useful, honest finding).
4. **A parallel bit-scatter for the byte-atomic case.** `compute_occupancy_kernel` uses a full
   `uint32_t` per node to sidestep CUDA's lack of native byte-granular `atomicOr`; implement the
   production "read the aligned word, shift the mask, atomicOr the word" trick for byte-granular
   writes instead, and measure the memory savings at the pathological cohort's largest depth
   (~966,000 nodes at D=11).
5. **[Ambitious] Block-wise parallel decode.** THEORY.md documents but does not implement the
   production answer to "decode resists parallelism": insert periodic byte-aligned restart points
   into the encoded stream so K independent decoders can each start from a known offset. Implement
   it, measure the rate cost (bits lost to restart-point padding) vs. the decode speedup.

## Limitations & honesty

- **Geometry only.** This codec compresses `(x, y, z)` — no intensity, color, semantic label, or
  timestamp. A real fleet-uplink payload usually carries more than geometry; PRACTICE.md §3 names
  this as a deliberate scoping decision, not an oversight.
- **No temporal (frame-to-frame) compression.** Every depth's octree is built completely from
  scratch; a production system would delta-encode against the previous uploaded version of the same
  map tile, a substantial additional saving this project does not attempt (THEORY.md "Where this
  sits in the real world").
- **Flat, context-free Huffman, not arithmetic/range coding.** As derived in THEORY.md "The math",
  Huffman pays a real `<1` bit/symbol penalty vs. the Shannon bound, and — a genuinely interesting,
  honestly-reported finding from this project's own development — the occupancy-byte histogram's
  entropy does NOT always favor the structured cloud over the pathological one in isolation (see
  THEORY.md "How we verify correctness" for the full, real story); the true compression advantage
  of structured data lives in Stage 1 (far fewer octree nodes needed), not Stage 2 alone.
- **Canonical Huffman codes are stored in a 32-bit field**, which would silently truncate a
  theoretical worst-case (Fibonacci-degenerate) 256-symbol code table exceeding 32 bits per code —
  never observed with real measured histograms at this project's scale, but named honestly
  (THEORY.md "Numerical considerations") rather than silently assumed away.
- **The pathological cohort is a designed negative control, not a claim about real LiDAR data.**
  No real sensor produces uniformly-scattered volumetric noise; it exists specifically to make the
  "surfaces are 2-D manifolds" argument falsifiable and measured, not merely asserted.
- **No motion of real hardware is possible from this project's output** — it produces compressed
  bytes and a reconstructed point cloud, nothing that commands an actuator. The safe-hardware-
  testing-ladder caveat (CLAUDE.md §1) is N/A here for exactly that reason (PRACTICE.md §3 states
  this explicitly).
- **Synthetic data throughout.** Both committed clouds are synthetic, generated by
  `scripts/make_synthetic.py` with a fixed seed; no public LiDAR/map dataset is used (see the "Data"
  section above and `data/README.md` for why).
