# 36.03 — Lattice-robot kinematics batches: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**The physical carrier this project abstracts away — and why that is an honest thing to do, up to a
point.** Every "module slides one cell" or "module pivots to a diagonal cell" this project checks the
legality of is, on real hardware, a small mechatronic miracle. A real lattice-robot module (the
lineage this project cites: M-TRAN, Roombots, and the broader crystalline/lattice-robot research
line) must contain, in a package the size of a fist to a large book:

- **Actuated connectors on every face** — latching mechanisms (hooks, permanent magnets,
  electropermanent magnets, or mechanical hermaphroditic connectors) that must engage and disengage
  ON COMMAND, under load, thousands of times, with millimeter-scale alignment tolerance.
- **Alignment funnels or compliant guides** — because two modules approaching each other under motor
  power will never arrive perfectly aligned; real designs chamfer the connector faces so a few
  millimeters of misalignment self-corrects during latching (see [`PRACTICE.md`](PRACTICE.md) §1–2 for
  the actual mechanical tolerances this demands).
- **Power and data pass-through on every face** — a module deep inside a lattice structure may have no
  direct path to a battery or a base-station radio; real lattice robots typically route power and
  communication cell-to-cell across the SAME faces the mechanical connectors use, so a "connector" is
  really a combined mechanical/electrical/data interface.
- **The actuator(s) that produce the motion itself** — a hinge motor (M-TRAN's rotational lattice
  motion), a linear or rotary drive at each connector (Roombots-style), or — for pure sliding-cube
  hardware concepts — a mechanism that has never been fully solved in general (see "Where this sits in
  the real world" below).

**This project's abstraction:** every one of those engineering problems is compressed into a single
boolean question — *"is this discrete move geometrically legal given who else is occupying the
lattice?"* — and the actual continuous motion, force, alignment, and timing are assumed away entirely.
This is the standard **sliding-cube model** used across the lattice self-reconfiguring robotics
literature specifically BECAUSE it lets researchers reason about reconfiguration algorithms (what
sequence of moves reaches a target shape, which shapes are even reachable) without first solving the
mechatronics — but it is important to say plainly that the abstraction is *generous* to the hardware:
a real move takes on the order of **seconds** (connector release → actuator sweep → connector
re-latch → verify), not the "instantaneous discrete step" this project checks the legality of. See
[`PRACTICE.md`](PRACTICE.md) §1 for what building the physical carrier actually involves.

**Engineering constraints a real robot imposes that this teaching core does NOT model:** connector
force budgets (how many modules can a single connector support before it must release?), backlash and
compliance in the hinge/connector mechanism, communication latency across a large lattice (a
100-module structure may need seconds to propagate a "module 57 has moved" update to every neighbor
that cares), and reliability at scale (a lattice robot with 1% per-move connector failure rate becomes
unreliable fast as module count and move count grow — the "self-repair" open problem named below).

## The math

**The lattice and its modules.** A configuration is a set of **kM = 24** distinct integer cells
`{(x_1,y_1,z_1), ..., (x_kM,y_kM,z_kM)} subset Z^3` — no physical units are assigned to a cell (a real
module's edge length is a hardware choice; see [`PRACTICE.md`](PRACTICE.md) §2), and no "up" direction
is privileged (see "Numerical considerations" for the isotropic-vs-gravity-biased scoping decision).

**Face adjacency (mechanical connection).** Modules at cells `a` and `b` are connected iff their
Manhattan distance is exactly 1:

```
adjacent(a, b)  <=>  |a.x-b.x| + |a.y-b.y| + |a.z-b.z| = 1
```

This is the ONLY distance that corresponds to two unit cubes sharing a face — distance 0 is total
overlap (illegal — Stage 1), distance 2 with two nonzero components is an **edge-diagonal** neighbor
(a legal MOVE target, but never a mechanical connection), and any larger distance is unrelated cells.

**A parity fact worth internalizing (it drove a real correction in this project's move rule — see
below): no cell is face-adjacent to two cells that are themselves face-adjacent to each other.** Proof
sketch: if `a` and `b = a+e` are adjacent (e a unit axis vector) and `c` is adjacent to both, then
`c = a+d` for some unit axis vector `d`, and also `c = b+d' = a+e+d'` for some unit axis vector `d'` —
forcing `d = e+d'`. Checking every case (`d'=e` gives `d=2e`, not unit; `d'=-e` gives `d=0`, not unit;
`d'` perpendicular to `e` gives `d` with two nonzero components, not unit) shows no valid `d` exists.
The grid graph is **bipartite** (colour cells by `(x+y+z) mod 2`; every edge connects opposite
colours), and bipartite graphs have no common neighbour between two adjacent vertices — the general
fact behind this specific one.

**The sliding-cube move model.** A module at cell `A` may move to an EMPTY cell `B` in exactly two
ways (this project's two "move families", `dir` 0–5 and 6–17 in [`kernels.cuh`](src/kernels.cuh)):

### SLIDE (linear move) — `B = A + e`, `e` one of the 6 face directions

```
   Before (side view, slicing through the f axis):        After:

   f=+1  [ W1 ][ W2 ]    <- the WALL, two modules          f=+1  [ W1 ][ W2 ]
   f= 0  [ A  ][    ]       spanning BOTH A's and           f= 0  [    ][ B  ]
              e-->             B's f-offset                            e-->

   W1 = A + f   (occupied)          B = A + e   (must be EMPTY before the move)
   W2 = B + f   (occupied)          module ends at B, still face-adjacent to W2
```

**Precondition:** `B` is empty, and there exists a perpendicular direction `f` (on a DIFFERENT axis
than `e`) such that **both** `A+f` and `B+f` are occupied.

**Why a TWO-module wall, not one (a correction made while building this project — CLAUDE.md's
"no black boxes, verify everything" in action):** an earlier draft of this precondition required only
`A+f` to be occupied, reasoning that "the wall stays at a fixed perpendicular offset throughout the
slide." That reasoning is WRONG: the fixed-offset observation is true only along the `f` axis, but the
module's position along the `e` axis also changes, from aligned with `A+f` to a distance of 1 away —
so a single wall cube loses face contact with the module by the time it reaches `B` (the parity fact
above proves this exactly: `A+f` and `B` are never face-adjacent). A single wall cube is only in
contact with the module's START, not its END. Requiring occupancy at BOTH `A+f` and `B+f` models a
genuine two-cell-long supporting surface, which — reasoning about the CONTINUOUS motion this discrete
step abstracts — keeps the module in contact with SOME point of the wall for the entire 1-cell slide,
start to finish. This is the physically defensible version, and it is what [`kernels.cu`](src/kernels.cu)
and [`reference_cpu.cpp`](src/reference_cpu.cpp) implement.

### CORNER (convex/pivot move) — `B = A + e + f`, `e` and `f` on different axes (12 such directions:
### 3 axis-pairs x 4 sign combinations — the cube's 12 edge-diagonals)

```
   Top-down view of the 2x2 cell block {A, A+e, A+f, B}:

        f
        ^
        |   +----+----+
        |   | Q  | B  |     Q = A + f   (must be EMPTY — the module's corner sweeps near it)
        |   +----+----+     P = A + e   (must be OCCUPIED — the pivot the module rotates around)
        |   | A  | P  |     B = A + e + f   (must be EMPTY — the destination)
        |   +----+----+
        +----------------> e

   The module at A rotates 90 degrees around the shared edge of pivot P,
   swinging from resting beside P (in the e direction) to resting on top
   of P (in the f direction) -- landing at B, which is face-adjacent to
   P via f (B = P + f), exactly the pivot's "other" face.
```

**Precondition:** `B` is empty, and **exactly one** of `{A+e, A+f}` is occupied (the occupied one is
the pivot `P`; the empty one, `Q`, is the corner the module's rotation sweeps past — if `Q` is ALSO
occupied, the rotation is physically blocked; if NEITHER is occupied, there is nothing to pivot
around). This is a single, symmetric integer test: `occupied(A+e) != occupied(A+f)`.

**Why a single pivot module suffices (unlike the slide):** rotation is a fundamentally different
motion than translation — the module sweeps around a FIXED point (the shared edge of `P`), so a single
neighbor is the correct minimal support, matching how the literature typically describes "pivot" or
"convex transition" moves for lattice-style modules.

**Articulation points (cut vertices).** Module `m` is an articulation point of the face-adjacency
graph iff removing it disconnects the remaining kM−1 modules into two or more pieces. An articulation
module can never legally move — its own departure would fracture the robot before it even lands
anywhere — so [Stage 4](src/kernels.cu) forces `legal_move = 0` for every direction of every
articulation module, independent of the SLIDE/CORNER geometry above.

## The algorithm

**Stage 1 — validity, O(kM log kM) [insertion sort O(kM^2) at this size]:** pack `(x,y,z)` into one
sortable 64-bit key (`kBias`-shifted so negative coordinates pack correctly — see
[`kernels.cu`](src/kernels.cu)), insertion-sort the kM=24 keys, scan for adjacent equal keys.

**Stage 2 — connectivity, O(kM^2):** textbook array-based BFS from module 0 over the face-adjacency
graph (adjacency tested on the fly, O(kM) per neighbour query — no adjacency list is built, since
building one costs the same O(kM^2) the direct approach already pays).

**Stage 3 — articulation points, O(kM^2), Tarjan's DFS low-link algorithm.** This is a classic
algorithm worth teaching properly, not just citing:

- `disc[u]` — the DFS **discovery time** of node `u` (the order the search first visits it: 0, 1, 2,
  ...).
- `low[u]` — the smallest discovery time reachable from `u`'s entire DFS subtree using **at most one
  non-tree ("back") edge** — informally, "how far up the tree can `u`'s subtree reach without going
  back through `u`'s own parent edge."
- **Tree edges** (`u -> v` where `v` was previously unvisited) descend the DFS; **back edges**
  (`u -> v` where `v` is an already-visited ANCESTOR, never the immediate parent) are the only other
  kind of edge a graph traversal from a tree can discover (there are no "cross edges" in an
  UNDIRECTED graph DFS — every non-tree edge connects a node to one of its own ancestors).
- **Non-root cut-vertex test:** node `u` (not the DFS root) is an articulation point iff it has a
  DFS-tree child `c` with `low[c] >= disc[u]` — meaning `c`'s entire subtree has NO back edge that
  escapes above `u`, so deleting `u` strands that subtree from the rest of the graph.
- **Root cut-vertex test:** the DFS root is an articulation point iff it has **2 or more** DFS-tree
  children — they can only reach each other by passing back through the root.

[`kernels.cu`](src/kernels.cu)'s `articulation_kernel` implements this **iteratively** (an explicit
stack plus a `next_child[]` resume index per node) rather than recursively — the standard
"recursion-to-state-machine" conversion, chosen so the per-thread stack depth and memory footprint are
fully visible in the code rather than hidden in the call stack. Complexity: each node's neighbour list
is scanned exactly once across the whole algorithm (the `next_child[]` counter only increases), giving
O(kM) amortized scan-steps per node and O(kM^2) = 576 total operations for kM=24 — the same order as
Stage 2's BFS.

**Stage 4 — move enumeration, O(kM · 18 · kM) = O(kM^2 · 18):** for every module (skipped entirely, in
O(1), if it is an articulation point) and every one of the 18 move directions, test the SLIDE or
CORNER precondition above — each test is O(kM) (a handful of `occupied()` point queries, each an
O(kM) linear scan). 24 modules x 18 directions x a few O(24) queries ≈ 10,000 operations per
configuration — trivial at this scale, discussed honestly in "The GPU mapping" below.

**The batch generator (seeded accretion), O(kM) expected, O(kM^2) worst case per configuration:**
start module 0 at the origin; for each subsequent module, pick a uniformly random already-placed
module and a uniformly random face direction, and place the new module there if the target cell is
free (bounded random retries, then a deterministic fallback scan that is GUARANTEED to find a free
cell — a connected set of fewer than kM cells on an infinite lattice always has at least one free
neighbour). Connectivity holds by construction: every module attaches to something already in the
graph.

## The GPU mapping

```
one thread = one CONFIGURATION k (not one module!)
grid = ceil(K/256) x 256           (repo default; ragged tail guarded)

per thread, entirely in LOCAL memory (no shared memory, no atomics,
no cross-thread communication of any kind — by construction, every
configuration is independent):

  Stage 1 (validity):      int64_t keys[24]                  (192 B)
  Stage 2 (connectivity):  bool visited[24], int queue[24]    (~120 B)
  Stage 3 (articulation):  int disc/low/parent/next_child[24] (384 B)
                           uint8_t artic[24], int stack[24]   (~120 B)
  Stage 4 (move enum):     no per-thread arrays beyond the loop
                           variables — output written directly
```

**Why "one thread per configuration" and not "one thread per module" or a parallel BFS/DFS across
threads:** the graphs here have kM=24 nodes. A parallel frontier-based BFS (the kind of algorithm
cuGraph or a GPU graph-analytics library would use on a graph with millions of nodes) needs
cross-thread/cross-block synchronization to build and advance a shared frontier — atomics or
cooperative-group barriers, memory traffic to a shared visited array, and load-balancing logic. All of
that machinery costs FAR more than the ~500 sequential integer comparisons a single thread needs to
BFS a 24-node graph start to finish. This is the **small-graph-per-thread regime**: exactly the same
scale argument 25.01 makes for keeping its own per-cell work serial rather than parallelizing an
already-tiny unit of work, and the same one 33.01 makes for small per-thread matrices — the honest
rule is "parallelize the largest independent unit that is still worth NOT parallelizing internally,"
and here that unit is "one whole configuration's worth of graph algorithms," not "one BFS step."

**Memory footprint honesty:** unlike 08.01's 4-float cart-pole state (fits trivially in registers),
this project's per-thread working set (roughly 800 bytes across all four kernels, though each kernel
only needs its own subset at a time) is large enough that the compiler will spill some of it to LOCAL
memory — physically the same DRAM as global memory, but L1/L2-cached and accessed per-thread with
addressing the compiler can often turn into efficient patterns. At kM=24 this spill is a non-issue
(the working set is a few hundred bytes, well within L1 cache reach for the whole warp's worth of
threads); it would become a real design question at kM in the hundreds — the point at which THEORY
recommends revisiting the one-thread-per-configuration mapping entirely (see "Where this sits in the
real world").

**Why no `__constant__` memory, unlike 09.01's robot model:** 09.01's joint-chain model is DATA loaded
once at runtime from a file — the textbook `__constant__` use case (broadcast a uniform read to every
thread). This project's lattice geometry (the 6 slide directions, the 12 corner directions) is not
data at all — it is a small, FIXED formula (`slide_delta()`, `corner_axes()` in
[`kernels.cu`](src/kernels.cu)), computed in a handful of integer instructions per direction. A
formula that cheap has no upload cost to amortize and no memory-hierarchy decision to make; using
`__constant__` memory here would add machinery to save work that costs less than the machinery itself.

**Why no CUDA library call:** every stage here is small-scale integer graph logic — there is no
BLAS/FFT/sort-network primitive that maps onto "run Tarjan's algorithm on a 24-node graph" the way
cuBLAS maps onto a batched matrix multiply. This project's four kernels are, deliberately, the
hand-written thing a library would not help with (CLAUDE.md §1's "no black boxes" cuts both ways: some
computations do not have a library shortcut, and pretending otherwise would be its own kind of
dishonesty).

## Numerical considerations

**All-integer — the shortest "numerical considerations" section in this repo, and that is the
point.** Every value in this pipeline — lattice coordinates, adjacency tests, validity, connectivity,
discovery times, low-links, move legality — is a 32-bit or 64-bit integer computed by addition,
subtraction, comparison, and `abs()`. There is no floating point ANYWHERE in the checked pipeline
(the vignette's compactness potential `Phi` is likewise `int64_t`, deliberately, to keep this
project's identity all-integer end to end — see [`main.cu`](src/main.cu) §8). Consequences:

- **No rounding, no ULP drift, no rel-tolerance gate.** The GPU-vs-CPU verify gate ([`main.cu`](src/main.cu))
  demands **bit-exact equality** on every one of the six output arrays, across all K=4096
  configurations — a strictly stronger and simpler guarantee than 08.01/09.01's `rel_tol = 1e-3`/`1e-6`
  gates, made possible only because the domain itself is discrete.
- **No determinism concerns from FMA fusion, intrinsic-vs-precise trig, or platform `libm`
  differences** (the concerns that shape 08.01/09.01's numerics sections) — integer arithmetic is
  bit-identical across any IEEE-conforming host and device compiler.
- **Coordinate range headroom, not precision, is the only "numerical" design decision here:**
  `pack_key()`'s bias (`1<<20`) and bit-width (21 bits/axis, ±2^20) are sized to comfortably contain
  the disconnect corruption's 100,000-cell translation (see [`main.cu`](src/main.cu) §"corrupt_disconnect")
  with wide margin, not to satisfy any accuracy requirement.
- **Isotropic, not gravity-biased (a scoping choice, stated once, cross-referenced everywhere it
  matters):** the move preconditions above assume no privileged "down" axis. Real hardware operates
  under gravity, and many published sliding-cube formalizations bake in a "floor" requirement (support
  must be BELOW the source or destination, specifically). This project's isotropic version is simpler
  to reason about, matches the theoretical reconfiguration-complexity literature's usual framing, and
  is explicitly named as a simplification relative to gravity-affected hardware (README "Limitations").

## How we verify correctness

Three independent layers, deliberately more than most flagships in this repo need, because this
project's all-integer domain makes strong cross-checks CHEAP to run for real (not just aspirational):

1. **GPU-vs-CPU, bit-exact, full batch (the repo's standard §5 gate, at its strongest form):** all
   four stages, all K=4096 configurations, zero tolerance. [`reference_cpu.cpp`](src/reference_cpu.cpp)'s
   four `*_cpu` functions are line-by-line sequential twins of the four kernels — a bug in indexing,
   thread mapping, or the ragged-tail guard shows up as ANY nonzero mismatch count, instantly.
2. **Injected-corruption detection (a designed negative control, not incidental):** the batch
   generator PROVES its clean configurations are valid+connected by construction (seeded accretion —
   see "The algorithm"), then deliberately corrupts a documented 10% with a duplicate-position defect
   (`corrupt_duplicate`) or a severed-connectivity defect (`corrupt_disconnect`), each engineered to be
   CLEAN in isolation (a duplicate corruption never disconnects anything real, because it always
   targets the most-recently-accreted, structurally-safe-to-remove module; a disconnect corruption
   never creates a duplicate, because it rigidly translates a whole subgroup by an offset far outside
   any accretion cluster's extent). The gate requires the pipeline to catch every single corrupted
   configuration and raise zero false alarms on the 3686 clean ones — a much stronger claim than
   "the algorithm looks right," because it exercises the FAILURE paths, not just the happy path.
3. **Brute-force cross-checks, independently coded, on a 128-configuration subset:**
   `articulation_bruteforce_cpu` re-derives cut vertices via the textbook "remove each module, re-BFS
   the rest" oracle — an algorithm with NOTHING in common with Tarjan low-link beyond "both correctly
   answer the same question." `move_precondition_bruteforce_cpu` re-derives move legality via an
   explicit occupied-cell array and a literal 12-row corner-direction table, instead of the fast
   path's on-the-fly scans and `corner_axes()` formula. A subset (not the full K=4096) is enough
   here — see "The algorithm" for these oracles' complexity: at kM=24 they cost nothing, so 128
   configurations (3072 articulation checks, 55,296 move-precondition checks) is chosen simply as
   "enough to be statistically convincing without slowing the demo down," not as a scaling necessity.

   This layer exists because layer 1 alone cannot catch a bug in the SHARED UNDERSTANDING of the
   rules — if `kernels.cu` and its `reference_cpu.cpp` twin both misimplement the same precondition
   (a translation slip is one risk; a conceptual misunderstanding of the rule is a different, sneakier
   one), they would agree with each other and still be wrong. An independently-shaped implementation
   is unlikely to reproduce the same conceptual mistake.

4. **The reconfiguration vignette's per-step re-verification (an end-to-end behavioural check):** every
   one of the 127 accepted moves in the demo run is followed by an independent
   `validity_cpu`+`connectivity_cpu` call on the WHOLE post-move configuration — not just a check that
   the moved module's own precondition held, but that the entire kM-module robot is still one
   physically connected piece. [`main.cu`](src/main.cu) hard-aborts (`std::exit`) if this
   invariant is ever violated, because that would indicate a logic bug in the pipeline the previous
   three layers did not have the opportunity to exercise (a legality check passing does not, by
   itself, guarantee the resulting WHOLE robot stays connected — see THEORY "The math" SLIDE
   discussion and README "Limitations").

## Where this sits in the real world

- **The open problems this teaching core deliberately stops short of:** REAL reconfiguration planning
  (project **36.01**, "Reconfiguration planning over enormous state spaces (GPU search)") must search
  a state space of size roughly `(number of reachable lattice shapes of kM cells)`, which grows
  super-exponentially in kM — general lattice-robot reconfiguration between two arbitrary shapes is
  known to be **PSPACE-hard** in several published formalizations of the sliding-cube-style models
  (the same complexity class as generalized Rush Hour and Sokoban — "there IS a solution, but finding
  it may require exploring an exponential fraction of the state space in the worst case"). Known
  POLYNOMIAL special cases exist in the literature (e.g., reconfiguring between two shapes that are
  both simply-connected/"tree-like" in specific senses, or under relaxed move sets) — this is an
  active research area, not a solved one, and this project does not attempt either the hardness proof
  or the polynomial special-case algorithms; it only builds the legality-checking substrate those
  algorithms would need as their per-state expansion oracle.
- **Distributed/decentralized control** (project **36.05**, "Emergent distributed control experiments
  at scale") is a second, orthogonal open problem this project does not touch: every module here has
  GLOBAL knowledge of the whole configuration (the batch kernels see all 24 module positions at once);
  real large lattice robots increasingly aim for LOCAL rules (each module only knows its immediate
  neighbours) that still produce coherent global reconfiguration — a much harder and less-solved
  design problem than the centralized planning this project's kinematics feed into.
- **Hardware reliability at scale** is the field's other honest open problem: demonstrated lattice/
  chain hybrid robots (M-TRAN, and successors in the same research lineage) have shown tens of modules
  reconfiguring; scaling connector reliability, power distribution, and communication latency to
  hundreds or thousands of modules — the module counts that would make lattice robots practically
  interesting (self-assembling structures, reconfigurable spacecraft, disaster-response rubble
  robots) — remains unsolved. **Self-repair** (detecting and routing around a failed module without
  human intervention) compounds the difficulty further and is very much a research-stage topic, not
  demonstrated at any meaningful scale.
- **No production lattice-robot system exists today** — see [`PRACTICE.md`](PRACTICE.md) §4 for the
  business/regulatory honesty this implies: this entire catalog domain (36. Modular &
  Self-Reconfigurable Robots) is, as of this writing, a research field with no commercial deployment,
  a genuinely different situation from most of this repository's other domains.
- **At larger kM, the GPU mapping itself would need revisiting:** this project's "one thread, one whole
  graph algorithm" design is honest about being tuned for kM in the tens (see "The GPU mapping"). A
  lattice robot with hundreds of modules would push per-thread local-memory pressure and per-thread
  work far enough that a hybrid mapping (e.g., one THREAD BLOCK per configuration, with the kM modules
  spread across the block's threads for the O(kM) inner loops) would likely win — a natural, documented
  extension this project does not implement.
