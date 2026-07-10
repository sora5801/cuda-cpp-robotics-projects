# 01.02 — Stereo depth: block matching, then Semi-Global Matching (SGM) kernels: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Two eyes, one triangle.** Hold a finger up and close one eye, then the other — your finger appears
to jump sideways against the background. That jump (the *disparity*) is larger the closer your finger
is. Stereo depth estimation is exactly that trick, made precise and automatic: two cameras, a known
baseline apart, each seeing the same 3-D point at a different image column; from the difference in
columns and the camera geometry, depth falls straight out of similar triangles.

**Epipolar geometry, and why rectification is the whole trick.** In general, if you know a point in the
left image, the matching point in the right image could be ANYWHERE along a line — the *epipolar
line*, the projection of the left camera's viewing ray into the right image. Searching a whole 2-D
image for a match would be needlessly expensive and needlessly ambiguous. **Rectification** — a pure
geometric warp of both images computed once from calibration (the focal lengths, the baseline, and the
relative pose between the two cameras) — re-projects both images onto a common image plane such that
every epipolar line becomes a horizontal image ROW. After rectification, a point at row `y`, column
`xL` in the left image can only match a point at the SAME row `y` in the right image, at some column
`xR <= xL`. The 2-D correspondence search collapses to a 1-D search along one row. This project starts
from an ALREADY-RECTIFIED pair (project 01.01/01.07's job in a real pipeline — the self-containment
rule, README "Limitations") — but every kernel here exploits the 1-D consequence directly: the cost
volume only ever looks along a row, never off it.

**Disparity and depth — deriving `Z = f·B/d`.** Let the two cameras share the same focal length `f`
(pixels) and be separated by a horizontal *baseline* `B` (meters), both looking in the same direction
(the standard rectified configuration). A 3-D point at depth `Z` (meters, along the optical axis)
projects to image column `xL = f·X/Z + cx` in the left camera and `xR = f·(X-B)/Z + cx` in the right
camera (pinhole projection; `cx` is the principal point, which cancels below). The **disparity**

```
d = xL - xR = f·B/Z            (pixels; d >= 0 for points in front of both cameras)
```

is *inversely* proportional to depth. Inverting: **`Z = f·B/d`** — the equation every stereo system
ultimately computes. Two engineering consequences fall directly out of this one line:

- **Depth resolution degrades as `Z²`, not linearly.** Differentiating, `dZ/dd = -f·B/d² = -Z²/(f·B)`.
  A ONE-PIXEL disparity error costs `Z²/(f·B)` meters of depth error — quadratically worse the farther
  away the point is. At `f=700 px`, `B=0.12 m` (a typical compact stereo rig, PRACTICE.md §2), a
  1-pixel error costs ~1.2 cm at `Z=1 m` but ~1.2 m at `Z=10 m`. This is WHY stereo depth is a
  near-field sensor by construction, not a limitation of any particular algorithm: the same disparity
  quantization (integer pixels, or sub-pixel-interpolated fractions of one) simply carries less and
  less depth information the farther the point is.
- **`d = 0` means `Z = infinity`.** The disparity range this project searches, `D = 64` levels
  (`kMaxDisp` in `kernels.cuh`), corresponds — at the illustrative `f, B` above — to depths from
  infinity (`d=0`) down to `700*0.12/63 ≈ 1.33 m` (the nearest resolvable point at `d=63`). Widening
  `D` extends the near range at the cost of `D` more Hamming distances per pixel in the cost volume
  (THEORY "The GPU mapping" — the cost is exactly linear in `D`).

This project's committed scene (`data/README.md`) is authored directly in disparity space rather than
from a metric 3-D scene, but the numbers still have a physical reading: at the illustrative `f=700 px,
B=0.12 m` above, the background's 4–18 px disparity range corresponds to depths of roughly **21.0 m
(far) down to 4.7 m (near)** — a receding ground plane, exactly as intended — and the three
foreground rectangles (26/40/52 px) correspond to roughly **3.2 m, 2.1 m, and 1.6 m**, progressively
nearer objects. (These are illustrative round-trip numbers from an assumed `f, B` — see PRACTICE.md §2
for real camera parameter ranges; the generator itself never needs them, because it authors disparity
directly — see "Numerical considerations" below for why that is a legitimate simplification.)

**The engineering constraints a real stereo rig imposes** (PRACTICE.md goes deeper on the hardware):
calibration accuracy (a fraction-of-a-pixel rectification error shows up as a systematic depth bias,
worse at range per the `Z²` law above), synchronization (both cameras must expose the SAME instant, or
a moving scene mismatches), baseline/field-of-view trade-offs (PRACTICE.md §2), and — the reason this
project exists at all — **radiometric mismatch**: no two real camera sensors, even from the same
production batch, report IDENTICAL brightness for the same physical point. Auto-exposure converges
slightly differently per camera, vignetting differs, gain and black-level calibration drift. Any
matching cost built from raw pixel differences (sum of absolute/squared differences, SAD/SSD) bakes
that systematic offset straight into every single comparison. The **census transform**, this project's
first kernel, exists specifically to make matching indifferent to exactly this kind of real-world
brightness mismatch — the next section makes that precise.

## The math

**Problem statement.** Given rectified images `I_L, I_R: [0,W) x [0,H) -> [0,255]`, find, for every
LEFT pixel `(x, y)`, the disparity `d(x,y) in {0, ..., D-1}` (or "no answer") that best explains
`I_L(x,y)` and `I_R(x-d, y)` as the same scene point, where "best" is defined by a cost function
`C(x, y, d)` this project builds in two stages.

**Stage 1 — the census signature.** For a `(2h+1) x (2h+1)` window (`h = kCensusHalf = 3`, a 7x7
window, `kCensusBits = 48` = `7*7 - 1` comparisons), define the census signature of pixel `(x,y)` as
the bit-string

```
census(x,y)_k = [ I(x + wx_k, y + wy_k) < I(x,y) ]     for k = 0..47, (wx_k, wy_k) enumerating the
                                                        7x7 window minus the center
```

— one bit per neighbor, 1 if that neighbor is DARKER than the center, 0 otherwise (a strict `<`; ties
clear the bit — an arbitrary but FIXED convention, shared exactly by the GPU kernel and the CPU oracle,
so it never causes a disagreement between them). This is a RELATIVE, ORDINAL encoding: it only ever
asks "which of these two pixels is brighter", never "what is the absolute brightness". A uniform
additive or multiplicative brightness shift between the left and right cameras — exactly the
radiometric mismatch named above — changes every pixel's intensity but preserves EVERY ordering
relationship between a pixel and its neighbors, so it changes NOTHING in the census signature.
(SAD/SSD have no such immunity: adding a constant `k` to every right-image pixel shifts every SAD term
by `|k|`, uniformly corrupting the cost volume.)

**Stage 2 — the Hamming-distance cost.** The matching cost for candidate disparity `d` at pixel
`(x,y)` is the Hamming distance between the two signatures:

```
C(x, y, d) = popcount( census_L(x,y) XOR census_R(x-d, y) )        in [0, 48]
```

— the number of window positions where the left and right neighborhoods DISAGREE about brightness
ordering. Two identical local neighborhoods (a correct match) score `C = 0`; two unrelated
neighborhoods score around `C ~ 24` (half the bits differ, by symmetry of a fair coin flip); `C = 48`
is maximal disagreement. `kCostInvalid = 255` is a SENTINEL (never a real Hamming distance, which
cannot exceed 48) marking "this candidate could not be evaluated" (a census-border pixel, or
`x - d < 0`) — kernels.cuh names every sentinel once, and every kernel checks it explicitly rather
than letting a fake "cost" contaminate a real comparison.

**Stage 3a — block matching (winner-take-all).** The simplest possible decision rule:

```
d_BM(x,y) = argmin_d C(x, y, d)
```

— no context, no neighbors, just "which disparity had the lowest cost, here, alone". Fast (an O(D)
scan per pixel, embarrassingly parallel) and exactly as blind as that description suggests: a locally
AMBIGUOUS neighborhood (several disparities with similarly low cost — the classic failure of
repetitive or low-detail texture) has no mechanism to borrow confidence from a well-matched neighbor.

**Stage 3b — Semi-Global Matching, the energy model.** SGM starts from a better-posed problem: instead
of minimizing cost independently per pixel, minimize a GLOBAL energy over the WHOLE disparity image
`D(.)` that also penalizes disagreement between spatial neighbors:

```
E(D) = Sum_p C(p, D(p))                                              (data term: same as before)
     + Sum_{q in N(p)} P1 * [ |D(p)-D(q)| == 1 ]                     (small jump: mild penalty)
     + Sum_{q in N(p)} P2 * [ |D(p)-D(q)|  > 1 ]     (P2 > P1)        (big jump: larger penalty)
```

`P1 < P2` is not a detail, it is the "Semi-" in Semi-Global Matching: a SMALL disparity change between
neighbors is normal (real surfaces slant and curve continuously — think of the receding ground plane
in this project's scene) and gets a light penalty, while a LARGE jump usually means a real depth
discontinuity (an object edge) and is allowed more freely, but only where the DATA actually supports
it (a small `C` at the jumped-to disparity) rather than as a free byproduct of an over-eager
smoothness prior. Minimizing `E(D)` EXACTLY over a full 2-D neighborhood graph is NP-hard (it is a
2-D Markov Random Field with a non-submodular-in-general prior) — SGM's entire contribution
(Hirschmüller 2008) is an efficient APPROXIMATION: instead of one global 2-D optimization, solve `R`
independent 1-D optimizations (one per PATH direction) exactly via dynamic programming, then sum the
results. Each 1-D path is solved by the recurrence

```
L_r(p,d) = C(p,d) + min( L_r(p-r,d),                         no jump
                          L_r(p-r,d-1) + P1,                  small jump, one way
                          L_r(p-r,d+1) + P1,                  small jump, other way
                          min_k L_r(p-r,k) + P2 )             any bigger jump, flat P2
                  - min_k L_r(p-r,k)                          running-min subtraction (bounds L_r —
                                                               see "Numerical considerations")
```

where `r` is the path's step direction (e.g. `r = (+1, 0)` for left-to-right) and `p-r` is `p`'s
PREDECESSOR along that path — this is genuinely a 1-D DYNAMIC PROGRAMMING recurrence, the same family
as edit distance or the knapsack problem, just over the "state" `d in {0..D-1}` at each step. Summing
`R` independent path directions,

```
L_sum(p, d) = Sum_{r in paths} L_r(p, d),          d_SGM(p) = argmin_d L_sum(p, d)
```

approximates the full 2-D smoothness prior by "smoothness along several 1-D crossings of each pixel" —
cheaper than the true 2-D problem by a huge margin, and (Hirschmüller's central empirical result, which
this project's own measured 63.35% -> 97.52% gap reproduces at small scale) good enough in practice to
close most of the gap to a real 2-D solution. This project uses `R = 4` paths (L→R, R→L, T→B, B→T);
production SGM typically uses `R = 8` (adding the 4 diagonals) — see "The GPU mapping" for exactly what
that costs and buys.

## The algorithm

Per demo run (`main.cu`'s stages, matched to the code):

1. **Census** — O(W·H·48) work (a 7×7 stencil per pixel), fully data-parallel across pixels.
2. **Cost volume** — O(W·H·D) Hamming distances (a popcount per (pixel, disparity) pair), fully
   data-parallel across (pixel, disparity) pairs.
3. **Block matching** — O(W·H·D) argmin work (data-parallel across pixels; each pixel's D-length scan
   is sequential but pixels are independent), plus a left-right check (O(W·H), see below).
4. **SGM aggregation** — O(R·W·H·D) work: `R=4` full sweeps of the cost volume, each sweep being
   O(scanline_length · D) SEQUENTIAL work per scanline (the recurrence above needs the previous step's
   full D-length result) but the ~300–400 scanlines within one sweep are independent of each other.
5. **SGM winner-take-all + left-right check + 3×3 median** — the same O(W·H·D) argmin as step 3, plus
   O(W·H) for the consistency check and the median filter (each pixel touches at most 9 neighbors).

**Serial cost** (a single CPU thread, exactly what `reference_cpu.cpp` measures): dominated by the cost
volume and the SGM aggregation, both O(W·H·D); on the reference machine, the full CPU pipeline
(census + cost volume + BOTH final disparity maps, i.e. steps 1–5 run TWICE, once for the isolated
VERIFY checkpoints and once inside the full pipeline oracle) takes ~560 ms.

**Parallel cost, and the one place it is NOT trivially "one thread per unit of work":** steps 1–3 and
5 map cleanly to one GPU thread per pixel (or per (pixel, disparity) pair, looped). Step 4 (SGM
aggregation) is the exception, and understanding exactly why is this project's second headline lesson
(after the BM-vs-SGM comparison itself) — see "The GPU mapping" immediately below.

**The left-right consistency check**, used after BOTH winner-take-all stages: having found
`d_L(x,y) = argmin_d cost_or_lsum(x, y, d)` (the LEFT-referenced disparity), independently compute
`d_R(xR,y)` (the RIGHT-referenced disparity — "if the right camera asked the same question, what would
IT answer, treating `xR` as the reference column?"). A left pixel's answer is kept only if
`|d_L(x,y) - d_R(x - d_L(x,y), y)| <= 1` — the two independent questions must agree about the SAME 3-D
point. This catches errors winner-take-all cannot see by construction (a wrong match can still have the
lowest cost among wrong candidates, especially in repetitive texture or occluded regions) — it needs
no ground truth, which is exactly why it is a real, run-time-usable check and not just a demo
convenience (a real robot's stereo node has no ground truth at run time either). `kernels.cuh`/
`kernels.cu` explain the SYMMETRIC-REUSE trick this project uses to get `d_R` for free from the SAME
cost volume, without building a second one.

## The GPU mapping

```
CENSUS            : one thread per PIXEL          — STENCIL  (reads a 7x7 neighborhood)
COST VOLUME       : one thread per PIXEL, loops D — MAP      (D independent popcounts)
SGM AGGREGATION   : one thread per SCANLINE       — SCAN     (sequential along the path; the
                                                               ~300-400 scanlines are the only
                                                               parallelism this stage has)
WTA / LR / MEDIAN : one thread per PIXEL          — MAP      (one independent decision each)
```

**Why the cost volume is stored D-MAJOR** (`cost[d*H*W + y*W + x]`, the full argument lives in
`kernels.cuh` — summarized here): the cost-VOLUME-CONSTRUCTION kernel is this project's single
largest MAP kernel (W·H·D work), and under D-major, its warp-wide write pattern (fixed `d`, 32
consecutive `x` values) is a single contiguous 32-byte span — perfectly coalesced. The alternative,
pixel-major (`cost[(y*W+x)*D + d]`), would make that SAME warp write to 32 addresses each `D=64` bytes
apart — a 64x-worse scatter for the repo's hottest kernel in this project. A genuinely free side
effect of that choice: the SGM VERTICAL paths (one thread per COLUMN) then ALSO read/write
`cost[d*H*W + y*W + x]` with consecutive threads = consecutive `x` = consecutive addresses at every
fixed `(d, y)` step — perfectly coalesced, for free. The HORIZONTAL paths (one thread per ROW) do
NOT get this for free: consecutive threads (rows `y, y+1`) are `W` elements apart under either layout,
so D-major's stride-`W` pattern is merely far cheaper than pixel-major's stride-`(W*D)` would have
been, not free. This asymmetry is stated, not hidden — a production implementation typically
transposes the buffer (or keeps a second, transposed copy) between horizontal and vertical passes to
get full coalescing on BOTH; this project keeps ONE buffer for a reader-verifiable kernel body and
names the cost it pays.

**"The parallelism tension SGM is famous for."** Every OTHER kernel in this pipeline parallelizes over
~110,592 independent pixels — the GPU's natural unit of work, saturating an RTX 2080 SUPER's 46 SMs
many times over. SGM's aggregation is fundamentally different: within ONE path direction, a pixel's
`L_r` value depends on its PREDECESSOR's full D-length result along that path — a genuine sequential
dependency chain, not a scheduling inconvenience. The only parallelism available is ACROSS scanlines
(~288 rows for a horizontal pass, ~384 columns for a vertical pass) — one to two orders of magnitude
less than "one thread per pixel" offers everywhere else. That mismatch is exactly why this project's
SGM aggregation measures ~60–80 ms against census+cost-volume's ~0.6–1.5 ms on the SAME image: not
because SGM does more total arithmetic (`R·W·H·D` vs. `W·H·D` — a small constant factor of 4), but
because it can only put a fraction of the GPU's SMs to work at once. Production SGM libraries (libSGM,
OpenCV CUDA StereoSGM) close most of this gap with techniques this project deliberately leaves as
exercises rather than folding into the teaching core: TILING each scanline across MULTIPLE threads
with an intra-block scan/reduction for the running minimum (turning "one thread marches 384 steps" into
"32 threads cooperate on 384 steps"), and processing several scanlines' worth of independent path
segments concurrently on different streaming multiprocessors via careful launch geometry. README
Exercise 3 (adding the 4 diagonal paths) and THEORY's own honesty about this gap are the same lesson
from two angles: SGM's quality comes with a real, structural GPU-mapping cost, and knowing exactly
where that cost lives is the point, not an embarrassment to paper over.

**Memory hierarchy used, and why:** GLOBAL memory only — no shared-memory tiling anywhere in this
project (a deliberate simplification named in `kernels.cu`'s comments: it keeps every kernel body a
direct, verifiable transcription of `reference_cpu.cpp`'s loops; README Exercise 4 is the tiled
version). REGISTERS/local memory hold each thread's working state: the census kernel's neighborhood
loop, the cost-volume kernel's per-`d` accumulator, and — the heaviest user — the SGM path kernel's
64-entry `prev`/`cur` arrays (256 bytes each; likely spilling to per-thread "local" memory, which is
cached but not a true register file at this size, a documented perf/clarity trade-off). No
`__constant__` memory (nothing here is read identically by every thread the way MPPI's nominal control
sequence is) and no texture memory (a natural fit for future work — bilinear-filtered texture fetches
are exactly what real-time stereo libraries use for the 2-D neighborhood reads, another honest
"exercise, not the teaching core" boundary).

## Numerical considerations

- **Everything here is INTEGER arithmetic — a genuine departure from most of this repository.** Pixel
  intensities are `uint8_t`; census comparisons are `<` on those integers; the cost is a `popcount`
  (an exact integer count of set bits); the SGM recurrence is integer `min`/`+`/`-`. There is
  **no floating point anywhere in the disparity-computing path** — no rounding to reason about, no
  ULP differences between GPU and CPU trigonometric intrinsics (contrast MPPI's `sinf`/`cosf` story),
  no FMA-fusion ambiguity. This is WHY main.cu's VERIFY stage can demand, and gets, **bit-for-bit
  equality** between the GPU and `reference_cpu.cpp` on every checkpoint — measured: 0 mismatches
  across census (221,184 signatures), the cost volume (7,077,888 entries), one SGM path
  (7,077,888 entries), and both final disparity maps (110,592 pixels each). Floating point appears
  EXACTLY ONCE in this whole project: the good-pixel-RATE percentage printed at the end, computed once
  from the (already-verified-identical) GPU disparity map — a diagnostic, never a value the VERIFY
  gate depends on.
- **The running-min subtraction is not cosmetic.** Without subtracting `min_k L_r(p-r,k)` at every
  step of the SGM recurrence, `L_r` would drift upward by roughly `P1` or `P2` EVERY step along an
  arbitrarily long path (384 or 288 steps here) — for `P2=48` that is a worst-case drift near 18,000,
  eventually risking `int32` overflow on a long enough image even though the MEANINGFUL signal (which
  `d` is smallest) never needs values that large. Subtracting the running minimum at each step keeps
  every `L_r(p,d)` within a small, bounded range near the DATA term's own scale (Hirschmüller 2008
  proves the bound formally) — the standard SGM implementation trick, applied here and explained at
  its point of use in `kernels.cu`.
- **Invalid-value sentinels are checked explicitly, everywhere, rather than relying on arithmetic to
  "naturally" produce a sane answer.** `kCensusInvalid` (all-ones `uint64_t`), `kCostInvalid` (255,
  outside the real 0–48 Hamming range), and `kInvalidDisp` (255, outside the real 0–63 disparity range)
  are each impossible to produce from real data, by construction — every kernel checks for them by
  name rather than hoping a large-but-legal value loses an argmin comparison by coincidence.
- **Occlusion is handled at TWO levels, and they answer different questions.** The GROUND TRUTH
  (`gt_valid.pgm`) marks pixels the SCENE says are genuinely invisible in the other view — known
  exactly, because the synthetic generator built the scene and can ask "did this pixel win the
  z-buffer" directly (`data/README.md`). The ALGORITHM's own left-right check marks pixels where BM or
  SGM's own two independent answers (left-referenced, right-referenced) disagree — a RUN-TIME,
  no-ground-truth-needed estimate of the same idea. They usually, but do not always, agree (a wrong
  match can occasionally still pass the LR check by coincidence, and a correct match near an occlusion
  boundary can occasionally fail it) — the ground-truth gate in `main.cu` measures the ALGORITHM's
  output against the SCENE's truth precisely to catch that gap, not just re-derive the same number
  the algorithm already computed about itself.
- **The scene is authored directly in disparity space, not derived from a metric 3-D model+camera.**
  A legitimate, documented simplification (not a limitation of the theory above, which the "physics
  first" section derives in full metric terms): `make_synthetic.py` picks integer disparities per
  scene region directly, and the "Z = f·B/d" physical reading in this file's first section is a
  POST-HOC illustrative interpretation (assuming a representative `f, B`), not something the generator
  needs to compute. This keeps the ground-truth pipeline exact-integer throughout (matching the
  algorithm's own all-integer nature above) and avoids a whole extra layer of floating-point
  camera-projection rounding that a metric-scene generator would introduce for no teaching benefit.

## How we verify correctness

Two independent, and independently NECESSARY, checks — this project can be numerically perfect and
still algorithmically unconvincing, or vice versa:

1. **VERIFY (GPU vs. CPU, exact equality).** `main.cu` runs census, the cost volume, ONE SGM
   aggregation path (L→R), and both final disparity maps (BM's full pipeline and SGM's full pipeline,
   each independently re-derived by `reference_cpu.cpp` from raw pixels — not by reusing the GPU's
   intermediate arrays, so the CPU path is a genuinely separate implementation) through both the GPU
   kernels and their CPU twins, and requires EXACT equality on every one of the five checkpoints (see
   "Numerical considerations" for why exact equality is the right bar here, not a tolerance). Measured:
   0 mismatches everywhere. This catches indexing bugs, D-major stride errors, off-by-one border
   handling, and left/right symmetric-reuse mistakes INSTANTLY — any such bug flips at least one bit
   somewhere, and this project has no rounding noise to hide behind.
2. **The ground-truth gate (RESULT).** Independently of VERIFY, the GPU's disparity outputs are scored
   against the scene's exact, dense ground truth: BM's good-pixel rate must clear 45% (measured
   63.35%), SGM's must clear 85% (measured 97.52%), and SGM must exceed BM by at least 15 points
   (measured 34.17). This catches everything the pointwise VERIFY check cannot: a badly-tuned P1/P2
   that makes SGM behave no better than BM while still being internally self-consistent, a
   left-right-check tolerance so loose it accepts garbage, or a scene design (README Exercise 2 invites
   breaking exactly this) that fails to actually exercise the smoothness prior.

The committed sample (`data/sample/`) makes both checks run fully offline, and the three demo artifacts
(`disparity_bm.pgm`, `disparity_sgm.pgm`, `error_map.pgm`) make the SECOND check's result inspectable
by eye, not just pass/fail (`demo/README.md`).

## Where this sits in the real world

- **libSGM (fixstars)** and **OpenCV's `cuda::StereoSGM`** are real-time, production CUDA
  implementations of the SAME algorithm family this project teaches: 8-direction (not 4) aggregation,
  heavily shared-memory-tiled kernels (this project's Exercise 4), and typically a Census+Hamming data
  term very close to this project's own — the biggest structural differences from this teaching version
  are exactly the ones named above: more paths, tiled memory access, and often sub-pixel disparity
  refinement (fitting a parabola to the cost curve around the integer minimum for finer-than-1-pixel
  precision, which this project's all-integer design deliberately forgoes — see "Numerical
  considerations").
- **NVIDIA VPI** ships stereo disparity estimation accelerated on Jetson's PVA (Programmable Vision
  Accelerator) as well as the GPU — the production answer, on embedded robotics compute, to "run this
  every camera frame within budget" (PRACTICE.md §2/§3 for the hardware tiers).
- **Active depth sensors** (structured-light or time-of-flight RGB-D cameras — think a wrist-mounted 3D
  camera in the manipulator-cell reference design) sidestep the correspondence PROBLEM entirely by
  projecting known structured light and measuring its return directly, at the cost of range (typically
  a few meters), outdoor robustness (sunlight swamps the projected pattern), and often higher power
  draw. Passive stereo (this project's family) is the answer when you need long range, outdoor
  operation, or cannot add an active emitter (covert operation, multi-sensor interference) — the
  trade-off named in README "In production" and grounded further in PRACTICE.md §2.
- **Mutual Information, not Hamming distance, as SGM's original data term** (Hirschmüller 2008) — MI is
  more robust to certain non-uniform radiometric distortions (e.g. a gamma curve difference between
  cameras) than census/Hamming, at a much higher per-pixel compute cost; most modern real-time
  implementations use Census/Hamming (as this project does) as the practical middle ground.
- **What the full production version adds beyond this teaching core:** 8-direction aggregation
  (Exercise 3), shared-memory-tiled census and aggregation kernels (Exercise 4), sub-pixel disparity
  refinement, confidence-based filtering beyond the left-right check used here, and — increasingly —
  learned matching costs and even fully learned stereo networks (RAFT-Stereo and similar), which trade
  this project's fully-interpretable Hamming-distance cost for a trained CNN feature comparison at
  significantly higher compute cost and (usually) higher accuracy on real, noisy, out-of-distribution
  scenes. This project's synthetic ground truth, by construction, cannot say which family wins on real
  photographs — Exercise 5 is the invitation to find out.
