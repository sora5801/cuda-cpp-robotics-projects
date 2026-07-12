// ===========================================================================
// kernels.cuh — interface for project 02.19
//               PointPillars/CenterPoint voxelization + scatter kernels
//               (Method A: atomic per-pillar slot claim.
//                Method B: sort + fixed-order deterministic truncation.)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates), kernels.cu (the
// GPU kernels + Thrust-based sort/compact plumbing), and reference_cpu.cpp
// (the independent CPU oracle twins). Every data-layout decision — the
// point-cloud layout, the pillar/voxel key formulas, the 9-D feature vector,
// the PFN-lite weights shape, the dense-canvas layout, the conv/NMS
// parameters — is defined HERE ONCE (CLAUDE.md §12; see 02.01's identical
// ruling in ITS kernels.cuh, which this project follows almost verbatim for
// the binning half of the pipeline).
//
// THE PROBLEM in one paragraph (THEORY.md "The problem" goes deeper)
// --------------------------------------------------------------------
// A raw LiDAR sweep is an unordered bag of N ~ 10^5 (x,y,z,intensity)
// points — exactly what project 02.01-02.18 upstream of this one produce
// after cleaning. Learned 3-D detectors (PointPillars, CenterPoint, and
// nearly everything TensorRT can run efficiently) want a DENSE, REGULAR
// tensor — a grid of fixed shape, the same shape every frame, so a
// convolutional/matmul-heavy network can run at a fixed FLOP count. This
// project is entirely about the ENGINEERING of that impedance mismatch:
// bin points into a bird's-eye-view (BEV) grid of "pillars" (a voxel that is
// infinitely tall — no z-split — the defining PointPillars simplification),
// cap each pillar's point count so the tensor has a fixed inner dimension,
// augment each kept point with geometric context a network cannot recover
// on its own, reduce each pillar to a small feature vector with a
// permutation-invariant pool, and SCATTER those per-pillar vectors back into
// a dense [C,H,W] canvas a conv head (here: a tiny hand-rolled one; in
// production: a learned CNN, then TensorRT — project 12.01) can consume.
// The catalog names two kernels explicitly: "voxelization" (steps 1-2 below)
// and "scatter" (step 5) — everything else in this header exists to make a
// runnable, end-to-end, TensorRT-free demo close the loop honestly.
//
// POINT CLOUD LAYOUT — float* points, interleaved, meters + reflectance:
//     points[i*4 + 0] = x, points[i*4+1] = y, points[i*4+2] = z,
//     points[i*4 + 3] = intensity (unitless reflectance, sensor-reported)
// This is the KITTI/PointPillars-native "raw velodyne .bin" layout (see
// data/README.md) — one field wider than 02.01/02.06's xyz-only PointCloud
// convention because pillarization's feature vector genuinely needs
// intensity (THEORY.md "The math" explains why).
//
// THE BEV GRID (bounded — the reason this project does NOT need 02.01's
// spatial hash table)
// --------------------------------------------------------------------------
// 02.01's voxel grid is UNBOUNDED (a LiDAR scan has no fixed extent), which
// is exactly why it needs a hash table: there is no way to allocate one
// array slot per possible voxel ahead of time. A BEV detector's input
// window, by contrast, is a DESIGN CHOICE — "we care about objects within
// +/-40 m of the ego vehicle" — so the grid is bounded and SMALL enough
// (kGridNX * kGridNY = 40,000 cells below) that every pillar gets a dense,
// unique array slot: cell index = iy*kGridNX + ix, no hashing, no
// collisions, no probing. THEORY.md "The GPU mapping" makes this contrast
// with 02.01 explicit — it is the single biggest data-structure difference
// between the two projects despite both being "bin points into cells."
//
//   kPillarSizeM = 0.4 m, kGridNX = kGridNY = 200 -> an 80 m x 80 m BEV
//   window, x,y in [kXMin, kXMin + kGridNX*kPillarSizeM) = [-40, 40) m each
//   (the real PointPillars paper uses a similar ~0.16-0.24 m pillar over a
//   ~70-80 m KITTI window; 0.4 m here keeps the demo's grid, and therefore
//   its runtime and printed artifacts, small while keeping the lesson
//   identical — a documented scope choice, README "Limitations").
//
// TWO BINNING METHODS, ONE 02.01 LINEAGE — the project's determinism lesson
// recast in ML-preprocessing form
// --------------------------------------------------------------------------
//   METHOD A — atomic per-pillar slot claim (atomic_bin_kernel below). Every
//   point atomically claims the NEXT free slot in its pillar via
//   atomicAdd(&point_count[cell], 1); if the claimed slot index is >= the
//   per-pillar cap (kMaxPointsPerPillar), the point is silently DROPPED
//   (never written — a real, deliberate truncation policy, not a bug). WHICH
//   points survive an over-full pillar depends on the ORDER atomicAdd calls
//   from different threads/blocks are serialized by the hardware scheduler —
//   an accident of execution, not a language guarantee (CUDA's memory model
//   promises the increments are atomic, never that they happen in thread-ID
//   order). main.cu's cap_truncation gate MEASURES this: it re-runs Method A
//   over several INPUT-ORDER PERMUTATIONS of the same logical point set (the
//   dominant, 100%-reproducible real-world source of this exact bug class —
//   packet reordering, multi-return interleaving, multi-sensor merge order —
//   see that gate's comment in main.cu and THEORY.md "Numerical
//   considerations" for why input-order shuffling stands in for raw
//   scheduler nondeterminism here) and reports how many of the kept points
//   differ. This is NOT an academic curiosity: which points a pillar keeps
//   changes that pillar's mean/offset FEATURES (below), which changes what
//   the network sees, run to run, from IDENTICAL sensor data — an ML
//   reproducibility bug hiding inside a "just a preprocessing kernel."
//
//   METHOD B — sort + fixed-order truncation (the sort/compact machinery in
//   kernels.cu, directly reusing 02.01's Method B lineage: compute keys ->
//   thrust::stable_sort_by_key -> mark segment boundaries -> compact). Points
//   are stable-sorted by pillar key, which turns "which points share a
//   pillar" into "a contiguous run in sorted order," and — because the sort
//   is STABLE — that run is always in ASCENDING ORIGINAL POINT INDEX order,
//   regardless of which order they arrived in. Truncation keeps the FIRST
//   kMaxPointsPerPillar points of that fixed order. Because the order is
//   fixed, Method B is BIT-EXACT: same input (in ANY arrival order, since
//   sorting removes order-dependence entirely) -> same kept points, every
//   time, on any GPU. Method B is the pipeline main.cu's primary VERIFY gate
//   and every downstream stage (features, PFN-lite, scatter, head) run
//   against.
//
// THE SAME MACHINERY, A DIFFERENT KEY FUNCTION (PointPillars vs CenterPoint)
// --------------------------------------------------------------------------
// launch_sort_and_compact() below is written generically over "N points, a
// per-point integer cell key, C possible cells" — it does not know or care
// whether the key came from pillar_key_of() (2-D, z-collapsed — the
// PointPillars encoder) or voxel_key_of() (3-D, z-split into kNumZBins
// bands — the coarse CenterPoint-style voxel encoder this project compares
// against). The catalog bullet names both networks in one breath because,
// at the KERNEL level, they share this exact binning machinery; they differ
// only in the key formula and in how many "layers" of pillars/voxels stack
// at one (ix,iy) — see THEORY.md "The problem" for the z-collapse trade this
// makes and main.cu's pillar_vs_voxel comparison for the measured memory/
// time cost of NOT collapsing z.
//
// Read this after: main.cu. Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // int32_t / uint32_t — exact-width integers for keys and RNG state
#include <cmath>     // std::floor (float overload == floorf; see pillar_coord)
#include <vector>    // std::vector<PeakCPU> — the CPU reference's peak-list output type

// ===========================================================================
// BEV grid geometry — single-sourced (main.cu, kernels.cu's device
// transcription, reference_cpu.cpp ALL read these same constants).
// ===========================================================================
constexpr float kPillarSizeM = 0.4f;     // pillar edge length in x AND y (m) — square pillars
constexpr int   kGridNX = 200;           // pillars along x
constexpr int   kGridNY = 200;           // pillars along y
constexpr float kXMin = -40.0f;          // BEV window: x in [kXMin, kXMin + kGridNX*kPillarSizeM) = [-40,40) m
constexpr float kYMin = -40.0f;          // BEV window: y in [kYMin, kYMin + kGridNY*kPillarSizeM) = [-40,40) m
constexpr float kZMin = -3.0f;           // points below this are out of the detector's vertical window
constexpr float kZMax = 5.0f;            // points above this are out of the detector's vertical window
constexpr int   kNumPillars = kGridNX * kGridNY;   // 40,000 — every pillar's dense array slot count

// CenterPoint-style 3-D voxel comparison: the SAME (x,y) grid, but z is now
// split into kNumZBins UNIFORM bands covering the SAME vertical window
// [kZMin,kZMax] pillars use, instead of collapsing z into one pillar.
// kZBandM is each band's thickness, derived (not hand-picked) so the two
// encodings cover an IDENTICAL physical volume — the only difference is
// whether z is collapsed (kNumZBins=1, pillars) or split (kNumZBins=2,
// voxels). At kZMin=-3, kZMax=5, kNumZBins=2, the band boundary lands at
// z = kZMin + kZBandM = -3 + 4 = 1.0 m — roughly separating ground/wheel-
// height returns (band 0) from car-body/roof returns (band 1) in this
// project's synthetic scene (THEORY.md "The math").
constexpr int   kNumZBins = 2;                                            // coarse: 2 z-bands
constexpr float kZBandM = (kZMax - kZMin) / static_cast<float>(kNumZBins); // uniform band thickness (m)
constexpr int   kNumVoxels = kNumPillars * kNumZBins;                     // 80,000 dense voxel slots

// Per-pillar point cap and the augmented feature width. kMaxPointsPerPillar
// is the SAME truncation constant both binning methods enforce (differently
// — see the file header); kNumPointFeatures = 9 is the classic PointPillars
// per-point feature vector: raw (x,y,z,intensity) + offset-from-pillar-mean
// (xc,yc,zc) + offset-from-pillar-geometric-center (xp,yp) — THEORY.md "The
// math" derives and names every one of the 9 terms.
constexpr int kMaxPointsPerPillar = 32;
constexpr int kNumPointFeatures   = 9;   // D: x,y,z,intensity,xc,yc,zc,xp,yp

// PFN-lite (Pillar Feature Net, teaching-scale stand-in for the real learned
// PFN — README "Limitations" and THEORY.md "Where this sits in the real
// world" both say so). Output channel layout per occupied pillar:
//   channel 0        = occupancy      = min(kept_count, cap) / cap        (EXPLICIT, hand-computed)
//   channel 1        = height extent  = (max_z - min_z among kept points) / kHeightNormM, clamped [0,1]
//                                                                          (EXPLICIT, hand-computed)
//   channels 2..5    = a fixed linear(D->4) + ReLU + max-pool-over-points  (the "real learned PFN" stand-in)
// kPfnChannels = 2 + kPfnLinOut is therefore the FULL channel count that
// gets scattered into the dense canvas.
constexpr int   kPfnLinOut    = 4;                            // learned-PFN-stand-in channel count
constexpr int   kPfnChannels  = 2 + kPfnLinOut;                // 6 total scattered channels
constexpr float kHeightNormM  = 2.0f;                          // height-extent normalizer (m); a 2 m tall
                                                                // return spread saturates channel 1 at 1.0

// Toy detection head (hand-designed 3x3 stencils — README/THEORY are explicit
// that these weights are HAND-PICKED to respond to spatially-clustered tall
// occupancy, never trained; see the file header's "impedance mismatch" note
// and THEORY.md "Where this sits in the real world" for what a LEARNED head
// would add). Two conv passes ("2-layer conv-ish", the catalog's closure
// requirement) with an elementwise occupancy gate between them.
constexpr int kPeakWindowR       = 2;    // local-max window is (2r+1)x(2r+1) = 5x5 pillars

// NMS suppression radius, in pillar units. MEASURED on this project's
// synthetic scene: because the cars are hollow boxes (4 side walls + roof,
// no solid interior — scripts/make_synthetic.py), occupancy/height-extent
// peak near the CORNERS, not the geometric center, so one car produces
// SEVERAL local maxima up to its full diagonal apart (length 4.2 m, width
// 1.8 m -> diagonal ~4.6 m). kNmsRadiusPillars must exceed that diagonal
// (12 pillars = 4.8 m here) so every car collapses to exactly one surviving
// peak; the six cars are >= 24 m apart (make_synthetic.py's CAR_CENTERS),
// far beyond 2x this radius, so no risk of merging two DIFFERENT cars.
constexpr int kNmsRadiusPillars  = 12;   // 4.8 m at kPillarSizeM — see the measurement note above

// The two conv passes' DESIGNED (never trained) weights — shared between
// main.cu's GPU path and its CPU reference calls so both run the identical
// head. Row-major 3x3, matching conv3x3_kernel/conv3x3_cpu's indexing
// [(ky+1)*3+(kx+1)].
//
// Layer 1 (smoothing, applied to channel 1 = height-extent): a normalized
// 3x3 box-ish blur. At an ISOLATED occupied pillar (empty neighbors), the
// center tap alone contributes 4/16 = 25% of the pillar's own value, so an
// isolated spike is heavily attenuated; at a pillar surrounded by other
// tall, occupied pillars (a real car spans ~11x5 pillars), neighbors
// contribute comparably, so a spatially-CLUSTERED tall region keeps most of
// its value. This is the "spatial coherence" signal the catalog's closure
// requirement needs (an isolated dense pillar, like the cap-stress test
// cell, must NOT read as an object) — THEORY.md "The math" derives the
// 25% figure.
constexpr float kSmoothKernel3x3[9] = {
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
    2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
};

// Layer 2 (sharpening, applied to the occupancy-gated smoothed plane): a
// unity-gain (weights sum to 1) sharpen/consolidate kernel — boosts a
// pixel relative to its immediate neighbors without changing a perfectly
// FLAT region's value (5 - 4*1 = 1), concentrating each object's response
// toward its footprint's interior before peak extraction.
constexpr float kSharpenKernel3x3[9] = {
     0.0f, -1.0f,  0.0f,
    -1.0f,  5.0f, -1.0f,
     0.0f, -1.0f,  0.0f,
};

// Detection threshold on the final heatmap. MEASURED on this project's
// committed sample (main.cu's [info] heatmap diagnostics line prints these
// same numbers every run): every car's STRONGEST surviving corner peak is
// >= 0.39, the isolated cap-stress pillar peaks at 0.179, and every
// clutter/ground pillar never clears even a much lower threshold at all
// (no candidate ever appears there). 0.22 sits roughly at the geometric
// mean of the two nearest real numbers (0.179 and 0.39) -- a wide,
// evidence-based margin on both sides, not a tuned-to-the-decimal constant.
constexpr float kDetectThreshold = 0.22f;

// ---------------------------------------------------------------------------
// pillar_coord — floor((p - min) / size) as an integer grid index along one
// axis. Same floor-not-truncate pitfall as 02.01's voxel_coord (its comment
// derives the negative-p bug this avoids; not repeated here — see
// 02.01/src/kernels.cuh "voxel_coord" for the full derivation this project
// cites rather than re-deriving).
// ---------------------------------------------------------------------------
inline int32_t pillar_coord(float p, float min_val, float size)
{
    return static_cast<int32_t>(std::floor((p - min_val) / size));
}

// ---------------------------------------------------------------------------
// pillar_key_of — the 2-D dense BEV cell index (PointPillars encoder).
//
// Parameters: x, y (m, sensor/BEV frame). Returns: iy*kGridNX + ix if the
// point's (ix,iy) lies inside the grid (0<=ix<kGridNX, 0<=iy<kGridNY),
// else -1 (an explicit "out of the detector's window" sentinel — the point
// is dropped from pillarization, exactly like a real detector's input
// filter; z-range filtering is the caller's job via z_in_range() below,
// since z does not participate in the pillar key at all — THE point of
// pillarization is that it does not).
// ---------------------------------------------------------------------------
inline int32_t pillar_key_of(float x, float y)
{
    const int32_t ix = pillar_coord(x, kXMin, kPillarSizeM);
    const int32_t iy = pillar_coord(y, kYMin, kPillarSizeM);
    if (ix < 0 || ix >= kGridNX || iy < 0 || iy >= kGridNY) return -1;
    return iy * kGridNX + ix;
}

// ---------------------------------------------------------------------------
// voxel_key_of — the 3-D dense grid index (CenterPoint-style coarse voxel
// encoder): the SAME (ix,iy) as pillar_key_of, plus an iz band. Layout
// iz*kNumPillars + (iy*kGridNX+ix) keeps every z-band's pillars contiguous —
// convenient for the memory-cost comparison in main.cu (band b occupies
// exactly [b*kNumPillars, (b+1)*kNumPillars) of key space).
// ---------------------------------------------------------------------------
inline int32_t voxel_key_of(float x, float y, float z)
{
    const int32_t ix = pillar_coord(x, kXMin, kPillarSizeM);
    const int32_t iy = pillar_coord(y, kYMin, kPillarSizeM);
    const int32_t iz = pillar_coord(z, kZMin, kZBandM);
    if (ix < 0 || ix >= kGridNX || iy < 0 || iy >= kGridNY || iz < 0 || iz >= kNumZBins) return -1;
    return iz * kNumPillars + (iy * kGridNX + ix);
}

// z_in_range — the vertical-window filter pillarization applies SEPARATELY
// from the (x,y) key (see pillar_key_of's comment: z never enters the key).
inline bool z_in_range(float z) { return z >= kZMin && z <= kZMax; }

// blocks_for — integer ceiling division (the same idiom 02.01/02.06/08.01 use).
inline int blocks_for(int count, int threads) { return (count + threads - 1) / threads; }

// ---------------------------------------------------------------------------
// PillarBinGPU — dense per-cell point storage, shared by Method A and
// Method B's device-side writers (kernels.cu). Sized by the CALLER
// (main.cu) to num_cells = kNumPillars (pillar binning only — the voxel
// comparison never materializes raw storage, see the file header's "same
// machinery" note and main.cu's pillar_vs_voxel section, which only needs
// occupied-cell COUNTS, not stored points).
// ---------------------------------------------------------------------------
struct PillarBinGPU {
    unsigned int* point_count;   // [num_cells] claimed-point counter (Method A: atomicAdd-incremented past
                                  // cap, so a value > cap_n IS the overflow signal, never clamped in place;
                                  // Method B: the true segment length, written once, never atomic)
    float* raw_points;           // [num_cells * cap_n * 4] each cell's kept points (x,y,z,intensity),
                                  // valid entries [0, min(point_count[cell], cap_n))
    int num_cells;               // kNumPillars for this project (voxel comparison does not use this struct)
    int cap_n;                   // kMaxPointsPerPillar
};

// ===========================================================================
// GPU kernel declarations — nvcc-only (cl.exe, compiling reference_cpu.cpp,
// has never heard of __global__ — same fence 02.01 uses, same reason).
// ===========================================================================
#ifdef __CUDACC__

// compute_pillar_keys_kernel — one thread per point: pillar_key_of(x,y) if
// z_in_range(z), else -1 (an out-of-window point, dropped from EVERY later
// stage). Device transcription of pillar_key_of/z_in_range above — the
// "shared token-for-token transcription" case 02.01's file header names;
// main.cu's VERIFY(keys) gate is the independent check that catches drift.
__global__ void compute_pillar_keys_kernel(int n, const float* __restrict__ points,
                                           int* __restrict__ keys);

// compute_voxel_keys_kernel — the 3-D-band twin of the kernel above, device
// transcription of voxel_key_of. Used only by the pillar_vs_voxel comparison.
__global__ void compute_voxel_keys_kernel(int n, const float* __restrict__ points,
                                          int* __restrict__ keys);

// reset_counts_kernel — one thread per CELL: zero point_count[cell] before a
// fresh Method-A pass (main.cu re-runs Method A across several input-order
// permutations for the cap_truncation determinism gate — each run needs a
// clean counter array).
__global__ void reset_counts_kernel(unsigned int* __restrict__ point_count, int num_cells);

// atomic_bin_kernel — Method A: one thread per point (key < 0 -> early
// return, out of window). atomicAdd claims this point's slot in its
// pillar; slots >= cap_n are claimed (the counter still increments, so
// point_count[cell] truthfully reports the total that ARRIVED) but not
// written to raw_points — the drop. Full walkthrough with the ordering
// story in kernels.cu.
__global__ void atomic_bin_kernel(int n, const float* __restrict__ points,
                                  const int* __restrict__ keys, PillarBinGPU bin);

// mark_boundaries_kernel — one thread per SORTED-ARRAY position: 1 where a
// new cell's run begins (position 0, or key changed from the previous
// position), 0 otherwise. Generic over pillar OR voxel keys (the "same
// machinery" note) — direct reuse of 02.01's Method B pattern.
__global__ void mark_boundaries_kernel(int n, const int* __restrict__ keys_sorted,
                                       int* __restrict__ is_start);

// gather_occupied_cell_kernel — one thread per OCCUPIED segment: read that
// segment's cell id (keys_sorted at its first position) into a dense,
// compacted occupied_cell[] list — the "pillar list" every later stage
// (features, PFN-lite, scatter) walks instead of the full 40,000-cell grid.
__global__ void gather_occupied_cell_kernel(int num_occupied, const int* __restrict__ seg_start,
                                            const int* __restrict__ keys_sorted,
                                            int* __restrict__ occupied_cell);

// sorted_bin_kernel — Method B's writer: one thread per OCCUPIED segment,
// walking its run in SORTED-ARRAY order (== ascending original point index,
// guaranteed by stable_sort) and copying the first min(run_len, cap_n)
// points into bin.raw_points at that pillar's dense slot. bin.point_count
// is set to the TRUE run length (not clamped) so downstream code can still
// see "how many points actually landed here" alongside "how many are kept."
__global__ void sorted_bin_kernel(int num_occupied, const int* __restrict__ seg_start, int n_sorted,
                                  const int* __restrict__ idx_sorted, const float* __restrict__ points,
                                  const int* __restrict__ occupied_cell, PillarBinGPU bin);

// augment_features_kernel — one thread per (occupied pillar, point slot)
// pair: emit the 9-D feature vector for slots < kept_count, zero-pad slots
// >= kept_count (the fixed [num_occ, cap_n, 9] tensor shape a real network
// requires; THEORY.md "The math" derives all 9 terms). mean_xyz is this
// pillar's mean over its KEPT points (computed by pfn_stats_kernel below
// first — augment MUST run after it).
__global__ void augment_features_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                                        PillarBinGPU bin, const float* __restrict__ mean_xyz,
                                        float* __restrict__ features_out);

// pfn_stats_kernel — one thread per occupied pillar: mean (x,y,z) over its
// kept points (feeds augment_features_kernel's offset-from-mean terms) AND
// the kept-point count min(point_count[cell],cap_n) (feeds pfn_lite_kernel,
// which never sees `bin` directly — see that kernel's comment for why).
// Small, sequential, per-thread loop (cap_n <= 32 points) — no parallel
// reduction needed at this scale (THEORY.md "The GPU mapping" explains why
// this is the right call here and when it would stop being one).
__global__ void pfn_stats_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                                 PillarBinGPU bin, float* __restrict__ mean_xyz_out,
                                 unsigned int* __restrict__ kept_count_out);

// pfn_lite_kernel — one thread per occupied pillar: the tiny fixed-weight
// "PFN-lite" (file header + kernels.cuh constants above describe the 6
// output channels). Reads the 9-D AUGMENTED features (not raw points) —
// exactly what a real PFN consumes.
__global__ void pfn_lite_kernel(int num_occupied, const float* __restrict__ features,
                                const unsigned int* __restrict__ kept_count,
                                const float* __restrict__ lin_w /*[kPfnLinOut*kNumPointFeatures]*/,
                                const float* __restrict__ lin_b /*[kPfnLinOut]*/,
                                float* __restrict__ pillar_feat_out /*[num_occupied*kPfnChannels]*/);

// scatter_kernel — the catalog's second named kernel: one thread per
// occupied pillar, looping kPfnChannels writes into the dense
// [kPfnChannels, kGridNY, kGridNX] canvas at that pillar's (iy,ix). Every
// cell NOT in occupied_cell[] stays at its caller-zeroed value — the sparse-
// to-dense trade THEORY.md "The GPU mapping" and main.cu's
// sparsity_economics gate both measure.
__global__ void scatter_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                               const float* __restrict__ pillar_feat, float* __restrict__ canvas);

// gather_kernel — the roundtrip check's other half: one thread per occupied
// pillar, reading canvas back at (iy,ix) into gathered_out — main.cu's
// layout_roundtrip gate compares this, entry for entry, against the
// pillar_feat that was scattered (a pure copy: must be BIT-IDENTICAL).
__global__ void gather_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                              const float* __restrict__ canvas, float* __restrict__ gathered_out);

// conv3x3_kernel — the "conv-as-stencil" kernel: one thread per output
// pixel (iy,ix), zero-padded boundary, 3x3 weighted sum + bias. Generic and
// reused for BOTH head layers with different weight arrays (the smoothing
// pass on the height-extent channel, and the sharpening pass on the gated
// plane) — one kernel, two designed weight sets, "2-layer conv-ish" per the
// catalog closure requirement.
__global__ void conv3x3_kernel(const float* __restrict__ in_plane, int h, int w,
                               const float* __restrict__ kernel3x3 /*[9], row-major*/, float bias,
                               float* __restrict__ out_plane);

// elementwise_mul_kernel — one thread per pixel: out[i] = a[i]*b[i]. Used as
// the head's occupancy GATE between the two conv passes (documented as
// fusion logic, not a third conv layer).
__global__ void elementwise_mul_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       int size, float* __restrict__ out);

// peak_extract_kernel — one thread per pixel: candidate = 1 iff value >
// threshold AND value is the STRICT max (ties broken by lowest flattened
// index, see kernels.cu) in its (2*kPeakWindowR+1)^2 neighborhood. main.cu's
// NMS pass (host, small candidate count) consumes is_candidate.
__global__ void peak_extract_kernel(const float* __restrict__ heatmap, int h, int w, float threshold,
                                    int window_r, unsigned char* __restrict__ is_candidate);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu; declarations
// here are plain C++, visible to main.cu regardless of compiler).
// ===========================================================================

void launch_compute_pillar_keys(int n, const float* d_points, int* d_keys);
void launch_compute_voxel_keys(int n, const float* d_points, int* d_keys);
void launch_reset_counts(unsigned int* d_point_count, int num_cells);
void launch_atomic_bin(int n, const float* d_points, const int* d_keys, PillarBinGPU bin);

// launch_sort_and_compact — the GENERIC sort+compact machinery (file header
// "same machinery, different key function"): filters out-of-window points
// (key < 0), stable-sorts the rest by key, marks segment boundaries, and
// compacts the occupied cell ids into occupied_cell_out. Every scratch/out
// array below is sized >= n by the caller (a safe upper bound — occupied
// cells and valid points can never exceed n).
//
//   d_keys_in          : [n] per-point cell key (pillar OR voxel — this
//                         function does not care which), -1 = out of window.
//   d_keys_scratch      : [n] SCRATCH — becomes the sorted, COMPACTED (valid-
//                         only) key array.
//   d_idx_scratch       : [n] SCRATCH — becomes the sorted permutation of
//                         ORIGINAL point indices (idx_scratch[k] = which
//                         input point now sits at sorted position k).
//   d_is_start_scratch  : [n] SCRATCH — the segment-boundary 0/1 mask.
//   d_seg_start_out     : [n] OUT (first *n_valid_out entries meaningful) —
//                         sorted-array offset where each occupied cell's run
//                         begins; d_seg_start_out[num_occupied] is set to
//                         n_valid so callers can compute the LAST segment's
//                         length uniformly (requires the array be sized
//                         >= n+1 — main.cu allocates n+1 for exactly this).
//   d_occupied_cell_out : [n] OUT (first return-value entries valid) — the
//                         compacted list of occupied cell ids, in ASCENDING
//                         cell-key order (a side effect of the sort).
//   n_valid_out         : OUT — how many of the n input points were in-
//                         window (key >= 0) and therefore participated.
//
// Returns: the number of occupied cells (== valid rows in *_out arrays).
int launch_sort_and_compact(int n, const int* d_keys_in, int* d_keys_scratch, int* d_idx_scratch,
                            int* d_is_start_scratch, int* d_seg_start_out, int* d_occupied_cell_out,
                            int* n_valid_out);

void launch_sorted_bin(int num_occupied, const int* d_seg_start, int n_sorted, const int* d_idx_sorted,
                       const float* d_points, const int* d_occupied_cell, PillarBinGPU bin);

void launch_pfn_stats(int num_occupied, const int* d_occupied_cell, PillarBinGPU bin,
                      float* d_mean_xyz_out, unsigned int* d_kept_count_out);
void launch_augment_features(int num_occupied, const int* d_occupied_cell, PillarBinGPU bin,
                             const float* d_mean_xyz, float* d_features_out);
void launch_pfn_lite(int num_occupied, const float* d_features, const unsigned int* d_kept_count,
                     const float* d_lin_w, const float* d_lin_b, float* d_pillar_feat_out);
void launch_scatter(int num_occupied, const int* d_occupied_cell, const float* d_pillar_feat,
                    float* d_canvas);
void launch_gather(int num_occupied, const int* d_occupied_cell, const float* d_canvas,
                   float* d_gathered_out);
void launch_conv3x3(const float* d_in_plane, int h, int w, const float* d_kernel3x3, float bias,
                    float* d_out_plane);
void launch_elementwise_mul(const float* d_a, const float* d_b, int size, float* d_out);
void launch_peak_extract(const float* d_heatmap, int h, int w, float threshold, int window_r,
                         unsigned char* d_is_candidate);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. Per the independence ruling
// (docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's file header): the key/
// feature-index FORMULAS above are single-sourced and shared; the
// ALGORITHMIC CORE (sorting, pooling, convolving, peak search) below is
// written independently, in its own loop structure, not a transcription of
// the GPU kernels.
// ===========================================================================

// The CPU pipeline is deliberately STAGE-GRANULAR (one function per GPU
// pipeline stage, mirrored dense-storage shapes but independently-looped
// bodies) rather than one monolithic call — so main.cu can VERIFY each
// stage (keys, binning, features, PFN, scatter, conv, peaks) separately,
// the same granular "VERIFY(keys)/VERIFY(method_b)/..." style 02.01 uses.

// pillar_keys_cpu — the twin of compute_pillar_keys_kernel, calling this
// header's OWN pillar_key_of/z_in_range. main.cu's VERIFY(keys) gate
// compares this, point for point, against the GPU's device-transcribed
// version (the gate that catches drift between the two copies).
void pillar_keys_cpu(int n, const float* points, int* keys_out);

// voxel_keys_cpu — the 3-D-band twin, same role for voxel_key_of.
void voxel_keys_cpu(int n, const float* points, int* keys_out);

// count_occupied_cpu — a tiny, fully independent (std::vector<bool>
// presence array, one pass) occupied-cell counter, used by main.cu's
// pillar_vs_voxel comparison to cross-check the GPU's sort-based occupied
// count for BOTH the pillar and the voxel grid without going through the
// sort/compact machinery at all — a genuinely different algorithm
// (presence-array vs sort) for a gate that must not share a bug with what
// it is checking.
int count_occupied_cpu(int n, const float* points, bool use_voxel_keys, int num_cells);

// sorted_bin_cpu — Method B's independent twin: std::stable_sort (NOT
// Thrust — a different library, a different algorithm class, satisfying
// "independent" for the sort itself) by pillar_key_of, then truncates each
// pillar's run to the first kMaxPointsPerPillar points in that stable
// (== ascending original index) order — the identical rule
// sorted_bin_kernel enforces via a different code path.
//
// Storage mirrors PillarBinGPU's DENSE, per-cell shape exactly (a shared
// data-layout contract, per the independence ruling — only the FORMULAS/
// shapes are shared, not the traversal code):
//   point_count_dense : [kNumPillars] OUT, TRUE arrival count per cell
//                        (caller must zero-initialize; untouched cells stay 0).
//   raw_points_dense  : [kNumPillars * kMaxPointsPerPillar * 4] OUT, kept
//                        points per cell (first min(count,cap) slots valid).
//   occupied_cell_out : [n] OUT (first return-value entries valid) — compacted
//                        occupied cell ids, ascending key order.
//   kept_count_out    : [n] OUT, aligned with occupied_cell_out.
//   mean_xyz_out      : [n*3] OUT, aligned with occupied_cell_out.
// Returns: the number of occupied pillars.
int sorted_bin_cpu(int n, const float* points,
                   unsigned int* point_count_dense, float* raw_points_dense,
                   int* occupied_cell_out, unsigned int* kept_count_out, float* mean_xyz_out);

// augment_features_cpu — the 9-D feature-tensor twin of augment_features_kernel
// (zero-padding beyond kept_count included). features_out is COMPACT,
// indexed by occupied-pillar rank (matching the GPU's layout exactly):
// [num_occupied * kMaxPointsPerPillar * kNumPointFeatures].
void augment_features_cpu(int num_occupied, const int* occupied_cell, const unsigned int* kept_count,
                          const float* mean_xyz, const float* raw_points_dense, float* features_out);

// pfn_lite_cpu — the fixed PFN-lite twin of pfn_lite_kernel. pillar_feat_out
// is [num_occupied * kPfnChannels], compact.
void pfn_lite_cpu(int num_occupied, const float* features, const unsigned int* kept_count,
                  const float* lin_w, const float* lin_b, float* pillar_feat_out);

// scatter_cpu — the twin of scatter_kernel. canvas_out is
// [kPfnChannels*kGridNY*kGridNX], caller-zeroed.
void scatter_cpu(int num_occupied, const int* occupied_cell, const float* pillar_feat, float* canvas_out);

// conv3x3_cpu — the twin of conv3x3_kernel (identical zero-padding policy,
// independently looped).
void conv3x3_cpu(const float* in_plane, int h, int w, const float* kernel3x3, float bias, float* out_plane);

// PeakCPU — one surviving detection after NMS.
struct PeakCPU { int iy; int ix; float score; };

// peak_extract_and_nms_cpu — independent host twin of peak_extract_kernel's
// local-max rule (identical tie-break: a strictly larger neighbor always
// wins; an equal neighbor wins only at a smaller flattened index) PLUS the
// NMS pass main.cu's GPU path also runs on the host (candidate counts are
// tiny after thresholding — no kernel needed on either path).
void peak_extract_and_nms_cpu(const float* heatmap, int h, int w, float threshold, int window_r,
                              int nms_radius_pillars, std::vector<PeakCPU>& peaks_out);

#endif // PROJECT_KERNELS_CUH
