// ===========================================================================
// kernels.cuh — interface for project 19.01
//               Parallel grasp-candidate scoring: antipodal sampling over
//               point clouds (two-finger parallel-jaw grippers)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the pipeline driver), kernels.cu (the GPU
// kernels), and reference_cpu.cpp (the CPU oracle twins). Everything all
// three must agree on — the point-cloud layout, the candidate/score record
// layouts, and every tuning constant that affects a bit-for-bit or
// tolerance-bound comparison — is defined HERE, once (CLAUDE.md §12: state
// layouts are single-sourced).
//
// The pipeline, in four stages (main.cu orchestrates; THEORY.md derives why)
// -------------------------------------------------------------------------
//   1. NORMALS   — PCA over k=16 nearest neighbors + in-register Jacobi
//                  eigensolve, ONE thread per cloud point. This is
//                  project 02.06's exact pattern (brute-force k-NN, cyclic
//                  Jacobi on the 3x3 covariance) — reused here and cited,
//                  not reinvented; see estimate_normals_kernel below for
//                  the one policy change (normal ORIENTATION: outward, not
//                  inward — kernels.cuh's "Numerics" note explains why).
//   2. CANDIDATES — sample K=4096 contact points p1 (one per thread, chosen
//                  by a deterministic hash of a candidate index — no RNG
//                  state, no host-generated stream to upload, unlike
//                  08.01's noise array), then walk the INWARD normal ray
//                  from p1 and brute-force-scan the WHOLE cloud for the
//                  best "exit point" p2 on the far side (nearest to the
//                  ray, within a coarse opposing-normal cone). This reuses
//                  02.06's find_correspondences SHAPE (one thread, one
//                  full-cloud scan, uniform loop bound -> no warp
//                  divergence) applied to a fundamentally different SEARCH
//                  (a ray-proximity query, not a nearest-neighbor query).
//   3. SCORING   — per candidate: the Coulomb friction-cone / force-closure
//                  test (THEORY.md derives it from first principles — the
//                  project's mathematical heart), the gripper stroke gate,
//                  and an approximate finger-clearance gate. One thread per
//                  candidate; a SECOND full-cloud scan per thread (the
//                  clearance check) — GPU MAPPING note below explains why
//                  this, and not a reduction, is the right shape here.
//   4. RANKING   — top-M selection by descending score. Host std::sort at
//                  K~4096: the same honest "this doesn't need a GPU kernel
//                  at this N" call project 12.01 makes for its greedy NMS
//                  step (see README "Prior art"). Lives in main.cu, not
//                  here, because it touches no device memory.
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters (docs/SYSTEM_DESIGN.md
// §3.6 PointCloud convention, the same layout every perception project in
// this repo speaks, 02.06 included):
//     xyz[i*3 + 0] = x, xyz[i*3 + 1] = y, xyz[i*3 + 2] = z
//
// NORMAL LAYOUT — float* normals, SAME interleaved layout, unit vectors,
// oriented OUTWARD (away from the object's interior — see "Numerics" below).
// Computed once per object cloud by launch_estimate_normals; read-only after.
//
// GraspCandidate / GraspScore — the two per-candidate record layouts. See
// their struct definitions below; both are plain arrays of POD structs
// (array-of-structs, not the parallel-arrays style 02.06 uses for its
// correspondence record) because a candidate's ~10 fields are always read
// and written together — one struct per candidate keeps every download,
// every CSV row, and every CPU/GPU comparison a single, obviously-matching
// unit instead of eight parallel float*/int* arrays that could silently
// drift out of index-alignment with each other.
//
// Why this header is (mostly) CUDA-qualifier-free — except grasp_hash_u32
// -----------------------------------------------------------------------
// Every function below except grasp_hash_u32 is a plain host-callable
// launcher or a POD struct: no __global__/__device__ signatures live here.
// grasp_hash_u32 is the ONE exception: it must run identically inside a
// __global__ kernel (generate_candidates_kernel, one call per thread) AND
// inside reference_cpu.cpp's plain-C++ CPU twin (compiled by cl.exe, which
// does not know the words __host__/__device__ at all — they are macros
// defined only by CUDA headers, which reference_cpu.cpp deliberately never
// includes, CLAUDE.md §5). The #ifdef __CUDACC__ fence below therefore
// compiles TWO copies of the identical function body: a __host__ __device__
// copy nvcc sees (usable from both kernels.cu's device code and any host
// code in nvcc-compiled files), and a plain `inline` copy cl.exe sees
// (host-only, no CUDA keywords at all). One body, two qualifier sets — the
// same "declarations behind a device-aware fence" idea the template's
// original comment describes, extended to a tiny function DEFINITION that
// both the GPU kernel and its CPU oracle must call bit-for-bit identically
// (candidate generation is required to match EXACTLY, not just within
// tolerance — see generate_candidates_kernel's header comment).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// PCA normal-estimation constants — IDENTICAL values and roles to project
// 02.06's kPcaK / kJacobiSweeps (see that project's kernels.cuh for the full
// sizing rationale: k=16 is a standard PCL-style local-neighborhood size;
// 8 Jacobi sweeps is generous overkill for a 3x3 symmetric eigensolve,
// converging in practice within 3-5 sweeps).
// ---------------------------------------------------------------------------
constexpr int kPcaK         = 16;
constexpr int kJacobiSweeps = 8;

// Repo-default block size for the three thread-per-item kernels in this
// project (normals, candidate generation, scoring) — a warp multiple with
// good occupancy on sm_75..sm_89, the same default 02.06/08.01 use for
// their non-reduction kernels.
constexpr int kThreadsPerBlock = 256;

// ---------------------------------------------------------------------------
// Candidate-generation geometry constants. These are ALGORITHM TUNING
// parameters (how the ray-proximity search is shaped), not physical object
// properties — unlike the gripper stroke and friction coefficient, which
// vary per object/gripper and are therefore passed as RUNTIME parameters
// (read from data/sample/objects_meta.csv, see main.cu) rather than baked
// in here. THEORY.md "The algorithm" derives each of these from the point
// spacing of this project's synthetic clouds (~1.4-2 mm on the box,
// cylinder, and sphere samples at their committed point counts).
//
// kSearchPerpTolM — the search ray is treated as a THIN CYLINDER, not an
//   infinite thin line: a candidate partner point qualifies if its
//   perpendicular distance to the ray is within this tolerance. 6 mm is
//   ~3-4x the point spacing on every committed sample cloud, generous
//   enough that noise and finite sampling density cannot cause a genuine
//   antipodal partner to be missed, tight enough that the search still
//   behaves like "points near this ray", not "points near this plane".
constexpr float kSearchPerpTolM = 0.006f;

// kSearchTMinM — the minimum distance along the ray a partner must lie at.
//   Without a floor, the search could pair p1 with one of its own close
//   neighbors (near-duplicate points a few sample-spacings away, which by
//   chance can have a noisy normal that satisfies the coarse cone test) —
//   a degenerate near-zero-width "grasp". 5 mm is ~2.5-3x the point
//   spacing, comfortably excluding self-neighborhood matches while staying
//   far below every valid grasp width in this project's sample objects
//   (40-60 mm).
constexpr float kSearchTMinM = 0.005f;

// kSearchTMaxM — the maximum ray distance searched. Set intentionally
//   LARGER than the gripper's maximum stroke (see objects_meta.csv,
//   gripper_w_max_m, 90 mm) so the search can still FIND far-apart
//   antipodal pairs (e.g. the box's 100 mm long axis) — the scoring
//   kernel's width gate is what REJECTS them (kernels.cuh's whole point:
//   "geometrically antipodal" and "gripper-feasible" are different, and
//   this project's demo needs the search to surface both kinds of
//   candidate to teach the difference).
constexpr float kSearchTMaxM = 0.13f;

// kGenConeCosThreshold — the COARSE opposing-normal prefilter applied
//   during the search (not the precise force-closure test — that is
//   score_candidates_kernel's job, using the ACTUAL friction coefficient
//   and the ACTUAL grasp axis, not just the two surface normals). A
//   candidate partner's normal n_j must satisfy dot(n1, n_j) <=
//   this threshold, i.e. the angle between n1 and n_j must exceed 140
//   degrees (cos(140 deg) = -0.766). This is deliberately LOOSER than any
//   reasonable friction cone (atan(mu) for mu in [0.2, 1.0] is 11-45
//   degrees off antiparallel) — its only job is to keep the search from
//   wasting the "nearest point on the ray" slot on a point whose normal
//   makes it geometrically impossible to be a grasp partner, before the
//   scoring kernel applies the real test.
constexpr float kGenConeCosThreshold = -0.766f;   // cos(140 deg)

// ---------------------------------------------------------------------------
// Finger-clearance approximation constants (README/THEORY: "collision-free
// finger approach approximated by a clearance check in a slab around the
// grasp axis"). The slab is a cylinder of radius kClearanceRadiusM around
// the SEGMENT from p1 to p2, with a "dead zone" of length kClearanceDeadzoneM
// at each end excluded (the region right at each contact patch is, by
// definition, full of cloud points near p1/p2 themselves and their local
// neighborhood — that is not an obstruction, it is the contact).
// ---------------------------------------------------------------------------
constexpr float kClearanceRadiusM   = 0.004f;   // 4 mm: illustrative finger-half-thickness + margin
constexpr float kClearanceDeadzoneM = 0.008f;   // 8 mm: excluded region at each end of the segment

// Number of randomly-sampled candidates per object (the catalog bullet's
// "K"). 4096 matches 08.01's rollout count — large enough for a stable
// top-M, small enough that a full K x n_cloud brute-force scan (both the
// generation search and the scoring clearance scan) stays a small-millisecond
// GPU cost at this project's cloud sizes (README "What this computes").
constexpr int kNumCandidates = 4096;

// Number of grasps kept after ranking (the catalog bullet's "M").
constexpr int kTopM = 10;

// Sentinel score for an INFEASIBLE candidate (idx2 not found, or any gate
// failed). Strictly below every possible FEASIBLE score: antipodal_cos (the
// feasible score, see GraspScore below) is a cosine, so it lies in [-1, 1];
// -2.0 can never be confused with a real, if poor, feasible grasp, and a
// descending sort by score always puts every feasible candidate ahead of
// every infeasible one.
constexpr float kRejectedScore = -2.0f;

// ---------------------------------------------------------------------------
// GraspCandidate — one antipodal candidate BEFORE scoring: which two cloud
// points (by index into the SAME cloud's xyz/normals arrays) form the pair.
//
//   idx1 — always a valid index in [0, n) — the sampled contact point.
//   idx2 — index of the found antipodal partner, or -1 if the ray search
//          (generate_candidates_kernel) found no qualifying point.
//
// Filled by launch_generate_candidates for indices [0, kNumCandidates); the
// tail of a candidate array MAY also hold hand-picked ADVERSARIAL entries
// main.cu writes directly on the host (README/THEORY "verification" — the
// box's deliberately non-antipodal, adjacent-face pairs) so that the SAME
// scoring kernel scores both the random and the adversarial candidates with
// no special-casing.
// ---------------------------------------------------------------------------
struct GraspCandidate {
    int idx1;
    int idx2;
};

// ---------------------------------------------------------------------------
// GraspScore — the full per-candidate scoring record; this project's single
// definition of grasp "quality" (THEORY.md "The math" derives every field).
//
//   width_m       — |p2 - p1|, meters. 0 if idx2 < 0 (no candidate found).
//   antipodal_cos — dot(n1, -n2): the cosine of the angle between contact
//                   1's outward normal and contact 2's INWARD normal. 1.0 =
//                   perfectly opposed surface normals (the ideal antipodal
//                   pair); -1.0 = normals point the same way (never a valid
//                   grasp). This IS the ranking score for feasible grasps
//                   (see `score` below) — a direct, cheap proxy for grasp
//                   quality, distinct from the full epsilon-quality metric
//                   named in THEORY.md "Where this sits in the real world".
//   theta1_deg    — angle (degrees) between the grasp axis (p1->p2) and
//                   contact 1's INWARD normal (-n1). Small = the finger at
//                   p1 pushes almost exactly along the grasp line.
//   theta2_deg    — angle (degrees) between the reversed axis (p2->p1) and
//                   contact 2's INWARD normal (-n2). Same idea, contact 2.
//   friction_ok   — 1 iff BOTH theta1_deg and theta2_deg are within
//                   atan(mu) of zero — the Coulomb friction-cone /
//                   two-contact force-closure test THEORY.md derives.
//   width_ok      — 1 iff width_m lies within [w_min_m, w_max_m] (the
//                   gripper's stroke range, from objects_meta.csv).
//   clearance_ok  — 1 iff no OTHER cloud point lies inside the finger-sweep
//                   slab around the p1-p2 segment (kClearanceRadiusM /
//                   kClearanceDeadzoneM above) — the approximated
//                   collision-free-approach gate.
//   feasible      — 1 iff idx2 >= 0 AND friction_ok AND width_ok AND
//                   clearance_ok — every gate passed.
//   score         — feasible ? antipodal_cos : kRejectedScore. The single
//                   field main.cu sorts by for top-M ranking.
// ---------------------------------------------------------------------------
struct GraspScore {
    float width_m;
    float antipodal_cos;
    float theta1_deg;
    float theta2_deg;
    unsigned char friction_ok;
    unsigned char width_ok;
    unsigned char clearance_ok;
    unsigned char feasible;
    float score;
};

// ---------------------------------------------------------------------------
// grasp_hash_u32 — a deterministic, STATELESS counter-based hash: given a
// fixed seed and a candidate index k, returns the SAME 32-bit value on every
// call, on every device, every run. Used ONLY to pick candidate k's contact
// point p1 (generate_candidates_kernel: idx1 = grasp_hash_u32(seed, k) % n).
//
// Why hash-based instead of a stateful PRNG stream (like 08.01's host
// xorshift32 + Box-Muller noise array)? Because 08.01's noise must be
// GENERATED sequentially on the host and UPLOADED (one long stream, order
// matters, 800 KB/tick — its own THEORY.md numerics note names this cost).
// A counter-based hash needs no state and no stream: thread k computes
// grasp_hash_u32(seed, k) independently or its own draw, with NO dependency
// on any other thread's draw and NO host-side generation step at all. This
// is the same design philosophy as cuRAND's Philox generator (a counter-
// based CBRNG: seed + counter -> pseudorandom output, embarrassingly
// parallel by construction) — see README "Prior art". The specific mixing
// function below is "triple32" (Chris Wellons' public-domain integer hash,
// chosen for the same reason 02.06 chose Jacobi over the closed-form cubic:
// it is simple, branch-free, and has no known degenerate inputs, not
// because it is the only correct choice).
//
// EXACTNESS REQUIREMENT (THEORY.md "How we verify correctness"): candidate
// generation is the one stage of this pipeline required to match the CPU
// oracle output EXACTLY, index for index — not within a tolerance. That
// only holds if this function computes IDENTICAL bits on GPU and CPU, which
// is why it is defined ONCE, here, behind the __CUDACC__ fence, rather than
// re-implemented independently in kernels.cu and reference_cpu.cpp (two
// hand-copies of even three lines of bit-twiddling are exactly how such a
// contract silently drifts).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__   // ---- device-aware section: nvcc-compiled files (kernels.cu, main.cu) ----
__host__ __device__ inline unsigned int grasp_hash_u32(unsigned int seed, unsigned int counter)
{
    unsigned int x = seed ^ (counter * 0x9E3779B9u);   // fold the counter in with a Weyl-sequence constant
    x ^= x >> 16;  x *= 0x7feb352du;                    // triple32: three xor-shift/multiply rounds is
    x ^= x >> 15;  x *= 0x846ca68bu;                    // enough to fully avalanche a 32-bit hash — every
    x ^= x >> 16;                                        // output bit depends on every input bit
    return x;
}
#else               // ---- reference_cpu.cpp: plain cl.exe host compilation, no CUDA keywords ----
inline unsigned int grasp_hash_u32(unsigned int seed, unsigned int counter)
{
    unsigned int x = seed ^ (counter * 0x9E3779B9u);
    x ^= x >> 16;  x *= 0x7feb352du;
    x ^= x >> 15;  x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
#endif

// blocks_for — integer ceiling division: how many kThreadsPerBlock-wide
// blocks cover `count` independent items. Same idiom as 02.06/08.01/33.01.
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ---------------------------------------------------------------------------
// launch_estimate_normals — per-cloud-point surface normal via PCA over its
// kPcaK nearest neighbors (brute-force, self included), eigen-decomposed
// with an in-register Jacobi sweep — project 02.06's exact pattern, cited
// and reused (kernels.cu's kernel header comment credits it precisely).
//
//   n, d_xyz     : the cloud (n*3 floats, meters).
//   ref_point    : a point INSIDE the object (its centroid works for every
//                  convex solid this project samples — box, cylinder,
//                  sphere), used to pick a consistent per-point sign.
//   d_normals    : DEVICE pointer, n*3 floats OUT, unit vectors, oriented
//                  OUTWARD (away from ref_point) — see kernels.cu's
//                  "Numerics: normal orientation" note for why this project
//                  flips 02.06's inward-pointing convention.
//
// Launch: one thread per cloud point, kThreadsPerBlock-thread blocks.
// ---------------------------------------------------------------------------
void launch_estimate_normals(int n, const float* d_xyz,
                             const float ref_point[3], float* d_normals);

// ---------------------------------------------------------------------------
// launch_generate_candidates — GPU ray-proximity search: for each of the
// first kNumCandidates threads, hash-select a contact point p1, then
// brute-force-scan the WHOLE cloud for the best antipodal partner along
// p1's inward-normal ray (kernels.cu documents the search precisely).
//
//   n, d_xyz, d_normals : the cloud and its outward unit normals.
//   seed                : hash seed (main.cu passes a fixed, documented
//                          constant — determinism is repo law, CLAUDE.md §12).
//   num_candidates       : how many candidates to generate (always
//                          kNumCandidates in this project; a parameter
//                          rather than the constant directly so the CPU
//                          twin and a future "generate fewer for a quick
//                          test" caller share one signature).
//   d_candidates          : DEVICE pointer, num_candidates GraspCandidate
//                          records OUT (indices [0, num_candidates)).
//
// Complexity: O(num_candidates * n), embarrassingly parallel across
// candidates — the honest brute-force teaching choice (README Exercise
// names a spatial hash / grid acceleration as the natural next step, the
// same trade 02.06 documents for its correspondence search).
// Launch: one thread per candidate, kThreadsPerBlock-thread blocks.
// ---------------------------------------------------------------------------
void launch_generate_candidates(int n, const float* d_xyz, const float* d_normals,
                                unsigned int seed, int num_candidates,
                                GraspCandidate* d_candidates);

// ---------------------------------------------------------------------------
// launch_score_candidates — per-candidate friction-cone, gripper-width, and
// clearance gates, composited into one ranking score (kernels.cu documents
// the exact math, mirroring GraspScore's field-by-field comment above).
//
//   n, d_xyz, d_normals : the cloud and its outward unit normals (SAME
//                          cloud the candidates were generated from).
//   d_candidates, num_candidates : the candidates to score — may include
//                          hand-picked adversarial entries appended by
//                          main.cu (the scoring kernel does not care how a
//                          candidate's idx1/idx2 were chosen).
//   mu                  : Coulomb friction coefficient (unitless, from
//                          objects_meta.csv; illustrative, PRACTICE.md §2
//                          dates and caveats any implied hardware value).
//   w_min_m, w_max_m     : gripper stroke range, meters (from
//                          objects_meta.csv).
//   d_scores             : DEVICE pointer, num_candidates GraspScore
//                          records OUT.
//
// Launch: one thread per candidate, kThreadsPerBlock-thread blocks.
// ---------------------------------------------------------------------------
void launch_score_candidates(int n, const float* d_xyz, const float* d_normals,
                             const GraspCandidate* d_candidates, int num_candidates,
                             float mu, float w_min_m, float w_max_m,
                             GraspScore* d_scores);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — the correctness oracle twins of the
// three launchers above. Same math, same layouts, plain single-threaded
// C++, no CUDA anywhere. main.cu's VERIFY stage runs these against the GPU
// kernels on the box object's exact inputs (THEORY.md "How we verify
// correctness" explains why the box is the representative case and why
// each stage's tolerance is what it is). All pointers below are HOST
// pointers.
// ---------------------------------------------------------------------------
void estimate_normals_cpu(int n, const float* xyz, const float ref_point[3], float* normals);

void generate_candidates_cpu(int n, const float* xyz, const float* normals,
                             unsigned int seed, int num_candidates,
                             GraspCandidate* candidates);

void score_candidates_cpu(int n, const float* xyz, const float* normals,
                          const GraspCandidate* candidates, int num_candidates,
                          float mu, float w_min_m, float w_max_m,
                          GraspScore* scores);

#endif // PROJECT_KERNELS_CUH
