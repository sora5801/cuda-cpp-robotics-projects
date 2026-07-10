// ===========================================================================
// kernels.cuh — interface for project 15.01
//               Minimum-snap trajectory optimization batched over waypoint
//               sets (quadrotor-style 2-D waypoint flight)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (batch generation, orchestration, the two
// verification stages, and the artifact writer), kernels.cu (the GPU
// per-problem solve), and reference_cpu.cpp (the CPU oracle twin). Every
// layout and every constraint-counting decision that all three files must
// agree on is defined HERE, once (CLAUDE.md §12).
//
// The problem in five lines (THEORY.md derives it properly)
// -----------------------------------------------------------
//   1. A waypoint SET is 5 waypoints (x,y) that a quadrotor must fly through.
//   2. Between consecutive waypoints sits a SEGMENT: a degree-7 polynomial
//      in a NORMALIZED time tau in [0,1] (4 segments total, one per axis).
//   3. Snap = the 4th time-derivative of position; minimizing its integral
//      is the classic "smooth as physically achievable" objective for a
//      differentially-flat quadrotor (THEORY.md §the-problem derives why).
//   4. Position-interpolation + zero-derivative endpoint conditions +
//      interior continuity turn "minimize snap" into ONE square 32x32
//      LINEAR SYSTEM per axis per waypoint set (THEORY.md §the-math counts
//      the 32 equations exactly) — no QP, no free variables, just Gaussian
//      elimination with partial pivoting.
//   5. One GPU thread solves ONE waypoint set (both axes) end to end: K
//      waypoint sets are fully independent, so K threads is the natural
//      map — the same thread-per-problem pattern 33.01 teaches, scaled from
//      N=6 (register-resident) to N=32 (LOCAL-memory-resident; see the
//      header comment in kernels.cu for why that distinction matters).
//
// STATE LAYOUT — SI units throughout (meters; tau is unitless normalized
// segment time), documented once here (CLAUDE.md §12):
//
//   WAYPOINTS (host+device), one set:  float wp[kNumWaypoints * 2]
//       wp[i*2+0] = x_i (m), wp[i*2+1] = y_i (m), i = 0..4, world frame
//       (x-forward, y-left per SYSTEM_DESIGN.md §3.2). A BATCH of K sets is
//       stored batch-contiguous: waypoints[k*kWaypointFloatsPerSet + ...].
//
//   COEFFICIENTS (host+device), one set: float coeffs[kCoeffsPerSet]
//       Axis-major, then segment-major, then power-major:
//         coeffs[axis*kSysN + seg*kCoeffsPerSegment + j]
//       axis: 0 = x, 1 = y.  seg: 0..3.  j: 0..7 (coefficient of tau^j).
//       Segment s's position law: p_s(tau) = sum_j coeffs[...][j] * tau^j,
//       tau in [0,1] mapped to real time by t = seg*kSegmentDurationS + tau
//       * kSegmentDurationS (equal segment times — a documented exercise
//       extends this to time-allocated, unequal segments).
//       A batch of K sets is batch-contiguous: coeffs[k*kCoeffsPerSet + ...].
//
// THE 32x32 LINEAR SYSTEM — constraint accounting (THEORY.md §the-math
// derives this by degrees-of-freedom counting; restated here as the single
// source of truth for row indices, used identically by kernels.cu and
// reference_cpu.cpp):
//
//   32 unknowns per axis = kNumSegments(4) * kCoeffsPerSegment(8).
//   Rows  0..7  (8):  position interpolation, 2 per segment (p_s(0)=wp_s,
//                     p_s(1)=wp_{s+1}).
//   Rows  8..13 (6):  GLOBAL endpoint conditions — zero velocity,
//                     acceleration, jerk (derivatives d=1,2,3) at the very
//                     start (segment 0, tau=0) and the very end (segment 3,
//                     tau=1). The physically-motivated "start and end at
//                     rest, not jerking" boundary condition.
//   Rows 14..31 (18): INTERIOR continuity at the 3 interior waypoints
//                     (between segments 0-1, 1-2, 2-3): derivatives
//                     d=1..6 (velocity, acceleration, jerk, snap, crackle,
//                     pop) must agree across the boundary. Going all the
//                     way to d=6 (not stopping at jerk or snap) is not a
//                     stylistic choice — it is FORCED by the DOF count:
//                     8 + 6 + 3*d_max = 32  =>  d_max = 6. THEORY.md §the-
//                     math shows this arithmetic and is honest about what
//                     it means relative to the "free-derivative" QP form
//                     of true minimum-snap (Mellinger & Kumar 2011).
//   Total: 8 + 6 + 18 = 32 equations for 32 unknowns — square, and (for
//   this constraint structure) independent of the waypoint VALUES, so it
//   is solved by plain Gaussian elimination with partial pivoting, never
//   singular for the well-posed constraint layout used here.
//
// Read this after: README.md, THEORY.md.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Problem-shape constants — shared verbatim by the GPU solver, the CPU
// oracle, and main.cu's batch generation / verification / artifact code.
// One source of truth (CLAUDE.md §12): a mismatch here would make "batched"
// solve the wrong-shaped problem silently.
// ---------------------------------------------------------------------------
constexpr int kNumWaypoints      = 5;   // waypoints per set (fixed, ratified scope)
constexpr int kNumSegments       = 4;   // = kNumWaypoints - 1
constexpr int kCoeffsPerSegment  = 8;   // degree-7 polynomial: c0..c7
constexpr int kSysN              = kNumSegments * kCoeffsPerSegment;  // 32: one axis's linear system size
constexpr int kNumAxes           = 2;   // 0 = x, 1 = y (2-D ratified scope; 3-D is README Exercise)
constexpr int kCoeffsPerSet      = kNumAxes * kSysN;              // 64 floats per waypoint set
constexpr int kWaypointFloatsPerSet = kNumWaypoints * 2;          // 10 floats per waypoint set (interleaved x,y)

constexpr int kNumInterior       = kNumSegments - 1;  // 3 interior waypoints (between segments)
constexpr int kMaxContinuityDeriv = 6;                // interior continuity goes up to POP (d=6) — see header derivation
constexpr int kMaxEndpointDeriv   = 3;                // endpoints pin velocity, accel, jerk (d=1..3) to zero

// Fixed, EQUAL segment duration (seconds) — "fixed equal segment times" per
// the ratified scope; proper time allocation (unequal, trajectory-aware
// segment durations) is a documented README exercise. Because every segment
// shares this SAME duration, continuity of the NORMALIZED-tau derivative at
// a segment boundary is equivalent to continuity of the PHYSICAL derivative
// (the constant 1/kSegmentDurationS^d scale factor cancels on both sides —
// THEORY.md §numerical-considerations spells this out) — that equivalence
// is what lets kernels.cu work entirely in tau and never divide by T.
constexpr float kSegmentDurationS = 1.0f;                                  // s, per segment
constexpr float kTrajDurationS    = kNumSegments * kSegmentDurationS;      // s, whole trajectory

// Batch / synthetic-generation defaults (main.cu; documented here so the
// PROBLEM: line and the data docs agree with the code that uses them).
constexpr int   kDefaultBatch   = 10000;  // waypoint sets per demo run
constexpr float kBoxHalfExtentM = 4.0f;   // random waypoints drawn from [-4,4]^2 m
constexpr float kMinSpacingM    = 0.75f;  // minimum consecutive-waypoint spacing, m

// ---------------------------------------------------------------------------
// launch_minsnap_batch — solve K independent minimum-snap waypoint sets on
// the GPU, one thread per set (both axes).
//
//   K            : number of waypoint sets (>= 1).
//   d_waypoints  : DEVICE pointer, K*kWaypointFloatsPerSet floats — layout
//                  above. Never written.
//   d_coeffs     : DEVICE pointer, K*kCoeffsPerSet floats, OUT — layout
//                  above. A set whose linear system was (unexpectedly)
//                  singular is filled with NaN (see kernels.cu) — the same
//                  fail-loud policy 33.01 uses, so a bug can never look like
//                  a plausible answer.
//
// Launch: one thread per waypoint set, 256-thread blocks (grid math +
// reasoning lives with the kernel in kernels.cu).
// ---------------------------------------------------------------------------
void launch_minsnap_batch(int K, const float* d_waypoints, float* d_coeffs);

// ---------------------------------------------------------------------------
// minsnap_batch_cpu — the CPU oracle twin (reference_cpu.cpp): identical
// constraint layout and identical Gaussian-elimination algorithm, sequential
// over k. main.cu runs it against the GPU on the SAME batch and requires
// agreement within a documented tolerance (the §5 GPU-vs-CPU gate).
// Pointers here are HOST pointers with the same shapes as the GPU twins.
// ---------------------------------------------------------------------------
void minsnap_batch_cpu(int K, const float* waypoints, float* coeffs);

#endif // PROJECT_KERNELS_CUH
