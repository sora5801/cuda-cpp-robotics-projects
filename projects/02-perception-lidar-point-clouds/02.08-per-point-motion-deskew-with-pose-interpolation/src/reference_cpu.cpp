// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.08
//                     (Per-point motion deskew with pose interpolation)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5):
//   1) It is the CORRECTNESS ORACLE: main.cu runs both paths and asserts
//      element-wise agreement within a documented tolerance.
//   2) It is the TEACHING BASELINE: the GPU version is a transformation OF
//      this file — reading this first, then kernels.cu, shows exactly what
//      parallelization changed (spoiler: the for-loop became threads; the
//      per-point math is byte-for-byte the SAME shared function).
//
// Independence ruling for THIS project (see the general ruling that follows
// — this file's header is the canonical place it is quoted, per repo
// convention, so a reader who opens only this file still gets the full
// argument):
//
//   * Data-layout contracts (the trajectory array layout, the point arrays'
//     layout, the Vec3/Quat/Pose structs) are single-sourced in
//     kernels.cuh — shared, not duplicated (CLAUDE.md §12).
//   * The quaternion/vector ALGEBRA (quat_slerp, quat_rotate, quat_mul, ...)
//     and the one-point deskew formula (deskew_one_point) are declared HD in
//     kernels.cuh and called VERBATIM from both this file and kernels.cu's
//     kernel — the "shared __host__ __device__ helper" exception CLAUDE.md
//     §5's independence ruling permits, because retyping quaternion algebra
//     by hand a second time would be pure token-for-token transcription
//     (exactly 01.10's precedent for its own camera-model primitives — see
//     that project's kernels.cuh for the same argument made about rotation
//     integration instead of rotation interpolation).
//   * The OUTER LOOP below — reading a point, calling deskew_one_point,
//     writing the result — IS typed independently here, a second time, as a
//     plain sequential for-loop (kernels.cu's kernel does the SAME three
//     steps per thread, but the two pieces of code are not copy-pasted from
//     one another).
//   * Because the twin comparison (VERIFY stage, main.cu) is therefore BLIND
//     to any bug living inside deskew_one_point/quat_slerp themselves (a bug
//     there would reproduce IDENTICALLY on both paths and the twins would
//     agree perfectly while both being wrong), this project carries the
//     INDEPENDENT gates the ruling requires: IDENTITY_CONTROL,
//     RESTORATION, SAMPLING_LESSON, and SLERP_CORRECTNESS (main.cu) all
//     compare against ANALYTIC ground truth from
//     ../scripts/make_synthetic.py's continuous trajectories, or against a
//     closed-form geodesic-angle formula — neither of which routes through
//     deskew_one_point at all. THEORY.md "How we verify correctness" walks
//     through why each gate catches a DIFFERENT class of bug.
//
// Rules for the loop below: plain C++17, no CUDA headers, no
// hand-vectorization, no OpenMP, no cleverness — clarity beats speed here,
// always (the GPU speed-up is only legible if this baseline is honest).
//
// Read this after: kernels.cu — then compare the two functions side by side
// (they are almost the same three lines, wrapped in a different loop shape).
// ===========================================================================

#include "kernels.cuh"   // deskew_cpu prototype: compiler-enforced signature
                         // agreement with what main.cu calls, and the shared
                         // HD deskew_one_point this loop calls into.

// ---------------------------------------------------------------------------
// deskew_cpu — sequential twin of deskew_kernel (kernels.cu).
//
// Parameters:
//   n_points  — point count (>= 0; 0 is a valid no-op).
//   t_points  — [n_points] host floats, firing time (s) per point.
//   xyz_local — [n_points*3] host floats, SKEWED local coordinates (m).
//   traj      — [n_samples*kTrajStride] host floats, the trajectory for THIS
//               call's regime (dense or sparse — see kernels.cuh). Unlike
//               the GPU path, this is a PLAIN host array, not a
//               __constant__-memory upload: there is no such thing as
//               "constant memory" on a CPU, so the trajectory is simply
//               passed as an ordinary argument (the same host/device
//               asymmetry 09.01's set_robot_model/batched_fk_cpu pair
//               documents for its own model buffer).
//   n_samples — trajectory sample count for this call.
//   ref_pose  — the precomputed pose at t_ref (computed once by main.cu,
//               shared with the GPU path — see main.cu "Two regimes, one
//               array").
//   xyz_out   — [n_points*3] host floats OUT, deskewed reference-frame coordinates (m).
//
// Complexity: O(n_points * log(n_samples)) — the binary search inside
// interpolate_pose, run once per point (kernels.cuh's find_bracket_index
// comment discusses why this never dominates at this project's n_samples).
// Side effects: overwrites xyz_out. Determinism: fully deterministic —
// there is no RNG anywhere in this pipeline; the only source of run-to-run
// float variation is compiler/ISA-level FMA fusion differences between
// cl.exe and nvcc, which is exactly what main.cu's VERIFY tolerance exists
// to absorb (THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
void deskew_cpu(int n_points, const float* t_points, const float* xyz_local,
                const float* traj, int n_samples, Pose ref_pose, float* xyz_out)
{
    // One loop, three lines — deliberately the simplest correct statement
    // of the computation, so a reader can verify it BY EYE against the
    // derivation in kernels.cuh's deskew_one_point comment. This is the
    // EXACT per-thread body of deskew_kernel, wrapped in a sequential loop
    // instead of a thread index — the parallelization changed nothing about
    // the math, only who iterates.
    for (int i = 0; i < n_points; ++i) {
        const Vec3 p_local{ xyz_local[i * 3 + 0], xyz_local[i * 3 + 1], xyz_local[i * 3 + 2] };
        const Vec3 p_out = deskew_one_point(traj, n_samples, ref_pose, t_points[i], p_local);
        xyz_out[i * 3 + 0] = p_out.x;
        xyz_out[i * 3 + 1] = p_out.y;
        xyz_out[i * 3 + 2] = p_out.z;
    }
}
