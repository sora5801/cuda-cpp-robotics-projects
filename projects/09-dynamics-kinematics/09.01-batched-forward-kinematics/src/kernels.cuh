// ===========================================================================
// kernels.cuh — interface for project 09.01
//               Batched forward kinematics (10⁵ configurations — the
//               foundation for everything above)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (driver), kernels.cu (GPU implementation),
// and reference_cpu.cpp (the correctness oracle). Everything the three files
// must agree on — the robot-model layout, the configuration layout, and the
// pose layout — is defined HERE, once (CLAUDE.md §12: state layouts are
// single-sourced).
//
// The robot model (what "forward kinematics" consumes)
// ----------------------------------------------------
// A serial chain of NJ revolute joints. Frames follow the repo conventions
// (docs/SYSTEM_DESIGN.md §interface conventions): right-handed, SI units,
// transforms named T_parent_child ("child expressed in parent"). Joint j
// contributes:
//
//     T_link(j-1)_link(j)(q_j) = T_fix(j) · Rot(axis_j, q_j)
//
// where T_fix(j) is the FIXED transform from the previous link frame to
// joint j's frame (the link geometry: where the next joint sits and how it
// is oriented — this is what a URDF <origin> tag encodes), and
// Rot(axis_j, q_j) is the joint's rotation by angle q_j (radians) about the
// unit vector axis_j expressed in joint j's own frame. The end-effector pose
// is the product down the chain:
//
//     T_base_ee(q) = Π_{j=0..NJ-1}  T_fix(j) · Rot(axis_j, q_j)
//
// MODEL LAYOUT — 10 floats per joint, flat array of NJ*10 floats:
//     [ tx ty tz   qw qx qy qz   ax ay az ]
//       ---------  -----------   --------
//       t of T_fix (m)           unit joint axis (joint frame)
//                  quaternion of T_fix's rotation, (w,x,y,z) — REPO ORDER,
//                  normalized (CLAUDE.md §12: order documented at every
//                  API boundary)
//
// CONFIGURATION LAYOUT: q[k*NJ + j] = angle of joint j in configuration k,
// radians, wrapped to (-π, π] by the producer (we only read them).
//
// POSE LAYOUT (the output; deliberately message-shaped like a ROS 2
// geometry_msgs/Pose): 7 floats per configuration,
//     pose[k*7 + 0..2] = p (m)        end-effector position in the base frame
//     pose[k*7 + 3..6] = (w,x,y,z)    end-effector orientation quaternion,
//                                     normalized, hemisphere NOT canonical-
//                                     ized (q and -q are the same rotation —
//                                     comparators must handle the double
//                                     cover; see main.cu).
//
// Why the model rides in __constant__ memory (and thus needs a setter):
// every thread of every block reads the SAME NJ*10 floats. Constant memory
// is cached and BROADCASTS a uniform read to a whole warp in one shot —
// the textbook use case. The kernels.cu translation unit owns the
// __constant__ symbols; hosts push the model through set_robot_model().
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// Hard cap on chain length, sized generously for serial arms (6-7 DoF is
// typical; humanoid limbs reach ~7; we leave headroom). It bounds the
// __constant__ buffer and the CPU oracle's stack scratch. Compile-time so
// both sides agree by construction.
constexpr int kMaxJoints = 16;

// Floats per joint in the flat model layout documented above.
constexpr int kModelStride = 10;

// Floats per output pose (3 position + 4 quaternion), documented above.
constexpr int kPoseStride = 7;

// ---------------------------------------------------------------------------
// set_robot_model — upload the robot description to GPU __constant__ memory.
//
//   nj    : number of joints, 1..kMaxJoints (anything else aborts loudly).
//   model : HOST pointer, nj*kModelStride floats in the layout above. The
//           fixed-rotation quaternions must be normalized and the joint axes
//           unit-length — the loader in main.cu validates this once at load
//           time so the per-thread inner loop doesn't have to.
//
// Must be called before launch_batched_fk (the launcher aborts if not).
// Cheap (a few hundred bytes); typically called once per robot, not per
// batch — mirroring how a real stack loads a URDF once at startup.
// ---------------------------------------------------------------------------
void set_robot_model(int nj, const float* model);

// ---------------------------------------------------------------------------
// launch_batched_fk — GPU forward kinematics for a batch of configurations.
//
//   count  : number of configurations K (>= 0; 0 is a valid no-op).
//   d_q    : DEVICE pointer, K*nj floats — joint angles (rad), layout above.
//   d_pose : DEVICE pointer, K*kPoseStride floats OUT — end-effector poses
//            (p in meters in the base frame, quaternion (w,x,y,z)).
//
// Launch configuration: one THREAD per configuration, 256-thread blocks,
// ceil(K/256) blocks — the same thread-per-problem pattern as project 33.01
// (see that project first if this is your entry point). Reasoning with the
// kernel in kernels.cu.
// ---------------------------------------------------------------------------
void launch_batched_fk(int count, const float* d_q, float* d_pose);

// ---------------------------------------------------------------------------
// CPU reference (defined in reference_cpu.cpp — the correctness oracle).
// Same math, same layouts, plain single-threaded C++; takes the model as an
// ordinary argument (no constant memory on a CPU — that difference IS the
// lesson of the pair). All pointers are HOST pointers.
// ---------------------------------------------------------------------------
void batched_fk_cpu(int nj, const float* model,
                    int count, const float* q, float* pose);

#endif // PROJECT_KERNELS_CUH
