// ===========================================================================
// kernels.cu — the one GPU kernel for project 01.10 (Rolling-shutter
//              correction using IMU rates)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the small host-side launch
// wrapper and the __constant__-memory row-LUT uploader. Everything this file
// needs to know about the camera model, the quaternion algebra, and the row
// homography is defined ONCE in kernels.cuh and included unchanged.
//
// Why only ONE kernel? (a design note, not an omission)
// -------------------------------------------------------
// Two computations happen before a single output pixel can be produced:
//   (1) integrate ~10 sparse (200 Hz) gyro samples into a dense orientation
//       trajectory, and collapse it into kImgH (=288) per-row relative
//       quaternions — a SEQUENTIAL recurrence (each fine integration step
//       depends on the previous one) over a tiny amount of data. Forcing
//       this onto the GPU would mean either one thread doing all the work
//       (no parallelism gained) or a parallel-scan-style reformulation that
//       would dwarf the actual computation in complexity, for data this
//       small. main.cu does it on the HOST, once per gyro variant.
//   (2) resolve, for each of kImgW*kImgH = 110,592 OUTPUT pixels, which raw
//       rolling-shutter pixel it came from, and bilinearly sample it — this
//       IS embarrassingly parallel (every pixel's answer is independent of
//       every other pixel's) and IS the computation this project exists to
//       teach on a GPU. That is rs_correct_kernel below.
// Step (1)'s output (the row LUT) is small enough (kImgH quaternions, ~4.6
// KiB) to live in __constant__ memory rather than a plain device buffer —
// see set_row_lut()'s comment for why that is a genuine, not just
// decorative, choice here.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>    // std::fprintf — the H-mismatch abort message in set_row_lut
#include <cstdlib>   // std::exit

// ---------------------------------------------------------------------------
// c_row_lut — the currently-active gyro variant's per-row relative-
// quaternion LUT, in GPU __constant__ memory.
//
// Why __constant__ and not a plain cudaMalloc'd buffer (like d_rs_frame)?
// Two reasons, one structural and one about the ACCESS PATTERN:
//   * Structural: __constant__ memory can only be WRITTEN from the host
//     (cudaMemcpyToSymbol) — never from a kernel. Since the row LUT is
//     built entirely on the host (kernels.cu's file header explains why),
//     there is no conflict: nothing ever needs to write it from device
//     code, so the read-only restriction costs nothing here.
//   * Access pattern: this project's kernel launch groups OUTPUT pixels
//     into 32(x)x8(y) blocks (see launch_rs_correct below), so every warp
//     (32 consecutive threads along x) shares the SAME yo, and therefore
//     starts its 3-iteration search with the SAME v_guess = yo — i.e., the
//     FIRST LUT read of every warp is to the EXACT SAME address. Constant
//     memory caches a single fetch and BROADCASTS it to the whole warp in
//     one transaction — the textbook constant-memory use case (same one
//     09.01's robot model relies on). Later iterations may drift the 32
//     threads' v_guess slightly apart (each pixel's own convergence), but
//     the constant cache still absorbs the resulting handful of distinct
//     addresses cheaply — nowhere near the cost of an L2-only global read
//     for 110,592 threads each issuing kFixedPointIters LUT reads.
// Sized exactly kImgH (288 entries * 16 bytes = 4,608 bytes) — comfortably
// under the ~64 KiB constant-memory budget with room to spare.
// ---------------------------------------------------------------------------
__constant__ Quat c_row_lut[kImgH];

// ---------------------------------------------------------------------------
// set_row_lut — see kernels.cuh for the full parameter contract. Uploads
// host_lut into c_row_lut above via cudaMemcpyToSymbol (the ONLY legal way
// to write __constant__ memory). Aborts loudly on an H mismatch rather than
// silently truncating/overrunning — a wrong H here would corrupt every
// row lookup for the rest of the run, and a loud abort is far cheaper to
// debug than a subtly-wrong image.
// ---------------------------------------------------------------------------
void set_row_lut(const Quat* host_lut, int H)
{
    if (H != kImgH) {
        std::fprintf(stderr, "set_row_lut: H=%d does not match kImgH=%d (kernels.cuh contract)\n", H, kImgH);
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_row_lut, host_lut, sizeof(Quat) * static_cast<size_t>(kImgH)));
}

// ---------------------------------------------------------------------------
// rs_correct_kernel — one thread per OUTPUT (reference/GS) pixel.
//
// Thread-to-data mapping: thread (blockIdx.x, threadIdx.x, blockIdx.y,
// threadIdx.y) owns output pixel
//     xo = blockIdx.x * blockDim.x + threadIdx.x
//     yo = blockIdx.y * blockDim.y + threadIdx.y
// A 2-D grid (rather than this repo's more common 1-D grid-stride map) is
// used specifically so that blockDim.x = 32 lines up ONE WARP with ONE
// output ROW — see c_row_lut's comment above for why that matters for the
// constant-memory broadcast.
//
// Per-thread work (independent of every other thread — the whole reason
// this is a trivial GPU map): run the kFixedPointIters-iteration row-time
// search described in kernels.cuh's file header, then bilinearly sample
// the raw RS frame at the resolved source pixel. Registers only — no
// shared memory (nothing is reused BETWEEN threads: each thread's search
// path through the row LUT is its own, so there is nothing to tile).
//
// Parameters: see launch_rs_correct's doc-comment in kernels.cuh — this
// kernel takes plain DEVICE pointers/dims; the launcher below owns the
// grid/block math.
// ---------------------------------------------------------------------------
__global__ void rs_correct_kernel(const unsigned char* __restrict__ rs_frame,
                                  unsigned char* __restrict__ corrected,
                                  unsigned char* __restrict__ valid_mask,
                                  float* __restrict__ iter_delta,
                                  int W, int H)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's output column
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's output row
    if (xo >= W || yo >= H) return;                          // guard the ragged edge blocks

    // v_guess: the row-time search state (file header derives the fixed-
    // point argument). Seeded at yo — a good initial guess because this
    // project's rotation rates only shift the true source row by a few
    // pixels away from yo (README "Expected output" reports the measured
    // shift), so the search starts close to its answer.
    float v_guess = static_cast<float>(yo);
    float xs = 0.0f, ys = 0.0f;    // resolved source pixel, updated each iteration
    float v_after_second_iter = 0.0f;  // snapshot for the convergence gate (see below)

    for (int it = 0; it < kFixedPointIters; ++it) {
        // Look up (and linearly interpolate) the relative rotation for the
        // CURRENT row guess, convert to a matrix once, apply it to the
        // FIXED output pixel (xo, yo) — only v_guess changes between
        // iterations, xo/yo never do (file header's "output pixel is
        // fixed; iterate on WHICH ROW's rotation to use" framing).
        const Quat q_rel = lerp_row_quat(c_row_lut, H, v_guess);
        float R[9];
        quat_to_mat3(q_rel, R);
        apply_row_rotation(R, static_cast<float>(xo), static_cast<float>(yo), xs, ys);

        // Snapshot the row estimate after the SECOND iteration (0-based
        // it == kFixedPointIters-2) so the delta below is "final minus
        // second-to-last" — exactly the "iteration-2 vs iteration-3 max
        // delta" README/THEORY.md report as the convergence diagnostic,
        // for the repo-standard kFixedPointIters = 3.
        if (it == kFixedPointIters - 2) v_after_second_iter = ys;
        v_guess = ys;   // the new row guess: wherever this iteration's source pixel landed
    }

    const int idx = yo * W + xo;
    iter_delta[idx] = fabsf(ys - v_after_second_iter);   // how much the LAST iteration still moved the guess

    int valid = 0;
    const float sample = bilinear_sample_gray(rs_frame, W, H, xs, ys, valid);
    // Round-to-nearest for the float->uint8 write (matches this repo's
    // usual "+0.5 then truncate" convention, e.g. 01.01's normalized_to_vis).
    corrected[idx] = valid ? static_cast<unsigned char>(sample + 0.5f) : static_cast<unsigned char>(0);
    valid_mask[idx] = static_cast<unsigned char>(valid);
}

// ---------------------------------------------------------------------------
// launch_rs_correct — owns the launch geometry (see kernels.cuh for the
// full parameter contract).
//
// Launch configuration reasoning:
//   block = (32, 8) = 256 threads — 32 along x to make blockDim.x exactly
//     one warp (the c_row_lut broadcast argument above requires this;
//     picking, say, (16,16) would split a warp across two output rows and
//     lose the "whole warp reads the same address" property at iteration
//     0). 8 along y keeps the total at 256, this repo's usual sweet spot.
//   grid = (ceil(W/32), ceil(H/8)) — covers every output pixel exactly
//     once; the kernel's own (xo>=W || yo>=H) guard handles the ragged
//     edge (kImgW=384 and kImgH=288 both divide evenly by 32 and 8
//     respectively for THIS project's fixed resolution, so the guard is
//     technically inert on the committed sample — kept anyway because the
//     kernel should not silently corrupt memory if someone changes kImgW/
//     kImgH without re-checking divisibility).
// ---------------------------------------------------------------------------
void launch_rs_correct(const unsigned char* d_rs_frame, unsigned char* d_corrected,
                       unsigned char* d_valid_mask, float* d_iter_delta, int W, int H)
{
    const dim3 block(32, 8);
    const dim3 grid((static_cast<unsigned>(W) + block.x - 1) / block.x,
                    (static_cast<unsigned>(H) + block.y - 1) / block.y);

    rs_correct_kernel<<<grid, block>>>(d_rs_frame, d_corrected, d_valid_mask, d_iter_delta, W, H);

    // Kernel launches report configuration errors asynchronously; catch it
    // at the launch site, not several calls later (CLAUDE.md paragraph 6.1
    // rule 7 — every launch in this repo is checked, visibly).
    CUDA_CHECK_LAST_ERROR("rs_correct_kernel launch");
}
