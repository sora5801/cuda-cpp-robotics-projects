// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 25.08
//               (Fuel-cell polarization + balance-of-plant sim)
//
// TEMPLATE PLACEHOLDER — replace with this project's real interface.
// TODO(scaffold): declare the real kernels/launchers here, each with a full
// doc-comment (purpose, params with units/frames, launch config, memory
// spaces) mirroring the definition in kernels.cu.
//
// Why ".cuh"?
// -----------
// The repo convention (CLAUDE.md §12): .cuh headers may contain CUDA-specific
// constructs (__global__, __device__, kernel launches) and are meant to be
// included from nvcc-compiled .cu files; plain .h headers stay host-only.
// This particular header is ALSO included by reference_cpu.cpp, which is
// compiled by the HOST compiler (cl.exe) — cl.exe does not know the word
// __global__, so the device-only declarations below are fenced behind
// #ifdef __CUDACC__ (a macro only nvcc defines). This is the standard trick
// for headers shared across the host/device boundary; you will see it in
// most projects in this repository.
//
// What belongs in this file
// -------------------------
// Declarations ONLY — the contract between translation units:
//   * __global__ kernel signatures (device side; nvcc-only),
//   * host-callable launch wrappers (visible to everyone),
//   * the CPU reference prototype (so main.cu and reference_cpu.cpp agree on
//     the signature at compile time instead of drifting apart silently).
// Definitions live in kernels.cu (GPU) and reference_cpu.cpp (CPU oracle).
//
// Read this after: main.cu.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH   // classic include guard: safe on every compiler
#define PROJECT_KERNELS_CUH   // (#pragma once also works; the guard is the more portable teaching choice)

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// saxpy_kernel — grid-stride map computing y[i] = a * x[i] + y[i].
// Full documentation (thread mapping, coalescing, numerics) sits with the
// definition in kernels.cu; headers carry the one-line summary, definitions
// carry the essay — so there is exactly one place to keep deeply in sync.
// x, y are DEVICE pointers of n floats; y is updated in place.
__global__ void saxpy_kernel(int n, float a,
                             const float* __restrict__ x,
                             float*       __restrict__ y);

#endif // __CUDACC__ --------------------------------------------------------

// launch_saxpy — host wrapper owning the grid/block math + post-launch error
// check. d_x/d_y are DEVICE pointers the caller allocated (see main.cu).
// Declared outside the __CUDACC__ fence: it is a plain host function, so any
// translation unit may call it (only its DEFINITION needs nvcc).
void launch_saxpy(int n, float a, const float* d_x, float* d_y);

// saxpy_cpu — the CPU correctness oracle (defined in reference_cpu.cpp).
// Same math, plain C++, single thread. x/y are HOST pointers of n floats;
// y is updated in place. Declared here so the compiler enforces that the GPU
// path and the oracle share one signature.
void saxpy_cpu(int n, float a, const float* x, float* y);

// TODO(scaffold): replace the three declarations above with this project's
// real kernel/launcher/reference interface.

#endif // PROJECT_KERNELS_CUH
