// ===========================================================================
// util/cuda_check.cuh — CUDA error-checking macros + device-info helper
//                       (project 13.05; copied from
//                       docs/PROJECT_TEMPLATE — see util/README.md for the
//                       deliberate-duplication rule)
//
// Role in the project
// -------------------
// EVERY CUDA runtime call and EVERY kernel launch in this repository is
// checked, visibly, at the call site (CLAUDE.md §6.1 rule 7). This header
// defines — ONCE per project — the two macros that make that cheap:
//
//   CUDA_CHECK(call)              wrap any cudaXxx(...) API call
//   CUDA_CHECK_LAST_ERROR(what)   place immediately after a kernel launch
//
// Why macros and not functions? Because a macro can capture __FILE__ and
// __LINE__ at the CALL SITE, so the error message points at the exact line
// that failed — a function would always report this header's line instead.
//
// What classes of failure do these catch?
// ---------------------------------------
// * cudaMalloc        -> cudaErrorMemoryAllocation (GPU out of memory).
// * cudaMemcpy        -> invalid pointers/sizes/directions (programming
//                        bugs), or a DEFERRED error from an earlier kernel —
//                        see the "sticky" note below.
// * kernel launches   -> errors are ASYNCHRONOUS. The launch itself returns
//                        immediately; an invalid configuration (too many
//                        threads, too much shared memory) is reported by
//                        cudaGetLastError() right after the launch, while a
//                        crash INSIDE the kernel (out-of-bounds access,
//                        assert) only surfaces at the next synchronizing
//                        call. CUDA_CHECK_LAST_ERROR catches the first class
//                        at the launch site; the second class is caught by
//                        CUDA_CHECK on the next cudaMemcpy/cudaEventSynchronize.
// * "Sticky" errors: once a kernel crashes, the CUDA context is poisoned and
//   EVERY later call returns the same error — which is why the message below
//   prints the failing call's text: the first report is the one that counts.
//
// Read this early — every other file in src/ uses these macros.
// ===========================================================================
#ifndef PROJECT_UTIL_CUDA_CHECK_CUH
#define PROJECT_UTIL_CUDA_CHECK_CUH

#include <cuda_runtime.h>  // cudaError_t, cudaGetErrorString, device queries
#include <cstdio>          // fprintf(stderr, ...) for the error report
#include <cstdlib>         // std::exit — fail HARD; a poisoned context can't continue

// ---------------------------------------------------------------------------
// CUDA_CHECK(call) — evaluate a CUDA runtime call and abort with a readable
// message if it did not return cudaSuccess.
//
// The do { ... } while (0) idiom: it wraps the multi-statement body so the
// macro behaves like a SINGLE statement in every syntactic position. Without
// it, this innocent-looking code breaks:
//
//     if (use_gpu)
//         CUDA_CHECK(cudaMalloc(&p, bytes));   // without do/while: only the
//     else                                     // first statement is guarded
//         ...                                  // and the 'else' won't parse!
//
// The while(0) loop runs exactly once and costs nothing; the trailing
// semicolon the caller writes completes the statement naturally.
//
// The double-underscore suffix on cuda_check_err__ avoids colliding with any
// caller variable that might share the name (macros paste text, not scopes).
// #call is the preprocessor "stringizer": it prints the literal text of the
// call so the message reads e.g.  CUDA error ... in 'cudaMalloc(&d_x, bytes)'.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t cuda_check_err__ = (call);                                 \
        if (cuda_check_err__ != cudaSuccess) {                                 \
            std::fprintf(stderr,                                               \
                "CUDA error %d (%s) at %s:%d in '%s'\n",                       \
                static_cast<int>(cuda_check_err__),                            \
                cudaGetErrorString(cuda_check_err__),                          \
                __FILE__, __LINE__, #call);                                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA_CHECK_LAST_ERROR(what) — place IMMEDIATELY after every kernel launch.
//
// Kernel launches (<<<...>>>) return void; their errors are retrieved via
// cudaGetLastError(), which also CLEARS the error flag (hence "last"). This
// catches launch-time failures — invalid grid/block dims, excessive shared
// memory, no compatible device code in the binary (the dreaded
// cudaErrorNoKernelImageForDevice when the GPU's sm_XX was not compiled in;
// see the CodeGeneration comment in build/*.vcxproj).
//
// 'what' is a short human label for the launch (e.g. "saxpy_kernel launch")
// so the report names the kernel, not just a file:line.
// ---------------------------------------------------------------------------
#define CUDA_CHECK_LAST_ERROR(what)                                            \
    do {                                                                       \
        cudaError_t cuda_check_err__ = cudaGetLastError();                     \
        if (cuda_check_err__ != cudaSuccess) {                                 \
            std::fprintf(stderr,                                               \
                "CUDA launch error %d (%s) at %s:%d after %s\n",               \
                static_cast<int>(cuda_check_err__),                            \
                cudaGetErrorString(cuda_check_err__),                          \
                __FILE__, __LINE__, (what));                                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// print_device_info — print a one-line "[info]" description of device 0.
//
// Purpose: (a) a friendly banner naming the GPU the demo ran on, and (b) the
// program's earliest loud failure point if NO CUDA device/driver is present
// (cudaGetDeviceProperties fails -> CUDA_CHECK aborts with a clear message
// instead of a mysterious crash later).
//
// The "[info]" prefix marks the line as NON-diffed demo output — device
// names differ across machines (see the output contract in main.cu).
// 'inline' because this lives in a header included by several .cu files;
// inline permits the multiple identical definitions the linker will see.
// ---------------------------------------------------------------------------
inline void print_device_info()
{
    cudaDeviceProp prop;                              // filled by the runtime: name, SM version, memory, ...
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));    // device 0: single-GPU assumption for demos
    std::printf("[info] GPU: %s (sm_%d%d, %zu MiB global memory)\n",
                prop.name,
                prop.major, prop.minor,               // compute capability, e.g. 7 and 5 -> sm_75
                prop.totalGlobalMem / (1024u * 1024u));
}

#endif // PROJECT_UTIL_CUDA_CHECK_CUH
