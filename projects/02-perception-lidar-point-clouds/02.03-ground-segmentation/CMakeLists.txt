# =============================================================================
# CMakeLists.txt — OPTIONAL cross-platform build for 02.03
#                  (Ground segmentation: RANSAC plane fit; Patchwork++-style GPU port)
#
# STATUS: bonus, not the deliverable. The Visual Studio solution in build/
# is this repo's REQUIRED build (CLAUDE.md paragraph 5); this file exists so
# Linux learners (and future CI compile checks) can build too. If this file
# and the .vcxproj ever disagree, the .vcxproj wins.
#
# Usage (from this project's root):
#     cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
#     cmake --build build-cmake -j
#     ./build-cmake/ground-segmentation
# (demo/run_demo.sh wraps exactly these steps.)
# =============================================================================

# 3.24 is the floor for the CMAKE_CUDA_ARCHITECTURES behavior we rely on and
# is old enough (2022) to be in every current distro's repositories.
cmake_minimum_required(VERSION 3.24)

# Declaring CUDA as a LANGUAGE (rather than find_package(CUDA), which is
# deprecated) makes CMake drive nvcc natively: .cu files just work, and
# CMake handles the device-link step if it is ever needed.
project(ground-segmentation LANGUAGES CXX CUDA)

# Repo standard: C++17 host code (CLAUDE.md paragraph 12). CUDA_STANDARD
# applies to the device-side compilation of .cu files — keep both in lockstep
# so host/device code can share headers without dialect surprises.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures — mirrors the .vcxproj CodeGeneration list: real code for
# sm_75 (Turing, the repo floor), sm_86 (Ampere), sm_89 (Ada). CMake appends
# PTX handling automatically for "75;86;89"; newer GPUs JIT from the newest
# PTX. Only set a default if the user/preset did not choose already, so
# `-DCMAKE_CUDA_ARCHITECTURES=native` still works for fast local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

# Default to an optimized build when the user does not say otherwise —
# single-config generators (Makefiles/Ninja) build UNOPTIMIZED with an empty
# CMAKE_BUILD_TYPE, which makes GPU-vs-CPU timing comparisons meaningless.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The single demo executable (CLAUDE.md paragraph 5: one runnable artifact).
# Same three sources as the VS project:
#   main.cu           — orchestration: args, data, transfers, verify, report
#   kernels.cu        — the GPU kernels + launch wrappers
#   reference_cpu.cpp — the plain-C++ correctness oracle (compiled as C++)
add_executable(ground-segmentation
  src/main.cu
  src/kernels.cu
  src/reference_cpu.cpp
)

# -lineinfo in optimized builds: source-line mapping for Nsight profiling at
# negligible cost (mirrors GenerateLineInfo in the .vcxproj). Device debug
# (-G) comes automatically in Debug builds via CMake's CUDA flags.
# NOTE: --use_fast_math is deliberately NOT set (see the .vcxproj comment —
# reproducible floats matter in robotics kernels).
target_compile_options(ground-segmentation PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>
)

# kernels.cu's Milestone-2 CZM pipeline uses Thrust (thrust::stable_sort_by_key
# / sequence / lower_bound) for the patch sort + vectorized boundary search —
# header-only (part of the CUDA Toolkit), so no target_link_libraries entry
# is needed for it. Thrust's CCCL headers DO need two extra flags when the
# HOST compiler nvcc calls is MSVC (cl.exe) — the same requirement the
# .vcxproj documents in full (ratified from project 02.01's identical fix):
# CCCL's preprocessor.h hard-requires MSVC's standard-conforming
# preprocessor (/Zc:preprocessor), and Thrust's dialect check needs
# __cplusplus to actually reflect /std:c++17 (/Zc:__cplusplus). This is an
# MSVC-specific requirement, not a CUDA one — a gcc/clang host compiler
# (the common case for this optional Linux CMake path) needs neither flag,
# so they are added ONLY when CMake has selected an MSVC host compiler
# (verified: this project's CMake configuration fails to compile kernels.cu
# with CCCL's own #error otherwise, on a Windows+MSVC CMake generator).
target_compile_options(ground-segmentation PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus>
)

# No target_link_libraries needed: with CUDA as an enabled language, CMake
# links the CUDA runtime automatically (CUDA_RUNTIME_LIBRARY defaults to
# Static — matching the .vcxproj's cudart_static.lib choice: the exe runs
# with only a driver installed).
