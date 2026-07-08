# BUILD_GUIDE.md — Installing the Toolchain & Building Any Project

> **What this is.** The canonical, copy-paste-friendly guide to getting from a bare Windows machine to
> a running demo for *any* project in `cuda-cpp-robotics-projects`. It is referenced by every project
> README (§6 "Build") and is the single place where toolchain knowledge lives — if a build step ever
> changes, this file changes in the same push (CLAUDE.md §5).
>
> **Who this is for.** A learner who knows C++ but may never have installed CUDA or built a Visual
> Studio solution from the command line. Every command below is meant to be pasted verbatim into
> **PowerShell** unless stated otherwise.
>
> **The reference machine** (the repo owner's, on which everything here was verified):
> Windows 11 · Visual Studio 2026 Community (v145 toolset) · CUDA Toolkit 13.3 ·
> NVIDIA GeForce RTX 2080 SUPER (8 GB, compute capability `sm_75`) · driver 591.86 · Python 3.12.

---

## Table of contents

1. [Prerequisites & install order](#1-prerequisites--install-order)
2. [Verifying the installation](#2-verifying-the-installation)
3. [Driver "CUDA Version" vs. Toolkit version — don't be spooked](#3-driver-cuda-version-vs-toolkit-version--dont-be-spooked)
4. [Building from the IDE (the normal path)](#4-building-from-the-ide-the-normal-path)
5. [Building from the command line (MSBuild)](#5-building-from-the-command-line-msbuild)
6. [GPU architectures: fatbins, PTX, and narrowing the codegen list](#6-gpu-architectures-fatbins-ptx-and-narrowing-the-codegen-list)
7. [Debug vs. Release — what each configuration actually does](#7-debug-vs-release--what-each-configuration-actually-does)
8. [Running the demos](#8-running-the-demos)
9. [Python for `tools/` and `scripts/`](#9-python-for-tools-and-scripts)
10. [Troubleshooting](#10-troubleshooting)
11. [A note on CI (no GPU in the cloud)](#11-a-note-on-ci-no-gpu-in-the-cloud)

---

## 1. Prerequisites & install order

**Order matters.** The CUDA installer detects installed Visual Studio versions and drops MSBuild
integration files into them. Install Visual Studio *first*, CUDA *second*. (If you did it backwards,
see [Troubleshooting](#10-troubleshooting) — the fix is to re-run the CUDA installer.)

### Step 1 — Visual Studio 2026 Community

1. Download **Visual Studio 2026 Community** (free) from Microsoft.
2. In the installer, select the workload **"Desktop development with C++"**. That workload brings the
   MSVC compiler (`cl.exe`), the Windows SDK, and MSBuild — everything the host-side (CPU) half of a
   CUDA program needs. `nvcc`, the CUDA compiler driver, does not compile host code itself; it hands
   the host portions of every `.cu` file to MSVC. No MSVC → no CUDA builds.
3. Finish the install and launch VS once so it completes first-run setup.

On the reference machine, VS lives at:

```
C:\Program Files\Microsoft Visual Studio\18\Community
```

("18" is the internal major version of Visual Studio 2026; you will see it again in paths below.
The C++ platform toolset it ships is **v145** — every `.vcxproj` in this repo targets it.)

### Step 2 — CUDA Toolkit 13.3

1. Download **CUDA Toolkit 13.3** from NVIDIA's developer site (choose Windows → x86_64 → your
   Windows version → exe installer).
2. Run the installer. The **Express** option is fine; it installs:
   - the toolkit itself (`nvcc`, headers, cuBLAS/cuFFT/cuRAND/… libraries) under
     `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` and sets the `CUDA_PATH`
     environment variable to it;
   - the **Visual Studio integration** — two MSBuild files, `CUDA 13.3.props` and
     `CUDA 13.3.targets`, copied into the VS build-customizations folder. Every `.vcxproj` in this
     repo imports these two files by name; they are what teach MSBuild to invoke `nvcc` on `.cu`
     files. This is the single most common thing to be missing on a broken machine — verify it below.
3. A sufficiently new **NVIDIA driver** is required at *run* time (the toolkit installer can update
   it, or use the standard GeForce/Studio driver). See [§3](#3-driver-cuda-version-vs-toolkit-version--dont-be-spooked)
   for why the driver's reported "CUDA Version" need not equal 13.3.

> **GPU requirement.** CUDA 13 dropped support for Maxwell, Pascal, and Volta GPUs. The oldest
> architecture it can target is **Turing (`sm_75`)** — e.g., GTX 16xx / RTX 20xx. The reference
> machine's RTX 2080 SUPER is exactly at this floor, which is why `sm_75` is the first entry in the repo's
> codegen list (see [§6](#6-gpu-architectures-fatbins-ptx-and-narrowing-the-codegen-list)). If your
> GPU is older than Turing, these projects will not run on it under CUDA 13.

### Step 3 — Python 3.12 (for tools and data scripts)

Install **Python 3.12** (from python.org or the Microsoft Store) and make sure `python` is on PATH.
It drives the repo tooling (`tools/*.py`) and every project's synthetic-data generator
(`scripts/make_synthetic.py`). See [§9](#9-python-for-tools-and-scripts).

---

## 2. Verifying the installation

Run these three checks in PowerShell. Each one verifies a different link in the chain — compiler,
build integration, and GPU/driver — and each can pass or fail independently, which is exactly why we
check all three.

### Check 1 — `nvcc` (the CUDA compiler is installed and on PATH)

```powershell
nvcc --version
```

Expected output (the reference machine's — your build date/time may differ, but the release line
should say **13.3** and the build should be **V13.3.33**):

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
...
Cuda compilation tools, release 13.3, V13.3.33
```

If `nvcc` is not found, the toolkit is not installed or its `bin` folder is not on PATH (the
installer normally adds it; a fresh terminal after installing usually fixes stale PATHs).

### Check 2 — the Visual Studio integration (MSBuild knows about CUDA)

This is the check people skip and then lose an afternoon to. Confirm the two integration files exist
in the VS 2026 build-customizations folder (this exact path is verified on the reference machine —
note the `v180` folder, which corresponds to VS 2026's internal version 18):

```powershell
Get-ChildItem "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v180\BuildCustomizations" -Filter "CUDA 13.3.*"
```

Expected: at least `CUDA 13.3.props` and `CUDA 13.3.targets` listed. If the folder has no
`CUDA 13.3.*` files, the CUDA installer did not find (or predates) your Visual Studio — see
[Troubleshooting](#10-troubleshooting), first row.

### Check 3 — the GPU and driver (`nvidia-smi`)

```powershell
nvidia-smi
```

This prints a table with your driver version, a "CUDA Version" (see §3 — it is *not* the toolkit
version), and your GPU's name. On the reference machine it shows an **NVIDIA GeForce RTX 2080
SUPER** (8 GB) with **driver 591.86** reporting **CUDA Version: 13.1**.

To map your GPU name to a compute capability (needed in [§6](#6-gpu-architectures-fatbins-ptx-and-narrowing-the-codegen-list)),
you can ask the driver directly:

```powershell
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Expected shape of the output (reference machine):

```
name, compute_cap
NVIDIA GeForce RTX 2080 SUPER, 7.5
```

`7.5` means `sm_75` (Turing). RTX 30xx is `8.6` (`sm_86`), RTX 40xx is `8.9` (`sm_89`).

---

## 3. Driver "CUDA Version" vs. Toolkit version — don't be spooked

On the reference machine, `nvidia-smi` reports **"CUDA Version: 13.1"** while the installed toolkit
is **13.3**. This looks like a mismatch. It is not a problem, and understanding why teaches you
something real about how CUDA is layered:

- The **toolkit** version (13.3) is the *compiler and libraries* you build with — what
  `nvcc --version` reports.
- The number `nvidia-smi` prints is the maximum CUDA version the installed **driver** natively
  advertises — a property of the driver (591.86 here), not of any toolkit.

These are two different pieces of software, and NVIDIA guarantees **minor-version compatibility
within a major CUDA family**: an application built with any CUDA 13.x toolkit runs on a driver that
supports any CUDA 13.x — the driver's user-mode stack and the 13.x runtime keep a stable contract
across minor versions. So binaries built with the 13.3 toolkit run fine on this 13.1-reporting
driver. (What would *not* be guaranteed is crossing a major version — e.g., a hypothetical CUDA 14
app on a 13.x driver.)

**Rule of thumb:** the two numbers only need to agree on the digit before the dot. If they do,
build and run without worry. If your driver is a whole major version behind your toolkit, update
the driver.

---

## 4. Building from the IDE (the normal path)

Every project ships a Visual Studio solution — this is the required, always-works build path
(CLAUDE.md §5). The optional `CMakeLists.txt`, where present, is a bonus for Linux learners; it is
never a substitute.

1. Open the project's solution in Visual Studio 2026:

   ```
   projects\<domain>\<SS.NN>-<slug>\build\<slug>.sln
   ```

   (Double-click it, or File → Open → Project/Solution.)

2. In the toolbar, set the configuration to **`Release`** and the platform to **`x64`**.
   Use **`Debug|x64`** instead when you want to step through kernels with Nsight — see
   [§7](#7-debug-vs-release--what-each-configuration-actually-does) for what actually changes.
   These are the only two configurations projects ship; there is no Win32/x86.

3. **Build → Build Solution** (Ctrl+Shift+B).

**Where the output lands.** Our `.vcxproj` files set `OutDir` to `$(ProjectDir)x64\<Config>\`, so
the executable appears next to the solution, under the project's own `build/` folder:

```
projects\<domain>\<SS.NN>-<slug>\build\x64\Release\<slug>.exe   (Release|x64)
projects\<domain>\<SS.NN>-<slug>\build\x64\Debug\<slug>.exe     (Debug|x64)
```

Intermediate files (`.obj`, logs) also land under that `x64\` tree, and every project's
`.gitignore` excludes it — build artifacts are never committed.

No manual configuration should ever be needed: the `.vcxproj` locates CUDA via the
`CUDA 13.3.props/.targets` integration and the `$(CUDA_PATH)` environment variable, never via
hardcoded paths. If Visual Studio complains on load, jump to [Troubleshooting](#10-troubleshooting).

---

## 5. Building from the command line (MSBuild)

Useful for scripting, for the `demo/run_demo.ps1` scripts (which do exactly this), and for building
without opening the IDE.

### Step 1 — find MSBuild with `vswhere`

`vswhere.exe` is a small locator utility that every Visual Studio install places at a **fixed,
version-independent path** — that is its whole job: you can always find it, and it can always find
Visual Studio. Ask it for the path to the newest MSBuild:

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
```

On the reference machine this prints:

```
C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe
```

### Step 2 — invoke MSBuild on the solution

The general shape (three flags to remember: configuration, platform, and `-m` for parallel builds):

```powershell
MSBuild.exe <path-to>\<slug>.sln /p:Configuration=Release /p:Platform=x64 -m
```

A complete, paste-able PowerShell example using the confirmed MSBuild path (capture the vswhere
result in a variable so the rest is copy-paste on any machine):

```powershell
# Locate MSBuild once (falls back to asking vswhere so this works on any VS install).
$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
# On the reference machine, $msbuild is:
#   C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe

# Build a project's solution: Release, 64-bit, multi-process (-m uses all cores).
& $msbuild "projects\08-control-systems\08.01-mppi-cart-pole\build\08.01-mppi-cart-pole.sln" `
    /p:Configuration=Release /p:Platform=x64 -m
```

(Substitute the real project path; the pattern is identical for every project. Add
`/p:Configuration=Debug` instead for a debug build. The resulting `.exe` lands in
`build\x64\Release\` exactly as in [§4](#4-building-from-the-ide-the-normal-path).)

**What each flag does, since nothing here is a black box:**

- `/p:Configuration=Release` — selects the Release configuration defined in the `.vcxproj`
  (optimized host code, no device debug; see §7).
- `/p:Platform=x64` — selects the 64-bit platform. CUDA on Windows is 64-bit only; there is no
  x86 configuration in these projects.
- `-m` — lets MSBuild schedule independent projects/translation units across all CPU cores.
  Harmless for a single small project, a big win for anything larger.

---

## 6. GPU architectures: fatbins, PTX, and narrowing the codegen list

### The repo standard

Every `.vcxproj` in this repo uses this CUDA **code generation** list:

```
compute_75,sm_75;compute_86,sm_86;compute_89,sm_89;compute_89,compute_89
```

Reading it: each `compute_XX,sm_XX` pair compiles the kernels **ahead of time** into native machine
code (SASS) for one GPU architecture — `sm_75` = Turing (RTX 20xx / GTX 16xx), `sm_86` = Ampere
(RTX 30xx), `sm_89` = Ada (RTX 40xx). The odd-looking **last entry `compute_89,compute_89`** embeds
**PTX** — no native code — and is deliberately last.

### What fatbins and PTX JIT are (the 3-sentence version)

`nvcc` bundles all of those compiled variants into one executable, called a **fat binary
("fatbin")**: at launch, the CUDA runtime picks the native code matching whatever GPU it finds, so
one `.exe` runs well on Turing, Ampere, and Ada without recompiling. **PTX** is CUDA's portable
virtual-ISA assembly — forward-compatible source the driver can compile *at run time* ("JIT",
just-in-time) into native code for architectures that didn't exist when we built. Embedding
`compute_89` PTX as the last fatbin entry is our forward-compatibility insurance: a future GPU
newer than Ada finds no matching native code, so the driver JIT-compiles the PTX and the program
still runs (with a small one-time compile pause on first launch).

### Why `sm_75` is the floor

CUDA 13 removed support for Maxwell, Pascal, and Volta. **Turing (`sm_75`) is the oldest
architecture CUDA 13.3 can compile for**, and — conveniently for testing honesty — it is exactly
what the reference machine's RTX 2080 SUPER is. There is nothing below `sm_75` in any codegen list in
this repo, and there cannot be.

### Narrowing the list for faster local builds

Compiling three native architectures plus PTX roughly multiplies device-code compile time by four.
While *studying and iterating* on one machine, you can compile for only your own GPU:

1. Find your compute capability:

   ```powershell
   nvidia-smi --query-gpu=name,compute_cap --format=csv
   ```

   (Reference machine: `7.5`, i.e. `sm_75`.)

2. In Visual Studio: project **Properties → CUDA C/C++ → Device → Code Generation** → replace the
   list with just your pair, e.g. `compute_75,sm_75`.

3. Rebuild. Iteration gets noticeably faster.

**Restore the full standard list before committing** — the committed `.vcxproj` must build fatbins
that run on other people's machines, not just yours. (The narrowed list is a local convenience,
never a repo change.)

---

## 7. Debug vs. Release — what each configuration actually does

Every project ships both configurations, and they differ in ways worth understanding:

| | `Debug\|x64` | `Release\|x64` |
|---|---|---|
| Host (CPU) code | No optimization, full debug info | MSVC `/O2` optimization |
| Device (GPU) code | `-G` — device-side debug: optimizations off, full kernel debug info | Optimized SASS |
| Line info | `-lineinfo` (with `-G`) so **Nsight** tools map GPU behavior back to source lines | (projects may keep `-lineinfo` for profiling; see each `.vcxproj`) |
| Use it for | Stepping through kernels in Nsight, chasing wrong answers | Timing, demos, the numbers in READMEs |

Two things to internalize:

- **Never time Debug builds.** `-G` disables device optimizations; Debug kernels can be an order of
  magnitude slower. Every timing figure printed by a demo, and every ms number in a README or
  push-note, comes from `Release|x64`.
- **`--use_fast_math` is OFF by default, everywhere.** That flag lets the compiler substitute
  faster, less-accurate math (fused/approximate divisions, sqrt, transcendentals) and breaks
  bit-reproducibility of floating-point results. Robotics code cares about reproducible floats —
  our demos assert GPU-vs-CPU agreement within a documented tolerance, and state estimators and
  integrators are sensitive to accumulation error. A project enables fast math **only** if it
  explicitly documents that its computation tolerates it (README §13 / THEORY.md numerics section),
  with the flag visibly set and commented in its `.vcxproj`.

---

## 8. Running the demos

Every project has a one-command demo (CLAUDE.md §6.3):

```powershell
# From anywhere — the script is location-independent:
& "projects\<domain>\<SS.NN>-<slug>\demo\run_demo.ps1"
```

What `run_demo.ps1` does, in order:

1. **Builds if needed** — if `build\x64\Release\<slug>.exe` is missing or stale, it invokes MSBuild
   exactly as in [§5](#5-building-from-the-command-line-msbuild) (locating it via `vswhere`).
2. **Runs on the committed sample** — the tiny dataset in `data\sample\` (synthetic by default,
   labeled as such), so the demo needs **no downloads and no extra SDKs**.
3. **Prints the result, the GPU-vs-CPU agreement check, and a timing line** — the executable runs
   both the CUDA kernels and the plain-C++ reference (`src\reference_cpu.cpp`) and asserts they
   agree within the project's documented tolerance. The timing line (CUDA-event-measured ms, and a
   speed-up where meaningful) is a teaching artifact, never a benchmark claim.
4. **Compares against `demo\expected_output.txt`** — what you see should match the committed
   expectation (within the documented tolerance for numeric lines). If it doesn't, something is
   wrong; the file is the project's ground truth for "the demo works".

Some demos also write a visual artifact (PNG/CSV/OBJ) into `demo\`; `demo\README.md` explains what
you are looking at.

**Linux learners:** projects that ship the optional `CMakeLists.txt` also ship a `demo\run_demo.sh`
with identical semantics (build via CMake if needed, run on the sample, check agreement, compare
expected output). The VS solution remains the canonical build; the CMake path is best-effort.

---

## 9. Python for `tools/` and `scripts/`

The repo's tooling (`tools\catalog.py`, `tools\scaffold.py`, `tools\verify_project.py`,
`tools\status.py`, `tools\new_pushnote.py`) and every project's data generator
(`scripts\make_synthetic.py`) run on **Python 3.12** (3.12.11 on the reference machine):

```powershell
python --version     # expect: Python 3.12.x
```

**No extra packages are required by the standard template** — the tools and the template's
synthetic-data scripts use only the Python standard library, precisely so that `pip install`
is never a prerequisite for building or demoing. If an individual project's script ever needs
more, that project's README must say so explicitly.

Typical invocations:

```powershell
python tools\verify_project.py projects\<domain>\<SS.NN>-<slug>    # Definition-of-Done checks
python projects\<domain>\<SS.NN>-<slug>\scripts\make_synthetic.py  # regenerate the sample data
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| VS says a project can't load, or MSBuild errors with *"...CUDA 13.3.props was not found"* / imported project not found | The CUDA↔VS integration is missing — the classic result of installing CUDA **before** Visual Studio (the installer had no VS to integrate into), or of a custom CUDA install that skipped the "Visual Studio Integration" component | Re-run the CUDA 13.3 installer *after* VS is installed and ensure the Visual Studio Integration component is selected. Then re-run Check 2 in [§2](#2-verifying-the-installation): `CUDA 13.3.props/.targets` must exist under `...\18\Community\MSBuild\Microsoft\VC\v180\BuildCustomizations\` |
| *"The build tools for v145 cannot be found"* or a platform-toolset error | The "Desktop development with C++" workload (which provides the v145 toolset) isn't installed, or you opened the solution in an older Visual Studio | Open the **Visual Studio Installer**, modify VS 2026, and add *Desktop development with C++*. Build with VS 2026 — the repo targets its v145 toolset |
| Build succeeds but the program prints a CUDA error like *"no kernel image is available for execution on the device"* at launch | Your GPU's architecture isn't covered by the compiled fatbin — most commonly a pre-Turing GPU (older than `sm_75`), which CUDA 13 no longer supports at all; or you narrowed the codegen list ([§6](#6-gpu-architectures-fatbins-ptx-and-narrowing-the-codegen-list)) to an arch that isn't yours | Check `nvidia-smi --query-gpu=name,compute_cap --format=csv`. If your compute capability is below 7.5, these projects cannot run on that GPU under CUDA 13. If it's ≥ 7.5, restore the full standard codegen list and rebuild |
| A long-running kernel makes the screen freeze, then the program dies with an error such as *"the launch timed out and was terminated"* | **Windows TDR** (Timeout Detection & Recovery): when the GPU also drives your display, Windows resets it if any single GPU operation blocks it for more than the default **~2 seconds**, killing your kernel mid-flight. Debug (`-G`) builds are the usual victims — unoptimized kernels can easily exceed 2 s | Prefer Release builds for long workloads; shrink the problem size (all demos are sized to finish well under the limit); or split work into shorter kernel launches. TDR *can* be reconfigured via registry/Nsight settings, but changing display-driver watchdog behavior has system-wide consequences — understand what you are doing first; it is never required for this repo's demos |
| `cudaMalloc` fails / *"out of memory"* on an 8 GB card | The reference GPU has 8 GB, and part of it is held by Windows and the desktop compositor; big grids/point clouds/batch sizes can exceed what's left | Demos are sized for 8 GB. If you enlarged parameters (a great exercise!), scale them back, process in chunks, or free intermediate buffers earlier. `nvidia-smi` shows current memory pressure |
| First build of a project is inexplicably slow (minutes), later builds fast | Real-time antivirus scanning every newly written `.obj`/`.exe` and every compiler process spawn — `nvcc` spawns many | Expect the first build to be slower; subsequent incremental builds are fast. If it's severe, an exclusion for your local working folder helps — that is a security trade-off on your machine, so it's your call, not this repo's advice |
| `nvcc` not found in a terminal that worked before / after a fresh install | PATH changes only apply to newly started shells | Open a fresh PowerShell. If still missing, confirm `CUDA_PATH` points at `...\CUDA\v13.3` and its `bin` is on PATH |

If a build or tooling assumption in this guide fails on your machine in a way this table doesn't
cover, that's a documentation bug: per CLAUDE.md §13, stop, figure it out (or ask), and capture the
fix *here* in the same push.

---

## 11. A note on CI (no GPU in the cloud)

If/when this repo runs GitHub Actions, understand what a green badge can and cannot mean:
**GitHub's hosted runners have no NVIDIA GPU.** CI can *compile* every CUDA project (a real and
useful check — it catches broken `.vcxproj` files, missing includes, and syntax errors on a clean
machine) but it can **never execute a kernel**. Running demos and verifying GPU-vs-CPU agreement is
strictly a **local** step on a real GPU, and every push-note states which GPU the verification ran
on. A green build badge therefore means "it compiles", never "the kernels were run" (CLAUDE.md §9).

---

*Everything in this guide was verified on the reference machine described at the top. Paths and
versions are real, not illustrative. If your machine differs (different VS edition, different
drive), the `vswhere` and `$(CUDA_PATH)`-based mechanisms above adapt automatically — that is why
the projects never hardcode paths.*
