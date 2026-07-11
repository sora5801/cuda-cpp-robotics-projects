# Push note — 2026-07-10-21: the Phase-1 standards retrospective

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

CLAUDE.md §11 requires that after the 36 flagships, the standards be reassessed and the
template/docs updated with what the flagships taught. This push is that retrospective — no new
projects, but the three rulings the flagship campaign earned, folded back into
`docs/PROJECT_TEMPLATE/` and `docs/BUILD_GUIDE.md` so all 469 remaining Phase-2 projects inherit
them instead of rediscovering them: (1) the **LNK4099 Debug-linker race** root-caused in 01.02 is
now suppressed — with its full explanation — in the template *and* in every still-pristine
scaffolded `.vcxproj` (469 files, exact-match-gated mechanical patch); (2) a new
**`src/util/paths.h`** ratifies 12.01's multi-candidate data/artifact path resolution and carries
the no-`<filesystem>`-under-nvcc rule from 07.09; (3) the **twin-vs-shared verification ruling**
distilled from the 13.03 shared-bug case study is now written where every future builder will
read it: layout contracts must be single-sourced, algorithmic cores default to independent twins,
and any shared `__host__ __device__` helper obliges the project to carry a gate that bypasses it.

## What changed

- **[docs/PROJECT_TEMPLATE/build/TEMPLATE.vcxproj](../docs/PROJECT_TEMPLATE/build/TEMPLATE.vcxproj)** —
  `/ignore:4099` in the **Debug** `<Link>` only (the warning never fires in Release), with the
  full root-cause XML comment: the CUDA 13.3 MSBuild integration's synthetic ClCompile
  placeholder doesn't reliably receive the computed PDB filename, an intermittent, cosmetic,
  environment-wide race (reproduced on 01.02 and 08.01) that a project's own settings cannot
  close. Also lists the new `paths.h` in ClInclude (and
  [TEMPLATE.vcxproj.filters](../docs/PROJECT_TEMPLATE/build/TEMPLATE.vcxproj.filters)).
- **469 × `projects/*/*/build/*.vcxproj`** — the same Debug-only suppression block inserted into
  every still-pristine scaffolded project file. Mechanical patch, exact-match anchored on the
  pristine Debug `<Link>` text so it *cannot* touch a file a builder has modified; done projects
  and 36.03 were excluded outright. Every patched file has an identical +25/−0 diff.
- **[docs/PROJECT_TEMPLATE/src/util/paths.h](../docs/PROJECT_TEMPLATE/src/util/paths.h)** (new) —
  `find_data_file()` / `resolve_out_dir()`: CLI-override → exe-relative (VS layout) →
  CWD-relative candidate resolution, ratified from 12.01's fix for the CMake-layout mismatch
  that shipped as a real bug. Its header also canonizes the 07.09 toolchain rule: never include
  `<filesystem>` in a translation unit nvcc compiles (hard EDG error). Wired into
  [util/README.md](../docs/PROJECT_TEMPLATE/src/util/README.md) and pointed to from
  [main.cu](../docs/PROJECT_TEMPLATE/src/main.cu)'s data-loading TODO.
- **[docs/PROJECT_TEMPLATE/src/reference_cpu.cpp](../docs/PROJECT_TEMPLATE/src/reference_cpu.cpp)** —
  the independence ruling (see below), with the 13.03 case study spelled out.
- **[docs/PROJECT_TEMPLATE/THEORY.md](../docs/PROJECT_TEMPLATE/THEORY.md)** — "How we verify
  correctness" stub upgraded to the two-tier doctrine: the CPU twin catches *parallelization*
  bugs; independent analytic/invariant/negative-control gates catch the *algorithmic* bugs the
  twin is blind to. Both tiers required.
- **[docs/BUILD_GUIDE.md](../docs/BUILD_GUIDE.md)** — two new §10 troubleshooting rows: the
  LNK4099 story (and why it is now suppressed repo-wide) and the nvcc/`<filesystem>` hard error
  with its `paths.h` remedy.

## The rulings (what to study here)

1. **Two-tier verification is doctrine, not habit.** In flagship 13.03, an identical
   variable-shadowing bug lived in *both* the GPU path and its CPU twin: element-wise agreement
   was perfect, and only the closed-form ramp-angle gate exposed the shared defect. Twin
   agreement proves the parallelization is faithful to the reference — only an independent gate
   proves the reference is right. The flagships that shared `__host__ __device__` physics
   helpers (10.03, 18.01, 27.04) all carried such gates; the ruling makes that obligatory
   whenever sharing is chosen.
2. **Suppress precisely, explain fully.** `/ignore:4099` is one linker code, Debug-only, carried
   with a 25-line root-cause comment — the difference between hiding a warning and closing one.
3. **Path resolution is a portability contract.** One exe, three launch layouts (IDE, run_demo,
   CMake); `paths.h` makes the candidate list explicit instead of baking in one layout's
   directory depth.

## How to build & run

Nothing new to run. To see the patch is build-neutral: any still-`todo` skeleton builds and its
placeholder demo passes, e.g.

```powershell
projects\01-perception-cameras-vision\01.01-full-gpu-image-pipeline\demo\run_demo.ps1
```

## Verification

On the owner's machine (RTX 2080 SUPER, CUDA 13.3, VS 2026 v145, 2026-07-10):

- Patch audit: 469 patched / 36 skipped (35 done + 36.03, then mid-build); `git diff --numstat`
  confirms every patched file is exactly **+25/−0**; idempotence guard (`/ignore:4099` already
  present → skip) verified by a second dry run patching 0.
- Build-neutrality: skeleton 01.01 rebuilt **Release|x64 and Debug|x64, zero errors, zero
  warnings**, placeholder demo `RESULT: PASS` after the patch.
- `paths.h` compiles in both cl.exe and nvcc translation units by construction (no `<filesystem>`,
  C++17-only); it is header-only and included by nothing until a real implementation adopts it.
- No project source, data, demo, or doc content outside the files listed above was touched.

## Known limitations / TODOs

- The 35 already-done flagships keep their as-verified `.vcxproj` files (only 01.02 carries the
  suppression, where it was root-caused); retrofitting the rest was deliberately skipped — their
  gates were green as shipped, and re-verifying 35 projects to silence a cosmetic Debug warning
  is not worth the churn. If any is revisited in a later phase, add the block then.
- Skeletons patched here still carry the *old* `util/` copies (no `paths.h`); Phase-2 builders
  copy `util/` fresh from the template per §10, so the header arrives with each real build-out.

## Next push preview

**Phase 2 opens.** The remaining 469 projects, domain by domain, easiest-first (★ → untagged →
[R&D]) per §11, with the same builder → independent lead gate sweep → push-note loop. First up:
domain 01 (perception/cameras), starting with its ★ entries.
