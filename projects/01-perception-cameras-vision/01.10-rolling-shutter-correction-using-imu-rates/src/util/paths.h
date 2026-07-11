// ===========================================================================
// paths.h — data-file and artifact-directory resolution for project
//           01.10 (Rolling-shutter correction using IMU rates; copied from
//           docs/PROJECT_TEMPLATE — see util/README.md for the deliberate-
//           duplication rule)
//
// Role in the project
// -------------------
// Host-only helpers that answer two questions every real project in this
// repository eventually asks:
//
//   1) "Where is data/sample/<file>?"   -> find_data_file()
//   2) "Where do I write demo/out/…?"   -> resolve_out_dir()
//
// The scaffolded SAXPY placeholder does neither (it synthesizes its input
// in memory and writes no artifacts), so nothing includes this header yet —
// include it from main.cu the moment the real implementation loads its
// committed sample or writes a PNG/CSV/PGM artifact (CLAUDE.md §6.3).
//
// Why multi-candidate resolution instead of one relative path (the lesson)
// ------------------------------------------------------------------------
// The same executable is launched from several places with several layouts:
//
//   * The REQUIRED Visual Studio build puts the exe at
//         build/x64/<Config>/<slug>.exe        (3 dirs below the project root)
//     so "exe_dir/../../.." finds the root — IF you launch the exe directly.
//   * The OPTIONAL CMake build (CLAUDE.md §5: bonus, never a substitute)
//     defaults to build-cmake/<slug> or build-cmake/<Config>/<slug> —
//     only 1–2 dirs deep, so the exe-relative formula lands one level too
//     high. This exact mismatch shipped as a real bug in an early flagship
//     and was root-caused in project 12.01, whose fix this header ratifies
//     template-wide (Phase-1 standards retrospective).
//   * Both demo/run_demo scripts invoke the exe with the PROJECT ROOT as the
//     working directory, so a plain CWD-relative candidate is the reliable
//     fallback for every build layout at once.
//
// Trying an ordered list of honest candidates — CLI override first, then
// exe-relative (VS layout), then CWD-relative — makes the demo work from
// the IDE debugger, from run_demo.ps1/.sh, and from a CMake build, without
// a single hardcoded absolute path (CLAUDE.md §5).
//
// Why not std::filesystem (a load-bearing toolchain fact)
// -------------------------------------------------------
// This header is included from main.cu, which nvcc compiles. nvcc 13.3's
// EDG front end chokes on MSVC's <filesystem> headers (root-caused in
// project 07.09 — a hard compile error, not a warning). The rule this repo
// ratified: NEVER include <filesystem> in a translation unit nvcc sees.
// Hence the classic <fstream>-probe + _mkdir/mkdir approach below: plain
// C++17-minus-filesystem that both cl.exe and nvcc compile everywhere.
//
// Read this after: util/cuda_check.cuh (same "small visible helper" spirit).
// ===========================================================================
#ifndef UTIL_PATHS_H
#define UTIL_PATHS_H

#include <string>    // std::string path arithmetic — no filesystem, see header
#include <vector>    // the ordered candidate lists
#include <fstream>   // std::ifstream as a portable "does this file exist?" probe
#include <cerrno>    // errno / EEXIST for the mkdir "already there" case

#ifdef _WIN32
#include <direct.h>      // _mkdir — Windows' single-level directory creation
#else
#include <sys/stat.h>    // mkdir(path, mode) — POSIX equivalent
#include <sys/types.h>
#endif

// ---------------------------------------------------------------------------
// project_root_from — derive the project root from argv[0], ASSUMING the
// required Visual Studio output layout build/x64/<Config>/<slug>.exe.
//
// Parameters:
//   argv0 — argv[0] as received by main(); may be null or bare ("slug.exe"
//           found via PATH), in which case there is no directory to strip
//           and we return "." (the CWD-relative candidates then do the work).
// Returns: "<exe_dir>/../../.." — the project root under the VS layout; a
//          harmless wrong guess under any other layout (callers always treat
//          this as ONE candidate among several, never as truth).
// Complexity: O(len). Side effects: none.
// ---------------------------------------------------------------------------
static inline std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    // Windows gives backslashes, the shells sometimes forward slashes —
    // accept either; we only need the LAST separator to strip the filename.
    const size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";   // bare name: no path info at all
    return exe.substr(0, cut) + "/../../..";    // build/x64/<Config> -> project root
}

// ---------------------------------------------------------------------------
// find_data_file — locate a committed sample file by trying honest candidates
// in priority order (see the file header for why a single path can't work).
//
// Parameters:
//   cli_dir  — optional user override directory from the project's CLI
//              ("" = none). Highest priority: an explicit ask always wins.
//   argv0    — argv[0], for the exe-relative (VS-layout) candidate.
//   filename — the file's name inside data/sample/, e.g. "scenario.csv".
// Returns: the first candidate that OPENS (an existence probe via ifstream —
//          openable is what callers actually need), or "" when none does;
//          callers must check for "" and fail with a readable message that
//          names the file (never a silent default — CLAUDE.md §13 honesty).
// Complexity: O(#candidates) opens. Side effects: none (probes are reads).
// ---------------------------------------------------------------------------
static inline std::string find_data_file(const std::string& cli_dir,
                                         const char* argv0,
                                         const char* filename)
{
    std::vector<std::string> candidates;
    if (!cli_dir.empty()) candidates.push_back(cli_dir + "/" + filename);
    // VS layout: exe three dirs below root (the REQUIRED build, CLAUDE.md §5).
    candidates.push_back(project_root_from(argv0) + "/data/sample/" + filename);
    // CWD-relative: both run_demo scripts launch with the project root as CWD,
    // and this is also what rescues the CMake layouts (see file header).
    candidates.push_back(std::string("data/sample/") + filename);
    candidates.push_back(std::string("../data/sample/") + filename);  // defensive: launched from build/
    for (const auto& c : candidates)
        if (std::ifstream(c, std::ios::binary).is_open()) return c;
    return "";   // not found anywhere honest — caller reports and fails loudly
}

// ---------------------------------------------------------------------------
// ensure_dir — create ONE directory level, succeeding if it already exists.
//
// Parameters: path — the directory to create (parent must already exist —
//             single-level _mkdir/mkdir semantics, deliberately: every
//             caller below targets "<something>/demo/out" where demo/ is a
//             COMMITTED folder in every project, so one level is enough).
// Returns: true when the directory exists on return (created now or before).
// Side effects: may create the directory. Complexity: one syscall.
// ---------------------------------------------------------------------------
static inline bool ensure_dir(const std::string& path)
{
#ifdef _WIN32
    const int r = _mkdir(path.c_str());
#else
    const int r = mkdir(path.c_str(), 0755);
#endif
    return r == 0 || errno == EEXIST;
}

// ---------------------------------------------------------------------------
// resolve_out_dir — pick the artifact directory demo/out/ with the SAME
// multi-candidate strategy find_data_file applies to inputs (ratified from
// project 12.01's fix — see the file header for the CMake-layout story).
//
// Parameters: argv0 — argv[0], for the exe-relative (VS-layout) candidate.
// Returns: the first candidate whose directory now exists. When every
//          candidate fails (exotic launch dir + read-only tree), returns the
//          first candidate anyway so the caller's artifact WRITE fails with
//          a readable per-file error — the failure surfaces at the write,
//          with a path in hand to print, instead of vanishing here.
// Side effects: creates demo/out under the winning candidate.
// ---------------------------------------------------------------------------
static inline std::string resolve_out_dir(const char* argv0)
{
    const std::vector<std::string> candidates = {
        project_root_from(argv0) + "/demo/out",   // VS OutDir layout (required build)
        "demo/out",                                // CWD-relative (both run_demo scripts)
        "../demo/out",                             // one level up, defensive fallback
    };
    for (const auto& c : candidates)
        if (ensure_dir(c)) return c;
    return candidates.front();   // exhausted; caller reports the write failure honestly
}

#endif // UTIL_PATHS_H
