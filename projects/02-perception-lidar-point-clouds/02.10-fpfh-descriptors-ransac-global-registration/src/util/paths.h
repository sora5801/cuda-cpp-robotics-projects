// ===========================================================================
// paths.h — data-file and artifact-directory resolution for project 02.10
//           (FPFH descriptors + RANSAC global registration; copied from
//           docs/PROJECT_TEMPLATE — see util/README.md for the deliberate-
//           duplication rule)
//
// Role in the project
// -------------------
// Host-only helpers that answer two questions this project's main.cu asks
// at startup: "where is data/sample/pairs_meta.csv (and the six .bin
// clouds it names)?" (find_data_file) and "where do I write demo/out/…?"
// (resolve_out_dir).
//
// Why multi-candidate resolution instead of one relative path, and why NOT
// std::filesystem — see docs/PROJECT_TEMPLATE/src/util/paths.h's original
// header (12.01's root-caused fix, ratified template-wide): the short
// version is that this project's exe is launched from several different
// working directories (VS debugger, run_demo.ps1/.sh, an optional CMake
// build) and nvcc 13.3's EDG front end cannot compile <filesystem>.
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
// ---------------------------------------------------------------------------
static inline std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    const size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

// ---------------------------------------------------------------------------
// find_data_file — locate a committed sample file by trying honest candidates
// in priority order (see the file header for why a single path can't work).
// ---------------------------------------------------------------------------
static inline std::string find_data_file(const std::string& cli_dir,
                                         const char* argv0,
                                         const char* filename)
{
    std::vector<std::string> candidates;
    if (!cli_dir.empty()) candidates.push_back(cli_dir + "/" + filename);
    candidates.push_back(project_root_from(argv0) + "/data/sample/" + filename);
    candidates.push_back(std::string("data/sample/") + filename);
    candidates.push_back(std::string("../data/sample/") + filename);
    for (const auto& c : candidates)
        if (std::ifstream(c, std::ios::binary).is_open()) return c;
    return "";
}

// ---------------------------------------------------------------------------
// ensure_dir — create ONE directory level, succeeding if it already exists.
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
// multi-candidate strategy find_data_file applies to inputs.
// ---------------------------------------------------------------------------
static inline std::string resolve_out_dir(const char* argv0)
{
    const std::vector<std::string> candidates = {
        project_root_from(argv0) + "/demo/out",
        "demo/out",
        "../demo/out",
    };
    for (const auto& c : candidates)
        if (ensure_dir(c)) return c;
    return candidates.front();
}

#endif // UTIL_PATHS_H
