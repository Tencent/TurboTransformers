#include "blas.h"
#include "absl/strings/str_cat.h"
#include <cstdlib>
#include <iostream>
#include <memory>

#if defined(__cplusplus) && __cplusplus >= 201703L && defined(__has_include)
#if __has_include(<filesystem>)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
#ifndef GHC_USE_STD_FS
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#endif

namespace fast_transformers {
namespace core {
#ifdef __APPLE__
static const char *dynlib_suffix_ = ".dylib";
#else
static const char *dynlib_suffix_ = ".so";
#endif

std::unique_ptr<CBlasFuncs, CBlasFuncDeleter> g_blas_funcs_;

void AutoInitBlas() {
  std::string openblas_libname = absl::StrCat("libopenblas", dynlib_suffix_);
  std::vector<std::string> mklml_libnames = {absl::StrCat("libmklml_intel", dynlib_suffix_), 
                        absl::StrCat("libmklml", dynlib_suffix_), absl::StrCat("libmklml_gnu", dynlib_suffix_)};
  char *conda_prefix = std::getenv("CONDA_PREFIX");
  if (conda_prefix != nullptr) {
    for(const std::string& mklml_libname: mklml_libnames) {
      auto p = fs::path(conda_prefix) / "lib" / mklml_libname;
      if (fs::exists(p)) {
        InitializeMKLMLLib(p.c_str());
        return;
      }
    }//for mklmkl
  }

  std::vector<fs::path> pathes = {fs::path("./"), fs::path("/usr/lib"),
                                  fs::path("/usr/local/lib"),
                                  fs::path("/usr/local/opt/openblas/lib")};

  for (auto &p : pathes) {
    for(const std::string& mklml_libname: mklml_libnames) {
      auto libpath = p / mklml_libname;
      if (fs::exists(libpath)) {
        InitializeOpenblasLib(libpath.c_str());
        return;
      }
    }//for mkl
  }//for path

  for (auto &p : pathes) {
    auto libpath = p / openblas_libname;
    if (fs::exists(libpath)) {
      InitializeOpenblasLib(libpath.c_str());
      return;
    }
  }

  throw std::runtime_error("Cannot initialize blas automatically");
}

static void InitializeBlasCommon(const char *filename) {
  void *lib = dlopen(filename, RTLD_LAZY | RTLD_LOCAL);
  FT_ENFORCE_NE(lib, nullptr, "Cannot load blas library %s", filename);

  g_blas_funcs_.reset(new CBlasFuncs());
  g_blas_funcs_->shared_library_ = lib;
  g_blas_funcs_->sgemm_ =
      reinterpret_cast<decltype(cblas_sgemm) *>(dlsym(lib, "cblas_sgemm"));
  FT_ENFORCE_NE(g_blas_funcs_->sgemm_, nullptr, "Cannot load cblas_sgemm");
}

void InitializeOpenblasLib(const char *filename) {
  std::cerr << "using openblas...\n";
  InitializeBlasCommon(filename);
}
void InitializeMKLMLLib(const char *filename) {
  std::cerr << "using mkl-ml...\n";
  InitializeBlasCommon(filename);
}
} // namespace core
} // namespace fast_transformers
