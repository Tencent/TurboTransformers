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
static const char *mklml_prefix = "libmklml";
#else
static const char *dynlib_suffix_ = ".so";
static const char *mklml_prefix = "libmklml_intel";
#endif

std::unique_ptr<CBlasFuncs, CBlasFuncDeleter> g_blas_funcs_;

void AutoInitBlas() {
  std::string openblas_libname = absl::StrCat("libopenblas", dynlib_suffix_);
  std::string mklml_libname = absl::StrCat(mklml_prefix, dynlib_suffix_);
  char *conda_prefix = std::getenv("CONDA_PREFIX");
  if (conda_prefix != nullptr) {
    auto p = fs::path(conda_prefix) / "lib" / mklml_libname;
    if (fs::exists(p)) {
      InitializeMKLMLLib(p.c_str());
      return;
    }
  }

  std::vector<fs::path> pathes = {fs::path("./"), fs::path("/usr/lib"),
                                  fs::path("/usr/local/lib"),
                                  fs::path("/usr/local/opt/openblas/lib")};

  for (auto &p : pathes) {
    auto libpath = p / mklml_libname;
    if (fs::exists(libpath)) {
      InitializeOpenblasLib(libpath.c_str());
      return;
    }
  }

  for (auto &p : pathes) {
    auto libpath = p / openblas_libname;
    if (fs::exists(libpath)) {
      InitializeOpenblasLib(libpath.c_str());
      return;
    }
  }

  throw std::runtime_error("Cannot initialize blas automatically");
}

/**
 * Load common blas routines from dynamic library file.
 *
 * Result is stored into g_blas_funcs_.
 * @param filename
 */
static void InitializeBlasCommon(const char *filename) {
  void *lib = dlopen(filename, RTLD_LAZY | RTLD_LOCAL);
  FT_ENFORCE_NE(lib, nullptr, "Cannot load blas library %s", filename);

  g_blas_funcs_.reset(new CBlasFuncs());
  g_blas_funcs_->shared_library_ = lib;
  g_blas_funcs_->sgemm_ =
      reinterpret_cast<decltype(cblas_sgemm) *>(dlsym(lib, "cblas_sgemm"));
  FT_ENFORCE_NE(g_blas_funcs_->sgemm_, nullptr, "Cannot load cblas_sgemm");

  g_blas_funcs_->sscal_ =
      reinterpret_cast<decltype(cblas_sscal) *>(dlsym(lib, "cblas_sscal"));
  FT_ENFORCE_NE(g_blas_funcs_->sscal_, nullptr, "Cannot load cblas_sscal");
}

void InitializeOpenblasLib(const char *filename) {
  std::cerr << "using openblas...\n";
  InitializeBlasCommon(filename);

  // Since openblas did not provide cblas_sgemm_batch, just use naive
  // implementation.
  g_blas_funcs_->sgemm_batch_ = naive_cblas_sgemm_batch;
}
void InitializeMKLMLLib(const char *filename) {
  std::cerr << "using mkl-ml...\n";
  InitializeBlasCommon(filename);
  g_blas_funcs_->sgemm_batch_ = reinterpret_cast<decltype(cblas_sgemm_batch) *>(
      dlsym(g_blas_funcs_->shared_library_, "cblas_sgemm_batch"));
  FT_ENFORCE_NE(g_blas_funcs_->sgemm_batch_, nullptr,
                "Cannot load cblas_sgemm_batch");
}

void naive_cblas_sgemm_batch(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE *transa_array,
                             CBLAS_TRANSPOSE *transb_array, int *m_array,
                             int *n_array, int *k_array,
                             const float *alpha_array, const float **a_array,
                             int *lda_array, const float **b_array,
                             int *ldb_array, const float *beta_array,
                             float **c_array, int *ldc_array, int group_count,
                             int *group_size) {
  int idx = 0;
  for (int i = 0; i < group_count; ++i) {
    auto alpha = alpha_array[i];
    auto beta = beta_array[i];
    for (int j = 0; j < group_size[i]; ++j) {
      Blas().sgemm_(Layout, transa_array[idx], transb_array[idx], m_array[idx],
                    n_array[idx], k_array[idx], alpha, a_array[idx],
                    lda_array[idx], b_array[idx], ldb_array[idx], beta,
                    c_array[idx], ldc_array[idx]);
      ++idx;
    }
  }
}

} // namespace core
} // namespace fast_transformers
