#include <memory>

#include "absl/strings/str_cat.h"
#include "blas.h"
#include <experimental/filesystem>
namespace fast_transformers {
namespace dynload {
#ifdef __APPLE__
static const char *dynlib_suffix_ = ".dylib";
#else
static const char *dynlib_suffix_ = ".so";
#endif

std::unique_ptr<CBlasFuncs> g_blas_funcs_;

namespace fs = std::experimental::filesystem;

void AutoInitBlas() {
  std::vector<fs::path> pathes = {fs::path("./"), fs::path("/usr/lib"),
                                  fs::path("/usr/local/lib"),
                                  fs::path("/usr/local/opt/openblas/lib")};
  std::string openblas_libname = absl::StrCat("libopenblas", dynlib_suffix_);

  for (auto &p : pathes) {
    auto libpath = p / openblas_libname;
    if (fs::exists(libpath)) {
      InitializeOpenblasLib(libpath.c_str());
      return;
    }
  }

  throw std::runtime_error("Cannot initialize blas automatically");
}

void InitializeOpenblasLib(const char *filename) {
  void *lib = dlopen(filename, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    throw std::runtime_error("Cannot load openblas");
  }

  g_blas_funcs_ = std::make_unique<CBlasFuncs>();
  g_blas_funcs_->shared_library_ = lib;
  g_blas_funcs_->sgemm_ =
      reinterpret_cast<decltype(cblas_sgemm) *>(dlsym(lib, "cblas_sgemm"));
}
} // namespace dynload
} // namespace fast_transformers
