#include "blas.h"
#include "absl/strings/str_cat.h"
#include <experimental/filesystem>
namespace fast_transformers {
namespace dynload {
#ifdef __APPLE__
static const char *dynlib_suffix_ = ".dylib";
#else
static const char *dynlib_suffix_ = ".so";
#endif

namespace details {
BlasProvider g_blas_provider_;
std::once_flag g_blas_once_;
} // namespace details

namespace fs = std::experimental::filesystem;

void AutoInitBlas() {
  std::vector<fs::path> pathes = {fs::path("/usr/lib"),
                                  fs::path("/usr/local/lib"),
                                  fs::path("/usr/local/opt/openblas/lib")};
  std::string openblas_libname = absl::StrCat("libopenblas", dynlib_suffix_);

  for (auto &p : pathes) {
    auto libpath = p / openblas_libname;
    if (fs::exists(libpath)) {
      InitializeBlas<Openblas>(libpath.c_str());
      return;
    }
  }

  throw std::runtime_error("Cannot initialize blas automatically");
}
} // namespace dynload
} // namespace fast_transformers
