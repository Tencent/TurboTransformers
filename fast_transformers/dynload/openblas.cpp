#include "fast_transformers/dynload/openblas.h"
#include "absl/strings/str_cat.h"
#include <dlfcn.h>
namespace fast_transformers {
namespace dynload {
struct Openblas::Impl {
  void *so_handle_;

  explicit Impl(const char *fn) {
    so_handle_ = dlopen(fn, RTLD_LAZY | RTLD_LOCAL);
    if (so_handle_ == nullptr) {
      throw std::runtime_error(absl::StrCat("Cannot load openblas on ", fn));
    }
  }
  ~Impl() { dlclose(so_handle_); }
};
Openblas::Openblas(const char *path) : m_(new Impl(path)) {}
Openblas::Openblas(Openblas &&o) : m_(std::move(o.m_)) {}
Openblas &Openblas::operator=(Openblas &&o) { m_ = std::move(o.m_); }
Openblas::~Openblas() = default;

} // namespace dynload
} // namespace fast_transformers
