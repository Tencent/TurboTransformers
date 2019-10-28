#pragma once

#include "absl/types/variant.h"
#include "fast_transformers/dynload/openblas.h"
#include <mutex>

namespace fast_transformers {
namespace dynload {

using BlasProvider = absl::variant<absl::monostate, Openblas>;

namespace details {
extern BlasProvider g_blas_provider_;
extern std::once_flag g_blas_once_;
} // namespace details

template <typename BlasProviderT> void InitializeBlas(const char *filename) {
  std::call_once(details::g_blas_once_, [filename] {
    details::g_blas_provider_ = BlasProviderT(filename);
  });
}

extern void AutoInitBlas();

} // namespace dynload
} // namespace fast_transformers
