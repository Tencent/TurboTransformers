#pragma once

#include "absl/types/variant.h"
#include "fast_transformers/dynload/cblas_fn.h"
#include <mutex>

namespace fast_transformers {
namespace dynload {

extern std::unique_ptr<CBlasFuncs> g_blas_funcs_;

void InitializeOpenblasLib(const char *filename);

void AutoInitBlas();

inline static CBlasFuncs &Blas() {
  if (g_blas_funcs_) {
    throw std::runtime_error("Must initialize blas lib");
  }
  return *g_blas_funcs_;
}

} // namespace dynload
} // namespace fast_transformers
