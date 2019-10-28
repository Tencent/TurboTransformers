#pragma once
#include "absl/types/variant.h"
#include "fast_transformers/core/cblas_fn.h"
#include "fast_transformers/core/enforce.h"
#include <mutex>

namespace fast_transformers {
namespace core {

extern std::unique_ptr<CBlasFuncs, CBlasFuncDeleter> g_blas_funcs_;

void InitializeOpenblasLib(const char *filename);

void AutoInitBlas();

inline static CBlasFuncs &Blas() {
  FT_ENFORCE_NE(g_blas_funcs_, nullptr, "Must initialize blas lib");
  return *g_blas_funcs_;
}

} // namespace core
} // namespace fast_transformers
