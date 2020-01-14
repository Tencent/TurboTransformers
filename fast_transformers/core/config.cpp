#include "fast_transformers/core/config.h"

namespace fast_transformers {
namespace core {
bool IsWithCUDA() {
#ifdef FT_WITH_CUDA
  return true;
#else
  return false;
#endif
}
BlasProvider GetBlasProvider() {
#ifdef FT_BLAS_USE_MKL
  return BlasProvider::MKL;
#elif defined(FT_BLAS_USE_OPENBLAS)
  return BlasProvider::OpenBlas;
#else
#error "unexpected code";
#endif
}
}  // namespace core
}  // namespace fast_transformers
