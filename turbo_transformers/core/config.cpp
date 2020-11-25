

#include "turbo_transformers/core/config.h"
#ifdef _OPENMP
#include "omp.h"
#endif
#include "blas.h"
#include "turbo_transformers/core/blas.h"

namespace turbo_transformers {
namespace core {
void SetNumThreads(int n_th) {
// The order seems important. Set MKL NUM_THREADS before OMP.
#ifdef TT_BLAS_USE_MKL
  mkl_set_num_threads(n_th);
#elif TT_BLAS_USE_OPENBLAS
  openblas_set_num_threads(n_th);
#elif TT_BLAS_USE_BLIS
#endif
#ifdef _OPENMP
  omp_set_num_threads(n_th);
#endif
}

BlasProvider GetBlasProvider() {
#ifdef TT_BLAS_USE_MKL
  return BlasProvider::MKL;
#elif defined(TT_BLAS_USE_OPENBLAS)
  return BlasProvider::OpenBlas;
#elif defined(TT_BLAS_USE_BLIS)
  return BlasProvider::BLIS;
#else
#error "unexpected code";
#endif
}

}  // namespace core
}  // namespace turbo_transformers
