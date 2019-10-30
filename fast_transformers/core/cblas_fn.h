#include "fast_transformers/core/cblas_defs.h"
#include <dlfcn.h>
namespace fast_transformers {
namespace core {

using BlasInt = int;

// NOTE(josephyu): These method just provide the interface of blas routines.
// do not try to invoke these method directly. Use `Blas().sgemm` instead.
extern void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB, BlasInt M, BlasInt N, BlasInt K,
                        float alpha, float *A, BlasInt lda, float *B,
                        BlasInt ldb, float beta, float *C, BlasInt ldc);

extern void
cblas_sgemm_batch(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE *transa_array,
                  CBLAS_TRANSPOSE *transb_array, BlasInt *m_array,
                  BlasInt *n_array, BlasInt *k_array, float *alpha_array,
                  float **a_array, BlasInt *lda_array, float **b_array,
                  BlasInt *ldb_array, float *beta_array, float **c_array,
                  BlasInt *ldc_array, BlasInt group_count, BlasInt *group_size);

extern void cblas_sscal(BlasInt N, float alpha, float *X, BlasInt incX);

struct CBlasFuncs {
  decltype(cblas_sgemm) *sgemm_;
  decltype(cblas_sgemm_batch) *sgemm_batch_;
  decltype(cblas_sscal) *sscal_;

  void *shared_library_;
};

struct CBlasFuncDeleter {
  void operator()(CBlasFuncs *f) const {
    if (f == nullptr)
      return;
    if (f->shared_library_) {
      dlclose(f->shared_library_);
    }
    delete f;
  }
};

} // namespace core
} // namespace fast_transformers
