#include "fast_transformers/core/cblas_defs.h"
#include <dlfcn.h>
namespace fast_transformers {
namespace core {

using blasint = int;

extern void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB, blasint M, blasint N, blasint K,
                        float alpha, float *A, blasint lda, float *B,
                        blasint ldb, float beta, float *C, blasint ldc);

extern void cblas_sgemm_batch(CBLAS_LAYOUT Layout,
                              CBLAS_TRANSPOSE *transa_array,
                              CBLAS_TRANSPOSE *transb_array, int *m_array,
                              int *n_array, int *k_array, float *alpha_array,
                              float **a_array, int *lda_array, float **b_array,
                              int *ldb_array, float *beta_array,
                              float **c_array, int *ldc_array, int group_count,
                              int *group_size);

extern void cblas_sscal(int N, float alpha, float *X, int incX);

struct CBlasFuncs {
  decltype(cblas_sgemm) *sgemm_;
  decltype(cblas_sgemm_batch) *sgemm_batch_;
  decltype(cblas_sscal) *sscal_;

  void *shared_library_;
};

struct CBlasFuncDeleter {
  void operator()(CBlasFuncs *f) {
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
