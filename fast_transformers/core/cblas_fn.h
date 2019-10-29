#include "fast_transformers/core/cblas_defs.h"
#include <dlfcn.h>
namespace fast_transformers {
namespace core {
extern void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const float alpha, const float *A,
                        const int lda, const float *B, const int ldb,
                        const float beta, float *C, const int ldc);

extern void cblas_sgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *transa_array,
    const CBLAS_TRANSPOSE *transb_array, const int *m_array, const int *n_array,
    const int *k_array, const float *alpha_array, const float **a_array,
    const int *lda_array, const float **b_array, const int *ldb_array,
    const float *beta_array, float **c_array, const int *ldc_array,
    const int group_count, const int *group_size);

struct CBlasFuncs {
  decltype(cblas_sgemm) *sgemm_;
  decltype(cblas_sgemm_batch) *sgemm_batch_;

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
