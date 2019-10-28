#include "fast_transformers/dynload/cblas_defs.h"
#include <dlfcn.h>
namespace fast_transformers {
namespace dynload {
extern void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const float alpha, const float *A,
                        const int lda, const float *B, const int ldb,
                        const float beta, float *C, const int ldc);
struct CBlasFuncs {
  decltype(cblas_sgemm) *sgemm_;

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

} // namespace dynload
} // namespace fast_transformers
