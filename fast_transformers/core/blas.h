#pragma once
#include "absl/types/variant.h"
#include "fast_transformers/core/cblas_fn.h"
#include "fast_transformers/core/enforce.h"
#include <mutex>

namespace fast_transformers {
namespace core {

extern std::unique_ptr<CBlasFuncs, CBlasFuncDeleter> g_blas_funcs_;

void InitializeOpenblasLib(const char *filename);
void InitializeMKLMLLib(const char *filename);

void AutoInitBlas();

inline static CBlasFuncs &Blas() {
  FT_ENFORCE_NE(g_blas_funcs_, nullptr, "Must initialize blas lib");
  return *g_blas_funcs_;
}

extern void
naive_cblas_sgemm_batch(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE *transa_array,
                        CBLAS_TRANSPOSE *transb_array, int *m_array,
                        int *n_array, int *k_array, float *alpha_array,
                        float **a_array, int *lda_array, float **b_array,
                        int *ldb_array, float *beta_array, float **c_array,
                        int *ldc_array, int group_count, int *group_size);

} // namespace core
} // namespace fast_transformers
