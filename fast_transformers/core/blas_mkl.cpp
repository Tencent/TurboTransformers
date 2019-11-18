#include "blas.h"
#include "mkl.h"

namespace fast_transformers {
namespace core {
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, BlasInt M, BlasInt N, BlasInt K,
                 float alpha, const float* A, BlasInt lda, const float* B,
                 BlasInt ldb, float beta, float* C, BlasInt ldc) {
  ::cblas_sgemm(static_cast<::CBLAS_LAYOUT>(layout),
                static_cast<::CBLAS_TRANSPOSE>(TransA),
                static_cast<::CBLAS_TRANSPOSE>(TransB), M, N, K, alpha, A, lda,
                B, ldb, beta, C, ldc);
}
void cblas_sgemm_batch(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE* transa_array,
                       CBLAS_TRANSPOSE* transb_array, BlasInt* m_array,
                       BlasInt* n_array, BlasInt* k_array,
                       const float* alpha_array, const float** a_array,
                       BlasInt* lda_array, const float** b_array,
                       BlasInt* ldb_array, const float* beta_array,
                       float** c_array, BlasInt* ldc_array, BlasInt group_count,
                       BlasInt* group_size) {
  ::cblas_sgemm_batch(static_cast<::CBLAS_LAYOUT>(Layout),
                      reinterpret_cast<::CBLAS_TRANSPOSE*>(transa_array),
                      reinterpret_cast<::CBLAS_TRANSPOSE*>(transb_array),
                      m_array, n_array, k_array, alpha_array, a_array,
                      lda_array, b_array, ldb_array, beta_array, c_array,
                      ldc_array, group_count, group_size);
}
void cblas_sscal(BlasInt N, float alpha, float* X, BlasInt incX) {
  ::cblas_sscal(N, alpha, X, incX);
}
void cblas_tanh(BlasInt N, float* X, float* Y) { ::vsTanh(N, X, Y); }

}  // namespace core
}  // namespace fast_transformers
