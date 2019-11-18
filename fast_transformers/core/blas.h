#pragma once
#include <stdint.h>
#include <mutex>
#include "absl/types/variant.h"
#include "fast_transformers/core/enforce.h"

namespace fast_transformers {
namespace core {

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113,
  CblasConjNoTrans = 114
};
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
using CBLAS_LAYOUT = CBLAS_ORDER;
using BlasInt = BLASINT;

extern void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB, BlasInt M, BlasInt N, BlasInt K,
                        float alpha, const float *A, BlasInt lda,
                        const float *B, BlasInt ldb, float beta, float *C,
                        BlasInt ldc);

// cblas_sgemm_batch
extern void cblas_sgemm_batch(CBLAS_LAYOUT Layout,
                              CBLAS_TRANSPOSE *transa_array,
                              CBLAS_TRANSPOSE *transb_array, BlasInt *m_array,
                              BlasInt *n_array, BlasInt *k_array,
                              const float *alpha_array, const float **a_array,
                              BlasInt *lda_array, const float **b_array,
                              BlasInt *ldb_array, const float *beta_array,
                              float **c_array, BlasInt *ldc_array,
                              BlasInt group_count, BlasInt *group_size);

extern void cblas_sscal(BlasInt N, float alpha, float *X, BlasInt incX);
extern void cblas_tanh(BlasInt N, float *X, float *Y);

}  // namespace core
}  // namespace fast_transformers
