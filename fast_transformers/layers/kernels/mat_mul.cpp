#include "mat_mul.h"

namespace fast_transformers {
namespace layers {
namespace kernels {
void Matmul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
            bool b_trans, float alpha, core::Tensor* out, float beta) {
  int a_rows = A.rows();
  int a_cols = A.cols();
  int b_rows = B.rows();
  int b_cols = B.cols();

  int M = a_trans ? a_cols : a_rows;
  int N = b_trans ? b_rows : b_cols;

  int K_a = a_trans ? a_rows : a_cols;
  int K_b = b_trans ? b_cols : b_rows;
  FT_ENFORCE_EQ(K_a, K_b, "matrix shape mismatch");

  CBLAS_TRANSPOSE transA = a_trans ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = b_trans ? CblasTrans : CblasNoTrans;

  int lda = (transA == CblasNoTrans) ? K_a : M;
  int ldb = (transB == CblasNoTrans) ? N : K_a;
  int ldc = N;

  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K_a, alpha, A.data<float>(),
              lda, B.data<float>(), ldb, beta, out->mutableData<float>(), ldc);
}
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
