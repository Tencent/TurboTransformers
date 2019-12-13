#include "mat_mul.h"

namespace fast_transformers {
namespace layers {
namespace kernels {
void MatMul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
            bool b_trans, float alpha, core::Tensor* out, float beta) {
  BlasInt a_rows = A.rows();
  BlasInt a_cols = A.cols();
  BlasInt b_rows = B.rows();
  BlasInt b_cols = B.cols();

  BlasInt M = a_trans ? a_cols : a_rows;
  BlasInt N = b_trans ? b_rows : b_cols;

  BlasInt K_a = a_trans ? a_rows : a_cols;
  BlasInt K_b = b_trans ? b_cols : b_rows;
  FT_ENFORCE_EQ(K_a, K_b, "matrix shape mismatch");

  CBLAS_TRANSPOSE transA = a_trans ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = b_trans ? CblasTrans : CblasNoTrans;

  BlasInt lda = (transA == CblasNoTrans) ? K_a : M;
  BlasInt ldb = (transB == CblasNoTrans) ? N : K_a;
  BlasInt ldc = N;

  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K_a, alpha, A.data<float>(),
              lda, B.data<float>(), ldb, beta, out->mutableData<float>(), ldc);
}
void BatchMatMul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
                 bool b_trans, float alpha, core::Tensor* C, float beta) {
  auto* A_shape = &A.shape(0);
  auto A_ndim = A.n_dim();
  auto* B_shape = &B.shape(0);
  auto B_ndim = B.n_dim();
  FT_ENFORCE_GT(A_ndim, 2, "A must at least be 3 dims");
  FT_ENFORCE_GT(B_ndim, 2, "B must at least be 3 dims");

  BlasInt a_rows = A_shape[A_ndim - 2];
  BlasInt a_cols = A_shape[A_ndim - 1];
  BlasInt b_rows = B_shape[B_ndim - 2];
  BlasInt b_cols = B_shape[B_ndim - 1];

  BlasInt a_batch_size = std::accumulate(A_shape, A_shape + A_ndim - 2, 1,
                                         std::multiplies<int64_t>());
  BlasInt b_batch_size = std::accumulate(B_shape, B_shape + B_ndim - 2, 1,
                                         std::multiplies<int64_t>());

  FT_ENFORCE_EQ(a_batch_size, b_batch_size, "BatchSize mismatch");

  BlasInt M = a_trans ? a_cols : a_rows;
  BlasInt N = b_trans ? b_rows : b_cols;
  BlasInt K_a = a_trans ? a_rows : a_cols;
  BlasInt K_b = b_trans ? b_cols : b_rows;
  FT_ENFORCE_EQ(K_a, K_b, "K mismatch");

  auto* C_shape = &C->shape(0);
  auto C_ndim = C->n_dim();

  BlasInt c_rows = C_shape[C_ndim - 2];
  BlasInt c_cols = C_shape[C_ndim - 1];
  BlasInt c_batch_size = std::accumulate(C_shape, C_shape + C_ndim - 2, 1,
                                         std::multiplies<int64_t>());

  FT_ENFORCE_EQ(c_rows, M, "C shape mismatch");
  FT_ENFORCE_EQ(c_cols, N, "C shape mismatch");
  FT_ENFORCE_EQ(c_batch_size, b_batch_size, "C BatchSize mismatch");

  BlasInt offsetA = a_rows * a_cols;
  BlasInt offsetB = b_rows * b_cols;
  BlasInt offsetC = c_rows * c_cols;

  std::unique_ptr<const float* []> A_array(new const float*[a_batch_size]);
  std::unique_ptr<const float* []> B_array(new const float*[b_batch_size]);
  std::unique_ptr<float* []> C_array(new float*[c_batch_size]);

  auto* a_ptr = A.data<float>();
  auto* b_ptr = B.data<float>();
  auto* c_ptr = C->mutableData<float>();

  for (int i = 0; i < a_batch_size; ++i) {
    A_array[i] = a_ptr + i * offsetA;
    B_array[i] = b_ptr + i * offsetB;
    C_array[i] = c_ptr + i * offsetC;
  }
  auto transA = a_trans ? CblasTrans : CblasNoTrans;
  auto transB = b_trans ? CblasTrans : CblasNoTrans;

  BlasInt lda = (transA == CblasNoTrans) ? K_a : M;
  BlasInt ldb = (transB == CblasNoTrans) ? N : K_a;
  BlasInt ldc = N;

  cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &M, &N, &K_a, &alpha,
                    A_array.get(), &lda, B_array.get(), &ldb, &beta,
                    C_array.get(), &ldc, 1, &a_batch_size);
}
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
