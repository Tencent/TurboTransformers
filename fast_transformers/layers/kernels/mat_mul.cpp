#include "mat_mul.h"

#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#endif

namespace fast_transformers {
namespace layers {
namespace kernels {
void MatMul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
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

  if (A.device_type() == kDLCPU && B.device_type() == kDLCPU &&
      out->device_type() == kDLCPU) {
    CBLAS_TRANSPOSE transA = a_trans ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = b_trans ? CblasTrans : CblasNoTrans;

    int lda = (transA == CblasNoTrans) ? K_a : M;
    int ldb = (transB == CblasNoTrans) ? N : K_a;
    int ldc = N;

    cblas_sgemm(CblasRowMajor, transA, transB, M, N, K_a, alpha,
                A.data<float>(), lda, B.data<float>(), ldb, beta,
                out->mutableData<float>(), ldc);
  } else if (A.device_type() == kDLGPU && B.device_type() == kDLGPU &&
             out->device_type() == kDLGPU) {
#ifdef WITH_CUDA
    cublasOperation_t transA = a_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = b_trans ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = (transA == CUBLAS_OP_N) ? K_a : M;
    int ldb = (transB == CUBLAS_OP_N) ? N : K_a;
    int ldc = N;

    core::DeviceContextPool& pool = core::CUDADeviceContext::GetInstance();
    gpu_ctx.CublasCall([&](cublasHandle_t handle) {
      cublasSgemm(handle, transB, transA, N, M, K_a, &alpha, B.data<float>(),
                  ldb, A.data<float>(), lda, &beta, out->mutableData<float>(),
                  ldc);
    });
#endif
  }
}
void BatchMatMul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
                 bool b_trans, float alpha, core::Tensor* C, float beta) {
  auto* A_shape = &A.shape(0);
  auto A_ndim = A.n_dim();
  auto* B_shape = &B.shape(0);
  auto B_ndim = B.n_dim();
  FT_ENFORCE_GT(A_ndim, 2, "A must at least be 3 dims");
  FT_ENFORCE_GT(B_ndim, 2, "B must at least be 3 dims");

  int a_rows = A_shape[A_ndim - 2];
  int a_cols = A_shape[A_ndim - 1];
  int b_rows = B_shape[B_ndim - 2];
  int b_cols = B_shape[B_ndim - 1];

  int a_batch_size = std::accumulate(A_shape, A_shape + A_ndim - 2, 1,
                                     std::multiplies<int64_t>());
  int b_batch_size = std::accumulate(B_shape, B_shape + B_ndim - 2, 1,
                                     std::multiplies<int64_t>());

  FT_ENFORCE_EQ(a_batch_size, b_batch_size, "BatchSize mismatch");

  int M = a_trans ? a_cols : a_rows;
  int N = b_trans ? b_rows : b_cols;
  int K_a = a_trans ? a_rows : a_cols;
  int K_b = b_trans ? b_cols : b_rows;
  FT_ENFORCE_EQ(K_a, K_b, "K mismatch");

  auto* C_shape = &C->shape(0);
  auto C_ndim = C->n_dim();

  int c_rows = C_shape[C_ndim - 2];
  int c_cols = C_shape[C_ndim - 1];
  int c_batch_size = std::accumulate(C_shape, C_shape + C_ndim - 2, 1,
                                     std::multiplies<int64_t>());

  FT_ENFORCE_EQ(c_rows, M, "C shape mismatch");
  FT_ENFORCE_EQ(c_cols, N, "C shape mismatch");
  FT_ENFORCE_EQ(c_batch_size, b_batch_size, "C BatchSize mismatch");

  int offsetA = a_rows * a_cols;
  int offsetB = b_rows * b_cols;
  int offsetC = c_rows * c_cols;

  std::unique_ptr<const float*[]> A_array(new const float*[a_batch_size]);
  std::unique_ptr<const float*[]> B_array(new const float*[b_batch_size]);
  std::unique_ptr<float*[]> C_array(new float*[c_batch_size]);

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

  int lda = (transA == CblasNoTrans) ? K_a : M;
  int ldb = (transB == CblasNoTrans) ? N : K_a;
  int ldc = N;

  cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &M, &N, &K_a, &alpha,
                    A_array.get(), &lda, B_array.get(), &ldb, &beta,
                    C_array.get(), &ldc, 1, &a_batch_size);
}
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
