// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#include "mat_mul.h"

#include "common.h"
#ifdef TT_WITH_CUDA
#include <cuda.h>

#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/cuda_enforce.cuh"
#endif
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {
void MatMul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
            bool b_trans, float alpha, core::Tensor* out, float beta,
            const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, A.device_type());
#endif
  BlasInt a_cols = A.shape(-1);
  BlasInt a_rows = A.numel() / a_cols;
  BlasInt b_cols = B.shape(-1);
  BlasInt b_rows = B.numel() / b_cols;

  BlasInt M = a_trans ? a_cols : a_rows;
  BlasInt N = b_trans ? b_rows : b_cols;

  BlasInt K_a = a_trans ? a_rows : a_cols;
  BlasInt K_b = b_trans ? b_cols : b_rows;
  TT_ENFORCE_EQ(K_a, K_b, "matrix shape mismatch %d vs %d", K_a, K_b);
  TT_ENFORCE(common::is_same_device_ctx(A.device_ctx(), B.device_ctx()),
             "MatMul error: the device of A and B is different.");
  TT_ENFORCE(common::is_same_device_ctx(A.device_ctx(), out->device_ctx()),
             "MatMul error: the device of A and out is different.");

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
#ifdef TT_WITH_CUDA
    cublasOperation_t transA = a_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = b_trans ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = (transA == CUBLAS_OP_N) ? K_a : M;
    int ldb = (transB == CUBLAS_OP_N) ? N : K_a;
    int ldc = N;

    auto& gpu_ctx =
        ::turbo_transformers::core::CUDADeviceContext::GetInstance();

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9010
    if (gpu_ctx.compute_major() >= 5) {

#if defined(WITH_TENSOR_CORE)      
      auto cublas_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      auto math_algo = CUBLAS_TENSOR_OP_MATH;
#else
      auto cublas_algo = CUBLAS_GEMM_DEFAULT;
      auto math_algo = CUBLAS_DEFAULT_MATH;
#endif  
      TT_ENFORCE_CUDA_SUCCESS(
          cublasSetMathMode(gpu_ctx.cublas_handle(), math_algo));
      TT_ENFORCE_CUDA_SUCCESS(cublasGemmEx(
          gpu_ctx.cublas_handle(), transB, transA, N, M, K_a, &alpha,
          B.data<float>(), CUDA_R_32F, ldb, A.data<float>(), CUDA_R_32F, lda,
          &beta, out->mutableData<float>(), CUDA_R_32F, ldc, CUDA_R_32F,
          cublas_algo));
      TT_ENFORCE_CUDA_SUCCESS(
          cublasSetMathMode(gpu_ctx.cublas_handle(), CUBLAS_DEFAULT_MATH));
    } else {
      TT_ENFORCE_CUDA_SUCCESS(cublasSgemmEx(
          gpu_ctx.cublas_handle(), transB, transA, N, M, K_a, &alpha,
          B.data<float>(), CUDA_R_32F, ldb, A.data<float>(), CUDA_R_32F, lda,
          &beta, out->mutableData<float>(), CUDA_R_32F, ldc));
    }
#else
    TT_ENFORCE_CUDA_SUCCESS(cublasSgemm(gpu_ctx.cublas_handle(), transB, transA,
                                        N, M, K_a, &alpha, B.data<float>(), ldb,
                                        A.data<float>(), lda, &beta,
                                        out->mutableData<float>(), ldc));
#endif
#else
    TT_THROW("CUDA is not supported for MatMul");
#endif
  } else {
    TT_THROW("device_type %d is not supported for MatMul", A.device_type());
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, A.device_type());
#endif
}
void BatchMatMul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
                 bool b_trans, float alpha, core::Tensor* C, float beta,
                 const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, A.device_type());
#endif
  auto* A_shape = &A.shape(0);
  auto A_ndim = A.n_dim();
  auto* B_shape = &B.shape(0);
  auto B_ndim = B.n_dim();
  TT_ENFORCE_GT(A_ndim, 2, "A must at least be 3 dims");
  TT_ENFORCE_GT(B_ndim, 2, "B must at least be 3 dims");

  BlasInt a_rows = A_shape[A_ndim - 2];
  BlasInt a_cols = A_shape[A_ndim - 1];
  BlasInt b_rows = B_shape[B_ndim - 2];
  BlasInt b_cols = B_shape[B_ndim - 1];

  BlasInt a_batch_size = std::accumulate(A_shape, A_shape + A_ndim - 2, 1,
                                         std::multiplies<int64_t>());
  BlasInt b_batch_size = std::accumulate(B_shape, B_shape + B_ndim - 2, 1,
                                         std::multiplies<int64_t>());

  TT_ENFORCE_EQ(a_batch_size, b_batch_size, "BatchSize mismatch");

  BlasInt M = a_trans ? a_cols : a_rows;
  BlasInt N = b_trans ? b_rows : b_cols;
  BlasInt K_a = a_trans ? a_rows : a_cols;
  BlasInt K_b = b_trans ? b_cols : b_rows;
  TT_ENFORCE_EQ(K_a, K_b, "K mismatch");

  auto* C_shape = &C->shape(0);
  auto C_ndim = C->n_dim();

  BlasInt c_rows = C_shape[C_ndim - 2];
  BlasInt c_cols = C_shape[C_ndim - 1];
  BlasInt c_batch_size = std::accumulate(C_shape, C_shape + C_ndim - 2, 1,
                                         std::multiplies<int64_t>());

  TT_ENFORCE_EQ(c_rows, M, "C shape mismatch");
  TT_ENFORCE_EQ(c_cols, N, "C shape mismatch");
  TT_ENFORCE_EQ(c_batch_size, b_batch_size, "C BatchSize mismatch");

  BlasInt offsetA = a_rows * a_cols;
  BlasInt offsetB = b_rows * b_cols;
  BlasInt offsetC = c_rows * c_cols;

  if (A.device_type() == kDLCPU && B.device_type() == kDLCPU &&
      C->device_type() == kDLCPU) {
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
  } else if (A.device_type() == kDLGPU && B.device_type() == kDLGPU &&
             C->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    auto transA = a_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transB = b_trans ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = (transA == CUBLAS_OP_N) ? K_a : M;
    int ldb = (transB == CUBLAS_OP_N) ? N : K_a;
    int ldc = N;
    auto& gpu_ctx =
        ::turbo_transformers::core::CUDADeviceContext::GetInstance();
    cublasSgemmStridedBatched(
        gpu_ctx.cublas_handle(), transB, transA, N, M, K_a, &alpha,
        B.data<float>(), ldb, offsetB, A.data<float>(), lda, offsetA, &beta,
        C->mutableData<float>(), ldc, offsetC, a_batch_size);
#endif
  } else {
    TT_THROW("device_type %d is not supported!", A.device_type());
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, A.device_type());
#endif
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
