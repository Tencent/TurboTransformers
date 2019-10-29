#pragma once
#include <cstddef>

namespace fast_transformers {
namespace core {
    template<typename T>
    int cpu_blas_gemm(
                   const bool TransA, const bool TransB,
                   const int M, const int N, const int K,
                   const float alpha,
                   const T *A, const int lda,
                   const T *B, const int ldb,
                   const float beta,
                   T *C, const int ldc);

    template<typename T>
    int cpu_blas_gemm_rowmajor(
                   const bool TransA, const bool TransB,
                   const int M, const int N, const int K,
                   const float alpha,
                   const T *A, const int lda,
                   const T *B, const int ldb,
                   const float beta,
                   T *C, const int ldc);

    template<typename T>
    int cpu_blas_gemm_batch(
                        const bool TransA, const bool TransB,
                         int m, int n, int k,
                         const float alpha,
                         const T **Aarray, int lda,
                         const T **Barray, int ldb,
                         const float beta,
                         T **Carray, int ldc,
                         int batchCount);

    template<typename T>
    int cpu_blas_gemm_strided_batch(
                                 const bool TransA, const bool TransB,
                                 int m, int n, int k,
                                 const float alpha,
                                 const T *A, int lda, long long int strideA,
                                 const T *B, int ldb, long long int strideB,
                                 const float beta,
                                 T *C, int ldc, long long int strideC,
                                 int batchCoun);

}//namespace core
}//namespace fast_transformers