#include "fast_transformers/core/math_function.h"
#include "fast_transformers/core/blas.h"

namespace fast_transformers {
namespace core {
    template<>
    int cpu_blas_gemm<float>(
                          const bool TransA, const bool TransB,
                          const int M, const int N, const int K,
                          const float alpha,
                          const float *A, const int lda,
                          const float *B, const int ldb,
                          const float beta,
                          float *C, const int ldc
                          ) {
                g_blas_funcs_->sgemm_(CblasRowMajor,
                        TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans,
                        N, M, K,
                        alpha,
                        B, ldb,
                        A, lda,
                        beta,
                        C, ldc);
                return 1;

    }

    template<>
    int cpu_blas_gemm_rowmajor<float>(
                          const bool TransA, const bool TransB,
                          const int M, const int N, const int K,
                          const float alpha,
                          const float *A, const int lda,
                          const float *B, const int ldb,
                          const float beta,
                          float *C, const int ldc
                          ) {
                g_blas_funcs_->sgemm_(CblasRowMajor,
                        TransA ? CblasTrans : CblasNoTrans, TransB ? CblasTrans : CblasNoTrans,
                        N, M, K,
                        alpha,
                        B, ldb,
                        A, lda,
                        beta,
                        C, ldc);
                return 1;
    }

/*
    template<>
    int cpu_blas_gemm_batch<float>(
                                const bool TransA, const bool TransB,
                                int m, int n, int k,
                                const float alpha,
                                const float **Aarray, int lda,
                                const float **Barray, int ldb,
                                const float beta,
                                float **Carray, int ldc,
                                int batchCount) {
            CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
            cblas_sgemm_batch(CblasColMajor,
                              &transA, &transB,
                              &m, &n, &k,
                              &alpha,
                              Aarray, &lda,
                              Barray, &ldb,
                              &beta,
                              Carray, &ldc,
                              1, &batchCount);
            return 1;
    }


    template<>
    int cpu_blas_gemm_strided_batch<float>(
                                        const bool TransA, const bool TransB,
                                        int m, int n, int k,
                                        const float alpha,
                                        const float *A, int lda, long long int strideA,
                                        const float *B, int ldb, long long int strideB,
                                        const float beta,
                                        float *C, int ldc, long long int strideC,
                                        int batchCount) {
            const float *A_Array[batchCount];
            const float *B_Array[batchCount];
            float *C_Array[batchCount];
            for (int i = 0; i < batchCount; ++i) {
                A_Array[i] = A + strideA * i;
                B_Array[i] = B + strideB * i;
                C_Array[i] = C + strideC * i;
            }

            CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
            cblas_sgemm_batch(CblasColMajor,
                              &transA, &transB,
                              &m, &n, &k,
                              &alpha,
                              A_Array, &lda,
                              B_Array, &ldb,
                              &beta,
                              C_Array, &ldc,
                              1, &batchCount);
            return 1;
    }
*/

}//namespace core
}//namespace fast_transformers