#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "fast_transformers/core/blas.h"

namespace fast_transformers {
namespace core {
static bool float_eq(float a, float b) { return std::abs(a - b) < 1e-5; }

TEST_CASE("blas-gemm") {
  AutoInitBlas();
  float A[] = {1, 2, 3, 4};
  float B[] = {2, 3, 4, 5};
  float C[] = {0, 0, 0, 0};

  Blas().sgemm_(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1, A, 2, B,
                2, 0, C, 2);
  REQUIRE(float_eq(C[0], 10));
  REQUIRE(float_eq(C[1], 13));
  REQUIRE(float_eq(C[2], 22));
  REQUIRE(float_eq(C[3], 29));
}
TEST_CASE("blas-batch-gemm") {
  AutoInitBlas();
  float A1[] = {1, 2, 3, 4};
  float B1[] = {2, 3, 4, 5};
  float C1[] = {0, 0, 0, 0};

  float A2[] = {1, 2, 3, 4};
  float B2[] = {2, 3, 4, 5};
  float C2[] = {0, 0, 0, 0};

  float *A[] = {A1, A2};
  float *B[] = {B1, B2};
  float *C[] = {C1, C2};

  BlasInt m[] = {2, 2};
  CBLAS_TRANSPOSE trans[] = {CblasNoTrans, CblasNoTrans};
  float alpha = 1;
  float beta = 0;
  BlasInt batch_size = 2;

  Blas().sgemm_batch_(CblasRowMajor, trans, trans, m, m, m, &alpha,
                      reinterpret_cast<float **>(A), m, B, m, &beta, C, m, 1,
                      &batch_size);
  REQUIRE(float_eq(C1[0], 10));
  REQUIRE(float_eq(C1[1], 13));
  REQUIRE(float_eq(C1[2], 22));
  REQUIRE(float_eq(C1[3], 29));
  REQUIRE(float_eq(C2[0], 10));
  REQUIRE(float_eq(C2[1], 13));
  REQUIRE(float_eq(C2[2], 22));
  REQUIRE(float_eq(C2[3], 29));
}

TEST_CASE("blas-sscal") {
  AutoInitBlas();
  float vec[] = {1, 2};
  Blas().sscal_(2, 2, vec, 1);
  REQUIRE(float_eq(vec[0], 2));
  REQUIRE(float_eq(vec[1], 4));
}

} // namespace core
} // namespace fast_transformers
