#define CATCH_CONFIG_MAIN
#include "fast_transformers/core/blas.h"
#include <chrono>
#include "catch2/catch.hpp"

namespace fast_transformers {
namespace core {
static bool float_eq(float a, float b) { return std::abs(a - b) < 1e-5; }

TEST_CASE("blas-gemm") {
  float A[] = {1, 2, 3, 4};
  float B[] = {2, 3, 4, 5};
  float C[] = {0, 0, 0, 0};

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1, A, 2, B, 2,
              0, C, 2);
  REQUIRE(float_eq(C[0], 10));
  REQUIRE(float_eq(C[1], 13));
  REQUIRE(float_eq(C[2], 22));
  REQUIRE(float_eq(C[3], 29));
}
TEST_CASE("blas-batch-gemm") {
  const float A1[] = {1, 2, 3, 4};
  const float B1[] = {2, 3, 4, 5};
  float C1[] = {0, 0, 0, 0};

  const float A2[] = {1, 2, 3, 4};
  const float B2[] = {2, 3, 4, 5};
  float C2[] = {0, 0, 0, 0};

  const float* A[] = {A1, A2};
  const float* B[] = {B1, B2};
  float* C[] = {C1, C2};

  BlasInt m[] = {2, 2};
  CBLAS_TRANSPOSE trans[] = {CblasNoTrans, CblasNoTrans};
  static float alpha = 1., beta = 0.;
  BlasInt batch_size = 2;

  cblas_sgemm_batch(CblasRowMajor, trans, trans, m, m, m, &alpha, A, m, B, m,
                    &beta, C, m, 1, &batch_size);
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
  float vec[] = {1, 2};
  cblas_sscal(2, 2, vec, 1);
  REQUIRE(float_eq(vec[0], 2));
  REQUIRE(float_eq(vec[1], 4));
}

TEST_CASE("blas-gemm-no-merge") {
  int m = 1 * 128, k = 12 * 64, n = 12 * 64;
  float* A = new float[m * k];
  float* B = new float[k * n];
  float* C = new float[m * n];
  static constexpr float alpha = 1., beta = 0.;

  auto start = std::chrono::system_clock::now();
  for (int it = 0; it < 100; ++it)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, k,
                B, k, beta, C, n);

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "blas-gemm-no-merge cost:"
            << double(duration.count()) *
                   std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << "sec" << std::endl;
}

TEST_CASE("blas-gemm-merge") {
  int m = 1 * 128, k = 12 * 64, n = 12 * 64 * 3;
  float* A = new float[m * k];
  float* B = new float[k * n];
  float* C = new float[m * n];
  static constexpr float alpha = 1., beta = 0.;

  auto start = std::chrono::system_clock::now();
  for (int it = 0; it < 100; ++it)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, k,
                B, k, beta, C, n);
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "blas-gemm-merge cost:"
            << double(duration.count()) *
                   std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << "sec" << std::endl;
}

TEST_CASE("blas-gemm-intermediate") {
  int m = 1 * 128, k = 12 * 64 * 4, n = 12 * 64;
  float* A = new float[m * k];
  float* B = new float[k * n];
  float* C = new float[m * n];
  static constexpr float alpha = 1., beta = 0.;

  auto start = std::chrono::system_clock::now();
  int step = 100;
  for (int it = 0; it < step; ++it)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, k,
                B, k, beta, C, n);
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double eslape = double(duration.count()) *
                  std::chrono::microseconds::period::num /
                  std::chrono::microseconds::period::den;
  double FLOP = m * n * k * 2. * step / 1e9;
  std::cout << "blas-gemm-intermediate cost:" << eslape << " sec, "
            << FLOP / eslape << " GFLOPS" << std::endl;
}

TEST_CASE("blas-gemm-intermediate-pad") {
  int k_pad = 0;
  int m = 1 * 128, k_param = 12 * 64 * 4, n_param = 12 * 64 * 4;
  int pad_size = 0;
  for (int k_pad = 0; k_pad <= pad_size; k_pad += 4) {
    int k = k_param + k_pad;
    int n = n_param;
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
    static constexpr float alpha = 1., beta = 0.;

    auto start = std::chrono::system_clock::now();
    int step = 100;
    for (int it = 0; it < step; ++it)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, k,
                  B, k, beta, C, n);
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double eslape = double(duration.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den;
    double FLOP = m * n * k_param * 2. * step / 1e9;
    std::cout << "[NoTrans-Trans] blas-gemm-intermediate pad, " << k_pad
              << " cost: " << eslape << " sec, " << FLOP / eslape << " GFLOPS"
              << std::endl;
  }
  std::cout << "<<<<<<<<<<<<<<<<<" << std::endl;
  for (int k_pad = 0; k_pad <= pad_size; k_pad += 4) {
    int k = k_param + k_pad;
    int n = n_param;
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
    static constexpr float alpha = 1., beta = 0.;

    auto start = std::chrono::system_clock::now();
    int step = 100;
    for (int it = 0; it < step; ++it)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A,
                  k, B, n, beta, C, n);
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double eslape = double(duration.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den;
    double FLOP = m * n * k_param * 2. * step / 1e9;
    std::cout << "[NoTrans-NoTrans] blas-gemm-intermediate pad, " << k_pad
              << " cost: " << eslape << " sec, " << FLOP / eslape << " GFLOPS"
              << std::endl;
  }
  std::cout << "<<<<<<<<<<<<<<<<<" << std::endl;
  for (int n_pad = 0; n_pad <= pad_size; n_pad += 4) {
    int k = k_param;
    int n = n_param + n_pad;
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
    static constexpr float alpha = 1., beta = 0.;

    auto start = std::chrono::system_clock::now();
    int step = 100;
    for (int it = 0; it < step; ++it)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A,
                  k, B, n, beta, C, n);
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double eslape = double(duration.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den;
    double FLOP = m * n * k_param * 2. * step / 1e9;
    std::cout << "[NoTrans-NoTrans] blas-gemm-intermediate n_pad, " << n_pad
              << " cost: " << eslape << " sec, " << FLOP / eslape << " GFLOPS"
              << std::endl;
  }
  std::cout << "<<<<<<<<<<<<<<<<<" << std::endl;
  for (int n_pad = 0; n_pad <= pad_size; n_pad += 4) {
    int k = k_param;
    int n = n_param + n_pad;
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
    static constexpr float alpha = 1., beta = 0.;

    auto start = std::chrono::system_clock::now();
    int step = 100;
    for (int it = 0; it < step; ++it)
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, A, m,
                  B, n, beta, C, n);
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double eslape = double(duration.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den;
    double FLOP = m * n * k_param * 2. * step / 1e9;
    std::cout << "[Trans-NoTrans] blas-gemm-intermediate n_pad, " << n_pad
              << " cost: " << eslape << " sec, " << FLOP / eslape << " GFLOPS"
              << std::endl;
  }
}

}  // namespace core
}  // namespace fast_transformers
