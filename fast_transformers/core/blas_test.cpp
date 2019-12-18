#define CATCH_CONFIG_MAIN
#include "fast_transformers/core/blas.h"

#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <ctime>

#include "catch2/catch.hpp"
#include "fast_transformers/core/tensor.h"
#include "fast_transformers/layers/kernels/mat_mul.h"

#ifdef WITH_CUDA
#include "fast_transformers/core/memory.h"
#include "fast_transformers/core/nvcommon.h"
#endif

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

inline void RandomMatrix(float* m, const int mSize, float LO = 0.,
                         float HI = 1.) {
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < mSize; i++)
    m[i] = LO + static_cast<float>(rand()) /
                    (static_cast<float>(RAND_MAX / (HI - LO)));
}

#ifdef WITH_CUDA
template <typename T>
inline void FillCPUGPU(Tensor& cpu_tensor, Tensor& gpu_tensor) {
  T* gpu_data = gpu_tensor.mutableData<T>();
  T* cpu_data = cpu_tensor.mutableData<T>();
  auto size = cpu_tensor.numel();
  srand((unsigned)time(NULL));
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = rand() / static_cast<T>(RAND_MAX);
  }
  fast_transformers::core::FT_Memcpy<T>(gpu_data, cpu_data, size,
                                        fast_transformers::core::FT_CPU2GPU);
}

template <typename T>
inline bool CompareCPUGPU(const Tensor& cpu_tensor, const Tensor& gpu_tensor) {
  const T* gpu_data = gpu_tensor.data<T>();
  const T* cpu_data = cpu_tensor.data<T>();
  auto size = cpu_tensor.numel();

  T* gpu_data_ref = new T[size];
  fast_transformers::core::FT_Memcpy<T>(gpu_data_ref, gpu_data, size,
                                        fast_transformers::core::FT_GPU2CPU);
  bool ret = true;
  double sum1 = 0., sum2 = 0.;
  for (int64_t i = 0; i < size; ++i) {
    if (fabs(gpu_data_ref[i] - cpu_data[i]) > 1e-3) {
      ret = false;
      break;
    }
    sum1 += gpu_data_ref[i];
    sum2 += cpu_data[i];
  }
  std::cout << "sum1: " << sum1 << ", sum2: " << sum2 << std::endl;
  delete[] gpu_data_ref;
  return ret;
}

TEST_CASE("check cpu and gpu correctness Trans Notrans") {
  int64_t k, n;
  std::vector<int64_t> test_list{5, 10, 15, 20};
  for (auto m : test_list) {
    k = 12 * 64, n = 12 * 64 * 3;
    std::initializer_list<int64_t> input_shape{m, k};
    std::initializer_list<int64_t> weight_shape{k, n};
    std::initializer_list<int64_t> output_shape{m, n};
    using fast_transformers::core::NewDLPackTensorT;

    core::Tensor cpu_input_tensor(
        NewDLPackTensorT<float>(input_shape, kDLCPU, 0));
    core::Tensor gpu_input_tensor(
        NewDLPackTensorT<float>(input_shape, kDLGPU, 0));

    FillCPUGPU<float>(cpu_input_tensor, gpu_input_tensor);

    core::Tensor cpu_weight_tensor(
        NewDLPackTensorT<float>(weight_shape, kDLCPU, 0));
    core::Tensor gpu_weight_tensor(
        NewDLPackTensorT<float>(weight_shape, kDLGPU, 0));
    FillCPUGPU<float>(cpu_weight_tensor, gpu_weight_tensor);

    core::Tensor cpu_output_tensor(
        NewDLPackTensorT<float>(output_shape, kDLCPU, 0));
    core::Tensor gpu_output_tensor(
        NewDLPackTensorT<float>(output_shape, kDLGPU, 0));
    FillCPUGPU<float>(cpu_output_tensor, gpu_output_tensor);

    layers::kernels::MatMul(cpu_input_tensor, false, cpu_weight_tensor, false,
                            1.0, &cpu_output_tensor, 0.0);

    layers::kernels::MatMul(gpu_input_tensor, false, gpu_weight_tensor, false,
                            1.0, &gpu_output_tensor, 0.0);

    CompareCPUGPU<float>(cpu_output_tensor, gpu_output_tensor);
  }
}

TEST_CASE("check cpu and gpu correctness NoTrans Trans") {
  int64_t k, n;
  std::vector<int64_t> test_list{5, 10, 15, 20};
  for (auto m : test_list) {
    k = 12 * 64, n = 12 * 64 * 3;
    std::initializer_list<int64_t> input_shape{m, k};
    std::initializer_list<int64_t> weight_shape{k, n};
    std::initializer_list<int64_t> output_shape{m, n};
    using fast_transformers::core::NewDLPackTensorT;

    core::Tensor cpu_input_tensor(
        NewDLPackTensorT<float>(input_shape, kDLCPU, 0));
    core::Tensor gpu_input_tensor(
        NewDLPackTensorT<float>(input_shape, kDLGPU, 0));

    FillCPUGPU<float>(cpu_input_tensor, gpu_input_tensor);

    core::Tensor cpu_weight_tensor(
        NewDLPackTensorT<float>(weight_shape, kDLCPU, 0));
    core::Tensor gpu_weight_tensor(
        NewDLPackTensorT<float>(weight_shape, kDLGPU, 0));
    FillCPUGPU<float>(cpu_weight_tensor, gpu_weight_tensor);

    core::Tensor cpu_output_tensor(
        NewDLPackTensorT<float>(output_shape, kDLCPU, 0));
    core::Tensor gpu_output_tensor(
        NewDLPackTensorT<float>(output_shape, kDLGPU, 0));
    FillCPUGPU<float>(cpu_output_tensor, gpu_output_tensor);

    layers::kernels::MatMul(cpu_input_tensor, false, cpu_weight_tensor, true,
                            1.0, &cpu_output_tensor, 0.0);

    layers::kernels::MatMul(gpu_input_tensor, false, gpu_weight_tensor, true,
                            1.0, &gpu_output_tensor, 0.0);

    CompareCPUGPU<float>(cpu_output_tensor, gpu_output_tensor);
  }
}

#endif

}  // namespace core
}  // namespace fast_transformers
