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
#include "turbo_transformers/layers/kernels/mat_mul.h"

#include <chrono>
#include <cstdlib>
#include <ctime>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/kernels/common.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

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

#ifdef TT_WITH_CUDA
void check_cpu_gpu_res(bool isTransB) {
  const std::vector<int64_t> m_list{5, 10, 15, 20};
  const std::vector<int64_t> n_list{12 * 64, 12 * 64 * 4};
  const std::vector<int64_t> k_list{12 * 64, 12 * 64 * 4};
  ;

  for (auto m : m_list) {
    for (auto n : n_list) {
      for (auto k : k_list) {
        std::initializer_list<int64_t> input_shape{m, k};
        std::initializer_list<int64_t> weight_shape{k, n};
        std::initializer_list<int64_t> weight_shape_trans{n, k};
        std::initializer_list<int64_t> output_shape{m, n};
        using turbo_transformers::core::NewDLPackTensorT;

        core::Tensor cpu_input_tensor(nullptr), gpu_input_tensor(nullptr);
        std::tie(cpu_input_tensor, gpu_input_tensor) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(input_shape);

        core::Tensor cpu_weight_tensor(nullptr), gpu_weight_tensor(nullptr);
        std::tie(cpu_weight_tensor, gpu_weight_tensor) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                isTransB ? weight_shape_trans : weight_shape);

        core::Tensor cpu_output_tensor(nullptr), gpu_output_tensor(nullptr);
        std::tie(cpu_output_tensor, gpu_output_tensor) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(output_shape);

        layers::kernels::MatMul(cpu_input_tensor, false, cpu_weight_tensor,
                                isTransB, 1.0, &cpu_output_tensor, 0.0);

        layers::kernels::MatMul(gpu_input_tensor, false, gpu_weight_tensor,
                                isTransB, 1.0, &gpu_output_tensor, 0.0);

        common::CheckResultOfCPUAndGPU<float>(cpu_output_tensor,
                                              gpu_output_tensor);
      }
    }
  }
}

TEST_CASE("matmul-gpu-test") {
  check_cpu_gpu_res(true);
  check_cpu_gpu_res(false);
}
#endif
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
