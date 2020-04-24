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
#include <cstdlib>
#include <ctime>
#include "benchmark_help.h"
#include "catch2/catch.hpp"
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

using layers::kernels::common::FillRandom;

static void MatmulBenchmarkHelper(DLDeviceType device_type, bool trans_weight,
                                  std::initializer_list<int64_t> weight_shape,
                                  std::vector<int64_t> m_list) {
  constexpr int n_step = 1000;
  const std::string device_name = device_type == kDLCPU ? "CPU" : "GPU";
  const std::string trans_name = trans_weight ? "Tran" : "NoTrans";

  int64_t k, n;
  std::vector<int64_t> weight_shape_vec(weight_shape);
  if (trans_weight) {
    n = weight_shape_vec[0];
    k = weight_shape_vec[1];
  } else {
    k = weight_shape_vec[0];
    n = weight_shape_vec[1];
  }
  for (auto m : m_list) {
    std::initializer_list<int64_t> input_shape{m, k};
    std::initializer_list<int64_t> output_shape{m, n};
    using turbo_transformers::core::NewDLPackTensorT;

    core::Tensor input_tensor(
        NewDLPackTensorT<float>(input_shape, device_type, 0));
    FillRandom<float>(input_tensor);

    core::Tensor weight_tensor(
        NewDLPackTensorT<float>(weight_shape, device_type, 0));
    FillRandom<float>(weight_tensor);

    core::Tensor output_tensor(
        NewDLPackTensorT<float>(output_shape, device_type, 0));
    FillRandom<float>(output_tensor);

    std::stringstream ss;
    ss << device_name << " " << trans_name << " MatMul " << m << ", " << k
       << ", " << n << " ";
    auto g_flops = m * n * k * 2 / 1e9;

    if (device_type == kDLGPU) {
#ifdef TT_WITH_CUDA
      auto flops = benchmark::TestFuncSpeed(
          [&]() {
            layers::kernels::MatMul(input_tensor, false, weight_tensor,
                                    trans_weight, 1.0, &output_tensor, 0.0);
          },
          n_step, ss.str(), g_flops, device_type);

      std::cout << ss.str() << " flops: " << flops << std::endl;
#endif
    } else {
      auto flops = benchmark::TestFuncSpeed(
          [&]() {
            layers::kernels::MatMul(input_tensor, false, weight_tensor,
                                    trans_weight, 1.0, &output_tensor, 0.0);
          },
          n_step, ss.str(), g_flops, device_type);
      std::cout << ss.str() << " flops: " << flops << std::endl;
    }
  }
}

#ifdef TT_WITH_CUDA

TEST_CASE("matmal-gpu-gemm7-benchmark") {
  std::cout << "=================================" << std::endl;
  std::cout << "GPU gemm7 Benchmark" << std::endl;
  int64_t k = 12 * 64 * 4, n = 12 * 64;
  DLDeviceType device_type = kDLGPU;
  std::vector<int64_t> m_list{10, 20, 40, 60, 80, 100, 120};
  std::cout << "weight trans" << std::endl;
  MatmulBenchmarkHelper(device_type, true, {n, k}, m_list);
  std::cout << "weight no trans" << std::endl;
  MatmulBenchmarkHelper(device_type, false, {k, n}, m_list);

  std::cout << "batch = 20" << std::endl;
  for (auto& m : m_list) m *= 20;
  std::cout << "weight trans" << std::endl;
  MatmulBenchmarkHelper(device_type, true, {n, k}, m_list);
  std::cout << "weight no trans" << std::endl;
  MatmulBenchmarkHelper(device_type, false, {k, n}, m_list);
}

TEST_CASE("matmal-gpu-gemm6-benchmark") {
  std::cout << "=================================" << std::endl;
  std::cout << "GPU gemm6 Benchmark" << std::endl;
  int64_t k = 12 * 64, n = 12 * 64 * 4;
  DLDeviceType device_type = kDLGPU;
  std::vector<int64_t> m_list{10, 20, 40, 60, 80, 100, 120};
  std::cout << "weight trans" << std::endl;
  MatmulBenchmarkHelper(device_type, true, {n, k}, m_list);
  std::cout << "weight no trans" << std::endl;
  MatmulBenchmarkHelper(device_type, false, {k, n}, m_list);

  std::cout << "batch = 20" << std::endl;
  for (auto& m : m_list) m *= 20;
  std::cout << "weight trans" << std::endl;
  MatmulBenchmarkHelper(device_type, true, {n, k}, m_list);
  std::cout << "weight no trans" << std::endl;
  MatmulBenchmarkHelper(device_type, false, {k, n}, m_list);
}

TEST_CASE("matmal-gpu-fused-gemm-benchmark") {
  std::cout << "=================================" << std::endl;
  std::cout << "GPU fused-gemm Benchmark" << std::endl;
  int64_t k = 12 * 64, n = 12 * 64 * 3;
  std::vector<int64_t> m_list{10, 20, 40, 60, 80, 100, 120};
  std::cout << "weight trans" << std::endl;
  MatmulBenchmarkHelper(kDLGPU, true, {n, k}, m_list);

  std::cout << "weight no trans" << std::endl;
  MatmulBenchmarkHelper(kDLGPU, false, {k, n}, m_list);

  for (auto& m : m_list) m *= 20;
  std::cout << "weight trans" << std::endl;
  MatmulBenchmarkHelper(kDLGPU, true, {n, k}, m_list);
  std::cout << "weight no trans" << std::endl;
  MatmulBenchmarkHelper(kDLGPU, false, {k, n}, m_list);
}

#endif

TEST_CASE("matmal-cpu-sqr-benchmark") {
  DLDeviceType device_type = kDLCPU;
  std::vector<int64_t> rows{1, 100, 1000};
  std::vector<int64_t> depths{1, 100, 1000};
  std::vector<int64_t> cols{1, 100, 1000};

  std::vector<std::tuple<int64_t, int64_t, int64_t>> benchmark_gemms;
  benchmark_gemms.emplace_back(10, 10, 10);
  benchmark_gemms.emplace_back(20, 20, 20);
  benchmark_gemms.emplace_back(30, 30, 30);
  benchmark_gemms.emplace_back(40, 40, 40);
  benchmark_gemms.emplace_back(50, 50, 50);
  benchmark_gemms.emplace_back(60, 60, 60);
  benchmark_gemms.emplace_back(64, 256, 147);
  benchmark_gemms.emplace_back(100, 100, 1);
  benchmark_gemms.emplace_back(100, 100, 100);
  benchmark_gemms.emplace_back(100, 1000, 100);
  benchmark_gemms.emplace_back(1000, 1000, 1);
  benchmark_gemms.emplace_back(1000, 1000, 10);
  benchmark_gemms.emplace_back(1000, 1000, 100);
  benchmark_gemms.emplace_back(1000, 1000, 1000);

  benchmark_gemms.emplace_back(10, 12 * 64, 12 * 64 * 4);
  benchmark_gemms.emplace_back(20, 12 * 64, 12 * 64 * 4);
  benchmark_gemms.emplace_back(40, 12 * 64, 12 * 64 * 4);
  benchmark_gemms.emplace_back(80, 12 * 64, 12 * 64 * 4);
  benchmark_gemms.emplace_back(100, 12 * 64, 12 * 64 * 4);
  benchmark_gemms.emplace_back(200, 12 * 64, 12 * 64 * 4);
  benchmark_gemms.emplace_back(300, 12 * 64, 12 * 64 * 4);

  benchmark_gemms.emplace_back(10, 12 * 64 * 4, 12 * 64);
  benchmark_gemms.emplace_back(20, 12 * 64 * 4, 12 * 64);
  benchmark_gemms.emplace_back(40, 12 * 64 * 4, 12 * 64);
  benchmark_gemms.emplace_back(80, 12 * 64 * 4, 12 * 64);
  benchmark_gemms.emplace_back(100, 12 * 64 * 4, 12 * 64);
  benchmark_gemms.emplace_back(200, 12 * 64 * 4, 12 * 64);
  benchmark_gemms.emplace_back(300, 12 * 64 * 4, 12 * 64);

  for (size_t i = 0; i < benchmark_gemms.size(); ++i) {
    auto benchmark_gemm = benchmark_gemms[i];
    int64_t depth, col, row;
    std::tie(row, depth, col) = benchmark_gemm;
    // std::cout << "weight trans" << std::endl;
    MatmulBenchmarkHelper(device_type, true, {depth, col}, {row});
    // std::cout << "weight no trans" << std::endl;
    // MatmulBenchmarkHelper(device_type, false, {col, depth}, {row});
  }
}
/*
TEST_CASE("matmal-cpu-benchmark") {
  std::cout << "=================================" << std::endl;
  std::cout << "CPU QKV MatMul Benchmark" << std::endl;
  int64_t k = 12 * 64, n = 12 * 64 * 3;
  std::vector<int64_t> m_list{10, 20, 40, 60, 80, 100, 120};
  MatmulBenchmarkHelper(kDLCPU, false, {k, n}, m_list);
  std::cout << std::endl;
}
*/

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
