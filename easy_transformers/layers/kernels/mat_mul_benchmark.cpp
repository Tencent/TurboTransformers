// Copyright 2020 Tencent
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define CATCH_CONFIG_MAIN
#include <chrono>
#include <cstdlib>
#include <ctime>

#include "catch2/catch.hpp"
#include "easy_transformers/core/tensor.h"
#include "easy_transformers/layers/kernels/mat_mul.h"
#include "easy_transformers/layers/kernels/test_helper.h"

namespace easy_transformers {
namespace core {

using ::easy_transformers::test::Fill;

inline void _CreateBenchmark(DLDeviceType device_type, bool trans_weight,
                             std::initializer_list<int64_t> weight_shape,
                             std::vector<int64_t> m_list) {
  const int step = 1000;
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
    using easy_transformers::core::NewDLPackTensorT;

    core::Tensor input_tensor(
        NewDLPackTensorT<float>(input_shape, device_type, 0));
    Fill<float>(input_tensor);

    core::Tensor weight_tensor(
        NewDLPackTensorT<float>(weight_shape, device_type, 0));
    Fill<float>(weight_tensor);

    core::Tensor output_tensor(
        NewDLPackTensorT<float>(output_shape, device_type, 0));
    Fill<float>(output_tensor);

    layers::kernels::MatMul(input_tensor, false, weight_tensor, trans_weight,
                            1.0, &output_tensor, 0.0);
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < step; ++i) {
      layers::kernels::MatMul(input_tensor, false, weight_tensor, trans_weight,
                              1.0, &output_tensor, 0.0);
    }
#ifdef FT_WITH_CUDA
    if (device_type == kDLGPU) cudaDeviceSynchronize();
#endif
    auto end = std::chrono::system_clock::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto elapse = double(duration.count()) *
                  std::chrono::microseconds::period::num /
                  std::chrono::microseconds::period::den / step;
    std::cout << m << "," << n << "," << k << ", mat_mul " << device_name
              << ", " << trans_name << " :" << elapse << " s, "
              << 2. * m * n * k / 1e9 / elapse << " GFlops" << std::endl;
  }
}

TEST_CASE("MatMul CPU benchmark") {
  int64_t k = 12 * 64, n = 12 * 64 * 3;
  std::vector<int64_t> m_list{10, 20, 40, 60, 80, 100, 120};
  _CreateBenchmark(kDLCPU, false, {k, n}, m_list);
  std::cout << std::endl;
}

TEST_CASE("Attention QKV MatMul GPU benchmark") {
  int64_t k = 12 * 64, n = 12 * 64 * 3;
  std::vector<int64_t> m_list{10, 20, 40, 60, 80, 100, 120};
  std::cout << "weight no trans" << std::endl;
  _CreateBenchmark(kDLGPU, false, {k, n}, m_list);
  std::cout << "weight trans" << std::endl;
  _CreateBenchmark(kDLGPU, true, {n, k}, m_list);

  std::cout << "batch = 20" << std::endl;
  for (auto& m : m_list) m *= 20;
  std::cout << "weight no trans" << std::endl;
  _CreateBenchmark(kDLGPU, false, {k, n}, m_list);
  std::cout << "weight trans" << std::endl;
  _CreateBenchmark(kDLGPU, true, {n, k}, m_list);
}

TEST_CASE("Attention Intermediate GPU benchmark") {
  std::cout << "Intermediate Layer Benchmark" << std::endl;
  int64_t k = 12 * 64, n = 12 * 64;
  DLDeviceType device_type = kDLGPU;
  std::vector<int64_t> m_list{10, 20, 40, 60, 80, 100, 120};
  std::cout << "weight trans" << std::endl;
  _CreateBenchmark(device_type, true, {n, k}, m_list);
  std::cout << "weight no trans" << std::endl;
  _CreateBenchmark(device_type, false, {k, n}, m_list);

  std::cout << "batch = 20" << std::endl;
  for (auto& m : m_list) m *= 20;
  std::cout << "weight trans" << std::endl;
  _CreateBenchmark(device_type, true, {n, k}, m_list);
  std::cout << "weight no trans" << std::endl;
  _CreateBenchmark(device_type, false, {k, n}, m_list);
}

}  // namespace core
}  // namespace easy_transformers
