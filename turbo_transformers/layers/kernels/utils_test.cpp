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
#include "turbo_transformers/layers/kernels/utils.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/common.h"
#endif
#include "catch2/catch.hpp"

namespace turbo_transformers {
namespace layers {
namespace kernels {

TEST_CASE("cpu-concate", "test1") {
  turbo_transformers::core::Tensor t1(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 1, 2, 2}));
  turbo_transformers::core::Tensor t2(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 1, 3, 2}));
  for (int i = 0; i < t1.numel(); ++i) {
    t1.mutableData<float>()[i] = i * 1.0;
  }

  for (int i = 0; i < t2.numel(); ++i) {
    t2.mutableData<float>()[i] = i * 100.0;
  }
  turbo_transformers::core::Tensor res1(nullptr), res2(nullptr), res3(nullptr);
  Concat<float>(t1, t2, 2, &res1);
  REQUIRE(res1.n_dim() == 4);
  REQUIRE(res1.numel() == 2 * 1 * 5 * 2);

  turbo_transformers::core::Tensor t3(
      turbo_transformers::core::NewDLPackTensorT<float>({4, 2}));
  turbo_transformers::core::Tensor t4(
      turbo_transformers::core::NewDLPackTensorT<float>({3, 2}));
  Concat<float>(t3, t4, 0, &res2);
  REQUIRE(res2.n_dim() == 2);
  REQUIRE(res2.numel() == 7 * 2);

  turbo_transformers::core::Tensor t5(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 3}));
  turbo_transformers::core::Tensor t6(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 4}));
  for (int i = 0; i < t5.numel(); ++i) {
    t5.mutableData<float>()[i] = i * 1.0;
  }

  for (int i = 0; i < t6.numel(); ++i) {
    t6.mutableData<float>()[i] = i * 100.0;
  }
  Concat<float>(t5, t6, 1, &res3);
  REQUIRE(res3.n_dim() == 2);
  REQUIRE(res3.numel() == 2 * 7);

  // t5.Print<float>(std::cerr);
  // t6.Print<float>(std::cerr);
  // res3.Print<float>(std::cerr);
  // REQUIRE(test_tensor.n_dim() == 2);
  // REQUIRE(test_tensor.numel() == 3 * 4);
}

#ifdef TT_WITH_CUDA
template <typename T, typename Func>
static void AddBiasTestHelper(int batch_size, int seq_length,
                              int hidden_size, const Func& func) {
  core::Tensor gpu_bias(nullptr), cpu_bias(nullptr), gpu_out(nullptr),
      cpu_out(nullptr);
  std::tie(cpu_bias, gpu_bias) =
      common::CreateAndFillRandomForCPUGPUTensors<T>({hidden_size});
  std::tie(cpu_out, gpu_out) = common::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, seq_length, hidden_size});
  func(cpu_bias, cpu_out, gpu_bias, gpu_out);
}

TEST_CASE("addbias-gpu-test") {
  for (auto hidden_size : {512, 12 * 64, 4096 * 2 + 1}) {
    for (auto batch_size : {1, 5}) {
      for (auto seq_length : {8, 120}) {
        AddBiasTestHelper<float>(
            batch_size, seq_length, hidden_size,
            [](core::Tensor& cpu_bias, core::Tensor& cpu_out,
               core::Tensor& gpu_bias, core::Tensor& gpu_out) {
              AddBias(cpu_bias, &cpu_out);
              AddBias(gpu_bias, &gpu_out);
              REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
            });
      }  // for
    }
  }
}

template <typename T, typename Func>
static void AddInputBiasTestHelper(int batch_size, int seq_length,
                                   int hidden_size, const Func& func) {
  core::Tensor gpu_input1(nullptr), cpu_input1(nullptr), gpu_input2(nullptr), cpu_input2(nullptr),
               gpu_bias(nullptr), cpu_bias(nullptr), gpu_out(nullptr), cpu_out(nullptr);
  std::tie(cpu_input1, gpu_input1) = common::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, seq_length, hidden_size});
  std::tie(cpu_input2, gpu_input2) = common::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, seq_length, hidden_size});
  std::tie(cpu_bias, gpu_bias) =
      common::CreateAndFillRandomForCPUGPUTensors<T>({hidden_size});
  std::tie(cpu_out, gpu_out) = common::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, seq_length, hidden_size});
  func(cpu_input1, cpu_input2, cpu_bias, cpu_out, gpu_input1, gpu_input2, gpu_bias, gpu_out);
}

TEST_CASE("addinputbias-gpu-test") {
  for (auto hidden_size : {512, 12 * 64, 4096 * 2 + 1}) {
    for (auto batch_size : {1, 5}) {
      for (auto seq_length : {8, 120}) {
        AddInputBiasTestHelper<float>(
            batch_size, seq_length, hidden_size,
            [](core::Tensor& cpu_input1, core::Tensor& cpu_input2,
               core::Tensor& cpu_bias, core::Tensor& cpu_out,
               core::Tensor& gpu_input1, core::Tensor& gpu_input2,
               core::Tensor& gpu_bias, core::Tensor& gpu_out) {
              AddInputBias(cpu_input1, cpu_input2, cpu_bias, &cpu_out);
              AddInputBias(gpu_input1, gpu_input2, gpu_bias, &gpu_out);
              REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
            });
      }  // for
    }
  }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
