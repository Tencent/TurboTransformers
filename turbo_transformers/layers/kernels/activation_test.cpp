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

#include "turbo_transformers/layers/kernels/activation.h"

#include "loguru.hpp"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/test_helper.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {
template <typename T>
void AddBiasGeluActNaive(const T* bias, T* out, int64_t m, int64_t n) {
  for (int64_t i = 0; i < m; ++i) {
    int64_t k = 0;
    for (int64_t j = n * i; j < n * (i + 1); ++j) {
      auto before_act =
          static_cast<float>(out[j]) + static_cast<float>(bias[k++]);
      out[j] = static_cast<T>(
          before_act * 0.5f *
          (1.0f + std::tanh(0.7978845608028654f *
                            (before_act + 0.044715f * before_act * before_act *
                                              before_act))));
    }
  }
}

template <typename T>
void AddBiasTanhActNaive(const T* bias, T* out, int64_t m, int64_t n) {
  for (int64_t i = 0; i < m; ++i) {
    int64_t k = 0;
    for (int64_t j = n * i; j < n * (i + 1); ++j) {
      auto before_act =
          static_cast<float>(out[j]) + static_cast<float>(bias[k++]);
      out[j] = static_cast<T>(std::tanh(before_act));
    }
  }
}

template <typename T>
void TestActResultAndSpeed(
    const std::function<void(const int step, int m, int64_t n,
                             const core::Tensor& bias, core::Tensor& out,
                             core::Tensor& out_parallel)>& func) {
  int64_t hidden_size = 12 * 64;
  const int step = 10;
  for (auto batch_size : {1, 20, 24}) {
    for (auto seq_length : {8, 16, 32, 48, 64, 128}) {
      auto m = batch_size * seq_length;
      auto n = hidden_size;

      auto bias = test::CreateTensorAndFillConstant<T>({n}, kDLCPU, 0, 0.01f);
      auto out = test::CreateTensorAndFillConstant<T>({m, n}, kDLCPU, 0, 0.02f);
      auto out_parallel =
          test::CreateTensorAndFillConstant<T>({m, n}, kDLCPU, 0, 0.02f);

      std::cout << "batch_size: " << batch_size
                << " seq_length: " << seq_length;
      func(step, m, n, bias, out, out_parallel);
      if (!test::CheckResultOfCPU<T>(out, out_parallel)) {
        TT_THROW("AddBiasGelu test failed");
      }
    }
  }
}

template <typename T>
void TestGelu(const int step, int m, int64_t n, const core::Tensor& bias,
              core::Tensor& out, core::Tensor& out_parallel) {
  test::TestFuncSpeed(
      [&]() {
        AddBiasGeluActNaive<T>(bias.data<T>(), out_parallel.mutableData<T>(), m,
                               n);
      },
      step, "AddBiasGeluActNaive", m * n * sizeof(T) / 1e9);

  test::TestFuncSpeed(
      [&]() { AddBiasAct<T, ActivationType::Gelu>(bias, &out); }, step,
      "AddBiasGeluAct OMP", m * n * sizeof(T) / 1e9);
}

template <typename T>
void TestTanh(const int step, int m, int64_t n, const core::Tensor& bias,
              core::Tensor& out, core::Tensor& out_parallel) {
  test::TestFuncSpeed(
      [&]() {
        AddBiasTanhActNaive<float>(bias.data<float>(),
                                   out_parallel.mutableData<float>(), m, n);
      },
      step, "AddBiasTanhActNaive", m * n * sizeof(float) / 1e9);

  test::TestFuncSpeed(
      [&]() { AddBiasAct<float, ActivationType::Tanh>(bias, &out); }, step,
      "AddBiasTanhAct OMP", m * n * sizeof(float) / 1e9);
}

TEST_CASE("activation CPU AddBiasGelu and AddBiasTanh benchmark") {
  TestActResultAndSpeed<float>(TestGelu<float>);
  TestActResultAndSpeed<float>(TestTanh<float>);
}

#ifdef TT_WITH_CUDA
template <typename T, typename Func>
void CheckResultOfGPUAndCPU(int batch_size, int seq_length, int hidden_size,
                            const Func& func) {
  core::Tensor gpu_bias(nullptr), cpu_bias(nullptr), gpu_out(nullptr),
      cpu_out(nullptr);
  std::tie(cpu_bias, gpu_bias) =
      test::CreateAndFillRandomForCPUGPUTensors<T>({hidden_size});
  std::tie(cpu_out, gpu_out) = test::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, seq_length, hidden_size});
  func(cpu_bias, cpu_out, gpu_bias, gpu_out);
}

TEST_CASE("activation-gpu-test") {
  for (auto hidden_size : {500, 12 * 64, 1000, 2000, 4096 * 2 + 1}) {
    for (auto batch_size : {1, 20, 24}) {
      for (auto seq_length : {8, 16, 32, 48, 64, 128}) {
        std::cout << "batch_size: " << batch_size
                  << " seq_length: " << seq_length
                  << " hidden_size: " << hidden_size;
        CheckResultOfGPUAndCPU<float>(
            batch_size, seq_length, hidden_size,
            [](core::Tensor& cpu_bias, core::Tensor& cpu_out,
               core::Tensor& gpu_bias, core::Tensor& gpu_out) {
              AddBiasAct<float, ActivationType::Gelu>(cpu_bias, &cpu_out);
              AddBiasAct<float, ActivationType::Gelu>(gpu_bias, &gpu_out);
              REQUIRE(test::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
            });

        CheckResultOfGPUAndCPU<float>(
            batch_size, seq_length, hidden_size,
            [](core::Tensor& cpu_bias, core::Tensor& cpu_out,
               core::Tensor& gpu_bias, core::Tensor& gpu_out) {
              AddBiasAct<float, ActivationType::Tanh>(cpu_bias, &cpu_out);
              AddBiasAct<float, ActivationType::Tanh>(gpu_bias, &gpu_out);
              REQUIRE(test::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
            });
        std::cout << " PASSED" << std::endl;
      }  // for
    }
  }
}

#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
