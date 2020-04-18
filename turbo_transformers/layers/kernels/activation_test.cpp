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

#include "turbo_transformers/layers/kernels/activation.h"

#include "loguru.hpp"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/common.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
template <typename T, typename Func>
static void ActivationTestHelper(int batch_size, int seq_length,
                                 int hidden_size, const Func& func) {
  core::Tensor gpu_bias(nullptr), cpu_bias(nullptr), gpu_out(nullptr),
      cpu_out(nullptr);
  std::tie(cpu_bias, gpu_bias) =
      common::CreateAndFillRandomForCPUGPUTensors<T>({hidden_size});
  std::tie(cpu_out, gpu_out) = common::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, seq_length, hidden_size});
  func(cpu_bias, cpu_out, gpu_bias, gpu_out);
}

TEST_CASE("activation-gpu-test") {
  for (auto hidden_size : {500, 12 * 64, 4096 * 2 + 1}) {
    for (auto batch_size : {1, 5}) {
      for (auto seq_length : {8, 120}) {
        ActivationTestHelper<float>(
            batch_size, seq_length, hidden_size,
            [](core::Tensor& cpu_bias, core::Tensor& cpu_out,
               core::Tensor& gpu_bias, core::Tensor& gpu_out) {
              AddBiasAct<float, ActivationType::Gelu>(cpu_bias, &cpu_out);
              AddBiasAct<float, ActivationType::Gelu>(gpu_bias, &gpu_out);
              REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
            });

        ActivationTestHelper<float>(
            batch_size, seq_length, hidden_size,
            [](core::Tensor& cpu_bias, core::Tensor& cpu_out,
               core::Tensor& gpu_bias, core::Tensor& gpu_out) {
              AddBiasAct<float, ActivationType::Tanh>(cpu_bias, &cpu_out);
              AddBiasAct<float, ActivationType::Tanh>(gpu_bias, &gpu_out);
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
