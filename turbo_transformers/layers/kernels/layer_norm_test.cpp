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

#include "turbo_transformers/layers/kernels/layer_norm.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/layers/kernels/common.h"

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
TEST_CASE("add_bias_layer_norm-test") {
  std::vector<int64_t> hidden_size_list{12 * 64, 2000};
  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};
  for (auto hidden_size : hidden_size_list)
    for (auto batch_size : batch_size_list)
      for (auto seq_length : seq_length_list) {
        core::Tensor gpu_input(nullptr), cpu_input(nullptr), gpu_bias(nullptr),
            cpu_bias(nullptr), gpu_out(nullptr), cpu_out(nullptr),
            gpu_gamma(nullptr), cpu_gamma(nullptr), gpu_beta(nullptr),
            cpu_beta(nullptr);
        std::tie(cpu_input, gpu_input) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, seq_length, hidden_size});
        std::tie(cpu_bias, gpu_bias) =
            common::CreateAndFillRandomForCPUGPUTensors<float>({hidden_size});
        std::tie(cpu_out, gpu_out) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, seq_length, hidden_size});
        std::tie(cpu_gamma, gpu_gamma) =
            common::CreateAndFillRandomForCPUGPUTensors<float>({hidden_size});
        std::tie(cpu_beta, gpu_beta) =
            common::CreateAndFillRandomForCPUGPUTensors<float>({hidden_size});

        {
          LayerNorm<float>(cpu_gamma, cpu_beta, &cpu_out);
          LayerNorm<float>(gpu_gamma, gpu_beta, &gpu_out);
        }
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));

        {
          AddBiasLayerNorm<float>(cpu_input, cpu_bias, cpu_gamma, cpu_beta,
                                  &cpu_out);
          AddBiasLayerNorm<float>(gpu_input, gpu_bias, gpu_gamma, gpu_beta,
                                  &gpu_out);
        }
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
      }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
