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

#include "benchmark_help.h"
#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/kernels/activation.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#include <string>

#include "turbo_transformers/layers/kernels/common.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

TEST_CASE("activation-benchmark") {
  const int n_step = 100;
  int64_t hidden_size = 12 * 64;
  for (auto batch_size : {1, 20, 24}) {
    for (auto seq_length : {8, 16, 32, 48, 64, 128}) {
      auto m = batch_size * seq_length;
      auto n = hidden_size;

      auto bias =
          common::CreateTensorAndFillConstant<float>({n}, kDLCPU, 0, 0.01f);
      auto out =
          common::CreateTensorAndFillConstant<float>({m, n}, kDLCPU, 0, 0.02f);
      benchmark::TestFuncSpeed(
          [&]() { AddBiasAct<float, ActivationType::Gelu>(bias, &out); },
          n_step,
          "CPU Gelu Activation " + std::to_string(m) + "," + std::to_string(n),
          m * n * sizeof(float) / 1e9, kDLCPU);

      benchmark::TestFuncSpeed(
          [&]() { AddBiasAct<float, ActivationType::Tanh>(bias, &out); },
          n_step,
          "CPU Tanh Activation " + std::to_string(m) + "," + std::to_string(n),
          m * n * sizeof(float) / 1e9, kDLCPU);
#ifdef TT_WITH_CUDA
      auto bias_gpu =
          common::CreateTensorAndFillConstant<float>({n}, kDLGPU, 0, 0.01f);
      auto out_gpu =
          common::CreateTensorAndFillConstant<float>({m, n}, kDLGPU, 0, 0.02f);
      benchmark::TestFuncSpeed(
          [&]() {
            AddBiasAct<float, ActivationType::Gelu>(bias_gpu, &out_gpu);
          },
          n_step,
          "GPU Gelu Activation " + std::to_string(m) + "," + std::to_string(n),
          m * n * sizeof(float) / 1e9, kDLGPU);

      benchmark::TestFuncSpeed(
          [&]() {
            AddBiasAct<float, ActivationType::Tanh>(bias_gpu, &out_gpu);
          },
          n_step,
          "GPU Tanh Activation " + std::to_string(m) + "," + std::to_string(n),
          m * n * sizeof(float) / 1e9, kDLGPU);
#endif
    }  // seq_length
  }    // for batch_size
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
