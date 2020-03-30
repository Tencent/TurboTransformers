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

#include "turbo_transformers/layers/kernels/transpose.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/aligned_scratchpad.h"
#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/test_helper.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
TEST_CASE("split_add_transpose CPU and GPU correctness") {
  int64_t num_attention_heads = 12;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{10, 20, 32, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
      std::tie(input_tensor_cpu, input_tensor_gpu) =
          test::CreateAndFillRandomForCPUGPUTensors<float>(
              {batch_size, seq_length, 3, num_attention_heads, 64});

      core::Tensor bias_tensor_cpu(nullptr), bias_tensor_gpu(nullptr);
      std::tie(bias_tensor_cpu, bias_tensor_gpu) =
          test::CreateAndFillRandomForCPUGPUTensors<float>(
              {3, num_attention_heads, 64});

      turbo_transformers::core::Tensor output_tensor_gpu(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {3, batch_size, num_attention_heads, seq_length, 64}, kDLGPU, 0));
      turbo_transformers::core::Tensor output_tensor_cpu(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {3, batch_size, num_attention_heads, seq_length, 64}, kDLCPU, 0));

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;

      SplitAddBiasTransposeForScore(&output_tensor_gpu, input_tensor_gpu,
                                    bias_tensor_gpu);
      SplitAddBiasTransposeForScore(&output_tensor_cpu, input_tensor_cpu,
                                    bias_tensor_cpu);
      REQUIRE(::turbo_transformers::test::CheckResultOfCPUAndGPU<float>(
          output_tensor_cpu, output_tensor_gpu));
    }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
