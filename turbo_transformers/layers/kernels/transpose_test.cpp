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

#include "turbo_transformers/layers/kernels/transpose.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/common.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
TEST_CASE("splitaddtranspose-gpu-test") {
  const std::vector<int64_t> num_attention_heads_list{12};
  const std::vector<int64_t> batch_size_list{1, 20, 24};
  const std::vector<int64_t> seq_length_list{10, 20, 32, 64, 128};

  for (auto num_attention_heads : num_attention_heads_list)
    for (auto batch_size : batch_size_list)
      for (auto seq_length : seq_length_list) {
        std::cout << num_attention_heads << ", " << batch_size << " ,"
                  << seq_length << std::endl;
        core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
        std::tie(input_tensor_cpu, input_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, seq_length, 3, num_attention_heads, 64});

        core::Tensor bias_tensor_cpu(nullptr), bias_tensor_gpu(nullptr);
        std::tie(bias_tensor_cpu, bias_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {3, num_attention_heads, 64});

        turbo_transformers::core::Tensor output_tensor_gpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {3, batch_size, num_attention_heads, seq_length, 64}, kDLGPU,
                0));
        turbo_transformers::core::Tensor output_tensor_cpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {3, batch_size, num_attention_heads, seq_length, 64}, kDLCPU,
                0));

        SplitAddBiasTransposeForScore(&output_tensor_gpu, input_tensor_gpu,
                                      bias_tensor_gpu);
        SplitAddBiasTransposeForScore(&output_tensor_cpu, input_tensor_cpu,
                                      bias_tensor_cpu);
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(output_tensor_cpu,
                                                      output_tensor_gpu));
      }
}

TEST_CASE("transpose-gpu-test") {
  const std::vector<int64_t> num_attention_heads_list{12};
  const std::vector<int64_t> batch_size_list{1, 20, 24};
  const std::vector<int64_t> seq_length_list{10, 20, 32, 64, 128};

  for (auto num_attention_heads : num_attention_heads_list)
    for (auto batch_size : batch_size_list)
      for (auto seq_length : seq_length_list) {
        std::cout << num_attention_heads << ", " << batch_size << " ,"
                  << seq_length << std::endl;
        core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
        std::tie(input_tensor_cpu, input_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, num_attention_heads, seq_length, 64});

        turbo_transformers::core::Tensor output_tensor_gpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, seq_length, num_attention_heads, 64}, kDLGPU, 0));
        turbo_transformers::core::Tensor output_tensor_cpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, seq_length, num_attention_heads, 64}, kDLCPU, 0));

        TransposeForScore(&output_tensor_gpu, input_tensor_gpu);
        TransposeForScore(&output_tensor_cpu, input_tensor_cpu);
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(output_tensor_cpu,
                                                      output_tensor_gpu));
      }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
