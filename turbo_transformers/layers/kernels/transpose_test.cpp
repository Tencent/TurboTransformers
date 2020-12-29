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

#include <string.h>

#include <chrono>
#include <vector>

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
  const std::vector<int64_t> batch_size_list{1, 12, 20};
  const std::vector<int64_t> seq_length_list{10, 20, 32, 64, 128};

  for (auto hidden_size : {64, 2000})
    for (auto num_attention_heads : num_attention_heads_list)
      for (auto batch_size : batch_size_list)
        for (auto seq_length : seq_length_list) {
          core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
          std::tie(input_tensor_cpu, input_tensor_gpu) =
              common::CreateAndFillRandomForCPUGPUTensors<float>(
                  {batch_size, seq_length, 3, num_attention_heads,
                   hidden_size});

          core::Tensor bias_tensor_cpu(nullptr), bias_tensor_gpu(nullptr);
          std::tie(bias_tensor_cpu, bias_tensor_gpu) =
              common::CreateAndFillRandomForCPUGPUTensors<float>(
                  {3, num_attention_heads, hidden_size});

          turbo_transformers::core::Tensor output_tensor_gpu(
              turbo_transformers::core::NewDLPackTensorT<float>(
                  {3, batch_size, num_attention_heads, seq_length, hidden_size},
                  kDLGPU, 0));
          turbo_transformers::core::Tensor output_tensor_cpu(
              turbo_transformers::core::NewDLPackTensorT<float>(
                  {3, batch_size, num_attention_heads, seq_length, hidden_size},
                  kDLCPU, 0));

          SplitAddBiasTransposeForScore(&output_tensor_gpu, input_tensor_gpu,
                                        bias_tensor_gpu);
          SplitAddBiasTransposeForScore(&output_tensor_cpu, input_tensor_cpu,
                                        bias_tensor_cpu);
          REQUIRE(common::CheckResultOfCPUAndGPU<float>(output_tensor_cpu,
                                                        output_tensor_gpu));
        }
}

TEST_CASE("transpose-gpu-test") {
  const std::vector<int64_t> num_attention_heads_list{12, 20, 24};
  const std::vector<int64_t> batch_size_list{
      1,
      20,
  };
  const std::vector<int64_t> seq_length_list{10, 32, 128};

  for (auto num_attention_heads : num_attention_heads_list)
    for (auto batch_size : batch_size_list)
      for (auto seq_length : seq_length_list) {
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

TEST_CASE("transpose-bias-gpu-test") {
  const std::vector<int64_t> num_attention_heads_list{12, 20, 24};
  const std::vector<int64_t> batch_size_list{
      1,
      20,
  };
  const std::vector<int64_t> seq_length_list{10, 32, 128};

  for (auto num_attention_heads : num_attention_heads_list)
    for (auto batch_size : batch_size_list)
      for (auto seq_length : seq_length_list) {
        core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
        core::Tensor bias_tensor_cpu(nullptr), bias_tensor_gpu(nullptr);
        std::tie(input_tensor_cpu, input_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, seq_length, num_attention_heads, 64});
        std::tie(bias_tensor_cpu, bias_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {num_attention_heads, 64});

        turbo_transformers::core::Tensor output_tensor_gpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, num_attention_heads, seq_length, 64}, kDLGPU, 0));
        turbo_transformers::core::Tensor output_tensor_cpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, num_attention_heads, seq_length, 64}, kDLCPU, 0));

        AddBiasTransposeForScore(input_tensor_gpu, bias_tensor_gpu,
                                 &output_tensor_gpu);
        AddBiasTransposeForScore(input_tensor_cpu, bias_tensor_cpu,
                                 &output_tensor_cpu);
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(output_tensor_cpu,
                                                      output_tensor_gpu));
      }
}
#endif

TEST_CASE("transpose-bias-gpu-pad-test") {
  const std::vector<int64_t> num_attention_heads_list{12, 20, 24};

  const std::vector<int64_t> seq_length_list{7, 7, 7};
  int64_t sum_seq_len =
      std::accumulate(seq_length_list.begin(), seq_length_list.end(), 0);
  int64_t max_seq_len =
      *std::max_element(seq_length_list.begin(), seq_length_list.end());
  int64_t batch_size = seq_length_list.size();

  constexpr int64_t model_dim = 64;
  for (auto num_attention_heads : num_attention_heads_list) {
    core::Tensor input_tensor_cpu(nullptr);
    core::Tensor bias_tensor_cpu(nullptr);

    input_tensor_cpu.Reshape<float>(
        {1, sum_seq_len, 3, num_attention_heads * model_dim}, kDLCPU, 0);
    bias_tensor_cpu.Reshape<float>({3, num_attention_heads, model_dim}, kDLCPU,
                                   0);

    common::FillRandom<float>(input_tensor_cpu);
    common::Fill<float>(bias_tensor_cpu.mutableData<float>(),
                        3 * num_attention_heads * model_dim, 0., kDLCPU);

    // for reference
    core::Tensor input_tensor_ref_cpu(nullptr);
    input_tensor_ref_cpu.Reshape<float>(
        {batch_size, max_seq_len, 3, num_attention_heads * model_dim}, kDLCPU,
        0);
    auto* input_ref_data = input_tensor_ref_cpu.mutableData<float>();
    const auto* input_data = input_tensor_cpu.data<float>();

    memset(input_ref_data, 0.,
           sizeof(float) * batch_size * max_seq_len * 3 * num_attention_heads *
               model_dim);
    int64_t acc_seq_len = 0;
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      auto seq_len = seq_length_list[batch_idx];
      for (int64_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        auto* dst = input_ref_data + (batch_idx * max_seq_len + seq_idx) * 3 *
                                         num_attention_heads * model_dim;
        const auto* src = input_data + (acc_seq_len + seq_idx) * 3 *
                                           num_attention_heads * model_dim;
        memcpy(dst, src, sizeof(float) * 3 * num_attention_heads * model_dim);
      }
      acc_seq_len += seq_len;
    }
    core::Tensor output_q_cpu_ref(
        turbo_transformers::core::NewDLPackTensorT<float>(
            {batch_size, num_attention_heads, max_seq_len * model_dim}, kDLCPU,
            0));
    core::Tensor output_k_cpu_ref(
        turbo_transformers::core::NewDLPackTensorT<float>(
            {batch_size, num_attention_heads, max_seq_len * model_dim}, kDLCPU,
            0));
    core::Tensor output_v_cpu_ref(
        turbo_transformers::core::NewDLPackTensorT<float>(
            {batch_size, num_attention_heads, max_seq_len * model_dim}, kDLCPU,
            0));

    SplitAddBiasTransposeForScore(input_tensor_ref_cpu, bias_tensor_cpu,
                                  output_q_cpu_ref, output_k_cpu_ref,
                                  output_v_cpu_ref);

    // real function call
    core::Tensor output_q_cpu(turbo_transformers::core::NewDLPackTensorT<float>(
        {batch_size, num_attention_heads, max_seq_len * model_dim}, kDLCPU, 0));
    core::Tensor output_k_cpu(turbo_transformers::core::NewDLPackTensorT<float>(
        {batch_size, num_attention_heads, max_seq_len * model_dim}, kDLCPU, 0));
    core::Tensor output_v_cpu(turbo_transformers::core::NewDLPackTensorT<float>(
        {batch_size, num_attention_heads, max_seq_len * model_dim}, kDLCPU, 0));
    SplitAddBiasTransposeForScorePad(input_tensor_cpu, bias_tensor_cpu,
                                     output_q_cpu, output_k_cpu, output_v_cpu,
                                     seq_length_list);

    REQUIRE(common::CheckResultOfCPU<float>(output_q_cpu, output_q_cpu_ref));
    REQUIRE(common::CheckResultOfCPU<float>(output_k_cpu, output_k_cpu_ref));
    REQUIRE(common::CheckResultOfCPU<float>(output_v_cpu, output_v_cpu_ref));
  }
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
