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

#include "prepare_bert_masks.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/common.h"

namespace turbo_transformers {
namespace layers {

#ifdef TT_WITH_CUDA
TEST_CASE("prepare_bert_masks CPU and GPU correctness") {
  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      core::Tensor gpu_inputs(nullptr), cpu_inputs(nullptr);
      std::tie(cpu_inputs, gpu_inputs) =
          kernels::common::CreateAndFillRandomForCPUGPUTensors<float>(
              {batch_size, seq_length});

      core::Tensor gpu_att_mask(nullptr), cpu_att_mask(nullptr),
          gpu_seq_type(nullptr), cpu_seq_type(nullptr);
      core::Tensor gpu_position_ids(nullptr), cpu_position_ids(nullptr),
          gpu_extended_attention_mask(nullptr),
          cpu_extended_attention_mask(nullptr);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      {
        PrepareBertMasks func;
        func(cpu_inputs, &cpu_att_mask, &cpu_seq_type, &cpu_position_ids,
             &cpu_extended_attention_mask);

        func(gpu_inputs, &gpu_att_mask, &gpu_seq_type, &gpu_position_ids,
             &gpu_extended_attention_mask);
      }
      REQUIRE(kernels::common::CheckResultOfCPUAndGPU<int64_t>(cpu_att_mask,
                                                               gpu_att_mask));
      REQUIRE(kernels::common::CheckResultOfCPUAndGPU<float>(
          cpu_extended_attention_mask, gpu_extended_attention_mask));
      REQUIRE(kernels::common::CheckResultOfCPUAndGPU<int64_t>(cpu_seq_type,
                                                               gpu_seq_type));
      REQUIRE(kernels::common::CheckResultOfCPUAndGPU<int64_t>(
          cpu_position_ids, gpu_position_ids));
    }  // for
}
#endif

}  // namespace layers
}  // namespace turbo_transformers
