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

#include "prepare_bert_masks.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/aligned_scratchpad.h"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/test_helper.h"

namespace turbo_transformers {
namespace layers {

#ifdef FT_WITH_CUDA
TEST_CASE("prepare_bert_masks CPU and GPU correctness") {
  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      turbo_transformers::core::Tensor gpu_inputs(
          turbo_transformers::core::NewDLPackTensorT<int64_t>(
              {batch_size, seq_length}, kDLGPU, 0));

      turbo_transformers::core::Tensor cpu_inputs(
          turbo_transformers::core::NewDLPackTensorT<int64_t>(
              {batch_size, seq_length}, kDLCPU, 0));

      turbo_transformers::core::Tensor gpu_att_mask(nullptr);
      turbo_transformers::core::Tensor cpu_att_mask(nullptr);
      turbo_transformers::core::Tensor gpu_seq_type(nullptr);
      turbo_transformers::core::Tensor cpu_seq_type(nullptr);

      turbo_transformers::core::Tensor gpu_position_ids(nullptr);
      turbo_transformers::core::Tensor cpu_position_ids(nullptr);
      turbo_transformers::core::Tensor gpu_extended_attention_mask(nullptr);
      turbo_transformers::core::Tensor cpu_extended_attention_mask(nullptr);

      ::turbo_transformers::test::FillDataForCPUGPUTensors<int64_t>(cpu_inputs,
                                                                    gpu_inputs);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      {
        PrepareBertMasks func;
        func(cpu_inputs, &cpu_att_mask, &cpu_seq_type, &cpu_position_ids,
             &cpu_extended_attention_mask);

        func(gpu_inputs, &gpu_att_mask, &gpu_seq_type, &gpu_position_ids,
             &gpu_extended_attention_mask);
      }
      REQUIRE(::turbo_transformers::test::CheckResultOfCPUAndGPU<int64_t>(
          cpu_att_mask, gpu_att_mask));
      REQUIRE(::turbo_transformers::test::CheckResultOfCPUAndGPU<float>(
          cpu_extended_attention_mask, gpu_extended_attention_mask));
      REQUIRE(::turbo_transformers::test::CheckResultOfCPUAndGPU<int64_t>(
          cpu_seq_type, gpu_seq_type));
      REQUIRE(::turbo_transformers::test::CheckResultOfCPUAndGPU<int64_t>(
          cpu_position_ids, gpu_position_ids));
    }  // for
}
#endif

}  // namespace layers
}  // namespace turbo_transformers
