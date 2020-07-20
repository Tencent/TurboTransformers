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

#include "turbo_transformers/layers/kernels/softmax.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/common.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
TEST_CASE("softmax-gpu-2Dmask-test") {
  int64_t num_attention_heads = 12;

  constexpr float scaler = 1.;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> from_seq_list{10,  20,  40,  60,  80,
                                     100, 200, 300, 400, 500};
  std::vector<int64_t> to_seq_list{10, 20, 40, 60, 80, 100};
  for (auto batch_size : batch_size_list)
    for (auto from_seq : from_seq_list)
      for (auto to_seq : to_seq_list) {
        core::Tensor qk_buf_cpu(nullptr), qk_buf_gpu(nullptr);
        std::tie(qk_buf_cpu, qk_buf_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, num_attention_heads, from_seq, to_seq});

        core::Tensor attr_mask_cpu(nullptr), attr_mask_gpu(nullptr);
        std::tie(attr_mask_cpu, attr_mask_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, 1, 1, to_seq});
        ApplyMaskAndSoftmax(&qk_buf_gpu, attr_mask_gpu, scaler);

        ApplyMaskAndSoftmax(&qk_buf_cpu, attr_mask_cpu, scaler);

        REQUIRE(common::CheckResultOfCPUAndGPU<float>(qk_buf_cpu, qk_buf_gpu));
      }
}


TEST_CASE("softmax-gpu-3Dmask-test") {
  int64_t num_attention_heads = 12;

  constexpr float scaler = 1.;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> from_seq_list{10,  20,  40,  60,  80,
                                     100, 200, 300, 400, 500};
  std::vector<int64_t> to_seq_list{10, 20, 40, 60, 80, 100};
  for (auto batch_size : batch_size_list)
    for (auto from_seq : from_seq_list)
      for (auto to_seq : to_seq_list) {
        core::Tensor qk_buf_cpu(nullptr), qk_buf_gpu(nullptr);
        std::tie(qk_buf_cpu, qk_buf_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, num_attention_heads, from_seq, to_seq});

        core::Tensor attr_mask_cpu(nullptr), attr_mask_gpu(nullptr);
        std::tie(attr_mask_cpu, attr_mask_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, 1, from_seq, to_seq});

        ApplyMaskAndSoftmax(&qk_buf_gpu, attr_mask_gpu, scaler);

        ApplyMaskAndSoftmax(&qk_buf_cpu, attr_mask_cpu, scaler);

        REQUIRE(common::CheckResultOfCPUAndGPU<float>(qk_buf_cpu, qk_buf_gpu));
      }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
