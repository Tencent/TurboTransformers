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

#pragma once
#include <stdint.h>
#include <turbo_transformers/core/tensor.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace turbo_transformers {
namespace layers {
namespace kernels {

/***
 * Transpose from (Batch, Seq_len, Head, Size_per_head) ->
 * (Batch, Head, Seq_len, Size_per_head)
 * def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor
 * **/
extern void TransposeForScore(core::Tensor* output, const core::Tensor& input,
                              const std::string name = "TransposeForScore");

extern void AddBiasTransposeForScore(
    const core::Tensor& input, const core::Tensor& bias, core::Tensor* output,
    const std::string name = "AddBiasTransposeForScore");

// input: (batch_size, seq_length, 3, head_num, *size_per_head)
// bias: (3, head_num, size_per_head)
// output: (3, batch_size, num_attention_heads, seq_length, size_per_head)
extern void SplitAddBiasTransposeForScore(
    core::Tensor* output, const core::Tensor& input_tensor,
    const core::Tensor& bias_tensor,
    const std::string name = "SplitAddBiasTransposeForScore");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
