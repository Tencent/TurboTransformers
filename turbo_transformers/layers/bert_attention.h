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
#include <memory>
#include <mutex>
#include <utility>

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/multi_headed_attention.h"

namespace turbo_transformers {
namespace layers {

class BertAttention : public MultiHeadedAttention {
 public:
  BertAttention(core::Tensor qkv_weight, core::Tensor qkv_bias,
                core::Tensor dense_weight, core::Tensor dense_bias,
                core::Tensor layer_norm_weight, core::Tensor layer_norm_bias,
                int64_t num_attention_heads)
      : MultiHeadedAttention(
            std::move(core::Tensor(nullptr)), std::move(core::Tensor(nullptr)),
            std::move(core::Tensor(nullptr)), std::move(core::Tensor(nullptr)),
            std::move(core::Tensor(nullptr)), std::move(core::Tensor(nullptr)),
            std::move(dense_weight), std::move(dense_bias),
            std::move(qkv_weight), std::move(qkv_bias),
            std::move(layer_norm_weight),  //(768)
            std::move(layer_norm_bias), num_attention_heads) {
    EnforceShapeAndType();
  }
  void EnforceShapeAndType() const;

  void operator()(const core::Tensor &input_tensor,
                  const core::Tensor &attention_mask, core::Tensor *output,
                  core::Tensor *attn = nullptr,
                  bool is_trans_weight = false) const;
};

}  // namespace layers
}  // namespace turbo_transformers
