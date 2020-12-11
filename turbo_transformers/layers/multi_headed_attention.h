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
#include <unordered_map>
#include <utility>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {

class MultiHeadedAttention {
 public:
  MultiHeadedAttention(core::Tensor k_weight, core::Tensor k_bias,
                       core::Tensor v_weight, core::Tensor v_bias,
                       core::Tensor q_weight, core::Tensor q_bias,
                       core::Tensor dense_weight, core::Tensor dense_bias,
                       core::Tensor qkv_weight, core::Tensor qkv_bias,
                       int64_t num_attention_heads)
      : k_weight_(std::move(k_weight)),  //(768, 768)
        k_bias_(std::move(k_bias)),
        v_weight_(std::move(v_weight)),  //(768, 768)
        v_bias_(std::move(v_bias)),
        q_weight_(std::move(q_weight)),  //(768, 768)
        q_bias_(std::move(q_bias)),
        dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)),
        qkv_weight_(std::move(qkv_weight)),
        qkv_bias_(std::move(qkv_bias)),
        layernorm_gamma_(nullptr),
        layernorm_beta_(nullptr),
        num_attention_heads_(num_attention_heads) {
    EnforceShapeAndType();
  }

  MultiHeadedAttention(core::Tensor k_weight, core::Tensor k_bias,
                       core::Tensor v_weight, core::Tensor v_bias,
                       core::Tensor q_weight, core::Tensor q_bias,
                       core::Tensor dense_weight, core::Tensor dense_bias,
                       core::Tensor qkv_weight, core::Tensor qkv_bias,
                       core::Tensor layernorm_gamma,
                       core::Tensor layernorm_beta,
                       int64_t num_attention_heads)
      : k_weight_(std::move(k_weight)),  //(768, 768)
        k_bias_(std::move(k_bias)),
        v_weight_(std::move(v_weight)),  //(768, 768)
        v_bias_(std::move(v_bias)),
        q_weight_(std::move(q_weight)),  //(768, 768)
        q_bias_(std::move(q_bias)),
        dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)),
        qkv_weight_(std::move(qkv_weight)),
        qkv_bias_(std::move(qkv_bias)),
        layernorm_gamma_(std::move(layernorm_gamma)),
        layernorm_beta_(std::move(layernorm_beta)),
        num_attention_heads_(num_attention_heads) {
    EnforceShapeAndType();
  }
  void EnforceShapeAndType() const;

  void SetContextFlag(
      const std::unordered_map<std::string, core::Tensor*>& layer_cache) const;

  template <bool is_self>
  void FuseGemm012AddBIasTranspose(
      const core::Tensor& query_tensor, const core::Tensor& value_tensor,
      const core::Tensor& key_tensor, bool pre_layernorm, bool is_trans_weight,
      std::unordered_map<std::string, core::Tensor*>& layer_cache,
      core::Tensor* q_out, core::Tensor* k_out, core::Tensor* v_out) const;

  void operator()(const core::Tensor& key_tensor,
                  const core::Tensor& value_tensor,
                  const core::Tensor& query_tensor,
                  const core::Tensor& attention_mask,
                  const std::string& attn_type, core::Tensor* output,
                  core::Tensor* att_score,
                  std::unordered_map<std::string, core::Tensor*> layer_cache,
                  bool pre_layernorm = false, bool post_layernorm = false,
                  bool post_add_input = false,
                  bool is_trans_weight = false) const;

 private:
  core::Tensor k_weight_;
  core::Tensor k_bias_;
  core::Tensor v_weight_;
  core::Tensor v_bias_;
  core::Tensor q_weight_;
  core::Tensor q_bias_;
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;

  core::Tensor qkv_weight_;  // store fused qkv weight/bias for "self" type
                             // attention computation.
  core::Tensor qkv_bias_;

  core::Tensor layernorm_gamma_;
  core::Tensor layernorm_beta_;

  int64_t num_attention_heads_;

  mutable int64_t batch_size_;
  mutable int64_t query_seq_length_;
  mutable int64_t key_seq_length_;
  mutable int64_t hidden_size_;

  mutable int64_t size_per_head_;
  mutable DLDeviceType devtype_;
  mutable int devid_;

  mutable bool layer_cache_not_none_{false};
  mutable bool memory_keys_not_none_{false};
  mutable bool memory_values_not_none_{false};
  mutable bool self_keys_not_none_{false};
  mutable bool self_values_not_none_{false};
  mutable bool memory_not_none_{false};
};

}  // namespace layers
}  // namespace turbo_transformers
