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

#include "turbo_transformers/layers/multi_headed_attention.h"

#include "loguru.hpp"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"
#include "turbo_transformers/layers/kernels/utils.h"

#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {

static std::mutex mutex_;

// context attn
template <>
void MultiHeadedAttention::FuseGemm012AddBIasTranspose<false>(
    const core::Tensor& query_tensor, const core::Tensor& value_tensor,
    const core::Tensor& key_tensor, bool pre_layernorm, bool is_trans_weight,
    std::unordered_map<std::string, core::Tensor*>& layer_cache,
    core::Tensor* q_out, core::Tensor* k_out, core::Tensor* v_out) const {
  core::Tensor temp_q_out(nullptr);
  core::Tensor temp_k_out(nullptr);
  core::Tensor temp_v_out(nullptr);

  core::Tensor temp_q_out2(nullptr);
  core::Tensor temp_v_out2(nullptr);
  core::Tensor temp_k_out2(nullptr);

  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(query_tensor.device_ctx(),
                                                    value_tensor.device_ctx()),
                true,
                "The query_tensor and value_tensor should have the same "
                "device type and device id.");
  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(query_tensor.device_ctx(),
                                                    key_tensor.device_ctx()),
                true,
                "The query_tensor and key_tensor should have the same "
                "device type and device id.");

  temp_q_out.Reshape<float>({batch_size_, query_seq_length_, hidden_size_},
                            devtype_, devid_, "context/gemm0/q_out1/Reshape");
  if (pre_layernorm) {
    temp_q_out2.Reshape<float>({batch_size_, query_seq_length_, hidden_size_},
                               devtype_, devid_,
                               "context/gemm0/q_out2/Reshape");
    core::Copy<float>(query_tensor, temp_q_out2,
                      "context/gemm0/prelayernorm/Copy");
    kernels::LayerNorm<float>(
        layernorm_gamma_, layernorm_beta_, &temp_q_out2, 1e-6,
        "context/gemm0/prelayernorm");  // q_out2 here is used as
    // layernormed_query TODO(jiaruifang)
    // 1e-6 should not be hard-coded
    kernels::MatMul(temp_q_out2, false, q_weight_, is_trans_weight, 1.0,
                    &temp_q_out, 0.0, "context/gemm0");
  } else {
    kernels::MatMul(query_tensor, false, q_weight_, is_trans_weight, 1.0,
                    &temp_q_out, 0.0, "context/gemm0");
  }
  temp_q_out.Reshape<float>(
      {batch_size_, query_seq_length_, num_attention_heads_, size_per_head_},
      devtype_, devid_, "context/AddBiasTransposeForScore/q_out1/Reshape");
  temp_q_out2.Reshape<float>(
      {batch_size_, num_attention_heads_, query_seq_length_, size_per_head_},
      devtype_, devid_, "context/AddBiasTransposeForScore/q_out2/Reshape");
  kernels::AddBiasTransposeForScore(temp_q_out, q_bias_, &temp_q_out2,
                                    "context/AddBiasTransposeForScore");
  *q_out = std::move(temp_q_out2);

  if (memory_not_none_) {
    k_out->Reshape<float>(
        {batch_size_, num_attention_heads_,
         layer_cache["memory_values"]->shape(2), size_per_head_},
        devtype_, devid_, "self/k/Reshape");
    v_out->Reshape<float>(
        {batch_size_, num_attention_heads_,
         layer_cache["memory_keys"]->shape(2), size_per_head_},
        devtype_, devid_, "self/v/Reshape");

    core::Copy<float>(*layer_cache["memory_values"], *v_out,
                      "context/memory_values/Copy");
    core::Copy<float>(*layer_cache["memory_keys"], *k_out,
                      "context/memory_keys/Copy");
  } else {
    temp_v_out.Reshape<float>({batch_size_, key_seq_length_, hidden_size_},
                              devtype_, devid_, "context/gemm1/v_out1/Reshape");
    temp_k_out.Reshape<float>({batch_size_, key_seq_length_, hidden_size_},
                              devtype_, devid_, "context/gemm2/k_out1/Reshape");

    kernels::MatMul(key_tensor, false, k_weight_, is_trans_weight, 1.0,
                    &temp_k_out, 0.0, "context/gemm1");
    kernels::MatMul(value_tensor, false, v_weight_, is_trans_weight, 1.0,
                    &temp_v_out, 0.0, "context/gemm2");
    temp_v_out.Reshape<float>(
        {batch_size_, key_seq_length_, num_attention_heads_, size_per_head_},
        devtype_, devid_, "context/gemm1/v_out1/Reshape");
    temp_k_out.Reshape<float>(
        {batch_size_, key_seq_length_, num_attention_heads_, size_per_head_},
        devtype_, devid_, "context/gemm2/k_out1/Reshape");

    if (layer_cache_not_none_) {
      layer_cache["memory_keys"]->Reshape<float>(
          {batch_size_, num_attention_heads_, key_seq_length_, size_per_head_},
          devtype_, devid_, "context/keys/AddBiasTransposeForScore/Reshape");
      layer_cache["memory_values"]->Reshape<float>(
          {batch_size_, num_attention_heads_, key_seq_length_, size_per_head_},
          devtype_, devid_, "context/values/AddBiasTransposeForScore/reshape");
      kernels::AddBiasTransposeForScore(
          temp_v_out, v_bias_, layer_cache["memory_values"],
          "context/values/AddBiasTransposeForScore");
      kernels::AddBiasTransposeForScore(
          temp_k_out, k_bias_, layer_cache["memory_keys"],
          "context/keys/AddBiasTransposeForScore");

      k_out->Reshape<float>(
          {batch_size_, num_attention_heads_,
           layer_cache["memory_values"]->shape(2), size_per_head_},
          devtype_, devid_, "self/k/Reshape");
      v_out->Reshape<float>(
          {batch_size_, num_attention_heads_,
           layer_cache["memory_keys"]->shape(2), size_per_head_},
          devtype_, devid_, "self/v/Reshape");

      core::Copy<float>(*layer_cache["memory_values"], *v_out,
                        "context/memory_values/Copy");
      core::Copy<float>(*layer_cache["memory_keys"], *k_out,
                        "context/memory_keys/Copy");
    } else {
      temp_v_out2.Reshape<float>(
          {batch_size_, num_attention_heads_, key_seq_length_, size_per_head_},
          devtype_, devid_, "context/values/AddBiasTransposeForScore/Reshape");
      temp_k_out2.Reshape<float>(
          {batch_size_, num_attention_heads_, key_seq_length_, size_per_head_},
          devtype_, devid_, "context/keys/AddBiasTransposeForScore/Reshape");
      kernels::AddBiasTransposeForScore(
          temp_v_out, v_bias_, &temp_v_out2,
          "context/values/AddBiasTransposeForScore");
      kernels::AddBiasTransposeForScore(
          temp_k_out, k_bias_, &temp_k_out2,
          "context/keys/AddBiasTransposeForScore");
      *v_out = std::move(temp_v_out2);
      *k_out = std::move(temp_k_out2);
    }
  }  // else
}


// self attn
template <>
void MultiHeadedAttention::FuseGemm012AddBIasTranspose<true>(
    const core::Tensor& query_tensor, const core::Tensor& value_tensor,
    const core::Tensor& key_tensor, bool pre_layernorm, bool is_trans_weight,
    std::unordered_map<std::string, core::Tensor*>& layer_cache,
    core::Tensor* q_out, core::Tensor* k_out, core::Tensor* v_out) const {
  core::Tensor tmp_qkv_out1(nullptr);

  q_out->Reshape<float>(
      {batch_size_, num_attention_heads_, query_seq_length_, size_per_head_},
      devtype_, devid_, "self/q/Reshape");
  k_out->Reshape<float>(
      {batch_size_, num_attention_heads_, query_seq_length_, size_per_head_},
      devtype_, devid_, "self/k/Reshape");
  v_out->Reshape<float>(
      {batch_size_, num_attention_heads_, query_seq_length_, size_per_head_},
      devtype_, devid_, "self/v/Reshape");

  tmp_qkv_out1.Reshape<float>({batch_size_, query_seq_length_, 3, hidden_size_},
                              devtype_, devid_, "self/qkv_out1/Reshape");
  if (pre_layernorm) {
    core::Tensor layernormed_query(nullptr);
    layernormed_query.Reshape<float>(
        {batch_size_, query_seq_length_, hidden_size_}, devtype_, devid_,
        "self/layernorm/Reshape");
    core::Copy<float>(query_tensor, layernormed_query, "self/layernorm/Copy");
    kernels::LayerNorm<float>(layernorm_gamma_, layernorm_beta_,
                              &layernormed_query, 1e-6);
    kernels::MatMul(layernormed_query, false, qkv_weight_, is_trans_weight, 1.0,
                    &tmp_qkv_out1, 0.0, "self/gemm012_fused");
  } else {
    kernels::MatMul(query_tensor, false, qkv_weight_, is_trans_weight, 1.0,
                    &tmp_qkv_out1, 0.0, "self/gemm012_fused");
  }

  kernels::SplitAddBiasTransposeForScore(tmp_qkv_out1, qkv_bias_, *q_out,
                                         *k_out, *v_out,
                                         "self/SplitAddBiasTransposeForScore");

  if (self_keys_not_none_) {
    core::Tensor tmp_k_out(nullptr);
    kernels::Concat<float>(*layer_cache["self_keys"], *k_out, 2, &tmp_k_out,
                           "self/keys/Concat");
    *k_out = std::move(tmp_k_out);
  }

  if (self_values_not_none_) {
    core::Tensor tmp_v_out(nullptr);
    kernels::Concat<float>(*layer_cache["self_values"], *v_out, 2, &tmp_v_out,
                           "self/values/Concat");
    *v_out = std::move(tmp_v_out);
  }
  if (layer_cache_not_none_) {
    layer_cache["self_keys"]->Reshape<float>(
        {batch_size_, num_attention_heads_, k_out->shape(2), size_per_head_},
        devtype_, devid_, "self/self_key/Reshape");
    layer_cache["self_values"]->Reshape<float>(
        {batch_size_, num_attention_heads_, v_out->shape(2), size_per_head_},
        devtype_, devid_, "self/self_value/Reshape");

    core::Copy<float>(*k_out, *layer_cache["self_keys"], "self/self_key/Copy");
    core::Copy<float>(*v_out, *layer_cache["self_values"],
                      "self/self_value/Copy");
  }
}

void MultiHeadedAttention::SetContextFlag(
    const std::unordered_map<std::string, core::Tensor*>& layer_cache) const {
  layer_cache_not_none_ = layer_cache.size() > 0 ? true : false;
  memory_keys_not_none_ = false;
  memory_values_not_none_ = false;
  self_keys_not_none_ = false;
  self_values_not_none_ = false;
  if (layer_cache_not_none_) {
    for (auto it = layer_cache.begin(); it != layer_cache.end(); ++it) {
      if (it->first == "memory_keys" && !it->second->is_null()) {
        memory_keys_not_none_ = true;
      }
      if (it->first == "memory_values" && !it->second->is_null()) {
        memory_values_not_none_ = true;
      }
      if (it->first == "self_keys" && !it->second->is_null()) {
        self_keys_not_none_ = true;
      }
      if (it->first == "self_values" && !it->second->is_null()) {
        self_values_not_none_ = true;
      }
    }
  }
  memory_not_none_ = memory_values_not_none_ && memory_keys_not_none_;
}

void MultiHeadedAttention::operator()(
    const core::Tensor& key_tensor, const core::Tensor& value_tensor,
    const core::Tensor& query_tensor, const core::Tensor& attention_mask,
    const std::string& attn_type, core::Tensor* output, core::Tensor* att_score,
    std::unordered_map<std::string, core::Tensor*> layer_cache,
    bool pre_layernorm, bool post_layernorm, bool post_add_input,
    bool is_trans_weight) const {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("MultiHeadedAttention_" + attn_type,
                            query_tensor.device_type());
#endif
  std::lock_guard<std::mutex> g(mutex_);

  TT_ENFORCE_EQ(key_tensor.n_dim(), 3,
                "The key_tensor should be a matrix with shape [batch_size_, "
                "key_seq_len, hidden_size_].");
  TT_ENFORCE_EQ(value_tensor.n_dim(), 3,
                "The value_tensor should be a matrix with shape [batch_size_, "
                "key_seq_len, hidden_size_].");
  TT_ENFORCE_EQ(query_tensor.n_dim(), 3,
                "The query_tensors should be a matrix with shape [batch_size_, "
                "query_seq_len, hidden_size_].");
  TT_ENFORCE_EQ(
      key_tensor.shape(0), value_tensor.shape(0),
      "The key_tensor and value_tensor should have the same hidden_size_");

  EnforceShapeAndType();
  batch_size_ = query_tensor.shape(0);
  query_seq_length_ =
      query_tensor.shape(1);  // query_seq_length_ = from_seq_Len

  if (attn_type == "context") {
    key_seq_length_ = key_tensor.shape(1);
  } else if (attn_type == "self") {
    key_seq_length_ = query_seq_length_;
  } else {
    TT_THROW("attn_type should be context or self.");
  }

  hidden_size_ = query_tensor.shape(2);
  size_per_head_ = hidden_size_ / num_attention_heads_;
  devtype_ = query_tensor.device_type();
  devid_ = query_tensor.device_id();

  SetContextFlag(layer_cache);

  // TODO we should caching allocated intermediate tensor.
  core::Tensor q_out{nullptr}, k_out{nullptr}, v_out{nullptr};
  if (attn_type == "context") {
    FuseGemm012AddBIasTranspose<false>(query_tensor, value_tensor, key_tensor,
                                       pre_layernorm, is_trans_weight,
                                       layer_cache, &q_out, &k_out, &v_out);
  } else if (attn_type == "self") {
    FuseGemm012AddBIasTranspose<true>(query_tensor, value_tensor, key_tensor,
                                      pre_layernorm, is_trans_weight,
                                      layer_cache, &q_out, &k_out, &v_out);
  } else {
    TT_THROW("%s is not support in MultiHeadedAttention\n", attn_type);
  }  // if (attn_type == "context")
  // 2) Calculate and scale scores.
  key_seq_length_ = k_out.shape(
      2);  // update for self type attn, since it will concat with cache.
  att_score->Reshape<float>(
      {batch_size_, num_attention_heads_, query_seq_length_,
       key_seq_length_},  // query_seq_length_ = from_seq_Len
      devtype_, devid_, "batch_gemm3/Reshape");

  const float scaler = 1.0f / std::sqrt(static_cast<float>(size_per_head_));
  kernels::BatchMatMul(q_out, false, k_out, true, scaler, att_score, 0.0,
                       "batch_gemm3");  //(B, num_head, q_len, k_len)
  // mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
  // scores = scores.masked_fill(mask, -1e18)
  // attn = self.softmax(scores).to(query.dtype)
  kernels::ApplyMaskAndSoftmax(
      att_score,
      attention_mask,  //(B, q_len, k_len) or (B, 1, k_len)
      1.0, "ApplyMaskAndSoftmax");

  // context_original = torch.matmul(drop_attn, value)
  core::Tensor context_layer(nullptr);
  context_layer.Reshape<float>(
      {batch_size_, num_attention_heads_, query_seq_length_, size_per_head_},
      devtype_, devid_, "ApplyMaskAndSoftmax/Reshape");

  kernels::BatchMatMul(*att_score, false, v_out, false, 1.0, &context_layer,
                       0.0, "batch_gemm4");
  // context = unshape(context_original)
  core::Tensor self_attr_out(nullptr);

  self_attr_out.Reshape<float>(
      {batch_size_, query_seq_length_, num_attention_heads_ * size_per_head_},
      devtype_, devid_, "batch_gemm4/Reshape");
  kernels::TransposeForScore(&self_attr_out, context_layer,
                             "TransposeForScore");
  // output = self.final_linear(context)
  output->Reshape<float>({batch_size_, query_seq_length_, hidden_size_},
                         devtype_, devid_, "gemm5/Reshape");

  kernels::MatMul(self_attr_out, false, dense_weight_, is_trans_weight, 1.0,
                  output, 0.0, "gemm5");

  if (false == post_add_input) {
    if (false == post_layernorm) {
      //+bias
      kernels::AddBias(dense_bias_, output, "AddBias");
    } else {
      //+bias+layernorm
      kernels::AddBiasLayerNorm<float>(query_tensor, dense_bias_,
                                       layernorm_gamma_,  // gemma
                                       layernorm_beta_, output, 1e-12,
                                       "AddBiasLayerNorm");
    }
  } else {
    //+input + bias
    kernels::AddInputBias(*output, query_tensor, dense_bias_, output);
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("MultiHeadedAttention_" + attn_type, devtype_);
#endif
}

void MultiHeadedAttention::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> qkv_weight_ <<<<<<<<<<<<" << std::endl;
    q_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> qkv_bias_ <<<<<<<<<<<<" << std::endl;
    q_bias_.Print<float>(os);
    os << ">>>>>>>>>>>> dense_weight_ <<<<<<<<<<<<" << std::endl;
    dense_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> dense_bias_ <<<<<<<<<<<<" << std::endl;
    dense_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace turbo_transformers
