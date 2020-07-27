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
                "The key_tensor should be a matrix with shape [batch_size, "
                "key_seq_len, hidden_size].");
  TT_ENFORCE_EQ(value_tensor.n_dim(), 3,
                "The value_tensor should be a matrix with shape [batch_size, "
                "key_seq_len, hidden_size].");
  TT_ENFORCE_EQ(query_tensor.n_dim(), 3,
                "The query_tensors should be a matrix with shape [batch_size, "
                "query_seq_len, hidden_size].");
  TT_ENFORCE_EQ(
      key_tensor.shape(0), value_tensor.shape(0),
      "The key_tensor and value_tensor should have the same hidden_size");

  EnforceShapeAndType();
  auto batch_size = query_tensor.shape(0);
  auto query_seq_length =
      query_tensor.shape(1);  // query_seq_length = from_seq_Len

  int64_t key_seq_length;
  if (attn_type == "context") {
    key_seq_length = key_tensor.shape(1);
  } else if (attn_type == "self") {
    key_seq_length = query_seq_length;
  } else {
    TT_THROW("attn_type should be context or self.");
  }

  auto hidden_size = query_tensor.shape(2);
  auto size_per_head = hidden_size / num_attention_heads_;
  auto devtype = query_tensor.device_type();
  auto devid = query_tensor.device_id();

  // TODO we should caching allocate intermediate tensor.
  core::Tensor *q_ptr{nullptr}, *k_ptr{nullptr}, *v_ptr{nullptr};
  core::Tensor q_out{nullptr}, k_out{nullptr}, v_out{nullptr};
  core::Tensor q_out1(nullptr);
  core::Tensor v_out1(nullptr);
  core::Tensor k_out1(nullptr);
  core::Tensor q_out2(nullptr);
  core::Tensor v_out2(nullptr);
  core::Tensor k_out2(nullptr);
  core::Tensor qkv_out1(nullptr);

  bool layer_cache_not_none = layer_cache.size() > 0 ? true : false;
  bool memory_keys_not_none = false, memory_values_not_none = false,
       self_keys_not_none = false, self_values_not_none = false;
  if (layer_cache_not_none) {
    for (auto it = layer_cache.begin(); it != layer_cache.end(); ++it) {
      if (it->first == "memory_keys" && !it->second->is_null()) {
        memory_keys_not_none = true;
      }
      if (it->first == "memory_values" && !it->second->is_null()) {
        memory_values_not_none = true;
      }
      if (it->first == "self_keys" && !it->second->is_null()) {
        self_keys_not_none = true;
      }
      if (it->first == "self_values" && !it->second->is_null()) {
        self_values_not_none = true;
      }
    }
  }
  bool memory_not_none = memory_values_not_none && memory_keys_not_none;
  if (attn_type == "context") {
    TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(
                      query_tensor.device_ctx(), value_tensor.device_ctx()),
                  true,
                  "The query_tensor and value_tensor should have the same "
                  "device type and device id.");
    TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(query_tensor.device_ctx(),
                                                      key_tensor.device_ctx()),
                  true,
                  "The query_tensor and key_tensor should have the same "
                  "device type and device id.");

    q_out1.Reshape<float>({batch_size, query_seq_length, hidden_size}, devtype,
                          devid, "context/gemm0/q_out1/Reshape");
    if (pre_layernorm) {
      q_out2.Reshape<float>({batch_size, query_seq_length, hidden_size},
                            devtype, devid, "context/gemm0/q_out2/Reshape");
      core::Copy<float>(query_tensor, q_out2,
                        "context/gemm0/prelayernorm/Copy");
      kernels::LayerNorm<float>(
          layernorm_gamma_, layernorm_beta_, &q_out2, 1e-6,
          "context/gemm0/prelayernorm");  // q_out2 here is used as
                                          // layernormed_query TODO(jiaruifang)
                                          // 1e-6 should not be hard-coded
      kernels::MatMul(q_out2, false, q_weight_, is_trans_weight, 1.0, &q_out1,
                      0.0, "context/gemm0");
    } else {
      kernels::MatMul(query_tensor, false, q_weight_, is_trans_weight, 1.0,
                      &q_out1, 0.0, "context/gemm0");
    }
    q_out1.Reshape<float>(
        {batch_size, query_seq_length, num_attention_heads_, size_per_head},
        devtype, devid, "context/AddBiasTransposeForScore/q_out1/Reshape");
    q_out2.Reshape<float>(
        {batch_size, num_attention_heads_, query_seq_length, size_per_head},
        devtype, devid, "context/AddBiasTransposeForScore/q_out2/Reshape");
    kernels::AddBiasTransposeForScore(q_out1, q_bias_, &q_out2,
                                      "context/AddBiasTransposeForScore");
    q_ptr = &q_out2;  // point to static memory space
    if (memory_not_none) {
      v_ptr = layer_cache["memory_values"];
      k_ptr = layer_cache["memory_keys"];
    } else {
      v_out1.Reshape<float>({batch_size, key_seq_length, hidden_size}, devtype,
                            devid, "context/gemm1/v_out1/Reshape");
      k_out1.Reshape<float>({batch_size, key_seq_length, hidden_size}, devtype,
                            devid, "context/gemm2/k_out1/Reshape");

      kernels::MatMul(key_tensor, false, k_weight_, is_trans_weight, 1.0,
                      &k_out1, 0.0, "context/gemm1");
      kernels::MatMul(value_tensor, false, v_weight_, is_trans_weight, 1.0,
                      &v_out1, 0.0, "context/gemm2");
      v_out1.Reshape<float>(
          {batch_size, key_seq_length, num_attention_heads_, size_per_head},
          devtype, devid, "context/gemm1/v_out1/Reshape");
      k_out1.Reshape<float>(
          {batch_size, key_seq_length, num_attention_heads_, size_per_head},
          devtype, devid, "context/gemm2/k_out1/Reshape");

      if (layer_cache_not_none) {
        layer_cache["memory_keys"]->Reshape<float>(
            {batch_size, num_attention_heads_, key_seq_length, size_per_head},
            devtype, devid, "context/keys/AddBiasTransposeForScore/Reshape");
        layer_cache["memory_values"]->Reshape<float>(
            {batch_size, num_attention_heads_, key_seq_length, size_per_head},
            devtype, devid, "context/values/AddBiasTransposeForScore/reshape");
        kernels::AddBiasTransposeForScore(
            v_out1, v_bias_, layer_cache["memory_values"],
            "context/values/AddBiasTransposeForScore");
        kernels::AddBiasTransposeForScore(
            k_out1, k_bias_, layer_cache["memory_keys"],
            "context/keys/AddBiasTransposeForScore");
        v_ptr = layer_cache["memory_values"];
        k_ptr = layer_cache["memory_keys"];
      } else {
        v_out2.Reshape<float>(
            {batch_size, num_attention_heads_, key_seq_length, size_per_head},
            devtype, devid, "context/values/AddBiasTransposeForScore/Reshape");
        k_out2.Reshape<float>(
            {batch_size, num_attention_heads_, key_seq_length, size_per_head},
            devtype, devid, "context/keys/AddBiasTransposeForScore/Reshape");
        kernels::AddBiasTransposeForScore(
            v_out1, v_bias_, &v_out2,
            "context/values/AddBiasTransposeForScore");
        kernels::AddBiasTransposeForScore(
            k_out1, k_bias_, &k_out2, "context/keys/AddBiasTransposeForScore");
        v_ptr = &v_out2;
        k_ptr = &k_out2;
      }
    }  // else
  } else if (attn_type == "self") {
    qkv_out1.Reshape<float>({batch_size, query_seq_length, 3, hidden_size},
                            devtype, devid, "self/qkv_out1/Reshape");
    if (pre_layernorm) {
      core::Tensor layernormed_query(nullptr);
      layernormed_query.Reshape<float>(
          {batch_size, query_seq_length, hidden_size}, devtype, devid,
          "self/layernorm/Reshape");
      core::Copy<float>(query_tensor, layernormed_query, "self/layernorm/Copy");
      kernels::LayerNorm<float>(layernorm_gamma_, layernorm_beta_,
                                &layernormed_query, 1e-6);
      kernels::MatMul(layernormed_query, false, qkv_weight_, is_trans_weight,
                      1.0, &qkv_out1, 0.0, "self/gemm012_fused");
    } else {
      kernels::MatMul(query_tensor, false, qkv_weight_, is_trans_weight, 1.0,
                      &qkv_out1, 0.0, "self/gemm012_fused");
    }
    q_out.Reshape<float>(
        {batch_size, num_attention_heads_, query_seq_length, size_per_head},
        devtype, devid, "self/q_out/Reshape");
    k_out.Reshape<float>(
        {batch_size, num_attention_heads_, query_seq_length, size_per_head},
        devtype, devid, "self/k_out/Reshape");
    v_out.Reshape<float>(
        {batch_size, num_attention_heads_, query_seq_length, size_per_head},
        devtype, devid, "self/v_out/Reshape");

    kernels::SplitAddBiasTransposeForScore(
        qkv_out1, qkv_bias_, q_out, k_out, v_out,
        "self/SplitAddBiasTransposeForScore");
    q_ptr = &q_out;
    if (self_keys_not_none) {
      kernels::Concat<float>(*layer_cache["self_keys"], k_out, 2, &k_out2,
                             "self/keys/Concat");
      k_ptr = &k_out2;
    } else {
      k_ptr = &k_out;
    }
    if (self_values_not_none) {
      kernels::Concat<float>(*layer_cache["self_values"], v_out, 2, &v_out2,
                             "self/values/Concat");
      v_ptr = &v_out2;
    } else {
      v_ptr = &v_out;
    }
    if (layer_cache_not_none) {
      layer_cache["self_keys"]->Reshape<float>(
          {batch_size, num_attention_heads_, k_ptr->shape(2), size_per_head},
          devtype, devid, "self/self_key/Reshape");
      layer_cache["self_values"]->Reshape<float>(
          {batch_size, num_attention_heads_, v_ptr->shape(2), size_per_head},
          devtype, devid, "self/self_value/Reshape");

      core::Copy<float>(*k_ptr, *layer_cache["self_keys"],
                        "self/self_key/Copy");
      core::Copy<float>(*v_ptr, *layer_cache["self_values"],
                        "self/self_value/Copy");
    }
  } else {
    TT_THROW("%s is not support in MultiHeadedAttention\n", attn_type);
  }  // if (attn_type == "context")
  // 2) Calculate and scale scores.
  key_seq_length = k_ptr->shape(
      2);  // update for self type attn, since it will concat with cache.
  att_score->Reshape<float>(
      {batch_size, num_attention_heads_, query_seq_length,
       key_seq_length},  // query_seq_length = from_seq_Len
      devtype, devid, "batch_gemm3/Reshape");

  const float scaler = 1.0f / std::sqrt(static_cast<float>(size_per_head));
  kernels::BatchMatMul(*q_ptr, false, *k_ptr, true, scaler, att_score, 0.0,
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
      {batch_size, num_attention_heads_, query_seq_length, size_per_head},
      devtype, devid, "ApplyMaskAndSoftmax/Reshape");

  kernels::BatchMatMul(*att_score, false, *v_ptr, false, 1.0, &context_layer,
                       0.0, "batch_gemm4");
  // context = unshape(context_original)
  core::Tensor self_attr_out(nullptr);

  self_attr_out.Reshape<float>(
      {batch_size, query_seq_length, num_attention_heads_ * size_per_head},
      devtype, devid, "batch_gemm4/Reshape");
  kernels::TransposeForScore(&self_attr_out, context_layer,
                             "TransposeForScore");
  // output = self.final_linear(context)
  output->Reshape<float>({batch_size, query_seq_length, hidden_size}, devtype,
                         devid, "gemm5/Reshape");

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
  profile_ctx.end_profile("MultiHeadedAttention_" + attn_type, devtype);
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
