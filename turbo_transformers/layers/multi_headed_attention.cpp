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
  profile_ctx.start_profile("MultiHeadedAttention_" + attn_type);
#endif
  std::lock_guard<std::mutex> g(mutex_);
  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(
                    query_tensor.device_ctx(), attention_mask.device_ctx()),
                true,
                "The query_tensor and attention_mask should have the same "
                "device type and device id.");

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
  auto devctx = query_tensor.device_ctx();

  // TODO we should caching allocate intermediate tensor.
  core::Tensor *q_ptr{nullptr}, *k_ptr{nullptr}, *v_ptr{nullptr};
  core::Tensor q_out1(nullptr);
  core::Tensor v_out1(nullptr);
  core::Tensor k_out1(nullptr);
  core::Tensor q_out2(nullptr);
  core::Tensor v_out2(nullptr);
  core::Tensor k_out2(nullptr);
  core::Tensor qkv_out1(nullptr);
  core::Tensor qkv_out2(nullptr);

  {
    if (attn_type == "context") {
#ifdef WITH_PERFTOOLS
      profile_ctx.start_profile("gemm_012+AddBiasTransposeForScore3");
      profile_ctx.start_profile("set_cache_flags");
#endif
      bool layer_cache_not_none = layer_cache.size() > 0 ? true : false;
      bool memory_keys_not_none = false, memory_values_not_none = false;
      if (layer_cache_not_none) {
        for (auto it = layer_cache.begin(); it != layer_cache.end(); ++it) {
          if (it->first == "memory_keys" && !it->second->is_null()) {
            memory_keys_not_none = true;
          }
          if (it->first == "memory_values" && !it->second->is_null()) {
            memory_values_not_none = true;
          }
        }
      }
      bool memory_not_none = memory_values_not_none && memory_keys_not_none;
#ifdef WITH_PERFTOOLS
      profile_ctx.end_profile("set_cache_flags");
#endif
      TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(
                        query_tensor.device_ctx(), value_tensor.device_ctx()),
                    true,
                    "The query_tensor and value_tensor should have the same "
                    "device type and device id.");
      TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(
                        query_tensor.device_ctx(), key_tensor.device_ctx()),
                    true,
                    "The query_tensor and key_tensor should have the same "
                    "device type and device id.");

      q_out1.Reshape<float>({batch_size, query_seq_length, hidden_size},
                            devtype, devid);
      if (pre_layernorm) {
        q_out2.Reshape<float>({batch_size, query_seq_length, hidden_size},
                              devtype, devid);
        core::Copy<float>(query_tensor, q_out2);
        kernels::LayerNorm<float>(
            layernorm_gamma_, layernorm_beta_, &q_out2,
            1e-6);  // q_out2 here is used as layernormed_query TODO(jiaruifang)
                    // 1e-6 should not be hard-coded
        kernels::MatMul(q_out2, false, q_weight_, is_trans_weight, 1.0, &q_out1,
                        0.0);
      } else {
        kernels::MatMul(query_tensor, false, q_weight_, is_trans_weight, 1.0,
                        &q_out1, 0.0);
      }
      q_out1.Reshape<float>(
          {batch_size, query_seq_length, num_attention_heads_, size_per_head},
          devtype, devid);
      q_out2.Reshape<float>(
          {batch_size, num_attention_heads_, query_seq_length, size_per_head},
          devtype, devid);
      kernels::AddBiasTransposeForScore(q_out1, q_bias_, &q_out2);
      q_ptr = &q_out2;  // point to static memory space

      if (memory_not_none) {
        v_ptr = layer_cache["memory_values"];
        k_ptr = layer_cache["memory_keys"];
      } else {
        v_out1.Reshape<float>({batch_size, key_seq_length, hidden_size},
                              devtype, devid);
        k_out1.Reshape<float>({batch_size, key_seq_length, hidden_size},
                              devtype, devid);

        kernels::MatMul(key_tensor, false, k_weight_, is_trans_weight, 1.0,
                        &k_out1, 0.0);
        kernels::MatMul(value_tensor, false, v_weight_, is_trans_weight, 1.0,
                        &v_out1, 0.0);
        v_out1.Reshape<float>(
            {batch_size, key_seq_length, num_attention_heads_, size_per_head},
            devtype, devid);
        k_out1.Reshape<float>(
            {batch_size, key_seq_length, num_attention_heads_, size_per_head},
            devtype, devid);

        if (layer_cache_not_none) {
          layer_cache["memory_keys"]->Reshape<float>(
              {batch_size, num_attention_heads_, key_seq_length, size_per_head},
              devtype, devid);
          layer_cache["memory_values"]->Reshape<float>(
              {batch_size, num_attention_heads_, key_seq_length, size_per_head},
              devtype, devid);
          kernels::AddBiasTransposeForScore(v_out1, v_bias_,
                                            layer_cache["memory_values"]);
          kernels::AddBiasTransposeForScore(k_out1, k_bias_,
                                            layer_cache["memory_keys"]);
          v_ptr = layer_cache["memory_values"];
          k_ptr = layer_cache["memory_keys"];
        } else {
          v_out2.Reshape<float>(
              {batch_size, num_attention_heads_, key_seq_length, size_per_head},
              devtype, devid);
          k_out2.Reshape<float>(
              {batch_size, num_attention_heads_, key_seq_length, size_per_head},
              devtype, devid);
          kernels::AddBiasTransposeForScore(v_out1, v_bias_, &v_out2);
          kernels::AddBiasTransposeForScore(k_out1, k_bias_, &k_out2);
          v_ptr = &v_out2;
          k_ptr = &k_out2;
        }
      }  // else

#ifdef WITH_PERFTOOLS
      profile_ctx.end_profile("gemm_012+AddBiasTransposeForScore3");
#endif
    } else if (attn_type == "self") {
      qkv_out1.Reshape<float>({3, batch_size, query_seq_length, hidden_size},
                              devtype, devid);

#ifdef WITH_PERFTOOLS
      profile_ctx.start_profile("gemm_fused");
#endif
      if (pre_layernorm) {
        core::Tensor layernormed_query(nullptr);
        layernormed_query.Reshape<float>(
            {batch_size, query_seq_length, hidden_size}, devtype, devid);
        core::Copy<float>(query_tensor, layernormed_query);
        kernels::LayerNorm<float>(layernorm_gamma_, layernorm_beta_,
                                  &layernormed_query, 1e-6);
        kernels::MatMul(layernormed_query, false, qkv_weight_, is_trans_weight,
                        1.0, &qkv_out1, 0.0);
      } else {
        kernels::MatMul(query_tensor, false, qkv_weight_, is_trans_weight, 1.0,
                        &qkv_out1, 0.0);
      }

#ifdef WITH_PERFTOOLS
      profile_ctx.end_profile("gemm_fused");
      profile_ctx.start_profile("SplitAddBiasTransposeForScore");
#endif
      qkv_out2.Reshape<float>({3, batch_size, num_attention_heads_,
                               query_seq_length, size_per_head},
                              devtype, devid);
      kernels::SplitAddBiasTransposeForScore(&qkv_out2, qkv_out1, qkv_bias_);
      q_ptr = new core::Tensor(
          qkv_out2[0]);  // copy temporary tensor to heap space.
      k_ptr = new core::Tensor(qkv_out2[1]);
      v_ptr = new core::Tensor(qkv_out2[2]);
#ifdef WITH_PERFTOOLS
      profile_ctx.end_profile("SplitAddBiasTransposeForScore");
#endif
    } else {
      TT_THROW("%s is not support in MultiHeadedAttention\n", attn_type);
    }
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.start_profile("batch_gemm0");
#endif
  // 2) Calculate and scale scores.
  // static core::TempTensor att_score_tmp;
  // core::Tensor& att_score = att_score_tmp.GetTensor(devctx);
  att_score->Reshape<float>(
      {batch_size, num_attention_heads_, query_seq_length,
       key_seq_length},  // query_seq_length = from_seq_Len
      devtype, devid);

#ifdef WITH_GPERFTOOLS
  profile_ctx.start_profile("batch_gemm0");
#endif

  const float scaler = 1.0f / std::sqrt(static_cast<float>(size_per_head));
  kernels::BatchMatMul(*q_ptr, false, *k_ptr, true, scaler, att_score,
                       0.0);  //(B, num_head, q_len, k_len)

#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("batch_gemm0");
#endif
  // mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
  // scores = scores.masked_fill(mask, -1e18)
  // attn = self.softmax(scores).to(query.dtype)
#ifdef WITH_PERFTOOLS
  profile_ctx.start_profile("ApplyMaskAndSoftmax");
#endif
  kernels::ApplyMaskAndSoftmax(
      att_score,
      attention_mask,  //(B, q_len, k_len) or (B, 1, k_len)
      1.0);

#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("ApplyMaskAndSoftmax");
#endif
  // context_original = torch.matmul(drop_attn, value)
  static core::TempTensor context_layer_tmpr;
  core::Tensor& context_layer = context_layer_tmpr.GetTensor(devctx);
  context_layer.Reshape<float>(
      {batch_size, num_attention_heads_, query_seq_length, size_per_head},
      devtype, devid);

#ifdef WITH_PERFTOOLS
  profile_ctx.start_profile("batch_gemm1");
#endif

  kernels::BatchMatMul(*att_score, false, *v_ptr, false, 1.0, &context_layer,
                       0.0);

#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("batch_gemm1");
  profile_ctx.start_profile("TransposeForScore");
#endif
  // context = unshape(context_original)
  static core::TempTensor self_attr_out_tmp;
  core::Tensor& self_attr_out = self_attr_out_tmp.GetTensor(devctx);

  self_attr_out.Reshape<float>(
      {batch_size, query_seq_length, num_attention_heads_ * size_per_head},
      devtype, devid);
  kernels::TransposeForScore(&self_attr_out, context_layer);
  // output = self.final_linear(context)
  output->Reshape<float>({batch_size, query_seq_length, hidden_size}, devtype,
                         devid);
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("TransposeForScore");
  profile_ctx.start_profile("gemm5");
#endif
  kernels::MatMul(self_attr_out, false, dense_weight_, is_trans_weight, 1.0,
                  output, 0.0);

#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("gemm5");
  profile_ctx.start_profile("AddBias");
#endif
  if (false == post_add_input) {
    if (false == post_layernorm) {
      //+bias
      kernels::AddBias(dense_bias_, output);
    } else {
      //+bias+layernorm
      kernels::AddBiasLayerNorm<float>(query_tensor, dense_bias_,
                                       layernorm_gamma_,  // gemma
                                       layernorm_beta_, output);
    }
  } else {
    //+input + bias
    kernels::AddInputBias(*output, query_tensor, dense_bias_, output);
  }

#ifdef __DEBUG__
  std::cerr << "let's see layer_cache in CPP" << std::endl;
  for (auto it = layer_cache.begin(); it != layer_cache.end(); ++it) {
    std::cerr << it->first << std::endl;
    if (it->second->is_null()) {
      std::cerr << "None" << std::endl;
    } else {
      it->second->Print<float>(std::cerr);
    }
  }
#endif

#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("AddBias");
  profile_ctx.end_profile("MultiHeadedAttention_" + attn_type);
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
