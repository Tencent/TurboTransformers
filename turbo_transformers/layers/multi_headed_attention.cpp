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
namespace turbo_transformers {
namespace layers {

static std::mutex mutex_;

// TODO(jiaruifang) call it context att, a type of multiheadedatt
void MultiHeadedAttention::operator()(const core::Tensor& key_tensor,
                                      const core::Tensor& value_tensor,
                                      const core::Tensor& query_tensor,
                                      const core::Tensor& attention_mask,
                                      core::Tensor* output) const {
  std::lock_guard<std::mutex> g(mutex_);
  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(
                    key_tensor.device_ctx(), attention_mask.device_ctx()),
                true,
                "The key_tensor and attention_mask should have the same "
                "device type and device id.");

  TT_ENFORCE_EQ(key_tensor.n_dim(), 3,
                "The key_tensor should be a matrix with shape [BatchSize, "
                "Key_Seq_Len, HiddenSize].");
  TT_ENFORCE_EQ(value_tensor.n_dim(), 3,
                "The value_tensor should be a matrix with shape [BatchSize, "
                "Value_Seq_Len, HiddenSize].");
  TT_ENFORCE_EQ(query_tensor.n_dim(), 3,
                "The query_tensors should be a matrix with shape [BatchSize, "
                "Query_Seq_Len, HiddenSize].");
  TT_ENFORCE_EQ(
      key_tensor.shape(0), value_tensor.shape(0),
      "The key_tensor and value_tensor should have the same hidden_size");

  EnforceShapeAndType();
  auto batch_size = query_tensor.shape(0);
  auto query_seq_length = query_tensor.shape(1);
  auto key_seq_length = key_tensor.shape(1);
  auto hidden_size = query_tensor.shape(2);
  auto size_per_head = hidden_size / num_attention_heads_;

  output->Reshape<float>({batch_size, key_seq_length, hidden_size},
                         query_tensor.device_type(), query_tensor.device_id());

  // 1. q = MatMul(query_tensor), k = MatMul(key_tensor), v =
  // MatMul(value_tensor)
  static core::TempTensor temp_q_tmp;
  core::Tensor& temp_q = temp_q_tmp.GetTensor(query_tensor.device_ctx());
  temp_q.Reshape<float>({batch_size, query_seq_length, hidden_size},
                        query_tensor.device_type(), query_tensor.device_id());

  static core::TempTensor temp_v_tmp;
  core::Tensor& temp_v = temp_v_tmp.GetTensor(value_tensor.device_ctx());
  temp_v.Reshape<float>({batch_size, key_seq_length, hidden_size},
                        value_tensor.device_type(), value_tensor.device_id());

  static core::TempTensor temp_k_tmp;
  core::Tensor& temp_k = temp_k_tmp.GetTensor(key_tensor.device_ctx());
  temp_k.Reshape<float>({batch_size, key_seq_length, hidden_size},
                        key_tensor.device_type(), key_tensor.device_id());

  kernels::MatMul(query_tensor, false, q_weight_, false, 1.0, &temp_q,
                  0.0);  //(B, seq, head*hiddden)
  kernels::MatMul(key_tensor, false, k_weight_, false, 1.0, &temp_k, 0.0);
  kernels::MatMul(value_tensor, false, v_weight_, false, 1.0, &temp_v, 0.0);
  temp_q.Reshape<float>(
      {batch_size, query_seq_length, num_attention_heads_, size_per_head},
      key_tensor.device_type(), key_tensor.device_id());
  temp_k.Reshape<float>(
      {batch_size, key_seq_length, num_attention_heads_, size_per_head},
      key_tensor.device_type(), key_tensor.device_id());
  temp_v.Reshape<float>(
      {batch_size, key_seq_length, num_attention_heads_, size_per_head},
      key_tensor.device_type(), key_tensor.device_id());
  // 2. qkv = transpose(temp_key + bias)
  // Since `SplitAddBiasTransposeForScore` does not support inplace,
  // qkv and temp_qkv cannot be same tensor
  static core::TempTensor q_tensor_tmp, k_tensor_tmp, v_tensor_tmp;
  core::Tensor& q = q_tensor_tmp.GetTensor(query_tensor.device_ctx());
  core::Tensor& k = k_tensor_tmp.GetTensor(key_tensor.device_ctx());
  core::Tensor& v = v_tensor_tmp.GetTensor(value_tensor.device_ctx());
  q.Reshape<float>(
      {batch_size, num_attention_heads_, query_seq_length, size_per_head},
      query_tensor.device_type(), query_tensor.device_id());
  k.Reshape<float>(
      {batch_size, num_attention_heads_, key_seq_length, size_per_head},
      key_tensor.device_type(), key_tensor.device_id());
  v.Reshape<float>(
      {batch_size, num_attention_heads_, key_seq_length, size_per_head},
      value_tensor.device_type(), value_tensor.device_id());
  kernels::AddBiasTransposeForScore(temp_q, q_bias_, &q);
  kernels::AddBiasTransposeForScore(temp_k, k_bias_, &k);
  kernels::AddBiasTransposeForScore(temp_v, v_bias_, &v);

  // 4. attn = self.softmax(scores).to(query.dtype)
  static core::TempTensor att_score_tmp;
  core::Tensor& att_score = att_score_tmp.GetTensor(query_tensor.device_ctx());
  att_score.Reshape<float>({batch_size, num_attention_heads_, query_seq_length,
                            key_seq_length},  // query_seq_length = from_seq_Len
                           query_tensor.device_type(),
                           query_tensor.device_id());
  kernels::BatchMatMul(q, false, k, true, 1.0, &att_score, 0.0);

  kernels::ApplyMaskAndSoftmax(
      &att_score, attention_mask,
      1 / std::sqrt(static_cast<float>(size_per_head)));

  // context_original = torch.matmul(drop_attn, value)
  static core::TempTensor context_layer_tmpr;
  core::Tensor& context_layer =
      context_layer_tmpr.GetTensor(query_tensor.device_ctx());
  context_layer.Reshape<float>(
      {batch_size, num_attention_heads_, query_seq_length, size_per_head},
      query_tensor.device_type(), query_tensor.device_id());
  kernels::BatchMatMul(att_score, false, v, false, 1.0, &context_layer, 0.0);

  // context = unshape(context_original)
  static core::TempTensor self_attr_out_tmp;
  core::Tensor& self_attr_out =
      self_attr_out_tmp.GetTensor(query_tensor.device_ctx());
  self_attr_out.Reshape<float>(
      {batch_size, key_seq_length, num_attention_heads_ * size_per_head},
      query_tensor.device_type(), query_tensor.device_id());

  kernels::TransposeForScore(&self_attr_out, context_layer);

  // output = self.final_linear(context)
  kernels::MatMul(self_attr_out, false, dense_weight_, false, 1.0, output, 0.0);
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
