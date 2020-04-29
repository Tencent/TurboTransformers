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
                                      const std::string& attn_type,
                                      core::Tensor* output) const {
  std::lock_guard<std::mutex> g(mutex_);
  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(
                    key_tensor.device_ctx(), attention_mask.device_ctx()),
                true,
                "The key_tensor and attention_mask should have the same "
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
  }
  auto hidden_size = query_tensor.shape(2);
  auto size_per_head = hidden_size / num_attention_heads_;
  auto query_device_type = query_tensor.device_type();
  auto query_device_id = query_tensor.device_id();

  output->Reshape<float>({batch_size, query_seq_length, hidden_size},
                         query_device_type, query_device_id);

  // 1. q = MatMul(query_tensor), k = MatMul(key_tensor), v =
  // MatMul(value_tensor)
  core::Tensor *q_ptr{nullptr}, *k_ptr{nullptr}, *v_ptr{nullptr};
  // 4. attn = self.softmax(scores).to(query.dtype)

  if (attn_type == "context") {
    static core::TempTensor q_out1_temp, v_out1_temp,
        k_out1_temp;  // intermediate results after matmul
    static core::TempTensor q_out2_temp, v_out2_temp,
        k_out2_temp;  // intermediate results after add bias and transpose
    core::Tensor& q_out1 = q_out1_temp.GetTensor(query_tensor.device_ctx());
    core::Tensor& v_out1 = v_out1_temp.GetTensor(value_tensor.device_ctx());
    core::Tensor& k_out1 = k_out1_temp.GetTensor(key_tensor.device_ctx());
    core::Tensor& q_out2 = q_out2_temp.GetTensor(query_tensor.device_ctx());
    core::Tensor& v_out2 = v_out2_temp.GetTensor(value_tensor.device_ctx());
    core::Tensor& k_out2 = k_out2_temp.GetTensor(key_tensor.device_ctx());

    q_out1.Reshape<float>({batch_size, query_seq_length, hidden_size},
                          query_device_type, query_device_id);
    v_out1.Reshape<float>({batch_size, key_seq_length, hidden_size},
                          query_device_type, query_device_id);
    k_out1.Reshape<float>({batch_size, key_seq_length, hidden_size},
                          query_device_type, query_device_id);

    kernels::MatMul(query_tensor, false, q_weight_, false, 1.0, &q_out1, 0.0);
    kernels::MatMul(key_tensor, false, k_weight_, false, 1.0, &v_out1, 0.0);
    kernels::MatMul(value_tensor, false, v_weight_, false, 1.0, &k_out1, 0.0);

    q_out1.Reshape<float>(
        {batch_size, query_seq_length, num_attention_heads_, size_per_head},
        query_device_type, query_device_id);
    v_out1.Reshape<float>(
        {batch_size, key_seq_length, num_attention_heads_, size_per_head},
        query_device_type, query_device_id);
    k_out1.Reshape<float>(
        {batch_size, key_seq_length, num_attention_heads_, size_per_head},
        query_device_type, query_device_id);

    q_out2.Reshape<float>(
        {batch_size, num_attention_heads_, query_seq_length, size_per_head},
        query_device_type, query_device_id);
    v_out2.Reshape<float>(
        {batch_size, num_attention_heads_, key_seq_length, size_per_head},
        query_device_type, query_device_id);
    k_out2.Reshape<float>(
        {batch_size, num_attention_heads_, key_seq_length, size_per_head},
        query_device_type, query_device_id);

    kernels::AddBiasTransposeForScore(q_out1, q_bias_, &q_out2);
    kernels::AddBiasTransposeForScore(v_out1, k_bias_, &v_out2);
    kernels::AddBiasTransposeForScore(k_out1, v_bias_, &k_out2);
    q_ptr = &q_out2;  // point to static memory space
    v_ptr = &v_out2;
    k_ptr = &k_out2;
  } else if (attn_type == "self") {
    static core::TempTensor qkv_out1_temp, qkv_out2_temp;
    core::Tensor& qkv_out1 = qkv_out1_temp.GetTensor(query_tensor.device_ctx());
    qkv_out1.Reshape<float>({3, batch_size, query_seq_length, hidden_size},
                            query_device_type, query_device_id);

    kernels::MatMul(query_tensor, false, qkv_weight_, false, 1.0, &qkv_out1,
                    0.0);

    core::Tensor& qkv_out2 = qkv_out2_temp.GetTensor(query_tensor.device_ctx());
    qkv_out2.Reshape<float>(
        {3, batch_size, num_attention_heads_, query_seq_length, size_per_head},
        query_device_type, query_device_id);

    kernels::SplitAddBiasTransposeForScore(&qkv_out2, qkv_out1, qkv_bias_);
    q_ptr =
        new core::Tensor(qkv_out2[0]);  // copy temporary tensor to heap space.
    k_ptr = new core::Tensor(qkv_out2[1]);
    v_ptr = new core::Tensor(qkv_out2[2]);
    qkv_out2[0].Print<float>(std::cerr);
  } else {
    TT_THROW("%s is not support in MultiHeadedAttention\n", attn_type);
  }

  static core::TempTensor att_score_tmp;
  core::Tensor& att_score = att_score_tmp.GetTensor(query_tensor.device_ctx());
  att_score.Reshape<float>({batch_size, num_attention_heads_, query_seq_length,
                            key_seq_length},  // query_seq_length = from_seq_Len
                           query_device_type, query_device_id);
  kernels::BatchMatMul(*q_ptr, false, *k_ptr, true, 1.0, &att_score, 0.0);

  kernels::ApplyMaskAndSoftmax(
      &att_score, attention_mask,
      1 / std::sqrt(static_cast<float>(size_per_head)));

  // context_original = torch.matmul(drop_attn, value)
  static core::TempTensor context_layer_tmpr;
  core::Tensor& context_layer =
      context_layer_tmpr.GetTensor(query_tensor.device_ctx());
  context_layer.Reshape<float>(
      {batch_size, num_attention_heads_, query_seq_length, size_per_head},
      query_device_type, query_device_id);
  kernels::BatchMatMul(att_score, false, *v_ptr, false, 1.0, &context_layer,
                       0.0);

  // context = unshape(context_original)
  static core::TempTensor self_attr_out_tmp;
  core::Tensor& self_attr_out =
      self_attr_out_tmp.GetTensor(query_tensor.device_ctx());

  self_attr_out.Reshape<float>(
      {batch_size, query_seq_length, num_attention_heads_ * size_per_head},
      query_device_type, query_device_id);

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
