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

#include "turbo_transformers/layers/bert_attention.h"

#include "loguru.hpp"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"
namespace turbo_transformers {
namespace layers {

void BertAttention::operator()(const core::Tensor& input_tensor,
                               const core::Tensor& attention_mask,
                               core::Tensor* output) const {
  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(
                    input_tensor.device_ctx(), attention_mask.device_ctx()),
                true,
                "The input_tensor and attention_mask should have the same "
                "device type and device id.");

  TT_ENFORCE_EQ(input_tensor.n_dim(), 3,
                "The input ids should be a matrix with shape [BatchSize, "
                "SeqLen, HiddenSize].");

  auto batch_size = input_tensor.shape(0);
  auto seq_length = input_tensor.shape(1);
  auto hidden_size = input_tensor.shape(2);
  auto size_per_head = hidden_size / num_attention_heads_;
  LOG_S(3) << "batch_size: " << batch_size
           << ", num_head: " << num_attention_heads_
           << ", seq_length: " << seq_length << ", hidden_size: " << hidden_size
           << ", size_per_head: " << size_per_head;
  output->Reshape<float>({batch_size, seq_length, hidden_size},
                         input_tensor.device_type(), input_tensor.device_id());

  // 1. temp_qkv = MatMul(input)
  core::TempTensor temp_qkv_tmp;
  core::Tensor& temp_qkv = temp_qkv_tmp.GetTensor(input_tensor.device_ctx());
  temp_qkv.Reshape<float>({3, batch_size, seq_length, hidden_size},
                          input_tensor.device_type(), input_tensor.device_id());
  kernels::MatMul(input_tensor, false, qkv_weight_, false, 1.0, &temp_qkv, 0.0);

  // 2. qkv = transpose(temp_qkv + bias)
  // Since `SplitAddBiasTransposeForScore` does not support inplace,
  // qkv and temp_qkv cannot be same tensor
  core::TempTensor qkv_tensor_tmp;
  core::Tensor& qkv = qkv_tensor_tmp.GetTensor(input_tensor.device_ctx());
  qkv.Reshape<float>(
      {3, batch_size, num_attention_heads_, seq_length, size_per_head},
      input_tensor.device_type(), input_tensor.device_id());

  kernels::SplitAddBiasTransposeForScore(&qkv, temp_qkv, qkv_bias_);

  // 3. q = qkv[0]; k = qkv[1]; v = qkv[2];
  auto q = qkv[0];
  auto k = qkv[1];
  auto v = qkv[2];

  // 4. att_score = softmax((q * k^T)*1/sqrt(size_per_head) + att_mask)
  core::TempTensor att_score_tmp;
  core::Tensor& att_score = att_score_tmp.GetTensor(input_tensor.device_ctx());
  att_score.Reshape<float>(
      {batch_size, num_attention_heads_, seq_length, seq_length},
      input_tensor.device_type(), input_tensor.device_id());

  kernels::BatchMatMul(q, false, k, true, 1.0, &att_score, 0.0);

  kernels::ApplyMaskAndSoftmax(
      &att_score, attention_mask,
      1 / std::sqrt(static_cast<float>(size_per_head)));

  // 5. ctx = v * att_score
  core::TempTensor context_layer_tmpr;
  core::Tensor& context_layer =
      context_layer_tmpr.GetTensor(input_tensor.device_ctx());
  context_layer.Reshape<float>(
      {batch_size, num_attention_heads_, seq_length, size_per_head},
      input_tensor.device_type(), input_tensor.device_id());
  kernels::BatchMatMul(att_score, false, v, false, 1.0, &context_layer, 0.0);

  // 6. self_att_out = transpose(ctx)
  core::TempTensor self_attr_out_tmp;
  core::Tensor& self_attr_out =
      self_attr_out_tmp.GetTensor(input_tensor.device_ctx());
  self_attr_out.Reshape<float>(
      {batch_size, seq_length, num_attention_heads_ * size_per_head},
      input_tensor.device_type(), input_tensor.device_id());

  kernels::TransposeForScore(&self_attr_out, context_layer);

  // 7. output = LayerNorm(MatMul(self_att_out) + Bias)
  kernels::MatMul(self_attr_out, false, dense_weight_, true, 1.0, output, 0.0);

  kernels::AddBiasLayerNorm<float>(input_tensor, dense_bias_,
                                   layer_norm_weight_,  // gemma
                                   layer_norm_bias_, output);
}

void BertAttention::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() > 3) {
    std::ostringstream os;
    layer_norm_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> layer_norm_bias <<<<<<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace turbo_transformers
