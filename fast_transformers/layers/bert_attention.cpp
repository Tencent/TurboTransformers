#include "fast_transformers/layers/bert_attention.h"

#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/memory.h"
#include "fast_transformers/layers/kernels/layer_norm.h"
#include "fast_transformers/layers/kernels/mat_mul.h"
#include "fast_transformers/layers/kernels/softmax.h"
#include "fast_transformers/layers/kernels/transpose.h"
#include "loguru.hpp"
namespace fast_transformers {
namespace layers {

void BertAttention::operator()(const core::Tensor& input_tensor,
                               const core::Tensor& attention_mask,
                               core::Tensor* output) const {
  FT_ENFORCE_EQ(input_tensor.n_dim(), 3,
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

  static core::Tensor query_buffer(nullptr);
  query_buffer.Reshape<float>({3, batch_size, seq_length, hidden_size});
  static core::Tensor qkv_tensor(nullptr);
  qkv_tensor.Reshape<float>(
      {3, batch_size, num_attention_heads_, seq_length, size_per_head});

  static core::Tensor attention_scores_tensor(nullptr);
  attention_scores_tensor.Reshape<float>(
      {batch_size, num_attention_heads_, seq_length, seq_length});

  static core::Tensor context_layer(nullptr);
  context_layer.Reshape<float>(
      {batch_size, num_attention_heads_, seq_length, size_per_head});

  FT_ENFORCE(output, "The output tensor should not be nullptr.");
  output->Reshape<float>({batch_size, seq_length, hidden_size});

  kernels::MatMul(input_tensor, false, qkv_weight_, true, 1.0, &query_buffer,
                  0.0);

  kernels::SplitAddBiasTransposeForScore(&qkv_tensor, query_buffer, qkv_bias_);

  auto q_tensor = qkv_tensor[0];
  auto k_tensor = qkv_tensor[1];
  auto v_tensor = qkv_tensor[2];

  // attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
  kernels::BatchMatMul(q_tensor, false, k_tensor, true, 1.0,
                       &attention_scores_tensor, 0.0);

  kernels::ApplyMaskAndSoftmax(&attention_scores_tensor, attention_mask,
                               1 / std::sqrt(size_per_head * 1.0f));

  // preprocess: attention_mask * -10000
  // attention_probs = self.dropout(attention_probs)
  // TODO

  // context_layer = torch.matmul(attention_probs, value_layer) -> context_layer
  kernels::BatchMatMul(attention_scores_tensor, false, v_tensor, false, 1.0,
                       &context_layer, 0.0);

  static core::Tensor self_attr_out(nullptr);
  self_attr_out.Reshape<float>(
      {batch_size, seq_length, num_attention_heads_ * size_per_head});

  // context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
  // new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
  // context_layer = context_layer.view(*new_context_layer_shape)
  kernels::TransposeForScore(&self_attr_out, context_layer);

  // self_outputs = (context_layer, attention_probs) if self.output_attentions
  // else (context_layer,) # self.output_attentions is nullptr hidden_states =
  // self.dense(self_outputs[0]) #context_layer
  kernels::MatMul(self_attr_out, false, dense_weight_, true, 1.0, output, 0.0);

  // attention_output = self.LayerNorm(hidden_states + input_tensor)
  kernels::AddBiasLayerNorm<float>(input_tensor, dense_bias_,
                                   layer_norm_weight_,  // gemma
                                   layer_norm_bias_, output);
  // outputs = (attention_output,) + self_outputs[1:]
}

void BertAttention::EnforceShapeAndType() const {
  FT_ENFORCE_EQ(layer_norm_weight_.device_type(), kDLCPU,
                "Only CPU supportted");
  if (loguru::current_verbosity_cutoff() > 3) {
    std::ostringstream os;
    layer_norm_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> layer_norm_bias <<<<<<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace fast_transformers
