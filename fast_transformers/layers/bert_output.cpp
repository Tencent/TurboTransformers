#include "fast_transformers/layers/bert_output.h"

#include <loguru.hpp>

#include "fast_transformers/core/memory.h"
#include "fast_transformers/layers/kernels/layer_norm.h"

namespace fast_transformers {
namespace layers {

// class BertOutput(nn.Module):
//     def __init__(self, config):
//         super(BertOutput, self).__init__()
//         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
//         self.LayerNorm = BertLayerNorm(config.hidden_size,
//         eps=config.layer_norm_eps) self.dropout =
//         nn.Dropout(config.hidden_dropout_prob)

//     def forward(self, hidden_states, input_tensor):
//         hidden_states = self.dense(hidden_states)
//         hidden_states = self.dropout(hidden_states)
//         hidden_states = self.LayerNorm(hidden_states + input_tensor)
//         return hidden_states
void BertOutput::operator()(const core::Tensor &hidden_states,
                            const core::Tensor &input_tensor,
                            core::Tensor *output_tensor) const {
  auto batch_size = hidden_states.shape(0);
  auto seq_length = hidden_states.shape(1);
  auto intermediate_size = hidden_states.shape(2);
  auto hidden_size = input_tensor.shape(2);

  auto m = batch_size * seq_length;
  auto k = intermediate_size;
  auto n = hidden_size;

  static constexpr float alpha = 1., beta = 0.;

  FT_ENFORCE(output_tensor, "The output tensor should not be nullptr.");
  output_tensor->Reshape<float>({batch_size, seq_length, hidden_size});

  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
      hidden_states.data<float>(),
      k,  // (batch_size*seq_length = m, intermediate_size = k)
      dense_weight_.data<float>(), k,
      beta,  //[hidden_size = n, intermediate_size = k] please transpose here!x
      output_tensor->mutableData<float>(),
      n);  // (batch_size*seq_length = m, hidden_size = n)

  kernels::AddBiasLayerNorm<float>(input_tensor, dense_bias_,
                                   layer_norm_weight_, layer_norm_bias_,
                                   output_tensor);
}

void BertOutput::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::stringstream ss;
    ss << "<<<<<<<< dense_weight_ <<<<<<<<<<";
    dense_weight_.Print<float>(ss);
    ss << "<<<<<<<< dense_bias <<<<<<<<<<";
    dense_bias_.Print<float>(ss);
    ss << "<<<<<<<< layer_norm_weight <<<<<<<<<<";
    layer_norm_weight_.Print<float>(ss);
    ss << "<<<<<<<< layer_norm_bias <<<<<<<<<<";
    layer_norm_bias_.Print<float>(ss);
    LOG_S(3) << ss.str();
  }
}

}  // namespace layers
}  // namespace fast_transformers
