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

#include "turbo_transformers/layers/bert_output.h"

#include <loguru.hpp>

#include "turbo_transformers/core/common.h"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"

namespace turbo_transformers {
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
  TT_ENFORCE_EQ(core::common::is_same_device_ctx(input_tensor.device_ctx(),
                                                 hidden_states.device_ctx()),
                true,
                "BertOutput: The input_tensor and hidden_states should have "
                "the same device type and device id.");
  output_tensor->Reshape<float>(
      {hidden_states.shape(0), hidden_states.shape(1), dense_weight_.shape(0)},
      hidden_states.device_type(), hidden_states.device_id());
  kernels::MatMul(hidden_states, false, dense_weight_, true, 1.0, output_tensor,
                  0.0);
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
}  // namespace turbo_transformers
