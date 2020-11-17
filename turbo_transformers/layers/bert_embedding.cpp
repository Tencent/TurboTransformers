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

#include "turbo_transformers/layers/bert_embedding.h"

#include "loguru.hpp"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/embedding.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"

namespace turbo_transformers {
namespace layers {

void BERTEmbedding::operator()(const core::Tensor &input_ids,
                               const core::Tensor &position_ids,
                               const core::Tensor &token_type_ids,
                               core::Tensor *output_tensor) const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> input_ids <<<<<<<<<<<<" << std::endl;
    input_ids.Print<int64_t>(os);
    os << ">>>>>>>>>>>> position_ids <<<<<<<<<<<<" << std::endl;
    position_ids.Print<int64_t>(os);
    os << ">>>>>>>>>>>> token_type_ids <<<<<<<<<<<<" << std::endl;
    token_type_ids.Print<int64_t>(os);
    LOG_S(3) << os.str();
  }

  TT_ENFORCE_EQ(
      input_ids.n_dim(), 2,
      "The input ids should be a matrix with shape [BatchSize, SeqLen].");
  auto batch_size = input_ids.shape(0);
  auto seq_length = input_ids.shape(1);
  // TODO 1. switch DeviceType::CPU 2. how should I set stride?
  auto hidden_size = word_embedings_.shape(1);

  TT_ENFORCE(output_tensor, "The output tensor should not be nullptr.");
  output_tensor->Reshape<float>({batch_size, seq_length, hidden_size},
                                input_ids.device_type(), input_ids.device_id(),
                                "BERTEmbedding/Reshape");
  LOG_S(3) << "Look up word embedding";

  kernels::LookupEmbedding</*Add=*/false>(output_tensor, word_embedings_,
                                          input_ids);

  LOG_S(3) << "Look up token type embedding";
  kernels::LookupEmbedding</*Add=*/true>(output_tensor, token_type_embeddings_,
                                         token_type_ids);
  LOG_S(3) << "Look up token position embedding";
  kernels::LookupEmbedding</*Add=*/true>(output_tensor, position_embeddings_,
                                         position_ids);

  kernels::LayerNorm<float>(layer_norm_weights_, layer_norm_bias_,
                            output_tensor);
}
void BERTEmbedding::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>> word_embedings_ <<<<<<<<" << std::endl;
    word_embedings_.Print<float>(os);
    os << ">>>>> position_embeddings_ <<<<<<<<" << std::endl;
    position_embeddings_.Print<float>(os);
    os << ">>>>> token_type_embeddings_ <<<<<<<<" << std::endl;
    token_type_embeddings_.Print<float>(os);
    os << ">>>>> layer_norm_weights_ <<<<<<<<" << std::endl;
    layer_norm_weights_.Print<float>(os);
    os << ">>>>> layer_norm_bias_ <<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}
}  // namespace layers
}  // namespace turbo_transformers
