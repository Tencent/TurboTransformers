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

#pragma once
#include <memory>

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/loaders/npz_load.h"

namespace turbo_transformers {
namespace layers {

class BERTEmbedding {
 public:
  BERTEmbedding(loaders::NPZMapView root, const std::string prefix,
                DLDeviceType dev)
      : word_embedings_(std::move(nullptr)),  // [vocab_size, hidden_size]
        position_embeddings_(
            std::move(nullptr)),  // [max_position_embeddings, hidden_size]
        token_type_embeddings_(
            std::move(nullptr)),  // [token_type_vocab_size, hidden_size]
        layer_norm_weights_(std::move(nullptr)),  // [hidden_size]
        layer_norm_bias_(std::move(nullptr)),     // [hidden_size]
        dropout_rate_(0) {
    //  params(std::move(npz), dev);
    auto npz = root.Sub(prefix);
    loaders::NPZLoader params(std::move(npz), dev);
    word_embedings_ = std::move(params["word_embeddings.weight"]);
    position_embeddings_ = std::move(params["position_embeddings.weight"]);
    token_type_embeddings_ = std::move(params["token_type_embeddings.weight"]);
    layer_norm_weights_ = std::move(params["LayerNorm.weight"]);
    layer_norm_bias_ = std::move(params["LayerNorm.bias"]);
    dropout_rate_ = 0;
  }

  BERTEmbedding(core::Tensor word_embeddings, core::Tensor position_embeddings,
                core::Tensor token_type_embeddings,
                core::Tensor layer_norm_weights, core::Tensor layer_norm_bias,
                float dropout_rate)
      : word_embedings_(
            std::move(word_embeddings)),  // [vocab_size, hidden_size]
        position_embeddings_(std::move(
            position_embeddings)),  // [max_position_embeddings, hidden_size]
        token_type_embeddings_(std::move(
            token_type_embeddings)),  // [token_type_vocab_size, hidden_size]
        layer_norm_weights_(std::move(layer_norm_weights)),  // [hidden_size]
        layer_norm_bias_(std::move(layer_norm_bias)),        // [hidden_size]
        dropout_rate_(dropout_rate) {
    EnforceShapeAndType();
  }

  void EnforceShapeAndType() const;

  void operator()(const core::Tensor &input_ids,
                  const core::Tensor &position_ids,
                  const core::Tensor &token_type_ids,
                  core::Tensor *output) const;

 private:
  core::Tensor word_embedings_;
  core::Tensor position_embeddings_;
  core::Tensor token_type_embeddings_;
  core::Tensor layer_norm_weights_;
  core::Tensor layer_norm_bias_;
  float dropout_rate_;
};

}  // namespace layers
}  // namespace turbo_transformers
