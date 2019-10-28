#include "fast_transformers/core/tensor.h"
#include <memory>

namespace fast_transformers {
namespace kernels {

class BERTEmbedding {
public:
  BERTEmbedding(core::Tensor word_embeddings, core::Tensor position_embeddings,
                core::Tensor token_type_embeddings,
                core::Tensor layer_norm_weights, core::Tensor layer_norm_bias,
                float dropout_rate)
      : word_embedings_(std::move(word_embeddings)),
        position_embeddings_(std::move(position_embeddings)),
        token_type_embeddings_(std::move(token_type_embeddings)),
        layer_norm_weights_(std::move(layer_norm_weights)),
        layer_norm_bias_(std::move(layer_norm_bias)),
        dropout_rate_(dropout_rate) {}

  core::Tensor operator()(const core::Tensor &input_ids,
                          const core::Tensor &attention_masks,
                          const core::Tensor &position_ids) const;

private:
  core::Tensor word_embedings_;
  core::Tensor position_embeddings_;
  core::Tensor token_type_embeddings_;
  core::Tensor layer_norm_weights_;
  core::Tensor layer_norm_bias_;
  float dropout_rate_;
};

} // namespace layers
} // namespace fast_transformers
