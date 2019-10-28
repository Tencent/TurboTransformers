#include "bert_embedding.h"

namespace fast_transformers {
namespace layers {
core::Tensor BERTEmbedding::operator()(const core::Tensor& input_ids, const core::Tensor& attention_masks, const core::Tensor& position_ids) const{
  return core::Tensor(nullptr);
}
}
}
