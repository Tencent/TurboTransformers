#pragma once
#include <memory>
#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {

class BertAttention {
 public:
  BertAttention(core::Tensor query_weight, core::Tensor query_bias,
                core::Tensor key_weight, core::Tensor key_bias,
                core::Tensor value_weight, core::Tensor value_bias,
                core::Tensor dense_weight, core::Tensor dense_bias,
                core::Tensor layer_norm_weight, core::Tensor layer_norm_bias,
                int64_t num_attention_heads)
      : query_weight_(std::move(query_weight)),  //(768, 768)
        query_bias_(std::move(query_bias)),      //(768)
        key_weight_(std::move(key_weight)),
        key_bias_(std::move(key_bias)),
        value_weight_(std::move(value_weight)),
        value_bias_(std::move(value_bias)),
        dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)),
        layer_norm_weight_(std::move(layer_norm_weight)),  //(768)
        layer_norm_bias_(std::move(layer_norm_bias)),
        num_attention_heads_(num_attention_heads) {
    EnforceShapeAndType();
  }
  void EnforceShapeAndType() const;

  core::Tensor operator()(const core::Tensor &input_tensor,
                          const core::Tensor &attention_mask,
                          const core::Tensor &head_mask) const;

 private:
  core::Tensor query_weight_;
  core::Tensor query_bias_;
  core::Tensor key_weight_;
  core::Tensor key_bias_;
  core::Tensor value_weight_;
  core::Tensor value_bias_;
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;
  core::Tensor layer_norm_weight_;
  core::Tensor layer_norm_bias_;
  int64_t num_attention_heads_;
};

}  // namespace layers
}  // namespace fast_transformers
