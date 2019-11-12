#pragma once
#include <glog/logging.h>
#include <memory>
#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {

class BertSelfAttention {
 public:
  BertSelfAttention(core::Tensor qkv_weight, core::Tensor qkv_bias,
                    core::Tensor dense_weight, core::Tensor dense_bias,
                    core::Tensor layer_norm_weight,
                    core::Tensor layer_norm_bias,
                    int64_t num_attention_heads)
      : qkv_weight_(std::move(qkv_weight)),  //(768, 768)
        qkv_bias_(std::move(qkv_bias)),
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
  core::Tensor qkv_weight_;
  core::Tensor qkv_bias_;
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;
  core::Tensor layer_norm_weight_;
  core::Tensor layer_norm_bias_;
  int64_t num_attention_heads_;
};

}  // namespace layers
}  // namespace fast_transformers
