#pragma once
#include <memory>
#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {

class BertOutput {
 public:
  BertOutput(core::Tensor dense_weight, core::Tensor dense_bias,
             core::Tensor layer_norm_weight, core::Tensor layer_norm_bias)
      : dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)),
        layer_norm_weight_(std::move(layer_norm_weight)),
        layer_norm_bias_(std::move(layer_norm_bias)) {
    EnforceShapeAndType();
  }
  void EnforceShapeAndType() const;

  core::Tensor operator()(const core::Tensor &hidden_states,
                          const core::Tensor &input_tensor) const;

 private:
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;
  core::Tensor layer_norm_weight_;
  core::Tensor layer_norm_bias_;
};

}  // namespace layers
}  // namespace fast_transformers
