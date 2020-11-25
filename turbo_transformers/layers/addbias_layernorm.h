

#pragma once
#include <memory>
#include <utility>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {

class FusedAddBiasLayerNorm {
 public:
  FusedAddBiasLayerNorm(core::Tensor dense_bias, core::Tensor layer_norm_weight,
                        core::Tensor layer_norm_bias)
      : bias(std::move(dense_bias)),
        norm_weight(std::move(layer_norm_weight)),
        norm_bias(std::move(layer_norm_bias)) {}

  void operator()(const core::Tensor& input_tensor, core::Tensor* output) const;

 private:
  core::Tensor bias;
  core::Tensor norm_weight;
  core::Tensor norm_bias;
};

}  // namespace layers
}  // namespace turbo_transformers
