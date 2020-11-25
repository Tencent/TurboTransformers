

#pragma once
#include <memory>
#include <utility>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {

class FusedAddBiasGELU {
 public:
  FusedAddBiasGELU(core::Tensor dense_bias) : bias(std::move(dense_bias)) {}

  void operator()(core::Tensor* output) const;

 private:
  core::Tensor bias;
};

}  // namespace layers
}  // namespace turbo_transformers
