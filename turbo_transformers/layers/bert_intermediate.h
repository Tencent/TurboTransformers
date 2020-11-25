

#pragma once
#include <memory>
#include <utility>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {

class BertIntermediate {
 public:
  BertIntermediate(core::Tensor dense_weight, core::Tensor dense_bias)
      : dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)) {
    EnforceShapeAndType();
  }

  void EnforceShapeAndType() const;
  void operator()(const core::Tensor& input_tensor, core::Tensor* output) const;

 private:
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;
};

}  // namespace layers
}  // namespace turbo_transformers
