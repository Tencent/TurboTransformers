

#pragma once

#include "turbo_transformers/core/tensor.h"
namespace turbo_transformers {
namespace layers {

class PrepareBertMasks {
 public:
  void operator()(const core::Tensor& inputs, core::Tensor* att_mask,
                  core::Tensor* seq_type, core::Tensor* position_ids,
                  core::Tensor* extended_attention_mask) const;
};

}  // namespace layers
}  // namespace turbo_transformers
