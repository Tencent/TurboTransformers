#pragma once
#include <cstdint>

#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {
extern void ApplyMaskAndSoftmax(core::Tensor* inout,
                                const core::Tensor& att_mask, float scale);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
