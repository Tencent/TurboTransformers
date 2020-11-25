

#pragma once
#include <cstdint>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {
extern void ApplyMaskAndSoftmax(core::Tensor* inout,
                                const core::Tensor& att_mask, float scale,
                                const std::string name = "ApplyMaskAndSoftmax");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
